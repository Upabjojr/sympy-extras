//! Port of `sympy_extras/logic/boolsimp.py`: SymPy's `simplify` of Boolean
//! formulas (`BooleanFunction._eval_simplify` and those of `And`, `Or` and
//! `Xor`): the univariate intervals, `simplify_logic`, the equations in
//! one symbol substituted, and the patterns of pairs and triples of
//! relations.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::core::function::{rebuild_evaluated, xreplace_evaluated};
use crate::core::logic::{and_, boolean, ite, not_, or_};
use crate::core::relational::{canonical, eq, ge, gt, le, lt, ne, reversed};
use crate::core::sorting::ordered;
use crate::error::{PolyError, PolyResult};
use crate::functions::elementary::complexes::abs_;
use crate::functions::elementary::miscellaneous::{max_, min_};
use crate::numbers::Rat;

type R = PolyResult<Basic>;
/// The simplification of a relation.
pub type RelationSimplifier<'a> = &'a dyn Fn(&Basic) -> R;
/// The size of a formula.
pub type Measure<'a> = &'a dyn Fn(&Basic) -> PolyResult<i64>;

struct Options<'a> {
    relation: RelationSimplifier<'a>,
    measure: Measure<'a>,
    ratio: f64,
    sympy_intervals: bool,
}

fn op(b: &Basic) -> Option<RelOp> {
    match b.head() {
        Head::Relational(o) => Some(*o),
        _ => None,
    }
}
fn is_rel(b: &Basic) -> bool {
    op(b).is_some()
}
fn is_atom_bool(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue | Head::BooleanFalse)
}
fn is_function(b: &Basic) -> bool {
    matches!(
        b.head(),
        Head::And | Head::Or | Head::Not | Head::Xor | Head::Implies | Head::Equivalent | Head::Ite
    )
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn relations_of(b: &Basic, filter: &dyn Fn(RelOp) -> bool) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for n in b.preorder() {
        if op(&n).is_some_and(filter) && !found.contains(&n) {
            found.push(n);
        }
    }
    found
}

/// SymPy's `simplify` of a Boolean formula (`simplify_boolean`).
pub fn simplify_boolean(
    formula: &Basic,
    relation: RelationSimplifier,
    measure: Measure,
    ratio: f64,
    sympy_intervals: bool,
) -> R {
    simplify(
        formula,
        &Options {
            relation,
            measure,
            ratio,
            sympy_intervals,
        },
    )
}

/// [`simplify_boolean`] with the default relation simplifier and measure.
pub fn simplify_boolean_default(formula: &Basic, ratio: f64) -> R {
    use crate::assumptions::refine::{measure, simplify_relational};
    let relation = |r: &Basic| simplify_relational(r, ratio, false, true);
    simplify_boolean(formula, &relation, &measure, ratio, false)
}

fn simplify(b: &Basic, o: &Options) -> R {
    match b.head() {
        Head::Relational(_) => (o.relation)(b),
        Head::And => simplify_and(b, o),
        Head::Or => simplify_or(b, o),
        Head::Xor => simplify_xor(b, o),
        _ if is_function(b) => simplify_function(b, o),
        _ => Ok(b.clone()),
    }
}

fn simplify_function(b: &Basic, o: &Options) -> R {
    let rv: Basic = simplify_univariate(b, o.sympy_intervals)?;
    if !is_function(&rv) {
        return simplify(&rv, o);
    }
    let args: Vec<Basic> = rv.args().iter().map(|a| simplify(a, o)).collect::<PolyResult<_>>()?;
    let mut rv: Basic = rebuild_evaluated(&rv, args)?;
    let relations: Vec<Basic> = ordered(relations_of(&rv, &|_| true));
    if !relations.is_empty() {
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for r in relations {
            let s: Basic = (o.relation)(&r)?;
            rule.insert(r, s);
        }
        rv = xreplace_evaluated(&rv, &rule)?;
    }
    crate::logic::boolalg::simplify_logic(&rv, None, false, None)
}

fn simplify_and(b: &Basic, o: &Options) -> R {
    let mut rv: Basic = simplify_function(b, o)?;
    if !matches!(rv.head(), Head::And) {
        return Ok(rv);
    }
    let relations: Vec<Basic> = rv.args().iter().filter(|a| is_rel(a)).cloned().collect();
    let others: Vec<Basic> = rv.args().iter().filter(|a| !is_rel(a)).cloned().collect();
    if relations.is_empty() {
        return Ok(rv);
    }
    let eqs: Vec<Basic> = relations.iter().filter(|r| op(r) == Some(RelOp::Eq)).cloned().collect();
    let other: Vec<Basic> = relations.iter().filter(|r| op(r) != Some(RelOp::Eq)).cloned().collect();
    if !eqs.is_empty() {
        rv = substituted_equations(&eqs, other, &others, o)?;
    }
    if !matches!(rv.head(), Head::And) {
        return Ok(rv);
    }
    apply_patterns(&rv, &and_patterns(), o.measure, &boolean(false), &and3_patterns())
}

fn linear_solution(e: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::mul::neg;
    let p: Basic = crate::core::add::add(&[e.args()[0].clone(), neg(&e.args()[1])?])?;
    let Some((constant, coefficients)) = crate::assumptions::refine::lin_eq2dict(&p, std::slice::from_ref(x))? else {
        return Ok(None);
    };
    let m: Basic = coefficients
        .iter()
        .find(|(s, _)| s == x)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| int(0));
    if m == int(0) || m.has(x) || constant.has(x) {
        return Ok(None);
    }
    let v: Basic = crate::core::mul::mul(&[neg(&constant)?, crate::core::power::pow(&m, &int(-1))?])?;
    Ok(Some(eq(x, &v)?))
}

fn names(r: &Basic) -> Vec<String> {
    let mut v: Vec<String> = r.free_symbols().iter().map(|s| s.to_string()).collect();
    v.sort();
    v
}

fn push_group(groups: &mut Vec<(usize, Vec<Basic>)>, k: usize, e: Basic) {
    match groups.iter_mut().find(|(g, _)| *g == k) {
        Some((_, v)) => v.push(e),
        None => groups.push((k, vec![e])),
    }
}

fn substituted_equations(eqs: &[Basic], other: Vec<Basic>, others: &[Basic], o: &Options) -> R {
    let mut reps: Vec<(Basic, Basic)> = Vec::new();
    let reps_map = |reps: &[(Basic, Basic)]| -> HashMap<Basic, Basic> { reps.iter().cloned().collect() };
    let ord: Vec<Basic> = ordered(eqs.to_vec());
    let mut sorted: Vec<Basic> = eqs.to_vec();
    sorted.sort_by_key(|r| (names(r), ord.iter().position(|q| q == r).unwrap_or(0)));
    let mut sifted: Vec<(usize, Vec<Basic>)> = Vec::new();
    for e in sorted {
        let k: usize = e.free_symbols().len();
        push_group(&mut sifted, k, e);
    }
    let mut found: Vec<Basic> = Vec::new();
    let mut nonlinear: Vec<Basic> = Vec::new();
    while let Some(pos) = sifted.iter().position(|(k, _)| *k == 1) {
        let (_, group) = sifted.remove(pos);
        for e_ in group {
            let mut e: Basic = e_;
            let x: Basic = e.free_symbols().into_iter().next().expect("one symbol");
            let has_rep = |reps: &[(Basic, Basic)]| reps.iter().any(|(s, _)| *s == x);
            if op(&e) == Some(RelOp::Eq)
                && (e.args()[0] != x || e.args()[1].free_symbols().contains(&x))
                && !has_rep(&reps)
            {
                if let Some(solved) = linear_solution(&e, &x)? {
                    if ((o.measure)(&solved)? as f64) <= o.ratio * (o.measure)(&e)? as f64 {
                        e = solved;
                    } else {
                        found.push(e);
                        continue;
                    }
                }
            }
            if let Some((_, v)) = reps.iter().find(|(s, _)| *s == x) {
                let rule: HashMap<Basic, Basic> = [(x.clone(), v.clone())].into_iter().collect();
                found.push(xreplace_evaluated(&e, &rule)?);
            } else if op(&e) == Some(RelOp::Eq) && e.args()[0] == x && !e.args()[1].free_symbols().contains(&x) {
                reps.push((x.clone(), e.args()[1].clone()));
                found.push(e);
            } else {
                nonlinear.push(e);
            }
        }
        let mut resifted: Vec<(usize, Vec<Basic>)> = Vec::new();
        let rule: HashMap<Basic, Basic> = reps_map(&reps);
        for (_, group) in &sifted {
            for old in group {
                let e2: Basic = xreplace_evaluated(old, &rule)?;
                push_group(&mut resifted, e2.free_symbols().len(), e2);
            }
        }
        sifted = resifted;
    }
    for (_, group) in sifted {
        found.extend(group);
    }
    let rule: HashMap<Basic, Basic> = reps_map(&reps);
    let nonlinear: Vec<Basic> = nonlinear
        .iter()
        .map(|n| xreplace_evaluated(n, &rule))
        .collect::<PolyResult<_>>()?;
    let other: Vec<Basic> = other
        .iter()
        .map(|n| xreplace_evaluated(n, &rule))
        .collect::<PolyResult<_>>()?;
    let mut all: Vec<Basic> = Vec::new();
    for f in found.iter().chain(nonlinear.iter()).chain(other.iter()) {
        all.push(canonical_of(f)?);
    }
    all.extend(others.iter().cloned());
    Ok(and_(&all))
}

/// The formula with its relations in canonical form (`_canonical`).
fn canonical_of(b: &Basic) -> R {
    if is_rel(b) {
        return canonical(b);
    }
    let relations: Vec<Basic> = relations_of(b, &|_| true);
    if relations.is_empty() {
        return Ok(b.clone());
    }
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for r in relations {
        let c: Basic = canonical(&r)?;
        rule.insert(r, c);
    }
    xreplace_evaluated(b, &rule)
}

fn simplify_or(b: &Basic, o: &Options) -> R {
    let lege: Vec<Basic> = relations_of(b, &|k| matches!(k, RelOp::Le | RelOp::Ge));
    if !lege.is_empty() {
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for r in lege {
            let v: Basic = or_(&[eq(&r.args()[0], &r.args()[1])?, strict(&r)?]);
            rule.insert(r, v);
        }
        return simplify(&xreplace_evaluated(b, &rule)?, o);
    }
    let rv: Basic = simplify_function(b, o)?;
    if !matches!(rv.head(), Head::Or) {
        return Ok(rv);
    }
    apply_patterns(&rv, &or_patterns(), o.measure, &boolean(true), &[])
}

fn strict(r: &Basic) -> R {
    match op(r) {
        Some(RelOp::Le) => lt(&r.args()[0], &r.args()[1]),
        Some(RelOp::Ge) => gt(&r.args()[0], &r.args()[1]),
        _ => Ok(r.clone()),
    }
}

fn simplify_xor(b: &Basic, o: &Options) -> R {
    use crate::logic::anf::{to_anf, xor_evaluated};
    let args: Vec<Basic> = b.args().iter().map(|a| simplify(a, o)).collect::<PolyResult<_>>()?;
    let rv: Basic = to_anf(&xor_evaluated(&args, true)?, true)?;
    if !matches!(rv.head(), Head::Xor) {
        return Ok(rv);
    }
    apply_xor_patterns(&rv, o.measure)
}

// ---------------------------------------------------------------------
// patterns

type Bindings = Vec<(&'static str, Basic)>;

fn w(b: &Bindings, name: &str) -> Basic {
    b.iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| v.clone())
        .expect("a bound name")
}

struct Pattern {
    relations: Vec<(RelOp, &'static str, &'static str)>,
    replacement: fn(&Bindings) -> R,
}

fn p(relations: &[(RelOp, &'static str, &'static str)], replacement: fn(&Bindings) -> R) -> Pattern {
    Pattern {
        relations: relations.to_vec(),
        replacement,
    }
}

fn match_(pattern: &Pattern, relations: &[&Basic]) -> PolyResult<Option<Bindings>> {
    let mut bound: Bindings = Vec::new();
    for ((cls, left, right), r) in pattern.relations.iter().zip(relations.iter()) {
        if op(r) != Some(*cls) {
            return Ok(None);
        }
        for (spec, value) in [(*left, &r.args()[0]), (*right, &r.args()[1])] {
            if let Some(name) = spec.strip_prefix('-') {
                match bound.iter().find(|(n, _)| *n == name) {
                    Some((_, v)) if crate::core::mul::neg(v)? == *value => {}
                    _ => return Ok(None),
                }
            } else if let Some((_, v)) = bound.iter().find(|(n, _)| *n == spec) {
                if v != value {
                    return Ok(None);
                }
            } else {
                bound.push((spec, value.clone()));
            }
        }
    }
    Ok(Some(bound))
}

fn real_split(relations: &[Basic]) -> (Vec<Basic>, Vec<Basic>) {
    let nonreal = |r: &Basic| r.free_symbols().iter().any(|s| is(s, Predicate::real) == Some(false));
    (
        relations.iter().filter(|r| !nonreal(r)).cloned().collect(),
        relations.iter().filter(|r| nonreal(r)).cloned().collect(),
    )
}

fn apply_patterns(rv: &Basic, patterns: &[Pattern], measure: Measure, dominating: &Basic, threeterm: &[Pattern]) -> R {
    let is_and: bool = matches!(rv.head(), Head::And);
    let relations: Vec<Basic> = rv.args().iter().filter(|a| is_rel(a)).cloned().collect();
    let others: Vec<Basic> = rv.args().iter().filter(|a| !is_rel(a)).cloned().collect();
    if relations.len() <= 1 {
        return Ok(rv.clone());
    }
    let (real, nonreal) = real_split(&relations);
    let mut rels: Vec<Basic> = real.iter().map(canonical).collect::<PolyResult<_>>()?;
    let combine = |args: &[Basic]| if is_and { and_(args) } else { or_(args) };
    if !threeterm.is_empty() && rels.len() >= 3 {
        match threeterm_(rels, threeterm, &combine, is_and, dominating, measure)? {
            None => return Ok(dominating.clone()),
            Some(f) => rels = f,
        }
    }
    let Some(found) = twoterm(rels, patterns, &combine, is_and, dominating, measure)? else {
        return Ok(dominating.clone());
    };
    let mut all: Vec<Basic> = Vec::new();
    for r in ordered(found).into_iter().filter(is_rel) {
        all.push(canonical_of(&r)?);
    }
    all.extend(others);
    all.extend(nonreal);
    Ok(combine(&all))
}

type Results = Vec<(i64, Vec<usize>, Basic)>;

fn best_of(results: &Results) -> &(i64, Vec<usize>, Basic) {
    let mut best: &(i64, Vec<usize>, Basic) = &results[0];
    for r in results {
        if r.0 > best.0 {
            best = r;
        }
    }
    best
}

fn replace(mut rels: Vec<Basic>, results: &Results, is_and: bool, dominating: &Basic) -> Vec<Basic> {
    let (_, indices, new) = best_of(results);
    let mut sorted: Vec<usize> = indices.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    for i in sorted {
        rels.remove(i);
    }
    if *new != not_(dominating) {
        let connective: bool = if is_and {
            matches!(new.head(), Head::And)
        } else {
            matches!(new.head(), Head::Or)
        };
        let parts: Vec<Basic> = if connective {
            new.args().to_vec()
        } else {
            vec![new.clone()]
        };
        rels.extend(parts.into_iter().filter(is_rel));
    }
    rels
}

fn has_minmax(b: &Basic) -> bool {
    b.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(Function::Max | Function::Min)))
}

fn usable(new: &Basic) -> bool {
    if matches!(new.head(), Head::Ite) || has_minmax(new) {
        return false;
    }
    let parts: Vec<Basic> = if matches!(new.head(), Head::And | Head::Or) {
        new.args().to_vec()
    } else {
        vec![new.clone()]
    };
    parts.iter().all(|p| is_rel(p) || is_atom_bool(p))
}

fn candidates(rels: &[Basic]) -> Vec<Vec<Basic>> {
    rels.iter()
        .map(|r| {
            if matches!(op(r), Some(RelOp::Eq | RelOp::Ne)) {
                vec![r.clone(), reversed(r)]
            } else {
                vec![r.clone()]
            }
        })
        .collect()
}

fn reoriented(rels: Vec<Basic>, flip: &[RelOp]) -> Vec<Basic> {
    let flipped: Vec<Basic> = rels
        .into_iter()
        .map(|r| {
            if op(&r).is_some_and(|k| flip.contains(&k)) {
                reversed(&r)
            } else {
                r
            }
        })
        .collect();
    ordered(flipped).into_iter().filter(is_rel).collect()
}

fn twoterm(
    rels_: Vec<Basic>,
    patterns: &[Pattern],
    combine: &dyn Fn(&[Basic]) -> Basic,
    is_and: bool,
    dominating: &Basic,
    measure: Measure,
) -> PolyResult<Option<Vec<Basic>>> {
    let mut rels: Vec<Basic> = rels_;
    let mut changed: bool = true;
    while changed && rels.len() >= 2 {
        changed = false;
        rels = reoriented(rels, &[RelOp::Ge, RelOp::Gt]);
        let cands: Vec<Vec<Basic>> = candidates(&rels);
        let mut results: Results = Vec::new();
        for i in 0..cands.len() {
            for j in i + 1..cands.len() {
                for pattern in patterns {
                    for p1 in &cands[i] {
                        for p2 in &cands[j] {
                            let Some(bound) = match_(pattern, &[p1, p2])? else {
                                continue;
                            };
                            let new: Basic = (pattern.replacement)(&bound)?;
                            if new == *dominating {
                                return Ok(None);
                            }
                            if usable(&new) {
                                let saving: i64 = measure(&combine(&[p1.clone(), p2.clone()]))? - measure(&new)?;
                                if saving > 0 {
                                    results.push((saving, vec![i, j], new));
                                }
                            }
                        }
                    }
                }
            }
        }
        if !results.is_empty() {
            rels = replace(rels, &results, is_and, dominating);
            changed = true;
        }
    }
    Ok(Some(rels))
}

fn threeterm_(
    rels_: Vec<Basic>,
    patterns: &[Pattern],
    combine: &dyn Fn(&[Basic]) -> Basic,
    is_and: bool,
    dominating: &Basic,
    measure: Measure,
) -> PolyResult<Option<Vec<Basic>>> {
    let mut rels: Vec<Basic> = rels_;
    let mut changed: bool = true;
    while changed && rels.len() >= 3 {
        changed = false;
        rels = reoriented(rels, &[RelOp::Le, RelOp::Lt]);
        let cands: Vec<Vec<Basic>> = candidates(&rels);
        let mut results: Results = Vec::new();
        let n: usize = cands.len();
        for i in 0..n {
            for j in 0..n {
                if j == i {
                    continue;
                }
                for k in 0..n {
                    if k == i || k == j {
                        continue;
                    }
                    for pattern in patterns {
                        for p1 in &cands[i] {
                            for p2 in &cands[j] {
                                for p3 in &cands[k] {
                                    let Some(bound) = match_(pattern, &[p1, p2, p3])? else {
                                        continue;
                                    };
                                    let new: Basic = (pattern.replacement)(&bound)?;
                                    if new == *dominating {
                                        return Ok(None);
                                    }
                                    if usable(&new) {
                                        let saving: i64 =
                                            measure(&combine(&[p1.clone(), p2.clone(), p3.clone()]))? - measure(&new)?;
                                        if saving > 0 {
                                            results.push((saving, vec![i, j, k], new));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if !results.is_empty() {
            rels = replace(rels, &results, is_and, dominating);
            changed = true;
        }
    }
    Ok(Some(rels))
}

fn apply_xor_patterns(rv: &Basic, measure: Measure) -> R {
    use crate::logic::anf::xor_evaluated;
    let relations: Vec<Basic> = rv.args().iter().filter(|a| is_rel(a)).cloned().collect();
    let others: Vec<Basic> = rv.args().iter().filter(|a| !is_rel(a)).cloned().collect();
    if relations.len() <= 1 {
        return Ok(rv.clone());
    }
    let (real, nonreal) = real_split(&relations);
    let mut rels: Vec<Basic> = real.iter().map(canonical).collect::<PolyResult<_>>()?;
    let mut replaced: Vec<Basic> = Vec::new();
    let patterns: Vec<Pattern> = xor_patterns();
    let mut changed: bool = true;
    while changed && rels.len() >= 2 {
        changed = false;
        rels = reoriented(rels, &[RelOp::Ge, RelOp::Gt]);
        let cands: Vec<Vec<Basic>> = candidates(&rels);
        let mut results: Results = Vec::new();
        for i in 0..cands.len() {
            for j in i + 1..cands.len() {
                for pattern in &patterns {
                    for p1 in &cands[i] {
                        for p2 in &cands[j] {
                            let Some(bound) = match_(pattern, &[p1, p2])? else {
                                continue;
                            };
                            let new: Basic = (pattern.replacement)(&bound)?;
                            if !matches!(new.head(), Head::Ite) && !has_minmax(&new) {
                                let saving: i64 =
                                    measure(&xor_evaluated(&[p1.clone(), p2.clone()], true)?)? - measure(&new)?;
                                if saving > 0 {
                                    results.push((saving, vec![i, j], new));
                                }
                            }
                        }
                    }
                }
            }
        }
        if !results.is_empty() {
            let (_, indices, new) = best_of(&results).clone();
            let mut sorted: Vec<usize> = indices;
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            for i in sorted {
                rels.remove(i);
            }
            let parts: Vec<Basic> = if matches!(new.head(), Head::Xor) {
                new.args().to_vec()
            } else {
                vec![new]
            };
            for part in parts {
                if is_rel(&part) {
                    rels.push(part);
                } else {
                    replaced.push(part);
                }
            }
            changed = true;
        }
    }
    let mut all: Vec<Basic> = Vec::new();
    let ordered_rels: Vec<Basic> = ordered(rels).into_iter().filter(is_rel).collect();
    for r in ordered_rels.iter().chain(replaced.iter()) {
        all.push(canonical_of(r)?);
    }
    all.extend(others);
    all.extend(nonreal);
    xor_evaluated(&all, true)
}

use RelOp::{Eq as EQ, Ge as GE, Gt as GT, Le as LE, Lt as LT, Ne as NE};

fn and_patterns() -> Vec<Pattern> {
    vec![
        p(&[(EQ, "a", "b"), (LT, "a", "b")], |_| Ok(boolean(false))),
        p(&[(LT, "b", "a"), (LT, "a", "b")], |_| Ok(boolean(false))),
        p(&[(EQ, "a", "b"), (LE, "b", "a")], |w_| eq(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "b", "a"), (LE, "a", "b")], |w_| eq(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "a", "b"), (LT, "a", "b")], |w_| lt(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "a", "b"), (NE, "a", "b")], |w_| lt(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LT, "a", "b"), (NE, "a", "b")], |w_| lt(&w(w_, "a"), &w(w_, "b"))),
        p(&[(EQ, "a", "b"), (EQ, "a", "-b")], |w_| {
            Ok(and_(&[eq(&w(w_, "a"), &int(0))?, eq(&w(w_, "b"), &int(0))?]))
        }),
        p(&[(LE, "b", "a"), (LE, "c", "a")], |w_| {
            ge(&w(w_, "a"), &max_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "b", "a"), (LT, "c", "a")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&gt(&b, &c)?, &ge(&a, &b)?, &gt(&a, &c)?))
        }),
        p(&[(LT, "b", "a"), (LT, "c", "a")], |w_| {
            gt(&w(w_, "a"), &max_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "a", "b"), (LE, "a", "c")], |w_| {
            le(&w(w_, "a"), &min_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "a", "b"), (LT, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&lt(&b, &c)?, &le(&a, &b)?, &lt(&a, &c)?))
        }),
        p(&[(LT, "a", "b"), (LT, "a", "c")], |w_| {
            lt(&w(w_, "a"), &min_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "a", "b"), (LE, "c", "a")], and_between_le),
        p(&[(LE, "c", "a"), (LE, "a", "b")], and_between_le),
        p(&[(LT, "a", "b"), (LT, "c", "a")], and_between_lt),
        p(&[(LT, "c", "a"), (LT, "a", "b")], and_between_lt),
        p(&[(LE, "a", "b"), (LT, "c", "a")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&le(&b, &c)?, &boolean(false), &and_(&[le(&a, &b)?, gt(&a, &c)?])))
        }),
        p(&[(LE, "c", "a"), (LT, "a", "b")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&le(&b, &c)?, &boolean(false), &and_(&[lt(&a, &b)?, ge(&a, &c)?])))
        }),
        p(&[(EQ, "a", "b"), (EQ, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&eq(&b, &c)?, &eq(&a, &b)?, &boolean(false)))
        }),
        p(&[(LT, "a", "b"), (LT, "-b", "a")], |w_| {
            let (a, b) = (w(w_, "a"), w(w_, "b"));
            Ok(ite(&gt(&b, &int(0))?, &lt(&abs_(&a)?, &b)?, &boolean(false)))
        }),
        p(&[(LE, "a", "b"), (LE, "-b", "a")], |w_| {
            let (a, b) = (w(w_, "a"), w(w_, "b"));
            Ok(ite(&ge(&b, &int(0))?, &le(&abs_(&a)?, &b)?, &boolean(false)))
        }),
    ]
}

fn and_between_le(w_: &Bindings) -> R {
    let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
    let inner: Basic = ite(&lt(&b, &c)?, &boolean(false), &and_(&[le(&a, &b)?, ge(&a, &c)?]));
    Ok(ite(&eq(&b, &c)?, &eq(&a, &b)?, &inner))
}

fn and_between_lt(w_: &Bindings) -> R {
    let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
    Ok(ite(&lt(&b, &c)?, &boolean(false), &and_(&[lt(&a, &b)?, gt(&a, &c)?])))
}

fn and3_patterns() -> Vec<Pattern> {
    fn two(
        f1: fn(&Basic, &Basic) -> R,
        x1: &str,
        y1: &str,
        f2: fn(&Basic, &Basic) -> R,
        x2: &str,
        y2: &str,
        w_: &Bindings,
    ) -> R {
        Ok(and_(&[f1(&w(w_, x1), &w(w_, y1))?, f2(&w(w_, x2), &w(w_, y2))?]))
    }
    vec![
        p(&[(GE, "a", "b"), (GE, "b", "c"), (GT, "c", "a")], |_| {
            Ok(boolean(false))
        }),
        p(&[(GE, "a", "b"), (GT, "b", "c"), (GT, "c", "a")], |_| {
            Ok(boolean(false))
        }),
        p(&[(GT, "a", "b"), (GT, "b", "c"), (GT, "c", "a")], |_| {
            Ok(boolean(false))
        }),
        p(&[(GE, "a", "b"), (GE, "a", "c"), (GE, "b", "c")], |w_| {
            two(ge, "a", "b", ge, "b", "c", w_)
        }),
        p(&[(GE, "a", "b"), (GE, "a", "c"), (GT, "b", "c")], |w_| {
            two(ge, "a", "b", gt, "b", "c", w_)
        }),
        p(&[(GE, "a", "b"), (GT, "a", "c"), (GT, "b", "c")], |w_| {
            two(ge, "a", "b", gt, "b", "c", w_)
        }),
        p(&[(GE, "a", "c"), (GT, "a", "b"), (GT, "b", "c")], |w_| {
            two(gt, "a", "b", gt, "b", "c", w_)
        }),
        p(&[(GE, "b", "c"), (GT, "a", "b"), (GT, "a", "c")], |w_| {
            two(gt, "a", "b", ge, "b", "c", w_)
        }),
        p(&[(GT, "a", "b"), (GT, "a", "c"), (GT, "b", "c")], |w_| {
            two(gt, "a", "b", gt, "b", "c", w_)
        }),
        p(&[(GE, "b", "a"), (GE, "c", "a"), (GE, "b", "c")], |w_| {
            two(ge, "c", "a", ge, "b", "c", w_)
        }),
        p(&[(GE, "b", "a"), (GE, "c", "a"), (GT, "b", "c")], |w_| {
            two(ge, "c", "a", gt, "b", "c", w_)
        }),
        p(&[(GE, "b", "a"), (GT, "c", "a"), (GT, "b", "c")], |w_| {
            two(gt, "c", "a", gt, "b", "c", w_)
        }),
        p(&[(GE, "c", "a"), (GT, "b", "a"), (GT, "b", "c")], |w_| {
            two(ge, "c", "a", gt, "b", "c", w_)
        }),
        p(&[(GE, "b", "c"), (GT, "b", "a"), (GT, "c", "a")], |w_| {
            two(gt, "c", "a", ge, "b", "c", w_)
        }),
        p(&[(GT, "b", "a"), (GT, "c", "a"), (GT, "b", "c")], |w_| {
            two(gt, "c", "a", gt, "b", "c", w_)
        }),
        p(&[(GE, "a", "b"), (GE, "b", "c"), (GE, "c", "a")], |w_| {
            two(eq, "a", "b", eq, "b", "c", w_)
        }),
    ]
}

fn or_patterns() -> Vec<Pattern> {
    vec![
        p(&[(LE, "b", "a"), (LE, "a", "b")], |_| Ok(boolean(true))),
        p(&[(LE, "b", "a"), (NE, "a", "b")], |_| Ok(boolean(true))),
        p(&[(EQ, "a", "b"), (LE, "a", "b")], |w_| le(&w(w_, "a"), &w(w_, "b"))),
        p(&[(EQ, "a", "b"), (LT, "a", "b")], |w_| le(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LT, "b", "a"), (LT, "a", "b")], |w_| ne(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LT, "b", "a"), (NE, "a", "b")], |w_| ne(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "a", "b"), (LT, "a", "b")], |w_| le(&w(w_, "a"), &w(w_, "b"))),
        p(&[(EQ, "a", "b"), (NE, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&eq(&b, &c)?, &boolean(true), &ne(&a, &c)?))
        }),
        p(&[(NE, "a", "b"), (NE, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&eq(&b, &c)?, &ne(&a, &b)?, &boolean(true)))
        }),
        p(&[(LE, "b", "a"), (LE, "c", "a")], |w_| {
            ge(&w(w_, "a"), &min_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "b", "a"), (LT, "c", "a")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&gt(&b, &c)?, &lt(&c, &a)?, &le(&b, &a)?))
        }),
        p(&[(LT, "b", "a"), (LT, "c", "a")], |w_| {
            gt(&w(w_, "a"), &min_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "a", "b"), (LE, "a", "c")], |w_| {
            le(&w(w_, "a"), &max_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "a", "b"), (LT, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&ge(&b, &c)?, &le(&a, &b)?, &lt(&a, &c)?))
        }),
        p(&[(LT, "a", "b"), (LT, "a", "c")], |w_| {
            lt(&w(w_, "a"), &max_(&[w(w_, "b"), w(w_, "c")])?)
        }),
        p(&[(LE, "a", "b"), (LE, "c", "a")], or_outside_le),
        p(&[(LE, "c", "a"), (LE, "a", "b")], or_outside_le),
        p(&[(LT, "a", "b"), (LT, "c", "a")], or_outside_lt),
        p(&[(LT, "c", "a"), (LT, "a", "b")], or_outside_lt),
        p(&[(LE, "a", "b"), (LT, "c", "a")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&ge(&b, &c)?, &boolean(true), &or_(&[le(&a, &b)?, gt(&a, &c)?])))
        }),
        p(&[(LE, "c", "a"), (LT, "a", "b")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(&ge(&b, &c)?, &boolean(true), &or_(&[lt(&a, &b)?, ge(&a, &c)?])))
        }),
        p(&[(LT, "b", "a"), (LT, "a", "-b")], |w_| {
            let (a, b) = (w(w_, "a"), w(w_, "b"));
            Ok(ite(&ge(&b, &int(0))?, &gt(&abs_(&a)?, &b)?, &boolean(true)))
        }),
        p(&[(LE, "b", "a"), (LE, "a", "-b")], |w_| {
            let (a, b) = (w(w_, "a"), w(w_, "b"));
            Ok(ite(&gt(&b, &int(0))?, &ge(&abs_(&a)?, &b)?, &boolean(true)))
        }),
    ]
}

fn or_outside_le(w_: &Bindings) -> R {
    let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
    Ok(ite(&ge(&b, &c)?, &boolean(true), &or_(&[le(&a, &b)?, ge(&a, &c)?])))
}

fn or_outside_lt(w_: &Bindings) -> R {
    let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
    Ok(ite(&gt(&b, &c)?, &boolean(true), &or_(&[lt(&a, &b)?, gt(&a, &c)?])))
}

fn xor_patterns() -> Vec<Pattern> {
    vec![
        p(&[(EQ, "a", "b"), (LE, "a", "b")], |w_| lt(&w(w_, "a"), &w(w_, "b"))),
        p(&[(EQ, "a", "b"), (LT, "a", "b")], |w_| le(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "a", "b"), (LT, "a", "b")], |w_| eq(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "a", "b"), (LE, "b", "a")], |w_| ne(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "b", "a"), (NE, "a", "b")], |w_| le(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LT, "b", "a"), (NE, "a", "b")], |w_| lt(&w(w_, "a"), &w(w_, "b"))),
        p(&[(LE, "b", "a"), (LE, "c", "a")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(and_(&[
                ge(&a, &min_(&[b.clone(), c.clone()])?)?,
                lt(&a, &max_(&[b, c])?)?,
            ]))
        }),
        p(&[(LE, "b", "a"), (LT, "c", "a")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(
                &gt(&b, &c)?,
                &and_(&[gt(&a, &c)?, lt(&a, &b)?]),
                &and_(&[ge(&a, &b)?, le(&a, &c)?]),
            ))
        }),
        p(&[(LT, "b", "a"), (LT, "c", "a")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(and_(&[
                gt(&a, &min_(&[b.clone(), c.clone()])?)?,
                le(&a, &max_(&[b, c])?)?,
            ]))
        }),
        p(&[(LE, "a", "b"), (LE, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(and_(&[
                le(&a, &max_(&[b.clone(), c.clone()])?)?,
                gt(&a, &min_(&[b, c])?)?,
            ]))
        }),
        p(&[(LE, "a", "b"), (LT, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(ite(
                &lt(&b, &c)?,
                &and_(&[lt(&a, &c)?, gt(&a, &b)?]),
                &and_(&[le(&a, &b)?, ge(&a, &c)?]),
            ))
        }),
        p(&[(LT, "a", "b"), (LT, "a", "c")], |w_| {
            let (a, b, c) = (w(w_, "a"), w(w_, "b"), w(w_, "c"));
            Ok(and_(&[
                lt(&a, &max_(&[b.clone(), c.clone()])?)?,
                ge(&a, &min_(&[b, c])?)?,
            ]))
        }),
    ]
}

// ---------------------------------------------------------------------
// univariate formulas

/// A formula in one symbol without equations as the union of the
/// intervals where it holds (`simplify_univariate`).
pub fn simplify_univariate(expr: &Basic, sympy_intervals: bool) -> R {
    if !is_function(expr) {
        return Ok(expr.clone());
    }
    if !relations_of(expr, &|k| matches!(k, RelOp::Eq | RelOp::Ne)).is_empty() {
        return Ok(expr.clone());
    }
    let free = expr.free_symbols();
    if free.len() != 1 {
        return Ok(expr.clone());
    }
    let x: Basic = free.into_iter().next().expect("one symbol");
    let truth: Option<Truth> = truth_intervals(expr, &x)?;
    if truth.as_ref().is_some_and(|t| t.isolated) {
        return Ok(expr.clone());
    }
    if sympy_intervals {
        return sympy_univariate(expr, &x);
    }
    let Some(truth) = truth else { return Ok(expr.clone()) };
    if truth.intervals.is_empty() {
        return Ok(boolean(false));
    }
    let mut args: Vec<Basic> = Vec::new();
    for (a, b, incl_a, incl_b) in truth.intervals {
        let c: Basic = if matches!(a.head(), Head::NegativeInfinity) {
            if matches!(b.head(), Head::Infinity) {
                boolean(true)
            } else {
                relation(if incl_b { RelOp::Le } else { RelOp::Lt }, &x, &b)?
            }
        } else if matches!(b.head(), Head::Infinity) {
            relation(if incl_a { RelOp::Ge } else { RelOp::Gt }, &x, &a)?
        } else {
            let lower: Basic = relation(if incl_a { RelOp::Le } else { RelOp::Lt }, &a, &x)?;
            let upper: Basic = relation(if incl_b { RelOp::Le } else { RelOp::Lt }, &x, &b)?;
            and_(&[lower, upper])
        };
        args.push(c);
    }
    Ok(or_(&args))
}

/// SymPy's `simplify_univariate` on the intervals of `Piecewise._intervals`,
/// the endpoints included as the formula says (`_sympy_univariate`).
fn sympy_univariate(expr: &Basic, x: &Basic) -> R {
    use crate::core::relational::{ge, gt, le, lt};
    use crate::functions::elementary::piecewise::{expr_cond_pair, piecewise_intervals, piecewise_node};
    let pw: Basic = piecewise_node(vec![expr_cond_pair(&Basic::zero(), expr)]);
    let Ok(pieces) = piecewise_intervals(&pw, x, true)? else {
        return Ok(expr.clone());
    };
    if pieces.is_empty() {
        return Ok(boolean(false));
    }
    let holds = |at: &Basic| -> PolyResult<bool> { Ok(crate::core::subs::subs(expr, x, at)? == boolean(true)) };
    let mut args: Vec<Basic> = Vec::new();
    for (a, b, _, _) in pieces {
        let b_infinite: bool = crate::core::expr::is(&b, crate::core::assumptions::Predicate::infinite) == Some(true);
        let c: Basic = if matches!(a.head(), Head::NegativeInfinity) {
            if matches!(b.head(), Head::Infinity) {
                boolean(true)
            } else if holds(&b)? {
                le(x, &b)?
            } else {
                lt(x, &b)?
            }
        } else {
            let incl_a: bool = holds(&a)?;
            let incl_b: bool = holds(&b)?;
            if incl_a && incl_b {
                if b_infinite {
                    ge(x, &a)?
                } else {
                    and_(&[le(&a, x)?, le(x, &b)?])
                }
            } else if incl_a {
                and_(&[le(&a, x)?, lt(x, &b)?])
            } else if incl_b {
                if b_infinite {
                    gt(x, &a)?
                } else {
                    and_(&[lt(&a, x)?, le(x, &b)?])
                }
            } else {
                and_(&[lt(&a, x)?, lt(x, &b)?])
            }
        };
        args.push(c);
    }
    Ok(or_(&args))
}

/// `cls(lhs, rhs)` as SymPy's operators build it (`_relation`).
fn relation(k: RelOp, lhs: &Basic, rhs: &Basic) -> R {
    let node: Basic = Basic::new(Head::Relational(k), vec![lhs.clone(), rhs.clone()]);
    let decided: Basic = crate::core::function::doit(&node)?;
    Ok(if is_atom_bool(&decided) { decided } else { node })
}

struct Solved {
    relation: Basic,
    p: Basic,
    roots: Vec<Basic>,
}

impl Solved {
    fn holds(&self, sign: i32) -> bool {
        match op(&self.relation) {
            Some(RelOp::Lt) => sign < 0,
            Some(RelOp::Le) => sign <= 0,
            Some(RelOp::Gt) => sign > 0,
            _ => sign >= 0,
        }
    }
}

fn solved(r: &Basic, x: &Basic) -> PolyResult<Option<Solved>> {
    use crate::core::mul::neg;
    if !matches!(op(r), Some(RelOp::Lt | RelOp::Le | RelOp::Gt | RelOp::Ge)) {
        return Ok(None);
    }
    let p: Basic = crate::core::add::add(&[r.args()[0].clone(), neg(&r.args()[1])?])?;
    if let Some((constant, coefficients)) = crate::assumptions::refine::lin_eq2dict(&p, std::slice::from_ref(x))? {
        let m: Basic = coefficients
            .iter()
            .find(|(s, _)| s == x)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| int(0));
        let number = crate::core::assumptions::is_number;
        if number(&m)
            && is(&m, Predicate::extended_real) == Some(true)
            && m != int(0)
            && number(&constant)
            && is(&constant, Predicate::extended_real) == Some(true)
        {
            let root: Basic = crate::core::mul::div(&neg(&constant)?, &m)?;
            return Ok(Some(Solved {
                relation: r.clone(),
                p,
                roots: vec![root],
            }));
        }
    }
    let poly = match crate::polys::polytools::Poly::new(&p, std::slice::from_ref(x)) {
        Ok(v) => v,
        Err(PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_)) => return Ok(None),
        Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => return Ok(None),
        Err(e) => return Err(e),
    };
    let degree: i64 = poly.degree(0)?;
    if degree < 1 || !poly.all_coeffs()?.iter().all(|c| c.is_rational_class()) {
        return Ok(None);
    }
    let found = crate::polys::polyroots::roots(&p, std::slice::from_ref(x), true, true, None)?;
    if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != degree {
        return Ok(None);
    }
    let mut real: Vec<Basic> = Vec::new();
    for (root, _) in found.iter() {
        match is(root, Predicate::extended_real) {
            Some(true) => real.push(root.clone()),
            Some(false) => {}
            None => return Ok(None),
        }
    }
    Ok(Some(Solved {
        relation: r.clone(),
        p,
        roots: real,
    }))
}

fn approximation(e: &Basic) -> Option<Rat> {
    let value = crate::numeric::reliable_value(e, 30, &[])?;
    let (re, im) = crate::core::evalf::parts(&value);
    if crate::assumptions::ivarith::real_to_rat(&im) != Rat::ZERO {
        return None;
    }
    Some(crate::assumptions::ivarith::real_to_rat(&re))
}

struct Truth {
    intervals: Vec<(Basic, Basic, bool, bool)>,
    isolated: bool,
}

fn truth_intervals(expr: &Basic, x: &Basic) -> PolyResult<Option<Truth>> {
    let relations: Vec<Basic> = ordered(relations_of(expr, &|_| true));
    let mut all: Vec<Solved> = Vec::new();
    for r in &relations {
        if !r.free_symbols().contains(x) {
            return Ok(None);
        }
        let Some(s) = solved(r, x)? else { return Ok(None) };
        all.push(s);
    }
    let mut points: Vec<(Rat, Basic)> = Vec::new();
    for s in &all {
        for root in &s.roots {
            if points.iter().any(|(_, p)| p == root) {
                continue;
            }
            let Some(a) = approximation(root) else { return Ok(None) };
            points.push((a, root.clone()));
        }
    }
    points.sort_by(|a, b| a.0.cmp(&b.0));
    let tiny: Rat = Rat::from_parts(crate::numbers::Int::ONE, dashu_int::UBig::from(10u8).pow(20));
    for k in 1..points.len() {
        if &points[k].0 - &points[k - 1].0 < tiny {
            return Ok(None);
        }
    }
    let sign_at = |s: &Solved, k: usize| -> PolyResult<Option<i32>> {
        let point: Basic = if k % 2 == 1 {
            let point: &Basic = &points[k / 2].1;
            if s.roots.iter().any(|r| r == point) {
                return Ok(Some(0));
            }
            point.clone()
        } else {
            let i: usize = k / 2;
            let sample: Rat = if points.is_empty() {
                Rat::ZERO
            } else if i == 0 {
                &points[0].0 - Rat::ONE
            } else if i == points.len() {
                &points[points.len() - 1].0 + Rat::ONE
            } else {
                (&points[i - 1].0 + &points[i].0) / Rat::from(2)
            };
            Basic::from_rational(sample)
        };
        let rule: HashMap<Basic, Basic> = [(x.clone(), point)].into_iter().collect();
        let at: Basic = xreplace_evaluated(&s.p, &rule)?;
        let Some(value) = crate::numeric::reliable_value(&at, 15, &[]) else {
            return Ok(None);
        };
        let (re, im) = crate::core::evalf::parts(&value);
        let (re, im) = (
            crate::assumptions::ivarith::real_to_rat(&re),
            crate::assumptions::ivarith::real_to_rat(&im),
        );
        if re == Rat::ZERO && im == Rat::ZERO {
            return Ok(None);
        }
        Ok(Some(if im == Rat::ZERO && re > Rat::ZERO { 1 } else { -1 }))
    };
    let mut truth: Vec<bool> = Vec::new();
    for k in 0..2 * points.len() + 1 {
        let mut values: HashMap<Basic, Basic> = HashMap::new();
        for s in &all {
            let Some(sign) = sign_at(s, k)? else { return Ok(None) };
            values.insert(s.relation.clone(), boolean(s.holds(sign)));
        }
        let decided: Basic = xreplace_evaluated(expr, &values)?;
        if !is_atom_bool(&decided) {
            return Ok(None);
        }
        truth.push(matches!(decided.head(), Head::BooleanTrue));
    }
    let mut intervals: Vec<(Basic, Basic, bool, bool)> = Vec::new();
    let mut k: usize = 0;
    while k < truth.len() {
        if !truth[k] {
            k += 1;
            continue;
        }
        let start: usize = k;
        while k + 1 < truth.len() && truth[k + 1] {
            k += 1;
        }
        let end: usize = k;
        if start == end && start % 2 == 1 {
            return Ok(Some(Truth {
                intervals: vec![],
                isolated: true,
            }));
        }
        let a: Basic = if start == 0 {
            Basic::negative_oo()
        } else {
            points[(start - 1) / 2].1.clone()
        };
        let b: Basic = if end == truth.len() - 1 {
            Basic::oo()
        } else {
            points[end / 2].1.clone()
        };
        intervals.push((a, b, start % 2 == 1, end % 2 == 1));
        k += 1;
    }
    Ok(Some(Truth {
        intervals,
        isolated: false,
    }))
}
