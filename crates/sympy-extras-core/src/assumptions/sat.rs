//! Port of `sympy_extras/assumptions/sat.py`: the satisfiability of
//! formulas mixing Boolean variables, polynomial relations and
//! memberships, the lazy approach of SMT solvers: a SAT solver (DPLL2) on
//! the propositional abstraction, each model checked against the theory
//! (the consistency of the predicates, the real solutions by Hermite's
//! count or the cylindrical algebraic decomposition, integer points).

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::assumptions::ask::facts_of;
use crate::assumptions::facts::{consequences, is_compound, normalize, to_polynomial, to_predicates, Facts};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::is;
use crate::core::function::xreplace_evaluated;
use crate::core::logic::{and_, boolean, not_};
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::cad::samplepoints::{compare_real, floor_scaled, Real};
use crate::polys::numberfields::qqbar::Algebraic;

/// A value of a model: a truth value or an expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Expr(Basic),
}

/// A model: Boolean variables to truth values, real variables to values.
pub type Model = Vec<(Basic, Value)>;
/// A witness point.
pub type Witness = Vec<(Basic, Basic)>;

/// The answer of [`satisfiable`].
#[derive(Debug, Clone, PartialEq)]
pub enum Sat {
    Model(Model),
    Unsatisfiable,
    Unknown,
}

enum Check {
    Witness(Witness),
    Unsatisfiable,
    Unknown,
}

fn symbols(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect()
}

fn sorted_symbols(s: &BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = s.iter().cloned().collect();
    v.sort_by_key(|s| s.to_string());
    v
}

fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}
fn is_false(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanFalse)
}
fn int(n: &Int) -> Basic {
    Basic::from_rational(Rat::from(n.clone()))
}

/// The non-propositional atoms replaced by Boolean dummies (`_abstract`).
fn abstract_(formula: &Basic) -> PolyResult<(Basic, Vec<(Basic, Basic)>)> {
    let mut atoms: Vec<(Basic, Basic)> = Vec::new();
    let mut mapping: HashMap<Basic, Basic> = HashMap::new();
    for atom in formula.preorder() {
        if !matches!(
            atom.head(),
            Head::Relational(_) | Head::Contains | Head::AppliedPredicate
        ) || mapping.contains_key(&atom)
        {
            continue;
        }
        let d: Basic = crate::core::symbol::Dummy::new(&format!("p{}", atoms.len()), &[]).0;
        atoms.push((d.clone(), atom.clone()));
        mapping.insert(atom, d);
    }
    Ok((xreplace_evaluated(formula, &mapping)?, atoms))
}

fn literals(model: &[(Basic, bool)], atoms: &[(Basic, Basic)]) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for (symbol, value) in model {
        if !matches!(symbol.head(), Head::Dummy(..)) {
            continue;
        }
        if let Some((_, atom)) = atoms.iter().find(|(d, _)| d == symbol) {
            out.push(if *value { atom.clone() } else { not_(atom) });
        }
    }
    out
}

/// A real algebraic number as a native one (`_real_algebraic`).
fn real_algebraic(a: &Basic) -> PolyResult<Real> {
    if let Some(number) = crate::polys::cad::indexedroot::algebraic_of(a)? {
        return Ok(number);
    }
    if a.is_mul() {
        let mut c: Rat = Rat::ONE;
        let mut result: Option<Algebraic> = None;
        for f in a.args() {
            if let (true, Some(r)) = (f.is_rational_class(), f.as_rational()) {
                c *= r.clone();
                continue;
            }
            let factor: Real = real_algebraic(f)?;
            result = Some(match result {
                None => (*factor).clone(),
                Some(r) => r.mul(&factor)?,
            });
        }
        return Ok(Arc::new(match result {
            None => Algebraic::from_rational(&c),
            Some(r) if c == Rat::ONE => r,
            // scaled_real: the same number as the product
            Some(r) => r.mul(&Algebraic::from_rational(&c))?,
        }));
    }
    if a.is_pow() && a.args()[1].is_integer_value() {
        let k: i64 = a.args()[1]
            .as_rational()
            .and_then(|r| i64::try_from(&crate::numbers::numer(r)).ok())
            .ok_or_else(|| value_error(format!("{} has a huge exponent", a)))?;
        let base: Real = real_algebraic(&a.args()[0])?;
        return Ok(Arc::new(if k >= 0 { base.pow(k)? } else { base.pow(-k)?.inverse()? }));
    }
    Err(value_error(format!(
        "{} is not a rational number, a real algebraic root or a product or power of them",
        a
    )))
}

fn floor_of(a: &Basic) -> PolyResult<Int> {
    if let (true, Some(r)) = (a.is_rational_class(), a.as_rational()) {
        let (n, d) = (crate::numbers::numer(r), Int::from(crate::numbers::denom(r)));
        let q: Int = &n / &d;
        return Ok(if crate::numbers::rat_is_negative(r) && Rat::from(q.clone()) != *r {
            q - Int::ONE
        } else {
            q
        });
    }
    floor_scaled(&real_algebraic(a)?, 0)
}

fn compare(a: &Basic, b: &Basic) -> PolyResult<i32> {
    if let (Some(x), Some(y)) = (a.as_rational(), b.as_rational()) {
        return Ok(x.cmp(y) as i32);
    }
    compare_real(&real_algebraic(a)?, &real_algebraic(b)?)
}

/// An integer in a union of intervals and points (`_integer_in`).
fn integer_in(sets: &Basic) -> PolyResult<Option<Basic>> {
    let parts: Vec<Basic> = if matches!(sets.head(), Head::Union) {
        sets.args().to_vec()
    } else {
        vec![sets.clone()]
    };
    for part in parts {
        if matches!(part.head(), Head::FiniteSet) {
            for v in part.args() {
                if is(v, Predicate::integer) == Some(true) {
                    return Ok(Some(v.clone()));
                }
            }
            continue;
        }
        if !matches!(part.head(), Head::Interval | Head::Reals) {
            continue;
        }
        let (a, b, left_open, right_open) = crate::sets::sets::interval_parts(&part);
        if matches!(a.head(), Head::NegativeInfinity) {
            if matches!(b.head(), Head::Infinity) {
                return Ok(Some(Basic::int(0)));
            }
            let mut c: Int = floor_of(&b)?;
            if right_open && int(&c) == b {
                c -= Int::ONE;
            }
            return Ok(Some(int(&c)));
        }
        let mut c: Int = floor_of(&a)?;
        if int(&c) != a || left_open {
            c += Int::ONE;
        }
        if matches!(b.head(), Head::Infinity) || compare(&int(&c), &b)? < 0 || (int(&c) == b && !right_open) {
            return Ok(Some(int(&c)));
        }
    }
    Ok(None)
}

fn subs_all(e: &Basic, pairs: &[(Basic, Basic)]) -> PolyResult<Basic> {
    let mut v: Basic = e.clone();
    for (s, value) in pairs {
        v = crate::core::subs::subs(&v, s, value)?;
    }
    Ok(v)
}

/// `Some(Some(point))`, `Some(None)` when there is none, `None` when
/// undecided (`_integer_witness`).
fn integer_witness(
    formula: &Basic,
    points: &[Witness],
    integer: &BTreeSet<Basic>,
    gens: &[Basic],
) -> PolyResult<Option<Option<Witness>>> {
    use crate::polys::cad::qe::{sample_points, solution_set, Quantifier};
    let integer_gens: Vec<&Basic> = gens.iter().filter(|g| integer.contains(*g)).collect();
    if integer_gens.len() == 1 {
        let n: Basic = integer_gens[0].clone();
        let others: Vec<Basic> = gens.iter().filter(|g| **g != n).cloned().collect();
        let prefix: Vec<(Quantifier, Basic)> = others.iter().map(|g| (Quantifier::Exists, g.clone())).collect();
        let sets: Option<Basic> = match solution_set(formula, &n, &prefix, None, true) {
            Ok(s) => Some(s),
            Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => None,
            Err(e) => return Err(e),
        };
        if let Some(sets) = sets {
            let Some(k) = integer_in(&sets)? else {
                return Ok(Some(None));
            };
            let mut witness: Witness = vec![(n.clone(), k.clone())];
            if !others.is_empty() {
                let points_ = sample_points(&crate::core::subs::subs(formula, &n, &k)?, &others, None, true)?;
                let Some(first) = points_.into_iter().next() else {
                    return Ok(None);
                };
                witness.extend(first);
            }
            return Ok(Some(Some(witness)));
        }
    }
    for point in points {
        let mut choices: Vec<Vec<Basic>> = Vec::new();
        for g in gens {
            let v: &Basic = &point.iter().find(|(s, _)| s == g).expect("a coordinate").1;
            if integer.contains(g) {
                let lo: Int = floor_of(v)?;
                let set: BTreeSet<Int> = [lo.clone(), &lo + Int::ONE, &lo - Int::ONE, &lo + Int::from(2u8)]
                    .into_iter()
                    .collect();
                choices.push(set.iter().map(int).collect());
            } else {
                choices.push(vec![v.clone()]);
            }
        }
        // itertools.product: the last coordinate fastest
        let total: usize = choices.iter().map(|c| c.len()).product();
        for index in 0..total {
            let mut rest: usize = index;
            let mut combo: Vec<Basic> = vec![Basic::int(0); choices.len()];
            for j in (0..choices.len()).rev() {
                combo[j] = choices[j][rest % choices[j].len()].clone();
                rest /= choices[j].len();
            }
            let candidate: Witness = gens.iter().cloned().zip(combo).collect();
            if is_true(&subs_all(formula, &candidate)?) {
                return Ok(Some(Some(candidate)));
            }
        }
    }
    Ok(None)
}

fn without_real_solution(formula: &Basic, gens: &[Basic]) -> PolyResult<bool> {
    if gens.len() < 2 {
        return Ok(false);
    }
    let decided = crate::timeout::attempt_within(Some(2.0), || {
        crate::polys::hermite::decide_zero_dimensional(formula, gens, 64)
    })?;
    Ok(decided == Some(Some(false)))
}

fn is_negated(lit: &Basic, head: fn(&Head) -> bool) -> bool {
    matches!(lit.head(), Head::Not) && head(lit.args()[0].head())
}

fn real_set(h: &Head) -> bool {
    matches!(
        h,
        Head::Naturals | Head::Naturals0 | Head::Integers | Head::Rationals | Head::Reals
    )
}

fn theory_check(lits: &[Basic], facts: &Facts) -> PolyResult<Check> {
    use crate::polys::cad::qe::sample_points;
    let mut predicates: Vec<Basic> = Vec::new();
    for lit in lits {
        if let Some(p) = to_predicates(lit)? {
            predicates.push(p);
        }
        if !is_compound(lit) {
            predicates.extend(consequences(lit)?);
        }
    }
    predicates.push(facts.predicates.clone());
    if !crate::assumptions::satask::predicates_consistent(&and_(&predicates))? {
        return Ok(Check::Unsatisfiable);
    }
    let mut real: BTreeSet<Basic> = facts.real.iter().cloned().collect();
    let mut integer: BTreeSet<Basic> = facts.integer.iter().cloned().collect();
    for lit in lits {
        if matches!(lit.head(), Head::Relational(op) if !matches!(op, RelOp::Eq | RelOp::Ne)) {
            real.extend(symbols(lit));
        }
        if matches!(lit.head(), Head::Contains) {
            let (x, s) = (&lit.args()[0], &lit.args()[1]);
            if x.is_symbol() {
                if real_set(s.head()) {
                    real.insert(x.clone());
                }
                if !matches!(s.head(), Head::Reals) {
                    integer.insert(x.clone());
                }
            }
        }
    }
    let mut candidates: BTreeSet<Basic> = real.clone();
    for lit in lits {
        let atom: &Basic = if matches!(lit.head(), Head::Not) {
            &lit.args()[0]
        } else {
            lit
        };
        if matches!(atom.head(), Head::Relational(_)) {
            candidates.extend(symbols(atom));
        }
    }
    let candidates: Vec<Basic> = candidates.into_iter().collect();
    let mut polynomial: Vec<Basic> = Vec::new();
    let mut unknown: bool = false;
    for lit in lits {
        match to_polynomial(lit, &candidates)? {
            None => {
                if (matches!(lit.head(), Head::Contains) || is_negated(lit, |h| matches!(h, Head::Contains)))
                    && to_predicates(lit)?.is_some()
                {
                    continue;
                }
                if matches!(lit.head(), Head::AppliedPredicate)
                    || is_negated(lit, |h| matches!(h, Head::AppliedPredicate))
                {
                    continue;
                }
                unknown = true;
            }
            Some(p) => {
                if !is_true(&p) {
                    polynomial.push(p);
                }
            }
        }
    }
    if !is_true(&facts.polynomial) {
        polynomial.push(facts.polynomial.clone());
    }
    let mut witness: Witness = Vec::new();
    if !polynomial.is_empty() {
        let formula: Basic = and_(&polynomial);
        let gens: Vec<Basic> = sorted_symbols(&symbols(&formula));
        let all_real: bool = gens.iter().all(|g| real.contains(g));
        if all_real && without_real_solution(&formula, &gens)? {
            return Ok(Check::Unsatisfiable);
        }
        let points: Vec<Witness> = sample_points(&formula, &gens, None, true)?;
        if points.is_empty() {
            return Ok(if all_real { Check::Unsatisfiable } else { Check::Unknown });
        }
        let integer_gens: Vec<&Basic> = gens.iter().filter(|g| integer.contains(*g)).collect();
        if !integer_gens.is_empty() {
            let value = |p: &Witness, g: &Basic| p.iter().find(|(s, _)| s == g).map(|(_, v)| v.clone());
            let exact: Option<&Witness> = points.iter().find(|p| {
                integer_gens
                    .iter()
                    .all(|g| value(p, g).is_some_and(|v| is(&v, Predicate::integer) == Some(true)))
            });
            match exact {
                Some(p) => witness = p.clone(),
                None => match integer_witness(&formula, &points, &integer, &gens)? {
                    None => return Ok(Check::Unknown),
                    Some(None) => return Ok(Check::Unsatisfiable),
                    Some(Some(w)) => witness = w,
                },
            }
        } else {
            witness = points[0].clone();
        }
    }
    if unknown {
        return Ok(Check::Unknown);
    }
    Ok(Check::Witness(witness))
}

fn prepared(formula: &Basic, assumptions: &[Basic], domain: Option<&Basic>) -> PolyResult<(Basic, Facts)> {
    let mut f: Basic = formula.clone();
    if f.preorder().iter().any(crate::assumptions::quantifiers::is_quantifier) {
        f = crate::assumptions::resolve::resolve(&f, domain, assumptions, None)?;
    }
    let syms: Vec<Basic> = f.free_symbols().into_iter().collect();
    let facts: Facts = facts_of(assumptions, domain, &syms)?;
    Ok((normalize(&f)?, facts))
}

fn models_of(formula: &Basic, facts: &Facts, all: bool) -> PolyResult<(Vec<Model>, bool, Option<bool>)> {
    // (models, unknown, decided by the abstraction alone: Some(false) for
    // an unsatisfiable one, Some(true) for a valid one)
    let (abstract_formula, atoms) = abstract_(formula)?;
    if is_false(&abstract_formula) {
        return Ok((vec![], false, Some(false)));
    }
    if is_true(&abstract_formula) {
        return Ok((vec![vec![]], false, Some(true)));
    }
    let clauses = crate::logic::cnf::to_clauses(&abstract_formula, None);
    let mut results: Vec<Model> = Vec::new();
    let mut unknown: bool = false;
    for model in crate::logic::dpll2::propositional_models(&clauses, None) {
        let lits: Vec<Basic> = literals(&model, &atoms);
        match theory_check(&lits, facts)? {
            Check::Unknown => {
                unknown = true;
                continue;
            }
            Check::Unsatisfiable => continue,
            Check::Witness(w) => {
                let mut full: Model = model
                    .iter()
                    .filter(|(s, _)| !atoms.iter().any(|(d, _)| d == s))
                    .map(|(s, v)| (s.clone(), Value::Bool(*v)))
                    .collect();
                for (k, v) in w {
                    match full.iter_mut().find(|(s, _)| *s == k) {
                        Some(entry) => entry.1 = Value::Expr(v),
                        None => full.push((k, Value::Expr(v))),
                    }
                }
                results.push(full);
                if !all {
                    return Ok((results, unknown, None));
                }
            }
        }
    }
    Ok((results, unknown, None))
}

/// Whether a formula can be true (`satisfiable`; Mathematica's
/// `SatisfiableQ` with `FindInstance`).
pub fn satisfiable(formula: &Basic, assumptions: &[Basic], domain: Option<&Basic>) -> PolyResult<Sat> {
    let (f, facts) = prepared(formula, assumptions, domain)?;
    let (mut models, unknown, _) = models_of(&f, &facts, false)?;
    if let Some(m) = models.pop() {
        return Ok(Sat::Model(m));
    }
    Ok(if unknown { Sat::Unknown } else { Sat::Unsatisfiable })
}

/// Every model consistent with the theory (`satisfiable(...,
/// all_models=True)`); `None` when undecided.
pub fn all_models(formula: &Basic, assumptions: &[Basic], domain: Option<&Basic>) -> PolyResult<Option<Vec<Model>>> {
    let (f, facts) = prepared(formula, assumptions, domain)?;
    let (models, unknown, _) = models_of(&f, &facts, true)?;
    Ok(if unknown && models.is_empty() {
        None
    } else {
        Some(models)
    })
}

/// Whether a formula is always true (`tautology`; `TautologyQ`).
pub fn tautology(formula: &Basic, assumptions: &[Basic], domain: Option<&Basic>) -> PolyResult<Option<bool>> {
    Ok(match satisfiable(&not_(formula), assumptions, domain)? {
        Sat::Unknown => None,
        Sat::Unsatisfiable => Some(true),
        Sat::Model(_) => Some(false),
    })
}

/// Values of the variables satisfying a formula (`find_instance`;
/// Mathematica's `FindInstance`); `domain` `None` is no domain.
pub fn find_instance(
    formula: &Basic,
    variables: &[Basic],
    domain: Option<&Basic>,
    assumptions: &[Basic],
    count: usize,
) -> PolyResult<Option<Vec<Witness>>> {
    for v in variables {
        if !v.is_symbol() {
            return Err(PolyError::Type(format!("expected a symbol, got {}", v)));
        }
    }
    if let Some(d) = domain {
        if !crate::sets::sets::is_set(d) {
            return Err(PolyError::Type(format!("domain must be a set, got {}", d)));
        }
    }
    let mut f: Basic = formula.clone();
    if f.preorder().iter().any(crate::assumptions::quantifiers::is_quantifier) {
        f = crate::assumptions::resolve::resolve(&f, Some(&crate::sets::sets::reals()), assumptions, None)?;
    }
    let facts: Facts = facts_of(assumptions, None, variables)?;
    f = normalize(&and_(&[f, facts.formula.clone()]))?;
    let vars: BTreeSet<Basic> = variables.iter().cloned().collect();
    if domain.is_some_and(|d| matches!(d.head(), Head::Reals)) {
        let mut real: BTreeSet<Basic> = facts.real.iter().cloned().collect();
        real.extend(vars.iter().cloned());
        let real: Vec<Basic> = real.into_iter().collect();
        if let Some(poly) = to_polynomial(&f, &real)? {
            if poly.free_symbols().iter().all(|s| vars.contains(s)) {
                if is_true(&poly) {
                    return Ok(Some(vec![variables
                        .iter()
                        .map(|v| (v.clone(), Basic::int(0)))
                        .collect()]));
                }
                if is_false(&poly) {
                    return Ok(Some(vec![]));
                }
                let mut points = crate::polys::cad::qe::sample_points(&poly, variables, None, true)?;
                points.truncate(count);
                return Ok(Some(points));
            }
        }
    }
    let Some(models) = all_models(&f, &[], domain)? else {
        return Ok(None);
    };
    let mut results: Vec<Witness> = Vec::new();
    for model in models {
        let mut assignment: Vec<(Basic, Basic)> = model
            .iter()
            .map(|(s, v)| {
                (
                    s.clone(),
                    match v {
                        Value::Bool(b) => boolean(*b),
                        Value::Expr(e) => e.clone(),
                    },
                )
            })
            .collect();
        let mut instance: Witness = Vec::new();
        for v in variables {
            let found: Option<&Basic> = assignment.iter().find(|(s, _)| s == v).map(|(_, e)| e);
            let value: Basic = match found {
                Some(e) if !crate::core::logic::is_boolean(e) => e.clone(),
                _ => Basic::int(0),
            };
            instance.push((v.clone(), value));
        }
        for (k, val) in &instance {
            match assignment.iter_mut().find(|(s, _)| s == k) {
                Some(entry) => entry.1 = val.clone(),
                None => assignment.push((k.clone(), val.clone())),
            }
        }
        if is_false(&subs_all(&f, &assignment)?) {
            continue;
        }
        if !results.contains(&instance) {
            results.push(instance);
        }
        if results.len() >= count {
            break;
        }
    }
    Ok(Some(results))
}
