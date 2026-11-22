//! `Piecewise`, ported from `sympy_extras.functions.elementary.piecewise`:
//! the node `Piecewise(ExprCondPair(e, c), ...)` and the evaluating
//! `piecewise`, which drops the false conditions and the pairs after a true
//! one, returns the expression of a single true pair (`nan` when no pair is
//! left), makes the relations canonical with their leading coefficient
//! divided out (`-2*x + 1 < 0` is `x > 1/2`), drops repeated conditions and
//! the redundant parts of a conjunction, collapses a nested `Piecewise`
//! with the same condition, and merges successive pairs with the same
//! expression (their conditions joined by `Or`). The derivative, the
//! conjugate and a power apply to the expressions; the facts are those all
//! the expressions agree on. `piecewise_fold` brings the `Piecewise` of a
//! sum, product or function to the top.
//!
//! `piecewise_intervals` (SymPy's `Piecewise._intervals`) and
//! `piecewise_simplify` are ported; not ported: the leading term.

// PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py (Piecewise, ExprCondPair,
// _piecewise_collapse_arguments, piecewise_fold), sympy/core/relational.py
// (_canonical_coeff), sympy/logic/boolalg.py (distribute_and_over_or) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::{HashMap, HashSet};

use crate::core::add::sub;
use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::{as_coeff_add, is_true};
use crate::core::function::rebuild_evaluated;
use crate::core::logic::{and_, false_, is_boolean, or_, true_, truth};
use crate::core::mul::div;
use crate::core::parameters::evaluate;
use crate::core::relational::{canonical, evaluate_relation, negated, rel};
use crate::core::sorting::{sort_key, SortKey};
use crate::error::{value_error, PolyError, PolyResult};

/// The pair `ExprCondPair(expr, cond)`.
pub fn expr_cond_pair(expr: &Basic, cond: &Basic) -> Basic {
    Basic::new(Head::ExprCondPair, vec![expr.clone(), cond.clone()])
}

/// The unevaluated `Piecewise` of the pairs.
pub fn piecewise_node(pairs: Vec<Basic>) -> Basic {
    Basic::new(Head::Piecewise, pairs)
}

fn pair_expr(p: &Basic) -> &Basic {
    &p.args()[0]
}

fn pair_cond(p: &Basic) -> &Basic {
    &p.args()[1]
}

fn is_piecewise(e: &Basic) -> bool {
    matches!(e.head(), Head::Piecewise)
}

fn type_error(message: &str) -> PolyError {
    PolyError::Type(message.to_string())
}

/// Whether `e` is an expression (not a boolean, a set, a tuple or a pair).
fn is_expression(e: &Basic) -> bool {
    !(is_boolean(e)
        || crate::sets::sets::is_set(e)
        || matches!(e.head(), Head::Tuple | Head::ExprCondPair | Head::Contains))
}

/// `ExprCondPair(expr, cond)` with its arguments checked (`_evaluated`).
pub fn checked_pair(expr: &Basic, cond: &Basic) -> PolyResult<Basic> {
    if !is_expression(expr) || !is_boolean(cond) {
        return Err(type_error("ExprCondPair takes an expression and a condition"));
    }
    Ok(expr_cond_pair(expr, cond))
}

/// The piecewise expression of the `(expr, cond)` pairs, evaluated.
pub fn piecewise(pairs: &[(Basic, Basic)]) -> PolyResult<Basic> {
    let mut given: Vec<Basic> = Vec::new();
    for (e, c) in pairs {
        given.push(checked_pair(e, c)?);
    }
    piecewise_of_pairs(given)
}

/// The piecewise expression of the `ExprCondPair` nodes, evaluated (the
/// rebuild of a `Piecewise`).
pub fn piecewise_of_pairs(given: Vec<Basic>) -> PolyResult<Basic> {
    if given.is_empty() {
        return Err(type_error("At least one (expr, cond) pair expected."));
    }
    if given.iter().any(|p| !matches!(p.head(), Head::ExprCondPair)) {
        return Err(type_error("Piecewise takes (expr, cond) pairs"));
    }
    if !evaluate() {
        return Ok(piecewise_node(given));
    }
    evaluate_pairs(given)
}

fn evaluate_pairs(given: Vec<Basic>) -> PolyResult<Basic> {
    // SymPy's __new__: the false conditions dropped, nothing after a true one
    let mut kept: Vec<Basic> = Vec::new();
    for p in given {
        match pair_cond(&p).head() {
            Head::BooleanFalse => continue,
            Head::BooleanTrue => {
                kept.push(p);
                break;
            }
            _ => kept.push(p),
        }
    }
    if let Some(value) = eval(&kept)? {
        return Ok(value);
    }
    Ok(piecewise_node(kept))
}

/// `Piecewise.eval`.
fn eval(args: &[Basic]) -> PolyResult<Option<Basic>> {
    if args.is_empty() {
        return Ok(Some(Basic::nan()));
    }
    if args.len() == 1 && matches!(pair_cond(&args[0]).head(), Head::BooleanTrue) {
        return Ok(Some(pair_expr(&args[0]).clone()));
    }
    let collapsed: Vec<Basic> = collapse_arguments(args)?;
    if collapsed.is_empty() {
        return Err(value_error(
            "There are no conditions (or none that are not trivially false) to define an expression.".to_string(),
        ));
    }
    if collapsed.as_slice() != args {
        return Ok(Some(evaluate_pairs(collapsed)?));
    }
    Ok(None)
}

/// `as_coeff_Mul(rational=True)`.
pub(crate) fn as_coeff_mul_rational(e: &Basic) -> (Basic, Basic) {
    if e.as_rational().is_some_and(|v| !v.is_zero()) {
        return (e.clone(), Basic::one());
    }
    if e.is_mul() && e.args()[0].as_rational().is_some() {
        return (e.args()[0].clone(), crate::core::mul::raw_mul(e.args()[1..].to_vec()));
    }
    (Basic::one(), e.clone())
}

fn op_of(r: &Basic) -> RelOp {
    match r.head() {
        Head::Relational(op) => *op,
        _ => unreachable!("a relation"),
    }
}

/// `_canonical_coeff`: `-2*x + 1 < 0` as `x > 1/2`.
fn canonical_coeff(r: &Basic) -> PolyResult<Basic> {
    let r: Basic = canonical(r)?;
    let (lhs, rhs) = (&r.args()[0], &r.args()[1]);
    let (b, rest) = {
        let (c, rest) = as_coeff_add(lhs);
        if c.as_rational().is_some() {
            (c, rest)
        } else {
            (Basic::zero(), lhs.clone())
        }
    };
    let (m, lhs) = as_coeff_mul_rational(&rest);
    if m == Basic::zero() {
        // a relation between numbers (unevaluated Eq(1, 1)): SymPy divides
        // by m = 0 and gets Eq(0, nan), False, so that its Piecewise drops a
        // true condition (issue #25); it is evaluated instead
        return evaluate_relation(op_of(&r), &r.args()[0], &r.args()[1]);
    }
    let rhs: Basic = div(&sub(rhs, &b)?, &m)?;
    let mut op: RelOp = op_of(&r);
    if is_true(&m, Predicate::negative) {
        op = op.reversed();
    }
    evaluate_relation(op, &lhs, &rhs)
}

/// Every relation of `c` replaced by its canonical coefficient form.
fn canonical_conditions(c: &Basic) -> PolyResult<Basic> {
    let mut relations: HashMap<Basic, Basic> = HashMap::new();
    let mut stack: Vec<Basic> = vec![c.clone()];
    while let Some(node) = stack.pop() {
        if matches!(node.head(), Head::Relational(_)) {
            if let std::collections::hash_map::Entry::Vacant(slot) = relations.entry(node.clone()) {
                slot.insert(canonical_coeff(&node)?);
            }
        } else if is_boolean(&node) {
            stack.extend(node.args().iter().cloned());
        }
    }
    if relations.is_empty() {
        return Ok(c.clone());
    }
    if matches!(c.head(), Head::Relational(_)) {
        return Ok(relations[c].clone());
    }
    // xreplace: the constructors do not evaluate
    let replaced: Basic = c.xreplace(&relations);
    if !is_boolean(&replaced) {
        return Err(type_error("a condition is expected"));
    }
    Ok(replaced)
}

/// The conjunctive normal form by distribution of `Or` over `And`.
fn distribute_and_over_or(e: &Basic) -> Basic {
    match e.head() {
        Head::Or => {
            for (k, arg) in e.args().iter().enumerate() {
                if matches!(arg.head(), Head::And) {
                    let others: Vec<Basic> = e
                        .args()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != k)
                        .map(|(_, a)| a.clone())
                        .collect();
                    let rest: Basic = or_(&others);
                    let parts: Vec<Basic> = arg
                        .args()
                        .iter()
                        .map(|c| distribute_and_over_or(&or_(&[c.clone(), rest.clone()])))
                        .collect();
                    return and_(&parts);
                }
            }
            e.clone()
        }
        Head::And => and_(&e.args().iter().map(distribute_and_over_or).collect::<Vec<Basic>>()),
        _ => e.clone(),
    }
}

fn weak(r: &Basic) -> Option<Basic> {
    match op_of(r) {
        RelOp::Lt => Some(rel(RelOp::Le, &r.args()[0], &r.args()[1])),
        RelOp::Gt => Some(rel(RelOp::Ge, &r.args()[0], &r.args()[1])),
        _ => None,
    }
}

/// `_piecewise_collapse_arguments`.
fn collapse_arguments(args: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut newargs: Vec<Basic> = Vec::new();
    let mut current: HashSet<Basic> = HashSet::new();
    for pair in args {
        let mut expr: Basic = pair_expr(pair).clone();
        let mut cond: Basic = canonical_conditions(pair_cond(pair))?;
        if is_piecewise(&expr) {
            let mut unmatching: Vec<Basic> = Vec::new();
            for inner in expr.clone().args() {
                if current.contains(pair_cond(inner)) {
                    continue;
                }
                if *pair_cond(inner) == cond {
                    if !matches!(pair_cond(inner).head(), Head::BooleanTrue) {
                        expr = if unmatching.is_empty() {
                            pair_expr(inner).clone()
                        } else {
                            let mut pairs: Vec<Basic> = unmatching.clone();
                            pairs.push(inner.clone());
                            evaluate_pairs(pairs)?
                        };
                    }
                    break;
                }
                unmatching.push(inner.clone());
            }
        }
        let mut candidates: Vec<Basic> = vec![cond.clone()];
        if matches!(cond.head(), Head::And) {
            candidates.extend(cond.args().iter().cloned());
        }
        if candidates.iter().any(|c| is_boolean(c) && current.contains(c)) {
            continue;
        }
        if matches!(cond.head(), Head::And) {
            let mut nonredundant: Vec<Basic> = Vec::new();
            let mut contradicted: bool = false;
            for c in cond.args() {
                if matches!(c.head(), Head::Relational(_)) {
                    if current.contains(&canonical(&negated(c))?) {
                        continue;
                    }
                    if let Some(w) = weak(c) {
                        if current.contains(&w) {
                            contradicted = true;
                            break;
                        }
                    }
                }
                nonredundant.push(c.clone());
            }
            cond = if contradicted { false_() } else { and_(&nonredundant) };
        } else if matches!(cond.head(), Head::Relational(_)) && current.contains(&canonical(&negated(&cond))?) {
            cond = true_();
        }
        current.insert(cond.clone());
        if let Some(last) = newargs.last() {
            if *pair_expr(last) == expr {
                let mut joined: Basic = or_(&[cond.clone(), pair_cond(last).clone()]);
                if matches!(joined.head(), Head::And | Head::Or) {
                    joined = distribute_and_over_or(&joined);
                }
                let n: usize = newargs.len();
                newargs[n - 1] = expr_cond_pair(&expr, &joined);
                continue;
            }
            if *pair_cond(last) == cond {
                continue;
            }
        }
        newargs.push(expr_cond_pair(&expr, &cond));
    }
    Ok(newargs)
}

/// Whether `e` contains a `Piecewise`.
pub fn has_piecewise(e: &Basic) -> bool {
    let mut stack: Vec<&Basic> = vec![e];
    while let Some(node) = stack.pop() {
        if is_piecewise(node) {
            return true;
        }
        stack.extend(node.args().iter());
    }
    false
}

fn common_prefix(lists: &[Vec<Basic>]) -> Vec<Basic> {
    let mut prefix: Vec<Basic> = Vec::new();
    let length: usize = lists.iter().map(Vec::len).min().unwrap_or(0);
    for i in 0..length {
        let first: &Basic = &lists[0][i];
        if lists.iter().all(|l| l[i] == *first) {
            prefix.push(first.clone());
        } else {
            break;
        }
    }
    prefix
}

fn as_expression(e: Basic) -> PolyResult<Basic> {
    if !is_expression(&e) {
        return Err(type_error("an expression is expected"));
    }
    Ok(e)
}

/// `piecewise_fold(expr, evaluate)`: the `Piecewise` parts of `expr`
/// brought to the top (`evaluate` `Some(true)` evaluates the result,
/// `Some(false)` keeps the node, `None` also drops the repeated conditions
/// and unwraps a single true pair, as SymPy's).
pub fn piecewise_fold(expr: &Basic, evaluate: Option<bool>) -> PolyResult<Basic> {
    if !has_piecewise(expr) {
        return Ok(expr.clone());
    }
    let mut new_args: Vec<(Basic, Basic)> = Vec::new();
    if is_piecewise(expr) {
        for pair in expr.args() {
            let (mut e, c) = (pair_expr(pair).clone(), pair_cond(pair).clone());
            if !is_piecewise(&e) {
                e = piecewise_fold(&e, Some(true))?;
            }
            if is_piecewise(&e) {
                for inner in e.args() {
                    new_args.push((
                        piecewise_fold(pair_expr(inner), Some(true))?,
                        and_(&[pair_cond(inner).clone(), c.clone()]),
                    ));
                }
            } else {
                new_args.push((e, c));
            }
        }
    } else {
        let parts: Vec<Basic> = if expr.is_add() || (expr.is_mul() && is_true(expr, Predicate::commutative)) {
            let pieces: Vec<&Basic> = expr.args().iter().filter(|a| is_piecewise(a)).collect();
            let mut args: Vec<Basic> = expr.args().iter().filter(|a| !is_piecewise(a)).cloned().collect();
            let mut keys: Vec<Vec<Basic>> = Vec::new();
            let mut groups: HashMap<Vec<Basic>, Vec<&Basic>> = HashMap::new();
            for piece in pieces {
                let key: Vec<Basic> = piece.args().iter().map(|p| pair_cond(p).clone()).collect();
                if !groups.contains_key(&key) {
                    keys.push(key.clone());
                }
                groups.entry(key).or_default().push(piece);
            }
            keys.sort_by_cached_key(|cs| cs.iter().map(sort_key).collect::<Vec<SortKey>>());
            for key in keys {
                let group: &Vec<&Basic> = &groups[&key];
                if group.len() > 1 {
                    let pargs: Vec<Vec<Basic>> = group.iter().map(|g| g.args().to_vec()).collect();
                    let conds: Vec<Vec<Basic>> = pargs
                        .iter()
                        .map(|a| a.iter().map(|p| pair_cond(p).clone()).collect())
                        .collect();
                    let common: Vec<Basic> = common_prefix(&conds);
                    let n: usize = common.len();
                    let mut collected: Vec<Basic> = Vec::new();
                    for (i, c) in common.iter().enumerate() {
                        let exprs: Vec<Basic> = pargs.iter().map(|a| pair_expr(&a[i]).clone()).collect();
                        collected.push(checked_pair(&as_expression(rebuild_evaluated(expr, exprs)?)?, c)?);
                    }
                    let mut remains: Vec<Basic> = Vec::new();
                    for a in &pargs {
                        if n == a.len() {
                            continue;
                        }
                        if matches!(pair_cond(&a[n]).head(), Head::BooleanTrue) {
                            remains.push(pair_expr(&a[n]).clone());
                        } else {
                            remains.push(piecewise_node(a[n..].to_vec()));
                        }
                    }
                    if !remains.is_empty() {
                        collected.push(checked_pair(
                            &as_expression(rebuild_evaluated(expr, remains)?)?,
                            &true_(),
                        )?);
                    }
                    args.push(piecewise_node(collected));
                    continue;
                }
                args.extend(group.iter().map(|g| (*g).clone()));
            }
            args
        } else {
            expr.args().to_vec()
        };
        let mut options: Vec<Vec<(Basic, Basic)>> = Vec::new();
        for part in &parts {
            let f: Basic = if is_expression(part) {
                piecewise_fold(part, Some(true))?
            } else {
                part.clone()
            };
            if is_piecewise(&f) {
                options.push(
                    f.args()
                        .iter()
                        .map(|p| (pair_expr(p).clone(), pair_cond(p).clone()))
                        .collect(),
                );
            } else {
                options.push(vec![(f, true_())]);
            }
        }
        // itertools.product: the last option varies fastest
        let mut index: Vec<usize> = vec![0; options.len()];
        loop {
            let expressions: Vec<Basic> = options.iter().zip(&index).map(|(o, &i)| o[i].0.clone()).collect();
            let conditions: Vec<Basic> = options.iter().zip(&index).map(|(o, &i)| o[i].1.clone()).collect();
            let rebuilt: Basic = as_expression(rebuild_evaluated(expr, expressions)?)?;
            new_args.push((rebuilt, and_(&conditions)));
            let mut done: bool = true;
            for k in (0..options.len()).rev() {
                index[k] += 1;
                if index[k] < options[k].len() {
                    done = false;
                    break;
                }
                index[k] = 0;
            }
            if done {
                break;
            }
        }
    }
    if evaluate.is_none() {
        // the first expression of each condition, in the order of the last
        // occurrences (a dict filled from the end)
        let mut order: Vec<Basic> = Vec::new();
        let mut seen: HashMap<Basic, Basic> = HashMap::new();
        for (e, c) in new_args.iter().rev() {
            if !seen.contains_key(c) {
                order.push(c.clone());
            }
            seen.insert(c.clone(), e.clone());
        }
        new_args = order.into_iter().rev().map(|c| (seen[&c].clone(), c)).collect();
    }
    let result: Basic = if evaluate != Some(false) {
        piecewise(&new_args)?
    } else {
        let mut pairs: Vec<Basic> = Vec::new();
        for (e, c) in &new_args {
            pairs.push(checked_pair(e, c)?);
        }
        if pairs.is_empty() {
            return Err(type_error("At least one (expr, cond) pair expected."));
        }
        piecewise_node(pairs)
    };
    if evaluate.is_none()
        && is_piecewise(&result)
        && result.args().len() == 1
        && truth(pair_cond(&result.args()[0])) == Some(true)
    {
        return Ok(pair_expr(&result.args()[0]).clone());
    }
    if is_piecewise(&result) && result.args().iter().any(|p| has_piecewise(pair_expr(p))) {
        return piecewise_fold(&result, Some(true));
    }
    Ok(result)
}

/// `f` applied to every expression, the conditions kept, evaluated
/// (`map_expressions`).
pub fn map_expressions(e: &Basic, f: &dyn Fn(&Basic) -> PolyResult<Basic>) -> PolyResult<Basic> {
    let mut pairs: Vec<(Basic, Basic)> = Vec::new();
    for p in e.args() {
        pairs.push((f(pair_expr(p))?, pair_cond(p).clone()));
    }
    piecewise(&pairs)
}

/// Each piece raised (`Piecewise._eval_power`).
pub fn piecewise_power(base: &Basic, exp: &Basic) -> PolyResult<Basic> {
    map_expressions(base, &|e: &Basic| crate::core::power::pow(e, exp))
}

/// The facts all the expressions agree on (`_eval_facts`).
pub fn piecewise_facts(e: &Basic) -> FactList {
    use Predicate as P;
    const AGREEING: [Predicate; 22] = [
        P::finite,
        P::complex,
        P::even,
        P::imaginary,
        P::integer,
        P::irrational,
        P::negative,
        P::nonnegative,
        P::nonpositive,
        P::nonzero,
        P::odd,
        P::polar,
        P::positive,
        P::extended_real,
        P::extended_positive,
        P::extended_negative,
        P::extended_nonzero,
        P::extended_nonpositive,
        P::extended_nonnegative,
        P::real,
        P::zero,
        P::commutative,
    ];
    let sub_facts: Vec<_> = e.args().iter().map(|p| facts_of(pair_expr(p))).collect();
    let mut facts: FactList = FactList::new();
    for name in AGREEING {
        let values: Vec<Option<bool>> = sub_facts.iter().map(|s| s.get(&name).copied()).collect();
        if values.iter().all(|v| *v == Some(true)) {
            facts.set(name, true);
        } else if values.iter().all(|v| *v == Some(false)) {
            facts.set(name, false);
        }
    }
    facts
}

// ---------------------------------------------------------------------
// the intervals of a Piecewise and piecewise_simplify

/// The pairs `(expression, condition)` of a `Piecewise` node.
fn pairs_of(pw: &Basic) -> Vec<(Basic, Basic)> {
    pw.args()
        .iter()
        .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
        .collect()
}

fn is_rel(e: &Basic, ops: &[RelOp]) -> bool {
    matches!(e.head(), Head::Relational(op) if ops.contains(op))
}

/// SymPy's `r.lts` and `r.gts`: the smaller and the greater side
/// (`_lts_gts`).
fn lts_gts(r: &Basic) -> (Basic, Basic) {
    if is_rel(r, &[RelOp::Lt, RelOp::Le]) {
        (r.args()[0].clone(), r.args()[1].clone())
    } else {
        (r.args()[1].clone(), r.args()[0].clone())
    }
}

/// The negations pushed to the relations and `Xor`, `Implies`,
/// `Equivalent` and `ITE` written with `And` and `Or` (`_to_nnf`).
fn to_nnf_pw(b: &Basic) -> PolyResult<Basic> {
    use crate::core::logic::not_;
    let xor_clauses = |args: &[Basic], first: usize| -> Vec<Basic> {
        let mut clauses: Vec<Basic> = Vec::new();
        let n: usize = args.len();
        let mut i: usize = first;
        while i <= n {
            for negated in combinations_of(n, i) {
                let parts: Vec<Basic> = args
                    .iter()
                    .enumerate()
                    .map(|(k, s)| if negated.contains(&k) { not_(s) } else { s.clone() })
                    .collect();
                clauses.push(or_(&parts));
            }
            i += 2;
        }
        clauses
    };
    let all =
        |v: &[Basic], f: &dyn Fn(&Basic) -> PolyResult<Basic>| -> PolyResult<Vec<Basic>> { v.iter().map(f).collect() };
    match b.head() {
        Head::Not => {
            if crate::logic::boolalg::is_literal(b) {
                return Ok(b.clone());
            }
            let inner: &Basic = &b.args()[0];
            match inner.head() {
                Head::And => Ok(or_(&all(inner.args(), &|a| to_nnf_pw(&not_(a)))?)),
                Head::Or => Ok(and_(&all(inner.args(), &|a| to_nnf_pw(&not_(a)))?)),
                Head::Implies => Ok(and_(&[
                    to_nnf_pw(&inner.args()[0])?,
                    to_nnf_pw(&not_(&inner.args()[1]))?,
                ])),
                Head::Equivalent => {
                    let negs: Vec<Basic> = inner.args().iter().map(not_).collect();
                    Ok(and_(&[to_nnf_pw(&or_(inner.args()))?, to_nnf_pw(&or_(&negs))?]))
                }
                Head::Xor => Ok(and_(&all(&xor_clauses(inner.args(), 1), &to_nnf_pw)?)),
                Head::Ite => {
                    let (c, t, e) = (&inner.args()[0], &inner.args()[1], &inner.args()[2]);
                    Ok(and_(&[
                        to_nnf_pw(&or_(&[c.clone(), not_(e)]))?,
                        to_nnf_pw(&or_(&[not_(c), not_(t)]))?,
                    ]))
                }
                _ => Ok(b.clone()),
            }
        }
        Head::And => Ok(and_(&all(b.args(), &to_nnf_pw)?)),
        Head::Or => Ok(or_(&all(b.args(), &to_nnf_pw)?)),
        Head::Xor => Ok(and_(&all(&xor_clauses(b.args(), 0), &to_nnf_pw)?)),
        Head::Implies => Ok(or_(&[to_nnf_pw(&not_(&b.args()[0]))?, to_nnf_pw(&b.args()[1])?])),
        Head::Equivalent => {
            let args: &[Basic] = b.args();
            let n: usize = args.len();
            let mut parts: Vec<Basic> = Vec::new();
            for k in 0..n {
                let (p, q) = (&args[k], &args[(k + 1) % n]);
                parts.push(to_nnf_pw(&or_(&[not_(p), q.clone()]))?);
            }
            Ok(and_(&parts))
        }
        Head::Ite => {
            let (c, t, e) = (&b.args()[0], &b.args()[1], &b.args()[2]);
            Ok(and_(&[
                to_nnf_pw(&or_(&[not_(c), t.clone()]))?,
                to_nnf_pw(&or_(&[c.clone(), e.clone()]))?,
            ]))
        }
        _ => Ok(b.clone()),
    }
}

/// The index sets of `itertools.combinations(range(n), k)`, in its order.
fn combinations_of(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    if k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i: isize = k as isize - 1;
        while i >= 0 && idx[i as usize] == i as usize + n - k {
            i -= 1;
        }
        if i < 0 {
            return out;
        }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// SymPy's `to_cnf` without simplification (`_to_cnf`).
fn to_cnf_pw(b: &Basic) -> PolyResult<Basic> {
    Ok(distribute_and_over_or(&to_nnf_pw(b)?))
}

/// The disjunctive normal form by distribution of `And` over `Or`
/// (`_distribute_or_over_and`).
fn distribute_or_over_and_pw(e: &Basic) -> Basic {
    match e.head() {
        Head::And => {
            for (k, arg) in e.args().iter().enumerate() {
                if matches!(arg.head(), Head::Or) {
                    let others: Vec<Basic> = e
                        .args()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != k)
                        .map(|(_, a)| a.clone())
                        .collect();
                    let rest: Basic = and_(&others);
                    let parts: Vec<Basic> = arg
                        .args()
                        .iter()
                        .map(|c| distribute_or_over_and_pw(&and_(&[c.clone(), rest.clone()])))
                        .collect();
                    return or_(&parts);
                }
            }
            e.clone()
        }
        Head::Or => or_(&e.args().iter().map(distribute_or_over_and_pw).collect::<Vec<_>>()),
        _ => e.clone(),
    }
}

/// `(lower, upper, expression, index)` of a piece on an interval.
pub type IntervalPiece = (Basic, Basic, Basic, usize);

/// The intervals of `sym` on which the pieces apply, or the reason a
/// condition cannot be written with intervals of `sym`
/// (`piecewise_intervals`, SymPy's `Piecewise._intervals`).
pub fn piecewise_intervals(pw: &Basic, sym: &Basic, err_on_eq: bool) -> PolyResult<Result<Vec<IntervalPiece>, String>> {
    use crate::core::relational::{gt, lt};
    use crate::functions::elementary::miscellaneous::{max_, min_};
    let oo: Basic = Basic::atom(Head::Infinity);
    let neg_oo: Basic = Basic::atom(Head::NegativeInfinity);
    let nonsymfail = |cond: &Basic| format!("A condition not involving {} appeared: {}", sym, cond);
    let mut relations: Vec<Basic> = Vec::new();
    for n in pw.preorder() {
        if matches!(n.head(), Head::Relational(_)) && !relations.contains(&n) {
            relations.push(n);
        }
    }
    let mut reps: HashMap<Basic, Basic> = HashMap::new();
    for r in crate::core::sorting::ordered(relations) {
        if !r.free_symbols().contains(sym) {
            return Ok(Err(nonsymfail(&r)));
        }
        let rv: Basic = match crate::solvers::inequalities::solve_inequality(&r, sym) {
            Ok(v) => v,
            Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {
                return Ok(Err(format!("Unable to solve relational {} for {}.", r, sym)))
            }
            Err(e) => return Err(e),
        };
        let rv: Basic = if matches!(rv.head(), Head::Relational(_)) {
            if rv.args()[0] != *sym || rv.args()[1].free_symbols().contains(sym) {
                return Ok(Err(format!("Unable to solve relational {} for {}.", r, sym)));
            }
            if is_rel(&rv, &[RelOp::Eq]) {
                false_()
            } else if is_rel(&rv, &[RelOp::Ne]) {
                match (lt(sym, &rv.args()[1]), gt(sym, &rv.args()[1])) {
                    (Ok(a), Ok(b)) => or_(&[a, b]),
                    (Err(PolyError::Type(_)), _) | (_, Err(PolyError::Type(_))) => true_(),
                    (Err(e), _) | (_, Err(e)) => return Err(e),
                }
            } else {
                rv
            }
        } else if rv == and_(&[lt(&neg_oo, sym)?, lt(sym, &oo)?]) {
            true_()
        } else {
            rv
        };
        reps.insert(r, rv);
    }
    let args: Vec<(Basic, Basic)> = pairs_of(pw).into_iter().map(|(e, c)| (e, c.xreplace(&reps))).collect();
    let mut expr_cond: Vec<(usize, Basic, Basic)> = Vec::new();
    let mut default: Option<(Basic, usize)> = None;
    for (i, (expr, cond)) in args.iter().enumerate() {
        if *cond == false_() {
            continue;
        }
        if *cond == true_() {
            default = Some((expr.clone(), i));
            break;
        }
        if is_rel(cond, &[RelOp::Eq]) {
            if err_on_eq {
                return Ok(Err(format!("encountered Eq condition: {}", cond)));
            }
            continue;
        }
        let mut cond: Basic = to_cnf_pw(cond)?;
        if matches!(cond.head(), Head::And) {
            cond = distribute_or_over_and_pw(&cond);
        }
        if matches!(cond.head(), Head::Or) {
            for o in cond.args() {
                if !is_rel(o, &[RelOp::Eq]) {
                    expr_cond.push((i, expr.clone(), o.clone()));
                }
            }
        } else if cond != false_() {
            expr_cond.push((i, expr.clone(), cond));
        }
    }
    let mut int_expr: Vec<IntervalPiece> = Vec::new();
    let mut k: usize = 0;
    while k < expr_cond.len() {
        let (iarg, expr, cond) = expr_cond[k].clone();
        k += 1;
        let (mut lower, mut upper): (Basic, Basic);
        if matches!(cond.head(), Head::And) {
            lower = neg_oo.clone();
            upper = oo.clone();
            let mut exclude: Vec<Basic> = Vec::new();
            for cond2 in cond.args() {
                if !matches!(cond2.head(), Head::Relational(_)) {
                    return Ok(Err("expecting only Relationals".to_string()));
                }
                if is_rel(cond2, &[RelOp::Eq]) {
                    lower = upper.clone();
                    if err_on_eq {
                        return Ok(Err("encountered secondary Eq condition".to_string()));
                    }
                    break;
                }
                if is_rel(cond2, &[RelOp::Ne]) {
                    let (l_, r_) = (&cond2.args()[0], &cond2.args()[1]);
                    if l_ == sym {
                        exclude.push(r_.clone());
                    } else if r_ == sym {
                        exclude.push(l_.clone());
                    } else {
                        return Ok(Err(nonsymfail(cond2)));
                    }
                    continue;
                }
                let (lts, gts) = lts_gts(cond2);
                if lts == *sym {
                    upper = min_(&[gts, upper])?;
                } else if gts == *sym {
                    lower = max_(&[lts, lower])?;
                } else {
                    return Ok(Err(nonsymfail(cond2)));
                }
            }
            if !exclude.is_empty() {
                let points: Vec<Basic> = crate::core::sorting::ordered(exclude);
                let mut previous: Basic = lower.clone();
                let mut bounds: Vec<(Basic, Basic)> = Vec::new();
                for e in points {
                    bounds.push((previous.clone(), e.clone()));
                    previous = e;
                }
                bounds.push((previous, upper.clone()));
                for (lo, hi) in bounds {
                    expr_cond.push((iarg, expr.clone(), and_(&[lt(&lo, sym)?, lt(sym, &hi)?])));
                }
                continue;
            }
        } else if matches!(cond.head(), Head::Relational(_)) && !is_rel(&cond, &[RelOp::Ne]) {
            (lower, upper) = lts_gts(&cond);
            if lower == *sym {
                lower = neg_oo.clone();
            } else if upper == *sym {
                upper = oo.clone();
            } else {
                return Ok(Err(nonsymfail(&cond)));
            }
        } else {
            return Ok(Err(format!("unrecognized condition: {}", cond)));
        }
        upper = max_(&[lower.clone(), upper])?;
        if err_on_eq && lower == upper {
            return Ok(Err("encountered Eq condition".to_string()));
        }
        if crate::core::relational::ge(&lower, &upper)? != true_() {
            int_expr.push((lower, upper, expr, iarg));
        }
    }
    if let Some((d, i)) = default {
        int_expr.push((neg_oo, oo, d, i));
    }
    let mut unique: Vec<IntervalPiece> = Vec::new();
    for piece in int_expr {
        if !unique.contains(&piece) {
            unique.push(piece);
        }
    }
    Ok(Ok(unique))
}

/// SymPy's `simplify` of a condition, natively (`_simplify_condition`).
pub(crate) fn simplify_condition(c: &Basic) -> PolyResult<Basic> {
    use crate::assumptions::refine::{measure, simplify_relational};
    let relation = |r: &Basic| simplify_relational(r, 1.7, false, true);
    crate::logic::boolsimp::simplify_boolean(c, &relation, &measure, 1.7, true)
}

/// The pieces simplified: for one variable the conditions rewritten as
/// disjoint intervals of it (`nan` where no piece applies), otherwise each
/// condition simplified; the expressions simplified
/// (`piecewise_simplify_arguments`).
pub fn piecewise_simplify_arguments(pw: &Basic, doit: bool) -> PolyResult<Basic> {
    use crate::core::relational::{ge, gt, le, lt};
    use crate::sets::sets::{complement, contains, empty_set, finite_set, inf, interval, is_subset, reals, union};
    let pairs: Vec<(Basic, Basic)> = pairs_of(pw);
    let first = pairs[0].1.free_symbols();
    let has_eq: bool = pw.preorder().iter().any(|n| is_rel(n, &[RelOp::Eq]));
    let mut new_args: Option<Vec<(Basic, Basic)>> = None;
    if first.len() == 1 && !has_eq {
        let x: Basic = first.into_iter().next().expect("one symbol");
        let include = |c: &Basic, a: &Basic| -> PolyResult<bool> {
            match crate::core::subs::subs(c, &x, a) {
                Ok(v) => Ok(v == true_()),
                Err(PolyError::Type(_)) => Ok(false),
                Err(e) => Err(e),
            }
        };
        let member = |s: &Basic, a: &Basic| -> PolyResult<bool> {
            let c: Basic = contains(s, a)?;
            if c == true_() {
                return Ok(true);
            }
            if c == false_() {
                return Ok(false);
            }
            Err(PolyError::Type(format!("did not evaluate to a bool: {}", c)))
        };
        let infinite = |e: &Basic| is_true(e, Predicate::infinite);
        if let Ok(abe) = piecewise_intervals(pw, &x, true)? {
            let mut out: Vec<(Basic, Basic)> = Vec::new();
            let mut covered: Basic = empty_set();
            for (a, b, e, i) in abe {
                let c: Basic = pairs[i].1.clone();
                let mut a: Basic = a;
                let mut incl_a: bool = include(&c, &a)?;
                let incl_b: bool = include(&c, &b)?;
                let iv: Basic = interval(&a, &b, !incl_a, !incl_b)?;
                let cset: Basic = complement(&iv, &covered)?;
                if cset == empty_set() {
                    continue;
                }
                match inf(&cset) {
                    Ok(v) => {
                        a = v;
                        incl_a = include(&c, &a)?;
                    }
                    Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {}
                    Err(e) => return Err(e),
                }
                let c: Basic = if incl_a && incl_b {
                    if infinite(&a) && infinite(&b) {
                        true_()
                    } else if infinite(&b) {
                        if member(&covered, &a)? {
                            gt(&x, &a)?
                        } else {
                            ge(&x, &a)?
                        }
                    } else if infinite(&a) {
                        le(&x, &b)?
                    } else if member(&covered, &a)? {
                        and_(&[lt(&a, &x)?, le(&x, &b)?])
                    } else {
                        and_(&[le(&a, &x)?, le(&x, &b)?])
                    }
                } else if incl_a {
                    if infinite(&a) {
                        lt(&x, &b)?
                    } else if member(&covered, &a)? {
                        and_(&[lt(&a, &x)?, lt(&x, &b)?])
                    } else {
                        and_(&[le(&a, &x)?, lt(&x, &b)?])
                    }
                } else if incl_b {
                    if infinite(&b) {
                        gt(&x, &a)?
                    } else {
                        and_(&[lt(&a, &x)?, le(&x, &b)?])
                    }
                } else if member(&covered, &a)? {
                    lt(&x, &b)?
                } else {
                    and_(&[lt(&a, &x)?, lt(&x, &b)?])
                };
                covered = union(&[covered, iv])?;
                if matches!(a.head(), Head::NegativeInfinity) && incl_a {
                    covered = union(&[covered, finite_set(&[Basic::atom(Head::NegativeInfinity)])])?;
                }
                if matches!(b.head(), Head::Infinity) && incl_b {
                    covered = union(&[covered, finite_set(std::slice::from_ref(&b))])?;
                }
                out.push((e, c));
            }
            if is_subset(&reals(), &covered)? != Some(true) {
                out.push((Basic::atom(Head::NaN), true_()));
            }
            new_args = Some(out);
        }
    }
    let new_args: Vec<(Basic, Basic)> = match new_args {
        Some(v) => v,
        None => pairs
            .iter()
            .map(|(e, c)| Ok((e.clone(), simplify_condition(c)?)))
            .collect::<PolyResult<_>>()?,
    };
    let _ = doit;
    let options = crate::simplify::simplify::Options {
        doit: false,
        ..crate::simplify::simplify::Options::default()
    };
    let mut fin: Vec<(Basic, Basic)> = Vec::new();
    for (e, c) in new_args {
        let newe: Basic = crate::simplify::simplify::simplify(&e, &options)?;
        fin.push((newe, c));
    }
    piecewise(&fin)
}

/// SymPy's `_diff_wrt`: a symbol, an undefined function, a derivative.
fn diff_wrt(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Symbol(_) | Head::Dummy(..) | Head::Undefined(_) | Head::Derivative
    )
}

fn blessed(e: &Basic) -> bool {
    let (l, r) = (&e.args()[0], &e.args()[1]);
    diff_wrt(l)
        && (diff_wrt(r)
            || r.as_rational().is_some()
            || matches!(
                r.head(),
                Head::Pi | Head::Exp1 | Head::EulerGamma | Head::Catalan | Head::GoldenRatio | Head::TribonacciConstant
            ))
}

fn eqs_and_others(cond: &Basic) -> (Vec<Basic>, Vec<Basic>) {
    if matches!(cond.head(), Head::And) {
        let eqs: Vec<Basic> = cond
            .args()
            .iter()
            .filter(|a| is_rel(a, &[RelOp::Eq]))
            .cloned()
            .collect();
        let others: Vec<Basic> = cond
            .args()
            .iter()
            .filter(|a| !is_rel(a, &[RelOp::Eq]))
            .cloned()
            .collect();
        return (eqs, others);
    }
    if is_rel(cond, &[RelOp::Eq]) {
        return (vec![cond.clone()], Vec::new());
    }
    (Vec::new(), Vec::new())
}

fn subs_pair(e: &Basic, eq: &Basic) -> PolyResult<Basic> {
    crate::core::subs::subs(e, &eq.args()[0], &eq.args()[1])
}

/// The expression of a piece with the equations of its condition
/// substituted (`_piecewise_simplify_eq_and`).
fn simplify_eq_and(mut args: Vec<(Basic, Basic)>) -> PolyResult<Vec<(Basic, Basic)>> {
    for slot in args.iter_mut() {
        let (mut expr, cond) = slot.clone();
        let (eqs, mut other) = eqs_and_others(&cond);
        if eqs.is_empty() {
            continue;
        }
        let mut eqs: Vec<Basic> = crate::core::sorting::ordered(eqs)
            .into_iter()
            .filter(|e| matches!(e.head(), Head::Relational(_)))
            .collect();
        let mut j: usize = 0;
        while j < eqs.len() {
            let e: Basic = eqs[j].clone();
            if blessed(&e) {
                expr = as_expression(subs_pair(&expr, &e)?)?;
                let mut rest: Vec<Basic> = Vec::new();
                for ei in &eqs[j + 1..] {
                    let r: Basic = subs_pair(ei, &e)?;
                    if matches!(r.head(), Head::Relational(_)) {
                        rest.push(r);
                    }
                }
                eqs.truncate(j + 1);
                eqs.extend(rest);
                other = other.iter().map(|ei| subs_pair(ei, &e)).collect::<PolyResult<_>>()?;
            }
            j += 1;
        }
        let mut all: Vec<Basic> = eqs;
        all.extend(other);
        *slot = (expr, and_(&all));
    }
    Ok(args)
}

/// A piece whose expression equals the next one under its equations takes
/// the next expression (`_piecewise_simplify_equal_to_next_segment`).
fn simplify_equal_to_next_segment(mut args: Vec<(Basic, Basic)>) -> PolyResult<Vec<(Basic, Basic)>> {
    let mut prevexpr: Option<Basic> = None;
    let n: usize = args.len();
    for i in (0..n).rev() {
        let (expr, cond) = args[i].clone();
        if let Some(prev) = prevexpr.clone() {
            let (eqs, other) = eqs_and_others(&cond);
            let (mut p, mut q) = (prev, expr.clone());
            if !eqs.is_empty() && other.is_empty() {
                for e in crate::core::sorting::ordered(eqs) {
                    if n == 2 || blessed(&e) {
                        p = as_expression(subs_pair(&p, &e)?)?;
                        q = as_expression(subs_pair(&q, &e)?)?;
                    }
                }
            }
            if p == q {
                args[i] = (args[i + 1].0.clone(), cond);
            } else {
                prevexpr = Some(expr);
            }
        } else {
            prevexpr = Some(expr);
        }
    }
    Ok(args)
}

/// A `Piecewise` simplified: its pieces, the equations of the conditions
/// substituted, the pieces equal to the next one merged
/// (`piecewise_simplify`).
pub fn piecewise_simplify(expr: &Basic, doit: bool) -> PolyResult<Basic> {
    if !matches!(expr.head(), Head::Piecewise) {
        return Ok(expr.clone());
    }
    let simplified: Basic = piecewise_simplify_arguments(expr, doit)?;
    if !matches!(simplified.head(), Head::Piecewise) {
        return Ok(simplified);
    }
    let args: Vec<(Basic, Basic)> = pairs_of(&simplified);
    let args = simplify_eq_and(args)?;
    let args = simplify_equal_to_next_segment(args)?;
    piecewise(&args)
}
