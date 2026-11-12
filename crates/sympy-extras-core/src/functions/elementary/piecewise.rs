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
//! Not ported: `piecewise_intervals`, `piecewise_simplify` (it simplifies
//! the conditions through SymPy), the leading term.

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
fn as_coeff_mul_rational(e: &Basic) -> (Basic, Basic) {
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
