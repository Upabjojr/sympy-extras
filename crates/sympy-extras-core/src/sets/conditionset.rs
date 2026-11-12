//! Port of `sympy_extras/sets/conditionset.py`: [`Head::ConditionSet`],
//! the elements of a base set at which a condition holds, as a condition
//! on an unknown (a symbol or a tuple of symbols, bound), and its
//! evaluating function [`condition_set`].

use std::collections::{BTreeSet, HashMap};

use crate::core::basic::{Basic, Head};
use crate::core::function::xreplace_evaluated;
use crate::core::logic::{and_, false_, truth};
use crate::core::subs::subs;
use crate::error::{value_error, PolyError, PolyResult};

use super::sets::{contains, contains_node, empty_set, finite_set, intersection, union};

/// The symbols of the unknown: a symbol or a tuple of symbols
/// (`_unknowns`).
pub fn unknowns(sym: &Basic) -> PolyResult<Vec<Basic>> {
    if sym.is_symbol() {
        return Ok(vec![sym.clone()]);
    }
    if matches!(sym.head(), Head::Tuple) && sym.args().iter().all(|s| s.is_symbol()) {
        return Ok(sym.args().to_vec());
    }
    Err(PolyError::Type(format!(
        "the unknown of a ConditionSet is a symbol or a tuple of symbols, got {}",
        sym
    )))
}

/// The unevaluated `ConditionSet(sym, condition, base_set)`.
pub fn condition_set_node(sym: &Basic, condition: &Basic, base_set: &Basic) -> Basic {
    Basic::new(
        Head::ConditionSet,
        vec![sym.clone(), condition.clone(), base_set.clone()],
    )
}

/// The free symbols: those of the condition but the unknowns, and those of
/// the base set.
pub fn condition_set_free_symbols(e: &Basic) -> BTreeSet<Basic> {
    let (sym, condition, base) = (&e.args()[0], &e.args()[1], &e.args()[2]);
    let bound: Vec<Basic> = unknowns(sym).unwrap_or_default();
    let mut out: BTreeSet<Basic> = condition
        .free_symbols()
        .into_iter()
        .filter(|s| !bound.contains(s))
        .collect();
    out.extend(base.free_symbols());
    out
}

/// `_contains`.
pub fn condition_set_contains(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (sym, condition, base) = (&s.args()[0], &s.args()[1], &s.args()[2]);
    if !sym.is_symbol() {
        return Ok(false_());
    }
    let decided: Basic = contains(base, other)?;
    let base_cond: Basic = if truth(&decided).is_some() {
        decided
    } else {
        contains_node(other, base)
    };
    if truth(&base_cond) == Some(false) {
        return Ok(false_());
    }
    Ok(and_(&[base_cond, subs(condition, sym, other)?]))
}

/// `as_relational`.
pub fn condition_set_as_relational(s: &Basic, x: &Basic) -> PolyResult<Basic> {
    let (sym, condition, base) = (&s.args()[0], &s.args()[1], &s.args()[2]);
    if !sym.is_symbol() {
        return Err(PolyError::Type(
            "the condition of a set of points is not a relation of an expression".to_string(),
        ));
    }
    Ok(and_(&[subs(condition, sym, x)?, contains(base, x)?]))
}

/// The set evaluated as SymPy's `ConditionSet` constructor does
/// (`condition_set`).
pub fn condition_set(sym: &Basic, condition: &Basic, base_set: &Basic) -> PolyResult<Basic> {
    let unknown: Vec<Basic> = unknowns(sym)?;
    let distinct: BTreeSet<&Basic> = unknown.iter().collect();
    if distinct.len() != unknown.len() {
        return Err(value_error("Duplicate symbols detected".to_string()));
    }
    let mut sym: Basic = sym.clone();
    let mut cond: Basic = condition.clone();
    match truth(&cond) {
        Some(true) => return Ok(base_set.clone()),
        Some(false) => return Ok(empty_set()),
        None => {}
    }
    if matches!(base_set.head(), Head::EmptySet) {
        return Ok(empty_set());
    }
    if sym.is_symbol() && truth(&contains(base_set, &sym)?) == Some(false) {
        return Err(PolyError::Type(format!(
            "sym `{}` is not in base_set `{}`",
            sym, base_set
        )));
    }
    let mut know: Option<Basic> = None;
    let mut base: Basic = base_set.clone();
    if matches!(base.head(), Head::FiniteSet) && sym.is_symbol() {
        let mut kept: Vec<Basic> = Vec::new();
        let mut undecided: Vec<Basic> = Vec::new();
        for e in base.args() {
            match truth(&subs(&cond, &sym, e)?) {
                Some(true) => kept.push(e.clone()),
                None => undecided.push(e.clone()),
                Some(false) => {}
            }
        }
        if undecided.is_empty() {
            return Ok(finite_set(&kept));
        }
        know = Some(finite_set(&kept));
        base = finite_set(&undecided);
    }
    if matches!(base.head(), Head::ConditionSet) {
        let (s, c, b) = (base.args()[0].clone(), base.args()[1].clone(), base.args()[2].clone());
        let inner: Vec<Basic> = unknowns(&s)?;
        let c_free: BTreeSet<Basic> = c.free_symbols();
        let cond_free: BTreeSet<Basic> = cond.free_symbols();
        if s == sym {
            cond = and_(&[cond, c]);
            base = b;
        } else if !unknown.iter().any(|u| c_free.contains(u)) {
            let reps: HashMap<Basic, Basic> = inner.iter().cloned().zip(unknown.iter().cloned()).collect();
            cond = and_(&[cond, xreplace_evaluated(&c, &reps)?]);
            base = b;
        } else if !inner.iter().any(|v| cond_free.contains(v)) {
            let reps: HashMap<Basic, Basic> = unknown.iter().cloned().zip(inner.iter().cloned()).collect();
            sym = xreplace_evaluated(&sym, &reps)?;
            cond = and_(&[xreplace_evaluated(&cond, &reps)?, c]);
            base = b;
        }
    }
    if matches!(cond.head(), Head::Contains) && sym == cond.args()[0] {
        let member_of: Basic = cond.args()[1].clone();
        return intersection(&[member_of, base]);
    }
    let rv: Basic = condition_set_node(&sym, &cond, &base);
    match know {
        None => Ok(rv),
        Some(k) => union(&[k, rv]),
    }
}
