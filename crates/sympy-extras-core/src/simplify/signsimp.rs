//! Port of `sympy_extras/simplify/signsimp.py`: `signsimp` and the
//! `sub_pre`/`sub_post` of SymPy's `cse_opts`. A sum which can extract a
//! minus sign is replaced by the unevaluated product `-1*(-sum)`, the
//! rebuilt products evaluated (which cancels opposite sums), the remaining
//! marks removed.

// PORTED FROM SYMPY: sympy/simplify/simplify.py (signsimp),
// sympy/simplify/cse_opts.py (sub_pre, sub_post) (SymPy 1.14), through
// sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::{HashMap, HashSet};

use crate::core::add::add;
use crate::core::basic::{Basic, Head};
use crate::core::function::{replace_where, xreplace_evaluated};
use crate::core::mul::mul;
use crate::core::printing::could_extract_minus_sign;
use crate::core::sorting::sort_key;
use crate::error::PolyResult;
use crate::numbers::*;

fn negated(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

fn is_integer_value(e: &Basic, v: i64) -> bool {
    matches!(e.head(), Head::Rational { value, integer_class: true } if *value == Rat::from(v))
}

fn sums_of(e: &Basic) -> Vec<Basic> {
    e.atoms(Some(&|n: &Basic| n.is_add()))
}

/// The sums which can extract a minus sign as `-1*(-sum)`, then the
/// remaining ones as `1*(-1)*(-sum)` (marks for [`sub_post`]).
pub fn sub_pre(e: &Basic) -> PolyResult<Basic> {
    let mut reps: HashMap<Basic, Basic> = HashMap::new();
    let mut ignore: HashSet<Basic> = HashSet::new();
    for a in sums_of(e) {
        if !could_extract_minus_sign(&a) {
            continue;
        }
        let na: Basic = negated(&a)?;
        if na.is_mul() {
            ignore.insert(a);
            continue;
        }
        reps.insert(a, Basic::new(Head::Mul, vec![Basic::negative_one(), na]));
    }
    let e: Basic = xreplace_evaluated(e, &reps)?;
    let mut negs: HashMap<Basic, Basic> = HashMap::new();
    let mut remaining: Vec<Basic> = sums_of(&e);
    remaining.sort_by_cached_key(sort_key);
    for a in remaining {
        if ignore.contains(&a) {
            continue;
        }
        if let Some(r) = reps.get(&a) {
            negs.insert(a, r.clone());
        } else if could_extract_minus_sign(&a) {
            let marked: Basic = Basic::new(Head::Mul, vec![Basic::one(), Basic::negative_one(), negated(&a)?]);
            negs.insert(a, marked);
        }
    }
    xreplace_evaluated(&e, &negs)
}

/// The marks `1*(-1)*rest` of [`sub_pre`] replaced by `-rest`.
pub fn sub_post(e: &Basic) -> PolyResult<Basic> {
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    for node in e.preorder() {
        if node.is_mul()
            && node.args().len() > 2
            && is_integer_value(&node.args()[0], 1)
            && is_integer_value(&node.args()[1], -1)
        {
            let rest: Vec<Basic> = node.args()[2..].to_vec();
            let inner: Basic = if rest.len() != 1 {
                Basic::new(Head::Mul, rest)
            } else {
                rest[0].clone()
            };
            replacements.push((node.clone(), negated(&inner)?));
        }
    }
    let mut e: Basic = e.clone();
    for (node, replacement) in replacements {
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        rule.insert(node, replacement);
        e = xreplace_evaluated(&e, &rule)?;
    }
    Ok(e)
}

fn double_negation(e: &Basic) -> bool {
    e.is_mul()
        && negated(&negated(e).unwrap_or_else(|_| e.clone()))
            .map(|d| d != *e)
            .unwrap_or(false)
}

fn undo(e: &Basic) -> PolyResult<Basic> {
    negated(&negated(e)?)
}

/// `e` with the sums which can extract a minus sign written with it
/// outside (SymPy's `signsimp`); with `evaluate = false` such a sum is the
/// unevaluated `-1*(-sum)`.
pub fn signsimp(e0: &Basic, evaluate: bool) -> PolyResult<Basic> {
    if e0.args().is_empty() {
        return Ok(e0.clone());
    }
    let e: Basic = replace_where(e0, &double_negation, &undo)?;
    let e: Basic = sub_post(&sub_pre(&e)?)?;
    if e.args().is_empty() {
        return Ok(e);
    }
    if e.is_add() {
        let mut terms: Vec<Basic> = Vec::new();
        for a in e.args() {
            terms.push(signsimp(a, evaluate)?);
        }
        let rv: Basic = add(&terms)?;
        if !evaluate && rv.is_add() && could_extract_minus_sign(&rv) {
            return Ok(Basic::new(Head::Mul, vec![Basic::negative_one(), negated(&rv)?]));
        }
        return Ok(rv);
    }
    if evaluate {
        return replace_where(&e, &double_negation, &undo);
    }
    Ok(e)
}
