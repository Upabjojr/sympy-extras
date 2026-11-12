//! B-splines, ported from `sympy_extras.functions.special.bsplines`:
//! `bspline_basis` and `bspline_basis_set`, the B-spline of degree `d` on
//! a knot sequence by the Cox--de Boor recursion, as a `Piecewise` on the
//! knot intervals (`x >= k_i` and `x <= k_{i+1}`), the pieces with the same
//! interval added and the expressions expanded.
//!
//! Not ported: `interpolating_spline` (it solves a linear system with
//! SymPy's `linsolve`).

// PORTED FROM SYMPY: sympy/functions/special/bsplines.py (bspline_basis,
// bspline_basis_set, _add_splines, _ivl) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::function::{expand_with, Hints};
use crate::core::logic::{and_, true_, truth};
use crate::core::mul::{div, mul};
use crate::core::relational::{eq, ge, le, lt};
use crate::error::{value_error, PolyError, PolyResult};
use crate::functions::elementary::piecewise::{checked_pair, piecewise, piecewise_fold, piecewise_node};

fn type_error(message: String) -> PolyError {
    PolyError::Type(message)
}

/// SymPy's `lts` and `gts`: the smaller and the larger side.
fn smaller_larger(r: &Basic) -> PolyResult<(Basic, Basic)> {
    match r.head() {
        Head::Relational(RelOp::Ge | RelOp::Gt) => Ok((r.args()[1].clone(), r.args()[0].clone())),
        Head::Relational(RelOp::Le | RelOp::Lt) => Ok((r.args()[0].clone(), r.args()[1].clone())),
        _ => Err(type_error(format!("unexpected relation: {}", r))),
    }
}

/// `_ivl`: the `(lower, upper)` of a condition `lower <= x <= upper`.
fn interval(cond: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    if matches!(cond.head(), Head::And) && cond.args().len() == 2 {
        let (mut a, mut b) = (&cond.args()[0], &cond.args()[1]);
        if matches!(a.head(), Head::Relational(_)) && matches!(b.head(), Head::Relational(_)) {
            if smaller_larger(a)?.0 == *x {
                std::mem::swap(&mut a, &mut b);
            }
            return Ok((smaller_larger(a)?.0, smaller_larger(b)?.1));
        }
    }
    Err(type_error(format!("unexpected cond type: {}", cond)))
}

fn expand(e: &Basic) -> PolyResult<Basic> {
    expand_with(e, true, Hints::DEFAULT)
}

/// SymPy's `Piecewise.expand()`: the expressions expanded, the Piecewise
/// evaluated again.
fn expanded(e: &Basic) -> PolyResult<Basic> {
    if matches!(e.head(), Head::Piecewise) {
        let mut pairs: Vec<(Basic, Basic)> = Vec::new();
        for p in e.args() {
            pairs.push((expand(&p.args()[0])?, p.args()[1].clone()));
        }
        return piecewise(&pairs);
    }
    expand(e)
}

fn is_zero(e: &Basic) -> bool {
    e.as_rational().is_some_and(|v| v.is_zero()) && e.is_number_node()
}

/// `_add_splines`.
fn add_splines(c: &Basic, b1: &Basic, d: &Basic, b2: &Basic, x: &Basic) -> PolyResult<Basic> {
    if is_zero(c) || is_zero(b1) {
        return expanded(&piecewise_fold(&mul(&[d.clone(), b2.clone()])?, Some(true))?);
    }
    if is_zero(d) || is_zero(b2) {
        return expanded(&piecewise_fold(&mul(&[c.clone(), b1.clone()])?, Some(true))?);
    }
    let p1: Basic = piecewise_fold(&mul(&[c.clone(), b1.clone()])?, Some(true))?;
    let p2: Basic = piecewise_fold(&mul(&[d.clone(), b2.clone()])?, Some(true))?;
    if !matches!(p1.head(), Head::Piecewise) || !matches!(p2.head(), Head::Piecewise) {
        return Err(type_error(
            "the B-spline pieces are expected to be Piecewise".to_string(),
        ));
    }
    let mut new_args: Vec<Basic> = Vec::new();
    let mut p2_args: Vec<Basic> = p2.args()[..p2.args().len() - 1].to_vec();
    for pair in &p1.args()[..p1.args().len() - 1] {
        let (mut expression, cond) = (pair.args()[0].clone(), pair.args()[1].clone());
        let lower: Basic = interval(&cond, x)?.0;
        for i in 0..p2_args.len() {
            let pair2: Basic = p2_args[i].clone();
            let (lower_2, upper_2) = interval(&pair2.args()[1], x)?;
            if pair2.args()[1] == cond {
                expression = add(&[expression, pair2.args()[0].clone()])?;
                p2_args.remove(i);
                break;
            }
            if truth(&lt(&lower_2, &lower)?) == Some(true) && truth(&le(&upper_2, &lower)?) == Some(true) {
                new_args.push(pair2.clone());
                p2_args.remove(i);
                break;
            }
        }
        new_args.push(checked_pair(&expression, &cond)?);
    }
    new_args.extend(p2_args);
    new_args.push(checked_pair(&Basic::zero(), &true_())?);
    // Piecewise(*new_args): the node, then expanded (evaluated)
    expanded(&piecewise_node(new_args))
}

/// The `n`-th B-spline of degree `d` on the knots, as a `Piecewise` in
/// `x`.
pub fn bspline_basis(d: i64, knots: &[Basic], n: i64, x: &Basic) -> PolyResult<Basic> {
    if n + d + 1 > knots.len() as i64 - 1 {
        return Err(value_error("n + d + 1 must not exceed len(knots) - 1".to_string()));
    }
    if d < 0 {
        return Err(value_error(format!("degree must be non-negative: {}", n)));
    }
    if n < 0 {
        return Err(not_negative_index());
    }
    let n_: usize = n as usize;
    let d_: usize = d as usize;
    if d == 0 {
        if knots[n_] == knots[n_ + 1] {
            return piecewise(&[(Basic::one(), eq(x, &knots[n_])?), (Basic::zero(), true_())]);
        }
        let cond: Basic = and_(&[ge(x, &knots[n_])?, le(x, &knots[n_ + 1])?]);
        return piecewise(&[(Basic::one(), cond), (Basic::zero(), true_())]);
    }
    let denominator: Basic = sub(&knots[n_ + d_ + 1], &knots[n_ + 1])?;
    let (b_coefficient, b2) = if !is_zero(&denominator) {
        (
            div(&sub(&knots[n_ + d_ + 1], x)?, &denominator)?,
            bspline_basis(d - 1, knots, n + 1, x)?,
        )
    } else {
        (Basic::zero(), Basic::zero())
    };
    let denominator: Basic = sub(&knots[n_ + d_], &knots[n_])?;
    let (a_coefficient, b1) = if !is_zero(&denominator) {
        (
            div(&sub(x, &knots[n_])?, &denominator)?,
            bspline_basis(d - 1, knots, n, x)?,
        )
    } else {
        (Basic::zero(), Basic::zero())
    };
    add_splines(&a_coefficient, &b1, &b_coefficient, &b2, x)
}

fn not_negative_index() -> PolyError {
    crate::error::not_ported("a negative knot index (Python wraps around)")
}

/// The `len(knots) - d - 1` B-splines of degree `d`.
pub fn bspline_basis_set(d: i64, knots: &[Basic], x: &Basic) -> PolyResult<Vec<Basic>> {
    let count: i64 = knots.len() as i64 - d - 1;
    (0..count.max(0)).map(|i| bspline_basis(d, knots, i, x)).collect()
}
