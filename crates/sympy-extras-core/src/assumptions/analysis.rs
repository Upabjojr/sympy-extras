//! Port of `sympy_extras/assumptions/analysis.py` (the first part): the
//! certified sign of a constant ([`certified_sign`]: the exact knowledge
//! of the core, a hidden zero of an algebraic number by its minimal
//! polynomial, then rigorous interval arithmetic and a value with a margin
//! over its precision when the numerical checks are enabled) and the
//! comparison of two constants.

use super::intervals::{evaluate_at, interval_sign};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::numbers::*;
use dashu_int::UBig;

fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

fn has_symbols(e: &Basic) -> bool {
    e.free_symbols()
        .iter()
        .any(|s| matches!(s.head(), Head::Symbol(_) | Head::Dummy(..)))
}

/// The sign of a constant by interval arithmetic at `digits` digits:
/// whether the arithmetic covers it, and the sign when the enclosure
/// excludes zero (`_interval_sign_at`).
fn interval_sign_at(value: &Basic, digits: usize) -> (bool, Option<i32>) {
    let prec: usize = crate::numeric::dps_to_prec(digits);
    match evaluate_at(value, &[], prec) {
        None => (false, None),
        Some(enclosure) => {
            let s: Option<i32> = interval_sign(&enclosure);
            (true, if s == Some(0) { None } else { s })
        }
    }
}

/// Whether `e` is built from rationals and `I` by sums, products and
/// rational powers.
fn is_algebraic_constant(e: &Basic) -> bool {
    if e.is_rational_class() || matches!(e.head(), Head::ImaginaryUnit) {
        return true;
    }
    if e.is_pow() {
        return e.args()[1].is_rational_class() && is_algebraic_constant(&e.args()[0]);
    }
    if e.is_add() || e.is_mul() {
        return e.args().iter().all(is_algebraic_constant);
    }
    false
}

/// Whether the algebraic number `e` is zero: its minimal polynomial is
/// `x`.
fn vanishes(e: &Basic) -> PolyResult<bool> {
    let coefficients: Option<Vec<Int>> =
        crate::timeout::attempt(|| crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(e))?;
    Ok(coefficients == Some(vec![Int::ONE, Int::ZERO]))
}

/// The sign of a constant expression (`certified_sign`); `None` when
/// undecided.
pub fn certified_sign(value: &Basic) -> PolyResult<Option<i32>> {
    if has_symbols(value) {
        return Ok(None);
    }
    match value.head() {
        Head::Infinity => return Ok(Some(1)),
        Head::NegativeInfinity => return Ok(Some(-1)),
        _ => {}
    }
    if yes(value, Predicate::zero) {
        return Ok(Some(0));
    }
    if yes(value, Predicate::positive) {
        return Ok(Some(1));
    }
    if yes(value, Predicate::negative) {
        return Ok(Some(-1));
    }
    if is_algebraic_constant(value) && vanishes(value)? {
        return Ok(Some(0));
    }
    let settings = crate::settings::settings();
    if !settings.numerical_checks || is(value, Predicate::real) == Some(false) {
        return Ok(None);
    }
    let p: usize = settings.precision;
    for digits in [p, 2 * p, 4 * p] {
        let (covered, s) = interval_sign_at(value, digits);
        if covered {
            if s.is_some() {
                return Ok(s);
            }
            continue;
        }
        // a value computed with a margin over its precision
        let Some(approximation) = crate::core::evalf::numeric_value(value, crate::numeric::dps_to_prec(digits)) else {
            return Ok(None);
        };
        let (re, im) = crate::core::evalf::parts(&approximation);
        let re_r: Rat = super::ivarith::real_to_rat(&re);
        if !super::ivarith::real_to_rat(&im).is_zero() {
            return Ok(None);
        }
        let threshold: Rat = Rat::from_parts(Int::ONE, UBig::from(10u8).pow(digits / 2));
        let magnitude: Rat = if rat_is_negative(&re_r) {
            -re_r.clone()
        } else {
            re_r.clone()
        };
        if magnitude > threshold {
            return Ok(Some(if rat_is_negative(&re_r) { -1 } else { 1 }));
        }
    }
    Ok(None)
}

/// The sign of `a - b` for constants, infinities included
/// (`certified_compare`).
pub fn certified_compare(a: &Basic, b: &Basic) -> PolyResult<Option<i32>> {
    if matches!(a.head(), Head::Infinity) || matches!(b.head(), Head::NegativeInfinity) {
        return Ok(Some(if a == b { 0 } else { 1 }));
    }
    if matches!(a.head(), Head::NegativeInfinity) || matches!(b.head(), Head::Infinity) {
        return Ok(Some(if a == b { 0 } else { -1 }));
    }
    certified_sign(&crate::core::add::sub(a, b)?)
}
