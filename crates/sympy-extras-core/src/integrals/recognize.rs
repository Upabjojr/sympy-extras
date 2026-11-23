//! Port of `sympy_extras/integrals/recognize.py`: symbolic-numeric
//! recognition of constants, closed forms guessed from high precision
//! values with PSLQ ([`crate::identification::pslq_tol`]), and of definite
//! integrals computed by two quadrature rules (mpmath's tanh-sinh and
//! Gauss-Legendre, [`crate::integrals::quadrature`]). A recognised closed
//! form is a conjecture, checked at a higher precision.
//!
//! The Rust core has no `Float`: `recognize_integral`'s
//! `Float(mpmath.nstr(value, digits), digits)` is the value rounded to
//! `digits` significant decimal digits, handed to [`recognize_value`]
//! with the digits its string shows (all of them).

#![allow(clippy::mutable_key_type)]

use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{parts, real_rat, Real};
use crate::error::PolyResult;
use crate::integrals::quadrature::{Point, Rule};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn p2(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn apply(f: Function, args: &[Basic]) -> R {
    crate::functions::apply_function(f, args)
}
fn prec_of(digits: usize) -> usize {
    crate::numeric::dps_to_prec(digits)
}
fn real(r: &Real, prec: usize) -> Real {
    real_rat(&crate::integrals::evalf::exact(r), prec)
}
fn ten_pow(k: i64, prec: usize) -> Real {
    let mut v: Rat = Rat::from(Int::ONE);
    for _ in 0..k.abs() {
        v *= Rat::from(Int::from(10));
    }
    if k < 0 {
        v = Rat::from(Int::ONE) / v;
    }
    real_rat(&v, prec)
}
fn abs_real(r: &Real) -> Real {
    if *r < Real::ZERO {
        -r.clone()
    } else {
        r.clone()
    }
}

/// The constants a closed form is searched in (`DEFAULT_CONSTANTS`).
pub fn default_constants() -> PolyResult<Vec<Basic>> {
    let pi: Basic = Basic::pi();
    let log2: Basic = apply(Function::Log, &[int(2)])?;
    let sqrt = |n: i64| p2(&int(n), &Basic::half());
    let quarter: Basic = Basic::from_rational(crate::numbers::rat(1, 4));
    let third: Basic = Basic::from_rational(crate::numbers::rat(1, 3));
    Ok(vec![
        pi.clone(),
        p2(&pi, &int(2))?,
        p2(&pi, &int(3))?,
        log2.clone(),
        apply(Function::Log, &[int(3)])?,
        m2(&pi, &log2)?,
        p2(&log2, &int(2))?,
        sqrt(2)?,
        sqrt(3)?,
        sqrt(5)?,
        m2(&sqrt(2)?, &pi)?,
        m2(&sqrt(3)?, &pi)?,
        Basic::atom(Head::EulerGamma),
        Basic::atom(Head::Catalan),
        apply(Function::Zeta, &[int(3)])?,
        Basic::e(),
        crate::core::mul::div(
            &p2(&apply(Function::Gamma, &[quarter])?, &int(2))?,
            &p2(&pi, &Basic::half())?,
        )?,
        p2(&apply(Function::Gamma, &[third])?, &int(3))?,
    ])
}

/// The largest coefficient accepted in an integer relation.
const MAX_COEFFICIENT: i64 = 1_000_000;

/// The guard digits of the numerical values of exact expressions.
const GUARD: usize = 20;

/// `value` as a real number to `digits` digits (computed with guard
/// digits); `None` when it is not a real number (`_to_mpf`).
fn to_mpf(value: &Basic, digits: usize) -> Option<Real> {
    let number = crate::core::evalf::numeric_value(value, prec_of(digits + GUARD))?;
    let (a, b) = parts(&number);
    if b != Real::ZERO {
        return None;
    }
    Some(real(&a, prec_of(digits)))
}

/// A closed form of the exact `value` as a rational combination of 1 and
/// at most two of the `constants`, found by PSLQ at `digits` digits and
/// checked at `digits + 15`; `None` when no relation with small integer
/// coefficients exists, or when numerical checks are off
/// (`recognize_constant`).
pub fn recognize_constant(value: &Basic, constants: &[Basic], digits: usize) -> OptR {
    if !crate::settings::settings().numerical_checks {
        return Ok(None);
    }
    if !value.free_symbols().is_empty() {
        return Ok(None);
    }
    if value.as_rational().is_some() {
        return Ok(Some(value.clone()));
    }
    let Some(target) = to_mpf(value, digits) else {
        return Ok(None);
    };
    let Some(candidate) = relation(&target, constants, digits)? else {
        return Ok(None);
    };
    // the check at a higher precision
    let (Some(precise), Some(guess)) = (to_mpf(value, digits + 15), to_mpf(&candidate, digits + 15)) else {
        return Ok(None);
    };
    let p: usize = prec_of(digits + 15);
    let bound: Real = ten_pow(-(digits as i64 + 10), p) * (real_rat(&Rat::from(Int::ONE), p) + abs_real(&precise));
    if abs_real(&real(&(&precise - &guess), p)) > bound {
        return Ok(None);
    }
    Ok(Some(candidate))
}

/// The closed form of an inexact value which shows `shown` significant
/// digits (`recognize_constant` of a `Float`): the relation at the digits
/// it carries is all the evidence.
pub fn recognize_value(target: &Real, shown: usize, constants: &[Basic], digits: usize) -> OptR {
    if !crate::settings::settings().numerical_checks {
        return Ok(None);
    }
    let digits: usize = digits.min(shown);
    if digits < 12 {
        return Ok(None);
    }
    let target: Real = real(target, prec_of(digits));
    relation(&target, constants, digits)
}

/// `target` as a rational combination of 1 and at most two of the
/// constants, by PSLQ on the short vectors, the single constants first
/// (`_relation`).
fn relation(target: &Real, constants: &[Basic], digits: usize) -> OptR {
    let mut values: Vec<Real> = Vec::new();
    for c in constants {
        let Some(v) = to_mpf(c, digits) else {
            return Ok(None);
        };
        values.push(v);
    }
    let prec: usize = prec_of(digits);
    let tol: Real = ten_pow(-(digits as i64 - 4), prec);
    let mut subsets: Vec<Vec<usize>> = vec![Vec::new()];
    subsets.extend((0..constants.len()).map(|i| vec![i]));
    for i in 0..constants.len() {
        for j in (i + 1)..constants.len() {
            subsets.push(vec![i, j]);
        }
    }
    let one: Real = real_rat(&Rat::from(Int::ONE), prec);
    for subset in subsets {
        let mut vector: Vec<Real> = vec![target.clone(), one.clone()];
        vector.extend(subset.iter().map(|&i| values[i].clone()));
        let Some(found) = crate::identification::pslq_tol(&vector, prec, MAX_COEFFICIENT, 10_000, Some(&tol)) else {
            continue;
        };
        if found[0] == 0 {
            continue;
        }
        let mut candidate: Basic = crate::core::mul::neg(&int(found[1]))?;
        for (m, &i) in found[2..].iter().zip(&subset) {
            candidate = crate::core::add::sub(&candidate, &m2(&int(*m), &constants[i])?)?;
        }
        return Ok(Some(crate::core::mul::div(&candidate, &int(found[0]))?));
    }
    Ok(None)
}

fn endpoint(e: &Basic, digits: usize) -> Option<Point> {
    match e.head() {
        Head::Infinity => Some(Point::PosInf),
        Head::NegativeInfinity => Some(Point::NegInf),
        _ => Some(Point::Finite(to_mpf(e, digits)?)),
    }
}

/// `Integral(f, (x, a, b))` by the tanh-sinh and Gauss-Legendre rules at
/// `digits` digits, returned only when the two agree to `digits - 5`
/// digits and the estimated error is as small (`quadrature`).
pub fn quadrature(f: &Basic, x: &Basic, a: &Basic, b: &Basic, digits: usize) -> PolyResult<Option<Real>> {
    let mut far_failure: bool = false;
    let found: Option<Real> = quadrature_values(f, x, a, b, digits, &mut far_failure);
    if found.is_none() && far_failure {
        return Err(crate::error::not_ported(
            "a value at a far quadrature node beyond the exponents of dashu's floats (mpmath's are unbounded)",
        ));
    }
    Ok(found)
}

fn quadrature_values(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    digits: usize,
    far_failure: &mut bool,
) -> Option<Real> {
    let prec: usize = prec_of(digits);
    let (lo, hi) = (endpoint(a, digits)?, endpoint(b, digits)?);
    let one: Real = real_rat(&Rat::from(Int::ONE), prec);
    let ten: Real = real_rat(&Rat::from(Int::from(10)), prec);
    let points: Vec<Point> = match (&lo, &hi) {
        (Point::NegInf, Point::PosInf) => vec![
            lo.clone(),
            Point::Finite(-one.clone()),
            Point::Finite(Real::ZERO),
            Point::Finite(one.clone()),
            hi.clone(),
        ],
        (Point::Finite(l), Point::PosInf) => vec![
            lo.clone(),
            Point::Finite(real(&(l + &one), prec)),
            Point::Finite(real(&(l + &ten), prec)),
            hi.clone(),
        ],
        (Point::NegInf, Point::Finite(h)) => vec![
            lo.clone(),
            Point::Finite(real(&(h - &ten), prec)),
            Point::Finite(real(&(h - &one), prec)),
            hi.clone(),
        ],
        _ => vec![lo.clone(), hi.clone()],
    };
    let far: Real = real_rat(&Rat::from(Int::ONE << 40usize), prec);
    let mut g = |t: &Real| -> Option<crate::integrals::quadrature::Value> {
        let z = crate::core::evalf::complex(t.clone(), Real::ZERO);
        let found = crate::integrals::evalf::value_at(f, x, z, prec);
        if found.is_none() && abs_real(t) > far {
            // an exponential beyond the range of dashu's exponents
            *far_failure = true;
        }
        Some(parts(&found?))
    };
    let (first, error) = crate::integrals::quadrature::quad_rule(&mut g, &points, prec, Rule::TanhSinh)?;
    let (second, _) = crate::integrals::quadrature::quad_rule(&mut g, &points, prec, Rule::GaussLegendre)?;
    if abs_real(&first.1) > ten_pow(-(digits as i64 - 5), prec) {
        return None;
    }
    let (first, second) = (first.0, second.0);
    let tolerance: Real = ten_pow(-(digits as i64 - 5), prec) * (&one + abs_real(&first));
    if abs_real(&real(&(&first - &second), prec)) > tolerance || abs_real(&error) > tolerance {
        return None;
    }
    Some(real(&first, prec))
}

/// `Float(mpmath.nstr(value, digits), digits)`: the value rounded to
/// `digits` significant digits, as the exact rational it shows, and the
/// number of digits its string shows (all of them).
fn shown_digits(value: &Real, digits: usize) -> (Rat, usize) {
    let r: Rat = crate::integrals::evalf::exact(value);
    if r == Rat::from(Int::ZERO) {
        return (r, 1);
    }
    let magnitude: f64 = value.to_f64().value().abs();
    let mut k: i64 = magnitude.log10().floor() as i64;
    let ten = |e: i64| -> Rat {
        let mut v: Rat = Rat::from(Int::ONE);
        for _ in 0..e.abs() {
            v *= Rat::from(Int::from(10));
        }
        if e < 0 {
            Rat::from(Int::ONE) / v
        } else {
            v
        }
    };
    let absolute: Rat = if r < Rat::from(Int::ZERO) {
        -r.clone()
    } else {
        r.clone()
    };
    // the exponent of the leading digit, exactly
    while absolute >= ten(k + 1) {
        k += 1;
    }
    while absolute < ten(k) {
        k -= 1;
    }
    let scale: Rat = ten(k - digits as i64 + 1);
    let scaled: Rat = &absolute / &scale;
    // round half to even
    let floor: Int = crate::numbers::numer(&scaled) / crate::numbers::denom(&scaled);
    let fraction: Rat = &scaled - &Rat::from(floor.clone());
    let half: Rat = crate::numbers::rat(1, 2);
    let mut mantissa: Int = floor.clone();
    if fraction > half || (fraction == half && (&floor % Int::from(2)) != Int::ZERO) {
        mantissa += Int::ONE;
    }
    let mut text: String = mantissa.to_string();
    let mut exponent: i64 = k - digits as i64 + 1;
    while text.len() > 1 && text.ends_with('0') {
        text.pop();
        exponent += 1;
    }
    // the Float of the string prints its precision: every one of the
    // digits is shown, the trailing zeros as well
    let shown: usize = digits;
    let m: Int = text.parse().expect("digits");
    let mut v: Rat = Rat::from(m) * ten(exponent);
    if r < Rat::from(Int::ZERO) {
        v = -v;
    }
    (v, shown)
}

/// A conjectured closed form of `Integral(f, (x, a, b))`: the integral
/// computed numerically to `digits` digits by two rules, recognised, then
/// checked at `digits + 15` digits by a third quadrature
/// (`recognize_integral`).
pub fn recognize_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, constants: &[Basic], digits: usize) -> OptR {
    if !crate::settings::settings().numerical_checks {
        return Ok(None);
    }
    let free = f.free_symbols();
    if free
        .iter()
        .chain(a.free_symbols().iter())
        .chain(b.free_symbols().iter())
        .any(|s| s != x)
    {
        return Ok(None);
    }
    let Some(value) = quadrature(f, x, a, b, digits)? else {
        return Ok(None);
    };
    let (shown_value, shown) = shown_digits(&value, digits);
    let Some(candidate) = recognize_value(&real_rat(&shown_value, prec_of(digits)), shown, constants, digits)? else {
        return Ok(None);
    };
    let (Some(precise), Some(guess)) = (quadrature(f, x, a, b, digits + 15)?, to_mpf(&candidate, digits + 15)) else {
        return Ok(None);
    };
    let p: usize = prec_of(digits + 15);
    let bound: Real = ten_pow(-(digits as i64 + 8), p) * (real_rat(&Rat::from(Int::ONE), p) + abs_real(&precise));
    if abs_real(&real(&(&precise - &guess), p)) > bound {
        return Ok(None);
    }
    Ok(Some(candidate))
}
