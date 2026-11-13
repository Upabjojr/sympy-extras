//! The spherical harmonics, ported from
//! `sympy_extras.functions.special.spherical_harmonics`: `Ynm(n, m, theta,
//! phi)` with the minus signs of `m`, `theta` and `phi` taken out, the
//! derivatives in `theta` and `phi`, the conjugate `(-1)**m*Ynm(n, -m,
//! theta, phi)`, the real harmonics `Znm` (into `Ynm` for a known sign of
//! `m`) and the values of `Ynm` (mpmath's `spherharm` for an integer
//! degree and order: a terminating hypergeometric sum).
//!
//! Not ported: the values of a non-integer degree or order, the expansion
//! into trigonometric polynomials, the rewrites, `as_real_imag`.

// PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, complex_real, complex_sqrt, parts, real_int, real_rat, Complex, Real};
use crate::core::expr::is_true;
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::{not_ported, PolyResult};
use crate::functions::elementary::exponential::exp;
use crate::numbers::{Int, Rat};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

fn sqrt(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::half())
}

/// The spherical harmonics.
pub fn is_spherical_harmonic(f: Function) -> bool {
    matches!(f, Function::Ynm | Function::Znm)
}

/// `Ynm(n, m, theta, phi)`, evaluated.
pub fn ynm(n: &Basic, m: &Basic, theta: &Basic, phi: &Basic) -> PolyResult<Basic> {
    spherical_harmonic(Function::Ynm, &[n.clone(), m.clone(), theta.clone(), phi.clone()])
}

/// `Znm(n, m, theta, phi)`, evaluated.
pub fn znm(n: &Basic, m: &Basic, theta: &Basic, phi: &Basic) -> PolyResult<Basic> {
    spherical_harmonic(Function::Znm, &[n.clone(), m.clone(), theta.clone(), phi.clone()])
}

/// `Ynm_c`: `conjugate(Ynm(n, m, theta, phi))`.
pub fn ynm_c(n: &Basic, m: &Basic, theta: &Basic, phi: &Basic) -> PolyResult<Basic> {
    crate::functions::elementary::complexes::conjugate(&ynm(n, m, theta, phi)?)
}

/// The evaluating function of the spherical harmonics.
pub fn spherical_harmonic(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    let (n, m, theta, phi) = (&args[0], &args[1], &args[2], &args[3]);
    let i: Basic = Basic::imaginary_unit();
    match f {
        Function::Ynm => {
            if could_extract_minus_sign(m) {
                let m: Basic = neg(m)?;
                let factor: Basic = mul(&[
                    pow(&Basic::negative_one(), &m)?,
                    exp(&mul(&[Basic::integer((-2).into()), i, m.clone(), phi.clone()])?)?,
                ])?;
                return mul(&[factor, ynm(n, &m, theta, phi)?]);
            }
            if could_extract_minus_sign(theta) {
                return ynm(n, m, &neg(theta)?, phi);
            }
            if could_extract_minus_sign(phi) {
                let phi: Basic = neg(phi)?;
                let factor: Basic = exp(&mul(&[Basic::integer((-2).into()), i, m.clone(), phi.clone()])?)?;
                return mul(&[factor, ynm(n, m, theta, &phi)?]);
            }
            Ok(node(f, args.to_vec()))
        }
        _ => {
            let root2: Basic = sqrt(&Basic::integer(2.into()))?;
            if is_true(m, Predicate::positive) {
                return div(&add(&[ynm(n, m, theta, phi)?, ynm_c(n, m, theta, phi)?])?, &root2);
            }
            if is_true(m, Predicate::zero) {
                return ynm(n, m, theta, phi);
            }
            if is_true(m, Predicate::negative) {
                return div(
                    &sub(&ynm(n, m, theta, phi)?, &ynm_c(n, m, theta, phi)?)?,
                    &mul(&[root2, i])?,
                );
            }
            Ok(node(f, args.to_vec()))
        }
    }
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn spherical_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::functions::elementary::trigonometric::cot;
    let args: &[Basic] = e.args();
    let (n, m, theta, phi) = (&args[0], &args[1], &args[2], &args[3]);
    let i: Basic = Basic::imaginary_unit();
    if f == Function::Ynm && argindex == 3 {
        let first: Basic = mul(&[m.clone(), cot(theta)?, ynm(n, m, theta, phi)?])?;
        let root: Basic = sqrt(&mul(&[sub(n, m)?, add(&[n.clone(), m.clone(), Basic::one()])?])?)?;
        let second: Basic = mul(&[
            root,
            exp(&mul(&[Basic::negative_one(), i, phi.clone()])?)?,
            ynm(n, &add(&[m.clone(), Basic::one()])?, theta, phi)?,
        ])?;
        return add(&[first, second]);
    }
    if f == Function::Ynm && argindex == 4 {
        return mul(&[i, m.clone(), ynm(n, m, theta, phi)?]);
    }
    let v: &Basic = &args[argindex - 1];
    let elsewhere: bool = args
        .iter()
        .enumerate()
        .any(|(k, w)| k != argindex - 1 && w.free_symbols().contains(v));
    if v.is_symbol() && !elsewhere {
        return Ok(crate::core::function::derivative_node(e, &[(v.clone(), 1)]));
    }
    Err(not_ported("a derivative as a Subs"))
}

/// The conjugate (`_eval_conjugate`) of `Ynm` (`Znm` has the default
/// rule).
pub fn spherical_conjugate(e: &Basic) -> PolyResult<Option<Basic>> {
    let args: &[Basic] = e.args();
    let (n, m, theta, phi) = (&args[0], &args[1], &args[2], &args[3]);
    Ok(Some(mul(&[
        pow(&Basic::negative_one(), m)?,
        ynm(n, &neg(m)?, theta, phi)?,
    ])?))
}

/// The facts (`_eval_facts`).
pub fn spherical_facts() -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    facts
}

fn integer_of(z: &Complex) -> Option<i64> {
    let (re, im) = parts(z);
    if im != Real::ZERO || re.clone() != re.floor() {
        return None;
    }
    re.to_int().value().try_into().ok()
}

fn c_int(n: i64, wp: usize) -> Complex {
    complex_real(real_int(&Int::from(n), wp), wp)
}

/// `sin` of a complex number by `exp` (real for a real argument).
fn sin(z: &Complex, wp: usize) -> Complex {
    let i: Complex = complex(Real::ZERO.with_precision(wp).value(), real_int(&Int::ONE, wp));
    let iz: Complex = &i * z;
    let s: Complex = &(&iz.exp() - &(-&iz).exp()) / &(&c_int(2, wp) * &i);
    if parts(z).1 == Real::ZERO {
        return complex(parts(&s).0, Real::ZERO.with_precision(wp).value());
    }
    s
}

fn zero(prec: usize) -> Complex {
    complex(
        Real::ZERO.with_precision(prec).value(),
        Real::ZERO.with_precision(prec).value(),
    )
}

/// The value of `Ynm` (mpmath's `spherharm`) for an integer degree and
/// order; `None` otherwise.
pub fn spherical_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    if f != Function::Ynm {
        return None;
    }
    let mut l: i64 = integer_of(&values[0])?;
    let m: i64 = integer_of(&values[1])?;
    let (theta, phi) = (&values[2], &values[3]);
    if l < 0 {
        l = -(l + 1);
    }
    let (tr, ti) = parts(theta);
    if tr == Real::ZERO && ti == Real::ZERO && m < 0 {
        return Some(zero(prec));
    }
    let am: i64 = m.abs();
    if am > l {
        return Some(zero(prec));
    }
    if l > 2000 {
        return None;
    }
    let wp: usize = prec + 40 + 4 * l as usize;
    let raise = |z: &Complex| -> Complex {
        let (a, b) = parts(z);
        complex(a.with_precision(wp).value(), b.with_precision(wp).value())
    };
    let (theta, phi) = (raise(theta), raise(phi));
    // (2l+1)*(l+|m|)!/(pi*(l-|m|)!), its square root
    let mut ratio: Int = Int::from(2 * l + 1);
    for k in (l - am + 1)..=(l + am) {
        ratio *= Int::from(k);
    }
    let pi: Real = Real::pi(wp);
    let root: Complex = complex_sqrt(&complex_real(real_int(&ratio, wp) / pi, wp), wp);
    let s: Complex = sin(&theta, wp);
    let s2: Complex = &s * &s;
    let mut power: Complex = c_int(1, wp);
    for _ in 0..(am / 2) {
        power = &power * &s2;
    }
    if am % 2 == 1 {
        power = &power * &complex_sqrt(&s2, wp);
    }
    // 1/(|m|! * 2**(|m| + 1))
    let mut denominator: Int = Int::from(2) << (am as usize);
    for k in 2..=am {
        denominator *= Int::from(k);
    }
    // 2F1(|m| - l, l + |m| + 1; |m| + 1; sin(theta/2)**2), a polynomial
    let half: Complex = &theta / &c_int(2, wp);
    let sh: Complex = sin(&half, wp);
    let z: Complex = &sh * &sh;
    let (a, b, c) = (am - l, l + am + 1, am + 1);
    let mut coefficient: Rat = Rat::from(Int::ONE);
    let mut zk: Complex = c_int(1, wp);
    let mut total: Complex = c_int(0, wp);
    for k in 0..=(l - am) {
        let term: Complex = &complex_real(real_rat(&coefficient, wp), wp) * &zk;
        total = &total + &term;
        coefficient *= crate::numbers::rat((a + k) * (b + k), (c + k) * (k + 1));
        zk = &zk * &z;
    }
    let i: Complex = complex(Real::ZERO.with_precision(wp).value(), real_int(&Int::ONE, wp));
    let e: Complex = (&(&i * &c_int(m, wp)) * &phi).exp();
    let mut value: Complex = &(&(&(&e * &root) * &power) * &total) / &c_int(1, wp);
    value = &value / &complex_real(real_int(&denominator, wp), wp);
    if m > 0 && m % 2 == 1 {
        value = -&value;
    }
    let (re, im) = parts(&value);
    Some(complex(
        re.with_precision(prec).value(),
        im.with_precision(prec).value(),
    ))
}
