//! The Bessel functions, part 1, ported from
//! `sympy_extras.functions.special.bessel`: `besselj`, `bessely`,
//! `besseli`, `besselk`, the Hankel functions `hankel1`, `hankel2` (no
//! values of their own, as in SymPy) and the helpers `_besseli`,
//! `_besselk` of the Gruntz algorithm (nodes without values). The values
//! at `z = 0` (by the order), at the infinities (and the imaginary ones
//! for `bessely`, `besseli`, `besselk`), a minus sign taken out of the
//! argument or of an integer order, `I` taken out of the argument for an
//! integer order; the derivative in the argument by the recurrence, the
//! conjugate off the negative axis, the extended-real facts.
//!
//! Values: `besselj`, `besseli` by `(z/2)**nu/gamma(nu + 1)*0F1(; nu + 1;
//! -+z**2/4)` (the series of `hyper`, so for a moderate argument);
//! `bessely`, `besselk`, `hankel1`, `hankel2` by the reflection formulas
//! for a non-integer order only (mpmath's limits at an integer order are
//! not ported). The core has no polar numbers: the branch factors are
//! always trivial. Not ported (part 2): the spherical Bessel functions,
//! the Airy functions, `marcumq`, `jn_zeros`; the asymptotic series.

// PORTED FROM SYMPY: sympy/functions/special/bessel.py (BesselBase, besselj, bessely,
// besseli, besselk, hankel1, hankel2, _besseli, _besselk) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, numeric_value, parts, real_int, Complex, Real};
use crate::core::expr::{is, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::PolyResult;
use crate::numbers::Int;

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

/// The Bessel heads of part 1.
pub fn is_bessel(f: Function) -> bool {
    matches!(
        f,
        Function::BesselJ
            | Function::BesselY
            | Function::BesselI
            | Function::BesselK
            | Function::Hankel1
            | Function::Hankel2
            | Function::BesselIs
            | Function::BesselKs
    )
}

pub fn besselj(nu: &Basic, z: &Basic) -> PolyResult<Basic> {
    bessel(Function::BesselJ, &[nu.clone(), z.clone()])
}

pub fn bessely(nu: &Basic, z: &Basic) -> PolyResult<Basic> {
    bessel(Function::BesselY, &[nu.clone(), z.clone()])
}

pub fn besseli(nu: &Basic, z: &Basic) -> PolyResult<Basic> {
    bessel(Function::BesselI, &[nu.clone(), z.clone()])
}

pub fn besselk(nu: &Basic, z: &Basic) -> PolyResult<Basic> {
    bessel(Function::BesselK, &[nu.clone(), z.clone()])
}

/// The evaluating function of the Bessel heads.
pub fn bessel(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    let value: Option<Basic> = match f {
        Function::BesselJ => besselj_eval(&args[0], &args[1])?,
        Function::BesselY => bessely_eval(&args[0], &args[1])?,
        Function::BesselI => besseli_eval(&args[0], &args[1])?,
        Function::BesselK => besselk_eval(&args[0], &args[1])?,
        _ => None, // hankel1, hankel2, _besseli, _besselk: no evaluation
    };
    Ok(value.unwrap_or_else(|| node(f, args.to_vec())))
}

/// SymPy's values of `besselj` and `besseli` at `z = 0`.
fn at_zero_regular(nu: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::re;
    if is_true(nu, Predicate::zero) {
        return Ok(Some(Basic::one()));
    }
    let re_nu: Basic = re(nu)?;
    if (is_true(nu, Predicate::integer) && is(nu, Predicate::zero) == Some(false))
        || is_true(&re_nu, Predicate::positive)
    {
        return Ok(Some(Basic::zero()));
    }
    if is_true(&re_nu, Predicate::negative) && !is_true(nu, Predicate::integer) {
        return Ok(Some(Basic::zoo()));
    }
    if is_true(nu, Predicate::imaginary) {
        return Ok(Some(Basic::nan()));
    }
    Ok(None)
}

/// SymPy's values of `bessely` and `besselk` at `z = 0`.
fn at_zero_singular(nu: &Basic, at_zero_order: Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::re;
    if is_true(nu, Predicate::zero) {
        return Ok(Some(at_zero_order));
    }
    let re_nu: Basic = re(nu)?;
    match is(&re_nu, Predicate::zero) {
        Some(false) => Ok(Some(Basic::zoo())),
        Some(true) => Ok(Some(Basic::nan())),
        None => Ok(None),
    }
}

/// `z/I` when `I` can be taken out of `z` and the rest is not zero.
fn rotated(z: &Basic) -> PolyResult<Option<Basic>> {
    let r: Option<Basic> = crate::core::expr::extract_multiplicatively(z, &Basic::imaginary_unit())?;
    Ok(r.filter(|r| !is_true(r, Predicate::zero)))
}

/// The branch handling of `besselj` and `besseli`: an integer order loses
/// the polar factors of the argument (the core has none: `unpolarify` is
/// a reevaluation), the order is unpolarified.
fn branch(f: Function, nu: &Basic, z: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::function::unpolarify;
    if is_true(nu, Predicate::integer) {
        let newz: Basic = unpolarify(z)?;
        if newz != *z {
            return Ok(Some(bessel(f, &[nu.clone(), newz])?));
        }
    }
    let nnu: Basic = unpolarify(nu)?;
    if *nu != nnu {
        return Ok(Some(bessel(f, &[nnu, z.clone()])?));
    }
    Ok(None)
}

fn is_infinity(z: &Basic) -> bool {
    matches!(z.head(), Head::Infinity | Head::NegativeInfinity)
}

/// `z**nu*(-z)**(-nu)*f(nu, -z)`.
fn minus_argument(f: Function, nu: &Basic, z: &Basic) -> PolyResult<Basic> {
    let minus_z: Basic = neg(z)?;
    mul(&[
        pow(z, nu)?,
        pow(&minus_z, &neg(nu)?)?,
        bessel(f, &[nu.clone(), minus_z])?,
    ])
}

fn besselj_eval(nu: &Basic, z: &Basic) -> PolyResult<Option<Basic>> {
    if is_true(z, Predicate::zero) {
        if let Some(v) = at_zero_regular(nu)? {
            return Ok(Some(v));
        }
    }
    if is_infinity(z) {
        return Ok(Some(Basic::zero()));
    }
    if could_extract_minus_sign(z) {
        return Ok(Some(minus_argument(Function::BesselJ, nu, z)?));
    }
    if is_true(nu, Predicate::integer) {
        if could_extract_minus_sign(nu) {
            let minus_nu: Basic = neg(nu)?;
            return Ok(Some(mul(&[
                pow(&Basic::negative_one(), &minus_nu)?,
                besselj(&minus_nu, z)?,
            ])?));
        }
        if let Some(r) = rotated(z)? {
            return Ok(Some(mul(&[pow(&Basic::imaginary_unit(), nu)?, besseli(nu, &r)?])?));
        }
    }
    branch(Function::BesselJ, nu, z)
}

fn bessely_eval(nu: &Basic, z: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::add::add;
    use crate::functions::elementary::exponential::exp;
    let i: Basic = Basic::imaginary_unit();
    if is_true(z, Predicate::zero) {
        if let Some(v) = at_zero_singular(nu, Basic::negative_oo())? {
            return Ok(Some(v));
        }
    }
    if is_infinity(z) {
        return Ok(Some(Basic::zero()));
    }
    let shifted: Basic = add(&[nu.clone(), Basic::one()])?;
    if *z == mul(&[i.clone(), Basic::oo()])? {
        let phase: Basic = exp(&mul(&[i, Basic::pi(), shifted, Basic::half()])?)?;
        return Ok(Some(mul(&[phase, Basic::oo()])?));
    }
    if *z == mul(&[i.clone(), Basic::negative_oo()])? {
        let phase: Basic = exp(&mul(&[Basic::negative_one(), i, Basic::pi(), shifted, Basic::half()])?)?;
        return Ok(Some(mul(&[phase, Basic::oo()])?));
    }
    if is_true(nu, Predicate::integer) && could_extract_minus_sign(nu) {
        let minus_nu: Basic = neg(nu)?;
        return Ok(Some(mul(&[
            pow(&Basic::negative_one(), &minus_nu)?,
            bessely(&minus_nu, z)?,
        ])?));
    }
    Ok(None)
}

fn besseli_eval(nu: &Basic, z: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::im;
    if is_true(z, Predicate::zero) {
        if let Some(v) = at_zero_regular(nu)? {
            return Ok(Some(v));
        }
    }
    if is_infinity(&im(z)?) {
        return Ok(Some(Basic::zero()));
    }
    if matches!(z.head(), Head::Infinity) {
        return Ok(Some(Basic::oo()));
    }
    if matches!(z.head(), Head::NegativeInfinity) {
        return Ok(Some(mul(&[pow(&Basic::negative_one(), nu)?, Basic::oo()])?));
    }
    if could_extract_minus_sign(z) {
        return Ok(Some(minus_argument(Function::BesselI, nu, z)?));
    }
    if is_true(nu, Predicate::integer) {
        if could_extract_minus_sign(nu) {
            return Ok(Some(besseli(&neg(nu)?, z)?));
        }
        if let Some(r) = rotated(z)? {
            return Ok(Some(mul(&[
                pow(&Basic::imaginary_unit(), &neg(nu)?)?,
                besselj(nu, &neg(&r)?)?,
            ])?));
        }
    }
    branch(Function::BesselI, nu, z)
}

fn besselk_eval(nu: &Basic, z: &Basic) -> PolyResult<Option<Basic>> {
    let i: Basic = Basic::imaginary_unit();
    if is_true(z, Predicate::zero) {
        if let Some(v) = at_zero_singular(nu, Basic::oo())? {
            return Ok(Some(v));
        }
    }
    if matches!(z.head(), Head::Infinity)
        || *z == mul(&[i.clone(), Basic::oo()])?
        || *z == mul(&[i, Basic::negative_oo()])?
    {
        return Ok(Some(Basic::zero()));
    }
    if is_true(nu, Predicate::integer) && could_extract_minus_sign(nu) {
        return Ok(Some(besselk(&neg(nu)?, z)?));
    }
    Ok(None)
}

/// The derivative in the argument `argindex` (from 1): in `z`, `b/2*f(nu
/// - 1, z) - a/2*f(nu + 1, z)`; in the order, `DefinedFunction.fdiff`
/// (the helpers `_besseli`, `_besselk` have only that).
pub fn bessel_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::core::add::{add, sub};
    if argindex != 2 || matches!(f, Function::BesselIs | Function::BesselKs) {
        return crate::functions::special::hyper::default_fdiff(e, argindex);
    }
    let (nu, z) = (&e.args()[0], &e.args()[1]);
    let (a, b): (i64, i64) = match f {
        Function::BesselI => (-1, 1),
        Function::BesselK => (1, -1),
        _ => (1, 1),
    };
    let lower: Basic = bessel(f, &[sub(nu, &Basic::one())?, z.clone()])?;
    let upper: Basic = bessel(f, &[add(&[nu.clone(), Basic::one()])?, z.clone()])?;
    sub(
        &mul(&[Basic::rational(Int::from(b), Int::from(2))?, lower])?,
        &mul(&[Basic::rational(Int::from(a), Int::from(2))?, upper])?,
    )
}

/// The conjugate (`_eval_conjugate`): `f(conjugate(nu), conjugate(z))` off
/// the negative axis (the Hankel functions exchanged); the helpers have
/// the default rule (real: itself, imaginary: minus itself).
pub fn bessel_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::conjugate;
    if matches!(f, Function::BesselIs | Function::BesselKs) {
        if is_true(e, Predicate::extended_real) {
            return Ok(Some(e.clone()));
        }
        if is_true(e, Predicate::imaginary) {
            return Ok(Some(neg(e)?));
        }
        return Ok(None);
    }
    let (nu, z) = (&e.args()[0], &e.args()[1]);
    if is(z, Predicate::extended_negative) != Some(false) {
        return Ok(None);
    }
    let target: Function = match f {
        Function::Hankel1 => Function::Hankel2,
        Function::Hankel2 => Function::Hankel1,
        g => g,
    };
    Ok(Some(bessel(target, &[conjugate(nu)?, conjugate(z)?])?))
}

/// The facts (`_eval_facts`): extended real for an integer order and a
/// real argument (`besselj`, `besseli`) or a positive one (`bessely`,
/// `besselk`).
pub fn bessel_facts(f: Function, args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    let on_real_axis: Option<Predicate> = match f {
        Function::BesselJ | Function::BesselI => Some(Predicate::extended_real),
        Function::BesselY | Function::BesselK => Some(Predicate::positive),
        _ => None,
    };
    if let Some(p) = on_real_axis {
        let nu_facts = crate::core::assumptions::facts_of(&args[0]);
        let z_facts = crate::core::assumptions::facts_of(&args[1]);
        if nu_facts.get(&Predicate::integer) == Some(&true) && z_facts.get(&p) == Some(&true) {
            facts.set(Predicate::extended_real, true);
        }
    }
    facts
}

// ---------------------------------------------------------------------
// values
// ---------------------------------------------------------------------

fn c_int(n: i64, wp: usize) -> Complex {
    crate::core::evalf::complex_real(real_int(&Int::from(n), wp), wp)
}

fn is_zero_value(v: &Complex) -> bool {
    let (re, im) = parts(v);
    re == Real::ZERO && im == Real::ZERO
}

fn is_real(v: &Complex) -> bool {
    parts(v).1 == Real::ZERO
}

fn is_integer_value(v: &Complex) -> bool {
    let (re, im) = parts(v);
    im == Real::ZERO && re.clone() == re.floor()
}

fn finish(v: &Complex, prec: usize, real: bool) -> Complex {
    let (re, im) = parts(v);
    let im: Real = if real {
        real_int(&Int::ZERO, prec)
    } else {
        im.with_precision(prec).value()
    };
    complex(re.with_precision(prec).value(), im)
}

/// `(z/2)**nu/gamma(nu + 1)*0F1(; nu + 1; sign*z**2/4)` (`sign = -1`:
/// `besselj`, `+1`: `besseli`) at the working precision.
fn regular(nu: &Complex, z: &Complex, sign: i64, wp: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::gamma;
    use crate::functions::special::hyper::hyper_value;
    let shifted: Complex = nu + &c_int(1, wp);
    if is_integer_value(&shifted) && parts(&shifted).0 <= Real::ZERO {
        return None; // 1/gamma vanishes: mpmath's limit is not ported
    }
    let half: Complex = z / &c_int(2, wp);
    let argument: Complex = &(&half * &half) * &c_int(sign, wp);
    let series: Complex = hyper_value(&[], std::slice::from_ref(&shifted), &argument, wp)?;
    let power: Complex = if is_zero_value(&half) {
        return None;
    } else {
        (nu * &half.ln()).exp()
    };
    Some(&(&power * &series) / &gamma(&shifted, wp)?)
}

/// `sin(pi*nu)` and `cos(pi*nu)`.
fn sin_cos_pi(nu: &Complex, wp: usize) -> Option<(Complex, Complex)> {
    let pi: Complex = numeric_value(&Basic::pi(), wp)?;
    let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
    let w: Complex = &(&pi * nu) * &i;
    let (a, b) = (w.exp(), (-&w).exp());
    let two: Complex = c_int(2, wp);
    Some((&(&a - &b) / &(&two * &i), &(&a + &b) / &two))
}

/// The value of a Bessel head at the values of its order and argument;
/// `None` where it is not ported or undefined.
pub fn bessel_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    let wp: usize = prec + 60;
    let raise = |v: &Complex| {
        let (re, im) = parts(v);
        complex(re.with_precision(wp).value(), im.with_precision(wp).value())
    };
    let (nu, z) = (raise(&values[0]), raise(&values[1]));
    if is_zero_value(&z) {
        return None;
    }
    let positive_real: bool = is_real(&z) && parts(&z).0 > Real::ZERO;
    let real_regular: bool = is_real(&nu) && is_real(&z) && (positive_real || is_integer_value(&nu));
    match f {
        Function::BesselJ => Some(finish(&regular(&nu, &z, -1, wp)?, prec, real_regular)),
        Function::BesselI => Some(finish(&regular(&nu, &z, 1, wp)?, prec, real_regular)),
        Function::BesselY | Function::BesselK | Function::Hankel1 | Function::Hankel2 => {
            if is_integer_value(&nu) {
                return None; // mpmath's limit at an integer order is not ported
            }
            let (s, c) = sin_cos_pi(&nu, wp)?;
            let minus_nu: Complex = -&nu;
            let real: bool = is_real(&nu) && positive_real;
            if f == Function::BesselK {
                let pi: Complex = numeric_value(&Basic::pi(), wp)?;
                let difference: Complex = &regular(&minus_nu, &z, 1, wp)? - &regular(&nu, &z, 1, wp)?;
                let value: Complex = &(&(&pi / &c_int(2, wp)) * &difference) / &s;
                return Some(finish(&value, prec, real));
            }
            let j: Complex = regular(&nu, &z, -1, wp)?;
            let y: Complex = &(&(&j * &c) - &regular(&minus_nu, &z, -1, wp)?) / &s;
            let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
            Some(match f {
                Function::BesselY => finish(&y, prec, real),
                Function::Hankel1 => finish(&(&j + &(&i * &y)), prec, false),
                _ => finish(&(&j - &(&i * &y)), prec, false),
            })
        }
        _ => None, // _besseli, _besselk: no value, as in SymPy
    }
}
