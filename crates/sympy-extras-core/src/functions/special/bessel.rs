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
//! not ported). The branch factor `exp_polar(2*pi*I*n)` of the argument
//! is taken out as `exp(2*n*pi*nu*I)` for a non-integer order.
//!
//! Part 2: the spherical Bessel functions `jn`, `yn`, `hn1`, `hn2` (values
//! at `0` and the infinities for `jn`, the derivative `f(n - 1, z) - f(n,
//! z)*(n + 1)/z`, values from `besselj`, `bessely` of order `n + 1/2`), the
//! Airy functions `airyai`, `airybi`, `airyaiprime`, `airybiprime`
//! (values at `0` and the infinities, derivatives, conjugates, SymPy's
//! complex parts, values by `0F1` for a moderate argument) and `marcumq`
//! (SymPy's closed forms, derivatives, no values). Not ported: `jn_zeros`
//! (floats: the Rust core has none), the Taylor terms and the asymptotic
//! series.

// PORTED FROM SYMPY: sympy/functions/special/bessel.py (BesselBase, besselj, bessely,
// besseli, besselk, hankel1, hankel2, _besseli, _besselk, jn, yn, hn1, hn2, airyai,
// airybi, airyaiprime, airybiprime, marcumq) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
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
            | Function::SphericalJ
            | Function::SphericalY
            | Function::SphericalHankel1
            | Function::SphericalHankel2
    )
}

/// The Airy heads.
pub fn is_airy(f: Function) -> bool {
    matches!(
        f,
        Function::AiryAi | Function::AiryBi | Function::AiryAiPrime | Function::AiryBiPrime
    )
}

fn is_spherical(f: Function) -> bool {
    matches!(
        f,
        Function::SphericalJ | Function::SphericalY | Function::SphericalHankel1 | Function::SphericalHankel2
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
        Function::SphericalJ => jn_eval(&args[0], &args[1])?,
        _ => None, // hankel1, hankel2, _besseli, _besselk, yn, hn1, hn2: no evaluation
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
/// the polar factors of the argument, another one takes out
/// `exp(2*n*pi*nu*I)` for the branch factor `exp_polar(2*pi*I*n)`; the
/// order is unpolarified.
fn branch(f: Function, nu: &Basic, z: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::function::unpolarify;
    if is_true(nu, Predicate::integer) {
        let newz: Basic = unpolarify(z)?;
        if newz != *z {
            return Ok(Some(bessel(f, &[nu.clone(), newz])?));
        }
    } else {
        let (newz, n) = crate::functions::elementary::polar::extract_branch_factor(z, false)?;
        if n != Basic::zero() {
            let phase: Basic = crate::functions::elementary::exponential::exp(&crate::core::mul::mul(&[
                Basic::int(2),
                n,
                Basic::pi(),
                nu.clone(),
                Basic::imaginary_unit(),
            ])?)?;
            return Ok(Some(crate::core::mul::mul(&[phase, bessel(f, &[nu.clone(), newz])?])?));
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
    if argindex != 2 || matches!(f, Function::BesselIs | Function::BesselKs) {
        return crate::functions::special::hyper::default_fdiff(e, argindex);
    }
    let (nu, z) = (&e.args()[0], &e.args()[1]);
    if is_spherical(f) {
        let lower: Basic = bessel(f, &[sub(nu, &Basic::one())?, z.clone()])?;
        let scaled: Basic = crate::core::mul::div(&mul(&[e.clone(), add(&[nu.clone(), Basic::one()])?])?, z)?;
        return sub(&lower, &scaled);
    }
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
        _ => None, // the Hankel and spherical functions, the helpers
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
        Function::SphericalJ | Function::SphericalY | Function::SphericalHankel1 | Function::SphericalHankel2 => {
            // sqrt(pi/(2*z))*besselj(n + 1/2, z) (bessely for yn)
            let pi: Complex = numeric_value(&Basic::pi(), wp)?;
            let root: Complex = crate::core::evalf::complex_sqrt(&(&pi / &(&c_int(2, wp) * &z)), wp);
            let order: Complex = &nu + &half_value(wp);
            let j = || bessel_value(Function::BesselJ, &[order.clone(), z.clone()], wp);
            let y = || bessel_value(Function::BesselY, &[order.clone(), z.clone()], wp);
            let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
            let real: bool = is_real(&nu) && positive_real;
            Some(match f {
                Function::SphericalJ => finish(&(&root * &j()?), prec, real),
                Function::SphericalY => finish(&(&root * &y()?), prec, real),
                Function::SphericalHankel1 => finish(&(&root * &(&j()? + &(&i * &y()?))), prec, false),
                _ => finish(&(&root * &(&j()? - &(&i * &y()?))), prec, false),
            })
        }
        _ => None, // _besseli, _besselk: no value, as in SymPy
    }
}

fn half_value(wp: usize) -> Complex {
    &c_int(1, wp) / &c_int(2, wp)
}

fn jn_eval(nu: &Basic, z: &Basic) -> PolyResult<Option<Basic>> {
    if is_true(z, Predicate::zero) {
        if is_true(nu, Predicate::zero) {
            return Ok(Some(Basic::one()));
        }
        if is_true(nu, Predicate::integer) {
            return Ok(Some(if is_true(nu, Predicate::positive) {
                Basic::zero()
            } else {
                Basic::zoo()
            }));
        }
    }
    if is_infinity(z) {
        return Ok(Some(Basic::zero()));
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// Airy functions
// ---------------------------------------------------------------------

pub fn airy(f: Function, z: &Basic) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, vec![z.clone()]));
    }
    Ok(airy_eval(f, z)?.unwrap_or_else(|| node(f, vec![z.clone()])))
}

/// `numerator/(3**power*gamma(gamma_argument))` (SymPy's values at 0).
fn airy_at_zero(power: (i64, i64), gamma_argument: (i64, i64), numerator: Basic) -> PolyResult<Basic> {
    use crate::functions::special::gamma_functions::gamma;
    let p: Basic = Basic::rational(Int::from(power.0), Int::from(power.1))?;
    let g: Basic = gamma(&Basic::rational(
        Int::from(gamma_argument.0),
        Int::from(gamma_argument.1),
    )?)?;
    crate::core::mul::div(&numerator, &mul(&[pow(&Basic::integer(Int::from(3)), &p)?, g])?)
}

fn airy_eval(f: Function, z: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::special::gamma_functions::gamma;
    if matches!(z.head(), Head::NaN) {
        return Ok(Some(Basic::nan()));
    }
    let (at_oo, at_minus_oo): (Option<Basic>, Option<Basic>) = match f {
        Function::AiryAi => (Some(Basic::zero()), Some(Basic::zero())),
        Function::AiryBi | Function::AiryBiPrime => (Some(Basic::oo()), Some(Basic::zero())),
        _ => (Some(Basic::zero()), None),
    };
    if matches!(z.head(), Head::Infinity) {
        if let Some(v) = at_oo {
            return Ok(Some(v));
        }
    }
    if matches!(z.head(), Head::NegativeInfinity) {
        if let Some(v) = at_minus_oo {
            return Ok(Some(v));
        }
    }
    if is_true(z, Predicate::zero) {
        return Ok(Some(match f {
            Function::AiryAi => airy_at_zero((2, 3), (2, 3), Basic::one())?,
            Function::AiryBi => airy_at_zero((1, 6), (2, 3), Basic::one())?,
            Function::AiryAiPrime => airy_at_zero((1, 3), (1, 3), Basic::negative_one())?,
            _ => crate::core::mul::div(
                &pow(&Basic::integer(Int::from(3)), &Basic::rational(Int::ONE, Int::from(6))?)?,
                &gamma(&Basic::rational(Int::ONE, Int::from(3))?)?,
            )?,
        }));
    }
    Ok(None)
}

/// The derivative of an Airy function.
pub fn airy_fdiff(f: Function, e: &Basic) -> PolyResult<Basic> {
    let z: &Basic = &e.args()[0];
    match f {
        Function::AiryAi => airy(Function::AiryAiPrime, z),
        Function::AiryBi => airy(Function::AiryBiPrime, z),
        Function::AiryAiPrime => mul(&[z.clone(), airy(Function::AiryAi, z)?]),
        _ => mul(&[z.clone(), airy(Function::AiryBi, z)?]),
    }
}

/// `f(conjugate(z))`.
pub fn airy_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    Ok(Some(airy(
        f,
        &crate::functions::elementary::complexes::conjugate(&e.args()[0])?,
    )?))
}

/// SymPy's complex parts `((f(z) + f(conj(z)))/2, I*(f(conj(z)) - f(z))/2)`.
pub fn airy_real_imag(f: Function, e: &Basic) -> PolyResult<(Basic, Basic)> {
    let z: &Basic = &e.args()[0];
    let f_z: Basic = airy(f, z)?;
    let f_conjugate: Basic = airy(f, &crate::functions::elementary::complexes::conjugate(z)?)?;
    let two: Basic = Basic::integer(Int::from(2));
    Ok((
        crate::core::mul::div(&add(&[f_z.clone(), f_conjugate.clone()])?, &two)?,
        crate::core::mul::div(&mul(&[Basic::imaginary_unit(), sub(&f_conjugate, &f_z)?])?, &two)?,
    ))
}

/// The facts: extended real as the argument.
pub fn airy_facts(args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    if let Some(real) = crate::core::assumptions::facts_of(&args[0]).get(&Predicate::extended_real) {
        facts.set(Predicate::extended_real, *real);
    }
    facts
}

/// The value of an Airy function by `0F1` (`Ai = c1*f - c2*g`, `Bi =
/// sqrt(3)*(c1*f + c2*g)` with `f = 0F1(; 2/3; z**3/9)`, `g = z*0F1(;
/// 4/3; z**3/9)`, and their derivatives), the precision raised for the
/// cancellation of the two terms.
pub fn airy_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::gamma;
    use crate::functions::special::hyper::hyper_value;
    let real: bool = is_real(&values[0]);
    let mut wp: usize = prec + 60;
    for _ in 0..4 {
        let raise = |v: &Complex| {
            let (re, im) = parts(v);
            complex(re.with_precision(wp).value(), im.with_precision(wp).value())
        };
        let z: Complex = raise(&values[0]);
        let third = |p: i64| &c_int(p, wp) / &c_int(3, wp);
        let w: Complex = &(&(&z * &z) * &z) / &c_int(9, wp);
        let f0f1 = |b: Complex| hyper_value(&[], std::slice::from_ref(&b), &w, wp);
        let c1: Complex = &c_int(1, wp) / &(&cube_root_power(2, wp) * &gamma(&third(2), wp)?);
        let c2: Complex = &c_int(1, wp) / &(&cube_root_power(1, wp) * &gamma(&third(1), wp)?);
        let (first, second): (Complex, Complex) = match f {
            Function::AiryAi | Function::AiryBi => (&c1 * &f0f1(third(2))?, &(&c2 * &z) * &f0f1(third(4))?),
            _ => (
                &(&c1 * &(&(&z * &z) / &c_int(2, wp))) * &f0f1(third(5))?,
                &c2 * &f0f1(third(1))?,
            ),
        };
        let value: Complex = match f {
            Function::AiryAi | Function::AiryAiPrime => &first - &second,
            _ => &crate::core::evalf::complex_sqrt(&c_int(3, wp), wp) * &(&first + &second),
        };
        let size = |v: &Complex| {
            let (a, b) = parts(v);
            let (a, b) = (a.to_f64().value(), b.to_f64().value());
            (a * a + b * b).sqrt()
        };
        let largest: f64 = size(&first).max(size(&second));
        let lost: f64 = if largest == 0.0 {
            0.0
        } else if size(&value) == 0.0 {
            f64::INFINITY
        } else {
            (largest / size(&value)).log2().max(0.0)
        };
        if lost.is_finite() && lost + ((prec + 20) as f64) < wp as f64 {
            return Some(finish(&value, prec, real));
        }
        wp = if lost.is_finite() {
            prec + 60 + lost.ceil() as usize
        } else {
            wp * 2
        };
    }
    None
}

/// `3**(k/3)`.
fn cube_root_power(k: i64, wp: usize) -> Complex {
    let three: Complex = c_int(3, wp);
    (&(&c_int(k, wp) / &three) * &three.ln()).exp()
}

// ---------------------------------------------------------------------
// marcumq
// ---------------------------------------------------------------------

pub fn marcumq(args: &[Basic]) -> PolyResult<Basic> {
    let n = || node(Function::MarcumQ, args.to_vec());
    if !evaluate() {
        return Ok(n());
    }
    Ok(marcumq_eval(&args[0], &args[1], &args[2])?.unwrap_or_else(n))
}

fn marcumq_eval(m: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::mul::div;
    use crate::functions::elementary::exponential::exp;
    use crate::functions::special::gamma_functions::{gamma, uppergamma};
    let two: Basic = Basic::integer(Int::from(2));
    let half: Basic = Basic::half();
    let zero = |e: &Basic| is_true(e, Predicate::zero);
    if zero(a) {
        if zero(m) && zero(b) {
            return Ok(Some(Basic::zero()));
        }
        let upper: Basic = uppergamma(m, &mul(&[pow(b, &two)?, half])?)?;
        return Ok(Some(div(&upper, &gamma(m)?)?));
    }
    if zero(m) && zero(b) {
        let decay: Basic = exp(&mul(&[pow(a, &two)?, half])?)?;
        return Ok(Some(sub(&Basic::one(), &div(&Basic::one(), &decay)?)?));
    }
    if a == b {
        let decay: Basic = exp(&mul(&[Basic::negative_one(), pow(a, &two)?])?)?;
        let a2: Basic = pow(a, &two)?;
        if *m == Basic::one() {
            let inner: Basic = add(&[Basic::one(), mul(&[decay, besseli(&Basic::zero(), &a2)?])?])?;
            return Ok(Some(mul(&[inner, half])?));
        }
        if *m == two {
            return Ok(Some(add(&[
                half.clone(),
                mul(&[half, decay.clone(), besseli(&Basic::zero(), &a2)?])?,
                mul(&[decay, besseli(&Basic::one(), &a2)?])?,
            ])?));
        }
    }
    Ok(None)
}

pub fn marcumq_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::core::mul::div;
    use crate::functions::elementary::exponential::exp;
    let (m, a, b) = (&e.args()[0], &e.args()[1], &e.args()[2]);
    let two: Basic = Basic::integer(Int::from(2));
    if argindex == 2 {
        let next: Basic = marcumq(&[add(&[Basic::one(), m.clone()])?, a.clone(), b.clone()])?;
        return mul(&[
            a.clone(),
            add(&[neg(&marcumq(&[m.clone(), a.clone(), b.clone()])?)?, next])?,
        ]);
    }
    if argindex == 3 {
        let ratio: Basic = div(&neg(&pow(b, m)?)?, &pow(a, &sub(m, &Basic::one())?)?)?;
        let decay: Basic = exp(&div(&neg(&add(&[pow(a, &two)?, pow(b, &two)?])?)?, &two)?)?;
        return mul(&[
            ratio,
            decay,
            besseli(&sub(m, &Basic::one())?, &mul(&[a.clone(), b.clone()])?)?,
        ]);
    }
    crate::functions::special::hyper::default_fdiff(e, argindex)
}

pub fn marcumq_facts(args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    if args.iter().all(|a| is_true(a, Predicate::zero)) {
        facts.set(Predicate::zero, true);
    }
    facts
}
