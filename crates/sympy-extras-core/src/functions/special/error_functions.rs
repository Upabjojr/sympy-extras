//! Port of the error functions of
//! `sympy_extras/functions/special/error_functions.py`: `erf`, `erfc`,
//! `erfi`, `erfinv`, `erfcinv`, `erf2` and `erf2inv` as function heads:
//! evaluation, derivatives, conjugates, real and imaginary parts, facts and
//! values (the Taylor series of `erf` with the bits its cancellation takes,
//! Newton's method for `erfinv`). The exponential, trigonometric and
//! logarithmic integrals and the Fresnel integrals are not ported yet.

use crate::core::add::{add, sub};
use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, complex_real, parts, real_int, Complex, Real};
use crate::core::expr::{extract_multiplicatively, is, is_true};
use crate::core::logic::truth;
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::PolyResult;
use crate::functions::elementary::exponential::exp;
use crate::numbers::Int;

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

fn minus(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

fn is_infinity(e: Option<&Basic>) -> bool {
    e.is_some_and(|e| matches!(e.head(), Head::Infinity | Head::NegativeInfinity))
}

fn is_f(e: &Basic, f: Function) -> bool {
    matches!(e.head(), Head::Function(g) if *g == f)
}

fn sqrt_pi() -> PolyResult<Basic> {
    pow(&Basic::pi(), &Basic::half())
}

fn erf_eval(a: &Basic) -> PolyResult<Basic> {
    match a.head() {
        Head::NaN => return Ok(Basic::nan()),
        Head::Infinity => return Ok(Basic::one()),
        Head::NegativeInfinity => return Ok(Basic::negative_one()),
        _ => {}
    }
    if is_true(a, Predicate::zero) {
        return Ok(Basic::zero());
    }
    if is_f(a, Function::Erfinv) {
        return Ok(a.args()[0].clone());
    }
    if is_f(a, Function::Erfcinv) {
        return sub(&Basic::one(), &a.args()[0]);
    }
    if is_infinity(extract_multiplicatively(a, &Basic::imaginary_unit())?.as_ref()) {
        return Ok(a.clone());
    }
    if could_extract_minus_sign(a) {
        return minus(&erf(&minus(a)?)?);
    }
    Ok(node(Function::Erf, vec![a.clone()]))
}

fn erfc_eval(a: &Basic) -> PolyResult<Basic> {
    match a.head() {
        Head::NaN => return Ok(Basic::nan()),
        Head::Infinity => return Ok(Basic::zero()),
        _ => {}
    }
    if is_true(a, Predicate::zero) {
        return Ok(Basic::one());
    }
    if is_f(a, Function::Erfinv) {
        return sub(&Basic::one(), &a.args()[0]);
    }
    if is_f(a, Function::Erfcinv) {
        return Ok(a.args()[0].clone());
    }
    if is_infinity(extract_multiplicatively(a, &Basic::imaginary_unit())?.as_ref()) {
        return minus(a);
    }
    if could_extract_minus_sign(a) {
        return sub(&int(2), &erfc(&minus(a)?)?);
    }
    Ok(node(Function::Erfc, vec![a.clone()]))
}

fn erfi_eval(z: &Basic) -> PolyResult<Basic> {
    if matches!(z.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if is_true(z, Predicate::zero) {
        return Ok(Basic::zero());
    }
    if matches!(z.head(), Head::Infinity) {
        return Ok(Basic::oo());
    }
    if could_extract_minus_sign(z) {
        return minus(&erfi(&minus(z)?)?);
    }
    let i: Basic = Basic::imaginary_unit();
    if let Some(rotated) = extract_multiplicatively(z, &i)? {
        if matches!(rotated.head(), Head::Infinity) {
            return Ok(i);
        }
        if is_f(&rotated, Function::Erfinv) {
            return mul(&[i, rotated.args()[0].clone()]);
        }
        if is_f(&rotated, Function::Erfcinv) {
            return mul(&[i, sub(&Basic::one(), &rotated.args()[0])?]);
        }
    }
    Ok(node(Function::Erfi, vec![z.clone()]))
}

fn erfinv_eval(z: &Basic) -> PolyResult<Basic> {
    if matches!(z.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if *z == Basic::negative_one() {
        return Ok(Basic::negative_oo());
    }
    if is_true(z, Predicate::zero) {
        return Ok(Basic::zero());
    }
    if *z == Basic::one() {
        return Ok(Basic::oo());
    }
    if is_f(z, Function::Erf) && is_true(&z.args()[0], Predicate::extended_real) {
        return Ok(z.args()[0].clone());
    }
    if let Some(negated) = extract_multiplicatively(z, &Basic::negative_one())? {
        if is_f(&negated, Function::Erf) && is_true(&negated.args()[0], Predicate::extended_real) {
            return minus(&negated.args()[0]);
        }
    }
    Ok(node(Function::Erfinv, vec![z.clone()]))
}

fn erfcinv_eval(z: &Basic) -> PolyResult<Basic> {
    if matches!(z.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if is_true(z, Predicate::zero) {
        return Ok(Basic::oo());
    }
    if *z == Basic::one() {
        return Ok(Basic::zero());
    }
    if *z == int(2) {
        return Ok(Basic::negative_oo());
    }
    Ok(node(Function::Erfcinv, vec![z.clone()]))
}

fn erf2_eval(x: &Basic, y: &Basic) -> PolyResult<Basic> {
    if matches!(x.head(), Head::NaN) || matches!(y.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if x == y {
        return Ok(Basic::zero());
    }
    let special = |e: &Basic| {
        matches!(e.head(), Head::Infinity | Head::NegativeInfinity) || (e.as_rational().is_some_and(|v| v.is_zero()))
    };
    let difference = || -> PolyResult<Basic> { sub(&erf(y)?, &erf(x)?) };
    if special(x) || special(y) {
        return difference();
    }
    if is_f(y, Function::Erf2inv) && y.args()[0] == *x {
        return Ok(y.args()[1].clone());
    }
    let real_infinite = |e: &Basic| is_true(e, Predicate::extended_real) && is_true(e, Predicate::infinite);
    if is_true(x, Predicate::zero) || is_true(y, Predicate::zero) || real_infinite(x) || real_infinite(y) {
        return difference();
    }
    let (sign_x, sign_y) = (could_extract_minus_sign(x), could_extract_minus_sign(y));
    if sign_x && sign_y {
        return minus(&erf2(&minus(x)?, &minus(y)?)?);
    }
    if sign_x || sign_y {
        return difference();
    }
    Ok(node(Function::Erf2, vec![x.clone(), y.clone()]))
}

fn erf2inv_eval(x: &Basic, y: &Basic) -> PolyResult<Basic> {
    if matches!(x.head(), Head::NaN) || matches!(y.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    let zero = |e: &Basic| is_true(e, Predicate::zero);
    if zero(x) && zero(y) {
        return Ok(Basic::zero());
    }
    if zero(x) && *y == Basic::one() {
        return Ok(Basic::oo());
    }
    if *x == Basic::one() && zero(y) {
        return Ok(Basic::one());
    }
    if zero(x) {
        return erfinv(y);
    }
    if matches!(x.head(), Head::Infinity) {
        return erfcinv(&minus(y)?);
    }
    if zero(y) {
        return Ok(x.clone());
    }
    if matches!(y.head(), Head::Infinity) {
        return erfinv(x);
    }
    Ok(node(Function::Erf2inv, vec![x.clone(), y.clone()]))
}

/// The heads of the module.
pub fn is_error_function(f: Function) -> bool {
    matches!(
        f,
        Function::Erf
            | Function::Erfc
            | Function::Erfi
            | Function::Erfinv
            | Function::Erfcinv
            | Function::Erf2
            | Function::Erf2inv
    )
}

/// The evaluating function of the heads of the module.
pub fn error_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::Erf => erf_eval(&args[0]),
        Function::Erfc => erfc_eval(&args[0]),
        Function::Erfi => erfi_eval(&args[0]),
        Function::Erfinv => erfinv_eval(&args[0]),
        Function::Erfcinv => erfcinv_eval(&args[0]),
        Function::Erf2 => erf2_eval(&args[0], &args[1]),
        _ => erf2inv_eval(&args[0], &args[1]),
    }
}

pub fn erf(x: &Basic) -> PolyResult<Basic> {
    error_function(Function::Erf, std::slice::from_ref(x))
}
pub fn erfc(x: &Basic) -> PolyResult<Basic> {
    error_function(Function::Erfc, std::slice::from_ref(x))
}
pub fn erfi(x: &Basic) -> PolyResult<Basic> {
    error_function(Function::Erfi, std::slice::from_ref(x))
}
pub fn erfinv(x: &Basic) -> PolyResult<Basic> {
    error_function(Function::Erfinv, std::slice::from_ref(x))
}
pub fn erfcinv(x: &Basic) -> PolyResult<Basic> {
    error_function(Function::Erfcinv, std::slice::from_ref(x))
}
pub fn erf2(x: &Basic, y: &Basic) -> PolyResult<Basic> {
    error_function(Function::Erf2, &[x.clone(), y.clone()])
}
pub fn erf2inv(x: &Basic, y: &Basic) -> PolyResult<Basic> {
    error_function(Function::Erf2inv, &[x.clone(), y.clone()])
}

fn gaussian(x: &Basic, sign: i64) -> PolyResult<Basic> {
    // sign*2*exp(-x**2)/sqrt(pi) style factors
    let e: Basic = exp(&mul(&[int(if sign == 0 { 1 } else { -1 }), pow(x, &int(2))?])?)?;
    Ok(e)
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn error_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let x: &Basic = &args[0];
    let half: Basic = Basic::half();
    match f {
        Function::Erf => div(&mul(&[int(2), gaussian(x, 1)?])?, &sqrt_pi()?),
        Function::Erfc => div(&mul(&[int(-2), gaussian(x, 1)?])?, &sqrt_pi()?),
        Function::Erfi => div(&mul(&[int(2), gaussian(x, 0)?])?, &sqrt_pi()?),
        Function::Erfinv => mul(&[sqrt_pi()?, exp(&pow(&erfinv(x)?, &int(2))?)?, half]),
        Function::Erfcinv => mul(&[
            Basic::negative_one(),
            sqrt_pi()?,
            exp(&pow(&erfcinv(x)?, &int(2))?)?,
            half,
        ]),
        Function::Erf2 => {
            if argindex == 1 {
                div(&mul(&[int(-2), gaussian(&args[0], 1)?])?, &sqrt_pi()?)
            } else {
                div(&mul(&[int(2), gaussian(&args[1], 1)?])?, &sqrt_pi()?)
            }
        }
        _ => {
            let (x, y) = (&args[0], &args[1]);
            let w: Basic = erf2inv(x, y)?;
            if argindex == 1 {
                exp(&sub(&pow(&w, &int(2))?, &pow(x, &int(2))?)?)
            } else {
                mul(&[sqrt_pi()?, half, exp(&pow(&w, &int(2))?)?])
            }
        }
    }
}

/// The conjugate (`_eval_conjugate`); `None` for the default rule.
pub fn error_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::conjugate;
    match f {
        Function::Erf | Function::Erfc | Function::Erfi => Ok(Some(error_function(f, &[conjugate(&e.args()[0])?])?)),
        Function::Erf2 => Ok(Some(erf2(&conjugate(&e.args()[0])?, &conjugate(&e.args()[1])?)?)),
        _ => Ok(None),
    }
}

/// The real and imaginary parts of `erf`, `erfc`, `erfi`
/// (`real_to_real_as_real_imag`); `None` for the other heads.
pub fn error_real_imag(f: Function, e: &Basic, deep: bool) -> PolyResult<Option<Option<(Basic, Basic)>>> {
    if !matches!(f, Function::Erf | Function::Erfc | Function::Erfi) {
        return Ok(None);
    }
    let a: &Basic = &e.args()[0];
    if is_true(a, Predicate::extended_real) {
        return Ok(Some(Some((e.clone(), Basic::zero()))));
    }
    let source: Basic = if deep {
        crate::core::function::expand_method(a)?
    } else {
        a.clone()
    };
    let Some((x, y)) = crate::functions::elementary::complexes::as_real_imag(&source, deep, None)? else {
        return Ok(Some(None));
    };
    let i: Basic = Basic::imaginary_unit();
    let plus: Basic = error_function(f, &[add(&[x.clone(), mul(&[i.clone(), y.clone()])?])?])?;
    let minus_: Basic = error_function(f, &[sub(&x, &mul(&[i.clone(), y])?)?])?;
    Ok(Some(Some((
        div(&add(&[plus.clone(), minus_.clone()])?, &int(2))?,
        div(&sub(&plus, &minus_)?, &mul(&[int(2), i])?)?,
    ))))
}

/// The facts (`_eval_facts`).
pub fn error_facts(f: Function, args: &[Basic]) -> FactList {
    use Predicate as P;
    let mut facts: FactList = FactList::new();
    facts.set(P::commutative, true);
    let a = facts_of(&args[0]);
    let get = |p: P| a.get(&p).copied();
    match f {
        Function::Erf => {
            if get(P::finite) == Some(true) || get(P::extended_real) == Some(true) {
                facts.set(P::finite, true);
            }
            if get(P::extended_real) == Some(true) {
                facts.set(P::real, true);
                for (name, source) in [
                    (P::zero, P::zero),
                    (P::positive, P::extended_positive),
                    (P::negative, P::extended_negative),
                ] {
                    if let Some(v) = get(source) {
                        facts.set(name, v);
                    }
                }
            }
        }
        Function::Erfc => {
            if get(P::extended_real) == Some(true) {
                facts.set(P::real, true);
            } else if get(P::imaginary) == Some(true) {
                facts.set(P::real, false);
            }
        }
        Function::Erfi => {
            if get(P::extended_real) == Some(true) {
                facts.set(P::extended_real, true);
            }
            if let Some(v) = get(P::zero) {
                facts.set(P::zero, v);
            }
        }
        Function::Erfinv => {
            if let Some(v) = get(P::zero) {
                facts.set(P::zero, v);
            }
        }
        Function::Erfcinv => {
            if let Some(v) = sub(&args[0], &Basic::one()).ok().and_then(|d| is(&d, P::zero)) {
                facts.set(P::zero, v);
            }
        }
        Function::Erf2 => {
            let (x, y) = (&args[0], &args[1]);
            if is_true(x, P::extended_real) && is_true(y, P::extended_real) {
                facts.set(P::extended_real, true);
            }
            if let Ok(equal) = crate::core::relational::eq(x, y) {
                match truth(&equal) {
                    Some(true) => facts.set(P::zero, true),
                    Some(false) => facts.set(P::zero, false),
                    None => {}
                }
            }
        }
        _ => {
            if is_true(&args[0], P::zero) && is_true(&args[1], P::zero) {
                facts.set(P::zero, true);
            }
        }
    }
    facts
}

fn modulus(z: &Complex) -> f64 {
    let (a, b) = parts(z);
    (a.to_f64().value().powi(2) + b.to_f64().value().powi(2)).sqrt()
}

fn c_int(n: i64, wp: usize) -> Complex {
    complex_real(real_int(&Int::from(n), wp), wp)
}

fn finish(value: &Complex, prec: usize, real: bool) -> Complex {
    let (re, im) = parts(value);
    let im: Real = if real {
        real_int(&Int::ZERO, prec)
    } else {
        im.with_precision(prec).value()
    };
    complex(re.with_precision(prec).value(), im)
}

/// `erf(z)` by its Taylor series, at `wp` bits plus those the series
/// cancels (`|z|**2/log(2)`).
fn erf_series(z: &Complex, wp: usize) -> Option<Complex> {
    let size: f64 = modulus(z);
    if size > 60.0 {
        return None;
    }
    let wp: usize = wp + (size * size * 1.443) as usize + 20;
    let (re, im) = parts(z);
    let z: Complex = complex(re.with_precision(wp).value(), im.with_precision(wp).value());
    let z2: Complex = &z * &z;
    // sum (-1)**n z**(2n+1)/(n!(2n+1))
    let mut power: Complex = z.clone();
    let mut total: Complex = z.clone();
    let mut n: i64 = 1;
    loop {
        power = &(-&(&power * &z2)) / &c_int(n, wp);
        let term: Complex = &power / &c_int(2 * n + 1, wp);
        total = &total + &term;
        let t: f64 = modulus(&term);
        if t == 0.0 || (n > 2 && t.log2() < -((wp + 10) as f64)) || n > 200_000 {
            break;
        }
        n += 1;
    }
    let pi: Real = Real::pi(wp);
    let factor: Real = real_int(&Int::from(2u8), wp) / pi.sqrt();
    Some(&total * &complex_real(factor, wp))
}

/// The inverse of `erf` on `(-1, 1)` by Newton's method.
fn erfinv_real(x: &Real, prec: usize) -> Option<Real> {
    let xf: f64 = x.to_f64().value();
    if xf.abs() >= 1.0 {
        return None;
    }
    // Giles' approximation as the start
    let w: f64 = -((1.0 - xf) * (1.0 + xf)).ln();
    let mut y: f64 = if w < 5.0 {
        let w = w - 2.5;
        let mut p = 2.81022636e-08;
        for c in [
            3.43273939e-07,
            -3.5233877e-06,
            -4.39150654e-06,
            0.00021858087,
            -0.00125372503,
            -0.00417768164,
            0.246640727,
            1.50140941,
        ] {
            p = c + p * w;
        }
        p * xf
    } else {
        let w = w.sqrt() - 3.0;
        let mut p = -0.000200214257;
        for c in [
            0.000100950558,
            0.00134934322,
            -0.00367342844,
            0.00573950773,
            -0.0076224613,
            0.00943887047,
            1.00167406,
            2.83297682,
        ] {
            p = c + p * w;
        }
        p * xf
    };
    if !y.is_finite() {
        y = 0.0;
    }
    let wp: usize = prec + 40;
    let mut current: Real = Real::try_from(y).ok()?.with_precision(wp).value();
    let target: Complex = complex_real(x.clone().with_precision(wp).value(), wp);
    let pi_root: Real = Real::pi(wp).sqrt();
    for _ in 0..200 {
        let value: Complex = erf_series(&complex_real(current.clone(), wp), wp)?;
        let residual: Real = parts(&(&value - &target)).0;
        let slope: Real =
            (real_int(&Int::from(2u8), wp) / pi_root.clone()) * (-(current.clone() * current.clone())).exp();
        let step: Real = residual / slope;
        current -= step.clone();
        let s: f64 = step.to_f64().value().abs();
        if s == 0.0 || s.log2() < -((wp - 10) as f64) {
            break;
        }
    }
    Some(current.with_precision(prec).value())
}

/// The value (`numeric`).
pub fn error_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    let wp: usize = prec + 20;
    let z: &Complex = &values[0];
    let real: bool = parts(z).1 == Real::ZERO;
    let one: Complex = c_int(1, wp);
    let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
    let value: Complex = match f {
        Function::Erf => erf_series(z, wp)?,
        Function::Erfc => &one - &erf_series(z, wp)?,
        Function::Erfi => &(-&i) * &erf_series(&(&i * z), wp)?,
        Function::Erfinv | Function::Erfcinv => {
            let (re, im) = parts(z);
            if im != Real::ZERO {
                return None;
            }
            let x: Real = if f == Function::Erfinv {
                re
            } else {
                real_int(&Int::ONE, wp) - re
            };
            if !(x > -real_int(&Int::ONE, wp) && x < real_int(&Int::ONE, wp)) {
                return None;
            }
            let y: Real = erfinv_real(&x, prec)?;
            return Some(complex(y, real_int(&Int::ZERO, prec)));
        }
        Function::Erf2 => {
            let real2: bool = real && parts(&values[1]).1 == Real::ZERO;
            let v: Complex = &erf_series(&values[1], wp)? - &erf_series(z, wp)?;
            return Some(finish(&v, prec, real2));
        }
        _ => return None,
    };
    Some(finish(&value, prec, real))
}
