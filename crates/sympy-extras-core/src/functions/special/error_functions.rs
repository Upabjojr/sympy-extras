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
        _ => default_conjugate(e),
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

// ---------------------------------------------------------------------
// part 2: the exponential, trigonometric and logarithmic integrals and the
// Fresnel integrals (without the polar arguments: the Rust core has no
// polar numbers, so the branch factors of SymPy are always trivial)

/// The heads of part 2.
pub fn is_integral_function(f: Function) -> bool {
    matches!(
        f,
        Function::Expint
            | Function::Ei
            | Function::Si
            | Function::Ci
            | Function::Shi
            | Function::Chi
            | Function::LogIntegral
            | Function::OffsetLogIntegral
            | Function::FresnelS
            | Function::FresnelC
    )
}

fn odd_integral(f: Function) -> bool {
    matches!(f, Function::Si | Function::Shi)
}

/// `TrigonometricIntegral.eval` (Si, Ci, Shi, Chi).
fn trigonometric_integral_eval(f: Function, z: &Basic) -> PolyResult<Basic> {
    let i: Basic = Basic::imaginary_unit();
    let pi: Basic = Basic::pi();
    let half: Basic = Basic::half();
    if is_true(z, Predicate::zero) {
        return Ok(match f {
            Function::Si | Function::Shi => Basic::zero(),
            _ => Basic::zoo(),
        });
    }
    if matches!(z.head(), Head::Infinity) {
        return match f {
            Function::Si => mul(&[pi, half]),
            Function::Ci => Ok(Basic::zero()),
            _ => Ok(Basic::oo()),
        };
    }
    if matches!(z.head(), Head::NegativeInfinity) {
        return match f {
            Function::Si => mul(&[Basic::negative_one(), pi, half]),
            Function::Ci => mul(&[i, pi]),
            Function::Shi => Ok(Basic::negative_oo()),
            _ => Ok(Basic::oo()),
        };
    }
    if odd_integral(f) {
        if let Some(nz) = extract_multiplicatively(z, &i)? {
            // i_factor(nz, 1)
            let other: Basic = if f == Function::Si {
                integral(Function::Shi, &nz)?
            } else {
                integral(Function::Si, &nz)?
            };
            return mul(&[i, other, Basic::one()]);
        }
        if let Some(nz) = extract_multiplicatively(z, &Basic::negative_one())? {
            return minus(&integral(f, &nz)?);
        }
    }
    Ok(node(f, vec![z.clone()]))
}

fn integral(f: Function, z: &Basic) -> PolyResult<Basic> {
    integral_function(f, std::slice::from_ref(z))
}

fn fresnel_eval(f: Function, z: &Basic) -> PolyResult<Basic> {
    if matches!(z.head(), Head::Infinity) {
        return Ok(Basic::half());
    }
    if is_true(z, Predicate::zero) {
        return Ok(Basic::zero());
    }
    let sign: i64 = if f == Function::FresnelS { -1 } else { 1 };
    let mut prefactor: Basic = Basic::one();
    let mut argument: Basic = z.clone();
    let mut changed: bool = false;
    if let Some(negated) = extract_multiplicatively(&argument, &Basic::negative_one())? {
        prefactor = minus(&prefactor)?;
        argument = negated;
        changed = true;
    }
    if let Some(rotated) = extract_multiplicatively(&argument, &Basic::imaginary_unit())? {
        prefactor = mul(&[int(sign), Basic::imaginary_unit(), prefactor])?;
        argument = rotated;
        changed = true;
    }
    if changed {
        return mul(&[prefactor, integral(f, &argument)?]);
    }
    Ok(node(f, vec![z.clone()]))
}

fn part2_eval(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    match f {
        Function::Expint => {
            let nu: &Basic = &args[0];
            let integer: bool = nu.as_rational().is_some_and(|v| crate::numbers::denom(v) == Int::ONE);
            let half_integer: bool = !integer
                && nu
                    .as_rational()
                    .is_some_and(|v| crate::numbers::denom(v) == Int::from(2));
            if (integer && is_true(nu, Predicate::nonpositive)) || half_integer {
                use crate::core::function::{expand_mul, unpolarify};
                use crate::functions::special::gamma_functions::uppergamma;
                let z: &Basic = &args[1];
                let one: Basic = Basic::one();
                let product: Basic = mul(&[pow(z, &sub(nu, &one)?)?, uppergamma(&sub(&one, nu)?, z)?])?;
                return unpolarify(&expand_mul(&product, true)?);
            }
            Ok(node(f, args.to_vec()))
        }
        Function::Ei => {
            let z: &Basic = &args[0];
            if is_true(z, Predicate::zero) {
                return Ok(Basic::negative_oo());
            }
            match z.head() {
                Head::Infinity => Ok(Basic::oo()),
                Head::NegativeInfinity => Ok(Basic::zero()),
                _ => Ok(node(f, args.to_vec())),
            }
        }
        Function::Si | Function::Ci | Function::Shi | Function::Chi => trigonometric_integral_eval(f, &args[0]),
        Function::LogIntegral => {
            let z: &Basic = &args[0];
            if is_true(z, Predicate::zero) {
                return Ok(Basic::zero());
            }
            if *z == Basic::one() {
                return Ok(Basic::negative_oo());
            }
            if matches!(z.head(), Head::Infinity) {
                return Ok(Basic::oo());
            }
            Ok(node(f, args.to_vec()))
        }
        Function::OffsetLogIntegral => {
            let z: &Basic = &args[0];
            if matches!(z.head(), Head::Infinity) {
                return Ok(Basic::oo());
            }
            if *z == int(2) {
                return Ok(Basic::zero());
            }
            Ok(node(f, args.to_vec()))
        }
        _ => fresnel_eval(f, &args[0]),
    }
}

/// The evaluating function of part 2.
pub fn integral_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    part2_eval(f, args)
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn integral_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::functions::elementary::exponential::log;
    use crate::functions::elementary::hyperbolic::{cosh, sinh};
    use crate::functions::elementary::trigonometric::{cos, sin};
    let args: &[Basic] = e.args();
    let z: &Basic = args.last().expect("an argument");
    match f {
        Function::Expint => {
            if argindex == 2 {
                return minus(&integral_function(
                    Function::Expint,
                    &[sub(&args[0], &Basic::one())?, z.clone()],
                )?);
            }
            let a: &Basic = &args[0];
            if a.is_symbol() && !z.free_symbols().contains(a) {
                return Ok(crate::core::function::derivative_node(e, &[(a.clone(), 1)]));
            }
            Err(crate::error::not_ported("a derivative as a Subs"))
        }
        Function::Ei => div(&exp(z)?, z),
        Function::Si => div(&sin(z)?, z),
        Function::Ci => div(&cos(z)?, z),
        Function::Shi => div(&sinh(z)?, z),
        Function::Chi => div(&cosh(z)?, z),
        Function::LogIntegral | Function::OffsetLogIntegral => div(&Basic::one(), &log(z)?),
        _ => {
            let w: Basic = mul(&[Basic::half(), Basic::pi(), pow(z, &int(2))?])?;
            if f == Function::FresnelS {
                sin(&w)
            } else {
                cos(&w)
            }
        }
    }
}

/// The conjugate (`_eval_conjugate`); `None` for the default rule.
pub fn integral_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::conjugate;
    let z: &Basic = &e.args()[0];
    match f {
        Function::LogIntegral => {
            if is_true(z, Predicate::extended_negative) {
                // the branch cut (-oo, 0): Python's override gives no rule
                return Ok(None);
            }
            Ok(Some(integral(f, &conjugate(z)?)?))
        }
        Function::FresnelS | Function::FresnelC => Ok(Some(integral(f, &conjugate(z)?)?)),
        _ => default_conjugate(e),
    }
}

/// The facts (`_eval_facts`).
pub fn integral_facts(f: Function, args: &[Basic]) -> FactList {
    use Predicate as P;
    let mut facts: FactList = FactList::new();
    facts.set(P::commutative, true);
    let z: &Basic = args.last().expect("an argument");
    match f {
        Function::Si | Function::Shi | Function::LogIntegral => {
            if is_true(z, P::zero) {
                facts.set(P::zero, true);
            }
        }
        Function::FresnelS | Function::FresnelC => {
            let a = facts_of(z);
            for p in [P::extended_real, P::zero] {
                if let Some(v) = a.get(&p) {
                    facts.set(p, *v);
                }
            }
        }
        _ => {}
    }
    facts
}

fn ln(z: &Complex) -> Complex {
    z.ln()
}

fn euler(wp: usize) -> Complex {
    complex_real(crate::core::evalf::euler_gamma(wp), wp)
}

/// `sum_{k >= 1} sign**k z**(step*k + offset)/(k-scaled)` helpers: the
/// series of the exponential and trigonometric integrals.
fn ei_series(z: &Complex, wp: usize) -> Complex {
    // sum z**k/(k*k!)
    let mut term: Complex = c_int(1, wp);
    let mut total: Complex = c_int(0, wp);
    let mut k: i64 = 1;
    loop {
        term = &(&term * z) / &c_int(k, wp);
        let t: Complex = &term / &c_int(k, wp);
        total = &total + &t;
        let size: f64 = modulus(&t);
        if size == 0.0 || (k > 2 && size.log2() < -((wp + 10) as f64)) || k > 400_000 {
            break;
        }
        k += 1;
    }
    total
}

/// `sum_k (sign)**k z**(2k + offset)/((2k + offset)*(2k + offset)!)`
/// (offset 1: Si, Shi; offset 0 from k = 1: Ci, Chi).
fn trig_series(z: &Complex, wp: usize, alternating: bool, odd: bool) -> Complex {
    let z2: Complex = z * z;
    let mut power: Complex = if odd { z.clone() } else { c_int(1, wp) };
    let mut total: Complex = if odd { z.clone() } else { c_int(0, wp) };
    let mut n: i64 = if odd { 1 } else { 0 };
    loop {
        // power = z**n/n!, from n to n + 2
        power = &(&power * &z2) / &c_int((n + 1) * (n + 2), wp);
        if alternating {
            power = -&power;
        }
        n += 2;
        let t: Complex = &power / &c_int(n, wp);
        total = &total + &t;
        let size: f64 = modulus(&t);
        if size == 0.0 || (n > 4 && size.log2() < -((wp + 10) as f64)) || n > 400_000 {
            break;
        }
    }
    total
}

fn bits_for(z: &Complex, factor: f64) -> usize {
    (modulus(z) * factor * 1.443) as usize + 20
}

fn raise(z: &Complex, wp: usize) -> Complex {
    let (re, im) = parts(z);
    complex(re.with_precision(wp).value(), im.with_precision(wp).value())
}

fn is_zero_value(z: &Complex) -> bool {
    let (re, im) = parts(z);
    re == Real::ZERO && im == Real::ZERO
}

/// mpmath's `ei(z) = EulerGamma + (log(z) - log(1/z))/2 + sum z**k/(k*k!)`.
fn ei_value(z: &Complex, wp: usize) -> Option<Complex> {
    if is_zero_value(z) || modulus(z) > 2000.0 {
        return None;
    }
    let wp: usize = wp + bits_for(z, 1.0);
    let z: Complex = raise(z, wp);
    let half: Complex = &c_int(1, wp) / &c_int(2, wp);
    let logs: Complex = &(&ln(&z) - &ln(&(&c_int(1, wp) / &z))) * &half;
    Some(&(&euler(wp) + &logs) + &ei_series(&z, wp))
}

/// `E_n(z)` for an integer `n >= 0`: `E_1 = -EulerGamma - log(z) -
/// sum (-z)**k/(k*k!)`, `E_0 = exp(-z)/z`, `E_{n+1} = (exp(-z) - z*E_n)/n`.
fn expint_value(n: i64, z: &Complex, wp: usize) -> Option<Complex> {
    if is_zero_value(z) {
        // E_n(0) = 1/(n - 1) for n >= 2, a pole otherwise
        if n >= 2 {
            return Some(&c_int(1, wp) / &c_int(n - 1, wp));
        }
        return None;
    }
    if modulus(z) > 2000.0 {
        return None;
    }
    let wp: usize = wp + bits_for(z, 1.0) + 2 * n as usize;
    let z: Complex = raise(z, wp);
    let minus_z: Complex = -&z;
    let e_minus: Complex = minus_z.exp();
    if n == 0 {
        return Some(&e_minus / &z);
    }
    let mut value: Complex = &(&(-&euler(wp)) - &ln(&z)) - &ei_series(&minus_z, wp);
    for k in 1..n {
        value = &(&e_minus - &(&z * &value)) / &c_int(k, wp);
    }
    Some(value)
}

/// The value (`numeric`) of part 2.
pub fn integral_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    let wp: usize = prec + 20;
    let z: &Complex = values.last().expect("an argument");
    let real: bool = parts(z).1 == Real::ZERO;
    let positive: bool = real && parts(z).0 > Real::ZERO;
    let value: Complex = match f {
        Function::Expint => {
            let (nr, ni) = parts(&values[0]);
            if ni != Real::ZERO || nr.clone() != nr.floor() || nr < Real::ZERO {
                return None;
            }
            let n: i64 = nr.to_int().value().try_into().ok()?;
            let v: Complex = expint_value(n.min(10_000), z, wp)?;
            return Some(finish(&v, prec, positive));
        }
        Function::Ei => {
            let v: Complex = ei_value(z, wp)?;
            return Some(finish(&v, prec, real));
        }
        Function::Si | Function::Shi => {
            let w: usize = wp + bits_for(z, 1.0);
            trig_series(&raise(z, w), w, f == Function::Si, true)
        }
        Function::Ci | Function::Chi => {
            if is_zero_value(z) {
                return None;
            }
            let w: usize = wp + bits_for(z, 1.0);
            let zz: Complex = raise(z, w);
            let series: Complex = trig_series(&zz, w, f == Function::Ci, false);
            let v: Complex = &(&euler(w) + &ln(&zz)) + &series;
            return Some(finish(&v, prec, positive));
        }
        Function::LogIntegral | Function::OffsetLogIntegral => {
            let one: Complex = c_int(1, wp + 20);
            if is_zero_value(&(z - &one)) {
                return None;
            }
            let mut v: Complex = if is_zero_value(z) {
                c_int(0, wp)
            } else {
                ei_value(&raise(z, wp + 20).ln(), wp)?
            };
            if f == Function::OffsetLogIntegral {
                v = &v - &ei_value(&c_int(2, wp + 20).ln(), wp)?;
            }
            return Some(finish(&v, prec, positive));
        }
        _ => {
            // fresnels: sum (-1)**n (pi/2)**(2n+1) z**(4n+3)/((2n+1)!(4n+3))
            // fresnelc: sum (-1)**n (pi/2)**(2n) z**(4n+1)/((2n)!(4n+1))
            let size: f64 = modulus(z);
            if size > 60.0 {
                return None;
            }
            let w: usize = wp + (size * size * std::f64::consts::FRAC_PI_2 * 1.443) as usize + 20;
            let zz: Complex = raise(z, w);
            let a: Complex = &(&complex_real(Real::pi(w), w) / &c_int(2, w)) * &(&zz * &zz);
            // t_m = a**m/m!, the terms use the odd m (S) or the even m (C)
            let s: bool = f == Function::FresnelS;
            let mut m: i64 = if s { 1 } else { 0 };
            let mut power: Complex = if s { a.clone() } else { c_int(1, w) };
            let mut total: Complex = c_int(0, w);
            let mut sign: bool = true;
            loop {
                let t: Complex = &power / &c_int(2 * m + 1, w);
                total = if sign { &total + &t } else { &total - &t };
                let ts: f64 = modulus(&t);
                if ts == 0.0 || (m > 4 && ts.log2() < -((w + 10) as f64)) || m > 400_000 {
                    break;
                }
                power = &(&(&power * &a) * &a) / &c_int((m + 1) * (m + 2), w);
                m += 2;
                sign = !sign;
            }
            &total * &zz
        }
    };
    Some(finish(&value, prec, real))
}

/// `DefinedFunction._eval_conjugate`: the node for a real value, its
/// negative for an imaginary one.
fn default_conjugate(e: &Basic) -> PolyResult<Option<Basic>> {
    if is_true(e, Predicate::extended_real) {
        return Ok(Some(e.clone()));
    }
    if is_true(e, Predicate::imaginary) {
        return Ok(Some(minus(e)?));
    }
    Ok(None)
}
