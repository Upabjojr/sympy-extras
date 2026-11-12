//! The elliptic integrals, ported from
//! `sympy_extras.functions.special.elliptic_integrals`: `elliptic_k(m)`,
//! `elliptic_f(z, m)`, `elliptic_e(m)` and `elliptic_e(z, m)`,
//! `elliptic_pi(n, m)` and `elliptic_pi(n, z, m)` with SymPy's special
//! values, the parity in `z`, the multiples of `pi/2`, the derivatives, the
//! conjugates off the cut `m > 1` and the values (mpmath's: `K` by the
//! complex AGM with principal square roots, the complete `E` by mpmath's
//! finite difference of `K`, the incomplete `F` and `E` by Carlson's `RF`
//! and `RD`).
//!
//! Not ported: the values of `elliptic_pi` (mpmath's `RJ` integrates
//! around the singularities of general arguments).

// PORTED FROM SYMPY: sympy/functions/special/elliptic_integrals.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// The values follow mpmath (BSD license): libhyper.mpc_ellipk, mpc_ellipe,
// libelefun.mpc_agm, functions.elliptic.ellipf, ellipe, RF_calc, RJ_calc.

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, complex_real, complex_sqrt, parts, real_int, Complex, Real};
use crate::core::expr::{is, is_true};
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::PolyResult;
use crate::functions::elementary::trigonometric::{sin, tan};
use crate::numbers::Int;

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

fn sqrt(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::half())
}

/// The elliptic integrals.
pub fn is_elliptic(f: Function) -> bool {
    matches!(
        f,
        Function::EllipticK | Function::EllipticF | Function::EllipticE | Function::EllipticPi
    )
}

/// Whether the arguments fit the head.
pub fn arity_fits(f: Function, n: usize) -> bool {
    match f {
        Function::EllipticK => n == 1,
        Function::EllipticF => n == 2,
        Function::EllipticE => n == 1 || n == 2,
        _ => n == 2 || n == 3,
    }
}

pub fn elliptic_k(m: &Basic) -> PolyResult<Basic> {
    elliptic(Function::EllipticK, std::slice::from_ref(m))
}

pub fn elliptic_f(z: &Basic, m: &Basic) -> PolyResult<Basic> {
    elliptic(Function::EllipticF, &[z.clone(), m.clone()])
}

fn elliptic_e1(m: &Basic) -> PolyResult<Basic> {
    elliptic(Function::EllipticE, std::slice::from_ref(m))
}

fn elliptic_e2(z: &Basic, m: &Basic) -> PolyResult<Basic> {
    elliptic(Function::EllipticE, &[z.clone(), m.clone()])
}

fn elliptic_pi2(n: &Basic, m: &Basic) -> PolyResult<Basic> {
    elliptic(Function::EllipticPi, &[n.clone(), m.clone()])
}

fn elliptic_pi3(n: &Basic, z: &Basic, m: &Basic) -> PolyResult<Basic> {
    elliptic(Function::EllipticPi, &[n.clone(), z.clone(), m.clone()])
}

fn is_real_infinity(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity | Head::NegativeInfinity)
}

fn extended_real_infinite(e: &Basic) -> bool {
    is_true(e, Predicate::extended_real) && is_true(e, Predicate::infinite)
}

/// `2*z/pi` when it is an integer.
fn two_z_over_pi_integer(z: &Basic) -> PolyResult<Option<Basic>> {
    let k: Basic = div(&mul(&[int(2), z.clone()])?, &Basic::pi())?;
    Ok(if is_true(&k, Predicate::integer) { Some(k) } else { None })
}

/// The product evaluated left to right (`_lmul`).
fn lmul(factors: &[Basic]) -> PolyResult<Basic> {
    let mut result: Basic = factors[0].clone();
    for f in &factors[1..] {
        result = mul(&[result, f.clone()])?;
    }
    Ok(result)
}

/// `sqrt(1 - m*sin(z)**2)`.
fn delta(m: &Basic, z: &Basic) -> PolyResult<Basic> {
    sqrt(&sub(&Basic::one(), &mul(&[m.clone(), pow(&sin(z)?, &int(2))?])?)?)
}

/// SymPy's `(m.is_real and (m - 1).is_positive) is False`.
fn off_cut(m: &Basic) -> PolyResult<bool> {
    let first: Option<bool> = is(m, Predicate::real);
    if first == Some(true) {
        return Ok(is(&sub(m, &Basic::one())?, Predicate::positive) == Some(false));
    }
    Ok(first == Some(false))
}

/// The evaluating function of the elliptic integrals.
pub fn elliptic(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    if let Some(value) = eval(f, args)? {
        return Ok(value);
    }
    Ok(node(f, args.to_vec()))
}

fn eval(f: Function, args: &[Basic]) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::sign;
    use crate::functions::elementary::inverse::atanh;
    use crate::functions::special::gamma_functions::gamma;
    let one: Basic = Basic::one();
    let half: Basic = Basic::half();
    let pi: Basic = Basic::pi();
    let i: Basic = Basic::imaginary_unit();
    Ok(Some(match (f, args.len()) {
        (Function::EllipticK, _) => {
            let m: &Basic = &args[0];
            if is_true(m, Predicate::zero) {
                mul(&[pi, half])?
            } else if *m == half {
                div(
                    &mul(&[int(8), pow(&pi, &Basic::rational(Int::from(3), Int::from(2))?)?])?,
                    &pow(&gamma(&Basic::rational(Int::from(-1), Int::from(4))?)?, &int(2))?,
                )?
            } else if *m == one {
                Basic::zoo()
            } else if *m == Basic::negative_one() {
                div(
                    &pow(&gamma(&Basic::rational(Int::from(1), Int::from(4))?)?, &int(2))?,
                    &mul(&[int(4), sqrt(&mul(&[int(2), pi])?)?])?,
                )?
            } else if is_real_infinity(m)
                || matches!(m.head(), Head::ComplexInfinity)
                || *m == mul(&[i.clone(), Basic::oo()])?
                || *m == mul(&[i, Basic::negative_oo()])?
            {
                Basic::zero()
            } else {
                return Ok(None);
            }
        }
        (Function::EllipticF, _) => {
            let (z, m) = (&args[0], &args[1]);
            if is_true(z, Predicate::zero) {
                Basic::zero()
            } else if is_true(m, Predicate::zero) {
                z.clone()
            } else if let Some(k) = two_z_over_pi_integer(z)? {
                mul(&[k, elliptic_k(m)?])?
            } else if is_real_infinity(m) {
                Basic::zero()
            } else if could_extract_minus_sign(z) {
                neg(&elliptic_f(&neg(z)?, m)?)?
            } else {
                return Ok(None);
            }
        }
        (Function::EllipticE, 2) => {
            let (z, m) = (&args[0], &args[1]);
            if is_true(m, Predicate::zero) {
                z.clone()
            } else if is_true(z, Predicate::zero) {
                Basic::zero()
            } else if let Some(k) = two_z_over_pi_integer(z)? {
                mul(&[k, elliptic_e1(m)?])?
            } else if is_real_infinity(m) {
                Basic::zoo()
            } else if could_extract_minus_sign(z) {
                neg(&elliptic_e2(&neg(z)?, m)?)?
            } else {
                return Ok(None);
            }
        }
        (Function::EllipticE, _) => {
            let m: &Basic = &args[0];
            if is_true(m, Predicate::zero) {
                mul(&[pi, half])?
            } else if *m == one {
                one
            } else {
                match m.head() {
                    Head::Infinity => mul(&[i, Basic::oo()])?,
                    Head::NegativeInfinity => Basic::oo(),
                    Head::ComplexInfinity => Basic::zoo(),
                    _ => return Ok(None),
                }
            }
        }
        (_, 3) => {
            let (n, z, m) = (&args[0], &args[1], &args[2]);
            if is_true(n, Predicate::zero) {
                elliptic_f(z, m)?
            } else if *n == one {
                let inner: Basic = sub(&mul(&[delta(m, z)?, tan(z)?])?, &elliptic_e2(z, m)?)?;
                add(&[elliptic_f(z, m)?, div(&inner, &sub(&one, m)?)?])?
            } else if let Some(k) = two_z_over_pi_integer(z)? {
                mul(&[k, elliptic_pi2(n, m)?])?
            } else if is_true(m, Predicate::zero) {
                let root: Basic = sqrt(&sub(n, &one)?)?;
                div(&atanh(&mul(&[root.clone(), tan(z)?])?)?, &root)?
            } else if n == m {
                add(&[
                    sub(&elliptic_f(z, n)?, &elliptic_pi3(&one, z, n)?)?,
                    div(&tan(z)?, &delta(n, z)?)?,
                ])?
            } else if is_real_infinity(n) || is_real_infinity(m) {
                Basic::zero()
            } else if could_extract_minus_sign(z) {
                neg(&elliptic_pi3(n, &neg(z)?, m)?)?
            } else if extended_real_infinite(m) || extended_real_infinite(n) {
                Basic::zero()
            } else {
                return Ok(None);
            }
        }
        _ => {
            let (n, m) = (&args[0], &args[1]);
            if is_true(n, Predicate::zero) {
                elliptic_k(m)?
            } else if *n == one {
                Basic::zoo()
            } else if is_true(m, Predicate::zero) {
                div(&pi, &mul(&[int(2), sqrt(&sub(&one, n)?)?])?)?
            } else if *m == one {
                div(&Basic::negative_oo(), &sign(&sub(n, &one)?)?)?
            } else if n == m {
                div(&elliptic_e1(n)?, &sub(&one, n)?)?
            } else if is_real_infinity(n)
                || is_real_infinity(m)
                || extended_real_infinite(m)
                || extended_real_infinite(n)
            {
                Basic::zero()
            } else {
                return Ok(None);
            }
        }
    }))
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn elliptic_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let one: Basic = Basic::one();
    match (f, args.len()) {
        (Function::EllipticK, _) => {
            let m: &Basic = &args[0];
            let top: Basic = sub(&elliptic_e1(m)?, &mul(&[sub(&one, m)?, elliptic_k(m)?])?)?;
            div(&top, &mul(&[int(2), m.clone(), sub(&one, m)?])?)
        }
        (Function::EllipticF, _) => {
            let (z, m) = (&args[0], &args[1]);
            let fm: Basic = delta(m, z)?;
            if argindex == 1 {
                return div(&one, &fm);
            }
            add(&[
                div(&elliptic_e2(z, m)?, &mul(&[int(2), m.clone(), sub(&one, m)?])?)?,
                neg(&div(&elliptic_f(z, m)?, &mul(&[int(2), m.clone()])?)?)?,
                neg(&div(
                    &sin(&mul(&[int(2), z.clone()])?)?,
                    &lmul(&[int(4), sub(&one, m)?, fm])?,
                )?)?,
            ])
        }
        (Function::EllipticE, 2) => {
            let (z, m) = (&args[0], &args[1]);
            if argindex == 1 {
                return delta(m, z);
            }
            div(
                &sub(&elliptic_e2(z, m)?, &elliptic_f(z, m)?)?,
                &mul(&[int(2), m.clone()])?,
            )
        }
        (Function::EllipticE, _) => {
            let m: &Basic = &args[0];
            div(&sub(&elliptic_e1(m)?, &elliptic_k(m)?)?, &mul(&[int(2), m.clone()])?)
        }
        (_, 3) => {
            let (n, z, m) = (&args[0], &args[1], &args[2]);
            let fm: Basic = delta(m, z)?;
            let fn_: Basic = sub(&one, &mul(&[n.clone(), pow(&sin(z)?, &int(2))?])?)?;
            let sin2z: Basic = sin(&mul(&[int(2), z.clone()])?)?;
            if argindex == 1 {
                let top: Basic = add(&[
                    elliptic_e2(z, m)?,
                    div(&mul(&[sub(m, n)?, elliptic_f(z, m)?])?, n)?,
                    div(&mul(&[sub(&pow(n, &int(2))?, m)?, elliptic_pi3(n, z, m)?])?, n)?,
                    neg(&div(&mul(&[n.clone(), fm, sin2z])?, &mul(&[int(2), fn_])?)?)?,
                ])?;
                return div(&top, &lmul(&[int(2), sub(m, n)?, sub(n, &one)?])?);
            }
            if argindex == 2 {
                return div(&one, &mul(&[fm, fn_])?);
            }
            let top: Basic = add(&[
                div(&elliptic_e2(z, m)?, &sub(m, &one)?)?,
                elliptic_pi3(n, z, m)?,
                neg(&div(&mul(&[m.clone(), sin2z])?, &lmul(&[int(2), sub(m, &one)?, fm])?)?)?,
            ])?;
            div(&top, &mul(&[int(2), sub(n, m)?])?)
        }
        _ => {
            let (n, m) = (&args[0], &args[1]);
            if argindex == 1 {
                let top: Basic = add(&[
                    elliptic_e1(m)?,
                    div(&mul(&[sub(m, n)?, elliptic_k(m)?])?, n)?,
                    div(&mul(&[sub(&pow(n, &int(2))?, m)?, elliptic_pi2(n, m)?])?, n)?,
                ])?;
                return div(&top, &lmul(&[int(2), sub(m, n)?, sub(n, &one)?])?);
            }
            let top: Basic = add(&[div(&elliptic_e1(m)?, &sub(m, &one)?)?, elliptic_pi2(n, m)?])?;
            div(&top, &mul(&[int(2), sub(n, m)?])?)
        }
    }
}

/// The conjugate (`_eval_conjugate`); `None` on the cut (no rule).
pub fn elliptic_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::conjugate;
    let args: &[Basic] = e.args();
    let allowed: bool = match (f, args.len()) {
        (Function::EllipticPi, 3) => off_cut(&args[0])? && off_cut(&args[2])?,
        (Function::EllipticPi, _) => true,
        _ => off_cut(args.last().expect("an argument"))?,
    };
    if !allowed {
        return Ok(None);
    }
    let mut conjugated: Vec<Basic> = Vec::new();
    for a in args {
        conjugated.push(conjugate(a)?);
    }
    Ok(Some(elliptic(f, &conjugated)?))
}

/// The facts (`_eval_facts`).
pub fn elliptic_facts(f: Function, args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    let zero: bool = match f {
        Function::EllipticK => is_true(&args[0], Predicate::infinite),
        Function::EllipticF => is_true(&args[0], Predicate::zero) || extended_real_infinite(&args[1]),
        _ => false,
    };
    if zero {
        facts.set(Predicate::zero, true);
    }
    facts
}

// ---------------------------------------------------------------------
// the values (mpmath's algorithms)

fn c_int(n: i64, wp: usize) -> Complex {
    complex_real(real_int(&Int::from(n), wp), wp)
}

fn zero_real(wp: usize) -> Real {
    Real::ZERO.with_precision(wp).value()
}

fn raise(z: &Complex, wp: usize) -> Complex {
    let (a, b) = parts(z);
    complex(a.with_precision(wp).value(), b.with_precision(wp).value())
}

fn is_zero(z: &Complex) -> bool {
    let (a, b) = parts(z);
    a == Real::ZERO && b == Real::ZERO
}

fn modulus(z: &Complex) -> f64 {
    let (a, b) = parts(z);
    a.to_f64().value().hypot(b.to_f64().value())
}

/// mpmath's `mag`: about `log2|z| + 1`.
fn mag(z: &Complex) -> i64 {
    let m: f64 = modulus(z);
    if m == 0.0 {
        return i64::MIN / 4;
    }
    m.log2().floor() as i64 + 1
}

fn pi_complex(wp: usize) -> Complex {
    complex_real(Real::pi(wp), wp)
}

/// `cos` and `sin` by `exp` (real for a real argument).
fn cos_sin(z: &Complex, wp: usize) -> (Complex, Complex) {
    let i: Complex = complex(zero_real(wp), real_int(&Int::ONE, wp));
    let iz: Complex = &i * z;
    let (a, b) = (iz.exp(), (-&iz).exp());
    let c: Complex = &(&a + &b) / &c_int(2, wp);
    let s: Complex = &(&a - &b) / &(&c_int(2, wp) * &i);
    if parts(z).1 == Real::ZERO {
        return (complex(parts(&c).0, zero_real(wp)), complex(parts(&s).0, zero_real(wp)));
    }
    (c, s)
}

/// mpmath's complex AGM `agm(1, a)`: principal square roots.
fn agm1(a: &Complex, wp: usize) -> Option<Complex> {
    if is_zero(a) {
        return None;
    }
    let one: Complex = c_int(1, wp);
    if -a == one {
        return None;
    }
    let mut x: Complex = one;
    let mut y: Complex = a.clone();
    let eps: f64 = -((wp as f64) - 10.0);
    for _ in 0..(4 * wp + 100) {
        let x1: Complex = &(&x + &y) / &c_int(2, wp);
        let y1: Complex = complex_sqrt(&(&x * &y), wp);
        x = x1;
        y = y1;
        let size: f64 = modulus(&x).max(modulus(&y));
        let err: f64 = modulus(&(&x - &y));
        if size == 0.0 || err == 0.0 || err.log2() < eps + size.log2() {
            return Some(x);
        }
    }
    None
}

/// `K(m) = pi/(2*agm(1, sqrt(1 - m)))`; `None` at `m = 1`.
fn ellipk(m: &Complex, wp: usize) -> Option<Complex> {
    let m: Complex = raise(m, wp);
    let one: Complex = c_int(1, wp);
    if m == one {
        return None;
    }
    let a: Complex = complex_sqrt(&(&one - &m), wp);
    let v: Complex = agm1(&a, wp)?;
    if is_zero(&v) {
        return None;
    }
    Some(&(&pi_complex(wp) / &v) / &c_int(2, wp))
}

/// mpmath's complete `E(m) = (1 - m)*(K(m) + 2*m*K'(m))`, `K'` by a finite
/// difference (backward on the real line up to 1, forward elsewhere).
fn ellipe(m: &Complex, wp: usize) -> Option<Complex> {
    let m: Complex = raise(m, wp);
    let (re, im) = parts(&m);
    let one: Complex = c_int(1, wp);
    if is_zero(&m) {
        return Some(&pi_complex(wp) / &c_int(2, wp));
    }
    if m == one {
        return Some(one);
    }
    let magnitude: i64 = mag(&m);
    let real_line: bool = im == Real::ZERO && re <= Real::ONE;
    if real_line && magnitude < -(wp as i64) {
        return Some(&pi_complex(wp) / &c_int(2, wp));
    }
    let p: i64 = magnitude.max(0) - wp as i64;
    let h: Complex = complex_real(
        Real::ONE.with_precision(2 * wp).value() * Real::from(2).powi(p.into()),
        2 * wp,
    );
    let k: Complex = ellipk(&m, 2 * wp)?;
    let difference: Complex = if real_line {
        &k - &ellipk(&(&raise(&m, 2 * wp) - &h), 2 * wp)?
    } else {
        &ellipk(&(&raise(&m, 2 * wp) + &h), 2 * wp)? - &k
    };
    let kdiff: Complex = &difference / &h;
    let t: Complex = &one - &m;
    let b: Complex = &(&kdiff * &m) * &c_int(2, wp);
    Some(&t * &(&k + &b))
}

/// Carlson's `RF(x, y, z)` (mpmath's `RF_calc`).
fn rf(x: &Complex, y: &Complex, z: &Complex, wp: usize) -> Option<Complex> {
    if y == z {
        return rc(x, y, wp);
    }
    if x == z {
        return rc(y, x, wp);
    }
    if x == y {
        return rc(z, x, wp);
    }
    let quarter: Complex = &c_int(1, wp) / &c_int(4, wp);
    let a0: Complex = &(&(x + y) + z) / &c_int(3, wp);
    let spread: f64 = modulus(&(&a0 - x)).max(modulus(&(&a0 - y))).max(modulus(&(&a0 - z)));
    // Q = (3r)**(-1/6)*spread with r = 2**(-wp + 10)
    let log_q: f64 = ((wp as f64) - 10.0 - 3f64.log2()) / 6.0 + spread.max(f64::MIN_POSITIVE).log2();
    let (mut xm, mut ym, mut zm) = (x.clone(), y.clone(), z.clone());
    let mut am: Complex = a0.clone();
    let mut pow4: Complex = c_int(1, wp);
    let mut log_pow4: f64 = 0.0;
    for _ in 0..(wp + 100) {
        let (xs, ys, zs) = (complex_sqrt(&xm, wp), complex_sqrt(&ym, wp), complex_sqrt(&zm, wp));
        let lm: Complex = &(&(&xs * &ys) + &(&xs * &zs)) + &(&ys * &zs);
        let am1: Complex = &(&am + &lm) * &quarter;
        xm = &(&xm + &lm) * &quarter;
        ym = &(&ym + &lm) * &quarter;
        zm = &(&zm + &lm) * &quarter;
        if spread == 0.0 || log_pow4 + log_q < modulus(&am).log2() {
            break;
        }
        am = am1;
        pow4 = &pow4 * &quarter;
        log_pow4 -= 2.0;
    }
    let t: Complex = &pow4 / &am;
    let xx: Complex = &(&a0 - x) * &t;
    let yy: Complex = &(&a0 - y) * &t;
    let zz: Complex = -&(&xx + &yy);
    let e2: Complex = &(&xx * &yy) - &(&zz * &zz);
    let e3: Complex = &(&xx * &yy) * &zz;
    let series: Complex = &(&(&(&(&c_int(9240, wp) - &(&c_int(924, wp) * &e2)) + &(&c_int(385, wp) * &(&e2 * &e2)))
        + &(&c_int(660, wp) * &e3))
        - &(&c_int(630, wp) * &(&e2 * &e3)))
        / &c_int(9240, wp);
    Some(&series / &complex_sqrt(&am, wp))
}

/// Carlson's `RC(x, y)` by its duplication (the principal value for a
/// negative real `y`, as mpmath's `RC_calc`).
fn rc(x: &Complex, y: &Complex, wp: usize) -> Option<Complex> {
    if is_zero(y) {
        return None;
    }
    let (yr, yi) = parts(y);
    if yi == Real::ZERO && yr < Real::ZERO {
        let factor: Complex = complex_sqrt(&(x / &(x - y)), wp);
        return Some(&factor * &rc(&(x - y), &(-y), wp)?);
    }
    if x == y {
        return Some(&c_int(1, wp) / &complex_sqrt(x, wp));
    }
    if is_zero(x) {
        return Some(&(&pi_complex(wp) / &complex_sqrt(y, wp)) / &c_int(2, wp));
    }
    let quarter: Complex = &c_int(1, wp) / &c_int(4, wp);
    let a0: Complex = &(x + &(y * &c_int(2, wp))) / &c_int(3, wp);
    let spread: f64 = modulus(&(&a0 - x));
    let log_q: f64 = ((wp as f64) - 10.0 - 3f64.log2()) / 8.0 + spread.max(f64::MIN_POSITIVE).log2();
    let (mut xm, mut ym) = (x.clone(), y.clone());
    let mut am: Complex = a0.clone();
    let mut log_pow4: f64 = 0.0;
    let mut pow4: Complex = c_int(1, wp);
    for _ in 0..(wp + 100) {
        if spread == 0.0 || log_pow4 + log_q < modulus(&am).log2() {
            break;
        }
        let lm: Complex = &(&(&complex_sqrt(&xm, wp) * &complex_sqrt(&ym, wp)) * &c_int(2, wp)) + &ym;
        am = &(&am + &lm) * &quarter;
        xm = &(&xm + &lm) * &quarter;
        ym = &(&ym + &lm) * &quarter;
        pow4 = &pow4 * &quarter;
        log_pow4 -= 2.0;
    }
    let s: Complex = &(&(y - &a0) * &pow4) / &am;
    let coefficients: [(i64, i64); 7] = [(3, 10), (1, 7), (3, 8), (9, 22), (159, 208), (9, 8), (0, 1)];
    let mut series: Complex = c_int(1, wp);
    let mut power: Complex = &s * &s;
    for (p, q) in coefficients.iter().take(6) {
        series = &series + &(&(&power * &c_int(*p, wp)) / &c_int(*q, wp));
        power = &power * &s;
    }
    Some(&series / &complex_sqrt(&am, wp))
}

/// Carlson's `RD(x, y, z) = RJ(x, y, z, z)` (mpmath's `RJ_calc`; with
/// `p = z` no integration is needed).
fn rd(x: &Complex, y: &Complex, z: &Complex, wp: usize) -> Option<Complex> {
    if is_zero(z) || (is_zero(x) as u8 + is_zero(y) as u8 + is_zero(z) as u8) > 1 {
        return None;
    }
    let p: Complex = z.clone();
    let quarter: Complex = &c_int(1, wp) / &c_int(4, wp);
    let a0: Complex = &(&(&(x + y) + z) + &(&p * &c_int(2, wp))) / &c_int(5, wp);
    let spread: f64 = modulus(&(&a0 - x))
        .max(modulus(&(&a0 - y)))
        .max(modulus(&(&a0 - z)))
        .max(modulus(&(&a0 - &p)));
    // Q = (r/4)**(-1/6)*spread
    let log_q: f64 = ((wp as f64) - 10.0 + 2.0) / 6.0 + spread.max(f64::MIN_POSITIVE).log2();
    let (mut xm, mut ym, mut zm, mut pm) = (x.clone(), y.clone(), z.clone(), p.clone());
    let mut am: Complex = a0.clone();
    let mut pow4: Complex = c_int(1, wp);
    let mut log_pow4: f64 = 0.0;
    let mut sum: Complex = c_int(0, wp);
    for _ in 0..(wp + 100) {
        let (sx, sy, sz, sp) = (
            complex_sqrt(&xm, wp),
            complex_sqrt(&ym, wp),
            complex_sqrt(&zm, wp),
            complex_sqrt(&pm, wp),
        );
        let lm: Complex = &(&(&sx * &sy) + &(&sx * &sz)) + &(&sy * &sz);
        let am1: Complex = &(&am + &lm) * &quarter;
        xm = &(&xm + &lm) * &quarter;
        ym = &(&ym + &lm) * &quarter;
        zm = &(&zm + &lm) * &quarter;
        pm = &(&pm + &lm) * &quarter;
        let dm: Complex = &(&(&sp + &sx) * &(&sp + &sy)) * &(&sp + &sz);
        if spread == 0.0 || log_pow4 + log_q < modulus(&am).log2() {
            break;
        }
        // delta = 0: RC(1, 1 + em) = RC(1, 1) = 1
        sum = &sum + &(&pow4 / &dm);
        pow4 = &pow4 * &quarter;
        log_pow4 -= 2.0;
        am = am1;
    }
    let t: Complex = &pow4 / &am;
    let xx: Complex = &(&a0 - x) * &t;
    let yy: Complex = &(&a0 - y) * &t;
    let zz: Complex = &(&a0 - z) * &t;
    let pp: Complex = &(-&(&(&xx + &yy) + &zz)) / &c_int(2, wp);
    let p2: Complex = &pp * &pp;
    let p3: Complex = &p2 * &pp;
    let xyz: Complex = &(&xx * &yy) * &zz;
    let e2: Complex = &(&(&(&xx * &yy) + &(&xx * &zz)) + &(&yy * &zz)) - &(&p2 * &c_int(3, wp));
    let e3: Complex = &(&xyz + &(&(&e2 * &pp) * &c_int(2, wp))) + &(&p3 * &c_int(4, wp));
    let e4: Complex = &(&(&(&xyz * &c_int(2, wp)) + &(&e2 * &pp)) + &(&p3 * &c_int(3, wp))) * &pp;
    let e5: Complex = &xyz * &p2;
    let series: Complex = &(&(&(&(&(&(&c_int(24024, wp) - &(&e2 * &c_int(5148, wp)))
        + &(&(&e2 * &e2) * &c_int(2457, wp)))
        + &(&e3 * &c_int(4004, wp)))
        - &(&(&e2 * &e3) * &c_int(4158, wp)))
        - &(&e4 * &c_int(3276, wp)))
        + &(&e5 * &c_int(2772, wp)))
        / &c_int(24024, wp);
    let root: Complex = complex_sqrt(&am, wp);
    let v1: Complex = &(&pow4 * &series) / &(&am * &root);
    Some(&v1 + &(&sum * &c_int(6, wp)))
}

/// The amplitude reduced into `|Re z| <= pi/2` (mpmath's `ellipf` and
/// `ellipe`): the shifted `z` and the number `d` of periods.
fn reduce(z: &Complex, wp: usize) -> (Complex, i64) {
    let (x, _) = parts(z);
    let half_pi: Real = Real::pi(wp) / Real::from(2);
    if x <= half_pi && -x.clone() <= half_pi {
        return (z.clone(), 0);
    }
    let ratio: Real = x / Real::pi(wp);
    let d: i64 = ratio.round().to_int().value().try_into().unwrap_or(0);
    (z - &(&pi_complex(wp) * &c_int(d, wp)), d)
}

fn ellipf(phi: &Complex, m: &Complex, wp: usize) -> Option<Complex> {
    if is_zero(m) {
        return Some(phi.clone());
    }
    if is_zero(phi) {
        return Some(c_int(0, wp));
    }
    let wp: usize = wp + mag(&complex_real(parts(phi).0, wp)).max(0) as usize;
    let (phi, m) = (raise(phi, wp), raise(m, wp));
    let (z, d) = reduce(&phi, wp);
    let one: Complex = c_int(1, wp);
    if m == one && d != 0 {
        return None;
    }
    let p: Complex = if d != 0 {
        &ellipk(&m, wp)? * &c_int(2 * d, wp)
    } else {
        c_int(0, wp)
    };
    let (c, s) = cos_sin(&z, wp);
    let value: Complex = &(&s * &rf(&(&c * &c), &(&one - &(&m * &(&s * &s))), &one, wp)?) + &p;
    Some(value)
}

fn ellipe_incomplete(phi: &Complex, m: &Complex, wp: usize) -> Option<Complex> {
    if is_zero(m) {
        return Some(phi.clone());
    }
    if is_zero(phi) {
        return Some(c_int(0, wp));
    }
    let wp: usize = wp + mag(&complex_real(parts(phi).0, wp)).max(0) as usize;
    let (phi, m) = (raise(phi, wp), raise(m, wp));
    let (z, d) = reduce(&phi, wp);
    let one: Complex = c_int(1, wp);
    let p: Complex = if d != 0 {
        &ellipe(&m, wp)? * &c_int(2 * d, wp)
    } else {
        c_int(0, wp)
    };
    let (c, s) = cos_sin(&z, wp);
    let (x, y) = (&c * &c, &one - &(&m * &(&s * &s)));
    let first: Complex = &s * &rf(&x, &y, &one, wp)?;
    let second: Complex = &(&(&(&m * &(&(&s * &s) * &s)) * &rd(&x, &y, &one, wp)?) / &c_int(3, wp)) * &c_int(-1, wp);
    Some(&(&first + &second) + &p)
}

/// The value (`numeric`); `None` for `elliptic_pi`.
pub fn elliptic_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    let wp: usize = prec + 40;
    let value: Complex = match (f, values.len()) {
        (Function::EllipticK, _) => ellipk(&values[0], wp)?,
        (Function::EllipticE, 1) => ellipe(&values[0], wp)?,
        (Function::EllipticF, _) => ellipf(&values[0], &values[1], wp)?,
        (Function::EllipticE, _) => ellipe_incomplete(&values[0], &values[1], wp)?,
        _ => return None,
    };
    let (re, im) = parts(&value);
    Some(complex(
        re.with_precision(prec).value(),
        im.with_precision(prec).value(),
    ))
}
