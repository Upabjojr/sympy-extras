//! The `expand(func=True)` rules of the special functions (SymPy's
//! `_eval_expand_func`), ported from `sympy_extras.functions._expand_func`
//! and `sympy_extras.functions.special._expand_func_bessel`; a rule
//! returns `None` where SymPy's rule does not apply (the node is left as it
//! is). Not ported (an error): the rewrites of `erfc`, `erfi`, `erf2` and
//! `expint` in terms of `erf` and `Ei`, the sums of `loggamma` and
//! `harmonic` at rationals, `lerchphi` at a rational `a` (`exp_polar`),
//! `hyperexpand` (`hyper` but `2F1(1)`, `meijerg`).

// PORTED FROM SYMPY: the _eval_expand_func methods of sympy/functions/special/gamma_functions.py
// (gamma, polygamma, loggamma, digamma, trigamma), beta_functions.py (beta), zeta_functions.py (zeta,
// polylog, lerchphi), sympy/functions/combinatorial/factorials.py (binomial), numbers.py (harmonic),
// sympy/functions/special/bessel.py (BesselBase, SphericalBesselBase, _jn, _yn),
// sympy/functions/special/spherical_harmonics.py (Ynm), sympy/functions/special/hyper.py (hyper)
// and sympy/core/numbers.py (GoldenRatio, TribonacciConstant); sympy/polys/orthopolys.py
// (dup_spherical_bessel_fn, dup_spherical_bessel_fn_minus) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use crate::core::add::{add, raw_add, sub};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::function::{diff, expand_method, expand_mul, expand_with, rebuild_evaluated, Hints};
use crate::core::mul::{div, mul, neg, raw_mul};
use crate::core::power::pow;
use crate::core::subs::subs;
use crate::error::{not_ported, PolyResult};
use crate::functions::apply_function;

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn one() -> Basic {
    Basic::one()
}

fn minus_one() -> Basic {
    Basic::negative_one()
}

fn q(p: i64, q: i64) -> PolyResult<Basic> {
    Basic::rational(p.into(), q.into())
}

fn inv(e: &Basic) -> PolyResult<Basic> {
    pow(e, &minus_one())
}

/// The binary product of Python's `a * b`.
fn m2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), b.clone()])
}

/// The binary sum of Python's `a + b`.
fn a2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), b.clone()])
}

fn call(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    apply_function(f, args)
}

fn sqrt(x: &Basic) -> PolyResult<Basic> {
    pow(x, &Basic::half())
}

/// The value of an `Integer` node.
fn to_int(e: &Basic) -> Option<i64> {
    match e.head() {
        Head::Rational {
            integer_class: true,
            value,
        } => i64::try_from(value.numerator()).ok(),
        _ => None,
    }
}

/// The numerator and denominator of a `Rational` node (an `Integer` too).
fn to_rat(e: &Basic) -> Option<(i64, i64)> {
    match e.head() {
        Head::Rational { value, .. } => Some((
            i64::try_from(value.numerator()).ok()?,
            i64::try_from(value.denominator()).ok()?,
        )),
        _ => None,
    }
}

fn is_number(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational { .. } | Head::Infinity | Head::NegativeInfinity | Head::NaN
    )
}

fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b) && !crate::sets::sets::is_set(b) && !matches!(b.head(), Head::Tuple)
}

fn expr_args(node: &Basic) -> Vec<Basic> {
    node.args().iter().filter(|a| is_expr(a)).cloned().collect()
}

type Rule = fn(&Basic) -> PolyResult<Option<Basic>>;

// PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma._eval_expand_func
fn gamma_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::factorials::rf;
    let a: &Basic = &node.args()[0];
    if let Some((p, qq)) = to_rat(a) {
        if p.abs() > qq {
            let n: i64 = p.div_euclid(qq);
            let r: Basic = q(p - n * qq, qq)?;
            return Ok(Some(mul(&[
                call(Function::Gamma, std::slice::from_ref(&r))?,
                rf(&r, &int(n))?,
            ])?));
        }
    }
    if a.is_add() {
        let (mut coeff, mut tail): (Basic, Vec<Basic>) = if a.args()[0].is_number_class() {
            (a.args()[0].clone(), a.args()[1..].to_vec())
        } else {
            (Basic::zero(), a.args().to_vec())
        };
        if let Some((p, qq)) = to_rat(&coeff) {
            if p != 0 && qq != 1 {
                let intpart: Basic = crate::functions::elementary::integers::floor(&coeff)?;
                tail.insert(0, add(&[coeff.clone(), neg(&intpart)?])?);
                if is_number(&intpart) {
                    coeff = intpart;
                }
            }
        }
        let rest: Basic = if tail.len() == 1 {
            tail[0].clone()
        } else {
            raw_add(tail)
        };
        return Ok(Some(mul(&[
            call(Function::Gamma, std::slice::from_ref(&rest))?,
            rf(&rest, &coeff)?,
        ])?));
    }
    Ok(Some(call(Function::Gamma, std::slice::from_ref(a))?))
}

// PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:polygamma._eval_expand_func
fn polygamma_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::factorials::factorial;
    use crate::functions::elementary::exponential::log;
    use crate::functions::elementary::trigonometric::{cos, cot, sin};
    let (n, z) = (&node.args()[0], &node.args()[1]);
    let polygamma = |a: &Basic, b: &Basic| call(Function::PolyGamma, &[a.clone(), b.clone()]);
    let ni: Option<i64> = to_int(n);
    if let Some(nv) = ni.filter(|v| *v >= 0) {
        if z.is_add() {
            let coeff: &Basic = &z.args()[0];
            if let Some(c) = to_int(coeff) {
                let e: Basic = neg(&add(&[n.clone(), one()])?)?;
                let tail: Basic = if c > 0 {
                    let terms: Vec<Basic> = (1..=c)
                        .map(|i| pow(&add(&[z.clone(), int(-i)])?, &e))
                        .collect::<PolyResult<_>>()?;
                    add(&terms)?
                } else {
                    let terms: Vec<Basic> = (0..-c)
                        .map(|i| pow(&add(&[z.clone(), int(i)])?, &e))
                        .collect::<PolyResult<_>>()?;
                    neg(&add(&terms)?)?
                };
                let shifted: Basic = polygamma(n, &add(&[z.clone(), neg(coeff)?])?)?;
                return Ok(Some(add(&[
                    shifted,
                    mul(&[pow(&minus_one(), n)?, factorial(n)?, tail])?,
                ])?));
            }
        } else if z.is_mul() {
            let factors: &[Basic] = z.args();
            let c: &Basic = &factors[0];
            let rest: Basic = if factors.len() == 2 {
                factors[1].clone()
            } else {
                raw_mul(factors[1..].to_vec())
            };
            if let Some(cv) = to_int(c).filter(|v| *v > 0) {
                let mut terms: Vec<Basic> = Vec::new();
                for i in 0..cv {
                    terms.push(polygamma(n, &add(&[rest.clone(), q(i, cv)?])?)?);
                }
                if nv == 0 {
                    return Ok(Some(add(&[mul(&[add(&terms)?, inv(c)?])?, log(c)?])?));
                }
                return Ok(Some(mul(&[add(&terms)?, inv(&pow(c, &int(nv + 1))?)?])?));
            }
        }
    }
    if ni == Some(0) {
        if let Some((p, qq)) = to_rat(z) {
            let pi: Basic = Basic::pi();
            let mut parts: Vec<Basic> = vec![
                neg(&Basic::atom(Head::EulerGamma))?,
                neg(&mul(&[
                    pi.clone(),
                    cot(&mul(&[int(p), pi.clone(), q(1, qq)?])?)?,
                    Basic::half(),
                ])?)?,
                neg(&log(&int(qq))?)?,
            ];
            let mut sum: Vec<Basic> = Vec::new();
            for k in 1..qq {
                let c: Basic = cos(&mul(&[int(2 * k), pi.clone(), int(p), q(1, qq)?])?)?;
                let l: Basic = log(&mul(&[int(2), sin(&mul(&[int(k), pi.clone(), q(1, qq)?])?)?])?)?;
                sum.push(mul(&[c, l])?);
            }
            parts.push(add(&sum)?);
            let part_1: Basic = add(&parts)?;
            if p > 0 {
                let m: Basic = crate::functions::elementary::integers::floor(z)?;
                if let Some(mv) = to_int(&m) {
                    let z0: Basic = add(&[z.clone(), neg(&m)?])?;
                    let terms: Vec<Basic> = (0..mv)
                        .map(|k| inv(&add(&[z0.clone(), int(k)])?))
                        .collect::<PolyResult<_>>()?;
                    return Ok(Some(add(&[part_1, add(&terms)?])?));
                }
            } else if p < 0 {
                let m: Basic = crate::functions::elementary::integers::floor(&add(&[one(), neg(z)?])?)?;
                if let Some(mv) = to_int(&m) {
                    let z0: Basic = add(&[z.clone(), m.clone()])?;
                    let terms: Vec<Basic> = (0..mv)
                        .map(|k| inv(&add(&[z0.clone(), int(-1 - k)])?))
                        .collect::<PolyResult<_>>()?;
                    return Ok(Some(add(&[part_1, neg(&add(&terms)?)?])?));
                }
            }
        }
    }
    if ni == Some(-1) {
        let half_log: Basic = mul(&[log(&mul(&[int(2), Basic::pi()])?)?, Basic::half()])?;
        return Ok(Some(add(&[
            call(Function::LogGamma, std::slice::from_ref(z))?,
            neg(&half_log)?,
        ])?));
    }
    if is(n, Predicate::integer) == Some(false) || is(n, Predicate::nonnegative) == Some(false) {
        // a derivative of zeta in its first argument
        let s: Basic = crate::core::Dummy::new("s", &[]).0;
        let zeta_sz: Basic = call(Function::Zeta, &[s.clone(), z.clone()])?;
        let dzt: Basic = subs(&diff(&zeta_sz, &[(s.clone(), 1)])?, &s, &add(&[n.clone(), one()])?)?;
        let factor: Basic = add(&[Basic::atom(Head::EulerGamma), call(Function::DiGamma, &[neg(n)?])?])?;
        let zeta_n: Basic = call(Function::Zeta, &[add(&[n.clone(), one()])?, z.clone()])?;
        let top: Basic = add(&[dzt, mul(&[factor, zeta_n])?])?;
        return Ok(Some(mul(&[top, inv(&call(Function::Gamma, &[neg(n)?])?)?])?));
    }
    Ok(Some(polygamma(n, z)?))
}

// PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_expand_func
fn loggamma_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    let z: &Basic = &node.args()[0];
    if let Some((p0, qq)) = to_rat(z) {
        let n: i64 = p0.div_euclid(qq);
        let p: i64 = p0 - n * qq;
        if 0 < p && p < qq {
            if n == 0 {
                return Ok(Some(call(Function::LogGamma, &[q(p, qq)?])?));
            }
            return Err(not_ported("expand_func of loggamma at a rational (a Sum)"));
        }
    }
    Ok(Some(node.clone()))
}

fn expand_func_full(e: &Basic) -> PolyResult<Basic> {
    // the function expand(e, func=True): its other hints are the defaults
    expand_with(
        e,
        true,
        Hints {
            func: true,
            ..Hints::DEFAULT
        },
    )
}

fn digamma_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    let pg: Basic = call(Function::PolyGamma, &[Basic::zero(), node.args()[0].clone()])?;
    Ok(Some(expand_func_full(&pg)?))
}

fn trigamma_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    let pg: Basic = call(Function::PolyGamma, &[one(), node.args()[0].clone()])?;
    Ok(Some(expand_func_full(&pg)?))
}

// PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:beta._eval_expand_func
fn beta_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    let a: Vec<Basic> = expr_args(node);
    let (x, y) = (&a[0], &a[1]);
    let g = |v: &Basic| call(Function::Gamma, std::slice::from_ref(v));
    Ok(Some(mul(&[g(x)?, g(y)?, inv(&g(&add(&[x.clone(), y.clone()])?)?)?])?))
}

// PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial._eval_expand_func
fn binomial_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::factorials::binomial;
    let (n, k0) = (&node.args()[0], &node.args()[1]);
    if is_number(n) {
        return Ok(Some(binomial(n, k0)?));
    }
    let mut k: Basic = k0.clone();
    let difference: Basic = add(&[n.clone(), neg(&k)?])?;
    if to_int(&difference).is_some() {
        k = difference;
    }
    if let Some(kv) = to_int(&k) {
        if kv == 0 {
            return Ok(Some(one()));
        }
        if kv < 0 {
            return Ok(Some(Basic::zero()));
        }
        let mut result: Basic = one();
        let mut fact: crate::numbers::Int = 1.into();
        for i in 1..=kv {
            result = mul(&[result, add(&[n.clone(), neg(&k)?, int(i)])?])?;
            fact *= crate::numbers::Int::from(i);
        }
        return Ok(Some(mul(&[result, inv(&Basic::integer(fact))?])?));
    }
    Ok(Some(binomial(n, &k)?))
}

// PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_expand_func
fn harmonic_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::numbers::harmonic;
    let args: Vec<Basic> = expr_args(node);
    let n: &Basic = &args[0];
    let m: Basic = if args.len() == 2 { args[1].clone() } else { one() };
    if m == one() {
        if n.is_add() {
            let off: &Basic = &n.args()[0];
            let nnew: Basic = add(&[n.clone(), neg(off)?])?;
            if let Some(o) = to_int(off) {
                let mut terms: Vec<Basic> = Vec::new();
                if o > 0 {
                    for i in (1..=o).rev() {
                        terms.push(inv(&add(&[nnew.clone(), int(i)])?)?);
                    }
                } else if o < 0 {
                    for i in ((o + 1)..=0).rev() {
                        terms.push(neg(&inv(&add(&[nnew.clone(), int(i)])?)?)?);
                    }
                }
                if o != 0 {
                    terms.push(harmonic(&nnew, None)?);
                    return Ok(Some(add(&terms)?));
                }
            }
        }
        if let Some((p0, qq)) = to_rat(n) {
            let u: i64 = p0.div_euclid(qq);
            let p: i64 = p0 - u * qq;
            if u >= 0 && 0 < p && p < qq {
                return Err(not_ported("expand_func of harmonic at a rational (a Sum)"));
            }
        }
    }
    Ok(Some(node.clone()))
}

// PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:zeta._eval_expand_func
fn zeta_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::numbers::harmonic;
    use crate::functions::special::zeta_functions::zeta;
    let args: Vec<Basic> = expr_args(node);
    let s: &Basic = &args[0];
    let a: Basic = if args.len() > 1 { args[1].clone() } else { one() };
    if is(&a, Predicate::integer) == Some(true) {
        if is(&a, Predicate::positive) == Some(true) {
            let h: Basic = harmonic(&add(&[a.clone(), minus_one()])?, Some(s))?;
            return Ok(Some(add(&[zeta(s, None)?, neg(&h)?])?));
        }
        if is(&a, Predicate::nonpositive) == Some(true)
            && (is(s, Predicate::integer) == Some(false) || is(s, Predicate::nonpositive) == Some(false))
        {
            return Ok(Some(Basic::atom(Head::NaN)));
        }
    }
    Ok(Some(node.clone()))
}

// PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:polylog._eval_expand_func
fn polylog_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::exponential::log;
    let (s, z) = (&node.args()[0], &node.args()[1]);
    if *s == one() {
        return Ok(Some(neg(&log(&add(&[one(), neg(z)?])?)?)?));
    }
    if let Some(sv) = to_int(s).filter(|v| *v <= 0) {
        let u: Basic = crate::core::Dummy::new("u", &[]).0;
        let mut start: Basic = mul(&[u.clone(), inv(&add(&[one(), neg(&u)?])?)?])?;
        for _ in 0..(-sv) {
            start = mul(&[u.clone(), diff(&start, &[(u.clone(), 1)])?])?;
        }
        return Ok(Some(subs(&expand_mul(&start, true)?, &u, z)?));
    }
    Ok(Some(crate::functions::special::zeta_functions::polylog(s, z)?))
}

fn golden_ratio_rule(_: &Basic) -> PolyResult<Option<Basic>> {
    Ok(Some(add(&[Basic::half(), mul(&[Basic::half(), sqrt(&int(5))?])?])?))
}

fn tribonacci_rule(_: &Basic) -> PolyResult<Option<Basic>> {
    let r33: Basic = sqrt(&int(33))?;
    let third: Basic = q(1, 3)?;
    let three_r: Basic = mul(&[int(3), r33])?;
    let first: Basic = pow(&add(&[int(19), neg(&three_r)?])?, &third)?;
    let second: Basic = pow(&add(&[int(19), three_r])?, &third)?;
    Ok(Some(mul(&[add(&[one(), first, second])?, third])?))
}

// PORTED FROM SYMPY: sympy/polys/orthopolys.py:dup_spherical_bessel_fn and dup_spherical_bessel_fn_minus
/// The coefficients (highest degree first) of SymPy's
/// `spherical_bessel_fn(n, x)`, a polynomial in `1/x`.
fn spherical_fn_coefficients(n: i64) -> Vec<i64> {
    fn lshift(f: &[i64]) -> Vec<i64> {
        if f.is_empty() {
            Vec::new()
        } else {
            let mut v: Vec<i64> = f.to_vec();
            v.push(0);
            v
        }
    }
    fn subtract(f: &[i64], g: &[i64]) -> Vec<i64> {
        let size: usize = f.len().max(g.len());
        let pad = |h: &[i64]| -> Vec<i64> {
            let mut v: Vec<i64> = vec![0; size - h.len()];
            v.extend_from_slice(h);
            v
        };
        let (f, g) = (pad(f), pad(g));
        let mut out: Vec<i64> = f.iter().zip(&g).map(|(a, b)| a - b).collect();
        while out.first() == Some(&0) {
            out.remove(0);
        }
        out
    }
    if n >= 0 {
        if n < 1 {
            return vec![1, 0];
        }
        let (mut m2, mut m1): (Vec<i64>, Vec<i64>) = (vec![1], vec![1, 0]);
        for i in 2..=n {
            let next: Vec<i64> = subtract(&lshift(&m1).iter().map(|c| c * (2 * i - 1)).collect::<Vec<_>>(), &m2);
            m2 = std::mem::replace(&mut m1, next);
        }
        return lshift(&m1);
    }
    let m: i64 = -n;
    let (mut m2, mut m1): (Vec<i64>, Vec<i64>) = (vec![1, 0], Vec::new());
    for i in 2..=m {
        let next: Vec<i64> = subtract(&lshift(&m1).iter().map(|c| c * (3 - 2 * i)).collect::<Vec<_>>(), &m2);
        m2 = std::mem::replace(&mut m1, next);
    }
    m1
}

/// SymPy's `spherical_bessel_fn(n, z)` as an expression.
fn spherical_fn(n: i64, z: &Basic) -> PolyResult<Basic> {
    let coefficients: Vec<i64> = spherical_fn_coefficients(n);
    let degree: i64 = coefficients.len() as i64 - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in coefficients.iter().enumerate() {
        if *c != 0 {
            terms.push(m2(&int(*c), &pow(z, &int(-(degree - i as i64)))?)?);
        }
    }
    add(&terms)
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:_jn
fn jn(n: i64, z: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::trigonometric::{cos, sin};
    let first: Basic = m2(&spherical_fn(n, z)?, &sin(z)?)?;
    let sign: Basic = pow(&minus_one(), &int(n + 1))?;
    let second: Basic = m2(&m2(&sign, &spherical_fn(-n - 1, z)?)?, &cos(z)?)?;
    a2(&first, &second)
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:_yn
fn yn(n: i64, z: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::trigonometric::{cos, sin};
    let sign: Basic = pow(&minus_one(), &int(n + 1))?;
    let first: Basic = m2(&m2(&sign, &spherical_fn(-n - 1, z)?)?, &sin(z)?)?;
    sub(&first, &m2(&spherical_fn(n, z)?, &cos(z)?)?)
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:SphericalBesselBase._eval_expand_func
fn spherical_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    let (nu, z) = (&node.args()[0], &node.args()[1]);
    let Some(n) = to_int(nu) else {
        return Ok(Some(node.clone()));
    };
    let f: Function = match node.head() {
        Head::Function(f) => *f,
        _ => unreachable!("a spherical Bessel function"),
    };
    match f {
        Function::SphericalJ => Ok(Some(jn(n, z)?)),
        Function::SphericalY => Ok(Some(yn(n, z)?)),
        _ => {
            let sign: i64 = if f == Function::SphericalHankel1 { 1 } else { -1 };
            let i_yn: Basic = m2(&m2(&int(sign), &Basic::imaginary_unit())?, &yn(n, z)?)?;
            Ok(Some(expand_method(&a2(&jn(n, z)?, &i_yn)?)?))
        }
    }
}

// PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:lerchphi._eval_expand_func
fn lerchphi_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::factorials::binomial;
    use crate::functions::special::zeta_functions::zeta;
    let (z, s, a) = (&node.args()[0], &node.args()[1], &node.args()[2]);
    if *z == one() {
        return Ok(Some(zeta(s, Some(a))?));
    }
    if let Some(sv) = to_int(s).filter(|v| *v <= 0) {
        let t: Basic = crate::core::Dummy::new("t", &[]).0;
        let m: i64 = -sv;
        let mut start: Basic = div(&one(), &sub(&one(), &t)?)?;
        let mut res: Basic = Basic::zero();
        // the coefficients of (t + a)**m, lowest degree first
        for k in 0..=m {
            let term: Basic = m2(&m2(&binomial(&int(m), &int(k))?, &pow(a, &int(m - k))?)?, &start)?;
            res = a2(&res, &term)?;
            start = m2(&t, &diff(&start, &[(t.clone(), 1)])?)?;
        }
        return Ok(Some(subs(&res, &t, z)?));
    }
    if matches!(a.head(), Head::Rational { .. }) {
        return Err(not_ported("expand_func of lerchphi at a rational a (exp_polar)"));
    }
    let pi_i: Basic = m2(&Basic::pi(), &Basic::imaginary_unit())?;
    let i: Basic = Basic::imaginary_unit();
    let minus_i: Basic = neg(&i)?;
    let exp_ratio: Option<(i64, i64)> = match z.head() {
        Head::Function(Function::Exp) => to_rat(&div(&z.args()[0], &pi_i)?),
        _ => None,
    };
    if exp_ratio.is_some() || *z == minus_one() || *z == i || *z == minus_i {
        let (pp, qq): (i64, i64) = if *z == minus_one() {
            (1, 2)
        } else if *z == i {
            (1, 4)
        } else if *z == minus_i {
            (-1, 4)
        } else {
            let two_pi_i: Basic = m2(&int(2), &pi_i)?;
            match to_rat(&div(&z.args()[0], &two_pi_i)?) {
                Some(r) => r,
                None => return Ok(Some(node.clone())),
            }
        };
        let mut terms: Vec<Basic> = Vec::new();
        for k in 0..qq {
            // exp(2*pi*I*k*pp/qq)/qq**s*zeta(s, (k + a)/qq), operator by operator
            let mut phase: Basic = m2(&m2(&m2(&int(2), &Basic::pi())?, &Basic::imaginary_unit())?, &int(k))?;
            phase = div(&m2(&phase, &int(pp))?, &int(qq))?;
            let exp_part: Basic = crate::functions::elementary::exponential::exp(&phase)?;
            let shifted: Basic = div(&a2(&int(k), a)?, &int(qq))?;
            terms.push(m2(&div(&exp_part, &pow(&int(qq), s)?)?, &zeta(s, Some(&shifted))?)?);
        }
        return Ok(Some(add(&terms)?));
    }
    Ok(Some(node.clone()))
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:BesselBase._eval_expand_func
/// The recurrence down to an order in `[-1, 1]`, with the signs of the
/// class (SymPy's `_a` and `_b`); the node itself when the order is not
/// real or already small.
fn bessel_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    Ok(Some(bessel_expand(node)?))
}

fn bessel_expand(node: &Basic) -> PolyResult<Basic> {
    let (nu, z) = (&node.args()[0], &node.args()[1]);
    let (a, b): (i64, i64) = match node.head() {
        Head::Function(Function::BesselI) => (-1, 1),
        Head::Function(Function::BesselK) => (1, -1),
        _ => (1, 1),
    };
    let at = |order: Basic| -> PolyResult<Basic> {
        let g: Basic = rebuild_evaluated(node, vec![order, z.clone()])?;
        if std::mem::discriminant(g.head()) == std::mem::discriminant(node.head())
            && g.class_name() == node.class_name()
        {
            return bessel_expand(&g);
        }
        Ok(g)
    };
    if is(nu, Predicate::real) == Some(true) {
        if is(&sub(nu, &one())?, Predicate::positive) == Some(true) {
            let first: Basic = m2(&int(-a * b), &at(sub(nu, &int(2))?)?)?;
            let second: Basic = m2(&m2(&int(2 * a), &sub(nu, &one())?)?, &at(sub(nu, &one())?)?)?;
            return a2(&first, &div(&second, z)?);
        }
        if is(&a2(nu, &one())?, Predicate::negative) == Some(true) {
            let first: Basic = m2(&m2(&int(2 * b), &a2(nu, &one())?)?, &at(a2(nu, &one())?)?)?;
            let second: Basic = m2(&int(-a * b), &at(a2(nu, &int(2))?)?)?;
            return a2(&div(&first, z)?, &second);
        }
    }
    Ok(node.clone())
}

// PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Ynm._eval_expand_func
/// `Ynm(n, m, theta, phi)` in terms of the associated Legendre function of
/// `cos(theta)`, `sin(theta)` for `sqrt(1 - cos(theta)**2)`.
pub(crate) fn ynm_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::factorials::factorial;
    use crate::functions::elementary::exponential::exp;
    use crate::functions::elementary::trigonometric::{cos, sin};
    let args: Vec<Basic> = expr_args(node);
    let (n, m, theta, phi) = (&args[0], &args[1], &args[2], &args[3]);
    let ratio: Basic = div(&a2(&m2(&int(2), n)?, &one())?, &m2(&int(4), &Basic::pi())?)?;
    let ratio: Basic = div(&m2(&ratio, &factorial(&sub(n, m)?)?)?, &factorial(&a2(n, m)?)?)?;
    let phase: Basic = exp(&m2(&m2(&Basic::imaginary_unit(), m)?, phi)?)?;
    let legendre: Basic = crate::functions::special::polynomials::assoc_legendre(n, m, &cos(theta)?)?;
    let rv: Basic = m2(&m2(&sqrt(&ratio)?, &phase)?, &legendre)?;
    let c2: Basic = pow(&cos(theta)?, &int(2))?;
    let old: Basic = sqrt(&a2(&neg(&c2)?, &one())?)?;
    Ok(Some(subs(&rv, &old, &sin(theta)?)?))
}

/// SymPy's `arg.match(c*(d*z**n)**m)` of the argument (`z` its only
/// symbol), `(c, d, n, m, z)` when it matches with `3*m` an integer
/// (`_airy_match`).
fn airy_match(node: &Basic) -> PolyResult<Option<[Basic; 5]>> {
    use crate::core::Wild;
    let arg: &Basic = &node.args()[0];
    let symbols = arg.free_symbols();
    if symbols.len() != 1 {
        return Ok(None);
    }
    let z: Basic = symbols.into_iter().next().expect("a symbol");
    if !z.is_symbol() {
        return Ok(None);
    }
    let wild = |name: &str| Wild::new(name, vec![z.clone()], Vec::new(), &[]).0;
    let (c, d, m, n) = (wild("c"), wild("d"), wild("m"), wild("n"));
    let pattern: Basic = m2(&c, &pow(&m2(&d, &pow(&z, &n)?)?, &m)?)?;
    let Some(found) = crate::core::matching::match_pattern(arg, &pattern, &[], false)? else {
        return Ok(None);
    };
    let get = |w: &Basic| found.iter().find(|(k, _)| k == w).map(|(_, v)| v.clone());
    let (Some(cv), Some(dv), Some(nv), Some(mv)) = (get(&c), get(&d), get(&n), get(&m)) else {
        return Ok(None);
    };
    if !(is_expr(&cv) && is_expr(&dv) && is_expr(&nv) && is_expr(&mv)) {
        return Ok(None);
    }
    if is(&m2(&int(3), &mv)?, Predicate::integer) != Some(true) {
        return Ok(None);
    }
    Ok(Some([cv, dv, nv, mv, z]))
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyai._eval_expand_func, airybi._eval_expand_func,
// airyaiprime._eval_expand_func, airybiprime._eval_expand_func
fn airy_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    let Some([c, d, n, m, z]) = airy_match(node)? else {
        return Ok(None);
    };
    let f: Function = match node.head() {
        Head::Function(f) => *f,
        _ => unreachable!("an Airy function"),
    };
    let half: Basic = Basic::half();
    let r3: Basic = sqrt(&int(3))?;
    let mn: Basic = m2(&m, &n)?;
    let inner: Basic = pow(&m2(&d, &pow(&z, &n)?)?, &m)?;
    let outer: Basic = m2(&pow(&d, &m)?, &pow(&z, &mn)?)?;
    let newarg: Basic = m2(&m2(&c, &pow(&d, &m)?)?, &pow(&z, &mn)?)?;
    let plus = |pf: &Basic| a2(pf, &one());
    let minus = |pf: &Basic| sub(pf, &one());
    let result: Basic = match f {
        Function::AiryAi | Function::AiryBi => {
            let pf: Basic = div(&inner, &outer)?;
            let ai: Basic = call(Function::AiryAi, std::slice::from_ref(&newarg))?;
            let bi: Basic = call(Function::AiryBi, std::slice::from_ref(&newarg))?;
            if f == Function::AiryAi {
                let first: Basic = m2(&plus(&pf)?, &ai)?;
                let second: Basic = m2(&div(&minus(&pf)?, &r3)?, &bi)?;
                m2(&half, &sub(&first, &second)?)?
            } else {
                let first: Basic = m2(&m2(&r3, &sub(&one(), &pf)?)?, &ai)?;
                let second: Basic = m2(&a2(&one(), &pf)?, &bi)?;
                m2(&half, &a2(&first, &second)?)?
            }
        }
        _ => {
            let pf: Basic = div(&outer, &inner)?;
            let ai: Basic = call(Function::AiryAiPrime, std::slice::from_ref(&newarg))?;
            let bi: Basic = call(Function::AiryBiPrime, std::slice::from_ref(&newarg))?;
            if f == Function::AiryAiPrime {
                let first: Basic = m2(&plus(&pf)?, &ai)?;
                let second: Basic = m2(&div(&minus(&pf)?, &r3)?, &bi)?;
                m2(&half, &a2(&first, &second)?)?
            } else {
                let first: Basic = m2(&m2(&r3, &minus(&pf)?)?, &ai)?;
                let second: Basic = m2(&plus(&pf)?, &bi)?;
                m2(&half, &a2(&first, &second)?)?
            }
        }
    };
    Ok(Some(result))
}

// PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper._eval_expand_func
fn hyper_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    let (ap, bq, z) = (node.args()[0].args(), node.args()[1].args(), &node.args()[2]);
    if ap.len() == 2 && bq.len() == 1 && *z == one() {
        let (a, b, c) = (&ap[0], &ap[1], &bq[0]);
        let g = |v: Basic| call(Function::Gamma, &[v]);
        return Ok(Some(mul(&[
            g(c.clone())?,
            g(add(&[c.clone(), neg(a)?, neg(b)?])?)?,
            inv(&g(add(&[c.clone(), neg(a)?])?)?)?,
            inv(&g(add(&[c.clone(), neg(b)?])?)?)?,
        ])?));
    }
    crate::simplify::hyperexpand::hyperexpand(node, false, crate::simplify::hyperexpand::Rewrite::Default, None)
        .map(Some)
}

/// meijerg._eval_expand_func
fn meijerg_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    crate::simplify::hyperexpand::hyperexpand(node, false, crate::simplify::hyperexpand::Rewrite::Default, None)
        .map(Some)
}

fn not_ported_rule(node: &Basic) -> PolyResult<Option<Basic>> {
    Err(not_ported(&format!("expand_func of {}", node.class_name())))
}

fn rule_for(e: &Basic) -> Option<Rule> {
    use Function as F;
    let rule: Rule = match e.head() {
        Head::Function(f) => match f {
            F::Gamma => gamma_rule,
            F::PolyGamma => polygamma_rule,
            F::LogGamma => loggamma_rule,
            F::DiGamma => digamma_rule,
            F::TriGamma => trigamma_rule,
            F::Beta => beta_rule,
            F::Binomial => binomial_rule,
            F::Harmonic => harmonic_rule,
            F::Zeta => zeta_rule,
            F::PolyLog => polylog_rule,
            F::LerchPhi => lerchphi_rule,
            F::BesselJ | F::BesselY | F::BesselI | F::BesselK | F::Hankel1 | F::Hankel2 => bessel_rule,
            F::SphericalJ | F::SphericalY | F::SphericalHankel1 | F::SphericalHankel2 => spherical_rule,
            F::Ynm => ynm_rule,
            // erf rewrites, Ei rewrites, pattern matching: not ported
            F::Erfc | F::Erfi | F::Erf2 | F::Expint => not_ported_rule,
            F::AiryAi | F::AiryBi | F::AiryAiPrime | F::AiryBiPrime => airy_rule,
            _ => return None,
        },
        Head::GoldenRatio => golden_ratio_rule,
        Head::TribonacciConstant => tribonacci_rule,
        Head::Hyper => hyper_rule,
        Head::MeijerG => meijerg_rule,
        _ => return None,
    };
    Some(rule)
}

// PORTED FROM SYMPY: sympy/core/expr.py:Expr._expand_hint with the _eval_expand_func methods
/// The special functions (their arguments first) expanded by their rules
/// (`_expand_func`).
pub fn expand_func(e: &Basic, deep: bool) -> PolyResult<Basic> {
    let mut e: Basic = e.clone();
    if deep && !e.args().is_empty() && !e.is_atom() {
        let mut args: Vec<Basic> = Vec::new();
        for a in e.args() {
            args.push(expand_func(a, deep)?);
        }
        if args
            .iter()
            .zip(e.args())
            .any(|(new, old)| !new.ptr_eq(old) && new != old)
        {
            e = rebuild_evaluated(&e, args)?;
        }
    }
    if !is_expr(&e) {
        return Ok(e);
    }
    if let Some(rule) = rule_for(&e) {
        if let Some(new) = rule(&e)? {
            if new != e {
                return Ok(new);
            }
        }
    }
    Ok(e)
}
