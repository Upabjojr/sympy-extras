//! Port of `sympy_extras/concrete/_evalf.py` (SymPy's `evalf_sum`,
//! `hypsum`, `check_convergence` and `Sum.euler_maclaurin`): the value of
//! `Sum(f, (i, a, b))` in `numeric_value`. A hypergeometric term summed
//! to infinity in fixed point (geometric or faster convergence) or by
//! mpmath's Richardson extrapolation (`nsum`, polynomial convergence); a
//! short finite range term by term; the rest by Euler–Maclaurin summation
//! with the quadrature of the remaining integral.

#![allow(clippy::mutable_key_type)]

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::evalf::{complex, numeric_value, parts, real_int, Complex, Real};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::integrals::quadrature::log10_abs;
use crate::numbers::{Int, Rat};

/// The most terms of a finite sum added one by one (`_DIRECT_TERMS * prec`
/// or `_MOST_DIRECT`).
const DIRECT_TERMS: i64 = 2;
const MOST_DIRECT: i64 = 4000;

fn real(n: i64, p: usize) -> Real {
    real_int(&Int::from(n), p)
}

fn rint(n: &Int, p: usize) -> Real {
    real_int(n, p)
}

fn at(x: &Real, p: usize) -> Real {
    x.clone().with_precision(p).value()
}

fn is_zero_real(r: &Real) -> bool {
    r.repr().significand().is_zero()
}

fn abs_real(r: &Real) -> Real {
    if *r < Real::ZERO {
        -r.clone()
    } else {
        r.clone()
    }
}

fn two_pow_neg(k: usize, p: usize) -> Real {
    real(1, p) / rint(&(Int::ONE << k), p)
}

fn as_int(e: &Basic) -> Option<i64> {
    match e.head() {
        Head::Rational {
            integer_class: true, ..
        } => i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok(),
        _ => None,
    }
}

fn subs(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Basic> {
    crate::core::subs::subs(e, old, new)
}

fn plus(a: &Basic, n: i64) -> PolyResult<Basic> {
    crate::core::add::add(&[a.clone(), Basic::int(n)])
}

/// The coefficients of the polynomial `e` in `n`, highest degree first,
/// when they are integers (`_integer_coefficients`).
fn integer_coefficients(e: &Basic, n: &Basic) -> Option<Vec<Int>> {
    let p = crate::polys::polytools::Poly::new(e, std::slice::from_ref(n)).ok()?;
    let coeffs: Vec<Basic> = p.all_coeffs().ok()?;
    let mut out: Vec<Int> = Vec::new();
    for c in coeffs {
        if !matches!(
            c.head(),
            Head::Rational {
                integer_class: true,
                ..
            }
        ) {
            return None;
        }
        out.push(crate::numbers::numer(c.as_rational()?));
    }
    Some(out)
}

/// The degree in `n` of a polynomial with numerical coefficients (`_degree`).
fn degree(e: &Basic, n: &Basic) -> Option<i64> {
    let ex: Basic = crate::core::function::expand_default(e).ok()?;
    let mut d: i64 = -1;
    for t in crate::core::expr::terms_of(&ex) {
        let (c, k) = crate::solvers::solvers::as_coeff_exponent(&t, n).ok()?;
        let k: i64 = as_int(&k)?;
        if c.has(n) || !crate::core::assumptions::is_number(&c) || k < 0 {
            return None;
        }
        if is(&c, Predicate::zero) != Some(true) {
            d = d.max(k);
        }
    }
    if d >= 0 {
        Some(d)
    } else {
        None
    }
}

fn real_value(e: &Basic, prec: usize) -> Option<Real> {
    let (re, im) = parts(&numeric_value(e, prec)?);
    if !is_zero_real(&im) {
        return None;
    }
    Some(re)
}

/// The sum over `n >= 0` of a term whose ratio `num/den` converges like a
/// power of `1/n!`, term by term (`_factorial_rate_sum`).
fn factorial_rate_sum(expr: &Basic, num: &Basic, den: &Basic, n: &Basic, prec: usize) -> Option<Real> {
    let (p, q) = (degree(num, n)?, degree(den, n)?);
    if q <= p {
        return None;
    }
    let wp: usize = prec + 20;
    let first: Basic = subs(expr, n, &Basic::zero()).ok()?;
    let term: Real = real_value(&first, wp)?;
    if is_zero_real(&term) {
        return None;
    }
    let mut total: Real = at(&term, wp);
    let mut current: Real = total.clone();
    let bound: Real = real(5, wp) * two_pow_neg(prec, wp);
    let mut k: i64 = 1;
    while abs_real(&current) > bound {
        crate::timeout::check().ok()?;
        let a: Basic = subs(num, n, &Basic::int(k - 1)).ok()?;
        let b: Basic = subs(den, n, &Basic::int(k - 1)).ok()?;
        let (va, vb) = (real_value(&a, wp)?, real_value(&b, wp)?);
        if is_zero_real(&vb) {
            return None;
        }
        current = at(&(&current * &va / &vb), wp);
        total = at(&(&total + &current), wp);
        k += 1;
    }
    Some(at(&total, prec))
}

fn horner(coeffs: &[Int], k: i64) -> Int {
    let mut value: Int = Int::ZERO;
    let kk: Int = Int::from(k);
    for c in coeffs {
        value = value * &kk + c;
    }
    value
}

fn has_natural_root(coeffs: &[Int]) -> bool {
    use dashu_base::Abs;
    let lead: Int = coeffs[0].clone().abs();
    let largest: Int = coeffs[1..].iter().map(|c| c.clone().abs()).max().unwrap_or(Int::ZERO);
    let bound: Int = Int::ONE + largest / lead + Int::ONE;
    let limit: i64 = i64::try_from(&bound).unwrap_or(i64::MAX).min(100_000);
    (0..=limit).any(|k| horner(coeffs, k) == Int::ZERO)
}

/// SymPy's `(h, g, p)` (`check_convergence`).
pub fn check_convergence(numer: &[Int], denom: &[Int]) -> (i64, Option<Rat>, Option<Rat>) {
    let (p, q) = (numer.len() as i64 - 1, denom.len() as i64 - 1);
    let rate: i64 = q - p;
    if rate != 0 {
        return (rate, None, None);
    }
    let constant: Rat = Rat::from_parts(denom[0].clone(), dashu_int::UBig::ONE) / Rat::from(numer[0].clone());
    let one: Rat = Rat::ONE;
    let magnitude: Rat = if constant < Rat::ZERO {
        -constant.clone()
    } else {
        constant.clone()
    };
    if magnitude != one {
        return (rate, Some(constant), None);
    }
    if p == 0 && q == 0 {
        return (rate, Some(constant), Some(Rat::ZERO));
    }
    let pp: Rat = Rat::from(&denom[1] - &numer[1]) / Rat::from(denom[0].clone());
    (rate, Some(constant), Some(pp))
}

fn floor_div(a: &Int, b: &Int) -> Int {
    let q: Int = a / b;
    let r: Int = a - &q * b;
    if r != Int::ZERO && ((r < Int::ZERO) != (*b < Int::ZERO)) {
        q - Int::ONE
    } else {
        q
    }
}

/// mpmath's `nstr(v, n)`: the value to `n` significant digits.
fn nstr(v: &Real, n: usize) -> String {
    if is_zero_real(v) {
        return "0".to_string();
    }
    let e: i64 = log10_abs(v).floor() as i64;
    let exact: Rat = crate::integrals::evalf::exact(v);
    let shift: i64 = n as i64 - 1 - e;
    let ten: Rat = Rat::from(Int::from(10));
    let mut scaled: Rat = exact;
    for _ in 0..shift.abs() {
        scaled = if shift > 0 { scaled * &ten } else { scaled / &ten };
    }
    let rounded: Int = scaled.round();
    format!("{}e{}", rounded, e)
}

/// `richardson(seq)`: the extrapolated limit and the largest weight.
fn richardson(seq: &[Real], p: usize) -> (Real, Real) {
    let sign = |r: Real| -> i8 {
        if is_zero_real(&r) {
            0
        } else if r < Real::ZERO {
            -1
        } else {
            1
        }
    };
    let n0: usize = seq.len();
    let s1: i8 = sign(&seq[n0 - 1] - &seq[n0 - 2]);
    let s2: i8 = sign(&seq[n0 - 2] - &seq[n0 - 3]);
    let seq: Vec<Real> = if s1 != s2 {
        seq.iter().step_by(2).cloned().collect()
    } else {
        seq.to_vec()
    };
    let n: usize = seq.len() / 2 - 1;
    let ni: i64 = n as i64;
    let mut fact: Int = Int::ONE;
    for k in 2..=n {
        fact *= Int::from(k);
    }
    let nn: Int = Int::from(ni).pow(n);
    let signed: Int = if n % 2 == 1 { -nn } else { nn };
    let mut c: Real = at(&(rint(&signed, p) / rint(&fact, p)), p);
    let mut s: Real = real(0, p);
    let mut maxc: Real = real(1, p);
    for k in 0..=n {
        let ki: i64 = k as i64;
        s = at(&(&s + &c * &seq[n + k]), p);
        let ac: Real = abs_real(&c);
        if ac > maxc {
            maxc = ac;
        }
        let up: Real = at(&rint(&Int::from(ki + ni + 1).pow(n), p), p);
        c = at(&(&c * real(ki - ni, p) * up), p);
        let down: Real = at(&rint(&Int::from(ki + ni).pow(n), p), p);
        c = at(&(&c / (real(1 + ki, p) * down)), p);
    }
    (s, maxc)
}

/// mpmath's `nsum(summand, [0, inf], method='richardson')` at `prec`
/// bits (`adaptive_extrapolation` with Richardson's method only).
fn nsum_richardson(summand: &mut dyn FnMut(i64, usize) -> Real, prec: usize) -> Real {
    let tol: Real = two_pow_neg(prec + 9, prec + 20);
    let dps: i64 = ((prec as f64 / std::f64::consts::LOG2_10).round() as i64 - 1).max(1);
    let maxterms: i64 = dps * 10;
    let wp: usize = (prec + 10) * 4;
    let eps_w: Real = two_pow_neg(wp - 1, wp);
    let mut try_richardson: bool = true;
    let mut last_richardson: Real = real(0, wp);
    let mut partial: Vec<Real> = Vec::new();
    let mut index: i64 = 0;
    let mut step: i64 = 0;
    let mut best: Real = real(0, wp);
    loop {
        if index >= maxterms {
            break;
        }
        step += 10;
        let mut psum: Real = partial.last().cloned().unwrap_or_else(|| real(0, wp));
        for k in index..index + step {
            psum = at(&(&psum + summand(k, wp)), wp);
            partial.push(psum.clone());
        }
        index += step;
        best = partial[partial.len() - 1].clone();
        let mut error: Real = abs_real(&(&best - &partial[partial.len() - 2]));
        if error <= tol {
            return at(&best, prec);
        }
        if try_richardson {
            let (value, maxc) = richardson(&partial, wp);
            let richardson_error: Real = abs_real(&(&value - &last_richardson));
            if richardson_error <= tol {
                return at(&value, prec);
            }
            last_richardson = value.clone();
            if &eps_w * &maxc > tol {
                try_richardson = false;
            }
            if richardson_error < error {
                error = richardson_error;
                best = value;
            }
        }
        let _ = error;
    }
    at(&best, prec)
}

/// The sum over `n >= start` of a hypergeometric term at `prec` bits
/// (`hypsum`); `None` when not decided here or when the sum diverges
/// (SymPy's `ValueError`).
pub fn hypsum(expr: &Basic, n: &Basic, start: i64, prec: usize) -> Option<Real> {
    let expr: Basic = if start != 0 {
        subs(expr, n, &plus(n, start).ok()?).ok()?
    } else {
        expr.clone()
    };
    let hs: Basic = crate::concrete::gosper::hypersimp(&expr, n).ok()??;
    let (num, den) = crate::core::expr::as_numer_denom(&hs).ok()?;
    let (Some(numer), Some(denom)) = (integer_coefficients(&num, n), integer_coefficients(&den, n)) else {
        return factorial_rate_sum(&expr, &num, &den, n, prec);
    };
    if numer.is_empty() || denom.is_empty() || numer[0] == Int::ZERO || denom[0] == Int::ZERO {
        return None;
    }
    let (h, g, p) = check_convergence(&numer, &denom);
    if h < 0 {
        return None; // diverges like (n!)^-h
    }
    if has_natural_root(&denom) {
        return None;
    }
    let eterm: Basic = subs(&expr, n, &Basic::zero()).ok()?;
    let value: Rat = match eterm.head() {
        Head::Rational { .. } => eterm.as_rational()?.clone(),
        _ => return None,
    };
    let (tp, tq): (Int, Int) = (crate::numbers::numer(&value), crate::numbers::denom(&value));
    let one: Rat = Rat::ONE;
    let gabs: Option<Rat> = g.as_ref().map(|v| if *v < Rat::ZERO { -v.clone() } else { v.clone() });
    if h > 0 || gabs.as_ref().is_some_and(|v| *v > one) {
        let mut term: Int = floor_div(&(&tp << prec), &tq);
        let mut s: Int = term.clone();
        let mut k: i64 = 1;
        let five: Int = Int::from(5);
        loop {
            let magnitude: Int = if term < Int::ZERO { -term.clone() } else { term.clone() };
            if magnitude <= five {
                break;
            }
            crate::timeout::check().ok()?;
            term *= horner(&numer, k - 1);
            term = floor_div(&term, &horner(&denom, k - 1));
            s += &term;
            k += 1;
        }
        return Some(rint(&s, prec) * two_pow_neg(prec, prec));
    }
    let g: Rat = g?;
    let alt: bool = g < Rat::ZERO;
    if gabs.as_ref().is_some_and(|v| *v < one) {
        return None; // diverges geometrically
    }
    let p: Rat = p?;
    if p < one || (p == one && !alt) {
        return None; // diverges like n^-p
    }
    let mut prec: usize = prec;
    let ndig: usize = ((prec as f64 / std::f64::consts::LOG2_10).round() as i64 - 1).max(1) as usize;
    let mut vold: Option<String> = None;
    loop {
        let prec2: usize = 4 * prec;
        let mut current: Int = floor_div(&(&tp << prec2), &tq);
        let mut summand = |k: i64, wp: usize| -> Real {
            if k != 0 {
                current *= horner(&numer, k - 1);
                current = floor_div(&current, &horner(&denom, k - 1));
            }
            at(&(rint(&current, wp) * two_pow_neg(prec2, wp)), wp)
        };
        let v: Real = nsum_richardson(&mut summand, prec);
        let vf: String = nstr(&v, ndig);
        if vold.as_ref() == Some(&vf) {
            return Some(v);
        }
        crate::timeout::check().ok()?;
        if prec > 1 << 16 {
            return None;
        }
        prec += prec;
        vold = Some(vf);
    }
}

fn fastlog(x: &Real) -> i64 {
    use dashu_base::{BitTest, UnsignedAbs};
    let r = x.repr();
    if r.significand().is_zero() {
        return -1_000_000_000;
    }
    r.exponent() as i64 + r.significand().unsigned_abs().bit_len() as i64
}

/// The value of `Sum(function, (i, a, b))` at `prec` bits (`evalf_sum`).
pub fn evalf_sum(function: &Basic, i: &Basic, a: &Basic, b: &Basic, prec: usize) -> Option<Complex> {
    let zero = |p: usize| complex(real(0, p), real(0, p));
    if is(function, Predicate::zero) == Some(true) {
        return Some(zero(prec));
    }
    let (ai, bi) = (as_int(a), as_int(b));
    if let (true, Some(start)) = (matches!(b.head(), Head::Infinity), ai) {
        if let Some(mut v) = hypsum(function, i, start, prec + 10) {
            if is_zero_real(&v) {
                return None;
            }
            if fastlog(&v) < -10 {
                let delta: i64 = prec as i64 - fastlog(&v);
                v = hypsum(function, i, start, delta.max(1) as usize)?;
            }
            return Some(complex(at(&v, prec), real(0, prec)));
        }
    } else if let (Some(lo), Some(hi)) = (ai, bi) {
        if hi - lo == -1 {
            return Some(zero(prec));
        }
        if 0 <= hi - lo && hi - lo < (DIRECT_TERMS * prec as i64).max(MOST_DIRECT) {
            let mut total: Complex = zero(prec + 10);
            for k in lo..=hi {
                crate::timeout::check().ok()?;
                let term: Basic = subs(function, i, &Basic::int(k)).ok()?;
                let value: Complex = numeric_value(&term, prec + 10)?;
                total = &total + &value;
            }
            return Some(total);
        }
    }
    euler_maclaurin_value(function, i, a, b, prec)
}

fn finite(z: &Complex) -> bool {
    let (re, im) = parts(z);
    re.repr().is_finite() && im.repr().is_finite()
}

fn modulus(z: &Complex) -> Real {
    let (re, im) = parts(z);
    crate::integrals::quadrature::modulus(&(re, im))
}

fn euler_maclaurin_value(function: &Basic, i: &Basic, a: &Basic, b: &Basic, prec: usize) -> Option<Complex> {
    if matches!(a.head(), Head::NegativeInfinity) {
        return None;
    }
    if !monotone_tail(function, i, &plus(a, 2 * prec as i64).ok()?, b) {
        return None;
    }
    let eps: Basic = Basic::from_rational(Rat::from_parts(Int::ONE, dashu_int::UBig::ONE << prec));
    let bound: Real = two_pow_neg(prec, prec + 10);
    let mut s: Basic = Basic::zero();
    for k in 1..5u32 {
        let m: i64 = (1i64 << k) * prec as i64;
        let (sum, err) = euler_maclaurin(function, i, a, b, m, m, &eps, false).ok()??;
        s = sum;
        if matches!(err.head(), Head::NaN) {
            return None;
        }
        let err_value: Complex = value3(&err)?;
        if modulus(&err_value) <= bound {
            break;
        }
    }
    let value: Complex = numeric_value(&s, prec + 10)?;
    if !finite(&value) {
        return None;
    }
    Some(value)
}

/// Whether `|f|` decreases at sample points of `(lo, hi)`, real and of one
/// sign (`_monotone_tail`).
fn monotone_tail(f: &Basic, i: &Basic, lo: &Basic, hi: &Basic) -> bool {
    let t: Basic = crate::core::symbol::Dummy::new("t", &[(Predicate::real, true)]).0;
    let Ok(g) = subs(f, i, &t) else { return false };
    let p: usize = 60;
    let Some(start) = numeric_value(lo, p) else {
        return false;
    };
    let infinite: bool = matches!(hi.head(), Head::Infinity);
    let end: Option<Complex> = if infinite { None } else { numeric_value(hi, p) };
    if end.is_none() && !infinite {
        return false;
    }
    let a0: Real = parts(&start).0;
    let points: Vec<Real> = match &end {
        Some(e) => {
            let b0: Real = parts(e).0;
            if b0 <= a0 {
                return true;
            }
            (0..=200)
                .map(|k| &a0 + (&b0 - &a0) * real(k, p) / real(200, p))
                .collect()
        }
        None => (0..48).map(|j| &a0 + rint(&(Int::ONE << j), p) - real(1, p)).collect(),
    };
    let tiny: Real = real(1, p) / rint(&Int::from(10_000_000_000i64), p);
    let mut values: Vec<Real> = Vec::new();
    for x in points {
        let z: Complex = complex(x, real(0, p));
        let Some(v) = crate::integrals::evalf::value_at(&g, &t, z, p) else {
            return false;
        };
        if !finite(&v) {
            return false;
        }
        let (re, im) = parts(&v);
        if abs_real(&im) > abs_real(&re) * &tiny {
            return false;
        }
        values.push(re);
    }
    let nonzero: Vec<&Real> = values.iter().filter(|v| !is_zero_real(v)).collect();
    if !nonzero.is_empty() && !(nonzero.iter().all(|v| **v > Real::ZERO) || nonzero.iter().all(|v| **v < Real::ZERO)) {
        return false;
    }
    let slack: Real = real(1, p) + real(1, p) / rint(&Int::from(1_000_000_000_000i64), p);
    values.windows(2).all(|w| abs_real(&w[1]) <= abs_real(&w[0]) * &slack)
}

/// SymPy's `e.evalf(3)` (`_value3`): the precision doubled until two values
/// agree to three digits.
fn value3(e: &Basic) -> Option<Complex> {
    let mut prec: usize = 40;
    let mut low: Complex = numeric_value(e, prec)?;
    while prec < 4000 {
        let high: Complex = numeric_value(e, 2 * prec)?;
        let d: Complex = &high - &low;
        let scale: Real = modulus(&high) / rint(&Int::from(10_000), 2 * prec);
        if modulus(&d) <= scale {
            return Some(high);
        }
        low = high;
        prec *= 2;
    }
    Some(low)
}

fn is_less(e: &Basic, eps: &Real) -> Option<bool> {
    let value: Complex = value3(e)?;
    if !finite(&value) {
        return Some(false);
    }
    Some(modulus(&value) < *eps)
}

fn decreasing(f: &Basic, i: &Basic, k: &Basic, term: &Basic) -> bool {
    let Some(here) = value3(term) else { return false };
    for step in [1, 10] {
        let Ok(next) = plus(k, step) else { return false };
        let Ok(at_next) = subs(f, i, &next) else { return false };
        match value3(&at_next) {
            Some(later) if modulus(&later) <= modulus(&here) => {}
            _ => return false,
        }
    }
    true
}

/// `Sum(function, (i, a, b)).euler_maclaurin(m, n, eps, eval_integral)`:
/// `(s, e)`, the approximation and the estimated error; `None` when a
/// comparison with `eps` is not decided (`euler_maclaurin`).
#[allow(clippy::too_many_arguments)]
pub fn euler_maclaurin(
    function: &Basic,
    i: &Basic,
    a: &Basic,
    b: &Basic,
    m: i64,
    n: i64,
    eps: &Basic,
    eval_integral: bool,
) -> PolyResult<Option<(Basic, Basic)>> {
    use crate::core::add::{add, sub};
    use crate::core::mul::{div, mul, neg};
    let eps_value: Option<Complex> = value3(eps);
    let eps_zero: bool = eps_value.as_ref().map_or(true, |v| v.is_zero());
    let eps_mpf: Real = eps_value.as_ref().map(modulus).unwrap_or_else(|| real(0, 60));
    let mut f: Basic = function.clone();
    let (mut a, mut b): (Basic, Basic) = (a.clone(), b.clone());
    let difference: Basic = sub(&a, &b)?;
    if is(&difference, Predicate::extended_positive) == Some(true) {
        if difference == Basic::one() {
            return Ok(Some((Basic::zero(), Basic::zero())));
        }
        let (na, nb) = (plus(&b, 1)?, plus(&a, -1)?);
        a = na;
        b = nb;
        f = neg(&f)?;
    }
    let at_ = |e: &Basic, v: &Basic| subs(e, i, v);
    let mut s: Basic = Basic::zero();
    let mut m: i64 = m;
    if m != 0 {
        if let (Some(lo), Some(hi)) = (as_int(&a), as_int(&b)) {
            m = m.min(hi - lo + 1);
        }
        let polynomial: bool = crate::core::expr::polynomial_like(
            &f,
            std::slice::from_ref(i),
            crate::core::expr::PolynomialKind::Polynomial,
        ) == Some(true);
        if eps_zero || polynomial {
            let mut terms: Vec<Basic> = Vec::new();
            for k in 0..m {
                crate::timeout::check()?;
                terms.push(at_(&f, &plus(&a, k)?)?);
            }
            s = add(&terms)?;
        } else {
            let mut term: Basic = at_(&f, &a)?;
            if term != Basic::zero() {
                let test: Option<bool> = is_less(&term, &eps_mpf);
                if test == Some(true) && decreasing(&f, i, &a, &term) {
                    return Ok(Some((s, crate::functions::elementary::complexes::abs_(&term)?)));
                }
                if test.is_none() {
                    return Ok(Some((term, Basic::zero())));
                }
            }
            s = term;
            for k in 1..m {
                crate::timeout::check()?;
                let point: Basic = plus(&a, k)?;
                term = at_(&f, &point)?;
                if term != Basic::zero() && is_less(&term, &eps_mpf) == Some(true) && decreasing(&f, i, &point, &term) {
                    return Ok(Some((s, crate::functions::elementary::complexes::abs_(&term)?)));
                }
                s = add(&[s, term.clone()])?;
            }
        }
        if plus(&sub(&b, &a)?, 1)? == Basic::int(m) {
            return Ok(Some((s, Basic::zero())));
        }
        a = plus(&a, m)?;
    }
    let x: Basic = crate::core::symbol::Dummy::new("x", &[]).0;
    let integrand: Basic = at_(&f, &x)?;
    let mut integral: Basic = crate::integrals::integral::integral(&integrand, &x, Some((&a, &b)))?;
    if eval_integral {
        integral = crate::core::function::doit(&integral)?;
    }
    s = add(&[s, integral])?;
    let infinite_b: bool = matches!(b.head(), Head::Infinity);
    let fpoint = |e: &Basic| -> PolyResult<(Basic, Basic)> {
        if infinite_b {
            return Ok((at_(e, &a)?, Basic::zero()));
        }
        Ok((at_(e, &a)?, at_(e, &b)?))
    };
    let (fa, fb) = fpoint(&f)?;
    let iterm: Basic = div(&add(&[fa, fb])?, &Basic::int(2))?;
    let mut g: Basic = crate::core::function::derivative(&f, i)?;
    let mut term: Basic = Basic::zero();
    for k in 1..=n + 1 {
        crate::timeout::check()?;
        let (ga, gb) = fpoint(&g)?;
        let coefficient: Basic = div(
            &crate::functions::combinatorial::numbers::bernoulli(&Basic::int(2 * k), None)?,
            &crate::functions::combinatorial::factorials::factorial(&Basic::int(2 * k))?,
        )?;
        term = mul(&[coefficient, sub(&gb, &ga)?])?;
        if k > n {
            break;
        }
        if !eps_zero && term != Basic::zero() {
            if let Some(value) = value3(&term) {
                if modulus(&value) < eps_mpf {
                    break;
                }
            }
        }
        s = add(&[s, term.clone()])?;
        g = crate::core::function::derivative(&crate::core::function::derivative(&g, i)?, i)?;
    }
    Ok(Some((
        add(&[s, iterm])?,
        crate::functions::elementary::complexes::abs_(&term)?,
    )))
}

/// The value of a `Sum` node in `numeric_value` (`Sum._numeric_node`): one
/// limit only.
pub fn sum_value(e: &Basic, prec: usize) -> Option<Complex> {
    let lims = crate::concrete::expr_with_limits::limits(e);
    if lims.len() != 1 {
        return None;
    }
    let (i, a, b) = &lims[0];
    evalf_sum(&e.args()[0], i, a, b, prec)
}
