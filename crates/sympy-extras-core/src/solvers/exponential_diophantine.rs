//! Port of `sympy_extras/solvers/exponential_diophantine.py`: Pillai's
//! equation `r a**x - s b**y = c` and the integer zeros of exponential
//! polynomials.
//!
//! **Pillai's equation.** For multiplicatively dependent bases the
//! equation is a question about powers of one integer; otherwise the
//! linear form in logarithms `x log a - y log b (+ log(r/s))` bounds the
//! exponents by the theorem of Baker and Wüstholz, the bound is reduced by
//! the continued fraction of `log b/log a` (Legendre) or by the lemma of
//! Baker and Davenport (in the form of Dujella and Pethő) for three
//! logarithms, and the remaining exponents are tried in exact integer
//! arithmetic: the list of solutions is complete.
//!
//! The floating-point steps (the logarithms, the partial quotients) run on
//! `dashu-float` instead of `mpmath`, at the binary precision of the same
//! numbers of decimal digits; the bounds keep their safety margins and the
//! partial quotients are checked at twice the digits, so the solutions are
//! those of Python.
//!
//! **Exponential polynomials.** All the integers `n` with
//! `sum p_i(n) a_i**n = 0` for polynomials `p_i` with rational
//! coefficients and distinct positive rational bases `a_i`: beyond an
//! explicit bound the term of the largest base dominates wherever its
//! polynomial does not vanish.
//!
//! References
//! ==========
//!
//! 1. A. Baker, G. Wüstholz, *Logarithmic forms and group varieties*,
//!    Journal für die reine und angewandte Mathematik 442 (1993), 19-62.
//! 2. B. M. M. de Weger, *Algorithms for Diophantine equations*, CWI Tract
//!    65, 1989, Chapters 3 and 4.
//! 3. A. Dujella, A. Pethő, *A generalization of a theorem of Baker and
//!    Davenport*, Quarterly Journal of Mathematics 49 (1998), 291-306.

use std::collections::{BTreeMap, BTreeSet};

use crate::core::evalf::{real_int, real_rat, Real};
use crate::error::{arithmetic_error, value_error, PolyResult};
use crate::ntheory::factorint;
use crate::numbers::*;
use crate::polys::domains::QQ;
use crate::polys::rootisolation::dup_isolate_real_roots_sqf;
use crate::polys::sqfreetools::dup_sqf_part;

/// How many convergents beyond the first large enough one are tried in the
/// reduction with three logarithms.
const CONVERGENTS: usize = 60;

/// A term `p(n) * base**n`: the coefficients of `p` (highest degree first)
/// and the positive rational base.
pub type ExponentialTerm = (Vec<Rat>, Rat);

// ---------------------------------------------------------------------------
// the floating-point helpers (mpmath's workdps)

/// mpmath's `dps_to_prec`: the bits of a precision of `dps` decimal digits.
fn dps_to_prec(dps: usize) -> usize {
    std::cmp::max(1, (((dps + 1) as f64) * std::f64::consts::LOG2_10).round() as usize)
}

/// Floats at a fixed precision.
struct Floats {
    prec: usize,
}

impl Floats {
    fn digits(dps: usize) -> Floats {
        Floats { prec: dps_to_prec(dps) }
    }

    fn int(&self, n: &Int) -> Real {
        real_int(n, self.prec)
    }

    fn small(&self, n: i64) -> Real {
        self.int(&Int::from(n))
    }

    fn rational(&self, r: &Rat) -> Real {
        real_rat(r, self.prec).with_precision(self.prec).value()
    }

    fn ln_int(&self, n: &Int) -> Real {
        self.int(n).ln()
    }

    fn max(&self, a: Real, b: Real) -> Real {
        if a >= b {
            a
        } else {
            b
        }
    }

    fn one(&self) -> Real {
        self.small(1)
    }
}

fn floor_int(x: &Real) -> Int {
    x.floor().to_int().value()
}

fn ceil_int(x: &Real) -> Int {
    x.ceil().to_int().value()
}

fn abs_real(x: Real) -> Real {
    if x < Real::ZERO {
        -x
    } else {
        x
    }
}

/// The decimal length of a non-negative integer (Python's `len(str(n))`).
fn decimal_length(n: &Int) -> usize {
    n.to_string().trim_start_matches('-').len()
}

fn to_usize(n: &Int) -> PolyResult<usize> {
    usize::try_from(n.clone()).map_err(|_| arithmetic_error(format!("the bound {} is too large", n)))
}

// ---------------------------------------------------------------------------
// Pillai's equation

/// `x >= 0` with `base**x == value`, or `None`.
fn exponent_of(value: &Int, base: &Int) -> Option<u64> {
    if *value < Int::ONE {
        return None;
    }
    let mut value: Int = value.clone();
    let mut x: u64 = 0;
    loop {
        let (q, r) = div_mod_floor(&value, base);
        if !r.is_zero() {
            break;
        }
        value = q;
        x += 1;
    }
    if value == Int::ONE {
        Some(x)
    } else {
        None
    }
}

/// `(g, i, j)` with `a = g**i` and `b = g**j` when `a` and `b` are
/// multiplicatively dependent, `None` otherwise.
fn common_base(a: &Int, b: &Int) -> Option<(Int, u64, u64)> {
    let (fa, fb) = (factorint(a, None), factorint(b, None));
    if !fa.keys().eq(fb.keys()) {
        return None;
    }
    let gcd_u = |x: u64, y: u64| -> u64 {
        let (mut x, mut y) = (x, y);
        while y != 0 {
            let t = x % y;
            x = y;
            y = t;
        }
        x
    };
    let ea: u64 = fa.values().fold(0, |g, e| gcd_u(g, *e));
    let eb: u64 = fb.values().fold(0, |g, e| gcd_u(g, *e));
    // a = ra**ea and b = rb**eb with ra, rb not perfect powers: dependent
    // exactly when ra == rb
    if fa.iter().any(|(p, e)| e * eb != fb[p] * ea) {
        return None;
    }
    let mut g: Int = Int::ONE;
    for (p, e) in &fa {
        g *= p.pow((e / ea) as usize);
    }
    Some((g, ea, eb))
}

/// The solutions of `g**(i x) - g**(j y) = c`.
fn dependent_solutions(g: &Int, i: u64, j: u64, c: &Int) -> Vec<(u64, u64)> {
    let mut solutions: Vec<(u64, u64)> = Vec::new();
    let size: Int = abs(c);
    let mut v: u64 = 0;
    while g.pow(v as usize) <= size {
        let mut w: u64 = 1;
        while g.pow(w as usize) - Int::ONE <= size {
            if g.pow(v as usize) * (g.pow(w as usize) - Int::ONE) == size {
                // the larger exponent is v + w
                let (u, low) = if *c > Int::ZERO { (v + w, v) } else { (v, v + w) };
                if u % i == 0 && low % j == 0 {
                    solutions.push((u / i, low / j));
                }
            }
            w += 1;
        }
        v += 1;
    }
    solutions
}

/// `Y0` such that no solution with `x, y >= 1` and `b**y >= 2 |c|` has
/// `y >= Y0` (Baker–Wüstholz for two logarithms).
fn baker_bound(a: &Int, b: &Int, c: &Int) -> PolyResult<Int> {
    let f: Floats = Floats::digits(50);
    let (log_a, log_b) = (f.ln_int(a), f.ln_int(b));
    let constant: Real = f.int(&(Int::from(18 * 6 * 8) * Int::from(32u8).pow(4)))
        * f.small(4).ln()
        * f.max(log_a.clone(), f.one())
        * f.max(log_b.clone(), f.one());
    let kappa: Real = f.max(f.one(), (&log_b + f.one()) / &log_a);
    let offset: Real = f.ln_int(&(Int::from(2u8) * abs(c)));
    // y log b - log(2 |c|) < C max(log(kappa y), 1): the largest solution of
    // the equality, by iteration from above
    let mut y: Real = f.int(&Int::from(10u8).pow(40));
    for _ in 0..200 {
        y = (&constant * f.max((&kappa * &y).ln(), f.one()) + &offset) / &log_b;
    }
    let bound: Int = ceil_int(&(y * f.rational(&rat(1001, 1000)))) + Int::from(2u8);
    let bound_real: Real = f.int(&bound);
    // beyond the bound the left-hand side grows faster
    if !(bound_real > &constant / &log_b
        && &bound_real * &log_b - &offset > &constant * f.max((&kappa * &bound_real).ln(), f.one()))
    {
        return Err(arithmetic_error("the Baker bound was not established".to_string()));
    }
    Ok(bound)
}

/// The partial quotients `a_0, ..., a_k` of `log b/log a`, `k` the first
/// index with a denominator `q_k >= limit`.
fn partial_quotients(a: &Int, b: &Int, limit: &Int, digits: usize) -> PolyResult<Vec<Int>> {
    let f: Floats = Floats::digits(digits);
    let mut theta: Real = f.ln_int(b) / f.ln_int(a);
    let mut quotients: Vec<Int> = Vec::new();
    let (mut q_previous, mut q): (Int, Int) = (Int::ONE, Int::ZERO);
    loop {
        let term: Int = floor_int(&theta);
        quotients.push(term.clone());
        let next: Int = &term * &q + &q_previous;
        q_previous = q;
        q = next;
        if q >= *limit {
            return Ok(quotients);
        }
        let fraction: Real = theta - f.int(&term);
        if fraction == Real::ZERO {
            return Err(arithmetic_error("the continued fraction ended".to_string()));
        }
        theta = f.one() / fraction;
    }
}

/// The partial quotients up to `limit`, at enough digits that doubling them
/// changes nothing.
fn stable_partial_quotients(a: &Int, b: &Int, limit: &Int) -> PolyResult<Vec<Int>> {
    let mut digits: usize = 3 * decimal_length(limit) + 60;
    let mut quotients: Vec<Int> = partial_quotients(a, b, limit, digits)?;
    while quotients != partial_quotients(a, b, limit, 2 * digits)? {
        // not enough digits for these partial quotients
        digits *= 2;
        quotients = partial_quotients(a, b, limit, digits)?;
    }
    Ok(quotients)
}

fn largest_after_first(quotients: &[Int]) -> PolyResult<Int> {
    quotients[1..]
        .iter()
        .max()
        .cloned()
        .ok_or_else(|| value_error("max() arg is an empty sequence".to_string()))
}

/// A bound on `y` for the solutions with `x, y >= 1` and `b**y >= 2 |c|`,
/// from the continued fraction of `log b/log a`.
fn reduced_bound(a: &Int, b: &Int, c: &Int, bound: &Int) -> PolyResult<Int> {
    let quotients: Vec<Int> = stable_partial_quotients(a, b, bound)?;
    let largest: Int = largest_after_first(&quotients)?;
    let f: Floats = Floats::digits(50);
    let (log_a, log_b) = (f.ln_int(a), f.ln_int(b));
    // x/y a convergent: b**y < 2 |c| (A + 2) Y0/log a
    let product: Int = Int::from(2u8) * abs(c) * (&largest + Int::from(2u8)) * bound;
    let y: Int = ceil_int(&((f.int(&product) / &log_a).ln() / &log_b)) + Int::ONE;
    // x/y no convergent: b**y <= 4 |c| y/log a, false from some y on; with
    // log a > 1/2 it is enough that b**y > 8 |c| y
    let mut small: usize = 1;
    while b.pow(small) <= Int::from(8u8) * abs(c) * Int::from(small) {
        small += 1;
    }
    Ok(std::cmp::max(y, Int::from(small)))
}

/// `(k, i, j)` with `k > 0` and `(r/s)**k = a**i b**j` when `r/s` is
/// multiplicatively dependent on the independent `a` and `b`, `None`
/// otherwise.
fn dependence(a: &Int, b: &Int, r: &Int, s: &Int) -> Option<(Int, Int, Int)> {
    let (fa, fb) = (factorint(a, None), factorint(b, None));
    let mut exponents: BTreeMap<Int, i64> = BTreeMap::new();
    for (prime, e) in factorint(r, None) {
        *exponents.entry(prime).or_insert(0) += e as i64;
    }
    for (prime, e) in factorint(s, None) {
        *exponents.entry(prime).or_insert(0) -= e as i64;
    }
    let primes: BTreeSet<Int> = fa.keys().chain(fb.keys()).cloned().collect();
    if exponents.iter().any(|(prime, e)| *e != 0 && !primes.contains(prime)) {
        return None;
    }
    let get = |map: &BTreeMap<Int, u64>, p: &Int| -> i64 { map.get(p).map_or(0, |e| *e as i64) };
    for p1 in &primes {
        for p2 in &primes {
            let determinant: i64 = get(&fa, p1) * get(&fb, p2) - get(&fa, p2) * get(&fb, p1);
            if determinant == 0 {
                continue;
            }
            let e1: i64 = exponents.get(p1).copied().unwrap_or(0);
            let e2: i64 = exponents.get(p2).copied().unwrap_or(0);
            let u: Rat = rat(e1 * get(&fb, p2) - e2 * get(&fb, p1), determinant);
            let v: Rat = rat(get(&fa, p1) * e2 - get(&fa, p2) * e1, determinant);
            for prime in &primes {
                let value: Rat = &u * rat(get(&fa, prime), 1) + &v * rat(get(&fb, prime), 1);
                if value != rat(exponents.get(prime).copied().unwrap_or(0), 1) {
                    return None;
                }
            }
            let k: Int = lcm(&denom(&u), &denom(&v));
            let scale: Rat = rat_from_int(&k);
            return Some((k.clone(), numer(&(&u * &scale)), numer(&(&v * &scale))));
        }
    }
    None
}

/// `(Y0, M)`: no solution of `r a**x - s b**y = c` with `y >= 1` and
/// `s b**y >= 2 |c|` has `y >= Y0`, and those below have `x <= M`
/// (Baker–Wüstholz for the three logarithms of `a`, `b` and `r/s`).
fn three_logarithms_baker(a: &Int, b: &Int, c: &Int, r: &Int, s: &Int) -> PolyResult<(Int, Int)> {
    let f: Floats = Floats::digits(50);
    let (log_a, log_b) = (f.ln_int(a), f.ln_int(b));
    let log_ratio: Real = abs_real((f.int(r) / f.int(s)).ln());
    let largest: &Int = if r > s { r } else { s };
    let constant: Real = f.int(&(Int::from(18 * 24 * 81) * Int::from(32u8).pow(5)))
        * f.small(6).ln()
        * f.max(log_a.clone(), f.one())
        * f.max(log_b.clone(), f.one())
        * f.max(f.ln_int(largest), f.one());
    let kappa: Real = f.max(f.one(), (&log_b + &log_ratio + f.one()) / &log_a);
    let offset: Real = f.ln_int(&(Int::from(2u8) * abs(c))) - f.ln_int(s);
    let mut y: Real = f.int(&Int::from(10u8).pow(60));
    for _ in 0..300 {
        y = (&constant * f.max((&kappa * &y).ln(), f.one()) + &offset) / &log_b;
    }
    let bound: Int = ceil_int(&(y * f.rational(&rat(1001, 1000)))) + Int::from(2u8);
    let bound_real: Real = f.int(&bound);
    if !(bound_real > &constant / &log_b
        && &bound_real * &log_b - &offset > &constant * f.max((&kappa * &bound_real).ln(), f.one()))
    {
        return Err(arithmetic_error("the Baker bound was not established".to_string()));
    }
    let largest_x: Int = ceil_int(&(&kappa * &bound_real)) + Int::ONE;
    Ok((bound, largest_x))
}

/// A bound on `y` when `(r/s)**k = a**i b**j`: the form has two logarithms
/// and continued fractions do.
fn dependent_coefficient_bound(a: &Int, b: &Int, c: &Int, s: &Int, k: &Int, j: &Int, bound: &Int) -> PolyResult<Int> {
    let largest: Int = k * bound + abs(j) + Int::ONE;
    let quotients: Vec<Int> = stable_partial_quotients(a, b, &largest)?;
    let top: Int = largest_after_first(&quotients)?;
    let f: Floats = Floats::digits(50);
    let product: Int = Int::from(2u8) * k * abs(c) * (&top + Int::from(2u8)) * &largest;
    let argument: Real = f.int(&product) / (f.int(s) * f.ln_int(a));
    let y: Int = ceil_int(&(argument.ln() / f.ln_int(b))) + Int::ONE;
    let mut small: usize = 1;
    while s * b.pow(small) <= Int::from(8u8) * k * abs(c) * (k * Int::from(small) + abs(j)) {
        small += 1;
    }
    Ok(std::cmp::max(std::cmp::max(y, Int::from(small)), abs(j) + Int::ONE))
}

/// A bound on `y` for the solutions of `r a**x - s b**y = c` with `y >= 1`
/// and `s b**y >= 2 |c|`, `r != s` coprime (the lemma of Baker and
/// Davenport in the form of Dujella and Pethő).
fn three_logarithms_bound(a: &Int, b: &Int, c: &Int, r: &Int, s: &Int) -> PolyResult<Int> {
    let (bound, largest_x) = three_logarithms_baker(a, b, c, r, s)?;
    if let Some((k, _, j)) = dependence(a, b, r, s) {
        return dependent_coefficient_bound(a, b, c, s, &k, &j, &bound);
    }
    let digits: usize = 2 * decimal_length(&largest_x) + 260;
    let f: Floats = Floats::digits(digits);
    let log_b: Real = f.ln_int(b);
    let theta: Real = f.ln_int(a) / &log_b;
    let mu: Real = (f.int(r) / f.int(s)).ln() / &log_b;
    let weight: Real = f.int(&(Int::from(2u8) * abs(c))) / (f.int(s) * &log_b);
    let tolerance: Real = f.one() / f.int(&Int::from(10u8).pow(30));
    let largest_x_real: Real = f.int(&largest_x);
    let six_x: Int = Int::from(6u8) * &largest_x;
    let mut rest: Real = theta.clone();
    let (mut q_previous, mut q): (Int, Int) = (Int::ONE, Int::ZERO);
    let mut tried: usize = 0;
    while tried < CONVERGENTS {
        let term: Int = floor_int(&rest);
        let next: Int = &term * &q + &q_previous;
        q_previous = q;
        q = next;
        let fraction: Real = rest - f.int(&term);
        if fraction == Real::ZERO {
            break;
        }
        rest = f.one() / fraction;
        if q <= six_x {
            continue;
        }
        tried += 1;
        if decimal_length(&q) + 40 > digits / 2 {
            break;
        }
        let q_real: Real = f.int(&q);
        let q_mu: Real = &q_real * &mu;
        let q_theta: Real = &q_real * &theta;
        let nearest_mu: Real = f.int(&q_mu.round().to_int().value());
        let nearest_theta: Real = f.int(&q_theta.round().to_int().value());
        let eps: Real = abs_real(&q_mu - nearest_mu) - &largest_x_real * abs_real(&q_theta - nearest_theta);
        if eps > tolerance {
            return Ok(floor_int(&((&weight * &q_real / eps).ln() / &log_b)) + Int::from(2u8));
        }
    }
    Err(arithmetic_error(format!(
        "the reduction with the convergents of log {}/log {} fails for {}/{}",
        a, b, r, s
    )))
}

/// All the pairs of integers `x, y >= 0` with `r*a**x - s*b**y == c`, for
/// integers `a, b >= 2`, `r, s >= 1` and `c != 0`, sorted; the list is
/// complete. `ValueError` for other arguments.
pub fn pillai(a: &Int, b: &Int, c: &Int, r: &Int, s: &Int) -> PolyResult<Vec<(u64, u64)>> {
    let two: Int = Int::from(2u8);
    if *a < two || *b < two {
        return Err(value_error("the bases must be at least 2".to_string()));
    }
    if *r < Int::ONE || *s < Int::ONE {
        return Err(value_error("the coefficients must be positive".to_string()));
    }
    if c.is_zero() {
        return Err(value_error(
            "c must not be zero (a**x = b**y has (0, 0) and, for dependent bases, infinitely many solutions)"
                .to_string(),
        ));
    }
    let common_factor: Int = gcd(r, s);
    if !mod_floor(c, &common_factor).is_zero() {
        return Ok(Vec::new());
    }
    let (r, s, c) = (
        div_floor(r, &common_factor),
        div_floor(s, &common_factor),
        div_floor(c, &common_factor),
    );
    let mut solutions: BTreeSet<(u64, u64)> = BTreeSet::new();
    let common: Option<(Int, u64, u64)> = common_base(a, b);
    if let Some((g, i, j)) = &common {
        if r == Int::ONE && s == Int::ONE {
            return Ok(dependent_solutions(g, *i, *j, &c)
                .into_iter()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect());
        }
        // a = g**i, b = g**j: the smaller of g**(i x), g**(j y) divides c
        let (mut x, mut power): (u64, Int) = (0, Int::ONE);
        while power <= abs(&c) {
            let (quotient, remainder) = div_mod_floor(&(&r * &power - &c), &s);
            if remainder.is_zero() {
                if let Some(y) = exponent_of(&quotient, b) {
                    solutions.insert((x, y));
                }
            }
            x += 1;
            power *= a;
        }
        let (mut y, mut power): (u64, Int) = (0, Int::ONE);
        while power <= abs(&c) {
            let (quotient, remainder) = div_mod_floor(&(&c + &s * &power), &r);
            if remainder.is_zero() {
                if let Some(x) = exponent_of(&quotient, a) {
                    solutions.insert((x, y));
                }
            }
            y += 1;
            power *= b;
        }
        return Ok(solutions.into_iter().collect());
    }
    let mut limit: Int = if r == Int::ONE && s == Int::ONE {
        reduced_bound(a, b, &c, &baker_bound(a, b, &c)?)?
    } else {
        three_logarithms_bound(a, b, &c, &r, &s)?
    };
    // the y with s b**y < 2 |c| are not covered by the bounds
    while &s * b.pow(to_usize(&limit)?) < &two * abs(&c) {
        limit += Int::ONE;
    }
    let mut power: Int = Int::ONE;
    for y in 0..=to_usize(&limit)? as u64 {
        let (quotient, remainder) = div_mod_floor(&(&c + &s * &power), &r);
        if remainder.is_zero() {
            if let Some(x) = exponent_of(&quotient, a) {
                solutions.insert((x, y));
            }
        }
        power *= b;
    }
    Ok(solutions.into_iter().collect())
}

// ---------------------------------------------------------------------------
// exponential polynomials

fn evaluate(p: &[Rat], n: i64) -> Rat {
    let point: Rat = rat(n, 1);
    let mut value: Rat = Rat::ZERO;
    for coefficient in p {
        value = value * &point + coefficient;
    }
    value
}

/// The terms with equal bases added, the zero terms dropped, sorted by
/// decreasing base; `ValueError` for a base which is not positive.
fn merged(terms: &[ExponentialTerm]) -> PolyResult<Vec<ExponentialTerm>> {
    let mut by_base: BTreeMap<Rat, Vec<Rat>> = BTreeMap::new();
    for (p, base) in terms {
        if *base <= Rat::ZERO {
            return Err(value_error("the bases must be positive".to_string()));
        }
        let current: Vec<Rat> = by_base.get(base).cloned().unwrap_or_default();
        let size: usize = std::cmp::max(current.len(), p.len());
        let mut total: Vec<Rat> = vec![Rat::ZERO; size];
        for (k, coefficient) in current.iter().rev().enumerate() {
            total[size - 1 - k] += coefficient;
        }
        for (k, coefficient) in p.iter().rev().enumerate() {
            total[size - 1 - k] += coefficient;
        }
        let start: usize = total.iter().position(|c| !c.is_zero()).unwrap_or(total.len());
        by_base.insert(base.clone(), total[start..].to_vec());
    }
    Ok(by_base
        .into_iter()
        .rev()
        .filter(|(_, p)| !p.is_empty())
        .map(|(base, p)| (p, base))
        .collect())
}

fn floor_rat(v: &Rat) -> Int {
    div_floor(&numer(v), &denom(v))
}

fn ceil_rat(v: &Rat) -> Int {
    -floor_rat(&-v)
}

fn small_int(n: &Int) -> PolyResult<i64> {
    i64::try_from(n.clone()).map_err(|_| arithmetic_error(format!("the root bound {} is too large", n)))
}

/// The integer roots of a nonzero polynomial: its real roots are isolated
/// in intervals shorter than one, whose integers are tried.
fn integer_roots(p: &[Rat]) -> PolyResult<Vec<i64>> {
    if p.len() <= 1 {
        return Ok(Vec::new());
    }
    let mut found: BTreeSet<i64> = BTreeSet::new();
    for (low, high) in dup_isolate_real_roots_sqf(&dup_sqf_part(p, &QQ), Some(&rat(1, 2))) {
        for candidate in small_int(&floor_rat(&low))?..=small_int(&ceil_rat(&high))? {
            if evaluate(p, candidate).is_zero() {
                found.insert(candidate);
            }
        }
    }
    Ok(found.into_iter().collect())
}

fn rat_abs(v: &Rat) -> Rat {
    if *v < Rat::ZERO {
        -v
    } else {
        v.clone()
    }
}

fn rat_pow(v: &Rat, n: usize) -> Rat {
    rat_from_parts(numer(v).pow(n), denom(v).pow(n))
}

/// `N >= 1` such that for every integer `n >= N` which is not a root of the
/// polynomial of the first term (the largest base), that term is larger in
/// absolute value than the sum of the others.
fn dominance_bound(terms: &[ExponentialTerm]) -> PolyResult<i64> {
    let (lead, base) = &terms[0];
    let mut denominator: Int = Int::ONE;
    for c in lead {
        denominator = lcm(&denominator, &denom(c));
    }
    let ratio: Rat = base / &terms[1].1;
    let excess: usize = terms[1..].iter().map(|(q, _)| q.len() - 1).max().expect("other terms");
    let mut norm: Rat = Rat::ZERO;
    for (q, _) in &terms[1..] {
        for c in q {
            norm += rat_abs(c);
        }
    }
    let weight: Rat = rat_from_int(&denominator) * norm;
    // log(ratio) >= 1 - 1/ratio
    let quotient: Rat = rat(excess as i64, 1) / (Rat::ONE - recip(&ratio));
    let mut n: i64 = std::cmp::max(small_int(&floor_rat(&quotient))? + 1, 1);
    while rat_pow(&ratio, n as usize) <= &weight * rat_pow(&rat(n, 1), excess) {
        n *= 2;
    }
    Ok(n)
}

/// The integers `n >= 0` at which the sum of the terms (merged, by
/// decreasing base, not empty) vanishes.
fn nonnegative_zeros(terms: &[ExponentialTerm]) -> PolyResult<Vec<i64>> {
    let lead: &[Rat] = &terms[0].0;
    if terms.len() == 1 {
        return Ok(integer_roots(lead)?.into_iter().filter(|n| *n >= 0).collect());
    }
    let bound: i64 = dominance_bound(terms)?;
    let mut zeros: Vec<i64> = Vec::new();
    let mut powers: Vec<Rat> = vec![Rat::ONE; terms.len()];
    for n in 0..bound {
        let mut total: Rat = Rat::ZERO;
        for ((q, _), power) in terms.iter().zip(&powers) {
            total += evaluate(q, n) * power;
        }
        if total.is_zero() {
            zeros.push(n);
        }
        powers = powers
            .iter()
            .zip(terms)
            .map(|(power, (_, base))| power * base)
            .collect();
    }
    // beyond the bound the sum can vanish only where the first polynomial
    // does, and there it is the sum of the other terms
    for n in nonnegative_zeros(&terms[1..])? {
        if n >= bound && evaluate(lead, n).is_zero() {
            zeros.push(n);
        }
    }
    Ok(zeros)
}

/// All the integers `n` (of either sign) with `sum p_i(n) a_i**n == 0`;
/// terms with the same base are added. `ValueError` when the sum is
/// identically zero or a base is not positive.
pub fn exponential_polynomial_roots(terms: &[ExponentialTerm]) -> PolyResult<Vec<i64>> {
    let merged_terms: Vec<ExponentialTerm> = merged(terms)?;
    if merged_terms.is_empty() {
        return Err(value_error("the sum is identically zero".to_string()));
    }
    let mut zeros: BTreeSet<i64> = nonnegative_zeros(&merged_terms)?.into_iter().collect();
    // n -> -n: the bases inverted, the polynomials reflected
    let reflected: Vec<ExponentialTerm> = merged_terms
        .iter()
        .map(|(p, base)| {
            let degree: usize = p.len() - 1;
            let q: Vec<Rat> = p
                .iter()
                .enumerate()
                .map(|(k, c)| if (degree - k) % 2 == 1 { -c } else { c.clone() })
                .collect();
            (q, recip(base))
        })
        .collect();
    let mirrored: Vec<ExponentialTerm> = merged(&reflected)?;
    for n in nonnegative_zeros(&mirrored)? {
        zeros.insert(-n);
    }
    Ok(zeros.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(a: i64, b: i64, c: i64, r: i64, s: i64) -> Vec<(u64, u64)> {
        pillai(&int(a), &int(b), &int(c), &int(r), &int(s)).unwrap()
    }

    #[test]
    fn docstring_examples() {
        assert_eq!(p(3, 2, 1, 1, 1), vec![(1, 1), (2, 3)]);
        assert_eq!(p(2, 3, 13, 1, 1), vec![(4, 1), (8, 5)]);
        assert_eq!(p(2, 5, 3, 1, 1), vec![(2, 0), (3, 1), (7, 3)]);
        assert_eq!(p(5, 7, 1, 1, 1), vec![]);
        assert_eq!(p(2, 3, -1, 1, 1), vec![(1, 1), (3, 2)]);
        assert_eq!(p(4, 2, 12, 1, 1), vec![(2, 2)]);
        assert_eq!(p(10, 3, 73, 1, 1), vec![(2, 3)]);
        assert_eq!(p(5, 2, -3, 1, 2), vec![(0, 1), (1, 2), (3, 6)]);
        assert_eq!(p(5, 11, 4, 5, 1), vec![(0, 0), (2, 2)]);
        let one: Vec<Rat> = vec![rat(1, 1)];
        let terms: Vec<ExponentialTerm> = vec![
            (one.clone(), rat(2, 1)),
            (vec![rat(-1, 1), rat(0, 1), rat(0, 1)], rat(1, 1)),
        ];
        assert_eq!(exponential_polynomial_roots(&terms).unwrap(), vec![2, 4]);
        let terms: Vec<ExponentialTerm> = vec![
            (one.clone(), rat(3, 1)),
            (one.clone(), rat(4, 1)),
            (vec![rat(-1, 1)], rat(5, 1)),
        ];
        assert_eq!(exponential_polynomial_roots(&terms).unwrap(), vec![2]);
        let terms: Vec<ExponentialTerm> = vec![
            (one.clone(), rat(2, 1)),
            (one.clone(), rat(1, 2)),
            (vec![rat(-5, 2)], rat(1, 1)),
        ];
        assert_eq!(exponential_polynomial_roots(&terms).unwrap(), vec![-1, 1]);
    }
}
