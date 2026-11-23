//! Port of mpmath's series extrapolation for complex sequences
//! (`mpmath/calculus/extrapolation.py`: `richardson`, `shanks` with
//! `randomized=True`, `adaptive_extrapolation` with the default method
//! `'r+s'`, `nsum` over `[n, inf]`) and of `quadosc`
//! (`mpmath/calculus/quadrature.py`), the oscillatory quadrature of
//! `integrals/definite`.

// PORTED FROM SYMPY: mpmath/calculus/extrapolation.py, mpmath/calculus/quadrature.py (mpmath 1.3, BSD license)

use crate::core::evalf::{real_int, Real};
use crate::integrals::quadrature::{modulus, quad_rule, Point, Rule, Value};
use crate::numbers::Int;

fn at(x: &Real, p: usize) -> Real {
    x.clone().with_precision(p).value()
}

fn real(n: i64, p: usize) -> Real {
    real_int(&Int::from(n), p)
}

fn is_zero(r: &Real) -> bool {
    r.repr().significand() == &Int::ZERO
}

fn two_pow_neg(k: usize, p: usize) -> Real {
    real(1, p) / real_int(&(Int::ONE << k), p)
}

fn cadd(a: &Value, b: &Value, p: usize) -> Value {
    (at(&(&a.0 + &b.0), p), at(&(&a.1 + &b.1), p))
}

fn csub(a: &Value, b: &Value, p: usize) -> Value {
    (at(&(&a.0 - &b.0), p), at(&(&a.1 - &b.1), p))
}

fn cscale(a: &Value, r: &Real, p: usize) -> Value {
    (at(&(&a.0 * r), p), at(&(&a.1 * r), p))
}

fn cabs(a: &Value, p: usize) -> Real {
    at(&modulus(a), p)
}

fn czero(p: usize) -> Value {
    (real(0, p), real(0, p))
}

fn cis_zero(a: &Value) -> bool {
    is_zero(&a.0) && is_zero(&a.1)
}

/// `1/b`.
fn crecip(b: &Value, p: usize) -> Value {
    let d: Real = at(&(&b.0 * &b.0 + &b.1 * &b.1), p);
    (at(&(&b.0 / &d), p), at(&(-&b.1 / &d), p))
}

/// mpmath's `sign`: `z/|z|`, `0` at zero.
fn csign(a: &Value, p: usize) -> Value {
    if cis_zero(a) {
        return czero(p);
    }
    if is_zero(&a.1) {
        let s: i64 = if a.0 < Real::ZERO { -1 } else { 1 };
        return (real(s, p), real(0, p));
    }
    let m: Real = cabs(a, p);
    (at(&(&a.0 / &m), p), at(&(&a.1 / &m), p))
}

/// `richardson(seq)`: the extrapolated limit and the largest weight.
fn richardson(seq: &[Value], p: usize) -> (Value, Real) {
    let n0: usize = seq.len();
    let d1: Value = csub(&seq[n0 - 1], &seq[n0 - 2], p);
    let d2: Value = csub(&seq[n0 - 2], &seq[n0 - 3], p);
    let seq: Vec<Value> = if csign(&d1, p) != csign(&d2, p) {
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
    let mut c: Real = at(&(real_int(&signed, p) / real_int(&fact, p)), p);
    let mut s: Value = czero(p);
    let mut maxc: Real = real(1, p);
    for k in 0..=n {
        let ki: i64 = k as i64;
        s = cadd(&s, &cscale(&seq[n + k], &c, p), p);
        let ac: Real = if c < Real::ZERO { -c.clone() } else { c.clone() };
        if ac > maxc {
            maxc = ac;
        }
        c = at(
            &(&c * real(ki - ni, p) * at(&real_int(&Int::from(ki + ni + 1).pow(n), p), p)),
            p,
        );
        c = at(
            &(&c / (real(1 + ki, p) * at(&real_int(&Int::from(ki + ni).pow(n), p), p))),
            p,
        );
    }
    (s, maxc)
}

/// `shanks(seq, table, randomized=True)`: the table of iterated Shanks
/// transformations extended to the new terms.
fn shanks(seq: &[Value], mut table: Vec<Vec<Value>>, p: usize) -> Vec<Vec<Value>> {
    let start: usize = table.len();
    let mut stop: usize = seq.len() - 1;
    if stop & 1 == 1 {
        stop -= 1;
    }
    let eps: Real = two_pow_neg(p - 1, p);
    let mut rnd = crate::random::PyRandom::from_int(&Int::from(start));
    for i in start..stop {
        let mut row: Vec<Value> = Vec::new();
        for j in 0..=i {
            let (a, mut b): (Value, Value) = if j == 0 {
                (czero(p), csub(&seq[i + 1], &seq[i], p))
            } else {
                let a: Value = if j == 1 {
                    seq[i].clone()
                } else {
                    table[i - 1][j - 2].clone()
                };
                (a, csub(&row[j - 1], &table[i - 1][j - 1], p))
            };
            if cis_zero(&b) {
                let k: i64 = 1 + rnd.getrandbits(10) as i64;
                b = (at(&(real(k, p) * &eps), p), real(0, p));
            }
            row.push(cadd(&a, &crecip(&b, p), p));
        }
        table.push(row);
    }
    table
}

/// `nsum(term, [0, inf])` with mpmath's default `'r+s'` adaptive
/// extrapolation at `prec` bits; the terms are computed at the working
/// precision `(prec + 10)*4` passed to `term`. `None` when a term has no
/// value.
pub fn nsum(term: &mut dyn FnMut(i64, usize) -> Option<Value>, prec: usize) -> Option<Value> {
    let eps: Real = two_pow_neg(prec - 1, prec + 20);
    let tol: Real = at(&(eps / real(1024, prec + 20)), prec + 20);
    let dps: i64 = ((prec as f64 / std::f64::consts::LOG2_10).round() as i64 - 1).max(1);
    let maxterms: i64 = dps * 10;
    let wp: usize = (prec + 10) * 4;
    let eps_w: Real = two_pow_neg(wp - 1, wp);
    let mut try_richardson: bool = true;
    let mut try_shanks: bool = true;
    let mut last_richardson: Value = czero(wp);
    let mut shanks_table: Vec<Vec<Value>> = Vec::new();
    let mut partial: Vec<Value> = Vec::new();
    let mut index: i64 = 0;
    let mut step: i64 = 0;
    let mut best: Value = czero(wp);
    let round = |v: &Value| -> Value { (at(&v.0, prec), at(&v.1, prec)) };
    loop {
        if index >= maxterms {
            break;
        }
        step += 10;
        let mut psum: Value = partial.last().cloned().unwrap_or_else(|| czero(wp));
        for k in index..index + step {
            psum = cadd(&psum, &term(k, wp)?, wp);
            partial.push(psum.clone());
        }
        index += step;
        best = partial[partial.len() - 1].clone();
        let mut error: Real = cabs(&csub(&best, &partial[partial.len() - 2], wp), wp);
        if error <= tol {
            return Some(round(&best));
        }
        if try_richardson {
            let (value, maxc) = richardson(&partial, wp);
            let richardson_error: Real = cabs(&csub(&value, &last_richardson, wp), wp);
            if richardson_error <= tol {
                return Some(round(&value));
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
        if try_shanks {
            shanks_table = shanks(&partial, shanks_table, wp);
            let row: &Vec<Value> = shanks_table.last().expect("a row");
            let (est1, maxc, shanks_error) = if row.len() == 2 {
                (row[1].clone(), real(0, wp), real(0, wp))
            } else {
                let n: usize = row.len();
                let est1: Value = row[n - 1].clone();
                let maxc: Real = cabs(&row[n - 2], wp);
                let err: Real = cabs(&csub(&est1, &row[n - 3], wp), wp);
                (est1, maxc, err)
            };
            if shanks_error <= tol {
                return Some(round(&est1));
            }
            if &eps_w * &maxc > tol {
                try_shanks = false;
            }
            if shanks_error < error {
                error = shanks_error;
                best = est1;
            }
        }
        let _ = error;
    }
    Some(round(&best))
}

/// `quadgl(f, [a, b])` at `prec` bits.
fn quadgl(f: &mut dyn FnMut(&Real) -> Option<Value>, a: &Real, b: &Real, prec: usize) -> Option<Value> {
    quad_rule(
        f,
        &[Point::Finite(a.clone()), Point::Finite(b.clone())],
        prec,
        Rule::GaussLegendre,
    )
    .map(|r| r.0)
}

/// `quadosc(f, [a, inf], period=period)` at `prec` bits: the integral
/// over `[a, period/2]`, then the half periods summed by [`nsum`].
pub fn quadosc(
    f: &mut dyn FnMut(&Real, usize) -> Option<Value>,
    a: &Real,
    period: &Real,
    prec: usize,
) -> Option<Value> {
    let zeros = |n: i64, p: usize| -> Real { at(&(real(n, p) * at(period, p) / real(2, p)), p) };
    let first: Value = quadgl(&mut |x: &Real| f(x, prec), a, &zeros(1, prec), prec)?;
    let mut term = |k: i64, wp: usize| -> Option<Value> {
        // the terms from n = 1 (standardize_infinite shifts by a = 1)
        let n: i64 = k + 1;
        quadgl(&mut |x: &Real| f(x, wp), &zeros(n, wp), &zeros(n + 1, wp), wp)
    };
    let rest: Value = nsum(&mut term, prec)?;
    Some(cadd(&first, &rest, prec))
}
