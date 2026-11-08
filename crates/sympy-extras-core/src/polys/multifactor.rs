//! Port of `sympy_extras/polys/multifactor.py`: factorisation of
//! multivariate polynomials over `ZZ` by Wang's enhanced extended Zassenhaus
//! algorithm (EEZ), on the sparse polynomials over `ZZ`.
//!
//! The content in the first variable is factored recursively; the
//! square-free part is evaluated at integer points of the other variables
//! until the univariate image is square-free with a leading coefficient
//! whose factors are told apart by their values; the image is factored over
//! `ZZ`, the leading coefficients are distributed among the factors, and the
//! factors are lifted one variable at a time by Hensel lifting with
//! multivariate diophantine equations; trial division gives the
//! multiplicities.
//!
//! The diophantine equations are solved modulo `nextprime(Mignotte bound)`,
//! which may exceed the machine words: they run over [`BigFiniteField`]
//! (Python's one `GF(p)` takes any modulus). The evaluation points come from
//! the seeded [`Rng`] instead of Python's Mersenne Twister; the factors are
//! sorted as SymPy sorts them, so the result does not depend on the points.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/multifactor.py):
// sympy/polys/factortools.py (dmp_zz_factor, dmp_zz_wang,
// dmp_zz_wang_non_divisors, dmp_zz_wang_test_points,
// dmp_zz_wang_lead_coeffs, dup_zz_diophantine, dmp_zz_diophantine,
// dmp_zz_wang_hensel_lifting, dmp_trial_division, dmp_zz_mignotte_bound),
// sympy/polys/sqfreetools.py (dmp_sqf_part), sympy/polys/polyutils.py
// (_sort_factors) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: sparse polynomials in all the variables instead of dense nested
// lists; a seeded local random generator for the evaluation points.
// ---------------------------------------------------------------------

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::ntheory::nextprime;
use crate::numbers::*;
use crate::random::Rng;

use super::densearith::{dup_add, dup_div, dup_mul, dup_mul_ground, dup_neg, dup_rem};
use super::densebasic::{dup_from_dict, dup_strip};
use super::densetools::dup_primitive;
use super::domains::{BigFiniteField, Domain, IntegerRing, SparsePolynomialRing, ZZ};
use super::euclidtools::dup_gcdex;
use super::factortools::{dup_zz_factor, dup_zz_factor_sqf};
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};
use super::sqfreetools::dup_sqf_p;

type Poly = PolyElement<IntegerRing>;
type Ring = Arc<PolyRing<IntegerRing>>;

/// The factors with their multiplicities.
pub type Factors = Vec<(Poly, usize)>;

pub const EEZ_NUMBER_OF_CONFIGS: usize = 3;
pub const EEZ_NUMBER_OF_TRIES: usize = 5;
pub const EEZ_MODULUS_STEP: i64 = 2;

/// `EvaluationFailed` and `ExtraneousFactors` of the Python code: the
/// internal failures which make the algorithm try other points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Failure {
    Evaluation,
    Extraneous,
    /// the `ArithmeticError` of an inexact division in the diophantine
    /// equations, which Python does not catch
    Inexact,
}

// ---------------------------------------------------------------------
// sparse versions of the dense helpers

fn degree(f: &Poly, i: usize) -> i64 {
    f.terms.keys().map(|m| m[i] as i64).max().unwrap_or(-1)
}

/// The leading coefficient in `x0`, a polynomial in the others.
fn lc_first(f: &Poly) -> Poly {
    let top: i64 = degree(f, 0);
    f.ring.from_dict(
        f.terms
            .iter()
            .filter(|(m, _)| m[0] as i64 == top)
            .map(|(m, c)| {
                let mut key: Monomial = m.clone();
                key[0] = 0;
                (key, c.clone())
            })
            .collect(),
    )
}

/// `f` without its terms of highest degree in `x0`.
fn without_leading(f: &Poly) -> Poly {
    let top: i64 = degree(f, 0);
    f.ring.from_dict(
        f.terms
            .iter()
            .filter(|(m, _)| m[0] as i64 != top)
            .map(|(m, c)| (m.clone(), c.clone()))
            .collect(),
    )
}

fn eval_var(f: &Poly, j: usize, a: &Int) -> Poly {
    let mut terms: Terms<Int> = Terms::new();
    for (m, c) in &f.terms {
        let mut key: Monomial = m.clone();
        key[j] = 0;
        let value: Int = c * a.pow(m[j] as usize);
        let total: Int = match terms.get(&key) {
            Some(existing) => existing + value,
            None => value,
        };
        terms.insert(key, total);
    }
    f.ring.from_dict(terms)
}

/// The last `len(A)` variables set to `A`.
fn eval_tail(f: &Poly, a: &[Int]) -> Poly {
    let n: usize = f.ring.ngens;
    let mut f: Poly = f.clone();
    for (k, value) in a.iter().enumerate() {
        f = eval_var(&f, n - a.len() + k, value);
    }
    f
}

fn constant(f: &Poly) -> Int {
    f.terms.get(&f.ring.zero_monom).cloned().unwrap_or(Int::ZERO)
}

/// The `k`-th derivative in `x_j` at `x_j = a`.
fn diff_eval(f: &Poly, k: usize, a: &Int, j: usize) -> Poly {
    let mut f: Poly = f.clone();
    for _ in 0..k {
        f = f.diff(j);
    }
    eval_var(&f, j, a)
}

fn symmetric(c: &Int, p: &Int) -> Int {
    let r: Int = mod_floor(c, p);
    if r > div_floor(p, &Int::from(2u8)) {
        r - p
    } else {
        r
    }
}

/// Symmetric residues of the coefficients modulo `p`.
fn trunc(f: &Poly, p: &Int) -> Poly {
    f.ring
        .from_dict(f.terms.iter().map(|(m, c)| (m.clone(), symmetric(c, p))).collect())
}

fn quo_ground(f: &Poly, c: &Int) -> Poly {
    f.ring
        .from_dict(f.terms.iter().map(|(m, v)| (m.clone(), div_floor(v, c))).collect())
}

fn content(f: &Poly) -> Int {
    let mut result: Int = Int::ZERO;
    for c in f.terms.values() {
        result = gcd(&result, c);
    }
    result
}

fn primitive(f: &Poly) -> (Int, Poly) {
    let c: Int = content(f);
    if c.is_zero() || c.is_one() {
        return (c, f.clone());
    }
    let part: Poly = quo_ground(f, &c);
    (c, part)
}

fn expand(h: &[Poly], ring: &Ring) -> Poly {
    let mut result: Poly = ring.one();
    for factor in h {
        result = result.mul(factor);
    }
    result
}

/// `x_j - a`.
fn variable_minus(ring: &Ring, j: usize, a: &Int) -> Poly {
    let monomial: Monomial = (0..ring.ngens).map(|i| if i == j { 1 } else { 0 }).collect();
    let mut terms: Terms<Int> = Terms::new();
    terms.insert(monomial, Int::ONE);
    terms.insert(ring.zero_monom.clone(), -a);
    ring.from_dict(terms)
}

/// A polynomial in `x0` only as a dense list.
fn to_dense(f: &Poly) -> Vec<Int> {
    let terms: BTreeMap<usize, Int> = f.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect();
    dup_from_dict(&terms, &ZZ)
}

fn from_dense(f: &[Int], ring: &Ring) -> Poly {
    let n: usize = f.len();
    ring.from_dict(
        f.iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| {
                let mut key: Monomial = vec![0; ring.ngens];
                key[0] = (n - 1 - i) as u32;
                (key, c.clone())
            })
            .collect(),
    )
}

/// `f / g` when the division is exact over `ZZ`, else `None`.
fn exquo(f: &Poly, g: &Poly) -> Option<Poly> {
    match f.div(std::slice::from_ref(g)) {
        Ok((mut quotients, remainder)) if remainder.is_zero() => Some(quotients.remove(0)),
        _ => None,
    }
}

/// A polynomial free of `x0` in the ring of the other variables.
fn drop_first(f: &Poly, ring: &Ring) -> Poly {
    ring.from_dict(f.terms.iter().map(|(m, c)| (m[1..].to_vec(), c.clone())).collect())
}

fn add_first(f: &Poly, ring: &Ring) -> Poly {
    ring.from_dict(
        f.terms
            .iter()
            .map(|(m, c)| {
                let mut key: Monomial = vec![0];
                key.extend_from_slice(m);
                (key, c.clone())
            })
            .collect(),
    )
}

/// The dense recursive representation of a polynomial as nested tuples,
/// which SymPy's lists compare as (`_sort_factors`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DenseKey {
    Int(Int),
    Tuple(Vec<DenseKey>),
}

/// The key SymPy sorts the factors by.
pub fn factor_sort_key(f: &Poly) -> DenseKey {
    dense_key(&f.terms, 0, f.ring.ngens)
}

fn dense_key(terms: &Terms<Int>, i: usize, n: usize) -> DenseKey {
    if terms.is_empty() {
        return DenseKey::Tuple(Vec::new());
    }
    let mut by_degree: BTreeMap<u32, Terms<Int>> = BTreeMap::new();
    for (m, c) in terms {
        by_degree.entry(m[i]).or_default().insert(m.clone(), c.clone());
    }
    let top: u32 = *by_degree.keys().next_back().expect("a nonzero polynomial");
    let empty: Terms<Int> = Terms::new();
    if i == n - 1 {
        return DenseKey::Tuple(
            (0..=top)
                .rev()
                .map(|d| {
                    let mut total: Int = Int::ZERO;
                    for v in by_degree.get(&d).unwrap_or(&empty).values() {
                        total += v;
                    }
                    DenseKey::Int(total)
                })
                .collect(),
        );
    }
    DenseKey::Tuple(
        (0..=top)
            .rev()
            .map(|d| dense_key(by_degree.get(&d).unwrap_or(&empty), i + 1, n))
            .collect(),
    )
}

fn sorted(mut factors: Factors) -> Factors {
    factors.sort_by(|a, b| {
        (degree(&a.0, 0) + 1, a.1, factor_sort_key(&a.0)).cmp(&(degree(&b.0, 0) + 1, b.1, factor_sort_key(&b.0)))
    });
    factors
}

// ---------------------------------------------------------------------
// the square-free part and the trial division

/// The square-free part over `ZZ` (`dmp_sqf_part`), primitive.
pub fn sqf_part(f: &Poly) -> PolyResult<Poly> {
    if f.is_zero() {
        return Ok(f.clone());
    }
    let f: Poly = if is_negative(&f.LC()) { f.neg() } else { f.clone() };
    let r: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::sympy(ZZ, f.ring.names.clone());
    let mut g: Poly = f.clone();
    for i in 0..f.ring.ngens {
        g = r.gcd(&g, &f.diff(i));
    }
    match exquo(&f, &g) {
        Some(quotient) => Ok(primitive(&quotient).1),
        None => Err(value_error("the gcd does not divide".to_string())),
    }
}

/// The multiplicities of the factors in `f` (`dmp_trial_division`).
pub fn trial_division(f: &Poly, factors: &[Poly]) -> PolyResult<Factors> {
    let mut result: Factors = Vec::new();
    let mut f: Poly = f.clone();
    for factor in factors {
        let mut k: usize = 0;
        while let Some(quotient) = exquo(&f, factor) {
            f = quotient;
            k += 1;
        }
        if k == 0 {
            return Err(value_error("trial division failed".to_string()));
        }
        result.push((factor.clone(), k));
    }
    Ok(sorted(result))
}

fn mignotte_bound(f: &Poly) -> Int {
    let a: Int = f.terms.values().map(abs).max().unwrap_or(Int::ZERO);
    let b: Int = abs(&f.LC());
    let n: i64 = (0..f.ring.ngens).map(|i| degree(f, i).max(0)).sum();
    isqrt(&Int::from(n + 1)) * Int::from(2u8).pow(n as usize) * a * b
}

// ---------------------------------------------------------------------
// Wang's algorithm

fn non_divisors(e: &[Int], cs: &Int, ct: &Int) -> Option<Vec<Int>> {
    let mut result: Vec<Int> = vec![cs * ct];
    for q0 in e {
        let mut q: Int = abs(q0);
        for r0 in result.iter().rev() {
            let mut r: Int = r0.clone();
            while !r.is_one() {
                r = gcd(&r, &q);
                q = div_floor(&q, &r);
            }
            if q.is_one() {
                return None;
            }
        }
        result.push(q);
    }
    Some(result[1..].to_vec())
}

/// `(c, h, E)` for the evaluation point, or `Evaluation` when it is unusable.
fn test_points(f: &Poly, t: &[Poly], ct: &Int, a: &[Int]) -> Result<(Int, Vec<Int>, Vec<Int>), Failure> {
    if eval_tail(&lc_first(f), a).is_zero() {
        return Err(Failure::Evaluation);
    }
    let g: Vec<Int> = to_dense(&eval_tail(f, a));
    if !dup_sqf_p(&g, &ZZ) {
        return Err(Failure::Evaluation);
    }
    let (mut c, mut h) = dup_primitive(&g, &ZZ);
    if is_negative(&h[0]) {
        c = -c;
        h = dup_neg(&h, &ZZ);
    }
    let e: Vec<Int> = t.iter().map(|ti| constant(&eval_tail(ti, a))).collect();
    if non_divisors(&e, &c, ct).is_none() {
        return Err(Failure::Evaluation);
    }
    Ok((c, h, e))
}

#[allow(clippy::type_complexity)]
fn lead_coeffs(
    f: &Poly,
    t: &[Poly],
    cs: &Int,
    e: &[Int],
    h: &[Vec<Int>],
    a: &[Int],
) -> Result<(Poly, Vec<Vec<Int>>, Vec<Poly>), Failure> {
    let ring: &Ring = &f.ring;
    let mut big_c: Vec<Poly> = Vec::new();
    let mut j: Vec<bool> = vec![false; e.len()];
    for hi in h {
        let mut c: Poly = ring.one();
        let mut d: Int = &hi[0] * cs;
        for i in (0..e.len()).rev() {
            let mut k: u64 = 0;
            while mod_floor(&d, &e[i]).is_zero() {
                d = div_floor(&d, &e[i]);
                k += 1;
            }
            if k != 0 {
                c = c.mul(&t[i].pow(k));
                j[i] = true;
            }
        }
        big_c.push(c);
    }
    if !j.iter().all(|x| *x) {
        return Err(Failure::Extraneous);
    }
    let mut cs: Int = cs.clone();
    let mut cc_list: Vec<Poly> = Vec::new();
    let mut hh: Vec<Vec<Int>> = Vec::new();
    for (c, hi) in big_c.iter().zip(h) {
        let mut d: Int = constant(&eval_tail(c, a));
        let lc: &Int = &hi[0];
        let mut hi: Vec<Int> = hi.clone();
        let cc: Int = if cs.is_one() {
            div_floor(lc, &d)
        } else {
            let g: Int = gcd(lc, &d);
            d = div_floor(&d, &g);
            let cc: Int = div_floor(lc, &g);
            hi = dup_mul_ground(&hi, &d, &ZZ);
            cs = div_floor(&cs, &d);
            cc
        };
        cc_list.push(c.mul_ground(&cc));
        hh.push(hi);
    }
    if cs.is_one() {
        return Ok((f.clone(), hh, cc_list));
    }
    let ccc: Vec<Poly> = cc_list.iter().map(|c| c.mul_ground(&cs)).collect();
    let hhh: Vec<Vec<Int>> = hh.iter().map(|hi| dup_mul_ground(hi, &cs, &ZZ)).collect();
    Ok((f.mul_ground(&cs.pow(h.len() - 1)), hhh, ccc))
}

fn to_int_poly(f: &[Int], p: &Int) -> Vec<Int> {
    let symmetric_coefficients: Vec<Int> = f.iter().map(|c| symmetric(c, p)).collect();
    dup_strip(&symmetric_coefficients, &ZZ)
}

fn reduce_mod(f: &[Int], k: &BigFiniteField) -> Vec<Int> {
    let residues: Vec<Int> = f.iter().map(|c| k.from_integer(c)).collect();
    dup_strip(&residues, k)
}

fn monomial_power(m: usize) -> Vec<Int> {
    let mut shift: Vec<Int> = vec![Int::ZERO; m + 1];
    shift[0] = Int::ONE;
    shift
}

/// `s_i` with `sum s_i prod_{j != i} F_j = x**m` modulo `p`, `deg s_i < deg
/// F_i` (`dup_zz_diophantine`).
fn dup_diophantine(f: &[Vec<Int>], m: usize, p: &Int) -> Vec<Vec<Int>> {
    let k: BigFiniteField = BigFiniteField::new(p.clone()).expect("a prime modulus");
    if f.len() == 2 {
        let (a, b) = (reduce_mod(&f[0], &k), reduce_mod(&f[1], &k));
        let (s, t, _) = dup_gcdex(&b, &a, &k).expect("a field");
        let shift: Vec<Int> = monomial_power(m);
        let s: Vec<Int> = dup_mul(&s, &shift, &k);
        let t: Vec<Int> = dup_mul(&t, &shift, &k);
        let (q, s) = dup_div(&s, &a, &k).expect("a nonzero divisor over a field");
        let t: Vec<Int> = dup_add(&t, &dup_mul(&q, &b, &k), &k);
        return vec![to_int_poly(&s, p), to_int_poly(&t, p)];
    }
    let mut g: Vec<Vec<Int>> = vec![f[f.len() - 1].clone()];
    for fi in f[1..f.len() - 1].iter().rev() {
        let product: Vec<Int> = dup_mul(fi, &g[0], &ZZ);
        g.insert(0, product);
    }
    let mut s_list: Vec<Vec<Int>> = Vec::new();
    let mut t_list: Vec<Vec<Int>> = vec![vec![Int::ONE]];
    for (fi, gi) in f.iter().zip(&g) {
        let mut solution: Vec<Vec<Int>> =
            diophantine_univariate(&[gi.clone(), fi.clone()], &t_list[t_list.len() - 1], p);
        let s: Vec<Int> = solution.pop().expect("two solutions");
        let t: Vec<Int> = solution.pop().expect("two solutions");
        t_list.push(t);
        s_list.push(s);
    }
    s_list.push(t_list[t_list.len() - 1].clone());
    let shift: Vec<Int> = monomial_power(m);
    s_list
        .iter()
        .zip(f)
        .map(|(s, fi)| {
            let (sp, fp) = (reduce_mod(s, &k), reduce_mod(fi, &k));
            let r: Vec<Int> = dup_rem(&dup_mul(&sp, &shift, &k), &fp, &k).expect("a nonzero divisor over a field");
            to_int_poly(&r, p)
        })
        .collect()
}

/// `dmp_zz_diophantine` with no evaluation point (dense univariate).
fn diophantine_univariate(f: &[Vec<Int>], c: &[Int], p: &Int) -> Vec<Vec<Int>> {
    let mut s: Vec<Vec<Int>> = vec![Vec::new(); f.len()];
    let n: usize = c.len();
    for (i, coefficient) in c.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let t: Vec<Vec<Int>> = dup_diophantine(f, n - 1 - i, p);
        for (j, tj) in t.iter().enumerate() {
            let scaled: Vec<Int> = dup_mul_ground(tj, coefficient, &ZZ);
            s[j] = to_int_poly(&dup_add(&s[j], &scaled, &ZZ), p);
        }
    }
    s
}

fn factorial(n: usize) -> Int {
    let mut result: Int = Int::ONE;
    for k in 2..=n {
        result *= Int::from(k);
    }
    result
}

/// `S_i` with `sum S_i prod_{j != i} F_j = c` modulo `p` and the ideal `((x_k
/// - A_k)**(d + 1))`, the polynomials in `x0, ..., x_len(A)` of the full
/// ring (`dmp_zz_diophantine`).
fn diophantine(f: &[Poly], c: &Poly, a: &[Int], d: i64, p: &Int) -> Result<Vec<Poly>, Failure> {
    let ring: Ring = Arc::clone(&c.ring);
    if a.is_empty() {
        let dense: Vec<Vec<Int>> = diophantine_univariate(&f.iter().map(to_dense).collect::<Vec<_>>(), &to_dense(c), p);
        return Ok(dense.iter().map(|s| from_dense(s, &ring)).collect());
    }
    let n: usize = a.len();
    let e: Poly = expand(f, &ring);
    let (last, rest) = (&a[n - 1], &a[..n - 1]);
    let mut b: Vec<Poly> = Vec::new();
    let mut g: Vec<Poly> = Vec::new();
    for fi in f {
        match exquo(&e, fi) {
            Some(quotient) => b.push(quotient),
            None => return Err(Failure::Inexact),
        }
        g.push(eval_var(fi, n, last));
    }
    let big_c: Poly = eval_var(c, n, last);
    let mut s: Vec<Poly> = diophantine(&g, &big_c, rest, d, p)?;
    let mut c: Poly = c.clone();
    for (si, bi) in s.iter().zip(&b) {
        c = c.sub(&si.mul(bi));
    }
    c = trunc(&c, p);
    let m: Poly = variable_minus(&ring, n, last);
    let mut big_m: Poly = ring.one();
    for k in 0..d.max(0) as usize {
        if c.is_zero() {
            break;
        }
        big_m = big_m.mul(&m);
        let mut big_c: Poly = diff_eval(&c, k + 1, last, n);
        if !big_c.is_zero() {
            big_c = quo_ground(&big_c, &factorial(k + 1));
            let t: Vec<Poly> = diophantine(&g, &big_c, rest, d, p)?
                .into_iter()
                .map(|ti| ti.mul(&big_m))
                .collect();
            s = s.iter().zip(&t).map(|(si, ti)| si.add(ti)).collect();
            for (ti, bi) in t.iter().zip(&b) {
                c = c.sub(&ti.mul(bi));
            }
            c = trunc(&c, p);
        }
    }
    Ok(s.iter().map(|si| trunc(si, p)).collect())
}

fn hensel_lifting(f: &Poly, h0: &[Vec<Int>], lc: &[Poly], a: &[Int], p: &Int) -> Result<Vec<Poly>, Failure> {
    let ring: Ring = Arc::clone(&f.ring);
    let n: usize = a.len();
    let mut s: Vec<Poly> = vec![f.clone()];
    for (i, value) in a[1..].iter().rev().enumerate() {
        let image: Poly = trunc(&eval_var(&s[0], n - i, value), p);
        s.insert(0, image);
    }
    let d: i64 = (1..ring.ngens)
        .map(|i| degree(f, i))
        .max()
        .expect("at least two variables");
    let mut h: Vec<Poly> = h0.iter().map(|hi| from_dense(hi, &ring)).collect();
    for ((j, si), value) in (2..n + 2).zip(&s).zip(a) {
        let g: Vec<Poly> = h.clone();
        let w: usize = j - 1;
        let (first, last) = (&a[..j - 2], &a[j - 1..]);
        for (hi, lci) in h.iter_mut().zip(lc) {
            let lead: Poly = trunc(&eval_tail(lci, last), p);
            let top: i64 = degree(hi, 0);
            let monomial: Monomial = (0..ring.ngens).map(|k| if k == 0 { top as u32 } else { 0 }).collect();
            let shifted: Poly = lead.mul(&ring.term(monomial, Int::ONE));
            *hi = shifted.add(&without_leading(hi));
        }
        let m: Poly = variable_minus(&ring, w, value);
        let mut big_m: Poly = ring.one();
        let mut c: Poly = si.sub(&expand(&h, &ring));
        let dj: i64 = degree(si, w);
        for k in 0..dj.max(0) as usize {
            if c.is_zero() {
                break;
            }
            big_m = big_m.mul(&m);
            let mut big_c: Poly = diff_eval(&c, k + 1, value, w);
            if !big_c.is_zero() {
                big_c = quo_ground(&big_c, &factorial(k + 1));
                let t: Vec<Poly> = diophantine(&g, &big_c, first, d, p)?;
                h = h
                    .iter()
                    .zip(&t)
                    .map(|(hi, ti)| trunc(&hi.add(&ti.mul(&big_m)), p))
                    .collect();
                c = trunc(&si.sub(&expand(&h, &ring)), p);
            }
        }
    }
    if expand(&h, &ring) != *f {
        return Err(Failure::Extraneous);
    }
    Ok(h)
}

/// A configuration of evaluation: the image, its content, the values of
/// the factors of the leading coefficient, the factors of the image and the
/// point.
type Config = (Vec<Int>, Int, Vec<Int>, Vec<Vec<Int>>, Vec<Int>);

/// The irreducible factors of a square-free primitive polynomial over `ZZ`
/// of positive degree in `x0` with at least two variables
/// (`dmp_zz_wang`).
pub fn zz_wang(f: &Poly, modulus: Option<i64>, seed: u64) -> PolyResult<Vec<Poly>> {
    let ring: Ring = Arc::clone(&f.ring);
    let mut rng: Rng = Rng::new(seed);
    let u: usize = ring.ngens - 1;
    let inner: Ring = PolyRing::new(ZZ, ring.names[1..].to_vec(), MonomialOrder::Lex)?;
    let (ct, t_inner) = zz_factor(&drop_first(&lc_first(f), &inner))?;
    let t: Vec<Poly> = t_inner.iter().map(|(ti, _)| add_first(ti, &ring)).collect();
    let b: Int = mignotte_bound(f);
    let p: Int = nextprime(&b);
    let mut modulus: i64 = modulus.unwrap_or(if u == 1 { 2 } else { 1 });
    let mut history: HashSet<Vec<Int>> = HashSet::new();
    let mut configs: Vec<Config> = Vec::new();
    let mut r: Option<usize> = None;
    let zeros: Vec<Int> = vec![Int::ZERO; u];
    if let Ok((cs, s, e)) = test_points(f, &t, &ct, &zeros) {
        let (_, h) = dup_zz_factor_sqf(&s, &ZZ)?;
        r = Some(h.len());
        if h.len() == 1 {
            return Ok(vec![f.clone()]);
        }
        configs.push((s, cs, e, h, zeros));
    }
    while configs.len() < EEZ_NUMBER_OF_CONFIGS {
        let mut filled: bool = false;
        for _ in 0..EEZ_NUMBER_OF_TRIES {
            let span: u64 = (2 * modulus + 1) as u64;
            let a: Vec<Int> = (0..u)
                .map(|_| Int::from(rng.randrange(span) as i64 - modulus))
                .collect();
            if !history.insert(a.clone()) {
                continue;
            }
            let (cs, s, e) = match test_points(f, &t, &ct, &a) {
                Ok(found) => found,
                Err(_) => continue,
            };
            let (_, h) = dup_zz_factor_sqf(&s, &ZZ)?;
            let rr: usize = h.len();
            match r {
                Some(current) if rr != current => {
                    if rr < current {
                        configs.clear();
                        r = Some(rr);
                    } else {
                        continue;
                    }
                }
                Some(_) => {}
                None => r = Some(rr),
            }
            if r == Some(1) {
                return Ok(vec![f.clone()]);
            }
            configs.push((s, cs, e, h, a));
            if configs.len() == EEZ_NUMBER_OF_CONFIGS {
                filled = true;
                break;
            }
        }
        if !filled {
            modulus += EEZ_MODULUS_STEP;
        }
    }
    let mut best: usize = 0;
    let mut best_norm: Option<Int> = None;
    for (i, (s, _, _, _, _)) in configs.iter().enumerate() {
        let norm: Int = s.iter().map(abs).max().unwrap_or(Int::ZERO);
        if best_norm.as_ref().map_or(true, |current| norm < *current) {
            best_norm = Some(norm);
            best = i;
        }
    }
    let (_, cs, e, h, a) = &configs[best];
    let lifted: Result<Vec<Poly>, Failure> =
        lead_coeffs(f, &t, cs, e, h, a).and_then(|(g, hh, lc)| hensel_lifting(&g, &hh, &lc, a, &p));
    let factors: Vec<Poly> = match lifted {
        Ok(factors) => factors,
        Err(Failure::Inexact) => return Err(value_error("inexact division".to_string())),
        Err(_) => return zz_wang(f, Some(modulus + 1), seed + 1),
    };
    Ok(factors
        .into_iter()
        .map(|hi| {
            let (_, hi) = primitive(&hi);
            if is_negative(&hi.LC()) {
                hi.neg()
            } else {
                hi
            }
        })
        .collect())
}

/// The content (with the sign of the leading coefficient) and the
/// irreducible factors with their multiplicities of a polynomial over `ZZ`,
/// sorted as SymPy sorts them (`dmp_zz_factor`).
pub fn zz_factor(f: &Poly) -> PolyResult<(Int, Factors)> {
    let ring: Ring = Arc::clone(&f.ring);
    if ring.ngens == 1 {
        let (coefficient, dense) = dup_zz_factor(&to_dense(f), &ZZ)?;
        return Ok((
            coefficient,
            dense.iter().map(|(g, k)| (from_dense(g, &ring), *k)).collect(),
        ));
    }
    if f.is_zero() {
        return Ok((Int::ZERO, Vec::new()));
    }
    let (mut cont, mut g) = primitive(f);
    if is_negative(&g.LC()) {
        cont = -cont;
        g = g.neg();
    }
    if (0..ring.ngens).all(|i| degree(&g, i) <= 0) {
        return Ok((cont, Vec::new()));
    }
    let r: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::sympy(ZZ, ring.names.clone());
    let inner_ring: Ring = PolyRing::new(ZZ, ring.names[1..].to_vec(), MonomialOrder::Lex)?;
    let content_first: Poly = x0_content(&g, &r);
    if !r.is_one(&content_first) {
        g = g.exquo(&content_first)?;
    }
    let mut factors: Factors = Vec::new();
    if degree(&g, 0) > 0 {
        g = sqf_part(&g)?;
        factors = trial_division(f, &zz_wang(&g, None, 0)?)?;
    }
    for (h, k) in zz_factor(&drop_first(&content_first, &inner_ring))?.1 {
        factors.insert(0, (add_first(&h, &ring), k));
    }
    Ok((cont, sorted(factors)))
}

/// The gcd of the coefficients in `x0` (polynomials in the other
/// variables, the sign of the ground leading coefficient positive).
fn x0_content(f: &Poly, r: &SparsePolynomialRing<IntegerRing>) -> Poly {
    let mut coefficients: BTreeMap<u32, Terms<Int>> = BTreeMap::new();
    for (m, c) in &f.terms {
        let mut key: Monomial = vec![0];
        key.extend_from_slice(&m[1..]);
        coefficients.entry(m[0]).or_default().insert(key, c.clone());
    }
    let ordered: Vec<Poly> = coefficients
        .into_values()
        .rev()
        .map(|terms| f.ring.from_dict(terms))
        .collect();
    let mut content: Poly = ordered[0].clone();
    for coefficient in &ordered[1..] {
        content = r.gcd(&content, coefficient);
        if r.is_one(&content) {
            break;
        }
    }
    if !content.is_zero() && is_negative(&content.LC()) {
        content = content.neg();
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctest() {
        // 2 x**3 y - 2 x y**3 + 4 x**2 - 4 y**2 = 2 (x - y)(x + y)(x y + 2)
        let r: Ring = PolyRing::new(ZZ, vec!["x".to_string(), "y".to_string()], MonomialOrder::Lex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let f = x
            .pow(3)
            .mul(y)
            .mul_ground(&int(2))
            .sub(&x.mul(&y.pow(3)).mul_ground(&int(2)))
            .add(&x.pow(2).mul_ground(&int(4)))
            .sub(&y.pow(2).mul_ground(&int(4)));
        let (c, factors) = zz_factor(&f).unwrap();
        assert_eq!(c, int(2));
        let shown: Vec<(String, usize)> = factors.iter().map(|(h, k)| (h.to_string(), *k)).collect();
        assert_eq!(
            shown,
            vec![
                ("x - y".to_string(), 1),
                ("x + y".to_string(), 1),
                ("x*y + 2".to_string(), 1)
            ]
        );
    }

    #[test]
    fn three_variables_with_powers() {
        let r: Ring = PolyRing::new(
            ZZ,
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            MonomialOrder::Lex,
        )
        .unwrap();
        let g = r.gens();
        let (x, y, z) = (&g[0], &g[1], &g[2]);
        let a = x.mul(y).add(z).add(&r.one());
        let b = x.sub(&z.pow(2)).mul_ground(&int(3)).add(y);
        let c = y.mul(z).sub(&r.from_int(2));
        let f = a.pow(2).mul(&b).mul(&c).mul_ground(&int(-6));
        let (coefficient, factors) = zz_factor(&f).unwrap();
        assert_eq!(coefficient, int(-6));
        let mut product: Poly = r.from_int(-6);
        for (h, k) in &factors {
            product = product.mul(&h.pow(*k as u64));
        }
        assert_eq!(product, f);
        assert_eq!(factors.len(), 3);
    }
}
