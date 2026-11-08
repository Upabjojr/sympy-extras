//! Port of `sympy_extras/polys/densearith.py`: dense univariate
//! polynomials over a typed domain, the arithmetic.
//!
//! Addition, multiplication (schoolbook; Karatsuba above a threshold),
//! powers, division with remainder over a field, pseudo-division over a
//! ring, exact division, and the operations with a ground element.
//!
//! References
//! ==========
//!
//! 1. D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
//!    Addison-Wesley, 1998, Section 4.6.1 (division of polynomials).
//! 2. K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//!    Algebra*, Kluwer, 1992, Section 2.5 (pseudo-division).

use crate::error::{value_error, zero_division, PolyResult};

use super::densebasic::dup_strip_vec;
use super::domains::Domain;

/// The degree above which multiplication uses Karatsuba's algorithm.
pub const KARATSUBA_THRESHOLD: usize = 32;

pub fn dup_neg<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    f.iter().map(|c| k.neg(c)).collect()
}

/// The coefficientwise operation of `f` and `g` aligned at the constant
/// terms, the leading part of the longer one transformed by `lead`.
fn combine<K: Domain>(
    f: &[K::Elem],
    g: &[K::Elem],
    k: &K,
    op: fn(&K, &K::Elem, &K::Elem) -> K::Elem,
    lead_g: fn(&K, &K::Elem) -> K::Elem,
) -> Vec<K::Elem> {
    let (df, dg) = (f.len(), g.len());
    if df == dg {
        let result: Vec<K::Elem> = f.iter().zip(g).map(|(a, b)| op(k, a, b)).collect();
        return dup_strip_vec(result, k);
    }
    let mut result: Vec<K::Elem> = Vec::with_capacity(df.max(dg));
    if df > dg {
        result.extend_from_slice(&f[..df - dg]);
        result.extend(f[df - dg..].iter().zip(g).map(|(a, b)| op(k, a, b)));
    } else {
        result.extend(g[..dg - df].iter().map(|b| lead_g(k, b)));
        result.extend(f.iter().zip(&g[dg - df..]).map(|(a, b)| op(k, a, b)));
    }
    result
}

pub fn dup_add<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() {
        return g.to_vec();
    }
    if g.is_empty() {
        return f.to_vec();
    }
    combine(f, g, k, |k, a, b| k.add(a, b), |_, b| b.clone())
}

pub fn dup_sub<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if g.is_empty() {
        return f.to_vec();
    }
    if f.is_empty() {
        return dup_neg(g, k);
    }
    combine(f, g, k, |k, a, b| k.sub(a, b), |k, b| k.neg(b))
}

/// `f + c x**n`.
pub fn dup_add_term<K: Domain>(f: &[K::Elem], c: &K::Elem, n: usize, k: &K) -> Vec<K::Elem> {
    if k.is_zero(c) {
        return f.to_vec();
    }
    let m: usize = f.len();
    if n >= m {
        let mut result: Vec<K::Elem> = Vec::with_capacity(n + 1);
        result.push(c.clone());
        result.resize(n - m + 1, k.zero());
        result.extend_from_slice(f);
        return result;
    }
    let mut result: Vec<K::Elem> = f.to_vec();
    result[m - 1 - n] = k.add(&result[m - 1 - n], c);
    dup_strip_vec(result, k)
}

pub fn dup_sub_term<K: Domain>(f: &[K::Elem], c: &K::Elem, n: usize, k: &K) -> Vec<K::Elem> {
    dup_add_term(f, &k.neg(c), n, k)
}

/// `f c x**n`.
pub fn dup_mul_term<K: Domain>(f: &[K::Elem], c: &K::Elem, n: usize, k: &K) -> Vec<K::Elem> {
    if f.is_empty() || k.is_zero(c) {
        return Vec::new();
    }
    let mut result: Vec<K::Elem> = f.iter().map(|a| k.mul(a, c)).collect();
    result.resize(f.len() + n, k.zero());
    result
}

pub fn dup_add_ground<K: Domain>(f: &[K::Elem], c: &K::Elem, k: &K) -> Vec<K::Elem> {
    dup_add_term(f, c, 0, k)
}

pub fn dup_sub_ground<K: Domain>(f: &[K::Elem], c: &K::Elem, k: &K) -> Vec<K::Elem> {
    dup_add_term(f, &k.neg(c), 0, k)
}

pub fn dup_mul_ground<K: Domain>(f: &[K::Elem], c: &K::Elem, k: &K) -> Vec<K::Elem> {
    if f.is_empty() || k.is_zero(c) {
        return Vec::new();
    }
    let result: Vec<K::Elem> = f.iter().map(|a| k.mul(a, c)).collect();
    dup_strip_vec(result, k)
}

/// The coefficients divided by `c` with the domain's `quo` (truncated in
/// `ZZ`).
pub fn dup_quo_ground<K: Domain>(f: &[K::Elem], c: &K::Elem, k: &K) -> PolyResult<Vec<K::Elem>> {
    if k.is_zero(c) {
        return Err(zero_division("division by zero"));
    }
    let mut result: Vec<K::Elem> = Vec::with_capacity(f.len());
    for a in f {
        result.push(k.quo(a, c)?);
    }
    Ok(dup_strip_vec(result, k))
}

/// The coefficients divided exactly by `c`.
pub fn dup_exquo_ground<K: Domain>(f: &[K::Elem], c: &K::Elem, k: &K) -> PolyResult<Vec<K::Elem>> {
    if k.is_zero(c) {
        return Err(zero_division("division by zero"));
    }
    let mut result: Vec<K::Elem> = Vec::with_capacity(f.len());
    for a in f {
        result.push(k.exquo(a, c)?);
    }
    Ok(dup_strip_vec(result, k))
}

/// `f x**n`.
pub fn dup_lshift<K: Domain>(f: &[K::Elem], n: usize, k: &K) -> Vec<K::Elem> {
    if f.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<K::Elem> = f.to_vec();
    result.resize(f.len() + n, k.zero());
    result
}

/// `f / x**n` (the low terms dropped).
pub fn dup_rshift<K: Domain>(f: &[K::Elem], n: usize, _k: &K) -> Vec<K::Elem> {
    if n < f.len() {
        f[..f.len() - n].to_vec()
    } else {
        Vec::new()
    }
}

fn schoolbook<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    let mut result: Vec<K::Elem> = vec![k.zero(); f.len() + g.len() - 1];
    for (i, a) in f.iter().enumerate() {
        if k.is_zero(a) {
            continue;
        }
        for (j, b) in g.iter().enumerate() {
            result[i + j] = k.add(&result[i + j], &k.mul(a, b));
        }
    }
    dup_strip_vec(result, k)
}

pub fn dup_mul<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() || g.is_empty() {
        return Vec::new();
    }
    if k.has_dense_kernels() {
        return k.dense_mul(f, g);
    }
    if std::ptr::eq(f, g) {
        return dup_sqr(f, k);
    }
    let (df, dg) = (f.len() - 1, g.len() - 1);
    if df.min(dg) < KARATSUBA_THRESHOLD {
        return schoolbook(f, g, k);
    }
    // Karatsuba: f = f1 x**n + f0, g = g1 x**n + g0
    let n: usize = df.max(dg).div_ceil(2);
    let low_part = |h: &[K::Elem]| -> Vec<K::Elem> {
        if n < h.len() {
            dup_strip_vec(h[h.len() - n..].to_vec(), k)
        } else {
            dup_strip_vec(h.to_vec(), k)
        }
    };
    let (f1, f0) = (dup_rshift(f, n, k), low_part(f));
    let (g1, g0) = (dup_rshift(g, n, k), low_part(g));
    let low: Vec<K::Elem> = dup_mul(&f0, &g0, k);
    let high: Vec<K::Elem> = dup_mul(&f1, &g1, k);
    let cross: Vec<K::Elem> = dup_mul(&dup_add(&f1, &f0, k), &dup_add(&g1, &g0, k), k);
    let middle: Vec<K::Elem> = dup_sub(&dup_sub(&cross, &low, k), &high, k);
    dup_add(
        &dup_add(&dup_lshift(&high, 2 * n, k), &dup_lshift(&middle, n, k), k),
        &low,
        k,
    )
}

pub fn dup_sqr<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() {
        return Vec::new();
    }
    if k.has_dense_kernels() {
        return k.dense_mul(f, f);
    }
    let df: usize = f.len() - 1;
    if df >= KARATSUBA_THRESHOLD {
        return schoolbook(f, f, k);
    }
    let mut result: Vec<K::Elem> = vec![k.zero(); 2 * df + 1];
    for (i, a) in f.iter().enumerate() {
        if k.is_zero(a) {
            continue;
        }
        result[2 * i] = k.add(&result[2 * i], &k.mul(a, a));
        for j in (i + 1)..(df + 1) {
            let b: &K::Elem = &f[j];
            if !k.is_zero(b) {
                let twice: K::Elem = k.mul(a, b);
                result[i + j] = k.add(&result[i + j], &k.add(&twice, &twice));
            }
        }
    }
    dup_strip_vec(result, k)
}

/// `f**n` by repeated squaring.
pub fn dup_pow<K: Domain>(f: &[K::Elem], n: usize, k: &K) -> Vec<K::Elem> {
    if n == 0 {
        return vec![k.one()];
    }
    let mut n: usize = n;
    let mut result: Vec<K::Elem> = vec![k.one()];
    let mut base: Vec<K::Elem> = f.to_vec();
    while n != 0 {
        if n & 1 == 1 {
            result = dup_mul(&result, &base, k);
        }
        n >>= 1;
        if n != 0 {
            base = dup_sqr(&base, k);
        }
    }
    result
}

/// The quotient and remainder of `f` by `g` over a field (over a ring, the
/// division proceeds while the leading coefficients divide exactly and
/// fails with `Value` otherwise).
pub fn dup_div<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<(Vec<K::Elem>, Vec<K::Elem>)> {
    if g.is_empty() {
        return Err(zero_division("polynomial division by zero"));
    }
    if k.has_dense_kernels() {
        return k.dense_div(f, g);
    }
    if f.len() < g.len() {
        return Ok((Vec::new(), f.to_vec()));
    }
    let (df, dg) = (f.len() - 1, g.len() - 1);
    let lc: &K::Elem = &g[0];
    // over a field the leading coefficient is inverted once
    let inverse: K::Elem = if k.is_field() { k.invert(lc)? } else { k.one() };
    let mut quotient: Vec<K::Elem> = vec![k.zero(); df - dg + 1];
    let mut remainder: Vec<K::Elem> = f.to_vec();
    for i in 0..(df - dg + 1) {
        if k.is_zero(&remainder[i]) {
            continue;
        }
        let q: K::Elem = if k.is_field() {
            k.mul(&remainder[i], &inverse)
        } else {
            k.exquo(&remainder[i], lc)?
        };
        for j in 1..(dg + 1) {
            remainder[i + j] = k.sub(&remainder[i + j], &k.mul(&q, &g[j]));
        }
        quotient[i] = q;
        remainder[i] = k.zero();
    }
    let tail: Vec<K::Elem> = remainder.split_off(df - dg + 1);
    Ok((dup_strip_vec(quotient, k), dup_strip_vec(tail, k)))
}

pub fn dup_quo<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    Ok(dup_div(f, g, k)?.0)
}

pub fn dup_rem<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    Ok(dup_div(f, g, k)?.1)
}

/// The exact quotient; `Value` when `g` does not divide `f`.
pub fn dup_exquo<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    let (q, r) = dup_div(f, g, k)?;
    if !r.is_empty() {
        return Err(value_error(format!("{:?} does not divide {:?}", g, f)));
    }
    Ok(q)
}

/// The pseudo-quotient and pseudo-remainder:
/// `lc(g)**(df - dg + 1) f = q g + r` with `deg r < deg g` (`q = 0`, `r = f` when `deg f < deg g`),
/// computed without divisions in the domain.
pub fn dup_pdiv<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<(Vec<K::Elem>, Vec<K::Elem>)> {
    if g.is_empty() {
        return Err(zero_division("polynomial division by zero"));
    }
    if f.len() < g.len() {
        return Ok((Vec::new(), f.to_vec()));
    }
    let dg: usize = g.len() - 1;
    let lc: &K::Elem = &g[0];
    let mut q: Vec<K::Elem> = Vec::new();
    let mut r: Vec<K::Elem> = f.to_vec();
    let mut n: usize = f.len() - g.len() + 1;
    while r.len() > dg {
        // r = lc r - lc(r) x**(dr - dg) g, q = lc q + lc(r) x**(dr - dg)
        let shift: usize = r.len() - 1 - dg;
        let lr: K::Elem = r[0].clone();
        q = dup_add_term(&dup_mul_ground(&q, lc, k), &lr, shift, k);
        r = dup_sub(&dup_mul_ground(&r, lc, k), &dup_mul_term(g, &lr, shift, k), k);
        n -= 1;
    }
    // the remaining factor of lc so that the multiplier is exactly lc**(df - dg + 1)
    if n > 0 {
        let scale: K::Elem = k.pow(lc, n as u64);
        q = dup_mul_ground(&q, &scale, k);
        r = dup_mul_ground(&r, &scale, k);
    }
    Ok((q, r))
}

/// The pseudo-remainder of `f` by `g`.
pub fn dup_prem<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    Ok(dup_pdiv(f, g, k)?.1)
}

/// The pseudo-quotient of `f` by `g`.
pub fn dup_pquo<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    Ok(dup_pdiv(f, g, k)?.0)
}

/// The pseudo-quotient when the pseudo-remainder is zero; `Value`
/// otherwise.
pub fn dup_pexquo<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    let (q, r) = dup_pdiv(f, g, k)?;
    if !r.is_empty() {
        return Err(value_error(format!("{:?} does not divide {:?}", g, f)));
    }
    Ok(q)
}

/// The largest absolute value of a coefficient (ordered domains).
pub fn dup_max_norm<K: Domain>(f: &[K::Elem], k: &K) -> K::Elem {
    let mut best: K::Elem = match f.first() {
        Some(c) => k.abs(c),
        None => return k.zero(),
    };
    for c in &f[1..] {
        let a: K::Elem = k.abs(c);
        if k.is_negative(&k.sub(&best, &a)) {
            best = a;
        }
    }
    best
}

/// The sum of the absolute values of the coefficients.
pub fn dup_l1_norm<K: Domain>(f: &[K::Elem], k: &K) -> K::Elem {
    let mut total: K::Elem = k.zero();
    for c in f {
        total = k.add(&total, &k.abs(c));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::super::densebasic::testing::{q, qq, zz};
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;
    use crate::polys::domains::{QQ, ZZ};

    #[test]
    fn doctests() {
        assert_eq!(dup_add(&zz(&[1, 2]), &zz(&[1, -2]), &ZZ), zz(&[2, 0]));
        assert_eq!(dup_mul(&zz(&[1, 1]), &zz(&[1, -1]), &ZZ), zz(&[1, 0, -1]));
        assert_eq!(
            dup_div(&qq(&[1, 0, -1]), &qq(&[2, 1]), &QQ).unwrap(),
            (vec![q(1, 2), q(-1, 4)], vec![q(-3, 4)])
        );
        assert_eq!(
            dup_pdiv(&zz(&[1, 0, -1]), &zz(&[2, 1]), &ZZ).unwrap(),
            (zz(&[2, -1]), zz(&[-3]))
        );
        assert_eq!(dup_exquo(&zz(&[1, 0, -1]), &zz(&[1, 1]), &ZZ).unwrap(), zz(&[1, -1]));
        assert_eq!(dup_pow(&zz(&[1, 1]), 3, &ZZ), zz(&[1, 3, 3, 1]));
        assert!(dup_exquo(&zz(&[1, 0, 1]), &zz(&[1, 1]), &ZZ).is_err());
        assert!(dup_div(&zz(&[1, 0, 1]), &[], &ZZ).is_err());
    }

    #[test]
    fn karatsuba_agrees_with_schoolbook() {
        let f: Vec<_> = (0..70).map(|i| q((i * 7 % 11) - 5, 1 + i % 3)).collect();
        let g: Vec<_> = (0..45).map(|i| q((i * 5 % 13) - 6, 1 + i % 4)).collect();
        assert_eq!(dup_mul(&f, &g, &QQ), schoolbook(&f, &g, &QQ));
        assert_eq!(dup_sqr(&g, &QQ), schoolbook(&g, &g, &QQ));
    }
}
