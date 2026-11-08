//! Port of `sympy_extras/polys/euclidtools.py`: Euclidean algorithms on
//! dense polynomials: gcds, resultants, subresultants.
//!
//! The Euclidean algorithm over a field ([`dup_gcd`], [`dup_gcdex`],
//! [`dup_invert`]), the subresultant polynomial remainder sequence over an
//! integral domain ([`dup_subresultants`], whose last element gives the gcd
//! by its primitive part), the resultant by the recursive identity on
//! pseudo-remainders ([`dup_resultant`]) and the discriminant.
//!
//! References
//! ==========
//!
//! 1. W. S. Brown, *The subresultant PRS algorithm*, ACM Transactions on
//!    Mathematical Software 4 (1978), 237-249.
//! 2. K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//!    Algebra*, Kluwer, 1992, Chapter 7.
//! 3. D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
//!    Addison-Wesley, 1998, Section 4.6.1.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/euclidtools.py):
// sympy/polys/euclidtools.py (dup_inner_subresultants, dup_prs_resultant)
// (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: typed; the sign of Res(f, g) for deg f < deg g corrected.
// ---------------------------------------------------------------------

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::densearith::{dup_div, dup_exquo, dup_exquo_ground, dup_mul, dup_mul_ground, dup_prem, dup_sub};
use super::densebasic::dup_degree;
use super::densetools::{dup_clear_denoms, dup_diff, dup_monic, dup_monic_field, dup_primitive};
use super::domains::{Domain, QQ, ZZ};

/// The pseudo-remainder by a nonzero polynomial, which cannot fail.
fn prem<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    dup_prem(f, g, k).expect("a nonzero divisor")
}

/// The monic gcd over a field by the Euclidean algorithm (each remainder
/// made monic, which keeps the coefficients small over `QQ`).
pub fn dup_euclid<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    let mut a: Vec<K::Elem> = dup_monic(f, k)?;
    let mut b: Vec<K::Elem> = dup_monic(g, k)?;
    while !b.is_empty() {
        let r: Vec<K::Elem> = dup_div(&a, &b, k)?.1;
        a = std::mem::replace(&mut b, dup_monic(&r, k)?);
    }
    Ok(a)
}

/// The Euclidean polynomial remainder sequence over a field.
pub fn dup_prs<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<Vec<K::Elem>>> {
    let mut sequence: Vec<Vec<K::Elem>> = vec![f.to_vec(), g.to_vec()];
    while !sequence[sequence.len() - 1].is_empty() {
        let n: usize = sequence.len();
        let r: Vec<K::Elem> = dup_div(&sequence[n - 2], &sequence[n - 1], k)?.1;
        sequence.push(r);
    }
    sequence.pop();
    Ok(sequence)
}

/// The subresultant polynomial remainder sequence of `f` and `g` over an
/// integral domain (Brown's algorithm): `r_{i+1} = prem(r_{i-1}, r_i) /
/// beta_i` with `beta_1 = (-1)**(delta_1 + 1)`, `psi_1 = -1` and, for
/// `i >= 2`, `psi_i = (-lc(r_{i-1}))**delta_{i-1} /
/// psi_{i-1}**(delta_{i-1} - 1)`, `beta_i = -lc(r_{i-1}) psi_i**delta_i`,
/// where `delta_i = deg r_{i-1} - deg r_i`; every division is exact. The
/// sequence ends with the last nonzero remainder.
pub fn dup_subresultants<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<Vec<K::Elem>> {
    if f.is_empty() {
        return if g.is_empty() { Vec::new() } else { vec![g.to_vec()] };
    }
    if g.is_empty() {
        return vec![f.to_vec()];
    }
    let (mut a, mut b): (Vec<K::Elem>, Vec<K::Elem>) = (f.to_vec(), g.to_vec());
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    let mut sequence: Vec<Vec<K::Elem>> = vec![a.clone(), b.clone()];
    let mut delta: usize = a.len() - b.len();
    let mut beta: K::Elem = if delta % 2 == 1 { k.one() } else { k.neg(&k.one()) };
    let mut psi: K::Elem = k.neg(&k.one());
    loop {
        let r: Vec<K::Elem> = prem(&a, &b, k);
        if r.is_empty() {
            break;
        }
        let r: Vec<K::Elem> = dup_exquo_ground(&r, &beta, k).expect("Brown's divisions are exact");
        sequence.push(r.clone());
        a = std::mem::replace(&mut b, r);
        let minus_lc: K::Elem = k.neg(&a[0]);
        psi = psi_update(&minus_lc, &psi, delta, k);
        delta = a.len() - b.len();
        beta = k.mul(&minus_lc, &k.pow(&psi, delta as u64));
    }
    sequence
}

/// `(-lc)**delta / psi**(delta - 1)`, an exact division (`psi` itself for
/// `delta = 0`).
fn psi_update<K: Domain>(minus_lc: &K::Elem, psi: &K::Elem, delta: usize, k: &K) -> K::Elem {
    if delta == 0 {
        return psi.clone();
    }
    if delta == 1 {
        return minus_lc.clone();
    }
    k.exquo(&k.pow(minus_lc, delta as u64), &k.pow(psi, (delta - 1) as u64))
        .expect("Brown's divisions are exact")
}

/// The greatest common divisor in canonical form: monic over a field (the
/// Euclidean algorithm; [`dup_qq_gcd`] is faster over `QQ`), and over an
/// integral domain the primitive part of the last subresultant times the
/// gcd of the contents, with the canonical unit of the domain (positive
/// leading coefficient over `ZZ`).
pub fn dup_gcd<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() && g.is_empty() {
        return Vec::new();
    }
    if k.is_field() {
        return dup_euclid(f, g, k).expect("the Euclidean algorithm does not fail over a field");
    }
    if f.is_empty() {
        return canonical(g, k);
    }
    if g.is_empty() {
        return canonical(f, k);
    }
    let (cf, pf) = dup_primitive(f, k);
    let (cg, pg) = dup_primitive(g, k);
    let content: K::Elem = k.gcd(&cf, &cg);
    let sequence: Vec<Vec<K::Elem>> = dup_subresultants(&pf, &pg, k);
    let last: &Vec<K::Elem> = sequence.last().expect("a nonempty sequence");
    let (_, part) = dup_primitive(last, k);
    dup_mul_ground(&canonical(&part, k), &content, k)
}

/// The monic gcd over `QQ` through the integers: the denominators cleared,
/// the subresultant gcd over `ZZ`, the result made monic.
pub fn dup_qq_gcd(f: &[Rat], g: &[Rat]) -> Vec<Rat> {
    let (_, fz) = dup_clear_denoms(f, &QQ);
    let (_, gz) = dup_clear_denoms(g, &QQ);
    let fi: Vec<Int> = fz.iter().map(|c| c.numerator().clone()).collect();
    let gi: Vec<Int> = gz.iter().map(|c| c.numerator().clone()).collect();
    let h: Vec<Int> = dup_gcd(&fi, &gi, &ZZ);
    let hq: Vec<Rat> = h.into_iter().map(Rat::from).collect();
    dup_monic_field(&hq, &QQ)
}

/// `f` times the canonical unit of its leading coefficient.
fn canonical<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() {
        return Vec::new();
    }
    let u: K::Elem = k.canonical_unit(&f[0]);
    if k.is_one(&u) {
        f.to_vec()
    } else {
        dup_mul_ground(f, &u, k)
    }
}

/// The least common multiple, in canonical form.
pub fn dup_lcm<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() || g.is_empty() {
        return Vec::new();
    }
    let h: Vec<K::Elem> = dup_gcd(f, g, k);
    let quotient: Vec<K::Elem> = dup_exquo(&dup_mul(f, g, k), &h, k).expect("the gcd divides the product");
    if k.is_field() {
        dup_monic_field(&quotient, k)
    } else {
        canonical(&quotient, k)
    }
}

/// `(s, t, h)` with `s f + t g = h` the monic gcd, over a field (the
/// extended Euclidean algorithm); `Value` over a ring.
pub fn dup_gcdex<K: Domain>(
    f: &[K::Elem],
    g: &[K::Elem],
    k: &K,
) -> PolyResult<(Vec<K::Elem>, Vec<K::Elem>, Vec<K::Elem>)> {
    if !k.is_field() {
        return Err(value_error(format!(
            "the extended Euclidean algorithm needs a field, got {}",
            k.name()
        )));
    }
    let (mut r0, mut r1): (Vec<K::Elem>, Vec<K::Elem>) = (f.to_vec(), g.to_vec());
    let (mut s0, mut s1): (Vec<K::Elem>, Vec<K::Elem>) = (vec![k.one()], Vec::new());
    let (mut t0, mut t1): (Vec<K::Elem>, Vec<K::Elem>) = (Vec::new(), vec![k.one()]);
    while !r1.is_empty() {
        let (q, r) = dup_div(&r0, &r1, k)?;
        r0 = std::mem::replace(&mut r1, r);
        let s2: Vec<K::Elem> = dup_sub(&s0, &dup_mul(&q, &s1, k), k);
        s0 = std::mem::replace(&mut s1, s2);
        let t2: Vec<K::Elem> = dup_sub(&t0, &dup_mul(&q, &t1, k), k);
        t0 = std::mem::replace(&mut t1, t2);
    }
    if r0.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let inverse: K::Elem = k.invert(&r0[0])?;
    Ok((
        dup_mul_ground(&s0, &inverse, k),
        dup_mul_ground(&t0, &inverse, k),
        dup_mul_ground(&r0, &inverse, k),
    ))
}

/// `(s, h)` with `s f = h (mod g)`, `h` the monic gcd.
pub fn dup_half_gcdex<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<(Vec<K::Elem>, Vec<K::Elem>)> {
    let (s, _, h) = dup_gcdex(f, g, k)?;
    Ok((s, h))
}

/// The inverse of `f` modulo `g` over a field; `Value` when they are not
/// coprime.
pub fn dup_invert<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    let (s, h) = dup_half_gcdex(f, g, k)?;
    if dup_degree(&h) != 0 {
        return Err(value_error(format!("{:?} and {:?} are not coprime", f, g)));
    }
    Ok(dup_div(&s, g, k)?.1)
}

/// The resultant, over any integral domain: with `m = deg f >= n = deg g`
/// and `r` the pseudo-remainder of `f` by `g` of degree `k` and content `c`
/// (`r = lc(g)**(m-n+1) f mod g`),
///
/// `Res(f, g) = (-1)**(m n) lc(g)**(m - k) c**n Res(g, r/c) / lc(g)**(n (m - n + 1))`,
///
/// the division being exact in the domain.
pub fn dup_resultant<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> K::Elem {
    if f.is_empty() || g.is_empty() {
        return k.zero();
    }
    let (m, n) = (f.len() - 1, g.len() - 1);
    if m < n {
        let value: K::Elem = dup_resultant(g, f, k);
        return if (m * n) % 2 == 1 { k.neg(&value) } else { value };
    }
    if n == 0 {
        return k.pow(&g[0], m as u64);
    }
    let r: Vec<K::Elem> = prem(f, g, k);
    if r.is_empty() {
        return k.zero();
    }
    let degree: usize = r.len() - 1;
    let (c, part) = dup_primitive(&r, k);
    let lc: &K::Elem = &g[0];
    let inner: K::Elem = dup_resultant(g, &part, k);
    let mut numerator: K::Elem = k.mul(&k.mul(&k.pow(lc, (m - degree) as u64), &k.pow(&c, n as u64)), &inner);
    if (m * n) % 2 == 1 {
        numerator = k.neg(&numerator);
    }
    k.exquo(&numerator, &k.pow(lc, (n * (m - n + 1)) as u64))
        .expect("the division of the resultant identity is exact")
}

/// The subresultant PRS of `f` and `g` (the one of higher degree first) and
/// the scalar subresultants `S` (`S[0] = 1`), with exact divisions only:
/// the last scalar is the resultant when the sequence ends with a
/// constant.
pub fn dup_inner_subresultants<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> (Vec<Vec<K::Elem>>, Vec<K::Elem>) {
    let (mut f, mut g): (Vec<K::Elem>, Vec<K::Elem>) = (f.to_vec(), g.to_vec());
    let (mut n, mut m): (isize, isize) = (dup_degree(&f), dup_degree(&g));
    if n < m {
        std::mem::swap(&mut f, &mut g);
        std::mem::swap(&mut n, &mut m);
    }
    if f.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if g.is_empty() {
        return (vec![f], vec![k.one()]);
    }
    let mut big_r: Vec<Vec<K::Elem>> = vec![f.clone(), g.clone()];
    let mut d: usize = (n - m) as usize;
    let mut b: K::Elem = k.pow(&k.neg(&k.one()), (d + 1) as u64);
    let mut h: Vec<K::Elem> = dup_mul_ground(&prem(&f, &g, k), &b, k);
    let mut lc: K::Elem = g[0].clone();
    let mut c: K::Elem = k.pow(&lc, d as u64);
    let mut big_s: Vec<K::Elem> = vec![k.one(), c.clone()];
    c = k.neg(&c);
    while !h.is_empty() {
        let degree: isize = dup_degree(&h);
        big_r.push(h.clone());
        f = std::mem::replace(&mut g, h);
        d = (m - degree) as usize;
        m = degree;
        b = k.mul(&k.neg(&lc), &k.pow(&c, d as u64));
        h = dup_exquo_ground(&prem(&f, &g, k), &b, k).expect("the subresultant divisions are exact");
        lc = g[0].clone();
        c = if d > 1 {
            k.exquo(&k.pow(&k.neg(&lc), d as u64), &k.pow(&c, (d - 1) as u64))
                .expect("the subresultant divisions are exact")
        } else {
            k.neg(&lc)
        };
        big_s.push(k.neg(&c));
    }
    (big_r, big_s)
}

/// The resultant by the subresultant PRS (no gcd computations), the
/// Sylvester determinant of `f` and `g` in this order.
pub fn dup_prs_resultant<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> K::Elem {
    if f.is_empty() || g.is_empty() {
        return k.zero();
    }
    let (big_r, big_s) = dup_inner_subresultants(f, g, k);
    if dup_degree(&big_r[big_r.len() - 1]) > 0 {
        return k.zero();
    }
    let mut value: K::Elem = big_s[big_s.len() - 1].clone();
    let (df, dg) = (dup_degree(f), dup_degree(g));
    if df < dg && (df * dg) % 2 == 1 {
        value = k.neg(&value);
    }
    value
}

/// `(-1)**(m (m - 1)/2) Res(f, f') / lc(f)` for `m = deg f` (`0` for a
/// constant).
pub fn dup_discriminant<K: Domain>(f: &[K::Elem], k: &K) -> K::Elem {
    let m: isize = dup_degree(f);
    if m <= 0 {
        return k.zero();
    }
    let mut value: K::Elem = dup_resultant(f, &dup_diff(f, 1, k), k);
    if (m * (m - 1) / 2) % 2 == 1 {
        value = k.neg(&value);
    }
    k.exquo(&value, &f[0])
        .expect("the leading coefficient divides the resultant")
}

#[cfg(test)]
mod tests {
    use super::super::densebasic::testing::{q, qq, zz};
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;
    use crate::polys::domains::GF;

    #[test]
    fn doctests() {
        assert_eq!(dup_gcd(&zz(&[1, 0, -1]), &zz(&[1, -2, 1]), &ZZ), zz(&[1, -1]));
        assert_eq!(dup_gcd(&zz(&[2, 0, -2]), &zz(&[4, -8, 4]), &ZZ), zz(&[2, -2]));
        assert_eq!(dup_resultant(&zz(&[1, 0, -2]), &zz(&[1, 0, 0, -3]), &ZZ), Int::from(1));
        assert_eq!(dup_discriminant(&zz(&[1, 1, 1]), &ZZ), Int::from(-3));
        assert_eq!(
            dup_gcdex(&qq(&[1, 0, -1]), &qq(&[1, -2]), &QQ).unwrap(),
            (vec![q(1, 3)], vec![q(-1, 3), q(-2, 3)], vec![q(1, 1)])
        );
        assert_eq!(dup_invert(&[1, 1], &[1, 0, 1], &GF(7).unwrap()).unwrap(), vec![3, 4]);
        assert_eq!(
            dup_prs_resultant(&zz(&[-7, -3]), &zz(&[-2, 6, 1, -6]), &ZZ),
            Int::from(1773)
        );
        assert!(dup_gcdex(&zz(&[1]), &zz(&[1]), &ZZ).is_err());
    }

    #[test]
    fn resultants_agree() {
        let polys: Vec<Vec<Int>> = vec![
            zz(&[3, -2, 0, 5]),
            zz(&[1, 4, 1]),
            zz(&[-2, 0, 7, 1, -1]),
            zz(&[5, 1]),
            zz(&[2]),
        ];
        for f in &polys {
            for g in &polys {
                assert_eq!(
                    dup_resultant(f, g, &ZZ),
                    dup_prs_resultant(f, g, &ZZ),
                    "{:?} {:?}",
                    f,
                    g
                );
            }
        }
    }
}
