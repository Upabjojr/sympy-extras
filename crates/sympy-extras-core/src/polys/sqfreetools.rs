//! Port of `sympy_extras/polys/sqfreetools.py`: square-free decomposition
//! of dense univariate polynomials.
//!
//! Over a domain of characteristic zero, Yun's algorithm: with `g = gcd(f,
//! f')`, `w = f/g` and `y = f'/g`, the factor of multiplicity `i` is
//! `gcd(w, y - w')` at step `i`, with `w` and `y` replaced by their
//! quotients. Over `GF(p)` the derivative may vanish (`f` is then a
//! polynomial in `x**p`, whose `p`-th root is taken by dropping the
//! exponents), so the factors of multiplicities divisible by `p` are found
//! recursively.
//!
//! References
//! ==========
//!
//! 1. D. Y. Y. Yun, *On square-free decomposition algorithms*, SYMSAC '76,
//!    26-35.
//! 2. K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//!    Algebra*, Kluwer, 1992, Section 8.2.

use super::densearith::{dup_exquo, dup_mul, dup_mul_ground, dup_sub};
use super::densebasic::dup_degree;
use super::densetools::{dup_decompose_content, dup_diff, dup_monic_field, dup_primitive};
use super::domains::Domain;
use super::euclidtools::dup_gcd;

/// A polynomial with its multiplicity in a factorisation.
pub type Factor<T> = (Vec<T>, usize);

/// The exact quotient of two polynomials known to divide.
fn exquo<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    dup_exquo(f, g, k).expect("an exact division of the square-free decomposition")
}

/// The content (with the sign of the leading coefficient in an ordered
/// domain, the leading coefficient over a field) and the primitive, or
/// monic, part.
pub fn dup_sqf_norm_content<K: Domain>(f: &[K::Elem], k: &K) -> (K::Elem, Vec<K::Elem>) {
    if f.is_empty() {
        return (k.zero(), Vec::new());
    }
    if k.is_field() {
        return (f[0].clone(), dup_monic_field(f, k));
    }
    dup_decompose_content(f, k)
}

/// The content and the square-free factors with their multiplicities, in
/// increasing order of multiplicity: `f = content * prod g_i**i`, each
/// `g_i` primitive (monic over a field) and square-free, pairwise coprime.
pub fn dup_sqf_list<K: Domain>(f: &[K::Elem], k: &K) -> (K::Elem, Vec<Factor<K::Elem>>) {
    let (content, part) = dup_sqf_norm_content(f, k);
    if dup_degree(&part) <= 0 {
        return (content, Vec::new());
    }
    if k.characteristic() == 0 {
        return (content, yun(&part, k));
    }
    let p: u64 = k.characteristic();
    (content, sqf_list_positive_characteristic(&part, k, p))
}

fn yun<K: Domain>(f: &[K::Elem], k: &K) -> Vec<Factor<K::Elem>> {
    let derivative: Vec<K::Elem> = dup_diff(f, 1, k);
    let g: Vec<K::Elem> = dup_gcd(f, &derivative, k);
    if dup_degree(&g) <= 0 {
        return vec![(f.to_vec(), 1)];
    }
    let mut w: Vec<K::Elem> = exquo(f, &g, k);
    let mut y: Vec<K::Elem> = exquo(&derivative, &g, k);
    let mut factors: Vec<Factor<K::Elem>> = Vec::new();
    let mut i: usize = 1;
    while dup_degree(&w) > 0 {
        let z: Vec<K::Elem> = dup_sub(&y, &dup_diff(&w, 1, k), k);
        let h: Vec<K::Elem> = dup_gcd(&w, &z, k);
        if dup_degree(&h) > 0 {
            w = exquo(&w, &h, k);
            y = exquo(&z, &h, k);
            factors.push((h, i));
        } else {
            y = z;
        }
        i += 1;
    }
    factors
}

/// `g` with `g(x)**p = f(x)` for `f` a polynomial in `x**p` over a prime
/// field (the coefficients are their own `p`-th roots).
fn pth_root<T: Clone + std::fmt::Debug>(f: &[T], p: u64) -> Vec<T> {
    let n: usize = f.len() - 1;
    let step: usize = p as usize;
    assert!(n % step == 0, "{:?} is not a polynomial in x**{}", f, p);
    (0..n + 1).step_by(step).map(|i| f[i].clone()).collect()
}

fn sqf_list_positive_characteristic<K: Domain>(f: &[K::Elem], k: &K, p: u64) -> Vec<Factor<K::Elem>> {
    let derivative: Vec<K::Elem> = dup_diff(f, 1, k);
    let mut factors: Vec<Factor<K::Elem>> = Vec::new();
    if derivative.is_empty() {
        // f = g(x**p) = g(x)**p over GF(p)
        let root: Vec<K::Elem> = pth_root(f, p);
        for (h, i) in sqf_list_positive_characteristic(&root, k, p) {
            factors.push((h, i * p as usize));
        }
        return factors;
    }
    let mut g: Vec<K::Elem> = dup_gcd(f, &derivative, k);
    let mut w: Vec<K::Elem> = exquo(f, &g, k);
    // w is the product of the factors with multiplicity prime to p, each once
    let mut i: usize = 1;
    while dup_degree(&w) > 0 {
        let g_w: Vec<K::Elem> = dup_gcd(&g, &w, k);
        let h: Vec<K::Elem> = exquo(&w, &g_w, k);
        if dup_degree(&h) > 0 {
            factors.push((h, i));
        }
        g = exquo(&g, &g_w, k);
        w = g_w;
        i += 1;
    }
    if dup_degree(&g) > 0 {
        // the remaining factors have multiplicities divisible by p
        let root: Vec<K::Elem> = pth_root(&g, p);
        for (h, j) in sqf_list_positive_characteristic(&root, k, p) {
            factors.push((h, j * p as usize));
        }
    }
    factors.sort_by_key(|item| item.1);
    factors
}

/// The product of the square-free factors (primitive over a ring, monic
/// over a field).
pub fn dup_sqf_part<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() {
        return Vec::new();
    }
    let (_, factors) = dup_sqf_list(f, k);
    let mut result: Vec<K::Elem> = vec![k.one()];
    for (g, _) in &factors {
        result = dup_mul(&result, g, k);
    }
    if k.is_field() {
        return dup_monic_field(&result, k);
    }
    let (_, part) = dup_primitive(&result, k);
    if !k.is_ordered() || !k.is_negative(&part[0]) {
        part
    } else {
        dup_mul_ground(&part, &k.neg(&k.one()), k)
    }
}

/// Whether `f` is square-free (the zero polynomial is not).
pub fn dup_sqf_p<K: Domain>(f: &[K::Elem], k: &K) -> bool {
    if f.is_empty() {
        return false;
    }
    if dup_degree(f) <= 0 {
        return true;
    }
    dup_degree(&dup_gcd(f, &dup_diff(f, 1, k), k)) <= 0
}

#[cfg(test)]
mod tests {
    use super::super::densebasic::testing::zz;
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;
    use crate::polys::domains::{GF, ZZ};

    #[test]
    fn doctests() {
        assert_eq!(
            dup_sqf_list(&zz(&[1, -4, 6, -4, 1]), &ZZ),
            (Int::from(1), vec![(zz(&[1, -1]), 4)])
        );
        assert_eq!(
            dup_sqf_list(&zz(&[2, 0, -4, 0, 2]), &ZZ),
            (Int::from(2), vec![(zz(&[1, 0, -1]), 2)])
        );
        assert_eq!(dup_sqf_part(&zz(&[1, -4, 6, -4, 1]), &ZZ), zz(&[1, -1]));
        assert!(dup_sqf_p(&zz(&[1, 0, -1]), &ZZ));
        assert!(!dup_sqf_p(&zz(&[1, 0, 0]), &ZZ));
        assert_eq!(
            dup_sqf_list(&[1, 0, 0, 0, 0, 0, 1], &GF(3).unwrap()),
            (1, vec![(vec![1, 0, 1], 3)])
        );
    }
}
