//! Port of `sympy_extras/polys/densebasic.py`: dense univariate
//! polynomials over a typed domain, the representation.
//!
//! A polynomial in `K[x]` is a vector of coefficients, highest degree first
//! (`[1, 0, -2]` is `x**2 - 2`), without leading zeros; the zero polynomial
//! is the empty vector. Functions take slices and return new vectors.

use std::collections::BTreeMap;

use crate::error::PolyResult;
use crate::numbers::*;

use super::domains::{ConvertFrom, Domain};

/// `f` without leading zeros (the zero polynomial is `[]`).
pub fn dup_strip<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    let i: usize = f.iter().take_while(|c| k.is_zero(c)).count();
    f[i..].to_vec()
}

/// [`dup_strip`] on an owned vector, without copying when nothing is
/// stripped.
pub(crate) fn dup_strip_vec<K: Domain>(f: Vec<K::Elem>, k: &K) -> Vec<K::Elem> {
    let i: usize = f.iter().take_while(|c| k.is_zero(c)).count();
    if i == 0 {
        return f;
    }
    f[i..].to_vec()
}

pub fn dup_zero<K: Domain>(_k: &K) -> Vec<K::Elem> {
    Vec::new()
}

pub fn dup_one<K: Domain>(k: &K) -> Vec<K::Elem> {
    vec![k.one()]
}

pub fn dup_constant<K: Domain>(c: &K::Elem, k: &K) -> Vec<K::Elem> {
    if k.is_zero(c) {
        Vec::new()
    } else {
        vec![c.clone()]
    }
}

/// `c x**n`.
pub fn dup_monomial<K: Domain>(c: &K::Elem, n: usize, k: &K) -> Vec<K::Elem> {
    if k.is_zero(c) {
        return Vec::new();
    }
    let mut result: Vec<K::Elem> = Vec::with_capacity(n + 1);
    result.push(c.clone());
    result.resize(n + 1, k.zero());
    result
}

pub fn dup_is_zero<T>(f: &[T]) -> bool {
    f.is_empty()
}

pub fn dup_is_one<K: Domain>(f: &[K::Elem], k: &K) -> bool {
    f.len() == 1 && k.is_one(&f[0])
}

pub fn dup_is_constant<T>(f: &[T]) -> bool {
    f.len() <= 1
}

/// The degree (`-1` for the zero polynomial).
pub fn dup_degree<T>(f: &[T]) -> isize {
    f.len() as isize - 1
}

/// The leading coefficient (`0` for the zero polynomial).
#[allow(non_snake_case)]
pub fn dup_LC<K: Domain>(f: &[K::Elem], k: &K) -> K::Elem {
    match f.first() {
        Some(c) => c.clone(),
        None => k.zero(),
    }
}

/// The trailing (constant) coefficient.
#[allow(non_snake_case)]
pub fn dup_TC<K: Domain>(f: &[K::Elem], k: &K) -> K::Elem {
    match f.last() {
        Some(c) => c.clone(),
        None => k.zero(),
    }
}

/// A sequence of coefficients (highest degree first) as a polynomial:
/// leading zeros removed.
pub fn dup_normal<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    dup_strip(f, k)
}

/// A list of integers (highest degree first) as a polynomial over `K`.
pub fn dup_from_list<K: Domain>(f: &[Int], k: &K) -> Vec<K::Elem> {
    let coefficients: Vec<K::Elem> = f.iter().map(|c| k.from_integer(c)).collect();
    dup_strip_vec(coefficients, k)
}

/// `{exponent: coefficient}` as a polynomial.
pub fn dup_from_dict<K: Domain>(f: &BTreeMap<usize, K::Elem>, k: &K) -> Vec<K::Elem> {
    let n: usize = match f.keys().next_back() {
        Some(n) => *n,
        None => return Vec::new(),
    };
    let mut result: Vec<K::Elem> = vec![k.zero(); n + 1];
    for (exponent, c) in f {
        result[n - exponent] = c.clone();
    }
    dup_strip_vec(result, k)
}

/// The nonzero terms as `{exponent: coefficient}`.
pub fn dup_to_dict<K: Domain>(f: &[K::Elem], k: &K) -> BTreeMap<usize, K::Elem> {
    let n: usize = f.len();
    f.iter()
        .enumerate()
        .filter(|(_, c)| !k.is_zero(c))
        .map(|(i, c)| (n - 1 - i, c.clone()))
        .collect()
}

/// The nonzero terms `(exponent, coefficient)`, highest first.
pub fn dup_terms<K: Domain>(f: &[K::Elem], k: &K) -> Vec<(usize, K::Elem)> {
    let n: usize = f.len();
    f.iter()
        .enumerate()
        .filter(|(_, c)| !k.is_zero(c))
        .map(|(i, c)| (n - 1 - i, c.clone()))
        .collect()
}

/// The polynomial over `K0` as a polynomial over `K1` (`Value` when a
/// coefficient has no image).
pub fn dup_convert<K0: Domain, K1: ConvertFrom<K0>>(f: &[K0::Elem], k0: &K0, k1: &K1) -> PolyResult<Vec<K1::Elem>> {
    let mut coefficients: Vec<K1::Elem> = Vec::with_capacity(f.len());
    for c in f {
        coefficients.push(k1.convert(c, k0)?);
    }
    Ok(dup_strip_vec(coefficients, k1))
}

/// The reciprocal polynomial `x**deg(f) f(1/x)`.
pub fn dup_reverse<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    let reversed: Vec<K::Elem> = f.iter().rev().cloned().collect();
    dup_strip_vec(reversed, k)
}

pub fn dup_copy<T: Clone>(f: &[T]) -> Vec<T> {
    f.to_vec()
}

/// The terms of degree `m <= k < n`.
pub fn dup_slice<K: Domain>(f: &[K::Elem], m: usize, n: usize, k: &K) -> Vec<K::Elem> {
    let terms: BTreeMap<usize, K::Elem> = dup_to_dict(f, k)
        .into_iter()
        .filter(|(e, _)| m <= *e && *e < n)
        .collect();
    dup_from_dict(&terms, k)
}

#[cfg(test)]
pub(crate) mod testing {
    #[allow(unused_imports)]
    use crate::numbers::*;

    pub fn zz(f: &[i64]) -> Vec<Int> {
        f.iter().map(|c| Int::from(*c)).collect()
    }

    pub fn q(p: i64, d: i64) -> Rat {
        rat_from_parts(Int::from(p), Int::from(d))
    }

    pub fn qq(f: &[i64]) -> Vec<Rat> {
        f.iter().map(|c| q(*c, 1)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{qq, zz};
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;
    use crate::polys::domains::{QQ, ZZ};

    #[test]
    fn doctests() {
        assert_eq!(dup_strip(&zz(&[0, 0, 1, 2]), &ZZ), zz(&[1, 2]));
        assert_eq!(dup_degree(&zz(&[1, 0, -2])), 2);
        assert_eq!(dup_degree::<Int>(&[]), -1);
        let d: BTreeMap<usize, Int> = [(3, Int::from(1)), (0, Int::from(-1))].into_iter().collect();
        assert_eq!(dup_from_dict(&d, &ZZ), zz(&[1, 0, 0, -1]));
        assert_eq!(dup_to_dict(&zz(&[1, 0, 0, -1]), &ZZ), d);
        assert_eq!(dup_convert(&zz(&[1, -2]), &ZZ, &QQ).unwrap(), qq(&[1, -2]));
        assert_eq!(dup_slice(&zz(&[1, 2, 3, 4]), 1, 3, &ZZ), zz(&[2, 3, 0]));
        assert_eq!(dup_reverse(&zz(&[1, 2, 0]), &ZZ), zz(&[2, 1]));
        assert_eq!(dup_monomial(&Int::from(3), 2, &ZZ), zz(&[3, 0, 0]));
    }
}
