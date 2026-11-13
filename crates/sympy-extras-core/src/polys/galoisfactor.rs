//! Port of `sympy_extras/polys/galoisfactor.py`: factorisation of
//! univariate polynomials over the finite fields `GF(p**n)`
//! ([`GaloisField`]), in the three classical stages: square-free
//! decomposition in characteristic `p` (the `p`-th power parts through the
//! `p`-th roots of the coefficients), distinct degrees (`gcd(f, x**(q**d) -
//! x)`), and equal degrees by Cantor and Zassenhaus (the trace map in
//! characteristic 2).
//!
//! The random choices come from a seeded generator (not Python's Mersenne
//! Twister); the factors are sorted, so the result is that of Python.
//!
//! References
//! ==========
//!
//! 1. D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring
//!    polynomials over finite fields*, Mathematics of Computation 36 (1981),
//!    587-592.
//! 2. J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
//!    Cambridge University Press, 2013, Chapter 14.

use crate::error::PolyResult;
use crate::numbers::*;
use crate::random::Rng;

use super::densearith::{dup_add, dup_div, dup_mul, dup_rem, dup_sub};
use super::densebasic::dup_degree;
use super::densetools::{dup_diff, dup_monic};
use super::domains::galoisfield::{GaloisElement, GaloisField};
use super::domains::Domain;
use super::euclidtools::dup_gcd;

/// A polynomial over `GF(p**n)`: a dense list of elements, highest degree
/// first.
pub type GaloisPoly = Vec<GaloisElement>;

/// The factors with their multiplicities (or degrees).
pub type GaloisFactors = Vec<(GaloisPoly, usize)>;

/// The seed of the random choices of the equal-degree stage.
const SEED: u64 = 20261110;

/// `f**n` modulo `g`, by repeated squaring.
pub fn gfq_pow_mod(f: &[GaloisElement], n: &Int, g: &[GaloisElement], k: &GaloisField) -> PolyResult<GaloisPoly> {
    let mut result: GaloisPoly = vec![k.one()];
    let mut base: GaloisPoly = dup_rem(f, g, k)?;
    let length: usize = bits(n) as usize;
    for i in 0..length {
        if bit(n, i) {
            result = dup_rem(&dup_mul(&result, &base, k), g, k)?;
        }
        if i + 1 < length {
            base = dup_rem(&dup_mul(&base, &base, k), g, k)?;
        }
    }
    Ok(result)
}

/// The leading coefficient and the square-free decomposition of `f`:
/// pairwise coprime monic square-free `g_i` with their multiplicities,
/// `f = lc prod g_i**i`, sorted by multiplicity, length, coefficients.
pub fn gfq_sqf_list(f: &[GaloisElement], k: &GaloisField) -> PolyResult<(GaloisElement, GaloisFactors)> {
    if f.is_empty() {
        return Ok((k.zero(), Vec::new()));
    }
    let lc: GaloisElement = f[0].clone();
    let mut f: GaloisPoly = dup_monic(f, k)?;
    let p: usize = k.p as usize;
    let mut result: GaloisFactors = Vec::new();
    let mut multiplier: usize = 1;
    while dup_degree(&f) > 0 {
        let derivative: GaloisPoly = dup_diff(&f, 1, k);
        if !derivative.is_empty() {
            let mut g: GaloisPoly = dup_gcd(&f, &derivative, k);
            let mut w: GaloisPoly = dup_div(&f, &g, k)?.0;
            let mut i: usize = 1;
            while dup_degree(&w) > 0 {
                let y: GaloisPoly = dup_gcd(&w, &g, k);
                let z: GaloisPoly = dup_div(&w, &y, k)?.0;
                if dup_degree(&z) > 0 {
                    result.push((dup_monic(&z, k)?, i * multiplier));
                }
                w = y.clone();
                g = dup_div(&g, &y, k)?.0;
                i += 1;
            }
            f = g;
        }
        if dup_degree(&f) > 0 {
            // f = h(x)**p: the p-th roots of the coefficients of x**(k p)
            let degree: usize = dup_degree(&f) as usize;
            f = (0..=degree / p).rev().map(|j| k.pth_root(&f[degree - j * p])).collect();
            multiplier *= p;
        }
    }
    result.sort_by(|a, b| (a.1, a.0.len(), &a.0).cmp(&(b.1, b.0.len(), &b.0)));
    Ok((lc, result))
}

/// The distinct-degree factorisation of a monic square-free `f`: the pairs
/// `(g, d)`, `g` the product of the irreducible factors of degree `d`.
pub fn gfq_ddf(f: &[GaloisElement], k: &GaloisField) -> PolyResult<GaloisFactors> {
    let mut factors: GaloisFactors = Vec::new();
    let x: GaloisPoly = vec![k.one(), k.zero()];
    let mut f: GaloisPoly = f.to_vec();
    let mut h: GaloisPoly = dup_rem(&x, &f, k)?;
    let mut d: usize = 1;
    while dup_degree(&f) >= 2 * d as isize {
        h = gfq_pow_mod(&h, &k.order, &f, k)?;
        let g: GaloisPoly = dup_gcd(&f, &dup_sub(&h, &x, k), k);
        if dup_degree(&g) > 0 {
            let g: GaloisPoly = dup_monic(&g, k)?;
            f = dup_div(&f, &g, k)?.0;
            factors.push((g, d));
            h = dup_rem(&h, &f, k)?;
        }
        d += 1;
    }
    if dup_degree(&f) > 0 {
        let degree: usize = dup_degree(&f) as usize;
        factors.push((dup_monic(&f, k)?, degree));
    }
    Ok(factors)
}

/// The irreducible factors of a monic square-free `f` all of whose
/// irreducible factors have the degree `d` (Cantor–Zassenhaus).
pub fn gfq_edf(f: &[GaloisElement], d: usize, k: &GaloisField, rng: &mut Rng) -> PolyResult<Vec<GaloisPoly>> {
    let mut factors: Vec<GaloisPoly> = Vec::new();
    let mut pending: Vec<GaloisPoly> = vec![f.to_vec()];
    while let Some(g) = pending.pop() {
        let n: usize = dup_degree(&g) as usize;
        if n == d {
            factors.push(g);
            continue;
        }
        loop {
            let mut h: GaloisPoly = (0..n).map(|_| k.random(rng)).collect();
            while h.first().is_some_and(|c| c.is_empty()) {
                h.remove(0);
            }
            if dup_degree(&h) < 1 {
                continue;
            }
            let part: GaloisPoly = if k.p == 2 {
                // the trace of GF(q**d) over GF(2)
                let mut trace: GaloisPoly = h.clone();
                let mut power: GaloisPoly = h;
                for _ in 0..k.degree * d - 1 {
                    power = dup_rem(&dup_mul(&power, &power, k), &g, k)?;
                    trace = dup_add(&trace, &power, k);
                }
                dup_gcd(&g, &trace, k)
            } else {
                let exponent: Int = (k.order.pow(d) - Int::ONE) / Int::from(2u8);
                let power: GaloisPoly = gfq_pow_mod(&h, &exponent, &g, k)?;
                dup_gcd(&g, &dup_sub(&power, &[k.one()], k), k)
            };
            let degree: isize = dup_degree(&part);
            if 0 < degree && degree < n as isize {
                let part: GaloisPoly = dup_monic(&part, k)?;
                let other: GaloisPoly = dup_div(&g, &part, k)?.0;
                pending.push(part);
                pending.push(other);
                break;
            }
        }
    }
    Ok(factors)
}

/// The leading coefficient and the monic irreducible factors of `f` over
/// `k` with their multiplicities, sorted by degree, then by coefficients.
pub fn gfq_factor(f: &[GaloisElement], k: &GaloisField) -> PolyResult<(GaloisElement, GaloisFactors)> {
    if f.is_empty() {
        return Ok((k.zero(), Vec::new()));
    }
    let mut rng: Rng = Rng::new(SEED);
    let (lc, parts) = gfq_sqf_list(f, k)?;
    let mut factors: GaloisFactors = Vec::new();
    for (g, multiplicity) in parts {
        for (h, d) in gfq_ddf(&g, k)? {
            for irreducible in gfq_edf(&h, d, k, &mut rng)? {
                factors.push((irreducible, multiplicity));
            }
        }
    }
    factors.sort_by(|a, b| (a.0.len(), &a.0, a.1).cmp(&(b.0.len(), &b.0, b.1)));
    Ok((lc, factors))
}

/// Whether `f` is irreducible over `k`.
pub fn gfq_irreducible_p(f: &[GaloisElement], k: &GaloisField) -> PolyResult<bool> {
    if dup_degree(f) < 1 {
        return Ok(false);
    }
    let (_, factors) = gfq_factor(f, k)?;
    Ok(factors.len() == 1 && factors[0].1 == 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docstring_examples() {
        let k = GaloisField::new(2, 2, None).unwrap();
        let f: GaloisPoly = vec![vec![1], vec![], vec![1], vec![1]];
        assert!(gfq_irreducible_p(&f, &k).unwrap());
        assert_eq!(gfq_factor(&f, &k).unwrap(), (vec![1], vec![(f.clone(), 1)]));
        assert_eq!(
            gfq_factor(&[vec![1], vec![1], vec![1]], &k).unwrap(),
            (
                vec![1],
                vec![(vec![vec![1], vec![1, 0]], 1), (vec![vec![1], vec![1, 1]], 1)]
            )
        );
        assert_eq!(
            gfq_factor(&[vec![1], vec![], vec![], vec![], vec![1]], &k).unwrap(),
            (vec![1], vec![(vec![vec![1], vec![1]], 4)])
        );
        assert_eq!(
            gfq_sqf_list(&[vec![1], vec![], vec![1], vec![]], &k).unwrap(),
            (vec![1], vec![(vec![vec![1], vec![]], 1), (vec![vec![1], vec![1]], 2)])
        );
        let k9 = GaloisField::new(3, 2, Some(&[int(1), int(0), int(1)])).unwrap();
        assert_eq!(
            gfq_factor(&[vec![1], vec![], vec![1]], &k9).unwrap(),
            (
                vec![1],
                vec![(vec![vec![1], vec![1, 0]], 1), (vec![vec![1], vec![2, 0]], 1)]
            )
        );
        let mut f: GaloisPoly = vec![vec![2]];
        f.extend((0..8).map(|_| Vec::new()));
        f.push(vec![1]);
        assert_eq!(
            gfq_factor(&f, &k9).unwrap(),
            (vec![2], vec![(vec![vec![1], vec![2]], 9)])
        );
    }
}
