//! Port of `sympy_extras/polys/galoistools.py`: factorisation of dense
//! polynomials over the prime fields `GF(p)`.
//!
//! * powers modulo a polynomial ([`gf_pow_mod`]) and the Frobenius map
//!   `h -> h**p mod f` by the precomputed powers of `x**p`
//!   ([`gf_frobenius_monomial_base`], [`gf_frobenius_map`]);
//! * the distinct-degree factorisation of a monic square-free `f`:
//!   `gcd(f, x**(p**d) - x)` collects the irreducible factors of degree `d`;
//! * the equal-degree factorisation of Cantor and Zassenhaus: for a product
//!   of irreducibles of the same degree `d`, a random `h` gives a proper
//!   factor `gcd(f, h**((p**d - 1)/2) - 1)` with probability at least one
//!   half for odd `p`, and `gcd(f, h + h**2 + ... + h**(2**(d-1)))` (the
//!   trace map) for `p = 2`;
//! * [`gf_factor_sqf`], [`gf_factor`] and Rabin's irreducibility test.
//!
//! The random choices come from [`Rng`] instead of Python's Mersenne
//! Twister; the factorisations are sorted, so they do not depend on them.
//!
//! References
//! ==========
//!
//! 1. J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
//!    Cambridge University Press, 2013, Chapter 14 (Algorithms 14.3, 14.8,
//!    14.10, 14.13).
//! 2. D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring
//!    polynomials over finite fields*, Mathematics of Computation 36
//!    (1981), 587-592.
//! 3. K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//!    Algebra*, Kluwer, 1992, Section 8.4.

use crate::numbers::*;
use crate::random::Rng;

use super::densearith::{dup_add, dup_div, dup_mul, dup_rem, dup_sqr, dup_sub};
use super::densebasic::{dup_degree, dup_strip_vec};
use super::densetools::dup_monic_field;
use super::domains::{Domain, FiniteField};
use super::euclidtools::dup_gcd;
use super::sqfreetools::{dup_sqf_list, Factor};

/// The seed of the random polynomials of the equal-degree splitting, so
/// that the factorisation is reproducible.
const SEED: u64 = 1;

/// A remainder by a nonzero polynomial over a field, which cannot fail.
fn rem(f: &[u64], g: &[u64], k: &FiniteField) -> Vec<u64> {
    dup_rem(f, g, k).expect("a nonzero divisor over a field")
}

fn quo(f: &[u64], g: &[u64], k: &FiniteField) -> Vec<u64> {
    dup_div(f, g, k).expect("a nonzero divisor over a field").0
}

/// The sort key of the Python code, `(len(h), h)`.
fn by_degree_then_coefficients(a: &Vec<u64>, b: &Vec<u64>) -> std::cmp::Ordering {
    (a.len(), a).cmp(&(b.len(), b))
}

/// `f**n mod g` by repeated squaring.
pub fn gf_pow_mod(f: &[u64], n: &Int, g: &[u64], k: &FiniteField) -> Vec<u64> {
    assert!(!is_negative(n), "negative exponent");
    let mut result: Vec<u64> = vec![k.one()];
    let mut base: Vec<u64> = rem(f, g, k);
    let length: u64 = bits(n);
    for i in 0..length as usize {
        if bit(n, i) {
            result = rem(&dup_mul(&result, &base, k), g, k);
        }
        if i + 1 < length as usize {
            base = rem(&dup_sqr(&base, k), g, k);
        }
    }
    result
}

/// `x**(i p) mod g` for `i = 0, ..., deg g - 1`: the matrix of the
/// Frobenius map on `GF(p)[x]/(g)`.
pub fn gf_frobenius_monomial_base(g: &[u64], k: &FiniteField) -> Vec<Vec<u64>> {
    let n: isize = dup_degree(g);
    if n <= 0 {
        return Vec::new();
    }
    let mut base: Vec<Vec<u64>> = vec![vec![k.one()]];
    let x_p: Vec<u64> = gf_pow_mod(&[k.one(), k.zero()], &Int::from(k.p()), g, k);
    for _ in 1..n {
        let next: Vec<u64> = rem(&dup_mul(&base[base.len() - 1], &x_p, k), g, k);
        base.push(next);
    }
    base
}

/// `f**p mod g` through the precomputed powers `x**(i p) mod g`: the
/// coefficients are their own `p`-th powers.
pub fn gf_frobenius_map(f: &[u64], g: &[u64], base: &[Vec<u64>], k: &FiniteField) -> Vec<u64> {
    let n: isize = dup_degree(g);
    let f: Vec<u64> = rem(f, g, k);
    let mut result: Vec<u64> = Vec::new();
    let m: usize = f.len();
    for i in 0..m {
        let c: u64 = f[m - 1 - i];
        if c != 0 {
            let term: Vec<u64> = base[i].iter().map(|b| k.mul(&c, b)).collect();
            result = dup_add(&result, &term, k);
        }
    }
    if n > 0 {
        dup_strip_vec(result, k)
    } else {
        Vec::new()
    }
}

/// The distinct-degree factorisation of a monic square-free `f`: the pairs
/// `(g, d)` with `g` the product of the irreducible factors of degree `d`,
/// for the degrees which occur.
pub fn gf_ddf(f: &[u64], k: &FiniteField) -> Vec<Factor<u64>> {
    let mut f: Vec<u64> = dup_monic_field(f, k);
    let mut factors: Vec<Factor<u64>> = Vec::new();
    let mut base: Vec<Vec<u64>> = gf_frobenius_monomial_base(&f, k);
    let x: Vec<u64> = vec![k.one(), k.zero()];
    let mut h: Vec<u64> = rem(&x, &f, k);
    let mut d: usize = 1;
    while dup_degree(&f) >= 2 * d as isize {
        h = gf_frobenius_map(&h, &f, &base, k);
        let g: Vec<u64> = dup_gcd(&f, &dup_sub(&h, &x, k), k);
        if dup_degree(&g) > 0 {
            f = quo(&f, &g, k);
            h = rem(&h, &f, k);
            base = gf_frobenius_monomial_base(&f, k);
            factors.push((g, d));
        }
        d += 1;
    }
    if dup_degree(&f) > 0 {
        let degree: usize = f.len() - 1;
        factors.push((f, degree));
    }
    factors
}

/// The equal-degree factorisation (Cantor–Zassenhaus) of a monic
/// square-free `f` all of whose irreducible factors have degree `d`: the
/// list of the factors.
pub fn gf_edf(f: &[u64], d: usize, k: &FiniteField, rng: &mut Rng) -> Vec<Vec<u64>> {
    if dup_degree(f) == d as isize {
        return vec![f.to_vec()];
    }
    let mut factors: Vec<Vec<u64>> = Vec::new();
    let mut to_split: Vec<Vec<u64>> = vec![f.to_vec()];
    while let Some(g) = to_split.pop() {
        if dup_degree(&g) == d as isize {
            factors.push(g);
            continue;
        }
        let split: [Vec<u64>; 2] = edf_split(&g, d, k, rng);
        to_split.extend(split);
    }
    factors.sort_by(by_degree_then_coefficients);
    factors
}

/// A proper factorisation `[g, f/g]` of `f` (found with probability at
/// least one half per trial).
fn edf_split(f: &[u64], d: usize, k: &FiniteField, rng: &mut Rng) -> [Vec<u64>; 2] {
    let n: usize = f.len() - 1;
    let p: u64 = k.p();
    loop {
        let h: Vec<u64> = dup_strip_vec((0..n).map(|_| rng.randrange(p)).collect(), k);
        if dup_degree(&h) < 1 {
            continue;
        }
        let g: Vec<u64> = if p == 2 {
            // the trace map h + h**2 + ... + h**(2**(d-1))
            let mut trace: Vec<u64> = h.clone();
            let mut power: Vec<u64> = h;
            for _ in 0..d - 1 {
                power = rem(&dup_sqr(&power, k), f, k);
                trace = dup_add(&trace, &power, k);
            }
            dup_gcd(f, &trace, k)
        } else {
            let exponent: Int = (Int::from(p).pow(d) - Int::ONE) / Int::from(2u8);
            let power: Vec<u64> = gf_pow_mod(&h, &exponent, f, k);
            dup_gcd(f, &dup_sub(&power, &[k.one()], k), k)
        };
        let degree: isize = dup_degree(&g);
        if 0 < degree && degree < n as isize {
            let cofactor: Vec<u64> = quo(f, &g, k);
            return [g, cofactor];
        }
    }
}

/// The leading coefficient and the monic irreducible factors of a
/// square-free `f` (each once), sorted by degree.
pub fn gf_factor_sqf(f: &[u64], k: &FiniteField) -> (u64, Vec<Vec<u64>>) {
    if f.is_empty() {
        return (k.zero(), Vec::new());
    }
    let lc: u64 = f[0];
    let f: Vec<u64> = dup_monic_field(f, k);
    if dup_degree(&f) <= 0 {
        return (lc, Vec::new());
    }
    let mut rng: Rng = Rng::new(SEED);
    let mut factors: Vec<Vec<u64>> = Vec::new();
    for (g, d) in gf_ddf(&f, k) {
        factors.extend(gf_edf(&g, d, k, &mut rng));
    }
    factors.sort_by(by_degree_then_coefficients);
    (lc, factors)
}

/// The leading coefficient and the monic irreducible factors with their
/// multiplicities, sorted by degree then lexicographically.
pub fn gf_factor(f: &[u64], k: &FiniteField) -> (u64, Vec<Factor<u64>>) {
    if f.is_empty() {
        return (k.zero(), Vec::new());
    }
    let (lc, sqf) = dup_sqf_list(f, k);
    let mut factors: Vec<Factor<u64>> = Vec::new();
    for (g, multiplicity) in &sqf {
        let (_, irreducibles) = gf_factor_sqf(g, k);
        for h in irreducibles {
            factors.push((h, *multiplicity));
        }
    }
    factors.sort_by(|a, b| (a.0.len(), &a.0, a.1).cmp(&(b.0.len(), &b.0, b.1)));
    (lc, factors)
}

/// Whether `f` is irreducible over `GF(p)` (Rabin's test: `x**(p**n) = x
/// mod f` and `gcd(f, x**(p**(n/q)) - x) = 1` for every prime `q` dividing
/// `n`).
pub fn gf_irreducible_p(f: &[u64], k: &FiniteField) -> bool {
    let n: isize = dup_degree(f);
    if n <= 0 {
        return false;
    }
    if n == 1 {
        return true;
    }
    let n: usize = n as usize;
    let f: Vec<u64> = dup_monic_field(f, k);
    let base: Vec<Vec<u64>> = gf_frobenius_monomial_base(&f, k);
    let x: Vec<u64> = rem(&[k.one(), k.zero()], &f, k);
    // x**(p**i) mod f for i = 1..n
    let mut powers: Vec<Vec<u64>> = vec![x.clone()];
    for _ in 0..n {
        let next: Vec<u64> = gf_frobenius_map(&powers[powers.len() - 1], &f, &base, k);
        powers.push(next);
    }
    if powers[n] != x {
        return false;
    }
    for q in prime_divisors(n) {
        let g: Vec<u64> = dup_gcd(&f, &dup_sub(&powers[n / q], &x, k), k);
        if dup_degree(&g) > 0 {
            return false;
        }
    }
    true
}

fn prime_divisors(n: usize) -> Vec<usize> {
    let mut n: usize = n;
    let mut result: Vec<usize> = Vec::new();
    let mut q: usize = 2;
    while q * q <= n {
        if n % q == 0 {
            result.push(q);
            while n % q == 0 {
                n /= q;
            }
        }
        q += 1;
    }
    if n > 1 {
        result.push(n);
    }
    result
}

/// A random monic polynomial of degree `n`.
pub fn gf_random(n: usize, k: &FiniteField, rng: &mut Rng) -> Vec<u64> {
    let mut f: Vec<u64> = vec![k.one()];
    f.extend((0..n).map(|_| rng.randrange(k.p())));
    f
}

/// A random monic irreducible polynomial of degree `n`.
pub fn gf_irreducible(n: usize, k: &FiniteField, rng: &mut Rng) -> Vec<u64> {
    loop {
        let f: Vec<u64> = gf_random(n, k, rng);
        if gf_irreducible_p(&f, k) {
            return f;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::GF;

    #[test]
    fn doctests() {
        let f: FiniteField = GF(5).unwrap();
        assert_eq!(
            gf_factor(&[1, 0, 0, 0, 4], &f),
            (
                1,
                vec![(vec![1, 1], 1), (vec![1, 2], 1), (vec![1, 3], 1), (vec![1, 4], 1)]
            )
        );
        assert_eq!(
            gf_factor(&[3, 0, 0, 3], &f),
            (3, vec![(vec![1, 1], 1), (vec![1, 4, 1], 1)])
        );
        assert_eq!(gf_factor(&[1, 3, 3, 1], &f), (1, vec![(vec![1, 1], 3)]));
        assert_eq!(gf_ddf(&[1, 0, 0, 0, 4], &f), vec![(vec![1, 0, 0, 0, 4], 1)]);
        assert!(gf_irreducible_p(&[1, 0, 2], &f));
        assert!(!gf_irreducible_p(&[1, 0, 1], &f));
    }

    #[test]
    fn characteristic_two_and_random_irreducibles() {
        let f2: FiniteField = GF(2).unwrap();
        // x**4 + x = x (x + 1) (x**2 + x + 1) over GF(2)
        assert_eq!(
            gf_factor(&[1, 0, 0, 1, 0], &f2),
            (1, vec![(vec![1, 0], 1), (vec![1, 1], 1), (vec![1, 1, 1], 1)])
        );
        let f: FiniteField = GF(7).unwrap();
        let mut rng: Rng = Rng::new(3);
        for n in 1..6 {
            let g: Vec<u64> = gf_irreducible(n, &f, &mut rng);
            let (_, factors) = gf_factor(&g, &f);
            assert_eq!(factors, vec![(g, 1)]);
        }
    }
}
