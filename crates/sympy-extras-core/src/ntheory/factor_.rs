//! Port of `sympy_extras/ntheory/factor_.py`: integer factorisation by trial
//! division, then Pollard's rho method with Brent's cycle detection, the
//! factors tested by Baillie–PSW.
//!
//! `factorint(n)` returns the prime factorisation as a map from primes to
//! exponents; a negative `n` has the factor `-1`, `0` gives `{0: 1}`. With
//! `limit`, only the primes up to it are removed by trial division and the
//! cofactor which remains is one entry (it may be composite), the
//! convention of SymPy's `factorint`.
//!
//! References
//! ==========
//!
//! 1. R. P. Brent, *An improved Monte Carlo factorization algorithm*, BIT 20
//!    (1980), 176-184.
//! 2. R. Crandall, C. Pomerance, *Prime Numbers: A Computational
//!    Perspective*, 2nd ed., Springer, 2005, Section 5.2.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/ntheory/factor_.py):
// sympy/ntheory/factor_.py (is_carmichael, find_carmichael_numbers_in_range,
// find_first_n_carmichaels) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// ---------------------------------------------------------------------

use std::collections::BTreeMap;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::primetest::{isprime, small_primes};

/// `(b, e)` with `b**e = n` and `e >= 2` maximal, for `n > 1`; `None` when
/// `n` is not a perfect power.
pub fn perfect_power(n: &Int) -> Option<(Int, u64)> {
    if *n < Int::from(4u8) {
        return None;
    }
    let mut best: Option<(Int, u64)> = None;
    for e in 2..bits(n) + 1 {
        let b: Int = integer_root(n, e);
        if b < Int::from(2u8) {
            break;
        }
        if b.pow(e as usize) == *n {
            best = Some((b, e));
        }
    }
    best
}

/// The integer part of `n**(1/e)`.
fn integer_root(n: &Int, e: u64) -> Int {
    if e == 2 {
        return isqrt(n);
    }
    let mut low: Int = Int::ONE;
    let mut high: Int = Int::from(2u8).pow((bits(n) / e + 1) as usize);
    while low < high {
        let mid: Int = div_floor(&(&low + &high + Int::ONE), &Int::from(2u8));
        if mid.pow(e as usize) <= *n {
            low = mid;
        } else {
            high = mid - Int::ONE;
        }
    }
    low
}

/// A nontrivial factor of the odd composite `n` (which is not a prime power)
/// by Pollard's rho with Brent's cycle detection and batched gcds, for the
/// maps `x -> x**2 + c` with `c = seed, seed + 1, ...` until one succeeds.
pub fn pollard_brent(n: &Int, seed: &Int) -> Int {
    let mut c: Int = seed.clone();
    let step = |v: &Int, c: &Int| -> Int { mod_floor(&(v * v + c), n) };
    loop {
        let mut y: Int = Int::from(2u8);
        let mut r: u64 = 1;
        let mut q: Int = Int::ONE;
        let mut g: Int = Int::ONE;
        let mut x: Int = y.clone();
        let mut ys: Int = y.clone();
        let batch: u64 = 128;
        while g.is_one() {
            x = y.clone();
            for _ in 0..r {
                y = step(&y, &c);
            }
            let mut k: u64 = 0;
            while k < r && g.is_one() {
                ys = y.clone();
                for _ in 0..batch.min(r - k) {
                    y = step(&y, &c);
                    q = mod_floor(&(&q * abs(&(&x - &y))), n);
                }
                g = gcd(&q, n);
                k += batch;
            }
            r *= 2;
        }
        if g == *n {
            // the batch overshot: step back one at a time
            g = Int::ONE;
            while g.is_one() {
                ys = step(&ys, &c);
                g = gcd(&abs(&(&x - &ys)), n);
            }
        }
        if g != *n {
            return g;
        }
        c += Int::ONE;
    }
}

/// Add the prime factors of `n > 1` (with no prime factor below 1000) to
/// `factors`, each `multiplicity` times.
fn factor_into(n: &Int, factors: &mut BTreeMap<Int, u64>, multiplicity: u64) {
    if n.is_one() {
        return;
    }
    if isprime(n) {
        *factors.entry(n.clone()).or_insert(0) += multiplicity;
        return;
    }
    if let Some((b, e)) = perfect_power(n) {
        factor_into(&b, factors, multiplicity * e);
        return;
    }
    let d: Int = pollard_brent(n, &Int::ONE);
    let mut n: Int = n.clone();
    let mut k: u64 = 0;
    while mod_floor(&n, &d).is_zero() {
        n = div_floor(&n, &d);
        k += 1;
    }
    factor_into(&d, factors, multiplicity * k);
    factor_into(&n, factors, multiplicity);
}

/// The prime factorisation of `n` as `{prime: exponent}`, sorted by the
/// primes (`-1` first for a negative `n`); with `limit`, trial division by
/// the primes up to `limit` only, the cofactor kept whole.
pub fn factorint(n: &Int, limit: Option<&Int>) -> BTreeMap<Int, u64> {
    let mut factors: BTreeMap<Int, u64> = BTreeMap::new();
    if n.is_zero() {
        factors.insert(Int::ZERO, 1);
        return factors;
    }
    let mut n: Int = n.clone();
    if is_negative(&n) {
        factors.insert(-Int::ONE, 1);
        n = -n;
    }
    let bound: Int = match limit {
        Some(l) => l.clone(),
        None => Int::from(997u16),
    };
    for p in small_primes() {
        let pi: Int = Int::from(*p);
        if pi > bound {
            break;
        }
        if mod_floor(&n, &pi).is_zero() {
            let mut k: u64 = 0;
            while mod_floor(&n, &pi).is_zero() {
                n = div_floor(&n, &pi);
                k += 1;
            }
            factors.insert(pi, k);
        }
    }
    if let Some(limit) = limit {
        if *limit > Int::from(997u16) {
            let mut p: Int = Int::from(1001u16);
            while p <= *limit && &p * &p <= n {
                if mod_floor(&n, &p).is_zero() {
                    let mut k: u64 = 0;
                    while mod_floor(&n, &p).is_zero() {
                        n = div_floor(&n, &p);
                        k += 1;
                    }
                    factors.insert(p.clone(), k);
                }
                p += Int::from(2u8);
            }
            if n > Int::ONE && &p * &p > n {
                *factors.entry(n.clone()).or_insert(0) += 1;
                n = Int::ONE;
            }
        }
    }
    if n > Int::ONE {
        if limit.is_none() {
            factor_into(&n, &mut factors, 1);
        } else {
            *factors.entry(n).or_insert(0) += 1;
        }
    }
    factors
}

/// Whether `n` is a Carmichael number (Korselt's criterion: odd, composite,
/// squarefree, `p - 1 | n - 1` for its prime factors).
pub fn is_carmichael(n: &Int) -> bool {
    if *n < Int::from(561u16) {
        return false;
    }
    let n_minus_one: Int = n - Int::ONE;
    bit(n, 0)
        && !isprime(n)
        && factorint(n, None)
            .iter()
            .all(|(p, e)| *e == 1 && mod_floor(&n_minus_one, &(p - Int::ONE)).is_zero())
}

/// The Carmichael numbers in `[x, y)` (`Value` unless `0 <= x <= y`).
pub fn find_carmichael_numbers_in_range(x: &Int, y: &Int) -> PolyResult<Vec<Int>> {
    if is_negative(x) || x > y {
        return Err(value_error(
            "The provided range is not valid. x and y must be non-negative integers and x <= y".to_string(),
        ));
    }
    let mut i: Int = if bit(x, 0) { x.clone() } else { x + Int::ONE };
    let mut found: Vec<Int> = Vec::new();
    while i < *y {
        if is_carmichael(&i) {
            found.push(i.clone());
        }
        i += Int::from(2u8);
    }
    Ok(found)
}

/// The first `n` Carmichael numbers.
pub fn find_first_n_carmichaels(n: usize) -> Vec<Int> {
    let mut i: Int = Int::from(561u16);
    let mut found: Vec<Int> = Vec::new();
    while found.len() < n {
        if is_carmichael(&i) {
            found.push(i.clone());
        }
        i += Int::from(2u8);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(Int, u64)]) -> BTreeMap<Int, u64> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn doctests() {
        let m61: Int = Int::from(2u8).pow(61) - Int::ONE;
        let n: Int = int(16 * 3 * 1000003) * &m61;
        assert_eq!(
            factorint(&n, None),
            map(&[(int(2), 4), (int(3), 1), (int(1000003), 1), (m61.clone(), 1)])
        );
        assert_eq!(
            factorint(&n, Some(&int(100))),
            map(&[(int(2), 4), (int(3), 1), (int(1000003) * &m61, 1)])
        );
        assert_eq!(
            factorint(&int(-12), None),
            map(&[(int(-1), 1), (int(2), 2), (int(3), 1)])
        );
        assert_eq!(factorint(&int(0), None), map(&[(int(0), 1)]));
        assert_eq!(factorint(&int(1), None), map(&[]));
        assert_eq!(factorint(&int(-1), None), map(&[(int(-1), 1)]));
        assert_eq!(factorint(&int(10403), None), map(&[(int(101), 1), (int(103), 1)]));
        let d: Int = pollard_brent(&int(10403), &int(1));
        assert!(d == int(101) || d == int(103));
        assert!(is_carmichael(&int(561)) && !is_carmichael(&int(563)));
        assert_eq!(
            find_carmichael_numbers_in_range(&int(1), &int(2000)).unwrap(),
            vec![int(561), int(1105), int(1729)]
        );
        assert_eq!(perfect_power(&int(1024)), Some((int(2), 10)));
        assert_eq!(perfect_power(&int(1000)), Some((int(10), 3)));
        assert_eq!(perfect_power(&int(1001)), None);
    }
}
