//! Port of `sympy_extras/polys/berlekampmassey.py`: the Berlekamp–Massey
//! algorithm, the shortest linear recurrence of a sequence over any field
//! of the typed layer, in `O(N**2)` field operations (Massey's formulation:
//! the current recurrence is corrected by a multiple of the last one which
//! failed whenever a discrepancy appears).
//!
//! References
//! ==========
//!
//! 1. J. L. Massey, *Shift-register synthesis and BCH decoding*, IEEE
//!    Transactions on Information Theory 15 (1969), 122-127.
//! 2. J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
//!    Cambridge University Press, 2013, Section 12.3.

use crate::error::PolyResult;

use super::domains::Domain;

/// The monic polynomial of least degree `L` (dense, highest degree first)
/// whose coefficients `1, c_1, ..., c_L` give a recurrence
/// `s[k + L] + c_1 s[k + L - 1] + ... + c_L s[k] = 0` satisfied by the whole
/// sequence, over the field `k`; `[1]` for the zero sequence. The error is
/// that of a division in a domain which is not a field.
pub fn berlekamp_massey<K: Domain>(sequence: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    // connection polynomials, lowest degree first: sum_i c[i] s[k - i] = 0
    let mut current: Vec<K::Elem> = vec![k.one()];
    let mut previous: Vec<K::Elem> = vec![k.one()];
    let mut length: usize = 0;
    let mut shift: usize = 1;
    let mut last: K::Elem = k.one();
    for index in 0..sequence.len() {
        let mut discrepancy: K::Elem = sequence[index].clone();
        for i in 1..=length {
            if i < current.len() {
                discrepancy = k.add(&discrepancy, &k.mul(&current[i], &sequence[index - i]));
            }
        }
        if k.is_zero(&discrepancy) {
            shift += 1;
            continue;
        }
        let factor: K::Elem = k.quo(&discrepancy, &last)?;
        let mut corrected: Vec<K::Elem> = current.clone();
        let wanted: usize = previous.len() + shift;
        if wanted > corrected.len() {
            corrected.resize(wanted, k.zero());
        }
        for (i, c) in previous.iter().enumerate() {
            corrected[i + shift] = k.sub(&corrected[i + shift], &k.mul(&factor, c));
        }
        if 2 * length <= index {
            previous = current;
            last = discrepancy;
            length = index + 1 - length;
            shift = 1;
        } else {
            shift += 1;
        }
        current = corrected;
    }
    // x**L + c_1 x**(L - 1) + ... + c_L
    let mut coefficients: Vec<K::Elem> = current;
    if coefficients.len() < length + 1 {
        coefficients.resize(length + 1, k.zero());
    }
    coefficients.truncate(length + 1);
    Ok(coefficients)
}

/// Whether the sequence satisfies the recurrence of the polynomial
/// `c_0 x**L + ... + c_L`: `sum_i c_i s[k + L - i] = 0` for every `k`.
pub fn satisfies_recurrence<K: Domain>(sequence: &[K::Elem], polynomial: &[K::Elem], k: &K) -> bool {
    let order: usize = polynomial.len() - 1;
    if sequence.len() < order {
        return true;
    }
    for start in 0..sequence.len() - order {
        let mut total: K::Elem = k.zero();
        for (i, c) in polynomial.iter().enumerate() {
            total = k.add(&total, &k.mul(c, &sequence[start + order - i]));
        }
        if !k.is_zero(&total) {
            return false;
        }
    }
    true
}

/// The sequence continued to `count` terms by its shortest recurrence.
pub fn extend_sequence<K: Domain>(sequence: &[K::Elem], k: &K, count: usize) -> PolyResult<Vec<K::Elem>> {
    let polynomial: Vec<K::Elem> = berlekamp_massey(sequence, k)?;
    let order: usize = polynomial.len() - 1;
    let mut result: Vec<K::Elem> = sequence.to_vec();
    while result.len() < count {
        let mut total: K::Elem = k.zero();
        for i in 1..=order {
            total = k.sub(&total, &k.mul(&polynomial[i], &result[result.len() - i]));
        }
        result.push(total);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers::rat;
    use crate::polys::domains::{FiniteField, QQ};

    #[test]
    fn docstring_examples() {
        let fib: Vec<_> = [0, 1, 1, 2, 3, 5, 8, 13].iter().map(|v| rat(*v, 1)).collect();
        assert_eq!(
            berlekamp_massey(&fib, &QQ).unwrap(),
            vec![rat(1, 1), rat(-1, 1), rat(-1, 1)]
        );
        assert_eq!(
            extend_sequence(&fib, &QQ, 11).unwrap()[8..],
            [rat(21, 1), rat(34, 1), rat(55, 1)]
        );
        let gf2 = FiniteField::new(2).unwrap();
        let bits: Vec<u64> = vec![1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1];
        assert_eq!(berlekamp_massey(&bits, &gf2).unwrap(), vec![1, 0, 1, 1]);
        let gf5 = FiniteField::new(5).unwrap();
        assert_eq!(berlekamp_massey(&[1, 2, 4, 3, 1, 2, 4, 3], &gf5).unwrap(), vec![1, 3]);
        assert_eq!(berlekamp_massey(&[0, 0, 0], &gf5).unwrap(), vec![1]);
        assert!(satisfies_recurrence(&[1, 2, 4, 3, 1], &[1, 3], &gf5));
        assert!(!satisfies_recurrence(&[1, 2, 4, 3, 2], &[1, 3], &gf5));
    }
}
