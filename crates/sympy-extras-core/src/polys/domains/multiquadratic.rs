//! Multiquadratic extensions `K(sqrt(d_1), ..., sqrt(d_k))` of a field `K`
//! of characteristic zero, ported from
//! `sympy_extras.polys.domains.multiquadratic`: an element is a map from
//! the subsets of the generators (bit masks) to nonzero coefficients; the
//! product of two basis elements is that of the symmetric difference times
//! the product of the `d_i` in the intersection; the inverse comes from the
//! conjugations (H. Cohen, *A Course in Computational Algebraic Number
//! Theory*, 4.2-4.3). The `d_i` must be independent modulo squares
//! (distinct primes, and `-1`).

use std::collections::BTreeMap;

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::Int;

use super::domain::Domain;

/// An element: the bit masks of the products of generators to their
/// nonzero coefficients.
pub type MultiquadraticElement<T> = BTreeMap<u64, T>;

/// `K(sqrt(d_1), ..., sqrt(d_k))` over the field `ground`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiquadraticField<K: Domain> {
    pub ground: K,
    pub radicands: Vec<i64>,
}

impl<K: Domain> MultiquadraticField<K> {
    pub fn new(ground: K, radicands: Vec<i64>) -> PolyResult<MultiquadraticField<K>> {
        if !ground.is_field() || ground.characteristic() != Int::ZERO {
            return Err(value_error(format!(
                "{} is not a field of characteristic zero",
                ground.name()
            )));
        }
        let mut seen: Vec<i64> = radicands.clone();
        seen.sort();
        seen.dedup();
        if seen.len() != radicands.len() || radicands.iter().any(|d| *d == 0 || *d == 1) {
            let shown: Vec<String> = radicands.iter().map(|d| d.to_string()).collect();
            return Err(value_error(format!(
                "the radicands [{}] are not independent",
                shown.join(", ")
            )));
        }
        Ok(MultiquadraticField { ground, radicands })
    }

    /// `sqrt(radicands[i])`.
    pub fn generator(&self, i: usize) -> MultiquadraticElement<K::Elem> {
        BTreeMap::from([(1u64 << i, self.ground.one())])
    }

    pub fn from_ground(&self, c: K::Elem) -> MultiquadraticElement<K::Elem> {
        if self.ground.is_zero(&c) {
            BTreeMap::new()
        } else {
            BTreeMap::from([(0, c)])
        }
    }

    fn square_part(&self, common: u64) -> i64 {
        let mut product: i64 = 1;
        let mut common: u64 = common;
        let mut i: usize = 0;
        while common != 0 {
            if common & 1 == 1 {
                product *= self.radicands[i];
            }
            common >>= 1;
            i += 1;
        }
        product
    }

    /// The image under the automorphism `sqrt(d_i) -> -sqrt(d_i)`.
    pub fn conjugate(&self, a: &MultiquadraticElement<K::Elem>, i: usize) -> MultiquadraticElement<K::Elem> {
        let bit: u64 = 1 << i;
        a.iter()
            .map(|(mask, c)| (*mask, if mask & bit != 0 { self.ground.neg(c) } else { c.clone() }))
            .collect()
    }
}

impl<K: Domain> Domain for MultiquadraticField<K> {
    type Elem = MultiquadraticElement<K::Elem>;

    fn name(&self) -> String {
        let roots: Vec<String> = self.radicands.iter().map(|d| format!("sqrt({})", d)).collect();
        format!("{}({})", self.ground.name(), roots.join(", "))
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        false
    }

    fn characteristic(&self) -> Int {
        Int::ZERO
    }

    fn zero(&self) -> Self::Elem {
        BTreeMap::new()
    }

    fn one(&self) -> Self::Elem {
        BTreeMap::from([(0, self.ground.one())])
    }

    fn is_zero(&self, a: &Self::Elem) -> bool {
        a.is_empty()
    }

    fn is_one(&self, a: &Self::Elem) -> bool {
        a.len() == 1 && a.get(&0).is_some_and(|c| self.ground.is_one(c))
    }

    fn is_unit(&self, a: &Self::Elem) -> bool {
        !a.is_empty()
    }

    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        let k: &K = &self.ground;
        let mut result: Self::Elem = a.clone();
        for (mask, c) in b {
            match result.get(mask) {
                None => {
                    result.insert(*mask, c.clone());
                }
                Some(known) => {
                    let total = k.add(known, c);
                    if k.is_zero(&total) {
                        result.remove(mask);
                    } else {
                        result.insert(*mask, total);
                    }
                }
            }
        }
        result
    }

    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.add(a, &self.neg(b))
    }

    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        a.iter().map(|(m, c)| (*m, self.ground.neg(c))).collect()
    }

    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        let k: &K = &self.ground;
        let mut result: Self::Elem = BTreeMap::new();
        for (m1, c1) in a {
            for (m2, c2) in b {
                let mut value = k.mul(c1, c2);
                let common: u64 = m1 & m2;
                if common != 0 {
                    value = k.mul(&value, &k.from_int(self.square_part(common)));
                }
                let mask: u64 = m1 ^ m2;
                if let Some(known) = result.get(&mask) {
                    value = k.add(known, &value);
                }
                if k.is_zero(&value) {
                    result.remove(&mask);
                } else {
                    result.insert(mask, value);
                }
            }
        }
        result
    }

    fn exquo(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<Self::Elem> {
        if a.is_empty() {
            if b.is_empty() {
                return Err(zero_division(&format!("division by zero in {}", self.name())));
            }
            return Ok(BTreeMap::new());
        }
        Ok(self.mul(a, &self.invert(b)?))
    }

    fn quo(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<Self::Elem> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &Self::Elem, b: &Self::Elem) -> PolyResult<Self::Elem> {
        if b.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(BTreeMap::new())
    }

    fn invert(&self, a: &Self::Elem) -> PolyResult<Self::Elem> {
        let Some(highest_mask) = a.keys().next_back() else {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        };
        if *highest_mask == 0 {
            return Ok(BTreeMap::from([(0, self.ground.invert(&a[&0])?)]));
        }
        let highest: usize = (63 - highest_mask.leading_zeros()) as usize;
        let conjugate = self.conjugate(a, highest);
        Ok(self.mul(&conjugate, &self.invert(&self.mul(a, &conjugate))?))
    }

    fn gcd(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        if a.is_empty() && b.is_empty() {
            BTreeMap::new()
        } else {
            self.one()
        }
    }

    fn gcdex(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<(Self::Elem, Self::Elem, Self::Elem)> {
        if !a.is_empty() {
            return Ok((self.invert(a)?, self.zero(), self.one()));
        }
        if !b.is_empty() {
            return Ok((self.zero(), self.invert(b)?, self.one()));
        }
        Ok((self.zero(), self.zero(), self.zero()))
    }

    fn size(&self, a: &Self::Elem) -> u64 {
        a.values().map(|c| self.ground.size(c)).sum::<u64>() + a.len() as u64
    }

    fn from_int(&self, n: i64) -> Self::Elem {
        self.from_ground(self.ground.from_int(n))
    }

    fn from_integer(&self, n: &Int) -> Self::Elem {
        self.from_ground(self.ground.from_integer(n))
    }

    fn to_str(&self, a: &Self::Elem) -> String {
        if a.is_empty() {
            return "0".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        for (mask, c) in a {
            let coefficient: String = self.ground.to_str(c);
            if *mask == 0 {
                parts.push(coefficient);
                continue;
            }
            let radical: String = format!("sqrt({})", self.square_part(*mask));
            parts.push(if self.ground.is_one(c) {
                radical
            } else {
                format!("{}*{}", coefficient, radical)
            });
        }
        parts.join(" + ")
    }
}
