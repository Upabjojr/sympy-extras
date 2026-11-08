//! The prime fields `GF(p)` for any modulus, with [`Int`] elements in `0,
//! ..., p - 1`.
//!
//! Python's `FiniteField` takes any modulus; the port splits it in two:
//! [`super::FiniteField`] with `u64` elements for the word-sized moduli of
//! the modular algorithms, and this domain for the moduli beyond the machine
//! words (the Hensel lifting of Wang's algorithm works modulo the next prime
//! above a Mignotte bound). The arithmetic is the same.

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::domain::{ConvertFrom, Domain};
use super::integerring::IntegerRing;

/// The field of integers modulo the prime `p` (the primality is not
/// checked, as in Python).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BigFiniteField {
    p: Int,
}

impl BigFiniteField {
    /// `Value` for a modulus below 2.
    pub fn new(p: Int) -> PolyResult<BigFiniteField> {
        if p < Int::from(2u8) {
            return Err(value_error(format!("the modulus must be a prime, got {}", p)));
        }
        Ok(BigFiniteField { p })
    }

    pub fn p(&self) -> &Int {
        &self.p
    }

    fn reduce(&self, n: &Int) -> Int {
        mod_floor(n, &self.p)
    }
}

impl Domain for BigFiniteField {
    type Elem = Int;

    fn name(&self) -> String {
        format!("GF({})", self.p)
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        false
    }

    fn characteristic(&self) -> Int {
        self.p.clone()
    }

    fn zero(&self) -> Int {
        Int::ZERO
    }

    fn one(&self) -> Int {
        Int::ONE
    }

    fn is_zero(&self, a: &Int) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Int) -> bool {
        a.is_one()
    }

    fn is_unit(&self, a: &Int) -> bool {
        !a.is_zero()
    }

    fn add(&self, a: &Int, b: &Int) -> Int {
        self.reduce(&(a + b))
    }

    fn sub(&self, a: &Int, b: &Int) -> Int {
        self.reduce(&(a - b))
    }

    fn neg(&self, a: &Int) -> Int {
        self.reduce(&-a)
    }

    fn mul(&self, a: &Int, b: &Int) -> Int {
        self.reduce(&(a * b))
    }

    fn exquo(&self, a: &Int, b: &Int) -> PolyResult<Int> {
        if b.is_zero() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(self.mul(a, &self.invert(b)?))
    }

    fn quo(&self, a: &Int, b: &Int) -> PolyResult<Int> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &Int, b: &Int) -> PolyResult<Int> {
        if b.is_zero() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(Int::ZERO)
    }

    fn invert(&self, a: &Int) -> PolyResult<Int> {
        if a.is_zero() {
            return Err(zero_division("0 has no inverse"));
        }
        mod_inverse(a, &self.p).ok_or_else(|| value_error("base is not invertible for the given modulus".to_string()))
    }

    fn gcd(&self, a: &Int, b: &Int) -> Int {
        if a.is_zero() && b.is_zero() {
            Int::ZERO
        } else {
            Int::ONE
        }
    }

    fn lcm(&self, a: &Int, b: &Int) -> Int {
        if a.is_zero() || b.is_zero() {
            Int::ZERO
        } else {
            Int::ONE
        }
    }

    fn gcdex(&self, a: &Int, b: &Int) -> PolyResult<(Int, Int, Int)> {
        if !a.is_zero() {
            return Ok((self.invert(a)?, Int::ZERO, Int::ONE));
        }
        if !b.is_zero() {
            return Ok((Int::ZERO, self.invert(b)?, Int::ONE));
        }
        Ok((Int::ZERO, Int::ZERO, Int::ZERO))
    }

    fn canonical_unit(&self, a: &Int) -> Int {
        if a.is_zero() {
            Int::ONE
        } else {
            self.invert(a).expect("a nonzero element of a prime field")
        }
    }

    fn from_int(&self, n: i64) -> Int {
        self.reduce(&Int::from(n))
    }

    fn from_integer(&self, n: &Int) -> Int {
        self.reduce(n)
    }

    fn to_str(&self, a: &Int) -> String {
        a.to_string()
    }
}

impl ConvertFrom<IntegerRing> for BigFiniteField {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<Int> {
        Ok(self.reduce(a))
    }
}

impl ConvertFrom<BigFiniteField> for IntegerRing {
    fn convert(&self, a: &Int, _k0: &BigFiniteField) -> PolyResult<Int> {
        Ok(a.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::densebasic::testing::zz;
    use crate::polys::domains::{FiniteField, GF};
    use crate::polys::euclidtools::dup_gcdex;

    #[test]
    fn agrees_with_the_word_sized_field() {
        let small: FiniteField = GF(10007).unwrap();
        let big: BigFiniteField = BigFiniteField::new(int(10007)).unwrap();
        let f: Vec<Int> = zz(&[3, 0, 7, 1]);
        let g: Vec<Int> = zz(&[5, 2, 9]);
        let (fs, gs): (Vec<u64>, Vec<u64>) = (vec![3, 0, 7, 1], vec![5, 2, 9]);
        let (s, t, h) = dup_gcdex(&f, &g, &big).unwrap();
        let (s2, t2, h2) = dup_gcdex(&fs, &gs, &small).unwrap();
        let back = |v: Vec<u64>| -> Vec<Int> { v.into_iter().map(Int::from).collect() };
        assert_eq!((s, t, h), (back(s2), back(t2), back(h2)));
        let huge: BigFiniteField = BigFiniteField::new(Int::from(2u8).pow(127) - Int::ONE).unwrap();
        let a: Int = Int::from(2u8).pow(100);
        assert!(huge.is_one(&huge.mul(&a, &huge.invert(&a).unwrap())));
    }
}
