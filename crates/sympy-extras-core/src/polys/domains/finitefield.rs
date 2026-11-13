//! Port of `sympy_extras/polys/domains/finitefield.py`: the prime fields
//! `GF(p)`, with the integers `0, ..., p - 1` as the elements.
//!
//! Python's elements are unbounded `int`s; here they are `u64` with every
//! product taken in `u128`, so any modulus below `2**64` works (the
//! modular algorithms of the package use word-sized primes).

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::domain::{ConvertFrom, Domain};
use super::integerring::IntegerRing;
use super::rationalfield::RationalField;

/// The field of integers modulo the prime `p` (the primality is not
/// checked, as in Python).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiniteField {
    p: u64,
}

impl FiniteField {
    /// `Value` for a modulus below 2.
    pub fn new(p: u64) -> PolyResult<FiniteField> {
        if p < 2 {
            return Err(value_error(format!("the modulus must be a prime, got {}", p)));
        }
        Ok(FiniteField { p })
    }

    /// The modulus.
    pub fn p(&self) -> u64 {
        self.p
    }

    fn reduce(&self, n: u128) -> u64 {
        (n % self.p as u128) as u64
    }

    /// `a**-1 mod p` by the extended Euclidean algorithm (Python's
    /// `pow(a, -1, p)`, a `ValueError` when `a` is not invertible).
    fn modular_inverse(&self, a: u64) -> PolyResult<u64> {
        let (mut r0, mut r1): (i128, i128) = (self.p as i128, (a % self.p) as i128);
        let (mut t0, mut t1): (i128, i128) = (0, 1);
        while r1 != 0 {
            let q: i128 = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        if r0 != 1 {
            return Err(value_error("base is not invertible for the given modulus".to_string()));
        }
        Ok(t0.rem_euclid(self.p as i128) as u64)
    }
}

/// The prime field with `p` elements.
#[allow(non_snake_case)]
pub fn GF(p: u64) -> PolyResult<FiniteField> {
    FiniteField::new(p)
}

impl Domain for FiniteField {
    type Elem = u64;

    fn to_expr(&self, a: &u64) -> crate::error::PolyResult<crate::core::Basic> {
        Ok(crate::core::Basic::integer(Int::from(*a)))
    }

    fn from_expr(&self, e: &crate::core::Basic) -> crate::error::PolyResult<u64> {
        if crate::core::expr::is_integer_class(e) {
            return Ok(self.from_integer(e.as_rational().expect("an integer").numerator()));
        }
        Err(crate::error::PolyError::Value(format!("{} is not an integer", e)))
    }

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
        Int::from(self.p)
    }

    fn zero(&self) -> u64 {
        0
    }

    fn one(&self) -> u64 {
        1
    }

    fn is_zero(&self, a: &u64) -> bool {
        *a == 0
    }

    fn is_one(&self, a: &u64) -> bool {
        *a == 1
    }

    fn is_unit(&self, a: &u64) -> bool {
        *a != 0
    }

    fn add(&self, a: &u64, b: &u64) -> u64 {
        self.reduce(*a as u128 + *b as u128)
    }

    fn sub(&self, a: &u64, b: &u64) -> u64 {
        self.reduce(*a as u128 + self.p as u128 - (*b % self.p) as u128)
    }

    fn neg(&self, a: &u64) -> u64 {
        self.reduce(self.p as u128 - (*a % self.p) as u128)
    }

    fn mul(&self, a: &u64, b: &u64) -> u64 {
        self.reduce(*a as u128 * *b as u128)
    }

    fn exquo(&self, a: &u64, b: &u64) -> PolyResult<u64> {
        if *b == 0 {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(self.mul(a, &self.modular_inverse(*b)?))
    }

    fn quo(&self, a: &u64, b: &u64) -> PolyResult<u64> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &u64, b: &u64) -> PolyResult<u64> {
        if *b == 0 {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(0)
    }

    fn invert(&self, a: &u64) -> PolyResult<u64> {
        if *a == 0 {
            return Err(zero_division("0 has no inverse"));
        }
        self.modular_inverse(*a)
    }

    fn pow_signed(&self, a: &u64, n: i64) -> PolyResult<u64> {
        if n < 0 {
            let inverse: u64 = self.modular_inverse(*a)?;
            return Ok(self.pow(&inverse, n.unsigned_abs()));
        }
        Ok(self.pow(a, n as u64))
    }

    fn gcd(&self, a: &u64, b: &u64) -> u64 {
        if *a == 0 && *b == 0 {
            0
        } else {
            1
        }
    }

    fn lcm(&self, a: &u64, b: &u64) -> u64 {
        if *a == 0 || *b == 0 {
            0
        } else {
            1
        }
    }

    fn gcdex(&self, a: &u64, b: &u64) -> PolyResult<(u64, u64, u64)> {
        if *a != 0 {
            return Ok((self.invert(a)?, 0, 1));
        }
        if *b != 0 {
            return Ok((0, self.invert(b)?, 1));
        }
        Ok((0, 0, 0))
    }

    fn canonical_unit(&self, a: &u64) -> u64 {
        if *a == 0 {
            1
        } else {
            self.invert(a).expect("a nonzero element of a prime field")
        }
    }

    fn has_dense_kernels(&self) -> bool {
        true
    }

    fn dense_mul(&self, f: &[u64], g: &[u64]) -> Vec<u64> {
        if f.is_empty() || g.is_empty() {
            return Vec::new();
        }
        let mut result: Vec<u64> = vec![0; f.len() + g.len() - 1];
        for (i, a) in f.iter().enumerate() {
            if *a == 0 {
                continue;
            }
            for (j, b) in g.iter().enumerate() {
                result[i + j] = self.add(&result[i + j], &self.mul(a, b));
            }
        }
        strip(result)
    }

    fn dense_div(&self, f: &[u64], g: &[u64]) -> PolyResult<(Vec<u64>, Vec<u64>)> {
        if g.is_empty() {
            return Err(zero_division("polynomial division by zero"));
        }
        if f.len() < g.len() {
            return Ok((Vec::new(), f.to_vec()));
        }
        let df: usize = f.len() - 1;
        let dg: usize = g.len() - 1;
        let inverse: u64 = self.modular_inverse(g[0])?;
        let mut quotient: Vec<u64> = vec![0; df - dg + 1];
        let mut remainder: Vec<u64> = f.iter().map(|c| c % self.p).collect();
        for i in 0..(df - dg + 1) {
            let c: u64 = remainder[i];
            if c == 0 {
                continue;
            }
            let q: u64 = self.mul(&c, &inverse);
            quotient[i] = q;
            for j in 1..(dg + 1) {
                remainder[i + j] = self.sub(&remainder[i + j], &self.mul(&q, &g[j]));
            }
        }
        let tail: Vec<u64> = remainder.split_off(df - dg + 1);
        Ok((strip(quotient), strip(tail)))
    }

    fn factor_dense(&self, f: &[u64]) -> PolyResult<(u64, Vec<(Vec<u64>, usize)>)> {
        Ok(crate::polys::galoistools::gf_factor(f, self))
    }

    fn from_int(&self, n: i64) -> u64 {
        (n as i128).rem_euclid(self.p as i128) as u64
    }

    fn from_integer(&self, n: &Int) -> u64 {
        u64::try_from(&mod_floor(n, &Int::from(self.p))).expect("a residue below the modulus")
    }

    fn to_str(&self, a: &u64) -> String {
        a.to_string()
    }

    fn elements(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        Box::new(0..self.p)
    }
}

fn strip(f: Vec<u64>) -> Vec<u64> {
    let i: usize = f.iter().take_while(|c| **c == 0).count();
    if i == 0 {
        return f;
    }
    f[i..].to_vec()
}

impl ConvertFrom<FiniteField> for FiniteField {
    fn convert(&self, a: &u64, k0: &FiniteField) -> PolyResult<u64> {
        if k0.p == self.p {
            return Ok(*a);
        }
        Err(value_error(format!(
            "no conversion from {} to {}",
            k0.name(),
            self.name()
        )))
    }
}

impl ConvertFrom<IntegerRing> for FiniteField {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<u64> {
        Ok(self.from_integer(a))
    }
}

impl ConvertFrom<RationalField> for FiniteField {
    fn convert(&self, a: &Rat, _k0: &RationalField) -> PolyResult<u64> {
        if a.is_int() {
            return Ok(self.from_integer(a.numerator()));
        }
        Err(value_error(format!("{} is not an integer", a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;

    #[test]
    fn doctests() {
        let f: FiniteField = GF(7).unwrap();
        assert_eq!(f.add(&5, &4), 2);
        assert_eq!(f.mul(&3, &5), 1);
        assert_eq!(f.invert(&3).unwrap(), 5);
        assert_eq!(f.pow(&3, 100), 4);
        assert_eq!(f.from_int(-1), 6);
        assert_eq!(GF(3).unwrap().elements().collect::<Vec<u64>>(), vec![0, 1, 2]);
        assert!(GF(1).is_err());
    }

    #[test]
    fn large_modulus() {
        let p: u64 = 18446744073709551557; // the largest prime below 2**64
        let f: FiniteField = GF(p).unwrap();
        let a: u64 = p - 1;
        assert_eq!(f.mul(&a, &a), 1);
        assert_eq!(f.add(&a, &a), p - 2);
        assert_eq!(f.mul(&f.invert(&12345).unwrap(), &12345), 1);
    }
}
