//! Port of `sympy_extras/polys/domains/polynomialring.py`: polynomial rings
//! `K[x]` as coefficient domains. A polynomial in several variables is a
//! polynomial over a polynomial ring (`K[y][x]`: `PolynomialRing<
//! PolynomialRing<K>>`), and the generic dense algorithms run on the tower.
//!
//! The depth of the tower is a type here, so it is fixed at compile time;
//! the algorithms whose number of variables is only known at run time use
//! the sparse rings ([`super::sparsepolynomialring`]), as the Python package
//! does when it needs one element type. The conversions to and from
//! expressions wait for the typed core.

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::super::densearith::{
    dup_add, dup_exquo, dup_exquo_ground, dup_mul, dup_neg, dup_pdiv, dup_pow, dup_pquo, dup_prem, dup_quo, dup_rem,
    dup_sub,
};
use super::super::densebasic::{dup_convert, dup_strip_vec};
use super::super::densetools::dup_diff;
use super::super::euclidtools::{dup_gcd, dup_gcdex, dup_lcm};
use super::domain::{ConvertFrom, Domain};
use super::finitefield::FiniteField;
use super::integerring::IntegerRing;
use super::rationalfield::RationalField;

/// The ring `K[x]` of dense univariate polynomials over the domain `K` (the
/// `ground` domain) in the variable named `variable`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialRing<K: Domain> {
    pub ground: K,
    pub variable: String,
}

impl<K: Domain> PolynomialRing<K> {
    pub fn new(ground: K, variable: &str) -> PolynomialRing<K> {
        PolynomialRing {
            ground,
            variable: variable.to_string(),
        }
    }

    /// The derivative.
    pub fn diff(&self, a: &[K::Elem]) -> Vec<K::Elem> {
        dup_diff(a, 1, &self.ground)
    }
}

impl<K: Domain> Domain for PolynomialRing<K> {
    type Elem = Vec<K::Elem>;

    fn name(&self) -> String {
        format!("{}[{}]", self.ground.name(), self.variable)
    }

    fn is_field(&self) -> bool {
        false
    }

    fn is_ordered(&self) -> bool {
        self.ground.is_ordered()
    }

    fn characteristic(&self) -> u64 {
        self.ground.characteristic()
    }

    fn zero(&self) -> Vec<K::Elem> {
        Vec::new()
    }

    fn one(&self) -> Vec<K::Elem> {
        vec![self.ground.one()]
    }

    fn is_zero(&self, a: &Vec<K::Elem>) -> bool {
        a.is_empty()
    }

    fn is_one(&self, a: &Vec<K::Elem>) -> bool {
        a.len() == 1 && self.ground.is_one(&a[0])
    }

    /// Whether the leading coefficient is negative (the sign of a
    /// polynomial, for the canonical forms).
    fn is_negative(&self, a: &Vec<K::Elem>) -> bool {
        !a.is_empty() && self.ground.is_negative(&a[0])
    }

    fn is_unit(&self, a: &Vec<K::Elem>) -> bool {
        a.len() == 1 && self.ground.is_unit(&a[0])
    }

    fn add(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> Vec<K::Elem> {
        dup_add(a, b, &self.ground)
    }

    fn sub(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> Vec<K::Elem> {
        dup_sub(a, b, &self.ground)
    }

    fn neg(&self, a: &Vec<K::Elem>) -> Vec<K::Elem> {
        dup_neg(a, &self.ground)
    }

    fn mul(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> Vec<K::Elem> {
        dup_mul(a, b, &self.ground)
    }

    fn pow(&self, a: &Vec<K::Elem>, n: u64) -> Vec<K::Elem> {
        dup_pow(a, n as usize, &self.ground)
    }

    fn pow_signed(&self, a: &Vec<K::Elem>, n: i64) -> PolyResult<Vec<K::Elem>> {
        if n < 0 {
            return Err(value_error(format!("negative exponent in {}", self.name())));
        }
        Ok(self.pow(a, n as u64))
    }

    fn exquo(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        if b.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        if self.ground.is_field() {
            return dup_exquo(a, b, &self.ground);
        }
        if a.is_empty() {
            return Ok(Vec::new());
        }
        if a.len() < b.len() {
            return Err(value_error(format!("{:?} does not divide {:?}", b, a)));
        }
        // over a ring: lc(b)**(deg a - deg b + 1) a = q b + r, and the
        // quotient is q divided exactly by that power when r vanishes
        let (q, r) = dup_pdiv(a, b, &self.ground)?;
        if !r.is_empty() {
            return Err(value_error(format!("{:?} does not divide {:?}", b, a)));
        }
        let scale: K::Elem = self.ground.pow(&b[0], (a.len() - b.len() + 1) as u64);
        dup_exquo_ground(&q, &scale, &self.ground)
    }

    fn quo(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        if self.ground.is_field() {
            dup_quo(a, b, &self.ground)
        } else {
            dup_pquo(a, b, &self.ground)
        }
    }

    fn rem(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        if self.ground.is_field() {
            dup_rem(a, b, &self.ground)
        } else {
            dup_prem(a, b, &self.ground)
        }
    }

    fn invert(&self, a: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        if self.is_unit(a) {
            return Ok(vec![self.ground.invert(&a[0])?]);
        }
        Err(value_error(format!("{:?} is not a unit of {}", a, self.name())))
    }

    fn gcd(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> Vec<K::Elem> {
        dup_gcd(a, b, &self.ground)
    }

    fn lcm(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> Vec<K::Elem> {
        dup_lcm(a, b, &self.ground)
    }

    fn gcdex(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> PolyResult<(Vec<K::Elem>, Vec<K::Elem>, Vec<K::Elem>)> {
        dup_gcdex(a, b, &self.ground)
    }

    fn canonical_unit(&self, a: &Vec<K::Elem>) -> Vec<K::Elem> {
        match a.first() {
            None => self.one(),
            Some(lc) => vec![self.ground.canonical_unit(lc)],
        }
    }

    fn from_int(&self, n: i64) -> Vec<K::Elem> {
        let c: K::Elem = self.ground.from_int(n);
        if self.ground.is_zero(&c) {
            Vec::new()
        } else {
            vec![c]
        }
    }

    fn from_integer(&self, n: &Int) -> Vec<K::Elem> {
        let c: K::Elem = self.ground.from_integer(n);
        if self.ground.is_zero(&c) {
            Vec::new()
        } else {
            vec![c]
        }
    }
}

/// A polynomial over `K0` in the same variable (`Value` when the variables
/// differ or a coefficient has no image).
impl<K0: Domain, K1: ConvertFrom<K0>> ConvertFrom<PolynomialRing<K0>> for PolynomialRing<K1> {
    fn convert(&self, a: &Vec<K0::Elem>, k0: &PolynomialRing<K0>) -> PolyResult<Vec<K1::Elem>> {
        if k0.variable != self.variable {
            return Err(value_error(format!(
                "no conversion from {} to {}",
                k0.name(),
                self.name()
            )));
        }
        dup_convert(a, &k0.ground, &self.ground)
    }
}

/// A ground element as a constant polynomial.
macro_rules! convert_ground {
    ($source:ty) => {
        impl<K: ConvertFrom<$source>> ConvertFrom<$source> for PolynomialRing<K> {
            fn convert(&self, a: &<$source as Domain>::Elem, k0: &$source) -> PolyResult<Vec<K::Elem>> {
                let c: K::Elem = self.ground.convert(a, k0)?;
                Ok(dup_strip_vec(vec![c], &self.ground))
            }
        }
    };
}

convert_ground!(IntegerRing);
convert_ground!(RationalField);
convert_ground!(FiniteField);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::densebasic::testing::zz;
    use crate::polys::domains::ZZ;

    #[test]
    fn doctests() {
        let r: PolynomialRing<IntegerRing> = PolynomialRing::new(ZZ, "y");
        assert_eq!(r.name(), "ZZ[y]");
        assert_eq!(r.mul(&zz(&[1, 1]), &zz(&[1, -1])), zz(&[1, 0, -1]));
        assert_eq!(r.gcd(&zz(&[2, 0, -2]), &zz(&[4, 4])), zz(&[2, 2]));
        let s: PolynomialRing<PolynomialRing<IntegerRing>> = PolynomialRing::new(r.clone(), "x");
        assert_eq!(s.name(), "ZZ[y][x]");
        // the gcd in ZZ[y][x] of (x + y)(x - y) and (x + y)**2 is x + y
        let f: Vec<Vec<Int>> = vec![zz(&[1]), vec![], zz(&[-1, 0, 0])];
        let g: Vec<Vec<Int>> = vec![zz(&[1]), zz(&[2, 0]), zz(&[1, 0, 0])];
        assert_eq!(s.gcd(&f, &g), vec![zz(&[1]), zz(&[1, 0])]);
        assert_eq!(
            s.exquo(&f, &vec![zz(&[1]), zz(&[1, 0])]).unwrap(),
            vec![zz(&[1]), zz(&[-1, 0])]
        );
        assert!(s.exquo(&f, &vec![zz(&[2]), zz(&[1, 0])]).is_err());
        assert_eq!(r.convert(&int(5), &ZZ).unwrap(), zz(&[5]));
    }
}
