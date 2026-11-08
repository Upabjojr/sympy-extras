//! Port of `sympy_extras/polys/domains/fractionfield.py`: fields of
//! fractions `K(x)` of polynomial rings, elements pairs `(numerator,
//! denominator)` of polynomials over `K`, reduced (coprime, the denominator
//! in canonical form).
//!
//! The conversions to and from expressions (`to_expr`, `from_expr`, and the
//! conversion between different fraction fields, which goes through them)
//! wait for the typed core.

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::super::densearith::{dup_exquo, dup_exquo_ground};
use super::super::densebasic::dup_is_one;
use super::super::euclidtools::dup_gcd;
use super::domain::{ConvertFrom, Domain};
use super::finitefield::FiniteField;
use super::integerring::IntegerRing;
use super::polynomialring::PolynomialRing;
use super::rationalfield::RationalField;

/// A reduced fraction of two dense polynomials over `T`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FracElement<T> {
    pub numer: Vec<T>,
    pub denom: Vec<T>,
}

/// The field of fractions `K(x)` of the polynomial ring `K[x]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FractionField<K: Domain> {
    pub ring: PolynomialRing<K>,
}

impl<K: Domain> FractionField<K> {
    pub fn new(ring: PolynomialRing<K>) -> FractionField<K> {
        FractionField { ring }
    }

    fn ground(&self) -> &K {
        &self.ring.ground
    }

    /// The reduced fraction `numer/denom` (`ZeroDivision` for a zero
    /// denominator).
    pub fn new_element(&self, numer: Vec<K::Elem>, denom: Vec<K::Elem>) -> PolyResult<FracElement<K::Elem>> {
        if denom.is_empty() {
            return Err(zero_division(&format!("zero denominator in {}", self.name())));
        }
        if numer.is_empty() {
            return Ok(FracElement {
                numer: Vec::new(),
                denom: self.ring.one(),
            });
        }
        let k: &K = self.ground();
        let (mut numer, mut denom): (Vec<K::Elem>, Vec<K::Elem>) = (numer, denom);
        let g: Vec<K::Elem> = dup_gcd(&numer, &denom, k);
        if !dup_is_one(&g, k) {
            numer = dup_exquo(&numer, &g, k).expect("the gcd divides the numerator");
            denom = dup_exquo(&denom, &g, k).expect("the gcd divides the denominator");
        }
        let u: Vec<K::Elem> = self.ring.canonical_unit(&denom);
        if !self.ring.is_one(&u) {
            numer = self.ring.mul(&numer, &u);
            denom = self.ring.mul(&denom, &u);
        }
        if k.is_field() {
            let lc: K::Elem = denom[0].clone();
            numer = dup_exquo_ground(&numer, &lc, k).expect("a nonzero element of a field");
            denom = dup_exquo_ground(&denom, &lc, k).expect("a nonzero element of a field");
        }
        Ok(FracElement { numer, denom })
    }

    fn reduced(&self, numer: Vec<K::Elem>, denom: Vec<K::Elem>) -> FracElement<K::Elem> {
        self.new_element(numer, denom)
            .expect("a product of nonzero denominators")
    }

    pub fn numer(&self, a: &FracElement<K::Elem>) -> Vec<K::Elem> {
        a.numer.clone()
    }

    pub fn denom(&self, a: &FracElement<K::Elem>) -> Vec<K::Elem> {
        a.denom.clone()
    }

    /// A polynomial as a fraction.
    pub fn from_ring(&self, a: &[K::Elem]) -> FracElement<K::Elem> {
        FracElement {
            numer: a.to_vec(),
            denom: self.ring.one(),
        }
    }
}

impl<K: Domain> Domain for FractionField<K> {
    type Elem = FracElement<K::Elem>;

    fn name(&self) -> String {
        format!("{}({})", self.ground().name(), self.ring.variable)
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        self.ring.is_ordered()
    }

    fn characteristic(&self) -> Int {
        self.ring.characteristic()
    }

    fn zero(&self) -> FracElement<K::Elem> {
        FracElement {
            numer: Vec::new(),
            denom: self.ring.one(),
        }
    }

    fn one(&self) -> FracElement<K::Elem> {
        FracElement {
            numer: self.ring.one(),
            denom: self.ring.one(),
        }
    }

    fn is_zero(&self, a: &FracElement<K::Elem>) -> bool {
        a.numer.is_empty()
    }

    fn is_one(&self, a: &FracElement<K::Elem>) -> bool {
        self.ring.is_one(&a.numer) && self.ring.is_one(&a.denom)
    }

    fn is_negative(&self, a: &FracElement<K::Elem>) -> bool {
        self.ring.is_negative(&a.numer)
    }

    fn is_unit(&self, a: &FracElement<K::Elem>) -> bool {
        !a.numer.is_empty()
    }

    fn add(&self, a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> FracElement<K::Elem> {
        let r: &PolynomialRing<K> = &self.ring;
        if a.denom == b.denom {
            return self.reduced(r.add(&a.numer, &b.numer), a.denom.clone());
        }
        self.reduced(
            r.add(&r.mul(&a.numer, &b.denom), &r.mul(&b.numer, &a.denom)),
            r.mul(&a.denom, &b.denom),
        )
    }

    fn sub(&self, a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> FracElement<K::Elem> {
        self.add(a, &self.neg(b))
    }

    fn neg(&self, a: &FracElement<K::Elem>) -> FracElement<K::Elem> {
        FracElement {
            numer: self.ring.neg(&a.numer),
            denom: a.denom.clone(),
        }
    }

    fn mul(&self, a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> FracElement<K::Elem> {
        let r: &PolynomialRing<K> = &self.ring;
        self.reduced(r.mul(&a.numer, &b.numer), r.mul(&a.denom, &b.denom))
    }

    fn exquo(&self, a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> PolyResult<FracElement<K::Elem>> {
        if b.numer.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        let r: &PolynomialRing<K> = &self.ring;
        self.new_element(r.mul(&a.numer, &b.denom), r.mul(&a.denom, &b.numer))
    }

    fn quo(&self, a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> PolyResult<FracElement<K::Elem>> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> PolyResult<FracElement<K::Elem>> {
        if b.numer.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(self.zero())
    }

    fn invert(&self, a: &FracElement<K::Elem>) -> PolyResult<FracElement<K::Elem>> {
        if a.numer.is_empty() {
            return Err(zero_division("0 has no inverse"));
        }
        self.new_element(a.denom.clone(), a.numer.clone())
    }

    fn gcd(&self, a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> FracElement<K::Elem> {
        if a.numer.is_empty() && b.numer.is_empty() {
            self.zero()
        } else {
            self.one()
        }
    }

    fn lcm(&self, a: &FracElement<K::Elem>, b: &FracElement<K::Elem>) -> FracElement<K::Elem> {
        if a.numer.is_empty() || b.numer.is_empty() {
            self.zero()
        } else {
            self.one()
        }
    }

    fn gcdex(
        &self,
        a: &FracElement<K::Elem>,
        b: &FracElement<K::Elem>,
    ) -> PolyResult<(FracElement<K::Elem>, FracElement<K::Elem>, FracElement<K::Elem>)> {
        if !a.numer.is_empty() {
            return Ok((self.invert(a)?, self.zero(), self.one()));
        }
        if !b.numer.is_empty() {
            return Ok((self.zero(), self.invert(b)?, self.one()));
        }
        Ok((self.zero(), self.zero(), self.zero()))
    }

    fn canonical_unit(&self, a: &FracElement<K::Elem>) -> FracElement<K::Elem> {
        if a.numer.is_empty() {
            self.one()
        } else {
            self.invert(a).expect("a nonzero fraction")
        }
    }

    fn from_int(&self, n: i64) -> FracElement<K::Elem> {
        FracElement {
            numer: self.ring.from_int(n),
            denom: self.ring.one(),
        }
    }

    fn from_integer(&self, n: &Int) -> FracElement<K::Elem> {
        FracElement {
            numer: self.ring.from_integer(n),
            denom: self.ring.one(),
        }
    }
}

impl<K: Domain> ConvertFrom<FractionField<K>> for FractionField<K> {
    /// The identity on this field; another field of fractions needs the
    /// conversion through expressions, not ported yet (`Value`).
    fn convert(&self, a: &FracElement<K::Elem>, k0: &FractionField<K>) -> PolyResult<FracElement<K::Elem>> {
        if k0 == self {
            return Ok(a.clone());
        }
        Err(value_error(format!(
            "no conversion from {} to {}",
            k0.name(),
            self.name()
        )))
    }
}

/// A polynomial or a ground element as a fraction.
macro_rules! convert_through_ring {
    ($source:ty, $($bound:tt)*) => {
        impl<K: Domain> ConvertFrom<$source> for FractionField<K>
        where
            K: ConvertFrom<$source>,
            $($bound)*
        {
            fn convert(&self, a: &<$source as Domain>::Elem, k0: &$source) -> PolyResult<FracElement<K::Elem>> {
                Ok(self.from_ring(&self.ring.convert(a, k0)?))
            }
        }
    };
}

convert_through_ring!(IntegerRing,);
convert_through_ring!(RationalField,);
convert_through_ring!(FiniteField,);

impl<K0: Domain, K: Domain> ConvertFrom<PolynomialRing<K0>> for FractionField<K>
where
    K: ConvertFrom<K0>,
{
    fn convert(&self, a: &Vec<K0::Elem>, k0: &PolynomialRing<K0>) -> PolyResult<FracElement<K::Elem>> {
        Ok(self.from_ring(&self.ring.convert(a, k0)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::densebasic::testing::{q, zz};
    use crate::polys::domains::{QQ, ZZ};

    #[test]
    fn doctests() {
        let f: FractionField<IntegerRing> = FractionField::new(PolynomialRing::new(ZZ, "x"));
        assert_eq!(f.name(), "ZZ(x)");
        let a = f.new_element(zz(&[1, 0, -1]), zz(&[1, 1])).unwrap();
        assert_eq!(
            a,
            FracElement {
                numer: zz(&[1, -1]),
                denom: zz(&[1])
            }
        );
        let inverse = f.invert(&f.new_element(zz(&[2, 0]), zz(&[1, 1])).unwrap()).unwrap();
        assert_eq!(
            inverse,
            FracElement {
                numer: zz(&[1, 1]),
                denom: zz(&[2, 0])
            }
        );
        let sum = f.add(
            &f.new_element(zz(&[1]), zz(&[1, 0])).unwrap(),
            &f.new_element(zz(&[1]), zz(&[1, 1])).unwrap(),
        );
        assert_eq!(
            sum,
            FracElement {
                numer: zz(&[2, 1]),
                denom: zz(&[1, 1, 0])
            }
        );
        assert!(f.new_element(zz(&[1]), Vec::new()).is_err());
    }

    #[test]
    fn rational_ground_and_signs() {
        // over QQ the denominator is monic; over ZZ its leading coefficient positive
        let g: FractionField<RationalField> = FractionField::new(PolynomialRing::new(QQ, "t"));
        let e = g.new_element(vec![q(3, 1)], vec![q(-2, 1), q(4, 1)]).unwrap();
        assert_eq!(
            e,
            FracElement {
                numer: vec![q(-3, 2)],
                denom: vec![q(1, 1), q(-2, 1)]
            }
        );
        let f: FractionField<IntegerRing> = FractionField::new(PolynomialRing::new(ZZ, "x"));
        let e = f.new_element(zz(&[3]), zz(&[-2, 4])).unwrap();
        assert_eq!(
            e,
            FracElement {
                numer: zz(&[-3]),
                denom: zz(&[2, -4])
            }
        );
        let one = f.mul(&e, &f.invert(&e).unwrap());
        assert!(f.is_one(&one));
        assert!(f.is_zero(&f.sub(&e, &e)));
        assert_eq!(f.convert(&int(4), &ZZ).unwrap(), f.from_int(4));
    }
}
