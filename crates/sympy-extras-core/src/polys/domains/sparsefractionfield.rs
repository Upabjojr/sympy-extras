//! Port of `sympy_extras/polys/domains/sparsefractionfield.py`: fields of
//! rational functions in several variables `K(y1, ..., yk)` on the sparse
//! polynomials of [`SparsePolynomialRing`]; the elements are pairs
//! `(numer, denom)` in lowest terms, the leading coefficient of the
//! denominator nonnegative (SymPy's `FracField`). The conversions from and
//! to expressions wait for the core.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/domains/
// sparsefractionfield.py): sympy/polys/fields.py (FracField, FracElement:
// new, the arithmetic, diff) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// ---------------------------------------------------------------------

use crate::error::{zero_division, PolyResult};
use crate::numbers::*;

use super::super::rings::PolyElement;
use super::domain::{ConvertFrom, Domain};
use super::integerring::IntegerRing;
use super::sparsepolynomialring::{SparseGround, SparsePolynomialRing};

/// `numer/denom` in lowest terms (built by [`SparseFractionField::new_element`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseFracElement<K: Domain> {
    pub numer: PolyElement<K>,
    pub denom: PolyElement<K>,
}

/// The field of fractions `K(y1, ..., yk)` of `K[y1, ..., yk]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseFractionField<K: SparseGround> {
    pub ring: SparsePolynomialRing<K>,
}

impl<K: SparseGround> SparseFractionField<K> {
    pub fn new(ring: SparsePolynomialRing<K>) -> SparseFractionField<K> {
        SparseFractionField { ring }
    }

    /// The fraction `numer/denom` in lowest terms; `ZeroDivision` for a
    /// zero denominator.
    pub fn new_element(&self, numer: &PolyElement<K>, denom: &PolyElement<K>) -> PolyResult<SparseFracElement<K>> {
        if denom.is_zero() {
            return Err(zero_division(&format!("zero denominator in {}", self.name())));
        }
        let (p, q) = self.ring.cancel(numer, denom);
        Ok(SparseFracElement { numer: p, denom: q })
    }

    fn fraction(&self, numer: &PolyElement<K>, denom: &PolyElement<K>) -> SparseFracElement<K> {
        self.new_element(numer, denom).expect("a nonzero denominator")
    }

    pub fn gens(&self) -> Vec<SparseFracElement<K>> {
        self.ring.ring.gens().iter().map(|g| self.from_ring(g)).collect()
    }

    pub fn from_ring(&self, a: &PolyElement<K>) -> SparseFracElement<K> {
        SparseFracElement {
            numer: a.clone(),
            denom: self.ring.one(),
        }
    }

    /// The derivative in the `i`-th variable.
    pub fn diff(&self, a: &SparseFracElement<K>, i: usize) -> SparseFracElement<K> {
        let numer: PolyElement<K> = a.numer.diff(i).mul(&a.denom).sub(&a.numer.mul(&a.denom.diff(i)));
        self.fraction(&numer, &a.denom.pow(2))
    }
}

impl<K: SparseGround> Domain for SparseFractionField<K> {
    type Elem = SparseFracElement<K>;

    /// `K(y1,...,yk)`.
    fn name(&self) -> String {
        format!("{}({})", self.ring.ground.name(), self.ring.names.join(","))
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        self.ring.ground.is_ordered()
    }

    fn characteristic(&self) -> Int {
        self.ring.characteristic()
    }

    fn zero(&self) -> SparseFracElement<K> {
        SparseFracElement {
            numer: self.ring.zero(),
            denom: self.ring.one(),
        }
    }

    fn one(&self) -> SparseFracElement<K> {
        SparseFracElement {
            numer: self.ring.one(),
            denom: self.ring.one(),
        }
    }

    fn is_zero(&self, a: &SparseFracElement<K>) -> bool {
        a.numer.is_zero()
    }

    fn is_one(&self, a: &SparseFracElement<K>) -> bool {
        self.ring.is_one(&a.numer) && self.ring.is_one(&a.denom)
    }

    fn is_negative(&self, a: &SparseFracElement<K>) -> bool {
        self.ring.is_negative(&a.numer)
    }

    fn is_unit(&self, a: &SparseFracElement<K>) -> bool {
        !a.numer.is_zero()
    }

    fn add(&self, a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> SparseFracElement<K> {
        if b.numer.is_zero() {
            return a.clone();
        }
        if a.numer.is_zero() {
            return b.clone();
        }
        if a.denom == b.denom {
            return self.fraction(&a.numer.add(&b.numer), &a.denom);
        }
        self.fraction(
            &a.numer.mul(&b.denom).add(&a.denom.mul(&b.numer)),
            &a.denom.mul(&b.denom),
        )
    }

    fn sub(&self, a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> SparseFracElement<K> {
        self.add(a, &self.neg(b))
    }

    fn neg(&self, a: &SparseFracElement<K>) -> SparseFracElement<K> {
        SparseFracElement {
            numer: a.numer.neg(),
            denom: a.denom.clone(),
        }
    }

    fn mul(&self, a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> SparseFracElement<K> {
        if a.numer.is_zero() || b.numer.is_zero() {
            return self.zero();
        }
        self.fraction(&a.numer.mul(&b.numer), &a.denom.mul(&b.denom))
    }

    /// `a**n`: the powers of a fraction in lowest terms are in lowest terms
    /// (the sign of the denominator made canonical).
    fn pow(&self, a: &SparseFracElement<K>, n: u64) -> SparseFracElement<K> {
        let (p, q) = self.ring.canonical_denominator(a.numer.pow(n), a.denom.pow(n));
        SparseFracElement { numer: p, denom: q }
    }

    fn pow_signed(&self, a: &SparseFracElement<K>, n: i64) -> PolyResult<SparseFracElement<K>> {
        if n >= 0 {
            return Ok(self.pow(a, n as u64));
        }
        if a.numer.is_zero() {
            return Err(zero_division("0 has no inverse"));
        }
        let m: u64 = n.unsigned_abs();
        let (p, q) = self.ring.canonical_denominator(a.denom.pow(m), a.numer.pow(m));
        Ok(SparseFracElement { numer: p, denom: q })
    }

    fn exquo(&self, a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> PolyResult<SparseFracElement<K>> {
        if b.numer.is_zero() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        self.new_element(&a.numer.mul(&b.denom), &a.denom.mul(&b.numer))
    }

    fn quo(&self, a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> PolyResult<SparseFracElement<K>> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> PolyResult<SparseFracElement<K>> {
        if b.numer.is_zero() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(self.zero())
    }

    fn invert(&self, a: &SparseFracElement<K>) -> PolyResult<SparseFracElement<K>> {
        if a.numer.is_zero() {
            return Err(zero_division("0 has no inverse"));
        }
        self.new_element(&a.denom, &a.numer)
    }

    fn gcd(&self, a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> SparseFracElement<K> {
        if a.numer.is_zero() && b.numer.is_zero() {
            self.zero()
        } else {
            self.one()
        }
    }

    fn lcm(&self, a: &SparseFracElement<K>, b: &SparseFracElement<K>) -> SparseFracElement<K> {
        if a.numer.is_zero() || b.numer.is_zero() {
            self.zero()
        } else {
            self.one()
        }
    }

    #[allow(clippy::type_complexity)]
    fn gcdex(
        &self,
        a: &SparseFracElement<K>,
        b: &SparseFracElement<K>,
    ) -> PolyResult<(SparseFracElement<K>, SparseFracElement<K>, SparseFracElement<K>)> {
        if !a.numer.is_zero() {
            return Ok((self.invert(a)?, self.zero(), self.one()));
        }
        if !b.numer.is_zero() {
            return Ok((self.zero(), self.invert(b)?, self.one()));
        }
        Ok((self.zero(), self.zero(), self.zero()))
    }

    fn canonical_unit(&self, a: &SparseFracElement<K>) -> SparseFracElement<K> {
        if a.numer.is_zero() {
            self.one()
        } else {
            self.invert(a).expect("a nonzero element")
        }
    }

    fn size(&self, a: &SparseFracElement<K>) -> u64 {
        self.ring.size(&a.numer) + self.ring.size(&a.denom)
    }

    fn from_int(&self, n: i64) -> SparseFracElement<K> {
        SparseFracElement {
            numer: self.ring.from_int(n),
            denom: self.ring.one(),
        }
    }

    fn from_integer(&self, n: &Int) -> SparseFracElement<K> {
        SparseFracElement {
            numer: self.ring.from_integer(n),
            denom: self.ring.one(),
        }
    }

    fn to_str(&self, a: &SparseFracElement<K>) -> String {
        if self.ring.is_one(&a.denom) {
            return a.numer.to_string();
        }
        format!("({})/({})", a.numer, a.denom)
    }
}

impl<K: SparseGround> ConvertFrom<SparseFractionField<K>> for SparseFractionField<K> {
    /// The same element in an equal field (Python converts between
    /// different fields through the expressions of the core).
    fn convert(&self, a: &SparseFracElement<K>, k0: &SparseFractionField<K>) -> PolyResult<SparseFracElement<K>> {
        if k0 != self {
            return Err(crate::error::value_error(format!(
                "the conversion from {} to {} goes through the core",
                k0.name(),
                self.name()
            )));
        }
        self.new_element(
            &self.ring.ring.from_dict(a.numer.terms.clone()),
            &self.ring.ring.from_dict(a.denom.terms.clone()),
        )
    }
}

impl<K: SparseGround> ConvertFrom<SparsePolynomialRing<K>> for SparseFractionField<K> {
    fn convert(&self, a: &PolyElement<K>, _k0: &SparsePolynomialRing<K>) -> PolyResult<SparseFracElement<K>> {
        Ok(self.from_ring(&self.ring.ring.from_dict(a.terms.clone())))
    }
}

impl<K: SparseGround> ConvertFrom<IntegerRing> for SparseFractionField<K> {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<SparseFracElement<K>> {
        Ok(self.from_integer(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::{RationalField, QQ};

    #[test]
    fn doctests() {
        let f: SparseFractionField<RationalField> =
            SparseFractionField::new(SparsePolynomialRing::sympy(QQ, vec!["x".to_string(), "y".to_string()]));
        assert_eq!(f.name(), "QQ(x,y)");
        let g = f.gens();
        let (x, y) = (&g[0], &g[1]);
        let two = f.from_int(2);
        // (x**2 - y**2)/(2*x + 2*y) = x/2 - y/2
        let a = f
            .exquo(
                &f.sub(&f.mul(x, x), &f.mul(y, y)),
                &f.add(&f.mul(&two, x), &f.mul(&two, y)),
            )
            .unwrap();
        assert_eq!(f.to_str(&a), "(x - y)/(2)");
        let s = f.add(&f.invert(x).unwrap(), &f.invert(y).unwrap());
        assert_eq!(f.to_str(&s), "(x + y)/(x*y)");
        let d = f.diff(&f.exquo(x, &f.add(x, y)).unwrap(), 0);
        assert_eq!(f.to_str(&d), "(y)/(x**2 + 2*x*y + y**2)");
        // the denominator's leading coefficient is made positive
        let n = f.exquo(&f.one(), &f.neg(x)).unwrap();
        assert_eq!(f.to_str(&n), "(-1)/(x)");
        assert_eq!(f.to_str(&f.pow_signed(&n, -2).unwrap()), "x**2");
        assert!(f.invert(&f.zero()).is_err());
    }
}
