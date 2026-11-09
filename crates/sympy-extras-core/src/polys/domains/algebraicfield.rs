//! Port of `sympy_extras/polys/domains/algebraicfield.py`: algebraic number
//! fields `QQ(alpha)` as coefficient domains. The elements are polynomials
//! in `alpha` with rational coefficients of degree below that of the
//! minimal polynomial (dense, highest degree first), reduced after every
//! product; inverses come from the extended Euclidean algorithm against the
//! minimal polynomial.
//!
//! `alpha_expr` is the root as a node of the typed core; the conversions
//! `to_expr`/`from_expr` (and so the conversion between fields with
//! different minimal polynomials) wait for its evaluating functions.
//!
//! References
//! ==========
//!
//! 1. H. Cohen, *A Course in Computational Algebraic Number Theory*,
//!    Springer, 1993, Section 4.2.

use crate::core::basic::Basic;
use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::super::densearith::{dup_add, dup_mul, dup_neg, dup_rem, dup_sub};
use super::super::densebasic::{dup_degree, dup_strip};
use super::super::densetools::dup_monic;
use super::super::euclidtools::{dup_invert, dup_resultant};
use super::super::sqfreetools::dup_sqf_part;
use super::domain::{ConvertFrom, Domain};
use super::integerring::IntegerRing;
use super::polynomialring::PolynomialRing;
use super::rationalfield::{RationalField, QQ};

/// An element of `QQ(alpha)`: a polynomial in `alpha` of degree below that
/// of the minimal polynomial.
pub type AlgebraicElement = Vec<Rat>;

/// The field `QQ(alpha)` with `alpha` a root of the monic irreducible
/// `minpoly` (highest degree first).
#[derive(Debug, Clone, PartialEq)]
pub struct AlgebraicField {
    pub minpoly: Vec<Rat>,
    pub degree: usize,
    pub alpha_expr: Basic,
}

impl AlgebraicField {
    /// `Value` unless the minimal polynomial is monic of positive degree.
    pub fn new(minpoly: Vec<Rat>, alpha_expr: Basic) -> PolyResult<AlgebraicField> {
        if dup_degree(&minpoly) < 1 || !minpoly[0].is_one() {
            return Err(value_error(
                "the minimal polynomial must be monic of positive degree".to_string(),
            ));
        }
        let degree: usize = dup_degree(&minpoly) as usize;
        Ok(AlgebraicField {
            minpoly,
            degree,
            alpha_expr,
        })
    }

    /// The generator itself.
    pub fn alpha(&self) -> AlgebraicElement {
        let x: Vec<Rat> = vec![Rat::ONE, Rat::ZERO];
        if self.degree > 1 {
            x
        } else {
            self.reduce(x)
        }
    }

    /// A list of coefficients as an element: leading zeros removed, reduced
    /// modulo the minimal polynomial.
    pub fn reduce(&self, a: Vec<Rat>) -> AlgebraicElement {
        let a: Vec<Rat> = dup_strip(&a, &QQ);
        if dup_degree(&a) < self.degree as isize {
            return a;
        }
        dup_rem(&a, &self.minpoly, &QQ).expect("a monic modulus")
    }

    pub fn from_rational(&self, q: &Rat) -> AlgebraicElement {
        if q.is_zero() {
            Vec::new()
        } else {
            vec![q.clone()]
        }
    }

    /// The norm `N(a)`, the product of the conjugates: the resultant of the
    /// minimal polynomial and `a` as a polynomial in `alpha`.
    pub fn norm(&self, a: &AlgebraicElement) -> Rat {
        if a.is_empty() {
            return Rat::ZERO;
        }
        dup_resultant(&self.minpoly, a, &QQ)
    }

    /// The characteristic polynomial of `a` (in a new variable `t`): the
    /// resultant in `alpha` of the minimal polynomial and `t - a(alpha)`, of
    /// degree `[K : QQ]`.
    pub fn characteristic_polynomial(&self, a: &AlgebraicElement) -> Vec<Rat> {
        let r: PolynomialRing<RationalField> = PolynomialRing::new(QQ, "t");
        // t - a(alpha) as a polynomial in alpha with coefficients in QQ[t]
        let t: Vec<Rat> = vec![Rat::ONE, Rat::ZERO];
        let mut coefficients: Vec<Vec<Rat>> = a
            .iter()
            .map(|c| if c.is_zero() { Vec::new() } else { vec![-c.clone()] })
            .collect();
        match coefficients.last_mut() {
            Some(last) => *last = dup_add(last, &t, &QQ),
            None => coefficients = vec![t],
        }
        let m_t: Vec<Vec<Rat>> = self
            .minpoly
            .iter()
            .map(|c| if c.is_zero() { Vec::new() } else { vec![c.clone()] })
            .collect();
        let result: Vec<Rat> = dup_resultant(&m_t, &dup_strip(&coefficients, &r), &r);
        dup_monic(&result, &QQ).expect("a nonzero resultant")
    }

    /// The minimal polynomial of `a` over `QQ`: the square-free part of its
    /// characteristic polynomial.
    pub fn minimal_polynomial(&self, a: &AlgebraicElement) -> Vec<Rat> {
        dup_sqf_part(&self.characteristic_polynomial(a), &QQ)
    }
}

impl Domain for AlgebraicField {
    type Elem = AlgebraicElement;

    /// `QQ<alpha>`.
    fn name(&self) -> String {
        format!("QQ<{}>", self.alpha_expr)
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

    fn zero(&self) -> AlgebraicElement {
        Vec::new()
    }

    fn one(&self) -> AlgebraicElement {
        vec![Rat::ONE]
    }

    fn is_zero(&self, a: &AlgebraicElement) -> bool {
        a.is_empty()
    }

    fn is_one(&self, a: &AlgebraicElement) -> bool {
        a.len() == 1 && a[0].is_one()
    }

    fn is_unit(&self, a: &AlgebraicElement) -> bool {
        !a.is_empty()
    }

    fn add(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        dup_add(a, b, &QQ)
    }

    fn sub(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        dup_sub(a, b, &QQ)
    }

    fn neg(&self, a: &AlgebraicElement) -> AlgebraicElement {
        dup_neg(a, &QQ)
    }

    fn mul(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        self.reduce(dup_mul(a, b, &QQ))
    }

    fn exquo(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        Ok(self.mul(a, &self.invert(b)?))
    }

    fn quo(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &AlgebraicElement, b: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        if b.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(Vec::new())
    }

    fn invert(&self, a: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        if a.is_empty() {
            return Err(zero_division("0 has no inverse"));
        }
        dup_invert(a, &self.minpoly, &QQ)
    }

    fn gcd(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        if a.is_empty() && b.is_empty() {
            Vec::new()
        } else {
            self.one()
        }
    }

    fn lcm(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        if a.is_empty() || b.is_empty() {
            Vec::new()
        } else {
            self.one()
        }
    }

    #[allow(clippy::type_complexity)]
    fn gcdex(
        &self,
        a: &AlgebraicElement,
        b: &AlgebraicElement,
    ) -> PolyResult<(AlgebraicElement, AlgebraicElement, AlgebraicElement)> {
        if !a.is_empty() {
            return Ok((self.invert(a)?, Vec::new(), self.one()));
        }
        if !b.is_empty() {
            return Ok((Vec::new(), self.invert(b)?, self.one()));
        }
        Ok((Vec::new(), Vec::new(), Vec::new()))
    }

    fn canonical_unit(&self, a: &AlgebraicElement) -> AlgebraicElement {
        if a.is_empty() {
            self.one()
        } else {
            self.invert(a).expect("a nonzero element")
        }
    }

    fn factor_dense(
        &self,
        f: &[AlgebraicElement],
    ) -> PolyResult<(AlgebraicElement, Vec<(Vec<AlgebraicElement>, usize)>)> {
        super::super::factortools::dup_ext_factor(f, self)
    }

    fn from_int(&self, n: i64) -> AlgebraicElement {
        self.from_rational(&Rat::from(n))
    }

    fn from_integer(&self, n: &Int) -> AlgebraicElement {
        self.from_rational(&rat_from_int(n))
    }
}

impl ConvertFrom<AlgebraicField> for AlgebraicField {
    /// The same element when the minimal polynomials agree; `Value`
    /// otherwise (Python converts through the expressions of the core).
    fn convert(&self, a: &AlgebraicElement, k0: &AlgebraicField) -> PolyResult<AlgebraicElement> {
        if k0.minpoly == self.minpoly {
            return Ok(a.clone());
        }
        Err(value_error(format!(
            "the conversion from {} to {} goes through the core",
            k0.name(),
            self.name()
        )))
    }
}

impl ConvertFrom<RationalField> for AlgebraicField {
    fn convert(&self, a: &Rat, _k0: &RationalField) -> PolyResult<AlgebraicElement> {
        Ok(self.from_rational(a))
    }
}

impl ConvertFrom<IntegerRing> for AlgebraicField {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<AlgebraicElement> {
        Ok(self.from_integer(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::basic::Head;

    pub(crate) fn sqrt2() -> AlgebraicField {
        let alpha: Basic = Basic::new(Head::Pow, vec![Basic::int(2), Basic::half()]);
        AlgebraicField::new(vec![rat(1, 1), rat(0, 1), rat(-2, 1)], alpha).unwrap()
    }

    #[test]
    fn doctests() {
        let k = sqrt2();
        let a: AlgebraicElement = vec![rat(1, 1), rat(1, 1)]; // 1 + sqrt(2)
        assert_eq!(k.mul(&a, &a), vec![rat(2, 1), rat(3, 1)]);
        assert_eq!(k.invert(&a).unwrap(), vec![rat(1, 1), rat(-1, 1)]);
        assert_eq!(k.norm(&a), rat(-1, 1));
        assert_eq!(k.minimal_polynomial(&a), vec![rat(1, 1), rat(-2, 1), rat(-1, 1)]);
        assert_eq!(k.mul(&k.alpha(), &k.alpha()), vec![rat(2, 1)]);
        assert!(k.invert(&Vec::new()).is_err());
        assert!(AlgebraicField::new(vec![rat(2, 1), rat(0, 1)], Basic::one()).is_err());
        // degree one: alpha is the rational root
        let q = AlgebraicField::new(vec![rat(1, 1), rat(-3, 1)], Basic::int(3)).unwrap();
        assert_eq!(q.alpha(), vec![rat(3, 1)]);
        assert_eq!(
            k.characteristic_polynomial(&vec![rat(3, 1)]),
            vec![rat(1, 1), rat(-6, 1), rat(9, 1)]
        );
    }
}
