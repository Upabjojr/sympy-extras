//! Simple algebraic extensions of any field, `K[t]/(f)` for a monic
//! irreducible `f` over the field `K`, ported from
//! `sympy_extras.polys.domains.quotientfield`: the elements are dense
//! polynomials over `K` of degree below that of `f` (highest degree first),
//! reduced after every product; inverses by the extended Euclidean
//! algorithm against `f` (an error at a zero divisor when `f` is not
//! irreducible). `AlgebraicField` with the rationals replaced by any field.

// Reference: K. O. Geddes, S. R. Czapor, G. Labahn, Algorithms for Computer Algebra,
// Kluwer, 1992, Section 8.8 (algebraic extensions).

use super::domain::Domain;
use crate::error::{value_error, zero_division, PolyError, PolyResult};
use crate::numbers::Int;
use crate::polys::densearith::{dup_add, dup_mul, dup_neg, dup_rem, dup_sub};
use crate::polys::densebasic::{dup_degree, dup_strip};
use crate::polys::euclidtools::dup_invert;

/// `K[t]/(modulus)`, `modulus` monic of positive degree and irreducible
/// over the field `K` (`QuotientField`).
#[derive(Debug, Clone, PartialEq)]
pub struct QuotientField<K: Domain> {
    pub ground: K,
    pub modulus: Vec<K::Elem>,
    pub degree: usize,
    name: String,
}

impl<K: Domain> QuotientField<K> {
    pub fn new(ground: K, modulus: Vec<K::Elem>) -> PolyResult<QuotientField<K>> {
        if !ground.is_field() {
            return Err(value_error(format!("{} is not a field", ground.name())));
        }
        if dup_degree(&modulus) < 1 || !ground.is_one(&modulus[0]) {
            return Err(value_error("the modulus must be monic of positive degree".to_string()));
        }
        let degree: usize = dup_degree(&modulus) as usize;
        let mut terms: Vec<String> = Vec::new();
        for (k, c) in modulus.iter().enumerate() {
            if ground.is_zero(c) {
                continue;
            }
            let n: usize = degree - k;
            let power: String = match n {
                0 => String::new(),
                1 => "t".to_string(),
                _ => format!("t**{}", n),
            };
            let text: String = if !power.is_empty() && ground.is_one(c) {
                power
            } else if !power.is_empty() {
                format!("{}*{}", ground.to_str(c), power)
            } else {
                ground.to_str(c)
            };
            terms.push(text);
        }
        let name: String = format!("{}[t]/({})", ground.name(), terms.join(" + ").replace("+ -", "- "));
        Ok(QuotientField {
            ground,
            modulus,
            degree,
            name,
        })
    }

    /// The class of `t`.
    pub fn generator(&self) -> Vec<K::Elem> {
        self.reduce(vec![self.ground.one(), self.ground.zero()])
    }

    /// The representative of degree below that of the modulus.
    pub fn reduce(&self, a: Vec<K::Elem>) -> Vec<K::Elem> {
        if a.len() <= self.degree {
            return dup_strip(&a, &self.ground);
        }
        dup_rem(&a, &self.modulus, &self.ground).expect("a monic modulus")
    }

    /// The image of an element of the ground field.
    pub fn from_ground(&self, c: &K::Elem) -> Vec<K::Elem> {
        dup_strip(std::slice::from_ref(c), &self.ground)
    }
}

impl<K: Domain> Domain for QuotientField<K> {
    type Elem = Vec<K::Elem>;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        false
    }

    fn characteristic(&self) -> Int {
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

    fn is_unit(&self, a: &Vec<K::Elem>) -> bool {
        !a.is_empty()
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
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        self.reduce(dup_mul(a, b, &self.ground))
    }

    fn invert(&self, a: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        if a.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name)));
        }
        if a.len() == 1 {
            return Ok(vec![self.ground.invert(&a[0])?]);
        }
        dup_invert(a, &self.modulus, &self.ground)
    }

    fn exquo(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        if a.is_empty() {
            if b.is_empty() {
                return Err(zero_division(&format!("division by zero in {}", self.name)));
            }
            return Ok(Vec::new());
        }
        Ok(self.mul(a, &self.invert(b)?))
    }

    fn quo(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> PolyResult<Vec<K::Elem>> {
        if b.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name)));
        }
        Ok(Vec::new())
    }

    fn gcd(&self, a: &Vec<K::Elem>, b: &Vec<K::Elem>) -> Vec<K::Elem> {
        if a.is_empty() && b.is_empty() {
            Vec::new()
        } else {
            self.one()
        }
    }

    #[allow(clippy::type_complexity)]
    fn gcdex(&self, _a: &Vec<K::Elem>, _b: &Vec<K::Elem>) -> PolyResult<(Vec<K::Elem>, Vec<K::Elem>, Vec<K::Elem>)> {
        // the Python class inherits Domain.gcdex, which raises
        Err(PolyError::NotImplemented(format!("gcdex in {}", self.name)))
    }

    fn size(&self, a: &Vec<K::Elem>) -> u64 {
        a.iter().map(|c| self.ground.size(c)).sum::<u64>() + a.len() as u64
    }

    fn from_int(&self, n: i64) -> Vec<K::Elem> {
        dup_strip(&[self.ground.from_int(n)], &self.ground)
    }

    fn from_integer(&self, n: &Int) -> Vec<K::Elem> {
        dup_strip(&[self.ground.from_integer(n)], &self.ground)
    }

    fn to_str(&self, a: &Vec<K::Elem>) -> String {
        format!(
            "[{}]",
            a.iter()
                .map(|c| self.ground.to_str(c))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers::Rat;
    use crate::polys::domains::QQ;

    #[test]
    fn sqrt_two() {
        // the docstring: QQ(sqrt(2))
        let f = QuotientField::new(QQ, vec![Rat::ONE, Rat::ZERO, Rat::from(-2)]).unwrap();
        let a = f.add(&f.generator(), &f.one());
        assert_eq!(f.mul(&a, &a), vec![Rat::from(2), Rat::from(3)]);
        assert_eq!(f.mul(&a, &f.invert(&a).unwrap()), f.one());
        assert_eq!(f.name(), "QQ[t]/(t**2 - 2)");
        assert!(QuotientField::new(QQ, vec![Rat::from(2), Rat::ONE]).is_err());
        assert!(f.invert(&f.zero()).is_err());
    }

    #[test]
    fn zero_divisor() {
        // t**2 - 1 is not irreducible: t - 1 has no inverse
        let f = QuotientField::new(QQ, vec![Rat::ONE, Rat::ZERO, Rat::from(-1)]).unwrap();
        assert!(f.invert(&vec![Rat::ONE, Rat::from(-1)]).is_err());
    }
}
