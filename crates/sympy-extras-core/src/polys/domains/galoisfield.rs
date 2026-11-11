//! Port of `sympy_extras/polys/domains/galoisfield.py`: the finite fields
//! `GF(p**n) = GF(p)[a]/(m(a))` for an irreducible `m` of degree `n`, as
//! coefficient domains.
//!
//! The elements are polynomials in the generator `a` of degree below `n`
//! (dense lists of residues in `[0, p)`, highest degree first, no leading
//! zeros), reduced after every product; inverses come from the extended
//! Euclidean algorithm against `m`. Without a modulus, the first monic
//! irreducible polynomial of degree `n` is used (its lower coefficients read
//! as a number in base `p`). The prime is a machine word, as in
//! [`FiniteField`].
//!
//! The Python conversion to expressions waits for the core; [`Domain::to_str`]
//! prints the same string (`2*a + 1`), the generator by its name.

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;
use crate::random::Rng;

use super::super::densearith::{dup_add, dup_mul, dup_neg, dup_rem, dup_sub};
use super::super::densebasic::dup_strip;
use super::super::euclidtools::dup_invert;
use super::super::galoistools::gf_irreducible_p;
use super::domain::{ConvertFrom, Domain};
use super::finitefield::FiniteField;
use super::integerring::IntegerRing;

/// An element of `GF(p**n)`: residues, highest degree first.
pub type GaloisElement = Vec<u64>;

/// The first monic irreducible polynomial of degree `n` over the prime
/// field, the lower coefficients read as a number in base `p`.
fn first_irreducible(n: usize, field: &FiniteField) -> Vec<u64> {
    if n == 1 {
        return vec![1, 0];
    }
    let p: u64 = field.p();
    let mut index: Int = Int::ONE;
    let p_int: Int = Int::from(p);
    loop {
        let mut digits: Vec<u64> = Vec::new();
        let mut value: Int = index.clone();
        for _ in 0..n {
            let (q, r) = div_mod_floor(&value, &p_int);
            digits.push(u64::try_from(r).expect("a residue"));
            value = q;
        }
        let mut candidate: Vec<u64> = vec![1];
        candidate.extend(digits.iter().rev());
        if gf_irreducible_p(&candidate, field) {
            return candidate;
        }
        index += Int::ONE;
    }
}

/// The field with `p**n` elements, as `GF(p)[a]/(modulus)` (`GaloisField`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaloisField {
    pub prime_field: FiniteField,
    pub p: u64,
    pub degree: usize,
    /// `p**n`.
    pub order: Int,
    /// Monic irreducible of degree `n` over `GF(p)`, highest degree first.
    pub modulus: Vec<u64>,
    /// The name of the generator in the printed elements (`a`).
    pub generator_name: String,
}

impl GaloisField {
    /// `GaloisField(p, n, modulus)`: `ValueError` for a degree below one, a
    /// modulus which is not monic of degree `n` or not irreducible, and for
    /// a `p` which is not a prime.
    pub fn new(p: u64, n: usize, modulus: Option<&[Int]>) -> PolyResult<GaloisField> {
        GaloisField::with_generator(p, n, modulus, "a")
    }

    /// [`GaloisField::new`] with the name of the generator.
    pub fn with_generator(p: u64, n: usize, modulus: Option<&[Int]>, generator: &str) -> PolyResult<GaloisField> {
        if n < 1 {
            return Err(value_error("the degree must be positive".to_string()));
        }
        let prime_field: FiniteField = FiniteField::new(p)?;
        let p_int: Int = Int::from(p);
        let modulus: Vec<u64> = match modulus {
            None => first_irreducible(n, &prime_field),
            Some(m) => m
                .iter()
                .map(|c| u64::try_from(mod_floor(c, &p_int)).expect("a residue"))
                .collect(),
        };
        if modulus.len() != n + 1 || modulus[0] != 1 {
            return Err(value_error(format!("the modulus must be monic of degree {}", n)));
        }
        if n > 1 && !gf_irreducible_p(&modulus, &prime_field) {
            return Err(value_error(format!("the modulus is not irreducible over GF({})", p)));
        }
        Ok(GaloisField {
            prime_field,
            p,
            degree: n,
            order: p_int.pow(n),
            modulus,
            generator_name: generator.to_string(),
        })
    }

    /// The class of `a`.
    pub fn generator(&self) -> GaloisElement {
        self.reduce(&[1, 0])
    }

    fn reduce(&self, a: &[u64]) -> GaloisElement {
        let a: Vec<u64> = dup_strip(&a.iter().map(|c| c % self.p).collect::<Vec<u64>>(), &self.prime_field);
        if a.len() <= self.degree {
            return a;
        }
        dup_rem(&a, &self.modulus, &self.prime_field).expect("a monic modulus")
    }

    /// `a**p`.
    pub fn frobenius(&self, a: &GaloisElement) -> GaloisElement {
        self.pow(a, self.p)
    }

    /// The element whose `p`-th power is `a`: `a**(q/p)`.
    pub fn pth_root(&self, a: &GaloisElement) -> GaloisElement {
        self.pow_int(a, &(&self.order / Int::from(self.p)))
    }

    /// `a**n` for an exponent of any size.
    pub fn pow_int(&self, a: &GaloisElement, n: &Int) -> GaloisElement {
        let mut result: GaloisElement = self.one();
        for i in (0..bits(n) as usize).rev() {
            result = self.mul(&result, &result);
            if bit(n, i) {
                result = self.mul(&result, a);
            }
        }
        result
    }

    /// The trace to the prime field: `a + a**p + ... + a**(p**(n-1))`.
    pub fn trace(&self, a: &GaloisElement) -> u64 {
        let mut total: GaloisElement = self.zero();
        let mut power: GaloisElement = a.clone();
        for _ in 0..self.degree {
            total = self.add(&total, &power);
            power = self.frobenius(&power);
        }
        total.first().copied().unwrap_or(0)
    }

    /// A uniformly random element.
    pub fn random(&self, rng: &mut Rng) -> GaloisElement {
        let digits: Vec<u64> = (0..self.degree).map(|_| rng.randrange(self.p)).collect();
        dup_strip(&digits, &self.prime_field)
    }
}

impl Domain for GaloisField {
    type Elem = GaloisElement;

    fn name(&self) -> String {
        if self.degree > 1 {
            format!("GF({}**{})", self.p, self.degree)
        } else {
            format!("GF({})", self.p)
        }
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

    fn zero(&self) -> GaloisElement {
        Vec::new()
    }

    fn one(&self) -> GaloisElement {
        vec![1]
    }

    fn is_zero(&self, a: &GaloisElement) -> bool {
        a.is_empty()
    }

    fn is_one(&self, a: &GaloisElement) -> bool {
        *a == [1]
    }

    fn is_unit(&self, a: &GaloisElement) -> bool {
        !a.is_empty()
    }

    fn add(&self, a: &GaloisElement, b: &GaloisElement) -> GaloisElement {
        dup_add(a, b, &self.prime_field)
    }

    fn sub(&self, a: &GaloisElement, b: &GaloisElement) -> GaloisElement {
        dup_sub(a, b, &self.prime_field)
    }

    fn neg(&self, a: &GaloisElement) -> GaloisElement {
        dup_neg(a, &self.prime_field)
    }

    fn mul(&self, a: &GaloisElement, b: &GaloisElement) -> GaloisElement {
        self.reduce(&dup_mul(a, b, &self.prime_field))
    }

    fn exquo(&self, a: &GaloisElement, b: &GaloisElement) -> PolyResult<GaloisElement> {
        Ok(self.mul(a, &self.invert(b)?))
    }

    fn quo(&self, a: &GaloisElement, b: &GaloisElement) -> PolyResult<GaloisElement> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &GaloisElement, b: &GaloisElement) -> PolyResult<GaloisElement> {
        if b.is_empty() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        Ok(Vec::new())
    }

    fn invert(&self, a: &GaloisElement) -> PolyResult<GaloisElement> {
        if a.is_empty() {
            return Err(zero_division("0 has no inverse"));
        }
        dup_invert(a, &self.modulus, &self.prime_field)
    }

    fn gcd(&self, a: &GaloisElement, b: &GaloisElement) -> GaloisElement {
        if a.is_empty() && b.is_empty() {
            Vec::new()
        } else {
            self.one()
        }
    }

    fn lcm(&self, a: &GaloisElement, b: &GaloisElement) -> GaloisElement {
        if a.is_empty() || b.is_empty() {
            Vec::new()
        } else {
            self.one()
        }
    }

    fn gcdex(&self, a: &GaloisElement, b: &GaloisElement) -> PolyResult<(GaloisElement, GaloisElement, GaloisElement)> {
        if !a.is_empty() {
            return Ok((self.invert(a)?, Vec::new(), self.one()));
        }
        if !b.is_empty() {
            return Ok((Vec::new(), self.invert(b)?, self.one()));
        }
        Ok((Vec::new(), Vec::new(), Vec::new()))
    }

    fn canonical_unit(&self, a: &GaloisElement) -> GaloisElement {
        if a.is_empty() {
            self.one()
        } else {
            self.invert(a).expect("a nonzero element")
        }
    }

    fn from_int(&self, n: i64) -> GaloisElement {
        self.from_integer(&Int::from(n))
    }

    fn from_integer(&self, n: &Int) -> GaloisElement {
        let r: u64 = u64::try_from(mod_floor(n, &Int::from(self.p))).expect("a residue");
        if r == 0 {
            Vec::new()
        } else {
            vec![r]
        }
    }

    fn to_str(&self, a: &GaloisElement) -> String {
        let n: usize = a.len().saturating_sub(1);
        let mut parts: Vec<String> = Vec::new();
        for (i, c) in a.iter().enumerate() {
            if *c == 0 {
                continue;
            }
            let k: usize = n - i;
            let power: String = if k == 1 {
                self.generator_name.clone()
            } else {
                format!("{}**{}", self.generator_name, k)
            };
            parts.push(if k == 0 {
                c.to_string()
            } else if *c == 1 {
                power
            } else {
                format!("{}*{}", c, power)
            });
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        }
    }

    fn elements(&self) -> Box<dyn Iterator<Item = GaloisElement> + '_> {
        let p: Int = Int::from(self.p);
        let mut index: Int = Int::ZERO;
        Box::new(std::iter::from_fn(move || {
            if index >= self.order {
                return None;
            }
            let mut digits: Vec<u64> = Vec::new();
            let mut value: Int = index.clone();
            while !value.is_zero() {
                let (q, r) = div_mod_floor(&value, &p);
                digits.push(u64::try_from(r).expect("a residue"));
                value = q;
            }
            digits.reverse();
            index += Int::ONE;
            Some(digits)
        }))
    }
}

impl ConvertFrom<GaloisField> for GaloisField {
    fn convert(&self, a: &GaloisElement, k0: &GaloisField) -> PolyResult<GaloisElement> {
        if k0.modulus == self.modulus && k0.p == self.p {
            return Ok(a.clone());
        }
        Err(value_error(format!(
            "no conversion from {} to {}",
            k0.name(),
            self.name()
        )))
    }
}

impl ConvertFrom<FiniteField> for GaloisField {
    fn convert(&self, a: &u64, _k0: &FiniteField) -> PolyResult<GaloisElement> {
        Ok(self.from_integer(&Int::from(*a)))
    }
}

impl ConvertFrom<IntegerRing> for GaloisField {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<GaloisElement> {
        Ok(self.from_integer(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docstring_examples() {
        let k = GaloisField::new(2, 3, None).unwrap();
        assert_eq!(
            (k.name(), k.modulus.clone(), k.order.clone()),
            ("GF(2**3)".to_string(), vec![1, 0, 1, 1], int(8))
        );
        let a = k.generator();
        assert_eq!(
            (k.mul(&a, &a), k.pow(&a, 3), k.pow(&a, 7)),
            (vec![1, 0, 0], vec![1, 1], vec![1])
        );
        assert_eq!(k.mul(&k.invert(&vec![1, 1, 0]).unwrap(), &vec![1, 1, 0]), vec![1]);
        assert_eq!(k.elements().count(), 8);
        let f = GaloisField::new(3, 2, Some(&[int(1), int(0), int(1)])).unwrap();
        assert_eq!(f.pow(&f.generator(), 2), vec![2]);
        assert_eq!(f.to_str(&vec![2, 1]), "2*a + 1");
        assert!(GaloisField::new(2, 2, Some(&[int(1), int(0), int(1)])).is_err());
    }
}
