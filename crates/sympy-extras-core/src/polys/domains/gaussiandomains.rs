//! Port of `sympy_extras/polys/domains/gaussiandomains.py` (itself ported
//! from SymPy's `sympy/polys/domains/gaussiandomains.py`, BSD 3-clause,
//! see `sympy_extras/LICENSE-SymPy`): the Gaussian integers `ZZ_I` and the
//! Gaussian rationals `QQ_I`.
//!
//! The elements are `x + y*I` with integer ([`GaussianInteger`]) or
//! rational ([`GaussianRational`]) parts. `ZZ_I` is a Euclidean domain: the
//! division rounds the exact quotient to the nearest Gaussian integer, the
//! gcd is the last nonzero remainder of the Euclidean algorithm made
//! canonical (in the first quadrant). `QQ_I` is a field. The conversions
//! from and to expressions wait for the core.

use std::fmt;

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::domain::{ConvertFrom, Domain};
use super::integerring::IntegerRing;
use super::rationalfield::RationalField;

/// `x + y*I` with integers `x`, `y`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GaussianInteger {
    pub x: Int,
    pub y: Int,
}

/// `x + y*I` with rationals `x`, `y`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GaussianRational {
    pub x: Rat,
    pub y: Rat,
}

impl GaussianInteger {
    pub fn new(x: Int, y: Int) -> GaussianInteger {
        GaussianInteger { x, y }
    }

    pub fn from_i64(x: i64, y: i64) -> GaussianInteger {
        GaussianInteger::new(Int::from(x), Int::from(y))
    }

    pub fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    /// `0` to `3`: the quadrant (`0`: `x > 0, y >= 0`).
    pub fn quadrant(&self) -> usize {
        quadrant(int_sign(&self.x), int_sign(&self.y))
    }
}

impl GaussianRational {
    pub fn new(x: Rat, y: Rat) -> GaussianRational {
        GaussianRational { x, y }
    }

    pub fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    pub fn quadrant(&self) -> usize {
        quadrant(rat_sign(&self.x), rat_sign(&self.y))
    }
}

impl fmt::Display for GaussianInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GaussianInteger({}, {})", self.x, self.y)
    }
}

impl fmt::Display for GaussianRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GaussianRational({}, {})", self.x, self.y)
    }
}

fn int_sign(a: &Int) -> i32 {
    if a.is_zero() {
        0
    } else if is_negative(a) {
        -1
    } else {
        1
    }
}

fn rat_sign(a: &Rat) -> i32 {
    if a.is_zero() {
        0
    } else if rat_is_negative(a) {
        -1
    } else {
        1
    }
}

/// SymPy's `GaussianElement.quadrant` on the signs of the parts.
fn quadrant(x: i32, y: i32) -> usize {
    if y > 0 {
        return if x > 0 { 0 } else { 1 };
    }
    if y < 0 {
        return if x < 0 { 2 } else { 3 };
    }
    if x >= 0 {
        0
    } else {
        2
    }
}

/// The index in `(1, I, -1, -I)` of the inverse of the unit of a quadrant
/// (Python's `units[-quadrant]`).
fn inverse_unit_index(q: usize) -> usize {
    (4 - q) % 4
}

/// The Gaussian integers `ZZ_I`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GaussianIntegerRing;

/// The Gaussian rationals `QQ_I`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GaussianRationalField;

#[allow(non_upper_case_globals)]
pub const ZZ_I: GaussianIntegerRing = GaussianIntegerRing;
#[allow(non_upper_case_globals)]
pub const QQ_I: GaussianRationalField = GaussianRationalField;

impl GaussianIntegerRing {
    /// The powers of `I`: `1, I, -1, -I`.
    pub fn units(&self) -> [GaussianInteger; 4] {
        [
            GaussianInteger::from_i64(1, 0),
            GaussianInteger::from_i64(0, 1),
            GaussianInteger::from_i64(-1, 0),
            GaussianInteger::from_i64(0, -1),
        ]
    }

    /// SymPy's `GaussianInteger.__divmod__`: the quotient rounded to the
    /// nearest Gaussian integer, `|r| < |b|`.
    pub fn divmod_gaussian(
        &self,
        a: &GaussianInteger,
        b: &GaussianInteger,
    ) -> PolyResult<(GaussianInteger, GaussianInteger)> {
        if b.is_zero() {
            return Err(zero_division(&format!("divmod({}, 0)", a)));
        }
        let (x, y) = (&b.x, &b.y);
        let p: Int = &a.x * x + &a.y * y;
        let q: Int = -(&a.x * y) + &a.y * x;
        let c: Int = x * x + y * y;
        let two_c: Int = Int::from(2u8) * &c;
        let qx: Int = div_floor(&(Int::from(2u8) * p + &c), &two_c);
        let qy: Int = div_floor(&(Int::from(2u8) * q + &c), &two_c);
        let quotient: GaussianInteger = GaussianInteger::new(qx, qy);
        let remainder: GaussianInteger = self.sub(a, &self.mul(&quotient, b));
        Ok((quotient, remainder))
    }

    /// The element times its canonical unit (SymPy's `normalize`).
    pub fn normalize(&self, a: &GaussianInteger) -> GaussianInteger {
        self.mul(a, &self.canonical_unit(a))
    }

    /// The lcm of the denominators of the parts (SymPy's `QQ_I.denom`).
    pub fn denom_of(&self, a: &GaussianRational) -> GaussianInteger {
        GaussianInteger::new(lcm(&denom(&a.x), &denom(&a.y)), Int::ZERO)
    }
}

/// `x + I*y` (`_gaussian_expr`).
fn gaussian_expr(x: crate::core::Basic, y: crate::core::Basic) -> crate::error::PolyResult<crate::core::Basic> {
    use crate::core::{add::add, mul::mul, Basic};
    add(&[x, mul(&[Basic::imaginary_unit(), y])?])
}

/// `(x, y)` of `x + y*I` (`_gaussian_parts`); `Value` otherwise.
fn gaussian_parts(e: &crate::core::Basic) -> crate::error::PolyResult<(crate::core::Basic, crate::core::Basic)> {
    use crate::core::expr::{as_coeff_add, as_coeff_mul};
    use crate::core::Basic;
    let (r, b) = as_coeff_add(e);
    if b.as_rational().is_some_and(|v| v.is_zero()) {
        return Ok((r, Basic::zero()));
    }
    let (c, rest) = as_coeff_mul(&b);
    if rest == Basic::imaginary_unit() {
        return Ok((r, c));
    }
    Err(crate::error::PolyError::Value(format!("{} is not Gaussian", e)))
}

impl Domain for GaussianIntegerRing {
    type Elem = GaussianInteger;

    fn to_expr(&self, a: &GaussianInteger) -> crate::error::PolyResult<crate::core::Basic> {
        gaussian_expr(
            crate::core::Basic::integer(a.x.clone()),
            crate::core::Basic::integer(a.y.clone()),
        )
    }

    fn from_expr(&self, e: &crate::core::Basic) -> crate::error::PolyResult<GaussianInteger> {
        let (x, y) = gaussian_parts(e)?;
        if !(crate::core::expr::is_integer_class(&x) && crate::core::expr::is_integer_class(&y)) {
            return Err(crate::error::PolyError::Value(format!(
                "{} is not a Gaussian integer",
                e
            )));
        }
        Ok(GaussianInteger {
            x: x.as_rational().expect("an integer").numerator().clone(),
            y: y.as_rational().expect("an integer").numerator().clone(),
        })
    }

    fn name(&self) -> String {
        "ZZ_I".to_string()
    }

    fn is_field(&self) -> bool {
        false
    }

    fn is_ordered(&self) -> bool {
        false
    }

    fn characteristic(&self) -> Int {
        Int::ZERO
    }

    fn zero(&self) -> GaussianInteger {
        GaussianInteger::from_i64(0, 0)
    }

    fn one(&self) -> GaussianInteger {
        GaussianInteger::from_i64(1, 0)
    }

    fn is_zero(&self, a: &GaussianInteger) -> bool {
        a.is_zero()
    }

    /// SymPy's `GaussianDomain.is_negative`.
    fn is_negative(&self, _a: &GaussianInteger) -> bool {
        false
    }

    fn is_unit(&self, a: &GaussianInteger) -> bool {
        self.units().contains(a)
    }

    fn add(&self, a: &GaussianInteger, b: &GaussianInteger) -> GaussianInteger {
        GaussianInteger::new(&a.x + &b.x, &a.y + &b.y)
    }

    fn sub(&self, a: &GaussianInteger, b: &GaussianInteger) -> GaussianInteger {
        GaussianInteger::new(&a.x - &b.x, &a.y - &b.y)
    }

    fn neg(&self, a: &GaussianInteger) -> GaussianInteger {
        GaussianInteger::new(-a.x.clone(), -a.y.clone())
    }

    fn mul(&self, a: &GaussianInteger, b: &GaussianInteger) -> GaussianInteger {
        GaussianInteger::new(&a.x * &b.x - &a.y * &b.y, &a.x * &b.y + &a.y * &b.x)
    }

    fn divmod(&self, a: &GaussianInteger, b: &GaussianInteger) -> PolyResult<(GaussianInteger, GaussianInteger)> {
        self.divmod_gaussian(a, b)
    }

    fn quo(&self, a: &GaussianInteger, b: &GaussianInteger) -> PolyResult<GaussianInteger> {
        Ok(self.divmod_gaussian(a, b)?.0)
    }

    fn rem(&self, a: &GaussianInteger, b: &GaussianInteger) -> PolyResult<GaussianInteger> {
        Ok(self.divmod_gaussian(a, b)?.1)
    }

    fn exquo(&self, a: &GaussianInteger, b: &GaussianInteger) -> PolyResult<GaussianInteger> {
        if b.is_zero() {
            return Err(zero_division(&format!("exquo({}, 0)", a)));
        }
        let (q, r) = self.divmod_gaussian(a, b)?;
        if !r.is_zero() {
            return Err(value_error(format!("{} does not divide {}", b, a)));
        }
        Ok(q)
    }

    fn invert(&self, a: &GaussianInteger) -> PolyResult<GaussianInteger> {
        if !self.is_unit(a) {
            return Err(value_error(format!("{} is not a unit", a)));
        }
        Ok(GaussianInteger::new(a.x.clone(), -a.y.clone()))
    }

    /// SymPy's `GaussianDomain.canonical_unit`: the inverse of the unit of
    /// the quadrant.
    fn canonical_unit(&self, a: &GaussianInteger) -> GaussianInteger {
        self.units()[inverse_unit_index(a.quadrant())].clone()
    }

    fn gcd(&self, a: &GaussianInteger, b: &GaussianInteger) -> GaussianInteger {
        let (mut a, mut b) = (a.clone(), b.clone());
        while !b.is_zero() {
            let r: GaussianInteger = self.rem(&a, &b).expect("a nonzero divisor");
            a = std::mem::replace(&mut b, r);
        }
        self.normalize(&a)
    }

    fn lcm(&self, a: &GaussianInteger, b: &GaussianInteger) -> GaussianInteger {
        // Python's quo, a ZeroDivisionError for a = b = 0
        self.quo(&self.mul(a, b), &self.gcd(a, b)).expect("lcm(0, 0)")
    }

    #[allow(clippy::type_complexity)]
    fn gcdex(
        &self,
        a: &GaussianInteger,
        b: &GaussianInteger,
    ) -> PolyResult<(GaussianInteger, GaussianInteger, GaussianInteger)> {
        let (mut a, mut b) = (a.clone(), b.clone());
        let (mut x_a, mut x_b, mut y_a, mut y_b) = (self.one(), self.zero(), self.zero(), self.one());
        while !b.is_zero() {
            let q: GaussianInteger = self.quo(&a, &b)?;
            let r: GaussianInteger = self.sub(&a, &self.mul(&q, &b));
            a = std::mem::replace(&mut b, r);
            let x: GaussianInteger = self.sub(&x_a, &self.mul(&q, &x_b));
            x_a = std::mem::replace(&mut x_b, x);
            let y: GaussianInteger = self.sub(&y_a, &self.mul(&q, &y_b));
            y_a = std::mem::replace(&mut y_b, y);
        }
        let unit: GaussianInteger = self.canonical_unit(&a);
        Ok((self.mul(&x_a, &unit), self.mul(&y_a, &unit), self.mul(&a, &unit)))
    }

    fn size(&self, a: &GaussianInteger) -> u64 {
        bits(&a.x).max(bits(&a.y)).max(1)
    }

    fn from_int(&self, n: i64) -> GaussianInteger {
        GaussianInteger::from_i64(n, 0)
    }

    fn from_integer(&self, n: &Int) -> GaussianInteger {
        GaussianInteger::new(n.clone(), Int::ZERO)
    }

    fn to_str(&self, a: &GaussianInteger) -> String {
        a.to_string()
    }
}

impl GaussianRationalField {
    pub fn units(&self) -> [GaussianRational; 4] {
        [
            GaussianRational::new(Rat::ONE, Rat::ZERO),
            GaussianRational::new(Rat::ZERO, Rat::ONE),
            GaussianRational::new(-Rat::ONE, Rat::ZERO),
            GaussianRational::new(Rat::ZERO, -Rat::ONE),
        ]
    }
}

impl Domain for GaussianRationalField {
    type Elem = GaussianRational;

    fn to_expr(&self, a: &GaussianRational) -> crate::error::PolyResult<crate::core::Basic> {
        gaussian_expr(
            crate::core::Basic::from_rational(a.x.clone()),
            crate::core::Basic::from_rational(a.y.clone()),
        )
    }

    fn from_expr(&self, e: &crate::core::Basic) -> crate::error::PolyResult<GaussianRational> {
        let (x, y) = gaussian_parts(e)?;
        match (x.as_rational(), y.as_rational()) {
            (Some(a), Some(b)) => Ok(GaussianRational {
                x: a.clone(),
                y: b.clone(),
            }),
            _ => Err(crate::error::PolyError::Value(format!(
                "{} is not a Gaussian rational",
                e
            ))),
        }
    }

    fn name(&self) -> String {
        "QQ_I".to_string()
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

    fn zero(&self) -> GaussianRational {
        GaussianRational::new(Rat::ZERO, Rat::ZERO)
    }

    fn one(&self) -> GaussianRational {
        GaussianRational::new(Rat::ONE, Rat::ZERO)
    }

    fn is_zero(&self, a: &GaussianRational) -> bool {
        a.is_zero()
    }

    fn is_negative(&self, _a: &GaussianRational) -> bool {
        false
    }

    fn add(&self, a: &GaussianRational, b: &GaussianRational) -> GaussianRational {
        GaussianRational::new(&a.x + &b.x, &a.y + &b.y)
    }

    fn sub(&self, a: &GaussianRational, b: &GaussianRational) -> GaussianRational {
        GaussianRational::new(&a.x - &b.x, &a.y - &b.y)
    }

    fn neg(&self, a: &GaussianRational) -> GaussianRational {
        GaussianRational::new(-a.x.clone(), -a.y.clone())
    }

    fn mul(&self, a: &GaussianRational, b: &GaussianRational) -> GaussianRational {
        GaussianRational::new(&a.x * &b.x - &a.y * &b.y, &a.x * &b.y + &a.y * &b.x)
    }

    /// SymPy's `GaussianRational.__truediv__`.
    fn exquo(&self, a: &GaussianRational, b: &GaussianRational) -> PolyResult<GaussianRational> {
        if b.is_zero() {
            return Err(zero_division(&format!("{} / 0", a)));
        }
        let (x, y) = (&b.x, &b.y);
        let c: Rat = x * x + y * y;
        Ok(GaussianRational::new(
            (&a.x * x + &a.y * y) / &c,
            (-(&a.x * y) + &a.y * x) / &c,
        ))
    }

    fn quo(&self, a: &GaussianRational, b: &GaussianRational) -> PolyResult<GaussianRational> {
        self.exquo(a, b)
    }

    fn rem(&self, a: &GaussianRational, b: &GaussianRational) -> PolyResult<GaussianRational> {
        if b.is_zero() {
            return Err(zero_division(&format!("{} % 0", a)));
        }
        Ok(self.zero())
    }

    fn invert(&self, a: &GaussianRational) -> PolyResult<GaussianRational> {
        self.exquo(&self.one(), a)
    }

    fn gcd(&self, a: &GaussianRational, b: &GaussianRational) -> GaussianRational {
        if a.is_zero() && b.is_zero() {
            self.zero()
        } else {
            self.one()
        }
    }

    fn lcm(&self, a: &GaussianRational, b: &GaussianRational) -> GaussianRational {
        if a.is_zero() || b.is_zero() {
            self.zero()
        } else {
            self.one()
        }
    }

    #[allow(clippy::type_complexity)]
    fn gcdex(
        &self,
        a: &GaussianRational,
        b: &GaussianRational,
    ) -> PolyResult<(GaussianRational, GaussianRational, GaussianRational)> {
        if !a.is_zero() {
            return Ok((self.invert(a)?, self.zero(), self.one()));
        }
        if !b.is_zero() {
            return Ok((self.zero(), self.invert(b)?, self.one()));
        }
        Ok((self.zero(), self.zero(), self.zero()))
    }

    fn canonical_unit(&self, a: &GaussianRational) -> GaussianRational {
        self.units()[inverse_unit_index(a.quadrant())].clone()
    }

    fn size(&self, a: &GaussianRational) -> u64 {
        [
            bits(&numer(&a.x)),
            bits(&denom(&a.x)),
            bits(&numer(&a.y)),
            bits(&denom(&a.y)),
            1,
        ]
        .into_iter()
        .max()
        .unwrap_or(1)
    }

    fn from_int(&self, n: i64) -> GaussianRational {
        GaussianRational::new(Rat::from(n), Rat::ZERO)
    }

    fn from_integer(&self, n: &Int) -> GaussianRational {
        GaussianRational::new(rat_from_int(n), Rat::ZERO)
    }

    fn to_str(&self, a: &GaussianRational) -> String {
        a.to_string()
    }
}

impl ConvertFrom<GaussianIntegerRing> for GaussianIntegerRing {
    fn convert(&self, a: &GaussianInteger, _k0: &GaussianIntegerRing) -> PolyResult<GaussianInteger> {
        Ok(a.clone())
    }
}

impl ConvertFrom<GaussianRationalField> for GaussianIntegerRing {
    /// `Value` unless both parts are integers.
    fn convert(&self, a: &GaussianRational, _k0: &GaussianRationalField) -> PolyResult<GaussianInteger> {
        if !denom(&a.x).is_one() || !denom(&a.y).is_one() {
            return Err(value_error(format!("{} is not a Gaussian integer", a)));
        }
        Ok(GaussianInteger::new(numer(&a.x), numer(&a.y)))
    }
}

impl ConvertFrom<IntegerRing> for GaussianIntegerRing {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<GaussianInteger> {
        Ok(self.from_integer(a))
    }
}

impl ConvertFrom<GaussianRationalField> for GaussianRationalField {
    fn convert(&self, a: &GaussianRational, _k0: &GaussianRationalField) -> PolyResult<GaussianRational> {
        Ok(a.clone())
    }
}

impl ConvertFrom<GaussianIntegerRing> for GaussianRationalField {
    fn convert(&self, a: &GaussianInteger, _k0: &GaussianIntegerRing) -> PolyResult<GaussianRational> {
        Ok(GaussianRational::new(rat_from_int(&a.x), rat_from_int(&a.y)))
    }
}

impl ConvertFrom<IntegerRing> for GaussianRationalField {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<GaussianRational> {
        Ok(self.from_integer(a))
    }
}

impl ConvertFrom<RationalField> for GaussianRationalField {
    fn convert(&self, a: &Rat, _k0: &RationalField) -> PolyResult<GaussianRational> {
        Ok(GaussianRational::new(a.clone(), Rat::ZERO))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(x: i64, y: i64) -> GaussianInteger {
        GaussianInteger::from_i64(x, y)
    }

    #[test]
    fn doctests() {
        let (a, b) = (g(3, 4), g(1, 2));
        assert_eq!(ZZ_I.mul(&a, &b), g(-5, 10));
        assert_eq!(ZZ_I.divmod(&a, &b).unwrap(), (g(2, 0), g(1, 0)));
        assert_eq!(ZZ_I.gcd(&g(4, 2), &g(2, 6)), g(4, 2));
        assert_eq!(ZZ_I.canonical_unit(&g(-2, 3)), g(0, -1));
        let qa: GaussianRational = QQ_I.convert(&a, &ZZ_I).unwrap();
        assert_eq!(
            QQ_I.invert(&qa).unwrap(),
            GaussianRational::new(rat(3, 25), rat(-4, 25))
        );
        let (s, t, h) = ZZ_I.gcdex(&g(4, 2), &g(2, 6)).unwrap();
        assert_eq!(ZZ_I.add(&ZZ_I.mul(&s, &g(4, 2)), &ZZ_I.mul(&t, &g(2, 6))), h);
        assert!(ZZ_I.exquo(&a, &b).is_err() && ZZ_I.invert(&a).is_err());
        assert_eq!(ZZ_I.invert(&g(0, 1)).unwrap(), g(0, -1));
        assert!(ZZ_I
            .convert(&GaussianRational::new(rat(1, 2), Rat::ZERO), &QQ_I)
            .is_err());
        assert_eq!(
            (g(0, 0).quadrant(), g(-1, 0).quadrant(), g(0, -1).quadrant()),
            (0, 2, 3)
        );
    }
}
