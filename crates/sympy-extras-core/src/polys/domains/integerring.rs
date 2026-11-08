//! Port of `sympy_extras/polys/domains/integerring.py`: the ring of
//! integers, with [`Int`] as the element type.

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::domain::{ConvertFrom, Domain};
use super::finitefield::FiniteField;
use super::rationalfield::RationalField;

/// The integers `ZZ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IntegerRing;

/// The ring of integers.
pub const ZZ: IntegerRing = IntegerRing;

impl Domain for IntegerRing {
    type Elem = Int;

    fn name(&self) -> String {
        "ZZ".to_string()
    }

    fn is_field(&self) -> bool {
        false
    }

    fn is_ordered(&self) -> bool {
        true
    }

    fn characteristic(&self) -> u64 {
        0
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

    fn is_negative(&self, a: &Int) -> bool {
        is_negative(a)
    }

    fn is_unit(&self, a: &Int) -> bool {
        abs(a).is_one()
    }

    fn add(&self, a: &Int, b: &Int) -> Int {
        a + b
    }

    fn sub(&self, a: &Int, b: &Int) -> Int {
        a - b
    }

    fn neg(&self, a: &Int) -> Int {
        -a
    }

    fn mul(&self, a: &Int, b: &Int) -> Int {
        a * b
    }

    fn pow(&self, a: &Int, n: u64) -> Int {
        a.pow(n as usize)
    }

    fn pow_signed(&self, a: &Int, n: i64) -> PolyResult<Int> {
        if n < 0 {
            return Err(value_error("negative exponent in ZZ".to_string()));
        }
        Ok(self.pow(a, n as u64))
    }

    fn exquo(&self, a: &Int, b: &Int) -> PolyResult<Int> {
        if b.is_zero() {
            return Err(zero_division("division by zero in ZZ"));
        }
        let (q, r) = div_rem_trunc(a, b);
        if !r.is_zero() {
            return Err(value_error(format!("{} does not divide {}", b, a)));
        }
        Ok(q)
    }

    /// The quotient of the truncated division (towards zero, so that
    /// `a = q b + r` with `|r| < |b|` and `r` of the sign of `a`).
    fn quo(&self, a: &Int, b: &Int) -> PolyResult<Int> {
        if b.is_zero() {
            return Err(zero_division("division by zero in ZZ"));
        }
        Ok(a / b)
    }

    fn rem(&self, a: &Int, b: &Int) -> PolyResult<Int> {
        let q: Int = self.quo(a, b)?;
        Ok(a - q * b)
    }

    fn invert(&self, a: &Int) -> PolyResult<Int> {
        if abs(a).is_one() {
            return Ok(a.clone());
        }
        Err(value_error(format!("{} is not a unit of ZZ", a)))
    }

    fn gcd(&self, a: &Int, b: &Int) -> Int {
        gcd(a, b)
    }

    fn lcm(&self, a: &Int, b: &Int) -> Int {
        if a.is_zero() || b.is_zero() {
            return Int::ZERO;
        }
        lcm(a, b)
    }

    /// `(s, t, g)` with `s a + t b = g = gcd(a, b) >= 0` (the extended
    /// Euclidean algorithm, with floor quotients as in Python).
    fn gcdex(&self, a: &Int, b: &Int) -> PolyResult<(Int, Int, Int)> {
        let (mut s0, mut s1): (Int, Int) = (Int::ONE, Int::ZERO);
        let (mut t0, mut t1): (Int, Int) = (Int::ZERO, Int::ONE);
        let (mut r0, mut r1): (Int, Int) = (a.clone(), b.clone());
        while !r1.is_zero() {
            let q: Int = div_floor(&r0, &r1);
            let r2: Int = &r0 - &q * &r1;
            r0 = std::mem::replace(&mut r1, r2);
            let s2: Int = &s0 - &q * &s1;
            s0 = std::mem::replace(&mut s1, s2);
            let t2: Int = &t0 - &q * &t1;
            t0 = std::mem::replace(&mut t1, t2);
        }
        if is_negative(&r0) {
            return Ok((-s0, -t0, -r0));
        }
        Ok((s0, t0, r0))
    }

    fn abs(&self, a: &Int) -> Int {
        abs(a)
    }

    fn canonical_unit(&self, a: &Int) -> Int {
        if is_negative(a) {
            -Int::ONE
        } else {
            Int::ONE
        }
    }

    fn size(&self, a: &Int) -> u64 {
        bits(a)
    }

    fn has_dense_kernels(&self) -> bool {
        true
    }

    fn dense_mul(&self, f: &[Int], g: &[Int]) -> Vec<Int> {
        if f.is_empty() || g.is_empty() {
            return Vec::new();
        }
        let mut result: Vec<Int> = vec![Int::ZERO; f.len() + g.len() - 1];
        for (i, a) in f.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in g.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        strip(result)
    }

    /// Division with the quotient coefficients obtained by exact division
    /// by the leading coefficient of `g` (`Value` when one is not exact).
    fn dense_div(&self, f: &[Int], g: &[Int]) -> PolyResult<(Vec<Int>, Vec<Int>)> {
        if g.is_empty() {
            return Err(zero_division("polynomial division by zero"));
        }
        if f.len() < g.len() {
            return Ok((Vec::new(), f.to_vec()));
        }
        let df: usize = f.len() - 1;
        let dg: usize = g.len() - 1;
        let lc: &Int = &g[0];
        let mut quotient: Vec<Int> = vec![Int::ZERO; df - dg + 1];
        let mut remainder: Vec<Int> = f.to_vec();
        for i in 0..(df - dg + 1) {
            if remainder[i].is_zero() {
                continue;
            }
            let (q, r) = div_mod_floor(&remainder[i], lc);
            if !r.is_zero() {
                return Err(value_error(format!("{} does not divide {}", lc, remainder[i])));
            }
            for j in 1..(dg + 1) {
                let product: Int = &q * &g[j];
                remainder[i + j] -= product;
            }
            quotient[i] = q;
        }
        let tail: Vec<Int> = remainder.split_off(df - dg + 1);
        Ok((strip(quotient), strip(tail)))
    }

    fn factor_dense(&self, f: &[Int]) -> PolyResult<(Int, Vec<(Vec<Int>, usize)>)> {
        crate::polys::factortools::dup_zz_factor(f, self)
    }

    fn isolate_real_roots_dense(&self, f: &[Int], eps: Option<&Rat>) -> Vec<((Rat, Rat), usize)> {
        let rational: Vec<Rat> = f.iter().map(rat_from_int).collect();
        crate::polys::rootisolation::dup_isolate_real_roots(&rational, eps)
    }

    fn count_real_roots_dense(&self, f: &[Int], inf: Option<&Rat>, sup: Option<&Rat>) -> i64 {
        let rational: Vec<Rat> = f.iter().map(rat_from_int).collect();
        crate::polys::rootisolation::dup_count_real_roots(&rational, inf, sup)
    }

    fn from_int(&self, n: i64) -> Int {
        Int::from(n)
    }

    fn from_integer(&self, n: &Int) -> Int {
        n.clone()
    }

    fn to_str(&self, a: &Int) -> String {
        a.to_string()
    }
}

fn strip(f: Vec<Int>) -> Vec<Int> {
    let i: usize = f.iter().take_while(|c| c.is_zero()).count();
    if i == 0 {
        return f;
    }
    f[i..].to_vec()
}

impl ConvertFrom<IntegerRing> for IntegerRing {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<Int> {
        Ok(a.clone())
    }
}

impl ConvertFrom<RationalField> for IntegerRing {
    fn convert(&self, a: &Rat, _k0: &RationalField) -> PolyResult<Int> {
        if a.is_int() {
            return Ok(a.numerator().clone());
        }
        Err(value_error(format!("{} is not an integer", a)))
    }
}

impl ConvertFrom<FiniteField> for IntegerRing {
    fn convert(&self, a: &u64, _k0: &FiniteField) -> PolyResult<Int> {
        Ok(Int::from(*a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;

    fn z(n: i64) -> Int {
        Int::from(n)
    }

    #[test]
    fn doctests() {
        assert_eq!(ZZ.gcd(&z(12), &z(-18)), z(6));
        assert_eq!(ZZ.gcdex(&z(12), &z(18)).unwrap(), (z(-1), z(1), z(6)));
        assert_eq!(ZZ.quo(&z(-7), &z(2)).unwrap(), z(-3));
        assert_eq!(ZZ.rem(&z(-7), &z(2)).unwrap(), z(-1));
        assert_eq!(ZZ.exquo(&z(12), &z(4)).unwrap(), z(3));
        assert!(ZZ.exquo(&z(12), &z(5)).is_err());
        assert!(ZZ.exquo(&z(12), &z(0)).is_err());
    }
}
