//! The number types of the port: the integers ([`Int`]) and the rationals
//! ([`Rat`]) of Python's `int` and `fractions.Fraction`, with the helpers
//! the algorithms use, so that the big number library is chosen in one
//! place.
//!
//! The library is `dashu` (pure Rust, MIT/Apache): its gcd (Lehmer's
//! algorithm) is faster than CPython's on large numbers, which matters for
//! the normalisation of every rational; `num-bigint`'s binary gcd was an
//! order of magnitude slower than CPython's at 50 000 bits.

use dashu_base::{Abs, BitTest, Gcd, Sign, UnsignedAbs};

/// An arbitrary precision integer (Python's `int`).
pub type Int = dashu_int::IBig;

/// An arbitrary precision rational number, always in lowest terms with a
/// positive denominator (Python's `fractions.Fraction`).
pub type Rat = dashu_ratio::RBig;

pub fn int(n: i64) -> Int {
    Int::from(n)
}

pub fn is_negative(a: &Int) -> bool {
    a.sign() == Sign::Negative && !a.is_zero()
}

pub fn abs(a: &Int) -> Int {
    a.clone().abs()
}

/// The nonnegative gcd (`gcd(0, 0) = 0`, as Python's `math.gcd`).
pub fn gcd(a: &Int, b: &Int) -> Int {
    if a.is_zero() {
        return abs(b);
    }
    if b.is_zero() {
        return abs(a);
    }
    Int::from(Gcd::gcd(a, b))
}

/// The nonnegative lcm (`0` when an argument is zero).
pub fn lcm(a: &Int, b: &Int) -> Int {
    if a.is_zero() || b.is_zero() {
        return Int::ZERO;
    }
    abs(&(a / gcd(a, b) * b))
}

/// The quotient and remainder of the division truncated towards zero.
pub fn div_rem_trunc(a: &Int, b: &Int) -> (Int, Int) {
    let q: Int = a / b;
    let r: Int = a - &q * b;
    (q, r)
}

/// The quotient rounded towards minus infinity (Python's `//`).
pub fn div_floor(a: &Int, b: &Int) -> Int {
    let (q, r) = div_rem_trunc(a, b);
    if !r.is_zero() && (is_negative(&r) != is_negative(b)) {
        q - Int::ONE
    } else {
        q
    }
}

/// The remainder of [`div_floor`] (Python's `%`: the sign of `b`).
pub fn mod_floor(a: &Int, b: &Int) -> Int {
    a - div_floor(a, b) * b
}

/// The floor quotient and remainder (Python's `divmod`).
pub fn div_mod_floor(a: &Int, b: &Int) -> (Int, Int) {
    let q: Int = div_floor(a, b);
    let r: Int = a - &q * b;
    (q, r)
}

/// The inverse of `a` modulo `m > 1` in `0..m` (Python's `pow(a, -1, m)`);
/// `None` when `a` and `m` are not coprime.
pub fn mod_inverse(a: &Int, m: &Int) -> Option<Int> {
    let (mut r0, mut r1): (Int, Int) = (mod_floor(a, m), m.clone());
    let (mut s0, mut s1): (Int, Int) = (Int::ONE, Int::ZERO);
    while !r1.is_zero() {
        let q: Int = div_floor(&r0, &r1);
        let r2: Int = &r0 - &q * &r1;
        r0 = std::mem::replace(&mut r1, r2);
        let s2: Int = &s0 - &q * &s1;
        s0 = std::mem::replace(&mut s1, s2);
    }
    if !r0.is_one() {
        return None;
    }
    Some(mod_floor(&s0, m))
}

/// `floor(sqrt(n))` (Python's `math.isqrt`).
pub fn isqrt(n: &Int) -> Int {
    assert!(!is_negative(n), "isqrt of a negative number");
    Int::from(dashu_base::SquareRoot::sqrt(&n.clone().unsigned_abs()))
}

/// Whether bit `i` of the nonnegative `n` is set.
pub fn bit(n: &Int, i: usize) -> bool {
    n.clone().unsigned_abs().bit(i)
}

/// The number of bits of `|a|` (Python's `int.bit_length`).
pub fn bits(a: &Int) -> u64 {
    a.clone().unsigned_abs().bit_len() as u64
}

/// The rational `p/q` in lowest terms (`q` nonzero).
pub fn rat(p: i64, q: i64) -> Rat {
    Rat::from_parts_signed(Int::from(p), Int::from(q))
}

pub fn rat_from_int(a: &Int) -> Rat {
    Rat::from(a.clone())
}

/// `p/q` in lowest terms for integers (`q` nonzero).
pub fn rat_from_parts(p: Int, q: Int) -> Rat {
    Rat::from_parts_signed(p, q)
}

pub fn numer(a: &Rat) -> Int {
    a.numerator().clone()
}

/// The (positive) denominator.
pub fn denom(a: &Rat) -> Int {
    Int::from(a.denominator().clone())
}

pub fn rat_is_negative(a: &Rat) -> bool {
    is_negative(a.numerator())
}

pub fn rat_is_integer(a: &Rat) -> bool {
    a.is_int()
}

/// `1/a` for nonzero `a`.
pub fn recip(a: &Rat) -> Rat {
    Rat::ONE / a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_semantics() {
        assert_eq!(div_floor(&int(-7), &int(2)), int(-4));
        assert_eq!(div_floor(&int(7), &int(-2)), int(-4));
        assert_eq!(div_floor(&int(-7), &int(-2)), int(3));
        assert_eq!(mod_floor(&int(-7), &int(2)), int(1));
        assert_eq!(div_rem_trunc(&int(-7), &int(2)), (int(-3), int(-1)));
        assert_eq!(gcd(&int(0), &int(0)), int(0));
        assert_eq!(gcd(&int(-12), &int(18)), int(6));
        assert_eq!(lcm(&int(-4), &int(6)), int(12));
        assert_eq!(bits(&int(-8)), 4);
        assert_eq!(bits(&int(0)), 0);
        assert_eq!(rat(-6, 4), rat(3, -2));
        assert_eq!(denom(&rat(3, -2)), int(2));
        assert!(rat_is_negative(&rat(3, -2)));
        assert!(!is_negative(&int(0)));
        assert_eq!(rat(1, 2).to_string(), "1/2");
        assert_eq!(mod_inverse(&int(-3), &int(7)), Some(int(2)));
        assert_eq!(mod_inverse(&int(4), &int(8)), None);
        assert_eq!(isqrt(&int(99)), int(9));
        assert_eq!(isqrt(&int(100)), int(10));
        assert!(bit(&int(5), 2) && !bit(&int(5), 1));
    }
}
