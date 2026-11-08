//! Port of the exact numbers of `sympy_extras/core/numbers.py`: `Integer`,
//! `Rational`, the infinities, `nan`, the named constants and `I`, as atoms
//! of the tree.
//!
//! `number_add` and `number_mul` are the arithmetic of the numbers (the
//! infinities and `nan` included). `Float` (arbitrary precision, `mpmath` in
//! Python) and `AlgebraicNumber` come later.

use crate::error::{zero_division, PolyResult};
use crate::numbers::*;

use super::basic::{Basic, Head};

impl Basic {
    /// `Integer(n)`.
    pub fn integer(n: Int) -> Basic {
        Basic::atom(Head::Rational {
            value: Rat::from(n),
            integer_class: true,
        })
    }

    /// `Integer(n)` of a machine integer.
    pub fn int(n: i64) -> Basic {
        Basic::integer(Int::from(n))
    }

    /// `Rational(p, q)` in lowest terms (`ZeroDivision` for `q = 0`).
    pub fn rational(p: Int, q: Int) -> PolyResult<Basic> {
        if q.is_zero() {
            return Err(zero_division(&format!("Rational({}, 0)", p)));
        }
        Ok(Basic::atom(Head::Rational {
            value: rat_from_parts(p, q),
            integer_class: false,
        }))
    }

    /// The `Integer` or `Rational` with the value of a fraction
    /// (`number_from_fraction`).
    pub fn from_rational(value: Rat) -> Basic {
        let integer_class: bool = value.is_int();
        Basic::atom(Head::Rational { value, integer_class })
    }

    pub fn zero() -> Basic {
        Basic::int(0)
    }

    pub fn one() -> Basic {
        Basic::int(1)
    }

    pub fn negative_one() -> Basic {
        Basic::int(-1)
    }

    pub fn half() -> Basic {
        Basic::from_rational(rat(1, 2))
    }

    /// `oo`.
    pub fn oo() -> Basic {
        Basic::atom(Head::Infinity)
    }

    /// `-oo`.
    pub fn negative_oo() -> Basic {
        Basic::atom(Head::NegativeInfinity)
    }

    /// `zoo`.
    pub fn zoo() -> Basic {
        Basic::atom(Head::ComplexInfinity)
    }

    pub fn nan() -> Basic {
        Basic::atom(Head::NaN)
    }

    pub fn pi() -> Basic {
        Basic::atom(Head::Pi)
    }

    /// `E`.
    pub fn e() -> Basic {
        Basic::atom(Head::Exp1)
    }

    pub fn euler_gamma() -> Basic {
        Basic::atom(Head::EulerGamma)
    }

    pub fn catalan() -> Basic {
        Basic::atom(Head::Catalan)
    }

    pub fn golden_ratio() -> Basic {
        Basic::atom(Head::GoldenRatio)
    }

    pub fn tribonacci_constant() -> Basic {
        Basic::atom(Head::TribonacciConstant)
    }

    /// `I`.
    pub fn imaginary_unit() -> Basic {
        Basic::atom(Head::ImaginaryUnit)
    }

    /// The sign of a real number: `-1`, `0` or `1`; `0` for the numbers
    /// which are not real (`_sign`).
    pub fn number_sign(&self) -> i32 {
        match self.head() {
            Head::Rational { value, .. } => {
                if value.is_zero() {
                    0
                } else if rat_is_negative(value) {
                    -1
                } else {
                    1
                }
            }
            Head::Infinity => 1,
            Head::NegativeInfinity => -1,
            _ => 0,
        }
    }
}

impl Basic {
    /// Whether the node is a number of Python's class `Number` (a rational,
    /// an infinity, `nan`; `Float` later).
    pub fn is_number_node(&self) -> bool {
        self.is_number_class()
    }

    /// Whether the number is a rational or a float (`is_finite_number`).
    pub fn is_finite_number(&self) -> bool {
        self.is_rational_class()
    }
}

fn expect_number(a: &Basic) {
    assert!(a.is_number_node(), "{} is not a Number", a);
}

/// The sum of two numbers.
pub fn number_add(a: &Basic, b: &Basic) -> Basic {
    expect_number(a);
    expect_number(b);
    let (ha, hb) = (a.head(), b.head());
    if matches!(ha, Head::NaN) || matches!(hb, Head::NaN) {
        return Basic::nan();
    }
    if matches!(ha, Head::ComplexInfinity) || matches!(hb, Head::ComplexInfinity) {
        if a.is_finite_number() && matches!(hb, Head::ComplexInfinity) {
            return Basic::zoo();
        }
        if b.is_finite_number() && matches!(ha, Head::ComplexInfinity) {
            return Basic::zoo();
        }
        return Basic::nan();
    }
    let infinite = |h: &Head| matches!(h, Head::Infinity | Head::NegativeInfinity);
    if infinite(ha) || infinite(hb) {
        let sa: i32 = if a.is_finite_number() { 0 } else { a.number_sign() };
        let sb: i32 = if b.is_finite_number() { 0 } else { b.number_sign() };
        if sa != 0 && sb != 0 && sa != sb {
            return Basic::nan();
        }
        let s: i32 = if sa != 0 { sa } else { sb };
        return if s > 0 { Basic::oo() } else { Basic::negative_oo() };
    }
    let (va, vb) = (
        a.as_rational().expect("a rational"),
        b.as_rational().expect("a rational"),
    );
    Basic::from_rational(va + vb)
}

/// The product of two numbers.
pub fn number_mul(a: &Basic, b: &Basic) -> Basic {
    expect_number(a);
    expect_number(b);
    let (ha, hb) = (a.head(), b.head());
    if matches!(ha, Head::NaN) || matches!(hb, Head::NaN) {
        return Basic::nan();
    }
    if !(a.is_finite_number() && b.is_finite_number()) {
        if (a.is_finite_number() && a.number_sign() == 0) || (b.is_finite_number() && b.number_sign() == 0) {
            return Basic::nan();
        }
        if matches!(ha, Head::ComplexInfinity) || matches!(hb, Head::ComplexInfinity) {
            return Basic::zoo();
        }
        let sign: i32 = a.number_sign() * b.number_sign();
        return if sign > 0 { Basic::oo() } else { Basic::negative_oo() };
    }
    let (va, vb) = (
        a.as_rational().expect("a rational"),
        b.as_rational().expect("a rational"),
    );
    Basic::from_rational(va * vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rationals() {
        let half = Basic::rational(int(2), int(4)).unwrap();
        assert_eq!(half.to_string(), "1/2");
        assert_eq!(half, Basic::half());
        // a Rational equals the Integer of the same value, the classes differ
        let three = Basic::rational(int(3), int(1)).unwrap();
        assert_eq!(three, Basic::int(3));
        assert_eq!(
            (three.class_name(), Basic::int(3).class_name()),
            ("Rational", "Integer")
        );
        assert!(Basic::rational(int(1), int(0)).is_err());
        assert_eq!(Basic::rational(int(3), int(-6)).unwrap().to_string(), "-1/2");
        assert_eq!(Basic::negative_oo().number_sign(), -1);
    }

    #[test]
    fn arithmetic() {
        let r = |p: i64, q: i64| Basic::from_rational(rat(p, q));
        assert_eq!(number_add(&r(1, 3), &r(2, 3)).to_string(), "1");
        assert_eq!(number_add(&Basic::oo(), &Basic::one()).to_string(), "oo");
        assert_eq!(number_add(&Basic::oo(), &Basic::negative_oo()).to_string(), "nan");
        assert_eq!(number_add(&Basic::zoo(), &Basic::one()).to_string(), "zoo");
        assert_eq!(number_add(&Basic::zoo(), &Basic::oo()).to_string(), "nan");
        assert_eq!(number_mul(&r(2, 3), &r(3, 4)).to_string(), "1/2");
        assert_eq!(number_mul(&Basic::oo(), &Basic::int(-2)).to_string(), "-oo");
        assert_eq!(number_mul(&Basic::oo(), &Basic::zero()).to_string(), "nan");
        assert_eq!(number_mul(&Basic::zoo(), &Basic::negative_oo()).to_string(), "zoo");
        assert_eq!(
            number_mul(&Basic::negative_oo(), &Basic::negative_oo()).to_string(),
            "oo"
        );
        // the class of the result follows the value
        assert_eq!(number_add(&r(1, 2), &r(1, 2)).class_name(), "Integer");
    }

    #[test]
    fn compare_order_of_numbers() {
        // Zero, One, Half, Infinity, NaN, NegativeOne, NegativeInfinity, Integer, Rational, ...
        let ordered: Vec<Basic> = vec![
            Basic::zero(),
            Basic::one(),
            Basic::half(),
            Basic::oo(),
            Basic::nan(),
            Basic::negative_one(),
            Basic::negative_oo(),
            Basic::int(-5),
            Basic::int(2),
            Basic::from_rational(rat(-7, 3)),
            Basic::from_rational(rat(1, 3)),
            Basic::e(),
            Basic::pi(),
            Basic::imaginary_unit(),
        ];
        for pair in ordered.windows(2) {
            assert_eq!(pair[0].compare(&pair[1]), -1, "{} < {}", pair[0], pair[1]);
        }
    }
}
