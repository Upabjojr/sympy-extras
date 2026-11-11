//! Port of the decompositions of `sympy_extras/core/expr.py` and of their
//! overrides in the classes of the core (`as_coeff_Mul`, `as_base_exp`,
//! `terms_of`, `factors_of`), and the properties of the facts (`is_real`,
//! `is_positive`, ...: the structural facts of [`super::assumptions`], as
//! Python's `Expr._ask`).

use crate::numbers::*;

use super::assumptions::{ask_structural, Predicate};
use super::basic::{Basic, Head};
use super::mul::{mul, raw_mul};

/// A fact of the node: `Some(true)`, `Some(false)` or unknown.
pub fn is(e: &Basic, p: Predicate) -> Option<bool> {
    ask_structural(e, p)
}

/// Whether the fact is known to hold (`e.is_xxx is True`).
pub fn is_true(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

/// Whether the fact is known to fail (`e.is_xxx is False`).
pub fn is_false(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(false)
}

/// Whether `e` is the rational number `value` (`Integer` or `Rational`).
pub fn is_rational_value(e: &Basic, value: i64) -> bool {
    e.as_rational().is_some_and(|r| *r == rat(value, 1))
}

/// Whether `e` is the integer zero (`isinstance(e, Integer) and e.p == 0`,
/// a `Rational` zero included: its value decides).
pub fn is_zero_number(e: &Basic) -> bool {
    e.as_rational().is_some_and(|r| r.is_zero())
}

/// Whether the node is of Python's class `Integer` (a `Rational` with
/// denominator one is not).
pub fn is_integer_class(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}

/// The terms of a sum, `[e]` for anything else.
pub fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// The factors of a product, `[e]` for anything else.
pub fn factors_of(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// The numerical coefficient and the rest of a product (`as_coeff_Mul`):
/// a number is `(self, 1)` (`-oo` is `(-1, oo)`), a product whose first
/// argument is a number other than `zoo` splits it off, anything else is
/// `(1, self)`.
pub fn as_coeff_mul(e: &Basic) -> (Basic, Basic) {
    match e.head() {
        Head::NegativeInfinity => (Basic::negative_one(), Basic::oo()),
        _ if e.is_number_node() => (e.clone(), Basic::one()),
        Head::Mul => {
            let first: &Basic = &e.args()[0];
            if first.is_number_node() && !matches!(first.head(), Head::ComplexInfinity) {
                (first.clone(), raw_mul(e.args()[1..].to_vec()))
            } else {
                (Basic::one(), e.clone())
            }
        }
        _ => (Basic::one(), e.clone()),
    }
}

/// The base and the exponent (`as_base_exp`): `(1/q)**e` is `q**(-e)`, a
/// product of powers with one integer exponent is `(product, exponent)`,
/// `I` is `(-1, 1/2)`, anything else is `(self, 1)`.
pub fn as_base_exp(e: &Basic) -> Result<(Basic, Basic), crate::error::PolyError> {
    match e.head() {
        Head::Pow => {
            let (base, exp) = (&e.args()[0], &e.args()[1]);
            if let Some(b) = base.as_rational() {
                if b.numerator().is_one() && denom(b) > Int::ONE {
                    return Ok((Basic::integer(denom(b)), mul(&[Basic::negative_one(), exp.clone()])?));
                }
            }
            Ok((base.clone(), exp.clone()))
        }
        Head::Mul => {
            // x**2*y**2 is (x*y)**2 when every factor has the same integer exponent
            let mut pairs: Vec<(Basic, Basic)> = Vec::new();
            for f in e.args() {
                pairs.push(as_base_exp(f)?);
            }
            let exponent: &Basic = &pairs[0].1;
            if pairs.iter().all(|(_, x)| x == exponent) {
                if let Some(r) = exponent.as_rational() {
                    if r.is_int() && *r != Rat::ONE {
                        let bases: Vec<Basic> = pairs.iter().map(|(b, _)| b.clone()).collect();
                        return Ok((mul(&bases)?, exponent.clone()));
                    }
                }
            }
            Ok((e.clone(), Basic::one()))
        }
        Head::ImaginaryUnit => Ok((Basic::negative_one(), Basic::half())),
        _ => Ok((e.clone(), Basic::one())),
    }
}
