//! Port of `sympy_extras/simplify/radsimp.py`, so far [`fraction`].

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function};
use crate::core::expr::{as_base_exp, as_numer_denom, factors_of, is_false, is_integer_class, is_rational_value, is_true};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::PolyResult;
use crate::numbers::*;

/// `(numerator, denominator)` of a product, without simplification
/// (`fraction` with `exact=False`).
pub fn fraction(e: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut numerator: Vec<Basic> = Vec::new();
    let mut denominator: Vec<Basic> = Vec::new();
    for term in factors_of(e) {
        if !is_false(&term, Predicate::commutative) && (term.is_pow() || term.is_function(Function::Exp)) {
            let (b, ex) = as_base_exp(&term)?;
            if is_true(&ex, Predicate::negative) {
                if is_integer_class(&ex) && is_rational_value(&ex, -1) {
                    denominator.push(b);
                } else {
                    denominator.push(pow(&b, &mul(&[Basic::negative_one(), ex])?)?);
                }
            } else if is_true(&ex, Predicate::positive) {
                numerator.push(term);
            } else if ex.is_mul() {
                let (n, d) = as_numer_denom(&term)?;
                if !(is_integer_class(&n) && is_rational_value(&n, 1)) {
                    numerator.push(n);
                }
                denominator.push(d);
            } else {
                numerator.push(term);
            }
        } else if let Some(v) = term.as_rational().filter(|v| denom(v) != Int::ONE) {
            if numer(v) != Int::ONE {
                numerator.push(Basic::integer(numer(v)));
            }
            denominator.push(Basic::integer(denom(v)));
        } else {
            numerator.push(term);
        }
    }
    Ok((mul(&numerator)?, mul(&denominator)?))
}
