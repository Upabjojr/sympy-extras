//! Port of the decision of `sympy_extras/core/relational.py` for the
//! orderings (`_compare`, `_order`): whether `a < b`, `a <= b`, ... holds,
//! is false, or stays undecided. The relational nodes themselves (`Lt`,
//! `Eq`, ...) are not in the Rust core yet.

use crate::error::PolyResult;

use super::add::sub;
use super::assumptions::Predicate;
use super::basic::{Basic, Head};
use super::expr::{is_comparable, is_true};

/// `-1`, `0` or `1` for two real numbers, `2` when undecidable (`_order`).
fn order(a: &Basic, b: &Basic) -> i32 {
    let undecidable = |x: &Basic| matches!(x.head(), Head::NaN | Head::ComplexInfinity);
    if undecidable(a) || undecidable(b) {
        return 2;
    }
    match (a.head(), b.head()) {
        (Head::Infinity, Head::Infinity) | (Head::NegativeInfinity, Head::NegativeInfinity) => return 0,
        (Head::Infinity, _) | (_, Head::NegativeInfinity) => return 1,
        (Head::NegativeInfinity, _) | (_, Head::Infinity) => return -1,
        _ => {}
    }
    match (a.as_rational(), b.as_rational()) {
        (Some(va), Some(vb)) => match va.cmp(vb) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        },
        _ => 2,
    }
}

/// The value of the relation whose true orders are `accepted` (`(-1,)` is
/// `<`, `(-1, 0)` is `<=`, ...): `Some(true)`, `Some(false)`, or `None`
/// when it stays unevaluated (`_compare`). Non-real sides (a `TypeError`
/// in Python) are undecided here.
pub fn compare(lhs: &Basic, rhs: &Basic, accepted: &[i32]) -> PolyResult<Option<bool>> {
    for side in [lhs, rhs] {
        if super::expr::is_false(side, Predicate::real) && !is_true(side, Predicate::extended_real) {
            return Ok(None);
        }
    }
    let infinite = |x: &Basic| matches!(x.head(), Head::Infinity | Head::NegativeInfinity);
    let mut result: i32 = 2;
    if lhs.is_number_class() && rhs.is_number_class() {
        result = order(lhs, rhs);
    } else if is_true(lhs, Predicate::extended_real)
        && is_true(rhs, Predicate::extended_real)
        && (infinite(rhs) || infinite(lhs))
    {
        let other: &Basic = if infinite(rhs) { lhs } else { rhs };
        let finite: bool = is_true(other, Predicate::finite);
        if matches!(rhs.head(), Head::Infinity) || matches!(lhs.head(), Head::NegativeInfinity) {
            result = if finite || accepted == [-1, 0] || accepted == [1] {
                -1
            } else {
                2
            };
        } else {
            result = if finite || accepted == [0, 1] || accepted == [-1] {
                1
            } else {
                2
            };
        }
    } else if (is_comparable(lhs) && is_comparable(rhs))
        || (is_true(lhs, Predicate::extended_real) && is_true(rhs, Predicate::extended_real))
    {
        let difference: Basic = sub(lhs, rhs)?;
        if difference.is_number_class() {
            result = order(&difference, &Basic::zero());
        } else {
            let yes = |p: Predicate| is_true(&difference, p);
            if yes(Predicate::extended_positive) {
                result = 1;
            } else if yes(Predicate::extended_negative) {
                result = -1;
            } else if yes(Predicate::zero) {
                result = 0;
            } else if yes(Predicate::extended_nonnegative) && accepted == [0, 1] {
                result = 1;
            } else if yes(Predicate::extended_nonpositive) && accepted == [-1, 0] {
                result = -1;
            } else if yes(Predicate::extended_nonnegative) && accepted == [-1] {
                result = 1;
            } else if yes(Predicate::extended_nonpositive) && accepted == [1] {
                result = -1;
            }
        }
    }
    if result == 2 {
        return Ok(None);
    }
    Ok(Some(accepted.contains(&result)))
}

/// Whether `a < b` evaluates to true (`Lt(a, b).doit() is true`).
pub fn lt_true(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(compare(a, b, &[-1])? == Some(true))
}
