//! Port of `sympy_extras/core/relational.py`: the relations of the core
//! ([`Head::Relational`]: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`), their
//! evaluating functions ([`eq`], [`ne`], [`lt`], [`le`], [`gt`], [`ge`]:
//! `true`, `false` or the relation), the decision of the orderings
//! (`_compare`, `_order`) and the transformations (`reversed`, `negated`,
//! `reversedsign`, `canonical`).

use crate::error::PolyResult;

use super::add::sub;
use super::assumptions::Predicate;
use super::basic::{Basic, Head, RelOp};
use super::expr::{is_comparable, is_true};
use super::logic::boolean;

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

/// The unevaluated relation `op(lhs, rhs)`.
pub fn rel(op: RelOp, lhs: &Basic, rhs: &Basic) -> Basic {
    Basic::new(Head::Relational(op), vec![lhs.clone(), rhs.clone()])
}

/// `a == b` decided: `Some(true)`, `Some(false)` or `None` (`_decide_equal`).
fn decide_equal(a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
    if a == b {
        return Ok(Some(true));
    }
    if a.is_number_class() && b.is_number_class() {
        if matches!(a.head(), Head::NaN) || matches!(b.head(), Head::NaN) {
            return Ok(Some(false));
        }
        if let (Some(va), Some(vb)) = (a.as_rational(), b.as_rational()) {
            return Ok(Some(va == vb));
        }
        return Ok(Some(false));
    }
    let difference: Basic = sub(a, b)?;
    if difference.is_number_class() {
        if let Some(v) = difference.as_rational() {
            return Ok(Some(v.is_zero()));
        }
        if !matches!(difference.head(), Head::NaN) {
            return Ok(Some(false));
        }
    }
    Ok(super::expr::is(&difference, Predicate::zero))
}

/// The evaluated equation `a == b` (`eq`).
pub fn eq(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    if !super::parameters::evaluate() {
        return Ok(rel(RelOp::Eq, a, b));
    }
    Ok(match decide_equal(a, b)? {
        Some(v) => boolean(v),
        None => rel(RelOp::Eq, a, b),
    })
}

/// The evaluated inequation `a != b` (`ne`).
pub fn ne(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    if !super::parameters::evaluate() {
        return Ok(rel(RelOp::Ne, a, b));
    }
    Ok(match decide_equal(a, b)? {
        Some(v) => boolean(!v),
        None => rel(RelOp::Ne, a, b),
    })
}

fn ordering(op: RelOp, a: &Basic, b: &Basic, accepted: &[i32]) -> PolyResult<Basic> {
    if !super::parameters::evaluate() {
        return Ok(rel(op, a, b));
    }
    for side in [a, b] {
        if super::expr::is_false(side, Predicate::real) && !is_true(side, Predicate::extended_real) {
            return Err(crate::error::PolyError::Type(format!(
                "invalid comparison of non-real {}",
                side
            )));
        }
    }
    Ok(match compare(a, b, accepted)? {
        Some(v) => boolean(v),
        None => rel(op, a, b),
    })
}

/// The evaluated `a < b` (`lt`).
pub fn lt(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    ordering(RelOp::Lt, a, b, &[-1])
}

/// The evaluated `a <= b` (`le`).
pub fn le(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    ordering(RelOp::Le, a, b, &[-1, 0])
}

/// The evaluated `a > b` (`gt`).
pub fn gt(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    ordering(RelOp::Gt, a, b, &[1])
}

/// The evaluated `a >= b` (`ge`).
pub fn ge(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    ordering(RelOp::Ge, a, b, &[0, 1])
}

/// The evaluating function of a relation (`_FUNCTIONS`).
pub fn evaluate_relation(op: RelOp, a: &Basic, b: &Basic) -> PolyResult<Basic> {
    match op {
        RelOp::Eq => eq(a, b),
        RelOp::Ne => ne(a, b),
        RelOp::Lt => lt(a, b),
        RelOp::Le => le(a, b),
        RelOp::Gt => gt(a, b),
        RelOp::Ge => ge(a, b),
    }
}

fn op_of(r: &Basic) -> RelOp {
    match r.head() {
        Head::Relational(op) => *op,
        _ => panic!("not a relation"),
    }
}

/// The relation with the sides exchanged (`reversed`).
pub fn reversed(r: &Basic) -> Basic {
    rel(op_of(r).reversed(), &r.args()[1], &r.args()[0])
}

/// The relation of the opposite meaning (`negated`).
pub fn negated(r: &Basic) -> Basic {
    rel(op_of(r).negated(), &r.args()[0], &r.args()[1])
}

/// Both sides negated (`reversedsign`: `-x > -1` for `x < 1`).
pub fn reversedsign(r: &Basic) -> PolyResult<Basic> {
    let op: RelOp = op_of(r);
    let (a, b) = (super::mul::neg(&r.args()[0])?, super::mul::neg(&r.args()[1])?);
    Ok(match op {
        RelOp::Eq | RelOp::Ne => rel(op, &a, &b),
        _ => rel(op.reversed(), &a, &b),
    })
}

/// SymPy's canonical form (`canonical`): a number on the right, the sides
/// in canonical order otherwise, no minus sign in front of the left side.
pub fn canonical(r: &Basic) -> PolyResult<Basic> {
    use super::assumptions::is_number;
    use super::printing::could_extract_minus_sign;
    use super::sorting::ordered;
    let mut r: Basic = r.clone();
    let (lhs, rhs) = (r.args()[0].clone(), r.args()[1].clone());
    if is_number(&rhs) {
        if rhs.is_number_class() && lhs.is_number_class() && order(&lhs, &rhs) == 1 {
            r = reversed(&r);
        }
    } else if is_number(&lhs) || ordered(vec![lhs.clone(), rhs.clone()]) != vec![lhs, rhs] {
        // a number on the right, otherwise the sides in canonical order
        r = reversed(&r);
    }
    if could_extract_minus_sign(&r.args()[0]) {
        return reversedsign(&r);
    }
    if !is_number(&r.args()[1]) && could_extract_minus_sign(&r.args()[1]) {
        let first: Basic = ordered(vec![r.args()[0].clone(), super::mul::neg(&r.args()[1])?]).remove(0);
        if first != r.args()[0] {
            return reversedsign(&reversed(&r));
        }
    }
    Ok(r)
}
