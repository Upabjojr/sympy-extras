//! Port of `sympy_extras/core/accumbounds.py`: accumulation bounds
//! [`Head::AccumBounds`] (`AccumBounds(a, b)`, the values a quantity
//! approaches without settling) and their interval arithmetic: the
//! evaluating [`accum_bounds`] and the operations [`ab_add`], [`ab_sub`],
//! [`ab_neg`], [`ab_mul`], [`ab_truediv`], [`ab_rtruediv`], [`ab_pow`],
//! [`ab_rpow`], [`ab_abs`], which the evaluating functions of the core
//! call (a bound is a coefficient in sums and products).

use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::{numer, Int};

use super::add::add;
use super::assumptions::{FactList, Predicate};
use super::basic::{Basic, Head};
use super::expr::{as_numer_denom, is_comparable, is_true};
use super::mul::mul;
use super::power::pow;

fn is_oo(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity)
}

fn is_moo(e: &Basic) -> bool {
    matches!(e.head(), Head::NegativeInfinity)
}

fn is_zero(e: &Basic) -> bool {
    is_true(e, Predicate::zero)
}

/// Whether the node is accumulation bounds.
pub fn is_bounds(e: &Basic) -> bool {
    matches!(e.head(), Head::AccumBounds)
}

fn min_of(e: &Basic) -> &Basic {
    &e.args()[0]
}

fn max_of(e: &Basic) -> &Basic {
    &e.args()[1]
}

fn min2(args: &[Basic]) -> PolyResult<Basic> {
    crate::functions::elementary::miscellaneous::min_(args)
}

fn max2(args: &[Basic]) -> PolyResult<Basic> {
    crate::functions::elementary::miscellaneous::max_(args)
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

fn sub(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), neg(b)?])
}

fn div(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}

fn inverse(a: &Basic) -> PolyResult<Basic> {
    pow(a, &Basic::negative_one())
}

/// `a < b` (`None` when undecided).
fn lt(a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
    Ok(super::expr::is(&sub(a, b)?, Predicate::extended_negative))
}

fn lt_true(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(lt(a, b)? == Some(true))
}

fn yes(e: &Basic, p: Predicate) -> bool {
    is_true(e, p)
}

/// The unevaluated `AccumBounds(min, max)`.
pub fn bounds_node(min: &Basic, max: &Basic) -> Basic {
    Basic::new(Head::AccumBounds, vec![min.clone(), max.clone()])
}

fn whole() -> Basic {
    bounds_node(&Basic::negative_oo(), &Basic::oo())
}

/// The evaluated bounds: a single value when `min == max`; the bounds
/// must be real and ordered (`accum_bounds`).
pub fn accum_bounds(lo: &Basic, hi: &Basic) -> PolyResult<Basic> {
    if !yes(lo, Predicate::extended_real) || !yes(hi, Predicate::extended_real) {
        return Err(value_error("Only real AccumulationBounds are supported".to_string()));
    }
    if hi == lo {
        return Ok(hi.clone());
    }
    let bad: bool = if super::assumptions::is_number(hi) && super::assumptions::is_number(lo) {
        is_comparable(hi) && is_comparable(lo) && lt(hi, lo)? == Some(true)
    } else {
        yes(&sub(hi, lo)?, Predicate::extended_negative)
    };
    if bad {
        return Err(value_error(
            "Lower limit should be smaller than upper limit".to_string(),
        ));
    }
    Ok(bounds_node(lo, hi))
}

/// `AccumulationBounds.__add__`.
pub fn ab_add(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    if is_bounds(other) {
        return accum_bounds(
            &add(&[lo.clone(), min_of(other).clone()])?,
            &add(&[hi.clone(), max_of(other).clone()])?,
        );
    }
    if (is_oo(other) && is_moo(lo)) || (is_moo(other) && is_oo(hi)) {
        return accum_bounds(&Basic::negative_oo(), &Basic::oo());
    }
    if yes(other, Predicate::extended_real) {
        if is_moo(lo) && is_oo(hi) {
            return accum_bounds(&Basic::negative_oo(), &Basic::oo());
        }
        if is_moo(lo) {
            return accum_bounds(&Basic::negative_oo(), &add(&[hi.clone(), other.clone()])?);
        }
        if is_oo(hi) {
            return accum_bounds(&add(&[lo.clone(), other.clone()])?, &Basic::oo());
        }
        return accum_bounds(&add(&[lo.clone(), other.clone()])?, &add(&[hi.clone(), other.clone()])?);
    }
    Ok(super::add::raw_add(vec![s.clone(), other.clone()]))
}

/// `AccumulationBounds.__neg__`.
pub fn ab_neg(s: &Basic) -> PolyResult<Basic> {
    accum_bounds(&neg(max_of(s))?, &neg(min_of(s))?)
}

/// `AccumulationBounds.__sub__`.
pub fn ab_sub(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    if is_bounds(other) {
        return accum_bounds(&sub(lo, max_of(other))?, &sub(hi, min_of(other))?);
    }
    if (is_moo(other) && is_moo(lo)) || (is_oo(other) && is_oo(hi)) {
        return accum_bounds(&Basic::negative_oo(), &Basic::oo());
    }
    if yes(other, Predicate::extended_real) {
        if is_moo(lo) && is_oo(hi) {
            return accum_bounds(&Basic::negative_oo(), &Basic::oo());
        }
        if is_moo(lo) {
            return accum_bounds(&Basic::negative_oo(), &sub(hi, other)?);
        }
        if is_oo(hi) {
            return accum_bounds(&sub(lo, other)?, &Basic::oo());
        }
        return accum_bounds(&sub(lo, other)?, &sub(hi, other)?);
    }
    Ok(super::add::raw_add(vec![s.clone(), neg(other)?]))
}

/// `AccumulationBounds.__mul__`.
pub fn ab_mul(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    if is_moo(lo) && is_oo(hi) {
        return Ok(s.clone());
    }
    if is_bounds(other) {
        if is_moo(min_of(other)) && is_oo(max_of(other)) {
            return Ok(other.clone());
        }
        let mut v: Vec<Basic> = Vec::new();
        for a in [lo, hi] {
            let vi: Basic = ab_mul(other, a)?;
            let items: Vec<Basic> = if is_bounds(&vi) { vi.args().to_vec() } else { vec![vi] };
            for t in items {
                if !v.contains(&t) {
                    v.push(t);
                }
            }
        }
        return accum_bounds(&min2(&v)?, &max2(&v)?);
    }
    let zero: Basic = Basic::zero();
    if is_oo(other) {
        if is_zero(lo) {
            return accum_bounds(&zero, &Basic::oo());
        }
        if is_zero(hi) {
            return accum_bounds(&Basic::negative_oo(), &zero);
        }
    }
    if is_moo(other) {
        if is_zero(lo) {
            return accum_bounds(&Basic::negative_oo(), &zero);
        }
        if is_zero(hi) {
            return accum_bounds(&zero, &Basic::oo());
        }
    }
    if yes(other, Predicate::extended_real) {
        if is_zero(other) {
            if is_oo(hi) {
                return accum_bounds(&zero, &Basic::oo());
            }
            if is_moo(lo) {
                return accum_bounds(&Basic::negative_oo(), &zero);
            }
            return Ok(zero);
        }
        if yes(other, Predicate::extended_positive) {
            return accum_bounds(&mul(&[lo.clone(), other.clone()])?, &mul(&[hi.clone(), other.clone()])?);
        }
        if yes(other, Predicate::extended_negative) {
            return accum_bounds(&mul(&[hi.clone(), other.clone()])?, &mul(&[lo.clone(), other.clone()])?);
        }
    }
    Ok(super::mul::raw_mul(vec![s.clone(), other.clone()]))
}

/// `AccumulationBounds.__truediv__`.
pub fn ab_truediv(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    let (oo, moo, zero) = (Basic::oo(), Basic::negative_oo(), Basic::zero());
    if is_bounds(other) {
        let (olo, ohi) = (min_of(other), max_of(other));
        if yes(olo, Predicate::positive) || yes(ohi, Predicate::negative) {
            let reciprocal: Basic = accum_bounds(&inverse(ohi)?, &inverse(olo)?)?;
            if !is_bounds(&reciprocal) {
                return Err(PolyError::Type(format!("bounds are expected, got {}", reciprocal)));
            }
            return ab_mul(s, &reciprocal);
        }
        let nonpositive = |e: &Basic| yes(e, Predicate::extended_nonpositive);
        let nonnegative = |e: &Basic| yes(e, Predicate::extended_nonnegative);
        if nonpositive(lo) && nonnegative(hi) && nonpositive(olo) && nonnegative(ohi) {
            if is_zero(lo) && is_zero(olo) {
                return accum_bounds(&zero, &oo);
            }
            if is_zero(hi) && is_zero(olo) {
                return accum_bounds(&moo, &zero);
            }
            return accum_bounds(&moo, &oo);
        }
        if yes(hi, Predicate::extended_negative) {
            if yes(olo, Predicate::extended_negative) {
                if is_zero(ohi) {
                    return accum_bounds(&div(hi, olo)?, &oo);
                }
                if yes(ohi, Predicate::extended_positive) {
                    return accum_bounds(&moo, &oo);
                }
            }
            if is_zero(olo) && yes(ohi, Predicate::extended_positive) {
                return accum_bounds(&moo, &div(hi, ohi)?);
            }
        }
        if yes(lo, Predicate::extended_positive) {
            if yes(olo, Predicate::extended_negative) {
                if is_zero(ohi) {
                    return accum_bounds(&moo, &div(lo, olo)?);
                }
                if yes(ohi, Predicate::extended_positive) {
                    return accum_bounds(&moo, &oo);
                }
            }
            if is_zero(olo) && yes(ohi, Predicate::extended_positive) {
                return accum_bounds(&div(lo, ohi)?, &oo);
            }
        }
    } else if yes(other, Predicate::extended_real) {
        if is_oo(other) || is_moo(other) {
            if is_moo(lo) && is_oo(hi) {
                return accum_bounds(&moo, &oo);
            }
            if is_oo(hi) {
                return accum_bounds(
                    &min2(&[zero.clone(), other.clone()])?,
                    &max2(&[zero.clone(), other.clone()])?,
                );
            }
            if is_moo(lo) {
                let minus: Basic = neg(other)?;
                return accum_bounds(&min2(&[zero.clone(), minus.clone()])?, &max2(&[zero, minus])?);
            }
        }
        if yes(other, Predicate::extended_positive) {
            return accum_bounds(&div(lo, other)?, &div(hi, other)?);
        }
        if yes(other, Predicate::extended_negative) {
            return accum_bounds(&div(hi, other)?, &div(lo, other)?);
        }
    }
    let reciprocal: Basic = if is_bounds(other) {
        ab_rtruediv(other, &Basic::one())?
    } else {
        inverse(other)?
    };
    if matches!(reciprocal.head(), Head::ComplexInfinity) {
        return Ok(super::mul::raw_mul(vec![s.clone(), reciprocal]));
    }
    ab_mul(s, &reciprocal)
}

/// `AccumulationBounds.__rtruediv__`: `other / s`.
pub fn ab_rtruediv(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    let (oo, moo) = (Basic::oo(), Basic::negative_oo());
    if yes(other, Predicate::extended_real) {
        if is_zero(other) {
            return Ok(Basic::zero());
        }
        if yes(lo, Predicate::extended_nonpositive) && yes(hi, Predicate::extended_nonnegative) {
            if is_zero(lo) {
                if yes(other, Predicate::extended_positive) {
                    return accum_bounds(&mul(&[other.clone(), inverse(hi)?])?, &oo);
                }
                if yes(other, Predicate::extended_negative) {
                    return accum_bounds(&moo, &mul(&[other.clone(), inverse(hi)?])?);
                }
            }
            if is_zero(hi) {
                if yes(other, Predicate::extended_positive) {
                    return accum_bounds(&moo, &mul(&[other.clone(), inverse(lo)?])?);
                }
                if yes(other, Predicate::extended_negative) {
                    return accum_bounds(&mul(&[other.clone(), inverse(lo)?])?, &oo);
                }
            }
            return accum_bounds(&moo, &oo);
        }
        let (a, b) = (div(other, lo)?, div(other, hi)?);
        return accum_bounds(&min2(&[a.clone(), b.clone()])?, &max2(&[a, b])?);
    }
    Ok(super::mul::raw_mul(vec![
        other.clone(),
        ab_pow(s, &Basic::negative_one())?,
    ]))
}

/// `AccumulationBounds.__pow__`.
pub fn ab_pow(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    let (oo, moo, zero, one) = (Basic::oo(), Basic::negative_oo(), Basic::zero(), Basic::one());
    let minus_one: Basic = Basic::negative_one();
    if is_oo(other) {
        if yes(lo, Predicate::extended_nonnegative) {
            if lt_true(hi, &one)? {
                return Ok(zero);
            }
            if lt_true(&one, lo)? {
                return Ok(oo);
            }
            return accum_bounds(&zero, &oo);
        }
        if yes(hi, Predicate::extended_negative) {
            if lt_true(&minus_one, lo)? {
                return Ok(zero);
            }
            if lt_true(hi, &minus_one)? {
                return Ok(Basic::zoo());
            }
            return Ok(Basic::nan());
        }
        if lt_true(&minus_one, lo)? {
            if lt_true(hi, &one)? {
                return Ok(zero);
            }
            return accum_bounds(&zero, &oo);
        }
        return accum_bounds(&moo, &oo);
    }
    if is_moo(other) {
        return pow(&ab_rtruediv(s, &one)?, &oo);
    }
    if yes(&sub(hi, lo)?, Predicate::nonnegative)
        && yes(lo, Predicate::nonnegative)
        && yes(other, Predicate::nonnegative)
    {
        return accum_bounds(&pow(lo, other)?, &pow(hi, other)?);
    }
    if is_zero(other) {
        return Ok(one);
    }
    if let Some(v) = other.as_rational().filter(|_| {
        matches!(
            other.head(),
            Head::Rational {
                integer_class: true,
                ..
            }
        )
    }) {
        let p: Int = numer(v);
        let (a, b) = (pow(lo, other)?, pow(hi, other)?);
        if yes(lo, Predicate::extended_positive) {
            return accum_bounds(&min2(&[a.clone(), b.clone()])?, &max2(&[a, b])?);
        }
        if yes(hi, Predicate::extended_negative) {
            return accum_bounds(&min2(&[b.clone(), a.clone()])?, &max2(&[b, a])?);
        }
        let negative: bool = p < Int::ZERO;
        if &p % Int::from(2) == Int::ZERO {
            if negative {
                if is_zero(lo) {
                    return accum_bounds(&b, &oo);
                }
                if is_zero(hi) {
                    return accum_bounds(&a, &oo);
                }
                return pow(&ab_rtruediv(s, &one)?, &neg(other)?);
            }
            return accum_bounds(&zero, &max2(&[a, b])?);
        }
        if negative {
            if is_zero(lo) {
                return accum_bounds(&b, &oo);
            }
            if is_zero(hi) {
                return accum_bounds(&moo, &a);
            }
            return pow(&ab_rtruediv(s, &one)?, &neg(other)?);
        }
        return accum_bounds(&a, &b);
    }
    if (super::assumptions::is_number(other) || yes(other, Predicate::rational))
        && yes(lo, Predicate::extended_nonnegative)
    {
        let (num, den) = as_numer_denom(other)?;
        if num == one {
            let root: Basic = inverse(&den)?;
            return accum_bounds(&pow(lo, &root)?, &pow(hi, &root)?);
        }
        if den != one {
            return pow(&pow(s, &num)?, &inverse(&den)?);
        }
    }
    if is_bounds(other) {
        let (olo, ohi) = (min_of(other), max_of(other));
        if yes(lo, Predicate::extended_positive)
            || (yes(lo, Predicate::extended_nonnegative) && yes(olo, Predicate::extended_nonnegative))
        {
            let p: Vec<Basic> = vec![pow(s, olo)?, pow(s, ohi)?];
            if !p.iter().any(|i| i.is_pow()) {
                let a: Vec<Basic> = p
                    .iter()
                    .flat_map(|i| {
                        if is_bounds(i) {
                            i.args().to_vec()
                        } else {
                            vec![i.clone()]
                        }
                    })
                    .collect();
                let (mut least, mut most) = (a[0].clone(), a[0].clone());
                for v in &a[1..] {
                    let (lt_least, gt_most) = (lt(v, &least)?, lt(&most, v)?);
                    let (Some(lt_least), Some(gt_most)) = (lt_least, gt_most) else {
                        return Ok(Basic::new(Head::Pow, vec![s.clone(), other.clone()]));
                    };
                    if lt_least {
                        least = v.clone();
                    }
                    if gt_most {
                        most = v.clone();
                    }
                }
                return accum_bounds(&least, &most);
            }
        }
    }
    Ok(Basic::new(Head::Pow, vec![s.clone(), other.clone()]))
}

/// `AccumulationBounds.__rpow__`: `other ** s`.
pub fn ab_rpow(s: &Basic, other: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    if yes(other, Predicate::real)
        && yes(other, Predicate::extended_nonnegative)
        && yes(&sub(hi, lo)?, Predicate::extended_positive)
    {
        if *other == Basic::one() {
            return Ok(Basic::one());
        }
        if yes(other, Predicate::extended_positive) {
            let (mut a, mut b) = (pow(other, lo)?, pow(other, hi)?);
            if lt_true(&b, &a)? {
                std::mem::swap(&mut a, &mut b);
            }
            return accum_bounds(&a, &b);
        }
        if is_zero(other) {
            if is_zero(lo) {
                return accum_bounds(&Basic::zero(), &Basic::one());
            }
            if yes(lo, Predicate::extended_positive) {
                return Ok(Basic::zero());
            }
        }
    }
    Ok(Basic::new(Head::Pow, vec![other.clone(), s.clone()]))
}

/// `AccumulationBounds.__abs__`.
pub fn ab_abs(s: &Basic) -> PolyResult<Basic> {
    let (lo, hi) = (min_of(s), max_of(s));
    if yes(hi, Predicate::extended_negative) {
        return ab_neg(s);
    }
    if yes(lo, Predicate::extended_negative) {
        let a: Basic = crate::functions::elementary::complexes::abs_(lo)?;
        return accum_bounds(&Basic::zero(), &max2(&[a, hi.clone()])?);
    }
    Ok(s.clone())
}

/// Whether `o` is in the bounds (`__contains__`; a `TypeError` when
/// undecided).
pub fn contains(s: &Basic, o: &Basic) -> PolyResult<bool> {
    let (lo, hi) = (min_of(s), max_of(s));
    if is_oo(o) || is_moo(o) {
        return Ok(is_moo(lo) || is_oo(hi));
    }
    let low = super::expr::is(&sub(lo, o)?, Predicate::extended_nonpositive);
    let high = super::expr::is(&sub(hi, o)?, Predicate::extended_nonnegative);
    match (low, high) {
        (Some(a), Some(b)) => Ok(a && b),
        _ => Err(PolyError::Type("input failed to evaluate".to_string())),
    }
}

/// Whether one of the points is in the bounds (`meets_points`).
pub fn meets_points(s: &Basic, points: &[Basic]) -> PolyResult<bool> {
    for p in points {
        if contains(s, p)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The common part of two bounds, `None` when empty (`intersection`).
pub fn intersection(s: &Basic, other: &Basic) -> PolyResult<Option<Basic>> {
    let (lo, hi, olo, ohi) = (min_of(s), max_of(s), min_of(other), max_of(other));
    if lt_true(hi, olo)? || lt_true(ohi, lo)? {
        return Ok(None);
    }
    if yes(&sub(lo, olo)?, Predicate::extended_nonpositive) {
        if yes(&sub(hi, ohi)?, Predicate::extended_nonpositive) {
            return Ok(Some(accum_bounds(olo, hi)?));
        }
        if lt_true(ohi, hi)? {
            return Ok(Some(other.clone()));
        }
    }
    if yes(&sub(olo, lo)?, Predicate::extended_nonpositive) {
        if lt_true(ohi, hi)? {
            return Ok(Some(accum_bounds(lo, ohi)?));
        }
        if lt_true(hi, ohi)? {
            return Ok(Some(s.clone()));
        }
    }
    Ok(None)
}

/// The union of two overlapping bounds, `None` otherwise (`union`).
pub fn union(s: &Basic, other: &Basic) -> PolyResult<Option<Basic>> {
    let (lo, hi, olo, ohi) = (min_of(s), max_of(s), min_of(other), max_of(other));
    if yes(&sub(lo, olo)?, Predicate::extended_nonpositive) && yes(&sub(hi, olo)?, Predicate::extended_nonnegative) {
        return Ok(Some(accum_bounds(lo, &max2(&[hi.clone(), ohi.clone()])?)?));
    }
    if yes(&sub(olo, lo)?, Predicate::extended_nonpositive) && yes(&sub(ohi, lo)?, Predicate::extended_nonnegative) {
        return Ok(Some(accum_bounds(olo, &max2(&[hi.clone(), ohi.clone()])?)?));
    }
    Ok(None)
}

/// The facts (`_eval_facts`): extended real, real when both ends are.
pub fn bounds_facts(e: &Basic) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::extended_real, true);
    facts.set(Predicate::commutative, true);
    if yes(min_of(e), Predicate::real) && yes(max_of(e), Predicate::real) {
        facts.set(Predicate::real, true);
    }
    facts
}

/// A sum with bounds: the numbers and the bounds make the coefficient,
/// the other terms are summed apart (`Add.flatten`, `_with_bounds`).
pub fn sum_with_bounds(stack: &[Basic]) -> PolyResult<Basic> {
    let mut flat: Vec<Basic> = Vec::new();
    for a in stack {
        if a.is_add() {
            flat.extend(a.args().iter().cloned());
        } else {
            flat.push(a.clone());
        }
    }
    let mut coeff: Basic = Basic::zero();
    let mut others: Vec<Basic> = Vec::new();
    for o in flat {
        if matches!(o.head(), Head::NaN) {
            return Ok(Basic::nan());
        }
        if o.is_number_node() {
            if coeff.is_number_node() {
                coeff = super::numbers::number_add(&coeff, &o);
            } else if is_bounds(&coeff) {
                coeff = ab_add(&coeff, &o)?;
            }
        } else if is_bounds(&o) {
            coeff = ab_add(&o, &coeff)?;
        } else {
            others.push(o);
        }
    }
    let rest: Basic = add(&others)?;
    if !is_bounds(&coeff) {
        if coeff.is_add() && coeff.args().iter().any(is_bounds) {
            // a sum of bounds and zoo: Python's add sends the sum back here
            // without end
            return Err(PolyError::Recursion("maximum recursion depth exceeded".to_string()));
        }
        return add(&[coeff, rest]);
    }
    if rest == Basic::zero() && rest.as_rational().is_some() {
        return Ok(coeff);
    }
    let mut terms: Vec<Basic> = vec![coeff];
    if rest.is_add() {
        terms.extend(rest.args().iter().cloned());
    } else {
        terms.push(rest);
    }
    Ok(super::add::raw_add(terms))
}

/// A product with bounds: the numbers and the bounds make the
/// coefficient (`Mul.flatten`, `_with_bounds`).
pub fn product_with_bounds(factors: &[Basic]) -> PolyResult<Basic> {
    let mut coeff: Basic = Basic::one();
    let mut others: Vec<Basic> = Vec::new();
    for o in flattened(factors) {
        if o.is_number_node() {
            if coeff.is_number_node() {
                coeff = super::numbers::number_mul(&coeff, &o);
            } else if is_bounds(&coeff) {
                coeff = ab_mul(&coeff, &o)?;
            }
        } else if is_bounds(&o) {
            coeff = ab_mul(&o, &coeff)?;
        } else {
            others.push(o);
        }
    }
    let rest: Basic = mul(&others)?;
    if !is_bounds(&coeff) {
        if coeff.is_mul() && coeff.args().iter().any(is_bounds) {
            // bounds times zoo: Python's mul returns the product alone, and
            // with other factors sends it back here without end
            if rest == Basic::one() && rest.as_rational().is_some() {
                return Ok(coeff);
            }
            return Err(PolyError::Recursion("maximum recursion depth exceeded".to_string()));
        }
        return mul(&[coeff, rest]);
    }
    if rest == Basic::one() && rest.as_rational().is_some() {
        return Ok(coeff);
    }
    let mut items: Vec<Basic> = vec![coeff];
    if rest.is_mul() {
        items.extend(rest.args().iter().cloned());
    } else {
        items.push(rest);
    }
    Ok(super::mul::raw_mul(items))
}

fn flattened(args: &[Basic]) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    for a in args {
        if a.is_mul() {
            result.extend(flattened(a.args()));
        } else {
            result.push(a.clone());
        }
    }
    result
}

/// `AccumBounds(-oo, oo)`.
pub fn everything() -> Basic {
    whole()
}
