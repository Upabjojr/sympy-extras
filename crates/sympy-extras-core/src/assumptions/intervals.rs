//! Port of `sympy_extras/assumptions/intervals.py`: rigorous interval
//! arithmetic on the expressions of the core ([`super::ivarith`], mpmath's
//! `iv` at 53 bits), the enclosure of an expression on a box
//! ([`evaluate`]) and the sign of an enclosure.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use super::ivarith::{Iv, E, PREC};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;

/// A box: for every variable its closed range (`-oo`, `oo` allowed).
pub type Box = Vec<(Basic, (Basic, Basic))>;

fn endpoint(value: &Basic, p: usize) -> Option<Iv> {
    match value.head() {
        Head::Infinity => Some(Iv::point(E::PInf)),
        Head::NegativeInfinity => Some(Iv::point(E::NInf)),
        _ => evaluate_at(value, &[], p),
    }
}

fn hull(lo: &Basic, hi: &Basic, p: usize) -> Option<Iv> {
    let (a, b) = (endpoint(lo, p)?, endpoint(hi, p)?);
    Some(Iv::new(a.a, b.b))
}

thread_local! {
    /// The working precision of the interval arithmetic (mpmath's global
    /// `iv.prec`, raised temporarily by the root isolation).
    static IV_PREC: std::cell::Cell<usize> = const { std::cell::Cell::new(PREC) };
}

/// The working precision of the interval arithmetic.
pub fn iv_prec() -> usize {
    IV_PREC.with(|p| p.get())
}

/// The precision raised to at least `bits` while the guard lives
/// (`_precision` of the root isolation).
pub struct PrecisionGuard {
    previous: usize,
}

impl PrecisionGuard {
    pub fn raise_to(bits: usize) -> PrecisionGuard {
        let previous: usize = iv_prec();
        IV_PREC.with(|p| p.set(bits.max(previous)));
        PrecisionGuard { previous }
    }
}

impl Drop for PrecisionGuard {
    fn drop(&mut self) {
        IV_PREC.with(|p| p.set(self.previous));
    }
}

/// An enclosure of the values of `expr` on the box, or `None`
/// (`evaluate`), at the working precision.
pub fn evaluate(expr: &Basic, b: &[(Basic, (Basic, Basic))]) -> Option<Iv> {
    evaluate_at(expr, b, iv_prec())
}

/// [`evaluate`] at `p` bits (mpmath's `iv.prec`).
pub fn evaluate_at(expr: &Basic, b: &[(Basic, (Basic, Basic))], p: usize) -> Option<Iv> {
    match expr.head() {
        Head::Symbol(_) | Head::Dummy(..) => {
            let (_, (lo, hi)) = b.iter().find(|(s, _)| s == expr)?;
            hull(lo, hi, p)
        }
        Head::Pi => Some(Iv::pi(p)),
        Head::Exp1 => Some(Iv::e(p)),
        Head::Add => {
            let mut total: Iv = Iv::point(E::zero());
            for a in expr.args() {
                total = total.add(&evaluate_at(a, b, p)?, p);
            }
            Some(total)
        }
        Head::Mul => {
            let mut product: Iv = Iv::point(E::one());
            for a in expr.args() {
                product = product.mul(&evaluate_at(a, b, p)?, p);
            }
            Some(product)
        }
        Head::Pow => power(&expr.args()[0], &expr.args()[1], b, p),
        Head::Function(f) => {
            let f: Function = *f;
            if !matches!(
                f,
                Function::Exp
                    | Function::Log
                    | Function::Sin
                    | Function::Cos
                    | Function::Tan
                    | Function::Atan
                    | Function::Sinh
                    | Function::Cosh
                    | Function::Tanh
                    | Function::Abs
            ) {
                return None;
            }
            let v: Iv = evaluate_at(&expr.args()[0], b, p)?;
            function(f, &v, p)
        }
        _ => {
            let r: &Rat = expr.as_rational()?;
            if expr.is_integer_value() {
                return Some(Iv::from_int(&numer(r), p));
            }
            Some(Iv::from_int(&numer(r), p).div(&Iv::from_int(&Int::from(denom(r)), p), p))
        }
    }
}

fn power(base: &Basic, exponent: &Basic, b: &[(Basic, (Basic, Basic))], p: usize) -> Option<Iv> {
    let bv: Iv = evaluate_at(base, b, p)?;
    let zero: E = E::zero();
    if let Some(r) = exponent.as_rational() {
        if exponent.is_integer_value() {
            let n: i64 = i64::try_from(&numer(r)).ok()?;
            if n >= 0 {
                return Some(bv.pow_int(n, p));
            }
            if bv.a > zero || bv.b < zero {
                return Some(Iv::point(E::one()).div(&bv.pow_int(-n, p), p));
            }
            return None;
        }
        let ratio: Iv = Iv::from_int(&numer(r), p).div(&Iv::from_int(&Int::from(denom(r)), p), p);
        if bv.a > zero {
            return Some(ratio.mul(&bv.log(p)?, p).exp(p));
        }
        let positive: bool = !rat_is_negative(r);
        if bv.a == zero && positive && bv.b > zero {
            let upper: Iv = ratio.mul(&Iv::point(bv.b.clone()).log(p)?, p).exp(p);
            return Some(Iv::new(E::zero(), upper.b));
        }
        if bv.a == zero && bv.b == zero && positive {
            return Some(Iv::point(E::zero()));
        }
        return None;
    }
    let e: Iv = evaluate_at(exponent, b, p)?;
    if bv.a <= zero {
        return None;
    }
    Some(e.mul(&bv.log(p)?, p).exp(p))
}

fn function(f: Function, v: &Iv, p: usize) -> Option<Iv> {
    let two: Iv = Iv::from_int(&Int::from(2u8), p);
    let one: Iv = Iv::from_int(&Int::ONE, p);
    match f {
        Function::Exp => Some(v.exp(p)),
        Function::Log => {
            if v.a > E::zero() {
                v.log(p)
            } else {
                None
            }
        }
        Function::Sin => Some(v.cos_sin(p).1),
        Function::Cos => Some(v.cos_sin(p).0),
        Function::Tan => Some(v.tan(p)),
        Function::Atan => Some(v.atan(p)),
        Function::Sinh => Some(v.exp(p).sub(&v.neg().exp(p), p).div(&two, p)),
        Function::Cosh => Some(v.exp(p).add(&v.neg().exp(p), p).div(&two, p)),
        Function::Tanh => {
            let w: Iv = two.mul(v, p).exp(p);
            Some(w.sub(&one, p).div(&w.add(&one, p), p))
        }
        Function::Abs => Some(v.abs(p)),
        _ => None,
    }
}

/// The sign of every number of the enclosure, `None` when it contains
/// numbers of different signs (`interval_sign`).
pub fn interval_sign(v: &Iv) -> Option<i32> {
    let zero: E = E::zero();
    if v.a > zero {
        return Some(1);
    }
    if v.b < zero {
        return Some(-1);
    }
    if v.a == zero && v.b == zero {
        return Some(0);
    }
    None
}

/// An endpoint as an exact number (`_to_expr`).
pub fn to_expr(value: &E) -> Basic {
    match value {
        E::PInf => Basic::oo(),
        E::NInf => Basic::negative_oo(),
        E::F(r) => Basic::from_rational(r.clone()),
    }
}

// -- branch and bound

/// The possible signs of an expression.
pub type Signs = BTreeSet<i32>;

/// For every variable of a box, whether its range is open at the left and
/// at the right end.
pub type Openness = Vec<(Basic, (bool, bool))>;

/// A corner: a value (or an infinity) for every variable.
pub type Corner = Vec<(Basic, Basic)>;

/// A way of finding the signs of an expression (a partial derivative).
pub type SignFinder<'a> = &'a dyn Fn(&Basic) -> PolyResult<Option<Signs>>;

fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

fn plain_symbols(e: &Basic) -> Vec<Basic> {
    e.free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_) | Head::Dummy(..)))
        .collect()
}

fn in_box(e: &Basic, b: &[(Basic, (Basic, Basic))]) -> bool {
    plain_symbols(e).iter().all(|s| b.iter().any(|(v, _)| v == s))
}

/// `a > b` for numbers (`TypeError` when undecided).
fn greater(a: &Basic, b: &Basic) -> PolyResult<bool> {
    let d: Basic = crate::core::add::sub(a, b)?;
    is(&d, Predicate::extended_positive)
        .ok_or_else(|| PolyError::Type(format!("cannot determine truth value of {} > {}", a, b)))
}

fn sub(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    crate::core::add::sub(a, b)
}

/// Bisect the box along its widest finite direction, or cut an infinite
/// direction at a finite point (`_split`).
fn split(b: &Box) -> PolyResult<Option<Vec<Box>>> {
    let mut widest: Option<usize> = None;
    let mut width: Basic = Basic::negative_one();
    for (k, (_, (lo, hi))) in b.iter().enumerate() {
        if yes(lo, Predicate::finite) && yes(hi, Predicate::finite) {
            let w: Basic = sub(hi, lo)?;
            if greater(&w, &width)? {
                width = w;
                widest = Some(k);
            }
        }
    }
    if let Some(k) = widest {
        if greater(&width, &Basic::zero())? {
            let (lo, hi) = b[k].1.clone();
            let mid: Basic = crate::core::mul::div(&crate::core::add::add(&[lo.clone(), hi.clone()])?, &Basic::int(2))?;
            let (mut left, mut right) = (b.clone(), b.clone());
            left[k].1 = (lo, mid.clone());
            right[k].1 = (mid, hi);
            return Ok(Some(vec![left, right]));
        }
    }
    for (k, (_, (lo, hi))) in b.iter().enumerate() {
        let (lf, hf) = (yes(lo, Predicate::finite), yes(hi, Predicate::finite));
        if !(lf && hf) {
            let cut: Basic = if !lf && !hf {
                Basic::zero()
            } else if !lf {
                sub(
                    &sub(hi, &Basic::one())?,
                    &crate::functions::elementary::complexes::abs_(hi)?,
                )?
            } else {
                crate::core::add::add(&[
                    lo.clone(),
                    Basic::one(),
                    crate::functions::elementary::complexes::abs_(lo)?,
                ])?
            };
            let (mut left, mut right) = (b.clone(), b.clone());
            left[k].1 = (lo.clone(), cut.clone());
            right[k].1 = (cut, hi.clone());
            return Ok(Some(vec![left, right]));
        }
    }
    Ok(None)
}

/// The signs of `expr` on the box by branch and bound, `None` when the
/// budget of boxes is spent (`signs_on_box`).
pub fn signs_on_box(expr: &Basic, b: &Box, max_boxes: usize, weak: bool) -> PolyResult<Option<Signs>> {
    if !in_box(expr, b) {
        return Ok(None);
    }
    let mut queue: Vec<Box> = vec![b.clone()];
    let mut signs: Signs = Signs::new();
    let mut examined: usize = 0;
    while let Some(current) = queue.pop() {
        examined += 1;
        if examined > max_boxes {
            return Ok(None);
        }
        let Some(value) = evaluate(expr, &current) else {
            return Ok(None);
        };
        if let Some(s) = interval_sign(&value) {
            signs.insert(s);
            continue;
        }
        if weak && value.a >= E::zero() {
            signs.extend([0, 1]);
            continue;
        }
        if weak && value.b <= E::zero() {
            signs.extend([-1, 0]);
            continue;
        }
        let Some(halves) = split(&current)? else {
            return Ok(None);
        };
        queue.extend(halves);
    }
    Ok(Some(signs))
}

/// A set containing the values of `expr` on the box (`range_on_box`).
pub fn range_on_box(expr: &Basic, b: &Box) -> PolyResult<Option<Basic>> {
    let Some(value) = evaluate(expr, b) else {
        return Ok(None);
    };
    let (lo, hi) = (to_expr(&value.a), to_expr(&value.b));
    if yes(&lo, Predicate::finite) && yes(&hi, Predicate::finite) && greater(&lo, &hi)? {
        return Ok(None);
    }
    Ok(Some(crate::sets::sets::interval(&lo, &hi, false, false)?))
}

type Corners = (Corner, Corner, bool, bool);

fn corners(
    expr: &Basic,
    b: &Box,
    openness: &Openness,
    partial_sign: Option<SignFinder>,
    max_boxes: usize,
) -> PolyResult<Option<Corners>> {
    let mut variables: Vec<Basic> = b.iter().map(|(v, _)| v.clone()).collect();
    variables.sort_by_key(|v| match v.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => v.to_string(),
    });
    if variables.is_empty() || !in_box(expr, b) {
        return Ok(None);
    }
    let (mut low, mut high): (Corner, Corner) = (Vec::new(), Vec::new());
    let (mut low_un, mut high_un) = (false, false);
    for v in &variables {
        let (lo, hi) = b.iter().find(|(w, _)| w == v).expect("a variable").1.clone();
        let (left_open, right_open) = openness
            .iter()
            .find(|(w, _)| w == v)
            .map(|(_, o)| *o)
            .unwrap_or((false, false));
        let derivative: Basic = crate::core::function::derivative(expr, v)?;
        let slope: Option<Signs> = match partial_sign {
            Some(f) => f(&derivative)?,
            None => signs_on_box(&derivative, b, max_boxes, true)?,
        };
        let Some(slope) = slope else { return Ok(None) };
        if slope.contains(&1) && slope.contains(&-1) {
            return Ok(None);
        }
        let strict: bool = !slope.contains(&0);
        if slope.contains(&1) {
            low.push((v.clone(), lo));
            high.push((v.clone(), hi));
            low_un = low_un || (strict && left_open);
            high_un = high_un || (strict && right_open);
        } else if slope.contains(&-1) {
            low.push((v.clone(), hi));
            high.push((v.clone(), lo));
            low_un = low_un || (strict && right_open);
            high_un = high_un || (strict && left_open);
        } else {
            low.push((v.clone(), lo.clone()));
            high.push((v.clone(), lo));
        }
    }
    Ok(Some((low, high, low_un, high_un)))
}

/// The exact range of `expr` on the box when it is monotone in every
/// variable (`monotone_range`).
pub fn monotone_range(
    expr: &Basic,
    b: &Box,
    openness: &Openness,
    partial_sign: Option<SignFinder>,
    max_boxes: usize,
) -> PolyResult<Option<Basic>> {
    let Some((low_c, high_c, low_un, high_un)) = corners(expr, b, openness, partial_sign, max_boxes)? else {
        return Ok(None);
    };
    let (Some(low), Some(high)) = (
        corner_value(expr, &low_c, Some(b))?,
        corner_value(expr, &high_c, Some(b))?,
    ) else {
        return Ok(None);
    };
    if is(&low, Predicate::real) == Some(false) || is(&high, Predicate::real) == Some(false) {
        return Ok(None);
    }
    let lo_open: bool = low_un || !yes(&low, Predicate::finite);
    let hi_open: bool = high_un || !yes(&high, Predicate::finite);
    Ok(Some(crate::sets::sets::interval(&low, &high, lo_open, hi_open)?))
}

/// The signs of `expr` on the box when it is monotone in every variable
/// (`corner_signs`).
pub fn corner_signs(
    expr: &Basic,
    b: &Box,
    openness: &Openness,
    partial_sign: Option<SignFinder>,
    max_boxes: usize,
) -> PolyResult<Option<Signs>> {
    let Some((low_c, high_c, low_un, high_un)) = corners(expr, b, openness, partial_sign, max_boxes)? else {
        return Ok(None);
    };
    let Some(low_value) = corner_value(expr, &low_c, Some(b))? else {
        return Ok(None);
    };
    let Some(low) = super::analysis::certified_sign(&low_value)? else {
        return Ok(None);
    };
    if low == 1 || (low == 0 && low_un) {
        return Ok(Some([1].into_iter().collect()));
    }
    let Some(high_value) = corner_value(expr, &high_c, Some(b))? else {
        return Ok(None);
    };
    let Some(high) = super::analysis::certified_sign(&high_value)? else {
        return Ok(None);
    };
    if high == -1 || (high == 0 && high_un) {
        return Ok(Some([-1].into_iter().collect()));
    }
    let mut signs: Signs = Signs::new();
    if low < 0 {
        signs.insert(-1);
    }
    if high > 0 {
        signs.insert(1);
    }
    if low <= 0 && 0 <= high {
        signs.insert(0);
    }
    Ok(Some(signs))
}

fn dummy_like(w: &Basic, assumptions: &[(Predicate, bool)]) -> Basic {
    let name: String = match w.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => w.to_string(),
    };
    crate::core::symbol::Dummy::new(&name, assumptions).0
}

/// The value of `expr` at a corner: the finite coordinates substituted,
/// the infinite ones taken as limits one at a time (`corner_value`).
pub fn corner_value(expr: &Basic, corner: &Corner, b: Option<&Box>) -> PolyResult<Option<Basic>> {
    let finite: std::collections::HashMap<Basic, Basic> = corner
        .iter()
        .filter(|(_, c)| yes(c, Predicate::finite))
        .map(|(v, c)| (v.clone(), c.clone()))
        .collect();
    let mut value: Basic = crate::core::function::xreplace_evaluated(expr, &finite)?;
    let mut infinite: Vec<Basic> = corner
        .iter()
        .map(|(v, _)| v.clone())
        .filter(|v| !finite.contains_key(v))
        .collect();
    infinite.sort_by_key(|v| v.to_string());
    for (i, v) in infinite.iter().enumerate() {
        let c: Basic = corner.iter().find(|(w, _)| w == v).expect("a coordinate").1.clone();
        let direction: &str = if matches!(c.head(), Head::Infinity) { "-" } else { "+" };
        let mut signed: Vec<(Basic, Basic)> = Vec::new();
        for w in &infinite[i + 1..] {
            if let Some(bx) = b {
                if let Some((_, (lo, hi))) = bx.iter().find(|(s, _)| s == w) {
                    let d: Basic = if yes(lo, Predicate::nonnegative) {
                        dummy_like(w, &[(Predicate::real, true), (Predicate::positive, true)])
                    } else if yes(hi, Predicate::nonpositive) {
                        dummy_like(w, &[(Predicate::real, true), (Predicate::negative, true)])
                    } else {
                        dummy_like(w, &[(Predicate::real, true)])
                    };
                    signed.push((w.clone(), d));
                }
            }
        }
        let forth: std::collections::HashMap<Basic, Basic> = signed.iter().cloned().collect();
        let current: Basic = crate::core::function::xreplace_evaluated(&value, &forth)?;
        let Some(found) = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            crate::core::limits::limit(&current, v, &c, direction)
        })?
        else {
            return Ok(None);
        };
        let back: std::collections::HashMap<Basic, Basic> =
            signed.iter().map(|(w, d)| (d.clone(), w.clone())).collect();
        value = crate::core::function::xreplace_evaluated(&found, &back)?;
    }
    Ok(if plain_symbols(&value).is_empty() {
        Some(value)
    } else {
        None
    })
}

/// The signs of a continuous `f` on an interval by bisection with interval
/// arithmetic (`isolate_signs`).
pub fn isolate_signs(f: &Basic, x: &Basic, piece: &Basic, max_boxes: usize) -> PolyResult<Option<Signs>> {
    let derivative: Basic = crate::core::function::derivative(f, x)?;
    let (a, b) = (piece.args()[0].clone(), piece.args()[1].clone());
    let left_open: bool = matches!(piece.args()[2].head(), Head::BooleanTrue);
    let right_open: bool = matches!(piece.args()[3].head(), Head::BooleanTrue);
    let mut queue: Vec<(Basic, Basic)> = vec![(a.clone(), b.clone())];
    let mut signs: Signs = Signs::new();
    let mut examined: usize = 0;
    while let Some((lo, hi)) = queue.pop() {
        examined += 1;
        if examined > max_boxes {
            return Ok(None);
        }
        let bx: Box = vec![(x.clone(), (lo.clone(), hi.clone()))];
        let Some(value) = evaluate(f, &bx) else { return Ok(None) };
        if let Some(s) = interval_sign(&value) {
            signs.insert(s);
            continue;
        }
        if yes(&lo, Predicate::finite) && yes(&hi, Predicate::finite) {
            let slope: Option<Iv> = evaluate(&derivative, &bx);
            if slope
                .as_ref()
                .is_some_and(|sl| !matches!(interval_sign(sl), None | Some(0)))
            {
                let left = super::analysis::certified_sign(&crate::core::subs::subs(f, x, &lo)?)?;
                let right = super::analysis::certified_sign(&crate::core::subs::subs(f, x, &hi)?)?;
                let (Some(left), Some(right)) = (left, right) else {
                    return Ok(None);
                };
                if left == right {
                    signs.insert(left);
                    continue;
                }
                if left == 0 {
                    if !(lo == a && left_open) {
                        signs.insert(0);
                    }
                    signs.insert(right);
                } else if right == 0 {
                    if !(hi == b && right_open) {
                        signs.insert(0);
                    }
                    signs.insert(left);
                } else {
                    signs.extend([left, 0, right]);
                }
                continue;
            }
        }
        let Some(halves) = split(&bx)? else { return Ok(None) };
        queue.extend(halves.into_iter().map(|h| h[0].1.clone()));
    }
    Ok(Some(signs))
}
