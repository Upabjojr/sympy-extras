//! Port of `sympy_extras/assumptions/analysis.py` (the first part): the
//! certified sign of a constant ([`certified_sign`]: the exact knowledge
//! of the core, a hidden zero of an algebraic number by its minimal
//! polynomial, then rigorous interval arithmetic and a value with a margin
//! over its precision when the numerical checks are enabled) and the
//! comparison of two constants.

#![allow(clippy::mutable_key_type)]

use super::intervals::{evaluate_at, interval_sign};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;
use dashu_int::UBig;

fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

fn has_symbols(e: &Basic) -> bool {
    e.free_symbols()
        .iter()
        .any(|s| matches!(s.head(), Head::Symbol(_) | Head::Dummy(..)))
}

/// The sign of a constant by interval arithmetic at `digits` digits:
/// whether the arithmetic covers it, and the sign when the enclosure
/// excludes zero (`_interval_sign_at`).
fn interval_sign_at(value: &Basic, digits: usize) -> (bool, Option<i32>) {
    let prec: usize = crate::numeric::dps_to_prec(digits);
    match evaluate_at(value, &[], prec) {
        None => (false, None),
        Some(enclosure) => {
            let s: Option<i32> = interval_sign(&enclosure);
            (true, if s == Some(0) { None } else { s })
        }
    }
}

/// Whether `e` is built from rationals and `I` by sums, products and
/// rational powers.
fn is_algebraic_constant(e: &Basic) -> bool {
    if e.is_rational_class() || matches!(e.head(), Head::ImaginaryUnit) {
        return true;
    }
    if e.is_pow() {
        return e.args()[1].is_rational_class() && is_algebraic_constant(&e.args()[0]);
    }
    if e.is_add() || e.is_mul() {
        return e.args().iter().all(is_algebraic_constant);
    }
    false
}

/// Whether the algebraic number `e` is zero: its minimal polynomial is
/// `x`.
fn vanishes(e: &Basic) -> PolyResult<bool> {
    let coefficients: Option<Vec<Int>> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(e)
    })?;
    Ok(coefficients == Some(vec![Int::ONE, Int::ZERO]))
}

/// The sign of a constant expression (`certified_sign`); `None` when
/// undecided.
pub fn certified_sign(value: &Basic) -> PolyResult<Option<i32>> {
    if has_symbols(value) {
        return Ok(None);
    }
    match value.head() {
        Head::Infinity => return Ok(Some(1)),
        Head::NegativeInfinity => return Ok(Some(-1)),
        _ => {}
    }
    if yes(value, Predicate::zero) {
        return Ok(Some(0));
    }
    if yes(value, Predicate::positive) {
        return Ok(Some(1));
    }
    if yes(value, Predicate::negative) {
        return Ok(Some(-1));
    }
    if is_algebraic_constant(value) && vanishes(value)? {
        return Ok(Some(0));
    }
    let settings = crate::settings::settings();
    if !settings.numerical_checks || is(value, Predicate::real) == Some(false) {
        return Ok(None);
    }
    let p: usize = settings.precision;
    for digits in [p, 2 * p, 4 * p] {
        let (covered, s) = interval_sign_at(value, digits);
        if covered {
            if s.is_some() {
                return Ok(s);
            }
            continue;
        }
        // a value computed with a margin over its precision
        let Some(approximation) = crate::core::evalf::numeric_value(value, crate::numeric::dps_to_prec(digits)) else {
            return Ok(None);
        };
        let (re, im) = crate::core::evalf::parts(&approximation);
        let re_r: Rat = super::ivarith::real_to_rat(&re);
        if !super::ivarith::real_to_rat(&im).is_zero() {
            return Ok(None);
        }
        let threshold: Rat = Rat::from_parts(Int::ONE, UBig::from(10u8).pow(digits / 2));
        let magnitude: Rat = if rat_is_negative(&re_r) {
            -re_r.clone()
        } else {
            re_r.clone()
        };
        if magnitude > threshold {
            return Ok(Some(if rat_is_negative(&re_r) { -1 } else { 1 }));
        }
    }
    Ok(None)
}

/// The sign of `a - b` for constants, infinities included
/// (`certified_compare`).
pub fn certified_compare(a: &Basic, b: &Basic) -> PolyResult<Option<i32>> {
    if matches!(a.head(), Head::Infinity) || matches!(b.head(), Head::NegativeInfinity) {
        return Ok(Some(if a == b { 0 } else { 1 }));
    }
    if matches!(a.head(), Head::NegativeInfinity) || matches!(b.head(), Head::Infinity) {
        return Ok(Some(if a == b { 0 } else { -1 }));
    }
    certified_sign(&crate::core::add::sub(a, b)?)
}

// -- the domains of the variables

use crate::core::basic::RelOp;
use crate::sets::sets::{empty_set, finite_set, intersection, interval, interval_node, interval_parts, reals, union};

fn plain_symbols(e: &Basic) -> Vec<Basic> {
    e.free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_) | Head::Dummy(..)))
        .collect()
}

fn is_interval(e: &Basic) -> bool {
    matches!(e.head(), Head::Interval | Head::Reals)
}

/// The interval described by a bound `x rel c` with a constant `c`
/// (`_interval_of_bound`).
fn interval_of_bound(atom: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let Head::Relational(op) = atom.head() else {
        return Ok(None);
    };
    if plain_symbols(atom) != vec![x.clone()] {
        return Ok(None);
    }
    let (lhs, rhs) = (&atom.args()[0], &atom.args()[1]);
    let (c, flipped) = if lhs == x && plain_symbols(rhs).is_empty() {
        (rhs.clone(), false)
    } else if rhs == x && plain_symbols(lhs).is_empty() {
        (lhs.clone(), true)
    } else {
        return Ok(None);
    };
    if !yes(&c, Predicate::real) {
        return Ok(None);
    }
    let mut kind: RelOp = *op;
    if flipped {
        kind = match kind {
            RelOp::Gt => RelOp::Lt,
            RelOp::Lt => RelOp::Gt,
            RelOp::Ge => RelOp::Le,
            RelOp::Le => RelOp::Ge,
            other => other,
        };
    }
    let (oo, moo) = (Basic::oo(), Basic::negative_oo());
    Ok(Some(match kind {
        RelOp::Gt => interval(&c, &oo, true, true)?,
        RelOp::Ge => interval(&c, &oo, false, false)?,
        RelOp::Lt => interval(&moo, &c, true, true)?,
        RelOp::Le => interval(&moo, &c, false, false)?,
        RelOp::Eq => finite_set(std::slice::from_ref(&c)),
        RelOp::Ne => union(&[interval(&moo, &c, true, true)?, interval(&c, &oo, true, true)?])?,
    }))
}

fn is_union_of_intervals(s: &Basic) -> bool {
    matches!(s.head(), Head::EmptySet)
        || is_interval(s)
        || (matches!(s.head(), Head::Union) && s.args().iter().all(is_interval))
}

/// The set of real values of `x` allowed by the facts which only concern
/// `x` (`domain_of`); `None` when some fact about `x` involves other
/// symbols or cannot be turned into intervals.
pub fn domain_of(x: &Basic, facts: &super::facts::Facts) -> PolyResult<Option<Basic>> {
    if !facts.real.contains(x) {
        return Ok(None);
    }
    let mut domain: Basic = reals();
    for atom in &facts.conjuncts {
        let symbols: Vec<Basic> = plain_symbols(atom);
        if !symbols.contains(x) {
            continue;
        }
        if symbols != vec![x.clone()] {
            return Ok(None);
        }
        let mut piece: Option<Basic> = None;
        if matches!(atom.head(), Head::Contains) && is_interval(&atom.args()[1]) {
            piece = Some(atom.args()[1].clone());
        }
        if piece.is_none() {
            piece = interval_of_bound(atom, x)?;
        }
        let mut on_roots: bool = false;
        if piece.is_none() && super::facts::to_polynomial(atom, &facts.real)?.is_some() {
            piece = cad_solution_set(atom, x)?;
            on_roots = true;
        }
        let Some(piece) = piece else { return Ok(None) };
        let mut found: Basic = intersection(&[domain.clone(), piece.clone()])?;
        if on_roots && !is_union_of_intervals(&found) {
            if let Some(certified) = certified_intersection(&domain, &piece)? {
                found = certified;
            }
        }
        domain = found;
    }
    if is_union_of_intervals(&domain) {
        return Ok(Some(domain));
    }
    Ok(None)
}

fn cad_solution_set(atom: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::polys::cad::qe::solution_set(atom, x, &[], None, true)
    })
}

fn intersect_pieces(a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    let (a0, a1, alo, aro) = interval_parts(a);
    let (b0, b1, blo, bro) = interval_parts(b);
    let (Some(starts), Some(ends)) = (certified_compare(&a0, &b0)?, certified_compare(&a1, &b1)?) else {
        return Ok(None);
    };
    let (start, left_open) = if starts > 0 || (starts == 0 && alo) {
        (a0, alo)
    } else {
        (b0, blo)
    };
    let (end, right_open) = if ends < 0 || (ends == 0 && aro) {
        (a1, aro)
    } else {
        (b1, bro)
    };
    let Some(order) = certified_compare(&start, &end)? else {
        return Ok(None);
    };
    if order > 0 {
        return Ok(Some(empty_set()));
    }
    if order == 0 {
        return Ok(Some(if left_open || right_open {
            empty_set()
        } else {
            finite_set(std::slice::from_ref(&start))
        }));
    }
    Ok(Some(interval_node(&start, &end, left_open, right_open)))
}

/// The intersection of two unions of intervals, their ends compared by
/// certified comparisons (`_certified_intersection`).
fn certified_intersection(a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    let (Some(first), Some(second)) = (pieces(a), pieces(b)) else {
        return Ok(None);
    };
    let mut parts: Vec<Basic> = Vec::new();
    for p in &first {
        for q in &second {
            let Some(part) = intersect_pieces(p, q)? else {
                return Ok(None);
            };
            if matches!(part.head(), Head::EmptySet) {
                continue;
            }
            if !is_interval(&part) {
                return Ok(None);
            }
            parts.push(part);
        }
    }
    if parts.is_empty() {
        return Ok(Some(empty_set()));
    }
    Ok(Some(union(&parts)?))
}

/// The intervals of a union of intervals (`_pieces`).
pub fn pieces(domain: &Basic) -> Option<Vec<Basic>> {
    if matches!(domain.head(), Head::Reals) {
        return Some(vec![interval_node(&Basic::negative_oo(), &Basic::oo(), true, true)]);
    }
    if is_interval(domain) {
        return Some(vec![domain.clone()]);
    }
    if matches!(domain.head(), Head::Union) {
        if domain.args().iter().all(is_interval) {
            return Some(domain.args().to_vec());
        }
        return None;
    }
    None
}

/// The value of a constant at `digits` digits as an exact rational
/// (`_rational(e.evalf(digits))`): the value rounded to the binary
/// precision of the digits, as the Python evaluation gives it.
pub(crate) fn evalf_rational(e: &Basic, digits: usize) -> Option<Rat> {
    let prec: usize = crate::numeric::dps_to_prec(digits);
    let v = crate::core::evalf::numeric_value(e, prec + 64)?;
    let (re, im) = crate::core::evalf::parts(&v);
    if !super::ivarith::real_to_rat(&im).is_zero() {
        return None;
    }
    Some(round_nearest(&super::ivarith::real_to_rat(&re), prec))
}

/// `r` rounded to `prec` bits, to the nearest (ties to even).
pub(crate) fn round_nearest(r: &Rat, prec: usize) -> Rat {
    let down: Rat = super::ivarith::round(r, prec, super::ivarith::Rnd::Floor);
    let up: Rat = super::ivarith::round(r, prec, super::ivarith::Rnd::Ceil);
    if down == up {
        return down;
    }
    let (dd, du) = (r - &down, &up - r);
    match dd.cmp(&du) {
        std::cmp::Ordering::Less => down,
        std::cmp::Ordering::Greater => up,
        std::cmp::Ordering::Equal => {
            // ties to an even mantissa: the one with more trailing zeros
            let half: Rat = (&down + &up) / Rat::from(2);
            if super::ivarith::round(&half, prec, super::ivarith::Rnd::Floor) == down {
                down
            } else {
                up
            }
        }
    }
}

/// A rational number strictly between the constants `p < q` (`_between`).
pub fn between(p: &Basic, q: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(p.head(), Head::NegativeInfinity) && matches!(q.head(), Head::Infinity) {
        return Ok(Some(Basic::zero()));
    }
    let precision: usize = crate::settings::settings().precision;
    for digits in [precision, 2 * precision] {
        let candidate: Option<Rat> = if matches!(p.head(), Head::NegativeInfinity) {
            evalf_rational(q, digits).map(|v| v - Rat::ONE)
        } else if matches!(q.head(), Head::Infinity) {
            evalf_rational(p, digits).map(|v| v + Rat::ONE)
        } else {
            match (evalf_rational(p, digits), evalf_rational(q, digits)) {
                (Some(a), Some(b)) => {
                    // (p.evalf + q.evalf)/2 as Python's Floats compute it
                    let prec: usize = crate::numeric::dps_to_prec(digits);
                    let sum: Rat = round_nearest(&(a + b), prec);
                    Some(sum / Rat::from(2))
                }
                _ => None,
            }
        };
        let Some(c) = candidate else {
            return Err(PolyError::Type("invalid input".to_string()));
        };
        let candidate: Basic = Basic::from_rational(c);
        if certified_compare(&candidate, p)? == Some(1) && certified_compare(q, &candidate)? == Some(1) {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// The constants sorted by certified comparisons (`_sorted_constants`).
pub fn sorted_constants(values: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let mut result: Vec<Basic> = Vec::new();
    for v in values {
        let mut position: Option<usize> = Some(result.len());
        for (i, w) in result.iter().enumerate() {
            let Some(c) = certified_compare(v, w)? else {
                return Ok(None);
            };
            if c == 0 {
                position = None;
                break;
            }
            if c < 0 {
                position = Some(i);
                break;
            }
        }
        if let Some(p) = position {
            result.insert(p, v.clone());
        }
    }
    Ok(Some(result))
}

/// Whether `f` is a rational function of `x` (`_is_rational_in`).
pub fn is_rational_in(f: &Basic, x: &Basic) -> bool {
    if !f.free_symbols().contains(x) || f == x {
        return true;
    }
    if f.is_pow() {
        return f.args()[1].is_integer_value() && f.args()[1].is_number_class() && is_rational_in(&f.args()[0], x);
    }
    if f.is_add() || f.is_mul() {
        return f.args().iter().all(|a| is_rational_in(a, x));
    }
    false
}

/// The zeros of a rational `f` in `domain` natively: whether they are
/// known, and the list (`_real_zeros_in`).
pub fn real_zeros_in(f: &Basic, x: &Basic, domain: &Basic) -> PolyResult<(bool, Vec<Basic>)> {
    let Some((solutions, _)) = crate::polys::rationalsolve::solveset_rational(f, x)? else {
        return Ok((false, vec![]));
    };
    let mut zeros: Vec<Basic> = Vec::new();
    for z in solutions {
        let member: Basic = crate::sets::sets::contains(domain, &z)?;
        match member.head() {
            Head::BooleanTrue => zeros.push(z),
            Head::BooleanFalse => {}
            _ => return Ok((false, vec![])),
        }
    }
    Ok((true, zeros))
}

// -- continuity and zeros

use super::intervals::Signs;
use crate::core::basic::Function;

fn signs_of(v: &[i32]) -> Signs {
    v.iter().copied().collect()
}

fn attempt_opt<T>(f: impl FnOnce() -> PolyResult<Option<T>>) -> PolyResult<Option<T>> {
    Ok(crate::timeout::attempt_within(crate::settings::settings().timeout, f)?.flatten())
}

fn signs_within(f: &Basic, x: &Basic, domain: &Basic, allowed: &[i32]) -> PolyResult<bool> {
    let allowed: Signs = signs_of(allowed);
    Ok(matches!(sign_on(f, x, domain, 1)?, Some(s) if s.is_subset(&allowed)))
}

fn either_side(f: &Basic, low: &Basic, high: &Basic, strict: bool, x: &Basic, domain: &Basic) -> PolyResult<bool> {
    let Some(ps) = pieces(domain) else { return Ok(false) };
    let below: &[i32] = if strict { &[-1] } else { &[-1, 0] };
    let above: &[i32] = if strict { &[1] } else { &[0, 1] };
    for piece in ps {
        let sub_low: Basic = crate::core::add::sub(f, low)?;
        let sub_high: Basic = crate::core::add::sub(f, high)?;
        if !(signs_within(&sub_low, x, &piece, below)? || signs_within(&sub_high, x, &piece, above)?) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `f` is continuous on `domain` (`_continuous_on`).
pub fn continuous_on(f: &Basic, x: &Basic, domain: &Basic) -> PolyResult<bool> {
    use Function as F;
    if !f.free_symbols().contains(x) || f == x {
        return Ok(true);
    }
    if f.is_add() || f.is_mul() {
        for a in f.args() {
            if !continuous_on(a, x, domain)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if f.is_pow() {
        return power_continuous_on(f, x, domain);
    }
    if f.args().len() != 1 {
        return Ok(false);
    }
    let argument: Basic = f.args()[0].clone();
    if !continuous_on(&argument, x, domain)? {
        return Ok(false);
    }
    let Head::Function(func) = f.head() else {
        return Ok(false);
    };
    let one: Basic = Basic::one();
    let sub = |a: &Basic, b: &Basic| crate::core::add::sub(a, b);
    let add = |a: &Basic, b: &Basic| crate::core::add::add(&[a.clone(), b.clone()]);
    Ok(match func {
        F::Exp | F::Sin | F::Cos | F::Atan | F::Sinh | F::Cosh | F::Tanh | F::Sech | F::Asinh | F::Abs => true,
        F::Coth | F::Csch | F::Acot => signs_within(&argument, x, domain, &[-1, 1])?,
        F::Tan | F::Sec => signs_within(
            &crate::functions::elementary::trigonometric::cos(&argument)?,
            x,
            domain,
            &[-1, 1],
        )?,
        F::Cot | F::Csc => signs_within(
            &crate::functions::elementary::trigonometric::sin(&argument)?,
            x,
            domain,
            &[-1, 1],
        )?,
        F::Log => signs_within(&argument, x, domain, &[1])?,
        F::Asin | F::Acos => {
            signs_within(&sub(&argument, &one)?, x, domain, &[-1, 0])?
                && signs_within(&add(&argument, &one)?, x, domain, &[0, 1])?
        }
        F::Acosh => signs_within(&sub(&argument, &one)?, x, domain, &[0, 1])?,
        F::Atanh => {
            signs_within(&sub(&argument, &one)?, x, domain, &[-1])?
                && signs_within(&add(&argument, &one)?, x, domain, &[1])?
        }
        F::Asech => {
            signs_within(&argument, x, domain, &[1])? && signs_within(&sub(&argument, &one)?, x, domain, &[-1, 0])?
        }
        F::Asec | F::Acsc => either_side(&argument, &Basic::negative_one(), &one, false, x, domain)?,
        F::Acoth => either_side(&argument, &Basic::negative_one(), &one, true, x, domain)?,
        _ => false,
    })
}

fn power_continuous_on(f: &Basic, x: &Basic, domain: &Basic) -> PolyResult<bool> {
    let (base, exponent) = (&f.args()[0], &f.args()[1]);
    if !exponent.free_symbols().contains(x) {
        if !continuous_on(base, x, domain)? {
            return Ok(false);
        }
        if let Some(r) = exponent.as_rational() {
            if exponent.is_integer_value() {
                if !rat_is_negative(r) {
                    return Ok(true);
                }
                if is_rational_in(base, x) {
                    let (found, poles) = real_zeros_in(base, x, domain)?;
                    if found {
                        return Ok(poles.is_empty());
                    }
                }
                return signs_within(base, x, domain, &[-1, 1]);
            }
            if denom(r) % 2u8 == 1u8 {
                return Ok(!rat_is_negative(r) || signs_within(base, x, domain, &[-1, 1])?);
            }
        }
        if yes(exponent, Predicate::positive) {
            return signs_within(base, x, domain, &[0, 1]);
        }
        return signs_within(base, x, domain, &[1]);
    }
    if !continuous_on(exponent, x, domain)? {
        return Ok(false);
    }
    if !base.free_symbols().contains(x) {
        if yes(exponent, Predicate::integer) {
            return Ok(matches!(certified_sign(base)?, Some(-1) | Some(1)));
        }
        return Ok(certified_sign(base)? == Some(1));
    }
    Ok(continuous_on(base, x, domain)? && signs_within(base, x, domain, &[1])?)
}

/// The solver of the relations polynomial in a kernel variable with
/// constant coefficients (`_native_solver`).
fn native_solver(formula: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    native_solver_pub(formula, v)
}

/// The solver of `_zero_set` (`_native_solver`), for the inequalities.
pub fn native_solver_pub(formula: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(formula.head(), Head::Relational(_)) {
        if let Some(linear) = crate::solvers::transcendental::linear_solution(formula, v)? {
            return Ok(Some(linear));
        }
        return quadratic_solution(formula, v);
    }
    if matches!(formula.head(), Head::And | Head::Or) {
        let mut parts: Vec<Basic> = Vec::new();
        for a in formula.args() {
            let Some(part) = native_solver(a, v)? else {
                return Ok(None);
            };
            parts.push(part);
        }
        return Ok(Some(if matches!(formula.head(), Head::And) {
            intersection(&parts)?
        } else {
            union(&parts)?
        }));
    }
    Ok(None)
}

fn quadratic_solution(relation: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    if !matches!(relation.head(), Head::Relational(RelOp::Eq)) {
        return Ok(None);
    }
    let e: Basic =
        crate::core::function::expand_default(&crate::core::add::sub(&relation.args()[0], &relation.args()[1])?)?;
    let Some(cs) = crate::solvers::transcendental::coefficients(&e, v)? else {
        return Ok(None);
    };
    if cs.iter().any(|(d, _)| *d > 2) || !cs.iter().any(|(d, _)| *d == 2) {
        return Ok(None);
    }
    let at = |k: i64| {
        cs.iter()
            .find(|(d, _)| *d == k)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Basic::zero)
    };
    let (a, b, c) = (at(2), at(1), at(0));
    for e in [&a, &b, &c] {
        if !(crate::core::assumptions::is_number(e) && yes(e, Predicate::real)) {
            return Ok(None);
        }
    }
    if matches!(certified_sign(&a)?, None | Some(0)) {
        return Ok(None);
    }
    use crate::core::mul::{div, mul};
    let four_ac: Basic = mul(&[mul(&[Basic::int(4), a.clone()])?, c.clone()])?;
    let discriminant: Basic = crate::core::add::sub(&crate::core::power::pow(&b, &Basic::int(2))?, &four_ac)?;
    let Some(sign) = certified_sign(&discriminant)? else {
        return Ok(None);
    };
    let two_a: Basic = mul(&[Basic::int(2), a.clone()])?;
    let minus_b: Basic = crate::core::mul::neg(&b)?;
    if sign < 0 {
        return Ok(Some(empty_set()));
    }
    if sign == 0 {
        return Ok(Some(finite_set(&[div(&minus_b, &two_a)?])));
    }
    let root: Basic = crate::core::power::pow(&discriminant, &Basic::half())?;
    Ok(Some(finite_set(&[
        div(&crate::core::add::sub(&minus_b, &root)?, &two_a)?,
        div(&crate::core::add::add(&[minus_b.clone(), root.clone()])?, &two_a)?,
    ])))
}

/// The real zeros of `f` by the native transcendental solver, factor by
/// factor (`_zero_set`).
fn zero_set(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let v: Basic = crate::core::symbol::Dummy::new("v", &[(Predicate::real, true)]).0;
    let rule: std::collections::HashMap<Basic, Basic> = [(x.clone(), v.clone())].into_iter().collect();
    let Some(found) = zero_set_of(&crate::core::function::xreplace_evaluated(f, &rule)?, &v)? else {
        return Ok(None);
    };
    let back: std::collections::HashMap<Basic, Basic> = [(v, x.clone())].into_iter().collect();
    Ok(Some(crate::core::function::xreplace_evaluated(&found, &back)?))
}

fn zero_set_of(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let g: Basic = crate::core::exprtools::factor_terms(f, false, true, false, false)?;
    if g.is_mul() {
        let mut parts: Vec<Basic> = Vec::new();
        for a in g.args() {
            if !a.free_symbols().contains(x) {
                continue;
            }
            let Some(part) = zero_set_of(a, x)? else {
                return Ok(None);
            };
            parts.push(part);
        }
        return Ok(Some(if parts.is_empty() { empty_set() } else { union(&parts)? }));
    }
    if g.is_pow() && !g.args()[1].free_symbols().contains(x) {
        if yes(&g.args()[1], Predicate::positive) {
            return zero_set_of(&g.args()[0], x);
        }
        if yes(&g.args()[1], Predicate::negative) {
            return Ok(Some(empty_set()));
        }
    }
    if matches!(g.head(), Head::Function(Function::Exp)) {
        return Ok(Some(empty_set()));
    }
    if let Some(h) = harmonic_zeros(&g, x)? {
        return Ok(Some(h));
    }
    let solver = |formula: &Basic, v: &Basic| native_solver(formula, v);
    crate::solvers::transcendental::solve_transcendental(
        &crate::core::relational::eq(&g, &Basic::zero())?,
        x,
        Some(&solver),
        None,
        false,
    )
}

/// The zeros of `a*sin(u) + b*cos(u)` with `u` linear (`_harmonic_zeros`).
fn harmonic_zeros(g: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let terms: Vec<Basic> = crate::core::expr::terms_of(g);
    let mut argument: Option<Basic> = None;
    let (mut a, mut b) = (Basic::zero(), Basic::zero());
    for t in &terms {
        let (coefficient, rest) = crate::core::expr::as_independent(t, x, false)?;
        let is_sin: bool = matches!(rest.head(), Head::Function(Function::Sin));
        if !(is_sin || matches!(rest.head(), Head::Function(Function::Cos))) {
            return Ok(None);
        }
        let arg: Basic = rest.args()[0].clone();
        match &argument {
            None => argument = Some(arg),
            Some(existing) if *existing != arg => return Ok(None),
            _ => {}
        }
        if is_sin {
            a = crate::core::add::add(&[a, coefficient])?;
        } else {
            b = crate::core::add::add(&[b, coefficient])?;
        }
    }
    let Some(argument) = argument else { return Ok(None) };
    if terms.len() != 2 {
        return Ok(None);
    }
    let slope: Basic = crate::core::function::derivative(&argument, x)?;
    if slope.has(x) || matches!(certified_sign(&slope)?, None | Some(0)) {
        return Ok(None);
    }
    for e in [&a, &b] {
        if !(crate::core::assumptions::is_number(e) && yes(e, Predicate::real)) {
            return Ok(None);
        }
    }
    if matches!(certified_sign(&a)?, None | Some(0)) {
        return Ok(None);
    }
    let offset: Basic = crate::core::subs::subs(&argument, x, &Basic::zero())?;
    let k: Basic = crate::core::symbol::Dummy::new("k", &[(Predicate::integer, true)]).0;
    use crate::core::mul::{div, mul};
    let atan: Basic = crate::functions::elementary::inverse::atan(&div(&crate::core::mul::neg(&b)?, &a)?)?;
    let numerator: Basic = crate::core::add::sub(
        &crate::core::add::add(&[atan, mul(&[k.clone(), Basic::pi()])?])?,
        &offset,
    )?;
    Ok(Some(crate::sets::fancysets::image_set(
        &Basic::new(Head::Tuple, vec![k]),
        &div(&numerator, &slope)?,
        &[Basic::atom(Head::Integers)],
    )?))
}

const MAX_PERIODIC_POINTS: i64 = 1000;

fn image_points(image: &Basic, k: &Basic, piece: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let a: Basic = crate::core::function::derivative(image, k)?;
    let b: Basic = crate::core::subs::subs(image, k, &Basic::zero())?;
    if a.has(k) || !plain_symbols(&a).is_empty() || !plain_symbols(&b).is_empty() {
        return Ok(None);
    }
    let Some(sa) = certified_sign(&a)? else { return Ok(None) };
    if sa == 0 {
        let member: Basic = crate::sets::sets::contains(piece, &b)?;
        return Ok(match member.head() {
            Head::BooleanTrue => Some(vec![b]),
            Head::BooleanFalse => Some(vec![]),
            _ => None,
        });
    }
    let (start, end, left_open, right_open) = interval_parts(piece);
    if !(yes(&start, Predicate::finite) && yes(&end, Predicate::finite)) {
        return Ok(None);
    }
    let digits: usize = crate::settings::settings().precision;
    let mut ends: Vec<f64> = Vec::new();
    for e in [&start, &end] {
        let value: Basic = crate::core::mul::div(&crate::core::add::sub(e, &b)?, &a)?;
        let Some(r) = evalf_rational(&value, digits) else {
            return Ok(None);
        };
        ends.push(rat_to_f64(&r));
    }
    let low: i64 = ends[0].min(ends[1]).trunc() as i64 - 2;
    let high: i64 = ends[0].max(ends[1]).trunc() as i64 + 2;
    if high - low > MAX_PERIODIC_POINTS {
        return Ok(None);
    }
    let mut points: Vec<Basic> = Vec::new();
    for j in low..=high {
        let point: Basic = crate::core::subs::subs(image, k, &Basic::int(j))?;
        let (Some(above), Some(below)) = (certified_compare(&point, &start)?, certified_compare(&end, &point)?) else {
            return Ok(None);
        };
        if (above > 0 || (above == 0 && !left_open)) && (below > 0 || (below == 0 && !right_open)) {
            points.push(point);
        }
    }
    Ok(Some(points))
}

fn rat_to_f64(r: &Rat) -> f64 {
    // the nearest double, as Python's float of a Float (only truncated
    // to an integer with a margin by the caller)
    let v: Rat = round_nearest(r, 53);
    let n: f64 = numer(&v).to_string().parse::<f64>().unwrap_or(f64::NAN);
    let d: f64 = denom(&v).to_string().parse::<f64>().unwrap_or(f64::NAN);
    n / d
}

fn points_in(zeros: &Basic, domain: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    match zeros.head() {
        Head::EmptySet => return Ok(Some(vec![])),
        Head::FiniteSet => {
            let mut points: Vec<Basic> = Vec::new();
            for z in zeros.args() {
                let member: Basic = crate::sets::sets::contains(domain, z)?;
                match member.head() {
                    Head::BooleanTrue => points.push(z.clone()),
                    Head::BooleanFalse => {}
                    _ => return Ok(None),
                }
            }
            return Ok(Some(points));
        }
        Head::Union => {
            let mut collected: Vec<Basic> = Vec::new();
            for part in zeros.args() {
                let Some(found) = points_in(part, domain)? else {
                    return Ok(None);
                };
                for z in found {
                    if !collected.contains(&z) {
                        collected.push(z);
                    }
                }
            }
            return Ok(Some(collected));
        }
        Head::ImageSet => {
            let (variables, image, base_sets) = (
                zeros.args()[0].args().to_vec(),
                zeros.args()[1].clone(),
                zeros.args()[2..].to_vec(),
            );
            if variables.len() != 1
                || base_sets.len() != 1
                || !matches!(base_sets[0].head(), Head::Integers)
                || matches!(image.head(), Head::Tuple)
            {
                return Ok(None);
            }
            let Some(ps) = pieces(domain) else { return Ok(None) };
            let mut listed: Vec<Basic> = Vec::new();
            for piece in ps {
                let Some(part) = image_points(&image, &variables[0], &piece)? else {
                    return Ok(None);
                };
                for z in part {
                    if !listed.contains(&z) {
                        listed.push(z);
                    }
                }
            }
            return Ok(Some(listed));
        }
        _ => {}
    }
    Ok(None)
}

fn is_discrete(zeros: &Basic, piece: &Basic) -> PolyResult<bool> {
    Ok(match zeros.head() {
        Head::FiniteSet | Head::EmptySet => true,
        Head::ImageSet => {
            let base_sets = &zeros.args()[2..];
            base_sets.len() == 1 && matches!(base_sets[0].head(), Head::Integers | Head::Naturals | Head::Naturals0)
        }
        Head::Interval | Head::Reals => {
            let i: Basic = intersection(&[zeros.clone(), piece.clone()])?;
            matches!(i.head(), Head::EmptySet | Head::FiniteSet)
        }
        Head::Union => {
            for a in zeros.args() {
                if !is_discrete(a, piece)? {
                    return Ok(false);
                }
            }
            true
        }
        Head::Intersection => {
            for a in zeros.args() {
                if is_discrete(a, piece)? {
                    return Ok(true);
                }
            }
            false
        }
        _ => false,
    })
}

/// The zeros of `f` in `domain` when they are finitely many and found
/// exactly (`_finite_zeros`).
pub fn finite_zeros(f: &Basic, x: &Basic, domain: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    if is_rational_in(f, x) {
        let (found, zeros) = real_zeros_in(f, x, domain)?;
        if found {
            return Ok(Some(zeros));
        }
    }
    let Some(solutions) = zero_set(f, x)? else {
        return Ok(None);
    };
    points_in(&solutions, domain)
}

/// Whether the zeros of `f` on the piece are isolated points
/// (`_has_discrete_zeros`).
pub fn has_discrete_zeros(f: &Basic, x: &Basic, piece: &Basic) -> PolyResult<bool> {
    if is_rational_in(f, x) && real_zeros_in(f, x, piece)?.0 {
        return Ok(true);
    }
    match zero_set(f, x)? {
        Some(s) => is_discrete(&s, piece),
        None => Ok(false),
    }
}

/// The set of signs which `f` takes on `domain` (`sign_on`).
pub fn sign_on(f: &Basic, x: &Basic, domain: &Basic, depth: usize) -> PolyResult<Option<Signs>> {
    let Some(ps) = pieces(domain) else { return Ok(None) };
    if !f.has(x) {
        return Ok(certified_sign(f)?.map(|s| signs_of(&[s])));
    }
    let continuous: Option<bool> =
        crate::timeout::attempt_within(crate::settings::settings().timeout, || continuous_on(f, x, domain))?;
    if continuous != Some(true) {
        return Ok(None);
    }
    let zero_list: Option<Vec<Basic>> = attempt_opt(|| finite_zeros(f, x, domain))?;
    if let Some(zeros) = &zero_list {
        if zeros.iter().all(|z| yes(z, Predicate::real)) {
            return signs_from_zeros(f, x, &ps, zeros);
        }
    }
    let bounded: Option<Signs> = signs_from_bounds(f, x, &ps)?;
    if bounded.as_ref().is_some_and(|b| b.len() == 1) {
        return Ok(bounded);
    }
    if depth > 0 {
        if let Some(monotone) = signs_from_monotonicity(f, x, &ps, depth)? {
            return Ok(Some(monotone));
        }
    }
    let mut isolated: Signs = Signs::new();
    for piece in &ps {
        let Some(part) = super::intervals::isolate_signs(f, x, piece, 400)? else {
            return Ok(bounded);
        };
        isolated.extend(part);
    }
    Ok(Some(isolated))
}

fn signs_from_bounds(f: &Basic, x: &Basic, ps: &[Basic]) -> PolyResult<Option<Signs>> {
    let mut signs: Signs = Signs::new();
    for piece in ps {
        let (start, end, _, _) = interval_parts(piece);
        let bounds: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            crate::core::subs::subs(f, x, &crate::core::accumbounds::accum_bounds(&start, &end)?)
        })?;
        let Some(bounds) = bounds else { return Ok(None) };
        let (low, high) = if crate::core::accumbounds::is_bounds(&bounds) {
            (bounds.args()[0].clone(), bounds.args()[1].clone())
        } else if plain_symbols(&bounds).is_empty() {
            (bounds.clone(), bounds.clone())
        } else {
            return Ok(None);
        };
        let (Some(ls), Some(hs)) = (certified_sign(&low)?, certified_sign(&high)?) else {
            return Ok(None);
        };
        if ls == 1 {
            signs.insert(1);
        } else if hs == -1 {
            signs.insert(-1);
        } else if ls == 0 && hs == 0 {
            signs.insert(0);
        } else if ls == 0 {
            signs.extend([0, 1]);
        } else if hs == 0 {
            signs.extend([-1, 0]);
        } else {
            signs.extend([-1, 0, 1]);
        }
    }
    Ok(Some(signs))
}

fn signs_from_zeros(f: &Basic, x: &Basic, ps: &[Basic], zeros: &[Basic]) -> PolyResult<Option<Signs>> {
    let mut signs: Signs = Signs::new();
    if !zeros.is_empty() {
        signs.insert(0);
    }
    for piece in ps {
        let (a, b, _, _) = interval_parts(piece);
        let mut inside: Vec<Basic> = Vec::new();
        for z in zeros {
            if certified_compare(z, &a)? == Some(1) && certified_compare(&b, z)? == Some(1) {
                inside.push(z.clone());
            }
        }
        let mut all: Vec<Basic> = vec![a.clone()];
        all.extend(inside);
        all.push(b.clone());
        let Some(ordered) = sorted_constants(&all)? else {
            return Ok(None);
        };
        if ordered.len() < 2 {
            let Some(s) = certified_sign(&crate::core::subs::subs(f, x, &ordered[0])?)? else {
                return Ok(None);
            };
            signs.insert(s);
            continue;
        }
        for w in ordered.windows(2) {
            let Some(sample) = between(&w[0], &w[1])? else {
                return Ok(None);
            };
            let Some(s) = certified_sign(&crate::core::subs::subs(f, x, &sample)?)? else {
                return Ok(None);
            };
            signs.insert(s);
        }
    }
    Ok(Some(signs))
}

/// The limit of `f` at a point, `None` when not found (`_endpoint_value`).
pub fn endpoint_value(f: &Basic, x: &Basic, point: &Basic, direction: &str) -> PolyResult<Option<Basic>> {
    let value: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::core::limits::limit(f, x, point, direction)
    })?;
    Ok(value.filter(|v| plain_symbols(v).is_empty()))
}

fn value_at(f: &Basic, x: &Basic, point: &Basic, direction: &str) -> PolyResult<Option<Basic>> {
    if yes(point, Predicate::finite) {
        let value: Basic = crate::core::subs::subs(f, x, point)?;
        if plain_symbols(&value).is_empty() && is(&value, Predicate::finite) != Some(false) {
            return Ok(Some(value));
        }
    }
    endpoint_value(f, x, point, direction)
}

fn monotone_signs(f: &Basic, x: &Basic, a: &Basic, b: &Basic, increasing: bool) -> PolyResult<Option<Signs>> {
    let (Some(left), Some(right)) = (value_at(f, x, a, "+")?, value_at(f, x, b, "-")?) else {
        return Ok(None);
    };
    let (low, high) = if increasing { (left, right) } else { (right, left) };
    let (Some(ls), Some(hs)) = (certified_sign(&low)?, certified_sign(&high)?) else {
        return Ok(None);
    };
    if ls >= 0 {
        return Ok(Some(signs_of(&[1])));
    }
    if hs <= 0 {
        return Ok(Some(signs_of(&[-1])));
    }
    Ok(Some(signs_of(&[-1, 0, 1])))
}

fn signs_from_monotonicity(f: &Basic, x: &Basic, ps: &[Basic], depth: usize) -> PolyResult<Option<Signs>> {
    let derivative: Basic = crate::core::function::derivative(f, x)?;
    let mut signs: Signs = Signs::new();
    for piece in ps {
        let (a, b, left_open, right_open) = interval_parts(piece);
        let critical: Option<Vec<Basic>> = attempt_opt(|| finite_zeros(&derivative, x, piece))?;
        if let Some(critical) = critical {
            if !critical.iter().all(|c| yes(c, Predicate::real)) {
                return Ok(None);
            }
            let mut inside: Vec<Basic> = Vec::new();
            for c in &critical {
                if certified_compare(c, &a)? == Some(1) && certified_compare(&b, c)? == Some(1) {
                    inside.push(c.clone());
                }
            }
            let mut all: Vec<Basic> = vec![a.clone()];
            all.extend(inside.iter().cloned());
            all.push(b.clone());
            let Some(ordered) = sorted_constants(&all)? else {
                return Ok(None);
            };
            for w in ordered.windows(2) {
                let Some(sample) = between(&w[0], &w[1])? else {
                    return Ok(None);
                };
                let slope = certified_sign(&crate::core::subs::subs(&derivative, x, &sample)?)?;
                let Some(slope) = slope.filter(|s| *s != 0) else {
                    return Ok(None);
                };
                let Some(part) = monotone_signs(f, x, &w[0], &w[1], slope == 1)? else {
                    return Ok(None);
                };
                signs.extend(part);
            }
            for c in &inside {
                let Some(value) = certified_sign(&crate::core::subs::subs(f, x, c)?)? else {
                    return Ok(None);
                };
                signs.insert(value);
            }
        } else {
            let Some(slope_signs) = sign_on(&derivative, x, piece, depth - 1)? else {
                return Ok(None);
            };
            if slope_signs == signs_of(&[0]) || (slope_signs.contains(&1) && slope_signs.contains(&-1)) {
                return Ok(None);
            }
            let strict: bool = !slope_signs.contains(&0) || has_discrete_zeros(&derivative, x, piece)?;
            if !strict {
                return Ok(None);
            }
            let Some(part) = monotone_signs(f, x, &a, &b, slope_signs.contains(&1))? else {
                return Ok(None);
            };
            signs.extend(part);
        }
        for (closed, point, direction) in [
            (!left_open && yes(&a, Predicate::finite), &a, "+"),
            (!right_open && yes(&b, Predicate::finite), &b, "-"),
        ] {
            if closed {
                let end_value: Option<Basic> = value_at(f, x, point, direction)?;
                let end_sign: Option<i32> = match end_value {
                    Some(v) => certified_sign(&v)?,
                    None => None,
                };
                let Some(s) = end_sign else { return Ok(None) };
                signs.insert(s);
            }
        }
    }
    Ok(Some(signs))
}

fn numeric_key(e: &Basic) -> PolyResult<(i32, Rat)> {
    match e.head() {
        Head::NegativeInfinity => return Ok((-1, Rat::ZERO)),
        Head::Infinity => return Ok((1, Rat::ZERO)),
        _ => {}
    }
    if let Some(r) = e.as_rational() {
        return Ok((0, r.clone()));
    }
    match evalf_rational(e, 20) {
        Some(r) => Ok((0, r)),
        None => Err(PolyError::Type(format!("Invalid comparison of non-real {}", e))),
    }
}

/// The box of the values of the symbols allowed by the facts, with the
/// openness of its ends (`box_of`).
pub fn box_of(
    symbols: &[Basic],
    facts: &super::facts::Facts,
) -> PolyResult<Option<(super::intervals::Box, super::intervals::Openness)>> {
    let mut bx: super::intervals::Box = Vec::new();
    let mut openness: super::intervals::Openness = Vec::new();
    for v in symbols {
        let Some(domain) = domain_of(v, facts)? else {
            return Ok(None);
        };
        let Some(ps) = pieces(&domain) else { return Ok(None) };
        if ps.is_empty() {
            return Ok(None);
        }
        let (mut first, mut last) = (ps[0].clone(), ps[0].clone());
        let mut first_key = numeric_key(&interval_parts(&first).0)?;
        let mut last_key = numeric_key(&interval_parts(&last).1)?;
        for p in &ps[1..] {
            let (start_key, end_key) = (numeric_key(&interval_parts(p).0)?, numeric_key(&interval_parts(p).1)?);
            if start_key < first_key {
                first = p.clone();
                first_key = start_key;
            }
            if last_key < end_key {
                last = p.clone();
                last_key = end_key;
            }
        }
        let (fs, _, flo, _) = interval_parts(&first);
        let (_, le, _, lro) = interval_parts(&last);
        bx.push((v.clone(), (fs, le)));
        openness.push((v.clone(), (flo, lro)));
    }
    Ok(Some((bx, openness)))
}

fn single_argument(f: &Basic, symbols: &[Basic]) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    let v: Basic = crate::core::symbol::Symbol::new("v", &[(Predicate::real, true)]).0;
    let mut wanted: Vec<Basic> = symbols.to_vec();
    wanted.sort();
    let mut candidates: Vec<Basic> = Vec::new();
    for e in f.preorder().into_iter().skip(1) {
        let mut s: Vec<Basic> = plain_symbols(&e);
        s.sort();
        if s == wanted && !e.is_symbol() && !matches!(e.head(), Head::Tuple) {
            candidates.push(e);
        }
    }
    let mut keyed: Vec<(i64, Basic)> = Vec::new();
    for c in candidates {
        keyed.push((-crate::core::function::count_ops(&c)?, c));
    }
    keyed.sort_by_key(|(k, _)| *k);
    for (_, u) in keyed {
        let rule: std::collections::HashMap<Basic, Basic> = [(u.clone(), v.clone())].into_iter().collect();
        let big_f: Basic = crate::core::function::xreplace_evaluated(f, &rule)?;
        if plain_symbols(&big_f) == vec![v.clone()] {
            return Ok(Some((big_f, u, v)));
        }
    }
    Ok(None)
}

/// The signs of an expression of several real variables on the box of
/// their assumptions (`signs_multivariate`).
pub fn signs_multivariate(f: &Basic, symbols: &[Basic], facts: &super::facts::Facts) -> PolyResult<Option<Signs>> {
    let Some((bx, openness)) = box_of(symbols, facts)? else {
        return Ok(None);
    };
    let partial_sign = |derivative: &Basic| -> PolyResult<Option<Signs>> {
        let inner: Vec<Basic> = plain_symbols(derivative);
        if inner.is_empty() {
            return Ok(certified_sign(derivative)?.map(|s| signs_of(&[s])));
        }
        if inner.len() == 1 {
            let w: &Basic = &inner[0];
            return match domain_of(w, facts)? {
                Some(domain) => sign_on(derivative, w, &domain, 0),
                None => Ok(None),
            };
        }
        super::intervals::signs_on_box(derivative, &bx, 120, true)
    };
    if let Some(signs) = super::intervals::corner_signs(f, &bx, &openness, Some(&partial_sign), 40)? {
        return Ok(Some(signs));
    }
    if let Some(signs) = super::intervals::signs_on_box(f, &bx, 120, false)? {
        return Ok(Some(signs));
    }
    if let Some((big_f, u, v)) = single_argument(f, symbols)? {
        let mut rng: Option<Basic> = super::intervals::monotone_range(&u, &bx, &openness, Some(&partial_sign), 40)?;
        if rng.is_none() {
            rng = super::intervals::range_on_box(&u, &bx)?;
        }
        if let Some(r) = rng {
            return sign_on(&big_f, &v, &r, 1);
        }
    }
    Ok(None)
}

fn real_on(
    f: &Basic,
    x: Option<&Basic>,
    domain: Option<&Basic>,
    facts: &super::facts::Facts,
    symbols: &[Basic],
) -> PolyResult<bool> {
    if f.has(&Basic::imaginary_unit()) {
        return Ok(false);
    }
    let mut checks: Vec<(Basic, Signs)> = Vec::new();
    for node in f.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Function(Function::Log)))) {
        checks.push((node.args()[0].clone(), signs_of(&[1])));
    }
    for node in f.atoms(Some(&|n: &Basic| n.is_pow())) {
        let exponent: &Basic = &node.args()[1];
        if let Some(r) = exponent.as_rational() {
            if !exponent.is_integer_value() && !node.args()[0].free_symbols().is_empty() {
                checks.push((
                    node.args()[0].clone(),
                    if rat_is_negative(r) {
                        signs_of(&[1])
                    } else {
                        signs_of(&[0, 1])
                    },
                ));
            }
        }
    }
    for (inner, allowed) in checks {
        let signs: Option<Signs> = match (x, domain) {
            (Some(x), Some(d)) => sign_on(&inner, x, d, 1)?,
            _ => signs_multivariate(&inner, symbols, facts)?,
        };
        if !matches!(signs, Some(s) if s.is_subset(&allowed)) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The truth of a relation under the facts, by the sign analysis of the
/// difference of its sides (`decide_relational`).
pub fn decide_relational(atom: &Basic, facts: &super::facts::Facts) -> PolyResult<Option<bool>> {
    let symbols: Vec<Basic> = plain_symbols(atom);
    if symbols.is_empty() {
        return Ok(None);
    }
    let Head::Relational(op) = atom.head() else {
        return Ok(None);
    };
    let f: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
    let signs: Option<Signs> = if symbols.len() == 1 {
        let x: &Basic = &symbols[0];
        let Some(domain) = domain_of(x, facts)? else {
            return Ok(None);
        };
        if matches!(domain.head(), Head::EmptySet) {
            return Ok(None);
        }
        if !real_on(&f, Some(x), Some(&domain), facts, &[])? {
            return Ok(None);
        }
        sign_on(&f, x, &domain, 1)?
    } else {
        let mut ordered: Vec<Basic> = symbols.clone();
        ordered.sort_by_key(|v| match v.head() {
            Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
            _ => v.to_string(),
        });
        if !real_on(&f, None, None, facts, &ordered)? {
            return Ok(None);
        }
        signs_multivariate(&f, &ordered, facts)?
    };
    let Some(signs) = signs else { return Ok(None) };
    let holds: Signs = signs_of(match op {
        RelOp::Gt => &[1],
        RelOp::Ge => &[0, 1],
        RelOp::Lt => &[-1],
        RelOp::Le => &[-1, 0],
        RelOp::Eq => &[0],
        RelOp::Ne => &[-1, 1],
    });
    if signs.is_subset(&holds) {
        return Ok(Some(true));
    }
    if signs.is_disjoint(&holds) {
        return Ok(Some(false));
    }
    Ok(None)
}
