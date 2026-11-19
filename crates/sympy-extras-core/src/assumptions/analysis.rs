//! Port of `sympy_extras/assumptions/analysis.py` (the first part): the
//! certified sign of a constant ([`certified_sign`]: the exact knowledge
//! of the core, a hidden zero of an algebraic number by its minimal
//! polynomial, then rigorous interval arithmetic and a value with a margin
//! over its precision when the numerical checks are enabled) and the
//! comparison of two constants.

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
    let coefficients: Option<Vec<Int>> =
        crate::timeout::attempt(|| crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(e))?;
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
    crate::timeout::attempt(|| crate::polys::cad::qe::solution_set(atom, x, &[], None, true))
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
fn evalf_rational(e: &Basic, digits: usize) -> Option<Rat> {
    let prec: usize = crate::numeric::dps_to_prec(digits);
    let v = crate::core::evalf::numeric_value(e, prec + 64)?;
    let (re, im) = crate::core::evalf::parts(&v);
    if !super::ivarith::real_to_rat(&im).is_zero() {
        return None;
    }
    Some(round_nearest(&super::ivarith::real_to_rat(&re), prec))
}

/// `r` rounded to `prec` bits, to the nearest (ties to even).
fn round_nearest(r: &Rat, prec: usize) -> Rat {
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
