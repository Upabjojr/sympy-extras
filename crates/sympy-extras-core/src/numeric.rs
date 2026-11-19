//! Numerical values which do not depend on a rounding error, ported from
//! `sympy_extras._numeric`: [`reliable_form`] writes `0` for the constant
//! sums which are proved hidden zeros and refuses (`None`) the constants
//! whose value hangs on a rounding (a sum without a significant digit
//! which is not proved zero, an argument on a cut on the side the rounding
//! chooses); [`reliable_value`] evaluates the others at doubling
//! precisions until two evaluations agree.
//!
//! The values are those of [`numeric_value`] (dashu floats; mpmath in the
//! Python), the sizes and thresholds Python floats as `f64`; the value is
//! a [`Complex`] at the precision of the digits asked (the Python's
//! `Float` expression; the Rust core has no `Float`). There are no foreign
//! nodes, hence no foreign minimal polynomials, and no time limit on the
//! proofs ([`crate::timeout::attempt`]).

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{numeric_value, parts, Complex, Real};
use crate::core::expr::{is, is_true};
use crate::error::{PolyError, PolyResult};
use crate::numbers::Int;

/// The additional digits of the evaluations of a part which looks like a
/// rounding error.
const MORE_DIGITS: [usize; 3] = [15, 30, 60];
/// The guard bits of every evaluation.
const GUARD_BITS: usize = 20;
/// The doublings of the precision allowed for the final value.
const DOUBLINGS: usize = 2;

/// mpmath's bits for `dps` decimal digits (`dps_to_prec`).
pub fn dps_to_prec(dps: usize) -> usize {
    1usize.max((((dps + 1) as f64) * 3.3219280948873626).round() as usize)
}

/// A sum without a significant digit which is not proved zero.
struct Undecided;

type Parts = (Real, Real);

fn is_finite(z: &Complex) -> bool {
    let (a, b) = parts(z);
    a.repr().is_finite() && b.repr().is_finite()
}

fn number(e: &Basic, prec: usize) -> Option<Complex> {
    numeric_value(e, prec).filter(is_finite)
}

fn value_parts(e: &Basic, digits: usize) -> Option<Parts> {
    number(e, dps_to_prec(digits) + GUARD_BITS).map(|z| parts(&z))
}

fn is_zero_real(x: &Real) -> bool {
    *x == Real::ZERO
}

fn size(x: &Real) -> f64 {
    x.to_f64().value().abs()
}

/// A function application (SymPy's `Function`, `Piecewise` among them).
fn is_function(node: &Basic) -> bool {
    matches!(
        node.head(),
        Head::Function(_) | Head::Undefined(_) | Head::Piecewise | Head::Hyper | Head::MeijerG
    )
}

/// Whether the parts of the value are rounding errors (`_noise`); `None`
/// when a value at a higher precision cannot be computed.
fn noise(e: &Basic, digits: usize, scale: f64, values: &Parts, structural: (bool, bool)) -> Option<(bool, bool)> {
    let threshold: f64 = 10f64.powi(-((digits * 3 / 5) as i32)) * scale;
    let suspect = |part: &Real, has_part: bool| -> bool {
        (!is_zero_real(part) && size(part) < threshold) || (is_zero_real(part) && has_part)
    };
    let suspects: [bool; 2] = [suspect(&values.0, structural.0), suspect(&values.1, structural.1)];
    if !suspects[0] && !suspects[1] {
        return Some((false, false));
    }
    let mut genuine: [bool; 2] = [false, false];
    let mut previous: Parts = values.clone();
    for more in MORE_DIGITS {
        let again: Parts = value_parts(e, digits + more)?;
        for k in 0..2 {
            if suspects[k] && !genuine[k] {
                let (first, second) = if k == 0 {
                    (&previous.0, &again.0)
                } else {
                    (&previous.1, &again.1)
                };
                if !is_zero_real(first) && !is_zero_real(second) && size(&(first - second)) <= 1e-5 * size(second) {
                    genuine[k] = true;
                }
            }
        }
        if (0..2).all(|k| genuine[k] || !suspects[k]) {
            break;
        }
        previous = again;
    }
    Some((suspects[0] && !genuine[0], suspects[1] && !genuine[1]))
}

/// Whether the node is an `Expr` of the Python (not a Boolean, a set, a
/// tuple or a pair of a piecewise).
fn is_expr(e: &Basic) -> bool {
    !(crate::core::logic::is_boolean(e)
        || crate::sets::sets::is_set(e)
        || matches!(e.head(), Head::Tuple | Head::ExprCondPair))
}

fn is_constant(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}

fn has_function(e: &Basic) -> bool {
    e.preorder().iter().any(is_function)
}

/// Whether `part` is proved zero: the core's `is_zero`, or the minimal
/// polynomial `x` of an algebraic number without functions
/// (`_proved_zero`).
pub fn proved_zero(part: &Basic) -> bool {
    match is(part, Predicate::zero) {
        Some(true) => return true,
        Some(false) => return false,
        None => {}
    }
    if !part.free_symbols().is_empty() || has_function(part) {
        return false;
    }
    match crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(part) {
        Ok(c) => c == vec![Int::ONE, Int::ZERO],
        Err(_) => false,
    }
}

/// Whether some term has a real part and an imaginary part which are not
/// exactly zero, and the size of the largest term (`_structural`).
fn structural(terms: &[Basic], digits: usize) -> Option<((bool, bool), f64)> {
    let (mut has_real, mut has_imaginary, mut scale) = (false, false, 0f64);
    for term in terms {
        let (a, b) = value_parts(term, digits)?;
        has_real = has_real || !is_zero_real(&a);
        has_imaginary = has_imaginary || !is_zero_real(&b);
        scale = scale.max(a.to_f64().value().hypot(b.to_f64().value()));
    }
    Some(((has_real, has_imaginary), scale))
}

fn resolved_sum(node: &Basic, digits: usize) -> Result<Basic, Undecided> {
    let Some(values) = value_parts(node, digits) else {
        return Ok(node.clone());
    };
    let (has_parts, scale) = structural(node.args(), digits).ok_or(Undecided)?;
    let noisy: (bool, bool) = noise(node, digits, scale, &values, has_parts).ok_or(Undecided)?;
    if !noisy.0 && !noisy.1 {
        return Ok(node.clone());
    }
    let exact: (Basic, Basic) = if is_zero_real(&values.1) && is_true(node, Predicate::extended_real) {
        (node.clone(), Basic::zero())
    } else {
        match crate::timeout::attempt_within(Some(2.0), || {
            crate::functions::elementary::complexes::as_real_imag(node, true, None)
        }) {
            Ok(Some(Some(split))) => split,
            _ => return Err(Undecided),
        }
    };
    let mut kept: Vec<Basic> = Vec::new();
    for (is_noise, part) in [(noisy.0, exact.0), (noisy.1, exact.1)] {
        if is_noise {
            if !proved_zero(&part) {
                return Err(Undecided);
            }
            kept.push(Basic::zero());
        } else {
            kept.push(part);
        }
    }
    let imaginary: Basic = crate::core::mul::mul(&[Basic::imaginary_unit(), kept[1].clone()]).map_err(|_| Undecided)?;
    crate::core::add::add(&[kept[0].clone(), imaginary]).map_err(|_| Undecided)
}

fn is_angular(node: &Basic) -> bool {
    matches!(node.head(), Head::Function(Function::Log | Function::Arg))
}

fn is_inverse(node: &Basic) -> bool {
    matches!(
        node.head(),
        Head::Function(
            Function::Asin
                | Function::Acos
                | Function::Atan
                | Function::Acot
                | Function::Asinh
                | Function::Acosh
                | Function::Atanh
                | Function::Acoth
        )
    )
}

/// Whether a constant argument of a logarithm, a fractional power or an
/// inverse function has a part which is a rounding error deciding the
/// branch (`_on_a_cut`).
fn on_a_cut(e: &Basic, digits: usize) -> bool {
    let mut stack: Vec<Basic> = vec![e.clone()];
    while let Some(node) = stack.pop() {
        stack.extend(node.args().iter().cloned());
        let (argument, angular): (Basic, bool) = if node.is_pow() {
            if is_true(&node.args()[1], Predicate::integer) {
                continue;
            }
            (node.args()[0].clone(), true)
        } else if is_angular(&node) && !node.args().is_empty() && is_expr(&node.args()[0]) {
            (node.args()[0].clone(), true)
        } else if is_inverse(&node) && !node.args().is_empty() && is_expr(&node.args()[0]) {
            (node.args()[0].clone(), false)
        } else {
            continue;
        };
        if !is_constant(&argument) || argument.as_rational().is_some() {
            continue;
        }
        let Some(values) = value_parts(&argument, digits) else {
            return true;
        };
        let mut has_parts: (bool, bool) = (false, false);
        if argument.is_add() {
            match structural(argument.args(), digits) {
                Some((found, _)) => has_parts = found,
                None => return true,
            }
        }
        let Some(noisy) = noise(&argument, digits, size(&values.0) + size(&values.1), &values, has_parts) else {
            return true;
        };
        if !noisy.0 && !noisy.1 {
            continue;
        }
        if !angular || (noisy.0 && noisy.1) {
            return true;
        }
        if noisy.1 && values.0 <= Real::ZERO {
            return true;
        }
        if noisy.0 && is_zero_real(&values.1) {
            return true;
        }
    }
    false
}

fn walk(node: &Basic, exposed: bool, digits: usize) -> Result<Basic, WalkError> {
    if node.args().is_empty() {
        return Ok(node.clone());
    }
    let inside: bool = exposed || is_function(node) || (node.is_pow() && !is_true(&node.args()[1], Predicate::integer));
    let mut changed: bool = false;
    let mut args: Vec<Basic> = Vec::new();
    for argument in node.args() {
        let new: Basic = if !is_expr(argument) {
            argument.clone()
        } else {
            walk(argument, inside, digits)?
        };
        changed = changed || !new.ptr_eq(argument);
        args.push(new);
    }
    let rebuilt: Basic = if changed {
        crate::core::function::rebuild_evaluated(node, args).map_err(WalkError::Error)?
    } else {
        node.clone()
    };
    if exposed && rebuilt.is_add() && is_constant(&rebuilt) {
        return resolved_sum(&rebuilt, digits).map_err(|_| WalkError::Undecided);
    }
    Ok(rebuilt)
}

enum WalkError {
    Undecided,
    Error(PolyError),
}

/// The expression with its constant sums which are hidden zeros written
/// `0`, or `None` when the value of a constant in it depends on a rounding
/// error (`reliable_form`, 20 digits by default in the Python).
pub fn reliable_form(e: &Basic, digits: usize) -> Option<Basic> {
    let resolved: Basic = match walk(e, false, digits) {
        Ok(r) => r,
        Err(WalkError::Undecided) => return None,
        Err(WalkError::Error(PolyError::Index(_))) => return None,
        Err(WalkError::Error(_)) => return None,
    };
    if on_a_cut(&resolved, digits) {
        return None;
    }
    Some(resolved)
}

fn abs_complex(z: &Complex) -> Real {
    let (a, b) = parts(z);
    (&a * &a + &b * &b).sqrt()
}

/// The value to `digits` digits at doubling precisions until two
/// evaluations agree (`_accurate`).
fn accurate(e: &Basic, digits: usize) -> Option<Complex> {
    let target: usize = dps_to_prec(digits);
    let mut prec: usize = target + GUARD_BITS;
    let mut value: Complex = number(e, prec)?;
    for _ in 0..DOUBLINGS {
        prec *= 2;
        let again: Complex = number(e, prec)?;
        let difference: Real = abs_complex(&(&again - &value));
        let bound: Real = {
            let magnitude: Real = abs_complex(&again);
            let shift: isize = 4 - target as isize;
            if shift >= 0 {
                magnitude * Real::from(1u64 << shift)
            } else {
                magnitude / Real::from(Int::ONE << (-shift) as usize)
            }
        };
        if difference <= bound {
            return Some(again);
        }
        value = again;
    }
    Some(value)
}

fn rounded(z: &Complex, prec: usize) -> Complex {
    let (a, b) = parts(z);
    crate::core::evalf::complex(a.with_precision(prec).value(), b.with_precision(prec).value())
}

/// The value with `digits` digits of `e` at the `values` of its symbols,
/// or `None` when it depends on a rounding error or is not a finite number
/// (`reliable_value`; a complex number at the precision of the digits).
pub fn reliable_value(e: &Basic, digits: usize, values: &[(Basic, Basic)]) -> Option<Complex> {
    let mut form: Option<Basic> = reliable_form(e, digits);
    if let (Some(f), false) = (&form, values.is_empty()) {
        let rule: HashMap<Basic, Basic> = values.iter().cloned().collect();
        let substituted: Basic = crate::core::function::xreplace_evaluated(f, &rule).ok()?;
        if !is_expr(&substituted) {
            return None;
        }
        form = reliable_form(&substituted, digits);
    }
    let form: Basic = form?;
    if !form.free_symbols().is_empty() {
        return None;
    }
    let z: Complex = accurate(&form, digits)?;
    Some(rounded(&z, dps_to_prec(digits)))
}

/// [`reliable_value`] as a pair of floats, `None` when it is not reliable
/// or does not fit (`reliable_complex`).
pub fn reliable_complex(e: &Basic, digits: usize, values: &[(Basic, Basic)]) -> Option<(f64, f64)> {
    let z: Complex = reliable_value(e, digits, values)?;
    let (a, b) = parts(&z);
    let (x, y) = (a.to_f64().value(), b.to_f64().value());
    if x.is_finite() && y.is_finite() {
        Some((x, y))
    } else {
        None
    }
}

/// Whether the number is proved zero (`proved_zero`).
pub fn proved_zero_number(e: &Basic) -> PolyResult<bool> {
    Ok(proved_zero(e))
}
