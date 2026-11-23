//! Port of `sympy_extras/integrals/parametric.py`: differentiation under
//! the integral sign (Feynman's trick). The derivative with respect to a
//! parameter is integrated by the definite integrator, integrated back in
//! the parameter, and the constant fixed at `p = 0`, `1` or at infinity.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::integrals::conditions::ConditionalValue;

type OptCV = PolyResult<Option<ConditionalValue>>;

/// The definite integrator which differentiation under the integral sign
/// calls, and the numerical check of a result (`ParametricIntegrator`).
pub trait ParametricIntegrator {
    /// `Integral(f, (x, a, b))` at the recursion depth `depth`.
    fn integrate(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV;
    /// Whether `value` agrees with a quadrature of the integral.
    fn verify(&self, value: &Basic, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<bool>>;
}

fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}

fn subs(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Basic> {
    crate::core::subs::subs(e, old, new)
}

fn has_head(e: &Basic, f: &dyn Fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| f(n.head()))
}

fn has_integral(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::Integral | Head::NonElementaryIntegral))
}

/// The parameters of `f` worth differentiating with respect to: the free
/// symbols other than `x` inside a function of `x`, sorted by name
/// (`candidate_parameters`).
pub fn candidate_parameters(f: &Basic, x: &Basic) -> Vec<Basic> {
    let mut found: BTreeSet<Basic> = BTreeSet::new();
    for node in f.atoms(Some(&|n: &Basic| {
        matches!(n.head(), Head::Function(_) | Head::Undefined(_))
    })) {
        if node.has(x) {
            found.extend(node.free_symbols());
        }
    }
    for node in f.atoms(Some(&|n: &Basic| n.is_pow())) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if base.has(x) && !exponent.has(x) {
            found.extend(exponent.free_symbols());
        }
        if exponent.has(x) && !base.has(x) {
            found.extend(base.free_symbols());
        }
    }
    found.remove(x);
    let mut v: Vec<Basic> = found.into_iter().collect();
    v.sort_by_key(|s| s.to_string());
    v
}

/// `Integral(f, (x, a, b))` by differentiation under the integral sign
/// with respect to `parameter` (`parametric_integral`); the integrator is
/// required (the default one of Python is `definite_integrator`).
#[allow(clippy::too_many_arguments)]
pub fn parametric_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    parameter: &Basic,
    assumptions: &[Basic],
    depth: usize,
    integrator: Option<&dyn ParametricIntegrator>,
) -> OptCV {
    let p: &Basic = parameter;
    if !f.has(p) || a.has(p) || b.has(p) {
        return Ok(None);
    }
    let derivative: Basic = crate::core::function::derivative(f, p)?;
    if derivative == Basic::zero() {
        return Ok(None);
    }
    let default: crate::integrals::definite::DefiniteIntegrator;
    let integrator: &dyn ParametricIntegrator = match integrator {
        Some(i) => i,
        None => {
            default = crate::integrals::definite::DefiniteIntegrator::new(assumptions);
            &default
        }
    };
    let Some(inner) = integrator.integrate(&derivative, x, a, b, depth + 1)? else {
        return Ok(None);
    };
    if has_integral(&inner.value) {
        return Ok(None);
    }
    let q: Basic = crate::integrals::definite::fresh("q", &[(Predicate::real, true)]);
    let Some(primitive) = crate::integrals::indefinite::closed_antiderivative(
        &subs(&inner.value, p, &q)?,
        &q,
        false,
        false,
        crate::settings::settings().timeout,
    )?
    else {
        return Ok(None);
    };
    for anchor in [Basic::zero(), Basic::one(), oo()] {
        let Some(value) = anchor_value(f, x, a, b, p, &anchor, integrator, assumptions, depth)? else {
            continue;
        };
        let Some(shift) = difference(&primitive, &q, p, &anchor, assumptions)? else {
            continue;
        };
        let result: Basic = crate::core::add::add(&[value.value.clone(), shift])?;
        if has_head(&result, &|h| {
            matches!(
                h,
                Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity | Head::Limit
            )
        }) {
            continue;
        }
        if crate::settings::settings().numerical_checks && integrator.verify(&result, f, x, a, b)? == Some(false) {
            return Ok(None);
        }
        let condition: Basic = crate::core::logic::and_(&[inner.condition.clone(), value.condition.clone()]);
        return Ok(Some(ConditionalValue::new(result, condition)));
    }
    Ok(None)
}

/// `I(anchor)` computed directly, or `0` at infinity when the integrand
/// decays there (`_anchor_value`).
#[allow(clippy::too_many_arguments)]
fn anchor_value(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    p: &Basic,
    anchor: &Basic,
    integrator: &dyn ParametricIntegrator,
    assumptions: &[Basic],
    depth: usize,
) -> OptCV {
    if *anchor == oo() {
        let vanishes: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            crate::assumptions::limits::limit(f, p, &oo(), "+", assumptions, None)
        })?;
        if vanishes.is_some_and(|v| v == Basic::zero()) && decays(f, x, p)? {
            return Ok(Some(ConditionalValue::new(Basic::zero(), crate::core::logic::true_())));
        }
        return Ok(None);
    }
    let specialised: Basic = subs(f, p, anchor)?;
    if has_head(&specialised, &|h| matches!(h, Head::NaN | Head::ComplexInfinity)) {
        return Ok(None);
    }
    integrator.integrate(&specialised, x, a, b, depth + 1)
}

/// Whether `f` carries a factor `exp(-p*g(x))` with `g > 0` (`_decays`).
fn decays(f: &Basic, x: &Basic, p: &Basic) -> PolyResult<bool> {
    for node in f.atoms(Some(&|n: &Basic| n.is_function(Function::Exp))) {
        let (coefficient, rest) = crate::core::expr::as_independent(&node.args()[0], p, false)?;
        if rest == *p && is(&coefficient, Predicate::negative) == Some(true) {
            return Ok(true);
        }
        if rest == *p && is(&crate::core::mul::neg(&coefficient)?, Predicate::positive) == Some(true) {
            return Ok(true);
        }
        let px: Basic = crate::core::mul::mul(&[p.clone(), x.clone()])?;
        if rest == px || rest == crate::core::mul::neg(&px)? {
            let ratio: Basic = crate::core::mul::div(&crate::core::mul::div(&rest, p)?, x)?;
            return Ok(is(&crate::core::mul::mul(&[coefficient, ratio])?, Predicate::negative) == Some(true));
        }
    }
    Ok(false)
}

/// `primitive(p) - primitive(anchor)`, the latter as a limit
/// (`_difference`).
fn difference(
    primitive: &Basic,
    q: &Basic,
    p: &Basic,
    anchor: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Basic>> {
    let at_p: Basic = subs(primitive, q, p)?;
    let at_anchor: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::assumptions::limits::limit(primitive, q, anchor, "+", assumptions, None)
    })?;
    let Some(at_anchor) = at_anchor else {
        return Ok(None);
    };
    if has_head(&at_anchor, &|h| {
        matches!(
            h,
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity | Head::Limit
        )
    }) {
        return Ok(None);
    }
    Ok(Some(crate::core::add::sub(&at_p, &at_anchor)?))
}
