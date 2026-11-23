//! Port of `sympy_extras/integrals/summability.py`: the Abel, Cesàro and
//! Gaussian means of a divergent oscillatory integral over an infinite
//! range: the integrand times the kernel `exp(-eps*x)`, `exp(-eps*x**2)` or
//! `(1 - x/R)**k` integrated by the definite integrator, and the limit as
//! the regulator tends to `0+` (to `oo`); `None` unless the limit is a
//! finite number free of the regulator. A convergent integral is its own
//! mean (the consistency theorems).

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;

type R = PolyResult<Basic>;
type OptCV = PolyResult<Option<ConditionalValue>>;

/// The summability methods (`METHODS`).
pub const METHODS: [&str; 3] = ["abel", "cesaro", "gaussian"];

/// The Cesàro orders tried in turn (`_CESARO_ORDERS`).
const CESARO_ORDERS: [i64; 3] = [1, 2, 3];

fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn neg_oo() -> Basic {
    Basic::atom(Head::NegativeInfinity)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn fresh(name: &str, positive: bool) -> Basic {
    if positive {
        crate::integrals::definite::fresh(name, &[(Predicate::positive, true)])
    } else {
        crate::integrals::definite::fresh(name, &[])
    }
}
fn has_head(e: &Basic, f: &dyn Fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| f(n.head()))
}

fn checked(method: &str) -> PolyResult<()> {
    if !METHODS.contains(&method) {
        return Err(PolyError::Value(format!(
            "method must be one of {}, got '{}'",
            METHODS.join(", "),
            method
        )));
    }
    Ok(())
}

/// `conditional_integral` within `seconds` (`_integral`).
fn integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic], seconds: Option<f64>) -> OptCV {
    Ok(crate::timeout::attempt_within(seconds, || {
        crate::integrals::definite::conditional_integral(f, x, a, b, assumptions, false, false, false)
    })?
    .flatten())
}

/// The `method` mean of `Integral(f, (x, a, b))` with the condition under
/// which it holds, or `None` (`summable_integral`).
pub fn summable_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, method: &str, assumptions: &[Basic]) -> OptCV {
    checked(method)?;
    if a == b {
        return Ok(Some(ConditionalValue::new(Basic::zero(), crate::core::logic::true_())));
    }
    if *a == oo() || *b == neg_oo() {
        return match summable_integral(f, x, b, a, method, assumptions)? {
            None => Ok(None),
            Some(found) => Ok(Some(found.scaled(&Basic::negative_one())?)),
        };
    }
    if *a == neg_oo() && *b == oo() {
        let Some(right) = oscillatory_tail(f, x, &Basic::zero(), method, assumptions)? else {
            return Ok(None);
        };
        let mirrored: Basic = subs(f, x, &crate::core::mul::neg(x)?)?;
        return match oscillatory_tail(&mirrored, x, &Basic::zero(), method, assumptions)? {
            None => Ok(None),
            Some(left) => Ok(Some(left.add(&right)?)),
        };
    }
    if *a == neg_oo() {
        // x = b - t maps (-oo, b) onto (0, oo)
        let t: Basic = fresh("t", false);
        let shifted: Basic = subs(f, x, &crate::core::add::sub(b, &t)?)?;
        return oscillatory_tail(&shifted, &t, &Basic::zero(), method, assumptions);
    }
    if *b == oo() {
        return oscillatory_tail(f, x, a, method, assumptions);
    }
    integral(f, x, a, b, assumptions, None)
}

fn infinite_or_bad(e: &Basic) -> bool {
    has_head(e, &|h| {
        matches!(
            h,
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity | Head::AccumBounds
        )
    })
}

/// The `method` mean of the tail `Integral(f, (x, a, oo))`
/// (`oscillatory_tail`).
pub fn oscillatory_tail(f: &Basic, x: &Basic, a: &Basic, method: &str, assumptions: &[Basic]) -> OptCV {
    checked(method)?;
    let t: Basic = fresh("t", true);
    let g: Basic = if *a != Basic::zero() {
        subs(f, x, &crate::core::add::add(&[t.clone(), a.clone()])?)?
    } else {
        subs(f, x, &t)?
    };
    // the consistency theorem: a convergent integral is its own mean
    let timeout: Option<f64> = crate::settings::settings().timeout;
    if let Some(found) = integral(&g, &t, &Basic::zero(), &oo(), assumptions, timeout)? {
        if !infinite_or_bad(&found.value) {
            return Ok(Some(found));
        }
    }
    if method == "cesaro" {
        for order in CESARO_ORDERS {
            if let Some(found) = cesaro_mean(&g, &t, order, assumptions)? {
                return Ok(Some(found));
            }
        }
        return Ok(None);
    }
    let eps: Basic = fresh("epsilon", true);
    let power: Basic = if method == "abel" {
        t.clone()
    } else {
        crate::core::power::pow(&t, &Basic::int(2))?
    };
    let kernel: Basic =
        crate::functions::elementary::exponential::exp(&crate::core::mul::neg(&mul(&[eps.clone(), power])?)?)?;
    regularised_limit(
        &mul(&[g, kernel])?,
        &t,
        &Basic::zero(),
        &oo(),
        &eps,
        &Basic::zero(),
        "+",
        assumptions,
    )
}

/// The `(C, order)` mean of `Integral(g, (t, 0, oo))` (`_cesaro_mean`).
fn cesaro_mean(g: &Basic, t: &Basic, order: i64, assumptions: &[Basic]) -> OptCV {
    let r: Basic = fresh("R", true);
    let kernel: Basic = crate::core::power::pow(
        &crate::core::add::sub(&Basic::one(), &crate::core::mul::div(t, &r)?)?,
        &Basic::int(order),
    )?;
    regularised_limit(
        &mul(&[g.clone(), kernel])?,
        t,
        &Basic::zero(),
        &r,
        &r,
        &oo(),
        "-",
        assumptions,
    )
}

/// `Integral(h, (t, lower, upper))` under the assumptions and `regulator
/// > 0`, then its limit as the regulator tends to `point`
/// (`_regularised_limit`).
#[allow(clippy::too_many_arguments)]
fn regularised_limit(
    h: &Basic,
    t: &Basic,
    lower: &Basic,
    upper: &Basic,
    regulator: &Basic,
    point: &Basic,
    direction: &str,
    assumptions: &[Basic],
) -> OptCV {
    let mut facts: Vec<Basic> = assumptions.to_vec();
    facts.push(crate::core::relational::gt(regulator, &Basic::zero())?);
    let timeout: Option<f64> = crate::settings::settings().timeout;
    let Some(inner) = integral(h, t, lower, upper, &facts, timeout)? else {
        return Ok(None);
    };
    if has_head(&inner.value, &|h| {
        matches!(h, Head::NaN | Head::ComplexInfinity | Head::AccumBounds)
    }) {
        return Ok(None);
    }
    let Some(condition) = crate::integrals::conditions::decide(&inner.condition, &facts)? else {
        return Ok(None);
    };
    if condition.has(regulator) {
        return Ok(None);
    }
    let value: Option<Basic> = crate::timeout::attempt_within(timeout, || {
        crate::assumptions::limits::limit(&inner.value, regulator, point, direction, &facts, None)
    })?;
    let Some(value) = value else {
        return Ok(None);
    };
    if value.has(regulator) || infinite_or_bad(&value) || has_head(&value, &|h| matches!(h, Head::Limit)) {
        return Ok(None);
    }
    if crate::core::expr::is(&value, Predicate::finite) == Some(false) {
        return Ok(None);
    }
    Ok(Some(ConditionalValue::new(value, without(&condition, regulator))))
}

/// The conjuncts of `condition` free of the regulator (`_without`).
fn without(condition: &Basic, regulator: &Basic) -> Basic {
    if *condition == crate::core::logic::true_() {
        return crate::core::logic::true_();
    }
    let parts: Vec<Basic> = if matches!(condition.head(), Head::And) {
        condition.args().to_vec()
    } else {
        vec![condition.clone()]
    };
    crate::core::logic::and_(&parts.into_iter().filter(|p| !p.has(regulator)).collect::<Vec<_>>())
}
