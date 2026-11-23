//! Port of `sympy_extras/integrals/laplace.py`: the operational rules of
//! the Laplace transform for `Integral(f(t)*exp(-s*t), (t, 0, oo))`: the
//! division by `t**n` (`n` outer integrals from `s` to infinity), the
//! multiplication by `t**n` (`(-1)**n` times the `n`-th derivative), the
//! convolution (and the integral rule as its case `q = 1`), the shift by a
//! step function, the exponential shift and the periodic rule. The
//! transforms of the pieces are computed by the definite integrator.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::integrals::conditions::ConditionalValue;

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;
type OptCV = PolyResult<Option<ConditionalValue>>;

/// The bound of the powers of `t` in the division and multiplication
/// rules (`_MAX_POWER`).
const MAX_POWER: i64 = 3;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn ask(q: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    crate::assumptions::ask::ask(q, assumptions, None)
}
fn budget() -> Option<f64> {
    crate::settings::settings().timeout
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn factors(f: &Basic) -> Vec<Basic> {
    if f.is_mul() {
        f.args().to_vec()
    } else {
        vec![f.clone()]
    }
}
fn has_function(e: &Basic, fs: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(f) if fs.contains(f)))
}
fn monomial(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    crate::integrals::mellin::monomial(e, x)
}

/// A dummy with a name of its own (`_fresh`).
fn fresh(name: &str, positive: bool) -> Basic {
    if positive {
        crate::integrals::definite::fresh(name, &[(Predicate::positive, true)])
    } else {
        crate::integrals::definite::fresh(name, &[])
    }
}

/// Whether `e` is an unevaluated `Integral` (`_is_integral`).
fn is_integral(e: &Basic) -> bool {
    matches!(e.head(), Head::Integral | Head::NonElementaryIntegral)
}

/// Whether `e` has an unevaluated `Integral` (`has_integral`).
pub fn has_integral(e: &Basic) -> bool {
    e.preorder().iter().any(is_integral)
}

/// The assumptions with the transform variable `s` renamed `u`
/// (`_renamed`).
fn renamed(assumptions: &[Basic], s: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(s.clone(), u.clone());
    let mut items: Vec<Basic> = Vec::new();
    for item in assumptions {
        items.push(crate::core::function::xreplace_evaluated(item, &rule)?);
    }
    items.push(gt(u, &Basic::zero())?);
    Ok(items)
}

/// `Integral(f*exp(-u*t), (t, 0, oo))` by the definite integrator, `None`
/// when it is not found or comes with an undecided condition
/// (`transform`).
pub fn transform(f: &Basic, t: &Basic, u: &Basic, assumptions: &[Basic]) -> R2 {
    let integrand: Basic = mul(&[f.clone(), exp(&neg(&mul(&[u.clone(), t.clone()])?)?)?])?;
    let options = crate::integrals::definite::Options {
        special_values: false,
        ..crate::integrals::definite::Options::default()
    };
    let value: Option<Basic> = crate::timeout::attempt_within(budget(), || {
        crate::integrals::definite::definite_integral(&integrand, t, &Basic::zero(), &oo(), assumptions, &options)
    })?;
    Ok(value.filter(|v| !v.preorder().iter().any(|n| matches!(n.head(), Head::Piecewise)) && !has_integral(v)))
}

/// `conditional_integral`, `None` for a value with an unevaluated
/// `Integral` (`_conditional_integral`).
fn conditional_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    let found = crate::integrals::definite::conditional_integral(f, x, a, b, assumptions, false, false, false)?;
    Ok(found.filter(|c| !has_integral(&c.value)))
}

/// `(n, h)` with `f == t**n*h` (`_power_of`).
fn power_of(f: &Basic, t: &Basic) -> PolyResult<(i64, Basic)> {
    let mut n: i64 = 0;
    let mut rest: Vec<Basic> = Vec::new();
    for g in factors(f) {
        if g == *t {
            n += 1;
        } else if g.is_pow() && g.args()[0] == *t && integer_of(&g.args()[1]).is_some() {
            n += integer_of(&g.args()[1]).unwrap_or(0);
        } else {
            rest.push(g);
        }
    }
    Ok((n, mul(&rest)?))
}

/// `L[h/t**n](s)`, `n` outer integrals (`_division_rule`).
fn division_rule(f: &Basic, t: &Basic, s: &Basic, assumptions: &[Basic]) -> OptCV {
    let (n, h) = power_of(f, t)?;
    if n >= 0 || -n > MAX_POWER {
        return Ok(None);
    }
    let u: Basic = fresh("u", true);
    let Some(mut value) = transform(&h, t, &u, &renamed(assumptions, s, &u)?)? else {
        return Ok(None);
    };
    let mut condition: Basic = crate::core::logic::true_();
    for _ in 0..-n {
        let w: Basic = fresh("w", true);
        let Some(outer) = conditional_integral(&value, &u, &w, &oo(), &renamed(assumptions, s, &w)?)? else {
            return Ok(None);
        };
        value = real_arguments(&subs(&outer.value, &w, &u)?, &renamed(assumptions, s, &u)?)?;
        condition = crate::core::logic::and_(&[condition, subs(&outer.condition, &w, s)?]);
    }
    Ok(Some(ConditionalValue::new(subs(&value, &u, s)?, condition)))
}

/// `L[t**n*h](s) = (-1)**n*H^(n)(s)` (`_multiplication_rule`).
fn multiplication_rule(f: &Basic, t: &Basic, s: &Basic, assumptions: &[Basic]) -> OptCV {
    let (n, h) = power_of(f, t)?;
    if n <= 0 || n > MAX_POWER {
        return Ok(None);
    }
    let u: Basic = fresh("u", true);
    let Some(value) = transform(&h, t, &u, &renamed(assumptions, s, &u)?)? else {
        return Ok(None);
    };
    let derivative: Basic = mul(&[
        crate::core::function::diff(&value, &[(u.clone(), n as u64)])?,
        crate::core::power::pow(&int(-1), &int(n))?,
    ])?;
    Ok(Some(ConditionalValue::new(
        subs(&derivative, &u, s)?,
        crate::core::logic::true_(),
    )))
}

/// `L[h(t)*theta(t - a)](s) = exp(-a*s)*L[h(t + a)](s)` for `a > 0`
/// (`_heaviside_rule`).
fn heaviside_rule(f: &Basic, t: &Basic, s: &Basic, assumptions: &[Basic]) -> OptCV {
    for g in factors(f) {
        if !(g.is_function(Function::Heaviside) && g.has(t)) {
            continue;
        }
        let argument: &Basic = &g.args()[0];
        if monomial(argument, t)?.is_some() {
            return Ok(None);
        }
        let (shift, rest) = crate::core::expr::as_independent(argument, t, true)?;
        if rest != *t {
            return Ok(None);
        }
        let a: Basic = neg(&shift)?;
        if ask(&gt(&a, &Basic::zero())?, assumptions)? != Some(true) {
            return Ok(None);
        }
        let h: Basic = subs(&div(f, &g)?, t, &add(&[t.clone(), a.clone()])?)?;
        let u: Basic = fresh("u", true);
        let Some(value) = transform(&h, t, &u, &renamed(assumptions, s, &u)?)? else {
            return Ok(None);
        };
        return Ok(Some(ConditionalValue::new(
            mul(&[exp(&neg(&mul(&[a, s.clone()])?)?)?, subs(&value, &u, s)?])?,
            crate::core::logic::true_(),
        )));
    }
    Ok(None)
}

/// `L[exp(c*t)*h](s) = H(s - c)` for `Re s > c` (`_exponential_rule`).
fn exponential_rule(f: &Basic, t: &Basic, s: &Basic, assumptions: &[Basic]) -> OptCV {
    for g in factors(f) {
        if !(g.is_function(Function::Exp) && g.has(t)) {
            continue;
        }
        let Some((c, power)) = monomial(&g.args()[0], t)? else {
            return Ok(None);
        };
        if power != Basic::one() {
            return Ok(None);
        }
        let u: Basic = fresh("u", true);
        let Some(value) = transform(&div(f, &g)?, t, &u, &renamed(assumptions, s, &u)?)? else {
            return Ok(None);
        };
        let shifted: Basic = sub(s, &c)?;
        return Ok(Some(ConditionalValue::new(
            subs(&value, &u, &shifted)?,
            gt(&shifted, &Basic::zero())?,
        )));
    }
    Ok(None)
}

/// `L[f](s) = Integral(f*exp(-s*t), (t, 0, T))/(1 - exp(-s*T))` for a
/// period `T` (`_periodic_rule`).
fn periodic_rule(f: &Basic, t: &Basic, s: &Basic, assumptions: &[Basic]) -> OptCV {
    if has_function(f, &[Function::Exp, Function::Heaviside]) || has_integral(f) {
        return Ok(None);
    }
    let found: Option<Option<Basic>> =
        crate::timeout::attempt_within(budget(), || crate::calculus::util::periodicity(f, t, false))?;
    let Some(found) = found.flatten() else {
        return Ok(None);
    };
    if found == Basic::zero() || found.has(t) {
        return Ok(None);
    }
    let mut period: Basic = found;
    // 2*pi/Abs(k) for sin(k*t): the sign of k is in the assumptions
    for node in period.atoms(Some(&|n: &Basic| n.is_function(Function::Abs))) {
        let argument: Basic = node.args()[0].clone();
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        if ask(&gt(&argument, &Basic::zero())?, assumptions)? == Some(true) {
            rule.insert(node.clone(), argument);
        } else if ask(&crate::core::relational::lt(&argument, &Basic::zero())?, assumptions)? == Some(true) {
            rule.insert(node.clone(), neg(&argument)?);
        }
        if !rule.is_empty() {
            period = crate::core::function::xreplace_evaluated(&period, &rule)?;
        }
    }
    if has_function(&period, &[Function::Abs]) {
        return Ok(None);
    }
    let u: Basic = fresh("u", true);
    let decay: Basic = exp(&neg(&mul(&[u.clone(), t.clone()])?)?)?;
    let Some(piece) = conditional_integral(
        &mul(&[f.clone(), decay])?,
        t,
        &Basic::zero(),
        &period,
        &renamed(assumptions, s, &u)?,
    )?
    else {
        return Ok(None);
    };
    let value: Basic = div(
        &piece.value,
        &sub(&Basic::one(), &exp(&neg(&mul(&[u.clone(), period])?)?)?)?,
    )?;
    Ok(Some(ConditionalValue::new(
        subs(&value, &u, s)?,
        subs(&piece.condition, &u, s)?,
    )))
}

/// `L[Integral(p(u)*q(t - u), (u, 0, t))](s) = P(s)*Q(s)`
/// (`laplace_of_convolution`).
pub fn laplace_of_convolution(f: &Basic, t: &Basic, s: &Basic, assumptions: &[Basic]) -> OptCV {
    if !is_integral(f) || f.args().len() != 2 {
        return Ok(None);
    }
    let (integrand, limits) = (&f.args()[0], &f.args()[1]);
    if !matches!(limits.head(), Head::Tuple) || limits.args().len() != 3 {
        return Ok(None);
    }
    let (variable, lower, upper) = (&limits.args()[0], &limits.args()[1], &limits.args()[2]);
    if !variable.is_symbol() || *lower != Basic::zero() || upper != t {
        return Ok(None);
    }
    let mut p: Basic = Basic::one();
    let mut q: Basic = Basic::one();
    let w: Basic = fresh("w", false);
    for g in factors(integrand) {
        if !g.has(t) {
            p = mul(&[p, g])?;
            continue;
        }
        let shifted: Basic = subs(&g, variable, &sub(t, &w)?)?;
        if shifted.has(t) {
            return Ok(None);
        }
        q = mul(&[q, subs(&shifted, &w, variable)?])?;
    }
    let v: Basic = fresh("v", true);
    let given: Vec<Basic> = renamed(assumptions, s, &v)?;
    let first: Option<Basic> = transform(&p, variable, &v, &given)?;
    let second: Option<Basic> = transform(&q, variable, &v, &given)?;
    let (Some(first), Some(second)) = (first, second) else {
        return Ok(None);
    };
    Ok(Some(ConditionalValue::new(
        subs(&mul(&[first, second])?, &v, s)?,
        crate::core::logic::true_(),
    )))
}

/// `arg(x + I*y)` with the signs of `x` and `y` decided under the
/// assumptions written as `atan(y/x)` (`x > 0`) or `atan(y/x) +- pi` (`x <
/// 0`), then simplified (`real_arguments`).
pub fn real_arguments(value: &Basic, assumptions: &[Basic]) -> R {
    use crate::functions::elementary::complexes::{im, re};
    use crate::functions::elementary::inverse::atan;
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    for node in value.atoms(Some(&|n: &Basic| n.is_function(Function::Arg))) {
        let z: &Basic = &node.args()[0];
        let (x, y) = (re(z)?, im(z)?);
        if has_function(&x, &[Function::Re, Function::Im]) || has_function(&y, &[Function::Re, Function::Im]) {
            continue;
        }
        let angle: Basic = atan(&div(&y, &x)?)?;
        if ask(&gt(&x, &Basic::zero())?, assumptions)? == Some(true) {
            replacement.insert(node.clone(), angle);
        } else if ask(&crate::core::relational::lt(&x, &Basic::zero())?, assumptions)? == Some(true) {
            if ask(&crate::core::relational::ge(&y, &Basic::zero())?, assumptions)? == Some(true) {
                replacement.insert(node.clone(), add(&[angle, Basic::pi()])?);
            } else if ask(&crate::core::relational::lt(&y, &Basic::zero())?, assumptions)? == Some(true) {
                replacement.insert(node.clone(), sub(&angle, &Basic::pi())?);
            }
        }
    }
    if replacement.is_empty() {
        return Ok(value.clone());
    }
    let rewritten: Basic = crate::core::function::xreplace_evaluated(value, &replacement)?;
    let simpler: Option<Basic> = crate::timeout::attempt_within(budget(), || {
        crate::simplify::simplify::simplify(&rewritten, &crate::simplify::simplify::Options::default())
    })?;
    Ok(simpler.unwrap_or(rewritten))
}

/// A rule of the module (`Rule`).
type Rule = fn(&Basic, &Basic, &Basic, &[Basic]) -> OptCV;

/// The rules in the order they are tried (`_RULES`).
const RULES: [Rule; 6] = [
    laplace_of_convolution,
    heaviside_rule,
    exponential_rule,
    periodic_rule,
    division_rule,
    multiplication_rule,
];

/// `Integral(f*exp(-s*t), (t, 0, oo))` by the first applicable rule
/// (`laplace_rules`).
pub fn laplace_rules(f: &Basic, t: &Basic, s: &Basic, assumptions: &[Basic]) -> OptCV {
    if f.is_add() {
        let mut total: ConditionalValue = ConditionalValue::new(Basic::zero(), crate::core::logic::true_());
        for term in f.args() {
            let Some(found) = laplace_rules(term, t, s, assumptions)? else {
                return Ok(None);
            };
            total = total.add(&found)?;
        }
        return Ok(Some(total));
    }
    for rule in RULES {
        if let Some(found) = rule(f, t, s, assumptions)? {
            let value: Basic = crate::integrals::marichev::tidy(
                &real_arguments(&found.value, assumptions)?,
                assumptions,
                &found.condition,
            )?;
            return Ok(Some(ConditionalValue::new(value, found.condition)));
        }
    }
    Ok(None)
}

/// `Integral(f, (t, a, b))` when it is a Laplace transform
/// `Integral(g(t)*exp(-s*t), (t, 0, oo))` with `s` a symbol or a positive
/// number, by [`laplace_rules`] on `g` (`laplace_integral`).
pub fn laplace_integral(f: &Basic, t: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    if *a != Basic::zero() || *b != oo() {
        return Ok(None);
    }
    for g in factors(f) {
        if !(g.is_function(Function::Exp) && g.has(t)) {
            continue;
        }
        let Some((c, power)) = monomial(&g.args()[0], t)? else {
            continue;
        };
        if power != Basic::one() {
            continue;
        }
        let s: Basic = neg(&c)?;
        let positive: bool = is(&s, Predicate::positive) == Some(true);
        if !(s.is_symbol() || positive) {
            continue;
        }
        if s.is_symbol() && !positive && ask(&gt(&s, &Basic::zero())?, assumptions)? != Some(true) {
            continue;
        }
        let rest: Basic = div(f, &g)?;
        if rest == Basic::one() || !rest.has(t) {
            continue;
        }
        return laplace_rules(&rest, t, &s, assumptions);
    }
    Ok(None)
}
