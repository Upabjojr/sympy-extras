//! Port of `sympy_extras/integrals/asymptotic.py`: asymptotic expansions
//! of parametric definite integrals for a large parameter — Watson's
//! lemma, Laplace's method (maxima of any even order, endpoint maxima),
//! the saddle points in the complex plane (confirmed by a quadrature),
//! the method of stationary phase and the uniform Airy and error function
//! forms for coalescing stationary points.
//!
//! The quadratures of the saddle check evaluate the integrand by
//! `numeric_value` with its variable bound to the node
//! ([`crate::integrals::evalf::value_at`]), with mpmath's tanh-sinh rule
//! ([`crate::integrals::quadrature::quad`]). Python's binary operators
//! evaluate at each step: the expressions are built with the binary
//! helpers in the same association.

#![allow(clippy::mutable_key_type)]
#![allow(clippy::too_many_arguments)]

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{rat, Int, Rat};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    Basic::zero()
}
fn one() -> Basic {
    Basic::one()
}
fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn neg_oo() -> Basic {
    Basic::atom(Head::NegativeInfinity)
}
fn imag() -> Basic {
    Basic::atom(Head::ImaginaryUnit)
}
fn pi() -> Basic {
    Basic::pi()
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn p2(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn ng(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn sqrt(a: &Basic) -> R {
    p2(a, &Basic::half())
}
fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn apply(f: Function, args: &[Basic]) -> R {
    crate::functions::apply_function(f, args)
}
fn gamma(a: &Basic) -> R {
    apply(Function::Gamma, std::slice::from_ref(a))
}
fn factorial(n: i64) -> R {
    apply(Function::Factorial, &[int(n)])
}
fn re(a: &Basic) -> R {
    crate::functions::elementary::complexes::re(a)
}
fn im(a: &Basic) -> R {
    crate::functions::elementary::complexes::im(a)
}
fn abs_(a: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(a)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn diff_n(e: &Basic, x: &Basic, n: u64) -> R {
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_method(e)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn ask(query: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    crate::assumptions::ask::ask(query, assumptions, None)
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
fn ge(a: &Basic, b: &Basic) -> R {
    crate::core::relational::ge(a, b)
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(crate::settings::settings().timeout, f)
}
fn infinite(e: &Basic) -> bool {
    *e == oo() || *e == neg_oo()
}
fn has_any(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn has_bad(e: &Basic) -> bool {
    has_any(e, &|n| {
        matches!(
            n.head(),
            Head::ComplexInfinity | Head::NaN | Head::Infinity | Head::NegativeInfinity
        )
    })
}
fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn terms(e: &Basic) -> Vec<Basic> {
    crate::core::expr::terms_of(e)
}
fn independent(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_independent(e, x, false)
}
fn real_value(e: &Basic) -> f64 {
    crate::core::evalf::numeric_value(e, 64)
        .map(|v| crate::core::evalf::parts(&v).0.to_f64().value())
        .unwrap_or(f64::NAN)
}
/// The rational number `p/q` (`_rational`).
fn rational(p: i64, q: i64) -> Basic {
    Basic::from_rational(rat(p, q))
}
/// `O(e)` as `t -> oo` (`order(e, (t, oo))`).
fn order_at_oo(e: &Basic, t: &Basic) -> R {
    crate::core::order::order(e, &[(t.clone(), oo())])
}
fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}

// ---------------------------------------------------------------------
// recognising the shapes

/// The exponent `a` of `rest == x**a`; `None` for another form
/// (`_exponent_of`).
fn exponent_of(rest: &Basic, x: &Basic) -> Option<Basic> {
    if *rest == one() {
        return Some(zero());
    }
    if rest == x {
        return Some(one());
    }
    if rest.is_pow() && rest.args()[0] == *x {
        return Some(rest.args()[1].clone());
    }
    None
}

/// The nonzero coefficients of `e` as a polynomial in `x` with expressions
/// as coefficients (`_coefficients`), by degree.
fn coefficients(e: &Basic, x: &Basic) -> PolyResult<Option<Vec<(i64, Basic)>>> {
    let mut found: Vec<(i64, Basic)> = Vec::new();
    for term in terms(&expand(e)?) {
        if term == zero() {
            continue;
        }
        let (coefficient, rest) = independent(&term, x)?;
        let Some(exponent) = exponent_of(&rest, x) else {
            return Ok(None);
        };
        let Some(k) = exponent
            .as_rational()
            .filter(|q| q.is_int() && **q >= Rat::from(Int::ZERO))
            .and_then(|q| i64::try_from(crate::numbers::numer(q)).ok())
        else {
            return Ok(None);
        };
        if coefficient.has(x) {
            return Ok(None);
        }
        match found.iter_mut().find(|(d, _)| *d == k) {
            Some(slot) => slot.1 = a2(&slot.1, &coefficient)?,
            None => found.push((k, coefficient)),
        }
    }
    Ok(Some(found.into_iter().filter(|(_, c)| *c != zero()).collect()))
}

fn degree(coefficients: &[(i64, Basic)]) -> i64 {
    coefficients.iter().map(|(k, _)| *k).max().unwrap_or(-1)
}

fn coefficient_of(coefficients: &[(i64, Basic)], k: i64) -> Basic {
    coefficients
        .iter()
        .find(|(d, _)| *d == k)
        .map(|(_, c)| c.clone())
        .unwrap_or_else(zero)
}

/// `(phi, h)` with `f == phi(x) * exp(t*h(x))`, `phi` and `h` free of `t`
/// (`exponential_part`).
pub fn exponential_part(f: &Basic, _x: &Basic, t: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let rewritten: Basic = crate::core::function::replace_where(f, &|e| e.is_pow() && e.args()[1].has(t), &|e| {
        exp(&m2(&e.args()[1], &log(&e.args()[0])?)?)
    })?;
    let mut phi: Basic = one();
    let mut argument: Basic = zero();
    for factor in mul_args(&rewritten) {
        if factor.is_function(Function::Exp) && factor.has(t) {
            argument = a2(&argument, &factor.args()[0])?;
        } else if factor.has(t) {
            return Ok(None);
        } else {
            phi = m2(&phi, &factor)?;
        }
    }
    if argument == zero() {
        return Ok(None);
    }
    let Some(polynomial) = coefficients(&argument, t)? else {
        return Ok(None);
    };
    if degree(&polynomial) != 1 {
        return Ok(None);
    }
    let h: Basic = coefficient_of(&polynomial, 1);
    let rest: Basic = coefficient_of(&polynomial, 0);
    if h.has(t) || rest.has(t) {
        return Ok(None);
    }
    Ok(Some((m2(&phi, &exp(&rest)?)?, h)))
}

/// `(phi, h, kind)` with `f == phi(x) * K(t*h(x))` for `K` one of
/// `exp(I*...)`, `cos` or `sin` (`oscillatory_part`).
pub fn oscillatory_part(f: &Basic, x: &Basic, t: &Basic) -> PolyResult<Option<(Basic, Basic, &'static str)>> {
    let _ = x;
    let mut phi: Basic = one();
    let mut found: Option<(Basic, &'static str)> = None;
    for factor in mul_args(f) {
        if !factor.has(t) {
            phi = m2(&phi, &factor)?;
            continue;
        }
        if found.is_some() {
            return Ok(None);
        }
        let kinds = [Function::Cos, Function::Sin, Function::Exp];
        if kinds.iter().any(|k| factor.is_function(*k)) && factor.args().len() == 1 {
            let mut argument: Basic = factor.args()[0].clone();
            if factor.is_function(Function::Exp) {
                argument = d2(&argument, &imag())?;
            }
            let Some(polynomial) = coefficients(&argument, t)? else {
                return Ok(None);
            };
            if degree(&polynomial) != 1 || polynomial.iter().any(|(k, _)| *k == 0) {
                return Ok(None);
            }
            let h: Basic = coefficient_of(&polynomial, 1);
            if h.has(t) || h.has(&imag()) {
                return Ok(None);
            }
            let kind: &'static str = if factor.is_function(Function::Cos) {
                "cos"
            } else if factor.is_function(Function::Sin) {
                "sin"
            } else {
                "exp"
            };
            found = Some((h, kind));
        } else {
            return Ok(None);
        }
    }
    Ok(found.map(|(h, kind)| (phi, h, kind)))
}

// ---------------------------------------------------------------------
// Watson's lemma

/// The first `count` terms `(c_k, a_k)` of the expansion of `phi` at `0`
/// and the exponent of the next one (`_puiseux_terms`).
fn puiseux_terms(phi: &Basic, x: &Basic, count: usize) -> PolyResult<Option<(Vec<(Basic, Basic)>, Option<Basic>)>> {
    if is_polynomial(phi, x) {
        let Some(mut polynomial) = coefficients(phi, x)? else {
            return Ok(None);
        };
        polynomial.sort_by_key(|(k, _)| *k);
        return Ok(Some((polynomial.into_iter().map(|(k, c)| (c, int(k))).collect(), None)));
    }
    let mut n: i64 = count as i64 + 2;
    for _ in 0..6 {
        let depth: i64 = n;
        let Some(expansion) =
            attempt(|| crate::core::series::series(phi, Some(x), &zero(), depth, "+", None, &zero()))?
        else {
            return Ok(None);
        };
        let found_order: Option<Basic> = crate::core::order::get_o(&expansion)?;
        let body: Basic = crate::core::order::remove_o(&expansion)?;
        let mut found: Vec<(Basic, Basic)> = Vec::new();
        for term in terms(&body) {
            let (coefficient, rest) = independent(&term, x)?;
            let Some(exponent) = exponent_of(&rest, x) else {
                return Ok(None);
            };
            if coefficient != zero() {
                found.push((coefficient, exponent));
            }
        }
        found.sort_by(|a, b| {
            real_value(&a.1)
                .partial_cmp(&real_value(&b.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let Some(o) = found_order else {
            return Ok(Some((found, None)));
        };
        let inner: Basic = independent(&o.args()[0], x)?.1;
        if exponent_of(&inner, x).is_none() {
            return Ok(None);
        }
        if found.len() > count {
            let next: Basic = found[count].1.clone();
            found.truncate(count);
            return Ok(Some((found, Some(next))));
        }
        n += count as i64 + 2;
    }
    Ok(None)
}

/// The expansion of `Integral(exp(-t*x)*phi(x), (x, 0, oo))` for
/// `t -> oo` by Watson's lemma, with `order` terms (`watson_lemma`).
pub fn watson_lemma(phi: &Basic, x: &Basic, t: &Basic, order: usize) -> OptR {
    let Some((found, next)) = puiseux_terms(phi, x, order)? else {
        return Ok(None);
    };
    let mut total: Basic = zero();
    for (coefficient, exponent) in &found {
        let e1: Basic = a2(exponent, &one())?;
        if !yes(&e1, Predicate::positive) {
            return Ok(None);
        }
        total = a2(&total, &d2(&m2(coefficient, &gamma(&e1)?)?, &p2(t, &e1)?)?)?;
    }
    let Some(next) = next else {
        return Ok(Some(total));
    };
    let power: Basic = s2(&ng(&next)?, &one())?;
    Ok(Some(a2(&total, &order_at_oo(&p2(t, &power)?, t)?)?))
}

// ---------------------------------------------------------------------
// Laplace's method

fn points_in(points: &[Basic], a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let mut inside: Vec<Basic> = Vec::new();
    for p in points {
        if is(p, Predicate::extended_real) == Some(false) {
            continue;
        }
        let below: Option<bool> = if *a == neg_oo() {
            Some(true)
        } else {
            ask(&gt(p, a)?, assumptions)?
        };
        let above: Option<bool> = if *b == oo() {
            Some(true)
        } else {
            ask(&lt(p, b)?, assumptions)?
        };
        let (Some(below), Some(above)) = (below, above) else {
            return Ok(None);
        };
        if below && above {
            inside.push(p.clone());
        }
    }
    Ok(Some(inside))
}

/// The zeros of `h'` inside `(a, b)` (`_stationary_points`).
fn stationary_points(
    h: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Vec<Basic>>> {
    let mut facts: Vec<Basic> = Vec::new();
    if *a != neg_oo() {
        facts.push(gt(x, a)?);
    }
    if *b != oo() {
        facts.push(lt(x, b)?);
    }
    facts.extend(assumptions.iter().cloned());
    let derivative: Basic = diff_n(h, x, 1)?;
    let reals: Basic = crate::sets::sets::reals();
    let zeros = attempt(|| {
        crate::assumptions::solve::solve(
            &[derivative],
            Some(std::slice::from_ref(x)),
            &facts,
            Some(&reals),
            false,
        )
    })?;
    let Some(zeros) = zeros else {
        return Ok(None);
    };
    if zeros == crate::sets::sets::empty_set() {
        return Ok(Some(Vec::new()));
    }
    if !matches!(zeros.head(), Head::FiniteSet) {
        return Ok(None);
    }
    points_in(zeros.args(), a, b, assumptions)
}

/// `(n, sign)` with `h^(n)(x0)` the first derivative of order at least
/// `start` which is not zero (`_first_nonzero_derivative`).
fn first_nonzero_derivative(
    h: &Basic,
    x: &Basic,
    x0: &Basic,
    start: u64,
    assumptions: &[Basic],
) -> PolyResult<Option<(i64, i64)>> {
    for n in start..9 {
        let value: Basic = simplify(&subs(&diff_n(h, x, n)?, x, x0)?)?;
        if value == zero() {
            continue;
        }
        if ask(&lt(&value, &zero())?, assumptions)? == Some(true) {
            return Ok(Some((n as i64, -1)));
        }
        if ask(&gt(&value, &zero())?, assumptions)? == Some(true) {
            return Ok(Some((n as i64, 1)));
        }
        return Ok(None);
    }
    Ok(None)
}

/// The series of the Laplace integral at a maximum of order `n` at `x0`
/// (`_power_expansion`).
fn power_expansion(
    phi: &Basic,
    h: &Basic,
    x: &Basic,
    x0: &Basic,
    t: &Basic,
    corrections: i64,
    n: i64,
    side: &str,
) -> PolyResult<Option<(Basic, Basic)>> {
    let u: Basic = Dummy::new("u", &[]).0;
    let orientation: Basic = if side == "left" { int(-1) } else { one() };
    let point: Basic = a2(x0, &m2(&orientation, &u)?)?;
    let c: Basic = d2(
        &m2(
            &ng(&p2(&orientation, &int(n))?)?,
            &subs(&diff_n(h, x, n as u64)?, x, x0)?,
        )?,
        &factorial(n)?,
    )?;
    let remainder: Basic = a2(
        &s2(&subs(h, x, &point)?, &subs(h, x, x0)?)?,
        &m2(&c, &p2(&u, &int(n))?)?,
    )?;
    let integrand: Basic = m2(&subs(phi, x, &point)?, &exp(&m2(t, &remainder)?)?)?;
    let kept_steps: i64 = n * corrections;
    let depth: i64 = (n + 1) * kept_steps + 1;
    let Some(expansion) = attempt(|| {
        crate::core::order::remove_o(&crate::core::series::series(
            &integrand,
            Some(&u),
            &zero(),
            depth,
            "+",
            None,
            &zero(),
        )?)
    })?
    else {
        return Ok(None);
    };
    let mut total: Basic = zero();
    for term in terms(&expand(&expansion)?) {
        let (coefficient, rest) = independent(&term, &u)?;
        let Some(k) = exponent_of(&rest, &u)
            .and_then(|e| {
                e.as_rational()
                    .filter(|q| q.is_int() && **q >= Rat::from(Int::ZERO))
                    .cloned()
            })
            .and_then(|q| i64::try_from(crate::numbers::numer(&q)).ok())
        else {
            return Ok(None);
        };
        if side == "both" && k % 2 != 0 {
            continue;
        }
        let r: Basic = rational(k + 1, n);
        let mut moment: Basic = d2(&gamma(&r)?, &m2(&int(n), &p2(&m2(t, &c)?, &r)?)?)?;
        if side == "both" {
            moment = m2(&int(2), &moment)?;
        }
        total = a2(&total, &m2(&coefficient, &moment)?)?;
    }
    // keep the powers of t from the leading t**(-1/n) down kept_steps steps
    let mut kept: Basic = zero();
    for term in terms(&expand(&total)?) {
        let (_, rest) = independent(&term, t)?;
        let Some(power) = exponent_of(&rest, t).and_then(|p| p.as_rational().cloned()) else {
            return Ok(None);
        };
        let step: Rat = -(power * Rat::from(Int::from(n)) + Rat::from(Int::ONE));
        if step >= Rat::from(Int::ZERO) && step <= Rat::from(Int::from(kept_steps)) {
            kept = a2(&kept, &term)?;
        }
    }
    let next_step: i64 = if side == "both" { kept_steps + 2 } else { kept_steps + 1 };
    Ok(Some((simplify(&kept)?, rational(-1 - next_step, n))))
}

fn limit(e: &Basic, x: &Basic, point: &Basic, direction: &str, assumptions: &[Basic]) -> R {
    crate::assumptions::limits::limit(e, x, point, direction, assumptions, None)
}

/// Whether `h` stays below `value` at the infinite ends (`_edges_below`).
fn edges_below(h: &Basic, x: &Basic, value: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    for endpoint in [a, b] {
        if !infinite(endpoint) {
            continue;
        }
        let edge: Option<Basic> = attempt(|| limit(h, x, endpoint, "+", assumptions))?;
        let Some(edge) = edge.filter(|e| !has_any(e, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity)))
        else {
            return Ok(false);
        };
        if edge == neg_oo() {
            continue;
        }
        if edge == oo() || ask(&lt(&edge, value)?, assumptions)? != Some(true) {
            return Ok(false);
        }
    }
    Ok(true)
}

type Candidate = (Basic, i64, &'static str);

/// The local maxima of `h` on the closed range (`_maxima`).
fn maxima(h: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<Candidate>>> {
    let Some(points) = stationary_points(h, x, a, b, assumptions)? else {
        return Ok(None);
    };
    let mut found: Vec<Candidate> = Vec::new();
    for x0 in points {
        let Some((n, sign)) = first_nonzero_derivative(h, x, &x0, 2, assumptions)? else {
            return Ok(None);
        };
        if n % 2 == 0 && sign < 0 {
            found.push((x0, n, "both"));
        }
    }
    for (endpoint, side) in [(a, "right"), (b, "left")] {
        if infinite(endpoint) {
            continue;
        }
        let mut singular: bool = false;
        for n in [1u64, 2] {
            if has_bad(&subs(&diff_n(h, x, n)?, x, endpoint)?) {
                singular = true;
            }
        }
        if singular {
            // a singular endpoint: no maximum when h falls to -oo there
            let direction: &str = if side == "right" { "+" } else { "-" };
            let edge: Option<Basic> = attempt(|| limit(h, x, endpoint, direction, assumptions))?;
            if edge == Some(neg_oo()) {
                continue;
            }
            return Ok(None);
        }
        let Some((n, sign)) = first_nonzero_derivative(h, x, endpoint, 1, assumptions)? else {
            return Ok(None);
        };
        let orientation: i64 = if side == "right" { 1 } else { -1 };
        if orientation.pow(n as u32) * sign < 0 {
            found.push((endpoint.clone(), n, side));
        }
    }
    Ok(Some(found))
}

/// The candidates where `h` is largest (`_global_maxima`).
fn global_maxima(
    candidates: &[Candidate],
    h: &Basic,
    x: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Vec<Candidate>>> {
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut best: Vec<Candidate> = vec![candidates[0].clone()];
    let mut height: Basic = subs(h, x, &candidates[0].0)?;
    for candidate in &candidates[1..] {
        let value: Basic = subs(h, x, &candidate.0)?;
        let difference: Basic = simplify(&s2(&value, &height)?)?;
        if difference == zero() {
            best.push(candidate.clone());
        } else if ask(&gt(&difference, &zero())?, assumptions)? == Some(true) {
            best = vec![candidate.clone()];
            height = value;
        } else if ask(&lt(&difference, &zero())?, assumptions)? != Some(true) {
            return Ok(None);
        }
    }
    Ok(Some(best))
}

/// The expansion of `Integral(phi*exp(t*h), (x, a, b))` for `t -> oo` by
/// Laplace's method, `order` terms (`laplace_method`).
pub fn laplace_method(
    phi: &Basic,
    h: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    order: i64,
    assumptions: &[Basic],
) -> OptR {
    let corrections: i64 = (order - 1).max(0);
    let Some(candidates) = maxima(h, x, a, b, assumptions)? else {
        return Ok(None);
    };
    if candidates.is_empty() {
        if *a == neg_oo() && *b == oo() && stationary_points(h, x, a, b, assumptions)?.is_none_or_empty() {
            return steepest_descent(phi, h, x, t, assumptions);
        }
        return Ok(None);
    }
    let Some(best) = global_maxima(&candidates, h, x, assumptions)? else {
        return Ok(None);
    };
    let height: Basic = subs(h, x, &best[0].0)?;
    if !edges_below(h, x, &height, a, b, assumptions)? {
        return Ok(None);
    }
    let mut total: Basic = zero();
    let mut scale: Option<Basic> = None;
    let growth: Basic = exp(&m2(t, &height)?)?;
    for (x0, n, side) in &best {
        let Some((series, next_power)) = power_expansion(phi, h, x, x0, t, corrections, *n, side)? else {
            return Ok(None);
        };
        if series == zero() {
            return Ok(None);
        }
        total = a2(&total, &m2(&growth, &series)?)?;
        let candidate_scale: Basic = m2(&growth, &p2(t, &next_power)?)?;
        let better: bool = match &scale {
            None => true,
            Some(s) => yes(&s2(&next_power, &power_of(s, t))?, Predicate::positive),
        };
        if better {
            scale = Some(candidate_scale);
        }
    }
    let Some(scale) = scale else {
        return Ok(None);
    };
    Ok(Some(a2(&total, &order_at_oo(&scale, t)?)?))
}

trait EmptyOrNone {
    fn is_none_or_empty(&self) -> bool;
}
impl EmptyOrNone for Option<Vec<Basic>> {
    fn is_none_or_empty(&self) -> bool {
        self.as_ref().map_or(true, |v| v.is_empty())
    }
}

/// The exponent of `t` in `e`, a product with one power of `t`
/// (`_power_of`).
fn power_of(e: &Basic, t: &Basic) -> Basic {
    for factor in mul_args(e) {
        if factor == *t {
            return one();
        }
        if factor.is_pow() && factor.args()[0] == *t {
            return factor.args()[1].clone();
        }
    }
    zero()
}

// ---------------------------------------------------------------------
// saddle points in the complex plane

/// The complex zeros of `h'` (`_saddle_points`).
fn saddle_points(h: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let derivative: Basic = diff_n(h, x, 1)?;
    let found =
        attempt(|| crate::assumptions::solve::solve(&[derivative], Some(std::slice::from_ref(x)), &[], None, false))?;
    let Some(found) = found.filter(|f| matches!(f.head(), Head::FiniteSet) && !f.args().is_empty()) else {
        return Ok(None);
    };
    let mut points: Vec<Basic> = Vec::new();
    for item in found.args() {
        if has_bad(item) {
            return Ok(None);
        }
        points.push(item.clone());
    }
    Ok(Some(points))
}

/// `phi(z0) exp(t h(z0)) sqrt(2 pi/(-t h''(z0)))` (`_leading_saddle_term`).
fn leading_saddle_term(phi: &Basic, h: &Basic, x: &Basic, z0: &Basic, t: &Basic) -> OptR {
    let second: Basic = simplify(&subs(&diff_n(h, x, 2)?, x, z0)?)?;
    if second == zero() || (yes(&second, Predicate::extended_real) && yes(&second, Predicate::positive)) {
        return Ok(None);
    }
    let height: Basic = expand(&subs(h, x, z0)?)?;
    let root: Basic = sqrt(&d2(&m2(&int(2), &pi())?, &m2(&ng(t)?, &second)?)?)?;
    Ok(Some(m2(
        &m2(&expand(&subs(phi, x, z0)?)?, &exp(&m2(t, &height)?)?)?,
        &root,
    )?))
}

/// Whether the sum of the leading `terms` agrees with the quadrature of
/// `Integral(phi exp(t h), (x, -oo, oo))` at two large values of `t`
/// (`_saddle_check`).
fn saddle_check(phi: &Basic, h: &Basic, x: &Basic, t: &Basic, candidate: &[Basic], level: f64) -> PolyResult<bool> {
    use crate::core::evalf::{complex, parts, real_rat, Complex};
    use crate::integrals::quadrature::Point;
    let mut errors: Vec<f64> = Vec::new();
    for value in [30i64, 120] {
        let digits: usize = 25 + (value as f64 * level.abs() / 2.3) as usize;
        let prec: usize = crate::numeric::dps_to_prec(digits);
        let g: Basic = subs(&m2(phi, &exp(&m2(t, h)?)?)?, t, &int(value))?;
        let mut points: Vec<Point> = vec![Point::NegInf];
        for p in -30..=30 {
            points.push(Point::Finite(real_rat(&rat(p, 10), prec)));
        }
        points.push(Point::PosInf);
        let mut f = |z: &crate::core::evalf::Real| -> Option<crate::integrals::quadrature::Value> {
            let v: Complex =
                crate::integrals::evalf::value_at(&g, x, complex(z.clone(), real_rat(&rat(0, 1), prec)), prec)?;
            Some(parts(&v))
        };
        let Some(((re_part, im_part), _)) = crate::integrals::quadrature::quad(&mut f, &points, prec) else {
            return Ok(false);
        };
        let exact: (f64, f64) = (re_part.to_f64().value(), im_part.to_f64().value());
        let mut values: Vec<(f64, f64)> = Vec::new();
        for term in candidate {
            let at: Basic = subs(term, t, &int(value))?;
            let Some(number) = crate::core::evalf::numeric_value(&at, prec) else {
                return Ok(false);
            };
            let (a, b) = parts(&number);
            values.push((a.to_f64().value(), b.to_f64().value()));
        }
        let envelope: f64 = values.iter().map(|(a, b)| (a * a + b * b).sqrt()).sum();
        if envelope == 0.0 || !envelope.is_finite() {
            return Ok(false);
        }
        let total: (f64, f64) = values.iter().fold((0.0, 0.0), |s, v| (s.0 + v.0, s.1 + v.1));
        let error: f64 = ((total.0 - exact.0).powi(2) + (total.1 - exact.1).powi(2)).sqrt() / envelope;
        errors.push(error);
    }
    Ok(errors[1] < 0.02 && errors[1] <= errors[0])
}

/// The leading term of `Integral(phi*exp(t*h), (x, -oo, oo))` for
/// `t -> oo` from the saddle points of `h` in the complex plane
/// (`steepest_descent`).
pub fn steepest_descent(phi: &Basic, h: &Basic, x: &Basic, t: &Basic, assumptions: &[Basic]) -> OptR {
    let Some(saddles) = saddle_points(h, x)? else {
        return Ok(None);
    };
    let mut found: Vec<(Basic, Basic)> = Vec::new();
    for z0 in &saddles {
        match leading_saddle_term(phi, h, x, z0, t)? {
            Some(term) => found.push((z0.clone(), term)),
            None => {
                if simplify(&subs(&diff_n(h, x, 2)?, x, z0)?)? == zero() {
                    // a degenerate saddle: an Airy-type contribution
                    return Ok(None);
                }
                // a vertical descent path
            }
        }
    }
    if found.is_empty() {
        return Ok(None);
    }
    if let Some(polynomial) = coefficients(h, x)? {
        if degree(&polynomial) == 2 && is_polynomial(phi, x) {
            let leading: Basic = coefficient_of(&polynomial, 2);
            if ask(&lt(&re(&leading)?, &zero())?, assumptions)? == Some(true) && found.len() == 1 {
                return Ok(Some(with_order(&found[0].1, &saddle_scale(h, x, &found[0].0, t)?, t)?));
            }
            return Ok(None);
        }
    }
    if !crate::settings::settings().numerical_checks || found.iter().any(|(z0, _)| !z0.free_symbols().is_empty()) {
        return Ok(None);
    }
    let mut levels: Vec<(f64, Vec<Basic>)> = Vec::new();
    for (z0, term) in &found {
        let Some(height) = crate::core::evalf::numeric_value(&re(&subs(h, x, z0)?)?, 60) else {
            return Ok(None);
        };
        let key: f64 = round8(crate::core::evalf::parts(&height).0.to_f64().value());
        match levels.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1.push(term.clone()),
            None => levels.push((key, vec![term.clone()])),
        }
    }
    levels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (key, group) in &levels {
        let mut candidates: Vec<Vec<Basic>> = vec![group.clone()];
        if group.len() > 1 {
            candidates.extend(group.iter().map(|t| vec![t.clone()]));
        }
        for candidate in candidates {
            if saddle_check(phi, h, x, t, &candidate, *key)? {
                let z0: &Basic = &found
                    .iter()
                    .find(|(_, term)| candidate.contains(term))
                    .expect("a member")
                    .0;
                let scale: Basic = saddle_scale(h, x, z0, t)?;
                return Ok(Some(with_order(&crate::core::add::add(&candidate)?, &scale, t)?));
            }
        }
    }
    Ok(None)
}

/// Python's `round(v, 8)`: half to even.
fn round8(v: f64) -> f64 {
    let y: f64 = v * 1e8;
    let mut r: f64 = y.round();
    if (y - y.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        r -= y.signum();
    }
    r / 1e8
}

/// `exp(t Re h(z0)) t**(-3/2)` (`_saddle_scale`).
fn saddle_scale(h: &Basic, x: &Basic, z0: &Basic, t: &Basic) -> R {
    let height: Basic = simplify(&re(&subs(h, x, z0)?)?)?;
    m2(&exp(&m2(t, &height)?)?, &p2(t, &rational(-3, 2))?)
}

/// `leading + O(scale)`, the sum left unevaluated when the order cannot
/// be compared with the terms (`_with_order`).
fn with_order(leading: &Basic, scale: &Basic, t: &Basic) -> R {
    let attempt_sum = || -> R { a2(leading, &order_at_oo(scale, t)?) };
    match attempt_sum() {
        Ok(v) => Ok(v),
        Err(PolyError::NotImplemented(m)) if m.starts_with(crate::error::NOT_PORTED_PREFIX) => {
            Err(PolyError::NotImplemented(m))
        }
        Err(PolyError::NotImplemented(_) | PolyError::Value(_) | PolyError::Type(_)) => Ok(Basic::new(
            Head::Add,
            vec![leading.clone(), crate::core::order::node(scale, &[(t.clone(), oo())])],
        )),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------
// stationary phase

/// The leading term of `Integral(phi*exp(I*t*h), (x, a, b))` (or with
/// `cos(t*h)`, `sin(t*h)`) for `t -> oo` (`stationary_phase`).
pub fn stationary_phase(
    phi: &Basic,
    h: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    kind: &str,
    assumptions: &[Basic],
) -> OptR {
    let Some(points) = stationary_points(h, x, a, b, assumptions)? else {
        return Ok(None);
    };
    if points.is_empty() {
        return Ok(None);
    }
    let mut total: Basic = zero();
    for x0 in &points {
        let second: Basic = subs(&diff_n(h, x, 2)?, x, x0)?;
        let positive: Option<bool> = ask(&gt(&second, &zero())?, assumptions)?;
        let negative: Option<bool> = ask(&lt(&second, &zero())?, assumptions)?;
        if positive != Some(true) && negative != Some(true) {
            return Ok(None);
        }
        let sign: Basic = if positive == Some(true) { one() } else { int(-1) };
        let phase: Basic = a2(&m2(t, &subs(h, x, x0)?)?, &d2(&m2(&sign, &pi())?, &int(4))?)?;
        let amplitude: Basic = m2(
            &subs(phi, x, x0)?,
            &sqrt(&d2(&m2(&int(2), &pi())?, &m2(&m2(t, &sign)?, &second)?)?)?,
        )?;
        let wave: Basic = match kind {
            "exp" => exp(&m2(&imag(), &phase)?)?,
            "cos" => apply(Function::Cos, &[phase])?,
            _ => apply(Function::Sin, &[phase])?,
        };
        total = a2(&total, &m2(&amplitude, &wave)?)?;
    }
    Ok(Some(a2(&total, &order_at_oo(&p2(t, &rational(-3, 2))?, t)?)?))
}

// ---------------------------------------------------------------------
// uniform expansions: coalescing stationary points

/// The real zeros of `h'` on the whole line, sorted
/// (`_real_stationary_points`).
fn real_stationary_points(h: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let Some(mut ordered) = stationary_points(h, x, &neg_oo(), &oo(), assumptions)? else {
        return Ok(None);
    };
    let n: usize = ordered.len();
    for i in 0..n {
        for j in 0..n.saturating_sub(1 + i) {
            if ask(&gt(&ordered[j], &ordered[j + 1])?, assumptions)? == Some(true) {
                ordered.swap(j, j + 1);
            } else if ask(&lt(&ordered[j], &ordered[j + 1])?, assumptions)? != Some(true) {
                return Ok(None);
            }
        }
    }
    Ok(Some(ordered))
}

fn in_range(p: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    let below: Option<bool> = if *a == neg_oo() {
        Some(true)
    } else {
        ask(&gt(p, a)?, assumptions)?
    };
    let above: Option<bool> = if *b == oo() {
        Some(true)
    } else {
        ask(&lt(p, b)?, assumptions)?
    };
    Ok(match (below, above) {
        (Some(x), Some(y)) => Some(x && y),
        _ => None,
    })
}

/// `value` as the expansion for `cos(t h)` or `sin(t h)`
/// (`_oscillatory_form`).
fn oscillatory_form(value: &Basic, kind: &str) -> R {
    match kind {
        "exp" => Ok(value.clone()),
        "cos" => simplify(&re(value)?),
        _ => simplify(&im(value)?),
    }
}

/// The leading term of the oscillatory integral uniform in the parameters
/// of `h` as its stationary points coalesce (`uniform_expansion`).
pub fn uniform_expansion(
    phi: &Basic,
    h: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    kind: &str,
    assumptions: &[Basic],
) -> OptR {
    let Some(points) = real_stationary_points(h, x, assumptions)? else {
        return Ok(None);
    };
    let mut inside: Vec<Basic> = Vec::new();
    let mut undecided: bool = false;
    for p in &points {
        match in_range(p, a, b, assumptions)? {
            Some(true) => inside.push(p.clone()),
            Some(false) => {}
            None => undecided = true,
        }
    }
    if undecided {
        return Ok(None);
    }
    if inside.len() == 2 {
        return airy_form(phi, h, x, &inside[0], &inside[1], t, kind, assumptions);
    }
    if points.len() == 1 && inside.len() <= 1 {
        let x0: &Basic = &points[0];
        if *a != neg_oo() && (*b == oo() || ask(&lt(&s2(x0, a)?, &s2(b, x0)?)?, assumptions)? == Some(true)) {
            return error_function_form(phi, h, x, x0, a, t, 1, kind, assumptions);
        }
        if *b != oo() {
            return error_function_form(phi, h, x, x0, b, t, -1, kind, assumptions);
        }
    }
    Ok(None)
}

fn airy_form(
    phi: &Basic,
    h: &Basic,
    x: &Basic,
    x1: &Basic,
    x2: &Basic,
    t: &Basic,
    kind: &str,
    assumptions: &[Basic],
) -> OptR {
    let (h1, h2) = (subs(h, x, x1)?, subs(h, x, x2)?);
    let difference: Basic = simplify(&s2(&h1, &h2)?)?;
    let s: Basic = if ask(&gt(&difference, &zero())?, assumptions)? == Some(true) {
        one()
    } else if ask(&lt(&difference, &zero())?, assumptions)? == Some(true) {
        int(-1)
    } else {
        return Ok(None);
    };
    let rho: Basic = d2(&a2(&h1, &h2)?, &int(2))?;
    let alpha: Basic = simplify(&p2(&m2(&m2(&rational(3, 4), &s)?, &difference)?, &rational(2, 3))?)?;
    let mut values: Vec<Basic> = Vec::new();
    for point in [x1, x2] {
        let second: Basic = simplify(&subs(&diff_n(h, x, 2)?, x, point)?)?;
        if second == zero() {
            return Ok(None);
        }
        values.push(m2(
            &subs(phi, x, point)?,
            &sqrt(&d2(&m2(&int(2), &sqrt(&alpha)?)?, &abs_(&second)?)?)?,
        )?);
    }
    let p0: Basic = simplify(&d2(&a2(&values[1], &values[0])?, &int(2))?)?;
    let q0: Basic = simplify(&d2(&s2(&values[1], &values[0])?, &m2(&int(2), &sqrt(&alpha)?)?)?)?;
    let argument: Basic = m2(&ng(&alpha)?, &p2(t, &rational(2, 3))?)?;
    let first: Basic = m2(
        &m2(&p0, &p2(t, &rational(-1, 3))?)?,
        &apply(Function::AiryAi, std::slice::from_ref(&argument))?,
    )?;
    let second: Basic = m2(
        &m2(&m2(&m2(&imag(), &s)?, &q0)?, &p2(t, &rational(-2, 3))?)?,
        &apply(Function::AiryAiPrime, &[argument])?,
    )?;
    let value: Basic = m2(
        &m2(&m2(&int(2), &pi())?, &exp(&m2(&m2(&imag(), t)?, &rho)?)?)?,
        &s2(&first, &second)?,
    )?;
    Ok(Some(with_order(
        &oscillatory_form(&value, kind)?,
        &p2(t, &rational(-4, 3))?,
        t,
    )?))
}

fn error_function_form(
    phi: &Basic,
    h: &Basic,
    x: &Basic,
    x0: &Basic,
    endpoint: &Basic,
    t: &Basic,
    orientation: i64,
    kind: &str,
    assumptions: &[Basic],
) -> OptR {
    let second: Basic = simplify(&subs(&diff_n(h, x, 2)?, x, x0)?)?;
    let sigma: Basic = if ask(&gt(&second, &zero())?, assumptions)? == Some(true) {
        one()
    } else if ask(&lt(&second, &zero())?, assumptions)? == Some(true) {
        int(-1)
    } else {
        return Ok(None);
    };
    let slope: Basic = subs(&diff_n(h, x, 1)?, x, endpoint)?;
    if slope == zero() {
        return Ok(None);
    }
    let gap: Basic = simplify(&m2(&sigma, &s2(&subs(h, x, endpoint)?, &subs(h, x, x0)?)?)?)?;
    if ask(&ge(&gap, &zero())?, assumptions)? != Some(true) {
        return Ok(None);
    }
    let Some(side) = ask(&gt(&m2(&int(orientation), &s2(x0, endpoint)?)?, &zero())?, assumptions)? else {
        return Ok(None);
    };
    let zeta: Basic = m2(&(if side { int(-1) } else { one() }), &sqrt(&m2(&int(2), &gap)?)?)?;
    let p0: Basic = d2(&subs(phi, x, x0)?, &sqrt(&abs_(&second)?)?)?;
    let endpoint_value: Basic = d2(
        &m2(&m2(&subs(phi, x, endpoint)?, &sigma)?, &zeta)?,
        &m2(&int(orientation), &slope)?,
    )?;
    let q0: Basic = simplify(&d2(&s2(&endpoint_value, &p0)?, &zeta)?)?;
    let phase: Basic = exp(&m2(&m2(&imag(), t)?, &subs(h, x, x0)?)?)?;
    let quarter: Basic = d2(&m2(&m2(&imag(), &sigma)?, &pi())?, &int(4))?;
    let back: Basic = d2(&m2(&m2(&ng(&imag())?, &sigma)?, &pi())?, &int(4))?;
    let erfc: Basic = apply(
        Function::Erfc,
        &[m2(&m2(&exp(&back)?, &sqrt(&d2(t, &int(2))?)?)?, &zeta)?],
    )?;
    let first: Basic = m2(
        &m2(
            &m2(&m2(&phase, &p0)?, &sqrt(&d2(&pi(), &m2(&int(2), t)?)?)?)?,
            &exp(&quarter)?,
        )?,
        &erfc,
    )?;
    let second_term: Basic = d2(
        &m2(
            &m2(&m2(&imag(), &sigma)?, &q0)?,
            &exp(&m2(&m2(&imag(), t)?, &subs(h, x, endpoint)?)?)?,
        )?,
        t,
    )?;
    let value: Basic = a2(&first, &second_term)?;
    Ok(Some(with_order(
        &oscillatory_form(&value, kind)?,
        &p2(t, &rational(-3, 2))?,
        t,
    )?))
}

// ---------------------------------------------------------------------
// the entry point

/// The first `order` terms of the expansion of `Integral(f, (x, a, b))` as
/// `parameter -> oo`, with an `Order` term, or `None` when no method
/// applies (`asymptotic_integral`).
pub fn asymptotic_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    parameter: &Basic,
    order: i64,
    assumptions: &[Basic],
    uniform: bool,
) -> OptR {
    let t: &Basic = parameter;
    if order < 1 {
        return Err(PolyError::Value("order must be at least 1".to_string()));
    }
    if let Some((phi, h, kind)) = oscillatory_part(f, x, t)? {
        if uniform {
            return uniform_expansion(&phi, &h, x, a, b, t, kind, assumptions);
        }
        return stationary_phase(&phi, &h, x, a, b, t, kind, assumptions);
    }
    let Some((phi, h)) = exponential_part(f, x, t)? else {
        return Ok(None);
    };
    // Watson's lemma: h = -c x on (0, b)
    let slope: Basic = d2(&ng(&h)?, x)?;
    if *a == zero() && !slope.has(x) && yes(&slope, Predicate::positive) {
        if let Some(scaled) = watson_lemma(&phi, x, t, order as usize)? {
            if slope != one() {
                return Ok(Some(subs(&scaled, t, &m2(&slope, t)?)?));
            }
            return Ok(Some(scaled));
        }
    }
    laplace_method(&phi, &h, x, a, b, t, order, assumptions)
}
