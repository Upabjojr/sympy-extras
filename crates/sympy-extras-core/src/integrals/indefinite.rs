//! Port of `sympy_extras/integrals/indefinite.py`: indefinite integration,
//! every antiderivative checked before it is returned. The typed methods
//! are tried in a fixed order (rational functions, radicals of a
//! quadratic, exponentials, trigonometric products, the Risch algorithm,
//! Trager's, the binomial differentials, the rewritten forms and the
//! substitutions, the heuristic Risch integrator, `manualintegrate`, the
//! Meijer G-function route), and a candidate `F` is returned only when
//! `F' - f` cancels, vanishes in the elementary tower, simplifies to zero
//! or vanishes at sample points where `f` is real ([`is_antiderivative`]).
//!
//! Python's `evalf(30)` of the integrand at a point followed by its
//! 128-bit value is the value at the precision of 30 digits (the Rust core
//! has no `Float`).

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::random::PyRandom;

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    Basic::zero()
}
fn rational(p: i64, q: i64) -> Basic {
    Basic::from_rational(crate::numbers::rat(p, q))
}
fn imag() -> Basic {
    Basic::atom(Head::ImaginaryUnit)
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
fn is_fn(e: &Basic, fs: &[Function]) -> bool {
    matches!(e.head(), Head::Function(f) if fs.contains(f))
}
fn has_any(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn has_heads(e: &Basic, fs: &[Function]) -> bool {
    has_any(e, &|n| is_fn(n, fs))
}
fn has_nan_zoo(e: &Basic) -> bool {
    has_any(e, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity))
}
fn has_nan_zoo_oo(e: &Basic) -> bool {
    has_any(e, &|n| {
        matches!(
            n.head(),
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
        )
    })
}
fn has_i(e: &Basic) -> bool {
    has_any(e, &|n| matches!(n.head(), Head::ImaginaryUnit))
}
fn is_false(b: &Basic) -> bool {
    *b == crate::core::logic::false_()
}
fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}
fn atoms_where(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if p(&n) && !found.contains(&n) {
            found.push(n);
        }
    }
    found
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn budget() -> Option<f64> {
    crate::settings::settings().timeout.map(|t| t / 4.0)
}
fn long_budget() -> Option<f64> {
    crate::settings::settings().timeout.map(|t| t / 2.0)
}
fn attempt<T>(seconds: Option<f64>, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(seconds, f)
}
fn is_unevaluated(e: &Basic) -> bool {
    has_any(e, &|n| matches!(n.head(), Head::Integral | Head::NonElementaryIntegral))
}

/// A dummy with a name of its own (`_fresh`).
fn fresh(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    Dummy::new(
        &format!("{}_indefinite_{}", name, COUNTER.fetch_add(1, Ordering::Relaxed)),
        assumptions,
    )
    .0
}

/// `e` with the replacements of `rule`, evaluated (`_replaced`).
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

/// Whether `F` is an antiderivative of `f`: `true` when `F' - f` cancels
/// or simplifies to zero, or vanishes numerically at random points
/// satisfying the assumptions; `false` when it does not vanish there;
/// `None` when nothing could be decided (`is_antiderivative`).
pub fn is_antiderivative(big_f: &Basic, f: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    if is_unevaluated(big_f) || has_nan_zoo(big_f) {
        return Ok(None);
    }
    let derivative: Basic = crate::core::function::diff(big_f, &[(x.clone(), 1)])?;
    let difference: Basic = s2(&almost_everywhere(&derivative)?, f)?;
    let reduced: Option<Basic> = attempt(budget(), || cancel(&difference))?;
    if reduced.as_ref() == Some(&zero()) {
        return Ok(Some(true));
    }
    // exactly, by the structure theorem
    if attempt(Some(EXACT_SECONDS), || exactly_zero(&difference, assumptions))? == Some(true) {
        return Ok(Some(true));
    }
    let base: Basic = reduced.unwrap_or_else(|| difference.clone());
    let simpler: Option<Basic> = attempt(budget(), || simplify(&base))?;
    if simpler.as_ref() == Some(&zero()) {
        return Ok(Some(true));
    }
    let both_signs: bool = has_i(big_f) || has_heads(big_f, &[Function::ExpPolar, Function::PolarLift]);
    vanishes_on_both_sides(&difference, f, x, assumptions, both_signs)
}

/// The budget of the exact zero test of [`is_antiderivative`].
const EXACT_SECONDS: f64 = 2.0;

/// Whether `difference` is zero in the tower of the structure theorem
/// (`_exactly_zero`).
fn exactly_zero(difference: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    use crate::simplify::structure::{is_not_elementary, Tower};
    let mut tower: Tower = Tower::new(assumptions);
    let result = tower.element(difference).and_then(|e| tower.vanishes(&e));
    match result {
        Ok(v) => Ok(v),
        Err(e) if is_not_elementary(&e) || matches!(e, PolyError::ZeroDivision(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

/// `derivative` away from the points where a `sign` factor jumps
/// (`_almost_everywhere`).
fn almost_everywhere(derivative: &Basic) -> R {
    let jump = |node: &Basic| -> bool {
        is_fn(node, &[Function::DiracDelta])
            || (matches!(node.head(), Head::Derivative) && is_fn(&node.args()[0], &[Function::Sign]))
    };
    if !has_any(derivative, &|n| {
        is_fn(n, &[Function::DiracDelta]) || matches!(n.head(), Head::Derivative)
    }) {
        return Ok(derivative.clone());
    }
    crate::core::function::replace_where(derivative, &jump, &|_| Ok(zero()))
}

/// The points where the difference of the derivatives is evaluated.
fn points() -> Vec<Basic> {
    [
        (-37, 10),
        (-19, 10),
        (-13, 10),
        (-2, 5),
        (-1, 20),
        (1, 20),
        (3, 5),
        (13, 10),
        (19, 10),
        (41, 10),
    ]
    .iter()
    .map(|&(p, q)| rational(p, q))
    .collect()
}

/// Whether `difference` vanishes at the fixed real points (the parameters
/// sampled under the assumptions) where `f` is a finite real number
/// (`_vanishes_on_both_sides`).
fn vanishes_on_both_sides(
    difference: &Basic,
    f: &Basic,
    x: &Basic,
    assumptions: &[Basic],
    both_signs: bool,
) -> PolyResult<Option<bool>> {
    let mut parameters: Vec<Basic> = difference.free_symbols().into_iter().filter(|s| s != x).collect();
    parameters.sort_by_key(symbol_name);
    let samples: usize = if parameters.is_empty() { 1 } else { 3 };
    let fixed: Vec<Basic> = points();
    let mut decided: bool = false;
    for index in 0..samples {
        let mut values: Vec<(Basic, Basic)> = Vec::new();
        if !parameters.is_empty() {
            let seed: String = if index == 0 {
                difference.to_string()
            } else {
                format!("{}#{}", difference, index)
            };
            let mut rng: PyRandom = PyRandom::from_string(&seed);
            match crate::integrals::conditions::sample_values(&parameters, assumptions, &mut rng)? {
                None => break,
                Some(found) => values = found,
            }
        }
        let (mut verdict, mut signs) = tested_points(difference, f, x, &values, assumptions, &fixed)?;
        if verdict.is_none() {
            let mut rng: PyRandom = PyRandom::from_string(&difference.to_string());
            let extra: Vec<Basic> = domain_points(f, x, &values, &mut rng, 6)?;
            if !extra.is_empty() {
                let mut all: Vec<Basic> = fixed.clone();
                all.extend(extra);
                (verdict, signs) = tested_points(difference, f, x, &values, assumptions, &all)?;
            }
        }
        if verdict == Some(false) {
            return Ok(Some(false));
        }
        if verdict == Some(true) && (!both_signs || signs.len() == 2) {
            decided = true;
        }
    }
    Ok(if decided { Some(true) } else { None })
}

/// Points where the radicands and the arguments of the logarithms of `f`
/// are positive (`_domain_points`).
fn domain_points(
    f: &Basic,
    x: &Basic,
    values: &[(Basic, Basic)],
    rng: &mut PyRandom,
    count: usize,
) -> PolyResult<Vec<Basic>> {
    let mut conditions: Vec<Basic> = Vec::new();
    for node in atoms_where(f, &|n| n.is_pow()) {
        let exponent: &Basic = &node.args()[1];
        if exponent.as_rational().is_some() && is(exponent, Predicate::integer) != Some(true) && node.args()[0].has(x) {
            conditions.push(crate::core::relational::gt(
                &replaced(&node.args()[0], values)?,
                &zero(),
            )?);
        }
    }
    for node in atoms_where(f, &|n| is_fn(n, &[Function::Log])) {
        let argument: &Basic = &node.args()[0];
        if argument.has(x) {
            conditions.push(crate::core::relational::gt(&replaced(argument, values)?, &zero())?);
        }
    }
    if conditions.is_empty() {
        return Ok(Vec::new());
    }
    let mut found_points: Vec<Basic> = Vec::new();
    for _ in 0..count {
        let found = attempt(budget(), || {
            crate::integrals::conditions::sample_values(std::slice::from_ref(x), &conditions, rng)
        })?
        .flatten();
        let Some(found) = found.filter(|v| !v.is_empty()) else {
            break;
        };
        let Some(point) = found.iter().find(|(k, _)| k == x).map(|(_, v)| v.clone()) else {
            break;
        };
        if !found_points.contains(&point) {
            found_points.push(point);
        }
    }
    Ok(found_points)
}

fn sign_of(point: &Basic) -> i32 {
    match crate::core::evalf::numeric_value(point, 64) {
        Some(v) if crate::core::evalf::parts(&v).0.to_f64().value() > 0.0 => 1,
        _ => -1,
    }
}

/// `(|value|, |im(value)|)` of a number at `prec` bits, `None` when it is
/// not a finite number.
fn magnitudes(v: &crate::core::evalf::Complex) -> Option<(f64, f64)> {
    let (a, b) = crate::core::evalf::parts(v);
    let (a, b) = (a.to_f64().value(), b.to_f64().value());
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    Some(((a * a + b * b).sqrt(), b.abs()))
}

/// The verdict of [`vanishes_on_both_sides`] at one sample of the
/// parameters and the given points, with the signs of the points tested
/// (`_tested_points`).
fn tested_points(
    difference: &Basic,
    f: &Basic,
    x: &Basic,
    values: &[(Basic, Basic)],
    facts: &[Basic],
    points: &[Basic],
) -> PolyResult<(Option<bool>, Vec<i32>)> {
    let mut tested: usize = 0;
    let mut signs: Vec<i32> = Vec::new();
    let integrand: Basic = replaced(f, values)?;
    let conditions: Vec<Basic> = if values.is_empty() {
        facts.to_vec()
    } else {
        facts
            .iter()
            .map(|fact| replaced(fact, values))
            .collect::<PolyResult<_>>()?
    };
    for point in points {
        let at: Vec<(Basic, Basic)> = vec![(x.clone(), point.clone())];
        // a point the assumptions on x exclude is not a test
        let mut excluded: bool = false;
        for fact in &conditions {
            if is_false(&replaced(fact, &at)?) {
                excluded = true;
                break;
            }
        }
        if excluded {
            continue;
        }
        let at_point: Basic = replaced(&integrand, &at)?;
        // Python's at_point.evalf(30), then its value at 128 bits
        let prec: usize = crate::numeric::dps_to_prec(30);
        let sample = attempt(budget(), || Ok(crate::core::evalf::numeric_value(&at_point, prec)))?.flatten();
        let Some(sample) = sample else {
            continue;
        };
        let Some((size, imaginary)) = magnitudes(&sample) else {
            continue;
        };
        let tiny: f64 = 1e-20;
        if size < tiny || imaginary > tiny || imaginary > size * tiny {
            continue;
        }
        // a point where the rounding chooses the side of a branch cut is
        // no test
        let mut at_both: Vec<(Basic, Basic)> = values.to_vec();
        at_both.retain(|(k, _)| k != x);
        at_both.push((x.clone(), point.clone()));
        let number = attempt(budget(), || {
            Ok(crate::numeric::reliable_value(difference, 30, &at_both))
        })?
        .flatten();
        let Some(number) = number else {
            continue;
        };
        let Some((residual, _)) = magnitudes(&number) else {
            continue;
        };
        if residual > size.max(1.0) / 1e15 {
            return Ok((Some(false), signs));
        }
        tested += 1;
        let s: i32 = sign_of(point);
        if !signs.contains(&s) {
            signs.push(s);
        }
    }
    Ok((if tested >= 2 { Some(true) } else { None }, signs))
}

/// `F` rewritten without `I` and polar numbers when a rewriting which is
/// still an antiderivative of `f` is found; `F` itself otherwise
/// (`real_form`).
pub fn real_form(big_f: &Basic, f: &Basic, x: &Basic, assumptions: &[Basic]) -> R {
    let complex = |e: &Basic| has_i(e) || has_heads(e, &[Function::ExpPolar]);
    if !complex(big_f) || has_i(f) {
        return Ok(big_f.clone());
    }
    let candidates: [&dyn Fn() -> R; 5] = [
        &|| conjugate_logarithms(big_f, x),
        &|| simplify(&conjugate_logarithms(big_f, x)?),
        &|| crate::simplify::simplify::logcombine(big_f, true),
        &|| simplify(big_f),
        &|| simplify(&crate::simplify::simplify::logcombine(big_f, true)?),
    ];
    for candidate in candidates {
        let Some(rewritten) = attempt(budget(), candidate)? else {
            continue;
        };
        if complex(&rewritten) {
            continue;
        }
        if is_antiderivative(&rewritten, f, x, assumptions)? == Some(true) {
            return Ok(rewritten);
        }
    }
    Ok(big_f.clone())
}

/// The pairs `a*log(p) - a*log(conjugate(p))` of `F` written as the
/// arctangents `2*Im(a)*atan(u/v)` (`conjugate_logarithms`).
pub fn conjugate_logarithms(big_f: &Basic, x: &Basic) -> R {
    let real: Basic = fresh(&symbol_name(x), &[(Predicate::real, true)]);
    let to_real: Vec<(Basic, Basic)> = vec![(x.clone(), real.clone())];
    let back: Vec<(Basic, Basic)> = vec![(real, x.clone())];
    let mut terms: Vec<(Basic, Basic, Basic)> = Vec::new();
    let mut others: Vec<Basic> = Vec::new();
    for term in crate::core::expr::terms_of(big_f) {
        let (coefficient, rest) = crate::core::expr::as_independent(&term, x, false)?;
        if is_fn(&rest, &[Function::Log]) && is(&coefficient, Predicate::imaginary) == Some(true) {
            let argument: Basic = replaced(&rest.args()[0], &to_real)?;
            let (u, v) = match crate::functions::elementary::complexes::as_real_imag(&argument, true, None)? {
                Some(parts) => parts,
                None => (
                    crate::functions::elementary::complexes::re(&argument)?,
                    crate::functions::elementary::complexes::im(&argument)?,
                ),
            };
            terms.push((coefficient, replaced(&u, &back)?, replaced(&v, &back)?));
        } else {
            others.push(term);
        }
    }
    let mut used: Vec<usize> = Vec::new();
    for i in 0..terms.len() {
        let (a, u, v) = terms[i].clone();
        if used.contains(&i) {
            continue;
        }
        if v == zero() || u == zero() {
            // not one of a pair: kept as it is (Python dropped it)
            others.push(m2(
                &a,
                &crate::functions::elementary::exponential::log(&a2(&u, &m2(&imag(), &v)?)?)?,
            )?);
            continue;
        }
        let mut paired: bool = false;
        for (j, (b, u2, v2)) in terms.iter().enumerate().skip(i + 1) {
            if used.contains(&j) || *u2 != u || a2(v2, &v)? != zero() || a2(&a, b)? != zero() {
                continue;
            }
            // log(u + I v) - log(u - I v) = 2 I atan2(v, u): the constant
            // dropped, and atan(u/v) is continuous where v does not vanish
            let im_a: Basic = crate::functions::elementary::complexes::im(&a)?;
            let atan: Basic = crate::functions::apply_function(Function::Atan, &[d2(&u, &v)?])?;
            others.push(m2(&m2(&int(2), &im_a)?, &atan)?);
            used.push(i);
            used.push(j);
            paired = true;
            break;
        }
        if !paired {
            others.push(m2(
                &a,
                &crate::functions::elementary::exponential::log(&a2(&u, &m2(&imag(), &v)?)?)?,
            )?);
        }
    }
    if used.is_empty() {
        return Ok(big_f.clone());
    }
    crate::core::add::add(&others)
}

// ---------------------------------------------------------------------
// the methods

/// The names of the methods, in the order they are tried (`METHODS`).
pub const METHODS: [&str; 12] = [
    "rational",
    "radicals",
    "exponential",
    "trigonometric",
    "risch",
    "trager",
    "binomial",
    "rewriting",
    "heurisch",
    "rewritten heurisch",
    "manual",
    "meijer",
];

/// The methods tried on a rewritten form or a substituted integrand
/// (`TYPED`).
pub const TYPED: [&str; 7] = [
    "rational",
    "radicals",
    "exponential",
    "trigonometric",
    "risch",
    "heurisch",
    "trager",
];

/// The typed methods without the heuristic (`_FAST_TYPED`).
const FAST_TYPED: [&str; 6] = [
    "rational",
    "radicals",
    "exponential",
    "trigonometric",
    "risch",
    "trager",
];

const HYPERBOLIC: [Function; 6] = [
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];
const INVERSE_HYPERBOLIC: [Function; 6] = [
    Function::Asinh,
    Function::Acosh,
    Function::Atanh,
    Function::Acoth,
    Function::Asech,
    Function::Acsch,
];

/// Whether `f` has hyperbolic functions and no inverse ones
/// (`_hyperbolic`).
fn hyperbolic(f: &Basic) -> bool {
    has_heads(f, &HYPERBOLIC) && !has_heads(f, &INVERSE_HYPERBOLIC)
}

fn method(name: &str, f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    match name {
        "rational" => {
            if crate::core::expr::polynomial_like(
                f,
                std::slice::from_ref(x),
                crate::core::expr::PolynomialKind::Rational,
            ) != Some(true)
            {
                return Ok(None);
            }
            let symbol: Basic = crate::core::symbol::Symbol::new("t", &[]).0;
            attempt(budget(), || {
                let integrand = crate::integrals::risch::rationaltools::Integrand::of(f)?;
                crate::integrals::risch::rationaltools::ratint(&integrand, x, &symbol, None)
            })
        }
        "radicals" => Ok(attempt(budget(), || {
            crate::integrals::radicals::quadratic_radical_antiderivative(f, x, assumptions, true)
        })?
        .flatten()),
        "exponential" => {
            if !has_heads(f, &[Function::Exp]) {
                return Ok(None);
            }
            Ok(attempt(budget(), || {
                crate::integrals::exponential::exponential_antiderivative(f, x, assumptions)
            })?
            .flatten())
        }
        "trigonometric" => Ok(attempt(budget(), || {
            crate::integrals::trigonometric::trigonometric_antiderivative(f, x)
        })?
        .flatten()),
        "risch" => {
            let found: Option<Basic> = attempt(budget(), || {
                crate::integrals::risch::risch::risch_integrate(f, x, None, "log", None, "piecewise")
            })?;
            Ok(found.filter(|v| !is_unevaluated(v)))
        }
        "heurisch" => Ok(attempt(budget(), || crate::integrals::heurisch::heurisch_cases(f, x))?.flatten()),
        "trager" => Ok(attempt(budget(), || crate::integrals::trager::trager_antiderivative(f, x))?.flatten()),
        "binomial" => Ok(attempt(budget(), || {
            crate::integrals::algebraic::binomial_hypergeometric_antiderivative(f, x)
        })?
        .flatten()),
        "rewriting" => rewriting(f, x, assumptions),
        "rewritten heurisch" => rewritten_heurisch(f, x, assumptions),
        "manual" => {
            let found: Option<Basic> = attempt(long_budget(), || {
                crate::integrals::manualintegrate::manualintegrate(f, x)
            })?;
            Ok(found.filter(|v| !has_any(v, &|n| matches!(n.head(), Head::Integral))))
        }
        "meijer" => Ok(attempt(long_budget(), || {
            crate::integrals::meijerint::meijer_antiderivative(f, x)
        })?
        .flatten()),
        _ => Ok(None),
    }
}

fn forms_of(f: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<(Vec<Basic>, Vec<Basic>)> {
    use crate::integrals::rewriting::{implied_assumptions, rewritten_forms};
    let forms: Vec<Basic> = attempt(budget(), || rewritten_forms(f, x, assumptions))?.unwrap_or_default();
    let mut implied: Vec<Basic> = assumptions.to_vec();
    implied.extend(attempt(budget(), || implied_assumptions(f, x, assumptions))?.unwrap_or_default());
    Ok((forms, implied))
}

/// The typed methods on the canonical forms of `f` and on the integrands
/// of its substitutions (`_rewriting`).
fn rewriting(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    use crate::integrals::rewriting::{power_substitutions, Substitution};
    let (forms, implied) = forms_of(f, x, assumptions)?;
    for form in &forms {
        if let Some(found) = verified_antiderivative(form, x, &implied, Some(&FAST_TYPED))? {
            return Ok(Some(found.0));
        }
    }
    // the substitutions of f and of its first canonical form
    let mut substitutions: Vec<Substitution> = attempt(budget(), || power_substitutions(f, x))?.unwrap_or_default();
    for form in forms.iter().take(1) {
        for extra in attempt(budget(), || power_substitutions(form, x))?.unwrap_or_default() {
            if substitutions.iter().all(|s| extra.back != s.back) {
                substitutions.push(extra);
            }
        }
    }
    // the facts about the parameters hold in the new variable, those about x do not
    let inner: Vec<Basic> = implied.iter().filter(|item| !item.has(x)).cloned().collect();
    for substitution in &substitutions {
        if !substitution.facts.is_empty() {
            continue;
        }
        if let Some(big_f) = substituted(substitution, &inner)? {
            return Ok(Some(big_f));
        }
    }
    // hyperbolic functions: the heuristic on their exponentials
    if hyperbolic(f) {
        for form in &forms {
            if let Some(found) = verified_antiderivative(form, x, &implied, Some(&["heurisch"]))? {
                return Ok(Some(found.0));
            }
        }
    }
    // the substitutions holding on a region each, assembled piecewise
    let mut pieces: Vec<(Basic, Basic)> = Vec::new();
    for substitution in &substitutions {
        if substitution.facts.is_empty() {
            continue;
        }
        let Some(big_f) = substituted(substitution, &inner)? else {
            return Ok(None);
        };
        let mut facts: Vec<Basic> = implied.clone();
        facts.extend(substitution.facts.iter().cloned());
        if is_antiderivative(&big_f, f, x, &facts)? != Some(true) {
            return Ok(None);
        }
        pieces.push((big_f, crate::core::logic::and_(&substitution.facts)));
    }
    if let Some(last) = pieces.last_mut() {
        last.1 = crate::core::logic::true_();
        return Ok(Some(crate::functions::elementary::piecewise::piecewise(&pieces)?));
    }
    Ok(None)
}

/// The heuristic Risch integrator on the canonical forms of `f`
/// (`_rewritten_heurisch`).
fn rewritten_heurisch(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    if hyperbolic(f) {
        return Ok(None);
    }
    let (forms, implied) = forms_of(f, x, assumptions)?;
    for form in &forms {
        if let Some(found) = verified_antiderivative(form, x, &implied, Some(&["heurisch"]))? {
            return Ok(Some(found.0));
        }
    }
    Ok(None)
}

/// The antiderivative of `f` through `substitution` (`_substituted`).
fn substituted(substitution: &crate::integrals::rewriting::Substitution, inner: &[Basic]) -> OptR {
    use crate::integrals::rewriting::{rewritten_forms, substitute_back};
    let (g, t) = (&substitution.integrand, &substitution.variable);
    let mut candidates: Vec<Basic> = vec![g.clone()];
    candidates.extend(attempt(budget(), || rewritten_forms(g, t, inner))?.unwrap_or_default());
    for candidate in &candidates {
        if let Some(found) = verified_antiderivative(candidate, t, inner, Some(&TYPED))? {
            return Ok(Some(substitute_back(&found.0, t, &substitution.back)?));
        }
    }
    Ok(None)
}

/// An antiderivative of `f` which checks by [`is_antiderivative`], with
/// the name of the method which found it; `None` when no method gives
/// one; `methods` restricts the names tried (`verified_antiderivative`).
pub fn verified_antiderivative(
    f: &Basic,
    x: &Basic,
    assumptions: &[Basic],
    methods: Option<&[&str]>,
) -> PolyResult<Option<(Basic, String)>> {
    for name in METHODS {
        if methods.is_some_and(|m| !m.contains(&name)) {
            continue;
        }
        let found: Option<Basic> = match method(name, f, x, assumptions) {
            Ok(v) => v,
            // the internals fail on some inputs
            Err(
                PolyError::ZeroDivision(_) | PolyError::Index(_) | PolyError::Recursion(_) | PolyError::Arithmetic(_),
            ) => None,
            Err(e) => return Err(e),
        };
        let Some(found) = found else {
            continue;
        };
        if is_unevaluated(&found) || has_nan_zoo_oo(&found) {
            continue;
        }
        let allowed = f.free_symbols();
        if found.free_symbols().iter().any(|s| s != x && !allowed.contains(s)) {
            // a constant of integration or a dummy leaked
            continue;
        }
        let found: Basic = real_form(&found, f, x, assumptions)?;
        if is_antiderivative(&found, f, x, assumptions)? == Some(true) {
            return Ok(Some((found, name.to_string())));
        }
    }
    Ok(None)
}

/// `Integral(f, x)` by the methods, each result checked; the unevaluated
/// `Integral` when none gives an antiderivative which checks; with
/// `special_values` the isolated values of the parameters get cases
/// (`indefinite_integral`).
pub fn indefinite_integral(f: &Basic, x: &Basic, assumptions: &[Basic], special_values: bool) -> R {
    indefinite(f, x, assumptions, special_values, true)
}

/// An antiderivative of `f` within `seconds` (the time limit of the
/// settings when `None`), with the cases of the special values when
/// `cases`; `None` when none checks or the time is up, and with `closed`
/// also when it has a `Piecewise` or a polar number
/// (`closed_antiderivative`).
pub fn closed_antiderivative(f: &Basic, x: &Basic, cases: bool, closed: bool, seconds: Option<f64>) -> OptR {
    let limit: Option<f64> = seconds.or(crate::settings::settings().timeout);
    let found: Option<Basic> = attempt(limit, || indefinite(f, x, &[], cases, true))?;
    let Some(found) = found.filter(|v| !is_unevaluated(v)) else {
        return Ok(None);
    };
    if closed && (has_any(&found, &|n| matches!(n.head(), Head::Piecewise)) || has_heads(&found, &[Function::ExpPolar]))
    {
        return Ok(None);
    }
    Ok(Some(found))
}

/// [`indefinite_integral`]; the cases of the special values in SymPy's
/// form with `sympy_style` (`_indefinite`).
fn indefinite(f: &Basic, x: &Basic, assumptions: &[Basic], special_values: bool, sympy_style: bool) -> R {
    let Some(found) = verified_antiderivative(f, x, assumptions, None)? else {
        return crate::integrals::integral::integral(f, x, None);
    };
    if !special_values {
        return Ok(found.0);
    }
    let mut parameters: Vec<Basic> = f.free_symbols().into_iter().filter(|s| s != x).collect();
    parameters.sort_by_key(symbol_name);
    let at_point = |point: &crate::special_values::Point, at: &[Basic]| -> OptR {
        Ok(Some(indefinite(&replaced(f, point)?, x, at, true, false)?))
    };
    // an error of the test (Python's raises through with_special_values)
    // is kept and raised after it
    let failure: std::cell::RefCell<Option<PolyError>> = std::cell::RefCell::new(None);
    let defined = |point: &crate::special_values::Point| -> bool {
        if failure.borrow().is_some() {
            return false;
        }
        match replaced(f, point) {
            Ok(g) => crate::special_values::defined_problem(&g, std::slice::from_ref(x)),
            Err(e) => {
                *failure.borrow_mut() = Some(e);
                false
            }
        }
    };
    let value: R = crate::special_values::with_special_values(
        &found.0,
        &parameters,
        assumptions,
        &at_point,
        Some(&defined),
        sympy_style,
        None,
        false,
        None,
    );
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }
    value
}
