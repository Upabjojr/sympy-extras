//! Port of `sympy_extras/integrals/antiderivative.py`: definite integrals
//! from an antiderivative, cut at the discontinuities of the
//! antiderivative and the singularities of the integrand inside the range
//! and evaluated by one-sided limits; the principal value and Hadamard's
//! finite part by the symmetric excision at the singular points.
//!
//! Python's binary operators evaluate at each step: the expressions are
//! built with the binary helpers in the same association.

#![allow(clippy::mutable_key_type)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

/// An antiderivative of `f` in `x` under the assumptions, by the exact
/// methods or (with the last argument `true`) by the heuristic ones
/// (`AntiderivativeStep`).
pub type AntiderivativeStep<'a> = &'a dyn Fn(&Basic, &Basic, &[Basic], bool) -> OptR;

static COUNTER: AtomicU64 = AtomicU64::new(0);

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
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
fn abs_(a: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(a)
}
fn im(a: &Basic) -> R {
    crate::functions::elementary::complexes::im(a)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn budget(share: f64) -> Option<f64> {
    timeout().map(|t| t / share)
}
fn attempt<T>(seconds: Option<f64>, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(seconds, f)
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
fn is_fn(e: &Basic, fs: &[Function]) -> bool {
    matches!(e.head(), Head::Function(f) if fs.contains(f))
}
fn has_any(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn has_fn(e: &Basic, fs: &[Function]) -> bool {
    has_any(e, &|n| is_fn(n, fs))
}
fn has_infinities(e: &Basic) -> bool {
    has_any(e, &|n| matches!(n.head(), Head::Infinity | Head::NegativeInfinity))
}
fn is_number(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}
fn is_unevaluated(e: &Basic) -> bool {
    has_any(e, &|n| matches!(n.head(), Head::Integral))
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
fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}
fn is_symbol(e: &Basic) -> bool {
    matches!(e.head(), Head::Symbol(_) | Head::Dummy(..))
}
fn xreplace(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn is_polynomial_in(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
}
fn reals() -> Basic {
    crate::sets::sets::reals()
}

/// A dummy with a name of its own (`_fresh`).
fn fresh(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    Dummy::new(
        &format!("{}_{}", name, COUNTER.fetch_add(1, Ordering::Relaxed)),
        assumptions,
    )
    .0
}

/// The methods of the indefinite dispatcher tried for a definite integral.
pub const METHODS: [&str; 9] = [
    "rational",
    "radicals",
    "exponential",
    "trigonometric",
    "risch",
    "trager",
    "binomial",
    "manual",
    "meijer",
];

/// The heuristic methods, tried late and under a larger budget.
const LATE_METHODS: [&str; 5] = ["heurisch", "rewriting", "rewritten heurisch", "manual", "meijer"];

/// The elementary functions of linear arguments which integrate takes at
/// once (`_ELEMENTARY`).
const ELEMENTARY: [Function; 5] = [
    Function::Sin,
    Function::Cos,
    Function::Exp,
    Function::Sinh,
    Function::Cosh,
];

/// An antiderivative of `f` by the radical table, by
/// [`elementary_antiderivative`] for a polynomial in elementary functions,
/// by parts for the shapes with `log(x)` or `erf`, and by the verified
/// methods of `indefinite` otherwise (`antiderivative`).
pub fn antiderivative(f: &Basic, x: &Basic, assumptions: &[Basic], late: bool) -> OptR {
    if let Some(found) = crate::integrals::radicals::quadratic_radical_antiderivative(f, x, &[], true)? {
        return Ok(Some(found));
    }
    if elementary_polynomial(f, x)? {
        let g: Basic = crate::core::function::replace_where(f, &|n| is_fn(n, &ELEMENTARY), &expanded_argument)?;
        if let Some(found) = elementary_antiderivative(&g, x, timeout())? {
            return Ok(Some(found));
        }
    }
    if let Some(found) = logarithm_by_parts(f, x)? {
        return Ok(Some(found));
    }
    if let Some(found) = error_function_by_parts(f, x)? {
        return Ok(Some(found));
    }
    let methods: Vec<&str> = if late { LATE_METHODS.to_vec() } else { METHODS.to_vec() };
    let share: f64 = if late { 4.0 } else { 8.0 };
    let verified = attempt(budget(share), || {
        crate::integrals::indefinite::verified_antiderivative(f, x, assumptions, Some(&methods))
    })?
    .flatten();
    Ok(verified.map(|v| v.0))
}

/// The methods tried on the terms of a polynomial in elementary functions
/// which the rules do not integrate (`_ELEMENTARY_METHODS`).
const ELEMENTARY_METHODS: [&str; 2] = ["exponential", "heurisch"];

/// An antiderivative of `f`, a polynomial in `x` and in elementary
/// functions of linear arguments, within `seconds`
/// (`elementary_antiderivative`).
pub fn elementary_antiderivative(f: &Basic, x: &Basic, seconds: Option<f64>) -> OptR {
    use crate::integrals::indefinite::{is_antiderivative, verified_antiderivative};
    let run = || -> OptR {
        let found: Basic = crate::integrals::manualintegrate::manualintegrate(f, x)?;
        if !is_unevaluated(&found) && is_antiderivative(&found, f, x, &[])? == Some(true) {
            return Ok(Some(found));
        }
        if let Some(v) = verified_antiderivative(f, x, &[], Some(&ELEMENTARY_METHODS))? {
            return Ok(Some(v.0));
        }
        let terms: Vec<Basic> = crate::core::expr::terms_of(&expand(f)?);
        if terms.len() < 2 {
            return Ok(None);
        }
        let mut total: Basic = zero();
        for term in &terms {
            let mut piece: Basic = crate::integrals::manualintegrate::manualintegrate(term, x)?;
            if is_unevaluated(&piece) || is_antiderivative(&piece, term, x, &[])? != Some(true) {
                let Some(v) = verified_antiderivative(term, x, &[], Some(&ELEMENTARY_METHODS))? else {
                    return Ok(None);
                };
                piece = v.0;
            }
            total = a2(&total, &piece)?;
        }
        Ok(Some(total))
    };
    Ok(attempt(seconds, run)?.flatten())
}

/// An elementary function with its argument expanded
/// (`_expanded_argument`).
fn expanded_argument(node: &Basic) -> R {
    let Head::Function(f) = node.head() else {
        return Ok(node.clone());
    };
    crate::functions::apply_function(*f, &[expand(&node.args()[0])?])
}

fn is_application(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Function(_) | Head::Undefined(_) | Head::Hyper | Head::MeijerG | Head::Piecewise
    )
}

/// Whether `f` is a polynomial in `x` and in sines, cosines, exponentials
/// and hyperbolic functions of arguments linear in `x`
/// (`elementary_polynomial`).
pub fn elementary_polynomial(f: &Basic, x: &Basic) -> PolyResult<bool> {
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for node in atoms_where(f, &is_application) {
        if !node.has(x) {
            continue;
        }
        if !is_fn(&node, &ELEMENTARY) || node.args().len() != 1 {
            return Ok(false);
        }
        if !linear(&node.args()[0], x)? {
            return Ok(false);
        }
        replacement.push((node.clone(), fresh("e", &[])));
    }
    let substituted: Basic = xreplace(f, &replacement)?;
    let mut variables: Vec<Basic> = vec![x.clone()];
    variables.extend(replacement.iter().map(|(_, s)| s.clone()));
    Ok(is_polynomial_in(&substituted, &variables))
}

/// Whether `argument` is a polynomial of degree one in `x` (`_linear`).
fn linear(argument: &Basic, x: &Basic) -> PolyResult<bool> {
    if !is_polynomial_in(argument, std::slice::from_ref(x)) {
        return Ok(false);
    }
    let slope: Basic = crate::core::function::diff(argument, &[(x.clone(), 1)])?;
    Ok(!slope.has(x) && is(&slope, Predicate::zero) != Some(true))
}

/// Whether `f` is of a shape the antiderivative comes at once for
/// (`quick_shape`).
pub fn quick_shape(f: &Basic, x: &Basic) -> PolyResult<bool> {
    Ok(elementary_polynomial(f, x)? || by_parts_shape(f, x)? || error_function_shape(f, x)?)
}

fn error_functions(f: &Basic) -> Vec<Basic> {
    mul_args(f)
        .into_iter()
        .filter(|e| e.is_function(Function::Erf))
        .collect()
}

fn error_function_shape(f: &Basic, x: &Basic) -> PolyResult<bool> {
    let errors: Vec<Basic> = error_functions(f);
    Ok(errors.len() == 1 && elementary_polynomial(&d2(f, &errors[0])?, x)?)
}

/// Whether `f` is a polynomial in `x` and exponentials times `log(x)`
/// (`by_parts_shape`).
pub fn by_parts_shape(f: &Basic, x: &Basic) -> PolyResult<bool> {
    let logarithms: Vec<Basic> = mul_args(f)
        .into_iter()
        .filter(|e| e.is_function(Function::Log))
        .collect();
    let log_x: Basic = log(x)?;
    if logarithms.len() != 1 || logarithms[0] != log_x {
        return Ok(false);
    }
    let g: Basic = d2(f, &log_x)?;
    Ok(!has_fn(&g, &[Function::Log]) && elementary_polynomial(&g, x)? && has_fn(&g, &[Function::Exp]))
}

/// `g(x)*erf(k*x)` by parts (`_error_function_by_parts`).
fn error_function_by_parts(f: &Basic, x: &Basic) -> OptR {
    let errors: Vec<Basic> = error_functions(f);
    if errors.len() != 1 {
        return Ok(None);
    }
    let argument: &Basic = &errors[0].args()[0];
    let (k, variable) = crate::core::expr::as_independent(argument, x, false)?;
    if variable != *x || k.has(x) {
        return Ok(None);
    }
    let g: Basic = d2(f, &errors[0])?;
    if has_fn(&g, &[Function::Erf]) || !elementary_polynomial(&g, x)? {
        return Ok(None);
    }
    let plain: Basic = fresh(&symbol_name(x), &[]);
    let g: Basic = subs(&g, x, &plain)?;
    let share: Option<f64> = budget(8.0);
    let Some(big_g) = elementary_antiderivative(&g, &plain, share)? else {
        return Ok(None);
    };
    let rewritten: Basic =
        crate::core::rewriting::rewrite(&big_g, &[], "exp", &crate::core::rewriting::RewriteHints::default())?;
    let gauss: Basic = exp(&m2(&ng(&p2(&k, &int(2))?)?, &p2(&plain, &int(2))?)?)?;
    let remainder: Basic = expand(&m2(&rewritten, &gauss)?)?;
    let measure = |e: &Basic| crate::core::function::count_ops(e);
    let mut big_h: Basic = zero();
    for term in crate::core::expr::terms_of(&remainder) {
        let combined: Basic =
            crate::simplify::powsimp::powsimp(&term, false, crate::simplify::powsimp::Combine::Exp, false, &measure)?;
        let Some(piece) = elementary_antiderivative(&combined, &plain, share)? else {
            return Ok(None);
        };
        big_h = a2(&big_h, &piece)?;
    }
    let erf: Basic = crate::functions::apply_function(Function::Erf, &[m2(&k, &plain)?])?;
    let second: Basic = m2(&d2(&m2(&int(2), &k)?, &p2(&Basic::pi(), &Basic::half())?)?, &big_h)?;
    Ok(Some(subs(&s2(&m2(&big_g, &erf)?, &second)?, &plain, x)?))
}

/// `g(x)*log(x)` by parts (`_logarithm_by_parts`).
fn logarithm_by_parts(f: &Basic, x: &Basic) -> OptR {
    if !by_parts_shape(f, x)? {
        return Ok(None);
    }
    let plain: Basic = fresh(&symbol_name(x), &[]);
    let g: Basic = subs(&d2(f, &log(x)?)?, x, &plain)?;
    let share: Option<f64> = budget(8.0);
    let Some(big_g) = elementary_antiderivative(&g, &plain, share)? else {
        return Ok(None);
    };
    let Some(big_h) = elementary_antiderivative(&cancel(&d2(&big_g, &plain)?)?, &plain, share)? else {
        return Ok(None);
    };
    Ok(Some(subs(&s2(&m2(&big_g, &log(&plain)?)?, &big_h)?, &plain, x)?))
}

/// Whether `p` is not real once the parameters the assumptions make real
/// are taken as such (`_not_real`).
fn not_real(p: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for q in p.free_symbols() {
        if !is_symbol(&q) {
            continue;
        }
        let name: String = symbol_name(&q);
        if ask(&gt(&q, &zero())?, assumptions)? == Some(true) {
            replacement.push((q.clone(), fresh(&name, &[(Predicate::positive, true)])));
        } else if ask(&lt(&q, &zero())?, assumptions)? == Some(true) {
            replacement.push((q.clone(), fresh(&name, &[(Predicate::negative, true)])));
        } else if yes(&q, Predicate::extended_real)
            || ask(&crate::assumptions::facts::element(&q, &reals())?, assumptions)? == Some(true)
        {
            replacement.push((q.clone(), fresh(&name, &[(Predicate::real, true)])));
        }
    }
    if replacement.is_empty() {
        return Ok(false);
    }
    let substituted: Basic = xreplace(p, &replacement)?;
    if is(&substituted, Predicate::extended_real) == Some(false) {
        return Ok(true);
    }
    let imaginary: Basic = im(&substituted)?;
    Ok(!has_fn(&imaginary, &[Function::Im]) && is(&imaginary, Predicate::zero) == Some(false))
}

/// The points strictly inside `(a, b)`, sorted; `None` when a position is
/// undecided (`_points_in`).
fn points_in(points: &[Basic], a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let mut inside: Vec<Basic> = Vec::new();
    for p in points {
        if is(p, Predicate::extended_real) == Some(false) || not_real(p, assumptions)? {
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
        if below && above && !inside.contains(p) {
            inside.push(p.clone());
        }
    }
    for i in 0..inside.len() {
        for j in (i + 1)..inside.len() {
            if ask(&lt(&inside[i], &inside[j])?, assumptions)?.is_none() {
                return Ok(None);
            }
        }
    }
    let mut keyed: Vec<(usize, Basic)> = Vec::new();
    for p in &inside {
        let mut below: usize = 0;
        for q in &inside {
            if ask(&lt(q, p)?, assumptions)? == Some(true) {
                below += 1;
            }
        }
        keyed.push((below, p.clone()));
    }
    keyed.sort_by_key(|k| k.0);
    Ok(Some(keyed.into_iter().map(|(_, p)| p).collect()))
}

/// The real zeros of `u` in `x` under the facts (`_real_zeros`).
fn real_zeros(u: &Basic, x: &Basic, facts: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let mut u: Basic = u.clone();
    while u.is_function(Function::Abs) {
        u = u.args()[0].clone();
    }
    if ask(&gt(&u, &zero())?, facts)? == Some(true) || ask(&lt(&u, &zero())?, facts)? == Some(true) {
        return Ok(Some(Vec::new()));
    }
    let r: Basic = reals();
    let zeros = attempt(timeout(), || {
        crate::assumptions::solve::solve(
            std::slice::from_ref(&u),
            Some(std::slice::from_ref(x)),
            facts,
            Some(&r),
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
    Ok(Some(zeros.args().to_vec()))
}

/// The real zeros of a target found first by a cheaper rule.
pub type Zeros<'a> = &'a dyn Fn(&Basic) -> PolyResult<Option<Vec<Basic>>>;

/// The real isolated singularities of `f` in `x` as SymPy's
/// `singularities` finds them, under the assumptions (`singularities`).
pub fn singularities(
    f: &Basic,
    x: &Basic,
    assumptions: &[Basic],
    zeros: Option<Zeros<'_>>,
) -> PolyResult<Option<Vec<Basic>>> {
    use crate::core::rewriting::{rewrite, RewriteHints};
    let hints = RewriteHints::default();
    let e: Basic = rewrite(
        f,
        &[Function::Sec, Function::Csc, Function::Cot, Function::Tan],
        "cos",
        &hints,
    )?;
    let e: Basic = rewrite(
        &e,
        &[Function::Sech, Function::Csch, Function::Coth, Function::Tanh],
        "cosh",
        &hints,
    )?;
    let mut targets: Vec<Basic> = Vec::new();
    for node in atoms_where(&e, &|n| n.is_pow()) {
        let exponent: &Basic = &node.args()[1];
        if yes(exponent, Predicate::infinite) {
            return Ok(None);
        }
        if yes(exponent, Predicate::negative) {
            targets.push(node.args()[0].clone());
        }
    }
    for node in atoms_where(f, &|n| is_fn(n, &[Function::Log, Function::Asech, Function::Acsch])) {
        targets.push(node.args()[0].clone());
    }
    for node in atoms_where(f, &|n| is_fn(n, &[Function::Atanh, Function::Acoth])) {
        targets.push(s2(&node.args()[0], &one())?);
        targets.push(a2(&node.args()[0], &one())?);
    }
    let mut points: Vec<Basic> = Vec::new();
    for u in &targets {
        if !u.has(x) {
            continue;
        }
        let mut found: Option<Vec<Basic>> = match zeros {
            Some(z) => z(u)?,
            None => None,
        };
        if found.is_none() {
            found = real_zeros(u, x, assumptions)?;
        }
        let Some(found) = found else {
            return Ok(None);
        };
        for p in found {
            if !points.contains(&p) {
                points.push(p);
            }
        }
    }
    Ok(Some(points))
}

/// The points of `(a, b)` where the antiderivative `F` may be
/// discontinuous (`discontinuities`).
pub fn discontinuities(
    big_f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Vec<Basic>>> {
    let bounded: Vec<Basic> = bounded(assumptions, x, a, b)?;
    let Some(mut candidates) = singularities(big_f, x, &bounded, None)? else {
        return Ok(None);
    };
    let mut arguments: Vec<Basic> = Vec::new();
    for node in atoms_where(big_f, &|n| is_fn(n, &[Function::Atan, Function::Acot])) {
        if node.has(x) {
            let (_, denominator) = crate::core::expr::as_numer_denom(&node.args()[0])?;
            if denominator.has(x) {
                arguments.push(denominator);
            }
        }
    }
    for node in atoms_where(big_f, &|n| is_fn(n, &[Function::Log])) {
        if node.has(x) {
            arguments.push(node.args()[0].clone());
        }
    }
    for node in atoms_where(big_f, &|n| {
        is_fn(n, &[Function::Abs, Function::Sign, Function::Heaviside])
    }) {
        if node.has(x) {
            arguments.push(node.args()[0].clone());
        }
    }
    for node in atoms_where(big_f, &|n| matches!(n.head(), Head::Piecewise)) {
        for pair in node.args() {
            if !matches!(pair.head(), Head::ExprCondPair) {
                continue;
            }
            for relation in atoms_where(&pair.args()[1], &|n| matches!(n.head(), Head::Relational(_))) {
                let difference: Basic = s2(&relation.args()[0], &relation.args()[1])?;
                if difference.has(x) {
                    arguments.push(difference);
                }
            }
        }
    }
    for u in &arguments {
        let Some(zeros) = real_zeros(u, x, &bounded)? else {
            return Ok(None);
        };
        candidates.extend(zeros);
    }
    points_in(&candidates, a, b, assumptions)
}

fn bounded(assumptions: &[Basic], x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Vec<Basic>> {
    let mut facts: Vec<Basic> = Vec::new();
    if *a != neg_oo() {
        facts.push(gt(x, a)?);
    }
    if *b != oo() {
        facts.push(lt(x, b)?);
    }
    facts.extend(assumptions.iter().cloned());
    Ok(facts)
}

/// The branch of a `Piecewise` antiderivative which holds on `(a, b)`
/// (`select_branch`).
pub fn select_branch(big_f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptR {
    let sample: Basic = if *a == neg_oo() && *b == oo() {
        zero()
    } else if *a == neg_oo() {
        s2(b, &one())?
    } else if *b == oo() {
        a2(a, &one())?
    } else {
        d2(&a2(a, b)?, &int(2))?
    };
    let mut result: Basic = big_f.clone();
    let r: Basic = reals();
    for node in atoms_where(big_f, &|n| matches!(n.head(), Head::Piecewise)) {
        let mut chosen: Option<Basic> = None;
        for pair in node.args() {
            if !matches!(pair.head(), Head::ExprCondPair) {
                return Ok(None);
            }
            let (value, condition) = (&pair.args()[0], &pair.args()[1]);
            if *condition == crate::core::logic::true_() {
                chosen = Some(value.clone());
                break;
            }
            // the same branch must hold on the whole range
            for atom in atoms_where(condition, &|n| matches!(n.head(), Head::Relational(_))) {
                let facts: Vec<Basic> = bounded(assumptions, x, a, b)?;
                let difference: Basic = s2(&atom.args()[0], &atom.args()[1])?;
                let zeros = attempt(timeout(), || {
                    crate::assumptions::solve::solve(
                        &[difference],
                        Some(std::slice::from_ref(x)),
                        &facts,
                        Some(&r),
                        false,
                    )
                })?;
                if zeros != Some(crate::sets::sets::empty_set()) {
                    return Ok(None);
                }
            }
            let Some(holds) = ask(&subs(condition, x, &sample)?, assumptions)? else {
                return Ok(None);
            };
            if holds {
                chosen = Some(value.clone());
                break;
            }
        }
        let Some(chosen) = chosen else {
            return Ok(None);
        };
        result = xreplace(&result, &[(node, chosen)])?;
    }
    Ok(Some(result))
}

/// Whether `condition` is built of equations free of `x`
/// (`_equalities_of_parameters`).
fn equalities_of_parameters(condition: &Basic, x: &Basic) -> bool {
    if condition.has(x) {
        return false;
    }
    if matches!(condition.head(), Head::Relational(crate::core::basic::RelOp::Eq)) {
        return true;
    }
    if matches!(condition.head(), Head::And | Head::Or) {
        return condition.args().iter().all(|c| equalities_of_parameters(c, x));
    }
    false
}

/// `F` with each `Piecewise` whose cases before the last are isolated
/// values of the parameters replaced by its last branch
/// (`generic_antiderivative`).
pub fn generic_antiderivative(big_f: &Basic, x: &Basic) -> R {
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for node in atoms_where(big_f, &|n| matches!(n.head(), Head::Piecewise)) {
        let pairs: Vec<Basic> = node
            .args()
            .iter()
            .filter(|p| matches!(p.head(), Head::ExprCondPair))
            .cloned()
            .collect();
        if pairs.len() != node.args().len()
            || pairs.len() < 2
            || pairs[pairs.len() - 1].args()[1] != crate::core::logic::true_()
        {
            continue;
        }
        if pairs[..pairs.len() - 1]
            .iter()
            .all(|p| equalities_of_parameters(&p.args()[1], x))
        {
            replacement.push((node.clone(), pairs[pairs.len() - 1].args()[0].clone()));
        }
    }
    if replacement.is_empty() {
        return Ok(big_f.clone());
    }
    let generic: Basic = xreplace(big_f, &replacement)?;
    if generic == *big_f {
        return Ok(generic);
    }
    generic_antiderivative(&generic, x)
}

/// Whether `F`, as a function of a real `x`, is proved continuous at
/// `x = point` under the assumptions, by its tree (`continuous_at`).
pub fn continuous_at(big_f: &Basic, x: &Basic, point: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    let nonzero = |u: &Basic| -> PolyResult<bool> {
        let at: Basic = subs(u, x, point)?;
        Ok(ask(&gt(&at, &zero())?, assumptions)? == Some(true) || ask(&lt(&at, &zero())?, assumptions)? == Some(true))
    };
    let positive =
        |u: &Basic| -> PolyResult<bool> { Ok(ask(&gt(&subs(u, x, point)?, &zero())?, assumptions)? == Some(true)) };
    let real = |u: &Basic| -> PolyResult<bool> {
        let q: Basic = crate::assumptions::facts::element(&subs(u, x, point)?, &reals())?;
        Ok(ask(&q, assumptions)? == Some(true))
    };
    fn continuous(
        e: &Basic,
        x: &Basic,
        nonzero: &dyn Fn(&Basic) -> PolyResult<bool>,
        positive: &dyn Fn(&Basic) -> PolyResult<bool>,
        real: &dyn Fn(&Basic) -> PolyResult<bool>,
    ) -> PolyResult<bool> {
        if !e.has(x) || e == x {
            return Ok(true);
        }
        if e.is_add() || e.is_mul() {
            for arg in e.args() {
                if !continuous(arg, x, nonzero, positive, real)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if e.is_pow() {
            let (base, exponent) = (&e.args()[0], &e.args()[1]);
            if !continuous(base, x, nonzero, positive, real)? || !continuous(exponent, x, nonzero, positive, real)? {
                return Ok(false);
            }
            if exponent.has(x) || !exponent.as_rational().is_some_and(|q| q.is_int()) {
                return positive(base);
            }
            return Ok(yes(exponent, Predicate::nonnegative) || nonzero(base)?);
        }
        if !is_application(e) || e.args().len() != 1 {
            return Ok(false);
        }
        let u: &Basic = &e.args()[0];
        if !continuous(u, x, nonzero, positive, real)? {
            return Ok(false);
        }
        if is_fn(
            e,
            &[
                Function::Abs,
                Function::Exp,
                Function::Sin,
                Function::Cos,
                Function::Sinh,
                Function::Cosh,
            ],
        ) {
            return Ok(true);
        }
        if is_fn(e, &[Function::Atan, Function::Tanh, Function::Asinh]) {
            return real(u);
        }
        if e.is_function(Function::Log) {
            if u.is_function(Function::Abs) {
                return nonzero(&u.args()[0]);
            }
            return positive(u);
        }
        Ok(false)
    }
    continuous(big_f, x, &nonzero, &positive, &real)
}

fn limit(e: &Basic, x: &Basic, point: &Basic, direction: &str, assumptions: &[Basic]) -> R {
    crate::assumptions::limits::limit(e, x, point, direction, assumptions, None)
}

/// `lim F(x)` as `x` tends to `point` from the side `'+'` or `'-'`;
/// `oo` or `-oo` for a signed infinity, `None` when the limit does not
/// exist or cannot be found (`one_sided_limit`).
pub fn one_sided_limit(big_f: &Basic, x: &Basic, point: &Basic, direction: &str, assumptions: &[Basic]) -> OptR {
    let has_limit = |v: &Basic| has_any(v, &|n| matches!(n.head(), Head::Limit));
    let value: Option<Basic> = if *point == oo() || *point == neg_oo() {
        attempt(timeout(), || limit(big_f, x, point, "+", assumptions))?
    } else if !is_number(point) && continuous_at(big_f, x, point, assumptions)? {
        Some(subs(big_f, x, point)?)
    } else {
        let v: Option<Basic> = attempt(timeout(), || limit(big_f, x, point, direction, assumptions))?;
        if (v.is_none() || v.as_ref().is_some_and(has_limit)) && !is_number(point) {
            Some(subs(big_f, x, point)?)
        } else {
            v
        }
    };
    let Some(value) = value else {
        return Ok(None);
    };
    if value == oo() || value == neg_oo() {
        return Ok(Some(value));
    }
    if has_any(&value, &|n| {
        matches!(
            n.head(),
            Head::Infinity
                | Head::NegativeInfinity
                | Head::ComplexInfinity
                | Head::NaN
                | Head::Limit
                | Head::Piecewise
                | Head::AccumBounds
        )
    }) {
        return Ok(None);
    }
    let allowed = big_f.free_symbols();
    let own = point.free_symbols();
    if value
        .free_symbols()
        .iter()
        .any(|s| !allowed.contains(s) && !own.contains(s))
    {
        return Ok(None);
    }
    Ok(Some(value))
}

/// The antiderivative on its branch of the range (`_prepared`).
fn prepared(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    late: bool,
    known: Option<&Basic>,
    step: Option<AntiderivativeStep<'_>>,
) -> OptR {
    let big_f: Option<Basic> = match (known, step) {
        (Some(k), _) => Some(k.clone()),
        (None, Some(s)) => s(f, x, assumptions, late)?,
        (None, None) => antiderivative(f, x, assumptions, late)?,
    };
    let Some(big_f) = big_f else {
        return Ok(None);
    };
    let big_f: Basic = cancelled_logarithms(&big_f, x)?;
    if has_any(&big_f, &|n| matches!(n.head(), Head::Piecewise)) {
        return select_branch(&big_f, x, a, b, assumptions);
    }
    Ok(Some(big_f))
}

fn ops(e: &Basic) -> i64 {
    crate::core::function::count_ops(e).unwrap_or(0)
}

/// `F` with the rational arguments of its logarithms in `x` cancelled
/// when that makes them smaller (`cancelled_logarithms`).
pub fn cancelled_logarithms(big_f: &Basic, x: &Basic) -> R {
    let query = |node: &Basic| -> bool {
        node.is_function(Function::Log)
            && node.has(x)
            && crate::core::expr::polynomial_like(
                &node.args()[0],
                std::slice::from_ref(x),
                crate::core::expr::PolynomialKind::Rational,
            ) == Some(true)
    };
    let value = |node: &Basic| -> R {
        let argument: &Basic = &node.args()[0];
        let cancelled: Basic = cancel(argument)?;
        if ops(&cancelled) < ops(argument) {
            log(&cancelled)
        } else {
            Ok(node.clone())
        }
    };
    crate::core::function::replace_where(big_f, &query, &value)
}

fn cut_points(
    f: &Basic,
    big_f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Vec<Basic>>> {
    let Some(points) = discontinuities(big_f, x, a, b, assumptions)? else {
        return Ok(None);
    };
    let Some(mut singular) = singularities(f, x, &bounded(assumptions, x, a, b)?, None)? else {
        return Ok(None);
    };
    singular.extend(points);
    points_in(&singular, a, b, assumptions)
}

/// `Integral(f, (x, a, b))` from an antiderivative, cut at the
/// discontinuities and singularities inside the range and evaluated by
/// one-sided limits (`antiderivative_integral`).
pub fn antiderivative_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    late: bool,
    known: Option<&Basic>,
    step: Option<AntiderivativeStep<'_>>,
) -> PolyResult<Option<ConditionalValue>> {
    let Some(ready) = prepared(f, x, a, b, assumptions, late, known, step)? else {
        return Ok(None);
    };
    let big_f: Basic = real_logarithms(&ready, x)?;
    let Some(more) = cut_points(f, &big_f, x, a, b, assumptions)? else {
        return Ok(None);
    };
    let mut bounds: Vec<Basic> = vec![a.clone()];
    bounds.extend(more);
    bounds.push(b.clone());
    let mut total: Basic = zero();
    for w in bounds.windows(2) {
        let (lower, upper) = (&w[0], &w[1]);
        let Some(right) = one_sided_limit(&big_f, x, upper, "-", assumptions)? else {
            return Ok(None);
        };
        let Some(left) = one_sided_limit(&big_f, x, lower, "+", assumptions)? else {
            return Ok(None);
        };
        // an infinite limit is checked against the sign of the integrand
        if (right == oo() || right == neg_oo()) && !integrand_has_sign(f, x, upper, "-", &right, assumptions)? {
            return Ok(None);
        }
        if (left == oo() || left == neg_oo()) && !integrand_has_sign(f, x, lower, "+", &ng(&left)?, assumptions)? {
            return Ok(None);
        }
        total = a2(&total, &s2(&right, &left)?)?;
    }
    if has_any(&total, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity)) {
        return Ok(None);
    }
    if not_real_number(&total)? && real_on_range(f, x, &bounds)? {
        return Ok(None);
    }
    Ok(Some(ConditionalValue::new(
        absorbed(&total)?,
        crate::core::logic::true_(),
    )))
}

/// The value of `e` to 15 digits as `(re, im)`, `None` when unreliable.
fn value15(e: &Basic) -> Option<(f64, f64)> {
    let v = crate::numeric::reliable_value(e, 15, &[])?;
    let (a, b) = crate::core::evalf::parts(&v);
    Some((a.to_f64().value(), b.to_f64().value()))
}

/// Whether `f` without parameters is real at a point inside every piece
/// between the bounds (`_real_on_range`).
fn real_on_range(f: &Basic, x: &Basic, bounds: &[Basic]) -> PolyResult<bool> {
    if f.free_symbols().iter().any(|s| s != x) {
        return Ok(false);
    }
    for w in bounds.windows(2) {
        let (lower, upper) = (&w[0], &w[1]);
        if !(is_number(lower) && is_number(upper)) {
            return Ok(false);
        }
        let sample: Basic = if *lower == neg_oo() && *upper == oo() {
            zero()
        } else if *lower == neg_oo() {
            s2(upper, &one())?
        } else if *upper == oo() {
            a2(lower, &one())?
        } else {
            d2(&a2(lower, upper)?, &int(2))?
        };
        let at: Basic = match subs(f, x, &sample) {
            Ok(v) => v,
            Err(PolyError::ZeroDivision(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        match value15(&at) {
            Some((_, 0.0)) => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// Whether `total` has an imaginary part known to be nonzero
/// (`_not_real_number`).
fn not_real_number(total: &Basic) -> PolyResult<bool> {
    let imaginary: Basic = im(total)?;
    Ok(!has_fn(&imaginary, &[Function::Im]) && is(&imaginary, Predicate::zero) == Some(false))
}

/// `log(u)` written `log(Abs(u))` for the real arguments `u` depending on
/// `x` (`real_logarithms`).
pub fn real_logarithms(big_f: &Basic, x: &Basic) -> R {
    let real: Basic = fresh(&symbol_name(x), &[(Predicate::real, true)]);
    let query = |node: &Basic| -> bool {
        if !node.is_function(Function::Log) || !node.has(x) || node.args()[0].is_function(Function::Abs) {
            return false;
        }
        match subs(&node.args()[0], x, &real) {
            Ok(v) => yes(&v, Predicate::extended_real),
            Err(_) => false,
        }
    };
    let value = |node: &Basic| -> R { log(&abs_(&node.args()[0])?) };
    crate::core::function::replace_where(big_f, &query, &value)
}

/// `total` with the finite numbers of a sum absorbed into its infinity
/// (`_absorbed`).
fn absorbed(total: &Basic) -> R {
    if *total == oo() || *total == neg_oo() || !has_infinities(total) {
        return Ok(total.clone());
    }
    let terms: Vec<Basic> = crate::core::expr::terms_of(total);
    let infinite: Vec<&Basic> = terms.iter().filter(|t| has_infinities(t)).collect();
    let finite: Vec<&Basic> = terms.iter().filter(|t| !has_infinities(t)).collect();
    if finite.is_empty() || !finite.iter().all(|t| is_number(t)) {
        return Ok(total.clone());
    }
    if infinite.iter().all(|t| **t == oo()) {
        return Ok(oo());
    }
    if infinite.iter().all(|t| **t == neg_oo()) {
        return Ok(neg_oo());
    }
    Ok(total.clone())
}

/// Whether `f` has the sign of `infinity` at three points approaching
/// `point` from `side`, the parameters sampled under the assumptions
/// (`_integrand_has_sign`).
fn integrand_has_sign(
    f: &Basic,
    x: &Basic,
    point: &Basic,
    side: &str,
    infinity: &Basic,
    assumptions: &[Basic],
) -> PolyResult<bool> {
    let mut parameters: Vec<Basic> = f
        .free_symbols()
        .union(&point.free_symbols())
        .filter(|s| *s != x)
        .cloned()
        .collect();
    parameters.sort_by_key(symbol_name);
    let symbols: Vec<Basic> = parameters.into_iter().filter(is_symbol).collect();
    let values: Vec<(Basic, Basic)> = if symbols.is_empty() {
        Vec::new()
    } else {
        let mut rng = crate::random::PyRandom::from_string(&format!("({}, {})", f, point));
        match crate::integrals::conditions::sample_values(&symbols, assumptions, &mut rng)? {
            None => return Ok(true),
            Some(v) => v,
        }
    };
    let g: Basic = xreplace(f, &values)?;
    let p: Basic = xreplace(point, &values)?;
    let samples: Vec<Basic> = if p == oo() {
        vec![int(10), int(100), int(1000)]
    } else if p == neg_oo() {
        vec![int(-10), int(-100), int(-1000)]
    } else {
        let step: i64 = if side == "+" { 1 } else { -1 };
        let mut v: Vec<Basic> = Vec::new();
        for k in [2u32, 4, 6] {
            let delta: Basic = Basic::from_rational(Rat::from_parts(
                Int::from(step),
                dashu_int::UBig::from(10u8).pow(k as usize),
            ));
            v.push(a2(&p, &delta)?);
        }
        v
    };
    let positive: bool = *infinity == oo();
    for sample in samples {
        let at: Basic = match subs(&g, x, &sample) {
            Ok(v) => v,
            Err(PolyError::ZeroDivision(_)) => return Ok(true),
            Err(e) => return Err(e),
        };
        let Some((re_part, im_part)) = value15(&at) else {
            return Ok(true);
        };
        if im_part != 0.0 {
            return Ok(true);
        }
        let wrong: bool = if positive { re_part < 0.0 } else { re_part > 0.0 };
        if wrong {
            return Ok(false);
        }
    }
    Ok(true)
}

use crate::numbers::{Int, Rat};

/// The jump of the antiderivative across a point, for the principal value
/// and the finite part.
type Jump<'a> = &'a dyn Fn(&Basic, &Basic) -> OptR;

/// The limits at the ends minus the jump of the antiderivative across
/// every point inside (`_regularized`).
fn regularized(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    step: Option<AntiderivativeStep<'_>>,
    jump: Jump<'_>,
) -> PolyResult<Option<ConditionalValue>> {
    let Some(big_f) = prepared(f, x, a, b, assumptions, false, None, step)? else {
        return Ok(None);
    };
    let Some(inside) = cut_points(f, &big_f, x, a, b, assumptions)? else {
        return Ok(None);
    };
    let right = one_sided_limit(&big_f, x, b, "-", assumptions)?;
    let left = one_sided_limit(&big_f, x, a, "+", assumptions)?;
    let (Some(right), Some(left)) = (right, left) else {
        return Ok(None);
    };
    if right == oo() || right == neg_oo() || left == oo() || left == neg_oo() {
        return Ok(None);
    }
    let mut total: Basic = s2(&right, &left)?;
    let epsilon: Basic = fresh("epsilon", &[(Predicate::positive, true)]);
    let allowed = big_f.free_symbols();
    for c in &inside {
        let excised: Basic = s2(
            &subs(&big_f, x, &a2(c, &epsilon)?)?,
            &subs(&big_f, x, &s2(c, &epsilon)?)?,
        )?;
        let Some(value) = jump(&excised, &epsilon)? else {
            return Ok(None);
        };
        if value.free_symbols().iter().any(|s| !allowed.contains(s)) {
            return Ok(None);
        }
        total = s2(&total, &value)?;
    }
    Ok(Some(ConditionalValue::new(total, crate::core::logic::true_())))
}

/// Cauchy's principal value of `Integral(f, (x, a, b))` from an
/// antiderivative (`principal_value_integral`).
pub fn principal_value_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    step: Option<AntiderivativeStep<'_>>,
) -> PolyResult<Option<ConditionalValue>> {
    let jump = |excised: &Basic, epsilon: &Basic| -> OptR {
        let value: Option<Basic> = attempt(timeout(), || limit(excised, epsilon, &zero(), "+", assumptions))?;
        let Some(value) = value else {
            return Ok(None);
        };
        if has_any(&value, &|n| {
            matches!(
                n.head(),
                Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity | Head::NaN | Head::Limit
            )
        }) {
            return Ok(None);
        }
        Ok(Some(value))
    };
    regularized(f, x, a, b, assumptions, step, &jump)
}

/// The constant term of the expansion of `jump` at `epsilon = 0`
/// (`_finite_part`).
fn finite_part(jump: &Basic, epsilon: &Basic) -> OptR {
    let expansion: Option<Basic> = attempt(timeout(), || {
        crate::core::order::remove_o(&crate::core::series::series(
            jump,
            Some(epsilon),
            &zero(),
            1,
            "+",
            None,
            &zero(),
        )?)
    })?;
    let Some(expansion) = expansion else {
        return Ok(None);
    };
    let mut constant: Basic = zero();
    for term in crate::core::expr::terms_of(&crate::core::function::expand_method(&expansion)?) {
        if !term.has(epsilon) {
            constant = a2(&constant, &term)?;
        }
    }
    if has_any(&constant, &|n| {
        matches!(
            n.head(),
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
        )
    }) {
        return Ok(None);
    }
    Ok(Some(constant))
}

/// Hadamard's finite part of `Integral(f, (x, a, b))`
/// (`finite_part_integral`).
pub fn finite_part_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    step: Option<AntiderivativeStep<'_>>,
) -> PolyResult<Option<ConditionalValue>> {
    let jump = |excised: &Basic, epsilon: &Basic| -> OptR { finite_part(excised, epsilon) };
    regularized(f, x, a, b, assumptions, step, &jump)
}
