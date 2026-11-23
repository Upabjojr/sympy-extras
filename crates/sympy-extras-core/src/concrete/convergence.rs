//! Port of `sympy_extras/concrete/convergence.py`: the condition on the
//! parameters under which `Sum(a_n, (n, n0, oo))` converges (the
//! counterpart of Mathematica's `SumConvergence`): the comparison with a
//! rational majorant, the term test, d'Alembert's ratio test with the
//! boundary `r = 1` examined at its points, Raabe's and Bertrand's tests,
//! Cauchy's root test, the comparison with the `p`-series, Leibniz's and
//! Dirichlet's tests, the integral test and the port of SymPy's
//! `Sum.is_convergent` for terms without parameters; the convergence of
//! infinite products.

// PORTED FROM SYMPY: sympy/concrete/summations.py (Sum.is_convergent), sympy/calculus/singularities.py
// (is_decreasing) (SymPy 1.14), through the Python port of sympy_extras.
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::PolyResult;

type R = PolyResult<Basic>;
/// A condition on the parameters, `None` when undecided (`Condition`).
pub type Condition = Option<Basic>;
type C = PolyResult<Condition>;

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
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn abs_(e: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::derivative(e, x)
}
fn true_() -> Basic {
    crate::core::logic::true_()
}
fn false_() -> Basic {
    crate::core::logic::false_()
}
fn is_true(b: &Basic) -> bool {
    *b == true_()
}
fn is_false(b: &Basic) -> bool {
    *b == false_()
}
fn and_(args: &[Basic]) -> Basic {
    crate::core::logic::and_(args)
}
fn or_(args: &[Basic]) -> Basic {
    crate::core::logic::or_(args)
}
fn rel(op: RelOp, a: &Basic, b: &Basic) -> R {
    use crate::core::relational as r;
    match op {
        RelOp::Eq => r::eq(a, b),
        RelOp::Ne => r::ne(a, b),
        RelOp::Lt => r::lt(a, b),
        RelOp::Le => r::le(a, b),
        RelOp::Gt => r::gt(a, b),
        RelOp::Ge => r::ge(a, b),
    }
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn within<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn is_number(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}
fn has_head(e: &Basic, f: &dyn Fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| f(n.head()))
}
fn has_function(e: &Basic, fs: &[Function]) -> bool {
    has_head(e, &|h| matches!(h, Head::Function(f) if fs.contains(f)))
}
fn is_infinity(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn dummy(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    crate::core::symbol::Dummy::new(name, assumptions).0
}
fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(data) | Head::Dummy(data, _) => data.name.clone(),
        _ => s.to_string(),
    }
}
fn positive_integer_dummy(name: &str) -> Basic {
    dummy(name, &[(Predicate::integer, true), (Predicate::positive, true)])
}
fn interval_from(start: &Basic) -> R {
    crate::sets::sets::interval(start, &oo(), false, false)
}

/// The assumptions on the parameters, the summation variable and its
/// lower bound, and a recursion depth for the case distinctions
/// (`_Context`).
#[derive(Clone)]
struct Context {
    n: Basic,
    lower: Basic,
    assumptions: Vec<Basic>,
    depth: usize,
}

impl Context {
    fn assume(&self, condition: &Basic) -> Context {
        let mut assumptions: Vec<Basic> = self.assumptions.clone();
        assumptions.push(condition.clone());
        Context {
            n: self.n.clone(),
            lower: self.lower.clone(),
            assumptions,
            depth: self.depth + 1,
        }
    }

    fn premise(&self) -> Basic {
        if self.assumptions.is_empty() {
            true_()
        } else {
            and_(&self.assumptions)
        }
    }

    fn ask(&self, statement: &Basic) -> PolyResult<Option<bool>> {
        crate::assumptions::ask::ask(statement, &[self.premise()], None)
    }

    /// The condition as `true`, `false`, or itself refined (`decide`).
    fn decide(&self, condition: &Basic) -> R {
        match self.ask(condition)? {
            Some(true) => return Ok(true_()),
            Some(false) => return Ok(false_()),
            None => {}
        }
        let premise: Basic = self.premise();
        let refined: Option<Basic> =
            within(|| crate::assumptions::refine::simplify(condition, &[premise], None, 1.7, false, true))?;
        Ok(refined.unwrap_or_else(|| condition.clone()))
    }

    /// Whether a statement in `n` holds for all large `n` (`eventually`).
    fn eventually(&self, statement: &Basic) -> PolyResult<Option<bool>> {
        for shift in [0, 4, 16, 64] {
            let start: Basic = add(&[self.lower.clone(), int(shift)])?;
            let premise: Basic = and_(&[self.premise(), rel(RelOp::Ge, &self.n, &start)?]);
            let symbols: Vec<Basic> = statement.free_symbols().into_iter().collect();
            let facts = match crate::assumptions::ask::facts_of(std::slice::from_ref(&premise), None, &symbols) {
                Ok(f) => f,
                Err(crate::error::PolyError::Value(_)) => return Ok(None),
                Err(e) => return Err(e),
            };
            let (abstracted, _, _) = crate::assumptions::refine::Refiner::new(facts).abstract_(statement)?;
            if is_true(&abstracted) {
                return Ok(Some(true));
            }
            if crate::assumptions::ask::ask(statement, &[premise], None)? == Some(true) {
                return Ok(Some(true));
            }
        }
        Ok(None)
    }
}

/// `e` with the keys of `rule` replaced, evaluated (`_replaced`).
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

/// The free symbols of `e` other than `n` (`_parameters`).
fn parameters(e: &Basic, n: &Basic) -> BTreeSet<Basic> {
    e.free_symbols()
        .into_iter()
        .filter(|s| s.is_symbol() && s != n)
        .collect()
}

/// `n` a positive integer dummy and the parameters without a sign real
/// dummies (`_real_dummies`).
fn real_dummies(e: &Basic, n: &Basic) -> Vec<(Basic, Basic)> {
    let mut forward: Vec<(Basic, Basic)> = vec![(n.clone(), positive_integer_dummy("n"))];
    for parameter in parameters(e, n) {
        if is(&parameter, Predicate::real).is_none() {
            let name: String = symbol_name(&parameter);
            forward.push((parameter, dummy(&name, &[(Predicate::real, true)])));
        }
    }
    forward
}

fn inverted(rule: &[(Basic, Basic)]) -> Vec<(Basic, Basic)> {
    rule.iter().map(|(s, d)| (d.clone(), s.clone())).collect()
}

/// A ratio or a power of terms simplified before its limit is taken
/// (`_clean`).
fn clean(e: &Basic, n: &Basic) -> R {
    let forward = real_dummies(e, n);
    let r: Basic = replaced(e, &forward)?;
    let cleaned: Option<Basic> = within(|| {
        let p: Basic = crate::simplify::powsimp::powsimp(
            &r,
            false,
            crate::simplify::powsimp::Combine::All,
            true,
            &crate::core::function::count_ops,
        )?;
        simplify(&crate::simplify::gammasimp::combsimp(&p)?)
    })?;
    match cleaned {
        None => Ok(e.clone()),
        Some(c) => replaced(&c, &inverted(&forward)),
    }
}

/// The limit at infinity under the assumptions (`_limit`).
fn limit(e: &Basic, ctx: &Context, cleaning: bool) -> PolyResult<Option<Basic>> {
    let cleaned: Basic = if cleaning { clean(e, &ctx.n)? } else { e.clone() };
    let premise: Basic = ctx.premise();
    let value: Option<Basic> =
        within(|| crate::assumptions::limits::limit(&cleaned, &ctx.n, &oo(), "+", &[premise], None))?;
    Ok(value.filter(|v| !has_head(v, &|h| matches!(h, Head::Limit)) && !v.has(&ctx.n)))
}

type Verdict<'a> = &'a dyn Fn(&Basic, &Context) -> C;

/// A verdict on a limit which may be a `Piecewise` in the parameters
/// (`_by_cases`).
fn by_cases(value: &Basic, ctx: &Context, verdict: Verdict) -> C {
    if !matches!(value.head(), Head::Piecewise) {
        return verdict(value, ctx);
    }
    if ctx.depth > 4 {
        return Ok(None);
    }
    let mut parts: Vec<Basic> = Vec::new();
    for pair in value.args() {
        let (e, c) = (&pair.args()[0], &pair.args()[1]);
        let Some(part) = verdict(e, &ctx.assume(c))? else {
            return Ok(None);
        };
        if !is_false(&part) {
            parts.push(and_(&[c.clone(), part]));
        }
    }
    Ok(Some(if parts.is_empty() { false_() } else { or_(&parts) }))
}

fn usable(c: &Condition) -> bool {
    matches!(c, Some(b) if !is_false(b))
}

/// The condition from a limit compared with 1 (`_compare_with_one`).
fn compare_with_one(
    value: &Basic,
    ctx: &Context,
    above: Condition,
    below: Condition,
    boundary: &dyn Fn(&Context) -> C,
    term: &Basic,
) -> C {
    if is_infinity(value) {
        return Ok(above);
    }
    if matches!(value.head(), Head::NegativeInfinity) {
        return Ok(below);
    }
    if value.free_symbols().is_empty() {
        return match crate::assumptions::analysis::certified_sign(&sub(value, &int(1))?)? {
            None => Ok(None),
            Some(s) if s < 0 => Ok(below),
            Some(s) if s > 0 => Ok(above),
            Some(_) => boundary(ctx),
        };
    }
    let less: Basic = ctx.decide(&rel(RelOp::Lt, value, &int(1))?)?;
    if is_true(&less) {
        return Ok(below);
    }
    let greater: Basic = ctx.decide(&rel(RelOp::Gt, value, &int(1))?)?;
    if is_true(&greater) {
        return Ok(above);
    }
    let equal: Basic = ctx.decide(&rel(RelOp::Eq, value, &int(1))?)?;
    if is_true(&equal) {
        return boundary(ctx);
    }
    let mut parts: Vec<Basic> = Vec::new();
    if !is_false(&less) && usable(&below) {
        parts.push(and_(&[less.clone(), below.clone().unwrap_or_else(false_)]));
    }
    if !is_false(&greater) && usable(&above) {
        parts.push(and_(&[greater.clone(), above.clone().unwrap_or_else(false_)]));
    }
    if !is_false(&equal) {
        let Some(on_boundary) = boundary_points(value, ctx, term)? else {
            return Ok(None);
        };
        if !is_false(&on_boundary) {
            parts.push(on_boundary);
        }
    }
    if (below.is_none() && !is_false(&less)) || (above.is_none() && !is_false(&greater)) {
        return Ok(None);
    }
    Ok(Some(if parts.is_empty() { false_() } else { or_(&parts) }))
}

/// The condition for convergence on `value = 1`, the points of the
/// boundary listed for one parameter (`_boundary_points`).
fn boundary_points(value: &Basic, ctx: &Context, term: &Basic) -> C {
    let mut found: Vec<Basic> = parameters(value, &ctx.n).into_iter().collect();
    found.sort_by_key(symbol_name);
    if found.len() != 1 || value.free_symbols().len() != 1 || ctx.depth > 4 {
        return Ok(None);
    }
    let p: Basic = found.remove(0);
    let equation: Basic = rel(RelOp::Eq, value, &int(1))?;
    let premise: Basic = ctx.premise();
    let points: Option<Basic> = within(|| {
        crate::assumptions::solve::solve(
            std::slice::from_ref(&equation),
            Some(std::slice::from_ref(&p)),
            &[premise],
            Some(&crate::sets::sets::reals()),
            false,
        )
    })?;
    let Some(points) = points.filter(|s| matches!(s.head(), Head::FiniteSet)) else {
        return Ok(None);
    };
    let mut parts: Vec<Basic> = Vec::new();
    for point in points.args() {
        if !point.free_symbols().is_empty() {
            return Ok(None);
        }
        let reduced: Context = Context {
            n: ctx.n.clone(),
            lower: ctx.lower.clone(),
            assumptions: ctx
                .assumptions
                .iter()
                .filter(|a| !a.free_symbols().contains(&p))
                .cloned()
                .collect(),
            depth: ctx.depth + 1,
        };
        let Some(at_point) = convergence(&replaced(term, &[(p.clone(), point.clone())])?, &reduced)? else {
            return Ok(None);
        };
        let here: Basic = rel(RelOp::Eq, &p, point)?;
        if is_true(&at_point) {
            parts.push(here);
        } else if !is_false(&at_point) {
            parts.push(and_(&[here, at_point]));
        }
    }
    Ok(Some(if parts.is_empty() { false_() } else { or_(&parts) }))
}

/// The factors of a product, the term itself otherwise (`_factors`).
fn factors(term: &Basic) -> Vec<Basic> {
    if term.is_mul() {
        term.args().to_vec()
    } else {
        vec![term.clone()]
    }
}

fn is_polynomial_in(e: &Basic, n: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(n),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}

/// `b_n` when `term` is `(-1)**(n + c)*b_n` or `cos(pi*n)*b_n`
/// (`_alternating`).
fn alternating(term: &Basic, n: &Basic) -> PolyResult<Option<Basic>> {
    let mut rest: Vec<Basic> = Vec::new();
    let mut found: bool = false;
    for f in factors(term) {
        if !found && f.is_pow() && f.args()[0] == Basic::negative_one() && is_polynomial_in(&f.args()[1], n) {
            let slope: Basic = diff(&f.args()[1], n)?;
            if integer_of(&slope).is_some_and(|s| s.abs() == 1) {
                found = true;
                continue;
            }
        }
        if !found
            && f.is_function(Function::Cos)
            && integer_of(&sub(&f.args()[0], &mul(&[Basic::pi(), n.clone()])?)?).is_some()
        {
            found = true;
            continue;
        }
        rest.push(f);
    }
    if !found {
        return Ok(None);
    }
    Ok(Some(mul(&rest)?))
}

/// `b_n` when `term` is `sin(c*n + d)*b_n` or `cos(c*n + d)*b_n` with `c`
/// real and not a multiple of `2*pi` (`_oscillating`).
fn oscillating(term: &Basic, n: &Basic) -> PolyResult<Option<Basic>> {
    let mut rest: Vec<Basic> = Vec::new();
    let mut found: bool = false;
    for f in factors(term) {
        if !found && (f.is_function(Function::Sin) || f.is_function(Function::Cos)) {
            let argument: &Basic = &f.args()[0];
            let c: Basic = diff(argument, n)?;
            if !c.free_symbols().contains(n)
                && diff(&sub(argument, &mul(&[c.clone(), n.clone()])?)?, n)? == Basic::zero()
                && is(&c, Predicate::real) == Some(true)
                && is(&div(&c, &mul(&[int(2), Basic::pi()])?)?, Predicate::integer) == Some(false)
            {
                found = true;
                continue;
            }
        }
        rest.push(f);
    }
    if !found {
        return Ok(None);
    }
    Ok(Some(mul(&rest)?))
}

/// Dirichlet's test (`_dirichlet_test`).
fn dirichlet_test(term: &Basic, ctx: &Context) -> C {
    let Some(b) = oscillating(term, &ctx.n)? else {
        return Ok(None);
    };
    Ok(match decreasing_to_zero(&b, ctx)? {
        Some(c) if is_true(&c) => Some(true_()),
        _ => None,
    })
}

/// The nonzero coefficients of `e` as a polynomial in `n` by degree
/// (`_coefficients`).
fn coefficients(e: &Basic, n: &Basic) -> PolyResult<Option<Vec<(i64, Basic)>>> {
    let mut found: Vec<(i64, Basic)> = Vec::new();
    let expanded: Basic = crate::core::function::expand_default(e)?;
    let summands: Vec<Basic> = if expanded.is_add() {
        expanded.args().to_vec()
    } else {
        vec![expanded]
    };
    for summand in summands {
        let (c, rest) = crate::core::expr::as_independent(&summand, n, false)?;
        let k: i64 = if rest == Basic::one() {
            0
        } else if rest == *n {
            1
        } else if rest.is_pow() && rest.args()[0] == *n && integer_of(&rest.args()[1]).is_some_and(|k| k > 0) {
            integer_of(&rest.args()[1]).unwrap_or(0)
        } else {
            return Ok(None);
        };
        match found.iter_mut().find(|(d, _)| *d == k) {
            Some((_, total)) => *total = add(&[total.clone(), c])?,
            None => found.push((k, c)),
        }
    }
    Ok(Some(found.into_iter().filter(|(_, c)| *c != Basic::zero()).collect()))
}

/// Whether `e` is a polynomial in `n` with real number coefficients
/// (`_real_polynomial`).
fn real_polynomial(e: &Basic, n: &Basic) -> PolyResult<bool> {
    if e.free_symbols().iter().any(|s| s != n) {
        return Ok(false);
    }
    Ok(match coefficients(e, n)? {
        None => false,
        Some(cs) => cs.iter().all(|(_, c)| is(c, Predicate::real) == Some(true)),
    })
}

/// A bound of `|term|` with the oscillating factors replaced by `1` and
/// the factors free of `n` dropped (`_majorant`).
fn majorant(term: &Basic, n: &Basic) -> PolyResult<Option<Basic>> {
    if !term.free_symbols().contains(n) {
        return Ok(if is_number(term) { Some(abs_(term)?) } else { None });
    }
    let signed_powers: bool = term
        .atoms(Some(&|p: &Basic| p.is_pow()))
        .iter()
        .any(|p| p.args()[0] == Basic::negative_one());
    if !has_function(term, &[Function::Sin, Function::Cos]) && !signed_powers {
        return Ok(Some(abs_(term)?));
    }
    if term.is_function(Function::Abs) {
        return majorant(&term.args()[0], n);
    }
    if term.is_function(Function::Sin) || term.is_function(Function::Cos) {
        return Ok(if real_polynomial(&term.args()[0], n)? {
            Some(Basic::one())
        } else {
            None
        });
    }
    if term.is_pow() {
        let (base, exponent) = (&term.args()[0], &term.args()[1]);
        if *base == Basic::negative_one() {
            return Ok(if real_polynomial(exponent, n)? {
                Some(Basic::one())
            } else {
                None
            });
        }
        if integer_of(exponent).is_some_and(|k| k > 0) {
            return match majorant(base, n)? {
                None => Ok(None),
                Some(bound) => Ok(Some(pow(&bound, exponent)?)),
            };
        }
        return Ok(None);
    }
    if term.is_mul() || term.is_add() {
        let mut bounds: Vec<Basic> = Vec::new();
        for a in term.args() {
            if term.is_mul() && !a.free_symbols().contains(n) {
                continue;
            }
            let Some(bound) = majorant(a, n)? else {
                return Ok(None);
            };
            bounds.push(bound);
        }
        return Ok(Some(if term.is_mul() { mul(&bounds)? } else { add(&bounds)? }));
    }
    Ok(None)
}

/// `e` with every `Abs(u)` replaced by `u` (`_without_abs_values`).
fn without_abs_values(e: &Basic) -> R {
    if e.args().is_empty() || e.is_atom() {
        return Ok(e.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(without_abs_values(a)?);
    }
    let rebuilt: Basic = if args.iter().zip(e.args()).any(|(new, old)| new != old) {
        crate::core::function::rebuild_evaluated(e, args)?
    } else {
        e.clone()
    };
    if rebuilt.is_function(Function::Abs) {
        return Ok(rebuilt.args()[0].clone());
    }
    Ok(rebuilt)
}

/// Absolute convergence by comparison with a rational majorant decaying
/// at least like `1/n**2` (`_rational_comparison`).
fn rational_comparison(term: &Basic, ctx: &Context) -> C {
    let Some(bound) = majorant(term, &ctx.n)? else {
        return Ok(None);
    };
    if bound.free_symbols().iter().any(|s| *s != ctx.n) {
        return Ok(None);
    }
    let rational: Basic = without_abs_values(&bound)?;
    let Some(cancelled) = within(|| crate::polys::polytools::cancel(&rational, &[]))? else {
        return Ok(None);
    };
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&cancelled)?;
    let (Some(top), Some(bottom)) = (coefficients(&numerator, &ctx.n)?, coefficients(&denominator, &ctx.n)?) else {
        return Ok(None);
    };
    if bottom.is_empty() {
        return Ok(None);
    }
    if top.is_empty() {
        return Ok(Some(true_()));
    }
    let excess: i64 =
        bottom.iter().map(|(k, _)| *k).max().unwrap_or(0) - top.iter().map(|(k, _)| *k).max().unwrap_or(0);
    Ok(if excess >= 2 { Some(true_()) } else { None })
}

/// Whether the terms have eventually a constant sign (`_constant_sign`).
fn constant_sign(term: &Basic, ctx: &Context) -> PolyResult<Option<bool>> {
    let replacements = real_dummies(term, &ctx.n);
    for form in 0..3 {
        let found: Option<Basic> = within(|| match form {
            0 => Ok(term.clone()),
            1 => crate::core::rewriting::rewrite(
                term,
                &[],
                "factorial",
                &crate::core::rewriting::RewriteHints::default(),
            ),
            _ => crate::simplify::gammasimp::combsimp(term),
        })?;
        let Some(found) = found else {
            continue;
        };
        let at_positive: Basic = replaced(&found, &replacements)?;
        if is(&at_positive, Predicate::nonnegative) == Some(true)
            || is(&at_positive, Predicate::nonpositive) == Some(true)
        {
            return Ok(Some(true));
        }
    }
    if parameters(term, &ctx.n).is_empty() {
        for shift in [0, 8, 64] {
            let domain: Basic = interval_from(&add(&[ctx.lower.clone(), int(shift)])?)?;
            let signs = within(|| crate::assumptions::analysis::sign_on(term, &ctx.n, &domain, 1))?.flatten();
            if let Some(signs) = signs {
                if !(signs.contains(&1) && signs.contains(&-1)) {
                    return Ok(Some(true));
                }
            }
        }
        return Ok(None);
    }
    if ctx.eventually(&rel(RelOp::Ge, term, &Basic::zero())?)? == Some(true)
        || ctx.eventually(&rel(RelOp::Le, term, &Basic::zero())?)? == Some(true)
    {
        return Ok(Some(true));
    }
    Ok(None)
}

/// The condition under which `b_n` decreases to zero eventually
/// (`_decreasing_to_zero`).
fn decreasing_to_zero(b: &Basic, ctx: &Context) -> C {
    let Some(value) = limit(b, ctx, true)? else {
        return Ok(None);
    };
    let verdict = |v: &Basic, c: &Context| -> C {
        match is(v, Predicate::zero) {
            Some(false) => return Ok(Some(false_())),
            None => return Ok(None),
            Some(true) => {}
        }
        let difference: Basic = sub(&replaced(b, &[(c.n.clone(), add(&[c.n.clone(), int(1)])?)])?, b)?;
        if c.eventually(&rel(RelOp::Le, &difference, &Basic::zero())?)? == Some(true) {
            return Ok(Some(true_()));
        }
        let derivative: Basic = diff(b, &c.n)?;
        if c.eventually(&rel(RelOp::Le, &derivative, &Basic::zero())?)? == Some(true) {
            return Ok(Some(true_()));
        }
        Ok(None)
    };
    by_cases(&value, ctx, &verdict)
}

/// The integral test for eventually positive decreasing terms
/// (`_integral_test`).
fn integral_test(term: &Basic, ctx: &Context) -> C {
    if !parameters(term, &ctx.n).is_empty() {
        return Ok(None);
    }
    if ctx.eventually(&rel(RelOp::Gt, term, &Basic::zero())?)? != Some(true) {
        return Ok(None);
    }
    if !decreasing_to_zero(term, ctx)?.is_some_and(|c| is_true(&c)) {
        return Ok(None);
    }
    let Some(value) = improper_integral_native(term, &ctx.n, &ctx.lower)? else {
        return Ok(None);
    };
    if value.has(&ctx.n) {
        return Ok(None);
    }
    if is_infinity(&value) {
        return Ok(Some(false_()));
    }
    if is(&value, Predicate::finite) == Some(true) {
        return Ok(Some(true_()));
    }
    Ok(None)
}

/// What the divergence of `sum |a_n|` means for `sum a_n`
/// (`_divergence_of_absolute`).
fn divergence_of_absolute(term: &Basic, ctx: &Context) -> C {
    if constant_sign(term, ctx)? == Some(true) {
        return Ok(Some(false_()));
    }
    if let Some(b) = alternating(term, &ctx.n)? {
        if let Some(decreasing) = decreasing_to_zero(&b, ctx)? {
            return Ok(Some(decreasing));
        }
    }
    term_test(term, ctx)
}

/// `|a_{n+1}/a_n|` (or its inverse), simplified (`_ratio`).
fn ratio(term: &Basic, ctx: &Context, inverse: bool) -> R {
    let shifted: Basic = replaced(term, &[(ctx.n.clone(), add(&[ctx.n.clone(), int(1)])?)])?;
    let r: Basic = if inverse {
        abs_(&div(term, &shifted)?)?
    } else {
        abs_(&div(&shifted, term)?)?
    };
    clean(&r, &ctx.n)
}

fn ratio_test(term: &Basic, ctx: &Context) -> C {
    let Some(r) = limit(&ratio(term, ctx, false)?, ctx, false)? else {
        return root_test(term, ctx);
    };
    let verdict = |value: &Basic, c: &Context| -> C {
        compare_with_one(
            value,
            c,
            Some(false_()),
            Some(true_()),
            &|c2: &Context| raabe_test(term, c2),
            term,
        )
    };
    by_cases(&r, ctx, &verdict)
}

fn root_test(term: &Basic, ctx: &Context) -> C {
    let root: Basic = pow(&abs_(term)?, &div(&int(1), &ctx.n)?)?;
    let Some(c) = limit(&root, ctx, true)? else {
        return power_comparison(term, ctx);
    };
    let verdict = |value: &Basic, cx: &Context| -> C {
        compare_with_one(
            value,
            cx,
            Some(false_()),
            Some(true_()),
            &|c2: &Context| power_comparison(term, c2),
            term,
        )
    };
    by_cases(&c, ctx, &verdict)
}

fn raabe_expression(term: &Basic, ctx: &Context) -> R {
    mul(&[ctx.n.clone(), sub(&ratio(term, ctx, true)?, &int(1))?])
}

fn raabe_test(term: &Basic, ctx: &Context) -> C {
    let Some(rho) = limit(&raabe_expression(term, ctx)?, ctx, false)? else {
        return power_comparison(term, ctx);
    };
    let verdict = |value: &Basic, c: &Context| -> C {
        compare_with_one(
            value,
            c,
            Some(true_()),
            divergence_of_absolute(term, c)?,
            &|c2: &Context| bertrand_test(term, c2),
            term,
        )
    };
    by_cases(&rho, ctx, &verdict)
}

fn bertrand_test(term: &Basic, ctx: &Context) -> C {
    let expression: Basic = mul(&[sub(&raabe_expression(term, ctx)?, &int(1))?, log(&ctx.n)?])?;
    let Some(value) = limit(&expression, ctx, false)? else {
        return power_comparison(term, ctx);
    };
    let verdict = |v: &Basic, c: &Context| -> C {
        compare_with_one(
            v,
            c,
            Some(true_()),
            divergence_of_absolute(term, c)?,
            &|c2: &Context| last_resort(term, c2),
            term,
        )
    };
    by_cases(&value, ctx, &verdict)
}

/// The comparison with the `p`-series through `lim log|a_n|/log n`
/// (`_power_comparison`).
fn power_comparison(term: &Basic, ctx: &Context) -> C {
    let exponent: Basic = div(&log(&abs_(term)?)?, &log(&ctx.n)?)?;
    let Some(q) = limit(&exponent, ctx, true)? else {
        return last_resort(term, ctx);
    };
    let verdict = |value: &Basic, c: &Context| -> C {
        compare_with_one(
            &neg(value)?,
            c,
            Some(true_()),
            divergence_of_absolute(term, c)?,
            &|c2: &Context| last_resort(term, c2),
            term,
        )
    };
    by_cases(&q, ctx, &verdict)
}

/// The integral of `term` over `(lower, oo)` from a native antiderivative
/// checked by differentiation (`_improper_integral_native`).
fn improper_integral_native(term: &Basic, n: &Basic, lower: &Basic) -> PolyResult<Option<Basic>> {
    let x: Basic = dummy("x", &[(Predicate::positive, true)]);
    let f: Basic = replaced(term, &[(n.clone(), x.clone())])?;
    let mut antiderivative: Option<Basic> = None;
    let found =
        within(|| crate::integrals::risch::risch::risch_integrate_separate(&f, &x, None, "log", None, "piecewise"))?;
    if let Some((elementary, rest)) = found {
        if rest == Basic::zero() {
            antiderivative = Some(elementary);
        }
    }
    if antiderivative.is_none() {
        antiderivative = within(|| crate::integrals::heurisch::heurisch_antiderivative(&f, &x, &[], 0, 2))?.flatten();
    }
    let Some(big_f) = antiderivative else {
        return Ok(None);
    };
    let check: Option<Basic> = within(|| crate::polys::polytools::cancel(&sub(&diff(&big_f, &x)?, &f)?, &[]))?;
    match check {
        None => return Ok(None),
        Some(c) if c != Basic::zero() && simplify(&c)? != Basic::zero() => return Ok(None),
        _ => {}
    }
    let at_infinity = within(|| crate::core::limits::limit(&big_f, &x, &oo(), "+"))?;
    let at_lower = within(|| crate::core::limits::limit(&big_f, &x, lower, "+"))?;
    let (Some(at_infinity), Some(at_lower)) = (at_infinity, at_lower) else {
        return Ok(None);
    };
    let unevaluated = |e: &Basic| has_head(e, &|h| matches!(h, Head::Limit));
    if unevaluated(&at_infinity) || unevaluated(&at_lower) {
        return Ok(None);
    }
    let finite = |e: &Basic| is(e, Predicate::finite) == Some(true);
    if is_infinity(&at_infinity) && finite(&at_lower) {
        return Ok(Some(oo()));
    }
    if finite(&at_infinity) && finite(&at_lower) {
        return Ok(Some(sub(&at_infinity, &at_lower)?));
    }
    Ok(None)
}

/// SymPy's `is_decreasing(f, Interval(lower, oo))` (`_is_decreasing_on`).
fn is_decreasing_on(f: &Basic, n: &Basic, lower: &Basic) -> PolyResult<bool> {
    let x: Basic = dummy("x", &[(Predicate::real, true)]);
    let derivative: Basic = diff(&replaced(f, &[(n.clone(), x.clone())])?, &x)?;
    let domain: Basic = interval_from(lower)?;
    let signs = within(|| crate::assumptions::analysis::sign_on(&derivative, &x, &domain, 1))?.flatten();
    Ok(signs.is_some_and(|s| !s.contains(&1)))
}

/// The native `limit_seq` (`_limit_seq_of`).
fn limit_seq_of(e: &Basic, n: &Basic) -> PolyResult<Option<Basic>> {
    Ok(within(|| crate::series::limitseq::limit_seq(e, Some(n), 5))?.flatten())
}

/// Whether a numerical limit is below (`true`) or above (`false`) the
/// bound (`_is_number_below`).
fn is_number_below(value: Option<&Basic>, bound: i64) -> PolyResult<Option<bool>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !is_number(value) || matches!(value.head(), Head::NaN) || has_head(value, &|h| matches!(h, Head::AccumBounds)) {
        return Ok(None);
    }
    Ok(
        match crate::assumptions::analysis::certified_sign(&sub(value, &int(bound))?)? {
            None | Some(0) => None,
            Some(s) => Some(s < 0),
        },
    )
}

fn wild(name: &str) -> Basic {
    crate::core::symbol::Wild::new(name, vec![], vec![], &[]).0
}

fn bound_to(bindings: &[(Basic, Basic)], w: &Basic) -> Option<Basic> {
    bindings.iter().find(|(k, _)| k == w).map(|(_, v)| v.clone())
}

/// The index sets of `itertools.combinations(range(n), size)`, in its
/// order.
fn combinations(n: usize, size: usize) -> Vec<Vec<usize>> {
    fn extend(start: usize, n: usize, size: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == size {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            extend(i + 1, n, size, current, out);
            current.pop();
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    extend(0, n, size, &mut Vec::new(), &mut out);
    out
}

/// SymPy's `Sum(term, (n, lower, oo)).is_convergent()` for a term without
/// parameters, on the core (`_is_convergent_native`).
fn is_convergent_native(term: &Basic, n: &Basic, lower: &Basic) -> PolyResult<Option<bool>> {
    let mut sequence_term: Basic = simplify(term)?;
    if sequence_term.free_symbols().iter().any(|s| s != n) {
        return Ok(None);
    }
    let sym: Basic = positive_integer_dummy(&symbol_name(n));
    sequence_term = replaced(&sequence_term, &[(n.clone(), sym.clone())])?;
    if matches!(sequence_term.head(), Head::Piecewise) {
        for pair in sequence_term.args() {
            if is_true(&pair.args()[1]) {
                return is_convergent_native(&pair.args()[0], &sym, lower);
            }
        }
        return Ok(None);
    }
    // the divergence test
    if let Some(v) = limit_seq_of(&sequence_term, &sym)? {
        if is(&v, Predicate::zero) == Some(false) {
            return Ok(Some(false));
        }
    }
    if let Some(v) = limit_seq_of(&abs_(&sequence_term)?, &sym)? {
        if is(&v, Predicate::zero) == Some(false) {
            return Ok(Some(false));
        }
    }
    let big_o: Option<Basic> = within(|| crate::core::order::order(&sequence_term, &[(sym.clone(), oo())]))?;
    let o_expr: Option<Basic> = big_o
        .filter(|o| matches!(o.head(), Head::Order))
        .map(|o| o.args()[0].clone());
    let (p, q, r) = (wild("p"), wild("q"), wild("r"));
    let matched = |e: &Basic, pattern: &Basic| crate::core::matching::match_pattern(e, pattern, &[], false);
    if let Some(o) = &o_expr {
        // the p-series test
        if let Some(b) = matched(o, &pow(&sym, &p)?)? {
            if let Some(exponent) = bound_to(&b, &p) {
                if let Some(below) = is_number_below(Some(&exponent), -1)? {
                    return Ok(Some(below));
                }
                if exponent == Basic::negative_one() {
                    return Ok(Some(false));
                }
            }
        }
        // the comparison with 1/(n**p*log(n)**q*log(log(n))**r)
        let reciprocal: Basic = div(&int(1), &sym)?;
        let loglog: Basic = log(&neg(&log(&reciprocal)?)?)?;
        let first: Basic = div(
            &int(1),
            &mul(&[pow(&sym, &p)?, pow(&log(&reciprocal)?, &q)?, pow(&loglog, &r)?])?,
        )?;
        let second: Basic = div(
            &int(1),
            &mul(&[pow(&sym, &p)?, pow(&neg(&log(&reciprocal)?)?, &q)?, pow(&loglog, &r)?])?,
        )?;
        let n_log = match matched(o, &first)? {
            Some(b) => Some(b),
            None => matched(o, &second)?,
        };
        if let Some(b) = n_log {
            if let (Some(ep), Some(eq_), Some(er)) = (bound_to(&b, &p), bound_to(&b, &q), bound_to(&b, &r)) {
                if is_number(&ep) && is_number(&eq_) && is_number(&er) {
                    let one: Basic = int(1);
                    let above = |e: &Basic| -> PolyResult<bool> { Ok(is_true(&rel(RelOp::Gt, e, &one)?)) };
                    if above(&ep)? || (ep == one && above(&eq_)?) || (ep == one && eq_ == one && above(&er)?) {
                        return Ok(Some(true));
                    }
                    return Ok(Some(false));
                }
            }
        }
    }
    // the limit comparison with 1/n
    if let Some(v) = limit_seq_of(&mul(&[sym.clone(), sequence_term.clone()])?, &sym)? {
        if is_number(&v) && crate::assumptions::analysis::certified_sign(&v)? == Some(1) {
            return Ok(Some(false));
        }
    }
    // the ratio test
    let next_term: Basic = replaced(&sequence_term, &[(sym.clone(), add(&[sym.clone(), int(1)])?)])?;
    let quotient: Basic = crate::simplify::gammasimp::combsimp(&crate::simplify::powsimp::powsimp_default(&div(
        &next_term,
        &sequence_term,
    )?)?)?;
    let lim_ratio: Option<Basic> = limit_seq_of(&quotient, &sym)?;
    let magnitude: Option<Basic> = match &lim_ratio {
        Some(v) => Some(abs_(v)?),
        None => None,
    };
    if let Some(below) = is_number_below(magnitude.as_ref(), 1)? {
        return Ok(Some(below));
    }
    // Raabe's test
    if lim_ratio.as_ref().is_some_and(|v| *v == Basic::one()) {
        let test_val: Basic = crate::simplify::gammasimp::gammasimp(&mul(&[
            sym.clone(),
            sub(&div(&sequence_term, &next_term)?, &int(1))?,
        ])?)?;
        if let Some(raabe) = is_number_below(limit_seq_of(&test_val, &sym)?.as_ref(), 1)? {
            return Ok(Some(!raabe));
        }
    }
    // the root test
    let root: Basic = pow(&abs_(&sequence_term)?, &div(&int(1), &sym)?)?;
    if let Some(below) = is_number_below(limit_seq_of(&root, &sym)?.as_ref(), 1)? {
        return Ok(Some(below));
    }
    // the alternating series test
    let pattern: Basic = mul(&[pow(&int(-1), &add(&[sym.clone(), p.clone()])?)?, q.clone()])?;
    if let Some(b) = matched(&sequence_term, &pattern)? {
        if let (Some(shift), Some(rest)) = (bound_to(&b, &p), bound_to(&b, &q)) {
            if !shift.has(&sym) && is_decreasing_on(&rest, &sym, lower)? {
                return Ok(Some(true));
            }
        }
    }
    // the integral test, for a monotone term
    if is_decreasing_on(&sequence_term, &sym, lower)? || is_decreasing_on(&neg(&sequence_term)?, &sym, lower)? {
        if let Some(integral) = improper_integral_native(&sequence_term, &sym, lower)? {
            if is_number(&integral) {
                return Ok(Some(is(&integral, Predicate::finite) == Some(true)));
            }
        }
    }
    // the Dirichlet and the bounded times convergent tests
    if let Some(o) = o_expr.filter(|o| o.is_mul()) {
        let args: Vec<Basic> = o.args().to_vec();
        let m: Basic = dummy("m", &[(Predicate::integer, true)]);
        for size in 1..args.len() {
            for chosen in combinations(args.len(), size) {
                let a_n: Basic = mul(&chosen.iter().map(|&i| args[i].clone()).collect::<Vec<_>>())?;
                let b_n: Basic = mul(&(0..args.len())
                    .filter(|i| !chosen.contains(i))
                    .map(|i| args[i].clone())
                    .collect::<Vec<_>>())?;
                if is_decreasing_on(&a_n, &sym, lower)? {
                    let limits = (sym.clone(), lower.clone(), m.clone());
                    let partial: Option<Basic> = within(|| crate::concrete::summations::summation(&b_n, &limits))?;
                    if let Some(partial) = partial {
                        if let Some(bounded) = limit_seq_of(&partial, &m)? {
                            if is(&bounded, Predicate::finite) == Some(true) {
                                return Ok(Some(true));
                            }
                        }
                    }
                }
                if let Some(lim_a) = limit_seq_of(&a_n, &sym)? {
                    let bounded: bool = is(&lim_a, Predicate::finite) == Some(true)
                        || (matches!(lim_a.head(), Head::AccumBounds)
                            && is(&sub(&lim_a.args()[1], &lim_a.args()[0])?, Predicate::finite) == Some(true));
                    if bounded && is_convergent_native(&abs_(&b_n)?, &sym, lower)? == Some(true) {
                        return Ok(Some(true));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// The term, alternating, Dirichlet and integral tests, and SymPy's own
/// decision for terms without parameters (`_last_resort`).
fn last_resort(term: &Basic, ctx: &Context) -> C {
    if let Some(divergent) = term_test(term, ctx)? {
        return Ok(Some(divergent));
    }
    if let Some(b) = alternating(term, &ctx.n)? {
        if decreasing_to_zero(&b, ctx)?.is_some_and(|c| is_true(&c)) {
            return Ok(Some(true_()));
        }
    }
    if let Some(found) = dirichlet_test(term, ctx)? {
        return Ok(Some(found));
    }
    if let Some(found) = integral_test(term, ctx)? {
        return Ok(Some(found));
    }
    if parameters(term, &ctx.n).is_empty() {
        // only True is a proof (sympy-extras#41)
        let verdict = within(|| is_convergent_native(term, &ctx.n, &ctx.lower))?.flatten();
        if verdict == Some(true) {
            return Ok(Some(true_()));
        }
    }
    Ok(None)
}

/// Whether `term` is an unbounded factor times `sin` or `cos` of a linear
/// argument in `n` (`_oscillates_unboundedly`).
fn oscillates_unboundedly(term: &Basic, ctx: &Context) -> PolyResult<bool> {
    if !term.is_mul() {
        return Ok(false);
    }
    let waves: Vec<Basic> = factors(term)
        .into_iter()
        .filter(|f| (f.is_function(Function::Sin) || f.is_function(Function::Cos)) && f.args()[0].has(&ctx.n))
        .collect();
    if waves.len() != 1 {
        return Ok(false);
    }
    let wave: &Basic = &waves[0];
    let Some(linear) = coefficients(&wave.args()[0], &ctx.n)? else {
        return Ok(false);
    };
    if linear.iter().map(|(k, _)| *k).max().unwrap_or(0) != 1 {
        return Ok(false);
    }
    let get = |d: i64| {
        linear
            .iter()
            .find(|(k, _)| *k == d)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Basic::zero)
    };
    let (a, b) = (get(1), get(0));
    if !a.free_symbols().is_empty() || !b.free_symbols().is_empty() || a == Basic::zero() {
        return Ok(false);
    }
    let turns: Basic = div(&a, &Basic::pi())?;
    if is(&turns, Predicate::integer) == Some(true) {
        let phase: Basic = if wave.is_function(Function::Sin) {
            crate::functions::elementary::trigonometric::sin(&b)?
        } else {
            crate::functions::elementary::trigonometric::cos(&b)?
        };
        if is(&phase, Predicate::zero) != Some(false) {
            return Ok(false);
        }
    }
    let rest: Basic = div(term, wave)?;
    let magnitude: Option<Basic> = limit(&abs_(&rest)?, ctx, true)?;
    Ok(magnitude.is_some_and(|m| is_infinity(&m)))
}

/// Divergence when the terms do not tend to zero; `None` otherwise
/// (`_term_test`).
fn term_test(term: &Basic, ctx: &Context) -> C {
    if oscillates_unboundedly(term, ctx)? {
        return Ok(Some(false_()));
    }
    let Some(value) = limit(term, ctx, true)? else {
        return Ok(None);
    };
    let verdict = |v: &Basic, c: &Context| -> C {
        if matches!(v.head(), Head::AccumBounds | Head::Infinity | Head::NegativeInfinity) {
            return Ok(Some(false_()));
        }
        if matches!(v.head(), Head::NaN) {
            return Ok(None);
        }
        if v.free_symbols().is_empty() {
            return Ok(if is(v, Predicate::zero) == Some(false) {
                Some(false_())
            } else {
                None
            });
        }
        Ok(if is_false(&c.decide(&rel(RelOp::Eq, v, &Basic::zero())?)?) {
            Some(false_())
        } else {
            None
        })
    };
    by_cases(&value, ctx, &verdict)
}

fn convergence(term: &Basic, ctx: &Context) -> C {
    if !term.has(&ctx.n) {
        return Ok(Some(ctx.decide(&rel(RelOp::Eq, term, &Basic::zero())?)?));
    }
    // a constant factor which may vanish makes the series trivially
    // convergent (and cancels from the ratios, so it is separated first)
    let (c, rest) = crate::core::expr::as_independent(term, &ctx.n, false)?;
    if !c.free_symbols().is_empty() {
        let zero: Basic = ctx.decide(&rel(RelOp::Eq, &c, &Basic::zero())?)?;
        if is_true(&zero) {
            return Ok(Some(true_()));
        }
        if !is_false(&zero) {
            let Some(nonzero) = ratio_test(&rest, &ctx.assume(&rel(RelOp::Ne, &c, &Basic::zero())?))? else {
                return Ok(None);
            };
            return Ok(Some(or_(&[zero, nonzero])));
        }
    }
    if rational_comparison(term, ctx)?.is_some_and(|c| is_true(&c)) {
        return Ok(Some(true_()));
    }
    ratio_test(term, ctx)
}

/// The condition simplified (`_finish`).
fn finish(condition: Condition, ctx: &Context) -> C {
    let Some(condition) = condition else {
        return Ok(None);
    };
    if is_true(&condition) || is_false(&condition) {
        return Ok(Some(condition));
    }
    let polynomial: Basic = without_abs(&condition)?;
    let given: Vec<Basic> = if ctx.assumptions.is_empty() {
        vec![]
    } else {
        vec![ctx.premise()]
    };
    if let Some(resolved) = within(|| crate::assumptions::resolve::resolve(&polynomial, None, &given, None))? {
        return Ok(Some(resolved));
    }
    let premise: Basic = ctx.premise();
    let simplified: Option<Basic> =
        within(|| crate::assumptions::refine::simplify(&condition, &[premise], None, 1.7, false, true))?;
    Ok(Some(simplified.unwrap_or(condition)))
}

/// `Abs(u) < c` written as `(u < c) & (u > -c)` and the like
/// (`_without_abs`).
fn without_abs(condition: &Basic) -> R {
    let mut result: Basic = condition.clone();
    for atom in condition.atoms(Some(&|e: &Basic| matches!(e.head(), Head::Relational(_)))) {
        let Head::Relational(op) = atom.head() else {
            continue;
        };
        let (lhs, rhs) = (&atom.args()[0], &atom.args()[1]);
        if !lhs.is_function(Function::Abs) || has_function(rhs, &[Function::Abs]) {
            continue;
        }
        let u: &Basic = &lhs.args()[0];
        let minus_u: Basic = neg(u)?;
        let replacement: Basic = match op {
            RelOp::Lt | RelOp::Le => and_(&[rel(*op, u, rhs)?, rel(*op, &minus_u, rhs)?]),
            RelOp::Gt | RelOp::Ge => or_(&[rel(*op, u, rhs)?, rel(*op, &minus_u, rhs)?]),
            RelOp::Eq => or_(&[rel(RelOp::Eq, u, rhs)?, rel(RelOp::Eq, u, &neg(rhs)?)?]),
            RelOp::Ne => and_(&[rel(RelOp::Ne, u, rhs)?, rel(RelOp::Ne, u, &neg(rhs)?)?]),
        };
        result = replaced(&result, &[(atom.clone(), replacement)])?;
    }
    Ok(result)
}

fn context(n: &Basic, lower: &Basic, assumptions: &[Basic]) -> Context {
    Context {
        n: n.clone(),
        lower: lower.clone(),
        assumptions: assumptions.to_vec(),
        depth: 0,
    }
}

/// The condition on the parameters under which `Sum(term, (n, lower,
/// oo))` converges: `true`, `false`, a Boolean in the parameters, or
/// `None` when the tests do not decide (`sum_convergence`).
pub fn sum_convergence(term: &Basic, n: &Basic, assumptions: &[Basic], lower: &Basic) -> C {
    let ctx: Context = context(n, lower, assumptions);
    finish(convergence(term, &ctx)?, &ctx)
}

/// Whether `Sum(term, (n, lower, oo))` converges under the assumptions
/// (`is_convergent`).
pub fn is_convergent(term: &Basic, n: &Basic, assumptions: &[Basic], lower: &Basic) -> PolyResult<Option<bool>> {
    Ok(match sum_convergence(term, n, assumptions, lower)? {
        Some(c) if is_true(&c) => Some(true),
        Some(c) if is_false(&c) => Some(false),
        _ => None,
    })
}

/// The condition under which the infinite product `Product(term, (n,
/// lower, oo))` converges to a nonzero limit (`product_convergence`).
pub fn product_convergence(term: &Basic, n: &Basic, assumptions: &[Basic], lower: &Basic) -> C {
    let ctx: Context = context(n, lower, assumptions);
    let b: Basic = sub(term, &int(1))?;
    if !b.has(n) {
        let decided: Basic = ctx.decide(&rel(RelOp::Eq, &b, &Basic::zero())?)?;
        return finish(Some(decided), &ctx);
    }
    if let Some(value) = limit(&b, &ctx, true)? {
        if !matches!(value.head(), Head::Piecewise)
            && value.free_symbols().is_empty()
            && is(&value, Predicate::zero) == Some(false)
        {
            return Ok(Some(false_()));
        }
    }
    let positive: Basic = positive_integer_dummy("n");
    let magnitude: Basic = replaced(
        &abs_(&replaced(&b, &[(n.clone(), positive.clone())])?)?,
        &[(positive, n.clone())],
    )?;
    let absolute: Condition = convergence(&magnitude, &ctx)?;
    if absolute.as_ref().is_some_and(is_true) {
        return Ok(Some(true_()));
    }
    // for an alternating factor the condition for sum |b| is sufficient,
    // not necessary (sympy-extras#39)
    if let Some(found) = alternating_product(&b, &ctx)? {
        return finish(Some(found), &ctx);
    }
    if let Some(a) = &absolute {
        if !is_true(a) && !is_false(a) {
            return finish(absolute, &ctx);
        }
    }
    if absolute.as_ref().is_some_and(is_false) && constant_sign(&b, &ctx)? == Some(true) {
        return Ok(Some(false_()));
    }
    let logarithmic: Condition = convergence(&log(term)?, &ctx)?;
    finish(logarithmic, &ctx)
}

/// `prod (1 + (-1)**n*c(n))` with `c > 0` decreasing to zero converges
/// exactly when `sum c**2` does (`_alternating_product`).
fn alternating_product(b: &Basic, ctx: &Context) -> C {
    let sign: Basic = pow(&int(-1), &ctx.n)?;
    let Some(c) = within(|| crate::polys::polytools::cancel(&div(b, &sign)?, &[]))? else {
        return Ok(None);
    };
    if c.atoms(Some(&|p: &Basic| p.is_pow()))
        .iter()
        .any(|p| p.args()[0] == Basic::negative_one())
    {
        return Ok(None);
    }
    let grows: Basic = positive_integer_dummy("n");
    let mut reals: Vec<(Basic, Basic)> = Vec::new();
    for s in parameters(&c, &ctx.n) {
        let name: String = symbol_name(&s);
        reals.push((s, dummy(&name, &[(Predicate::real, true)])));
    }
    let at: Basic = replaced(&replaced(&c, &[(ctx.n.clone(), grows)])?, &reals)?;
    if is(&at, Predicate::positive) != Some(true) {
        return Ok(None);
    }
    convergence(&pow(&c, &int(2))?, ctx)
}
