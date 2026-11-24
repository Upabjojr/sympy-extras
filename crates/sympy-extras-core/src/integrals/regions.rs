//! Integrals over regions given by inequalities, ported from
//! `sympy_extras.integrals.regions`: the integral of a function over the
//! semialgebraic set of a condition by cylindrical algebraic decomposition
//! (the cells of full dimension as iterated integrals with explicit
//! bounds, the parameter cells as cases), after the routes of polar,
//! scaled and cylindrical coordinates, of the bounds solved for the last
//! variable, of polytopes (Dirichlet's formula on a triangulation), of a
//! radial integrand in a symbolic dimension and of the Hausdorff measure
//! on the varieties of the equations.

// PORTED FROM SYMPY: sympy/solvers/solvers.py (_solve, the polynomial case)
// (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: typed, on core expressions, the roots of polys.polyroots
// (solutions only); the rest of the module is original code.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::polys::cad::lifting::{Cad, CadCell};
use crate::polys::cad::samplepoints::Real;

type R = PolyResult<Basic>;

/// The numerical tolerance for matching a branch to a root at the sample
/// point (`_TOLERANCE`).
const TOLERANCE: f64 = 1e-20;

/// The working precision (bits) of the numerical matching of the branches
/// (`_PREC`).
const PREC: usize = 120;

/// The measures of integration: the Lebesgue measure of `R^n` and the
/// Hausdorff measure on the variety of the equations (`_MEASURES`).
pub const MEASURES: [&str; 2] = ["lebesgue", "hausdorff"];

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn sqrt(e: &Basic) -> R {
    pow(e, &Basic::from_rational(crate::numbers::rat(1, 2)))
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn integer_class(e: &Basic) -> Option<i64> {
    if !matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) {
        return None;
    }
    e.to_string().parse::<i64>().ok()
}

/// A dummy with a name of its own: the typed polynomials map their symbols
/// by name, so two dummies of one name would be one symbol (`_fresh`).
pub fn fresh(name: &str, positive: bool) -> Basic {
    let n: usize = COUNTER.fetch_add(1, Ordering::Relaxed);
    let assumptions: &[(Predicate, bool)] = if positive { &[(Predicate::positive, true)] } else { &[] };
    crate::core::symbol::Dummy::new(&format!("{}{}", name, n), assumptions).0
}

/// `e.xreplace(rule)` with the rebuilt nodes evaluated (`_evaluated`).
#[allow(clippy::mutable_key_type)]
fn evaluated(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: std::collections::HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

fn symbol_name(s: &Basic) -> String {
    crate::core::printing::sstr(s)
}

/// The number of operations of `e` (`_ops`).
fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

/// The radial variable among the symbols of a condition: the one named
/// `r`, or the only one (`_radial_symbol`).
pub fn radial_symbol(names: &[Basic]) -> R {
    for name in names {
        if symbol_name(name) == "r" {
            return Ok(name.clone());
        }
    }
    match names.len() {
        0 => Ok(crate::core::symbol::Symbol::new("r", &[]).0),
        1 => Ok(names[0].clone()),
        _ => Err(PolyError::Value(format!(
            "the radial variable of a condition in {:?} must be given",
            names.iter().map(symbol_name).collect::<Vec<String>>()
        ))),
    }
}

// -- polynomials in the integration variables with any coefficients

/// A polynomial in the integration variables with coefficients free of
/// them: the exponents of each term and its coefficient, in the order of
/// the first appearance (`Coefficients`).
pub type Coefficients = Vec<(Vec<i64>, Basic)>;

/// `e` as a polynomial in `names` whose coefficients are free of them, the
/// zero terms dropped; `None` when it is not one (`_coefficients`).
pub fn coefficients(e: &Basic, names: &[Basic]) -> PolyResult<Option<Coefficients>> {
    let mut terms: Coefficients = Vec::new();
    for term in crate::core::expr::terms_of(&expand(e)?) {
        let (coefficient, dependent) = independent_of_all(&term, names)?;
        let mut exponents: Vec<i64> = vec![0; names.len()];
        for part in crate::core::expr::factors_of(&dependent) {
            if part == Basic::one() {
                continue;
            }
            let (base, exponent) = crate::core::expr::as_base_exp(&part)?;
            let Some(i) = names.iter().position(|n| *n == base) else {
                return Ok(None);
            };
            match integer_class(&exponent) {
                Some(p) if p >= 0 => exponents[i] += p,
                _ => return Ok(None),
            }
        }
        match terms.iter_mut().find(|(k, _)| *k == exponents) {
            Some(entry) => entry.1 = add(&[entry.1.clone(), coefficient])?,
            None => terms.push((exponents, coefficient)),
        }
    }
    Ok(Some(terms.into_iter().filter(|(_, v)| *v != Basic::zero()).collect()))
}

/// `e.as_independent(*names, as_Add=False)`.
fn independent_of_all(e: &Basic, names: &[Basic]) -> PolyResult<(Basic, Basic)> {
    let depends = |t: &Basic| names.iter().any(|s| t.has(s));
    if !e.is_mul() {
        return Ok(if depends(e) {
            (Basic::one(), e.clone())
        } else {
            (e.clone(), Basic::one())
        });
    }
    let (mut free, mut dependent): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for f in e.args() {
        if depends(f) {
            dependent.push(f.clone());
        } else {
            free.push(f.clone());
        }
    }
    Ok((mul(&free)?, mul(&dependent)?))
}

fn coefficient(terms: &Coefficients, key: &[i64]) -> Basic {
    terms
        .iter()
        .find(|(k, _)| k == key)
        .map_or_else(Basic::zero, |(_, v)| v.clone())
}

fn total_degree(terms: &Coefficients) -> i64 {
    terms.iter().map(|(k, _)| k.iter().sum::<i64>()).max().unwrap_or(-1)
}

/// The exponents of `x_i**e` among `n` variables (`_unit`).
fn unit(n: usize, i: usize, e: i64) -> Vec<i64> {
    (0..n).map(|j| if j == i { e } else { 0 }).collect()
}

/// A root of a polynomial in `x_k` over a cell: the `index`-th real root of
/// `poly` at the sample point, and its value (`_Bound`).
#[derive(Clone, Debug)]
struct Bound {
    poly: Basic,
    value: Real,
}

/// The bounds of one cell of full dimension, level by level, as explicit
/// expressions (`-oo`/`oo` for an unbounded sector) (`_Stack`).
#[derive(Clone, Debug)]
struct Stack {
    bounds: Vec<(Basic, Basic)>,
}

fn compare(a: &Real, b: &Real) -> PolyResult<std::cmp::Ordering> {
    use crate::polys::cad::samplepoints::compare_real;
    Ok(match compare_real(a, b)? {
        s if s.is_negative() => std::cmp::Ordering::Less,
        s if s.is_positive() => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    })
}

/// The sorted distinct real roots of the polynomials of a level over the
/// sample point of the parent cell, each attributed to the first
/// polynomial having it (`_root_sequence`).
fn root_sequence(parent: &CadCell, polys: &[Basic], gens: &[Basic]) -> PolyResult<Vec<Bound>> {
    let sample = parent.sample()?;
    let mut found: Vec<Bound> = Vec::new();
    for poly in polys {
        let Some(roots) = sample.real_roots(poly, gens)? else {
            continue;
        };
        for root in roots {
            let mut seen: bool = false;
            for b in &found {
                if compare(&root, &b.value)? == std::cmp::Ordering::Equal {
                    seen = true;
                    break;
                }
            }
            if !seen {
                found.push(Bound {
                    poly: poly.clone(),
                    value: root,
                });
            }
        }
    }
    let mut failure: Option<PolyError> = None;
    found.sort_by(|a, b| match compare(&a.value, &b.value) {
        Ok(o) => o,
        Err(e) => {
            failure.get_or_insert(e);
            std::cmp::Ordering::Equal
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    Ok(found)
}

/// A real algebraic number as an expression: a rational number, else the
/// root object of its minimal polynomial in the variable `name`
/// (`_number`).
fn number(a: &Real, name: &str) -> R {
    crate::polys::cad::indexedroot::real_root(a, name)
}

fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    let limit: Option<f64> = crate::settings::settings().timeout;
    crate::timeout::attempt_within(limit, f)
}

/// The root as an expression in the earlier variables: the solution of the
/// polynomial for the last generator which takes the value of the root at
/// the sample point of the parent cell (`_explicit_root`).
fn explicit_root(bound: &Bound, parent: &CadCell, gens: &[Basic]) -> PolyResult<Option<Basic>> {
    let x: &Basic = &gens[gens.len() - 1];
    let equation: &Basic = &bound.poly;
    let earlier: &[Basic] = &gens[..gens.len() - 1];
    if gens.len() == 1 || !earlier.iter().any(|g| equation.has(g)) {
        return Ok(Some(radical_form_of(&number(&bound.value, &symbol_name(x))?)?));
    }
    let mut point: Vec<(Basic, Basic)> = Vec::new();
    for (g, p) in earlier.iter().zip(&parent.point) {
        point.push((g.clone(), number(p, &symbol_name(g))?));
    }
    let Some(target) = crate::core::evalf::numeric_value(&number(&bound.value, &symbol_name(x))?, PREC) else {
        return Ok(None);
    };
    let target_re: f64 = crate::core::evalf::parts(&target).0.to_f64().value();
    let trigonometric: Option<Vec<Basic>> = trigonometric_roots(equation, x)?;
    let solved: Option<Vec<Basic>> = attempt(|| solutions(equation, x))?;
    for candidates in [trigonometric, solved].into_iter().flatten() {
        let mut matches: Vec<Basic> = Vec::new();
        for c in candidates {
            // a form for the other sign of p, or a value whose branch a
            // rounding error chooses
            let Some(z) = crate::numeric::reliable_value(&c, 30, &point) else {
                continue;
            };
            let (re, im) = crate::core::evalf::parts(&z);
            let (re, im): (f64, f64) = (re.to_f64().value(), im.to_f64().value());
            // Cardano's form of a real root has a tiny imaginary part
            if im.abs() >= TOLERANCE {
                continue;
            }
            if (re - target_re).abs() < TOLERANCE {
                matches.push(c);
            }
        }
        if matches.len() == 1 {
            return Ok(matches.into_iter().next());
        }
    }
    Ok(None)
}

/// The solutions of a polynomial equation in `x` as SymPy's `solve` gives
/// them: the explicit roots, simplified when there are at most two
/// (`_solutions`).
fn solutions(equation: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let found = match crate::polys::polyroots::roots(equation, std::slice::from_ref(x), true, true, None) {
        Ok(f) => f,
        Err(PolyError::Polys(..)) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let candidates: Vec<Basic> = found.iter().map(|(r, _)| r.clone()).collect();
    if candidates.len() <= 2 {
        return candidates.iter().map(simplify).collect();
    }
    Ok(candidates)
}

/// A numeric root object of degree at most four written in radicals when
/// the radicals are real and short, else the root object itself
/// (`_radical_form`).
fn radical_form_of(root: &Basic) -> R {
    if !matches!(root.head(), Head::IndexedRoot) {
        return Ok(root.clone());
    }
    let (expr, variable) = (&root.args()[0], &root.args()[1]);
    let Some(terms) = coefficients(expr, std::slice::from_ref(variable))? else {
        return Ok(root.clone());
    };
    if total_degree(&terms) > 4 {
        return Ok(root.clone());
    }
    let candidate: Option<Option<Basic>> = attempt(|| crate::polys::roots::radical_form(root))?;
    match candidate.flatten() {
        Some(c)
            if !matches!(c.head(), Head::IndexedRoot)
                && !c.has(&Basic::imaginary_unit())
                && crate::core::printing::sstr(&c).len() <= 200 =>
        {
            Ok(c)
        }
        _ => Ok(root.clone()),
    }
}

/// The roots of a cubic in `x` in Viète's trigonometric and hyperbolic
/// forms for the depressed cubic `t**3 + p*t + q`, all shifted by
/// `-b/(3*a)`; the candidates real at the sample point are selected by the
/// caller; `None` for another degree (`_trigonometric_roots`).
fn trigonometric_roots(poly: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    use crate::functions::elementary::hyperbolic::{cosh, sinh};
    use crate::functions::elementary::inverse::{acos, acosh, asinh};
    use crate::functions::elementary::trigonometric::cos;
    let Some(terms) = coefficients(poly, std::slice::from_ref(x))? else {
        return Ok(None);
    };
    if total_degree(&terms) != 3 {
        return Ok(None);
    }
    let (a, b, c, d) = (
        coefficient(&terms, &[3]),
        coefficient(&terms, &[2]),
        coefficient(&terms, &[1]),
        coefficient(&terms, &[0]),
    );
    let three: Basic = int(3);
    let p: Basic = div(
        &sub(&mul(&[three.clone(), a.clone(), c.clone()])?, &pow(&b, &int(2))?)?,
        &mul(&[three.clone(), pow(&a, &int(2))?])?,
    )?;
    let q: Basic = div(
        &add(&[
            mul(&[int(2), pow(&b, &int(3))?])?,
            mul(&[int(-9), a.clone(), b.clone(), c.clone()])?,
            mul(&[int(27), pow(&a, &int(2))?, d])?,
        ])?,
        &mul(&[int(27), pow(&a, &int(3))?])?,
    )?;
    if p == Basic::zero() {
        return Ok(None);
    }
    let shift: Basic = neg(&div(&b, &mul(&[three.clone(), a])?)?)?;
    let radius: Basic = mul(&[int(2), sqrt(&neg(&div(&p, &three)?)?)?])?;
    let ratio = |s: &Basic| -> R {
        div(
            &mul(&[three.clone(), q.clone(), s.clone()])?,
            &mul(&[int(2), p.clone()])?,
        )
    };
    let angle: Basic = div(&acos(&ratio(&sqrt(&div(&int(-3), &p)?)?)?)?, &three)?;
    let mut found: Vec<Basic> = Vec::new();
    for k in 0..3 {
        let phase: Basic = div(&mul(&[int(2), Basic::pi(), int(k)])?, &three)?;
        found.push(add(&[
            mul(&[radius.clone(), cos(&sub(&angle, &phase)?)?])?,
            shift.clone(),
        ])?);
    }
    let hyperbolic: Basic = sinh(&div(&asinh(&ratio(&sqrt(&div(&three, &p)?)?)?)?, &three)?)?;
    found.push(add(&[
        mul(&[int(-2), sqrt(&div(&p, &three)?)?, hyperbolic])?,
        shift.clone(),
    ])?);
    let abs_q: Basic = crate::functions::elementary::complexes::abs_(&q)?;
    let inner: Basic = div(
        &mul(&[int(-3), abs_q, sqrt(&div(&int(-3), &p)?)?])?,
        &mul(&[int(2), p.clone()])?,
    )?;
    found.push(add(&[
        mul(&[
            int(-2),
            crate::functions::elementary::complexes::sign(&q)?,
            sqrt(&neg(&div(&p, &three)?)?)?,
            cosh(&div(&acosh(&inner)?, &three)?)?,
        ])?,
        shift,
    ])?);
    Ok(Some(found))
}

/// A one-dimensional integrator: `f` over `(x, a, b)` under the
/// assumptions, `None` when it is not computed; a `Value` error for bounds
/// which contradict the assumptions (an empty cell) (`Integrator`).
pub type Integrator<'a> = &'a dyn Fn(&Basic, &Basic, &Basic, &Basic, &[Basic]) -> PolyResult<Option<Basic>>;

/// The default integrator: `definite_integral` without the cases of the
/// special values of the parameters; `None` when it keeps an `Integral`
/// (`_definite`).
pub fn definite(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    let options = crate::integrals::definite::Options {
        special_values: false,
        ..Default::default()
    };
    let value: Basic = crate::integrals::definite::definite_integral(f, x, a, b, assumptions, &options)?;
    let unevaluated: bool = value
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Integral | Head::IntegralByRanges));
    Ok(if unevaluated { None } else { Some(value) })
}

fn oo() -> Basic {
    Basic::oo()
}
fn neg_oo() -> Basic {
    neg(&Basic::oo()).expect("-oo")
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
fn eq(a: &Basic, b: &Basic) -> R {
    crate::core::relational::eq(a, b)
}
fn and_(v: &[Basic]) -> R {
    Ok(crate::core::logic::and_(v))
}
fn is_false(b: &Basic) -> bool {
    *b == crate::core::logic::false_()
}

fn poly_exprs(polys: &[crate::polys::polytools::Poly]) -> PolyResult<Vec<Basic>> {
    polys.iter().map(|p| p.as_expr()).collect()
}

/// The ancestors of a cell from level 1 to its own.
fn ancestors(cell: &std::sync::Arc<CadCell>) -> Vec<std::sync::Arc<CadCell>> {
    let mut found: Vec<std::sync::Arc<CadCell>> = Vec::new();
    let mut current: Option<std::sync::Arc<CadCell>> = Some(cell.clone());
    while let Some(c) = current {
        if c.level() == 0 {
            break;
        }
        current = c.parent.clone();
        found.push(c);
    }
    found.reverse();
    found
}

/// The explicit bounds of a cell in the levels from `first` (1-based)
/// upwards, all of them sectors but the levels `skip` (sections, left
/// out), whose variables are replaced in the bounds of the later levels by
/// `replace` (`_stack`).
fn stack(
    cell: &std::sync::Arc<CadCell>,
    cad: &Cad,
    first: usize,
    skip: &[usize],
    replace: &[(Basic, Basic)],
) -> PolyResult<Option<Stack>> {
    let chain: Vec<std::sync::Arc<CadCell>> = ancestors(cell);
    let mut bounds: Vec<(Basic, Basic)> = Vec::new();
    for level in first..=cad.gens.len() {
        if skip.contains(&level) {
            continue;
        }
        let node: &std::sync::Arc<CadCell> = &chain[level - 1];
        let Some(parent) = node.parent.clone() else {
            return Ok(None);
        };
        if node.is_section() {
            return Ok(None);
        }
        let gens: &[Basic] = &cad.gens[..level];
        let roots: Vec<Bound> = root_sequence(&parent, &poly_exprs(&cad.projection[level - 1])?, gens)?;
        // the sector lies between root position-1 and position
        let position: usize = (node.index[node.index.len() - 1] - 1) / 2;
        let mut lower: Basic = neg_oo();
        let mut upper: Basic = oo();
        if position >= 1 {
            let Some(found) = explicit_root(&roots[position - 1], &parent, gens)? else {
                return Ok(None);
            };
            lower = found;
        }
        if position < roots.len() {
            let Some(found) = explicit_root(&roots[position], &parent, gens)? else {
                return Ok(None);
            };
            upper = found;
        }
        if !replace.is_empty() {
            lower = evaluated(&lower, replace)?;
            upper = evaluated(&upper, replace)?;
        }
        bounds.push((lower, upper));
    }
    Ok(Some(Stack { bounds }))
}

/// `ask` which treats contradictory assumptions (an empty cell of the
/// decomposition under its parameter condition) as undecided (`_ask`).
fn ask(query: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    match crate::assumptions::ask::ask(query, assumptions, None) {
        Ok(v) => Ok(v),
        Err(PolyError::Value(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Square roots (and other rational powers) of products with factors known
/// positive under the assumptions split into powers of the factors, so
/// that the integrand of an inner integral has the parameters outside the
/// radicals (`_split_roots`).
fn split_roots(expr: &Basic, assumptions: &[Basic]) -> R {
    if !expr.preorder().iter().any(|n| n.is_pow()) {
        return Ok(expr.clone());
    }
    let rewrite = |node: &Basic| -> R {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        let Some(q) = exponent.as_rational() else {
            return Ok(node.clone());
        };
        if q.is_int() {
            return Ok(node.clone());
        }
        let Some(base) =
            crate::timeout::attempt_within(None, || crate::polys::polytools::factor(base, &[], false, true))?
        else {
            return Ok(node.clone());
        };
        let factors: Vec<Basic> = if base.is_mul() {
            base.args().to_vec()
        } else {
            vec![base.clone()]
        };
        // (a factor known positive, its exponent)
        let mut positive: Vec<(Basic, Basic)> = Vec::new();
        let mut rest: Vec<Basic> = Vec::new();
        let mut sign: i64 = 1;
        for f in factors {
            let (mut f_, mut power) = (f.clone(), Basic::one());
            // a reciprocal (x - 1)**-1 is split by the sign of x - 1
            if f.is_pow() && is(&f.args()[1], Predicate::integer) == Some(true) {
                f_ = f.args()[0].clone();
                power = f.args()[1].clone();
            }
            let number: bool = crate::core::assumptions::is_number(&f_);
            if is(&f_, Predicate::positive) == Some(true)
                || (!number && ask(&gt(&f_, &Basic::zero())?, assumptions)? == Some(true))
            {
                positive.push((f_, power));
            } else if is(&f_, Predicate::negative) == Some(true)
                || (!number && ask(&lt(&f_, &Basic::zero())?, assumptions)? == Some(true))
            {
                if is(&power, Predicate::odd) == Some(true) {
                    sign = -sign;
                }
                positive.push((neg(&f_)?, power));
            } else {
                rest.push(pow(&f_, &power)?);
            }
        }
        if positive.is_empty() {
            return Ok(node.clone());
        }
        let mut result: Basic = Basic::one();
        for (f_, power) in &positive {
            result = mul(&[result, pow(f_, &mul(&[power.clone(), exponent.clone()])?)?])?;
        }
        let remaining: Basic = mul(&[int(sign), mul(&rest)?])?;
        mul(&[result, pow(&remaining, exponent)?])
    };
    crate::core::function::replace_where(expr, &|n: &Basic| n.is_pow(), &rewrite)
}

/// Whether `value` has logarithms of radicals or more than a few dozen
/// operations, the marks of an antiderivative taken along a complex branch
/// (`_unwieldy`).
fn unwieldy(value: &Basic) -> PolyResult<bool> {
    if ops(value)? > 40 {
        return Ok(true);
    }
    Ok(value
        .preorder()
        .iter()
        .any(|n| n.is_function(crate::core::basic::Function::Log) && n.args()[0].preorder().iter().any(|m| m.is_pow())))
}

/// The iterated integral over the stack, innermost variable first; a
/// sector between opposite bounds `(-h, h)` rescaled to `(-1, 1)` by `x =
/// h*u` when the integrator does not answer (`_iterated`).
fn iterated(
    integrand: &Basic,
    variables: &[Basic],
    stack: &Stack,
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let mut value: Basic = integrand.clone();
    for k in (0..variables.len()).rev() {
        let x: &Basic = &variables[k];
        let (lower, upper) = stack.bounds[k].clone();
        let mut outer: Vec<Basic> = assumptions.to_vec();
        for (variable, (lo, hi)) in variables.iter().zip(&stack.bounds).take(k) {
            if *lo != neg_oo() {
                outer.push(gt(variable, lo)?);
            }
            if *hi != oo() {
                outer.push(lt(variable, hi)?);
            }
        }
        if !value.has(x) {
            if lower == neg_oo() || upper == oo() {
                return Ok(None);
            }
            value = mul(&[value, sub(&upper, &lower)?])?;
            continue;
        }
        let mut inner: Vec<Basic> = outer.clone();
        if lower != neg_oo() {
            inner.push(gt(x, &lower)?);
        }
        if upper != oo() {
            inner.push(lt(x, &upper)?);
        }
        // an empty cell under its parameter condition
        if inner.iter().any(is_false) {
            return Ok(Some(Basic::zero()));
        }
        let piece: Basic = split_roots(&value, &inner)?;
        let mut result: Option<Basic> = match integrator(&piece, x, &lower, &upper, &outer) {
            Ok(r) => r,
            // contradictory bounds: an empty cell
            Err(PolyError::Value(_)) => return Ok(Some(Basic::zero())),
            Err(e) => return Err(e),
        };
        let bad = |r: &Option<Basic>| -> PolyResult<bool> {
            Ok(match r {
                None => true,
                Some(v) => v.has(&Basic::imaginary_unit()) || unwieldy(v)?,
            })
        };
        if piece != value && bad(&result)? {
            // the split roots under a < 0 gave a complex form where the
            // product gives it plainly
            let unsplit: Option<Basic> = match integrator(&value, x, &lower, &upper, &outer) {
                Ok(r) => r,
                Err(PolyError::Value(_)) => return Ok(Some(Basic::zero())),
                Err(e) => return Err(e),
            };
            if let Some(u) = unsplit {
                let better: bool = match &result {
                    None => true,
                    Some(r) => ops(&u)? < ops(r)?,
                };
                if better {
                    result = Some(u);
                }
            }
        }
        if result.is_none()
            && lower == neg(&upper)?
            && upper != oo()
            && ask(&gt(&upper, &Basic::zero())?, &outer)? == Some(true)
        {
            let u: Basic = fresh("u", false);
            let mut extra: Vec<Basic> = outer.clone();
            extra.extend([gt(&upper, &Basic::zero())?, gt(&u, &int(-1))?, lt(&u, &int(1))?]);
            let substituted: Basic = mul(&[
                crate::core::subs::subs(&value, x, &mul(&[upper.clone(), u.clone()])?)?,
                upper.clone(),
            ])?;
            let rescaled: Basic = split_roots(&substituted, &extra)?;
            result = integrator(&rescaled, &u, &int(-1), &int(1), &outer)?;
        }
        let Some(r) = result else { return Ok(None) };
        value = r;
    }
    Ok(Some(value))
}

/// The condition of the parameter cell of `cell`: the sign conditions of
/// the projection polynomials of the first `levels` levels; with
/// `explicit`, the cell bounded by its roots in explicit form where they
/// have one (`_parameter_condition`).
fn parameter_condition(cell: &std::sync::Arc<CadCell>, cad: &Cad, levels: usize, explicit: bool) -> R {
    cell_condition(cell, cad, 1, levels, explicit)
}

/// The sign conditions of the projection polynomials of the levels `first`
/// to `last` (1-based) at the ancestor of `cell` of level `last`; with
/// `explicit`, each level's variable between its roots in explicit form
/// where they have one (`_cell_condition`).
fn cell_condition(cell: &std::sync::Arc<CadCell>, cad: &Cad, first: usize, last: usize, explicit: bool) -> R {
    let mut ancestor: Option<std::sync::Arc<CadCell>> = Some(cell.clone());
    while let Some(a) = ancestor.clone() {
        if a.level() <= last {
            break;
        }
        ancestor = a.parent.clone();
    }
    let Some(ancestor) = ancestor else {
        return Ok(crate::core::logic::true_());
    };
    let mut parts: Vec<Basic> = Vec::new();
    for level in first..=last {
        let node: std::sync::Arc<CadCell> = ancestor_at(&ancestor, level);
        if explicit {
            if let Some(found) = explicit_cell(&node, cad, level)? {
                parts.extend(found);
                continue;
            }
        }
        let gens: &[Basic] = &cad.gens[..level];
        let sample = node.sample()?;
        for poly in &cad.projection[level - 1] {
            let expression: Basic = poly.as_expr()?;
            let s = sample.sign(&expression, gens)?;
            parts.push(if s.is_positive() {
                gt(&expression, &Basic::zero())?
            } else if s.is_negative() {
                lt(&expression, &Basic::zero())?
            } else {
                eq(&expression, &Basic::zero())?
            });
        }
    }
    and_(&parts)
}

/// The relations placing the variable of `level` in the cell `node` of
/// that level, the roots as expressions in the earlier variables; `None`
/// when a root has no explicit form (`_explicit_cell`).
fn explicit_cell(node: &std::sync::Arc<CadCell>, cad: &Cad, level: usize) -> PolyResult<Option<Vec<Basic>>> {
    let Some(parent) = node.parent.clone() else {
        return Ok(None);
    };
    let gens: &[Basic] = &cad.gens[..level];
    let x: &Basic = &gens[gens.len() - 1];
    let roots: Vec<Bound> = root_sequence(&parent, &poly_exprs(&cad.projection[level - 1])?, gens)?;
    let last: usize = node.index[node.index.len() - 1];
    if node.is_section() {
        let position: i64 = (last / 2) as i64 - 1;
        if position < 0 || position as usize >= roots.len() {
            return Ok(None);
        }
        let Some(root) = explicit_root(&roots[position as usize], &parent, gens)? else {
            return Ok(None);
        };
        return Ok(Some(vec![eq(x, &root)?]));
    }
    let position: usize = (last - 1) / 2;
    let mut parts: Vec<Basic> = Vec::new();
    if position >= 1 {
        let Some(lower) = explicit_root(&roots[position - 1], &parent, gens)? else {
            return Ok(None);
        };
        parts.push(gt(x, &lower)?);
    }
    if position < roots.len() {
        let Some(upper) = explicit_root(&roots[position], &parent, gens)? else {
            return Ok(None);
        };
        parts.push(lt(x, &upper)?);
    }
    Ok(Some(parts))
}

fn ancestor_at(cell: &std::sync::Arc<CadCell>, level: usize) -> std::sync::Arc<CadCell> {
    let mut current: std::sync::Arc<CadCell> = cell.clone();
    while current.level() > level {
        let Some(parent) = current.parent.clone() else { break };
        current = parent;
    }
    current
}

// -- polar, spherical and scaled coordinates

use crate::core::basic::RelOp;

fn relation_op(e: &Basic) -> Option<RelOp> {
    match e.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}

fn is_inequality(e: &Basic) -> bool {
    matches!(relation_op(e), Some(RelOp::Lt | RelOp::Le | RelOp::Gt | RelOp::Ge))
}

fn is_less(e: &Basic) -> bool {
    matches!(relation_op(e), Some(RelOp::Lt | RelOp::Le))
}

/// `lhs - rhs` of a relation.
fn difference(atom: &Basic) -> R {
    sub(&atom.args()[0], &atom.args()[1])
}

/// The relations of a conjunction of strict or weak inequalities; `None`
/// for any other formula (`_atoms`).
fn atoms(formula: &Basic) -> Option<Vec<Basic>> {
    let parts: Vec<Basic> = if matches!(formula.head(), Head::And) {
        formula.args().to_vec()
    } else {
        vec![formula.clone()]
    };
    if parts.iter().all(is_inequality) {
        Some(parts)
    } else {
        None
    }
}

/// `x**2 + y**2 + ...` (`_radius`).
fn radius_of(names: &[Basic]) -> R {
    add(&names
        .iter()
        .map(|v| pow(v, &int(2)))
        .collect::<PolyResult<Vec<Basic>>>()?)
}

/// `(upper, value)` when the relation bounds `x**2 + y**2 (+ z**2)`:
/// `upper` for a bound from above (`rho**2 < value`) (`_radial_atom`).
fn radial_atom(atom: &Basic, names: &[Basic]) -> PolyResult<Option<(bool, Basic)>> {
    let d: Basic = difference(atom)?;
    let Some(terms) = coefficients(&d, names)? else {
        return Ok(None);
    };
    let c: Basic = coefficient(&terms, &unit(names.len(), 0, 2));
    if c == Basic::zero() {
        return Ok(None);
    }
    let rest: Basic = expand(&sub(&d, &mul(&[c.clone(), radius_of(names)?])?)?)?;
    if names.iter().any(|n| rest.has(n)) {
        return Ok(None);
    }
    // c * rho**2 + rest < 0 (or <=, >, >=)
    let bound: Basic = neg(&div(&rest, &c)?)?;
    let positive: bool = is(&c, Predicate::positive) == Some(true);
    let negative: bool = is(&c, Predicate::negative) == Some(true);
    if !positive && !negative {
        return Ok(None);
    }
    Ok(Some((is_less(atom) == positive, bound)))
}

fn has_nan_or_zoo(e: &Basic) -> bool {
    e.has(&Basic::nan()) || e.has(&Basic::zoo())
}

/// `g(rho)` with `f == g(sqrt(x**2 + y**2 + ...))`, or `None` when `f` is
/// not a function of the distance to the origin (`_radial_integrand`).
fn radial_integrand(f: &Basic, names: &[Basic], rho: &Basic) -> PolyResult<Option<Basic>> {
    let radius: Basic = sqrt(&radius_of(names)?)?;
    let on = |first: &Basic| -> Vec<(Basic, Basic)> {
        let mut rule: Vec<(Basic, Basic)> = vec![(names[0].clone(), first.clone())];
        for v in &names[1..] {
            rule.push((v.clone(), Basic::zero()));
        }
        rule
    };
    let candidate: Basic = evaluated(f, &on(&radius))?;
    if has_nan_or_zoo(&candidate) {
        return Ok(None);
    }
    let d: Basic = sub(f, &candidate)?;
    let simpler: Option<Basic> = attempt(|| simplify(&d))?;
    if simpler.as_ref() != Some(&Basic::zero()) {
        // at points of every orthant, and under rotations: Heaviside(x)
        // agrees with 1 = Heaviside(rho) on the positive orthant alone
        if !crate::integrals::conditions::numerically_equal(f, &candidate, &[], 3)?
            || !crate::integrals::axisymmetric::rotation_invariant(f, names, 3)?
        {
            return Ok(None);
        }
    }
    Ok(Some(evaluated(f, &on(rho))?))
}

/// The area of the unit sphere `S^(n-1)`, `2 pi**(n/2)/gamma(n/2)`
/// (`_sphere_area`).
fn sphere_area(n: &Basic) -> R {
    let half: Basic = div(n, &int(2))?;
    div(
        &mul(&[int(2), pow(&Basic::pi(), &half)?])?,
        &crate::functions::special::gamma_functions::gamma(&half)?,
    )
}

fn refine(e: &Basic, assumptions: &[Basic]) -> R {
    crate::assumptions::refine::refine(e, assumptions, None)
}

/// The integral over a disc, an annulus or a ball (in any dimension) of a
/// function of the distance to the origin, in polar (spherical)
/// coordinates (`_radial`).
fn radial(
    f: &Basic,
    formula: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let Some(atoms) = atoms(formula).filter(|a| !a.is_empty()) else {
        return Ok(None);
    };
    let (mut lower, mut upper): (Option<Basic>, Option<Basic>) = (None, None);
    for atom in &atoms {
        let Some((is_upper, value)) = radial_atom(atom, names)? else {
            return Ok(None);
        };
        let slot: &mut Option<Basic> = if is_upper { &mut upper } else { &mut lower };
        if slot.is_some() {
            return Ok(None);
        }
        *slot = Some(value);
    }
    let Some(upper) = upper else { return Ok(None) };
    let rho: Basic = fresh("rho", true);
    let Some(g) = radial_integrand(f, names, &rho)? else {
        return Ok(None);
    };
    let positive: Option<bool> = ask(&gt(&upper, &Basic::zero())?, assumptions)?;
    if positive == Some(false) {
        return Ok(Some(Basic::zero()));
    }
    let mut inner: Vec<Basic> = assumptions.to_vec();
    if positive.is_none() {
        inner.push(gt(&upper, &Basic::zero())?);
    }
    let start: Basic = match &lower {
        None => Basic::zero(),
        Some(lower) => {
            if ask(&crate::core::relational::ge(lower, &Basic::zero())?, &inner)? != Some(true)
                || ask(&lt(lower, &upper)?, &inner)? != Some(true)
            {
                return Ok(None);
            }
            sqrt(lower)?
        }
    };
    let n: i64 = names.len() as i64;
    let measure: Basic = mul(&[sphere_area(&int(n))?, pow(&rho, &int(n - 1))?])?;
    let mut radius: Basic = sqrt(&upper)?;
    if let Some(refined) = attempt(|| refine(&radius, &inner))? {
        radius = refined;
    }
    let Some(mut value) = integrator(&mul(&[g, measure])?, &rho, &start, &radius, &inner)? else {
        return Ok(None);
    };
    if let Some(tidy) = attempt(|| refine(&value, &inner))? {
        value = tidy;
    }
    if positive.is_none() {
        return Ok(Some(conditional(&value, &upper, assumptions)?));
    }
    Ok(Some(value))
}

fn sorted_by_name(symbols: impl IntoIterator<Item = Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().collect();
    v.sort_by_key(symbol_name);
    v
}

/// `value` where `bound > 0` and `0` elsewhere: the plain `value` when
/// `bound` is never negative and `value` vanishes wherever `bound` does,
/// the `Piecewise` otherwise (`_conditional`).
fn conditional(value: &Basic, bound: &Basic, assumptions: &[Basic]) -> R {
    let parameters: Vec<Basic> = sorted_by_name(bound.free_symbols());
    // the parameters are real (a**2 >= 0 is undecided for a complex a)
    let mut real: Vec<Basic> = assumptions.to_vec();
    for p in &parameters {
        real.push(crate::assumptions::facts::element(p, &Basic::atom(Head::Reals))?);
    }
    if ask(&crate::core::relational::ge(bound, &Basic::zero())?, &real)? == Some(true) {
        let zeros: Option<Vec<Vec<(Basic, Basic)>>> = if parameters.is_empty() {
            Some(Vec::new())
        } else {
            attempt(|| zeros_of(bound, &parameters))?.flatten()
        };
        if let Some(zeros) = zeros {
            let mut all: bool = true;
            for zero in &zeros {
                if simplify(&evaluated(value, zero)?)? != Basic::zero() {
                    all = false;
                    break;
                }
            }
            if all {
                return Ok(value.clone());
            }
        }
    }
    crate::functions::elementary::piecewise::piecewise(&[
        (value.clone(), gt(bound, &Basic::zero())?),
        (Basic::zero(), crate::core::logic::true_()),
    ])
}

/// The points of the parameters where `bound` vanishes, solved for the
/// first parameter of which it is a polynomial with all its roots
/// explicit, or for a single parameter the finite solution set of the
/// solver under assumptions; `None` otherwise (`_zeros`).
fn zeros_of(bound: &Basic, parameters: &[Basic]) -> PolyResult<Option<Vec<Vec<(Basic, Basic)>>>> {
    for p in parameters {
        let Some(terms) = coefficients(bound, std::slice::from_ref(p))? else {
            continue;
        };
        if total_degree(&terms) < 1 {
            continue;
        }
        let found = match crate::polys::polyroots::roots(bound, std::slice::from_ref(p), true, true, None) {
            Ok(f) => f,
            Err(PolyError::Polys(..)) => continue,
            Err(e) => return Err(e),
        };
        if found.iter().map(|(_, m)| *m as i64).sum::<i64>() == total_degree(&terms) {
            return Ok(Some(found.iter().map(|(r, _)| vec![(p.clone(), r.clone())]).collect()));
        }
    }
    if parameters.len() != 1 {
        return Ok(None);
    }
    let p: &Basic = &parameters[0];
    let where_: Basic = crate::assumptions::solve::solve(
        &[eq(bound, &Basic::zero())?],
        Some(std::slice::from_ref(p)),
        &[],
        None,
        false,
    )?;
    if !matches!(where_.head(), Head::FiniteSet | Head::EmptySet) {
        return Ok(None);
    }
    Ok(Some(
        where_.args().iter().map(|v| vec![(p.clone(), v.clone())]).collect(),
    ))
}

/// A diagonal quadratic form with the square completed (`_form_atom`):
/// `(upper, coefficients, shifts, value)` for `sum(c_i*(x_i + s_i)**2) <
/// value` (`upper`) or `> value`, all `c_i` positive.
type Form = (bool, Vec<Basic>, Vec<Basic>, Basic);

fn form_atom(atom: &Basic, names: &[Basic], assumptions: &[Basic]) -> PolyResult<Option<Form>> {
    let mut d: Basic = difference(atom)?;
    let Some(terms) = coefficients(&d, names)? else {
        return Ok(None);
    };
    if total_degree(&terms) != 2 {
        return Ok(None);
    }
    let n: usize = names.len();
    let mut cs: Vec<Basic> = Vec::new();
    let mut linear: Vec<Basic> = Vec::new();
    for i in 0..n {
        let c: Basic = coefficient(&terms, &unit(n, i, 2));
        if c == Basic::zero() {
            return Ok(None);
        }
        cs.push(c);
        linear.push(coefficient(&terms, &unit(n, i, 1)));
    }
    // a cross term
    if terms
        .iter()
        .any(|(m, _)| m.iter().sum::<i64>() == 2 && m.iter().max() == Some(&1))
    {
        return Ok(None);
    }
    let mut sign: Option<bool> = None;
    for c in &cs {
        let positive: bool =
            is(c, Predicate::positive) == Some(true) || ask(&gt(c, &Basic::zero())?, assumptions)? == Some(true);
        let negative: bool =
            is(c, Predicate::negative) == Some(true) || ask(&lt(c, &Basic::zero())?, assumptions)? == Some(true);
        if !positive && !negative {
            return Ok(None);
        }
        match sign {
            None => sign = Some(positive),
            Some(s) if s != positive => return Ok(None),
            _ => {}
        }
    }
    let Some(sign) = sign else { return Ok(None) };
    if !sign {
        cs = cs.iter().map(neg).collect::<PolyResult<_>>()?;
        linear = linear.iter().map(neg).collect::<PolyResult<_>>()?;
        d = neg(&d)?;
    }
    let shifts: Vec<Basic> = linear
        .iter()
        .zip(&cs)
        .map(|(b, c)| div(b, &mul(&[int(2), c.clone()])?))
        .collect::<PolyResult<_>>()?;
    let mut completed: Basic = Basic::zero();
    for ((c, v), t) in cs.iter().zip(names).zip(&shifts) {
        completed = add(&[
            completed,
            mul(&[c.clone(), pow(&add(&[v.clone(), t.clone()])?, &int(2))?])?,
        ])?;
    }
    let rest: Basic = expand(&sub(&d, &completed)?)?;
    if names.iter().any(|n| rest.has(n)) {
        return Ok(None);
    }
    Ok(Some((is_less(atom) == sign, cs, shifts, neg(&rest)?)))
}

/// The integral over an ellipse or ellipsoid of a function which is radial
/// after the affine change of variables `x_i = u_i/sqrt(c_i) - s_i`,
/// through [`radial`] (`_scaled_radial`).
fn scaled_radial(
    f: &Basic,
    formula: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    if names.len() != 2 && names.len() != 3 {
        return Ok(None);
    }
    let Some(atoms) = atoms(formula).filter(|a| !a.is_empty()) else {
        return Ok(None);
    };
    type Part = (Vec<Basic>, Vec<Basic>, Basic);
    let (mut upper, mut lower): (Option<Part>, Option<Part>) = (None, None);
    for atom in &atoms {
        let Some((is_upper, cs, shifts, value)) = form_atom(atom, names, assumptions)? else {
            return Ok(None);
        };
        let slot: &mut Option<Part> = if is_upper { &mut upper } else { &mut lower };
        if slot.is_some() {
            return Ok(None);
        }
        *slot = Some((cs, shifts, value));
    }
    let Some((cs, shifts, value)) = upper else {
        return Ok(None);
    };
    // the polar route already looked at it
    if cs.iter().all(|c| *c == Basic::one()) && shifts.iter().all(|t| *t == Basic::zero()) {
        return Ok(None);
    }
    let us: Vec<Basic> = names.iter().map(|v| fresh(&symbol_name(v), false)).collect();
    let mut substitution: Vec<(Basic, Basic)> = Vec::new();
    let mut jacobian: Basic = Basic::one();
    for ((v, u), (c, t)) in names.iter().zip(&us).zip(cs.iter().zip(&shifts)) {
        substitution.push((v.clone(), sub(&div(u, &sqrt(c)?)?, t)?));
        jacobian = div(&jacobian, &sqrt(c)?)?;
    }
    let radius: Basic = radius_of(&us)?;
    let mut parts: Vec<Basic> = vec![lt(&radius, &value)?];
    if let Some((low_cs, low_shifts, low_value)) = lower {
        let ratio: Basic = div(&low_cs[0], &cs[0])?;
        for (c, d) in cs.iter().zip(&low_cs) {
            if simplify(&sub(d, &mul(&[ratio.clone(), c.clone()])?)?)? != Basic::zero() {
                return Ok(None);
            }
        }
        for (t, t_) in shifts.iter().zip(&low_shifts) {
            if simplify(&sub(t, t_)?)? != Basic::zero() {
                return Ok(None);
            }
        }
        if ask(&gt(&ratio, &Basic::zero())?, assumptions)? != Some(true) {
            return Ok(None);
        }
        parts.push(gt(&radius, &div(&low_value, &ratio)?)?);
    }
    let g: Basic = mul(&[evaluated(f, &substitution)?, jacobian])?;
    radial(&g, &and_(&parts)?, &us, assumptions, integrator)
}

// -- cylindrical coordinates and the bounds solved for the last variable

fn reals() -> Basic {
    Basic::atom(Head::Reals)
}

fn solve_in(statement: &Basic, x: &Basic, assumptions: &[Basic]) -> R {
    crate::assumptions::solve::solve(
        std::slice::from_ref(statement),
        Some(std::slice::from_ref(x)),
        assumptions,
        Some(&reals()),
        false,
    )
}

/// The integral in cylindrical coordinates: a disc or annulus condition on
/// `x**2 + y**2` (or none), bounds on `z` linear in `z` which depend on
/// `x, y` through `x**2 + y**2` only, and an integrand `g(x**2 + y**2, z)`
/// (`_cylindrical`).
fn cylindrical(
    f: &Basic,
    formula: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    if names.len() != 3 {
        return Ok(None);
    }
    let Some(atoms) = atoms(formula).filter(|a| !a.is_empty()) else {
        return Ok(None);
    };
    let (x, y, z) = (&names[0], &names[1], &names[2]);
    let plane: Vec<Basic> = vec![x.clone(), y.clone()];
    let (mut lower_radius, mut upper_radius): (Option<Basic>, Option<Basic>) = (None, None);
    let (mut lowers, mut uppers): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    let rho: Basic = fresh("rho", true);
    let in_plane = |e: &Basic| e.has(x) || e.has(y);
    for atom in &atoms {
        if !atom.has(z) {
            let Some((is_upper, value)) = radial_atom(atom, &plane)? else {
                return Ok(None);
            };
            let slot: &mut Option<Basic> = if is_upper { &mut upper_radius } else { &mut lower_radius };
            if slot.is_some() {
                return Ok(None);
            }
            *slot = Some(value);
            continue;
        }
        let Some((is_upper, expression)) = linear_bound(atom, z, assumptions)? else {
            return Ok(None);
        };
        let radial: Option<Basic> = if in_plane(&expression) {
            radial_integrand(&expression, &plane, &rho)?
        } else {
            Some(expression)
        };
        let Some(radial) = radial else { return Ok(None) };
        if is_upper {
            uppers.push(radial);
        } else {
            lowers.push(radial);
        }
    }
    let g: Option<Basic> = if in_plane(f) {
        radial_integrand(f, &plane, &rho)?
    } else {
        Some(f.clone())
    };
    let Some(g) = g else { return Ok(None) };
    let mut lower_z: Basic = neg_oo();
    let mut upper_z: Basic = oo();
    if !lowers.is_empty() {
        let Some(found) = extreme(&lowers, true, assumptions)? else {
            return Ok(None);
        };
        lower_z = found;
    }
    if !uppers.is_empty() {
        let Some(found) = extreme(&uppers, false, assumptions)? else {
            return Ok(None);
        };
        upper_z = found;
    }
    // the range of rho: inside the annulus, where the bounds on z are ordered
    let mut conditions: Vec<Basic> = Vec::new();
    if let Some(u) = &upper_radius {
        if ask(&gt(u, &Basic::zero())?, assumptions)? != Some(true) {
            return Ok(None);
        }
        conditions.push(lt(&rho, &sqrt(u)?)?);
    }
    if let Some(l) = &lower_radius {
        if ask(&crate::core::relational::ge(l, &Basic::zero())?, assumptions)? != Some(true) {
            return Ok(None);
        }
        conditions.push(gt(&rho, &sqrt(l)?)?);
    }
    if lower_z != neg_oo() && upper_z != oo() {
        conditions.push(lt(&lower_z, &upper_z)?);
    }
    if conditions.is_empty() {
        return Ok(None);
    }
    let mut with_rho: Vec<Basic> = assumptions.to_vec();
    with_rho.push(gt(&rho, &Basic::zero())?);
    let condition: Basic = and_(&conditions)?;
    let Some(where_) = attempt(|| solve_in(&condition, &rho, &with_rho))? else {
        return Ok(None);
    };
    // rho > 0 is not applied by solve to rho**2 < a with a symbolic
    let positive_axis: Basic = crate::sets::sets::interval(&Basic::zero(), &oo(), true, true)?;
    let Some(pieces) = interval_conditions(&crate::sets::sets::intersection(&[where_, positive_axis])?, &rho)? else {
        return Ok(None);
    };
    let mut total: Basic = Basic::zero();
    for piece in &pieces {
        let Some((start, stop)) = piece_limits(piece, &rho) else {
            return Ok(None);
        };
        let mut rho_assumptions: Vec<Basic> = assumptions.to_vec();
        rho_assumptions.push(gt(&rho, &start)?);
        if stop != oo() {
            rho_assumptions.push(lt(&rho, &stop)?);
        }
        let Some(inner) = integrator(
            &mul(&[g.clone(), rho.clone()])?,
            z,
            &lower_z,
            &upper_z,
            &rho_assumptions,
        )?
        else {
            return Ok(None);
        };
        let Some(outer) = integrator(&mul(&[int(2), Basic::pi(), inner])?, &rho, &start, &stop, assumptions)? else {
            return Ok(None);
        };
        total = add(&[total, outer])?;
    }
    Ok(Some(total))
}

/// `(start, stop)` of a conjunction `rho > start & rho < stop` (`0` and
/// `oo` when a side is missing) (`_piece_limits`).
fn piece_limits(piece: &Basic, rho: &Basic) -> Option<(Basic, Basic)> {
    let mut start: Basic = Basic::zero();
    let mut stop: Basic = oo();
    let parts: Vec<Basic> = if matches!(piece.head(), Head::And) {
        piece.args().to_vec()
    } else if *piece == crate::core::logic::true_() {
        Vec::new()
    } else {
        vec![piece.clone()]
    };
    for part in &parts {
        let op = relation_op(part)?;
        let (lhs, rhs) = (&part.args()[0], &part.args()[1]);
        let greater: bool = matches!(op, RelOp::Gt | RelOp::Ge);
        let less: bool = matches!(op, RelOp::Lt | RelOp::Le);
        if greater && lhs == rho {
            start = rhs.clone();
        } else if less && lhs == rho {
            stop = rhs.clone();
        } else if less && rhs == rho {
            start = lhs.clone();
        } else if greater && rhs == rho {
            stop = lhs.clone();
        } else {
            return None;
        }
    }
    Some((start, stop))
}

/// `(upper, bound)` for a relation linear in `y`: `y < bound` (`upper`) or
/// `y > bound`; `None` when it is not linear in `y` or the sign of the
/// coefficient of `y` is not known (`_linear_bound`).
fn linear_bound(atom: &Basic, y: &Basic, assumptions: &[Basic]) -> PolyResult<Option<(bool, Basic)>> {
    let d: Basic = difference(atom)?;
    let Some(terms) = coefficients(&d, std::slice::from_ref(y))? else {
        return Ok(None);
    };
    if total_degree(&terms) != 1 {
        return Ok(None);
    }
    let (p, q) = (coefficient(&terms, &[1]), coefficient(&terms, &[0]));
    // p*y + q < 0 (Lt, Le) or > 0 (Gt, Ge)
    let positive: bool =
        is(&p, Predicate::positive) == Some(true) || ask(&gt(&p, &Basic::zero())?, assumptions)? == Some(true);
    let negative: bool =
        is(&p, Predicate::negative) == Some(true) || ask(&lt(&p, &Basic::zero())?, assumptions)? == Some(true);
    if !positive && !negative {
        return Ok(None);
    }
    Ok(Some((is_less(atom) == positive, neg(&div(&q, &p)?)?)))
}

/// The bounds `(upper, bound)` on `y` from a relation: the linear one, else
/// the finite endpoints of the interval of `y` which the solver finds
/// under the assumptions; `None` when the solution is not one interval
/// (`_solved_bound`).
fn solved_bound(atom: &Basic, y: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<(bool, Basic)>>> {
    if let Some(linear) = linear_bound(atom, y, assumptions)? {
        return Ok(Some(vec![linear]));
    }
    let Some(where_) = attempt(|| solve_in(atom, y, assumptions))? else {
        return Ok(None);
    };
    if !matches!(where_.head(), Head::Interval) {
        return Ok(None);
    }
    let (left, right) = (&where_.args()[0], &where_.args()[1]);
    let mut found: Vec<(bool, Basic)> = Vec::new();
    if *left != neg_oo() {
        found.push((false, left.clone()));
    }
    if *right != oo() {
        found.push((true, right.clone()));
    }
    Ok(Some(found))
}

/// The largest (smallest) of the bounds, when the assumptions order them;
/// `None` otherwise (`_extreme`).
fn extreme(bounds: &[Basic], largest: bool, assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    use crate::core::relational::{ge, le};
    let mut best: Basic = bounds[0].clone();
    for other in &bounds[1..] {
        let beats: Basic = if largest { ge(other, &best)? } else { le(other, &best)? };
        if ask(&beats, assumptions)? == Some(true) {
            best = other.clone();
            continue;
        }
        let loses: Basic = if largest { le(other, &best)? } else { ge(other, &best)? };
        if ask(&loses, assumptions)? != Some(true) {
            return Ok(None);
        }
    }
    Ok(Some(best))
}

/// A set of real numbers as a list of conjunctions `a < x < b`, one per
/// interval (points are dropped: measure zero); `None` for a set which is
/// not a union of intervals (`_interval_conditions`).
fn interval_conditions(where_: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    match where_.head() {
        Head::Reals => Ok(Some(vec![crate::core::logic::true_()])),
        Head::EmptySet | Head::FiniteSet => Ok(Some(Vec::new())),
        Head::Interval => {
            let (left, right) = (&where_.args()[0], &where_.args()[1]);
            let mut parts: Vec<Basic> = Vec::new();
            if *left != neg_oo() {
                parts.push(gt(x, left)?);
            }
            if *right != oo() {
                parts.push(lt(x, right)?);
            }
            Ok(Some(vec![and_(&parts)?]))
        }
        Head::Union => {
            let mut pieces: Vec<Basic> = Vec::new();
            for member in where_.args() {
                let Some(found) = interval_conditions(member, x)? else {
                    return Ok(None);
                };
                pieces.extend(found);
            }
            Ok(Some(pieces))
        }
        _ => Ok(None),
    }
}

/// The integral through the bounds of the last variable solved from the
/// relations (Fubini) (`_solved_bounds`).
fn solved_bounds(
    f: &Basic,
    formula: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let Some(atoms) = atoms(formula).filter(|a| !a.is_empty()) else {
        return Ok(None);
    };
    let y: &Basic = &names[names.len() - 1];
    let outer_names: Vec<Basic> = names[..names.len() - 1].to_vec();
    let mut outer_atoms: Vec<Basic> = Vec::new();
    let (mut lowers, mut uppers): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    let mut outer_assumptions: Vec<Basic> = assumptions.to_vec();
    for atom in &atoms {
        if !atom.has(y) {
            outer_atoms.push(atom.clone());
            outer_assumptions.push(atom.clone());
        }
    }
    for atom in &atoms {
        if !atom.has(y) {
            continue;
        }
        let Some(found) = solved_bound(atom, y, &outer_assumptions)? else {
            return Ok(None);
        };
        for (is_upper, bound) in found {
            if is_upper {
                uppers.push(bound);
            } else {
                lowers.push(bound);
            }
        }
    }
    // the region lies where the bounds are real: y < log(x) restricts x
    for bound in lowers.iter().chain(&uppers) {
        let Some(conditions) = reality_conditions(bound)? else {
            return Ok(None);
        };
        for condition in conditions {
            if condition != crate::core::logic::true_() && !outer_atoms.contains(&condition) {
                outer_atoms.push(condition.clone());
                outer_assumptions.push(condition);
            }
        }
    }
    let mut lower: Basic = neg_oo();
    let mut upper: Basic = oo();
    if !lowers.is_empty() {
        let Some(found) = extreme(&lowers, true, &outer_assumptions)? else {
            return Ok(None);
        };
        lower = found;
    }
    if !uppers.is_empty() {
        let Some(found) = extreme(&uppers, false, &outer_assumptions)? else {
            return Ok(None);
        };
        upper = found;
    }
    if outer_names.is_empty() {
        if !outer_atoms.is_empty() {
            return Ok(None);
        }
        return integrator(f, y, &lower, &upper, assumptions);
    }
    // where the two bounds are in the right order
    let mut pieces: Vec<Basic> = vec![crate::core::logic::true_()];
    if lower != neg_oo() && upper != oo() {
        let ordered: Option<bool> = ask(&lt(&lower, &upper)?, &outer_assumptions)?;
        if ordered == Some(false) {
            return Ok(Some(Basic::zero()));
        }
        if ordered.is_none() {
            if outer_names.len() != 1 {
                return Ok(None);
            }
            let x: &Basic = &outer_names[0];
            let condition: Basic = lt(&lower, &upper)?;
            let Some(where_) = attempt(|| solve_in(&condition, x, &outer_assumptions))? else {
                return Ok(None);
            };
            let Some(found) = interval_conditions(&where_, x)? else {
                return Ok(None);
            };
            pieces = found;
        }
    }
    let mut inner_assumptions: Vec<Basic> = outer_assumptions.clone();
    if lower != neg_oo() {
        inner_assumptions.push(gt(y, &lower)?);
    }
    if upper != oo() {
        inner_assumptions.push(lt(y, &upper)?);
    }
    let Some(inner) = integrator(f, y, &lower, &upper, &inner_assumptions)? else {
        return Ok(None);
    };
    let mut total: Basic = Basic::zero();
    for piece in &pieces {
        let mut parts: Vec<Basic> = outer_atoms.clone();
        parts.push(piece.clone());
        let condition: Basic = and_(&parts)?;
        let value: Basic = if condition == crate::core::logic::true_() {
            // the other variables are unbounded: the whole line each,
            // innermost first
            let mut value: Basic = inner.clone();
            for v in outer_names.iter().rev() {
                let Some(next) = integrator(&value, v, &neg_oo(), &oo(), assumptions)? else {
                    return Ok(None);
                };
                value = next;
            }
            value
        } else {
            let found: Basic = integrate_by_ranges(
                &inner,
                &condition,
                Some(&outer_names),
                assumptions,
                Some(integrator),
                "lebesgue",
                None,
            )?;
            if found
                .preorder()
                .iter()
                .any(|n| matches!(n.head(), Head::IntegralByRanges))
            {
                return Ok(None);
            }
            found
        };
        total = add(&[total, value])?;
    }
    Ok(Some(total))
}

// -- radial integrals in a symbolic dimension

/// The functions real at every real argument (`_REAL_FUNCTIONS`).
fn real_function(f: crate::core::basic::Function) -> bool {
    use crate::core::basic::Function as F;
    matches!(
        f,
        F::Exp
            | F::Sin
            | F::Cos
            | F::Tan
            | F::Sinh
            | F::Cosh
            | F::Tanh
            | F::Atan
            | F::Asinh
            | F::Abs
            | F::Sign
            | F::Floor
            | F::Ceiling
    )
}

/// The conditions under which `e` is real: `u > 0` for `log(u)`, `u >= 0`
/// for a rational power, `-1 <= u <= 1` for `asin`, `acos`, and so on;
/// `None` for a function whose reality is not known
/// (`_reality_conditions`).
fn reality_conditions(e: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    use crate::core::basic::Function as F;
    use crate::core::relational::{ge, le};
    // SymPy's Piecewise is a function of unknown reality
    if e.preorder().iter().any(|n| matches!(n.head(), Head::Piecewise)) {
        return Ok(None);
    }
    let mut conditions: Vec<Basic> = Vec::new();
    let mut applications: Vec<Basic> = e.atoms(Some(&|n: &Basic| {
        matches!(n.head(), Head::Function(_) | Head::Undefined(_))
    }));
    applications.sort_by_key(crate::core::sorting::sort_key);
    for node in &applications {
        if node.args().is_empty() {
            continue;
        }
        let argument: &Basic = &node.args()[0];
        match node.head() {
            Head::Function(F::Log) => conditions.push(gt(argument, &Basic::zero())?),
            Head::Function(F::Asin | F::Acos) => {
                conditions.push(ge(argument, &int(-1))?);
                conditions.push(le(argument, &int(1))?);
            }
            Head::Function(F::Acosh) => conditions.push(ge(argument, &int(1))?),
            Head::Function(F::Atanh) => {
                conditions.push(gt(argument, &int(-1))?);
                conditions.push(lt(argument, &int(1))?);
            }
            Head::Function(f) if real_function(*f) => {}
            _ => return Ok(None),
        }
    }
    let mut powers: Vec<Basic> = e.atoms(Some(&|n: &Basic| n.is_pow()));
    powers.sort_by_key(crate::core::printing::sstr);
    for power in &powers {
        let (base, exponent) = (&power.args()[0], &power.args()[1]);
        if let Some(q) = exponent.as_rational() {
            if !q.is_int() && !base.free_symbols().is_empty() {
                let positive_numerator: bool = *q.numerator() > crate::numbers::Int::ZERO;
                conditions.push(if positive_numerator {
                    ge(base, &Basic::zero())?
                } else {
                    gt(base, &Basic::zero())?
                });
                continue;
            }
        }
        if !exponent.free_symbols().is_empty() && is(base, Predicate::positive) != Some(true) {
            return Ok(None);
        }
    }
    Ok(Some(conditions))
}

/// `Integral(f(r), R^n)` over the ball, annulus or whole space that
/// `formula` describes through bounds on the radial variable `r`: the area
/// of the unit sphere times `Integral(f * r**(n - 1), (r, a, R))`
/// (`_symbolic_radial`).
fn symbolic_radial(
    f: &Basic,
    formula: &Basic,
    r: &Basic,
    n: &Basic,
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let atoms: Vec<Basic> = if *formula == crate::core::logic::true_() {
        Vec::new()
    } else {
        let Some(found) = atoms(formula) else { return Ok(None) };
        found
    };
    let (mut lower, mut upper): (Option<Basic>, Option<Basic>) = (None, None);
    for atom in &atoms {
        let Some((is_upper, value)) = linear_bound(atom, r, assumptions)? else {
            return Ok(None);
        };
        let slot: &mut Option<Basic> = if is_upper { &mut upper } else { &mut lower };
        if slot.is_some() {
            return Ok(None);
        }
        *slot = Some(value);
    }
    let mut inner: Vec<Basic> = assumptions.to_vec();
    inner.push(gt(n, &Basic::zero())?);
    let mut start: Basic = lower.clone().unwrap_or_else(Basic::zero);
    if let Some(l) = &lower {
        if ask(&crate::core::relational::ge(l, &Basic::zero())?, &inner)? != Some(true) {
            return Ok(None);
        }
    }
    let mut end: Basic = upper.clone().unwrap_or_else(oo);
    if let Some(u) = &upper {
        if ask(&lt(&start, u)?, &inner)? != Some(true) {
            return Ok(None);
        }
    }
    let (mut f, mut r) = (f.clone(), r.clone());
    if f.has(&r) && ask(&gt(&r, &Basic::zero())?, &inner)?.is_none() {
        let rho: Basic = fresh("rho", true);
        let rule: Vec<(Basic, Basic)> = vec![(r.clone(), rho.clone())];
        f = evaluated(&f, &rule)?;
        start = evaluated(&start, &rule)?;
        end = evaluated(&end, &rule)?;
        r = rho;
    }
    let weight: Basic = pow(&r, &sub(n, &int(1))?)?;
    let Some(radial) = integrator(&mul(&[f, weight])?, &r, &start, &end, &inner)? else {
        return Ok(None);
    };
    let total: Basic = mul(&[sphere_area(n)?, radial])?;
    Ok(Some(attempt(|| simplify(&total))?.unwrap_or(total)))
}

// -- polynomials over polytopes

use crate::numbers::Rat;

fn rat_expr(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}

fn rat_int(n: i64) -> Rat {
    crate::numbers::rat(n, 1)
}

/// A bounded polytope `A x <= b` with rational `A`, `b` free of the
/// variables, and its vertices (`_Polytope`).
struct Polytope {
    rows: Vec<Vec<Rat>>,
    rhs: Vec<Basic>,
    vertices: Vec<Vec<Basic>>,
}

impl Polytope {
    /// Whether the vertex lies on the hyperplane of the facet (`on_facet`).
    fn on_facet(&self, vertex: usize, facet: usize) -> PolyResult<bool> {
        Ok(expand(&sub(
            &dot(&self.rows[facet], &self.vertices[vertex])?,
            &self.rhs[facet],
        )?)? == Basic::zero())
    }
}

fn dot(row: &[Rat], point: &[Basic]) -> R {
    let mut terms: Vec<Basic> = Vec::new();
    for (a, v) in row.iter().zip(point) {
        terms.push(mul(&[rat_expr(a), v.clone()])?);
    }
    add(&terms)
}

/// The inequalities as rows `a . x <= b`; `None` when one is not linear in
/// the variables with rational coefficients (`_linear_rows`).
fn linear_rows(atoms: &[Basic], names: &[Basic]) -> PolyResult<Option<(Vec<Vec<Rat>>, Vec<Basic>)>> {
    let n: usize = names.len();
    let (mut rows, mut rhs): (Vec<Vec<Rat>>, Vec<Basic>) = (Vec::new(), Vec::new());
    for atom in atoms {
        let mut d: Basic = difference(atom)?;
        if matches!(relation_op(atom), Some(RelOp::Gt | RelOp::Ge)) {
            d = neg(&d)?;
        }
        let Some(terms) = coefficients(&d, names)? else {
            return Ok(None);
        };
        if total_degree(&terms) > 1 {
            return Ok(None);
        }
        let mut row: Vec<Rat> = Vec::new();
        for i in 0..n {
            let Some(c) = coefficient(&terms, &unit(n, i, 1)).as_rational().cloned() else {
                return Ok(None);
            };
            row.push(c);
        }
        if row.iter().all(|c| *c == Rat::ZERO) {
            return Ok(None);
        }
        rows.push(row);
        rhs.push(neg(&coefficient(&terms, &vec![0; n]))?);
    }
    Ok(Some((rows, rhs)))
}

/// Whether `A x <= b` is bounded: no direction `d != 0` with `A d <= 0`, by
/// linear programming over the recession cone cut to the unit cube
/// (`_bounded`).
fn bounded(rows: &[Vec<Rat>]) -> PolyResult<Option<bool>> {
    use crate::solvers::simplex::{linear_program, Status};
    let d: usize = rows[0].len();
    let mut a: Vec<Vec<Rat>> = rows.to_vec();
    for sign in [1, -1] {
        for i in 0..d {
            a.push((0..d).map(|j| if j == i { rat_int(sign) } else { Rat::ZERO }).collect());
        }
    }
    let mut b: Vec<Rat> = vec![Rat::ZERO; rows.len()];
    b.extend(vec![Rat::ONE; 2 * d]);
    for j in 0..d {
        for orientation in [1, -1] {
            let c: Vec<Rat> = (0..d)
                .map(|i| if i == j { rat_int(orientation) } else { Rat::ZERO })
                .collect();
            let found = linear_program(&c, &a, &b)?;
            let (Status::Optimal, Some(value)) = (found.status, found.value) else {
                return Ok(None);
            };
            if value > Rat::ZERO {
                return Ok(Some(false));
            }
        }
    }
    Ok(Some(true))
}

/// The solution of the square system `matrix . x = rhs` with a rational
/// matrix, the coordinates expanded; `None` when the matrix is singular
/// (`_solve_rational`).
fn solve_rational(matrix: &[Vec<Rat>], rhs: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let n: usize = matrix.len();
    let mut inverse: Vec<Vec<Rat>> = (0..n)
        .map(|i| (0..n).map(|j| if j == i { Rat::ONE } else { Rat::ZERO }).collect())
        .collect();
    let mut rows: Vec<Vec<Rat>> = matrix.to_vec();
    for col in 0..n {
        let Some(pivot) = (col..n).find(|&i| rows[i][col] != Rat::ZERO) else {
            return Ok(None);
        };
        rows.swap(col, pivot);
        inverse.swap(col, pivot);
        let scale: Rat = rows[col][col].clone();
        rows[col] = rows[col].iter().map(|a| a / &scale).collect();
        inverse[col] = inverse[col].iter().map(|a| a / &scale).collect();
        for i in 0..n {
            if i != col && rows[i][col] != Rat::ZERO {
                let factor: Rat = rows[i][col].clone();
                let (pivot_row, pivot_inverse) = (rows[col].clone(), inverse[col].clone());
                rows[i] = rows[i]
                    .iter()
                    .zip(&pivot_row)
                    .map(|(a, b)| a - &(&factor * b))
                    .collect();
                inverse[i] = inverse[i]
                    .iter()
                    .zip(&pivot_inverse)
                    .map(|(a, b)| a - &(&factor * b))
                    .collect();
            }
        }
    }
    Ok(Some(
        (0..n)
            .map(|k| expand(&dot(&inverse[k], rhs)?))
            .collect::<PolyResult<_>>()?,
    ))
}

/// The vertices of `A x <= b`: the intersections of `d` facet hyperplanes
/// which satisfy the other inequalities, decided under the assumptions
/// (`None` when undecided) (`_vertices`).
fn vertices_of(rows: &[Vec<Rat>], rhs: &[Basic], assumptions: &[Basic]) -> PolyResult<Option<Vec<Vec<Basic>>>> {
    let d: usize = rows[0].len();
    let mut vertices: Vec<Vec<Basic>> = Vec::new();
    for subset in combinations(rows.len(), d) {
        let chosen: Vec<Vec<Rat>> = subset.iter().map(|&i| rows[i].clone()).collect();
        let values: Vec<Basic> = subset.iter().map(|&i| rhs[i].clone()).collect();
        let Some(coordinates) = solve_rational(&chosen, &values)? else {
            continue;
        };
        let mut feasible: bool = true;
        for i in 0..rows.len() {
            if subset.contains(&i) {
                continue;
            }
            let slack: Basic = expand(&sub(&dot(&rows[i], &coordinates)?, &rhs[i])?)?;
            if crate::core::assumptions::is_number(&slack) {
                if is(&slack, Predicate::positive) == Some(true) {
                    feasible = false;
                    break;
                }
                continue;
            }
            match ask(&crate::core::relational::le(&slack, &Basic::zero())?, assumptions)? {
                None => return Ok(None),
                Some(false) => {
                    feasible = false;
                    break;
                }
                Some(true) => {}
            }
        }
        if feasible && !vertices.contains(&coordinates) {
            vertices.push(coordinates);
        }
    }
    Ok(Some(vertices))
}

/// The `k`-subsets of `0..n` in lexicographic order (`combinations`).
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn go(start: usize, n: usize, k: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            go(i + 1, n, k, current, out);
            current.pop();
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    if k <= n {
        go(0, n, k, &mut Vec::new(), &mut out);
    }
    out
}

/// The rank of a matrix of polynomials in the parameters: the largest
/// order of a minor which does not expand to zero (`_rank`).
fn rank(matrix: &[Vec<Basic>]) -> PolyResult<usize> {
    if matrix.is_empty() {
        return Ok(0);
    }
    let (rows, cols) = (matrix.len(), matrix[0].len());
    for k in (1..=rows.min(cols)).rev() {
        for chosen in combinations(rows, k) {
            for columns in combinations(cols, k) {
                let minor: Vec<Vec<Basic>> = chosen
                    .iter()
                    .map(|&i| columns.iter().map(|&j| matrix[i][j].clone()).collect())
                    .collect();
                if expand(&crate::solvers::ode_cases::determinant(&minor)?)? != Basic::zero() {
                    return Ok(k);
                }
            }
        }
    }
    Ok(0)
}

/// The dimension of the affine hull of the points (`_affine_rank`).
fn affine_rank(points: &[Vec<Basic>]) -> PolyResult<usize> {
    if points.len() < 2 {
        return Ok(0);
    }
    let base: &Vec<Basic> = &points[0];
    let mut differences: Vec<Vec<Basic>> = Vec::new();
    for p in &points[1..] {
        differences.push(
            (0..base.len())
                .map(|k| sub(&p[k], &base[k]))
                .collect::<PolyResult<_>>()?,
        );
    }
    rank(&differences)
}

/// The pulling triangulation of a face of dimension `k` given by its
/// vertices: cones from the first vertex over the triangulations of the
/// facets of the face not containing it (`_pulling`).
fn pulling(polytope: &Polytope, face: &[usize], k: usize) -> PolyResult<Vec<Vec<usize>>> {
    if k == 0 {
        return Ok(vec![vec![face[0]]]);
    }
    let apex: usize = face[0];
    let mut simplices: Vec<Vec<usize>> = Vec::new();
    let mut seen: Vec<Vec<usize>> = Vec::new();
    for facet in 0..polytope.rows.len() {
        if polytope.on_facet(apex, facet)? {
            continue;
        }
        let mut on_facet: Vec<usize> = Vec::new();
        for &v in face {
            if polytope.on_facet(v, facet)? {
                on_facet.push(v);
            }
        }
        let mut key: Vec<usize> = on_facet.clone();
        key.sort_unstable();
        if on_facet.len() < k || seen.contains(&key) {
            continue;
        }
        let points: Vec<Vec<Basic>> = on_facet.iter().map(|&v| polytope.vertices[v].clone()).collect();
        if affine_rank(&points)? != k - 1 {
            continue;
        }
        seen.push(key);
        for simplex in pulling(polytope, &on_facet, k - 1)? {
            let mut cone: Vec<usize> = vec![apex];
            cone.extend(simplex);
            simplices.push(cone);
        }
    }
    Ok(simplices)
}

fn factorial_u(n: i64) -> crate::numbers::Int {
    let mut f: crate::numbers::Int = crate::numbers::Int::ONE;
    for i in 2..=n {
        f *= crate::numbers::Int::from(i);
    }
    f
}

/// `Integral(f, simplex)` by the affine map from the standard simplex and
/// Dirichlet's formula `Integral(u**a) = prod(a_i!) / (|a| + d)!`; `None`
/// when the sign of the Jacobian is undecided (`_simplex_integral`).
fn simplex_integral(
    f: &Basic,
    names: &[Basic],
    vertices: &[Vec<Basic>],
    assumptions: &[Basic],
) -> PolyResult<Option<Basic>> {
    let d: usize = names.len();
    let base: &Vec<Basic> = &vertices[0];
    let us: Vec<Basic> = (0..d).map(|i| fresh(&format!("u{}_", i), false)).collect();
    let mut substitution: Vec<(Basic, Basic)> = Vec::new();
    for (k, v) in names.iter().enumerate() {
        let mut terms: Vec<Basic> = vec![base[k].clone()];
        for (i, u) in us.iter().enumerate() {
            terms.push(mul(&[sub(&vertices[i + 1][k], &base[k])?, u.clone()])?);
        }
        substitution.push((v.clone(), add(&terms)?));
    }
    let mut matrix: Vec<Vec<Basic>> = Vec::new();
    for i in 0..d {
        matrix.push(
            (0..d)
                .map(|k| sub(&vertices[i + 1][k], &base[k]))
                .collect::<PolyResult<_>>()?,
        );
    }
    let jacobian: Basic = expand(&crate::solvers::ode_cases::determinant(&matrix)?)?;
    if jacobian == Basic::zero() {
        return Ok(Some(Basic::zero()));
    }
    let volume: Basic = if crate::core::assumptions::is_number(&jacobian) {
        if is(&jacobian, Predicate::extended_real) != Some(true) {
            return Ok(None);
        }
        crate::functions::elementary::complexes::abs_(&jacobian)?
    } else if ask(&gt(&jacobian, &Basic::zero())?, assumptions)? == Some(true) {
        jacobian
    } else if ask(&lt(&jacobian, &Basic::zero())?, assumptions)? == Some(true) {
        neg(&jacobian)?
    } else {
        return Ok(None);
    };
    let Some(pulled) = coefficients(&evaluated(f, &substitution)?, &us)? else {
        return Ok(None);
    };
    let mut total: Basic = Basic::zero();
    for (monomial, c) in &pulled {
        let mut numerator: crate::numbers::Int = crate::numbers::Int::ONE;
        for a in monomial {
            numerator *= factorial_u(*a);
        }
        let denominator = factorial_u(monomial.iter().sum::<i64>() + d as i64);
        let weight: Basic = Basic::from_rational(Rat::from(numerator) / Rat::from(denominator));
        total = add(&[total, mul(&[c.clone(), weight])?])?;
    }
    Ok(Some(mul(&[volume, total])?))
}

/// A polynomial over a bounded polytope, summed over the simplices of the
/// pulling triangulation; `None` when the region is not a bounded polytope
/// with the vertices decided, or the integrand not a polynomial
/// (`_polytope`).
fn polytope(f: &Basic, formula: &Basic, names: &[Basic], assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    let Some(atoms) = atoms(formula).filter(|a| !a.is_empty()) else {
        return Ok(None);
    };
    if atoms.iter().any(|atom| !names.iter().any(|n| atom.has(n))) {
        return Ok(None);
    }
    let Some((rows, rhs)) = linear_rows(&atoms, names)? else {
        return Ok(None);
    };
    if coefficients(f, names)?.is_none() {
        return Ok(None);
    }
    if bounded(&rows)? != Some(true) {
        return Ok(None);
    }
    let Some(vertices) = vertices_of(&rows, &rhs, assumptions)? else {
        return Ok(None);
    };
    if vertices.is_empty() {
        return Ok(Some(Basic::zero()));
    }
    let d: usize = names.len();
    if affine_rank(&vertices)? < d {
        return Ok(Some(Basic::zero()));
    }
    let all: Vec<usize> = (0..vertices.len()).collect();
    let shape = Polytope { rows, rhs, vertices };
    let mut total: Basic = Basic::zero();
    for simplex in pulling(&shape, &all, d)? {
        if simplex.len() != d + 1 {
            return Ok(None);
        }
        let points: Vec<Vec<Basic>> = simplex.iter().map(|&i| shape.vertices[i].clone()).collect();
        let Some(value) = simplex_integral(f, names, &points, assumptions)? else {
            return Ok(None);
        };
        total = add(&[total, value])?;
    }
    let tidy: Option<Basic> = attempt(|| crate::polys::polytools::factor(&total, &[], false, true))?;
    Ok(Some(match tidy {
        Some(t) => t,
        None => expand(&total)?,
    }))
}

// -- the driver

/// The integral of `integrand` over the region `condition`
/// (`integrate_by_ranges`): the routes of polar, scaled and cylindrical
/// coordinates, polytopes, rotational symmetry, the decomposition, the
/// bounds solved for the last variable and the other orders of the
/// variables; the unevaluated `IntegralByRanges` when none applies.
#[allow(clippy::too_many_arguments)]
pub fn integrate_by_ranges(
    integrand: &Basic,
    condition: &Basic,
    variables: Option<&[Basic]>,
    assumptions: &[Basic],
    integrator: Option<Integrator>,
    measure: &str,
    dimension: Option<&Basic>,
) -> R {
    integrate_by_ranges_with(
        integrand,
        condition,
        variables,
        assumptions,
        integrator,
        measure,
        dimension,
        None,
    )
}

/// [`integrate_by_ranges`] with the `parameters` excluded from the
/// variables found in the condition.
#[allow(clippy::too_many_arguments)]
pub fn integrate_by_ranges_with(
    integrand: &Basic,
    condition: &Basic,
    variables: Option<&[Basic]>,
    assumptions: &[Basic],
    integrator: Option<Integrator>,
    measure: &str,
    dimension: Option<&Basic>,
    parameters: Option<&[Basic]>,
) -> R {
    let node: Basic = integral_by_ranges(integrand, condition, variables, measure, dimension, parameters)?;
    let default_integrator = definite;
    let inner: Integrator = integrator.unwrap_or(&default_integrator);
    let (f, names) = (node.args()[0].clone(), node.args()[2].args().to_vec());
    let mut formula: Basic = crate::assumptions::facts::normalize(&node.args()[1])?;
    if measure == MEASURES[0] && dimension.is_none() {
        formula = full_measure(&formula, &names)?;
        if is_false(&formula) {
            return Ok(Basic::zero());
        }
    }
    let extra: &[Basic] = assumptions;
    if measure == "hausdorff" {
        if let Some(revolved) =
            crate::integrals::axisymmetric::axisymmetric_integral(&f, &formula, &names, extra, measure, inner)?
        {
            return Ok(revolved);
        }
        return Ok(hausdorff(&f, &formula, &names, extra, inner)?.unwrap_or(node));
    }
    if let Some(n) = dimension {
        return Ok(symbolic_radial(&f, &formula, &names[0], n, extra, inner)?.unwrap_or(node));
    }
    if let Some(v) = radial(&f, &formula, &names, extra, inner)? {
        return Ok(v);
    }
    if let Some(v) = scaled_radial(&f, &formula, &names, extra, inner)? {
        return Ok(v);
    }
    if let Some(v) = cylindrical(&f, &formula, &names, extra, inner)? {
        return Ok(v);
    }
    if let Some(v) = polytope(&f, &formula, &names, extra)? {
        return Ok(v);
    }
    if let Some(v) =
        crate::integrals::axisymmetric::axisymmetric_integral(&f, &formula, &names, extra, "lebesgue", inner)?
    {
        return Ok(v);
    }
    if let Some(v) = decomposed(&f, &formula, &names, extra, inner)? {
        return Ok(v);
    }
    if let Some(v) = solved_bounds(&f, &formula, &names, extra, inner)? {
        return Ok(v);
    }
    for order in other_orders(&names) {
        // a bound with no explicit form in the last variable may be
        // explicit in another
        if let Some(v) = attempt(|| decomposed(&f, &formula, &order, extra, inner))?.flatten() {
            return Ok(v);
        }
    }
    Ok(node)
}

/// The condition without its equations and non-equations in the
/// integration variables (`Eq` false, `Ne` true): they change the set on a
/// set of Lebesgue measure zero (`_full_measure`).
fn full_measure(formula: &Basic, names: &[Basic]) -> R {
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    let equations: Vec<Basic> = formula.atoms(Some(&|n: &Basic| matches!(relation_op(n), Some(RelOp::Eq | RelOp::Ne))));
    for atom in equations {
        let d: Basic = difference(&atom)?;
        if names.iter().any(|n| d.free_symbols().contains(n)) {
            let value: Basic = if matches!(relation_op(&atom), Some(RelOp::Eq)) {
                crate::core::logic::false_()
            } else {
                crate::core::logic::true_()
            };
            replacements.push((atom, value));
        }
    }
    if replacements.is_empty() {
        return Ok(formula.clone());
    }
    evaluated(formula, &replacements)
}

/// The other orders of the variables to try: every permutation of up to
/// three variables, the reversed order beyond (`_other_orders`).
fn other_orders(names: &[Basic]) -> Vec<Vec<Basic>> {
    if names.len() < 2 {
        return Vec::new();
    }
    if names.len() > 3 {
        return vec![names.iter().rev().cloned().collect()];
    }
    permutations(names)
        .into_iter()
        .filter(|order| order.as_slice() != names)
        .collect()
}

/// The permutations in Python's `itertools.permutations` order.
fn permutations(items: &[Basic]) -> Vec<Vec<Basic>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out: Vec<Vec<Basic>> = Vec::new();
    for i in 0..items.len() {
        let mut rest: Vec<Basic> = items.to_vec();
        let first: Basic = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, first.clone());
            out.push(tail);
        }
    }
    out
}

/// The compiled condition and the decomposition sign-invariant for its
/// polynomials; `None` when the condition is not polynomial
/// (`_decomposition`).
fn decomposition(formula: &Basic, gens: &[Basic]) -> PolyResult<Option<(crate::polys::cad::qe::Compiled, Cad)>> {
    let mut polys: Vec<crate::polys::polytools::Poly> = Vec::new();
    let mut attempt_it = || -> PolyResult<(crate::polys::cad::qe::Compiled, Cad)> {
        let compiled = crate::polys::cad::qe::compile(formula, gens, &mut polys)?;
        let exprs: Vec<Basic> = poly_exprs(&polys)?;
        let cad: Cad = crate::polys::cad::lifting::cylindrical_algebraic_decomposition(&exprs, gens, None)?;
        Ok((compiled, cad))
    };
    match attempt_it() {
        Ok(v) => Ok(Some(v)),
        Err(PolyError::Value(_) | PolyError::Type(_) | PolyError::Polys(..)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// The integral through the cylindrical algebraic decomposition, or `None`
/// when the condition is not polynomial, a bound has no explicit form or
/// an inner integral is not computed (`_decomposed`).
fn decomposed(
    f: &Basic,
    formula: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let parameters: Vec<Basic> = sorted_by_name(formula.free_symbols().into_iter().filter(|s| !names.contains(s)));
    let mut gens: Vec<Basic> = parameters.clone();
    gens.extend(names.iter().cloned());
    let Some((compiled, cad)) = decomposition(formula, &gens)? else {
        return Ok(None);
    };
    let m: usize = parameters.len();
    // the values by the parameter cell, in the order of the first one
    let mut values: Vec<(std::sync::Arc<CadCell>, Basic)> = Vec::new();
    for cell in cad.cells() {
        let signs = cell.signs.lock().expect("signs").clone();
        if !compiled.call(&signs) {
            continue;
        }
        // a section: measure zero
        if cell.index[m..].iter().any(|i| i % 2 == 0) {
            continue;
        }
        let base: std::sync::Arc<CadCell> = if m > 0 { ancestor_at(cell, m) } else { cell.clone() };
        let condition: Basic = if m > 0 {
            parameter_condition(cell, &cad, m, false)?
        } else {
            crate::core::logic::true_()
        };
        // a refuted case: its bounds need no explicit form
        if m > 0 && ask(&condition, assumptions)? == Some(false) {
            continue;
        }
        let Some(stack) = stack(cell, &cad, m + 1, &[], &[])? else {
            return Ok(None);
        };
        let mut with_condition: Vec<Basic> = assumptions.to_vec();
        if m > 0 {
            with_condition.push(condition.clone());
        }
        let Some(mut value) = iterated(f, names, &stack, &with_condition, integrator)? else {
            return Ok(None);
        };
        if m > 0 {
            if let Some(refined) = attempt(|| refine(&value, &with_condition))? {
                value = refined;
            }
        }
        match values.iter_mut().find(|(c, _)| std::sync::Arc::ptr_eq(c, &base)) {
            Some(entry) => entry.1 = add(&[entry.1.clone(), value])?,
            None => values.push((base, value)),
        }
    }
    Ok(Some(cases(&values, &cad, m, assumptions)?))
}

fn all_odd(index: &[usize]) -> bool {
    index.iter().all(|i| i % 2 == 1)
}

/// The sum of the values of the cells over the parameter cells, as a
/// `Piecewise` over the conditions of the parameter cells (the plain sum
/// without parameters) (`_cases`).
fn cases(values: &[(std::sync::Arc<CadCell>, Basic)], cad: &Cad, m: usize, assumptions: &[Basic]) -> R {
    if m == 0 {
        return add(&values.iter().map(|(_, v)| v.clone()).collect::<Vec<Basic>>());
    }
    let mut found: Vec<(Basic, Basic, std::sync::Arc<CadCell>)> = Vec::new();
    for base in cad.cells_at(m)? {
        let condition: Basic = parameter_condition(base, cad, m, true)?;
        if ask(&condition, assumptions)? == Some(false) {
            continue;
        }
        let value: Basic = values
            .iter()
            .find(|(c, _)| std::sync::Arc::ptr_eq(c, base))
            .map_or_else(Basic::zero, |(_, v)| v.clone());
        found.push((value, condition, base.clone()));
    }
    if found.is_empty() {
        return Ok(Basic::zero());
    }
    if found.len() == 1 {
        return Ok(found[0].0.clone());
    }
    if let Some(whole) = one_expression(&found)? {
        return Ok(whole);
    }
    let mut grouped: Vec<(Basic, Vec<Basic>)> = Vec::new();
    for (value, condition, _) in &found {
        match grouped.iter_mut().find(|(v, _)| v == value) {
            Some(entry) => entry.1.push(condition.clone()),
            None => grouped.push((value.clone(), vec![condition.clone()])),
        }
    }
    // the most frequent value last, as the otherwise branch (stable)
    grouped.sort_by_key(|(_, conditions)| std::cmp::Reverse(conditions.len()));
    let mut branches: Vec<(Basic, Basic)> = Vec::new();
    let count: usize = grouped.len();
    for (i, (value, conditions)) in grouped.into_iter().enumerate() {
        let condition: Basic = if i + 1 < count {
            crate::core::logic::or_(&conditions)
        } else {
            crate::core::logic::true_()
        };
        branches.push((value, condition));
    }
    crate::functions::elementary::piecewise::piecewise(&branches)
}

/// The common value of the cases when the full-dimensional parameter cells
/// all have the same value and each lower-dimensional cell's value is that
/// expression at the cell; `None` otherwise (`_one_expression`).
fn one_expression(found: &[(Basic, Basic, std::sync::Arc<CadCell>)]) -> PolyResult<Option<Basic>> {
    let sectors: Vec<&Basic> = found
        .iter()
        .filter(|(_, _, b)| all_odd(&b.index))
        .map(|(v, _, _)| v)
        .collect();
    if sectors.is_empty() || sectors[1..].iter().any(|v| *v != sectors[0]) {
        return Ok(None);
    }
    let common: Basic = sectors[0].clone();
    for (value, condition, base) in found {
        if all_odd(&base.index) {
            continue;
        }
        let atoms: Vec<Basic> = if matches!(condition.head(), Head::And) {
            condition.args().to_vec()
        } else {
            vec![condition.clone()]
        };
        let mut rule: Vec<(Basic, Basic)> = Vec::new();
        for atom in &atoms {
            if matches!(relation_op(atom), Some(RelOp::Eq)) && atom.args()[0].is_symbol() {
                rule.push((atom.args()[0].clone(), atom.args()[1].clone()));
            }
        }
        if rule.is_empty() {
            return Ok(None);
        }
        let restricted: Option<Basic> =
            attempt(|| simplify(&sub(&evaluated(&common, &rule)?, &evaluated(value, &rule)?)?))?;
        if restricted != Some(Basic::zero()) {
            return Ok(None);
        }
    }
    Ok(Some(common))
}

/// The conjunction without the atoms implied by the others (`_pruned`).
fn pruned(condition: &Basic) -> R {
    let parts: Vec<Basic> = if matches!(condition.head(), Head::And) {
        condition.args().to_vec()
    } else {
        vec![condition.clone()]
    };
    let mut kept: Vec<Basic> = parts.clone();
    for (i, part) in parts.iter().enumerate() {
        // `other is not part`: by position among the original atoms
        let position: Option<usize> = kept.iter().position(|k| k == part);
        let rest: Vec<Basic> = match position {
            Some(p) => kept
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != p)
                .map(|(_, k)| k.clone())
                .collect(),
            None => kept.clone(),
        };
        let _ = i;
        if !rest.is_empty() && ask(part, &rest)? == Some(true) {
            kept = rest;
        }
    }
    and_(&kept)
}

/// The integral of `f` with the Hausdorff measure over a cell of dimension
/// `n - k` which is a section at `k` levels: each section variable the
/// explicit branch of its root, the Gram determinant of the graph
/// parametrisation as the weight (`_section_value`).
#[allow(clippy::too_many_arguments)]
fn section_value(
    cell: &std::sync::Arc<CadCell>,
    cad: &Cad,
    m: usize,
    f: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let n: usize = cad.gens.len();
    let sections: Vec<usize> = ((m + 1)..=n).filter(|&level| cell.index[level - 1] % 2 == 0).collect();
    if sections.is_empty() {
        return Ok(None);
    }
    let mut branches: Vec<(Basic, Basic)> = Vec::new();
    for &k in &sections {
        let node: std::sync::Arc<CadCell> = ancestor_at(cell, k);
        let Some(parent) = node.parent.clone() else {
            return Ok(None);
        };
        let gens: &[Basic] = &cad.gens[..k];
        let roots: Vec<Bound> = root_sequence(&parent, &poly_exprs(&cad.projection[k - 1])?, gens)?;
        let position: i64 = (node.index[node.index.len() - 1] / 2) as i64 - 1;
        if position < 0 || position as usize >= roots.len() {
            return Ok(None);
        }
        let Some(phi) = explicit_root(&roots[position as usize], &parent, gens)? else {
            return Ok(None);
        };
        let value: Basic = evaluated(&phi, &branches)?;
        branches.push((names[k - m - 1].clone(), value));
    }
    let solved: Vec<Basic> = sections.iter().map(|&k| names[k - m - 1].clone()).collect();
    let rest: Vec<Basic> = names.iter().filter(|v| !solved.contains(v)).cloned().collect();
    let branch_of = |v: &Basic| -> Basic {
        branches
            .iter()
            .find(|(s, _)| s == v)
            .map(|(_, b)| b.clone())
            .expect("a branch")
    };
    let mut jacobian: Vec<Vec<Basic>> = Vec::new();
    for v in &solved {
        jacobian.push(
            rest.iter()
                .map(|u| diff1(&branch_of(v), u))
                .collect::<PolyResult<_>>()?,
        );
    }
    let mut gram: Basic = if rest.is_empty() {
        Basic::one()
    } else {
        let mut matrix: Vec<Vec<Basic>> = Vec::new();
        for i in 0..rest.len() {
            let mut row: Vec<Basic> = Vec::new();
            for j in 0..rest.len() {
                let entry: Basic = if i == j { Basic::one() } else { Basic::zero() };
                let mut terms: Vec<Basic> = vec![entry];
                for jac in &jacobian {
                    terms.push(mul(&[jac[i].clone(), jac[j].clone()])?);
                }
                row.push(add(&terms)?);
            }
            matrix.push(row);
        }
        crate::solvers::ode_cases::determinant(&matrix)?
    };
    if let Some(simpler) = attempt(|| simplify(&gram))? {
        gram = simpler;
    }
    let mut g: Basic = mul(&[evaluated(f, &branches)?, sqrt(&gram)?])?;
    // a point: the counting measure
    if rest.is_empty() {
        return Ok(Some(g));
    }
    let Some(stack) = stack(cell, cad, m + 1, &sections, &branches)? else {
        return Ok(None);
    };
    let mut facts: Vec<Basic> = assumptions.to_vec();
    for (v, (lower, upper)) in rest.iter().zip(&stack.bounds) {
        if *lower != neg_oo() {
            facts.push(gt(v, lower)?);
        }
        if *upper != oo() {
            facts.push(lt(v, upper)?);
        }
    }
    // sqrt(-1/(x**2 - 1)) sqrt(1 - x**2) = 1 on (-1, 1)
    g = split_roots(&g, &facts)?;
    if sections == [n] && rest.len() >= 2 {
        // the base cell as a region (a disc under a surface goes to polar
        // coordinates)
        let condition: Basic = pruned(&cell_condition(cell, cad, m + 1, n - 1, false)?)?;
        let found: Basic = integrate_by_ranges(
            &g,
            &condition,
            Some(&rest),
            assumptions,
            Some(integrator),
            "lebesgue",
            None,
        )?;
        if !found
            .preorder()
            .iter()
            .any(|n| matches!(n.head(), Head::IntegralByRanges))
        {
            return Ok(Some(found));
        }
    }
    iterated(&g, &rest, &stack, assumptions, integrator)
}

fn diff1(e: &Basic, x: &Basic) -> R {
    crate::core::function::derivative(e, x)
}

/// The integral with the `(n-k)`-dimensional Hausdorff measure on the
/// variety of the equations of the condition, through the decomposition;
/// `None` when the condition has no equation, a branch is not explicit or
/// an integral is not computed (`_hausdorff`).
fn hausdorff(
    f: &Basic,
    formula: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let parts: Vec<Basic> = if matches!(formula.head(), Head::And) {
        formula.args().to_vec()
    } else {
        vec![formula.clone()]
    };
    if !parts.iter().any(|p| matches!(relation_op(p), Some(RelOp::Eq))) {
        return Ok(None);
    }
    if !parts.iter().all(|p| {
        matches!(
            relation_op(p),
            Some(RelOp::Eq | RelOp::Lt | RelOp::Le | RelOp::Gt | RelOp::Ge)
        )
    }) {
        return Ok(None);
    }
    let parameters: Vec<Basic> = sorted_by_name(formula.free_symbols().into_iter().filter(|s| !names.contains(s)));
    let mut gens: Vec<Basic> = parameters.clone();
    gens.extend(names.iter().cloned());
    let Some((compiled, cad)) = decomposition(formula, &gens)? else {
        return Ok(None);
    };
    let m: usize = parameters.len();
    let holding: Vec<std::sync::Arc<CadCell>> = cad
        .cells()
        .iter()
        .filter(|c| compiled.call(&c.signs.lock().expect("signs")))
        .cloned()
        .collect();
    if holding.is_empty() {
        return Ok(Some(Basic::zero()));
    }
    let sections_of = |c: &CadCell| c.index[m..].iter().filter(|i| *i % 2 == 0).count();
    let codimension: usize = holding.iter().map(|c| sections_of(c)).min().unwrap_or(0);
    if codimension == 0 {
        return Ok(None);
    }
    let mut values: Vec<(std::sync::Arc<CadCell>, Basic)> = Vec::new();
    for cell in &holding {
        // not of dimension n - k: measure zero
        if sections_of(cell) != codimension {
            continue;
        }
        let base: std::sync::Arc<CadCell> = if m > 0 { ancestor_at(cell, m) } else { cell.clone() };
        let condition: Basic = if m > 0 {
            parameter_condition(cell, &cad, m, false)?
        } else {
            crate::core::logic::true_()
        };
        if m > 0 && ask(&condition, assumptions)? == Some(false) {
            continue;
        }
        let mut with_condition: Vec<Basic> = assumptions.to_vec();
        if m > 0 {
            with_condition.push(condition.clone());
        }
        let Some(mut value) = section_value(cell, &cad, m, f, names, &with_condition, integrator)? else {
            return Ok(None);
        };
        if m > 0 {
            if let Some(refined) = attempt(|| refine(&value, &with_condition))? {
                value = refined;
            }
        }
        match values.iter_mut().find(|(c, _)| std::sync::Arc::ptr_eq(c, &base)) {
            Some(entry) => entry.1 = add(&[entry.1.clone(), value])?,
            None => values.push((base, value)),
        }
    }
    Ok(Some(cases(&values, &cad, m, assumptions)?))
}

// -- the node

/// `IntegralByRanges(integrand, condition, Tuple(variables)[, Symbol(measure)[, dimension]])`:
/// the arguments checked and stored, nothing evaluated; the variables the
/// symbols of the condition but the `parameters`, by name, when not given
/// (the radial variable with a dimension) (`IntegralByRanges.__init__`).
pub fn integral_by_ranges(
    integrand: &Basic,
    condition: &Basic,
    variables: Option<&[Basic]>,
    measure: &str,
    dimension: Option<&Basic>,
    parameters: Option<&[Basic]>,
) -> R {
    let excluded: Vec<Basic> = parameters.map(|p| p.to_vec()).unwrap_or_default();
    for p in &excluded {
        if !p.is_symbol() {
            return Err(PolyError::Type(format!("expected a symbol, got {}", p)));
        }
    }
    let names: Vec<Basic> = match variables {
        None => {
            let mut names: Vec<Basic> =
                sorted_by_name(condition.free_symbols().into_iter().filter(|v| !excluded.contains(v)));
            if dimension.is_some() {
                names = vec![radial_symbol(&names)?];
            }
            names
        }
        Some(vs) => {
            for v in vs {
                if !v.is_symbol() {
                    return Err(PolyError::Type(format!("expected a symbol, got {}", v)));
                }
            }
            if let Some(clash) = vs.iter().find(|v| excluded.contains(v)) {
                return Err(PolyError::Value(format!(
                    "{} cannot be both an integration variable and a parameter",
                    clash
                )));
            }
            vs.to_vec()
        }
    };
    if names.is_empty() {
        return Err(PolyError::Value(
            "no integration variable: the condition has no symbols".to_string(),
        ));
    }
    if !MEASURES.contains(&measure) {
        return Err(PolyError::Value(format!(
            "measure must be one of lebesgue, hausdorff, got '{}'",
            measure
        )));
    }
    if dimension.is_some() && names.len() != 1 {
        return Err(PolyError::Value(
            "with a dimension the integral is radial, in one variable".to_string(),
        ));
    }
    let mut args: Vec<Basic> = vec![integrand.clone(), condition.clone(), Basic::new(Head::Tuple, names)];
    if measure != MEASURES[0] || dimension.is_some() {
        args.push(crate::core::symbol::Symbol::new(measure, &[]).0);
    }
    if let Some(n) = dimension {
        args.push(n.clone());
    }
    Ok(Basic::new(Head::IntegralByRanges, args))
}

/// The value of the node, or the node itself when it cannot be computed
/// (`IntegralByRanges.doit`).
pub fn doit(node: &Basic) -> R {
    let args: &[Basic] = node.args();
    let measure: String = if args.len() > 3 {
        symbol_name(&args[3])
    } else {
        MEASURES[0].to_string()
    };
    let dimension: Option<&Basic> = args.get(4);
    integrate_by_ranges(&args[0], &args[1], Some(args[2].args()), &[], None, &measure, dimension)
}

/// The free symbols: those of the integrand, the condition and the
/// dimension but the integration variables (`free_symbols`).
#[allow(clippy::mutable_key_type)]
pub fn free_symbols(node: &Basic) -> std::collections::BTreeSet<Basic> {
    let args: &[Basic] = node.args();
    let mut symbols: std::collections::BTreeSet<Basic> = args[0].free_symbols();
    symbols.extend(args[1].free_symbols());
    if let Some(n) = args.get(4) {
        symbols.extend(n.free_symbols());
    }
    for v in args[2].args() {
        symbols.remove(v);
    }
    symbols
}
