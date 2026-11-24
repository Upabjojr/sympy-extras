//! Port of `sympy_extras/solvers/charpit.py`: complete integrals of first
//! order partial differential equations `F(x, y, u, p, q) = 0` through
//! the standard first integrals of Charpit's method (Clairaut's form,
//! `F(p, q) = 0`, `F(u, p, q) = 0`, the separable form, `p = a` or `q = a`
//! as a first integral).

#![allow(clippy::mutable_key_type)]

use crate::core::basic::{Basic, Head, RelOp};
use crate::error::{PolyError, PolyResult};
use crate::numbers::Rat;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
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
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1)])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn eq(a: &Basic, b: &Basic) -> Basic {
    crate::core::relational::rel(RelOp::Eq, a, b)
}
fn is_eq(e: &Basic) -> bool {
    matches!(e.head(), Head::Relational(RelOp::Eq))
}
fn has_any(e: &Basic, symbols: &[&Basic]) -> bool {
    symbols.iter().any(|s| e.has(s))
}
fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn factors_of(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}

/// `F` with `u`, `p`, `q` replaced by symbols (`_Equation`).
pub struct Equation {
    pub f: Basic,
    pub x: Basic,
    pub y: Basic,
    pub u: Basic,
    pub p: Basic,
    pub q: Basic,
}

/// `lhs - rhs` of an equation, or the expression itself (`_difference`).
fn difference(equation: &Basic) -> R {
    if is_eq(equation) {
        return sub(&equation.args()[0], &equation.args()[1]);
    }
    if crate::core::logic::is_boolean(equation) || matches!(equation.head(), Head::Tuple) {
        return Err(PolyError::Type(format!(
            "an equation or an expression is expected, got {}",
            equation
        )));
    }
    Ok(equation.clone())
}

/// `e` with the keys of `rule` replaced, evaluated (`_replaced`).
pub(crate) fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let result: Basic = crate::core::function::xreplace_evaluated(e, &rule.iter().cloned().collect())?;
    if crate::core::logic::is_boolean(&result) || matches!(result.head(), Head::Tuple) {
        return Err(PolyError::Type(format!("expected an expression, got {}", result)));
    }
    Ok(result)
}

/// `F` with `u`, `p`, `q` as symbols (`_as_symbols`).
fn as_symbols(equation: &Basic, f: &Basic) -> PolyResult<Equation> {
    if f.args().len() != 2 || !f.args().iter().all(|a| a.is_symbol()) {
        return Err(PolyError::Value(format!(
            "a function of two symbols is expected, got {}",
            f
        )));
    }
    let (x, y) = (f.args()[0].clone(), f.args()[1].clone());
    let lhs: Basic = difference(equation)?;
    let (u, p, q) = (dummy("u"), dummy("p"), dummy("q"));
    let derivatives: Vec<(Basic, Basic)> = vec![(diff(f, &x)?, p.clone()), (diff(f, &y)?, q.clone())];
    let first: Basic = replaced(&lhs, &derivatives)?;
    let has_derivative = |e: &Basic| e.preorder().iter().any(|n| matches!(n.head(), Head::Derivative));
    if has_derivative(&first) {
        return Err(PolyError::Value(format!(
            "only first order derivatives of {} are allowed",
            f
        )));
    }
    let second: Basic = replaced(&first, &[(f.clone(), u.clone())])?;
    if second.has(f) {
        return Err(PolyError::Value(format!(
            "only first order derivatives of {} are allowed",
            f
        )));
    }
    Ok(Equation {
        f: second,
        x,
        y,
        u,
        p,
        q,
    })
}

/// Whether `solution` satisfies the equation (`check_complete_integral`).
pub fn check_complete_integral(equation: &Basic, f: &Basic, solution: &Basic) -> PolyResult<bool> {
    let expression: Basic = if is_eq(solution) {
        solution.args()[1].clone()
    } else {
        solution.clone()
    };
    if crate::core::logic::is_boolean(&expression) || matches!(expression.head(), Head::Tuple) {
        return Err(PolyError::Type(format!(
            "an equation or an expression is expected, got {}",
            solution
        )));
    }
    satisfies(&as_symbols(equation, f)?, &expression)
}

fn satisfies(e: &Equation, expression: &Basic) -> PolyResult<bool> {
    let value: Basic = replaced(
        &replaced(
            &e.f,
            &[
                (e.p.clone(), diff(expression, &e.x)?),
                (e.q.clone(), diff(expression, &e.y)?),
            ],
        )?,
        &[(e.u.clone(), expression.clone())],
    )?;
    Ok(simplify(&value)? == Basic::zero())
}

/// A complete integral through the standard first integrals
/// (`_first_integral`).
fn first_integral(e: &Equation, a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    let (f, x, y, u, p, q) = (&e.f, &e.x, &e.y, &e.u, &e.p, &e.q);
    let mut candidates: Vec<Basic> = Vec::new();
    // Clairaut: u = p x + q y + g(p, q)
    if is_clairaut(f, e)? {
        candidates.push(add(&[
            mul(&[a.clone(), x.clone()])?,
            mul(&[b.clone(), y.clone()])?,
            clairaut_term(f, e, a, b)?,
        ])?);
    }
    // F(p, q) = 0
    if !has_any(f, &[x, y, u]) {
        for q_value in solutions(&replaced(f, &[(p.clone(), a.clone())])?, q)? {
            candidates.push(add(&[
                mul(&[a.clone(), x.clone()])?,
                mul(&[q_value, y.clone()])?,
                b.clone(),
            ])?);
        }
    }
    // F(u, p, q) = 0: q = a p
    if !has_any(f, &[x, y]) && f.has(u) {
        for p_value in solutions(&replaced(f, &[(q.clone(), mul(&[a.clone(), p.clone()])?)])?, p)? {
            if p_value == Basic::zero() {
                continue;
            }
            let Some(integral) = antiderivative(&pow(&p_value, &int(-1))?, u)? else {
                continue;
            };
            // integral(u) = x + a y + b: solved for u when possible
            let relation: Basic = add(&[
                integral,
                crate::core::mul::neg(x)?,
                crate::core::mul::neg(&mul(&[a.clone(), y.clone()])?)?,
                crate::core::mul::neg(b)?,
            ])?;
            candidates.extend(solutions(&relation, u)?);
        }
    }
    // separable F = A(x, p) - B(y, q)
    if let Some((big_a, big_b)) = separable(f, e)? {
        for p_value in solutions(&sub(&big_a, a)?, p)? {
            for q_value in solutions(&sub(&big_b, a)?, q)? {
                if has_any(&p_value, &[y, q, u]) || has_any(&q_value, &[x, p, u]) {
                    continue;
                }
                let first = antiderivative(&p_value, x)?;
                let second = antiderivative(&q_value, y)?;
                let (Some(first), Some(second)) = (first, second) else {
                    continue;
                };
                candidates.push(add(&[first, second, b.clone()])?);
            }
        }
    }
    // p = a or q = a as a first integral
    for (variable, other) in [(p, q), (q, p)] {
        let along: &Basic = if variable == p { x } else { y };
        let derivative: Basic = add(&[diff(f, along)?, mul(&[variable.clone(), diff(f, u)?])?])?;
        if simplify(&derivative)? != Basic::zero() {
            continue;
        }
        for other_value in solutions(&replaced(f, &[(variable.clone(), a.clone())])?, other)? {
            let (p_value, q_value) = if variable == p {
                (a.clone(), other_value)
            } else {
                (other_value, a.clone())
            };
            if let Some(solution) = integrate_exact(&p_value, &q_value, e, b)? {
                candidates.push(solution);
            }
        }
    }
    for candidate in candidates {
        if satisfies(e, &candidate)? {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// `u` with `du = p dx + q dy` (`_integrate_exact`).
fn integrate_exact(p_value: &Basic, q_value: &Basic, e: &Equation, b: &Basic) -> PolyResult<Option<Basic>> {
    let (x, y, u) = (&e.x, &e.y, &e.u);
    if p_value.has(u) || q_value.has(u) {
        return Ok(None);
    }
    if simplify(&sub(&diff(p_value, y)?, &diff(q_value, x)?)?)? != Basic::zero() {
        return Ok(None);
    }
    let Some(first) = antiderivative(p_value, x)? else {
        return Ok(None);
    };
    let rest: Basic = simplify(&sub(q_value, &diff(&first, y)?)?)?;
    let Some(second) = antiderivative(&rest, y)? else {
        return Ok(None);
    };
    Ok(Some(add(&[first, second, b.clone()])?))
}

/// An antiderivative of `e` in `x`, term by term for a sum of powers,
/// else the native one (`_antiderivative`).
fn antiderivative(e: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    if let Some(native) = power_antiderivative(e, x)? {
        return Ok(Some(native));
    }
    crate::integrals::indefinite::closed_antiderivative(e, x, false, false, None)
}

/// `e` integrated term by term when it is a sum of terms `c*x**n`
/// (`_power_antiderivative`).
pub(crate) fn power_antiderivative(e: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    if !e.has(x) {
        return Ok(Some(mul(&[e.clone(), x.clone()])?));
    }
    let (constant, e) = crate::core::expr::as_independent(e, x, false)?;
    let expanded: Basic = expand(&e)?;
    let mut pieces: Vec<Basic> = Vec::new();
    for term in terms_of(&expanded) {
        let (c, dependent) = crate::core::expr::as_independent(&term, x, false)?;
        let mut n: Rat = Rat::from(0);
        for factor in factors_of(&dependent) {
            if factor == *x {
                n += Rat::from(1);
            } else if factor.is_pow() && factor.args()[0] == *x && factor.args()[1].as_rational().is_some() {
                n += factor.args()[1].as_rational().expect("a rational exponent").clone();
            } else if factor != Basic::one() {
                return Ok(None);
            }
        }
        if n == Rat::from(-1) {
            pieces.push(mul(&[c, crate::functions::elementary::exponential::log(x)?])?);
        } else {
            let m: Basic = Basic::from_rational(n + Rat::from(1));
            pieces.push(mul(&[c, pow(x, &m)?, pow(&m, &int(-1))?])?);
        }
    }
    Ok(Some(mul(&[constant, add(&pieces)?])?))
}

/// The solutions of `expression = 0` for `variable` (`_solutions`).
fn solutions(expression: &Basic, variable: &Basic) -> PolyResult<Vec<Basic>> {
    if let Some(native) =
        crate::timeout::attempt_within(timeout(), || polynomial_solutions(expression, variable))?.flatten()
    {
        return Ok(native);
    }
    Ok(crate::timeout::attempt_within(timeout(), || {
        crate::solvers::solvers::solve_equation(expression, variable)
    })?
    .flatten()
    .unwrap_or_default())
}

/// The roots of the numerator which do not cancel the denominator, when
/// the numerator is a polynomial with all its roots found
/// (`_polynomial_solutions`).
fn polynomial_solutions(expression: &Basic, variable: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let (numerator, denominator) =
        crate::simplify::radsimp::fraction(&crate::polys::rationaltools::together(expression, false, true)?)?;
    if !numerator.has(variable) {
        return Ok(None);
    }
    let polynomial = match crate::polys::polytools::Poly::new(&numerator, std::slice::from_ref(variable)) {
        Ok(p) => p,
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let degree: i64 = polynomial.degree(0)?;
    if degree < 1 || polynomial.all_coeffs()?.iter().any(|c| c.has(variable)) {
        return Ok(None);
    }
    let found = crate::polys::polyroots::roots(&numerator, std::slice::from_ref(variable), true, true, None)?;
    if found.values().sum::<usize>() as i64 != degree {
        return Ok(None);
    }
    let count: usize = found.len();
    let mut values: Vec<Basic> = Vec::new();
    for (root, _) in found.iter() {
        if denominator.has(variable)
            && simplify(&replaced(&denominator, &[(variable.clone(), root.clone())])?)? == Basic::zero()
        {
            continue;
        }
        // as SymPy's solve: more than two roots are not simplified
        let value: Basic = if count <= 2 { simplify(root)? } else { root.clone() };
        if !values.contains(&value) {
            values.push(value);
        }
    }
    values.sort_by_key(crate::core::sorting::sort_key);
    Ok(Some(values))
}

fn is_clairaut(f: &Basic, e: &Equation) -> PolyResult<bool> {
    let (x, y, u, p, q) = (&e.x, &e.y, &e.u, &e.p, &e.q);
    let linear: Basic = expand(f)?;
    let coefficient: Basic = diff(&linear, u)?;
    let free = coefficient.free_symbols();
    if [x, y, u, p, q].iter().any(|s| free.contains(*s)) {
        return Ok(false);
    }
    if simplify(&add(&[diff(&linear, x)?, mul(&[coefficient.clone(), p.clone()])?])?)? != Basic::zero() {
        return Ok(false);
    }
    if simplify(&add(&[diff(&linear, y)?, mul(&[coefficient.clone(), q.clone()])?])?)? != Basic::zero() {
        return Ok(false);
    }
    Ok(coefficient != Basic::zero())
}

/// `f(a, b)` with `F = c*(u - p x - q y - f(p, q))` (`_clairaut_term`).
fn clairaut_term(f: &Basic, e: &Equation, a: &Basic, b: &Basic) -> R {
    let (x, y, u, p, q) = (&e.x, &e.y, &e.u, &e.p, &e.q);
    let c: Basic = diff(&expand(f)?, u)?;
    let rest: Basic = expand(&add(&[
        div(f, &c)?,
        crate::core::mul::neg(u)?,
        mul(&[p.clone(), x.clone()])?,
        mul(&[q.clone(), y.clone()])?,
    ])?)?;
    crate::core::mul::neg(&replaced(&rest, &[(p.clone(), a.clone()), (q.clone(), b.clone())])?)
}

/// `(A(x, p), B(y, q))` with `F = A - B` (`_separable`).
fn separable(f: &Basic, e: &Equation) -> PolyResult<Option<(Basic, Basic)>> {
    let (x, y, u, p, q) = (&e.x, &e.y, &e.u, &e.p, &e.q);
    if f.has(u) {
        return Ok(None);
    }
    let mut a_terms: Vec<Basic> = Vec::new();
    let mut b_terms: Vec<Basic> = Vec::new();
    for term in terms_of(&expand(f)?) {
        if !has_any(&term, &[y, q]) {
            a_terms.push(term);
        } else if !has_any(&term, &[x, p]) {
            b_terms.push(crate::core::mul::neg(&term)?);
        } else {
            return Ok(None);
        }
    }
    if a_terms.is_empty() || b_terms.is_empty() {
        return Ok(None);
    }
    Ok(Some((add(&a_terms)?, add(&b_terms)?)))
}

/// A complete integral `Eq(f, phi(x, y, a, b))` (`complete_integral`).
pub fn complete_integral(
    equation: &Basic,
    f: &Basic,
    constants: Option<(&Basic, &Basic)>,
) -> PolyResult<Option<Basic>> {
    let e: Equation = as_symbols(equation, f)?;
    let default_a: Basic = crate::core::symbol::Symbol::new("a", &[]).0;
    let default_b: Basic = crate::core::symbol::Symbol::new("b", &[]).0;
    let (a, b) = constants.unwrap_or((&default_a, &default_b));
    Ok(first_integral(&e, a, b)?.map(|solution| eq(f, &solution)))
}
