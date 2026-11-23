//! Port of `sympy_extras/solvers/second_order.py`: second order
//! equations `y'' = Phi(x, y, y')` through integrating factors, Lie's
//! linearisation test and transformation, and the rectification of two
//! commuting point symmetries.

#![allow(clippy::mutable_key_type)]

use std::rc::Rc;
use std::sync::Arc;

use crate::core::basic::{Basic, Head, RelOp, UndefinedFunction};
use crate::error::{PolyError, PolyResult};
use crate::solvers::kovacic::{failure, replaced};
use crate::solvers::lie::{JetSpace, Symmetry};
use crate::solvers::linear_ode::normal;

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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn diff2(e: &Basic, x: &Basic, y: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1), (y.clone(), 1)])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
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
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn function_of(name: &str, x: &Basic) -> R {
    Arc::new(UndefinedFunction::new(name, None, &[])).apply(vec![x.clone()])
}
fn is_zero(e: &Basic) -> bool {
    *e == Basic::zero()
}

/// The coefficients of `e` as a polynomial in `v` from the leading one
/// down (`[0]` for zero), `None` when `e` is no polynomial in `v`
/// (`_all_coeffs`).
fn all_coeffs(e: &Basic, v: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let terms = match crate::solvers::kovacic::coefficients(e, v) {
        Ok(t) => t,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(err) => return Err(err),
    };
    let Some(&top) = terms.keys().next_back() else {
        return Ok(Some(vec![Basic::zero()]));
    };
    Ok(Some(
        (0..=top)
            .rev()
            .map(|k| terms.get(&k).cloned().unwrap_or_else(Basic::zero))
            .collect(),
    ))
}

/// `coefficients` padded with leading zeros to `n` entries.
fn padded(mut coefficients: Vec<Basic>, n: usize) -> Vec<Basic> {
    while coefficients.len() < n {
        coefficients.insert(0, Basic::zero());
    }
    coefficients
}

/// A first integral `R(x, y, y') = C1` and its integrating factor
/// (`FirstIntegral`).
#[derive(Clone, Debug)]
pub struct FirstIntegral {
    pub factor: Basic,
    pub integral: Basic,
    pub x: Basic,
    pub y: Basic,
    pub p: Basic,
}

impl FirstIntegral {
    /// `R(x, f, f') = C1` (`equation`).
    pub fn equation(&self, f: &Basic) -> R {
        let x: Basic = match f.args().first() {
            Some(x) if x.is_symbol() => x.clone(),
            _ => return Err(PolyError::Value("the function must depend on a symbol".to_string())),
        };
        let derivative: Basic = crate::core::function::derivative_node(f, &[(x.clone(), 1)]);
        let big_r: Basic = replaced(
            &replaced(&self.integral, &[(self.p.clone(), derivative)])?,
            &[(self.y.clone(), f.clone()), (self.x.clone(), x)],
        )?;
        Ok(eq(&big_r, &symbol("C1")))
    }

    /// `FirstIntegral(factor, integral)` (`__repr__`).
    pub fn repr(&self) -> String {
        format!("FirstIntegral({}, {})", self.factor, self.integral)
    }
}

/// A point transformation `t = phi(x, y)`, `u = psi(x, y)` to
/// `u'' = rhs(u')` (`Linearization`).
#[derive(Clone, Debug)]
pub struct Linearization {
    pub t: Basic,
    pub u: Basic,
    pub x: Basic,
    pub y: Basic,
    pub v: Basic,
    pub rhs: Basic,
}

impl Linearization {
    /// (`__repr__`)
    pub fn repr(&self) -> String {
        if is_zero(&self.rhs) {
            format!("Linearization(t={}, u={})", self.t, self.u)
        } else {
            format!("Linearization(t={}, u={}, rhs={})", self.t, self.u, self.rhs)
        }
    }
}

fn as_zero(equation: &Basic) -> R {
    if is_eq(equation) {
        return sub(&equation.args()[0], &equation.args()[1]);
    }
    if crate::core::logic::is_boolean(equation) {
        return Err(PolyError::Type(format!("{} is not an equation", equation)));
    }
    Ok(equation.clone())
}

fn has_derivative(e: &Basic) -> bool {
    e.preorder().iter().any(|n| matches!(n.head(), Head::Derivative))
}

/// `(Phi, x, y, p)` with the equation as `y'' = Phi(x, y, p)`
/// (`second_order_rhs`).
pub fn second_order_rhs(equation: &Basic, f: &Basic) -> PolyResult<(Basic, Basic, Basic, Basic)> {
    let x: Basic = match f.args().first() {
        Some(x) if x.is_symbol() => x.clone(),
        _ => return Err(PolyError::Value("the function must depend on a symbol".to_string())),
    };
    let lhs: Basic = as_zero(equation)?;
    let (y, p, q) = (dummy("y"), dummy("p"), dummy("q"));
    let second: Basic = crate::core::function::derivative_node(f, &[(x.clone(), 2)]);
    let first: Basic = crate::core::function::derivative_node(f, &[(x.clone(), 1)]);
    let r: Basic = replaced(
        &replaced(&replaced(&lhs, &[(second, q.clone())])?, &[(first, p.clone())])?,
        &[(f.clone(), y.clone())],
    )?;
    if has_derivative(&r) || r.has(f) {
        return Err(PolyError::Value(format!(
            "{} is not a second order equation in {}",
            equation, f
        )));
    }
    let Some(c) = all_coeffs(&r, &q)?.filter(|c| c.len() == 2) else {
        return Err(PolyError::Value(format!(
            "{} is not linear in the second derivative",
            equation
        )));
    };
    Ok((normal(&neg(&div(&c[1], &c[0])?)?)?, x, y, p))
}

fn integrate(e: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    crate::integrals::indefinite::closed_antiderivative(e, v, true, false, None)
}

fn dsolve(equation: &Basic, f: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    Ok(attempt(|| crate::solvers::dsolve::dsolve(equation, f))?.filter(|v| !v.is_empty()))
}

fn solve(e: &Basic, v: &Basic) -> PolyResult<Vec<Basic>> {
    Ok(attempt(|| crate::solvers::solvers::solve_equation(e, v))?
        .flatten()
        .unwrap_or_default())
}

fn constants_of(e: &Basic) -> Vec<Basic> {
    let mut constants: Vec<Basic> = e
        .free_symbols()
        .into_iter()
        .filter(|s| s.to_string().starts_with('C'))
        .collect();
    constants.sort_by_key(|s| s.to_string());
    constants
}

/// A nonzero particular solution from a general one (`_particular`).
fn particular(solution: &Basic) -> PolyResult<Option<Basic>> {
    if !is_eq(solution) {
        return Ok(None);
    }
    let rhs: Basic = solution.args()[1].clone();
    let constants: Vec<Basic> = constants_of(&rhs);
    if constants.is_empty() {
        return Ok(Some(rhs));
    }
    for chosen in &constants {
        let values: Vec<(Basic, Basic)> = constants
            .iter()
            .map(|c| (c.clone(), if c == chosen { Basic::one() } else { Basic::zero() }))
            .collect();
        let candidate: Basic = simplify(&replaced(&rhs, &values)?)?;
        if !is_zero(&candidate) {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// A nonzero solution of `w'' + c1 w' + c0 w = 0` (`_solve_linear`).
fn solve_linear(c1: &Basic, c0: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let w: Basic = function_of("W", x)?;
    let equation: Basic = add(&[
        diff(&w, x, 2)?,
        mul(&[c1.clone(), diff(&w, x, 1)?])?,
        mul(&[c0.clone(), w.clone()])?,
    ])?;
    let Some(solution) = dsolve(&equation, &w)?.filter(|s| s.len() == 1) else {
        return Ok(None);
    };
    particular(&solution[0])
}

/// An integrating factor `mu(x, y)` and its first integral
/// (`integrating_factor_xy`).
pub fn integrating_factor_xy(phi: &Basic, x: &Basic, y: &Basic, p: &Basic) -> PolyResult<Option<FirstIntegral>> {
    let Some(c) = all_coeffs(&expand(phi)?, p)?.filter(|c| c.len() <= 3) else {
        return Ok(None);
    };
    let c: Vec<Basic> = padded(c, 3);
    let (a, b, cc) = (&c[0], &c[1], &c[2]);
    let Some(big_a) = integrate(a, y)? else {
        return Ok(None);
    };
    let e: Basic = exp(&neg(&big_a)?)?;
    let ex: Basic = div(&diff(&e, x, 1)?, &e)?;
    let exx: Basic = div(&diff(&e, x, 2)?, &e)?;
    let c1: Basic = normal(&add(&[b.clone(), mul(&[int(2), ex.clone()])?])?)?;
    let c0: Basic = normal(&add(&[
        diff(b, x, 1)?,
        mul(&[b.clone(), ex])?,
        exx,
        neg(&diff(cc, y, 1)?)?,
        mul(&[a.clone(), cc.clone()])?,
    ])?)?;
    if c1.has(y) || c0.has(y) {
        return Ok(None);
    }
    let Some(m) = solve_linear(&c1, &c0, x)? else {
        return Ok(None);
    };
    let mu: Basic = simplify(&mul(&[m, e])?)?;
    let Some(s1) = integrate(&neg(&mul(&[cc.clone(), mu.clone()])?)?, x)? else {
        return Ok(None);
    };
    let remainder: Basic = simplify(&add(&[
        neg(&mul(&[b.clone(), mu.clone()])?)?,
        neg(&diff(&mu, x, 1)?)?,
        neg(&diff(&s1, y, 1)?)?,
    ])?)?;
    if remainder.has(x) {
        return Ok(None);
    }
    let Some(s2) = integrate(&remainder, y)? else {
        return Ok(None);
    };
    let integral: Basic = simplify(&add(&[mul(&[mu.clone(), p.clone()])?, s1, s2])?)?;
    Ok(Some(FirstIntegral {
        factor: mu,
        integral,
        x: x.clone(),
        y: y.clone(),
        p: p.clone(),
    }))
}

fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// `Phi = g(p) L(x, y, p)` (`_factor_in_p`).
fn factor_in_p(phi: &Basic, x: &Basic, y: &Basic, p: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let only_p = |e: &Basic| e.has(p) && !(e.has(x) || e.has(y));
    let mut g_parts: Vec<Basic> = Vec::new();
    let mut rest_parts: Vec<Basic> = Vec::new();
    for f in mul_args(&crate::core::exprtools::factor_terms(phi, false, true, false, true)?) {
        if only_p(&f) {
            g_parts.push(f);
        } else {
            rest_parts.push(f);
        }
    }
    let rest: Basic = mul(&rest_parts)?;
    let (numerator, denominator) =
        crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&rest, false, true)?)?;
    let mut polynomial_parts: Vec<Basic> = Vec::new();
    for (part, sign) in [(numerator, 1i64), (denominator, -1)] {
        let (constant, factors) = match crate::polys::polytools::factor_list(&part, &[x.clone(), y.clone(), p.clone()])
        {
            Ok(found) => found,
            Err(e) if failure(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        polynomial_parts.push(pow(&constant, &int(sign))?);
        for (base, exponent) in factors {
            let power: Basic = pow(&base, &mul(&[int(sign), exponent])?)?;
            if only_p(&base) {
                g_parts.push(power);
            } else {
                polynomial_parts.push(power);
            }
        }
    }
    Ok(Some((mul(&g_parts)?, mul(&polynomial_parts)?)))
}

/// An integrating factor `mu(y')` and its first integral
/// (`integrating_factor_p`).
pub fn integrating_factor_p(phi: &Basic, x: &Basic, y: &Basic, p: &Basic) -> PolyResult<Option<FirstIntegral>> {
    let Some((g, big_l)) = factor_in_p(phi, x, y, p)? else {
        return Ok(None);
    };
    let Some(c) = all_coeffs(&expand(&big_l)?, p)?.filter(|c| c.len() <= 2) else {
        return Ok(None);
    };
    let (big_b, big_a) = if c.len() == 2 {
        (c[0].clone(), c[1].clone())
    } else {
        (Basic::zero(), c[0].clone())
    };
    if !is_zero(&simplify(&sub(&diff(&big_a, y, 1)?, &diff(&big_b, x, 1)?)?)?) {
        return Ok(None);
    }
    let mu: Basic = pow(&g, &int(-1))?;
    let big_m: Option<Basic> = integrate(&mu, p)?;
    let s1: Option<Basic> = integrate(&neg(&big_a)?, x)?;
    let (Some(big_m), Some(s1)) = (big_m, s1) else {
        return Ok(None);
    };
    let remainder: Basic = simplify(&sub(&neg(&big_b)?, &diff(&s1, y, 1)?)?)?;
    if remainder.has(x) {
        return Ok(None);
    }
    let Some(s2) = integrate(&remainder, y)? else {
        return Ok(None);
    };
    Ok(Some(FirstIntegral {
        factor: mu,
        integral: add(&[big_m, s1, s2])?,
        x: x.clone(),
        y: y.clone(),
        p: p.clone(),
    }))
}

type Cubic = (Basic, Basic, Basic, Basic);

/// `(a, b, c, d)` of `y'' + a y'^3 + b y'^2 + c y' + d = 0`
/// (`_cubic_coefficients`).
fn cubic_coefficients(phi: &Basic, p: &Basic) -> PolyResult<Option<Cubic>> {
    let Some(c) = all_coeffs(&expand(&neg(phi)?)?, p)?.filter(|c| c.len() <= 4) else {
        return Ok(None);
    };
    let c: Vec<Basic> = padded(c, 4);
    Ok(Some((c[0].clone(), c[1].clone(), c[2].clone(), c[3].clone())))
}

fn lie_conditions(k: &Cubic, x: &Basic, y: &Basic) -> PolyResult<(Basic, Basic)> {
    let (a, b, c, d) = k;
    let d1 = |e: &Basic, v: &Basic| diff(e, v, 1);
    let first: Basic = add(&[
        mul(&[int(3), diff(a, x, 2)?])?,
        mul(&[int(-2), diff2(b, x, y)?])?,
        diff(c, y, 2)?,
        mul(&[int(-3), d1(a, x)?, c.clone()])?,
        mul(&[int(3), d1(a, y)?, d.clone()])?,
        mul(&[int(2), d1(b, x)?, b.clone()])?,
        mul(&[int(-3), d1(c, x)?, a.clone()])?,
        neg(&mul(&[d1(c, y)?, b.clone()])?)?,
        mul(&[int(6), d1(d, y)?, a.clone()])?,
    ])?;
    let second: Basic = add(&[
        diff(b, x, 2)?,
        mul(&[int(-2), diff2(c, x, y)?])?,
        mul(&[int(3), diff(d, y, 2)?])?,
        mul(&[int(-6), d1(a, x)?, d.clone()])?,
        mul(&[d1(b, x)?, c.clone()])?,
        mul(&[int(3), d1(b, y)?, d.clone()])?,
        mul(&[int(-2), d1(c, y)?, c.clone()])?,
        mul(&[int(-3), d1(d, x)?, a.clone()])?,
        mul(&[int(3), d1(d, y)?, b.clone()])?,
    ])?;
    Ok((simplify(&first)?, simplify(&second)?))
}

/// Lie's test (`is_linearizable`).
pub fn is_linearizable(phi: &Basic, x: &Basic, y: &Basic, p: &Basic) -> PolyResult<bool> {
    let Some(k) = cubic_coefficients(phi, p)? else {
        return Ok(false);
    };
    let (first, second) = lie_conditions(&k, x, y)?;
    Ok(is_zero(&first) && is_zero(&second))
}

/// `(phi(x), psi(x, y))` taking `y'' + b y'^2 + c y' + d = 0` to `u'' = 0`
/// (`_fibre_preserving`).
fn fibre_preserving(k: &Cubic, x: &Basic, y: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let (a, b, c, d) = k;
    if !is_zero(a) || !is_zero(&simplify(&sub(&diff(c, y, 1)?, &mul(&[int(2), diff(b, x, 1)?])?)?)?) {
        return Ok(None);
    }
    let Some(big_b) = integrate(b, y)? else {
        return Ok(None);
    };
    let bx: Basic = diff(&big_b, x, 1)?;
    let q: Basic = simplify(&add(&[
        diff(&big_b, x, 2)?,
        neg(&pow(&bx, &int(2))?)?,
        mul(&[c.clone(), bx.clone()])?,
        neg(&diff(d, y, 1)?)?,
        neg(&mul(&[b.clone(), d.clone()])?)?,
    ])?)?;
    let coefficient1: Basic = simplify(&sub(c, &mul(&[int(2), bx.clone()])?)?)?;
    if q.has(y) || coefficient1.has(y) {
        return Ok(None);
    }
    let Some(w) = solve_linear(&coefficient1, &neg(&q)?, x)? else {
        return Ok(None);
    };
    let kk: Basic = pow(&w, &int(-1))?;
    let r: Basic = simplify(&add(&[
        mul(&[int(2), bx])?,
        neg(c)?,
        neg(&mul(&[int(2), div(&diff(&w, x, 1)?, &w)?])?)?,
    ])?)?;
    if r.has(y) {
        return Ok(None);
    }
    let eb: Basic = exp(&big_b)?;
    let Some(psi_y) = integrate(&eb, y)? else {
        return Ok(None);
    };
    let (k1, k2) = (diff(&kk, x, 1)?, diff(&kk, x, 2)?);
    let (p1, p2) = (diff(&psi_y, x, 1)?, diff(&psi_y, x, 2)?);
    let inner: Basic = sub(
        &add(&[
            mul(&[k2, psi_y.clone()])?,
            mul(&[int(2), k1.clone(), p1.clone()])?,
            mul(&[kk.clone(), p2])?,
        ])?,
        &mul(&[r.clone(), add(&[mul(&[k1, psi_y.clone()])?, mul(&[kk.clone(), p1])?])?])?,
    )?;
    let remainder: Basic = simplify(&sub(&mul(&[d.clone(), kk.clone(), eb])?, &inner)?)?;
    if remainder.has(y) {
        return Ok(None);
    }
    let Some(big_r) = integrate(&r, x)? else {
        return Ok(None);
    };
    let phi_prime: Basic = exp(&big_r)?;
    let Some(phi) = integrate(&phi_prime, x)? else {
        return Ok(None);
    };
    let mut l_prime: Basic = Basic::zero();
    if !is_zero(&remainder) {
        let Some(inner) = integrate(&div(&remainder, &phi_prime)?, x)? else {
            return Ok(None);
        };
        l_prime = mul(&[phi_prime, inner])?;
    }
    let l: Basic = if !is_zero(&l_prime) {
        match integrate(&l_prime, x)? {
            Some(l) => l,
            None => return Ok(None),
        }
    } else {
        Basic::zero()
    };
    let psi: Basic = simplify(&add(&[mul(&[kk, psi_y])?, l])?)?;
    Ok(Some((phi, psi)))
}

/// `y'' = Phi` in the variables `t(x, y)`, `u(x, y)` as `u'' = rhs(v)`
/// (`transformed_rhs`).
pub fn transformed_rhs(
    phi: &Basic,
    t: &Basic,
    u: &Basic,
    x: &Basic,
    y: &Basic,
    p: &Basic,
    v: &Basic,
) -> PolyResult<Option<Basic>> {
    let total = |e: &Basic| -> R { add(&[diff(e, x, 1)?, mul(&[diff(e, y, 1)?, p.clone()])?]) };
    let dt: Basic = total(t)?;
    let du: Basic = total(u)?;
    let big_v: Basic = normal(&div(&du, &dt)?)?;
    let dv: Basic = add(&[total(&big_v)?, mul(&[diff(&big_v, p, 1)?, phi.clone()])?])?;
    let second: Basic = normal(&div(&dv, &dt)?)?;
    let solutions: Vec<Basic> = attempt(|| solve(&sub(&big_v, v)?, p))?.unwrap_or_default();
    for value in solutions {
        let candidate: Option<Basic> = attempt(|| simplify(&replaced(&second, &[(p.clone(), value.clone())])?))?;
        if let Some(candidate) = candidate {
            if !candidate.has(x) && !candidate.has(y) {
                return Ok(Some(candidate));
            }
        }
    }
    Ok(None)
}

/// A point transformation taking `y'' = Phi` to `u'' = 0` (`linearize`).
pub fn linearize(phi: &Basic, x: &Basic, y: &Basic, p: &Basic) -> PolyResult<Option<Linearization>> {
    let Some(k) = cubic_coefficients(phi, p)? else {
        return Ok(None);
    };
    let (first, second) = lie_conditions(&k, x, y)?;
    if !is_zero(&first) || !is_zero(&second) {
        return Ok(None);
    }
    let v: Basic = dummy("v");
    let made = |(t, u): (Basic, Basic)| Linearization {
        t,
        u,
        x: x.clone(),
        y: y.clone(),
        v: v.clone(),
        rhs: Basic::zero(),
    };
    if let Some(found) = fibre_preserving(&k, x, y)? {
        return Ok(Some(made(found)));
    }
    // x as a function of y: y' = 1/x', y'' = -x''/x'^3
    let q: Basic = dummy("q");
    let swapped: Basic = normal(&mul(&[
        neg(&replaced(phi, &[(p.clone(), pow(&q, &int(-1))?)])?)?,
        pow(&q, &int(3))?,
    ])?)?;
    let Some(swapped_k) = cubic_coefficients(&swapped, &q)? else {
        return Ok(None);
    };
    Ok(fibre_preserving(&swapped_k, y, x)?.map(made))
}

// ---------------------------------------------------------------------------
// commuting symmetries

fn sym_add(a: &Symmetry, b: &Symmetry) -> PolyResult<Symmetry> {
    let xi: Vec<Basic> =
        a.xi.iter()
            .zip(&b.xi)
            .map(|(p, q)| add(&[p.clone(), q.clone()]))
            .collect::<PolyResult<_>>()?;
    let eta: Vec<Basic> = a
        .eta
        .iter()
        .zip(&b.eta)
        .map(|(p, q)| add(&[p.clone(), q.clone()]))
        .collect::<PolyResult<_>>()?;
    Symmetry::new(&a.jets, xi, eta)
}

fn sym_scale(a: &Symmetry, c: &Basic) -> PolyResult<Symmetry> {
    let xi: Vec<Basic> =
        a.xi.iter()
            .map(|p| mul(&[c.clone(), p.clone()]))
            .collect::<PolyResult<_>>()?;
    let eta: Vec<Basic> = a
        .eta
        .iter()
        .map(|p| mul(&[c.clone(), p.clone()]))
        .collect::<PolyResult<_>>()?;
    Symmetry::new(&a.jets, xi, eta)
}

fn combination(symmetries: &[Symmetry], coefficients: &[Basic]) -> PolyResult<Symmetry> {
    let mut result: Symmetry = sym_scale(&symmetries[0], &coefficients[0])?;
    for k in 1..symmetries.len() {
        result = sym_add(&result, &sym_scale(&symmetries[k], &coefficients[k])?)?;
    }
    Ok(result)
}

/// The values of the unknowns of a linear system (`_linear_solution`).
fn linear_solution(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    if let Some(unique) = crate::solvers::linsolve::solve_linear_unique(equations, unknowns)? {
        return Ok(Some(unique));
    }
    let solution = attempt(|| crate::solvers::solvers::solve_system(equations, unknowns))?.flatten();
    let Some(first) = solution.and_then(|s| s.into_iter().next()) else {
        return Ok(None);
    };
    Ok(Some(
        unknowns
            .iter()
            .map(|c| {
                first
                    .iter()
                    .find(|(k, _)| k == c)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(Basic::zero)
            })
            .collect(),
    ))
}

/// The coefficients of an expression linear in the unknowns
/// (`_coefficients_in`).
fn coefficients_in(e: &Basic, x: &Basic, y: &Basic) -> PolyResult<Vec<Basic>> {
    if let Some(terms) = crate::solvers::lie::polynomial_terms(e, &[x.clone(), y.clone()])? {
        if terms.is_empty() {
            return Ok(vec![Basic::zero()]);
        }
        return Ok(terms.into_iter().map(|(_, c)| c).collect());
    }
    let mut collected: Vec<(Basic, Basic)> = Vec::new();
    let parts: Vec<Basic> = if e.is_add() { e.args().to_vec() } else { vec![e.clone()] };
    for term in parts {
        let depends = |e: &Basic| e.has(x) || e.has(y);
        let (constant, function) = if term.is_mul() {
            let (inside, outside): (Vec<Basic>, Vec<Basic>) = term.args().iter().cloned().partition(|a| depends(a));
            (mul(&outside)?, mul(&inside)?)
        } else if depends(&term) {
            (Basic::one(), term.clone())
        } else {
            (term.clone(), Basic::one())
        };
        match collected.iter_mut().find(|(f, _)| *f == function) {
            Some((_, c)) => *c = add(&[c.clone(), constant])?,
            None => collected.push((function, constant)),
        }
    }
    Ok(collected.into_iter().map(|(_, c)| c).collect())
}

fn span_equations(combination: &Symmetry, target: &Symmetry, x: &Basic, y: &Basic) -> PolyResult<Vec<Basic>> {
    let mut equations: Vec<Basic> = Vec::new();
    for (lhs, rhs) in [
        (&combination.xi[0], &target.xi[0]),
        (&combination.eta[0], &target.eta[0]),
    ] {
        equations.extend(coefficients_in(&expand(&sub(lhs, rhs)?)?, x, y)?);
    }
    Ok(equations)
}

type Structure = Vec<Vec<Vec<Basic>>>;

/// The structure constants (`_structure`).
fn structure(symmetries: &[Symmetry], x: &Basic, y: &Basic) -> PolyResult<Option<Structure>> {
    let m: usize = symmetries.len();
    let coefficients: Vec<Basic> = (0..m).map(|k| symbol(&format!("c{}", k))).collect();
    let combined: Symmetry = combination(symmetries, &coefficients)?;
    let mut result: Structure = Vec::new();
    for i in 0..m {
        let mut row: Vec<Vec<Basic>> = Vec::new();
        for j in 0..m {
            let bracket: Symmetry = symmetries[i].commutator(&symmetries[j])?;
            let equations: Vec<Basic> = span_equations(&combined, &bracket, x, y)?;
            let Some(values) = linear_solution(&equations, &coefficients)? else {
                return Ok(None);
            };
            row.push(values);
        }
        result.push(row);
    }
    Ok(Some(result))
}

fn coordinates(target: &Symmetry, symmetries: &[Symmetry], x: &Basic, y: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let coefficients: Vec<Basic> = (0..symmetries.len()).map(|k| symbol(&format!("c{}", k))).collect();
    let combined: Symmetry = combination(symmetries, &coefficients)?;
    linear_solution(&span_equations(&combined, target, x, y)?, &coefficients)
}

/// Two commuting, pointwise independent symmetries (`commuting_pair`).
#[allow(clippy::needless_range_loop)]
pub fn commuting_pair(symmetries: &[Symmetry], x: &Basic, y: &Basic) -> PolyResult<Option<(Symmetry, Symmetry)>> {
    let m: usize = symmetries.len();
    if m < 2 {
        return Ok(None);
    }
    let independent = |a: &Symmetry, b: &Symmetry| -> PolyResult<bool> {
        Ok(!is_zero(&simplify(&sub(
            &mul(&[a.xi[0].clone(), b.eta[0].clone()])?,
            &mul(&[b.xi[0].clone(), a.eta[0].clone()])?,
        )?)?))
    };
    let commute = |a: &Symmetry, b: &Symmetry| -> PolyResult<bool> {
        let bracket: Symmetry = a.commutator(b)?;
        for c in bracket.xi.iter().chain(&bracket.eta) {
            if !is_zero(&simplify(c)?) {
                return Ok(false);
            }
        }
        Ok(true)
    };
    let mut candidates: Vec<Symmetry> = symmetries.to_vec();
    for i in 0..m {
        for j in i + 1..m {
            candidates.push(sym_add(&symmetries[i], &symmetries[j])?);
        }
    }
    for (i, a) in candidates.iter().enumerate() {
        for b in &candidates[i + 1..] {
            if commute(a, b)? && independent(a, b)? {
                return Ok(Some((a.clone(), b.clone())));
            }
        }
    }
    let Some(c) = structure(symmetries, x, y)? else {
        return Ok(None);
    };
    let mut generic: Vec<Symmetry> = Vec::new();
    for k in 0..3 {
        let mut s: Symmetry = symmetries[0].clone();
        for (i, symmetry) in symmetries.iter().enumerate().skip(1) {
            s = sym_add(&s, &sym_scale(symmetry, &int((i + k + 1) as i64))?)?;
        }
        generic.push(s);
    }
    for a_sym in &generic {
        let Some(a) = coordinates(a_sym, symmetries, x, y)? else {
            continue;
        };
        let mut rows: Vec<Vec<Basic>> = Vec::new();
        for k in 0..m {
            let mut row: Vec<Basic> = Vec::new();
            for j in 0..m {
                let terms: Vec<Basic> = (0..m)
                    .map(|i| mul(&[a[i].clone(), c[i][j][k].clone()]))
                    .collect::<PolyResult<_>>()?;
                row.push(add(&terms)?);
            }
            rows.push(row);
        }
        for vector in crate::solvers::linear_ode::nullspace(&rows, m)? {
            let mut b: Symmetry = sym_scale(&symmetries[0], &vector[0])?;
            for j in 1..m {
                b = sym_add(&b, &sym_scale(&symmetries[j], &vector[j])?)?;
            }
            if !b.is_zero() && independent(a_sym, &b)? {
                return Ok(Some((a_sym.clone(), b)));
            }
        }
    }
    Ok(None)
}

/// `F` with `F_x = fx`, `F_y = fy` (`_potential`).
fn potential(fx: &Basic, fy: &Basic, x: &Basic, y: &Basic) -> PolyResult<Option<Basic>> {
    if !is_zero(&simplify(&sub(&diff(fx, y, 1)?, &diff(fy, x, 1)?)?)?) {
        return Ok(None);
    }
    let Some(f1) = integrate(fx, x)? else {
        return Ok(None);
    };
    let remainder: Basic = simplify(&sub(fy, &diff(&f1, y, 1)?)?)?;
    if remainder.has(x) {
        return Ok(None);
    }
    let Some(f2) = integrate(&remainder, y)? else {
        return Ok(None);
    };
    Ok(Some(simplify(&add(&[f1, f2])?)?))
}

/// The representation of a symmetry (`Symmetry.__repr__`).
pub fn symmetry_repr(s: &Symmetry) -> String {
    let tuple = |values: &[Basic]| -> String {
        if values.len() == 1 {
            format!("({},)", values[0])
        } else {
            format!(
                "({})",
                values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
            )
        }
    };
    format!("Symmetry(xi={}, eta={})", tuple(&s.xi), tuple(&s.eta))
}

/// The point symmetries of an ordinary differential equation, sorted by
/// the number of operations (`ode.ode_symmetries`).
pub fn ode_symmetries(ode: &Basic, y: &Basic, degree: u32, basis: &[Basic]) -> PolyResult<Vec<Symmetry>> {
    let found: Vec<Symmetry> =
        crate::solvers::lie::symmetries(std::slice::from_ref(ode), std::slice::from_ref(y), degree, basis, true)?;
    let mut keyed: Vec<((i64, String), Symmetry)> = Vec::new();
    for s in found {
        let all: Vec<Basic> = s.xi.iter().chain(&s.eta).cloned().collect();
        let ops: i64 = crate::core::function::count_ops(&Basic::new(Head::Tuple, all))?;
        keyed.push(((ops, symmetry_repr(&s)), s));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, s)| s).collect())
}

/// Coordinates rectifying two commuting point symmetries
/// (`rectify_symmetries`).
pub fn rectify_symmetries(
    phi: &Basic,
    x: &Basic,
    y: &Basic,
    p: &Basic,
    f: &Basic,
    degree: u32,
) -> PolyResult<Option<Linearization>> {
    let equation: Basic = sub(
        &crate::core::function::derivative_node(f, &[(x.clone(), 2)]),
        &replaced(
            &replaced(
                phi,
                &[(p.clone(), crate::core::function::derivative_node(f, &[(x.clone(), 1)]))],
            )?,
            &[(y.clone(), f.clone())],
        )?,
    )?;
    let Some(found) = attempt(|| ode_symmetries(&equation, f, degree, &[]))?.filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    let jets: Rc<JetSpace> = found[0].jets.clone();
    let ysym: Basic = jets.u[0].clone();
    let Some((x1, x2)) = commuting_pair(&found, x, &ysym)? else {
        return Ok(None);
    };
    let (xi1, eta1, xi2, eta2) = (&x1.xi[0], &x1.eta[0], &x2.xi[0], &x2.eta[0]);
    let delta: Basic = simplify(&sub(
        &mul(&[xi1.clone(), eta2.clone()])?,
        &mul(&[xi2.clone(), eta1.clone()])?,
    )?)?;
    let t: Option<Basic> = potential(
        &normal(&div(eta2, &delta)?)?,
        &normal(&neg(&div(xi2, &delta)?)?)?,
        x,
        &ysym,
    )?;
    let u: Option<Basic> = potential(
        &normal(&neg(&div(eta1, &delta)?)?)?,
        &normal(&div(xi1, &delta)?)?,
        x,
        &ysym,
    )?;
    let (Some(t), Some(u)) = (t, u) else {
        return Ok(None);
    };
    let (t, u) = (
        replaced(&t, &[(ysym.clone(), y.clone())])?,
        replaced(&u, &[(ysym, y.clone())])?,
    );
    let v: Basic = dummy("v");
    let Some(rhs) = transformed_rhs(phi, &t, &u, x, y, p, &v)? else {
        return Ok(None);
    };
    Ok(Some(Linearization {
        t,
        u,
        x: x.clone(),
        y: y.clone(),
        v,
        rhs,
    }))
}

/// The solutions of `u'' = rhs(u')` pulled back
/// (`_solutions_of_linearization`).
fn solutions_of_linearization(l: &Linearization, f: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let t: Basic = dummy("t");
    let u: Basic = function_of("U", &t)?;
    let general: Vec<Basic> = if is_zero(&l.rhs) {
        vec![add(&[mul(&[symbol("C1"), t.clone()])?, symbol("C2")])?]
    } else {
        let equation: Basic = eq(
            &diff(&u, &t, 2)?,
            &replaced(&l.rhs, &[(l.v.clone(), diff(&u, &t, 1)?)])?,
        );
        let Some(solved) = dsolve(&equation, &u)? else {
            return Ok(None);
        };
        let general: Vec<Basic> = solved
            .iter()
            .filter(|s| is_eq(s) && s.args()[0] == u)
            .map(|s| s.args()[1].clone())
            .collect();
        if general.is_empty() {
            return Ok(None);
        }
        general
    };
    let mut results: Vec<Basic> = Vec::new();
    for expression in general {
        let relation: Basic = sub(&l.u, &replaced(&expression, &[(t.clone(), l.t.clone())])?)?;
        let explicit: Vec<Basic> = attempt(|| solve(&relation, &l.y))?.unwrap_or_default();
        if !explicit.is_empty() {
            for value in explicit {
                results.push(eq(f, &replaced(&value, &[(l.x.clone(), x.clone())])?));
            }
        } else {
            results.push(eq(
                &replaced(&relation, &[(l.x.clone(), x.clone()), (l.y.clone(), f.clone())])?,
                &Basic::zero(),
            ));
        }
    }
    Ok(if results.is_empty() { None } else { Some(results) })
}

/// Whether `f` occurs in the integrand of an `Integral` over its own
/// argument (`_unknown_under_integral`).
fn unknown_under_integral(candidate: &Basic, f: &Basic) -> bool {
    let x: &Basic = &f.args()[0];
    candidate
        .preorder()
        .iter()
        .any(|node| matches!(node.head(), Head::Integral) && node.args()[0].has(f) && node.args()[1] == *x)
}

/// Solutions of a second order equation through an integrating factor,
/// its linearisation or two commuting symmetries (`dsolve_second_order`).
pub fn dsolve_second_order(
    equation: &Basic,
    f: &Basic,
    degree: u32,
    special_values: bool,
) -> PolyResult<Option<Vec<Basic>>> {
    let found: Option<Vec<Basic>> = dsolve_second_order_inner(equation, f, degree)?;
    if special_values {
        if let Some(found) = found.as_ref().filter(|v| !v.is_empty()) {
            let solve = |at: &Basic| -> PolyResult<Vec<Basic>> {
                Ok(dsolve_second_order(at, f, degree, true)?.unwrap_or_default())
            };
            if let Some(cased) = crate::solvers::ode_cases::solution_cases(found, equation, f, &solve)? {
                return Ok(Some(vec![cased]));
            }
        }
    }
    Ok(found)
}

type Method = fn(&Basic, &Basic, &Basic, &Basic) -> PolyResult<Option<FirstIntegral>>;

fn dsolve_second_order_inner(equation: &Basic, f: &Basic, degree: u32) -> PolyResult<Option<Vec<Basic>>> {
    let (phi, x, y, p) = second_order_rhs(equation, f)?;
    let methods: [Method; 2] = [integrating_factor_xy, integrating_factor_p];
    for method in methods {
        let Some(first_integral) = attempt(|| method(&phi, &x, &y, &p))?.flatten() else {
            continue;
        };
        let reduced: Basic = first_integral.equation(f)?;
        let Some(found) = dsolve(&reduced, f)? else {
            return Ok(Some(vec![reduced]));
        };
        if found.iter().any(|s| unknown_under_integral(s, f)) {
            return Ok(Some(vec![reduced]));
        }
        return Ok(Some(found));
    }
    let mut l: Option<Linearization> = attempt(|| linearize(&phi, &x, &y, &p))?.flatten();
    if l.is_none() {
        l = attempt(|| rectify_symmetries(&phi, &x, &y, &p, f, degree))?.flatten();
    }
    let Some(l) = l else {
        return Ok(None);
    };
    solutions_of_linearization(&l, f, &x)
}
