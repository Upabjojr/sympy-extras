//! Port of `sympy_extras/solvers/first_order.py`: first order equations
//! beyond the classical hints: Riccati equations through their linear
//! second order equation, Chini's and Abel's equations in their
//! integrable classes, the Prelle–Singer procedure, and the parametric
//! solutions of d'Alembert–Lagrange equations.

#![allow(clippy::mutable_key_type)]

use std::sync::Arc;

use crate::core::basic::{Basic, Head, RelOp, UndefinedFunction};
use crate::error::{PolyError, PolyResult};
use crate::solvers::abel::{all_coeffs, degree, replace_any};
use crate::solvers::kovacic::replaced;
use crate::solvers::linear_ode::{has_integral, normal};

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;

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
fn is_rational_function(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}
fn powdenest(e: &Basic) -> R {
    crate::simplify::powsimp::powdenest(e, true, false)
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

/// `F(x, y, p) = 0` with symbols for `f` and `f'`
/// (`_derivative_polynomial`).
pub(crate) fn derivative_polynomial(equation: &Basic, f: &Basic) -> PolyResult<(Basic, Basic, Basic, Basic)> {
    let x: Basic = match f.args().first() {
        Some(x) if x.is_symbol() => x.clone(),
        _ => return Err(PolyError::Value("the function must depend on a symbol".to_string())),
    };
    let lhs: Basic = as_zero(equation)?;
    let (u, p) = (dummy("y"), dummy("p"));
    let derivative: Basic = crate::core::function::derivative_node(f, &[(x.clone(), 1)]);
    let in_p: Basic = replaced(&lhs, &[(derivative, p.clone())])?;
    let has_derivative = |e: &Basic| e.preorder().iter().any(|n| matches!(n.head(), Head::Derivative));
    // the derivatives left are looked for before f is replaced
    if has_derivative(&in_p) {
        return Err(PolyError::Value(format!("a first order equation in {} is expected", f)));
    }
    let replaced_e: Basic = replaced(&in_p, &[(f.clone(), u.clone())])?;
    if has_derivative(&replaced_e) || replaced_e.has(f) {
        return Err(PolyError::Value(format!("a first order equation in {} is expected", f)));
    }
    Ok((replaced_e, x, u, p))
}

/// An antiderivative, the unevaluated `Integral` when none is found
/// (`_integrate`).
fn integrate(e: &Basic, v: &Basic) -> R {
    match crate::integrals::indefinite::closed_antiderivative(e, v, false, false, None)? {
        Some(found) => Ok(found),
        None => crate::integrals::integral::integral(e, v, None),
    }
}

/// `p` solved from `F = 0` when `F` is linear in `p` (`_explicit_rhs`).
fn explicit_rhs(big_f: &Basic, p: &Basic) -> R2 {
    let Some(c) = all_coeffs(big_f, p, true)?.filter(|c| degree(c) == 1) else {
        return Ok(None);
    };
    Ok(Some(normal(&neg(&div(&c[1], &c[0])?)?)?))
}

/// An implicit solution of Chini's equation when its invariant is constant
/// (`chini_ode`).
pub fn chini_ode(equation: &Basic, f: &Basic) -> R2 {
    let (big_f, x, u, p) = derivative_polynomial(equation, f)?;
    let Some(rhs) = explicit_rhs(&big_f, &p)? else {
        return Ok(None);
    };
    chini(&rhs, &x, &u, f)
}

/// SymPy's `real_root(arg, n)` (`miscellaneous.real_root`).
fn real_root(a: &Basic, m: &Basic) -> R {
    use crate::core::relational::eq as eqv;
    let root = |b: &Basic| pow(b, &pow(m, &int(-1))?);
    let one: Basic = int(1);
    let first: Basic = crate::core::logic::or_(&[eqv(m, &one)?, eqv(m, &int(-1))?]);
    let second: Basic = crate::core::logic::and_(&[
        eqv(&crate::functions::elementary::complexes::im(a)?, &Basic::zero())?,
        eqv(&crate::core::modulo::modulo(m, &int(2))?, &one)?,
    ]);
    let sign: Basic = crate::functions::elementary::complexes::sign(a)?;
    let absolute: Basic = crate::functions::elementary::complexes::abs_(a)?;
    crate::functions::elementary::piecewise::piecewise(&[
        (root(a)?, first),
        (mul(&[sign, root(&absolute)?])?, second),
        (root(a)?, crate::core::logic::true_()),
    ])
}

fn chini(rhs: &Basic, x: &Basic, u: &Basic, f: &Basic) -> R2 {
    let mut terms: Vec<(Basic, Basic)> = Vec::new();
    let expanded: Basic = expand(rhs)?;
    let parts: Vec<Basic> = if expanded.is_add() {
        expanded.args().to_vec()
    } else {
        vec![expanded]
    };
    for term in parts {
        let (coefficient, part) = crate::core::expr::as_independent(&term, u, false)?;
        let exponent: Basic = if part == Basic::one() {
            Basic::zero()
        } else if part == *u {
            Basic::one()
        } else if part.is_pow() && part.args()[0] == *u && !part.args()[1].has(u) && !part.args()[1].has(x) {
            part.args()[1].clone()
        } else {
            return Ok(None);
        };
        match terms.iter_mut().find(|(e, _)| *e == exponent) {
            Some((_, c)) => *c = add(&[c.clone(), coefficient])?,
            None => terms.push((exponent, coefficient)),
        }
    }
    let exponents: Vec<Basic> = terms
        .iter()
        .map(|(e, _)| e.clone())
        .filter(|e| *e != Basic::zero() && *e != Basic::one())
        .collect();
    if exponents.len() != 1 {
        return Ok(None);
    }
    let n: Basic = exponents[0].clone();
    if !crate::core::assumptions::is_number(&n) || n == Basic::zero() {
        return Ok(None);
    }
    let get = |k: &Basic| {
        terms
            .iter()
            .find(|(e, _)| e == k)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Basic::zero)
    };
    let (fn_, g, h) = (get(&n), get(&Basic::one()), get(&Basic::zero()));
    if fn_ == Basic::zero() || h == Basic::zero() {
        return Ok(None);
    }
    // y = v exp(G), G = Integral(g): v' = fn exp((n-1)G) v**n + h exp(-G)
    let big_g: Basic = if g != Basic::zero() {
        integrate(&g, x)?
    } else {
        Basic::zero()
    };
    if has_integral(&big_g) {
        return Ok(None);
    }
    let fx: Basic = mul(&[fn_, exp(&mul(&[sub(&n, &int(1))?, big_g.clone()])?)?])?;
    let hx: Basic = mul(&[h, exp(&neg(&big_g)?)?])?;
    let ratio: Basic = normal(&div(&hx, &fx)?)?;
    let odd_integer: bool = crate::core::expr::is(&n, crate::core::assumptions::Predicate::integer) == Some(true)
        && crate::core::expr::is(&n, crate::core::assumptions::Predicate::odd) == Some(true);
    let s: Basic = if odd_integer {
        // the real n-th root (the principal root of a negative number is complex)
        powdenest(&real_root(&ratio, &n)?)?
    } else {
        powdenest(&pow(&ratio, &pow(&n, &int(-1))?)?)?
    };
    let c: Basic = simplify(&powdenest(&div(&diff(&s, x, 1)?, &hx)?)?)?;
    if c.has(x) {
        return Ok(None);
    }
    let w: Basic = dummy("w");
    let left: Basic = integrate(
        &pow(&add(&[pow(&w, &n)?, int(1), neg(&mul(&[c, w.clone()])?)?])?, &int(-1))?,
        &w,
    )?;
    let right: Basic = integrate(&normal(&div(&hx, &s)?)?, x)?;
    let c1: Basic = symbol("C1");
    let w_value: Basic = div(&mul(&[f.clone(), exp(&neg(&big_g)?)?])?, &s)?;
    Ok(Some(eq(
        &crate::core::subs::subs(&left, &w, &w_value)?,
        &add(&[right, c1])?,
    )))
}

/// Whether an implicit relation returned for an Abel equation defines
/// solutions of it (`_abel_relation_holds`).
fn abel_relation_holds(
    solution: &Basic,
    c: &crate::solvers::abel::Coefficients,
    x: &Basic,
    f: &Basic,
) -> PolyResult<bool> {
    if !is_eq(solution) {
        return Ok(true);
    }
    let relation: Basic = sub(&solution.args()[0], &solution.args()[1])?;
    crate::solvers::abel::defines_solution(&relation, c, x, f, &symbol("C1"))
}

/// An implicit solution of Abel's equations of the first and second kind
/// in the constant-invariant cases (`abel_ode`).
pub fn abel_ode(equation: &Basic, f: &Basic) -> R2 {
    use crate::solvers::abel::{abel_by_invariants, abel_coefficients, constant_invariant};
    let (big_f, x, u, p) = derivative_polynomial(equation, f)?;
    let Some(in_p) = all_coeffs(&big_f, &p, true)?.filter(|c| degree(c) == 1) else {
        return Ok(None);
    };
    let (a1, a0) = (&in_p[0], &in_p[1]);
    // a1(x, y) y' + a0(x, y) = 0
    if !a1.has(&u) {
        let rhs: Basic = normal(&neg(&div(a0, a1)?)?)?;
        let first_kind: Basic = eq(&diff(f, &x, 1)?, &replaced(&rhs, &[(u.clone(), f.clone())])?);
        let coefficients = match abel_coefficients(&first_kind, f) {
            Ok((c, _)) => c,
            Err(PolyError::Value(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        if constant_invariant(&coefficients, &x)? != Some(false) {
            let solution: Option<Basic> = abel_first_kind(&rhs, &x, &u, f, f)?;
            // the relation can lose x altogether and imply y' = 0
            if let Some(solution) = solution {
                if abel_relation_holds(&solution, &coefficients, &x, f)? {
                    return Ok(Some(solution));
                }
            }
        }
        // a non-constant invariant: the integrable classes recognised
        // through the invariants
        return abel_by_invariants(equation, f);
    }
    // second kind: a1 linear in y, a0 quadratic in y
    let p1: Option<Vec<Basic>> = all_coeffs(a1, &u, true)?;
    let p0: Option<Vec<Basic>> = all_coeffs(&neg(a0)?, &u, true)?;
    let (Some(p1), Some(p0)) = (p1, p0) else {
        return Ok(None);
    };
    if degree(&p1) != 1 || degree(&p0) > 2 {
        return Ok(None);
    }
    let lead: &Basic = &p1[0];
    let g: Basic = normal(&div(&p1[1], lead)?)?;
    let rhs: Basic = normal(&neg(&div(a0, lead)?)?)?;
    // y + g = 1/w: w' = -(y + g)' w**2 = -(rhs/(y + g) - g') w**2
    let w: Basic = dummy("w");
    let v: Basic = sub(&pow(&w, &int(-1))?, &g)?;
    let w_rhs: Basic = normal(&neg(&mul(&[
        sub(
            &mul(&[replaced(&rhs, &[(u.clone(), v)])?, w.clone()])?,
            &diff(&g, &x, 1)?,
        )?,
        pow(&w, &int(2))?,
    ])?)?)?;
    let big_w: Basic = function_of("W", &x)?;
    let w_equation: Basic = eq(&diff(&big_w, &x, 1)?, &replaced(&w_rhs, &[(w.clone(), big_w.clone())])?);
    let coefficients = match abel_coefficients(&w_equation, &big_w) {
        Ok((c, _)) => c,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let value: Basic = pow(&add(&[f.clone(), g.clone()])?, &int(-1))?;
    if constant_invariant(&coefficients, &x)? != Some(false) {
        if let Some(solution) = abel_first_kind(&w_rhs, &x, &w, &value, f)? {
            let back: Basic = replace_any(&solution, &[(f.clone(), sub(&pow(&big_w, &int(-1))?, &g)?)])?;
            if abel_relation_holds(&back, &coefficients, &x, &big_w)? {
                return Ok(Some(solution));
            }
        }
    }
    let Some(first_kind) = abel_by_invariants(&w_equation, &big_w)?.filter(is_eq) else {
        return Ok(None);
    };
    let back: Vec<(Basic, Basic)> = vec![(big_w, value)];
    Ok(Some(eq(
        &crate::core::subs::subs_mapping(&first_kind.args()[0], &back)?,
        &crate::core::subs::subs_mapping(&first_kind.args()[1], &back)?,
    )))
}

/// `u' = rhs(x, u)` cubic in `u`, `u` standing for `value`
/// (`_abel_first_kind`).
fn abel_first_kind(rhs: &Basic, x: &Basic, u: &Basic, value: &Basic, _f: &Basic) -> R2 {
    let Some(c) = all_coeffs(&expand(rhs)?, u, true)?.filter(|c| degree(c) == 3) else {
        return Ok(None);
    };
    let (f3, f2) = (&c[0], &c[1]);
    if *f2 == Basic::zero() {
        return chini_for(rhs, x, u, value);
    }
    let shift: Basic = normal(&div(f2, &mul(&[int(3), f3.clone()])?)?)?;
    let v: Basic = dummy("v");
    // u = v - shift: v' = rhs(x, v - shift) + shift'
    let new_rhs: Basic = expand(&add(&[
        replaced(rhs, &[(u.clone(), sub(&v, &shift)?)])?,
        diff(&shift, x, 1)?,
    ])?)?;
    chini_for(&new_rhs, x, &v, &add(&[value.clone(), shift])?)
}

fn chini_for(rhs: &Basic, x: &Basic, u: &Basic, value: &Basic) -> R2 {
    let placeholder: Basic = function_of("Y", x)?;
    let Some(solution) = chini(rhs, x, u, &placeholder)? else {
        return Ok(None);
    };
    Ok(Some(replace_any(&solution, &[(placeholder, value.clone())])?))
}

/// The parametric solution `[Eq(x, X(p)), Eq(y, Y(p))]` of a
/// d'Alembert–Lagrange equation (`lagrange_ode`).
pub fn lagrange_ode(equation: &Basic, f: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let (big_f, x, u, p) = derivative_polynomial(equation, f)?;
    let Some(in_u) = all_coeffs(&big_f, &u, true)?.filter(|c| degree(c) == 1) else {
        return Ok(None);
    };
    let rhs: Basic = normal(&neg(&div(&in_u[1], &in_u[0])?)?)?; // y = rhs(x, p)
    let Some(in_x) = all_coeffs(&rhs, &x, true)?.filter(|c| degree(c) == 1) else {
        return Ok(None);
    };
    let (fp, gp) = (&in_x[0], &in_x[1]);
    if fp.has(&x) || gp.has(&x) {
        return Ok(None);
    }
    if normal(&sub(fp, &p)?)? == Basic::zero() {
        return Ok(None); // Clairaut: SymPy's hint
    }
    let parameter: Basic = symbol("p");
    let (fp, gp) = (
        replaced(fp, &[(p.clone(), parameter.clone())])?,
        replaced(gp, &[(p, parameter.clone())])?,
    );
    let big_x: Basic = function_of("X", &parameter)?;
    let denominator: Basic = sub(&parameter, &fp)?;
    let linear: Basic = eq(
        &sub(
            &diff(&big_x, &parameter, 1)?,
            &div(&mul(&[big_x.clone(), diff(&fp, &parameter, 1)?])?, &denominator)?,
        )?,
        &div(&diff(&gp, &parameter, 1)?, &denominator)?,
    );
    let found: Option<Vec<Basic>> = attempt(|| crate::solvers::dsolve::dsolve(&linear, &big_x))?;
    let Some(solved) = found.filter(|v| v.len() == 1).map(|mut v| v.remove(0)) else {
        return Ok(None);
    };
    if solved.args()[0] != big_x {
        return Ok(None);
    }
    let x_value: Basic = solved.args()[1].clone();
    if has_integral(&x_value) {
        return Ok(None);
    }
    Ok(Some(vec![
        eq(&x, &x_value),
        eq(f, &simplify(&add(&[mul(&[x_value, fp])?, gp])?)?),
    ]))
}

/// The general solution of a Riccati equation with rational coefficients
/// through its linear second order equation (`riccati_ode`).
pub fn riccati_ode(equation: &Basic, f: &Basic, special_values: bool) -> R2 {
    let found: Option<Basic> = riccati(equation, f)?;
    if special_values {
        if let Some(found) = &found {
            let solve = |at: &Basic| -> PolyResult<Vec<Basic>> { Ok(riccati_ode(at, f, true)?.into_iter().collect()) };
            if let Some(cased) =
                crate::solvers::ode_cases::solution_cases(std::slice::from_ref(found), equation, f, &solve)?
            {
                return Ok(Some(cased));
            }
        }
    }
    Ok(found)
}

fn riccati(equation: &Basic, f: &Basic) -> R2 {
    let (big_f, x, u, p) = derivative_polynomial(equation, f)?;
    let Some(rhs) = explicit_rhs(&big_f, &p)? else {
        return Ok(None);
    };
    let Some(c) = all_coeffs(&expand(&rhs)?, &u, true)?.filter(|c| degree(c) == 2) else {
        return Ok(None);
    };
    let (a, b, cc) = (&c[0], &c[1], &c[2]);
    if ![a, b, cc].iter().all(|e| is_rational_function(e, &x)) {
        return Ok(None);
    }
    let w: Basic = function_of("u", &x)?;
    let linear: Basic = add(&[
        diff(&w, &x, 2)?,
        neg(&mul(&[
            add(&[div(&diff(a, &x, 1)?, a)?, b.clone()])?,
            diff(&w, &x, 1)?,
        ])?)?,
        mul(&[a.clone(), cc.clone(), w.clone()])?,
    ])?;
    // the special functions are recognised at once; Kovacic's algorithm is
    // slow to fail on equations without Liouvillian solutions
    let mut solutions: Option<Vec<Basic>> = attempt(|| crate::solvers::special::special_solutions(&linear, &w))?
        .flatten()
        .filter(|v| !v.is_empty());
    if solutions.is_none() {
        solutions = attempt(|| crate::solvers::linear_ode::dsolve_linear(&linear, &w, true, true, false))?
            .filter(|v| !v.is_empty());
    }
    let Some(solutions) = solutions else {
        return Ok(None);
    };
    let c1: Basic = symbol("C1");
    let (numerator, denominator) = if solutions.len() >= 2 {
        let (u1, u2) = (&solutions[0], &solutions[1]);
        (
            add(&[diff(u1, &x, 1)?, mul(&[c1.clone(), diff(u2, &x, 1)?])?])?,
            mul(&[a.clone(), add(&[u1.clone(), mul(&[c1, u2.clone()])?])?])?,
        )
    } else {
        // a particular solution only
        let u1: &Basic = &solutions[0];
        (diff(u1, &x, 1)?, mul(&[a.clone(), u1.clone()])?)
    };
    let mut value: Basic = div(&neg(&numerator)?, &denominator)?;
    if is_rational_function(&value, &x) {
        value = normal(&value)?;
    }
    Ok(Some(eq(f, &value)))
}

/// The Prelle–Singer procedure (`darboux.prelle_singer`).
fn prelle_singer(equation: &Basic, f: &Basic) -> R2 {
    crate::solvers::darboux::prelle_singer(equation, f, 4, None)
}

type Method = fn(&Basic, &Basic) -> R2;

/// The Riccati, Chini/Abel, Prelle–Singer or Lagrange solution of a first
/// order equation, else `None` (`dsolve_first_order`).
pub fn dsolve_first_order(equation: &Basic, f: &Basic, special_values: bool) -> R2 {
    let methods: [Method; 4] = [riccati, abel_ode, chini_ode, prelle_singer];
    for method in methods {
        if let Some(solution) = attempt(|| method(equation, f))?.flatten() {
            if special_values && is_eq(&solution) {
                let solve = |at: &Basic| -> PolyResult<Vec<Basic>> {
                    Ok(dsolve_first_order(at, f, true)?.into_iter().collect())
                };
                if let Some(cased) =
                    crate::solvers::ode_cases::solution_cases(std::slice::from_ref(&solution), equation, f, &solve)?
                {
                    return Ok(Some(cased));
                }
            }
            return Ok(Some(solution));
        }
    }
    if let Some(parametric) = attempt(|| lagrange_ode(equation, f))?
        .flatten()
        .filter(|v| !v.is_empty())
    {
        return Ok(Some(Basic::new(Head::Tuple, parametric)));
    }
    Ok(None)
}
