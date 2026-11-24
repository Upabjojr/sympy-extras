//! Port of `sympy_extras/solvers/ode.py`: ordinary differential equations
//! through their point symmetries (canonical coordinates, the reduction
//! of the order), the check of a solution as SymPy's `checkodesol`
//! proves it, and `solve_ode`: the classical methods with the first order
//! solvers and the symmetry method as fallbacks.

#![allow(clippy::mutable_key_type)]

use std::rc::Rc;
use std::sync::Arc;

use crate::core::basic::{Basic, Head, RelOp, UndefinedFunction};
use crate::error::{PolyError, PolyResult};
use crate::solvers::kovacic::{failure, replaced};
use crate::solvers::lie::{JetSpace, Symmetry};
use crate::solvers::linear_ode::{has_integral, normal};
use crate::solvers::second_order::ode_symmetries;

type R = PolyResult<Basic>;
type Timeout = Option<f64>;

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
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn trigsimp(e: &Basic) -> R {
    crate::simplify::trigsimp::trigsimp(e, crate::simplify::trigsimp::Method::Matching)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
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
fn is_zero(e: &Basic) -> bool {
    *e == Basic::zero()
}
fn function_of(name: &str, x: &Basic) -> R {
    Arc::new(UndefinedFunction::new(name, None, &[])).apply(vec![x.clone()])
}
fn has_head(e: &Basic, test: fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| test(n.head()))
}
fn has_derivative(e: &Basic) -> bool {
    has_head(e, |h| matches!(h, Head::Derivative))
}
fn has_order(e: &Basic) -> bool {
    has_head(e, |h| matches!(h, Head::Order))
}

/// The time limit of the settings (`_limit` of the default marker).
pub fn default_timeout() -> Timeout {
    crate::settings::settings().timeout
}

fn attempt<T>(timeout: Timeout, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout, f)
}

/// `f()` with the settings carrying the limit (`_with_limit`).
fn with_limit<T>(timeout: Timeout, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    let changes = match timeout {
        None => crate::settings::Changes {
            no_timeout: true,
            ..Default::default()
        },
        Some(t) => crate::settings::Changes {
            timeout: Some(t),
            ..Default::default()
        },
    };
    let _guard = crate::settings::configure(&changes)?;
    attempt(timeout, f)
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

fn dsolve(equation: &Basic, f: &Basic, timeout: Timeout) -> PolyResult<Option<Vec<Basic>>> {
    attempt(timeout, || crate::solvers::dsolve::dsolve(equation, f))
}

fn solve(e: &Basic, v: &Basic, timeout: Timeout) -> PolyResult<Vec<Basic>> {
    Ok(attempt(timeout, || crate::solvers::solvers::solve_equation(e, v))?
        .flatten()
        .unwrap_or_default())
}

fn integral(e: &Basic, v: &Basic, timeout: Timeout) -> PolyResult<Option<Basic>> {
    Ok(with_limit(timeout, || {
        crate::integrals::indefinite::closed_antiderivative(e, v, true, false, None)
    })?
    .flatten())
}

fn quadrature(e: &Basic, v: &Basic, timeout: Timeout) -> R {
    match integral(e, v, timeout)? {
        Some(found) => Ok(found),
        None => crate::integrals::integral::integral(e, v, None),
    }
}

fn sorted_by_name(symbols: impl IntoIterator<Item = Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().collect();
    v.sort_by_key(|s| s.to_string());
    v.dedup();
    v
}

// ---------------------------------------------------------------------------
// the symmetries and the canonical coordinates

/// Functions appearing in the equation (`_natural_basis`).
fn natural_basis(e: &Basic, x: &Basic, ysym: &Basic) -> Vec<Basic> {
    let mut atoms: Vec<Basic> = e
        .preorder()
        .into_iter()
        .filter(|n| matches!(n.head(), Head::Function(_)))
        .collect();
    atoms.sort_by_key(|a| a.to_string());
    let mut found: Vec<Basic> = Vec::new();
    for f in atoms {
        let symbols = f.free_symbols();
        if !(symbols.contains(x) || symbols.contains(ysym)) {
            continue;
        }
        if !found.contains(&f) {
            found.push(f);
        }
    }
    found.truncate(4);
    found
}

fn constants(taken: &[Basic], n: usize) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    let mut k: usize = 1;
    while result.len() < n {
        let c: Basic = symbol(&format!("C{}", k));
        if !taken.contains(&c) {
            result.push(c);
        }
        k += 1;
    }
    result
}

/// An invariant `r(x, y)` of `dy/dx = f(x, y)` (`_first_integral`).
fn first_integral(
    ode_rhs: &Basic,
    big_y: &Basic,
    x: &Basic,
    ysym: &Basic,
    timeout: Timeout,
) -> PolyResult<Option<Basic>> {
    let Some(sol) = dsolve(&eq(&diff(big_y, x, 1)?, ode_rhs), big_y, timeout)? else {
        return Ok(None);
    };
    for candidate in sol {
        let found: Vec<Basic> = candidate
            .free_symbols()
            .into_iter()
            .filter(|c| c.to_string().starts_with('C'))
            .collect();
        if found.len() != 1 || !is_eq(&candidate) {
            continue;
        }
        let values: Vec<Basic> = solve(&sub(&candidate.args()[0], &candidate.args()[1])?, &found[0], timeout)?;
        if let Some(first) = values.first() {
            return Ok(Some(simplify(&subs(first, big_y, ysym)?)?));
        }
    }
    Ok(None)
}

/// Canonical coordinates `(r, s)` of a symmetry (`canonical_coordinates`).
pub fn canonical_coordinates(symmetry: &Symmetry, timeout: Timeout) -> PolyResult<Option<(Basic, Basic)>> {
    let jets: &Rc<JetSpace> = &symmetry.jets;
    if jets.p() != 1 || jets.q() != 1 {
        return Err(PolyError::Value(
            "an ordinary differential equation with one unknown is expected".to_string(),
        ));
    }
    let (x, ysym) = (jets.x[0].clone(), jets.u[0].clone());
    let (xi, eta) = (&symmetry.xi[0], &symmetry.eta[0]);
    if is_zero(xi) && is_zero(eta) {
        return Ok(None);
    }
    let (r, s): (Option<Basic>, Option<Basic>) = if is_zero(xi) {
        (Some(x.clone()), integral(&pow(eta, &int(-1))?, &ysym, timeout)?)
    } else if is_zero(eta) {
        (Some(ysym.clone()), integral(&pow(xi, &int(-1))?, &x, timeout)?)
    } else {
        let big_y: Basic = function_of("Y", &x)?;
        let Some(r) = first_integral(&subs(&div(eta, xi)?, &ysym, &big_y)?, &big_y, &x, &ysym, timeout)? else {
            return Ok(None);
        };
        let s: Option<Basic> = transversal(&r, xi, eta, &x, &ysym, timeout)?;
        (Some(r), s)
    };
    let (Some(r), Some(s)) = (r, s) else {
        return Ok(None);
    };
    if has_integral(&s) {
        return Ok(None);
    }
    let check = attempt(timeout, || -> PolyResult<(Basic, Basic)> {
        Ok((
            simplify(&add(&[
                mul(&[xi.clone(), diff(&r, &x, 1)?])?,
                mul(&[eta.clone(), diff(&r, &ysym, 1)?])?,
            ])?)?,
            simplify(&add(&[
                mul(&[xi.clone(), diff(&s, &x, 1)?])?,
                mul(&[eta.clone(), diff(&s, &ysym, 1)?])?,
                int(-1),
            ])?)?,
        ))
    })?;
    match check {
        Some((a, b)) if is_zero(&a) && is_zero(&b) => Ok(Some((r, s))),
        _ => Ok(None),
    }
}

/// `s` with `X s = 1` (`_transversal`).
fn transversal(
    r: &Basic,
    xi: &Basic,
    eta: &Basic,
    x: &Basic,
    ysym: &Basic,
    timeout: Timeout,
) -> PolyResult<Option<Basic>> {
    let rsym: Basic = symbol("r");
    for (solved, other, coefficient) in [(ysym, x, xi), (x, ysym, eta)] {
        for value in solve(&sub(r, &rsym)?, solved, timeout)? {
            let integrand: Basic = normal(&subs(&pow(coefficient, &int(-1))?, solved, &value)?)?;
            if let Some(s) = integral(&integrand, other, timeout)? {
                return Ok(Some(subs(&s, &rsym, r)?));
            }
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// the reduction of the order

/// The reduction of an ordinary differential equation by a symmetry
/// (`ReducedODE`).
#[derive(Clone, Debug)]
pub struct ReducedOde {
    pub r: Basic,
    pub s: Basic,
    pub variable: Basic,
    pub v: Basic,
    pub ode: Basic,
}

impl ReducedOde {
    /// (`__repr__`)
    pub fn repr(&self) -> String {
        format!("ReducedODE(r={}, s={}, ode={})", self.r, self.s, self.ode)
    }
}

/// `expr` simplified, if it does not depend on `symbols` (`_free_of`).
fn free_of(expr: &Basic, symbols: &[Basic], timeout: Timeout) -> PolyResult<Option<Basic>> {
    let depends = |e: &Basic| e.free_symbols().iter().any(|s| symbols.contains(s));
    let mut e: Basic = normal(expr)?;
    if depends(&e) {
        let Some(simplified) = attempt(timeout, || simplify(&e))? else {
            return Ok(None);
        };
        e = simplified;
    }
    Ok(if depends(&e) { None } else { Some(e) })
}

/// Reduce the order with a point symmetry (`reduce_order`).
pub fn reduce_order(ode: &Basic, y: &Basic, symmetry: &Symmetry, timeout: Timeout) -> PolyResult<Option<ReducedOde>> {
    let jets: Rc<JetSpace> = JetSpace::new(std::slice::from_ref(y))?;
    let (x, ysym) = (jets.x[0].clone(), jets.u[0].clone());
    let equation: Basic = jets.to_jet(&as_zero(ode)?)?;
    let n: u32 = jets.order(&equation);
    if n == 0 {
        return Ok(None);
    }
    let Some((r, s)) = canonical_coordinates(symmetry, timeout)? else {
        return Ok(None);
    };
    let leading: Vec<(Basic, Basic)> =
        crate::solvers::lie::leading_derivatives_branch(&jets, std::slice::from_ref(&equation))?;
    let y_n: Basic = jets.jet(0, &[n]);
    let Some(omega) = leading.iter().find(|(k, _)| *k == y_n).map(|(_, v)| v.clone()) else {
        return Ok(None);
    };
    let d_r: Basic = jets.total_derivative(&r, 0)?;
    if is_zero(&d_r) {
        return Ok(None);
    }
    let mut s_k: Vec<Basic> = vec![s.clone()];
    for _ in 0..n {
        let last: Basic = s_k.last().expect("nonempty").clone();
        s_k.push(normal(&div(&jets.total_derivative(&last, 0)?, &d_r)?)?);
    }
    let rsym: Basic = symbol("r");
    let v_k: Vec<Basic> = (0..n).map(|k| symbol(&format!("v_{}", k))).collect();
    let mut substitutions: Vec<(Basic, Basic)> = Vec::new();
    let mut eliminated: Option<Basic> = None;
    for candidate in [&ysym, &x] {
        let values: Vec<Basic> = solve(&sub(&r, &rsym)?, candidate, timeout)?;
        if let Some(first) = values.first() {
            substitutions.push((candidate.clone(), first.clone()));
            eliminated = Some(candidate.clone());
            break;
        }
    }
    let Some(eliminated) = eliminated else {
        return Ok(None);
    };
    let remaining: Vec<Basic> = [x.clone(), ysym.clone()]
        .into_iter()
        .filter(|s| *s != eliminated)
        .collect();
    for k in 1..n {
        let y_k: Basic = jets.jet(0, &[k]);
        let equation: Basic = sub(&replaced(&s_k[k as usize], &substitutions)?, &v_k[(k - 1) as usize])?;
        let solved: Vec<Basic> = solve(&equation, &y_k, timeout)?;
        let Some(first) = solved.first() else {
            return Ok(None);
        };
        substitutions.push((y_k, first.clone()));
    }
    let mut top: Basic = replaced(&s_k[n as usize], &[(y_n, omega)])?;
    for _ in 0..n {
        let new: Basic = replaced(&top, &substitutions)?;
        if new == top {
            break;
        }
        top = new;
    }
    let Some(mut g) = free_of(&top, &remaining, timeout)? else {
        return Ok(None);
    };
    let v: Basic = function_of("v", &rsym)?;
    let reduced: Basic = if n == 1 {
        sub(&v, &g)?
    } else {
        for k in (1..n).rev() {
            let replacement: Basic = if k > 1 {
                crate::core::function::derivative_node(&v, &[(rsym.clone(), (k - 1) as u64)])
            } else {
                v.clone()
            };
            g = subs(&g, &v_k[(k - 1) as usize], &replacement)?;
        }
        sub(
            &crate::core::function::derivative_node(&v, &[(rsym.clone(), (n - 1) as u64)]),
            &g,
        )?
    };
    Ok(Some(ReducedOde {
        r,
        s,
        variable: rsym,
        v,
        ode: reduced,
    }))
}

// ---------------------------------------------------------------------------
// the check of a solution

/// The variables and counts of a derivative node.
fn variable_counts(d: &Basic) -> Vec<(Basic, u64)> {
    d.args()[1..]
        .iter()
        .map(|pair| {
            let count: u64 = pair.args()[1]
                .as_rational()
                .and_then(|r| u64::try_from(&crate::numbers::numer(r)).ok())
                .unwrap_or(1);
            (pair.args()[0].clone(), count)
        })
        .collect()
}

fn derivatives_in(e: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for node in e.preorder() {
        if matches!(node.head(), Head::Derivative) && !found.contains(&node) {
            found.push(node);
        }
    }
    found
}

/// `eq` with `func` replaced by `new` and its derivatives by those of
/// `new` (SymPy's `sub_func_doit`).
fn sub_func_doit(e: &Basic, func: &Basic, new: &Basic) -> R {
    let mut reps: Vec<(Basic, Basic)> = vec![(func.clone(), new.clone())];
    for d in derivatives_in(e) {
        let value: Basic = if d.args()[0] == *func {
            crate::core::function::diff(new, &variable_counts(&d))?
        } else {
            crate::core::function::doit(&replaced(&d, &[(func.clone(), new.clone())])?)?
        };
        if !reps.iter().any(|(k, _)| *k == d) {
            reps.push((d, value));
        }
    }
    replaced(e, &reps)
}

/// The solutions of `e = 0` for `target` (`_solved_for`).
fn solved_for(e: &Basic, target: &Basic, timeout: Timeout) -> PolyResult<Option<Vec<Basic>>> {
    let d: Basic = dummy("f");
    let in_d: Basic = replaced(e, &[(target.clone(), d.clone())])?;
    let found: Option<Vec<Basic>> = attempt(timeout, || crate::solvers::solvers::solve_equation(&in_d, &d))?.flatten();
    match found {
        None => Ok(None),
        Some(values) => Ok(Some(
            values.iter().map(|v| subs(v, &d, target)).collect::<PolyResult<_>>()?,
        )),
    }
}

/// The solution of `e = 0` for `target` when `e` is linear in it
/// (`_linear_solution`).
fn linear_solution(e: &Basic, target: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let d: Basic = dummy("f");
    let linear: Basic = replaced(e, &[(target.clone(), d.clone())])?;
    let a: Basic = diff(&linear, &d, 1)?;
    if is_zero(&a) || a.has(&d) {
        return Ok(None);
    }
    let b: Basic = subs(&linear, &d, &Basic::zero())?;
    Ok(Some(vec![normal(&neg(&div(&b, &a)?)?)?]))
}

/// Whether `solution` solves the equation, as SymPy's `checkodesol`
/// proves it (`check_ode_solution`).
pub fn check_ode_solution(
    ode: &Basic,
    solution: &Basic,
    f: &Basic,
    timeout: Timeout,
    solve_for_func: bool,
) -> PolyResult<Option<bool>> {
    Ok(attempt(timeout, || checkodesol(ode, solution, f, timeout, solve_for_func))?.flatten())
}

fn check_failure(e: &PolyError) -> bool {
    failure(e) || matches!(e, PolyError::Arithmetic(_))
}

/// `False` when the residual has a reliable nonzero value at a sample
/// point of `x > 0` (`_nonzero_residual`).
fn nonzero_residual(residual: &Basic, x: &Basic) -> PolyResult<Option<bool>> {
    if !crate::settings::settings().numerical_checks {
        return Ok(None);
    }
    let condition: Basic = crate::core::relational::gt(x, &Basic::zero())?;
    match crate::integrals::conditions::numerical_verdict(residual, &Basic::zero(), &[], &condition, 3) {
        Ok(Some(false)) => Ok(Some(false)),
        Ok(_) => Ok(None),
        Err(e) if check_failure(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

fn checkodesol(
    ode: &Basic,
    solution: &Basic,
    func: &Basic,
    timeout: Timeout,
    solve_for_func: bool,
) -> PolyResult<Option<bool>> {
    if func.args().len() != 1 || !func.args()[0].is_symbol() {
        return Err(PolyError::Value(format!(
            "func must be a function of one variable, not {}",
            func
        )));
    }
    let x: Basic = func.args()[0].clone();
    let (lhs_ode, rhs_ode) = if is_eq(ode) {
        (ode.args()[0].clone(), ode.args()[1].clone())
    } else {
        (ode.clone(), Basic::zero())
    };
    let (mut sol_lhs, mut sol_rhs) = if is_eq(solution) {
        (solution.args()[0].clone(), solution.args()[1].clone())
    } else {
        (func.clone(), solution.clone())
    };
    if is_eq(solution) && sol_rhs == *func {
        std::mem::swap(&mut sol_lhs, &mut sol_rhs);
    }
    let whole: Basic = sub(&lhs_ode, &rhs_ode)?;
    let order: usize = crate::solvers::ode_cases::ode_order(&whole, func);
    let solved: bool = sol_lhs == *func && !sol_rhs.has(func);
    let relation: Basic = sub(&sol_lhs, &sol_rhs)?;
    if !solved && !has_order(&relation) {
        if let Some(implicit) = derivatives_substituted(&lhs_ode, &rhs_ode, &relation, func, &x, order, timeout)? {
            if off_the_relation(&implicit, &relation, func, &x)? == Some(false) {
                return Ok(Some(false));
            }
            if zero_residual(&implicit, func)? {
                return Ok(Some(true));
            }
            if let Some(reduced) = modulo_relation(&implicit, &relation, func, &x, &whole)? {
                if zero_residual(&reduced, func)? {
                    return Ok(Some(true));
                }
            }
        }
    }
    if solve_for_func && !solved {
        if let Some(values) = solved_for(&relation, func, timeout)?.filter(|v| !v.is_empty()) {
            let mut verdicts: Vec<Option<bool>> = Vec::new();
            for value in &values {
                verdicts.push(checkodesol(ode, &eq(func, value), func, timeout, false)?);
            }
            if verdicts.contains(&Some(false)) {
                return Ok(Some(false));
            }
            if verdicts.iter().all(|v| *v == Some(true)) {
                return Ok(Some(true));
            }
        }
    }
    if has_order(&sol_lhs) || has_order(&sol_rhs) {
        return Ok(None);
    }
    let mut residual: Option<Basic> = None;
    // first pass: the solution substituted into the equation
    if sol_lhs == *func {
        let r: Basic = sub_func_doit(&whole, func, &sol_rhs)?;
        let substituted: Basic = crate::simplify::simplify::besselsimp(&r)?;
        let rewritten: Basic = crate::core::rewriting::rewrite(
            &substituted,
            &[],
            "exp",
            &crate::core::rewriting::RewriteHints::default(),
        )?;
        let simplified: Basic = simplify(&rewritten)?;
        let s: Basic = if !is_zero(&simplified) {
            crate::core::function::expand_with(
                &simplified,
                true,
                crate::core::function::Hints {
                    force: true,
                    ..crate::core::function::Hints::DEFAULT
                },
            )?
        } else {
            Basic::zero()
        };
        residual = Some(r);
        if is_zero(&s) {
            return Ok(Some(true));
        }
    }
    // second pass: the n-th derivatives equal
    let s: Basic = simplify(&add(&[
        trigsimp(&sub(
            &diff(&sol_lhs, &x, order as u64)?,
            &diff(&sol_rhs, &x, order as u64)?,
        )?)?,
        neg(&trigsimp(&lhs_ode)?)?,
        trigsimp(&rhs_ode)?,
    ])?)?;
    if is_zero(&s) {
        return Ok(Some(true));
    }
    if !solved {
        return Ok(None);
    }
    // third pass: the derivatives of the solution substituted
    if let Some(derived) = derivatives_substituted(&lhs_ode, &rhs_ode, &relation, func, &x, order, timeout)? {
        if zero_residual(&derived, func)? {
            return Ok(Some(true));
        }
    }
    match residual {
        None => Ok(None),
        Some(r) => nonzero_residual(&r, &x),
    }
}

/// The residual with the derivatives of `func` replaced by those of the
/// relation (`_derivatives_substituted`).
#[allow(clippy::too_many_arguments)]
fn derivatives_substituted(
    lhs_ode: &Basic,
    rhs_ode: &Basic,
    relation: &Basic,
    func: &Basic,
    x: &Basic,
    order: usize,
    timeout: Timeout,
) -> PolyResult<Option<Basic>> {
    let mut diffsols: Vec<Option<Basic>> = vec![None; order + 1];
    if relation.has(func) && !has_derivative(relation) {
        if let Some(explicit) = linear_solution(relation, func)? {
            if !explicit[0].has(func) {
                diffsols[0] = Some(explicit[0].clone());
            }
        }
    }
    for i in 1..=order {
        if i == 1 {
            let dr: Basic = diff(relation, x, 1)?;
            let df: Basic = diff(func, x, 1)?;
            let sdf: Option<Vec<Basic>> = match linear_solution(&dr, &df)? {
                Some(v) if !v.is_empty() => Some(v),
                _ => solved_for(&dr, &df, timeout)?,
            };
            let Some(first) = sdf.and_then(|v| v.into_iter().next()) else {
                return Ok(None);
            };
            diffsols[1] = Some(first);
        } else {
            let previous: Basic = diffsols[i - 1].clone().expect("the lower derivative");
            diffsols[i] = Some(diff(&previous, x, 1)?);
        }
    }
    let (mut lhs, mut rhs) = (lhs_ode.clone(), rhs_ode.clone());
    let lowest: usize = if diffsols[0].is_some() { 0 } else { 1 };
    for i in (lowest..=order).rev() {
        let target: Basic = if i > 0 { diff(func, x, i as u64)? } else { func.clone() };
        let value: Basic = diffsols[i].clone().expect("a value");
        lhs = sub_func_doit(&lhs, &target, &value)?;
        rhs = sub_func_doit(&rhs, &target, &value)?;
    }
    Ok(Some(sub(&lhs, &rhs)?))
}

/// Whether `simplify` proves the residual zero (`_zero_residual`).
fn zero_residual(residual: &Basic, func: &Basic) -> PolyResult<bool> {
    if is_zero(&simplify(residual)?) {
        return Ok(true);
    }
    let num: Basic = trigsimp(&crate::core::expr::as_numer_denom(residual)?.0)?;
    let d: Basic = dummy("func");
    let num: Basic = replaced(&num, &[(func.clone(), d.clone())])?;
    let (positive, reps) = crate::simplify::simplify::posify(&num)?;
    let reps: Vec<(Basic, Basic)> = reps.into_iter().collect();
    Ok(is_zero(&replaced(
        &replaced(&simplify(&positive)?, &reps)?,
        &[(d, func.clone())],
    )?))
}

/// The first candidate in which `relation` is linear, with its value
/// (`_linear_variable`).
fn linear_variable(relation: &Basic, candidates: &[Basic]) -> PolyResult<Option<(Basic, Basic)>> {
    for v in candidates {
        if !relation.has(v) {
            continue;
        }
        let a: Basic = diff(relation, v, 1)?;
        if is_zero(&a) || a.has(v) {
            continue;
        }
        return Ok(Some((
            v.clone(),
            normal(&div(&neg(&subs(relation, v, &Basic::zero())?)?, &a)?)?,
        )));
    }
    Ok(None)
}

/// The constants of the relation first, then the parameters, then `last`
/// (`_constants_first`).
fn constants_first(relation: &Basic, ode: &Basic, x: &Basic, last: &[Basic]) -> Vec<Basic> {
    let names: Vec<Basic> = sorted_by_name(
        relation
            .free_symbols()
            .into_iter()
            .filter(|s| s.is_symbol() && s != x && !last.contains(s)),
    );
    let taken = ode.free_symbols();
    let mut result: Vec<Basic> = names.iter().filter(|s| !taken.contains(*s)).cloned().collect();
    result.extend(names.iter().filter(|s| taken.contains(*s)).cloned());
    result.extend(last.iter().cloned());
    result
}

/// The residual with a constant of the relation replaced by its value
/// (`_modulo_relation`).
fn modulo_relation(
    residual: &Basic,
    relation: &Basic,
    func: &Basic,
    x: &Basic,
    ode: &Basic,
) -> PolyResult<Option<Basic>> {
    if has_derivative(relation) {
        return Ok(None);
    }
    let y: Basic = dummy("y");
    let big_f: Basic = replaced(relation, &[(func.clone(), y.clone())])?;
    let in_ode = ode.free_symbols();
    let candidates: Vec<Basic> = constants_first(&big_f, ode, x, &[])
        .into_iter()
        .filter(|s| !in_ode.contains(s))
        .collect();
    let Some((v, value)) = linear_variable(&big_f, &candidates)? else {
        return Ok(None);
    };
    if !residual.has(&v) {
        return Ok(None);
    }
    Ok(Some(replaced(
        residual,
        &[(v, replaced(&value, &[(y, func.clone())])?)],
    )?))
}

/// The points of the relation at which an implicit residual is evaluated
/// (`_RELATION_SAMPLES`).
const RELATION_SAMPLES: usize = 4;

/// `False` when the residual of an implicit solution has a reliable
/// nonzero value at a point of the relation (`_off_the_relation`).
fn off_the_relation(residual: &Basic, relation: &Basic, func: &Basic, x: &Basic) -> PolyResult<Option<bool>> {
    if !crate::settings::settings().numerical_checks || has_derivative(residual) || has_derivative(relation) {
        return Ok(None);
    }
    let y: Basic = dummy("y");
    let big_r: Basic = replaced(residual, &[(func.clone(), y.clone())])?;
    let big_f: Basic = replaced(relation, &[(func.clone(), y.clone())])?;
    if has_head(&big_r, |h| matches!(h, Head::Undefined(_))) {
        return Ok(None);
    }
    let Some((v, value)) = linear_variable(&big_f, &constants_first(&big_f, &big_r, x, &[x.clone(), y.clone()]))?
    else {
        return Ok(None);
    };
    let on_relation: Basic = replaced(&big_r, &[(v.clone(), value.clone())])?;
    let slope: Basic = replaced(&diff(&big_f, &y, 1)?, &[(v.clone(), value.clone())])?;
    let terms: Vec<Basic> = if on_relation.is_add() {
        on_relation.args().to_vec()
    } else {
        vec![on_relation.clone()]
    };
    let mut all_symbols = big_r.free_symbols();
    all_symbols.extend(big_f.free_symbols());
    let symbols: Vec<Basic> = sorted_by_name(all_symbols.into_iter().filter(|s| s.is_symbol() && *s != v));
    let mut rng = crate::random::PyRandom::from_string(&format!("({}, {})", residual, relation));
    let positive_variable: bool = v == *x || v == y;
    for _ in 0..RELATION_SAMPLES {
        let point: Vec<(Basic, Basic)> = symbols
            .iter()
            .map(|s| {
                let p: i64 = rng.randint(3, 19);
                (
                    s.clone(),
                    Basic::rational(crate::numbers::Int::from(p), crate::numbers::Int::from(4)).expect("4"),
                )
            })
            .collect();
        let value_at = |e: &Basic| crate::numeric::reliable_complex(e, 30, &point);
        let Some(v0) = value_at(&value) else {
            continue;
        };
        if v0.1 != 0.0 || (positive_variable && v0.0 <= 0.0) {
            continue;
        }
        let d = value_at(&slope);
        let r = value_at(&on_relation);
        let (Some(d), Some(r)) = (d, r) else {
            continue;
        };
        if (d.0 == 0.0 && d.1 == 0.0) || d.1 != 0.0 {
            continue;
        }
        let mut scale: f64 = 1.0;
        for t in &terms {
            match value_at(t) {
                Some(z) if z.1 == 0.0 => scale = scale.max(z.0.abs()),
                _ => {
                    // a complex value: a branch of a logarithm or of a root
                    scale = 0.0;
                    break;
                }
            }
        }
        if scale > 0.0 && r.0.hypot(r.1) > 1e-12 * scale {
            return Ok(Some(false));
        }
    }
    Ok(None)
}

fn verified(ode: &Basic, candidate: &Basic, y: &Basic, timeout: Timeout) -> PolyResult<bool> {
    Ok(check_ode_solution(ode, candidate, y, timeout, true)? == Some(true))
}

// ---------------------------------------------------------------------------
// the solvers

/// Solve the reduced equation and go back (`_solve_reduced`).
fn solve_reduced(
    ode: &Basic,
    y: &Basic,
    reduction: &ReducedOde,
    check: bool,
    timeout: Timeout,
) -> PolyResult<Vec<Basic>> {
    let jets: Rc<JetSpace> = JetSpace::new(std::slice::from_ref(y))?;
    let ysym: Basic = jets.u[0].clone();
    let (rsym, v) = (&reduction.variable, &reduction.v);
    let n: u32 = jets.order(&jets.to_jet(&as_zero(ode)?)?);
    let v_solutions: Vec<Basic> = if n == 1 {
        vec![sub(v, &reduction.ode)?]
    } else {
        let Some(sols) = dsolve(&reduction.ode, v, timeout)? else {
            return Ok(Vec::new());
        };
        sols.iter()
            .filter(|s| is_eq(s) && s.args()[0] == *v)
            .map(|s| s.args()[1].clone())
            .collect()
    };
    let mut results: Vec<Basic> = Vec::new();
    for big_v in v_solutions {
        let s_r: Basic = quadrature(&big_v, rsym, timeout)?;
        let mut taken: Vec<Basic> = s_r.free_symbols().into_iter().collect();
        taken.extend(as_zero(ode)?.free_symbols());
        let c: Basic = constants(&taken, 1).remove(0);
        let lhs: Basic = reduction.s.clone();
        let rhs: Basic = add(&[subs(&s_r, rsym, &reduction.r)?, c])?;
        let mut candidates: Vec<Basic> = Vec::new();
        for value in solve(&sub(&lhs, &rhs)?, &ysym, timeout)? {
            candidates.push(eq(y, &subs(&value, &ysym, y)?));
        }
        if candidates.is_empty() {
            candidates.push(eq(&subs(&lhs, &ysym, y)?, &subs(&rhs, &ysym, y)?));
        }
        for candidate in candidates {
            if check && !verified(ode, &candidate, y, timeout)? {
                continue;
            }
            results.push(candidate);
        }
    }
    Ok(results)
}

/// Solve an ordinary differential equation through its point symmetries
/// (`dsolve_lie`).
#[allow(clippy::too_many_arguments)]
pub fn dsolve_lie(
    ode: &Basic,
    y: &Basic,
    degree: u32,
    basis: &[Basic],
    given: Option<&[Symmetry]>,
    check: bool,
    timeout: Timeout,
    special_values: bool,
) -> PolyResult<Vec<Basic>> {
    let syms: Vec<Symmetry> = match given {
        Some(given) => given.to_vec(),
        None => {
            let mut syms: Vec<Symmetry> = ode_symmetries(ode, y, degree, basis)?;
            if syms.is_empty() && basis.is_empty() {
                let jets: Rc<JetSpace> = JetSpace::new(std::slice::from_ref(y))?;
                let natural: Vec<Basic> = natural_basis(&jets.to_jet(&as_zero(ode)?)?, &jets.x[0], &jets.u[0]);
                if !natural.is_empty() {
                    syms = ode_symmetries(ode, y, degree, &natural)?;
                }
            }
            syms
        }
    };
    let mut solutions: Vec<Basic> = Vec::new();
    for sym in &syms {
        let Some(reduction) = reduce_order(ode, y, sym, timeout)? else {
            continue;
        };
        for sol in solve_reduced(ode, y, &reduction, check, timeout)? {
            if !solutions.contains(&sol) {
                solutions.push(sol);
            }
        }
        if !solutions.is_empty() {
            break;
        }
    }
    if special_values && given.is_none() {
        let solve = |at: &Basic| dsolve_lie(at, y, degree, basis, None, check, timeout, true);
        if let Some(cased) = crate::solvers::ode_cases::solution_cases(&solutions, ode, y, &solve)? {
            return Ok(vec![cased]);
        }
    }
    Ok(solutions)
}

/// Solve an ordinary differential equation: the classical methods, the
/// first order solvers, the symmetry method (`solve_ode`).
pub fn solve_ode(
    ode: &Basic,
    y: &Basic,
    degree: u32,
    check: bool,
    timeout: Timeout,
    special_values: bool,
) -> PolyResult<Vec<Basic>> {
    let found: Vec<Basic> = solve_ode_inner(ode, y, degree, check, timeout)?;
    if special_values {
        let solve = |at: &Basic| solve_ode(at, y, degree, check, timeout, true);
        if let Some(cased) = crate::solvers::ode_cases::solution_cases(&found, ode, y, &solve)? {
            return Ok(vec![cased]);
        }
    }
    Ok(found)
}

fn solve_ode_inner(ode: &Basic, y: &Basic, degree: u32, check: bool, timeout: Timeout) -> PolyResult<Vec<Basic>> {
    let found: Vec<Basic> = dsolve(ode, y, timeout)?.unwrap_or_default();
    let mut accepted: Vec<Basic> = Vec::new();
    for candidate in found {
        if accepted.contains(&candidate) || has_head(&candidate, |h| matches!(h, Head::Order | Head::NaN)) {
            continue;
        }
        if check && !verified(ode, &candidate, y, timeout)? {
            continue;
        }
        accepted.push(candidate);
    }
    if !accepted.is_empty() {
        return Ok(accepted);
    }
    if let Some(first) = first_order(ode, y, check, timeout)? {
        return Ok(vec![first]);
    }
    dsolve_lie(ode, y, degree, &[], None, check, timeout, false)
}

/// The solution of a first order equation by the first order solvers
/// (`_first_order`).
fn first_order(ode: &Basic, y: &Basic, check: bool, timeout: Timeout) -> PolyResult<Option<Basic>> {
    let equation: Basic = as_zero(ode)?;
    let orders: Vec<u64> = derivatives_in(&equation)
        .iter()
        .filter(|d| d.args()[0] == *y)
        .map(|d| variable_counts(d).iter().map(|(_, n)| *n).max().unwrap_or(0))
        .collect();
    if orders.iter().max() != Some(&1) {
        return Ok(None);
    }
    let found: Option<Basic> = attempt(timeout, || {
        crate::solvers::first_order::dsolve_first_order(ode, y, false)
    })?
    .flatten();
    let Some(found) = found.filter(is_eq) else {
        return Ok(None);
    };
    if check && !verified(ode, &found, y, timeout)? {
        return Ok(None);
    }
    Ok(Some(found))
}
