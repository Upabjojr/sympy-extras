//! Port of `sympy_extras/solvers/pde.py`: point symmetries of partial
//! differential equations, similarity reductions in two independent
//! variables, and the group invariant solutions.

#![allow(clippy::mutable_key_type)]

use std::rc::Rc;
use std::sync::Arc;

use crate::core::basic::{Basic, Head, RelOp, UndefinedFunction};
use crate::error::{PolyError, PolyResult};
use crate::solvers::lie::{Jet, JetSpace, Symmetry};

type R = PolyResult<Basic>;
type Timeout = Option<f64>;

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
    crate::core::mul::div(a, b)
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1)])
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}
fn eq(a: &Basic, b: &Basic) -> Basic {
    crate::core::relational::rel(RelOp::Eq, a, b)
}
fn is_eq(e: &Basic) -> bool {
    matches!(e.head(), Head::Relational(RelOp::Eq))
}
fn undefined(name: &str) -> Arc<UndefinedFunction> {
    Arc::new(UndefinedFunction::new(name, None, &[]))
}
fn attempt<T>(timeout: Timeout, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout, f)
}
fn not_implemented(message: String) -> PolyError {
    PolyError::NotImplemented(message)
}
/// A genuine `NotImplementedError` (not a part which is not ported yet).
fn is_genuine_not_implemented(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if !m.starts_with(crate::error::NOT_PORTED_PREFIX))
}
fn as_zero(equation: &Basic) -> R {
    if is_eq(equation) {
        return sub(&equation.args()[0], &equation.args()[1]);
    }
    Ok(equation.clone())
}
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    crate::core::function::xreplace_evaluated(e, &rule.iter().cloned().collect())
}

/// The time limit of the settings (`_limit` of the default marker).
pub fn default_timeout() -> Timeout {
    crate::settings::settings().timeout
}

/// The point symmetries of a partial differential equation, sorted by
/// the number of operations (`pde_symmetries`).
pub fn pde_symmetries(
    pde: &Basic,
    u: &Basic,
    degree: u32,
    basis: &[Basic],
    superposition: bool,
) -> PolyResult<Vec<Symmetry>> {
    let found: Vec<Symmetry> = crate::solvers::lie::symmetries(
        std::slice::from_ref(pde),
        std::slice::from_ref(u),
        degree,
        basis,
        superposition,
    )?;
    let mut keyed: Vec<((i64, String), Symmetry)> = Vec::new();
    for s in found {
        let all: Vec<Basic> = s.xi.iter().chain(&s.eta).cloned().collect();
        let ops: i64 = crate::core::function::count_ops(&Basic::new(Head::Tuple, all))?;
        keyed.push(((ops, crate::solvers::second_order::symmetry_repr(&s)), s));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, s)| s).collect())
}

/// A similarity reduction (`Reduction`).
#[derive(Clone, Debug)]
pub struct Reduction {
    pub z: Basic,
    pub variable: Basic,
    pub f: Basic,
    pub solution: Basic,
    pub ode: Basic,
}

impl Reduction {
    /// (`__repr__`)
    pub fn repr(&self) -> String {
        format!("Reduction(z={}, solution={}, ode={})", self.z, self.solution, self.ode)
    }
}

fn dsolve(equation: &Basic, f: &Basic, timeout: Timeout) -> PolyResult<Option<Vec<Basic>>> {
    attempt(timeout, || crate::solvers::dsolve::dsolve(equation, f))
}

/// The solutions of `e = 0` for `v`, natively when linear (`_solve`).
fn solve(e: &Basic, v: &Basic, timeout: Timeout) -> PolyResult<Vec<Basic>> {
    let a: Basic = diff(e, v)?;
    if a != Basic::zero() && !a.has(v) {
        let b: Basic = subs(e, v, &Basic::zero())?;
        if !b.has(v) && !crate::core::logic::is_boolean(&b) {
            return Ok(vec![div(&neg(&b)?, &a)?]);
        }
    }
    Ok(attempt(timeout, || crate::solvers::solvers::solve_equation(e, v))?
        .flatten()
        .unwrap_or_default())
}

/// The integration constants `C1, C2, ...` (`_constants`).
fn constants(e: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = e
        .free_symbols()
        .into_iter()
        .filter(|c| c.to_string().starts_with('C'))
        .collect();
    found.sort_by_key(|c| c.to_string());
    found
}

/// An invariant `z(x)` of the flow on the two independent variables and
/// the parametrizing index (`_invariant_of_flow`).
fn invariant_of_flow(xi: &[Basic], x: &[Basic], timeout: Timeout) -> PolyResult<Option<(usize, Basic)>> {
    let nonzero: Vec<usize> = (0..xi.len()).filter(|i| xi[*i] != Basic::zero()).collect();
    if nonzero.is_empty() {
        return Ok(None);
    }
    if nonzero.len() == 1 {
        let i: usize = nonzero[0];
        return Ok(Some((i, x[1 - i].clone())));
    }
    // prefer a constant xi as the parametrizing direction
    let i: usize = *nonzero
        .iter()
        .min_by_key(|k| (!crate::core::assumptions::is_number(&xi[**k]), xi[**k].to_string()))
        .expect("a nonzero infinitesimal");
    let j: usize = 1 - i;
    let big_x: Basic = undefined("X").apply(vec![x[i].clone()])?;
    let rhs: Basic = subs(&div(&xi[j], &xi[i])?, &x[j], &big_x)?;
    let Some(sols) = dsolve(&eq(&diff(&big_x, &x[i])?, &rhs), &big_x, timeout)? else {
        return Ok(None);
    };
    for s in sols {
        let cs: Vec<Basic> = constants(&s);
        if cs.len() != 1 || !is_eq(&s) {
            continue;
        }
        let values: Vec<Basic> = solve(&sub(&s.args()[0], &s.args()[1])?, &cs[0], timeout)?;
        if let Some(first) = values.first() {
            let z: Basic = subs(first, &big_x, &x[j])?;
            return Ok(Some((i, simplify(&z)?)));
        }
    }
    Ok(None)
}

/// Reduce a partial differential equation in two independent variables
/// to an ordinary differential equation with a point symmetry
/// (`similarity_reduction`).
pub fn similarity_reduction(pde: &Basic, u: &Basic, symmetry: &Symmetry, timeout: Timeout) -> PolyResult<Reduction> {
    let jets: Rc<JetSpace> = JetSpace::new(std::slice::from_ref(u))?;
    if jets.q() != 1 {
        return Err(PolyError::Value("one dependent variable is expected".to_string()));
    }
    if jets.p() != 2 {
        return Err(not_implemented(
            "similarity reductions are implemented for two independent variables".to_string(),
        ));
    }
    let x: &[Basic] = &jets.x;
    let usym: Basic = jets.u[0].clone();
    let xi: &[Basic] = &symmetry.xi;
    let eta: &Basic = &symmetry.eta[0];
    if xi.iter().any(|e| e.has(&usym)) {
        return Err(not_implemented(format!(
            "the infinitesimals of the independent variables depend on {}",
            usym
        )));
    }
    if xi.iter().all(|e| *e == Basic::zero()) {
        return Err(not_implemented(
            "the symmetry moves no independent variable: no similarity variable".to_string(),
        ));
    }
    let Some((i, z)) = invariant_of_flow(xi, x, timeout)? else {
        return Err(not_implemented(format!(
            "no invariant of {} was found",
            crate::solvers::second_order::symmetry_repr(symmetry)
        )));
    };
    let j: usize = 1 - i;
    let zsym: Basic = symbol("z");
    let f_function: Arc<UndefinedFunction> = undefined("F");
    let fz: Basic = f_function.clone().apply(vec![zsym.clone()])?;
    // the invariant involving u: du/dx_i = eta/xi^i along the characteristics
    let fzx: Basic = f_function.clone().apply(vec![z.clone()])?;
    let phi: Basic = if *eta == Basic::zero() {
        fzx.clone()
    } else {
        let big_u: Basic = undefined("U").apply(vec![x[i].clone()])?;
        if xi[i] == Basic::zero() {
            return Err(not_implemented(
                "the characteristic direction has a vanishing infinitesimal".to_string(),
            ));
        }
        let xj_values: Vec<Basic> = solve(&sub(&z, &zsym)?, &x[j], timeout)?;
        let Some(xj) = xj_values.first().cloned() else {
            return Err(not_implemented(format!(
                "cannot solve the invariant {} for {}",
                z, x[j]
            )));
        };
        let rhs: Basic = crate::core::subs::subs_mapping(
            &div(eta, &xi[i])?,
            &[(usym.clone(), big_u.clone()), (x[j].clone(), xj.clone())],
        )?;
        let Some(sols) = dsolve(&eq(&diff(&big_u, &x[i])?, &rhs), &big_u, timeout)? else {
            return Err(not_implemented(format!(
                "cannot integrate the characteristic equation for {}",
                usym
            )));
        };
        let mut found: Option<Basic> = None;
        for s in sols {
            let cs: Vec<Basic> = constants(&s);
            if cs.len() != 1 || !is_eq(&s) {
                continue;
            }
            found = Some(subs(&subs(&subs(&s.args()[1], &cs[0], &fzx)?, &x[j], &xj)?, &zsym, &z)?);
            break;
        }
        match found {
            Some(p) => p,
            None => {
                return Err(not_implemented(format!(
                    "cannot integrate the characteristic equation for {}",
                    usym
                )));
            }
        }
    };
    // u = phi(x, F(z(x))) substituted, with F_k the derivatives of F
    let equation: Basic = jets.to_jet(&as_zero(pde)?)?;
    let order: usize = jets.order(&equation) as usize;
    let fk: Vec<Basic> = (0..=order).map(|k| symbol(&format!("F_{}", k))).collect();
    let ansatz: Basic = subs(&phi, &fzx, &fk[0])?;
    let dz: Vec<Basic> = x.iter().map(|v| diff(&z, v)).collect::<PolyResult<_>>()?;
    let total = |e: &Basic, m: usize| -> R {
        let mut result: Basic = diff(e, &x[m])?;
        for k in 0..order {
            result = add(&[result, mul(&[diff(e, &fk[k])?, fk[k + 1].clone(), dz[m].clone()])?])?;
        }
        Ok(result)
    };
    let mut values: Vec<(Basic, Basic)> = vec![(usym.clone(), ansatz.clone())];
    let jets_in: Vec<Jet> = jets.jets_in(&equation);
    for (a, multi) in jets_in {
        let mut e: Basic = ansatz.clone();
        for (m, count) in multi.iter().enumerate() {
            for _ in 0..*count {
                e = total(&e, m)?;
            }
        }
        values.push((jets.jet(a, &multi), e));
    }
    let reduced: Basic = replaced(&equation, &values)?;
    // the independent variables eliminated in favour of z
    let Some(mut ode) = eliminate(&reduced, &fk, x, &[j, i], &z, &zsym, timeout)? else {
        return Err(not_implemented(
            "the reduced equation still depends on the independent variables".to_string(),
        ));
    };
    let derivatives: Vec<(Basic, Basic)> = (0..=order)
        .map(|k| {
            let value: Basic = if k > 0 {
                crate::core::function::derivative_node(&fz, &[(zsym.clone(), k as u64)])
            } else {
                fz.clone()
            };
            (fk[k].clone(), value)
        })
        .collect();
    // an overall factor removed (SymPy's factor_list in the derivatives
    // of F, the highest first)
    let mut gens: Vec<Basic> = (1..=order).rev().map(|k| fk[k].clone()).collect();
    gens.push(fk[0].clone());
    match crate::polys::polytools::factor_list(&ode, &gens) {
        Ok((_, factors)) => {
            let relevant: Vec<Basic> = factors
                .iter()
                .filter(|(f, _)| fk.iter().any(|g| f.has(g)))
                .map(|(f, m)| pow(f, m))
                .collect::<PolyResult<_>>()?;
            if relevant.len() == 1 {
                ode = crate::core::function::expand_default(&relevant[0])?;
            }
        }
        Err(PolyError::Polys(..)) => {}
        Err(e) => return Err(e),
    }
    let ode: Basic = replaced(&ode, &derivatives)?;
    Ok(Reduction {
        z,
        variable: zsym,
        f: fz,
        solution: phi,
        ode,
    })
}

/// One independent variable substituted by its value in terms of `z`,
/// the coefficients normalized by the leading one (`_eliminate`).
#[allow(clippy::too_many_arguments)]
fn eliminate(
    reduced: &Basic,
    fk: &[Basic],
    x: &[Basic],
    order: &[usize],
    z: &Basic,
    zsym: &Basic,
    timeout: Timeout,
) -> PolyResult<Option<Basic>> {
    let Some(terms) = crate::solvers::lie::polynomial_terms(reduced, fk)? else {
        return Ok(None);
    };
    for &k in order {
        for value in solve(&sub(z, zsym)?, &x[k], timeout)? {
            let coefficients: Vec<Basic> = terms
                .iter()
                .map(|(_, c)| subs(c, &x[k], &value))
                .collect::<PolyResult<_>>()?;
            let Some(leading) = coefficients.first().cloned() else {
                continue;
            };
            if leading == Basic::zero() {
                continue;
            }
            let mut normalized: Vec<Basic> = Vec::new();
            for c in &coefficients {
                let e: Option<Basic> = attempt(timeout, || simplify(&div(c, &leading)?))?;
                match e {
                    Some(e) if !e.free_symbols().iter().any(|s| x.contains(s)) => normalized.push(e),
                    _ => {
                        normalized.clear();
                        break;
                    }
                }
            }
            if !normalized.is_empty() {
                let mut result: Vec<Basic> = Vec::new();
                for ((monomial, _), c) in terms.iter().zip(normalized) {
                    let mut term: Basic = c;
                    for (f, n) in fk.iter().zip(monomial) {
                        term = mul(&[term, pow(f, &Basic::int(i64::from(*n)))?])?;
                    }
                    result.push(term);
                }
                return Ok(Some(add(&result)?));
            }
        }
    }
    Ok(None)
}

/// Whether `u = value` solves the equation (`check_pde_solution`).
pub fn check_pde_solution(pde: &Basic, u: &Basic, value: &Basic, timeout: Timeout) -> PolyResult<bool> {
    let equation: Basic = as_zero(pde)?;
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    for d in equation.preorder() {
        if matches!(d.head(), Head::Derivative) && d.args()[0] == *u && !replacements.iter().any(|(k, _)| *k == d) {
            let variables: Vec<(Basic, u64)> = d.args()[1..]
                .iter()
                .map(|pair| {
                    let count: u64 = pair.args()[1].to_string().parse::<u64>().unwrap_or(1);
                    (pair.args()[0].clone(), count)
                })
                .collect();
            replacements.push((d.clone(), crate::core::function::diff(value, &variables)?));
        }
    }
    replacements.push((u.clone(), value.clone()));
    let substituted: Basic = replaced(&equation, &replacements)?;
    if substituted == Basic::zero() {
        return Ok(true);
    }
    Ok(attempt(timeout, || simplify(&substituted))? == Some(Basic::zero()))
}

/// Group invariant solutions of a partial differential equation
/// (`pdsolve_lie`).
pub fn pdsolve_lie(
    pde: &Basic,
    u: &Basic,
    given: Option<&[Symmetry]>,
    degree: u32,
    check: bool,
    timeout: Timeout,
) -> PolyResult<Vec<Basic>> {
    let syms: Vec<Symmetry> = match given {
        Some(s) => s.to_vec(),
        None => pde_symmetries(pde, u, degree, &[], false)?,
    };
    let mut solutions: Vec<Basic> = Vec::new();
    let mut seen: Vec<Basic> = Vec::new();
    for sym in &syms {
        let reduction: Reduction = match similarity_reduction(pde, u, sym, timeout) {
            Ok(r) => r,
            Err(e) if is_genuine_not_implemented(&e) => continue,
            Err(e) => return Err(e),
        };
        let Some(ode_solutions) = dsolve(&reduction.ode, &reduction.f, timeout)? else {
            continue;
        };
        for s in ode_solutions {
            if !is_eq(&s) || s.args()[0] != reduction.f {
                continue;
            }
            let f_value: Basic = subs(&s.args()[1], &reduction.variable, &reduction.z)?;
            let applied: Basic = match reduction.f.head() {
                Head::Undefined(g) => g.clone().apply(vec![reduction.z.clone()])?,
                _ => reduction.f.clone(),
            };
            let value: Basic = subs(&reduction.solution, &applied, &f_value)?;
            if seen.contains(&value) {
                continue;
            }
            seen.push(value.clone());
            if check && !check_pde_solution(pde, u, &value, timeout)? {
                continue;
            }
            solutions.push(eq(u, &value));
        }
    }
    Ok(solutions)
}
