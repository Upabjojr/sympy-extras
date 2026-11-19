//! Port of `sympy_extras/solvers/solveset.py`: SymPy's `solveset` of one
//! relation in one unknown over the real or the complex numbers (the
//! cases the callers of the package meet): rational relations with
//! numbers as coefficients by all the roots of the numerator, the real
//! ones told apart exactly or by reliable values and the signs between
//! them; with parameters, the roots of an equation, linear inequalities
//! and even binomials; transcendental relations by
//! [`super::transcendental`]; the inverse images of the periodic functions
//! over the complex numbers. A `ConditionSet` when it is not solved.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::sets::sets::{empty_set, finite_set, intersection, interval, reals, union};

type R = PolyResult<Basic>;

fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn add2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn mul2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn hints() -> crate::core::function::Hints {
    crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    }
}

fn fraction(f: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::simplify::radsimp::fraction(&crate::polys::rationaltools::together(f, false, true)?)
}

fn flipped(op: RelOp) -> RelOp {
    match op {
        RelOp::Lt => RelOp::Gt,
        RelOp::Le => RelOp::Ge,
        RelOp::Gt => RelOp::Lt,
        RelOp::Ge => RelOp::Le,
        other => other,
    }
}

fn distributed(r: &Basic) -> R {
    let candidate: Basic = crate::core::function::expand_mul(r, true)?;
    if crate::core::function::count_ops(&candidate)? < crate::core::function::count_ops(r)? {
        return Ok(candidate);
    }
    Ok(r.clone())
}

/// The roots which are numbers written `a + b*I` (`_expanded_numbers`).
fn expanded_numbers(found: &[Basic]) -> PolyResult<Vec<Basic>> {
    if found.iter().any(|r| !r.free_symbols().is_empty()) {
        return found
            .iter()
            .map(|r| {
                let (_, d) = crate::core::expr::as_numer_denom(r)?;
                if d != Basic::one() {
                    distributed(r)
                } else {
                    Ok(r.clone())
                }
            })
            .collect();
    }
    found
        .iter()
        .map(|r| {
            crate::core::function::expand_with(
                r,
                true,
                crate::core::function::Hints {
                    complex: true,
                    ..hints()
                },
            )
        })
        .collect()
}

fn numeric(e: &Basic) -> PolyResult<Option<(f64, f64)>> {
    let settings = crate::settings::settings();
    if !settings.numerical_checks {
        return Ok(None);
    }
    Ok(crate::timeout::attempt(|| Ok(crate::numeric::reliable_complex(e, settings.precision, &[])))?.flatten())
}

fn modulus(z: (f64, f64)) -> f64 {
    (z.0 * z.0 + z.1 * z.1).sqrt()
}

/// The real roots among the distinct roots of a polynomial, and those
/// undecided (`_real_roots`).
fn real_roots(found: &[Basic], real_coefficients: bool) -> PolyResult<(Vec<Basic>, Vec<Basic>)> {
    let mut real: Vec<Basic> = Vec::new();
    let mut undecided: Vec<Basic> = Vec::new();
    let mut values: Vec<Option<(f64, f64)>> = Vec::new();
    for r in found {
        values.push(if is(r, Predicate::real).is_some() {
            None
        } else {
            numeric(r)?
        });
    }
    let tolerance: f64 = 10f64.powi(-((crate::settings::settings().precision / 2) as i32));
    for (i, (r, value)) in found.iter().zip(&values).enumerate() {
        match is(r, Predicate::real) {
            Some(true) => {
                real.push(r.clone());
                continue;
            }
            Some(false) => continue,
            None => {}
        }
        let Some(v) = value else {
            undecided.push(r.clone());
            continue;
        };
        if v.1.abs() > tolerance * modulus(*v).max(1.0) {
            continue;
        }
        let isolated: bool = values.iter().enumerate().all(|(j, other)| match other {
            None => true,
            Some(o) => j == i || modulus((o.0 - v.0, o.1 - v.1)) > 4.0 * tolerance * modulus(*v).max(1.0),
        });
        if real_coefficients && isolated {
            real.push(r.clone());
        } else {
            undecided.push(r.clone());
        }
    }
    Ok((real, undecided))
}

fn sorted_real(points: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let mut result: Vec<Basic> = Vec::new();
    for p in points {
        let mut position: usize = 0;
        while position < result.len() {
            let Some(s) = super::isolation::compare_points(p, &result[position])? else {
                return Ok(None);
            };
            if s < 0 {
                break;
            }
            position += 1;
        }
        result.insert(position, p.clone());
    }
    Ok(Some(result))
}

fn sign_at(f: &Basic, x: &Basic, point: &Basic) -> PolyResult<Option<i32>> {
    let value: Basic = crate::core::subs::subs(f, x, point)?;
    crate::assumptions::analysis::certified_sign(&value)
}

fn max_degree(cs: &[(i64, Basic)]) -> i64 {
    cs.iter().map(|(d, _)| *d).max().unwrap_or(0)
}

/// The real solutions of `f op 0` for `f` rational in `x` with numbers as
/// coefficients (`real_polynomial_solutions`).
pub fn real_polynomial_solutions(f: &Basic, x: &Basic, op: RelOp) -> PolyResult<Option<Basic>> {
    use super::solvers::{coefficients, polynomial_roots, zero_truth};
    let (numerator, denominator) = fraction(f)?;
    let other = |e: &Basic| e.free_symbols().iter().any(|s| s != x);
    if other(&numerator) || other(&denominator) {
        return Ok(None);
    }
    let (Some(cs), Some(dcs)) = (coefficients(&numerator, x)?, coefficients(&denominator, x)?) else {
        return Ok(None);
    };
    if cs.is_empty() {
        return Ok(Some(if matches!(op, RelOp::Eq | RelOp::Le | RelOp::Ge) {
            reals()
        } else {
            empty_set()
        }));
    }
    let degree: i64 = max_degree(&cs);
    let real_coefficients: bool = cs.iter().chain(&dcs).all(|(_, c)| yes(c, Predicate::real));
    let mut roots: Vec<Basic> = Vec::new();
    if degree > 0 {
        let Some(found) = polynomial_roots(&numerator, x, degree as usize)? else {
            return Ok(None);
        };
        let (r, undecided) = real_roots(&expanded_numbers(&found)?, real_coefficients)?;
        if !undecided.is_empty() {
            return Ok(None);
        }
        roots = r;
    }
    let mut poles: Vec<Basic> = Vec::new();
    if max_degree(&dcs) > 0 {
        let Some(found) = polynomial_roots(&denominator, x, max_degree(&dcs) as usize)? else {
            return Ok(None);
        };
        let (p, undecided) = real_roots(&found, real_coefficients)?;
        if !undecided.is_empty() {
            return Ok(None);
        }
        poles = p;
    }
    let mut kept: Vec<Basic> = Vec::new();
    for r in roots {
        let mut is_pole: bool = false;
        for p in &poles {
            if zero_truth(&sub(&r, p)?)? == Some(true) {
                is_pole = true;
                break;
            }
        }
        if !is_pole {
            kept.push(r);
        }
    }
    let roots: Vec<Basic> = kept;
    if op == RelOp::Eq {
        return Ok(Some(if roots.is_empty() {
            empty_set()
        } else {
            finite_set(&roots)
        }));
    }
    if !real_coefficients {
        return Ok(None);
    }
    let mut all: Vec<Basic> = roots.clone();
    all.extend(poles.iter().cloned());
    let Some(ordered) = sorted_real(&all)? else {
        return Ok(None);
    };
    let mut bounds: Vec<Basic> = vec![Basic::negative_oo()];
    bounds.extend(ordered);
    bounds.push(Basic::oo());
    if op == RelOp::Ne {
        let parts: Vec<Basic> = bounds
            .windows(2)
            .map(|w| interval(&w[0], &w[1], true, true))
            .collect::<PolyResult<_>>()?;
        return Ok(Some(union(&parts)?));
    }
    let wanted: i32 = if matches!(op, RelOp::Gt | RelOp::Ge) { 1 } else { -1 };
    let strict: bool = matches!(op, RelOp::Gt | RelOp::Lt);
    let mut parts: Vec<Basic> = Vec::new();
    for w in bounds.windows(2) {
        let Some(point) = super::isolation::point_between(&w[0], &w[1])? else {
            return Ok(None);
        };
        let Some(s) = sign_at(f, x, &point)? else {
            return Ok(None);
        };
        if s == wanted {
            parts.push(interval(&w[0], &w[1], true, true)?);
        }
    }
    if !strict {
        for r in &roots {
            parts.push(finite_set(std::slice::from_ref(r)));
        }
    }
    Ok(Some(if parts.is_empty() { empty_set() } else { union(&parts)? }))
}

fn parametric_real(f: &Basic, x: &Basic, op: RelOp) -> PolyResult<Option<Basic>> {
    use super::solvers::{coefficients, polynomial_roots};
    let (numerator, denominator) = fraction(f)?;
    let Some(cs) = coefficients(&numerator, x)? else {
        return Ok(None);
    };
    if cs.iter().any(|(_, c)| c.has(x)) || denominator.has(x) {
        return Ok(None);
    }
    if cs.is_empty() {
        return Ok(Some(if matches!(op, RelOp::Eq | RelOp::Le | RelOp::Ge) {
            reals()
        } else {
            empty_set()
        }));
    }
    let degree: i64 = max_degree(&cs);
    let at = |k: i64| {
        cs.iter()
            .find(|(d, _)| *d == k)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Basic::zero)
    };
    if op == RelOp::Eq {
        if degree == 0 {
            return Ok(Some(empty_set()));
        }
        let Some(found) = polynomial_roots(&numerator, x, degree as usize)? else {
            return Ok(None);
        };
        let kept: Vec<Basic> = found.iter().filter(|r| yes(r, Predicate::real)).cloned().collect();
        let undecided: Vec<Basic> = found
            .iter()
            .filter(|r| is(r, Predicate::real).is_none())
            .cloned()
            .collect();
        let mut result: Basic = if kept.is_empty() {
            empty_set()
        } else {
            finite_set(&kept)
        };
        if !undecided.is_empty() {
            result = union(&[result, intersection(&[finite_set(&undecided), reals()])?])?;
        }
        return Ok(Some(result));
    }
    if cs.iter().any(|(_, c)| !yes(c, Predicate::real)) {
        return Ok(None);
    }
    let lead: Basic = at(degree);
    if !yes(&lead, Predicate::positive) && !yes(&lead, Predicate::negative) {
        return Ok(None);
    }
    let op: RelOp = if yes(&lead, Predicate::negative) {
        flipped(op)
    } else {
        op
    };
    let (oo, moo) = (Basic::oo(), Basic::negative_oo());
    if degree == 1 {
        let root: Basic = div(&neg(&at(0))?, &lead)?;
        return Ok(match op {
            RelOp::Gt | RelOp::Ge => Some(interval(&root, &oo, op == RelOp::Gt, true)?),
            RelOp::Lt | RelOp::Le => Some(interval(&moo, &root, true, op == RelOp::Lt)?),
            _ => None,
        });
    }
    if cs.iter().all(|(d, _)| *d == 0 || *d == degree) && degree % 2 == 0 {
        let c: Basic = div(&neg(&at(0))?, &lead)?;
        if !yes(&c, Predicate::positive) {
            return Ok(None);
        }
        let r: Basic = crate::core::power::pow(
            &c,
            &Basic::rational(crate::numbers::Int::ONE, crate::numbers::Int::from(degree))?,
        )?;
        let nr: Basic = neg(&r)?;
        return Ok(match op {
            RelOp::Lt | RelOp::Le => Some(interval(&nr, &r, op == RelOp::Lt, op == RelOp::Lt)?),
            RelOp::Gt | RelOp::Ge => Some(union(&[
                interval(&moo, &nr, true, op == RelOp::Gt)?,
                interval(&r, &oo, op == RelOp::Gt, true)?,
            ])?),
            _ => None,
        });
    }
    Ok(None)
}

fn applications(f: &Basic) -> Vec<Basic> {
    f.atoms(Some(&|n: &Basic| {
        matches!(n.head(), Head::Function(_) | Head::Undefined(_))
    }))
}

/// Whether `x` occurs in `f` only in sums, products and powers with
/// exponents free of `x` (`_algebraic`).
fn algebraic(f: &Basic, x: &Basic) -> bool {
    if applications(f).iter().any(|a| a.has(x)) {
        return false;
    }
    !f.atoms(Some(&|n: &Basic| n.is_pow()))
        .iter()
        .any(|p| p.args()[1].has(x))
}

fn without_periodic(f: &Basic, x: &Basic) -> bool {
    applications(f).iter().all(|a| {
        !a.has(x)
            || matches!(
                a.head(),
                Head::Function(Function::Exp | Function::Log | Function::LambertW)
            )
    })
}

fn candidates_real(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let found: Option<Vec<Basic>> = crate::timeout::attempt(|| super::solvers::solve_equation(f, x))?.flatten();
    let Some(found) = found else { return Ok(None) };
    let tolerance: f64 = 10f64.powi(-((crate::settings::settings().precision / 2) as i32));
    let mut kept: Vec<Basic> = Vec::new();
    for v in found {
        let mut real: Option<bool> = is(&v, Predicate::real);
        if real.is_none() {
            let Some(value) = numeric(&v)? else { return Ok(None) };
            if value.1.abs() > tolerance * modulus(value).max(1.0) {
                real = Some(false);
            } else {
                return Ok(None);
            }
        }
        if real == Some(false) {
            continue;
        }
        match super::solvers::checksol(f, x, &v)? {
            Some(false) => continue,
            None => return Ok(None),
            Some(true) => kept.push(v),
        }
    }
    Ok(Some(if kept.is_empty() {
        empty_set()
    } else {
        finite_set(&kept)
    }))
}

/// The solver given to the transcendental solver (`_formula_solver`).
fn formula_solver(formula: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    match formula.head() {
        Head::BooleanTrue => return Ok(Some(reals())),
        Head::BooleanFalse => return Ok(Some(empty_set())),
        Head::Relational(_) => return solve_real(formula, v),
        Head::And | Head::Or => {
            let mut parts: Vec<Basic> = Vec::new();
            for a in formula.args() {
                let Some(part) = formula_solver(a, v)? else {
                    return Ok(None);
                };
                parts.push(part);
            }
            return Ok(Some(if matches!(formula.head(), Head::And) {
                intersection(&parts)?
            } else {
                union(&parts)?
            }));
        }
        _ => {}
    }
    Ok(None)
}

fn rel_op(e: &Basic) -> RelOp {
    match e.head() {
        Head::Relational(op) => *op,
        _ => unreachable!("a relation"),
    }
}

fn solve_real(relation: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    use super::solvers::coefficients;
    let f: Basic = sub(&relation.args()[0], &relation.args()[1])?;
    let op: RelOp = rel_op(relation);
    if !f.has(x) {
        return Ok(None);
    }
    let (numerator, denominator) = fraction(&f)?;
    let rational: bool = coefficients(&numerator, x)?.is_some() && coefficients(&denominator, x)?.is_some();
    if rational {
        if f.free_symbols().iter().all(|s| s == x) {
            return real_polynomial_solutions(&f, x, op);
        }
        return parametric_real(&f, x, op);
    }
    let solver = |formula: &Basic, v: &Basic| formula_solver(formula, v);
    let found: Option<Basic> = crate::timeout::attempt(|| {
        super::transcendental::solve_transcendental(relation, x, Some(&solver), None, true)
    })?
    .flatten();
    if found.is_some() {
        return Ok(found);
    }
    if op == RelOp::Eq && without_periodic(&f, x) {
        return candidates_real(&f, x);
    }
    if op == RelOp::Eq {
        return periodic_real(&f, x);
    }
    Ok(None)
}

// -- the complex equations

fn complex_branches(g: &Basic, r: &Basic) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    use crate::functions::elementary::{inverse as h, inverse as t};
    let pi: Basic = Basic::pi();
    let i: Basic = Basic::imaginary_unit();
    let two_pi: Basic = mul2(&crate::core::basic::Basic::int(2), &pi)?;
    let two_pi_i: Basic = mul2(&two_pi, &i)?;
    let Head::Function(f) = g.head() else { return Ok(None) };
    Ok(match f {
        Function::Exp => {
            if *r == Basic::zero() {
                Some(vec![])
            } else {
                Some(vec![(crate::functions::elementary::exponential::log(r)?, two_pi_i)])
            }
        }
        Function::Sin => {
            let a: Basic = t::asin(r)?;
            Some(vec![(a.clone(), two_pi.clone()), (sub(&pi, &a)?, two_pi)])
        }
        Function::Cos => {
            let a: Basic = t::acos(r)?;
            Some(vec![(a.clone(), two_pi.clone()), (sub(&two_pi, &a)?, two_pi)])
        }
        Function::Tan => Some(vec![(t::atan(r)?, pi)]),
        Function::Sinh => {
            let a: Basic = h::asinh(r)?;
            Some(vec![
                (a.clone(), two_pi_i.clone()),
                (sub(&mul2(&i, &pi)?, &a)?, two_pi_i),
            ])
        }
        Function::Cosh => {
            let a: Basic = h::acosh(r)?;
            Some(vec![(a.clone(), two_pi_i.clone()), (sub(&two_pi_i, &a)?, two_pi_i)])
        }
        Function::Tanh => Some(vec![(h::atanh(r)?, mul2(&pi, &i)?)]),
        _ => None,
    })
}

/// `(g, r)` with `f = 0` equivalent to `g = r` (`_peel`).
fn peel(f: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut lhs: Basic = f.clone();
    let mut r: Basic = Basic::zero();
    for _ in 0..50 {
        if lhs.is_add() {
            let (independent, dependent) = crate::core::expr::as_independent(&lhs, x, true)?;
            if independent == Basic::zero() {
                break;
            }
            r = sub(&r, &independent)?;
            lhs = dependent;
            continue;
        }
        if lhs.is_mul() {
            let (independent, dependent) = crate::core::expr::as_independent(&lhs, x, false)?;
            if independent == Basic::one() {
                break;
            }
            r = div(&r, &independent)?;
            lhs = dependent;
            continue;
        }
        break;
    }
    Ok((lhs, r))
}

fn periodic(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let (g, r) = peel(f, x)?;
    if !matches!(g.head(), Head::Function(_)) || g.args().len() != 1 || r.has(x) {
        return Ok(None);
    }
    let u: Basic = g.args()[0].clone();
    let Some(branches) = complex_branches(&g, &r)? else {
        return Ok(None);
    };
    let n: Basic = crate::core::symbol::Dummy::new("n", &[(Predicate::integer, true)]).0;
    let a: Basic = crate::core::function::derivative(&u, x)?;
    let linear: bool = !a.has(x) && a != Basic::zero();
    let b: Basic = if linear {
        crate::core::subs::subs(&u, x, &Basic::zero())?
    } else {
        Basic::zero()
    };
    let mut images: Vec<Basic> = Vec::new();
    let mut seen: Vec<Basic> = Vec::new();
    for (value, period) in branches {
        let target: Basic = add2(&value, &mul2(&n, &period)?)?;
        let expressions: Vec<Basic> = if linear {
            vec![crate::polys::polytools::cancel(&div(&sub(&target, &b)?, &a)?, &[])?]
        } else {
            let found: Option<Vec<Basic>> =
                crate::timeout::attempt(|| super::solvers::solve_equation(&sub(&u, &target)?, x))?.flatten();
            let Some(found) = found else { return Ok(None) };
            found
        };
        for expression in expressions {
            let mut duplicate: bool = false;
            for other in &seen {
                if crate::polys::polytools::cancel(&sub(&expression, other)?, &[])? == Basic::zero() {
                    duplicate = true;
                    break;
                }
            }
            if duplicate {
                continue;
            }
            seen.push(expression.clone());
            images.push(crate::sets::fancysets::image_set(
                &Basic::new(Head::Tuple, vec![n.clone()]),
                &expression,
                &[Basic::atom(Head::Integers)],
            )?);
        }
    }
    Ok(Some(if images.is_empty() {
        empty_set()
    } else {
        union(&images)?
    }))
}

fn periodic_real(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let (g, r) = peel(f, x)?;
    if !matches!(g.head(), Head::Function(Function::Sin | Function::Cos | Function::Tan))
        || r.has(x)
        || !yes(&r, Predicate::real)
    {
        return Ok(None);
    }
    if !matches!(g.head(), Head::Function(Function::Tan)) {
        let bound: Basic =
            crate::core::relational::le(&crate::functions::elementary::complexes::abs_(&r)?, &Basic::one())?;
        if !matches!(bound.head(), Head::BooleanTrue) {
            return Ok(None);
        }
    }
    let Some(found) = periodic(f, x)? else { return Ok(None) };
    let pieces: Vec<Basic> = if matches!(found.head(), Head::Union) {
        found.args().to_vec()
    } else {
        vec![found.clone()]
    };
    for piece in &pieces {
        if !matches!(piece.head(), Head::ImageSet) {
            return Ok(None);
        }
        let image: &Basic = &piece.args()[1];
        if matches!(image.head(), Head::Tuple) || is(image, Predicate::real) == Some(false) {
            return Ok(None);
        }
    }
    Ok(Some(found))
}

fn solve_complex(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    use super::solvers::{checksol, coefficients, polynomial_roots, solve_equation, zero_truth};
    let (numerator, denominator) = fraction(f)?;
    let cs = coefficients(&numerator, x)?;
    if let Some(cs) = cs.filter(|c| !c.iter().any(|(_, v)| v.has(x))) {
        if coefficients(&denominator, x)?.is_some() {
            let degree: i64 = max_degree(&cs);
            if degree == 0 {
                return Ok(Some(empty_set()));
            }
            let mut found: Option<Vec<Basic>> = polynomial_roots(&numerator, x, degree as usize)?;
            if found.is_none() {
                found = crate::timeout::attempt(|| solve_equation(&numerator, x))?.flatten();
            }
            let Some(found) = found else { return Ok(None) };
            let found: Vec<Basic> = expanded_numbers(&found)?;
            let mut kept: Vec<Basic> = Vec::new();
            for r in found {
                if !denominator.has(x) || zero_truth(&crate::core::subs::subs(&denominator, x, &r)?)? != Some(true) {
                    kept.push(r);
                }
            }
            return Ok(Some(if kept.is_empty() {
                empty_set()
            } else {
                finite_set(&kept)
            }));
        }
    }
    if let Some(p) = periodic(f, x)? {
        return Ok(Some(p));
    }
    if algebraic(f, x) {
        let candidates: Option<Vec<Basic>> = crate::timeout::attempt(|| solve_equation(f, x))?.flatten();
        let Some(candidates) = candidates else { return Ok(None) };
        let mut kept: Vec<Basic> = Vec::new();
        for v in candidates {
            match checksol(f, x, &v)? {
                Some(false) => continue,
                None => return Ok(None),
                Some(true) => kept.push(v),
            }
        }
        return Ok(Some(if kept.is_empty() {
            empty_set()
        } else {
            finite_set(&kept)
        }));
    }
    Ok(None)
}

/// The solutions of a relation in `x` in `domain` (`solveset`): the
/// `ConditionSet` of the relation when it is not solved.
pub fn solveset(relation: &Basic, x: &Basic, domain: &Basic) -> R {
    match relation.head() {
        Head::BooleanTrue => return Ok(domain.clone()),
        Head::BooleanFalse => return Ok(empty_set()),
        _ => {}
    }
    let unsolved: Basic = crate::sets::conditionset::condition_set(x, relation, domain)?;
    if !matches!(relation.head(), Head::Relational(_)) || !relation.has(x) {
        return Ok(unsolved);
    }
    let is_reals: bool = matches!(domain.head(), Head::Reals);
    let real: bool = is_reals || crate::sets::sets::is_subset(domain, &reals())? == Some(true);
    if real {
        let Some(found) = solve_real(relation, x)? else {
            return Ok(unsolved);
        };
        return if is_reals {
            Ok(found)
        } else {
            intersection(&[found, domain.clone()])
        };
    }
    if rel_op(relation) != RelOp::Eq {
        return Ok(unsolved);
    }
    let Some(found) = solve_complex(&sub(&relation.args()[0], &relation.args()[1])?, x)? else {
        return Ok(unsolved);
    };
    if matches!(domain.head(), Head::Complexes) {
        return Ok(found);
    }
    intersection(&[found, domain.clone()])
}
