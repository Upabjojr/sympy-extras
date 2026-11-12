//! Port of `sympy_extras/polys/rationalsolve.py`: SymPy's `solveset` over
//! the complex numbers restricted to the equations rational in the
//! unknown ([`solveset_rational`]): the solutions in radicals in the order
//! of SymPy's `FiniteSet`, and the points excluded as roots of a
//! denominator. The steps of SymPy's `_solveset` are kept, since the form
//! of the solutions depends on them; everything else gives `None`.

// PORTED FROM SYMPY: sympy/solvers/solveset.py (_solveset, _invert_complex,
// _solve_as_rational, _solve_as_poly, domain_check, _domain_check, FiniteSet
// ordering) (SymPy 1.14), through sympy_extras; BSD 3-clause license, see
// sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::{as_independent, is_true, polynomial_like, PolynomialKind};
use crate::core::function::{expand_default, expand_with, Hints};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::core::sorting::ordered;
use crate::error::{PolyError, PolyResult};

use super::polyroots::roots;
use super::polytools::degree;

/// The solutions in SymPy's order and the excluded points (SymPy's
/// `Complement(FiniteSet(*solutions), FiniteSet(*excluded))`).
pub type Solutions = (Vec<Basic>, Vec<Basic>);

fn div(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}

fn is_integer_value(e: &Basic, v: i64) -> bool {
    matches!(e.head(), Head::Rational { integer_class: true, value } if *value == crate::numbers::Rat::from(v))
}

/// The solutions of `f = 0` for `x` as SymPy's `solveset` gives them,
/// `None` outside the rational equations.
pub fn solveset_rational(f: &Basic, x: &Basic) -> PolyResult<Option<Solutions>> {
    let Some((solutions, excluded)) = solveset(f, x)? else {
        return Ok(None);
    };
    let (_, dependent) = as_independent(f, x, true)?;
    let (_, fx) = as_independent(&dependent, x, false)?;
    let mut kept: Vec<Basic> = Vec::new();
    for s in solutions {
        if !is_true(&s, Predicate::infinite) && domain_check(&fx, x, &s)? {
            kept.push(s);
        }
    }
    Ok(Some((kept, excluded)))
}

fn finite_set(elements: Vec<Basic>) -> Vec<Basic> {
    let mut unique: Vec<Basic> = Vec::new();
    for e in elements {
        if !unique.contains(&e) {
            unique.push(e);
        }
    }
    ordered(unique)
}

fn union(parts: Vec<Solutions>) -> Option<Solutions> {
    if parts.iter().any(|(_, excluded)| !excluded.is_empty()) {
        return None;
    }
    Some((finite_set(parts.into_iter().flat_map(|(s, _)| s).collect()), Vec::new()))
}

fn infinite(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
    )
}

/// `_solveset`.
fn solveset(f: &Basic, x: &Basic) -> PolyResult<Option<Solutions>> {
    let mut f: Basic = f.clone();
    if f.is_mul() {
        let (coeff, rest) = as_independent(&f, x, false)?;
        f = if infinite(&coeff) {
            super::rationaltools::together(&f, false, true)?
        } else {
            rest
        };
    } else if f.is_add() {
        let (a, h) = as_independent(&f, x, true)?;
        let (m, h) = as_independent(&h, x, false)?;
        if !(infinite(&m) || is_integer_value(&m, 0)) {
            f = add(&[div(&a, &m)?, h])?;
        }
    }
    if is_true(&expand_default(&f)?, Predicate::zero) {
        return Ok(None);
    }
    if !f.has(x) {
        return Ok(Some((Vec::new(), Vec::new())));
    }
    // _is_finite_with_finite_vars of a rational expression: a polynomial in
    // all its symbols
    let finite = |m: &Basic| {
        let syms: Vec<Basic> = m.free_symbols().into_iter().collect();
        polynomial_like(m, &syms, PolynomialKind::Polynomial) == Some(true)
    };
    if f.is_mul() && f.args().iter().all(finite) {
        let mut parts: Vec<Solutions> = Vec::new();
        for m in f.args() {
            let Some(part) = solveset(m, x)? else { return Ok(None) };
            parts.push(part);
        }
        return Ok(union(parts));
    }
    if !(polynomial_like(&f, std::slice::from_ref(x), PolynomialKind::Rational) == Some(true) && !f.has(&Basic::nan()))
    {
        return Ok(None);
    }
    let Some((lhs, rhs)) = invert_complex(&f, &Basic::zero(), x)? else {
        return Ok(None);
    };
    if lhs == *x {
        let simplified: Basic = crate::simplify::signsimp::signsimp(&rhs, true)?;
        let (content, primitive) = crate::core::exprtools::as_content_primitive(&simplified, false, true)?;
        return Ok(Some((vec![mul(&[content, primitive])?], Vec::new())));
    }
    let equation: Basic = add(&[lhs, mul(&[Basic::negative_one(), rhs])?])?;
    if equation == f {
        return solve_as_rational(&equation, x);
    }
    solveset(&equation, x)
}

/// `(lhs, rhs)` with `f = y` equivalent to `lhs = rhs` (`_invert_complex`,
/// its `Add`, `Mul` and `Pow` cases); `None` for no solution.
fn invert_complex(f: &Basic, y: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let (mut f, mut y) = (f.clone(), y.clone());
    loop {
        if f == *x {
            return Ok(Some((f, y)));
        }
        if f.is_add() {
            let (g, h) = as_independent(&f, x, true)?;
            if !is_integer_value(&g, 0) {
                y = add(&[y, mul(&[Basic::negative_one(), g])?])?;
                f = h;
                continue;
            }
        }
        if f.is_mul() {
            let (g, h) = as_independent(&f, x, false)?;
            if !is_integer_value(&g, 1) {
                if infinite(&g) {
                    return Ok(None);
                }
                y = div(&y, &g)?;
                f = h;
                continue;
            }
        }
        if f.is_pow() && is_integer_value(&y, 0) {
            let e: &Basic = &f.args()[1];
            if e.as_rational().is_some() && is_true(e, Predicate::positive) {
                f = f.args()[0].clone();
                continue;
            }
        }
        return Ok(Some((f, y)));
    }
}

/// `_solve_as_rational`.
fn solve_as_rational(f: &Basic, x: &Basic) -> PolyResult<Option<Solutions>> {
    let mut expanded: Basic = f.clone();
    loop {
        let again: Basic =
            crate::core::function::expand_mul(&crate::core::function::expand_multinomial(&expanded, true)?, true)?;
        if again == expanded {
            break;
        }
        expanded = again;
    }
    let (g, h) = crate::simplify::radsimp::fraction(&super::rationaltools::together(&expanded, true, true)?)?;
    if !h.has(x) {
        return solve_as_poly(&g, x);
    }
    let (Some(valid), Some(invalid)) = (solveset(&g, x)?, solveset(&h, x)?) else {
        return Ok(None);
    };
    if !valid.1.is_empty() || !invalid.1.is_empty() {
        return Ok(None);
    }
    let removed: Vec<Basic> = invalid.0;
    let kept: Vec<Basic> = valid.0.into_iter().filter(|s| !removed.contains(s)).collect();
    Ok(Some((kept, removed)))
}

/// `_solve_as_poly`: a polynomial in `x` with all its roots in radicals
/// (of degree two at most with symbols in the coefficients).
fn solve_as_poly(f: &Basic, x: &Basic) -> PolyResult<Option<Solutions>> {
    if polynomial_like(f, std::slice::from_ref(x), PolynomialKind::Polynomial) != Some(true) {
        return Ok(None);
    }
    let attempt = || -> PolyResult<Option<(i64, super::polyroots::RootsResult)>> {
        let n: i64 = degree(f, x)?;
        let others: bool = f.free_symbols().iter().any(|s| s != x);
        if others && n > 2 {
            return Ok(None);
        }
        Ok(Some((n, roots(f, std::slice::from_ref(x), true, true, None)?)))
    };
    let (n, found) = match attempt() {
        Ok(Some(r)) => r,
        Ok(None) => return Ok(None),
        // BasePolynomialError: the classes of polys.polyerrors
        Err(PolyError::Polys(..)) | Err(PolyError::HeuristicGCDFailed(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if (found.values().sum::<usize>() as i64) < n {
        return Ok(None);
    }
    let mut solutions: Vec<Basic> = finite_set(found.keys().cloned().collect());
    if solutions.iter().all(|s| s.free_symbols().is_empty()) {
        let hints = Hints {
            mul: false,
            multinomial: false,
            power_base: false,
            power_exp: false,
            log: false,
            force: false,
            complex: true,
        };
        let mut expanded: Vec<Basic> = Vec::new();
        for s in solutions {
            expanded.push(expand_with(&s, true, hints)?);
        }
        solutions = finite_set(expanded);
    }
    Ok(Some((solutions, Vec::new())))
}

/// False when `f` or one of its subexpressions becomes infinite at `x = p`
/// (`_domain_check`).
fn domain_check(f: &Basic, x: &Basic, p: &Basic) -> PolyResult<bool> {
    if f.args().is_empty() {
        return Ok(true);
    }
    let value: Basic = crate::core::subs::subs(f, x, p)?;
    if is_true(&value, Predicate::infinite) {
        return Ok(false);
    }
    for g in f.args() {
        if !domain_check(g, x, p)? {
            return Ok(false);
        }
    }
    Ok(true)
}
