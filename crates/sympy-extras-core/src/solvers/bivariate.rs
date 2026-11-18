//! Port of `sympy_extras/solvers/bivariate.py` (SymPy's
//! `_solve_lambert`): the solutions of the equations of Lambert type in one
//! unknown, `a*log(b*X + c) + d*X + f = 0` with `X = g(x)` solved by
//! `X = -c/b + (a/d)*W(d/(a*b)*exp((c*d - b*f)/(a*b)))` on the branches
//! `k = 0` and, when real, `k = -1`; `None` when the equation is not of a
//! Lambert form. The solutions are not checked.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_independent, is};
use crate::core::symbol::Dummy;
use crate::error::PolyResult;
use crate::functions::elementary::exponential::{exp, lambertw, log};

type R = PolyResult<Basic>;
type O = PolyResult<Option<Vec<Basic>>>;

fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
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
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn mexpand(e: &Basic) -> R {
    crate::simplify::sqrtdenest::mexpand(e)
}

fn expand_log(e: &Basic, force: bool) -> R {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: true,
        force,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}

fn count_function(e: &Basic, f: Function) -> usize {
    e.preorder().iter().filter(|n| n.is_function(f)).count()
}

fn count_pow(e: &Basic) -> usize {
    e.preorder().iter().filter(|n| n.is_pow()).count()
}

#[derive(Clone, Copy)]
enum Kind {
    Log,
    Exp,
    Pow,
}

fn is_kind(e: &Basic, k: Kind) -> bool {
    match k {
        Kind::Log => e.is_function(Function::Log),
        Kind::Exp => e.is_function(Function::Exp),
        Kind::Pow => e.is_pow(),
    }
}

/// The subexpression of the kind containing `x` with the most nested
/// occurrences of the kind (`_mostfunc`).
fn mostfunc(lhs: &Basic, k: Kind, x: &Basic) -> Option<Basic> {
    let found: Vec<Basic> = lhs
        .atoms(Some(&|n: &Basic| is_kind(n, k)))
        .into_iter()
        .filter(|a| a.free_symbols().contains(x))
        .collect();
    match found.len() {
        0 => None,
        1 => Some(found[0].clone()),
        _ => {
            let candidates: Vec<Basic> = crate::core::sorting::ordered(found);
            let count = |a: &Basic| match k {
                Kind::Log => count_function(a, Function::Log),
                Kind::Exp => count_function(a, Function::Exp),
                Kind::Pow => count_pow(a),
            };
            let mut best: Option<(usize, Basic)> = None;
            for c in candidates {
                let n: usize = count(&c);
                if best.as_ref().map_or(true, |(b, _)| n > *b) {
                    best = Some((n, c));
                }
            }
            best.map(|(_, c)| c)
        }
    }
}

fn factor_terms(e: &Basic) -> R {
    crate::core::exprtools::factor_terms(e, false, false, false, true)
}

/// `(a, b, X)` with `arg = a*X + b`, `a` and `b` free of `x` (`_linab`).
fn linab(arg: &Basic, x: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    let arg: Basic = factor_terms(&crate::core::function::expand_default(arg)?)?;
    let (independent, dependent) = as_independent(&arg, x, arg.is_add())?;
    if arg.is_mul() && dependent.is_add() {
        let (a, b, big_x) = linab(&dependent, x)?;
        return Ok((mul(&[independent.clone(), a])?, mul(&[independent, b])?, big_x));
    }
    let (mut a, b, mut big_x): (Basic, Basic, Basic);
    if !arg.is_add() {
        a = independent;
        b = Basic::zero();
        big_x = dependent;
    } else {
        b = independent;
        let (aa, xx) = as_independent(&factor_terms(&dependent)?, x, false)?;
        a = aa;
        big_x = xx;
    }
    if crate::core::printing::could_extract_minus_sign(&big_x) {
        a = neg(&a)?;
        big_x = neg(&big_x)?;
    }
    Ok((a, b, big_x))
}

fn real(e: &Basic) -> Option<bool> {
    if let Some(r) = is(e, Predicate::real) {
        return Some(r);
    }
    let settings = crate::settings::settings();
    if !settings.numerical_checks || !crate::core::assumptions::is_number(e) {
        return None;
    }
    crate::numeric::reliable_complex(e, settings.precision, &[]).map(|(_, im)| im == 0.0)
}

/// The solutions of `a*log(b*X + c) + d*X + f = 0` (`_lambert`).
fn lambert(eq: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let mut eq: Basic = mexpand(&expand_log(eq, true)?)?;
    let Some(mut mainlog) = mostfunc(&eq, Kind::Log, x) else {
        return Ok(vec![]);
    };
    let mut other: Basic = subs(&eq, &mainlog, &Basic::zero())?;
    if neg(&other)?.is_function(Function::Log) {
        eq = subs(&sub(&eq, &other)?, &mainlog, &mainlog.args()[0])?;
        let inner: Basic = mainlog.args()[0].clone();
        if !inner.is_function(Function::Log) {
            return Ok(vec![]);
        }
        mainlog = inner;
        other = neg(&neg(&other)?.args()[0])?;
        eq = add(&[eq, other.clone()])?;
    }
    if !other.free_symbols().contains(x) {
        return Ok(vec![]);
    }
    let (d, f, x2) = linab(&other, x)?;
    let logterm: Basic =
        crate::simplify::radsimp::collect(&sub(&eq, &other)?, std::slice::from_ref(&mainlog), None, Some(false))?;
    let Some(a) = crate::core::expr::as_coefficient(&logterm, &mainlog)? else {
        return Ok(vec![]);
    };
    if a.free_symbols().contains(x) {
        return Ok(vec![]);
    }
    let logarg: Basic = mainlog.args()[0].clone();
    let (b, c, x1) = linab(&logarg, x)?;
    if x1 != x2 {
        return Ok(vec![]);
    }
    let u: Basic = Dummy::new("rhs", &[]).0;
    let Some(inverses) = super::solvers::solve_equation(&sub(&x1, &u)?, x)? else {
        return Ok(vec![]);
    };
    let exponent: Basic = div(
        &div(&sub(&mul(&[c.clone(), d.clone()])?, &mul(&[b.clone(), f])?)?, &a)?,
        &b,
    )?;
    let (num, den) = crate::core::expr::as_numer_denom(&exponent)?;
    let (p, den) = crate::core::expr::as_coeff_mul(&den);
    let e: Basic = exp(&div(&num, &den)?)?;
    let t: Basic = Dummy::new("t", &[]).0;
    let Some(pn) = p
        .as_rational()
        .filter(|v| crate::core::expr::is_integer_class(&p) && *v >= &crate::numbers::Rat::ONE)
        .cloned()
    else {
        return Ok(vec![]);
    };
    let tees: Vec<Basic> = if pn.is_one() {
        vec![e]
    } else {
        match crate::polys::polyroots::ex_roots(&sub(&crate::core::power::pow(&t, &p)?, &e)?, &t) {
            Ok(r) => r,
            Err(crate::error::PolyError::NotImplemented(_)) => return Ok(vec![]),
            Err(err) => return Err(err),
        }
    };
    let mut solutions: Vec<Basic> = Vec::new();
    for root in tees {
        let argument: Basic = mul(&[div(&d, &mul(&[a.clone(), b.clone()])?)?, root])?;
        for k in [-1i64, 0] {
            let w: Basic = if k != 0 {
                lambertw(&argument, Some(&Basic::int(k)))?
            } else {
                lambertw(&argument, None)?
            };
            if k != 0 && real(&w) != Some(true) {
                continue;
            }
            let rhs: Basic = add(&[neg(&div(&c, &b)?)?, mul(&[div(&a, &d)?, w])?])?;
            for xu in &inverses {
                solutions.push(subs(xu, &u, &rhs)?);
            }
        }
    }
    Ok(solutions)
}

fn lambert_generators(e: &Basic, x: &Basic) -> Vec<Basic> {
    super::solvers::generators(e, x)
        .into_iter()
        .filter(|g| g.is_function(Function::Exp) || g.is_function(Function::Log) || (g.is_pow() && g.args()[1].has(x)))
        .collect()
}

fn uniq(values: Vec<Basic>) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    for v in values {
        if !result.contains(&v) {
            result.push(v);
        }
    }
    result
}

fn solve_even_degree_expr(expr: &Basic, t: &Basic, x: &Basic, depth: usize) -> O {
    let at = |s: i64| -> R {
        let value: Basic = mul(&[Basic::int(s), x.clone()])?;
        Ok(expr.xreplace(&HashMap::from([(t.clone(), value)])))
    };
    let (negative, positive) = (at(-1)?, at(1)?);
    let mut solutions: Vec<Basic> = solve_lambert_inner(&negative, x, depth + 1)?.unwrap_or_default();
    if positive != negative {
        if let Some(more) = solve_lambert_inner(&positive, x, depth + 1)? {
            solutions.extend(more);
        }
    }
    Ok(if solutions.is_empty() {
        None
    } else {
        Some(uniq(solutions))
    })
}

fn even_powers(lhs: &Basic, x: &Basic, t: &Basic) -> R {
    let query =
        |a: &Basic| a.is_pow() && a.args()[0] == *x && crate::core::expr::is_true(&a.args()[1], Predicate::even);
    let value = |a: &Basic| crate::core::power::pow(t, &a.args()[1]);
    crate::core::function::replace_where(lhs, &query, &value)
}

/// The solutions of `f = 0` in terms of the Lambert W function, `None`
/// when `f` is not of a Lambert form (`solve_lambert`).
pub fn solve_lambert(f: &Basic, x: &Basic) -> O {
    solve_lambert_inner(f, x, 0)
}

fn assumptions_of(x: &Basic) -> Vec<(Predicate, bool)> {
    match x.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.assumptions.iter().map(|(p, v)| (*p, *v)).collect(),
        _ => vec![],
    }
}

fn solve_lambert_inner(f: &Basic, x: &Basic, depth: usize) -> O {
    use crate::simplify::radsimp::collect;
    if depth > 4 {
        return Ok(None);
    }
    let (nrhs, mut lhs) = as_independent(f, x, true)?;
    let mut rhs: Basic = neg(&nrhs)?;
    if lambert_generators(f, x).is_empty() {
        return Ok(None);
    }
    if lhs.is_add() || lhs.is_mul() {
        let t: Basic = Dummy::new("t", &assumptions_of(x)).0;
        lhs = even_powers(&lhs, x, &t)?;
        if lhs.is_add() && lhs.has(&t) {
            let t_independent: Basic = subs(&lhs, &t, &Basic::zero())?;
            let t_term: Basic = sub(&lhs, &t_independent)?;
            let rhs_: Basic = sub(&rhs, &t_independent)?;
            if !t_term.is_add() && rhs_ != Basic::zero() && is(&t_term, Predicate::finite) != Some(false) {
                let equation: Basic = expand_log(&sub(&log(&t_term)?, &log(&rhs_)?)?, false)?;
                return solve_even_degree_expr(&equation, &t, x, depth);
            }
        } else if lhs.is_mul() && rhs != Basic::zero() {
            lhs = expand_log(&log(&lhs)?, true)?;
            rhs = log(&rhs)?;
            if lhs.has(&t) && lhs.is_add() {
                return solve_even_degree_expr(&sub(&lhs, &rhs)?, &t, x, depth);
            }
        }
        lhs = lhs.xreplace(&HashMap::from([(t, x.clone())]));
    }
    lhs = crate::simplify::powsimp::powsimp_default(&crate::polys::polytools::factor(&lhs, &[], true, true)?)?;
    let r: Basic = Dummy::new("r", &[]).0;
    let (i, inverted) = super::solvers::invert(&sub(&lhs, &r)?, x, &Basic::zero())?;
    lhs = inverted;
    rhs = i.xreplace(&HashMap::from([(r, rhs)]));
    let mut solutions: Vec<Basic> = Vec::new();
    if let Some(mainlog) = mostfunc(&lhs, Kind::Log, x) {
        if lhs.is_mul() && rhs != Basic::zero() {
            solutions = lambert(&sub(&log(&lhs)?, &log(&rhs)?)?, x)?;
        } else if lhs.is_add() {
            let other: Basic = subs(&lhs, &mainlog, &Basic::zero())?;
            let has_power: bool = other
                .atoms(Some(&|n: &Basic| n.is_pow()))
                .iter()
                .any(|a| a.free_symbols().contains(x));
            if other != Basic::zero() && !other.is_add() && has_power {
                let difference: Basic = if rhs == Basic::zero() {
                    sub(&log(&other)?, &log(&sub(&other, &lhs)?)?)?
                } else {
                    sub(&log(&sub(&lhs, &other)?)?, &log(&sub(&rhs, &other)?)?)?
                };
                solutions = lambert(&expand_log(&difference, false)?, x)?;
            } else {
                solutions = lambert(&sub(&lhs, &rhs)?, x)?;
            }
        }
    }
    if solutions.is_empty() {
        if let Some(mainexp) = mostfunc(&lhs, Kind::Exp, x) {
            lhs = collect(&lhs, std::slice::from_ref(&mainexp), None, Some(false))?;
            if lhs.is_mul() && rhs != Basic::zero() {
                solutions = lambert(&expand_log(&sub(&log(&lhs)?, &log(&rhs)?)?, false)?, x)?;
            } else if lhs.is_add() {
                let other: Basic = subs(&lhs, &mainexp, &Basic::zero())?;
                let mut mainterm: Basic = sub(&lhs, &other)?;
                rhs = sub(&rhs, &other)?;
                if crate::core::printing::could_extract_minus_sign(&mainterm)
                    && crate::core::printing::could_extract_minus_sign(&rhs)
                {
                    mainterm = neg(&mainterm)?;
                    rhs = neg(&rhs)?;
                }
                solutions = lambert(&expand_log(&sub(&log(&mainterm)?, &log(&rhs)?)?, false)?, x)?;
            }
        }
    }
    if solutions.is_empty() {
        if let Some(mainpow) = mostfunc(&lhs, Kind::Pow, x) {
            if mainpow.is_pow() && mainpow.args()[1].free_symbols().contains(x) {
                lhs = collect(&lhs, std::slice::from_ref(&mainpow), None, Some(false))?;
                if lhs.is_mul() && rhs != Basic::zero() {
                    solutions = lambert(&expand_log(&sub(&log(&lhs)?, &log(&rhs)?)?, false)?, x)?;
                } else if lhs.is_add() {
                    let other: Basic = subs(&lhs, &mainpow, &Basic::zero())?;
                    let mainterm: Basic = sub(&lhs, &other)?;
                    rhs = sub(&rhs, &other)?;
                    solutions = lambert(&expand_log(&sub(&log(&mainterm)?, &log(&rhs)?)?, false)?, x)?;
                }
            }
        }
    }
    if solutions.is_empty() {
        return Ok(None);
    }
    let mut result: Vec<Basic> = uniq(solutions);
    result.sort_by_cached_key(crate::core::sorting::sort_key);
    Ok(Some(result))
}
