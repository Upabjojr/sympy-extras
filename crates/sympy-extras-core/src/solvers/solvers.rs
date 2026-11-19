//! Port of `sympy_extras/solvers/solvers.py` (parts of SymPy's
//! `solvers.py`): [`solve_equation`], the solutions of one equation in one
//! unknown as SymPy's `solve(f, x)` lists them (a product factor by
//! factor, the roots of a polynomial numerator, a polynomial in one
//! generator, radicals of one base removed by a resultant, the
//! transcendental equations by inversion, factoring, `logcombine` and the
//! Lambert forms of `bivariate`), every candidate checked as `checksol`
//! does; and [`solve_system`], polynomial systems by substitution after a
//! lexicographic Gröbner basis. `None` where SymPy raises
//! `NotImplementedError`.

#![allow(clippy::mutable_key_type, clippy::type_complexity)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_independent, is, is_true};
use crate::core::sorting::sort_key;
use crate::core::symbol::Dummy;
use crate::error::PolyResult;
use crate::functions::elementary::exponential::{exp, log};
use crate::numbers::*;

type R = PolyResult<Basic>;
type O = PolyResult<Option<Vec<Basic>>>;

/// The deepest recursion of the solver (`_MAX_DEPTH`).
const MAX_DEPTH: usize = 12;

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
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn is_one(e: &Basic) -> bool {
    *e == Basic::one()
}
fn integer_of(e: &Basic) -> Option<i64> {
    if !crate::core::expr::is_integer_class(e) {
        return None;
    }
    e.as_rational().and_then(|v| i64::try_from(&numer(v)).ok())
}
fn is_application(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(_) | Head::Undefined(_))
}
fn has_application(e: &Basic) -> bool {
    e.preorder().iter().any(is_application)
}
fn count_function(e: &Basic, f: Function) -> usize {
    e.preorder().iter().filter(|n| n.is_function(f)).count()
}
fn count_applications(e: &Basic) -> usize {
    e.preorder().iter().filter(|n| is_application(n)).count()
}

/// `(c, e)` with `self == c*x**e`, `c` free of `x`, or `(self, 0)`
/// (`as_coeff_exponent`).
pub fn as_coeff_exponent(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let s: Basic = crate::simplify::radsimp::collect(e, std::slice::from_ref(x), None, Some(false))?;
    let (c, p): (Basic, Vec<Basic>) = if s.is_mul() {
        let free: Vec<Basic> = s.args().iter().filter(|f| !f.has(x)).cloned().collect();
        let p: Vec<Basic> = s.args().iter().filter(|f| f.has(x)).cloned().collect();
        let c: Basic = match free.len() {
            0 => Basic::one(),
            1 => free[0].clone(),
            _ => crate::core::mul::raw_mul(free),
        };
        (c, p)
    } else if s.has(x) {
        (Basic::one(), vec![s.clone()])
    } else {
        (s.clone(), vec![])
    };
    if p.len() == 1 {
        let (b, ex) = crate::core::expr::as_base_exp(&p[0])?;
        if b == *x {
            return Ok((c, ex));
        }
    }
    Ok((s, Basic::zero()))
}

// -- the checks

fn numeric_truth(value: &Basic) -> Option<bool> {
    let settings = crate::settings::settings();
    if !settings.numerical_checks {
        return None;
    }
    let digits: usize = settings.precision;
    let (re, im) = crate::numeric::reliable_complex(value, digits, &[])?;
    let size: f64 = re.hypot(im);
    if !size.is_finite() {
        return None;
    }
    if size > 10f64.powi(-((digits * 2 / 3) as i32)) {
        return Some(false);
    }
    Some(true)
}

/// Whether `x = v` solves `f = 0`: `Some(true)`, `Some(false)` when
/// provably not, `None` when undecided (`checksol`).
pub fn checksol(f: &Basic, x: &Basic, v: &Basic) -> PolyResult<Option<bool>> {
    let value: Basic = match subs(f, x, v) {
        Ok(value) => value,
        Err(crate::error::PolyError::Type(_))
        | Err(crate::error::PolyError::Value(_))
        | Err(crate::error::PolyError::ZeroDivision(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if value.free_symbols().contains(x) {
        return Ok(None);
    }
    zero_truth(&value)
}

/// Whether an expression is zero: structurally or after `cancel`, not
/// zero for positive symbols (`posify`), or by a reliable value
/// (`zero_truth`).
pub fn zero_truth(value: &Basic) -> PolyResult<Option<bool>> {
    if *value == Basic::zero() {
        return Ok(Some(true));
    }
    if is_true(value, Predicate::infinite) {
        return Ok(Some(false));
    }
    if crate::core::assumptions::is_number(value) {
        if let Some(z) = is(value, Predicate::zero) {
            return Ok(Some(z));
        }
        return Ok(numeric_truth(value));
    }
    let cancelled: Option<Basic> =
        crate::timeout::attempt_within(crate::settings::settings().timeout, || cancel(value))?;
    if cancelled.is_some_and(|c| c == Basic::zero()) {
        return Ok(Some(true));
    }
    let (positive, _) = crate::simplify::simplify::posify(value)?;
    if is(&positive, Predicate::zero) == Some(false) {
        return Ok(Some(false));
    }
    Ok(None)
}

fn zeroes_denominator(denominator: &Basic, x: &Basic, v: &Basic) -> PolyResult<bool> {
    if !denominator.has(x) {
        return Ok(false);
    }
    Ok(checksol(denominator, x, v)? == Some(true))
}

// -- polynomials

pub(crate) fn coefficients(e: &Basic, v: &Basic) -> PolyResult<Option<Vec<(i64, Basic)>>> {
    let ex: Basic = expand(e)?;
    let terms: Vec<Basic> = if ex.is_add() { ex.args().to_vec() } else { vec![ex] };
    let mut gathered: Vec<(i64, Vec<Basic>)> = Vec::new();
    for term in terms {
        let (c, k) = as_coeff_exponent(&term, v)?;
        let Some(kk) = integer_of(&k).filter(|kk| *kk >= 0) else {
            return Ok(None);
        };
        if c.has(v) {
            return Ok(None);
        }
        match gathered.iter().position(|(d, _)| *d == kk) {
            Some(i) => gathered[i].1.push(c),
            None => gathered.push((kk, vec![c])),
        }
    }
    let mut result: Vec<(i64, Basic)> = Vec::new();
    for (degree, cs) in gathered {
        let coefficient: Basic = add(&cs)?;
        if coefficient != Basic::zero() {
            result.push((degree, coefficient));
        }
    }
    Ok(Some(result))
}

fn polynomial_of(coefficients: &[(i64, Basic)], x: &Basic) -> R {
    let terms: Vec<Basic> = coefficients
        .iter()
        .map(|(k, c)| mul(&[c.clone(), pow(x, &int(*k))?]))
        .collect::<PolyResult<_>>()?;
    add(&terms)
}

fn distinct(values: Vec<Basic>) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    for v in values {
        if !result.contains(&v) {
            result.push(v);
        }
    }
    result
}

pub(crate) fn polynomial_roots(e: &Basic, x: &Basic, degree: usize) -> O {
    let found = match crate::polys::polyroots::roots(e, std::slice::from_ref(x), true, true, None) {
        Ok(f) => Some(f),
        Err(err) if crate::polys::rootoftools::is_failure(&err) => None,
        Err(err) => return Err(err),
    };
    if let Some(found) = found {
        let total: usize = found.iter().map(|(_, k)| *k).sum();
        if total == degree {
            return Ok(Some(found.iter().map(|(r, _)| r.clone()).collect()));
        }
        let free = e.free_symbols();
        if free.iter().all(|c| is_true(c, Predicate::rational)) && free.iter().all(|s| s == x) {
            let listed = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
                crate::polys::complexrootof::all_roots(e, Some(x), true)
            })?;
            if let Some(listed) = listed {
                if listed.len() == degree {
                    return Ok(Some(distinct(listed)));
                }
            }
        }
        return Ok(None);
    }
    let listed: Vec<Basic> = match crate::polys::polyroots::ex_roots(&expand(e)?, x) {
        Ok(l) => l,
        Err(err) if crate::polys::rootoftools::is_failure(&err) => match roots_in_power(e, x)? {
            Some(d) => d,
            None => return Ok(None),
        },
        Err(err) => return Err(err),
    };
    if listed.len() != degree {
        return Ok(None);
    }
    Ok(Some(distinct(listed)))
}

fn roots_of_unity(n: usize) -> PolyResult<Option<Vec<Basic>>> {
    let w: Basic = Dummy::new("w", &[]).0;
    let unity: Vec<Basic> = crate::polys::polyroots::roots_list(
        &sub(&pow(&w, &int(n as i64))?, &Basic::one())?,
        std::slice::from_ref(&w),
        true,
        true,
        None,
    )?;
    Ok(if unity.len() == n { Some(unity) } else { None })
}

fn roots_in_power(e: &Basic, x: &Basic) -> O {
    let Some(cs) = coefficients(e, x)? else { return Ok(None) };
    let mut g: i64 = 0;
    for (k, _) in &cs {
        g = num_gcd(g, *k);
    }
    if g < 2 {
        return Ok(None);
    }
    let y: Basic = Dummy::new("y", &[]).0;
    let reduced: Vec<Basic> = cs
        .iter()
        .map(|(k, c)| mul(&[c.clone(), pow(&y, &int(k / g))?]))
        .collect::<PolyResult<_>>()?;
    let inner: Vec<Basic> = match crate::polys::polyroots::ex_roots(&expand(&add(&reduced)?)?, &y) {
        Ok(i) => i,
        Err(err) if crate::polys::rootoftools::is_failure(&err) => return Ok(None),
        Err(err) => return Err(err),
    };
    let Some(unity) = roots_of_unity(g as usize)? else {
        return Ok(None);
    };
    let mut result: Vec<Basic> = Vec::new();
    for value in inner {
        let principal: Basic = pow(&value, &Basic::from_rational(rat(1, g)))?;
        for u in &unity {
            result.push(mul(&[principal.clone(), u.clone()])?);
        }
    }
    Ok(Some(result))
}

fn num_gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

// -- the inversion of the outer functions

fn inverse(f: &Basic) -> Option<fn(&Basic) -> R> {
    use crate::functions::elementary::hyperbolic as h;
    use crate::functions::elementary::inverse as inv;
    use crate::functions::elementary::trigonometric as t;
    use Function as F;
    let Head::Function(g) = f.head() else { return None };
    Some(match g {
        F::Tan => inv::atan,
        F::Cot => inv::acot,
        F::Sec => inv::asec,
        F::Csc => inv::acsc,
        F::Asin => t::sin,
        F::Acos => t::cos,
        F::Atan => t::tan,
        F::Acot => t::cot,
        F::Asec => t::sec,
        F::Acsc => t::csc,
        F::Sinh => inv::asinh,
        F::Cosh => inv::acosh,
        F::Tanh => inv::atanh,
        F::Coth => inv::acoth,
        F::Sech => inv::asech,
        F::Csch => inv::acsch,
        F::Asinh => h::sinh,
        F::Acosh => h::cosh,
        F::Atanh => h::tanh,
        F::Acoth => h::coth,
        F::Asech => h::sech,
        F::Acsch => h::csch,
        _ => return None,
    })
}

/// `(r, g)` with `f = rhs` equivalent to `g = r`, the free terms and
/// factors and the outer invertible functions moved to the right
/// (`invert`, SymPy's `_invert`).
pub fn invert(f: &Basic, x: &Basic, rhs: &Basic) -> PolyResult<(Basic, Basic)> {
    let (mut lhs, mut r) = (f.clone(), rhs.clone());
    for _ in 0..100 {
        if lhs == *x || !lhs.has(x) {
            break;
        }
        if lhs.is_add() {
            let (independent, dependent) = as_independent(&lhs, x, true)?;
            if independent != Basic::zero() {
                r = sub(&r, &independent)?;
                lhs = dependent;
                continue;
            }
            break;
        }
        if lhs.is_mul() {
            let (independent, dependent) = as_independent(&lhs, x, false)?;
            if independent != Basic::one() {
                r = div(&r, &independent)?;
                lhs = dependent;
                continue;
            }
            break;
        }
        if lhs.is_pow() {
            let (base, e) = (lhs.args()[0].clone(), lhs.args()[1].clone());
            if !e.has(x) {
                if crate::core::expr::is_integer_class(&e) && r != Basic::zero() {
                    break;
                }
                r = pow(&r, &div(&Basic::one(), &e)?)?;
                lhs = base;
                continue;
            }
            if !base.has(x) {
                r = div(&log(&r)?, &log(&base)?)?;
                lhs = e;
                continue;
            }
            break;
        }
        if lhs.is_function(Function::Exp) {
            r = log(&r)?;
            lhs = lhs.args()[0].clone();
            continue;
        }
        if lhs.is_function(Function::Log) {
            r = exp(&r)?;
            lhs = lhs.args()[0].clone();
            continue;
        }
        if lhs.is_function(Function::LambertW) && lhs.args().len() == 1 {
            r = mul(&[r.clone(), exp(&r)?])?;
            lhs = lhs.args()[0].clone();
            continue;
        }
        if let Some(inv) = inverse(&lhs).filter(|_| lhs.args().len() == 1) {
            r = inv(&r)?;
            lhs = lhs.args()[0].clone();
            continue;
        }
        break;
    }
    Ok((r, lhs))
}

// -- generators

fn as_base_q(g: &Basic) -> PolyResult<(Basic, i64)> {
    if g.is_function(Function::Exp) {
        let (c, rest) = crate::core::expr::as_coeff_mul(&g.args()[0]);
        if c.is_rational_class() && c != Basic::one() {
            return Ok((
                exp(&rest)?,
                i64::try_from(&denom(c.as_rational().expect("a rational"))).unwrap_or(1),
            ));
        }
        return Ok((g.clone(), 1));
    }
    if g.is_pow() {
        let (base, e) = (g.args()[0].clone(), g.args()[1].clone());
        if e.is_rational_class() {
            return Ok((
                base,
                i64::try_from(&denom(e.as_rational().expect("a rational"))).unwrap_or(1),
            ));
        }
        let (c, rest) = crate::core::expr::as_coeff_mul(&e);
        if c.is_rational_class() && c != Basic::one() {
            return Ok((
                pow(&base, &rest)?,
                i64::try_from(&denom(c.as_rational().expect("a rational"))).unwrap_or(1),
            ));
        }
    }
    Ok((g.clone(), 1))
}

/// The generators containing `x` of `e` read as a polynomial
/// (`_generators`).
pub fn generators(e: &Basic, x: &Basic) -> Vec<Basic> {
    fn visit(a: &Basic, x: &Basic, found: &mut Vec<Basic>) {
        if !a.has(x) {
            return;
        }
        if a == x {
            if !found.contains(x) {
                found.push(x.clone());
            }
            return;
        }
        if a.is_add() || a.is_mul() {
            for b in a.args() {
                visit(b, x, found);
            }
            return;
        }
        if a.is_pow() && crate::core::expr::is_integer_class(&a.args()[1]) {
            visit(&a.args()[0], x, found);
            return;
        }
        if !matches!(a.head(), Head::Tuple) && !crate::core::logic::is_boolean(a) && !found.contains(a) {
            found.push(a.clone());
        }
    }
    let mut found: Vec<Basic> = Vec::new();
    visit(e, x, &mut found);
    found
}

fn exponent_in(g: &Basic, base: &Basic, q: i64) -> PolyResult<Option<i64>> {
    if base.is_function(Function::Exp) && g.is_function(Function::Exp) {
        let r: Basic = cancel(&div(&g.args()[0], &base.args()[0])?)?;
        let k: Basic = mul(&[r.clone(), int(q)])?;
        return Ok(if r.is_rational_class() { integer_of(&k) } else { None });
    }
    let (b, e) = crate::core::expr::as_base_exp(g)?;
    let (bb, be) = crate::core::expr::as_base_exp(base)?;
    if b != bb {
        return Ok(None);
    }
    let r: Basic = cancel(&div(&e, &be)?)?;
    if r.is_rational_class() {
        return Ok(integer_of(&mul(&[r, int(q)])?));
    }
    Ok(None)
}

fn by_generator(numerator: &Basic, x: &Basic, depth: usize) -> O {
    let gens: Vec<Basic> = generators(numerator, x);
    if gens.is_empty() || gens.as_slice() == std::slice::from_ref(x) {
        return Ok(None);
    }
    let pairs: Vec<(Basic, i64)> = gens.iter().map(as_base_q).collect::<PolyResult<_>>()?;
    let mut bases: Vec<Basic> = Vec::new();
    for (b, _) in &pairs {
        if !bases.contains(b) {
            bases.push(b.clone());
        }
    }
    if gens.len() == 1 && !(is_application(&gens[0]) || gens[0].is_pow() || gens[0].is_symbol()) {
        return Ok(Some(vec![]));
    }
    let t: Basic = Dummy::new("t", &[]).0;
    let (replaced, generator): (Basic, Basic);
    if bases.len() == 1 && bases[0] != *x {
        let base: Basic = bases[0].clone();
        let mut q: i64 = 1;
        for (_, qi) in &pairs {
            q = q / num_gcd(q, *qi) * qi;
        }
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for g in &gens {
            let Some(k) = exponent_in(g, &base, q)? else {
                return Ok(None);
            };
            rule.insert(g.clone(), pow(&t, &int(k))?);
        }
        replaced = numerator.xreplace(&rule);
        if replaced.has(x) {
            return Ok(None);
        }
        generator = if q == 1 {
            base
        } else if base.is_function(Function::Exp) {
            exp(&div(&base.args()[0], &int(q))?)?
        } else {
            pow(&base, &Basic::from_rational(rat(1, q)))?
        };
    } else {
        let mut functions: Vec<Basic> = gens.iter().filter(|g| is_application(g)).cloned().collect();
        functions.sort_by_cached_key(|g| (count_applications(g), sort_key(g)));
        let Some(first) = functions.first().cloned() else {
            return Ok(None);
        };
        generator = first;
        replaced = subs(numerator, &generator, &t)?;
        if replaced.has(x) {
            return Ok(None);
        }
    }
    let Some(values) = solve_inner(&replaced, &t, depth + 1)? else {
        return Ok(None);
    };
    let mut result: Vec<Basic> = Vec::new();
    for value in values {
        let Some(found) = solve_inner(&sub(&generator, &value)?, x, depth + 1)? else {
            return Ok(None);
        };
        result.extend(found);
    }
    Ok(Some(result))
}

// -- radicals

fn is_rational_function(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}

fn fraction_of_together(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::simplify::radsimp::fraction(&crate::polys::rationaltools::together(e, false, true)?)
}

fn unrad(numerator: &Basic, x: &Basic, depth: usize) -> O {
    let gens: Vec<Basic> = generators(numerator, x).into_iter().filter(|g| g != x).collect();
    if gens.is_empty() {
        return Ok(None);
    }
    let mut bases: Vec<Basic> = Vec::new();
    let mut q: i64 = 1;
    for g in &gens {
        if !g.is_pow() || !g.args()[1].is_rational_class() || crate::core::expr::is_integer_class(&g.args()[1]) {
            return Ok(None);
        }
        let b: &Basic = &g.args()[0];
        if !is_rational_function(b, x) {
            return Ok(None);
        }
        if !bases.contains(b) {
            bases.push(b.clone());
        }
        let gq: i64 = i64::try_from(&denom(g.args()[1].as_rational().expect("a rational"))).unwrap_or(1);
        q = q / num_gcd(q, gq) * gq;
    }
    if bases.len() != 1 {
        return Ok(None);
    }
    let base: Basic = bases[0].clone();
    let t: Basic = Dummy::new("t", &[]).0;
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for g in &gens {
        let k: Basic = mul(&[g.args()[1].clone(), int(q)])?;
        if integer_of(&k).is_none() {
            return Ok(None);
        }
        rule.insert(g.clone(), pow(&t, &k)?);
    }
    let polynomial: Basic = fraction_of_together(&numerator.xreplace(&rule))?.0;
    let (base_numerator, base_denominator) = fraction_of_together(&base)?;
    if crate::core::expr::polynomial_like(
        &polynomial,
        &[t.clone(), x.clone()],
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return Ok(None);
    }
    let defining: Basic = sub(&mul(&[base_denominator.clone(), pow(&t, &int(q))?])?, &base_numerator)?;
    let mut eliminated: Option<Basic> = None;
    if q == 2 {
        eliminated = square_eliminated(&polynomial, &t, &base_numerator, &base_denominator)?;
    }
    if eliminated.is_none() {
        eliminated = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            crate::polys::polytools::resultant_expr(
                &expand(&polynomial)?,
                &expand(&defining)?,
                std::slice::from_ref(&t),
            )
        })?;
    }
    let Some(eliminated) = eliminated else { return Ok(None) };
    if !eliminated.has(x) {
        return Ok(if eliminated != Basic::zero() {
            Some(vec![])
        } else {
            None
        });
    }
    let Some(candidates) = solve_inner(&eliminated, x, depth + 1)? else {
        return Ok(None);
    };
    let free = eliminated.free_symbols();
    if !(free.len() == 1 && free.contains(x)) {
        return Ok(Some(candidates));
    }
    Ok(Some(checked_by_roots(numerator, x, candidates, &eliminated)?))
}

fn value_of(e: &Basic) -> Option<(f64, f64)> {
    crate::numeric::reliable_complex(e, 30, &[])
}

fn checked_by_roots(
    numerator: &Basic,
    x: &Basic,
    candidates: Vec<Basic>,
    eliminated: &Basic,
) -> PolyResult<Vec<Basic>> {
    let Some(proxies) = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::polys::complexrootof::all_roots(eliminated, Some(x), false)
    })?
    else {
        return Ok(candidates);
    };
    let values: Vec<Option<(f64, f64)>> = proxies.iter().map(value_of).collect();
    let mut kept: Vec<Basic> = Vec::new();
    for c in candidates {
        if checksol(numerator, x, &c)?.is_some() || !crate::core::assumptions::is_number(&c) {
            kept.push(c);
            continue;
        }
        let Some((vr, vi)) = value_of(&c) else {
            kept.push(c);
            continue;
        };
        let size: f64 = vr.hypot(vi).max(1.0);
        let near: Vec<&Basic> = proxies
            .iter()
            .zip(&values)
            .filter(|(_, w)| w.is_some_and(|(wr, wi)| (wr - vr).hypot(wi - vi) < 1e-12 * size))
            .map(|(p, _)| p)
            .collect();
        if near.len() == 1 && checksol(numerator, x, near[0])? == Some(false) {
            continue;
        }
        kept.push(c);
    }
    Ok(kept)
}

fn square_eliminated(polynomial: &Basic, t: &Basic, n: &Basic, d: &Basic) -> PolyResult<Option<Basic>> {
    let Some(cs) = coefficients(polynomial, t)? else {
        return Ok(None);
    };
    let top: i64 = cs.iter().map(|(k, _)| *k).max().unwrap_or(0);
    let k: i64 = top / 2;
    let (mut even, mut odd): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for (degree, c) in &cs {
        let term: Basic = mul(&[c.clone(), pow(n, &int(degree / 2))?, pow(d, &int(k - degree / 2))?])?;
        if degree % 2 == 1 {
            odd.push(term);
        } else {
            even.push(term);
        }
    }
    let (a, b) = (add(&even)?, add(&odd)?);
    Ok(Some(expand(&sub(
        &mul(&[d.clone(), pow(&a, &int(2))?])?,
        &mul(&[n.clone(), pow(&b, &int(2))?])?,
    )?)?))
}

// -- the transcendental equations

fn tsolve(numerator: &Basic, x: &Basic, depth: usize) -> O {
    let (rhs, lhs) = invert(numerator, x, &Basic::zero())?;
    if lhs == *x {
        return Ok(Some(vec![rhs]));
    }
    if lhs.is_add() {
        let difference: Basic = sub(&lhs, &rhs)?;
        let factored: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            crate::polys::polytools::factor(
                &crate::simplify::powsimp::powdenest(&difference, false, false)?,
                &[],
                false,
                true,
            )
        })?;
        if let Some(factored) = factored {
            if factored.is_mul() && factored.args().iter().filter(|m| m.has(x)).count() > 1 {
                return solve_inner(&factored, x, depth + 1);
            }
        }
        if rhs != Basic::zero() {
            let combined: Basic = crate::simplify::simplify::logcombine(&lhs, true)?;
            if count_function(&combined, Function::Log) != count_function(&lhs, Function::Log) {
                if combined.is_function(Function::Log) {
                    return solve_inner(&sub(&combined.args()[0], &exp(&rhs)?)?, x, depth + 1);
                }
                return solve_inner(&sub(&combined, &rhs)?, x, depth + 1);
            }
        }
    } else if lhs.is_pow() {
        let (base, e) = (lhs.args()[0].clone(), lhs.args()[1].clone());
        if let Some(n) = integer_of(&e) {
            if n >= 2 && rhs != Basic::zero() && !rhs.has(x) {
                return nth_roots(&base, &rhs, n as usize, x, depth);
            }
            if sub(&lhs, &rhs)? != *numerator {
                return solve_inner(&sub(&lhs, &rhs)?, x, depth + 1);
            }
        }
        if rhs == Basic::zero() && e.has(x) && base.has(x) {
            let Some(found) = solve_inner(&base, x, depth + 1)? else {
                return Ok(None);
            };
            let mut kept: Vec<Basic> = Vec::new();
            for s in found {
                if subs(&e, x, &s)? != Basic::zero() {
                    kept.push(s);
                }
            }
            return Ok(Some(kept));
        }
        if base.has(x) && e.has(x) {
            return solve_inner(&sub(&mul(&[e, log(&base)?])?, &log(&rhs)?)?, x, depth + 1);
        }
    } else if lhs.is_function(Function::Exp)
        || lhs.is_function(Function::Log)
        || (lhs.is_function(Function::LambertW) && lhs.args().len() == 1)
    {
        return solve_inner(&sub(&lhs, &rhs)?, x, depth + 1);
    }
    if let Some(inverses) = multi_inverses(&lhs, &rhs)? {
        let argument: Basic = lhs.args()[0].clone();
        let mut result: Vec<Basic> = Vec::new();
        for value in inverses {
            let Some(found) = solve_inner(&sub(&argument, &value)?, x, depth + 1)? else {
                return Ok(None);
            };
            result.extend(found);
        }
        return Ok(Some(distinct(result)));
    }
    super::bivariate::solve_lambert(&sub(&lhs, &rhs)?, x)
}

fn multi_inverses(f: &Basic, r: &Basic) -> O {
    use crate::functions::elementary::inverse as inv;
    let pi: Basic = Basic::pi();
    if f.is_function(Function::Sin) && f.args().len() == 1 {
        let a: Basic = inv::asin(r)?;
        return Ok(Some(vec![a.clone(), sub(&pi, &a)?]));
    }
    if f.is_function(Function::Cos) && f.args().len() == 1 {
        let a: Basic = inv::acos(r)?;
        return Ok(Some(vec![a.clone(), sub(&mul(&[int(2), pi])?, &a)?]));
    }
    Ok(None)
}

fn nth_roots(base: &Basic, r: &Basic, n: usize, x: &Basic, depth: usize) -> O {
    let Some(unity) = roots_of_unity(n)? else {
        return Ok(None);
    };
    let principal: Basic = principal_root(r, n)?;
    let slope: Basic = crate::core::function::derivative(base, x)?;
    let linear: bool = slope != Basic::zero() && !slope.has(x);
    let offset: Basic = if linear {
        subs(base, x, &Basic::zero())?
    } else {
        Basic::zero()
    };
    let mut result: Vec<Basic> = Vec::new();
    for u in unity {
        if linear {
            let shifted: Basic = sub(&mul(&[principal.clone(), u])?, &offset)?;
            result.push(if !is_one(&slope) {
                div(&shifted, &slope)?
            } else {
                shifted
            });
            continue;
        }
        let Some(found) = solve_inner(&sub(base, &mul(&[principal.clone(), u])?)?, x, depth + 1)? else {
            return Ok(None);
        };
        result.extend(found);
    }
    Ok(Some(result))
}

fn principal_root(r: &Basic, n: usize) -> R {
    let e: Basic = Basic::from_rational(rat(1, n as i64));
    if !r.is_mul() {
        return pow(r, &e);
    }
    let (mut outside, mut inside): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for f in r.args() {
        if is_true(f, Predicate::positive) {
            outside.push(f.clone());
        } else {
            inside.push(f.clone());
        }
    }
    if outside.is_empty() {
        return pow(r, &e);
    }
    let powered: Vec<Basic> = outside.iter().map(|f| pow(f, &e)).collect::<PolyResult<_>>()?;
    mul(&[mul(&powered)?, pow(&mul(&inside)?, &e)?])
}

// -- the solver

fn simplified(v: &Basic) -> R {
    let found = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::simplify::simplify::simplify(v, &crate::simplify::simplify::Options::default())
    })?;
    Ok(found.unwrap_or_else(|| v.clone()))
}

/// The solutions of `f = 0` (`None`: no method), unchecked except for the
/// denominators (`_solve`).
pub(crate) fn solve_inner(f: &Basic, x: &Basic, depth: usize) -> O {
    if depth > MAX_DEPTH {
        return Ok(None);
    }
    if !f.has(x) {
        return Ok(Some(vec![]));
    }
    if f == x {
        return Ok(Some(vec![Basic::zero()]));
    }
    let negative_exponent = |m: &Basic| m.is_pow() && is_true(&m.args()[1], Predicate::negative);
    if f.is_mul() {
        let mut found: Vec<Basic> = Vec::new();
        for m in f.args() {
            if !m.has(x) {
                if is_true(m, Predicate::infinite) {
                    return Ok(Some(vec![]));
                }
                continue;
            }
            if negative_exponent(m) && !m.args()[1].has(x) {
                continue;
            }
            let Some(solved) = solve_inner(m, x, depth + 1)? else {
                return Ok(None);
            };
            found.extend(solved);
        }
        let denominators: Vec<Basic> = f
            .args()
            .iter()
            .filter(|m| negative_exponent(m) && m.args()[0].has(x))
            .map(|m| m.args()[0].clone())
            .collect();
        let mut result: Vec<Basic> = Vec::new();
        for v in distinct(found) {
            let mut zeroes: bool = false;
            for d in &denominators {
                if zeroes_denominator(d, x, &v)? {
                    zeroes = true;
                    break;
                }
            }
            if !zeroes {
                result.push(v);
            }
        }
        return Ok(Some(result));
    }
    let (numerator, denominator) = fraction_of_together(f)?;
    if !numerator.has(x) {
        return Ok(Some(vec![]));
    }
    let mut candidates: Option<Vec<Basic>> = None;
    let mut checked: bool = true;
    if let Some(cs) = coefficients(&numerator, x)? {
        if !cs.iter().any(|(_, c)| c.has(x)) {
            let degree: i64 = cs.iter().map(|(k, _)| *k).max().unwrap_or(0);
            if degree == 0 {
                return Ok(Some(vec![]));
            }
            candidates = polynomial_roots(&polynomial_of(&cs, x)?, x, degree as usize)?;
            checked = candidates.is_none();
        }
    }
    if candidates.is_none() {
        candidates = by_generator(&numerator, x, depth)?;
    }
    if candidates.is_none() {
        candidates = unrad(&numerator, x, depth)?;
    }
    if candidates.is_none() {
        candidates = tsolve(&numerator, x, depth)?;
    }
    let Some(candidates) = candidates else { return Ok(None) };
    let mut result: Vec<Basic> = Vec::new();
    for v in distinct(candidates) {
        if v.free_symbols().contains(x) {
            continue;
        }
        if zeroes_denominator(&denominator, x, &v)? {
            continue;
        }
        if checked && checksol(&numerator, x, &v)? == Some(false) {
            continue;
        }
        result.push(v);
    }
    Ok(Some(result))
}

/// The solutions of `f = 0` for `x` (SymPy's `solve(f, x)`), sorted;
/// two or fewer simplified; those contradicting the assumptions of `x`
/// dropped; `None` when no method applies (`solve_equation`).
pub fn solve_equation(f: &Basic, x: &Basic) -> O {
    let Some(mut found) = solve_inner(f, x, 0)? else {
        return Ok(None);
    };
    if found.len() <= 2 {
        found = distinct(found.iter().map(simplified).collect::<PolyResult<_>>()?);
    }
    let mut kept: Vec<Basic> = found
        .into_iter()
        .filter(|v| crate::core::assumptions::check_against(v, x) != Some(false))
        .collect();
    kept.sort_by_cached_key(sort_key);
    Ok(Some(kept))
}

// -- systems

/// A solution of a system: the values of the unknowns solved.
pub type Solution = Vec<(Basic, Basic)>;

fn unknowns_in(e: &Basic, unknowns: &[Basic]) -> Vec<Basic> {
    unknowns.iter().filter(|u| e.has(u)).cloned().collect()
}

fn choice(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Option<(usize, Basic, Vec<(i64, Basic)>)>> {
    let mut best: Option<((i64, i64, usize), usize, Basic, Vec<(i64, Basic)>)> = None;
    for (i, e) in equations.iter().enumerate() {
        let present: Vec<Basic> = unknowns_in(e, unknowns);
        for u in &present {
            let Some(cs) = coefficients(e, u)? else { continue };
            if cs.iter().any(|(_, c)| c.has(u)) {
                continue;
            }
            let degree: i64 = cs.iter().map(|(k, _)| *k).max().unwrap_or(0);
            if degree < 1 {
                continue;
            }
            let lead: &Basic = &cs.iter().find(|(k, _)| *k == degree).expect("the leading term").1;
            let constrained: i64 = i64::from(present.iter().any(|v| lead.has(v)));
            let key = (constrained, degree, present.len());
            if best.as_ref().map_or(true, |(b, ..)| key < *b) {
                best = Some((key, i, u.clone(), cs));
            }
        }
    }
    Ok(best.map(|(_, i, u, cs)| (i, u, cs)))
}

fn numerators(equations: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    for e in equations {
        let n: Basic = expand(&fraction_of_together(e)?.0)?;
        if n != Basic::zero() && !result.contains(&n) {
            result.push(n);
        }
    }
    Ok(result)
}

fn nonzero_constant(e: &Basic, unknowns: &[Basic]) -> bool {
    unknowns_in(e, unknowns).is_empty() && *e != Basic::zero()
}

fn substitute(equations: &[Basic], u: &Basic, value: &Basic) -> PolyResult<Vec<Basic>> {
    let mut replaced: Vec<Basic> = Vec::new();
    for e in equations {
        let r: Basic = subs(e, u, value)?;
        replaced.push(if has_application(&r) { r } else { cancel(&r)? });
    }
    numerators(&replaced)
}

fn solve_low(cs: &[(i64, Basic)], u: &Basic) -> O {
    let degree: i64 = cs.iter().map(|(k, _)| *k).max().unwrap_or(0);
    let get = |k: i64| {
        cs.iter()
            .find(|(d, _)| *d == k)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Basic::zero)
    };
    let sqrt = |e: &Basic| pow(e, &Basic::half());
    if degree == 1 {
        return Ok(Some(vec![cancel(&div(&neg(&get(0))?, &get(1))?)?]));
    }
    if degree == 2 {
        let (a, b, c0) = (get(2), get(1), get(0));
        if b == Basic::zero() {
            let r: Basic = sqrt(&cancel(&div(&neg(&c0)?, &a)?)?)?;
            return Ok(Some(distinct(vec![neg(&r)?, r])));
        }
        let d: Basic = sqrt(&cancel(&sub(&pow(&b, &int(2))?, &mul(&[int(4), a.clone(), c0])?)?)?)?;
        let two_a: Basic = mul(&[int(2), a])?;
        let centre: Basic = cancel(&div(&neg(&b)?, &two_a)?)?;
        let offset: Basic = div(&d, &two_a)?;
        return Ok(Some(distinct(vec![sub(&centre, &offset)?, add(&[centre, offset])?])));
    }
    solve_equation(&polynomial_of(cs, u)?, u)
}

fn branches(equations: &[Basic], unknowns: &[Basic], depth: usize, out: &mut Vec<Solution>) -> PolyResult<bool> {
    if depth > 3 * unknowns.len() + 6 || out.len() > 200 {
        return Ok(false);
    }
    let equations: Vec<Basic> = numerators(equations)?;
    if equations.iter().any(|e| nonzero_constant(e, unknowns)) {
        return Ok(true);
    }
    let equations: Vec<Basic> = equations
        .into_iter()
        .filter(|e| !unknowns_in(e, unknowns).is_empty())
        .collect();
    if equations.is_empty() {
        out.push(vec![]);
        return Ok(true);
    }
    let Some((index, u, cs)) = choice(&equations, unknowns)? else {
        return Ok(false);
    };
    let e: Basic = equations[index].clone();
    let others: Vec<Basic> = equations
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != index)
        .map(|(_, e)| e.clone())
        .collect();
    let degree: i64 = cs.iter().map(|(k, _)| *k).max().unwrap_or(0);
    let lead: Basic = cs
        .iter()
        .find(|(k, _)| *k == degree)
        .expect("the leading term")
        .1
        .clone();
    let remaining: Vec<Basic> = unknowns.iter().filter(|v| **v != u).cloned().collect();
    let values: Option<Vec<Basic>> = if degree > 2 || has_application(&lead) {
        solve_equation(&e, &u)?
    } else {
        solve_low(&cs, &u)?
    };
    let Some(values) = values else { return Ok(false) };
    let mut complete: bool = true;
    for value in values {
        let mut sub_out: Vec<Solution> = Vec::new();
        if !branches(&substitute(&others, &u, &value)?, &remaining, depth + 1, &mut sub_out)? {
            complete = false;
            continue;
        }
        for s in sub_out {
            let mut v: Basic = value.clone();
            for (w, wv) in &s {
                v = subs(&v, w, wv)?;
            }
            if !has_application(&v) {
                v = cancel(&v)?;
            }
            let mut solution: Solution = s.clone();
            match solution.iter().position(|(k, _)| *k == u) {
                Some(i) => solution[i].1 = v,
                None => solution.push((u.clone(), v)),
            }
            out.push(solution);
        }
    }
    if !unknowns_in(&lead, unknowns).is_empty() {
        let rest: Vec<(i64, Basic)> = cs.iter().filter(|(k, _)| *k != degree).cloned().collect();
        let mut system: Vec<Basic> = others.clone();
        system.push(lead.clone());
        system.push(polynomial_of(&rest, &u)?);
        let mut sub_out: Vec<Solution> = Vec::new();
        if branches(&system, unknowns, depth + 1, &mut sub_out)? {
            out.extend(sub_out);
        } else {
            complete = false;
        }
    }
    Ok(complete)
}

fn verified(solution: &Solution, equations: &[Basic]) -> PolyResult<bool> {
    let rule: HashMap<Basic, Basic> = solution.iter().cloned().collect();
    for e in equations {
        let value: Basic =
            crate::core::subs::subs_mapping(e, &rule.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<Vec<_>>())?;
        if zero_truth(&value)? == Some(false) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn groebner(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    use crate::polys::domains::QQ;
    use crate::polys::ideals::Ideal;
    use crate::polys::rings::MonomialOrder;
    if equations.is_empty()
        || equations
            .iter()
            .any(|e| e.free_symbols().iter().any(|s| !unknowns.contains(s)))
    {
        return Ok(None);
    }
    if equations.iter().any(|e| {
        crate::core::expr::polynomial_like(e, unknowns, crate::core::expr::PolynomialKind::Polynomial) != Some(true)
    }) {
        return Ok(None);
    }
    crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        let ideal = Ideal::new(equations, unknowns, QQ, MonomialOrder::Lex)?;
        let graded = ideal.groebner_basis(Some(&MonomialOrder::Grevlex))?;
        if graded.iter().any(|g| g.LM().iter().all(|e| *e == 0)) {
            return Ok(vec![Basic::one()]);
        }
        ideal.groebner_basis(None)?.iter().map(|g| ideal.to_expr(g)).collect()
    })
}

/// The solutions of the equations (each `= 0`) for the unknowns, as
/// SymPy's `solve(equations, unknowns, dict=True)`; `None` when an
/// equation cannot be solved (`solve_system`).
pub fn solve_system(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Option<Vec<Solution>>> {
    let mut system: Vec<Basic> = numerators(equations)?;
    if let Some(basis) = groebner(&system, unknowns)? {
        if basis.iter().any(|g| nonzero_constant(g, unknowns)) {
            return Ok(Some(vec![]));
        }
        system = basis;
    }
    let mut out: Vec<Solution> = Vec::new();
    if !branches(&system, unknowns, 0, &mut out)? {
        return Ok(None);
    }
    let mut result: Vec<Solution> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    for s in out {
        if !verified(&s, equations)? {
            continue;
        }
        let mut pairs: Vec<(String, String)> = s.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        pairs.sort();
        let key: String = format!("{:?}", pairs);
        if keys.contains(&key) {
            continue;
        }
        keys.push(key);
        result.push(
            unknowns
                .iter()
                .filter_map(|u| s.iter().find(|(k, _)| k == u).cloned())
                .collect(),
        );
    }
    result.sort_by_cached_key(|s| {
        unknowns
            .iter()
            .map(|u| {
                sort_key(
                    &s.iter()
                        .find(|(k, _)| k == u)
                        .map(|(_, v)| v.clone())
                        .unwrap_or_else(|| u.clone()),
                )
            })
            .collect::<Vec<_>>()
    });
    Ok(Some(result))
}
