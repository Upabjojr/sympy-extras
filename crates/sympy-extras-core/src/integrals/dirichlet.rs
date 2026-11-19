//! Port of `sympy_extras/integrals/dirichlet.py`: integrals of
//! trigonometric sums over powers of the variable (Dirichlet, Frullani and
//! Borwein integrals) on the half lines and the line: the sum vanishing to
//! order `n` at `0`, `n - 1` integrations by parts bring the integral to
//! `pi/2` per sine and `-log(w)` per cosine.

use crate::assumptions::ask::ask;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_independent, terms_of};
use crate::error::PolyResult;
use crate::integrals::conditions::ConditionalValue;

type R = PolyResult<Basic>;

/// `sin` or `cos`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Sin,
    Cos,
}

/// A constant and the coefficients of `sin(w*x)` and `cos(w*x)`
/// (`TrigonometricSum`).
pub type TrigonometricSum = (Basic, Vec<((Basic, Kind), Basic)>);

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn trig(f: Function, a: &Basic) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), vec![a.clone()]), vec![a.clone()])
}
fn has_fn(e: &Basic, fs: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(f) if fs.contains(f)))
}

fn add_term(terms: &mut Vec<((Basic, Kind), Basic)>, w: &Basic, kind: Kind, c: &Basic) -> PolyResult<()> {
    match terms.iter_mut().find(|((v, k), _)| v == w && *k == kind) {
        Some(entry) => entry.1 = a2(&entry.1, c)?,
        None => terms.push(((w.clone(), kind), a2(&int(0), c)?)),
    }
    Ok(())
}

/// `g` as a constant plus a sum of sines and cosines of `w*x + phase`
/// with `w > 0` (`trigonometric_sum`); `None` otherwise.
pub fn trigonometric_sum(g: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<TrigonometricSum>> {
    let mut g: Basic = g.clone();
    if has_fn(&g, &[Function::Exp]) {
        g = crate::core::rewriting::rewrite(&g, &[], "cos", &crate::core::rewriting::RewriteHints::default())?;
    }
    let mut expanded: Basic = g;
    for _ in 0..8 {
        expanded = crate::core::function::expand_method(&crate::simplify::fu::tr8(&expanded, true)?)?;
        let mut done: bool = true;
        for term in terms_of(&expanded) {
            let (_, rest) = as_independent(&term, x, false)?;
            if rest.is_pow() || rest.is_mul() {
                done = false;
                break;
            }
        }
        if done {
            break;
        }
    }
    let mut constant: Basic = int(0);
    let mut terms: Vec<((Basic, Kind), Basic)> = Vec::new();
    for term in terms_of(&expanded) {
        let (mut c, function) = as_independent(&term, x, false)?;
        if function == int(1) {
            constant = a2(&constant, &c)?;
            continue;
        }
        let kind: Kind = match function.head() {
            Head::Function(Function::Sin) => Kind::Sin,
            Head::Function(Function::Cos) => Kind::Cos,
            _ => return Ok(None),
        };
        let argument: &Basic = &function.args()[0];
        let polynomial: bool = crate::core::expr::polynomial_like(
            argument,
            std::slice::from_ref(x),
            crate::core::expr::PolynomialKind::Polynomial,
        ) == Some(true);
        if !polynomial || !of_degree_one(argument, x)? {
            return Ok(None);
        }
        let mut w: Basic = crate::core::expr::coeff(argument, x)?;
        let mut phase: Basic = crate::core::subs::subs(argument, x, &int(0))?;
        if w.has(x) || phase.has(x) {
            return Ok(None);
        }
        if ask(&crate::core::relational::gt(&w, &int(0))?, assumptions, None)? != Some(true) {
            if ask(&crate::core::relational::lt(&w, &int(0))?, assumptions, None)? != Some(true) {
                return Ok(None);
            }
            w = neg(&w)?;
            phase = neg(&phase)?;
            if kind == Kind::Sin {
                c = neg(&c)?;
            }
        }
        if phase == int(0) {
            add_term(&mut terms, &w, kind, &c)?;
        } else if kind == Kind::Sin {
            add_term(&mut terms, &w, Kind::Sin, &m2(&c, &trig(Function::Cos, &phase)?)?)?;
            add_term(&mut terms, &w, Kind::Cos, &m2(&c, &trig(Function::Sin, &phase)?)?)?;
        } else {
            add_term(&mut terms, &w, Kind::Cos, &m2(&c, &trig(Function::Cos, &phase)?)?)?;
            add_term(
                &mut terms,
                &w,
                Kind::Sin,
                &m2(&neg(&c)?, &trig(Function::Sin, &phase)?)?,
            )?;
        }
    }
    terms.retain(|(_, v)| *v != int(0));
    Ok(Some((constant, terms)))
}

fn of_degree_one(p: &Basic, x: &Basic) -> PolyResult<bool> {
    let d: Basic = crate::core::function::expand_method(&crate::core::function::derivative(p, x)?)?;
    Ok(!d.has(x) && d != int(0))
}

fn vanishes_to_order(total: &TrigonometricSum, n: i64) -> PolyResult<bool> {
    let (constant, terms) = total;
    for m in 0..n {
        let mut coefficient: Basic = if m == 0 { constant.clone() } else { int(0) };
        for ((w, kind), c) in terms {
            if (m % 2 == 0 && *kind == Kind::Cos) || (m % 2 == 1 && *kind == Kind::Sin) {
                coefficient = a2(&coefficient, &m2(c, &crate::core::power::pow(w, &int(m))?)?)?;
            }
        }
        let simplified: Basic =
            crate::simplify::simplify::simplify(&coefficient, &crate::simplify::simplify::Options::default())?;
        if simplified != int(0) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn derivative(total: &TrigonometricSum) -> PolyResult<TrigonometricSum> {
    let mut derived: Vec<((Basic, Kind), Basic)> = Vec::new();
    for ((w, kind), c) in &total.1 {
        if *kind == Kind::Sin {
            add_term(&mut derived, w, Kind::Cos, &m2(c, w)?)?;
        } else {
            // derived.get(..., 0) - c * w
            match derived.iter_mut().find(|((v, k), _)| v == w && *k == Kind::Sin) {
                Some(entry) => entry.1 = crate::core::add::sub(&entry.1, &m2(c, w)?)?,
                None => derived.push(((w.clone(), Kind::Sin), crate::core::add::sub(&int(0), &m2(c, w)?)?)),
            }
        }
    }
    Ok((int(0), derived))
}

fn reflected(total: &TrigonometricSum, n: i64) -> PolyResult<TrigonometricSum> {
    let (constant, terms) = total;
    let factor: Basic = if n % 2 == 0 { int(1) } else { int(-1) };
    let mut out: Vec<((Basic, Kind), Basic)> = Vec::new();
    for ((w, kind), c) in terms {
        let value: Basic = m2(&factor, &if *kind == Kind::Sin { neg(c)? } else { c.clone() })?;
        out.push(((w.clone(), *kind), value));
    }
    Ok((m2(&factor, constant)?, out))
}

fn half_line(total: &TrigonometricSum, n: i64) -> PolyResult<Option<Basic>> {
    if !vanishes_to_order(total, n)? {
        return Ok(None);
    }
    if n == 1 && total.0 != int(0) {
        return Ok(None);
    }
    let mut current: TrigonometricSum = total.clone();
    for _ in 0..n - 1 {
        current = derivative(&current)?;
    }
    let mut value: Basic = int(0);
    for ((w, kind), c) in &current.1 {
        let piece: Basic = if *kind == Kind::Sin {
            crate::core::mul::div(&m2(&Basic::pi(), c)?, &int(2))?
        } else {
            m2(&neg(c)?, &crate::functions::elementary::exponential::log(w)?)?
        };
        value = a2(&value, &piece)?;
    }
    let f: Basic = crate::functions::combinatorial::factorials::factorial(&int(n - 1))?;
    Ok(Some(crate::simplify::simplify::logcombine(
        &crate::core::mul::div(&value, &f)?,
        true,
    )?))
}

/// `Integral(f, (x, a, b))` for a trigonometric sum over `x**n` on
/// `(0, oo)`, `(-oo, 0)` or `(-oo, oo)` (`dirichlet_integral`).
pub fn dirichlet_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    let (zero, oo, moo) = (int(0), Basic::oo(), Basic::negative_oo());
    let ranges: bool = (*a == zero && *b == oo) || (*a == moo && *b == zero) || (*a == moo && *b == oo);
    if !ranges || !has_fn(f, &[Function::Sin, Function::Cos, Function::Exp]) {
        return Ok(None);
    }
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&crate::polys::polytools::cancel(f, &[])?)?;
    let Some((scale, n)) = crate::integrals::mellin::monomial(&denominator, x)? else {
        return Ok(None);
    };
    if !n.is_integer_value() {
        return Ok(None);
    }
    let n: i64 = i64::try_from(&crate::numbers::numer(n.as_rational().expect("an integer"))).expect("small");
    if n < 1 {
        return Ok(None);
    }
    let Some(total) = trigonometric_sum(&crate::core::mul::div(&numerator, &scale)?, x, assumptions)? else {
        return Ok(None);
    };
    let mut value: Basic = int(0);
    if *b == oo {
        let Some(right) = half_line(&total, n)? else {
            return Ok(None);
        };
        value = a2(&value, &right)?;
    }
    if *a == moo {
        let Some(left) = half_line(&reflected(&total, n)?, n)? else {
            return Ok(None);
        };
        value = a2(&value, &left)?;
    }
    let expanded: Basic = crate::core::function::expand_with(
        &value,
        true,
        crate::core::function::Hints {
            log: true,
            ..crate::core::function::Hints::DEFAULT
        },
    )?;
    Ok(Some(ConditionalValue::new(expanded, crate::core::logic::boolean(true))))
}
