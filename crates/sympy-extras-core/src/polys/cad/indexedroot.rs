//! Port of `sympy_extras/polys/cad/indexedroot.py`: the node
//! `IndexedRoot(f, t, k)` of the core ([`Head::IndexedRoot`]), the `k`-th
//! distinct real root from 0 of the polynomial `f` in the bound variable
//! `t`, whose coefficients may depend on other symbols. [`indexed_root`]
//! evaluates: the bound variable written as the dummy kept for its name
//! ([`bound_variable`]), a root with rational coefficients as a rational,
//! in radicals when `radical_form` allows, or as the canonical node of its
//! minimal polynomial ([`real_root`]). The numerical value of a root with
//! irrational numerical coefficients (mpmath's `polyroots` in the Python)
//! is not ported (`None`).

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Head};
use crate::core::evalf::{complex_real, real_rat, Complex};
use crate::core::symbol::{Dummy, Symbol};
use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::super::polytools::Poly;
use super::super::rootisolation::dup_count_real_roots;
use super::samplepoints::{rational, rational_real_roots, Real};

thread_local! {
    static BOUND: RefCell<HashMap<String, Basic>> = RefCell::new(HashMap::new());
    static ALGEBRAIC: RefCell<Vec<(Basic, Real)>> = const { RefCell::new(Vec::new()) };
}

/// How many real algebraic numbers of canonical roots are kept.
const KEPT: usize = 20000;

/// The bound variable of the roots in the variable called `name`: one
/// real dummy per name, so that equal roots are equal expressions.
pub fn bound_variable(name: &str) -> Basic {
    BOUND.with(|b| {
        b.borrow_mut()
            .entry(name.to_string())
            .or_insert_with(|| Dummy::new(name, &[(Predicate::real, true)]).0)
            .clone()
    })
}

/// The unevaluated node `IndexedRoot(f, t, k)`.
pub fn node(f: &Basic, t: &Basic, k: usize) -> PolyResult<Basic> {
    if !t.is_symbol() {
        return Err(value_error(
            "IndexedRoot(f, t, k) takes a symbol t and an index k, from 0".to_string(),
        ));
    }
    Ok(Basic::new(
        Head::IndexedRoot,
        vec![f.clone(), t.clone(), Basic::int(k as i64)],
    ))
}

/// The parts `(f, t, k)` of a node.
pub fn parts(e: &Basic) -> (&Basic, &Basic, usize) {
    let k: usize = e.args()[2]
        .as_rational()
        .and_then(|v| usize::try_from(&numer(v)).ok())
        .unwrap_or(0);
    (&e.args()[0], &e.args()[1], k)
}

fn variable_name(t: &Basic) -> String {
    match t.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Wild(w) => w.symbol.name.clone(),
        _ => t.to_string(),
    }
}

fn remember(node: &Basic, a: &Real) {
    ALGEBRAIC.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(position) = cache.iter().position(|(n, _)| n == node) {
            cache.remove(position);
        }
        cache.push((node.clone(), a.clone()));
        while cache.len() > KEPT {
            cache.remove(0);
        }
    })
}

fn remembered(node: &Basic) -> Option<Real> {
    ALGEBRAIC.with(|cache| cache.borrow().iter().find(|(n, _)| n == node).map(|(_, a)| a.clone()))
}

/// `f` as a polynomial in `t` with rational coefficients, of positive
/// degree (`_rational_poly`).
fn rational_poly(f: &Basic, t: &Basic) -> Option<Poly> {
    let poly: Poly = Poly::new(f, std::slice::from_ref(t)).ok()?;
    if !poly.domain.params.is_empty() || poly.degree(0).ok()? < 1 {
        return None;
    }
    Some(poly)
}

/// The real algebraic number of a rational or of an `IndexedRoot` with
/// rational coefficients (`algebraic_of`); `None` otherwise.
pub fn algebraic_of(e: &Basic) -> PolyResult<Option<Real>> {
    if let Some(v) = e.as_rational() {
        return Ok(Some(rational(v)));
    }
    if !matches!(e.head(), Head::IndexedRoot) || !e.free_symbols().is_empty() {
        return Ok(None);
    }
    if let Some(found) = remembered(e) {
        return Ok(Some(found));
    }
    let (f, t, k) = parts(e);
    let Some(poly) = rational_poly(f, t) else {
        return Ok(None);
    };
    let roots = rational_real_roots(&poly)?;
    if k >= roots.len() {
        return Ok(None);
    }
    let a: Real = roots[k].0.clone();
    remember(e, &a);
    Ok(Some(a))
}

/// The real algebraic number as an expression: a rational, or the
/// canonical `IndexedRoot` of its minimal polynomial in the variable
/// called `name` (`real_root`).
pub fn real_root(a: &Real, name: &str) -> PolyResult<Basic> {
    if a.is_rational() {
        return Ok(Basic::from_rational(a.rational_value()?));
    }
    let t: Basic = bound_variable(name);
    let minpoly: Vec<Int> = a.minpoly();
    let (lo, _) = a.real_interval()?;
    let rational_minpoly: Vec<Rat> = minpoly.iter().map(rat_from_int).collect();
    let index: i64 = dup_count_real_roots(&rational_minpoly, None, Some(&lo));
    let d: usize = minpoly.len() - 1;
    let terms: Vec<(Vec<u32>, Rat)> = minpoly
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| (vec![(d - i) as u32], rat_from_int(c)))
        .collect();
    let poly: Poly = Poly::from_dict(&terms, std::slice::from_ref(&t))?;
    let result: Basic = node(&poly.as_expr()?, &t, index as usize)?;
    if remembered(&result).is_none() {
        remember(&result, a);
    }
    Ok(result)
}

/// [`real_root`] in radicals when `radical_form` allows (`real_number`).
pub fn real_number(a: &Real, name: &str) -> PolyResult<Basic> {
    let n: Basic = real_root(a, name)?;
    if !matches!(n.head(), Head::IndexedRoot) {
        return Ok(n);
    }
    Ok(super::super::roots::radical_form(&n)?.unwrap_or(n))
}

/// `IndexedRoot(f, t, k)` evaluated (`indexed_root`).
pub fn indexed_root(f: &Basic, t: &Basic, k: usize) -> PolyResult<Basic> {
    if !t.is_symbol() {
        return Err(value_error(
            "IndexedRoot(f, t, k) takes a symbol t and an index k, from 0".to_string(),
        ));
    }
    let bound: Basic = bound_variable(&variable_name(t));
    let mut expression: Basic = f.clone();
    if *t != bound {
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        rule.insert(t.clone(), bound.clone());
        expression = expression.xreplace(&rule);
    }
    let free = expression.free_symbols();
    if expression.has(&bound) && free.len() == 1 && free.contains(&bound) {
        if let Some(value) = numerical_root(&expression, &bound, k)? {
            return Ok(value);
        }
    }
    node(&expression, &bound, k)
}

fn numerical_root(expression: &Basic, t: &Basic, index: usize) -> PolyResult<Option<Basic>> {
    let Some(poly) = rational_poly(expression, t) else {
        return Ok(None);
    };
    let roots = rational_real_roots(&poly)?;
    if index >= roots.len() {
        return Ok(None);
    }
    Ok(Some(real_number(&roots[index].0, &variable_name(t))?))
}

/// The polynomial and the index of the root as SymPy's `CRootOf` has them
/// (the minimal polynomial in a plain symbol, the index among its real
/// roots), for a root with rational coefficients which is not rational
/// (`polynomial_root`).
pub fn polynomial_root(e: &Basic) -> PolyResult<Option<(Poly, usize)>> {
    if !e.free_symbols().is_empty() {
        return Ok(None);
    }
    let Some(a) = algebraic_of(e)? else { return Ok(None) };
    if a.is_rational() {
        return Ok(None);
    }
    let (_, t, _) = parts(e);
    let canonical: Basic = real_root(&a, &variable_name(t))?;
    if !matches!(canonical.head(), Head::IndexedRoot) {
        return Ok(None);
    }
    let (f, v, k) = parts(&canonical);
    let plain: Basic = Symbol::new(&variable_name(v), &[]).0;
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(v.clone(), plain.clone());
    Ok(Some((Poly::new(&f.xreplace(&rule), std::slice::from_ref(&plain))?, k)))
}

/// The facts of a node (`_eval_facts`).
pub fn indexed_root_facts(e: &Basic) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    facts.set(Predicate::extended_real, true);
    if !e.free_symbols().is_empty() {
        return facts;
    }
    let Ok(Some(a)) = algebraic_of(e) else { return facts };
    let Ok(s) = a.sign() else { return facts };
    for (p, v) in [
        (Predicate::real, true),
        (Predicate::finite, true),
        (Predicate::zero, s == 0),
        (Predicate::positive, s > 0),
        (Predicate::negative, s < 0),
    ] {
        facts.set(p, v);
    }
    facts
}

/// The value of a real algebraic number at `prec` bits: the middle of its
/// interval refined to a relative width below `2**-prec`
/// (`_algebraic_value`).
pub fn algebraic_value(a: &Arc<crate::polys::numberfields::qqbar::Algebraic>, prec: usize) -> Option<Complex> {
    let mut eps: Rat = rat_from_parts(Int::ONE, Int::ONE << (prec + 4));
    let floor: Rat = rat_from_parts(Int::ONE, Int::ONE << (4 * prec + 64));
    let middle: Rat = loop {
        let (lo, hi, _, _) = a.enclosure(&eps).ok()?;
        let middle: Rat = (&lo + &hi) / rat(2, 1);
        let magnitude: Rat = if rat_is_negative(&middle) {
            -middle.clone()
        } else {
            middle.clone()
        };
        let relative: Rat = &magnitude * rat_from_parts(Int::ONE, Int::ONE << (prec + 2));
        if middle.is_zero() || &hi - &lo <= relative || eps < floor {
            break middle;
        }
        eps = if magnitude < Rat::ONE {
            &eps * &magnitude
        } else {
            &eps / rat(2, 1)
        };
    };
    Some(complex_real(real_rat(&middle, prec), prec))
}

/// The numerical value of a node without free symbols (`_numeric_node`):
/// that of its real algebraic number; `None` for irrational coefficients.
pub fn indexed_root_value(e: &Basic, prec: usize) -> Option<Complex> {
    if !e.free_symbols().is_empty() {
        return None;
    }
    let a: Real = algebraic_of(e).ok()??;
    algebraic_value(&a, prec)
}

/// `str` of a node: the bound variable shown as the plain symbol of its
/// name.
pub fn shown_parts(e: &Basic) -> (Basic, Basic, usize) {
    let (f, t, k) = parts(e);
    let plain: Basic = Symbol::new(&variable_name(t), &[]).0;
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(t.clone(), plain.clone());
    (f.xreplace(&rule), plain, k)
}
