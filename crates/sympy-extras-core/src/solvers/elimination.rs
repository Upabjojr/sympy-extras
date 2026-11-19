//! Port of `sympy_extras/solvers/elimination.py`: elimination of
//! variables from polynomial, trigonometric, exponential and radical
//! equations: the algebraic dependencies as relations (the abstraction of
//! `simplify/normalform.rs`), the denominators saturated, a Gröbner basis
//! of the elimination ideal; implicitisation of parametrisations.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::basic::{Basic, Head, RelOp};
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::linalg::collect_kernels;
use crate::polys::domains::rationalfield::QQ;
use crate::polys::ideals::Ideal;
use crate::polys::rings::MonomialOrder;
use crate::simplify::normalform::{exponentials_of, normal_form, numer_denom, Abstraction};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}

fn expand_trig(e: &Basic) -> R {
    use crate::core::function::Hints;
    let hints = Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: true,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}

/// `lhs - rhs` of an equation, or the expression itself.
fn expression(equation: &Basic) -> R {
    if matches!(equation.head(), Head::Relational(RelOp::Eq)) {
        return crate::core::add::sub(&equation.args()[0], &equation.args()[1]);
    }
    Ok(equation.clone())
}

fn sorted(mut v: Vec<Basic>) -> Vec<Basic> {
    v.sort_by_cached_key(crate::core::sorting::sort_key);
    v
}

fn plain_symbols(e: &Basic) -> Vec<Basic> {
    e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect()
}

/// The polynomial consequences of the equations free of the variables
/// (`eliminate`).
pub fn eliminate(equations: &[Basic], variables: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut abstraction: Abstraction = Abstraction::new();
    let mut back: Vec<(Basic, Basic)> = Vec::new();
    let expanded: Vec<Basic> = equations
        .iter()
        .map(|q| expand_trig(&expression(q)?))
        .collect::<PolyResult<_>>()?;
    let i: Basic = Basic::atom(Head::ImaginaryUnit);
    if expanded.iter().any(|e| e.has(&i)) {
        return Err(PolyError::NotImplemented(
            "elimination with the imaginary unit in the equations".to_string(),
        ));
    }
    let mut numerators: Vec<Basic> = Vec::new();
    let mut denominators: Vec<Basic> = Vec::new();
    for e in exponentials_of(&expanded, &mut back)? {
        let (numerator, denominator) = numer_denom(&abstraction.abstract_(&e)?)?;
        if numerator != int(0) {
            numerators.push(numerator);
        }
        if !denominator.is_rational_class() {
            denominators.push(denominator);
        }
    }
    let mut relations: Vec<Basic> = abstraction.relations.clone();
    let mut found: Vec<Basic> = Vec::new();
    for e in numerators.iter().chain(denominators.iter()).chain(relations.iter()) {
        collect_kernels(e, &mut found);
    }
    for (k, v) in &abstraction.back {
        match back.iter_mut().find(|(a, _)| a == k) {
            Some(entry) => entry.1 = v.clone(),
            None => back.push((k.clone(), v.clone())),
        }
    }
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for kernel in &found {
        if !kernel.is_symbol() {
            let placeholder: Basic = crate::core::symbol::Dummy::new("k", &[]).0;
            rule.insert(kernel.clone(), placeholder.clone());
            back.push((placeholder, kernel.clone()));
        }
    }
    if !rule.is_empty() {
        let replace = |v: &[Basic]| -> PolyResult<Vec<Basic>> {
            v.iter()
                .map(|e| crate::core::function::xreplace_evaluated(e, &rule))
                .collect()
        };
        numerators = replace(&numerators)?;
        denominators = replace(&denominators)?;
        relations = replace(&relations)?;
    }
    let mut generators: Vec<Basic> = numerators.clone();
    generators.extend(relations.iter().cloned());
    let mut symbols: Vec<Basic> = Vec::new();
    for e in generators.iter().chain(denominators.iter()) {
        for s in plain_symbols(e) {
            if !symbols.contains(&s) {
                symbols.push(s);
            }
        }
    }
    let mut removed: Vec<Basic> = symbols.iter().filter(|s| variables.contains(s)).cloned().collect();
    for (placeholder, original) in &back {
        if symbols.contains(placeholder)
            && original.free_symbols().iter().any(|s| variables.contains(s))
            && !removed.contains(placeholder)
        {
            removed.push(placeholder.clone());
        }
    }
    if !denominators.is_empty() {
        let saturation: Basic = crate::core::symbol::Dummy::new("z", &[]).0;
        let product: Basic = crate::core::mul::mul(&denominators)?;
        generators.push(crate::core::add::sub(
            &int(1),
            &crate::core::mul::mul(&[saturation.clone(), product])?,
        )?);
        symbols.push(saturation.clone());
        removed.push(saturation);
    }
    if generators.is_empty() {
        return Ok(Vec::new());
    }
    if symbols.is_empty() {
        return Ok(vec![int(1)]);
    }
    let kept: Vec<Basic> = sorted(symbols.iter().filter(|s| !removed.contains(s)).cloned().collect());
    let dropped: Vec<Basic> = sorted(removed.clone());
    let mut all: Vec<Basic> = dropped.clone();
    all.extend(kept.iter().cloned());
    let whole = Ideal::new(&generators, &all, QQ, MonomialOrder::Grevlex)?;
    if kept.is_empty() {
        return Ok(if whole.is_whole_ring()? {
            vec![int(1)]
        } else {
            Vec::new()
        });
    }
    let ideal = if dropped.is_empty() {
        whole
    } else {
        whole.eliminate(&dropped)?
    };
    let mut results: Vec<Basic> = Vec::new();
    for g in ideal.groebner_basis(None)? {
        let mut polynomial: Basic = numer_denom(&ideal.to_expr(&g)?)?.0;
        for (placeholder, original) in back.iter().rev() {
            if polynomial.has(placeholder) {
                polynomial = crate::core::subs::subs(&polynomial, placeholder, original)?;
            }
        }
        if normal_form(&polynomial)? != int(0) {
            results.push(polynomial);
        }
    }
    Ok(sorted(results))
}

/// The implicit equations of a parametrisation (`implicitize`).
pub fn implicitize(parametrization: &[Basic], parameters: &[Basic], coordinates: &[Basic]) -> PolyResult<Vec<Basic>> {
    if parametrization.len() != coordinates.len() {
        return Err(value_error(format!(
            "{} coordinates for {} functions",
            coordinates.len(),
            parametrization.len()
        )));
    }
    if parameters.iter().any(|p| coordinates.contains(p)) {
        return Err(value_error("a parameter cannot be a coordinate".to_string()));
    }
    let equations: Vec<Basic> = coordinates
        .iter()
        .zip(parametrization.iter())
        .map(|(x, f)| crate::core::add::sub(x, f))
        .collect::<PolyResult<_>>()?;
    eliminate(&equations, parameters)
}
