//! Port of `sympy_extras/assumptions/resolve.py`: quantifier elimination
//! over the reals (virtual substitution, then the cylindrical algebraic
//! decomposition), the integers (Cooper's algorithm, with the real
//! relaxation of closed formulas) and the complex numbers (comprehensive
//! Gröbner systems); the result simplified under the assumptions.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::assumptions::ask::facts_of;
use crate::assumptions::facts::{normalize, to_polynomial, Facts};
use crate::assumptions::quantifiers::{prenex, Prefix, QuantifierKind};
use crate::core::basic::{Basic, Function, Head};
use crate::core::logic::{and_, boolean};
use crate::error::{value_error, PolyError, PolyResult};
use crate::polys::cad::projection::Method;

type R = PolyResult<Basic>;

fn symbols(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect()
}

fn is_atom_bool(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue | Head::BooleanFalse)
}

fn is_membership_in(c: &Basic, s: &Basic) -> bool {
    matches!(c.head(), Head::Contains) && c.args()[1] == *s
}

fn qe_prefix(prefix: &Prefix) -> Vec<(crate::polys::cad::qe::Quantifier, Basic)> {
    use crate::polys::cad::qe::Quantifier as Q;
    prefix
        .iter()
        .map(|(k, v)| {
            (
                if *k == QuantifierKind::ForAll {
                    Q::Forall
                } else {
                    Q::Exists
                },
                v.clone(),
            )
        })
        .collect()
}

/// Eliminate the quantifiers of a formula (`resolve`); `domain` `None` is
/// the reals.
pub fn resolve(formula: &Basic, domain: Option<&Basic>, assumptions: &[Basic], method: Option<Method>) -> R {
    let reals: Basic = crate::sets::sets::reals();
    let domain_: &Basic = domain.unwrap_or(&reals);
    match domain_.head() {
        Head::Integers => return resolve_integers(formula, assumptions),
        Head::Complexes => return resolve_complexes(formula, assumptions),
        Head::Reals => {}
        _ => {
            return Err(PolyError::NotImplemented(
                "quantifier elimination is only implemented over the real numbers, the complex numbers and, for linear formulas, the integers"
                    .to_string(),
            ))
        }
    }
    let (prefix, matrix) = prenex(formula)?;
    let bound: Vec<Basic> = prefix.iter().map(|(_, v)| v.clone()).collect();
    let mut given: BTreeSet<Basic> = matrix.free_symbols();
    given.extend(bound.iter().cloned());
    let given: Vec<Basic> = given.into_iter().collect();
    let facts: Facts = facts_of(assumptions, Some(&reals), &given)?;
    let matrix: Basic = normalize(&matrix)?;
    let mut real: BTreeSet<Basic> = facts.real.iter().cloned().collect();
    real.extend(bound.iter().cloned());
    real.extend(symbols(&matrix));
    let real: Vec<Basic> = real.into_iter().collect();
    let Some(mut poly) = to_polynomial(&matrix, &real)? else {
        return Err(value_error(format!(
            "the formula is not a Boolean combination of polynomial relations with rational coefficients: {}",
            matrix
        )));
    };
    let premise: Option<Basic> = to_polynomial(&facts.formula, &real)?;
    if let Some(p) = &premise {
        if !matches!(p.head(), Head::BooleanTrue) {
            poly = and_(&[p.clone(), poly]);
        }
    }
    let mut free: Vec<Basic> = symbols(&poly).into_iter().filter(|s| !bound.contains(s)).collect();
    free.sort_by_key(|s| s.to_string());
    if is_atom_bool(&poly) {
        return Ok(poly);
    }
    if prefix.is_empty() && free.is_empty() {
        if matches!(poly.head(), Head::Relational(_)) {
            return Err(PolyError::Type(format!("cannot determine the truth value of {}", poly)));
        }
        return Ok(boolean(true));
    }
    let (poly, remaining) =
        crate::polys::virtual_substitution::virtual_substitution_elimination(&poly, &prefix, 12, 2)?;
    if is_atom_bool(&poly) {
        return Ok(poly);
    }
    let eliminated: bool = remaining.len() < prefix.len();
    let mut result: Basic = if remaining.is_empty() && free.len() > 2 {
        poly
    } else {
        crate::polys::cad::qe::quantifier_elimination(&poly, &qe_prefix(&remaining), Some(&free), method, true)?
    };
    if eliminated {
        result = crate::polys::virtual_substitution::with_sides(&result)?;
    }
    refine_boolean_(result, &facts, premise.as_ref())
}

fn resolve_integers(formula: &Basic, assumptions: &[Basic]) -> R {
    use crate::solvers::integers::{is_presburger, presburger_quantifier_elimination};
    let integers_set: Basic = Basic::atom(Head::Integers);
    let (prefix, matrix) = prenex(formula)?;
    let bound: Vec<Basic> = prefix.iter().map(|(_, v)| v.clone()).collect();
    let mut given: BTreeSet<Basic> = matrix.free_symbols();
    given.extend(bound.iter().cloned());
    let given: Vec<Basic> = given.into_iter().collect();
    let facts: Facts = facts_of(assumptions, Some(&integers_set), &given)?;
    let mut matrix: Basic = normalize(&matrix)?;
    let mut integers: BTreeSet<Basic> = facts.integer.iter().cloned().collect();
    integers.extend(bound.iter().cloned());
    integers.extend(symbols(&matrix));
    let integers: Vec<Basic> = integers.into_iter().collect();
    let kept: Vec<Basic> = facts
        .conjuncts
        .iter()
        .filter(|c| symbols(c).iter().any(|s| !bound.contains(s)) && !is_membership_in(c, &integers_set))
        .cloned()
        .collect();
    let premise: Basic = and_(&kept);
    let trivial: bool = matches!(premise.head(), Head::BooleanTrue);
    if !trivial {
        matrix = and_(&[premise.clone(), matrix]);
    }
    if !is_presburger(&matrix, &integers)? {
        if let Some(relaxed) = real_relaxation(formula, &prefix, assumptions)? {
            return Ok(relaxed);
        }
        return Err(value_error(format!(
            "over the integers the formula must be a Boolean combination of linear relations with integer coefficients and divisibilities: {}",
            matrix
        )));
    }
    let mut result: Basic = presburger_quantifier_elimination(&matrix, &prefix)?;
    if !trivial && !is_atom_bool(&result) {
        result = crate::assumptions::refine::refine_boolean_under(&result, &facts)?;
    }
    Ok(result)
}

fn real_relaxation(formula: &Basic, prefix: &Prefix, assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    let kinds: BTreeSet<&'static str> = prefix.iter().map(|(k, _)| k.name()).collect();
    let bound: Vec<Basic> = prefix.iter().map(|(_, v)| v.clone()).collect();
    let has_mod: bool = formula
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(Function::Mod)));
    if kinds.len() != 1 || has_mod || symbols(formula).iter().any(|s| !bound.contains(s)) {
        return Ok(None);
    }
    let reals: Basic = crate::sets::sets::reals();
    let over_reals: Option<Basic> = match crate::timeout::attempt(|| resolve(formula, Some(&reals), assumptions, None))
    {
        Ok(v) => v,
        Err(PolyError::NotImplemented(_) | PolyError::Value(_) | PolyError::Type(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let kind: &str = kinds.iter().next().copied().unwrap_or("");
    Ok(match over_reals {
        Some(v) if matches!(v.head(), Head::BooleanTrue) && kind == "forall" => Some(boolean(true)),
        Some(v) if matches!(v.head(), Head::BooleanFalse) && kind == "exists" => Some(boolean(false)),
        _ => None,
    })
}

fn resolve_complexes(formula: &Basic, assumptions: &[Basic]) -> R {
    use crate::polys::comprehensive::{complex_quantifier_elimination, reduce_complex, Quantifier as CQ};
    let complexes: Basic = Basic::atom(Head::Complexes);
    let (prefix, matrix) = prenex(formula)?;
    let bound: Vec<Basic> = prefix.iter().map(|(_, v)| v.clone()).collect();
    let mut given: BTreeSet<Basic> = matrix.free_symbols();
    given.extend(bound.iter().cloned());
    let given: Vec<Basic> = given.into_iter().collect();
    let facts: Facts = facts_of(assumptions, Some(&complexes), &given)?;
    let matrix: Basic = normalize(&matrix)?;
    let cprefix: Vec<(CQ, Basic)> = prefix
        .iter()
        .map(|(k, v)| {
            (
                if *k == QuantifierKind::ForAll {
                    CQ::Forall
                } else {
                    CQ::Exists
                },
                v.clone(),
            )
        })
        .collect();
    let conditions: Vec<Basic> = facts
        .conjuncts
        .iter()
        .filter(|c| symbols(c).iter().any(|s| !bound.contains(s)) && !is_membership_in(c, &complexes))
        .cloned()
        .collect();
    if conditions.is_empty() {
        return complex_quantifier_elimination(&matrix, &cprefix);
    }
    let premise: Basic = reduce_complex(&and_(&conditions))?;
    let result: Basic = complex_quantifier_elimination(&and_(&[premise.clone(), matrix]), &cprefix)?;
    let parts: Vec<Basic> = if matches!(premise.head(), Head::And) {
        premise.args().to_vec()
    } else {
        vec![premise]
    };
    let known: HashMap<Basic, Basic> = parts.into_iter().map(|c| (c, boolean(true))).collect();
    reduce_complex(&crate::core::function::xreplace_evaluated(&result, &known)?)
}

fn refine_boolean_(result: Basic, facts: &Facts, premise: Option<&Basic>) -> R {
    if let Some(p) = premise {
        if !matches!(p.head(), Head::BooleanTrue) && !is_atom_bool(&result) {
            let refined: Basic = crate::assumptions::refine::refine_boolean_under(&result, facts)?;
            return crate::polys::virtual_substitution::with_sides(&refined);
        }
    }
    Ok(result)
}
