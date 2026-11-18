//! The named sets of numbers, ported from `sympy_extras.sets.fancysets`:
//! the singletons `Naturals`, `Naturals0`, `Integers`, `Rationals`,
//! `Complexes` (memberships decided from the assumptions) and the
//! unevaluated `ImageSet` (no membership rules), with [`image_set`]
//! evaluating as SymPy's constructor.

// PORTED FROM SYMPY: sympy/sets/fancysets.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::core::logic::{boolean, false_, true_};
use crate::error::{value_error, PolyError, PolyResult};

/// Whether the head is one of the named sets of numbers.
pub fn is_named_set(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Naturals | Head::Naturals0 | Head::Integers | Head::Rationals | Head::Complexes
    )
}

/// The printed name of a named set (`set_name`).
pub fn set_name(e: &Basic) -> &'static str {
    match e.head() {
        Head::Naturals => "Naturals",
        Head::Naturals0 => "Naturals0",
        Head::Integers => "Integers",
        Head::Rationals => "Rationals",
        _ => "Complexes",
    }
}

fn tfn(value: Option<bool>) -> Option<Basic> {
    value.map(boolean)
}

/// The membership of an expression (`_contains`).
pub fn named_contains(s: &Basic, x: &Basic) -> Option<Basic> {
    match s.head() {
        Head::Naturals => {
            if is(x, Predicate::positive) == Some(true) && is(x, Predicate::integer) == Some(true) {
                return Some(true_());
            }
            if is(x, Predicate::integer) == Some(false) || is(x, Predicate::positive) == Some(false) {
                return Some(false_());
            }
            None
        }
        Head::Naturals0 => {
            if is(x, Predicate::integer) == Some(true) && is(x, Predicate::nonnegative) == Some(true) {
                return Some(true_());
            }
            if is(x, Predicate::integer) == Some(false) || is(x, Predicate::nonnegative) == Some(false) {
                return Some(false_());
            }
            None
        }
        Head::Integers => tfn(is(x, Predicate::integer)),
        Head::Rationals => tfn(is(x, Predicate::rational)),
        _ => tfn(is(x, Predicate::complex)),
    }
}

/// The infimum of a named set (`Complexes` has none).
pub fn named_inf(s: &Basic) -> PolyResult<Basic> {
    match s.head() {
        Head::Naturals => Ok(Basic::one()),
        Head::Naturals0 => Ok(Basic::zero()),
        Head::Integers | Head::Rationals => Ok(Basic::negative_oo()),
        _ => Err(PolyError::NotImplemented(format!("({})._inf", set_name(s)))),
    }
}

/// The supremum of a named set.
pub fn named_sup(s: &Basic) -> PolyResult<Basic> {
    match s.head() {
        Head::Complexes => Err(PolyError::NotImplemented("(Complexes)._sup".to_string())),
        _ => Ok(Basic::oo()),
    }
}

/// The unevaluated `ImageSet(variables, image, *base_sets)`.
pub fn image_set_node(variables: &Basic, image: &Basic, base_sets: &[Basic]) -> PolyResult<Basic> {
    if !variables.args().iter().all(|v| v.is_symbol()) {
        return Err(PolyError::Type(format!(
            "the variables of an ImageSet are symbols, got {}",
            variables
        )));
    }
    let expression = |e: &Basic| !matches!(e.head(), Head::Tuple) && !crate::sets::sets::is_set(e);
    if !(expression(image) || (matches!(image.head(), Head::Tuple) && image.args().iter().all(expression))) {
        return Err(PolyError::Type(format!(
            "the image of an ImageSet is an expression or a tuple of them, got {}",
            image
        )));
    }
    let mut args: Vec<Basic> = vec![variables.clone(), image.clone()];
    args.extend(base_sets.iter().cloned());
    Ok(Basic::new(Head::ImageSet, args))
}

/// The image of the base sets, evaluated as SymPy's `ImageSet` does: the
/// base set itself for the identity, a single point for a constant function
/// (`image_set`).
pub fn image_set(variables: &Basic, image: &Basic, base_sets: &[Basic]) -> PolyResult<Basic> {
    use crate::sets::sets::{empty_set, finite_set, fuzzy_or, is_empty, point_set};
    if variables.args().len() != base_sets.len() {
        return Err(value_error("Incompatible signature".to_string()));
    }
    if base_sets.len() == 1 && variables.args() == std::slice::from_ref(image) {
        return Ok(base_sets[0].clone());
    }
    let free: BTreeSet<Basic> = image.free_symbols();
    if !variables.args().iter().any(|v| free.contains(v)) {
        let mut values: Vec<Option<bool>> = Vec::new();
        for s in base_sets {
            values.push(is_empty(s)?);
        }
        match fuzzy_or(values) {
            Some(true) => return Ok(empty_set()),
            Some(false) => {
                if matches!(image.head(), Head::Tuple) {
                    return point_set(std::slice::from_ref(image));
                }
                return Ok(finite_set(std::slice::from_ref(image)));
            }
            None => {}
        }
    }
    image_set_node(variables, image, base_sets)
}

/// The arguments of an `ImageSet` rebuild checked (`_image_set_args`).
pub fn rebuild_image_set(args: &[Basic]) -> PolyResult<Basic> {
    if !matches!(args[0].head(), Head::Tuple) {
        return Err(PolyError::Type(format!(
            "the variables of an ImageSet are a tuple, got {}",
            args[0]
        )));
    }
    if !args[2..].iter().all(crate::sets::sets::is_set) {
        return Err(PolyError::Type("the base sets of an ImageSet are sets".to_string()));
    }
    image_set(&args[0], &args[1], &args[2..])
}

/// The free symbols of an `ImageSet`: those of the image but the
/// variables, and those of the base sets.
pub fn image_set_free_symbols(e: &Basic) -> BTreeSet<Basic> {
    let variables: &[Basic] = e.args()[0].args();
    let mut result: BTreeSet<Basic> = e.args()[1]
        .free_symbols()
        .into_iter()
        .filter(|s| !variables.contains(s))
        .collect();
    for s in &e.args()[2..] {
        result.extend(s.free_symbols());
    }
    result
}

/// `ImageSet(Lambda(x, image), base sets)` (`StrPrinter._print_ImageSet`).
pub fn print_image_set(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let variables: &[Basic] = e.args()[0].args();
    let shown: String = if variables.len() == 1 {
        doprint(&variables[0])
    } else {
        format!(
            "({})",
            variables.iter().map(doprint).collect::<Vec<String>>().join(", ")
        )
    };
    let bases: Vec<String> = e.args()[2..].iter().map(doprint).collect();
    format!(
        "ImageSet(Lambda({}, {}), {})",
        shown,
        doprint(&e.args()[1]),
        bases.join(", ")
    )
}
