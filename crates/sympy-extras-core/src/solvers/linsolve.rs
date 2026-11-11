//! Port of `sympy_extras/solvers/linsolve.py`: the unique solution of a
//! linear system with rational function coefficients
//! ([`solve_linear_unique`], the reduced row echelon form over
//! `QQ(other symbols)`), the root of a linear equation ([`linear_root`],
//! [`solve_one`]) and the solution of a numerical linear system
//! ([`numeric_linear_solution`]).

// a node caches its keys in `OnceLock`s that the hash never reads
#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, HashMap};

use crate::core::basic::Basic;
use crate::core::function::diff;
use crate::core::mul::{div, neg};
use crate::core::sorting::sort_key;
use crate::core::subs::{subs, subs_mapping};
use crate::core::Symbol;
use crate::error::{PolyError, PolyResult};
use crate::polys::domains::domain::{Domain, SparseRow};
use crate::polys::domains::rationalfield::QQ;
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;

/// The values of `unknowns` (in their order) solving the linear
/// `equations` (each `= 0`) when they are unique; `None` for an
/// inconsistent or underdetermined system, an equation not linear in the
/// unknowns, or a coefficient outside `QQ(other symbols)`
/// (`solve_linear_unique`).
pub fn solve_linear_unique(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    if unknowns.is_empty() {
        return Ok(None);
    }
    let mut others: Vec<Basic> = Vec::new();
    for e in equations {
        for s in e.free_symbols() {
            if !unknowns.contains(&s) && !others.contains(&s) {
                others.push(s);
            }
        }
    }
    others.sort_by_cached_key(sort_key);
    // unique names for the other symbols, one placeholder when there is none
    let mut names: Vec<String> = (0..others.len()).map(|i| format!("_ls{}", i)).collect();
    if names.is_empty() {
        names.push("_ls".to_string());
    }
    let renamed: HashMap<Basic, Basic> = others
        .iter()
        .zip(&names)
        .map(|(s, name)| (s.clone(), Symbol::new(name, &[]).0))
        .collect();
    let restored: HashMap<Basic, Basic> = others
        .iter()
        .zip(&names)
        .map(|(s, name)| (Symbol::new(name, &[]).0, s.clone()))
        .collect();
    let field: SparseFractionField<crate::polys::domains::rationalfield::RationalField> =
        SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names));
    let zeros: HashMap<Basic, Basic> = unknowns.iter().map(|u| (u.clone(), Basic::zero())).collect();
    let mut rows: Vec<SparseRow<SparseFracElement<_>>> = Vec::new();
    for e in equations {
        let mut row: SparseRow<SparseFracElement<_>> = BTreeMap::new();
        let mut entries: Vec<Basic> = Vec::new();
        for u in unknowns {
            entries.push(diff(e, &[(u.clone(), 1)])?);
        }
        entries.push(e.xreplace(&zeros));
        for (column, entry) in entries.iter().enumerate() {
            if unknowns.iter().any(|u| entry.has(u)) {
                return Ok(None); // not linear in the unknowns
            }
            let value = match field.from_expr(&entry.xreplace(&renamed)) {
                Ok(v) => v,
                Err(PolyError::Value(_)) => return Ok(None), // a coefficient outside QQ(others)
                Err(other) => return Err(other),
            };
            if !field.is_zero(&value) {
                row.insert(column, value);
            }
        }
        if !row.is_empty() {
            rows.push(row);
        }
    }
    let echelon = field.sparse_echelon(&rows);
    let count: usize = unknowns.len();
    let leading = |r: &SparseRow<SparseFracElement<_>>| *r.keys().next().expect("a nonzero row");
    if echelon.iter().any(|r| leading(r) == count) {
        return Ok(None); // inconsistent
    }
    let mut pivots: Vec<usize> = echelon.iter().map(leading).collect();
    pivots.sort();
    if pivots != (0..count).collect::<Vec<usize>>() {
        return Ok(None); // underdetermined
    }
    let mut solution: Vec<Basic> = Vec::new();
    for r in &echelon {
        let value = match r.get(&count) {
            Some(constant) => field.neg(constant),
            None => field.zero(),
        };
        solution.push(field.to_expr(&value)?.xreplace(&restored));
    }
    Ok(Some(solution))
}

/// The root `-b/a` of `e = a*v + b` for a nonzero rational `a`
/// (`linear_root`).
pub fn linear_root(e: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    let a: Basic = diff(e, &[(v.clone(), 1)])?;
    match a.as_rational() {
        Some(x) if !x.is_zero() => {}
        _ => return Ok(None),
    }
    let b: Basic = subs(e, v, &Basic::zero())?;
    if b.has(v) {
        return Ok(None);
    }
    Ok(Some(div(&neg(&b)?, &a)?))
}

/// SymPy's `solve(e, v)` when it is known natively: `[]` for an equation
/// free of `v`, the root of [`linear_root`]; `None` otherwise
/// (`solve_one`).
pub fn solve_one(e: &Basic, v: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    if !e.has(v) {
        return Ok(Some(Vec::new()));
    }
    Ok(linear_root(e, v)?.map(|root| vec![root]))
}

/// The unique solution for all the free symbols of `equations` when they
/// are linear with rational coefficients and constants
/// (`numeric_linear_solution`).
pub fn numeric_linear_solution(equations: &[Basic]) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    let mut unknowns: Vec<Basic> = Vec::new();
    for e in equations {
        for s in e.free_symbols() {
            if !unknowns.contains(&s) {
                unknowns.push(s);
            }
        }
    }
    unknowns.sort_by_cached_key(sort_key);
    if unknowns.is_empty() {
        return Ok(None);
    }
    let zeros: Vec<(Basic, Basic)> = unknowns.iter().map(|u| (u.clone(), Basic::zero())).collect();
    for e in equations {
        if subs_mapping(e, &zeros)?.as_rational().is_none() {
            return Ok(None);
        }
        for u in &unknowns {
            if diff(e, &[(u.clone(), 1)])?.as_rational().is_none() {
                return Ok(None);
            }
        }
    }
    let Some(values) = solve_linear_unique(equations, &unknowns)? else {
        return Ok(None);
    };
    Ok(Some(unknowns.into_iter().zip(values).collect()))
}
