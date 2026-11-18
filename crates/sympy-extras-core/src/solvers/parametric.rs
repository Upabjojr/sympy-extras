//! Polynomial systems with parameters, ported from
//! `sympy_extras.solvers.parametric`: the cases of the parameters, read
//! from a triangular decomposition in the sense of Lazard with the
//! unknowns greater than the parameters. In a chain, the polynomials whose
//! main variable is a parameter are the equations of the case, the
//! initials (and the inequations of the system) which do not vanish are
//! its inequations, and the polynomials in the unknowns give them one
//! after the other, the smallest first: by a division (degree one) or the
//! quadratic formula (degree two); a higher degree leaves the solutions as
//! the equations of the chain. An inequation in the unknowns implied by
//! the other conditions is not written.

// References: C. Chen, M. Moreno Maza, Algorithms for computing triangular decomposition of
// polynomial systems, J. Symbolic Comput. 47 (2012); C. Chen, O. Golubitsky, F. Lemaire,
// M. Moreno Maza, W. Pan, Comprehensive triangular decomposition, CASC 2007.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use crate::core::add::{add, sub};
use crate::core::basic::{Basic, Head};
use crate::core::mul::{div, mul, neg};
use crate::core::power::pow;
use crate::core::relational::{eq, ne};
use crate::error::PolyResult;
use crate::polys::polytools::{cancel, factor_list, Poly};
use crate::polys::regularchains::regularchain::{triangularize, RegularChain};

/// A case of a system with parameters (`ParametricCase`): its condition
/// (the equations and inequations of the parameters, and the inequations
/// of the unknowns which the others do not imply), the polynomials of the
/// chain in the unknowns, and their solutions as a finite set of tuples of
/// the unknowns (`None` with a polynomial of degree three or more).
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricCase {
    pub condition: Basic,
    pub solutions: Option<Basic>,
    pub equations: Vec<Basic>,
}

impl ParametricCase {
    /// The case as a formula (`as_formula`).
    pub fn as_formula(&self) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = vec![self.condition.clone()];
        for e in &self.equations {
            parts.push(eq(e, &Basic::zero())?);
        }
        Ok(crate::core::logic::and_(&parts))
    }
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

/// The irreducible factors of a polynomial which is not zero (`_factors`).
fn factors(h: &Basic) -> PolyResult<Vec<Basic>> {
    Ok(factor_list(h, &[])?.1.into_iter().map(|(f, _)| f).collect())
}

fn has_any(e: &Basic, symbols: &[Basic]) -> bool {
    symbols.iter().any(|s| e.has(s))
}

/// The inequations in the parameters, and those in the unknowns which the
/// chain and the others do not imply (`_kept_inequations`).
fn kept_inequations(
    chain: &RegularChain,
    inequations: &[Basic],
    symbols: &[Basic],
    unknowns: &[Basic],
) -> PolyResult<Vec<Basic>> {
    let mut kept: Vec<Basic> = inequations.to_vec();
    for h in inequations {
        if !has_any(h, unknowns) {
            continue;
        }
        let others: Vec<Basic> = kept.iter().filter(|o| *o != h).cloned().collect();
        let mut system: Vec<Basic> = chain.polys()?;
        system.push(h.clone());
        if triangularize(&system, symbols, &others, false)?.is_empty() {
            kept = others;
        }
    }
    Ok(kept)
}

/// The complex solutions of polynomial `equations` with rational
/// coefficients in the `unknowns` for every value of the `parameters` (the
/// other symbols, sorted by name, by default): the cases of the
/// parameters, each with its solutions (`parametric_cases`).
pub fn parametric_cases(
    equations: &[Basic],
    unknowns: &[Basic],
    parameters: Option<&[Basic]>,
    inequations: &[Basic],
) -> PolyResult<Vec<ParametricCase>> {
    let names: Vec<Basic> = unknowns.to_vec();
    let parameters: Vec<Basic> = match parameters {
        Some(p) => p.to_vec(),
        None => {
            let mut found: BTreeSet<Basic> = BTreeSet::new();
            for e in equations.iter().chain(inequations) {
                found.extend(e.free_symbols());
            }
            let mut p: Vec<Basic> = found.into_iter().filter(|s| !names.contains(s)).collect();
            p.sort_by_key(symbol_name);
            p
        }
    };
    let mut symbols: Vec<Basic> = names.clone();
    symbols.extend(parameters.iter().filter(|p| !names.contains(p)).cloned());
    let mut cases: Vec<ParametricCase> = Vec::new();
    for chain in triangularize(equations, &symbols, inequations, false)? {
        let mut nonzero: Vec<Basic> = Vec::new();
        let mut sources: Vec<Basic> = chain.initials()?;
        sources.extend(inequations.iter().cloned());
        for h in &sources {
            for factor in factors(h)? {
                if !factor.free_symbols().is_empty() && !nonzero.contains(&factor) {
                    nonzero.push(factor);
                }
            }
        }
        let nonzero: Vec<Basic> = kept_inequations(&chain, &nonzero, &symbols, &names)?;
        let mut conditions: Vec<Basic> = Vec::new();
        let mut in_unknowns: Vec<(Basic, Basic)> = Vec::new();
        for (p, v) in chain.polys()?.into_iter().zip(chain.main_variables()) {
            if names.contains(&v) {
                in_unknowns.push((p, v));
            } else {
                conditions.push(solved_parameter(&p, &v)?);
            }
        }
        for h in &nonzero {
            conditions.push(ne(h, &Basic::zero())?);
        }
        let solutions: Option<Basic> = families(&in_unknowns, &names)?;
        cases.push(ParametricCase {
            condition: crate::core::logic::and_(&conditions),
            solutions,
            equations: in_unknowns.into_iter().map(|(p, _)| p).collect(),
        });
    }
    Ok(cases)
}

/// The equation of a case, solved for its main variable when that one
/// occurs alone in it with a constant coefficient (`_solved_parameter`).
fn solved_parameter(p: &Basic, v: &Basic) -> PolyResult<Basic> {
    let poly: Poly = Poly::new(p, std::slice::from_ref(v))?;
    if poly.degree(0)? == 1 {
        let coefficients: Vec<Basic> = poly.all_coeffs()?;
        let (lc, tc) = (&coefficients[0], &coefficients[coefficients.len() - 1]);
        if lc.free_symbols().is_empty() && tc.free_symbols().is_empty() {
            return eq(v, &div(&neg(tc)?, lc)?);
        }
    }
    eq(p, &Basic::zero())
}

/// The unknowns from the polynomials of a chain, the smallest main
/// variable first; `None` when one has degree three or more (`_families`).
fn families(polys: &[(Basic, Basic)], unknowns: &[Basic]) -> PolyResult<Option<Basic>> {
    let mut families: Vec<Vec<(Basic, Basic)>> = vec![Vec::new()];
    for (p, v) in polys {
        let poly: Poly = Poly::new(p, std::slice::from_ref(v))?;
        if poly.degree(0)? > 2 {
            return Ok(None);
        }
        let mut extended: Vec<Vec<(Basic, Basic)>> = Vec::new();
        for family in &families {
            let mut coefficients: Vec<Basic> = Vec::new();
            for c in poly.all_coeffs()? {
                coefficients.push(substituted(&c, family)?);
            }
            let values: Vec<Basic> = if coefficients.len() == 2 {
                vec![cancel(&div(&neg(&coefficients[1])?, &coefficients[0])?, &[])?]
            } else {
                let (a, b, c) = (&coefficients[0], &coefficients[1], &coefficients[2]);
                let two: Basic = Basic::integer(2.into());
                let four_ac: Basic = mul(&[mul(&[Basic::integer(4.into()), a.clone()])?, c.clone()])?;
                let discriminant: Basic = sub(&pow(b, &two)?, &four_ac)?;
                let root: Basic = pow(&cancel(&discriminant, &[])?, &Basic::half())?;
                let two_a: Basic = mul(&[two.clone(), a.clone()])?;
                let first: Basic = div(&neg(b)?, &two_a)?;
                let second: Basic = div(&root, &two_a)?;
                vec![sub(&first, &second)?, add(&[first, second])?]
            };
            for value in values {
                let mut longer: Vec<(Basic, Basic)> = family.clone();
                match longer.iter_mut().find(|(k, _)| k == v) {
                    Some(entry) => entry.1 = value,
                    None => longer.push((v.clone(), value)),
                }
                extended.push(longer);
            }
        }
        families = extended;
    }
    let mut points: Vec<Basic> = Vec::new();
    for family in &families {
        let values: Vec<Basic> = unknowns
            .iter()
            .map(|s| {
                family
                    .iter()
                    .find(|(k, _)| k == s)
                    .map(|(_, x)| x.clone())
                    .unwrap_or_else(|| s.clone())
            })
            .collect();
        points.push(Basic::new(Head::Tuple, values));
    }
    Ok(Some(crate::sets::sets::point_set(&points)?))
}

/// `e` with the values of the unknowns found so far, evaluated
/// (`_substituted`).
fn substituted(e: &Basic, values: &[(Basic, Basic)]) -> PolyResult<Basic> {
    if values.is_empty() {
        return Ok(e.clone());
    }
    let rule: std::collections::HashMap<Basic, Basic> = values.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &rule)
}
