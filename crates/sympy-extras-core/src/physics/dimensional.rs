//! Dimensional analysis, ported from `sympy_extras.physics.dimensional`:
//! the dimensionless groups of the Buckingham Pi theorem (an LLL reduced
//! basis of the integer kernel of the dimension matrix, the first nonzero
//! exponent of each positive) and the form a physical law must take.
//! Dimensions are lists of (base dimension, rational exponent); the
//! quantities a list of (name, dimension), in their order.

// Reference: E. Buckingham, On physically similar systems; illustrations of the use of
// dimensional equations, Phys. Rev. 4 (1914).

use crate::core::assumptions::Predicate;
use crate::core::basic::Basic;
use crate::error::{value_error, PolyResult};
use crate::numbers::{Int, Rat};

/// A dimension: the exponents of the base dimensions.
pub type Dimension = Vec<(String, Rat)>;
/// The quantities of a problem, by name, in their order.
pub type Quantities = Vec<(String, Dimension)>;

fn d(exponents: &[(&str, i64)]) -> Dimension {
    exponents.iter().map(|(b, e)| (b.to_string(), Rat::from(*e))).collect()
}

/// The dimension of a usual quantity in mass M, length L, time T, current
/// I, temperature K (`SI`).
pub fn si(name: &str) -> Option<Dimension> {
    Some(match name {
        "dimensionless" => d(&[]),
        "mass" => d(&[("M", 1)]),
        "length" => d(&[("L", 1)]),
        "time" => d(&[("T", 1)]),
        "current" => d(&[("I", 1)]),
        "temperature" => d(&[("K", 1)]),
        "area" => d(&[("L", 2)]),
        "volume" => d(&[("L", 3)]),
        "velocity" => d(&[("L", 1), ("T", -1)]),
        "acceleration" => d(&[("L", 1), ("T", -2)]),
        "frequency" => d(&[("T", -1)]),
        "density" => d(&[("M", 1), ("L", -3)]),
        "force" => d(&[("M", 1), ("L", 1), ("T", -2)]),
        "pressure" => d(&[("M", 1), ("L", -1), ("T", -2)]),
        "energy" => d(&[("M", 1), ("L", 2), ("T", -2)]),
        "power" => d(&[("M", 1), ("L", 2), ("T", -3)]),
        "momentum" => d(&[("M", 1), ("L", 1), ("T", -1)]),
        "angular momentum" => d(&[("M", 1), ("L", 2), ("T", -1)]),
        "dynamic viscosity" => d(&[("M", 1), ("L", -1), ("T", -1)]),
        "kinematic viscosity" => d(&[("L", 2), ("T", -1)]),
        "surface tension" => d(&[("M", 1), ("T", -2)]),
        "charge" => d(&[("I", 1), ("T", 1)]),
        "voltage" => d(&[("M", 1), ("L", 2), ("T", -3), ("I", -1)]),
        "resistance" => d(&[("M", 1), ("L", 2), ("T", -3), ("I", -2)]),
        "capacitance" => d(&[("M", -1), ("L", -2), ("T", 4), ("I", 2)]),
        "inductance" => d(&[("M", 1), ("L", 2), ("T", -2), ("I", -2)]),
        "thermal conductivity" => d(&[("M", 1), ("L", 1), ("T", -3), ("K", -1)]),
        "specific heat" => d(&[("L", 2), ("T", -2), ("K", -1)]),
        "gravitational constant" => d(&[("M", -1), ("L", 3), ("T", -2)]),
        "action" => d(&[("M", 1), ("L", 2), ("T", -1)]),
        _ => return None,
    })
}

fn exponent_of(dimension: &Dimension, base: &str) -> Rat {
    dimension
        .iter()
        .find(|(b, _)| b == base)
        .map(|(_, e)| e.clone())
        .unwrap_or(Rat::ZERO)
}

/// `(bases, D)`: the base dimensions which occur (sorted) and the integer
/// matrix with a row for each and a column for each quantity, the rows
/// scaled when the exponents are fractions (`dimension_matrix`).
pub fn dimension_matrix(quantities: &Quantities) -> (Vec<String>, Vec<Vec<Int>>) {
    let mut bases: Vec<String> = Vec::new();
    for (_, dimension) in quantities {
        for (base, exponent) in dimension {
            if *exponent != Rat::ZERO && !bases.contains(base) {
                bases.push(base.clone());
            }
        }
    }
    bases.sort();
    let mut rows: Vec<Vec<Int>> = Vec::new();
    for base in &bases {
        let line: Vec<Rat> = quantities
            .iter()
            .map(|(_, dimension)| exponent_of(dimension, base))
            .collect();
        let mut scale: Int = Int::ONE;
        for entry in &line {
            scale = crate::core::intfunc::ilcm(&[scale, Int::from(entry.denominator().clone())]);
        }
        rows.push(
            line.iter()
                .map(|entry| (entry * Rat::from(scale.clone())).numerator().clone())
                .collect(),
        );
    }
    (bases, rows)
}

/// The exponent vectors of a basis of the dimensionless products, the
/// first nonzero exponent of each positive, sorted (`pi_exponents`).
pub fn pi_exponents(quantities: &Quantities) -> PolyResult<Vec<Vec<Int>>> {
    let (_, rows) = dimension_matrix(quantities);
    let basis = crate::polys::lattice::integer_kernel(&rows, quantities.len())?;
    let first = |v: &Vec<Int>| v.iter().position(|e| !e.is_zero()).unwrap_or(v.len());
    let mut oriented: Vec<Vec<Int>> = basis
        .into_iter()
        .map(|v| {
            let lead_negative: bool = v.iter().find(|e| !e.is_zero()).is_some_and(|e| *e < Int::ZERO);
            if lead_negative {
                v.into_iter().map(|e| -e).collect()
            } else {
                v
            }
        })
        .collect();
    oriented.sort_by(|a, b| (first(a), a).cmp(&(first(b), b)));
    Ok(oriented)
}

fn product(names: &[String], exponents: &[Rat]) -> PolyResult<Basic> {
    let mut factors: Vec<Basic> = Vec::new();
    for (name, exponent) in names.iter().zip(exponents) {
        if *exponent != Rat::ZERO {
            let symbol: Basic = crate::core::Symbol::new(name, &[(Predicate::positive, true)]).0;
            factors.push(crate::core::power::pow(
                &symbol,
                &Basic::from_rational(exponent.clone()),
            )?);
        }
    }
    if factors.is_empty() {
        return Ok(Basic::one());
    }
    crate::core::mul::mul(&factors)
}

/// A complete set of independent dimensionless products of the quantities
/// (positive symbols with their names), with small integer exponents
/// (`pi_groups`).
pub fn pi_groups(quantities: &Quantities) -> PolyResult<Vec<Basic>> {
    let names: Vec<String> = quantities.iter().map(|(n, _)| n.clone()).collect();
    let mut groups: Vec<Basic> = Vec::new();
    for vector in pi_exponents(quantities)? {
        let exponents: Vec<Rat> = vector.into_iter().map(Rat::from).collect();
        groups.push(product(&names, &exponents)?);
    }
    Ok(groups)
}

/// The dimension of the product of the quantities raised to the given
/// exponents, the base dimensions with nonzero exponents (`dimension_of`).
pub fn dimension_of(exponents: &[(String, Rat)], quantities: &Quantities) -> PolyResult<Dimension> {
    let mut total: Dimension = Vec::new();
    for (name, exponent) in exponents {
        let Some((_, dimension)) = quantities.iter().find(|(n, _)| n == name) else {
            return Err(value_error(format!("no quantity '{}'", name)));
        };
        for (base, power) in dimension {
            let added: Rat = power * exponent;
            match total.iter_mut().find(|(b, _)| b == base) {
                Some(entry) => entry.1 = &entry.1 + added,
                None => total.push((base.clone(), added)),
            }
        }
    }
    Ok(total.into_iter().filter(|(_, p)| *p != Rat::ZERO).collect())
}

/// Whether the product with these exponents has no dimension
/// (`is_dimensionless`).
pub fn is_dimensionless(exponents: &[(String, Rat)], quantities: &Quantities) -> PolyResult<bool> {
    Ok(dimension_of(exponents, quantities)?.is_empty())
}

/// The product of powers of the other quantities with the dimension of
/// `target`, when the dimensions determine it (`solve_for`).
pub fn solve_for(target: &str, quantities: &Quantities) -> PolyResult<Basic> {
    let Some((_, dimension)) = quantities.iter().find(|(n, _)| n == target) else {
        return Err(value_error(format!("no quantity '{}'", target)));
    };
    let others: Quantities = quantities.iter().filter(|(n, _)| n != target).cloned().collect();
    if !pi_exponents(&others)?.is_empty() {
        return Err(value_error(format!(
            "the other quantities have a dimensionless group: {} is not determined by the dimensions",
            target
        )));
    }
    let mut ordered: Quantities = others.clone();
    ordered.push((target.to_string(), dimension.clone()));
    let vectors: Vec<Vec<Int>> = pi_exponents(&ordered)?;
    if vectors.is_empty() {
        return Err(value_error(format!(
            "no product of the other quantities has the dimension of {}",
            target
        )));
    }
    if vectors.len() != 1 {
        return Err(value_error("not enough values to unpack".to_string()));
    }
    let vector: &Vec<Int> = &vectors[0];
    let own: Int = vector[vector.len() - 1].clone();
    let names: Vec<String> = others.iter().map(|(n, _)| n.clone()).collect();
    let exponents: Vec<Rat> = vector[..vector.len() - 1]
        .iter()
        .map(|e| Rat::from_parts_signed(-e.clone(), own.clone()))
        .collect();
    product(&names, &exponents)
}
