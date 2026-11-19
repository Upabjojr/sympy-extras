//! Port of `sympy_extras/matrices/pencils.py`: the generalised eigenvalue
//! problem `A v = x B v` of a regular pencil: `det(A - x*B)`, its
//! eigenspaces in `K[t]/(factor)` and the infinite eigenvalue of
//! multiplicity `n - deg det(A - x*B)`.

use crate::core::basic::{Basic, Head};
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::domainmatrix::DomainMatrix;
use crate::matrices::eigen::{
    coefficients, column, extension, field_nullspace, roots_in_radicals, Eigenspace, Field, FunctionEntries,
    RationalEntries,
};
use crate::matrices::linalg::{det, kernels as kernels_of, normal, square, PolynomialLift};
use crate::numbers::numer;
use crate::polys::domains::domain::Domain;

fn check(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<usize> {
    let n: usize = square(a)?;
    if square(b)? != n {
        return Err(value_error(format!(
            "the matrices of a pencil have the same size: ({}, {}) and ({}, {})",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    Ok(n)
}

fn fresh(a: &ExprMatrix, b: &ExprMatrix) -> Basic {
    let mut used: Vec<String> = Vec::new();
    for m in [a, b] {
        for e in &m.entries {
            for s in e.free_symbols() {
                if matches!(s.head(), Head::Symbol(_)) {
                    used.push(s.to_string());
                }
            }
        }
    }
    let (mut name, mut k) = ("_lambda".to_string(), 0);
    while used.contains(&name) {
        name = format!("_lambda{}", k);
        k += 1;
    }
    crate::core::symbol::Symbol::new(&name, &[]).0
}

/// The numerator of `det(A - x*B)` (`pencil_polynomial`).
pub fn pencil_polynomial(a: &ExprMatrix, b: &ExprMatrix, x: &Basic) -> PolyResult<Basic> {
    let n: usize = check(a, b)?;
    let unknown: Basic = fresh(a, b);
    let determinant: Basic = if n > 0 {
        det(&a.sub(&b.scale(&unknown)?)?.applyfunc(&normal)?)?
    } else {
        Basic::int(1)
    };
    let together: Basic = crate::polys::rationaltools::together(&normal(&determinant)?, false, true)?;
    let numerator: Basic = crate::core::function::expand_default(&crate::core::expr::as_numer_denom(&together)?.0)?;
    crate::core::subs::subs(&numerator, &unknown, x)
}

fn spaces<F: Field>(
    field: &F,
    a: &ExprMatrix,
    b: &ExprMatrix,
    factors: &[(Basic, usize)],
    unknown: &Basic,
    x: &Basic,
) -> PolyResult<Vec<Eigenspace>> {
    let mut out: Vec<Eigenspace> = Vec::new();
    for (factor, multiplicity) in factors {
        let f = extension(field, factor, unknown)?;
        let mut rows = Vec::new();
        for i in 0..a.rows {
            let mut row = Vec::new();
            for (p, q) in a.row(i).iter().zip(b.row(i).iter()) {
                row.push(f.sub(
                    &f.from_ground(&field.element(p)?),
                    &f.mul(&f.generator(), &f.from_ground(&field.element(q)?)),
                ));
            }
            rows.push(row);
        }
        let mut basis: Vec<ExprMatrix> = Vec::new();
        for vector in field_nullspace(&DomainMatrix::new(rows, f.clone(), Some(a.cols))?)? {
            basis.push(column(field, &vector, x)?);
        }
        out.push(Eigenspace {
            factor: crate::core::subs::subs(factor, unknown, x)?,
            symbol: x.clone(),
            multiplicity: *multiplicity,
            basis,
        });
    }
    Ok(out)
}

#[allow(clippy::type_complexity)]
fn factors_of(a: &ExprMatrix, b: &ExprMatrix, unknown: &Basic) -> PolyResult<(Vec<Basic>, Vec<(Basic, usize)>, usize)> {
    let kernels: Vec<Basic> = kernels_of(&[a, b]);
    if !kernels.iter().all(|k| k.is_symbol()) {
        return Err(PolyError::NotImplemented(
            "pencils of matrices whose entries are not rational functions of symbols".to_string(),
        ));
    }
    let polynomial: Basic = pencil_polynomial(a, b, unknown)?;
    if polynomial == Basic::int(0) {
        return Err(value_error(
            "the pencil is singular: det(A - x*B) vanishes identically".to_string(),
        ));
    }
    let degree: usize = coefficients(&polynomial, unknown)?.len() - 1;
    let mut gens: Vec<Basic> = vec![unknown.clone()];
    gens.extend(kernels.iter().cloned());
    let (_, listed) = crate::polys::polytools::factor_list(&polynomial, &gens)?;
    let mut factors: Vec<(Basic, usize)> = Vec::new();
    for (factor, exponent) in listed {
        if factor.has(unknown) && exponent.is_integer_value() {
            let k: usize = usize::try_from(&numer(exponent.as_rational().expect("an integer")))
                .map_err(|_| value_error("multiplicity".to_string()))?;
            factors.push((factor, k));
        }
    }
    Ok((kernels, factors, degree))
}

/// The generalised eigenspaces of `A v = x B v` (`generalized_eigenspaces`).
pub fn generalized_eigenspaces(a: &ExprMatrix, b: &ExprMatrix, x: &Basic) -> PolyResult<Vec<Eigenspace>> {
    check(a, b)?;
    let unknown: Basic = fresh(a, b);
    let (kernels, factors, _) = factors_of(a, b, &unknown)?;
    if !kernels.is_empty() {
        return spaces(
            &FunctionEntries::new(PolynomialLift::new(&kernels)),
            a,
            b,
            &factors,
            &unknown,
            x,
        );
    }
    spaces(&RationalEntries::new(), a, b, &factors, &unknown, x)
}

fn explicit(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<Vec<(Basic, Eigenspace)>> {
    let mut pairs: Vec<(Basic, Eigenspace)> = Vec::new();
    for space in generalized_eigenspaces(a, b, &fresh(a, b))? {
        let Some(values) = roots_in_radicals(&space.factor, &space.symbol)? else {
            return Err(PolyError::NotImplemented(format!(
                "the roots of {} are not expressed in radicals: see generalized_eigenspaces",
                space.factor
            )));
        };
        for value in values {
            pairs.push((value, space.clone()));
        }
    }
    pairs.sort_by_cached_key(|p| crate::core::sorting::sort_key(&p.0));
    Ok(pairs)
}

/// The generalised eigenvalues with their multiplicities, `oo` for the
/// infinite ones (`generalized_eigenvalues`).
pub fn generalized_eigenvalues(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<Vec<(Basic, usize)>> {
    let n: usize = check(a, b)?;
    let (_, _, degree) = factors_of(a, b, &fresh(a, b))?;
    let mut result: Vec<(Basic, usize)> = Vec::new();
    for (value, space) in explicit(a, b)? {
        match result.iter_mut().find(|(v, _)| *v == value) {
            Some(entry) => entry.1 = space.multiplicity,
            None => result.push((value, space.multiplicity)),
        }
    }
    if degree < n {
        let oo: Basic = Basic::oo();
        match result.iter_mut().find(|(v, _)| *v == oo) {
            Some(entry) => entry.1 = n - degree,
            None => result.push((oo, n - degree)),
        }
    }
    Ok(result)
}

/// `(value, multiplicity, eigenvectors)` (`generalized_eigenvectors`).
#[allow(clippy::type_complexity)]
pub fn generalized_eigenvectors(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<Vec<(Basic, usize, Vec<ExprMatrix>)>> {
    explicit(a, b)?
        .into_iter()
        .map(|(v, s)| Ok((v.clone(), s.multiplicity, s.vectors(&v)?)))
        .collect()
}
