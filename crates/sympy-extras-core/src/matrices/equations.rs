//! Port of `sympy_extras/matrices/equations.py`: linear matrix equations
//! `sum A_k X B_k = C` by vectorisation (`vec(A X B) = (B^T (x) A) vec(X)`):
//! Sylvester, continuous and discrete Lyapunov equations, the commutant.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::basic::{Basic, Head};
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::linalg::{linear_solve, normal, square};

/// A particular solution and a basis of the homogeneous solutions
/// (`MatrixSolution`).
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixSolution {
    pub particular: ExprMatrix,
    pub homogeneous: Vec<ExprMatrix>,
}

impl MatrixSolution {
    pub fn is_unique(&self) -> bool {
        self.homogeneous.is_empty()
    }
}

/// `A (x) B` (`kronecker_product`).
pub fn kronecker_product(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..a.rows {
        for k in 0..b.rows {
            for j in 0..a.cols {
                for l in 0..b.cols {
                    entries.push(crate::core::mul::mul(&[a.at(i, j).clone(), b.at(k, l).clone()])?);
                }
            }
        }
    }
    ExprMatrix::new(a.rows * b.rows, a.cols * b.cols, entries)
}

/// The columns stacked (`vec`).
pub fn vec(m: &ExprMatrix) -> ExprMatrix {
    let mut entries: Vec<Basic> = Vec::new();
    for j in 0..m.cols {
        for i in 0..m.rows {
            entries.push(m.at(i, j).clone());
        }
    }
    ExprMatrix::column(entries)
}

/// The inverse of [`vec`] (`unvec`).
pub fn unvec(v: &ExprMatrix, rows: usize, cols: usize) -> PolyResult<ExprMatrix> {
    if v.cols != 1 || v.rows != rows * cols {
        return Err(value_error(format!("a column of {} entries is expected", rows * cols)));
    }
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            entries.push(v.at(j * rows + i, 0).clone());
        }
    }
    ExprMatrix::new(rows, cols, entries)
}

/// The conjugate transpose (`adjoint`).
pub fn adjoint(m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let transposed: ExprMatrix = m.transpose();
    let i: Basic = Basic::atom(Head::ImaginaryUnit);
    if !transposed.has(&i) {
        return Ok(transposed);
    }
    let rule: HashMap<Basic, Basic> = [(i.clone(), crate::core::mul::neg(&i)?)].into_iter().collect();
    transposed
        .applyfunc(&|e| crate::core::function::xreplace_evaluated(e, &rule))?
        .applyfunc(&normal)
}

/// The solutions of `sum A_k X B_k = C` (`linear_matrix_equation`).
pub fn linear_matrix_equation(
    terms: &[(ExprMatrix, ExprMatrix)],
    c: &ExprMatrix,
) -> PolyResult<Option<MatrixSolution>> {
    if terms.is_empty() {
        return Err(value_error("an equation without terms".to_string()));
    }
    let (rows, cols) = (terms[0].0.cols, terms[0].1.rows);
    for (a, b) in terms {
        if a.cols != rows || b.rows != cols || a.rows != c.rows || b.cols != c.cols {
            return Err(value_error("the shapes of the terms do not agree".to_string()));
        }
    }
    let mut system: ExprMatrix = kronecker_product(&terms[0].1.transpose(), &terms[0].0)?;
    for (a, b) in &terms[1..] {
        system = system.add(&kronecker_product(&b.transpose(), a)?)?;
    }
    let Some(solution) = linear_solve(&system, &vec(c))? else {
        return Ok(None);
    };
    Ok(Some(MatrixSolution {
        particular: unvec(&solution.particular, rows, cols)?,
        homogeneous: solution
            .nullspace
            .iter()
            .map(|v| unvec(v, rows, cols))
            .collect::<PolyResult<_>>()?,
    }))
}

/// `A X + X B = C` (`sylvester_solve`).
pub fn sylvester_solve(a: &ExprMatrix, b: &ExprMatrix, c: &ExprMatrix) -> PolyResult<Option<MatrixSolution>> {
    let (m, n) = (square(a)?, square(b)?);
    linear_matrix_equation(&[(a.clone(), ExprMatrix::eye(n)), (ExprMatrix::eye(m), b.clone())], c)
}

/// `A X + X A^H = -Q` (`lyapunov_solve`).
pub fn lyapunov_solve(a: &ExprMatrix, q: &ExprMatrix) -> PolyResult<Option<MatrixSolution>> {
    let identity: ExprMatrix = ExprMatrix::eye(square(a)?);
    linear_matrix_equation(&[(a.clone(), identity.clone()), (identity, adjoint(a)?)], &q.neg()?)
}

/// `A X A^H - X = -Q` (`discrete_lyapunov_solve`).
pub fn discrete_lyapunov_solve(a: &ExprMatrix, q: &ExprMatrix) -> PolyResult<Option<MatrixSolution>> {
    let identity: ExprMatrix = ExprMatrix::eye(square(a)?);
    linear_matrix_equation(&[(a.clone(), adjoint(a)?), (identity.neg()?, identity)], &q.neg()?)
}

/// A basis of the matrices commuting with `A` (`commuting_matrices`).
pub fn commuting_matrices(a: &ExprMatrix) -> PolyResult<Vec<ExprMatrix>> {
    let n: usize = square(a)?;
    let identity: ExprMatrix = ExprMatrix::eye(n);
    match linear_matrix_equation(
        &[(a.clone(), identity.clone()), (identity.neg()?, a.clone())],
        &ExprMatrix::zeros(n, n),
    )? {
        Some(solution) => Ok(solution.homogeneous),
        None => Err(PolyError::Arithmetic(
            "a homogeneous system without solution".to_string(),
        )),
    }
}
