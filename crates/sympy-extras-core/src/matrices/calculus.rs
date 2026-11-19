//! Port of `sympy_extras/matrices/calculus.py`: Jacobian and gradient,
//! the (bordered) Hessian, the Wronskian, and the classification of a
//! critical point by the inertia of the Hessian.

#![allow(clippy::mutable_key_type, clippy::needless_range_loop)]

use std::collections::HashMap;

use crate::core::basic::Basic;
use crate::error::{value_error, PolyResult};
use crate::matrices::dense::ExprMatrix;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn diff(e: &Basic, v: &Basic) -> R {
    crate::core::function::derivative(e, v)
}
fn tidy(e: &Basic) -> R {
    crate::simplify::normalform::normal_form(e)
}

/// The Jacobian matrix of the functions (`jacobian`).
pub fn jacobian(functions: &[Basic], variables: &[Basic]) -> PolyResult<ExprMatrix> {
    let mut entries: Vec<Basic> = Vec::new();
    for f in functions {
        for v in variables {
            entries.push(diff(f, v)?);
        }
    }
    ExprMatrix::new(functions.len(), variables.len(), entries)
}

/// The gradient as a column (`gradient`).
pub fn gradient(f: &Basic, variables: &[Basic]) -> PolyResult<ExprMatrix> {
    Ok(ExprMatrix::column(
        variables.iter().map(|v| diff(f, v)).collect::<PolyResult<_>>()?,
    ))
}

/// The Hessian, bordered by the gradients of the constraints (`hessian`).
pub fn hessian(f: &Basic, variables: &[Basic], constraints: &[Basic]) -> PolyResult<ExprMatrix> {
    let n: usize = variables.len();
    let first: Vec<Basic> = variables.iter().map(|v| diff(f, v)).collect::<PolyResult<_>>()?;
    let h: Vec<Vec<Basic>> = first
        .iter()
        .map(|d| variables.iter().map(|v| diff(d, v)).collect::<PolyResult<_>>())
        .collect::<PolyResult<_>>()?;
    if constraints.is_empty() {
        return ExprMatrix::new(n, n, h.into_iter().flatten().collect());
    }
    let border: Vec<Vec<Basic>> = constraints
        .iter()
        .map(|g| variables.iter().map(|v| diff(g, v)).collect::<PolyResult<_>>())
        .collect::<PolyResult<_>>()?;
    let m: usize = border.len();
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for i in 0..m {
        let mut row: Vec<Basic> = vec![int(0); m];
        row.extend(border[i].iter().cloned());
        rows.push(row);
    }
    for i in 0..n {
        let mut row: Vec<Basic> = (0..m).map(|k| border[k][i].clone()).collect();
        row.extend(h[i].iter().cloned());
        rows.push(row);
    }
    ExprMatrix::new(m + n, m + n, rows.into_iter().flatten().collect())
}

/// The Wronskian of the functions (`wronskian`).
pub fn wronskian(functions: &[Basic], x: &Basic) -> R {
    let n: usize = functions.len();
    if n == 0 {
        return Ok(int(1));
    }
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    let mut current: Vec<Basic> = functions.to_vec();
    for _ in 0..n {
        let next: Vec<Basic> = current.iter().map(|f| diff(f, x)).collect::<PolyResult<_>>()?;
        rows.push(std::mem::replace(&mut current, next));
    }
    tidy(&crate::matrices::linalg::det(&ExprMatrix::new(
        n,
        n,
        rows.into_iter().flatten().collect(),
    )?)?)
}

/// `"minimum"`, `"maximum"`, `"saddle"` or `None` (degenerate) at a
/// critical point (`classify_critical_point`).
pub fn classify_critical_point(f: &Basic, variables: &[Basic], point: &[Basic]) -> PolyResult<Option<&'static str>> {
    if point.len() != variables.len() {
        return Err(value_error(format!(
            "{} coordinates for {} variables",
            point.len(),
            variables.len()
        )));
    }
    let rule: HashMap<Basic, Basic> = variables.iter().cloned().zip(point.iter().cloned()).collect();
    let at_point = |e: &Basic| -> R {
        tidy(&crate::core::function::doit(
            &crate::core::function::xreplace_evaluated(e, &rule)?,
        )?)
    };
    for v in variables {
        if at_point(&diff(f, v)?)? != int(0) {
            let shown: Vec<String> = point.iter().map(|p| p.to_string()).collect();
            return Err(value_error(format!(
                "[{}] is not a critical point: the gradient does not vanish there",
                shown.join(", ")
            )));
        }
    }
    let h: ExprMatrix = hessian(f, variables, &[])?.applyfunc(&at_point)?;
    let (positive, negative, zero) = crate::matrices::decompositions::inertia(&h)?;
    if positive > 0 && negative > 0 {
        return Ok(Some("saddle"));
    }
    if zero > 0 {
        return Ok(None);
    }
    Ok(Some(if positive > 0 { "minimum" } else { "maximum" }))
}
