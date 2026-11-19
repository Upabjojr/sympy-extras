//! Port of `sympy_extras/matrices/statespace.py`: linear time-invariant
//! systems `x' = A x + B u, y = C x + D u`: controllability and
//! observability matrices, transfer functions, pole placement by
//! Ackermann's formula, the controllable canonical realisation of a
//! scalar transfer function, the Kalman decompositions and minimal
//! realisations.

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::coefficients;
use crate::matrices::linalg::{inverse, normal, rank, rref, square};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn shape(m: &ExprMatrix) -> String {
    format!("({}, {})", m.rows, m.cols)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}

fn hstack(blocks: &[ExprMatrix]) -> PolyResult<ExprMatrix> {
    let rows: usize = blocks[0].rows;
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..rows {
        for b in blocks {
            entries.extend(b.row(i).iter().cloned());
        }
    }
    ExprMatrix::new(rows, blocks.iter().map(|b| b.cols).sum(), entries)
}

/// `[B, A B, ..., A**(n-1) B]` (`controllability_matrix`).
pub fn controllability_matrix(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let n: usize = square(a)?;
    if b.rows != n {
        return Err(value_error(format!(
            "B has {} rows for a state of dimension {}",
            b.rows, n
        )));
    }
    let mut blocks: Vec<ExprMatrix> = vec![b.clone()];
    for _ in 0..n.saturating_sub(1) {
        let next: ExprMatrix = a.matmul(&blocks[blocks.len() - 1])?.applyfunc(&normal)?;
        blocks.push(next);
    }
    if n > 0 {
        hstack(&blocks)
    } else {
        Ok(b.clone())
    }
}

/// `[C; C A; ...; C A**(n-1)]` (`observability_matrix`).
pub fn observability_matrix(a: &ExprMatrix, c: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let n: usize = square(a)?;
    if c.cols != n {
        return Err(value_error(format!(
            "C has {} columns for a state of dimension {}",
            c.cols, n
        )));
    }
    Ok(controllability_matrix(&a.transpose(), &c.transpose())?.transpose())
}

pub fn is_controllable(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<bool> {
    Ok(rank(&controllability_matrix(a, b)?)? == a.rows)
}

pub fn is_observable(a: &ExprMatrix, c: &ExprMatrix) -> PolyResult<bool> {
    Ok(rank(&observability_matrix(a, c)?)? == a.rows)
}

/// `C (s I - A)**-1 B + D` (`transfer_function`).
pub fn transfer_function(
    a: &ExprMatrix,
    b: &ExprMatrix,
    c: &ExprMatrix,
    s: &Basic,
    d: Option<&ExprMatrix>,
) -> PolyResult<ExprMatrix> {
    let n: usize = square(a)?;
    if b.rows != n || c.cols != n {
        return Err(value_error(format!(
            "shapes: A {}, B {}, C {}",
            shape(a),
            shape(b),
            shape(c)
        )));
    }
    if [a, b, c].iter().any(|m| m.entries.iter().any(|e| e.has(s))) {
        return Err(value_error(format!("{} occurs in the matrices of the system", s)));
    }
    let mut result: ExprMatrix = if n == 0 {
        ExprMatrix::zeros(c.rows, b.cols)
    } else {
        let resolvent: ExprMatrix = inverse(&ExprMatrix::eye(n).scale(s)?.sub(a)?)?;
        c.matmul(&resolvent)?.matmul(b)?
    };
    if let Some(d) = d {
        if d.shape() != (c.rows, b.cols) {
            return Err(value_error(format!(
                "D has the shape {} for {} outputs and {} inputs",
                shape(d),
                c.rows,
                b.cols
            )));
        }
        result = result.add(d)?;
    }
    result.applyfunc(&normal)
}

/// The target of a pole placement: a polynomial in `s` or its roots.
#[derive(Debug, Clone)]
pub enum Target {
    Polynomial(Basic, Option<Basic>),
    Poles(Vec<Basic>),
}

/// The feedback `K` with `det(s I - (A - B K))` the target polynomial
/// (Ackermann's formula, `pole_placement`).
pub fn pole_placement(a: &ExprMatrix, b: &ExprMatrix, target: &Target) -> PolyResult<ExprMatrix> {
    let n: usize = square(a)?;
    if b.shape() != (n, 1) {
        return Err(value_error(format!(
            "pole placement by the formula of Ackermann is for one input: B has the shape {}",
            shape(b)
        )));
    }
    let mut cs: Vec<Basic> = match target {
        Target::Polynomial(p, s) => {
            let Some(s) = s else {
                return Err(value_error("the variable of the polynomial is needed".to_string()));
            };
            let polynomial: Basic = expand(p)?;
            let mut cs: Vec<Basic> = coefficients(&polynomial, s)?;
            if cs[0] != int(1) {
                let lead: Basic = cs[0].clone();
                cs = cs
                    .iter()
                    .map(|c| normal(&crate::core::mul::div(c, &lead)?))
                    .collect::<PolyResult<_>>()?;
            }
            cs
        }
        Target::Poles(poles) => {
            let mut cs: Vec<Basic> = vec![int(1)];
            for root in poles {
                let mut shifted: Vec<Basic> = cs.clone();
                shifted.push(int(0));
                let mut next: Vec<Basic> = Vec::new();
                for i in 0..shifted.len() {
                    let t: Basic = if i > 0 {
                        crate::core::mul::mul(&[root.clone(), cs[i - 1].clone()])?
                    } else {
                        int(0)
                    };
                    next.push(crate::core::add::sub(&shifted[i], &t)?);
                }
                cs = next;
            }
            cs.iter().map(normal).collect::<PolyResult<_>>()?
        }
    };
    if cs.len() != n + 1 {
        return Err(value_error(format!("a polynomial of degree {} is needed", n)));
    }
    if n == 0 {
        return Ok(ExprMatrix::zeros(1, 0));
    }
    let controllability: ExprMatrix = controllability_matrix(a, b)?;
    let inverted: ExprMatrix = match inverse(&controllability) {
        Ok(m) => m,
        Err(PolyError::Value(_)) => return Err(value_error("the pair (A, B) is not controllable".to_string())),
        Err(e) => return Err(e),
    };
    let first: Basic = cs.remove(0);
    let mut value: ExprMatrix = ExprMatrix::eye(n).scale(&first)?;
    for c in &cs {
        value = value
            .matmul(a)?
            .add(&ExprMatrix::eye(n).scale(c)?)?
            .applyfunc(&normal)?;
    }
    let last: ExprMatrix = ExprMatrix::new(1, n, inverted.row(n - 1).to_vec())?;
    last.matmul(&value)?.applyfunc(&normal)
}

/// The controllable canonical realisation of a proper scalar transfer
/// function (`realization`).
pub fn realization(g: &Basic, s: &Basic) -> PolyResult<(ExprMatrix, ExprMatrix, ExprMatrix, ExprMatrix)> {
    let (numerator, denominator) = crate::simplify::radsimp::fraction(&normal(g)?)?;
    let expanded: Basic = expand(&numerator)?;
    let top: Vec<Basic> = if expanded == int(0) {
        vec![int(0)]
    } else {
        coefficients(&expanded, s)?
    };
    let bottom: Vec<Basic> = coefficients(&expand(&denominator)?, s)?;
    let n: usize = bottom.len() - 1;
    if top.len() - 1 > n {
        return Err(value_error(format!(
            "{} is not proper: the degree of the numerator exceeds that of the denominator",
            g
        )));
    }
    let leading: Basic = bottom[0].clone();
    let a: Vec<Basic> = bottom
        .iter()
        .map(|c| normal(&crate::core::mul::div(c, &leading)?))
        .collect::<PolyResult<_>>()?;
    let mut b: Vec<Basic> = vec![int(0); n + 1 - top.len()];
    for c in &top {
        b.push(normal(&crate::core::mul::div(c, &leading)?)?);
    }
    let direct: Basic = b[0].clone();
    let rest: Vec<Basic> = (1..=n)
        .map(|k| {
            normal(&crate::core::add::sub(
                &b[k],
                &crate::core::mul::mul(&[direct.clone(), a[k].clone()])?,
            )?)
        })
        .collect::<PolyResult<_>>()?;
    let mut am: Vec<Basic> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            am.push(if i == n - 1 {
                crate::core::mul::neg(&a[n - j])?
            } else if j == i + 1 {
                int(1)
            } else {
                int(0)
            });
        }
    }
    let bm: Vec<Basic> = (0..n).map(|i| if i == n - 1 { int(1) } else { int(0) }).collect();
    let cm: Vec<Basic> = (0..n).map(|j| rest[n - 1 - j].clone()).collect();
    Ok((
        ExprMatrix::new(n, n, am)?,
        ExprMatrix::new(n, 1, bm)?,
        ExprMatrix::new(1, n, cm)?,
        ExprMatrix::new(1, 1, vec![direct])?,
    ))
}

fn completed_basis(m: &ExprMatrix) -> PolyResult<(ExprMatrix, usize)> {
    let n: usize = m.rows;
    let extended: ExprMatrix = if m.cols > 0 {
        hstack(&[m.clone(), ExprMatrix::eye(n)])?
    } else {
        ExprMatrix::eye(n)
    };
    let (_, pivots) = rref(&extended)?;
    let r: usize = pivots.iter().filter(|c| **c < m.cols).count();
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..n {
        for j in &pivots {
            entries.push(extended.at(i, *j).clone());
        }
    }
    Ok((ExprMatrix::new(n, n, entries)?, r))
}

/// `(T, T**-1 A T, T**-1 B, r)` with the controllable part first
/// (`controllable_decomposition`).
pub fn controllable_decomposition(
    a: &ExprMatrix,
    b: &ExprMatrix,
) -> PolyResult<(ExprMatrix, ExprMatrix, ExprMatrix, usize)> {
    let (t, r) = completed_basis(&controllability_matrix(a, b)?)?;
    let inverted: ExprMatrix = inverse(&t)?;
    let at: ExprMatrix = inverted.matmul(a)?.matmul(&t)?.applyfunc(&normal)?;
    let bt: ExprMatrix = inverted.matmul(b)?.applyfunc(&normal)?;
    Ok((t, at, bt, r))
}

/// The dual decomposition (`observable_decomposition`).
pub fn observable_decomposition(
    a: &ExprMatrix,
    c: &ExprMatrix,
) -> PolyResult<(ExprMatrix, ExprMatrix, ExprMatrix, usize)> {
    let (s, _, _, r) = controllable_decomposition(&a.transpose(), &c.transpose())?;
    let t: ExprMatrix = inverse(&s.transpose())?;
    let at: ExprMatrix = s.transpose().matmul(a)?.matmul(&t)?.applyfunc(&normal)?;
    let ct: ExprMatrix = c.matmul(&t)?.applyfunc(&normal)?;
    Ok((t, at, ct, r))
}

fn block(m: &ExprMatrix, rows: usize, cols: usize) -> PolyResult<ExprMatrix> {
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            entries.push(m.at(i, j).clone());
        }
    }
    ExprMatrix::new(rows, cols, entries)
}

/// A minimal realisation (`minimal_realization`).
pub fn minimal_realization(
    a: &ExprMatrix,
    b: &ExprMatrix,
    c: &ExprMatrix,
) -> PolyResult<(ExprMatrix, ExprMatrix, ExprMatrix)> {
    if c.cols != a.rows {
        return Err(value_error(format!(
            "C has {} columns for a state of dimension {}",
            c.cols, a.rows
        )));
    }
    let (t, a1, b1, r) = controllable_decomposition(a, b)?;
    let c1: ExprMatrix = c.matmul(&t)?.applyfunc(&normal)?;
    let ac: ExprMatrix = block(&a1, r, r)?;
    let bc: ExprMatrix = block(&b1, r, b.cols)?;
    let cc: ExprMatrix = block(&c1, c.rows, r)?;
    let (_, a2, c2, q) = observable_decomposition(&ac, &cc)?;
    let (s, _, _, _) = controllable_decomposition(&ac.transpose(), &cc.transpose())?;
    let b2: ExprMatrix = s.transpose().matmul(&bc)?.applyfunc(&normal)?;
    Ok((block(&a2, q, q)?, block(&b2, q, b.cols)?, block(&c2, c.rows, q)?))
}
