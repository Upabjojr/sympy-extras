//! Port of `sympy_extras/calculus/interpolation.py`: exact interpolation
//! and fitting: divided differences and the Newton form, Hermite
//! interpolation with derivatives, cubic splines (natural, clamped,
//! not-a-knot, periodic) as a `Piecewise`, least squares in a basis.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::is;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::linalg::{linear_solve, normal};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}

fn newton(nodes: &[Basic], coefficients: &[Basic], x: &Basic) -> R {
    let mut result: Basic = int(0);
    for (node, c) in nodes.iter().rev().zip(coefficients.iter().rev()) {
        result = a2(&m2(&result, &s2(x, node)?)?, c)?;
    }
    normal(&expand(&result)?)
}

/// The divided differences `f[x0], f[x0, x1], ...` (`divided_differences`).
pub fn divided_differences(points: &[(Basic, Basic)]) -> PolyResult<Vec<Basic>> {
    let nodes: Vec<Basic> = points.iter().map(|(p, _)| p.clone()).collect();
    let mut table: Vec<Basic> = points.iter().map(|(_, v)| v.clone()).collect();
    let n: usize = nodes.len();
    for i in 0..n {
        for j in 0..i {
            if normal(&s2(&nodes[i], &nodes[j])?)? == int(0) {
                return Err(value_error("the nodes of interpolation are not distinct".to_string()));
            }
        }
    }
    let mut result: Vec<Basic> = Vec::new();
    for level in 0..n {
        result.push(table[0].clone());
        let mut next: Vec<Basic> = Vec::new();
        for i in 0..n - level - 1 {
            next.push(normal(&d2(
                &s2(&table[i + 1], &table[i])?,
                &s2(&nodes[i + level + 1], &nodes[i])?,
            )?)?);
        }
        table = next;
    }
    Ok(result)
}

/// The interpolating polynomial in the Newton form (`interpolating_polynomial`).
pub fn interpolating_polynomial(points: &[(Basic, Basic)], x: &Basic) -> R {
    let nodes: Vec<Basic> = points.iter().map(|(p, _)| p.clone()).collect();
    newton(&nodes, &divided_differences(points)?, x)
}

/// The polynomial with the given values and derivatives at the nodes
/// (`hermite_interpolation`).
pub fn hermite_interpolation(data: &[(Basic, Vec<Basic>)], x: &Basic) -> R {
    let mut nodes: Vec<Basic> = Vec::new();
    let mut owner: Vec<usize> = Vec::new();
    let mut derivatives: Vec<Vec<Basic>> = Vec::new();
    for (index, (node, values)) in data.iter().enumerate() {
        if values.is_empty() {
            return Err(value_error("a node without a value".to_string()));
        }
        derivatives.push(values.clone());
        for _ in 0..values.len() {
            nodes.push(node.clone());
            owner.push(index);
        }
    }
    for (i, (first, _)) in data.iter().enumerate() {
        for (second, _) in &data[..i] {
            if normal(&s2(first, second)?)? == int(0) {
                return Err(value_error("the nodes of interpolation are not distinct".to_string()));
            }
        }
    }
    let n: usize = nodes.len();
    let mut table: Vec<Basic> = (0..n).map(|i| derivatives[owner[i]][0].clone()).collect();
    let mut coefficients: Vec<Basic> = Vec::new();
    let mut factorial: i64 = 1;
    for level in 0..n {
        coefficients.push(table[0].clone());
        if level + 1 == n {
            break;
        }
        factorial *= level as i64 + 1;
        let mut following: Vec<Basic> = Vec::new();
        for i in 0..n - level - 1 {
            if owner[i] == owner[i + level + 1] {
                following.push(normal(&d2(&derivatives[owner[i]][level + 1], &int(factorial))?)?);
            } else {
                following.push(normal(&d2(
                    &s2(&table[i + 1], &table[i])?,
                    &s2(&nodes[i + level + 1], &nodes[i])?,
                )?)?);
            }
        }
        table = following;
    }
    newton(&nodes, &coefficients, x)
}

/// The boundary condition of a cubic spline.
#[derive(Debug, Clone)]
pub enum Boundary {
    Natural,
    Clamped(Option<(Basic, Basic)>),
    NotAKnot,
    Periodic,
    Unknown(String),
}

/// The cubic spline through the points (`cubic_spline`).
pub fn cubic_spline(points: &[(Basic, Basic)], x: &Basic, boundary: &Boundary) -> R {
    let nodes: Vec<Basic> = points.iter().map(|(p, _)| p.clone()).collect();
    let values: Vec<Basic> = points.iter().map(|(_, v)| v.clone()).collect();
    if nodes.len() < 2 {
        return Err(value_error("a spline needs two points at least".to_string()));
    }
    let n: usize = nodes.len() - 1;
    let h: Vec<Basic> = (0..n)
        .map(|i| normal(&s2(&nodes[i + 1], &nodes[i])?))
        .collect::<PolyResult<_>>()?;
    if h.iter().any(|step| is(step, Predicate::positive) != Some(true)) {
        return Err(value_error(
            "the nodes must be increasing (with known signs of their differences)".to_string(),
        ));
    }
    let slope: Vec<Basic> = (0..n)
        .map(|i| normal(&d2(&s2(&values[i + 1], &values[i])?, &h[i])?))
        .collect::<PolyResult<_>>()?;
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    let mut rhs: Vec<Basic> = Vec::new();
    let mut row = |entries: &[(usize, Basic)], value: Basic| -> PolyResult<()> {
        let mut line: Vec<Basic> = vec![int(0); n + 1];
        for (position, entry) in entries {
            line[*position] = a2(&line[*position], entry)?;
        }
        rows.push(line);
        rhs.push(value);
        Ok(())
    };
    for i in 1..n {
        row(
            &[
                (i - 1, h[i - 1].clone()),
                (i, m2(&int(2), &a2(&h[i - 1], &h[i])?)?),
                (i + 1, h[i].clone()),
            ],
            m2(&int(6), &s2(&slope[i], &slope[i - 1])?)?,
        )?;
    }
    match boundary {
        Boundary::Natural => {
            row(&[(0, int(1))], int(0))?;
            row(&[(n, int(1))], int(0))?;
        }
        Boundary::Clamped(slopes) => {
            let Some((first, last)) = slopes else {
                return Err(value_error(
                    "a clamped spline needs the slopes at the two ends".to_string(),
                ));
            };
            row(
                &[(0, m2(&int(2), &h[0])?), (1, h[0].clone())],
                m2(&int(6), &s2(&slope[0], first)?)?,
            )?;
            row(
                &[(n - 1, h[n - 1].clone()), (n, m2(&int(2), &h[n - 1])?)],
                m2(&int(6), &s2(last, &slope[n - 1])?)?,
            )?;
        }
        Boundary::NotAKnot => {
            if n < 3 {
                return interpolating_polynomial(points, x);
            }
            row(
                &[
                    (0, h[1].clone()),
                    (1, crate::core::mul::neg(&a2(&h[0], &h[1])?)?),
                    (2, h[0].clone()),
                ],
                int(0),
            )?;
            row(
                &[
                    (n - 2, h[n - 1].clone()),
                    (n - 1, crate::core::mul::neg(&a2(&h[n - 2], &h[n - 1])?)?),
                    (n, h[n - 2].clone()),
                ],
                int(0),
            )?;
        }
        Boundary::Periodic => {
            if normal(&s2(&values[0], &values[n])?)? != int(0) {
                return Err(value_error(
                    "a periodic spline needs equal values at the two ends".to_string(),
                ));
            }
            row(&[(0, int(1)), (n, int(-1))], int(0))?;
            row(
                &[
                    (0, m2(&int(2), &h[0])?),
                    (1, h[0].clone()),
                    (n - 1, h[n - 1].clone()),
                    (n, m2(&int(2), &h[n - 1])?),
                ],
                m2(&int(6), &s2(&slope[0], &slope[n - 1])?)?,
            )?;
        }
        Boundary::Unknown(name) => return Err(value_error(format!("unknown boundary condition '{}'", name))),
    }
    let solution = linear_solve(&ExprMatrix::from_rows(&rows)?, &ExprMatrix::column(rhs))?;
    let solution = match solution {
        Some(s) if s.is_unique() => s,
        _ => return Err(PolyError::Arithmetic("the spline is not determined".to_string())),
    };
    let mm: Vec<Basic> = (0..=n).map(|i| solution.particular.at(i, 0).clone()).collect();
    let mut pieces: Vec<Basic> = Vec::new();
    for i in 0..n {
        let t: Basic = s2(x, &nodes[i])?;
        let inner: Basic = d2(&m2(&h[i], &a2(&m2(&int(2), &mm[i])?, &mm[i + 1])?)?, &int(6))?;
        let linear: Basic = m2(&s2(&slope[i], &inner)?, &t)?;
        let quadratic: Basic = d2(&m2(&mm[i], &pw(&t, &int(2))?)?, &int(2))?;
        let cubic_term: Basic = d2(&m2(&s2(&mm[i + 1], &mm[i])?, &pw(&t, &int(3))?)?, &m2(&int(6), &h[i])?)?;
        let cubic: Basic = a2(&a2(&a2(&values[i], &linear)?, &quadratic)?, &cubic_term)?;
        pieces.push(normal(&expand(&cubic)?)?);
    }
    let mut same: bool = true;
    for piece in &pieces {
        if normal(&s2(piece, &pieces[0])?)? != int(0) {
            same = false;
            break;
        }
    }
    if same {
        return Ok(pieces[0].clone());
    }
    let mut branches: Vec<(Basic, Basic)> = Vec::new();
    for (i, piece) in pieces[..pieces.len() - 1].iter().enumerate() {
        branches.push((
            piece.clone(),
            Basic::new(Head::Relational(RelOp::Le), vec![x.clone(), nodes[i + 1].clone()]),
        ));
    }
    branches.push((pieces[pieces.len() - 1].clone(), crate::core::logic::boolean(true)));
    crate::functions::elementary::piecewise::piecewise(&branches)
}

/// The least squares combination of the basis functions (`least_squares`).
pub fn least_squares(points: &[(Basic, Basic)], x: &Basic, basis: &[Basic]) -> R {
    if basis.is_empty() {
        return Err(value_error("an empty basis".to_string()));
    }
    let mut design: Vec<Vec<Basic>> = Vec::new();
    for (p, _) in points {
        design.push(
            basis
                .iter()
                .map(|f| normal(&crate::core::subs::subs(f, x, p)?))
                .collect::<PolyResult<_>>()?,
        );
    }
    let k: usize = basis.len();
    let mut gram: Vec<Vec<Basic>> = Vec::new();
    for i in 0..k {
        let mut line: Vec<Basic> = Vec::new();
        for j in 0..k {
            let terms: Vec<Basic> = design
                .iter()
                .map(|row| m2(&row[i], &row[j]))
                .collect::<PolyResult<_>>()?;
            line.push(normal(&crate::core::add::add(&terms)?)?);
        }
        gram.push(line);
    }
    let mut moments: Vec<Basic> = Vec::new();
    for i in 0..k {
        let terms: Vec<Basic> = design
            .iter()
            .zip(points.iter())
            .map(|(row, (_, v))| m2(&row[i], v))
            .collect::<PolyResult<_>>()?;
        moments.push(normal(&crate::core::add::add(&terms)?)?);
    }
    let solution = linear_solve(&ExprMatrix::from_rows(&gram)?, &ExprMatrix::column(moments))?;
    let solution = match solution {
        Some(s) if s.is_unique() => s,
        _ => {
            return Err(value_error(
                "the least squares fit is not unique: the functions are dependent on these abscissas".to_string(),
            ))
        }
    };
    let terms: Vec<Basic> = (0..k)
        .map(|i| m2(solution.particular.at(i, 0), &basis[i]))
        .collect::<PolyResult<_>>()?;
    crate::core::add::add(&terms)
}

/// The least squares polynomial of a degree (`polynomial_fit`).
pub fn polynomial_fit(points: &[(Basic, Basic)], x: &Basic, degree: i64) -> R {
    if degree < 0 {
        return Err(value_error("a negative degree".to_string()));
    }
    let basis: Vec<Basic> = (0..=degree).map(|k| pw(x, &int(k))).collect::<PolyResult<_>>()?;
    least_squares(points, x, &basis)
}
