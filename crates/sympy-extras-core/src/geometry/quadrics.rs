//! Port of `sympy_extras/geometry/quadrics.py`: classification of conics
//! and quadrics `x^T A x + 2 b^T x + c = 0` by the ranks and signatures of
//! `A` and of the extended matrix `[[A, b], [b^T, c]]` (Sylvester's
//! inertia), with the center, and for the central conics the canonical
//! form, the semi-axes and the eccentricity, for the parabola its
//! parameter.

use crate::core::assumptions::Predicate;
use crate::core::basic::Basic;
use crate::core::expr::is;
use crate::error::{value_error, PolyResult};
use crate::matrices::decompositions::inertia;
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::sqrt;
use crate::matrices::linalg::{det, linear_solve, normal};

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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn shown(v: &[Basic]) -> String {
    format!(
        "[{}]",
        v.iter().map(|s| s.to_string()).collect::<Vec<String>>().join(", ")
    )
}

/// `(A, M)`: the matrix of the quadratic part and the extended matrix
/// (`quadric_matrices`).
pub fn quadric_matrices(equation: &Basic, variables: &[Basic]) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let polynomial: Basic = crate::core::function::expand_default(equation)?;
    let n: usize = variables.len();
    let half: Basic = d2(&int(1), &int(2))?;
    let at_origin = |e: &Basic| -> R {
        let mut e: Basic = e.clone();
        for v in variables {
            e = crate::core::subs::subs(&e, v, &int(0))?;
        }
        Ok(e)
    };
    let first: Vec<Basic> = variables
        .iter()
        .map(|v| crate::core::function::derivative(&polynomial, v))
        .collect::<PolyResult<_>>()?;
    let mut a: Vec<Vec<Basic>> = Vec::new();
    for f in &first {
        a.push(
            variables
                .iter()
                .map(|v| normal(&m2(&crate::core::function::derivative(f, v)?, &half)?))
                .collect::<PolyResult<_>>()?,
        );
    }
    let b: Vec<Basic> = first
        .iter()
        .map(|f| normal(&m2(&at_origin(f)?, &half)?))
        .collect::<PolyResult<_>>()?;
    let c: Basic = normal(&at_origin(&polynomial)?)?;
    let not_quadratic = || {
        value_error(format!(
            "{} is not a polynomial of degree at most two in {}",
            equation,
            shown(variables)
        ))
    };
    if a.iter()
        .flatten()
        .any(|e| e.free_symbols().iter().any(|s| variables.contains(s)))
    {
        return Err(not_quadratic());
    }
    let mut rebuilt: Basic = c.clone();
    for i in 0..n {
        rebuilt = a2(&rebuilt, &m2(&m2(&int(2), &b[i])?, &variables[i])?)?;
        for j in 0..n {
            rebuilt = a2(&rebuilt, &m2(&m2(&a[i][j], &variables[i])?, &variables[j])?)?;
        }
    }
    if normal(&s2(&rebuilt, &polynomial)?)? != int(0) {
        return Err(not_quadratic());
    }
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for i in 0..n {
        let mut row: Vec<Basic> = a[i].clone();
        row.push(b[i].clone());
        rows.push(row);
    }
    let mut last: Vec<Basic> = b.clone();
    last.push(c);
    rows.push(last);
    Ok((ExprMatrix::from_rows(&a)?, ExprMatrix::from_rows(&rows)?))
}

fn signature(m: &ExprMatrix) -> PolyResult<(usize, usize)> {
    let (positive, negative, _) = inertia(m)?;
    Ok((positive + negative, positive.abs_diff(negative)))
}

/// A classified conic (`Conic`).
#[derive(Debug, Clone)]
pub struct Conic {
    pub kind: String,
    pub center: Option<Vec<Basic>>,
    pub semi_axes: Option<(Basic, Basic)>,
    pub eccentricity: Option<Basic>,
    pub parameter: Option<Basic>,
    pub canonical: Option<(Basic, Basic, Basic)>,
}

/// A classified quadric (`Quadric`).
#[derive(Debug, Clone)]
pub struct Quadric {
    pub kind: String,
    pub center: Option<Vec<Basic>>,
}

fn center(a: &ExprMatrix, m: &ExprMatrix) -> PolyResult<Option<Vec<Basic>>> {
    let n: usize = a.rows;
    if det(a)? == int(0) {
        return Ok(None);
    }
    let rhs: Vec<Basic> = (0..n).map(|i| neg(m.at(i, n))).collect::<PolyResult<_>>()?;
    let Some(solution) = linear_solve(a, &ExprMatrix::column(rhs))? else {
        return Ok(None);
    };
    Ok(Some((0..n).map(|i| solution.particular.at(i, 0).clone()).collect()))
}

fn conic_kind(a: &ExprMatrix, m: &ExprMatrix) -> PolyResult<&'static str> {
    let (rank_a, sign_a) = signature(a)?;
    let (rank_m, sign_m) = signature(m)?;
    Ok(match rank_m {
        3 => {
            if rank_a == 2 {
                if sign_a == 0 {
                    "hyperbola"
                } else if sign_m == 3 {
                    "empty set"
                } else if *a.at(0, 1) == int(0) && normal(&s2(a.at(0, 0), a.at(1, 1))?)? == int(0) {
                    "circle"
                } else {
                    "ellipse"
                }
            } else {
                "parabola"
            }
        }
        2 => match rank_a {
            2 => {
                if sign_a == 2 {
                    "point"
                } else {
                    "two intersecting lines"
                }
            }
            1 => {
                if sign_m == 0 {
                    "two parallel lines"
                } else {
                    "empty set"
                }
            }
            _ => "line",
        },
        1 => {
            if rank_a == 1 {
                "double line"
            } else {
                "empty set"
            }
        }
        _ => "plane",
    })
}

/// The kind of a conic with its invariants (`classify_conic`).
pub fn classify_conic(equation: &Basic, x: &Basic, y: &Basic) -> PolyResult<Conic> {
    let (a, m) = quadric_matrices(equation, &[x.clone(), y.clone()])?;
    let kind: &str = conic_kind(&a, &m)?;
    let c = center(&a, &m)?;
    let mut conic = Conic {
        kind: kind.to_string(),
        center: c,
        semi_axes: None,
        eccentricity: None,
        parameter: None,
        canonical: None,
    };
    let positive = |e: &Basic| is(e, Predicate::positive) == Some(true);
    if matches!(kind, "ellipse" | "circle" | "hyperbola") {
        let (delta, big) = (det(&a)?, det(&m)?);
        let trace: Basic = normal(&a2(a.at(0, 0), a.at(1, 1))?)?;
        let root: Basic = sqrt(&normal(&s2(&pw(&trace, &int(2))?, &m2(&int(4), &delta)?)?)?)?;
        let first: Basic = normal(&d2(&s2(&trace, &root)?, &int(2))?)?;
        let second: Basic = normal(&d2(&a2(&trace, &root)?, &int(2))?)?;
        let constant: Basic = normal(&d2(&big, &delta)?)?;
        conic.canonical = Some((first.clone(), second.clone(), constant.clone()));
        let squares: Vec<Basic> = vec![
            normal(&d2(&neg(&constant)?, &first)?)?,
            normal(&d2(&neg(&constant)?, &second)?)?,
        ];
        if kind == "hyperbola" {
            let p0: bool = positive(&squares[0]);
            let transverse: Basic = if p0 { squares[0].clone() } else { squares[1].clone() };
            let conjugate: Basic = if p0 { squares[1].clone() } else { squares[0].clone() };
            conic.semi_axes = Some((sqrt(&transverse)?, sqrt(&normal(&neg(&conjugate)?)?)?));
            conic.eccentricity = Some(sqrt(&normal(&s2(&int(1), &d2(&conjugate, &transverse)?)?)?)?);
        } else {
            let (mut larger, mut smaller) = (squares[0].clone(), squares[1].clone());
            if positive(&s2(&smaller, &larger)?) {
                std::mem::swap(&mut larger, &mut smaller);
            }
            conic.semi_axes = Some((sqrt(&larger)?, sqrt(&smaller)?));
            conic.eccentricity = Some(sqrt(&normal(&s2(&int(1), &d2(&smaller, &larger)?)?)?)?);
        }
    } else if kind == "parabola" {
        let trace: Basic = normal(&a2(a.at(0, 0), a.at(1, 1))?)?;
        conic.parameter = Some(sqrt(&normal(&d2(&neg(&det(&m)?)?, &pw(&trace, &int(3))?)?)?)?);
    }
    Ok(conic)
}

fn quadric_kind(a: &ExprMatrix, m: &ExprMatrix) -> PolyResult<&'static str> {
    let (rank_a, sign_a) = signature(a)?;
    let (rank_m, sign_m) = signature(m)?;
    Ok(match rank_m {
        4 => {
            if rank_a == 3 {
                if sign_a == 3 {
                    if sign_m == 2 {
                        "ellipsoid"
                    } else {
                        "empty set"
                    }
                } else if sign_m == 0 {
                    "hyperboloid of one sheet"
                } else {
                    "hyperboloid of two sheets"
                }
            } else if sign_a == 2 {
                "elliptic paraboloid"
            } else {
                "hyperbolic paraboloid"
            }
        }
        3 => match rank_a {
            3 => {
                if sign_a == 3 {
                    "point"
                } else {
                    "cone"
                }
            }
            2 => {
                if sign_a == 0 {
                    "hyperbolic cylinder"
                } else if sign_m == 1 {
                    "elliptic cylinder"
                } else {
                    "empty set"
                }
            }
            _ => "parabolic cylinder",
        },
        2 => match rank_a {
            2 => {
                if sign_a == 2 {
                    "line"
                } else {
                    "two intersecting planes"
                }
            }
            1 => {
                if sign_m == 0 {
                    "two parallel planes"
                } else {
                    "empty set"
                }
            }
            _ => "plane",
        },
        1 => {
            if rank_a == 1 {
                "double plane"
            } else {
                "empty set"
            }
        }
        _ => "space",
    })
}

/// The kind of a quadric with its center (`classify_quadric`).
pub fn classify_quadric(equation: &Basic, x: &Basic, y: &Basic, z: &Basic) -> PolyResult<Quadric> {
    let (a, m) = quadric_matrices(equation, &[x.clone(), y.clone(), z.clone()])?;
    Ok(Quadric {
        kind: quadric_kind(&a, &m)?.to_string(),
        center: center(&a, &m)?,
    })
}
