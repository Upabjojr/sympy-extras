//! Port of `sympy_extras/diffgeom/metric.py`: a (pseudo-)Riemannian metric
//! in coordinates: inverse, determinant, raising and lowering indices,
//! Christoffel symbols, Riemann, Ricci, Einstein and Weyl tensors, the
//! Ricci and Kretschmann scalars, geodesic and Killing equations,
//! covariant derivative, divergence, gradient and Laplace–Beltrami
//! operator.

#![allow(clippy::needless_range_loop, clippy::mutable_key_type)]

use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::basic::{Basic, UndefinedFunction};
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;

type R = PolyResult<Basic>;
type Rank3 = Vec<Vec<Vec<Basic>>>;
type Rank4 = Vec<Vec<Vec<Vec<Basic>>>>;
/// The simplification applied to every result.
pub type Simplifier = fn(&Basic) -> R;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    int(0)
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
fn sum(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn diff(e: &Basic, v: &Basic) -> R {
    crate::core::function::derivative(e, v)
}

/// The default simplifier, `normal_form`.
pub fn default_simplifier(e: &Basic) -> R {
    crate::simplify::normalform::normal_form(e)
}

/// A metric in coordinates (`Metric`).
#[derive(Debug, Clone)]
pub struct Metric {
    pub g: ExprMatrix,
    pub coordinates: Vec<Basic>,
    pub dimension: usize,
    pub simplifier: Simplifier,
    inverse: OnceCell<ExprMatrix>,
    christoffel: OnceCell<Rank3>,
    riemann: OnceCell<Rank4>,
    ricci: OnceCell<ExprMatrix>,
}

impl Metric {
    pub fn new(g: &ExprMatrix, coordinates: &[Basic], simplifier: Option<Simplifier>) -> PolyResult<Metric> {
        let n: usize = coordinates.len();
        if g.shape() != (n, n) {
            return Err(value_error(format!(
                "a {} x {} matrix for {} coordinates",
                g.rows, g.cols, n
            )));
        }
        for i in 0..n {
            if coordinates[..i].contains(&coordinates[i]) {
                return Err(value_error("the coordinates are not distinct".to_string()));
            }
        }
        let simplifier: Simplifier = simplifier.unwrap_or(default_simplifier);
        for i in 0..n {
            for j in 0..i {
                if simplifier(&s2(g.at(i, j), g.at(j, i))?)? != zero() {
                    return Err(value_error("the matrix of a metric is symmetric".to_string()));
                }
            }
        }
        Ok(Metric {
            g: g.clone(),
            coordinates: coordinates.to_vec(),
            dimension: n,
            simplifier,
            inverse: OnceCell::new(),
            christoffel: OnceCell::new(),
            riemann: OnceCell::new(),
            ricci: OnceCell::new(),
        })
    }

    /// The metric of a line element `ds**2` in the differentials
    /// (`from_line_element`).
    pub fn from_line_element(
        ds2: &Basic,
        coordinates: &[Basic],
        differentials: &[Basic],
        simplifier: Option<Simplifier>,
    ) -> PolyResult<Metric> {
        let n: usize = coordinates.len();
        if differentials.len() != n {
            return Err(value_error(format!(
                "{} differentials for {} coordinates",
                differentials.len(),
                n
            )));
        }
        let form: Basic = crate::core::function::expand_default(ds2)?;
        let mut rows: Vec<Vec<Basic>> = Vec::new();
        let mut total: Basic = zero();
        for i in 0..n {
            let mut line: Vec<Basic> = Vec::new();
            for j in 0..n {
                let entry: Basic = if i == j {
                    crate::core::expr::coeff(&form, &crate::core::power::pow(&differentials[i], &int(2))?)?
                } else {
                    d2(
                        &crate::core::expr::coeff(
                            &crate::core::expr::coeff(&form, &differentials[i])?,
                            &differentials[j],
                        )?,
                        &int(2),
                    )?
                };
                total = a2(&total, &m2(&m2(&entry, &differentials[i])?, &differentials[j])?)?;
                line.push(entry);
            }
            rows.push(line);
        }
        let leftover: bool = rows.iter().flatten().any(|e| differentials.iter().any(|d| e.has(d)));
        if leftover || crate::core::function::expand_default(&s2(&form, &total)?)? != zero() {
            return Err(value_error(format!(
                "{} is not a quadratic form in the differentials",
                ds2
            )));
        }
        Metric::new(&ExprMatrix::from_rows(&rows)?, coordinates, simplifier)
    }

    fn simplify(&self, e: &Basic) -> R {
        (self.simplifier)(e)
    }

    /// The inverse matrix (`inverse`).
    pub fn inverse(&self) -> PolyResult<ExprMatrix> {
        if let Some(i) = self.inverse.get() {
            return Ok(i.clone());
        }
        let n: usize = self.dimension;
        let diagonal: bool = (0..n).all(|i| (0..n).all(|j| i == j || *self.g.at(i, j) == zero()));
        let found: ExprMatrix = if diagonal {
            if (0..n).any(|i| *self.g.at(i, i) == zero()) {
                return Err(value_error("the metric is degenerate".to_string()));
            }
            let mut entries: Vec<Basic> = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    entries.push(if i == j {
                        self.simplify(&d2(&int(1), self.g.at(i, i))?)?
                    } else {
                        zero()
                    });
                }
            }
            ExprMatrix::new(n, n, entries)?
        } else {
            match crate::matrices::linalg::inverse(&self.g) {
                Ok(m) => m.applyfunc(&self.simplifier)?,
                Err(PolyError::Value(_)) => return Err(value_error("the metric is degenerate".to_string())),
                Err(e) => return Err(e),
            }
        };
        let _ = self.inverse.set(found.clone());
        Ok(found)
    }

    pub fn determinant(&self) -> R {
        self.simplify(&crate::matrices::linalg::det(&self.g)?)
    }

    fn vector(&self, components: &[Basic]) -> PolyResult<Vec<Basic>> {
        if components.len() != self.dimension {
            return Err(value_error(format!(
                "{} components in dimension {}",
                components.len(),
                self.dimension
            )));
        }
        Ok(components.to_vec())
    }

    pub fn lower(&self, vector: &[Basic]) -> PolyResult<Vec<Basic>> {
        let v: Vec<Basic> = self.vector(vector)?;
        let n: usize = self.dimension;
        (0..n)
            .map(|i| {
                let terms: Vec<Basic> = (0..n).map(|j| m2(self.g.at(i, j), &v[j])).collect::<PolyResult<_>>()?;
                self.simplify(&sum(&terms)?)
            })
            .collect()
    }

    pub fn raise_index(&self, covector: &[Basic]) -> PolyResult<Vec<Basic>> {
        let w: Vec<Basic> = self.vector(covector)?;
        let n: usize = self.dimension;
        let inverse: ExprMatrix = self.inverse()?;
        (0..n)
            .map(|i| {
                let terms: Vec<Basic> = (0..n).map(|j| m2(inverse.at(i, j), &w[j])).collect::<PolyResult<_>>()?;
                self.simplify(&sum(&terms)?)
            })
            .collect()
    }

    fn connection(&self) -> PolyResult<Rank3> {
        if let Some(t) = self.christoffel.get() {
            return Ok(t.clone());
        }
        let n: usize = self.dimension;
        let x: &[Basic] = &self.coordinates;
        let inverse: ExprMatrix = self.inverse()?;
        let mut dg: Rank3 = Vec::new();
        for l in 0..n {
            let mut plane: Vec<Vec<Basic>> = Vec::new();
            for j in 0..n {
                plane.push(
                    (0..n)
                        .map(|k| diff(self.g.at(j, k), &x[l]))
                        .collect::<PolyResult<_>>()?,
                );
            }
            dg.push(plane);
        }
        let mut table: Rank3 = vec![vec![vec![zero(); n]; n]; n];
        for j in 0..n {
            for k in j..n {
                let first: Vec<Basic> = (0..n)
                    .map(|l| s2(&a2(&dg[j][l][k], &dg[k][l][j])?, &dg[l][j][k]))
                    .collect::<PolyResult<_>>()?;
                for i in 0..n {
                    let mut terms: Vec<Basic> = Vec::new();
                    for l in 0..n {
                        if *inverse.at(i, l) != zero() && first[l] != zero() {
                            terms.push(m2(inverse.at(i, l), &first[l])?);
                        }
                    }
                    let value: Basic = if terms.is_empty() {
                        zero()
                    } else {
                        self.simplify(&d2(&sum(&terms)?, &int(2))?)?
                    };
                    table[i][j][k] = value.clone();
                    table[i][k][j] = value;
                }
            }
        }
        let _ = self.christoffel.set(table.clone());
        Ok(table)
    }

    pub fn christoffel(&self, i: usize, j: usize, k: usize) -> R {
        Ok(self.connection()?[i][j][k].clone())
    }

    /// The nonzero symbols with `j <= k` (`christoffel_symbols`).
    #[allow(clippy::type_complexity)]
    pub fn christoffel_symbols(&self) -> PolyResult<Vec<((usize, usize, usize), Basic)>> {
        let n: usize = self.dimension;
        let table: Rank3 = self.connection()?;
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in j..n {
                    if table[i][j][k] != zero() {
                        out.push(((i, j, k), table[i][j][k].clone()));
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn christoffel_first_kind(&self, i: usize, j: usize, k: usize) -> R {
        let (x, g) = (&self.coordinates, &self.g);
        let value: Basic = s2(
            &a2(&diff(g.at(i, k), &x[j])?, &diff(g.at(i, j), &x[k])?)?,
            &diff(g.at(j, k), &x[i])?,
        )?;
        self.simplify(&d2(&value, &int(2))?)
    }

    fn curvature(&self) -> PolyResult<Rank4> {
        if let Some(t) = self.riemann.get() {
            return Ok(t.clone());
        }
        let n: usize = self.dimension;
        let x: &[Basic] = &self.coordinates;
        let gamma: Rank3 = self.connection()?;
        let mut table: Rank4 = vec![vec![vec![vec![zero(); n]; n]; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in k + 1..n {
                        let mut terms: Vec<Basic> =
                            vec![diff(&gamma[i][l][j], &x[k])?, neg(&diff(&gamma[i][k][j], &x[l])?)?];
                        for m in 0..n {
                            if gamma[i][k][m] != zero() && gamma[m][l][j] != zero() {
                                terms.push(m2(&gamma[i][k][m], &gamma[m][l][j])?);
                            }
                            if gamma[i][l][m] != zero() && gamma[m][k][j] != zero() {
                                terms.push(m2(&neg(&gamma[i][l][m])?, &gamma[m][k][j])?);
                            }
                        }
                        let value: Basic = self.simplify(&sum(&terms)?)?;
                        table[i][j][l][k] = if value != zero() {
                            self.simplify(&neg(&value)?)?
                        } else {
                            zero()
                        };
                        table[i][j][k][l] = value;
                    }
                }
            }
        }
        let _ = self.riemann.set(table.clone());
        Ok(table)
    }

    pub fn riemann(&self, i: usize, j: usize, k: usize, l: usize) -> R {
        Ok(self.curvature()?[i][j][k][l].clone())
    }

    pub fn riemann_lower(&self, i: usize, j: usize, k: usize, l: usize) -> R {
        let table: Rank4 = self.curvature()?;
        let terms: Vec<Basic> = (0..self.dimension)
            .map(|m| m2(self.g.at(i, m), &table[m][j][k][l]))
            .collect::<PolyResult<_>>()?;
        self.simplify(&sum(&terms)?)
    }

    /// The nonzero components with `k < l` (`riemann_components`).
    #[allow(clippy::type_complexity)]
    pub fn riemann_components(&self) -> PolyResult<Vec<((usize, usize, usize, usize), Basic)>> {
        let n: usize = self.dimension;
        let table: Rank4 = self.curvature()?;
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in k + 1..n {
                        if table[i][j][k][l] != zero() {
                            out.push(((i, j, k, l), table[i][j][k][l].clone()));
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn ricci(&self) -> PolyResult<ExprMatrix> {
        if let Some(r) = self.ricci.get() {
            return Ok(r.clone());
        }
        let n: usize = self.dimension;
        let table: Rank4 = self.curvature()?;
        let mut entries: Vec<Basic> = Vec::new();
        for j in 0..n {
            for l in 0..n {
                let terms: Vec<Basic> = (0..n).map(|i| table[i][j][i][l].clone()).collect();
                entries.push(self.simplify(&sum(&terms)?)?);
            }
        }
        let found: ExprMatrix = ExprMatrix::new(n, n, entries)?;
        let _ = self.ricci.set(found.clone());
        Ok(found)
    }

    pub fn ricci_scalar(&self) -> R {
        let n: usize = self.dimension;
        let (ricci, inverse) = (self.ricci()?, self.inverse()?);
        let mut terms: Vec<Basic> = Vec::new();
        for j in 0..n {
            for l in 0..n {
                terms.push(m2(inverse.at(j, l), ricci.at(j, l))?);
            }
        }
        self.simplify(&sum(&terms)?)
    }

    pub fn einstein(&self) -> PolyResult<ExprMatrix> {
        let n: usize = self.dimension;
        let (ricci, scalar) = (self.ricci()?, self.ricci_scalar()?);
        let mut entries: Vec<Basic> = Vec::new();
        for j in 0..n {
            for l in 0..n {
                entries.push(self.simplify(&s2(ricci.at(j, l), &d2(&m2(&scalar, self.g.at(j, l))?, &int(2))?)?)?);
            }
        }
        ExprMatrix::new(n, n, entries)
    }

    fn lowered(&self) -> PolyResult<Rank4> {
        let n: usize = self.dimension;
        let table: Rank4 = self.curvature()?;
        let mut out: Rank4 = vec![vec![vec![vec![zero(); n]; n]; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let terms: Vec<Basic> = (0..n)
                            .map(|m| m2(self.g.at(i, m), &table[m][j][k][l]))
                            .collect::<PolyResult<_>>()?;
                        out[i][j][k][l] = self.simplify(&sum(&terms)?)?;
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn kretschmann(&self) -> R {
        let n: usize = self.dimension;
        let inverse: ExprMatrix = self.inverse()?;
        let lowered: Rank4 = self.lowered()?;
        let mut raised: Rank4 = lowered.clone();
        for position in 0..4 {
            let mut updated: Rank4 = vec![vec![vec![vec![zero(); n]; n]; n]; n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for l in 0..n {
                            let index: [usize; 4] = [i, j, k, l];
                            let mut terms: Vec<Basic> = Vec::new();
                            for m in 0..n {
                                let factor: &Basic = inverse.at(index[position], m);
                                if *factor == zero() {
                                    continue;
                                }
                                let mut source: [usize; 4] = index;
                                source[position] = m;
                                let component: &Basic = &raised[source[0]][source[1]][source[2]][source[3]];
                                if *component != zero() {
                                    terms.push(m2(factor, component)?);
                                }
                            }
                            if !terms.is_empty() {
                                updated[i][j][k][l] = sum(&terms)?;
                            }
                        }
                    }
                }
            }
            raised = updated;
        }
        let mut terms: Vec<Basic> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        if lowered[i][j][k][l] != zero() {
                            terms.push(m2(&lowered[i][j][k][l], &raised[i][j][k][l])?);
                        }
                    }
                }
            }
        }
        self.simplify(&sum(&terms)?)
    }

    pub fn weyl(&self, i: usize, j: usize, k: usize, l: usize) -> R {
        let n: usize = self.dimension;
        if n < 3 {
            return Err(value_error(
                "the Weyl tensor is defined from the dimension 3 on".to_string(),
            ));
        }
        let (g, ricci, scalar) = (&self.g, self.ricci()?, self.ricci_scalar()?);
        let a: Basic = a2(
            &s2(
                &s2(&m2(g.at(i, k), ricci.at(j, l))?, &m2(g.at(i, l), ricci.at(j, k))?)?,
                &m2(g.at(j, k), ricci.at(i, l))?,
            )?,
            &m2(g.at(j, l), ricci.at(i, k))?,
        )?;
        let b: Basic = s2(&m2(g.at(i, k), g.at(j, l))?, &m2(g.at(i, l), g.at(j, k))?)?;
        let nn: i64 = n as i64;
        let value: Basic = a2(
            &s2(&self.riemann_lower(i, j, k, l)?, &d2(&a, &int(nn - 2))?)?,
            &d2(&m2(&scalar, &b)?, &int((nn - 1) * (nn - 2)))?,
        )?;
        self.simplify(&value)
    }

    pub fn is_flat(&self) -> PolyResult<bool> {
        Ok(self.riemann_components()?.is_empty())
    }

    /// `x_i'' + Gamma^i_jk x_j' x_k' = 0` along `x(t)` (`geodesic_equations`).
    pub fn geodesic_equations(&self, parameter: &Basic) -> PolyResult<Vec<Basic>> {
        let n: usize = self.dimension;
        let gamma: Rank3 = self.connection()?;
        let paths: Vec<Basic> = self
            .coordinates
            .iter()
            .map(|c| Arc::new(UndefinedFunction::new(&c.to_string(), None, &[])).apply(vec![parameter.clone()]))
            .collect::<PolyResult<_>>()?;
        let rule: HashMap<Basic, Basic> = self.coordinates.iter().cloned().zip(paths.iter().cloned()).collect();
        let velocity: Vec<Basic> = paths.iter().map(|p| diff(p, parameter)).collect::<PolyResult<_>>()?;
        let mut equations: Vec<Basic> = Vec::new();
        for i in 0..n {
            let mut terms: Vec<Basic> = vec![diff(&velocity[i], parameter)?];
            for j in 0..n {
                for k in j..n {
                    if gamma[i][j][k] != zero() {
                        let weight: i64 = if j == k { 1 } else { 2 };
                        let at: Basic = crate::core::function::xreplace_evaluated(&gamma[i][j][k], &rule)?;
                        terms.push(m2(&m2(&m2(&int(weight), &at)?, &velocity[j])?, &velocity[k])?);
                    }
                }
            }
            equations.push(sum(&terms)?);
        }
        Ok(equations)
    }

    pub fn covariant_derivative(&self, vector: &[Basic]) -> PolyResult<ExprMatrix> {
        let v: Vec<Basic> = self.vector(vector)?;
        let n: usize = self.dimension;
        let x: &[Basic] = &self.coordinates;
        let gamma: Rank3 = self.connection()?;
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let terms: Vec<Basic> = (0..n).map(|k| m2(&gamma[i][j][k], &v[k])).collect::<PolyResult<_>>()?;
                entries.push(self.simplify(&a2(&diff(&v[i], &x[j])?, &sum(&terms)?)?)?);
            }
        }
        ExprMatrix::new(n, n, entries)
    }

    pub fn divergence(&self, vector: &[Basic]) -> R {
        let derivative: ExprMatrix = self.covariant_derivative(vector)?;
        let terms: Vec<Basic> = (0..self.dimension).map(|i| derivative.at(i, i).clone()).collect();
        self.simplify(&sum(&terms)?)
    }

    pub fn gradient(&self, f: &Basic) -> PolyResult<Vec<Basic>> {
        let derivatives: Vec<Basic> = self.coordinates.iter().map(|x| diff(f, x)).collect::<PolyResult<_>>()?;
        self.raise_index(&derivatives)
    }

    /// The Laplace–Beltrami operator (`laplacian`).
    pub fn laplacian(&self, f: &Basic) -> R {
        let n: usize = self.dimension;
        let x: &[Basic] = &self.coordinates;
        let (gamma, inverse) = (self.connection()?, self.inverse()?);
        let first: Vec<Basic> = (0..n).map(|k| diff(f, &x[k])).collect::<PolyResult<_>>()?;
        let mut terms: Vec<Basic> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if *inverse.at(i, j) == zero() {
                    continue;
                }
                let correction: Vec<Basic> = (0..n)
                    .map(|k| m2(&gamma[k][i][j], &first[k]))
                    .collect::<PolyResult<_>>()?;
                let second: Basic = s2(&diff(&first[j], &x[i])?, &sum(&correction)?)?;
                terms.push(m2(inverse.at(i, j), &second)?);
            }
        }
        self.simplify(&sum(&terms)?)
    }

    pub fn killing_equations(&self, vector: &[Basic]) -> PolyResult<ExprMatrix> {
        let n: usize = self.dimension;
        let x: &[Basic] = &self.coordinates;
        let gamma: Rank3 = self.connection()?;
        let xi: Vec<Basic> = self.lower(vector)?;
        let mut covariant: Vec<Vec<Basic>> = Vec::new();
        for i in 0..n {
            let mut row: Vec<Basic> = Vec::new();
            for j in 0..n {
                let correction: Vec<Basic> = (0..n).map(|k| m2(&gamma[k][i][j], &xi[k])).collect::<PolyResult<_>>()?;
                row.push(s2(&diff(&xi[j], &x[i])?, &sum(&correction)?)?);
            }
            covariant.push(row);
        }
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                entries.push(self.simplify(&a2(&covariant[i][j], &covariant[j][i])?)?);
            }
        }
        ExprMatrix::new(n, n, entries)
    }

    pub fn is_killing(&self, vector: &[Basic]) -> PolyResult<bool> {
        Ok(self.killing_equations(vector)?.entries.iter().all(|e| *e == zero()))
    }
}
