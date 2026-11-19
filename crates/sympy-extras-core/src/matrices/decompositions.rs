//! Port of `sympy_extras/matrices/decompositions.py`: LU with row
//! pivoting, LDL and Cholesky of Hermitian matrices, Gram–Schmidt and QR,
//! rank factorisations, the congruence diagonalisation of a Hermitian
//! matrix with its inertia (Sylvester), definiteness and the leading
//! principal minors; computed in the field of the entries of
//! `matrices/eigen.rs`, the pivots certified nonzero.

#![allow(clippy::needless_range_loop)]

use crate::core::assumptions::Predicate;
use crate::core::basic::Basic;
use crate::core::expr::is;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::{sqrt, Field, FunctionEntries, GaussianEntries, RationalEntries};
use crate::matrices::linalg::{complex_lift, det, kernels as kernels_of, rref, square, PolynomialLift};
use crate::numbers::{numer, Int};
use crate::polys::domains::domain::Domain;

fn int(n: i64) -> Basic {
    Basic::int(n)
}

/// `P*M = L*U` (`LUDecomposition`).
#[derive(Debug, Clone, PartialEq)]
pub struct LUDecomposition {
    pub l: ExprMatrix,
    pub u: ExprMatrix,
    pub permutation: Vec<usize>,
    pub rank: usize,
}

impl LUDecomposition {
    /// The permutation matrix (`P`).
    pub fn p(&self) -> ExprMatrix {
        let n: usize = self.permutation.len();
        let entries: Vec<Basic> = (0..n * n)
            .map(|k| {
                if self.permutation[k / n] == k % n {
                    int(1)
                } else {
                    int(0)
                }
            })
            .collect();
        ExprMatrix {
            rows: n,
            cols: n,
            entries,
        }
    }

    pub fn repr(&self) -> String {
        let p: Vec<String> = self.permutation.iter().map(|k| k.to_string()).collect();
        let shown: String = if p.len() == 1 {
            format!("({},)", p[0])
        } else {
            format!("({})", p.join(", "))
        };
        format!("LUDecomposition({}, {}, {})", self.l.repr(), self.u.repr(), shown)
    }
}

/// The algorithms over a field of the entries (`_Worker`).
struct Worker<F: Field> {
    field: F,
    exact: bool,
}

type Elem<F> = <<F as Field>::D as Domain>::Elem;

impl<F: Field> Worker<F> {
    fn rows(&self, m: &ExprMatrix) -> PolyResult<Vec<Vec<Elem<F>>>> {
        (0..m.rows)
            .map(|i| m.row(i).iter().map(|e| self.field.element(e)).collect())
            .collect()
    }

    fn matrix(&self, rows: &[Vec<Elem<F>>], ncols: usize) -> PolyResult<ExprMatrix> {
        let mut entries: Vec<Basic> = Vec::new();
        for row in rows {
            for a in row {
                entries.push(self.field.expr(a)?);
            }
        }
        ExprMatrix::new(rows.len(), ncols, entries)
    }

    fn pivot(&self, a: Elem<F>) -> PolyResult<Elem<F>> {
        if !self.exact {
            let e: Basic = self.field.expr(&a)?;
            let nonzero_rational: bool = e.is_rational_class() && e != int(0);
            if !nonzero_rational && crate::simplify::structure::is_zero(&e, &[])? != Some(false) {
                return Err(PolyError::NotImplemented(format!(
                    "the pivot {} is not proved nonzero: simplify the entries of the matrix first",
                    e
                )));
            }
        }
        Ok(a)
    }

    fn inner(&self, u: &[Elem<F>], v: &[Elem<F>]) -> Elem<F> {
        let k: &F::D = self.field.domain();
        let mut total = k.zero();
        for (a, b) in u.iter().zip(v.iter()) {
            if !k.is_zero(a) && !k.is_zero(b) {
                total = k.add(&total, &k.mul(&self.field.conjugate(a), b));
            }
        }
        total
    }

    fn hermitian(&self, rows: &[Vec<Elem<F>>]) -> bool {
        let n: usize = rows.len();
        (0..n).all(|i| (i..n).all(|j| rows[i][j] == self.field.conjugate(&rows[j][i])))
    }

    fn eliminate(&self, rows: &mut [Vec<Elem<F>>], target: usize, pivot_row: &[Elem<F>], factor: &Elem<F>) {
        let k: &F::D = self.field.domain();
        rows[target] = rows[target]
            .iter()
            .zip(pivot_row.iter())
            .map(|(a, b)| {
                if k.is_zero(b) {
                    a.clone()
                } else {
                    k.sub(a, &k.mul(factor, b))
                }
            })
            .collect();
    }

    fn lu(&self, m: &ExprMatrix) -> PolyResult<LUDecomposition> {
        let k: &F::D = self.field.domain();
        let (mm, n) = m.shape();
        let mut rows = self.rows(m)?;
        let mut lower: Vec<Vec<Elem<F>>> = vec![vec![k.zero(); mm]; mm];
        let mut permutation: Vec<usize> = (0..mm).collect();
        let mut r: usize = 0;
        for column in 0..n {
            if r == mm {
                break;
            }
            let Some(chosen) = (r..mm).find(|i| !k.is_zero(&rows[*i][column])) else {
                continue;
            };
            if chosen != r {
                rows.swap(r, chosen);
                lower.swap(r, chosen);
                permutation.swap(r, chosen);
            }
            let inverse = k.invert(&self.pivot(rows[r][column].clone())?)?;
            let pivot_row: Vec<Elem<F>> = rows[r].clone();
            for i in r + 1..mm {
                if k.is_zero(&rows[i][column]) {
                    continue;
                }
                let factor = k.mul(&rows[i][column], &inverse);
                lower[i][r] = factor.clone();
                self.eliminate(&mut rows, i, &pivot_row, &factor);
            }
            r += 1;
        }
        for (i, row) in lower.iter_mut().enumerate() {
            row[i] = k.one();
        }
        Ok(LUDecomposition {
            l: self.matrix(&lower, mm)?,
            u: self.matrix(&rows, n)?,
            permutation,
            rank: r,
        })
    }

    fn is_hermitian(&self, m: &ExprMatrix) -> PolyResult<bool> {
        Ok(m.rows == m.cols && self.hermitian(&self.rows(m)?))
    }

    fn ldl(&self, m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
        let k: &F::D = self.field.domain();
        let n: usize = m.rows;
        let mut rows = self.rows(m)?;
        if !self.hermitian(&rows) {
            return Err(value_error("the matrix is not symmetric (Hermitian)".to_string()));
        }
        let mut lower: Vec<Vec<Elem<F>>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { k.one() } else { k.zero() }).collect())
            .collect();
        let mut diagonal: Vec<Elem<F>> = Vec::new();
        for kk in 0..n {
            let pivot = rows[kk][kk].clone();
            if k.is_zero(&pivot) {
                return Err(value_error(format!(
                    "the leading principal minor of order {} is zero: no LDL decomposition without pivoting (see congruence_diagonalize)",
                    kk + 1
                )));
            }
            let inverse = k.invert(&self.pivot(pivot.clone())?)?;
            diagonal.push(pivot);
            let pivot_row: Vec<Elem<F>> = rows[kk].clone();
            for i in kk + 1..n {
                if k.is_zero(&rows[i][kk]) {
                    continue;
                }
                let factor = k.mul(&rows[i][kk], &inverse);
                lower[i][kk] = factor.clone();
                self.eliminate(&mut rows, i, &pivot_row, &factor);
            }
        }
        let d: Vec<Vec<Elem<F>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { diagonal[i].clone() } else { k.zero() })
                    .collect()
            })
            .collect();
        Ok((self.matrix(&lower, n)?, self.matrix(&d, n)?))
    }

    #[allow(clippy::type_complexity)]
    fn orthogonalise(
        &self,
        columns: &[Vec<Elem<F>>],
    ) -> PolyResult<(Vec<Vec<Elem<F>>>, Vec<Elem<F>>, Vec<Vec<Elem<F>>>, Vec<usize>)> {
        let k: &F::D = self.field.domain();
        let n: usize = columns.len();
        let mut vectors: Vec<Vec<Elem<F>>> = Vec::new();
        let mut norms: Vec<Elem<F>> = Vec::new();
        let mut inverses: Vec<Elem<F>> = Vec::new();
        let mut coefficients: Vec<Vec<Elem<F>>> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        for (kk, column) in columns.iter().enumerate() {
            let mut residual: Vec<Elem<F>> = column.clone();
            for (j, u) in vectors.iter().enumerate() {
                let product = self.inner(u, column);
                if k.is_zero(&product) {
                    continue;
                }
                let c = k.mul(&product, &inverses[j]);
                coefficients[j][kk] = c.clone();
                residual = residual
                    .iter()
                    .zip(u.iter())
                    .map(|(a, b)| {
                        if k.is_zero(b) {
                            a.clone()
                        } else {
                            k.sub(a, &k.mul(&c, b))
                        }
                    })
                    .collect();
            }
            if residual.iter().all(|a| k.is_zero(a)) {
                continue;
            }
            let norm = self.pivot(self.inner(&residual, &residual))?;
            vectors.push(residual);
            inverses.push(k.invert(&norm)?);
            norms.push(norm);
            let mut line: Vec<Elem<F>> = vec![k.zero(); n];
            line[kk] = k.one();
            coefficients.push(line);
            pivots.push(kk);
        }
        Ok((vectors, norms, coefficients, pivots))
    }

    fn qr(&self, m: &ExprMatrix, normalize: bool) -> PolyResult<(ExprMatrix, ExprMatrix, Vec<usize>)> {
        let (mm, n) = m.shape();
        let rows = self.rows(m)?;
        let columns: Vec<Vec<Elem<F>>> = (0..n).map(|j| (0..mm).map(|i| rows[i][j].clone()).collect()).collect();
        let (vectors, norms, coefficients, pivots) = self.orthogonalise(&columns)?;
        let r: usize = vectors.len();
        let lengths: Vec<Basic> = norms
            .iter()
            .map(|norm| {
                if normalize {
                    sqrt(&self.field.expr(norm)?)
                } else {
                    Ok(int(1))
                }
            })
            .collect::<PolyResult<_>>()?;
        let mut q: Vec<Basic> = Vec::new();
        for i in 0..mm {
            for j in 0..r {
                q.push(crate::core::mul::div(&self.field.expr(&vectors[j][i])?, &lengths[j])?);
            }
        }
        let mut rr: Vec<Basic> = Vec::new();
        for j in 0..r {
            for kk in 0..n {
                rr.push(crate::core::mul::mul(&[
                    self.field.expr(&coefficients[j][kk])?,
                    lengths[j].clone(),
                ])?);
            }
        }
        Ok((ExprMatrix::new(mm, r, q)?, ExprMatrix::new(r, n, rr)?, pivots))
    }

    fn congruence(&self, m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
        let k: &F::D = self.field.domain();
        let n: usize = m.rows;
        let mut a = self.rows(m)?;
        if !self.hermitian(&a) {
            return Err(value_error("the matrix is not symmetric (Hermitian)".to_string()));
        }
        let mut p: Vec<Vec<Elem<F>>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { k.one() } else { k.zero() }).collect())
            .collect();
        let add_column =
            |a: &mut Vec<Vec<Elem<F>>>, p: &mut Vec<Vec<Elem<F>>>, target: usize, source: usize, c: &Elem<F>| {
                for row in a.iter_mut() {
                    if !k.is_zero(&row[source]) {
                        row[target] = k.add(&row[target], &k.mul(c, &row[source]));
                    }
                }
                for row in p.iter_mut() {
                    if !k.is_zero(&row[source]) {
                        row[target] = k.add(&row[target], &k.mul(c, &row[source]));
                    }
                }
                let cc = self.field.conjugate(c);
                let src: Vec<Elem<F>> = a[source].clone();
                a[target] = a[target]
                    .iter()
                    .zip(src.iter())
                    .map(|(x, b)| {
                        if k.is_zero(b) {
                            x.clone()
                        } else {
                            k.add(x, &k.mul(&cc, b))
                        }
                    })
                    .collect();
            };
        let swap = |a: &mut Vec<Vec<Elem<F>>>, p: &mut Vec<Vec<Elem<F>>>, i: usize, j: usize| {
            for row in a.iter_mut() {
                row.swap(i, j);
            }
            for row in p.iter_mut() {
                row.swap(i, j);
            }
            a.swap(i, j);
        };
        for kk in 0..n {
            if k.is_zero(&a[kk][kk]) {
                let mut chosen: Option<usize> = (kk + 1..n).find(|j| !k.is_zero(&a[*j][*j]));
                if chosen.is_none() {
                    let found: Option<(usize, usize)> = (kk..n)
                        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                        .find(|(i, j)| !k.is_zero(&a[*i][*j]));
                    let Some((i, j)) = found else { break };
                    let entry = a[i][j].clone();
                    let c = if self.field.conjugate(&entry) == entry {
                        k.one()
                    } else {
                        self.field.conjugate(&entry)
                    };
                    add_column(&mut a, &mut p, i, j, &c);
                    chosen = Some(i);
                }
                let chosen: usize = chosen.expect("chosen");
                if chosen != kk {
                    swap(&mut a, &mut p, kk, chosen);
                }
            }
            let inverse = k.invert(&self.pivot(a[kk][kk].clone())?)?;
            for i in kk + 1..n {
                if !k.is_zero(&a[i][kk]) {
                    let c = k.neg(&self.field.conjugate(&k.mul(&a[i][kk], &inverse)));
                    add_column(&mut a, &mut p, i, kk, &c);
                }
            }
        }
        Ok((self.matrix(&p, n)?, self.matrix(&a, n)?))
    }
}

macro_rules! dispatch {
    ($matrices:expr, |$w:ident| $body:expr) => {{
        let kernels: Vec<Basic> = kernels_of($matrices);
        if let Some(g) = complex_lift(&kernels)? {
            let $w = Worker {
                field: GaussianEntries::new(g),
                exact: true,
            };
            $body
        } else if !kernels.is_empty() {
            let lift: PolynomialLift = PolynomialLift::new(&kernels);
            let exact: bool = lift.exact;
            let $w = Worker {
                field: FunctionEntries::new(lift),
                exact,
            };
            $body
        } else {
            let $w = Worker {
                field: RationalEntries::new(),
                exact: true,
            };
            $body
        }
    }};
}

/// `P*M = L*U` with row pivoting (`lu`).
pub fn lu(m: &ExprMatrix) -> PolyResult<LUDecomposition> {
    dispatch!(&[m], |w| w.lu(m))
}

/// Whether `M` is Hermitian (`is_hermitian`).
pub fn is_hermitian(m: &ExprMatrix) -> PolyResult<bool> {
    dispatch!(&[m], |w| w.is_hermitian(m))
}

/// `M = L*D*L^H` (`ldl`).
pub fn ldl(m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    square(m)?;
    dispatch!(&[m], |w| w.ldl(m))
}

/// `M = L*L^H` (`cholesky`).
pub fn cholesky(m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let (l, d) = ldl(m)?;
    let n: usize = m.rows;
    let mut roots: Vec<Basic> = Vec::new();
    for kk in 0..n {
        let pivot: &Basic = d.at(kk, kk);
        if let Some(r) = pivot.as_rational() {
            if numer(r) <= Int::from(0) {
                return Err(value_error(format!(
                    "the matrix is not positive definite: the pivot {}",
                    pivot
                )));
            }
        }
        roots.push(sqrt(pivot)?);
    }
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            entries.push(if j <= i {
                crate::core::mul::mul(&[l.at(i, j).clone(), roots[j].clone()])?
            } else {
                int(0)
            });
        }
    }
    ExprMatrix::new(n, n, entries)
}

/// Gram–Schmidt (`gram_schmidt`).
pub fn gram_schmidt(vectors: &[ExprMatrix], normalize: bool) -> PolyResult<Vec<ExprMatrix>> {
    if vectors.is_empty() {
        return Ok(Vec::new());
    }
    let size: usize = vectors[0].rows;
    if vectors.iter().any(|v| v.cols != 1 || v.rows != size) {
        return Err(value_error(
            "column vectors of the same length are expected".to_string(),
        ));
    }
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..size {
        for v in vectors {
            entries.push(v.at(i, 0).clone());
        }
    }
    let m: ExprMatrix = ExprMatrix::new(size, vectors.len(), entries)?;
    let (q, _, _) = dispatch!(&[&m], |w| w.qr(&m, normalize))?;
    Ok((0..q.cols)
        .map(|j| ExprMatrix::column((0..size).map(|i| q.at(i, j).clone()).collect()))
        .collect())
}

/// `M = Q*R` (`qr`).
pub fn qr(m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let (q, r, _) = dispatch!(&[m], |w| w.qr(m, true))?;
    Ok((q, r))
}

/// `M = B*C` with `B` the pivot columns (`rank_factorization`).
pub fn rank_factorization(m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let (reduced, pivots) = rref(m)?;
    let r: usize = pivots.len();
    let mut b: Vec<Basic> = Vec::new();
    for i in 0..m.rows {
        for j in &pivots {
            b.push(m.at(i, *j).clone());
        }
    }
    let mut c: Vec<Basic> = Vec::new();
    for i in 0..r {
        for j in 0..m.cols {
            c.push(reduced.at(i, j).clone());
        }
    }
    Ok((ExprMatrix::new(m.rows, r, b)?, ExprMatrix::new(r, m.cols, c)?))
}

/// `(P, D)` with `P^H*M*P = D` diagonal (`congruence_diagonalize`).
pub fn congruence_diagonalize(m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    square(m)?;
    dispatch!(&[m], |w| w.congruence(m))
}

/// `(positive, negative, zero)` of a Hermitian matrix (`inertia`).
pub fn inertia(m: &ExprMatrix) -> PolyResult<(usize, usize, usize)> {
    let (_, d) = congruence_diagonalize(m)?;
    let (mut positive, mut negative, mut zero) = (0, 0, 0);
    for kk in 0..d.rows {
        let entry: &Basic = d.at(kk, kk);
        if is(entry, Predicate::positive) == Some(true) {
            positive += 1;
        } else if is(entry, Predicate::negative) == Some(true) {
            negative += 1;
        } else if is(entry, Predicate::zero) == Some(true) {
            zero += 1;
        } else {
            return Err(value_error(format!("the sign of {} is not known", entry)));
        }
    }
    Ok((positive, negative, zero))
}

pub fn is_positive_definite(m: &ExprMatrix) -> PolyResult<bool> {
    Ok(inertia(m)?.0 == m.rows)
}

pub fn is_positive_semidefinite(m: &ExprMatrix) -> PolyResult<bool> {
    Ok(inertia(m)?.1 == 0)
}

/// The leading principal minors (`leading_principal_minors`).
pub fn leading_principal_minors(m: &ExprMatrix) -> PolyResult<Vec<Basic>> {
    let n: usize = square(m)?;
    let mut out: Vec<Basic> = Vec::new();
    for k in 1..=n {
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..k {
            for j in 0..k {
                entries.push(m.at(i, j).clone());
            }
        }
        out.push(det(&ExprMatrix::new(k, k, entries)?)?);
    }
    Ok(out)
}
