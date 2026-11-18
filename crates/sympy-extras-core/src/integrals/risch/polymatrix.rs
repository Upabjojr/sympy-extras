//! Matrices over the field of the Risch algorithm, ported from
//! `sympy_extras.integrals.risch.polymatrix`: [`FieldMatrix`], an
//! immutable dense matrix of [`RationalFunction`] elements with the
//! operations the parametric problems use (joins, products, the reduced
//! row echelon form and the nullspace by Gauss–Jordan elimination over the
//! field). It takes the place of SymPy's `PolyMatrix`.

use crate::error::{value_error, PolyError, PolyResult};

use super::tower::RationalFunction;

/// An immutable `rows x cols` matrix over the field of the tower, stored
/// row by row (`FieldMatrix`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldMatrix {
    pub rows: usize,
    pub cols: usize,
    entries: Vec<RationalFunction>,
}

impl std::fmt::Display for FieldMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.rows == 0 {
            return write!(f, "FieldMatrix({}, {}, [])", self.rows, self.cols);
        }
        let rows: Vec<String> = (0..self.rows)
            .map(|i| {
                let row: Vec<String> = (0..self.cols)
                    .map(|j| self.entries[i * self.cols + j].to_string())
                    .collect();
                format!("[{}]", row.join(", "))
            })
            .collect();
        write!(f, "FieldMatrix([{}])", rows.join(", "))
    }
}

fn shapes(message: &str, a: &FieldMatrix, b: &FieldMatrix) -> PolyError {
    value_error(format!(
        "{} of shapes ({}, {}) and ({}, {})",
        message, a.rows, a.cols, b.rows, b.cols
    ))
}

impl FieldMatrix {
    pub fn new(rows: usize, cols: usize, entries: Vec<RationalFunction>) -> PolyResult<FieldMatrix> {
        if entries.len() != rows * cols {
            return Err(value_error(format!(
                "{} entries for a {}x{} matrix",
                entries.len(),
                rows,
                cols
            )));
        }
        Ok(FieldMatrix { rows, cols, entries })
    }

    pub fn from_function(rows: usize, cols: usize, f: impl Fn(usize, usize) -> RationalFunction) -> FieldMatrix {
        let entries: Vec<RationalFunction> = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .map(|(i, j)| f(i, j))
            .collect();
        FieldMatrix { rows, cols, entries }
    }

    /// The matrix of a list of rows (`cols` for an empty list).
    pub fn from_rows(rows: &[Vec<RationalFunction>], cols: usize) -> PolyResult<FieldMatrix> {
        if rows.is_empty() {
            return FieldMatrix::new(0, cols, Vec::new());
        }
        let n: usize = rows[0].len();
        if rows.iter().any(|r| r.len() != n) {
            return Err(value_error("rows of different lengths".to_string()));
        }
        FieldMatrix::new(rows.len(), n, rows.iter().flatten().cloned().collect())
    }

    pub fn row_vector(entries: &[RationalFunction]) -> FieldMatrix {
        FieldMatrix {
            rows: 1,
            cols: entries.len(),
            entries: entries.to_vec(),
        }
    }

    pub fn column(entries: &[RationalFunction]) -> FieldMatrix {
        FieldMatrix {
            rows: entries.len(),
            cols: 1,
            entries: entries.to_vec(),
        }
    }

    pub fn zeros(m: usize, n: usize) -> FieldMatrix {
        FieldMatrix {
            rows: m,
            cols: n,
            entries: vec![RationalFunction::zero(); m * n],
        }
    }

    pub fn eye(n: usize) -> FieldMatrix {
        FieldMatrix::from_function(n, n, |i, j| {
            if i == j {
                RationalFunction::one()
            } else {
                RationalFunction::zero()
            }
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn flat(&self) -> &[RationalFunction] {
        &self.entries
    }

    /// The entry `(i, j)`, negative indices from the end (`entry`).
    pub fn entry(&self, i: i64, j: i64) -> PolyResult<&RationalFunction> {
        let r: i64 = if i < 0 { i + self.rows as i64 } else { i };
        let c: i64 = if j < 0 { j + self.cols as i64 } else { j };
        if !(0 <= r && r < self.rows as i64 && 0 <= c && c < self.cols as i64) {
            return Err(PolyError::Index(format!("index ({}, {}) out of range", i, j)));
        }
        Ok(&self.entries[r as usize * self.cols + c as usize])
    }

    fn at(&self, i: usize, j: usize) -> &RationalFunction {
        &self.entries[i * self.cols + j]
    }

    /// The `k`-th entry in the order of the rows.
    pub fn element(&self, k: usize) -> &RationalFunction {
        &self.entries[k]
    }

    /// The submatrix of the given rows and columns (`__getitem__` with
    /// slices).
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> FieldMatrix {
        let entries: Vec<RationalFunction> = rows
            .iter()
            .flat_map(|r| cols.iter().map(move |c| self.at(*r, *c).clone()))
            .collect();
        FieldMatrix {
            rows: rows.len(),
            cols: cols.len(),
            entries,
        }
    }

    pub fn row(&self, i: usize) -> FieldMatrix {
        self.submatrix(&[i], &(0..self.cols).collect::<Vec<usize>>())
    }

    pub fn col(&self, j: usize) -> FieldMatrix {
        self.submatrix(&(0..self.rows).collect::<Vec<usize>>(), &[j])
    }

    pub fn neg(&self) -> FieldMatrix {
        FieldMatrix {
            rows: self.rows,
            cols: self.cols,
            entries: self.entries.iter().map(|e| e.neg()).collect(),
        }
    }

    pub fn add(&self, other: &FieldMatrix) -> PolyResult<FieldMatrix> {
        if self.shape() != other.shape() {
            return Err(shapes("matrices", self, other));
        }
        let entries: Vec<RationalFunction> = self
            .entries
            .iter()
            .zip(&other.entries)
            .map(|(a, b)| a.add(b))
            .collect::<PolyResult<_>>()?;
        FieldMatrix::new(self.rows, self.cols, entries)
    }

    pub fn sub(&self, other: &FieldMatrix) -> PolyResult<FieldMatrix> {
        if self.shape() != other.shape() {
            return Err(shapes("matrices", self, other));
        }
        let entries: Vec<RationalFunction> = self
            .entries
            .iter()
            .zip(&other.entries)
            .map(|(a, b)| a.sub(b))
            .collect::<PolyResult<_>>()?;
        FieldMatrix::new(self.rows, self.cols, entries)
    }

    /// The matrix product.
    pub fn mul(&self, other: &FieldMatrix) -> PolyResult<FieldMatrix> {
        if self.cols != other.rows {
            return Err(shapes("matrices", self, other));
        }
        let mut entries: Vec<RationalFunction> = Vec::with_capacity(self.rows * other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut total: RationalFunction = RationalFunction::zero();
                for k in 0..self.cols {
                    let a: &RationalFunction = self.at(i, k);
                    if !a.is_zero() {
                        let b: &RationalFunction = other.at(k, j);
                        if !b.is_zero() {
                            total = total.add(&a.mul(b)?)?;
                        }
                    }
                }
                entries.push(total);
            }
        }
        FieldMatrix::new(self.rows, other.cols, entries)
    }

    /// The product by a scalar.
    pub fn scale(&self, c: &RationalFunction) -> PolyResult<FieldMatrix> {
        let entries: Vec<RationalFunction> = self.entries.iter().map(|e| e.mul(c)).collect::<PolyResult<_>>()?;
        FieldMatrix::new(self.rows, self.cols, entries)
    }

    pub fn div(&self, c: &RationalFunction) -> PolyResult<FieldMatrix> {
        self.scale(&c.invert()?)
    }

    pub fn row_join(&self, other: &FieldMatrix) -> PolyResult<FieldMatrix> {
        if self.rows != other.rows {
            if self.is_empty() && self.cols == 0 {
                return Ok(other.clone());
            }
            return Err(shapes("row_join", self, other));
        }
        let mut entries: Vec<RationalFunction> = Vec::with_capacity(self.len() + other.len());
        for i in 0..self.rows {
            entries.extend_from_slice(&self.entries[i * self.cols..(i + 1) * self.cols]);
            entries.extend_from_slice(&other.entries[i * other.cols..(i + 1) * other.cols]);
        }
        FieldMatrix::new(self.rows, self.cols + other.cols, entries)
    }

    pub fn col_join(&self, other: &FieldMatrix) -> PolyResult<FieldMatrix> {
        if self.cols != other.cols {
            if self.is_empty() && self.rows == 0 {
                return Ok(other.clone());
            }
            return Err(shapes("col_join", self, other));
        }
        let mut entries: Vec<RationalFunction> = self.entries.clone();
        entries.extend_from_slice(&other.entries);
        FieldMatrix::new(self.rows + other.rows, self.cols, entries)
    }

    pub fn applyfunc(&self, f: impl Fn(&RationalFunction) -> PolyResult<RationalFunction>) -> PolyResult<FieldMatrix> {
        let entries: Vec<RationalFunction> = self.entries.iter().map(f).collect::<PolyResult<_>>()?;
        FieldMatrix::new(self.rows, self.cols, entries)
    }

    pub fn transpose(&self) -> FieldMatrix {
        let entries: Vec<RationalFunction> = (0..self.cols)
            .flat_map(|j| (0..self.rows).map(move |i| (i, j)))
            .map(|(i, j)| self.at(i, j).clone())
            .collect();
        FieldMatrix {
            rows: self.cols,
            cols: self.rows,
            entries,
        }
    }

    /// The reduced row echelon form and the pivot columns (Gauss–Jordan
    /// elimination over the field).
    pub fn rref(&self) -> PolyResult<(FieldMatrix, Vec<usize>)> {
        let mut rows: Vec<Vec<RationalFunction>> = (0..self.rows)
            .map(|i| self.entries[i * self.cols..(i + 1) * self.cols].to_vec())
            .collect();
        let mut pivots: Vec<usize> = Vec::new();
        let mut r: usize = 0;
        for c in 0..self.cols {
            if r >= self.rows {
                break;
            }
            let Some(p) = (r..self.rows).find(|i| !rows[*i][c].is_zero()) else {
                continue;
            };
            rows.swap(r, p);
            let inverse: RationalFunction = rows[r][c].invert()?;
            rows[r] = rows[r]
                .iter()
                .map(|e| if e.is_zero() { Ok(e.clone()) } else { e.mul(&inverse) })
                .collect::<PolyResult<_>>()?;
            for i in 0..self.rows {
                if i != r && !rows[i][c].is_zero() {
                    let factor: RationalFunction = rows[i][c].clone();
                    let updated: Vec<RationalFunction> = rows[i]
                        .iter()
                        .zip(&rows[r])
                        .map(|(a, b)| {
                            if b.is_zero() {
                                Ok(a.clone())
                            } else {
                                a.sub(&factor.mul(b)?)
                            }
                        })
                        .collect::<PolyResult<_>>()?;
                    rows[i] = updated;
                }
            }
            pivots.push(c);
            r += 1;
        }
        Ok((
            FieldMatrix::new(self.rows, self.cols, rows.into_iter().flatten().collect())?,
            pivots,
        ))
    }

    /// A basis of the column vectors `v` with `M v = 0`: one for each
    /// column without a pivot, `1` there (SymPy's basis).
    pub fn nullspace(&self) -> PolyResult<Vec<FieldMatrix>> {
        let (reduced, pivots) = self.rref()?;
        let mut basis: Vec<FieldMatrix> = Vec::new();
        for free in 0..self.cols {
            if pivots.contains(&free) {
                continue;
            }
            let mut v: Vec<RationalFunction> = vec![RationalFunction::zero(); self.cols];
            v[free] = RationalFunction::one();
            for (k, c) in pivots.iter().enumerate() {
                let value: &RationalFunction = reduced.at(k, free);
                if !value.is_zero() {
                    v[*c] = value.neg();
                }
            }
            basis.push(FieldMatrix::column(&v));
        }
        Ok(basis)
    }

    pub fn rank(&self) -> PolyResult<usize> {
        Ok(self.rref()?.1.len())
    }
}
