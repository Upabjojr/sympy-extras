//! Port of `sympy_extras/matrices/dense.py`: a dense immutable matrix of
//! expressions (`ExprMatrix`), the entries in row-major order; the
//! construction, the arithmetic in the order of the Python's operators,
//! insertion of rows and columns, `subs`, `diff`, `applyfunc`, `has`. The
//! linear algebra is that of `matrices/linalg.rs` and the eigenvalues
//! that of `matrices/eigen.rs`.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};

type R = PolyResult<Basic>;

/// A dense matrix of expressions (`ExprMatrix`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExprMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<Basic>,
}

impl ExprMatrix {
    pub fn new(rows: usize, cols: usize, entries: Vec<Basic>) -> PolyResult<ExprMatrix> {
        if entries.len() != rows * cols {
            return Err(value_error(format!(
                "{} entries for a {} x {} matrix",
                entries.len(),
                rows,
                cols
            )));
        }
        Ok(ExprMatrix { rows, cols, entries })
    }

    /// A column vector (`column`).
    pub fn column(entries: Vec<Basic>) -> ExprMatrix {
        ExprMatrix {
            rows: entries.len(),
            cols: 1,
            entries,
        }
    }

    /// The matrix of the rows (`from_rows`).
    pub fn from_rows(rows: &[Vec<Basic>]) -> PolyResult<ExprMatrix> {
        let cols: usize = rows.first().map(|r| r.len()).unwrap_or(0);
        if rows.iter().any(|r| r.len() != cols) {
            return Err(value_error("rows of different lengths".to_string()));
        }
        ExprMatrix::new(rows.len(), cols, rows.iter().flatten().cloned().collect())
    }

    pub fn zeros(rows: usize, cols: usize) -> ExprMatrix {
        ExprMatrix {
            rows,
            cols,
            entries: vec![Basic::zero(); rows * cols],
        }
    }

    pub fn eye(n: usize) -> ExprMatrix {
        let entries: Vec<Basic> = (0..n * n)
            .map(|k| if k / n == k % n { Basic::one() } else { Basic::zero() })
            .collect();
        ExprMatrix {
            rows: n,
            cols: n,
            entries,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The entry `(i, j)` (`__getitem__`).
    pub fn get(&self, i: usize, j: usize) -> PolyResult<&Basic> {
        if i >= self.rows || j >= self.cols {
            return Err(PolyError::Index(format!(
                "entry ({}, {}) of a {} x {} matrix",
                i, j, self.rows, self.cols
            )));
        }
        Ok(&self.entries[i * self.cols + j])
    }

    pub fn at(&self, i: usize, j: usize) -> &Basic {
        &self.entries[i * self.cols + j]
    }

    pub fn row(&self, i: usize) -> &[Basic] {
        &self.entries[i * self.cols..(i + 1) * self.cols]
    }

    /// The matrix with one entry changed (`with_entry`).
    pub fn with_entry(&self, i: usize, j: usize, value: Basic) -> ExprMatrix {
        let mut entries: Vec<Basic> = self.entries.clone();
        entries[i * self.cols + j] = value;
        ExprMatrix {
            rows: self.rows,
            cols: self.cols,
            entries,
        }
    }

    /// `ExprMatrix([[...], ...])` (`__repr__`).
    pub fn repr(&self) -> String {
        let rows: Vec<String> = (0..self.rows)
            .map(|i| {
                format!(
                    "[{}]",
                    self.row(i)
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            })
            .collect();
        format!("ExprMatrix([{}])", rows.join(", "))
    }

    pub fn transpose(&self) -> ExprMatrix {
        let mut entries: Vec<Basic> = Vec::new();
        for j in 0..self.cols {
            for i in 0..self.rows {
                entries.push(self.at(i, j).clone());
            }
        }
        ExprMatrix {
            rows: self.cols,
            cols: self.rows,
            entries,
        }
    }

    fn map(&self, f: &dyn Fn(&Basic) -> R) -> PolyResult<ExprMatrix> {
        Ok(ExprMatrix {
            rows: self.rows,
            cols: self.cols,
            entries: self.entries.iter().map(f).collect::<PolyResult<_>>()?,
        })
    }

    /// `self + other` (`__add__`).
    pub fn add(&self, other: &ExprMatrix) -> PolyResult<ExprMatrix> {
        if self.shape() != other.shape() {
            return Err(value_error(format!(
                "shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let entries: Vec<Basic> = self
            .entries
            .iter()
            .zip(other.entries.iter())
            .map(|(a, b)| crate::core::add::add(&[a.clone(), b.clone()]))
            .collect::<PolyResult<_>>()?;
        Ok(ExprMatrix {
            rows: self.rows,
            cols: self.cols,
            entries,
        })
    }

    /// `self - other`: `self + (-other)` (`__sub__`).
    pub fn sub(&self, other: &ExprMatrix) -> PolyResult<ExprMatrix> {
        self.add(&other.neg()?)
    }

    pub fn neg(&self) -> PolyResult<ExprMatrix> {
        self.map(&|a| crate::core::mul::neg(a))
    }

    /// `self * c` for a scalar (`__mul__`).
    pub fn scale(&self, c: &Basic) -> PolyResult<ExprMatrix> {
        self.map(&|a| crate::core::mul::mul(&[a.clone(), c.clone()]))
    }

    /// `c * self` (`__rmul__`).
    pub fn rscale(&self, c: &Basic) -> PolyResult<ExprMatrix> {
        self.map(&|a| crate::core::mul::mul(&[c.clone(), a.clone()]))
    }

    /// `self / c` (`__truediv__`).
    pub fn div(&self, c: &Basic) -> PolyResult<ExprMatrix> {
        self.map(&|a| crate::core::mul::div(a, c))
    }

    /// The matrix product (`__mul__`).
    pub fn matmul(&self, other: &ExprMatrix) -> PolyResult<ExprMatrix> {
        if self.cols != other.rows {
            return Err(value_error(format!(
                "shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..self.rows {
            let r: &[Basic] = self.row(i);
            for j in 0..other.cols {
                let terms: Vec<Basic> = (0..self.cols)
                    .map(|k| crate::core::mul::mul(&[r[k].clone(), other.at(k, j).clone()]))
                    .collect::<PolyResult<_>>()?;
                entries.push(crate::core::add::add(&terms)?);
            }
        }
        Ok(ExprMatrix {
            rows: self.rows,
            cols: other.cols,
            entries,
        })
    }

    pub fn col_insert(&self, pos: usize, other: &ExprMatrix) -> PolyResult<ExprMatrix> {
        if other.rows != self.rows {
            return Err(value_error(format!(
                "{} rows inserted in a matrix of {}",
                other.rows, self.rows
            )));
        }
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..self.rows {
            let row: &[Basic] = self.row(i);
            let p: usize = pos.min(row.len());
            entries.extend(row[..p].iter().cloned());
            entries.extend(other.row(i).iter().cloned());
            entries.extend(row[p..].iter().cloned());
        }
        Ok(ExprMatrix {
            rows: self.rows,
            cols: self.cols + other.cols,
            entries,
        })
    }

    pub fn row_insert(&self, pos: usize, other: &ExprMatrix) -> PolyResult<ExprMatrix> {
        if self.rows > 0 && other.cols != self.cols {
            return Err(value_error(format!(
                "{} columns inserted in a matrix of {}",
                other.cols, self.cols
            )));
        }
        let p: usize = pos.min(self.rows);
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..p {
            entries.extend(self.row(i).iter().cloned());
        }
        entries.extend(other.entries.iter().cloned());
        for i in p..self.rows {
            entries.extend(self.row(i).iter().cloned());
        }
        Ok(ExprMatrix {
            rows: self.rows + other.rows,
            cols: other.cols,
            entries,
        })
    }

    pub fn applyfunc(&self, f: &dyn Fn(&Basic) -> R) -> PolyResult<ExprMatrix> {
        self.map(f)
    }

    /// Every entry substituted (`subs` with pairs, in order).
    pub fn subs(&self, pairs: &[(Basic, Basic)]) -> PolyResult<ExprMatrix> {
        self.map(&|a| crate::core::subs::subs_pairs(a, pairs))
    }

    pub fn xreplace(&self, rule: &HashMap<Basic, Basic>) -> ExprMatrix {
        ExprMatrix {
            rows: self.rows,
            cols: self.cols,
            entries: self.entries.iter().map(|a| a.xreplace(rule)).collect(),
        }
    }

    pub fn diff(&self, x: &Basic) -> PolyResult<ExprMatrix> {
        self.map(&|a| crate::core::function::derivative(a, x))
    }

    pub fn has(&self, pattern: &Basic) -> bool {
        self.entries.iter().any(|a| a.has(pattern))
    }

    /// The matrix as `Tuple(rows, cols, Tuple(entries))` for the bridge.
    pub fn as_tuple(&self) -> Basic {
        use crate::core::basic::Head;
        Basic::new(
            Head::Tuple,
            vec![
                Basic::int(self.rows as i64),
                Basic::int(self.cols as i64),
                Basic::new(Head::Tuple, self.entries.clone()),
            ],
        )
    }

    /// The inverse of [`ExprMatrix::as_tuple`].
    pub fn from_tuple(t: &Basic) -> PolyResult<ExprMatrix> {
        let a = t.args();
        let size = |b: &Basic| -> PolyResult<usize> {
            b.to_string()
                .parse::<usize>()
                .map_err(|_| value_error(format!("a size is expected, got {}", b)))
        };
        if a.len() != 3 {
            return Err(value_error(format!("a matrix tuple is expected, got {}", t)));
        }
        ExprMatrix::new(size(&a[0])?, size(&a[1])?, a[2].args().to_vec())
    }
}
