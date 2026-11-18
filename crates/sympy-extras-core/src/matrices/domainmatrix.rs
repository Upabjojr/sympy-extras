//! Exact linear algebra over a typed domain, ported from
//! `sympy_extras.matrices.domainmatrix`: a dense [`DomainMatrix`] with
//! Bareiss' fraction-free elimination (the determinant, the fraction-free
//! Gauss–Jordan form `rref_den` and what is read off it: rank, null space,
//! inverse, linear systems) and Berkowitz' division-free characteristic
//! polynomial (and the adjugate of a singular matrix by Cayley–Hamilton).
//!
//! References: E. H. Bareiss, *Sylvester's identity and multistep
//! integer-preserving Gaussian elimination*, Math. Comp. 22 (1968);
//! K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//! Algebra* (1992), 9.3; S. J. Berkowitz, Inf. Proc. Letters 18 (1984).

use crate::error::{value_error, PolyResult};
use crate::polys::domains::domain::Domain;

/// A dense `nrows` by `ncols` matrix over `domain` (immutable: every
/// operation returns a new matrix).
#[derive(Debug, Clone)]
pub struct DomainMatrix<K: Domain> {
    pub rows: Vec<Vec<K::Elem>>,
    pub nrows: usize,
    pub ncols: usize,
    pub domain: K,
}

impl<K: Domain + Clone> PartialEq for DomainMatrix<K>
where
    K::Elem: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.domain.name() == other.domain.name() && self.shape() == other.shape() && self.rows == other.rows
    }
}

impl<K: Domain + Clone> DomainMatrix<K>
where
    K::Elem: Clone + PartialEq,
{
    /// `DomainMatrix(rows, domain, ncols)`.
    pub fn new(rows: Vec<Vec<K::Elem>>, domain: K, ncols: Option<usize>) -> PolyResult<DomainMatrix<K>> {
        let width: usize = rows.first().map(|r| r.len()).unwrap_or(ncols.unwrap_or(0));
        if let Some(c) = ncols {
            if c != width {
                return Err(value_error(format!("rows of length {} for {} columns", width, c)));
            }
        }
        if rows.iter().any(|r| r.len() != width) {
            return Err(value_error("rows of different lengths".to_string()));
        }
        Ok(DomainMatrix {
            nrows: rows.len(),
            ncols: width,
            rows,
            domain,
        })
    }

    fn built(&self, rows: Vec<Vec<K::Elem>>, ncols: usize) -> DomainMatrix<K> {
        DomainMatrix {
            nrows: rows.len(),
            ncols,
            rows,
            domain: self.domain.clone(),
        }
    }

    pub fn zeros(nrows: usize, ncols: usize, domain: K) -> DomainMatrix<K> {
        let rows: Vec<Vec<K::Elem>> = (0..nrows).map(|_| vec![domain.zero(); ncols]).collect();
        DomainMatrix {
            rows,
            nrows,
            ncols,
            domain,
        }
    }

    pub fn eye(n: usize, domain: K) -> DomainMatrix<K> {
        let rows: Vec<Vec<K::Elem>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { domain.one() } else { domain.zero() })
                    .collect()
            })
            .collect();
        DomainMatrix {
            rows,
            nrows: n,
            ncols: n,
            domain,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// `repr`: `DomainMatrix([[...], ...], K)`.
    pub fn to_str(&self) -> String {
        let k: &K = &self.domain;
        let body: Vec<String> = self
            .rows
            .iter()
            .map(|row| {
                format!(
                    "[{}]",
                    row.iter().map(|a| k.to_str(a)).collect::<Vec<String>>().join(", ")
                )
            })
            .collect();
        format!("DomainMatrix([{}], {})", body.join(", "), k.name())
    }

    fn square(&self) -> PolyResult<usize> {
        if self.nrows != self.ncols {
            return Err(value_error(format!(
                "a {} x {} matrix is not square",
                self.nrows, self.ncols
            )));
        }
        Ok(self.nrows)
    }

    fn shapes_error(&self, other: &DomainMatrix<K>) -> crate::error::PolyError {
        value_error(format!(
            "shapes ({}, {}) and ({}, {})",
            self.nrows, self.ncols, other.nrows, other.ncols
        ))
    }

    // -- arithmetic

    pub fn add(&self, other: &DomainMatrix<K>) -> PolyResult<DomainMatrix<K>> {
        if self.shape() != other.shape() {
            return Err(self.shapes_error(other));
        }
        let k: &K = &self.domain;
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(r, s)| r.iter().zip(s).map(|(a, b)| k.add(a, b)).collect())
            .collect();
        Ok(self.built(rows, self.ncols))
    }

    pub fn sub(&self, other: &DomainMatrix<K>) -> PolyResult<DomainMatrix<K>> {
        if self.shape() != other.shape() {
            return Err(self.shapes_error(other));
        }
        let k: &K = &self.domain;
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(r, s)| r.iter().zip(s).map(|(a, b)| k.sub(a, b)).collect())
            .collect();
        Ok(self.built(rows, self.ncols))
    }

    pub fn neg(&self) -> DomainMatrix<K> {
        let k: &K = &self.domain;
        let rows = self.rows.iter().map(|r| r.iter().map(|a| k.neg(a)).collect()).collect();
        self.built(rows, self.ncols)
    }

    pub fn mul(&self, other: &DomainMatrix<K>) -> PolyResult<DomainMatrix<K>> {
        if self.ncols != other.nrows {
            return Err(self.shapes_error(other));
        }
        let k: &K = &self.domain;
        let columns: Vec<Vec<K::Elem>> = other.transpose().rows;
        let mut product: Vec<Vec<K::Elem>> = Vec::new();
        for r in &self.rows {
            let mut line: Vec<K::Elem> = Vec::new();
            for column in &columns {
                let mut total: K::Elem = k.zero();
                for (a, b) in r.iter().zip(column) {
                    if !k.is_zero(a) && !k.is_zero(b) {
                        total = k.add(&total, &k.mul(a, b));
                    }
                }
                line.push(total);
            }
            product.push(line);
        }
        Ok(self.built(product, other.ncols))
    }

    pub fn pow(&self, n: i64) -> PolyResult<DomainMatrix<K>> {
        let size: usize = self.square()?;
        if n < 0 {
            return Err(value_error(
                "negative power of a matrix over a ring: use inv_den".to_string(),
            ));
        }
        let mut n: i64 = n;
        let mut result: DomainMatrix<K> = DomainMatrix::eye(size, self.domain.clone());
        let mut base: DomainMatrix<K> = self.clone();
        while n != 0 {
            if n & 1 == 1 {
                result = result.mul(&base)?;
            }
            n >>= 1;
            if n != 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    /// The matrix times the element `c`.
    pub fn scale(&self, c: &K::Elem) -> DomainMatrix<K> {
        let k: &K = &self.domain;
        let rows = self
            .rows
            .iter()
            .map(|r| r.iter().map(|a| k.mul(a, c)).collect())
            .collect();
        self.built(rows, self.ncols)
    }

    pub fn transpose(&self) -> DomainMatrix<K> {
        let rows = (0..self.ncols)
            .map(|j| (0..self.nrows).map(|i| self.rows[i][j].clone()).collect())
            .collect();
        DomainMatrix {
            rows,
            nrows: self.ncols,
            ncols: self.nrows,
            domain: self.domain.clone(),
        }
    }

    /// The columns of `other` appended.
    pub fn hstack(&self, other: &DomainMatrix<K>) -> PolyResult<DomainMatrix<K>> {
        if self.nrows != other.nrows {
            return Err(self.shapes_error(other));
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(r, s)| r.iter().chain(s).cloned().collect())
            .collect();
        Ok(self.built(rows, self.ncols + other.ncols))
    }

    // -- fraction-free elimination

    /// The row from `start` on with the smallest nonzero entry in the
    /// column.
    fn pivot_row(&self, rows: &[Vec<K::Elem>], start: usize, column: usize) -> Option<usize> {
        let k: &K = &self.domain;
        let mut best: Option<(usize, u64)> = None;
        for (i, row) in rows.iter().enumerate().skip(start) {
            let a: &K::Elem = &row[column];
            if k.is_zero(a) {
                continue;
            }
            let size: u64 = k.size(a);
            if best.map_or(true, |(_, s)| size < s) {
                best = Some((i, size));
            }
        }
        best.map(|(i, _)| i)
    }

    /// The determinant, by Bareiss' fraction-free elimination.
    pub fn det(&self) -> PolyResult<K::Elem> {
        let n: usize = self.square()?;
        let k: &K = &self.domain;
        let mut rows: Vec<Vec<K::Elem>> = self.rows.clone();
        let mut previous: K::Elem = k.one();
        let mut negative: bool = false;
        for kk in 0..n {
            let Some(chosen) = self.pivot_row(&rows, kk, kk) else {
                return Ok(k.zero());
            };
            if chosen != kk {
                rows.swap(kk, chosen);
                negative = !negative;
            }
            let pivot: K::Elem = rows[kk][kk].clone();
            let pivot_row: Vec<K::Elem> = rows[kk].clone();
            for row in rows.iter_mut().skip(kk + 1) {
                let factor: K::Elem = row[kk].clone();
                if k.is_zero(&factor) {
                    for entry in row.iter_mut().take(n).skip(kk + 1) {
                        *entry = k.exquo(&k.mul(&pivot, entry), &previous)?;
                    }
                } else {
                    for j in kk + 1..n {
                        let value = k.sub(&k.mul(&pivot, &row[j]), &k.mul(&factor, &pivot_row[j]));
                        row[j] = k.exquo(&value, &previous)?;
                    }
                }
                row[kk] = k.zero();
            }
            previous = pivot;
        }
        Ok(if negative { k.neg(&previous) } else { previous })
    }

    /// `(R, d, pivots)`: `R/d` is the reduced row echelon form, `pivots` the
    /// pivot columns (fraction-free Gauss–Jordan elimination).
    pub fn rref_den(&self) -> PolyResult<(DomainMatrix<K>, K::Elem, Vec<usize>)> {
        let (reduced, den, pivots, _) = self.elimination()?;
        Ok((reduced, den, pivots))
    }

    /// [`rref_den`](Self::rref_den) followed by the pivots used, in order.
    pub fn elimination(&self) -> PolyResult<(DomainMatrix<K>, K::Elem, Vec<usize>, Vec<K::Elem>)> {
        let k: &K = &self.domain;
        let mut rows: Vec<Vec<K::Elem>> = self.rows.clone();
        let mut previous: K::Elem = k.one();
        let mut pivots: Vec<usize> = Vec::new();
        let mut values: Vec<K::Elem> = Vec::new();
        let mut r: usize = 0;
        for column in 0..self.ncols {
            if r == self.nrows {
                break;
            }
            let Some(chosen) = self.pivot_row(&rows, r, column) else {
                continue;
            };
            if chosen != r {
                rows.swap(r, chosen);
            }
            let pivot_row: Vec<K::Elem> = rows[r].clone();
            let pivot: K::Elem = pivot_row[column].clone();
            for (i, row) in rows.iter_mut().enumerate() {
                if i == r {
                    continue;
                }
                let factor: K::Elem = row[column].clone();
                if k.is_zero(&factor) {
                    if !k.is_one(&previous) || !k.is_one(&pivot) {
                        let mut updated: Vec<K::Elem> = Vec::new();
                        for a in row.iter() {
                            updated.push(if k.is_zero(a) {
                                a.clone()
                            } else {
                                k.exquo(&k.mul(&pivot, a), &previous)?
                            });
                        }
                        *row = updated;
                    }
                } else {
                    let mut updated: Vec<K::Elem> = Vec::new();
                    for (a, b) in row.iter().zip(&pivot_row) {
                        // the columns of an appended identity are mostly zero
                        if k.is_zero(b) {
                            updated.push(if k.is_zero(a) {
                                a.clone()
                            } else {
                                k.exquo(&k.mul(&pivot, a), &previous)?
                            });
                        } else if k.is_zero(a) {
                            updated.push(k.exquo(&k.neg(&k.mul(&factor, b)), &previous)?);
                        } else {
                            updated.push(k.exquo(&k.sub(&k.mul(&pivot, a), &k.mul(&factor, b)), &previous)?);
                        }
                    }
                    *row = updated;
                }
            }
            previous = pivot.clone();
            pivots.push(column);
            values.push(pivot);
            r += 1;
        }
        let unit: K::Elem = k.canonical_unit(&previous);
        if !k.is_one(&unit) {
            previous = k.mul(&previous, &unit);
            rows = rows
                .iter()
                .map(|row| row.iter().map(|a| k.mul(a, &unit)).collect())
                .collect();
        }
        Ok((self.built(rows, self.ncols), previous, pivots, values))
    }

    /// The reduced row echelon form and the pivot columns (over a field).
    pub fn rref(&self) -> PolyResult<(DomainMatrix<K>, Vec<usize>)> {
        let k: &K = &self.domain;
        if !k.is_field() {
            return Err(value_error(format!(
                "rref needs a field: use rref_den over {}",
                k.name()
            )));
        }
        let (reduced, den, pivots) = self.rref_den()?;
        let inverse: K::Elem = k.invert(&den)?;
        Ok((reduced.scale(&inverse), pivots))
    }

    pub fn rank(&self) -> PolyResult<usize> {
        Ok(self.rref_den()?.2.len())
    }

    /// A basis of the (right) null space, without fractions.
    pub fn nullspace(&self) -> PolyResult<Vec<Vec<K::Elem>>> {
        let k: &K = &self.domain;
        let (reduced, den, pivots) = self.rref_den()?;
        let mut basis: Vec<Vec<K::Elem>> = Vec::new();
        for free in 0..self.ncols {
            if pivots.contains(&free) {
                continue;
            }
            let mut vector: Vec<K::Elem> = vec![k.zero(); self.ncols];
            vector[free] = den.clone();
            for (i, column) in pivots.iter().enumerate() {
                vector[*column] = k.neg(&reduced.rows[i][free]);
            }
            basis.push(vector);
        }
        Ok(basis)
    }

    /// `(N, d)` with `A N = N A = d I`; `ValueError` for a singular matrix.
    pub fn inv_den(&self) -> PolyResult<(DomainMatrix<K>, K::Elem)> {
        let n: usize = self.square()?;
        let k: K = self.domain.clone();
        let (reduced, den, pivots) = self.hstack(&DomainMatrix::eye(n, k))?.rref_den()?;
        if pivots != (0..n).collect::<Vec<usize>>() {
            return Err(value_error("the matrix is singular".to_string()));
        }
        let rows: Vec<Vec<K::Elem>> = reduced.rows.iter().map(|row| row[n..].to_vec()).collect();
        Ok((self.built(rows, n), den))
    }

    /// The inverse (over a field).
    pub fn inv(&self) -> PolyResult<DomainMatrix<K>> {
        let k: &K = &self.domain;
        if !k.is_field() {
            return Err(value_error(format!("inv needs a field: use inv_den over {}", k.name())));
        }
        let (numerator, den) = self.inv_den()?;
        Ok(numerator.scale(&k.invert(&den)?))
    }

    /// `(X, d)` with `A X = d B`, or `None` when some system has no
    /// solution.
    pub fn solve_den(&self, rhs: &DomainMatrix<K>) -> PolyResult<Option<(DomainMatrix<K>, K::Elem)>> {
        if rhs.nrows != self.nrows {
            return Err(self.shapes_error(rhs));
        }
        let k: &K = &self.domain;
        let (reduced, den, pivots) = self.hstack(rhs)?.rref_den()?;
        if pivots.iter().any(|c| *c >= self.ncols) {
            return Ok(None);
        }
        let mut solution: Vec<Vec<K::Elem>> = (0..self.ncols).map(|_| vec![k.zero(); rhs.ncols]).collect();
        for (i, column) in pivots.iter().enumerate() {
            solution[*column] = reduced.rows[i][self.ncols..].to_vec();
        }
        Ok(Some((self.built(solution, rhs.ncols), den)))
    }

    // -- division-free

    /// The coefficients of the characteristic polynomial `det(lambda I -
    /// A)`, highest degree first, by Berkowitz' division-free algorithm.
    pub fn charpoly(&self) -> PolyResult<Vec<K::Elem>> {
        let n: usize = self.square()?;
        let k: &K = &self.domain;
        let mut polynomial: Vec<K::Elem> = vec![k.one()];
        for kk in 0..n {
            let last: Vec<K::Elem> = self.rows[kk][..kk].to_vec();
            let column: Vec<K::Elem> = (0..kk).map(|i| self.rows[i][kk].clone()).collect();
            let mut toeplitz: Vec<K::Elem> = vec![k.one(), k.neg(&self.rows[kk][kk])];
            let mut vector: Vec<K::Elem> = column;
            for step in 0..kk {
                let mut total: K::Elem = k.zero();
                for (a, b) in last.iter().zip(&vector) {
                    total = k.add(&total, &k.mul(a, b));
                }
                toeplitz.push(k.neg(&total));
                if step + 1 < kk {
                    let mut updated: Vec<K::Elem> = Vec::new();
                    for i in 0..kk {
                        let mut entry: K::Elem = k.zero();
                        for (a, b) in self.rows[i][..kk].iter().zip(&vector) {
                            entry = k.add(&entry, &k.mul(a, b));
                        }
                        updated.push(entry);
                    }
                    vector = updated;
                }
            }
            let mut product: Vec<K::Elem> = Vec::new();
            for i in 0..kk + 2 {
                let mut total: K::Elem = k.zero();
                let low: usize = i.saturating_sub(kk + 1);
                for j in low..=i.min(kk) {
                    total = k.add(&total, &k.mul(&toeplitz[i - j], &polynomial[j]));
                }
                product.push(total);
            }
            polynomial = product;
        }
        Ok(polynomial)
    }

    /// The adjugate: `A adj(A) = det(A) I`.
    pub fn adjugate(&self) -> PolyResult<DomainMatrix<K>> {
        let n: usize = self.square()?;
        let k: &K = &self.domain;
        if n == 0 {
            return Ok(self.clone());
        }
        let determinant: K::Elem = self.det()?;
        if !k.is_zero(&determinant) {
            let (numerator, den) = self.inv_den()?;
            // N/d is the inverse and adj = det * inverse; det = +-d
            if determinant == den {
                return Ok(numerator);
            }
            let mut rows: Vec<Vec<K::Elem>> = Vec::new();
            for r in &numerator.rows {
                let mut line: Vec<K::Elem> = Vec::new();
                for a in r {
                    line.push(k.exquo(&k.mul(a, &determinant), &den)?);
                }
                rows.push(line);
            }
            return Ok(self.built(rows, n));
        }
        // Cayley-Hamilton: adj(A) = (-1)**(n-1) (A**(n-1) + c1 A**(n-2) + ... + c_(n-1) I)
        let coefficients: Vec<K::Elem> = self.charpoly()?;
        let identity: DomainMatrix<K> = DomainMatrix::eye(n, k.clone());
        let mut total: DomainMatrix<K> = identity.clone();
        for c in &coefficients[1..n] {
            total = self.mul(&total)?.add(&identity.scale(c))?;
        }
        Ok(if n % 2 == 1 { total } else { total.neg() })
    }
}
