//! Port of `sympy_extras/polys/lattice.py`: lattice basis reduction over the
//! integers (the LLL algorithm in de Weger's all-integer form, Cohen's
//! Algorithm 2.6.7) and the Hermite normal form, with the linear algebra
//! over the integers built on the same row operations: the integer kernel,
//! the integer solutions of `A x = b` and the Smith normal form.
//!
//! The vectors are rows of [`Int`]; every division of the algorithm is exact
//! (the Gram determinants `d_i` and `lambda_ij = d_j mu_ij` are integers), so
//! there are no fractions in the reduction and no limit on the entries. The
//! rounding divisions are Python's floor division ([`div_floor`]).

use crate::error::{arithmetic_error, value_error, PolyResult};
use crate::numbers::*;

/// An integer row vector.
pub type Vector = Vec<Int>;

/// An LLL reduced basis with its Gram determinants (`LatticeReduction`).
///
/// `basis` are the reduced row vectors, `gram` the determinants
/// `d_0 = 1, d_1, ..., d_n` of the leading Gram matrices, so that the squared
/// length of the `i`-th Gram–Schmidt vector (from 0) is
/// `gram[i + 1]/gram[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeReduction {
    pub basis: Vec<Vector>,
    pub gram: Vec<Int>,
}

impl LatticeReduction {
    /// The squared lengths of the Gram–Schmidt vectors.
    pub fn norms_squared(&self) -> Vec<Rat> {
        (0..self.basis.len())
            .map(|i| rat_from_parts(self.gram[i + 1].clone(), self.gram[i].clone()))
            .collect()
    }

    /// The first vectors `b_1, ..., b_t` of the basis, `t` the smallest index
    /// such that every later Gram–Schmidt vector is longer than the bound:
    /// every lattice vector `v` with `|v|**2 <= bound_squared` is an integer
    /// combination of them.
    pub fn short_part(&self, bound_squared: &Rat) -> Vec<Vector> {
        let mut t: usize = self.basis.len();
        while t > 0 && rat_from_int(&self.gram[t]) > bound_squared * rat_from_int(&self.gram[t - 1]) {
            t -= 1;
        }
        self.basis[..t].to_vec()
    }
}

fn dot(u: &[Int], v: &[Int]) -> Int {
    let mut total: Int = Int::ZERO;
    for (a, c) in u.iter().zip(v) {
        total += a * c;
    }
    total
}

fn rat_dot(u: &[Rat], v: &[Rat]) -> Rat {
    let mut total: Rat = Rat::ZERO;
    for (a, c) in u.iter().zip(v) {
        total += a * c;
    }
    total
}

/// The index of the first row of `indices` whose entry in `column` has the
/// smallest absolute value (Python's `min` with a key keeps the first).
fn smallest_in_column(work: &[Vector], indices: &[usize], column: usize) -> usize {
    let mut best: usize = indices[0];
    for &i in &indices[1..] {
        if abs(&work[i][column]) < abs(&work[best][column]) {
            best = i;
        }
    }
    best
}

/// Euclid's algorithm on a column: subtracts multiples of the row with the
/// smallest entry from the others (the entries from `start` on) until at most
/// one row has a nonzero entry in the column.
fn euclid_on_column(work: &mut [Vector], column: usize, start: usize) {
    loop {
        let candidates: Vec<usize> = (0..work.len()).filter(|&i| !work[i][column].is_zero()).collect();
        if candidates.len() <= 1 {
            break;
        }
        let smallest: usize = smallest_in_column(work, &candidates, column);
        let pivot: Vector = work[smallest].clone();
        for &i in &candidates {
            if i == smallest {
                continue;
            }
            let q: Int = div_floor(&work[i][column], &pivot[column]);
            if !q.is_zero() {
                let row: &mut Vector = &mut work[i];
                for j in start..row.len() {
                    row[j] -= &q * &pivot[j];
                }
            }
        }
    }
}

/// The Hermite normal form of the lattice spanned by the integer row
/// vectors: the basis in row echelon form with positive pivots and, above
/// each pivot, entries in `[0, pivot)`. It depends on the lattice only.
/// `ValueError` for vectors of different lengths.
pub fn hermite_normal_form(rows: &[Vector]) -> PolyResult<Vec<Vector>> {
    let mut work: Vec<Vector> = rows.to_vec();
    let width: usize = work.first().map_or(0, |row| row.len());
    if work.iter().any(|row| row.len() != width) {
        return Err(value_error("the vectors must have the same length".to_string()));
    }
    let mut result: Vec<Vector> = Vec::new();
    for column in 0..width {
        euclid_on_column(&mut work, column, column);
        let position: Option<usize> = work.iter().position(|row| !row[column].is_zero());
        let Some(position) = position else {
            continue;
        };
        let mut pivot: Vector = work.remove(position);
        if is_negative(&pivot[column]) {
            pivot = pivot.iter().map(|v| -v).collect();
        }
        for previous in result.iter_mut() {
            let q: Int = div_floor(&previous[column], &pivot[column]);
            if !q.is_zero() {
                for j in column..width {
                    previous[j] -= &q * &pivot[j];
                }
            }
        }
        result.push(pivot);
    }
    Ok(result)
}

/// de Weger's integral LLL on the rows `b` (in place), `delta = num/den`:
/// `None` when the rows are linearly dependent, the Gram determinants
/// otherwise.
fn integral_lll(b: &mut [Vector], num: &Int, den: &Int) -> Option<Vec<Int>> {
    let n: usize = b.len();
    // d[i + 1] is the Gram determinant d_i of the text, d[0] = 1
    let mut d: Vec<Int> = vec![Int::ONE; n + 1];
    let mut lam: Vec<Vec<Int>> = vec![vec![Int::ZERO; n]; n];
    if n == 0 {
        return Some(d);
    }
    d[1] = dot(&b[0], &b[0]);
    if d[1].is_zero() {
        return None;
    }
    let mut k: usize = 1;
    let mut kmax: usize = 0;
    while k < n {
        if k > kmax {
            kmax = k;
            for j in 0..=k {
                let mut u: Int = dot(&b[k], &b[j]);
                for i in 0..j {
                    u = div_floor(&(&d[i + 1] * &u - &lam[k][i] * &lam[j][i]), &d[i]);
                }
                if j < k {
                    lam[k][j] = u;
                } else {
                    d[k + 1] = u;
                }
            }
            if d[k + 1].is_zero() {
                return None;
            }
        }
        reduce(b, &mut lam, &d, k, k - 1);
        let l: Int = lam[k][k - 1].clone();
        if den * (&d[k + 1] * &d[k - 1] + &l * &l) < num * &d[k] * &d[k] {
            swap(b, &mut lam, &mut d, k, kmax);
            k = std::cmp::max(1, k - 1);
        } else {
            for j in (0..k.saturating_sub(1)).rev() {
                reduce(b, &mut lam, &d, k, j);
            }
            k += 1;
        }
    }
    Some(d)
}

/// Size reduction of `b_k` against `b_l`.
fn reduce(b: &mut [Vector], lam: &mut [Vec<Int>], d: &[Int], k: usize, l: usize) {
    let dl: &Int = &d[l + 1];
    let value: Int = lam[k][l].clone();
    if Int::from(2) * abs(&value) <= *dl {
        return;
    }
    let q: Int = div_floor(&(Int::from(2) * &value + dl), &(Int::from(2) * dl));
    let bl: Vector = b[l].clone();
    for (a, c) in b[k].iter_mut().zip(&bl) {
        *a -= &q * c;
    }
    lam[k][l] = value - &q * dl;
    // l < k: the row l is in the lower part
    let (low, high) = lam.split_at_mut(k);
    for (a, c) in high[0].iter_mut().zip(&low[l]).take(l) {
        *a -= &q * c;
    }
}

/// Exchange `b_k` and `b_{k-1}` and update the Gram data.
fn swap(b: &mut [Vector], lam: &mut [Vec<Int>], d: &mut [Int], k: usize, kmax: usize) {
    b.swap(k, k - 1);
    {
        let (low, high) = lam.split_at_mut(k);
        for (a, c) in high[0].iter_mut().zip(low[k - 1].iter_mut()).take(k - 1) {
            std::mem::swap(a, c);
        }
    }
    let l: Int = lam[k][k - 1].clone();
    let dk1: Int = d[k].clone();
    let dk: Int = d[k + 1].clone();
    let new: Int = div_floor(&(&d[k - 1] * &dk + &l * &l), &dk1);
    for row in lam.iter_mut().take(kmax + 1).skip(k + 1) {
        let t: Int = row[k].clone();
        row[k] = div_floor(&(&dk * &row[k - 1] - &l * &t), &dk1);
        row[k - 1] = div_floor(&(&new * &t + &l * &row[k]), &dk);
    }
    d[k] = new;
}

/// The default parameter of the Lovász condition, `3/4`.
pub fn default_delta() -> Rat {
    rat(3, 4)
}

/// An LLL reduced basis of the lattice spanned by the integer row vectors,
/// with its Gram determinants (`lll_reduction`). The generators need not be
/// independent (a basis is extracted first), zero vectors are ignored;
/// `1/4 < delta <= 1`. `ValueError` for a bad `delta` or vectors of
/// different lengths.
pub fn lll_reduction(rows: &[Vector], delta: &Rat) -> PolyResult<LatticeReduction> {
    if !(*delta > rat(1, 4) && *delta <= rat(1, 1)) {
        return Err(value_error("delta must be in (1/4, 1]".to_string()));
    }
    let mut b: Vec<Vector> = rows
        .iter()
        .filter(|row| row.iter().any(|v| !v.is_zero()))
        .cloned()
        .collect();
    if let Some(first) = b.first() {
        let width: usize = first.len();
        if b.iter().any(|row| row.len() != width) {
            return Err(value_error("the vectors must have the same length".to_string()));
        }
    }
    let (num, den) = (numer(delta), denom(delta));
    let gram: Vec<Int> = match integral_lll(&mut b, &num, &den) {
        Some(gram) => gram,
        None => {
            b = hermite_normal_form(rows)?;
            match integral_lll(&mut b, &num, &den) {
                Some(gram) => gram,
                None => return Err(arithmetic_error("the Hermite normal form is not a basis".to_string())),
            }
        }
    };
    Ok(LatticeReduction { basis: b, gram })
}

/// An LLL reduced basis of the lattice spanned by the integer row vectors
/// (exact integer arithmetic, entries of any size).
pub fn lll(rows: &[Vector], delta: &Rat) -> PolyResult<Vec<Vector>> {
    Ok(lll_reduction(rows, delta)?.basis)
}

/// Whether the rows are independent and LLL reduced with the parameter
/// `delta` (checked from the definition, with rational Gram–Schmidt
/// coefficients).
pub fn is_lll_reduced(rows: &[Vector], delta: &Rat) -> bool {
    let half: Rat = rat(1, 2);
    let mut star: Vec<Vec<Rat>> = Vec::new();
    let mut norms: Vec<Rat> = Vec::new();
    let mut previous_mu: Rat = Rat::ZERO;
    for (i, row) in rows.iter().enumerate() {
        let v: Vec<Rat> = row.iter().map(rat_from_int).collect();
        let mut current: Vec<Rat> = v.clone();
        for j in 0..i {
            let mu: Rat = rat_dot(&v, &star[j]) / &norms[j];
            if mu > half || mu < -&half {
                return false;
            }
            current = current.iter().zip(&star[j]).map(|(a, c)| a - &mu * c).collect();
            if j + 1 == i {
                previous_mu = mu;
            }
        }
        let norm: Rat = rat_dot(&current, &current);
        if norm.is_zero() {
            return false;
        }
        if i > 0 && norm < (delta - &previous_mu * &previous_mu) * &norms[i - 1] {
            return false;
        }
        star.push(current);
        norms.push(norm);
    }
    true
}

// ---------------------------------------------------------------------------
// linear systems over the integers

/// The echelon rows (`h + u`), the kernel vectors and the pivot positions.
type TransposedEchelon = (Vec<Vector>, Vec<Vector>, Vec<usize>);

/// Unimodular row operations on the transpose of the matrix, carried along
/// on the identity: rows `(h, u)` with `h = A u`. Returns the rows with
/// `h != 0` in echelon form (as `h + u`), the `u` of the rows with `h = 0`
/// (a basis of the kernel), and the pivot positions.
fn transposed_echelon(matrix: &[Vector], unknowns: usize) -> PolyResult<TransposedEchelon> {
    let equations: usize = matrix.len();
    if matrix.iter().any(|row| row.len() != unknowns) {
        return Err(value_error(format!("the rows must have {} entries", unknowns)));
    }
    let mut work: Vec<Vector> = (0..unknowns)
        .map(|j| {
            let mut row: Vector = (0..equations).map(|i| matrix[i][j].clone()).collect();
            row.extend((0..unknowns).map(|k| if k == j { Int::ONE } else { Int::ZERO }));
            row
        })
        .collect();
    let mut echelon: Vec<Vector> = Vec::new();
    let mut pivots: Vec<usize> = Vec::new();
    for column in 0..equations {
        euclid_on_column(&mut work, column, 0);
        let Some(position) = work.iter().position(|row| !row[column].is_zero()) else {
            continue;
        };
        echelon.push(work.remove(position));
        pivots.push(column);
    }
    let kernel: Vec<Vector> = work.into_iter().map(|row| row[equations..].to_vec()).collect();
    Ok((echelon, kernel, pivots))
}

/// An LLL reduced basis of the integer vectors `x` with `A x = 0`, `A` the
/// integer matrix with these rows and `unknowns` columns; it generates all
/// the integer solutions.
pub fn integer_kernel(matrix: &[Vector], unknowns: usize) -> PolyResult<Vec<Vector>> {
    let (_, kernel, _) = transposed_echelon(matrix, unknowns)?;
    lll(&kernel, &default_delta())
}

/// The integer solutions of a linear system: `particular` plus the integer
/// combinations of the vectors of `kernel` (an LLL reduced basis).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerSolutions {
    pub particular: Vector,
    pub kernel: Vec<Vector>,
}

/// Python's rounding `(2*p + q)//(2*q)` of a rational `p/q`.
fn round_half_up(a: &Rat) -> Int {
    let (p, q) = (numer(a), denom(a));
    div_floor(&(Int::from(2) * p + &q), &(Int::from(2) * q))
}

/// `vector` minus a lattice vector close to it (Babai's nearest plane
/// algorithm on a reduced basis): a short representative of its class.
fn nearest_plane(vector: &[Int], basis: &[Vector]) -> Vector {
    let mut star: Vec<Vec<Rat>> = Vec::new();
    let mut norms: Vec<Rat> = Vec::new();
    for (i, row) in basis.iter().enumerate() {
        let r: Vec<Rat> = row.iter().map(rat_from_int).collect();
        let mut current: Vec<Rat> = r.clone();
        for j in 0..i {
            let mu: Rat = rat_dot(&r, &star[j]) / &norms[j];
            current = current.iter().zip(&star[j]).map(|(a, c)| a - &mu * c).collect();
        }
        norms.push(rat_dot(&current, &current));
        star.push(current);
    }
    let mut result: Vector = vector.to_vec();
    for i in (0..basis.len()).rev() {
        let r: Vec<Rat> = result.iter().map(rat_from_int).collect();
        let coefficient: Rat = rat_dot(&r, &star[i]) / &norms[i];
        let q: Int = round_half_up(&coefficient);
        if !q.is_zero() {
            result = result.iter().zip(&basis[i]).map(|(a, c)| a - &q * c).collect();
        }
    }
    result
}

/// All the integer solutions of `A x = b`, or `None` when there is none
/// (`ValueError` without one right-hand side per row).
pub fn solve_integer_system(matrix: &[Vector], rhs: &[Int], unknowns: usize) -> PolyResult<Option<IntegerSolutions>> {
    let equations: usize = matrix.len();
    if rhs.len() != equations {
        return Err(value_error("one right-hand side for each row".to_string()));
    }
    let (echelon, kernel, pivots) = transposed_echelon(matrix, unknowns)?;
    let mut rest: Vector = rhs.to_vec();
    let mut particular: Vector = vec![Int::ZERO; unknowns];
    for (row, &column) in echelon.iter().zip(&pivots) {
        if rest[..column].iter().any(|v| !v.is_zero()) {
            return Ok(None);
        }
        let (q, r) = div_mod_floor(&rest[column], &row[column]);
        if !r.is_zero() {
            return Ok(None);
        }
        if !q.is_zero() {
            rest = rest.iter().zip(&row[..equations]).map(|(a, c)| a - &q * c).collect();
            particular = particular
                .iter()
                .zip(&row[equations..])
                .map(|(a, c)| a + &q * c)
                .collect();
        }
    }
    if rest.iter().any(|v| !v.is_zero()) {
        return Ok(None);
    }
    let reduced: Vec<Vector> = lll(&kernel, &default_delta())?;
    Ok(Some(IntegerSolutions {
        particular: nearest_plane(&particular, &reduced),
        kernel: reduced,
    }))
}

/// The invariant factors `d_1 | d_2 | ...` (positive) of an integer matrix:
/// the nonzero diagonal of its Smith normal form. `ValueError` for rows of
/// different lengths.
pub fn smith_normal_form(matrix: &[Vector]) -> PolyResult<Vector> {
    let mut work: Vec<Vector> = matrix.to_vec();
    let width: usize = work.first().map_or(0, |row| row.len());
    if work.iter().any(|row| row.len() != width) {
        return Err(value_error("the rows must have the same length".to_string()));
    }
    let mut factors: Vector = Vec::new();
    let mut start: usize = 0;
    while start < work.len() && start < width {
        // the entry of smallest absolute value of the remaining block
        let mut best: Option<(usize, usize)> = None;
        for i in start..work.len() {
            for j in start..width {
                let smaller: bool = match best {
                    None => true,
                    Some((bi, bj)) => abs(&work[i][j]) < abs(&work[bi][bj]),
                };
                if !work[i][j].is_zero() && smaller {
                    best = Some((i, j));
                }
            }
        }
        let Some((bi, bj)) = best else {
            break;
        };
        work.swap(start, bi);
        for row in work.iter_mut() {
            row.swap(start, bj);
        }
        let pivot: Int = work[start][start].clone();
        let mut clean: bool = true;
        for i in start + 1..work.len() {
            let q: Int = div_floor(&work[i][start], &pivot);
            if !q.is_zero() {
                let top: Vector = work[start].clone();
                for (a, c) in work[i].iter_mut().zip(&top) {
                    *a -= &q * c;
                }
            }
            if !work[i][start].is_zero() {
                clean = false;
            }
        }
        for j in start + 1..width {
            let q: Int = div_floor(&work[start][j], &pivot);
            if !q.is_zero() {
                for row in work.iter_mut() {
                    let t: Int = &q * &row[start];
                    row[j] -= t;
                }
            }
            if !work[start][j].is_zero() {
                clean = false;
            }
        }
        if !clean {
            // a smaller entry appeared: choose the pivot again
            continue;
        }
        // the pivot must divide the rest of the block
        let offender: Option<usize> =
            (start + 1..work.len()).find(|&i| (start + 1..width).any(|j| !mod_floor(&work[i][j], &pivot).is_zero()));
        if let Some(offender) = offender {
            let other: Vector = work[offender].clone();
            for (a, c) in work[start].iter_mut().zip(&other) {
                *a += c;
            }
            continue;
        }
        factors.push(abs(&pivot));
        start += 1;
    }
    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: &[&[i64]]) -> Vec<Vector> {
        rows.iter().map(|r| r.iter().map(|&v| int(v)).collect()).collect()
    }

    #[test]
    fn docstring_examples() {
        let d: Rat = default_delta();
        assert_eq!(
            lll(&m(&[&[1, 0, 3], &[0, 1, 5], &[0, 0, 7]]), &d).unwrap(),
            m(&[&[-1, -1, -1], &[-1, 2, 0], &[-2, 0, 1]])
        );
        assert_eq!(
            lll(&m(&[&[2, 0], &[0, 2], &[1, 1]]), &d).unwrap(),
            m(&[&[1, 1], &[-1, 1]])
        );
        assert_eq!(
            hermite_normal_form(&m(&[&[4, 6], &[6, 4]])).unwrap(),
            m(&[&[2, 8], &[0, 10]])
        );
        let r = lll_reduction(&m(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]), &d).unwrap();
        assert_eq!(r.gram, vec![int(1), int(1), int(2), int(9)]);
        assert_eq!(r.short_part(&rat(2, 1)), m(&[&[0, 1, 0], &[1, 0, 1]]));
        assert_eq!(
            integer_kernel(&m(&[&[2, 4, 6]]), 3).unwrap(),
            m(&[&[-1, -1, 1], &[-2, 1, 0]])
        );
        let s = solve_integer_system(&m(&[&[1, 2, 3], &[4, 5, 6]]), &[int(6), int(15)], 3)
            .unwrap()
            .unwrap();
        assert_eq!(
            (s.particular, s.kernel),
            (m(&[&[1, 1, 1]])[0].clone(), m(&[&[1, -2, 1]]))
        );
        assert_eq!(
            smith_normal_form(&m(&[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]])).unwrap(),
            vec![int(2), int(6), int(12)]
        );
        assert!(!is_lll_reduced(&m(&[&[1, 0], &[5, 1]]), &d));
    }
}
