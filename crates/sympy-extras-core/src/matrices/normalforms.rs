//! Port of `sympy_extras/matrices/normalforms.py`: normal forms of
//! polynomial matrices over the field of the coefficients: the Smith form
//! with its unimodular transformations, the Hermite and Popov forms, the
//! rank; invariant factors, minimal polynomial, companion and Frobenius
//! (rational canonical) forms, elementary divisors, similarity.

#![allow(clippy::needless_range_loop)]

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::{
    characteristic_factors, coefficients as coefficients_of, fresh_symbol_pub, Field, FunctionEntries, GaussianEntries,
    RationalEntries,
};
use crate::matrices::linalg::{complex_lift, kernels as kernels_of, square, PolynomialLift};
use crate::polys::densearith::{dup_add, dup_div, dup_mul, dup_mul_ground, dup_neg, dup_rem};
use crate::polys::domains::domain::Domain;

type R = PolyResult<Basic>;
type Coefficients = Vec<Basic>;
type PolynomialRows = Vec<Vec<Coefficients>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}

/// `S = U M V` with `S` the Smith form (`SmithDecomposition`).
#[derive(Debug, Clone, PartialEq)]
pub struct SmithDecomposition {
    pub s: ExprMatrix,
    pub u: ExprMatrix,
    pub v: ExprMatrix,
}

impl SmithDecomposition {
    /// The nonzero diagonal entries (`invariants`).
    pub fn invariants(&self) -> Vec<Basic> {
        (0..self.s.rows.min(self.s.cols))
            .map(|i| self.s.at(i, i).clone())
            .filter(|e| *e != int(0))
            .collect()
    }
}

type Elem<F> = <<F as Field>::D as Domain>::Elem;
type PMatrix<F> = Vec<Vec<Vec<Elem<F>>>>;

struct Reduction<F: Field> {
    s: PMatrix<F>,
    u: PMatrix<F>,
    v: PMatrix<F>,
    u_inverse: PMatrix<F>,
}

struct Worker<F: Field> {
    field: F,
}

impl<F: Field> Worker<F> {
    fn k(&self) -> &F::D {
        self.field.domain()
    }

    fn polynomial(&self, coefficients: &[Basic]) -> PolyResult<Vec<Elem<F>>> {
        let mut result: Vec<Elem<F>> = coefficients
            .iter()
            .map(|c| self.field.element(c))
            .collect::<PolyResult<_>>()?;
        while !result.is_empty() && self.k().is_zero(&result[0]) {
            result.remove(0);
        }
        Ok(result)
    }

    fn expression(&self, f: &[Elem<F>], x: &Basic) -> R {
        let degree: i64 = f.len() as i64 - 1;
        let mut terms: Vec<Basic> = Vec::new();
        for (k, c) in f.iter().enumerate() {
            terms.push(crate::core::mul::mul(&[
                self.field.expr(c)?,
                crate::core::power::pow(x, &int(degree - k as i64))?,
            ])?);
        }
        crate::core::add::add(&terms)
    }

    fn matrix(&self, rows: &[Vec<Vec<Elem<F>>>], ncols: usize, x: &Basic) -> PolyResult<ExprMatrix> {
        let mut entries: Vec<Basic> = Vec::new();
        for row in rows {
            for f in row {
                entries.push(self.expression(f, x)?);
            }
        }
        ExprMatrix::new(rows.len(), ncols, entries)
    }

    fn identity(&self, n: usize) -> PMatrix<F> {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { vec![self.k().one()] } else { Vec::new() })
                    .collect()
            })
            .collect()
    }

    fn rows_of(&self, rows: &PolynomialRows) -> PolyResult<PMatrix<F>> {
        rows.iter()
            .map(|row| row.iter().map(|f| self.polynomial(f)).collect())
            .collect()
    }

    fn combine(&self, target: &[Vec<Elem<F>>], source: &[Vec<Elem<F>>], q: &[Elem<F>]) -> Vec<Vec<Elem<F>>> {
        let k = self.k();
        target
            .iter()
            .zip(source.iter())
            .map(|(a, b)| {
                if b.is_empty() {
                    a.clone()
                } else {
                    dup_add(a, &dup_mul(q, b, k), k)
                }
            })
            .collect()
    }

    fn reduce(&self, mut b: PMatrix<F>, ncols: usize) -> PolyResult<Reduction<F>> {
        let k = self.k();
        let (m, n) = (b.len(), ncols);
        let (mut u, mut v, mut w) = (self.identity(m), self.identity(n), self.identity(m));
        for t in 0..m.min(n) {
            let mut finished: bool = false;
            loop {
                let mut best: Option<(usize, usize)> = None;
                for i in t..m {
                    for j in t..n {
                        if !b[i][j].is_empty() && best.map_or(true, |(bi, bj)| b[i][j].len() < b[bi][bj].len()) {
                            best = Some((i, j));
                        }
                    }
                }
                let Some((bi, bj)) = best else {
                    finished = true;
                    break;
                };
                if bi != t {
                    b.swap(t, bi);
                    u.swap(t, bi);
                    for row in w.iter_mut() {
                        row.swap(t, bi);
                    }
                }
                if bj != t {
                    for matrix in [&mut b, &mut v] {
                        for row in matrix.iter_mut() {
                            row.swap(t, bj);
                        }
                    }
                }
                let pivot: Vec<Elem<F>> = b[t][t].clone();
                let mut again: bool = false;
                for i in t + 1..m {
                    if !b[i][t].is_empty() {
                        let quotient = dup_div(&b[i][t], &pivot, k)?.0;
                        let q = dup_neg(&quotient, k);
                        self.add_row(&mut b, &mut u, &mut w, i, t, &q);
                        again = again || !b[i][t].is_empty();
                    }
                }
                for j in t + 1..n {
                    if !b[t][j].is_empty() {
                        let quotient = dup_div(&b[t][j], &pivot, k)?.0;
                        let q = dup_neg(&quotient, k);
                        for matrix in [&mut b, &mut v] {
                            for row in matrix.iter_mut() {
                                if !row[t].is_empty() {
                                    row[j] = dup_add(&row[j], &dup_mul(&q, &row[t], k), k);
                                }
                            }
                        }
                        again = again || !b[t][j].is_empty();
                    }
                }
                if again {
                    continue;
                }
                let mut culprit: Option<usize> = None;
                'search: for i in t + 1..m {
                    for j in t + 1..n {
                        if !b[i][j].is_empty() && !dup_rem(&b[i][j], &pivot, k)?.is_empty() {
                            culprit = Some(i);
                            break 'search;
                        }
                    }
                }
                let Some(c) = culprit else { break };
                self.add_row(&mut b, &mut u, &mut w, t, c, &[k.one()]);
            }
            if finished {
                break;
            }
            let leading = b[t][t][0].clone();
            if !k.is_one(&leading) {
                let inverse = k.invert(&leading)?;
                b[t] = b[t].iter().map(|a| dup_mul_ground(a, &inverse, k)).collect();
                u[t] = u[t].iter().map(|a| dup_mul_ground(a, &inverse, k)).collect();
                for row in w.iter_mut() {
                    row[t] = dup_mul_ground(&row[t], &leading, k);
                }
            }
        }
        Ok(Reduction {
            s: b,
            u,
            v,
            u_inverse: w,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn add_row(
        &self,
        b: &mut PMatrix<F>,
        u: &mut PMatrix<F>,
        w: &mut PMatrix<F>,
        target: usize,
        source: usize,
        q: &[Elem<F>],
    ) {
        let k = self.k();
        b[target] = self.combine(&b[target], &b[source], q);
        u[target] = self.combine(&u[target], &u[source], q);
        let negative = dup_neg(q, k);
        for row in w.iter_mut() {
            if !row[target].is_empty() {
                row[source] = dup_add(&row[source], &dup_mul(&negative, &row[target], k), k);
            }
        }
    }

    fn smith(&self, rows: &PolynomialRows, ncols: usize, x: &Basic) -> PolyResult<SmithDecomposition> {
        let result = self.reduce(self.rows_of(rows)?, ncols)?;
        Ok(SmithDecomposition {
            s: self.matrix(&result.s, ncols, x)?,
            u: self.matrix(&result.u, rows.len(), x)?,
            v: self.matrix(&result.v, ncols, x)?,
        })
    }

    fn hermite(
        &self,
        rows: &PolynomialRows,
        ncols: usize,
        x: &Basic,
    ) -> PolyResult<(ExprMatrix, ExprMatrix, Vec<usize>)> {
        let k = self.k();
        let mut b: PMatrix<F> = self.rows_of(rows)?;
        let m: usize = b.len();
        let mut u: PMatrix<F> = self.identity(m);
        let mut pivots: Vec<usize> = Vec::new();
        let mut r: usize = 0;
        for column in 0..ncols {
            if r == m {
                break;
            }
            loop {
                let mut best: Option<usize> = None;
                for i in r..m {
                    if !b[i][column].is_empty() && best.map_or(true, |bi| b[i][column].len() < b[bi][column].len()) {
                        best = Some(i);
                    }
                }
                let Some(bi) = best else { break };
                if bi != r {
                    b.swap(r, bi);
                    u.swap(r, bi);
                }
                let mut again: bool = false;
                for i in r + 1..m {
                    if !b[i][column].is_empty() {
                        let q = dup_neg(&dup_div(&b[i][column], &b[r][column], k)?.0, k);
                        b[i] = self.combine(&b[i], &b[r], &q);
                        u[i] = self.combine(&u[i], &u[r], &q);
                        again = again || !b[i][column].is_empty();
                    }
                }
                if !again {
                    break;
                }
            }
            if b[r][column].is_empty() {
                continue;
            }
            let leading = b[r][column][0].clone();
            if !k.is_one(&leading) {
                let inverse = k.invert(&leading)?;
                b[r] = b[r].iter().map(|a| dup_mul_ground(a, &inverse, k)).collect();
                u[r] = u[r].iter().map(|a| dup_mul_ground(a, &inverse, k)).collect();
            }
            for i in 0..r {
                if !b[i][column].is_empty() {
                    let quotient = dup_div(&b[i][column], &b[r][column], k)?.0;
                    if !quotient.is_empty() {
                        let q = dup_neg(&quotient, k);
                        b[i] = self.combine(&b[i], &b[r], &q);
                        u[i] = self.combine(&u[i], &u[r], &q);
                    }
                }
            }
            pivots.push(column);
            r += 1;
        }
        Ok((self.matrix(&b, ncols, x)?, self.matrix(&u, m, x)?, pivots))
    }

    fn popov(&self, rows: &PolynomialRows, ncols: usize, x: &Basic) -> PolyResult<(ExprMatrix, ExprMatrix)> {
        let k = self.k();
        let mut b: PMatrix<F> = self.rows_of(rows)?;
        let m: usize = b.len();
        let mut u: PMatrix<F> = self.identity(m);
        let pivot = |b: &PMatrix<F>, i: usize| -> (i64, i64) {
            let (mut degree, mut column) = (-1i64, -1i64);
            for (j, entry) in b[i].iter().enumerate() {
                if !entry.is_empty() && entry.len() as i64 > degree {
                    degree = entry.len() as i64 - 1;
                    column = j as i64;
                }
            }
            (degree, column)
        };
        loop {
            let mut seen: Vec<(i64, usize)> = Vec::new();
            let mut clash: Option<(usize, usize)> = None;
            for i in 0..m {
                let (_, column) = pivot(&b, i);
                if column < 0 {
                    continue;
                }
                if let Some((_, first)) = seen.iter().find(|(c, _)| *c == column) {
                    clash = Some((*first, i));
                    break;
                }
                seen.push((column, i));
            }
            let Some((first, second)) = clash else { break };
            let (low, high) = if pivot(&b, first).0 <= pivot(&b, second).0 {
                (first, second)
            } else {
                (second, first)
            };
            let column: usize = pivot(&b, low).1 as usize;
            let shift: usize = (pivot(&b, high).0 - pivot(&b, low).0) as usize;
            let factor = k.neg(&k.mul(&b[high][column][0], &k.invert(&b[low][column][0])?));
            let mut q: Vec<Elem<F>> = vec![factor];
            q.extend(std::iter::repeat(k.zero()).take(shift));
            b[high] = self.combine(&b[high], &b[low], &q);
            u[high] = self.combine(&u[high], &u[low], &q);
        }
        for i in 0..m {
            let (_, column) = pivot(&b, i);
            if column >= 0 && !k.is_one(&b[i][column as usize][0]) {
                let inverse = k.invert(&b[i][column as usize][0])?;
                b[i] = b[i].iter().map(|a| dup_mul_ground(a, &inverse, k)).collect();
                u[i] = u[i].iter().map(|a| dup_mul_ground(a, &inverse, k)).collect();
            }
        }
        let mut ended: bool = false;
        for _ in 0..64 * (m + 1) * (m + 1) {
            let mut changed: bool = false;
            for r in 0..m {
                let (degree, column) = pivot(&b, r);
                if column < 0 {
                    continue;
                }
                let column: usize = column as usize;
                for i in 0..m {
                    if i != r && !b[i][column].is_empty() && b[i][column].len() as i64 > degree {
                        let q = dup_neg(&dup_div(&b[i][column], &b[r][column], k)?.0, k);
                        b[i] = self.combine(&b[i], &b[r], &q);
                        u[i] = self.combine(&u[i], &u[r], &q);
                        changed = true;
                    }
                }
            }
            if !changed {
                ended = true;
                break;
            }
        }
        if !ended {
            return Err(PolyError::Arithmetic(
                "the reduction to the Popov form did not end".to_string(),
            ));
        }
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by_key(|i| {
            let c = pivot(&b, *i).1;
            (c < 0, c)
        });
        let bo: PMatrix<F> = order.iter().map(|i| b[*i].clone()).collect();
        let uo: PMatrix<F> = order.iter().map(|i| u[*i].clone()).collect();
        Ok((self.matrix(&bo, ncols, x)?, self.matrix(&uo, m, x)?))
    }

    #[allow(clippy::type_complexity)]
    fn characteristic(&self, a: &ExprMatrix) -> PolyResult<(Vec<Vec<Elem<F>>>, Reduction<F>)> {
        let k = self.k();
        let n: usize = a.rows;
        let values: Vec<Vec<Elem<F>>> = (0..n)
            .map(|i| {
                a.row(i)
                    .iter()
                    .map(|e| self.field.element(e))
                    .collect::<PolyResult<_>>()
            })
            .collect::<PolyResult<_>>()?;
        let mut b: PMatrix<F> = Vec::new();
        for i in 0..n {
            let mut line: Vec<Vec<Elem<F>>> = Vec::new();
            for j in 0..n {
                let constant = k.neg(&values[i][j]);
                if i == j {
                    line.push(vec![k.one(), constant]);
                } else {
                    line.push(if k.is_zero(&constant) {
                        Vec::new()
                    } else {
                        vec![constant]
                    });
                }
            }
            b.push(line);
        }
        let reduction = self.reduce(b, n)?;
        Ok((values, reduction))
    }

    fn invariants(&self, a: &ExprMatrix) -> PolyResult<Vec<Coefficients>> {
        let (_, result) = self.characteristic(a)?;
        (0..a.rows)
            .filter(|i| result.s[*i][*i].len() > 1)
            .map(|i| result.s[i][i].iter().map(|c| self.field.expr(c)).collect())
            .collect()
    }

    fn multiplicities(&self, a: &ExprMatrix, factor: &[Basic]) -> PolyResult<Vec<usize>> {
        let k = self.k();
        let f = self.polynomial(factor)?;
        let (_, result) = self.characteristic(a)?;
        let mut found: Vec<usize> = Vec::new();
        for i in 0..a.rows {
            let mut s = result.s[i][i].clone();
            let mut exponent: usize = 0;
            while s.len() > 1 {
                let (quotient, remainder) = dup_div(&s, &f, k)?;
                if !remainder.is_empty() {
                    break;
                }
                s = quotient;
                exponent += 1;
            }
            if exponent > 0 {
                found.push(exponent);
            }
        }
        Ok(found)
    }

    fn frobenius(&self, a: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
        let k = self.k();
        let n: usize = a.rows;
        let (values, result) = self.characteristic(a)?;
        let image = |v: &[Elem<F>]| -> Vec<Elem<F>> {
            values
                .iter()
                .map(|row| {
                    let mut total = k.zero();
                    for (x, y) in row.iter().zip(v.iter()) {
                        if !k.is_zero(x) && !k.is_zero(y) {
                            total = k.add(&total, &k.mul(x, y));
                        }
                    }
                    total
                })
                .collect()
        };
        let mut columns: Vec<Vec<Elem<F>>> = Vec::new();
        let mut frob: Vec<Vec<Elem<F>>> = vec![vec![k.zero(); n]; n];
        for i in 0..n {
            let s = &result.s[i][i];
            let degree: i64 = s.len() as i64 - 1;
            if degree < 1 {
                continue;
            }
            let degree: usize = degree as usize;
            let mut generator: Vec<Elem<F>> = vec![k.zero(); n];
            let top: usize = (0..n).map(|r| result.u_inverse[r][i].len()).max().unwrap_or(0);
            for kk in 0..top {
                generator = image(&generator);
                for r in 0..n {
                    let f = &result.u_inverse[r][i];
                    let position: i64 = kk as i64 - (top as i64 - f.len() as i64);
                    if position >= 0 {
                        generator[r] = k.add(&generator[r], &f[position as usize]);
                    }
                }
            }
            let start: usize = columns.len();
            let mut vector = generator;
            for kk in 0..degree {
                columns.push(vector.clone());
                vector = image(&vector);
                if kk > 0 {
                    frob[start + kk][start + kk - 1] = k.one();
                }
                frob[start + kk][start + degree - 1] = k.neg(&s[degree - kk]);
            }
        }
        if columns.len() != n {
            return Err(PolyError::Arithmetic(format!(
                "the degrees of the invariant factors do not add up to {}",
                n
            )));
        }
        let mut p: Vec<Basic> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                p.push(self.field.expr(&columns[j][i])?);
            }
        }
        let f_entries: Vec<Basic> = frob
            .iter()
            .flatten()
            .map(|c| self.field.expr(c))
            .collect::<PolyResult<_>>()?;
        Ok((ExprMatrix::new(n, n, p)?, ExprMatrix::new(n, n, f_entries)?))
    }
}

enum Forms {
    Gaussian(Worker<GaussianEntries>),
    Function(Worker<FunctionEntries>),
    Rational(Worker<RationalEntries>),
}

macro_rules! each {
    ($f:expr, |$w:ident| $body:expr) => {
        match $f {
            Forms::Gaussian($w) => $body,
            Forms::Function($w) => $body,
            Forms::Rational($w) => $body,
        }
    };
}

fn forms(coefficients: &[Basic]) -> PolyResult<Forms> {
    let kernels: Vec<Basic> = kernels_of(&[&ExprMatrix::new(1, coefficients.len(), coefficients.to_vec())?]);
    if let Some(g) = complex_lift(&kernels)? {
        return Ok(Forms::Gaussian(Worker {
            field: GaussianEntries::new(g),
        }));
    }
    if !kernels.is_empty() {
        if !kernels.iter().all(|k| k.is_symbol()) {
            return Err(PolyError::NotImplemented(
                "normal forms of a matrix whose entries are not rational functions of symbols, I and square roots of rationals".to_string(),
            ));
        }
        return Ok(Forms::Function(Worker {
            field: FunctionEntries::new(PolynomialLift::new(&kernels)),
        }));
    }
    Ok(Forms::Rational(Worker {
        field: RationalEntries::new(),
    }))
}

fn entry_coefficients(e: &Basic, x: &Basic) -> PolyResult<Coefficients> {
    let expanded: Basic = crate::core::function::expand_default(e)?;
    if expanded == int(0) {
        Ok(vec![int(0)])
    } else {
        coefficients_of(&expanded, x)
    }
}

fn polynomial_expr(coefficients: &[Basic], x: &Basic) -> R {
    let degree: i64 = coefficients.len() as i64 - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (k, c) in coefficients.iter().enumerate() {
        terms.push(crate::core::mul::mul(&[
            c.clone(),
            crate::core::power::pow(x, &int(degree - k as i64))?,
        ])?);
    }
    crate::core::add::add(&terms)
}

fn polynomial_rows(m: &ExprMatrix, x: &Basic) -> PolyResult<(PolynomialRows, Vec<Basic>)> {
    let rows: PolynomialRows = (0..m.rows)
        .map(|i| m.row(i).iter().map(|e| entry_coefficients(e, x)).collect())
        .collect::<PolyResult<_>>()?;
    let flat: Vec<Basic> = rows.iter().flatten().flatten().cloned().collect();
    Ok((rows, flat))
}

/// The Smith normal form with `S = U M V` (`smith_normal_form`).
pub fn smith_normal_form(m: &ExprMatrix, x: &Basic) -> PolyResult<SmithDecomposition> {
    let (rows, flat) = polynomial_rows(m, x)?;
    each!(forms(&flat)?, |w| w.smith(&rows, m.cols, x))
}

/// The Hermite normal form `H = U M` (`hermite_normal_form`).
pub fn hermite_normal_form(m: &ExprMatrix, x: &Basic) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let (rows, flat) = polynomial_rows(m, x)?;
    let (h, u, _) = each!(forms(&flat)?, |w| w.hermite(&rows, m.cols, x))?;
    Ok((h, u))
}

/// The Popov form `P = U M` (`popov_form`).
pub fn popov_form(m: &ExprMatrix, x: &Basic) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let (rows, flat) = polynomial_rows(m, x)?;
    each!(forms(&flat)?, |w| w.popov(&rows, m.cols, x))
}

/// The rank over the rational functions (`polynomial_matrix_rank`).
pub fn polynomial_matrix_rank(m: &ExprMatrix, x: &Basic) -> PolyResult<usize> {
    let (rows, flat) = polynomial_rows(m, x)?;
    Ok(each!(forms(&flat)?, |w| w.hermite(&rows, m.cols, x))?.2.len())
}

/// Whether `M` is unimodular (`is_unimodular`).
pub fn is_unimodular(m: &ExprMatrix, x: &Basic) -> PolyResult<bool> {
    let n: usize = square(m)?;
    Ok(hermite_normal_form(m, x)?.0 == ExprMatrix::eye(n))
}

/// The invariant factors of `x I - A` (`invariant_factors`).
pub fn invariant_factors(a: &ExprMatrix, x: &Basic) -> PolyResult<Vec<Basic>> {
    square(a)?;
    let found: Vec<Coefficients> = each!(forms(&a.entries)?, |w| w.invariants(a))?;
    found.iter().map(|f| polynomial_expr(f, x)).collect()
}

/// The minimal polynomial (`matrix_minimal_polynomial`).
pub fn matrix_minimal_polynomial(a: &ExprMatrix, x: &Basic) -> R {
    let factors: Vec<Basic> = invariant_factors(a, x)?;
    Ok(factors.last().cloned().unwrap_or_else(|| int(1)))
}

/// The companion matrix of `p` (`companion_matrix`).
pub fn companion_matrix(p: &Basic, x: &Basic) -> PolyResult<ExprMatrix> {
    let cs: Vec<Basic> = coefficients_of(p, x)?;
    let n: usize = cs.len() - 1;
    if n < 1 {
        return Err(value_error(format!(
            "{} is not a polynomial of positive degree in {}",
            p, x
        )));
    }
    let leading: Basic = cs[0].clone();
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if j == n - 1 {
                entries.push(crate::core::mul::div(&crate::core::mul::neg(&cs[n - i])?, &leading)?);
            } else {
                entries.push(int(if i == j + 1 { 1 } else { 0 }));
            }
        }
    }
    ExprMatrix::new(n, n, entries)
}

/// `(P, F)` with `F = P**-1 A P` the Frobenius form (`frobenius_form`).
pub fn frobenius_form(a: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    square(a)?;
    each!(forms(&a.entries)?, |w| w.frobenius(a))
}

/// The elementary divisors `(factor, exponent)` (`elementary_divisors`).
pub fn elementary_divisors(a: &ExprMatrix, x: &Basic) -> PolyResult<Vec<(Basic, usize)>> {
    square(a)?;
    let unknown: Basic = fresh_symbol_pub(a);
    let (_, factors) = characteristic_factors(a, &unknown)?;
    let f = forms(&a.entries)?;
    let mut result: Vec<(Basic, usize)> = Vec::new();
    for (factor, _) in &factors {
        let cs: Vec<Basic> = coefficients_of(factor, &unknown)?;
        let mut exponents: Vec<usize> = each!(&f, |w| w.multiplicities(a, &cs))?;
        exponents.sort();
        for exponent in exponents {
            result.push((crate::core::subs::subs(factor, &unknown, x)?, exponent));
        }
    }
    Ok(result)
}

/// Whether `A` and `B` are similar (`is_similar`).
pub fn is_similar(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<bool> {
    if square(a)? != square(b)? {
        return Ok(false);
    }
    let mut all: Vec<Basic> = a.entries.clone();
    all.extend(b.entries.iter().cloned());
    let f = forms(&all)?;
    Ok(each!(&f, |w| w.invariants(a))? == each!(&f, |w| w.invariants(b))?)
}
