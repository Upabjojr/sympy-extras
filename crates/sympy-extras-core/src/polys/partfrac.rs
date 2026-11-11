//! Port of `sympy_extras/polys/partfrac.py`: [`apart`], the partial
//! fraction decomposition of a univariate rational function by
//! undetermined coefficients over the factors of the denominator.

// PORTED FROM SYMPY: sympy/polys/partfrac.py (apart, apart_undetermined_coeffs)
// (SymPy 1.14), through sympy_extras; BSD 3-clause license, see
// sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::basic::Basic;
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::{not_ported, polys_error, value_error, PolyResult, PolysErrorKind};

use super::polytools::{cancel, common_gens, factor, Poly};

fn is_zero(e: &Basic) -> bool {
    e.is_number_class() && e.as_rational().is_some_and(|v| v.is_zero())
}

fn cancelled(e: &Basic) -> PolyResult<Basic> {
    cancel(e, &[])
}

/// The unique solution of a square linear system with rational function
/// entries (Gauss-Jordan elimination, every entry cancelled).
fn solve_linear(matrix: &[Vec<Basic>], rhs: &[Basic]) -> PolyResult<Vec<Basic>> {
    let n: usize = rhs.len();
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for (row, r) in matrix.iter().zip(rhs) {
        let mut entries: Vec<Basic> = Vec::new();
        for v in row {
            entries.push(cancelled(v)?);
        }
        entries.push(cancelled(r)?);
        rows.push(entries);
    }
    for col in 0..n {
        let Some(pivot) = (col..n).find(|&i| !is_zero(&rows[i][col])) else {
            return Err(value_error(
                "singular system in the partial fraction decomposition".to_string(),
            ));
        };
        rows.swap(col, pivot);
        let inverse: Basic = pow(&rows[col][col], &Basic::negative_one())?;
        let mut scaled: Vec<Basic> = Vec::new();
        for v in &rows[col] {
            scaled.push(cancelled(&mul(&[v.clone(), inverse.clone()])?)?);
        }
        rows[col] = scaled;
        for i in 0..n {
            if i != col && !is_zero(&rows[i][col]) {
                let factor_i: Basic = rows[i][col].clone();
                let mut reduced: Vec<Basic> = Vec::new();
                for (v, w) in rows[i].iter().zip(&rows[col]) {
                    let term: Basic = mul(&[Basic::negative_one(), factor_i.clone(), w.clone()])?;
                    reduced.push(cancelled(&add(&[v.clone(), term])?)?);
                }
                rows[i] = reduced;
            }
        }
    }
    Ok(rows.into_iter().map(|row| row[n].clone()).collect())
}

/// `P/Q` as a sum of `h/f**i` over the factors `f` of `Q`
/// (`apart_undetermined_coeffs`).
fn undetermined_coefficients(p: &Poly, q: &Poly) -> PolyResult<Basic> {
    let x: Basic = q.gens[0].clone();
    let (_, factors) = q.factor_list()?;
    let mut blocks: Vec<(usize, Poly, Poly, usize)> = Vec::new();
    for (f, k) in factors {
        let n: usize = f.degree(0)?.max(0) as usize;
        let mut current: Poly = q.clone();
        for i in 1..=k {
            current = current.exquo(&f)?;
            blocks.push((n, current.clone(), f.clone(), i));
        }
    }
    let size: usize = blocks.iter().map(|b| b.0).sum();
    let mut columns: Vec<Vec<Basic>> = Vec::new();
    for (n, current, _, _) in &blocks {
        for j in 0..*n {
            let monomial: Poly = Poly::new(&pow(&x, &Basic::int((n - 1 - j) as i64))?, std::slice::from_ref(&x))?;
            let shifted: Poly = current.mul(&monomial)?;
            let mut column: Vec<Basic> = Vec::new();
            for k in 0..size {
                column.push(shifted.nth(&[k as u32])?);
            }
            columns.push(column);
        }
    }
    let matrix: Vec<Vec<Basic>> = (0..size)
        .map(|r| columns.iter().map(|c| c[r].clone()).collect())
        .collect();
    let mut rhs: Vec<Basic> = Vec::new();
    for k in 0..size {
        rhs.push(p.nth(&[k as u32])?);
    }
    let solution: Vec<Basic> = solve_linear(&matrix, &rhs)?;
    let mut result: Basic = Basic::zero();
    let mut position: usize = 0;
    for (n, _, f, i) in &blocks {
        let mut terms: Vec<Basic> = Vec::new();
        for j in 0..*n {
            terms.push(mul(&[
                solution[position + j].clone(),
                pow(&x, &Basic::int((n - 1 - j) as i64))?,
            ])?);
        }
        let h: Basic = add(&terms)?;
        position += n;
        let power: Basic = pow(&f.as_expr()?, &Basic::int(*i as i64))?;
        result = add(&[result, mul(&[h, pow(&power, &Basic::negative_one())?])?])?;
    }
    Ok(result)
}

/// The partial fraction decomposition of `e` in `x` (the only generator
/// when `x` is `None`); `full` (`RootSum`), non-commutative and
/// multivariate decompositions are `NotImplemented`.
pub fn apart(e: &Basic, x: Option<&Basic>, full: bool) -> PolyResult<Basic> {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    if full {
        return Err(not_ported("the full partial fraction decomposition (RootSum)"));
    }
    if crate::core::expr::is_false(e, crate::core::assumptions::Predicate::commutative) {
        return Err(not_ported("apart of noncommutative expressions"));
    }
    let (numer, denom) = crate::core::expr::as_numer_denom(e)?;
    let gens: Vec<Basic> = match x {
        Some(x) => vec![x.clone()],
        None => common_gens(&numer, &denom)?,
    };
    if x.is_none() && gens.iter().any(crate::core::assumptions::is_number) {
        return Err(polys_error(
            PolysErrorKind::CoercionFailed,
            "algebraic coefficients (extension=True) are not ported yet".to_string(),
        ));
    }
    if gens.len() > 1 {
        let c: Basic = cancelled(e)?;
        if c != *e {
            return apart(&c, x, full);
        }
        return Err(not_ported("multivariate partial fraction decomposition"));
    }
    let (common, p, q) = Poly::new(&numer, &gens)?.cancel(&Poly::new(&denom, &gens)?)?;
    let (polynomial, p) = p.div(&q)?;
    let partial: Basic = if q.degree(0)? <= 1 {
        mul(&[p.as_expr()?, pow(&q.as_expr()?, &Basic::negative_one())?])?
    } else {
        undetermined_coefficients(&p, &q)?
    };
    let parts: Vec<Basic> = if partial.is_add() {
        partial.args().to_vec()
    } else {
        vec![partial]
    };
    let mut terms: Vec<Basic> = Vec::new();
    for t in parts {
        terms.push(factor(&t, &[], false, true)?);
    }
    mul(&[common, add(&[polynomial.as_expr()?, add(&terms)?])?])
}
