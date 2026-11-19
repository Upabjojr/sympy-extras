//! Port of `sympy_extras/solvers/stability.py`: stability of polynomials
//! and matrices by the Routh–Hurwitz criterion (all roots in the left half
//! plane, or in the unit disc through the Möbius map `x -> (x + 1)/(x -
//! 1)`): Hurwitz matrices and determinants, the Routh array, the
//! conditions on the parameters.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::is;
use crate::core::logic::boolean;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::{coefficients as coefficients_of, fresh_symbol_pub};
use crate::matrices::linalg::{charpoly, det, normal};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}

/// The unevaluated `Ne(a, 0)`, `Gt(a, 0)` of the Python.
fn relation(op: RelOp, a: &Basic) -> Basic {
    Basic::new(Head::Relational(op), vec![a.clone(), int(0)])
}

fn polynomial_coefficients(p: &Basic, x: &Basic) -> PolyResult<(Vec<Basic>, Basic)> {
    let (numerator, denominator) = crate::simplify::radsimp::fraction(&normal(p)?)?;
    if denominator.has(x) {
        return Err(value_error(format!("{} is not a polynomial in {}", p, x)));
    }
    let expanded: Basic = expand(&numerator)?;
    if expanded == int(0) {
        return Err(value_error("the zero polynomial".to_string()));
    }
    Ok((coefficients_of(&expanded, x)?, denominator))
}

/// The Hurwitz matrix of `p` (`hurwitz_matrix`).
pub fn hurwitz_matrix(p: &Basic, x: &Basic) -> PolyResult<ExprMatrix> {
    let (cs, _) = polynomial_coefficients(p, x)?;
    let n: i64 = cs.len() as i64 - 1;
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let k: i64 = 2 * j - i + 1;
            entries.push(if (0..=n).contains(&k) {
                cs[k as usize].clone()
            } else {
                int(0)
            });
        }
    }
    ExprMatrix::new(n as usize, n as usize, entries)
}

/// Its leading principal minors (`hurwitz_determinants`).
pub fn hurwitz_determinants(p: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let h: ExprMatrix = hurwitz_matrix(p, x)?;
    let mut out: Vec<Basic> = Vec::new();
    for k in 1..=h.rows {
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..k {
            for j in 0..k {
                entries.push(h.at(i, j).clone());
            }
        }
        out.push(det(&ExprMatrix::new(k, k, entries)?)?);
    }
    Ok(out)
}

/// The Routh array (`routh_array`).
pub fn routh_array(p: &Basic, x: &Basic) -> PolyResult<Vec<Vec<Basic>>> {
    let (cs, _) = polynomial_coefficients(p, x)?;
    let n: usize = cs.len() - 1;
    let width: usize = n / 2 + 1;
    let padded = |mut row: Vec<Basic>| -> Vec<Basic> {
        while row.len() < width {
            row.push(int(0));
        }
        row
    };
    let mut rows: Vec<Vec<Basic>> = vec![padded(cs.iter().step_by(2).cloned().collect())];
    if n >= 1 {
        rows.push(padded(cs.iter().skip(1).step_by(2).cloned().collect()));
    }
    for _ in 2..=n {
        let (above, last) = (rows[rows.len() - 2].clone(), rows[rows.len() - 1].clone());
        let pivot: Basic = last[0].clone();
        if pivot == int(0) {
            return Err(value_error(
                "a zero in the first column of the Routh array: use stability_conditions".to_string(),
            ));
        }
        let mut row: Vec<Basic> = Vec::new();
        for k in 0..width - 1 {
            let difference: Basic = crate::core::add::sub(&m2(&pivot, &above[k + 1])?, &m2(&above[0], &last[k + 1])?)?;
            row.push(normal(&crate::core::mul::div(&difference, &pivot)?)?);
        }
        rows.push(padded(row));
    }
    Ok(rows)
}

fn moebius(cs: &[Basic], x: &Basic) -> PolyResult<Vec<Basic>> {
    let n: i64 = cs.len() as i64 - 1;
    let (plus, minus) = (
        crate::core::add::add(&[x.clone(), int(1)])?,
        crate::core::add::sub(x, &int(1))?,
    );
    let mut terms: Vec<Basic> = Vec::new();
    for (k, c) in cs.iter().enumerate() {
        terms.push(crate::core::mul::mul(&[
            c.clone(),
            pw(&plus, &int(n - k as i64))?,
            pw(&minus, &int(k as i64))?,
        ])?);
    }
    let total: Basic = crate::core::add::add(&terms)?;
    let expanded: Basic = expand(&total)?;
    let result: Vec<Basic> = if expanded != int(0) {
        coefficients_of(&expanded, x)?
    } else {
        vec![int(0)]
    };
    let mut out: Vec<Basic> = vec![int(0); (n as usize + 1).saturating_sub(result.len())];
    out.extend(result);
    Ok(out)
}

/// A polynomial or a matrix (its characteristic polynomial).
#[derive(Debug, Clone)]
pub enum Subject {
    Polynomial(Basic),
    Matrix(ExprMatrix),
}

/// The conditions on the parameters for all the roots to lie in the left
/// half plane (the unit disc when `discrete`) (`stability_conditions`).
pub fn stability_conditions(p: &Subject, x: Option<&Basic>, discrete: bool, simplify: bool) -> R {
    let (variable, polynomial): (Basic, Basic) = match p {
        Subject::Matrix(m) => {
            let v: Basic = match x {
                Some(x) => x.clone(),
                None => fresh_symbol_pub(m),
            };
            let c: Basic = charpoly(m, &v)?;
            (v, c)
        }
        Subject::Polynomial(e) => match x {
            None => return Err(value_error("the variable of the polynomial is needed".to_string())),
            Some(x) => (x.clone(), e.clone()),
        },
    };
    let (mut cs, denominator) = polynomial_coefficients(&polynomial, &variable)?;
    let mut conditions: Vec<Basic> = Vec::new();
    if !denominator.free_symbols().is_empty() {
        conditions.push(relation(RelOp::Ne, &denominator));
    }
    if discrete {
        cs = moebius(&cs, &variable)?;
    }
    let n: usize = cs.len() - 1;
    let leading: Basic = cs[0].clone();
    if leading == int(0) {
        return Ok(boolean(false));
    }
    if !leading.free_symbols().is_empty() {
        conditions.push(relation(RelOp::Ne, &leading));
    }
    if n > 0 {
        let mut terms: Vec<Basic> = Vec::new();
        for (k, c) in cs.iter().enumerate() {
            terms.push(crate::core::mul::mul(&[
                c.clone(),
                pw(&variable, &int((n - k) as i64))?,
            ])?);
        }
        let rebuilt: Basic = crate::core::add::add(&terms)?;
        for (index, minor) in hurwitz_determinants(&rebuilt, &variable)?.iter().enumerate() {
            let order: i64 = index as i64 + 1;
            let quantity: Basic = normal(&m2(minor, &pw(&leading, &int(order))?)?)?;
            if quantity.free_symbols().is_empty() {
                if is(&quantity, Predicate::positive) != Some(true) {
                    return Ok(boolean(false));
                }
                continue;
            }
            conditions.push(relation(RelOp::Gt, &quantity));
        }
    }
    if conditions.is_empty() {
        return Ok(boolean(true));
    }
    let formula: Basic = if conditions.len() == 1 {
        conditions[0].clone()
    } else {
        Basic::new(Head::And, conditions.clone())
    };
    if !simplify {
        return Ok(formula);
    }
    match crate::assumptions::resolve::resolve(&formula, None, &[], None) {
        Ok(r) => Ok(r),
        Err(PolyError::Value(_)) => Ok(formula),
        Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => Ok(formula),
        Err(e) => Err(e),
    }
}

/// Whether `p` is stable; `ValueError` when it depends on parameters
/// (`is_stable`).
pub fn is_stable(p: &Subject, x: Option<&Basic>, discrete: bool) -> PolyResult<bool> {
    let result: Basic = stability_conditions(p, x, discrete, true)?;
    if result == boolean(true) {
        return Ok(true);
    }
    if result == boolean(false) {
        return Ok(false);
    }
    Err(value_error(format!("the stability depends on parameters: {}", result)))
}
