//! Port of `radical_form` and `polynomial_root` of
//! `sympy_extras/polys/roots.py`: a root object (an `IndexedRoot` with
//! rational coefficients; SymPy's `CRootOf` is not in the core) written in
//! radicals when SymPy's `CRootOf(..., radicals=True)` would: the roots of
//! linear, quadratic and binomial polynomials, and those of `roots` for a
//! polynomial whose decomposition has small components, the candidate
//! identified numerically. SymPy's fallback over `EX` is `NotImplemented`.

use crate::core::basic::{Basic, Head};
use crate::core::evalf::{numeric_value, parts, real_int, Real};
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;

use super::polyroots::{decompose_parts, is_cyclotomic_poly, roots, roots_binomial, roots_linear, roots_quadratic};
use super::polytools::Poly;

/// The polynomial and index of a root object; `None` for any other node.
pub fn polynomial_root(e: &Basic) -> PolyResult<Option<(Poly, usize)>> {
    if matches!(e.head(), Head::IndexedRoot) {
        return super::cad::indexedroot::polynomial_root(e);
    }
    Ok(None)
}

fn small_components(poly: &Poly) -> PolyResult<bool> {
    for component in decompose_parts(poly)? {
        if component.degree(0)? <= 2 || component.length() <= 2 || is_cyclotomic_poly(&component)? {
            continue;
        }
        return Ok(false);
    }
    Ok(true)
}

fn trivial_roots(poly: &Poly) -> PolyResult<Option<Vec<Basic>>> {
    let n: i64 = poly.degree(0)?;
    if n == 1 {
        return Ok(Some(roots_linear(poly)?));
    }
    if n == 2 {
        return Ok(Some(roots_quadratic(poly)?));
    }
    let constant: Basic = poly.nth(&[0])?;
    if poly.length() == 2 && !(constant.as_rational().is_some_and(|v| v.is_zero())) {
        return Ok(Some(roots_binomial(poly)?));
    }
    Ok(None)
}

/// The root object in radicals, or `None` when it stays a root object
/// (`radical_form`).
pub fn radical_form(root: &Basic) -> PolyResult<Option<Basic>> {
    let Some((poly, index)) = polynomial_root(root)? else {
        return Ok(None);
    };
    if let Some(direct) = trivial_roots(&poly)? {
        return Ok(direct.get(index).cloned());
    }
    if poly.degree(0)? <= 4 || !small_components(&poly)? {
        return Ok(None);
    }
    let listed = match roots(&poly.as_expr()?, std::slice::from_ref(&poly.gens[0]), true, true, None) {
        Ok(r) => r,
        Err(PolyError::NotImplemented(_)) => {
            return Err(crate::error::not_ported(
                "roots over EX in radical_form (SymPy's roots)",
            ))
        }
        Err(e) => return Err(e),
    };
    let mut candidates: Vec<Basic> = Vec::new();
    for (r, m) in listed.iter() {
        candidates.extend(std::iter::repeat(r.clone()).take(*m));
    }
    if candidates.len() as i64 != poly.degree(0)? {
        return Ok(None);
    }
    identify(root, &candidates)
}

/// The candidate closest to the root if clearly closer than the others
/// (`_identify`, at the 30 digits of `settings.precision`).
fn identify(root: &Basic, candidates: &[Basic]) -> PolyResult<Option<Basic>> {
    const DIGITS: i64 = 30;
    // mpmath's dps_to_prec(30)
    let prec: usize = 103;
    let Some(value) = numeric_value(root, prec) else {
        return Ok(None);
    };
    let mut distances: Vec<(Real, Basic)> = Vec::new();
    for c in candidates {
        let Some(v) = numeric_value(c, prec) else {
            return Ok(None);
        };
        let d = &v - &value;
        let (a, b) = parts(&d);
        distances.push(((&a * &a + &b * &b).sqrt(), c.clone()));
    }
    distances.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let power = |k: i64| -> Real {
        let ten: Real = real_int(&Int::from(10u8), prec + 20);
        let mut p: Real = real_int(&Int::ONE, prec + 20);
        for _ in 0..k {
            p = &p / &ten;
        }
        p
    };
    if distances.len() > 1 && distances[1].0 < power(DIGITS / 3) {
        return Ok(None);
    }
    if distances[0].0 > power(DIGITS * 2 / 3) {
        return Ok(None);
    }
    Ok(Some(distances[0].1.clone()))
}
