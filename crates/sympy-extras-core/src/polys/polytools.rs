//! Port of the helpers of `sympy_extras/polys/polytools.py` which need no
//! expressions: `_factor_all` (the factorisation over `QQ` in all the
//! variables of a ring) and `_resultant_first` (the resultant in the first
//! variable). The class `Poly` and the rest of the module wait for the core.

use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::densebasic::dup_from_dict;
use super::domains::{IntegerRing, RationalField, SparsePolynomialRing, QQ, ZZ};
use super::euclidtools::{dup_prs_resultant, dup_resultant};
use super::multifactor as mf;
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

/// The coefficient and the integer primitive factors (positive leading
/// coefficients), sorted, of `f` over `QQ` in all the variables of its ring
/// (SymPy's `dup_factor_list`/`dmp_factor_list` over `ZZ` or `QQ`; the
/// port of `polytools._factor_all`); `ValueError` for zero.
pub fn factor_all(f: &PolyElement<RationalField>) -> PolyResult<(Rat, mf::Factors)> {
    if f.is_zero() {
        return Err(value_error("min() arg is an empty sequence".to_string()));
    }
    let names: Vec<String> = f.ring.names.clone();
    let n: usize = names.len();
    let exponents: Vec<u32> = (0..n)
        .map(|i| f.terms.keys().map(|m| m[i]).min().expect("a nonzero polynomial"))
        .collect();
    let shifted: Terms<Rat> = f
        .terms
        .iter()
        .map(|(m, c)| (m.iter().zip(&exponents).map(|(e, s)| e - s).collect(), c.clone()))
        .collect();
    let mut content: Rat = Rat::ZERO;
    for c in shifted.values() {
        let (cn, cd) = (numer(&content), denom(&content));
        let (vn, vd) = (numer(c), denom(c));
        content = rat_from_parts(gcd(&cn, &vn), &cd * &vd / gcd(&cd, &vd));
    }
    let ints: Terms<Int> = shifted
        .iter()
        .map(|(m, c)| (m.clone(), numer(&(c / &content))))
        .collect();
    let ring: Arc<PolyRing<IntegerRing>> = PolyRing::new(ZZ, names.clone(), MonomialOrder::Lex)?;
    let (coefficient, mut factors): (Int, mf::Factors) = if n == 1 {
        // dup_zz_factor, sorted below as the Python sorts it
        mf::zz_factor(&ring.from_dict(ints))?
    } else {
        let used: Vec<usize> = (0..n).filter(|&i| ints.keys().any(|m| m[i] != 0)).collect();
        if used.is_empty() {
            let zero: Monomial = vec![0; n];
            (ints.get(&zero).cloned().unwrap_or(Int::ZERO), Vec::new())
        } else {
            let sub: Arc<PolyRing<IntegerRing>> =
                PolyRing::new(ZZ, used.iter().map(|&i| names[i].clone()).collect(), MonomialOrder::Lex)?;
            let g: PolyElement<IntegerRing> = sub.from_dict(
                ints.iter()
                    .map(|(m, c)| (used.iter().map(|&i| m[i]).collect(), c.clone()))
                    .collect(),
            );
            let (coefficient, sub_factors) = mf::zz_factor(&g)?;
            let mut factors: mf::Factors = Vec::new();
            for (h, k) in sub_factors {
                let mut terms: Terms<Int> = Terms::new();
                for (m, value) in &h.terms {
                    let mut full: Monomial = vec![0; n];
                    for (position, e) in used.iter().zip(m) {
                        full[*position] = *e;
                    }
                    terms.insert(full, value.clone());
                }
                factors.push((ring.from_dict(terms), k));
            }
            (coefficient, factors)
        }
    };
    for i in (0..n).rev() {
        if exponents[i] != 0 {
            let monomial: Monomial = (0..n).map(|k| u32::from(k == i)).collect();
            factors.insert(0, (ring.term(monomial, Int::ONE), exponents[i] as usize));
        }
    }
    // the key of the univariate branch (degree, multiplicity, dense
    // coefficients) is that of `_sorted` in one variable
    Ok((rat_from_int(&coefficient) * content, mf::sorted(factors)))
}

/// The resultant in the first variable of two polynomials over `QQ`, as a
/// polynomial in all the variables (constant in the first); `integral`
/// computes it over `ZZ` (the coefficients must be integers).
pub fn resultant_first(
    f: &PolyElement<RationalField>,
    g: &PolyElement<RationalField>,
    names: &[String],
    integral: bool,
) -> PolyElement<RationalField> {
    if names.len() == 1 {
        let rf: Vec<Rat> = dup_from_dict(&f.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect(), &QQ);
        let rg: Vec<Rat> = dup_from_dict(&g.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect(), &QQ);
        let value: Rat = dup_resultant(&rf, &rg, &QQ);
        let mut terms: Terms<Rat> = Terms::new();
        terms.insert(vec![0], value);
        return f.ring.from_dict(terms);
    }
    if integral {
        let rz: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::new(ZZ, names.to_vec());
        let inner: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::new(ZZ, names[1..].to_vec());
        let to_int = |p: &PolyElement<RationalField>| -> PolyElement<IntegerRing> {
            rz.ring
                .from_dict(p.terms.iter().map(|(m, c)| (m.clone(), numer(c))).collect())
        };
        let r: PolyElement<IntegerRing> = dup_prs_resultant(
            &rz.to_univariate(&to_int(f), &inner.ring, &inner),
            &rz.to_univariate(&to_int(g), &inner.ring, &inner),
            &inner,
        );
        return f.ring.from_dict(
            r.terms
                .iter()
                .map(|(m, c)| {
                    let mut key: Monomial = vec![0];
                    key.extend_from_slice(m);
                    (key, rat_from_int(c))
                })
                .collect(),
        );
    }
    let rq: SparsePolynomialRing<RationalField> = SparsePolynomialRing::new(QQ, names.to_vec());
    let inner: SparsePolynomialRing<RationalField> = SparsePolynomialRing::new(QQ, names[1..].to_vec());
    let r: PolyElement<RationalField> = dup_prs_resultant(
        &rq.to_univariate(&rq.ring.from_dict(f.terms.clone()), &inner.ring, &inner),
        &rq.to_univariate(&rq.ring.from_dict(g.terms.clone()), &inner.ring, &inner),
        &inner,
    );
    f.ring.from_dict(
        r.terms
            .iter()
            .map(|(m, c)| {
                let mut key: Monomial = vec![0];
                key.extend_from_slice(m);
                (key, c.clone())
            })
            .collect(),
    )
}
