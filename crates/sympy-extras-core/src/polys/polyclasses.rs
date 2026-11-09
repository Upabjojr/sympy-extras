//! Port of `sympy_extras/polys/polyclasses.py`: [`DensePoly`], a typed
//! univariate polynomial over the dense representation, with the
//! algorithms of the dense modules as methods.
//!
//! A polynomial in several variables is a `DensePoly` over a
//! [`PolynomialRing`](super::domains::PolynomialRing) domain. The
//! conversions `from_expr`/`to_expr` wait for the evaluating functions of
//! the core (`add`, `mul`, `pow`). The binary operations return
//! `Err(Value)` for polynomials over different domains or in different
//! variables, where Python raises `ValueError`.

use std::fmt;

use crate::core::symbol::Symbol;
use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::densearith as da;
use super::densebasic as db;
use super::densetools as dt;
use super::domains::{ConvertFrom, Domain};
use super::euclidtools as et;
use super::sqfreetools as sq;

/// A univariate polynomial `rep` (coefficients highest degree first, no
/// leading zeros) over `domain` in the variable `gen`.
#[derive(Debug, Clone)]
pub struct DensePoly<K: Domain> {
    pub rep: Vec<K::Elem>,
    pub domain: K,
    pub gen: Symbol,
}

/// A factorisation: the leading constant and the factors with their
/// multiplicities.
pub type DenseFactors<K> = (<K as Domain>::Elem, Vec<(DensePoly<K>, usize)>);

impl<K: Domain> PartialEq for DensePoly<K> {
    fn eq(&self, other: &Self) -> bool {
        self.rep == other.rep && self.domain == other.domain && self.gen == other.gen
    }
}

impl<K: Domain> fmt::Display for DensePoly<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let coefficients: Vec<String> = self.rep.iter().map(|c| self.domain.to_str(c)).collect();
        write!(
            f,
            "DensePoly([{}], {}, {})",
            coefficients.join(", "),
            self.domain.name(),
            self.gen.0
        )
    }
}

impl<K: Domain> DensePoly<K> {
    pub fn new(rep: &[K::Elem], domain: K, gen: Symbol) -> DensePoly<K> {
        DensePoly {
            rep: db::dup_strip(rep, &domain),
            domain,
            gen,
        }
    }

    /// The polynomial with the given integer coefficients (highest degree
    /// first) over `domain`.
    pub fn from_list(coefficients: &[Int], gen: Symbol, domain: K) -> DensePoly<K> {
        let rep: Vec<K::Elem> = db::dup_from_list(coefficients, &domain);
        DensePoly { rep, domain, gen }
    }

    /// The same polynomial over another domain.
    pub fn convert<K1: ConvertFrom<K>>(&self, domain: K1) -> PolyResult<DensePoly<K1>> {
        let rep: Vec<K1::Elem> = db::dup_convert(&self.rep, &self.domain, &domain)?;
        Ok(DensePoly::new(&rep, domain, self.gen.clone()))
    }

    /// A polynomial with the domain and variable of `self`.
    pub fn with_rep(&self, rep: Vec<K::Elem>) -> DensePoly<K> {
        DensePoly {
            rep: db::dup_strip_vec(rep, &self.domain),
            domain: self.domain.clone(),
            gen: self.gen.clone(),
        }
    }

    fn same(&self, other: &DensePoly<K>) -> PolyResult<()> {
        if other.domain != self.domain || other.gen != self.gen {
            return Err(value_error(
                "polynomials over different domains or in different variables".to_string(),
            ));
        }
        Ok(())
    }

    // -- basic data

    pub fn degree(&self) -> isize {
        db::dup_degree(&self.rep)
    }

    #[allow(non_snake_case)]
    pub fn LC(&self) -> K::Elem {
        db::dup_LC(&self.rep, &self.domain)
    }

    #[allow(non_snake_case)]
    pub fn TC(&self) -> K::Elem {
        db::dup_TC(&self.rep, &self.domain)
    }

    pub fn is_zero(&self) -> bool {
        self.rep.is_empty()
    }

    pub fn is_one(&self) -> bool {
        db::dup_is_one(&self.rep, &self.domain)
    }

    pub fn is_monic(&self) -> bool {
        !self.rep.is_empty() && self.domain.is_one(&self.rep[0])
    }

    pub fn is_squarefree(&self) -> bool {
        sq::dup_sqf_p(&self.rep, &self.domain)
    }

    /// The coefficient of `gen**n`.
    pub fn coeff(&self, n: usize) -> K::Elem {
        let d: isize = self.degree();
        if n as isize > d {
            return self.domain.zero();
        }
        self.rep[d as usize - n].clone()
    }

    pub fn terms(&self) -> Vec<(usize, K::Elem)> {
        db::dup_terms(&self.rep, &self.domain)
    }

    // -- arithmetic

    pub fn neg(&self) -> DensePoly<K> {
        self.with_rep(da::dup_neg(&self.rep, &self.domain))
    }

    pub fn add(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(da::dup_add(&self.rep, &other.rep, &self.domain)))
    }

    pub fn sub(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(da::dup_sub(&self.rep, &other.rep, &self.domain)))
    }

    pub fn mul(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(da::dup_mul(&self.rep, &other.rep, &self.domain)))
    }

    pub fn pow(&self, n: usize) -> DensePoly<K> {
        self.with_rep(da::dup_pow(&self.rep, n, &self.domain))
    }

    pub fn mul_ground(&self, c: &K::Elem) -> DensePoly<K> {
        self.with_rep(da::dup_mul_ground(&self.rep, c, &self.domain))
    }

    pub fn exquo_ground(&self, c: &K::Elem) -> PolyResult<DensePoly<K>> {
        Ok(self.with_rep(da::dup_exquo_ground(&self.rep, c, &self.domain)?))
    }

    /// Quotient and remainder (over a field, or with exact divisions of the
    /// leading coefficients).
    pub fn div(&self, other: &DensePoly<K>) -> PolyResult<(DensePoly<K>, DensePoly<K>)> {
        self.same(other)?;
        let (q, r) = da::dup_div(&self.rep, &other.rep, &self.domain)?;
        Ok((self.with_rep(q), self.with_rep(r)))
    }

    pub fn quo(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        Ok(self.div(other)?.0)
    }

    pub fn rem(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        Ok(self.div(other)?.1)
    }

    /// The exact quotient; `Value` when `other` does not divide `self`.
    pub fn exquo(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(da::dup_exquo(&self.rep, &other.rep, &self.domain)?))
    }

    pub fn pdiv(&self, other: &DensePoly<K>) -> PolyResult<(DensePoly<K>, DensePoly<K>)> {
        self.same(other)?;
        let (q, r) = da::dup_pdiv(&self.rep, &other.rep, &self.domain)?;
        Ok((self.with_rep(q), self.with_rep(r)))
    }

    pub fn prem(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        Ok(self.pdiv(other)?.1)
    }

    // -- calculus and transformations

    pub fn diff(&self, m: usize) -> DensePoly<K> {
        self.with_rep(dt::dup_diff(&self.rep, m, &self.domain))
    }

    pub fn integrate(&self, m: usize) -> PolyResult<DensePoly<K>> {
        Ok(self.with_rep(dt::dup_integrate(&self.rep, m, &self.domain)?))
    }

    pub fn eval(&self, a: &K::Elem) -> K::Elem {
        dt::dup_eval(&self.rep, a, &self.domain)
    }

    /// `f(x + a)`.
    pub fn shift(&self, a: &K::Elem) -> DensePoly<K> {
        self.with_rep(dt::dup_shift(&self.rep, a, &self.domain))
    }

    /// `f(a x)`.
    pub fn scale(&self, a: &K::Elem) -> DensePoly<K> {
        self.with_rep(dt::dup_scale(&self.rep, a, &self.domain))
    }

    pub fn compose(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(dt::dup_compose(&self.rep, &other.rep, &self.domain)))
    }

    /// `f(-x)`.
    pub fn mirror(&self) -> DensePoly<K> {
        self.with_rep(dt::dup_mirror(&self.rep, &self.domain))
    }

    pub fn reverse(&self) -> DensePoly<K> {
        self.with_rep(db::dup_reverse(&self.rep, &self.domain))
    }

    // -- contents and normal forms

    pub fn content(&self) -> K::Elem {
        dt::dup_content(&self.rep, &self.domain)
    }

    pub fn primitive(&self) -> (K::Elem, DensePoly<K>) {
        let (c, part) = dt::dup_primitive(&self.rep, &self.domain);
        (c, self.with_rep(part))
    }

    pub fn monic(&self) -> PolyResult<DensePoly<K>> {
        Ok(self.with_rep(dt::dup_monic(&self.rep, &self.domain)?))
    }

    // -- gcd, resultants

    pub fn gcd(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(et::dup_gcd(&self.rep, &other.rep, &self.domain)))
    }

    pub fn lcm(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(et::dup_lcm(&self.rep, &other.rep, &self.domain)))
    }

    /// `(s, t, h)` with `s f + t g = h` the monic gcd (fields).
    pub fn gcdex(&self, other: &DensePoly<K>) -> PolyResult<(DensePoly<K>, DensePoly<K>, DensePoly<K>)> {
        self.same(other)?;
        let (s, t, h) = et::dup_gcdex(&self.rep, &other.rep, &self.domain)?;
        Ok((self.with_rep(s), self.with_rep(t), self.with_rep(h)))
    }

    /// The inverse of `self` modulo `other` (fields).
    pub fn invert(&self, other: &DensePoly<K>) -> PolyResult<DensePoly<K>> {
        self.same(other)?;
        Ok(self.with_rep(et::dup_invert(&self.rep, &other.rep, &self.domain)?))
    }

    pub fn resultant(&self, other: &DensePoly<K>) -> PolyResult<K::Elem> {
        self.same(other)?;
        Ok(et::dup_resultant(&self.rep, &other.rep, &self.domain))
    }

    pub fn discriminant(&self) -> K::Elem {
        et::dup_discriminant(&self.rep, &self.domain)
    }

    pub fn subresultants(&self, other: &DensePoly<K>) -> PolyResult<Vec<DensePoly<K>>> {
        self.same(other)?;
        Ok(et::dup_subresultants(&self.rep, &other.rep, &self.domain)
            .into_iter()
            .map(|r| self.with_rep(r))
            .collect())
    }

    // -- factorisation

    pub fn sqf_part(&self) -> DensePoly<K> {
        self.with_rep(sq::dup_sqf_part(&self.rep, &self.domain))
    }

    pub fn sqf_list(&self) -> DenseFactors<K> {
        let (c, factors) = sq::dup_sqf_list(&self.rep, &self.domain);
        (c, factors.into_iter().map(|(g, k)| (self.with_rep(g), k)).collect())
    }

    /// The irreducible factorisation over the domain (`ZZ`, `QQ`, `GF(p)`).
    pub fn factor_list(&self) -> PolyResult<DenseFactors<K>> {
        let (c, factors) = self.domain.factor_dense(&self.rep)?;
        Ok((c, factors.into_iter().map(|(g, k)| (self.with_rep(g), k)).collect()))
    }

    pub fn is_irreducible(&self) -> PolyResult<bool> {
        if self.degree() < 1 {
            return Ok(false);
        }
        let (_, factors) = self.factor_list()?;
        Ok(factors.len() == 1 && factors[0].1 == 1 && factors[0].0.degree() == self.degree())
    }

    // -- real roots (ZZ and QQ)

    /// Isolating intervals of the real roots with their multiplicities
    /// (refined below the width `eps` when given).
    pub fn real_root_intervals(&self, eps: Option<&Rat>) -> Vec<((Rat, Rat), usize)> {
        self.domain.isolate_real_roots_dense(&self.rep, eps)
    }

    pub fn count_real_roots(&self, inf: Option<&Rat>, sup: Option<&Rat>) -> i64 {
        self.domain.count_real_roots_dense(&self.rep, inf, sup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::{IntegerRing, QQ, ZZ};

    fn ints(v: &[i64]) -> Vec<Int> {
        v.iter().map(|c| Int::from(*c)).collect()
    }

    #[test]
    fn doctests() {
        let x = Symbol::new("x", &[]);
        let f: DensePoly<IntegerRing> = DensePoly::from_list(&ints(&[1, 0, 0, 0, -1]), x.clone(), ZZ);
        assert_eq!(f.to_string(), "DensePoly([1, 0, 0, 0, -1], ZZ, x)");
        let (c, factors) = f.factor_list().unwrap();
        let shown: Vec<(String, usize)> = factors.iter().map(|(g, k)| (g.to_string(), *k)).collect();
        assert_eq!(c, Int::ONE);
        assert_eq!(
            shown,
            vec![
                ("DensePoly([1, -1], ZZ, x)".to_string(), 1),
                ("DensePoly([1, 1], ZZ, x)".to_string(), 1),
                ("DensePoly([1, 0, 1], ZZ, x)".to_string(), 1)
            ]
        );
        let (_, sqf) = f.mul(&f).unwrap().sqf_list();
        assert_eq!(sqf, vec![(f.clone(), 2)]);
        let g = DensePoly::from_list(&ints(&[1, 0, -1]), x.clone(), ZZ);
        assert_eq!(f.gcd(&g).unwrap(), g);
        assert_eq!(f.count_real_roots(None, None), 2);
        assert_eq!(
            f.real_root_intervals(None),
            vec![((rat(-2, 1), rat(0, 1)), 1), ((rat(0, 1), rat(2, 1)), 1)]
        );
        let p = DensePoly::from_list(&ints(&[1, 0, -2]), x.clone(), QQ);
        let q = DensePoly::from_list(&ints(&[1, -3]), x.clone(), QQ);
        assert_eq!(p.resultant(&q).unwrap(), rat(7, 1));
        // data, operations and errors
        assert_eq!((f.degree(), f.coeff(4), f.coeff(7)), (4, Int::ONE, Int::ZERO));
        assert!(f.is_monic() && f.is_squarefree() && !f.mul(&f).unwrap().is_squarefree());
        assert!(!f.is_irreducible().unwrap() && g.sub(&f).unwrap().degree() == 4);
        assert!(f
            .add(&DensePoly::from_list(&ints(&[1]), Symbol::new("y", &[]), ZZ))
            .is_err());
        assert_eq!(q.convert(ZZ).unwrap().rep, ints(&[1, -3]));
        assert_eq!(
            f.convert(QQ).unwrap().rep,
            vec![rat(1, 1), rat(0, 1), rat(0, 1), rat(0, 1), rat(-1, 1)]
        );
        assert_eq!(DensePoly::new(&ints(&[0, 0, 3]), ZZ, x).rep, ints(&[3]));
    }
}
