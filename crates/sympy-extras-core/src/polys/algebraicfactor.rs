//! Port of `sympy_extras/polys/algebraicfactor.py`: factorisation and gcds
//! of multivariate polynomials over a number field `K = QQ(alpha)`
//! ([`AlgebraicField`]) by Trager's algorithm.
//!
//! The polynomial is made monic and square-free; a shift `x -> x - s alpha`
//! makes the norm `Res_y(m(y), f(X, y))` square-free over `QQ`; the norm is
//! factored over `QQ` (Wang's algorithm), each of its factors gives an
//! irreducible factor by a gcd over `K`, shifted back, and trial division
//! gives the multiplicities. [`AlgebraicPolynomialRing`] is the ring
//! `K[x_1, ..., x_n]` as a domain: no order, the gcd monic, the gcd in
//! several variables through rings of the same kind.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/algebraicfactor.py):
// sympy/polys/factortools.py (dmp_ext_factor, dmp_trial_division,
// dmp_factor_list), sympy/polys/sqfreetools.py (dmp_sqf_norm,
// _dmp_sqf_norm_shifts, dmp_sqf_p, dmp_sqf_part), sympy/polys/densetools.py
// (dmp_shift) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: sparse polynomials instead of dense nested lists, strict typing.
// ---------------------------------------------------------------------

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::densebasic::dup_from_dict;
use super::domains::{AlgebraicElement, AlgebraicField, Domain, RationalField, SparsePolynomialRing, QQ};
use super::euclidtools::dup_gcd;
use super::polytools::{factor_all, resultant_first};
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

/// A polynomial over a number field.
pub type ExtPoly = PolyElement<AlgebraicField>;

/// Factors with their multiplicities.
pub type ExtFactors = Vec<(ExtPoly, usize)>;

/// The ring `K[x_1, ..., x_n]` over a number field `K` as a domain: the
/// field has no order, so the canonical unit of a polynomial is the inverse
/// of its leading coefficient (the gcd is monic), and the gcd in several
/// variables recurses through rings of the same kind.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgebraicPolynomialRing {
    pub field: AlgebraicField,
    pub names: Vec<String>,
    pub ring: Arc<PolyRing<AlgebraicField>>,
}

impl AlgebraicPolynomialRing {
    pub fn new(field: AlgebraicField, names: Vec<String>) -> PolyResult<AlgebraicPolynomialRing> {
        let ring: Arc<PolyRing<AlgebraicField>> = PolyRing::new(field.clone(), names.clone(), MonomialOrder::Lex)?;
        Ok(AlgebraicPolynomialRing { field, names, ring })
    }

    fn inner(&self) -> AlgebraicPolynomialRing {
        AlgebraicPolynomialRing::new(self.field.clone(), self.names[1..].to_vec()).expect("at least one variable")
    }

    /// `a` as a dense polynomial in its first variable over the ring of the
    /// other ones.
    fn to_univariate(&self, a: &ExtPoly, inner: &AlgebraicPolynomialRing) -> Vec<ExtPoly> {
        let mut by_degree: BTreeMap<usize, Terms<AlgebraicElement>> = BTreeMap::new();
        for (m, c) in &a.terms {
            by_degree
                .entry(m[0] as usize)
                .or_default()
                .insert(m[1..].to_vec(), c.clone());
        }
        let coefficients: BTreeMap<usize, ExtPoly> = by_degree
            .into_iter()
            .map(|(d, terms)| (d, inner.ring.from_dict(terms)))
            .collect();
        dup_from_dict(&coefficients, inner)
    }

    fn from_univariate(&self, f: &[ExtPoly]) -> ExtPoly {
        let n: usize = f.len();
        let mut terms: Terms<AlgebraicElement> = Terms::new();
        for (i, c) in f.iter().enumerate() {
            for (m, value) in &c.terms {
                let mut key: Monomial = vec![(n - 1 - i) as u32];
                key.extend_from_slice(m);
                terms.insert(key, value.clone());
            }
        }
        self.ring.from_dict(terms)
    }

    fn gcd_monomial(&self, term: &ExtPoly, g: &ExtPoly) -> ExtPoly {
        let (m0, c0) = term.terms.iter().next().expect("a term");
        let mut monomial: Monomial = m0.clone();
        let mut coefficient: AlgebraicElement = c0.clone();
        for (m, c) in &g.terms {
            monomial = monomial.iter().zip(m).map(|(i, j)| *i.min(j)).collect();
            coefficient = self.field.gcd(&coefficient, c);
        }
        self.ring.term(monomial, coefficient)
    }

    fn gcd_general(&self, a: &ExtPoly, b: &ExtPoly) -> ExtPoly {
        if self.names.len() == 1 {
            let dense = |p: &ExtPoly| -> Vec<AlgebraicElement> {
                dup_from_dict(
                    &p.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect(),
                    &self.field,
                )
            };
            let g: Vec<AlgebraicElement> = dup_gcd(&dense(a), &dense(b), &self.field);
            let n: usize = g.len();
            return self.ring.from_dict(
                g.into_iter()
                    .enumerate()
                    .map(|(i, c)| (vec![(n - 1 - i) as u32], c))
                    .collect(),
            );
        }
        let inner: AlgebraicPolynomialRing = self.inner();
        let g: Vec<ExtPoly> = dup_gcd(&self.to_univariate(a, &inner), &self.to_univariate(b, &inner), &inner);
        self.from_univariate(&g)
    }
}

impl Domain for AlgebraicPolynomialRing {
    type Elem = ExtPoly;

    fn name(&self) -> String {
        format!("{}[{}]", self.field.name(), self.names.join(","))
    }

    fn is_field(&self) -> bool {
        false
    }

    fn is_ordered(&self) -> bool {
        false
    }

    fn characteristic(&self) -> Int {
        Int::ZERO
    }

    fn zero(&self) -> ExtPoly {
        self.ring.zero()
    }

    fn one(&self) -> ExtPoly {
        self.ring.one()
    }

    fn is_zero(&self, a: &ExtPoly) -> bool {
        a.is_zero()
    }

    fn is_negative(&self, _a: &ExtPoly) -> bool {
        false
    }

    fn add(&self, a: &ExtPoly, b: &ExtPoly) -> ExtPoly {
        a.add(b)
    }

    fn sub(&self, a: &ExtPoly, b: &ExtPoly) -> ExtPoly {
        a.sub(b)
    }

    fn neg(&self, a: &ExtPoly) -> ExtPoly {
        a.neg()
    }

    fn mul(&self, a: &ExtPoly, b: &ExtPoly) -> ExtPoly {
        a.mul(b)
    }

    fn exquo(&self, a: &ExtPoly, b: &ExtPoly) -> PolyResult<ExtPoly> {
        a.exquo(b)
    }

    fn quo(&self, a: &ExtPoly, b: &ExtPoly) -> PolyResult<ExtPoly> {
        Ok(a.div(std::slice::from_ref(b))?
            .0
            .into_iter()
            .next()
            .expect("one quotient"))
    }

    fn rem(&self, a: &ExtPoly, b: &ExtPoly) -> PolyResult<ExtPoly> {
        a.rem(std::slice::from_ref(b))
    }

    fn invert(&self, a: &ExtPoly) -> PolyResult<ExtPoly> {
        if a.is_ground() && !a.is_zero() {
            return Ok(self.ring.ground(self.field.invert(&a.LC())?));
        }
        Err(value_error(format!("{} is not a unit of {}", a, self.name())))
    }

    /// SymPy's `PolyElement.gcd`, the canonical gcd monic.
    fn gcd(&self, a: &ExtPoly, b: &ExtPoly) -> ExtPoly {
        if a.is_zero() && b.is_zero() {
            return self.zero();
        }
        if a.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return a.clone();
        }
        if a.terms.len() == 1 {
            return self.gcd_monomial(a, b);
        }
        if b.terms.len() == 1 {
            return self.gcd_monomial(b, a);
        }
        self.gcd_general(a, b)
    }

    fn gcdex(&self, _a: &ExtPoly, _b: &ExtPoly) -> PolyResult<(ExtPoly, ExtPoly, ExtPoly)> {
        Err(value_error(format!("no extended gcd over {}", self.name())))
    }

    fn canonical_unit(&self, a: &ExtPoly) -> ExtPoly {
        if a.is_zero() {
            return self.one();
        }
        self.ring
            .ground(self.field.invert(&a.LC()).expect("a nonzero leading coefficient"))
    }

    fn from_int(&self, n: i64) -> ExtPoly {
        self.ring.ground(self.field.from_int(n))
    }

    fn from_integer(&self, n: &Int) -> ExtPoly {
        self.ring.ground(self.field.from_integer(n))
    }

    fn to_str(&self, a: &ExtPoly) -> String {
        a.to_string()
    }
}

/// The monic gcd of two polynomials over a number field (zero when both are
/// zero).
pub fn ext_gcd(f: &ExtPoly, g: &ExtPoly) -> PolyResult<ExtPoly> {
    if f.is_zero() {
        return g.monic();
    }
    if g.is_zero() {
        return f.monic();
    }
    let s: AlgebraicPolynomialRing = AlgebraicPolynomialRing::new(f.ring.domain.clone(), f.ring.names.clone())?;
    let common: ExtPoly = s.gcd(&s.ring.from_dict(f.terms.clone()), &s.ring.from_dict(g.terms.clone()));
    f.ring.from_dict(common.terms).monic()
}

/// The square-free part of `f`, monic (`dmp_sqf_part`).
fn sqf_part(f: &ExtPoly) -> PolyResult<ExtPoly> {
    let mut g: ExtPoly = f.clone();
    for i in 0..f.ring.ngens {
        let d: ExtPoly = f.diff(i);
        if d.is_zero() {
            continue;
        }
        g = ext_gcd(&g, &d)?;
        if g.is_ground() {
            return f.monic();
        }
    }
    f.exquo(&g)?.monic()
}

/// The binomial coefficient `comb(n, k)`.
fn comb(n: u32, k: u32) -> Int {
    let mut result: Int = Int::ONE;
    for i in 0..k {
        result = result * Int::from(n - i) / Int::from(i + 1);
    }
    result
}

/// `f(x_1 + a_1, ..., x_n + a_n)` (`dmp_shift`).
fn shift(f: &ExtPoly, a: &[AlgebraicElement]) -> ExtPoly {
    let k: &AlgebraicField = &f.ring.domain;
    let mut terms: Terms<AlgebraicElement> = f.terms.clone();
    for (i, ai) in a.iter().enumerate() {
        if ai.is_empty() {
            continue;
        }
        let mut shifted: Terms<AlgebraicElement> = Terms::new();
        for (m, c) in &terms {
            let e: u32 = m[i];
            for j in 0..=e {
                let coefficient: AlgebraicElement =
                    k.mul(c, &k.mul(&k.from_integer(&comb(e, j)), &k.pow(ai, u64::from(e - j))));
                let mut monomial: Monomial = m.clone();
                monomial[i] = j;
                let total: AlgebraicElement = match shifted.get(&monomial) {
                    Some(previous) => k.add(previous, &coefficient),
                    None => coefficient,
                };
                shifted.insert(monomial, total);
            }
        }
        terms = shifted.into_iter().filter(|(_, c)| !c.is_empty()).collect();
    }
    f.ring.from_dict(terms)
}

/// `Res_y(m(y), f(X, y))` over `QQ`, the coefficients of `f` written as
/// polynomials in `y = alpha`.
fn norm(f: &ExtPoly) -> PolyResult<PolyElement<RationalField>> {
    let k: &AlgebraicField = &f.ring.domain;
    let mut names: Vec<String> = vec!["_y".to_string()];
    names.extend(f.ring.names.iter().cloned());
    let ring: Arc<PolyRing<RationalField>> = PolyRing::new(QQ, names.clone(), MonomialOrder::Lex)?;
    let zeros: Vec<u32> = vec![0; f.ring.ngens];
    let n: usize = k.minpoly.len() - 1;
    let mut minimal: Terms<Rat> = Terms::new();
    for (j, c) in k.minpoly.iter().enumerate() {
        if !c.is_zero() {
            let mut m: Monomial = vec![(n - j) as u32];
            m.extend_from_slice(&zeros);
            minimal.insert(m, c.clone());
        }
    }
    let mut terms: Terms<Rat> = Terms::new();
    for (m, c) in &f.terms {
        let d: usize = c.len() - 1;
        for (j, value) in c.iter().enumerate() {
            if !value.is_zero() {
                let mut key: Monomial = vec![(d - j) as u32];
                key.extend_from_slice(m);
                terms.insert(key, value.clone());
            }
        }
    }
    let result: PolyElement<RationalField> =
        resultant_first(&ring.from_dict(minimal), &ring.from_dict(terms), &names, false);
    let target: Arc<PolyRing<RationalField>> = PolyRing::new(QQ, f.ring.names.clone(), MonomialOrder::Lex)?;
    Ok(target.from_dict(result.terms.iter().map(|(m, c)| (m[1..].to_vec(), c.clone())).collect()))
}

/// Whether `r` is square-free over `QQ` (`dmp_sqf_p`).
fn sqf_p(r: &PolyElement<RationalField>) -> bool {
    if r.is_zero() {
        return true;
    }
    let q: SparsePolynomialRing<RationalField> = SparsePolynomialRing::sympy(QQ, r.ring.names.clone());
    for i in 0..r.ring.ngens {
        let d: PolyElement<RationalField> = r.diff(i);
        if !d.is_zero() && q.gcd(r, &d).degree_in(i) != 0 {
            return false;
        }
    }
    true
}

/// `(s, g, r)`: `g = f(X - s*alpha)` with the square-free norm `r`, the
/// shifts in SymPy's order (`dmp_sqf_norm`, `_dmp_sqf_norm_shifts`).
fn sqf_norm(f: &ExtPoly) -> PolyResult<(Vec<i64>, ExtPoly, PolyElement<RationalField>)> {
    let k: &AlgebraicField = &f.ring.domain;
    let n: usize = f.ring.ngens;
    let try_shift = |s: &[i64]| -> PolyResult<Option<(ExtPoly, PolyElement<RationalField>)>> {
        let g: ExtPoly = if s.iter().any(|si| *si != 0) {
            let a: Vec<AlgebraicElement> = s.iter().map(|si| k.mul(&k.from_int(-si), &k.alpha())).collect();
            shift(f, &a)
        } else {
            f.clone()
        };
        let r: PolyElement<RationalField> = norm(&g)?;
        Ok(if sqf_p(&r) { Some((g, r)) } else { None })
    };
    let s0: Vec<i64> = vec![0; n];
    if let Some((g, r)) = try_shift(&s0)? {
        return Ok((s0, g, r));
    }
    let mut by_degree: Vec<(i64, usize)> = (0..n).map(|i| (f.degree_in(i), i)).filter(|(d, _)| *d > 0).collect();
    by_degree.sort();
    for (_, i) in by_degree {
        let mut s: Vec<i64> = vec![0; n];
        s[i] = 1;
        if let Some((g, r)) = try_shift(&s)? {
            return Ok((s, g, r));
        }
    }
    let mut j: i64 = 0;
    loop {
        j += 1;
        let s: Vec<i64> = vec![j; n];
        if let Some((g, r)) = try_shift(&s)? {
            return Ok((s, g, r));
        }
    }
}

/// The multiplicities of the factors in `f` (`dmp_trial_division`).
fn trial_division(f: &ExtPoly, factors: &[ExtPoly]) -> PolyResult<ExtFactors> {
    let mut result: ExtFactors = Vec::new();
    let mut f: ExtPoly = f.clone();
    for factor in factors {
        let mut k: usize = 0;
        loop {
            let (q, r) = f.div(std::slice::from_ref(factor))?;
            if !r.is_zero() {
                break;
            }
            f = q.into_iter().next().expect("one quotient");
            k += 1;
        }
        if k == 0 {
            return Err(value_error("trial division failed".to_string()));
        }
        result.push((factor.clone(), k));
    }
    Ok(result)
}

/// The sort key: the degree in the first variable, the multiplicity, the
/// terms in decreasing order.
type SortKey = (i64, usize, Vec<(Monomial, AlgebraicElement)>);

fn sort_key(item: &(ExtPoly, usize)) -> SortKey {
    let (f, k) = item;
    let mut terms: Vec<(Monomial, AlgebraicElement)> = f.terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect();
    terms.sort();
    terms.reverse();
    (f.degree_in(0), *k, terms)
}

/// The leading coefficient and the monic irreducible factors (without
/// multiplicities) of a polynomial whose monomial content is one
/// (`dmp_ext_factor`).
fn ext_factor(f: &ExtPoly) -> PolyResult<(AlgebraicElement, Vec<ExtPoly>)> {
    let k: &AlgebraicField = &f.ring.domain;
    let lc: AlgebraicElement = f.LC();
    let f: ExtPoly = f.monic()?;
    if f.is_ground() {
        return Ok((lc, Vec::new()));
    }
    let f: ExtPoly = sqf_part(&f)?;
    let (s, mut g, r) = sqf_norm(&f)?;
    let (_, rational_factors) = factor_all(&r)?;
    if rational_factors.len() == 1 {
        return Ok((lc, vec![f]));
    }
    let mut factors: Vec<ExtPoly> = Vec::new();
    let back: Vec<AlgebraicElement> = s.iter().map(|si| k.mul(&k.from_int(*si), &k.alpha())).collect();
    for (h, _) in &rational_factors {
        let hk: ExtPoly = f
            .ring
            .from_dict(h.terms.iter().map(|(m, c)| (m.clone(), k.from_integer(c))).collect());
        let common: ExtPoly = ext_gcd(&hk, &g)?;
        g = g.exquo(&common)?;
        factors.push(if s.iter().any(|si| *si != 0) {
            shift(&common, &back)
        } else {
            common
        });
    }
    Ok((lc, factors))
}

/// The coefficient and the monic irreducible factors of `f` over its number
/// field, with their multiplicities: the variables dividing `f` split off
/// first, then Trager's algorithm; sorted by the degree in the first
/// variable, the multiplicity and the terms (`dmp_factor_list`).
pub fn ext_factor_list(f: &ExtPoly) -> PolyResult<(AlgebraicElement, ExtFactors)> {
    let ring: &Arc<PolyRing<AlgebraicField>> = &f.ring;
    let k: &AlgebraicField = &ring.domain;
    let n: usize = ring.ngens;
    if f.is_zero() {
        return Ok((k.zero(), Vec::new()));
    }
    let j: Vec<u32> = (0..n)
        .map(|i| f.terms.keys().map(|m| m[i]).min().expect("a term"))
        .collect();
    let g: ExtPoly = ring.from_dict(
        f.terms
            .iter()
            .map(|(m, c)| (m.iter().zip(&j).map(|(e, ji)| e - ji).collect(), c.clone()))
            .collect(),
    );
    let (lc, irreducible) = ext_factor(&g)?;
    let mut factors: ExtFactors = if irreducible.is_empty() {
        Vec::new()
    } else {
        trial_division(&g.monic()?, &irreducible)?
    };
    for (i, ji) in j.iter().enumerate() {
        if *ji != 0 {
            let monomial: Monomial = (0..n).map(|t| u32::from(t == i)).collect();
            factors.push((ring.term(monomial, k.one()), *ji as usize));
        }
    }
    factors.sort_by_cached_key(sort_key);
    Ok((lc, factors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::symbol::Symbol;

    #[test]
    fn sqrt2_examples() {
        let k = AlgebraicField::new(vec![rat(1, 1), rat(0, 1), rat(-2, 1)], Symbol::new("sqrt2", &[]).0).unwrap();
        let r = AlgebraicPolynomialRing::new(k.clone(), vec!["x".into(), "y".into()]).unwrap();
        let g = r.ring.gens();
        let (x, y) = (&g[0], &g[1]);
        let s = r.ring.ground(k.alpha());
        let lhs = x.sub(&s.mul(y)).mul(&x.add(y));
        let rhs = x.sub(&s.mul(y)).mul(&x.sub(y));
        assert_eq!(r.gcd(&lhs, &rhs), x.sub(&s.mul(y)));
        // x**2 - 2 y**2 = (x - sqrt2 y)(x + sqrt2 y)
        let f = x.mul(x).sub(&y.mul(y).mul_ground(&k.from_int(2)));
        let (_, factors) = ext_factor_list(&f).unwrap();
        assert_eq!(factors.len(), 2);
        assert!(factors.iter().all(|(h, m)| h.degree() == 1 && *m == 1));
    }
}
