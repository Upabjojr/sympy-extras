//! Port of `sympy_extras/polys/multisqf.py`: square-free decomposition of
//! multivariate polynomials over `ZZ`, `QQ`, `ZZ[a, ...]` and `QQ[a, ...]`
//! (Yun's algorithm in the first variable after its content, the content
//! decomposed recursively).
//!
//! The Python class hierarchy `SqfDomain` is the trait [`SqfDomain`], with
//! [`IntegerSqfDomain`], [`RationalSqfDomain`] and [`ParameterSqfDomain`]:
//! the gcd of SymPy's `dmp_inner_gcd` over the domain and the split of the
//! ground content; the coefficient is a polynomial constant in the
//! generators.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/multisqf.py):
// sympy/polys/sqfreetools.py (dup_sqf_list, dmp_sqf_list) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: sparse polynomials; the domain operations behind SqfDomain.
// ---------------------------------------------------------------------

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::numbers::*;

use super::domains::{
    Domain, IntegerRing, IteratedPolynomialRing, RationalField, SparseGround, SparsePolynomialRing, QQ, ZZ,
};
use super::rings::{Monomial, PolyElement, PolyRing, Terms};

/// The polynomials in the generators (the parameters of a polynomial domain
/// after them) with SymPy's operations of the domain.
pub trait SqfDomain: Sized {
    type Ground: Domain;

    fn ring(&self) -> &Arc<PolyRing<Self::Ground>>;

    fn n_gens(&self) -> usize;

    fn gcd(&self, f: &PolyElement<Self::Ground>, g: &PolyElement<Self::Ground>) -> PolyElement<Self::Ground>;

    /// `(coeff, f / coeff)`: the ground content with the sign of the leading
    /// coefficient over a ring, the leading coefficient over a field.
    fn ground_split(&self, f: &PolyElement<Self::Ground>) -> (PolyElement<Self::Ground>, PolyElement<Self::Ground>);

    /// The domain without the first generator.
    fn sub(&self) -> Self;

    fn degree(&self, f: &PolyElement<Self::Ground>) -> i64 {
        f.terms.keys().map(|m| m[0] as i64).max().unwrap_or(-1)
    }

    fn drop_first(&self, f: &PolyElement<Self::Ground>, other: &Self) -> PolyElement<Self::Ground> {
        other
            .ring()
            .from_dict(f.terms.iter().map(|(m, c)| (m[1..].to_vec(), c.clone())).collect())
    }

    fn lift(&self, f: &PolyElement<Self::Ground>) -> PolyElement<Self::Ground> {
        self.ring().from_dict(
            f.terms
                .iter()
                .map(|(m, c)| {
                    let mut key: Monomial = vec![0];
                    key.extend_from_slice(m);
                    (key, c.clone())
                })
                .collect(),
        )
    }

    /// SymPy's `dmp_content`: the gcd of the coefficients in `x0` (fold from
    /// the leading one, stopping at `1`), the sign of its ground leading
    /// coefficient made positive; constant in `x0`.
    fn content_first(&self, f: &PolyElement<Self::Ground>) -> PolyElement<Self::Ground> {
        let mut coefficients: BTreeMap<u32, Terms<<Self::Ground as Domain>::Elem>> = BTreeMap::new();
        for (m, c) in &f.terms {
            let mut key: Monomial = vec![0];
            key.extend_from_slice(&m[1..]);
            coefficients.entry(m[0]).or_default().insert(key, c.clone());
        }
        let ordered: Vec<PolyElement<Self::Ground>> = coefficients
            .into_values()
            .rev()
            .map(|terms| self.ring().from_dict(terms))
            .collect();
        let one: PolyElement<Self::Ground> = self.ring().one();
        let mut content: PolyElement<Self::Ground> = ordered[0].clone();
        for coefficient in &ordered[1..] {
            content = self.gcd(&content, coefficient);
            if content == one {
                break;
            }
        }
        if self.negative(&content) {
            content = content.neg();
        }
        content
    }

    fn negative(&self, f: &PolyElement<Self::Ground>) -> bool {
        !f.is_zero() && self.ring().domain.is_negative(&f.LC())
    }
}

/// `ZZ[x, ...]` with SymPy's gcd (`IntegerPolynomialRing`).
#[derive(Debug, Clone)]
pub struct IntegerSqfDomain {
    pub domain: SparsePolynomialRing<IntegerRing>,
    pub names: Vec<String>,
}

impl IntegerSqfDomain {
    pub fn new(names: Vec<String>) -> IntegerSqfDomain {
        IntegerSqfDomain {
            domain: SparsePolynomialRing::sympy(ZZ, names.clone()),
            names,
        }
    }
}

impl SqfDomain for IntegerSqfDomain {
    type Ground = IntegerRing;

    fn ring(&self) -> &Arc<PolyRing<IntegerRing>> {
        &self.domain.ring
    }

    fn n_gens(&self) -> usize {
        self.names.len()
    }

    fn gcd(&self, f: &PolyElement<IntegerRing>, g: &PolyElement<IntegerRing>) -> PolyElement<IntegerRing> {
        self.domain.gcd(f, g)
    }

    fn ground_split(&self, f: &PolyElement<IntegerRing>) -> (PolyElement<IntegerRing>, PolyElement<IntegerRing>) {
        let mut c: Int = Int::ZERO;
        for v in f.terms.values() {
            c = gcd(&c, v);
        }
        if is_negative(&f.LC()) {
            c = -c;
        }
        let part: PolyElement<IntegerRing> = self
            .ring()
            .from_dict(f.terms.iter().map(|(m, v)| (m.clone(), div_floor(v, &c))).collect());
        (self.ring().ground(c), part)
    }

    fn sub(&self) -> IntegerSqfDomain {
        IntegerSqfDomain::new(self.names[1..].to_vec())
    }
}

/// `QQ[x, ...]` with SymPy's gcd made monic (`RationalPolynomialRing`).
#[derive(Debug, Clone)]
pub struct RationalSqfDomain {
    pub domain: SparsePolynomialRing<RationalField>,
    pub names: Vec<String>,
}

impl RationalSqfDomain {
    pub fn new(names: Vec<String>) -> RationalSqfDomain {
        RationalSqfDomain {
            domain: SparsePolynomialRing::sympy(QQ, names.clone()),
            names,
        }
    }
}

impl SqfDomain for RationalSqfDomain {
    type Ground = RationalField;

    fn ring(&self) -> &Arc<PolyRing<RationalField>> {
        &self.domain.ring
    }

    fn n_gens(&self) -> usize {
        self.names.len()
    }

    fn gcd(&self, f: &PolyElement<RationalField>, g: &PolyElement<RationalField>) -> PolyElement<RationalField> {
        let h: PolyElement<RationalField> = self.domain.gcd(f, g);
        if h.is_zero() {
            h
        } else {
            h.exquo_ground(&h.LC()).expect("a nonzero rational")
        }
    }

    fn ground_split(&self, f: &PolyElement<RationalField>) -> (PolyElement<RationalField>, PolyElement<RationalField>) {
        let lc: Rat = f.LC();
        let part: PolyElement<RationalField> = f.exquo_ground(&lc).expect("a nonzero rational");
        (self.ring().ground(lc), part)
    }

    fn sub(&self) -> RationalSqfDomain {
        RationalSqfDomain::new(self.names[1..].to_vec())
    }
}

/// The generators over `K = ZZ[a, ...]` or `QQ[a, ...]`.
#[derive(Debug, Clone)]
pub struct ParameterSqfDomain<K: SparseGround> {
    pub iterated: IteratedPolynomialRing<K>,
    pub params: SparsePolynomialRing<K>,
    pub gen_names: Vec<String>,
}

impl<K: SparseGround> ParameterSqfDomain<K> {
    pub fn new(params: SparsePolynomialRing<K>, gen_names: Vec<String>) -> ParameterSqfDomain<K> {
        let iterated: IteratedPolynomialRing<K> = IteratedPolynomialRing::new(params.clone(), gen_names.clone());
        ParameterSqfDomain {
            iterated,
            params,
            gen_names,
        }
    }
}

impl<K: SparseGround> SqfDomain for ParameterSqfDomain<K> {
    type Ground = K;

    fn ring(&self) -> &Arc<PolyRing<K>> {
        self.iterated.ring()
    }

    fn n_gens(&self) -> usize {
        self.gen_names.len()
    }

    fn gcd(&self, f: &PolyElement<K>, g: &PolyElement<K>) -> PolyElement<K> {
        self.iterated.gcd(f, g)
    }

    fn ground_split(&self, f: &PolyElement<K>) -> (PolyElement<K>, PolyElement<K>) {
        let mut content: PolyElement<K> = self.iterated.ground_content(f);
        if self.params.is_negative(&self.iterated.ground_LC(f)) {
            content = content.neg();
        }
        let embedded: PolyElement<K> = self.iterated.embed_params(&content);
        let part: PolyElement<K> = f.exquo(&embedded).expect("the content divides");
        (embedded, part)
    }

    fn negative(&self, f: &PolyElement<K>) -> bool {
        self.params.is_negative(&self.iterated.ground_LC(f))
    }

    fn sub(&self) -> ParameterSqfDomain<K> {
        ParameterSqfDomain::new(self.params.clone(), self.gen_names[1..].to_vec())
    }
}

fn exact<K: Domain>(f: &PolyElement<K>, g: &PolyElement<K>) -> PolyElement<K> {
    f.exquo(g).expect("an exact division of the square-free decomposition")
}

/// `(coeff, [(g, k)])` with `f = coeff * prod g**k`, the `g` square-free and
/// pairwise coprime, in increasing multiplicity (`all` keeps the trivial
/// factors).
pub fn sqf_list<D: SqfDomain>(
    f: &PolyElement<D::Ground>,
    d: &D,
    all: bool,
) -> (PolyElement<D::Ground>, Vec<(PolyElement<D::Ground>, usize)>) {
    if f.is_zero() {
        return (d.ring().zero(), Vec::new());
    }
    let (mut coeff, mut f) = d.ground_split(f);
    if d.degree(&f) <= 0 && d.n_gens() == 1 {
        return (coeff, Vec::new());
    }
    let deg: i64 = d.degree(&f);
    if deg < 0 {
        return (coeff, Vec::new());
    }
    let mut result: BTreeMap<usize, PolyElement<D::Ground>> = BTreeMap::new();
    let one: PolyElement<D::Ground> = d.ring().one();
    let mut content: PolyElement<D::Ground> = one.clone();
    if d.n_gens() > 1 {
        content = d.content_first(&f);
        if content != one {
            f = exact(&f, &content);
        }
    }
    if deg != 0 {
        let h: PolyElement<D::Ground> = f.diff(0);
        let g: PolyElement<D::Ground> = d.gcd(&f, &h);
        let (mut p, mut q) = (exact(&f, &g), exact(&h, &g));
        let mut i: usize = 1;
        loop {
            let h: PolyElement<D::Ground> = q.sub(&p.diff(0));
            if h.is_zero() {
                result.insert(i, p);
                break;
            }
            let g: PolyElement<D::Ground> = d.gcd(&p, &h);
            p = exact(&p, &g);
            q = exact(&h, &g);
            if all || d.degree(&g) > 0 {
                result.insert(i, g);
            }
            i += 1;
        }
    }
    if d.n_gens() == 1 {
        return (coeff, result.into_iter().map(|(i, g)| (g, i)).collect());
    }
    let s: D = d.sub();
    let (coeff_content, result_content) = sqf_list(&d.drop_first(&content, &s), &s, all);
    coeff = coeff.mul(&d.lift(&coeff_content));
    for (factor, i) in result_content {
        let lifted: PolyElement<D::Ground> = d.lift(&factor);
        let combined: PolyElement<D::Ground> = match result.get(&i) {
            Some(existing) => existing.mul(&lifted),
            None => lifted,
        };
        result.insert(i, combined);
    }
    (coeff, result.into_iter().map(|(i, g)| (g, i)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctest() {
        // 2 (x + y)**2 (x - y)**3 (y + 1)**2
        let d: IntegerSqfDomain = IntegerSqfDomain::new(vec!["x".to_string(), "y".to_string()]);
        let r = d.ring().clone();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let f = x
            .add(y)
            .pow(2)
            .mul(&x.sub(y).pow(3))
            .mul(&y.add(&r.one()).pow(2))
            .mul_ground(&int(2));
        let (c, factors) = sqf_list(&f, &d, false);
        assert_eq!(c.to_string(), "2");
        let shown: Vec<(String, usize)> = factors.iter().map(|(g, k)| (g.to_string(), *k)).collect();
        assert_eq!(
            shown,
            vec![("x*y + x + y**2 + y".to_string(), 2), ("x - y".to_string(), 3)]
        );
    }
}
