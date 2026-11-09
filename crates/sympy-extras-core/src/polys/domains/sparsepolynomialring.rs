//! Port of `sympy_extras/polys/domains/sparsepolynomialring.py`: the
//! polynomial ring `K[y1, ..., yk]` as a domain whose elements are sparse
//! polynomials ([`PolyElement`]), with SymPy's conventions for the gcd and
//! the lcm of its elements.
//!
//! The Python classes map as follows:
//!
//! * `SparsePolynomialRing(K, names)`: [`SparsePolynomialRing::new`] (the
//!   subresultant gcd);
//! * `IntegerPolynomialRing(names)`, `RationalPolynomialRing(names)`:
//!   [`SparsePolynomialRing::sympy`] over `ZZ` and `QQ`, whose gcd follows
//!   SymPy's `PolyElement` (the heuristic gcd over `ZZ`; the ground gcd of
//!   rationals and monic gcds over `QQ`), given by the [`SparseGround`]
//!   hooks of the ground domain;
//! * `IteratedPolynomialRing(params, gen_names)`:
//!   [`IteratedPolynomialRing`], one type for every number of generators
//!   (with no generator left it is its parameter ring, so that the
//!   recursion has one element type, as in Python).
//!
//! The conversions to and from expressions wait for the typed core.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/domains/
// sparsepolynomialring.py): sympy/polys/rings.py (PolyElement.cofactors,
// _gcd_zero, _gcd_monom, _gcd_QQ, lcm, content),
// sympy/polys/domains/polynomialring.py (canonical_unit),
// sympy/polys/domains/field.py (Field.gcd, Field.lcm),
// sympy/polys/euclidtools.py (dup_rr_prs_gcd, dmp_rr_prs_gcd,
// _dup_rr_trivial_gcd, _dmp_rr_trivial_gcd, _dmp_simplify_gcd,
// dmp_content, dup_rr_lcm, dmp_rr_lcm), sympy/polys/densetools.py
// (dmp_ground_content) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: sparse elements in all the variables instead of dense nested
// lists; the heuristic gcd over ZZ replaced by the subresultant gcd when it
// fails (the same canonical result).
// ---------------------------------------------------------------------

use std::sync::Arc;

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::super::densebasic::dup_from_dict;
use super::super::euclidtools::{dup_gcd, dup_subresultants};
use super::super::heuristicgcd::{deflate, heugcd, inflate};
use super::super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};
use super::domain::Domain;
use super::finitefield::FiniteField;
use super::integerring::{IntegerRing, ZZ};
use super::rationalfield::RationalField;

/// Which gcd the ring computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GcdFlavor {
    /// `SparsePolynomialRing`: the subresultant gcd variable after variable.
    Subresultant,
    /// `IntegerPolynomialRing`, `RationalPolynomialRing`: SymPy's gcd of
    /// `PolyElement` (the [`SparseGround`] hooks of the ground domain).
    Sympy,
}

/// `K[y1, ..., yk]` with sparse elements (lex order).
#[derive(Debug, Clone)]
pub struct SparsePolynomialRing<K: Domain> {
    pub ground: K,
    pub names: Vec<String>,
    pub ring: Arc<PolyRing<K>>,
    pub flavor: GcdFlavor,
}

impl<K: Domain> PartialEq for SparsePolynomialRing<K> {
    fn eq(&self, other: &Self) -> bool {
        self.flavor == other.flavor && self.ground == other.ground && self.names == other.names
    }
}

/// The ground domains of the sparse polynomial rings: the hooks by which
/// SymPy's conventions (the `Sympy` flavor) differ from the generic ones.
pub trait SparseGround: Domain {
    /// The gcd of two coefficients in the contents (`_ground_gcd`).
    fn sympy_ground_gcd(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.gcd(a, b)
    }

    /// The lcm of two contents (`_ground_lcm`).
    fn sympy_ground_lcm(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.lcm(a, b)
    }

    /// The gcd of two polynomials neither of which is a term
    /// (`_gcd_general`).
    fn sympy_gcd_general(
        ring: &SparsePolynomialRing<Self>,
        a: &PolyElement<Self>,
        b: &PolyElement<Self>,
    ) -> PolyElement<Self> {
        ring.subresultant_gcd(a, b)
    }

    /// `cancel`, when it is not the generic one
    /// (`RationalPolynomialRing.cancel`).
    #[allow(clippy::type_complexity)]
    fn sympy_cancel(
        _ring: &SparsePolynomialRing<Self>,
        _f: &PolyElement<Self>,
        _g: &PolyElement<Self>,
    ) -> Option<(PolyElement<Self>, PolyElement<Self>)> {
        None
    }

    /// The lcm, when it is not the generic one.
    fn sympy_lcm(
        _ring: &SparsePolynomialRing<Self>,
        _a: &PolyElement<Self>,
        _b: &PolyElement<Self>,
    ) -> Option<PolyElement<Self>> {
        None
    }
}

impl SparseGround for IntegerRing {
    /// The heuristic gcd on the deflated polynomials, the subresultant gcd
    /// when it fails (`IntegerPolynomialRing._gcd_general`).
    fn sympy_gcd_general(
        ring: &SparsePolynomialRing<IntegerRing>,
        a: &PolyElement<IntegerRing>,
        b: &PolyElement<IntegerRing>,
    ) -> PolyElement<IntegerRing> {
        let (steps, deflated) = deflate(&[a.clone(), b.clone()]);
        let h: PolyElement<IntegerRing> = match heugcd(&deflated[0], &deflated[1]) {
            Ok((h, _, _)) => h,
            Err(_) => ring.subresultant_gcd(&deflated[0], &deflated[1]),
        };
        inflate(&h, &steps)
    }
}

impl SparseGround for RationalField {
    /// The gcd of the numerators over the lcm of the denominators
    /// (SymPy's `Field.gcd`).
    fn sympy_ground_gcd(&self, a: &Rat, b: &Rat) -> Rat {
        rat_from_parts(gcd(a.numerator(), b.numerator()), lcm(&denom(a), &denom(b)))
    }

    /// The lcm of the numerators over the gcd of the denominators
    /// (SymPy's `Field.lcm`).
    fn sympy_ground_lcm(&self, a: &Rat, b: &Rat) -> Rat {
        rat_from_parts(lcm(a.numerator(), b.numerator()), gcd(&denom(a), &denom(b)))
    }

    /// The gcd of the polynomials with cleared denominators over `ZZ`, made
    /// monic (`RationalPolynomialRing._gcd_general`).
    fn sympy_gcd_general(
        ring: &SparsePolynomialRing<RationalField>,
        a: &PolyElement<RationalField>,
        b: &PolyElement<RationalField>,
    ) -> PolyElement<RationalField> {
        let z: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::sympy(ZZ, ring.names.clone());
        let cleared = |p: &PolyElement<RationalField>| -> PolyElement<IntegerRing> {
            let mut d: Int = Int::ONE;
            for c in p.terms.values() {
                d = lcm(&d, &denom(c));
            }
            z.ring.from_dict(
                p.terms
                    .iter()
                    .map(|(m, c)| (m.clone(), numer(&(c * Rat::from(d.clone())))))
                    .collect(),
            )
        };
        let h: PolyElement<IntegerRing> = IntegerRing::sympy_gcd_general(&z, &cleared(a), &cleared(b));
        let result: PolyElement<RationalField> = ring
            .ring
            .from_dict(h.terms.iter().map(|(m, c)| (m.clone(), rat_from_int(c))).collect());
        result
            .exquo_ground(&result.LC())
            .expect("a nonzero leading coefficient of a field")
    }

    /// The denominators cleared, the cofactors over `ZZ`, the contents
    /// divided by their gcd, the sign of `q` canonical.
    fn sympy_cancel(
        ring: &SparsePolynomialRing<RationalField>,
        f: &PolyElement<RationalField>,
        g: &PolyElement<RationalField>,
    ) -> Option<(PolyElement<RationalField>, PolyElement<RationalField>)> {
        if f.is_zero() {
            return Some((f.clone(), ring.one()));
        }
        let z: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::sympy(ZZ, ring.names.clone());
        let clear = |p: &PolyElement<RationalField>| -> (Int, PolyElement<IntegerRing>) {
            let mut d: Int = Int::ONE;
            for c in p.terms.values() {
                d = lcm(&d, &denom(c));
            }
            let cleared = z.ring.from_dict(
                p.terms
                    .iter()
                    .map(|(m, c)| (m.clone(), numer(&(c * rat_from_int(&d)))))
                    .collect(),
            );
            (d, cleared)
        };
        let (cq, fz) = clear(f);
        let (cp, gz) = clear(g);
        let h: PolyElement<IntegerRing> = z.gcd(&fz, &gz);
        let pz: PolyElement<IntegerRing> = fz.exquo(&h).expect("the gcd divides");
        let qz: PolyElement<IntegerRing> = gz.exquo(&h).expect("the gcd divides");
        let c: Int = gcd(&cp, &cq);
        let (cp, cq): (Int, Int) = (&cp / &c, &cq / &c);
        let p = ring.ring.from_dict(
            pz.terms
                .iter()
                .map(|(m, v)| (m.clone(), rat_from_int(&(v * &cp))))
                .collect(),
        );
        let q = ring.ring.from_dict(
            qz.terms
                .iter()
                .map(|(m, v)| (m.clone(), rat_from_int(&(v * &cq))))
                .collect(),
        );
        Some(ring.canonical_denominator(p, q))
    }

    /// The monic `a b / gcd(a, b)`.
    fn sympy_lcm(
        ring: &SparsePolynomialRing<RationalField>,
        a: &PolyElement<RationalField>,
        b: &PolyElement<RationalField>,
    ) -> Option<PolyElement<RationalField>> {
        if a.is_zero() || b.is_zero() {
            return Some(ring.zero());
        }
        let h: PolyElement<RationalField> = a.mul(b).exquo(&ring.gcd(a, b)).expect("the gcd divides the product");
        Some(
            h.exquo_ground(&h.LC())
                .expect("a nonzero leading coefficient of a field"),
        )
    }
}

impl SparseGround for FiniteField {}

impl<K: SparseGround> SparsePolynomialRing<K> {
    /// `SparsePolynomialRing(ground, names)`.
    pub fn new(ground: K, names: Vec<String>) -> SparsePolynomialRing<K> {
        Self::with_flavor(ground, names, GcdFlavor::Subresultant)
    }

    /// `IntegerPolynomialRing(names)` over `ZZ`, `RationalPolynomialRing(names)`
    /// over `QQ`: SymPy's gcd.
    pub fn sympy(ground: K, names: Vec<String>) -> SparsePolynomialRing<K> {
        Self::with_flavor(ground, names, GcdFlavor::Sympy)
    }

    fn with_flavor(ground: K, names: Vec<String>, flavor: GcdFlavor) -> SparsePolynomialRing<K> {
        let ring: Arc<PolyRing<K>> =
            PolyRing::new(ground.clone(), names.clone(), MonomialOrder::Lex).expect("at least one variable");
        SparsePolynomialRing {
            ground,
            names,
            ring,
            flavor,
        }
    }

    /// `(p, q)` with `p/q == f/g` in lowest terms, the leading coefficient
    /// of `q` canonical (nonnegative).
    pub fn cancel(&self, f: &PolyElement<K>, g: &PolyElement<K>) -> (PolyElement<K>, PolyElement<K>) {
        if self.flavor == GcdFlavor::Sympy {
            if let Some(result) = K::sympy_cancel(self, f, g) {
                return result;
            }
        }
        if f.is_zero() {
            return (f.clone(), self.ring.one());
        }
        let h: PolyElement<K> = self.gcd(f, g);
        let p: PolyElement<K> = f.exquo(&h).expect("the gcd divides");
        let q: PolyElement<K> = g.exquo(&h).expect("the gcd divides");
        self.canonical_denominator(p, q)
    }

    /// `(-p, -q)` when the leading coefficient of `q` is negative (SymPy
    /// multiplies by the canonical unit of `q`).
    pub fn canonical_denominator(&self, p: PolyElement<K>, q: PolyElement<K>) -> (PolyElement<K>, PolyElement<K>) {
        if !q.is_zero() && self.ground.is_negative(&q.LC()) {
            return (p.neg(), q.neg());
        }
        (p, q)
    }

    pub fn is_nonnegative(&self, a: &PolyElement<K>) -> bool {
        a.is_zero() || !self.ground.is_negative(&a.LC())
    }

    // -- dense views

    /// `a` as a dense polynomial in `y1` with coefficients in `inner = K[y2,
    /// ..., yk]` (given by its ring and its domain).
    pub fn to_univariate<D: Domain<Elem = PolyElement<K>>>(
        &self,
        a: &PolyElement<K>,
        inner_ring: &Arc<PolyRing<K>>,
        inner: &D,
    ) -> Vec<PolyElement<K>> {
        to_univariate(a, inner_ring, inner)
    }

    pub fn from_univariate(&self, f: &[PolyElement<K>]) -> PolyElement<K> {
        from_univariate(&self.ring, f)
    }

    fn ground_univariate(&self, a: &PolyElement<K>) -> Vec<K::Elem> {
        let terms: std::collections::BTreeMap<usize, K::Elem> =
            a.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect();
        dup_from_dict(&terms, &self.ground)
    }

    fn from_ground_univariate(&self, f: &[K::Elem]) -> PolyElement<K> {
        let n: usize = f.len();
        self.ring.from_dict(
            f.iter()
                .enumerate()
                .map(|(i, c)| (vec![(n - 1 - i) as u32], c.clone()))
                .collect(),
        )
    }

    // -- the gcd of SymPy's PolyElement

    fn ground_gcd(&self, a: &K::Elem, b: &K::Elem) -> K::Elem {
        match self.flavor {
            GcdFlavor::Subresultant => self.ground.gcd(a, b),
            GcdFlavor::Sympy => self.ground.sympy_ground_gcd(a, b),
        }
    }

    fn ground_lcm(&self, a: &K::Elem, b: &K::Elem) -> K::Elem {
        match self.flavor {
            GcdFlavor::Subresultant => self.ground.lcm(a, b),
            GcdFlavor::Sympy => self.ground.sympy_ground_lcm(a, b),
        }
    }

    /// The ground gcd of the coefficients (SymPy's `PolyElement.content`).
    pub fn content(&self, a: &PolyElement<K>) -> K::Elem {
        let mut result: K::Elem = self.ground.zero();
        for c in a.terms.values() {
            result = self.ground_gcd(&result, c);
        }
        result
    }

    pub fn primitive(&self, a: &PolyElement<K>) -> (K::Elem, PolyElement<K>) {
        let c: K::Elem = self.content(a);
        if self.ground.is_zero(&c) || self.ground.is_one(&c) {
            return (c, a.clone());
        }
        let part: PolyElement<K> = a.exquo_ground(&c).expect("the content divides the coefficients");
        (c, part)
    }

    fn gcd_monomial(&self, term: &PolyElement<K>, g: &PolyElement<K>) -> PolyElement<K> {
        let (m0, c0) = term.terms.iter().next().expect("a term");
        let mut monomial: Monomial = m0.clone();
        let mut coefficient: K::Elem = c0.clone();
        for (m, c) in &g.terms {
            monomial = monomial.iter().zip(m).map(|(i, j)| *i.min(j)).collect();
            coefficient = self.ground_gcd(&coefficient, c);
        }
        self.ring.term(monomial, coefficient)
    }

    /// The canonical gcd (the primitive part of the last subresultant times
    /// the gcd of the contents, positive leading coefficient), each variable
    /// in turn (`SparsePolynomialRing._gcd_general`).
    pub fn subresultant_gcd(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        if self.names.len() == 1 {
            let g: Vec<K::Elem> = dup_gcd(&self.ground_univariate(a), &self.ground_univariate(b), &self.ground);
            return self.from_ground_univariate(&g);
        }
        let inner: SparsePolynomialRing<K> = SparsePolynomialRing::new(self.ground.clone(), self.names[1..].to_vec());
        let g: Vec<PolyElement<K>> = dup_gcd(
            &to_univariate(a, &inner.ring, &inner),
            &to_univariate(b, &inner.ring, &inner),
            &inner,
        );
        self.from_univariate(&g)
    }

    fn gcd_general(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        match self.flavor {
            GcdFlavor::Subresultant => self.subresultant_gcd(a, b),
            GcdFlavor::Sympy => K::sympy_gcd_general(self, a, b),
        }
    }
}

/// `a` as a dense polynomial in its first variable over the ring of the
/// other ones.
fn to_univariate<K: Domain, D: Domain<Elem = PolyElement<K>>>(
    a: &PolyElement<K>,
    inner_ring: &Arc<PolyRing<K>>,
    inner: &D,
) -> Vec<PolyElement<K>> {
    let mut by_degree: std::collections::BTreeMap<usize, Terms<K::Elem>> = std::collections::BTreeMap::new();
    for (m, c) in &a.terms {
        by_degree
            .entry(m[0] as usize)
            .or_default()
            .insert(m[1..].to_vec(), c.clone());
    }
    let coefficients: std::collections::BTreeMap<usize, PolyElement<K>> = by_degree
        .into_iter()
        .map(|(d, terms)| (d, inner_ring.from_dict(terms)))
        .collect();
    dup_from_dict(&coefficients, inner)
}

fn from_univariate<K: Domain>(ring: &Arc<PolyRing<K>>, f: &[PolyElement<K>]) -> PolyElement<K> {
    let mut terms: Terms<K::Elem> = Terms::new();
    let n: usize = f.len();
    for (i, c) in f.iter().enumerate() {
        for (m, value) in &c.terms {
            let mut key: Monomial = vec![(n - 1 - i) as u32];
            key.extend_from_slice(m);
            terms.insert(key, value.clone());
        }
    }
    ring.from_dict(terms)
}

impl<K: SparseGround> Domain for SparsePolynomialRing<K> {
    type Elem = PolyElement<K>;

    fn name(&self) -> String {
        format!("{}[{}]", self.ground.name(), self.names.join(","))
    }

    fn is_field(&self) -> bool {
        false
    }

    fn is_ordered(&self) -> bool {
        self.ground.is_ordered()
    }

    fn characteristic(&self) -> Int {
        self.ground.characteristic()
    }

    fn zero(&self) -> PolyElement<K> {
        self.ring.zero()
    }

    fn one(&self) -> PolyElement<K> {
        self.ring.one()
    }

    fn is_zero(&self, a: &PolyElement<K>) -> bool {
        a.is_zero()
    }

    fn is_negative(&self, a: &PolyElement<K>) -> bool {
        !a.is_zero() && self.ground.is_negative(&a.LC())
    }

    fn is_unit(&self, a: &PolyElement<K>) -> bool {
        a.is_ground() && !a.is_zero() && self.ground.is_unit(&a.LC())
    }

    fn add(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        a.add(b)
    }

    fn sub(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        a.sub(b)
    }

    fn neg(&self, a: &PolyElement<K>) -> PolyElement<K> {
        a.neg()
    }

    fn mul(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        a.mul(b)
    }

    fn exquo(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        if b.is_zero() {
            return Err(zero_division(&format!("division by zero in {}", self.name())));
        }
        a.exquo(b)
    }

    fn quo(&self, _a: &PolyElement<K>, _b: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        panic!("quo is not implemented for {}", self.name())
    }

    fn rem(&self, _a: &PolyElement<K>, _b: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        panic!("rem is not implemented for {}", self.name())
    }

    fn invert(&self, a: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        if self.is_unit(a) {
            return Ok(self.ring.ground(self.ground.invert(&a.LC())?));
        }
        Err(value_error(format!("{} is not a unit of {}", a, self.name())))
    }

    /// SymPy's `PolyElement.gcd`: `b` or `-b` (nonnegative leading
    /// coefficient) when `a = 0`; the monomial gcd times the ground gcd of
    /// the coefficients when one of them is a term; the canonical gcd
    /// otherwise.
    fn gcd(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        if a.is_zero() && b.is_zero() {
            return self.zero();
        }
        if a.is_zero() {
            return if self.is_nonnegative(b) { b.clone() } else { b.neg() };
        }
        if b.is_zero() {
            return if self.is_nonnegative(a) { a.clone() } else { a.neg() };
        }
        if a.terms.len() == 1 {
            return self.gcd_monomial(a, b);
        }
        if b.terms.len() == 1 {
            return self.gcd_monomial(b, a);
        }
        self.gcd_general(a, b)
    }

    /// SymPy's `PolyElement.lcm`: `f g / gcd(f, g)` of the primitive parts
    /// times the lcm of the contents (monic over `QQ` with SymPy's gcd).
    fn lcm(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        if self.flavor == GcdFlavor::Sympy {
            if let Some(result) = K::sympy_lcm(self, a, b) {
                return result;
            }
        }
        if a.is_zero() || b.is_zero() {
            return self.zero();
        }
        let (ca, pa) = self.primitive(a);
        let (cb, pb) = self.primitive(b);
        let c: K::Elem = self.ground_lcm(&ca, &cb);
        pa.mul(&pb)
            .exquo(&self.gcd(&pa, &pb))
            .expect("the gcd divides the product")
            .mul_ground(&c)
    }

    fn gcdex(
        &self,
        _a: &PolyElement<K>,
        _b: &PolyElement<K>,
    ) -> PolyResult<(PolyElement<K>, PolyElement<K>, PolyElement<K>)> {
        panic!("gcdex is not implemented for {}", self.name())
    }

    /// `-1` when the leading coefficient is negative, `1` otherwise.
    fn canonical_unit(&self, a: &PolyElement<K>) -> PolyElement<K> {
        if !a.is_zero() && self.ground.is_negative(&a.LC()) {
            return self.one().neg();
        }
        self.one()
    }

    fn size(&self, a: &PolyElement<K>) -> u64 {
        let total: u64 = a.terms.values().map(|c| self.ground.size(c)).sum();
        if total == 0 {
            1
        } else {
            total
        }
    }

    fn from_int(&self, n: i64) -> PolyElement<K> {
        self.ring.from_int(n)
    }

    fn from_integer(&self, n: &Int) -> PolyElement<K> {
        self.ring.ground(self.ground.from_integer(n))
    }

    fn to_str(&self, a: &PolyElement<K>) -> String {
        a.to_string()
    }
}

/// `K[x1, ..., xn]` over a polynomial ring `K = k[y1, ..., ym]` with SymPy's
/// gcd of dense polynomials over a polynomial domain (`dmp_rr_prs_gcd`):
/// contents taken variable after variable down to `K`, whose gcd is
/// [`SparsePolynomialRing`]'s. The elements are sparse polynomials in `x1,
/// ..., xn, y1, ..., ym`. With no generator (`n = 0`) the ring is `K`
/// itself, the coefficient ring of the recursion.
#[derive(Debug, Clone)]
pub struct IteratedPolynomialRing<K: Domain> {
    pub params: SparsePolynomialRing<K>,
    pub gen_names: Vec<String>,
    pub n_gens: usize,
    /// The ring in all the variables (`params` itself when `n = 0`), with
    /// the generic conventions of `SparsePolynomialRing`.
    pub base: SparsePolynomialRing<K>,
}

impl<K: Domain> PartialEq for IteratedPolynomialRing<K> {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params && self.gen_names == other.gen_names
    }
}

impl<K: SparseGround> IteratedPolynomialRing<K> {
    pub fn new(params: SparsePolynomialRing<K>, gen_names: Vec<String>) -> IteratedPolynomialRing<K> {
        let n_gens: usize = gen_names.len();
        let base: SparsePolynomialRing<K> = if n_gens == 0 {
            params.clone()
        } else {
            let mut names: Vec<String> = gen_names.clone();
            names.extend(params.names.iter().cloned());
            SparsePolynomialRing::new(params.ground.clone(), names)
        };
        IteratedPolynomialRing {
            params,
            gen_names,
            n_gens,
            base,
        }
    }

    pub fn ring(&self) -> &Arc<PolyRing<K>> {
        &self.base.ring
    }

    /// The leading coefficient in `x1, ..., xn` (lex), an element of `K`
    /// (SymPy's `dmp_ground_LC`).
    #[allow(non_snake_case)]
    pub fn ground_LC(&self, a: &PolyElement<K>) -> PolyElement<K> {
        if a.is_zero() {
            return self.params.zero();
        }
        let n: usize = self.n_gens;
        let top: &[u32] = a.terms.keys().map(|m| &m[..n]).max().expect("a nonzero polynomial");
        let top: Monomial = top.to_vec();
        self.params.ring.from_dict(
            a.terms
                .iter()
                .filter(|(m, _)| m[..n] == top[..])
                .map(|(m, c)| (m[n..].to_vec(), c.clone()))
                .collect(),
        )
    }

    /// The coefficients in `x1`: `K[x2, ..., xn]`, or `K`.
    fn inner(&self) -> IteratedPolynomialRing<K> {
        IteratedPolynomialRing::new(self.params.clone(), self.gen_names[1..].to_vec())
    }

    /// An element of the inner ring as a constant in `x1`.
    fn lift(&self, c: &PolyElement<K>) -> PolyElement<K> {
        self.ring().from_dict(
            c.terms
                .iter()
                .map(|(m, v)| {
                    let mut key: Monomial = vec![0];
                    key.extend_from_slice(m);
                    (key, v.clone())
                })
                .collect(),
        )
    }

    /// An element of `K` in this ring.
    pub fn embed_params(&self, c: &PolyElement<K>) -> PolyElement<K> {
        self.ring().from_dict(
            c.terms
                .iter()
                .map(|(m, v)| {
                    let mut key: Monomial = vec![0; self.n_gens];
                    key.extend_from_slice(m);
                    (key, v.clone())
                })
                .collect(),
        )
    }

    fn univariate(&self, a: &PolyElement<K>, inner: &IteratedPolynomialRing<K>) -> Vec<PolyElement<K>> {
        to_univariate(a, inner.ring(), inner)
    }

    fn content(&self, f: &[PolyElement<K>], inner: &IteratedPolynomialRing<K>) -> PolyElement<K> {
        if f.is_empty() {
            return inner.zero();
        }
        if self.n_gens == 1 {
            let mut content: PolyElement<K> = inner.zero();
            for c in f {
                content = inner.gcd(&content, c);
                if inner.is_one(&content) {
                    break;
                }
            }
            return content;
        }
        let mut content: PolyElement<K> = f[0].clone();
        for c in &f[1..] {
            content = inner.gcd(&content, c);
            if inner.is_one(&content) {
                break;
            }
        }
        if inner.n_gens > 0 && self.params.is_negative(&inner.ground_LC(&content)) {
            return content.neg();
        }
        content
    }

    fn primitive(
        &self,
        f: Vec<PolyElement<K>>,
        inner: &IteratedPolynomialRing<K>,
    ) -> (PolyElement<K>, Vec<PolyElement<K>>) {
        let content: PolyElement<K> = self.content(&f, inner);
        if f.is_empty() || inner.is_one(&content) {
            return (content, f);
        }
        let parts: Vec<PolyElement<K>> = f
            .iter()
            .map(|c| inner.exquo(c, &content).expect("the content divides the coefficients"))
            .collect();
        (content, parts)
    }

    /// The gcd in `K` of the coefficients in `x1, ..., xn`, folded as SymPy
    /// folds it over the dense representation (the gcd of `QQ[y]` is not
    /// associative).
    pub fn ground_content(&self, a: &PolyElement<K>) -> PolyElement<K> {
        let k: &SparsePolynomialRing<K> = &self.params;
        if a.is_zero() {
            return k.zero();
        }
        let inner: IteratedPolynomialRing<K> = self.inner();
        let mut content: PolyElement<K> = k.zero();
        for c in self.univariate(a, &inner) {
            let part: PolyElement<K> = if inner.n_gens > 0 { inner.ground_content(&c) } else { c };
            content = k.gcd(&content, &part);
            if k.is_one(&content) {
                break;
            }
        }
        content
    }
}

impl<K: SparseGround> Domain for IteratedPolynomialRing<K> {
    type Elem = PolyElement<K>;

    fn name(&self) -> String {
        if self.n_gens == 0 {
            return self.params.name();
        }
        format!("{}[{}]", self.params.name(), self.gen_names.join(","))
    }

    fn is_field(&self) -> bool {
        false
    }

    fn is_ordered(&self) -> bool {
        self.base.is_ordered()
    }

    fn characteristic(&self) -> Int {
        self.base.characteristic()
    }

    fn zero(&self) -> PolyElement<K> {
        self.base.zero()
    }

    fn one(&self) -> PolyElement<K> {
        self.base.one()
    }

    fn is_zero(&self, a: &PolyElement<K>) -> bool {
        a.is_zero()
    }

    fn is_negative(&self, a: &PolyElement<K>) -> bool {
        self.base.is_negative(a)
    }

    fn is_unit(&self, a: &PolyElement<K>) -> bool {
        self.base.is_unit(a)
    }

    fn add(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        a.add(b)
    }

    fn sub(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        a.sub(b)
    }

    fn neg(&self, a: &PolyElement<K>) -> PolyElement<K> {
        a.neg()
    }

    fn mul(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        a.mul(b)
    }

    fn exquo(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        self.base.exquo(a, b)
    }

    fn quo(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        self.base.quo(a, b)
    }

    fn rem(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        self.base.rem(a, b)
    }

    fn invert(&self, a: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        self.base.invert(a)
    }

    fn gcd(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        if self.n_gens == 0 {
            return self.params.gcd(a, b);
        }
        let k: &SparsePolynomialRing<K> = &self.params;
        if a.is_zero() && b.is_zero() {
            return self.zero();
        }
        if a.is_zero() {
            return if k.is_nonnegative(&self.ground_LC(b)) {
                b.clone()
            } else {
                b.neg()
            };
        }
        if b.is_zero() {
            return if k.is_nonnegative(&self.ground_LC(a)) {
                a.clone()
            } else {
                a.neg()
            };
        }
        let inner: IteratedPolynomialRing<K> = self.inner();
        let big_f: Vec<PolyElement<K>> = self.univariate(a, &inner);
        let big_g: Vec<PolyElement<K>> = self.univariate(b, &inner);
        if self.n_gens > 1 {
            if self.is_one(a) || self.is_one(b) {
                return self.one();
            }
            if big_f.len() == 1 || big_g.len() == 1 {
                let (left, right) = if big_f.len() == 1 && big_g.len() == 1 {
                    (big_f[0].clone(), big_g[0].clone())
                } else if big_f.len() == 1 {
                    (big_f[0].clone(), self.content(&big_g, &inner))
                } else {
                    (self.content(&big_f, &inner), big_g[0].clone())
                };
                return self.lift(&inner.gcd(&left, &right));
            }
        }
        let (fc, big_f) = self.primitive(big_f, &inner);
        let (gc, big_g) = self.primitive(big_g, &inner);
        let sequence: Vec<Vec<PolyElement<K>>> = dup_subresultants(&big_f, &big_g, &inner);
        let h: Vec<PolyElement<K>> = sequence.last().expect("a nonempty sequence").clone();
        let mut c: PolyElement<K> = inner.gcd(&fc, &gc);
        let (_, h) = self.primitive(h, &inner);
        if self.n_gens == 1 {
            c = c.mul(&k.canonical_unit(&h[0]));
            let scaled: Vec<PolyElement<K>> = h.iter().map(|hc| hc.mul(&c)).collect();
            return from_univariate(self.ring(), &scaled);
        }
        let scaled: Vec<PolyElement<K>> = h.iter().map(|hc| hc.mul(&c)).collect();
        let result: PolyElement<K> = from_univariate(self.ring(), &scaled);
        let unit: PolyElement<K> = k.canonical_unit(&self.ground_LC(&result));
        if k.is_one(&unit) {
            result
        } else {
            result.mul(&self.embed_params(&unit))
        }
    }

    fn lcm(&self, a: &PolyElement<K>, b: &PolyElement<K>) -> PolyElement<K> {
        if self.n_gens == 0 {
            return self.params.lcm(a, b);
        }
        let k: &SparsePolynomialRing<K> = &self.params;
        if a.is_zero() || b.is_zero() {
            return self.zero();
        }
        let (fc, gc) = (self.ground_content(a), self.ground_content(b));
        let f: PolyElement<K> = if k.is_one(&fc) {
            a.clone()
        } else {
            a.exquo(&self.embed_params(&fc)).expect("the content divides")
        };
        let g: PolyElement<K> = if k.is_one(&gc) {
            b.clone()
        } else {
            b.exquo(&self.embed_params(&gc)).expect("the content divides")
        };
        let mut c: PolyElement<K> = k.lcm(&fc, &gc);
        let h: PolyElement<K> = f.mul(&g).exquo(&self.gcd(&f, &g)).expect("the gcd divides the product");
        if self.n_gens == 1 {
            c = c.mul(&k.canonical_unit(&self.ground_LC(&h)));
        }
        h.mul(&self.embed_params(&c))
    }

    fn gcdex(
        &self,
        a: &PolyElement<K>,
        b: &PolyElement<K>,
    ) -> PolyResult<(PolyElement<K>, PolyElement<K>, PolyElement<K>)> {
        self.base.gcdex(a, b)
    }

    fn canonical_unit(&self, a: &PolyElement<K>) -> PolyElement<K> {
        self.base.canonical_unit(a)
    }

    fn size(&self, a: &PolyElement<K>) -> u64 {
        self.base.size(a)
    }

    fn from_int(&self, n: i64) -> PolyElement<K> {
        self.base.from_int(n)
    }

    fn from_integer(&self, n: &Int) -> PolyElement<K> {
        self.base.from_integer(n)
    }

    fn to_str(&self, a: &PolyElement<K>) -> String {
        a.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::QQ;

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    fn zz_poly(r: &Arc<PolyRing<IntegerRing>>, terms: &[(&[u32], i64)]) -> PolyElement<IntegerRing> {
        r.from_dict(terms.iter().map(|(m, c)| (m.to_vec(), int(*c))).collect())
    }

    fn qq_poly(r: &Arc<PolyRing<RationalField>>, terms: &[(&[u32], i64, i64)]) -> PolyElement<RationalField> {
        r.from_dict(terms.iter().map(|(m, p, q)| (m.to_vec(), rat(*p, *q))).collect())
    }

    #[test]
    fn doctests() {
        // (x + y)(x - 2y) and (x + y)(3x + y)
        let r = SparsePolynomialRing::new(ZZ, names(&["x", "y"]));
        let f = zz_poly(&r.ring, &[(&[2, 0], 1), (&[1, 1], -1), (&[0, 2], -2)]);
        let g = zz_poly(&r.ring, &[(&[2, 0], 3), (&[1, 1], 4), (&[0, 2], 1)]);
        assert_eq!(r.gcd(&f, &g).to_string(), "x + y");
        assert_eq!(r.name(), "ZZ[x,y]");
        // IntegerPolynomialRing: gcd(x**4 - y**2, x**6 + y**3) = x**2 + y
        let z = SparsePolynomialRing::sympy(ZZ, names(&["x", "y"]));
        let a = zz_poly(&z.ring, &[(&[4, 0], 1), (&[0, 2], -1)]);
        let b = zz_poly(&z.ring, &[(&[6, 0], 1), (&[0, 3], 1)]);
        assert_eq!(z.gcd(&a, &b).to_string(), "x**2 + y");
        // RationalPolynomialRing: gcd(-a/2, a/3) = a/6, gcd(2a + 2, a/3 + 1/3) = a + 1
        let q = SparsePolynomialRing::sympy(QQ, names(&["a"]));
        let p1 = qq_poly(&q.ring, &[(&[1], -1, 2)]);
        let p2 = qq_poly(&q.ring, &[(&[1], 1, 3)]);
        assert_eq!(q.gcd(&p1, &p2).to_string(), "1/6*a");
        let p3 = qq_poly(&q.ring, &[(&[1], 2, 1), (&[0], 2, 1)]);
        let p4 = qq_poly(&q.ring, &[(&[1], 1, 3), (&[0], 1, 3)]);
        assert_eq!(q.gcd(&p3, &p4).to_string(), "a + 1");
        // IteratedPolynomialRing(RationalPolynomialRing(['y']), ['x']):
        // gcd(y x/2 + y, x/3) = 1/6
        let it = IteratedPolynomialRing::new(SparsePolynomialRing::sympy(QQ, names(&["y"])), names(&["x"]));
        let e1 = qq_poly(it.ring(), &[(&[1, 1], 1, 2), (&[0, 1], 1, 1)]);
        let e2 = qq_poly(it.ring(), &[(&[1, 0], 1, 3)]);
        assert_eq!(it.gcd(&e1, &e2).to_string(), "1/6");
        assert_eq!(it.name(), "QQ[y][x]");
    }
}
