//! Port of `sympy_extras/polys/differential/polynomial.py`: differential
//! polynomials over a [`DifferentialRing`] (coefficients in `ZZ[p, x]`,
//! fraction-free), their leaders, initials and separants, derivatives, and
//! Ritt's reduction (pseudo-remainders, partial and full remainders by a
//! chain).

// a node caches its keys in `OnceLock`s that the hash never reads
#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::core::add::add;
use crate::core::basic::Basic;
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::PolyResult;
use crate::numbers::*;
use crate::polys::domains::domain::Domain;

use super::ring::{
    integer_product, integer_quotient, monomial_product, DifferentialRing, IntegerCoefficient, Jet, JetMonomial, Terms,
};

/// The rank of a differential polynomial: the key of its leader and its
/// degree.
pub type Rank = (Vec<i64>, u32);

/// A polynomial in the derivatives of the functions of a
/// [`DifferentialRing`] (`DifferentialPolynomial`).
#[derive(Debug, Clone)]
pub struct DifferentialPolynomial {
    pub ring: Rc<DifferentialRing>,
    pub terms: Terms,
    derivatives: RefCell<HashMap<usize, DifferentialPolynomial>>,
}

impl PartialEq for DifferentialPolynomial {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl Eq for DifferentialPolynomial {}

impl Hash for DifferentialPolynomial {
    /// `hash(frozenset(self.terms))`: the monomials, in any order.
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut keys: Vec<&JetMonomial> = self.terms.keys().collect();
        keys.sort();
        keys.hash(state);
    }
}

fn accumulate(terms: &mut Terms, m: JetMonomial, c: IntegerCoefficient) {
    let value: IntegerCoefficient = match terms.get(&m) {
        Some(existing) => existing.add(&c),
        None => c,
    };
    if value.is_zero() {
        terms.remove(&m);
    } else {
        terms.insert(m, value);
    }
}

impl DifferentialPolynomial {
    pub fn new(ring: Rc<DifferentialRing>, terms: Terms) -> DifferentialPolynomial {
        DifferentialPolynomial {
            ring,
            terms,
            derivatives: RefCell::new(HashMap::new()),
        }
    }

    /// The differential polynomial of an expression (`from_expr`).
    pub fn from_expr(ring: Rc<DifferentialRing>, equation: &Basic) -> PolyResult<DifferentialPolynomial> {
        let terms: Terms = ring.integral(&ring.terms(equation)?);
        Ok(DifferentialPolynomial::new(ring, terms).primitive())
    }

    /// The expression, the terms by decreasing monomial (`to_expr`).
    pub fn to_expr(&self) -> PolyResult<Basic> {
        let mut monomials: Vec<&JetMonomial> = self.terms.keys().collect();
        monomials.sort_by_cached_key(|m| std::cmp::Reverse(self.monomial_key(m)));
        let mut parts: Vec<Basic> = Vec::new();
        for m in monomials {
            let mut term: Basic = self.ring.integer_expr(self.terms.get(m).expect("a term"), None)?;
            for (jet, e) in m {
                term = mul(&[term, pow(&self.ring.to_expr(jet), &Basic::int(i64::from(*e)))?])?;
            }
            parts.push(term);
        }
        if parts.is_empty() {
            return Ok(Basic::zero());
        }
        add(&parts)
    }

    pub fn is_nonzero(&self) -> bool {
        !self.terms.is_empty()
    }

    // -- arithmetic

    /// The element `c` of the ring of coefficients.
    pub fn constant(&self, c: IntegerCoefficient) -> DifferentialPolynomial {
        let mut terms: Terms = Terms::new();
        if !c.is_zero() {
            terms.insert(Vec::new(), c);
        }
        DifferentialPolynomial::new(self.ring.clone(), terms)
    }

    pub fn neg(&self) -> DifferentialPolynomial {
        DifferentialPolynomial::new(
            self.ring.clone(),
            self.terms.iter().map(|(m, c)| (m.clone(), c.neg())).collect(),
        )
    }

    pub fn add(&self, other: &DifferentialPolynomial) -> DifferentialPolynomial {
        let mut terms: Terms = self.terms.clone();
        for (m, c) in other.terms.iter() {
            accumulate(&mut terms, m.clone(), c.clone());
        }
        DifferentialPolynomial::new(self.ring.clone(), terms)
    }

    pub fn sub(&self, other: &DifferentialPolynomial) -> DifferentialPolynomial {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &DifferentialPolynomial) -> DifferentialPolynomial {
        let mut terms: Terms = Terms::new();
        for (m, c) in self.terms.iter() {
            for (k, d) in other.terms.iter() {
                accumulate(&mut terms, monomial_product(m, k), integer_product(c, d));
            }
        }
        DifferentialPolynomial::new(self.ring.clone(), terms)
    }

    pub fn pow(&self, n: u32) -> DifferentialPolynomial {
        let mut result: DifferentialPolynomial = self.constant(self.ring.integers.one());
        for _ in 0..n {
            result = result.mul(self);
        }
        result
    }

    /// The polynomial divided by the gcd of its coefficients, with a
    /// positive leading coefficient (`primitive`).
    pub fn primitive(&self) -> DifferentialPolynomial {
        if self.terms.is_empty() {
            return self.clone();
        }
        let z = &self.ring.integers;
        let one: IntegerCoefficient = z.one();
        let values: Vec<&IntegerCoefficient> = self.terms.values().collect();
        let mut divisor: IntegerCoefficient = values[0].clone();
        for c in &values[1..] {
            if divisor == one {
                break;
            }
            divisor = z.gcd(&divisor, c);
        }
        let first: &IntegerCoefficient = self.terms.get(self.highest_monomial()).expect("a term");
        if is_negative(&first.LC()) != is_negative(&divisor.LC()) {
            divisor = divisor.neg();
        }
        if divisor == one {
            return self.clone();
        }
        let terms: Terms = self
            .terms
            .iter()
            .map(|(m, c)| {
                (
                    m.clone(),
                    integer_quotient(c, &divisor).expect("the gcd divides the coefficients"),
                )
            })
            .collect();
        DifferentialPolynomial::new(self.ring.clone(), terms)
    }

    /// The highest monomial (`max(self.terms, key=self._monomial_key)`, the
    /// first of the equal ones).
    fn highest_monomial(&self) -> &JetMonomial {
        let mut best: Option<(&JetMonomial, Vec<(Vec<i64>, u32)>)> = None;
        for m in self.terms.keys() {
            let key = self.monomial_key(m);
            let greater: bool = match &best {
                None => true,
                Some((_, best_key)) => key > *best_key,
            };
            if greater {
                best = Some((m, key));
            }
        }
        best.expect("a term").0
    }

    fn monomial_key(&self, m: &JetMonomial) -> Vec<(Vec<i64>, u32)> {
        let mut key: Vec<(Vec<i64>, u32)> = m.iter().map(|(jet, e)| (self.ring.rank_key(jet), *e)).collect();
        key.sort_by(|a, b| b.cmp(a));
        key
    }

    // -- structure

    /// Whether the polynomial lies in the field of coefficients.
    pub fn is_ground(&self) -> bool {
        self.terms.keys().all(|m| m.is_empty())
    }

    /// The derivatives which occur.
    pub fn jets(&self) -> HashSet<Jet> {
        self.terms
            .keys()
            .flat_map(|m| m.iter().map(|(j, _)| j.clone()))
            .collect()
    }

    /// The highest derivative; `Value` for an element of the field.
    pub fn leader(&self) -> PolyResult<Jet> {
        let jets: Vec<Jet> = self.jets().into_iter().collect();
        if jets.is_empty() {
            return Err(crate::error::PolyError::Value(
                "an element of the field of coefficients has no leader".to_string(),
            ));
        }
        // the rank keys are distinct: the order of the set does not matter
        Ok(self.ring.highest(&jets))
    }

    pub fn leader_expr(&self) -> PolyResult<Basic> {
        Ok(self.ring.to_expr(&self.leader()?))
    }

    /// The degree in a derivative.
    pub fn degree_in(&self, v: &Jet) -> u32 {
        self.terms
            .keys()
            .flat_map(|m| m.iter())
            .filter(|(w, _)| w == v)
            .map(|(_, e)| *e)
            .max()
            .unwrap_or(0)
    }

    /// The degree in the leader.
    pub fn degree(&self) -> PolyResult<u32> {
        Ok(self.degree_in(&self.leader()?))
    }

    /// The rank: the leader and the degree in it.
    pub fn rank(&self) -> PolyResult<Rank> {
        Ok((self.ring.rank_key(&self.leader()?), self.degree()?))
    }

    /// The coefficients as a polynomial in one derivative, by degree, in
    /// the order of first occurrence (`coefficients`).
    pub fn coefficients(&self, jet: &Jet) -> Vec<(u32, DifferentialPolynomial)> {
        let mut parts: Vec<(u32, Terms)> = Vec::new();
        for (m, c) in self.terms.iter() {
            let mut e: u32 = 0;
            let mut rest: JetMonomial = m.clone();
            if let Some(k) = m.iter().position(|(w, _)| w == jet) {
                e = m[k].1;
                rest.remove(k);
            }
            match parts.iter_mut().find(|(d, _)| *d == e) {
                Some((_, terms)) => terms.insert(rest, c.clone()),
                None => {
                    let mut terms: Terms = Terms::new();
                    terms.insert(rest, c.clone());
                    parts.push((e, terms));
                }
            }
        }
        parts
            .into_iter()
            .map(|(e, terms)| (e, DifferentialPolynomial::new(self.ring.clone(), terms)))
            .collect()
    }

    fn coefficient_of(&self, jet: &Jet, e: u32) -> DifferentialPolynomial {
        self.coefficients(jet)
            .into_iter()
            .find(|(d, _)| *d == e)
            .expect("a coefficient")
            .1
    }

    /// The coefficient of the highest power of the leader.
    pub fn initial(&self) -> PolyResult<DifferentialPolynomial> {
        Ok(self.coefficient_of(&self.leader()?, self.degree()?))
    }

    /// The partial derivative with respect to the leader.
    pub fn separant(&self) -> PolyResult<DifferentialPolynomial> {
        let v: Jet = self.leader()?;
        let mut result: DifferentialPolynomial = self.constant(self.ring.integers.zero());
        for (e, c) in self.coefficients(&v) {
            if e > 0 {
                let scaled: DifferentialPolynomial = c.scaled(&self.ring.integers.from_int(i64::from(e)));
                result = result.add(&scaled.mul(&self.power(&v, e - 1)));
            }
        }
        Ok(result)
    }

    /// The product with an element of the ring of coefficients.
    pub fn scaled(&self, c: &IntegerCoefficient) -> DifferentialPolynomial {
        if c.is_zero() {
            return self.constant(c.clone());
        }
        DifferentialPolynomial::new(
            self.ring.clone(),
            self.terms
                .iter()
                .map(|(m, d)| (m.clone(), integer_product(d, c)))
                .collect(),
        )
    }

    /// The power of a derivative.
    pub fn power(&self, jet: &Jet, e: u32) -> DifferentialPolynomial {
        let mut terms: Terms = Terms::new();
        terms.insert(
            if e > 0 { vec![(jet.clone(), e)] } else { Vec::new() },
            self.ring.integers.one(),
        );
        DifferentialPolynomial::new(self.ring.clone(), terms)
    }

    /// The derivative with respect to the variable `i` (cached).
    pub fn derivative(&self, i: usize) -> DifferentialPolynomial {
        if let Some(cached) = self.derivatives.borrow().get(&i) {
            return cached.clone();
        }
        let ring = &self.ring;
        let mut terms: Terms = Terms::new();
        for (m, c) in self.terms.iter() {
            let d: IntegerCoefficient = ring.integer_derivative(c, i);
            if !d.is_zero() {
                accumulate(&mut terms, m.clone(), d);
            }
            for (k, (jet, e)) in m.iter().enumerate() {
                let mut rest: JetMonomial = m.clone();
                rest.remove(k);
                let factor: JetMonomial = if *e > 1 { vec![(jet.clone(), e - 1)] } else { Vec::new() };
                let product: JetMonomial = monomial_product(
                    &monomial_product(&rest, &factor),
                    &vec![(ring.differentiate(jet, i), 1)],
                );
                accumulate(&mut terms, product, c.mul_ground(&Int::from(*e)));
            }
        }
        let result: DifferentialPolynomial = DifferentialPolynomial::new(ring.clone(), terms);
        self.derivatives.borrow_mut().insert(i, result.clone());
        result
    }

    /// The derivative of the given orders.
    pub fn prolonged(&self, theta: &[u32]) -> DifferentialPolynomial {
        let mut result: DifferentialPolynomial = self.clone();
        for (i, k) in theta.iter().enumerate() {
            for _ in 0..*k {
                result = result.derivative(i);
            }
        }
        result
    }

    // -- reduction

    /// The pseudo-remainder of the division by `divisor` as polynomials in
    /// the derivative `jet` (`pseudo_remainder`).
    pub fn pseudo_remainder(&self, divisor: &DifferentialPolynomial, jet: &Jet) -> DifferentialPolynomial {
        let d: u32 = divisor.degree_in(jet);
        let lead: DifferentialPolynomial = divisor.coefficient_of(jet, d);
        let mut remainder: DifferentialPolynomial = self.clone();
        while remainder.is_nonzero() {
            let e: u32 = remainder.degree_in(jet);
            if e < d {
                break;
            }
            let top: DifferentialPolynomial = remainder.coefficient_of(jet, e);
            remainder = lead.mul(&remainder).sub(&top.mul(&self.power(jet, e - d)).mul(divisor));
        }
        remainder
    }

    /// Whether no proper derivative of the leader of `other` occurs.
    pub fn is_partially_reduced(&self, other: &DifferentialPolynomial) -> PolyResult<bool> {
        let v: Jet = other.leader()?;
        Ok(!self.jets().iter().any(|w| *w != v && self.ring.is_derivative_of(w, &v)))
    }

    /// Whether the polynomial is partially reduced with respect to `other`
    /// and of lower degree in its leader.
    pub fn is_reduced(&self, other: &DifferentialPolynomial) -> PolyResult<bool> {
        Ok(self.is_partially_reduced(other)? && self.degree_in(&other.leader()?) < other.degree()?)
    }

    /// Ritt's partial remainder by a chain (`partial_remainder`).
    pub fn partial_remainder(&self, chain: &[DifferentialPolynomial]) -> PolyResult<DifferentialPolynomial> {
        let ring = &self.ring;
        let mut leaders: Vec<Jet> = Vec::new();
        for a in chain {
            leaders.push(a.leader()?);
        }
        let mut remainder: DifferentialPolynomial = self.clone();
        while remainder.is_nonzero() {
            let mut found: Option<(Jet, usize)> = None;
            for w in remainder.jets() {
                for (k, v) in leaders.iter().enumerate() {
                    if w != *v && ring.is_derivative_of(&w, v) {
                        let higher: bool = match &found {
                            None => true,
                            Some((f, _)) => ring.rank_key(&w) > ring.rank_key(f),
                        };
                        if higher {
                            found = Some((w.clone(), k));
                        }
                        break;
                    }
                }
            }
            let Some((w, k)) = found else { break };
            let theta: Vec<u32> = w.1.iter().zip(&leaders[k].1).map(|(j, l)| j - l).collect();
            remainder = remainder.pseudo_remainder(&chain[k].prolonged(&theta), &w).primitive();
        }
        Ok(remainder)
    }

    /// Ritt's full remainder by an autoreduced set (`remainder`).
    pub fn remainder(&self, chain: &[DifferentialPolynomial]) -> PolyResult<DifferentialPolynomial> {
        let mut remainder: DifferentialPolynomial = self.partial_remainder(chain)?;
        let mut ordered: Vec<(Vec<i64>, &DifferentialPolynomial)> = Vec::new();
        for a in chain {
            ordered.push((self.ring.rank_key(&a.leader()?), a));
        }
        // Python's stable sort by decreasing key
        ordered.sort_by(|x, y| y.0.cmp(&x.0));
        for (_, a) in ordered {
            if !remainder.is_nonzero() {
                break;
            }
            remainder = remainder.pseudo_remainder(a, &a.leader()?).primitive();
        }
        Ok(remainder)
    }
}
