//! Port of `sympy_extras/polys/regularchains/operations.py`: the operations
//! on squarefree regular chains behind the triangular decompositions
//! (Chen and Moreno Maza): `regularize`, `regular_gcd`, `intersect`,
//! `extend`, and `triangularize`, recursive from the top of the chain; in the
//! Kalkbrener mode (`bound`) the chains of greater height are dropped.
//!
//! The order of the results is Python's: the pool of the irreducible factors
//! met keeps their order of insertion (a Python dict), the chains are told
//! apart by position where Python compares identities, and the caches are
//! keyed by the terms of the polynomials.
//!
//! References
//! ==========
//!
//! 1. C. Chen, M. Moreno Maza, *Algorithms for computing triangular
//!    decomposition of polynomial systems*, Journal of Symbolic Computation
//!    47 (2012), 610-642.
//! 2. M. Kalkbrener, *A generalized Euclidean algorithm for computing
//!    triangular representations of algebraic varieties*, Journal of
//!    Symbolic Computation 15 (1993), 143-167.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::error::{arithmetic_error, PolyError, PolyResult};
use crate::ntheory::nextprime;
use crate::numbers::*;

use super::super::domains::IntegerRing;
use super::super::rings::{PolyRing, Terms};
use super::recursive::{
    coefficient, degree, distinct_factors, exact_quotient, initial, main_variable, max_norm, normalize, primitive_part,
    pseudo_divide, pseudo_remainder, sort_key, sqf_part, subresultant_chain, tail, ZPoly,
};

/// Polynomials with a coefficient above this are not factored.
fn factor_norm() -> Int {
    Int::from(10u8).pow(6)
}

type Key = Terms<Int>;

fn key_of(p: &ZPoly) -> Key {
    p.terms.clone()
}

/// A triangular set: at most one polynomial per main variable; `polys[i]` is
/// the polynomial whose main variable has the index `i`, or `None`.
#[derive(Debug, Clone)]
pub struct Chain {
    pub polys: Vec<Option<ZPoly>>,
    pub height: usize,
}

impl PartialEq for Chain {
    fn eq(&self, other: &Chain) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Chain {}

impl std::hash::Hash for Chain {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.polys.iter().flatten().map(|p| p.to_string()).collect();
        write!(f, "Chain({})", parts.join(", "))
    }
}

impl Chain {
    pub fn new(polys: Vec<Option<ZPoly>>) -> Chain {
        let height: usize = polys.iter().filter(|p| p.is_some()).count();
        Chain { polys, height }
    }

    fn key(&self) -> Vec<Option<Key>> {
        self.polys.iter().map(|p| p.as_ref().map(key_of)).collect()
    }

    pub fn get(&self, i: usize) -> Option<&ZPoly> {
        self.polys[i].as_ref()
    }

    /// The polynomials whose main variable is smaller than the variable `i`.
    pub fn below(&self, i: usize) -> Chain {
        let mut polys: Vec<Option<ZPoly>> = vec![None; i + 1];
        polys.extend(self.polys[i + 1..].iter().cloned());
        Chain::new(polys)
    }

    /// The polynomials whose main variable is greater than the variable `i`,
    /// by increasing main variable.
    pub fn above(&self, i: usize) -> Vec<ZPoly> {
        self.polys[..i].iter().rev().flatten().cloned().collect()
    }

    pub fn with_poly(&self, i: usize, p: ZPoly) -> Chain {
        let mut polys: Vec<Option<ZPoly>> = self.polys.clone();
        polys[i] = Some(p);
        Chain::new(polys)
    }

    pub fn union(&self, polys: &[ZPoly]) -> Chain {
        let mut result: Vec<Option<ZPoly>> = self.polys.clone();
        for p in polys {
            result[main_variable(p)] = Some(p.clone());
        }
        Chain::new(result)
    }

    /// The polynomials by increasing main variable.
    pub fn elements(&self) -> Vec<ZPoly> {
        self.polys.iter().rev().flatten().cloned().collect()
    }
}

/// A chain with whether the polynomial is zero (`true`) or regular modulo it.
pub type Regularized = (Chain, bool);

/// A regular gcd (`None` on a chain of greater height), the chain, and the
/// polynomial outside whose zeros the gcd specializes.
pub type RegularGcd = (Option<ZPoly>, Chain, ZPoly);

/// `p/q` when `q` divides `p`, `None` otherwise.
fn divided(p: &ZPoly, q: &ZPoly) -> PolyResult<Option<ZPoly>> {
    match exact_quotient(p, q) {
        Ok(quotient) => Ok(Some(quotient)),
        Err(PolyError::Value(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn distinct(chains: Vec<Chain>) -> Vec<Chain> {
    let mut seen: HashSet<Chain> = HashSet::new();
    let mut result: Vec<Chain> = Vec::new();
    for chain in chains {
        if seen.insert(chain.clone()) {
            result.push(chain);
        }
    }
    result
}

/// The operations on the squarefree regular chains of a polynomial ring over
/// the integers, with their caches; `bound` is the greatest height of the
/// chains which are kept (the Kalkbrener mode), `None` to keep them all.
pub struct Decomposer {
    pub ring: Arc<PolyRing<IntegerRing>>,
    pub n: usize,
    pub bound: Option<usize>,
    pub empty: Chain,
    regularized: HashMap<(Key, Chain), Vec<Regularized>>,
    intersected: HashMap<(Key, Chain), Vec<Chain>>,
    subresultant_cache: HashMap<(Key, Key, usize), Vec<ZPoly>>,
    factor_cache: HashMap<Key, Vec<ZPoly>>,
    /// The irreducible factors met so far, with their values at the point,
    /// in their order of insertion.
    pool: Vec<(ZPoly, Int)>,
    pool_index: HashMap<Key, usize>,
    point: Vec<Int>,
}

impl Decomposer {
    pub fn new(ring: Arc<PolyRing<IntegerRing>>, bound: Option<usize>) -> Decomposer {
        let n: usize = ring.ngens;
        Decomposer {
            ring,
            n,
            bound,
            empty: Chain::new(vec![None; n]),
            regularized: HashMap::new(),
            intersected: HashMap::new(),
            subresultant_cache: HashMap::new(),
            factor_cache: HashMap::new(),
            pool: Vec::new(),
            pool_index: HashMap::new(),
            point: (0..n).map(|i| nextprime(&Int::from(100 * (i + 1)))).collect(),
        }
    }

    // ------------------------------------------------------------------
    // elementary operations

    pub fn kept(&self, chain: &Chain) -> bool {
        self.bound.is_none_or_le(chain.height)
    }

    fn in_pool(&self, q: &ZPoly) -> bool {
        self.pool_index.contains_key(&key_of(q))
    }

    /// Distinct factors of `p` with the same zeros (only an optimization: a
    /// polynomial with large coefficients is divided by the known
    /// irreducible factors, and only a small cofactor is factored).
    pub fn factors(&mut self, p: &ZPoly) -> PolyResult<Vec<ZPoly>> {
        if let Some(found) = self.factor_cache.get(&key_of(p)) {
            return Ok(found.clone());
        }
        let found: Vec<ZPoly> = if p.is_ground() {
            Vec::new()
        } else if max_norm(p) < factor_norm() {
            self.irreducible_factors(p)?
        } else {
            let mut found: Vec<ZPoly> = Vec::new();
            let mut rest: ZPoly = p.clone();
            let mut value: Int = self.value(&rest);
            let pool: Vec<(ZPoly, Int)> = self.pool.clone();
            for (q, at) in &pool {
                if at.is_zero() || !mod_floor(&value, at).is_zero() {
                    continue;
                }
                let Some(mut quotient) = divided(&rest, q)? else {
                    continue;
                };
                found.push(q.clone());
                loop {
                    rest = quotient;
                    match divided(&rest, q)? {
                        Some(next) => quotient = next,
                        None => break,
                    }
                }
                value = self.value(&rest);
            }
            let rest: ZPoly = normalize(&rest);
            if !rest.is_ground() {
                // the content in the main variable is cheap to split off
                let primitive: ZPoly = primitive_part(&rest, main_variable(&rest))?;
                let mut parts: Vec<ZPoly> = vec![primitive.clone()];
                if primitive != rest {
                    parts.push(normalize(&exact_quotient(&rest, &primitive)?));
                }
                for part in parts {
                    if max_norm(&part) < factor_norm() || part != rest {
                        for q in self.factors(&part)? {
                            if !found.contains(&q) {
                                found.push(q);
                            }
                        }
                    } else {
                        found.push(part);
                    }
                }
            }
            found
        };
        self.factor_cache.insert(key_of(p), found.clone());
        Ok(found)
    }

    fn irreducible_factors(&mut self, p: &ZPoly) -> PolyResult<Vec<ZPoly>> {
        let found: Vec<ZPoly> = distinct_factors(p)?;
        for q in &found {
            if !self.in_pool(q) {
                let value: Int = self.value(q);
                self.pool_index.insert(key_of(q), self.pool.len());
                self.pool.push((q.clone(), value));
            }
        }
        Ok(found)
    }

    /// The value at a fixed integer point: a divisor divides there.
    fn value(&self, p: &ZPoly) -> Int {
        let mut value: Int = Int::ZERO;
        for (m, c) in &p.terms {
            let mut term: Int = c.clone();
            for (a, e) in self.point.iter().zip(m) {
                if *e != 0 {
                    term *= a.pow(*e as usize);
                }
            }
            value += term;
        }
        value
    }

    pub fn subresultants(&mut self, f: &ZPoly, g: &ZPoly, i: usize) -> PolyResult<Vec<ZPoly>> {
        let key = (key_of(f), key_of(g), i);
        if let Some(found) = self.subresultant_cache.get(&key) {
            return Ok(found.clone());
        }
        let found: Vec<ZPoly> = subresultant_chain(f, g, i)?;
        self.subresultant_cache.insert(key, found.clone());
        Ok(found)
    }

    /// The pseudo-remainder of `p` by the chain, normalized: zero if and only
    /// if `p` is in the saturated ideal.
    pub fn reduce(&self, p: &ZPoly, chain: &Chain) -> PolyResult<ZPoly> {
        let mut p: ZPoly = p.clone();
        for (i, t) in chain.polys.iter().enumerate() {
            if p.is_zero() {
                break;
            }
            if let Some(t) = t {
                if degree(&p, i) >= degree(t, i) {
                    p = pseudo_remainder(&p, t, i)?;
                }
            }
        }
        Ok(normalize(&p))
    }

    // ------------------------------------------------------------------
    // regularize

    /// Split the chain so that `p` is zero or regular modulo each part.
    pub fn regularize(&mut self, p: &ZPoly, chain: &Chain) -> PolyResult<Vec<Regularized>> {
        if p.is_ground() {
            return Ok(vec![(chain.clone(), p.is_zero())]);
        }
        let key = (key_of(p), chain.clone());
        if let Some(found) = self.regularized.get(&key) {
            return Ok(found.clone());
        }
        let r: ZPoly = self.reduce(p, chain)?;
        let result: Vec<Regularized> = if r.is_zero() {
            vec![(chain.clone(), true)]
        } else if r.is_ground() {
            vec![(chain.clone(), false)]
        } else {
            let mut result: Vec<Regularized> = Vec::new();
            let factors: Vec<ZPoly> = self.factors(&r)?;
            let undecided: Vec<(Chain, Option<bool>)> = if factors.len() == 1 {
                self.regularize_reduced(&factors[0], chain)?
            } else {
                // a product is zero where a factor is, regular where all are
                let mut undecided: Vec<(Chain, Option<bool>)> = Vec::new();
                let mut pending: Vec<Chain> = vec![chain.clone()];
                for f in &factors {
                    let mut following: Vec<Chain> = Vec::new();
                    for e in &pending {
                        for (e2, zero) in self.regularize(f, e)? {
                            if e2.height > chain.height {
                                undecided.push((e2, None));
                            } else if zero {
                                undecided.push((e2, Some(true)));
                            } else {
                                following.push(e2);
                            }
                        }
                    }
                    pending = following;
                }
                undecided.extend(pending.into_iter().map(|e| (e, Some(false))));
                undecided
            };
            for (e, status) in undecided {
                if !self.kept(&e) {
                    continue;
                }
                match status {
                    // a smaller dimension: nothing is inherited
                    None => result.extend(self.regularize(p, &e)?),
                    Some(s) => result.push((e, s)),
                }
            }
            result
        };
        self.regularized.insert(key, result.clone());
        Ok(result)
    }

    /// `regularize` for `f` reduced with respect to the chain and not
    /// constant; the chains of greater height come with `None`.
    fn regularize_reduced(&mut self, f: &ZPoly, chain: &Chain) -> PolyResult<Vec<(Chain, Option<bool>)>> {
        let v: usize = main_variable(f);
        let t: Option<ZPoly> = chain.get(v).cloned();
        let c: Chain = chain.below(v);
        let above: Vec<ZPoly> = chain.above(v);
        let top: Vec<ZPoly> = match &t {
            None => above.clone(),
            Some(t) => {
                let mut top: Vec<ZPoly> = vec![t.clone()];
                top.extend(above.iter().cloned());
                top
            }
        };
        let mut out: Vec<(Chain, Option<bool>)> = Vec::new();
        let a: ZPoly = initial(f);
        for (e, zero) in self.regularize(&a, &c)? {
            if e.height > c.height {
                out.extend(self.extend(&e, &top)?.into_iter().map(|t2| (t2, None)));
                continue;
            }
            if zero {
                let t2: Chain = e.union(&top);
                for (e2, zero2) in self.regularize(&tail(f), &t2)? {
                    let status: Option<bool> = if e2.height == chain.height { Some(zero2) } else { None };
                    out.push((e2, status));
                }
                continue;
            }
            let Some(t) = &t else {
                // a regular leading coefficient in a free variable
                out.push((e.union(&top), Some(false)));
                continue;
            };
            for (g, big_f, lost) in self.regular_gcd(t, f, v, &e)? {
                let Some(g) = g else {
                    out.extend(self.extend(&big_f, &top)?.into_iter().map(|t2| (t2, None)));
                    continue;
                };
                if degree(&g, v) == 0 {
                    out.push((big_f.union(&top), Some(false)));
                    continue;
                }
                let (_, q, rho) = pseudo_divide(t, &g, v)?;
                if !self.reduce(&rho, &big_f)?.is_zero() {
                    return Err(arithmetic_error(
                        "the regular gcd does not divide the polynomial of the chain".to_string(),
                    ));
                }
                let q: ZPoly = primitive_part(&self.reduce(&q, &big_f)?, v)?;
                out.push((big_f.with_poly(v, g).union(&above), Some(true)));
                // the chain is squarefree: f is invertible modulo the cofactor
                out.push((big_f.with_poly(v, q).union(&above), Some(false)));
                for f2 in self.intersect(&lost, &big_f)? {
                    out.extend(self.extend(&f2, &top)?.into_iter().map(|t2| (t2, None)));
                }
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // regular gcd

    /// Regular gcds of `p` and `q`, of main variable `v` and with initials
    /// regular modulo the chain, whose variables are smaller.
    pub fn regular_gcd(&mut self, p: &ZPoly, q: &ZPoly, v: usize, chain: &Chain) -> PolyResult<Vec<RegularGcd>> {
        let (p, q) = if degree(p, v) < degree(q, v) { (q, p) } else { (p, q) };
        let subresultants: Vec<ZPoly> = self.subresultants(p, q, v)?;
        let zero: ZPoly = self.ring.zero();
        let mut out: Vec<RegularGcd> = Vec::new();
        let mut pending: Vec<Chain> = vec![chain.clone()];
        for (j, s_poly) in subresultants.iter().enumerate() {
            let s: ZPoly = coefficient(s_poly, v, j as i64);
            let mut following: Vec<Chain> = Vec::new();
            for f in &pending {
                for (f2, vanishes) in self.regularize(&s, f)? {
                    if f2.height > chain.height {
                        out.push((None, f2, zero.clone()));
                    } else if vanishes {
                        following.push(f2);
                    } else {
                        let g: ZPoly = self.reduce(s_poly, &f2)?;
                        if degree(&g, v) != j as i64 {
                            return Err(arithmetic_error(
                                "a regular principal subresultant coefficient was reduced to zero".to_string(),
                            ));
                        }
                        let lead: ZPoly = coefficient(&g, v, j as i64);
                        out.push((Some(primitive_part(&g, v)?), f2, lead));
                    }
                }
            }
            pending = following;
            if pending.is_empty() {
                break;
            }
        }
        for f in &pending {
            let g: ZPoly = self.reduce(q, f)?;
            let d: i64 = degree(&g, v);
            if d != degree(q, v) {
                return Err(arithmetic_error("a regular initial was reduced to zero".to_string()));
            }
            let lead: ZPoly = coefficient(&g, v, d);
            out.push((Some(primitive_part(&g, v)?), f.clone(), lead));
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // building chains

    /// Put `f` on top of the chain: the chains with the squarefree part of
    /// `f`.
    pub fn add(&mut self, chain: &Chain, f: &ZPoly) -> PolyResult<Vec<Chain>> {
        let v: usize = main_variable(f);
        let g: ZPoly = self.reduce(f, chain)?;
        let d: i64 = degree(&g, v);
        if d != degree(f, v) {
            return Err(arithmetic_error("a regular initial was reduced to zero".to_string()));
        }
        let g: ZPoly = primitive_part(&g, v)?;
        if d == 1 {
            return Ok(vec![chain.with_poly(v, g)]);
        }
        if chain.height == 0 {
            let mut squarefree: ZPoly = self.ring.one();
            for part in self.factors(&g)? {
                let factor: ZPoly = if self.in_pool(&part) { part } else { sqf_part(&part)? };
                squarefree = squarefree.mul(&factor);
            }
            return Ok(vec![chain.with_poly(v, squarefree)]);
        }
        let mut out: Vec<Chain> = Vec::new();
        for (common, big_f, lost) in self.regular_gcd(&g, &g.diff(v), v, chain)? {
            let Some(common) = common else {
                out.extend(self.extend(&big_f, std::slice::from_ref(f))?);
                continue;
            };
            if degree(&common, v) == 0 {
                out.push(big_f.with_poly(v, g.clone()));
                continue;
            }
            let (_, q, rho) = pseudo_divide(&g, &common, v)?;
            if !self.reduce(&rho, &big_f)?.is_zero() {
                return Err(arithmetic_error(
                    "the regular gcd does not divide the polynomial".to_string(),
                ));
            }
            let reduced: ZPoly = primitive_part(&self.reduce(&q, &big_f)?, v)?;
            out.push(big_f.with_poly(v, reduced));
            for f2 in self.intersect(&lost, &big_f)? {
                out.extend(self.extend(&f2, std::slice::from_ref(f))?);
            }
        }
        Ok(out.into_iter().filter(|t| self.kept(t)).collect())
    }

    /// Put the polynomials, of increasing main variables greater than the
    /// ones of the chain, on top of it; the parts where an initial vanishes
    /// identically are dropped.
    pub fn extend(&mut self, chain: &Chain, polys: &[ZPoly]) -> PolyResult<Vec<Chain>> {
        let mut chains: Vec<Chain> = if self.kept(chain) {
            vec![chain.clone()]
        } else {
            Vec::new()
        };
        for f in polys {
            let a: ZPoly = initial(f);
            let mut following: Vec<Chain> = Vec::new();
            for big_a in &chains {
                for (a2, zero) in self.regularize(&a, big_a)? {
                    if !zero {
                        following.extend(self.add(&a2, f)?);
                    }
                }
            }
            chains = distinct(following);
        }
        Ok(chains)
    }

    // ------------------------------------------------------------------
    // intersect

    /// The zeros of `p` in the quasi-component of the chain.
    pub fn intersect(&mut self, p: &ZPoly, chain: &Chain) -> PolyResult<Vec<Chain>> {
        if p.is_zero() {
            return Ok(vec![chain.clone()]);
        }
        if p.is_ground() || !self.kept(chain) {
            return Ok(Vec::new());
        }
        let key = (key_of(p), chain.clone());
        if let Some(found) = self.intersected.get(&key) {
            return Ok(found.clone());
        }
        let mut out: Vec<Chain> = Vec::new();
        for f in self.factors(p)? {
            out.extend(self.intersect_with(&f, &f, chain)?);
        }
        let result: Vec<Chain> = distinct(out).into_iter().filter(|t| self.kept(t)).collect();
        self.intersected.insert(key, result.clone());
        Ok(result)
    }

    /// The zeros of `fr` in the quasi-component, where a regular multiple of
    /// `f` is a multiple of `fr` modulo the saturated ideal (`_intersect`).
    fn intersect_with(&mut self, f: &ZPoly, fr: &ZPoly, chain: &Chain) -> PolyResult<Vec<Chain>> {
        let mut out: Vec<Chain> = Vec::new();
        for (e, zero) in self.regularize(fr, chain)? {
            if e.height > chain.height {
                out.extend(self.intersect(f, &e)?);
            } else if zero {
                out.push(e);
            } else {
                let reduced: ZPoly = self.reduce(fr, &e)?;
                for h in self.factors(&reduced)? {
                    out.extend(self.intersect_regular(f, &h, &e)?);
                }
            }
        }
        Ok(out)
    }

    /// The same for `h` reduced, not constant and regular modulo the chain.
    fn intersect_regular(&mut self, f: &ZPoly, h: &ZPoly, chain: &Chain) -> PolyResult<Vec<Chain>> {
        let u: usize = main_variable(h);
        let t: Option<ZPoly> = chain.get(u).cloned();
        let c: Chain = chain.below(u);
        let above: Vec<ZPoly> = chain.above(u);
        let mut out: Vec<Chain> = Vec::new();
        if let Some(t) = t {
            // common zeros of h and t lie over the zeros of their resultant
            let resultant: ZPoly = self.subresultants(&t, h, u)?[0].clone();
            if resultant.is_zero() {
                return Err(arithmetic_error(
                    "a regular polynomial has a common factor with the chain".to_string(),
                ));
            }
            let mut top: Vec<ZPoly> = vec![t.clone()];
            top.extend(above.iter().cloned());
            for e in self.intersect(&resultant, &c)? {
                for t2 in self.extend(&e, &top)? {
                    out.extend(self.intersect(f, &t2)?);
                }
            }
            return Ok(out);
        }
        let a: ZPoly = initial(h);
        if !a.is_ground() {
            // where the initial vanishes
            let rest: ZPoly = tail(h);
            for t2 in self.intersect(&a, chain)? {
                if t2.height > chain.height {
                    out.extend(self.intersect(f, &t2)?);
                } else {
                    out.extend(self.intersect_with(f, &rest, &t2)?);
                }
            }
        }
        let mut with_h: Vec<ZPoly> = vec![h.clone()];
        with_h.extend(above.iter().cloned());
        for (e, zero) in self.regularize(&a, &c)? {
            if zero {
                continue;
            }
            if e.height > c.height {
                for r in self.extend(&e, &with_h)? {
                    out.extend(self.intersect(f, &r)?);
                }
                continue;
            }
            for big_a in self.add(&e, h)? {
                for r in self.extend(&big_a, &above)? {
                    if r.height == chain.height + 1 {
                        out.push(r);
                    } else {
                        out.extend(self.intersect(f, &r)?);
                    }
                }
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // decompositions

    /// Whether `p` is regular modulo the saturated ideal of the chain.
    pub fn is_regular(&mut self, p: &ZPoly, chain: &Chain) -> PolyResult<bool> {
        Ok(self
            .regularize(p, chain)?
            .iter()
            .all(|(e, zero)| e.height != chain.height || !zero))
    }

    /// A sufficient condition for the quasi-component of `first` to be
    /// contained in the one of `second` (in its closure with `closures`).
    pub fn inside(&mut self, first: &Chain, second: &Chain, closures: bool) -> PolyResult<bool> {
        if second.height > first.height {
            return Ok(false);
        }
        let elements: Vec<ZPoly> = second.elements();
        for p in &elements {
            if !self.reduce(p, first)?.is_zero() {
                return Ok(false);
            }
        }
        for p in &elements {
            let a: ZPoly = initial(p);
            if a.is_ground() {
                continue;
            }
            if closures {
                if !self.is_regular(&a, first)? {
                    return Ok(false);
                }
            } else if !self.intersect(&a, first)?.is_empty() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Remove the chains which [`Decomposer::inside`] finds in another one
    /// (the chains told apart by position, as Python by identity).
    pub fn irredundant(&mut self, chains: &[Chain], closures: bool) -> PolyResult<Vec<Chain>> {
        let mut kept: Vec<usize> = (0..chains.len()).collect();
        let mut order: Vec<usize> = (0..chains.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(chains[i].height));
        for i in order {
            let mut redundant: bool = false;
            for &other in &kept.clone() {
                if other != i && self.inside(&chains[i], &chains[other], closures)? {
                    redundant = true;
                    break;
                }
            }
            if redundant {
                kept.retain(|&k| k != i);
            }
        }
        Ok(kept.into_iter().map(|i| chains[i].clone()).collect())
    }

    /// Make the chains without free variables disjoint.
    pub fn separate_points(&mut self, chains: &[Chain]) -> PolyResult<Vec<Chain>> {
        let mut result: Vec<Chain> = Vec::new();
        for chain in chains {
            let mut parts: Vec<Chain> = vec![chain.clone()];
            if chain.height == self.n {
                for other in result.clone() {
                    if other.height == self.n {
                        let mut next: Vec<Chain> = Vec::new();
                        for part in &parts {
                            next.extend(self.outside(part, &other)?);
                        }
                        parts = next;
                    }
                }
            }
            result.extend(parts);
        }
        Ok(result)
    }

    fn outside(&mut self, chain: &Chain, other: &Chain) -> PolyResult<Vec<Chain>> {
        let mut out: Vec<Chain> = Vec::new();
        let mut pending: Vec<Chain> = vec![chain.clone()];
        for p in other.elements() {
            let mut following: Vec<Chain> = Vec::new();
            for part in &pending {
                for (e, zero) in self.regularize(&p, part)? {
                    if zero {
                        following.push(e);
                    } else {
                        out.push(e);
                    }
                }
            }
            pending = following;
        }
        Ok(out)
    }

    /// Chains whose quasi-components cover the zeros of the equations where
    /// no inequation vanishes (in the Kalkbrener mode, the closures cover
    /// them); every inequation is regular modulo every chain.
    pub fn triangularize(&mut self, equations: &[ZPoly], inequations: &[ZPoly]) -> PolyResult<Vec<Chain>> {
        let mut polys: Vec<ZPoly> = equations.iter().filter(|p| !p.is_zero()).map(normalize).collect();
        polys.sort_by_cached_key(|p| (std::cmp::Reverse(main_variable(p)), p.terms.len(), sort_key(p)));
        let mut chains: Vec<Chain> = vec![self.empty.clone()];
        for p in &polys {
            let mut next: Vec<Chain> = Vec::new();
            for t in &chains {
                next.extend(self.intersect(p, t)?);
            }
            chains = self.irredundant(&distinct(next), false)?;
        }
        for h in inequations {
            let mut next: Vec<Chain> = Vec::new();
            for t in &chains {
                for (e, zero) in self.regularize(h, t)? {
                    if !zero {
                        next.push(e);
                    }
                }
            }
            chains = distinct(next);
        }
        if self.bound.is_some() {
            chains = self.irredundant(&chains, true)?;
        }
        chains.sort_by_cached_key(order_key);
        self.separate_points(&chains)
    }
}

/// The greatest dimension first, then the greatest degree (`_order`).
fn order_key(chain: &Chain) -> (usize, std::cmp::Reverse<Int>, Vec<(usize, String)>) {
    let elements: Vec<ZPoly> = chain.elements();
    let mut total: Int = Int::ONE;
    for p in &elements {
        total *= Int::from(degree(p, main_variable(p)));
    }
    (
        chain.height,
        std::cmp::Reverse(total),
        elements.iter().map(|p| (p.terms.len(), p.to_string())).collect(),
    )
}

/// `bound is None or height <= bound`.
trait BoundCheck {
    fn is_none_or_le(&self, height: usize) -> bool;
}

impl BoundCheck for Option<usize> {
    fn is_none_or_le(&self, height: usize) -> bool {
        match self {
            None => true,
            Some(b) => height <= *b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::ZZ;
    use crate::polys::rings::MonomialOrder;

    #[test]
    fn triangularize_small_systems() {
        let ring = PolyRing::new(ZZ, vec!["x".into(), "y".into()], MonomialOrder::Lex).unwrap();
        let g = ring.gens();
        let (x, y) = (&g[0], &g[1]);
        let mut d = Decomposer::new(Arc::clone(&ring), None);
        // x*y = 0, y**2 = y: three components
        let chains = d.triangularize(&[x.mul(y), y.mul(y).sub(y)], &[]).unwrap();
        assert!(!chains.is_empty());
        for chain in &chains {
            for p in [x.mul(y), y.mul(y).sub(y)] {
                assert!(d.reduce(&p, chain).unwrap().is_zero());
            }
        }
    }
}
