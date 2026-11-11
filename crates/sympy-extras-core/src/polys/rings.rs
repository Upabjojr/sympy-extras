//! Port of `sympy_extras/polys/rings.py`: sparse multivariate polynomials
//! over a typed domain, the ring `K[x_1, ..., x_n]` with a monomial order,
//! elements as maps from exponent vectors to nonzero coefficients.
//!
//! This is the representation of the Gröbner basis algorithms: leading
//! terms under the order, division with remainder by a list of polynomials
//! (the multivariate reduction), and the arithmetic.
//!
//! The Python orders are key functions on exponent tuples (`lex`, `grlex`,
//! `grevlex`, and the `WeightOrder`, `BlockOrder` of `orderings.py` and the
//! homogeneous order of `modulargroebner.py`); here they are the closed
//! enum [`MonomialOrder`], whose [`MonomialOrder::cmp`] compares two
//! monomials as Python compares their keys, without building the keys.
//!
//! The conversions to and from expressions (`from_expr`, `to_expr`) wait for
//! the port of the typed core.
//!
//! References
//! ==========
//!
//! 1. D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*, 4th
//!    ed., Springer, 2015, Chapter 2 (monomial orders, the division
//!    algorithm).

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::domains::Domain;

/// The exponents of a monomial (Python's `tuple[int, ...]`).
pub type Monomial = Vec<u32>;

/// The terms of a polynomial: monomial -> nonzero coefficient.
pub type Terms<T> = BTreeMap<Monomial, T>;

pub fn monomial_mul(a: &[u32], b: &[u32]) -> Monomial {
    a.iter().zip(b).map(|(i, j)| i + j).collect()
}

/// `a/b` (`Value` when `b` does not divide `a`).
pub fn monomial_div(a: &[u32], b: &[u32]) -> PolyResult<Monomial> {
    if a.iter().zip(b).any(|(i, j)| i < j) {
        return Err(value_error(format!("{:?} does not divide {:?}", b, a)));
    }
    Ok(a.iter().zip(b).map(|(i, j)| i - j).collect())
}

/// Whether `a` divides `b`.
pub fn monomial_divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(i, j)| i <= j)
}

pub fn monomial_lcm(a: &[u32], b: &[u32]) -> Monomial {
    a.iter().zip(b).map(|(i, j)| *i.max(j)).collect()
}

pub fn monomial_gcd(a: &[u32], b: &[u32]) -> Monomial {
    a.iter().zip(b).map(|(i, j)| *i.min(j)).collect()
}

fn total_degree(m: &[u32]) -> u64 {
    m.iter().map(|e| *e as u64).sum()
}

/// A monomial order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MonomialOrder {
    /// Lexicographic.
    Lex,
    /// Total degree, then lexicographic.
    Grlex,
    /// Total degree, then reverse lexicographic.
    Grevlex,
    /// `orderings.WeightOrder`: the weighted degree `sum(w_i e_i)` (weights
    /// nonnegative), ties broken by `tail`.
    Weight {
        weights: Vec<Int>,
        tail: Box<MonomialOrder>,
    },
    /// `orderings.BlockOrder`: consecutive blocks of variables, each with
    /// its own order, the first block first.
    Block(Vec<(MonomialOrder, usize)>),
    /// The homogeneous order of `modulargroebner.py`: total degree first,
    /// then `base` on the first `nvars` exponents.
    Homogeneous { base: Box<MonomialOrder>, nvars: usize },
}

impl MonomialOrder {
    /// `orderings.WeightOrder(weights, tail)`; `Value` for a negative weight.
    pub fn weight(weights: Vec<Int>, tail: MonomialOrder) -> PolyResult<MonomialOrder> {
        if weights.iter().any(is_negative) {
            return Err(value_error("the weights must be non-negative".to_string()));
        }
        Ok(MonomialOrder::Weight {
            weights,
            tail: Box::new(tail),
        })
    }

    /// `orderings.BlockOrder(blocks)`; `Value` for an empty block.
    pub fn block(blocks: Vec<(MonomialOrder, usize)>) -> PolyResult<MonomialOrder> {
        if blocks.iter().any(|(_, size)| *size == 0) {
            return Err(value_error("block sizes must be positive".to_string()));
        }
        Ok(MonomialOrder::Block(blocks))
    }

    /// `orderings.elimination_order(neliminate, nvars, order)`: the block
    /// order eliminating the first `neliminate` of `nvars` variables.
    pub fn elimination(neliminate: usize, nvars: usize, order: MonomialOrder) -> MonomialOrder {
        if neliminate == 0 || neliminate >= nvars {
            return order;
        }
        MonomialOrder::Block(vec![(order.clone(), neliminate), (order, nvars - neliminate)])
    }

    /// How `a` compares with `b` (a larger monomial is `Greater`), as the
    /// Python keys compare.
    pub fn cmp(&self, a: &[u32], b: &[u32]) -> Ordering {
        match self {
            MonomialOrder::Lex => a.cmp(b),
            MonomialOrder::Grlex => total_degree(a).cmp(&total_degree(b)).then_with(|| a.cmp(b)),
            MonomialOrder::Grevlex => total_degree(a).cmp(&total_degree(b)).then_with(|| {
                // the key is (-e for e in reversed(m)): the last differing
                // exponent decides, the smaller one being the larger monomial
                for (x, y) in a.iter().rev().zip(b.iter().rev()) {
                    match y.cmp(x) {
                        Ordering::Equal => continue,
                        other => return other,
                    }
                }
                Ordering::Equal
            }),
            MonomialOrder::Weight { weights, tail } => {
                let weighted = |m: &[u32]| -> Int {
                    let mut total: Int = Int::ZERO;
                    for (w, e) in weights.iter().zip(m) {
                        total += w * Int::from(*e);
                    }
                    total
                };
                weighted(a).cmp(&weighted(b)).then_with(|| tail.cmp(a, b))
            }
            MonomialOrder::Block(blocks) => {
                let mut start: usize = 0;
                for (order, size) in blocks {
                    let end: usize = start + size;
                    match order.cmp(&a[start..end], &b[start..end]) {
                        Ordering::Equal => start = end,
                        other => return other,
                    }
                }
                Ordering::Equal
            }
            MonomialOrder::Homogeneous { base, nvars } => total_degree(a)
                .cmp(&total_degree(b))
                .then_with(|| base.cmp(&a[..*nvars], &b[..*nvars])),
        }
    }

    /// Whether the squares of the variables are above the variables (the
    /// total degree comes first), the test of `interop.typed_order`.
    pub fn is_graded(&self, ngens: usize) -> bool {
        match self {
            MonomialOrder::Lex => false,
            MonomialOrder::Grlex | MonomialOrder::Grevlex | MonomialOrder::Homogeneous { .. } => true,
            _ => {
                let unit = |i: usize, e: u32| -> Monomial { (0..ngens).map(|k| if k == i { e } else { 0 }).collect() };
                (0..ngens).all(|i| (0..ngens).all(|j| self.cmp(&unit(i, 2), &unit(j, 1)) == Ordering::Greater))
            }
        }
    }
}

impl fmt::Display for MonomialOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MonomialOrder::Lex => write!(f, "lex"),
            MonomialOrder::Grlex => write!(f, "grlex"),
            MonomialOrder::Grevlex => write!(f, "grevlex"),
            MonomialOrder::Weight { weights, tail } => {
                let w: Vec<String> = weights.iter().map(|w| w.to_string()).collect();
                if w.len() == 1 {
                    write!(f, "WeightOrder(({},), {})", w[0], tail)
                } else {
                    write!(f, "WeightOrder(({}), {})", w.join(", "), tail)
                }
            }
            MonomialOrder::Block(blocks) => {
                let parts: Vec<String> = blocks.iter().map(|(o, s)| format!("({}, {})", o, s)).collect();
                write!(f, "BlockOrder({})", parts.join(", "))
            }
            MonomialOrder::Homogeneous { base, nvars } => write!(f, "HomogeneousOrder({}, {})", base, nvars),
        }
    }
}

/// The polynomial ring over `domain` in the variables `names` with the
/// monomial order `order`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyRing<K: Domain> {
    pub domain: K,
    pub names: Vec<String>,
    pub order: MonomialOrder,
    pub ngens: usize,
    pub zero_monom: Monomial,
}

impl<K: Domain> PolyRing<K> {
    /// `Value` for a ring without variables.
    pub fn new(domain: K, names: Vec<String>, order: MonomialOrder) -> PolyResult<Arc<PolyRing<K>>> {
        if names.is_empty() {
            return Err(value_error("a polynomial ring needs at least one variable".to_string()));
        }
        let ngens: usize = names.len();
        Ok(Arc::new(PolyRing {
            domain,
            names,
            order,
            ngens,
            zero_monom: vec![0; ngens],
        }))
    }

    /// `K[x,y]`.
    pub fn name(&self) -> String {
        format!("{}[{}]", self.domain.name(), self.names.join(","))
    }

    pub fn zero(self: &Arc<Self>) -> PolyElement<K> {
        PolyElement {
            terms: Terms::new(),
            ring: Arc::clone(self),
        }
    }

    pub fn one(self: &Arc<Self>) -> PolyElement<K> {
        self.ground(self.domain.one())
    }

    pub fn gens(self: &Arc<Self>) -> Vec<PolyElement<K>> {
        (0..self.ngens)
            .map(|i| {
                let monomial: Monomial = (0..self.ngens).map(|j| if j == i { 1 } else { 0 }).collect();
                self.term(monomial, self.domain.one())
            })
            .collect()
    }

    pub fn term(self: &Arc<Self>, monomial: Monomial, coefficient: K::Elem) -> PolyElement<K> {
        let mut terms: Terms<K::Elem> = Terms::new();
        if !self.domain.is_zero(&coefficient) {
            terms.insert(monomial, coefficient);
        }
        PolyElement {
            terms,
            ring: Arc::clone(self),
        }
    }

    pub fn ground(self: &Arc<Self>, c: K::Elem) -> PolyElement<K> {
        self.term(self.zero_monom.clone(), c)
    }

    pub fn from_int(self: &Arc<Self>, n: i64) -> PolyElement<K> {
        self.ground(self.domain.from_int(n))
    }

    /// The polynomial of the terms, the zero coefficients dropped.
    pub fn from_dict(self: &Arc<Self>, terms: Terms<K::Elem>) -> PolyElement<K> {
        let terms: Terms<K::Elem> = terms.into_iter().filter(|(_, c)| !self.domain.is_zero(c)).collect();
        PolyElement {
            terms,
            ring: Arc::clone(self),
        }
    }

    pub fn with_order(&self, order: MonomialOrder) -> Arc<PolyRing<K>> {
        Arc::new(PolyRing { order, ..self.clone() })
    }
}

/// A sparse polynomial: `terms` maps exponent vectors to nonzero
/// coefficients of the ring's domain.
#[derive(Debug, Clone)]
pub struct PolyElement<K: Domain> {
    pub terms: Terms<K::Elem>,
    pub ring: Arc<PolyRing<K>>,
}

impl<K: Domain> PartialEq for PolyElement<K> {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms && (Arc::ptr_eq(&self.ring, &other.ring) || self.ring == other.ring)
    }
}

impl<K: Domain> fmt::Display for PolyElement<K> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(out, "0");
        }
        let k: &K = &self.ring.domain;
        for (i, m) in self.monomials().iter().enumerate() {
            let c: &K::Elem = &self.terms[m];
            let negative: bool = k.is_ordered() && k.is_negative(c);
            let magnitude: K::Elem = if negative { k.neg(c) } else { c.clone() };
            let factors: Vec<String> = self
                .ring
                .names
                .iter()
                .zip(m)
                .filter(|(_, e)| **e > 0)
                .map(|(name, e)| {
                    if *e > 1 {
                        format!("{}**{}", name, e)
                    } else {
                        name.clone()
                    }
                })
                .collect();
            let body: String = if !factors.is_empty() && k.is_one(&magnitude) {
                factors.join("*")
            } else {
                let mut parts: Vec<String> = vec![k.to_str(&magnitude)];
                parts.extend(factors);
                parts.join("*")
            };
            if i == 0 {
                write!(out, "{}{}", if negative { "-" } else { "" }, body)?;
            } else {
                write!(out, "{}{}", if negative { " - " } else { " + " }, body)?;
            }
        }
        Ok(())
    }
}

impl<K: Domain> PolyElement<K> {
    fn with_terms(&self, terms: Terms<K::Elem>) -> PolyElement<K> {
        PolyElement {
            terms,
            ring: Arc::clone(&self.ring),
        }
    }

    fn domain(&self) -> &K {
        &self.ring.domain
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    // -- leading data

    /// The monomials in decreasing order.
    pub fn monomials(&self) -> Vec<Monomial> {
        let mut monomials: Vec<Monomial> = self.terms.keys().cloned().collect();
        let order: &MonomialOrder = &self.ring.order;
        monomials.sort_by(|a, b| order.cmp(b, a));
        monomials
    }

    /// The terms in decreasing order of their monomials.
    pub fn ordered_terms(&self) -> Vec<(Monomial, K::Elem)> {
        self.monomials()
            .into_iter()
            .map(|m| {
                let c: K::Elem = self.terms[&m].clone();
                (m, c)
            })
            .collect()
    }

    /// The leading monomial (the zero monomial for the zero polynomial).
    #[allow(non_snake_case)]
    pub fn LM(&self) -> Monomial {
        let order: &MonomialOrder = &self.ring.order;
        match self.terms.keys().max_by(|a, b| order.cmp(a, b)) {
            Some(m) => m.clone(),
            None => self.ring.zero_monom.clone(),
        }
    }

    /// The leading coefficient (zero for the zero polynomial).
    #[allow(non_snake_case)]
    pub fn LC(&self) -> K::Elem {
        let order: &MonomialOrder = &self.ring.order;
        match self.terms.iter().max_by(|a, b| order.cmp(a.0, b.0)) {
            Some((_, c)) => c.clone(),
            None => self.domain().zero(),
        }
    }

    /// The leading term.
    #[allow(non_snake_case)]
    pub fn LT(&self) -> (Monomial, K::Elem) {
        let m: Monomial = self.LM();
        let c: K::Elem = self.coeff(&m);
        (m, c)
    }

    /// The total degree (`-1` for zero).
    pub fn degree(&self) -> i64 {
        self.terms.keys().map(|m| total_degree(m) as i64).max().unwrap_or(-1)
    }

    pub fn degree_in(&self, i: usize) -> i64 {
        self.terms.keys().map(|m| m[i] as i64).max().unwrap_or(-1)
    }

    pub fn is_ground(&self) -> bool {
        self.terms.is_empty() || (self.terms.len() == 1 && self.terms.contains_key(&self.ring.zero_monom))
    }

    pub fn is_monic(&self) -> bool {
        !self.terms.is_empty() && self.domain().is_one(&self.LC())
    }

    pub fn coeff(&self, monomial: &[u32]) -> K::Elem {
        match self.terms.get(monomial) {
            Some(c) => c.clone(),
            None => self.domain().zero(),
        }
    }

    // -- arithmetic

    pub fn neg(&self) -> PolyElement<K> {
        let k: &K = self.domain();
        self.with_terms(self.terms.iter().map(|(m, c)| (m.clone(), k.neg(c))).collect())
    }

    fn add_into(k: &K, result: &mut Terms<K::Elem>, m: Monomial, c: K::Elem) {
        match result.get(&m) {
            Some(existing) => {
                let s: K::Elem = k.add(existing, &c);
                if k.is_zero(&s) {
                    result.remove(&m);
                } else {
                    result.insert(m, s);
                }
            }
            None => {
                result.insert(m, c);
            }
        }
    }

    pub fn add(&self, other: &PolyElement<K>) -> PolyElement<K> {
        let k: &K = self.domain();
        let mut result: Terms<K::Elem> = self.terms.clone();
        for (m, c) in &other.terms {
            Self::add_into(k, &mut result, m.clone(), c.clone());
        }
        self.with_terms(result)
    }

    pub fn sub(&self, other: &PolyElement<K>) -> PolyElement<K> {
        let k: &K = self.domain();
        let mut result: Terms<K::Elem> = self.terms.clone();
        for (m, c) in &other.terms {
            Self::add_into(k, &mut result, m.clone(), k.neg(c));
        }
        self.with_terms(result)
    }

    pub fn mul(&self, other: &PolyElement<K>) -> PolyElement<K> {
        let k: &K = self.domain();
        let mut result: Terms<K::Elem> = Terms::new();
        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                Self::add_into(k, &mut result, monomial_mul(m1, m2), k.mul(c1, c2));
            }
        }
        self.with_terms(result)
    }

    /// `self**n` by repeated squaring.
    pub fn pow(&self, n: u64) -> PolyElement<K> {
        let mut n: u64 = n;
        let mut result: PolyElement<K> = self.ring.one();
        let mut base: PolyElement<K> = self.clone();
        while n != 0 {
            if n & 1 == 1 {
                result = result.mul(&base);
            }
            n >>= 1;
            if n != 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    pub fn mul_term(&self, monomial: &[u32], coefficient: &K::Elem) -> PolyElement<K> {
        let k: &K = self.domain();
        if k.is_zero(coefficient) {
            return self.ring.zero();
        }
        // the coefficients of an integral domain stay nonzero
        let terms: Terms<K::Elem> = self
            .terms
            .iter()
            .map(|(m, c)| (monomial_mul(m, monomial), k.mul(c, coefficient)))
            .filter(|(_, c)| !k.is_zero(c))
            .collect();
        self.with_terms(terms)
    }

    pub fn mul_ground(&self, coefficient: &K::Elem) -> PolyElement<K> {
        let zero: Monomial = self.ring.zero_monom.clone();
        self.mul_term(&zero, coefficient)
    }

    pub fn mul_monom(&self, monomial: &[u32]) -> PolyElement<K> {
        self.with_terms(
            self.terms
                .iter()
                .map(|(m, c)| (monomial_mul(m, monomial), c.clone()))
                .collect(),
        )
    }

    /// The coefficients divided exactly by `coefficient`.
    pub fn exquo_ground(&self, coefficient: &K::Elem) -> PolyResult<PolyElement<K>> {
        let k: &K = self.domain();
        let mut terms: Terms<K::Elem> = Terms::new();
        for (m, c) in &self.terms {
            terms.insert(m.clone(), k.exquo(c, coefficient)?);
        }
        Ok(self.with_terms(terms))
    }

    /// `self` divided by its leading coefficient (a field).
    pub fn monic(&self) -> PolyResult<PolyElement<K>> {
        if self.terms.is_empty() {
            return Ok(self.clone());
        }
        let inverse: K::Elem = self.domain().invert(&self.LC())?;
        Ok(self.mul_ground(&inverse))
    }

    /// The content (gcd of the coefficients) and the primitive part.
    pub fn primitive(&self) -> (K::Elem, PolyElement<K>) {
        let k: &K = self.domain();
        let mut content: K::Elem = k.zero();
        for c in self.terms.values() {
            content = k.gcd(&content, c);
        }
        if k.is_zero(&content) || k.is_one(&content) {
            return (content, self.clone());
        }
        let part: PolyElement<K> = self
            .exquo_ground(&content)
            .expect("the content divides the coefficients");
        (content, part)
    }

    /// The division algorithm: quotients `q_i` and a remainder `r` with
    /// `self = sum q_i g_i + r`, no term of `r` divisible by a leading
    /// monomial of the divisors (the leading coefficients must divide
    /// exactly: a field, or divisors with unit leading coefficients; `Value`
    /// otherwise).
    pub fn div(&self, divisors: &[PolyElement<K>]) -> PolyResult<(Vec<PolyElement<K>>, PolyElement<K>)> {
        let k: &K = self.domain();
        let mut quotients: Vec<PolyElement<K>> = divisors.iter().map(|_| self.ring.zero()).collect();
        let mut remainder: Terms<K::Elem> = Terms::new();
        let mut p: PolyElement<K> = self.clone();
        let leading: Vec<(Monomial, K::Elem)> = divisors.iter().map(|g| (g.LM(), g.LC())).collect();
        while !p.terms.is_empty() {
            let (m, c) = p.LT();
            let mut divided: bool = false;
            for (i, (lm, lc)) in leading.iter().enumerate() {
                if !divisors[i].terms.is_empty() && monomial_divides(lm, &m) {
                    let q_monomial: Monomial = monomial_div(&m, lm)?;
                    let q_coefficient: K::Elem = k.exquo(&c, lc)?;
                    quotients[i] = quotients[i].add(&self.ring.term(q_monomial.clone(), q_coefficient.clone()));
                    p = p.sub(&divisors[i].mul_term(&q_monomial, &q_coefficient));
                    divided = true;
                    break;
                }
            }
            if !divided {
                p.terms.remove(&m);
                remainder.insert(m, c);
            }
        }
        Ok((quotients, self.with_terms(remainder)))
    }

    pub fn rem(&self, divisors: &[PolyElement<K>]) -> PolyResult<PolyElement<K>> {
        Ok(self.div(divisors)?.1)
    }

    /// The exact quotient by one polynomial; `Value` otherwise.
    pub fn exquo(&self, other: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
        let (mut quotients, remainder) = self.div(std::slice::from_ref(other))?;
        if !remainder.terms.is_empty() {
            return Err(value_error(format!("{} does not divide {}", other, self)));
        }
        Ok(quotients.remove(0))
    }

    // -- calculus and evaluation

    /// The derivative with respect to the `i`-th variable.
    pub fn diff(&self, i: usize) -> PolyElement<K> {
        let k: &K = self.domain();
        let mut result: Terms<K::Elem> = Terms::new();
        for (m, c) in &self.terms {
            if m[i] > 0 {
                let d: K::Elem = k.mul(c, &k.from_int(m[i] as i64));
                if !k.is_zero(&d) {
                    let mut shifted: Monomial = m.clone();
                    shifted[i] -= 1;
                    result.insert(shifted, d);
                }
            }
        }
        self.with_terms(result)
    }

    /// The polynomial with the `i`-th variable set to `a` (still an element
    /// of the same ring, constant in that variable).
    pub fn evaluate(&self, i: usize, a: &K::Elem) -> PolyElement<K> {
        let k: &K = self.domain();
        let mut result: PolyElement<K> = self.ring.zero();
        for (m, c) in &self.terms {
            let value: K::Elem = k.mul(c, &k.pow(a, m[i] as u64));
            let mut monomial: Monomial = m.clone();
            monomial[i] = 0;
            result = result.add(&self.ring.term(monomial, value));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::{QQ, ZZ};

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn doctests() {
        let r = PolyRing::new(QQ, names(&["x", "y"]), MonomialOrder::Grevlex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let f = x.pow(2).mul(y).sub(&r.one());
        assert_eq!(f.LM(), vec![2, 1]);
        assert_eq!(f.LC(), rat(1, 1));
        assert_eq!(f.degree(), 3);
        assert_eq!(f.to_string(), "x**2*y - 1");
        let lhs = x.add(y).pow(2);
        let rhs = x.pow(2).add(&x.mul(y).mul_ground(&QQ.from_int(2))).add(&y.pow(2));
        assert_eq!(lhs, rhs);
        let (q, rem) = x.pow(2).mul(y).add(x).div(&[x.mul(y).sub(&r.one())]).unwrap();
        assert_eq!(
            (q[0].to_string(), rem.to_string()),
            ("x".to_string(), "2*x".to_string())
        );
    }

    #[test]
    fn orders_compare_as_the_python_keys() {
        // grevlex: x*z**2 > y**3? degrees 3 = 3; reversed negated keys:
        // x*z**2 -> (-2, 0, -1), y**3 -> (0, -3, 0): y**3 is larger
        assert_eq!(MonomialOrder::Grevlex.cmp(&[1, 0, 2], &[0, 3, 0]), Ordering::Less);
        assert_eq!(MonomialOrder::Grlex.cmp(&[1, 0, 2], &[0, 3, 0]), Ordering::Greater);
        assert_eq!(MonomialOrder::Lex.cmp(&[0, 5], &[1, 0]), Ordering::Less);
        let w = MonomialOrder::weight(vec![int(1), int(3)], MonomialOrder::Lex).unwrap();
        assert_eq!(w.cmp(&[3, 0], &[0, 1]), Ordering::Greater);
        assert_eq!(w.cmp(&[2, 0], &[0, 1]), Ordering::Less);
        assert_eq!(w.to_string(), "WeightOrder((1, 3), lex)");
        let b = MonomialOrder::block(vec![(MonomialOrder::Grevlex, 1), (MonomialOrder::Grevlex, 2)]).unwrap();
        assert_eq!(b.cmp(&[1, 0, 0], &[0, 5, 5]), Ordering::Greater);
        assert_eq!(b.to_string(), "BlockOrder((grevlex, 1), (grevlex, 2))");
        assert!(!b.is_graded(3) && !w.is_graded(2) && !MonomialOrder::Lex.is_graded(2));
        let graded_weight = MonomialOrder::weight(vec![int(1), int(1)], MonomialOrder::Lex).unwrap();
        assert!(graded_weight.is_graded(2));
        assert!(MonomialOrder::weight(vec![int(-1)], MonomialOrder::Lex).is_err());
    }

    #[test]
    fn integer_arithmetic_and_exact_division() {
        let r = PolyRing::new(ZZ, names(&["x", "y"]), MonomialOrder::Lex).unwrap();
        let g = r.gens();
        let f = g[0].add(&g[1]).mul(&g[0].mul_ground(&int(2)).sub(&g[1]));
        assert_eq!(f.exquo(&g[0].add(&g[1])).unwrap().to_string(), "2*x - y");
        assert!(f.exquo(&g[0].mul_ground(&int(3)).add(&g[1])).is_err());
        assert_eq!(f.diff(0).to_string(), "4*x + y");
        assert_eq!(f.evaluate(1, &int(1)).to_string(), "2*x**2 + x - 1");
        assert_eq!(f.primitive().0, int(1));
        assert_eq!(f.mul_ground(&int(6)).primitive().0, int(6));
    }
}
