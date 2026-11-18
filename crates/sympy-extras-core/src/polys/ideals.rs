//! Ideals of polynomial rings over a field, ported from
//! `sympy_extras.polys.ideals`: [`Ideal`] with its Gröbner bases (cached
//! per order; `lex` and `grlex` bases of zero-dimensional ideals by FGLM
//! from the `grevlex` one), membership, the arithmetic (sums, products,
//! powers, intersections, quotients, saturations), elimination, the
//! Hilbert series, polynomial, dimension and degree, the quotient algebra
//! of a zero-dimensional ideal (standard monomials, multiplication
//! matrices, minimal polynomials), the radical in dimension zero, the
//! maximality test and the change of order (FGLM or the Gröbner walk).
//!
//! The decompositions of `idealdecomposition` (the radical in positive
//! dimension, minimal and associated primes, primary decompositions) are
//! reached over the rationals by downcasting (`NotImplemented` over the
//! other fields, as in the Python), and so is the rational univariate
//! representation of `rur`. The Gröbner bases are computed by F4 directly (the
//! Python falls back on the modular algorithm after a time limit; the
//! reduced basis is the same).

#![allow(clippy::mutable_key_type)]

use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;

use crate::core::basic::{Basic, Head};
use crate::core::symbol::{Dummy, Symbol};
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::{Int, Rat};

use super::domains::{Domain, RationalField};
use super::rings::{
    monomial_div, monomial_divides, monomial_mul, Monomial, MonomialOrder, PolyElement, PolyRing, Terms,
};

fn is_graded(order: &MonomialOrder) -> bool {
    matches!(order, MonomialOrder::Grevlex | MonomialOrder::Grlex)
}

fn not_implemented(message: &str) -> PolyError {
    PolyError::NotImplemented(message.to_string())
}

// ----------------------------------------------------------------------
// separating forms

/// The coefficients `(c_2, ..., c_n)` of the linear forms of
/// [`separating_forms`] in `nvars` variables, in the same order (nothing
/// for fewer than two variables).
pub fn separating_coefficients(nvars: usize, points: usize) -> Vec<Vec<i64>> {
    let mut result: Vec<Vec<i64>> = Vec::new();
    if nvars < 2 {
        return result;
    }
    let pairs: usize = points * points.saturating_sub(1) / 2;
    let bound: i64 = pairs.div_ceil(2) as i64;
    for height in 0..=bound {
        let mut values: Vec<i64> = vec![0];
        for k in 1..=height {
            values.push(k);
            values.push(-k);
        }
        // the cartesian product, the last coordinate fastest
        let mut tails: Vec<Vec<i64>> = vec![Vec::new()];
        for _ in 0..nvars - 1 {
            tails = tails
                .iter()
                .flat_map(|t| values.iter().map(move |v| t.iter().copied().chain([*v]).collect()))
                .collect();
        }
        for tail in tails {
            if height == 0 || tail.iter().map(|c| c.abs()).max() == Some(height) {
                result.push(tail);
            }
        }
    }
    result
}

/// Linear forms `x_1 + c_2 x_2 + ... + c_n x_n` with small integer
/// coefficients by increasing height, among which one separates any
/// `points` distinct points (`separating_forms`).
pub fn separating_forms(variables: &[Basic], points: usize) -> PolyResult<Vec<Basic>> {
    if variables.len() < 2 {
        return Ok(vec![variables.first().cloned().unwrap_or_else(Basic::zero)]);
    }
    let mut forms: Vec<Basic> = Vec::new();
    for tail in separating_coefficients(variables.len(), points) {
        let mut terms: Vec<Basic> = vec![variables[0].clone()];
        for (c, v) in tail.iter().zip(&variables[1..]) {
            if *c != 0 {
                terms.push(crate::core::mul::mul(&[Basic::int(*c), v.clone()])?);
            }
        }
        forms.push(crate::core::add::add(&terms)?);
    }
    Ok(forms)
}

// ----------------------------------------------------------------------
// Gröbner bases and normal forms

/// The reduced Gröbner basis of `polys` in `ring` (over a field), sorted by
/// decreasing leading monomial (`groebner`).
pub fn groebner<K: Domain>(polys: &[PolyElement<K>], ring: &Arc<PolyRing<K>>) -> PolyResult<Vec<PolyElement<K>>> {
    let nonzero: Vec<PolyElement<K>> = polys.iter().filter(|f| !f.is_zero()).cloned().collect();
    if nonzero.is_empty() {
        return Ok(Vec::new());
    }
    let mut basis: Vec<PolyElement<K>> = super::f4::f4(&nonzero, ring)?;
    basis.retain(|g| !g.is_zero());
    let order: MonomialOrder = ring.order.clone();
    basis.sort_by(|a, b| order.cmp(&b.LM(), &a.LM()));
    Ok(basis)
}

/// The remainder of the division of `f` by `basis` over a field, for the
/// order of the ring of `f` (`normal_form` of `groebnerwalk`).
pub fn normal_form<K: Domain>(f: &PolyElement<K>, basis: &[PolyElement<K>]) -> PolyResult<PolyElement<K>> {
    let k: &K = &f.ring.domain;
    let order: &MonomialOrder = &f.ring.order;
    let mut divisors: Vec<(Monomial, K::Elem, Vec<(Monomial, K::Elem)>)> = Vec::new();
    for g in basis {
        if !g.is_zero() {
            let lm: Monomial = g.LM();
            let inverse: K::Elem = k.invert(&g.terms[&lm])?;
            divisors.push((
                lm,
                inverse,
                g.terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect(),
            ));
        }
    }
    let mut p: Terms<K::Elem> = f.terms.clone();
    let mut remainder: Terms<K::Elem> = Terms::new();
    while !p.is_empty() {
        let m: Monomial = p.keys().max_by(|a, b| order.cmp(a, b)).expect("not empty").clone();
        let c: K::Elem = p[&m].clone();
        let mut divided: bool = false;
        for (lm, inverse, terms) in &divisors {
            if monomial_divides(lm, &m) {
                let q: Monomial = monomial_div(&m, lm)?;
                let factor: K::Elem = k.mul(&c, inverse);
                for (gm, gc) in terms {
                    let t: Monomial = monomial_mul(gm, &q);
                    let product: K::Elem = k.mul(&factor, gc);
                    let value: K::Elem = match p.get(&t) {
                        None => k.neg(&product),
                        Some(old) => k.sub(old, &product),
                    };
                    if k.is_zero(&value) {
                        p.remove(&t);
                    } else {
                        p.insert(t, value);
                    }
                }
                divided = true;
                break;
            }
        }
        if !divided {
            p.remove(&m);
            remainder.insert(m, c);
        }
    }
    Ok(f.ring.from_dict(remainder))
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

/// A dense univariate polynomial (highest degree first) as an expression
/// in `x`.
pub fn dense_to_expr<K: Domain>(rep: &[K::Elem], k: &K, x: &Basic) -> PolyResult<Basic> {
    let n: usize = rep.len().saturating_sub(1);
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in rep.iter().enumerate() {
        if k.is_zero(c) {
            continue;
        }
        let e: usize = n - i;
        let power: Basic = match e {
            0 => Basic::one(),
            1 => x.clone(),
            _ => crate::core::power::pow(x, &Basic::int(e as i64))?,
        };
        terms.push(crate::core::mul::mul(&[k.to_expr(c)?, power])?);
    }
    crate::core::add::add(&terms)
}

// ----------------------------------------------------------------------
// ideals

/// An ideal of a polynomial ring over a field, given by generators
/// (`Ideal`). The Gröbner bases are computed lazily and cached per order.
#[derive(Debug, Clone)]
pub struct Ideal<K: Domain> {
    pub symbols: Vec<Basic>,
    pub domain: K,
    pub order: MonomialOrder,
    names: Vec<String>,
    renamed: Vec<(Basic, Basic)>,
    rings: RefCell<Vec<(MonomialOrder, Arc<PolyRing<K>>)>>,
    bases: RefCell<Vec<(MonomialOrder, Vec<PolyElement<K>>)>>,
    pub gens: Vec<PolyElement<K>>,
}

impl<K: Domain + 'static> std::fmt::Display for Ideal<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gens: Vec<String> = self
            .exprs()
            .map(|v| v.iter().map(|e| e.to_string()).collect())
            .unwrap_or_default();
        let symbols: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
        write!(f, "Ideal([{}], {})", gens.join(", "), symbols.join(", "))
    }
}

/// A polynomial given to an ideal: a core expression or a ring element.
pub enum PolyLike<'a, K: Domain> {
    Expr(&'a Basic),
    Element(&'a PolyElement<K>),
}

impl<K: Domain + 'static> Ideal<K> {
    fn empty(symbols: &[Basic], domain: K, order: MonomialOrder) -> PolyResult<Ideal<K>> {
        if symbols.is_empty() {
            return Err(value_error("no variables: give the symbols of the ring".to_string()));
        }
        if !domain.is_field() {
            return Err(value_error(format!(
                "the coefficient domain of an ideal must be a field, got {}",
                domain.name()
            )));
        }
        let mut names: Vec<String> = Vec::new();
        let mut renamed: Vec<(Basic, Basic)> = Vec::new();
        for s in symbols {
            let base: String = symbol_name(s);
            let (mut name, mut k) = (base.clone(), 0);
            while names.contains(&name) {
                k += 1;
                name = format!("{}_{}", base, k);
            }
            if name != base {
                renamed.push((s.clone(), Symbol::new(&name, &[]).0));
            }
            names.push(name);
        }
        Ok(Ideal {
            symbols: symbols.to_vec(),
            domain,
            order,
            names,
            renamed,
            rings: RefCell::new(Vec::new()),
            bases: RefCell::new(Vec::new()),
            gens: Vec::new(),
        })
    }

    /// The ideal of expressions in the `symbols` (all the free symbols of
    /// the generators, sorted by name, when empty).
    pub fn new(gens: &[Basic], symbols: &[Basic], domain: K, order: MonomialOrder) -> PolyResult<Ideal<K>> {
        let variables: Vec<Basic> = if symbols.is_empty() {
            let mut found: Vec<Basic> = Vec::new();
            for g in gens {
                for s in g.free_symbols() {
                    if !found.contains(&s) {
                        found.push(s);
                    }
                }
            }
            found.sort_by_key(symbol_name);
            found
        } else {
            symbols.to_vec()
        };
        let mut ideal: Ideal<K> = Ideal::empty(&variables, domain, order)?;
        let mut polys: Vec<PolyElement<K>> = Vec::new();
        for g in gens {
            let p: PolyElement<K> = ideal.poly(&PolyLike::Expr(g), None)?;
            if !p.is_zero() {
                polys.push(p);
            }
        }
        ideal.gens = polys;
        Ok(ideal)
    }

    /// The ideal generated by the polynomials with these terms
    /// (`from_terms`).
    pub fn from_terms(
        terms: &[Terms<K::Elem>],
        symbols: &[Basic],
        domain: K,
        order: MonomialOrder,
    ) -> PolyResult<Ideal<K>> {
        let mut result: Ideal<K> = Ideal::empty(symbols, domain, order)?;
        let r: Arc<PolyRing<K>> = result.ring(None)?;
        result.gens = terms
            .iter()
            .filter(|t| !t.is_empty())
            .map(|t| r.from_dict(t.clone()))
            .collect();
        Ok(result)
    }

    /// The polynomial ring with the given order (the default one if
    /// omitted).
    pub fn ring(&self, order: Option<&MonomialOrder>) -> PolyResult<Arc<PolyRing<K>>> {
        let key: MonomialOrder = order.cloned().unwrap_or_else(|| self.order.clone());
        if let Some((_, r)) = self.rings.borrow().iter().find(|(o, _)| *o == key) {
            return Ok(Arc::clone(r));
        }
        let r: Arc<PolyRing<K>> = PolyRing::new(self.domain.clone(), self.names.clone(), key.clone())?;
        self.rings.borrow_mut().push((key, Arc::clone(&r)));
        Ok(r)
    }

    fn cached_basis(&self, key: &MonomialOrder) -> Option<Vec<PolyElement<K>>> {
        self.bases
            .borrow()
            .iter()
            .find(|(o, _)| o == key)
            .map(|(_, b)| b.clone())
    }

    fn store_basis(&self, key: MonomialOrder, basis: Vec<PolyElement<K>>) {
        let mut bases = self.bases.borrow_mut();
        bases.retain(|(o, _)| *o != key);
        bases.push((key, basis));
    }

    /// The reduced Gröbner basis for the order, sorted by decreasing
    /// leading monomial (`_basis`).
    fn basis(&self, order: Option<&MonomialOrder>) -> PolyResult<Vec<PolyElement<K>>> {
        let key: MonomialOrder = order.cloned().unwrap_or_else(|| self.order.clone());
        if let Some(found) = self.cached_basis(&key) {
            return Ok(found);
        }
        let r: Arc<PolyRing<K>> = self.ring(Some(&key))?;
        let mut g: Vec<PolyElement<K>> = match self.converted_basis(&key)? {
            Some(converted) => converted,
            None => {
                let gens: Vec<PolyElement<K>> = self.gens.iter().map(|g| r.from_dict(g.terms.clone())).collect();
                groebner(&gens, &r)?
            }
        };
        g.retain(|p| !p.is_zero());
        g.sort_by(|a, b| key.cmp(&b.LM(), &a.LM()));
        self.store_basis(key, g.clone());
        Ok(g)
    }

    /// The basis for `lex` or `grlex` of a zero-dimensional ideal by FGLM
    /// from the `grevlex` one (`_converted_basis`).
    fn converted_basis(&self, order: &MonomialOrder) -> PolyResult<Option<Vec<PolyElement<K>>>> {
        if !matches!(order, MonomialOrder::Lex | MonomialOrder::Grlex) || self.gens.is_empty() {
            return Ok(None);
        }
        let graded: Vec<PolyElement<K>> = self.basis(Some(&MonomialOrder::Grevlex))?;
        let mut pure: Vec<usize> = Vec::new();
        for g in &graded {
            let lm: Monomial = g.LM();
            let total: u32 = lm.iter().sum();
            for (i, e) in lm.iter().enumerate() {
                if *e > 0 && *e == total && !pure.contains(&i) {
                    pure.push(i);
                }
            }
        }
        if pure.len() < self.symbols.len() {
            return Ok(None);
        }
        let r: Arc<PolyRing<K>> = self.ring(Some(order))?;
        let converted = super::fglm::fglm(&graded, &self.ring(Some(&MonomialOrder::Grevlex))?, order)?;
        Ok(Some(converted.iter().map(|g| r.from_dict(g.terms.clone())).collect()))
    }

    /// The reduced Gröbner basis for the order, sorted by decreasing
    /// leading monomial (`groebner_basis`).
    pub fn groebner_basis(&self, order: Option<&MonomialOrder>) -> PolyResult<Vec<PolyElement<K>>> {
        self.basis(order)
    }

    /// Record the reduced Gröbner basis for the default order
    /// (`_set_basis`).
    fn set_basis(&self, basis: &[Terms<K::Elem>]) -> PolyResult<()> {
        let r: Arc<PolyRing<K>> = self.ring(None)?;
        let mut g: Vec<PolyElement<K>> = basis
            .iter()
            .filter(|t| !t.is_empty())
            .map(|t| r.from_dict(t.clone()))
            .collect();
        let order: MonomialOrder = self.order.clone();
        g.sort_by(|a, b| order.cmp(&b.LM(), &a.LM()));
        self.store_basis(order, g);
        Ok(())
    }

    /// `f` as an element of the ring with the order (`_poly`).
    pub fn poly(&self, f: &PolyLike<K>, order: Option<&MonomialOrder>) -> PolyResult<PolyElement<K>> {
        let r: Arc<PolyRing<K>> = self.ring(order)?;
        match f {
            PolyLike::Element(p) => {
                if p.ring.ngens != self.symbols.len() {
                    return Err(value_error(format!(
                        "{} is not an element of a ring in {} variables",
                        p,
                        self.symbols.len()
                    )));
                }
                Ok(r.from_dict(p.terms.clone()))
            }
            PolyLike::Expr(e) => {
                let e: Basic = if self.renamed.is_empty() {
                    (*e).clone()
                } else {
                    let rule: std::collections::HashMap<Basic, Basic> = self.renamed.iter().cloned().collect();
                    crate::core::function::xreplace_evaluated(e, &rule)?
                };
                Ok(r.from_dict(super::polyutils::dict_from_expr(&e, &self.names, &self.domain)?))
            }
        }
    }

    fn poly_expr(&self, e: &Basic) -> PolyResult<PolyElement<K>> {
        self.poly(&PolyLike::Expr(e), None)
    }

    /// An element of the ring as a core expression in the symbols.
    pub fn to_expr(&self, f: &PolyElement<K>) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (m, c) in &f.terms {
            let mut factors: Vec<Basic> = vec![self.domain.to_expr(c)?];
            for (s, e) in self.symbols.iter().zip(m) {
                if *e == 1 {
                    factors.push(s.clone());
                } else if *e > 1 {
                    factors.push(crate::core::power::pow(s, &Basic::int(*e as i64))?);
                }
            }
            parts.push(crate::core::mul::mul(&factors)?);
        }
        if parts.is_empty() {
            return Ok(Basic::zero());
        }
        crate::core::add::add(&parts)
    }

    /// The ideal of the same ring (or of the ring in `symbols`) generated by
    /// ring elements (`_new`).
    fn derived(&self, gens: &[PolyElement<K>], symbols: Option<&[Basic]>) -> PolyResult<Ideal<K>> {
        let variables: &[Basic] = symbols.unwrap_or(&self.symbols);
        let terms: Vec<Terms<K::Elem>> = gens.iter().map(|g| g.terms.clone()).collect();
        Ideal::from_terms(&terms, variables, self.domain.clone(), self.order.clone())
    }

    /// The generators as core expressions.
    pub fn exprs(&self) -> PolyResult<Vec<Basic>> {
        self.gens.iter().map(|g| self.to_expr(g)).collect()
    }

    // -- membership and equality

    /// The normal form of `f` modulo the Gröbner basis for the order.
    pub fn normal_form(&self, f: &PolyLike<K>, order: Option<&MonomialOrder>) -> PolyResult<PolyElement<K>> {
        normal_form(&self.poly(f, order)?, &self.basis(order)?)
    }

    /// The normal form of `f` as a core expression (`reduce`).
    pub fn reduce(&self, f: &PolyLike<K>, order: Option<&MonomialOrder>) -> PolyResult<Basic> {
        self.to_expr(&self.normal_form(f, order)?)
    }

    /// Whether the polynomial `f` belongs to the ideal.
    pub fn contains(&self, f: &PolyLike<K>) -> PolyResult<bool> {
        Ok(self.normal_form(f, None)?.is_zero())
    }

    pub fn is_zero(&self) -> PolyResult<bool> {
        Ok(self.basis(None)?.is_empty())
    }

    pub fn is_whole_ring(&self) -> PolyResult<bool> {
        let g: Vec<PolyElement<K>> = self.basis(None)?;
        Ok(g.len() == 1 && g[0].is_ground())
    }

    fn same_ring(&self, other: &Ideal<K>) -> PolyResult<()> {
        if other.symbols != self.symbols {
            return Err(value_error("the ideals live in different rings".to_string()));
        }
        Ok(())
    }

    /// Equality of ideals (`__eq__`): the same reduced Gröbner basis.
    pub fn equals(&self, other: &Ideal<K>) -> PolyResult<bool> {
        if other.symbols != self.symbols || other.domain != self.domain {
            return Ok(false);
        }
        let mine: Vec<Terms<K::Elem>> = self.basis(None)?.into_iter().map(|g| g.terms).collect();
        let theirs: Vec<Terms<K::Elem>> = other.basis(Some(&self.order))?.into_iter().map(|g| g.terms).collect();
        Ok(mine == theirs)
    }

    /// Whether the ideal is contained in `other`.
    pub fn subset(&self, other: &Ideal<K>) -> PolyResult<bool> {
        self.same_ring(other)?;
        for g in &self.gens {
            if !other.contains(&PolyLike::Element(g))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // -- arithmetic

    pub fn add(&self, other: &Ideal<K>) -> PolyResult<Ideal<K>> {
        self.same_ring(other)?;
        let mut gens: Vec<PolyElement<K>> = self.gens.clone();
        gens.extend(other.gens.iter().cloned());
        self.derived(&gens, None)
    }

    pub fn mul(&self, other: &Ideal<K>) -> PolyResult<Ideal<K>> {
        self.same_ring(other)?;
        let products: Vec<PolyElement<K>> = self
            .gens
            .iter()
            .flat_map(|f| other.gens.iter().map(move |g| f.mul(g)))
            .collect();
        self.derived(&products, None)
    }

    /// The product by a polynomial.
    pub fn mul_poly(&self, other: &PolyLike<K>) -> PolyResult<Ideal<K>> {
        let h: PolyElement<K> = self.poly(other, None)?;
        let products: Vec<PolyElement<K>> = self.gens.iter().map(|g| g.mul(&h)).collect();
        self.derived(&products, None)
    }

    pub fn pow(&self, n: usize) -> PolyResult<Ideal<K>> {
        let mut result: Ideal<K> = self.derived(&[self.ring(None)?.one()], None)?;
        for _ in 0..n {
            result = result.mul(self)?;
        }
        Ok(result)
    }

    // -- elimination and the operations built on it

    /// The elimination ideal in the variables other than `symbols`, as an
    /// ideal of that ring (`eliminate`).
    pub fn eliminate(&self, symbols: &[Basic]) -> PolyResult<Ideal<K>> {
        for s in symbols {
            if !self.symbols.contains(s) {
                return Err(value_error(format!("{} is not a variable of the ring", s)));
            }
        }
        if symbols.is_empty() {
            return Ok(self.clone());
        }
        let positions: Vec<usize> = symbols
            .iter()
            .map(|s| self.symbols.iter().position(|t| t == s).expect("a variable"))
            .collect();
        let kept: Vec<usize> = (0..self.symbols.len()).filter(|i| !positions.contains(i)).collect();
        let permutation: Vec<usize> = positions.iter().chain(&kept).copied().collect();
        let k: usize = symbols.len();
        let r: Arc<PolyRing<K>> = PolyRing::new(
            self.domain.clone(),
            permutation.iter().map(|i| self.names[*i].clone()).collect(),
            MonomialOrder::elimination(k, permutation.len(), MonomialOrder::Grevlex),
        )?;
        let gens: Vec<PolyElement<K>> = self
            .gens
            .iter()
            .map(|g| {
                r.from_dict(
                    g.terms
                        .iter()
                        .map(|(m, c)| (permutation.iter().map(|i| m[*i]).collect(), c.clone()))
                        .collect(),
                )
            })
            .collect();
        let basis: Vec<PolyElement<K>> = groebner(&gens, &r)?;
        let found: Vec<Terms<K::Elem>> = basis
            .iter()
            .filter(|g| g.terms.keys().all(|m| m[..k].iter().all(|e| *e == 0)))
            .map(|g| g.terms.iter().map(|(m, c)| (m[k..].to_vec(), c.clone())).collect())
            .collect();
        let kept_symbols: Vec<Basic> = kept.iter().map(|i| self.symbols[*i].clone()).collect();
        let result: Ideal<K> = Ideal::from_terms(&found, &kept_symbols, self.domain.clone(), self.order.clone())?;
        if self.order == MonomialOrder::Grevlex {
            let terms: Vec<Terms<K::Elem>> = result.gens.iter().map(|g| g.terms.clone()).collect();
            result.set_basis(&terms)?;
        }
        Ok(result)
    }

    fn new_variable(&self) -> Basic {
        Dummy::new("t", &[]).0
    }

    /// The ideal of `K[t, x]` generated by the generators and the given
    /// polynomials of `K[t, x]` (`_extended`).
    fn extended(&self, polys: &[Terms<K::Elem>]) -> PolyResult<Ideal<K>> {
        let mut terms: Vec<Terms<K::Elem>> = self.gens.iter().map(|g| shifted(&g.terms, 0)).collect();
        terms.extend(polys.iter().cloned());
        let mut symbols: Vec<Basic> = vec![self.new_variable()];
        symbols.extend(self.symbols.iter().cloned());
        Ideal::from_terms(&terms, &symbols, self.domain.clone(), self.order.clone())
    }

    /// The terms of `1 - t*f` in `K[t, x]`.
    fn one_minus_t_times(&self, f: &PolyElement<K>) -> Terms<K::Elem> {
        let mut terms: Terms<K::Elem> = f
            .terms
            .iter()
            .map(|(m, c)| (prefixed(m, 1), self.domain.neg(c)))
            .collect();
        terms.insert(vec![0; self.symbols.len() + 1], self.domain.one());
        terms
    }

    /// The intersection with `other`, as `(tI + (1 - t)J) ∩ K[x]`.
    pub fn intersect(&self, other: &Ideal<K>) -> PolyResult<Ideal<K>> {
        self.same_ring(other)?;
        let k: &K = &self.domain;
        let mut gens: Vec<Terms<K::Elem>> = self.gens.iter().map(|g| shifted(&g.terms, 1)).collect();
        for g in &other.gens {
            let mut terms: Terms<K::Elem> = shifted(&g.terms, 0);
            for (m, c) in &g.terms {
                terms.insert(prefixed(m, 1), k.neg(c));
            }
            gens.push(terms);
        }
        let t: Basic = self.new_variable();
        let mut symbols: Vec<Basic> = vec![t.clone()];
        symbols.extend(self.symbols.iter().cloned());
        Ideal::from_terms(&gens, &symbols, self.domain.clone(), self.order.clone())?.eliminate(&[t])
    }

    fn divisor_of(&self, other: &IdealOrPoly<K>) -> PolyResult<Ideal<K>> {
        match other {
            IdealOrPoly::Ideal(i) => Ok((*i).clone()),
            IdealOrPoly::Poly(p) => self.derived(&[self.poly(p, None)?], None),
        }
    }

    /// The ideal quotient `I : J` (`quotient`).
    pub fn quotient(&self, other: &IdealOrPoly<K>) -> PolyResult<Ideal<K>> {
        let divisor: Ideal<K> = self.divisor_of(other)?;
        self.same_ring(&divisor)?;
        let r: Arc<PolyRing<K>> = self.ring(None)?;
        let mut result: Option<Ideal<K>> = None;
        for f in &divisor.gens {
            let g_f: PolyElement<K> = r.from_dict(f.terms.clone());
            let common: Ideal<K> = self.intersect(&self.derived(std::slice::from_ref(&g_f), None)?)?;
            let mut quotients: Vec<PolyElement<K>> = Vec::new();
            for g in common.basis(None)? {
                let (q, rem) = r.from_dict(g.terms.clone()).div(std::slice::from_ref(&g_f))?;
                if !rem.is_zero() {
                    return Err(crate::error::arithmetic_error(
                        "inexact division in the ideal quotient".to_string(),
                    ));
                }
                quotients.push(q[0].monic()?);
            }
            let part: Ideal<K> = self.derived(&quotients, None)?;
            result = Some(match result {
                None => part,
                Some(previous) => previous.intersect(&part)?,
            });
        }
        match result {
            Some(found) => Ok(found),
            None => self.derived(&[r.one()], None),
        }
    }

    /// The saturation `I : J**oo` (`saturate`).
    pub fn saturate(&self, other: &IdealOrPoly<K>) -> PolyResult<Ideal<K>> {
        let divisor: Ideal<K> = self.divisor_of(other)?;
        if divisor.gens.len() == 1 {
            let extended: Ideal<K> = self.extended(&[self.one_minus_t_times(&divisor.gens[0])])?;
            let t: Basic = extended.symbols[0].clone();
            return extended.eliminate(&[t]);
        }
        let mut current: Ideal<K> = self.clone();
        loop {
            let following: Ideal<K> = current.quotient(&IdealOrPoly::Ideal(&divisor))?;
            if following.equals(&current)? {
                return Ok(following);
            }
            current = following;
        }
    }

    /// Whether `f` belongs to the radical (Rabinowitsch's trick).
    pub fn radical_contains(&self, f: &PolyLike<K>) -> PolyResult<bool> {
        let p: PolyElement<K> = self.poly(f, None)?;
        self.extended(&[self.one_minus_t_times(&p)])?.is_whole_ring()
    }

    // -- dimension, Hilbert series

    /// The leading monomials of the Gröbner basis.
    pub fn leading_monomials(&self, order: Option<&MonomialOrder>) -> PolyResult<Vec<Monomial>> {
        Ok(self.basis(order)?.iter().map(|g| g.LM()).collect())
    }

    fn graded_order(&self) -> MonomialOrder {
        if is_graded(&self.order) {
            self.order.clone()
        } else {
            MonomialOrder::Grevlex
        }
    }

    /// `(q, dim)`: the numerator of the Hilbert series divided by the
    /// largest power of `1 - t` (lowest degree first) and the Krull
    /// dimension (`hilbert_data`).
    pub fn hilbert_data(&self) -> PolyResult<(Vec<Int>, i64)> {
        let mut numerator: Vec<Int> =
            super::bigatti::hilbert_numerator_coefficients(&self.leading_monomials(Some(&self.graded_order()))?);
        let mut s: i64 = 0;
        while !numerator.is_empty() && numerator.iter().fold(Int::ZERO, |a, c| a + c).is_zero() {
            let mut quotient: Vec<Int> = Vec::new();
            let mut running: Int = Int::ZERO;
            for c in &numerator[..numerator.len() - 1] {
                running += c;
                quotient.push(running.clone());
            }
            numerator = quotient;
            s += 1;
        }
        Ok((numerator, self.symbols.len() as i64 - s))
    }

    /// The Hilbert series of `K[x]/I` as `q(t)/(1 - t)**d`.
    pub fn hilbert_series(&self, t: Option<&Basic>) -> PolyResult<Basic> {
        let variable: Basic = t.cloned().unwrap_or_else(|| Symbol::new("t", &[]).0);
        let (q, dim) = self.hilbert_data()?;
        if q.is_empty() {
            return Ok(Basic::zero());
        }
        let rep: Vec<Int> = q.iter().rev().cloned().collect();
        let numerator: Basic = dense_to_expr(&rep, &super::domains::ZZ, &variable)?;
        if dim == 0 {
            return Ok(numerator);
        }
        let one_minus_t: Basic =
            crate::core::add::add(&[Basic::one(), crate::core::mul::mul(&[Basic::int(-1), variable])?])?;
        crate::core::mul::mul(&[numerator, crate::core::power::pow(&one_minus_t, &Basic::int(-dim))?])
    }

    /// The Krull dimension of `K[x]/I` (`-1` for the whole ring).
    pub fn dimension(&self) -> PolyResult<i64> {
        if self.is_whole_ring()? {
            return Ok(-1);
        }
        Ok(self.hilbert_data()?.1)
    }

    pub fn is_zero_dimensional(&self) -> PolyResult<bool> {
        Ok(self.dimension()? == 0)
    }

    /// The degree of the ideal.
    pub fn degree(&self) -> PolyResult<Int> {
        Ok(self.hilbert_data()?.0.iter().fold(Int::ZERO, |a, c| a + c))
    }

    /// The Hilbert polynomial of `K[x]/I`.
    pub fn hilbert_polynomial(&self, d: Option<&Basic>) -> PolyResult<Basic> {
        let variable: Basic = d.cloned().unwrap_or_else(|| Symbol::new("d", &[]).0);
        let (q, dim) = self.hilbert_data()?;
        if dim <= 0 {
            return Ok(Basic::zero());
        }
        let dim: usize = dim as usize;
        let mut total: Vec<Rat> = vec![Rat::ZERO; dim];
        let factorial: Int = (1..dim).fold(Int::ONE, |p, j| p * Int::from(j));
        for (i, c) in q.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let mut term: Vec<Rat> = vec![crate::numbers::rat_from_parts(c.clone(), factorial.clone())];
            for j in 1..dim {
                let mut next: Vec<Rat> = vec![Rat::ZERO; term.len() + 1];
                for (k, a) in term.iter().enumerate() {
                    next[k] += a * Rat::from(Int::from(j as i64 - i as i64));
                    next[k + 1] += a;
                }
                term = next;
            }
            for (k, a) in term.iter().enumerate() {
                total[k] += a;
            }
        }
        let rep: Vec<Rat> = total.into_iter().rev().collect();
        let rep: Vec<Rat> = super::densebasic::dup_strip(&rep, &super::domains::QQ);
        dense_to_expr(&rep, &super::domains::QQ, &variable)
    }

    // -- zero-dimensional ideals

    fn require_zero_dimensional(&self) -> PolyResult<()> {
        if !self.is_zero_dimensional()? {
            return Err(not_implemented("the ideal is not zero-dimensional"));
        }
        Ok(())
    }

    /// The exponents of the standard monomials, by increasing monomial
    /// (`standard_exponents`).
    pub fn standard_exponents(&self, order: Option<&MonomialOrder>) -> PolyResult<Vec<Monomial>> {
        self.require_zero_dimensional()?;
        let lms: Vec<Monomial> = self.leading_monomials(order)?;
        let n: usize = self.symbols.len();
        let key: MonomialOrder = self.ring(order)?.order.clone();
        let mut found: Vec<Monomial> = Vec::new();
        let mut seen: std::collections::BTreeSet<Monomial> = std::collections::BTreeSet::new();
        let mut queue: std::collections::VecDeque<Monomial> = std::collections::VecDeque::from([vec![0; n]]);
        while let Some(m) = queue.pop_front() {
            if seen.contains(&m) || lms.iter().any(|lm| monomial_divides(lm, &m)) {
                continue;
            }
            seen.insert(m.clone());
            found.push(m.clone());
            for i in 0..n {
                let mut next: Monomial = m.clone();
                next[i] += 1;
                queue.push_back(next);
            }
        }
        found.sort_by(|a, b| key.cmp(a, b));
        Ok(found)
    }

    /// The standard monomials as core expressions.
    pub fn standard_monomials(&self, order: Option<&MonomialOrder>) -> PolyResult<Vec<Basic>> {
        let r: Arc<PolyRing<K>> = self.ring(order)?;
        self.standard_exponents(order)?
            .into_iter()
            .map(|m| self.to_expr(&r.term(m, self.domain.one())))
            .collect()
    }

    pub fn vector_space_dimension(&self) -> PolyResult<usize> {
        Ok(self.standard_exponents(None)?.len())
    }

    /// The matrix (as rows) of multiplication by `f` in the basis of the
    /// standard monomials.
    pub fn multiplication_matrix(
        &self,
        f: &PolyLike<K>,
        order: Option<&MonomialOrder>,
    ) -> PolyResult<Vec<Vec<K::Elem>>> {
        let g_basis: Vec<PolyElement<K>> = self.basis(order)?;
        let monomials: Vec<Monomial> = self.standard_exponents(order)?;
        let g: PolyElement<K> = self.poly(f, order)?;
        let mut rows: Vec<Vec<K::Elem>> = vec![vec![self.domain.zero(); monomials.len()]; monomials.len()];
        for (j, m) in monomials.iter().enumerate() {
            let p: PolyElement<K> = normal_form(&g.mul_monom(m), &g_basis)?;
            for (mm, c) in &p.terms {
                let i: usize = monomials.iter().position(|x| x == mm).expect("a standard monomial");
                rows[i][j] = c.clone();
            }
        }
        Ok(rows)
    }

    /// The monic polynomial of least degree with `p(f)` in the ideal, dense
    /// (highest degree first) (`minimal_polynomial`).
    pub fn minimal_polynomial(&self, f: &PolyLike<K>) -> PolyResult<Vec<K::Elem>> {
        self.require_zero_dimensional()?;
        let m: Vec<Vec<K::Elem>> = self.multiplication_matrix(f, None)?;
        let k: &K = &self.domain;
        let dim: usize = m.len();
        let mut pivots: Vec<(usize, Vec<K::Elem>, Vec<(usize, K::Elem)>)> = Vec::new();
        let mut v: Vec<K::Elem> = (0..dim).map(|i| if i == 0 { k.one() } else { k.zero() }).collect();
        let mut step: usize = 0;
        loop {
            let mut vector: Vec<K::Elem> = v.clone();
            let mut combination: Vec<(usize, K::Elem)> = vec![(step, k.one())];
            for (pivot, row, row_combination) in &pivots {
                let c: K::Elem = vector[*pivot].clone();
                if k.is_zero(&c) {
                    continue;
                }
                vector = vector.iter().zip(row).map(|(a, b)| k.sub(a, &k.mul(&c, b))).collect();
                for (j, b) in row_combination {
                    let product: K::Elem = k.mul(&c, b);
                    match combination.iter_mut().find(|(i, _)| i == j) {
                        Some(entry) => entry.1 = k.sub(&entry.1, &product),
                        None => combination.push((*j, k.neg(&product))),
                    }
                }
            }
            let Some(first) = vector.iter().position(|a| !k.is_zero(a)) else {
                return Ok((0..=step)
                    .rev()
                    .map(|j| {
                        combination
                            .iter()
                            .find(|(i, _)| *i == j)
                            .map(|(_, c)| c.clone())
                            .unwrap_or_else(|| k.zero())
                    })
                    .collect());
            };
            let inverse: K::Elem = k.invert(&vector[first])?;
            pivots.push((
                first,
                vector.iter().map(|a| k.mul(a, &inverse)).collect(),
                combination.iter().map(|(j, b)| (*j, k.mul(b, &inverse))).collect(),
            ));
            v = m.iter().map(|row| dot(row, &v, k)).collect();
            step += 1;
        }
    }

    /// The monic generator of `I ∩ K[x]` for a variable `x`, dense
    /// (`univariate`).
    pub fn univariate(&self, x: &Basic) -> PolyResult<Vec<K::Elem>> {
        if !self.symbols.contains(x) {
            return Err(value_error(format!("{} is not a variable of the ring", x)));
        }
        self.minimal_polynomial(&PolyLike::Expr(x))
    }

    /// The radical: by Seidenberg's lemma in dimension zero, by the
    /// regular chains of `idealdecomposition` otherwise (`radical`).
    pub fn radical(&self) -> PolyResult<Ideal<K>> {
        if self.is_whole_ring()? {
            return self.reduced();
        }
        if !self.is_zero_dimensional()? {
            let rational = self.rational("ideals of positive dimension are decomposed over the rationals only")?;
            return Ok(Ideal::back(super::idealdecomposition::radical(rational)?));
        }
        use super::densearith::dup_quo;
        use super::densetools::dup_diff;
        use super::euclidtools::dup_gcd;
        let r: Arc<PolyRing<K>> = self.ring(None)?;
        let n: usize = self.symbols.len();
        let k: &K = &self.domain;
        let mut extra: Vec<PolyElement<K>> = Vec::new();
        for (i, s) in self.symbols.iter().enumerate() {
            let p: Vec<K::Elem> = self.univariate(s)?;
            let sqf: Vec<K::Elem> = dup_quo(&p, &dup_gcd(&p, &dup_diff(&p, 1, k), k), k)?;
            let d: usize = sqf.len() - 1;
            let mut terms: Terms<K::Elem> = Terms::new();
            for (j, c) in sqf.iter().enumerate() {
                if !k.is_zero(c) {
                    terms.insert(
                        (0..n).map(|kk| if kk == i { (d - j) as u32 } else { 0 }).collect(),
                        c.clone(),
                    );
                }
            }
            extra.push(r.from_dict(terms));
        }
        let mut gens: Vec<PolyElement<K>> = self.basis(None)?;
        gens.extend(extra);
        self.derived(&gens, None)?.reduced()
    }

    /// The same ideal generated by its reduced Gröbner basis.
    pub fn reduced(&self) -> PolyResult<Ideal<K>> {
        let g: Vec<PolyElement<K>> = self.basis(None)?;
        let result: Ideal<K> = self.derived(&g, None)?;
        result.set_basis(&g.iter().map(|p| p.terms.clone()).collect::<Vec<_>>())?;
        Ok(result)
    }

    pub fn is_radical(&self) -> PolyResult<bool> {
        self.radical()?.equals(self)
    }

    /// The height (codimension) of a proper ideal.
    pub fn height(&self) -> PolyResult<i64> {
        if self.is_whole_ring()? {
            return Err(value_error("the whole ring has no height".to_string()));
        }
        Ok(self.symbols.len() as i64 - self.dimension()?)
    }

    /// Whether `K[x]/I` is a field (`is_maximal`).
    pub fn is_maximal(&self) -> PolyResult<bool> {
        if self.is_whole_ring()? || !self.is_zero_dimensional()? {
            return Ok(false);
        }
        if !self.is_radical()? {
            return Ok(false);
        }
        let dim: usize = self.vector_space_dimension()?;
        if dim == 1 {
            return Ok(true);
        }
        let r: Arc<PolyRing<K>> = self.ring(None)?;
        let k: &K = &self.domain;
        let n: usize = self.symbols.len();
        let units: Vec<Monomial> = (0..n)
            .map(|i| (0..n).map(|kk| if kk == i { 1 } else { 0 }).collect())
            .collect();
        let tails: Vec<Vec<i64>> = if n >= 2 {
            separating_coefficients(n, dim)
        } else {
            vec![Vec::new()]
        };
        for tail in tails {
            let mut terms: Terms<K::Elem> = Terms::new();
            terms.insert(units[0].clone(), k.one());
            for (c, unit) in tail.iter().zip(&units[1..]) {
                if *c != 0 {
                    terms.insert(unit.clone(), k.from_int(*c));
                }
            }
            let p: Vec<K::Elem> = self.minimal_polynomial(&PolyLike::Element(&r.from_dict(terms)))?;
            if p.len() - 1 < dim {
                continue;
            }
            let (_, factors) = k.factor_dense(&p)?;
            return Ok(factors.len() == 1 && factors[0].1 == 1);
        }
        Err(crate::error::arithmetic_error(
            "no separating linear form was found".to_string(),
        ))
    }

    /// Whether the ideal is prime: maximal in dimension zero, its own only
    /// minimal prime otherwise.
    pub fn is_prime(&self) -> PolyResult<bool> {
        if self.is_whole_ring()? {
            return Ok(false);
        }
        if self.is_zero_dimensional()? {
            return self.is_maximal();
        }
        let primes: Vec<Ideal<K>> = self.minimal_primes()?;
        Ok(primes.len() == 1 && primes[0].equals(self)?)
    }

    /// The ideal over the rationals (the same type), for the
    /// decompositions of `idealdecomposition`.
    fn rational(&self, message: &str) -> PolyResult<&Ideal<RationalField>> {
        (self as &dyn Any)
            .downcast_ref::<Ideal<RationalField>>()
            .ok_or_else(|| not_implemented(message))
    }

    fn back(ideal: Ideal<RationalField>) -> Ideal<K> {
        *(Box::new(ideal) as Box<dyn Any>)
            .downcast::<Ideal<K>>()
            .expect("an ideal over the rationals")
    }

    /// The equidimensional parts of the radical, by decreasing dimension.
    pub fn equidimensional_parts(&self) -> PolyResult<Vec<Ideal<K>>> {
        let rational = self.rational("ideals of positive dimension are decomposed over the rationals only")?;
        Ok(super::idealdecomposition::equidimensional_parts(rational)?
            .into_iter()
            .map(Ideal::back)
            .collect())
    }

    /// The minimal primes, by decreasing dimension.
    pub fn minimal_primes(&self) -> PolyResult<Vec<Ideal<K>>> {
        let rational = self.rational("ideals of positive dimension are decomposed over the rationals only")?;
        Ok(super::idealdecomposition::minimal_primes(rational)?
            .into_iter()
            .map(Ideal::back)
            .collect())
    }

    /// An irredundant primary decomposition, as pairs of a primary ideal
    /// and its associated prime.
    pub fn primary_decomposition(&self) -> PolyResult<Vec<(Ideal<K>, Ideal<K>)>> {
        let rational = self.rational("ideals are decomposed over the rationals only")?;
        Ok(super::idealdecomposition::primary_decomposition(rational)?
            .into_iter()
            .map(|(q, p)| (Ideal::back(q), Ideal::back(p)))
            .collect())
    }

    /// The associated primes, minimal and embedded, by decreasing dimension.
    pub fn associated_primes(&self) -> PolyResult<Vec<Ideal<K>>> {
        let rational = self.rational("ideals are decomposed over the rationals only")?;
        Ok(super::idealdecomposition::associated_primes(rational)?
            .into_iter()
            .map(Ideal::back)
            .collect())
    }

    /// The rational univariate representation (zero-dimensional ideals
    /// over the rationals, `rur`).
    pub fn rur(&self, t: Option<&Basic>) -> PolyResult<super::rur::RationalUnivariateRepresentation> {
        let rational = self.rational("real zeros are counted for ideals over QQ")?;
        super::rur::rur(&super::hermite::System::Ideal(rational), t)
    }

    /// Whether the ideal is primary.
    pub fn is_primary(&self) -> PolyResult<bool> {
        let rational = self.rational("ideals are decomposed over the rationals only")?;
        super::idealdecomposition::is_primary(rational)
    }

    /// The reduced Gröbner basis for another order, by FGLM for
    /// zero-dimensional ideals and the Gröbner walk otherwise
    /// (`change_order`).
    pub fn change_order(&self, target: &MonomialOrder) -> PolyResult<Vec<PolyElement<K>>> {
        if self.cached_basis(target).is_some() {
            return self.groebner_basis(Some(target));
        }
        let g: Vec<PolyElement<K>> = self.basis(None)?;
        let r: Arc<PolyRing<K>> = self.ring(None)?;
        let r2: Arc<PolyRing<K>> = self.ring(Some(target))?;
        let simple =
            |o: &MonomialOrder| matches!(o, MonomialOrder::Lex | MonomialOrder::Grevlex | MonomialOrder::Grlex);
        let converted: Vec<PolyElement<K>> = if self.is_zero_dimensional()? && simple(target) && simple(&self.order) {
            super::fglm::fglm(&g, &r, target)?
        } else {
            super::groebnerwalk::groebner_walk(&g, &r, target, false, super::groebnerwalk::WalkLift::NormalForm)?
        };
        self.store_basis(
            target.clone(),
            converted.iter().map(|p| r2.from_dict(p.terms.clone())).collect(),
        );
        self.groebner_basis(Some(target))
    }

    /// The ideal of an expression (a convenience for [`PolyLike::Expr`]).
    pub fn contains_expr(&self, e: &Basic) -> PolyResult<bool> {
        Ok(self.normal_form(&PolyLike::Expr(e), None)?.is_zero() && self.poly_expr(e).is_ok())
    }
}

/// An ideal or a single polynomial (the divisor of a quotient or a
/// saturation).
pub enum IdealOrPoly<'a, K: Domain> {
    Ideal(&'a Ideal<K>),
    Poly(PolyLike<'a, K>),
}

fn prefixed(m: &[u32], first: u32) -> Monomial {
    std::iter::once(first).chain(m.iter().copied()).collect()
}

fn shifted<T: Clone>(terms: &Terms<T>, first: u32) -> Terms<T> {
    terms.iter().map(|(m, c)| (prefixed(m, first), c.clone())).collect()
}

fn dot<K: Domain>(row: &[K::Elem], v: &[K::Elem], k: &K) -> K::Elem {
    let mut total: K::Elem = k.zero();
    for (a, b) in row.iter().zip(v) {
        if !k.is_zero(a) && !k.is_zero(b) {
            total = k.add(&total, &k.mul(a, b));
        }
    }
    total
}

/// One operation on an ideal for the bindings: `op` with the generators
/// `gens` in `symbols` over `domain`, the default order `order`, and the
/// further arguments `extra` (expressions; a `Tuple` stands for a list).
/// The results are expressions (a `Tuple` for a list, a boolean or an
/// integer for the predicates and the counts).
pub fn ideal_operation<K: Domain + 'static>(
    op: &str,
    gens: &[Basic],
    symbols: &[Basic],
    domain: K,
    order: MonomialOrder,
    extra: &[Basic],
) -> PolyResult<Basic> {
    let ideal: Ideal<K> = Ideal::new(gens, symbols, domain.clone(), order.clone())?;
    let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
    let exprs = |i: &Ideal<K>, g: &[PolyElement<K>]| -> PolyResult<Basic> {
        Ok(tuple(g.iter().map(|p| i.to_expr(p)).collect::<PolyResult<_>>()?))
    };
    let other = || Ideal::new(extra[0].args(), symbols, domain.clone(), order.clone());
    let boolean = crate::core::logic::boolean;
    let order_of = |b: &Basic| MonomialOrder::named(&b.to_string());
    Ok(match op {
        "groebner_basis" => {
            let o: Option<MonomialOrder> = extra.first().map(order_of).transpose()?;
            exprs(&ideal, &ideal.groebner_basis(o.as_ref())?)?
        }
        "change_order" => exprs(&ideal, &ideal.change_order(&order_of(&extra[0])?)?)?,
        "reduce" => ideal.reduce(&PolyLike::Expr(&extra[0]), None)?,
        "contains" => boolean(ideal.contains(&PolyLike::Expr(&extra[0]))?),
        "radical_contains" => boolean(ideal.radical_contains(&PolyLike::Expr(&extra[0]))?),
        "eliminate" => {
            let result: Ideal<K> = ideal.eliminate(extra[0].args())?;
            tuple(vec![
                exprs(&result, &result.groebner_basis(None)?)?,
                tuple(result.symbols.clone()),
            ])
        }
        "intersect" => {
            let result: Ideal<K> = ideal.intersect(&other()?)?;
            exprs(&result, &result.groebner_basis(None)?)?
        }
        "quotient" | "saturate" => {
            let o: Ideal<K> = other()?;
            let divisor: IdealOrPoly<K> = IdealOrPoly::Ideal(&o);
            let result: Ideal<K> = if op == "quotient" {
                ideal.quotient(&divisor)?
            } else {
                ideal.saturate(&divisor)?
            };
            exprs(&result, &result.groebner_basis(None)?)?
        }
        "sum" | "product" => {
            let o: Ideal<K> = other()?;
            let result: Ideal<K> = if op == "sum" { ideal.add(&o)? } else { ideal.mul(&o)? };
            exprs(&result, &result.groebner_basis(None)?)?
        }
        "power" => {
            let n: usize = extra[0]
                .to_string()
                .parse()
                .map_err(|_| value_error("an exponent".to_string()))?;
            let result: Ideal<K> = ideal.pow(n)?;
            exprs(&result, &result.groebner_basis(None)?)?
        }
        "equals" => boolean(ideal.equals(&other()?)?),
        "subset" => boolean(ideal.subset(&other()?)?),
        "hilbert_series" => ideal.hilbert_series(extra.first())?,
        "hilbert_polynomial" => ideal.hilbert_polynomial(extra.first())?,
        "dimension" => Basic::int(ideal.dimension()?),
        "degree" => Basic::integer(ideal.degree()?),
        "height" => Basic::int(ideal.height()?),
        "standard_monomials" => tuple(ideal.standard_monomials(None)?),
        "vector_space_dimension" => Basic::int(ideal.vector_space_dimension()? as i64),
        "multiplication_matrix" => {
            let rows: Vec<Vec<K::Elem>> = ideal.multiplication_matrix(&PolyLike::Expr(&extra[0]), None)?;
            tuple(
                rows.iter()
                    .map(|r| Ok(tuple(r.iter().map(|c| domain.to_expr(c)).collect::<PolyResult<_>>()?)))
                    .collect::<PolyResult<_>>()?,
            )
        }
        "minimal_polynomial" => dense_to_expr(
            &ideal.minimal_polynomial(&PolyLike::Expr(&extra[0]))?,
            &domain,
            &extra[1],
        )?,
        "univariate" => dense_to_expr(&ideal.univariate(&extra[0])?, &domain, &extra[0])?,
        "radical" => {
            let result: Ideal<K> = ideal.radical()?;
            exprs(&result, &result.groebner_basis(None)?)?
        }
        "is_radical" => boolean(ideal.is_radical()?),
        "is_primary" => boolean(ideal.is_primary()?),
        "equidimensional_parts" | "minimal_primes" | "associated_primes" => {
            let found: Vec<Ideal<K>> = match op {
                "equidimensional_parts" => ideal.equidimensional_parts()?,
                "minimal_primes" => ideal.minimal_primes()?,
                _ => ideal.associated_primes()?,
            };
            tuple(
                found
                    .iter()
                    .map(|i| exprs(i, &i.groebner_basis(None)?))
                    .collect::<PolyResult<_>>()?,
            )
        }
        "primary_decomposition" => {
            let found: Vec<(Ideal<K>, Ideal<K>)> = ideal.primary_decomposition()?;
            tuple(
                found
                    .iter()
                    .map(|(q, p)| {
                        Ok(tuple(vec![
                            exprs(q, &q.groebner_basis(None)?)?,
                            exprs(p, &p.groebner_basis(None)?)?,
                        ]))
                    })
                    .collect::<PolyResult<_>>()?,
            )
        }
        "is_maximal" => boolean(ideal.is_maximal()?),
        "is_prime" => boolean(ideal.is_prime()?),
        "is_zero_dimensional" => boolean(ideal.is_zero_dimensional()?),
        "is_whole_ring" => boolean(ideal.is_whole_ring()?),
        _ => return Err(value_error(format!("unknown ideal operation {}", op))),
    })
}
