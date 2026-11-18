//! The Rosenfeld–Gröbner algorithm, ported from
//! `sympy_extras.polys.differential.rosenfeld_groebner`: the radical of a
//! differential ideal as an intersection of regular differential ideals
//! `[A] : H**oo` (Boulier, Lazard, Ollivier, Petitot), by Ritt's process
//! with splittings on the initials and separants, the factors of the
//! equations split one at a time; membership by Rosenfeld's lemma (the
//! partial remainder modulo the squarefree regular chains of `(A) : H**oo`).

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::core::basic::{Basic, Head};
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};

use super::super::domains::{Domain, RationalField, SparseFractionField};
use super::super::ideals::Ideal;
use super::super::regularchains::regularchain::{triangularize, RegularChain};
use super::super::rings::Terms as SymbolTerms;
use super::polynomial::{DifferentialPolynomial, Rank};
use super::ring::{Coefficient, DifferentialRing, Jet, Terms};

type Field = SparseFractionField<RationalField>;

/// Symbols standing for the derivatives in the polynomial rings
/// (`JetSymbols`).
#[derive(Debug, Clone)]
pub struct JetSymbols {
    ring: Rc<DifferentialRing>,
    symbols: RefCell<Vec<(Jet, Basic)>>,
}

impl JetSymbols {
    pub fn new(ring: Rc<DifferentialRing>) -> JetSymbols {
        JetSymbols {
            ring,
            symbols: RefCell::new(Vec::new()),
        }
    }

    /// The symbol of a derivative.
    pub fn symbol(&self, jet: &Jet) -> Basic {
        if let Some((_, s)) = self.symbols.borrow().iter().find(|(j, _)| j == jet) {
            return s.clone();
        }
        let (a, orders) = jet;
        let name: String = match self.ring.functions[*a as usize].head() {
            Head::Undefined(u) => u.name.clone(),
            _ => self.ring.functions[*a as usize].to_string(),
        };
        let digits: String = orders.iter().map(|j| j.to_string()).collect();
        let s: Basic = Dummy::new(&format!("{}_{}", name, digits), &[]).0;
        self.symbols.borrow_mut().push((jet.clone(), s.clone()));
        s
    }

    /// The polynomial as an expression in the symbols.
    pub fn expr(&self, p: &DifferentialPolynomial) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (monomial, c) in p.terms.iter() {
            let mut factors: Vec<Basic> = vec![self.ring.integer_expr(c, None)?];
            for (jet, e) in monomial {
                let s: Basic = self.symbol(jet);
                factors.push(if *e > 1 {
                    crate::core::power::pow(&s, &Basic::int(*e as i64))?
                } else {
                    s
                });
            }
            parts.push(crate::core::mul::mul(&factors)?);
        }
        if parts.is_empty() {
            return Ok(Basic::zero());
        }
        crate::core::add::add(&parts)
    }

    /// The differential polynomial of an expression in the symbols.
    pub fn polynomial(&self, expr: &Basic) -> PolyResult<DifferentialPolynomial> {
        let jets: HashMap<Basic, Jet> = self
            .symbols
            .borrow()
            .iter()
            .map(|(j, s)| (s.clone(), j.clone()))
            .collect();
        let terms: Terms = self.ring.integral(&self.ring.terms_of(expr, &jets)?);
        Ok(DifferentialPolynomial::new(Rc::clone(&self.ring), terms))
    }

    /// The terms of a polynomial in the derivatives `jets` (after `shift`
    /// other variables), with coefficients in the field.
    fn terms(&self, p: &DifferentialPolynomial, jets: &[Jet], shift: usize) -> SymbolTerms<Coefficient> {
        let position: HashMap<&Jet, usize> = jets.iter().enumerate().map(|(k, j)| (j, shift + k)).collect();
        let mut result: SymbolTerms<Coefficient> = SymbolTerms::new();
        for (monomial, c) in p.terms.iter() {
            let mut exponents: Vec<u32> = vec![0; shift + jets.len()];
            for (jet, e) in monomial {
                exponents[position[jet]] = *e;
            }
            result.insert(exponents, self.ring.field_element(c));
        }
        result
    }

    /// The symbols created so far.
    pub fn known(&self) -> Vec<Basic> {
        self.symbols.borrow().iter().map(|(_, s)| s.clone()).collect()
    }
}

fn sorted_jets(ring: &DifferentialRing, jets: HashSet<Jet>) -> Vec<Jet> {
    let mut ordered: Vec<Jet> = jets.into_iter().collect();
    ordered.sort_by_key(|j| std::cmp::Reverse(ring.rank_key(j)));
    ordered
}

/// A regular differential system `A = 0`, `H != 0`: a component of the
/// decomposition (`RegularDifferentialSystem`).
#[derive(Debug, Clone)]
pub struct RegularDifferentialSystem {
    pub ring: Rc<DifferentialRing>,
    pub chain: Vec<DifferentialPolynomial>,
    pub nonzero: Vec<DifferentialPolynomial>,
    symbols: JetSymbols,
    basis: RefCell<Option<Vec<Basic>>>,
    chains: RefCell<Option<Vec<RegularChain>>>,
}

impl RegularDifferentialSystem {
    pub fn new(
        ring: Rc<DifferentialRing>,
        chain: Vec<DifferentialPolynomial>,
        nonzero: Vec<DifferentialPolynomial>,
    ) -> Self {
        RegularDifferentialSystem {
            symbols: JetSymbols::new(Rc::clone(&ring)),
            ring,
            chain,
            nonzero,
            basis: RefCell::new(None),
            chains: RefCell::new(None),
        }
    }

    /// The expressions which vanish, by increasing leader (`equations`).
    pub fn equations(&self) -> PolyResult<Vec<Basic>> {
        let hidden: Vec<Jet> = self.ring.constancy();
        let mut result: Vec<Basic> = Vec::new();
        for p in &self.chain {
            let leader: Jet = p.leader()?;
            if !hidden.iter().any(|jet| self.ring.is_derivative_of(&leader, jet)) {
                result.push(p.to_expr()?);
            }
        }
        Ok(result)
    }

    /// The expressions which do not vanish.
    pub fn inequations(&self) -> PolyResult<Vec<Basic>> {
        self.nonzero.iter().map(|p| p.to_expr()).collect()
    }

    /// The leaders of the equations.
    pub fn leaders(&self) -> PolyResult<Vec<Basic>> {
        self.chain.iter().map(|p| p.leader_expr()).collect()
    }

    fn jets(&self) -> HashSet<Jet> {
        let mut jets: HashSet<Jet> = HashSet::new();
        for p in self.chain.iter().chain(&self.nonzero) {
            jets.extend(p.jets());
        }
        jets
    }

    /// A Gröbner basis of `(A) : H**oo` in symbols standing for the
    /// derivatives (`saturation`).
    pub fn saturation(&self) -> PolyResult<Vec<Basic>> {
        if let Some(found) = self.basis.borrow().as_ref() {
            return Ok(found.clone());
        }
        let k: &Field = &self.ring.domain;
        let ordered: Vec<Jet> = sorted_jets(&self.ring, self.jets());
        let symbols: Vec<Basic> = ordered.iter().map(|j| self.symbols.symbol(j)).collect();
        let order = super::super::rings::MonomialOrder::Grevlex;
        let ideal: Ideal<Field> = if self.nonzero.is_empty() {
            let terms: Vec<SymbolTerms<Coefficient>> =
                self.chain.iter().map(|p| self.symbols.terms(p, &ordered, 0)).collect();
            Ideal::from_terms(&terms, &symbols, k.clone(), order)?
        } else {
            let one: Terms = Terms::from_iter([(Vec::new(), self.ring.integers.ring.one())]);
            let mut product: DifferentialPolynomial = DifferentialPolynomial::new(Rc::clone(&self.ring), one);
            for h in &self.nonzero {
                product = product.mul(h);
            }
            let mut last: SymbolTerms<Coefficient> = SymbolTerms::new();
            for (m, c) in self.symbols.terms(&product, &ordered, 1) {
                let mut key: Vec<u32> = m.clone();
                key[0] = 1;
                last.insert(key, k.neg(&c));
            }
            let origin: Vec<u32> = vec![0; ordered.len() + 1];
            let value: Coefficient = match last.get(&origin) {
                Some(c) => k.add(c, &k.one()),
                None => k.one(),
            };
            if k.is_zero(&value) {
                last.remove(&origin);
            } else {
                last.insert(origin, value);
            }
            let t: Basic = Dummy::new("t", &[]).0;
            let mut gens: Vec<SymbolTerms<Coefficient>> =
                self.chain.iter().map(|p| self.symbols.terms(p, &ordered, 1)).collect();
            gens.push(last);
            let mut all: Vec<Basic> = vec![t.clone()];
            all.extend(symbols.iter().cloned());
            Ideal::from_terms(&gens, &all, k.clone(), order)?.eliminate(&[t])?
        };
        let exprs: Vec<Basic> = ideal.exprs()?;
        *self.basis.borrow_mut() = Some(exprs.clone());
        Ok(exprs)
    }

    fn ordered(&self, jets: HashSet<Jet>) -> Vec<Basic> {
        let mut symbols: Vec<Basic> = sorted_jets(&self.ring, jets)
            .iter()
            .map(|j| self.symbols.symbol(j))
            .collect();
        symbols.extend(self.ring.variables.iter().cloned());
        symbols.extend(self.ring.parameters.iter().cloned());
        symbols
    }

    /// Squarefree regular chains with `(A) : H**oo` the intersection of
    /// their saturated ideals (`regular_chains`).
    pub fn regular_chains(&self) -> PolyResult<Vec<RegularChain>> {
        if let Some(found) = self.chains.borrow().as_ref() {
            return Ok(found.clone());
        }
        let coefficients: Vec<Basic> = self
            .ring
            .variables
            .iter()
            .chain(&self.ring.parameters)
            .cloned()
            .collect();
        let equations: Vec<Basic> = self
            .chain
            .iter()
            .map(|p| self.symbols.expr(p))
            .collect::<PolyResult<_>>()?;
        let inequations: Vec<Basic> = self
            .nonzero
            .iter()
            .map(|h| self.symbols.expr(h))
            .collect::<PolyResult<_>>()?;
        let symbols: Vec<Basic> = self.ordered(self.jets());
        let chains: Vec<RegularChain> = triangularize(&equations, &symbols, &inequations, true)?
            .into_iter()
            .filter(|c| !c.main_variables().iter().any(|v| coefficients.contains(v)))
            .collect();
        *self.chains.borrow_mut() = Some(chains.clone());
        Ok(chains)
    }

    /// Whether the system has solutions: `1` is not in `(A) : H**oo`.
    pub fn is_consistent(&self) -> PolyResult<bool> {
        Ok(!self
            .saturation()?
            .iter()
            .any(|g| g.free_symbols().is_empty() && *g != Basic::zero()))
    }

    /// Ritt's remainder of a differential polynomial by the equations.
    pub fn remainder(&self, expr: &Basic) -> PolyResult<Basic> {
        DifferentialPolynomial::from_expr(Rc::clone(&self.ring), expr)?
            .remainder(&self.chain)?
            .to_expr()
    }

    /// Whether the differential polynomial lies in `[A] : H**oo`.
    pub fn contains(&self, expr: &Basic) -> PolyResult<bool> {
        let p: DifferentialPolynomial = DifferentialPolynomial::from_expr(Rc::clone(&self.ring), expr)?;
        if !p.remainder(&self.chain)?.is_nonzero() {
            return Ok(true);
        }
        let partial: DifferentialPolynomial = p.partial_remainder(&self.chain)?;
        let mut jets: HashSet<Jet> = partial.jets();
        jets.extend(self.jets());
        let target: Basic = self.symbols.expr(&partial)?;
        let symbols: Vec<Basic> = self.ordered(jets);
        for chain in self.regular_chains()? {
            if !chain.with_symbols(&symbols)?.contains(&target)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The radical differential ideal of a system, as the intersection of the
/// regular differential ideals of its components (`RadicalDifferentialIdeal`).
#[derive(Debug, Clone)]
pub struct RadicalDifferentialIdeal {
    pub ring: Rc<DifferentialRing>,
    pub components: Vec<RegularDifferentialSystem>,
}

impl RadicalDifferentialIdeal {
    pub fn is_consistent(&self) -> bool {
        !self.components.is_empty()
    }

    /// Whether the differential polynomial vanishes on every solution.
    pub fn contains(&self, expr: &Basic) -> PolyResult<bool> {
        for component in &self.components {
            if !component.contains(expr)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// An autoreduced subset of lowest rank (`_characteristic_set`).
fn characteristic_set(polynomials: &[DifferentialPolynomial]) -> PolyResult<Vec<DifferentialPolynomial>> {
    let mut keyed: Vec<((Rank, usize), &DifferentialPolynomial)> = Vec::new();
    for p in polynomials {
        keyed.push(((p.rank()?, p.terms.len()), p));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    let mut chain: Vec<DifferentialPolynomial> = Vec::new();
    for (_, p) in keyed {
        let mut reduced: bool = true;
        for a in &chain {
            if !p.is_reduced(a)? {
                reduced = false;
                break;
            }
        }
        if reduced {
            chain.push(p.clone());
        }
    }
    Ok(chain)
}

/// The cross derivatives of the pairs with leaders which are derivatives
/// of the same function (`_delta_polynomials`).
fn delta_polynomials(chain: &[DifferentialPolynomial]) -> PolyResult<Vec<DifferentialPolynomial>> {
    let mut result: Vec<DifferentialPolynomial> = Vec::new();
    for (k, a) in chain.iter().enumerate() {
        for b in &chain[k + 1..] {
            let (v, w) = (a.leader()?, b.leader()?);
            if v.0 != w.0 {
                continue;
            }
            let common: Vec<u32> = v.1.iter().zip(&w.1).map(|(i, j)| *i.max(j)).collect();
            let first: DifferentialPolynomial =
                a.prolonged(&common.iter().zip(&v.1).map(|(c, i)| c - i).collect::<Vec<u32>>());
            let second: DifferentialPolynomial =
                b.prolonged(&common.iter().zip(&w.1).map(|(c, j)| c - j).collect::<Vec<u32>>());
            result.push(b.separant()?.mul(&first).sub(&a.separant()?.mul(&second)));
        }
    }
    Ok(result)
}

fn distinct(polynomials: &[DifferentialPolynomial]) -> Vec<DifferentialPolynomial> {
    let mut result: Vec<DifferentialPolynomial> = Vec::new();
    for p in polynomials {
        if p.is_nonzero() && !result.contains(p) {
            result.push(p.clone());
        }
    }
    result
}

fn is_polynomial_error(e: &PolyError) -> bool {
    matches!(e, PolyError::Polys(..))
}

/// The irreducible factors of differential polynomials, remembered
/// (`_Factorizer`).
struct Factorizer<'a> {
    symbols: &'a JetSymbols,
    known: HashMap<DifferentialPolynomial, Vec<DifferentialPolynomial>>,
}

impl<'a> Factorizer<'a> {
    fn factors(&mut self, p: &DifferentialPolynomial) -> PolyResult<Vec<DifferentialPolynomial>> {
        if let Some(found) = self.known.get(p) {
            return Ok(found.clone());
        }
        let mut result: Vec<DifferentialPolynomial> = Vec::new();
        if !p.is_ground() {
            match super::super::polytools::factor_list(&self.symbols.expr(p)?, &[]) {
                Ok((_, found)) => {
                    for (f, _) in found {
                        let q: DifferentialPolynomial = self.symbols.polynomial(&f)?.primitive();
                        if !q.is_ground() && !result.contains(&q) {
                            result.push(q);
                        }
                    }
                }
                Err(e) if is_polynomial_error(&e) => result = vec![p.clone()],
                Err(e) => return Err(e),
            }
        }
        self.known.insert(p.clone(), result.clone());
        for q in &result {
            self.known.entry(q.clone()).or_insert_with(|| vec![q.clone()]);
        }
        Ok(result)
    }

    fn nonzero(&mut self, polynomials: &[DifferentialPolynomial]) -> PolyResult<Vec<DifferentialPolynomial>> {
        let mut result: Vec<DifferentialPolynomial> = Vec::new();
        for h in polynomials {
            for f in self.factors(h)? {
                if !result.contains(&f) {
                    result.push(f);
                }
            }
        }
        Ok(result)
    }
}

/// The equation without its factors which are free of its leader and known
/// not to vanish (`_without_factors`).
fn without_factors(
    symbols: &JetSymbols,
    a: &DifferentialPolynomial,
    nonzero: &[DifferentialPolynomial],
) -> PolyResult<DifferentialPolynomial> {
    let leader: Jet = a.leader()?;
    let expr: Basic = symbols.expr(a)?;
    let known: Vec<Basic> = symbols.known();
    let mut gens: Vec<Basic> = expr.free_symbols().into_iter().filter(|s| known.contains(s)).collect();
    gens.sort_by_key(|s| match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    });
    let factors = match super::super::polytools::factor_list(&expr, &gens) {
        Ok((_, f)) => f,
        Err(e) if is_polynomial_error(&e) => return Ok(a.clone()),
        Err(e) => return Err(e),
    };
    let mut kept: Option<DifferentialPolynomial> = None;
    let mut dropped: bool = false;
    for (f, e) in factors {
        let q: DifferentialPolynomial = symbols.polynomial(&f)?.primitive();
        if !q.jets().contains(&leader) && (nonzero.contains(&q) || q.is_ground()) {
            dropped = dropped || !q.is_ground();
            continue;
        }
        let exponent: u32 = e
            .as_rational()
            .filter(|v| v.is_int())
            .and_then(|v| u32::try_from(v.numerator()).ok())
            .ok_or_else(|| PolyError::Type(format!("an integer multiplicity is expected, got {}", e)))?;
        let power: DifferentialPolynomial = q.pow(exponent);
        kept = Some(match kept {
            None => power,
            Some(k) => k.mul(&power),
        });
    }
    match kept {
        Some(k) if dropped => Ok(k.primitive()),
        _ => Ok(a.clone()),
    }
}

/// The radical of the differential ideal of a polynomial system of
/// differential equations, as an intersection of regular differential
/// ideals (`rosenfeld_groebner`).
pub fn rosenfeld_groebner(
    equations: &[Basic],
    functions: &[Basic],
    variables: Option<&[Basic]>,
    ranking: Option<&[Vec<Basic>]>,
    inequations: &[Basic],
) -> PolyResult<RadicalDifferentialIdeal> {
    let probe: DifferentialRing = DifferentialRing::new(functions, variables, ranking, &[])?;
    let mut parameters: Vec<Basic> = Vec::new();
    for e in equations.iter().chain(inequations) {
        for s in e.free_symbols() {
            if !probe.variables.contains(&s) && !parameters.contains(&s) {
                parameters.push(s);
            }
        }
    }
    parameters.sort_by_key(|s| match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    });
    let ring: Rc<DifferentialRing> = Rc::new(DifferentialRing::new(functions, variables, ranking, &parameters)?);
    let symbols: JetSymbols = JetSymbols::new(Rc::clone(&ring));
    let one = ring.integers.ring.one();
    let mut start: Vec<DifferentialPolynomial> = Vec::new();
    for e in equations {
        start.push(DifferentialPolynomial::from_expr(Rc::clone(&ring), e)?.primitive());
    }
    for jet in ring.constancy() {
        start.push(DifferentialPolynomial::new(
            Rc::clone(&ring),
            Terms::from_iter([(vec![(jet, 1)], one.clone())]),
        ));
    }
    let mut avoid: Vec<DifferentialPolynomial> = Vec::new();
    for e in inequations {
        avoid.push(DifferentialPolynomial::from_expr(Rc::clone(&ring), e)?.primitive());
    }
    if avoid.iter().any(|q| !q.is_nonzero()) {
        return Ok(RadicalDifferentialIdeal {
            ring,
            components: Vec::new(),
        });
    }
    let mut factorizer: Factorizer = Factorizer {
        symbols: &symbols,
        known: HashMap::new(),
    };
    let initial_nonzero: Vec<DifferentialPolynomial> = factorizer.nonzero(&avoid)?;
    let mut todo: Vec<(Vec<DifferentialPolynomial>, Vec<DifferentialPolynomial>)> =
        vec![(distinct(&start), initial_nonzero)];
    let mut components: Vec<RegularDifferentialSystem> = Vec::new();
    while let Some((p_set, h_set)) = todo.pop() {
        if p_set.iter().any(|p| p.is_ground()) {
            continue;
        }
        let mut split: bool = false;
        for (k, p) in p_set.iter().enumerate() {
            let factors: Vec<DifferentialPolynomial> = factorizer
                .factors(p)?
                .into_iter()
                .filter(|f| !h_set.contains(f))
                .collect();
            if factors.len() == 1 && factors[0] == *p {
                continue;
            }
            let mut before: Vec<DifferentialPolynomial> = Vec::new();
            for f in factors {
                let mut replaced: Vec<DifferentialPolynomial> = p_set[..k].to_vec();
                replaced.push(f.clone());
                replaced.extend(p_set[k + 1..].iter().cloned());
                let mut nonzero: Vec<DifferentialPolynomial> = h_set.clone();
                nonzero.extend(before.iter().cloned());
                todo.push((distinct(&replaced), distinct(&nonzero)));
                before.push(f);
            }
            split = true;
            break;
        }
        if split {
            continue;
        }
        let chain: Vec<DifferentialPolynomial> = characteristic_set(&p_set)?;
        let mut others: Vec<DifferentialPolynomial> = p_set.iter().filter(|p| !chain.contains(p)).cloned().collect();
        others.extend(delta_polynomials(&chain)?);
        let remainders: Vec<DifferentialPolynomial> =
            others.iter().map(|p| p.remainder(&chain)).collect::<PolyResult<_>>()?;
        let r_set: Vec<DifferentialPolynomial> = distinct(&remainders);
        let mut initials_separants: Vec<DifferentialPolynomial> = Vec::new();
        for a in &chain {
            initials_separants.push(a.initial()?);
            initials_separants.push(a.separant()?);
        }
        let splitting: Vec<DifferentialPolynomial> = factorizer.nonzero(&initials_separants)?;
        let mut before: Vec<DifferentialPolynomial> = Vec::new();
        for h in &splitting {
            if !h_set.contains(h) {
                let mut more: Vec<DifferentialPolynomial> = p_set.clone();
                more.push(h.clone());
                let mut nonzero: Vec<DifferentialPolynomial> = h_set.clone();
                nonzero.extend(before.iter().cloned());
                todo.push((distinct(&more), distinct(&nonzero)));
                before.push(h.clone());
            }
        }
        let mut vanishing: bool = false;
        for h in &h_set {
            if !h.remainder(&chain)?.is_nonzero() {
                vanishing = true;
                break;
            }
        }
        if vanishing {
            continue;
        }
        if !r_set.is_empty() {
            if !r_set.iter().any(|r| r.is_ground()) {
                let mut more: Vec<DifferentialPolynomial> = chain.clone();
                more.extend(r_set.iter().cloned());
                let mut nonzero: Vec<DifferentialPolynomial> = h_set.clone();
                nonzero.extend(splitting.iter().cloned());
                todo.push((distinct(&more), distinct(&nonzero)));
            }
            continue;
        }
        let mut partials: Vec<DifferentialPolynomial> = Vec::new();
        for h in h_set.iter().chain(&splitting) {
            partials.push(h.partial_remainder(&chain)?);
        }
        let mut nonzero: Vec<DifferentialPolynomial> = Vec::new();
        for h in factorizer.nonzero(&partials)? {
            if !h.remainder(&chain)?.is_ground() {
                nonzero.push(h);
            }
        }
        let mut reduced: Vec<(Rank, DifferentialPolynomial)> = Vec::new();
        for a in &chain {
            let q: DifferentialPolynomial = without_factors(&symbols, a, &nonzero)?;
            reduced.push((q.rank()?, q));
        }
        reduced.sort_by(|a, b| a.0.cmp(&b.0));
        let component: RegularDifferentialSystem =
            RegularDifferentialSystem::new(Rc::clone(&ring), reduced.into_iter().map(|(_, q)| q).collect(), nonzero);
        if component.is_consistent()? {
            components.push(component);
        }
    }
    let mut keyed: Vec<((i64, Vec<Rank>), RegularDifferentialSystem)> = Vec::new();
    for c in components {
        let ranks: Vec<Rank> = c.chain.iter().map(|p| p.rank()).collect::<PolyResult<_>>()?;
        keyed.push(((-(c.chain.len() as i64), ranks), c));
    }
    keyed.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(RadicalDifferentialIdeal {
        ring,
        components: keyed.into_iter().map(|(_, c)| c).collect(),
    })
}
