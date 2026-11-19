//! Regular chains of polynomials with rational coefficients on the
//! expressions of the core, ported from
//! `sympy_extras.polys.regularchains.regularchain`: the expression face of
//! the chains of [`super::operations`] over the integers, and
//! `triangularize`. Not ported: `saturated_ideal` (the ideals),
//! `numerical_solutions` and `solutions` (mpmath).

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;
use std::sync::Arc;

use crate::core::basic::{Basic, Head};
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::domains::{IntegerRing, ZZ};
use crate::polys::rings::{MonomialOrder, PolyRing, Terms};

use super::operations::{Chain, Decomposer};
use super::recursive::{degree, initial, main_variable, normalize, ZPoly};

/// An equation: an expression which vanishes, or an equality (`_as_expr`).
fn as_expr(equation: &Basic) -> PolyResult<Basic> {
    match equation.head() {
        Head::Relational(crate::core::basic::RelOp::Eq) => {
            crate::core::add::sub(&equation.args()[0], &equation.args()[1])
        }
        Head::BooleanTrue | Head::BooleanFalse => {
            Err(PolyError::Type(format!("an equation is expected, got {}", equation)))
        }
        _ if crate::core::logic::is_boolean(equation) || crate::sets::sets::is_set(equation) => {
            Err(PolyError::Type(format!("an equation is expected, got {}", equation)))
        }
        _ => Ok(equation.clone()),
    }
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Wild(w) => w.symbol.name.clone(),
        _ => s.to_string(),
    }
}

fn sorted_by_name(symbols: impl IntoIterator<Item = Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().collect();
    v.sort_by_key(symbol_name);
    v
}

/// The polynomial over the integers of an equation: the numerator of a
/// polynomial with rational coefficients (`_polynomial`).
fn polynomial(ring: &Arc<PolyRing<IntegerRing>>, symbols: &[Basic], equation: &Basic) -> PolyResult<ZPoly> {
    let expr: Basic = as_expr(equation)?;
    let unknown: Vec<Basic> = sorted_by_name(expr.free_symbols().into_iter().filter(|s| !symbols.contains(s)));
    if !unknown.is_empty() {
        let names: Vec<String> = unknown.iter().map(|s| s.to_string()).collect();
        return Err(value_error(format!(
            "symbols which are not variables: [{}]",
            names.join(", ")
        )));
    }
    let poly = crate::polys::polytools::Poly::new(&expr, symbols)
        .map_err(|_| value_error(format!("not a polynomial in the variables: {}", expr)))?;
    let mut terms: Vec<(Vec<u32>, Rat)> = Vec::new();
    for (m, c) in poly.terms()? {
        match c.as_rational() {
            Some(v) => terms.push((m, v.clone())),
            None => {
                return Err(value_error(format!(
                    "not a polynomial with rational coefficients: {}",
                    expr
                )))
            }
        }
    }
    let mut denominator: Int = Int::ONE;
    for (_, c) in &terms {
        let d: Int = Int::from(c.denominator().clone());
        denominator = crate::core::intfunc::ilcm(&[denominator, d]);
    }
    let mut map: Terms<Int> = Terms::new();
    for (m, c) in terms {
        let scaled: Rat = c * Rat::from(denominator.clone());
        map.insert(m, scaled.numerator().clone());
    }
    Ok(ring.from_dict(map))
}

/// The expression of a polynomial in the actual symbols (`_expr`).
fn expr_of(p: &ZPoly, symbols: &[Basic]) -> PolyResult<Basic> {
    use crate::core::add::add;
    use crate::core::mul::mul;
    use crate::core::power::pow;
    let mut parts: Vec<Basic> = Vec::new();
    for (m, c) in &p.terms {
        let mut factors: Vec<Basic> = vec![Basic::integer(c.clone())];
        for (s, e) in symbols.iter().zip(m) {
            if *e > 1 {
                factors.push(pow(s, &Basic::integer(Int::from(*e)))?);
            } else if *e == 1 {
                factors.push(s.clone());
            }
        }
        parts.push(mul(&factors)?);
    }
    if parts.is_empty() {
        return Ok(Basic::zero());
    }
    add(&parts)
}

/// The given variables followed by the other symbols, as parameters
/// (`_variables`).
fn variables(equations: &[Basic], symbols: &[Basic]) -> PolyResult<Vec<Basic>> {
    for s in symbols {
        if !s.is_symbol() {
            return Err(PolyError::Type(format!("a symbol is expected, got {}", s)));
        }
    }
    let mut vars: Vec<Basic> = symbols.to_vec();
    let distinct: BTreeSet<Basic> = vars.iter().cloned().collect();
    if distinct.len() != vars.len() {
        return Err(value_error("repeated variables".to_string()));
    }
    let mut found: BTreeSet<Basic> = BTreeSet::new();
    for e in equations {
        found.extend(as_expr(e)?.free_symbols());
    }
    vars.extend(sorted_by_name(found.into_iter().filter(|s| !distinct.contains(s))));
    if vars.is_empty() {
        return Err(value_error("no variables".to_string()));
    }
    Ok(vars)
}

fn ring_of(symbols: &[Basic]) -> PolyResult<Arc<PolyRing<IntegerRing>>> {
    PolyRing::new(ZZ, symbols.iter().map(symbol_name).collect(), MonomialOrder::Lex)
}

/// A squarefree regular chain of polynomials with rational coefficients
/// (`RegularChain`): the variables, the greatest first, and the chain over
/// the integers with the decomposer which made it.
#[derive(Clone)]
pub struct RegularChain {
    pub symbols: Vec<Basic>,
    ring: Arc<PolyRing<IntegerRing>>,
    decomposer: Rc<RefCell<Decomposer>>,
    chain: Chain,
}

impl PartialEq for RegularChain {
    fn eq(&self, other: &RegularChain) -> bool {
        self.symbols == other.symbols && self.chain == other.chain
    }
}

impl std::fmt::Debug for RegularChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let polys: Vec<String> = match self.polys() {
            Ok(p) => p.iter().map(|e| e.to_string()).collect(),
            Err(_) => vec!["?".to_string()],
        };
        let symbols: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
        write!(f, "RegularChain([{}], {})", polys.join(", "), symbols.join(", "))
    }
}

impl RegularChain {
    /// The chain of `polys` (distinct main variables), checked to be a
    /// squarefree regular chain (`RegularChain(polys, *symbols)`).
    pub fn new(polys: &[Basic], symbols: &[Basic]) -> PolyResult<RegularChain> {
        let vars: Vec<Basic> = variables(polys, symbols)?;
        let ring = ring_of(&vars)?;
        let mut decomposer: Decomposer = Decomposer::new(Arc::clone(&ring), None);
        let mut elements: Vec<ZPoly> = Vec::new();
        for p in polys {
            elements.push(normalize(&polynomial(&ring, &vars, p)?));
        }
        let levels: Vec<usize> = elements.iter().map(main_variable).collect();
        if levels.iter().any(|l| *l == ring.ngens) {
            return Err(value_error("a constant polynomial".to_string()));
        }
        let distinct: BTreeSet<usize> = levels.iter().copied().collect();
        if distinct.len() != levels.len() {
            return Err(value_error("two polynomials with the same main variable".to_string()));
        }
        let mut pairs: Vec<(usize, ZPoly)> = levels.into_iter().zip(elements).collect();
        pairs.sort_by_key(|(level, _)| std::cmp::Reverse(*level));
        let mut chain: Chain = decomposer.empty.clone();
        for (level, p) in pairs {
            if !decomposer.is_regular(&initial(&p), &chain)? {
                return Err(value_error(format!(
                    "the initial {} is a zero divisor modulo the polynomials below",
                    expr_of(&initial(&p), &vars)?
                )));
            }
            let added: Vec<ZPoly> = decomposer
                .add(&chain, &p)?
                .iter()
                .filter_map(|t| t.get(level).cloned())
                .collect();
            if added.len() != 1 || degree(&added[0], level) != degree(&p, level) {
                return Err(value_error(format!(
                    "{} is not squarefree modulo the polynomials below",
                    expr_of(&p, &vars)?
                )));
            }
            chain = chain.with_poly(level, p);
        }
        Ok(RegularChain {
            symbols: vars,
            ring,
            decomposer: Rc::new(RefCell::new(decomposer)),
            chain,
        })
    }

    fn wrap(&self, chain: Chain) -> RegularChain {
        RegularChain {
            symbols: self.symbols.clone(),
            ring: Arc::clone(&self.ring),
            decomposer: Rc::clone(&self.decomposer),
            chain,
        }
    }

    fn element(&self, f: &Basic) -> PolyResult<ZPoly> {
        polynomial(&self.ring, &self.symbols, f)
    }

    fn expr(&self, p: &ZPoly) -> PolyResult<Basic> {
        expr_of(p, &self.symbols)
    }

    /// The chain of polynomials over the integers.
    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    /// The polynomials, by increasing main variable.
    pub fn polys(&self) -> PolyResult<Vec<Basic>> {
        self.chain.elements().iter().map(|p| self.expr(p)).collect()
    }

    /// The same chain in a ring with more variables, which are free; the
    /// variables of the chain must come in the same order among the new
    /// ones (`with_symbols`).
    pub fn with_symbols(&self, symbols: &[Basic]) -> PolyResult<RegularChain> {
        for (i, s) in symbols.iter().enumerate() {
            if symbols[..i].contains(s) {
                return Err(value_error("repeated variables".to_string()));
            }
        }
        let kept: Vec<Basic> = symbols.iter().filter(|s| self.symbols.contains(s)).cloned().collect();
        if kept != self.symbols {
            let names: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
            return Err(value_error(format!(
                "the variables ({}) must be kept, in this order",
                names.join(", ")
            )));
        }
        let ring = ring_of(symbols)?;
        let decomposer: Decomposer = Decomposer::new(Arc::clone(&ring), None);
        let mut polys: Vec<ZPoly> = Vec::new();
        for p in self.polys()? {
            polys.push(polynomial(&ring, symbols, &p)?);
        }
        let chain: Chain = decomposer.empty.union(&polys);
        Ok(RegularChain {
            symbols: symbols.to_vec(),
            ring,
            decomposer: Rc::new(RefCell::new(decomposer)),
            chain,
        })
    }

    /// The variables of the ring of the chain.
    pub fn symbols(&self) -> &[Basic] {
        &self.symbols
    }

    /// The main variables of the polynomials, in the same order.
    pub fn main_variables(&self) -> Vec<Basic> {
        self.chain
            .elements()
            .iter()
            .map(|p| self.symbols[main_variable(p)].clone())
            .collect()
    }

    /// The variables which are the main variable of no polynomial.
    pub fn free_variables(&self) -> Vec<Basic> {
        self.symbols
            .iter()
            .enumerate()
            .filter(|(i, _)| self.chain.get(*i).is_none())
            .map(|(_, s)| s.clone())
            .collect()
    }

    /// The leading coefficients in the main variables.
    pub fn initials(&self) -> PolyResult<Vec<Basic>> {
        self.chain.elements().iter().map(|p| self.expr(&initial(p))).collect()
    }

    /// The number of polynomials.
    pub fn height(&self) -> usize {
        self.chain.height
    }

    /// The number of free variables.
    pub fn dimension(&self) -> usize {
        self.symbols.len() - self.chain.height
    }

    /// The saturated ideal `(T) : h^oo`, `h` the product of the initials
    /// (`saturated_ideal`).
    pub fn saturated_ideal(&self) -> PolyResult<crate::polys::ideals::Ideal<crate::polys::domains::RationalField>> {
        use crate::polys::ideals::{Ideal, IdealOrPoly, PolyLike};
        let polys: Vec<Basic> = self.polys()?;
        let gens: Vec<Basic> = if polys.is_empty() {
            vec![Basic::zero()]
        } else {
            polys.clone()
        };
        let ideal = Ideal::new(&gens, &self.symbols, crate::polys::domains::QQ, MonomialOrder::Grevlex)?;
        if polys.is_empty() {
            return Ok(ideal);
        }
        let product: Basic = crate::core::mul::mul(&self.initials()?)?;
        ideal.saturate(&IdealOrPoly::Poly(PolyLike::Expr(&product)))
    }

    /// The generator of the saturated ideal in `v` alone (`_univariate`).
    pub fn univariate(&self, v: &Basic) -> PolyResult<Basic> {
        let rep: Vec<Rat> = self.saturated_ideal()?.univariate(v)?;
        let d: usize = rep.len().saturating_sub(1);
        let mut terms: Vec<Basic> = Vec::new();
        for (k, c) in rep.iter().enumerate() {
            terms.push(crate::core::mul::mul(&[
                Basic::from_rational(c.clone()),
                crate::core::power::pow(v, &Basic::int((d - k) as i64))?,
            ])?);
        }
        crate::core::add::add(&terms)
    }

    /// The points of a chain without free variables, exactly
    /// (`solutions`): see `solutions::exact_solutions`.
    pub fn solutions(&self, real: bool) -> PolyResult<Vec<super::solutions::Point>> {
        if self.dimension() > 0 {
            return Err(value_error(format!(
                "the chain has the free variables {:?}: its solutions are not isolated",
                self.free_variables()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
            )));
        }
        let fallback = |v: &Basic| self.univariate(v);
        super::solutions::exact_solutions(&self.polys()?, &self.main_variables(), real, Some(&fallback))
    }

    /// The product of the degrees in the main variables.
    pub fn degree(&self) -> i64 {
        self.chain
            .elements()
            .iter()
            .map(|p| degree(p, main_variable(p)))
            .product()
    }

    /// The pseudo-remainder of `f` by the chain, primitive over the
    /// integers (`reduce`).
    pub fn reduce(&self, f: &Basic) -> PolyResult<Basic> {
        let r: ZPoly = self.decomposer.borrow().reduce(&self.element(f)?, &self.chain)?;
        self.expr(&r)
    }

    /// Whether `f` is in the saturated ideal (`contains`).
    pub fn contains(&self, f: &Basic) -> PolyResult<bool> {
        Ok(self
            .decomposer
            .borrow()
            .reduce(&self.element(f)?, &self.chain)?
            .is_zero())
    }

    /// Whether `f` is not a zero divisor modulo the saturated ideal.
    pub fn is_regular(&self, f: &Basic) -> PolyResult<bool> {
        let p: ZPoly = self.element(f)?;
        self.decomposer.borrow_mut().is_regular(&p, &self.chain)
    }

    /// The chains modulo which `f` is zero and those modulo which it is
    /// regular (`regularize`).
    pub fn regularize(&self, f: &Basic) -> PolyResult<(Vec<RegularChain>, Vec<RegularChain>)> {
        let p: ZPoly = self.element(f)?;
        let parts = self.decomposer.borrow_mut().regularize(&p, &self.chain)?;
        let (mut zero, mut regular) = (Vec::new(), Vec::new());
        for (chain, vanishes) in parts {
            if vanishes {
                zero.push(self.wrap(chain));
            } else {
                regular.push(self.wrap(chain));
            }
        }
        Ok((zero, regular))
    }

    /// Chains covering the zeros of `f` in the quasi-component (`intersect`).
    pub fn intersect(&self, f: &Basic) -> PolyResult<Vec<RegularChain>> {
        let p: ZPoly = self.element(f)?;
        let mut d = self.decomposer.borrow_mut();
        let chains: Vec<Chain> = d.intersect(&p, &self.chain)?;
        let chains: Vec<Chain> = d.irredundant(&chains, false)?;
        drop(d);
        Ok(chains.into_iter().map(|c| self.wrap(c)).collect())
    }
}

/// A triangular decomposition of the zeros of polynomial equations
/// (`triangularize`): the variables the greatest first, the other symbols
/// after them as parameters; `kalkbrener` for the union of the closures
/// in place of the quasi-components.
pub fn triangularize(
    equations: &[Basic],
    symbols: &[Basic],
    inequations: &[Basic],
    kalkbrener: bool,
) -> PolyResult<Vec<RegularChain>> {
    let mut all: Vec<Basic> = equations.to_vec();
    all.extend(inequations.iter().cloned());
    let vars: Vec<Basic> = variables(&all, symbols)?;
    let ring = ring_of(&vars)?;
    let mut polys: Vec<ZPoly> = Vec::new();
    for p in equations {
        polys.push(polynomial(&ring, &vars, p)?);
    }
    let mut nonzero: Vec<ZPoly> = Vec::new();
    for p in inequations {
        nonzero.push(polynomial(&ring, &vars, p)?);
    }
    if nonzero.iter().any(|h| h.is_zero()) {
        return Ok(Vec::new());
    }
    let bound: Option<usize> = if kalkbrener {
        Some(polys.iter().filter(|p| !p.is_zero()).count())
    } else {
        None
    };
    let chains: Vec<Chain> = Decomposer::new(Arc::clone(&ring), bound).triangularize(&polys, &nonzero)?;
    // the chains which are returned keep every chain of their own operations
    let decomposer = Rc::new(RefCell::new(Decomposer::new(Arc::clone(&ring), None)));
    Ok(chains
        .into_iter()
        .map(|chain| RegularChain {
            symbols: vars.clone(),
            ring: Arc::clone(&ring),
            decomposer: Rc::clone(&decomposer),
            chain,
        })
        .collect())
}
