//! Port of `sympy_extras/polys/differential/ring.py`: differential
//! polynomial rings `K{u_1, ..., u_m}` over `K = QQ(p)(x)` with the
//! derivations `d/dx_i`, the derivatives as [`Jet`]s (the index of the
//! function and the orders), the block rankings (elimination between the
//! blocks, orderly inside), and the conversion of expressions of the core.
//!
//! The dicts of the Python module are insertion-ordered maps
//! ([`OrderedMap`]), as their order reaches the results. An `Equality`
//! (not in the Rust core) is given as `lhs - rhs`.

// a node caches its keys in `OnceLock`s that the hash never reads
#![allow(clippy::mutable_key_type)]

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::core::add::add;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is_integer_class;
use crate::core::function::{derivative_counts, derivative_node, doit};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::core::sorting::sort_key;
use crate::error::{zero_division, PolyError, PolyResult};
use crate::numbers::*;
use crate::orderedmap::OrderedMap;
use crate::polys::domains::domain::Domain;
use crate::polys::domains::integerring::{IntegerRing, ZZ};
use crate::polys::domains::rationalfield::{RationalField, QQ};
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;
use crate::polys::rings::{Monomial, PolyElement, Terms as RingTerms};

/// A derivative: the index of the function (`-1` for the unit) and the
/// orders in the variables.
pub type Jet = (i64, Monomial);
/// A product of powers of derivatives, sorted by derivative.
pub type JetMonomial = Vec<(Jet, u32)>;
/// An element of the field of coefficients `QQ(p, x)`.
pub type Coefficient = SparseFracElement<RationalField>;
/// An element of `ZZ[p, x]`, the coefficients of the differential
/// polynomials.
pub type IntegerCoefficient = PolyElement<IntegerRing>;
/// A differential polynomial with coefficients in the field.
pub type FieldTerms = OrderedMap<JetMonomial, Coefficient>;
/// A differential polynomial with coefficients in `ZZ[p, x]`.
pub type Terms = OrderedMap<JetMonomial, IntegerCoefficient>;

/// The pseudo derivative standing for the constant `1` in a linear form
/// (`unit_jet`).
pub fn unit_jet(n: usize) -> Jet {
    (-1, vec![0; n])
}

/// The product of two monomials in the derivatives (`monomial_product`).
pub fn monomial_product(m: &JetMonomial, k: &JetMonomial) -> JetMonomial {
    if m.is_empty() {
        return k.clone();
    }
    if k.is_empty() {
        return m.clone();
    }
    let mut powers: std::collections::BTreeMap<Jet, u32> = m.iter().cloned().collect();
    for (jet, e) in k {
        *powers.entry(jet.clone()).or_insert(0) += e;
    }
    powers.into_iter().collect()
}

/// The product of two elements of `ZZ[p, x]` (`integer_product`).
pub fn integer_product(a: &IntegerCoefficient, b: &IntegerCoefficient) -> IntegerCoefficient {
    let mut result: RingTerms<Int> = RingTerms::new();
    for (m1, c1) in &a.terms {
        for (m2, c2) in &b.terms {
            let m: Monomial = m1.iter().zip(m2).map(|(i, j)| i + j).collect();
            let value: Int = result.remove(&m).unwrap_or(Int::ZERO) + c1 * c2;
            result.insert(m, value);
        }
    }
    result.retain(|_, c| !c.is_zero());
    PolyElement {
        terms: result,
        ring: a.ring.clone(),
    }
}

/// The exact quotient of two elements of `ZZ[p, x]` (`integer_quotient`):
/// `Value` when `b` does not divide `a`.
pub fn integer_quotient(a: &IntegerCoefficient, b: &IntegerCoefficient) -> PolyResult<IntegerCoefficient> {
    if b.terms.is_empty() {
        return Err(zero_division("division by zero"));
    }
    let not_dividing = || PolyError::Value("the divisor does not divide the dividend".to_string());
    if b.terms.len() == 1 {
        let (n, d) = b.terms.iter().next().expect("one term");
        let mut result: RingTerms<Int> = RingTerms::new();
        for (m, c) in &a.terms {
            let (q, r) = (div_floor(c, d), mod_floor(c, d));
            if !r.is_zero() || m.iter().zip(n).any(|(i, j)| i < j) {
                return Err(not_dividing());
            }
            result.insert(m.iter().zip(n).map(|(i, j)| i - j).collect(), q);
        }
        return Ok(PolyElement {
            terms: result,
            ring: a.ring.clone(),
        });
    }
    // the rings of this module are lex ordered: the leading monomial is the
    // largest exponent vector
    let leading: Monomial = b.terms.keys().max().expect("a term").clone();
    let lc: Int = b.terms[&leading].clone();
    let mut rest: HashMap<Monomial, Int> = a.terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect();
    let mut heap: BinaryHeap<Monomial> = rest.keys().cloned().collect();
    let mut quotient: RingTerms<Int> = RingTerms::new();
    while !rest.is_empty() {
        let Some(m) = heap.pop() else { break };
        let Some(value) = rest.get(&m).cloned() else {
            continue;
        };
        let (q, r) = (div_floor(&value, &lc), mod_floor(&value, &lc));
        if !r.is_zero() || m.iter().zip(&leading).any(|(i, j)| i < j) {
            return Err(not_dividing());
        }
        let shift: Monomial = m.iter().zip(&leading).map(|(i, j)| i - j).collect();
        for (k, e) in &b.terms {
            let product: Monomial = shift.iter().zip(k).map(|(i, j)| i + j).collect();
            match rest.get(&product).cloned() {
                None => {
                    rest.insert(product.clone(), -(&q * e));
                    heap.push(product);
                }
                Some(s) => {
                    let s: Int = s - &q * e;
                    if s.is_zero() {
                        rest.remove(&product);
                    } else {
                        rest.insert(product, s);
                    }
                }
            }
        }
        quotient.insert(shift, q);
    }
    Ok(PolyElement {
        terms: quotient,
        ring: a.ring.clone(),
    })
}

/// Why an expression could not be converted (`_NotPolynomial`,
/// `_NotRational`, or an error of the arithmetic).
enum ConvertError {
    NotPolynomial,
    NotRational,
    Other(PolyError),
}

impl From<PolyError> for ConvertError {
    fn from(e: PolyError) -> Self {
        ConvertError::Other(e)
    }
}

/// A differential polynomial ring with a ranking (`DifferentialRing`).
#[derive(Debug, Clone)]
pub struct DifferentialRing {
    pub functions: Vec<Basic>,
    pub variables: Vec<Basic>,
    pub parameters: Vec<Basic>,
    /// The symbols of the coefficients: the parameters, then the variables.
    pub symbols: Vec<Basic>,
    /// The field of coefficients `QQ(p, x)`.
    pub domain: SparseFractionField<RationalField>,
    /// `ZZ[p, x]`, the coefficients of the differential polynomials.
    pub integers: SparsePolynomialRing<IntegerRing>,
    rationals: SparsePolynomialRing<RationalField>,
    index: HashMap<Basic, usize>,
    generators: Vec<Coefficient>,
    pub blocks: Vec<Vec<usize>>,
    level: HashMap<usize, (i64, i64)>,
}

fn value_error(message: String) -> PolyError {
    PolyError::Value(message)
}

impl DifferentialRing {
    /// The ring of the `functions` (applied undefined functions of distinct
    /// symbols), the `variables` in the order of the ranking (the arguments
    /// in their order of appearance by default), the blocks of the
    /// `ranking` (the highest first; one block by default) and the constant
    /// `parameters`. Python's `TypeError`s are `Value` here.
    pub fn new(
        functions: &[Basic],
        variables: Option<&[Basic]>,
        ranking: Option<&[Vec<Basic>]>,
        parameters: &[Basic],
    ) -> PolyResult<DifferentialRing> {
        if functions.is_empty() {
            return Err(value_error("at least one function is expected".to_string()));
        }
        let mut found: Vec<Basic> = Vec::new();
        let mut heads: HashSet<String> = HashSet::new();
        for f in functions {
            let Head::Undefined(u) = f.head() else {
                return Err(value_error(format!(
                    "an applied undefined function such as u(x, y) is expected, got {}",
                    f
                )));
            };
            if !heads.insert(u.name.clone()) {
                return Err(value_error(format!("the function {} is given twice", u.name)));
            }
            for arg in f.args() {
                if !matches!(arg.head(), Head::Symbol(_) | Head::Dummy(..) | Head::Wild(_)) {
                    return Err(value_error(format!("the arguments of {} must be symbols", f)));
                }
                if !found.contains(arg) {
                    found.push(arg.clone());
                }
            }
            let distinct: HashSet<&Basic> = f.args().iter().collect();
            if distinct.len() != f.args().len() {
                return Err(value_error(format!("the arguments of {} must be distinct", f)));
            }
        }
        let chosen: Vec<Basic> = match variables {
            None => found.clone(),
            Some(vs) => {
                for s in vs {
                    if !s.is_symbol() {
                        return Err(value_error(format!("the variables must be symbols, got {}", s)));
                    }
                }
                if let Some(missing) = found.iter().find(|s| !vs.contains(s)) {
                    return Err(value_error(format!("{} is not among the variables", missing)));
                }
                vs.to_vec()
            }
        };
        let distinct: HashSet<&Basic> = chosen.iter().collect();
        if distinct.len() != chosen.len() {
            return Err(value_error("the variables must be distinct".to_string()));
        }
        let parameters: Vec<Basic> = parameters.iter().filter(|p| !chosen.contains(p)).cloned().collect();
        let mut symbols: Vec<Basic> = parameters.clone();
        symbols.extend(chosen.iter().cloned());
        let names: Vec<String> = symbols.iter().map(symbol_name).collect();
        let domain = SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names.clone()));
        let integers = SparsePolynomialRing::sympy(ZZ, names.clone());
        let rationals = SparsePolynomialRing::sympy(QQ, names);
        let index: HashMap<Basic, usize> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let generators: Vec<Coefficient> = domain.gens();
        let mut ring = DifferentialRing {
            functions: functions.to_vec(),
            variables: chosen,
            parameters,
            symbols,
            domain,
            integers,
            rationals,
            index,
            generators,
            blocks: Vec::new(),
            level: HashMap::new(),
        };
        ring.blocks = ring.make_blocks(ranking)?;
        let count: i64 = ring.blocks.len() as i64;
        for (b, block) in ring.blocks.iter().enumerate() {
            for (position, a) in block.iter().enumerate() {
                ring.level
                    .insert(*a, (count - b as i64, (block.len() - position) as i64));
            }
        }
        Ok(ring)
    }

    fn make_blocks(&self, ranking: Option<&[Vec<Basic>]>) -> PolyResult<Vec<Vec<usize>>> {
        let Some(ranking) = ranking else {
            return Ok(vec![(0..self.functions.len()).collect()]);
        };
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for block in ranking {
            let mut indices: Vec<usize> = Vec::new();
            for f in block {
                let Some(a) = self.functions.iter().position(|g| g == f) else {
                    return Err(value_error(format!(
                        "{} of the ranking is not a function of the ring",
                        f
                    )));
                };
                if !seen.insert(a) {
                    return Err(value_error(format!("{} occurs twice in the ranking", f)));
                }
                indices.push(a);
            }
            if !indices.is_empty() {
                blocks.push(indices);
            }
        }
        if seen.len() != self.functions.len() {
            let absent = self
                .functions
                .iter()
                .enumerate()
                .find(|(a, _)| !seen.contains(a))
                .expect("absent");
            return Err(value_error(format!("{} is not in the ranking", absent.1)));
        }
        Ok(blocks)
    }

    // -- derivatives and the ranking

    /// The number of derivations.
    pub fn n(&self) -> usize {
        self.variables.len()
    }

    /// Whether a derivative of higher order is always higher.
    pub fn is_orderly(&self) -> bool {
        self.blocks.len() == 1
    }

    /// The key which sorts the derivatives by the ranking (`rank_key`).
    pub fn rank_key(&self, jet: &Jet) -> Vec<i64> {
        let (a, j) = jet;
        let mut key: Vec<i64> = if *a < 0 {
            vec![-1, 0, 0]
        } else {
            let (block, position) = self.level[&(*a as usize)];
            vec![block, j.iter().map(|&e| i64::from(e)).sum(), position]
        };
        key.extend(j.iter().map(|&e| i64::from(e)));
        key
    }

    /// The highest of the derivatives (the first of the highest ones, as
    /// Python's `max`).
    pub fn highest(&self, jets: &[Jet]) -> Jet {
        let mut best: &Jet = &jets[0];
        let mut best_key: Vec<i64> = self.rank_key(best);
        for jet in &jets[1..] {
            let key: Vec<i64> = self.rank_key(jet);
            if key > best_key {
                best = jet;
                best_key = key;
            }
        }
        best.clone()
    }

    /// The derivative of a derivative with respect to the variable `i`.
    pub fn differentiate(&self, jet: &Jet, i: usize) -> Jet {
        let mut orders: Monomial = jet.1.clone();
        orders[i] += 1;
        (jet.0, orders)
    }

    /// Whether `jet` is a derivative of `other` (or equal to it).
    pub fn is_derivative_of(&self, jet: &Jet, other: &Jet) -> bool {
        jet.0 == other.0 && jet.1.iter().zip(&other.1).all(|(j, k)| j >= k)
    }

    /// The derivatives which vanish because a function does not depend on a
    /// variable (`constancy`).
    pub fn constancy(&self) -> Vec<Jet> {
        let mut result: Vec<Jet> = Vec::new();
        for (a, f) in self.functions.iter().enumerate() {
            for (i, s) in self.variables.iter().enumerate() {
                if !f.args().contains(s) {
                    let mut orders: Monomial = vec![0; self.n()];
                    orders[i] = 1;
                    result.push((a as i64, orders));
                }
            }
        }
        result
    }

    // -- coefficients

    /// An element of [`Self::integers`] as an element of the field.
    pub fn field_element(&self, c: &IntegerCoefficient) -> Coefficient {
        let terms: RingTerms<Rat> = c.terms.iter().map(|(m, e)| (m.clone(), rat_from_int(e))).collect();
        self.domain.from_ring(&self.rationals.ring.from_dict(terms))
    }

    /// The expression of an element of [`Self::integers`], with the values
    /// `at` of some symbols substituted.
    pub fn integer_expr(&self, c: &IntegerCoefficient, at: Option<&HashMap<Basic, Basic>>) -> PolyResult<Basic> {
        let terms: Vec<(Monomial, Rat)> = c.terms.iter().map(|(m, e)| (m.clone(), rat_from_int(e))).collect();
        self.polynomial_expr(&terms, at)
    }

    /// The expression of an element of the field.
    pub fn coefficient_expr(&self, c: &Coefficient) -> PolyResult<Basic> {
        let numer: Vec<(Monomial, Rat)> = c.numer.terms.iter().map(|(m, e)| (m.clone(), e.clone())).collect();
        let numerator: Basic = self.polynomial_expr(&numer, None)?;
        if self.rationals.is_one(&c.denom) {
            return Ok(numerator);
        }
        let denom: Vec<(Monomial, Rat)> = c.denom.terms.iter().map(|(m, e)| (m.clone(), e.clone())).collect();
        mul(&[
            numerator,
            pow(&self.polynomial_expr(&denom, None)?, &Basic::negative_one())?,
        ])
    }

    fn polynomial_expr(&self, terms: &[(Monomial, Rat)], at: Option<&HashMap<Basic, Basic>>) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (m, c) in terms {
            let mut factors: Vec<Basic> = vec![Basic::from_rational(c.clone())];
            for (s, e) in self.symbols.iter().zip(m) {
                if *e > 0 {
                    let value: Basic = at.and_then(|a| a.get(s)).cloned().unwrap_or_else(|| s.clone());
                    factors.push(if *e > 1 {
                        pow(&value, &Basic::int(i64::from(*e)))?
                    } else {
                        value
                    });
                }
            }
            parts.push(mul(&factors)?);
        }
        if parts.is_empty() {
            return Ok(Basic::zero());
        }
        add(&parts)
    }

    /// An expression as an element of the field of coefficients.
    pub fn coefficient(&self, expr: &Basic) -> PolyResult<Coefficient> {
        match self.convert(expr, &HashMap::new()) {
            Ok(terms) => Ok(terms.get(&Vec::new()).cloned().unwrap_or_else(|| self.domain.zero())),
            Err(ConvertError::Other(e)) => Err(e),
            Err(_) => Err(value_error(format!(
                "{} is not a rational function of the variables {} and the parameters {}",
                expr,
                list(&self.variables),
                list(&self.parameters)
            ))),
        }
    }

    /// The derivative of a coefficient with respect to the variable `i`.
    pub fn coefficient_derivative(&self, c: &Coefficient, i: usize) -> Coefficient {
        self.domain.diff(c, self.parameters.len() + i)
    }

    /// The derivative of an element of [`Self::integers`].
    pub fn integer_derivative(&self, c: &IntegerCoefficient, i: usize) -> IntegerCoefficient {
        c.diff(self.parameters.len() + i)
    }

    /// The terms multiplied by an element of the field which makes the
    /// coefficients elements of [`Self::integers`] (`integral`).
    pub fn integral(&self, terms: &FieldTerms) -> Terms {
        if terms.is_empty() {
            return Terms::new();
        }
        let values: Vec<&Coefficient> = terms.values().collect();
        let mut multiple: PolyElement<RationalField> = values[0].denom.clone();
        for c in &values[1..] {
            multiple = self.rationals.lcm(&multiple, &c.denom);
        }
        let factor: Coefficient = self.domain.from_ring(&multiple);
        let mut cleared: Vec<(JetMonomial, PolyElement<RationalField>)> = Vec::new();
        let mut denominator: Int = Int::ONE;
        for (m, c) in terms.iter() {
            let scaled: Coefficient = self.domain.mul(c, &factor);
            let q: PolyElement<RationalField> = scaled.numer.mul_ground(&recip(&scaled.denom.LC()));
            for e in q.terms.values() {
                let d: Int = denom(e);
                denominator = &denominator / gcd(&denominator, &d) * &d;
            }
            cleared.push((m.clone(), q));
        }
        let mut result: Terms = Terms::new();
        for (m, q) in cleared {
            let coefficients: RingTerms<Int> = q
                .terms
                .iter()
                .map(|(k, e)| {
                    let scaled: Rat = e * rat_from_int(&denominator);
                    (k.clone(), numer(&scaled))
                })
                .collect();
            result.insert(m, self.integers.ring.from_dict(coefficients));
        }
        result
    }

    // -- conversion

    /// The derivative which an applied function or a `Derivative` of one
    /// stands for, `None` for another expression.
    pub fn jet(&self, expr: &Basic) -> Option<Jet> {
        if matches!(expr.head(), Head::Undefined(_)) {
            return self
                .functions
                .iter()
                .position(|f| f == expr)
                .map(|a| (a as i64, vec![0; self.n()]));
        }
        if matches!(expr.head(), Head::Derivative) {
            let a: usize = self.functions.iter().position(|f| *f == expr.args()[0])?;
            let mut orders: Monomial = vec![0; self.n()];
            for (s, count) in derivative_counts(expr) {
                let i: usize = self.variables.iter().position(|v| *v == s)?;
                orders[i] += u32::try_from(count).ok()?;
            }
            return Some((a as i64, orders));
        }
        None
    }

    /// The expression of a derivative.
    pub fn to_expr(&self, jet: &Jet) -> Basic {
        let (a, j) = jet;
        if *a < 0 {
            return Basic::one();
        }
        let counts: Vec<(Basic, u64)> = self
            .variables
            .iter()
            .zip(j)
            .filter(|(_, e)| **e > 0)
            .map(|(s, e)| (s.clone(), u64::from(*e)))
            .collect();
        let f: &Basic = &self.functions[*a as usize];
        if counts.is_empty() {
            return f.clone();
        }
        derivative_node(f, &counts)
    }

    fn convert(&self, e: &Basic, jets: &HashMap<Basic, Jet>) -> Result<FieldTerms, ConvertError> {
        let k = &self.domain;
        if let Some(jet) = jets.get(e) {
            return Ok(std::iter::once((vec![(jet.clone(), 1)], k.one())).collect());
        }
        if let Some(v) = e.as_rational() {
            if v.is_zero() {
                return Ok(FieldTerms::new());
            }
            let c: Coefficient = k.from_ring(&self.rationals.ring.ground(v.clone()));
            return Ok(std::iter::once((Vec::new(), c)).collect());
        }
        if e.is_symbol() {
            let i: usize = *self.index.get(e).ok_or(ConvertError::NotRational)?;
            return Ok(std::iter::once((Vec::new(), self.generators[i].clone())).collect());
        }
        match e.head() {
            Head::Add => {
                let mut result: FieldTerms = FieldTerms::new();
                for arg in e.args() {
                    for (m, c) in self.convert(arg, jets)?.iter() {
                        let value: Coefficient = match result.get(m) {
                            Some(existing) => k.add(existing, c),
                            None => c.clone(),
                        };
                        if k.is_zero(&value) {
                            result.remove(m);
                        } else {
                            result.insert(m.clone(), value);
                        }
                    }
                }
                Ok(result)
            }
            Head::Mul => {
                let mut product: FieldTerms = std::iter::once((Vec::new(), k.one())).collect();
                for arg in e.args() {
                    product = self.multiply(&product, &self.convert(arg, jets)?);
                }
                Ok(product)
            }
            Head::Pow if is_integer_class(&e.args()[1]) => {
                let base: FieldTerms = self.convert(&e.args()[0], jets)?;
                let n: &Int = e.args()[1].as_rational().expect("an integer").numerator();
                if is_negative(n) {
                    if base.keys().any(|m| !m.is_empty()) {
                        return Err(ConvertError::NotPolynomial);
                    }
                    let c: Coefficient = base.get(&Vec::new()).cloned().unwrap_or_else(|| k.zero());
                    if k.is_zero(&c) {
                        return Err(ConvertError::Other(zero_division(&format!(
                            "division by zero in {}",
                            e
                        ))));
                    }
                    let exponent: i64 =
                        i64::try_from(n).map_err(|_| ConvertError::Other(value_error("a huge exponent".into())))?;
                    return Ok(std::iter::once((Vec::new(), k.pow_signed(&c, exponent)?)).collect());
                }
                let count: u64 =
                    u64::try_from(n).map_err(|_| ConvertError::Other(value_error("a huge exponent".into())))?;
                let mut power: FieldTerms = std::iter::once((Vec::new(), k.one())).collect();
                for _ in 0..count {
                    power = self.multiply(&power, &base);
                }
                Ok(power)
            }
            _ => Err(ConvertError::NotRational),
        }
    }

    fn multiply(&self, a: &FieldTerms, b: &FieldTerms) -> FieldTerms {
        let k = &self.domain;
        let mut result: FieldTerms = FieldTerms::new();
        for (m, c) in a.iter() {
            for (n, d) in b.iter() {
                let product: JetMonomial = monomial_product(m, n);
                let mut value: Coefficient = k.mul(c, d);
                if let Some(existing) = result.get(&product) {
                    value = k.add(existing, &value);
                }
                if k.is_zero(&value) {
                    result.remove(&product);
                } else {
                    result.insert(product, value);
                }
            }
        }
        result
    }

    /// The terms of an expression in which the keys of `jets` stand for the
    /// derivatives (`terms_of`).
    pub fn terms_of(&self, expr: &Basic, jets: &HashMap<Basic, Jet>) -> PolyResult<FieldTerms> {
        self.convert(expr, jets).map_err(|e| self.conversion_error(expr, e))
    }

    fn conversion_error(&self, expr: &Basic, e: ConvertError) -> PolyError {
        match e {
            ConvertError::NotPolynomial => value_error(format!("{} is not a polynomial in the derivatives", expr)),
            ConvertError::NotRational => value_error(format!(
                "{} is not a polynomial in the derivatives with rational coefficients",
                expr
            )),
            ConvertError::Other(e) => e,
        }
    }

    /// The monomials and coefficients of a differential polynomial given as
    /// an expression (`terms`), in SymPy's order of the terms of a `Poly`.
    pub fn terms(&self, equation: &Basic) -> PolyResult<FieldTerms> {
        let expr: Basic = doit(equation)?;
        let mut atoms: Vec<Basic> = Vec::new();
        for node in expr.preorder() {
            if matches!(node.head(), Head::Derivative | Head::Undefined(_)) && !atoms.contains(&node) {
                atoms.push(node);
            }
        }
        atoms.sort_by_cached_key(sort_key);
        let mut jets: HashMap<Basic, Jet> = HashMap::new();
        for atom in &atoms {
            match self.jet(atom) {
                Some(jet) => {
                    jets.insert(atom.clone(), jet);
                }
                None => {
                    let is_undefined: bool = matches!(atom.head(), Head::Undefined(_));
                    let involves: bool = atom
                        .preorder()
                        .iter()
                        .any(|n| matches!(n.head(), Head::Undefined(_)) && self.functions.contains(n));
                    if is_undefined || involves {
                        if matches!(atom.head(), Head::Derivative) && self.functions.contains(&atom.args()[0]) {
                            return Err(value_error(format!(
                                "{} is not a derivative in the variables of the ring",
                                atom
                            )));
                        }
                        if is_undefined {
                            return Err(value_error(format!("{} is not a function of the ring", atom)));
                        }
                    }
                    return Err(value_error(format!(
                        "{} is not a derivative of a function of the ring",
                        atom
                    )));
                }
            }
        }
        let mut extra: Vec<Basic> = expr
            .free_symbols()
            .into_iter()
            .filter(|s| !self.variables.contains(s) && !self.parameters.contains(s))
            .collect();
        if !extra.is_empty() {
            extra.sort_by_key(symbol_name);
            return Err(value_error(format!(
                "{} is neither a variable nor a parameter of the ring",
                extra[0]
            )));
        }
        if atoms.is_empty() {
            let c: Coefficient = self.coefficient(&expr)?;
            let mut result: FieldTerms = FieldTerms::new();
            if !self.domain.is_zero(&c) {
                result.insert(Vec::new(), c);
            }
            return Ok(result);
        }
        let result: FieldTerms = self
            .convert(&expr, &jets)
            .map_err(|e| self.conversion_error(&expr, e))?;
        // SymPy's order of the terms of a Poly in the atoms sorted by
        // sort_key: increasing exponent vectors
        let ordered: Vec<Jet> = atoms.iter().map(|a| jets[a].clone()).collect();
        let exponents = |m: &JetMonomial| -> Vec<u32> {
            let powers: HashMap<&Jet, u32> = m.iter().map(|(j, e)| (j, *e)).collect();
            let mut seen: HashSet<&Jet> = HashSet::new();
            let mut vector: Vec<u32> = Vec::new();
            for jet in &ordered {
                vector.push(if seen.contains(jet) {
                    0
                } else {
                    powers.get(jet).copied().unwrap_or(0)
                });
                seen.insert(jet);
            }
            vector
        };
        let mut keys: Vec<JetMonomial> = result.keys().cloned().collect();
        keys.sort_by_cached_key(|m| exponents(m));
        Ok(keys
            .into_iter()
            .map(|m| {
                let c: Coefficient = result.get(&m).expect("a term").clone();
                (m, c)
            })
            .collect())
    }

    /// The expression of a differential polynomial (`from_terms`).
    pub fn from_terms(&self, terms: &FieldTerms) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (monomial, c) in terms.iter() {
            let mut term: Basic = self.coefficient_expr(c)?;
            for (jet, e) in monomial {
                term = mul(&[term, pow(&self.to_expr(jet), &Basic::int(i64::from(*e)))?])?;
            }
            parts.push(term);
        }
        if parts.is_empty() {
            return Ok(Basic::zero());
        }
        add(&parts)
    }

    /// The highest derivative of a differential polynomial (`leader`).
    pub fn leader(&self, equation: &Basic) -> PolyResult<Basic> {
        let jets: Vec<Jet> = self
            .terms(equation)?
            .keys()
            .flat_map(|m| m.iter().map(|(j, _)| j.clone()))
            .collect();
        if jets.is_empty() {
            return Err(value_error("the differential polynomial has no derivative".to_string()));
        }
        Ok(self.to_expr(&self.highest(&jets)))
    }
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Wild(w) => w.symbol.name.clone(),
        _ => s.to_string(),
    }
}

/// A Python list of expressions as printed in the messages.
fn list(items: &[Basic]) -> String {
    format!(
        "[{}]",
        items.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ")
    )
}
