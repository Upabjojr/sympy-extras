//! Port of `sympy_extras/solvers/darboux.py`: Darboux polynomials,
//! exponential factors, Darboux integrating factors and first integrals
//! of polynomial vector fields `Q d/dx + P d/dy`, and the Prelle–Singer
//! procedure for `y' = P(x, y)/Q(x, y)`.
//!
//! Python's generic `_Coefficients[T]` is the trait [`Coefficients`] with
//! one implementation per field (`QQ`, a number field, `QQ(a, b, ...)`);
//! `_Context` is the object-safe trait [`Context`].

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::domains::algebraicfield::{AlgebraicElement, AlgebraicField};
use crate::polys::domains::domain::{Domain, SparseRow};
use crate::polys::domains::rationalfield::{RationalField, QQ};
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;
use crate::polys::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

type R = PolyResult<Basic>;
type Pair<K> = (PolyElement<K>, PolyElement<K>);
type Triple<K> = (PolyElement<K>, PolyElement<K>, PolyElement<K>);
/// A point of the unknowns: their values by index (`_Point`).
type Point<E> = BTreeMap<usize, E>;
/// A solution of a linear system (`_Affine`).
type Affine<E> = (Vec<E>, Vec<Vec<E>>, Vec<usize>);
/// A row of the parametric system (`_Row`).
type Row<K> = Vec<PolyElement<K>>;
/// The constraints at a leaf of the parametric elimination (`_Leaf`).
type Leaf<K> = (Vec<PolyElement<K>>, Vec<PolyElement<K>>, Vec<(usize, PolyElement<K>)>);
/// The exponent vector of `x**i y**j` (`_Monomial`).
type XY = (u32, u32);

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1)])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}
/// The core's `xreplace`, which does not evaluate the rebuilt nodes
/// (Python's `Basic.xreplace`; the order of the arguments stays).
fn raw_xreplace(e: &Basic, rule: &std::collections::HashMap<Basic, Basic>) -> R {
    Ok(e.xreplace(rule))
}
fn xreplace(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    raw_xreplace(e, &rule.iter().cloned().collect())
}
fn eq(a: &Basic, b: &Basic) -> Basic {
    crate::core::relational::rel(RelOp::Eq, a, b)
}
fn value_error(message: String) -> PolyError {
    PolyError::Value(message)
}
fn args_of_add(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn args_of_mul(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn name_of(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r: &Rat = e.as_rational()?;
    if !crate::numbers::rat_is_integer(r) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn is_integer_node(e: &Basic) -> bool {
    e.as_rational().is_some_and(crate::numbers::rat_is_integer)
}

// ---------------------------------------------------------------------------
// the results

/// A Darboux polynomial and its cofactor (`DarbouxPolynomial`).
#[derive(Clone, Debug, PartialEq)]
pub struct DarbouxPolynomial {
    pub polynomial: Basic,
    pub cofactor: Basic,
}

impl DarbouxPolynomial {
    /// (`__repr__`)
    pub fn repr(&self) -> String {
        format!("DarbouxPolynomial({}, {})", self.polynomial, self.cofactor)
    }
}

/// An exponential factor `exp(A/B)` and its cofactor (`ExponentialFactor`).
#[derive(Clone, Debug)]
pub struct ExponentialFactor {
    pub numerator: Basic,
    pub denominator: Basic,
    pub cofactor: Basic,
}

impl ExponentialFactor {
    /// `A/B` (`argument`).
    pub fn argument(&self) -> R {
        div(&self.numerator, &self.denominator)
    }

    /// `exp(A/B)` (`factor`).
    pub fn factor(&self) -> R {
        crate::functions::elementary::exponential::exp(&self.argument()?)
    }

    /// (`__repr__`)
    pub fn repr(&self) -> String {
        format!(
            "ExponentialFactor({}, {}, {})",
            self.numerator, self.denominator, self.cofactor
        )
    }
}

// ---------------------------------------------------------------------------
// the coefficient fields

/// The field of the coefficients: its domain, the factorisation over it
/// and the conversions of its elements (`_Coefficients`).
pub trait Coefficients {
    type K: Domain;

    fn domain(&self) -> &Self::K;

    /// The monic irreducible factors of positive degree of `f`, each once,
    /// in SymPy's order (`factor`).
    fn factor(&self, f: &PolyElement<Self::K>) -> PolyResult<Vec<PolyElement<Self::K>>>;

    /// `e` with the parameters named as inside the field (`rename`).
    fn rename(&self, e: &Basic) -> R {
        Ok(e.clone())
    }

    /// The coefficient `e` as an element (`element`).
    fn element(&self, e: &Basic) -> PolyResult<<Self::K as Domain>::Elem> {
        self.domain().from_expr(&self.rename(e)?)
    }

    /// The element as an expression (`to_expr`).
    fn to_expr(&self, c: &<Self::K as Domain>::Elem) -> R {
        self.domain().to_expr(c)
    }

    /// The element as a rational number, when it is one (`rational`).
    fn rational(&self, _c: &<Self::K as Domain>::Elem) -> Option<Rat> {
        None
    }

    /// A key of the element (`key`).
    fn key(&self, c: &<Self::K as Domain>::Elem) -> String {
        self.domain().to_str(c)
    }

    fn is_rationals(&self) -> bool {
        false
    }

    /// The roots in `K` of `f`, a polynomial in the variable `index` alone
    /// (`roots`).
    fn roots(&self, f: &PolyElement<Self::K>, index: usize) -> PolyResult<Vec<<Self::K as Domain>::Elem>> {
        let k: &Self::K = self.domain();
        let mut found = Vec::new();
        for h in self.factor(f)? {
            if h.degree() != 1 {
                continue;
            }
            let n: usize = h.ring.ngens;
            let constant = h.coeff(&vec![0; n]);
            let unit: Vec<u32> = (0..n).map(|i| u32::from(i == index)).collect();
            found.push(k.neg(&k.exquo(&constant, &h.coeff(&unit))?));
        }
        Ok(found)
    }
}

/// `QQ` (`_Rationals`).
pub struct Rationals;

impl Coefficients for Rationals {
    type K = RationalField;

    fn domain(&self) -> &RationalField {
        &QQ
    }

    fn factor(&self, f: &PolyElement<RationalField>) -> PolyResult<Vec<PolyElement<RationalField>>> {
        if f.is_ground() {
            return Ok(Vec::new());
        }
        let (_, factors) = crate::polys::polytools::factor_all(f)?;
        let mut result = Vec::new();
        for (h, _) in factors {
            if h.degree() > 0 {
                let terms: Terms<Rat> = h
                    .terms
                    .iter()
                    .map(|(m, c)| (m.clone(), crate::numbers::rat_from_int(c)))
                    .collect();
                result.push(f.ring.from_dict(terms).monic()?);
            }
        }
        Ok(result)
    }

    fn rational(&self, c: &Rat) -> Option<Rat> {
        Some(c.clone())
    }

    fn is_rationals(&self) -> bool {
        true
    }
}

/// A number field with the images of its generating algebraic numbers
/// (`_NumberField`).
pub struct NumberField {
    pub algebraic: AlgebraicField,
    pub images: Vec<(Basic, AlgebraicElement)>,
    radicals: Vec<(Basic, (i64, AlgebraicElement))>,
}

impl NumberField {
    pub fn new(k: AlgebraicField, images: Option<Vec<(Basic, AlgebraicElement)>>) -> NumberField {
        let images: Vec<(Basic, AlgebraicElement)> = images.unwrap_or_else(|| vec![(k.alpha_expr.clone(), k.alpha())]);
        let mut radicals: Vec<(Basic, (i64, AlgebraicElement))> = Vec::new();
        for (g, image) in &images {
            if g.is_pow() && g.args()[0].as_rational().is_some() {
                if let Some(r) = g.args()[1].as_rational() {
                    let (p, q) = (crate::numbers::numer(r), crate::numbers::denom(r));
                    if p == Int::from(1) && q > Int::from(1) {
                        let q: i64 = i64::try_from(&q).unwrap_or(i64::MAX);
                        let base: Basic = g.args()[0].clone();
                        match radicals.iter_mut().find(|(b, _)| *b == base) {
                            Some(entry) => entry.1 = (q, image.clone()),
                            None => radicals.push((base, (q, image.clone()))),
                        }
                    }
                }
            }
        }
        NumberField {
            algebraic: k,
            images,
            radicals,
        }
    }

    fn signed_pow(&self, a: &AlgebraicElement, k: i64) -> PolyResult<AlgebraicElement> {
        let field: &AlgebraicField = &self.algebraic;
        if k >= 0 {
            Ok(field.pow(a, k as u64))
        } else {
            Ok(field.pow(&field.invert(a)?, k.unsigned_abs()))
        }
    }
}

impl Coefficients for NumberField {
    type K = AlgebraicField;

    fn domain(&self) -> &AlgebraicField {
        &self.algebraic
    }

    fn element(&self, e: &Basic) -> PolyResult<AlgebraicElement> {
        let k: &AlgebraicField = &self.algebraic;
        if let Some(r) = e.as_rational() {
            return Ok(k.from_rational(r));
        }
        if let Some((_, image)) = self.images.iter().find(|(g, _)| g == e) {
            return Ok(image.clone());
        }
        if e.is_pow() {
            if let Some(r) = e.args()[1].as_rational() {
                let (p, q) = (crate::numbers::numer(r), crate::numbers::denom(r));
                if let Some((_, (order, image))) = self.radicals.iter().find(|(b, _)| *b == e.args()[0]) {
                    let q_small: i64 = i64::try_from(&q).unwrap_or(i64::MAX);
                    if order % q_small == 0 {
                        let p_small: i64 = i64::try_from(&p).unwrap_or(i64::MAX);
                        return self.signed_pow(image, p_small * (order / q_small));
                    }
                }
                if let Some(k_exp) = integer_of(&e.args()[1]) {
                    let base: AlgebraicElement = self.element(&e.args()[0])?;
                    return self.signed_pow(&base, k_exp);
                }
            }
        }
        if e.is_add() {
            let mut total: AlgebraicElement = k.zero();
            for term in e.args() {
                total = k.add(&total, &self.element(term)?);
            }
            return Ok(total);
        }
        if e.is_mul() {
            let mut product: AlgebraicElement = k.one();
            for factor in e.args() {
                product = k.mul(&product, &self.element(factor)?);
            }
            return Ok(product);
        }
        Err(value_error(format!("{} is not an element of {}", e, k.name())))
    }

    fn to_expr(&self, c: &AlgebraicElement) -> R {
        expand(&self.algebraic.to_expr(c)?)
    }

    fn factor(&self, f: &PolyElement<AlgebraicField>) -> PolyResult<Vec<PolyElement<AlgebraicField>>> {
        if f.is_ground() {
            return Ok(Vec::new());
        }
        let (_, factors) = crate::polys::algebraicfactor::ext_factor_list(f)?;
        factors
            .into_iter()
            .filter(|(h, _)| h.degree() > 0)
            .map(|(h, _)| h.monic())
            .collect()
    }

    fn rational(&self, c: &AlgebraicElement) -> Option<Rat> {
        match c.len() {
            0 => Some(Rat::from(0)),
            1 => Some(c[0].clone()),
            _ => None,
        }
    }
}

type ParameterField = SparseFractionField<RationalField>;

/// `QQ(a, b, ...)` with the parameters renamed (`_Parameters`).
pub struct Parameters {
    pub field: ParameterField,
    pub names: Vec<String>,
    forward: Vec<(Basic, Basic)>,
    restore: Vec<(Basic, Basic)>,
}

impl Parameters {
    pub fn new(parameters: &[Basic]) -> Parameters {
        let names: Vec<String> = (0..parameters.len()).map(|i| format!("_darboux_p{}", i)).collect();
        let field: ParameterField = SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names.clone()));
        let forward = parameters
            .iter()
            .zip(&names)
            .map(|(p, n)| (p.clone(), symbol(n)))
            .collect();
        let restore = parameters
            .iter()
            .zip(&names)
            .map(|(p, n)| (symbol(n), p.clone()))
            .collect();
        Parameters {
            field,
            names,
            forward,
            restore,
        }
    }
}

impl Coefficients for Parameters {
    type K = ParameterField;

    fn domain(&self) -> &ParameterField {
        &self.field
    }

    fn rename(&self, e: &Basic) -> R {
        raw_xreplace(e, &self.forward.iter().cloned().collect())
    }

    fn to_expr(&self, c: &SparseFracElement<RationalField>) -> R {
        raw_xreplace(&self.field.to_expr(c)?, &self.restore.iter().cloned().collect())
    }

    fn rational(&self, c: &SparseFracElement<RationalField>) -> Option<Rat> {
        if !(c.numer.is_ground() && c.denom.is_ground()) {
            return None;
        }
        let zero: Monomial = c.numer.ring.zero_monom.clone();
        let (n, d) = (c.numer.coeff(&zero), c.denom.coeff(&zero));
        Some(n / d)
    }

    fn factor(&self, f: &PolyElement<ParameterField>) -> PolyResult<Vec<PolyElement<ParameterField>>> {
        if f.is_ground() {
            return Ok(Vec::new());
        }
        let ring: &SparsePolynomialRing<RationalField> = &self.field.ring;
        let mut common: PolyElement<RationalField> = ring.ring.one();
        for c in f.terms.values() {
            common = ring.lcm(&common, &c.denom);
        }
        let n: usize = f.ring.ngens;
        let mut joint_names: Vec<String> = f.ring.names.clone();
        joint_names.extend(self.names.iter().cloned());
        let joint: Arc<PolyRing<RationalField>> = PolyRing::new(QQ, joint_names, MonomialOrder::Lex)?;
        let mut terms: Terms<Rat> = Terms::new();
        for (m, c) in &f.terms {
            let numerator: PolyElement<RationalField> = c.numer.mul(&ring.exquo(&common, &c.denom)?);
            for (pm, value) in &numerator.terms {
                let mut monomial: Monomial = m.clone();
                monomial.extend(pm.iter().copied());
                terms.insert(monomial, value.clone());
            }
        }
        let (_, factors) = crate::polys::polytools::factor_all(&joint.from_dict(terms))?;
        let mut result = Vec::new();
        for (h, _) in factors {
            if h.terms.keys().all(|m| m[..n].iter().all(|e| *e == 0)) {
                continue;
            }
            let mut grouped: BTreeMap<Monomial, Terms<Rat>> = BTreeMap::new();
            for (m, integer) in &h.terms {
                grouped
                    .entry(m[..n].to_vec())
                    .or_default()
                    .insert(m[n..].to_vec(), crate::numbers::rat_from_int(integer));
            }
            let back: Terms<SparseFracElement<RationalField>> = grouped
                .into_iter()
                .map(|(m, g)| (m, self.field.from_ring(&ring.ring.from_dict(g))))
                .collect();
            result.push(f.ring.from_dict(back).monic()?);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// polynomials of the unknowns: substitutions and values

/// `p` with the variables `values` replaced by polynomials of the same
/// ring (`_substituted`).
fn substituted<K: Domain>(p: &PolyElement<K>, values: &BTreeMap<usize, PolyElement<K>>) -> PolyElement<K> {
    if values.is_empty() || p.terms.is_empty() {
        return p.clone();
    }
    let ring: &Arc<PolyRing<K>> = &p.ring;
    let mut powers: BTreeMap<(usize, u32), PolyElement<K>> = BTreeMap::new();
    let mut result: PolyElement<K> = ring.zero();
    for (m, c) in &p.terms {
        let base: Monomial = m
            .iter()
            .enumerate()
            .map(|(i, e)| if values.contains_key(&i) { 0 } else { *e })
            .collect();
        let mut term: PolyElement<K> = ring.term(base, c.clone());
        for (i, e) in m.iter().enumerate() {
            if *e > 0 {
                if let Some(value) = values.get(&i) {
                    let power = powers.entry((i, *e)).or_insert_with(|| value.pow(u64::from(*e)));
                    term = term.mul(power);
                }
            }
        }
        result = result.add(&term);
    }
    result
}

/// `p` with the variables of `point` set to its values (`_evaluated`).
fn evaluated<K: Domain>(p: &PolyElement<K>, point: &Point<K::Elem>) -> PolyElement<K> {
    if point.is_empty() || p.terms.is_empty() {
        return p.clone();
    }
    let k: &K = &p.ring.domain;
    let mut terms: Terms<K::Elem> = Terms::new();
    for (m, c) in &p.terms {
        let mut value: K::Elem = c.clone();
        for (i, e) in m.iter().enumerate() {
            if *e > 0 {
                if let Some(v) = point.get(&i) {
                    value = k.mul(&value, &k.pow(v, u64::from(*e)));
                }
            }
        }
        if k.is_zero(&value) {
            continue;
        }
        let monomial: Monomial = m
            .iter()
            .enumerate()
            .map(|(i, e)| if point.contains_key(&i) { 0 } else { *e })
            .collect();
        let s: K::Elem = k.add(terms.get(&monomial).unwrap_or(&k.zero()), &value);
        if k.is_zero(&s) {
            terms.remove(&monomial);
        } else {
            terms.insert(monomial, s);
        }
    }
    PolyElement {
        terms,
        ring: Arc::clone(&p.ring),
    }
}

/// The value of `p` at a point of all its variables (`_value`).
fn value<K: Domain>(p: &PolyElement<K>, point: &Point<K::Elem>) -> PolyResult<K::Elem> {
    let e: PolyElement<K> = evaluated(p, point);
    if !e.is_ground() {
        return Err(value_error(format!("an unknown is left in {}", p)));
    }
    Ok(e.coeff(&p.ring.zero_monom))
}

/// The rows and right-hand sides of the equations of degree at most one
/// in the unknowns `gens` (`_linear_rows`).
fn linear_rows<K: Domain>(equations: &[PolyElement<K>], gens: &[usize], k: &K) -> (Vec<Vec<K::Elem>>, Vec<K::Elem>) {
    let mut rows: Vec<Vec<K::Elem>> = Vec::new();
    let mut rhs: Vec<K::Elem> = Vec::new();
    for equation in equations {
        if equation.degree() > 1 {
            continue;
        }
        let mut row: Vec<K::Elem> = vec![k.zero(); gens.len()];
        let mut constant: K::Elem = k.zero();
        for (monomial, coefficient) in &equation.terms {
            if monomial.iter().all(|e| *e == 0) {
                constant = coefficient.clone();
            } else {
                let variable: usize = monomial.iter().position(|e| *e == 1).expect("a linear monomial");
                let column: usize = gens
                    .iter()
                    .position(|g| *g == variable)
                    .expect("an unknown of the system");
                row[column] = coefficient.clone();
            }
        }
        rows.push(row);
        rhs.push(k.neg(&constant));
    }
    (rows, rhs)
}

/// The solutions of `rows * v = rhs`: a particular solution, a basis of
/// the nullspace and the free columns, or `None` (`_solve_affine`).
fn solve_affine<K: Domain>(rows: &[Vec<K::Elem>], rhs: &[K::Elem], n: usize, k: &K) -> Option<Affine<K::Elem>> {
    if rows.is_empty() {
        let identity: Vec<Vec<K::Elem>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { k.one() } else { k.zero() }).collect())
            .collect();
        return Some((vec![k.zero(); n], identity, (0..n).collect()));
    }
    let mut sparse: Vec<SparseRow<K::Elem>> = Vec::new();
    for (row, value) in rows.iter().zip(rhs) {
        let entries: SparseRow<K::Elem> = row
            .iter()
            .chain(std::iter::once(value))
            .enumerate()
            .filter(|(_, c)| !k.is_zero(c))
            .map(|(j, c)| (j, c.clone()))
            .collect();
        if !entries.is_empty() {
            sparse.push(entries);
        }
    }
    let echelon: Vec<SparseRow<K::Elem>> = k.sparse_echelon(&sparse);
    let pivots: Vec<usize> = echelon
        .iter()
        .map(|r| *r.keys().next().expect("a nonzero row"))
        .collect();
    if pivots.contains(&n) {
        return None;
    }
    let mut particular: Vec<K::Elem> = vec![k.zero(); n];
    for (r, column) in echelon.iter().zip(&pivots) {
        particular[*column] = r.get(&n).cloned().unwrap_or_else(|| k.zero());
    }
    let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
    let mut basis: Vec<Vec<K::Elem>> = Vec::new();
    for &column in &free {
        let mut vector: Vec<K::Elem> = vec![k.zero(); n];
        vector[column] = k.one();
        for (r, pivot) in echelon.iter().zip(&pivots) {
            vector[*pivot] = k.neg(&r.get(&column).cloned().unwrap_or_else(|| k.zero()));
        }
        basis.push(vector);
    }
    Some((particular, basis, free))
}

// ---------------------------------------------------------------------------
// the vector field

/// The vector field `Q d/dx + P d/dy` over the coefficient field
/// (`_Field`).
pub struct VectorField<C: Coefficients> {
    pub c: C,
    pub ring: Arc<PolyRing<C::K>>,
    pub p: PolyElement<C::K>,
    pub q: PolyElement<C::K>,
    pub x: Basic,
    pub y: Basic,
    pub degree: i64,
}

impl<C: Coefficients> VectorField<C> {
    fn k(&self) -> &C::K {
        self.c.domain()
    }

    /// `D f = Q f_x + P f_y` (`apply`).
    fn apply(&self, f: &PolyElement<C::K>) -> PolyElement<C::K> {
        self.q.mul(&f.diff(0)).add(&self.p.mul(&f.diff(1)))
    }

    /// `Q_x + P_y` (`divergence`).
    fn divergence(&self) -> PolyElement<C::K> {
        self.q.diff(0).add(&self.p.diff(1))
    }

    /// `D f / f` when the division is exact (`cofactor_of`).
    fn cofactor_of(&self, f: &PolyElement<C::K>) -> PolyResult<Option<PolyElement<C::K>>> {
        let (quotients, remainder) = self.apply(f).div(std::slice::from_ref(f))?;
        if !remainder.terms.is_empty() {
            return Ok(None);
        }
        Ok(quotients.into_iter().next())
    }

    /// The terms of total degree `n` (`homogeneous_part`).
    fn homogeneous_part(&self, f: &PolyElement<C::K>, n: i64) -> PolyElement<C::K> {
        PolyElement {
            terms: f
                .terms
                .iter()
                .filter(|(m, _)| m.iter().map(|e| i64::from(*e)).sum::<i64>() == n)
                .map(|(m, c)| (m.clone(), c.clone()))
                .collect(),
            ring: Arc::clone(&self.ring),
        }
    }

    fn monomial(&self, m: XY) -> PolyElement<C::K> {
        self.ring.term(vec![m.0, m.1], self.k().one())
    }

    fn zero(&self) -> PolyElement<C::K> {
        self.ring.zero()
    }

    fn one(&self) -> PolyElement<C::K> {
        self.ring.one()
    }

    /// `f` as an expression in `x` and `y` (`expr`).
    fn expr(&self, f: &PolyElement<C::K>) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (m, c) in f.terms.iter().rev() {
            terms.push(mul(&[
                self.c.to_expr(c)?,
                pow(&self.x, &int(i64::from(m[0])))?,
                pow(&self.y, &int(i64::from(m[1])))?,
            ])?);
        }
        if terms.is_empty() {
            Ok(Basic::zero())
        } else {
            add(&terms)
        }
    }
}

/// The exponent vectors of the monomials of total degree at most `n`
/// (`_monomials`).
fn monomials(n: i64) -> Vec<XY> {
    let mut result: Vec<XY> = Vec::new();
    for k in 0..=n.max(-1) {
        for i in (0..=k).rev() {
            result.push((i as u32, (k - i) as u32));
        }
    }
    result
}

fn xy_sum(m: &XY) -> i64 {
    i64::from(m.0) + i64::from(m.1)
}

/// `e` as a polynomial in `x, y` over the field (`_xy_poly`).
fn xy_poly<C: Coefficients>(
    e: &Basic,
    x: &Basic,
    y: &Basic,
    c: &C,
    ring: &Arc<PolyRing<C::K>>,
) -> PolyResult<PolyElement<C::K>> {
    let k: &C::K = c.domain();
    let mut terms: Terms<<C::K as Domain>::Elem> = Terms::new();
    for term in args_of_add(&expand(e)?) {
        if term == Basic::zero() {
            continue;
        }
        let mut coefficient_parts: Vec<Basic> = Vec::new();
        let (mut i, mut j) = (0u32, 0u32);
        for factor in args_of_mul(&term) {
            if !(factor.has(x) || factor.has(y)) {
                coefficient_parts.push(factor);
                continue;
            }
            let (base, exponent) = if factor.is_pow() {
                (factor.args()[0].clone(), factor.args()[1].clone())
            } else {
                (factor.clone(), Basic::one())
            };
            let power: Option<i64> = integer_of(&exponent).filter(|p| *p >= 0);
            let Some(power) = power.filter(|_| base == *x || base == *y) else {
                return Err(value_error(format!("{} is not a polynomial in {} and {}", e, x, y)));
            };
            if base == *x {
                i += power as u32;
            } else {
                j += power as u32;
            }
        }
        let value = c.element(&mul(&coefficient_parts)?)?;
        let key: Monomial = vec![i, j];
        let total = k.add(terms.get(&key).unwrap_or(&k.zero()), &value);
        if k.is_zero(&total) {
            terms.remove(&key);
        } else {
            terms.insert(key, total);
        }
    }
    Ok(PolyElement {
        terms,
        ring: Arc::clone(ring),
    })
}

/// `(P, Q, x, y)` of `y' = P(x, y)/Q(x, y)`, or `None` (`vector_field`).
pub fn vector_field(equation: &Basic, f: &Basic) -> PolyResult<Option<(Basic, Basic, Basic, Basic)>> {
    let x: Basic = match f.args().first() {
        Some(x) if x.is_symbol() => x.clone(),
        _ => return Err(value_error("the function must depend on a symbol".to_string())),
    };
    let lhs: Basic = if matches!(equation.head(), Head::Relational(RelOp::Eq)) {
        sub(&equation.args()[0], &equation.args()[1])?
    } else if !crate::core::logic::is_boolean(equation) && !matches!(equation.head(), Head::Tuple) {
        equation.clone()
    } else {
        return Err(value_error(format!("an equation is expected, got {}", equation)));
    };
    let u: Basic = crate::core::symbol::Dummy::new("y", &[]).0;
    let p: Basic = crate::core::symbol::Dummy::new("p", &[]).0;
    let derivative: Basic = crate::core::function::diff(f, &[(x.clone(), 1)])?;
    let replaced: Basic = raw_xreplace(
        &raw_xreplace(&lhs, &[(derivative, p.clone())].into_iter().collect())?,
        &[(f.clone(), u.clone())].into_iter().collect(),
    )?;
    if replaced.preorder().iter().any(|n| matches!(n.head(), Head::Derivative)) || replaced.has(f) {
        return Err(value_error(format!("a first order equation in {} is expected", f)));
    }
    let (mut a0, mut a1) = (Basic::zero(), Basic::zero());
    for term in args_of_add(&expand(&replaced)?) {
        let (coefficient, dependent) = crate::core::expr::as_independent(&term, &p, false)?;
        if dependent == Basic::one() {
            a0 = add(&[a0, coefficient])?;
        } else if dependent == p {
            a1 = add(&[a1, coefficient])?;
        } else {
            return Ok(None);
        }
    }
    if a1 == Basic::zero() {
        return Ok(None);
    }
    let cancelled: Basic = crate::polys::polytools::cancel(&neg(&div(&a0, &a1)?)?, &[])?;
    let (big_p, big_q) = crate::simplify::radsimp::fraction(&cancelled)?;
    let variables: Vec<Basic> = vec![x.clone(), u.clone()];
    let polynomial = |e: &Basic| {
        crate::core::expr::polynomial_like(e, &variables, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
    };
    if !polynomial(&big_p) || !polynomial(&big_q) || big_q == Basic::zero() {
        return Ok(None);
    }
    let name: String = match f.head() {
        Head::Undefined(g) => g.name.clone(),
        _ => f.to_string(),
    };
    let taken: HashSet<String> = big_p
        .free_symbols()
        .into_iter()
        .chain(big_q.free_symbols())
        .filter(|s| *s != u)
        .map(|s| name_of(&s))
        .collect();
    let y: Basic = if taken.contains(&name) {
        crate::core::symbol::Dummy::new(&name, &[]).0
    } else {
        symbol(&name)
    };
    let named: Vec<(Basic, Basic)> = vec![(u, y.clone())];
    Ok(Some((
        raw_xreplace(&big_p, &named.iter().cloned().collect())?,
        raw_xreplace(&big_q, &named.iter().cloned().collect())?,
        x,
        y,
    )))
}

// ---------------------------------------------------------------------------
// the Darboux polynomials

/// The unknown coefficients of one chart (`_Unknowns`).
struct Unknowns<'a, C: Coefficients> {
    field: &'a VectorField<C>,
    ring: Arc<PolyRing<C::K>>,
    f_terms: BTreeMap<XY, PolyElement<C::K>>,
    g_terms: BTreeMap<XY, PolyElement<C::K>>,
    a: Vec<usize>,
    b: Vec<usize>,
}

impl<C: Coefficients> Unknowns<'_, C> {
    /// The coefficients of `D f - g f` (`equations`).
    fn equations(&self) -> Vec<PolyElement<C::K>> {
        let field = self.field;
        let mut collected: BTreeMap<XY, PolyElement<C::K>> = BTreeMap::new();
        let zero: PolyElement<C::K> = self.ring.zero();
        for (m, c) in &self.f_terms {
            for (ij, k) in &field.apply(&field.monomial(*m)).terms {
                let key: XY = (ij[0], ij[1]);
                let entry = collected.entry(key).or_insert_with(|| zero.clone());
                *entry = entry.add(&c.mul_ground(k));
            }
            for (n, d) in &self.g_terms {
                let product: XY = (m.0 + n.0, m.1 + n.1);
                let entry = collected.entry(product).or_insert_with(|| zero.clone());
                *entry = entry.sub(&c.mul(d));
            }
        }
        collected.into_values().rev().filter(|e| !e.terms.is_empty()).collect()
    }
}

/// The linear equations solved and substituted until only nonlinear ones
/// remain (`_Propagation`).
struct Propagation<K: Domain> {
    k: K,
    a: Vec<usize>,
    b: Vec<usize>,
    gens: Vec<usize>,
    values: BTreeMap<usize, PolyElement<K>>,
    equations: Vec<PolyElement<K>>,
    consistent: bool,
}

impl<K: Domain> Propagation<K> {
    fn new(equations: Vec<PolyElement<K>>, a: &[usize], b: &[usize], k: &K, b_first: bool) -> Propagation<K> {
        let gens: Vec<usize> = if b_first { [b, a].concat() } else { [a, b].concat() };
        let mut propagation = Propagation {
            k: k.clone(),
            a: a.to_vec(),
            b: b.to_vec(),
            gens,
            values: BTreeMap::new(),
            equations,
            consistent: true,
        };
        propagation.run();
        propagation
    }

    fn run(&mut self) {
        while self.consistent {
            let (rows, rhs) = linear_rows(&self.equations, &self.gens, &self.k);
            if rows.is_empty() {
                return;
            }
            let Some((particular, basis, free)) = solve_affine(&rows, &rhs, self.gens.len(), &self.k) else {
                self.consistent = false;
                return;
            };
            let mut substitution: BTreeMap<usize, PolyElement<K>> = BTreeMap::new();
            let ring: Arc<PolyRing<K>> = Arc::clone(&self.equations[0].ring);
            let gens: Vec<PolyElement<K>> = ring.gens();
            for (i, gen) in self.gens.iter().enumerate() {
                if free.contains(&i) {
                    continue;
                }
                let mut v: PolyElement<K> = ring.ground(particular[i].clone());
                for (vector, column) in basis.iter().zip(&free) {
                    if !self.k.is_zero(&vector[i]) {
                        v = v.add(&gens[self.gens[*column]].mul_ground(&vector[i]));
                    }
                }
                substitution.insert(*gen, v);
            }
            if substitution.is_empty() {
                return;
            }
            self.substitute(substitution);
        }
    }

    fn substitute(&mut self, substitution: BTreeMap<usize, PolyElement<K>>) {
        for v in self.values.values_mut() {
            *v = substituted(v, &substitution);
        }
        self.values.extend(substitution.iter().map(|(g, v)| (*g, v.clone())));
        self.gens.retain(|g| !substitution.contains_key(g));
        self.a.retain(|g| !substitution.contains_key(g));
        self.b.retain(|g| !substitution.contains_key(g));
        let mut remaining: Vec<PolyElement<K>> = Vec::new();
        for equation in &self.equations {
            let replaced: PolyElement<K> = substituted(equation, &substitution);
            if replaced.terms.is_empty() {
                continue;
            }
            if self.gens.is_empty() || replaced.is_ground() {
                // a nonzero constant: no solution
                self.consistent = false;
                return;
            }
            remaining.push(replaced);
        }
        self.equations = remaining;
    }

    /// The value of a polynomial in the unknowns at a point of the free
    /// ones (`value_of`).
    fn value_of(&self, p: &PolyElement<K>, point: &Point<K::Elem>) -> PolyResult<K::Elem> {
        value(&substituted(p, &self.values), point)
    }
}

/// The unknown coefficients of a Darboux polynomial of degree `n`
/// (`_chart_equations`).
fn chart_equations<'a, C: Coefficients>(
    field: &'a VectorField<C>,
    n: i64,
    top: &PolyElement<C::K>,
    top_cofactor: &PolyElement<C::K>,
    fixed: Option<&BTreeMap<XY, i64>>,
) -> PolyResult<Unknowns<'a, C>> {
    let k: &C::K = field.k();
    let mut a_names: Vec<String> = Vec::new();
    for monomial in monomials(n) {
        if xy_sum(&monomial) == n && fixed.is_none_or_contains(&monomial) {
            continue;
        }
        a_names.push(format!("a_{}_{}", monomial.0, monomial.1));
    }
    let b_monomials: Vec<XY> = monomials(field.degree - 1)
        .into_iter()
        .filter(|m| xy_sum(m) != field.degree - 1)
        .collect();
    let b_names: Vec<String> = b_monomials.iter().map(|m| format!("b_{}_{}", m.0, m.1)).collect();
    let mut names: Vec<String> = [a_names.clone(), b_names].concat();
    if names.is_empty() {
        names = vec!["u".to_string()];
    }
    let ring: Arc<PolyRing<C::K>> = PolyRing::new(k.clone(), names, MonomialOrder::Lex)?;
    let gens: Vec<PolyElement<C::K>> = ring.gens();
    let mut f_terms: BTreeMap<XY, PolyElement<C::K>> = BTreeMap::new();
    let mut a: Vec<usize> = Vec::new();
    for monomial in monomials(n) {
        if xy_sum(&monomial) == n {
            if let Some(value) = fixed.and_then(|f| f.get(&monomial)) {
                f_terms.insert(monomial, ring.from_int(*value));
                continue;
            }
            if fixed.is_none() {
                f_terms.insert(monomial, ring.ground(top.coeff(&[monomial.0, monomial.1])));
                continue;
            }
        }
        let index: usize = a.len();
        a.push(index);
        f_terms.insert(monomial, gens[index].clone());
    }
    let mut g_terms: BTreeMap<XY, PolyElement<C::K>> = BTreeMap::new();
    let mut b: Vec<usize> = Vec::new();
    for monomial in monomials(field.degree - 1) {
        if xy_sum(&monomial) == field.degree - 1 {
            g_terms.insert(monomial, ring.ground(top_cofactor.coeff(&[monomial.0, monomial.1])));
            continue;
        }
        let index: usize = a_names.len() + b.len();
        b.push(index);
        g_terms.insert(monomial, gens[index].clone());
    }
    Ok(Unknowns {
        field,
        ring,
        f_terms,
        g_terms,
        a,
        b,
    })
}

trait FixedMap {
    fn is_none_or_contains(&self, m: &XY) -> bool;
}

impl FixedMap for Option<&BTreeMap<XY, i64>> {
    fn is_none_or_contains(&self, m: &XY) -> bool {
        match self {
            None => true,
            Some(f) => f.contains_key(m),
        }
    }
}

/// A share of the enclosing limit for one optional step
/// (`_share_of_time`).
fn share_of_time() -> Option<f64> {
    let limit: Option<f64> = crate::timeout::remaining_time().or(crate::settings::settings().timeout);
    limit.map(|l| l / 4.0)
}

fn attempt<T>(seconds: Option<f64>, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(seconds, f)
}

/// Gaussian elimination with case splits on the pivots
/// (`_ParametricElimination`).
struct ParametricElimination<K: Domain> {
    ring: Arc<PolyRing<K>>,
    n: usize,
    leaves: Vec<Leaf<K>>,
    exceeded: bool,
}

impl<K: Domain> ParametricElimination<K> {
    /// The number of leaves after which the elimination is given up.
    const LIMIT: usize = 400;

    fn run(
        &mut self,
        rows: Vec<Row<K>>,
        columns: Vec<usize>,
        equations: Vec<PolyElement<K>>,
        nonzero: Vec<PolyElement<K>>,
        fixed: Vec<(usize, PolyElement<K>)>,
    ) -> PolyResult<()> {
        if self.exceeded {
            return Ok(());
        }
        let mut live: Vec<Row<K>> = Vec::new();
        let mut equations: Vec<PolyElement<K>> = equations.into_iter().filter(|e| !e.terms.is_empty()).collect();
        if equations.iter().any(|e| e.is_ground()) {
            return Ok(());
        }
        for row in rows {
            if row[..self.n].iter().all(|entry| entry.terms.is_empty()) {
                let constant: &PolyElement<K> = &row[self.n];
                if !constant.terms.is_empty() {
                    if constant.is_ground() {
                        return Ok(());
                    }
                    equations.push(constant.clone());
                }
            } else {
                live.push(row);
            }
        }
        if live.is_empty() || columns.is_empty() {
            self.leaves.push((equations, nonzero, fixed));
            if self.leaves.len() > Self::LIMIT {
                self.exceeded = true;
            }
            return Ok(());
        }
        if let Some((i, j)) = Self::constant_pivot(&live, &columns) {
            let eliminated: Vec<Row<K>> = self.eliminated(&live, i, j, &nonzero)?;
            let rest: Vec<usize> = columns.iter().copied().filter(|c| *c != j).collect();
            return self.run(eliminated, rest, equations, nonzero, fixed);
        }
        let (i, j) = Self::smallest_pivot(&live, &columns)?;
        let p: PolyElement<K> = live[i][j].clone();
        // the pivot vanishes
        if let Some((index, v)) = self.substitution(&p)? {
            let values: BTreeMap<usize, PolyElement<K>> = [(index, v.clone())].into_iter().collect();
            let rows: Vec<Row<K>> = live
                .iter()
                .map(|row| row.iter().map(|entry| substituted(entry, &values)).collect())
                .collect();
            let mut fixed_more = fixed.clone();
            fixed_more.push((index, v));
            self.run(
                rows,
                columns.clone(),
                equations.iter().map(|e| substituted(e, &values)).collect(),
                nonzero.iter().map(|e| substituted(e, &values)).collect(),
                fixed_more,
            )?;
        } else {
            let mut zeroed: Vec<Row<K>> = live.clone();
            zeroed[i][j] = self.ring.zero();
            let mut more: Vec<PolyElement<K>> = equations.clone();
            more.push(p.clone());
            self.run(zeroed, columns.clone(), more, nonzero.clone(), fixed.clone())?;
        }
        // the pivot does not vanish
        let mut nonzero_more: Vec<PolyElement<K>> = nonzero;
        nonzero_more.push(p);
        let eliminated: Vec<Row<K>> = self.eliminated(&live, i, j, &nonzero_more)?;
        let rest: Vec<usize> = columns.iter().copied().filter(|c| *c != j).collect();
        self.run(eliminated, rest, equations, nonzero_more, fixed)
    }

    fn constant_pivot(rows: &[Row<K>], columns: &[usize]) -> Option<(usize, usize)> {
        for (i, row) in rows.iter().enumerate() {
            for &j in columns {
                let entry: &PolyElement<K> = &row[j];
                if !entry.terms.is_empty() && entry.is_ground() {
                    return Some((i, j));
                }
            }
        }
        None
    }

    fn smallest_pivot(rows: &[Row<K>], columns: &[usize]) -> PolyResult<(usize, usize)> {
        let mut best: Option<(i64, usize, usize)> = None;
        for (i, row) in rows.iter().enumerate() {
            for &j in columns {
                let entry: &PolyElement<K> = &row[j];
                if !entry.terms.is_empty() {
                    let degree: i64 = entry.degree();
                    if best.as_ref().is_none_or_less(degree) {
                        best = Some((degree, i, j));
                    }
                }
            }
        }
        best.map(|(_, i, j)| (i, j))
            .ok_or_else(|| value_error("no pivot".to_string()))
    }

    /// `p = 0` solved for a parameter when `p` has degree one
    /// (`_substitution`).
    fn substitution(&self, p: &PolyElement<K>) -> PolyResult<Option<(usize, PolyElement<K>)>> {
        if p.degree() != 1 {
            return Ok(None);
        }
        let k: &K = &self.ring.domain;
        for (index, gen) in self.ring.gens().iter().enumerate() {
            let coefficient: K::Elem = p.coeff(&gen.LM());
            if !k.is_zero(&coefficient) {
                let rest: PolyElement<K> = p.sub(&gen.mul_ground(&coefficient));
                return Ok(Some((index, rest.mul_ground(&k.neg(&k.invert(&coefficient)?)))));
            }
        }
        Ok(None)
    }

    /// The rows with the unknown `j` eliminated by the pivot row `i`
    /// (`_eliminated`).
    fn eliminated(&self, rows: &[Row<K>], i: usize, j: usize, nonzero: &[PolyElement<K>]) -> PolyResult<Vec<Row<K>>> {
        let pivot_row: &Row<K> = &rows[i];
        let p: &PolyElement<K> = &pivot_row[j];
        let mut result: Vec<Row<K>> = Vec::new();
        for (k, row) in rows.iter().enumerate() {
            if k == i {
                continue;
            }
            let c: &PolyElement<K> = &row[j];
            if c.terms.is_empty() {
                result.push(row.clone());
                continue;
            }
            let combined: Row<K> = row
                .iter()
                .zip(pivot_row)
                .map(|(entry, pivot_entry)| entry.mul(p).sub(&pivot_entry.mul(c)))
                .collect();
            result.push(Self::reduced(combined, nonzero)?);
        }
        Ok(result)
    }

    /// The row divided by the common factors known to be nonzero
    /// (`_reduced`).
    fn reduced(mut row: Row<K>, nonzero: &[PolyElement<K>]) -> PolyResult<Row<K>> {
        let entries: Vec<&PolyElement<K>> = row.iter().filter(|e| !e.terms.is_empty()).collect();
        if entries.is_empty() || entries.iter().any(|e| e.is_ground()) {
            return Ok(row);
        }
        for factor in nonzero {
            if factor.is_ground() {
                continue;
            }
            'divide: loop {
                let mut quotients: Row<K> = Vec::new();
                for entry in &row {
                    if entry.terms.is_empty() {
                        quotients.push(entry.clone());
                        continue;
                    }
                    let (q, r) = entry.div(std::slice::from_ref(factor))?;
                    if !r.terms.is_empty() {
                        break 'divide;
                    }
                    quotients.push(q.into_iter().next().expect("a quotient"));
                }
                row = quotients;
            }
        }
        Ok(row)
    }
}

trait LessDegree {
    fn is_none_or_less(&self, degree: i64) -> bool;
}

impl LessDegree for Option<&(i64, usize, usize)> {
    fn is_none_or_less(&self, degree: i64) -> bool {
        match self {
            None => true,
            Some(best) => degree < best.0,
        }
    }
}

/// The values of the parameters on the solutions of the nonlinear
/// remainder (`_parameter_points`).
fn parameter_points<C: Coefficients>(
    propagation: &Propagation<C::K>,
    parameters: &[usize],
    unknowns: &[usize],
    c: &C,
) -> PolyResult<Option<Vec<Point<<C::K as Domain>::Elem>>>> {
    let k: &C::K = c.domain();
    let (a, b) = (unknowns, parameters);
    if b.is_empty() {
        return Ok(Some(vec![Point::new()]));
    }
    let ring: Arc<PolyRing<C::K>> = PolyRing::new(
        k.clone(),
        (0..b.len()).map(|i| format!("b{}", i)).collect(),
        MonomialOrder::Lex,
    )?;
    let column: BTreeMap<usize, usize> = a.iter().enumerate().map(|(i, g)| (*g, i)).collect();
    let n: usize = a.len();
    let mut rows: Vec<Row<C::K>> = Vec::new();
    let mut constraints: Vec<PolyElement<C::K>> = Vec::new();
    for equation in &propagation.equations {
        let mut split: BTreeMap<usize, Terms<<C::K as Domain>::Elem>> = BTreeMap::new();
        for (monomial, coefficient) in &equation.terms {
            let a_part: Vec<(usize, u32)> = monomial
                .iter()
                .enumerate()
                .filter(|(g, e)| **e > 0 && column.contains_key(g))
                .map(|(g, e)| (g, *e))
                .collect();
            if a_part.iter().map(|(_, e)| *e).sum::<u32>() > 1 {
                return Ok(None);
            }
            let target: usize = a_part.first().map(|(g, _)| column[g]).unwrap_or(n);
            let b_monomial: Monomial = b.iter().map(|g| monomial[*g]).collect();
            split.entry(target).or_default().insert(b_monomial, coefficient.clone());
        }
        if a.is_empty() {
            constraints.push(ring.from_dict(split.remove(&n).unwrap_or_default()));
            continue;
        }
        let mut row: Row<C::K> = vec![ring.zero(); n + 1];
        for (target, terms) in split {
            row[target] = ring.from_dict(terms);
        }
        rows.push(row);
    }
    let mut elimination = ParametricElimination {
        ring: Arc::clone(&ring),
        n,
        leaves: Vec::new(),
        exceeded: false,
    };
    elimination.run(rows, (0..n).collect(), constraints, Vec::new(), Vec::new())?;
    if elimination.exceeded {
        return Ok(None);
    }
    let mut points: Vec<Point<<C::K as Domain>::Elem>> = Vec::new();
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    for (equations, nonzero, fixed) in &elimination.leaves {
        let fixed_indices: HashSet<usize> = fixed.iter().map(|(i, _)| *i).collect();
        let free: Vec<usize> = (0..b.len()).filter(|i| !fixed_indices.contains(i)).collect();
        if equations.is_empty() && !free.is_empty() {
            // a continuum of cofactors
            continue;
        }
        'values: for values in leaf_points(equations, &free, c)? {
            let mut point: Point<<C::K as Domain>::Elem> = free.iter().copied().zip(values).collect();
            for (index, v) in fixed.iter().rev() {
                let at: <C::K as Domain>::Elem = value(v, &point)?;
                point.insert(*index, at);
            }
            for e in nonzero {
                if k.is_zero(&value(e, &point)?) {
                    continue 'values;
                }
            }
            let key: Vec<String> = (0..b.len()).map(|i| c.key(&point[&i])).collect();
            if !seen.insert(key) {
                continue;
            }
            points.push((0..b.len()).map(|i| (b[i], point[&i].clone())).collect());
        }
    }
    Ok(Some(points))
}

/// The number of combinations of factors up to which a leaf system is
/// solved case by case (`_CASES`).
const CASES: usize = 12;

/// `p` in `ring`, whose variables are the `indices` of its own
/// (`_restricted`).
fn restricted<K: Domain>(p: &PolyElement<K>, indices: &[usize], ring: &Arc<PolyRing<K>>) -> PolyElement<K> {
    PolyElement {
        terms: p
            .terms
            .iter()
            .map(|(m, c)| (indices.iter().map(|i| m[*i]).collect(), c.clone()))
            .collect(),
        ring: Arc::clone(ring),
    }
}

/// The solutions in `K` of the equations of a leaf (`_leaf_points`).
fn leaf_points<C: Coefficients>(
    equations: &[PolyElement<C::K>],
    free: &[usize],
    c: &C,
) -> PolyResult<Vec<Vec<<C::K as Domain>::Elem>>> {
    if free.is_empty() {
        return Ok(if equations.iter().all(|e| e.terms.is_empty()) {
            vec![Vec::new()]
        } else {
            Vec::new()
        });
    }
    let k: &C::K = c.domain();
    let ring: Arc<PolyRing<C::K>> = PolyRing::new(
        k.clone(),
        free.iter().map(|i| format!("b{}", i)).collect(),
        MonomialOrder::Lex,
    )?;
    let restricted_eqs: Vec<PolyElement<C::K>> = equations.iter().map(|e| restricted(e, free, &ring)).collect();
    let mut cases: Vec<Vec<PolyElement<C::K>>> = Vec::new();
    for e in &restricted_eqs {
        let factors = c.factor(e)?;
        if factors.is_empty() {
            return Ok(Vec::new());
        }
        cases.push(factors);
    }
    cases.sort_by_key(|f| f.len());
    let combinations: usize = cases.iter().map(|f| f.len()).product();
    let highest: i64 = cases.iter().flatten().map(|f| f.degree()).max().unwrap_or(0);
    let degree: i64 = restricted_eqs.iter().map(|e| e.degree()).max().unwrap_or(0);
    let n: usize = free.len();
    let points = if combinations <= CASES && 2 * highest <= degree {
        attempt(share_of_time(), || points_of_cases(&cases, &[], n, c))?
    } else {
        let products: Vec<PolyElement<C::K>> = cases
            .iter()
            .map(|factors| factors.iter().fold(ring.one(), |acc, f| acc.mul(f)))
            .collect();
        attempt(share_of_time(), || points_in_domain(&products, n, c))?.flatten()
    };
    Ok(points.unwrap_or_default())
}

/// The points of the cases, one factor of each equation in turn
/// (`_points_of_cases`).
fn points_of_cases<C: Coefficients>(
    cases: &[Vec<PolyElement<C::K>>],
    chosen: &[PolyElement<C::K>],
    n: usize,
    c: &C,
) -> PolyResult<Vec<Vec<<C::K as Domain>::Elem>>> {
    if cases.is_empty() {
        return Ok(points_in_domain(chosen, n, c)?.unwrap_or_default());
    }
    let mut found = Vec::new();
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    for factor in &cases[0] {
        let mut more: Vec<PolyElement<C::K>> = chosen.to_vec();
        more.push(factor.clone());
        for point in points_of_cases(&cases[1..], &more, n, c)? {
            let key: Vec<String> = point.iter().map(|v| c.key(v)).collect();
            if seen.insert(key) {
                found.push(point);
            }
        }
    }
    Ok(found)
}

/// The reduced lex Gröbner basis of a zero-dimensional system in the
/// first `n` variables by FGLM, `[1]` when inconsistent, `None` when not
/// zero-dimensional (`_lex_basis`).
fn lex_basis<K: Domain>(polys: &[PolyElement<K>], n: usize) -> PolyResult<Option<Vec<PolyElement<K>>>> {
    let ring: &Arc<PolyRing<K>> = &polys[0].ring;
    let graded: Arc<PolyRing<K>> =
        PolyRing::new(ring.domain.clone(), ring.names[..n].to_vec(), MonomialOrder::Grevlex)?;
    let indices: Vec<usize> = (0..n).collect();
    let restricted_polys: Vec<PolyElement<K>> = polys.iter().map(|p| restricted(p, &indices, &graded)).collect();
    let basis: Vec<PolyElement<K>> = crate::polys::ideals::groebner(&restricted_polys, &graded)?;
    if basis.iter().any(|g| g.is_ground() && !g.terms.is_empty()) {
        return Ok(Some(vec![ring.one()]));
    }
    if !crate::polys::fglm::is_zero_dimensional(&basis, &graded) {
        return Ok(None);
    }
    let pad: usize = ring.ngens - n;
    let lex: Vec<PolyElement<K>> = crate::polys::fglm::fglm(&basis, &graded, &MonomialOrder::Lex)?;
    Ok(Some(
        lex.into_iter()
            .map(|g| PolyElement {
                terms: g
                    .terms
                    .into_iter()
                    .map(|(mut m, c)| {
                        m.extend(std::iter::repeat(0).take(pad));
                        (m, c)
                    })
                    .collect(),
                ring: Arc::clone(ring),
            })
            .collect(),
    ))
}

/// The monic gcd of two polynomials in one variable (`_univariate_gcd`).
fn univariate_gcd<K: Domain>(f: &PolyElement<K>, g: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
    let (mut f, mut g) = (f.clone(), g.clone());
    while !g.terms.is_empty() {
        let r: PolyElement<K> = f.rem(std::slice::from_ref(&g))?;
        f = g;
        g = r;
    }
    f.monic()
}

/// The points with coordinates in `K` of a zero-dimensional system in the
/// first `n` variables (`_points_in_domain`).
fn points_in_domain<C: Coefficients>(
    polys: &[PolyElement<C::K>],
    n: usize,
    c: &C,
) -> PolyResult<Option<Vec<Vec<<C::K as Domain>::Elem>>>> {
    let mut polys: Vec<PolyElement<C::K>> = polys.iter().filter(|p| !p.terms.is_empty()).cloned().collect();
    if n == 0 {
        return Ok(Some(if polys.is_empty() { vec![Vec::new()] } else { Vec::new() }));
    }
    if polys.is_empty() {
        return Ok(None);
    }
    if polys.iter().any(|p| p.is_ground()) {
        return Ok(Some(Vec::new()));
    }
    let last: usize = n - 1;
    let univariate = |p: &PolyElement<C::K>| p.terms.keys().all(|m| m[..last].iter().all(|e| *e == 0));
    let mut alone: Vec<PolyElement<C::K>> = polys.iter().filter(|p| univariate(p)).cloned().collect();
    if alone.is_empty() {
        let Some(basis) = lex_basis(&polys, n)? else {
            return Ok(None);
        };
        if basis.len() == 1 && basis[0].is_ground() {
            return Ok(Some(Vec::new()));
        }
        polys = basis;
        alone = polys.iter().filter(|p| univariate(p)).cloned().collect();
        if alone.is_empty() {
            return Ok(None);
        }
    }
    let mut polynomial: PolyElement<C::K> = alone[0].clone();
    for p in &alone[1..] {
        polynomial = univariate_gcd(&polynomial, p)?;
    }
    let others: Vec<PolyElement<C::K>> = polys.iter().filter(|p| !univariate(p)).cloned().collect();
    let mut points = Vec::new();
    for root in c.roots(&polynomial, last)? {
        let at: Point<<C::K as Domain>::Elem> = [(last, root.clone())].into_iter().collect();
        let rest: Vec<PolyElement<C::K>> = others.iter().map(|p| evaluated(p, &at)).collect();
        let Some(below) = points_in_domain(&rest, last, c)? else {
            return Ok(None);
        };
        for mut point in below {
            point.push(root.clone());
            points.push(point);
        }
    }
    Ok(Some(points))
}

/// A few members of an affine space (`_members`).
fn members<K: Domain>(particular: &[K::Elem], basis: &[Vec<K::Elem>], k: &K) -> PolyResult<Vec<Vec<K::Elem>>> {
    let sparse: Vec<K::Elem> = sparse_member(particular, basis, k)?;
    let mut result: Vec<Vec<K::Elem>> = vec![sparse.clone()];
    for vector in basis {
        result.push(sparse.iter().zip(vector).map(|(p, v)| k.add(p, v)).collect());
    }
    if basis.len() > 1 {
        let mut generic: Vec<K::Elem> = sparse;
        for (i, vector) in basis.iter().enumerate() {
            let scale: K::Elem = k.from_int(i as i64 + 1);
            generic = generic
                .iter()
                .zip(vector)
                .map(|(p, v)| k.add(p, &k.mul(&scale, v)))
                .collect();
        }
        result.push(generic);
    }
    Ok(result)
}

/// The member with the fewest nonzero entries reached greedily
/// (`_sparse_member`).
fn sparse_member<K: Domain>(particular: &[K::Elem], basis: &[Vec<K::Elem>], k: &K) -> PolyResult<Vec<K::Elem>> {
    let mut member: Vec<K::Elem> = particular.to_vec();
    for _ in 0..2 {
        for vector in basis {
            let mut best: Vec<K::Elem> = member.clone();
            for (i, v) in vector.iter().enumerate() {
                if k.is_zero(v) || k.is_zero(&member[i]) {
                    continue;
                }
                let u: K::Elem = k.neg(&k.exquo(&member[i], v)?);
                let candidate: Vec<K::Elem> = member
                    .iter()
                    .zip(vector)
                    .map(|(p, w)| k.add(p, &k.mul(&u, w)))
                    .collect();
                if nonzero_entries(&candidate, k) < nonzero_entries(&best, k) {
                    best = candidate;
                }
            }
            member = best;
        }
    }
    Ok(member)
}

fn nonzero_entries<K: Domain>(vector: &[K::Elem], k: &K) -> usize {
    vector.iter().filter(|v| !k.is_zero(v)).count()
}

/// The Darboux polynomials of one chart (`_chart_solutions`).
fn chart_solutions<C: Coefficients>(unknowns: &Unknowns<'_, C>) -> PolyResult<Vec<Pair<C::K>>> {
    let field = unknowns.field;
    let k: &C::K = field.k();
    let equations: Vec<PolyElement<C::K>> = unknowns.equations();
    let mut propagation: Propagation<C::K> = Propagation::new(equations.clone(), &unknowns.a, &unknowns.b, k, false);
    if !propagation.consistent {
        return Ok(Vec::new());
    }
    let (mut parameters, mut free) = (propagation.b.clone(), propagation.a.clone());
    let points: Vec<Point<<C::K as Domain>::Elem>> = if !propagation.equations.is_empty() {
        let swapped: Propagation<C::K> = Propagation::new(equations, &unknowns.a, &unknowns.b, k, true);
        if swapped.consistent && swapped.a.len() < propagation.b.len() {
            parameters = swapped.a.clone();
            free = swapped.b.clone();
            propagation = swapped;
        }
        match parameter_points(&propagation, &parameters, &free, &field.c)? {
            Some(points) => points,
            None => return Ok(Vec::new()),
        }
    } else {
        vec![Point::new()]
    };
    let mut found: Vec<Pair<C::K>> = Vec::new();
    for point in points {
        let remaining: Vec<PolyElement<C::K>> = propagation
            .equations
            .iter()
            .map(|e| evaluated(e, &point))
            .filter(|e| !e.terms.is_empty())
            .collect();
        if remaining.iter().any(|e| e.is_ground() || e.degree() > 1) {
            continue;
        }
        let (rows, rhs) = linear_rows(&remaining, &free, k);
        let Some((particular, basis, _)) = solve_affine(&rows, &rhs, free.len(), k) else {
            continue;
        };
        for member in members(&particular, &basis, k)? {
            let mut values: Point<<C::K as Domain>::Elem> = point.clone();
            values.extend(free.iter().copied().zip(member));
            for gen in &parameters {
                values.entry(*gen).or_insert_with(|| k.zero());
            }
            let mut f_terms: Terms<<C::K as Domain>::Elem> = Terms::new();
            for (m, c) in &unknowns.f_terms {
                f_terms.insert(vec![m.0, m.1], propagation.value_of(c, &values)?);
            }
            let mut g_terms: Terms<<C::K as Domain>::Elem> = Terms::new();
            for (m, c) in &unknowns.g_terms {
                g_terms.insert(vec![m.0, m.1], propagation.value_of(c, &values)?);
            }
            let f: PolyElement<C::K> = field.ring.from_dict(f_terms);
            let g: PolyElement<C::K> = field.ring.from_dict(g_terms);
            if f.degree() <= 0 {
                continue;
            }
            if field.apply(&f) != g.mul(&f) {
                continue;
            }
            found.push((f, g));
        }
    }
    Ok(found)
}

/// `P_d`, `Q_d` and `H = x P_d - y Q_d` (`_top_parts`).
fn top_parts<C: Coefficients>(field: &VectorField<C>) -> Triple<C::K> {
    let d: i64 = field.degree;
    let p_d: PolyElement<C::K> = field.homogeneous_part(&field.p, d);
    let q_d: PolyElement<C::K> = field.homogeneous_part(&field.q, d);
    let h: PolyElement<C::K> = field.monomial((1, 0)).mul(&p_d).sub(&field.monomial((0, 1)).mul(&q_d));
    (p_d, q_d, h)
}

/// The possible highest degree parts of a Darboux polynomial of degree
/// `n` (`_top_candidates`).
fn top_candidates<C: Coefficients>(
    field: &VectorField<C>,
    n: i64,
    h: &PolyElement<C::K>,
) -> PolyResult<Vec<PolyElement<C::K>>> {
    let irreducible: Vec<(PolyElement<C::K>, i64)> = field
        .c
        .factor(h)?
        .into_iter()
        .map(|f| {
            let d = f.degree();
            (f, d)
        })
        .collect();
    let mut candidates: Vec<PolyElement<C::K>> = Vec::new();
    fn extend<K: Domain>(
        irreducible: &[(PolyElement<K>, i64)],
        index: usize,
        degree_left: i64,
        product: PolyElement<K>,
        candidates: &mut Vec<PolyElement<K>>,
    ) {
        if degree_left == 0 {
            if candidates.iter().all(|c| product != *c) {
                candidates.push(product);
            }
            return;
        }
        for k in index..irreducible.len() {
            let (h, degree) = &irreducible[k];
            if *degree <= degree_left {
                extend(irreducible, k, degree_left - degree, product.mul(h), candidates);
            }
        }
    }
    extend(&irreducible, 0, n, field.one(), &mut candidates);
    Ok(candidates)
}

/// Darboux polynomials of total degree `n` with their cofactors
/// (`_darboux_of_degree`).
fn darboux_of_degree<C: Coefficients>(field: &VectorField<C>, n: i64) -> PolyResult<Vec<Pair<C::K>>> {
    let (p_d, q_d, h) = top_parts(field);
    let mut found: Vec<Pair<C::K>> = Vec::new();
    if !h.terms.is_empty() {
        for top in top_candidates(field, n, &h)? {
            let d_top: PolyElement<C::K> = q_d.mul(&top.diff(0)).add(&p_d.mul(&top.diff(1)));
            let (quotients, remainder) = d_top.div(std::slice::from_ref(&top))?;
            if !remainder.terms.is_empty() {
                continue;
            }
            found.extend(chart_solutions(&chart_equations(field, n, &top, &quotients[0], None)?)?);
        }
        return Ok(found);
    }
    // dicritical: D_d = k (x d/dx + y d/dy), the cofactor's top part is N k
    let k_part: PolyElement<C::K> = if !q_d.terms.is_empty() {
        q_d.div(&[field.monomial((1, 0))])?.0.remove(0)
    } else {
        p_d.div(&[field.monomial((0, 1))])?.0.remove(0)
    };
    let top_cofactor: PolyElement<C::K> = k_part.mul_ground(&field.k().from_int(n));
    let top_monomials: Vec<XY> = (0..=n).map(|j| ((n - j) as u32, j as u32)).collect();
    for leading in 0..=(n as usize) {
        let mut fixed: BTreeMap<XY, i64> = BTreeMap::new();
        for m in &top_monomials[..leading] {
            fixed.insert(*m, 0);
        }
        fixed.insert(top_monomials[leading], 1);
        found.extend(chart_solutions(&chart_equations(
            field,
            n,
            &field.zero(),
            &top_cofactor,
            Some(&fixed),
        )?)?);
    }
    Ok(found)
}

/// The irreducible Darboux polynomials of degree at most `degree`
/// (`_darboux`).
fn darboux<C: Coefficients>(field: &VectorField<C>, degree: i64) -> PolyResult<Vec<Pair<C::K>>> {
    let mut found: Vec<Pair<C::K>> = Vec::new();
    for n in 1..=degree {
        add_darboux(field, n, &mut found)?;
    }
    Ok(found)
}

/// The irreducible Darboux polynomials of degree `n` added to `found`
/// (`_add_darboux`).
fn add_darboux<C: Coefficients>(field: &VectorField<C>, n: i64, found: &mut Vec<Pair<C::K>>) -> PolyResult<()> {
    for (f, _) in darboux_of_degree(field, n)? {
        for h in field.c.factor(&f)? {
            if found.iter().any(|(g, _)| h == *g) {
                continue;
            }
            if let Some(cofactor) = field.cofactor_of(&h)? {
                found.push((h, cofactor));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// exponential factors

/// `1` and the products of Darboux polynomials of total degree at most
/// `degree`, with their cofactors (`_denominators`).
fn denominators<C: Coefficients>(field: &VectorField<C>, polynomials: &[Pair<C::K>], degree: i64) -> Vec<Pair<C::K>> {
    let mut result: Vec<Pair<C::K>> = vec![(field.one(), field.zero())];
    let degrees: Vec<i64> = polynomials.iter().map(|(f, _)| f.degree()).collect();
    for size in 1..=degree.max(0) as usize {
        for choice in combinations_with_replacement(polynomials.len(), size) {
            if choice.iter().map(|i| degrees[*i]).sum::<i64>() > degree {
                continue;
            }
            let (mut b, mut g) = (field.one(), field.zero());
            for i in choice {
                b = b.mul(&polynomials[i].0);
                g = g.add(&polynomials[i].1);
            }
            result.push((b, g));
        }
    }
    result
}

/// Python's `itertools.combinations_with_replacement(range(n), size)`.
fn combinations_with_replacement(n: usize, size: usize) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    if n == 0 {
        return result;
    }
    let mut indices: Vec<usize> = vec![0; size];
    loop {
        result.push(indices.clone());
        let Some(i) = (0..size).rev().find(|i| indices[*i] != n - 1) else {
            return result;
        };
        let next: usize = indices[i] + 1;
        for v in indices.iter_mut().skip(i) {
            *v = next;
        }
    }
}

/// The exponential factors `exp(A/B)` with `deg A <= bound`
/// (`_exponential_with_denominator`).
fn exponential_with_denominator<C: Coefficients>(
    field: &VectorField<C>,
    b: &PolyElement<C::K>,
    g_b: &PolyElement<C::K>,
    bound: i64,
) -> Vec<Pair<C::K>> {
    let k: &C::K = field.k();
    let a_monomials: Vec<XY> = monomials(bound);
    let l_monomials: Vec<XY> = monomials(field.degree - 1);
    let mut columns: Vec<Terms<<C::K as Domain>::Elem>> = Vec::new();
    for m in &a_monomials {
        let monomial: PolyElement<C::K> = field.monomial(*m);
        columns.push(field.apply(&monomial).sub(&g_b.mul(&monomial)).terms);
    }
    for m in &l_monomials {
        columns.push(b.mul(&field.monomial(*m)).neg().terms);
    }
    let image: Vec<Monomial> = {
        let mut all: Vec<Monomial> = columns.iter().flat_map(|c| c.keys().cloned()).collect();
        all.sort();
        all.dedup();
        all
    };
    let rows: Vec<Vec<<C::K as Domain>::Elem>> = image
        .iter()
        .map(|monomial| {
            columns
                .iter()
                .map(|c| c.get(monomial).cloned().unwrap_or_else(|| k.zero()))
                .collect()
        })
        .collect();
    let zeros: Vec<<C::K as Domain>::Elem> = vec![k.zero(); rows.len()];
    let Some((_, basis, _)) = solve_affine(&rows, &zeros, columns.len(), k) else {
        return Vec::new();
    };
    let mut result: Vec<Pair<C::K>> = Vec::new();
    for vector in basis {
        let a: PolyElement<C::K> = field.ring.from_dict(
            a_monomials
                .iter()
                .enumerate()
                .map(|(i, m)| (vec![m.0, m.1], vector[i].clone()))
                .collect(),
        );
        let l: PolyElement<C::K> = field.ring.from_dict(
            l_monomials
                .iter()
                .enumerate()
                .map(|(i, m)| (vec![m.0, m.1], vector[a_monomials.len() + i].clone()))
                .collect(),
        );
        if l.terms.is_empty() || a.terms.is_empty() {
            continue;
        }
        result.push((a, l));
    }
    result
}

/// The exponential factors `(A, B, L)` (`_exponential`).
fn exponential<C: Coefficients>(field: &VectorField<C>, polynomials: &[Pair<C::K>], degree: i64) -> Vec<Triple<C::K>> {
    let mut found: Vec<Triple<C::K>> = Vec::new();
    for (b, g_b) in denominators(field, polynomials, degree) {
        let bound: i64 = degree.max(field.degree) + b.degree();
        for (a, l) in exponential_with_denominator(field, &b, &g_b, bound) {
            // exp(B A0/B) is exp(A0) again
            if found.iter().any(|(a0, b0, _)| a.mul(b0) == a0.mul(&b)) {
                continue;
            }
            found.push((a, b.clone(), l));
        }
    }
    found
}

// ---------------------------------------------------------------------------
// the exponents

fn ops(e: &Basic) -> i64 {
    crate::core::function::count_ops(e).unwrap_or(0)
}

type Simplicity = (usize, usize, i64, String);

/// The sort key of the simplest candidate (`_simplicity`).
fn simplicity(e: &Basic) -> Simplicity {
    let mut atoms: Vec<Basic> = e.preorder();
    atoms.sort_by_key(|a| a.to_string());
    atoms.dedup();
    let exponentials: usize = atoms
        .iter()
        .filter(|a| matches!(a.head(), Head::Function(Function::Exp)))
        .count();
    let fractional: usize = atoms
        .iter()
        .filter(|a| a.is_pow() && !is_integer_node(&a.args()[1]))
        .count();
    (exponentials, fractional, ops(e), e.to_string())
}

fn simplest(candidates: Vec<Basic>) -> Option<Basic> {
    let mut best: Option<(Simplicity, Basic)> = None;
    for c in candidates {
        let key: Simplicity = simplicity(&c);
        if best.as_ref().is_none_or_greater(&key) {
            best = Some((key, c));
        }
    }
    best.map(|(_, c)| c)
}

trait GreaterKey {
    fn is_none_or_greater(&self, key: &Simplicity) -> bool;
}

impl GreaterKey for Option<&(Simplicity, Basic)> {
    fn is_none_or_greater(&self, key: &Simplicity) -> bool {
        match self {
            None => true,
            Some((best, _)) => key < best,
        }
    }
}

/// The Darboux polynomials, the exponential factors and the linear
/// algebra of their exponents (`_Darboux`).
struct Darboux<'a, C: Coefficients> {
    field: &'a VectorField<C>,
    polynomials: Vec<Pair<C::K>>,
    exponentials: Vec<Triple<C::K>>,
    cofactors: Vec<PolyElement<C::K>>,
    monomials: Vec<XY>,
    rows: Vec<Vec<<C::K as Domain>::Elem>>,
}

impl<'a, C: Coefficients> Darboux<'a, C> {
    fn new(field: &'a VectorField<C>, polynomials: Vec<Pair<C::K>>, degree: i64) -> Darboux<'a, C> {
        let exponentials: Vec<Triple<C::K>> = exponential(field, &polynomials, degree);
        let cofactors: Vec<PolyElement<C::K>> = polynomials
            .iter()
            .map(|(_, g)| g.clone())
            .chain(exponentials.iter().map(|(_, _, l)| l.clone()))
            .collect();
        let monomials: Vec<XY> = monomials(field.degree - 1);
        let rows = monomials
            .iter()
            .map(|m| cofactors.iter().map(|g| g.coeff(&[m.0, m.1])).collect())
            .collect();
        Darboux {
            field,
            polynomials,
            exponentials,
            cofactors,
            monomials,
            rows,
        }
    }

    /// The vector scaled to integers when its entries are rational
    /// (`_integers`).
    fn integers(&self, vector: &[<C::K as Domain>::Elem]) -> Option<Vec<Basic>> {
        let mut rationals: Vec<Rat> = Vec::new();
        for v in vector {
            rationals.push(self.field.c.rational(v)?);
        }
        let mut scale: Int = Int::from(1);
        for q in &rationals {
            scale = crate::numbers::lcm(&scale, &crate::numbers::denom(q));
        }
        Some(
            rationals
                .iter()
                .map(|q| Basic::integer(crate::numbers::numer(q) * (scale.clone() / crate::numbers::denom(q))))
                .collect(),
        )
    }

    /// The simplest combination of the cofactors vanishing
    /// (`first_integral`).
    fn first_integral(&self) -> PolyResult<Option<Basic>> {
        if self.cofactors.is_empty() {
            return Ok(None);
        }
        let k: &C::K = self.field.k();
        let zeros: Vec<<C::K as Domain>::Elem> = vec![k.zero(); self.rows.len()];
        let Some((_, basis, _)) = solve_affine(&self.rows, &zeros, self.cofactors.len(), k) else {
            return Ok(None);
        };
        if basis.is_empty() {
            return Ok(None);
        }
        let mut candidates: Vec<Basic> = Vec::new();
        for vector in &basis {
            let mut exponents: Vec<Basic> = match self.integers(vector) {
                Some(scaled) => scaled,
                None => vector
                    .iter()
                    .map(|v| self.field.c.to_expr(v))
                    .collect::<PolyResult<_>>()?,
            };
            let first: Basic = exponents
                .iter()
                .find(|e| **e != Basic::zero())
                .cloned()
                .unwrap_or_else(Basic::one);
            if crate::core::expr::is(&first, crate::core::assumptions::Predicate::negative) == Some(true) {
                exponents = exponents.iter().map(neg).collect::<PolyResult<_>>()?;
            }
            let candidate: Basic = self.product(&exponents, true)?;
            if !is_constant_integral(&tidy(&candidate)?, &self.field.x, &self.field.y) {
                candidates.push(candidate);
            }
        }
        Ok(simplest(candidates))
    }

    /// The simplest Darboux integrating factor (`integrating_factor`).
    fn integrating_factor(&self) -> PolyResult<Option<Basic>> {
        let k: &C::K = self.field.k();
        let divergence: PolyElement<C::K> = self.field.divergence();
        let rhs: Vec<<C::K as Domain>::Elem> = self
            .monomials
            .iter()
            .map(|m| k.neg(&divergence.coeff(&[m.0, m.1])))
            .collect();
        if self.cofactors.is_empty() {
            if rhs.iter().all(|v| k.is_zero(v)) {
                return Ok(Some(Basic::one()));
            }
            return Ok(None);
        }
        let Some((particular, basis, _)) = solve_affine(&self.rows, &rhs, self.cofactors.len(), k) else {
            return Ok(None);
        };
        let mut all: Vec<Vec<<C::K as Domain>::Elem>> = vec![particular.clone()];
        all.extend(members(&particular, &basis, k)?);
        let mut candidates: Vec<Basic> = Vec::new();
        for member in all {
            let exponents: Vec<Basic> = member
                .iter()
                .map(|v| self.field.c.to_expr(v))
                .collect::<PolyResult<_>>()?;
            candidates.push(self.product(&exponents, false)?);
        }
        Ok(simplest(candidates))
    }

    /// `prod f_i**c_i exp(sum e_j A_j/B_j)`, or its logarithm
    /// (`_product`).
    fn product(&self, exponents: &[Basic], logarithmic: bool) -> R {
        let field = self.field;
        let n: usize = self.polynomials.len();
        let mut argument: Basic = Basic::zero();
        for ((a, b, _), e) in self.exponentials.iter().zip(&exponents[n..]) {
            if *e != Basic::zero() {
                argument = add(&[argument, div(&mul(&[e.clone(), field.expr(a)?])?, &field.expr(b)?)?])?;
            }
        }
        if logarithmic && argument != Basic::zero() {
            let mut terms: Vec<Basic> = vec![argument];
            for ((f, _), c) in self.polynomials.iter().zip(&exponents[..n]) {
                if *c != Basic::zero() {
                    terms.push(mul(&[
                        c.clone(),
                        crate::functions::elementary::exponential::log(&field.expr(f)?)?,
                    ])?);
                }
            }
            return add(&terms);
        }
        let mut product: Basic = Basic::one();
        for ((f, _), c) in self.polynomials.iter().zip(&exponents[..n]) {
            if *c != Basic::zero() {
                product = mul(&[product, pow(&field.expr(f)?, c)?])?;
            }
        }
        if argument != Basic::zero() {
            product = mul(&[product, crate::functions::elementary::exponential::exp(&argument)?])?;
        }
        Ok(product)
    }
}

// ---------------------------------------------------------------------------
// the quadratures

fn closed_antiderivative(e: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    crate::integrals::indefinite::closed_antiderivative(e, v, false, false, None)
}

/// `Integral(e, v)` with the symbols real first, then complex
/// (`_integrate`).
fn integrate(e: &Basic, v: &Basic, x: &Basic, y: &Basic) -> PolyResult<Option<Basic>> {
    let real_dummy = |s: &Basic| {
        crate::core::symbol::Dummy::new(&name_of(s), &[(crate::core::assumptions::Predicate::real, true)]).0
    };
    let mut real: Vec<(Basic, Basic)> = e
        .free_symbols()
        .into_iter()
        .map(|s| (s.clone(), real_dummy(&s)))
        .collect();
    for s in [x, y] {
        if !real.iter().any(|(k, _)| k == s) {
            real.push((s.clone(), real_dummy(s)));
        }
    }
    let back: Vec<(Basic, Basic)> = real.iter().map(|(s, d)| (d.clone(), s.clone())).collect();
    let variable: Basic = real
        .iter()
        .find(|(s, _)| s == v)
        .map(|(_, d)| d.clone())
        .expect("the variable");
    let integrand: Basic = raw_xreplace(e, &real.iter().cloned().collect())?;
    let budget = || -> Option<f64> {
        match crate::timeout::remaining_time() {
            None => crate::settings::settings().timeout,
            Some(limit) => Some(limit / 2.0),
        }
    };
    if let Some(result) = attempt(budget(), || closed_antiderivative(&integrand, &variable))?.flatten() {
        if let Some(generic) = generic_branch(&result)? {
            let candidate: Basic = raw_xreplace(&generic, &back.iter().cloned().collect())?;
            if is_antiderivative(&candidate, e, v)? {
                return Ok(Some(candidate));
            }
        }
    }
    let Some(result) = attempt(budget(), || closed_antiderivative(e, v))?.flatten() else {
        return Ok(None);
    };
    match generic_branch(&result)? {
        Some(generic) if is_antiderivative(&generic, e, v)? => Ok(Some(generic)),
        _ => Ok(None),
    }
}

/// Whether `f` is an antiderivative of `e` in closed form
/// (`_is_antiderivative`).
fn is_antiderivative(f: &Basic, e: &Basic, v: &Basic) -> PolyResult<bool> {
    if f.preorder().iter().any(|n| matches!(n.head(), Head::Derivative)) {
        return Ok(false);
    }
    is_zero(&sub(&diff(f, v)?, e)?)
}

/// `e` with every `Piecewise` replaced by its generic branch
/// (`_generic_branch`).
fn generic_branch(e: &Basic) -> PolyResult<Option<Basic>> {
    let mut e: Basic = e.clone();
    let mut pieces: Vec<Basic> = e
        .preorder()
        .into_iter()
        .filter(|n| matches!(n.head(), Head::Piecewise))
        .collect();
    pieces.sort_by_key(|p| p.to_string());
    pieces.dedup();
    for piece in pieces {
        let mut chosen: Option<Basic> = None;
        for pair in piece.args() {
            if !matches!(pair.head(), Head::ExprCondPair) {
                continue;
            }
            let condition: &Basic = &pair.args()[1];
            if *condition == crate::core::logic::true_() || matches!(condition.head(), Head::Relational(RelOp::Ne)) {
                chosen = Some(pair.args()[0].clone());
                break;
            }
        }
        let Some(chosen) = chosen else {
            return Ok(None);
        };
        e = raw_xreplace(&e, &[(piece, chosen)].into_iter().collect())?;
    }
    Ok(Some(e))
}

fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}

fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}

/// `cancel`, then `simplify` within a share of the time (`_is_zero`).
fn is_zero(e: &Basic) -> PolyResult<bool> {
    let Some(reduced) = attempt(share_of_time(), || cancel(e))? else {
        return Ok(false);
    };
    if reduced == Basic::zero() {
        return Ok(true);
    }
    Ok(attempt(share_of_time(), || simplify(&reduced))? == Some(Basic::zero()))
}

/// `I` with `I_x = -R P` and `I_y = R Q` (`_first_integral_from_factor`).
fn first_integral_from_factor<C: Coefficients>(field: &VectorField<C>, r: &Basic) -> PolyResult<Option<Basic>> {
    let (x, y) = (&field.x, &field.y);
    let (p, q) = (field.expr(&field.p)?, field.expr(&field.q)?);
    let Some(u) = integrate(&neg(&mul(&[r.clone(), p.clone()])?)?, x, x, y)? else {
        return Ok(None);
    };
    let v: Basic = sub(&mul(&[r.clone(), q.clone()])?, &diff(&u, y)?)?;
    let Some(mut v) = attempt(share_of_time(), || cancel(&v))? else {
        return Ok(None);
    };
    if v.has(x) {
        match attempt(share_of_time(), || simplify(&v))? {
            Some(s) if !s.has(x) => v = s,
            _ => return Ok(None),
        }
    }
    let w: Basic = if v != Basic::zero() {
        match integrate(&v, y, x, y)? {
            Some(w) => w,
            None => return Ok(None),
        }
    } else {
        Basic::zero()
    };
    let integral: Basic = add(&[u, w])?;
    if !is_zero(&add(&[
        mul(&[q, diff(&integral, x)?])?,
        mul(&[p, diff(&integral, y)?])?,
    ])?)? {
        return Ok(None);
    }
    Ok(Some(integral))
}

fn is_constant_integral(integral: &Basic, x: &Basic, y: &Basic) -> bool {
    !(integral.has(x) || integral.has(y))
}

/// A first integral in a readable form (`_tidy`).
fn tidy(integral: &Basic) -> R {
    let Some(cancelled) = attempt(share_of_time(), || cancel(integral))? else {
        return Ok(integral.clone());
    };
    // min by the operations: the first of equal ones
    Ok(if ops(&cancelled) < ops(integral) {
        cancelled
    } else {
        integral.clone()
    })
}

// ---------------------------------------------------------------------------
// the field of the coefficients and the procedure

/// A vector field over a coefficient field whose element type is hidden
/// (`_Context`).
pub trait Context {
    fn polynomials(&self, degree: i64) -> PolyResult<Vec<DarbouxPolynomial>>;
    fn exponential_factors(&self, degree: i64) -> PolyResult<Vec<ExponentialFactor>>;
    fn integrating_factor(&self, degree: i64) -> PolyResult<Option<Basic>>;
    /// The first integral with the Darboux polynomials of the lowest
    /// degree which gives one, over this field (`first_integral_over`).
    fn first_integral_over(&self, degree: i64) -> PolyResult<Option<Basic>>;
    /// The square root of the discriminant of the first irreducible
    /// quadratic factor of `x P_d - y Q_d` over the rationals
    /// (`quadratic_extension`).
    fn quadratic_extension(&self) -> PolyResult<Option<Basic>>;
    /// The same field over `QQ(extension)` (`extended`).
    fn extended(&self, extension: &Basic) -> PolyResult<Box<dyn Context>>;

    /// The Prelle–Singer loop, then over the field of the quadratic
    /// factors (`first_integral`).
    fn first_integral(&self, degree: i64) -> PolyResult<Option<Basic>> {
        if let Some(integral) = self.first_integral_over(degree)? {
            return Ok(Some(integral));
        }
        let Some(extension) = self.quadratic_extension()? else {
            return Ok(None);
        };
        self.extended(&extension)?.first_integral_over(degree)
    }
}

/// The procedure on a field over `K` (`_Engine`).
pub struct Engine<C: Coefficients> {
    field: VectorField<C>,
    p: Basic,
    q: Basic,
}

impl<C: Coefficients> Context for Engine<C> {
    fn polynomials(&self, degree: i64) -> PolyResult<Vec<DarbouxPolynomial>> {
        let field = &self.field;
        darboux(field, degree)?
            .iter()
            .map(|(f, g)| {
                Ok(DarbouxPolynomial {
                    polynomial: field.expr(f)?,
                    cofactor: field.expr(g)?,
                })
            })
            .collect()
    }

    fn exponential_factors(&self, degree: i64) -> PolyResult<Vec<ExponentialFactor>> {
        let field = &self.field;
        exponential(field, &darboux(field, degree)?, degree)
            .iter()
            .map(|(a, b, l)| {
                Ok(ExponentialFactor {
                    numerator: field.expr(a)?,
                    denominator: field.expr(b)?,
                    cofactor: field.expr(l)?,
                })
            })
            .collect()
    }

    fn integrating_factor(&self, degree: i64) -> PolyResult<Option<Basic>> {
        Darboux::new(&self.field, darboux(&self.field, degree)?, degree).integrating_factor()
    }

    fn first_integral_over(&self, degree: i64) -> PolyResult<Option<Basic>> {
        let field = &self.field;
        let mut polynomials: Vec<Pair<C::K>> = Vec::new();
        for n in 1..=degree {
            add_darboux(field, n, &mut polynomials)?;
            let d: Darboux<'_, C> = Darboux::new(field, polynomials.clone(), n);
            if let Some(direct) = d.first_integral()? {
                return Ok(Some(tidy(&direct)?));
            }
            let Some(r) = d.integrating_factor()? else {
                continue;
            };
            if let Some(integral) = first_integral_from_factor(field, &r)? {
                if !is_constant_integral(&integral, &field.x, &field.y) {
                    return Ok(Some(tidy(&integral)?));
                }
            }
        }
        Ok(None)
    }

    fn quadratic_extension(&self) -> PolyResult<Option<Basic>> {
        let field = &self.field;
        if !field.c.is_rationals() {
            return Ok(None);
        }
        let (_, _, h) = top_parts(field);
        if h.terms.is_empty() {
            return Ok(None);
        }
        let k: &C::K = field.k();
        for f in field.c.factor(&h)? {
            if f.degree() != 2 {
                continue;
            }
            // h(x, 1), or h(1, y) when it drops a degree
            let collect = |index: usize| -> BTreeMap<u32, <C::K as Domain>::Elem> {
                let mut in_x: BTreeMap<u32, <C::K as Domain>::Elem> = BTreeMap::new();
                for (m, c) in &f.terms {
                    let entry = in_x.entry(m[index]).or_insert_with(|| k.zero());
                    *entry = k.add(entry, c);
                }
                in_x
            };
            let mut in_x = collect(0);
            if k.is_zero(in_x.get(&2).unwrap_or(&k.zero())) {
                in_x = collect(1);
            }
            let get = |i: u32| in_x.get(&i).cloned().unwrap_or_else(|| k.zero());
            let (a, b, c) = (get(2), get(1), get(0));
            let discriminant = field
                .c
                .rational(&k.sub(&k.mul(&b, &b), &k.mul(&k.from_int(4), &k.mul(&a, &c))));
            match discriminant {
                Some(d) if d < Rat::from(0) => {
                    return Ok(Some(pow(&Basic::from_rational(d), &Basic::half())?));
                }
                // reducible, or a double line: nothing new
                _ => continue,
            }
        }
        Ok(None)
    }

    fn extended(&self, extension: &Basic) -> PolyResult<Box<dyn Context>> {
        field_of(&self.p, &self.q, &self.field.x, &self.field.y, Some(extension))
    }
}

/// The number field of an algebraic number (`_number_field`).
fn number_field(alpha: &Basic) -> PolyResult<AlgebraicField> {
    let coefficients: Vec<Int> = crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(alpha)?;
    let lead: Int = coefficients[0].clone();
    AlgebraicField::new(
        coefficients
            .iter()
            .map(|c| crate::numbers::rat_from_parts(c.clone(), lead.clone()))
            .collect(),
        alpha.clone(),
    )
}

/// `f(t)` in `K` (`_composed`).
fn composed(f: &[Rat], t: &AlgebraicElement, k: &AlgebraicField) -> AlgebraicElement {
    let mut value: AlgebraicElement = k.zero();
    for c in f {
        value = k.add(&k.mul(&value, t), &k.from_rational(c));
    }
    value
}

/// The number field of several algebraic numbers by a primitive element
/// (`_primitive_field`).
fn primitive_field(generators: &[Basic]) -> PolyResult<NumberField> {
    let mut theta: Basic = generators[0].clone();
    let mut k: AlgebraicField = number_field(&theta)?;
    let mut images: Vec<(Basic, AlgebraicElement)> = vec![(theta.clone(), k.alpha())];
    for g in &generators[1..] {
        let big_g: AlgebraicField = number_field(g)?;
        if big_g.degree == 1 {
            images.push((g.clone(), k.from_rational(&(-big_g.minpoly[1].clone()))));
            continue;
        }
        let mut found = false;
        for c in 1..(4 * k.degree * big_g.degree + 2) as i64 {
            let candidate: Basic = add(&[theta.clone(), mul(&[int(c), g.clone()])?])?;
            let l: AlgebraicField = number_field(&candidate)?;
            // m(theta_L - c t) as a polynomial in t over L, by Horner
            let mut shifted: Vec<AlgebraicElement> = Vec::new();
            for coefficient in &k.minpoly {
                let mut product: Vec<AlgebraicElement> = if shifted.is_empty() {
                    Vec::new()
                } else {
                    vec![l.zero(); shifted.len() + 1]
                };
                for (i, u) in shifted.iter().enumerate() {
                    product[i] = l.add(&product[i], &l.mul(u, &l.from_int(-c)));
                    product[i + 1] = l.add(&product[i + 1], &l.mul(u, &l.alpha()));
                }
                if product.is_empty() {
                    product = if *coefficient != Rat::from(0) {
                        vec![l.from_rational(coefficient)]
                    } else {
                        Vec::new()
                    };
                } else {
                    let last: usize = product.len() - 1;
                    product[last] = l.add(&product[last], &l.from_rational(coefficient));
                }
                shifted = product;
            }
            let target: Vec<AlgebraicElement> = big_g.minpoly.iter().map(|v| l.from_rational(v)).collect();
            let common: Vec<AlgebraicElement> = crate::polys::euclidtools::dup_gcd(&shifted, &target, &l);
            if common.len() != 2 {
                continue;
            }
            let image: AlgebraicElement = l.neg(&l.quo(&common[1], &common[0])?);
            let previous: AlgebraicElement = l.sub(&l.alpha(), &l.mul(&l.from_int(c), &image));
            images = images
                .into_iter()
                .map(|(h, v)| (h, composed(&v, &previous, &l)))
                .collect();
            images.push((g.clone(), image));
            theta = candidate;
            k = l;
            found = true;
            break;
        }
        if !found {
            return Err(PolyError::Arithmetic("no primitive element was found".to_string()));
        }
    }
    Ok(NumberField::new(k, Some(images)))
}

/// The parameters, the radicals of rational bases and `I` among the
/// coefficients (`_coefficient_atoms`).
fn coefficient_atoms(
    e: &Basic,
    x: &Basic,
    y: &Basic,
    parameters: &mut Vec<Basic>,
    radicals: &mut Vec<(Rat, Int)>,
    imaginary: &mut bool,
) {
    if e.as_rational().is_some() || e == x || e == y {
        return;
    }
    if matches!(e.head(), Head::ImaginaryUnit) {
        *imaginary = true;
        return;
    }
    if e.is_symbol() {
        if !parameters.contains(e) {
            parameters.push(e.clone());
        }
        return;
    }
    if e.is_pow() {
        if let (Some(base), Some(exponent)) = (e.args()[0].as_rational(), e.args()[1].as_rational()) {
            let q: Int = crate::numbers::denom(exponent);
            match radicals.iter_mut().find(|(b, _)| b == base) {
                Some(entry) => entry.1 = crate::numbers::lcm(&entry.1, &q),
                None => radicals.push((base.clone(), q)),
            }
            return;
        }
        if is_integer_node(&e.args()[1]) {
            coefficient_atoms(&e.args()[0], x, y, parameters, radicals, imaginary);
            return;
        }
    }
    if e.is_add() || e.is_mul() {
        for a in e.args() {
            coefficient_atoms(a, x, y, parameters, radicals, imaginary);
        }
        return;
    }
    if !(e.has(x) || e.has(y)) && !parameters.contains(e) {
        parameters.push(e.clone());
    }
}

fn engine<C: Coefficients + 'static>(c: C, p: &Basic, q: &Basic, x: &Basic, y: &Basic) -> PolyResult<Box<dyn Context>> {
    let ring: Arc<PolyRing<C::K>> = PolyRing::new(
        c.domain().clone(),
        vec!["_darboux_x".to_string(), "_darboux_y".to_string()],
        MonomialOrder::Lex,
    )?;
    let pp: PolyElement<C::K> = xy_poly(p, x, y, &c, &ring)?;
    let qq: PolyElement<C::K> = xy_poly(q, x, y, &c, &ring)?;
    if qq.terms.is_empty() {
        return Err(value_error("Q must not be zero".to_string()));
    }
    let degree: i64 = pp.degree().max(qq.degree());
    Ok(Box::new(Engine {
        field: VectorField {
            c,
            ring,
            p: pp,
            q: qq,
            x: x.clone(),
            y: y.clone(),
            degree,
        },
        p: p.clone(),
        q: q.clone(),
    }))
}

/// The field `Q d/dx + P d/dy` over the field of its coefficients
/// (`_field_of`).
pub fn field_of(p: &Basic, q: &Basic, x: &Basic, y: &Basic, extension: Option<&Basic>) -> PolyResult<Box<dyn Context>> {
    let error: String = format!("polynomials in {} and {} over a field are expected", x, y);
    let mut parameters: Vec<Basic> = Vec::new();
    let mut radicals: Vec<(Rat, Int)> = Vec::new();
    let mut imaginary: bool = false;
    for e in [p, q] {
        coefficient_atoms(&expand(e)?, x, y, &mut parameters, &mut radicals, &mut imaginary);
    }
    let mut generators: Vec<Basic> = extension.into_iter().cloned().collect();
    if imaginary {
        generators.push(Basic::atom(Head::ImaginaryUnit));
    }
    radicals.sort_by(|a, b| {
        (crate::numbers::numer(&a.0), crate::numbers::denom(&a.0))
            .cmp(&(crate::numbers::numer(&b.0), crate::numbers::denom(&b.0)))
    });
    for (base, q_root) in &radicals {
        if *q_root > Int::from(1) {
            generators.push(pow(
                &Basic::from_rational(base.clone()),
                &Basic::rational(Int::from(1), q_root.clone())?,
            )?);
        }
    }
    let built = || -> PolyResult<Box<dyn Context>> {
        if !generators.is_empty() {
            if !parameters.is_empty() {
                return Err(value_error(
                    "coefficients with parameters and algebraic numbers are not supported".to_string(),
                ));
            }
            let c: NumberField = if generators.len() == 1 {
                NumberField::new(number_field(&generators[0])?, None)
            } else {
                primitive_field(&generators)?
            };
            return engine(c, p, q, x, y);
        }
        if !parameters.is_empty() {
            let mut sorted: Vec<Basic> = parameters.clone();
            sorted.sort_by_key(crate::core::sorting::sort_key);
            return engine(Parameters::new(&sorted), p, q, x, y);
        }
        engine(Rationals, p, q, x, y)
    };
    match built() {
        Ok(context) => Ok(context),
        Err(e) if e.to_string() == "Q must not be zero" => Err(e),
        Err(PolyError::Value(_) | PolyError::Type(_) | PolyError::Arithmetic(_)) => Err(value_error(error)),
        Err(e) if crate::solvers::kovacic::failure(&e) => Err(value_error(error)),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// the public functions

/// The irreducible Darboux polynomials of degree at most `degree`
/// (`darboux_polynomials`).
pub fn darboux_polynomials(
    p: &Basic,
    q: &Basic,
    x: &Basic,
    y: &Basic,
    degree: i64,
    extension: Option<&Basic>,
) -> PolyResult<Vec<DarbouxPolynomial>> {
    field_of(p, q, x, y, extension)?.polynomials(degree)
}

/// The exponential factors (`exponential_factors`).
pub fn exponential_factors(
    p: &Basic,
    q: &Basic,
    x: &Basic,
    y: &Basic,
    degree: i64,
    extension: Option<&Basic>,
) -> PolyResult<Vec<ExponentialFactor>> {
    field_of(p, q, x, y, extension)?.exponential_factors(degree)
}

/// A Darboux integrating factor (`darboux_integrating_factor`).
pub fn darboux_integrating_factor(
    p: &Basic,
    q: &Basic,
    x: &Basic,
    y: &Basic,
    degree: i64,
    extension: Option<&Basic>,
) -> PolyResult<Option<Basic>> {
    field_of(p, q, x, y, extension)?.integrating_factor(degree)
}

/// A first integral by the Prelle–Singer procedure
/// (`darboux_first_integral`).
pub fn darboux_first_integral(
    p: &Basic,
    q: &Basic,
    x: &Basic,
    y: &Basic,
    degree: i64,
    extension: Option<&Basic>,
) -> PolyResult<Option<Basic>> {
    field_of(p, q, x, y, extension)?.first_integral(degree)
}

/// The implicit (or explicit) solution of `y' = P/Q` by the Prelle–Singer
/// procedure (`prelle_singer`).
pub fn prelle_singer(equation: &Basic, f: &Basic, degree: i64, extension: Option<&Basic>) -> PolyResult<Option<Basic>> {
    let Some((p, q, x, y)) = vector_field(equation, f)? else {
        return Ok(None);
    };
    let context: Box<dyn Context> = match field_of(&p, &q, &x, &y, extension) {
        Ok(c) => c,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let timeout: Option<f64> = crate::settings::settings().timeout;
    let Some(integral) = attempt(timeout, || context.first_integral(degree))?.flatten() else {
        return Ok(None);
    };
    let c1: Basic = symbol("C1");
    let implicit: Basic = eq(&xreplace(&integral, &[(y.clone(), f.clone())])?, &c1);
    let Some(explicit) = explicit(&integral, &y, &c1)? else {
        return Ok(Some(implicit));
    };
    Ok(Some(eq(f, &xreplace(&explicit, &[(y, f.clone())])?)))
}

/// `y` from `I(x, y) = C1` when it has one root (`_explicit`).
fn explicit(integral: &Basic, y: &Basic, c1: &Basic) -> PolyResult<Option<Basic>> {
    if crate::core::expr::polynomial_like(
        integral,
        std::slice::from_ref(y),
        crate::core::expr::PolynomialKind::Rational,
    ) != Some(true)
    {
        return Ok(None);
    }
    let relation: Basic = sub(integral, c1)?;
    let Some(found) = attempt(share_of_time(), || {
        crate::polys::rationalsolve::solveset_rational(&relation, y)
    })?
    .flatten() else {
        return Ok(None);
    };
    let (solutions, excluded) = found;
    if solutions.len() != 1 {
        return Ok(None);
    }
    let solution: Basic = solutions[0].clone();
    if solution.has(y) || excluded.contains(&solution) {
        return Ok(None);
    }
    Ok(Some(
        attempt(share_of_time(), || simplify(&solution))?.unwrap_or(solution),
    ))
}
