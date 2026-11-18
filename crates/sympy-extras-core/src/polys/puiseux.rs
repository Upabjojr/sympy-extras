//! Puiseux expansions of the branches of a plane curve, ported from
//! `sympy_extras.polys.puiseux` (Newton–Puiseux with Duval's rational
//! expansions over number fields). This part: the truncated Laurent
//! series, the embeddings of number fields, the fields built from a
//! minimal polynomial (the generator written as the largest real root),
//! the factorisation over a number field and the adjunction of a root by
//! a primitive element (exactly: a resultant and a gcd, no root evaluated
//! numerically).
//!
//! A number is an element of an `AlgebraicField`, its coordinates in the
//! powers of the generator (highest degree first, `[]` zero); the
//! rationals are the field [`rationals`] of degree one.

// References: R. J. Walker, Algebraic curves, Princeton 1950, chapter IV; D. Duval, Rational
// Puiseux expansions, Compositio Mathematica 70 (1989); B. M. Trager, Algebraic factoring and
// rational function integration, SYMSAC 1976.

use std::collections::BTreeMap;

use crate::core::basic::Basic;
use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::densearith::{dup_add, dup_mul, dup_pow};
use crate::polys::densebasic::{dup_degree, dup_strip};
use crate::polys::densetools::{dup_diff, dup_monic};
use crate::polys::domains::{AlgebraicField, Domain, PolynomialRing, QQ};
use crate::polys::euclidtools::{dup_gcd, dup_resultant};

/// An element of a number field: its coordinates in the powers of the
/// generator, highest degree first (`[]` is zero).
pub type Number = Vec<Rat>;

/// The rationals as the number field of degree one (`RATIONALS`): the
/// elements are `[]` and `[q]`, the rationals of every number field.
pub fn rationals() -> AlgebraicField {
    AlgebraicField::new(vec![Rat::ONE, Rat::ZERO], Basic::zero()).expect("a monic modulus")
}

fn min_order(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(a), Some(b)) => Some(a.min(b)),
    }
}

/// A truncated Laurent series `sum c_k t^k` with coefficients in the
/// number field `k`, the coefficients known for `k < order` (`None`: an
/// exact Laurent polynomial) (`LaurentSeries`).
#[derive(Debug, Clone, PartialEq)]
pub struct LaurentSeries {
    pub field: AlgebraicField,
    pub order: Option<i64>,
    pub terms: BTreeMap<i64, Number>,
}

impl LaurentSeries {
    pub fn new(field: AlgebraicField, terms: BTreeMap<i64, Number>, order: Option<i64>) -> LaurentSeries {
        let terms: BTreeMap<i64, Number> = terms
            .into_iter()
            .filter(|(k, c)| !c.is_empty() && order.map_or(true, |o| *k < o))
            .collect();
        LaurentSeries { field, order, terms }
    }

    pub fn constant(field: &AlgebraicField, c: Number) -> LaurentSeries {
        LaurentSeries::new(field.clone(), BTreeMap::from([(0, c)]), None)
    }

    pub fn monomial(field: &AlgebraicField, c: Number, k: i64) -> LaurentSeries {
        LaurentSeries::new(field.clone(), BTreeMap::from([(k, c)]), None)
    }

    /// The least exponent with a nonzero coefficient, `None` when no term
    /// is known.
    pub fn valuation(&self) -> Option<i64> {
        self.terms.keys().next().copied()
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn coefficient(&self, k: i64) -> Number {
        self.terms.get(&k).cloned().unwrap_or_default()
    }

    pub fn add(&self, other: &LaurentSeries) -> LaurentSeries {
        let k = &self.field;
        let mut terms: BTreeMap<i64, Number> = self.terms.clone();
        for (j, c) in &other.terms {
            let sum: Number = k.add(&terms.get(j).cloned().unwrap_or_default(), c);
            terms.insert(*j, sum);
        }
        LaurentSeries::new(k.clone(), terms, min_order(self.order, other.order))
    }

    pub fn neg(&self) -> LaurentSeries {
        let terms = self.terms.iter().map(|(j, c)| (*j, self.field.neg(c))).collect();
        LaurentSeries::new(self.field.clone(), terms, self.order)
    }

    pub fn sub(&self, other: &LaurentSeries) -> LaurentSeries {
        self.add(&other.neg())
    }

    pub fn scaled(&self, c: &Number) -> LaurentSeries {
        let terms = self.terms.iter().map(|(j, v)| (*j, self.field.mul(c, v))).collect();
        LaurentSeries::new(self.field.clone(), terms, self.order)
    }

    /// The series times `t^k`.
    pub fn shifted(&self, k: i64) -> LaurentSeries {
        let terms = self.terms.iter().map(|(j, c)| (j + k, c.clone())).collect();
        LaurentSeries::new(self.field.clone(), terms, self.order.map(|o| o + k))
    }

    pub fn mul(&self, other: &LaurentSeries) -> LaurentSeries {
        let k = &self.field;
        let (v1, v2) = (self.valuation(), other.valuation());
        if (self.order.is_none() && self.terms.is_empty()) || (other.order.is_none() && other.terms.is_empty()) {
            // an exact zero: the product is exactly zero
            return LaurentSeries::new(k.clone(), BTreeMap::new(), None);
        }
        let order: Option<i64> = if self.order.is_none() && other.order.is_none() {
            None
        } else {
            let mut candidates: Vec<i64> = Vec::new();
            if let Some(o) = other.order {
                // the unknown part of other times the known part of self
                candidates.push(o + v1.unwrap_or(self.order.unwrap_or(0)));
            }
            if let Some(o) = self.order {
                candidates.push(o + v2.unwrap_or(other.order.unwrap_or(0)));
            }
            candidates.into_iter().min()
        };
        let mut terms: BTreeMap<i64, Number> = BTreeMap::new();
        for (j, c) in &self.terms {
            for (l, d) in &other.terms {
                if order.is_some_and(|o| j + l >= o) {
                    continue;
                }
                let sum: Number = k.add(&terms.get(&(j + l)).cloned().unwrap_or_default(), &k.mul(c, d));
                terms.insert(j + l, sum);
            }
        }
        LaurentSeries::new(k.clone(), terms, order)
    }

    pub fn truncated(&self, order: i64) -> LaurentSeries {
        let known: i64 = self.order.map_or(order, |o| o.min(order));
        LaurentSeries::new(self.field.clone(), self.terms.clone(), Some(known))
    }

    pub fn power(&self, n: u64) -> LaurentSeries {
        let mut result: LaurentSeries = LaurentSeries::constant(&self.field, self.field.one());
        let mut base: LaurentSeries = self.clone();
        let mut n: u64 = n;
        while n > 0 {
            if n % 2 == 1 {
                result = result.mul(&base);
            }
            n /= 2;
            if n > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// The inverse series, known `order` terms beyond its valuation.
    pub fn inverse(&self, order: i64) -> PolyResult<LaurentSeries> {
        let Some(v) = self.valuation() else {
            return Err(zero_division("inverse of a series which is zero to its order"));
        };
        let k = &self.field;
        let unit: LaurentSeries = self.shifted(-v);
        let a: BTreeMap<i64, Number> = unit
            .terms
            .iter()
            .filter(|(j, _)| **j < order)
            .map(|(j, c)| (*j, c.clone()))
            .collect();
        let inv0: Number = k.invert(&unit.coefficient(0))?;
        let mut b: BTreeMap<i64, Number> = BTreeMap::from([(0, inv0.clone())]);
        for n in 1..order {
            let mut total: Number = k.zero();
            for (j, a_j) in &a {
                if 0 < *j && *j <= n {
                    if let Some(b_prev) = b.get(&(n - j)) {
                        total = k.add(&total, &k.mul(a_j, b_prev));
                    }
                }
            }
            if !total.is_empty() {
                b.insert(n, k.neg(&k.mul(&inv0, &total)));
            }
        }
        let known: i64 = unit.order.map_or(order, |o| o.min(order));
        Ok(LaurentSeries::new(k.clone(), b, Some(known)).shifted(-v))
    }

    /// The series with `t` replaced by `t^q`.
    pub fn substituted(&self, q: i64) -> LaurentSeries {
        let terms = self.terms.iter().map(|(j, c)| (j * q, c.clone())).collect();
        LaurentSeries::new(self.field.clone(), terms, self.order.map(|o| o * q))
    }

    pub fn converted(&self, embedding: &Embedding) -> LaurentSeries {
        let terms = self.terms.iter().map(|(j, c)| (*j, embedding.apply(c))).collect();
        LaurentSeries::new(embedding.target.clone(), terms, self.order)
    }

    /// `sum c_k t^(k/exponent_scale)` as an expression (`t^(-k/...)` with
    /// `reciprocal`) (`as_expr`).
    pub fn as_expr(&self, t: &Basic, exponent_scale: i64, reciprocal: bool) -> PolyResult<Basic> {
        let sign: i64 = if reciprocal { -1 } else { 1 };
        let mut parts: Vec<Basic> = Vec::new();
        for (j, c) in &self.terms {
            let exponent: Basic = Basic::rational(Int::from(sign * j), Int::from(exponent_scale))?;
            let power: Basic = crate::core::power::pow(t, &exponent)?;
            parts.push(crate::core::mul::mul(&[self.field.to_expr(c)?, power])?);
        }
        crate::core::add::add(&parts)
    }
}

/// The embedding of a number field `source` into a number field `target`
/// containing it, by the image of the generator of the source (none for
/// the rationals, nor when the fields are the same) (`Embedding`).
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub source: AlgebraicField,
    pub target: AlgebraicField,
    pub image: Option<Number>,
}

impl Embedding {
    pub fn new(source: AlgebraicField, target: AlgebraicField, image: Option<Number>) -> PolyResult<Embedding> {
        let needed: bool = source.degree > 1 && source != target;
        if needed && image.is_none() {
            return Err(value_error("the image of the generator is needed".to_string()));
        }
        Ok(Embedding {
            source,
            target,
            image: if needed { image } else { None },
        })
    }

    /// The image of `c` (`__call__`).
    pub fn apply(&self, c: &Number) -> Number {
        if self.source == self.target {
            return c.clone();
        }
        let target = &self.target;
        match &self.image {
            // the rationals into any field
            None => c.first().map_or_else(|| target.zero(), |q| target.from_rational(q)),
            Some(image) => {
                // c(image) by Horner's rule
                let mut total: Number = target.zero();
                for a in c {
                    total = target.add(&target.mul(&total, image), &target.from_rational(a));
                }
                total
            }
        }
    }

    /// The composition with `other`, an embedding of `target` (`then`).
    pub fn then(&self, other: &Embedding) -> PolyResult<Embedding> {
        if other.source != self.target {
            return Err(value_error("the embeddings do not compose".to_string()));
        }
        if self.source == self.target {
            return Ok(other.clone());
        }
        if other.source == other.target {
            return Ok(self.clone());
        }
        match &self.image {
            None => Embedding::new(self.source.clone(), other.target.clone(), None),
            Some(image) => Embedding::new(self.source.clone(), other.target.clone(), Some(other.apply(image))),
        }
    }
}

/// The generator of the number field (one for the rationals).
pub fn generator(k: &AlgebraicField) -> Number {
    if k.degree > 1 {
        k.alpha()
    } else {
        k.one()
    }
}

/// The degree of the number field over the rationals.
pub fn degree(k: &AlgebraicField) -> usize {
    k.degree
}

/// The rational coordinates of `c`, low degree first, padded to the
/// degree (`_coefficient_list`).
fn coefficient_list(c: &Number, k: &AlgebraicField) -> Vec<Rat> {
    let mut coefficients: Vec<Rat> = c.iter().rev().cloned().collect();
    while coefficients.len() < k.degree {
        coefficients.push(Rat::ZERO);
    }
    coefficients
}

fn monic(f: &[Rat]) -> PolyResult<Vec<Rat>> {
    dup_monic(&dup_strip(f, &QQ), &QQ)
}

/// The largest real root of the monic irreducible `n` (the first root in
/// SymPy's order when it has none), in radicals for a linear, quadratic
/// or binomial polynomial; `theta` when not real and not in radicals
/// (`_generator_expression`).
fn generator_expression(n: &[Rat]) -> PolyResult<Basic> {
    use crate::polys::cad::indexedroot::{bound_variable, node};
    let mut denominator: Int = Int::ONE;
    for c in n {
        denominator = crate::core::intfunc::ilcm(&[denominator, Int::from(c.denominator().clone())]);
    }
    let t: Basic = bound_variable("T");
    let d: usize = n.len() - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in n.iter().enumerate() {
        let integer: Int = (c * Rat::from(denominator.clone())).numerator().clone();
        if !integer.is_zero() {
            let power: Basic = crate::core::power::pow(&t, &Basic::integer(Int::from(d - i)))?;
            terms.push(crate::core::mul::mul(&[Basic::integer(integer), power])?);
        }
    }
    let expr: Basic = crate::core::add::add(&terms)?;
    let poly = crate::polys::polytools::Poly::new(&expr, std::slice::from_ref(&t))?;
    let real: i64 = crate::polys::rootisolation::dup_count_real_roots(n, None, None);
    let index: usize = if real > 0 { (real - 1) as usize } else { 0 };
    if let Some(trivial) = crate::polys::roots::trivial_roots(&poly)? {
        return Ok(trivial[index].clone());
    }
    if real > 0 {
        return node(&poly.as_expr()?, &t, index);
    }
    Ok(crate::core::Symbol::new("theta", &[]).0)
}

/// `QQ(Theta)` for `Theta` a root of the irreducible `n` (rational
/// coefficients, highest degree first), the generator written as the
/// largest real root; the rationals for a linear `n`
/// (`field_from_polynomial`).
pub fn field_from_polynomial(n: &[Rat]) -> PolyResult<AlgebraicField> {
    let monic: Vec<Rat> = monic(n)?;
    if monic.len() <= 2 {
        return Ok(rationals());
    }
    let alpha: Basic = generator_expression(&monic)?;
    AlgebraicField::new(monic, alpha)
}

/// The irreducible factors over `k` of the polynomial with the given
/// coefficients (highest first), monic, with multiplicities
/// (`factor_over`).
pub fn factor_over(k: &AlgebraicField, coefficients: &[Number]) -> PolyResult<Vec<(Vec<Number>, usize)>> {
    let f: Vec<Number> = dup_strip(coefficients, k);
    if dup_degree(&f) < 1 {
        return Ok(Vec::new());
    }
    if k.degree == 1 {
        let rational: Vec<Rat> = f.iter().map(|c| c.first().cloned().unwrap_or(Rat::ZERO)).collect();
        let (_, factors) = QQ.factor_dense(&rational)?;
        return Ok(factors
            .into_iter()
            .map(|(factor, m)| (factor.iter().map(|c| k.from_rational(c)).collect(), m))
            .collect());
    }
    Ok(k.factor_dense(&f)?.1)
}

/// `k^d psi((T - theta)/k; theta)` (`psi(T; theta)` for `k = 0`) as a
/// polynomial in `theta` over `QQ[T]`, highest degree first
/// (`_shifted_psi`).
fn shifted_psi(
    psi: &[Number],
    k: i64,
    d: usize,
    qt: &PolynomialRing<crate::polys::domains::RationalField>,
) -> Vec<Vec<Rat>> {
    let mut total: Vec<Vec<Rat>> = Vec::new();
    for (i, c) in psi.iter().rev().enumerate() {
        if c.is_empty() {
            continue;
        }
        let lifted: Vec<Vec<Rat>> = c
            .iter()
            .map(|a| if a.is_zero() { Vec::new() } else { vec![a.clone()] })
            .collect();
        let coefficient: Vec<Vec<Rat>> = dup_strip(&lifted, qt);
        let term: Vec<Vec<Rat>> = if k == 0 {
            // times T^i
            let mut monomial: Vec<Rat> = vec![Rat::ONE];
            monomial.extend(std::iter::repeat(Rat::ZERO).take(i));
            coefficient.iter().map(|a| qt.mul(a, &monomial)).collect()
        } else {
            // times k^(d - i) (T - theta)^i, T - theta = -theta + T
            let linear: Vec<Vec<Rat>> = vec![vec![Rat::from(-1)], vec![Rat::ONE, Rat::ZERO]];
            let power: Vec<Vec<Rat>> = dup_pow(&linear, i, qt);
            let scale: Vec<Rat> = vec![Rat::from(Int::from(k).pow(d - i))];
            dup_mul(&coefficient, &power, qt)
                .iter()
                .map(|a| qt.mul(a, &scale))
                .collect()
        };
        total = dup_add(&total, &dup_strip(&term, qt), qt);
    }
    total
}

/// The polynomial `c` with rational coefficients at `value`
/// (`_evaluate_rational`).
fn evaluate_rational(c: &[Rat], value: &Number, l: &AlgebraicField) -> Number {
    let mut total: Number = l.zero();
    for a in c {
        total = l.add(&l.mul(&total, value), &l.from_rational(a));
    }
    total
}

/// The field `L = K(u)` for a root `u` of `psi` (coefficients in `k`,
/// highest degree first) irreducible over `k`, the embedding of `k` into
/// `L` and `u` (`adjoin_root`).
pub fn adjoin_root(k: &AlgebraicField, psi: &[Number]) -> PolyResult<(AlgebraicField, Embedding, Number)> {
    let d: usize = psi.len() - 1;
    if d == 1 {
        let u: Number = k.neg(&k.quo(&psi[1], &psi[0])?);
        return Ok((k.clone(), Embedding::new(k.clone(), k.clone(), None)?, u));
    }
    if k.degree == 1 {
        let rational: Vec<Rat> = psi.iter().map(|c| c.first().cloned().unwrap_or(Rat::ZERO)).collect();
        let l: AlgebraicField = field_from_polynomial(&rational)?;
        let u: Number = generator(&l);
        return Ok((l.clone(), Embedding::new(k.clone(), l, None)?, u));
    }
    let qt = PolynomialRing::new(QQ, "T");
    let m_k: Vec<Vec<Rat>> = k
        .minpoly
        .iter()
        .map(|c| if c.is_zero() { Vec::new() } else { vec![c.clone()] })
        .collect();
    let bound: i64 = (2 * d * k.degree + 3) as i64;
    let mut found: Option<(i64, Vec<Rat>, Vec<Vec<Rat>>)> = None;
    for shift in 0..bound {
        // Theta = theta + k u: P(T, theta) = k^d psi((T - theta)/k)
        let p: Vec<Vec<Rat>> = shifted_psi(psi, shift, d, &qt);
        let n: Vec<Rat> = dup_resultant(&p, &m_k, &qt);
        if dup_degree(&n) == (d * k.degree) as isize && dup_degree(&dup_gcd(&n, &dup_diff(&n, 1, &QQ), &QQ)) == 0 {
            found = Some((shift, n, p));
            break;
        }
    }
    let Some((shift, n, p)) = found else {
        return Err(value_error("no primitive element found".to_string()));
    };
    let l: AlgebraicField = field_from_polynomial(&n)?;
    let theta: Number = generator(&l);
    // the image t of theta: the common root of m_K(t) and P(Theta, t)
    let m_k_l: Vec<Number> = k.minpoly.iter().map(|c| l.from_rational(c)).collect();
    let p_l: Vec<Number> = dup_strip(
        &p.iter().map(|c| evaluate_rational(c, &theta, &l)).collect::<Vec<_>>(),
        &l,
    );
    let common: Vec<Number> = dup_gcd(&m_k_l, &p_l, &l);
    if dup_degree(&common) != 1 {
        return Err(value_error("the image of the generator is not unique".to_string()));
    }
    let image: Number = l.neg(&l.quo(&common[1], &common[0])?);
    let u: Number = if shift == 0 {
        theta
    } else {
        l.quo(&l.sub(&theta, &image), &l.from_int(shift))?
    };
    let into: Embedding = Embedding::new(k.clone(), l.clone(), Some(image))?;
    let mut value: Number = l.zero();
    for c in psi {
        value = l.add(&l.mul(&value, &u), &into.apply(c));
    }
    if !value.is_empty() {
        return Err(value_error(
            "the adjoined root does not satisfy its polynomial".to_string(),
        ));
    }
    Ok((l, into, u))
}

/// The number field generated by an algebraic number given as an
/// expression (radicals, `I`, a real `IndexedRoot`) and the number as its
/// generator (`field_of`).
pub fn field_of(point: &Basic) -> PolyResult<(AlgebraicField, Number)> {
    use crate::core::basic::Head;
    if let (Head::Rational { .. }, Some(v)) = (point.head(), point.as_rational()) {
        let q = rationals();
        let element: Number = q.from_rational(v);
        return Ok((q, element));
    }
    let integers: Vec<Int> = if matches!(point.head(), Head::IndexedRoot) {
        match crate::polys::cad::indexedroot::algebraic_of(point)? {
            Some(number) => number.minpoly(),
            None => return Err(value_error(format!("{} is not an algebraic number", point))),
        }
    } else {
        crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(point)?
    };
    let minpoly: Vec<Rat> = monic(&integers.into_iter().map(Rat::from).collect::<Vec<_>>())?;
    if minpoly.len() <= 2 {
        let q = rationals();
        let element: Number = q.from_rational(&-minpoly[1].clone());
        return Ok((q, element));
    }
    let k: AlgebraicField = AlgebraicField::new(minpoly, point.clone())?;
    let alpha: Number = k.alpha();
    Ok((k, alpha))
}

/// The minimal polynomial over the rationals of an element of `k`, monic,
/// highest degree first (`minimal_polynomial_of`).
pub fn minimal_polynomial_of(c: &Number, k: &AlgebraicField) -> Vec<Rat> {
    if k.degree == 1 {
        return vec![Rat::ONE, -c.first().cloned().unwrap_or(Rat::ZERO)];
    }
    k.minimal_polynomial(c)
}

/// `c` of `l` as an element of the subfield `into.source`, `None` when it
/// is not in it (`in_subfield`).
pub fn in_subfield(c: &Number, l: &AlgebraicField, into: &Embedding) -> Option<Number> {
    let k: &AlgebraicField = &into.source;
    if k == l {
        return Some(c.clone());
    }
    if k.degree == 1 {
        let coefficients: Vec<Rat> = coefficient_list(c, l);
        if coefficients[1..].iter().any(|q| !q.is_zero()) {
            return None;
        }
        return Some(k.from_rational(&coefficients[0]));
    }
    let d: usize = k.degree;
    let mut columns: Vec<Vec<Rat>> = Vec::new();
    let mut power: Number = l.one();
    let image: Number = into.apply(&generator(k));
    for _ in 0..d {
        columns.push(coefficient_list(&power, l));
        power = l.mul(&power, &image);
    }
    let target: Vec<Rat> = coefficient_list(c, l);
    let mut rows: Vec<BTreeMap<usize, Rat>> = Vec::new();
    for (i, t) in target.iter().enumerate() {
        let mut row: BTreeMap<usize, Rat> = (0..d)
            .filter(|j| !columns[*j][i].is_zero())
            .map(|j| (j, columns[j][i].clone()))
            .collect();
        if !t.is_zero() {
            row.insert(d, t.clone());
        }
        rows.push(row);
    }
    let nonempty: Vec<BTreeMap<usize, Rat>> = rows.into_iter().filter(|r| !r.is_empty()).collect();
    let echelon = QQ.sparse_echelon(&nonempty);
    let mut solution: Vec<Rat> = vec![Rat::ZERO; d];
    for row in &echelon {
        let pivot: usize = *row.keys().next().expect("a nonempty row");
        if pivot == d {
            return None;
        }
        solution[pivot] = row.get(&d).cloned().unwrap_or(Rat::ZERO);
    }
    solution.reverse();
    Some(dup_strip(&solution, &QQ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Number {
        vec![Rat::from(n)]
    }

    #[test]
    fn laurent_series() {
        let k = rationals();
        // the docstring: 3*t**2 + 1/t, and in t**(1/2)
        let s = LaurentSeries::new(k.clone(), BTreeMap::from([(-1, q(1)), (2, q(3))]), Some(4));
        let t = crate::core::Symbol::new("t", &[]).0;
        assert_eq!(s.as_expr(&t, 1, false).unwrap().to_string(), "3*t**2 + 1/t");
        assert_eq!(s.as_expr(&t, 2, false).unwrap().to_string(), "3*t + 1/sqrt(t)");
        // (1 - t) * (1 + t + t**2 + ...) = 1 to the order
        let one_minus_t = LaurentSeries::new(k.clone(), BTreeMap::from([(0, q(1)), (1, q(-1))]), None);
        let inverse = one_minus_t.inverse(5).unwrap();
        assert_eq!(inverse.order, Some(5));
        assert!((0..5).all(|j| inverse.coefficient(j) == q(1)));
        let product = one_minus_t.mul(&inverse);
        assert_eq!(product.terms, BTreeMap::from([(0, q(1))]));
        assert_eq!(product.order, Some(5));
        assert_eq!(one_minus_t.power(2).coefficient(1), q(-2));
        assert!(LaurentSeries::new(k, BTreeMap::new(), Some(3)).inverse(2).is_err());
    }

    #[test]
    fn rationals_field() {
        assert_eq!(rationals().name(), "QQ");
        assert_eq!(generator(&rationals()), q(1));
    }
}
