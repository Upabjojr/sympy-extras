//! Port of `sympy_extras/integrals/trager.py`: Trager's algorithm for the
//! integrands rational in `x` and in `y = sqrt(P(x))` (the Hermite
//! reduction on the curve `w**2 = Q(x)`, the residues and the functions
//! with a given divisor), and the roots of index `n >= 3` by components on
//! the curve `y**n = P`.
//!
//! The Python module is generic over the field of the constants (`QQ`,
//! the rational functions of the parameters, a number field, both, `EX`):
//! here [`Field`] is generic over the [`Domain`] and the five cases are
//! monomorphised where Python dispatches at run time.

// References: B. M. Trager, Integration of algebraic functions, PhD thesis, MIT, 1984;
// M. Bronstein, Symbolic integration tutorial, ISSAC 1998, section 4.

#![allow(clippy::mutable_key_type)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::symbol::{Dummy, Symbol};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::densebasic::dup_strip;
use crate::polys::domains::{AlgebraicField, Domain, PolynomialRing};
use crate::polys::euclidtools::dup_resultant;
use crate::polys::numberfields::qqbar::{algebraic_roots, primitive_element, Algebraic};
use crate::polys::polyclasses::DensePoly;

type R = PolyResult<Basic>;

/// An element of a number field, its coordinates on the powers of the
/// generator, highest degree first.
pub type Number = Vec<Rat>;

/// The multiples of a residue class tried (Mazur's bound for a curve of
/// genus one over the rationals).
pub const TORSION_BOUND: i64 = 12;
/// The degree beyond which the field of the constants is not built.
pub const MAX_FIELD_DEGREE: usize = 32;

static COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------
// expressions

pub(crate) fn int(n: i64) -> Basic {
    Basic::int(n)
}
pub(crate) fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
pub(crate) fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
pub(crate) fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
pub(crate) fn neg(a: &Basic) -> R {
    mul(&[int(-1), a.clone()])
}
pub(crate) fn sub(a: &Basic, b: &Basic) -> R {
    add(&[a.clone(), neg(b)?])
}
pub(crate) fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pw(b, &int(-1))?])
}
pub(crate) fn sqrt(a: &Basic) -> R {
    pw(a, &Basic::half())
}
pub(crate) fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
pub(crate) fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
pub(crate) fn fraction(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::simplify::radsimp::fraction(e)
}
pub(crate) fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1)])
}

/// The errors which the Python module catches (`_FAILURES`): the
/// polynomial errors, `NotImplementedError`, `ValueError`, `TypeError`,
/// `ZeroDivisionError`, `ArithmeticError`; a part not ported yet is not
/// one of them.
pub(crate) fn is_failure(e: &PolyError) -> bool {
    match e {
        PolyError::NotImplemented(m) => !m.starts_with(crate::error::NOT_PORTED_PREFIX),
        PolyError::ZeroDivision(_)
        | PolyError::Value(_)
        | PolyError::HeuristicGCDFailed(_)
        | PolyError::Arithmetic(_)
        | PolyError::Type(_)
        | PolyError::Polys(_, _) => true,
        _ => false,
    }
}

/// `Ok(None)` for the errors of [`is_failure`].
pub(crate) fn caught<T>(r: PolyResult<T>) -> PolyResult<Option<T>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if is_failure(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

fn arithmetic_error(message: &str) -> PolyError {
    PolyError::Arithmetic(message.to_string())
}

/// A dummy of a fresh name (`_fresh`).
pub(crate) fn fresh(name: &str, real: bool, positive: bool) -> Basic {
    let label: String = format!("_trager_{}_{}", name, COUNTER.fetch_add(1, Ordering::Relaxed));
    if positive {
        return Dummy::new(&label, &[(Predicate::positive, true)]).0;
    }
    if real {
        return Dummy::new(&label, &[(Predicate::real, true)]).0;
    }
    Dummy::new(&label, &[]).0
}

/// `e` with the evaluated replacements of `rule` (`_replaced`).
pub(crate) fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

pub(crate) fn is_pow(e: &Basic) -> bool {
    matches!(e.head(), Head::Pow)
}

/// The powers in `e` (`_pows`), in the order of the tree.
pub(crate) fn pows(e: &Basic) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if is_pow(&n) && !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

fn exponent_rational(node: &Basic) -> Option<Rat> {
    node.args()[1].as_rational().cloned()
}

/// Whether the exponent of the power is a half integer (`_half_integer`).
pub(crate) fn half_integer(node: &Basic) -> bool {
    matches!(exponent_rational(node), Some(q) if crate::numbers::denom(&q) == Int::from(2))
}

/// The rational exponent of the power (`_exponent`).
pub(crate) fn exponent(node: &Basic) -> PolyResult<Rat> {
    exponent_rational(node).ok_or_else(|| PolyError::Type(format!("{} has no rational exponent", node)))
}

/// The free symbols (the dummies too: Python's `Dummy` is a `Symbol`)
/// sorted by name (`_symbols`).
pub(crate) fn symbols(e: &Basic) -> Vec<Basic> {
    let mut out: Vec<Basic> = e
        .free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_) | Head::Dummy(_, _)))
        .collect();
    out.sort_by_key(|s| s.to_string());
    out
}

pub(crate) fn is_polynomial(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
}

pub(crate) fn is_rational_function(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Rational) == Some(true)
}

fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// The coefficients of the powers of `y` in `e`, a polynomial in `y`
/// (`_powers_of`; `Value` when it is not one), in the order the powers
/// first appear.
pub(crate) fn powers_of(e: &Basic, y: &Basic) -> PolyResult<Vec<(i64, Basic)>> {
    let mut collected: Vec<(i64, Vec<Basic>)> = Vec::new();
    for term in crate::core::expr::terms_of(&expand(e)?) {
        let mut k: i64 = 0;
        let mut rest: Vec<Basic> = Vec::new();
        for factor in mul_args(&term) {
            if factor == *y {
                k += 1;
            } else if is_pow(&factor) && factor.args()[0] == *y {
                match factor.args()[1].as_rational() {
                    Some(q) if q.is_int() && *q >= Rat::ZERO => {
                        k += i64::try_from(crate::numbers::numer(q))
                            .map_err(|_| PolyError::Value("a large power".to_string()))?;
                    }
                    _ => return Err(PolyError::Value(format!("{} is not a polynomial in {}", e, y))),
                }
            } else if factor.has(y) {
                return Err(PolyError::Value(format!("{} is not a polynomial in {}", e, y)));
            } else {
                rest.push(factor);
            }
        }
        let product: Basic = mul(&rest)?;
        match collected.iter_mut().find(|(j, _)| *j == k) {
            Some((_, v)) => v.push(product),
            None => collected.push((k, vec![product])),
        }
    }
    collected.into_iter().map(|(k, terms)| Ok((k, add(&terms)?))).collect()
}

pub(crate) fn power_degree(p: &[(i64, Basic)]) -> i64 {
    p.iter().map(|(k, _)| *k).max().unwrap_or(0)
}

pub(crate) fn power_get(p: &[(i64, Basic)], k: i64) -> Basic {
    p.iter()
        .find(|(j, _)| *j == k)
        .map(|(_, c)| c.clone())
        .unwrap_or_else(Basic::zero)
}

// ---------------------------------------------------------------------
// the number fields of the constants

/// The algebraic number of a constant expression, the root of its minimal
/// polynomial nearest to its value, `None` when it is not one or the
/// nearest root is not isolated (`_algebraic_number`).
pub(crate) fn algebraic_number(e: &Basic) -> PolyResult<Option<Algebraic>> {
    if *e == Basic::imaginary_unit() {
        return Ok(Some(Algebraic::imaginary_unit()));
    }
    if let Some(q) = e.as_rational() {
        return Ok(Some(Algebraic::from_rational(q)));
    }
    let Some(coefficients) = caught(crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(e))? else {
        return Ok(None);
    };
    let Some(value) = crate::core::evalf::numeric_value(e, crate::numeric::dps_to_prec(30)) else {
        return Ok(None);
    };
    let (vr, vi) = crate::core::evalf::parts(&value);
    let (vr, vi): (f64, f64) = (vr.to_f64().value(), vi.to_f64().value());
    let Some(candidates_roots) = caught(algebraic_roots(&coefficients))? else {
        return Ok(None);
    };
    let mut candidates: Vec<(f64, Algebraic)> = Vec::new();
    for r in candidates_roots {
        let (a, b) = r.to_complex()?;
        candidates.push((((a - vr).powi(2) + (b - vi).powi(2)).sqrt(), r));
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let magnitude: f64 = (vr * vr + vi * vi).sqrt();
    if candidates.len() > 1 && candidates[1].0 < 1e-12 * (1.0 + magnitude) {
        return Ok(None);
    }
    Ok(Some(candidates.swap_remove(0).1))
}

fn horner(coefficients: &[Rat], point: &Number, k: &AlgebraicField) -> Number {
    let mut result: Number = k.zero();
    for c in coefficients {
        result = k.add(&k.mul(&result, point), &k.from_rational(c));
    }
    result
}

fn monic_minpoly(minpoly: &[Int]) -> Vec<Rat> {
    let lead: Rat = Rat::from(minpoly[0].clone());
    minpoly.iter().map(|c| Rat::from(c.clone()) / &lead).collect()
}

/// The number field `QQ(theta)` of the constants `generators`, `theta` a
/// primitive element (`_NumberField`).
#[derive(Debug, Clone)]
pub struct NumberField {
    pub theta: Basic,
    pub field: AlgebraicField,
    pub degree: usize,
    pub theta_expr: Basic,
    pub coordinates: Vec<(Basic, Number)>,
    pub forms: Vec<(Basic, Basic)>,
}

impl NumberField {
    /// `Arithmetic` when a generator is not an algebraic number or the
    /// field is larger than `bound`.
    pub fn new(generators: &[Basic], bound: usize) -> PolyResult<NumberField> {
        let theta: Basic = fresh("theta", false, false);
        let mut values: Vec<Algebraic> = Vec::new();
        for g in generators {
            match algebraic_number(g)? {
                Some(v) => values.push(v),
                None => return Err(arithmetic_error(&format!("{} is not an algebraic number", g))),
            }
        }
        let mut gamma: Algebraic = values[0].clone();
        let mut gamma_expr: Basic = generators[0].clone();
        let mut coordinates: Vec<Number> = vec![vec![Rat::ONE, Rat::ZERO]];
        for (g, value) in generators[1..].iter().zip(&values[1..]) {
            let (new, p, q) = primitive_element(&gamma, value)?;
            if new.degree() > bound {
                return Err(arithmetic_error("the field of the constants is too large"));
            }
            let k: AlgebraicField = AlgebraicField::new(monic_minpoly(&new.minpoly()), theta.clone())?;
            let gamma_in_new: Number = horner(&p, &k.alpha(), &k);
            coordinates = coordinates.iter().map(|c| horner(c, &gamma_in_new, &k)).collect();
            coordinates.push(horner(&q, &k.alpha(), &k));
            let mut c: i64 = 0;
            if !new.equals(&gamma)? {
                let limit: i64 = (4 * gamma.degree() * value.degree() + 2) as i64;
                let mut found: Option<i64> = None;
                for j in 1..limit {
                    let candidate: Algebraic = gamma.add(&value.mul(&Algebraic::from_rational(&Rat::from(j)))?)?;
                    if candidate.equals(&new)? {
                        found = Some(j);
                        break;
                    }
                }
                c = found.ok_or_else(|| arithmetic_error("no integer combination gives the primitive element"))?;
            }
            gamma_expr = add(&[gamma_expr, mul(&[int(c), g.clone()])?])?;
            gamma = new;
        }
        if gamma.degree() > bound {
            return Err(arithmetic_error("the field of the constants is too large"));
        }
        let minpoly: Vec<Rat> = monic_minpoly(&gamma.minpoly());
        let field: AlgebraicField = AlgebraicField::unordered(minpoly.clone(), theta.clone())?;
        let degree: usize = minpoly.len() - 1;
        let mut coords: Vec<(Basic, Number)> = Vec::new();
        let mut forms: Vec<(Basic, Basic)> = Vec::new();
        for (g, element) in generators.iter().zip(coordinates) {
            let reduced: Number = field.reduce(element);
            forms.push((g.clone(), field.to_expr(&reduced)?));
            coords.push((g.clone(), reduced));
        }
        Ok(NumberField {
            theta,
            field,
            degree,
            theta_expr: gamma_expr,
            coordinates: coords,
            forms,
        })
    }
}

// ---------------------------------------------------------------------
// the fields of the constants

/// A field of the constants with the conversions of the core expressions
/// (`_Field`).
#[derive(Debug, Clone)]
pub struct Field<D: Domain> {
    pub domain: D,
    pub number_field: Option<NumberField>,
    pub parameters: Vec<Basic>,
    pub rationals: bool,
    pub restore: Vec<(Basic, Basic)>,
}

fn plain_symbol(p: &Basic) -> Basic {
    match p.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => Symbol::new(&d.name, &[]).0,
        _ => p.clone(),
    }
}

pub(crate) fn name_of(p: &Basic) -> String {
    match p.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => p.to_string(),
    }
}

impl<D: Domain> Field<D> {
    pub fn new(domain: D, number_field: Option<NumberField>, parameters: Vec<Basic>, rationals: bool) -> Field<D> {
        let restore: Vec<(Basic, Basic)> = parameters
            .iter()
            .filter(|p| plain_symbol(p) != **p)
            .map(|p| (plain_symbol(p), p.clone()))
            .collect();
        Field {
            domain,
            number_field,
            parameters,
            rationals,
            restore,
        }
    }

    /// The element of a constant expression.
    pub fn element(&self, e: &Basic) -> PolyResult<D::Elem> {
        match &self.number_field {
            Some(f) => self.domain.from_expr(&replaced(e, &f.forms)?),
            None => self.domain.from_expr(e),
        }
    }

    /// The expression of an element.
    pub fn expr(&self, c: &D::Elem) -> R {
        let e: Basic = replaced(&self.domain.to_expr(c)?, &self.restore)?;
        match &self.number_field {
            None => Ok(e),
            Some(f) => expand(&replaced(&e, &[(f.theta.clone(), f.theta_expr.clone())])?),
        }
    }

    /// A polynomial over another field of the constants, over this one.
    pub fn poly<K: Domain>(&self, p: &DensePoly<K>) -> PolyResult<DensePoly<D>> {
        let rep: Vec<D::Elem> = p
            .rep
            .iter()
            .map(|c| self.element(&p.domain.to_expr(c)?))
            .collect::<PolyResult<_>>()?;
        Ok(DensePoly::new(&rep, self.domain.clone(), p.gen.clone()))
    }

    /// The polynomial in `x` of an expression.
    pub fn from_expr(&self, e: &Basic, x: &Basic) -> PolyResult<DensePoly<D>> {
        let coefficients: Vec<(i64, Basic)> = powers_of(e, x)?;
        let degree: i64 = power_degree(&coefficients);
        let mut rep: Vec<D::Elem> = Vec::new();
        for k in (0..=degree).rev() {
            rep.push(self.element(&power_get(&coefficients, k))?);
        }
        Ok(DensePoly::new(&rep, self.domain.clone(), Symbol(x.clone())))
    }

    /// The expression of a polynomial.
    pub fn poly_expr(&self, p: &DensePoly<D>) -> R {
        let x: &Basic = &p.gen.0;
        let n: isize = p.degree();
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in p.rep.iter().enumerate() {
            if self.domain.is_zero(c) {
                continue;
            }
            let k: isize = n - i as isize;
            let power: Basic = match k {
                0 => Basic::one(),
                1 => x.clone(),
                _ => pw(x, &int(k as i64))?,
            };
            terms.push(mul(&[self.expr(c)?, power])?);
        }
        add(&terms)
    }
}

pub type Rationals = crate::polys::domains::RationalField;
pub type Parametric = crate::polys::domains::SparseFractionField<crate::polys::domains::RationalField>;
pub type AlgebraicParametric = crate::polys::domains::SparseFractionField<AlgebraicField>;
pub type Expressions = crate::polys::domains::ExpressionDomain;

/// `QQ` (`_rationals`).
pub fn rationals() -> Field<Rationals> {
    Field::new(crate::polys::domains::QQ, None, Vec::new(), true)
}

/// `QQ(parameters)` (`_parametric`).
pub fn parametric(parameters: &[Basic]) -> Field<Parametric> {
    let names: Vec<String> = parameters.iter().map(name_of).collect();
    let ring = crate::polys::domains::SparsePolynomialRing::sympy(crate::polys::domains::QQ, names);
    Field::new(
        crate::polys::domains::SparseFractionField::new(ring),
        None,
        parameters.to_vec(),
        false,
    )
}

/// The number field (`_algebraic`).
pub fn algebraic(f: &NumberField) -> Field<AlgebraicField> {
    Field::new(f.field.clone(), Some(f.clone()), Vec::new(), false)
}

/// The number field with the parameters (`_algebraic_parametric`).
pub fn algebraic_parametric(f: &NumberField, parameters: &[Basic]) -> Field<AlgebraicParametric> {
    let names: Vec<String> = parameters.iter().map(name_of).collect();
    let ring = crate::polys::domains::SparsePolynomialRing::new(f.field.clone(), names);
    Field::new(
        crate::polys::domains::SparseFractionField::new(ring),
        Some(f.clone()),
        parameters.to_vec(),
        false,
    )
}

/// `EX` (`_expressions`).
pub fn expressions() -> Field<Expressions> {
    Field::new(crate::polys::domains::EX, None, Vec::new(), false)
}

/// The parameters of the expressions, whether `I` is among the constants
/// and whether other constants are (`_constant_kind`).
pub fn constant_kind(exprs: &[Basic], x: &Basic) -> PolyResult<(Vec<Basic>, bool, bool)> {
    let mut parameters: Vec<Basic> = Vec::new();
    for e in exprs {
        for s in symbols(e) {
            if s != *x && !parameters.contains(&s) {
                parameters.push(s);
            }
        }
    }
    parameters.sort_by_key(|s| s.to_string());
    let mut names: Vec<String> = vec![name_of(x)];
    names.extend(parameters.iter().map(name_of));
    let mut gaussian: bool = false;
    let mut parts: Vec<Basic> = Vec::new();
    for e in exprs {
        let (n, d) = fraction(&cancel(e)?)?;
        parts.push(n);
        parts.push(d);
    }
    let gaussian_field: AlgebraicField = gaussian_field();
    for e in &parts {
        match crate::polys::polyutils::dict_from_expr(e, &names, &crate::polys::domains::QQ) {
            Ok(_) => continue,
            Err(PolyError::Value(_)) => {}
            Err(err) => return Err(err),
        }
        match crate::polys::polyutils::dict_from_expr(e, &names, &gaussian_field) {
            Ok(_) => gaussian = true,
            Err(PolyError::Value(_)) => return Ok((parameters, false, true)),
            Err(err) => return Err(err),
        }
    }
    Ok((parameters, gaussian, false))
}

/// `QQ(I)` as a number field of the constants (`_gaussian`).
pub fn gaussian() -> PolyResult<NumberField> {
    NumberField::new(&[Basic::imaginary_unit()], MAX_FIELD_DEGREE)
}

fn gaussian_field() -> AlgebraicField {
    AlgebraicField::new(vec![Rat::ONE, Rat::ZERO, Rat::ONE], Basic::imaginary_unit()).expect("a monic modulus")
}

// ---------------------------------------------------------------------
// the integrand

/// The integrand `A + B y` with `y = sqrt(base)` (`_Parsed`).
#[derive(Debug, Clone)]
pub struct Parsed {
    pub base: Basic,
    pub a: Basic,
    pub b: Basic,
}

/// The integrand as `A + B sqrt(P)` with `P` a polynomial in `x`, `None`
/// when it is not one (`_parse`).
pub fn parse(f: &Basic, x: &Basic) -> PolyResult<Option<Parsed>> {
    let mut bases: Vec<Basic> = Vec::new();
    for node in pows(f) {
        if half_integer(&node) {
            let base: &Basic = &node.args()[0];
            if !base.has(x) {
                continue;
            }
            if !is_polynomial(base, std::slice::from_ref(x)) {
                return Ok(None);
            }
            if !bases.contains(base) {
                bases.push(base.clone());
            }
        }
    }
    if bases.len() > 1 {
        bases.sort_by_key(|b| b.to_string());
        let Some(combined) = combined_roots(f, x, &bases)? else {
            return Ok(None);
        };
        return parse(&combined, x);
    }
    if bases.len() != 1 {
        return Ok(None);
    }
    let base: Basic = bases.swap_remove(0);
    let y: Basic = fresh("y", false, false);
    let to_y = |node: &Basic| -> R {
        if is_pow(node) && node.args()[0] == base && half_integer(node) {
            return pw(
                &y,
                &int(i64::try_from(crate::numbers::numer(&exponent(node)?)).unwrap_or(1)),
            );
        }
        Ok(node.clone())
    };
    let g: Basic = crate::core::traversal::bottom_up(f, &to_y, false)?;
    if !is_rational_function(&g, &[x.clone(), y.clone()]) {
        return Ok(None);
    }
    let p = match powers_of(&base, x) {
        Ok(p) => p,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if power_degree(&p) < 1 {
        return Ok(None);
    }
    let (numerator, denominator) = fraction(&cancel(&g)?)?;
    let reduced = (|| -> PolyResult<_> {
        Ok((
            reduce_modulo_curve(&numerator, &y, &base)?,
            reduce_modulo_curve(&denominator, &y, &base)?,
        ))
    })();
    let ((n0, n1), (d0, d1)) = match reduced {
        Ok(v) => v,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let norm: Basic = expand(&sub(&pw(&d0, &int(2))?, &mul(&[pw(&d1, &int(2))?, base.clone()])?)?)?;
    if norm == Basic::zero() {
        return Ok(None);
    }
    let a: Basic = cancel(&div(
        &sub(
            &mul(&[n0.clone(), d0.clone()])?,
            &mul(&[n1.clone(), d1.clone(), base.clone()])?,
        )?,
        &norm,
    )?)?;
    let b: Basic = cancel(&div(&sub(&mul(&[n1, d0])?, &mul(&[n0, d1])?)?, &norm)?)?;
    Ok(Some(Parsed { base, a, b }))
}

/// Several square roots of polynomials made one, the root of their
/// product, when every term has all or none of them (`_combined_roots`).
pub fn combined_roots(f: &Basic, x: &Basic, bases: &[Basic]) -> PolyResult<Option<Basic>> {
    let ys: Vec<Basic> = (0..bases.len())
        .map(|i| fresh(&format!("Y{}", i), false, false))
        .collect();
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for node in pows(f) {
        let base: &Basic = &node.args()[0];
        if let Some(i) = bases.iter().position(|b| b == base) {
            if half_integer(&node) {
                let k: i64 = i64::try_from(crate::numbers::numer(&exponent(&node)?)).unwrap_or(1);
                let m: i64 = k.rem_euclid(2);
                replacement.push((
                    node.clone(),
                    mul(&[pw(base, &int((k - m) / 2))?, pw(&ys[i], &int(m))?])?,
                ));
            }
        }
    }
    let g: Basic = replaced(f, &replacement)?;
    let mut gens: Vec<Basic> = vec![x.clone()];
    gens.extend(ys.iter().cloned());
    if !is_rational_function(&g, &gens) {
        return Ok(None);
    }
    let (numerator, denominator) = fraction(&cancel(&g)?)?;
    let y: Basic = fresh("y", false, false);
    let product: Basic = mul(bases)?;
    let mut rewritten: Vec<Basic> = Vec::new();
    for part in [numerator, denominator] {
        let Some(poly) = caught(crate::polys::polytools::Poly::new(&part, &ys))? else {
            return Ok(None);
        };
        let Some(terms) = caught(poly.terms())? else {
            return Ok(None);
        };
        let mut total: Vec<Basic> = Vec::new();
        for (monomial, coefficient) in terms {
            let parities: Vec<u32> = monomial.iter().map(|e| e % 2).collect();
            let mut factors: Vec<Basic> = Vec::new();
            for (i, e) in monomial.iter().enumerate() {
                factors.push(pw(&bases[i], &int(((e - e % 2) / 2) as i64))?);
            }
            let factor: Basic = mul(&factors)?;
            if parities.iter().all(|p| *p != 0) {
                total.push(mul(&[coefficient, factor, y.clone()])?);
            } else if parities.iter().all(|p| *p == 0) {
                total.push(mul(&[coefficient, factor])?);
            } else {
                return Ok(None);
            }
        }
        rewritten.push(add(&total)?);
    }
    Ok(Some(replaced(
        &div(&rewritten[0], &rewritten[1])?,
        &[(y, sqrt(&product)?)],
    )?))
}

/// `N` modulo `y**2 - P` as `(n0, n1)` with `N = n0 + n1 y`
/// (`_reduce_modulo_curve`).
pub fn reduce_modulo_curve(n: &Basic, y: &Basic, p: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut n0: Vec<Basic> = Vec::new();
    let mut n1: Vec<Basic> = Vec::new();
    for (k, coefficient) in powers_of(n, y)? {
        if k % 2 == 0 {
            n0.push(mul(&[coefficient, pw(p, &int(k / 2))?])?);
        } else {
            n1.push(mul(&[coefficient, pw(p, &int((k - 1) / 2))?])?);
        }
    }
    Ok((expand(&add(&n0)?)?, expand(&add(&n1)?)?))
}

/// `sqrt(P)` as `d w` with `w**2 = Q` squarefree (`_Radical`).
pub struct Radical<'a, D: Domain> {
    pub f: &'a Field<D>,
    pub x: Basic,
    pub p: DensePoly<D>,
    pub a: Basic,
    pub b: Basic,
    pub d: DensePoly<D>,
    pub q: DensePoly<D>,
    pub p_expr: Basic,
    pub d_expr: Basic,
    pub q_expr: Basic,
}

impl<'a, D: Domain> Radical<'a, D> {
    pub fn new(f: &'a Field<D>, x: &Basic, p: DensePoly<D>, a: &Basic, b: &Basic) -> PolyResult<Radical<'a, D>> {
        let (d, q) = split_square(&p)?;
        let p_expr: Basic = f.poly_expr(&p)?;
        let d_expr: Basic = f.poly_expr(&d)?;
        let q_expr: Basic = f.poly_expr(&q)?;
        Ok(Radical {
            f,
            x: x.clone(),
            p,
            a: a.clone(),
            b: b.clone(),
            d,
            q,
            p_expr,
            d_expr,
            q_expr,
        })
    }

    /// `w = sqrt(P)/d`.
    pub fn w(&self) -> R {
        div(&sqrt(&self.p_expr)?, &self.d_expr)
    }

    /// `B d Q` as `M/D` (the differential `M/(D w)`).
    pub fn algebraic_numerator(&self) -> PolyResult<(DensePoly<D>, DensePoly<D>)> {
        let e: Basic = cancel(&mul(&[self.b.clone(), self.d_expr.clone(), self.q_expr.clone()])?)?;
        let (numerator, denominator) = fraction(&e)?;
        Ok((
            self.f.from_expr(&numerator, &self.x)?,
            self.f.from_expr(&denominator, &self.x)?,
        ))
    }
}

/// `P = d**2 Q` with `Q` squarefree (`_split_square`).
pub fn split_square<D: Domain>(p: &DensePoly<D>) -> PolyResult<(DensePoly<D>, DensePoly<D>)> {
    let (content, factors) = p.sqf_list();
    let k: &D = &p.domain;
    let mut d: DensePoly<D> = p.with_rep(vec![k.one()]);
    let mut q: DensePoly<D> = p.with_rep(dup_strip(&[content], k));
    for (factor, multiplicity) in factors {
        d = d.mul(&factor.pow(multiplicity / 2))?;
        if multiplicity % 2 == 1 {
            q = q.mul(&factor)?;
        }
    }
    Ok((d, q))
}

/// `c x**j` (`_monomial`).
pub fn monomial<D: Domain>(c: D::Elem, j: usize, k: &D, x: &Symbol) -> DensePoly<D> {
    let mut rep: Vec<D::Elem> = vec![c];
    rep.extend(std::iter::repeat(k.zero()).take(j));
    DensePoly::new(&rep, k.clone(), x.clone())
}

/// What the Hermite reduction leaves (`_Reduction`): the derivatives
/// `S w/T` taken out, the remainder `r/(D w)` and the polynomial part.
#[derive(Debug, Clone)]
pub struct Reduction<D: Domain> {
    pub terms: Vec<(DensePoly<D>, DensePoly<D>)>,
    pub r: DensePoly<D>,
    pub d: DensePoly<D>,
    pub polynomial: DensePoly<D>,
}

fn one_like<D: Domain>(p: &DensePoly<D>) -> DensePoly<D> {
    p.with_rep(vec![p.domain.one()])
}

fn ground<D: Domain>(k: &D, n: i64) -> D::Elem {
    k.from_int(n)
}

/// The Hermite reduction of `M/(D w)` on `w**2 = Q` (`_hermite`).
pub fn hermite<D: Domain>(
    m: &DensePoly<D>,
    d: &DensePoly<D>,
    q: &DensePoly<D>,
    x: &Symbol,
) -> PolyResult<Reduction<D>> {
    let mut terms: Vec<(DensePoly<D>, DensePoly<D>)> = Vec::new();
    let one: DensePoly<D> = one_like(q);
    let (mut m, mut d) = (m.clone(), d.clone());
    loop {
        crate::timeout::check()?;
        let common: DensePoly<D> = m.gcd(&d)?;
        m = m.exquo(&common)?;
        d = d.exquo(&common)?;
        let lead: D::Elem = d.LC();
        m = m.exquo_ground(&lead)?;
        d = d.monic()?;
        let Some((m2, d2, s, t)) = reduction_step(&m, &d, q, x)? else {
            break;
        };
        m = m2;
        d = d2;
        terms.push((s, t));
    }
    let (p, r1) = m.div(&d)?;
    let (s, r2) = reduce_at_infinity(&p, q, x)?;
    if !s.is_zero() {
        terms.push((s, one));
    }
    let r: DensePoly<D> = r2.mul(&d)?.add(&r1)?;
    let common: DensePoly<D> = r.gcd(&d)?;
    Ok(Reduction {
        terms,
        r: r.exquo(&common)?,
        d: d.exquo(&common)?,
        polynomial: r2,
    })
}

/// One step of the reduction at a multiple or ramified factor of `D`
/// (`_reduction_step`): `(M, D, S, T)`.
fn reduction_step<D: Domain>(
    m: &DensePoly<D>,
    d: &DensePoly<D>,
    q: &DensePoly<D>,
    _x: &Symbol,
) -> PolyResult<Option<(DensePoly<D>, DensePoly<D>, DensePoly<D>, DensePoly<D>)>> {
    let k: &D = &q.domain;
    let (_, mut factors) = d.sqf_list();
    factors.sort_by_key(|(_, e)| std::cmp::Reverse(*e));
    for (v, e) in factors {
        let ramified: DensePoly<D> = v.gcd(q)?;
        if ramified.degree() > 0 {
            let v: DensePoly<D> = ramified;
            let u_: DensePoly<D> = d.exquo(&v.pow(e))?;
            let q_: DensePoly<D> = q.exquo(&v)?;
            let inverse_: DensePoly<D> = inverse(
                &u_.mul(&v.diff(1))?.mul(&q_)?.mul_ground(&ground(k, 1 - 2 * e as i64)),
                &v,
            )?;
            let s: DensePoly<D> = m.mul(&inverse_)?.mul_ground(&ground(k, 2)).rem(&v)?;
            let subtracted: DensePoly<D> = s
                .diff(1)
                .mul(&q_)?
                .mul(&v)?
                .mul_ground(&ground(k, 2))
                .add(&s.mul(&q.diff(1))?)?
                .sub(&s.mul(&v.diff(1))?.mul(&q_)?.mul_ground(&ground(k, 2 * e as i64)))?;
            let m_: DensePoly<D> = m.sub(&u_.mul(&subtracted)?.exquo_ground(&ground(k, 2))?)?.exquo(&v)?;
            let vk: DensePoly<D> = v.pow(e);
            return Ok(Some((m_, u_.mul(&v.pow(e - 1))?, s, vk)));
        }
        if e >= 2 {
            let u_: DensePoly<D> = d.exquo(&v.pow(e))?;
            let inverse_: DensePoly<D> =
                inverse(&u_.mul(&v.diff(1))?.mul(q)?.mul_ground(&ground(k, e as i64 - 1)), &v)?;
            let s: DensePoly<D> = m.mul(&inverse_)?.rem(&v)?.neg();
            let subtracted: DensePoly<D> = s
                .diff(1)
                .mul(q)?
                .mul(&v)?
                .mul_ground(&ground(k, 2))
                .add(&s.mul(&q.diff(1))?.mul(&v)?)?
                .sub(&s.mul(&v.diff(1))?.mul(q)?.mul_ground(&ground(k, 2 * (e as i64 - 1))))?;
            let m_: DensePoly<D> = m.sub(&u_.mul(&subtracted)?.exquo_ground(&ground(k, 2))?)?.exquo(&v)?;
            return Ok(Some((m_, u_.mul(&v.pow(e - 1))?, s, v.pow(e - 1))));
        }
    }
    Ok(None)
}

/// The polynomial part lowered below `deg Q - 1` by the derivatives of
/// `S w` (`_reduce_at_infinity`): `(S, p)`.
fn reduce_at_infinity<D: Domain>(
    p: &DensePoly<D>,
    q: &DensePoly<D>,
    x: &Symbol,
) -> PolyResult<(DensePoly<D>, DensePoly<D>)> {
    let k: &D = &q.domain;
    let mut s: DensePoly<D> = q.with_rep(Vec::new());
    let n: isize = q.degree();
    let two: D::Elem = ground(k, 2);
    let mut p: DensePoly<D> = p.clone();
    while !p.is_zero() && p.degree() >= n - 1 {
        crate::timeout::check()?;
        let j: isize = p.degree() - n + 1;
        let c: D::Elem = k.exquo(
            &k.mul(&two, &p.LC()),
            &k.mul(&q.LC(), &ground(k, 2 * j as i64 + n as i64)),
        )?;
        let mono: DensePoly<D> = monomial(c, j as usize, k, x);
        s = s.add(&mono)?;
        p = p.sub(
            &mono
                .diff(1)
                .mul(q)?
                .mul_ground(&two)
                .add(&mono.mul(&q.diff(1))?)?
                .exquo_ground(&two)?,
        )?;
    }
    Ok((s, p))
}

/// The inverse of `p` modulo `modulus` (`_inverse`; `NotInvertible` as a
/// polynomial error).
pub fn inverse<D: Domain>(p: &DensePoly<D>, modulus: &DensePoly<D>) -> PolyResult<DensePoly<D>> {
    let (s, _, h) = p.gcdex(modulus)?;
    if h.degree() != 0 {
        return Err(crate::error::polys_error(
            crate::error::PolysErrorKind::NotInvertible,
            "zero divisor".to_string(),
        ));
    }
    s.exquo_ground(&h.LC())
}

/// `Res_x(F, G0 + z**power G1)` as a polynomial in `z` (`_resultant_in`).
pub fn resultant_in<D: Domain>(
    f: &DensePoly<D>,
    g0: &DensePoly<D>,
    g1: &DensePoly<D>,
    power: usize,
    z: &Basic,
) -> DensePoly<D> {
    let k: &D = &f.domain;
    let kz: PolynomialRing<D> = PolynomialRing::new(k.clone(), &name_of(z));
    let lifted = |p: &DensePoly<D>| -> Vec<Vec<D::Elem>> {
        p.rep.iter().map(|c| dup_strip(std::slice::from_ref(c), k)).collect()
    };
    let mut zp: Vec<D::Elem> = vec![k.one()];
    zp.extend(std::iter::repeat(k.zero()).take(power));
    let shifted: Vec<Vec<D::Elem>> = lifted(g1).iter().map(|c| kz.mul(c, &zp)).collect();
    let g: Vec<Vec<D::Elem>> = dup_strip(&dense_add(&lifted(g0), &shifted, &kz), &kz);
    let f_: Vec<Vec<D::Elem>> = lifted(f);
    let resultant: Vec<D::Elem> = dup_resultant(&f_, &g, &kz);
    DensePoly::new(&resultant, k.clone(), Symbol(z.clone()))
}

fn dense_add<D: Domain>(f: &[D::Elem], g: &[D::Elem], k: &D) -> Vec<D::Elem> {
    let n: usize = f.len().max(g.len());
    let mut f_: Vec<D::Elem> = vec![k.zero(); n - f.len()];
    f_.extend(f.iter().cloned());
    let mut g_: Vec<D::Elem> = vec![k.zero(); n - g.len()];
    g_.extend(g.iter().cloned());
    let sum: Vec<D::Elem> = f_.iter().zip(&g_).map(|(a, b)| k.add(a, b)).collect();
    dup_strip(&sum, k)
}

/// The Rothstein–Trager resultant `Res_x(D, r**2 - z**2 D'**2 Q)`
/// (`_residue_polynomial`).
pub fn residue_polynomial<D: Domain>(
    r: &DensePoly<D>,
    d: &DensePoly<D>,
    q: &DensePoly<D>,
    z: &Basic,
) -> PolyResult<DensePoly<D>> {
    Ok(resultant_in(d, &r.mul(r)?, &d.diff(1).pow(2).mul(q)?.neg(), 2, z))
}

// ---------------------------------------------------------------------
// the residues

/// The irreducible factors of the resultant as expressions in `z`
/// (`_factor_exprs`).
fn factor_exprs<D: Domain>(f: &Field<D>, r: &DensePoly<D>, z: &Basic) -> PolyResult<Vec<Basic>> {
    match r.factor_list() {
        Ok((_, factors)) => return factors.iter().map(|(g, _)| f.poly_expr(g)).collect(),
        Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {}
        Err(e) => return Err(e),
    }
    let (numerator, _) = fraction(&cancel(&f.poly_expr(r)?)?)?;
    match caught(crate::polys::polytools::factor_list(
        &numerator,
        std::slice::from_ref(z),
    ))? {
        Some((_, found)) => Ok(found.into_iter().map(|(e, _)| e).collect()),
        None => Ok(vec![numerator]),
    }
}

fn roots_of(e: &Basic, z: &Basic) -> PolyResult<Option<Vec<(Basic, usize)>>> {
    let found = caught(crate::polys::polyroots::roots(
        e,
        std::slice::from_ref(z),
        true,
        true,
        None,
    ))?;
    Ok(found.map(|m| m.iter().map(|(r, k)| (r.clone(), *k)).collect()))
}

/// The roots of `e` in `z` in closed form, `None` when they are not all
/// found (`_closed_roots`).
pub fn closed_roots(e: &Basic, z: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let coefficients: Vec<(i64, Basic)> = powers_of(e, z)?;
    let degree: i64 = power_degree(&coefficients);
    if let Some(found) = roots_of(e, z)? {
        let total: i64 = found.iter().map(|(_, k)| *k as i64).sum();
        if total == degree {
            return Ok(Some(found.into_iter().map(|(r, _)| r).collect()));
        }
        return Ok(None);
    }
    let mut g: i64 = 0;
    for (k, _) in &coefficients {
        g = gcd_i64(g, *k);
    }
    if g > 1 && degree > g {
        let t: Basic = fresh("t", false, false);
        let mut terms: Vec<Basic> = Vec::new();
        for (k, c) in &coefficients {
            terms.push(mul(&[c.clone(), pw(&t, &int(k / g))?])?);
        }
        let Some(inner) = closed_roots(&add(&terms)?, &t)? else {
            return Ok(None);
        };
        let mut found: Vec<Basic> = Vec::new();
        for root in inner {
            let Some(powers) = binomial_roots(&root, g, &t)? else {
                return Ok(None);
            };
            found.extend(powers);
        }
        return Ok(Some(found));
    }
    let a: Basic = power_get(&coefficients, degree);
    let b: Basic = power_get(&coefficients, degree - 1);
    if degree == 1 {
        return Ok(Some(vec![cancel(&div(&neg(&b)?, &a)?)?]));
    }
    if degree == 2 {
        let c: Basic = power_get(&coefficients, 0);
        let root: Basic = sqrt(&expand(&sub(&pw(&b, &int(2))?, &mul(&[int(4), a.clone(), c])?)?)?)?;
        let two_a: Basic = mul(&[int(2), a])?;
        return Ok(Some(vec![
            expand(&div(&sub(&neg(&b)?, &root)?, &two_a)?)?,
            expand(&div(&add(&[neg(&b)?, root])?, &two_a)?)?,
        ]));
    }
    Ok(None)
}

pub(crate) fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i64(b, a % b)
    }
}

/// The `n`-th roots of `value` (`_binomial_roots`).
fn binomial_roots(value: &Basic, n: i64, z: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    if n == 2 {
        let root: Basic = sqrt(value)?;
        return Ok(Some(vec![neg(&root)?, root]));
    }
    let Some(unity) = roots_of(&sub(&pw(z, &int(n))?, &int(1))?, z)? else {
        return Ok(None);
    };
    if unity.iter().map(|(_, k)| *k as i64).sum::<i64>() != n {
        return Ok(None);
    }
    let principal: Basic = pw(
        value,
        &Basic::from_rational(Rat::from_parts(Int::ONE, dashu_int::UBig::from(n as u64))),
    )?;
    let mut out: Vec<Basic> = Vec::new();
    for (u, _) in unity {
        out.push(crate::core::function::expand_mul(&mul(&[u, principal.clone()])?, true)?);
    }
    Ok(Some(out))
}

/// The nonzero residues, the roots of the factors of the resultant in
/// closed form, `None` when some are not found (`_residue_values`).
fn residue_values<D: Domain>(f: &Field<D>, r: &DensePoly<D>, z: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let mut values: Vec<Basic> = Vec::new();
    for e in factor_exprs(f, r, z)? {
        let coefficients: Vec<(i64, Basic)> = powers_of(&e, z)?;
        let degree: i64 = power_degree(&coefficients);
        if degree == 0 {
            continue;
        }
        if degree == 1 && power_get(&coefficients, 0) == Basic::zero() {
            continue;
        }
        let Some(found) = closed_roots(&e, z)? else {
            return Ok(None);
        };
        for root in found {
            values.push(denested(&root)?);
        }
    }
    Ok(Some(values.into_iter().filter(|v| *v != Basic::zero()).collect()))
}

/// The radicals of a value with parameters denested (`_denested`).
pub(crate) fn denested(value: &Basic) -> R {
    if value.free_symbols().is_empty() {
        return Ok(value.clone());
    }
    crate::simplify::powsimp::powdenest(&crate::polys::polytools::factor(value, &[], false, true)?, true, false)
}

/// The algebraic numbers among the values: `I`, the radicals of numbers,
/// the numbers which are not rational (`_algebraic_atoms`), sorted.
fn algebraic_atoms(values: &[Basic]) -> Vec<Basic> {
    let mut numbers: Vec<Basic> = Vec::new();
    let i: Basic = Basic::imaginary_unit();
    for value in values {
        for node in value.preorder() {
            if !(is_pow(&node) || node == i) || !node.free_symbols().is_empty() {
                continue;
            }
            if node == i {
                if !numbers.contains(&i) {
                    numbers.push(i.clone());
                }
            } else if let Some(q) = node.args()[1].as_rational() {
                if crate::numbers::denom(q) > Int::ONE && !numbers.contains(&node) {
                    numbers.push(node.clone());
                }
            }
        }
        if value.free_symbols().is_empty() && value.as_rational().is_none() && !numbers.contains(value) {
            numbers.push(value.clone());
        }
    }
    numbers.sort_by_key(|n| n.to_string());
    numbers
}

/// The residues which are rational multiples of `rho`: the points
/// `(q, G, W)` above the roots of `G` on the branch `w = W mod G` with the
/// residue `q rho`, and the multiple at infinity (`_ResidueClass`).
#[derive(Debug, Clone)]
pub struct ResidueClass<U: Domain> {
    pub rho: Basic,
    pub finite: Vec<(Rat, DensePoly<U>, DensePoly<U>)>,
    pub q_infinity: Rat,
}

impl<U: Domain> ResidueClass<U> {
    /// The common denominator of the multiples.
    pub fn denominator(&self) -> Int {
        let mut d: Int = crate::numbers::denom(&self.q_infinity);
        for (q, _, _) in &self.finite {
            d = crate::numbers::lcm(&d, &crate::numbers::denom(q));
        }
        d
    }
}

/// The residues grouped by their rational ratios (`_classes`); `None` when
/// a residue has no point above it.
fn classes<U: Domain>(
    values: &[U::Elem],
    infinity: Option<&U::Elem>,
    l: &Field<U>,
    r: &DensePoly<U>,
    d: &DensePoly<U>,
    q: &DensePoly<U>,
) -> PolyResult<Option<Vec<ResidueClass<U>>>> {
    let k: &U = &l.domain;
    let mut representatives: Vec<U::Elem> = Vec::new();
    let mut all_values: Vec<U::Elem> = values.to_vec();
    if let Some(v) = infinity {
        all_values.push(v.clone());
    }
    for value in &all_values {
        let mut placed: bool = false;
        for representative in &representatives {
            if l.expr(&k.exquo(value, representative)?)?.as_rational().is_some() {
                placed = true;
                break;
            }
        }
        if !placed {
            representatives.push(value.clone());
        }
    }
    let dd: DensePoly<U> = d.diff(1);
    let mut out: Vec<ResidueClass<U>> = Vec::new();
    for representative in &representatives {
        let rho: Basic = l.expr(representative)?;
        let mut finite: Vec<(Rat, DensePoly<U>, DensePoly<U>)> = Vec::new();
        let mut seen: Vec<Rat> = Vec::new();
        for value in values {
            let Some(ratio) = l.expr(&k.exquo(value, representative)?)?.as_rational().cloned() else {
                continue;
            };
            let qq: Rat = if ratio < Rat::ZERO {
                -ratio.clone()
            } else {
                ratio.clone()
            };
            if seen.contains(&qq) {
                continue;
            }
            seen.push(qq.clone());
            let residue: U::Elem = if ratio > Rat::ZERO { value.clone() } else { k.neg(value) };
            let g: DensePoly<U> = d.gcd(
                &r.mul(r)?
                    .sub(&dd.pow(2).mul(q)?.mul_ground(&k.mul(&residue, &residue)))?,
            )?;
            if g.degree() == 0 {
                return Ok(None);
            }
            let w: DensePoly<U> = r.mul(&inverse(&dd.mul_ground(&residue), &g)?)?.rem(&g)?;
            finite.push((qq, g, w));
        }
        let mut q_infinity: Rat = Rat::ZERO;
        if let Some(v) = infinity {
            if let Some(ratio) = l.expr(&k.exquo(v, representative)?)?.as_rational() {
                q_infinity = ratio.clone();
            }
        }
        out.push(ResidueClass {
            rho,
            finite,
            q_infinity,
        });
    }
    Ok(Some(out))
}

/// `W` lifted from `W**2 = Q mod G` to a root modulo `G**N` by Newton's
/// iteration (`_lift`).
fn lift<D: Domain>(w: &DensePoly<D>, g: &DensePoly<D>, q: &DensePoly<D>, n: usize) -> PolyResult<DensePoly<D>> {
    let modulus: DensePoly<D> = g.pow(n);
    let two: D::Elem = ground(&q.domain, 2);
    let bits: usize = (usize::BITS - (n.saturating_sub(1)).leading_zeros()) as usize;
    let steps: usize = (bits + 1).max(1);
    let mut w: DensePoly<D> = w.clone();
    for _ in 0..steps {
        crate::timeout::check()?;
        w = w
            .add(&q.mul(&inverse(&w, &modulus)?)?)?
            .exquo_ground(&two)?
            .rem(&modulus)?;
    }
    Ok(w)
}

/// The coefficients of `sqrt(Q/lc(Q))` in `1/x` up to `order`
/// (`_series_at_infinity`).
fn series_at_infinity<D: Domain>(q: &DensePoly<D>, order: usize) -> PolyResult<Vec<D::Elem>> {
    let k: &D = &q.domain;
    let mut qs: Vec<D::Elem> = q.monic()?.rep;
    while qs.len() < order + 1 {
        qs.push(k.zero());
    }
    let mut sigma: Vec<D::Elem> = vec![k.one()];
    let two: D::Elem = ground(k, 2);
    for j in 1..=order {
        let mut total: D::Elem = if j < qs.len() { qs[j].clone() } else { k.zero() };
        for i in 1..j {
            total = k.sub(&total, &k.mul(&sigma[i], &sigma[j - i]));
        }
        sigma.push(k.exquo(&total, &two)?);
    }
    Ok(sigma)
}

/// The coefficients from the constant term up, padded to `size`
/// (`_coefficients`).
fn coefficients<D: Domain>(p: &DensePoly<D>, size: usize) -> Vec<D::Elem> {
    let mut out: Vec<D::Elem> = p.rep.iter().rev().cloned().collect();
    while out.len() < size {
        out.push(p.domain.zero());
    }
    out
}

fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// A function `u + v w` with the divisor `n Delta` of the class
/// (`_function_with_divisor`), `None` when there is none of the degrees
/// bounded by the divisor.
fn function_with_divisor<U: Domain>(
    cls: &ResidueClass<U>,
    n: i64,
    q: &DensePoly<U>,
    s: Option<&U::Elem>,
    x: &Symbol,
) -> PolyResult<Option<(DensePoly<U>, DensePoly<U>)>> {
    let k: &U = &q.domain;
    let deg_q: i64 = q.degree() as i64;
    let m: i64 = deg_q / 2;
    let numer = |r: &Rat| -> i64 { i64::try_from(crate::numbers::numer(r)).unwrap_or(i64::MAX) };
    let denom = |r: &Rat| -> i64 { i64::try_from(crate::numbers::denom(r)).unwrap_or(1) };
    let mut sigma: i64 = 0;
    for (qq, g, _) in &cls.finite {
        sigma += floor_div(numer(qq) * n, denom(qq)) * g.degree() as i64;
    }
    // Python's int() of a fraction: towards zero
    let q_inf: i64 = (numer(&cls.q_infinity) * n) / denom(&cls.q_infinity);
    let (du, dv, e_plus, e_minus): (i64, i64, i64, i64) = if deg_q % 2 == 1 {
        (sigma, sigma - m - 1, sigma, sigma)
    } else {
        let (e_plus, e_minus) = (sigma - q_inf, sigma + q_inf);
        let e: i64 = e_plus.max(e_minus);
        (e, e - m, e_plus, e_minus)
    };
    if du < 0 {
        return Ok(None);
    }
    let (nu, nv): (usize, usize) = ((du + 1) as usize, (dv + 1).max(0) as usize);
    let unknowns: usize = nu + nv;
    let mut rows: Vec<Vec<U::Elem>> = Vec::new();
    for (qq, g, w) in &cls.finite {
        let big_n: usize = floor_div(2 * numer(qq) * n, denom(qq)) as usize;
        let modulus: DensePoly<U> = g.pow(big_n);
        let w_n: DensePoly<U> = lift(w, g, q, big_n)?;
        let size: usize = modulus.degree() as usize;
        let mut columns: Vec<Vec<U::Elem>> = Vec::new();
        for i in 0..nu {
            columns.push(coefficients(&monomial(k.one(), i, k, x).rem(&modulus)?, size));
        }
        for i in 0..nv {
            columns.push(coefficients(
                &monomial(k.one(), i, k, x).mul(&w_n)?.rem(&modulus)?,
                size,
            ));
        }
        for j in 0..size {
            rows.push(columns.iter().map(|c| c[j].clone()).collect());
        }
    }
    if deg_q % 2 == 0 && q_inf != 0 {
        let Some(s) = s else {
            return Ok(None);
        };
        let order: i64 = (nv as i64 - 1 + m - e_plus.min(e_minus) - 1).max(0);
        let sigma_series: Vec<U::Elem> = series_at_infinity(q, order as usize)?;
        for (sign, bound) in [(k.one(), e_plus), (k.neg(&k.one()), e_minus)] {
            for j in bound + 1..=du {
                let mut row: Vec<U::Elem> = (0..nu as i64)
                    .map(|i| if i == j { k.one() } else { k.zero() })
                    .collect();
                for kk in 0..nv as i64 {
                    let index: i64 = kk + m - j;
                    let value: U::Elem = if index >= 0 && (index as usize) < sigma_series.len() {
                        sigma_series[index as usize].clone()
                    } else {
                        k.zero()
                    };
                    row.push(k.mul(&k.mul(&sign, s), &value));
                }
                rows.push(row);
            }
        }
    }
    let vector: Vec<U::Elem> = if rows.is_empty() {
        let mut v: Vec<U::Elem> = vec![k.one()];
        v.extend(std::iter::repeat(k.zero()).take(unknowns - 1));
        v
    } else {
        let space: Vec<Vec<U::Elem>> = k.nullspace(&rows, unknowns)?;
        if space.is_empty() {
            return Ok(None);
        }
        space.into_iter().next().expect("a vector")
    };
    let u_rep: Vec<U::Elem> = vector[..nu].iter().rev().cloned().collect();
    let v_rep: Vec<U::Elem> = vector[nu..nu + nv].iter().rev().cloned().collect();
    let u: DensePoly<U> = DensePoly::new(&u_rep, k.clone(), x.clone());
    let v: DensePoly<U> = DensePoly::new(&v_rep, k.clone(), x.clone());
    if u.is_zero() && v.is_zero() {
        return Ok(None);
    }
    let norm: DensePoly<U> = u.mul(&u)?.sub(&v.mul(&v)?.mul(q)?)?;
    let mut expected: DensePoly<U> = one_like(q);
    for (qq, g, _) in &cls.finite {
        expected = expected.mul(&g.pow(floor_div(2 * numer(qq) * n, denom(qq)) as usize))?;
    }
    if norm.is_zero() || norm.degree() != expected.degree() || !norm.sub(&expected.mul_ground(&norm.LC()))?.is_zero() {
        return Ok(None);
    }
    Ok(Some((u, v)))
}

// ---------------------------------------------------------------------
// the elements a0 + a1 w of the function field

/// `a0 + a1 w` with `w**2 = Q` (`_Pair`).
#[derive(Debug, Clone)]
pub struct Pair {
    pub a0: Basic,
    pub a1: Basic,
    pub q: Basic,
    pub x: Basic,
}

impl Pair {
    pub fn new(a0: Basic, a1: Basic, q: &Basic, x: &Basic) -> Pair {
        Pair {
            a0,
            a1,
            q: q.clone(),
            x: x.clone(),
        }
    }

    pub fn mul(&self, other: &Pair) -> PolyResult<Pair> {
        let a0: Basic = cancel(&add(&[
            mul(&[self.a0.clone(), other.a0.clone()])?,
            mul(&[self.a1.clone(), other.a1.clone(), self.q.clone()])?,
        ])?)?;
        let a1: Basic = cancel(&add(&[
            mul(&[self.a0.clone(), other.a1.clone()])?,
            mul(&[self.a1.clone(), other.a0.clone()])?,
        ])?)?;
        Ok(Pair::new(a0, a1, &self.q, &self.x))
    }

    pub fn sub(&self, other: &Pair) -> PolyResult<Pair> {
        Ok(Pair::new(
            cancel(&sub(&self.a0, &other.a0)?)?,
            cancel(&sub(&self.a1, &other.a1)?)?,
            &self.q,
            &self.x,
        ))
    }

    pub fn scaled(&self, c: &Basic) -> PolyResult<Pair> {
        Ok(Pair::new(
            cancel(&mul(&[c.clone(), self.a0.clone()])?)?,
            cancel(&mul(&[c.clone(), self.a1.clone()])?)?,
            &self.q,
            &self.x,
        ))
    }

    pub fn derivative(&self) -> PolyResult<Pair> {
        let a1: Basic = cancel(&add(&[
            diff(&self.a1, &self.x)?,
            div(
                &mul(&[self.a1.clone(), diff(&self.q, &self.x)?])?,
                &mul(&[int(2), self.q.clone()])?,
            )?,
        ])?)?;
        Ok(Pair::new(cancel(&diff(&self.a0, &self.x)?)?, a1, &self.q, &self.x))
    }

    pub fn inverse(&self) -> PolyResult<Pair> {
        let norm: Basic = cancel(&sub(
            &pw(&self.a0, &int(2))?,
            &mul(&[pw(&self.a1, &int(2))?, self.q.clone()])?,
        )?)?;
        Ok(Pair::new(
            cancel(&div(&self.a0, &norm)?)?,
            cancel(&div(&neg(&self.a1)?, &norm)?)?,
            &self.q,
            &self.x,
        ))
    }

    pub fn logarithmic_derivative(&self) -> PolyResult<Pair> {
        self.derivative()?.mul(&self.inverse()?)
    }

    pub fn is_zero(&self) -> bool {
        self.a0 == Basic::zero() && self.a1 == Basic::zero()
    }
}

/// What the integration of `A + B sqrt(P)` found (`_Result`): the
/// rational part, the terms `S w/T` of the Hermite reduction, the
/// logarithms, the remainder not integrated, and whether it is proved not
/// elementary (`None` when undecided).
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub p: Basic,
    pub d: Basic,
    pub q: Basic,
    pub rational_part: Basic,
    pub rational_logs: Basic,
    pub terms: Vec<(Basic, Basic)>,
    pub logs: Vec<Basic>,
    pub remainder: Pair,
    pub nonelementary: Option<bool>,
    pub generators: Vec<Basic>,
}

impl IntegrationResult {
    pub fn w(&self) -> R {
        div(&sqrt(&self.p)?, &self.d)
    }

    pub fn exact_part(&self) -> R {
        let w: Basic = self.w()?;
        let mut parts: Vec<Basic> = vec![self.rational_part.clone()];
        for (s, t) in &self.terms {
            parts.push(div(&mul(&[s.clone(), w.clone()])?, t)?);
        }
        add(&parts)
    }

    pub fn elementary(&self) -> R {
        let mut parts: Vec<Basic> = vec![self.exact_part()?, self.rational_logs.clone()];
        parts.extend(self.logs.iter().cloned());
        add(&parts)
    }

    pub fn remainder_expr(&self) -> R {
        add(&[
            self.remainder.a0.clone(),
            div(
                &mul(&[
                    cancel(&mul(&[self.remainder.a1.clone(), self.q.clone()])?)?,
                    self.d.clone(),
                ])?,
                &sqrt(&self.p)?,
            )?,
        ])
    }
}

fn ratint_expr(f: &Basic, x: &Basic) -> R {
    let symbol: Basic = Symbol::new("t", &[]).0;
    let integrand = crate::integrals::risch::rationaltools::Integrand::of(f)?;
    crate::integrals::risch::rationaltools::ratint(&integrand, x, &symbol, None)
}

/// The integral of `A + B sqrt(P)`, `None` when the integrand is not one
/// (`_integrate`).
pub fn integrate(f: &Basic, x: &Basic, logarithms: bool) -> PolyResult<Option<IntegrationResult>> {
    let Some(parsed) = parse(f, x)? else {
        return Ok(None);
    };
    let (an, ad) = fraction(&parsed.a)?;
    let (bn, bd) = fraction(&parsed.b)?;
    let exprs: Vec<Basic> = vec![parsed.base.clone(), an, ad, bn, bd];
    let (parameters, is_gaussian, constants) = constant_kind(&exprs, x)?;
    let result: IntegrationResult = if constants {
        integrate_over(&expressions(), &parsed, x, logarithms)?
    } else if is_gaussian {
        let g: NumberField = gaussian()?;
        if !parameters.is_empty() {
            integrate_over(&algebraic_parametric(&g, &parameters), &parsed, x, logarithms)?
        } else {
            integrate_over(&algebraic(&g), &parsed, x, logarithms)?
        }
    } else if !parameters.is_empty() {
        integrate_over(&parametric(&parameters), &parsed, x, logarithms)?
    } else {
        integrate_over(&rationals(), &parsed, x, logarithms)?
    };
    Ok(Some(result))
}

/// The integration over the field of the constants (`_integrate_over`).
fn integrate_over<D: Domain>(
    f: &Field<D>,
    parsed: &Parsed,
    x: &Basic,
    logarithms: bool,
) -> PolyResult<IntegrationResult> {
    let xs: Symbol = Symbol(x.clone());
    let radical: Radical<D> = Radical::new(f, x, f.from_expr(&parsed.base, x)?, &parsed.a, &parsed.b)?;
    let q: &DensePoly<D> = &radical.q;
    let a: &Basic = &radical.a;
    let (numerator, denominator) = fraction(&cancel(a)?)?;
    let (rational_part, rational_logs): (Basic, Basic) = if *a == Basic::zero() {
        (Basic::zero(), Basic::zero())
    } else if logarithms {
        (ratint_expr(a, x)?, Basic::zero())
    } else {
        let (p, qq) = (f.from_expr(&numerator, x)?, f.from_expr(&denominator, x)?);
        let (quotient, rest) = p.div(&qq)?;
        let (part, logs) =
            crate::integrals::risch::rationaltools::ratint_ratpart(&f.poly_expr(&rest)?, &f.poly_expr(&qq)?, x)?;
        (add(&[part, f.poly_expr(&quotient.integrate(1)?)?])?, logs)
    };
    if q.degree() == 0 {
        let constant: Basic = sqrt(&radical.q_expr)?;
        let g: Basic = mul(&[radical.b.clone(), radical.d_expr.clone(), constant])?;
        let extra: Basic = if logarithms { ratint_expr(&g, x)? } else { Basic::zero() };
        let orientation: Basic = if radical.d.degree() >= 1 {
            crate::functions::elementary::complexes::sign(&radical.d_expr)?
        } else {
            Basic::one()
        };
        let a0: Basic = if logarithms {
            Basic::zero()
        } else {
            mul(&[g, orientation.clone()])?
        };
        let remainder: Pair = Pair::new(a0, Basic::zero(), &radical.q_expr, x);
        return Ok(IntegrationResult {
            p: radical.p_expr.clone(),
            d: radical.d_expr.clone(),
            q: radical.q_expr.clone(),
            rational_part: add(&[rational_part, mul(&[orientation, extra])?])?,
            rational_logs,
            terms: Vec::new(),
            logs: Vec::new(),
            remainder,
            nonelementary: Some(false),
            generators: Vec::new(),
        });
    }
    let (m, d) = radical.algebraic_numerator()?;
    let reduction: Reduction<D> = hermite(&m, &d, q, &xs)?;
    let mut terms: Vec<(Basic, Basic)> = Vec::new();
    for (s, t) in &reduction.terms {
        terms.push((f.poly_expr(s)?, f.poly_expr(t)?));
    }
    let q_expr: Basic = radical.q_expr.clone();
    let remainder: Pair = Pair::new(
        Basic::zero(),
        cancel(&div(
            &f.poly_expr(&reduction.r)?,
            &mul(&[f.poly_expr(&reduction.d)?, q_expr.clone()])?,
        )?)?,
        &q_expr,
        x,
    );
    let half: isize = q.degree() / 2;
    let done = |terms: Vec<(Basic, Basic)>,
                logs: Vec<Basic>,
                remainder: Pair,
                verdict: Option<bool>,
                generators: Vec<Basic>| {
        IntegrationResult {
            p: radical.p_expr.clone(),
            d: radical.d_expr.clone(),
            q: q_expr.clone(),
            rational_part: rational_part.clone(),
            rational_logs: rational_logs.clone(),
            terms,
            logs,
            remainder,
            nonelementary: verdict,
            generators,
        }
    };
    if reduction.polynomial.degree() >= half && half >= 1 {
        return Ok(done(terms, Vec::new(), remainder, Some(true), Vec::new()));
    }
    if !logarithms || reduction.r.is_zero() {
        let verdict: Option<bool> = if reduction.r.is_zero() { Some(false) } else { None };
        return Ok(done(terms, Vec::new(), remainder, verdict, Vec::new()));
    }
    let (logs, remainder, verdict, generators) = logarithmic_part(f, &reduction, q, x, remainder, &radical.w()?)?;
    Ok(done(terms, logs, remainder, verdict, generators))
}

/// The logarithms, the remainder, the verdict and the generators of the
/// residues when undecided (`_Logs`).
pub type Logs = (Vec<Basic>, Pair, Option<bool>, Vec<Basic>);

/// The logarithmic part (`_logarithmic_part`): the residues and the field
/// they generate, then [`logarithms_in`].
fn logarithmic_part<D: Domain>(
    f: &Field<D>,
    reduction: &Reduction<D>,
    q: &DensePoly<D>,
    x: &Basic,
    remainder: Pair,
    w: &Basic,
) -> PolyResult<Logs> {
    let (r, d) = (&reduction.r, &reduction.d);
    let z: Basic = fresh("z", false, false);
    let big_r: DensePoly<D> = residue_polynomial(r, d, q, &z)?;
    let Some(values) = residue_values(f, &big_r, &z)? else {
        return Ok((Vec::new(), remainder, None, Vec::new()));
    };
    let m: isize = q.degree() / 2;
    let mut infinity: Option<Basic> = None;
    let mut s_value: Option<Basic> = None;
    if q.degree() % 2 == 0 {
        let s: Basic = denested(&sqrt(&f.expr(&q.LC())?)?)?;
        if reduction.polynomial.degree() == m - 1 {
            infinity = Some(div(&neg(&f.expr(&reduction.polynomial.LC())?)?, &s)?);
        }
        s_value = Some(s);
    }
    if values.is_empty() && infinity.is_none() {
        let verdict: bool = !remainder.is_zero();
        return Ok((Vec::new(), remainder, Some(verdict), Vec::new()));
    }
    let mut generators: Vec<Basic> = values.clone();
    if let Some(v) = &infinity {
        generators.push(v.clone());
    }
    if let Some(v) = &s_value {
        generators.push(v.clone());
    }
    let undecided: Logs = (
        Vec::new(),
        remainder.clone(),
        None,
        generators
            .iter()
            .filter(|v| !v.free_symbols().is_empty())
            .cloned()
            .collect(),
    );
    let numbers: Vec<Basic> = algebraic_atoms(&generators);
    let parameters: &[Basic] = &f.parameters;
    let input = LogInput {
        values: &values,
        infinity: infinity.as_ref(),
        s_value: s_value.as_ref(),
    };
    if numbers.is_empty() {
        if !parameters.is_empty() {
            return logarithms_in(
                &parametric(parameters),
                f,
                reduction,
                q,
                x,
                remainder,
                w,
                &input,
                undecided,
            );
        }
        return logarithms_in(&rationals(), f, reduction, q, x, remainder, w, &input, undecided);
    }
    let n: NumberField = match NumberField::new(&numbers, MAX_FIELD_DEGREE) {
        Ok(n) => n,
        Err(PolyError::Arithmetic(_)) => return Ok(undecided),
        Err(e) => return Err(e),
    };
    if !parameters.is_empty() {
        return logarithms_in(
            &algebraic_parametric(&n, parameters),
            f,
            reduction,
            q,
            x,
            remainder,
            w,
            &input,
            undecided,
        );
    }
    logarithms_in(&algebraic(&n), f, reduction, q, x, remainder, w, &input, undecided)
}

struct LogInput<'a> {
    values: &'a [Basic],
    infinity: Option<&'a Basic>,
    s_value: Option<&'a Basic>,
}

/// The logarithms over the field `L` of the residues (`_logarithms_in`).
fn logarithms_in<U: Domain, D: Domain>(
    l: &Field<U>,
    f: &Field<D>,
    reduction: &Reduction<D>,
    q0: &DensePoly<D>,
    x: &Basic,
    mut remainder: Pair,
    w: &Basic,
    input: &LogInput,
    undecided: Logs,
) -> PolyResult<Logs> {
    let converted = (|| -> PolyResult<(Vec<U::Elem>, Option<U::Elem>, Option<U::Elem>)> {
        let elements: Vec<U::Elem> = input.values.iter().map(|v| l.element(v)).collect::<PolyResult<_>>()?;
        let infinity: Option<U::Elem> = input.infinity.map(|v| l.element(v)).transpose()?;
        let s: Option<U::Elem> = input.s_value.map(|v| l.element(v)).transpose()?;
        Ok((elements, infinity, s))
    })();
    let (elements, infinity, s) = match caught(converted)? {
        Some(v) => v,
        None => return Ok(undecided),
    };
    let k: &U = &l.domain;
    let xs: Symbol = Symbol(x.clone());
    let (r, d, q) = (l.poly(&reduction.r)?, l.poly(&reduction.d)?, l.poly(q0)?);
    let Some(found_classes) = classes(&elements, infinity.as_ref(), l, &r, &d, &q)? else {
        return Ok((Vec::new(), remainder, None, Vec::new()));
    };
    let genus_one: bool = matches!(q.degree(), 3 | 4);
    let mut logs: Vec<Basic> = Vec::new();
    let mut undecided_: bool = false;
    let q_expr: Basic = f.poly_expr(q0)?;
    for cls in &found_classes {
        let n0: i64 = i64::try_from(cls.denominator()).unwrap_or(i64::MAX);
        let mut n: i64 = n0;
        let mut found: Option<(DensePoly<U>, DensePoly<U>)> = None;
        for j in 1..=TORSION_BOUND {
            n = j * n0;
            found = function_with_divisor(cls, n, &q, s.as_ref(), &xs)?;
            if found.is_some() {
                break;
            }
        }
        let Some((mut u, mut v)) = found else {
            if !(genus_one && l.rationals) {
                undecided_ = true;
            }
            continue;
        };
        let mut rho: Basic = cls.rho.clone();
        if !v.is_zero() && crate::core::expr::is(&l.expr(&k.mul(&u.LC(), &v.LC()))?, Predicate::negative) == Some(true)
        {
            v = v.neg();
            rho = neg(&rho)?;
        }
        if crate::core::expr::is(&l.expr(&u.LC())?, Predicate::negative) == Some(true) {
            u = u.neg();
            v = v.neg();
        }
        let mut denominator: DensePoly<U> = one_like(&q);
        for (qq, g, _) in &cls.finite {
            let e: i64 = (i64::try_from(crate::numbers::numer(qq)).unwrap_or(0) * n)
                .div_euclid(i64::try_from(crate::numbers::denom(qq)).unwrap_or(1));
            denominator = denominator.mul(&g.pow(e as usize))?;
        }
        let (u_expr, v_expr, d_expr) = (l.poly_expr(&u)?, l.poly_expr(&v)?, l.poly_expr(&denominator)?);
        let scale: Basic = div(&rho, &int(n))?;
        logs.push(mul(&[
            scale.clone(),
            crate::functions::elementary::exponential::log(&div(
                &add(&[u_expr.clone(), mul(&[v_expr.clone(), w.clone()])?])?,
                &d_expr,
            )?)?,
        ])?);
        let pair: Pair = Pair::new(div(&u_expr, &d_expr)?, div(&v_expr, &d_expr)?, &q_expr, x);
        remainder = remainder.sub(&pair.logarithmic_derivative()?.scaled(&scale)?)?;
    }
    if remainder.is_zero() {
        return Ok((logs, remainder, Some(false), Vec::new()));
    }
    let verdict: Option<bool> = if undecided_ { None } else { Some(true) };
    Ok((logs, remainder, verdict, Vec::new()))
}

// ---------------------------------------------------------------------
// the verification and the public functions

fn settings_timeout() -> Option<f64> {
    crate::settings::settings().timeout
}

/// Whether `e` is positive at `values` (`_positive_at`).
fn positive_at(e: &Basic, values: &[(Basic, Basic)]) -> PolyResult<bool> {
    let v: Basic = replaced(e, values)?;
    if let Some(q) = v.as_rational() {
        return Ok(*q > Rat::ZERO);
    }
    if !v.free_symbols().is_empty() {
        return Ok(crate::core::expr::is(&v, Predicate::positive) == Some(true));
    }
    let Some(found) = crate::numeric::reliable_value(&v, 30, &[]) else {
        return Ok(false);
    };
    let (re, im) = crate::core::evalf::parts(&found);
    Ok(im == crate::core::evalf::Real::ZERO && re > crate::core::evalf::Real::ZERO)
}

fn without_deltas(e: &Basic) -> R {
    crate::core::traversal::bottom_up(
        e,
        &|node: &Basic| {
            let delta: bool = matches!(node.head(), Head::Function(crate::core::basic::Function::DiracDelta));
            let sign_derivative: bool = matches!(node.head(), Head::Derivative)
                && matches!(
                    node.args()[0].head(),
                    Head::Function(crate::core::basic::Function::Sign)
                );
            Ok(if delta || sign_derivative {
                Basic::zero()
            } else {
                node.clone()
            })
        },
        false,
    )
}

/// Whether `F' = f` numerically at random points where the radicands are
/// positive (`_verified`).
pub fn verified(big_f: &Basic, f: &Basic, x: &Basic, p: &Basic) -> PolyResult<bool> {
    use crate::core::evalf::{parts, real_rat, Real};
    let derivative: Basic = without_deltas(&diff(big_f, x)?)?;
    let mut rng = crate::random::PyRandom::from_string(&f.to_string());
    let parameters: Vec<Basic> = symbols(f).into_iter().filter(|s| s != x).collect();
    let mut radicands: Vec<Basic> = vec![p.clone()];
    for node in pows(f) {
        if node.has(x) && half_integer(&node) {
            radicands.push(node.args()[0].clone());
        }
    }
    let mut checked: usize = 0;
    let tolerance: Real = real_rat(&(Rat::ONE / Rat::from(Int::from(10u8).pow(20))), 140);
    for _ in 0..40 {
        crate::timeout::check()?;
        let mut values: Vec<(Basic, Basic)> = Vec::new();
        for s in &parameters {
            values.push((
                s.clone(),
                Basic::from_rational(Rat::from_parts(
                    Int::from(rng.randint(-400, 400)),
                    dashu_int::UBig::from(100u8),
                )),
            ));
        }
        values.push((
            x.clone(),
            Basic::from_rational(Rat::from_parts(
                Int::from(rng.randint(-400, 400)),
                dashu_int::UBig::from(100u8),
            )),
        ));
        let step = (|| -> PolyResult<Option<(crate::core::evalf::Complex, crate::core::evalf::Complex)>> {
            for radicand in &radicands {
                if !positive_at(radicand, &values)? {
                    return Ok(None);
                }
            }
            let expected = crate::numeric::reliable_value(f, 30, &values);
            let found = crate::numeric::reliable_value(&derivative, 30, &values);
            Ok(match (expected, found) {
                (Some(e), Some(g)) => Some((e, g)),
                _ => None,
            })
        })();
        let Some(Some((e, g))) = caught(step)? else {
            continue;
        };
        let (er, ei) = parts(&e);
        let (gr, gi) = parts(&g);
        let (dr, di) = (&gr - &er, &gi - &ei);
        let magnitude: Real = {
            let m2: Real = &(&er * &er) + &(&ei * &ei);
            m2.sqrt()
        };
        let bound: Real = &tolerance * &(&Real::ONE + &magnitude);
        if &(&dr * &dr) + &(&di * &di) > &bound * &bound {
            return Ok(false);
        }
        checked += 1;
        if checked >= 4 {
            return Ok(true);
        }
    }
    Ok(checked > 0)
}

/// `(F, h)` with `f = F' + h`, `F` elementary and `h` the part not
/// integrated, verified (`trager_reduce`); `None` when the integrand is
/// none of those handled.
pub fn trager_reduce(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let (mut g, mut rational_roots) = polynomial_radicands(f, x)?;
    let first: Option<IntegrationResult> =
        crate::timeout::attempt_within(settings_timeout(), || integrate(&g, x, true))?.flatten();
    let Some(mut result) = first else {
        let found = crate::timeout::attempt_within(settings_timeout(), || index_n(f, x))?.flatten();
        if found.is_some() {
            return Ok(found);
        }
        return Ok(crate::timeout::attempt_within(settings_timeout(), || {
            crate::integrals::curves::curve_reduce(f, x)
        })?
        .flatten());
    };
    let mut back: Vec<(Basic, Basic)> = Vec::new();
    let original: Basic = result.p.clone();
    if result.nonelementary.is_none() && !result.generators.is_empty() {
        let params: Vec<Basic> = symbols(&g).into_iter().filter(|s| s != x).collect();

        if let Some((substitution, back_)) = rational_parameters(&result.generators, &params)? {
            let g_: Basic = replaced(&g, &substitution)?;
            let again = crate::timeout::attempt_within(settings_timeout(), || integrate(&g_, x, true))?.flatten();
            if let Some(again) = again {
                if again.nonelementary.is_some() {
                    result = again;
                    g = g_;
                    let mut roots_: Vec<(Basic, Basic)> = Vec::new();
                    for (product, y) in &rational_roots {
                        roots_.push((expand(&replaced(product, &substitution)?)?, replaced(y, &substitution)?));
                    }
                    rational_roots = roots_;
                    back = back_;
                }
            }
        }
    }
    let mut p: Basic = result.p.clone();
    let mut big_f: Basic = restored(&original_roots(&result.elementary()?, &g, x, &p)?, &rational_roots)?;
    let mut h: Basic = restored(&original_roots(&result.remainder_expr()?, &g, x, &p)?, &rational_roots)?;
    if !back.is_empty() {
        let p_: Basic = replaced(&p, &back)?;
        big_f = replaced(&big_f, &back)?;
        h = replaced(&h, &back)?;
        let pairs: Vec<(Basic, Basic)> = vec![(expand(&p_)?, sqrt(&original)?)];
        big_f = restored(&big_f, &pairs)?;
        h = restored(&h, &pairs)?;
        p = original;
    }
    if !verified(&big_f, &sub(f, &h)?, x, &p)? {
        return Ok(None);
    }
    Ok(Some((big_f, h)))
}

/// An elementary antiderivative, `None` when none is found
/// (`trager_antiderivative`).
pub fn trager_antiderivative(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let Some((big_f, h)) = trager_reduce(f, x)? else {
        return Ok(None);
    };
    if h != Basic::zero() {
        return Ok(None);
    }
    Ok(Some(big_f))
}

/// The Hermite reduction `(g, h)` with `f = g' + h`, `h` with simple poles
/// only, verified (`hermite_reduce_algebraic`).
pub fn hermite_reduce_algebraic(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some(Some(result)) = crate::timeout::attempt_within(settings_timeout(), || integrate(f, x, false))? else {
        return Ok(None);
    };
    let g: Basic = result.exact_part()?;
    let h: Basic = add(&[result.rational_logs.clone(), result.remainder_expr()?])?;
    if !verified(&g, &sub(f, &h)?, x, &result.p)? {
        return Ok(None);
    }
    Ok(Some((g, h)))
}

/// Whether the integral is proved not elementary (`None` when undecided)
/// (`is_nonelementary_algebraic`).
pub fn is_nonelementary_algebraic(f: &Basic, x: &Basic) -> PolyResult<Option<bool>> {
    if let Some(binomial) = crate::integrals::algebraic::binomial_exponents(f, x)? {
        if symbols(f).iter().all(|s| s == x) {
            return Ok(Some(!crate::integrals::algebraic::chebyshev_elementary(&binomial)));
        }
    }
    let Some(result) = crate::timeout::attempt_within(settings_timeout(), || integrate(f, x, true))?.flatten() else {
        let verdict = crate::timeout::attempt_within(settings_timeout(), || {
            crate::integrals::curves::is_nonelementary_on_curve(f, x)
        })?;
        return Ok(verdict.flatten());
    };
    if result.nonelementary == Some(false) {
        return Ok(if trager_antiderivative(f, x)?.is_some() {
            Some(false)
        } else {
            None
        });
    }
    Ok(result.nonelementary)
}

/// The square roots of rational functions made roots of polynomials
/// (`_polynomial_radicands`): `sqrt(N/D)**k` as `sqrt(N D)**k/D**k`, with
/// the pairs to restore them.
fn polynomial_radicands(f: &Basic, x: &Basic) -> PolyResult<(Basic, Vec<(Basic, Basic)>)> {
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    let mut pairs: Vec<(Basic, Basic)> = Vec::new();
    let xs: [Basic; 1] = [x.clone()];
    for node in pows(f) {
        if !(half_integer(&node) && node.has(x)) {
            continue;
        }
        let base: &Basic = &node.args()[0];
        if is_polynomial(base, &xs) || !is_rational_function(base, &xs) {
            continue;
        }
        let (numerator, denominator) = fraction(&cancel(base)?)?;
        if !(is_polynomial(&numerator, &xs) && is_polynomial(&denominator, &xs) && denominator.has(x)) {
            continue;
        }
        let k: i64 = i64::try_from(crate::numbers::numer(&exponent(&node)?)).unwrap_or(1);
        let product: Basic = expand(&mul(&[numerator, denominator.clone()])?)?;
        replacements.push((
            node.clone(),
            div(&pw(&product, &node.args()[1])?, &pw(&denominator, &int(k))?)?,
        ));
        pairs.push((product, mul(&[denominator, sqrt(base)?])?));
    }
    Ok((replaced(f, &replacements)?, pairs))
}

/// The roots of the products back to the original radicals (`_restored`).
fn restored(big_f: &Basic, pairs: &[(Basic, Basic)]) -> R {
    if pairs.is_empty() {
        return Ok(big_f.clone());
    }
    let restore = |node: &Basic| -> R {
        if is_pow(node) && half_integer(node) {
            let base: Basic = expand(&node.args()[0])?;
            for (product, y) in pairs {
                if base == *product {
                    let k: i64 = i64::try_from(crate::numbers::numer(&exponent(node)?)).unwrap_or(1);
                    let m: i64 = k.rem_euclid(2);
                    return mul(&[pw(&node.args()[0], &int((k - m) / 2))?, pw(y, &int(m))?]);
                }
            }
        }
        Ok(node.clone())
    };
    crate::core::traversal::bottom_up(big_f, &restore, false)
}

/// Rational parameters which make the residues rational: a square root
/// of an expression linear in a symbol `p` as `delta` with
/// `p = (delta**2 - offset)/slope` (`_rational_parameters`).
fn rational_parameters(
    values: &[Basic],
    parameters: &[Basic],
) -> PolyResult<Option<(Vec<(Basic, Basic)>, Vec<(Basic, Basic)>)>> {
    let mut substitution: Vec<(Basic, Basic)> = Vec::new();
    let mut back: Vec<(Basic, Basic)> = Vec::new();
    let mut current: Vec<Basic> = values.to_vec();
    let symbolic_half = |n: &Basic| half_integer(n) && !n.free_symbols().is_empty();
    for _ in 0..8 {
        let mut nodes: Vec<Basic> = Vec::new();
        for value in &current {
            for node in pows(value) {
                if symbolic_half(&node) {
                    nodes.push(node);
                }
            }
        }
        let mut inner: Vec<Basic> = nodes
            .iter()
            .filter(|node| !pows(&node.args()[0]).iter().any(&symbolic_half))
            .cloned()
            .collect();
        if inner.is_empty() {
            break;
        }
        inner.sort_by_key(|n| n.to_string());
        let node: Basic = inner.swap_remove(0);
        let (numerator, denominator) = fraction(&cancel(&node.args()[0])?)?;
        let nd: Basic = mul(&[numerator, denominator.clone()])?;
        let syms: Vec<Basic> = symbols(&nd);
        let Some((content, factors)) = caught(crate::polys::polytools::factor_list(&nd, &syms))? else {
            return Ok(None);
        };
        let mut e0: Vec<Basic> = Vec::new();
        let mut square: Vec<Basic> = Vec::new();
        for (factor, multiplicity) in factors {
            let Some(m) = multiplicity
                .as_rational()
                .filter(|q| q.is_int())
                .map(crate::numbers::numer)
            else {
                return Ok(None);
            };
            let m: i64 = i64::try_from(&m).unwrap_or(0);
            e0.push(pw(&factor, &int(m.rem_euclid(2)))?);
            square.push(pw(&factor, &int(m.div_euclid(2)))?);
        }
        let e0: Basic = mul(&e0)?;
        let square: Basic = mul(&square)?;
        let mut candidates: Vec<Basic> = symbols(&e0);
        candidates.sort_by_key(|p| (!parameters.contains(p), p.to_string()));
        let mut chosen: Option<Basic> = None;
        let mut linear: Vec<(i64, Basic)> = Vec::new();
        for p in &candidates {
            let Ok(l) = powers_of(&e0, p) else { continue };
            if power_degree(&l) == 1 {
                linear = l;
                chosen = Some(p.clone());
                break;
            }
        }
        let Some(chosen) = chosen else {
            return Ok(None);
        };
        let (slope, offset) = (power_get(&linear, 1), power_get(&linear, 0));
        let delta: Basic = fresh("delta", false, true);
        let value: Basic = div(&sub(&pw(&delta, &int(2))?, &offset)?, &slope)?;
        back.push((delta.clone(), replaced(&sqrt(&e0)?, &back)?));
        let rule: Vec<(Basic, Basic)> = vec![(chosen.clone(), value.clone())];
        let mut new_substitution: Vec<(Basic, Basic)> = Vec::new();
        for (q, e) in &substitution {
            new_substitution.push((q.clone(), replaced(e, &rule)?));
        }
        substitution = new_substitution;
        if parameters.contains(&chosen) {
            substitution.push((chosen.clone(), value.clone()));
        }
        let root: Basic = div(&mul(&[sqrt(&content)?, square, delta])?, &denominator)?;
        let mut replacement: Vec<(Basic, Basic)> = Vec::new();
        for other in &nodes {
            if other.args()[0] == node.args()[0] && !replacement.iter().any(|(k, _)| k == other) {
                let k: i64 = i64::try_from(crate::numbers::numer(&exponent(other)?)).unwrap_or(1);
                replacement.push((other.clone(), pw(&root, &int(k))?));
            }
        }
        let mut next: Vec<Basic> = Vec::new();
        for c in &current {
            next.push(cancel(&replaced(&replaced(c, &replacement)?, &rule)?)?);
        }
        current = next;
    }
    if substitution.is_empty() {
        return Ok(None);
    }
    Ok(Some((substitution, back)))
}

/// The root of the product of several radicands back to the product of
/// their roots (`_original_roots`).
fn original_roots(big_f: &Basic, f: &Basic, x: &Basic, p: &Basic) -> R {
    let xs: [Basic; 1] = [x.clone()];
    let mut bases: Vec<Basic> = Vec::new();
    for node in pows(f) {
        if node.has(x) && half_integer(&node) && is_polynomial(&node.args()[0], &xs) && !bases.contains(&node.args()[0])
        {
            bases.push(node.args()[0].clone());
        }
    }
    if bases.len() < 2 {
        return Ok(big_f.clone());
    }
    bases.sort_by_key(|b| b.to_string());
    let roots_: Vec<Basic> = bases.iter().map(sqrt).collect::<PolyResult<_>>()?;
    let y: Basic = mul(&roots_)?;
    let p_: Basic = expand(p)?;
    let restore = |node: &Basic| -> R {
        if is_pow(node) && half_integer(node) && expand(&node.args()[0])? == p_ {
            let k: i64 = i64::try_from(crate::numbers::numer(&exponent(node)?)).unwrap_or(1);
            let m: i64 = k.rem_euclid(2);
            return mul(&[pw(&node.args()[0], &int((k - m) / 2))?, pw(&y, &int(m))?]);
        }
        Ok(node.clone())
    };
    crate::core::traversal::bottom_up(big_f, &restore, false)
}

// ---------------------------------------------------------------------
// the roots of index n >= 3

/// A rational `y` with `y' + f y = g`, `None` when there is none
/// (`_rational_risch_de`).
fn rational_risch_de(f: &Basic, g: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    use crate::integrals::risch::risch::{frac_in, DifferentialExtension};
    use crate::integrals::risch::tower::{RationalFunction, TowerPoly};
    let fe: RationalFunction = RationalFunction::from_expr(f)?;
    let ge: RationalFunction = RationalFunction::from_expr(g)?;
    let solve = || -> R {
        let mut de: DifferentialExtension =
            DifferentialExtension::manual(vec![TowerPoly::one(x)], Vec::new(), Vec::new(), None, None, None);
        let (fa, fd) = frac_in(&fe, x, false)?;
        let (ga, gd) = frac_in(&ge, x, false)?;
        let (ya, yd) = match crate::integrals::risch::rde::risch_de(&fa, &fd, &ga, &gd, &mut de) {
            Ok(v) => v,
            Err(e) if crate::integrals::risch::risch::is_non_elementary(&e) => {
                return Err(PolyError::NotImplemented("no rational solution".to_string()))
            }
            Err(e) => return Err(e),
        };
        ya.as_element()?.div(&yd.as_element()?)?.as_expr()
    };
    crate::timeout::attempt_within(settings_timeout(), solve)
}

/// The integrand by components on `y**n = P`, `n >= 3` (`_index_n`).
fn index_n(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some((p_, n, components)) = root_components(f, x)? else {
        return Ok(None);
    };
    let mut big_f: Basic = if components[0] != Basic::zero() {
        ratint_expr(&components[0], x)?
    } else {
        Basic::zero()
    };
    let mut h: Basic = Basic::zero();
    for k in 1..n {
        let a: &Basic = &components[k as usize];
        if *a == Basic::zero() {
            continue;
        }
        let coefficient: Basic = cancel(&div(&mul(&[int(k), diff(&p_, x)?])?, &mul(&[int(n), p_.clone()])?)?)?;
        let root_k: Basic = pw(&p_, &Basic::from_rational(crate::numbers::rat(k, n)))?;
        match rational_risch_de(&coefficient, &cancel(a)?, x)? {
            None => match component_with_logarithms(a, &p_, n, k, x)? {
                None => h = add(&[h, mul(&[a.clone(), root_k])?])?,
                Some(found) => big_f = add(&[big_f, found])?,
            },
            Some(y) => big_f = add(&[big_f, mul(&[cancel(&y)?, root_k])?])?,
        }
    }
    if !verified(&big_f, &sub(f, &h)?, x, &p_)? {
        return Ok(None);
    }
    Ok(Some((big_f, h)))
}

pub const INDEX_N_FIELD_DEGREE: usize = 12;
const FIELD_SECONDS: f64 = 2.0;

type QPoly = DensePoly<Rationals>;
type KPoly = DensePoly<AlgebraicField>;

/// The component `A y**k` with its logarithms, verified
/// (`_component_with_logarithms`).
fn component_with_logarithms(a: &Basic, base: &Basic, n: i64, k: i64, x: &Basic) -> PolyResult<Option<Basic>> {
    let g: i64 = gcd_i64(k, n);
    let (n, k) = if g > 1 { (n / g, k / g) } else { (n, k) };
    if n == 2 {
        let Some(result) = integrate(&mul(&[a.clone(), sqrt(base)?])?, x, true)? else {
            return Ok(None);
        };
        if !result.remainder.is_zero() {
            return Ok(None);
        }
        return Ok(Some(result.elementary()?));
    }
    if symbols(base).iter().chain(symbols(a).iter()).any(|s| s != x) {
        return Ok(None);
    }
    let p: QPoly = match rationals().from_expr(base, x) {
        Ok(p) => p,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some((terms, m, d)) = hermite_index_n(a, &p, n, k, x)? else {
        return Ok(None);
    };
    let y: Basic = pw(base, &Basic::from_rational(crate::numbers::rat(1, n)))?;
    let yk: Basic = pw(&y, &int(k))?;
    let field: Field<Rationals> = rationals();
    let mut exact: Vec<Basic> = Vec::new();
    for (s, t) in &terms {
        exact.push(mul(&[div(&field.poly_expr(s)?, &field.poly_expr(t)?)?, yk.clone()])?);
    }
    let exact: Basic = add(&exact)?;
    let found: Basic = if m.is_zero() {
        exact
    } else {
        let Some(logs) = logarithms_index_n(&m, &d, &p, n, k, x)? else {
            return Ok(None);
        };
        add(&[exact, logs])?
    };
    if !verified(&found, &mul(&[a.clone(), yk])?, x, base)? {
        return Ok(None);
    }
    Ok(Some(found))
}

fn qq(q: Rat) -> Rat {
    q
}

/// The Hermite reduction of `A y**k` on `y**n = P` (`_hermite_index_n`):
/// the terms `S y**k/T` and the remainder `M y**k/(D P)` with simple poles.
fn hermite_index_n(
    a: &Basic,
    p: &QPoly,
    n: i64,
    k: i64,
    x: &Basic,
) -> PolyResult<Option<(Vec<(QPoly, QPoly)>, QPoly, QPoly)>> {
    let f: Field<Rationals> = rationals();
    let xs: Symbol = Symbol(x.clone());
    let (numerator, denominator) = fraction(&cancel(a)?)?;
    let mut m: QPoly = f.from_expr(&numerator, x)?;
    let mut d: QPoly = f.from_expr(&denominator, x)?;
    let mut j: usize = 0;
    loop {
        let (quotient, remainder) = d.div(p)?;
        if !remainder.is_zero() {
            break;
        }
        d = quotient;
        j += 1;
    }
    if j == 0 {
        m = m.mul(p)?;
        j = 1;
    }
    let mut terms: Vec<(QPoly, QPoly)> = Vec::new();
    let kn: Rat = crate::numbers::rat(k, n);
    let pd: QPoly = p.diff(1);
    while j >= 2 {
        crate::timeout::check()?;
        let c: Rat = Rat::from(j as i64 - 1) - &kn;
        let s: QPoly = m.mul(&inverse(&d.mul(&pd)?.mul_ground(&c), p)?)?.rem(p)?.neg();
        m = m
            .sub(&d.mul(&s.diff(1).mul(p)?.sub(&s.mul(&pd)?.mul_ground(&c))?)?)?
            .exquo(p)?;
        terms.push((s, p.pow(j - 1)));
        j -= 1;
    }
    loop {
        crate::timeout::check()?;
        let common: QPoly = m.gcd(&d)?;
        m = m.exquo(&common)?;
        d = d.exquo(&common)?;
        let (_, factors) = d.sqf_list();
        let mut ramified: Option<(QPoly, usize)> = None;
        for (v, e) in &factors {
            let g: QPoly = v.gcd(p)?;
            if g.degree() > 0 {
                ramified = Some((g, *e));
                break;
            }
        }
        if let Some((v, e)) = ramified {
            let u_: QPoly = d.exquo(&v.pow(e))?;
            let p1: QPoly = p.exquo(&v)?;
            let c: Rat = Rat::from(e as i64) - &kn;
            let s: QPoly = m
                .mul(&inverse(&u_.mul(&v.diff(1))?.mul(&p1)?.mul_ground(&c), &v)?)?
                .rem(&v)?
                .neg();
            let subtracted: QPoly = s
                .diff(1)
                .mul(&v)?
                .mul(&p1)?
                .add(&s.mul(&v)?.mul(&p1.diff(1))?.mul_ground(&kn))?
                .sub(&s.mul(&v.diff(1))?.mul(&p1)?.mul_ground(&c))?;
            m = m.sub(&u_.mul(&subtracted)?)?.exquo(&v)?;
            d = u_.mul(&v.pow(e - 1))?;
            terms.push((s, v.pow(e)));
            continue;
        }
        let multiple: Vec<(QPoly, usize)> = factors.into_iter().filter(|(_, e)| *e >= 2).collect();
        if multiple.is_empty() {
            break;
        }
        // max by the multiplicity, the first of the largest
        let mut best: usize = 0;
        for (i, (_, e)) in multiple.iter().enumerate() {
            if *e > multiple[best].1 {
                best = i;
            }
        }
        let (v, e) = multiple[best].clone();
        let u_: QPoly = d.exquo(&v.pow(e))?;
        let c: Rat = Rat::from(e as i64 - 1);
        let s: QPoly = m
            .mul(&inverse(&u_.mul(&v.diff(1))?.mul(p)?.mul_ground(&c), &v)?)?
            .rem(&v)?
            .neg();
        let subtracted: QPoly = s
            .diff(1)
            .mul(&v)?
            .mul(p)?
            .sub(&s.mul(&v.diff(1))?.mul(p)?.mul_ground(&c))?
            .add(&s.mul(&pd)?.mul(&v)?.mul_ground(&kn))?;
        m = m.sub(&u_.mul(&subtracted)?)?.exquo(&v)?;
        d = u_.mul(&v.pow(e - 1))?;
        terms.push((s, v.pow(e - 1)));
    }
    let (mut q, r) = m.div(&d)?;
    let dp: i64 = p.degree() as i64;
    let mut s: QPoly = p.with_rep(Vec::new());
    while !q.is_zero() && q.degree() as i64 >= dp - 1 {
        crate::timeout::check()?;
        let i: i64 = q.degree() as i64 - dp + 1;
        let c: Rat = q.LC() / (p.LC() * (Rat::from(i) + crate::numbers::rat(k * dp, n)));
        let mono: QPoly = monomial(qq(c), i as usize, &crate::polys::domains::QQ, &xs);
        s = s.add(&mono)?;
        q = q.sub(&mono.diff(1).mul(p)?.add(&mono.mul(&pd)?.mul_ground(&kn))?)?;
    }
    if !s.is_zero() {
        terms.push((s, one_like(p)));
    }
    if !q.is_zero() && n * (q.degree() as i64 + 1 - dp) + k * dp >= gcd_i64(n, dp) {
        return Ok(None);
    }
    let m: QPoly = q.mul(&d)?.add(&r)?;
    let common: QPoly = m.gcd(&d)?;
    Ok(Some((terms, m.exquo(&common)?, d.exquo(&common)?)))
}

/// The places above the roots of `G` on the branch `y = W mod G`, with
/// the residue `rho` (`_Places`): the branch lifted on demand.
struct Places {
    g: KPoly,
    w: KPoly,
    rho: Number,
    group: usize,
    p: KPoly,
    n: usize,
    lifted: std::cell::RefCell<(usize, KPoly, KPoly)>,
}

impl Places {
    fn new(g: KPoly, w: KPoly, rho: Number, group: usize, p: KPoly, n: usize) -> PolyResult<Places> {
        let k: &AlgebraicField = &p.domain;
        let start: KPoly = inverse(&w.pow(n - 1).mul_ground(&k.from_int(n as i64)), &g)?;
        let lifted = std::cell::RefCell::new((1, w.clone(), start));
        Ok(Places {
            g,
            w,
            rho,
            group,
            p,
            n,
            lifted,
        })
    }

    /// The branch modulo `G**N` (Newton's iteration on `y**n = P`).
    fn branch(&self, big_n: usize) -> PolyResult<KPoly> {
        let (mut precision, mut w, mut inv) = self.lifted.borrow().clone();
        let k: &AlgebraicField = &self.p.domain;
        let n_: Number = k.from_int(self.n as i64);
        let two: KPoly = self.g.with_rep(vec![k.from_int(2)]);
        while precision < big_n {
            crate::timeout::check()?;
            precision = (2 * precision).min(big_n);
            let modulus: KPoly = self.g.pow(precision);
            w = w
                .sub(&w.pow(self.n).sub(&self.p)?.mul(&inv)?.rem(&modulus)?)?
                .rem(&modulus)?;
            inv = inv
                .mul(&two.sub(&w.pow(self.n - 1).mul_ground(&n_).mul(&inv)?.rem(&modulus)?)?)?
                .rem(&modulus)?;
        }
        *self.lifted.borrow_mut() = (precision, w.clone(), inv);
        w.rem(&self.g.pow(big_n))
    }
}

/// A point at infinity where `y ~ s x**(deg P/n)`, with the residue `rho`
/// (`_PlaceAtInfinity`).
#[derive(Clone)]
struct PlaceAtInfinity {
    s: Number,
    rho: Number,
}

/// `(beta/m) (power log U - log denominator)` with `U = sum U_i y**i`
/// (`_LogTermN`).
#[derive(Clone)]
struct LogTermN {
    beta: Number,
    m: i64,
    u: Vec<KPoly>,
    denominator: KPoly,
    power: i64,
}

/// The logarithmic part of `M y**k/(D P)` on `y**n = P`
/// (`_logarithms_index_n`).
fn logarithms_index_n(m: &QPoly, d: &QPoly, p: &QPoly, n: i64, k: i64, x: &Basic) -> PolyResult<Option<Basic>> {
    let xs: Symbol = Symbol(x.clone());
    let dp: i64 = p.degree() as i64;
    let at_infinity: bool = dp % n == 0;
    let tau: Basic = fresh("tau", false, false);
    let nu: usize = n as usize;
    let big_r: QPoly = resultant_in(
        d,
        &m.pow(nu).neg(),
        &d.diff(1).pow(nu).mul(&p.pow((n - k) as usize))?,
        1,
        &tau,
    );
    if big_r.degree() >= 1 {
        let (_, factors) = big_r.factor_list()?;
        if factors.iter().map(|(f, _)| f.degree()).max().unwrap_or(0) > 2 {
            return Ok(None);
        }
    }
    let norms: Vec<Basic> = if big_r.degree() >= 1 {
        match residue_values(&rationals(), &big_r, &tau)? {
            Some(v) => v,
            None => return Ok(None),
        }
    } else {
        Vec::new()
    };
    let z: Basic = fresh("z", false, false);
    let Some(zeta) = primitive_root_of_unity(n, &z)? else {
        return Ok(None);
    };
    let mut generators: Vec<Basic> = vec![zeta.clone()];
    let mut principal_roots: Vec<Basic> = Vec::new();
    for value in &norms {
        let Some(root) = preferably_real_root(value, n, &z)? else {
            return Ok(None);
        };
        principal_roots.push(root.clone());
        generators.push(root);
    }
    let kappa: Rat = if at_infinity {
        residue_at_infinity(m, d, p, n, k)?
    } else {
        Rat::ZERO
    };
    let mut s0: Option<Basic> = None;
    if kappa != Rat::ZERO {
        let Some(s) = preferably_real_root(&Basic::from_rational(p.LC()), n, &z)? else {
            return Ok(None);
        };
        generators.push(s.clone());
        s0 = Some(s);
    }
    let mut distinct: Vec<Basic> = Vec::new();
    for g in &generators {
        if g.as_rational().is_none() && !distinct.contains(g) {
            distinct.push(g.clone());
        }
    }
    distinct.sort_by_key(|g| g.to_string());
    let found_field = crate::timeout::attempt_within(Some(FIELD_SECONDS), || {
        NumberField::new(&distinct, INDEX_N_FIELD_DEGREE)
    })?;
    let Some(nf) = found_field else {
        return Ok(None);
    };
    if nf.degree > INDEX_N_FIELD_DEGREE {
        return Ok(None);
    }
    let l: Field<AlgebraicField> = algebraic(&nf);
    let kf: AlgebraicField = nf.field.clone();
    let coordinate = |e: &Basic| -> PolyResult<Number> {
        if e.as_rational().is_some() {
            return l.element(e);
        }
        nf.coordinates
            .iter()
            .find(|(g, _)| g == e)
            .map(|(_, c)| c.clone())
            .ok_or_else(|| PolyError::Value(format!("{} is not a generator", e)))
    };
    let (m_, d_, p_) = (l.poly(m)?, l.poly(d)?, l.poly(p)?);
    let zeta_: Number = coordinate(&zeta)?;
    let (a, b) = bezout(k, n);
    let mut finite: Vec<Places> = Vec::new();
    let dd: KPoly = d_.diff(1);
    for (group, root) in principal_roots.iter().enumerate() {
        let mut rho: Number = coordinate(root)?;
        let norm: Number = kf.pow(&rho, n as u64);
        let g: KPoly = d_.gcd(
            &m_.pow(nu)
                .sub(&dd.pow(nu).mul(&p_.pow((n - k) as usize))?.mul_ground(&norm))?,
        )?;
        if g.degree() == 0 {
            return Ok(None);
        }
        for _ in 0..n {
            let w_k: KPoly = dd.mul(&p_)?.mul(&inverse(&m_, &g)?)?.mul_ground(&rho).rem(&g)?;
            let w: KPoly = power_modulo(&w_k, a, &g)?.mul(&power_modulo(&p_, b, &g)?)?.rem(&g)?;
            finite.push(Places::new(g.clone(), w, rho.clone(), group, p_.clone(), nu)?);
            rho = kf.mul(&rho, &zeta_);
        }
    }
    let mut infinite: Vec<PlaceAtInfinity> = Vec::new();
    if let Some(s0) = &s0 {
        let mut s: Number = coordinate(s0)?;
        for _ in 0..n {
            let rho: Number = kf.neg(&kf.mul(&kf.pow(&s, k as u64), &kf.from_rational(&kappa)));
            infinite.push(PlaceAtInfinity { s: s.clone(), rho });
            s = kf.mul(&s, &zeta_);
        }
    }
    if finite.is_empty() && infinite.is_empty() {
        return Ok(None);
    }
    let mut logs: Option<Vec<LogTermN>> = orbit_logarithms(&finite, &infinite, &zeta_, &p_, nu, &xs)?;
    let matches = |logs: &Option<Vec<LogTermN>>| -> PolyResult<bool> {
        match logs {
            None => Ok(false),
            Some(l) => logarithmic_derivative_matches(l, &m_, &d_, &p_, nu, k as usize, &xs),
        }
    };
    if !matches(&logs)? {
        let mut groups: Vec<usize> = finite.iter().map(|p| p.group).collect();
        groups.sort();
        groups.dedup();
        let orbits: usize = groups.len() + usize::from(!infinite.is_empty());
        if orbits == 1 {
            return Ok(None);
        }
        logs = basis_logarithms(&finite, &infinite, &kf, &p_, nu, &xs)?;
        if !matches(&logs)? {
            return Ok(None);
        }
    }
    Ok(Some(real_logarithms(&logs.expect("logarithms"), &l, p, nu, x)?))
}

/// The logarithms of the orbits of places under `y -> zeta y`, each with
/// a function of divisor `m (Q_0 - sum Q_l/n)` (`_orbit_logarithms`).
fn orbit_logarithms(
    finite: &[Places],
    infinite: &[PlaceAtInfinity],
    zeta: &Number,
    p: &KPoly,
    n: usize,
    x: &Symbol,
) -> PolyResult<Option<Vec<LogTermN>>> {
    let k: &AlgebraicField = &p.domain;
    let mut logs: Vec<LogTermN> = Vec::new();
    let mut groups: Vec<usize> = finite.iter().map(|pl| pl.group).collect();
    groups.sort();
    groups.dedup();
    let mut orbits: Vec<(Vec<&Places>, Vec<PlaceAtInfinity>)> = Vec::new();
    for group in groups {
        orbits.push((finite.iter().filter(|pl| pl.group == group).collect(), Vec::new()));
    }
    if !infinite.is_empty() {
        orbits.push((Vec::new(), infinite.to_vec()));
    }
    let mut coefficients: Vec<Rat> = vec![crate::numbers::rat(n as i64 - 1, n as i64)];
    coefficients.extend(std::iter::repeat(crate::numbers::rat(-1, n as i64)).take(n - 1));
    for (places, points) in &orbits {
        let mut found: Option<(Vec<KPoly>, KPoly)> = None;
        let mut m: usize = n;
        for multiple in 1..=TORSION_BOUND as usize {
            m = multiple * n;
            let cf: &[Rat] = if places.is_empty() { &[] } else { &coefficients };
            let ci: &[Rat] = if points.is_empty() { &[] } else { &coefficients };
            found = function_with_divisor_n(places, cf, points, ci, m as i64, p, n, x, 0)?;
            if found.is_some() {
                break;
            }
        }
        let Some((mut v, denominator)) = found else {
            return Ok(None);
        };
        let mut power: i64 = 1;
        if !places.is_empty() {
            let (v2, power2) = root_on_the_curve(&v, places[0], m, p, n, x)?;
            v = v2;
            power = power2;
        }
        for l in 0..n {
            let u: Vec<KPoly> = v
                .iter()
                .enumerate()
                .map(|(i, vi)| vi.mul_ground(&k.pow(zeta, (l * i) as u64)))
                .collect();
            let rho: Option<Number> = if !places.is_empty() {
                let w: KPoly = places[0]
                    .w
                    .mul_ground(&k.pow(zeta, ((n - l) % n) as u64))
                    .rem(&places[0].g)?;
                let mut found_rho: Option<Number> = None;
                for place in places {
                    if place.w.sub(&w)?.is_zero() {
                        found_rho = Some(place.rho.clone());
                        break;
                    }
                }
                found_rho
            } else {
                let s: Number = k.mul(&points[0].s, &k.pow(zeta, ((n - l) % n) as u64));
                points.iter().find(|pt| pt.s == s).map(|pt| pt.rho.clone())
            };
            let Some(rho) = rho else {
                return Ok(None);
            };
            logs.push(LogTermN {
                beta: rho,
                m: m as i64,
                u,
                denominator: denominator.clone(),
                power,
            });
        }
    }
    Ok(Some(logs))
}

/// A root `w` of the function on the curve, `V = w**mu` up to a constant,
/// the largest `mu` first (`_root_on_the_curve`).
fn root_on_the_curve(
    v: &[KPoly],
    place: &Places,
    m: usize,
    p: &KPoly,
    n: usize,
    x: &Symbol,
) -> PolyResult<(Vec<KPoly>, i64)> {
    let d: usize = p.degree() as usize;
    let g: usize = gcd_i64(n as i64, d as i64) as usize;
    let big_b: usize = n / g * (m / n) * place.g.degree() as usize;
    let mut candidates: Vec<usize> = (2..=m).filter(|mu| m % mu == 0 && big_b % mu == 0).collect();
    candidates.sort_by(|a, b| b.cmp(a));
    for mu in candidates {
        let Some((w, _)) = function_with_divisor_n(
            &[place],
            &[Rat::ONE],
            &[],
            &[],
            (m / mu) as i64,
            p,
            n,
            x,
            (big_b / mu) as i64,
        )?
        else {
            continue;
        };
        let power: Vec<KPoly> = power_on_the_curve(&w, mu, p, n)?;
        let pairs: Vec<(&KPoly, &KPoly)> = v
            .iter()
            .zip(&power)
            .filter(|(a, b)| !a.is_zero() || !b.is_zero())
            .collect();
        if pairs.is_empty() || !pairs.iter().all(|(a, b)| !a.is_zero() && !b.is_zero()) {
            continue;
        }
        let (a0, b0) = pairs[0];
        let mut proportional: bool = true;
        for (a, b) in &pairs {
            if !a.mul(b0)?.sub(&b.mul(a0)?)?.is_zero() {
                proportional = false;
                break;
            }
        }
        if proportional {
            return Ok((w, mu as i64));
        }
    }
    Ok((v.to_vec(), 1))
}

/// `w**mu` on the curve `y**n = P` as the coefficients of `1, y, ...`
/// (`_power_on_the_curve`).
fn power_on_the_curve(w: &[KPoly], mu: usize, p: &KPoly, n: usize) -> PolyResult<Vec<KPoly>> {
    let zero: KPoly = p.with_rep(Vec::new());
    let mut result: Vec<KPoly> = vec![one_like(p)];
    result.extend(std::iter::repeat(zero.clone()).take(n - 1));
    for _ in 0..mu {
        let mut product: Vec<KPoly> = vec![zero.clone(); 2 * n - 1];
        for (i, a) in result.iter().enumerate() {
            for (j, b) in w.iter().enumerate() {
                product[i + j] = product[i + j].add(&a.mul(b)?)?;
            }
        }
        for i in (n..=2 * n - 2).rev() {
            let term: KPoly = product[i].mul(p)?;
            product[i - n] = product[i - n].add(&term)?;
        }
        product.truncate(n);
        result = product;
    }
    Ok(result)
}

/// Trager's general form over a rational basis of the residues
/// (`_basis_logarithms`).
fn basis_logarithms(
    finite: &[Places],
    infinite: &[PlaceAtInfinity],
    k: &AlgebraicField,
    p: &KPoly,
    n: usize,
    x: &Symbol,
) -> PolyResult<Option<Vec<LogTermN>>> {
    let mut residues: Vec<Number> = finite.iter().map(|pl| pl.rho.clone()).collect();
    residues.extend(infinite.iter().map(|pt| pt.rho.clone()));
    let Some((chosen, coordinates)) = rational_basis(&residues, k) else {
        return Ok(None);
    };
    let finite_refs: Vec<&Places> = finite.iter().collect();
    let mut logs: Vec<LogTermN> = Vec::new();
    for (j, index) in chosen.iter().enumerate() {
        let c_finite: Vec<Rat> = coordinates[..finite.len()].iter().map(|row| row[j].clone()).collect();
        let c_infinite: Vec<Rat> = coordinates[finite.len()..].iter().map(|row| row[j].clone()).collect();
        let mut n0: Int = Int::ONE;
        for c in c_finite.iter().chain(&c_infinite) {
            n0 = crate::numbers::lcm(&n0, &crate::numbers::denom(c));
        }
        let n0: i64 = i64::try_from(&n0).unwrap_or(i64::MAX);
        let mut found: Option<(Vec<KPoly>, KPoly)> = None;
        let mut m: i64 = n0;
        for multiple in 1..=TORSION_BOUND {
            m = multiple * n0;
            found = function_with_divisor_n(&finite_refs, &c_finite, infinite, &c_infinite, m, p, n, x, 0)?;
            if found.is_some() {
                break;
            }
        }
        let Some((u, denominator)) = found else {
            return Ok(None);
        };
        logs.push(LogTermN {
            beta: residues[*index].clone(),
            m,
            u,
            denominator,
            power: 1,
        });
    }
    Ok(Some(logs))
}

fn sorted_by_str(mut v: Vec<Basic>) -> Vec<Basic> {
    v.sort_by_key(|e| e.to_string());
    v
}

/// An `n`-th root of `value`, a real one when there is one
/// (`_preferably_real_root`).
fn preferably_real_root(value: &Basic, n: i64, z: &Basic) -> PolyResult<Option<Basic>> {
    let candidates: Vec<Basic> = if value.as_rational().is_some() {
        let Some(found) = roots_of(&sub(&pw(z, &int(n))?, value)?, z)? else {
            return Ok(None);
        };
        if found.iter().map(|(_, k)| *k as i64).sum::<i64>() != n {
            return Ok(None);
        }
        sorted_by_str(found.into_iter().map(|(r, _)| r).collect())
    } else {
        let binomial: Option<Vec<Basic>> = if n != 2 { binomial_roots(value, n, z)? } else { None };
        let Some(b) = binomial else {
            return Ok(None);
        };
        sorted_by_str(b)
    };
    for root in &candidates {
        if crate::core::expr::is(root, Predicate::real) == Some(true) {
            return Ok(Some(root.clone()));
        }
    }
    Ok(candidates.into_iter().next())
}

/// A primitive `n`-th root of unity in closed form
/// (`_primitive_root_of_unity`).
fn primitive_root_of_unity(n: i64, z: &Basic) -> PolyResult<Option<Basic>> {
    let Some(found) = roots_of(&sub(&pw(z, &int(n))?, &int(1))?, z)? else {
        return Ok(None);
    };
    if found.iter().map(|(_, k)| *k as i64).sum::<i64>() != n {
        return Ok(None);
    }
    let divisors: Vec<i64> = (1..n).filter(|d| n % d == 0).collect();
    for root in sorted_by_str(found.into_iter().map(|(r, _)| r).collect()) {
        let mut primitive: bool = true;
        for d in &divisors {
            let value = crate::numeric::reliable_complex(&sub(&expand(&pw(&root, &int(*d))?)?, &int(1))?, 30, &[]);
            match value {
                Some((a, b)) if (a * a + b * b).sqrt() > 1e-10 => {}
                _ => {
                    primitive = false;
                    break;
                }
            }
        }
        if primitive {
            return Ok(Some(root));
        }
    }
    Ok(None)
}

/// `(a, b)` with `a k + b n = gcd(k, n)` (`_bezout`).
fn bezout(k: i64, n: i64) -> (i64, i64) {
    let (mut r, mut r_) = (k, n);
    let (mut a, mut a_) = (1i64, 0i64);
    let (mut b, mut b_) = (0i64, 1i64);
    while r_ != 0 {
        let quotient: i64 = r.div_euclid(r_);
        (r, r_) = (r_, r - quotient * r_);
        (a, a_) = (a_, a - quotient * a_);
        (b, b_) = (b_, b - quotient * b_);
    }
    (a, b)
}

/// `p**exponent mod G`, a negative exponent through the inverse
/// (`_power_modulo`).
fn power_modulo<D: Domain>(p: &DensePoly<D>, exponent: i64, g: &DensePoly<D>) -> PolyResult<DensePoly<D>> {
    let base: DensePoly<D> = if exponent >= 0 {
        p.rem(g)?
    } else {
        inverse(&p.rem(g)?, g)?
    };
    let mut result: DensePoly<D> = one_like(p);
    for _ in 0..exponent.unsigned_abs() {
        result = result.mul(&base)?.rem(g)?;
    }
    Ok(result)
}

fn series_multiply<D: Domain>(a: &[D::Elem], b: &[D::Elem], order: usize, k: &D) -> Vec<D::Elem> {
    let mut product: Vec<D::Elem> = vec![k.zero(); order + 1];
    for (i, ai) in a.iter().take(order + 1).enumerate() {
        for (j, bj) in b.iter().take(order + 1 - i).enumerate() {
            product[i + j] = k.add(&product[i + j], &k.mul(ai, bj));
        }
    }
    product
}

fn series_inverse<D: Domain>(p: &[D::Elem], order: usize, k: &D) -> PolyResult<Vec<D::Elem>> {
    let mut inverse_: Vec<D::Elem> = vec![k.exquo(&k.one(), &p[0])?];
    for i in 1..=order {
        let mut total: D::Elem = k.zero();
        for a in 1..=i {
            if a < p.len() {
                total = k.add(&total, &k.mul(&p[a], &inverse_[i - a]));
            }
        }
        inverse_.push(k.exquo(&k.neg(&total), &p[0])?);
    }
    Ok(inverse_)
}

fn series_power<D: Domain>(p: &[D::Elem], exponent: &Rat, order: usize, k: &D) -> PolyResult<Vec<D::Elem>> {
    let mut sigma: Vec<D::Elem> = vec![k.one()];
    for i in 1..=order {
        let mut total: D::Elem = k.zero();
        for a in 0..i {
            if i - a < p.len() {
                let c: Rat = exponent * Rat::from((i - a) as i64) - Rat::from(a as i64);
                let cc: D::Elem = k.exquo(
                    &k.from_integer(&crate::numbers::numer(&c)),
                    &k.from_integer(&crate::numbers::denom(&c)),
                )?;
                total = k.add(&total, &k.mul(&k.mul(&sigma[a], &p[i - a]), &cc));
            }
        }
        sigma.push(k.exquo(&total, &k.from_int(i as i64))?);
    }
    Ok(sigma)
}

fn series_of_the_root<D: Domain>(p: &DensePoly<D>, exponent: &Rat, order: usize) -> PolyResult<Vec<D::Elem>> {
    series_power(&p.monic()?.rep, exponent, order, &p.domain)
}

/// The coefficient of the differential at infinity which gives the
/// residues there (`_residue_at_infinity`).
fn residue_at_infinity(m: &QPoly, d: &QPoly, p: &QPoly, n: i64, k: i64) -> PolyResult<Rat> {
    let kq = crate::polys::domains::QQ;
    let dp: i64 = p.degree() as i64;
    let order: i64 = (k * dp).div_euclid(n) + 1;
    let shift: i64 = d.degree() as i64 + dp - m.degree() as i64;
    if shift > order {
        return Ok(Rat::ZERO);
    }
    let order_: usize = order as usize;
    let reversed_m: Vec<Rat> = m.rep.iter().rev().cloned().collect();
    let reversed_dp: Vec<Rat> = d.mul(p)?.rep.iter().rev().cloned().collect();
    let h: Vec<Rat> = series_multiply(&reversed_m, &series_inverse(&reversed_dp, order_, &kq)?, order_, &kq);
    let shift_: usize = shift.max(0) as usize;
    let mut shifted: Vec<Rat> = vec![Rat::ZERO; shift_];
    shifted.extend(h.into_iter().take(order_ + 1 - shift_));
    let product: Vec<Rat> = series_multiply(
        &shifted,
        &series_of_the_root(p, &crate::numbers::rat(k, n), order_)?,
        order_,
        &kq,
    );
    Ok(product[order_].clone())
}

/// The reduced row echelon form of rational rows and the pivot columns
/// (`_rref`).
fn rref(rows: &[Vec<Rat>]) -> (Vec<Vec<Rat>>, Vec<usize>) {
    let mut matrix: Vec<Vec<Rat>> = rows.to_vec();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r: usize = 0;
    let columns: usize = matrix.first().map(|row| row.len()).unwrap_or(0);
    for c in 0..columns {
        let Some(pivot) = (r..matrix.len()).find(|i| matrix[*i][c] != Rat::ZERO) else {
            continue;
        };
        matrix.swap(r, pivot);
        let lead: Rat = matrix[r][c].clone();
        matrix[r] = matrix[r].iter().map(|v| v / &lead).collect();
        for i in 0..matrix.len() {
            if i != r && matrix[i][c] != Rat::ZERO {
                let factor: Rat = matrix[i][c].clone();
                let pivot_row: Vec<Rat> = matrix[r].clone();
                matrix[i] = matrix[i].iter().zip(&pivot_row).map(|(a, b)| a - &factor * b).collect();
            }
        }
        pivots.push(c);
        r += 1;
        if r == matrix.len() {
            break;
        }
    }
    (matrix, pivots)
}

/// A basis of the rational span of the values and their coordinates on
/// it (`_rational_basis`).
fn rational_basis(values: &[Number], k: &AlgebraicField) -> Option<(Vec<usize>, Vec<Vec<Rat>>)> {
    let degree: usize = k.degree;
    let vectors: Vec<Vec<Rat>> = values
        .iter()
        .map(|value| {
            let mut c: Vec<Rat> = value.iter().rev().cloned().collect();
            c.resize(degree.max(c.len()), Rat::ZERO);
            c
        })
        .collect();
    let mut chosen: Vec<usize> = Vec::new();
    let mut rank: usize = 0;
    for (index, vector) in vectors.iter().enumerate() {
        let mut rows: Vec<Vec<Rat>> = chosen.iter().map(|i| vectors[*i].clone()).collect();
        rows.push(vector.clone());
        let (_, pivots) = rref(&rows);
        if pivots.len() > rank {
            chosen.push(index);
            rank += 1;
        }
    }
    let mut columns: Vec<Vec<Rat>> = chosen.iter().map(|i| vectors[*i].clone()).collect();
    columns.extend(vectors.iter().cloned());
    let rows: Vec<Vec<Rat>> = (0..degree)
        .map(|r| columns.iter().map(|c| c[r].clone()).collect())
        .collect();
    let (reduced, pivots) = rref(&rows);
    if pivots.iter().take(chosen.len()).cloned().collect::<Vec<_>>() != (0..chosen.len()).collect::<Vec<_>>() {
        return None;
    }
    let mut coordinates: Vec<Vec<Rat>> = Vec::new();
    for index in 0..values.len() {
        let column: usize = chosen.len() + index;
        coordinates.push((0..chosen.len()).map(|r| reduced[r][column].clone()).collect());
    }
    Some((chosen, coordinates))
}

fn int_of_rat_toward_zero(q: &Rat) -> i64 {
    let (n, d) = (crate::numbers::numer(q), crate::numbers::denom(q));
    i64::try_from(&(n / d)).unwrap_or(0)
}

/// A function `U/denominator`, `U = sum U_i y**i`, with the divisor
/// `m (sum c_j Q_j)` at the places and points (`_function_with_divisor_n`).
fn function_with_divisor_n(
    finite: &[&Places],
    c_finite: &[Rat],
    infinite: &[PlaceAtInfinity],
    c_infinite: &[Rat],
    m: i64,
    p: &KPoly,
    n: usize,
    x: &Symbol,
    extra_pole: i64,
) -> PolyResult<Option<(Vec<KPoly>, KPoly)>> {
    let k: &AlgebraicField = &p.domain;
    let d: i64 = p.degree() as i64;
    let g: i64 = gcd_i64(n as i64, d);
    let e: i64 = n as i64 / g;
    let mut pole_orders: Vec<(usize, i64)> = Vec::new();
    for (place, c) in finite.iter().zip(c_finite) {
        let order_: i64 = if *c < Rat::ZERO {
            int_of_rat_toward_zero(&(-c * Rat::from(m)))
        } else {
            0
        };
        match pole_orders.iter_mut().find(|(gr, _)| *gr == place.group) {
            Some(entry) => entry.1 = entry.1.max(order_),
            None => pole_orders.push((place.group, order_)),
        }
    }
    let mut denominator: KPoly = one_like(p);
    let mut groups: Vec<(usize, KPoly)> = Vec::new();
    for place in finite {
        match groups.iter_mut().find(|(gr, _)| *gr == place.group) {
            Some(entry) => entry.1 = place.g.clone(),
            None => groups.push((place.group, place.g.clone())),
        }
    }
    let group_poly = |gr: usize| -> &KPoly { &groups.iter().find(|(g_, _)| *g_ == gr).expect("a group").1 };
    let mut big_b: i64 = 0;
    for (group, order_) in &pole_orders {
        if *order_ != 0 {
            denominator = denominator.mul(&group_poly(*group).pow(*order_ as usize))?;
            big_b += e * order_ * group_poly(*group).degree() as i64;
        }
    }
    let required_infinity: Vec<i64> = if !infinite.is_empty() {
        c_infinite
            .iter()
            .map(|c| -big_b - extra_pole + int_of_rat_toward_zero(&(c * Rat::from(m))))
            .collect()
    } else {
        vec![-big_b - extra_pole]
    };
    let pole: i64 = -required_infinity.iter().min().copied().unwrap_or(0);
    let bounds: Vec<i64> = (0..n as i64).map(|i| (pole - i * (d / g)).div_euclid(e)).collect();
    let mut offsets: Vec<usize> = Vec::new();
    let mut unknowns: usize = 0;
    for i in 0..n {
        offsets.push(unknowns);
        unknowns += (bounds[i] + 1).max(0) as usize;
    }
    if unknowns == 0 {
        return Ok(None);
    }
    let order_of = |gr: usize| -> i64 {
        pole_orders
            .iter()
            .find(|(g_, _)| *g_ == gr)
            .map(|(_, o)| *o)
            .unwrap_or(0)
    };
    let mut rows: Vec<Vec<Number>> = Vec::new();
    for (place, c) in finite.iter().zip(c_finite) {
        let big_n: i64 = int_of_rat_toward_zero(&(c * Rat::from(m))) + order_of(place.group);
        if big_n <= 0 {
            continue;
        }
        let modulus: KPoly = place.g.pow(big_n as usize);
        let size: usize = modulus.degree() as usize;
        let w_n: KPoly = place.branch(big_n as usize)?;
        let mut powers: Vec<KPoly> = vec![one_like(p)];
        for _ in 1..n {
            let next: KPoly = powers[powers.len() - 1].mul(&w_n)?.rem(&modulus)?;
            powers.push(next);
        }
        let mut columns: Vec<Vec<Number>> = Vec::new();
        for i in 0..n {
            for a in 0..=bounds[i] {
                if a < 0 {
                    break;
                }
                columns.push(coefficients(
                    &monomial(k.one(), a as usize, k, x).mul(&powers[i])?.rem(&modulus)?,
                    size,
                ));
            }
        }
        for r in 0..size {
            rows.push(columns.iter().map(|col| col[r].clone()).collect());
        }
    }
    let max_required: i64 = required_infinity.iter().max().copied().unwrap_or(0);
    if !infinite.is_empty() && max_required > -pole {
        let order: i64 = pole + max_required;
        let series: Vec<Vec<Number>> = (0..n)
            .map(|i| series_of_the_root(p, &crate::numbers::rat(i as i64, n as i64), order as usize))
            .collect::<PolyResult<_>>()?;
        for (point, required) in infinite.iter().zip(&required_infinity) {
            for r in 0..(required + pole).max(0) {
                let mut row: Vec<Number> = Vec::new();
                for i in 0..n {
                    for a in 0..=bounds[i] {
                        if a < 0 {
                            break;
                        }
                        let index: i64 = -pole + r + a + (i as i64 * d).div_euclid(n as i64);
                        let value: Number = if index >= 0 && index <= order {
                            series[i][index as usize].clone()
                        } else {
                            k.zero()
                        };
                        row.push(k.mul(&k.pow(&point.s, i as u64), &value));
                    }
                }
                rows.push(row);
            }
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let space: Vec<Vec<Number>> = k.nullspace(&rows, unknowns)?;
    let Some(mut vector) = space.into_iter().next() else {
        return Ok(None);
    };
    let Some(lead) = vector.iter().find(|entry| !k.is_zero(entry)).cloned() else {
        return Ok(None);
    };
    vector = vector
        .iter()
        .map(|entry| k.exquo(entry, &lead))
        .collect::<PolyResult<_>>()?;
    let mut u: Vec<KPoly> = Vec::new();
    for i in 0..n {
        if bounds[i] >= 0 {
            let slice: Vec<Number> = vector[offsets[i]..offsets[i] + bounds[i] as usize + 1]
                .iter()
                .rev()
                .cloned()
                .collect();
            u.push(DensePoly::new(&slice, k.clone(), x.clone()));
        } else {
            u.push(p.with_rep(Vec::new()));
        }
    }
    let big_norm: KPoly = norm(&u, p, n)?;
    let mut expected: KPoly = one_like(p);
    let mut exponents: Vec<(usize, Rat)> = Vec::new();
    for (place, c) in finite.iter().zip(c_finite) {
        match exponents.iter_mut().find(|(g_, _)| *g_ == place.group) {
            Some(entry) => entry.1 = &entry.1 + c,
            None => exponents.push((place.group, c.clone())),
        }
    }
    for (group, gp) in &groups {
        let total: Rat = exponents
            .iter()
            .find(|(g_, _)| g_ == group)
            .map(|(_, r)| r.clone())
            .unwrap_or(Rat::ZERO);
        let exponent_: i64 = int_of_rat_toward_zero(&(total * Rat::from(m))) + n as i64 * order_of(*group);
        expected = expected.mul(&gp.pow(exponent_.max(0) as usize))?;
    }
    if big_norm.is_zero()
        || big_norm.degree() != expected.degree()
        || !big_norm.sub(&expected.mul_ground(&big_norm.LC()))?.is_zero()
    {
        return Ok(None);
    }
    Ok(Some((u, denominator)))
}

/// Whether the sum of the logarithmic derivatives is the differential
/// `M y**k/(D P)` exactly on the curve (`_logarithmic_derivative_matches`).
fn logarithmic_derivative_matches(
    logs: &[LogTermN],
    m: &KPoly,
    d: &KPoly,
    p: &KPoly,
    n: usize,
    k: usize,
    x: &Symbol,
) -> PolyResult<bool> {
    use crate::polys::domains::{FracElement, FractionField};
    let kf: &AlgebraicField = &p.domain;
    let kx: FractionField<AlgebraicField> = FractionField::new(PolynomialRing::new(kf.clone(), &name_of(&x.0)));
    let y: Symbol = Symbol(fresh("y", false, false));
    let fraction_of = |q: &KPoly| -> FracElement<Number> { kx.from_ring(&q.rep) };
    let in_y = |coefficients: &[(usize, FracElement<Number>)]| -> DensePoly<FractionField<AlgebraicField>> {
        let degree: usize = coefficients.iter().map(|(i, _)| *i).max().unwrap_or(0);
        let rep: Vec<FracElement<Number>> = (0..=degree)
            .rev()
            .map(|i| {
                coefficients
                    .iter()
                    .find(|(j, _)| *j == i)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(|| kx.zero())
            })
            .collect();
        DensePoly::new(&rep, kx.clone(), y.clone())
    };
    let p_: FracElement<Number> = fraction_of(p);
    let modulus = in_y(&[(n, kx.one()), (0, kx.neg(&p_))]);
    let mut total = in_y(&[]);
    let pd: KPoly = p.diff(1);
    for term in logs {
        let mut uc: Vec<(usize, FracElement<Number>)> = Vec::new();
        let mut duc: Vec<(usize, FracElement<Number>)> = Vec::new();
        for (i, ui) in term.u.iter().enumerate() {
            if ui.is_zero() {
                continue;
            }
            uc.push((i, fraction_of(ui)));
            let c: Number = kf.exquo(&kf.from_int(i as i64), &kf.from_int(n as i64))?;
            let numerator: KPoly = ui.diff(1).mul(p)?.add(&ui.mul(&pd)?.mul_ground(&c))?;
            duc.push((i, kx.exquo(&fraction_of(&numerator), &p_)?));
        }
        let u = in_y(&uc);
        let du = in_y(&duc);
        let mut dlog = du
            .mul(&inverse(&u, &modulus)?)?
            .rem(&modulus)?
            .mul_ground(&kx.from_int(term.power));
        if term.denominator.degree() >= 1 {
            let piece: FracElement<Number> =
                kx.exquo(&fraction_of(&term.denominator.diff(1)), &fraction_of(&term.denominator))?;
            dlog = dlog.sub(&in_y(&[(0, piece)]))?;
        }
        let beta: FracElement<Number> = kx.from_ring(&if kf.is_zero(&term.beta) {
            Vec::new()
        } else {
            vec![term.beta.clone()]
        });
        total = total.add(&dlog.mul_ground(&kx.exquo(&beta, &kx.from_int(term.m))?))?;
    }
    let expected = in_y(&[(k, kx.exquo(&fraction_of(m), &kx.mul(&fraction_of(d), &p_))?)]);
    Ok(total.sub(&expected)?.is_zero())
}

/// The norm of `U = sum U_i y**i` from the function field to `K[x]`
/// (`_norm`).
fn norm(u: &[KPoly], p: &KPoly, n: usize) -> PolyResult<KPoly> {
    let k: &AlgebraicField = &p.domain;
    let kx: PolynomialRing<AlgebraicField> = PolynomialRing::new(k.clone(), &name_of(&p.gen.0));
    let mut curve: Vec<Vec<Number>> = vec![vec![k.one()]];
    curve.extend(std::iter::repeat(Vec::new()).take(n - 1));
    curve.push(p.neg().rep);
    let function: Vec<Vec<Number>> = dup_strip(&u.iter().rev().map(|ui| ui.rep.clone()).collect::<Vec<_>>(), &kx);
    let resultant: Vec<Number> = dup_resultant(&curve, &function, &kx);
    Ok(DensePoly::new(&resultant, k.clone(), p.gen.clone()))
}

/// The logarithms as an expression, their real form (logarithms of sums
/// of squares and arc tangents) when the parts are real
/// (`_real_logarithms`).
fn real_logarithms(logs: &[LogTermN], l: &Field<AlgebraicField>, p: &QPoly, n: usize, x: &Basic) -> R {
    use crate::functions::elementary::complexes::as_real_imag;
    use crate::functions::elementary::exponential::log;
    let big_x: Basic = fresh("x", true, false);
    let big_y: Basic = fresh("y", true, false);
    let p_expr: Basic = rationals().poly_expr(p)?;
    let y: Basic = pw(&p_expr, &Basic::from_rational(crate::numbers::rat(1, n as i64)))?;
    let mut total: Vec<Basic> = Vec::new();
    let mut complex_form: Vec<Basic> = Vec::new();
    let mut real: bool = true;
    let p_x: Basic = replaced(&p_expr, &[(x.clone(), big_x.clone())])?;
    let back: Vec<(Basic, Basic)> = vec![(big_x.clone(), x.clone()), (big_y.clone(), y)];
    let to_x: Vec<(Basic, Basic)> = vec![(x.clone(), big_x.clone())];
    let parts_of = |e: &Basic| -> PolyResult<(Basic, Basic)> {
        let (a, b) = as_real_imag(e, true, None)?.unwrap_or_else(|| (e.clone(), Basic::zero()));
        Ok((a, b))
    };
    let has_complex = |e: &Basic| -> bool {
        e.has(&Basic::imaginary_unit())
            || e.preorder().iter().any(|n| {
                matches!(
                    n.head(),
                    Head::Function(crate::core::basic::Function::Re | crate::core::basic::Function::Im)
                )
            })
    };
    for term in logs {
        let beta: Basic = div(&l.expr(&term.beta)?, &int(term.m))?;
        let mut u_terms: Vec<Basic> = Vec::new();
        for (i, ui) in term.u.iter().enumerate() {
            u_terms.push(mul(&[l.poly_expr(ui)?, pw(&big_y, &int(i as i64))?])?);
        }
        let u: Basic = replaced(&add(&u_terms)?, &to_x)?;
        let denominator: Basic = replaced(&l.poly_expr(&term.denominator)?, &to_x)?;
        complex_form.push(replaced(
            &mul(&[
                beta.clone(),
                sub(&mul(&[int(term.power), log(&u)?])?, &log(&denominator)?)?,
            ])?,
            &back,
        )?);
        let (b_r, b_i) = parts_of(&beta)?;
        let (u_r, u_i) = parts_of(&expand(&u)?)?;
        let (u_r, u_i) = (expand(&u_r)?, expand(&u_i)?);
        let (d_r, d_i) = parts_of(&expand(&denominator)?)?;
        let (d_r, d_i) = (expand(&d_r)?, expand(&d_i)?);
        if [&u_r, &u_i, &d_r, &d_i, &b_r, &b_i]
            .iter()
            .any(|part| has_complex(part))
        {
            real = false;
            continue;
        }
        let mut contribution: Vec<Basic> = Vec::new();
        if b_r != Basic::zero() {
            let squares_u: Basic = add(&[pw(&u_r, &int(2))?, pw(&u_i, &int(2))?])?;
            let squares_d: Basic = add(&[pw(&d_r, &int(2))?, pw(&d_i, &int(2))?])?;
            contribution.push(div(
                &mul(&[
                    b_r.clone(),
                    sub(
                        &mul(&[int(term.power), log_of_factors(&squares_u, &big_x, &big_y, n, &p_x)?])?,
                        &log_of_factors(&squares_d, &big_x, &big_y, n, &p_x)?,
                    )?,
                ])?,
                &int(2),
            )?);
        }
        if b_i != Basic::zero() {
            for (numerator, denominator_, sign) in [(&u_i, &u_r, -term.power), (&d_i, &d_r, 1)] {
                if *numerator != Basic::zero() && *denominator_ != Basic::zero() {
                    let factor = |e: &Basic| crate::polys::polytools::factor(e, &[], false, true);
                    let at: Basic = crate::functions::apply_function(
                        crate::core::basic::Function::Atan,
                        &[div(&factor(numerator)?, &factor(denominator_)?)?],
                    )?;
                    contribution.push(mul(&[int(sign), b_i.clone(), at])?);
                }
            }
        }
        let contribution: Basic = add(&contribution)?;
        for part in crate::core::expr::terms_of(&contribution) {
            total.push(crate::core::function::expand_mul(&part, false)?);
        }
    }
    if !real {
        return add(&complex_form);
    }
    replaced(&add(&total)?, &back)
}

/// `log(p)` as the sum of the logarithms of its factors, `p` reduced on
/// `Y**n = P(X)` (`_log_of_factors`).
fn log_of_factors(p: &Basic, big_x: &Basic, big_y: &Basic, n: usize, p_x: &Basic) -> R {
    use crate::functions::elementary::exponential::log;
    let attempt = || -> PolyResult<(Basic, Vec<(Basic, Basic)>)> {
        let mut coefficients: Vec<(i64, Basic)> = powers_of(p, big_y)?;
        coefficients.sort_by_key(|c| std::cmp::Reverse(c.0));
        let mut reduced: std::collections::BTreeMap<i64, Basic> = coefficients.iter().cloned().collect();
        let keys: Vec<i64> = coefficients.iter().map(|(k, _)| *k).collect();
        for i in keys {
            if i >= n as i64 {
                if let Some(c) = reduced.remove(&i) {
                    let old: Basic = reduced.get(&(i - n as i64)).cloned().unwrap_or_else(Basic::zero);
                    reduced.insert(i - n as i64, add(&[old, mul(&[c, p_x.clone()])?])?);
                }
            }
        }
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in &reduced {
            terms.push(mul(&[c.clone(), pw(big_y, &int(*i))?])?);
        }
        let reduced: Basic = expand(&add(&terms)?)?;
        crate::polys::polytools::factor_list(&reduced, &[big_x.clone(), big_y.clone()])
    };
    let Some((coefficient, factors)) = caught(attempt())? else {
        return log(p);
    };
    let mut total: Vec<Basic> = Vec::new();
    for (factor, multiplicity) in factors {
        total.push(mul(&[multiplicity, log(&factor)?])?);
    }
    if coefficient.has(big_x) || coefficient.has(big_y) {
        total.push(log(&coefficient)?);
    }
    add(&total)
}

/// The integrand as `sum A_k y**k` with `y = P**(1/n)`, `n >= 3`, `P`
/// squarefree (`_root_components`).
fn root_components(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, i64, Vec<Basic>)>> {
    let mut bases: Vec<(Basic, Int)> = Vec::new();
    let xs: [Basic; 1] = [x.clone()];
    for node in pows(f) {
        let Some(q) = node.args()[1].as_rational() else {
            continue;
        };
        if crate::numbers::denom(q) > Int::ONE && node.has(x) {
            let base: &Basic = &node.args()[0];
            if !is_polynomial(base, &xs) {
                return Ok(None);
            }
            let den: Int = crate::numbers::denom(q);
            match bases.iter_mut().find(|(b, _)| b == base) {
                Some(entry) => entry.1 = crate::numbers::lcm(&entry.1, &den),
                None => bases.push((base.clone(), den)),
            }
        }
    }
    if bases.len() != 1 {
        return Ok(None);
    }
    let (base, n) = bases.swap_remove(0);
    let n: i64 = i64::try_from(&n).unwrap_or(0);
    if n < 3 {
        return Ok(None);
    }
    let parameters: Vec<Basic> = symbols(f).into_iter().filter(|s| s != x).collect();
    let squarefree = if !parameters.is_empty() {
        caught(
            parametric(&parameters)
                .from_expr(&base, x)
                .map(|p| (p.degree(), p.is_squarefree())),
        )?
    } else {
        caught(rationals().from_expr(&base, x).map(|p| (p.degree(), p.is_squarefree())))?
    };
    let Some((degree, is_squarefree)) = squarefree else {
        return Ok(None);
    };
    if degree < 1 || !is_squarefree {
        return Ok(None);
    }
    let y: Basic = fresh("y", false, false);
    let to_y = |node: &Basic| -> R {
        if is_pow(node) && node.args()[0] == base {
            if let Some(q) = node.args()[1].as_rational() {
                let p_: i64 = i64::try_from(crate::numbers::numer(q)).unwrap_or(0);
                let q_: i64 = i64::try_from(crate::numbers::denom(q)).unwrap_or(1);
                return pw(&y, &int((p_ * n).div_euclid(q_)));
            }
        }
        Ok(node.clone())
    };
    let g: Basic = crate::core::traversal::bottom_up(f, &to_y, false)?;
    if !is_rational_function(&g, &[x.clone(), y.clone()]) {
        return Ok(None);
    }
    let (numerator, denominator) = fraction(&cancel(&g)?)?;
    let mut field_parameters: Vec<Basic> = vec![x.clone()];
    field_parameters.extend(parameters.iter().cloned());
    let kf: Field<Parametric> = parametric(&field_parameters);
    let reduced = caught((|| -> PolyResult<DensePoly<Parametric>> {
        let modulus = kf.from_expr(&sub(&pw(&y, &int(n))?, &base)?, &y)?;
        let inv = inverse(&kf.from_expr(&denominator, &y)?, &modulus)?;
        kf.from_expr(&numerator, &y)?.mul(&inv)?.rem(&modulus)
    })())?;
    let Some(reduced) = reduced else {
        return Ok(None);
    };
    let mut components: Vec<Basic> = Vec::new();
    for k in 0..n as usize {
        components.push(cancel(&kf.expr(&reduced.coeff(k))?)?);
    }
    Ok(Some((base, n, components)))
}
