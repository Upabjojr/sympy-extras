//! Port of `sympy_extras/integrals/telescoping.py`: the Almkvist–Zeilberger
//! algorithm (differential creative telescoping) for hyperexponential
//! integrands, the differential equation of a parametric integral with its
//! boundary values, and its solution with the constants fixed by values of
//! the integral (`holonomic_integral`).
//!
//! The rational functions of `x` have their coefficients in `QQ_I(t, ...)`
//! ([`FunctionField`]), factored over `QQ` or over `QQ_I` as SymPy factors
//! them.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp, UndefinedFunction};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult, NOT_PORTED_PREFIX};
use crate::integrals::conditions::ConditionalValue;
use crate::numbers::{Int, Rat};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::gaussiandomains::{GaussianRational, GaussianRationalField, QQ_I};
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;
use crate::polys::polyclasses::DensePoly;
use crate::polys::rings::PolyElement;

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;

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
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    if n == 0 {
        return Ok(e.clone());
    }
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn together(e: &Basic) -> R {
    crate::polys::rationaltools::together(e, false, true)
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn is_rational_function(e: &Basic, symbols: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, symbols, crate::core::expr::PolynomialKind::Rational) == Some(true)
}
fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}
fn integer_of(e: &Basic) -> Option<i64> {
    if !matches!(e.head(), Head::Rational { value, .. } if value.is_int()) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn function_of(name: &str, t: &Basic) -> R {
    Arc::new(UndefinedFunction::new(name, None, &[])).apply(vec![t.clone()])
}
fn has_head(e: &Basic, pred: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(pred)
}

/// The trigonometric and hyperbolic functions written as exponentials
/// (`_TRIGONOMETRIC`).
const TRIGONOMETRIC: [Function; 12] = [
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];

fn has_trigonometric(e: &Basic) -> bool {
    has_head(e, &|n| TRIGONOMETRIC.iter().any(|f| n.is_function(*f)))
}

/// What the polynomial layer raises on an expression it cannot take
/// (`_POLYNOMIAL_FAILURES`).
pub(crate) fn polynomial_failure(e: &PolyError) -> bool {
    match e {
        PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_) | PolyError::ZeroDivision(_) => true,
        PolyError::NotImplemented(m) => !m.starts_with(NOT_PORTED_PREFIX),
        _ => false,
    }
}

/// The output of the Almkvist–Zeilberger algorithm for `F(x, t)`
/// (`DifferentialTelescoper`).
#[derive(Clone, Debug)]
pub struct DifferentialTelescoper {
    pub term: Basic,
    pub x: Basic,
    pub t: Basic,
    pub coefficients: Vec<Basic>,
    pub certificate: Basic,
}

impl DifferentialTelescoper {
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// `sum_j a_j(t) I^(j)(t)` (`operator`).
    pub fn operator(&self, name: &str) -> R {
        let f: Basic = function_of(name, &self.t)?;
        let mut terms: Vec<Basic> = Vec::new();
        for (j, a) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[a.clone(), diff(&f, &self.t, j as u64)?])?);
        }
        add(&terms)
    }

    /// Verify `sum_j a_j d^j F/dt^j = d(R F)/dx` (`check`).
    pub fn check(&self) -> PolyResult<bool> {
        if self.check_by_terms()? {
            return Ok(true);
        }
        let f: &Basic = &self.term;
        let mut terms: Vec<Basic> = Vec::new();
        for (j, a) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[a.clone(), diff(f, &self.t, j as u64)?])?);
        }
        let lhs: Basic = add(&terms)?;
        let mut difference: Basic = div(
            &sub(&lhs, &diff(&mul(&[self.certificate.clone(), f.clone()])?, &self.x, 1)?)?,
            f,
        )?;
        if has_trigonometric(&difference) {
            difference = crate::core::rewriting::rewrite(&difference, &[], "exp", &Default::default())?;
        }
        let difference: Basic = cancel(&expand(&difference)?)?;
        Ok(difference == Basic::zero() || simplify(&difference)? == Basic::zero())
    }

    /// The identity proved on hyperexponential terms (`_check_by_terms`).
    fn check_by_terms(&self) -> PolyResult<bool> {
        let result = (|| -> PolyResult<bool> {
            let (x, t) = (&self.x, &self.t);
            let xt: [Basic; 2] = [x.clone(), t.clone()];
            let mut g: Basic = mul(&[self.certificate.clone(), self.term.clone()])?;
            if !is_rational_function(&self.certificate, &xt) {
                g = cancel(&g)?;
            }
            let mut groups: Vec<(Basic, Basic)> = Vec::new();
            let collect = |groups: &mut Vec<(Basic, Basic)>, term: &Basic, c: &Basic| -> PolyResult<()> {
                for (s, total) in groups.iter_mut() {
                    let ratio: Basic = cancel(&div(term, s)?)?;
                    if is_rational_function(&ratio, &xt) {
                        *total = add(&[total.clone(), mul(&[ratio, c.clone()])?])?;
                        return Ok(());
                    }
                }
                groups.push((term.clone(), c.clone()));
                Ok(())
            };
            for term in terms(&self.term, x, t)? {
                let Some(p) = logarithmic_derivative(&term, t)? else {
                    return Ok(false);
                };
                let mut pj: Basic = Basic::one();
                let mut total: Basic = self.coefficients[0].clone();
                for a in &self.coefficients[1..] {
                    pj = cancel(&add(&[diff(&pj, t, 1)?, mul(&[p.clone(), pj.clone()])?])?)?;
                    total = add(&[total, mul(&[a.clone(), pj.clone()])?])?;
                }
                collect(&mut groups, &term, &total)?;
            }
            for u in terms(&g, x, t)? {
                let Some(q) = logarithmic_derivative(&u, x)? else {
                    return Ok(false);
                };
                collect(&mut groups, &u, &neg(&q)?)?;
            }
            if groups.is_empty() {
                return Ok(false);
            }
            for (_, total) in &groups {
                if cancel(total)? != Basic::zero() {
                    return Ok(false);
                }
            }
            Ok(true)
        })();
        match result {
            Ok(v) => Ok(v),
            Err(e) if polynomial_failure(&e) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// `diff(F, x)/F` as a rational function of `x`, or `None`
/// (`logarithmic_derivative`).
pub fn logarithmic_derivative(f: &Basic, x: &Basic) -> R2 {
    if *f == Basic::zero() {
        return Ok(None);
    }
    let ratio = (|| -> R {
        let mut ratio: Basic = cancel(&div(&diff(f, x, 1)?, f)?)?;
        if has_head(&ratio, &|n| n.is_function(Function::Exp)) && !is_rational_function(&ratio, std::slice::from_ref(x))
        {
            // powers written as exponentials of logarithms
            ratio = cancel(&expand(&ratio)?)?;
        }
        Ok(ratio)
    })();
    let ratio: Basic = match ratio {
        Ok(r) => r,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    if !is_rational_function(&ratio, std::slice::from_ref(x)) {
        return Ok(None);
    }
    Ok(Some(ratio))
}

/// Whether both logarithmic derivatives of `F` are rational functions of
/// `x` and `t` (`is_hyperexponential`).
pub fn is_hyperexponential(f: &Basic, x: &Basic, t: &Basic) -> PolyResult<bool> {
    let xt: [Basic; 2] = [x.clone(), t.clone()];
    let q: Option<Basic> = logarithmic_derivative(f, x)?;
    let p: Option<Basic> = logarithmic_derivative(f, t)?;
    Ok(match (q, p) {
        (Some(q), Some(p)) => is_rational_function(&q, &xt) && is_rational_function(&p, &xt),
        _ => false,
    })
}

// ---------------------------------------------------------------------------
// rational functions of x over QQ_I(t, ...)

/// An element of the field of the coefficients (`Element`).
pub type Element = SparseFracElement<GaussianRationalField>;
/// A polynomial in `x` over it (`KPoly`).
pub type KPoly = DensePoly<FunctionField>;

/// The field `QQ_I(y1, ..., yk)` of the coefficients with the
/// factorisation of the polynomials over it in one more variable
/// (`_FunctionField`).
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionField {
    pub field: SparseFractionField<GaussianRationalField>,
    pub variables: Vec<String>,
    pub gaussian: bool,
}

impl FunctionField {
    fn new(names: &[String], gaussian: bool) -> FunctionField {
        FunctionField {
            field: SparseFractionField::new(SparsePolynomialRing::sympy(QQ_I, names.to_vec())),
            variables: names.to_vec(),
            gaussian,
        }
    }
}

impl Domain for FunctionField {
    type Elem = Element;

    fn name(&self) -> String {
        self.field.name()
    }
    fn is_field(&self) -> bool {
        true
    }
    fn is_ordered(&self) -> bool {
        self.field.is_ordered()
    }
    fn characteristic(&self) -> Int {
        self.field.characteristic()
    }
    fn zero(&self) -> Element {
        self.field.zero()
    }
    fn one(&self) -> Element {
        self.field.one()
    }
    fn to_expr(&self, a: &Element) -> R {
        self.field.to_expr(a)
    }
    fn from_expr(&self, e: &Basic) -> PolyResult<Element> {
        self.field.from_expr(e)
    }
    fn is_zero(&self, a: &Element) -> bool {
        self.field.is_zero(a)
    }
    fn is_one(&self, a: &Element) -> bool {
        self.field.is_one(a)
    }
    fn is_negative(&self, a: &Element) -> bool {
        self.field.is_negative(a)
    }
    fn is_unit(&self, a: &Element) -> bool {
        self.field.is_unit(a)
    }
    fn add(&self, a: &Element, b: &Element) -> Element {
        self.field.add(a, b)
    }
    fn sub(&self, a: &Element, b: &Element) -> Element {
        self.field.sub(a, b)
    }
    fn neg(&self, a: &Element) -> Element {
        self.field.neg(a)
    }
    fn mul(&self, a: &Element, b: &Element) -> Element {
        self.field.mul(a, b)
    }
    fn pow(&self, a: &Element, n: u64) -> Element {
        self.field.pow(a, n)
    }
    fn pow_signed(&self, a: &Element, n: i64) -> PolyResult<Element> {
        self.field.pow_signed(a, n)
    }
    fn exquo(&self, a: &Element, b: &Element) -> PolyResult<Element> {
        self.field.exquo(a, b)
    }
    fn quo(&self, a: &Element, b: &Element) -> PolyResult<Element> {
        self.field.quo(a, b)
    }
    fn rem(&self, a: &Element, b: &Element) -> PolyResult<Element> {
        self.field.rem(a, b)
    }
    fn invert(&self, a: &Element) -> PolyResult<Element> {
        self.field.invert(a)
    }
    fn gcd(&self, a: &Element, b: &Element) -> Element {
        self.field.gcd(a, b)
    }
    fn lcm(&self, a: &Element, b: &Element) -> Element {
        self.field.lcm(a, b)
    }
    fn gcdex(&self, a: &Element, b: &Element) -> PolyResult<(Element, Element, Element)> {
        self.field.gcdex(a, b)
    }
    fn canonical_unit(&self, a: &Element) -> Element {
        self.field.canonical_unit(a)
    }
    fn size(&self, a: &Element) -> u64 {
        self.field.size(a)
    }
    fn from_int(&self, n: i64) -> Element {
        self.field.from_int(n)
    }
    fn from_integer(&self, n: &Int) -> Element {
        self.field.from_integer(n)
    }
    fn to_str(&self, a: &Element) -> String {
        self.field.to_str(a)
    }

    /// Over `QQ` (the factors of the numerator over `QQ[x, y]` which depend
    /// on `x`) unless `gaussian`, over `QQ_I` then (`factor_dense`).
    fn factor_dense(&self, f: &[Element]) -> PolyResult<(Element, Vec<(Vec<Element>, usize)>)> {
        if f.is_empty() {
            return Ok((self.zero(), Vec::new()));
        }
        let ring = &self.field.ring;
        let mut common = ring.one();
        for c in f {
            common = ring.lcm(&common, &c.denom);
        }
        let n: usize = f.len() - 1;
        let mut terms: BTreeMap<Vec<u32>, GaussianRational> = BTreeMap::new();
        for (i, c) in f.iter().enumerate() {
            let scaled = c.numer.mul(&common.exquo(&c.denom)?);
            for (m, value) in &scaled.terms {
                let mut monomial: Vec<u32> = vec![(n - i) as u32];
                monomial.extend(m.iter().copied());
                terms.insert(monomial, value.clone());
            }
        }
        let mut variables: Vec<String> = vec!["_kfx".to_string()];
        variables.extend(self.variables.iter().cloned());
        let real: bool = terms.values().all(|c| c.y == Rat::ZERO);
        let found: Vec<(BTreeMap<Vec<u32>, GaussianRational>, usize)> = if self.gaussian || !real {
            let g_ring =
                crate::polys::rings::PolyRing::new(QQ_I, variables.clone(), crate::polys::rings::MonomialOrder::Lex)?;
            let element: PolyElement<GaussianRationalField> = g_ring.from_dict(terms.into_iter().collect());
            let (_, factors) = crate::polys::gaussianfactor::gaussian_factor_list(&element, false)?;
            factors
                .into_iter()
                .map(|(g, k)| (g.terms.clone().into_iter().collect(), k))
                .collect()
        } else {
            let rational: Vec<(Vec<u32>, Rat)> = terms.iter().map(|(m, c)| (m.clone(), c.x.clone())).collect();
            let gens: Vec<Basic> = variables.iter().map(|v| symbol(v)).collect();
            let poly = crate::polys::polytools::Poly::from_dict(&rational, &gens)?;
            let (_, poly_factors) = poly.factor_list()?;
            poly_factors
                .into_iter()
                .map(|(g, k)| {
                    let gaussian: BTreeMap<Vec<u32>, GaussianRational> = g
                        .rep
                        .terms
                        .iter()
                        .map(|(m, c)| {
                            (
                                m.clone(),
                                GaussianRational {
                                    x: c.clone(),
                                    y: Rat::ZERO,
                                },
                            )
                        })
                        .collect();
                    (gaussian, k)
                })
                .collect()
        };
        let mut result: Vec<(Vec<Element>, usize)> = Vec::new();
        let mut product: Element = self.one();
        for (g, k) in found {
            let degree: u32 = g.keys().map(|m| m[0]).max().unwrap_or(0);
            if degree == 0 {
                continue;
            }
            let mut by_degree: BTreeMap<u32, Vec<(Vec<u32>, GaussianRational)>> = BTreeMap::new();
            for (m, value) in &g {
                by_degree
                    .entry(m[0])
                    .or_default()
                    .push((m[1..].to_vec(), value.clone()));
            }
            let mut dense: Vec<Element> = Vec::new();
            for i in 0..=degree {
                let entries = by_degree.get(&(degree - i)).cloned().unwrap_or_default();
                dense.push(
                    self.field
                        .from_ring(&ring.ring.from_dict(entries.into_iter().collect())),
                );
            }
            product = self.mul(&product, &self.pow(&dense[0], k as u64));
            result.push((dense, k));
        }
        Ok((self.quo(&f[0], &product)?, result))
    }
}

/// The generators of `e` as a rational function over `QQ_I`
/// (`_generators`).
fn generators(e: &Basic, x: &Basic, found: &mut Vec<Basic>) -> PolyResult<()> {
    if e.is_add() || e.is_mul() {
        for a in e.args() {
            if !crate::core::logic::is_boolean(a) {
                generators(a, x, found)?;
            }
        }
        return Ok(());
    }
    if e.is_pow() && integer_of(&e.args()[1]).is_some() {
        return generators(&e.args()[0], x, found);
    }
    if matches!(e.head(), Head::Rational { .. } | Head::ImaginaryUnit) {
        return Ok(());
    }
    if e == x {
        return Ok(());
    }
    if e.is_number_node() || e.has(x) {
        return Err(PolyError::Value(format!("{} is not a rational function of {}", e, x)));
    }
    if !found.contains(e) {
        found.push(e.clone());
    }
    Ok(())
}

/// Rational functions of `x` with their coefficients in the field of the
/// generators of the given expressions (`_Context`).
#[derive(Clone, Debug)]
pub(crate) struct Context {
    pub x: Basic,
    pub generators: Vec<Basic>,
    pub gaussian: bool,
    renamed: HashMap<Basic, Basic>,
    restored: HashMap<Basic, Basic>,
    pub variable: Basic,
    pub field: FunctionField,
}

impl Context {
    pub(crate) fn new(exprs: &[Basic], x: &Basic, given: &[Basic], gaussian: bool) -> PolyResult<Context> {
        let mut found: Vec<Basic> = given.to_vec();
        for e in exprs {
            generators(e, x, &mut found)?;
        }
        if given.is_empty() {
            found.sort_by_cached_key(crate::core::sorting::sort_key);
        }
        let i_unit: Basic = Basic::atom(Head::ImaginaryUnit);
        let gaussian: bool = gaussian || exprs.iter().any(|e| e.has(&i_unit));
        let mut names: Vec<String> = (0..found.len()).map(|i| format!("_k{}", i)).collect();
        if names.is_empty() {
            names.push("_k".to_string());
        }
        let mut renamed: HashMap<Basic, Basic> =
            found.iter().zip(&names).map(|(g, n)| (g.clone(), symbol(n))).collect();
        let mut restored: HashMap<Basic, Basic> =
            found.iter().zip(&names).map(|(g, n)| (symbol(n), g.clone())).collect();
        let variable: Basic = symbol("_kx");
        renamed.insert(x.clone(), variable.clone());
        restored.insert(variable.clone(), x.clone());
        Ok(Context {
            x: x.clone(),
            generators: found,
            gaussian,
            renamed,
            restored,
            variable,
            field: FunctionField::new(&names, gaussian),
        })
    }

    /// Whether the expressions are rational functions over the field
    /// (`covers`).
    pub(crate) fn covers(&self, exprs: &[Basic]) -> PolyResult<bool> {
        let mut found: Vec<Basic> = self.generators.clone();
        for e in exprs {
            generators(e, &self.x, &mut found)?;
        }
        Ok(found.len() == self.generators.len())
    }

    /// The context of the generators of `exprs` too (`extended`).
    pub(crate) fn extended(&self, exprs: &[Basic], gaussian: bool) -> PolyResult<Context> {
        let mut found: Vec<Basic> = self.generators.clone();
        for e in exprs {
            generators(e, &self.x, &mut found)?;
        }
        Context::new(exprs, &self.x, &found, self.gaussian || gaussian)
    }

    /// A polynomial of the context `other` in this one (`convert`).
    pub(crate) fn convert(&self, p: &KPoly, other: &Context) -> PolyResult<KPoly> {
        if std::ptr::eq(self, other) {
            return Ok(p.clone());
        }
        self.poly(&other.poly_expr(p)?)
    }

    fn internal(&self, e: &Basic) -> Basic {
        e.xreplace(&self.renamed)
    }

    pub(crate) fn restore(&self, e: &Basic) -> R {
        crate::core::function::xreplace_evaluated(e, &self.restored)
    }

    /// An expression free of `x` as an element of the field (`element`).
    pub(crate) fn element(&self, e: &Basic) -> PolyResult<Element> {
        self.field.from_expr(&self.internal(e))
    }

    pub(crate) fn element_expr(&self, c: &Element) -> R {
        self.restore(&self.field.to_expr(c)?)
    }

    /// A polynomial in `x` with coefficients in the field (`poly`).
    pub(crate) fn poly(&self, e: &Basic) -> PolyResult<KPoly> {
        let name: String = self.variable.to_string();
        let ring = crate::polys::domains::polynomialring::PolynomialRing::new(self.field.clone(), &name);
        let rep: Vec<Element> = ring.from_expr(&expand(&self.internal(e))?)?;
        Ok(DensePoly::new(
            &rep,
            self.field.clone(),
            crate::core::Symbol::new(&name, &[]),
        ))
    }

    pub(crate) fn from_elements(&self, coefficients: &[Element]) -> KPoly {
        let name: String = self.variable.to_string();
        DensePoly::new(coefficients, self.field.clone(), crate::core::Symbol::new(&name, &[]))
    }

    pub(crate) fn constant(&self, c: &Element) -> KPoly {
        self.from_elements(std::slice::from_ref(c))
    }

    pub(crate) fn poly_expr(&self, p: &KPoly) -> R {
        let name: String = self.variable.to_string();
        let ring = crate::polys::domains::polynomialring::PolynomialRing::new(self.field.clone(), &name);
        self.restore(&ring.to_expr(&p.rep)?)
    }

    /// Numerator and denominator of a rational function of `x`
    /// (`fraction`).
    pub(crate) fn fraction(&self, e: &Basic) -> PolyResult<(KPoly, KPoly)> {
        let (numerator, denominator) = numer_denom(&together(e)?)?;
        Ok((self.poly(&numerator)?, self.poly(&denominator)?))
    }

    pub(crate) fn fraction_expr(&self, numerator: &KPoly, denominator: &KPoly) -> R {
        cancel(&div(&self.poly_expr(numerator)?, &self.poly_expr(denominator)?)?)
    }
}

pub(crate) fn context(exprs: &[Basic], x: &Basic) -> PolyResult<Option<Context>> {
    match Context::new(exprs, x, &[], false) {
        Ok(c) => Ok(Some(c)),
        Err(PolyError::Value(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// The canonical basis of the null space of a matrix of rational
/// functions over the field of the generators of the entries, `None` when
/// an entry is not a rational function (`_nullspace`).
pub(crate) fn nullspace(rows: &[Vec<Basic>], columns: usize) -> PolyResult<Option<Vec<Vec<Basic>>>> {
    let entries: Vec<Basic> = rows.iter().flatten().cloned().collect();
    let Some(k) = context(&entries, &dummy("_kn"))? else {
        return Ok(None);
    };
    let converted = rows
        .iter()
        .map(|row| row.iter().map(|e| k.element(e)).collect::<PolyResult<Vec<Element>>>())
        .collect::<PolyResult<Vec<Vec<Element>>>>();
    let converted: Vec<Vec<Element>> = match converted {
        Ok(c) => c,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let basis: Vec<Vec<Basic>> = k
        .field
        .nullspace(&converted, columns)?
        .iter()
        .map(|vector| vector.iter().map(|v| k.element_expr(v)).collect::<PolyResult<_>>())
        .collect::<PolyResult<_>>()?;
    Ok(Some(basis))
}

fn positive_integer(e: &Basic) -> PolyResult<Option<i64>> {
    Ok(integer_of(&cancel(e)?).filter(|v| *v > 0))
}

/// `(factor, multiplicity)` for the monic irreducible factors of `P`
/// which depend on the variable (`_irreducibles`).
fn irreducibles(p: &KPoly) -> PolyResult<Vec<(KPoly, usize)>> {
    let (_, factors) = p.factor_list()?;
    let mut result: Vec<(KPoly, usize)> = Vec::new();
    for (d, k) in factors {
        if d.degree() > 0 {
            result.push((d.monic()?, k));
        }
    }
    Ok(result)
}

/// The polynomial `D` the denominator of `R` divides (`_denominator_bound`).
fn denominator_bound(k: &Context, a: &KPoly, b: &KPoly, denominators: &[KPoly]) -> PolyResult<KPoly> {
    // factor -> (order in B, max order in the p_j)
    let mut factors: Vec<(KPoly, usize, usize)> = irreducibles(b)?.into_iter().map(|(d, m)| (d, m, 0)).collect();
    for l in denominators {
        for (d, m) in irreducibles(l)? {
            match factors.iter_mut().find(|(e, _, _)| *e == d) {
                Some(entry) => entry.2 = entry.2.max(m),
                None => factors.push((d, 0, m)),
            }
        }
    }
    let mut result: KPoly = k.constant(&k.field.one());
    for (d, order_b, order_s) in factors {
        let mut m: i64;
        if order_b <= 1 {
            m = order_s as i64 - 1;
            if order_b == 1 {
                // a simple pole of q with a positive integer residue rho
                // allows a pole of order rho of R
                let cofactor: KPoly = b.quo(&d)?;
                let residue = (|| -> PolyResult<KPoly> { a.mul(&cofactor.invert(&d)?)?.rem(&d) })();
                let residue: Option<KPoly> = match residue {
                    Ok(r) => Some(r),
                    Err(e) if polynomial_failure(&e) => None,
                    Err(e) => return Err(e),
                };
                if let Some(residue) = residue.filter(|r| r.degree() <= 0) {
                    let value: Basic = if residue.is_zero() {
                        Basic::zero()
                    } else {
                        k.element_expr(&residue.LC())?
                    };
                    if let Some(rho) = positive_integer(&value)? {
                        m = m.max(rho);
                    }
                }
            }
        } else {
            m = order_s as i64 - order_b as i64;
        }
        if m > 0 {
            result = result.mul(&d.pow(m as usize))?;
        }
    }
    Ok(result)
}

/// The bound on the degree of the numerator `N` of `R = N/D`
/// (`_numerator_degree`).
fn numerator_degree(k: &Context, a: &KPoly, b: &KPoly, d: &KPoly, degree_s: i64) -> PolyResult<i64> {
    let degree_q: Option<i64> = if a.is_zero() {
        None
    } else {
        Some(a.degree() as i64 - b.degree() as i64)
    };
    let dd: i64 = d.degree() as i64;
    let n: i64 = match degree_q {
        Some(q) if q >= 0 => degree_s + dd - q,
        Some(-1) => {
            let mut n: i64 = degree_s + dd + 1;
            // R' + q R ~ (n - deg D + rho_oo) x^(n - deg D - 1) may cancel
            let rho: Basic = k.element_expr(&k.field.quo(&a.LC(), &b.LC())?)?;
            if let Some(zero) = positive_integer(&neg(&rho)?)? {
                n = n.max(dd + zero);
            } else if rho == Basic::zero() {
                n = n.max(dd);
            }
            n
        }
        _ => degree_s + dd + 1,
    };
    Ok(n.max(0))
}

type Telescoped = (Vec<Basic>, Basic);

/// `(a_0, ..., a_J, R)` for the given order `J`, or `None` (`_telescope`).
fn telescope(q: &Basic, p: &Basic, x: &Basic, t: &Basic, order: usize) -> PolyResult<Option<Telescoped>> {
    let mut ps: Vec<Basic> = vec![Basic::one()];
    for _ in 0..order {
        let last: Basic = ps.last().expect("nonempty").clone();
        ps.push(cancel(&add(&[diff(&last, t, 1)?, mul(&[p.clone(), last])?])?)?);
    }
    let mut exprs: Vec<Basic> = vec![q.clone()];
    exprs.extend(ps.iter().cloned());
    let Some(k) = context(&exprs, x)? else {
        return Ok(None);
    };
    let fractions = (|| -> PolyResult<((KPoly, KPoly), Vec<(KPoly, KPoly)>)> {
        let ab = k.fraction(q)?;
        let fs: Vec<(KPoly, KPoly)> = ps.iter().map(|pj| k.fraction(pj)).collect::<PolyResult<_>>()?;
        Ok((ab, fs))
    })();
    let ((a, b), fractions) = match fractions {
        Ok(v) => v,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let denominators: Vec<KPoly> = fractions.iter().map(|(_, l)| l.clone()).collect();
    let d: KPoly = denominator_bound(&k, &a, &b, &denominators)?;
    let degree_s: i64 = fractions
        .iter()
        .filter(|(pp, _)| !pp.is_zero())
        .map(|(pp, l)| pp.degree() as i64 - l.degree() as i64)
        .max()
        .unwrap_or(0);
    let n: i64 = numerator_degree(&k, &a, &b, &d, degree_s)?;
    let mut l: KPoly = k.constant(&k.field.one());
    for (_, lj) in &fractions {
        l = l.lcm(lj)?;
    }
    // B L (N' D - N D') + A L N D = B D^2 sum_j a_j (L / L_j) P_j
    let mut columns: Vec<KPoly> = Vec::new();
    let x_: KPoly = k.from_elements(&[k.field.one(), k.field.zero()]);
    let d_prime: KPoly = d.diff(1);
    let bl: KPoly = b.mul(&l)?;
    let al: KPoly = a.mul(&l)?;
    for i in 0..=n {
        let monomial: KPoly = x_.pow(i as usize);
        let derivative: KPoly = if i != 0 {
            k.from_elements(&[k.field.from_int(i)]).mul(&x_.pow((i - 1) as usize))?
        } else {
            k.constant(&k.field.zero())
        };
        let inner: KPoly = derivative.mul(&d)?.sub(&monomial.mul(&d_prime)?)?;
        columns.push(bl.mul(&inner)?.add(&al.mul(&monomial)?.mul(&d)?)?);
    }
    for (pp, lj) in &fractions {
        columns.push(b.mul(&d)?.mul(&d)?.mul(&l.quo(lj)?)?.mul(pp)?.neg());
    }
    let height: isize = columns.iter().map(|c| c.degree()).max().unwrap_or(-1) + 1;
    if height <= 0 {
        return Ok(None);
    }
    let height: usize = height as usize;
    let rows: Vec<Vec<Element>> = (0..height)
        .map(|r| {
            columns
                .iter()
                .map(|c| {
                    let index: isize = height as isize - 1 - r as isize;
                    if index >= 0 && index <= c.degree() {
                        c.coeff(index as usize)
                    } else {
                        k.field.zero()
                    }
                })
                .collect()
        })
        .collect();
    let n1: usize = (n + 1) as usize;
    for vector in k.field.nullspace(&rows, columns.len())? {
        let a_values: Vec<Basic> = vector[n1..]
            .iter()
            .map(|v| k.element_expr(v))
            .collect::<PolyResult<_>>()?;
        if a_values.iter().all(|v| *v == Basic::zero()) {
            continue;
        }
        let n_coefficients: Vec<Element> = vector[..n1].iter().rev().cloned().collect();
        let n_value: KPoly = k.from_elements(&n_coefficients);
        let r: Basic = k.fraction_expr(&n_value, &d)?;
        let (normalized_values, factor) = normalized(&a_values, t)?;
        return Ok(Some((normalized_values, cancel(&mul(&[r, factor])?)?)));
    }
    Ok(None)
}

/// The coefficients cleared of denominators and common factors, and the
/// factor they were multiplied by (`_normalized`).
pub(crate) fn normalized(values: &[Basic], t: &Basic) -> PolyResult<(Vec<Basic>, Basic)> {
    let denominators: Vec<Basic> = values
        .iter()
        .map(|v| Ok(numer_denom(&together(v)?)?.1))
        .collect::<PolyResult<_>>()?;
    let attempt_scale = (|| -> PolyResult<(Basic, Vec<Basic>, Basic)> {
        let scale: Basic = crate::polys::polytools::lcm_list(&denominators, &[])?;
        let scaled: Vec<Basic> = values
            .iter()
            .map(|v| cancel(&mul(&[v.clone(), scale.clone()])?))
            .collect::<PolyResult<_>>()?;
        let nonzero: Vec<Basic> = scaled.iter().filter(|v| **v != Basic::zero()).cloned().collect();
        let common: Basic = crate::polys::polytools::gcd_list(&nonzero, &[])?;
        Ok((scale, scaled, common))
    })();
    let (scale, scaled, common) = match attempt_scale {
        Ok(v) => v,
        Err(e) if polynomial_failure(&e) => (Basic::one(), values.to_vec(), Basic::one()),
        Err(e) => return Err(e),
    };
    let mut result: Vec<Basic> = scaled
        .iter()
        .map(|v| cancel(&div(v, &common)?))
        .collect::<PolyResult<_>>()?;
    let mut factor: Basic = cancel(&div(&scale, &common)?)?;
    // a positive leading coefficient of the highest order term
    let lead: Basic = result.last().expect("nonempty").clone();
    let mut negative: bool = is(&lead, Predicate::negative) == Some(true);
    if !negative && !crate::core::assumptions::is_number(&lead) && is_polynomial(&lead, t) {
        negative = match crate::polys::polytools::Poly::new(&lead, std::slice::from_ref(t)).and_then(|p| p.LC()) {
            Ok(lc) => is(&lc, Predicate::negative) == Some(true),
            Err(e) if polynomial_failure(&e) => false,
            Err(e) => return Err(e),
        };
    }
    if negative {
        result = result.iter().map(neg).collect::<PolyResult<_>>()?;
        factor = neg(&factor)?;
    }
    Ok((result, factor))
}

/// `F` as a sum of hyperexponential terms, or `[]` (`_terms`).
pub(crate) fn terms(f: &Basic, x: &Basic, t: &Basic) -> PolyResult<Vec<Basic>> {
    let mut g: Basic = f.clone();
    if has_trigonometric(&g) {
        g = crate::core::rewriting::rewrite(&g, &[], "exp", &Default::default())?;
    }
    let g: Basic = expand(&g)?;
    let found: Vec<Basic> = terms_of(&g);
    let mut all: bool = true;
    for term in &found {
        if !is_hyperexponential(term, x, t)? {
            all = false;
            break;
        }
    }
    if all {
        return Ok(found);
    }
    if is_hyperexponential(f, x, t)? {
        return Ok(vec![f.clone()]);
    }
    Ok(Vec::new())
}

/// The Almkvist–Zeilberger algorithm for a hyperexponential `F(x, t)`, or
/// a sum of hyperexponential terms sharing the same equation
/// (`almkvist_zeilberger`).
pub fn almkvist_zeilberger(
    f: &Basic,
    x: &Basic,
    t: &Basic,
    max_order: usize,
    min_order: usize,
) -> PolyResult<Option<DifferentialTelescoper>> {
    let found_terms: Vec<Basic> = terms(f, x, t)?;
    if found_terms.is_empty() {
        return Ok(None);
    }
    let mut found: Vec<(Vec<Basic>, Basic, Basic)> = Vec::new();
    for term in found_terms {
        let (Some(q), Some(p)) = (logarithmic_derivative(&term, x)?, logarithmic_derivative(&term, t)?) else {
            return Ok(None);
        };
        let mut result: Option<Telescoped> = None;
        for order in min_order..=max_order {
            result = telescope(&q, &p, x, t, order)?;
            if result.is_some() {
                break;
            }
        }
        let Some((coefficients, r)) = result else {
            return Ok(None);
        };
        found.push((coefficients, r, term));
    }
    let coefficients: Vec<Basic> = found[0].0.clone();
    for (other, _, _) in &found[1..] {
        if other.len() != coefficients.len() {
            return Ok(None);
        }
        for (a, b) in other.iter().zip(&coefficients) {
            if cancel(&sub(a, b)?)? != Basic::zero() {
                // the terms of the sum satisfy different equations
                return Ok(None);
            }
        }
    }
    let certificate: Basic = if found.len() == 1 {
        found[0].1.clone()
    } else {
        let mut parts: Vec<Basic> = Vec::new();
        for (_, r, term) in &found {
            parts.push(mul(&[r.clone(), term.clone()])?);
        }
        cancel(&div(&add(&parts)?, f)?)?
    };
    Ok(Some(DifferentialTelescoper {
        term: f.clone(),
        x: x.clone(),
        t: t.clone(),
        coefficients,
        certificate,
    }))
}

// ---------------------------------------------------------------------------
// the differential equation of the integral, and its solution

fn boundary(g: &Basic, x: &Basic, point: &Basic, direction: &str, assumptions: &[Basic]) -> R2 {
    let value: Option<Basic> =
        attempt(|| crate::assumptions::limits::limit(g, x, point, direction, assumptions, None))?;
    Ok(value.filter(|v| {
        !has_head(v, &|n| {
            matches!(
                n.head(),
                Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity | Head::Piecewise
            )
        }) && !v.has(x)
    }))
}

/// `sum_j a_j I^(j) = [G]_a^b`, `None` when a boundary value is not finite
/// (`_equation`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn equation(
    coefficients: &[Basic],
    g: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    assumptions: &[Basic],
    name: &str,
) -> R2 {
    let (Some(upper), Some(lower)) = (
        boundary(g, x, b, "-", assumptions)?,
        boundary(g, x, a, "+", assumptions)?,
    ) else {
        return Ok(None);
    };
    let f: Basic = function_of(name, t)?;
    let mut terms: Vec<Basic> = Vec::new();
    for (j, c) in coefficients.iter().enumerate() {
        terms.push(mul(&[c.clone(), diff(&f, t, j as u64)?])?);
    }
    let operator: Basic = add(&terms)?;
    Ok(Some(crate::core::relational::rel(
        RelOp::Eq,
        &operator,
        &simplify(&sub(&upper, &lower)?)?,
    )))
}

/// The linear differential equation satisfied by `Integral(F, (x, a, b))`
/// (`holonomic_ode`).
pub fn holonomic_ode(f: &Basic, x: &Basic, a: &Basic, b: &Basic, t: &Basic, assumptions: &[Basic], name: &str) -> R2 {
    let Some(telescoper) = almkvist_zeilberger(f, x, t, 4, 0)? else {
        return Ok(None);
    };
    let g: Basic = mul(&[telescoper.certificate.clone(), f.clone()])?;
    equation(&telescoper.coefficients, &g, x, a, b, t, assumptions, name)
}

/// `I^(j)(t0)` for `j < order` by integrating `d^j F/dt^j` at `t = t0`
/// (`_initial_values`).
#[allow(clippy::too_many_arguments)]
fn initial_values(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    t0: &Basic,
    order: usize,
    assumptions: &[Basic],
) -> PolyResult<Option<Vec<Basic>>> {
    let mut values: Vec<Basic> = Vec::new();
    let options = crate::integrals::definite::Options {
        special_values: false,
        ..Default::default()
    };
    for j in 0..order {
        let integrand: Basic = crate::core::subs::subs(&diff(f, t, j as u64)?, t, t0)?;
        let value: Option<Basic> =
            attempt(|| crate::integrals::definite::definite_integral(&integrand, x, a, b, assumptions, &options))?;
        let Some(value) = value else {
            return Ok(None);
        };
        let bad: bool = crate::solvers::linear_ode::has_integral(&value)
            || has_head(&value, &|n| {
                matches!(
                    n.head(),
                    Head::Piecewise | Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
                )
            });
        if bad {
            return Ok(None);
        }
        values.push(value);
    }
    Ok(Some(values))
}

type General = (Basic, Vec<Basic>);

/// The general solution of `sum_j a_j I^(j) = rhs` with its constants
/// (`_general_solution`).
pub(crate) fn general_solution(
    coefficients: &[Basic],
    rhs: &Basic,
    t: &Basic,
    eq: &Basic,
) -> PolyResult<Option<General>> {
    use crate::solvers::linear_ode::{dsolve_linear, has_integral, LinearOperator};
    let order: usize = coefficients.len() - 1;
    if order == 0 {
        return Ok(Some((cancel(&div(rhs, &coefficients[0])?)?, Vec::new())));
    }
    let y: Basic = function_of("_y", t)?;
    let mut terms: Vec<Basic> = Vec::new();
    for (j, c) in coefficients.iter().enumerate() {
        terms.push(mul(&[c.clone(), diff(&y, t, j as u64)?])?);
    }
    let homogeneous: Basic = add(&terms)?;
    let basis: Option<Vec<Basic>> = attempt(|| dsolve_linear(&homogeneous, &y, true, false, false))?;
    if let Some(basis) = basis.filter(|b| b.len() == order && !b.iter().any(has_integral)) {
        let l: LinearOperator = LinearOperator::new(coefficients, t);
        let particular: Option<Basic> = if *rhs != Basic::zero() {
            attempt(|| crate::solvers::factorization::variation_of_parameters(&l, &basis, rhs))?
        } else {
            Some(Basic::zero())
        };
        if let Some(particular) = particular.filter(|p| !has_integral(p)) {
            let constants: Vec<Basic> = (0..order).map(|j| dummy(&format!("_C{}", j + 1))).collect();
            let mut parts: Vec<Basic> = vec![particular];
            for (c, s) in constants.iter().zip(&basis) {
                parts.push(mul(&[c.clone(), s.clone()])?);
            }
            return Ok(Some((add(&parts)?, constants)));
        }
    }
    let unknown: Basic = function_of("I", t)?;
    Ok(attempt(|| crate::solvers::dsolve::explicit_solution(eq, &unknown))?.flatten())
}

/// The unique solution of linear equations in the constants
/// (`_solve_constants`).
pub(crate) fn solve_constants(equations: &[Basic], constants: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    if let Some(found) = crate::solvers::linsolve::solve_linear_unique(equations, constants)? {
        return Ok(Some(found));
    }
    let zeros: HashMap<Basic, Basic> = constants.iter().map(|c| (c.clone(), Basic::zero())).collect();
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for e in equations {
        let row: Vec<Basic> = constants.iter().map(|c| diff(e, c, 1)).collect::<PolyResult<_>>()?;
        let constant: Basic = crate::core::function::xreplace_evaluated(e, &zeros)?;
        if row.iter().any(|v| constants.iter().any(|c| v.has(c))) {
            return Ok(None);
        }
        let mut full: Vec<Basic> = row;
        full.push(neg(&constant)?);
        rows.push(full);
    }
    let (reduced, pivots) = match crate::solvers::recurr::row_reduce(&rows) {
        Ok(v) => v,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    if pivots != (0..constants.len()).collect::<Vec<usize>>() {
        return Ok(None);
    }
    for row in &reduced[pivots.len()..] {
        if !crate::solvers::recurr::is_zero(row.last().expect("nonempty"))? {
            return Ok(None);
        }
    }
    Ok(Some(
        (0..constants.len())
            .map(|i| reduced[i].last().expect("nonempty").clone())
            .collect(),
    ))
}

/// The solution of the equation of the integral with its constants fixed
/// by the initial values (`_solve`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    coefficients: &[Basic],
    eq: &Basic,
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    assumptions: &[Basic],
) -> R2 {
    let rhs: Basic = eq.args()[1].clone();
    let Some((solution, constants)) = general_solution(coefficients, &rhs, t, eq)? else {
        return Ok(None);
    };
    let order: usize = coefficients.len() - 1;
    if constants.len() != order {
        return Ok(None);
    }
    if constants.is_empty() {
        return Ok(Some(solution));
    }
    let points: [Basic; 4] = [int(1), int(2), Basic::half(), int(3)];
    for t0 in &points {
        let Some(values) = initial_values(f, x, a, b, t, t0, order, assumptions)? else {
            continue;
        };
        let mut equations: Vec<Basic> = Vec::new();
        for (j, v) in values.iter().enumerate() {
            equations.push(sub(
                &crate::core::subs::subs(&diff(&solution, t, j as u64)?, t, t0)?,
                v,
            )?);
        }
        let Some(solved) = attempt(|| solve_constants(&equations, &constants))?.flatten() else {
            continue;
        };
        let assignment: HashMap<Basic, Basic> = constants.iter().cloned().zip(solved).collect();
        return Ok(Some(crate::core::function::xreplace_evaluated(&solution, &assignment)?));
    }
    Ok(None)
}

/// Whether the numerical check (when it is on) does not reject the value
/// (`_verified`).
pub(crate) fn verified(
    value: &Basic,
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<bool> {
    if !crate::settings::settings().numerical_checks {
        return Ok(true);
    }
    Ok(crate::integrals::definite::verify_numerically(value, f, x, a, b, assumptions, 2, &[])? != Some(false))
}

/// `Integral(F, (x, a, b))` as a function of the parameter `t` by the
/// holonomic method (`holonomic_integral`).
pub fn holonomic_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    let Some(telescoper) = almkvist_zeilberger(f, x, t, 4, 0)? else {
        return Ok(None);
    };
    let g: Basic = mul(&[telescoper.certificate.clone(), f.clone()])?;
    let Some(eq) = equation(&telescoper.coefficients, &g, x, a, b, t, assumptions, "I")? else {
        return Ok(None);
    };
    let Some(solution) = solve(&telescoper.coefficients, &eq, f, x, a, b, t, assumptions)? else {
        return Ok(None);
    };
    let value: Basic = crate::integrals::marichev::tidy(&solution, assumptions, &crate::core::logic::true_())?;
    if !verified(&value, f, x, a, b, assumptions)? {
        return Ok(None);
    }
    Ok(Some(ConditionalValue::new(value, crate::core::logic::true_())))
}
