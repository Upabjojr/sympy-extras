//! Port of `sympy_extras/concrete/karr.py`: Karr's algorithm for
//! indefinite summation. The summand is analysed, a ΠΣ-field containing it
//! is built by adjoining a generator for every product or sum not already
//! expressible in the field ([`crate::concrete::pisigma`]), and the
//! telescoping equation `g(k + 1) - g(k) = f(k)` is solved there. The
//! isolated values of the parameters at which the closed form is undefined
//! get cases of their own ([`crate::special_values`]). The last resort of
//! [`summation`], creative telescoping (`concrete/creative`), is not
//! ported and raises not ported when it is reached.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::concrete::pisigma::{Constants, Kind, PiSigmaElement, PiSigmaField};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;

/// A point of the parameters: their values (`SumPoint`).
pub type SumPoint = Vec<(Basic, Basic)>;

/// The seconds given to the search for the poles of a summand at a point
/// (`_POLE_SECONDS`).
const POLE_SECONDS: f64 = 2.0;
/// The seconds given to the evaluation of a case at one sample point
/// (`_CHECK_SECONDS`).
const CHECK_SECONDS: f64 = 2.0;
/// The number of values of the upper limit (or of the index) checked
/// (`_CHECKS`).
const CHECKS: i64 = 3;
/// The relative difference beyond which a case disagrees with the direct
/// sum (`_TOLERANCE`).
const TOLERANCE: f64 = 1e-9;

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
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn doit(e: &Basic) -> R {
    crate::core::function::doit(e)
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn has_head(e: &Basic, f: &dyn Fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| f(n.head()))
}
fn has_bad(e: &Basic) -> bool {
    has_head(e, &|h| {
        matches!(
            h,
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
        )
    })
}
fn has_nan_or_zoo(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::NaN | Head::ComplexInfinity))
}
fn has_sum(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::Sum))
}
fn is_harmonic(e: &Basic) -> bool {
    e.is_function(Function::Harmonic)
}
fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(data) | Head::Dummy(data, _) => data.name.clone(),
        _ => s.to_string(),
    }
}
fn by_name(symbols: BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().filter(|s| s.is_symbol()).collect();
    v.sort_by_key(symbol_name);
    v
}
fn symbols_of(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect()
}
fn replaced(e: &Basic, point: &[(Basic, Basic)]) -> R {
    let map: HashMap<Basic, Basic> = point.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn is_polynomial_in(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}
fn is_not_ported(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if m.starts_with(crate::error::NOT_PORTED_PREFIX))
}
fn value_error(message: String) -> PolyError {
    PolyError::Value(message)
}
/// The errors the conversion into the field may raise (`ValueError`,
/// `TypeError`, `ZeroDivisionError`, the polynomial errors).
fn conversion_failure(e: &PolyError) -> bool {
    matches!(
        e,
        PolyError::Value(_) | PolyError::Type(_) | PolyError::ZeroDivision(_) | PolyError::Polys(..)
    )
}
fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}
fn sum_limits(e: &Basic) -> Vec<(Basic, Basic, Basic)> {
    crate::concrete::expr_with_limits::limits(e)
}
fn sum_function(e: &Basic) -> Basic {
    e.args()[0].clone()
}

/// The subexpressions of `expr` depending on `k` which are not rational
/// operations (`_atoms`).
fn atoms(expr: &Basic, k: &Basic) -> BTreeSet<Basic> {
    if !expr.has(k) || expr == k {
        return BTreeSet::new();
    }
    if expr.is_add() || expr.is_mul() {
        let mut result: BTreeSet<Basic> = BTreeSet::new();
        for a in expr.args() {
            result.extend(atoms(a, k));
        }
        return result;
    }
    if expr.is_pow() && integer_of(&expr.args()[1]).is_some() {
        return atoms(&expr.args()[0], k);
    }
    let mut one: BTreeSet<Basic> = BTreeSet::new();
    one.insert(expr.clone());
    one
}

/// `c` if `arg == k + c` with `c` an integer (`_linear_shift`).
fn linear_shift(arg: &Basic, k: &Basic) -> PolyResult<Option<i64>> {
    Ok(integer_of(&sub(arg, k)?))
}

/// `(a, b)` if `arg == a*k + b` with `a` a positive integer and `b` an
/// integer (`_linear`).
fn linear(arg: &Basic, k: &Basic) -> PolyResult<Option<(i64, i64)>> {
    let arg: Basic = crate::core::function::expand_method(arg)?;
    let a: Basic = crate::core::expr::coeff(&arg, k)?;
    let b: Basic = crate::core::function::expand_method(&sub(&arg, &mul(&[a.clone(), k.clone()])?)?)?;
    Ok(match (integer_of(&a), integer_of(&b)) {
        (Some(a), Some(b)) if a > 0 => Some((a, b)),
        _ => None,
    })
}

fn evaluate_at(expr: &Basic, k: &Basic, k0: i64) -> R2 {
    let value: Basic = subs(expr, k, &int(k0))?;
    let value: Basic = match doit(&value) {
        Ok(v) => v,
        Err(e) if is_not_ported(&e) => return Err(e),
        Err(_) => return Ok(None),
    };
    if has_bad(&value) {
        return Ok(None);
    }
    Ok(Some(value))
}

/// Builds a ΠΣ-field containing given expressions (`_Builder`).
struct Builder<Cs: Constants> {
    k: Basic,
    field: PiSigmaField<Cs>,
    /// atom -> expression in k and the generator symbols, in insertion
    /// order
    known: Vec<(Basic, Basic)>,
}

impl<Cs: Constants> Builder<Cs> {
    fn lookup(&self, atom: &Basic) -> Option<Basic> {
        self.known.iter().find(|(a, _)| a == atom).map(|(_, e)| e.clone())
    }

    fn remember(&mut self, atom: &Basic, e: Basic) {
        if self.lookup(atom).is_none() {
            self.known.push((atom.clone(), e));
        }
    }

    /// The element of the field equal to `expr`, adjoining generators as
    /// needed (`convert`).
    fn convert(&mut self, expr: &Basic) -> PolyResult<PiSigmaElement> {
        let mut found: Vec<(i64, String, Basic)> = Vec::new();
        for a in atoms(expr, &self.k) {
            found.push((ops(&a)?, a.to_string(), a));
        }
        found.sort_by(|p, q| (p.0, &p.1).cmp(&(q.0, &q.1)));
        for (_, _, atom) in found {
            if self.lookup(&atom).is_none() {
                let e: Basic = self.represent(&atom)?;
                self.remember(&atom, e);
            }
        }
        let rule: HashMap<Basic, Basic> = self.known.iter().cloned().collect();
        let rewritten: Basic = expr.xreplace(&rule);
        match self.field.from_expr(&rewritten) {
            Ok(e) => Ok(e),
            Err(e) if conversion_failure(&e) => Err(value_error(format!(
                "{} is not a rational expression in the summation index, the parameters and the \
                 sequences of the field: {}",
                expr, e
            ))),
            Err(e) => Err(e),
        }
    }

    fn represent(&mut self, atom: &Basic) -> R {
        let k: Basic = self.k.clone();
        // sums and products with a shifted upper limit are expressed
        // through the unshifted ones
        let (canonical, shift) = self.canonical(atom)?;
        if shift != 0 {
            if self.lookup(&canonical).is_none() {
                let e: Basic = self.represent(&canonical)?;
                self.remember(&canonical, e);
            }
            let element: PiSigmaElement = self.field.from_expr(&self.lookup(&canonical).expect("known"))?;
            return self.field.to_expr(&self.field.sigma(&element, shift)?, false, false);
        }
        if is_harmonic(atom) {
            let order: Basic = if atom.args().len() > 1 {
                atom.args()[1].clone()
            } else {
                Basic::one()
            };
            if let Some((a, b)) = linear(&atom.args()[0], &k)? {
                if !order.has(&k) {
                    // H_{a(k+1)+b} - H_{ak+b} = the a reciprocals in between
                    let mut beta: Basic = Basic::zero();
                    for i in 1..=a {
                        let base: Basic = add(&[mul(&[int(a), k.clone()])?, int(b + i)])?;
                        beta = add(&[beta, div(&int(1), &pow(&base, &order)?)?])?;
                    }
                    return self.sigma_of(atom, &beta);
                }
            }
        }
        for kind in [Kind::Sigma, Kind::Pi] {
            let matching: bool = match kind {
                Kind::Sigma => matches!(atom.head(), Head::Sum),
                Kind::Pi => matches!(atom.head(), Head::Product),
            };
            if matching {
                let lims = sum_limits(atom);
                if lims.len() == 1 {
                    let (j, lo, hi) = &lims[0];
                    let function: Basic = sum_function(atom);
                    if *hi == k && !lo.has(&k) && !function.has(&k) {
                        let next: Basic = subs(&function, j, &add(&[k.clone(), int(1)])?)?;
                        return match kind {
                            Kind::Sigma => self.sigma_of(atom, &next),
                            Kind::Pi => self.pi_of(atom, &next),
                        };
                    }
                }
            }
        }
        if let Some(ratio) = crate::concrete::gosper::hypersimp(atom, &k)? {
            if ratio != Basic::zero() {
                return self.pi_of(atom, &ratio);
            }
        }
        Err(value_error(format!(
            "{} is not a hypergeometric term or an indefinite sum in {}",
            atom, k
        )))
    }

    /// `(atom with k in place of k + c, c)` (`_canonical`).
    fn canonical(&self, atom: &Basic) -> PolyResult<(Basic, i64)> {
        let k: &Basic = &self.k;
        if is_harmonic(atom) {
            if let Some((a, b)) = linear(&atom.args()[0], k)? {
                let b0: i64 = b.rem_euclid(a);
                if b != b0 {
                    let argument: Basic = add(&[mul(&[int(a), k.clone()])?, int(b0)])?;
                    let mut args: Vec<Basic> = vec![argument];
                    args.extend(atom.args()[1..].iter().cloned());
                    let shifted: Basic = Basic::new(Head::Function(Function::Harmonic), args);
                    return Ok((shifted, (b - b0).div_euclid(a)));
                }
            }
        }
        if matches!(atom.head(), Head::Sum | Head::Product) {
            let lims = sum_limits(atom);
            if lims.len() == 1 {
                let (j, lo, hi) = &lims[0];
                let function: Basic = sum_function(atom);
                if let Some(shift) = linear_shift(hi, k)? {
                    if shift != 0 && !lo.has(k) && !function.has(k) {
                        let unshifted: Basic = crate::concrete::expr_with_limits::node(
                            atom.head().clone(),
                            &function,
                            &[(j.clone(), lo.clone(), k.clone())],
                        )?;
                        return Ok((unshifted, shift));
                    }
                }
            }
        }
        Ok((atom.clone(), 0))
    }

    /// Whether `c` is a constant of the field (`_is_constant_expr`).
    fn is_constant_expr(&self, c: &Basic) -> PolyResult<bool> {
        match self.field.from_expr(c) {
            Ok(e) => Ok(self.field.is_constant(&e)),
            Err(PolyError::Value(_) | PolyError::Type(_) | PolyError::ZeroDivision(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// The constant `atom - w` (or `atom/w`) by evaluation at a point
    /// (`_initial_constant`).
    fn initial_constant(&self, atom: &Basic, w_expr: &Basic, ratio: bool) -> R2 {
        for k0 in 0..12 {
            let (Some(a0), Some(w0)) = (evaluate_at(atom, &self.k, k0)?, evaluate_at(w_expr, &self.k, k0)?) else {
                continue;
            };
            let c: Basic = if ratio {
                if w0 == Basic::zero() {
                    continue;
                }
                cancel(&div(&a0, &w0)?)?
            } else {
                cancel(&sub(&a0, &w0)?)?
            };
            if !self.is_constant_expr(&c)? {
                return Ok(None);
            }
            // the identity must hold for all k: check at another point
            let k1: i64 = k0 + 1;
            if let (Some(a1), Some(w1)) = (evaluate_at(atom, &self.k, k1)?, evaluate_at(w_expr, &self.k, k1)?) {
                let check: Basic = if ratio {
                    cancel(&sub(&div(&a1, &w1)?, &c)?)?
                } else {
                    cancel(&sub(&sub(&a1, &w1)?, &c)?)?
                };
                if check != Basic::zero() {
                    return Ok(None);
                }
            }
            return Ok(Some(c));
        }
        Ok(None)
    }

    fn sigma_of(&mut self, atom: &Basic, beta: &Basic) -> R {
        let b: PiSigmaElement = self.convert(beta)?;
        if let Some(w) = self.field.telescope(&b)? {
            let w_expr: Basic = self.field.to_expr(&w, true, false)?;
            if let Some(c) = self.initial_constant(atom, &w_expr, false)? {
                return add(&[self.field.to_expr(&w, false, false)?, c]);
            }
        }
        let (field, symbol) = self.field.extend(Kind::Sigma, &b, atom, Some("sigma"), None)?;
        self.field = field;
        Ok(symbol)
    }

    fn pi_of(&mut self, atom: &Basic, alpha: &Basic) -> R {
        let a: PiSigmaElement = self.convert(alpha)?;
        if let Some(sols) = self.field.solve_embedded(&a, &[], None)? {
            for (_, w) in sols {
                if w.is_nonzero() {
                    let w_expr: Basic = self.field.to_expr(&w, true, false)?;
                    if let Some(c) = self.initial_constant(atom, &w_expr, true)? {
                        return mul(&[c, self.field.to_expr(&w, false, false)?]);
                    }
                }
            }
        }
        let (field, symbol) = self.field.extend(Kind::Pi, &a, atom, Some("pi"), None)?;
        self.field = field;
        Ok(symbol)
    }
}

/// The degree of `expr` in `atom`, an upper bound (`_degree_in`).
fn degree_in(expr: &Basic, atom: &Basic) -> i64 {
    if expr == atom {
        return 1;
    }
    if !expr.has(atom) {
        return 0;
    }
    if expr.is_add() {
        return expr.args().iter().map(|a| degree_in(a, atom)).max().unwrap_or(0);
    }
    if expr.is_mul() {
        return expr.args().iter().map(|a| degree_in(a, atom)).sum();
    }
    if expr.is_pow() {
        if let Some(e) = integer_of(&expr.args()[1]) {
            return e.abs() * degree_in(&expr.args()[0], atom);
        }
    }
    1
}

fn harmonics(k: &Basic, upto: i64) -> PolyResult<Vec<Basic>> {
    let mut out: Vec<Basic> = Vec::new();
    for m in 1..=upto {
        out.push(crate::functions::combinatorial::numbers::harmonic(k, Some(&int(m)))?);
    }
    Ok(out)
}

/// Harmonic numbers to adjoin when the telescoping fails
/// (`_auto_extensions`).
fn auto_extensions(f: &Basic, k: &Basic) -> PolyResult<Vec<Basic>> {
    let found: Vec<Basic> = atoms(f, k)
        .into_iter()
        .filter(|a| is_harmonic(a) || matches!(a.head(), Head::Sum))
        .collect();
    if found.is_empty() {
        // a rational summand with no closed form in the rational functions
        // sums to harmonic numbers (sympy-extras#34)
        let (_, denominator) = crate::core::expr::as_numer_denom(f)?;
        if !denominator.has(k)
            || crate::core::expr::polynomial_like(
                f,
                std::slice::from_ref(k),
                crate::core::expr::PolynomialKind::Rational,
            ) != Some(true)
        {
            return Ok(Vec::new());
        }
        let factored =
            crate::polys::polytools::Poly::new(&denominator, std::slice::from_ref(k)).and_then(|p| p.factor_list());
        let factors = match factored {
            Ok((_, factors)) => factors,
            Err(PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_)) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let multiplicity: i64 = factors.iter().map(|(_, m)| *m as i64).max().unwrap_or(0);
        return harmonics(k, multiplicity);
    }
    let mut order: i64 = 1;
    for a in &found {
        if is_harmonic(a) && a.args().len() > 1 {
            if let Some(m) = integer_of(&a.args()[1]) {
                order = order.max(m);
            }
        }
    }
    let degree: i64 = found.iter().map(|a| degree_in(f, a)).max().unwrap_or(0);
    harmonics(k, order + degree.max(1))
}

/// What a computation does with the field built for `f`: generic over the
/// constants of the field.
trait WithField {
    type Out;
    fn run<Cs: Constants>(&self, builder: Builder<Cs>) -> PolyResult<Self::Out>;
}

/// The field for the parameters (rational or parametric constants), the
/// computation run in it.
fn with_field<W: WithField>(k: &Basic, params: &[Basic], w: &W) -> PolyResult<W::Out> {
    if params.is_empty() {
        let field = crate::concrete::pisigma::rational_pisigma_field(k)?;
        w.run(Builder {
            k: k.clone(),
            field,
            known: Vec::new(),
        })
    } else {
        let field = crate::concrete::pisigma::parametric_pisigma_field(k, params)?;
        w.run(Builder {
            k: k.clone(),
            field,
            known: Vec::new(),
        })
    }
}

fn parameters_of(f: &Basic, k: &Basic, extensions: &[Basic]) -> Vec<Basic> {
    let mut symbols: BTreeSet<Basic> = f.free_symbols();
    for e in extensions {
        symbols.extend(e.free_symbols());
    }
    symbols.remove(k);
    by_name(symbols)
}

/// The telescoper of `f` as expressions: `(g, sigma(g))` in `k`, the atoms
/// represented in the field; `None` when `f` has none.
struct Telescoped {
    g: Basic,
    shifted: Basic,
}

struct TelescopeJob<'a> {
    f: &'a Basic,
    extensions: Vec<Basic>,
}

impl WithField for TelescopeJob<'_> {
    type Out = (Option<Telescoped>, Vec<Basic>);
    fn run<Cs: Constants>(&self, mut builder: Builder<Cs>) -> PolyResult<Self::Out> {
        for e in &self.extensions {
            builder.convert(e)?;
        }
        let element: PiSigmaElement = builder.convert(self.f)?;
        let known: Vec<Basic> = builder.known.iter().map(|(a, _)| a.clone()).collect();
        let Some(g) = builder.field.telescope(&element)? else {
            return Ok((None, known));
        };
        let g_expr: Basic = builder.field.to_expr(&g, true, false)?;
        let shifted: Basic = builder.field.to_expr(&builder.field.sigma(&g, 1)?, true, false)?;
        Ok((Some(Telescoped { g: g_expr, shifted }), known))
    }
}

/// A telescoper for `f`, the automatic extensions tried when the first
/// attempt fails (`_telescope`).
fn telescope(f: &Basic, k: &Basic, extensions: &[Basic], auto: bool) -> PolyResult<Option<Telescoped>> {
    let job = TelescopeJob {
        f,
        extensions: extensions.to_vec(),
    };
    let (found, known) = with_field(k, &parameters_of(f, k, extensions), &job)?;
    if found.is_some() || !auto {
        return Ok(found);
    }
    let extra: Vec<Basic> = auto_extensions(f, k)?
        .into_iter()
        .filter(|e| !extensions.contains(e) && !known.contains(e))
        .collect();
    if extra.is_empty() {
        return Ok(None);
    }
    let mut all: Vec<Basic> = extensions.to_vec();
    all.extend(extra);
    let job = TelescopeJob {
        f,
        extensions: all.clone(),
    };
    Ok(with_field(k, &parameters_of(f, k, &all), &job)?.0)
}

/// The indefinite sum of `f` by Karr's algorithm: `g` with `g(k + 1) -
/// g(k) == f`, or `None` (`karr_term`).
pub fn karr_term(f: &Basic, k: &Basic, extensions: &[Basic], auto: bool, special_values: bool) -> R2 {
    if !k.is_symbol() {
        return Err(PolyError::Type(format!("{} is not a symbol", k)));
    }
    let Some(found) = telescope(f, k, extensions, auto)? else {
        return Ok(None);
    };
    let value: Basic = found.g;
    if !special_values {
        return Ok(Some(value));
    }
    let at_point = |point: &SumPoint| -> R2 {
        let rule: HashMap<Basic, Basic> = point.iter().cloned().collect();
        let shifted: Vec<Basic> = extensions.iter().map(|e| e.xreplace(&rule)).collect();
        karr_term(&f.xreplace(&rule), k, &shifted, auto, true)
    };
    Ok(Some(with_special_values_sum(
        &value,
        f,
        std::slice::from_ref(k),
        &[],
        &at_point,
        false,
        None,
    )?))
}

/// The definite sum `sum_{k=a}^{b} f(k)` by Karr's algorithm, `g(b + 1) -
/// g(a)` (`karr_sum`); `bounds` `None` is the indefinite sum.
pub fn karr_sum(
    f: &Basic,
    index: &Basic,
    bounds: Option<(&Basic, &Basic)>,
    extensions: &[Basic],
    auto: bool,
    special_values: bool,
) -> R2 {
    let Some((a, b)) = bounds else {
        return karr_term(f, index, extensions, auto, special_values);
    };
    if !index.is_symbol() {
        return Err(PolyError::Type(format!("{} is not a symbol", index)));
    }
    let Some(found) = telescope(f, index, extensions, auto)? else {
        return Ok(None);
    };
    // g(b + 1) is sigma(g) at b, which keeps the generators unshifted
    let upper: Basic = subs(&found.shifted, index, b)?;
    let lower: Basic = subs(&found.g, index, a)?;
    let closed: Basic = crate::core::exprtools::factor_terms(
        &cancel(&sub(&doit(&upper)?, &doit(&lower)?)?)?,
        false,
        false,
        true,
        false,
    )?;
    let repaired: Basic = repaired_at_poles(&closed, &found.g, f, index, a, b)?;
    if !special_values {
        return Ok(Some(repaired));
    }
    let at_point = |point: &SumPoint| -> R2 {
        let rule: HashMap<Basic, Basic> = point.iter().cloned().collect();
        let shifted: Vec<Basic> = extensions.iter().map(|e| e.xreplace(&rule)).collect();
        karr_sum(&f.xreplace(&rule), index, Some((a, b)), &shifted, auto, true)
    };
    Ok(Some(with_special_values_sum(
        &repaired,
        f,
        std::slice::from_ref(index),
        &[a.clone(), b.clone()],
        &at_point,
        true,
        None,
    )?))
}

/// The sample values of the parameters left free by a point
/// (`_SAMPLES`).
fn sample(i: usize) -> R {
    let samples: [(i64, i64); 4] = [(17, 7), (23, 9), (31, 11), (41, 13)];
    let (p, q) = samples[i % 4];
    add(&[Basic::rational(p.into(), q.into())?, int((i / 4) as i64)])
}

/// The powers of `0` with an exponent in `k` (`_zero_powers`).
fn zero_powers(term: &Basic, k: &Basic) -> Vec<Basic> {
    term.preorder()
        .into_iter()
        .filter(|node| node.is_pow() && is(&node.args()[0], Predicate::zero) == Some(true) && node.args()[1].has(k))
        .collect()
}

/// The sum of `term` with its powers `0**(c*k + d)` written by their
/// values (`_sum_of_zero_powers`).
fn sum_of_zero_powers(term: &Basic, k: &Basic, lower: &Basic, upper: &Basic) -> R2 {
    if integer_of(lower).is_none() {
        return Ok(None);
    }
    let mut zeroed: Vec<(Basic, Basic)> = Vec::new();
    let mut at_lower: bool = false;
    for power in zero_powers(term, k) {
        let exponent: &Basic = &power.args()[1];
        if !is_polynomial_in(exponent, k) {
            return Ok(None);
        }
        let polynomial = crate::polys::polytools::Poly::new(exponent, std::slice::from_ref(k))?;
        if polynomial.degree(0)? != 1 {
            return Ok(None);
        }
        let coefficients: Vec<Basic> = polynomial.all_coeffs()?;
        let (c, d) = (coefficients[0].clone(), coefficients[1].clone());
        if !integer_of(&c).is_some_and(|c| c > 0) || d.as_rational().is_none() {
            return Ok(None);
        }
        let vanishing: Basic = div(&crate::core::mul::neg(&d)?, &c)?;
        if crate::core::relational::gt(&vanishing, lower)? == crate::core::logic::true_() {
            // 0 to a negative power inside the range
            return Ok(None);
        }
        at_lower = at_lower || vanishing == *lower;
        zeroed.push((power.clone(), Basic::zero()));
    }
    let rest: Basic = replaced(term, &zeroed)?;
    let mut value: Basic = if rest.has(k) {
        crate::concrete::summations::summation(&rest, &(k.clone(), lower.clone(), upper.clone()))?
    } else {
        mul(&[rest.clone(), add(&[sub(upper, lower)?, int(1)])?])?
    };
    if at_lower {
        value = add(&[value, sub(&subs(term, k, lower)?, &subs(&rest, k, lower)?)?])?;
    }
    Ok(Some(value))
}

/// Whether `term` has a pole at an integer of the range
/// (`_pole_in_range`).
fn pole_in_range(term: &Basic, k: &Basic, lower: &Basic, upper: &Basic) -> PolyResult<bool> {
    let Some(low) = integer_of(lower) else {
        return Ok(false);
    };
    let together: Basic = crate::polys::rationaltools::together(term, false, true)?;
    let (_, denominator) = crate::core::expr::as_numer_denom(&together)?;
    if !denominator.has(k) || !is_polynomial_in(&denominator, k) {
        return Ok(false);
    }
    let factored = crate::timeout::attempt_within(Some(POLE_SECONDS), || {
        crate::polys::polytools::factor_list(&denominator, std::slice::from_ref(k))
    })?;
    let Some((_, factors)) = factored else {
        return Ok(false);
    };
    for (factor, _) in factors {
        let polynomial = crate::polys::polytools::Poly::new(&factor, std::slice::from_ref(k))?;
        if polynomial.degree(0)? != 1 {
            continue;
        }
        let coefficients: Vec<Basic> = polynomial.all_coeffs()?;
        let root: Basic = cancel(&div(&crate::core::mul::neg(&coefficients[1])?, &coefficients[0])?)?;
        if let Some(r) = integer_of(&root) {
            if r >= low && integer_of(upper).map_or(true, |u| r <= u) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The values of the symbol of `upper` at which a sum from `lower` is
/// checked, with the upper limit there (`_upper_values`).
fn upper_values(lower: i64, upper: &Basic) -> PolyResult<Option<Vec<(SumPoint, i64)>>> {
    if let Some(u) = integer_of(upper) {
        return Ok(Some(vec![(Vec::new(), u)]));
    }
    let symbols: Vec<Basic> = by_name(symbols_of(upper));
    if symbols.len() != 1 || !is_polynomial_in(upper, &symbols[0]) {
        return Ok(None);
    }
    let m: &Basic = &symbols[0];
    let polynomial = crate::polys::polytools::Poly::new(upper, std::slice::from_ref(m))?;
    if polynomial.degree(0)? != 1 {
        return Ok(None);
    }
    let coefficients: Vec<Basic> = polynomial.all_coeffs()?;
    let (Some(c), Some(d)) = (integer_of(&coefficients[0]), integer_of(&coefficients[1])) else {
        return Ok(None);
    };
    if c <= 0 {
        return Ok(None);
    }
    let first: i64 = 0i64.max(-(d - lower + 1).div_euclid(c));
    Ok(Some(
        (first..first + CHECKS)
            .map(|v| (vec![(m.clone(), int(v))], c * v + d))
            .collect(),
    ))
}

fn magnitude(e: &Basic) -> Option<f64> {
    crate::numeric::reliable_complex(e, 15, &[]).map(|(a, b)| a.hypot(b))
}

/// Whether `value` is refuted by the sum computed term by term
/// (`_disagrees`).
fn disagrees(
    value: &Basic,
    term: &Basic,
    k: &Basic,
    bounds: Option<(&Basic, &Basic)>,
    indefinite: bool,
) -> PolyResult<bool> {
    let mut checks: Vec<(Basic, Basic)> = Vec::new();
    if let Some((lower, upper)) = bounds {
        let Some(low) = integer_of(lower) else {
            return Ok(false);
        };
        let Some(uppers) = upper_values(low, upper)? else {
            return Ok(false);
        };
        for (at, top) in uppers {
            let shifted: Basic = replaced(term, &at)?;
            let mut terms: Vec<Basic> = Vec::new();
            for i in low..=top {
                terms.push(subs(&shifted, k, &int(i))?);
            }
            checks.push((replaced(value, &at)?, add(&terms)?));
        }
    } else if indefinite {
        for i in 2..2 + CHECKS {
            checks.push((
                sub(&subs(value, k, &int(i + 1))?, &subs(value, k, &int(i))?)?,
                subs(term, k, &int(i))?,
            ));
        }
    } else {
        return Ok(false);
    }
    for (case, direct) in checks {
        let mut others: BTreeSet<Basic> = symbols_of(&case);
        others.extend(symbols_of(&direct));
        others.remove(k);
        let mut samples: Vec<(Basic, Basic)> = Vec::new();
        for (i, s) in by_name(others).into_iter().enumerate() {
            samples.push((s, sample(i)?));
        }
        let evaluated: Option<Basic> =
            crate::timeout::attempt_within(Some(CHECK_SECONDS), || doit(&replaced(&case, &samples)?))?;
        let Some(evaluated) = evaluated else {
            continue;
        };
        let at_direct: Basic = replaced(&direct, &samples)?;
        let (Some(difference), Some(size)) = (magnitude(&sub(&evaluated, &at_direct)?), magnitude(&at_direct)) else {
            continue;
        };
        if difference > TOLERANCE * (1.0 + size) {
            return Ok(true);
        }
    }
    Ok(false)
}

type SumCompute<'a> = &'a dyn Fn(&SumPoint) -> R2;

/// The value of a sum of `f` with a case for each isolated value of the
/// parameters of the summand at which it is undefined
/// (`_sum_special_values_native`).
fn sum_special_values_native(
    value: &Basic,
    f: &Basic,
    indices: &[Basic],
    limits: &[Basic],
    compute: SumCompute,
    keep_unevaluated: bool,
    defined: Option<&dyn Fn(&SumPoint) -> bool>,
) -> R {
    let mut excluded: BTreeSet<Basic> = indices.iter().cloned().collect();
    for limit in limits {
        excluded.extend(symbols_of(limit));
    }
    let parameters: Vec<Basic> = by_name(symbols_of(f).into_iter().filter(|s| !excluded.contains(s)).collect());
    if parameters.is_empty() {
        return Ok(value.clone());
    }
    let index: &Basic = &indices[0];
    let bounds: Option<(&Basic, &Basic)> = if limits.len() == 2 {
        Some((&limits[0], &limits[1]))
    } else {
        None
    };
    let indefinite: bool = limits.is_empty() && indices.len() == 1;
    let defined_at = |point: &SumPoint| -> bool {
        let Ok(term) = replaced(f, point) else {
            return false;
        };
        if has_nan_or_zoo(&term) {
            return false;
        }
        if let Some((lo, hi)) = bounds {
            if pole_in_range(&term, index, lo, hi).unwrap_or(false) {
                return false;
            }
        }
        defined.map_or(true, |d| d(point))
    };
    let checked = |point: &crate::special_values::Point, _at: &[Basic]| -> R2 {
        let term: Basic = replaced(f, point)?;
        let special: Option<Basic> = match bounds {
            Some((lo, hi)) if !zero_powers(&term, index).is_empty() => sum_of_zero_powers(&term, index, lo, hi)?,
            _ => compute(point)?,
        };
        let Some(special) = special else {
            return Ok(None);
        };
        if !crate::settings::settings().numerical_checks {
            return Ok(Some(special));
        }
        if disagrees(&special, &term, index, bounds, indefinite)? {
            return Ok(None);
        }
        Ok(Some(special))
    };
    crate::special_values::with_special_values(
        value,
        &parameters,
        &[],
        &checked,
        Some(&defined_at),
        false,
        None,
        keep_unevaluated,
        None,
    )
}

/// `with_sum_special_values` for core expressions (`_with_special_values`).
pub(crate) fn with_special_values_sum(
    value: &Basic,
    f: &Basic,
    indices: &[Basic],
    limits: &[Basic],
    compute: SumCompute,
    keep_unevaluated: bool,
    defined: Option<&dyn Fn(&SumPoint) -> bool>,
) -> R {
    let mut excluded: BTreeSet<Basic> = indices.iter().cloned().collect();
    for limit in limits {
        excluded.extend(limit.free_symbols());
    }
    if f.free_symbols().iter().all(|s| excluded.contains(s)) {
        return Ok(value.clone());
    }
    sum_special_values_native(value, f, indices, limits, compute, keep_unevaluated, defined)
}

/// The closed form with the values where it is false restored
/// (`_repaired_at_poles`, sympy-extras#35).
fn repaired_at_poles(closed: &Basic, antidifference: &Basic, f: &Basic, index: &Basic, a: &Basic, b: &Basic) -> R {
    let mut symbols: BTreeSet<Basic> = antidifference.free_symbols();
    symbols.remove(index);
    let parameters: Vec<Basic> = by_name(symbols);
    if parameters.len() != 1 || !b.is_symbol() || *b != parameters[0] {
        return Ok(closed.clone());
    }
    let parameter: &Basic = &parameters[0];
    let together: Basic = crate::polys::rationaltools::together(antidifference, false, true)?;
    let (_, denominator) = crate::core::expr::as_numer_denom(&together)?;
    let poles = match crate::polys::polyroots::roots(&denominator, std::slice::from_ref(parameter), true, true, None) {
        Ok(p) => p,
        Err(PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_)) => return Ok(closed.clone()),
        Err(e) => return Err(e),
    };
    let mut found: Vec<Basic> = poles.iter().map(|(r, _)| r.clone()).collect();
    found.sort_by_key(|r| r.to_string());
    let mut branches: Vec<(Basic, Basic)> = Vec::new();
    for pole in found {
        if !integer_of(&pole).is_some_and(|p| p >= 0) {
            continue;
        }
        let summand: Basic = subs(f, parameter, &pole)?;
        let direct: Basic = doit(&crate::concrete::expr_with_limits::node(
            Head::Sum,
            &summand,
            &[(index.clone(), a.clone(), pole.clone())],
        )?)?;
        if matches!(direct.head(), Head::Sum) {
            continue;
        }
        if cancel(&sub(&direct, &subs(closed, parameter, &pole)?)?)? != Basic::zero() {
            branches.push((direct, crate::core::relational::eq(parameter, &pole)?));
        }
    }
    if branches.is_empty() {
        return Ok(closed.clone());
    }
    branches.push((closed.clone(), crate::core::logic::true_()));
    crate::functions::elementary::piecewise::piecewise(&branches)
}

/// A summation limit `(k, a, b)`.
pub type Limit = (Basic, Basic, Basic);

/// Summation with the ported `summation` and Karr's algorithm
/// (`summation`).
pub fn summation(f: &Basic, limits: &[Limit], extensions: &[Basic], auto: bool, special_values: bool) -> R {
    for (k, _, _) in limits {
        if !k.is_symbol() {
            return Err(PolyError::Type(format!("{} is not a symbol", k)));
        }
    }
    let tuples: Vec<Basic> = limits
        .iter()
        .map(|(k, a, b)| Basic::new(Head::Tuple, vec![k.clone(), a.clone(), b.clone()]))
        .collect();
    let s: Basic = crate::concrete::expr_with_limits::sum_(f, &tuples)?;
    let mut result: Basic = if matches!(s.head(), Head::Sum) {
        crate::concrete::summations::sum_doit(&s, false)?
    } else {
        s
    };
    if has_sum(&result) {
        result = karr_on_sums(&result, extensions, auto)?;
    }
    if !special_values {
        return Ok(result);
    }
    let indices: Vec<Basic> = limits.iter().map(|(k, _, _)| k.clone()).collect();
    let bounds: Vec<Basic> = limits.iter().flat_map(|(_, a, b)| [a.clone(), b.clone()]).collect();
    let at_point = |point: &SumPoint| -> R2 {
        let rule: HashMap<Basic, Basic> = point.iter().cloned().collect();
        let at_limits: Vec<Limit> = limits
            .iter()
            .map(|(k, a, b)| (k.clone(), a.xreplace(&rule), b.xreplace(&rule)))
            .collect();
        Ok(Some(summation(&f.xreplace(&rule), &at_limits, extensions, auto, true)?))
    };
    with_special_values_sum(&result, f, &indices, &bounds, &at_point, true, None)
}

/// The sums of `expr` with one limit evaluated by Karr's algorithm
/// (`_karr_on_sums`).
fn karr_on_sums(expr: &Basic, extensions: &[Basic], auto: bool) -> R {
    if expr.args().is_empty() || expr.is_atom() {
        return Ok(expr.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in expr.args() {
        args.push(karr_on_sums(a, extensions, auto)?);
    }
    let rebuilt: Basic = if args.iter().zip(expr.args()).any(|(new, old)| new != old) {
        crate::core::function::rebuild_evaluated(expr, args)?
    } else {
        expr.clone()
    };
    if !matches!(rebuilt.head(), Head::Sum) {
        return Ok(rebuilt);
    }
    let lims = sum_limits(&rebuilt);
    if lims.len() != 1 {
        return Ok(rebuilt);
    }
    let (k, a, b) = &lims[0];
    let function: Basic = sum_function(&rebuilt);
    let value: Option<Basic> = match karr_sum(&function, k, Some((a, b)), extensions, auto, false) {
        Ok(v) => v,
        Err(PolyError::Value(_)) => None,
        Err(e) => return Err(e),
    };
    if value.is_none() {
        let mut limit_symbols: BTreeSet<Basic> = a.free_symbols();
        limit_symbols.extend(b.free_symbols());
        limit_symbols.remove(k);
        if function.free_symbols().iter().any(|s| limit_symbols.contains(s)) {
            // the summand depends on a limit: creative telescoping
            return Err(crate::error::not_ported("concrete/creative (definite_sum)"));
        }
    }
    Ok(value.unwrap_or(rebuilt))
}
