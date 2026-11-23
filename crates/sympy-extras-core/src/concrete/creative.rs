//! Port of `sympy_extras/concrete/creative.py`: definite summation by
//! creative telescoping in ΠΣ-fields (Zeilberger's algorithm for
//! hypergeometric summands, Schneider's extension for summands with
//! harmonic numbers and other indefinite sums), the recurrence of the sums
//! completed with the boundary terms, and its d'Alembertian solution with
//! the constants fixed by the sum computed directly.
//!
//! The field of Python's `CreativeTelescoper` is generic here (rational or
//! parametric constants): the values the definite sum needs from it (the
//! shifted certificate, the poles, the check) are computed while it is at
//! hand, their errors kept for the caller which asks for them.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::concrete::karr::{SumPoint, WithField};
use crate::concrete::pisigma::{Constants, Kind, PiSigmaElement};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, UndefinedFunction};
use crate::core::expr::{is, polynomial_like, PolynomialKind};
use crate::error::{PolyError, PolyResult, NOT_PORTED_PREFIX};
use crate::numbers::{Int, Rat};
use crate::polys::polytools::Poly;

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;

/// The seconds given to one simplification or one call of a solver
/// (`_STEP_SECONDS`).
const STEP_SECONDS: f64 = 10.0;
/// The values of `n` beyond the initial values at which the closed form is
/// compared with the sum (`_CHECKS`).
const CHECKS: i64 = 5;

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
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn subs_mapping(e: &Basic, pairs: &[(Basic, Basic)]) -> R {
    crate::core::subs::subs_mapping(e, pairs)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn combsimp(e: &Basic) -> R {
    crate::simplify::gammasimp::combsimp(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn together(e: &Basic) -> R {
    crate::polys::rationaltools::together(e, false, true)
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn expand_func(e: &Basic) -> R {
    crate::functions::expand_func::expand_func(e, true)
}
fn doit(e: &Basic) -> R {
    crate::core::function::doit(e)
}
fn powsimp(e: &Basic) -> R {
    crate::simplify::powsimp::powsimp_default(e)
}
fn factor_terms(e: &Basic) -> R {
    crate::core::exprtools::factor_terms(e, false, false, false, true)
}
fn rewrite(e: &Basic, target: &str) -> R {
    crate::core::rewriting::rewrite(e, &[], target, &crate::core::rewriting::RewriteHints::default())
}
fn factor_list(e: &Basic, gens: &[Basic]) -> PolyResult<(Basic, Vec<(Basic, Basic)>)> {
    crate::polys::polytools::factor_list(e, gens)
}
fn tuple(v: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, v)
}
fn harmonic(n: &Basic, m: &Basic) -> R {
    crate::functions::combinatorial::numbers::harmonic(n, Some(m))
}
fn is_rational_function(e: &Basic, symbols: &[Basic]) -> bool {
    polynomial_like(e, symbols, PolynomialKind::Rational) == Some(true)
}
fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    polynomial_like(e, std::slice::from_ref(x), PolynomialKind::Polynomial) == Some(true)
}
fn integer_of(e: &Basic) -> Option<i64> {
    if !matches!(e.head(), Head::Rational { value, .. } if value.is_int()) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
}
fn rational_of(e: &Basic) -> Option<Rat> {
    match e.head() {
        Head::Rational { value, .. } => Some(value.clone()),
        _ => None,
    }
}
/// The integer of `e` (`_int`), `TypeError` otherwise.
fn int_of(e: &Basic) -> PolyResult<i64> {
    integer_of(e).ok_or_else(|| PolyError::Type(format!("{} is not an integer", e)))
}
fn has_head(e: &Basic, pred: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(pred)
}
fn has_sum(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::Sum))
}
fn has_harmonic(e: &Basic) -> bool {
    has_head(e, &|n| n.is_function(Function::Harmonic))
}
fn symbol(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    crate::core::symbol::Symbol::new(name, assumptions).0
}
fn dummy(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    crate::core::symbol::Dummy::new(name, assumptions).0
}
fn name_of(s: &Basic) -> String {
    s.to_string()
}
fn sorted_symbols(symbols: BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().collect();
    v.sort_by_key(name_of);
    v
}
fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}
/// Whether `e` is a negative rational number (`_negative`).
fn negative(e: &Basic) -> bool {
    rational_of(e).is_some_and(|r| r < Rat::ZERO)
}
/// The errors of the polynomial and simplification routines on an
/// expression outside their scope (`_OUT_OF_SCOPE`).
fn out_of_scope(e: &PolyError) -> bool {
    match e {
        PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_) => true,
        PolyError::NotImplemented(m) => !m.starts_with(NOT_PORTED_PREFIX),
        _ => false,
    }
}
/// `f()` within `seconds`, `None` when the time is up or the routines give
/// up (`_within`).
fn within(f: impl FnOnce() -> R, seconds: f64) -> R2 {
    crate::timeout::attempt_within(Some(seconds), f)
}
fn poly(e: &Basic, x: &Basic) -> PolyResult<Poly> {
    Poly::new(e, std::slice::from_ref(x))
}
/// The key of the shortest candidate (`(_ops(e), len(str(e)))`).
fn shortest(candidates: Vec<Basic>) -> PolyResult<Basic> {
    let mut best: Option<((i64, usize), Basic)> = None;
    for c in candidates {
        let key: (i64, usize) = (ops(&c)?, c.to_string().chars().count());
        if best.as_ref().map_or(true, |(b, _)| key < *b) {
            best = Some((key, c));
        }
    }
    Ok(best.expect("a candidate").1)
}
/// `to_expr`: `TypeError` for an object which is no expression.
fn to_expr(e: &Basic) -> R {
    if !!crate::core::logic::is_boolean(e) && !matches!(e.head(), Head::Tuple) {
        return Err(PolyError::Type(format!("an expression is expected, got {}", e)));
    }
    Ok(e.clone())
}

fn finite(value: &Basic) -> bool {
    !has_head(value, &|n| {
        matches!(
            n.head(),
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
        )
    })
}

/// The poles `k = alpha*n + beta` of the representations in the field and
/// the integer poles in `n` of their coefficients.
type Poles = (Vec<(Rat, Rat)>, Vec<i64>);

/// The result of creative telescoping for a summand `f(n, k)`
/// (`CreativeTelescoper`): `sum_i c_i(n) f(n + i, k) = g(n, k + 1) -
/// g(n, k)`.
#[derive(Clone, Debug)]
pub struct CreativeTelescoper {
    pub term: Basic,
    pub k: Basic,
    pub n: Basic,
    pub coefficients: Vec<Basic>,
    pub certificate: Basic,
    shifted: Result<Basic, PolyError>,
    checked: Result<bool, PolyError>,
    poles: Result<Poles, PolyError>,
}

impl CreativeTelescoper {
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// The left hand side `sum_i c_i(n) S(n + i)` of the recurrence
    /// (`recurrence`).
    pub fn recurrence(&self, name: &str) -> R {
        let f: Arc<UndefinedFunction> = Arc::new(UndefinedFunction::new(name, None, &[]));
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[
                c.clone(),
                f.apply(vec![add(&[self.n.clone(), int(i as i64)])?])?,
            ])?);
        }
        add(&terms)
    }

    /// `g(n, k + 1)` written with the sequences of the field at `k`
    /// (`shifted_certificate`).
    pub fn shifted_certificate(&self) -> R {
        self.shifted.clone()
    }

    /// Whether the telescoping relation holds in the field (`check`).
    pub fn check(&self) -> PolyResult<bool> {
        self.checked.clone()
    }
}

// ----------------------------------------------------------------------
// the summand as an element of a PiSigma-field over Q(n)

/// Whether `expr` contains harmonic numbers, sums or products depending on
/// `k` (`_has_sums`).
fn has_sums(expr: &Basic, k: &Basic) -> bool {
    let sums = |a: &Basic| {
        has_head(a, &|n| {
            n.is_function(Function::Harmonic) || matches!(n.head(), Head::Sum | Head::Product)
        })
    };
    crate::concrete::karr::atoms(expr, k).iter().any(sums)
}

/// `(u, T, R)` with `f = u*T*R` (`_split`).
fn split(f: &Basic, k: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    let (mut u, mut t, mut r) = (Basic::one(), Basic::one(), Basic::one());
    let factors: Vec<Basic> = if f.is_mul() { f.args().to_vec() } else { vec![f.clone()] };
    for factor in factors {
        if !factor.has(k) {
            u = mul(&[u, factor])?;
        } else if has_sums(&factor, k) || crate::concrete::karr::atoms(&factor, k).is_empty() {
            r = mul(&[r, factor])?;
        } else {
            t = mul(&[t, factor])?;
        }
    }
    Ok((u, t, r))
}

/// `num/den` as a rational function of the `symbols`, or `None`
/// (`_rational_ratio`).
fn rational_ratio(num: &Basic, den: &Basic, symbols: &[Basic]) -> R2 {
    if num == den {
        return Ok(Some(Basic::one()));
    }
    let quotient: Basic = div(num, den)?;
    let ratio: Option<Basic> = within(|| cancel(&combsimp(&quotient)?), STEP_SECONDS)?;
    if let Some(r) = ratio.filter(|r| is_rational_function(r, symbols)) {
        return Ok(Some(r));
    }
    let second: Option<Basic> = within(|| cancel(&combsimp(&rewrite(&quotient, "gamma")?)?), STEP_SECONDS)?;
    Ok(second.filter(|r| is_rational_function(r, symbols)))
}

/// The coefficients scaled to polynomials without a common factor, the
/// leading one with a positive leading coefficient, and the scale
/// (`_normalized`).
fn normalized(values: &[Basic], n: &Basic) -> PolyResult<(Vec<Basic>, Basic)> {
    let mut denominators: Vec<Basic> = Vec::new();
    for v in values {
        if *v != Basic::zero() {
            denominators.push(numer_denom(&together(v)?)?.1);
        }
    }
    let mut scale: Basic = if denominators.is_empty() {
        Basic::one()
    } else {
        crate::polys::polytools::lcm_list(&denominators, &[])?
    };
    let mut scaled: Vec<Basic> = values
        .iter()
        .map(|v| cancel(&mul(&[v.clone(), scale.clone()])?))
        .collect::<PolyResult<_>>()?;
    let numerators: Vec<Basic> = scaled.iter().filter(|v| **v != Basic::zero()).cloned().collect();
    if !numerators.is_empty() {
        let common: Basic = crate::polys::polytools::gcd_list(&numerators, &[])?;
        if common != Basic::zero() {
            scale = cancel(&div(&scale, &common)?)?;
            scaled = scaled
                .iter()
                .map(|v| cancel(&div(v, &common)?))
                .collect::<PolyResult<_>>()?;
        }
    }
    let leading: Basic = scaled
        .iter()
        .rev()
        .find(|v| **v != Basic::zero())
        .cloned()
        .unwrap_or_else(Basic::one);
    let mut symbols: Vec<Basic> = sorted_symbols(leading.free_symbols());
    if symbols.is_empty() {
        symbols.push(n.clone());
    }
    match Poly::new(&leading, &symbols).and_then(|p| p.LC()) {
        Ok(lc) => {
            if negative(&lc) {
                scale = neg(&scale)?;
                scaled = scaled.iter().map(neg).collect::<PolyResult<_>>()?;
            }
        }
        Err(e) if out_of_scope(&e) => {}
        Err(e) => return Err(e),
    }
    Ok((scaled, scale))
}

/// Creative telescoping of `f(n, k)` in a ΠΣ-field
/// (`creative_telescoping`): coefficients `c_0(n), ..., c_r(n)` and a
/// certificate for the least order `r <= order`, or `None`.
pub fn creative_telescoping(
    f: &Basic,
    k: &Basic,
    n: &Basic,
    order: usize,
    extensions: &[Basic],
) -> PolyResult<Option<CreativeTelescoper>> {
    let f: &Basic = &to_expr(f)?;
    if !k.is_symbol() || !n.is_symbol() {
        return Err(PolyError::Type(
            "the index and the parameter must be symbols".to_string(),
        ));
    }
    let mut result: Option<CreativeTelescoper> = creative_telescoping_with(f, k, n, order, extensions)?;
    if result.is_none() {
        let (_, _, r) = split(f, k)?;
        if has_sums(&r, k) {
            let extra: Vec<Basic> = crate::concrete::karr::auto_extensions(&r, k)?
                .into_iter()
                .filter(|e| !extensions.contains(e))
                .collect();
            if !extra.is_empty() {
                let mut all: Vec<Basic> = extensions.to_vec();
                all.extend(extra);
                result = creative_telescoping_with(f, k, n, order, &all)?;
            }
        }
    }
    Ok(result)
}

struct TelescopingJob<'a> {
    f: &'a Basic,
    k: &'a Basic,
    n: &'a Basic,
    order: usize,
    extensions: &'a [Basic],
    u: Basic,
    t: Basic,
    r: Basic,
    others: Vec<Basic>,
}

impl WithField for TelescopingJob<'_> {
    type Out = Option<CreativeTelescoper>;

    fn run<Cs: Constants>(&self, mut builder: crate::concrete::karr::Builder<Cs>) -> PolyResult<Self::Out> {
        let (f, k, n) = (self.f, self.k, self.n);
        let mut tau: Basic = Basic::one();
        if self.t != Basic::one() {
            let alpha: Option<Basic> = crate::concrete::gosper::hypersimp(&self.t, k)?;
            let Some(alpha) = alpha.filter(|a| is_rational_function(a, std::slice::from_ref(k))) else {
                return Err(PolyError::Value(format!("{} is not hypergeometric in {}", self.t, k)));
            };
            let value: PiSigmaElement = match builder.field.from_expr(&alpha) {
                Ok(v) => v,
                Err(e @ (PolyError::ZeroDivision(_) | PolyError::Polys(..))) => {
                    return Err(PolyError::Value(format!(
                        "{} is not hypergeometric in {}: {}",
                        self.t, k, e
                    )))
                }
                Err(e) => return Err(e),
            };
            let (field, symbol) = builder.field.extend(Kind::Pi, &value, &self.t, None, None)?;
            builder.field = field;
            tau = symbol;
        }
        for e in self.extensions {
            builder.convert(e)?;
        }
        let mut mu_symbols: Vec<Basic> = vec![n.clone()];
        mu_symbols.extend(self.others.iter().cloned());
        let mut rho_symbols: Vec<Basic> = vec![n.clone(), k.clone()];
        rho_symbols.extend(self.others.iter().cloned());
        let mut converted: Vec<Basic> = Vec::new();
        for r in 0..=self.order {
            let shifted_n: Basic = add(&[n.clone(), int(r as i64)])?;
            let mu: Option<Basic> = rational_ratio(&subs(&self.u, n, &shifted_n)?, &self.u, &mu_symbols)?;
            let rho: Option<Basic> = rational_ratio(&subs(&self.t, n, &shifted_n)?, &self.t, &rho_symbols)?;
            let (Some(mu), Some(rho)) = (mu, rho) else {
                return Err(PolyError::Value(format!(
                    "the shifts of {} in {} are not rational multiples of it",
                    f, n
                )));
            };
            let product: Basic = mul(&[mu, rho, tau.clone(), subs(&self.r, n, &shifted_n)?])?;
            let element: PiSigmaElement = builder.convert(&product)?;
            converted.push(builder.field.to_expr(&element, false, false)?);
            let field = &builder.field;
            let fs: Vec<PiSigmaElement> = converted
                .iter()
                .map(|e| field.from_expr(e))
                .collect::<PolyResult<_>>()?;
            let Some(sols) = field.solve_embedded(&field.one, &fs, None)? else {
                return Ok(None);
            };
            for (c, g) in sols {
                let values: Vec<Basic> = c
                    .iter()
                    .map(|ci| field.to_expr(ci, false, false))
                    .collect::<PolyResult<_>>()?;
                if values.iter().all(|v| *v == Basic::zero()) {
                    continue;
                }
                let (coefficients, scale) = normalized(&values, n)?;
                let g: PiSigmaElement = field.mul(&g, &field.from_expr(&scale)?);
                let certificate: Basic = mul(&[self.u.clone(), field.to_expr(&g, true, true)?])?;
                let sigma_g: PolyResult<PiSigmaElement> = field.sigma(&g, 1);
                let shifted: Result<Basic, PolyError> = sigma_g
                    .clone()
                    .and_then(|s| mul(&[self.u.clone(), field.to_expr(&s, true, true)?]));
                let checked: Result<bool, PolyError> = (|| {
                    let mut lhs: PiSigmaElement = field.zero.clone();
                    for (ci, fi) in coefficients.iter().zip(&fs) {
                        lhs = field.add(&lhs, &field.mul(&field.from_expr(ci)?, fi));
                    }
                    let s: PiSigmaElement = sigma_g.clone()?;
                    // the difference is compared with zero: equal elements
                    // over Q(n) may compare different
                    Ok(field.sub(&field.sub(&s, &g), &lhs) == field.zero)
                })();
                let poles: Result<Poles, PolyError> = poles_of(field, k, n, &fs, &g);
                return Ok(Some(CreativeTelescoper {
                    term: f.clone(),
                    k: k.clone(),
                    n: n.clone(),
                    coefficients,
                    certificate,
                    shifted,
                    checked,
                    poles,
                }));
            }
        }
        Ok(None)
    }
}

fn creative_telescoping_with(
    f: &Basic,
    k: &Basic,
    n: &Basic,
    order: usize,
    extensions: &[Basic],
) -> PolyResult<Option<CreativeTelescoper>> {
    let (u, t, r) = split(f, k)?;
    let mut free: BTreeSet<Basic> = f.free_symbols();
    free.remove(k);
    free.remove(n);
    let others: Vec<Basic> = sorted_symbols(free);
    let mut symbols: BTreeSet<Basic> = f.free_symbols();
    symbols.insert(n.clone());
    for e in extensions {
        symbols.extend(e.free_symbols());
    }
    symbols.remove(k);
    let params: Vec<Basic> = sorted_symbols(symbols);
    let job = TelescopingJob {
        f,
        k,
        n,
        order,
        extensions,
        u,
        t,
        r,
        others,
    };
    crate::concrete::karr::with_field(k, &params, &job)
}

// ----------------------------------------------------------------------
// definite sums

/// `(m, s)` with `expr == m*n + s`, integers `m >= 0` and `s` (`_linear`).
fn linear(expr: &Basic, n: &Basic) -> PolyResult<Option<(i64, i64)>> {
    let p: Poly = match poly(expr, n) {
        Ok(p) => p,
        Err(e) if out_of_scope(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    if p.degree(0)? > 1 {
        return Ok(None);
    }
    let coeffs: Vec<Basic> = p.all_coeffs()?;
    let (m, s) = if coeffs.len() == 2 {
        (coeffs[0].clone(), coeffs[1].clone())
    } else {
        (Basic::zero(), coeffs[0].clone())
    };
    match (integer_of(&m), integer_of(&s)) {
        (Some(m), Some(s)) if m >= 0 => Ok(Some((m, s))),
        _ => Ok(None),
    }
}

/// `Sum(f, (k, lo, hi))` term by term with Karr's convention for
/// `hi < lo`, or `None` when a term is not defined (`_direct`).
fn direct(f: &Basic, k: &Basic, lo: i64, hi: i64) -> R2 {
    if hi < lo - 1 {
        return Ok(match direct(f, k, hi + 1, lo - 1)? {
            None => None,
            Some(v) => Some(neg(&v)?),
        });
    }
    let mut total: Basic = Basic::zero();
    for i in lo..=hi {
        let mut term: Basic = subs(f, k, &int(i))?;
        if has_head(&term, &|n| matches!(n.head(), Head::Sum | Head::Product)) {
            term = doit(&term)?;
        }
        if !finite(&term) {
            return Ok(None);
        }
        total = add(&[total, term])?;
    }
    Ok(Some(total))
}

/// `expr` with the binomials and harmonic numbers of shifted arguments
/// expanded and the hypergeometric terms simplified (`_tidy`).
fn tidy(expr: &Basic) -> R {
    Ok(within(|| combsimp(&expand_func(expr)?), STEP_SECONDS)?.unwrap_or_else(|| expr.clone()))
}

fn harmonic_order(h: &Basic) -> Basic {
    if h.args().len() > 1 {
        h.args()[1].clone()
    } else {
        Basic::one()
    }
}

/// `harmonic(x + c, r)` for integers `c` written through `harmonic(x, r)`
/// and `c` reciprocal powers (`_unshifted_harmonics`).
fn unshifted_harmonics(expr: &Basic) -> R {
    let rule = |h: &Basic| -> R {
        let x: &Basic = &h.args()[0];
        let order: Basic = harmonic_order(h);
        let (c, y) = crate::core::expr::as_coeff_add(x);
        let Some(c) = integer_of(&c).filter(|c| *c != 0) else {
            return Ok(h.clone());
        };
        if crate::core::assumptions::is_number(&y) {
            return Ok(h.clone());
        }
        let base: Basic = harmonic(&y, &order)?;
        let reciprocal = |i: i64| -> R { pow(&pow(&add(&[y.clone(), int(i)])?, &order)?, &int(-1)) };
        if c > 0 {
            let terms: Vec<Basic> = (1..=c).map(reciprocal).collect::<PolyResult<_>>()?;
            return add(&[base, add(&terms)?]);
        }
        let terms: Vec<Basic> = (0..-c).map(|i| reciprocal(-i)).collect::<PolyResult<_>>()?;
        sub(&base, &add(&terms)?)
    };
    crate::core::function::replace_where(expr, &|e: &Basic| e.is_function(Function::Harmonic), &rule)
}

/// `RisingFactorial(-x, n)` as `(-1)**n*FallingFactorial(x, n)`, the
/// falling factorials over factorials as binomials (`_falling`).
fn falling(expr: &Basic) -> R {
    let rule = |e: &Basic| -> R {
        let (x, m) = (&e.args()[0], &e.args()[1]);
        mul(&[
            pow(&int(-1), m)?,
            crate::functions::combinatorial::factorials::ff(&neg(x)?, m)?,
        ])
    };
    let replaced: Basic =
        crate::core::function::replace_where(expr, &|e: &Basic| e.is_function(Function::RisingFactorial), &rule)?;
    rewrite(&powsimp(&replaced)?, "binomial")
}

/// `expr` with `harmonic(n, r)` written through `harmonic(n + shift, r)`,
/// expanded (`_shifted_harmonics`).
fn shifted_harmonics(expr: &Basic, n: &Basic, shift: i64) -> R {
    let rule = |h: &Basic| -> R {
        let order: Basic = harmonic_order(h);
        if h.args()[0] != *n || integer_of(&order).is_none() {
            return Ok(h.clone());
        }
        let moved: Basic = harmonic(&add(&[n.clone(), int(shift)])?, &order)?;
        sub(&moved, &sub(&unshifted_harmonics(&moved)?, h)?)
    };
    cancel(&crate::core::function::replace_where(
        expr,
        &|e: &Basic| e.is_function(Function::Harmonic),
        &rule,
    )?)
}

/// The closed form simplified, the gamma functions of half-integer
/// arguments removed with the duplication formula and factorials or
/// binomials preferred when smaller (`_nicer`).
fn nicer(solution: &Basic, n: &Basic) -> R {
    use crate::functions::special::gamma_functions::gamma;
    let half: Basic = Basic::half();
    let third: Basic = Basic::rational(Int::ONE, Int::from(3))?;
    let pi: Basic = Basic::atom(Head::Pi);
    let rule = |g: &Basic| -> R {
        let shifted: Basic = sub(&g.args()[0], &half)?;
        let integral: bool = is(&shifted, Predicate::integer) == Some(true) || {
            is_polynomial(&shifted, n)
                && poly(&shifted, n)
                    .and_then(|p| p.all_coeffs())
                    .ok()
                    .and_then(|c| c.last().cloned())
                    .is_some_and(|c| is(&c, Predicate::integer) == Some(true))
                && is(&shifted, Predicate::integer) != Some(false)
        };
        if integral {
            let doubled: Basic = mul(&[int(2), shifted.clone()])?;
            return div(
                &mul(&[
                    gamma(&doubled)?,
                    pow(&pi, &half)?,
                    pow(&int(2), &sub(&int(1), &doubled)?)?,
                ])?,
                &gamma(&shifted)?,
            );
        }
        Ok(g.clone())
    };
    let duplicate = |e: &Basic| -> R {
        crate::core::function::replace_where(e, &|g: &Basic| g.is_function(Function::Gamma), &rule)
    };
    let triplicate = |e: &Basic| -> R {
        // gamma(z + 1/3) gamma(z + 2/3) = 2 pi 3**(1/2 - 3z) gamma(3z)/gamma(z)
        let gammas: Vec<Basic> = e
            .atoms(Some(&|g: &Basic| g.is_function(Function::Gamma)))
            .iter()
            .map(|g| g.args()[0].clone())
            .collect();
        for a in &gammas {
            for b in &gammas {
                let z: Basic = sub(a, &third)?;
                if sub(b, a)? == third && is_polynomial(&z, n) {
                    let three_z: Basic = mul(&[int(3), z.clone()])?;
                    let formula: Basic = div(
                        &mul(&[
                            int(2),
                            pi.clone(),
                            pow(&int(3), &sub(&half, &three_z)?)?,
                            gamma(&three_z)?,
                        ])?,
                        &mul(&[gamma(&z)?, gamma(b)?])?,
                    )?;
                    return subs_mapping(e, &[(gamma(a)?, formula)]);
                }
            }
        }
        Ok(e.clone())
    };
    let mut candidates: Vec<Basic> = vec![simplify(&combsimp(solution)?)?];
    candidates.push(combsimp(&duplicate(&candidates[0])?)?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(combsimp(&triplicate(&last)?)?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(combsimp(&rewrite(&last, "factorial")?)?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(rewrite(&last, "binomial")?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(simplify(&last)?);
    shortest(candidates)
}

/// The shortest of a few simplifications of the closed form (`_simplest`).
fn simplest(expr: &Basic, n: &Basic) -> R {
    let mut candidates: Vec<Basic> = vec![expr.clone()];
    let mut starts: Vec<Basic> = vec![expr.clone()];
    let transformations: [fn(&Basic) -> R; 2] = [unshifted_harmonics, falling];
    for transformation in transformations {
        if let Some(value) = within(|| transformation(expr), STEP_SECONDS)? {
            if value != *expr {
                starts.push(value);
            }
        }
    }
    if has_harmonic(expr) && !has_sum(expr) {
        for shift in [1, -1] {
            let base: Basic = if starts.len() > 1 {
                starts[1].clone()
            } else {
                expr.clone()
            };
            if let Some(value) = within(|| shifted_harmonics(&base, n, shift), STEP_SECONDS)? {
                starts.push(value);
            }
        }
    }
    let to_factorials = |e: &Basic| -> R {
        crate::core::function::replace_where(
            e,
            &|g: &Basic| {
                g.is_function(Function::Gamma)
                    && sub(&g.args()[0], &int(1))
                        .map(|a| is(&a, Predicate::nonnegative) == Some(true))
                        .unwrap_or(false)
            },
            &|g: &Basic| crate::functions::combinatorial::factorials::factorial(&sub(&g.args()[0], &int(1))?),
        )
    };
    let with_nicer: bool = !has_sum(expr) && !has_harmonic(expr);
    for start in starts {
        candidates.push(start.clone());
        let simplifications: Vec<Box<dyn Fn(&Basic) -> R>> = {
            let mut v: Vec<Box<dyn Fn(&Basic) -> R>> = vec![
                Box::new(|e: &Basic| factor_terms(&combsimp(e)?)),
                Box::new(|e: &Basic| {
                    if !has_sum(e) {
                        factor_terms(&cancel(e)?)
                    } else {
                        Ok(e.clone())
                    }
                }),
                Box::new(|e: &Basic| factor_terms(&combsimp(&expand(e)?)?)),
                Box::new(to_factorials),
            ];
            if with_nicer {
                v.push(Box::new(|e: &Basic| nicer(e, n)));
            }
            v
        };
        for simplification in &simplifications {
            if let Some(value) = within(|| simplification(&start), STEP_SECONDS)? {
                candidates.push(value);
            }
        }
    }
    shortest(candidates)
}

/// The integer roots of the factors of `expr` which are polynomials in `n`
/// alone (`_integer_roots`).
fn integer_roots(expr: &Basic, n: &Basic) -> PolyResult<Vec<i64>> {
    let mut roots: Vec<i64> = Vec::new();
    let factors: Vec<(Basic, Basic)> = match factor_list(expr, &[]) {
        Ok((_, factors)) => factors,
        Err(e) if out_of_scope(&e) => return Ok(roots),
        Err(e) => return Err(e),
    };
    for (factor, _) in factors {
        let symbols: BTreeSet<Basic> = factor.free_symbols();
        if symbols.len() != 1 || !symbols.contains(n) || !is_polynomial(&factor, n) {
            continue;
        }
        let p: Poly = poly(&factor, n)?;
        if p.degree(0)? == 1 {
            let coeffs: Vec<Basic> = p.all_coeffs()?;
            let root: Basic = neg(&div(&coeffs[1], &coeffs[0])?)?;
            if let Some(v) = integer_of(&root) {
                roots.push(v);
            }
        }
    }
    Ok(roots)
}

/// The poles of the representations in the field (`_poles`).
fn poles_of<Cs: Constants>(
    field: &crate::concrete::pisigma::PiSigmaField<Cs>,
    k: &Basic,
    n: &Basic,
    fs: &[PiSigmaElement],
    g: &PiSigmaElement,
) -> PolyResult<Poles> {
    let mut elements: Vec<PiSigmaElement> = fs.to_vec();
    elements.push(g.clone());
    elements.push(field.sigma(g, 1)?);
    for e in &field.extensions {
        elements.push(field.from_expr(&e.value)?);
    }
    let gens: BTreeSet<Basic> = field.extensions.iter().map(|e| e.symbol.clone()).collect();
    let mut in_k: Vec<(Rat, Rat)> = Vec::new();
    let mut in_n: Vec<i64> = Vec::new();
    for element in &elements {
        for (is_numer, part) in [(true, field.numer_expr(element)?), (false, field.denom_expr(element)?)] {
            let (numerator, denominator) = numer_denom(&together(&part)?)?;
            in_n.extend(integer_roots(&denominator, n)?);
            if is_numer {
                continue;
            }
            let factors: Vec<(Basic, Basic)> = match factor_list(&numerator, &[]) {
                Ok((_, factors)) => factors,
                Err(e) if out_of_scope(&e) => continue,
                Err(e) => return Err(e),
            };
            for (factor, _) in factors {
                let symbols: BTreeSet<Basic> = factor.free_symbols();
                let with_gens: bool = symbols.iter().any(|s| gens.contains(s));
                if with_gens || !symbols.contains(k) {
                    if !with_gens && !symbols.contains(k) && symbols.len() == 1 && symbols.contains(n) {
                        in_n.extend(integer_roots(&factor, n)?);
                    }
                    continue;
                }
                let p: Poly = poly(&factor, k)?;
                if p.degree(0)? != 1 {
                    continue;
                }
                let coeffs: Vec<Basic> = p.all_coeffs()?;
                let root: Basic = cancel(&neg(&div(&coeffs[1], &coeffs[0])?)?)?;
                if !root.free_symbols().iter().all(|s| s == n) {
                    continue;
                }
                let q: Poly = match poly(&root, n) {
                    Ok(q) => q,
                    Err(e) if out_of_scope(&e) => continue,
                    Err(e) => return Err(e),
                };
                if q.degree(0)? > 1 {
                    continue;
                }
                let qc: Vec<Basic> = q.all_coeffs()?;
                let (alpha, beta) = if qc.len() == 2 {
                    (qc[0].clone(), qc[1].clone())
                } else {
                    (Basic::zero(), qc[0].clone())
                };
                if let (Some(alpha), Some(beta)) = (rational_of(&alpha), rational_of(&beta)) {
                    in_k.push((alpha, beta));
                }
            }
        }
    }
    Ok((in_k, in_n))
}

/// A pole of the certificate inside the summation range (`_Unsupported`).
struct Unsupported;

fn ceil(x: &Rat) -> i64 {
    let (num, den) = (crate::numbers::numer(x), crate::numbers::denom(x));
    let q: Int = -crate::numbers::div_floor(&(-num), &den);
    i64::try_from(&q).unwrap_or(i64::MAX)
}

fn rat(v: i64) -> Rat {
    Rat::from(v)
}

fn is_integer_rat(r: &Rat) -> bool {
    r.is_int()
}

fn floor_int(r: &Rat) -> i64 {
    let (num, den) = (crate::numbers::numer(r), crate::numbers::denom(r));
    // int() of a Fraction truncates towards zero
    let q: Int = crate::numbers::div_rem_trunc(&num, &den).0;
    i64::try_from(&q).unwrap_or(0)
}

/// `(j0, j1, n_min)`: the telescoped range is `[a(n + r) + j0, b(n) - j1]`,
/// free of poles for `n >= n_min` (`_window`).
fn window(poles: &[(Rat, Rat)], lower: (i64, i64), upper: (i64, i64), r: i64) -> Result<(i64, i64, i64), Unsupported> {
    let (m1, s1) = lower;
    let (m2, s2) = upper;
    let bottom: i64 = s1 + m1 * r;
    let (mut j0, mut j1, mut n_min) = (0i64, 0i64, 0i64);
    for (alpha, beta) in poles {
        if is_integer_rat(alpha) && !is_integer_rat(beta) {
            continue;
        }
        if *alpha == rat(m1) && m1 < m2 {
            if *beta >= rat(bottom + j0) {
                j0 = floor_int(beta) - bottom + 1;
            }
        } else if *alpha == rat(m2) && m1 < m2 {
            if *beta <= rat(s2 - j1) {
                j1 = s2 - floor_int(beta) + 1;
            }
        } else if m1 == m2 && *alpha == rat(m1) {
            if rat(bottom) <= *beta && *beta <= rat(s2) {
                return Err(Unsupported);
            }
        } else if rat(m1) < *alpha && *alpha < rat(m2) {
            return Err(Unsupported);
        }
    }
    for (alpha, beta) in poles {
        if is_integer_rat(alpha) && !is_integer_rat(beta) {
            continue;
        }
        if *alpha < rat(m1) {
            let bound: Rat = (beta.clone() - rat(bottom) - rat(j0) + rat(1)) / (rat(m1) - alpha.clone());
            n_min = n_min.max(ceil(&bound));
        } else if *alpha > rat(m2) {
            let bound: Rat = (rat(s2) - rat(j1) - beta.clone() + rat(1)) / (alpha.clone() - rat(m2));
            n_min = n_min.max(ceil(&bound));
        }
    }
    // a nonempty telescoped range: bottom + j0 <= b(n) - j1 + 1
    if m2 > m1 {
        n_min = n_min.max(ceil(&(rat(bottom + j0 + j1 - 1 - s2) / rat(m2 - m1))));
    } else if bottom + j0 > s2 - j1 + 1 {
        return Err(Unsupported);
    }
    Ok((j0, j1, n_min))
}

/// A summation index which does not occur in `expr` (`_index`).
fn index(expr: &Basic) -> Basic {
    let names: BTreeSet<String> = expr.free_symbols().iter().map(name_of).collect();
    for name in ["j", "i", "l", "m", "p", "q"] {
        if !names.contains(name) {
            return symbol(name, &[(Predicate::integer, true)]);
        }
    }
    dummy("j", &[(Predicate::integer, true)])
}

fn sum_node(term: &Basic, j: &Basic, lo: &Basic, hi: &Basic) -> R {
    crate::concrete::expr_with_limits::sum_(term, &[tuple(vec![j.clone(), lo.clone(), hi.clone()])])
}

/// `Sum(term, (j, lo, hi))` with the index shifted to the simplest summand
/// (`_prettier_sum`).
fn prettier_sum(term: &Basic, j: &Basic, lo: &Basic, hi: &Basic) -> R {
    let mut best: Option<((i64, usize), Basic)> = None;
    let rational: bool = is_rational_function(term, std::slice::from_ref(j));
    for d in -2i64..3 {
        let moved: Basic = subs(term, j, &sub(j, &int(d))?)?;
        let shifted: Basic = if rational { cancel(&moved)? } else { combsimp(&moved)? };
        let candidate: Basic = sum_node(&shifted, j, &add(&[lo.clone(), int(d)])?, &add(&[hi.clone(), int(d)])?)?;
        let function: Basic = if matches!(candidate.head(), Head::Sum) {
            candidate.args()[0].clone()
        } else {
            candidate.clone()
        };
        let key: (i64, usize) = (ops(&function)?, function.to_string().chars().count());
        if best.as_ref().map_or(true, |(b, _)| key < *b) {
            best = Some((key, candidate));
        }
    }
    Ok(best.expect("five candidates").1)
}

/// `harmonic(d*j)` for the leading coefficients `d > 1` of the linear
/// factors of the denominator of `term` (`_harmonic_extensions`).
fn harmonic_extensions(term: &Basic, j: &Basic) -> PolyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    let denominator: Basic = numer_denom(&together(term)?)?.1;
    let factors: Vec<(Basic, Basic)> = match factor_list(&denominator, &[]) {
        Ok((_, factors)) => factors,
        Err(e) if out_of_scope(&e) => return Ok(result),
        Err(e) => return Err(e),
    };
    for (factor, _) in factors {
        if factor.has(j) && is_polynomial(&factor, j) && poly(&factor, j)?.degree(0)? == 1 {
            let d: Basic = poly(&factor, j)?.LC()?;
            if let Some(d) = integer_of(&d).filter(|d| d.abs() > 1) {
                result.push(crate::functions::combinatorial::numbers::harmonic(
                    &mul(&[int(d.abs()), j.clone()])?,
                    None,
                )?);
            }
        }
    }
    Ok(result)
}

/// `sum_{j=lo}^{hi} term`: a closed form by Karr's algorithm, else the sum
/// of the closed forms of the terms, the others left as sums (`_sum`).
fn sum(term: &Basic, j: &Basic, lo: &Basic, hi: &Basic) -> R {
    let term: Basic = tidy(term)?;
    if term == Basic::zero() {
        return Ok(Basic::zero());
    }
    let karr = |t: &Basic| -> R2 {
        let compute = || -> R {
            let limits: [Basic; 3] = [j.clone(), lo.clone(), hi.clone()];
            let mut value: Option<Basic> =
                crate::concrete::karr::karr_sum(t, j, Some((&limits[1], &limits[2])), &[], true, false)?;
            if value.is_none() {
                let extra: Vec<Basic> = harmonic_extensions(t, j)?;
                if !extra.is_empty() {
                    value = crate::concrete::karr::karr_sum(t, j, Some((&limits[1], &limits[2])), &extra, true, false)?;
                }
            }
            let Some(value) = value else {
                return Err(PolyError::NotImplemented("no closed form".to_string()));
            };
            // g(n) - g(0) for the sum up to n - 1 has harmonic(n - 1),
            // undefined at n = 0
            cancel(&unshifted_harmonics(&value)?)
        };
        within(compute, STEP_SECONDS)
    };
    if let Some(whole) = karr(&term)? {
        return Ok(whole);
    }
    let mut total: Basic = Basic::zero();
    let mut rest: Basic = Basic::zero();
    let expanded: Option<Basic> = within(|| expand(&term), STEP_SECONDS)?;
    let source: Basic = expanded.unwrap_or_else(|| term.clone());
    let parts: Vec<Basic> = if source.is_add() {
        source.args().to_vec()
    } else {
        vec![source]
    };
    for t in parts {
        let t: Basic = tidy(&powsimp(&t)?)?;
        match karr(&t)? {
            None => rest = add(&[rest, t])?,
            Some(value) => total = add(&[total, value])?,
        }
    }
    if rest != Basic::zero() {
        total = add(&[total, prettier_sum(&tidy(&rest)?, j, lo, hi)?])?;
    }
    Ok(total)
}

/// `prod_{j=n0}^{n-1} ratio(j)` for a rational `ratio`, or `None` when a
/// factor vanishes or has a pole for some `j >= n0` (`_product`).
fn product(ratio: &Basic, n: &Basic, n0: i64) -> R2 {
    use crate::functions::special::gamma_functions::gamma;
    let j: Basic = dummy("j", &[(Predicate::integer, true)]);
    let at_j: Basic = subs(ratio, n, &j)?;
    let (numerator, denominator) = numer_denom(&together(&at_j)?)?;
    let mut constant: Basic = Basic::one();
    let mut exponents: Vec<(Basic, i64)> = Vec::new();
    for (part, sign) in [(numerator, 1i64), (denominator, -1i64)] {
        let (coefficient, factors) = match factor_list(&part, std::slice::from_ref(&j)) {
            Ok(found) => found,
            Err(e) if out_of_scope(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        constant = mul(&[constant, pow(&coefficient, &int(sign))?])?;
        for (factor, multiplicity) in factors {
            let p: Poly = poly(&factor, &j)?;
            if p.degree(0)? != 1 {
                let limit = (j.clone(), int(n0), sub(n, &int(1))?);
                let value: Option<Basic> = within(|| crate::concrete::products::product(&at_j, &limit), STEP_SECONDS)?;
                return match value {
                    Some(v) if !has_head(&v, &|x| matches!(x.head(), Head::Product)) => Ok(Some(tidy(&v)?)),
                    _ => Ok(None),
                };
            }
            let coeffs: Vec<Basic> = p.all_coeffs()?;
            let (lead, tail) = (&coeffs[0], &coeffs[1]);
            let m: i64 = int_of(&multiplicity)?;
            constant = mul(&[constant, pow(lead, &int(sign * m))?])?;
            let a: Basic = cancel(&div(tail, lead)?)?;
            if integer_of(&a).is_some_and(|v| n0 + v <= 0) {
                return Ok(None);
            }
            match exponents.iter_mut().find(|(b, _)| *b == a) {
                Some((_, e)) => *e += sign * m,
                None => exponents.push((a, sign * m)),
            }
        }
    }
    let mut result: Basic = pow(&constant, &sub(n, &int(n0))?)?;
    let mut classes: Vec<Vec<Basic>> = Vec::new();
    for (a, _) in &exponents {
        let mut placed: bool = false;
        for members in classes.iter_mut() {
            if integer_of(&cancel(&sub(a, &members[0])?)?).is_some() {
                members.push(a.clone());
                placed = true;
                break;
            }
        }
        if !placed {
            classes.push(vec![a.clone()]);
        }
    }
    for members in &classes {
        let first: &Basic = &members[0];
        let mut base: Basic = first.clone();
        let mut least: i64 = i64::MAX;
        for a in members {
            let d: i64 = int_of(&cancel(&sub(a, first)?)?)?;
            if d < least {
                least = d;
                base = a.clone();
            }
        }
        let mut total: i64 = 0;
        for a in members {
            let e: i64 = exponents.iter().find(|(b, _)| b == a).map(|(_, e)| *e).unwrap_or(0);
            total += e;
            let shift: i64 = int_of(&cancel(&sub(a, &base)?)?)?;
            // Gamma(x + a)/Gamma(x + base) = (x + base)*...*(x + a - 1)
            let mut factors: Vec<Basic> = Vec::new();
            for i in 0..shift {
                factors.push(div(
                    &add(&[n.clone(), base.clone(), int(i)])?,
                    &add(&[int(n0), base.clone(), int(i)])?,
                )?);
            }
            let rising: Basic = mul(&factors)?;
            result = mul(&[result, pow(&rising, &int(e))?])?;
        }
        if total != 0 && crate::core::assumptions::is_number(&base) {
            let ratio_: Basic = div(
                &gamma(&add(&[n.clone(), base.clone()])?)?,
                &gamma(&add(&[int(n0), base.clone()])?)?,
            )?;
            result = mul(&[result, pow(&ratio_, &int(total))?])?;
        } else if total != 0 {
            let rf: Basic =
                crate::functions::combinatorial::factorials::rf(&add(&[int(n0), base.clone()])?, &sub(n, &int(n0))?)?;
            result = mul(&[result, pow(&rf, &int(total))?])?;
        }
    }
    let special: bool = has_head(&result, &|x| {
        x.is_function(Function::Gamma) || x.is_function(Function::RisingFactorial)
    });
    Ok(Some(if special { result } else { cancel(&result)? }))
}

/// A hypergeometric solution of the recurrence with coefficients `c`, in
/// its nicest form (`_hypergeometric_solution`).
fn hypergeometric_solution(c: &[Basic], n: &Basic) -> R2 {
    let first = || -> R {
        for term in crate::solvers::recurr::rsolve_hypergeometric(c, n)? {
            if term != Basic::zero() {
                return Ok(term);
            }
        }
        Err(PolyError::NotImplemented("no hypergeometric solution".to_string()))
    };
    let Some(value) = within(first, 4.0 * STEP_SECONDS)? else {
        return Ok(None);
    };
    let nicer_value: Option<Basic> = within(|| nicer(&value, n), STEP_SECONDS)?;
    Ok(Some(nicer_value.unwrap_or(value)))
}

/// The general solution of `sum_i c_i(n) S(n + i) = rhs(n)` for `n >= n0`
/// in d'Alembertian terms, with its constants (`_general_solution`).
fn general_solution(c: &[Basic], rhs: &Basic, n: &Basic, n0: i64) -> PolyResult<Option<(Basic, Vec<Basic>)>> {
    let r: usize = c.len() - 1;
    if r == 0 {
        return Ok(Some((cancel(&div(rhs, &c[0])?)?, Vec::new())));
    }
    let constant: Basic = dummy("C", &[]);
    if r == 1 {
        let ratio: Basic = cancel(&neg(&div(&c[0], &c[1])?)?)?;
        let Some(p) = product(&ratio, n, n0)? else {
            return Ok(None);
        };
        let beta: Basic = cancel(&div(rhs, &c[1])?)?;
        if beta == Basic::zero() {
            return Ok(Some((mul(&[constant.clone(), p])?, vec![constant])));
        }
        let j: Basic = index(&add(&[rhs.clone(), p.clone()])?);
        let term: Basic = tidy(&div(&subs(&beta, n, &j)?, &subs(&p, n, &add(&[j.clone(), int(1)])?)?)?)?;
        let summed: Basic = sum(&term, &j, &int(n0), &sub(n, &int(1))?)?;
        return Ok(Some((mul(&[p, add(&[constant.clone(), summed])?])?, vec![constant])));
    }
    let Some(h) = hypergeometric_solution(c, n)? else {
        return Ok(None);
    };
    let quotient = || -> R {
        crate::concrete::gosper::hypersimp(&h, n)?
            .ok_or_else(|| PolyError::NotImplemented("not hypergeometric".to_string()))
    };
    let q: Option<Basic> = within(quotient, STEP_SECONDS)?;
    let rho: Option<Basic> = match q.filter(|q| is_rational_function(q, std::slice::from_ref(n))) {
        Some(q) => Some(q),
        None => rational_ratio(&subs(&h, n, &add(&[n.clone(), int(1)])?)?, &h, std::slice::from_ref(n))?,
    };
    let Some(rho) = rho else {
        return Ok(None);
    };
    // S = h*y, z = y(n + 1) - y(n): sum_t d_t z(n + t) = rhs/h with
    // d_t = sum_{i > t} c_i Q_i, Q_i = h(n + i)/h(n)
    let mut q_list: Vec<Basic> = vec![Basic::one()];
    for i in 1..=r {
        let shifted: Basic = subs(&rho, n, &add(&[n.clone(), int(i as i64 - 1)])?)?;
        let next: Basic = cancel(&mul(&[q_list.last().expect("nonempty").clone(), shifted])?)?;
        q_list.push(next);
    }
    let mut d: Vec<Basic> = Vec::new();
    for t in 0..r {
        let mut terms: Vec<Basic> = Vec::new();
        for i in t + 1..=r {
            terms.push(mul(&[c[i].clone(), q_list[i].clone()])?);
        }
        d.push(cancel(&add(&terms)?)?);
    }
    let Some((z, mut constants)) = general_solution(&d, &tidy(&div(rhs, &h)?)?, n, n0)? else {
        return Ok(None);
    };
    let j: Basic = index(&add(&[z.clone(), rhs.clone()])?);
    let summed: Basic = sum(&subs(&z, n, &j)?, &j, &int(n0), &sub(n, &int(1))?)?;
    constants.push(constant.clone());
    Ok(Some((mul(&[h, add(&[constant, summed])?])?, constants)))
}

/// The value of `e` at the `values` of its symbols as a complex number,
/// `None` when it is not a reliable finite number (`_numeric`).
fn numeric(e: &Basic, digits: usize, values: &SumPoint) -> Option<(f64, f64)> {
    if !values.is_empty() {
        return crate::numeric::reliable_complex(e, digits, values);
    }
    if !e.free_symbols().is_empty() {
        return None;
    }
    let z = crate::core::evalf::numeric_value(e, (3.33 * digits as f64) as usize + 20)?;
    let (re_part, im_part) = crate::core::evalf::parts(&z);
    let (re_value, im_value) = (re_part.to_f64().value(), im_part.to_f64().value());
    if !re_value.is_finite() || !im_value.is_finite() {
        return None;
    }
    Some((re_value, im_value))
}

/// Whether `direct == value`: exactly when it is proved, else numerically
/// at sample values of the symbols (`_agree`).
fn agree(direct_value: &Basic, value: &Basic) -> PolyResult<Option<bool>> {
    let difference: Basic = sub(direct_value, value)?;
    let exact: Option<Basic> = within(|| cancel(&combsimp(&expand_func(&difference)?)?), STEP_SECONDS)?;
    if exact == Some(Basic::zero()) {
        return Ok(Some(true));
    }
    let digits: usize = crate::settings::settings().precision;
    let symbols: Vec<Basic> = sorted_symbols(difference.free_symbols());
    let samples: Vec<SumPoint> = if symbols.is_empty() {
        vec![Vec::new()]
    } else {
        let denominators: [i64; 8] = [7, 11, 13, 17, 19, 23, 29, 31];
        let mut out: Vec<SumPoint> = Vec::new();
        for primes in [[31i64, 37, 41, 43, 47, 53, 59, 61], [67, 71, 73, 79, 83, 89, 97, 101]] {
            let mut point: SumPoint = Vec::new();
            for ((s, p), q) in symbols.iter().zip(primes).zip(denominators) {
                point.push((s.clone(), Basic::rational(Int::from(p), Int::from(q))?));
            }
            out.push(point);
        }
        out
    };
    for values in &samples {
        let (Some(error), Some(size)) = (
            numeric(&difference, digits, values),
            numeric(direct_value, digits, values),
        ) else {
            return Ok(None);
        };
        let magnitude = |z: (f64, f64)| z.0.hypot(z.1);
        if magnitude(error) > 10f64.powi(-(digits as i32 - 10)) * magnitude(size).max(1.0) {
            return Ok(Some(false));
        }
    }
    Ok(Some(true))
}

/// The definite sum `sum_{k=a}^{b} f(n, k)` by creative telescoping in a
/// ΠΣ-field, or `None` (`definite_sum`).
pub fn definite_sum(
    f: &Basic,
    limits: &[Basic],
    n: Option<&Basic>,
    order: usize,
    extensions: &[Basic],
    special_values: bool,
) -> R2 {
    if limits.len() != 3 {
        let shown: Vec<String> = limits.iter().map(|l| l.to_string()).collect();
        return Err(PolyError::Value(format!(
            "the limits are (k, a, b), not ({})",
            shown.join(", ")
        )));
    }
    let f: &Basic = &to_expr(f)?;
    let k: &Basic = &limits[0];
    if !k.is_symbol() {
        return Err(PolyError::Type(format!("{} is not a symbol", k)));
    }
    let (a, b) = (&limits[1], &limits[2]);
    let n: Basic = match n {
        Some(n) => n.clone(),
        None => {
            let mut free: BTreeSet<Basic> = a.free_symbols();
            free.extend(b.free_symbols());
            free.remove(k);
            let candidates: Vec<Basic> = sorted_symbols(free);
            if candidates.len() != 1 {
                return Ok(None);
            }
            candidates[0].clone()
        }
    };
    // work with an integer n and k, for which binomial(n, n) and the like
    // evaluate
    let big_n: Basic = dummy(
        &name_of(&n),
        &[(Predicate::integer, true), (Predicate::nonnegative, true)],
    );
    let big_k: Basic = dummy(&name_of(k), &[(Predicate::integer, true)]);
    let g: Basic = subs_mapping(f, &[(n.clone(), big_n.clone()), (k.clone(), big_k.clone())])?;
    let lower: Option<(i64, i64)> = linear(&subs_mapping(a, &[(n.clone(), big_n.clone())])?, &big_n)?;
    let upper: Option<(i64, i64)> = linear(&subs_mapping(b, &[(n.clone(), big_n.clone())])?, &big_n)?;
    let (Some(lower), Some(upper)) = (lower, upper) else {
        return Ok(None);
    };
    if lower.0 > upper.0 || (lower.0 == 0 && upper.0 == 0) {
        return Ok(None);
    }
    let (u, t, r) = split(&g, &big_k)?;
    let mut h: Basic = mul(&[t, r])?;
    let mut result: Option<Basic> = definite_sum_terms(&h, &big_k, &big_n, lower, upper, order, extensions)?;
    if result.is_none() && !h.is_add() {
        if let Some(expanded) = within(|| expand(&h), STEP_SECONDS)? {
            if expanded.is_add() {
                h = expanded;
            }
        }
    }
    if result.is_none() && h.is_add() {
        let mut total: Basic = Basic::zero();
        for term in h.args() {
            let Some(value) = definite_sum_terms(term, &big_k, &big_n, lower, upper, order, extensions)? else {
                return Ok(None);
            };
            total = add(&[total, value])?;
        }
        result = Some(total);
    }
    let Some(result) = result else {
        return Ok(None);
    };
    let closed: Basic = subs_mapping(&mul(&[u, result])?, &[(big_n, n.clone()), (big_k, k.clone())])?;
    if !special_values {
        return Ok(Some(closed));
    }
    let at_point = |point: &SumPoint| -> R2 {
        let shifted_limits: [Basic; 3] = [k.clone(), a.clone(), b.clone()];
        let moved: Vec<Basic> = extensions
            .iter()
            .map(|e| subs_mapping(e, point))
            .collect::<PolyResult<_>>()?;
        definite_sum(&subs_mapping(f, point)?, &shifted_limits, Some(&n), order, &moved, true)
    };
    Ok(Some(crate::concrete::karr::with_special_values_sum(
        &closed,
        f,
        &[k.clone(), n.clone()],
        &[a.clone(), b.clone()],
        &at_point,
        true,
        None,
    )?))
}

/// `(m, s)` evaluated at `v`: `m*v + s`.
fn at(limit: (i64, i64), v: &Basic) -> R {
    add(&[mul(&[int(limit.0), v.clone()])?, int(limit.1)])
}

/// [`definite_sum`] of one summand (`_definite_sum`).
fn definite_sum_terms(
    f: &Basic,
    k: &Basic,
    n: &Basic,
    lower: (i64, i64),
    upper: (i64, i64),
    order: usize,
    extensions: &[Basic],
) -> R2 {
    let z: CreativeTelescoper = match creative_telescoping(f, k, n, order, extensions) {
        Ok(Some(z)) => z,
        Ok(None) | Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let r: i64 = z.order() as i64;
    let c: &Vec<Basic> = &z.coefficients;
    let ((m1, _), (m2, _)) = (lower, upper);
    let (poles, poles_in_n) = z.poles.clone()?;
    let Ok((j0, j1, n_min)) = window(&poles, lower, upper, r) else {
        return Ok(None);
    };
    let bottom: Basic = add(&[at(lower, &add(&[n.clone(), int(r)])?)?, int(j0)])?;
    let top: Basic = sub(&at(upper, n)?, &int(j1))?;
    // sum_i c_i S(n + i) = g(n, top + 1) - g(n, bottom) + the terms of each
    // S(n + i) outside [bottom, top]
    let mut rhs: Basic = sub(
        &subs(&z.shifted_certificate()?, k, &top)?,
        &subs(&z.certificate, k, &bottom)?,
    )?;
    for (i, ci) in c.iter().enumerate() {
        if *ci == Basic::zero() {
            continue;
        }
        let i: i64 = i as i64;
        let fi: Basic = subs(f, n, &add(&[n.clone(), int(i)])?)?;
        let below: i64 = m1 * (r - i) + j0;
        let above: i64 = m2 * i + j1;
        let mut extra: Vec<Basic> = Vec::new();
        let start: Basic = at(lower, &add(&[n.clone(), int(i)])?)?;
        for t in 0..below {
            extra.push(subs(&fi, k, &add(&[start.clone(), int(t)])?)?);
        }
        for t in 0..above {
            extra.push(subs(&fi, k, &add(&[top.clone(), int(1 + t)])?)?);
        }
        rhs = add(&[rhs, mul(&[ci.clone(), add(&extra)?])?])?;
    }
    let rhs: Basic = tidy(&rhs)?;
    if !finite(&rhs) {
        return Ok(None);
    }
    let mut starts: Vec<i64> = vec![n_min, 0];
    starts.extend(poles_in_n.iter().map(|v| v + 1));
    starts.extend(
        integer_roots(c.last().expect("nonempty"), n)?
            .into_iter()
            .map(|v| v + 1),
    );
    starts.extend(
        integer_roots(&numer_denom(&together(&rhs)?)?.1, n)?
            .into_iter()
            .map(|v| v + 1),
    );
    if r == 1 {
        starts.extend(integer_roots(&c[0], n)?.into_iter().map(|v| v + 1));
    }
    let n0: i64 = *starts.iter().max().expect("nonempty");
    let direct_at = |v: i64| -> R2 {
        let value: Basic = int(v);
        let lo: i64 = int_of(&at(lower, &value)?)?;
        let hi: i64 = int_of(&at(upper, &value)?)?;
        direct(&subs(f, n, &value)?, k, lo, hi)
    };
    // a recurrence in steps of d > 1 is one recurrence for each residue
    // class of n modulo d
    let mut d: i64 = 0;
    for (i, ci) in c.iter().enumerate() {
        if *ci != Basic::zero() {
            d = gcd(d, i as i64);
        }
    }
    if d <= 1 {
        let Some((solution, exceptions)) = solve_and_check(c, &rhs, n, n0, &direct_at)? else {
            return Ok(None);
        };
        if exceptions.is_empty() {
            return Ok(Some(solution));
        }
        let mut pieces: Vec<(Basic, Basic)> = Vec::new();
        for (v, value) in exceptions {
            pieces.push((value, crate::core::relational::eq(n, &int(v))?));
        }
        pieces.push((solution, crate::core::logic::true_()));
        return Ok(Some(crate::functions::elementary::piecewise::piecewise(&pieces)?));
    }
    let m: Basic = dummy("m", &[(Predicate::integer, true), (Predicate::nonnegative, true)]);
    let mut pieces: Vec<(Basic, Basic)> = Vec::new();
    let mut branches: Vec<(Basic, Basic)> = Vec::new();
    for residue in 0..d {
        let point: Basic = add(&[mul(&[int(d), m.clone()])?, int(residue)])?;
        let class_c: Vec<Basic> = (0..=(r / d) as usize)
            .map(|i| subs(&c[d as usize * i], n, &point))
            .collect::<PolyResult<_>>()?;
        let class_rhs: Basic = subs(&rhs, n, &point)?;
        let m0: i64 = 0i64.max(-((residue - n0).div_euclid(d)));
        let class_direct = |v: i64| direct_at(d * v + residue);
        let Some((solution, exceptions)) = solve_and_check(&class_c, &class_rhs, &m, m0, &class_direct)? else {
            return Ok(None);
        };
        for (v, value) in exceptions {
            pieces.push((value, crate::core::relational::eq(n, &int(d * v + residue))?));
        }
        let branch: Basic = simplest(&subs(&solution, &m, &div(&sub(n, &int(residue))?, &int(d))?)?, n)?;
        let condition: Basic = if residue == d - 1 {
            crate::core::logic::true_()
        } else {
            crate::core::relational::eq(&crate::core::modulo::modulo(n, &int(d))?, &int(residue))?
        };
        branches.push((branch, condition));
    }
    pieces.extend(branches);
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&pieces)?))
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn sum_limits(node: &Basic) -> Vec<(Basic, Basic, Basic)> {
    crate::concrete::expr_with_limits::limits(node)
}

/// Whether an unevaluated `Sum` of `expr` has, at `n = v`, an upper limit
/// below its lower limit minus one (`_reversed_at`).
fn reversed_at(expr: &Basic, n: &Basic, v: i64) -> PolyResult<bool> {
    for node in expr.atoms(Some(&|e: &Basic| matches!(e.head(), Head::Sum))) {
        for (_, lo, hi) in sum_limits(&node) {
            let length: Basic = subs(&add(&[hi, neg(&lo)?, int(1)])?, n, &int(v))?;
            if crate::core::assumptions::is_number(&length) && is(&length, Predicate::negative) == Some(true) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// `expr` with the lower limit of every `Sum(t, (j, a, b(n)))` lowered to
/// `b(start) + 1` when it is above (`_sums_ordered_from`).
fn sums_ordered_from(expr: &Basic, n: &Basic, start: i64) -> R2 {
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    for node in expr.atoms(Some(&|e: &Basic| matches!(e.head(), Head::Sum))) {
        let lims = sum_limits(&node);
        if lims.len() != 1 {
            continue;
        }
        let (j, lo, hi) = &lims[0];
        let Some(lo_value) = integer_of(lo) else {
            continue;
        };
        if !is_polynomial(hi, n) {
            continue;
        }
        let p: Poly = poly(hi, n)?;
        if p.degree(0)? > 1 || negative(&p.LC()?) {
            continue;
        }
        let bottom: Basic = add(&[subs(hi, n, &int(start))?, int(1)])?;
        let Some(bottom_value) = integer_of(&bottom) else {
            continue;
        };
        if bottom_value >= lo_value {
            continue;
        }
        let term: Basic = node.args()[0].clone();
        let mut taken: Vec<Basic> = Vec::new();
        for t in bottom_value..lo_value {
            let value: Basic = subs(&term, j, &int(t))?;
            if !finite(&value) {
                return Ok(None);
            }
            taken.push(value);
        }
        replacements.push((node.clone(), sub(&sum_node(&term, j, &bottom, hi)?, &add(&taken)?)?));
    }
    if replacements.is_empty() {
        Ok(Some(expr.clone()))
    } else {
        Ok(Some(subs_mapping(expr, &replacements)?))
    }
}

type Checked = Option<(Basic, Vec<(i64, Basic)>)>;

/// The solution of the recurrence with the initial values of the sum,
/// checked against the sum at further values and below `n0`
/// (`_solve_and_check`).
fn solve_and_check(c: &[Basic], rhs: &Basic, n: &Basic, n0: i64, direct_at: &dyn Fn(i64) -> R2) -> PolyResult<Checked> {
    let r: i64 = c.len() as i64 - 1;
    let mut general: Option<(Basic, Vec<Basic>)> = None;
    let mut n0: i64 = n0;
    // the products of the solution may vanish at the first values: the
    // start is moved on
    for start in n0..n0 + 4 {
        general = general_solution(c, rhs, n, start)?;
        if general.is_some() {
            n0 = start;
            break;
        }
    }
    let Some((mut solution, constants)) = general else {
        return Ok(None);
    };
    let value_at = |s: &Basic, v: i64| -> R2 {
        let at_v: Basic = subs(s, n, &int(v))?;
        within(|| doit(&at_v), STEP_SECONDS)
    };
    if !constants.is_empty() {
        let mut equations: Vec<Basic> = Vec::new();
        for v in n0..n0 + r + CHECKS {
            let value: Option<Basic> = direct_at(v)?;
            let at_v: Option<Basic> = value_at(&solution, v)?;
            let (Some(value), Some(at_v)) = (value, at_v.filter(finite)) else {
                continue;
            };
            equations.push(sub(&at_v, &value)?);
            if equations.len() == constants.len() {
                break;
            }
        }
        if equations.len() < constants.len() {
            return Ok(None);
        }
        let solve = || -> PolyResult<Vec<Basic>> {
            crate::solvers::linsolve::solve_linear_unique(&equations, &constants)?
                .ok_or_else(|| PolyError::NotImplemented("no unique solution".to_string()))
        };
        let solved: Option<Vec<Basic>> = crate::timeout::attempt_within(Some(STEP_SECONDS), solve)?;
        let Some(solved) = solved else {
            return Ok(None);
        };
        if solved.iter().any(|v| constants.iter().any(|c| v.has(c))) {
            return Ok(None);
        }
        let assignment: Vec<(Basic, Basic)> = constants.iter().cloned().zip(solved).collect();
        solution = subs_mapping(&solution, &assignment)?;
    }
    solution = simplest(&solution, n)?;
    let last: i64 = n0 + r + CHECKS;
    if has_sum(&solution) {
        // the unevaluated sums ordered from the first n possible on
        for start in 0..last {
            if let Some(lowered) = sums_ordered_from(&solution, n, start)? {
                if lowered != solution {
                    solution = simplest(&lowered, n)?;
                }
                break;
            }
        }
        if reversed_at(&solution, n, last)? || reversed_at(&solution, n, last + 100)? {
            return Ok(None);
        }
    }
    // the check: from n0 on the recurrence determines the values, below n0
    // the closed form may fail
    let mut exceptions: Vec<(i64, Basic)> = Vec::new();
    for v in 0..last {
        let expected: Option<Basic> = direct_at(v)?;
        if reversed_at(&solution, n, v)? {
            let Some(expected) = expected else {
                return Ok(None);
            };
            exceptions.push((v, expected));
            continue;
        }
        let at_v: Option<Basic> = value_at(&solution, v)?;
        let verdict: Option<bool> = match (&expected, at_v.filter(finite)) {
            (Some(e), Some(a)) => agree(e, &a)?,
            _ => None,
        };
        if v >= n0 {
            if verdict != Some(true) {
                return Ok(None);
            }
        } else if verdict != Some(true) {
            if let Some(expected) = expected {
                exceptions.push((v, expected));
            }
        }
    }
    Ok(Some((solution, exceptions)))
}
