//! Port of `sympy_extras/integrals/brackets.py`: Ramanujan's master
//! theorem and the method of brackets, Mellin transforms from power
//! series. The coefficients of one residue class of the power series are
//! a hypergeometric term guessed from the first coefficients and proved by
//! the differential equation it implies ([`formal_series_terms`]); the
//! master theorem continues them to `k = -s`, and the method of brackets
//! eliminates one index of a product of two series and sums the other.
//!
//! Python's binary operators evaluate at each step: the expressions are
//! built with the binary helpers in the same association.

#![allow(clippy::mutable_key_type)]

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::integrals::mellin::{monomial, MellinTransform};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    Basic::zero()
}
fn one() -> Basic {
    Basic::one()
}
fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn neg_oo() -> Basic {
    Basic::atom(Head::NegativeInfinity)
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn p2(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn ng(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn no(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(false)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn gamma(a: &Basic) -> R {
    crate::functions::apply_function(Function::Gamma, std::slice::from_ref(a))
}
fn factorial(a: &Basic) -> R {
    crate::functions::apply_function(Function::Factorial, std::slice::from_ref(a))
}
fn abs_(a: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(a)
}
fn gammasimp(e: &Basic) -> R {
    crate::simplify::gammasimp::gammasimp(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}
fn is_integer_node(e: &Basic) -> bool {
    e.as_rational().is_some_and(|q| q.is_int())
}
fn int_value(e: &Basic) -> Option<i64> {
    e.as_rational()
        .filter(|q| q.is_int())
        .and_then(|q| i64::try_from(crate::numbers::numer(q)).ok())
}
fn is_pole_error(e: &PolyError) -> bool {
    matches!(e, PolyError::Pole(_))
}
fn not_ported(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if m.starts_with(crate::error::NOT_PORTED_PREFIX))
}
fn real_value(e: &Basic) -> f64 {
    crate::core::evalf::numeric_value(e, 64)
        .map(|v| crate::core::evalf::parts(&v).0.to_f64().value())
        .unwrap_or(f64::NAN)
}
fn series(e: &Basic, x: &Basic, n: i64) -> R {
    crate::core::series::series(e, Some(x), &zero(), n, "+", None, &zero())
}
fn leadterm(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::series::leadterm(e, x, None, &zero())
}
fn remove_o(e: &Basic) -> R {
    crate::core::order::remove_o(e)
}
/// `e.coeff(x, n)`.
fn coeff(e: &Basic, x: &Basic, n: i64) -> R {
    crate::core::expr::coeff(e, &p2(x, &int(n))?)
}

/// The power series of a function about `0` in the form
/// `f(x) = x^r sum_{j >= 0} c(j) (x^m)^j` (`SeriesCoefficient`).
#[derive(Debug, Clone)]
pub struct SeriesCoefficient {
    pub coefficient: Basic,
    pub index: Basic,
    pub period: i64,
    pub offset: Basic,
}

impl SeriesCoefficient {
    pub fn repr(&self) -> String {
        format!(
            "SeriesCoefficient({}, {}, period={}, offset={})",
            self.coefficient, self.index, self.period, self.offset
        )
    }

    /// `phi(j) = (-1)^j j! c(j)` of the master theorem, the gamma function
    /// in place of the factorials; `None` when a factor `(-1)^j` remains
    /// (`phi`).
    pub fn phi(&self) -> PolyResult<Option<Basic>> {
        let j: &Basic = &self.index;
        let sign: Basic = p2(&int(-1), j)?;
        let product: Basic = m2(&m2(&sign, &factorial(j)?)?, &self.coefficient)?;
        let measure = |e: &Basic| crate::core::function::count_ops(e);
        let value: Basic = crate::simplify::powsimp::powsimp(
            &product,
            false,
            crate::simplify::powsimp::Combine::Exp,
            false,
            &measure,
        )?;
        let value: Basic = gammasimp(&crate::core::rewriting::rewrite(
            &value,
            &[],
            "gamma",
            &crate::core::rewriting::RewriteHints::default(),
        )?)?;
        if value.has(&sign) || has_sign_power(&value, j) {
            return Ok(None);
        }
        Ok(Some(value))
    }
}

/// The power series of a function (`FormalSeriesTerms`).
#[derive(Debug, Clone)]
pub struct FormalSeriesTerms {
    pub formula: Basic,
    pub index: Basic,
    pub start: Basic,
    pub independent: Basic,
}

impl FormalSeriesTerms {
    pub fn repr(&self) -> String {
        format!(
            "FormalSeriesTerms({}, {}, start={}, independent={})",
            self.formula, self.index, self.start, self.independent
        )
    }
}

/// Whether `e` still has a power of a negative number with the index in
/// the exponent (`_has_sign_power`).
fn has_sign_power(e: &Basic, j: &Basic) -> bool {
    e.preorder()
        .iter()
        .any(|n| n.is_pow() && yes(&n.args()[0], Predicate::negative) && n.args()[1].has(j))
}

/// A `Piecewise` coefficient read as `(value, m, r)` (`_residue_class`).
fn residue_class(formula: &Basic, k: &Basic) -> Option<(Basic, i64, i64)> {
    if !matches!(formula.head(), Head::Piecewise) {
        return Some((formula.clone(), 1, 0));
    }
    let mut found: Option<(Basic, i64, i64)> = None;
    for pair in formula.args() {
        let (value, condition) = (&pair.args()[0], &pair.args()[1]);
        if *value == zero() {
            continue;
        }
        if matches!(condition.head(), Head::BooleanTrue) && found.is_none() {
            return Some((value.clone(), 1, 0));
        }
        if !matches!(condition.head(), Head::Relational(RelOp::Eq)) {
            return None;
        }
        let (lhs, rhs) = (&condition.args()[0], &condition.args()[1]);
        if !(lhs.is_function(Function::Mod) && lhs.args()[0] == *k) {
            return None;
        }
        let (modulus, remainder) = (int_value(&lhs.args()[1])?, int_value(rhs)?);
        if !is_integer_node(&lhs.args()[1]) || !is_integer_node(rhs) {
            return None;
        }
        if found.is_some() {
            return None;
        }
        found = Some((value.clone(), modulus, remainder));
    }
    found
}

/// `e.subs(old, new)`, `None` when the substitution fails (`_replaced`).
fn replaced(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Option<Basic>> {
    match subs(e, old, new) {
        Ok(v) => Ok(Some(v)),
        Err(PolyError::Value(_) | PolyError::Type(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

const PROBE_TERMS: i64 = 12;
const CLASS_TERMS: i64 = 12;
const MAX_TERMS: i64 = 80;

/// The first `count` Taylor coefficients of `f` about 0 by the native
/// series (`_taylor_coefficients`).
fn taylor_coefficients(f: &Basic, x: &Basic, count: i64) -> PolyResult<Option<Vec<Basic>>> {
    let truncated = match attempt(|| remove_o(&series(f, x, count)?)) {
        Ok(v) => v,
        Err(e) if is_pole_error(&e) || matches!(e, PolyError::Value(_) | PolyError::Type(_)) => return Ok(None),
        Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some(truncated) = truncated.filter(|t| is_polynomial(t, x)) else {
        return Ok(None);
    };
    let expanded: Basic = crate::core::function::expand_method(&truncated)?;
    if !is_polynomial(&expanded, x) {
        return Ok(None);
    }
    let mut coefficients: Vec<Basic> = Vec::new();
    for n in 0..count {
        let c: Basic = if n != 0 {
            coeff(&expanded, x, n)?
        } else {
            subs(&expanded, x, &zero())?
        };
        if c.has(x) {
            return Ok(None);
        }
        coefficients.push(c);
    }
    Ok(Some(coefficients))
}

/// Whether a constant is zero, `None` when undecided (`_is_zero_constant`).
fn is_zero_constant(c: &Basic) -> PolyResult<Option<bool>> {
    if *c == zero() {
        return Ok(Some(true));
    }
    if let Some(z) = is(c, Predicate::zero) {
        return Ok(Some(z));
    }
    Ok(crate::assumptions::analysis::certified_sign(c)?.map(|s| s == 0))
}

/// The exponents of the nonzero coefficients (`_support`).
fn support(coefficients: &[Basic]) -> PolyResult<Option<Vec<i64>>> {
    let mut found: Vec<i64> = Vec::new();
    for (n, c) in coefficients.iter().enumerate() {
        match is_zero_constant(c)? {
            None => return Ok(None),
            Some(false) => found.push(n as i64),
            Some(true) => {}
        }
    }
    Ok(Some(found))
}

/// The polynomial of a list of integers, highest degree first
/// (`_polynomial_of`).
fn polynomial_of(coefficients: &[Int], t: &Basic) -> R {
    let mut value: Basic = zero();
    for c in coefficients {
        value = a2(&m2(&value, t)?, &Basic::integer(c.clone()))?;
    }
    Ok(value)
}

/// `polynomial(J) g` for `J = (x d/dx - s)/m` (`_apply_in_theta`).
fn apply_in_theta(polynomial: &Basic, t: &Basic, g: &Basic, x: &Basic, s: i64, m: i64) -> R {
    let poly = crate::polys::polytools::Poly::new(polynomial, std::slice::from_ref(t))?;
    let mut coefficients: Vec<Basic> = poly.all_coeffs()?;
    coefficients.reverse();
    let mut result: Basic = zero();
    let mut power: Basic = g.clone();
    for coefficient in coefficients {
        if coefficient != zero() {
            result = a2(&result, &m2(&coefficient, &power)?)?;
        }
        let derivative: Basic = crate::core::function::diff(&power, &[(x.clone(), 1)])?;
        power = d2(&s2(&m2(x, &derivative)?, &m2(&int(s), &power)?)?, &int(m))?;
    }
    Ok(result)
}

/// Whether the expression is recognised as zero (`_vanishes`).
fn vanishes(e: &Basic) -> PolyResult<bool> {
    let expanded = attempt(|| crate::polys::polytools::cancel(&crate::core::function::expand_method(e)?, &[]))?;
    if expanded == Some(zero()) {
        return Ok(true);
    }
    Ok(attempt(|| simplify(e))? == Some(zero()))
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// The power series of `f` about 0 when its nonzero coefficients form one
/// residue class and are a hypergeometric term there
/// (`formal_series_terms`).
pub fn formal_series_terms(f: &Basic, x: &Basic) -> PolyResult<Option<FormalSeriesTerms>> {
    let Some(probe) = taylor_coefficients(f, x, PROBE_TERMS)? else {
        return Ok(None);
    };
    let Some(sup) = support(&probe)? else {
        return Ok(None);
    };
    if sup.len() < 2 {
        return Ok(None);
    }
    let s: i64 = sup[0];
    let mut m: i64 = 0;
    for n in &sup[1..] {
        m = gcd(m, n - s);
    }
    let count: i64 = s + m * CLASS_TERMS + m;
    if count > MAX_TERMS {
        return Ok(None);
    }
    let Some(coefficients) = taylor_coefficients(f, x, count)? else {
        return Ok(None);
    };
    let Some(sup) = support(&coefficients)? else {
        return Ok(None);
    };
    if sup.iter().any(|n| (n - s) % m != 0) || sup[0] != s {
        return Ok(None);
    }
    let class: Vec<i64> = (s..count).filter(|n| (n - s) % m == 0).collect();
    if class != sup {
        // a zero inside the class: not a hypergeometric term
        return Ok(None);
    }
    let e0: Basic = coefficients[s as usize].clone();
    let mut ratios: Vec<Rat> = Vec::new();
    for n in &sup {
        let q: Basic = crate::polys::polytools::cancel(&d2(&coefficients[*n as usize], &e0)?, &[])?;
        let Some(r) = q.as_rational() else {
            return Ok(None);
        };
        ratios.push(r.clone());
    }
    let Some(recurrence) =
        crate::concrete::guess::guess_recurrence(&ratios, Some(1), None, crate::concrete::guess::MARGIN)?
    else {
        return Ok(None);
    };
    if recurrence.order() != 1 {
        return Ok(None);
    }
    let i: Basic = Dummy::new("i", &[]).0;
    let big_p: Basic = ng(&polynomial_of(&recurrence.coefficients[0], &i)?)?;
    let big_q: Basic = polynomial_of(&recurrence.coefficients[1], &i)?;
    let roots = |e: &Basic| -> PolyResult<crate::polys::polyroots::RootsResult> {
        let poly = crate::polys::polytools::Poly::new(e, std::slice::from_ref(&i))?;
        crate::polys::polyroots::roots(&poly.as_expr()?, std::slice::from_ref(&i), true, true, None)
    };
    let (p_roots, q_roots) = (roots(&big_p)?, roots(&big_q)?);
    let p_poly = crate::polys::polytools::Poly::new(&big_p, std::slice::from_ref(&i))?;
    let q_poly = crate::polys::polytools::Poly::new(&big_q, std::slice::from_ref(&i))?;
    if p_roots.values().sum::<usize>() as i64 != p_poly.degree(0)?
        || q_roots.values().sum::<usize>() as i64 != q_poly.degree(0)?
    {
        return Ok(None);
    }
    for root in q_roots.keys() {
        if let Some(r) = root.as_rational() {
            if r.is_int() && *r >= Rat::from(Int::ZERO) {
                return Ok(None);
            }
            let n_free: Basic = a2(&m2(&int(m), &a2(root, &one())?)?, &int(s))?;
            if let Some(v) = int_value(&n_free) {
                if v >= count {
                    return Ok(None);
                }
            }
        }
    }
    // the differential equation of the class, checked on f
    let t: Basic = Dummy::new("t", &[]).0;
    let shifted_q: Basic = subs(&big_q, &i, &s2(&t, &one())?)?;
    let shifted_p: Basic = subs(&big_p, &i, &t)?;
    let left: Basic = apply_in_theta(&shifted_q, &t, &s2(f, &m2(&e0, &p2(x, &int(s))?)?)?, x, s, m)?;
    let right: Basic = m2(&p2(x, &int(m))?, &apply_in_theta(&shifted_p, &t, f, x, s, m)?)?;
    if !vanishes(&s2(&left, &right)?)? {
        return Ok(None);
    }
    let k: Basic = Dummy::new("k", &[(Predicate::integer, true), (Predicate::nonnegative, true)]).0;
    let j: Basic = d2(&s2(&k, &int(s))?, &int(m))?;
    let mut value: Basic = m2(&e0, &p2(&d2(&p_poly.LC()?, &q_poly.LC()?)?, &j)?)?;
    let rf = |a: &Basic| crate::functions::combinatorial::factorials::rf(a, &j);
    for (root, multiplicity) in p_roots.iter() {
        value = m2(&value, &p2(&rf(&ng(root)?)?, &int(*multiplicity as i64))?)?;
    }
    for (root, multiplicity) in q_roots.iter() {
        value = d2(&value, &p2(&rf(&ng(root)?)?, &int(*multiplicity as i64))?)?;
    }
    let mut formula: Basic = value.clone();
    if m > 1 {
        // SymPy's Eq(Mod(k, m), s % m), unevaluated
        let condition: Basic = crate::core::relational::rel(
            RelOp::Eq,
            &Basic::new(Head::Function(Function::Mod), vec![k.clone(), int(m)]),
            &int(s.rem_euclid(m)),
        );
        formula = crate::functions::elementary::piecewise::piecewise(&[
            (value, condition),
            (zero(), crate::core::logic::true_()),
        ])?;
    }
    Ok(Some(FormalSeriesTerms {
        formula,
        index: k,
        start: zero(),
        independent: zero(),
    }))
}

/// The power series of `f` about 0 as a [`SeriesCoefficient`]
/// (`taylor_coefficient`).
pub fn taylor_coefficient(f: &Basic, x: &Basic) -> PolyResult<Option<SeriesCoefficient>> {
    if is_polynomial(f, x) {
        return Ok(None);
    }
    let Some(series_terms) = formal_series_terms(f, x)? else {
        return Ok(None);
    };
    let k: &Basic = &series_terms.index;
    let Some((value, m, r)) = residue_class(&series_terms.formula, k) else {
        return Ok(None);
    };
    // the independent terms must agree with the formula
    for term in crate::core::expr::terms_of(&series_terms.independent) {
        if term == zero() {
            continue;
        }
        let (coefficient, rest) = crate::core::expr::as_independent(&term, x, false)?;
        let power: Basic = match monomial(&rest, x)? {
            None => {
                if rest == one() {
                    zero()
                } else {
                    return Ok(None);
                }
            }
            Some((_, p)) => p,
        };
        let Some(pv) = int_value(&power) else {
            return Ok(None);
        };
        if (pv - r).rem_euclid(m) != 0 {
            return Ok(None);
        }
        let Some(at_power) = replaced(&value, k, &power)? else {
            return Ok(None);
        };
        if simplify(&s2(&at_power, &coefficient)?)? != zero() {
            return Ok(None);
        }
    }
    // and the whole must agree with the Taylor expansion for the first terms
    let Some(first) = int_value(&series_terms.start) else {
        return Ok(None);
    };
    let terms: i64 = 6.max(first + 2 * m);
    let truncated = match attempt(|| remove_o(&series(f, x, terms)?)) {
        Ok(v) => v,
        Err(e) if is_pole_error(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some(truncated) = truncated else {
        return Ok(None);
    };
    for k0 in 0..terms {
        let actual: Basic = if k0 != 0 {
            coeff(&truncated, x, k0)?
        } else {
            subs(&truncated, x, &zero())?
        };
        let predicted: Basic = if (k0 - r).rem_euclid(m) != 0 {
            zero()
        } else {
            let Some(at_k0) = replaced(&value, k, &int(k0))? else {
                return Ok(None);
            };
            if no(&at_k0, Predicate::finite) {
                zero()
            } else {
                at_k0
            }
        };
        if simplify(&s2(&predicted, &actual)?)? != zero() {
            return Ok(None);
        }
    }
    let j: Basic = Dummy::new("j", &[(Predicate::integer, true), (Predicate::nonnegative, true)]).0;
    let Some(coefficient) = replaced(&value, k, &a2(&m2(&int(m), &j)?, &int(r))?)? else {
        return Ok(None);
    };
    Ok(Some(SeriesCoefficient {
        coefficient,
        index: j,
        period: m,
        offset: int(r),
    }))
}

/// Whether `f` decays faster than the powers met in practice
/// (`_decays_fast`).
fn decays_fast(f: &Basic, x: &Basic) -> PolyResult<bool> {
    let decays = attempt(|| crate::core::limits::limit(&m2(f, &p2(x, &int(50))?)?, x, &oo(), "+"))?;
    Ok(decays == Some(zero()))
}

/// `p - q` for `f = C x**q T(a x**p)` with `T` a sine or a cosine
/// (`_oscillation_order`).
fn oscillation_order(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let mut q: Basic = zero();
    let mut oscillating: Option<Basic> = None;
    for factor in mul_args(f) {
        if !factor.has(x) {
            continue;
        }
        if factor.is_function(Function::Sin) || factor.is_function(Function::Cos) {
            if oscillating.is_some() {
                return Ok(None);
            }
            oscillating = Some(factor.args()[0].clone());
            continue;
        }
        let Some((c, e)) = monomial(&factor, x)? else {
            return Ok(None);
        };
        if c != one() {
            return Ok(None);
        }
        q = a2(&q, &e)?;
    }
    let Some(oscillating) = oscillating else {
        return Ok(None);
    };
    let Some((a, p)) = monomial(&oscillating, x)? else {
        return Ok(None);
    };
    let positive_rational: bool = p.as_rational().is_some_and(|r| *r > Rat::from(Int::ZERO));
    if !positive_rational || a.has(x) || !yes(&a, Predicate::extended_real) || is(&a, Predicate::zero) != Some(false) {
        return Ok(None);
    }
    Ok(Some(s2(&p, &q)?))
}

fn fast_or_oscillation(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    if decays_fast(f, x)? {
        return Ok(Some(oo()));
    }
    oscillation_order(f, x)
}

/// `e` such that `f ~ C x**(-e)` as `x -> oo` (`_order_at_infinity`).
fn order_at_infinity(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let y: Basic = Dummy::new("y", &[(Predicate::positive, true)]).0;
    let g: Basic = subs(f, x, &d2(&one(), &y)?)?;
    let leading = match attempt(|| remove_o(&series(&g, &y, 1)?)) {
        Ok(v) => v,
        // an essential singularity at infinity: exponential decay or oscillation
        Err(e) if is_pole_error(&e) => return fast_or_oscillation(f, x),
        Err(e) => return Err(e),
    };
    let leading: Basic = match leading {
        Some(l) if l != zero() => l,
        _ => {
            // the leading term is beyond the first order
            let found = match attempt(|| leadterm(&g, &y)) {
                Ok(v) => v,
                Err(e)
                    if is_pole_error(&e)
                        || matches!(e, PolyError::Value(_))
                        || (matches!(e, PolyError::NotImplemented(_)) && !not_ported(&e)) =>
                {
                    None
                }
                Err(e) => return Err(e),
            };
            if let Some((_, ex)) = found {
                if !ex.has(&y) {
                    return Ok(Some(ex));
                }
            }
            return fast_or_oscillation(f, x);
        }
    };
    if matches!(leading.head(), Head::Order) {
        return fast_or_oscillation(f, x);
    }
    let mut exponents: Vec<Basic> = Vec::new();
    for term in crate::core::expr::terms_of(&leading) {
        if term.has(x) && monomial(&term, x)?.is_some() {
            return Ok(None);
        }
        let rest: Basic = crate::core::expr::as_independent(&term, &y, false)?.1;
        // powers of log(y) do not change the exponent of convergence
        let kept: Vec<Basic> = mul_args(&rest)
            .into_iter()
            .filter(|a| {
                let log_y = |b: &Basic| b.is_function(Function::Log) && b.args()[0] == y;
                !(log_y(a) || (a.is_pow() && log_y(&a.args()[0])))
            })
            .collect();
        let rest: Basic = crate::core::function::doit(&crate::core::mul::mul(&kept)?)?;
        if !rest.has(&y) {
            exponents.push(zero());
            continue;
        }
        let Some((_, e)) = monomial(&rest, &y)? else {
            return Ok(None);
        };
        exponents.push(e);
    }
    if exponents.is_empty() {
        return Ok(None);
    }
    let mut best: Basic = exponents[0].clone();
    for e in &exponents[1..] {
        if real_value(e) < real_value(&best) {
            best = e.clone();
        }
    }
    Ok(Some(best))
}

/// The exponent of the first term of the series of `f` at 0
/// (`_order_at_zero`).
fn order_at_zero(f: &Basic, x: &Basic, offset: &Basic) -> PolyResult<Basic> {
    let found = match attempt(|| leadterm(f, x)) {
        Ok(v) => v,
        Err(e)
            if is_pole_error(&e)
                || matches!(e, PolyError::Value(_))
                || (matches!(e, PolyError::NotImplemented(_)) && !not_ported(&e)) =>
        {
            return Ok(offset.clone())
        }
        Err(e) => return Err(e),
    };
    let Some((_, e)) = found else {
        return Ok(offset.clone());
    };
    if e.has(x) || !is_integer_node(&e) {
        return Ok(offset.clone());
    }
    Ok(if yes(&s2(&e, offset)?, Predicate::positive) {
        e
    } else {
        offset.clone()
    })
}

/// The strip of convergence (`_strip`).
fn strip(f: &Basic, x: &Basic, offset: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some(upper) = order_at_infinity(f, x)? else {
        return Ok(None);
    };
    if upper.has(x) {
        return Ok(None);
    }
    Ok(Some((ng(&order_at_zero(f, x, offset)?)?, upper)))
}

/// The Mellin transform of `f` by Ramanujan's master theorem, with the
/// strip in which the integral converges (`ramanujan_master_theorem`).
pub fn ramanujan_master_theorem(f: &Basic, x: &Basic, s: &Basic) -> PolyResult<Option<MellinTransform>> {
    let Some(c) = taylor_coefficient(f, x)? else {
        return Ok(None);
    };
    let Some(phi) = c.phi()? else {
        return Ok(None);
    };
    // f = x^r g(x^m): M[f](s) = M[g]((s + r)/m) / m
    let sigma: Basic = d2(&a2(s, &c.offset)?, &int(c.period))?;
    let value: Basic = gammasimp(&d2(
        &m2(&gamma(&sigma)?, &subs(&phi, &c.index, &ng(&sigma)?)?)?,
        &int(c.period),
    )?)?;
    let Some((lower, upper)) = strip(f, x, &c.offset)? else {
        return Ok(None);
    };
    if upper != oo() && lower != neg_oo() && no(&s2(&upper, &lower)?, Predicate::positive) {
        return Ok(None);
    }
    Ok(Some(MellinTransform {
        transform: value,
        strip: (lower, upper),
        condition: crate::core::logic::true_(),
    }))
}

/// `(g(y), beta, gamma)`: a factor `g(beta x**gamma)` of a product.
type Factor = (Basic, Basic, Basic);

fn is_function_node(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Function(_) | Head::Undefined(_) | Head::Hyper | Head::MeijerG | Head::Piecewise
    )
}

/// `f` as `constant * x**alpha * prod g_i(beta_i x**gamma_i)`
/// (`_factors`).
fn factors(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Vec<Factor>)>> {
    let mut constant: Basic = one();
    let mut alpha: Basic = zero();
    let mut found: Vec<Factor> = Vec::new();
    for e in mul_args(f) {
        if !e.has(x) {
            constant = m2(&constant, &e)?;
            continue;
        }
        if let Some((c, p)) = monomial(&e, x)? {
            constant = m2(&constant, &c)?;
            alpha = a2(&alpha, &p)?;
            continue;
        }
        let argument: Option<Basic> = if is_function_node(&e) && !e.args().is_empty() {
            let last: &Basic = &e.args()[e.args().len() - 1];
            if crate::core::logic::is_boolean(last) || matches!(last.head(), Head::Tuple | Head::ExprCondPair) {
                None
            } else {
                Some(last.clone())
            }
        } else if e.is_pow() && e.args()[0].is_add() {
            Some(e.args()[0].clone())
        } else {
            None
        };
        let Some(argument) = argument else {
            return Ok(None);
        };
        let rest: Basic = crate::core::expr::as_independent(&argument, x, true)?.1;
        let Some((mut beta, g_)) = monomial(&rest, x)? else {
            return Ok(None);
        };
        // a positive scale, so that g is read on (0, oo)
        if yes(&beta, Predicate::negative) {
            beta = ng(&beta)?;
        }
        let y: Basic = Dummy::new("y", &[(Predicate::positive, true)]).0;
        let inner: Basic = p2(&d2(&y, &beta)?, &d2(&one(), &g_)?)?;
        found.push((subs(&e, x, &inner)?, beta, g_));
    }
    Ok(Some((constant, alpha, found)))
}

/// The transform of `g(beta x**gamma)` from that of `g` (`_scaled`).
fn scaled(transform: &MellinTransform, beta: &Basic, g_: &Basic, s: &Basic) -> PolyResult<MellinTransform> {
    let value: Basic = d2(
        &m2(
            &subs(&transform.transform, s, &d2(s, g_)?)?,
            &p2(beta, &d2(&ng(s)?, g_)?)?,
        )?,
        &abs_(g_)?,
    )?;
    let (lower, upper) = &transform.strip;
    let strip: (Basic, Basic) = if yes(g_, Predicate::positive) {
        (m2(lower, g_)?, m2(upper, g_)?)
    } else {
        (m2(upper, g_)?, m2(lower, g_)?)
    };
    Ok(MellinTransform {
        transform: value,
        strip,
        condition: transform.condition.clone(),
    })
}

fn shifted(transform: &MellinTransform, alpha: &Basic, constant: &Basic, s: &Basic) -> PolyResult<MellinTransform> {
    let value: Basic = m2(constant, &subs(&transform.transform, s, &a2(s, alpha)?)?)?;
    let (lower, upper) = &transform.strip;
    Ok(MellinTransform {
        transform: value,
        strip: (s2(lower, alpha)?, s2(upper, alpha)?),
        condition: transform.condition.clone(),
    })
}

/// The Mellin transform of a product of two series-expandable factors by
/// the method of brackets (`method_of_brackets`).
pub fn method_of_brackets(
    f: &Basic,
    x: &Basic,
    s: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<MellinTransform>> {
    let Some((constant, alpha, fs)) = factors(f, x)? else {
        return Ok(None);
    };
    if fs.len() != 2 {
        return Ok(None);
    }
    let mut series_: Vec<(SeriesCoefficient, Basic, Basic)> = Vec::new();
    let mut strips: Vec<(Basic, Basic)> = Vec::new();
    for (g, beta, g_) in &fs {
        let Some(variable) = variable(g) else {
            return Ok(None);
        };
        let Some(c) = taylor_coefficient(g, &variable)? else {
            return Ok(None);
        };
        if c.phi()?.is_none() {
            return Ok(None);
        }
        let Some(single) = ramanujan_master_theorem(g, &variable, s)? else {
            return Ok(None);
        };
        let sc: MellinTransform = scaled(&single, beta, g_, s)?;
        series_.push((c, beta.clone(), g_.clone()));
        strips.push(sc.strip);
    }
    let lower: Basic = max_of(&strips[0].0, &strips[1].0)?;
    let upper: Basic = min_of(&strips[0].1, &strips[1].1)?;
    if lower != neg_oo() && upper != oo() && no(&s2(&upper, &lower)?, Predicate::positive) {
        return Ok(None);
    }
    let (c1, beta1, g1) = series_[0].clone();
    let (c2, beta2, g2) = series_[1].clone();
    let (Some(phi1), Some(phi2)) = (c1.phi()?, c2.phi()?) else {
        return Ok(None);
    };
    for first in [0, 1] {
        let (ca, cb) = if first == 0 { (&c1, &c2) } else { (&c2, &c1) };
        let (phia, phib) = if first == 0 { (&phi1, &phi2) } else { (&phi2, &phi1) };
        let (betaa, betab) = if first == 0 { (&beta1, &beta2) } else { (&beta2, &beta1) };
        let (ga, gb) = if first == 0 { (&g1, &g2) } else { (&g2, &g1) };
        let n: &Basic = &ca.index;
        // the exponent of x: ga (ma n + ra) + gb (mb k + rb) + alpha; eliminate k
        let big_a: Basic = m2(ga, &int(ca.period))?;
        let big_b: Basic = m2(gb, &int(cb.period))?;
        let big_c: Basic = a2(&a2(&m2(ga, &ca.offset)?, &m2(gb, &cb.offset)?)?, &alpha)?;
        let k_star: Basic = d2(&ng(&a2(&a2(&m2(&big_a, n)?, &big_c)?, s)?)?, &big_b)?;
        let mut term: Basic = m2(phia, &p2(&int(-1), n)?)?;
        term = d2(&term, &factorial(n)?)?;
        term = m2(&term, &p2(betaa, &a2(&m2(&int(ca.period), n)?, &ca.offset)?)?)?;
        term = m2(&term, &subs(phib, &cb.index, &k_star)?)?;
        term = m2(&term, &p2(betab, &a2(&m2(&int(cb.period), &k_star)?, &cb.offset)?)?)?;
        term = m2(&term, &gamma(&ng(&k_star)?)?)?;
        term = d2(&term, &abs_(&big_b)?)?;
        let term: Basic = gammasimp(&term)?;
        let limit = (n.clone(), zero(), oo());
        let Some(mut total) = attempt(|| crate::concrete::summations::summation(&term, &limit))? else {
            continue;
        };
        if matches!(total.head(), Head::Piecewise) {
            // the closed form where the series converges, continued
            let Some(closed) = closed_branch(&total) else {
                continue;
            };
            total = closed;
        }
        if total.preorder().iter().any(|n| matches!(n.head(), Head::Sum)) {
            continue;
        }
        if total.preorder().iter().any(|n| matches!(n.head(), Head::Hyper)) && !converges(&total, assumptions)? {
            continue;
        }
        let summed: Basic = total.clone();
        let expanded = attempt(|| {
            crate::simplify::hyperexpand::hyperexpand(
                &summed,
                false,
                crate::simplify::hyperexpand::Rewrite::Default,
                None,
            )
        })?;
        let value: Basic = gammasimp(&m2(&constant, &expanded.unwrap_or(total))?)?;
        if let Some(decay) = order_at_infinity(f, x)? {
            if !decay.has(x) {
                // the true strip of the product: from its order at 0 and its decay
                let low: Basic = ng(&a2(&a2(&m2(&g1, &c1.offset)?, &m2(&g2, &c2.offset)?)?, &alpha)?)?;
                let up: Basic = s2(&decay, &alpha)?;
                return Ok(Some(MellinTransform {
                    transform: value,
                    strip: (low, up),
                    condition: crate::core::logic::true_(),
                }));
            }
        }
        return Ok(Some(MellinTransform {
            transform: value,
            strip: (s2(&lower, &alpha)?, s2(&upper, &alpha)?),
            condition: crate::core::logic::true_(),
        }));
    }
    Ok(None)
}

/// The first branch of a `Piecewise` without the unevaluated sum
/// (`_closed_branch`).
fn closed_branch(total: &Basic) -> Option<Basic> {
    for pair in total.args() {
        let e: &Basic = &pair.args()[0];
        if !e.preorder().iter().any(|n| matches!(n.head(), Head::Sum)) {
            return Some(e.clone());
        }
    }
    None
}

fn variable(g: &Basic) -> Option<Basic> {
    let symbols: Vec<Basic> = g
        .free_symbols()
        .into_iter()
        .filter(|v| matches!(v.head(), Head::Dummy(..)))
        .collect();
    if symbols.len() == 1 {
        symbols.into_iter().next()
    } else {
        None
    }
}

/// Whether every hypergeometric series in `e` converges (`_converges`).
fn converges(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    for node in e.preorder() {
        if !matches!(node.head(), Head::Hyper) {
            continue;
        }
        let (p, q) = (node.args()[0].args().len(), node.args()[1].args().len());
        if p <= q {
            continue;
        }
        if p == q + 1 {
            let query: Basic = crate::core::relational::lt(&abs_(&node.args()[2])?, &one())?;
            if crate::assumptions::ask::ask(&query, assumptions, None)? == Some(true) {
                continue;
            }
        }
        return Ok(false);
    }
    Ok(true)
}

fn max_of(a: &Basic, b: &Basic) -> R {
    if *a == neg_oo() {
        return Ok(b.clone());
    }
    if *b == neg_oo() {
        return Ok(a.clone());
    }
    let d: Basic = s2(a, b)?;
    if yes(&d, Predicate::nonnegative) {
        return Ok(a.clone());
    }
    if yes(&d, Predicate::negative) {
        return Ok(b.clone());
    }
    crate::functions::elementary::miscellaneous::max_(&[a.clone(), b.clone()])
}

fn min_of(a: &Basic, b: &Basic) -> R {
    if *a == oo() {
        return Ok(b.clone());
    }
    if *b == oo() {
        return Ok(a.clone());
    }
    let d: Basic = s2(a, b)?;
    if yes(&d, Predicate::nonpositive) {
        return Ok(a.clone());
    }
    if yes(&d, Predicate::positive) {
        return Ok(b.clone());
    }
    crate::functions::elementary::miscellaneous::min_(&[a.clone(), b.clone()])
}

/// The Mellin transform of `f` from its power series: the master theorem
/// for one factor, the method of brackets for two
/// (`mellin_transform_series`).
pub fn mellin_transform_series(
    f: &Basic,
    x: &Basic,
    s: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<MellinTransform>> {
    let Some((constant, alpha, fs)) = factors(f, x)? else {
        return Ok(None);
    };
    if fs.len() == 1 {
        let (g, beta, g_) = &fs[0];
        let Some(variable) = variable(g) else {
            return Ok(None);
        };
        let Some(single) = ramanujan_master_theorem(g, &variable, s)? else {
            return Ok(None);
        };
        return Ok(Some(shifted(&scaled(&single, beta, g_, s)?, &alpha, &constant, s)?));
    }
    if fs.len() == 2 {
        return method_of_brackets(f, x, s, assumptions);
    }
    Ok(None)
}
