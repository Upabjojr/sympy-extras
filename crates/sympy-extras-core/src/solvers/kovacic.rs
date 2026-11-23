//! Port of `sympy_extras/solvers/kovacic.py`: Kovacic's algorithm for the
//! Liouvillian solutions of `z'' = r z` with `r` rational, and the
//! solutions of second order linear homogeneous equations with rational
//! coefficients it gives.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, HashMap};

use crate::core::basic::{Basic, Head, RelOp};
use crate::error::{PolyError, PolyResult, NOT_PORTED_PREFIX};

type R = PolyResult<Basic>;

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
fn sqrt(e: &Basic) -> R {
    pow(e, &Basic::half())
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
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
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn rat(p: i64, q: i64) -> R {
    Basic::rational(crate::numbers::Int::from(p), crate::numbers::Int::from(q))
}
fn integer_of(e: &Basic) -> Option<i64> {
    if !matches!(e.head(), Head::Rational { value, .. } if value.is_int()) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
}
fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b) && !crate::sets::sets::is_set(b) && !matches!(b.head(), Head::Tuple)
}
/// What the polynomial routines raise on coefficients they cannot take
/// (`_FAILURES`).
pub(crate) fn failure(e: &PolyError) -> bool {
    match e {
        PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_) | PolyError::ZeroDivision(_) => true,
        PolyError::NotImplemented(m) => !m.starts_with(NOT_PORTED_PREFIX),
        _ => false,
    }
}

/// A Liouvillian solution `exp(Integral(omega))` of `z'' = r z`
/// (`KovacicSolution`).
#[derive(Clone, Debug)]
pub struct KovacicSolution {
    pub case: u8,
    pub omega: Option<Basic>,
    pub omega_polynomial: Basic,
    pub solution: Basic,
}

/// `e` with the keys of `rule` replaced, evaluated (`_replaced`).
pub(crate) fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    let value: Basic = crate::core::function::xreplace_evaluated(e, &map)?;
    if !is_expr(&value) {
        return Err(PolyError::Type(format!("{} is not an expression", value)));
    }
    Ok(value)
}

fn sum(terms: &[Basic]) -> R {
    let mut total: Basic = Basic::zero();
    for t in terms {
        total = add(&[total, t.clone()])?;
    }
    Ok(total)
}

fn product(factors: &[Basic]) -> R {
    let mut total: Basic = Basic::one();
    for f in factors {
        total = mul(&[total, f.clone()])?;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// polynomials in one variable with any coefficients

/// `k` for `e = x**k` (`0` for `1`), `None` otherwise
/// (`_monomial_degree`).
fn monomial_degree(e: &Basic, x: &Basic) -> Option<i64> {
    if *e == Basic::one() {
        return Some(0);
    }
    if e == x {
        return Some(1);
    }
    if e.is_pow() && e.args()[0] == *x {
        return integer_of(&e.args()[1]);
    }
    None
}

/// The nonzero coefficients of the expanded `e` as a polynomial in `x`;
/// `ValueError` when `e` is no polynomial in `x` (`_coefficients`).
pub(crate) fn coefficients(e: &Basic, x: &Basic) -> PolyResult<BTreeMap<i64, Basic>> {
    let mut collected: Vec<(i64, Vec<Basic>)> = Vec::new();
    for term in terms_of(&expand(e)?) {
        if term == Basic::zero() {
            continue;
        }
        let (coefficient, rest) = crate::core::expr::as_independent(&term, x, false)?;
        let Some(k) = monomial_degree(&rest, x).filter(|k| *k >= 0) else {
            return Err(PolyError::Value(format!("{} is not a polynomial in {}", e, x)));
        };
        match collected.iter_mut().find(|(j, _)| *j == k) {
            Some((_, parts)) => parts.push(coefficient),
            None => collected.push((k, vec![coefficient])),
        }
    }
    let mut result: BTreeMap<i64, Basic> = BTreeMap::new();
    for (k, parts) in collected {
        let c: Basic = sum(&parts)?;
        if c != Basic::zero() {
            result.insert(k, c);
        }
    }
    Ok(result)
}

/// A polynomial in `x` given by its coefficients (`_Polynomial`).
#[derive(Clone, Debug)]
struct Polynomial {
    expr: Basic,
    coefficients: BTreeMap<i64, Basic>,
}

impl Polynomial {
    fn new(e: &Basic, x: &Basic) -> PolyResult<Polynomial> {
        Ok(Polynomial {
            expr: e.clone(),
            coefficients: coefficients(e, x)?,
        })
    }
    fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }
    fn degree(&self) -> i64 {
        self.coefficients.keys().next_back().copied().unwrap_or(-1)
    }
    fn lc(&self) -> Basic {
        self.coefficients
            .values()
            .next_back()
            .cloned()
            .unwrap_or_else(Basic::zero)
    }
    fn all_coeffs(&self) -> Vec<Basic> {
        (0..=self.degree())
            .rev()
            .map(|k| self.coefficients.get(&k).cloned().unwrap_or_else(Basic::zero))
            .collect()
    }
}

/// `e` with the numbers which are not rational and the functions of the
/// parameters in its coefficients replaced by symbols, and the
/// replacements back (`_generic_coefficients`).
pub(crate) fn generic_coefficients(e: &Basic, x: &Basic) -> PolyResult<(Basic, Vec<(Basic, Basic)>)> {
    let mut forward: Vec<(Basic, Basic)> = Vec::new();
    fn walk(node: &Basic, x: &Basic, forward: &mut Vec<(Basic, Basic)>) -> R {
        if node.is_symbol() || matches!(node.head(), Head::Rational { .. }) {
            return Ok(node.clone());
        }
        if node.is_add() {
            let parts: Vec<Basic> = node
                .args()
                .iter()
                .filter(|a| is_expr(a))
                .map(|a| walk(a, x, forward))
                .collect::<PolyResult<_>>()?;
            return sum(&parts);
        }
        if node.is_mul() {
            let parts: Vec<Basic> = node
                .args()
                .iter()
                .filter(|a| is_expr(a))
                .map(|a| walk(a, x, forward))
                .collect::<PolyResult<_>>()?;
            return product(&parts);
        }
        if node.is_pow() && integer_of(&node.args()[1]).is_some_and(|k| k >= 0) {
            return pow(&walk(&node.args()[0], x, forward)?, &node.args()[1]);
        }
        if node.has(x) {
            return Ok(node.clone()); // no polynomial in x: Poly refuses it
        }
        if let Some((_, d)) = forward.iter().find(|(n, _)| n == node) {
            return Ok(d.clone());
        }
        let d: Basic = dummy(&format!("g{}", forward.len()));
        forward.push((node.clone(), d.clone()));
        Ok(d)
    }
    let walked: Basic = walk(&expand(e)?, x, &mut forward)?;
    let back: Vec<(Basic, Basic)> = forward.into_iter().map(|(n, d)| (d, n)).collect();
    Ok((walked, back))
}

fn native_roots(e: &Basic, x: &Basic) -> PolyResult<Vec<(Basic, usize)>> {
    let poly = crate::polys::polytools::Poly::new(e, std::slice::from_ref(x))?;
    let found = crate::polys::polyroots::roots(&poly.as_expr()?, std::slice::from_ref(x), true, true, None)?;
    Ok(found.iter().map(|(r, m)| (r.clone(), *m)).collect())
}

/// The roots of a polynomial in `x` which SymPy's `roots` finds, with their
/// multiplicities; coefficients with irrational numbers or functions of the
/// parameters taken as generators of the domain (`_roots`).
pub(crate) fn roots(e: &Basic, x: &Basic) -> PolyResult<Vec<(Basic, usize)>> {
    match native_roots(e, x) {
        Ok(found) => return Ok(found),
        Err(PolyError::Polys(..)) => {}
        Err(err) => return Err(err),
    }
    let (generic, back) = generic_coefficients(e, x)?;
    let found = native_roots(&generic, x)?;
    let mut result: Vec<(Basic, usize)> = Vec::new();
    for (r, m) in found {
        let value: Basic = replaced(&r, &back)?;
        match result.iter_mut().find(|(v, _)| *v == value) {
            Some((_, n)) => *n += m,
            None => result.push((value, m)),
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------

/// Numerator and denominator polynomials of a rational function
/// (`_rational`).
fn rational(e: &Basic, x: &Basic) -> PolyResult<(Polynomial, Polynomial)> {
    let (num, den) = crate::core::expr::as_numer_denom(&cancel(e)?)?;
    match (Polynomial::new(&num, x), Polynomial::new(&den, x)) {
        (Ok(n), Ok(d)) => Ok((n, d)),
        (Err(PolyError::Value(_)), _) | (_, Err(PolyError::Value(_))) => {
            Err(PolyError::Value(format!("{} is not a rational function of {}", e, x)))
        }
        (Err(err), _) | (_, Err(err)) => Err(err),
    }
}

/// `r` with `z'' = r z` the normal form of `y'' + p y' + q y = 0`
/// (`normal_form`).
pub fn normal_form(p: &Basic, q: &Basic, x: &Basic) -> R {
    cancel(&add(&[
        div(&pow(p, &int(2))?, &int(4))?,
        div(&diff(p, x, 1)?, &int(2))?,
        neg(q)?,
    ])?)
}

type Poles = Vec<(Basic, usize)>;

/// The poles with their orders, or `None` (`_poles`).
fn poles_of(den: &Polynomial, x: &Basic) -> PolyResult<Option<Poles>> {
    let found = roots(&den.expr, x)?;
    if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != den.degree() {
        return Ok(None);
    }
    Ok(Some(found))
}

/// The order of `r` at infinity (`_order_at_infinity`).
fn order_at_infinity(num: &Polynomial, den: &Polynomial) -> i64 {
    den.degree() - num.degree()
}

/// The coefficient of `(x - c)**-2` at a pole of order 2 (`_coefficient`).
fn coefficient(r: &Basic, c: &Basic, x: &Basic) -> R {
    let value: Basic = crate::core::subs::subs(&cancel(&mul(&[r.clone(), pow(&sub(x, c)?, &int(2))?])?)?, x, c)?;
    if !is_expr(&value) {
        return Err(PolyError::Type(format!("{} is not an expression", value)));
    }
    Ok(value)
}

/// The coefficient of `x**-2` at infinity when the order there is 2
/// (`_coefficient_at_infinity`).
fn coefficient_at_infinity(num: &Polynomial, den: &Polynomial) -> R {
    div(&num.lc(), &den.lc())
}

/// `e` simplified when it may be a number (`_as_number`).
fn as_number(e: &Basic) -> R {
    let expanded: Basic = expand(e)?;
    if !expanded.free_symbols().is_empty() {
        return Ok(expanded);
    }
    simplify(&expanded)
}

fn integers(candidates: &[Basic]) -> PolyResult<Vec<i64>> {
    let mut result: Vec<i64> = Vec::new();
    for c in candidates {
        if let Some(v) = integer_of(&as_number(c)?) {
            if !result.contains(&v) {
                result.push(v);
            }
        }
    }
    Ok(result)
}

/// The exponent of `t` in a term `c t**k` of a Laurent polynomial
/// (`_t_degree`).
fn t_degree(term: &Basic, t: &Basic) -> PolyResult<i64> {
    let (_, rest) = crate::core::expr::as_independent(term, t, false)?;
    if rest == Basic::one() {
        return Ok(0);
    }
    if rest == *t {
        return Ok(1);
    }
    if rest.is_pow() && rest.args()[0] == *t {
        if let Some(k) = integer_of(&rest.args()[1]) {
            return Ok(k);
        }
    }
    Err(PolyError::Value(format!(
        "{} is not a term of a Laurent polynomial in {}",
        term, t
    )))
}

/// The coefficient of `t**n` (SymPy's `coeff(t, n)`).
fn coeff_power(e: &Basic, t: &Basic, n: i64) -> R {
    if n == 0 {
        return Ok(crate::core::expr::as_independent(e, t, true)?.0);
    }
    crate::core::expr::coeff(e, &pow(t, &int(n))?)
}

fn series_without_order(e: &Basic, t: &Basic, n: i64) -> R {
    let s: Basic = crate::core::series::series(e, Some(t), &Basic::zero(), n, "+", None, &Basic::zero())?;
    crate::core::order::remove_o(&s)
}

// ---------------------------------------------------------------------------
// case 1

fn case1(
    r: &Basic,
    num: &Polynomial,
    den: &Polynomial,
    poles: &Poles,
    x: &Basic,
) -> PolyResult<Option<KovacicSolution>> {
    if let Some(result) = case1_kovacic(r, num, den, poles, x)? {
        return Ok(Some(result));
    }
    case1_riccati(r, x)
}

/// The polar part of the Laurent expansion of `sqrt(r)` at a pole of even
/// order and the ratio of the alpha data (`_laurent_sqrt`).
fn laurent_sqrt(r: &Basic, c: &Basic, order: usize, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let nu: i64 = (order / 2) as i64;
    let t: Basic = dummy("t");
    let shifted: Basic = replaced(r, &[(x.clone(), add(&[c.clone(), t.clone()])?)])?;
    if attempt(|| series_without_order(&shifted, &t, 1))?.is_none() {
        return Ok(None);
    }
    let Some(root) = attempt(|| series_without_order(&sqrt(&shifted)?, &t, 0))? else {
        return Ok(None);
    };
    let mut polar_terms: Vec<Basic> = Vec::new();
    for term in terms_of(&root) {
        if t_degree(&term, &t)? + nu < nu {
            polar_terms.push(term);
        }
    }
    let polar: Basic = sum(&polar_terms)?;
    if polar == Basic::zero() {
        return Ok(None);
    }
    let a: Basic = coeff_power(&polar, &t, -nu)?;
    let rest: Basic = expand(&sub(&shifted, &pow(&polar, &int(2))?)?)?;
    let b: Basic = coeff_power(&rest, &t, -nu - 1)?;
    Ok(Some((replaced(&polar, &[(t.clone(), sub(x, c)?)])?, div(&b, &a)?)))
}

/// The sign choices of the local data (`cartesian(*([(1, -1)]*n))`).
fn sign_choices(n: usize) -> Vec<Vec<i64>> {
    let mut result: Vec<Vec<i64>> = vec![Vec::new()];
    for _ in 0..n {
        let mut next: Vec<Vec<i64>> = Vec::new();
        for choice in &result {
            for s in [1, -1] {
                let mut c: Vec<i64> = choice.clone();
                c.push(s);
                next.push(c);
            }
        }
        result = next;
    }
    result
}

fn case1_kovacic(
    r: &Basic,
    num: &Polynomial,
    den: &Polynomial,
    poles: &Poles,
    x: &Basic,
) -> PolyResult<Option<KovacicSolution>> {
    // local data at the poles: (c, [sqrt r]_c, alpha+, alpha-)
    let mut local: Vec<(Basic, Basic, Basic, Basic)> = Vec::new();
    for (c, order) in poles {
        if *order == 1 {
            local.push((c.clone(), Basic::zero(), Basic::one(), Basic::one()));
        } else if *order == 2 {
            let b: Basic = coefficient(r, c, x)?;
            let root: Basic = sqrt(&add(&[int(1), mul(&[int(4), b])?])?)?;
            local.push((
                c.clone(),
                Basic::zero(),
                div(&add(&[int(1), root.clone()])?, &int(2))?,
                div(&sub(&int(1), &root)?, &int(2))?,
            ));
        } else if order % 2 == 0 {
            let Some((polar, ratio)) = laurent_sqrt(r, c, *order, x)? else {
                return Ok(None);
            };
            let nu: Basic = int((order / 2) as i64);
            local.push((
                c.clone(),
                polar,
                div(&add(&[ratio.clone(), nu.clone()])?, &int(2))?,
                div(&add(&[neg(&ratio)?, nu])?, &int(2))?,
            ));
        } else {
            return Ok(None);
        }
    }
    // at infinity
    let o: i64 = order_at_infinity(num, den);
    let infinity: (Basic, Basic, Basic) = if o > 2 {
        (Basic::zero(), Basic::zero(), Basic::one())
    } else if o == 2 {
        let b: Basic = coefficient_at_infinity(num, den)?;
        let root: Basic = sqrt(&add(&[int(1), mul(&[int(4), b])?])?)?;
        (
            Basic::zero(),
            div(&add(&[int(1), root.clone()])?, &int(2))?,
            div(&sub(&int(1), &root)?, &int(2))?,
        )
    } else if o % 2 == 0 {
        let nu: i64 = -o / 2;
        let t: Basic = dummy("t");
        let at_infinity: Basic = replaced(r, &[(x.clone(), pow(&t, &int(-1))?)])?;
        let Some(found) = attempt(|| series_without_order(&sqrt(&at_infinity)?, &t, 1))? else {
            return Ok(None);
        };
        let mut part_terms: Vec<Basic> = Vec::new();
        for term in terms_of(&found) {
            if t_degree(&term, &t)? + nu <= nu {
                part_terms.push(term);
            }
        }
        let polynomial_part: Basic = sum(&part_terms)?;
        if polynomial_part == Basic::zero() {
            return Ok(None);
        }
        let a: Basic = coeff_power(&polynomial_part, &t, -nu)?;
        let rest: Basic = expand(&sub(&at_infinity, &pow(&polynomial_part, &int(2))?)?)?;
        let b: Basic = coeff_power(&rest, &t, -nu + 1)?;
        let ratio: Basic = div(&b, &a)?;
        (
            replaced(&polynomial_part, &[(t.clone(), pow(x, &int(-1))?)])?,
            div(&sub(&ratio, &int(nu))?, &int(2))?,
            div(&sub(&neg(&ratio)?, &int(nu))?, &int(2))?,
        )
    } else {
        return Ok(None);
    };
    for choice in sign_choices(local.len() + 1) {
        let s_inf: i64 = *choice.last().expect("one sign at infinity");
        let alpha_inf: &Basic = if s_inf == 1 { &infinity.1 } else { &infinity.2 };
        let mut alphas: Vec<Basic> = Vec::new();
        for (entry, s) in local.iter().zip(&choice) {
            alphas.push(if *s == 1 { entry.2.clone() } else { entry.3.clone() });
        }
        let d: Basic = as_number(&sub(alpha_inf, &sum(&alphas)?)?)?;
        let Some(d) = integer_of(&d).filter(|d| *d >= 0) else {
            continue;
        };
        let mut parts: Vec<Basic> = vec![mul(&[int(s_inf), infinity.0.clone()])?];
        for (entry, s) in local.iter().zip(&choice) {
            let alpha: &Basic = if *s == 1 { &entry.2 } else { &entry.3 };
            parts.push(mul(&[int(*s), entry.1.clone()])?);
            parts.push(div(alpha, &sub(x, &entry.0)?)?);
        }
        let omega: Basic = cancel(&sum(&parts)?)?;
        let (p, unknowns) = monic_polynomial(d, x)?;
        let equation: Basic = add(&[
            diff(&p, x, 2)?,
            mul(&[int(2), omega.clone(), diff(&p, x, 1)?])?,
            mul(&[add(&[diff(&omega, x, 1)?, pow(&omega, &int(2))?, neg(r)?])?, p.clone()])?,
        ])?;
        let Some(values) = solve_linear(&equation, &unknowns, x)? else {
            continue;
        };
        let p_value: Basic = replaced(&p, &values)?;
        let full: Basic = cancel(&add(&[omega, div(&diff(&p_value, x, 1)?, &p_value)?])?)?;
        if cancel(&add(&[diff(&full, x, 1)?, pow(&full, &int(2))?, neg(r)?])?)? != Basic::zero() {
            continue;
        }
        let w: Basic = symbol("w");
        return Ok(Some(KovacicSolution {
            case: 1,
            omega: Some(full.clone()),
            omega_polynomial: sub(&w, &full)?,
            solution: exp_integral(&full, x)?,
        }));
    }
    Ok(None)
}

/// The rational solutions `omega` of `omega' + omega**2 = r` given by the
/// hyperexponential solutions of `z'' = r z` (`rational_riccati`).
pub fn rational_riccati(r: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    use crate::solvers::linear_ode::{hyperexponential_solutions, log_derivative, LinearOperator};
    let found = (|| -> PolyResult<Vec<Basic>> {
        let operator: LinearOperator = LinearOperator::from_equation(&equation_of(r, x)?, &unknown(x)?)?;
        hyperexponential_solutions(&operator, 200)
    })();
    let found: Vec<Basic> = match found {
        Ok(found) => found,
        Err(e) if failure(&e) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut solutions: Vec<Basic> = Vec::new();
    for z in found {
        let omega: Basic = log_derivative(&z, x)?;
        let check = (|| -> PolyResult<bool> {
            Ok(cancel(&add(&[diff(&omega, x, 1)?, pow(&omega, &int(2))?, neg(r)?])?)? != Basic::zero())
        })();
        match check {
            Ok(true) => continue,
            Ok(false) => {}
            Err(e) if failure(&e) => continue,
            Err(e) => return Err(e),
        }
        if !solutions.contains(&omega) {
            solutions.push(omega);
        }
    }
    Ok(solutions)
}

fn unknown(x: &Basic) -> R {
    std::sync::Arc::new(crate::core::basic::UndefinedFunction::new("_z", None, &[])).apply(vec![x.clone()])
}

fn equation_of(r: &Basic, x: &Basic) -> R {
    let z: Basic = unknown(x)?;
    sub(&diff(&z, x, 2)?, &mul(&[r.clone(), z])?)
}

fn case1_riccati(r: &Basic, x: &Basic) -> PolyResult<Option<KovacicSolution>> {
    let Some(solutions) = attempt(|| rational_riccati(r, x))? else {
        return Ok(None);
    };
    for solution in solutions {
        let omega: Basic = cancel(&solution)?;
        if omega.free_symbols().iter().any(|s| s != x && s.is_symbol()) {
            continue;
        }
        if cancel(&add(&[diff(&omega, x, 1)?, pow(&omega, &int(2))?, neg(r)?])?)? != Basic::zero() {
            continue;
        }
        let w: Basic = symbol("w");
        return Ok(Some(KovacicSolution {
            case: 1,
            omega: Some(omega.clone()),
            omega_polynomial: sub(&w, &omega)?,
            solution: exp_integral(&omega, x)?,
        }));
    }
    Ok(None)
}

fn exp_integral(omega: &Basic, x: &Basic) -> R {
    match crate::integrals::indefinite::closed_antiderivative(omega, x, false, true, None)? {
        Some(integral) => exp(&integral),
        None => exp(&crate::integrals::integral::integral(omega, x, None)?),
    }
}

// ---------------------------------------------------------------------------
// cases 2 and 3

type Sets = (Vec<(Basic, Vec<i64>)>, Vec<i64>);

/// The sets `E_c` of the poles and `E_oo` for case 2 (`n = 2`) or case 3
/// (`n = 4, 6, 12`) (`_sets`).
fn sets(r: &Basic, num: &Polynomial, den: &Polynomial, poles: &Poles, x: &Basic, n: i64) -> PolyResult<Option<Sets>> {
    let family = |root: &Basic| -> PolyResult<Vec<i64>> {
        if n == 2 {
            let candidates: Vec<Basic> = [0i64, 2, -2]
                .iter()
                .map(|k| add(&[int(2), mul(&[int(*k), root.clone()])?]))
                .collect::<PolyResult<_>>()?;
            integers(&candidates)
        } else {
            let mut candidates: Vec<Basic> = Vec::new();
            for k in -n / 2..=n / 2 {
                candidates.push(add(&[int(6), mul(&[rat(12, n)?, int(k), root.clone()])?])?);
            }
            integers(&candidates)
        }
    };
    let mut e: Vec<(Basic, Vec<i64>)> = Vec::new();
    for (c, order) in poles {
        let values: Vec<i64> = if *order == 1 {
            vec![if n == 2 { 4 } else { 12 }]
        } else if *order == 2 {
            let b: Basic = coefficient(r, c, x)?;
            family(&sqrt(&add(&[int(1), mul(&[int(4), b])?])?)?)?
        } else {
            if n != 2 {
                return Ok(None);
            }
            vec![*order as i64]
        };
        if values.is_empty() {
            return Ok(None);
        }
        e.push((c.clone(), values));
    }
    let o: i64 = order_at_infinity(num, den);
    let e_inf: Vec<i64> = if o > 2 {
        if n == 2 {
            vec![0, 2, 4]
        } else {
            vec![12]
        }
    } else if o == 2 {
        let b: Basic = coefficient_at_infinity(num, den)?;
        family(&sqrt(&add(&[int(1), mul(&[int(4), b])?])?)?)?
    } else {
        if n != 2 {
            return Ok(None);
        }
        vec![o]
    };
    if e_inf.is_empty() {
        return Ok(None);
    }
    Ok(Some((e, e_inf)))
}

fn monic_polynomial(d: i64, x: &Basic) -> PolyResult<(Basic, Vec<Basic>)> {
    let coefficients: Vec<Basic> = (0..d).map(|i| dummy(&format!("p{}", i))).collect();
    let mut terms: Vec<Basic> = vec![pow(x, &int(d))?];
    for (i, c) in coefficients.iter().enumerate() {
        terms.push(mul(&[c.clone(), pow(x, &int(i as i64))?])?);
    }
    Ok((sum(&terms)?, coefficients))
}

/// Coefficients making a polynomial in `x` with coefficients linear in the
/// unknowns vanish (`_solve_linear`).
fn solve_linear(equation: &Basic, unknowns: &[Basic], x: &Basic) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    let num: Basic = crate::core::expr::as_numer_denom(&cancel(equation)?)?.0;
    let conditions: Vec<Basic> = match Polynomial::new(&num, x) {
        Ok(p) => p.all_coeffs().into_iter().filter(|c| *c != Basic::zero()).collect(),
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if conditions.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if unknowns.is_empty() {
        return Ok(None);
    }
    let zeros: Vec<(Basic, Basic)> = unknowns.iter().map(|u| (u.clone(), Basic::zero())).collect();
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    let mut rhs: Vec<Basic> = Vec::new();
    for condition in &conditions {
        let row: Vec<Basic> = unknowns
            .iter()
            .map(|u| diff(condition, u, 1))
            .collect::<PolyResult<_>>()?;
        if row.iter().any(|entry| unknowns.iter().any(|u| entry.has(u))) {
            return Ok(None); // not linear in the unknowns
        }
        rows.push(row);
        rhs.push(neg(&replaced(condition, &zeros)?)?);
    }
    let Some(solution) = crate::solvers::recurr::solve_linear(&rows, &rhs, unknowns.len())? else {
        return Ok(None);
    };
    Ok(Some(unknowns.iter().cloned().zip(solution).collect()))
}

fn family_products(sets: &[(Basic, Vec<i64>)]) -> Vec<Vec<i64>> {
    let mut result: Vec<Vec<i64>> = vec![Vec::new()];
    for (_, values) in sets {
        let mut next: Vec<Vec<i64>> = Vec::new();
        for f in &result {
            for v in values {
                let mut g: Vec<i64> = f.clone();
                g.push(*v);
                next.push(g);
            }
        }
        result = next;
    }
    result
}

fn case2(
    r: &Basic,
    num: &Polynomial,
    den: &Polynomial,
    poles: &Poles,
    x: &Basic,
) -> PolyResult<Option<KovacicSolution>> {
    let Some((e, e_inf)) = sets(r, num, den, poles, x, 2)? else {
        return Ok(None);
    };
    for e_value in &e_inf {
        for family in family_products(&e) {
            let d2: i64 = e_value - family.iter().sum::<i64>();
            if d2 < 0 || d2 % 2 != 0 {
                continue;
            }
            let d: i64 = d2 / 2;
            let mut theta_terms: Vec<Basic> = Vec::new();
            for (value, (c, _)) in family.iter().zip(&e) {
                theta_terms.push(div(&rat(*value, 2)?, &sub(x, c)?)?);
            }
            let theta: Basic = sum(&theta_terms)?;
            let (p, unknowns) = monic_polynomial(d, x)?;
            let theta1: Basic = diff(&theta, x, 1)?;
            let equation: Basic = add(&[
                diff(&p, x, 3)?,
                mul(&[int(3), theta.clone(), diff(&p, x, 2)?])?,
                mul(&[
                    add(&[
                        mul(&[int(3), pow(&theta, &int(2))?])?,
                        mul(&[int(3), theta1.clone()])?,
                        mul(&[int(-4), r.clone()])?,
                    ])?,
                    diff(&p, x, 1)?,
                ])?,
                mul(&[
                    add(&[
                        diff(&theta, x, 2)?,
                        mul(&[int(3), theta.clone(), theta1])?,
                        pow(&theta, &int(3))?,
                        mul(&[int(-4), r.clone(), theta.clone()])?,
                        mul(&[int(-2), diff(r, x, 1)?])?,
                    ])?,
                    p.clone(),
                ])?,
            ])?;
            let Some(values) = solve_linear(&equation, &unknowns, x)? else {
                continue;
            };
            let p_value: Basic = replaced(&p, &values)?;
            let phi: Basic = cancel(&add(&[theta, div(&diff(&p_value, x, 1)?, &p_value)?])?)?;
            let w: Basic = symbol("w");
            let constant: Basic = cancel(&add(&[
                div(&diff(&phi, x, 1)?, &int(2))?,
                div(&pow(&phi, &int(2))?, &int(2))?,
                neg(r)?,
            ])?)?;
            let polynomial: Basic = add(&[pow(&w, &int(2))?, neg(&mul(&[phi.clone(), w.clone()])?)?, constant])?;
            let discriminant: Basic = cancel(&sub(
                &pow(&phi, &int(2))?,
                &mul(&[
                    int(4),
                    add(&[
                        div(&diff(&phi, x, 1)?, &int(2))?,
                        div(&pow(&phi, &int(2))?, &int(2))?,
                        neg(r)?,
                    ])?,
                ])?,
            )?)?;
            let omega: Basic = div(&add(&[phi, sqrt(&discriminant)?])?, &int(2))?;
            return Ok(Some(KovacicSolution {
                case: 2,
                omega: Some(omega.clone()),
                omega_polynomial: polynomial,
                solution: exp_integral(&omega, x)?,
            }));
        }
    }
    Ok(None)
}

fn case3(
    r: &Basic,
    num: &Polynomial,
    den: &Polynomial,
    poles: &Poles,
    x: &Basic,
) -> PolyResult<Option<KovacicSolution>> {
    if poles.iter().any(|(_, order)| *order > 2) || order_at_infinity(num, den) < 2 {
        return Ok(None);
    }
    for n in [4i64, 6, 12] {
        let Some((e, e_inf)) = sets(r, num, den, poles, x, n)? else {
            continue;
        };
        let linear: Vec<Basic> = e.iter().map(|(c, _)| sub(x, c)).collect::<PolyResult<_>>()?;
        let s_poly: Basic = product(&linear)?;
        for e_value in &e_inf {
            for family in family_products(&e) {
                let d12: i64 = n * (e_value - family.iter().sum::<i64>());
                if d12 < 0 || d12 % 12 != 0 {
                    continue;
                }
                let d: i64 = d12 / 12;
                let mut theta_terms: Vec<Basic> = Vec::new();
                for (value, (c, _)) in family.iter().zip(&e) {
                    theta_terms.push(div(&int(*value), &sub(x, c)?)?);
                }
                let theta: Basic = mul(&[rat(n, 12)?, sum(&theta_terms)?])?;
                let (p, unknowns) = monic_polynomial(d, x)?;
                // Kovacic's recursion, P_n = -P down to P_{-1} which must vanish
                let mut polynomials: BTreeMap<i64, Basic> = BTreeMap::new();
                polynomials.insert(n, neg(&p)?);
                let s1: Basic = diff(&s_poly, x, 1)?;
                for i in (0..=n).rev() {
                    let p_i: Basic = polynomials[&i].clone();
                    let p_next: Basic = polynomials.get(&(i + 1)).cloned().unwrap_or_else(Basic::zero);
                    let value: Basic = cancel(&add(&[
                        neg(&mul(&[s_poly.clone(), diff(&p_i, x, 1)?])?)?,
                        mul(&[
                            sub(
                                &mul(&[int(n - i), s1.clone()])?,
                                &mul(&[s_poly.clone(), theta.clone()])?,
                            )?,
                            p_i.clone(),
                        ])?,
                        neg(&mul(&[
                            int((n - i) * (i + 1)),
                            pow(&s_poly, &int(2))?,
                            r.clone(),
                            p_next,
                        ])?)?,
                    ])?)?;
                    polynomials.insert(i - 1, value);
                }
                let Some(values) = solve_linear(&polynomials[&-1], &unknowns, x)? else {
                    continue;
                };
                let w: Basic = symbol("w");
                let mut terms: Vec<Basic> = Vec::new();
                for i in 0..=n {
                    let factorial: Basic = crate::functions::combinatorial::factorials::factorial(&int(n - i))?;
                    terms.push(mul(&[
                        pow(&s_poly, &int(i))?,
                        div(&replaced(&polynomials[&i], &values)?, &factorial)?,
                        pow(&w, &int(i))?,
                    ])?);
                }
                let polynomial: Basic = cancel(&sum(&terms)?)?;
                let omega: Option<Basic> = algebraic_root(&polynomial, &w)?;
                let integrand: Basic = match &omega {
                    Some(o) => o.clone(),
                    None => root_placeholder(x)?,
                };
                let solution: Basic = exp(&crate::integrals::integral::integral(&integrand, x, None)?)?;
                return Ok(Some(KovacicSolution {
                    case: 3,
                    omega,
                    omega_polynomial: polynomial,
                    solution,
                }));
            }
        }
    }
    Ok(None)
}

/// `omega` when its polynomial cannot be solved in radicals
/// (`_root_placeholder`).
fn root_placeholder(x: &Basic) -> R {
    std::sync::Arc::new(crate::core::basic::UndefinedFunction::new("omega", None, &[])).apply(vec![x.clone()])
}

/// A root of a polynomial in `w` with rational function coefficients, in
/// radicals when the native `roots` finds them (`_algebraic_root`).
fn algebraic_root(polynomial: &Basic, w: &Basic) -> PolyResult<Option<Basic>> {
    let limit: Option<f64> = timeout().map(|t| t.min(5.0));
    let numerator: Basic = crate::core::expr::as_numer_denom(polynomial)?.0;
    let Some(found) = crate::timeout::attempt_within(limit, || roots(&numerator, w))? else {
        return Ok(None);
    };
    for (root, _) in found {
        if !root.preorder().iter().any(|n| matches!(n.head(), Head::Piecewise)) {
            return Ok(Some(root));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------

/// A Liouvillian solution of `z'' = r z` by Kovacic's algorithm, or `None`
/// (`liouvillian_solution`).
pub fn liouvillian_solution(r: &Basic, x: &Basic) -> PolyResult<Option<KovacicSolution>> {
    let (num, den) = rational(r, x)?;
    if num.is_zero() {
        return Ok(Some(KovacicSolution {
            case: 1,
            omega: Some(Basic::zero()),
            omega_polynomial: symbol("w"),
            solution: Basic::one(),
        }));
    }
    let Some(poles) = poles_of(&den, x)? else {
        return Ok(None);
    };
    let o: i64 = order_at_infinity(&num, &den);
    // the necessary conditions of Kovacic's theorem
    if poles.iter().all(|(_, k)| *k == 1 || k % 2 == 0) && (o % 2 == 0 || o > 2) {
        if let Some(result) = case1(r, &num, &den, &poles, x)? {
            return Ok(Some(result));
        }
    }
    if let Some(result) = case2(r, &num, &den, &poles, x)? {
        return Ok(Some(result));
    }
    case3(r, &num, &den, &poles, x)
}

/// Whether `e` does not depend on `x` (`_is_constant`).
fn is_constant(e: &Basic, x: &Basic) -> PolyResult<bool> {
    if !e.has(x) {
        return Ok(true);
    }
    Ok(attempt(|| simplify(&diff(e, x, 1)?))? == Some(Basic::zero()))
}

/// The second solution by reduction of order (`_second_solution`).
fn second_solution(y1: &Basic, p: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    use crate::integrals::indefinite::closed_antiderivative;
    let Some(inner) = closed_antiderivative(p, x, false, false, None)? else {
        return Ok(None);
    };
    let integrand: Basic = simplify(&div(&exp(&neg(&inner)?)?, &pow(y1, &int(2))?)?)?;
    if let Some(outer) = closed_antiderivative(&integrand, x, false, true, None)? {
        let candidate: Basic = simplify(&mul(&[y1.clone(), outer])?)?;
        let difference: Option<Basic> = attempt(|| simplify(&sub(&diff(&div(&candidate, y1)?, x, 1)?, &integrand)?))?;
        if difference == Some(Basic::zero()) {
            return Ok(Some(candidate));
        }
    }
    Ok(Some(mul(&[
        y1.clone(),
        crate::integrals::integral::integral(&integrand, x, None)?,
    ])?))
}

/// The Liouvillian solutions of a second order linear homogeneous equation
/// with rational coefficients (`dsolve_kovacic`).
pub fn dsolve_kovacic(equation: &Basic, f: &Basic, special_values: bool) -> PolyResult<Option<Vec<Basic>>> {
    let Some(found) = dsolve_kovacic_plain(equation, f)? else {
        return Ok(None);
    };
    if !special_values {
        return Ok(Some(found));
    }
    let solve_at = |at: &Basic| -> PolyResult<Option<Vec<Basic>>> { dsolve_kovacic(at, f, true) };
    Ok(Some(crate::solvers::ode_cases::basis_cases(
        &found,
        &crate::solvers::ode_cases::as_zero(equation)?,
        f,
        &solve_at,
    )?))
}

fn dsolve_kovacic_plain(equation: &Basic, f: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let x: Basic = match f.args().first() {
        Some(x) if x.is_symbol() => x.clone(),
        _ => return Err(PolyError::Value("the function must depend on a symbol".to_string())),
    };
    let lhs: Basic = if matches!(equation.head(), Head::Relational(RelOp::Eq)) {
        sub(&equation.args()[0], &equation.args()[1])?
    } else if is_expr(equation) {
        equation.clone()
    } else {
        return Err(PolyError::Type(format!("{} is not an equation", equation)));
    };
    let expanded: Basic = expand(&lhs)?;
    let (d2, d1, d0) = (diff(f, &x, 2)?, diff(f, &x, 1)?, f.clone());
    let a: Basic = crate::core::expr::coeff(&expanded, &d2)?;
    let rest: Basic = sub(&expanded, &mul(&[a.clone(), d2.clone()])?)?;
    let b: Basic = crate::core::expr::coeff(&rest, &d1)?;
    let rest: Basic = sub(&rest, &mul(&[b.clone(), d1.clone()])?)?;
    let c: Basic = crate::core::expr::coeff(&rest, &d0)?;
    if a == Basic::zero()
        || expand(&sub(&rest, &mul(&[c.clone(), d0.clone()])?)?)? != Basic::zero()
        || [&a, &b, &c].iter().any(|e| e.has(f))
    {
        return Ok(None);
    }
    for e in [&a, &b, &c] {
        if crate::core::expr::polynomial_like(e, std::slice::from_ref(&x), crate::core::expr::PolynomialKind::Rational)
            != Some(true)
        {
            return Ok(None);
        }
    }
    let p: Basic = cancel(&div(&b, &a)?)?;
    let q: Basic = cancel(&div(&c, &a)?)?;
    let r: Basic = normal_form(&p, &q, &x)?;
    let Some(z) = liouvillian_solution(&r, &x)? else {
        return Ok(None);
    };
    let Some(omega) = z.omega.clone() else {
        return Ok(None);
    };
    let Some(integral_p) = crate::integrals::indefinite::closed_antiderivative(&p, &x, true, false, None)? else {
        return Ok(None);
    };
    let factor: Basic = exp(&div(&neg(&integral_p)?, &int(2))?)?;
    let y1: Basic = normalized(&simplify(&mul(&[z.solution.clone(), factor.clone()])?)?, &x)?;
    let mut solutions: Vec<Basic> = vec![y1.clone()];
    if z.case == 3 {
        return Ok(Some(vec![mul(&[z.solution, factor])?]));
    }
    if z.case == 2 {
        // the conjugate root gives the second solution
        let w: Basic = symbol("w");
        let other: Basic = cancel(&sub(
            &neg(&crate::core::expr::coeff(&z.omega_polynomial, &w)?)?,
            &omega,
        )?)?;
        let y2: Basic = normalized(&simplify(&mul(&[exp_integral(&other, &x)?, factor.clone()])?)?, &x)?;
        if !is_constant(&simplify(&div(&y2, &y1)?)?, &x)? {
            solutions.push(y2);
        }
    }
    if solutions.len() == 1 {
        if let Some(second) = second_solution(&y1, &p, &x)? {
            solutions.push(normalized(&second, &x)?);
        }
    }
    Ok(Some(solutions))
}

/// `y` without its constant factor (`_normalized`).
fn normalized(y: &Basic, x: &Basic) -> R {
    let (constant, rest) = crate::core::expr::as_independent(y, x, false)?;
    if constant != Basic::zero() && rest != Basic::zero() && !constant.has(x) {
        return Ok(rest);
    }
    Ok(y.clone())
}
