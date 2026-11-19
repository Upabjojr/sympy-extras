//! Port of `sympy_extras/integrals/exponential.py`: antiderivatives of
//! exponentials of powers (`x**(v-1)*exp(a*x**n)`: `Ei`, recursions,
//! `erf`/`erfi`, upper incomplete gamma, `hyper`), of quadratics (shifted
//! to the first family), of rational functions of an exponential
//! (`ratint` in `u = exp(c*x)`, binomial radicals), of nested powers,
//! rational functions times an exponential (partial fractions and `Ei`),
//! composite Gaussians and `exp(alpha*x**2 + beta/x**2)`. Every result is
//! checked: `cancel`, `simplify`, then numerically. The expressions are
//! built in the order of the Python's operators.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::assumptions::ask::ask;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function};
use crate::core::expr::{as_independent, factors_of, is, terms_of};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;
type Coefficients = Vec<(i64, Basic)>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rat(p: i64, q: i64) -> Basic {
    Basic::from_rational(Rat::from(Int::from(p)) / Rat::from(Int::from(q)))
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn sqrt(a: &Basic) -> R {
    pw(a, &rat(1, 2))
}
fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn ei(a: &Basic) -> R {
    crate::functions::special::error_functions::integral_function(Function::Ei, std::slice::from_ref(a))
}
fn erf(a: &Basic) -> R {
    crate::functions::special::error_functions::erf(a)
}
fn erfi(a: &Basic) -> R {
    crate::functions::special::error_functions::erfi(a)
}
fn uppergamma(a: &Basic, z: &Basic) -> R {
    crate::functions::special::gamma_functions::uppergamma(a, z)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::derivative(e, x)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn is_exp(e: &Basic) -> bool {
    e.is_function(Function::Exp)
}
fn has_exp(e: &Basic) -> bool {
    e.preorder().iter().any(is_exp)
}
fn atoms_of(e: &Basic, test: impl Fn(&Basic) -> bool) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if test(&n) && !found.contains(&n) {
            found.push(n);
        }
    }
    found
}
fn integer_of(e: &Basic) -> Option<i64> {
    if e.is_integer_value() {
        i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
    } else {
        None
    }
}
fn budget() -> Option<f64> {
    crate::settings::settings().timeout.map(|t| t / 4.0)
}
fn powsimp_exp(e: &Basic) -> R {
    crate::simplify::powsimp::powsimp(
        e,
        false,
        crate::simplify::powsimp::Combine::Exp,
        false,
        &crate::core::function::count_ops,
    )
}
fn dummy(name: &str, positive: bool) -> Basic {
    let facts: &[(Predicate, bool)] = if positive { &[(Predicate::positive, true)] } else { &[] };
    crate::core::symbol::Dummy::new(name, facts).0
}

/// Whether `F' = f`: `cancel`, `simplify`, then numerically (`_checks`).
fn checks(big_f: &Basic, f: &Basic, x: &Basic, facts: &[Basic]) -> PolyResult<bool> {
    let difference: Basic = s2(&diff(big_f, x)?, f)?;
    let reduced: Option<Basic> = crate::timeout::attempt_within(budget(), || cancel(&difference))?;
    if reduced.as_ref().is_some_and(|r| *r == int(0)) {
        return Ok(true);
    }
    let target: Basic = reduced.unwrap_or_else(|| difference.clone());
    let simpler: Option<Basic> = crate::timeout::attempt_within(budget(), || {
        crate::simplify::simplify::simplify(&target, &crate::simplify::simplify::Options::default())
    })?;
    if simpler.as_ref().is_some_and(|r| *r == int(0)) {
        return Ok(true);
    }
    crate::integrals::conditions::numerically_equal(&diff(big_f, x)?, f, facts, 3)
}

fn is_power_of(e: &Basic, x: &Basic) -> bool {
    e.is_pow() && e.args()[0] == *x && !e.args()[1].has(x)
}

fn coefficients(e: &Basic, x: &Basic) -> PolyResult<Option<Coefficients>> {
    let mut found: Coefficients = Vec::new();
    for term in terms_of(&expand(e)?) {
        let (c, monomial) = as_independent(&term, x, false)?;
        let k: i64 = if monomial == int(1) {
            0
        } else if monomial == *x {
            1
        } else if monomial.is_pow()
            && monomial.args()[0] == *x
            && integer_of(&monomial.args()[1]).is_some_and(|k| k > 0)
        {
            integer_of(&monomial.args()[1]).expect("an integer")
        } else {
            return Ok(None);
        };
        match found.iter_mut().find(|(n, _)| *n == k) {
            Some(entry) => entry.1 = a2(&entry.1, &c)?,
            None => found.push((k, a2(&int(0), &c)?)),
        }
    }
    found.retain(|(_, c)| *c != int(0));
    Ok(Some(found))
}

fn degree(c: &Coefficients) -> i64 {
    c.iter().map(|(k, _)| *k).max().unwrap_or(-1)
}
fn get(c: &Coefficients, k: i64) -> Basic {
    c.iter()
        .find(|(n, _)| *n == k)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| int(0))
}

fn from_coefficients(c: &Coefficients, x: &Basic) -> R {
    let mut total: Basic = int(0);
    for (k, v) in c {
        total = a2(&total, &m2(v, &pw(x, &int(*k))?)?)?;
    }
    Ok(total)
}

fn divmod(p: &Coefficients, q: &Coefficients) -> PolyResult<(Coefficients, Coefficients)> {
    let mut remainder: Coefficients = p.clone();
    let mut quotient: Coefficients = Vec::new();
    let n: i64 = degree(q);
    let leading: Basic = get(q, n);
    while !remainder.is_empty() && degree(&remainder) >= n {
        let m: i64 = degree(&remainder);
        let c: Basic = cancel(&d2(&get(&remainder, m), &leading)?)?;
        quotient.push((m - n, c.clone()));
        for (k, qk) in q {
            let index: i64 = k + m - n;
            let value: Basic = cancel(&s2(&get(&remainder, index), &m2(&c, qk)?)?)?;
            if value == int(0) {
                remainder.retain(|(j, _)| *j != index);
            } else {
                match remainder.iter_mut().find(|(j, _)| *j == index) {
                    Some(entry) => entry.1 = value,
                    None => remainder.push((index, value)),
                }
            }
        }
        remainder.retain(|(j, _)| *j != m);
    }
    Ok((quotient, remainder))
}

/// The coefficients made symbols for `roots` (`_Generalizer`).
struct Generalizer {
    dummies: Vec<(Basic, Basic)>,
}

impl Generalizer {
    fn call(&mut self, e: &Basic) -> R {
        if e.is_add() {
            let mut total: Basic = int(0);
            for a in e.args() {
                total = a2(&total, &self.call(a)?)?;
            }
            return Ok(total);
        }
        if e.is_mul() {
            let mut product: Basic = int(1);
            for a in e.args() {
                product = m2(&product, &self.call(a)?)?;
            }
            return Ok(product);
        }
        if e.is_pow() && e.args()[1].is_integer_value() {
            return pw(&self.call(&e.args()[0])?, &e.args()[1]);
        }
        if e.is_symbol() || e.is_rational_class() {
            return Ok(e.clone());
        }
        if let Some((_, d)) = self.dummies.iter().find(|(o, _)| o == e) {
            return Ok(d.clone());
        }
        let d: Basic = dummy("g", false);
        self.dummies.push((e.clone(), d.clone()));
        Ok(d)
    }

    fn back(&self, e: &Basic) -> R {
        let rule: HashMap<Basic, Basic> = self.dummies.iter().map(|(o, d)| (d.clone(), o.clone())).collect();
        crate::core::function::xreplace_evaluated(e, &rule)
    }
}

fn roots(c: &Coefficients, x: &Basic) -> PolyResult<Option<Vec<(Basic, i64)>>> {
    let mut general = Generalizer { dummies: Vec::new() };
    let mut generalized: Coefficients = Vec::new();
    for (k, v) in c {
        generalized.push((*k, general.call(v)?));
    }
    let polynomial: Basic = from_coefficients(&generalized, x)?;
    let found = match crate::polys::polyroots::roots(&polynomial, std::slice::from_ref(x), true, true, None) {
        Ok(f) => f,
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let total: i64 = found.iter().map(|(_, m)| *m as i64).sum();
    if total != degree(c) {
        return Ok(None);
    }
    let mut restored: Vec<(Basic, i64)> = Vec::new();
    for (root, m) in found.iter() {
        let r: Basic = general.back(root)?;
        match restored.iter_mut().find(|(k, _)| *k == r) {
            Some(entry) => entry.1 += *m as i64,
            None => restored.push((r, *m as i64)),
        }
    }
    Ok(Some(restored))
}

/// The antiderivative of `x**(v-1)*exp(a*x**n)` (`_power_exponential_raw`).
fn power_exponential_raw(v: &Basic, a: &Basic, n: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let s: Basic = cancel(&d2(v, n)?)?;
    let axn: Basic = m2(a, &pw(x, n)?)?;
    if *v == int(0) {
        return Ok(Some(d2(&ei(&axn)?, n)?));
    }
    let an: Basic = m2(a, n)?;
    if let Some(si) = integer_of(&s) {
        if si > 0 {
            if si == 1 {
                return Ok(Some(d2(&exp(&axn)?, &an)?));
            }
            let vn: Basic = s2(v, n)?;
            let Some(lower) = power_exponential_raw(&vn, a, n, x, assumptions)? else {
                return Ok(None);
            };
            // x**(v - n) * exp(a * x**n) / (a * n) - (v - n) * lower / (a * n)
            let first: Basic = d2(&m2(&pw(x, &vn)?, &exp(&axn)?)?, &an)?;
            let second: Basic = d2(&m2(&vn, &lower)?, &an)?;
            return Ok(Some(s2(&first, &second)?));
        }
        if si < 0 {
            let Some(higher) = power_exponential_raw(&a2(v, n)?, a, n, x, assumptions)? else {
                return Ok(None);
            };
            // x**v * exp(a * x**n) / v - a * n * higher / v
            let first: Basic = d2(&m2(&pw(x, v)?, &exp(&axn)?)?, v)?;
            let second: Basic = d2(&m2(&an, &higher)?, v)?;
            return Ok(Some(s2(&first, &second)?));
        }
    }
    let ax2: Basic = m2(a, &pw(x, &int(2))?)?;
    if *n == int(2) {
        if let Some(vi) = integer_of(v) {
            if vi.rem_euclid(2) == 1 && vi != 1 {
                let two_a: Basic = m2(&int(2), a)?;
                if vi > 1 {
                    let v2: Basic = s2(v, &int(2))?;
                    let Some(lower) = power_exponential_raw(&v2, a, n, x, assumptions)? else {
                        return Ok(None);
                    };
                    let first: Basic = d2(&m2(&pw(x, &v2)?, &exp(&ax2)?)?, &two_a)?;
                    let second: Basic = d2(&m2(&v2, &lower)?, &two_a)?;
                    return Ok(Some(s2(&first, &second)?));
                }
                let Some(higher) = power_exponential_raw(&a2(v, &int(2))?, a, n, x, assumptions)? else {
                    return Ok(None);
                };
                let first: Basic = d2(&m2(&pw(x, v)?, &exp(&ax2)?)?, v)?;
                let second: Basic = d2(&m2(&two_a, &higher)?, v)?;
                return Ok(Some(s2(&first, &second)?));
            }
        }
    }
    let negative: Option<bool> = ask(&crate::core::relational::lt(a, &int(0))?, assumptions, None)?;
    let positive: Option<bool> = ask(&crate::core::relational::gt(a, &int(0))?, assumptions, None)?;
    let real_everywhere: bool = integer_of(n).is_some_and(|k| k.rem_euclid(2) == 0)
        || ask(&crate::core::relational::gt(x, &int(0))?, assumptions, None)? == Some(true);
    if *n == int(2) && *v == int(1) {
        // sqrt(pi) * erf(sqrt(-a) * x) / (2 * sqrt(-a)), with erfi above
        let pi: Basic = Basic::pi();
        let sa: Basic = sqrt(&neg(a)?)?;
        let below: Basic = d2(&m2(&sqrt(&pi)?, &erf(&m2(&sa, x)?)?)?, &m2(&int(2), &sa)?)?;
        let sb: Basic = sqrt(a)?;
        let above: Basic = d2(&m2(&sqrt(&pi)?, &erfi(&m2(&sb, x)?)?)?, &m2(&int(2), &sb)?)?;
        if negative == Some(true) {
            return Ok(Some(below));
        }
        if positive == Some(true) {
            return Ok(Some(above));
        }
        return Ok(Some(crate::functions::elementary::piecewise::piecewise(&[
            (below, crate::core::relational::lt(a, &int(0))?),
            (above, crate::core::logic::boolean(true)),
        ])?));
    }
    if negative == Some(true) && real_everywhere {
        // -(-a)**(-s) * uppergamma(s, -a * x**n) / n
        let factor: Basic = neg(&pw(&neg(a)?, &neg(&s)?)?)?;
        let g: Basic = uppergamma(&s, &m2(&neg(a)?, &pw(x, n)?)?)?;
        return Ok(Some(d2(&m2(&factor, &g)?, n)?));
    }
    // x**v * hyper((s,), (s + 1,), a * x**n) / v
    let h: Basic = crate::functions::special::hyper::hyper(std::slice::from_ref(&s), &[a2(&s, &int(1))?], &axn)?;
    Ok(Some(d2(&m2(&pw(x, v)?, &h)?, v)?))
}

fn monomial_exponential(term: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Basic, Basic)>> {
    let mut coefficient: Basic = int(1);
    let mut power: Basic = int(0);
    let mut exponent: Option<Basic> = None;
    for f in factors_of(term) {
        if !f.has(x) {
            coefficient = m2(&coefficient, &f)?;
        } else if f == *x {
            power = a2(&power, &int(1))?;
        } else if is_power_of(&f, x) {
            power = a2(&power, &f.args()[1])?;
        } else if is_exp(&f) && exponent.is_none() {
            exponent = Some(f.args()[0].clone());
        } else {
            return Ok(None);
        }
    }
    let Some(exponent) = exponent else { return Ok(None) };
    let (constant, rest) = as_independent(&exponent, x, true)?;
    let ft: Basic = crate::core::exprtools::factor_terms(&rest, false, false, false, true)?;
    let (a, monomial) = as_independent(&ft, x, false)?;
    let n: Basic = if monomial == *x {
        int(1)
    } else if is_power_of(&monomial, x) {
        monomial.args()[1].clone()
    } else {
        return Ok(None);
    };
    if a == int(0) || n == int(0) {
        return Ok(None);
    }
    Ok(Some((m2(&coefficient, &exp(&constant)?)?, a2(&power, &int(1))?, a, n)))
}

/// `x**(v-1)*exp(a*x**n + b)` and sums of such (`power_exponential`).
pub fn power_exponential(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let combined: Basic = powsimp_exp(&expand(f)?)?;
    let mut total: Basic = int(0);
    for term in terms_of(&combined) {
        if !has_exp(&term) {
            let (coefficient, monomial) = as_independent(&term, x, false)?;
            if monomial == int(1) {
                total = a2(&total, &m2(&coefficient, x)?)?;
            } else if monomial == *x {
                total = a2(&total, &d2(&m2(&coefficient, &pw(x, &int(2))?)?, &int(2))?)?;
            } else if is_power_of(&monomial, x) {
                let e: Basic = monomial.args()[1].clone();
                let piece: Basic = if e == int(-1) {
                    m2(&coefficient, &log(x)?)?
                } else {
                    let e1: Basic = a2(&e, &int(1))?;
                    d2(&m2(&coefficient, &pw(x, &e1)?)?, &e1)?
                };
                total = a2(&total, &piece)?;
            } else {
                return Ok(None);
            }
            continue;
        }
        let Some((c, v, a, n)) = monomial_exponential(&term, x)? else {
            return Ok(None);
        };
        let Some(piece) = power_exponential_raw(&v, &a, &n, x, assumptions)? else {
            return Ok(None);
        };
        total = a2(&total, &m2(&c, &piece)?)?;
    }
    if !checks(&total, f, x, assumptions)? {
        return Ok(None);
    }
    Ok(Some(total))
}

fn quadratic_term(term: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let mut found: Option<Basic> = None;
    let mut coefficient: Basic = int(1);
    let mut m: Basic = int(0);
    for g in factors_of(term) {
        if !g.has(x) {
            coefficient = m2(&coefficient, &g)?;
        } else if g == *x {
            m = a2(&m, &int(1))?;
        } else if g.is_pow() && g.args()[0] == *x && integer_of(&g.args()[1]).is_some_and(|k| k > 0) {
            m = a2(&m, &g.args()[1])?;
        } else if is_exp(&g) && found.is_none() {
            found = Some(g.args()[0].clone());
        } else {
            return Ok(None);
        }
    }
    let Some(found) = found else { return Ok(None) };
    if crate::core::expr::polynomial_like(
        &found,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return Ok(None);
    }
    let Some(poly) = coefficients(&found, x)? else {
        return Ok(None);
    };
    if degree(&poly) != 2 {
        return Ok(None);
    }
    let (a, b, c) = (get(&poly, 2), get(&poly, 1), get(&poly, 0));
    if b == int(0) {
        return Ok(None);
    }
    let h: Basic = d2(&b, &m2(&int(2), &a)?)?;
    let t: Basic = dummy("t", false);
    // (t - h)**m * exp(a * t**2) * exp(c - b**2 / (4 * a))
    let shifted: Basic = expand(&m2(
        &m2(&pw(&s2(&t, &h)?, &m)?, &exp(&m2(&a, &pw(&t, &int(2))?)?)?)?,
        &exp(&s2(&c, &d2(&pw(&b, &int(2))?, &m2(&int(4), &a)?)?)?)?,
    )?)?;
    let Some(inner) = power_exponential(&shifted, &t, assumptions)? else {
        return Ok(None);
    };
    Ok(Some(m2(&coefficient, &subs(&inner, &t, &a2(x, &h)?)?)?))
}

fn quadratic_exponential(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let combined: Basic = powsimp_exp(&expand(f)?)?;
    let mut total: Basic = int(0);
    for term in terms_of(&combined) {
        let Some(piece) = quadratic_term(&term, x, assumptions)? else {
            return Ok(None);
        };
        total = a2(&total, &piece)?;
    }
    if !checks(&total, f, x, assumptions)? {
        return Ok(None);
    }
    Ok(Some(total))
}

fn exponential_generator(f: &Basic, x: &Basic) -> OptR {
    let mut rates: Vec<Basic> = Vec::new();
    for node in atoms_of(f, is_exp) {
        let argument: Basic = node.args()[0].clone();
        if !argument.has(x) {
            continue;
        }
        let (k, rest) = as_independent(&argument, x, false)?;
        if rest != *x {
            return Ok(None);
        }
        rates.push(k);
    }
    if rates.is_empty() {
        return Ok(None);
    }
    let mut ratios: Vec<Basic> = Vec::new();
    for k in &rates {
        let r: Basic = cancel(&d2(k, &rates[0])?)?;
        if !r.is_rational_class() {
            return Ok(None);
        }
        ratios.push(r);
    }
    let mut common: Basic = ratios[0].clone();
    for r in &ratios[1..] {
        common = crate::polys::polytools::gcd_expr(&common, r, &[])?;
    }
    let mut c: Basic = m2(&rates[0], &common)?;
    if crate::core::printing::could_extract_minus_sign(&c) {
        c = neg(&c)?;
    }
    Ok(Some(c))
}

fn ratint(f: &Basic, x: &Basic) -> OptR {
    let symbol: Basic = crate::core::symbol::Symbol::new("t", &[]).0;
    crate::timeout::attempt_within(budget(), || {
        let integrand = crate::integrals::risch::rationaltools::Integrand::of(f)?;
        crate::integrals::risch::rationaltools::ratint(&integrand, x, &symbol, None)
    })
}

fn is_rational_in(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}

/// Rational functions of `exp(c*x)` (`exponential_rational`).
pub fn exponential_rational(f0: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let f: Basic = powsimp_exp(f0)?;
    let Some(c) = exponential_generator(&f, x)? else {
        return Ok(None);
    };
    let u: Basic = dummy("u", true);
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    for node in atoms_of(&f, is_exp) {
        let argument: Basic = node.args()[0].clone();
        if argument.has(x) {
            let k: Basic = as_independent(&argument, x, false)?.0;
            replacement.insert(node, pw(&u, &cancel(&d2(&k, &c)?)?)?);
        }
    }
    let g: Basic = crate::core::function::xreplace_evaluated(&f, &replacement)?;
    if g.has(x) {
        return Ok(None);
    }
    let integrand: Basic = cancel(&d2(&g, &m2(&c, &u)?)?)?;
    let found: Option<Basic> = if is_rational_in(&integrand, &u) {
        ratint(&integrand, &u)?
    } else {
        binomial_in(&integrand, &u)?
    };
    let Some(big_f) = found else { return Ok(None) };
    let mut result: Basic = subs(&big_f, &u, &exp(&m2(&c, x)?)?)?;
    result = crate::core::function::replace_where(
        &result,
        &|e: &Basic| e.is_function(Function::Log) && is_exp(&e.args()[0]),
        &|e: &Basic| Ok(e.args()[0].args()[0].clone()),
    )?;
    if !checks(&result, f0, x, assumptions)? {
        return Ok(None);
    }
    Ok(Some(result))
}

fn binomial_in(integrand: &Basic, u: &Basic) -> OptR {
    let fractional =
        |n: &Basic| n.is_pow() && n.args()[1].is_rational_class() && !n.args()[1].is_integer_value() && n.has(u);
    let mut radical: Option<Basic> = None;
    for node in atoms_of(integrand, |n| n.is_pow()) {
        if fractional(&node) {
            if let Some(r) = &radical {
                if node.args()[0] != r.args()[0] {
                    return Ok(None);
                }
            }
            radical = Some(node);
        }
    }
    let Some(radical) = radical else { return Ok(None) };
    let base: Basic = radical.args()[0].clone();
    let Some(linear) = coefficients(&base, u)? else {
        return Ok(None);
    };
    if degree(&linear) != 1 {
        return Ok(None);
    }
    let (b, a) = (get(&linear, 1), get(&linear, 0));
    let mut exponents: Vec<(Basic, Rat)> = Vec::new();
    for node in atoms_of(integrand, |n| n.is_pow()) {
        if node.args()[1].is_rational_class() && node.has(u) && node.args()[0] == base {
            let e: Rat = node.args()[1].as_rational().expect("rational").clone();
            exponents.push((node, e));
        }
    }
    let den = |r: &Rat| i64::try_from(&Int::from(crate::numbers::denom(r))).expect("small");
    let num = |r: &Rat| i64::try_from(&crate::numbers::numer(r)).expect("small");
    let q: i64 = exponents.iter().map(|(_, e)| den(e)).max().unwrap_or(1);
    let t: Basic = dummy("t", true);
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    for (node, e) in &exponents {
        replacement.insert(node.clone(), pw(&t, &int(num(e) * (q / den(e))))?);
    }
    // integrand.xreplace(...).subs(u, (t**q - a) / b) * q * t**(q - 1) / b
    let replaced: Basic = crate::core::function::xreplace_evaluated(integrand, &replacement)?;
    let substituted: Basic = subs(&replaced, u, &d2(&s2(&pw(&t, &int(q))?, &a)?, &b)?)?;
    let in_t: Basic = cancel(&d2(&m2(&m2(&substituted, &int(q))?, &pw(&t, &int(q - 1))?)?, &b)?)?;
    if !is_rational_in(&in_t, &t) {
        return Ok(None);
    }
    let Some(g) = ratint(&in_t, &t)? else { return Ok(None) };
    Ok(Some(subs(&g, &t, &pw(&base, &rat(1, q))?)?))
}

fn linear_bases(f: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let mut found: Vec<Basic> = Vec::new();
    for node in atoms_of(f, |n| n.is_pow()) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if exponent.has(x) || !base.has(x) || *base == *x {
            continue;
        }
        if let Some(poly) = coefficients(base, x)? {
            if degree(&poly) == 1 && !found.contains(base) {
                found.push(base.clone());
            }
        }
    }
    for base in factors_of(f) {
        let polynomial: bool = crate::core::expr::polynomial_like(
            &base,
            std::slice::from_ref(x),
            crate::core::expr::PolynomialKind::Polynomial,
        ) == Some(true);
        if base.has(x) && base != *x && polynomial && !found.contains(&base) {
            if let Some(poly) = coefficients(&base, x)? {
                if degree(&poly) == 1 {
                    found.push(base);
                }
            }
        }
    }
    Ok(found)
}

type Route = fn(&Basic, &Basic, &[Basic]) -> OptR;

fn shifted_exponential(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let routes: [Route; 2] = [power_exponential, quadratic_exponential];
    for base in linear_bases(f, x)? {
        let Some(poly) = coefficients(&base, x)? else { continue };
        let (a, b) = (get(&poly, 1), get(&poly, 0));
        let t: Basic = dummy("t", false);
        let shifted: Basic = d2(&subs(f, x, &d2(&s2(&t, &b)?, &a)?)?, &a)?;
        for route in routes {
            let Some(found) = crate::timeout::attempt_within(budget(), || route(&shifted, &t, assumptions))?.flatten()
            else {
                continue;
            };
            let candidate: Basic = subs(&found, &t, &base)?;
            if checks(&candidate, f, x, assumptions)? {
                return Ok(Some(candidate));
            }
        }
    }
    Ok(None)
}

/// The routes of the module in order (`exponential_antiderivative`).
pub fn exponential_antiderivative(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    if !has_exp(f) {
        return Ok(None);
    }
    let routes: [Route; 8] = [
        power_exponential,
        quadratic_exponential,
        shifted_exponential,
        exponential_rational,
        nested_power_exponential,
        rational_exponential,
        reciprocal_square_exponential,
        composite_gaussian,
    ];
    for route in routes {
        if let Some(found) = crate::timeout::attempt_within(budget(), || route(f, x, assumptions))?.flatten() {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

fn nested_power(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let mut d: Basic = int(1);
    let mut depth: usize = 0;
    let mut node: Basic = e.clone();
    while node.is_pow() && !node.args()[1].has(x) {
        d = m2(&d, &node.args()[1])?;
        depth += 1;
        node = node.args()[0].clone();
    }
    if node != *x || depth < 2 {
        return Ok(None);
    }
    Ok(Some((e.clone(), d)))
}

type NestedTerm = (Basic, Basic, Basic, Basic, Basic, Basic);

fn nested_term(term: &Basic, x: &Basic) -> PolyResult<Option<NestedTerm>> {
    let mut coefficient: Basic = int(1);
    let mut prefactor: Basic = int(1);
    let mut deg: Basic = int(0);
    let mut nested: bool = false;
    let mut exponent: Option<Basic> = None;
    for f in factors_of(term) {
        let inner: Option<(Basic, Basic)> = nested_power(&f, x)?;
        if !f.has(x) {
            coefficient = m2(&coefficient, &f)?;
        } else if f == *x {
            prefactor = m2(&prefactor, &f)?;
            deg = a2(&deg, &int(1))?;
        } else if is_power_of(&f, x) {
            prefactor = m2(&prefactor, &f)?;
            deg = a2(&deg, &f.args()[1])?;
        } else if f.is_pow() && inner.is_some() {
            prefactor = m2(&prefactor, &f)?;
            deg = a2(&deg, &inner.expect("inner").1)?;
            nested = true;
        } else if is_exp(&f) && exponent.is_none() {
            exponent = Some(f.args()[0].clone());
        } else {
            return Ok(None);
        }
    }
    let Some(exponent) = exponent else { return Ok(None) };
    let (constant, rest) = as_independent(&exponent, x, true)?;
    let ft: Basic = crate::core::exprtools::factor_terms(&rest, false, false, false, true)?;
    let (a, monomial) = as_independent(&ft, x, false)?;
    let (w, d): (Basic, Basic) = if let Some(found) = nested_power(&monomial, x)? {
        nested = true;
        found
    } else if monomial == *x {
        (x.clone(), int(1))
    } else if is_power_of(&monomial, x) {
        (monomial.clone(), monomial.args()[1].clone())
    } else {
        return Ok(None);
    };
    if !nested || a == int(0) || d == int(0) {
        return Ok(None);
    }
    Ok(Some((m2(&coefficient, &exp(&constant)?)?, prefactor, deg, a, w, d)))
}

/// `x**q*exp(a*(x**r)**s)` and kin (`nested_power_exponential`).
pub fn nested_power_exponential(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let combined: Basic = powsimp_exp(&expand(f)?)?;
    let mut total: Basic = int(0);
    for term in terms_of(&combined) {
        let Some((c, v, q, a, w, d)) = nested_term(&term, x)? else {
            return Ok(None);
        };
        let u: Basic = m2(&neg(&a)?, &w)?;
        let s_: Basic = d2(&a2(&q, &int(1))?, &d)?;
        // c * (-x * V * u**(-s_) * uppergamma(s_, u) / d)
        let inner: Basic = d2(
            &m2(&m2(&m2(&neg(x)?, &v)?, &pw(&u, &neg(&s_)?)?)?, &uppergamma(&s_, &u)?)?,
            &d,
        )?;
        total = a2(&total, &m2(&c, &inner)?)?;
    }
    if !checks(&total, f, x, assumptions)? {
        return Ok(None);
    }
    Ok(Some(total))
}

/// A rational function times `exp(c*x + d)` (`rational_exponential`).
pub fn rational_exponential(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let exponentials: Vec<Basic> = atoms_of(f, |n| is_exp(n) && n.has(x));
    if exponentials.len() != 1 {
        return Ok(None);
    }
    let e0: Basic = exponentials[0].clone();
    let argument: Basic = e0.args()[0].clone();
    let (d, rest) = as_independent(&argument, x, true)?;
    let (c, variable) = as_independent(&rest, x, false)?;
    if variable != *x || c.has(x) {
        return Ok(None);
    }
    let r: Basic = cancel(&d2(f, &e0)?)?;
    if has_exp(&r) || !is_rational_in(&r, x) {
        return Ok(None);
    }
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&r)?;
    let (Some(q), Some(p)) = (coefficients(&denominator, x)?, coefficients(&numerator, x)?) else {
        return Ok(None);
    };
    if degree(&q) < 1 {
        return Ok(None);
    }
    let Some(found) = roots(&q, x)? else { return Ok(None) };
    let (quotient, remainder) = divmod(&p, &q)?;
    let mut total: Basic = int(0);
    if !quotient.is_empty() {
        let Some(polynomial) = power_exponential(&m2(&from_coefficients(&quotient, x)?, &e0)?, x, assumptions)? else {
            return Ok(None);
        };
        total = a2(&total, &polynomial)?;
    }
    let leading: Basic = get(&q, degree(&q));
    let rest_: Basic = from_coefficients(&remainder, x)?;
    for (root, multiplicity) in &found {
        let mut others: Basic = leading.clone();
        for (other, om) in &found {
            if other != root {
                others = m2(&others, &pw(&s2(x, other)?, &int(*om))?)?;
            }
        }
        let stripped: Basic = d2(&rest_, &others)?;
        for j in (1..=*multiplicity).rev() {
            let order: i64 = multiplicity - j;
            let mut derivative: Basic = stripped.clone();
            for _ in 0..order {
                derivative = diff(&derivative, x)?;
            }
            let mut coefficient: Basic = if order != 0 {
                d2(
                    &subs(&derivative, x, root)?,
                    &crate::functions::combinatorial::factorials::factorial(&int(order))?,
                )?
            } else {
                subs(&stripped, x, root)?
            };
            coefficient = cancel(&coefficient)?;
            if coefficient == int(0) {
                continue;
            }
            total = a2(
                &total,
                &m2(&m2(&coefficient, &exp(&d)?)?, &exponential_over_power(&c, root, j, x)?)?,
            )?;
        }
    }
    if !checks(&total, f, x, assumptions)? {
        return Ok(None);
    }
    Ok(Some(total))
}

fn exponential_over_power(c: &Basic, r: &Basic, k: i64, x: &Basic) -> R {
    if k == 1 {
        return m2(&exp(&m2(c, r)?)?, &ei(&m2(c, &s2(x, r)?)?)?);
    }
    // -exp(c * x) / ((k - 1) * (x - r)**(k - 1)) + c / (k - 1) * lower
    let first: Basic = d2(
        &neg(&exp(&m2(c, x)?)?)?,
        &m2(&int(k - 1), &pw(&s2(x, r)?, &int(k - 1))?)?,
    )?;
    let second: Basic = m2(&d2(c, &int(k - 1))?, &exponential_over_power(c, r, k - 1, x)?)?;
    a2(&first, &second)
}

fn coeff_of(e: &Basic, p: &Basic) -> R {
    crate::core::expr::coeff(e, p)
}

fn linear_pair(equations: &[Basic], p: &Basic, q: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let mut rows: Vec<(Basic, Basic, Basic)> = Vec::new();
    for e in equations {
        let e_: Basic = expand(e)?;
        let (a, b) = (coeff_of(&e_, p)?, coeff_of(&e_, q)?);
        let rest: Basic = cancel(&s2(&s2(&e_, &m2(&a, p)?)?, &m2(&b, q)?)?)?;
        let has_pq = |v: &Basic| v.has(p) || v.has(q);
        if has_pq(&a) || has_pq(&b) || has_pq(&rest) {
            return Ok(None);
        }
        rows.push((a, b, rest));
    }
    for i in 0..rows.len() {
        let (a1, b1, c1) = rows[i].clone();
        for (a2_, b2_, c2_) in &rows[i + 1..] {
            let determinant: Basic = cancel(&s2(&m2(&a1, &b2_)?, &m2(&a2_, &b1)?)?)?;
            if determinant == int(0) {
                continue;
            }
            let pv: Basic = cancel(&d2(&s2(&m2(&b1, &c2_)?, &m2(&b2_, &c1)?)?, &determinant)?)?;
            let qv: Basic = cancel(&d2(&s2(&m2(&a2_, &c1)?, &m2(&a1, &c2_)?)?, &determinant)?)?;
            for (a, b, c) in &rows {
                if cancel(&a2(&a2(&m2(a, &pv)?, &m2(b, &qv)?)?, c)?)? != int(0) {
                    return Ok(None);
                }
            }
            return Ok(Some((pv, qv)));
        }
    }
    Ok(None)
}

fn independent_of_two(e: &Basic, x: &Basic, l: &Basic, as_add: bool) -> PolyResult<(Basic, Basic)> {
    let depends = |t: &Basic| t.has(x) || t.has(l);
    if as_add {
        if !e.is_add() {
            return Ok(if depends(e) {
                (int(0), e.clone())
            } else {
                (e.clone(), int(0))
            });
        }
        let free: Vec<Basic> = e.args().iter().filter(|a| !depends(a)).cloned().collect();
        let dep: Vec<Basic> = e.args().iter().filter(|a| depends(a)).cloned().collect();
        return Ok((crate::core::add::add(&free)?, crate::core::add::add(&dep)?));
    }
    if !e.is_mul() {
        return Ok(if depends(e) {
            (int(1), e.clone())
        } else {
            (e.clone(), int(1))
        });
    }
    let free: Vec<Basic> = e.args().iter().filter(|a| !depends(a)).cloned().collect();
    let dep: Vec<Basic> = e.args().iter().filter(|a| depends(a)).cloned().collect();
    Ok((crate::core::mul::mul(&free)?, crate::core::mul::mul(&dep)?))
}

/// `exp(a*g(x)**2 + c)` times a multiple of `g'` (`composite_gaussian`).
pub fn composite_gaussian(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let exponentials: Vec<Basic> = atoms_of(f, |n| is_exp(n) && n.has(x));
    if exponentials.len() != 1 {
        return Ok(None);
    }
    let e0: Basic = exponentials[0].clone();
    let l: Basic = dummy("L", false);
    let argument: Basic = expand(&subs(&e0.args()[0], &log(x)?, &l)?)?;
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&cancel(&argument)?)?;
    let poly_in = |e: &Basic, gens: &[Basic]| {
        crate::core::expr::polynomial_like(e, gens, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
    };
    if !poly_in(&denominator, std::slice::from_ref(x)) || !poly_in(&numerator, &[x.clone(), l.clone()]) {
        return Ok(None);
    }
    let (constant, dependent) = independent_of_two(&argument, x, &l, true)?;
    let rest: Basic = cancel(&d2(f, &e0)?)?;
    let candidates: [(Basic, Basic); 2] = [(int(0), argument.clone()), (constant, dependent)];
    for (c, body) in candidates {
        let Some(factored) =
            crate::timeout::attempt_within(budget(), || crate::polys::polytools::factor(&body, &[], false, true))?
        else {
            continue;
        };
        let (a, square) = independent_of_two(&factored, x, &l, false)?;
        if a == int(0) || a.has(x) {
            continue;
        }
        let mut g: Basic = int(1);
        for piece in factors_of(&square) {
            if !(piece.is_pow() && integer_of(&piece.args()[1]).is_some_and(|k| k.rem_euclid(2) == 0)) {
                g = int(0);
                break;
            }
            g = m2(&g, &pw(&piece.args()[0], &d2(&piece.args()[1], &int(2))?)?)?;
        }
        if g == int(0) {
            continue;
        }
        g = subs(&g, &l, &log(x)?)?;
        let k: Basic = cancel(&d2(&rest, &diff(&g, x)?)?)?;
        if !k.has(x) {
            let total: Basic = m2(&k, &gaussian_primitive(&a, &g, &c)?)?;
            if checks(&total, f, x, assumptions)? {
                return Ok(Some(total));
            }
        }
        let terms: Vec<Basic> = terms_of(&expand(&g)?);
        if terms.len() == 2 {
            let (u, v) = (terms[0].clone(), terms[1].clone());
            let product: Basic = cancel(&m2(&u, &v)?)?;
            if !product.has(x) && u.has(x) && v.has(x) {
                let (p_, q_) = (dummy("p", false), dummy("q", false));
                let difference: Basic = cancel(&s2(&s2(&rest, &m2(&p_, &diff(&u, x)?)?)?, &m2(&q_, &diff(&v, x)?)?)?)?;
                let numerator_: Basic = crate::core::expr::as_numer_denom(&difference)?.0;
                let solution: Option<(Basic, Basic)> = match coefficients(&numerator_, x)? {
                    None => None,
                    Some(cs) => linear_pair(&cs.iter().map(|(_, v)| v.clone()).collect::<Vec<Basic>>(), &p_, &q_)?,
                };
                if let Some((pv, qv)) = solution {
                    if !pv.has(x) && !qv.has(x) {
                        let h: Basic = s2(&u, &v)?;
                        // (pv + qv) / 2 * G(a, g, c) + (pv - qv) / 2 * G(a, h, c + 4 * a * product)
                        let first: Basic = m2(&d2(&a2(&pv, &qv)?, &int(2))?, &gaussian_primitive(&a, &g, &c)?)?;
                        let shifted_c: Basic = a2(&c, &m2(&m2(&int(4), &a)?, &product)?)?;
                        let second: Basic =
                            m2(&d2(&s2(&pv, &qv)?, &int(2))?, &gaussian_primitive(&a, &h, &shifted_c)?)?;
                        let total: Basic = a2(&first, &second)?;
                        if checks(&total, f, x, assumptions)? {
                            return Ok(Some(total));
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

fn gaussian_primitive(a: &Basic, g: &Basic, c: &Basic) -> R {
    let pi: Basic = Basic::pi();
    if is(a, Predicate::positive) == Some(true) {
        // sqrt(pi) * exp(c) * erfi(sqrt(a) * g) / (2 * sqrt(a))
        let sa: Basic = sqrt(a)?;
        return d2(
            &m2(&m2(&sqrt(&pi)?, &exp(c)?)?, &erfi(&m2(&sa, g)?)?)?,
            &m2(&int(2), &sa)?,
        );
    }
    let big_a: Basic = sqrt(&neg(a)?)?;
    d2(
        &m2(&m2(&sqrt(&pi)?, &exp(c)?)?, &erf(&m2(&big_a, g)?)?)?,
        &m2(&int(2), &big_a)?,
    )
}

fn coeff_power(e: &Basic, x: &Basic, n: i64) -> R {
    let target: Basic = pw(x, &int(n))?;
    let mut found: Vec<Basic> = Vec::new();
    for term in terms_of(e) {
        let mut factors: Vec<Basic> = factors_of(&term);
        if let Some(k) = factors.iter().position(|f| *f == target) {
            factors.remove(k);
            found.push(crate::core::mul::mul(&factors)?);
        }
    }
    crate::core::add::add(&found)
}

/// `c*exp(alpha*x**2 + beta/x**2 + gamma)` (`reciprocal_square_exponential`).
pub fn reciprocal_square_exponential(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let (c, rest) = as_independent(f, x, false)?;
    if !is_exp(&rest) {
        return Ok(None);
    }
    let argument: Basic = expand(&rest.args()[0])?;
    let (gamma_, dependent) = as_independent(&argument, x, true)?;
    let alpha: Basic = coeff_power(&dependent, x, 2)?;
    let beta: Basic = coeff_power(&dependent, x, -2)?;
    if alpha == int(0) || beta == int(0) {
        return Ok(None);
    }
    let check: Basic = expand(&s2(
        &s2(&dependent, &m2(&alpha, &pw(x, &int(2))?)?)?,
        &d2(&beta, &pw(x, &int(2))?)?,
    )?)?;
    if check != int(0) || alpha.has(x) || beta.has(x) {
        return Ok(None);
    }
    let (big_a, big_b) = (sqrt(&neg(&alpha)?)?, sqrt(&neg(&beta)?)?);
    let pi: Basic = Basic::pi();
    // c * exp(gamma) * sqrt(pi) / (4 * A) * (exp(2AB) erf(A x + B/x) + exp(-2AB) erf(A x - B/x))
    let lead: Basic = d2(&m2(&m2(&c, &exp(&gamma_)?)?, &sqrt(&pi)?)?, &m2(&int(4), &big_a)?)?;
    let ab2: Basic = m2(&m2(&int(2), &big_a)?, &big_b)?;
    let ax: Basic = m2(&big_a, x)?;
    let bx: Basic = d2(&big_b, x)?;
    let plus: Basic = m2(&exp(&ab2)?, &erf(&a2(&ax, &bx)?)?)?;
    let minus: Basic = m2(&exp(&neg(&ab2)?)?, &erf(&s2(&ax, &bx)?)?)?;
    let total: Basic = m2(&lead, &a2(&plus, &minus)?)?;
    if !checks(&total, f, x, assumptions)? {
        return Ok(None);
    }
    Ok(Some(total))
}
