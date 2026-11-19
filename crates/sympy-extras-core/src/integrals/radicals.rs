//! Port of `sympy_extras/integrals/radicals.py`: real antiderivatives of
//! `x^n Q(x)^{m/2}` for a quadratic `Q`, by the reduction formulas of
//! Gradshteyn and Ryzhik 2.26 (arcsines, inverse hyperbolic sines and
//! logarithms in their real forms), and Euler's substitutions for the
//! rational functions of `x` and `sqrt(Q)` beyond them. The expressions
//! are built in the order of the Python's operators (two arguments at a
//! time), which decides their printed forms.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::assumptions::ask::ask;
use crate::core::assumptions::Predicate;
use crate::core::basic::Basic;
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn half(m: i64) -> Basic {
    Basic::from_rational(Rat::from(Int::from(m)) / Rat::from(Int::from(2)))
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
    pw(a, &half(1))
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}

/// The coefficients of a polynomial in `x` (`_coefficients`).
pub(crate) fn coefficients(e: &Basic, x: &Basic) -> PolyResult<Option<Vec<(i64, Basic)>>> {
    if crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return Ok(None);
    }
    let mut found: Vec<(i64, Basic)> = Vec::new();
    let expanded: Basic = crate::core::function::expand_default(e)?;
    for term in crate::core::expr::terms_of(&expanded) {
        let (coefficient, rest) = crate::core::expr::as_independent(&term, x, false)?;
        let n: i64 = if rest == int(1) {
            0
        } else if rest == *x {
            1
        } else if rest.is_pow()
            && rest.args()[0] == *x
            && rest.args()[1].is_integer_value()
            && is(&rest.args()[1], Predicate::positive) == Some(true)
        {
            i64::try_from(&crate::numbers::numer(
                rest.args()[1].as_rational().expect("an integer"),
            ))
            .expect("a small degree")
        } else {
            return Ok(None);
        };
        match found.iter_mut().find(|(k, _)| *k == n) {
            Some(entry) => entry.1 = a2(&entry.1, &coefficient)?,
            None => found.push((n, coefficient)),
        }
    }
    found.retain(|(_, c)| *c != int(0));
    Ok(Some(found))
}

fn degree(coefficients: &[(i64, Basic)]) -> i64 {
    coefficients.iter().map(|(n, _)| *n).max().unwrap_or(-1)
}

fn coefficient(coefficients: &[(i64, Basic)], n: i64) -> Basic {
    coefficients
        .iter()
        .find(|(k, _)| *k == n)
        .map(|(_, c)| c.clone())
        .unwrap_or_else(|| int(0))
}

fn quadratic_coefficients(q: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    let Some(c) = coefficients(q, x)? else { return Ok(None) };
    if !matches!(degree(&c), 1 | 2) {
        return Ok(None);
    }
    Ok(Some((coefficient(&c, 2), coefficient(&c, 1), coefficient(&c, 0))))
}

/// The antiderivatives of `x^n Q^{m/2}` for one `Q` (`_Table`).
struct Table {
    a: Basic,
    b: Basic,
    c: Basic,
    x: Basic,
    q: Basic,
    discriminant: Basic,
    assumptions: Vec<Basic>,
    memo: HashMap<(i64, i64), Option<Basic>>,
}

impl Table {
    fn new(a: Basic, b: Basic, c: Basic, x: Basic, assumptions: &[Basic]) -> PolyResult<Table> {
        // a * x**2 + b * x + c, b**2 - 4 * a * c
        let q: Basic = a2(&a2(&m2(&a, &pw(&x, &int(2))?)?, &m2(&b, &x)?)?, &c)?;
        let discriminant: Basic = s2(&pw(&b, &int(2))?, &m2(&m2(&int(4), &a)?, &c)?)?;
        Ok(Table {
            a,
            b,
            c,
            x,
            q,
            discriminant,
            assumptions: assumptions.to_vec(),
            memo: HashMap::new(),
        })
    }

    fn positive(&self, e: &Basic) -> PolyResult<bool> {
        Ok(is(e, Predicate::positive) == Some(true)
            || ask(&crate::core::relational::gt(e, &int(0))?, &self.assumptions, None)? == Some(true))
    }

    fn negative(&self, e: &Basic) -> PolyResult<bool> {
        Ok(is(e, Predicate::negative) == Some(true)
            || ask(&crate::core::relational::lt(e, &int(0))?, &self.assumptions, None)? == Some(true))
    }

    /// `2 * a * x + b`.
    fn linear(&self) -> R {
        a2(&m2(&m2(&int(2), &self.a)?, &self.x)?, &self.b)
    }

    /// The antiderivative of `Q^{m/2}` (`power`).
    fn power(&mut self, m: i64) -> OptR {
        let (a, b, x, q) = (self.a.clone(), self.b.clone(), self.x.clone(), self.q.clone());
        let disc: Basic = self.discriminant.clone();
        if a == int(0) {
            if b == int(0) {
                return Ok(None);
            }
            let e: Basic = a2(&half(m), &int(1))?;
            return Ok(Some(d2(&pw(&q, &e)?, &m2(&b, &e)?)?));
        }
        if m == -1 {
            if self.negative(&a)? {
                if !self.positive(&disc)? {
                    return Ok(None);
                }
                let arg: Basic = d2(&self.linear()?, &sqrt(&disc)?)?;
                let asin: Basic = crate::functions::elementary::inverse::asin(&arg)?;
                return Ok(Some(d2(&neg(&asin)?, &sqrt(&neg(&a)?)?)?));
            }
            if self.positive(&a)? {
                if self.negative(&disc)? {
                    let arg: Basic = d2(&self.linear()?, &sqrt(&neg(&disc)?)?)?;
                    let asinh: Basic = crate::functions::elementary::inverse::asinh(&arg)?;
                    return Ok(Some(d2(&asinh, &sqrt(&a)?)?));
                }
                if self.positive(&disc)? {
                    // log(2 * sqrt(a) * sqrt(Q) + 2 * a * x + b) / sqrt(a)
                    let first: Basic = m2(&m2(&int(2), &sqrt(&a)?)?, &sqrt(&q)?)?;
                    let inner: Basic = a2(&a2(&first, &m2(&m2(&int(2), &a)?, &x)?)?, &b)?;
                    return Ok(Some(d2(&log(&inner)?, &sqrt(&a)?)?));
                }
                if disc == int(0) {
                    return Ok(None);
                }
            }
            return Ok(None);
        }
        if m == -3 {
            if disc == int(0) {
                return Ok(None);
            }
            // -2 * (2 * a * x + b) / (self.discriminant * sqrt(Q))
            let top: Basic = m2(&int(-2), &self.linear()?)?;
            return Ok(Some(d2(&top, &m2(&disc, &sqrt(&q)?)?)?));
        }
        if m < -3 {
            let k: i64 = (-m - 1) / 2;
            let Some(previous) = self.power(m + 2)? else {
                return Ok(None);
            };
            if disc == int(0) {
                return Ok(None);
            }
            let scale: Basic = m2(&int(2 * k - 1), &disc)?;
            let first: Basic = d2(
                &m2(&m2(&int(-2), &self.linear()?)?, &pw(&q, &a2(&half(m), &int(1))?)?)?,
                &scale,
            )?;
            let second: Basic = d2(&m2(&m2(&m2(&int(8), &a)?, &int(k - 1))?, &previous)?, &scale)?;
            return Ok(Some(s2(&first, &second)?));
        }
        let Some(previous) = self.power(m - 2)? else {
            return Ok(None);
        };
        // (2*a*x + b) * Q**(m/2) / (2 * a * (m + 1)) - m * disc * previous / (4 * a * (m + 1))
        let first: Basic = d2(
            &m2(&self.linear()?, &pw(&q, &half(m))?)?,
            &m2(&m2(&int(2), &a)?, &int(m + 1))?,
        )?;
        let second: Basic = d2(
            &m2(&m2(&int(m), &disc)?, &previous)?,
            &m2(&m2(&int(4), &a)?, &int(m + 1))?,
        )?;
        Ok(Some(s2(&first, &second)?))
    }

    /// The antiderivative of `x^n Q^{m/2}` (`moment`).
    fn moment(&mut self, n: i64, m: i64) -> OptR {
        if let Some(v) = self.memo.get(&(n, m)) {
            return Ok(v.clone());
        }
        self.memo.insert((n, m), None);
        let value: Option<Basic> = self.moment_(n, m)?;
        self.memo.insert((n, m), value.clone());
        Ok(value)
    }

    fn moment_(&mut self, n: i64, m: i64) -> OptR {
        let (a, b, c, x, q) = (
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.x.clone(),
            self.q.clone(),
        );
        if n == 0 {
            return self.power(m);
        }
        if n < 0 {
            return self.negative_(n, m);
        }
        if a == int(0) {
            if b == int(0) {
                return Ok(None);
            }
            let (Some(higher), Some(lower)) = (self.moment(n - 1, m + 2)?, self.moment(n - 1, m)?) else {
                return Ok(None);
            };
            return Ok(Some(d2(&s2(&higher, &m2(&c, &lower)?)?, &b)?));
        }
        if n == 1 {
            let Some(lower) = self.moment(0, m)? else {
                return Ok(None);
            };
            // Q**(m/2 + 1) / (a * (m + 2)) - b * lower / (2 * a)
            let first: Basic = d2(&pw(&q, &a2(&half(m), &int(1))?)?, &m2(&a, &int(m + 2))?)?;
            let second: Basic = d2(&m2(&b, &lower)?, &m2(&int(2), &a)?)?;
            return Ok(Some(s2(&first, &second)?));
        }
        let leading: Basic = m2(&a, &int(n + m + 1))?;
        if leading != int(0) {
            let (Some(first), Some(second)) = (self.moment(n - 1, m)?, self.moment(n - 2, m)?) else {
                return Ok(None);
            };
            // (x**(n-1) * Q**(m/2+1) - b * (n + m/2) * first - c * (n - 1) * second) / leading
            let t1: Basic = m2(&pw(&x, &int(n - 1))?, &pw(&q, &a2(&half(m), &int(1))?)?)?;
            let t2: Basic = m2(&m2(&b, &a2(&int(n), &half(m))?)?, &first)?;
            let t3: Basic = m2(&m2(&c, &int(n - 1))?, &second)?;
            return Ok(Some(d2(&s2(&s2(&t1, &t2)?, &t3)?, &leading)?));
        }
        let (Some(higher), Some(first), Some(second)) = (
            self.moment(n - 2, m + 2)?,
            self.moment(n - 1, m)?,
            self.moment(n - 2, m)?,
        ) else {
            return Ok(None);
        };
        Ok(Some(d2(&s2(&s2(&higher, &m2(&b, &first)?)?, &m2(&c, &second)?)?, &a)?))
    }

    fn negative_(&mut self, n: i64, m: i64) -> OptR {
        let (a, b, c, x, q) = (
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.x.clone(),
            self.q.clone(),
        );
        let disc: Basic = self.discriminant.clone();
        if c == int(0) {
            return Ok(None);
        }
        if n == -1 {
            if m == -1 {
                if self.positive(&c)? {
                    // -log((2 * sqrt(c) * sqrt(Q) + b * x + 2 * c) / x) / sqrt(c)
                    let inner: Basic = a2(
                        &a2(&m2(&m2(&int(2), &sqrt(&c)?)?, &sqrt(&q)?)?, &m2(&b, &x)?)?,
                        &m2(&int(2), &c)?,
                    )?;
                    return Ok(Some(d2(&neg(&log(&d2(&inner, &x)?)?)?, &sqrt(&c)?)?));
                }
                if self.negative(&c)? && self.positive(&disc)? {
                    // asin((2 * c + b * x) / (sqrt(x**2) * sqrt(disc))) / sqrt(-c)
                    let top: Basic = a2(&m2(&int(2), &c)?, &m2(&b, &x)?)?;
                    let bottom: Basic = m2(&sqrt(&pw(&x, &int(2))?)?, &sqrt(&disc)?)?;
                    let asin: Basic = crate::functions::elementary::inverse::asin(&d2(&top, &bottom)?)?;
                    return Ok(Some(d2(&asin, &sqrt(&neg(&c)?)?)?));
                }
                return Ok(None);
            }
            if m > 0 {
                let (Some(first), Some(second), Some(third)) =
                    (self.moment(1, m - 2)?, self.moment(0, m - 2)?, self.moment(-1, m - 2)?)
                else {
                    return Ok(None);
                };
                return Ok(Some(a2(&a2(&m2(&a, &first)?, &m2(&b, &second)?)?, &m2(&c, &third)?)?));
            }
            let (Some(higher), Some(first), Some(second)) =
                (self.moment(-1, m + 2)?, self.moment(1, m)?, self.moment(0, m)?)
            else {
                return Ok(None);
            };
            return Ok(Some(d2(&s2(&s2(&higher, &m2(&a, &first)?)?, &m2(&b, &second)?)?, &c)?));
        }
        let (Some(first), Some(second)) = (self.moment(n + 1, m)?, self.moment(n + 2, m)?) else {
            return Ok(None);
        };
        // (x**(n+1) * Q**(m/2+1) - b * (n + 2 + m/2) * first - a * (n + m + 3) * second) / (c * (n + 1))
        let t1: Basic = m2(&pw(&x, &int(n + 1))?, &pw(&q, &a2(&half(m), &int(1))?)?)?;
        let t2: Basic = m2(&m2(&b, &a2(&int(n + 2), &half(m))?)?, &first)?;
        let t3: Basic = m2(&m2(&a, &int(n + m + 3))?, &second)?;
        Ok(Some(d2(&s2(&s2(&t1, &t2)?, &t3)?, &m2(&c, &int(n + 1))?)?))
    }
}

/// `base` as `root**k` for an odd `k > 1` when it is one (`_odd_power`).
fn odd_power(base: &Basic, x: &Basic) -> PolyResult<(Basic, i64)> {
    let poly = match crate::polys::polytools::Poly::new(base, std::slice::from_ref(x)) {
        Ok(p) => p,
        Err(PolyError::Polys(..)) => return Ok((base.clone(), 1)),
        Err(e) => return Err(e),
    };
    if poly.degree(0)? < 2 {
        return Ok((base.clone(), 1));
    }
    let (content, factors) = poly.sqf_list()?;
    if factors.len() != 1 {
        return Ok((base.clone(), 1));
    }
    let (root, k) = &factors[0];
    let k: i64 = *k as i64;
    if k % 2 == 0 || k == 1 || content != int(1) {
        return Ok((base.clone(), 1));
    }
    Ok((root.as_expr()?, k))
}

/// A radical term `(coefficient, n, radicand, m)` (`RadicalTerm`).
type RadicalTerm = (Basic, i64, Option<Basic>, i64);

fn is_poly(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}

fn small(e: &Basic) -> i64 {
    i64::try_from(&crate::numbers::numer(e.as_rational().expect("a rational"))).expect("a small number")
}

/// The terms `c x^n Q^{m/2}` of a product (`_terms`).
fn terms(term: &Basic, x: &Basic) -> PolyResult<Option<Vec<RadicalTerm>>> {
    let mut coefficient: Basic = int(1);
    let mut polynomial: Basic = int(1);
    let mut base: Option<Basic> = None;
    let mut m: i64 = 0;
    let mut shift: i64 = 0;
    let mut radicals: Vec<(Basic, i64)> = Vec::new();
    for f in crate::core::expr::factors_of(term) {
        if !f.has(x) {
            coefficient = m2(&coefficient, &f)?;
            continue;
        }
        if f.is_pow()
            && f.args()[1].is_rational_class()
            && crate::numbers::denom(f.args()[1].as_rational().expect("rational")) == Int::from(2u8)
        {
            let (radicand, power) = odd_power(&f.args()[0], x)?;
            radicals.push((radicand, small(&f.args()[1]) * power));
            continue;
        }
        if f.is_pow() && f.args()[0] == *x && f.args()[1].is_integer_value() {
            shift += small(&f.args()[1]);
            continue;
        }
        if f.is_pow() && f.args()[1].is_integer_value() && small(&f.args()[1]) < 0 && is_poly(&f.args()[0], x) {
            radicals.push((f.args()[0].clone(), 2 * small(&f.args()[1])));
            continue;
        }
        if is_poly(&f, x) {
            polynomial = m2(&polynomial, &f)?;
            continue;
        }
        return Ok(None);
    }
    if radicals.len() > 1 && radicals.iter().all(|(r, _)| *r == radicals[0].0) {
        let total: i64 = radicals.iter().map(|(_, p)| p).sum();
        radicals = vec![(radicals[0].0.clone(), total)];
    }
    if radicals.len() == 1 {
        base = Some(radicals[0].0.clone());
        m = radicals[0].1;
        if m % 2 == 0 {
            return Ok(None);
        }
    } else if radicals.len() > 1 {
        for (r, p) in &radicals {
            let linear = coefficients(r, x)?;
            if *p < 1 || linear.as_ref().map_or(true, |l| degree(l) != 1) {
                return Ok(None);
            }
        }
        let mut product: Basic = int(1);
        for (r, p) in &radicals {
            polynomial = m2(&polynomial, &pw(r, &int((p - 1) / 2))?)?;
            product = m2(&product, r)?;
        }
        base = Some(crate::core::function::expand_default(&product)?);
        m = 1;
    }
    let Some(c) = coefficients(&polynomial, x)? else {
        return Ok(None);
    };
    let mut ns: Vec<i64> = c.iter().map(|(n, _)| *n).collect();
    ns.sort_unstable_by(|p, q| q.cmp(p));
    let mut found: Vec<RadicalTerm> = Vec::new();
    for n in ns {
        found.push((m2(&coefficient, &coefficient_of(&c, n))?, n + shift, base.clone(), m));
    }
    Ok(Some(found))
}

fn coefficient_of(c: &[(i64, Basic)], n: i64) -> Basic {
    coefficient(c, n)
}

fn with_assumptions(facts: Vec<Basic>, assumptions: &[Basic]) -> Vec<Basic> {
    let mut out: Vec<Basic> = facts;
    out.extend(assumptions.iter().cloned());
    out
}

/// A real antiderivative of a sum of terms `c x^n Q^{m/2}`, by the
/// reduction formulas, else by Euler's substitutions when `euler`
/// (`quadratic_radical_antiderivative`); `None` when it is not found.
pub fn quadratic_radical_antiderivative(f: &Basic, x: &Basic, assumptions: &[Basic], euler: bool) -> OptR {
    let found: Option<Basic> = table_antiderivative(f, x, assumptions)?;
    if found.is_some() || !euler {
        return Ok(found);
    }
    euler_substitution_antiderivative(f, x, assumptions)
}

fn table_antiderivative(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    let mut all: Vec<RadicalTerm> = Vec::new();
    for summand in crate::core::expr::terms_of(f) {
        let Some(found) = terms(&summand, x)? else {
            return Ok(None);
        };
        all.extend(found);
    }
    let mut tables: Vec<(Basic, Table)> = Vec::new();
    for (_, _, radicand, _) in &all {
        let Some(radicand) = radicand else { continue };
        if tables.iter().any(|(r, _)| r == radicand) {
            continue;
        }
        let Some((a, b, c)) = quadratic_coefficients(radicand, x)? else {
            return Ok(None);
        };
        tables.push((radicand.clone(), Table::new(a, b, c, x.clone(), assumptions)?));
    }
    let mut total: Basic = int(0);
    for (coefficient, n, radicand, m) in &all {
        let Some(radicand) = radicand else {
            let term: Basic = if *n == -1 {
                m2(coefficient, &log(x)?)?
            } else {
                d2(&m2(coefficient, &pw(x, &int(n + 1))?)?, &int(n + 1))?
            };
            total = a2(&total, &term)?;
            continue;
        };
        if m % 2 == 0 {
            return Ok(None);
        }
        let table: &mut Table = &mut tables.iter_mut().find(|(r, _)| r == radicand).expect("a table").1;
        let Some(value) = table.moment(*n, *m)? else {
            return Ok(None);
        };
        total = a2(&total, &m2(coefficient, &value)?)?;
    }
    let mut facts: Vec<Basic> = Vec::new();
    for (_, t) in &tables {
        facts.push(crate::core::relational::gt(&t.q, &int(0))?);
    }
    let facts: Vec<Basic> = with_assumptions(facts, assumptions);
    let derivative: Basic = crate::core::function::derivative(&total, x)?;
    if !crate::integrals::conditions::numerically_equal(&derivative, f, &facts, 3)? {
        return Ok(None);
    }
    Ok(Some(total))
}

/// The radicand `Q` and `f` as a rational function of `x` and `R =
/// sqrt(Q)` (`_radicand`).
fn radicand(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    use crate::core::symbol::{Dummy, Symbol};
    let r: Basic = Dummy::new("R", &[(Predicate::positive, true)]).0;
    let mut radicands: Vec<Basic> = Vec::new();
    let mut degrees: Vec<(Basic, i64)> = Vec::new();
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    let mut powers: Vec<Basic> = Vec::new();
    for node in f.preorder() {
        if node.is_pow() && !powers.contains(&node) {
            powers.push(node);
        }
    }
    for node in &powers {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if !base.has(x) || !exponent.is_rational_class() || exponent.is_integer_value() {
            continue;
        }
        let q_den = crate::numbers::denom(exponent.as_rational().expect("rational"));
        if q_den != Int::from(2u8) || !is_poly(base, x) {
            return Ok(None);
        }
        let (base, power) = odd_power(base, x)?;
        let p_: i64 = small(exponent) * power;
        let Some(c) = coefficients(&base, x)? else {
            return Ok(None);
        };
        let d: i64 = degree(&c);
        if !matches!(d, 1 | 2) {
            return Ok(None);
        }
        if !radicands.contains(&base) {
            radicands.push(base.clone());
            degrees.push((base.clone(), d));
        }
        let rem: i64 = p_.rem_euclid(2);
        replacement.insert(
            node.clone(),
            m2(&pw(&base, &int((p_ - rem) / 2))?, &pw(&r, &int(rem))?)?,
        );
    }
    let symbol_r: Basic = Symbol::new("R", &[]).0;
    let rational = |g: &Basic, gens: &[Basic]| {
        crate::core::expr::polynomial_like(g, gens, crate::core::expr::PolynomialKind::Rational) == Some(true)
    };
    let degree_of = |b: &Basic| degrees.iter().find(|(k, _)| k == b).map(|(_, d)| *d).unwrap_or(0);
    if radicands.len() == 2 && radicands.iter().all(|b| degree_of(b) == 1) {
        let q: Basic = crate::core::function::expand_default(&m2(&radicands[0], &radicands[1])?)?;
        let first: Basic = radicands[0].clone();
        let s: Basic = Dummy::new("S", &[(Predicate::positive, true)]).0;
        let t: Basic = Dummy::new("T", &[(Predicate::positive, true)]).0;
        let root_of = |b: &Basic| if *b == first { s.clone() } else { t.clone() };
        for node in &powers {
            if !replacement.contains_key(node) {
                continue;
            }
            let (base, exponent) = (&node.args()[0], &node.args()[1]);
            if !exponent.is_rational_class() {
                return Ok(None);
            }
            let (base, power) = odd_power(base, x)?;
            let p_: i64 = small(exponent) * power;
            let rem: i64 = p_.rem_euclid(2);
            replacement.insert(
                node.clone(),
                m2(&pw(&base, &int((p_ - rem) / 2))?, &pw(&root_of(&base), &int(rem))?)?,
            );
        }
        let mut g: Basic = f.xreplace(&replacement);
        g = crate::core::subs::subs(&g, &t, &d2(&r, &s)?)?;
        g = crate::core::subs::subs(&g, &pw(&s, &int(2))?, &first)?;
        g = crate::core::subs::subs(&g, &pw(&s, &int(-2))?, &d2(&int(1), &first)?)?;
        if g.has(&s) || !rational(&g, &[x.clone(), r.clone()]) {
            return Ok(None);
        }
        return Ok(Some((q, crate::core::subs::subs(&g, &r, &symbol_r)?)));
    }
    if radicands.len() != 1 {
        return Ok(None);
    }
    let q: Basic = radicands[0].clone();
    let g: Basic = f.xreplace(&replacement);
    if !rational(&g, &[x.clone(), r.clone()]) {
        return Ok(None);
    }
    Ok(Some((q, crate::core::subs::subs(&g, &r, &symbol_r)?)))
}

/// An antiderivative of a rational function of `x` and `sqrt(Q)` by an
/// Euler substitution (`euler_substitution_antiderivative`).
pub fn euler_substitution_antiderivative(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptR {
    use crate::core::symbol::{Dummy, Symbol};
    let Some((q, g)) = radicand(f, x)? else { return Ok(None) };
    let r: Basic = Symbol::new("R", &[]).0;
    let Some((a, b, c)) = quadratic_coefficients(&q, x)? else {
        return Ok(None);
    };
    let t: Basic = Dummy::new("t", &[(Predicate::positive, true)]).0;
    let positive = |e: &Basic| -> PolyResult<bool> {
        Ok(is(e, Predicate::positive) == Some(true)
            || ask(&crate::core::relational::gt(e, &int(0))?, assumptions, None)? == Some(true))
    };
    let t2: Basic = pw(&t, &int(2))?;
    let mut candidates: Vec<(Basic, Basic, Basic)> = Vec::new();
    if a == int(0) {
        if b == int(0) {
            return Ok(None);
        }
        candidates.push((d2(&s2(&t2, &c)?, &b)?, t.clone(), sqrt(&q)?));
    } else {
        if positive(&a)? {
            // (t**2 - c) / (2 * sqrt(a) * t + b)
            let x_t: Basic = d2(&s2(&t2, &c)?, &a2(&m2(&m2(&int(2), &sqrt(&a)?)?, &t)?, &b)?)?;
            let root_t: Basic = s2(&t, &m2(&sqrt(&a)?, &x_t)?)?;
            let back: Basic = a2(&sqrt(&q)?, &m2(&sqrt(&a)?, x)?)?;
            candidates.push((x_t, root_t, back));
        }
        let mut real_roots: Vec<Basic> = Vec::new();
        for root in roots_of(&q, x)? {
            let real: Option<bool> = is(&root, Predicate::extended_real);
            if real == Some(true)
                || (real.is_none()
                    && ask(
                        &crate::assumptions::facts::element(&root, &crate::sets::sets::reals())?,
                        assumptions,
                        None,
                    )? == Some(true))
            {
                real_roots.push(root);
            }
        }
        if real_roots.len() == 2 {
            let (r1, r2) = (real_roots[0].clone(), real_roots[1].clone());
            // (a * r2 - t**2 * r1) / (a - t**2)
            let x_t: Basic = d2(&s2(&m2(&a, &r2)?, &m2(&t2, &r1)?)?, &s2(&a, &t2)?)?;
            let root_t: Basic = m2(&t, &s2(&x_t, &r1)?)?;
            let back: Basic = d2(&sqrt(&q)?, &s2(x, &r1)?)?;
            candidates.push((x_t, root_t, back));
        }
        if positive(&c)? {
            // (2 * t * sqrt(c) - b) / (a - t**2)
            let x_t: Basic = d2(&s2(&m2(&m2(&int(2), &t)?, &sqrt(&c)?)?, &b)?, &s2(&a, &t2)?)?;
            let root_t: Basic = a2(&m2(&x_t, &t)?, &sqrt(&c)?)?;
            let back: Basic = d2(&s2(&sqrt(&q)?, &sqrt(&c)?)?, x)?;
            candidates.push((x_t, root_t, back));
        }
    }
    let facts: Vec<Basic> = with_assumptions(vec![crate::core::relational::gt(&q, &int(0))?], assumptions);
    let symbol_t: Basic = Symbol::new("t", &[]).0;
    for (x_t, root_t, back) in candidates {
        let substituted: Basic = crate::core::subs::subs_pairs(&g, &[(x.clone(), x_t.clone()), (r.clone(), root_t)])?;
        let h: Basic =
            crate::polys::polytools::cancel(&m2(&substituted, &crate::core::function::derivative(&x_t, &t)?)?, &[])?;
        if crate::core::expr::polynomial_like(
            &h,
            std::slice::from_ref(&t),
            crate::core::expr::PolynomialKind::Rational,
        ) != Some(true)
        {
            continue;
        }
        let primitive: Option<Basic> = match crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            let integrand = crate::integrals::risch::rationaltools::Integrand::of(&h)?;
            crate::integrals::risch::rationaltools::ratint(&integrand, &t, &symbol_t, None)
        }) {
            Ok(p) => p,
            Err(PolyError::Index(_) | PolyError::Value(_) | PolyError::ZeroDivision(_)) => continue,
            Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => continue,
            Err(e) => return Err(e),
        };
        let Some(primitive) = primitive else { continue };
        let result: Basic = crate::core::subs::subs(&primitive, &t, &back)?;
        let derivative: Basic = crate::core::function::derivative(&result, x)?;
        if crate::integrals::conditions::numerically_equal(&derivative, f, &facts, 3)? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

fn roots_of(q: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    if let Err(PolyError::Polys(..)) = crate::polys::polytools::Poly::new(q, std::slice::from_ref(x)) {
        return Ok(vec![]);
    }
    let found = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::polys::polyroots::roots(q, std::slice::from_ref(x), true, true, None)
    })?;
    Ok(found
        .map(|r| r.iter().map(|(k, _)| k.clone()).collect())
        .unwrap_or_default())
}
