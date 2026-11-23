//! Port of `sympy_extras/integrals/algebraic.py`: the rationalising
//! substitutions of algebraic integrands (Euler's substitutions, the
//! roots of a Möbius function), the binomial differentials
//! `x**m*(a + b*x**n)**p` (Chebyshev's cases) and their antiderivatives by
//! `2F1` where they are not elementary, and the definite integrals through
//! them (`rationalised_integral`, `algebraic_integral`, by the definite
//! integrator of `integrals/definite`).

#![allow(clippy::too_many_arguments, clippy::mutable_key_type)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A dummy with a name of its own (`_fresh`).
fn fresh(name: &str, facts: &[(Predicate, bool)]) -> Basic {
    let k: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
    crate::core::symbol::Dummy::new(&format!("{}{}", name, k), facts).0
}

fn int(n: i64) -> Basic {
    Basic::int(n)
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
/// A product built two factors at a time from the left, as Python's `*`.
fn chain(v: &[Basic]) -> R {
    let mut acc: Basic = v[0].clone();
    for f in &v[1..] {
        acc = m2(&acc, f)?;
    }
    Ok(acc)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn rat(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}
fn rational_of(e: &Basic) -> Option<Rat> {
    match e.head() {
        Head::Rational { .. } => e.as_rational().cloned(),
        _ => None,
    }
}
fn is_integer_node(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn denominator(r: &Rat) -> Int {
    crate::numbers::denom(r)
}
fn numerator(r: &Rat) -> Int {
    crate::numbers::numer(r)
}
fn is_whole(r: &Rat) -> bool {
    denominator(r) == Int::ONE
}
fn to_i64(n: &Int) -> PolyResult<i64> {
    i64::try_from(n).map_err(|_| PolyError::Value("an exponent too large".to_string()))
}

/// A rationalising substitution `x = X(t)`: the variable, the map, its
/// inverse and the radicals as rational functions of `t`
/// (`Rationalisation`).
#[derive(Clone, Debug)]
pub struct Rationalisation {
    pub t: Basic,
    pub x_of_t: Basic,
    pub t_of_x: Basic,
    pub radicals: Vec<(Basic, Basic)>,
}

fn coefficients(e: &Basic, x: &Basic) -> PolyResult<Option<BTreeMap<i64, Basic>>> {
    if crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return Ok(None);
    }
    let mut found: BTreeMap<i64, Basic> = BTreeMap::new();
    for term in crate::core::expr::terms_of(&crate::core::function::expand_default(e)?) {
        let (coefficient, rest) = crate::core::expr::as_independent(&term, x, false)?;
        let n: i64 = if rest == Basic::one() {
            0
        } else if rest == *x {
            1
        } else if rest.is_pow() && rest.args()[0] == *x && is_integer_node(&rest.args()[1]) {
            match rational_of(&rest.args()[1]) {
                Some(r) if r > Rat::ZERO => to_i64(&numerator(&r))?,
                _ => return Ok(None),
            }
        } else {
            return Ok(None);
        };
        let value: Basic = match found.get(&n) {
            Some(c) => a2(c, &coefficient)?,
            None => coefficient,
        };
        found.insert(n, value);
    }
    Ok(Some(found.into_iter().filter(|(_, c)| *c != Basic::zero()).collect()))
}

fn polynomial(e: &Basic, x: &Basic) -> PolyResult<Option<BTreeMap<i64, Basic>>> {
    Ok(coefficients(e, x)?.filter(|f| !f.is_empty()))
}

fn degree(c: &BTreeMap<i64, Basic>) -> i64 {
    c.keys().next_back().copied().unwrap_or(-1)
}

fn coefficient(c: &BTreeMap<i64, Basic>, n: i64) -> Basic {
    c.get(&n).cloned().unwrap_or_else(Basic::zero)
}

fn radical_atoms(f: &Basic, x: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for node in f.preorder() {
        if node.is_pow()
            && node.has(x)
            && matches!(
                node.args()[1].head(),
                Head::Rational {
                    integer_class: false,
                    ..
                }
            )
            && !found.contains(&node)
        {
            found.push(node);
        }
    }
    found
}

fn exponent(atom: &Basic) -> Rat {
    rational_of(&atom.args()[1]).expect("a rational exponent")
}

fn real_roots(a2_: &Basic, b2: &Basic, c2: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let (a, b, c) = (rational_of(a2_), rational_of(b2), rational_of(c2));
    let (Some(a), Some(b), Some(c)) = (a, b, c) else {
        return Err(PolyError::Type("expected a rational number".to_string()));
    };
    let discriminant: Rat = &b * &b - Rat::from(Int::from(4)) * &a * &c;
    if discriminant <= Rat::ZERO {
        return Ok(None);
    }
    let y: Basic = fresh("y", &[]);
    let quadratic: Basic = crate::core::add::add(&[m2(a2_, &pw(&y, &int(2))?)?, m2(b2, &y)?, c2.clone()])?;
    let found: Vec<Basic> = crate::polys::polyroots::roots_quadratic(&crate::polys::polytools::Poly::new(
        &quadratic,
        std::slice::from_ref(&y),
    )?)?;
    if found.len() != 2 {
        return Ok(None);
    }
    let (mut r1, mut r2) = (found[0].clone(), found[1].clone());
    if yes(&s2(&r1, &r2)?, Predicate::extended_positive) {
        std::mem::swap(&mut r1, &mut r2);
    }
    Ok(Some((r1, r2)))
}

fn ask_positive(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    let query: Basic = crate::core::relational::gt(e, &Basic::zero())?;
    Ok(crate::assumptions::ask::ask(&query, assumptions, None)? == Some(true))
}

fn sqrt(e: &Basic) -> R {
    pw(e, &d2(&int(1), &int(2))?)
}

/// Euler's substitutions applicable to an integrand whose radicals are
/// half-integer powers of one polynomial of degree one or two
/// (`euler_substitutions`), in the order `a > 0`, `c > 0`, real roots.
pub fn euler_substitutions(f: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Vec<Rationalisation>> {
    let atoms: Vec<Basic> = radical_atoms(f, x);
    if atoms.is_empty() || atoms.iter().any(|a| denominator(&exponent(a)) != Int::from(2)) {
        return Ok(Vec::new());
    }
    let mut bases: Vec<BTreeMap<i64, Basic>> = Vec::new();
    for atom in &atoms {
        let Some(p) = polynomial(&atom.args()[0], x)? else {
            return Ok(Vec::new());
        };
        if degree(&p) > 2 {
            return Ok(Vec::new());
        }
        bases.push(p);
    }
    let q0: Basic = atoms[0].args()[0].clone();
    let mut ratios: Vec<Basic> = Vec::new();
    for atom in &atoms {
        let ratio: Basic = cancel(&d2(&atom.args()[0], &q0)?)?;
        if ratio.has(x) || !ask_positive(&ratio, assumptions)? {
            return Ok(Vec::new());
        }
        ratios.push(ratio);
    }
    let q: Basic = crate::core::function::expand_default(&q0)?;
    let u: Basic = fresh("u", &[]);
    let c = &bases[0];
    let u2: Basic = pw(&u, &int(2))?;
    let mut maps: Vec<(Basic, Basic, Basic)> = Vec::new();
    if degree(c) == 1 {
        let (b1, c1) = (coefficient(c, 1), coefficient(c, 0));
        maps.push((d2(&s2(&u2, &c1)?, &b1)?, sqrt(&q)?, u.clone()));
    } else {
        let (a2_, b2, c2) = (coefficient(c, 2), coefficient(c, 1), coefficient(c, 0));
        if ask_positive(&a2_, assumptions)? {
            let sa: Basic = sqrt(&a2_)?;
            let x_of_u: Basic = d2(&s2(&u2, &c2)?, &s2(&b2, &chain(&[int(2), sa.clone(), u.clone()])?)?)?;
            let u_of_x: Basic = s2(&sqrt(&q)?, &m2(&sa, x)?)?;
            maps.push((x_of_u.clone(), u_of_x, a2(&m2(&sa, &x_of_u)?, &u)?));
        }
        if ask_positive(&c2, assumptions)? {
            let sc: Basic = sqrt(&c2)?;
            let x_of_u: Basic = d2(&s2(&chain(&[int(2), sc.clone(), u.clone()])?, &b2)?, &s2(&a2_, &u2)?)?;
            let u_of_x: Basic = d2(&s2(&sqrt(&q)?, &sc)?, x)?;
            maps.push((x_of_u.clone(), u_of_x, a2(&m2(&x_of_u, &u)?, &sc)?));
        }
        if [&a2_, &b2, &c2].iter().all(|v| rational_of(v).is_some()) {
            if let Some((r1, r2)) = real_roots(&a2_, &b2, &c2)? {
                let x_of_u: Basic = d2(&s2(&m2(&a2_, &r2)?, &m2(&r1, &u2)?)?, &s2(&a2_, &u2)?)?;
                let u_of_x: Basic = d2(&sqrt(&q)?, &s2(x, &r1)?)?;
                maps.push((x_of_u.clone(), u_of_x, m2(&s2(&x_of_u, &r1)?, &u)?));
            }
        }
    }
    let mut substitutions: Vec<Rationalisation> = Vec::new();
    for (x_of_u, u_of_x, y) in maps {
        let y_: Basic = cancel(&y)?;
        let mut radicals: Vec<(Basic, Basic)> = Vec::new();
        for (atom, ratio) in atoms.iter().zip(&ratios) {
            let e: Rat = exponent(atom);
            radicals.push((
                atom.clone(),
                m2(
                    &pw(ratio, &rat(&e))?,
                    &pw(&y_, &Basic::from_rational(Rat::from(numerator(&e))))?,
                )?,
            ));
        }
        substitutions.push(Rationalisation {
            t: u.clone(),
            x_of_t: cancel(&x_of_u)?,
            t_of_x: u_of_x,
            radicals,
        });
    }
    Ok(substitutions)
}

/// The first of [`euler_substitutions`] (`euler_substitution`).
pub fn euler_substitution(f: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Rationalisation>> {
    Ok(euler_substitutions(f, x, assumptions)?.into_iter().next())
}

fn lcm(a: i64, b: i64) -> i64 {
    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 {
            a.abs()
        } else {
            gcd(b, a % b)
        }
    }
    (a / gcd(a, b) * b).abs()
}

/// The substitution `t**n = M(x)` for an integrand whose radicals are
/// rational powers of one Möbius (or linear) function of `x` or of its
/// reciprocal (`moebius_root`).
pub fn moebius_root(f: &Basic, x: &Basic) -> PolyResult<Option<Rationalisation>> {
    let atoms: Vec<Basic> = radical_atoms(f, x);
    if atoms.is_empty() {
        return Ok(None);
    }
    let base0: Basic = atoms[0].args()[0].clone();
    let (numer, denom) = crate::core::expr::as_numer_denom(&cancel(&base0)?)?;
    let (Some(pn), Some(pd)) = (polynomial(&numer, x)?, polynomial(&denom, x)?) else {
        return Ok(None);
    };
    if degree(&pn) > 1 || degree(&pd) > 1 {
        return Ok(None);
    }
    let mut n: i64 = 1;
    let mut powers: Vec<(Basic, Rat)> = Vec::new();
    for atom in &atoms {
        let base: &Basic = &atom.args()[0];
        let sign: i64 = if cancel(&d2(base, &base0)?)? == Basic::one() {
            1
        } else if cancel(&m2(base, &base0)?)? == Basic::one() {
            -1
        } else {
            return Ok(None);
        };
        let e: Rat = exponent(atom);
        powers.push((atom.clone(), Rat::from(Int::from(sign)) * &e));
        n = lcm(n, to_i64(&denominator(&e))?);
    }
    let t: Basic = fresh("t", &[(Predicate::positive, true)]);
    let (a1, b1) = (coefficient(&pn, 1), coefficient(&pn, 0));
    let (c1, d1) = (coefficient(&pd, 1), coefficient(&pd, 0));
    let tn: Basic = pw(&t, &int(n))?;
    let below: Basic = s2(&a1, &m2(&c1, &tn)?)?;
    if below == Basic::zero() {
        return Ok(None);
    }
    let x_of_t: Basic = cancel(&d2(&s2(&m2(&d1, &tn)?, &b1)?, &below)?)?;
    if !x_of_t.has(&t) {
        return Ok(None);
    }
    let mut radicals: Vec<(Basic, Basic)> = Vec::new();
    for (atom, k) in powers {
        let power: Rat = k * Rat::from(Int::from(n));
        radicals.push((
            atom,
            pw(
                &t,
                &Basic::from_rational(Rat::from(numerator(&power) / denominator(&power))),
            )?,
        ));
    }
    let t_of_x: Basic = pw(&base0, &d2(&int(1), &int(n))?)?;
    Ok(Some(Rationalisation {
        t,
        x_of_t,
        t_of_x,
        radicals,
    }))
}

fn monomial_exponent(monomial: &Basic, x: &Basic) -> Option<Basic> {
    if monomial == x {
        return Some(Basic::one());
    }
    if monomial.is_pow() && monomial.args()[0] == *x && rational_of(&monomial.args()[1]).is_some() {
        return Some(monomial.args()[1].clone());
    }
    None
}

/// `(C, m, a, b, n, p)` with `f == C*x**m*(a + b*x**n)**p` and the binomial
/// power not an integer power of a polynomial (`binomial_differential`).
pub fn binomial_differential(f: &Basic, x: &Basic) -> PolyResult<Option<[Basic; 6]>> {
    let (c, rest) = crate::core::expr::as_independent(f, x, false)?;
    let mut m: Basic = Basic::zero();
    let mut binomial: Option<(Basic, Basic, Basic, Basic)> = None;
    for factor in crate::core::expr::factors_of(&rest) {
        if factor == *x {
            m = a2(&m, &Basic::one())?;
            continue;
        }
        if factor.is_pow() && factor.args()[0] == *x && rational_of(&factor.args()[1]).is_some() {
            m = a2(&m, &factor.args()[1])?;
            continue;
        }
        if factor.is_pow()
            && rational_of(&factor.args()[1]).is_some()
            && factor.args()[0].is_add()
            && binomial.is_none()
        {
            let terms: &[Basic] = factor.args()[0].args();
            if terms.len() != 2 {
                return Ok(None);
            }
            let constant: Vec<&Basic> = terms.iter().filter(|t| !t.has(x)).collect();
            if constant.len() != 1 {
                return Ok(None);
            }
            let a_: Basic = constant[0].clone();
            let other: &Basic = if terms[1].ptr_eq(constant[0]) {
                &terms[0]
            } else {
                &terms[1]
            };
            let (coefficient_, monomial) = crate::core::expr::as_independent(other, x, false)?;
            let Some(power) = monomial_exponent(&monomial, x) else {
                return Ok(None);
            };
            binomial = Some((a_, coefficient_, power, factor.args()[1].clone()));
            continue;
        }
        return Ok(None);
    }
    let Some((a0, b0, n0, p0)) = binomial else {
        return Ok(None);
    };
    if is_integer_node(&p0) && is_integer_node(&m) && is_integer_node(&n0) {
        return Ok(None);
    }
    Ok(Some([c, m, a0, b0, n0, p0]))
}

/// `x**m*(a + b*x**n)**p` with `m`, `n`, `p` rational
/// (`BinomialDifferential`).
#[derive(Clone, Debug)]
pub struct BinomialDifferential {
    pub m: Rat,
    pub a: Basic,
    pub b: Basic,
    pub n: Rat,
    pub p: Rat,
}

type BinomialPower = (Basic, Basic, Rat, Rat);

fn binomial_powers(e: &Basic, x: &Basic) -> PolyResult<Option<(Rat, Vec<BinomialPower>)>> {
    if !e.has(x) {
        return Ok(Some((Rat::ZERO, Vec::new())));
    }
    if e == x {
        return Ok(Some((Rat::ONE, Vec::new())));
    }
    if e.is_pow() {
        let Some(exponent_) = rational_of(&e.args()[1]) else {
            return Ok(None);
        };
        let Some((m, powers)) = binomial_powers(&e.args()[0], x)? else {
            return Ok(None);
        };
        return Ok(Some((
            m * &exponent_,
            powers
                .into_iter()
                .map(|(a, b, n, k)| (a, b, n, k * &exponent_))
                .collect(),
        )));
    }
    if e.is_mul() {
        let mut total: Rat = Rat::ZERO;
        let mut collected: Vec<BinomialPower> = Vec::new();
        for factor in e.args() {
            let Some((m, powers)) = binomial_powers(factor, x)? else {
                return Ok(None);
            };
            total += m;
            collected.extend(powers);
        }
        return Ok(Some((total, collected)));
    }
    if e.is_add() {
        let mut terms: Vec<(Rat, Basic)> = Vec::new();
        for term in e.args() {
            let (c, monomial) = crate::core::expr::as_independent(term, x, false)?;
            let k: Rat = if monomial == Basic::one() {
                Rat::ZERO
            } else {
                let Some(power) = monomial_exponent(&monomial, x) else {
                    return Ok(None);
                };
                rational_of(&power).expect("a rational")
            };
            match terms.iter_mut().find(|(kk, _)| *kk == k) {
                Some(entry) => entry.1 = a2(&entry.1, &c)?,
                None => terms.push((k, c)),
            }
        }
        let mut present: Vec<Rat> = terms
            .iter()
            .filter(|(_, c)| *c != Basic::zero())
            .map(|(k, _)| k.clone())
            .collect();
        present.sort();
        if present.len() == 1 {
            return Ok(Some((present[0].clone(), Vec::new())));
        }
        if present.len() != 2 {
            return Ok(None);
        }
        let (k1, k2) = (present[0].clone(), present[1].clone());
        let get = |k: &Rat| {
            terms
                .iter()
                .find(|(kk, _)| kk == k)
                .map(|(_, c)| c.clone())
                .expect("a term")
        };
        return Ok(Some((k1.clone(), vec![(get(&k1), get(&k2), &k2 - &k1, Rat::ONE)])));
    }
    Ok(None)
}

fn nonzero_coefficient(c: &Basic) -> bool {
    if crate::core::assumptions::is_number(c) {
        return is(c, Predicate::zero) == Some(false) && yes(c, Predicate::finite);
    }
    is(c, Predicate::zero) != Some(true) && is(c, Predicate::finite) != Some(false)
}

/// `f` as one binomial differential, the proportional binomials merged
/// (`binomial_exponents`).
pub fn binomial_exponents(f: &Basic, x: &Basic) -> PolyResult<Option<BinomialDifferential>> {
    let Some((m, powers)) = binomial_powers(f, x)? else {
        return Ok(None);
    };
    let mut merged: Vec<BinomialPower> = Vec::new();
    'outer: for (a, b, n, e) in powers {
        for entry in merged.iter_mut() {
            if entry.2 == n && cancel(&s2(&m2(&a, &entry.1)?, &m2(&entry.0, &b)?)?)? == Basic::zero() {
                entry.3 = &entry.3 + &e;
                continue 'outer;
            }
        }
        merged.push((a, b, n, e));
    }
    let remaining: Vec<BinomialPower> = merged.into_iter().filter(|p| p.3 != Rat::ZERO).collect();
    if remaining.len() != 1 {
        return Ok(None);
    }
    let (a, b, n, p) = remaining.into_iter().next().expect("one");
    if !(nonzero_coefficient(&a) && nonzero_coefficient(&b)) {
        return Ok(None);
    }
    Ok(Some(BinomialDifferential { m, a, b, n, p }))
}

/// Chebyshev's theorem: the antiderivative is elementary when `p`,
/// `(m + 1)/n` or `(m + 1)/n + p` is an integer (`chebyshev_elementary`).
pub fn chebyshev_elementary(binomial: &BinomialDifferential) -> bool {
    let q: Rat = (&binomial.m + Rat::ONE) / &binomial.n;
    is_whole(&binomial.p) || is_whole(&q) || is_whole(&(&q + &binomial.p))
}

fn scaled_power(a: &Basic, b: &Basic, w: &Basic, p: &Basic) -> R {
    let minus_p: Basic = neg(p)?;
    if yes(a, Predicate::extended_positive) {
        return m2(&pw(a, p)?, &pw(&a2(a, &m2(b, w)?)?, &minus_p)?);
    }
    if yes(a, Predicate::extended_negative) {
        let na: Basic = neg(a)?;
        return m2(&pw(&na, p)?, &pw(&s2(&na, &m2(b, w)?)?, &minus_p)?);
    }
    pw(&a2(&Basic::one(), &d2(&m2(b, w)?, a)?)?, &minus_p)
}

fn real_at(f: &Basic, x: &Basic, point: &Basic) -> PolyResult<bool> {
    let rule: std::collections::HashMap<Basic, Basic> = [(x.clone(), point.clone())].into_iter().collect();
    let value = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::core::function::xreplace_evaluated(f, &rule)
    })?;
    Ok(value.is_some_and(|v| yes(&v, Predicate::extended_real)))
}

/// The antiderivative of a binomial differential which Chebyshev's
/// theorem does not make elementary, by `2F1` in `z = -b x**n/a` and in
/// `1/z` beyond `|z| = 1`, joined where both forms are real
/// (`binomial_hypergeometric_antiderivative`).
pub fn binomial_hypergeometric_antiderivative(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::special::hyper::hyper;
    let Some(binomial) = binomial_exponents(f, x)? else {
        return Ok(None);
    };
    if chebyshev_elementary(&binomial) {
        return Ok(None);
    }
    let (m_, n_, p_) = (binomial.m.clone(), binomial.n.clone(), binomial.p.clone());
    let (m, n, p) = (rat(&m_), rat(&n_), rat(&p_));
    let (a, b) = (&binomial.a, &binomial.b);
    let q: Basic = rat(&((&m_ + Rat::ONE) / &n_));
    let ratio: Basic = d2(&neg(b)?, a)?;
    let xn: Basic = pw(x, &n)?;
    let z: Basic = m2(&ratio, &xn)?;
    let m2_: Basic = rat(&(&m_ + &n_ * &p_));
    let q2: Basic = rat(&(-((&m_ + Rat::ONE) / &n_ + &p_)));
    let inner_factor: Basic = m2(f, &scaled_power(a, b, &xn, &p)?)?;
    let outer_factor: Basic = m2(f, &scaled_power(b, a, &pw(x, &neg(&n)?)?, &p)?)?;
    let one: Basic = Basic::one();
    let minus_p: Basic = neg(&p)?;
    let inside: Basic = m2(
        &d2(&m2(x, &inner_factor)?, &a2(&m, &one)?)?,
        &hyper(&[minus_p.clone(), q.clone()], &[a2(&q, &one)?], &z)?,
    )?;
    let outside: Basic = m2(
        &d2(&m2(x, &outer_factor)?, &a2(&m2_, &one)?)?,
        &hyper(&[minus_p, q2.clone()], &[a2(&q2, &one)?], &d2(&one, &z)?)?,
    )?;
    if !crate::core::assumptions::is_number(&ratio) {
        return Ok(Some(parametric_cases(&inside, &outside, &ratio, x, &n_)?));
    }
    if !yes(&ratio, Predicate::extended_real) {
        return Ok(Some(inside));
    }
    let mut pieces: Vec<(Basic, Basic)> = Vec::new();
    for side in [1i64, -1] {
        if side == -1 && denominator(&n_) != Int::ONE {
            continue;
        }
        if !yes(&m2(&ratio, &pw(&int(side), &n)?)?, Predicate::extended_positive) {
            continue;
        }
        let boundary: Basic = pw(
            &crate::functions::elementary::complexes::abs_(&ratio)?,
            &d2(&int(-1), &n)?,
        )?;
        let ceiling = crate::functions::elementary::integers::ceiling;
        let beyond: Basic = m2(&int(side), &ceiling(&m2(&int(2), &boundary)?)?)?;
        let k: Basic = ceiling(&d2(&int(2), &boundary)?)?;
        let k: i64 = rational_of(&k)
            .filter(is_whole)
            .map(|r| to_i64(&numerator(&r)))
            .transpose()?
            .ok_or_else(|| PolyError::Type(format!("int() of {}", k)))?;
        let within: Basic = m2(&int(side), &d2(&int(1), &int(k))?)?;
        if !real_at(f, x, &beyond)? {
            continue;
        }
        let mut piece: Basic = outside.clone();
        if real_at(f, x, &within)? && p_ > Rat::from(Int::from(-1)) {
            let point: Basic = m2(&int(side), &boundary)?;
            let Some(joined) = joined(x, &point, side, &inner_factor, &outer_factor, &m, &q, &m2_, &q2, &p)? else {
                continue;
            };
            piece = a2(&outside, &joined)?;
        }
        let condition: Basic = if side == 1 {
            crate::core::relational::gt(x, &boundary)?
        } else {
            crate::core::relational::lt(x, &neg(&boundary)?)?
        };
        pieces.push((piece, condition));
    }
    if pieces.is_empty() {
        return Ok(Some(inside));
    }
    pieces.push((inside, crate::core::logic::true_()));
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&pieces)?))
}

fn parametric_cases(inside: &Basic, outside: &Basic, ratio: &Basic, x: &Basic, n: &Rat) -> R {
    let mut free: Vec<Basic> = ratio.free_symbols().into_iter().collect();
    free.sort_by_key(|s| s.to_string());
    let real: std::collections::HashMap<Basic, Basic> = free
        .into_iter()
        .map(|s| {
            let name: String = match s.head() {
                Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
                _ => s.to_string(),
            };
            let d: Basic = fresh(&name, &[(Predicate::real, true), (Predicate::nonzero, true)]);
            (s, d)
        })
        .collect();
    if !yes(
        &crate::core::function::xreplace_evaluated(ratio, &real)?,
        Predicate::extended_real,
    ) {
        return Ok(inside.clone());
    }
    let nb: Basic = rat(n);
    let one: Basic = Basic::one();
    let condition: Basic = if denominator(n) == Int::ONE {
        let even: bool = (numerator(n) % Int::from(2)) == Int::ZERO;
        if even && yes(ratio, Predicate::extended_nonpositive) {
            return Ok(inside.clone());
        }
        crate::core::relational::gt(&m2(ratio, &pw(x, &nb)?)?, &one)?
    } else {
        if yes(ratio, Predicate::extended_nonpositive) {
            return Ok(inside.clone());
        }
        let absolute: Basic = crate::functions::elementary::complexes::abs_(x)?;
        crate::core::logic::and_(&[
            crate::core::relational::gt(x, &Basic::zero())?,
            crate::core::relational::gt(&m2(ratio, &pw(&absolute, &nb)?)?, &one)?,
        ])
    };
    if condition == crate::core::logic::false_() {
        return Ok(inside.clone());
    }
    crate::functions::elementary::piecewise::piecewise(&[
        (outside.clone(), condition),
        (inside.clone(), crate::core::logic::true_()),
    ])
}

fn joined(
    x: &Basic,
    point: &Basic,
    side: i64,
    inner_factor: &Basic,
    outer_factor: &Basic,
    m: &Basic,
    q: &Basic,
    m2_: &Basic,
    q2: &Basic,
    p: &Basic,
) -> PolyResult<Option<Basic>> {
    use crate::functions::special::gamma_functions::gamma;
    let (before, after) = if side == 1 { ("-", "+") } else { ("+", "-") };
    let timeout: Option<f64> = crate::settings::settings().timeout;
    let inner = crate::timeout::attempt_within(timeout, || {
        crate::assumptions::limits::limit(inner_factor, x, point, before, &[], None)
    })?;
    let outer = crate::timeout::attempt_within(timeout, || {
        crate::assumptions::limits::limit(outer_factor, x, point, after, &[], None)
    })?;
    let bad = |v: &Basic| {
        !crate::core::assumptions::is_number(v)
            || v.preorder()
                .iter()
                .any(|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity | Head::Limit))
            || !yes(v, Predicate::finite)
    };
    let (Some(inner), Some(outer)) = (inner, outer) else {
        return Ok(None);
    };
    if bad(&inner) || bad(&outer) {
        return Ok(None);
    }
    let one: Basic = Basic::one();
    let first: Basic = d2(
        &chain(&[point.clone(), inner, gamma(&a2(q, &one)?)?, gamma(&a2(p, &one)?)?])?,
        &m2(
            &gamma(&crate::core::add::add(&[q.clone(), p.clone(), one.clone()])?)?,
            &a2(m, &one)?,
        )?,
    )?;
    let second: Basic = d2(
        &chain(&[point.clone(), outer, gamma(&a2(q2, &one)?)?, gamma(&a2(p, &one)?)?])?,
        &m2(
            &gamma(&crate::core::add::add(&[q2.clone(), p.clone(), one.clone()])?)?,
            &a2(m2_, &one)?,
        )?,
    )?;
    Ok(Some(s2(&first, &second)?))
}

// -- the transformed integral

fn bound(t_of_x: &Basic, x: &Basic, point: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    let value: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::assumptions::limits::limit(t_of_x, x, point, "+", assumptions, None)
    })?;
    let Some(value) = value else {
        return Ok(None);
    };
    let bad: bool = value
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity | Head::Limit));
    let free = t_of_x.free_symbols();
    if bad || value.free_symbols().iter().any(|s| !free.contains(s)) {
        return Ok(None);
    }
    Ok(Some(value))
}

fn is_rational_in(g: &Basic, t: &Basic) -> bool {
    for node in g.atoms(Some(&|n: &Basic| n.is_pow())) {
        if node.has(t) && !is_integer_node(&node.args()[1]) {
            return false;
        }
    }
    crate::core::expr::polynomial_like(g, std::slice::from_ref(t), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}

/// The integral of the transformed integrand over the image of the range
/// (`_transformed`).
fn transformed(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    substitution: &Rationalisation,
    assumptions: &[Basic],
    g: Option<Basic>,
) -> PolyResult<Option<crate::integrals::conditions::ConditionalValue>> {
    let t: &Basic = &substitution.t;
    let g: Basic = match g {
        Some(g) => g,
        None => {
            let rule: std::collections::HashMap<Basic, Basic> = substitution.radicals.iter().cloned().collect();
            let replaced: Basic = crate::core::function::xreplace_evaluated(f, &rule)?;
            if !radical_atoms(&replaced, x).is_empty() {
                return Ok(None);
            }
            m2(
                &crate::core::subs::subs(&replaced, x, &substitution.x_of_t)?,
                &crate::core::function::derivative(&substitution.x_of_t, t)?,
            )?
        }
    };
    let g: Basic = cancel(&g)?;
    if !is_rational_in(&g, t) {
        return Ok(None);
    }
    let (Some(lower), Some(upper)) = (
        bound(&substitution.t_of_x, x, a, assumptions)?,
        bound(&substitution.t_of_x, x, b, assumptions)?,
    ) else {
        return Ok(None);
    };
    if lower == upper {
        return Ok(None);
    }
    let Some(found) =
        crate::integrals::definite::conditional_integral(&g, t, &lower, &upper, assumptions, false, false, false)?
    else {
        return Ok(None);
    };
    let value: Basic = crate::integrals::marichev::tidy(&found.value, assumptions, &found.condition)?;
    if crate::settings::settings().numerical_checks
        && crate::integrals::definite::verify_numerically(&value, f, x, a, b, assumptions, 2, &[])? == Some(false)
    {
        return Ok(None);
    }
    Ok(Some(crate::integrals::conditions::ConditionalValue::new(
        value,
        found.condition,
    )))
}

/// `Integral(f, (x, a, b))` through a rationalising substitution
/// (`rationalised_integral`).
pub fn rationalised_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    substitution: &Rationalisation,
    assumptions: &[Basic],
) -> PolyResult<Option<crate::integrals::conditions::ConditionalValue>> {
    transformed(f, x, a, b, substitution, assumptions, None)
}

/// Chebyshev's three cases (`_binomial_integral`).
fn binomial_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<crate::integrals::conditions::ConditionalValue>> {
    let Some([c, m, a_, b_, n, p]) = binomial_differential(f, x)? else {
        return Ok(None);
    };
    let (Some(m_), Some(n_), Some(p_)) = (rational_of(&m), rational_of(&n), rational_of(&p)) else {
        return Ok(None);
    };
    if n_ == Rat::ZERO {
        return Ok(None);
    }
    let t: Basic = fresh("t", &[(Predicate::positive, true)]);
    let q: Rat = (&m_ + Rat::ONE) / &n_;
    let (g, t_of_x): (Basic, Basic) = if is_whole(&p_) {
        let k: i64 = lcm(to_i64(&denominator(&m_))?, to_i64(&denominator(&n_))?);
        let kr: Rat = Rat::from(k);
        let g: Basic = chain(&[
            c.clone(),
            pw(&t, &rat(&(&m_ * &kr)))?,
            pw(&a2(&a_, &m2(&b_, &pw(&t, &rat(&(&n_ * &kr)))?)?)?, &p)?,
            int(k),
            pw(&t, &int(k - 1))?,
        ])?;
        (g, pw(x, &rat(&(Rat::ONE / kr)))?)
    } else if is_whole(&q) {
        let s: i64 = to_i64(&denominator(&p_))?;
        let qm1: i64 = to_i64(&numerator(&(&q - Rat::ONE)))?;
        let g: Basic = chain(&[
            d2(&m2(&c, &int(s))?, &m2(&n, &b_)?)?,
            pw(&t, &int(s - 1))?,
            pw(&t, &rat(&(Rat::from(s) * &p_)))?,
            pw(&d2(&s2(&pw(&t, &int(s))?, &a_)?, &b_)?, &int(qm1))?,
        ])?;
        (
            g,
            pw(&a2(&a_, &m2(&b_, &pw(x, &n)?)?)?, &rat(&(Rat::ONE / Rat::from(s))))?,
        )
    } else if is_whole(&(&q + &p_)) {
        let s: i64 = to_i64(&denominator(&p_))?;
        let exponent: i64 = -to_i64(&numerator(&(&q + &p_)))? - 1;
        let g: Basic = chain(&[
            d2(&m2(&neg(&c)?, &int(s))?, &m2(&n, &a_)?)?,
            pw(&t, &rat(&(Rat::from(s - 1) + Rat::from(s) * &p_)))?,
            pw(&d2(&s2(&pw(&t, &int(s))?, &b_)?, &a_)?, &int(exponent))?,
        ])?;
        let t_of_x: Basic = pw(
            &a2(&m2(&a_, &pw(x, &neg(&n)?)?)?, &b_)?,
            &rat(&(Rat::ONE / Rat::from(s))),
        )?;
        (g, t_of_x)
    } else {
        return Ok(None);
    };
    let substitution: Rationalisation = Rationalisation {
        t,
        x_of_t: Basic::zero(),
        t_of_x,
        radicals: Vec::new(),
    };
    transformed(f, x, a, b, &substitution, assumptions, Some(g))
}

/// `Integral(f, (x, a, b))` for an algebraic integrand of genus zero
/// (`algebraic_integral`): Möbius roots, Euler's substitutions, Chebyshev's
/// binomial differentials.
pub fn algebraic_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<crate::integrals::conditions::ConditionalValue>> {
    if radical_atoms(f, x).is_empty() {
        return Ok(None);
    }
    if let Some(substitution) = moebius_root(f, x)? {
        if let Some(found) = transformed(f, x, a, b, &substitution, assumptions, None)? {
            return Ok(Some(found));
        }
    }
    let mut best: Option<(i64, crate::integrals::conditions::ConditionalValue)> = None;
    for candidate in euler_substitutions(f, x, assumptions)? {
        if let Some(found) = transformed(f, x, a, b, &candidate, assumptions, None)? {
            let size: i64 = crate::core::function::count_ops(&found.value)?;
            if best.as_ref().is_none_or_larger(size) {
                best = Some((size, found));
            }
        }
    }
    if let Some((_, found)) = best {
        return Ok(Some(found));
    }
    binomial_integral(f, x, a, b, assumptions)
}

trait Larger {
    fn is_none_or_larger(&self, size: i64) -> bool;
}
impl Larger for Option<&(i64, crate::integrals::conditions::ConditionalValue)> {
    fn is_none_or_larger(&self, size: i64) -> bool {
        match self {
            None => true,
            Some((s, _)) => size < *s,
        }
    }
}
