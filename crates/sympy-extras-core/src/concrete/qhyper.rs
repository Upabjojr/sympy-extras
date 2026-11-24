//! q-hypergeometric summation, ported from `sympy_extras.concrete.qhyper`:
//! the q-Pochhammer symbol `(a; q)_k = prod_{i=0}^{k-1} (1 - a q**i)` (the
//! node `Function::QPochhammer`, evaluated by [`qpochhammer`]), the
//! q-binomial coefficients, and the q-analogues of Gosper's and
//! Zeilberger's algorithms.

use crate::core::basic::{Basic, Function, Head};
use crate::error::PolyResult;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}

/// `isinstance(k, Integer)`: its value.
fn integer_class(k: &Basic) -> Option<i64> {
    if !matches!(
        k.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) {
        return None;
    }
    k.to_string().parse::<i64>().ok()
}

/// The node `QPochhammer(a, q, k)`, unevaluated.
pub fn qpochhammer_node(a: &Basic, q: &Basic, k: &Basic) -> Basic {
    Basic::new(
        Head::Function(Function::QPochhammer),
        vec![a.clone(), q.clone(), k.clone()],
    )
}

/// `QPochhammer.eval`: the product for an integer `k` (its reciprocal
/// form for a negative one), `1` for `a = 0`, the node otherwise.
pub fn qpochhammer_eval(a: &Basic, q: &Basic, k: &Basic) -> R {
    let factor = |i: i64| -> R { sub(&Basic::one(), &mul(&[a.clone(), pow(q, &int(i))?])?) };
    if let Some(m) = integer_class(k) {
        if m >= 0 {
            return mul(&(0..m).map(factor).collect::<PolyResult<Vec<Basic>>>()?);
        }
    }
    if *a == Basic::zero() {
        return Ok(Basic::one());
    }
    if let Some(m) = integer_class(k) {
        // (a; q)_{-m} = 1/prod_{i=1}^{m} (1 - a q**(-i))
        let product: Basic = mul(&(1..=-m).map(|i| factor(-i)).collect::<PolyResult<Vec<Basic>>>()?)?;
        return div(&Basic::one(), &product);
    }
    Ok(qpochhammer_node(a, q, k))
}

/// `(a; q)_k`, evaluated (`qpochhammer`).
pub fn qpochhammer(a: &Basic, q: &Basic, k: &Basic) -> R {
    qpochhammer_eval(a, q, k)
}

/// The q-binomial coefficient `(q; q)_n / ((q; q)_k (q; q)_{n-k})`,
/// cancelled (`qbinomial`).
pub fn qbinomial(n: &Basic, k: &Basic, q: &Basic) -> R {
    let numerator: Basic = qpochhammer(q, q, n)?;
    let denominator: Basic = mul(&[qpochhammer(q, q, k)?, qpochhammer(q, q, &sub(n, k)?)?])?;
    crate::polys::polytools::cancel(&div(&numerator, &denominator)?, &[])
}

// -- the q-Gosper and q-Zeilberger algorithms

use std::collections::HashMap;

use crate::concrete::gosper::UPoly;
use crate::concrete::karr::SumPoint;
use crate::core::symbol::Dummy;
use crate::error::PolyError;

fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn factor(e: &Basic) -> R {
    crate::polys::polytools::factor(e, &[], false, true)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn dummy(name: &str) -> Basic {
    Dummy::new(name, &[]).0
}
fn is_qpochhammer(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(Function::QPochhammer))
}
fn has_nan_or_zoo(e: &Basic) -> bool {
    e.has(&Basic::nan()) || e.has(&Basic::zoo())
}

/// `e` with the keys of `rule` replaced, the rebuilt nodes evaluated
/// (`_evaluated`).
#[allow(clippy::mutable_key_type)]
fn evaluated(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

/// The q-Pochhammer symbols of `e`, by their sort key (`_symbols`).
fn symbols(e: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = e.atoms(Some(&is_qpochhammer));
    found.sort_by_key(crate::core::sorting::sort_key);
    found.dedup();
    found
}

/// `(a; q)_{m+c}` through `(a; q)_m` (`_step`).
fn step(symbol: &Basic, a: &Basic, q: &Basic, m: &Basic, c: i64) -> R {
    let factor = |e: &Basic| -> R { sub(&Basic::one(), &mul(&[a.clone(), pow(q, e)?])?) };
    if c >= 0 {
        let mut factors: Vec<Basic> = vec![symbol.clone()];
        for i in 0..c {
            factors.push(factor(&add(&[m.clone(), int(i)])?)?);
        }
        return mul(&factors);
    }
    let mut factors: Vec<Basic> = Vec::new();
    for i in 1..=-c {
        factors.push(factor(&sub(m, &int(i))?)?);
    }
    div(symbol, &mul(&factors)?)
}

fn arguments(symbol: &Basic) -> (Basic, Basic, Basic) {
    let a: &[Basic] = symbol.args();
    (a[0].clone(), a[1].clone(), a[2].clone())
}

/// `e` with `k` replaced by `k + 1`, the q-Pochhammer symbols with
/// arguments linear in `k` rewritten through `(a; q)_{m+1} = (a; q)_m (1 -
/// a q**m)`; `None` for an argument which is not (`_shift_qpochhammer`).
fn shift_qpochhammer(e: &Basic, k: &Basic) -> PolyResult<Option<Basic>> {
    let k1: Basic = add(&[k.clone(), Basic::one()])?;
    let mut opaque: Vec<(Basic, Basic)> = Vec::new();
    let mut back: Vec<(Basic, Basic)> = Vec::new();
    for symbol in symbols(e) {
        let (a, q, m) = arguments(&symbol);
        let Some(c) = integer_class(&sub(&subs(&m, k, &k1)?, &m)?) else {
            return Ok(None);
        };
        let d: Basic = dummy("P");
        opaque.push((symbol.clone(), d.clone()));
        back.push((d, step(&symbol, &a, &q, &m, c)?));
    }
    let shifted: Basic = subs(&evaluated(e, &opaque)?, k, &k1)?;
    Ok(Some(evaluated(&shifted, &back)?))
}

fn rational_in(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}

/// `t(k+1)/t(k)` as a rational function of `Q = q**k` with the symbol `Q`,
/// `None` when the term is not q-hypergeometric (`q_ratio`).
pub fn q_ratio(term: &Basic, k: &Basic, q: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some(shifted) = shift_qpochhammer(term, k)? else {
        return Ok(None);
    };
    let opaque: Vec<(Basic, Basic)> = symbols(term).into_iter().map(|s| (s, dummy("P"))).collect();
    let big_q: Basic = dummy("Q");
    let quotient: Basic = evaluated(&div(&shifted, term)?, &opaque)?;
    let rational: Basic = cancel(&in_q(&quotient, k, q, &big_q)?)?;
    if opaque.iter().any(|(_, d)| rational.has(d)) {
        return Ok(None);
    }
    if rational.has(k) || !rational_in(&rational, &big_q) {
        return Ok(None);
    }
    Ok(Some((rational, big_q)))
}

/// `e` with the powers of the same base combined and every `q**(c k + d)`
/// written as `Q**c q**d` (`_in_Q`).
fn in_q(e: &Basic, k: &Basic, q: &Basic, big_q: &Basic) -> R {
    let combined: Basic = crate::simplify::powsimp::powsimp_default(e)?;
    let hints = crate::core::function::Hints {
        power_base: false,
        ..crate::core::function::Hints::DEFAULT
    };
    let expanded: Basic = crate::core::function::expand_with(&combined, true, hints)?;
    let is_power = |p: &Basic| p.is_pow() && p.args()[0] == *q && p.args()[1].has(k);
    crate::core::function::replace_where(&expanded, &is_power, &|p: &Basic| split_power(p, k, q, big_q))
}

/// `e` as a factored rational function of `q**k` (`normal_in`).
pub fn normal_in(e: &Basic, k: &Basic, q: &Basic) -> R {
    let big_q: Basic = dummy("Q");
    let rational: Basic = cancel(&in_q(e, k, q, &big_q)?)?;
    subs(&factor(&rational)?, &big_q, &pow(q, k)?)
}

/// `q**(c*k + d)` as `Q**c q**d` for an integer `c` (`_split_power`).
fn split_power(e: &Basic, k: &Basic, q: &Basic, big_q: &Basic) -> R {
    if !e.is_pow() {
        return Ok(e.clone());
    }
    let Ok(poly) = UPoly::of(&e.args()[1], k) else {
        return Ok(e.clone());
    };
    if poly.degree() != 1 {
        return Ok(e.clone());
    }
    let (d, c) = (poly.c[0].clone(), poly.c[1].clone());
    let Some(c) = integer_class(&c) else {
        return Ok(e.clone());
    };
    mul(&[pow(big_q, &int(c))?, pow(q, &d)?])
}

/// `n` with `e == q**n` for an integer `n` in `[-bound, bound]`, or `None`
/// (`_q_power`).
fn q_power(e: &Basic, q: &Basic, bound: i64) -> PolyResult<Option<i64>> {
    let e: Basic = cancel(e)?;
    if e == Basic::one() {
        return Ok(Some(0));
    }
    if e.is_pow() && e.args()[0] == *q {
        if let Some(n) = integer_class(&e.args()[1]) {
            return Ok(Some(n));
        }
    }
    for n in -bound..=bound {
        if cancel(&sub(&e, &pow(q, &int(n))?)?)? == Basic::zero() {
            return Ok(Some(n));
        }
    }
    Ok(None)
}

/// `p(s Q)` (`_scaled`).
fn scaled(p: &UPoly, s: &Basic) -> PolyResult<UPoly> {
    let coeffs: Vec<Basic> =
        p.c.iter()
            .enumerate()
            .map(|(i, c)| mul(&[c.clone(), pow(s, &int(i as i64))?]))
            .collect::<PolyResult<_>>()?;
    UPoly::new(&coeffs)
}

/// The monic gcd of two polynomials in `Q` over the rational functions of
/// the parameters: the gcd of their numerators as polynomials in all the
/// symbols, Euclid's algorithm where that fails (`_gcd`).
fn upoly_gcd(f: &UPoly, g: &UPoly, big_q: &Basic) -> PolyResult<UPoly> {
    let attempt = || -> PolyResult<UPoly> {
        let numerator = |p: &UPoly| -> R { Ok(crate::core::expr::as_numer_denom(&cancel(&p.as_expr(big_q)?)?)?.0) };
        let common: Basic = crate::polys::polytools::gcd_expr(&numerator(f)?, &numerator(g)?, &[])?;
        UPoly::of(&common, big_q)?.monic()
    };
    match attempt() {
        Ok(p) => Ok(p),
        Err(PolyError::Value(_) | PolyError::Type(_) | PolyError::NotImplemented(_)) => f.gcd(g),
        // BasePolynomialError
        Err(PolyError::Polys(..)) => f.gcd(g),
        Err(e) => Err(e),
    }
}

/// `a, b, c` with `num/den == (a/b) c(qQ)/c(Q)` and `gcd(a(Q), b(q**j Q))
/// = 1` for `0 <= j <= bound` (`_normal_form`).
fn normal_form(num: &UPoly, den: &UPoly, q: &Basic, big_q: &Basic, bound: i64) -> PolyResult<(UPoly, UPoly, UPoly)> {
    let (mut a, mut b, mut c) = (num.clone(), den.clone(), UPoly::new(&[Basic::one()])?);
    for j in 0..=bound {
        loop {
            let g: UPoly = upoly_gcd(&a, &scaled(&b, &pow(q, &int(j))?)?, big_q)?;
            if g.degree() == 0 {
                break;
            }
            a = a.quo(&g)?;
            b = b.quo(&scaled(&g, &pow(q, &int(-j))?)?)?;
            for i in 1..=j {
                c = c.times(&scaled(&g, &pow(q, &int(-i))?)?)?;
            }
        }
    }
    Ok((a, b, c))
}

fn valuation(p: &UPoly) -> i64 {
    p.c.iter()
        .position(|c| *c != Basic::zero())
        .expect("a nonzero polynomial") as i64
}

fn trailing(p: &UPoly) -> Basic {
    p.c[valuation(p) as usize].clone()
}

/// The lowest and highest powers of `Q` in a Laurent polynomial solution
/// `f` of `a(Q) f(qQ) - b(Q/q) f(Q) = c(Q)` (`_degree_bounds`).
fn degree_bounds(
    a: &UPoly,
    b_shift: &UPoly,
    degree_c: i64,
    valuation_c: i64,
    q: &Basic,
) -> PolyResult<Option<(i64, i64)>> {
    let (da, db) = (a.degree(), b_shift.degree());
    let mut highs: Vec<i64> = vec![degree_c - da.max(db)];
    if da == db {
        if let Some(power) = q_power(&div(&b_shift.lc(), &a.lc())?, q, 30)? {
            highs.push(power);
        }
    }
    let (va, vb) = (valuation(a), valuation(b_shift));
    let mut lows: Vec<i64> = vec![valuation_c - va.min(vb)];
    if va == vb {
        if let Some(power) = q_power(&div(&trailing(b_shift), &trailing(a))?, q, 30)? {
            lows.push(power);
        }
    }
    let (high, low) = (*highs.iter().max().expect("one"), *lows.iter().min().expect("one"));
    if high < low {
        return Ok(None);
    }
    Ok(Some((low, high)))
}

/// The output of the q-Zeilberger algorithm: the coefficients `a_j(q**n)`
/// of the recurrence and the certificate `R` with `sum_j a_j t(n+j, k) =
/// G(k+1) - G(k)`, `G = R t(n, k)` (`QTelescoper`).
#[derive(Clone, Debug)]
pub struct QTelescoper {
    pub term: Basic,
    pub n: Basic,
    pub k: Basic,
    pub q: Basic,
    pub coefficients: Vec<Basic>,
    pub certificate: Basic,
}

impl QTelescoper {
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// The recurrence `sum_j a_j S(n + j)` in the function `name`.
    pub fn recurrence(&self, name: &str) -> R {
        let f = std::sync::Arc::new(crate::core::basic::UndefinedFunction::new(name, None, &[]));
        let mut terms: Vec<Basic> = Vec::new();
        for (j, a) in self.coefficients.iter().enumerate() {
            let at: Basic = f.apply(vec![add(&[self.n.clone(), int(j as i64)])?])?;
            terms.push(mul(&[a.clone(), at])?);
        }
        add(&terms)
    }
}

/// Solve the parametrised q-Gosper equation for the term whose ratio is
/// `ratio * P(qQ)/P(Q)`: the rational certificate `b(Q/q) f(Q)/(c(Q)
/// P(Q))` and the values of the unknowns (`_solve_qgosper`).
fn solve_qgosper(
    ratio: &Basic,
    big_q: &Basic,
    q: &Basic,
    p: &Basic,
    unknowns: &[Basic],
) -> PolyResult<Option<(Basic, Vec<(Basic, Basic)>)>> {
    let (num, den) = crate::core::expr::as_numer_denom(&cancel(ratio)?)?;
    let polys = || -> PolyResult<(UPoly, UPoly, UPoly)> {
        Ok((UPoly::of(&num, big_q)?, UPoly::of(&den, big_q)?, UPoly::of(p, big_q)?))
    };
    let (num_poly, den_poly, p_poly) = match polys() {
        Ok(v) => v,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let (a, b, c) = normal_form(&num_poly, &den_poly, q, big_q, 20)?;
    let b_shift: UPoly = scaled(&b, &div(&Basic::one(), q)?)?;
    let big_c: UPoly = c.times(&p_poly)?;
    let Some((low, high)) = degree_bounds(&a, &b_shift, big_c.degree(), valuation(&big_c), q)? else {
        return Ok(None);
    };
    let f_coefficients: Vec<Basic> = (low..=high).map(|i| dummy(&format!("f{}", i))).collect();
    let f: Basic = add(&(low..=high)
        .zip(&f_coefficients)
        .map(|(i, u)| mul(&[u.clone(), pow(big_q, &int(i))?]))
        .collect::<PolyResult<Vec<Basic>>>()?)?;
    let q_big_q: Basic = mul(&[q.clone(), big_q.clone()])?;
    let equation: Basic = expand(&sub(
        &sub(
            &mul(&[a.as_expr(big_q)?, subs(&f, big_q, &q_big_q)?])?,
            &mul(&[b_shift.as_expr(big_q)?, f.clone()])?,
        )?,
        &big_c.as_expr(big_q)?,
    )?)?;
    // clear the negative powers
    let equation: Basic = expand(&mul(&[equation, pow(big_q, &int(0i64.max(-low)))?])?)?;
    let rows: Vec<Basic> = match UPoly::of(&equation, big_q) {
        Ok(p) => p.c,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if !unknowns.is_empty() {
        let mut variables: Vec<Basic> = f_coefficients.clone();
        variables.extend(unknowns.iter().cloned());
        let matrix: Vec<Vec<Basic>> = rows
            .iter()
            .map(|row| {
                variables
                    .iter()
                    .map(|u| cancel(&crate::core::function::derivative(row, u)?))
                    .collect::<PolyResult<Vec<Basic>>>()
            })
            .collect::<PolyResult<_>>()?;
        for vector in crate::concrete::zeilberger::nullspace(&matrix, variables.len())? {
            let values: Vec<Basic> = vector.iter().map(cancel).collect::<PolyResult<_>>()?;
            let nf: usize = f_coefficients.len();
            if values[nf..].iter().all(|v| *v == Basic::zero()) {
                continue;
            }
            let f_rule: Vec<(Basic, Basic)> = f_coefficients
                .iter()
                .cloned()
                .zip(values[..nf].iter().cloned())
                .collect();
            let f_value: Basic = evaluated(&f, &f_rule)?;
            let assignment: Vec<(Basic, Basic)> = unknowns.iter().cloned().zip(values[nf..].iter().cloned()).collect();
            let p_value: Basic = evaluated(p, &assignment)?;
            let certificate: Basic = cancel(&div(
                &mul(&[b_shift.as_expr(big_q)?, f_value])?,
                &mul(&[c.as_expr(big_q)?, p_value])?,
            )?)?;
            return Ok(Some((certificate, assignment)));
        }
        return Ok(None);
    }
    // the inhomogeneous system for the coefficients of f alone
    let equations: Vec<Basic> = rows.into_iter().filter(|r| *r != Basic::zero()).collect();
    if equations.is_empty() {
        return Ok(None);
    }
    let Some(solution) = crate::concrete::gosper::solve_linear(&equations, &f_coefficients)? else {
        return Ok(None);
    };
    // the free coefficients (a solution of the homogeneous equation) are 0,
    // in the values of the others too
    let free: Vec<(Basic, Basic)> = f_coefficients
        .iter()
        .filter(|u| !solution.iter().any(|(s, _)| s == *u))
        .map(|u| (u.clone(), Basic::zero()))
        .collect();
    let mut values: Vec<(Basic, Basic)> = Vec::new();
    for u in &f_coefficients {
        let v: Basic = solution
            .iter()
            .find(|(s, _)| s == u)
            .map_or_else(Basic::zero, |(_, v)| v.clone());
        values.push((u.clone(), evaluated(&v, &free)?));
    }
    let f_value: Basic = evaluated(&f, &values)?;
    let certificate: Basic = cancel(&div(
        &mul(&[b_shift.as_expr(big_q)?, f_value])?,
        &mul(&[c.as_expr(big_q)?, p.clone()])?,
    )?)?;
    Ok(Some((certificate, Vec::new())))
}

fn not_q_hypergeometric(t: &Basic, k: &Basic) -> PolyError {
    PolyError::Value(format!("{} is not q-hypergeometric in {}", t, k))
}

fn in_point(point: &SumPoint, q: &Basic) -> bool {
    point.iter().any(|(s, _)| s == q)
}

/// `z(k)` with `z(k+1) - z(k) = term`, a q-hypergeometric term, or `None`
/// when there is none; with `special_values` the isolated values of the
/// parameters where `z` is undefined get cases of their own
/// (`qgosper_term`).
pub fn qgosper_term(term: &Basic, k: &Basic, q: &Basic, special_values: bool) -> PolyResult<Option<Basic>> {
    let Some((ratio, big_q)) = q_ratio(term, k, q)? else {
        return Err(not_q_hypergeometric(term, k));
    };
    let Some((certificate, _)) = solve_qgosper(&ratio, &big_q, q, &Basic::one(), &[])? else {
        return Ok(None);
    };
    let mut z: Basic = mul(&[subs(&certificate, &big_q, &pow(q, k)?)?, term.clone()])?;
    if !matches!(z.head(), Head::NaN | Head::ComplexInfinity) {
        z = factor(&z)?;
    }
    if !special_values {
        return Ok(Some(z));
    }
    let at_point = |point: &SumPoint| -> PolyResult<Option<Basic>> {
        let at_term: Basic = evaluated(term, point)?;
        if in_point(point, q) {
            return crate::concrete::karr::karr_term(&at_term, k, &[], true, true);
        }
        qgosper_term(&at_term, k, q, true)
    };
    Ok(Some(crate::concrete::karr::with_special_values_sum(
        &z,
        term,
        std::slice::from_ref(k),
        &[],
        &at_point,
        false,
        None,
    )?))
}

/// `sum_{k=lo}^{hi} term` in closed form by the q-Gosper algorithm, or
/// `None` (`qgosper_sum`).
pub fn qgosper_sum(
    term: &Basic,
    limits: (&Basic, &Basic, &Basic),
    q: &Basic,
    special_values: bool,
) -> PolyResult<Option<Basic>> {
    let (k, lo, hi) = limits;
    let Some(z) = qgosper_term(term, k, q, false)? else {
        return Ok(None);
    };
    let value: Basic = sub(&subs(&z, k, &add(&[hi.clone(), Basic::one()])?)?, &subs(&z, k, lo)?)?;
    if has_nan_or_zoo(&value) {
        // an antidifference undefined at a limit
        return Ok(None);
    }
    if !special_values {
        return Ok(Some(value));
    }
    let at_point = |point: &SumPoint| -> PolyResult<Option<Basic>> {
        let at_term: Basic = evaluated(term, point)?;
        if in_point(point, q) {
            let limit = (k.clone(), lo.clone(), hi.clone());
            return Ok(Some(crate::concrete::karr::summation(
                &at_term,
                &[limit],
                &[],
                true,
                true,
            )?));
        }
        qgosper_sum(&at_term, (k, lo, hi), q, true)
    };
    Ok(Some(crate::concrete::karr::with_special_values_sum(
        &value,
        term,
        std::slice::from_ref(k),
        &[lo.clone(), hi.clone()],
        &at_point,
        false,
        None,
    )?))
}

/// The q-Zeilberger algorithm: a q-recurrence in `n` for `sum_k term` with
/// its certificate, of order at most `max_order` (`qzeilberger`).
pub fn qzeilberger(term: &Basic, n: &Basic, k: &Basic, q: &Basic, max_order: usize) -> PolyResult<Option<QTelescoper>> {
    let Some((ratio, big_q)) = q_ratio(term, k, q)? else {
        return Err(not_q_hypergeometric(term, k));
    };
    let mut ratios: Vec<Basic> = Vec::new();
    for j in 0..=max_order {
        let shifted: Basic = subs(term, n, &add(&[n.clone(), int(j as i64)])?)?;
        let Some(pair) = ratio_in_q(&shifted, term, k, q, &big_q)? else {
            return Err(not_q_hypergeometric(term, n));
        };
        ratios.push(pair);
    }
    for order in 1..=max_order {
        let unknowns: Vec<Basic> = (0..=order).map(|j| dummy(&format!("a{}", j))).collect();
        let denominators: Vec<Basic> = ratios[..=order]
            .iter()
            .map(|r| Ok(crate::core::expr::as_numer_denom(r)?.1))
            .collect::<PolyResult<_>>()?;
        let d: Basic = crate::polys::polytools::lcm_list(&denominators, &[])?;
        let mut parts: Vec<Basic> = Vec::new();
        for (u, r) in unknowns.iter().zip(&ratios[..=order]) {
            parts.push(mul(&[u.clone(), cancel(&mul(&[r.clone(), d.clone()])?)?])?);
        }
        let p: Basic = expand(&add(&parts)?)?;
        let shifted_d: Basic = subs(&d, &big_q, &mul(&[q.clone(), big_q.clone()])?)?;
        let reduced: Basic = cancel(&div(&mul(&[ratio.clone(), d.clone()])?, &shifted_d)?)?;
        let Some((certificate, assignment)) = solve_qgosper(&reduced, &big_q, q, &p, &unknowns)? else {
            continue;
        };
        let value_of = |u: &Basic| -> Basic {
            assignment
                .iter()
                .find(|(s, _)| s == u)
                .map_or_else(Basic::zero, |(_, v)| v.clone())
        };
        let coefficients: Vec<Basic> = unknowns
            .iter()
            .map(|u| normal_in(&value_of(u), n, q))
            .collect::<PolyResult<_>>()?;
        // G = certificate * T = certificate * P/D * t
        let r: Basic = cancel(&div(&mul(&[certificate, evaluated(&p, &assignment)?])?, &d)?)?;
        return Ok(Some(QTelescoper {
            term: term.clone(),
            n: n.clone(),
            k: k.clone(),
            q: q.clone(),
            coefficients,
            certificate: subs(&r, &big_q, &pow(q, k)?)?,
        }));
    }
    Ok(None)
}

/// `f/g` as a rational function of `Q = q**k`, the q-Pochhammer symbols
/// cancelled through their shift rules (`_ratio_in_Q`).
fn ratio_in_q(f: &Basic, g: &Basic, k: &Basic, q: &Basic, big_q: &Basic) -> PolyResult<Option<Basic>> {
    let bases: Vec<Basic> = symbols(g);
    let opaque: Vec<(Basic, Basic)> = bases.iter().map(|s| (s.clone(), dummy("P"))).collect();
    // express the symbols of f through those of g: (a; q)_{m + c} = (a; q)_m * ...
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    for symbol in symbols(f) {
        let (a, qq, m) = arguments(&symbol);
        for base in &bases {
            let (a2, qq2, m2) = arguments(base);
            if a2 == a && qq2 == qq {
                if let Some(c) = integer_class(&sub(&m, &m2)?) {
                    replacements.push((symbol.clone(), step(base, &a, &qq, &m2, c)?));
                    break;
                }
            }
        }
    }
    let f_: Basic = evaluated(f, &replacements)?;
    let rational: Basic = cancel(&in_q(&evaluated(&div(&f_, g)?, &opaque)?, k, q, big_q)?)?;
    if opaque.iter().any(|(_, d)| rational.has(d)) {
        return Ok(None);
    }
    if rational.has(k) || !rational_in(&rational, big_q) {
        return Ok(None);
    }
    Ok(Some(rational))
}
