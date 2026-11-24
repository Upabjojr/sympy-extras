//! Formal power series, ported from `sympy_extras.series.formal` (SymPy's
//! `sympy.series.formal`): the formula of the coefficients of the power
//! series of a function by the rational algorithm, by a simple
//! differential equation and its recurrence (the hypergeometric
//! algorithm), and the series built on it.

// PORTED FROM SYMPY: sympy/series/formal.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: strict typing; the helpers of the Python port (its linsolve,
// full partial fractions, the rsolve with initial values).

use std::sync::Arc;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head, UndefinedFunction};
use crate::core::expr::is_true;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;

/// A formula of the coefficients: `(formula, independent terms, first
/// index)`.
pub type Formula = (Basic, Basic, Basic);

/// Hypergeometric solutions: pairs `(term, exponent)`.
pub type Solutions = Vec<(Basic, Basic)>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    if n == 0 {
        return Ok(e.clone());
    }
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn independent(e: &Basic, x: &Basic, as_add: bool) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_independent(e, x, as_add)
}
fn rational_in(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}
fn has_infinity(e: &Basic) -> bool {
    e.has(&Basic::zoo()) || e.has(&Basic::oo()) || e.has(&Basic::nan())
}

/// `isinstance(e, Integer)`: its value.
fn integer_class(e: &Basic) -> Option<i64> {
    if !matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) {
        return None;
    }
    e.to_string().parse::<i64>().ok()
}

/// `a < b` decided (SymPy raises `TypeError` when it cannot) (`_lt`).
fn lt(a: &Basic, b: &Basic) -> PolyResult<bool> {
    let d: Basic = sub(b, a)?;
    if is_true(&d, Predicate::extended_positive) {
        return Ok(true);
    }
    if is_true(&d, Predicate::extended_nonpositive) {
        return Ok(false);
    }
    Err(PolyError::Type(format!(
        "cannot determine truth value of {} < {}",
        a, b
    )))
}

/// `isinstance(e, Number) and e == 0` (`_is_zero`).
fn is_zero(e: &Basic) -> bool {
    e.is_number_class() && *e == Basic::zero()
}

/// SymPy's `e.as_coeff_add(*deps)`: the terms free of `deps` added, and
/// the others (`_coeff_add`).
fn coeff_add(e: &Basic, deps: &[Basic]) -> PolyResult<(Basic, Vec<Basic>)> {
    let mut free: Vec<Basic> = Vec::new();
    let mut dependent: Vec<Basic> = Vec::new();
    for t in crate::core::expr::terms_of(e) {
        if deps.iter().any(|d| t.has(d)) {
            dependent.push(t);
        } else {
            free.push(t);
        }
    }
    Ok((add(&free)?, dependent))
}

/// `Derivative(g(x), x, i)`: `g(x)` itself for `i = 0` (`_derivative`).
fn derivative_of(gx: &Basic, x: &Basic, i: u64) -> Basic {
    if i == 0 {
        return gx.clone();
    }
    crate::core::function::derivative_node(gx, &[(x.clone(), i)])
}

fn limit0(e: &Basic, x: &Basic) -> R {
    crate::core::limits::limit(e, x, &Basic::zero(), "+")
}

/// SymPy's `integrate(e, x)`: an antiderivative, the unevaluated integral
/// when there is none (`_integrate`).
fn integrate(e: &Basic, x: &Basic) -> R {
    if !e.has(x) {
        return mul(&[e.clone(), x.clone()]);
    }
    crate::integrals::indefinite::indefinite_integral(e, x, &[], true)
}

/// The roots of the polynomial `p` in `k` (none for a constant), in their
/// order (`_roots`).
fn roots_of(p: &Basic, k: &Basic) -> PolyResult<Vec<(Basic, usize)>> {
    if !p.has(k) {
        return Ok(Vec::new());
    }
    match crate::polys::polyroots::roots(p, std::slice::from_ref(k), true, true, None) {
        Ok(found) => Ok(found.iter().map(|(r, m)| (r.clone(), *m)).collect()),
        Err(PolyError::Polys(..)) => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

fn numerator(e: &Basic) -> R {
    Ok(numer_denom(e)?.0)
}

/// SymPy's `linsolve` as `fps` reads it: the unknowns with their values,
/// the free unknowns with themselves; empty when there is no solution
/// (Gauss-Jordan elimination, `cancel` deciding the zeros) (`_linsolve`).
fn linsolve(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Vec<(Basic, Basic)>> {
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for e in equations {
        let mut row: Vec<Basic> = Vec::new();
        for u in unknowns {
            let c: Basic = cancel(&diff(e, u, 1)?)?;
            if unknowns.iter().any(|v| c.has(v)) {
                return Ok(Vec::new());
            }
            row.push(c);
        }
        let mut at_zero: Basic = e.clone();
        for u in unknowns {
            at_zero = subs(&at_zero, u, &Basic::zero())?;
        }
        row.push(cancel(&at_zero)?);
        rows.push(row);
    }
    let n: usize = unknowns.len();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r: usize = 0;
    for c in 0..n {
        let mut p: Option<usize> = None;
        for (i, row) in rows.iter().enumerate().skip(r) {
            if cancel(&row[c])? != Basic::zero() {
                p = Some(i);
                break;
            }
        }
        let Some(p) = p else { continue };
        rows.swap(r, p);
        let lead: Basic = rows[r][c].clone();
        rows[r] = rows[r]
            .iter()
            .map(|v| cancel(&div(v, &lead)?))
            .collect::<PolyResult<_>>()?;
        for i in 0..rows.len() {
            if i != r && rows[i][c] != Basic::zero() {
                let factor: Basic = rows[i][c].clone();
                let pivot_row: Vec<Basic> = rows[r].clone();
                rows[i] = rows[i]
                    .iter()
                    .zip(&pivot_row)
                    .map(|(a, b)| cancel(&sub(a, &mul(&[factor.clone(), b.clone()])?)?))
                    .collect::<PolyResult<_>>()?;
            }
        }
        pivots.push(c);
        r += 1;
    }
    for row in &rows[r..] {
        if cancel(&row[n])? != Basic::zero() {
            return Ok(Vec::new());
        }
    }
    let mut solution: Vec<(Basic, Basic)> = unknowns.iter().map(|u| (u.clone(), u.clone())).collect();
    for (i, &c) in pivots.iter().enumerate() {
        let mut value: Basic = neg(&rows[i][n])?;
        for j in 0..n {
            if j != c && !pivots[..i].contains(&j) && rows[i][j] != Basic::zero() {
                value = sub(&value, &mul(&[rows[i][j].clone(), unknowns[j].clone()])?)?;
            }
        }
        solution[c].1 = cancel(&value)?;
    }
    Ok(solution)
}

/// SymPy's `rsolve` with initial values: `None` when no closed form is
/// found (`_rsolve`).
fn rsolve(equation: &Basic, y: &Basic, init: &[(Basic, Basic)]) -> PolyResult<Option<Basic>> {
    match crate::solvers::recurr::rsolve_linear(equation, y, Some(init)) {
        Ok(v) => Ok(v),
        Err(PolyError::Value(_) | PolyError::NotImplemented(_) | PolyError::Type(_) | PolyError::ZeroDivision(_)) => {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// The quotient of the polynomial of coefficients `d` (highest first) by
/// `x - a`, `a` a root (synthetic division) (`_taylor_shift_division`).
fn taylor_shift_division(d: &[Basic], a: &Basic) -> PolyResult<Vec<Basic>> {
    let mut quotient: Vec<Basic> = Vec::new();
    let mut carry: Basic = Basic::zero();
    for c in &d[..d.len() - 1] {
        carry = expand(&add(&[c.clone(), mul(&[a.clone(), carry])?])?)?;
        quotient.push(carry.clone());
    }
    Ok(quotient)
}

/// The full partial fraction decomposition of the rational function `f`
/// with numerical coefficients over the roots of its denominator; `None`
/// when a root is not found or a coefficient is symbolic (`_apart_full`).
fn apart_full(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    use crate::polys::polytools::Poly;
    let (num, den) = numer_denom(&cancel(f)?)?;
    if num
        .free_symbols()
        .iter()
        .chain(den.free_symbols().iter())
        .any(|s| s != x)
    {
        return Ok(None);
    }
    let gens: &[Basic] = std::slice::from_ref(x);
    let (p, q) = (Poly::new(&num, gens)?, Poly::new(&den, gens)?);
    let (polynomial, remainder) = p.div(&q)?;
    let mut terms: Vec<Basic> = vec![polynomial.as_expr()?];
    let (_, factors) = q.sqf_list()?;
    let whole: Basic = div(&remainder.as_expr()?, &q.as_expr()?)?;
    for (d, n) in factors {
        if d.degree(0)? <= 0 {
            continue;
        }
        let found: Vec<(Basic, usize)> = roots_of(&d.as_expr()?, x)?;
        if found.iter().map(|(_, m)| *m).sum::<usize>() as i64 != d.degree(0)? {
            return Ok(None);
        }
        let cofactor: Basic = cancel(&mul(&[whole.clone(), pow(&d.as_expr()?, &int(n as i64))?])?)?;
        let coefficients: Vec<Basic> = d.all_coeffs()?;
        for (a, _) in &found {
            let quotient: Vec<Basic> = taylor_shift_division(&coefficients, a)?;
            let mut w_terms: Vec<Basic> = Vec::new();
            for (i, c) in quotient.iter().enumerate() {
                w_terms.push(mul(&[c.clone(), pow(x, &int((coefficients.len() - 2 - i) as i64))?])?);
            }
            let w: Basic = add(&w_terms)?;
            let g: Basic = div(&cofactor, &pow(&w, &int(n as i64))?)?;
            let mut derivative: Basic = g;
            for j in 0..n {
                if j > 0 {
                    derivative = div(&diff(&derivative, x, 1)?, &int(j as i64))?;
                }
                let c: Basic = crate::simplify::radsimp::radsimp(&expand(&subs(&derivative, x, a)?)?, true, 4)?;
                terms.push(div(&c, &pow(&sub(x, a)?, &int((n - j) as i64))?)?);
            }
        }
    }
    Ok(Some(add(&terms)?))
}

fn apart(f: &Basic, x: &Basic, full: bool) -> PolyResult<Option<Basic>> {
    if full {
        if let Some(found) = apart_full(f, x)? {
            return Ok(Some(found));
        }
    }
    match crate::polys::partfrac::apart(f, Some(x), false) {
        Ok(v) => Ok(Some(v)),
        Err(PolyError::NotImplemented(_) | PolyError::Value(_) | PolyError::Type(_) | PolyError::ZeroDivision(_)) => {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

// -- the rational algorithm

/// The formula of the coefficients of the power series of `f` at 0 by the
/// rational algorithm, when `f` or one of its first `order` derivatives is
/// a rational function: `(formula, independent terms, first index)`
/// (`rational_algorithm`).
pub fn rational_algorithm(f: &Basic, x: &Basic, k: &Basic, order: usize, full: bool) -> PolyResult<Option<Formula>> {
    use crate::functions::combinatorial::factorials::factorial;
    let mut derivative: Basic = f.clone();
    let mut ds: Vec<Basic> = Vec::new();
    for i in 0..=order {
        if i > 0 {
            derivative = diff(&derivative, x, 1)?;
        }
        if rational_in(&derivative, x) {
            let mut coeff: Basic = Basic::zero();
            let mut sep: Basic = Basic::zero();
            let Some(terms) = apart(&derivative, x, full)? else {
                return Ok(None);
            };
            for t in crate::core::expr::terms_of(&terms) {
                let (mut num, mut den) = numer_denom(&t)?;
                if !den.has(x) {
                    sep = add(&[sep, t.clone()])?;
                    continue;
                }
                if den.is_mul() {
                    let (ind, rest) = independent(&den, x, false)?;
                    den = rest;
                    num = div(&num, &ind)?;
                }
                let (base, exponent) = crate::core::expr::as_base_exp(&den)?;
                let Some(j) = integer_class(&exponent) else {
                    return Ok(None);
                };
                let (a, xterms) = coeff_add(&base, std::slice::from_ref(x))?;
                if is_zero(&a) {
                    sep = add(&[sep, t.clone()])?;
                    continue;
                }
                // an irreducible quadratic (kept by apart without full) gave
                // a wrong formula in SymPy; the algorithm declines
                if xterms.len() != 1 {
                    return Ok(None);
                }
                let (xc, rest) = independent(&xterms[0], x, false)?;
                if rest != *x {
                    return Ok(None);
                }
                let a: Basic = neg(&div(&a, &xc)?)?;
                num = div(&num, &pow(&xc, &int(j))?)?;
                // binomial(j + k - 1, k) rewritten with factorials
                let binom: Basic = div(
                    &factorial(&add(&[int(j - 1), k.clone()])?)?,
                    &mul(&[factorial(k)?, factorial(&int(j - 1))?])?,
                )?;
                let sign: Basic = pow(&int(-1), &int(j))?;
                let term: Basic = div(&mul(&[sign, num, binom])?, &pow(&a, &add(&[int(j), k.clone()])?)?)?;
                coeff = add(&[coeff, term])?;
            }
            if is_true(&coeff, Predicate::zero) {
                return Ok(None);
            }
            if coeff.has(x) || has_infinity(&coeff) {
                return Ok(None);
            }
            for j in 0..i {
                coeff = div(&coeff, &add(&[k.clone(), int(j as i64 + 1)])?)?;
                sep = integrate(&sep, x)?;
                let previous: Basic = ds.pop().expect("a derivative");
                // the constant of integration
                sep = add(&[sep.clone(), limit0(&sub(&previous, &sep)?, x)?])?;
            }
            let shifted: Basic = subs(&coeff, k, &sub(k, &int(i as i64))?)?;
            return Ok(Some((shifted, sep, int(i as i64))));
        }
        ds.push(derivative.clone());
    }
    Ok(None)
}

/// The terms grouped into sums whose quotients are not rational functions
/// of `x` (`rational_independent`).
pub fn rational_independent(terms: &[Basic], x: &Basic) -> PolyResult<Vec<Basic>> {
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let mut ind: Vec<Basic> = vec![terms[0].clone()];
    for t in &terms[1..] {
        let n: Basic = independent(t, x, true)?.1;
        let mut placed: bool = false;
        for term in ind.iter_mut() {
            let d: Basic = independent(term, x, true)?.1;
            let q: Basic = cancel(&div(&n, &d)?)?;
            if rational_in(&q, x) {
                *term = add(&[term.clone(), t.clone()])?;
                placed = true;
                break;
            }
        }
        if !placed {
            ind.push(t.clone());
        }
    }
    Ok(ind)
}

/// The undefined function `name`.
pub fn undefined(name: &str) -> Arc<UndefinedFunction> {
    Arc::new(UndefinedFunction::new(name, None, &[]))
}

fn apply(g: &Arc<UndefinedFunction>, arg: &Basic) -> R {
    g.apply(vec![arg.clone()])
}

/// The simple differential equations `g**(k) + sum A_j g**(j) = 0` with
/// rational coefficients which `f` satisfies, for `k` up to `order`, one
/// at a time as Python's generator yields them (`simpleDE`).
pub struct SimpleDe {
    f: Basic,
    x: Basic,
    gx: Basic,
    a: Vec<Basic>,
    order: usize,
    k: usize,
    found: bool,
}

impl SimpleDe {
    pub fn new(f: &Basic, x: &Basic, g: &Arc<UndefinedFunction>, order: usize) -> PolyResult<SimpleDe> {
        Ok(SimpleDe {
            f: f.clone(),
            x: x.clone(),
            gx: apply(g, x)?,
            a: (0..order)
                .map(|i| crate::core::symbol::Symbol::new(&format!("a{}", i), &[]).0)
                .collect(),
            order,
            k: 0,
            found: false,
        })
    }

    fn make(&self, k: usize) -> PolyResult<(Basic, Basic)> {
        let (f, x) = (&self.f, &self.x);
        let mut eq_terms: Vec<Basic> = vec![diff(f, x, k as u64)?];
        let mut de_terms: Vec<Basic> = vec![derivative_of(&self.gx, x, k as u64)];
        for (i, ai) in self.a.iter().enumerate().take(k) {
            eq_terms.push(mul(&[ai.clone(), diff(f, x, i as u64)?])?);
            de_terms.push(mul(&[ai.clone(), derivative_of(&self.gx, x, i as u64)])?);
        }
        Ok((add(&eq_terms)?, add(&de_terms)?))
    }

    /// The next equation `(DE, k)`, `None` when there are no more.
    pub fn next_de(&mut self) -> PolyResult<Option<(Basic, usize)>> {
        while self.k < self.order {
            self.k += 1;
            let k: usize = self.k;
            let (eq_, mut de) = self.make(k)?;
            let terms: Vec<Basic> = crate::core::printing::ordered_terms(&crate::core::expr::terms_of(&expand(&eq_)?));
            let ind: Vec<Basic> = rational_independent(&terms, &self.x)?;
            if self.found || ind.len() == k {
                let sol: Vec<(Basic, Basic)> = linsolve(&ind, &self.a[..k])?;
                if !sol.is_empty() {
                    self.found = true;
                    de = crate::core::subs::subs_mapping(&de, &sol)?;
                }
                de = numerator(&de)?;
                let factored: Basic = crate::polys::polytools::factor(&de, &[], false, true)?;
                de = dependent_factor_derivative(&factored)?;
                let first: Basic = crate::core::function::derivative_node(&self.gx, &[(self.x.clone(), 1)]);
                return Ok(Some((crate::simplify::radsimp::collect(&de, &[first], None, None)?, k)));
            }
        }
        Ok(None)
    }
}

/// All the equations of [`SimpleDe`] (`list(simpleDE(...))`).
pub fn simple_de(f: &Basic, x: &Basic, g: &Arc<UndefinedFunction>, order: usize) -> PolyResult<Vec<(Basic, usize)>> {
    let mut iter: SimpleDe = SimpleDe::new(f, x, g, order)?;
    let mut result: Vec<(Basic, usize)> = Vec::new();
    while let Some(item) = iter.next_de()? {
        result.push(item);
    }
    Ok(result)
}

/// The first factor of `e` with a `Derivative` (SymPy's
/// `e.as_coeff_mul(Derivative)[1][0]`) (`_dependent_factor`).
fn dependent_factor_derivative(e: &Basic) -> R {
    for f in crate::core::expr::factors_of(e) {
        if f.preorder().iter().any(|n| matches!(n.head(), Head::Derivative)) {
            return Ok(f);
        }
    }
    Err(PolyError::Value(format!("no factor of {} has Derivative", e)))
}

/// The applied unknown function of a differential equation, the first by
/// the sort key (`_applied`).
fn applied(de: &Basic) -> R {
    let mut found: Vec<Basic> = de.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Undefined(_))));
    found.sort_by_key(crate::core::sorting::sort_key);
    found
        .into_iter()
        .next()
        .ok_or_else(|| PolyError::Value(format!("no unknown function in {}", de)))
}

/// The total order of a derivative, 0 for anything else (`_order_of`).
fn order_of(d: &Basic) -> PolyResult<i64> {
    if !matches!(d.head(), Head::Derivative) {
        return Ok(0);
    }
    let mut total: i64 = 0;
    for (_, n) in crate::core::function::derivative_orders(d) {
        total += integer_class(&n).ok_or_else(|| PolyError::Value(format!("a symbolic order in {}", d)))?;
    }
    Ok(total)
}

/// The recurrence of a differential equation with constant coefficients:
/// `f**(j) -> r(k + j)`, the lowest term `r(k)` (`exp_re`).
pub fn exp_re(de: &Basic, r: &Arc<UndefinedFunction>, k: &Basic) -> R {
    let mut re: Basic = Basic::zero();
    let g: Basic = applied(de)?;
    let mut mini: Option<i64> = None;
    for t in crate::core::expr::terms_of(de) {
        let (coeff, d) = independent(&t, &g, false)?;
        let j: i64 = order_of(&d)?;
        if mini.map_or(true, |m| j < m) {
            mini = Some(j);
        }
        re = add(&[re, mul(&[coeff, apply(r, &add(&[k.clone(), int(j)])?)?])?])?;
    }
    if let Some(m) = mini.filter(|m| *m != 0) {
        re = subs(&re, k, &sub(k, &int(m))?)?;
    }
    Ok(re)
}

/// The recurrence of a differential equation with polynomial
/// coefficients: `x**l f**(j) -> rf(k + 1 - l, j) r(k + j - l)`, the
/// lowest term `r(k)` (`hyper_re`).
pub fn hyper_re(de: &Basic, r: &Arc<UndefinedFunction>, k: &Basic) -> R {
    use crate::functions::combinatorial::factorials::rf;
    let mut re: Basic = Basic::zero();
    let g: Basic = applied(de)?;
    let mut xs: Vec<Basic> = g.free_symbols().into_iter().collect();
    xs.sort_by_key(crate::core::sorting::sort_key);
    let x: Basic = xs[0].clone();
    let mut mini: Option<Basic> = None;
    for t in crate::core::expr::terms_of(&expand(de)?) {
        let (coeff, d) = independent(&t, &g, false)?;
        let (c, v) = independent(&coeff, &x, false)?;
        let l: Basic = crate::solvers::solvers::as_coeff_exponent(&v, &x)?.1;
        let j: i64 = order_of(&d)?;
        let shift: Basic = sub(&add(&[k.clone(), int(1)])?, &l)?;
        let at: Basic = sub(&add(&[k.clone(), int(j)])?, &l)?;
        re = add(&[re, mul(&[c, rf(&shift, &int(j))?, apply(r, &at)?])?])?;
        let jl: Basic = sub(&int(j), &l)?;
        let smaller: bool = match &mini {
            None => true,
            Some(m) => lt(&jl, m)?,
        };
        if smaller {
            mini = Some(jl);
        }
    }
    if let Some(m) = mini {
        re = subs(&re, k, &sub(k, &m)?)?;
    }
    let m: Basic = crate::core::symbol::Wild::new("m", Vec::new(), Vec::new(), &[]).0;
    crate::simplify::radsimp::collect(&re, &[apply(r, &add(&[k.clone(), m])?)?], None, None)
}

// -- the hypergeometric algorithm

/// The function and the recurrence `Q(k) a(k + m) = P(k) a(k)` under the
/// transformations of [`rsolve_hypergeometric`] (`_State`).
#[derive(Clone, Debug)]
struct State {
    f: Basic,
    p: Basic,
    q: Basic,
    m: Basic,
}

fn transformation_a(s: &State, x: &Basic, k: &Basic, shift: &Basic) -> PolyResult<State> {
    let k_shift: Basic = add(&[k.clone(), shift.clone()])?;
    Ok(State {
        f: mul(&[s.f.clone(), pow(x, &neg(shift)?)?])?,
        p: subs(&s.p, k, &k_shift)?,
        q: subs(&s.q, k, &k_shift)?,
        m: s.m.clone(),
    })
}

fn transformation_c(s: &State, x: &Basic, k: &Basic, scale: &Basic) -> PolyResult<State> {
    let k_scale: Basic = div(k, scale)?;
    Ok(State {
        f: subs(&s.f, x, &pow(x, scale)?)?,
        p: subs(&s.p, k, &k_scale)?,
        q: subs(&s.q, k, &k_scale)?,
        m: mul(&[s.m.clone(), scale.clone()])?,
    })
}

fn transformation_e(s: &State, x: &Basic, k: &Basic) -> PolyResult<State> {
    let k1: Basic = add(&[k.clone(), int(1)])?;
    Ok(State {
        f: diff(&s.f, x, 1)?,
        p: mul(&[subs(&s.p, k, &k1)?, add(&[k.clone(), s.m.clone(), int(1)])?])?,
        q: mul(&[subs(&s.q, k, &k1)?, k1.clone()])?,
        m: s.m.clone(),
    })
}

fn apply_shift(sol: &Solutions, shift: &Basic) -> PolyResult<Solutions> {
    sol.iter()
        .map(|(res, cond)| Ok((res.clone(), add(&[cond.clone(), shift.clone()])?)))
        .collect()
}

fn apply_scale(sol: &Solutions, scale: &Basic) -> PolyResult<Solutions> {
    sol.iter()
        .map(|(res, cond)| Ok((res.clone(), div(cond, scale)?)))
        .collect()
}

/// The term `res*x**cond` integrated is `res*x**(cond + 1)/(cond + 1)`;
/// SymPy divides by the coefficient of `k` in `cond` as well
/// (sympy/sympy#31275) (`_apply_integrate`).
fn apply_integrate(sol: &Solutions) -> PolyResult<Solutions> {
    sol.iter()
        .map(|(res, cond)| {
            let c1: Basic = add(&[cond.clone(), int(1)])?;
            Ok((div(res, &c1)?, c1))
        })
        .collect()
}

fn derivative_at_zero(f: &Basic, x: &Basic, i: i64) -> R {
    use crate::functions::combinatorial::factorials::factorial;
    div(&limit0(&diff(f, x, i as u64)?, x)?, &factorial(&int(i))?)
}

/// The formulas for `f` (`_compute_formula`).
fn compute_formula(s: &State, x: &Basic, k: &Basic, k_max: i64) -> PolyResult<Option<Solutions>> {
    use crate::functions::combinatorial::factorials::rf;
    let Some(m) = integer_class(&s.m) else {
        return Ok(None);
    };
    let mut sol: Solutions = Vec::new();
    let reciprocal: Basic = div(&int(1), k)?;
    for i in (k_max + 1)..(k_max + m + 1) {
        if i < 0 {
            continue;
        }
        let r: Basic = derivative_at_zero(&s.f, x, i)?;
        if is_true(&r, Predicate::zero) {
            continue;
        }
        let kterm: Basic = add(&[mul(&[int(m), k.clone()])?, int(i)])?;
        let p: Basic = subs(&s.p, k, &kterm)?;
        let q: Basic = subs(&s.q, k, &kterm)?;
        let c1: Basic = crate::core::series::leadterm(&subs(&p, k, &reciprocal)?, k, None, &Basic::zero())?.0;
        let c2: Basic = crate::core::series::leadterm(&subs(&q, k, &reciprocal)?, k, None, &Basic::zero())?.0;
        let mut res: Basic = mul(&[r, pow(&neg(&div(&c1, &c2)?)?, k)?])?;
        let mut up: Vec<Basic> = Vec::new();
        for (root, n) in roots_of(&p, k)? {
            up.push(pow(&rf(&neg(&root)?, k)?, &int(n as i64))?);
        }
        let mut down: Vec<Basic> = Vec::new();
        for (root, n) in roots_of(&q, k)? {
            down.push(pow(&rf(&neg(&root)?, k)?, &int(n as i64))?);
        }
        res = div(&mul(&[res, mul(&up)?])?, &mul(&down)?)?;
        sol.push((res, kterm));
    }
    Ok(Some(sol))
}

fn rational_denominator(r: &Basic) -> Option<i64> {
    let q = r.as_rational()?;
    q.denominator().to_string().parse::<i64>().ok()
}

fn lcm(a: i64, b: i64) -> i64 {
    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 {
            a.abs()
        } else {
            gcd(b, a % b)
        }
    }
    a / gcd(a, b) * b
}

fn is_unevaluated_limit(e: &Basic) -> bool {
    matches!(e.head(), Head::Limit)
}

/// The recursive part of [`rsolve_hypergeometric`]: `(formulas,
/// independent terms, highest power of x among them)` or `None`
/// (`_rsolve_hypergeometric`).
fn rsolve_hypergeometric_inner(
    f: &Basic,
    x: &Basic,
    p: &Basic,
    q: &Basic,
    k: &Basic,
    m: &Basic,
) -> PolyResult<Option<(Solutions, Basic, Basic)>> {
    use crate::functions::elementary::miscellaneous::{max_, min_};
    let mut s: State = State {
        f: f.clone(),
        p: p.clone(),
        q: q.clone(),
        m: m.clone(),
    };
    // transformation c
    let mut all_roots: Vec<Basic> = roots_of(p, k)?.into_iter().map(|(r, _)| r).collect();
    for (r, _) in roots_of(q, k)? {
        if !all_roots.contains(&r) {
            all_roots.push(r);
        }
    }
    let denominators: Vec<i64> = all_roots.iter().filter_map(rational_denominator).collect();
    let scale: Basic = int(denominators.iter().copied().fold(1, lcm));
    s = transformation_c(&s, x, k, &scale)?;
    // transformation a
    let qroots: Vec<Basic> = roots_of(&s.q, k)?.into_iter().map(|(r, _)| r).collect();
    let k_min: Basic = if qroots.is_empty() {
        Basic::zero()
    } else {
        min_(&qroots)?
    };
    let shift: Basic = add(&[k_min, s.m.clone()])?;
    s = transformation_a(&s, x, k, &shift)?;
    let l_: Basic = limit0(&mul(&[x.clone(), s.f.clone()])?, x)?;
    if !is_unevaluated_limit(&l_) && l_ != Basic::zero() {
        return Ok(None);
    }
    let qroots: Vec<Basic> = roots_of(&s.q, k)?.into_iter().map(|(r, _)| r).collect();
    let k_max_e: Basic = if qroots.is_empty() {
        Basic::zero()
    } else {
        max_(&qroots)?
    };
    let (Some(k_max), Some(m_int)) = (integer_class(&k_max_e), integer_class(&s.m)) else {
        return Ok(None);
    };
    let mut ind: Basic = Basic::zero();
    let mut mp: Basic = neg(&Basic::oo())?;
    for i in 0..(k_max + m_int + 1) {
        let r: Basic = derivative_at_zero(&s.f, x, i)?;
        if crate::core::expr::is(&r, Predicate::finite) == Some(false) {
            let old_f: Basic = s.f.clone();
            s = transformation_a(&s, x, k, &int(i))?;
            s = transformation_e(&s, x, k)?;
            let Some((sol, ind_inner, mp_inner)) = rsolve_hypergeometric_inner(&s.f, x, &s.p, &s.q, k, &s.m)? else {
                return Ok(None);
            };
            let mut sol: Solutions = apply_integrate(&sol)?;
            sol = apply_shift(&sol, &int(i))?;
            let mut ind: Basic = integrate(&ind_inner, x)?;
            // the constant of integration
            let constant: Basic = limit0(&sub(&div(&old_f, &pow(x, &int(i))?)?, &ind)?, x)?;
            if crate::core::expr::is(&constant, Predicate::finite) != Some(true) {
                return Ok(None);
            }
            ind = expand(&mul(&[add(&[ind, constant])?, pow(x, &int(i))?])?)?;
            let mut mp: Basic = add(&[mp_inner, int(1 + i)])?;
            // the series so far is that of old_f, the function after the
            // transformations c and a of this call: undone as for the
            // regular case (SymPy returns without undoing them)
            sol = apply_scale(&apply_shift(&sol, &shift)?, &scale)?;
            ind = subs(
                &expand(&mul(&[ind, pow(x, &shift)?])?)?,
                x,
                &pow(x, &div(&int(1), &scale)?)?,
            )?;
            mp = div(&add(&[mp, shift.clone()])?, &scale)?;
            return Ok(Some((sol, ind, mp)));
        }
        if !is_zero(&r) {
            let power: Basic = add(&[int(i), shift.clone()])?;
            ind = add(&[ind, mul(&[r, pow(x, &power)?])?])?;
            let pow_x: Basic = div(&power, &scale)?;
            if lt(&mp, &pow_x)? {
                // the highest power of x
                mp = pow_x;
            }
        }
    }
    ind = subs(&ind, x, &pow(x, &div(&int(1), &scale)?)?)?;
    let Some(formulas) = compute_formula(&s, x, k, k_max)? else {
        return Ok(None);
    };
    let formulas: Solutions = apply_scale(&apply_shift(&formulas, &shift)?, &scale)?;
    Ok(Some((formulas, ind, mp)))
}

/// The coefficients of the series of `f` from the recurrence `Q(k) a(k +
/// m) = P(k) a(k)` of hypergeometric type: `(formula, independent terms,
/// first index)` or `None` (`rsolve_hypergeometric`).
pub fn rsolve_hypergeometric(
    f: &Basic,
    x: &Basic,
    p: &Basic,
    q: &Basic,
    k: &Basic,
    m: &Basic,
) -> PolyResult<Option<Formula>> {
    use crate::functions::elementary::integers::{ceiling, floor, frac};
    let Some((sol_list, mut ind, mp)) = rsolve_hypergeometric_inner(f, x, p, q, k, m)? else {
        return Ok(None);
    };
    let mut sol_dict: Vec<(Basic, Basic)> = Vec::new();
    for (mut res, cond) in sol_list {
        let (j, mk) = crate::core::expr::as_coeff_add(&cond);
        let c: Basic = crate::core::expr::coeff(&mk, k)?;
        let mut jj: Basic = j.clone();
        if crate::core::expr::is(&j, Predicate::integer) == Some(false) {
            res = mul(&[res, pow(x, &frac(&j)?)?])?;
            jj = floor(&j)?;
        }
        res = subs(&res, k, &div(&sub(k, &jj)?, &c)?)?;
        let condition: Basic = crate::core::relational::eq(
            &crate::core::modulo::modulo(k, &c)?,
            &crate::core::modulo::modulo(&jj, &c)?,
        )?;
        match sol_dict.iter_mut().find(|(cnd, _)| *cnd == condition) {
            Some(entry) => entry.1 = add(&[entry.1.clone(), res])?,
            None => sol_dict.push((condition, res)),
        }
    }
    let mut pairs: Vec<(Basic, Basic)> = sol_dict.into_iter().map(|(cond, res)| (res, cond)).collect();
    pairs.push((Basic::zero(), crate::core::logic::true_()));
    let sol: Basic = crate::functions::elementary::piecewise::piecewise(&pairs)?;
    let mut s: Basic = if matches!(mp.head(), Head::NegativeInfinity) {
        Basic::zero()
    } else if crate::core::expr::is(&mp, Predicate::integer) == Some(false) {
        ceiling(&mp)?
    } else {
        add(&[mp.clone(), int(1)])?
    };
    if lt(&s, &Basic::zero())? {
        let term: Basic = mul(&[sol.clone(), pow(x, k)?])?;
        let seq = crate::series::sequences::sequence(
            &crate::series::sequences::SeqSource::Formula(term),
            Some(&crate::series::sequences::SeqLimits::Variable(
                k.clone(),
                s.clone(),
                int(-1),
            )),
        )?;
        let mut terms: Vec<Basic> = vec![ind];
        terms.extend(seq.slice(0, None, 1)?);
        ind = add(&terms)?;
        s = Basic::zero();
    }
    Ok(Some((sol, ind, s)))
}

/// The applications of `g` in a recurrence, by the sort key
/// (`_function_terms`).
fn function_terms(re: &Basic, g: &Arc<UndefinedFunction>) -> Vec<Basic> {
    let head: Head = Head::Undefined(g.clone());
    let mut found: Vec<Basic> = re.atoms(Some(&|n: &Basic| n.head().same_head(&head)));
    found.sort_by_key(crate::core::sorting::sort_key);
    found
}

/// See [`rsolve_hypergeometric`] (`_solve_hyper_RE`).
fn solve_hyper_re(
    f: &Basic,
    x: &Basic,
    re: &Basic,
    g: &Arc<UndefinedFunction>,
    k: &Basic,
) -> PolyResult<Option<Formula>> {
    let terms: Vec<Basic> = crate::core::expr::terms_of(re);
    if terms.len() != 2 {
        return Ok(None);
    }
    let gs: Vec<Basic> = function_terms(re, g);
    if gs.len() != 2 {
        return Ok(None);
    }
    let (mut p, mut q) = (
        crate::core::expr::coeff(re, &gs[0])?,
        crate::core::expr::coeff(re, &gs[1])?,
    );
    let mut m: Basic = sub(&gs[1].args()[0], &gs[0].args()[0])?;
    let Some(mv) = integer_class(&m) else {
        return Ok(None);
    };
    if mv < 0 {
        std::mem::swap(&mut p, &mut q);
        m = int(-mv);
    }
    rsolve_hypergeometric(f, x, &p, &q, k, &m)
}

/// The values `g(i)` of the first `n` derivatives at 0, each divided by
/// `i!` when `scaled` (the initial values of `_solve_explike_DE` and
/// `_solve_simple`).
fn initial_values(
    f: &Basic,
    x: &Basic,
    g: &Arc<UndefinedFunction>,
    n: usize,
    scaled: bool,
) -> PolyResult<Vec<(Basic, Basic)>> {
    use crate::functions::combinatorial::factorials::factorial;
    let mut init: Vec<(Basic, Basic)> = Vec::new();
    let mut f: Basic = f.clone();
    for i in 0..n {
        if i > 0 {
            f = diff(&f, x, 1)?;
        }
        let mut value: Basic = limit0(&f, x)?;
        if scaled {
            value = div(&value, &factorial(&int(i as i64))?)?;
        }
        init.push((apply(g, &int(i as i64))?, value));
    }
    Ok(init)
}

/// Solves a differential equation with constant coefficients
/// (`_solve_explike_DE`).
fn solve_explike_de(
    f: &Basic,
    x: &Basic,
    de: &Basic,
    g: &Arc<UndefinedFunction>,
    k: &Basic,
) -> PolyResult<Option<Formula>> {
    use crate::functions::combinatorial::factorials::factorial;
    let gx: Basic = applied(de)?;
    for t in crate::core::expr::terms_of(de) {
        let (coeff, _) = independent(&t, &gx, false)?;
        if !coeff.free_symbols().is_empty() {
            return Ok(None);
        }
    }
    let re: Basic = exp_re(de, g, k)?;
    let init: Vec<(Basic, Basic)> = initial_values(f, x, g, crate::core::expr::terms_of(&re).len(), false)?;
    match rsolve(&re, &apply(g, k)?, &init)? {
        Some(sol) if sol != Basic::zero() => Ok(Some((div(&sol, &factorial(k)?)?, Basic::zero(), Basic::zero()))),
        _ => Ok(None),
    }
}

/// Turns the differential equation into a recurrence solved by `rsolve`
/// (`_solve_simple`).
fn solve_simple(
    f: &Basic,
    x: &Basic,
    de: &Basic,
    g: &Arc<UndefinedFunction>,
    k: &Basic,
) -> PolyResult<Option<Formula>> {
    let re: Basic = hyper_re(de, g, k)?;
    let init: Vec<(Basic, Basic)> = initial_values(f, x, g, crate::core::expr::terms_of(&re).len(), true)?;
    match rsolve(&re, &apply(g, k)?, &init)? {
        Some(sol) if sol != Basic::zero() => Ok(Some((sol, Basic::zero(), Basic::zero()))),
        _ => Ok(None),
    }
}

fn sorted_symbols(syms: &[Basic]) -> Vec<Basic> {
    let mut v: Vec<Basic> = syms.to_vec();
    v.sort_by_key(crate::core::sorting::sort_key);
    v
}

/// `e.subs(dict(sol))`.
fn subs_solution(e: &Basic, sol: Vec<(Basic, Basic)>) -> R {
    crate::core::subs::subs_mapping(e, &sol)
}

/// The first factor of `e` with an applied undefined function (`_dependent_factor(e, AppliedUndef)`).
fn dependent_factor_applied(e: &Basic) -> R {
    for f in crate::core::expr::factors_of(e) {
        if f.preorder().iter().any(|n| matches!(n.head(), Head::Undefined(_))) {
            return Ok(f);
        }
    }
    Err(PolyError::Value(format!("no factor of {} has AppliedUndef", e)))
}

fn factor(e: &Basic) -> R {
    crate::polys::polytools::factor(e, &[], false, true)
}

/// Turns a differential equation with free parameters into one with
/// constant coefficients (`_transform_explike_DE`).
fn transform_explike_de(de: &Basic, g: &Arc<UndefinedFunction>, x: &Basic, order: usize, syms: &[Basic]) -> R {
    let gx: Basic = apply(g, x)?;
    let mut eqs: Vec<Basic> = Vec::new();
    let highest_coeff: Basic = crate::core::expr::coeff(de, &derivative_of(&gx, x, order as u64))?;
    for i in 0..order {
        let coeff: Basic = crate::core::expr::coeff(de, &derivative_of(&gx, x, i as u64))?;
        let collected: Basic = crate::simplify::radsimp::collect(
            &expand(&div(&coeff, &highest_coeff)?)?,
            std::slice::from_ref(x),
            None,
            None,
        )?;
        eqs.extend(crate::core::expr::terms_of(&collected));
    }
    let mut temp: Vec<Basic> = Vec::new();
    let mut broke: bool = false;
    for e in &eqs {
        if e.has(x) {
            broke = true;
            break;
        }
        if e.preorder().iter().any(|n| n.is_symbol()) {
            temp.push(e.clone());
        }
    }
    if !broke {
        eqs = temp;
    }
    let mut de: Basic = de.clone();
    if !eqs.is_empty() {
        let sol: Vec<(Basic, Basic)> = linsolve(&eqs, &sorted_symbols(syms))?;
        if !sol.is_empty() {
            de = subs_solution(&de, sol)?;
            de = dependent_factor_derivative(&factor(&de)?)?;
            let first: Basic = crate::core::function::derivative_node(&gx, &[(x.clone(), 1)]);
            de = crate::simplify::radsimp::collect(&de, &[first], None, None)?;
        }
    }
    Ok(de)
}

/// Turns a differential equation with free parameters into a recurrence
/// of hypergeometric type (`_transform_DE_RE`).
fn transform_de_re(de: &Basic, g: &Arc<UndefinedFunction>, k: &Basic, order: usize, syms: &[Basic]) -> R {
    let mut re: Basic = hyper_re(de, g, k)?;
    let mut eqs: Vec<Basic> = Vec::new();
    for i in 1..order {
        eqs.push(crate::core::expr::coeff(
            &re,
            &apply(g, &add(&[k.clone(), int(i as i64)])?)?,
        )?);
    }
    let sol: Vec<(Basic, Basic)> = linsolve(&eqs, &sorted_symbols(syms))?;
    if !sol.is_empty() {
        let m: Basic = crate::core::symbol::Wild::new("m", Vec::new(), Vec::new(), &[]).0;
        re = subs_solution(&re, sol)?;
        re = crate::simplify::radsimp::collect(
            &numerator(&factor(&re)?)?,
            &[apply(g, &add(&[k.clone(), m])?)?],
            None,
            None,
        )?;
        re = dependent_factor_applied(&re)?;
        for i in 1..order {
            if crate::core::expr::coeff(&re, &apply(g, &add(&[k.clone(), int(i as i64)])?)?)? != Basic::zero() {
                re = subs(&re, k, &sub(k, &int(i as i64))?)?;
                break;
            }
        }
    }
    Ok(re)
}

fn free_except(e: &Basic, x: &Basic) -> Vec<Basic> {
    e.free_symbols().into_iter().filter(|s| s != x).collect()
}

/// The coefficients of the series of `f` from its differential equation
/// `DE`: by a recurrence of two terms, or as an equation with constant
/// coefficients (`solve_de`).
pub fn solve_de(
    f: &Basic,
    x: &Basic,
    de: &Basic,
    order: usize,
    g: &Arc<UndefinedFunction>,
    k: &Basic,
) -> PolyResult<Option<Formula>> {
    let syms: Vec<Basic> = free_except(de, x);
    let re: Basic = if syms.is_empty() {
        hyper_re(de, g, k)?
    } else {
        transform_de_re(de, g, k, order, &syms)?
    };
    if free_except(&re, k).is_empty() {
        if let Some(sol) = solve_hyper_re(f, x, &re, g, k)? {
            return Ok(Some(sol));
        }
    }
    let de: Basic = if syms.is_empty() {
        de.clone()
    } else {
        transform_explike_de(de, g, x, order, &syms)?
    };
    if free_except(&de, x).is_empty() {
        return solve_explike_de(f, x, &de, g, k);
    }
    Ok(None)
}

/// The coefficients of the series of `f` by the hypergeometric algorithm:
/// a differential equation, its recurrence, the solution of the
/// recurrence (`hyper_algorithm`).
pub fn hyper_algorithm(f: &Basic, x: &Basic, k: &Basic, order: usize) -> PolyResult<Option<Formula>> {
    let g: Arc<UndefinedFunction> = undefined("g");
    let mut des: Vec<Basic> = Vec::new();
    // simpleDE is a generator in Python: the equations come one at a time,
    // and the first solution stops it
    let mut equations: SimpleDe = SimpleDe::new(f, x, &g, order)?;
    while let Some((de, i)) = equations.next_de()? {
        if let Some(sol) = solve_de(f, x, &de, i, &g, k)? {
            return Ok(Some(sol));
        }
        if free_except(&de, x).is_empty() {
            des.push(de);
        }
    }
    // nothing works: plain rsolve
    for de in &des {
        if let Some(sol) = solve_simple(f, x, de, &g, k)? {
            return Ok(Some(sol));
        }
    }
    Ok(None)
}

// -- the driver

use crate::series::sequences::{Seq, SeqFormula, SeqLimits, SeqSource};

/// The coefficients, the powers and the independent terms of a series
/// (`SeriesTerms`).
pub type SeriesTerms = (SeqFormula, SeqFormula, Basic);

fn as_formula_seq(s: Seq) -> PolyResult<SeqFormula> {
    match s {
        Seq::Formula(f) => Ok(f),
        other => Err(PolyError::Type(format!(
            "a sequence of a formula is expected, got {}",
            other
        ))),
    }
}

/// `SeqFormula.subs(old, new)`: the bound variable stays.
fn seq_subs(s: &SeqFormula, old: &Basic, new: &Basic) -> PolyResult<SeqFormula> {
    if *old == s.variable {
        return Ok(s.clone());
    }
    Ok(SeqFormula {
        formula: subs(&s.formula, old, new)?,
        variable: s.variable.clone(),
        start: subs(&s.start, old, new)?,
        stop: subs(&s.stop, old, new)?,
    })
}

fn formula_from(formula: &Basic, k: &Basic, start: &Basic, stop: &Basic) -> PolyResult<SeqFormula> {
    SeqFormula::new(
        formula,
        Some(&SeqLimits::Variable(k.clone(), start.clone(), stop.clone())),
    )
}

/// The coefficients `ak` moved `shift` places up: `a(k - shift)` from
/// `start + shift` on (`_shifted`).
fn shifted(ak: &SeqFormula, shift: i64) -> PolyResult<SeqFormula> {
    if shift == 0 {
        return Ok(ak.clone());
    }
    let k: &Basic = &ak.variable;
    formula_from(
        &subs(&ak.formula, k, &sub(k, &int(shift))?)?,
        k,
        &add(&[ak.start.clone(), int(shift)])?,
        &ak.stop,
    )
}

fn powsimp(e: &Basic) -> R {
    crate::simplify::powsimp::powsimp_default(e)
}

/// The integer `j` with `xk(k) == common(k + j)` for the powers `x**(k +
/// c)` of two series, `None` when they do not differ by an integer power
/// of `x` (`_power_shift`).
fn power_shift(xk: &SeqFormula, common: &SeqFormula, x: &Basic) -> PolyResult<Option<i64>> {
    if xk.formula == common.formula
        && xk.variable == common.variable
        && xk.start == common.start
        && xk.stop == common.stop
    {
        return Ok(Some(0));
    }
    let k: Basic = crate::core::symbol::Dummy::new("k", &[]).0;
    let at = |s: &SeqFormula| -> R { Seq::Formula(s.clone()).coeff(&k) };
    let ratio: Basic = powsimp(&div(&at(xk)?, &at(common)?)?)?;
    if ratio == Basic::one() {
        return Ok(Some(0));
    }
    let (base, exponent) = crate::core::expr::as_base_exp(&ratio)?;
    Ok(if base == *x { integer_class(&exponent) } else { None })
}

/// `e.as_independent(*syms, as_Add=False)`: the factors free of all the
/// symbols and the others.
fn independent_of_all(e: &Basic, syms: &[Basic]) -> PolyResult<(Basic, Basic)> {
    let depends = |t: &Basic| syms.iter().any(|s| t.has(s));
    if !e.is_mul() {
        return Ok(if depends(e) {
            (Basic::one(), e.clone())
        } else {
            (e.clone(), Basic::one())
        });
    }
    let (mut free, mut dependent): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for f in e.args() {
        if depends(f) {
            dependent.push(f.clone());
        } else {
            free.push(f.clone());
        }
    }
    Ok((mul(&free)?, mul(&dependent)?))
}

fn constant_term(p: &Basic, x: &Basic) -> R {
    let terms: Vec<Basic> = crate::core::expr::terms_of(&expand(p)?)
        .into_iter()
        .filter(|t| !t.has(x))
        .collect();
    add(&terms)
}

/// See [`compute_fps`] (`_compute_fps`).
#[allow(clippy::too_many_arguments)]
fn compute_fps_inner(
    f: &Basic,
    x: &Basic,
    x0: &Basic,
    dir: i64,
    hyper: bool,
    order: usize,
    rational: bool,
    full: bool,
) -> PolyResult<Option<SeriesTerms>> {
    let one_over_x: Basic = div(&int(1), x)?;
    if matches!(x0.head(), Head::Infinity | Head::NegativeInfinity) {
        let d: i64 = if matches!(x0.head(), Head::Infinity) { 1 } else { -1 };
        let temp: Basic = subs(f, x, &one_over_x)?;
        let Some((ak, xk, ind)) = compute_fps_inner(&temp, x, &Basic::zero(), d, hyper, order, rational, full)? else {
            return Ok(None);
        };
        return Ok(Some((ak, seq_subs(&xk, x, &one_over_x)?, subs(&ind, x, &one_over_x)?)));
    }
    if *x0 != Basic::zero() || dir == -1 {
        let (rep, back): (Basic, Basic) = if dir == -1 {
            (add(&[neg(x)?, x0.clone()])?, add(&[neg(x)?, x0.clone()])?)
        } else {
            (add(&[x.clone(), x0.clone()])?, sub(x, x0)?)
        };
        let temp: Basic = subs(f, x, &rep)?;
        let Some((ak, xk, ind)) = compute_fps_inner(&temp, x, &Basic::zero(), 1, hyper, order, rational, full)? else {
            return Ok(None);
        };
        return Ok(Some((ak, seq_subs(&xk, x, &back)?, subs(&ind, x, &back)?)));
    }
    let polynomial: bool = crate::core::expr::polynomial_like(
        f,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true);
    if polynomial {
        let k: Basic = crate::core::symbol::Dummy::new("k", &[]).0;
        let ak: SeqFormula = formula_from(&coeff_node(f, x, &k), &k, &int(1), &Basic::oo())?;
        let xk: SeqFormula = formula_from(&pow(x, &k)?, &k, &Basic::zero(), &Basic::oo())?;
        return Ok(Some((ak, xk, constant_term(f, x)?)));
    }
    // the terms of a sum separately: the different algorithms on different
    // terms widen the range of the functions
    if f.is_add() {
        let mut found: bool = false;
        let mut ak_seq: Seq = Seq::Formula(SeqFormula::new(
            &Basic::zero(),
            Some(&SeqLimits::Range(Basic::zero(), Basic::oo())),
        )?);
        let mut xk_found: Option<SeqFormula> = None;
        let mut sum_ind: Basic = Basic::zero();
        for t in crate::core::expr::terms_of(f) {
            let Some((res_ak, res_xk, res_ind)) =
                compute_fps_inner(&t, x, &Basic::zero(), 1, hyper, order, rational, full)?
            else {
                sum_ind = add(&[sum_ind, t])?;
                continue;
            };
            if !found {
                found = true;
                xk_found = Some(res_xk.clone());
            }
            let common: SeqFormula = xk_found.clone().expect("found");
            // the coefficients of two terms are added on the same powers of
            // x only (sympy/sympy#31275)
            let mut new_ak: SeqFormula = res_ak;
            let Some(shift) = power_shift(&res_xk, &common, x)? else {
                return Ok(None);
            };
            if shift < 0 {
                ak_seq = Seq::Formula(shifted(&as_formula_seq(ak_seq)?, -shift)?);
                xk_found = Some(res_xk.clone());
            } else {
                new_ak = shifted(&new_ak, shift)?;
            }
            let new_seq: Seq = Seq::Formula(new_ak);
            let (s, e, seq) = if lt(&ak_seq.start()?, &new_seq.start()?)? {
                (ak_seq.start()?, new_seq.start()?, ak_seq.clone())
            } else {
                (new_seq.start()?, ak_seq.start()?, new_seq.clone())
            };
            let (Some(si), Some(ei)) = (integer_class(&s), integer_class(&e)) else {
                return Ok(None);
            };
            let xk_seq: Seq = Seq::Formula(xk_found.clone().expect("found"));
            let firsts: Vec<Basic> = seq.slice(0, Some(ei - si), 1)?;
            let powers: Vec<Basic> = xk_seq.slice(si, Some(ei), 1)?;
            let products: Vec<Basic> = firsts
                .iter()
                .zip(&powers)
                .map(|(a, b)| mul(&[a.clone(), b.clone()]))
                .collect::<PolyResult<_>>()?;
            let save: Basic = add(&products)?;
            ak_seq = ak_seq.plus(&new_seq)?;
            sum_ind = add(&[sum_ind, res_ind, save])?;
        }
        if let (true, Some(xk)) = (found, xk_found) {
            return Ok(Some((as_formula_seq(ak_seq)?, xk, sum_ind)));
        }
        return Ok(None);
    }
    // the symbolic factor, if any, comes out of the function
    let syms: Vec<Basic> = free_except(f, x);
    let (f, mut symb) = {
        let (free, dependent) = independent_of_all(&expand(f)?, &syms)?;
        // as_independent(*syms) puts the factors free of syms first: the
        // function is that part
        (free, dependent)
    };
    // from here on x0 = 0 and dir = 1
    let k: Basic = crate::core::symbol::Dummy::new("k", &[]).0;
    let mut formula: Option<Formula> = None;
    if rational {
        formula = rational_algorithm(&f, x, &k, order, full)?;
    }
    if formula.is_none() && hyper {
        formula = hyper_algorithm(&f, x, &k, order)?;
    }
    let Some((coefficients, independent_terms, first_index)) = formula else {
        return Ok(None);
    };
    symb = if is_true(&symb, Predicate::zero) {
        Basic::one()
    } else {
        powsimp(&symb)?
    };
    let first: Seq = crate::series::sequences::sequence(
        &SeqSource::Formula(coefficients),
        Some(&SeqLimits::Variable(k.clone(), first_index, Basic::oo())),
    )?;
    if matches!(first, Seq::Empty) {
        return Ok(None);
    }
    let ak: SeqFormula = as_formula_seq(first)?;
    let xk: SeqFormula = formula_from(
        &powsimp(&mul(&[pow(x, &k)?, symb.clone()])?)?,
        &k,
        &Basic::zero(),
        &Basic::oo(),
    )?;
    let ind: Basic = powsimp(&mul(&[independent_terms, symb])?)?;
    Ok(Some((ak, xk, ind)))
}

/// The coefficients `ak`, the powers `xk` (sequences) and the independent
/// terms of the formal power series of `f` at `x0` (from the right for
/// `dir = 1`, from the left for `-1`), by the rational algorithm, then the
/// hypergeometric one; `None` when neither applies (`compute_fps`).
#[allow(clippy::too_many_arguments)]
pub fn compute_fps(
    f: &Basic,
    x: &Basic,
    x0: &Basic,
    dir: i64,
    hyper: bool,
    order: usize,
    rational: bool,
    full: bool,
) -> PolyResult<Option<SeriesTerms>> {
    if dir != 1 && dir != -1 {
        return Err(PolyError::Value("Dir must be '+' or '-'".to_string()));
    }
    if !f.has(x) {
        return Ok(None);
    }
    compute_fps_inner(f, x, x0, dir, hyper, order, rational, full)
}

// -- Coeff

/// The node `Coeff(p, x, n)`, unevaluated.
pub fn coeff_node(p: &Basic, x: &Basic, n: &Basic) -> Basic {
    Basic::new(
        Head::Function(crate::core::basic::Function::Coeff),
        vec![p.clone(), x.clone(), n.clone()],
    )
}

/// `Coeff.eval`: the `n`-th coefficient of the polynomial `p` in the
/// symbol `x` for an integer `n`, the node otherwise.
pub fn coeff_eval(p: &Basic, x: &Basic, n: &Basic) -> R {
    let polynomial: bool = x.is_symbol()
        && crate::core::expr::polynomial_like(
            p,
            std::slice::from_ref(x),
            crate::core::expr::PolynomialKind::Polynomial,
        ) == Some(true);
    if let (Some(nv), true) = (integer_class(n), polynomial) {
        if nv < 0 {
            return Ok(Basic::zero());
        }
        if nv == 0 {
            return constant_term(p, x);
        }
        return crate::core::expr::coeff(p, &pow(x, &int(nv))?);
    }
    Ok(coeff_node(p, x, n))
}

/// `Coeff(p, x, n)`, evaluated (`coeff_of`).
pub fn coeff_of(p: &Basic, x: &Basic, n: &Basic) -> R {
    coeff_eval(p, x, n)
}

// -- the series

fn bell_incomplete_poly(
    n: usize,
    k: usize,
    values: &[Basic],
    memo: &mut std::collections::HashMap<(usize, usize), Basic>,
) -> R {
    if n == 0 && k == 0 {
        return Ok(Basic::one());
    }
    if n == 0 || k == 0 {
        return Ok(Basic::zero());
    }
    if let Some(v) = memo.get(&(n, k)) {
        return Ok(v.clone());
    }
    let mut s: Basic = Basic::zero();
    let mut a: Basic = Basic::one();
    for m in 1..(n - k + 2) {
        let inner: Basic = bell_incomplete_poly(n - m, k - 1, values, memo)?;
        s = add(&[s, mul(&[a.clone(), inner, values[m - 1].clone()])?])?;
        a = mul(&[a, div(&int((n - m) as i64), &int(m as i64))?])?;
    }
    let result: Basic = crate::core::function::expand_mul(&s, true)?;
    memo.insert((n, k), result.clone());
    Ok(result)
}

/// The incomplete (partial) Bell polynomial `B_{n,k}(x_1, x_2, ...)` at
/// `values` (SymPy's `bell(n, k, values)`) (`bell_incomplete`).
pub fn bell_incomplete(n: usize, k: usize, values: &[Basic]) -> R {
    bell_incomplete_poly(n, k, values, &mut std::collections::HashMap::new())
}

/// The formal power series of a function: its formula of coefficients
/// `ak`, the powers `xk` and the independent terms `ind`. No computation is
/// done: [`fps`] computes the series (`FormalPowerSeries`).
#[derive(Clone, Debug)]
pub struct FormalPowerSeries {
    pub function: Basic,
    pub x: Basic,
    pub x0: Basic,
    /// SymPy's `dir`: `1` from the right, `-1` from the left.
    pub direction: Basic,
    pub ak: SeqFormula,
    pub xk: SeqFormula,
    pub ind: Basic,
}

use crate::series::series_class::SeriesBase;

impl SeriesBase for FormalPowerSeries {
    fn start(&self) -> R {
        Ok(Basic::zero())
    }
    fn stop(&self) -> R {
        Ok(Basic::oo())
    }
    fn length(&self) -> R {
        Ok(Basic::oo())
    }
    fn eval_term(&self, pt: &Basic) -> R {
        let mut term: Basic = match (
            Seq::Formula(self.xk.clone()).coeff(pt),
            Seq::Formula(self.ak.clone()).coeff(pt),
        ) {
            (Ok(pt_xk), Ok(pt_ak)) => {
                // the coefficients simplified
                let simplified: Basic =
                    crate::simplify::simplify::simplify(&pt_ak, &crate::simplify::simplify::Options::default())?;
                mul(&[simplified, pt_xk])?
            }
            (Err(PolyError::Index(_)), _) | (_, Err(PolyError::Index(_))) => Basic::zero(),
            (Err(e), _) | (_, Err(e)) => return Err(e),
        };
        if !is_zero(&self.ind) {
            let mut ind: Basic = Basic::zero();
            let sym: Vec<Basic> = self.free_symbols();
            for t in crate::core::expr::terms_of(&self.ind) {
                let mut pow_x: Basic = self.get_pow_x(&t)?;
                if sym.iter().any(|s| pow_x.has(s)) {
                    pow_x = coeff_add(&pow_x, &sym)?.0;
                }
                let next: Basic = add(&[pt.clone(), int(1)])?;
                if (*pt == Basic::zero() && lt(&pow_x, &int(1))?) || (!lt(&pow_x, pt)? && lt(&pow_x, &next)?) {
                    ind = add(&[ind, t.clone()])?;
                }
            }
            term = add(&[term, ind])?;
        }
        crate::simplify::radsimp::collect(&term, std::slice::from_ref(&self.x), None, None)
    }
}

impl FormalPowerSeries {
    /// The free symbols of the series: those of its arguments but the
    /// bound variables of the sequences (`free_symbols`).
    pub fn free_symbols(&self) -> Vec<Basic> {
        let mut found: Vec<Basic> = Vec::new();
        let mut push = |e: &Basic, bound: Option<&Basic>| {
            for s in e.free_symbols() {
                if Some(&s) != bound && !found.contains(&s) {
                    found.push(s);
                }
            }
        };
        push(&self.function, None);
        push(&self.x, None);
        push(&self.x0, None);
        for s in [&self.ak, &self.xk] {
            push(&s.formula, Some(&s.variable));
            push(&s.start, None);
            push(&s.stop, None);
        }
        push(&self.ind, None);
        found.sort_by_key(crate::core::sorting::sort_key);
        found
    }

    fn k(&self) -> &Basic {
        &self.ak.variable
    }

    /// The coefficients from 1 on (`ak_seq`).
    pub fn ak_seq(&self) -> PolyResult<SeqFormula> {
        formula_from(&self.ak.formula, self.k(), &int(1), &Basic::oo())
    }

    /// `factorial(k)` from 1 on (`fact_seq`).
    pub fn fact_seq(&self) -> PolyResult<SeqFormula> {
        formula_from(
            &crate::functions::combinatorial::factorials::factorial(self.k())?,
            self.k(),
            &int(1),
            &Basic::oo(),
        )
    }

    /// `ak_seq * fact_seq` (`bell_coeff_seq`).
    pub fn bell_coeff_seq(&self) -> PolyResult<Seq> {
        Seq::Formula(self.ak_seq()?).times(&Seq::Formula(self.fact_seq()?))
    }

    /// The periodic signs `-1, 1, ...` from 1 on (`sign_seq`).
    pub fn sign_seq(&self) -> PolyResult<Seq> {
        crate::series::sequences::sequence(
            &SeqSource::Periodical(vec![int(-1), int(1)]),
            Some(&SeqLimits::Variable(self.k().clone(), int(1), Basic::oo())),
        )
    }

    /// The infinite representation of the series, a `Sum` (`infinite`).
    pub fn infinite(&self) -> R {
        let term: Basic = mul(&[self.ak.formula.clone(), self.xk.formula.clone()])?;
        // the class Sum, unevaluated (the function sum_ folds the Piecewise)
        let limits = (self.k().clone(), self.ak.start.clone(), self.ak.stop.clone());
        add(&[
            self.ind.clone(),
            crate::concrete::expr_with_limits::node(Head::Sum, &term, &[limits])?,
        ])
    }

    /// The power of `x` in a term (`_get_pow_x`).
    fn get_pow_x(&self, term: &Basic) -> R {
        let dependent: Basic = independent(term, &self.x, false)?.1;
        let (xterm, pow_x) = crate::core::expr::as_base_exp(&dependent)?;
        if !xterm.has(&self.x) {
            return Ok(Basic::zero());
        }
        Ok(pow_x)
    }

    /// The series truncated before `x**n`, without the order term
    /// (`polynomial`).
    pub fn polynomial(&self, n: i64) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        let sym: Vec<Basic> = self.free_symbols();
        let mut i: i64 = 0;
        loop {
            let t: Basic = self.item(i)?;
            let mut xp: Basic = self.get_pow_x(&t)?;
            if sym.iter().any(|s| xp.has(s)) {
                xp = coeff_add(&xp, &sym)?.0;
            }
            if !lt(&xp, &int(n))? {
                break;
            }
            if crate::core::expr::is(&xp, Predicate::integer) == Some(true) && i == n + 1 {
                break;
            }
            if !is_zero(&t) {
                terms.push(t);
            }
            i += 1;
        }
        add(&terms)
    }

    /// The series truncated at `O(x**n)` (`truncate`).
    pub fn truncate(&self, n: i64) -> R {
        let x0: Basic = if matches!(self.x0.head(), Head::NegativeInfinity) {
            Basic::oo()
        } else {
            self.x0.clone()
        };
        let pt_xk: Basic = Seq::Formula(self.xk.clone()).coeff(&int(n))?;
        let o: Basic = crate::core::order::order(&pt_xk, &[(self.x.clone(), x0)])?;
        add(&[self.polynomial(n)?, o])
    }

    /// The term of index 0 (`zero_coeff`).
    pub fn zero_coeff(&self) -> R {
        self.eval_term(&Basic::zero())
    }

    /// The first nonzero term (`_eval_as_leading_term`).
    pub fn leading_term(&self) -> R {
        let mut i: i64 = 0;
        loop {
            let t: Basic = self.item(i)?;
            if !is_zero(&t) {
                return Ok(t);
            }
            i += 1;
        }
    }

    fn ak_formula_pieces(&self) -> Vec<(Basic, Basic)> {
        let formula: &Basic = &self.ak.formula;
        if matches!(formula.head(), Head::Piecewise) {
            return formula
                .args()
                .iter()
                .filter(|p| matches!(p.head(), Head::ExprCondPair))
                .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
                .collect();
        }
        vec![(formula.clone(), crate::core::logic::true_())]
    }

    /// The derivative of the series in `x` (`_eval_derivative`).
    pub fn derivative(&self, x: &Basic) -> PolyResult<FormalPowerSeries> {
        let f: Basic = diff(&self.function, x, 1)?;
        let ind: Basic = diff(&self.ind, x, 1)?;
        let pow_xk: Basic = self.get_pow_x(&self.xk.formula)?;
        let ak: &SeqFormula = &self.ak;
        let k: &Basic = &ak.variable;
        let k1: Basic = add(&[k.clone(), int(1)])?;
        let start: Basic = sub(&ak.start, &int(1))?;
        let formula: Basic = if ak.formula.has(x) {
            let mut form: Vec<(Basic, Basic)> = Vec::new();
            for (e, c) in self.ak_formula_pieces() {
                let mut temp: Basic = Basic::zero();
                for t in crate::core::expr::terms_of(&e) {
                    let pow_x: Basic = self.get_pow_x(&t)?;
                    temp = add(&[temp, mul(&[t.clone(), add(&[pow_xk.clone(), pow_x])?])?])?;
                }
                form.push((temp, c));
            }
            subs(&crate::functions::elementary::piecewise::piecewise(&form)?, k, &k1)?
        } else {
            subs(&mul(&[ak.formula.clone(), pow_xk])?, k, &k1)?
        };
        let new: Seq = crate::series::sequences::sequence(
            &SeqSource::Formula(formula),
            Some(&SeqLimits::Variable(k.clone(), start, ak.stop.clone())),
        )?;
        Ok(FormalPowerSeries {
            function: f,
            x: self.x.clone(),
            x0: self.x0.clone(),
            direction: self.direction.clone(),
            ak: as_formula_seq(new)?,
            xk: self.xk.clone(),
            ind,
        })
    }
}

/// What the operations on a series give: an expression (the function
/// when it does not depend on `x`), or a series.
#[derive(Clone, Debug)]
pub enum Fps {
    Expr(Basic),
    Series(FormalPowerSeries),
}

impl FormalPowerSeries {
    fn with(&self, function: Basic, ak: SeqFormula, xk: SeqFormula, ind: Basic) -> FormalPowerSeries {
        FormalPowerSeries {
            function,
            x: self.x.clone(),
            x0: self.x0.clone(),
            direction: self.direction.clone(),
            ak,
            xk,
            ind,
        }
    }

    /// The indefinite integral of the series in `x` (`integrate(x)`).
    pub fn integrate(&self, x: &Basic) -> PolyResult<FormalPowerSeries> {
        let f: Basic = integrate(&self.function, x)?;
        let mut ind: Basic = integrate(&self.ind, x)?;
        // the constant of integration
        ind = add(&[ind.clone(), limit0(&sub(&f, &ind)?, x)?])?;
        let pow_xk: Basic = self.get_pow_x(&self.xk.formula)?;
        let ak: &SeqFormula = &self.ak;
        let k: &Basic = &ak.variable;
        let k1: Basic = sub(k, &int(1))?;
        let start: Basic = add(&[ak.start.clone(), int(1)])?;
        let formula: Basic = if ak.formula.has(x) {
            let mut form: Vec<(Basic, Basic)> = Vec::new();
            for (e, c) in self.ak_formula_pieces() {
                let mut temp: Basic = Basic::zero();
                for t in crate::core::expr::terms_of(&e) {
                    let pow_x: Basic = self.get_pow_x(&t)?;
                    temp = add(&[temp, div(&t, &add(&[pow_xk.clone(), pow_x, int(1)])?)?])?;
                }
                form.push((temp, c));
            }
            subs(&crate::functions::elementary::piecewise::piecewise(&form)?, k, &k1)?
        } else {
            subs(&div(&ak.formula, &add(&[pow_xk, int(1)])?)?, k, &k1)?
        };
        let new: Seq = crate::series::sequences::sequence(
            &SeqSource::Formula(formula),
            Some(&SeqLimits::Variable(k.clone(), start, ak.stop.clone())),
        )?;
        Ok(self.with(f, as_formula_seq(new)?, self.xk.clone(), ind))
    }

    /// The definite integral of the function over `(x, a, b)`
    /// (`integrate((x, a, b))`).
    pub fn integrate_definite(&self, x: &Basic, a: &Basic, b: &Basic) -> R {
        crate::integrals::definite::definite_integral(
            &self.function,
            x,
            a,
            b,
            &[],
            &crate::integrals::definite::Options::default(),
        )
    }

    /// The sum with another series (`__add__`).
    pub fn plus(&self, o: &FormalPowerSeries) -> PolyResult<Fps> {
        if self.direction != o.direction {
            return Err(PolyError::Value(
                "Both series should be calculated from the same direction.".to_string(),
            ));
        }
        if self.x0 != o.x0 {
            return Err(PolyError::Value(
                "Both series should be calculated about the same point.".to_string(),
            ));
        }
        let (x, y) = (&self.x, &o.x);
        let f: Basic = add(&[self.function.clone(), subs(&o.function, y, x)?])?;
        if !f.free_symbols().contains(x) {
            return Ok(Fps::Expr(f));
        }
        // on the same powers of x only (see compute_fps)
        let shift: Option<i64> = power_shift(&seq_subs(&o.xk, y, x)?, &self.xk, x)?;
        let shift: i64 = match shift {
            Some(s) if s >= 0 => s,
            Some(_) if y == x => return o.plus(self),
            // an unevaluated Add of the two series in Python
            _ => {
                return Err(crate::error::not_ported(
                    "the sum of two formal power series on other powers",
                ))
            }
        };
        let other_ak: SeqFormula = shifted(&o.ak, shift)?;
        let ak: Seq = Seq::Formula(self.ak.clone()).plus(&Seq::Formula(other_ak.clone()))?;
        let (seq, s, e) = if lt(&other_ak.start, &self.ak.start)? {
            (other_ak.clone(), other_ak.start.clone(), self.ak.start.clone())
        } else {
            (self.ak.clone(), self.ak.start.clone(), other_ak.start.clone())
        };
        let (Some(si), Some(ei)) = (integer_class(&s), integer_class(&e)) else {
            return Err(PolyError::Value(
                "the series start at points which are not integers".to_string(),
            ));
        };
        let firsts: Vec<Basic> = Seq::Formula(seq).slice(0, Some(ei - si), 1)?;
        let powers: Vec<Basic> = Seq::Formula(self.xk.clone()).slice(si, Some(ei), 1)?;
        let products: Vec<Basic> = firsts
            .iter()
            .zip(&powers)
            .map(|(a, b)| mul(&[a.clone(), b.clone()]))
            .collect::<PolyResult<_>>()?;
        let ind: Basic = add(&[self.ind.clone(), o.ind.clone(), add(&products)?])?;
        Ok(Fps::Series(self.with(f, as_formula_seq(ak)?, self.xk.clone(), ind)))
    }

    /// The sum with an expression free of `x` (`__add__`).
    pub fn plus_expr(&self, o: &Basic) -> PolyResult<FormalPowerSeries> {
        if o.has(&self.x) {
            return Err(crate::error::not_ported(
                "the sum of a formal power series and a function of x",
            ));
        }
        Ok(self.with(
            add(&[self.function.clone(), o.clone()])?,
            self.ak.clone(),
            self.xk.clone(),
            add(&[self.ind.clone(), o.clone()])?,
        ))
    }

    /// `-self` (`__neg__`).
    pub fn neg(&self) -> PolyResult<FormalPowerSeries> {
        let ak: SeqFormula = as_formula_seq(Seq::Formula(self.ak.clone()).neg()?)?;
        Ok(self.with(neg(&self.function)?, ak, self.xk.clone(), neg(&self.ind)?))
    }

    /// `self - o` (`__sub__`).
    pub fn minus(&self, o: &FormalPowerSeries) -> PolyResult<Fps> {
        self.plus(&o.neg()?)
    }

    /// The series multiplied by an expression free of `x` (`__mul__`).
    pub fn scale(&self, o: &Basic) -> PolyResult<FormalPowerSeries> {
        if o.has(&self.x) {
            return Err(crate::error::not_ported(
                "the product of a formal power series and a function of x",
            ));
        }
        let ak: SeqFormula = as_formula_seq(Seq::Formula(self.ak.clone()).coeff_mul(o)?)?;
        Ok(self.with(
            mul(&[self.function.clone(), o.clone()])?,
            ak,
            self.xk.clone(),
            mul(&[self.ind.clone(), o.clone()])?,
        ))
    }

    fn check_compatible(&self, b: &FormalPowerSeries) -> PolyResult<()> {
        if self.direction != b.direction {
            return Err(PolyError::Value(
                "Both series should be calculated from the same direction.".to_string(),
            ));
        }
        if self.x0 != b.x0 {
            return Err(PolyError::Value(
                "Both series should be calculated about the same point.".to_string(),
            ));
        }
        if self.x != b.x {
            return Err(PolyError::Value("Both series should have the same symbol.".to_string()));
        }
        Ok(())
    }

    /// The sequence of the incomplete Bell polynomials `B_{n,j}(b_1, b_2,
    /// ...)`, `j = 1, ..., n`, of the coefficients times the factorials
    /// (`coeff_bell`).
    pub fn coeff_bell(&self, n: usize) -> PolyResult<Seq> {
        let values: Vec<Basic> = self.bell_coeff_seq()?.slice(0, Some(n as i64), 1)?;
        let mut inner: Vec<Basic> = Vec::new();
        for j in 1..=n {
            inner.push(bell_incomplete(n, j, &values[..n - j + 1])?);
        }
        let k: Basic = crate::core::symbol::Dummy::new("k", &[]).0;
        crate::series::sequences::sequence(
            &SeqSource::Periodical(inner),
            Some(&SeqLimits::Variable(k, int(1), Basic::oo())),
        )
    }
}

/// The products, compositions and inverses of formal power series: their
/// terms up to an order, no formula (`FiniteFormalPowerSeries` and its
/// classes).
#[derive(Clone, Debug)]
pub enum FiniteFps {
    Product(FormalPowerSeries, FormalPowerSeries),
    Compose(FormalPowerSeries, FormalPowerSeries),
    Inverse(FormalPowerSeries),
}

impl FiniteFps {
    fn ffps(&self) -> &FormalPowerSeries {
        match self {
            FiniteFps::Product(f, _) | FiniteFps::Compose(f, _) | FiniteFps::Inverse(f) => f,
        }
    }

    /// The function of the series (`function`).
    pub fn function(&self) -> R {
        match self {
            FiniteFps::Product(f, g) => mul(&[f.function.clone(), g.function.clone()]),
            FiniteFps::Compose(f, g) => subs(&f.function, &f.x, &g.function),
            FiniteFps::Inverse(f) => div(&int(1), &f.function),
        }
    }

    /// The first `n` terms (`_eval_terms`, `polynomial`).
    pub fn polynomial(&self, n: usize) -> R {
        match self {
            FiniteFps::Product(f, g) => {
                // the coefficient of x**j from the j-th term, which has the
                // independent terms and is zero before the start of ak (SymPy
                // reads ak.formula from 0: fps(cos(x))*fps(log(1 + x)) was nan)
                let coeffs = |s: &FormalPowerSeries| -> PolyResult<Vec<Basic>> {
                    let mut c: Vec<Basic> = Vec::new();
                    for j in 0..n as i64 {
                        let xj: Basic = Seq::Formula(s.xk.clone()).coeff(&int(j))?;
                        c.push(cancel(&div(&s.item(j)?, &xj)?)?);
                    }
                    Ok(c)
                };
                let (a, b) = (coeffs(f)?, coeffs(g)?);
                let mut terms: Vec<Basic> = Vec::new();
                for i in 0..n {
                    let parts: Vec<Basic> = (0..=i)
                        .map(|j| mul(&[a[j].clone(), b[i - j].clone()]))
                        .collect::<PolyResult<_>>()?;
                    let xi: Basic = Seq::Formula(f.xk.clone()).coeff(&int(i as i64))?;
                    terms.push(mul(&[add(&parts)?, xi])?);
                }
                add(&terms)
            }
            FiniteFps::Compose(f, g) => self.bell_terms(n, &f.bell_coeff_seq()?, g),
            FiniteFps::Inverse(f) => self.bell_terms(n, &self.aux_seq()?, f),
        }
    }

    /// The series truncated at `O(x**n)` (`truncate`).
    pub fn truncate(&self, n: usize) -> R {
        let ffps: &FormalPowerSeries = self.ffps();
        let pt_xk: Basic = Seq::Formula(ffps.xk.clone()).coeff(&int(n as i64))?;
        let o: Basic = crate::core::order::order(&pt_xk, &[(ffps.x.clone(), ffps.x0.clone())])?;
        add(&[self.polynomial(n)?, o])
    }

    /// `sign_seq * fact_seq * inv**(-(k + 1))` (`aux_seq`).
    fn aux_seq(&self) -> PolyResult<Seq> {
        let ffps: &FormalPowerSeries = self.ffps();
        let k: &Basic = &ffps.xk.variable;
        let inv: Basic = ffps.zero_coeff()?;
        let inv_seq: SeqFormula = formula_from(
            &pow(&inv, &neg(&add(&[k.clone(), int(1)])?)?)?,
            k,
            &int(1),
            &Basic::oo(),
        )?;
        ffps.sign_seq()?
            .times(&Seq::Formula(ffps.fact_seq()?))?
            .times(&Seq::Formula(inv_seq))
    }

    fn bell_terms(&self, n: usize, aux: &Seq, inner: &FormalPowerSeries) -> R {
        let ffps: &FormalPowerSeries = self.ffps();
        let mut terms: Vec<Basic> = vec![ffps.zero_coeff()?];
        let fact: Seq = Seq::Formula(ffps.fact_seq()?);
        for i in 1..n {
            let seq: Seq = aux.times(&inner.coeff_bell(i)?)?;
            let first: Basic = add(&seq.slice(0, Some(i as i64), 1)?)?;
            let xi: Basic = Seq::Formula(ffps.xk.clone()).coeff(&int(i as i64))?;
            terms.push(mul(&[div(&first, &fact.get(i as i64 - 1)?)?, xi])?);
        }
        add(&terms)
    }
}

impl FormalPowerSeries {
    /// The product with another series (`product`).
    pub fn product(&self, other: &FormalPowerSeries) -> PolyResult<FiniteFps> {
        self.check_compatible(other)?;
        Ok(FiniteFps::Product(self.clone(), other.clone()))
    }

    /// The composition `f(g(x))` with the series of `g`, which has no
    /// constant term (`compose`).
    pub fn compose(&self, other: &FormalPowerSeries) -> PolyResult<FiniteFps> {
        self.check_compatible(other)?;
        let mut constant: Basic = other.eval_term(&Basic::zero())?;
        if constant.has(&other.x) {
            constant = independent(&constant, &other.x, false)?.0;
        }
        if !is_zero(&constant) {
            return Err(PolyError::Value(
                "The formal power series of the inner function should not have any constant coefficient term."
                    .to_string(),
            ));
        }
        Ok(FiniteFps::Compose(self.clone(), other.clone()))
    }

    /// The series of `1/f` (`inverse`).
    pub fn inverse(&self) -> PolyResult<FiniteFps> {
        if is_true(&self.eval_term(&Basic::zero())?, Predicate::zero) {
            return Err(PolyError::Value(
                "Constant coefficient should exist for an inverse of a formal power series to exist.".to_string(),
            ));
        }
        Ok(FiniteFps::Inverse(self.clone()))
    }
}

/// The formal power series of `f` in `x` about `x0`: a series with an
/// explicit formula of the coefficients (see [`compute_fps`]), `f` itself
/// when none is found (`fps`); `x` the only free symbol of `f` when not
/// given.
#[allow(clippy::too_many_arguments)]
pub fn fps(
    f: &Basic,
    x: Option<&Basic>,
    x0: &Basic,
    dir: i64,
    hyper: bool,
    order: usize,
    rational: bool,
    full: bool,
) -> PolyResult<Fps> {
    let x: Basic = match x {
        Some(x) => x.clone(),
        None => {
            let free: Vec<Basic> = f.free_symbols().into_iter().collect();
            match free.len() {
                0 => return Ok(Fps::Expr(f.clone())),
                1 => free[0].clone(),
                _ => {
                    return Err(PolyError::NotImplemented(
                        "multivariate formal power series".to_string(),
                    ))
                }
            }
        }
    };
    let Some((ak, xk, ind)) = compute_fps(f, &x, x0, dir, hyper, order, rational, full)? else {
        return Ok(Fps::Expr(f.clone()));
    };
    Ok(Fps::Series(FormalPowerSeries {
        function: f.clone(),
        x,
        x0: x0.clone(),
        direction: int(dir),
        ak,
        xk,
        ind,
    }))
}
