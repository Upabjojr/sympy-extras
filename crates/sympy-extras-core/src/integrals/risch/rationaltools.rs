//! Port of `sympy_extras/integrals/risch/rationaltools.py` (SymPy's
//! `rationaltools` in the version of Aaron Meurer's pull requests):
//! integration of rational functions by Hermite's reduction
//! (Horowitz–Ostrogradsky) and the logarithmic part of Lazard, Rioboo and
//! Trager, with Rioboo's conversion of the complex logarithms to real
//! arctangents. The coefficients are rational (`QQ`) or expressions of the
//! core (`EX`, or `EXR` when they have square roots of numbers); the
//! algorithms are generic over the domain.
//!
//! References: M. Bronstein, Symbolic Integration I, 2nd ed., Springer
//! 2005, pp. 35-70.

#![allow(clippy::type_complexity, clippy::needless_range_loop, clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::{as_numer_denom, is, is_true};
use crate::core::symbol::Dummy;
use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;
use crate::polys::densearith::{dup_add, dup_div, dup_exquo, dup_mul, dup_neg, dup_rem, dup_sub};
use crate::polys::densebasic::{dup_LC, dup_degree, dup_strip};
use crate::polys::densetools::{dup_clear_denoms, dup_diff, dup_integrate, dup_monic};
use crate::polys::domains::expressiondomain::{ExpressionDomain, EX, EXR};
use crate::polys::domains::{Domain, PolynomialRing, QQ, ZZ};
use crate::polys::euclidtools::{dup_gcd, dup_gcdex, dup_inner_subresultants, dup_invert};
use crate::polys::rootoftools::{expr_of_coefficients, has_radical_number, polynomial_coefficients, root_sum};
use crate::polys::sqfreetools::dup_sqf_list;

type R = PolyResult<Basic>;

fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn inv(e: &Basic) -> R {
    crate::core::power::pow(e, &Basic::negative_one())
}
fn cancel(e: &Basic, gens: &[Basic]) -> R {
    crate::polys::polytools::cancel(e, gens)
}

fn is_rational(cs: &[Basic]) -> bool {
    cs.iter().all(|c| c.is_rational_class())
}

fn qq(cs: &[Basic]) -> PolyResult<Vec<Rat>> {
    let values: Vec<Rat> = cs.iter().map(|c| QQ.from_expr(c)).collect::<PolyResult<_>>()?;
    Ok(dup_strip(&values, &QQ))
}

fn ex(cs: &[Basic], k: &ExpressionDomain) -> Vec<Basic> {
    dup_strip(cs, k)
}

fn has_square_root(c: &Basic) -> bool {
    c.preorder().iter().any(|n| {
        n.is_pow()
            && n.args()[1].is_rational_class()
            && n.args()[1].as_rational().is_some_and(|v| denom(v) == Int::from(2))
            && n.args()[0].free_symbols().is_empty()
    })
}

fn expression_domain(cs: &[Basic]) -> ExpressionDomain {
    if cs.iter().any(has_square_root) {
        EXR
    } else {
        EX
    }
}

fn exprs<K: Domain>(f: &[K::Elem], k: &K) -> PolyResult<Vec<Basic>> {
    f.iter().map(|c| k.to_expr(c)).collect()
}

fn as_expr<K: Domain>(f: &[K::Elem], k: &K, x: &Basic) -> R {
    expr_of_coefficients(&exprs(f, k)?, x)
}

fn lift<K: Domain>(f: &[K::Elem], k: &K) -> Vec<Vec<K::Elem>> {
    f.iter()
        .map(|c| if k.is_zero(c) { vec![] } else { vec![c.clone()] })
        .collect()
}

fn bivariate_expr<K: Domain>(h: &[Vec<K::Elem>], k: &K, x: &Basic, t: &Basic) -> R {
    let cs: Vec<Basic> = h.iter().map(|c| as_expr(c, k, t)).collect::<PolyResult<_>>()?;
    sympy_expr(&cs, x, false)
}

// -- SymPy's normalisations of QQ and EX

fn cancel_qq(f: &[Rat], g: &[Rat]) -> PolyResult<(Vec<Rat>, Vec<Rat>, Int, Int)> {
    let (cq, big_f) = dup_clear_denoms(f, &QQ);
    let (cp, big_g) = dup_clear_denoms(g, &QQ);
    let fz: Vec<Int> = big_f.iter().map(numer).collect();
    let gz: Vec<Int> = big_g.iter().map(numer).collect();
    let h: Vec<Int> = dup_gcd(&fz, &gz, &ZZ);
    let mut pz: Vec<Int> = if h.is_empty() {
        fz.clone()
    } else {
        dup_exquo(&fz, &h, &ZZ)?
    };
    let mut qz: Vec<Int> = if h.is_empty() {
        gz.clone()
    } else {
        dup_exquo(&gz, &h, &ZZ)?
    };
    let common: Int = gcd(&cp, &cq);
    let (mut cp, cq) = (&cp / &common, &cq / &common);
    let p_neg: bool = pz.first().is_some_and(|c| *c < Int::ZERO);
    let q_neg: bool = qz.first().is_some_and(|c| *c < Int::ZERO);
    if p_neg && q_neg {
        pz = pz.iter().map(|c| -c.clone()).collect();
        qz = qz.iter().map(|c| -c.clone()).collect();
    } else if p_neg {
        cp = -cp;
        pz = pz.iter().map(|c| -c.clone()).collect();
    } else if q_neg {
        cp = -cp;
        qz = qz.iter().map(|c| -c.clone()).collect();
    }
    Ok((
        pz.into_iter().map(Rat::from).collect(),
        qz.into_iter().map(Rat::from).collect(),
        cp,
        cq,
    ))
}

/// SymPy's `Poly.cancel(include=False)` (`_cancel`).
fn cancel_poly<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> PolyResult<(Basic, Vec<K::Elem>, Vec<K::Elem>)> {
    if k.name() == "QQ" {
        let big_f: Vec<Rat> = exprs(f, k)?
            .iter()
            .map(|c| QQ.from_expr(c))
            .collect::<PolyResult<_>>()?;
        let big_g: Vec<Rat> = exprs(g, k)?
            .iter()
            .map(|c| QQ.from_expr(c))
            .collect::<PolyResult<_>>()?;
        let (p, q, cp, cq) = cancel_qq(&big_f, &big_g)?;
        let back = |v: &[Rat]| -> PolyResult<Vec<K::Elem>> {
            v.iter()
                .map(|c| k.from_expr(&Basic::from_rational(c.clone())))
                .collect()
        };
        return Ok((Basic::from_rational(rat_from_parts(cp, cq)), back(&p)?, back(&q)?));
    }
    let h: Vec<K::Elem> = if !f.is_empty() && !g.is_empty() {
        dup_monic(&dup_gcd(f, g, k), k)?
    } else if !f.is_empty() || !g.is_empty() {
        vec![k.one()]
    } else {
        vec![]
    };
    let mut p: Vec<K::Elem> = if h.is_empty() { f.to_vec() } else { dup_div(f, &h, k)?.0 };
    let mut q: Vec<K::Elem> = if h.is_empty() { g.to_vec() } else { dup_div(g, &h, k)?.0 };
    let mut sign: i64 = 1;
    let p_neg: bool = !p.is_empty() && k.is_negative(&dup_LC(&p, k));
    let q_neg: bool = !q.is_empty() && k.is_negative(&dup_LC(&q, k));
    if p_neg && q_neg {
        p = dup_neg(&p, k);
        q = dup_neg(&q, k);
    } else if p_neg {
        sign = -sign;
        p = dup_neg(&p, k);
    } else if q_neg {
        sign = -sign;
        q = dup_neg(&q, k);
    }
    Ok((Basic::int(sign), p, q))
}

fn integer_primitive<K: Domain>(f: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    let values: Vec<Rat> = exprs(f, k)?
        .iter()
        .map(|c| QQ.from_expr(c))
        .collect::<PolyResult<_>>()?;
    let mut denominator: Int = Int::ONE;
    for v in &values {
        let d: Int = denom(v);
        let g: Int = gcd(&denominator, &d);
        denominator = &denominator * &d / g;
    }
    let numerators: Vec<Int> = values
        .iter()
        .map(|v| numer(&(v * Rat::from(denominator.clone()))))
        .collect();
    let mut content: Int = Int::ZERO;
    for n in &numerators {
        content = gcd(&content, n);
    }
    if content.is_zero() {
        content = Int::ONE;
    }
    let result: Vec<K::Elem> = numerators
        .iter()
        .map(|n| k.from_expr(&Basic::integer(div_floor(n, &content))))
        .collect::<PolyResult<_>>()?;
    Ok(dup_strip(&result, k))
}

fn is_negative_real(c: &Basic) -> bool {
    if let Some(v) = c.as_rational().filter(|_| c.is_rational_class()) {
        return rat_is_negative(v);
    }
    if !is_true(c, Predicate::extended_real) {
        return false;
    }
    let mut negative: Option<bool> = is(c, Predicate::extended_negative);
    if negative.is_none() && c.free_symbols().is_empty() {
        negative = crate::numeric::reliable_complex(c, 30, &[]).map(|(re, _)| re < 0.0);
    }
    negative == Some(true)
}

fn has_symbolic_denominator(c: &Basic) -> PolyResult<bool> {
    Ok(!as_numer_denom(c)?.1.is_rational_class())
}

/// The coefficients as SymPy's `Poly(...).as_expr()` writes them over the
/// domain `construct_domain` gives them (`sympy_form`).
pub fn sympy_form(cs: &[Basic], field: bool) -> PolyResult<Vec<Basic>> {
    let nonzero: Vec<&Basic> = cs.iter().filter(|c| **c != Basic::zero()).collect();
    if nonzero.iter().all(|c| c.is_rational_class()) || nonzero.iter().any(|c| has_radical_number(c)) {
        return Ok(cs.to_vec());
    }
    let mut symbolic: bool = field;
    for c in &nonzero {
        if has_symbolic_denominator(c)? {
            symbolic = true;
        }
    }
    if symbolic {
        return cs
            .iter()
            .map(|c| {
                if *c != Basic::zero() {
                    cancel(c, &[])
                } else {
                    Ok(c.clone())
                }
            })
            .collect();
    }
    Ok(cs.to_vec())
}

fn sympy_expr(cs: &[Basic], x: &Basic, field: bool) -> R {
    expr_of_coefficients(&sympy_form(cs, field)?, x)
}

fn primitive_expr(cs: Vec<Basic>) -> PolyResult<Vec<Basic>> {
    let nonzero: Vec<Basic> = cs.iter().filter(|c| **c != Basic::zero()).cloned().collect();
    if nonzero.is_empty() {
        return Ok(cs);
    }
    if is_rational(&nonzero) {
        return exprs(&integer_primitive(&qq(&cs)?, &QQ)?, &QQ);
    }
    if nonzero.iter().any(has_radical_number) {
        return Ok(cs);
    }
    for c in &nonzero {
        if !as_numer_denom(c)?.1.free_symbols().is_empty() {
            return Ok(cs);
        }
    }
    let content: Basic = match crate::polys::polytools::gcd_list(&nonzero, &[]) {
        Ok(c) => c,
        Err(e) if crate::polys::rootoftools::is_failure(&e) => return Ok(cs),
        Err(e) => return Err(e),
    };
    if content == Basic::one() || content == Basic::zero() {
        return Ok(cs);
    }
    let ic: Basic = inv(&content)?;
    cs.iter()
        .map(|c| cancel(&mul(&[c.clone(), ic.clone()])?, &[]))
        .collect()
}

// -- Horowitz–Ostrogradsky

fn solve<K: Domain>(rows: &[Vec<K::Elem>], rhs: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    let n: usize = rows.len();
    let mut m: Vec<Vec<K::Elem>> = rows
        .iter()
        .zip(rhs)
        .map(|(r, b)| r.iter().cloned().chain(std::iter::once(b.clone())).collect())
        .collect();
    for col in 0..n {
        let Some(pivot) = (col..n).find(|i| !k.is_zero(&m[*i][col])) else {
            return Err(value_error(
                "the system of Horowitz and Ostrogradsky is singular".to_string(),
            ));
        };
        m.swap(col, pivot);
        let inverse: K::Elem = k.invert(&m[col][col])?;
        m[col] = m[col].iter().map(|c| k.mul(&inverse, c)).collect();
        for i in 0..n {
            if i != col && !k.is_zero(&m[i][col]) {
                let factor: K::Elem = m[i][col].clone();
                let pivot_row: Vec<K::Elem> = m[col].clone();
                m[i] = m[i]
                    .iter()
                    .zip(&pivot_row)
                    .map(|(a, b)| k.sub(a, &k.mul(&factor, b)))
                    .collect();
            }
        }
    }
    Ok((0..n).map(|i| m[i][n].clone()).collect())
}

fn coefficient<K: Domain>(f: &[K::Elem], kk: usize, k: &K) -> K::Elem {
    let n: isize = dup_degree(f);
    if (kk as isize) <= n {
        f[(n - kk as isize) as usize].clone()
    } else {
        k.zero()
    }
}

/// `(A, u, B, v)` with `f/g = (A/u)' + B/v`, `v` square-free (`_ratpart`).
fn ratpart<K: Domain>(
    f: &[K::Elem],
    g: &[K::Elem],
    k: &K,
) -> PolyResult<(Vec<K::Elem>, Vec<K::Elem>, Vec<K::Elem>, Vec<K::Elem>)> {
    let u: Vec<K::Elem> = dup_monic(&dup_gcd(g, &dup_diff(g, 1, k), k), k)?;
    let v: Vec<K::Elem> = dup_div(g, &u, k)?.0;
    let (n, m) = (dup_degree(&u).max(0) as usize, dup_degree(&v).max(0) as usize);
    let w: Vec<K::Elem> = dup_div(&dup_mul(&dup_diff(&u, 1, k), &v, k), &u, k)?.0;
    let size: usize = n + m;
    let mut columns: Vec<Vec<K::Elem>> = Vec::new();
    for i in 0..n {
        let mut power: Vec<K::Elem> = vec![k.one()];
        power.extend(std::iter::repeat(k.zero()).take(i));
        let derivative: Vec<K::Elem> = dup_diff(&power, 1, k);
        columns.push(dup_sub(&dup_mul(&derivative, &v, k), &dup_mul(&power, &w, k), k));
    }
    for j in 0..m {
        let mut power: Vec<K::Elem> = vec![k.one()];
        power.extend(std::iter::repeat(k.zero()).take(j));
        columns.push(dup_mul(&power, &u, k));
    }
    let rows: Vec<Vec<K::Elem>> = (0..size)
        .map(|kk| columns.iter().map(|c| coefficient(c, kk, k)).collect())
        .collect();
    let rhs: Vec<K::Elem> = (0..size).map(|kk| coefficient(f, kk, k)).collect();
    let solution: Vec<K::Elem> = if size > 0 { solve(&rows, &rhs, k)? } else { vec![] };
    let a: Vec<K::Elem> = dup_strip(&solution[..n].iter().rev().cloned().collect::<Vec<_>>(), k);
    let b: Vec<K::Elem> = dup_strip(&solution[n..].iter().rev().cloned().collect::<Vec<_>>(), k);
    Ok((a, u, b, v))
}

fn ratpart_exprs<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let (a, u, b, v) = ratpart(f, g, k)?;
    let rat_part: Basic = cancel_in(&mul(&[as_expr(&a, k, x)?, inv(&as_expr(&u, k, x)?)?])?, x)?;
    let log_part: Basic = cancel_in(&mul(&[as_expr(&b, k, x)?, inv(&as_expr(&v, k, x)?)?])?, x)?;
    Ok((rat_part, log_part))
}

/// SymPy's `cancel(e, x)`, over `EX` for algebraic numbers (`_cancel_in`).
fn cancel_in(e: &Basic, x: &Basic) -> R {
    if !has_radical_number(e) {
        return cancel(e, std::slice::from_ref(x));
    }
    let domain: ExpressionDomain = expression_domain(std::slice::from_ref(e));
    let factored: Basic =
        crate::core::exprtools::factor_terms(&crate::simplify::signsimp::signsimp(e, true)?, true, false, false, true)?;
    let (p, q) = as_numer_denom(&factored)?;
    let (Ok(pc), Ok(qc)) = (polynomial_coefficients(&p, x), polynomial_coefficients(&q, x)) else {
        return cancel(e, std::slice::from_ref(x));
    };
    let (big_p, big_q) = (ex(&pc, &domain), ex(&qc, &domain));
    let (sign, big_p, big_q) = cancel_poly(&big_p, &big_q, &domain)?;
    mul(&[
        sign,
        mul(&[as_expr(&big_p, &domain, x)?, inv(&as_expr(&big_q, &domain, x)?)?])?,
    ])
}

// -- the logarithmic part

type Bivariate<T> = Vec<Vec<T>>;

/// Lazard–Rioboo–Trager on `f/g` (`_logpart`).
fn logpart<K: Domain>(
    f: &[K::Elem],
    g: &[K::Elem],
    k: &K,
    t: &Basic,
) -> PolyResult<Vec<(Bivariate<K::Elem>, Vec<K::Elem>)>> {
    let name: String = match t.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => t.to_string(),
    };
    let ring: PolynomialRing<K> = PolynomialRing::new(k.clone(), &name);
    let a: Bivariate<K::Elem> = lift(g, k);
    let derivative: Vec<K::Elem> = dup_diff(g, 1, k);
    let size: usize = f.len().max(derivative.len());
    let pad = |v: &[K::Elem]| -> Vec<K::Elem> {
        std::iter::repeat(k.zero())
            .take(size - v.len())
            .chain(v.iter().cloned())
            .collect()
    };
    let (pf, pd) = (pad(f), pad(&derivative));
    let b_raw: Bivariate<K::Elem> = pf
        .iter()
        .zip(&pd)
        .map(|(c, d)| dup_strip(&[k.neg(d), c.clone()], k))
        .collect();
    let b: Bivariate<K::Elem> = dup_strip(&b_raw, &ring);
    let (sequence, scalars) = dup_inner_subresultants(&a, &b, &ring);
    let res: Vec<K::Elem> = if sequence.is_empty() || dup_degree(sequence.last().expect("a remainder")) > 0 {
        vec![]
    } else {
        scalars.last().cloned().unwrap_or_default()
    };
    if res.is_empty() {
        return Err(value_error("BUG: the resultant is zero".to_string()));
    }
    let mut by_degree: Vec<(isize, Bivariate<K::Elem>)> = Vec::new();
    for r in &sequence {
        let d: isize = dup_degree(r);
        match by_degree.iter().position(|(k2, _)| *k2 == d) {
            Some(i) => by_degree[i].1 = r.clone(),
            None => by_degree.push((d, r.clone())),
        }
    }
    let rational: bool = is_rational(&exprs(&res, k)?);
    let (big_c, factors) = dup_sqf_list(&res, k);
    let c_negative: bool = is_negative_real(&k.to_expr(&big_c)?);
    let mut normalised: Vec<(Vec<K::Elem>, usize)> = Vec::new();
    for (idx, (q, i)) in factors.into_iter().enumerate() {
        let q: Vec<K::Elem> = if rational {
            let p: Vec<K::Elem> = integer_primitive(&q, k)?;
            if idx == 0 && c_negative {
                dup_neg(&p, k)
            } else {
                p
            }
        } else if idx == 0 && c_negative {
            q.iter().map(|c| k.mul(c, &big_c)).collect()
        } else {
            q
        };
        normalised.push((q, i));
    }
    let mut result: Vec<(Bivariate<K::Elem>, Vec<K::Elem>)> = Vec::new();
    for (q, i) in normalised {
        if dup_degree(g) == i as isize {
            result.push((a.clone(), q));
            continue;
        }
        let mut h: Bivariate<K::Elem> = by_degree
            .iter()
            .find(|(d, _)| *d == i as isize)
            .map(|(_, r)| r.clone())
            .ok_or_else(|| value_error(format!("no subresultant of degree {}", i)))?;
        let h_lc: Vec<K::Elem> = h[0].clone();
        let (_, lc_factors) = dup_sqf_list(&h_lc, k);
        for (factor, j) in lc_factors {
            let common: Vec<K::Elem> = dup_gcd(&factor, &q, k);
            if dup_degree(&common) > 0 {
                let divisor: Vec<K::Elem> = ring.pow(&common, j as u64);
                h = h.iter().map(|c| ring.exquo(c, &divisor)).collect::<PolyResult<_>>()?;
            }
        }
        let inverse: Vec<K::Elem> = dup_invert(&h_lc, &q, k)?;
        let mut coefficients: Bivariate<K::Elem> = Vec::new();
        let mut first: bool = true;
        for c in &h {
            if c.is_empty() {
                coefficients.push(vec![]);
            } else if first {
                coefficients.push(vec![k.one()]);
                first = false;
            } else {
                coefficients.push(dup_rem(&dup_mul(&inverse, c, k), &q, k)?);
            }
        }
        result.push((coefficients, q));
    }
    Ok(result)
}

// -- the public functions

fn dummy_of(symbol: &Basic) -> Basic {
    match symbol.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => Dummy::new(&d.name, &d.given).0,
        _ => Dummy::new(&symbol.to_string(), &[]).0,
    }
}

/// The rational function to integrate: `numerator/denominator`, and the
/// expression whose atoms decide `real` by default.
pub struct Integrand<'a> {
    pub numerator: Basic,
    pub denominator: Basic,
    pub atoms_of: Vec<&'a Basic>,
}

impl<'a> Integrand<'a> {
    /// `f` as `p/q` (`as_numer_denom`).
    pub fn of(f: &'a Basic) -> PolyResult<Integrand<'a>> {
        let (numerator, denominator) = as_numer_denom(f)?;
        Ok(Integrand {
            numerator,
            denominator,
            atoms_of: vec![f],
        })
    }
    /// The pair `(p, q)`.
    pub fn pair(p: &'a Basic, q: &'a Basic) -> Integrand<'a> {
        Integrand {
            numerator: p.clone(),
            denominator: q.clone(),
            atoms_of: vec![p, q],
        }
    }
}

/// An antiderivative of a rational function in `x`: the polynomial and
/// rational parts and the logarithmic part with the roots written out
/// where they are found (real arctangents when `real`, by default when
/// every atom but `x` is real), a `RootSum` in a dummy named after
/// `symbol` elsewhere (`ratint`).
pub fn ratint(f: &Integrand, x: &Basic, symbol: &Basic, real: Option<bool>) -> R {
    let nc: Vec<Basic> = polynomial_coefficients(&f.numerator, x)?;
    let dc: Vec<Basic> = polynomial_coefficients(&f.denominator, x)?;
    let all: Vec<Basic> = nc.iter().chain(&dc).cloned().collect();
    if is_rational(&all) {
        return ratint_in(&qq(&nc)?, &qq(&dc)?, &QQ, f, x, symbol, real);
    }
    let e: ExpressionDomain = expression_domain(&all);
    ratint_in(&ex(&nc, &e), &ex(&dc, &e), &e, f, x, symbol, real)
}

fn ratint_in<K: Domain>(
    p0: &[K::Elem],
    q0: &[K::Elem],
    k: &K,
    f: &Integrand,
    x: &Basic,
    symbol: &Basic,
    real: Option<bool>,
) -> R {
    if q0.is_empty() {
        return Err(zero_division("the denominator of the integrand is zero"));
    }
    let (coeff, p, q) = cancel_poly(p0, q0, k)?;
    let (poly, p) = dup_div(&p, &q, k)?;
    let mut result: Basic = as_expr(&dup_integrate(&poly, 1, k)?, k, x)?;
    if p.is_empty() {
        return mul(&[coeff, result]);
    }
    let (g, h) = ratpart_exprs(&p, &q, k, x)?;
    let (pe, qe) = as_numer_denom(&h)?;
    let pc: Vec<Basic> = polynomial_coefficients(&pe, x)?;
    let qc: Vec<Basic> = polynomial_coefficients(&qe, x)?;
    let all: Vec<Basic> = pc.iter().chain(&qc).cloned().collect();
    let (quotient, eps) = if k.name() == "QQ" && is_rational(&all) {
        logarithmic_terms(&qq(&pc)?, &qq(&qc)?, &QQ, f, x, symbol, real)?
    } else {
        let e: ExpressionDomain = expression_domain(&all);
        logarithmic_terms(&ex(&pc, &e), &ex(&qc, &e), &e, f, x, symbol, real)?
    };
    result = add(&[result, g, quotient])?;
    if let Some(eps) = eps {
        result = add(&[result, eps])?;
    }
    mul(&[coeff, result])
}

fn atoms_real(f: &Integrand, x: &Basic) -> bool {
    let mut atoms: Vec<Basic> = Vec::new();
    for e in &f.atoms_of {
        for a in e.atoms(None) {
            if !atoms.contains(&a) {
                atoms.push(a);
            }
        }
    }
    atoms
        .iter()
        .filter(|a| *a != x)
        .all(|a| !crate::core::logic::is_boolean(a) && is_true(a, Predicate::extended_real))
}

fn logarithmic_terms<K: Domain>(
    p: &[K::Elem],
    q: &[K::Elem],
    k: &K,
    f: &Integrand,
    x: &Basic,
    symbol: &Basic,
    real: Option<bool>,
) -> PolyResult<(Basic, Option<Basic>)> {
    let (quotient, r) = dup_div(p, q, k)?;
    let integral: Basic = sympy_expr(&exprs(&dup_integrate(&quotient, 1, k)?, k)?, x, true)?;
    if r.is_empty() {
        return Ok((integral, None));
    }
    let t: Basic = dummy_of(symbol);
    let parts = logpart(&r, q, k, &t)?;
    let real: bool = real.unwrap_or_else(|| atoms_real(f, x));
    let mut eps: Basic = Basic::zero();
    for (h, qq_) in parts {
        let q_expr: Basic = as_expr(&qq_, k, &t)?;
        let mut h_coefficients: Vec<Basic> = h.iter().map(|c| as_expr(c, k, &t)).collect::<PolyResult<_>>()?;
        if !h_coefficients.iter().any(|c| c.has(&t)) {
            h_coefficients = primitive_expr(h_coefficients)?;
        }
        let h_expr: Basic = sympy_expr(&h_coefficients, x, false)?;
        if real {
            if let Some(r) = log_to_real(&h_expr, &q_expr, x, &t, false)? {
                eps = add(&[eps, r])?;
                continue;
            }
        }
        let body: Basic = mul(&[t.clone(), crate::functions::elementary::exponential::log(&h_expr)?])?;
        eps = add(&[eps, root_sum(&q_expr, Some((&t, &body)), Some(&t), true, true)?])?;
    }
    Ok((integral, Some(eps)))
}

/// Horowitz–Ostrogradsky: `A` and `B` with `f/g = A' + B`, `B` of
/// square-free denominator (`ratint_ratpart`).
pub fn ratint_ratpart(f: &Basic, g: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let fc: Vec<Basic> = polynomial_coefficients(f, x)?;
    let gc: Vec<Basic> = polynomial_coefficients(g, x)?;
    let all: Vec<Basic> = fc.iter().chain(&gc).cloned().collect();
    if is_rational(&all) {
        return ratpart_exprs(&qq(&fc)?, &qq(&gc)?, &QQ, x);
    }
    let e: ExpressionDomain = expression_domain(&all);
    ratpart_exprs(&ex(&fc, &e), &ex(&gc, &e), &e, x)
}

/// The antiderivative of the quotient of `f` by `g` and the remainder
/// (`polynomial_part`).
pub fn polynomial_part(f: &Basic, g: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    fn part<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K, x: &Basic) -> PolyResult<(Basic, Basic)> {
        let (quotient, remainder) = dup_div(f, g, k)?;
        Ok((
            as_expr(&dup_integrate(&quotient, 1, k)?, k, x)?,
            as_expr(&remainder, k, x)?,
        ))
    }
    let fc: Vec<Basic> = polynomial_coefficients(f, x)?;
    let gc: Vec<Basic> = polynomial_coefficients(g, x)?;
    let all: Vec<Basic> = fc.iter().chain(&gc).cloned().collect();
    if is_rational(&all) {
        return part(&qq(&fc)?, &qq(&gc)?, &QQ, x);
    }
    let e: ExpressionDomain = expression_domain(&all);
    part(&ex(&fc, &e), &ex(&gc, &e), &e, x)
}

/// Lazard–Rioboo–Trager: the pairs `(S_i, q_i)` (`ratint_logpart`).
pub fn ratint_logpart(f: &Basic, g: &Basic, x: &Basic, t: Option<&Basic>) -> PolyResult<Vec<(Basic, Basic)>> {
    let variable: Basic = match t {
        Some(t) => t.clone(),
        None => Dummy::new("t", &[]).0,
    };
    fn pairs<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K, x: &Basic, t: &Basic) -> PolyResult<Vec<(Basic, Basic)>> {
        logpart(f, g, k, t)?
            .iter()
            .map(|(h, q)| Ok((bivariate_expr(h, k, x, t)?, as_expr(q, k, t)?)))
            .collect()
    }
    let fc: Vec<Basic> = polynomial_coefficients(f, x)?;
    let gc: Vec<Basic> = polynomial_coefficients(g, x)?;
    let all: Vec<Basic> = fc.iter().chain(&gc).cloned().collect();
    if is_rational(&all) {
        return pairs(&qq(&fc)?, &qq(&gc)?, &QQ, x, &variable);
    }
    let e: ExpressionDomain = expression_domain(&all);
    pairs(&ex(&fc, &e), &ex(&gc, &e), &e, x, &variable)
}

// -- real forms

/// Rioboo's conversion: a sum of arctangents `h` with
/// `h' = (I log((f + I g)/(f - I g)))'` (`log_to_atan`).
pub fn log_to_atan(f: &Basic, g: &Basic, x: &Basic) -> R {
    let fc: Vec<Basic> = polynomial_coefficients(f, x)?;
    let gc: Vec<Basic> = polynomial_coefficients(g, x)?;
    let all: Vec<Basic> = fc.iter().chain(&gc).cloned().collect();
    if is_rational(&all) {
        return log_to_atan_in(&qq(&fc)?, &qq(&gc)?, &QQ, x);
    }
    log_to_atan_in(&ex(&fc, &EX), &ex(&gc, &EX), &EX, x)
}

fn log_to_atan_in<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K, x: &Basic) -> R {
    let atan = crate::functions::elementary::inverse::atan;
    let (f, g): (Vec<K::Elem>, Vec<K::Elem>) = if dup_degree(f) < dup_degree(g) {
        (dup_neg(g, k), f.to_vec())
    } else {
        (f.to_vec(), g.to_vec())
    };
    let (p, q) = dup_div(&f, &g, k)?;
    if q.is_empty() {
        return mul(&[Basic::int(2), atan(&sympy_expr(&exprs(&p, k)?, x, true)?)?]);
    }
    let (s, t, h) = dup_gcdex(&g, &dup_neg(&f, k), k)?;
    let u: Vec<K::Elem> = dup_div(&dup_add(&dup_mul(&f, &s, k), &dup_mul(&g, &t, k), k), &h, k)?.0;
    let a: Basic = mul(&[Basic::int(2), atan(&sympy_expr(&exprs(&u, k)?, x, true)?)?])?;
    add(&[a, log_to_atan_in(&s, &t, k, x)?])
}

fn replaced(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    crate::core::function::xreplace_evaluated(e, rule)
}

fn generic_roots(q: &Basic, t: &Basic, radicals: bool) -> PolyResult<Option<Vec<(Basic, usize)>>> {
    let (generic, back) = crate::polys::rootoftools::generic_constants(q, radicals)?;
    if back.is_empty() {
        return Ok(None);
    }
    let result = match crate::polys::polyroots::roots(&generic, std::slice::from_ref(t), true, true, None) {
        Ok(r) => r,
        Err(e) if crate::polys::rootoftools::is_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut found: Vec<(Basic, usize)> = Vec::new();
    for (r, k) in result.iter() {
        let value: Basic = replaced(r, &back)?;
        match found.iter().position(|(v, _)| *v == value) {
            Some(i) => found[i].1 += *k,
            None => found.push((value, *k)),
        }
    }
    Ok(Some(found))
}

fn roots_of(q: &Basic, t: &Basic) -> PolyResult<Option<Vec<(Basic, usize)>>> {
    use crate::polys::rootoftools::{linear_root, quadratic_roots, Polynomial};
    let degree: usize = polynomial_coefficients(q, t)?.len().saturating_sub(1);
    let numerator: Basic = as_numer_denom(&cancel(q, std::slice::from_ref(t))?)?.0;
    let found: Vec<(Basic, usize)> =
        match crate::polys::polyroots::roots(&numerator, std::slice::from_ref(t), true, true, None) {
            Ok(result) => result.iter().map(|(r, k)| (r.clone(), *k)).collect(),
            Err(e) if crate::polys::rootoftools::is_failure(&e) => {
                let cleared: Vec<Basic> = polynomial_coefficients(&numerator, t)?;
                let radical: bool = has_radical_number(&numerator);
                if radical && degree == 1 {
                    vec![(linear_root(&Polynomial::new(None, cleared, t, HashMap::new())?)?, 1)]
                } else if radical && degree == 2 {
                    let mut found: Vec<(Basic, usize)> = Vec::new();
                    for r in quadratic_roots(&Polynomial::new(None, cleared, t, HashMap::new())?)? {
                        match found.iter().position(|(v, _)| *v == r) {
                            Some(i) => found[i].1 += 1,
                            None => found.push((r, 1)),
                        }
                    }
                    found
                } else {
                    match generic_roots(&numerator, t, radical)? {
                        Some(g) => g,
                        None => return Ok(None),
                    }
                }
            }
            Err(e) => return Err(e),
        };
    if found.iter().map(|(_, k)| *k).sum::<usize>() != degree {
        return Ok(None);
    }
    Ok(Some(found))
}

fn roots_real_complex(q: &Basic, t: &Basic) -> PolyResult<Option<(Vec<(Basic, usize)>, Vec<((Basic, Basic), usize)>)>> {
    let Some(rs) = roots_of(q, t)? else { return Ok(None) };
    let get = |r: &Basic| rs.iter().find(|(v, _)| v == r).map(|(_, k)| *k);
    let mut reals: Vec<(Basic, usize)> = Vec::new();
    let mut complexes: Vec<((Basic, Basic), usize)> = Vec::new();
    let mut remaining: Vec<Basic> = rs.iter().map(|(r, _)| r.clone()).collect();
    while let Some(r) = remaining.pop() {
        let r_c: Basic = crate::functions::elementary::complexes::conjugate(&r)?;
        if r != r_c && get(&r_c).is_some() {
            if get(&r_c) != get(&r) {
                return Err(value_error("conjugate roots of different multiplicities".to_string()));
            }
            if let Some(i) = remaining.iter().position(|v| *v == r_c) {
                remaining.remove(i);
            }
            let (r_re, r_im) = match crate::functions::elementary::complexes::as_real_imag(&r, true, None)? {
                Some(parts) => parts,
                None => (
                    crate::functions::elementary::complexes::re(&r)?,
                    crate::functions::elementary::complexes::im(&r)?,
                ),
            };
            let first: Basic =
                crate::core::sorting::ordered(vec![r_im.clone(), crate::core::mul::neg(&r_im)?]).remove(0);
            let key = (r_re, first);
            let k: usize = get(&r).expect("a root");
            match complexes.iter().position(|(c, _)| *c == key) {
                Some(i) => complexes[i].1 = k,
                None => complexes.push((key, k)),
            }
        } else {
            let k: usize = get(&r).expect("a root");
            match reals.iter().position(|(c, _)| *c == r) {
                Some(i) => reals[i].1 = k,
                None => reals.push((r, k)),
            }
        }
    }
    Ok(Some((reals, complexes)))
}

/// Rioboo's conversion of a sum of logarithms to real functions; `None`
/// when the roots of `q` are not all found, and, with `complex_only`,
/// when none comes in a conjugate pair (`log_to_real`).
pub fn log_to_real(h: &Basic, q: &Basic, x: &Basic, t: &Basic, complex_only: bool) -> PolyResult<Option<Basic>> {
    let Some((reals, complexes)) = roots_real_complex(q, t)? else {
        return Ok(None);
    };
    if complex_only && complexes.is_empty() {
        return Ok(None);
    }
    let u: Basic = Dummy::new("u", &[]).0;
    let v: Basic = Dummy::new("v", &[]).0;
    let i: Basic = Basic::imaginary_unit();
    let shifted: Basic = replaced(
        h,
        &HashMap::from([(t.clone(), add(&[u.clone(), mul(&[i.clone(), v.clone()])?])?)]),
    )?;
    let big_h: Basic = crate::core::function::expand_default(&shifted)?;
    let h_map: Vec<(Basic, Basic)> =
        crate::simplify::radsimp::collect_dict(&big_h, std::slice::from_ref(&i), None, Some(false))?;
    let get = |k: &Basic| {
        h_map
            .iter()
            .find(|(a, _)| a == k)
            .map(|(_, b)| b.clone())
            .unwrap_or_else(Basic::zero)
    };
    let (a, b) = (get(&Basic::one()), get(&i));
    let mut result: Basic = Basic::zero();
    for ((r_u, r_v), _) in &complexes {
        let rule: HashMap<Basic, Basic> = HashMap::from([(u.clone(), r_u.clone()), (v.clone(), r_v.clone())]);
        let (a_expr, b_expr) = (replaced(&a, &rule)?, replaced(&b, &rule)?);
        let ac: Vec<Basic> = polynomial_coefficients(&a_expr, x)?;
        let bc: Vec<Basic> = polynomial_coefficients(&b_expr, x)?;
        let all: Vec<Basic> = ac.iter().chain(&bc).cloned().collect();
        let term: Basic = if is_rational(&all) {
            complex_pair(&qq(&ac)?, &qq(&bc)?, &QQ, x, r_u, r_v)?
        } else {
            complex_pair(&ex(&ac, &EX), &ex(&bc, &EX), &EX, x, r_u, r_v)?
        };
        result = add(&[result, term])?;
    }
    for (r, _) in &reals {
        let at: Basic = crate::core::subs::subs(h, t, r)?;
        result = add(&[
            result,
            mul(&[r.clone(), crate::functions::elementary::exponential::log(&at)?])?,
        ])?;
    }
    Ok(Some(result))
}

fn complex_pair<K: Domain>(a: &[K::Elem], b: &[K::Elem], k: &K, x: &Basic, r_u: &Basic, r_v: &Basic) -> R {
    let mut field: bool = false;
    for c in a.iter().chain(b) {
        if has_symbolic_denominator(&k.to_expr(c)?)? {
            field = true;
        }
    }
    let ab: Basic = sympy_expr(&exprs(&dup_add(&dup_mul(a, a, k), &dup_mul(b, b, k), k), k)?, x, field)?;
    let mut result: Basic = mul(&[r_u.clone(), crate::functions::elementary::exponential::log(&ab)?])?;
    if !a.is_empty() && !b.is_empty() {
        result = add(&[result, mul(&[r_v.clone(), log_to_atan_in(a, b, k, x)?])?])?;
    }
    Ok(result)
}
