//! Port of `sympy_extras/polys/polyroots.py`: [`roots`], the roots of a
//! univariate polynomial in radicals with their multiplicities. The power
//! of the variable and the content come out, a common scale of the
//! coefficients and an integer basis are removed ([`preprocess_roots`]);
//! the polynomial is factored and each factor solved by the formulas:
//! linear, quadratic, binomial, cyclotomic, Cardano's and Ferrari's (with
//! Euler's resolvent), or after a functional decomposition. The quartic
//! whose resolvent sign is undecided (a `Piecewise` in the Python) and
//! the algebraic domain `EX` beyond binomials are `NotImplemented`.

// PORTED FROM SYMPY: sympy/polys/polyroots.py (roots, roots_linear, roots_quadratic,
// roots_binomial, roots_cubic, roots_quartic, _roots_quartic_euler, roots_cyclotomic,
// _inv_totient_estimate, preprocess_roots, _integer_basis), sympy/polys/densetools.py
// (dup_decompose), sympy/polys/factortools.py (dup_cyclotomic_p), sympy/core/mul.py
// (expand_2arg), sympy/core/add.py (_unevaluated_Add) (SymPy 1.14), through
// sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, HashMap};

use crate::core::add::add;
use crate::core::assumptions::{is_number, Predicate};
use crate::core::basic::{Basic, Head};
use crate::core::expr::{as_coeff_mul, as_independent, is_false, is_true};
use crate::core::function::{expand_with, rebuild_evaluated, xreplace_evaluated, Hints};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::core::sorting::ordered;
use crate::core::Dummy;
use crate::error::{arithmetic_error, not_ported, polys_error, value_error, PolyError, PolyResult, PolysErrorKind};
use crate::numbers::*;
use crate::orderedmap::OrderedMap;

use super::densearith::{dup_div, dup_mul, dup_neg, dup_sub};
use super::densebasic::dup_strip;
use super::densetools::dup_mirror;
use super::domains::{IntegerRing, RationalField, SparsePolynomialRing, QQ, ZZ};
use super::factortools::{dup_zz_cyclotomic_poly, dup_zz_factor};
use super::polytools::{cancel, decompose_with_gens, factor, Poly, PolyOrExpr};
use super::sqfreetools::dup_sqf_part;

/// The roots with their multiplicities, in insertion order (a dict).
pub type RootsResult = OrderedMap<Basic, usize>;

/// The filter of the roots: `'Z'`, `'Q'`, `'R'`, `'I'` or `'C'`.
pub type RootFilter<'a> = Option<&'a str>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}

fn q(n: i64, d: i64) -> Basic {
    if d == 1 {
        return int(n);
    }
    Basic::from_rational(rat(n, d))
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[int(-1), e.clone()])
}

fn inverse(e: &Basic) -> PolyResult<Basic> {
    pow(e, &int(-1))
}

fn sqrt(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::half())
}

/// `root(e, n)`: the principal `n`-th root.
fn root(e: &Basic, n: i64) -> PolyResult<Basic> {
    pow(e, &q(1, n))
}

fn is_zero_number(e: &Basic) -> bool {
    matches!(e.head(), Head::Rational { integer_class: true, value } if value.is_zero())
}

fn is_integer_one(e: &Basic) -> bool {
    matches!(e.head(), Head::Rational { integer_class: true, value } if *value == Rat::ONE)
}

fn rational(c: &Basic) -> PolyResult<Rat> {
    match c.as_rational() {
        Some(v) => Ok(v.clone()),
        None => Err(polys_error(
            PolysErrorKind::CoercionFailed,
            format!("{} is not rational", c),
        )),
    }
}

/// `expand(e, **hints)` with the defaults of the function `expand`.
fn expand_hints(e: &Basic, hints: Hints) -> PolyResult<Basic> {
    expand_with(e, true, hints)
}

fn expand_complex(e: &Basic) -> PolyResult<Basic> {
    expand_hints(
        e,
        Hints {
            complex: true,
            ..Hints::DEFAULT
        },
    )
}

fn divisors(n: &Int) -> Vec<Int> {
    let mut result: Vec<Int> = vec![Int::ONE];
    for (p, e) in crate::ntheory::factor_::factorint(n, None) {
        let mut next: Vec<Int> = Vec::new();
        for d in &result {
            let mut power: Int = Int::ONE;
            for _ in 0..=e {
                next.push(d * &power);
                power *= &p;
            }
        }
        result = next;
    }
    result.sort();
    result
}

/// SymPy's `dom.is_Numerical`: `ZZ` or `QQ`.
fn numerical(f: &Poly) -> bool {
    f.domain.params.is_empty()
}

/// `_unevaluated_Add`: the sums flattened, the numbers summed first, the
/// other terms in the compare order, unevaluated.
fn unevaluated_add(args: &[Basic]) -> PolyResult<Basic> {
    let mut pending: Vec<Basic> = args.to_vec();
    let mut rest: Vec<Basic> = Vec::new();
    let mut number: Basic = Basic::zero();
    while let Some(a) = pending.pop() {
        if a.is_add() {
            pending.extend(a.args().iter().cloned());
        } else if a.is_number_class() {
            number = add(&[number, a])?;
        } else {
            rest.push(a);
        }
    }
    rest.sort_by(|a, b| a.compare_key().cmp(b.compare_key()));
    if !(number.is_number_class() && number.as_rational().is_some_and(|v| v.is_zero())) {
        rest.insert(0, number);
    }
    if rest.len() == 1 {
        return Ok(rest.remove(0));
    }
    Ok(Basic::new(Head::Add, rest))
}

/// `expand_2arg`: a number times a sum distributed, bottom up, the sum
/// unevaluated.
fn expand_2arg(e: &Basic) -> PolyResult<Basic> {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(expand_2arg(a)?);
    }
    let rebuilt: Basic = rebuild_evaluated(e, args)?;
    if rebuilt.is_mul() {
        let (c, r) = as_coeff_mul(&rebuilt);
        if r.is_add() && !is_integer_one(&c) {
            let mut terms: Vec<Basic> = Vec::new();
            for t in r.args() {
                terms.push(mul(&[c.clone(), t.clone()])?);
            }
            return unevaluated_add(&terms);
        }
    }
    Ok(rebuilt)
}

/// The root of a linear polynomial (`roots_linear`).
pub fn roots_linear(f: &Poly) -> PolyResult<Vec<Basic>> {
    let mut r: Basic = mul(&[int(-1), f.nth(&[0])?, inverse(&f.nth(&[1])?)?])?;
    if !numerical(f) {
        r = factor(&r, &[], false, true)?;
    }
    Ok(vec![r])
}

/// `sqrt(d)` with the even powers of the factors taken out (`_sqrt`).
fn sqrt_taken(d: &Basic) -> PolyResult<Basic> {
    let factors: Vec<Basic> = if d.is_mul() { d.args().to_vec() } else { vec![d.clone()] };
    let mut taken: Vec<Basic> = Vec::new();
    let mut other: Vec<Basic> = Vec::new();
    for di in factors {
        if di.is_pow() {
            if let Head::Rational {
                integer_class: true,
                value,
            } = di.args()[1].head()
            {
                let p: Int = numer(value);
                if (&p % Int::from(2u8)).is_zero() {
                    taken.push(pow(&di.args()[0], &Basic::integer(&p / Int::from(2u8)))?);
                    continue;
                }
            }
        }
        other.push(di);
    }
    if !taken.is_empty() {
        return mul(&[mul(&taken)?, sqrt(&mul(&other)?)?]);
    }
    sqrt(d)
}

/// The roots of a quadratic polynomial (`roots_quadratic`).
pub fn roots_quadratic(f: &Poly) -> PolyResult<Vec<Basic>> {
    let coefficients: Vec<Basic> = f.all_coeffs()?;
    let (a, b, c) = (&coefficients[0], &coefficients[1], &coefficients[2]);
    let numerical: bool = numerical(f);
    let simplified = |e: Basic| -> PolyResult<Basic> {
        if numerical {
            Ok(e)
        } else {
            factor(&e, &[], false, true)
        }
    };
    let (mut r0, mut r1): (Basic, Basic);
    if is_zero_number(c) {
        r0 = Basic::zero();
        r1 = mul(&[int(-1), b.clone(), inverse(a)?])?;
        if !numerical {
            r1 = simplified(r1)?;
        } else if is_true(&r1, Predicate::negative) {
            std::mem::swap(&mut r0, &mut r1);
        }
    } else if is_zero_number(b) {
        let r: Basic = simplified(mul(&[int(-1), c.clone(), inverse(a)?])?)?;
        let big_r: Basic = sqrt_taken(&r)?;
        r0 = neg(&big_r)?;
        r1 = big_r;
    } else {
        let mut d: Basic = add(&[pow(b, &int(2))?, mul(&[int(-4), a.clone(), c.clone()])?])?;
        let big_a: Basic = mul(&[int(2), a.clone()])?;
        let mut big_b: Basic = mul(&[int(-1), b.clone(), inverse(&big_a)?])?;
        if !numerical {
            d = simplified(d)?;
            big_b = simplified(big_b)?;
        }
        let big_d: Basic = crate::core::exprtools::factor_terms(
            &mul(&[sqrt_taken(&d)?, inverse(&big_a)?])?,
            false,
            false,
            false,
            true,
        )?;
        r0 = add(&[big_b.clone(), neg(&big_d)?])?;
        r1 = add(&[big_b, big_d])?;
        if is_true(a, Predicate::negative) {
            std::mem::swap(&mut r0, &mut r1);
        } else if !numerical {
            r0 = expand_2arg(&r0)?;
            r1 = expand_2arg(&r1)?;
        }
    }
    Ok(vec![r0, r1])
}

/// The roots of `a*x**n + b` (`roots_binomial`).
pub fn roots_binomial(f: &Poly) -> PolyResult<Vec<Basic>> {
    let n: i64 = f.degree(0)?;
    binomial(n, &f.nth(&[n as u32])?, &f.nth(&[0])?)
}

fn binomial(n: i64, a: &Basic, b: &Basic) -> PolyResult<Vec<Basic>> {
    use crate::functions::elementary::exponential::exp;
    let base: Basic = neg(&cancel(&mul(&[b.clone(), inverse(a)?])?, &[])?)?;
    let mut alpha: Basic = root(&base, n)?;
    if is_number(&alpha) {
        alpha = expand_complex(&alpha)?;
    }
    let negative_base: bool = is_true(&base, Predicate::negative);
    let even: bool = n % 2 == 0;
    let mut ks: Vec<i64> = Vec::new();
    let mut imax: i64 = n / 2;
    if even {
        ks.push(imax);
        imax -= 1;
    }
    if !negative_base {
        ks.push(0);
    }
    for i in (1..=imax).rev() {
        if negative_base {
            ks.extend([i, -i]);
        } else {
            ks.extend([-i, i]);
        }
    }
    if negative_base {
        ks.push(0);
    }
    let d: Basic = mul(&[
        int(2),
        Basic::imaginary_unit(),
        Basic::pi(),
        if n != 1 { q(1, n) } else { int(1) },
    ])?;
    let mut result: Vec<Basic> = Vec::new();
    for k in ks {
        let zeta: Basic = expand_complex(&exp(&mul(&[int(k), d.clone()])?)?)?;
        result.push(expand_hints(
            &mul(&[alpha.clone(), zeta])?,
            Hints {
                power_base: false,
                ..Hints::DEFAULT
            },
        )?);
    }
    Ok(result)
}

/// `_inv_totient_estimate`: bounds on `n` with `phi(n) = m` (floats as in
/// the Python).
fn inv_totient_estimate(m: i64) -> (i64, i64) {
    let primes: Vec<Int> = divisors(&Int::from(m))
        .into_iter()
        .map(|d| d + Int::ONE)
        .filter(crate::ntheory::primetest::isprime)
        .collect();
    let (mut a, mut b): (Int, Int) = (Int::ONE, Int::ONE);
    for p in &primes {
        a *= p;
        b *= p - Int::ONE;
    }
    let to_f64 = |x: &Int| -> f64 { x.to_string().parse::<f64>().unwrap_or(f64::INFINITY) };
    let lower: i64 = m;
    let upper: i64 = (m as f64 * (to_f64(&a) / to_f64(&b))).ceil() as i64;
    let (mut big_p, mut p): (Int, Int) = (Int::from(2u8), Int::from(2u8));
    let mut found: Vec<Int> = Vec::new();
    while big_p <= Int::from(upper) {
        p = crate::ntheory::generate::nextprime(&p);
        found.push(p.clone());
        big_p *= &p;
    }
    big_p = &big_p / &p;
    let mut b: Int = Int::ONE;
    for p in &found[..found.len().saturating_sub(1)] {
        b *= p - Int::ONE;
    }
    let upper: i64 = (m as f64 * (to_f64(&big_p) / to_f64(&b))).ceil() as i64;
    (lower, upper)
}

/// The roots of a cyclotomic polynomial (`roots_cyclotomic`).
pub fn roots_cyclotomic(f: &Poly) -> PolyResult<Vec<Basic>> {
    use crate::functions::elementary::exponential::exp;
    let (lower, upper) = inv_totient_estimate(f.degree(0)?);
    let mut coefficients: Vec<Int> = Vec::new();
    for c in f.all_coeffs()? {
        coefficients.push(numer(&rational(&c)?));
    }
    let Some(n) = (lower..=upper).find(|&n| dup_zz_cyclotomic_poly(n as usize) == coefficients) else {
        return Err(arithmetic_error(
            "failed to find index of a cyclotomic polynomial".to_string(),
        ));
    };
    let h: i64 = n / 2;
    let mut ks: Vec<i64> = (1..=n)
        .filter(|i| gcd(&Int::from(*i), &Int::from(n)) == Int::ONE)
        .collect();
    ks.sort_by_key(|&k| if k <= h { (k, -1) } else { ((k - n).abs(), 1) });
    let d: Basic = mul(&[
        int(2),
        Basic::imaginary_unit(),
        Basic::pi(),
        if n != 1 { q(1, n) } else { int(1) },
    ])?;
    let mut result: Vec<Basic> = Vec::new();
    for k in ks.into_iter().rev() {
        result.push(expand_complex(&exp(&mul(&[int(k), d.clone()])?)?)?);
    }
    Ok(result)
}

fn is_cyclotomic(f: &Poly) -> PolyResult<bool> {
    if !f.domain.params.is_empty() || f.gens.len() != 1 {
        return Ok(false);
    }
    let mut ints: Vec<Int> = Vec::new();
    for c in f.all_coeffs()? {
        let v: Rat = rational(&c)?;
        if !v.is_int() {
            return Ok(false);
        }
        ints.push(numer(&v));
    }
    dup_cyclotomic_p(&ints)
}

/// `dup_cyclotomic_p`.
fn dup_cyclotomic_p(f: &[Int]) -> PolyResult<bool> {
    let last: &Int = f.last().expect("a polynomial");
    if f[0] != Int::ONE || (*last != Int::ONE && *last != -Int::ONE) {
        return Ok(false);
    }
    let (coeff, factors) = dup_zz_factor(f, &ZZ)?;
    if coeff != Int::ONE || factors.len() != 1 || factors[0].0 != f || factors[0].1 != 1 {
        return Ok(false);
    }
    dup_cyclotomic_p_irreducible(f)
}

fn dup_cyclotomic_p_irreducible(f: &[Int]) -> PolyResult<bool> {
    let k: &IntegerRing = &ZZ;
    let n: usize = f.len() - 1;
    let g: Vec<Int> = (n % 2..=n).step_by(2).map(|i| f[i].clone()).collect();
    let h: Vec<Int> = if n >= 1 {
        ((n - 1) % 2..n).step_by(2).map(|i| f[i].clone()).collect()
    } else {
        Vec::new()
    };
    let g: Vec<Int> = dup_strip(&g, k);
    let g2: Vec<Int> = dup_mul(&g, &g, k);
    let mut h2: Vec<Int> = dup_mul(&h, &h, k);
    h2.push(Int::ZERO);
    let mut big_f: Vec<Int> = dup_sub(&g2, &h2, k);
    if big_f.first().is_some_and(is_negative) {
        big_f = dup_neg(&big_f, k);
    }
    if big_f == f {
        return Ok(true);
    }
    let mut m: Vec<Int> = dup_mirror(f, k);
    if m.first().is_some_and(is_negative) {
        m = dup_neg(&m, k);
    }
    if big_f == m && dup_cyclotomic_p(&m)? {
        return Ok(true);
    }
    let big_g: Vec<Int> = dup_sqf_part(&big_f, k);
    if dup_mul(&big_g, &big_g, k) == big_f && dup_cyclotomic_p(&big_g)? {
        return Ok(true);
    }
    Ok(false)
}

/// `_integer_basis`: the integer `b` with `f(x) = b**n g(x/b)` for an
/// integer `g`.
fn integer_basis(f: &Poly) -> PolyResult<Option<Int>> {
    let terms = f.terms()?;
    if terms.is_empty() {
        // the Python's IndexError on the zero polynomial
        return Err(value_error(
            "list index out of range (the roots of the zero polynomial)".to_string(),
        ));
    }
    let mut monoms: Vec<i64> = terms.iter().map(|(m, _)| i64::from(m[0])).collect();
    let mut coeffs: Vec<Int> = Vec::new();
    for (_, c) in &terms {
        coeffs.push(abs(&numer(&rational(c)?)));
    }
    if coeffs[0] < coeffs[coeffs.len() - 1] {
        coeffs.reverse();
        let n: i64 = monoms[0];
        monoms = monoms.iter().rev().map(|i| n - i).collect();
    } else {
        return Ok(None);
    }
    monoms.pop();
    coeffs.pop();
    if monoms.len() == 1 {
        let (r, exact) = integer_nthroot(&coeffs[0], monoms[0] as u32);
        return Ok(if exact { Some(r) } else { None });
    }
    let mut g: Int = Int::ZERO;
    for c in &coeffs {
        g = gcd(&g, c);
    }
    for div in divisors(&g).into_iter().skip(1).rev() {
        if monoms
            .iter()
            .zip(&coeffs)
            .all(|(m, c)| (c % div.pow(*m as usize)).is_zero())
        {
            return Ok(Some(div));
        }
    }
    Ok(None)
}

/// `integer_nthroot(y, n)`: the integer part of the `n`-th root and
/// whether it is exact.
fn integer_nthroot(y: &Int, n: u32) -> (Int, bool) {
    if *y == Int::ZERO || *y == Int::ONE || n == 1 {
        return (y.clone(), true);
    }
    if n == 2 {
        let r: Int = isqrt(y);
        let exact: bool = &r * &r == *y;
        return (r, exact);
    }
    let shift: usize = (bits(y) as usize).div_ceil(n as usize);
    let mut r: Int = Int::ONE << shift;
    loop {
        let s: Int = (Int::from(n - 1) * &r + y / r.pow(n as usize - 1)) / Int::from(n);
        if s >= r {
            break;
        }
        r = s;
    }
    while r.pow(n as usize) > *y {
        r -= Int::ONE;
    }
    while (&r + Int::ONE).pow(n as usize) <= *y {
        r += Int::ONE;
    }
    let exact: bool = r.pow(n as usize) == *y;
    (r, exact)
}

/// `(coeff, g)`: the roots of `f` are `coeff` times those of `g`
/// (`preprocess_roots`).
pub fn preprocess_roots(f: &Poly) -> PolyResult<(Basic, Poly)> {
    let mut coeff: Basic = Basic::one();
    let x: Basic = f.gens[0].clone();
    let expr: Basic = cleared_primitive(f)?;
    let mut poly: Poly = Poly::new(&expr, std::slice::from_ref(&x))?;
    let params: Vec<Basic> = poly.domain.params.clone();
    if !params.is_empty() && poly.coeffs()?.iter().all(|c| !c.is_add()) {
        let mut gens: Vec<Basic> = vec![x.clone()];
        gens.extend(params.iter().cloned());
        let mut injected: Poly = Poly::new(&poly.as_expr()?, &gens)?;
        let monoms = injected.monoms()?;
        let base: Vec<u32> = monoms.iter().map(|m| m[0]).collect();
        for (index, gen) in params.iter().enumerate() {
            let mut strip: Vec<u32> = monoms.iter().map(|m| m[index + 1]).collect();
            let mut reverse: bool = false;
            if strip[0] < strip[strip.len() - 1] {
                strip.reverse();
                reverse = true;
            }
            let mut ratio: Option<i64> = None;
            let mut ok: bool = true;
            for (a, b) in base.iter().zip(&strip) {
                if *a == 0 && *b == 0 {
                    continue;
                }
                if *a == 0 || *b == 0 || b % a != 0 {
                    ok = false;
                    break;
                }
                let current: i64 = i64::from(b / a);
                match ratio {
                    None => ratio = Some(current),
                    Some(r) if r != current => {
                        ok = false;
                        break;
                    }
                    _ => {}
                }
            }
            if let (true, Some(mut r)) = (ok, ratio) {
                if reverse {
                    r = -r;
                }
                let position: usize = injected.gens.iter().position(|g| g == gen).expect("a generator");
                injected = match injected.eval(position, &Basic::one())? {
                    PolyOrExpr::Poly(p) => p,
                    PolyOrExpr::Expr(e) => Poly::new(&e, std::slice::from_ref(&x))?,
                };
                coeff = mul(&[coeff, pow(gen, &int(-r))?])?;
            }
        }
        poly = Poly::new(&injected.as_expr()?, std::slice::from_ref(&x))?;
    }
    if poly.domain.params.is_empty() && poly.gens.len() == 1 && poly.domain.integral {
        if let Some(basis) = integer_basis(&poly)? {
            let n: i64 = poly.degree(0)?;
            let mut terms: Vec<Basic> = Vec::new();
            for (m, c) in poly.terms()? {
                let divisor: Int = basis.pow((n - i64::from(m[0])) as usize);
                terms.push(mul(&[
                    Basic::integer(div_floor(&numer(&rational(&c)?), &divisor)),
                    pow(&x, &int(i64::from(m[0])))?,
                ])?);
            }
            poly = Poly::new(&add(&terms)?, std::slice::from_ref(&x))?;
            coeff = mul(&[coeff, Basic::integer(basis)])?;
        }
    }
    Ok((coeff, poly))
}

/// `f` with the denominators cleared and the content divided out
/// (`_cleared_primitive`).
fn cleared_primitive(f: &Poly) -> PolyResult<Basic> {
    let mut d: Int = Int::ONE;
    for c in f.rep.terms.values() {
        let den: Int = denom(c);
        d = &d * &den / gcd(&d, &den);
    }
    let scale: Rat = rat_from_int(&d);
    let scaled: Vec<(Vec<u32>, Rat)> = f.rep.terms.iter().map(|(m, c)| (m.clone(), c * &scale)).collect();
    let mut g: Int = Int::ZERO;
    for (_, c) in &scaled {
        g = gcd(&g, &numer(c));
    }
    if g.is_zero() {
        g = Int::ONE;
    }
    let divisor: Rat = rat_from_int(&g);
    let rep = f
        .rep
        .ring
        .from_dict(scaled.into_iter().map(|(m, c)| (m, c / &divisor)).collect());
    let p: Poly = Poly {
        rep,
        gens: f.gens.clone(),
        domain: f.domain.clone(),
    };
    let mut content: Basic = Basic::one();
    if !f.domain.params.is_empty() {
        content = parameter_content(&p)?;
    }
    let mut expr: Basic = p.as_expr()?;
    if !is_integer_one(&content) {
        expr = cancel(&mul(&[expr, inverse(&content)?])?, &[])?;
    }
    Ok(expr)
}

fn parameter_content(p: &Poly) -> PolyResult<Basic> {
    use super::domains::sparsepolynomialring::IteratedPolynomialRing;
    let n: usize = p.gens.len();
    let names: Vec<String> = p.rep.ring.names.clone();
    let ring = IteratedPolynomialRing::new(
        SparsePolynomialRing::sympy(ZZ, names[n..].to_vec()),
        names[..n].to_vec(),
    );
    let element = ring
        .ring()
        .from_dict(p.rep.terms.iter().map(|(m, c)| (m.clone(), numer(c))).collect());
    let content = ring.ground_content(&element);
    let terms: BTreeMap<Vec<u32>, Rat> = content
        .terms
        .iter()
        .map(|(m, c)| (m.clone(), rat_from_int(c)))
        .collect();
    p.coefficient_expr(&terms)
}

/// `[g1, ..., gn]` with `f = g1(g2(...gn))` over `QQ` (`dup_decompose`; a
/// polynomial domain is not decomposed).
fn decompose(f: &Poly) -> PolyResult<Vec<Poly>> {
    if !f.domain.params.is_empty() {
        return Ok(vec![f.clone()]);
    }
    let x: Basic = f.gens[0].clone();
    let mut dense: Vec<Rat> = Vec::new();
    for c in f.all_coeffs()? {
        dense.push(rational(&c)?);
    }
    let mut parts: Vec<Vec<Rat>> = Vec::new();
    while let Some((g, h)) = decompose_once(&dense)? {
        dense = g;
        parts.insert(0, h);
    }
    let mut result: Vec<Poly> = Vec::new();
    for p in std::iter::once(&dense).chain(parts.iter()) {
        result.push(Poly::new(&from_dense(p, &x)?, std::slice::from_ref(&x))?);
    }
    Ok(result)
}

fn from_dense(f: &[Rat], x: &Basic) -> PolyResult<Basic> {
    let n: usize = f.len() - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in f.iter().enumerate() {
        if !c.is_zero() {
            terms.push(mul(&[Basic::from_rational(c.clone()), pow(x, &int((n - i) as i64))?])?);
        }
    }
    add(&terms)
}

fn decompose_once(f: &[Rat]) -> PolyResult<Option<(Vec<Rat>, Vec<Rat>)>> {
    let df: usize = f.len() - 1;
    for s in 2..df {
        if df % s != 0 {
            continue;
        }
        let h: Vec<Rat> = right_decompose(f, s);
        if let Some(g) = left_decompose(f, &h)? {
            return Ok(Some((g, h)));
        }
    }
    Ok(None)
}

fn right_decompose(f: &[Rat], s: usize) -> Vec<Rat> {
    let n: usize = f.len() - 1;
    let lc: &Rat = &f[0];
    let raw: HashMap<usize, Rat> = f
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| (n - i, c.clone()))
        .collect();
    let mut g: HashMap<usize, Rat> = HashMap::new();
    g.insert(s, Rat::ONE);
    let r: usize = n / s;
    for i in 1..s {
        let mut coeff: Rat = Rat::ZERO;
        for j in 0..i {
            let (Some(a), Some(b)) = (raw.get(&(n + j - i)), g.get(&(s - j))) else {
                continue;
            };
            coeff += rat_from_int(&(Int::from(i as i64) - Int::from((r * j) as i64))) * a * b;
        }
        let value: Rat = coeff / (rat_from_int(&Int::from((i * r) as i64)) * lc);
        g.insert(s - i, value);
    }
    let dense: Vec<Rat> = (0..=s).rev().map(|k| g.get(&k).cloned().unwrap_or(Rat::ZERO)).collect();
    dup_strip(&dense, &QQ)
}

fn left_decompose(f: &[Rat], h: &[Rat]) -> PolyResult<Option<Vec<Rat>>> {
    let k: &RationalField = &QQ;
    let mut g: BTreeMap<usize, Rat> = BTreeMap::new();
    let mut i: usize = 0;
    let mut f: Vec<Rat> = f.to_vec();
    while !f.is_empty() {
        let (quotient, remainder) = dup_div(&f, h, k)?;
        if remainder.len() > 1 {
            return Ok(None);
        }
        g.insert(i, remainder.first().cloned().unwrap_or(Rat::ZERO));
        f = quotient;
        i += 1;
    }
    let top: usize = g.keys().max().copied().unwrap_or(0);
    let dense: Vec<Rat> = (0..=top)
        .rev()
        .map(|k| g.get(&k).cloned().unwrap_or(Rat::ZERO))
        .collect();
    Ok(Some(dup_strip(&dense, k)))
}

fn listed(result: &RootsResult) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for zero in ordered(result.keys().cloned().collect()) {
        let m: usize = *result.get(&zero).expect("a root");
        out.extend(std::iter::repeat(zero).take(m));
    }
    out
}

/// `roots([c_n, ..., c_0], multiple=True)` in a dummy variable
/// (`_roots_of_coefficients`).
fn roots_of_coefficients(coefficients: &[Basic], cubics: bool) -> PolyResult<Vec<Basic>> {
    let x: Basic = Dummy::new("x", &[]).0;
    let n: usize = coefficients.len() - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in coefficients.iter().enumerate() {
        terms.push(mul(&[c.clone(), pow(&x, &int((n - i) as i64))?])?);
    }
    let f: Poly = Poly::new(&add(&terms)?, std::slice::from_ref(&x))?;
    let result: RootsResult = roots_impl(&f, cubics, true, None, true)?;
    Ok(listed(&result))
}

/// `_mexpand`: `expand_mul(expand_multinomial(e))`.
fn mexpand(e: &Basic) -> PolyResult<Basic> {
    crate::core::function::expand_mul(&crate::core::function::expand_multinomial(e, true)?, true)
}

/// The roots of a cubic by Cardano's formula (`roots_cubic`, `trig=False`).
pub fn roots_cubic(f: &Poly) -> PolyResult<Vec<Basic>> {
    let coefficients: Vec<Basic> = f.monic()?.all_coeffs()?;
    let (a, b, c) = (&coefficients[1], &coefficients[2], &coefficients[3]);
    if is_zero_number(c) {
        let found: Vec<Basic> = roots_of_coefficients(&[int(1), a.clone(), b.clone()], true)?;
        return Ok(vec![found[0].clone(), Basic::zero(), found[1].clone()]);
    }
    let p: Basic = add(&[b.clone(), mul(&[int(-1), pow(a, &int(2))?, q(1, 3)])?])?;
    let qq: Basic = add(&[
        c.clone(),
        mul(&[int(-1), a.clone(), b.clone(), q(1, 3)])?,
        mul(&[q(2, 27), pow(a, &int(3))?])?,
    ])?;
    let pon3: Basic = mul(&[p.clone(), q(1, 3)])?;
    let aon3: Basic = mul(&[a.clone(), q(1, 3)])?;
    let mut u1: Option<Basic> = None;
    if is_zero_number(&p) {
        if is_zero_number(&qq) {
            let r: Basic = neg(&aon3)?;
            return Ok(vec![r.clone(), r.clone(), r]);
        }
        u1 = Some(if is_true(&qq, Predicate::positive) {
            neg(&root(&qq, 3)?)?
        } else {
            root(&neg(&qq)?, 3)?
        });
    } else if is_zero_number(&qq) {
        let found: Vec<Basic> = roots_of_coefficients(&[int(1), int(0), p.clone()], true)?;
        let minus: Basic = neg(&aon3)?;
        let mut out: Vec<Basic> = Vec::new();
        for t in [found[0].clone(), Basic::zero(), found[1].clone()] {
            out.push(add(&[t, minus.clone()])?);
        }
        return Ok(out);
    } else if is_true(&qq, Predicate::real) && is_true(&qq, Predicate::negative) {
        let inner: Basic = add(&[
            mul(&[int(-1), qq.clone(), q(1, 2)])?,
            sqrt(&add(&[mul(&[pow(&qq, &int(2))?, q(1, 4)])?, pow(&pon3, &int(3))?])?)?,
        ])?;
        u1 = Some(neg(&root(&inner, 3)?)?);
    }
    let coeff: Basic = mul(&[Basic::imaginary_unit(), sqrt(&int(3))?, q(1, 2)])?;
    let Some(u1) = u1 else {
        let units: Vec<Basic> = vec![
            int(1),
            add(&[q(-1, 2), coeff.clone()])?,
            add(&[q(-1, 2), neg(&coeff)?])?,
        ];
        let (bb, cc, dd) = (a, b, c);
        let d0: Basic = add(&[pow(bb, &int(2))?, mul(&[int(-3), cc.clone()])?])?;
        let d1: Basic = add(&[
            mul(&[int(2), pow(bb, &int(3))?])?,
            mul(&[int(-9), bb.clone(), cc.clone()])?,
            mul(&[int(27), dd.clone()])?,
        ])?;
        let discriminant: Basic = sqrt(&add(&[pow(&d1, &int(2))?, mul(&[int(-4), pow(&d0, &int(3))?])?])?)?;
        let big_c: Basic = root(&mul(&[add(&[d1.clone(), discriminant])?, q(1, 2)])?, 3)?;
        let mut out: Vec<Basic> = Vec::new();
        for uk in units {
            let sum: Basic = add(&[
                bb.clone(),
                mul(&[uk.clone(), big_c.clone()])?,
                mul(&[d0.clone(), inverse(&big_c)?, inverse(&uk)?])?,
            ])?;
            out.push(mul(&[int(-1), sum, q(1, 3)])?);
        }
        return Ok(out);
    };
    let u2: Basic = mul(&[u1.clone(), add(&[q(-1, 2), coeff.clone()])?])?;
    let u3: Basic = mul(&[u1.clone(), add(&[q(-1, 2), neg(&coeff)?])?])?;
    let minus_aon3: Basic = neg(&aon3)?;
    let mut out: Vec<Basic> = Vec::new();
    for u in [u1, u2, u3] {
        if is_zero_number(&p) {
            out.push(add(&[u, minus_aon3.clone()])?);
        } else {
            out.push(add(&[
                neg(&u)?,
                mul(&[pon3.clone(), inverse(&u)?])?,
                minus_aon3.clone(),
            ])?);
        }
    }
    Ok(out)
}

/// Euler's resolvent with a rational root (`_roots_quartic_euler`).
fn roots_quartic_euler(p: &Basic, qq: &Basic, r: &Basic, a: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let x: Basic = Dummy::new("x", &[]).0;
    let eq: Basic = add(&[
        mul(&[int(64), pow(&x, &int(3))?])?,
        mul(&[int(32), p.clone(), pow(&x, &int(2))?])?,
        mul(&[
            add(&[mul(&[int(4), pow(p, &int(2))?])?, mul(&[int(-16), r.clone()])?])?,
            x.clone(),
        ])?,
        mul(&[int(-1), pow(qq, &int(2))?])?,
    ])?;
    let found: RootsResult = roots_impl(&Poly::new(&eq, std::slice::from_ref(&x))?, false, true, None, false)?;
    let candidates: Vec<Basic> = found
        .keys()
        .filter(|s| is_true(s, Predicate::rational) && is_true(s, Predicate::nonzero))
        .cloned()
        .collect();
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut big_r: Basic = candidates[0].clone();
    for s in &candidates[1..] {
        if rational(s)? > rational(&big_r)? {
            big_r = s.clone();
        }
    }
    let c1: Basic = sqrt(&big_r)?;
    let big_b: Basic = mul(&[
        int(-1),
        qq.clone(),
        c1.clone(),
        inverse(&mul(&[int(4), big_r.clone()])?)?,
    ])?;
    let big_a: Basic = add(&[neg(&big_r)?, mul(&[int(-1), p.clone(), q(1, 2)])?])?;
    let c2: Basic = sqrt(&add(&[big_a.clone(), big_b.clone()])?)?;
    let c3: Basic = sqrt(&add(&[big_a, neg(&big_b)?])?)?;
    let minus_a: Basic = neg(a)?;
    Ok(Some(vec![
        add(&[c1.clone(), neg(&c2)?, minus_a.clone()])?,
        add(&[neg(&c1)?, neg(&c3)?, minus_a.clone()])?,
        add(&[neg(&c1)?, c3, minus_a.clone()])?,
        add(&[c1, c2, minus_a])?,
    ]))
}

/// The roots of a quartic by Ferrari's method (`roots_quartic`).
pub fn roots_quartic(f: &Poly) -> PolyResult<Vec<Basic>> {
    let coefficients: Vec<Basic> = f.monic()?.all_coeffs()?;
    let (a, b, c, d) = (&coefficients[1], &coefficients[2], &coefficients[3], &coefficients[4]);
    if is_zero_number(d) {
        let mut out: Vec<Basic> = vec![Basic::zero()];
        out.extend(roots_of_coefficients(&[int(1), a.clone(), b.clone(), c.clone()], true)?);
        return Ok(out);
    }
    if !is_zero_number(a) && pow(&mul(&[c.clone(), inverse(a)?])?, &int(2))? == *d {
        let x: Basic = f.gens[0].clone();
        let m: Basic = mul(&[c.clone(), inverse(a)?])?;
        let gens = std::slice::from_ref(&x);
        let gq: Poly = Poly::new(
            &add(&[
                pow(&x, &int(2))?,
                mul(&[a.clone(), x.clone()])?,
                b.clone(),
                mul(&[int(-2), m.clone()])?,
            ])?,
            gens,
        )?;
        let found: Vec<Basic> = roots_quadratic(&gq)?;
        let h1: Poly = Poly::new(
            &add(&[
                pow(&x, &int(2))?,
                mul(&[int(-1), found[0].clone(), x.clone()])?,
                m.clone(),
            ])?,
            gens,
        )?;
        let h2: Poly = Poly::new(
            &add(&[pow(&x, &int(2))?, mul(&[int(-1), found[1].clone(), x.clone()])?, m])?,
            gens,
        )?;
        let mut out: Vec<Basic> = roots_quadratic(&h1)?;
        out.extend(roots_quadratic(&h2)?);
        return Ok(out);
    }
    let a2: Basic = pow(a, &int(2))?;
    let e: Basic = add(&[b.clone(), mul(&[int(-3), a2.clone(), q(1, 8)])?])?;
    let ff: Basic = mexpand(&add(&[
        c.clone(),
        mul(&[
            a.clone(),
            add(&[mul(&[a2.clone(), q(1, 8)])?, mul(&[int(-1), b.clone(), q(1, 2)])?])?,
        ])?,
    ])?)?;
    let aon4: Basic = mul(&[a.clone(), q(1, 4)])?;
    let inner: Basic = add(&[
        mul(&[
            a.clone(),
            add(&[
                mul(&[int(3), a2.clone(), q(1, 64)])?,
                mul(&[int(-1), b.clone(), q(1, 4)])?,
            ])?,
        ])?,
        c.clone(),
    ])?;
    let g: Basic = mexpand(&add(&[d.clone(), mul(&[int(-1), aon4.clone(), inner])?])?)?;
    let minus_aon4: Basic = neg(&aon4)?;
    let shifted = |ys: Vec<Basic>| -> PolyResult<Vec<Basic>> {
        let mut out: Vec<Basic> = Vec::new();
        for t in ys {
            out.push(add(&[t, minus_aon4.clone()])?);
        }
        Ok(out)
    };
    if is_true(&ff, Predicate::zero) {
        let found: Vec<Basic> = roots_of_coefficients(&[int(1), e.clone(), g.clone()], true)?;
        let (y1, y2) = (sqrt(&found[0])?, sqrt(&found[1])?);
        return shifted(vec![neg(&y1)?, neg(&y2)?, y1, y2]);
    }
    if is_true(&g, Predicate::zero) {
        let mut ys: Vec<Basic> = vec![Basic::zero()];
        ys.extend(roots_of_coefficients(&[int(1), int(0), e.clone(), ff.clone()], true)?);
        return shifted(ys);
    }
    if let Some(solutions) = roots_quartic_euler(&e, &ff, &g, &aon4)? {
        if !solutions.is_empty() {
            return Ok(solutions);
        }
    }
    let p: Basic = add(&[mul(&[int(-1), pow(&e, &int(2))?, q(1, 12)])?, neg(&g)?])?;
    let qq: Basic = add(&[
        mul(&[int(-1), pow(&e, &int(3))?, q(1, 108)])?,
        mul(&[e.clone(), g.clone(), q(1, 3)])?,
        mul(&[int(-1), pow(&ff, &int(2))?, q(1, 8)])?,
    ])?;
    let third: Basic = q(1, 3);
    let answers = |y: &Basic| -> PolyResult<Vec<Basic>> {
        let w: Basic = sqrt(&add(&[e.clone(), mul(&[int(2), y.clone()])?])?)?;
        let arg1: Basic = add(&[mul(&[int(3), e.clone()])?, mul(&[int(2), y.clone()])?])?;
        let arg2: Basic = mul(&[int(2), ff.clone(), inverse(&w)?])?;
        let mut result: Vec<Basic> = Vec::new();
        for s in [-1, 1] {
            let r0: Basic = sqrt(&mul(&[int(-1), add(&[arg1.clone(), mul(&[int(s), arg2.clone()])?])?])?)?;
            for t in [-1, 1] {
                let half: Basic = mul(&[
                    add(&[mul(&[int(s), w.clone()])?, mul(&[int(-t), r0.clone()])?])?,
                    q(1, 2),
                ])?;
                result.push(add(&[half, minus_aon4.clone()])?);
            }
        }
        Ok(result)
    };
    let p: Basic = mexpand(&p)?;
    let y1: Basic = add(&[mul(&[e.clone(), q(-5, 6)])?, neg(&pow(&qq, &third)?)?])?;
    if is_true(&p, Predicate::zero) {
        return answers(&y1);
    }
    let r_root: Basic = sqrt(&add(&[
        mul(&[pow(&qq, &int(2))?, q(1, 4)])?,
        mul(&[pow(&p, &int(3))?, q(1, 27)])?,
    ])?)?;
    let r: Basic = add(&[mul(&[int(-1), qq.clone(), q(1, 2)])?, r_root])?;
    let u: Basic = pow(&r, &third)?;
    let y2: Basic = add(&[
        mul(&[e.clone(), q(-5, 6)])?,
        u.clone(),
        mul(&[int(-1), p.clone(), inverse(&u)?, q(1, 3)])?,
    ])?;
    if is_false(&p, Predicate::zero) {
        return answers(&y2);
    }
    Err(not_ported(
        "Piecewise (the roots of a quartic whose resolvent may vanish)",
    ))
}

/// The roots of `f` in its generators `gens` with their multiplicities
/// (`roots`; `multiple` lists each root as many times, see [`roots_list`]).
pub fn roots(f: &Basic, gens: &[Basic], cubics: bool, quartics: bool, filter: RootFilter) -> PolyResult<RootsResult> {
    let p: Poly = match Poly::new(f, gens) {
        Ok(p) => p,
        Err(PolyError::Polys(PolysErrorKind::GeneratorsNeeded, _)) => return Ok(RootsResult::new()),
        Err(e) => return Err(e),
    };
    if !p.gens[0].is_symbol() {
        return Err(polys_error(
            PolysErrorKind::PolynomialError,
            "generator must be a Symbol".to_string(),
        ));
    }
    roots_impl(&p, cubics, quartics, filter, false)
}

/// `roots(..., multiple=True)`: each root repeated by its multiplicity, in
/// SymPy's `ordered` order.
pub fn roots_list(
    f: &Basic,
    gens: &[Basic],
    cubics: bool,
    quartics: bool,
    filter: RootFilter,
) -> PolyResult<Vec<Basic>> {
    Ok(listed(&roots(f, gens, cubics, quartics, filter)?))
}

fn roots_impl(f: &Poly, cubics: bool, quartics: bool, filter: RootFilter, from_list: bool) -> PolyResult<RootsResult> {
    let n: i64 = f.degree(0)?;
    if !from_list && f.length() == 2 && n > 2 {
        if let Some(rescaled) = rescaled_binomial(f, cubics, quartics, filter)? {
            return Ok(rescaled);
        }
    }
    if f.gens.len() > 1 {
        return Err(polys_error(
            PolysErrorKind::PolynomialError,
            "multivariate polynomials are not supported".to_string(),
        ));
    }
    // SymPy renames the generator to Dummy('x0')
    let x: Basic = Dummy::new("x0", &[]).0;
    let gens = std::slice::from_ref(&x);
    let mut renamed: HashMap<Basic, Basic> = HashMap::new();
    renamed.insert(f.gens[0].clone(), x.clone());
    let mut f: Poly = Poly::new(&xreplace_evaluated(&f.as_expr()?, &renamed)?, gens)?;
    let k: u32 = if f.is_zero() {
        0
    } else {
        f.monoms()?.iter().map(|m| m[0]).min().unwrap_or(0)
    };
    if k > 0 {
        f = Poly::new(
            &cancel(&mul(&[f.as_expr()?, pow(&x, &int(-i64::from(k)))?])?, &[])?,
            gens,
        )?;
    }
    let mut zeros: RootsResult = RootsResult::new();
    if k > 0 {
        zeros.insert(Basic::zero(), k as usize);
    }
    let (coeff, f) = preprocess_roots(&f)?;
    let mut result: RootsResult = RootsResult::new();
    let mut update = |r: Basic, multiplicity: usize, zeros: &mut RootsResult| {
        if is_zero_number(&r) {
            let total: usize = zeros.get(&Basic::zero()).copied().unwrap_or(0) + multiplicity;
            zeros.insert(Basic::zero(), total);
        }
        let total: usize = result.get(&r).copied().unwrap_or(0) + multiplicity;
        result.insert(r, total);
    };
    if !f.is_ground() {
        if f.degree(0)? == 1 {
            update(roots_linear(&f)?.remove(0), 1, &mut zeros);
        } else if f.length() == 2 {
            let found: Vec<Basic> = if f.degree(0)? == 2 {
                roots_quadratic(&f)?
            } else {
                roots_binomial(&f)?
            };
            for r in found {
                update(r, 1, &mut zeros);
            }
        } else {
            let (_, factors) = Poly::new(&f.as_expr()?, &[])?.factor_list()?;
            if factors.len() == 1 && f.degree(0)? == 2 {
                for r in roots_quadratic(&f)? {
                    update(r, 1, &mut zeros);
                }
            } else if factors.len() == 1 && factors[0].1 == 1 {
                for r in try_decompose(&f, cubics, quartics)? {
                    update(r, 1, &mut zeros);
                }
            } else {
                for (g, m) in factors {
                    for r in try_heuristics(&Poly::new(&g.as_expr()?, gens)?, cubics, quartics)? {
                        update(r, m, &mut zeros);
                    }
                }
            }
        }
    }
    if !is_integer_one(&coeff) {
        let mut scaled: RootsResult = RootsResult::new();
        for (r, m) in result.iter() {
            scaled.insert(mul(&[coeff.clone(), r.clone()])?, *m);
        }
        result = scaled;
    }
    if let Some(name) = filter.filter(|name| *name != "C") {
        let mut kept: RootsResult = RootsResult::new();
        for (r, m) in result.iter() {
            let keep: bool = match name {
                "Z" => matches!(
                    r.head(),
                    Head::Rational {
                        integer_class: true,
                        ..
                    }
                ),
                "Q" => r.as_rational().is_some(),
                "R" => {
                    let (nn, dd) = crate::core::expr::as_numer_denom(r)?;
                    is_true(&nn, Predicate::real) && is_true(&dd, Predicate::real)
                }
                "I" => is_true(r, Predicate::imaginary),
                _ => return Err(value_error(format!("Invalid filter: {}", name))),
            };
            if keep {
                kept.insert(r.clone(), *m);
            }
        }
        result = kept;
    }
    for (r, m) in zeros.iter() {
        result.insert(r.clone(), *m);
    }
    Ok(result)
}

/// SymPy's check for `c*gen**m - foo**n`: the integer powers of sums in
/// the constant replaced by positive dummies (`_rescaled_binomial`).
fn rescaled_binomial(f: &Poly, cubics: bool, quartics: bool, filter: RootFilter) -> PolyResult<Option<RootsResult>> {
    if f.gens.len() != 1 {
        return Err(not_ported("roots of a binomial in several generators"));
    }
    let (con, dep) = as_independent(&f.as_expr()?, &f.gens[0], true)?;
    let fcon: Basic = neg(&factor(&neg(&con)?, &[], false, true)?)?;
    if fcon == con {
        return Ok(None);
    }
    let factors: Vec<Basic> = if fcon.is_mul() {
        fcon.args().to_vec()
    } else {
        vec![fcon.clone()]
    };
    let mut bases: Vec<(Basic, Basic)> = Vec::new();
    for i in factors {
        if i.is_pow() {
            let (b, e) = crate::core::expr::as_base_exp(&i)?;
            if matches!(
                e.head(),
                Head::Rational {
                    integer_class: true,
                    ..
                }
            ) && b.is_add()
            {
                bases.push((b, Dummy::new("Dummy", &[(Predicate::positive, true)]).0));
            }
        }
    }
    if bases.is_empty() {
        return Ok(None);
    }
    let rule: HashMap<Basic, Basic> = bases.iter().cloned().collect();
    let replaced: Basic = xreplace_evaluated(&add(&[dep, fcon])?, &rule)?;
    let inner: RootsResult = roots_impl(&Poly::new(&replaced, &f.gens)?, cubics, quartics, filter, false)?;
    let back: HashMap<Basic, Basic> = bases.into_iter().map(|(b, d)| (d, b)).collect();
    let mut out: RootsResult = RootsResult::new();
    for (r, m) in inner.iter() {
        let restored: Basic = xreplace_evaluated(r, &back)?;
        out.insert(
            crate::core::exprtools::factor_terms(&restored, false, false, false, true)?,
            *m,
        );
    }
    Ok(Some(out))
}

fn try_decompose(f: &Poly, cubics: bool, quartics: bool) -> PolyResult<Vec<Basic>> {
    let parts: Vec<Poly> = decompose(f)?;
    let mut found: Vec<Basic> = try_heuristics(&parts[0], cubics, quartics)?;
    for g in &parts[1..] {
        let previous: Vec<Basic> = std::mem::take(&mut found);
        for r in previous {
            let shifted: Basic = add(&[g.as_expr()?, neg(&r)?])?;
            match Poly::new(&shifted, std::slice::from_ref(&f.gens[0])) {
                Ok(h) => found.extend(try_heuristics(&h, cubics, quartics)?),
                Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
                    found.extend(ex_heuristics(&shifted, &f.gens[0])?)
                }
                Err(e) => return Err(e),
            }
        }
    }
    Ok(found)
}

/// `_try_heuristics` over `EX` for monomials and binomials (`_ex_heuristics`).
fn ex_heuristics(e: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let mut coefficients: BTreeMap<u32, Basic> = BTreeMap::new();
    for (m, cs) in decompose_with_gens(e, std::slice::from_ref(x))? {
        let c: Basic = add(&cs)?;
        if !(c.is_number_class() && c.as_rational().is_some_and(|v| v.is_zero())) {
            coefficients.insert(m[0], c);
        }
    }
    let n: u32 = coefficients.keys().max().copied().unwrap_or(0);
    if n == 0 {
        return Ok(Vec::new());
    }
    if coefficients.len() == 1 {
        return Ok(vec![Basic::zero(); n as usize]);
    }
    if coefficients.len() == 2 && coefficients.contains_key(&0) && n >= 2 {
        return binomial(i64::from(n), &coefficients[&n], &coefficients[&0]);
    }
    Err(not_ported("roots over the algebraic domain EX need simplify"))
}

fn try_heuristics(f: &Poly, cubics: bool, quartics: bool) -> PolyResult<Vec<Basic>> {
    if f.is_ground() {
        return Ok(Vec::new());
    }
    if f.length() == 1 {
        return Ok(vec![Basic::zero(); f.degree(0)? as usize]);
    }
    let cancelled = |rs: Vec<Basic>| -> PolyResult<Vec<Basic>> {
        let mut out: Vec<Basic> = Vec::new();
        for r in rs {
            out.push(cancel(&r, &[])?);
        }
        Ok(out)
    };
    if f.length() == 2 {
        if f.degree(0)? == 1 {
            return cancelled(roots_linear(f)?);
        }
        return roots_binomial(f);
    }
    let mut result: Vec<Basic> = Vec::new();
    let x: Basic = f.gens[0].clone();
    let mut f: Poly = f.clone();
    for i in [-1, 1] {
        if let PolyOrExpr::Expr(value) = f.eval(0, &int(i))? {
            if is_zero_number(&value) {
                f = f.exquo(&Poly::new(&add(&[x.clone(), int(-i)])?, std::slice::from_ref(&x))?)?;
                result.push(int(i));
                break;
            }
        }
    }
    let n: i64 = f.degree(0)?;
    if n == 1 {
        result.extend(cancelled(roots_linear(&f)?)?);
    } else if n == 2 {
        result.extend(cancelled(roots_quadratic(&f)?)?);
    } else if is_cyclotomic(&f)? {
        result.extend(roots_cyclotomic(&f)?);
    } else if n == 3 && cubics {
        result.extend(roots_cubic(&f)?);
    } else if n == 4 && quartics {
        result.extend(roots_quartic(&f)?);
    }
    Ok(result)
}
