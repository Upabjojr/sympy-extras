//! Port of `sympy_extras/integrals/validated.py`: validated numerical
//! integration, a value with a rigorous error bound.
//!
//! Every panel of a subdivision of the range is integrated by the Simpson
//! rule (error at most `h**5/2880 max |f''''|`) or the five-point
//! Gauss–Legendre rule (error at most `h**11 (5!)**4/(11 (10!)**3) max
//! |f**(10)|`), whichever bound is smaller; the derivatives and the values
//! at the nodes are enclosed by the interval arithmetic of
//! [`crate::assumptions::ivarith`] (mpmath's `iv`), so that the sum over
//! the panels is an enclosure of the integral. The panels are bisected
//! where the bound is largest until the total is below the tolerance. A
//! panel without a finite derivative bound is bounded by its width times
//! the range of `f`; an endpoint where even the range is infinite by the
//! model `Integral(|c(v)| v**alpha |log v|**j, (v, 0, h))`; removable
//! singularities through the Taylor remainder; algebraic endpoint
//! singularities are removed by `x = a + (b - a) u**q`; infinite ranges
//! are mapped onto `[0, 1]` (algebraic or exponential decay, a stretched
//! exponential after `x = s**(1/k)`), and oscillatory tails are integrated
//! by parts with Bonnet's bound on the remainder.
//!
//! The differences with Python: the errors and tolerances are exact
//! dyadic rationals (rounded up to 53 bits) instead of mpmath floats, the
//! final enclosure is widened by outward rounding instead of mpmath's
//! nearest rounding, the nodes of the Gauss rule come from Newton's method
//! in exact rationals instead of `findroot`, and the result is the exact
//! midpoint and radius of the enclosure instead of `Float`s (the core has
//! none). The enclosures are therefore rigorous but not identical, bit for
//! bit, to Python's.

#![allow(clippy::mutable_key_type)]

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::time::{Duration, Instant};

use crate::assumptions::intervals::{evaluate, iv_prec, PrecisionGuard};
use crate::assumptions::ivarith::{round, Iv, Rnd, E};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{denom, numer, rat_is_negative, Int, Rat};

type R = PolyResult<Basic>;
type OptIv = PolyResult<Option<Iv>>;

/// The largest number of panels before the refinement stops.
const MAX_PANELS: usize = 6000;
/// The nodes of the Gauss–Legendre rule.
const GAUSS_NODES: usize = 5;
/// The largest size of a derivative still enclosed on every panel.
const MAX_OPS: i64 = 5000;
/// How many times an oscillatory tail is integrated by parts at most.
const MAX_PARTS: usize = 10;

const FUNCTIONS: [Function; 10] = [
    Function::Exp,
    Function::Log,
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Atan,
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Abs,
];

thread_local! {
    /// When the whole computation must stop refining (`_deadline`).
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    static RULES: RefCell<HashMap<(usize, usize), Option<Rule>>> = RefCell::new(HashMap::new());
    static REMOVABLE: RefCell<HashMap<(Basic, Basic, Basic), Option<Basic>>> = RefCell::new(HashMap::new());
}

// -- small helpers

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    Basic::zero()
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
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn at(e: &Basic, x: &Basic, c: &Basic) -> R {
    crate::core::subs::subs(e, x, c)
}
fn diff_n(e: &Basic, x: &Basic, n: u64) -> R {
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn expand_power_exp(e: &Basic) -> R {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: true,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn together(e: &Basic) -> R {
    crate::polys::rationaltools::together(e, false, true)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn rat_basic(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}
fn add_args(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn has_function(e: &Basic, fs: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(f) if fs.contains(f)))
}
fn function_of(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}

/// The time allowed to one symbolic preparation (`_budget`).
fn budget() -> Option<f64> {
    crate::settings::settings().timeout.map(|t| t / 8.0)
}

fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(budget(), f)
}

fn out_of_time() -> bool {
    DEADLINE.with(|d| d.get()).is_some_and(|d| Instant::now() > d)
}

fn p() -> usize {
    iv_prec()
}

fn finite(v: &Option<Iv>) -> bool {
    matches!(v, Some(Iv { a: E::F(_), b: E::F(_) }))
}

fn ivi(n: i64) -> Iv {
    Iv::from_int(&Int::from(n), p())
}

/// A rational number as an interval (`_interval`).
fn ivr(r: &Rat) -> Iv {
    let n: Iv = Iv::from_int(&numer(r), p());
    let d: Int = denom(r);
    if d == Int::ONE {
        return n;
    }
    n.div(&Iv::from_int(&d, p()), p())
}

fn iadd(a: &Iv, b: &Iv) -> Iv {
    a.add(b, p())
}
fn isub(a: &Iv, b: &Iv) -> Iv {
    a.sub(b, p())
}
fn imul(a: &Iv, b: &Iv) -> Iv {
    a.mul(b, p())
}
fn idiv(a: &Iv, b: &Iv) -> Iv {
    a.div(b, p())
}
fn ipow(a: &Iv, n: i64) -> Iv {
    a.pow_int(n, p())
}
fn iabs(a: &Iv) -> Iv {
    a.abs(p())
}

fn e_add(a: &E, b: &E) -> E {
    match (a, b) {
        (E::F(x), E::F(y)) => E::F(round(&(x + y), 53, Rnd::Ceil)),
        (E::NInf, _) | (_, E::NInf) => E::NInf,
        _ => E::PInf,
    }
}

fn e_half_width(v: &Iv) -> E {
    match (&v.a, &v.b) {
        (E::F(a), E::F(b)) => E::F(round(&((b - a) / Rat::from(2)), 53, Rnd::Ceil)),
        _ => E::PInf,
    }
}

fn e_le(a: &E, t: &Rat) -> bool {
    *a <= E::F(t.clone())
}

fn integer(e: &Basic) -> Option<i64> {
    let r: &Rat = e.as_rational()?;
    if denom(r) != Int::ONE {
        return None;
    }
    i64::try_from(&numer(r)).ok()
}

fn rational(e: &Basic) -> Option<Rat> {
    e.as_rational().cloned()
}

fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

fn factorial(n: u64) -> Int {
    (1..=n).fold(Int::ONE, |acc, k| acc * Int::from(k))
}

// -- polynomials in the variable

/// The coefficients of `expr` as a polynomial in `x`, lowest degree first
/// (`_polynomial`).
fn polynomial(expr: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let expanded: Basic = expand(expr)?;
    let mut collected: std::collections::BTreeMap<usize, Vec<Basic>> = std::collections::BTreeMap::new();
    for term in add_args(&expanded) {
        let mut degree: usize = 0;
        let mut rest: Vec<Basic> = Vec::new();
        for f in mul_args(&term) {
            if f == *x {
                degree += 1;
            } else if f.is_pow() && f.args()[0] == *x {
                match integer(&f.args()[1]) {
                    Some(n) if n >= 0 => degree += n as usize,
                    _ => return Ok(None),
                }
            } else if f.free_symbols().contains(x) {
                return Ok(None);
            } else {
                rest.push(f);
            }
        }
        collected.entry(degree).or_default().push(mul_all(&rest)?);
    }
    let top: usize = collected.keys().max().copied().unwrap_or(0);
    let mut coefficients: Vec<Basic> = Vec::new();
    for k in 0..=top {
        coefficients.push(add_all(collected.get(&k).map(|v| v.as_slice()).unwrap_or(&[]))?);
    }
    while coefficients.len() > 1 && *coefficients.last().expect("nonempty") == zero() {
        coefficients.pop();
    }
    Ok(Some(coefficients))
}

fn degree(p: &[Basic]) -> i64 {
    if p.len() > 1 || p[0] != zero() {
        p.len() as i64 - 1
    } else {
        -1
    }
}

/// The quotient by `x - c` and the remainder (`_divided`).
fn divided(p: &[Basic], c: &Basic) -> PolyResult<(Vec<Basic>, Basic)> {
    let mut quotient: Vec<Basic> = Vec::new();
    let mut carry: Basic = zero();
    for coefficient in p.iter().rev() {
        carry = a2(&m2(&carry, c)?, coefficient)?;
        quotient.push(carry.clone());
    }
    let remainder: Basic = quotient.pop().expect("nonempty");
    quotient.reverse();
    if quotient.is_empty() {
        quotient.push(zero());
    }
    Ok((quotient, remainder))
}

fn expr_of(p: &[Basic], x: &Basic) -> R {
    let terms: Vec<Basic> = p
        .iter()
        .enumerate()
        .map(|(k, c)| m2(c, &p2(x, &int(k as i64))?))
        .collect::<PolyResult<_>>()?;
    add_all(&terms)
}

/// The coefficients as rationals, highest degree first
/// (`_rational_coefficients`).
fn rational_coefficients(p: &[Basic]) -> Option<Vec<Rat>> {
    p.iter().rev().map(rational).collect()
}

// -- enclosures

/// An enclosure of `expr` on `[lo, hi]` (`_enclose`).
fn enclose(expr: &Basic, x: &Basic, lo: &Basic, hi: &Basic, removable: bool) -> OptIv {
    let mut value: Option<Iv> = evaluate(expr, &[(x.clone(), (lo.clone(), hi.clone()))]);
    if value.is_none() {
        value = extended(expr, x, lo, hi)?;
    }
    if finite(&value) || !removable {
        return Ok(value);
    }
    let fixed: Option<Iv> = removable_value(expr, x, lo, hi)?;
    Ok(fixed.or(value))
}

fn enclose_r(expr: &Basic, x: &Basic, lo: &Rat, hi: &Rat) -> OptIv {
    enclose(expr, x, &rat_basic(lo), &rat_basic(hi), true)
}

fn extended(expr: &Basic, x: &Basic, lo: &Basic, hi: &Basic) -> OptIv {
    if expr.is_add() || expr.is_mul() {
        let mut total: Option<Iv> = None;
        for part in expr.args() {
            let Some(value) = enclose(part, x, lo, hi, false)? else {
                return Ok(None);
            };
            total = Some(match total {
                None => value,
                Some(t) if expr.is_add() => iadd(&t, &value),
                Some(t) => imul(&t, &value),
            });
        }
        return Ok(total);
    }
    if expr.is_pow() {
        return extended_power(&expr.args()[0], &expr.args()[1], x, lo, hi);
    }
    if let Some(f) = function_of(expr).filter(|f| FUNCTIONS.contains(f)) {
        let Some(argument) = enclose(&expr.args()[0], x, lo, hi, false)? else {
            return Ok(None);
        };
        return Ok(apply(f, &argument));
    }
    Ok(None)
}

fn apply(f: Function, v: &Iv) -> Option<Iv> {
    let zero_e: E = E::zero();
    match f {
        Function::Exp => Some(v.exp(p())),
        Function::Log => {
            if v.a > zero_e {
                v.log(p())
            } else if v.a == zero_e && v.b > zero_e {
                Some(Iv::new(E::NInf, Iv::point(v.b.clone()).log(p())?.b))
            } else {
                None
            }
        }
        Function::Sin => Some(v.cos_sin(p()).1),
        Function::Cos => Some(v.cos_sin(p()).0),
        Function::Tan => Some(v.tan(p())),
        Function::Atan => Some(v.atan(p())),
        Function::Sinh => Some(idiv(&isub(&v.exp(p()), &v.neg().exp(p())), &ivi(2))),
        Function::Cosh => Some(idiv(&iadd(&v.exp(p()), &v.neg().exp(p())), &ivi(2))),
        Function::Tanh => {
            let w: Iv = imul(&ivi(2), v).exp(p());
            Some(idiv(&isub(&w, &ivi(1)), &iadd(&w, &ivi(1))))
        }
        Function::Abs => Some(iabs(v)),
        _ => None,
    }
}

fn extended_power(base_: &Basic, exponent_: &Basic, x: &Basic, lo: &Basic, hi: &Basic) -> OptIv {
    let Some(base) = enclose(base_, x, lo, hi, false)? else {
        return Ok(None);
    };
    let zero_e: E = E::zero();
    if let Some(n) = integer(exponent_) {
        if n >= 0 {
            return Ok(Some(ipow(&base, n)));
        }
        if base.a > zero_e || base.b < zero_e {
            return Ok(Some(idiv(&ivi(1), &ipow(&base, -n))));
        }
        if base.a == zero_e && base.b > zero_e {
            let q: Iv = idiv(&ivi(1), &ipow(&Iv::point(base.b.clone()), -n));
            return Ok(Some(Iv::new(q.a, E::PInf)));
        }
        if base.b == zero_e && base.a < zero_e {
            let q: Iv = idiv(&ivi(1), &ipow(&Iv::point(base.a.clone()), -n));
            return Ok(Some(if n % 2 == 0 {
                Iv::new(q.a, E::PInf)
            } else {
                Iv::new(E::NInf, q.b)
            }));
        }
        return Ok(None);
    }
    let Some(exponent) = enclose(exponent_, x, lo, hi, false)? else {
        return Ok(None);
    };
    if base.a > zero_e {
        let Some(l) = base.log(p()) else { return Ok(None) };
        return Ok(Some(imul(&exponent, &l).exp(p())));
    }
    if crate::core::assumptions::is_number(exponent_) && yes(exponent_, Predicate::extended_real) {
        if base.a == zero_e && base.b > zero_e {
            let Some(l) = Iv::point(base.b.clone()).log(p()) else {
                return Ok(None);
            };
            let at_top: Iv = imul(&exponent, &l).exp(p());
            if exponent.a > zero_e {
                return Ok(Some(Iv::new(E::zero(), at_top.b)));
            }
            if exponent.b < zero_e {
                return Ok(Some(Iv::new(at_top.a, E::PInf)));
            }
        }
        if base.a == zero_e && base.b == zero_e && exponent.a > zero_e {
            return Ok(Some(Iv::point(E::zero())));
        }
        return Ok(None);
    }
    // a nonnegative base and a nonnegative exponent: the power increases
    // with the base, and 1 covers 0**0
    if base.a < zero_e || exponent.a < zero_e {
        return Ok(None);
    }
    let mut top: E = if exponent.a == zero_e { E::one() } else { E::zero() };
    if base.b > zero_e {
        let Some(l) = Iv::point(base.b.clone()).log(p()) else {
            return Ok(None);
        };
        for e in [&exponent.a, &exponent.b] {
            let candidate: Iv = imul(&Iv::point(e.clone()), &l).exp(p());
            if candidate.b > top {
                top = candidate.b;
            }
        }
    }
    Ok(Some(Iv::new(E::zero(), top)))
}

/// `g**(n)/n!` when `expr = g/(x - c)**n` with `g` and its first `n - 1`
/// derivatives vanishing at `c` (`_removable_form`).
fn removable_form(expr: &Basic, x: &Basic, c: &Basic) -> PolyResult<Option<Basic>> {
    let key = (expr.clone(), x.clone(), c.clone());
    if let Some(found) = REMOVABLE.with(|m| m.borrow().get(&key).cloned()) {
        return Ok(found);
    }
    let form: Option<Basic> = attempt(|| find_removable_form(expr, x, c))?.flatten();
    REMOVABLE.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() > 2000 {
            m.clear();
        }
        m.insert(key, form.clone());
    });
    Ok(form)
}

fn find_removable_form(expr: &Basic, x: &Basic, c: &Basic) -> PolyResult<Option<Basic>> {
    let (numerator, denominator) = crate::core::expr::as_numer_denom(expr)?;
    if at(&denominator, x, c)? != zero() {
        return Ok(None);
    }
    let Some(mut q) = polynomial(&denominator, x)? else {
        return Ok(None);
    };
    let mut n: u64 = 0;
    while degree(&q) > 0 {
        let (quotient, remainder) = divided(&q, c)?;
        if remainder != zero() {
            break;
        }
        q = quotient;
        n += 1;
    }
    if n == 0 {
        return Ok(None);
    }
    let mut derivative: Basic = cancelled(&d2(&numerator, &expr_of(&q, x)?)?)?;
    for _ in 0..n {
        if at(&derivative, x, c)? != zero() {
            return Ok(None);
        }
        derivative = diff_n(&derivative, x, 1)?;
    }
    Ok(Some(d2(&derivative, &Basic::from_rational(Rat::from(factorial(n))))?))
}

/// `cancel(e)`, or `e` where the typed polynomials have no domain
/// (`_cancelled`).
fn cancelled(e: &Basic) -> R {
    match cancel(e) {
        Err(PolyError::Polys(..)) => Ok(e.clone()),
        other => other,
    }
}

fn removable_value(expr: &Basic, x: &Basic, lo: &Basic, hi: &Basic) -> OptIv {
    if !(expr.is_mul() || expr.is_pow() || expr.is_add()) || out_of_time() {
        return Ok(None);
    }
    for c in [lo, hi] {
        if let Some(form) = removable_form(expr, x, c)? {
            let value: Option<Iv> = enclose(&form, x, lo, hi, false)?;
            if finite(&value) {
                return Ok(value);
            }
        }
    }
    Ok(None)
}

// -- the Gauss–Legendre rule

/// The `n`-point Gauss–Legendre rule on `[0, 1]` (`_Rule`).
#[derive(Clone)]
struct Rule {
    nodes: Vec<(Rat, Rat)>,
    weights: Vec<Iv>,
    order: i64,
    constant: Iv,
}

fn legendre(n: usize, t: &Iv) -> (Iv, Iv) {
    let (mut previous, mut current) = (ivi(1), t.clone());
    for k in 1..n as i64 {
        let next: Iv = idiv(
            &isub(&imul(&imul(&ivi(2 * k + 1), t), &current), &imul(&ivi(k), &previous)),
            &ivi(k + 1),
        );
        previous = current;
        current = next;
    }
    (current, previous)
}

fn legendre_rat(n: usize, t: &Rat) -> (Rat, Rat) {
    let (mut previous, mut current) = (Rat::ONE, t.clone());
    for k in 1..n as i64 {
        let next: Rat = (Rat::from(2 * k + 1) * t * &current - Rat::from(k) * &previous) / Rat::from(k + 1);
        previous = current;
        current = next;
    }
    (current, previous)
}

/// mpmath's `prec_to_dps`.
fn prec_to_dps(prec: usize) -> usize {
    1usize.max(((prec as f64) / 3.3219280948873626).round() as usize - 1)
}

fn gauss_rule(n: usize) -> Option<Rule> {
    let key = (n, prec_to_dps(p()));
    if let Some(r) = RULES.with(|m| m.borrow().get(&key).cloned()) {
        return r;
    }
    let rule: Option<Rule> = make_rule(n);
    RULES.with(|m| m.borrow_mut().insert(key, rule.clone()));
    rule
}

fn from_f64(v: f64) -> Rat {
    let scaled: f64 = (v * (2f64).powi(60)).round();
    Rat::from_parts(Int::from(scaled as i64), dashu_int::UBig::ONE << 60)
}

/// The nodes found by Newton's method, each enclosed in an interval of
/// radius `10**-(dps + 12)` verified by a sign change of `P_n` at a
/// precision raised by twenty digits (`_make_rule`).
fn make_rule(n: usize) -> Option<Rule> {
    let dps: usize = prec_to_dps(p());
    let order: i64 = 2 * n as i64;
    let mut nodes: Vec<(Rat, Rat)> = Vec::new();
    let mut weights: Vec<Iv> = Vec::new();
    {
        let wprec: usize = crate::numeric::dps_to_prec(dps + 20);
        let _guard = PrecisionGuard::set(wprec);
        let delta: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(10u8).pow(dps + 12));
        let n_r: Rat = Rat::from(n as i64);
        for i in 1..=n {
            let guess: f64 = (std::f64::consts::PI * (4 * i - 1) as f64 / (4 * n + 2) as f64).cos();
            let mut root: Rat = from_f64(guess);
            for _ in 0..60 {
                let (pn, pn1) = legendre_rat(n, &root);
                let derivative: Rat = &n_r * (&root * &pn - &pn1) / (&root * &root - Rat::ONE);
                let step: Rat = pn / derivative;
                root = round(&(&root - &step), wprec + 16, Rnd::Down);
                let size: Rat = if rat_is_negative(&step) { -step } else { step };
                if size < Rat::from_parts(Int::ONE, dashu_int::UBig::ONE << (wprec + 8)) {
                    break;
                }
            }
            let lo: Rat = round(&(&root - &delta), wprec, Rnd::Floor);
            let hi: Rat = round(&(&root + &delta), wprec, Rnd::Ceil);
            let left: Iv = legendre(n, &ivr(&lo)).0;
            let right: Iv = legendre(n, &ivr(&hi)).0;
            let zero_e: E = E::zero();
            if !((left.b < zero_e && zero_e < right.a) || (right.b < zero_e && zero_e < left.a)) {
                return None;
            }
            let t: Iv = Iv::new(ivr(&lo).a, ivr(&hi).b);
            let (pn, pn1) = legendre(n, &t);
            let derivative: Iv = idiv(
                &imul(&ivi(n as i64), &isub(&imul(&t, &pn), &pn1)),
                &isub(&ipow(&t, 2), &ivi(1)),
            );
            weights.push(idiv(
                &ivi(2),
                &imul(&isub(&ivi(1), &ipow(&t, 2)), &ipow(&derivative, 2)),
            ));
            let half: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(2u8));
            nodes.push(((Rat::ONE + &lo) * &half, (Rat::ONE + &hi) * &half));
        }
    }
    let mut ordered: Vec<(Rat, Rat)> = nodes.clone();
    ordered.sort();
    if ordered.windows(2).any(|w| w[0].1 >= w[1].0) {
        return None;
    }
    let nf: Int = factorial(n as u64);
    let constant: Rat = Rat::from(nf.pow(4)) / Rat::from(Int::from(2 * n as i64 + 1) * factorial(2 * n as u64).pow(3));
    Some(Rule {
        nodes,
        weights,
        order,
        constant: ivr(&constant),
    })
}

// -- the endpoint model

/// The integrand near an endpoint as a sum of terms `c(v) v**alpha
/// log(v)**j` (`_Model`).
struct Model {
    v: Basic,
    terms: Vec<(Iv, u32, Basic)>,
}

impl Model {
    fn bound(&self, h: &Rat) -> PolyResult<Option<E>> {
        if *h >= Rat::ONE {
            return Ok(None);
        }
        let width: Iv = ivr(h);
        let mut total: Iv = ivi(0);
        for (alpha, j, coefficient) in &self.terms {
            let c: Option<Iv> = enclose(coefficient, &self.v, &zero(), &rat_basic(h), true)?;
            if !finite(&c) {
                return Ok(None);
            }
            let Some(integral) = model_integral(alpha, *j, &width) else {
                return Ok(None);
            };
            total = iadd(&total, &imul(&iabs(&c.expect("finite")), &integral));
        }
        Ok(Some(total.b))
    }
}

/// `Integral(v**alpha*(-log(v))**j, (v, 0, h))` for `h < 1`
/// (`_model_integral`).
fn model_integral(alpha: &Iv, j: u32, h: &Iv) -> Option<Iv> {
    let a1: Iv = iadd(alpha, &ivi(1));
    let lh: Iv = h.log(p())?;
    let power: Iv = imul(&a1, &lh).exp(p());
    let minus_log: Iv = lh.neg();
    let mut value: Iv = idiv(&power, &a1);
    for k in 1..=j as i64 {
        value = iadd(
            &idiv(&imul(&power, &ipow(&minus_log, k)), &a1),
            &idiv(&imul(&ivi(k), &value), &a1),
        );
    }
    Some(value)
}

fn model(f: &Basic, x: &Basic, end: &Rat, side: Side) -> PolyResult<Option<Model>> {
    let v: Basic = Dummy::new("v", &[(Predicate::positive, true)]).0;
    let point: Basic = match side {
        Side::Left => a2(&rat_basic(end), &v)?,
        Side::Right => s2(&rat_basic(end), &v)?,
    };
    let mut g: Basic = at(f, x, &point)?;
    if let Some(expanded) = attempt(|| expand(&g))? {
        g = expanded;
    }
    let log_v: Basic = log(&v)?;
    let mut terms: Vec<(Iv, u32, Basic)> = Vec::new();
    for term in add_args(&g) {
        let mut alpha: Basic = zero();
        let mut j: u32 = 0;
        let mut rest: Vec<Basic> = Vec::new();
        for factor in mul_args(&term) {
            let (base, exponent) = if factor.is_pow() {
                (factor.args()[0].clone(), factor.args()[1].clone())
            } else {
                (factor.clone(), Basic::one())
            };
            let n: Option<i64> = integer(&exponent);
            if base == v && crate::core::assumptions::is_number(&exponent) && yes(&exponent, Predicate::extended_real) {
                alpha = a2(&alpha, &exponent)?;
            } else if base == log_v && n.is_some_and(|n| n > 0) {
                j += n.expect("checked") as u32;
            } else {
                rest.push(factor);
            }
        }
        if !yes(&a2(&alpha, &Basic::one())?, Predicate::positive) {
            return Ok(None);
        }
        let Some(alpha_) = enclose(&alpha, &v, &zero(), &zero(), true)? else {
            return Ok(None);
        };
        terms.push((alpha_, j, mul_all(&rest)?));
    }
    Ok(Some(Model { v, terms }))
}

// -- centred forms

/// An expression as `N(x)/D(x)` with polynomial `N` and `D` whose
/// coefficients may be transcendental, evaluated by the Taylor shift to
/// the midpoint of the panel (`_Form`).
struct Form {
    x: Basic,
    numerator: Vec<Basic>,
    denominator: Vec<Basic>,
}

impl Form {
    fn evaluate(&self, lo: &Rat, hi: &Rat) -> OptIv {
        let mid: Iv = ivr(&((lo + hi) / Rat::from(2)));
        let radius: Iv = idiv(&ivr(&(hi - lo)), &ivi(2));
        let offset: Iv = Iv::new(
            match &radius.b {
                E::F(r) => E::F(-r.clone()),
                _ => E::NInf,
            },
            radius.b.clone(),
        );
        let mut values: Vec<Iv> = Vec::new();
        for coefficients in [&self.numerator, &self.denominator] {
            let mut enclosed: Vec<Iv> = Vec::new();
            for c in coefficients {
                let value: Option<Iv> = enclose_r(c, &self.x, lo, hi)?;
                if !finite(&value) {
                    return Ok(None);
                }
                enclosed.push(value.expect("finite"));
            }
            values.push(sum_of_powers(&shifted(enclosed, &mid), &offset));
        }
        let (n, d) = (&values[0], &values[1]);
        if d.a <= E::zero() && E::zero() <= d.b {
            return Ok(None);
        }
        Ok(Some(idiv(n, d)))
    }
}

fn shifted(mut a: Vec<Iv>, mid: &Iv) -> Vec<Iv> {
    let n: usize = a.len();
    for i in 0..n.saturating_sub(1) {
        for j in (i..=n - 2).rev() {
            a[j] = iadd(&a[j], &imul(mid, &a[j + 1]));
        }
    }
    a
}

fn sum_of_powers(coefficients: &[Iv], offset: &Iv) -> Iv {
    let mut total: Iv = ivi(0);
    for (k, c) in coefficients.iter().enumerate() {
        total = iadd(&total, &imul(c, &ipow(offset, k as i64)));
    }
    total
}

/// `expr` with its transcendental subexpressions in `x` replaced by
/// dummies (`_hidden`).
fn hidden(expr: &Basic, x: &Basic, back: &mut HashMap<Basic, Basic>) -> R {
    if !expr.free_symbols().contains(x) || expr == x {
        return Ok(expr.clone());
    }
    if expr.is_add() {
        let args: Vec<Basic> = expr
            .args()
            .iter()
            .map(|a| hidden(a, x, back))
            .collect::<PolyResult<_>>()?;
        return add_all(&args);
    }
    if expr.is_mul() {
        let args: Vec<Basic> = expr
            .args()
            .iter()
            .map(|a| hidden(a, x, back))
            .collect::<PolyResult<_>>()?;
        return mul_all(&args);
    }
    if expr.is_pow() && integer(&expr.args()[1]).is_some_and(|n| n > 0) {
        return p2(&hidden(&expr.args()[0], x, back)?, &expr.args()[1]);
    }
    let dummy: Basic = Dummy::new("g", &[]).0;
    back.insert(dummy.clone(), expr.clone());
    Ok(dummy)
}

fn form(expr: &Basic, x: &Basic) -> PolyResult<Option<Form>> {
    Ok(attempt(|| make_form(expr, x))?.flatten())
}

fn make_form(expr: &Basic, x: &Basic) -> PolyResult<Option<Form>> {
    let combined: Basic = together(expr)?;
    let (n, d) = crate::core::expr::as_numer_denom(&combined)?;
    let mut parts: Vec<Vec<Basic>> = Vec::new();
    for part in [n, d] {
        let mut back: HashMap<Basic, Basic> = HashMap::new();
        let h: Basic = hidden(&part, x, &mut back)?;
        let Some(q) = polynomial(&h, x)? else {
            return Ok(None);
        };
        if degree(&q) > 80 {
            return Ok(None);
        }
        parts.push(
            q.iter()
                .map(|c| crate::core::function::xreplace_evaluated(c, &back))
                .collect::<PolyResult<_>>()?,
        );
    }
    let denominator: Vec<Basic> = parts.pop().expect("two parts");
    let numerator: Vec<Basic> = parts.pop().expect("two parts");
    Ok(Some(Form {
        x: x.clone(),
        numerator,
        denominator,
    }))
}

// -- panels

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Side {
    Left,
    Right,
}

/// An integrand on `[a, b]` with the derivatives of the bounds, the rule
/// and the endpoint models (`_Problem`).
struct Problem {
    f: Basic,
    x: Basic,
    a: Rat,
    b: Rat,
    fourth: Option<Basic>,
    rule: Option<Rule>,
    high: Option<Basic>,
    forms: HashMap<Basic, Option<Form>>,
    models: RefCell<HashMap<Side, Option<std::rc::Rc<Model>>>>,
}

impl Problem {
    fn new(f: &Basic, x: &Basic, a: &Rat, b: &Rat) -> PolyResult<Problem> {
        let fourth: Option<Basic> = attempt(|| diff_n(f, x, 4))?;
        let rule: Option<Rule> = gauss_rule(GAUSS_NODES);
        let mut high: Option<Basic> = None;
        if rule.is_some() && !out_of_time() {
            if let Some(derivative) = attempt(|| diff_n(f, x, 2 * GAUSS_NODES as u64))? {
                if ops(&derivative)? <= MAX_OPS {
                    high = Some(derivative);
                }
            }
        }
        let mut forms: HashMap<Basic, Option<Form>> = HashMap::new();
        for expr in [Some(f.clone()), fourth.clone(), high.clone()].into_iter().flatten() {
            if !out_of_time() && !forms.contains_key(&expr) {
                let made: Option<Form> = form(&expr, x)?;
                forms.insert(expr, made);
            }
        }
        Ok(Problem {
            f: f.clone(),
            x: x.clone(),
            a: a.clone(),
            b: b.clone(),
            fourth,
            rule,
            high,
            forms,
            models: RefCell::new(HashMap::new()),
        })
    }

    fn bound(&self, expr: &Basic, lo: &Rat, hi: &Rat) -> OptIv {
        if let Some(Some(form)) = self.forms.get(expr) {
            let value: Option<Iv> = form.evaluate(lo, hi)?;
            if finite(&value) {
                return Ok(value);
            }
        }
        enclose_r(expr, &self.x, lo, hi)
    }

    fn model(&self, side: Side) -> PolyResult<Option<std::rc::Rc<Model>>> {
        if out_of_time() {
            return Ok(None);
        }
        if let Some(m) = self.models.borrow().get(&side) {
            return Ok(m.clone());
        }
        let end: &Rat = if side == Side::Left { &self.a } else { &self.b };
        let made: Option<std::rc::Rc<Model>> = model(&self.f, &self.x, end, side)?.map(std::rc::Rc::new);
        self.models.borrow_mut().insert(side, made.clone());
        Ok(made)
    }
}

/// A panel with its enclosure and error bound (`_Panel`).
struct Panel {
    lo: Rat,
    hi: Rat,
    value: Iv,
    error: E,
    priority: E,
}

impl Panel {
    fn new(lo: Rat, hi: Rat, value: Iv, error: E) -> Panel {
        let priority: E = if matches!(error, E::F(_)) {
            e_add(&error, &e_half_width(&value))
        } else {
            error.clone()
        };
        Panel {
            lo,
            hi,
            value,
            error,
            priority,
        }
    }
}

impl PartialEq for Panel {
    fn eq(&self, other: &Panel) -> bool {
        self.priority == other.priority
    }
}
impl Eq for Panel {}
impl PartialOrd for Panel {
    fn partial_cmp(&self, other: &Panel) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Panel {
    // the largest priority first (a max-heap)
    fn cmp(&self, other: &Panel) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

fn simpson_value(problem: &Problem, lo: &Rat, hi: &Rat) -> OptIv {
    let mid: Rat = (lo + hi) / Rat::from(2);
    let mut values: Vec<Iv> = Vec::new();
    for q in [lo, &mid, hi] {
        let v: Option<Iv> = enclose_r(&problem.f, &problem.x, q, q)?;
        if !finite(&v) {
            return Ok(None);
        }
        values.push(v.expect("finite"));
    }
    let sum: Iv = iadd(&iadd(&values[0], &imul(&ivi(4), &values[1])), &values[2]);
    Ok(Some(imul(&idiv(&ivr(&(hi - lo)), &ivi(6)), &sum)))
}

fn gauss_value(problem: &Problem, lo: &Rat, hi: &Rat) -> OptIv {
    let Some(rule) = &problem.rule else {
        return Ok(None);
    };
    let width: Rat = hi - lo;
    let mut total: Iv = ivi(0);
    for ((node_lo, node_hi), weight) in rule.nodes.iter().zip(&rule.weights) {
        let value: Option<Iv> = enclose_r(
            &problem.f,
            &problem.x,
            &(lo + &width * node_lo),
            &(lo + &width * node_hi),
        )?;
        if !finite(&value) {
            return Ok(None);
        }
        total = iadd(&total, &imul(weight, &value.expect("finite")));
    }
    Ok(Some(imul(&idiv(&ivr(&width), &ivi(2)), &total)))
}

#[derive(Clone, Copy)]
enum PanelRule {
    Gauss,
    Simpson,
}

/// The panel with the rule of the smaller bound, the range bound, the
/// endpoint model, or an infinite error (`_panel`).
fn panel(problem: &Problem, lo: &Rat, hi: &Rat) -> PolyResult<Panel> {
    let width: Iv = ivr(&(hi - lo));
    let mut candidates: Vec<(E, PanelRule)> = Vec::new();
    if let (Some(high), Some(rule)) = (&problem.high, &problem.rule) {
        let bound: Option<Iv> = problem.bound(high, lo, hi)?;
        if finite(&bound) {
            let error: Iv = imul(
                &imul(&ipow(&width, rule.order + 1), &rule.constant),
                &iabs(&bound.expect("finite")),
            );
            candidates.push((error.b, PanelRule::Gauss));
        }
    }
    let bound: Option<Iv> = match &problem.fourth {
        None => None,
        Some(fourth) => problem.bound(fourth, lo, hi)?,
    };
    if finite(&bound) {
        let error: Iv = imul(&idiv(&ipow(&width, 5), &ivi(2880)), &iabs(&bound.expect("finite")));
        candidates.push((error.b, PanelRule::Simpson));
    }
    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    for (error, rule) in candidates {
        let value: Option<Iv> = match rule {
            PanelRule::Gauss => gauss_value(problem, lo, hi)?,
            PanelRule::Simpson => simpson_value(problem, lo, hi)?,
        };
        if let Some(value) = value {
            return Ok(Panel::new(lo.clone(), hi.clone(), value, error));
        }
    }
    let values: Option<Iv> = problem.bound(&problem.f, lo, hi)?;
    if finite(&values) {
        return Ok(Panel::new(
            lo.clone(),
            hi.clone(),
            imul(&width, &values.expect("finite")),
            E::zero(),
        ));
    }
    for (side, end, touching) in [(Side::Left, &problem.a, lo), (Side::Right, &problem.b, hi)] {
        if touching == end {
            if let Some(m) = problem.model(side)? {
                if let Some(bounded) = m.bound(&(hi - lo))? {
                    return Ok(Panel::new(lo.clone(), hi.clone(), ivi(0), bounded));
                }
            }
        }
    }
    Ok(Panel::new(lo.clone(), hi.clone(), ivi(0), E::PInf))
}

/// The heap of panels with the running estimate of the radius
/// (`_Subdivision`); the estimate is exact here, so it never drifts.
struct Subdivision {
    heap: BinaryHeap<Panel>,
    estimate: E,
    unbounded: usize,
}

impl Subdivision {
    fn push(&mut self, panel: Panel) {
        if matches!(panel.error, E::F(_)) {
            self.estimate = match (&self.estimate, &panel.priority) {
                (E::F(a), E::F(b)) => E::F(a + b),
                _ => E::PInf,
            };
        } else {
            self.unbounded += 1;
        }
        self.heap.push(panel);
    }

    fn pop(&mut self) -> Panel {
        let panel: Panel = self.heap.pop().expect("a panel");
        if matches!(panel.error, E::F(_)) {
            if let (E::F(a), E::F(b)) = (&self.estimate, &panel.priority) {
                self.estimate = E::F(a - b);
            }
        } else {
            self.unbounded -= 1;
        }
        panel
    }

    fn exact(&mut self) -> (Iv, E) {
        let mut total: Iv = ivi(0);
        let mut spread: E = E::zero();
        for panel in self.heap.iter() {
            total = iadd(&total, &panel.value);
            spread = e_add(&spread, &panel.error);
        }
        self.estimate = e_add(&e_half_width(&total), &spread);
        (total, spread)
    }
}

fn widened(total: &Iv, spread: &E) -> Iv {
    match spread {
        E::F(s) => {
            let point: Iv = Iv::new(E::F(-s.clone()), E::F(s.clone()));
            iadd(total, &point)
        }
        _ => Iv::new(E::NInf, E::PInf),
    }
}

/// An interval containing `Integral(f, (x, a, b))` for rational `a < b`,
/// refined until its width is below `tolerance` or the panel budget or the
/// time limit is spent; `None` when a narrow panel still cannot be bounded
/// (`enclosure`).
pub fn enclosure(f: &Basic, x: &Basic, a: &Rat, b: &Rat, tolerance: &Rat, max_panels: usize) -> OptIv {
    let problem: Problem = Problem::new(f, x, a, b)?;
    let mut panels: Subdivision = Subdivision {
        heap: BinaryHeap::new(),
        estimate: E::zero(),
        unbounded: 0,
    };
    panels.push(panel(&problem, a, b)?);
    let mut iterations: usize = 0;
    let span: Rat = b - a;
    while panels.heap.len() < max_panels {
        if panels.unbounded == 0 && (e_le(&panels.estimate, tolerance) || iterations % 64 == 0) {
            let (total, spread) = panels.exact();
            if e_le(&e_add(&e_half_width(&total), &spread), tolerance) {
                break;
            }
        }
        iterations += 1;
        if out_of_time() {
            break;
        }
        crate::timeout::check()?;
        let worst: Panel = panels.pop();
        if !matches!(worst.error, E::F(_)) && &worst.hi - &worst.lo < tolerance * &span {
            return Ok(None);
        }
        let mid: Rat = (&worst.lo + &worst.hi) / Rat::from(2);
        panels.push(panel(&problem, &worst.lo, &mid)?);
        panels.push(panel(&problem, &mid, &worst.hi)?);
    }
    if panels.unbounded > 0 {
        return Ok(None);
    }
    let (total, spread) = panels.exact();
    Ok(Some(widened(&total, &spread)))
}

// -- substitutions

/// Powers with rational exponents of a factored base, denested
/// (`_denest`).
fn denest(g: &Basic) -> R {
    let replaced: Basic = crate::core::traversal::bottom_up(
        g,
        &|node: &Basic| {
            if node.is_pow() && node.args()[1].as_rational().is_some_and(|r| denom(r) > Int::ONE) {
                let base: Basic = crate::polys::polytools::factor(&expand(&node.args()[0])?, &[], false, true)?;
                return p2(&base, &node.args()[1]);
            }
            Ok(node.clone())
        },
        false,
    )?;
    crate::simplify::powsimp::powdenest(&replaced, true, false)
}

fn lcm(v: &BTreeSet<i64>) -> i64 {
    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    v.iter().fold(1, |acc, x| acc / gcd(acc, *x) * x)
}

/// `f` with `x = a + (b - a) u**k` (or `x = b - (b - a) u**k`) on `u` in
/// `[0, 1]`, removing powers with rational exponents of bases vanishing at
/// that end (`_remove_power_singularity`).
fn remove_power_singularity(f: &Basic, x: &Basic, a: &Rat, b: &Rat, side: Side) -> PolyResult<Option<(Basic, Basic)>> {
    let end: Basic = rat_basic(if side == Side::Left { a } else { b });
    let mut denominators: BTreeSet<i64> = BTreeSet::new();
    for node in f.atoms(Some(&|n: &Basic| n.is_pow())) {
        let Some(r) = node.args()[1].as_rational() else {
            continue;
        };
        let q: Int = denom(r);
        if q > Int::ONE && q <= Int::from(12) && *r > Rat::from(-1) && at(&node.args()[0], x, &end)? == zero() {
            denominators.insert(i64::try_from(&q).expect("small"));
        }
    }
    if denominators.is_empty() {
        return Ok(None);
    }
    let k: i64 = lcm(&denominators);
    let u: Basic = Dummy::new("u", &[(Predicate::positive, true)]).0;
    let scale: Basic = rat_basic(&(b - a));
    let uk: Basic = p2(&u, &int(k))?;
    let image: Basic = if side == Side::Left {
        a2(&rat_basic(a), &m2(&scale, &uk)?)?
    } else {
        s2(&rat_basic(b), &m2(&scale, &uk)?)?
    };
    let g: Basic = m2(&m2(&m2(&at(f, x, &image)?, &scale)?, &int(k))?, &p2(&u, &int(k - 1))?)?;
    let denested: Option<Basic> = attempt(|| denest(&g))?;
    Ok(Some((denested.unwrap_or(g), u)))
}

fn desingularised(f: &Basic, x: &Basic, lo: &Rat, hi: &Rat) -> PolyResult<(Basic, Basic, Rat, Rat)> {
    let (mut g, mut u, mut a, mut b) = (f.clone(), x.clone(), lo.clone(), hi.clone());
    for side in [Side::Left, Side::Right] {
        if let Some((h, v)) = remove_power_singularity(&g, &u, &a, &b, side)? {
            g = h;
            u = v;
            a = Rat::ZERO;
            b = Rat::ONE;
        }
    }
    Ok((g, u, a, b))
}

fn rounded(value: &E, rnd: Rnd) -> Option<Rat> {
    match value {
        E::F(r) => Some(round(r, crate::numeric::dps_to_prec(40), rnd)),
        _ => None,
    }
}

type Bounds = (Rat, Rat, Vec<(Rat, Rat)>);

/// Rational bounds inside `[a, b]` and the strips at irrational endpoints
/// (`_bounds`).
fn bounds(a: &Basic, b: &Basic) -> PolyResult<Option<Bounds>> {
    let mut strips: Vec<(Rat, Rat)> = Vec::new();
    let dummy: Basic = Dummy::new("x", &[]).0;
    let lo: Rat = match rational(a) {
        Some(r) => r,
        None => {
            let Some(value) = enclose(a, &dummy, &zero(), &zero(), true)? else {
                return Ok(None);
            };
            let (Some(top), Some(bottom)) = (rounded(&value.b, Rnd::Ceil), rounded(&value.a, Rnd::Floor)) else {
                return Ok(None);
            };
            strips.push((bottom, top.clone()));
            top
        }
    };
    let hi: Rat = match rational(b) {
        Some(r) => r,
        None => {
            let Some(value) = enclose(b, &dummy, &zero(), &zero(), true)? else {
                return Ok(None);
            };
            let (Some(bottom), Some(top)) = (rounded(&value.a, Rnd::Floor), rounded(&value.b, Rnd::Ceil)) else {
                return Ok(None);
            };
            strips.push((bottom.clone(), top));
            bottom
        }
    };
    Ok(Some((lo, hi, strips)))
}

fn infinite(e: &Basic) -> bool {
    *e == oo() || *e == neg_oo()
}

/// An enclosure of `Integral(f, (x, a, b))` for any bounds (`_enclosed`).
fn enclosed(f: &Basic, x: &Basic, a: &Basic, b: &Basic, tolerance: &Rat) -> OptIv {
    if a == b {
        return Ok(Some(ivi(0)));
    }
    if out_of_time() {
        return Ok(None);
    }
    if infinite(a) || infinite(b) {
        return infinite_range(f, x, a, b, tolerance);
    }
    let Some((lo, hi, strips)) = bounds(a, b)? else {
        return Ok(None);
    };
    if lo > hi {
        return Ok(enclosed(f, x, b, a, tolerance)?.map(|v| v.neg()));
    }
    let (g, u, lo_, hi_) = desingularised(f, x, &lo, &hi)?;
    let mut result: Option<Iv> = enclosure(&g, &u, &lo_, &hi_, tolerance, MAX_PANELS)?;
    if result.is_none() && (g != *f || u != *x) {
        result = enclosure(f, x, &lo, &hi, tolerance, MAX_PANELS)?;
    }
    let Some(mut result) = result else {
        return Ok(None);
    };
    for (strip_lo, strip_hi) in &strips {
        let values: Option<Iv> = enclose_r(f, x, strip_lo, strip_hi)?;
        if !finite(&values) {
            return Ok(None);
        }
        result = iadd(&result, &imul(&ivr(&(strip_hi - strip_lo)), &values.expect("finite")));
    }
    Ok(Some(result))
}

// -- infinite ranges

fn tidy(g: &Basic) -> R {
    Ok(attempt(|| cancel(&expand_power_exp(g)?))?.unwrap_or_else(|| g.clone()))
}

fn half(t: &Rat) -> Rat {
    t / Rat::from(2)
}

/// `x = a + t/(1 - t)` on `[0, 1]` (`_rational_map`).
fn rational_map(f: &Basic, x: &Basic, a: &Basic, tolerance: &Rat) -> OptIv {
    let t: Basic = Dummy::new("t", &[(Predicate::positive, true)]).0;
    let one_minus: Basic = s2(&Basic::one(), &t)?;
    let g: Basic = d2(&at(f, x, &a2(a, &d2(&t, &one_minus)?)?)?, &p2(&one_minus, &int(2))?)?;
    let simpler: Option<Basic> = attempt(|| simplify(&cancel(&g)?))?;
    enclosed(&simpler.unwrap_or(g), &t, &zero(), &Basic::one(), tolerance)
}

/// `x = a - log(1 - t)` on `[0, 1]` (`_exponential_map`).
fn exponential_map(f: &Basic, x: &Basic, a: &Basic, tolerance: &Rat) -> OptIv {
    let t: Basic = Dummy::new("t", &[(Predicate::positive, true)]).0;
    let one_minus: Basic = s2(&Basic::one(), &t)?;
    let g: Basic = tidy(&d2(&at(f, x, &s2(a, &log(&one_minus)?)?)?, &one_minus)?)?;
    enclosed(&g, &t, &zero(), &Basic::one(), tolerance)
}

fn ceiling_of(a: &Basic) -> PolyResult<Option<i64>> {
    if let Some(r) = rational(a) {
        let (n, d) = (numer(&r), denom(&r));
        let q: Int = if rat_is_negative(&r) {
            -((-n) / &d)
        } else {
            (n + &d - Int::ONE) / &d
        };
        return Ok(i64::try_from(&q).ok());
    }
    let c: Basic = crate::functions::apply_function(Function::Ceiling, std::slice::from_ref(a))?;
    Ok(integer(&c))
}

/// A decay `exp(-c x**k)`, `k >= 2` (`_stretched`).
fn stretched(f: &Basic, x: &Basic, a: &Basic, tolerance: &Rat) -> OptIv {
    let mut degrees: BTreeSet<i64> = BTreeSet::new();
    for node in f.atoms(Some(&|n: &Basic| n.is_function(Function::Exp))) {
        let argument: &Basic = &node.args()[0];
        if !argument.free_symbols().contains(x) {
            continue;
        }
        let Some(q) = polynomial(argument, x)? else {
            return Ok(None);
        };
        if degree(&q) < 2 || !yes(q.last().expect("nonempty"), Predicate::negative) {
            return Ok(None);
        }
        degrees.insert(degree(&q));
    }
    if degrees.len() != 1 {
        return Ok(None);
    }
    let k: i64 = *degrees.iter().next().expect("one");
    let Some(start) = ceiling_of(a)? else {
        return Ok(None);
    };
    let cut: Basic = int(start.max(1));
    let Some(finite_part) = enclosed(f, x, a, &cut, &half(tolerance))? else {
        return Ok(None);
    };
    let s: Basic = Dummy::new("s", &[(Predicate::positive, true)]).0;
    let root: Basic = Basic::from_rational(crate::numbers::rat(1, k));
    let g: Basic = d2(
        &m2(&at(f, x, &p2(&s, &root)?)?, &p2(&s, &s2(&root, &Basic::one())?)?)?,
        &int(k),
    )?;
    let tail: Option<Iv> = exponential_map(&g, &s, &p2(&cut, &int(k))?, &half(tolerance))?;
    Ok(tail.map(|t| iadd(&finite_part, &t)))
}

/// An antiderivative of `c sin(w x + d)` or `c cos(w x + d)`
/// (`_trigonometric_antiderivative`).
fn trigonometric_antiderivative(e: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let mut rest: Vec<Basic> = Vec::new();
    let mut found: Option<Basic> = None;
    for factor in mul_args(e) {
        if (factor.is_function(Function::Sin) || factor.is_function(Function::Cos)) && found.is_none() {
            found = Some(factor);
        } else if factor.free_symbols().contains(x) {
            return Ok(None);
        } else {
            rest.push(factor);
        }
    }
    let Some(found) = found else {
        return Ok(None);
    };
    let argument: Basic = found.args()[0].clone();
    let Some(q) = polynomial(&argument, x)? else {
        return Ok(None);
    };
    if degree(&q) != 1 {
        return Ok(None);
    }
    let c: Basic = d2(&mul_all(&rest)?, &q[1])?;
    if found.is_function(Function::Sin) {
        let cos: Basic = crate::functions::apply_function(Function::Cos, std::slice::from_ref(&argument))?;
        return Ok(Some(m2(&ng(&c)?, &cos)?));
    }
    let sin: Basic = crate::functions::apply_function(Function::Sin, std::slice::from_ref(&argument))?;
    Ok(Some(m2(&c, &sin)?))
}

/// `f = g(x) sin(w x + c)` (or `cos`): the tail integrated by parts `k`
/// times, the remainder bounded by Bonnet's mean value theorem
/// (`_oscillatory`).
fn oscillatory(f: &Basic, x: &Basic, a: &Basic, tolerance: &Rat) -> OptIv {
    let mut factors: Vec<Basic> = mul_args(f);
    let trigonometric: Vec<usize> = factors
        .iter()
        .enumerate()
        .filter(|(_, g)| g.is_function(Function::Sin) || g.is_function(Function::Cos))
        .map(|(i, _)| i)
        .collect();
    if trigonometric.len() != 1 {
        return Ok(None);
    }
    let mut phi: Basic = factors.remove(trigonometric[0]);
    let mut g: Basic = mul_all(&factors)?;
    if has_function(&g, &[Function::Sin, Function::Cos, Function::Tan]) {
        return Ok(None);
    }
    let argument: Basic = phi.args()[0].clone();
    let Some(q) = polynomial(&argument, x)? else {
        return Ok(None);
    };
    if degree(&q) != 1 {
        return Ok(None);
    }
    let mut w: Basic = q[1].clone();
    if !crate::core::assumptions::is_number(&w) {
        return Ok(None);
    }
    if yes(&w, Predicate::negative) {
        // sin(-y) = -sin(y), cos(-y) = cos(y), the nodes kept unevaluated
        let negated: Basic = ng(&argument)?;
        if phi.is_function(Function::Sin) {
            phi = Basic::new(Head::Function(Function::Sin), vec![negated]);
            g = ng(&g)?;
        } else {
            phi = Basic::new(Head::Function(Function::Cos), vec![negated]);
        }
        w = ng(&w)?;
    }
    let w_: Option<Iv> = enclose(&w, x, &zero(), &zero(), true)?;
    if !finite(&w_) || w_.as_ref().expect("finite").a <= E::zero() {
        return Ok(None);
    }
    let w_: Iv = w_.expect("finite");
    let mut derivatives: Vec<Basic> = vec![g];
    let mut antiderivatives: Vec<Basic> = vec![phi];
    let Some(start) = ceiling_of(a)? else {
        return Ok(None);
    };
    for m in 0..24u32 {
        let big_a: Basic = int((start + 1) * (1i64 << m));
        for k in 0..=MAX_PARTS {
            if out_of_time() {
                return Ok(None);
            }
            while derivatives.len() <= k + 1 {
                let next: Basic = diff_n(derivatives.last().expect("nonempty"), x, 1)?;
                derivatives.push(next);
            }
            while antiderivatives.len() <= k {
                let Some(next) = trigonometric_antiderivative(antiderivatives.last().expect("nonempty"), x)? else {
                    return Ok(None);
                };
                antiderivatives.push(next);
            }
            let at_a: Option<Iv> = enclose(&derivatives[k], x, &big_a, &big_a, true)?;
            if !finite(&at_a) {
                break;
            }
            let bound: Iv = idiv(&imul(&ivi(2), &iabs(&at_a.expect("finite"))), &ipow(&w_, k as i64 + 1));
            let remainder: E = bound.b;
            if !e_le(&remainder, &half(tolerance)) {
                continue;
            }
            if !monotone_to_zero(&derivatives[..k + 2], x, &big_a)? {
                continue;
            }
            let mut tail: Iv = ivi(0);
            for i in 0..k {
                let sign: Basic = int(if i % 2 == 0 { -1 } else { 1 });
                let term: Basic = m2(&m2(&sign, &derivatives[i])?, &antiderivatives[i + 1])?;
                let value: Option<Iv> = enclose(&term, x, &big_a, &big_a, true)?;
                if !finite(&value) {
                    return Ok(None);
                }
                tail = iadd(&tail, &value.expect("finite"));
            }
            let Some(finite_part) = enclosed(f, x, a, &big_a, &half(tolerance))? else {
                return Ok(None);
            };
            return Ok(Some(iadd(&finite_part, &widened(&tail, &remainder))));
        }
    }
    Ok(None)
}

/// The sign of `expr` on `[A, oo)` (`_sign_on_ray`).
fn sign_on_ray(expr: &Basic, x: &Basic, big_a: &Basic) -> PolyResult<Option<i32>> {
    if *expr == zero() {
        return Ok(Some(0));
    }
    Ok(attempt(|| find_sign_on_ray(expr, x, big_a))?.flatten())
}

/// The number of distinct roots of `p` (highest degree first) in
/// `[A, oo)` (`_roots_on_ray`).
fn roots_on_ray(q: &[Rat], big_a: &Rat) -> i64 {
    let mut at_a: Rat = Rat::ZERO;
    for c in q {
        at_a = at_a * big_a + c;
    }
    crate::polys::rootisolation::dup_count_real_roots(q, Some(big_a), None)
        + if at_a == Rat::ZERO && q.len() > 1 { 1 } else { 0 }
}

fn find_sign_on_ray(expr: &Basic, x: &Basic, big_a: &Basic) -> PolyResult<Option<i32>> {
    let combined: Basic = together(expr)?;
    let (n, d) = crate::core::expr::as_numer_denom(&combined)?;
    let mut sign: i32 = 1;
    for part in [n, d] {
        for factor in mul_args(&part) {
            let (mut base, mut exponent) = (factor.clone(), 1i64);
            if factor.is_pow() {
                if let Some(k) = integer(&factor.args()[1]) {
                    base = factor.args()[0].clone();
                    exponent = k;
                }
            }
            let factor_sign: Option<i32> = if !base.free_symbols().contains(x) {
                enclose(&base, x, &zero(), &zero(), true)?.and_then(|v| interval_sign(&v))
            } else {
                let q: Option<Vec<Basic>> = polynomial(&base, x)?;
                match q.as_deref().and_then(rational_coefficients) {
                    Some(rational) => {
                        if roots_on_ray(&rational, big_a.as_rational().expect("an integer")) != 0 {
                            return Ok(None);
                        }
                        enclose(&base, x, big_a, big_a, true)?.and_then(|v| interval_sign(&v))
                    }
                    None => enclose(&base, x, big_a, &oo(), true)?.and_then(|v| interval_sign(&v)),
                }
            };
            match factor_sign {
                None | Some(0) => return Ok(None),
                Some(s) => {
                    if exponent % 2 != 0 {
                        sign *= s;
                    }
                }
            }
        }
    }
    Ok(Some(sign))
}

fn interval_sign(value: &Iv) -> Option<i32> {
    if value.a > E::zero() {
        return Some(1);
    }
    if value.b < E::zero() {
        return Some(-1);
    }
    None
}

/// Whether the last derivative but one has a constant sign and the last
/// the opposite sign (or vanishes) on `[A, oo)`, all of them vanishing at
/// infinity (`_monotone_to_zero`).
fn monotone_to_zero(derivatives: &[Basic], x: &Basic, big_a: &Basic) -> PolyResult<bool> {
    let n: usize = derivatives.len();
    let last: Option<i32> = sign_on_ray(&derivatives[n - 2], x, big_a)?;
    let next: Option<i32> = sign_on_ray(&derivatives[n - 1], x, big_a)?;
    let (Some(last), Some(next)) = (last, next) else {
        return Ok(false);
    };
    if last * next > 0 {
        return Ok(false);
    }
    for derivative in &derivatives[..n - 1] {
        let at_infinity: Option<Basic> = attempt(|| crate::core::limits::limit(derivative, x, &oo(), "+"))?;
        if at_infinity != Some(zero()) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// An infinite range (`_infinite`).
fn infinite_range(f: &Basic, x: &Basic, a: &Basic, b: &Basic, tolerance: &Rat) -> OptIv {
    if *a == neg_oo() && *b == oo() {
        let left: Option<Iv> = infinite_range(f, x, &neg_oo(), &zero(), &half(tolerance))?;
        let right: Option<Iv> = infinite_range(f, x, &zero(), &oo(), &half(tolerance))?;
        return Ok(match (left, right) {
            (Some(l), Some(r)) => Some(iadd(&l, &r)),
            _ => None,
        });
    }
    if *a == neg_oo() {
        return infinite_range(&at(f, x, &ng(x)?)?, x, &ng(b)?, &oo(), tolerance);
    }
    if *a == oo() {
        return Ok(infinite_range(f, x, b, &oo(), tolerance)?.map(|v| v.neg()));
    }
    if *b != oo() {
        return Ok(None);
    }
    let has_exp: bool = has_function(f, &[Function::Exp]);
    let mut result: Option<Iv> = oscillatory(f, x, a, tolerance)?;
    if result.is_none() && has_exp {
        result = stretched(f, x, a, tolerance)?;
    }
    let maps: [fn(&Basic, &Basic, &Basic, &Rat) -> OptIv; 2] = if has_exp {
        [exponential_map, rational_map]
    } else {
        [rational_map, exponential_map]
    };
    for mapping in maps {
        if result.is_some() || out_of_time() {
            break;
        }
        result = mapping(f, x, a, tolerance)?;
    }
    Ok(result)
}

// -- the interface

/// Restores the deadline when dropped.
struct DeadlineGuard(Option<Instant>);
impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        DEADLINE.with(|d| d.set(self.0));
    }
}

/// The enclosure of `Integral(f, (x, a, b))` refined to a quarter of
/// `10**(-digits)` (`validated_integral` without the conversion to
/// floats): `None` when the integrand has parameters, uses functions the
/// interval evaluator does not know, or cannot be bounded on some panel.
pub fn validated_enclosure(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    digits: usize,
) -> PolyResult<Option<(Rat, Rat)>> {
    let mut free: BTreeSet<Basic> = f.free_symbols();
    free.remove(x);
    if !free.is_empty() || !a.free_symbols().is_empty() || !b.free_symbols().is_empty() {
        return Ok(None);
    }
    if a == b {
        return Ok(Some((Rat::ZERO, Rat::ZERO)));
    }
    let tolerance: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(10u8).pow(digits)) / Rat::from(4);
    let _precision = PrecisionGuard::set(crate::numeric::dps_to_prec(digits + 10));
    let _deadline = DeadlineGuard(DEADLINE.with(|d| d.get()));
    if let Some(t) = crate::settings::settings().timeout {
        if DEADLINE.with(|d| d.get()).is_none() {
            let scaled: f64 = (t * crate::settings::settings().time_scale).max(0.0);
            DEADLINE.with(|d| d.set(Some(Instant::now() + Duration::from_secs_f64(scaled))));
        }
    }
    let Some(result) = enclosed(f, x, a, b, &tolerance)? else {
        return Ok(None);
    };
    match (&result.a, &result.b) {
        (E::F(lo), E::F(hi)) => Ok(Some((lo.clone(), hi.clone()))),
        _ => Ok(None),
    }
}

/// `(value, error)` with `|Integral(f, (x, a, b)) - value| <= error`
/// proved by interval arithmetic, the error aimed at `10**(-digits)`
/// (`validated_integral`): the exact midpoint of the enclosure and its
/// radius rounded up to 20 bits.
pub fn validated_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, digits: usize) -> PolyResult<Option<(Rat, Rat)>> {
    let Some((lo, hi)) = validated_enclosure(f, x, a, b, digits)? else {
        return Ok(None);
    };
    let mid: Rat = (&lo + &hi) / Rat::from(2);
    let radius: Rat = round(&((&hi - &lo) / Rat::from(2)), 20, Rnd::Ceil);
    Ok(Some((mid, radius)))
}
