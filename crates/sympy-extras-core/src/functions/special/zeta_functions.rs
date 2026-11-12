//! Port of `sympy_extras/functions/special/zeta_functions.py`: the
//! Riemann and Hurwitz zeta [`Function::Zeta`], the Dirichlet eta
//! [`Function::DirichletEta`], the polylogarithm [`Function::PolyLog`],
//! the Lerch transcendent [`Function::LerchPhi`] (a node with its
//! derivatives), Riemann's xi [`Function::RiemannXi`] and the Stieltjes
//! constants [`Function::Stieltjes`]: evaluation, derivatives, facts and
//! values (zeta and eta by Euler-Maclaurin, xi through them, polylog by its
//! series inside `|z| < 3/4`; mpmath's other polylog formulas, lerchphi and
//! stieltjes have no value here). Not ported, as in Python: the polar
//! arguments of polylog, the expansions.

use crate::core::add::{add, sub};
use crate::core::assumptions::{facts_of, is_number, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, complex_real, parts, real_int, Complex, Real};
use crate::core::expr::{equals, is, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::{not_ported, PolyResult};
use crate::functions::elementary::exponential::log;
use crate::numbers::{denom, numer, Int, Rat};

use super::gamma_numeric::{bernoulli, gamma as gamma_value, hurwitz_zeta, polylog_series};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

fn number(r: &Rat) -> PolyResult<Basic> {
    if denom(r) == Int::ONE {
        return Ok(Basic::integer(numer(r)));
    }
    Basic::rational(numer(r), denom(r))
}

fn div(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}

fn as_int(e: &Basic) -> Option<i64> {
    let v = e.as_rational()?;
    if denom(v) == Int::ONE {
        i64::try_from(&numer(v)).ok()
    } else {
        None
    }
}

fn binomial_int(n: i64, k: i64) -> Int {
    let mut r: Int = Int::ONE;
    for i in 0..k {
        r = r * Int::from(n - i) / Int::from(i + 1);
    }
    r
}

/// The Bernoulli number with SymPy's convention `B_1 = 1/2`
/// (`bernoulli_number`).
pub fn bernoulli_number(n: usize) -> Rat {
    let b: Rat = bernoulli(n);
    if n == 1 {
        -b
    } else {
        b
    }
}

/// `B_n(x) = sum(binomial(n, k)*b_k*x**(n - k))` with `b_1 = -1/2`
/// (`bernoulli_polynomial`).
pub fn bernoulli_polynomial(n: i64, x: &Basic) -> PolyResult<Basic> {
    let mut terms: Vec<Basic> = Vec::new();
    for k in 0..=n {
        let b: Rat = bernoulli(k as usize);
        if b != Rat::ZERO {
            let c: Rat = Rat::from(binomial_int(n, k)) * b;
            terms.push(mul(&[number(&c)?, pow(x, &int(n - k))?])?);
        }
    }
    add(&terms)
}

/// `sum(1/k**m, k = 1..n)`.
fn harmonic_number(n: i64, m: i64) -> Rat {
    let mut total: Rat = Rat::ZERO;
    for k in 1..=n {
        let mut p: Int = Int::ONE;
        for _ in 0..m {
            p *= Int::from(k);
        }
        total += Rat::ONE / Rat::from(p);
    }
    total
}

fn has_zeta(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(Function::Zeta)) || e.args().iter().any(has_zeta)
}

fn huge() -> crate::error::PolyError {
    not_ported("a zeta of a huge integer")
}

/// `Zeta.eval`.
fn zeta_eval(args: &[Basic]) -> PolyResult<Basic> {
    let s: &Basic = &args[0];
    let given: Option<&Basic> = args.get(1);
    if given == Some(&Basic::one()) {
        return zeta(s, None);
    }
    if matches!(s.head(), Head::NaN) || given.is_some_and(|g| matches!(g.head(), Head::NaN)) {
        return Ok(Basic::nan());
    }
    if *s == Basic::one() {
        return Ok(Basic::zoo());
    }
    if matches!(s.head(), Head::Infinity) {
        return Ok(Basic::one());
    }
    if given.is_some_and(|g| matches!(g.head(), Head::Infinity)) {
        return Ok(Basic::zero());
    }
    let a: Basic = given.cloned().unwrap_or_else(Basic::one);
    let n: Option<i64> = as_int(s);
    if let Some(n) = n.filter(|n| *n <= 0) {
        if n < -2000 {
            return Err(huge());
        }
        return div(&bernoulli_polynomial(1 - n, &a)?, &sub(s, &Basic::one())?);
    }
    let unevaluated = || node(Function::Zeta, args.to_vec());
    if a == Basic::one() {
        if let Some(n) = n.filter(|n| n % 2 == 0) {
            if n > 2000 {
                return Err(huge());
            }
            let mut f: Int = Int::ONE;
            for i in 2..=n {
                f *= Int::from(i);
            }
            let coefficient: Rat = -bernoulli_number(n as usize) / Rat::from(Int::from(2) * f);
            let base: Basic = mul(&[int(2), Basic::pi(), Basic::imaginary_unit()])?;
            return mul(&[number(&coefficient)?, pow(&base, s)?]);
        }
        return Ok(unevaluated());
    }
    let k: Option<i64> = as_int(&a);
    if let (Some(n), Some(k)) = (n, k) {
        if k > 0 {
            if k > 100_000 || n > 2000 {
                return Err(huge());
            }
            return sub(&zeta(s, None)?, &number(&harmonic_number(k - 1, n))?);
        }
    }
    if k.is_some_and(|k| k <= 0)
        && (crate::core::expr::is_false(s, Predicate::integer)
            || crate::core::expr::is_false(s, Predicate::nonpositive))
    {
        return Ok(Basic::nan());
    }
    Ok(unevaluated())
}

/// `DirichletEta.eval`.
fn eta_eval(args: &[Basic]) -> PolyResult<Basic> {
    let s: &Basic = &args[0];
    let a: Option<&Basic> = args.get(1);
    if a == Some(&Basic::one()) {
        return dirichlet_eta(s, None);
    }
    let one: Basic = Basic::one();
    let two_power: Basic = pow(&int(2), &sub(&one, s)?)?;
    match a {
        None => {
            if *s == one {
                return log(&int(2));
            }
            let z: Basic = zeta(s, None)?;
            if !has_zeta(&z) {
                return mul(&[sub(&one, &two_power)?, z]);
            }
            Ok(node(Function::DirichletEta, args.to_vec()))
        }
        Some(a) => {
            let half: Basic = div(&add(&[a.clone(), one.clone()])?, &int(2))?;
            if *s == one {
                use super::gamma_functions::digamma;
                return add(&[sub(&log(&int(2))?, &digamma(a)?)?, digamma(&half)?]);
            }
            let (z1, z2) = (zeta(s, Some(a))?, zeta(s, Some(&half))?);
            if !has_zeta(&z1) && !has_zeta(&z2) {
                return sub(&z1, &mul(&[two_power, z2])?);
            }
            Ok(node(Function::DirichletEta, args.to_vec()))
        }
    }
}

/// `polylog(2, z)` at special points (SymPy's `_dilogtable`).
fn dilogarithm_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let one: Basic = Basic::one();
    let i: Basic = Basic::imaginary_unit();
    let pi: Basic = Basic::pi();
    let catalan: Basic = Basic::catalan();
    let five: Basic = pow(&int(5), &Basic::half())?;
    let half: Basic = Basic::half();
    let log2: Basic = log(&int(2))?;
    let g: Basic = div(&sub(&five, &one)?, &int(2))?;
    let big: Basic = div(&add(&[five.clone(), one.clone()])?, &int(2))?;
    let pi2: Basic = pow(&pi, &int(2))?;
    let sq = |e: &Basic| pow(e, &int(2));
    let neg = |e: &Basic| mul(&[Basic::negative_one(), e.clone()]);
    Ok(vec![
        (half, sub(&div(&pi2, &int(12))?, &div(&sq(&log2)?, &int(2))?)?),
        (
            int(2),
            sub(&div(&pi2, &int(4))?, &mul(&[i.clone(), pi.clone(), log2.clone()])?)?,
        ),
        (neg(&g)?, add(&[div(&pi2, &int(-15))?, div(&sq(&log(&g)?)?, &int(2))?])?),
        (neg(&big)?, sub(&div(&pi2, &int(-10))?, &sq(&log(&big)?)?)?),
        (
            div(&sub(&int(3), &five)?, &int(2))?,
            sub(&div(&pi2, &int(15))?, &sq(&log(&g)?)?)?,
        ),
        (g.clone(), sub(&div(&pi2, &int(10))?, &sq(&log(&g)?)?)?),
        (
            i.clone(),
            sub(&mul(&[i.clone(), catalan.clone()])?, &div(&pi2, &int(48))?)?,
        ),
        (
            neg(&i)?,
            sub(
                &mul(&[Basic::negative_one(), i.clone(), catalan.clone()])?,
                &div(&pi2, &int(48))?,
            )?,
        ),
        (
            sub(&one, &i)?,
            add(&[
                div(&pi2, &int(16))?,
                mul(&[Basic::negative_one(), i.clone(), catalan.clone()])?,
                mul(&[
                    Basic::negative_one(),
                    div(&mul(&[pi.clone(), i.clone()])?, &int(4))?,
                    log2.clone(),
                ])?,
            ])?,
        ),
        (
            add(&[one.clone(), i.clone()])?,
            add(&[
                div(&pi2, &int(16))?,
                mul(&[i.clone(), catalan.clone()])?,
                mul(&[div(&mul(&[pi.clone(), i.clone()])?, &int(4))?, log2.clone()])?,
            ])?,
        ),
        (
            div(&sub(&one, &i)?, &int(2))?,
            add(&[
                div(&sq(&log2)?, &int(-8))?,
                div(&mul(&[pi.clone(), i.clone(), log2.clone()])?, &int(8))?,
                div(&mul(&[int(5), pi2.clone()])?, &int(96))?,
                mul(&[Basic::negative_one(), i.clone(), catalan.clone()])?,
            ])?,
        ),
    ])
}

/// `PolyLog.eval`.
fn polylog_eval(s: &Basic, z: &Basic) -> PolyResult<Basic> {
    let one: Basic = Basic::one();
    if is_number(z) {
        if *z == one {
            return zeta(s, None);
        }
        if *z == Basic::negative_one() {
            return mul(&[Basic::negative_one(), dirichlet_eta(s, None)?]);
        }
        if is_true(z, Predicate::zero) {
            return Ok(Basic::zero());
        }
        if *s == int(2) {
            for (key, value) in dilogarithm_table()? {
                if key == *z {
                    return Ok(value);
                }
            }
        }
    }
    if is_true(z, Predicate::zero) {
        return Ok(Basic::zero());
    }
    let is_one: bool = equals(z, &one)?;
    if is_one {
        return zeta(s, None);
    }
    if is_true(s, Predicate::zero) {
        return div(z, &sub(&one, z)?);
    }
    if *s == Basic::negative_one() {
        return div(z, &pow(&sub(&one, z)?, &int(2))?);
    }
    Ok(node(Function::PolyLog, vec![s.clone(), z.clone()]))
}

/// `RiemannXi.eval`.
fn xi_eval(s: &Basic) -> PolyResult<Basic> {
    use super::gamma_functions::gamma;
    let z: Basic = zeta(s, None)?;
    if *s == Basic::zero() || *s == Basic::one() {
        return Ok(Basic::half());
    }
    if !matches!(z.head(), Head::Function(Function::Zeta)) {
        let half_s: Basic = div(s, &int(2))?;
        let top: Basic = mul(&[mul(&[s.clone(), sub(s, &Basic::one())?])?, gamma(&half_s)?, z])?;
        return div(&top, &mul(&[int(2), pow(&Basic::pi(), &half_s)?])?);
    }
    Ok(node(Function::RiemannXi, vec![s.clone()]))
}

/// `Stieltjes.eval`.
fn stieltjes_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    let a: Option<&Basic> = args.get(1);
    if let Some(a) = a {
        if matches!(a.head(), Head::NaN) {
            return Ok(Basic::nan());
        }
        if as_int(a).is_some_and(|k| k <= 0) {
            return Ok(Basic::zoo());
        }
    }
    let default_a: bool = a.map_or(true, |a| *a == Basic::one());
    if n.is_number_node() {
        if matches!(n.head(), Head::NaN) {
            return Ok(Basic::nan());
        }
        if is_true(n, Predicate::negative) {
            return Ok(Basic::zoo());
        }
        if as_int(n).is_none() {
            return Ok(Basic::zoo());
        }
        if is_true(n, Predicate::zero) && default_a {
            return Ok(Basic::euler_gamma());
        }
    }
    if is_true(n, Predicate::extended_negative) {
        return Ok(Basic::zoo());
    }
    if is_true(n, Predicate::zero) && default_a {
        return Ok(Basic::euler_gamma());
    }
    if crate::core::expr::is_false(n, Predicate::integer) {
        return Ok(Basic::zoo());
    }
    Ok(node(Function::Stieltjes, args.to_vec()))
}

/// The heads of the module.
pub fn is_zeta_function(f: Function) -> bool {
    matches!(
        f,
        Function::Zeta
            | Function::DirichletEta
            | Function::PolyLog
            | Function::LerchPhi
            | Function::RiemannXi
            | Function::Stieltjes
    )
}

/// The evaluating function of the heads of the module.
pub fn zeta_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::Zeta => zeta_eval(args),
        Function::DirichletEta => eta_eval(args),
        Function::PolyLog => polylog_eval(&args[0], &args[1]),
        Function::RiemannXi => xi_eval(&args[0]),
        Function::Stieltjes => stieltjes_eval(args),
        _ => Ok(node(f, args.to_vec())),
    }
}

/// `zeta(s)` or `zeta(s, a)`, evaluated.
pub fn zeta(s: &Basic, a: Option<&Basic>) -> PolyResult<Basic> {
    match a {
        None => zeta_function(Function::Zeta, std::slice::from_ref(s)),
        Some(a) => zeta_function(Function::Zeta, &[s.clone(), a.clone()]),
    }
}

/// `dirichlet_eta(s)` or `dirichlet_eta(s, a)`, evaluated.
pub fn dirichlet_eta(s: &Basic, a: Option<&Basic>) -> PolyResult<Basic> {
    match a {
        None => zeta_function(Function::DirichletEta, std::slice::from_ref(s)),
        Some(a) => zeta_function(Function::DirichletEta, &[s.clone(), a.clone()]),
    }
}

pub fn polylog(s: &Basic, z: &Basic) -> PolyResult<Basic> {
    zeta_function(Function::PolyLog, &[s.clone(), z.clone()])
}

pub fn lerchphi(z: &Basic, s: &Basic, a: &Basic) -> PolyResult<Basic> {
    zeta_function(Function::LerchPhi, &[z.clone(), s.clone(), a.clone()])
}

pub fn riemann_xi(s: &Basic) -> PolyResult<Basic> {
    zeta_function(Function::RiemannXi, std::slice::from_ref(s))
}

pub fn stieltjes(n: &Basic, a: Option<&Basic>) -> PolyResult<Basic> {
    match a {
        None => zeta_function(Function::Stieltjes, std::slice::from_ref(n)),
        Some(a) => zeta_function(Function::Stieltjes, &[n.clone(), a.clone()]),
    }
}

/// `DefinedFunction.fdiff`: `Derivative(f(x), x)` when the argument is a
/// symbol found in no other argument.
fn generic_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let a: &Basic = &args[argindex - 1];
    let elsewhere: bool = args
        .iter()
        .enumerate()
        .any(|(i, v)| i != argindex - 1 && v.free_symbols().contains(a));
    if a.is_symbol() && !elsewhere {
        return Ok(crate::core::function::derivative_node(e, &[(a.clone(), 1)]));
    }
    Err(not_ported("a derivative as a Subs"))
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn zeta_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let one: Basic = Basic::one();
    match (f, argindex) {
        (Function::Zeta, 2) => {
            let (s, a) = (&args[0], &args[1]);
            mul(&[
                Basic::negative_one(),
                s.clone(),
                zeta(&add(&[s.clone(), one])?, Some(a))?,
            ])
        }
        (Function::PolyLog, 2) => {
            let (s, z) = (&args[0], &args[1]);
            div(&polylog(&sub(s, &one)?, z)?, z)
        }
        (Function::LerchPhi, 3) => {
            let (z, s, a) = (&args[0], &args[1], &args[2]);
            mul(&[
                Basic::negative_one(),
                s.clone(),
                lerchphi(z, &add(&[s.clone(), one])?, a)?,
            ])
        }
        (Function::LerchPhi, 1) => {
            let (z, s, a) = (&args[0], &args[1], &args[2]);
            let top: Basic = sub(
                &lerchphi(z, &sub(s, &one)?, a)?,
                &mul(&[a.clone(), lerchphi(z, s, a)?])?,
            )?;
            div(&top, z)
        }
        _ => generic_fdiff(e, argindex),
    }
}

/// The facts (`_eval_facts`).
pub fn zeta_facts(f: Function, args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    match f {
        Function::Zeta => {
            if let Some(pole) = sub(&args[0], &Basic::one()).ok().and_then(|d| is(&d, Predicate::zero)) {
                facts.set(Predicate::finite, !pole);
            }
        }
        Function::PolyLog if facts_of(&args[1]).get(&Predicate::zero) == Some(&true) => {
            facts.set(Predicate::zero, true);
        }
        _ => {}
    }
    facts
}

fn c_int(n: i64, wp: usize) -> Complex {
    complex_real(real_int(&Int::from(n), wp), wp)
}

fn is_value(z: &Complex, n: i64) -> bool {
    let (re, im) = parts(z);
    im == Real::ZERO && re == real_int(&Int::from(n), 64)
}

/// The value (`numeric`).
pub fn zeta_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    let wp: usize = prec + 40;
    match f {
        Function::Zeta => {
            let a: Complex = values.get(1).cloned().unwrap_or_else(|| c_int(1, wp));
            hurwitz_zeta(&values[0], &a, prec)
        }
        Function::DirichletEta => {
            let s: &Complex = &values[0];
            let one: Complex = c_int(1, wp);
            let two_power: Complex = (&(&one - s) * &c_int(2, wp).ln()).exp();
            match values.get(1) {
                None => {
                    if is_value(s, 1) {
                        // altzeta(1) = log(2)
                        let l: Complex = c_int(2, wp).ln();
                        let (re, _) = parts(&l);
                        return Some(complex(re.with_precision(prec).value(), real_int(&Int::ZERO, prec)));
                    }
                    let z: Complex = hurwitz_zeta(s, &one, prec + 20)?;
                    let value: Complex = &(&one - &two_power) * &z;
                    let (re, im) = parts(&value);
                    Some(complex(
                        re.with_precision(prec).value(),
                        im.with_precision(prec).value(),
                    ))
                }
                Some(a) => {
                    let half: Complex = &(a + &one) / &c_int(2, wp);
                    let value: Complex =
                        &hurwitz_zeta(s, a, prec + 20)? - &(&two_power * &hurwitz_zeta(s, &half, prec + 20)?);
                    let (re, im) = parts(&value);
                    Some(complex(
                        re.with_precision(prec).value(),
                        im.with_precision(prec).value(),
                    ))
                }
            }
        }
        Function::PolyLog => polylog_series(&values[0], &values[1], prec),
        Function::RiemannXi => {
            let s: &Complex = &values[0];
            if is_value(s, 0) || is_value(s, 1) {
                return Some(complex(
                    crate::core::evalf::real_rat(&crate::numbers::rat(1, 2), prec),
                    real_int(&Int::ZERO, prec),
                ));
            }
            let one: Complex = c_int(1, wp);
            let half_s: Complex = s / &c_int(2, wp);
            let pi: Complex = complex_real(Real::pi(wp), wp);
            let top: Complex =
                &(&(s * &(s - &one)) * &gamma_value(&half_s, prec + 20)?) * &hurwitz_zeta(s, &one, prec + 20)?;
            let bottom: Complex = &c_int(2, wp) * &(&half_s * &pi.ln()).exp();
            let value: Complex = &top / &bottom;
            let (re, im) = parts(&value);
            Some(complex(
                re.with_precision(prec).value(),
                im.with_precision(prec).value(),
            ))
        }
        _ => None,
    }
}
