//! Port of the gamma functions of
//! `sympy_extras/functions/special/gamma_functions.py`: [`Function::Gamma`],
//! [`Function::LogGamma`], [`Function::PolyGamma`], [`Function::DiGamma`],
//! [`Function::TriGamma`]: the evaluation (gamma at the integers and the
//! half-integers, loggamma at the integers and the positive half-integers,
//! polygamma(0, z) at the integers by the harmonic numbers and at the
//! rationals of denominator up to 6 by Gauss' digamma theorem, the poles,
//! the infinities, polygamma(-1, z)), the derivatives, the conjugates, the
//! facts and the values (by [`super::gamma_numeric`]). `polygamma(n, z)`
//! at the integers and at 1/2 for `n >= 1` needs `zeta` (not ported yet);
//! lowergamma, uppergamma and multigamma are not ported.

use crate::core::add::{add, sub};
use crate::core::assumptions::{facts_of, is_number, FactList, Facts, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{parts, Complex, Real};
use crate::core::expr::{as_numer_denom, extract_multiplicatively, is, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::{not_ported, PolyResult};
use crate::functions::combinatorial::factorials::factorial;
use crate::functions::elementary::exponential::log;
use crate::numbers::{denom, numer, Int, Rat};

use super::gamma_numeric;

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

/// A rational number as SymPy's `number_from_fraction` makes it (an
/// `Integer` for an integral value).
fn number(p: Int, q: Int) -> PolyResult<Basic> {
    let r: Rat = Rat::from(p) / Rat::from(q);
    if denom(&r) == Int::ONE {
        return Ok(Basic::integer(numer(&r)));
    }
    Basic::rational(numer(&r), denom(&r))
}

fn sqrt_pi() -> PolyResult<Basic> {
    pow(&Basic::pi(), &Basic::half())
}

fn integer_value(e: &Basic) -> Option<Int> {
    let v = e.as_rational()?;
    if denom(v) == Int::ONE {
        Some(numer(v))
    } else {
        None
    }
}

/// `Gamma.eval`.
fn gamma_eval(z: &Basic) -> PolyResult<Basic> {
    match z.head() {
        Head::NaN => return Ok(Basic::nan()),
        Head::Infinity => return Ok(Basic::oo()),
        _ => {}
    }
    if let Some(v) = integer_value(z) {
        return if v > Int::ZERO {
            factorial(&sub(z, &Basic::one())?)
        } else {
            Ok(Basic::zoo())
        };
    }
    if let Some(r) = z.as_rational() {
        if denom(r) == Int::from(2) {
            let p: Int = numer(r);
            let positive: bool = p > Int::ZERO;
            let mut n: Int = if positive { p.clone() } else { -p.clone() };
            n = &n / Int::from(2);
            let (k, mut coefficient): (Int, Int) = if positive {
                (n.clone(), Int::ONE)
            } else {
                n += Int::ONE;
                let even: bool = &n % Int::from(2) == Int::ZERO;
                (n.clone(), if even { Int::ONE } else { -Int::ONE })
            };
            let kk: i64 = i64::try_from(&k).map_err(|_| not_ported("gamma of a huge half-integer"))?;
            let mut odd: i64 = 3;
            while odd < 2 * kk {
                coefficient *= Int::from(odd);
                odd += 2;
            }
            let two_n: Int = Int::from(2).pow(usize::try_from(&n).map_err(|_| not_ported("a huge power"))?);
            let c: Basic = if positive {
                number(coefficient, two_n)?
            } else {
                number(two_n, coefficient)?
            };
            return mul(&[c, sqrt_pi()?]);
        }
    }
    Ok(node(Function::Gamma, vec![z.clone()]))
}

/// `LogGamma.eval`.
fn loggamma_eval(z: &Basic) -> PolyResult<Basic> {
    if is_true(z, Predicate::integer) {
        if is_true(z, Predicate::nonpositive) {
            return Ok(Basic::oo());
        }
        if is_true(z, Predicate::positive) {
            return log(&gamma(z)?);
        }
    } else if is_true(z, Predicate::rational) {
        let (p, q) = as_numer_denom(z)?;
        if is_true(&p, Predicate::positive) && q == int(2) {
            let inner: Basic = mul(&[
                sqrt_pi()?,
                pow(&int(2), &sub(&Basic::one(), &p)?)?,
                gamma(&p)?,
                pow(
                    &gamma(&mul(&[add(&[p.clone(), Basic::one()])?, Basic::half()])?)?,
                    &Basic::negative_one(),
                )?,
            ])?;
            return log(&inner);
        }
    }
    match z.head() {
        Head::Infinity => return Ok(Basic::oo()),
        Head::NegativeInfinity | Head::ComplexInfinity => return Ok(Basic::zoo()),
        _ => {}
    }
    if is_number(z) && is_true(z, Predicate::infinite) {
        return Ok(Basic::zoo());
    }
    if matches!(z.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    Ok(node(Function::LogGamma, vec![z.clone()]))
}

/// The harmonic number `H_n`.
fn harmonic(n: i64) -> Rat {
    let mut total: Rat = Rat::ZERO;
    for k in 1..=n {
        total += Rat::ONE / Rat::from(Int::from(k));
    }
    total
}

fn rat_basic(p: i64, q: i64) -> PolyResult<Basic> {
    number(Int::from(p), Int::from(q))
}

/// `polygamma(0, p/q)` by Gauss' digamma theorem (`_digamma_rational`).
fn digamma_rational(z: &Basic, p: i64, q: i64) -> PolyResult<Basic> {
    use crate::functions::elementary::integers::floor;
    use crate::functions::elementary::trigonometric::{cos, cot, sin};
    let pi: Basic = Basic::pi();
    let mut sum_terms: Vec<Basic> = Vec::new();
    for k in 1..q {
        let c: Basic = cos(&mul(&[rat_basic(2 * k * p, q)?, pi.clone()])?)?;
        let l: Basic = log(&mul(&[int(2), sin(&mul(&[rat_basic(k, q)?, pi.clone()])?)?])?)?;
        sum_terms.push(mul(&[c, l])?);
    }
    let part: Basic = add(&[
        mul(&[Basic::negative_one(), Basic::euler_gamma()])?,
        mul(&[
            Basic::negative_one(),
            pi.clone(),
            cot(&mul(&[rat_basic(p, q)?, pi.clone()])?)?,
            Basic::half(),
        ])?,
        mul(&[Basic::negative_one(), log(&int(q))?])?,
        add(&sum_terms)?,
    ])?;
    if p > 0 {
        let count: Basic = floor(z)?;
        let z0: Basic = sub(z, &count)?;
        let n: i64 = integer_value(&count).and_then(|v| i64::try_from(&v).ok()).unwrap_or(0);
        let mut terms: Vec<Basic> = Vec::new();
        for k in 0..n {
            terms.push(pow(&add(&[z0.clone(), int(k)])?, &Basic::negative_one())?);
        }
        return add(&[part, add(&terms)?]);
    }
    let count: Basic = floor(&sub(&Basic::one(), z)?)?;
    let z0: Basic = add(&[z.clone(), count.clone()])?;
    let n: i64 = integer_value(&count).and_then(|v| i64::try_from(&v).ok()).unwrap_or(0);
    let mut terms: Vec<Basic> = Vec::new();
    for k in 0..n {
        terms.push(pow(&sub(&sub(&z0, &Basic::one())?, &int(k))?, &Basic::negative_one())?);
    }
    sub(&part, &add(&terms)?)
}

/// `PolyGamma.eval`.
fn polygamma_eval(n: &Basic, z: &Basic) -> PolyResult<Basic> {
    if matches!(n.head(), Head::NaN) || matches!(z.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if matches!(z.head(), Head::Infinity) {
        return Ok(if is_true(n, Predicate::zero) {
            Basic::oo()
        } else {
            Basic::zero()
        });
    }
    if integer_value(z).is_some_and(|v| v <= Int::ZERO) {
        return Ok(Basic::zoo());
    }
    if *n == Basic::negative_one() {
        let half_log: Basic = mul(&[log(&mul(&[int(2), Basic::pi()])?)?, Basic::half()])?;
        return sub(&loggamma(z)?, &half_log);
    }
    if is_true(n, Predicate::zero) {
        if matches!(z.head(), Head::NegativeInfinity) {
            return Ok(Basic::oo());
        }
        if let Some(imaginary) = extract_multiplicatively(z, &Basic::imaginary_unit())? {
            if matches!(imaginary.head(), Head::Infinity | Head::NegativeInfinity) {
                return Ok(Basic::oo());
            }
        }
        if let Some(r) = z.as_rational() {
            let (p, q) = (numer(r), denom(r));
            if q == Int::ONE {
                let p: i64 = i64::try_from(&p).map_err(|_| not_ported("a huge harmonic number"))?;
                if p > 100_000 {
                    return Err(not_ported("a huge harmonic number"));
                }
                let h: Rat = harmonic(p - 1);
                return sub(&number(numer(&h), denom(&h))?, &Basic::euler_gamma());
            }
            if q <= Int::from(6) {
                let (p, q) = (
                    i64::try_from(&p).map_err(|_| not_ported("a huge rational"))?,
                    i64::try_from(&q).unwrap_or(6),
                );
                return digamma_rational(z, p, q);
            }
        }
        return Ok(node(Function::PolyGamma, vec![n.clone(), z.clone()]));
    }
    if is_true(n, Predicate::integer) && is_true(n, Predicate::nonnegative) {
        let integral: bool = integer_value(z).is_some();
        if integral || *z == Basic::half() {
            return Err(not_ported("zeta (polygamma at the integers and at 1/2)"));
        }
    }
    Ok(node(Function::PolyGamma, vec![n.clone(), z.clone()]))
}

/// The evaluating function of the heads of the module.
pub fn gamma_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::Gamma => gamma_eval(&args[0]),
        Function::LogGamma => loggamma_eval(&args[0]),
        Function::PolyGamma => polygamma_eval(&args[0], &args[1]),
        Function::DiGamma => polygamma(&Basic::zero(), &args[0]),
        _ => polygamma(&Basic::one(), &args[0]),
    }
}

/// Whether `f` is a head of the module.
pub fn is_gamma_function(f: Function) -> bool {
    matches!(
        f,
        Function::Gamma | Function::LogGamma | Function::PolyGamma | Function::DiGamma | Function::TriGamma
    )
}

pub fn gamma(z: &Basic) -> PolyResult<Basic> {
    gamma_function(Function::Gamma, std::slice::from_ref(z))
}

pub fn loggamma(z: &Basic) -> PolyResult<Basic> {
    gamma_function(Function::LogGamma, std::slice::from_ref(z))
}

pub fn polygamma(n: &Basic, z: &Basic) -> PolyResult<Basic> {
    gamma_function(Function::PolyGamma, &[n.clone(), z.clone()])
}

pub fn digamma(z: &Basic) -> PolyResult<Basic> {
    gamma_function(Function::DiGamma, std::slice::from_ref(z))
}

pub fn trigamma(z: &Basic) -> PolyResult<Basic> {
    gamma_function(Function::TriGamma, std::slice::from_ref(z))
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn gamma_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    match f {
        Function::Gamma => mul(&[gamma(&args[0])?, polygamma(&Basic::zero(), &args[0])?]),
        Function::LogGamma => polygamma(&Basic::zero(), &args[0]),
        Function::PolyGamma => {
            if argindex == 2 {
                return polygamma(&add(&[args[0].clone(), Basic::one()])?, &args[1]);
            }
            Err(not_ported("the derivative of polygamma in its order"))
        }
        Function::DiGamma => polygamma(&Basic::one(), &args[0]),
        _ => polygamma(&int(2), &args[0]),
    }
}

/// The conjugate (`_eval_conjugate`); `None` for the default rule.
pub fn gamma_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::conjugate;
    let z: &Basic = &e.args()[0];
    match f {
        Function::Gamma => Ok(Some(gamma(&conjugate(z)?)?)),
        Function::LogGamma => {
            if is_true(z, Predicate::zero) || matches!(z.head(), Head::NegativeInfinity) {
                return Ok(None);
            }
            Ok(Some(loggamma(&conjugate(z)?)?))
        }
        _ => Ok(None),
    }
}

fn known(a: &Facts, p: Predicate) -> Option<bool> {
    a.get(&p).copied()
}

fn fuzzy_and(values: &[Option<bool>]) -> Option<bool> {
    if values.contains(&Some(false)) {
        return Some(false);
    }
    if values.iter().all(|v| *v == Some(true)) {
        return Some(true);
    }
    None
}

fn polygamma_facts(n: &Basic, z: &Basic) -> FactList {
    use Predicate as P;
    let (a, b) = (facts_of(n), facts_of(z));
    let mut facts: FactList = FactList::new();
    facts.set(P::commutative, true);
    if known(a, P::positive) == Some(true) && known(b, P::positive) == Some(true) {
        facts.set(P::real, true);
    }
    let negative_integer: Option<bool> = fuzzy_and(&[known(b, P::negative), known(b, P::integer)]);
    let complex: Option<bool> = fuzzy_and(&[known(b, P::complex), negative_integer.map(|v| !v)]);
    if let Some(v) = complex {
        facts.set(P::complex, v);
    }
    if known(a, P::positive) == Some(true) {
        if known(a, P::odd) == Some(true) && known(b, P::real) == Some(true) {
            facts.set(P::positive, true);
            facts.set(P::negative, false);
        } else if known(a, P::even) == Some(true) && known(b, P::positive) == Some(true) {
            facts.set(P::positive, false);
            facts.set(P::negative, true);
        }
    }
    facts
}

/// The facts (`_eval_facts`).
pub fn gamma_facts(f: Function, args: &[Basic]) -> FactList {
    use Predicate as P;
    match f {
        Function::Gamma => {
            let x: &Basic = &args[0];
            let a: &Facts = facts_of(x);
            let mut facts: FactList = FactList::new();
            facts.set(P::commutative, true);
            let nonpositive_integer: bool =
                known(a, P::nonpositive) == Some(true) && known(a, P::integer) == Some(true);
            if nonpositive_integer || integer_value(x).is_some_and(|v| v <= Int::ZERO) {
                facts.set(P::real, false);
            } else if known(a, P::positive) == Some(true) || known(a, P::noninteger) == Some(true) {
                facts.set(P::real, true);
            }
            if known(a, P::positive) == Some(true) {
                facts.set(P::positive, true);
            } else if known(a, P::noninteger) == Some(true) {
                if let Some(even) = crate::functions::elementary::integers::floor(x)
                    .ok()
                    .and_then(|fl| is(&fl, P::even))
                {
                    facts.set(P::positive, even);
                }
            }
            facts
        }
        Function::LogGamma => {
            let a: &Facts = facts_of(&args[0]);
            let mut facts: FactList = FactList::new();
            facts.set(P::commutative, true);
            if known(a, P::positive) == Some(true) {
                facts.set(P::real, true);
            } else if known(a, P::nonpositive) == Some(true) {
                facts.set(P::real, false);
            }
            facts
        }
        Function::PolyGamma => polygamma_facts(&args[0], &args[1]),
        Function::DiGamma => polygamma_facts(&Basic::zero(), &args[0]),
        _ => polygamma_facts(&Basic::one(), &args[0]),
    }
}

/// The value (`numeric`).
pub fn gamma_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    match f {
        Function::Gamma => gamma_numeric::gamma(&values[0], prec),
        Function::LogGamma => gamma_numeric::loggamma(&values[0], prec),
        Function::PolyGamma => {
            let (nr, ni) = parts(&values[0]);
            if ni != Real::ZERO || nr.clone() != nr.floor() || nr < Real::ZERO {
                return None;
            }
            let m: u64 = nr.to_int().value().try_into().ok()?;
            gamma_numeric::polygamma(m, &values[1], prec)
        }
        Function::DiGamma => gamma_numeric::polygamma(0, &values[0], prec),
        _ => gamma_numeric::polygamma(1, &values[0], prec),
    }
}
