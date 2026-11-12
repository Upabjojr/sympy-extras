//! Port of `sympy_extras/functions/combinatorial/factorials.py`: the
//! factorial, the binomial coefficient, the rising and falling factorials,
//! the subfactorial and the double factorial as function heads, with their
//! evaluation, facts and values (by the gamma function of
//! [`crate::functions::special::gamma_numeric`]). Not ported until the
//! gamma functions are: the derivatives (`polygamma`), the gamma form of
//! `binomial(n, k)` for a non-integer number `k`, the value of the
//! subfactorial at a non-integer (`gammainc`).

use crate::core::add::{add, sub};
use crate::core::assumptions::{facts_of, FactList, Facts, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::Complex;
use crate::core::expr::{is, is_true};
use crate::core::function::expand;
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::{not_ported, value_error, PolyResult};
use crate::functions::special::gamma_numeric::gamma;
use crate::numbers::{denom, numer, Int};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

/// `e` as an integer when it is an `Integer` (`_int`).
fn as_int(e: &Basic) -> Option<Int> {
    let v = e.as_rational()?;
    if denom(v) == Int::ONE {
        Some(numer(v))
    } else {
        None
    }
}

fn small(n: &Int) -> Option<i64> {
    i64::try_from(n).ok()
}

fn factorial_int(n: u64) -> Int {
    let mut result: Int = Int::ONE;
    for i in 2..=n {
        result *= Int::from(i);
    }
    result
}

fn comb(n: u64, k: u64) -> Int {
    if k > n {
        return Int::ZERO;
    }
    let k: u64 = k.min(n - k);
    let mut result: Int = Int::ONE;
    for i in 0..k {
        result = result * Int::from(n - i) / Int::from(i + 1);
    }
    result
}

fn huge() -> crate::error::PolyError {
    not_ported("a factorial of a huge integer")
}

/// `Factorial.eval`.
fn factorial_eval(n: &Basic) -> PolyResult<Basic> {
    if is_true(n, Predicate::zero) {
        return Ok(Basic::one());
    }
    if matches!(n.head(), Head::Infinity) {
        return Ok(Basic::oo());
    }
    if let Some(v) = as_int(n) {
        if v < Int::ZERO {
            return Ok(Basic::zoo());
        }
        let v: u64 = u64::try_from(&v).map_err(|_| huge())?;
        if v > 1_000_000 {
            return Err(huge());
        }
        return Ok(Basic::integer(factorial_int(v)));
    }
    Ok(node(Function::Factorial, vec![n.clone()]))
}

/// `binomial._eval` for a number `n` and an integer `k` (`_numerical`).
fn binomial_numerical(n: &Basic, k: &Basic) -> PolyResult<Option<Basic>> {
    let Some(kk) = as_int(k) else { return Ok(None) };
    if let Some(nn) = as_int(n) {
        if nn >= Int::ZERO {
            if kk > nn {
                return Ok(Some(Basic::zero()));
            }
            let (nn, kk) = (
                u64::try_from(&nn).map_err(|_| huge())?,
                u64::try_from(&kk).map_err(|_| huge())?,
            );
            return Ok(Some(Basic::integer(comb(nn, kk))));
        }
    }
    let kk: u64 = u64::try_from(&kk).map_err(|_| huge())?;
    if kk > 100_000 {
        return Err(huge());
    }
    let mut d: Basic = sub(n, k)?;
    let mut result: Basic = Basic::one();
    for _ in 1..=kk {
        d = add(&[d, Basic::one()])?;
        result = mul(&[result, d.clone()])?;
    }
    Ok(Some(mul(&[
        result,
        pow(&Basic::integer(factorial_int(kk)), &Basic::negative_one())?,
    ])?))
}

/// `Binomial.eval`.
fn binomial_eval(n: &Basic, k: &Basic) -> PolyResult<Basic> {
    let d: Basic = sub(n, k)?;
    let n_nonnegative: Option<bool> = is(n, Predicate::nonnegative);
    let n_integer: Option<bool> = is(n, Predicate::integer);
    let side: bool = n_nonnegative == Some(true) || n_integer == Some(false);
    if is_true(k, Predicate::zero) || (side && is_true(&d, Predicate::zero)) {
        return Ok(Basic::one());
    }
    if is_true(&sub(k, &Basic::one())?, Predicate::zero) || (side && is_true(&sub(&d, &Basic::one())?, Predicate::zero))
    {
        return Ok(n.clone());
    }
    if is_true(k, Predicate::integer) {
        if is_true(k, Predicate::negative)
            || (n_nonnegative == Some(true) && n_integer == Some(true) && is_true(&d, Predicate::negative))
        {
            return Ok(Basic::zero());
        }
        if crate::core::assumptions::is_number(n) {
            return Ok(match binomial_numerical(n, k)? {
                None => node(Function::Binomial, vec![n.clone(), k.clone()]),
                Some(r) if r == Basic::zero() => r,
                Some(r) => expand(&r, true, true, true)?,
            });
        }
        return Ok(node(Function::Binomial, vec![n.clone(), k.clone()]));
    }
    if n_nonnegative == Some(false) && n_integer == Some(true) {
        return Ok(Basic::zoo());
    }
    if crate::core::assumptions::is_number(k) {
        use crate::functions::special::gamma_functions::gamma;
        let denominator: Basic = mul(&[
            gamma(&add(&[k.clone(), Basic::one()])?)?,
            gamma(&add(&[d.clone(), Basic::one()])?)?,
        ])?;
        return mul(&[
            gamma(&add(&[n.clone(), Basic::one()])?)?,
            pow(&denominator, &Basic::negative_one())?,
        ]);
    }
    Ok(node(Function::Binomial, vec![n.clone(), k.clone()]))
}

/// `(x + sign*i)` multiplied over the offsets (`_product`).
fn product(x: &Basic, offsets: std::ops::Range<i64>, sign: i64) -> PolyResult<Basic> {
    let mut result: Basic = Basic::one();
    for i in offsets {
        result = mul(&[result, add(&[x.clone(), int(sign * i)])?])?;
    }
    Ok(result)
}

/// `RisingFactorial.eval` and `FallingFactorial.eval` for an integer `k`
/// (`_factorial_power`).
fn factorial_power(x: &Basic, k: &Basic, rising: bool) -> PolyResult<Option<Basic>> {
    let Some(kk) = as_int(k) else { return Ok(None) };
    let kk: i64 = small(&kk).filter(|v| v.abs() <= 100_000).ok_or_else(huge)?;
    if kk == 0 {
        return Ok(Some(Basic::one()));
    }
    if kk > 0 {
        if matches!(x.head(), Head::Infinity) {
            return Ok(Some(Basic::oo()));
        }
        if matches!(x.head(), Head::NegativeInfinity) {
            return Ok(Some(if kk % 2 != 0 { Basic::negative_oo() } else { Basic::oo() }));
        }
        return Ok(Some(product(x, 0..kk, if rising { 1 } else { -1 })?));
    }
    if matches!(x.head(), Head::Infinity | Head::NegativeInfinity) {
        return Ok(Some(Basic::oo()));
    }
    Ok(Some(pow(
        &product(x, 1..(-kk + 1), if rising { -1 } else { 1 })?,
        &Basic::negative_one(),
    )?))
}

fn rising_eval(x: &Basic, k: &Basic) -> PolyResult<Basic> {
    if matches!(x.head(), Head::NaN) || matches!(k.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if *x == Basic::one() {
        return factorial(k);
    }
    if let Some(v) = factorial_power(x, k, true)? {
        return Ok(v);
    }
    if crate::core::expr::is_false(k, Predicate::integer)
        && is_true(x, Predicate::integer)
        && is_true(x, Predicate::negative)
    {
        return Ok(Basic::zero());
    }
    Ok(node(Function::RisingFactorial, vec![x.clone(), k.clone()]))
}

fn falling_eval(x: &Basic, k: &Basic) -> PolyResult<Basic> {
    if matches!(x.head(), Head::NaN) || matches!(k.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if is_true(k, Predicate::integer) && x == k {
        return factorial(x);
    }
    Ok(factorial_power(x, k, false)?.unwrap_or_else(|| node(Function::FallingFactorial, vec![x.clone(), k.clone()])))
}

fn subfactorial_eval(n: &Basic) -> PolyResult<Basic> {
    if let Some(v) = as_int(n) {
        if v >= Int::ZERO {
            let v: u64 = u64::try_from(&v).map_err(|_| huge())?;
            if v > 100_000 {
                return Err(huge());
            }
            if v == 0 {
                return Ok(Basic::one());
            }
            let (mut previous, mut current) = (Int::ONE, Int::ZERO);
            for i in 2..=v {
                let next: Int = Int::from(i - 1) * (&current + &previous);
                previous = current;
                current = next;
            }
            return Ok(Basic::integer(current));
        }
    }
    if matches!(n.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if matches!(n.head(), Head::Infinity) {
        return Ok(Basic::oo());
    }
    Ok(node(Function::Subfactorial, vec![n.clone()]))
}

fn div(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}

fn factorial2_eval(n: &Basic) -> PolyResult<Basic> {
    if !n.is_number_node() {
        return Ok(node(Function::Factorial2, vec![n.clone()]));
    }
    let message = || value_error("argument must be nonnegative integer or negative odd integer".to_string());
    let Some(v) = as_int(n) else { return Err(message()) };
    let odd: bool = &v % Int::from(2) != Int::ZERO;
    if v >= Int::ZERO {
        if !odd {
            let k: Basic = Basic::integer(&v / Int::from(2));
            return mul(&[pow(&int(2), &k)?, factorial(&k)?]);
        }
        return div(&factorial(n)?, &factorial2(&sub(n, &Basic::one())?)?);
    }
    if odd {
        let sign: Basic = pow(&Basic::negative_one(), &div(&sub(&Basic::one(), n)?, &int(2))?)?;
        return div(
            &mul(&[n.clone(), sign])?,
            &factorial2(&mul(&[Basic::negative_one(), n.clone()])?)?,
        );
    }
    Err(message())
}

/// The evaluating function of the heads of the module.
pub fn combinatorial(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::Factorial => factorial_eval(&args[0]),
        Function::Binomial => binomial_eval(&args[0], &args[1]),
        Function::RisingFactorial => rising_eval(&args[0], &args[1]),
        Function::FallingFactorial => falling_eval(&args[0], &args[1]),
        Function::Subfactorial => subfactorial_eval(&args[0]),
        _ => factorial2_eval(&args[0]),
    }
}

/// Whether `f` is a head of the module.
pub fn is_combinatorial(f: Function) -> bool {
    matches!(
        f,
        Function::Factorial
            | Function::Binomial
            | Function::RisingFactorial
            | Function::FallingFactorial
            | Function::Subfactorial
            | Function::Factorial2
    )
}

pub fn factorial(n: &Basic) -> PolyResult<Basic> {
    combinatorial(Function::Factorial, std::slice::from_ref(n))
}

pub fn binomial(n: &Basic, k: &Basic) -> PolyResult<Basic> {
    combinatorial(Function::Binomial, &[n.clone(), k.clone()])
}

pub fn rf(x: &Basic, k: &Basic) -> PolyResult<Basic> {
    combinatorial(Function::RisingFactorial, &[x.clone(), k.clone()])
}

pub fn ff(x: &Basic, k: &Basic) -> PolyResult<Basic> {
    combinatorial(Function::FallingFactorial, &[x.clone(), k.clone()])
}

pub fn subfactorial(n: &Basic) -> PolyResult<Basic> {
    combinatorial(Function::Subfactorial, std::slice::from_ref(n))
}

pub fn factorial2(n: &Basic) -> PolyResult<Basic> {
    combinatorial(Function::Factorial2, std::slice::from_ref(n))
}

/// The derivative in the argument `argindex` (from 1, `fdiff`): by
/// `polygamma` for the factorial and the binomial, the generic
/// `Derivative(f(x), x)` of `DefinedFunction.fdiff` for the others.
pub fn combinatorial_derivative(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::functions::special::gamma_functions::{gamma, polygamma};
    let args: &[Basic] = e.args();
    let one: Basic = Basic::one();
    match f {
        Function::Factorial => {
            let n: Basic = add(&[args[0].clone(), one])?;
            mul(&[gamma(&n)?, polygamma(&Basic::zero(), &n)?])
        }
        Function::Binomial => {
            let (n, k) = (&args[0], &args[1]);
            let psi = |x: Basic| polygamma(&Basic::zero(), &x);
            let n_k1: Basic = add(&[sub(n, k)?, one.clone()])?;
            let difference: Basic = if argindex == 1 {
                sub(&psi(add(&[n.clone(), one.clone()])?)?, &psi(n_k1)?)?
            } else {
                sub(&psi(n_k1)?, &psi(add(&[k.clone(), one.clone()])?)?)?
            };
            mul(&[binomial(n, k)?, difference])
        }
        _ => {
            let a: &Basic = &args[argindex - 1];
            let elsewhere: bool = args
                .iter()
                .enumerate()
                .any(|(i, v)| i != argindex - 1 && v.free_symbols().contains(a));
            if a.is_symbol() && !elsewhere {
                return Ok(crate::core::function::derivative_node(e, &[(a.clone(), 1)]));
            }
            Err(not_ported("the derivative of a factorial as a Subs"))
        }
    }
}

fn known(a: &Facts, p: Predicate) -> Option<bool> {
    a.get(&p).copied()
}

/// The facts (`_eval_facts`).
pub fn combinatorial_facts(f: Function, args: &[Basic]) -> FactList {
    use Predicate as P;
    let mut facts: FactList = FactList::new();
    facts.set(P::commutative, true);
    let a: &Facts = facts_of(&args[0]);
    let yes = |p: P| known(a, p) == Some(true);
    match f {
        Function::Factorial => {
            if yes(P::integer) && yes(P::nonnegative) {
                facts.set(P::integer, true);
                facts.set(P::positive, true);
                if let Some(v) = sub(&args[0], &int(2)).ok().and_then(|d| is(&d, P::nonnegative)) {
                    facts.set(P::even, v);
                }
                if let Some(v) = sub(&args[0], &int(3)).ok().and_then(|d| is(&d, P::nonnegative)) {
                    facts.set(P::composite, v);
                }
            }
            if yes(P::nonnegative) || yes(P::noninteger) {
                facts.set(P::real, true);
            }
        }
        Function::Binomial => {
            let k: &Facts = facts_of(&args[1]);
            if yes(P::integer) && known(k, P::integer) == Some(true) {
                facts.set(P::integer, true);
                if yes(P::nonnegative) || known(k, P::negative) == Some(true) || known(k, P::even) == Some(true) {
                    facts.set(P::nonnegative, true);
                } else if known(k, P::even) == Some(false) {
                    facts.set(P::nonnegative, false);
                }
            } else if known(k, P::integer) == Some(false) {
                facts.set(P::integer, false);
            }
        }
        Function::RisingFactorial | Function::FallingFactorial => {
            let k: &Facts = facts_of(&args[1]);
            let values: [Option<bool>; 3] = [known(a, P::integer), known(k, P::integer), known(k, P::nonnegative)];
            if values.iter().all(|v| *v == Some(true)) {
                facts.set(P::integer, true);
            } else if values.contains(&Some(false)) {
                facts.set(P::integer, false);
            }
        }
        Function::Subfactorial => {
            if yes(P::odd) && yes(P::nonnegative) {
                facts.set(P::even, true);
            }
            if yes(P::integer) && yes(P::nonnegative) {
                facts.set(P::integer, true);
                facts.set(P::nonnegative, true);
            }
            if yes(P::even) && yes(P::nonnegative) {
                facts.set(P::odd, true);
            }
        }
        _ => factorial2_facts(&args[0], &mut facts),
    }
    facts
}

fn factorial2_facts(n: &Basic, facts: &mut FactList) {
    use Predicate as P;
    let t = |e: &Basic, p: P| is_true(e, p);
    if t(n, P::integer) {
        if t(n, P::odd) {
            facts.set(P::even, false);
        } else if t(n, P::even) {
            if t(n, P::positive) {
                facts.set(P::even, true);
            } else if t(n, P::zero) {
                facts.set(P::even, false);
            }
        }
        if add(&[n.clone(), Basic::one()]).is_ok_and(|e| t(&e, P::nonnegative)) {
            facts.set(P::integer, true);
            facts.set(P::positive, true);
        } else if t(n, P::odd) {
            if let Some(v) = add(&[n.clone(), int(3)]).ok().and_then(|e| is(&e, P::nonnegative)) {
                facts.set(P::integer, v);
            }
            if let Some(v) = add(&[n.clone(), Basic::one()])
                .and_then(|e| div(&e, &int(2)))
                .ok()
                .and_then(|e| is(&e, P::even))
            {
                facts.set(P::positive, v);
            }
        }
    }
    if t(n, P::odd) {
        if let Some(v) = add(&[n.clone(), int(3)]).ok().and_then(|e| is(&e, P::nonnegative)) {
            facts.set(P::odd, v);
        }
    } else if t(n, P::even) {
        if t(n, P::positive) {
            facts.set(P::odd, false);
        } else if t(n, P::zero) {
            facts.set(P::odd, true);
        }
    }
}

fn c_add(a: &Complex, n: i64, prec: usize) -> Complex {
    a + &crate::core::evalf::complex_real(crate::core::evalf::real_int(&Int::from(n), prec + 40), prec + 40)
}

/// The value (`numeric`), by the gamma function; `None` at its poles and
/// for the subfactorial (`gammainc`, not ported).
pub fn combinatorial_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    match f {
        Function::Factorial => gamma(&c_add(&values[0], 1, prec), prec),
        Function::Binomial => {
            let (n, k) = (&values[0], &values[1]);
            gamma_product(
                &[c_add(n, 1, prec)],
                &[c_add(k, 1, prec), c_add(&(n - k), 1, prec)],
                prec,
            )
        }
        Function::RisingFactorial => {
            let (x, k) = (&values[0], &values[1]);
            gamma_product(&[x + k], std::slice::from_ref(x), prec)
        }
        Function::FallingFactorial => {
            let (x, k) = (&values[0], &values[1]);
            gamma_product(&[c_add(x, 1, prec)], &[c_add(&(x - k), 1, prec)], prec)
        }
        Function::Subfactorial => {
            // mpmath's gammainc(n + 1, -1)/e: gamma(a)/e - (-1)**a*sum_k (-1)**k/(a)_(k+1)
            use crate::core::evalf::{complex, complex_real, real_int};
            let wp: usize = prec + 40;
            let one: Complex = complex_real(real_int(&Int::ONE, wp), wp);
            let a: Complex = &values[0] + &one;
            let (ar, ai) = crate::core::evalf::parts(&a);
            if ai == crate::core::evalf::Real::ZERO && ar <= crate::core::evalf::Real::ZERO && ar.clone() == ar.floor()
            {
                return Some(subfactorial_at_pole(&ar, prec));
            }
            let g: Complex = gamma(&a, wp)?;
            let e: Complex = one.exp();
            let mut term: Complex = &one / &a;
            let mut total: Complex = term.clone();
            let mut k: i64 = 1;
            loop {
                let ak: Complex = &a + &complex_real(real_int(&Int::from(k), wp), wp);
                term = &(-&term) / &ak;
                total = &total + &term;
                let (tr, ti) = crate::core::evalf::parts(&term);
                let size: f64 = tr.to_f64().value().abs().max(ti.to_f64().value().abs());
                if size == 0.0 || size.log2() < -((wp + 10) as f64) || k > 100_000 {
                    break;
                }
                k += 1;
            }
            let pi: crate::core::evalf::Real = crate::core::evalf::Real::pi(wp);
            let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
            let sign: Complex = (&(&i * &complex_real(pi, wp)) * &a).exp();
            let value: Complex = &(&g / &e) - &(&sign * &total);
            let (re, im) = crate::core::evalf::parts(&value);
            Some(complex(
                re.with_precision(prec).value(),
                im.with_precision(prec).value(),
            ))
        }
        Function::Factorial2 => {
            // mpmath's fac2: 2**(x/2)*(pi/2)**((cos(pi*x) - 1)/4)*gamma(x/2 + 1)
            use crate::core::evalf::{complex, complex_real, real_int, real_rat};
            let wp: usize = prec + 40;
            let x: &Complex = &values[0];
            let half: Complex = complex_real(real_rat(&crate::numbers::rat(1, 2), wp), wp);
            let quarter: Complex = complex_real(real_rat(&crate::numbers::rat(1, 4), wp), wp);
            let one: Complex = complex_real(real_int(&Int::ONE, wp), wp);
            let two: Complex = complex_real(real_int(&Int::from(2u8), wp), wp);
            let pi: Complex = complex_real(crate::core::evalf::Real::pi(wp), wp);
            let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
            let pix: Complex = &pi * x;
            let cos: Complex = &(&(&i * &pix).exp() + &(-(&i * &pix)).exp()) * &half;
            let exponent: Complex = &(&cos - &one) * &quarter;
            let g: Complex = gamma(&(&(x * &half) + &one), wp)?;
            let a: Complex = (&(x * &half) * &two.ln()).exp();
            let b: Complex = (&exponent * &(&pi * &half).ln()).exp();
            let value: Complex = &(&a * &b) * &g;
            let (re, im) = crate::core::evalf::parts(&value);
            let (_, xi) = crate::core::evalf::parts(x);
            let im = if xi == crate::core::evalf::Real::ZERO {
                real_int(&Int::ZERO, prec)
            } else {
                im
            };
            Some(complex(
                re.with_precision(prec).value(),
                im.with_precision(prec).value(),
            ))
        }
        _ => None,
    }
}

/// `gammainc(-N, -1)/e` for an integer `N >= 0` (a pole of `gamma`):
/// `(-1)**N/N! * (E1(-1) - e*sum_{k < N} (-1)**k*k!/(-1)**(k + 1))/e`,
/// with `E1(-1) = -Ei(1) - I*pi` and `Ei(1) = EulerGamma + sum 1/(k*k!)`.
fn subfactorial_at_pole(a: &crate::core::evalf::Real, prec: usize) -> Complex {
    use crate::core::evalf::{complex, complex_real, euler_gamma, real_int, Real};
    let wp: usize = prec + 40;
    let n: u64 = (-a.clone()).to_int().value().try_into().unwrap_or(0);
    // Ei(1)
    let mut ei: Real = euler_gamma(wp);
    let mut factorial: Real = real_int(&Int::ONE, wp);
    let mut k: u64 = 1;
    loop {
        factorial *= real_int(&Int::from(k), wp);
        let term: Real = real_int(&Int::ONE, wp) / (real_int(&Int::from(k), wp) * factorial.clone());
        ei += term.clone();
        if term.to_f64().value().log2() < -((wp + 10) as f64) {
            break;
        }
        k += 1;
    }
    let e1: Complex = complex(-ei, -Real::pi(wp));
    // e*sum_{k < N} (-1)**k*k!/(-1)**(k + 1) = -e*sum k!
    let mut total: Real = real_int(&Int::ZERO, wp);
    let mut kf: Real = real_int(&Int::ONE, wp);
    for k in 0..n {
        if k > 0 {
            kf *= real_int(&Int::from(k), wp);
        }
        total += kf.clone();
    }
    let e: Real = real_int(&Int::ONE, wp).exp();
    let inner: Complex = &e1 + &complex_real(e.clone() * total, wp);
    let mut nf: Real = real_int(&Int::ONE, wp);
    for k in 2..=n {
        nf *= real_int(&Int::from(k), wp);
    }
    let sign: Real = if n % 2 == 1 {
        -real_int(&Int::ONE, wp)
    } else {
        real_int(&Int::ONE, wp)
    };
    let value: Complex = &inner * &complex_real(sign / (nf * e), wp);
    let (re, im) = crate::core::evalf::parts(&value);
    complex(re.with_precision(prec).value(), im.with_precision(prec).value())
}

fn is_gamma_pole(z: &Complex) -> bool {
    let (re, im) = crate::core::evalf::parts(z);
    im == crate::core::evalf::Real::ZERO && re <= crate::core::evalf::Real::ZERO && re.clone() == re.floor()
}

/// `prod gamma(a) / prod gamma(b)` as mpmath's `gammaprod`: zero when the
/// denominator has more poles, infinite (`None`) when the numerator has.
pub(crate) fn gamma_product(a: &[Complex], b: &[Complex], prec: usize) -> Option<Complex> {
    let pa: usize = a.iter().filter(|z| is_gamma_pole(z)).count();
    let pb: usize = b.iter().filter(|z| is_gamma_pole(z)).count();
    if pb > pa {
        use crate::core::evalf::{complex, real_int};
        return Some(complex(real_int(&Int::ZERO, prec), real_int(&Int::ZERO, prec)));
    }
    if pa > pb {
        return None;
    }
    // as many poles above as below: mpmath's limit with one epsilon,
    // gamma(-m + eps) ~ (-1)**m/(m!*eps)
    let mut pole_ratio: crate::numbers::Rat = crate::numbers::Rat::ONE;
    let residue = |z: &Complex| -> crate::numbers::Rat {
        let m: u64 = (-crate::core::evalf::parts(z).0)
            .to_int()
            .value()
            .try_into()
            .unwrap_or(0);
        let mut f: Int = Int::ONE;
        for i in 2..=m {
            f *= Int::from(i);
        }
        let sign: crate::numbers::Rat = if m % 2 == 1 {
            -crate::numbers::Rat::ONE
        } else {
            crate::numbers::Rat::ONE
        };
        sign / crate::numbers::Rat::from(f)
    };
    for z in a.iter().filter(|z| is_gamma_pole(z)) {
        pole_ratio *= residue(z);
    }
    for z in b.iter().filter(|z| is_gamma_pole(z)) {
        pole_ratio /= residue(z);
    }
    let a: Vec<Complex> = a.iter().filter(|z| !is_gamma_pole(z)).cloned().collect();
    let b: Vec<Complex> = b.iter().filter(|z| !is_gamma_pole(z)).cloned().collect();
    let (a, b) = (&a[..], &b[..]);
    let mut top: Complex =
        crate::core::evalf::complex_real(crate::core::evalf::real_int(&Int::ONE, prec + 40), prec + 40);
    for z in a {
        top = &top * &gamma(z, prec + 20)?;
    }
    let mut bottom: Complex =
        crate::core::evalf::complex_real(crate::core::evalf::real_int(&Int::ONE, prec + 40), prec + 40);
    for z in b {
        bottom = &bottom * &gamma(z, prec + 20)?;
    }
    let ratio: Complex =
        crate::core::evalf::complex_real(crate::core::evalf::real_rat(&pole_ratio, prec + 40), prec + 40);
    Some(&(&top / &bottom) * &ratio)
}
