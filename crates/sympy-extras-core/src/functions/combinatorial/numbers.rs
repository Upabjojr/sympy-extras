//! Port of the combinatorial numbers of
//! `sympy_extras/functions/combinatorial/numbers.py`: `catalan`,
//! `fibonacci` (and the Fibonacci polynomials), `lucas`, `harmonic`,
//! `bernoulli`, `euler`, `bell`, `genocchi` (numbers and polynomials) and
//! `partition`: evaluation, derivatives, facts and values (`catalan` by the
//! gamma function, `fibonacci` and `lucas` by Binet's formula, `harmonic`
//! by digamma and zeta; mpmath's `bell` of a non-integer is not ported).
//! The number-theoretic functions of the module are not ported yet.

use crate::core::add::{add, sub};
use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, complex_real, parts, real_int, Complex, Real};
use crate::core::expr::{is, is_true};
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::{not_ported, value_error, PolyError, PolyResult};
use crate::numbers::{denom, numer, Int, Rat};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

fn big(n: Int) -> Basic {
    Basic::integer(n)
}

fn number(r: &Rat) -> PolyResult<Basic> {
    if denom(r) == Int::ONE {
        return Ok(Basic::integer(numer(r)));
    }
    Basic::rational(numer(r), denom(r))
}

fn as_int(e: &Basic) -> Option<i64> {
    let v = e.as_rational()?;
    if denom(v) == Int::ONE {
        i64::try_from(&numer(v)).ok()
    } else {
        None
    }
}

fn huge() -> PolyError {
    not_ported("a combinatorial number of a huge index")
}

fn checked(k: i64) -> PolyResult<i64> {
    if k.abs() > 20_000 {
        return Err(huge());
    }
    Ok(k)
}

fn binomial(n: i64, k: i64) -> Int {
    let mut r: Int = Int::ONE;
    for i in 0..k {
        r = r * Int::from(n - i) / Int::from(i + 1);
    }
    r
}

fn fibonacci_number(n: i64) -> Int {
    let (mut a, mut b) = (Int::ZERO, Int::ONE);
    for _ in 0..n {
        let next: Int = &a + &b;
        a = b;
        b = next;
    }
    a
}

fn stirling2(n: i64, k: i64) -> Int {
    let mut total: Int = Int::ZERO;
    for i in 0..=k {
        let mut p: Int = Int::ONE;
        for _ in 0..n {
            p *= Int::from(k - i);
        }
        let term: Int = binomial(k, i) * p;
        if i % 2 == 0 {
            total += term;
        } else {
            total -= term;
        }
    }
    let mut f: Int = Int::ONE;
    for i in 2..=k {
        f *= Int::from(i);
    }
    total / f
}

fn partitions(n: i64) -> Int {
    let n: usize = n as usize;
    let mut p: Vec<Int> = vec![Int::ZERO; n + 1];
    p[0] = Int::ONE;
    for m in 1..=n {
        let mut total: Int = Int::ZERO;
        let mut k: usize = 1;
        loop {
            let (g1, g2) = (k * (3 * k - 1) / 2, k * (3 * k + 1) / 2);
            if g1 > m {
                break;
            }
            let positive: bool = k % 2 == 1;
            let add_term = |t: &mut Int, v: &Int| {
                if positive {
                    *t += v
                } else {
                    *t -= v
                }
            };
            add_term(&mut total, &p[m - g1]);
            if g2 <= m {
                add_term(&mut total, &p[m - g2]);
            }
            k += 1;
        }
        p[m] = total;
    }
    p[n].clone()
}

fn bernoulli_number(n: usize) -> Rat {
    crate::functions::special::zeta_functions::bernoulli_number(n)
}

/// The coefficients of the Bernoulli polynomial, from the constant term
/// (`_bernoulli_coefficients`, `b_1 = -1/2`).
fn bernoulli_coefficients(n: i64) -> Vec<Rat> {
    let mut coefficients: Vec<Rat> = vec![Rat::ZERO; n as usize + 1];
    for k in 0..=n {
        let b: Rat = crate::functions::special::gamma_numeric::bernoulli(k as usize);
        coefficients[(n - k) as usize] = Rat::from(binomial(n, k)) * b;
    }
    coefficients
}

fn polynomial(coefficients: &[Rat], x: &Basic) -> PolyResult<Basic> {
    let mut terms: Vec<Basic> = Vec::new();
    for (j, c) in coefficients.iter().enumerate() {
        if *c != Rat::ZERO {
            terms.push(mul(&[number(c)?, pow(x, &int(j as i64))?])?);
        }
    }
    add(&terms)
}

fn two_power(j: i64) -> Rat {
    let mut r: Int = Int::ONE;
    for _ in 0..j {
        r *= Int::from(2);
    }
    Rat::from(r)
}

fn euler_coefficients(n: i64) -> Vec<Rat> {
    let b: Vec<Rat> = bernoulli_coefficients(n + 1);
    let scale: Rat = Rat::from(Int::from(2)) / Rat::from(Int::from(n + 1));
    let top: Rat = two_power(n + 1);
    b.iter()
        .enumerate()
        .map(|(j, c)| &scale * c * (Rat::ONE - &top / two_power(j as i64)))
        .take(n as usize + 1)
        .collect()
}

fn genocchi_coefficients(n: i64) -> Vec<Rat> {
    let b: Vec<Rat> = bernoulli_coefficients(n);
    let top: Rat = two_power(n);
    b.iter()
        .enumerate()
        .map(|(j, c)| Rat::from(Int::from(2)) * c * (&top / two_power(j as i64) - Rat::ONE))
        .collect()
}

/// The Euler number `E_k` (mpmath's `eulernum(k, exact=True)`), from the
/// Euler polynomial: `E_k = 2**k*E_k(1/2)`.
fn euler_number(k: i64) -> Rat {
    let coefficients: Vec<Rat> = euler_coefficients(k);
    let half: Rat = Rat::ONE / Rat::from(Int::from(2));
    let mut value: Rat = Rat::ZERO;
    let mut power: Rat = Rat::ONE;
    for c in &coefficients {
        value += c * &power;
        power *= &half;
    }
    value * two_power(k)
}

fn catalan_eval(n: &Basic) -> PolyResult<Basic> {
    use crate::functions::special::gamma_functions::gamma;
    let integral_nonnegative: bool = as_int(n).is_some_and(|k| k >= 0);
    if integral_nonnegative || (is_true(n, Predicate::noninteger) && is_true(n, Predicate::negative)) {
        let half: Basic = Basic::half();
        let top: Basic = mul(&[pow(&int(4), n)?, gamma(&add(&[n.clone(), half.clone()])?)?])?;
        let bottom: Basic = mul(&[gamma(&half)?, gamma(&add(&[n.clone(), int(2)])?)?])?;
        return div(&top, &bottom);
    }
    if is_true(n, Predicate::integer) && is_true(n, Predicate::negative) {
        let next: Basic = add(&[n.clone(), Basic::one()])?;
        if is_true(&next, Predicate::negative) {
            return Ok(Basic::zero());
        }
        if is_true(&next, Predicate::zero) {
            return Basic::rational(Int::from(-1), Int::from(2));
        }
    }
    Ok(node(Function::Catalan, vec![n.clone()]))
}

fn fibonacci_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    if matches!(n.head(), Head::Infinity) {
        return Ok(Basic::oo());
    }
    let Some(k) = as_int(n) else {
        return Ok(node(Function::Fibonacci, args.to_vec()));
    };
    let k: i64 = checked(k)?;
    if args.len() == 1 {
        if k < 0 {
            return mul(&[pow(&Basic::negative_one(), &int(k + 1))?, fibonacci(&int(-k), None)?]);
        }
        return Ok(big(fibonacci_number(k)));
    }
    if k < 1 {
        return Err(value_error(
            "Fibonacci polynomials are defined only for positive integer indices.".to_string(),
        ));
    }
    let x: &Basic = &args[1];
    let mut previous: Vec<Int> = vec![Int::ONE];
    let mut current: Vec<Int> = vec![Int::ZERO, Int::ONE];
    if k == 1 {
        current = previous.clone();
    }
    for _ in 0..(k - 2).max(0) {
        let mut shifted: Vec<Int> = vec![Int::ZERO];
        shifted.extend(current.iter().cloned());
        let following: Vec<Int> = (0..shifted.len())
            .map(|i| &shifted[i] + previous.get(i).cloned().unwrap_or(Int::ZERO))
            .collect();
        previous = current;
        current = following;
    }
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in current.iter().enumerate() {
        if *c != Int::ZERO {
            terms.push(mul(&[big(c.clone()), pow(x, &int(i as i64))?])?);
        }
    }
    add(&terms)
}

fn lucas_eval(n: &Basic) -> PolyResult<Basic> {
    if matches!(n.head(), Head::Infinity) {
        return Ok(Basic::oo());
    }
    if as_int(n).is_some() {
        let one: Basic = Basic::one();
        return add(&[
            fibonacci(&add(&[n.clone(), one.clone()])?, None)?,
            fibonacci(&sub(n, &one)?, None)?,
        ]);
    }
    Ok(node(Function::Lucas, vec![n.clone()]))
}

fn bernoulli_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    let x: Option<&Basic> = args.get(1);
    if x == Some(&Basic::one()) {
        return bernoulli(n, None);
    }
    if is_true(n, Predicate::zero) {
        return Ok(Basic::one());
    }
    if crate::core::expr::is_false(n, Predicate::integer) || crate::core::expr::is_false(n, Predicate::nonnegative) {
        if x.is_some_and(|x| as_int(x).is_some_and(|v| v <= 0)) {
            return Ok(Basic::nan());
        }
        return Ok(node(Function::Bernoulli, args.to_vec()));
    }
    let k: Option<i64> = as_int(n);
    match x {
        None => {
            if *n == Basic::one() {
                return Ok(Basic::half());
            }
            if is_true(n, Predicate::odd) && is_true(&sub(n, &Basic::one())?, Predicate::positive) {
                return Ok(Basic::zero());
            }
            if let Some(k) = k {
                return number(&bernoulli_number(checked(k)? as usize));
            }
            Ok(node(Function::Bernoulli, args.to_vec()))
        }
        Some(x) => {
            if let Some(k) = k {
                return crate::functions::special::zeta_functions::bernoulli_polynomial(checked(k)?, x);
            }
            Ok(node(Function::Bernoulli, args.to_vec()))
        }
    }
}

fn harmonic_eval(args: &[Basic]) -> PolyResult<Basic> {
    use crate::core::logic::truth;
    use crate::core::relational::{gt, le};
    use crate::functions::special::zeta_functions::zeta;
    let n: &Basic = &args[0];
    let given: Option<&Basic> = args.get(1);
    if given == Some(&Basic::one()) {
        return harmonic(n, None);
    }
    let m: Basic = given.cloned().unwrap_or_else(Basic::one);
    if is_true(n, Predicate::zero) {
        return Ok(Basic::zero());
    }
    if is_true(&m, Predicate::zero) {
        return Ok(n.clone());
    }
    if matches!(n.head(), Head::Infinity) {
        if is_true(&m, Predicate::negative) {
            return Ok(Basic::nan());
        }
        if truth(&le(&m, &Basic::one())?) == Some(true) {
            return Ok(Basic::oo());
        }
        if truth(&gt(&m, &Basic::one())?) == Some(true) {
            return zeta(&m, None);
        }
        return Ok(node(Function::Harmonic, args.to_vec()));
    }
    let order: Option<i64> = as_int(&m);
    if let Some(order) = order.filter(|o| *o <= 0) {
        let degree: Basic = int(1 - checked(order)?);
        let top: Basic = sub(
            &bernoulli(&degree, Some(&add(&[n.clone(), Basic::one()])?))?,
            &bernoulli(&degree, None)?,
        )?;
        return div(&top, &degree);
    }
    let Some(k) = as_int(n) else {
        return Ok(node(Function::Harmonic, args.to_vec()));
    };
    if k < 0
        && (crate::core::expr::is_false(&m, Predicate::integer)
            || crate::core::expr::is_false(&m, Predicate::nonpositive))
    {
        return Ok(if m == Basic::one() { Basic::zoo() } else { Basic::nan() });
    }
    if k >= 0 {
        let k: i64 = checked(k)?;
        if let Some(order) = order {
            let mut total: Rat = Rat::ZERO;
            for j in 1..=k {
                let mut p: Int = Int::ONE;
                for _ in 0..order {
                    p *= Int::from(j);
                }
                total += Rat::ONE / Rat::from(p);
            }
            return number(&total);
        }
        let mut terms: Vec<Basic> = Vec::new();
        for j in 1..=k {
            terms.push(pow(&int(j), &mul(&[Basic::negative_one(), m.clone()])?)?);
        }
        return add(&terms);
    }
    Ok(node(Function::Harmonic, args.to_vec()))
}

fn euler_eval(args: &[Basic]) -> PolyResult<Basic> {
    use crate::functions::special::gamma_functions::digamma;
    let n: &Basic = &args[0];
    let x: Option<&Basic> = args.get(1);
    if is_true(n, Predicate::zero) {
        return Ok(Basic::one());
    }
    if *n == Basic::negative_one() {
        return match x {
            None => div(&Basic::pi(), &int(2)),
            Some(x) => sub(
                &digamma(&div(&add(&[x.clone(), Basic::one()])?, &int(2))?)?,
                &digamma(&div(x, &int(2))?)?,
            ),
        };
    }
    if crate::core::expr::is_false(n, Predicate::integer) || crate::core::expr::is_false(n, Predicate::nonnegative) {
        return Ok(node(Function::Euler, args.to_vec()));
    }
    let k: Option<i64> = as_int(n);
    match x {
        None => {
            if is_true(n, Predicate::odd) && is_true(n, Predicate::positive) {
                return Ok(Basic::zero());
            }
            if let Some(k) = k {
                return number(&euler_number(checked(k)?));
            }
            Ok(node(Function::Euler, args.to_vec()))
        }
        Some(x) => {
            if let Some(k) = k {
                return polynomial(&euler_coefficients(checked(k)?), x);
            }
            Ok(node(Function::Euler, args.to_vec()))
        }
    }
}

fn bell_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    let x: Option<&Basic> = args.get(1);
    if matches!(n.head(), Head::Infinity) {
        if x.is_none() {
            return Ok(Basic::oo());
        }
        return Err(value_error("Bell polynomial is not defined".to_string()));
    }
    if is_true(n, Predicate::negative) || crate::core::expr::is_false(n, Predicate::integer) {
        return Err(value_error("a non-negative integer expected".to_string()));
    }
    let Some(k) = as_int(n) else {
        return Ok(node(Function::Bell, args.to_vec()));
    };
    let k: i64 = checked(k)?;
    match x {
        None => {
            let mut total: Int = Int::ZERO;
            for j in 0..=k {
                total += stirling2(k, j);
            }
            Ok(big(total))
        }
        Some(x) => {
            let mut terms: Vec<Basic> = Vec::new();
            for j in 0..=k {
                let s: Int = stirling2(k, j);
                if s != Int::ZERO {
                    terms.push(mul(&[big(s), pow(x, &int(j))?])?);
                }
            }
            add(&terms)
        }
    }
}

fn genocchi_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    let x: Option<&Basic> = args.get(1);
    if x == Some(&Basic::one()) {
        return genocchi(n, None);
    }
    if crate::core::expr::is_false(n, Predicate::integer) || crate::core::expr::is_false(n, Predicate::nonnegative) {
        return Ok(node(Function::Genocchi, args.to_vec()));
    }
    let k: Option<i64> = as_int(n);
    match x {
        None => {
            if is_true(n, Predicate::odd) && is_true(&sub(n, &Basic::one())?, Predicate::positive) {
                return Ok(Basic::zero());
            }
            if k.is_some() {
                checked(k.unwrap_or(0))?;
                return mul(&[int(2), sub(&Basic::one(), &pow(&int(2), n)?)?, bernoulli(n, None)?]);
            }
            Ok(node(Function::Genocchi, args.to_vec()))
        }
        Some(x) => {
            if let Some(k) = k {
                return polynomial(&genocchi_coefficients(checked(k)?), x);
            }
            Ok(node(Function::Genocchi, args.to_vec()))
        }
    }
}

fn partition_eval(n: &Basic) -> PolyResult<Basic> {
    if crate::core::expr::is_false(n, Predicate::integer) {
        return Err(PolyError::Type("n should be an integer".to_string()));
    }
    if is_true(n, Predicate::negative) {
        return Ok(Basic::zero());
    }
    if is_true(n, Predicate::zero) || *n == Basic::one() {
        return Ok(Basic::one());
    }
    if let Some(k) = as_int(n) {
        return Ok(big(partitions(checked(k)?)));
    }
    Ok(node(Function::Partition, vec![n.clone()]))
}

/// The heads of the module.
pub fn is_combinatorial_number(f: Function) -> bool {
    matches!(
        f,
        Function::Catalan
            | Function::Fibonacci
            | Function::Lucas
            | Function::Harmonic
            | Function::Bernoulli
            | Function::Euler
            | Function::Bell
            | Function::Genocchi
            | Function::Partition
    )
}

/// The evaluating function of the heads of the module.
pub fn combinatorial_number(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::Catalan => catalan_eval(&args[0]),
        Function::Fibonacci => fibonacci_eval(args),
        Function::Lucas => lucas_eval(&args[0]),
        Function::Harmonic => harmonic_eval(args),
        Function::Bernoulli => bernoulli_eval(args),
        Function::Euler => euler_eval(args),
        Function::Bell => bell_eval(args),
        Function::Genocchi => genocchi_eval(args),
        _ => partition_eval(&args[0]),
    }
}

fn with_optional(f: Function, n: &Basic, x: Option<&Basic>) -> PolyResult<Basic> {
    match x {
        None => combinatorial_number(f, std::slice::from_ref(n)),
        Some(x) => combinatorial_number(f, &[n.clone(), x.clone()]),
    }
}

pub fn catalan(n: &Basic) -> PolyResult<Basic> {
    combinatorial_number(Function::Catalan, std::slice::from_ref(n))
}
pub fn fibonacci(n: &Basic, x: Option<&Basic>) -> PolyResult<Basic> {
    with_optional(Function::Fibonacci, n, x)
}
pub fn lucas(n: &Basic) -> PolyResult<Basic> {
    combinatorial_number(Function::Lucas, std::slice::from_ref(n))
}
pub fn harmonic(n: &Basic, m: Option<&Basic>) -> PolyResult<Basic> {
    with_optional(Function::Harmonic, n, m)
}
pub fn bernoulli(n: &Basic, x: Option<&Basic>) -> PolyResult<Basic> {
    with_optional(Function::Bernoulli, n, x)
}
pub fn euler(n: &Basic, x: Option<&Basic>) -> PolyResult<Basic> {
    with_optional(Function::Euler, n, x)
}
pub fn bell(n: &Basic, x: Option<&Basic>) -> PolyResult<Basic> {
    with_optional(Function::Bell, n, x)
}
pub fn genocchi(n: &Basic, x: Option<&Basic>) -> PolyResult<Basic> {
    with_optional(Function::Genocchi, n, x)
}
pub fn partition(n: &Basic) -> PolyResult<Basic> {
    combinatorial_number(Function::Partition, std::slice::from_ref(n))
}

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
pub fn number_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::functions::special::gamma_functions::polygamma;
    let args: &[Basic] = e.args();
    match (f, argindex) {
        (Function::Catalan, _) => {
            let n: &Basic = &args[0];
            let psi = |x: Basic| polygamma(&Basic::zero(), &x);
            let difference: Basic = sub(
                &psi(add(&[n.clone(), Basic::half()])?)?,
                &psi(add(&[n.clone(), int(2)])?)?,
            )?;
            mul(&[
                catalan(n)?,
                add(&[difference, crate::functions::elementary::exponential::log(&int(4))?])?,
            ])
        }
        (Function::Harmonic, 1) => {
            let n: &Basic = &args[0];
            let m: Basic = args.get(1).cloned().unwrap_or_else(Basic::one);
            let z: Basic = crate::functions::special::zeta_functions::zeta(
                &add(&[m.clone(), Basic::one()])?,
                Some(&add(&[n.clone(), Basic::one()])?),
            )?;
            mul(&[m, z])
        }
        _ => generic_fdiff(e, argindex),
    }
}

/// The facts (`_eval_facts`).
pub fn number_facts(f: Function, args: &[Basic]) -> FactList {
    use Predicate as P;
    let mut facts: FactList = FactList::new();
    facts.set(P::commutative, true);
    let n: &Basic = &args[0];
    let a = facts_of(n);
    let yes = |p: P| a.get(&p) == Some(&true);
    match f {
        Function::Catalan => {
            if yes(P::integer) && yes(P::nonnegative) {
                facts.set(P::integer, true);
            }
            if yes(P::nonnegative) {
                facts.set(P::positive, true);
            }
            if yes(P::integer) && sub(n, &int(3)).is_ok_and(|d| is_true(&d, P::positive)) {
                facts.set(P::composite, true);
            }
        }
        Function::Genocchi => {
            if args.len() > 1 && args[1] != Basic::one() {
                return facts;
            }
            if is_true(n, P::integer) && is_true(n, P::nonnegative) {
                facts.set(P::integer, true);
                let half: Option<Basic> = div(n, &int(2)).ok();
                if is_true(n, P::odd) {
                    if let Some(v) = sub(n, &Basic::one()).ok().and_then(|d| is(&d, P::positive)) {
                        facts.set(P::negative, !v);
                        facts.set(P::odd, !v);
                        facts.set(P::even, v);
                    }
                    facts.set(P::positive, false);
                } else {
                    if is_true(n, P::zero) || is_true(n, P::odd) {
                        facts.set(P::positive, false);
                    } else if let Some(v) = half.as_ref().and_then(|h| is(h, P::even)) {
                        facts.set(P::positive, v);
                    }
                    if let Some(v) = half.as_ref().and_then(|h| is(h, P::odd)) {
                        facts.set(P::negative, v);
                    }
                    if is_true(n, P::even) {
                        if let Some(z) = is(n, P::zero) {
                            facts.set(P::even, z);
                            facts.set(P::odd, !z);
                        }
                    }
                }
            }
            if let Some(v) = sub(n, &int(8)).ok().and_then(|d| is(&d, P::zero)) {
                facts.set(P::prime, v);
            }
        }
        Function::Partition if is_true(n, P::nonnegative) => {
            facts.set(P::positive, true);
        }
        _ => {}
    }
    facts
}

fn c_int(n: i64, wp: usize) -> Complex {
    complex_real(real_int(&Int::from(n), wp), wp)
}

fn finish(value: &Complex, prec: usize) -> Complex {
    let (re, im) = parts(value);
    complex(re.with_precision(prec).value(), im.with_precision(prec).value())
}

/// mpmath's `fibonacci(z) = (phi**z - cos(pi*z)*phi**(-z))/sqrt(5)`.
fn fibonacci_value(z: &Complex, wp: usize) -> Complex {
    let five: Complex = c_int(5, wp);
    let root5: Complex = crate::core::evalf::complex_sqrt(&five, wp);
    let phi: Complex = &(&c_int(1, wp) + &root5) / &c_int(2, wp);
    let log_phi: Complex = phi.ln();
    let a: Complex = (z * &log_phi).exp();
    let b: Complex = (-(z * &log_phi)).exp();
    let pi: Complex = complex_real(Real::pi(wp), wp);
    let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
    let pz: Complex = &(&i * &pi) * z;
    let cos: Complex = &(&pz.exp() + &(-&pz).exp()) / &c_int(2, wp);
    &(&a - &(&cos * &b)) / &root5
}

fn is_integral(z: &Complex) -> Option<i64> {
    let (re, im) = parts(z);
    if im != Real::ZERO || re.clone() != re.floor() {
        return None;
    }
    re.to_int().value().try_into().ok()
}

/// The value (`numeric`).
pub fn number_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::{gamma, hurwitz_zeta, polygamma};
    let wp: usize = prec + 40;
    let z: &Complex = &values[0];
    let real: bool = parts(z).1 == Real::ZERO && values.iter().all(|v| parts(v).1 == Real::ZERO);
    let value: Complex = match f {
        Function::Catalan => {
            let half: Complex = &c_int(1, wp) / &c_int(2, wp);
            let four_n: Complex = (z * &c_int(4, wp).ln()).exp();
            let top: Complex = &four_n * &gamma(&(z + &half), wp)?;
            let bottom: Complex = &gamma(&half, wp)? * &gamma(&(z + &c_int(2, wp)), wp)?;
            &top / &bottom
        }
        Function::Fibonacci => {
            if values.len() != 1 {
                return None;
            }
            fibonacci_value(z, wp)
        }
        Function::Lucas => {
            let one: Complex = c_int(1, wp);
            &fibonacci_value(&(z + &one), wp) + &fibonacci_value(&(z - &one), wp)
        }
        Function::Harmonic => {
            let one: Complex = c_int(1, wp);
            let m: Complex = values.get(1).cloned().unwrap_or_else(|| one.clone());
            if is_integral(&m) == Some(1) {
                // mpmath's harmonic(n) = digamma(n + 1) + EulerGamma
                let g: Complex = complex_real(crate::core::evalf::euler_gamma(wp), wp);
                &polygamma(0, &(z + &one), wp)? + &g
            } else {
                &hurwitz_zeta(&m, &one, wp)? - &hurwitz_zeta(&m, &(z + &one), wp)?
            }
        }
        Function::Bernoulli | Function::Euler | Function::Genocchi => {
            // the integers are evaluated exactly; mpmath has no other values
            return None;
        }
        Function::Partition => {
            let k: i64 = is_integral(z)?;
            if k < 0 {
                return Some(c_int(0, prec));
            }
            let v: Int = partitions(k.min(20_000));
            complex_real(real_int(&v, wp), wp)
        }
        _ => return None,
    };
    let mut result: Complex = finish(&value, prec);
    if real && matches!(f, Function::Fibonacci | Function::Lucas | Function::Partition) {
        let (re, _) = parts(&result);
        result = complex(re, real_int(&Int::ZERO, prec));
    }
    Some(result)
}
