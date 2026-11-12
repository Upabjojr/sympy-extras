//! Port of the number-theoretic functions and the remaining numbers of
//! `sympy_extras/functions/combinatorial/numbers.py`: `totient`,
//! `reduced_totient`, `divisor_sigma`, `udivisor_sigma`, `primenu`,
//! `primeomega`, `mobius`, `primepi`, `legendre_symbol`, `jacobi_symbol`,
//! `kronecker_symbol`, `tribonacci` (and its polynomials), `motzkin` and
//! `andre`: the evaluation by the factorisation of `crate::ntheory`, the
//! errors of SymPy's checks and the facts. None has a numerical value of
//! its own, as in Python.

use std::collections::BTreeMap;

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_base_exp, is_false, is_true};
use crate::core::logic::truth;
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::relational::lt;
use crate::error::{not_ported, value_error, PolyError, PolyResult};
use crate::ntheory::factor_::factorint;
use crate::ntheory::generate::primerange;
use crate::ntheory::primetest::{jacobi_symbol, powmod};
use crate::numbers::{denom, numer, Int, Rat};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

fn as_int(e: &Basic) -> Option<Int> {
    let v = e.as_rational()?;
    if denom(v) == Int::ONE {
        Some(numer(v))
    } else {
        None
    }
}

fn type_error(message: &str) -> PolyError {
    PolyError::Type(message.to_string())
}

/// `_check_positive_integer`.
fn check_positive_integer(n: &Basic, name: &str) -> PolyResult<()> {
    if is_false(n, Predicate::integer) {
        return Err(type_error(&format!("{} should be an integer", name)));
    }
    if is_false(n, Predicate::positive) {
        return Err(value_error(format!("{} should be a positive integer", name)));
    }
    Ok(())
}

fn factors(n: &Int) -> BTreeMap<Int, u64> {
    factorint(n, None)
}

fn int_pow(base: &Int, e: u64) -> Int {
    let mut r: Int = Int::ONE;
    for _ in 0..e {
        r *= base;
    }
    r
}

fn gcd(a: &Int, b: &Int) -> Int {
    let (mut a, mut b) = (a.clone(), b.clone());
    while b != Int::ZERO {
        let r: Int = &a % &b;
        a = b;
        b = r;
    }
    if a < Int::ZERO {
        -a
    } else {
        a
    }
}

fn lcm(a: &Int, b: &Int) -> Int {
    a * b / gcd(a, b)
}

fn is_one(e: &Basic) -> bool {
    *e == Basic::one()
}

fn totient_eval(f: Function, n: &Basic) -> PolyResult<Basic> {
    check_positive_integer(n, "n")?;
    if is_one(n) {
        return Ok(Basic::one());
    }
    if is_true(n, Predicate::prime) {
        return sub(n, &Basic::one());
    }
    let Some(k) = as_int(n) else {
        return Ok(node(f, vec![n.clone()]));
    };
    if f == Function::Totient {
        let mut product: Int = Int::ONE;
        for (p, e) in factors(&k) {
            product *= int_pow(&p, e - 1) * (&p - Int::ONE);
        }
        return Ok(Basic::integer(product));
    }
    let mut k: Int = k;
    let mut twos: u64 = 0;
    while &k % Int::from(2) == Int::ZERO {
        k = &k / Int::from(2);
        twos += 1;
    }
    let t: Int = if twos == 0 {
        Int::ONE
    } else if twos > 2 {
        int_pow(&Int::from(2), twos - 2)
    } else {
        Int::from(twos)
    };
    let mut result: Int = t;
    for (p, e) in factors(&k) {
        result = lcm(&result, &((&p - Int::ONE) * int_pow(&p, e - 1)));
    }
    Ok(Basic::integer(result))
}

fn check_order(k: &Basic) -> PolyResult<()> {
    if is_false(k, Predicate::integer) {
        return Err(type_error("k should be an integer"));
    }
    if is_false(k, Predicate::nonnegative) {
        return Err(value_error("k should be a nonnegative integer".to_string()));
    }
    Ok(())
}

fn divisor_sigma_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    let k: Basic = args.get(1).cloned().unwrap_or_else(Basic::one);
    check_positive_integer(n, "n")?;
    check_order(&k)?;
    if is_true(n, Predicate::prime) {
        return add(&[Basic::one(), pow(n, &k)?]);
    }
    if is_one(n) {
        return Ok(Basic::one());
    }
    let Some(value) = as_int(n) else {
        return Ok(node(Function::DivisorSigma, args.to_vec()));
    };
    let fs: BTreeMap<Int, u64> = factors(&value);
    if is_true(&k, Predicate::zero) {
        let items: Vec<Basic> = fs.values().map(|e| int(*e as i64 + 1)).collect();
        return mul(&items);
    }
    if let Some(order) = as_int(&k) {
        let order: u64 = u64::try_from(&order).map_err(|_| not_ported("a huge divisor order"))?;
        if order > 10_000 {
            return Err(not_ported("a huge divisor order"));
        }
        let mut product: Int = Int::ONE;
        for (p, e) in &fs {
            product *= (int_pow(p, order * (e + 1)) - Int::ONE) / (int_pow(p, order) - Int::ONE);
        }
        return Ok(Basic::integer(product));
    }
    if is_false(&k, Predicate::zero) {
        let mut items: Vec<Basic> = Vec::new();
        for (p, e) in &fs {
            let mut terms: Vec<Basic> = Vec::new();
            for i in 0..=*e {
                terms.push(pow(&Basic::integer(p.clone()), &mul(&[k.clone(), int(i as i64)])?)?);
            }
            items.push(add(&terms)?);
        }
        return mul(&items);
    }
    Ok(node(Function::DivisorSigma, args.to_vec()))
}

fn udivisor_sigma_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    let k: Basic = args.get(1).cloned().unwrap_or_else(Basic::one);
    check_positive_integer(n, "n")?;
    check_order(&k)?;
    if is_true(n, Predicate::prime) {
        return add(&[Basic::one(), pow(n, &k)?]);
    }
    let Some(value) = as_int(n) else {
        return Ok(node(Function::UDivisorSigma, args.to_vec()));
    };
    let mut items: Vec<Basic> = Vec::new();
    for (p, e) in factors(&value) {
        items.push(add(&[
            Basic::one(),
            pow(&Basic::integer(p), &mul(&[k.clone(), int(e as i64)])?)?,
        ])?);
    }
    mul(&items)
}

fn prime_count_eval(f: Function, n: &Basic) -> PolyResult<Basic> {
    check_positive_integer(n, "n")?;
    if is_true(n, Predicate::prime) {
        return Ok(Basic::one());
    }
    if is_one(n) {
        return Ok(Basic::zero());
    }
    let Some(value) = as_int(n) else {
        return Ok(node(f, vec![n.clone()]));
    };
    let fs: BTreeMap<Int, u64> = factors(&value);
    Ok(if f == Function::PrimeNu {
        int(fs.len() as i64)
    } else {
        int(fs.values().sum::<u64>() as i64)
    })
}

fn mobius_eval(n: &Basic) -> PolyResult<Basic> {
    check_positive_integer(n, "n")?;
    if is_true(n, Predicate::prime) {
        return Ok(Basic::negative_one());
    }
    if is_one(n) {
        return Ok(Basic::one());
    }
    let factors_of: Vec<Basic> = if n.is_mul() { n.args().to_vec() } else { vec![n.clone()] };
    let mut result: Option<Basic> = None;
    for factor in factors_of {
        let (m, e) = as_base_exp(&factor)?;
        let fine: bool = is_true(&m, Predicate::integer)
            && is_true(&m, Predicate::positive)
            && is_true(&e, Predicate::integer)
            && is_true(&e, Predicate::positive);
        if !fine {
            return Ok(node(Function::Mobius, vec![n.clone()]));
        }
        let relation: Option<bool> = truth(&lt(&Basic::one(), &e)?);
        if relation == Some(true) {
            result = Some(Basic::zero());
            continue;
        }
        if let Some(value) = as_int(&m) {
            let fs: BTreeMap<Int, u64> = factors(&value);
            if fs.values().any(|v| *v > 1) {
                result = Some(Basic::zero());
            } else if relation == Some(false) {
                let sign: Basic = if fs.len() % 2 == 1 {
                    Basic::negative_one()
                } else {
                    Basic::one()
                };
                result = Some(match result {
                    None => sign,
                    Some(r) => mul(&[r, sign])?,
                });
            }
        }
    }
    Ok(result.unwrap_or_else(|| node(Function::Mobius, vec![n.clone()])))
}

fn primepi_eval(n: &Basic) -> PolyResult<Basic> {
    match n.head() {
        Head::Infinity => return Ok(Basic::oo()),
        Head::NegativeInfinity => return Ok(Basic::zero()),
        _ => {}
    }
    if is_false(n, Predicate::real) {
        return Err(type_error("n should be a real"));
    }
    if truth(&lt(n, &int(2))?) == Some(true) {
        return Ok(Basic::zero());
    }
    let Some(v) = n.as_rational() else {
        return Ok(node(Function::PrimePi, vec![n.clone()]));
    };
    // int(): the quotient truncated toward zero
    let bound: Int = numer(v) / denom(v);
    if bound > Int::from(10_000_000) {
        return Err(not_ported("primepi of a huge bound"));
    }
    let count: usize = primerange(&Int::from(2), &(bound + Int::ONE)).count();
    Ok(int(count as i64))
}

fn legendre(a: &Int, p: &Int) -> i64 {
    let r: Int = ((a % p) + p) % p;
    let value: Int = powmod(&r, &((p - Int::ONE) / Int::from(2)), p);
    if value == p - Int::ONE {
        -1
    } else {
        i64::try_from(&value).unwrap_or(0)
    }
}

fn kronecker(a: &Int, n: &Int) -> PolyResult<i64> {
    if *n == Int::ZERO {
        let abs: Int = if *a < Int::ZERO { -a.clone() } else { a.clone() };
        return Ok(if abs == Int::ONE { 1 } else { 0 });
    }
    let mut sign: i64 = 1;
    let mut n: Int = n.clone();
    if n < Int::ZERO {
        n = -n;
        if *a < Int::ZERO {
            sign = -1;
        }
    }
    let mut twos: u64 = 0;
    while &n % Int::from(2) == Int::ZERO {
        n = &n / Int::from(2);
        twos += 1;
    }
    if twos > 0 {
        if a % Int::from(2) == Int::ZERO {
            return Ok(0);
        }
        let r: Int = ((a % Int::from(8)) + Int::from(8)) % Int::from(8);
        if twos % 2 == 1 && (r == Int::from(3) || r == Int::from(5)) {
            sign = -sign;
        }
    }
    let j: i64 = if n > Int::ONE { jacobi_symbol(a, &n)? as i64 } else { 1 };
    Ok(sign * j)
}

fn mod_is_zero(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(is_true(&crate::core::modulo::modulo(a, b)?, Predicate::zero))
}

fn legendre_eval(a: &Basic, p: &Basic) -> PolyResult<Basic> {
    if is_false(a, Predicate::integer) {
        return Err(type_error("a should be an integer"));
    }
    if is_false(p, Predicate::integer) {
        return Err(type_error("p should be an integer"));
    }
    if is_false(p, Predicate::prime) || is_false(p, Predicate::odd) {
        return Err(value_error("p should be an odd prime integer".to_string()));
    }
    if mod_is_zero(a, p)? {
        return Ok(Basic::zero());
    }
    if is_one(a) {
        return Ok(Basic::one());
    }
    if let (Some(aa), Some(pp)) = (as_int(a), as_int(p)) {
        return Ok(int(legendre(&aa, &pp)));
    }
    Ok(node(Function::LegendreSymbol, vec![a.clone(), p.clone()]))
}

fn jacobi_eval(m: &Basic, n: &Basic) -> PolyResult<Basic> {
    if is_false(m, Predicate::integer) {
        return Err(type_error("m should be an integer"));
    }
    if is_false(n, Predicate::integer) {
        return Err(type_error("n should be an integer"));
    }
    if is_false(n, Predicate::positive) || is_false(n, Predicate::odd) {
        return Err(value_error("n should be an odd positive integer".to_string()));
    }
    if is_one(m) || is_one(n) {
        return Ok(Basic::one());
    }
    if mod_is_zero(m, n)? {
        return Ok(Basic::zero());
    }
    if let (Some(mm), Some(nn)) = (as_int(m), as_int(n)) {
        return Ok(int(jacobi_symbol(&mm, &nn)? as i64));
    }
    Ok(node(Function::JacobiSymbol, vec![m.clone(), n.clone()]))
}

fn kronecker_eval(a: &Basic, n: &Basic) -> PolyResult<Basic> {
    if is_false(a, Predicate::integer) {
        return Err(type_error("a should be an integer"));
    }
    if is_false(n, Predicate::integer) {
        return Err(type_error("n should be an integer"));
    }
    if is_one(a) || is_one(n) {
        return Ok(Basic::one());
    }
    if let (Some(aa), Some(nn)) = (as_int(a), as_int(n)) {
        return Ok(int(kronecker(&aa, &nn)?));
    }
    Ok(node(Function::KroneckerSymbol, vec![a.clone(), n.clone()]))
}

fn small(n: &Basic) -> PolyResult<Option<i64>> {
    match as_int(n) {
        None => Ok(None),
        Some(v) => {
            let k: i64 = i64::try_from(&v).map_err(|_| not_ported("a huge index"))?;
            if k.abs() > 20_000 {
                return Err(not_ported("a huge index"));
            }
            Ok(Some(k))
        }
    }
}

fn tribonacci_eval(args: &[Basic]) -> PolyResult<Basic> {
    let n: &Basic = &args[0];
    if matches!(n.head(), Head::Infinity) {
        return Ok(Basic::oo());
    }
    let Some(k) = small(n)? else {
        return Ok(node(Function::Tribonacci, args.to_vec()));
    };
    if k < 0 {
        return Err(value_error(
            "Tribonacci polynomials are defined only for non-negative integer indices.".to_string(),
        ));
    }
    if args.len() == 1 {
        let (mut a, mut b, mut c) = (Int::ZERO, Int::ONE, Int::ONE);
        for _ in 0..k {
            let next: Int = &a + &b + &c;
            a = b;
            b = c;
            c = next;
        }
        return Ok(Basic::integer(a));
    }
    let x: &Basic = &args[1];
    let mut polys: Vec<Vec<Int>> = vec![vec![Int::ZERO], vec![Int::ONE], vec![Int::ZERO, Int::ZERO, Int::ONE]];
    for _ in 3..=k {
        let len: usize = polys.len();
        let (p1, p2, p3) = (&polys[len - 1], &polys[len - 2], &polys[len - 3]);
        let size: usize = (p1.len() + 2).max(p2.len() + 1).max(p3.len());
        let mut new: Vec<Int> = vec![Int::ZERO; size];
        for (i, c) in p1.iter().enumerate() {
            new[i + 2] += c;
        }
        for (i, c) in p2.iter().enumerate() {
            new[i + 1] += c;
        }
        for (i, c) in p3.iter().enumerate() {
            new[i] += c;
        }
        polys.push(new);
    }
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in polys[k as usize].iter().enumerate() {
        if *c != Int::ZERO {
            terms.push(mul(&[Basic::integer(c.clone()), pow(x, &int(i as i64))?])?);
        }
    }
    add(&terms)
}

fn motzkin_number(n: i64) -> Int {
    let mut values: Vec<Int> = vec![Int::ONE, Int::ONE];
    for k in 2..=n {
        let k_u: usize = k as usize;
        let next: Int =
            (Int::from(2 * k + 1) * &values[k_u - 1] + Int::from(3 * k - 3) * &values[k_u - 2]) / Int::from(k + 2);
        values.push(next);
    }
    values[n as usize].clone()
}

fn motzkin_eval(n: &Basic) -> PolyResult<Basic> {
    let k: Option<i64> = small(n)?;
    match k {
        Some(k) if k >= 0 => Ok(Basic::integer(motzkin_number((k - 1).max(0)))),
        _ => Err(value_error(
            "The provided number must be a positive integer".to_string(),
        )),
    }
}

fn andre_eval(n: &Basic) -> PolyResult<Basic> {
    use crate::functions::special::zeta_functions::zeta;
    match n.head() {
        Head::NaN => return Ok(Basic::nan()),
        Head::Infinity => return Ok(Basic::oo()),
        _ => {}
    }
    if is_true(n, Predicate::zero) {
        return Ok(Basic::one());
    }
    let k: Option<i64> = small(n)?;
    if k == Some(-1) {
        return mul(&[
            Basic::negative_one(),
            crate::functions::elementary::exponential::log(&int(2))?,
        ]);
    }
    if k == Some(-2) {
        return mul(&[int(-2), Basic::catalan()]);
    }
    if let Some(k) = k {
        if k >= 0 && k % 2 == 0 {
            let e: Basic = super::numbers::euler(n, None)?;
            return crate::functions::elementary::complexes::abs_(&e);
        }
        if k % 2 != 0 {
            let m: i64 = -k - 1;
            let two_m: Rat = rat_pow(&Rat::from(Int::from(2)), m);
            let four_m: Rat = rat_pow(&Rat::from(Int::from(4)), m);
            let ratio: Rat = (Rat::ONE - two_m) / four_m;
            let number: Basic = if denom(&ratio) == Int::ONE {
                Basic::integer(numer(&ratio))
            } else {
                Basic::rational(numer(&ratio), denom(&ratio))?
            };
            return mul(&[
                pow(&Basic::imaginary_unit(), &int(m))?,
                number,
                zeta(&mul(&[Basic::negative_one(), n.clone()])?, None)?,
            ]);
        }
    }
    Ok(node(Function::Andre, vec![n.clone()]))
}

fn rat_pow(base: &Rat, e: i64) -> Rat {
    let mut r: Rat = Rat::ONE;
    for _ in 0..e.unsigned_abs() {
        r *= base;
    }
    if e < 0 {
        Rat::ONE / r
    } else {
        r
    }
}

/// The heads of the module.
pub fn is_number_theoretic(f: Function) -> bool {
    matches!(
        f,
        Function::Totient
            | Function::ReducedTotient
            | Function::DivisorSigma
            | Function::UDivisorSigma
            | Function::PrimeNu
            | Function::PrimeOmega
            | Function::Mobius
            | Function::PrimePi
            | Function::LegendreSymbol
            | Function::JacobiSymbol
            | Function::KroneckerSymbol
            | Function::Tribonacci
            | Function::Motzkin
            | Function::Andre
    )
}

/// The evaluating function of the heads of the module.
pub fn number_theoretic(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::Totient | Function::ReducedTotient => totient_eval(f, &args[0]),
        Function::DivisorSigma => divisor_sigma_eval(args),
        Function::UDivisorSigma => udivisor_sigma_eval(args),
        Function::PrimeNu | Function::PrimeOmega => prime_count_eval(f, &args[0]),
        Function::Mobius => mobius_eval(&args[0]),
        Function::PrimePi => primepi_eval(&args[0]),
        Function::LegendreSymbol => legendre_eval(&args[0], &args[1]),
        Function::JacobiSymbol => jacobi_eval(&args[0], &args[1]),
        Function::KroneckerSymbol => kronecker_eval(&args[0], &args[1]),
        Function::Tribonacci => tribonacci_eval(args),
        Function::Motzkin => motzkin_eval(&args[0]),
        _ => andre_eval(&args[0]),
    }
}

/// The facts (`_eval_facts`).
pub fn number_theoretic_facts(f: Function, args: &[Basic]) -> FactList {
    use Predicate as P;
    let mut facts: FactList = FactList::new();
    match f {
        Function::Tribonacci | Function::Motzkin => {
            facts.set(P::commutative, true);
        }
        Function::Andre => {
            facts.set(P::commutative, true);
            let n: &Basic = &args[0];
            if is_true(n, P::integer) && is_true(n, P::nonnegative) {
                facts.set(P::integer, true);
            }
            if is_true(n, P::nonnegative) {
                facts.set(P::positive, true);
            }
        }
        _ => {
            for p in [
                P::commutative,
                P::integer,
                P::rational,
                P::real,
                P::extended_real,
                P::finite,
                P::complex,
                P::algebraic,
            ] {
                facts.set(p, true);
            }
            match f {
                Function::Totient | Function::ReducedTotient | Function::DivisorSigma | Function::UDivisorSigma => {
                    facts.set(P::positive, true);
                }
                Function::PrimeNu | Function::PrimeOmega | Function::PrimePi => {
                    facts.set(P::nonnegative, true);
                }
                _ => {}
            }
        }
    }
    facts
}
