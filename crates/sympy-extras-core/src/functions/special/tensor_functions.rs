//! Port of `sympy_extras/functions/special/tensor_functions.py`: the
//! Kronecker delta [`Function::KroneckerDelta`] and the Levi-Civita symbol
//! [`Function::LeviCivita`]: evaluation (the delta decided by the
//! difference of its indices, ordered by the canonical key; the symbol of
//! integers by `prod(prod(a_j - a_i, j > i)/i!)`, zero for a repeated
//! index), the facts of an integer, the powers of a delta.

use crate::core::add::sub;
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{is, is_true};
use crate::core::parameters::evaluate;
use crate::core::sorting::sort_key;
use crate::error::PolyResult;
use crate::numbers::{denom, numer, Int, Rat};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

/// `KroneckerDelta.eval`.
fn kronecker_eval(i: &Basic, j: &Basic) -> PolyResult<Basic> {
    match is(&sub(i, j)?, Predicate::zero) {
        Some(true) => return Ok(Basic::one()),
        Some(false) => return Ok(Basic::zero()),
        None => {}
    }
    if sort_key(j) < sort_key(i) {
        return kronecker_delta(j, i);
    }
    Ok(node(Function::KroneckerDelta, vec![i.clone(), j.clone()]))
}

/// The Levi-Civita symbol of integers (`eval_levicivita`).
pub fn eval_levicivita(args: &[Int]) -> PolyResult<Basic> {
    let mut value: Rat = Rat::ONE;
    let mut factorial: Int = Int::ONE;
    for i in 0..args.len() {
        if i > 0 {
            factorial *= Int::from(i);
        }
        let mut inner: Int = Int::ONE;
        for j in i + 1..args.len() {
            inner *= &args[j] - &args[i];
        }
        value *= Rat::from(inner) / Rat::from(factorial.clone());
    }
    if denom(&value) == Int::ONE {
        return Ok(Basic::integer(numer(&value)));
    }
    Basic::rational(numer(&value), denom(&value))
}

fn levicivita_eval(args: &[Basic]) -> PolyResult<Basic> {
    let integers: Vec<Int> = args
        .iter()
        .filter_map(|a| a.as_rational().filter(|v| denom(v) == Int::ONE).map(numer))
        .collect();
    if integers.len() == args.len() {
        return eval_levicivita(&integers);
    }
    let mut distinct: Vec<&Basic> = Vec::new();
    for a in args {
        if !distinct.contains(&a) {
            distinct.push(a);
        }
    }
    if distinct.len() != args.len() {
        return Ok(Basic::zero());
    }
    Ok(node(Function::LeviCivita, args.to_vec()))
}

/// The evaluating function of the two heads.
pub fn tensor_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::KroneckerDelta => kronecker_eval(&args[0], &args[1]),
        _ => levicivita_eval(args),
    }
}

pub fn kronecker_delta(i: &Basic, j: &Basic) -> PolyResult<Basic> {
    tensor_function(Function::KroneckerDelta, &[i.clone(), j.clone()])
}

pub fn levi_civita(args: &[Basic]) -> PolyResult<Basic> {
    tensor_function(Function::LeviCivita, args)
}

/// The facts of an integer (`_INTEGER_FACTS`).
pub fn tensor_facts() -> FactList {
    use Predicate as P;
    let mut facts: FactList = FactList::new();
    for (p, v) in [
        (P::commutative, true),
        (P::integer, true),
        (P::rational, true),
        (P::real, true),
        (P::extended_real, true),
        (P::finite, true),
        (P::complex, true),
        (P::hermitian, true),
        (P::algebraic, true),
        (P::imaginary, false),
        (P::infinite, false),
        (P::noninteger, false),
        (P::irrational, false),
        (P::transcendental, false),
    ] {
        facts.set(p, v);
    }
    facts
}

/// A power of a delta (`_eval_power`): the delta for a positive exponent,
/// `1/delta` for a negative one other than `-1`.
pub fn kronecker_power(base: &Basic, exponent: &Basic) -> PolyResult<Option<Basic>> {
    if is_true(exponent, Predicate::positive) {
        return Ok(Some(base.clone()));
    }
    if is_true(exponent, Predicate::negative) && *exponent != Basic::negative_one() {
        return Ok(Some(crate::core::mul::div(&Basic::one(), base)?));
    }
    Ok(None)
}
