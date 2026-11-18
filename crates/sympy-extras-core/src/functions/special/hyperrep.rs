//! The representing functions of hyperexpand's formula tables, ported
//! from `sympy_extras.functions.special.hyperrep`: the heads
//! `HyperRep_power1`, `HyperRep_power2`, `HyperRep_log1`,
//! `HyperRep_atanh`, `HyperRep_asin1`, `HyperRep_asin2`,
//! `HyperRep_sqrts1`, `HyperRep_sqrts2`, `HyperRep_log2`,
//! `HyperRep_cosasin`, `HyperRep_sinasin` (the parameters unpolarified,
//! commutative, no values) and [`rewrite_nonrep`] (SymPy's
//! `rewrite('nonrepsmall')` and `rewrite('nonrep')`: the small-argument
//! closed form, or a `Piecewise` of the big-argument form for `|z| > 1`
//! and the small one).
//!
//! The core has no polar numbers: the branch number of the argument is
//! always `0`, so only the `_small` forms and the `_big` forms at `n = 0`
//! are reached (the `_minus` forms need `exp_polar(I*pi)`). The forms are
//! built with Python's operators in their order (`a*b/c` left to right).

// PORTED FROM SYMPY: sympy/functions/special/hyper.py (HyperRep and its
// subclasses) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is_true;
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::PolyResult;
use crate::numbers::Int;

/// The `HyperRep` heads.
pub fn is_hyperrep(f: Function) -> bool {
    matches!(
        f,
        Function::HyperRepPower1
            | Function::HyperRepPower2
            | Function::HyperRepLog1
            | Function::HyperRepAtanh
            | Function::HyperRepAsin1
            | Function::HyperRepAsin2
            | Function::HyperRepSqrts1
            | Function::HyperRepSqrts2
            | Function::HyperRepLog2
            | Function::HyperRepCosasin
            | Function::HyperRepSinasin
    )
}

/// The number of arguments of a `HyperRep` head.
pub fn hyperrep_arity(f: Function) -> usize {
    match f {
        Function::HyperRepPower1
        | Function::HyperRepPower2
        | Function::HyperRepSqrts1
        | Function::HyperRepSqrts2
        | Function::HyperRepCosasin
        | Function::HyperRepSinasin => 2,
        _ => 1,
    }
}

/// The evaluating function (`HyperRep.eval`): the parameters unpolarified
/// (the core has no polar numbers: a reevaluation).
pub fn hyperrep(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    let node = |args: Vec<Basic>| Basic::new(Head::Function(f), args);
    if !evaluate() {
        return Ok(node(args.to_vec()));
    }
    let mut newargs: Vec<Basic> = Vec::new();
    for a in &args[..args.len() - 1] {
        newargs.push(crate::core::function::unpolarify(a)?);
    }
    newargs.push(args[args.len() - 1].clone());
    Ok(node(newargs))
}

pub fn hyperrep_facts() -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    facts
}

// Python's operators, one binary operation each
fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}
fn m2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), b.clone()])
}
fn sqrt(x: &Basic) -> PolyResult<Basic> {
    pow(x, &Basic::half())
}
fn pi() -> Basic {
    Basic::pi()
}
fn i() -> Basic {
    Basic::imaginary_unit()
}
fn exp(x: &Basic) -> PolyResult<Basic> {
    crate::functions::elementary::exponential::exp(x)
}
fn log(x: &Basic) -> PolyResult<Basic> {
    crate::functions::elementary::exponential::log(x)
}

/// The closed form for small arguments (`_small`).
fn small(f: Function, p: &[Basic], x: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::inverse::{asin, atanh};
    use crate::functions::elementary::trigonometric::{cos, sin};
    let one: Basic = Basic::one();
    let two: Basic = int(2);
    Ok(match f {
        // (1 - x)**a
        Function::HyperRepPower1 => pow(&sub(&one, x)?, &p[0])?,
        // 2**(2*a - 1) * (1 + sqrt(1 - x))**(1 - 2*a)
        Function::HyperRepPower2 => {
            let a: &Basic = &p[0];
            let first: Basic = pow(&two, &sub(&m2(&two, a)?, &one)?)?;
            let second: Basic = pow(&a2(&one, &sqrt(&sub(&one, x)?)?)?, &sub(&one, &m2(&two, a)?)?)?;
            m2(&first, &second)?
        }
        // log(1 - x)
        Function::HyperRepLog1 => log(&sub(&one, x)?)?,
        // atanh(sqrt(x)) / sqrt(x)
        Function::HyperRepAtanh => div(&atanh(&sqrt(x)?)?, &sqrt(x)?)?,
        // asin(sqrt(x)) / sqrt(x)
        Function::HyperRepAsin1 => div(&asin(&sqrt(x)?)?, &sqrt(x)?)?,
        Function::HyperRepAsin2 => div(
            &small(Function::HyperRepAsin1, &[], x)?,
            &small(Function::HyperRepPower1, &[Basic::half()], x)?,
        )?,
        // ((1 - sqrt(x))**(2*a) + (1 + sqrt(x))**(2*a)) / 2
        Function::HyperRepSqrts1 => {
            let a2a: Basic = m2(&two, &p[0])?;
            let s: Basic = sqrt(x)?;
            let sum: Basic = a2(&pow(&sub(&one, &s)?, &a2a)?, &pow(&a2(&one, &s)?, &a2a)?)?;
            div(&sum, &two)?
        }
        // sqrt(x) * ((1 - sqrt(x))**(2*a) - (1 + sqrt(x))**(2*a)) / 2
        Function::HyperRepSqrts2 => {
            let a2a: Basic = m2(&two, &p[0])?;
            let s: Basic = sqrt(x)?;
            let difference: Basic = sub(&pow(&sub(&one, &s)?, &a2a)?, &pow(&a2(&one, &s)?, &a2a)?)?;
            div(&m2(&s, &difference)?, &two)?
        }
        // log(HALF + sqrt(1 - x)/2)
        Function::HyperRepLog2 => log(&a2(&Basic::half(), &div(&sqrt(&sub(&one, x)?)?, &two)?)?)?,
        // cos(2*p[0]*asin(sqrt(x)))
        Function::HyperRepCosasin => cos(&m2(&m2(&two, &p[0])?, &asin(&sqrt(x)?)?)?)?,
        // sqrt(x)/sqrt(1 - x) * sin(2*p[0]*asin(sqrt(x)))
        Function::HyperRepSinasin => {
            let ratio: Basic = div(&sqrt(x)?, &sqrt(&sub(&one, x)?)?)?;
            m2(&ratio, &sin(&m2(&m2(&two, &p[0])?, &asin(&sqrt(x)?)?)?)?)?
        }
        _ => unreachable!("a HyperRep head"),
    })
}

/// The closed form for big arguments on the branch `n = 0` (`_big`).
fn big(f: Function, p: &[Basic], x: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::hyperbolic::{cosh, sinh};
    use crate::functions::elementary::inverse::{acosh, acoth, asin};
    let one: Basic = Basic::one();
    let two: Basic = int(2);
    let n: Basic = Basic::zero();
    // 2*n - 1
    let odd_shift: Basic = sub(&m2(&two, &n)?, &one)?;
    Ok(match f {
        Function::HyperRepPower1 => {
            let a: &Basic = &p[0];
            if is_true(a, Predicate::integer) {
                return small(f, p, x);
            }
            // (x - 1)**a * exp((2*n - 1)*pi*I*a)
            let phase: Basic = exp(&m2(&m2(&m2(&odd_shift, &pi())?, &i())?, a)?)?;
            m2(&pow(&sub(x, &one)?, a)?, &phase)?
        }
        Function::HyperRepPower2 => {
            // n even: sgn = -1;
            // 2**(2*a - 1) * (1 + sgn*I*sqrt(x - 1))**(1 - 2*a) * exp(-2*n*pi*I*a)
            let a: &Basic = &p[0];
            let first: Basic = pow(&two, &sub(&m2(&two, a)?, &one)?)?;
            let inner: Basic = a2(&one, &m2(&m2(&int(-1), &i())?, &sqrt(&sub(x, &one)?)?)?)?;
            let second: Basic = pow(&inner, &sub(&one, &m2(&two, a)?)?)?;
            let phase: Basic = exp(&m2(&m2(&m2(&m2(&int(-2), &n)?, &pi())?, &i())?, a)?)?;
            m2(&m2(&first, &second)?, &phase)?
        }
        // log(x - 1) + (2*n - 1)*pi*I
        Function::HyperRepLog1 => a2(&log(&sub(x, &one)?)?, &m2(&m2(&odd_shift, &pi())?, &i())?)?,
        // n even: (acoth(sqrt(x)) + I*pi/2) / sqrt(x)
        Function::HyperRepAtanh => {
            let top: Basic = a2(&acoth(&sqrt(x)?)?, &div(&m2(&i(), &pi())?, &two)?)?;
            div(&top, &sqrt(x)?)?
        }
        // (-1)**n * ((HALF - n)*pi/sqrt(x) + I*acosh(sqrt(x))/sqrt(x))
        Function::HyperRepAsin1 => {
            let first: Basic = div(&m2(&sub(&Basic::half(), &n)?, &pi())?, &sqrt(x)?)?;
            let second: Basic = div(&m2(&i(), &acosh(&sqrt(x)?)?)?, &sqrt(x)?)?;
            m2(&pow(&int(-1), &n)?, &a2(&first, &second)?)?
        }
        Function::HyperRepAsin2 => div(
            &big(Function::HyperRepAsin1, &[], x)?,
            &big(Function::HyperRepPower1, &[Basic::half()], x)?,
        )?,
        // n even: ((sqrt(x) + 1)**(2*a)*exp(2*pi*I*n*a)
        //          + (sqrt(x) - 1)**(2*a)*exp(2*pi*I*(n - 1)*a)) / 2
        Function::HyperRepSqrts1 => {
            let a: &Basic = &p[0];
            let a2a: Basic = m2(&two, a)?;
            let s: Basic = sqrt(x)?;
            let two_pi_i: Basic = m2(&m2(&two, &pi())?, &i())?;
            let first: Basic = m2(&pow(&a2(&s, &one)?, &a2a)?, &exp(&m2(&m2(&two_pi_i, &n)?, a)?)?)?;
            let second: Basic = m2(
                &pow(&sub(&s, &one)?, &a2a)?,
                &exp(&m2(&m2(&two_pi_i, &sub(&n, &one)?)?, a)?)?,
            )?;
            div(&a2(&first, &second)?, &two)?
        }
        // n even: sqrt(x)/2 * ((sqrt(x) - 1)**(2*a)*exp(2*pi*I*a*(n - 1))
        //                      - (sqrt(x) + 1)**(2*a)*exp(2*pi*I*a*n))
        Function::HyperRepSqrts2 => {
            let a: &Basic = &p[0];
            let a2a: Basic = m2(&two, a)?;
            let s: Basic = sqrt(x)?;
            let two_pi_i_a: Basic = m2(&m2(&m2(&two, &pi())?, &i())?, a)?;
            let first: Basic = m2(&pow(&sub(&s, &one)?, &a2a)?, &exp(&m2(&two_pi_i_a, &sub(&n, &one)?)?)?)?;
            let second: Basic = m2(&pow(&a2(&s, &one)?, &a2a)?, &exp(&m2(&two_pi_i_a, &n)?)?)?;
            m2(&div(&s, &two)?, &sub(&first, &second)?)?
        }
        // n even: (n - HALF)*pi*I + log(sqrt(x)/2) + I*asin(1/sqrt(x))
        Function::HyperRepLog2 => {
            let first: Basic = m2(&m2(&sub(&n, &Basic::half())?, &pi())?, &i())?;
            let second: Basic = log(&div(&sqrt(x)?, &two)?)?;
            let third: Basic = m2(&i(), &asin(&div(&one, &sqrt(x)?)?)?)?;
            a2(&a2(&first, &second)?, &third)?
        }
        // cosh(2*a*acosh(sqrt(x)) + a*pi*I*(2*n - 1))
        Function::HyperRepCosasin => {
            let a: &Basic = &p[0];
            let first: Basic = m2(&m2(&two, a)?, &acosh(&sqrt(x)?)?)?;
            let second: Basic = m2(&m2(&m2(a, &pi())?, &i())?, &odd_shift)?;
            cosh(&a2(&first, &second)?)?
        }
        // -1/sqrt(1 - 1/x) * sinh(2*a*acosh(sqrt(x)) + a*pi*I*(2*n - 1))
        Function::HyperRepSinasin => {
            let a: &Basic = &p[0];
            let factor: Basic = div(&int(-1), &sqrt(&sub(&one, &div(&one, x)?)?)?)?;
            let first: Basic = m2(&m2(&two, a)?, &acosh(&sqrt(x)?)?)?;
            let second: Basic = m2(&m2(&m2(a, &pi())?, &i())?, &odd_shift)?;
            m2(&factor, &sinh(&a2(&first, &second)?)?)?
        }
        _ => unreachable!("a HyperRep head"),
    })
}

/// The closed form of a `HyperRep` node: for small arguments
/// (`nonrepsmall`), or on every branch (`nonrep`: the big form for `|x| >
/// 1`).
fn closed_form(f: Function, e: &Basic, small_only: bool) -> PolyResult<Basic> {
    use crate::core::relational::gt;
    use crate::functions::elementary::complexes::abs_;
    let args: &[Basic] = e.args();
    let (p, x) = (&args[..args.len() - 1], &args[args.len() - 1]);
    let s: Basic = small(f, p, x)?;
    if small_only {
        return Ok(s);
    }
    let b: Basic = big(f, p, x)?;
    if b == s {
        return Ok(s);
    }
    crate::functions::elementary::piecewise::piecewise(&[
        (b, gt(&abs_(x)?, &Basic::one())?),
        (s, crate::core::logic::true_()),
    ])
}

/// `e` with its `HyperRep` nodes replaced by their closed forms (SymPy's
/// `rewrite('nonrepsmall')`, or `rewrite('nonrep')` when `small` is
/// false), bottom up.
pub fn rewrite_nonrep(e: &Basic, small_only: bool) -> PolyResult<Basic> {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(rewrite_nonrep(a, small_only)?);
    }
    let rebuilt: Basic = if args.as_slice() != e.args() {
        crate::core::function::rebuild_evaluated(e, args)?
    } else {
        e.clone()
    };
    match rebuilt.head() {
        Head::Function(f) if is_hyperrep(*f) => closed_form(*f, &rebuilt, small_only),
        _ => Ok(rebuilt),
    }
}
