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
//! The branch number of the argument comes from `extract_branch_factor`:
//! the `_small`/`_big` forms on the integer branches, the `_minus` forms
//! on the half-integer ones (`exp_polar(I*pi)`). The forms are built with
//! Python's operators in their order (`a*b/c` left to right).

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

/// The closed form for small arguments of `-x` on the branch of
/// `exp_polar(I*pi)` (`_small_minus`).
fn small_minus(f: Function, p: &[Basic], x: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::hyperbolic::{cosh, sinh};
    use crate::functions::elementary::inverse::{asinh, atan};
    use crate::functions::elementary::trigonometric::{cos, sin};
    let one: Basic = Basic::one();
    let two: Basic = int(2);
    Ok(match f {
        // (1 + x)**a
        Function::HyperRepPower1 => pow(&a2(&one, x)?, &p[0])?,
        // 2**(2*a - 1) * (1 + sqrt(1 + x))**(1 - 2*a)
        Function::HyperRepPower2 => {
            let a: &Basic = &p[0];
            let first: Basic = pow(&two, &sub(&m2(&two, a)?, &one)?)?;
            let second: Basic = pow(&a2(&one, &sqrt(&a2(&one, x)?)?)?, &sub(&one, &m2(&two, a)?)?)?;
            m2(&first, &second)?
        }
        Function::HyperRepLog1 => log(&a2(&one, x)?)?,
        Function::HyperRepAtanh => div(&atan(&sqrt(x)?)?, &sqrt(x)?)?,
        Function::HyperRepAsin1 => div(&asinh(&sqrt(x)?)?, &sqrt(x)?)?,
        Function::HyperRepAsin2 => div(
            &small_minus(Function::HyperRepAsin1, &[], x)?,
            &small_minus(Function::HyperRepPower1, &[Basic::half()], x)?,
        )?,
        // (1 + x)**a * cos(2*a*atan(sqrt(x)))
        Function::HyperRepSqrts1 => {
            let a: &Basic = &p[0];
            m2(&pow(&a2(&one, x)?, a)?, &cos(&m2(&m2(&two, a)?, &atan(&sqrt(x)?)?)?)?)?
        }
        // sqrt(x) * (1 + x)**a * sin(2*a*atan(sqrt(x)))
        Function::HyperRepSqrts2 => {
            let a: &Basic = &p[0];
            let first: Basic = m2(&sqrt(x)?, &pow(&a2(&one, x)?, a)?)?;
            m2(&first, &sin(&m2(&m2(&two, a)?, &atan(&sqrt(x)?)?)?)?)?
        }
        Function::HyperRepLog2 => log(&a2(&Basic::half(), &div(&sqrt(&a2(&one, x)?)?, &two)?)?)?,
        Function::HyperRepCosasin => cosh(&m2(&m2(&two, &p[0])?, &asinh(&sqrt(x)?)?)?)?,
        // -sqrt(x)/sqrt(1 + x) * sinh(2*p[0]*asinh(sqrt(x)))
        Function::HyperRepSinasin => {
            let ratio: Basic = div(&mul(&[int(-1), sqrt(x)?])?, &sqrt(&a2(&one, x)?)?)?;
            m2(&ratio, &sinh(&m2(&m2(&two, &p[0])?, &asinh(&sqrt(x)?)?)?)?)?
        }
        _ => unreachable!("a HyperRep head"),
    })
}

fn is_even(n: &Basic) -> bool {
    is_true(n, Predicate::even)
}

/// The closed form for big arguments on the branch `n` (`_big`).
fn big(f: Function, p: &[Basic], x: &Basic, n: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::hyperbolic::{cosh, sinh};
    use crate::functions::elementary::inverse::{acosh, acoth, asin};
    let one: Basic = Basic::one();
    let two: Basic = int(2);
    // 2*n - 1
    let odd_shift: Basic = sub(&m2(&two, n)?, &one)?;
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
            // 2**(2*a - 1) * (1 + sgn*I*sqrt(x - 1))**(1 - 2*a) * exp(-2*n*pi*I*a)
            let a: &Basic = &p[0];
            let (sgn, n): (i64, Basic) = if is_true(n, Predicate::odd) {
                (1, sub(n, &one)?)
            } else {
                (-1, n.clone())
            };
            let first: Basic = pow(&two, &sub(&m2(&two, a)?, &one)?)?;
            let inner: Basic = a2(&one, &m2(&m2(&int(sgn), &i())?, &sqrt(&sub(x, &one)?)?)?)?;
            let second: Basic = pow(&inner, &sub(&one, &m2(&two, a)?)?)?;
            let phase: Basic = exp(&m2(&m2(&m2(&m2(&int(-2), &n)?, &pi())?, &i())?, a)?)?;
            m2(&m2(&first, &second)?, &phase)?
        }
        // log(x - 1) + (2*n - 1)*pi*I
        Function::HyperRepLog1 => a2(&log(&sub(x, &one)?)?, &m2(&m2(&odd_shift, &pi())?, &i())?)?,
        // (acoth(sqrt(x)) +- I*pi/2) / sqrt(x)
        Function::HyperRepAtanh => {
            let shift: Basic = div(&m2(&i(), &pi())?, &two)?;
            let top: Basic = if is_even(n) {
                a2(&acoth(&sqrt(x)?)?, &shift)?
            } else {
                sub(&acoth(&sqrt(x)?)?, &shift)?
            };
            div(&top, &sqrt(x)?)?
        }
        // (-1)**n * ((HALF - n)*pi/sqrt(x) + I*acosh(sqrt(x))/sqrt(x))
        Function::HyperRepAsin1 => {
            let first: Basic = div(&m2(&sub(&Basic::half(), n)?, &pi())?, &sqrt(x)?)?;
            let second: Basic = div(&m2(&i(), &acosh(&sqrt(x)?)?)?, &sqrt(x)?)?;
            m2(&pow(&int(-1), n)?, &a2(&first, &second)?)?
        }
        Function::HyperRepAsin2 => div(
            &big(Function::HyperRepAsin1, &[], x, n)?,
            &big(Function::HyperRepPower1, &[Basic::half()], x, n)?,
        )?,
        Function::HyperRepSqrts1 => {
            let a: &Basic = &p[0];
            let a2a: Basic = m2(&two, a)?;
            let s: Basic = sqrt(x)?;
            if is_even(n) {
                // ((sqrt(x) + 1)**(2*a)*exp(2*pi*I*n*a) + (sqrt(x) - 1)**(2*a)*exp(2*pi*I*(n - 1)*a)) / 2
                let two_pi_i: Basic = m2(&m2(&two, &pi())?, &i())?;
                let first: Basic = m2(&pow(&a2(&s, &one)?, &a2a)?, &exp(&m2(&m2(&two_pi_i, n)?, a)?)?)?;
                let second: Basic = m2(
                    &pow(&sub(&s, &one)?, &a2a)?,
                    &exp(&m2(&m2(&two_pi_i, &sub(n, &one)?)?, a)?)?,
                )?;
                div(&a2(&first, &second)?, &two)?
            } else {
                // n = n - 1: ((sqrt(x) - 1)**(2*a)*exp(2*pi*I*a*(n + 1)) + (sqrt(x) + 1)**(2*a)*exp(2*pi*I*a*n)) / 2
                let n: Basic = sub(n, &one)?;
                let two_pi_i_a: Basic = m2(&m2(&m2(&two, &pi())?, &i())?, a)?;
                let first: Basic = m2(&pow(&sub(&s, &one)?, &a2a)?, &exp(&m2(&two_pi_i_a, &a2(&n, &one)?)?)?)?;
                let second: Basic = m2(&pow(&a2(&s, &one)?, &a2a)?, &exp(&m2(&two_pi_i_a, &n)?)?)?;
                div(&a2(&first, &second)?, &two)?
            }
        }
        Function::HyperRepSqrts2 => {
            // sqrt(x)/2 * ((sqrt(x) - 1)**(2*a)*exp(2*pi*I*a*(n - 1)) - (sqrt(x) + 1)**(2*a)*exp(2*pi*I*a*n)),
            // with n - 1 for n and (n + 1) for (n - 1) on an odd branch
            let a: &Basic = &p[0];
            let a2a: Basic = m2(&two, a)?;
            let s: Basic = sqrt(x)?;
            let two_pi_i_a: Basic = m2(&m2(&m2(&two, &pi())?, &i())?, a)?;
            let (low, high): (Basic, Basic) = if is_even(n) {
                (sub(n, &one)?, n.clone())
            } else {
                let m: Basic = sub(n, &one)?;
                (a2(&m, &one)?, m)
            };
            let first: Basic = m2(&pow(&sub(&s, &one)?, &a2a)?, &exp(&m2(&two_pi_i_a, &low)?)?)?;
            let second: Basic = m2(&pow(&a2(&s, &one)?, &a2a)?, &exp(&m2(&two_pi_i_a, &high)?)?)?;
            m2(&div(&s, &two)?, &sub(&first, &second)?)?
        }
        // (n - HALF)*pi*I + log(sqrt(x)/2) +- I*asin(1/sqrt(x))
        Function::HyperRepLog2 => {
            let first: Basic = m2(&m2(&sub(n, &Basic::half())?, &pi())?, &i())?;
            let second: Basic = log(&div(&sqrt(x)?, &two)?)?;
            let third: Basic = m2(&i(), &asin(&div(&one, &sqrt(x)?)?)?)?;
            if is_even(n) {
                a2(&a2(&first, &second)?, &third)?
            } else {
                sub(&a2(&first, &second)?, &third)?
            }
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

/// The closed form for big arguments of `-x` on the branch `n` (`_big_minus`).
fn big_minus(f: Function, p: &[Basic], x: &Basic, n: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::hyperbolic::{cosh, sinh};
    use crate::functions::elementary::inverse::{asinh, atan};
    use crate::functions::elementary::trigonometric::{cos, sin};
    let one: Basic = Basic::one();
    let two: Basic = int(2);
    let two_pi_i: Basic = m2(&m2(&two, &pi())?, &i())?;
    Ok(match f {
        Function::HyperRepPower1 => {
            let a: &Basic = &p[0];
            if is_true(a, Predicate::integer) {
                return small_minus(f, p, x);
            }
            // (1 + x)**a * exp(2*n*pi*I*a)
            let phase: Basic = exp(&m2(&m2(&m2(&m2(&two, n)?, &pi())?, &i())?, a)?)?;
            m2(&pow(&a2(&one, x)?, a)?, &phase)?
        }
        Function::HyperRepPower2 => {
            // sgn * 2**(2*a - 1) * (sqrt(1 + x) + sgn)**(1 - 2*a) * exp(-2*pi*I*a*n)
            let a: &Basic = &p[0];
            let sgn: Basic = int(if is_true(n, Predicate::odd) { -1 } else { 1 });
            let first: Basic = m2(&sgn, &pow(&two, &sub(&m2(&two, a)?, &one)?)?)?;
            let second: Basic = pow(&a2(&sqrt(&a2(&one, x)?)?, &sgn)?, &sub(&one, &m2(&two, a)?)?)?;
            let phase: Basic = exp(&m2(&m2(&m2(&m2(&int(-2), &pi())?, &i())?, a)?, n)?)?;
            m2(&m2(&first, &second)?, &phase)?
        }
        // log(1 + x) + 2*n*pi*I
        Function::HyperRepLog1 => a2(&log(&a2(&one, x)?)?, &m2(&m2(&m2(&two, n)?, &pi())?, &i())?)?,
        Function::HyperRepAtanh => {
            if is_even(n) {
                div(&atan(&sqrt(x)?)?, &sqrt(x)?)?
            } else {
                div(&sub(&atan(&sqrt(x)?)?, &pi())?, &sqrt(x)?)?
            }
        }
        // (-1)**n * (asinh(sqrt(x))/sqrt(x) + n*pi*I/sqrt(x))
        Function::HyperRepAsin1 => {
            let first: Basic = div(&asinh(&sqrt(x)?)?, &sqrt(x)?)?;
            let second: Basic = div(&m2(&m2(n, &pi())?, &i())?, &sqrt(x)?)?;
            m2(&pow(&int(-1), n)?, &a2(&first, &second)?)?
        }
        Function::HyperRepAsin2 => div(
            &big_minus(Function::HyperRepAsin1, &[], x, n)?,
            &big_minus(Function::HyperRepPower1, &[Basic::half()], x, n)?,
        )?,
        // (1 + x)**a * exp(2*pi*I*n*a) * cos(2*a*atan(sqrt(x)) [- 2*pi*a])
        Function::HyperRepSqrts1 | Function::HyperRepSqrts2 => {
            let a: &Basic = &p[0];
            let front: Basic = m2(&pow(&a2(&one, x)?, a)?, &exp(&m2(&m2(&two_pi_i, n)?, a)?)?)?;
            let mut angle: Basic = m2(&m2(&two, a)?, &atan(&sqrt(x)?)?)?;
            if !is_even(n) {
                angle = sub(&angle, &m2(&m2(&two, &pi())?, a)?)?;
            }
            if f == Function::HyperRepSqrts1 {
                m2(&front, &cos(&angle)?)?
            } else {
                m2(&m2(&front, &sqrt(x)?)?, &sin(&angle)?)?
            }
        }
        Function::HyperRepLog2 => {
            let first: Basic = m2(&m2(&pi(), &i())?, n)?;
            let root: Basic = div(&sqrt(&a2(&one, x)?)?, &two)?;
            if is_even(n) {
                a2(&first, &log(&a2(&Basic::half(), &root)?)?)?
            } else {
                a2(&first, &log(&sub(&root, &Basic::half())?)?)?
            }
        }
        // cosh(2*a*asinh(sqrt(x)) + 2*a*pi*I*n)
        Function::HyperRepCosasin => {
            let a: &Basic = &p[0];
            let first: Basic = m2(&m2(&two, a)?, &asinh(&sqrt(x)?)?)?;
            let second: Basic = m2(&m2(&m2(&m2(&two, a)?, &pi())?, &i())?, n)?;
            cosh(&a2(&first, &second)?)?
        }
        // -1/sqrt(1 + 1/x) * sinh(2*a*asinh(sqrt(x)) + 2*a*pi*I*n)
        Function::HyperRepSinasin => {
            let a: &Basic = &p[0];
            let factor: Basic = div(&int(-1), &sqrt(&a2(&one, &div(&one, x)?)?)?)?;
            let first: Basic = m2(&m2(&two, a)?, &asinh(&sqrt(x)?)?)?;
            let second: Basic = m2(&m2(&m2(&m2(&two, a)?, &pi())?, &i())?, n)?;
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
    let p: &[Basic] = &args[..args.len() - 1];
    let (x, n) = crate::functions::elementary::polar::extract_branch_factor(&args[args.len() - 1], true)?;
    let integral: bool = matches!(
        n.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    );
    if small_only {
        return if integral {
            small(f, p, &x)
        } else {
            small_minus(f, p, &x)
        };
    }
    let (s, b) = if integral {
        (small(f, p, &x)?, big(f, p, &x, &n)?)
    } else {
        let n: Basic = sub(&n, &Basic::half())?;
        (small_minus(f, p, &x)?, big_minus(f, p, &x, &n)?)
    };
    if b == s {
        return Ok(s);
    }
    crate::functions::elementary::piecewise::piecewise(&[
        (b, gt(&abs_(&x)?, &Basic::one())?),
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
