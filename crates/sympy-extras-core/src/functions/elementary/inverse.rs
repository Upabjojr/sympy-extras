//! Port of the inverse functions of
//! `sympy_extras/functions/elementary/trigonometric.py` (`asin`, `acos`,
//! `atan`, `acot`, `asec`, `acsc`) and of `hyperbolic.py` (`asinh`,
//! `acosh`, `atanh`, `acoth`, `asech`, `acsch`): the nodes of
//! [`Head::Function`], their evaluation (the values at the numbers and the
//! tables of SymPy, the odd symmetry, the exchange of the two families on
//! imaginary arguments, the inverse of a direct function of a comparable
//! angle), facts, derivatives and conjugates. The values which are
//! `AccumBounds` (`atan(zoo)`, `atanh(zoo)`, `asech(zoo)`) are
//! `NotImplemented`; the series are not ported.

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py (asin, acos, atan,
// acot, asec, acsc, InverseTrigonometricFunction tables),
// sympy/functions/elementary/hyperbolic.py (asinh, acosh, atanh, acoth, asech,
// acsch, the tables) (SymPy 1.14), through sympy_extras; BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::HashMap;

use crate::core::add::add;
use crate::core::assumptions::{facts_of, is_number, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{numeric_value, parts, Complex};
use crate::core::expr::{as_coefficient, is_comparable, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::{not_ported, PolyResult};
use crate::numbers::*;

use super::exponential::log;

use Predicate::*;

type Table = HashMap<Basic, Basic>;

fn node(f: Function, x: Basic) -> Basic {
    Basic::new(Head::Function(f), vec![x])
}

fn apply(f: Function, x: &Basic) -> PolyResult<Basic> {
    if evaluate() {
        if let Some(value) = eval(f, x)? {
            return Ok(value);
        }
    }
    Ok(node(f, x.clone()))
}

pub fn asin(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Asin, x)
}

pub fn acos(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Acos, x)
}

pub fn atan(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Atan, x)
}

pub fn acot(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Acot, x)
}

pub fn asec(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Asec, x)
}

pub fn acsc(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Acsc, x)
}

pub fn asinh(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Asinh, x)
}

pub fn acosh(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Acosh, x)
}

pub fn atanh(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Atanh, x)
}

pub fn acoth(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Acoth, x)
}

pub fn asech(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Asech, x)
}

pub fn acsch(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Acsch, x)
}

/// Whether `f` is one of the twelve inverse functions.
pub fn is_inverse(f: Function) -> bool {
    matches!(
        f,
        Function::Asin
            | Function::Acos
            | Function::Atan
            | Function::Acot
            | Function::Asec
            | Function::Acsc
            | Function::Asinh
            | Function::Acosh
            | Function::Atanh
            | Function::Acoth
            | Function::Asech
            | Function::Acsch
    )
}

/// Whether `f` is an inverse trigonometric function (the module of its
/// Python class is `trigonometric`, `hyperbolic` otherwise).
pub fn is_inverse_trigonometric(f: Function) -> bool {
    matches!(
        f,
        Function::Asin | Function::Acos | Function::Atan | Function::Acot | Function::Asec | Function::Acsc
    )
}

// ---------------------------------------------------------------------
// small constructors

fn int(n: i64) -> Basic {
    Basic::int(n)
}

fn q(p: i64, d: i64) -> Basic {
    Basic::from_rational(rat_from_parts(Int::from(p), Int::from(d)))
}

fn i() -> Basic {
    Basic::imaginary_unit()
}

fn pi() -> Basic {
    Basic::pi()
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

fn sqrt(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::half())
}

fn s(n: i64) -> PolyResult<Basic> {
    sqrt(&int(n))
}

fn sub(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), neg(b)?])
}

fn div(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}

/// `c*pi`.
fn pi_times(c: Basic) -> PolyResult<Basic> {
    mul(&[c, pi()])
}

fn half_pi() -> PolyResult<Basic> {
    pi_times(Basic::half())
}

fn is_rational_value(e: &Basic, v: i64) -> bool {
    e.as_rational().is_some_and(|r| *r == Rat::from(v))
}

fn is_zero(e: &Basic) -> bool {
    is_true(e, zero)
}

fn imaginary_coefficient(arg: &Basic) -> PolyResult<Option<Basic>> {
    as_coefficient(arg, &i())
}

/// `(f, true)` for `arg = -f` written as the product `-1*f`
/// (`_negated_node`).
fn negated_node(arg: &Basic) -> (Basic, bool) {
    if arg.is_mul() && arg.args().len() == 2 && is_rational_value(&arg.args()[0], -1) {
        return (arg.args()[1].clone(), true);
    }
    (arg.clone(), false)
}

fn function_of(arg: &Basic, f: Function) -> Option<Basic> {
    if arg.is_function(f) && is_comparable(&arg.args()[0]) {
        return Some(arg.args()[0].clone());
    }
    None
}

// ---------------------------------------------------------------------
// numerical comparisons of angles

/// Whether the comparable `a` is greater than `b` (`_gt`).
fn gt(a: &Basic, b: &Basic) -> PolyResult<bool> {
    if a == b {
        return Ok(false);
    }
    let Some(value) = numeric_value(&sub(a, b)?, 200) else {
        return Ok(false);
    };
    Ok(parts(&value).0 > crate::core::evalf::Real::ZERO)
}

/// `angle % period` for a comparable angle (`_mod`).
fn modulo(angle: &Basic, period: &Basic) -> PolyResult<Basic> {
    let Some(value) = numeric_value(&div(angle, period)?, 200) else {
        return Ok(angle.clone());
    };
    let k: Int = parts(&value).0.floor().to_int().value();
    if k.is_zero() {
        return Ok(angle.clone());
    }
    add(&[angle.clone(), mul(&[Basic::integer(-k), period.clone()])?])
}

/// A comparable angle reduced into `[-pi/2, pi/2]` with the same sine
/// (`_asin_angle`).
fn asin_angle(ang: &Basic) -> PolyResult<Basic> {
    let mut ang: Basic = modulo(ang, &pi_times(int(2))?)?;
    if gt(&ang, &pi())? {
        ang = sub(&pi(), &ang)?;
    }
    if gt(&ang, &half_pi()?)? {
        ang = sub(&pi(), &ang)?;
    }
    if gt(&neg(&half_pi()?)?, &ang)? {
        ang = sub(&neg(&pi())?, &ang)?;
    }
    Ok(ang)
}

/// A comparable angle reduced into `[0, pi]` with the same cosine (of the
/// opposite sign when `minus`) (`_acos_angle`).
fn acos_angle(ang: &Basic, minus: bool) -> PolyResult<Basic> {
    let mut ang: Basic = ang.clone();
    if minus {
        ang = sub(&pi(), &ang)?;
    }
    ang = modulo(&ang, &pi_times(int(2))?)?;
    if gt(&ang, &pi())? {
        ang = sub(&pi_times(int(2))?, &ang)?;
    }
    Ok(ang)
}

/// `ang - pi` when `ang > pi/2`.
fn into_half_turn(ang: Basic) -> PolyResult<Basic> {
    if gt(&ang, &half_pi()?)? {
        return sub(&ang, &pi());
    }
    Ok(ang)
}

// ---------------------------------------------------------------------
// the tables of SymPy

thread_local! {
    static TABLES: RefCell<HashMap<&'static str, Table>> = RefCell::new(HashMap::new());
}

fn lookup(
    name: &'static str,
    build: fn() -> PolyResult<Vec<(Basic, Basic)>>,
    key: &Basic,
) -> PolyResult<Option<Basic>> {
    let built: bool = TABLES.with(|t| t.borrow().contains_key(name));
    if !built {
        let entries: Table = build()?.into_iter().collect();
        TABLES.with(|t| t.borrow_mut().insert(name, entries));
    }
    Ok(TABLES.with(|t| t.borrow()[name].get(key).cloned()))
}

fn asin_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let (s2, s3, s5, s6) = (s(2)?, s(3)?, s(5)?, s(6)?);
    let half: Basic = Basic::half();
    let pi_q = |p: i64, d: i64| pi_times(q(p, d));
    Ok(vec![
        (div(&s3, &int(2))?, pi_q(1, 3)?),
        (div(&s2, &int(2))?, pi_q(1, 4)?),
        (div(&int(1), &s2)?, pi_q(1, 4)?),
        (sqrt(&div(&sub(&int(5), &s5)?, &int(8))?)?, pi_q(1, 5)?),
        (
            div(&mul(&[s2.clone(), sqrt(&sub(&int(5), &s5)?)?])?, &int(4))?,
            pi_q(1, 5)?,
        ),
        (sqrt(&div(&add(&[int(5), s5.clone()])?, &int(8))?)?, pi_q(2, 5)?),
        (
            div(&mul(&[s2.clone(), sqrt(&add(&[int(5), s5.clone()])?)?])?, &int(4))?,
            pi_q(2, 5)?,
        ),
        (half.clone(), pi_q(1, 6)?),
        (div(&sqrt(&sub(&int(2), &s2)?)?, &int(2))?, pi_q(1, 8)?),
        (sqrt(&sub(&half, &div(&s2, &int(4))?)?)?, pi_q(1, 8)?),
        (div(&sqrt(&add(&[int(2), s2.clone()])?)?, &int(2))?, pi_q(3, 8)?),
        (sqrt(&add(&[half.clone(), div(&s2, &int(4))?])?)?, pi_q(3, 8)?),
        (div(&sub(&s5, &int(1))?, &int(4))?, pi_q(1, 10)?),
        (div(&sub(&int(1), &s5)?, &int(4))?, pi_q(-1, 10)?),
        (div(&add(&[s5.clone(), int(1)])?, &int(4))?, pi_q(3, 10)?),
        (sub(&div(&s6, &int(4))?, &div(&s2, &int(4))?)?, pi_q(1, 12)?),
        (add(&[div(&neg(&s6)?, &int(4))?, div(&s2, &int(4))?])?, pi_q(-1, 12)?),
        (div(&sub(&s3, &int(1))?, &s(8)?)?, pi_q(1, 12)?),
        (div(&sub(&int(1), &s3)?, &s(8)?)?, pi_q(-1, 12)?),
        (add(&[div(&s6, &int(4))?, div(&s2, &int(4))?])?, pi_q(5, 12)?),
        (div(&add(&[int(1), s3.clone()])?, &s(8)?)?, pi_q(5, 12)?),
    ])
}

fn atan_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let (s2, s3, s5) = (s(2)?, s(3)?, s(5)?);
    let pi_q = |p: i64, d: i64| pi_times(q(p, d));
    let two_s5: Basic = mul(&[int(2), s5.clone()])?;
    Ok(vec![
        (div(&s3, &int(3))?, pi_q(1, 6)?),
        (div(&int(1), &s3)?, pi_q(1, 6)?),
        (s3.clone(), pi_q(1, 3)?),
        (sub(&s2, &int(1))?, pi_q(1, 8)?),
        (sub(&int(1), &s2)?, pi_q(-1, 8)?),
        (add(&[int(1), s2.clone()])?, pi_q(3, 8)?),
        (sqrt(&sub(&int(5), &two_s5)?)?, pi_q(1, 5)?),
        (sqrt(&add(&[int(5), two_s5.clone()])?)?, pi_q(2, 5)?),
        (sqrt(&sub(&int(1), &div(&two_s5, &int(5))?)?)?, pi_q(1, 10)?),
        (sqrt(&add(&[int(1), div(&two_s5, &int(5))?])?)?, pi_q(3, 10)?),
        (sub(&int(2), &s3)?, pi_q(1, 12)?),
        (add(&[int(-2), s3.clone()])?, pi_q(-1, 12)?),
        (add(&[int(2), s3.clone()])?, pi_q(5, 12)?),
    ])
}

fn acsc_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let (s2, s3, s5, s6) = (s(2)?, s(3)?, s(5)?, s(6)?);
    let pi_q = |p: i64, d: i64| pi_times(q(p, d));
    let s5_8: Basic = div(&s5, &int(8))?;
    let two_s5_5: Basic = div(&mul(&[int(2), s5.clone()])?, &int(5))?;
    let two_s2: Basic = mul(&[int(2), s2.clone()])?;
    Ok(vec![
        (div(&mul(&[int(2), s3.clone()])?, &int(3))?, pi_q(1, 3)?),
        (s2.clone(), pi_q(1, 4)?),
        (sqrt(&add(&[int(2), two_s5_5.clone()])?)?, pi_q(1, 5)?),
        (div(&int(1), &sqrt(&sub(&q(5, 8), &s5_8)?)?)?, pi_q(1, 5)?),
        (sqrt(&sub(&int(2), &two_s5_5)?)?, pi_q(2, 5)?),
        (div(&int(1), &sqrt(&add(&[q(5, 8), s5_8.clone()])?)?)?, pi_q(2, 5)?),
        (int(2), pi_q(1, 6)?),
        (sqrt(&add(&[int(4), two_s2.clone()])?)?, pi_q(1, 8)?),
        (div(&int(2), &sqrt(&sub(&int(2), &s2)?)?)?, pi_q(1, 8)?),
        (sqrt(&sub(&int(4), &two_s2)?)?, pi_q(3, 8)?),
        (div(&int(2), &sqrt(&add(&[int(2), s2.clone()])?)?)?, pi_q(3, 8)?),
        (add(&[int(1), s5.clone()])?, pi_q(1, 10)?),
        (sub(&s5, &int(1))?, pi_q(3, 10)?),
        (neg(&sub(&s5, &int(1))?)?, pi_q(-3, 10)?),
        (add(&[s6.clone(), s2.clone()])?, pi_q(1, 12)?),
        (sub(&s6, &s2)?, pi_q(5, 12)?),
        (neg(&sub(&s6, &s2)?)?, pi_q(-5, 12)?),
    ])
}

fn acosh_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let (s2, s3, s5) = (s(2)?, s(3)?, s(5)?);
    let pi_q = |p: i64, d: i64| pi_times(q(p, d));
    let half: Basic = Basic::half();
    let one_s2: Basic = add(&[int(1), s2.clone()])?;
    let s8: Basic = s(8)?;
    let r_2s2: Basic = sqrt(&add(&[int(2), s2.clone()])?)?;
    let r_2ms2: Basic = sqrt(&sub(&int(2), &s2)?)?;
    let a12: Basic = div(&add(&[int(1), s3.clone()])?, &mul(&[int(2), s2.clone()])?)?;
    let a5: Basic = div(&add(&[s5.clone(), int(1)])?, &int(4))?;
    let a512: Basic = div(&sub(&s3, &int(1))?, &s8)?;
    Ok(vec![
        (i(), log(&mul(&[i(), one_s2.clone()])?)?),
        (neg(&i())?, log(&mul(&[neg(&i())?, one_s2])?)?),
        (half.clone(), pi_q(1, 3)?),
        (neg(&half)?, pi_q(2, 3)?),
        (div(&s2, &int(2))?, pi_q(1, 4)?),
        (div(&neg(&s2)?, &int(2))?, pi_q(3, 4)?),
        (div(&int(1), &s2)?, pi_q(1, 4)?),
        (div(&int(-1), &s2)?, pi_q(3, 4)?),
        (div(&s3, &int(2))?, pi_q(1, 6)?),
        (div(&neg(&s3)?, &int(2))?, pi_q(5, 6)?),
        (a512.clone(), pi_q(5, 12)?),
        (div(&neg(&sub(&s3, &int(1))?)?, &s8)?, pi_q(7, 12)?),
        (div(&r_2s2, &int(2))?, pi_q(1, 8)?),
        (div(&neg(&r_2s2)?, &int(2))?, pi_q(7, 8)?),
        (div(&r_2ms2, &int(2))?, pi_q(3, 8)?),
        (div(&neg(&r_2ms2)?, &int(2))?, pi_q(5, 8)?),
        (a12.clone(), pi_q(1, 12)?),
        (
            div(&neg(&add(&[int(1), s3.clone()])?)?, &mul(&[int(2), s2.clone()])?)?,
            pi_q(11, 12)?,
        ),
        (a5.clone(), pi_q(1, 5)?),
        (div(&neg(&add(&[s5.clone(), int(1)])?)?, &int(4))?, pi_q(4, 5)?),
    ])
}

fn acsch_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let (s2, s3, s5, s6) = (s(2)?, s(3)?, s(5)?, s(6)?);
    let pi_q = |p: i64, d: i64| pi_times(q(p, d));
    let times_i = |e: Basic| mul(&[i(), e]);
    Ok(vec![
        (i(), pi_q(-1, 2)?),
        (times_i(add(&[s2.clone(), s6.clone()])?)?, pi_q(-1, 12)?),
        (times_i(add(&[int(1), s5.clone()])?)?, pi_q(-1, 10)?),
        (mul(&[i(), div(&int(2), &sqrt(&sub(&int(2), &s2)?)?)?])?, pi_q(-1, 8)?),
        (mul(&[i(), int(2)])?, pi_q(-1, 6)?),
        (times_i(sqrt(&add(&[int(2), div(&int(2), &s5)?])?)?)?, pi_q(-1, 5)?),
        (times_i(s2.clone())?, pi_q(-1, 4)?),
        (times_i(sub(&s5, &int(1))?)?, pi_q(-3, 10)?),
        (mul(&[i(), div(&int(2), &s3)?])?, pi_q(-1, 3)?),
        (
            mul(&[i(), div(&int(2), &sqrt(&add(&[int(2), s2.clone()])?)?)?])?,
            pi_q(-3, 8)?,
        ),
        (times_i(sqrt(&sub(&int(2), &div(&int(2), &s5)?)?)?)?, pi_q(-2, 5)?),
        (times_i(sub(&s6, &s2)?)?, pi_q(-5, 12)?),
        (
            int(2),
            mul(&[neg(&i())?, log(&div(&add(&[int(1), s5.clone()])?, &int(2))?)?])?,
        ),
    ])
}

fn asech_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let (s2, s3, s5, s6) = (s(2)?, s(3)?, s(5)?, s(6)?);
    let pi_q = |p: i64, d: i64| pi_times(q(p, d));
    let half_pi_i: Basic = mul(&[pi(), i(), Basic::half()])?;
    let log_1s2: Basic = log(&add(&[int(1), s2.clone()])?)?;
    let a10: Basic = sqrt(&sub(&int(2), &div(&int(2), &s5)?)?)?;
    let a8: Basic = div(&int(2), &sqrt(&add(&[int(2), s2.clone()])?)?)?;
    let a310: Basic = sqrt(&add(&[int(2), div(&int(2), &s5)?])?)?;
    let a38: Basic = sqrt(&mul(&[int(2), add(&[int(2), s2.clone()])?])?)?;
    Ok(vec![
        (i(), add(&[neg(&half_pi_i)?, log_1s2.clone()])?),
        (neg(&i())?, add(&[half_pi_i.clone(), log_1s2])?),
        (sub(&s6, &s2)?, pi_q(1, 12)?),
        (sub(&s2, &s6)?, pi_q(11, 12)?),
        (a10.clone(), pi_q(1, 10)?),
        (neg(&a10)?, pi_q(9, 10)?),
        (a8.clone(), pi_q(1, 8)?),
        (neg(&a8)?, pi_q(7, 8)?),
        (div(&int(2), &s3)?, pi_q(1, 6)?),
        (div(&int(-2), &s3)?, pi_q(5, 6)?),
        (sub(&s5, &int(1))?, pi_q(1, 5)?),
        (sub(&int(1), &s5)?, pi_q(4, 5)?),
        (s2.clone(), pi_q(1, 4)?),
        (neg(&s2)?, pi_q(3, 4)?),
        (a310.clone(), pi_q(3, 10)?),
        (neg(&a310)?, pi_q(7, 10)?),
        (int(2), pi_q(1, 3)?),
        (int(-2), pi_q(2, 3)?),
        (a38.clone(), pi_q(3, 8)?),
        (neg(&a38)?, pi_q(5, 8)?),
        (add(&[int(1), s5.clone()])?, pi_q(2, 5)?),
        (sub(&int(-1), &s5)?, pi_q(3, 5)?),
        (add(&[s6.clone(), s2.clone()])?, pi_q(5, 12)?),
        (sub(&neg(&s6)?, &s2)?, pi_q(7, 12)?),
        (mul(&[i(), Basic::oo()])?, neg(&half_pi_i)?),
        (mul(&[i(), Basic::negative_oo()])?, half_pi_i),
    ])
}

// ---------------------------------------------------------------------
// evaluation

/// `cls.eval(arg)` of the inverse function `f`.
pub fn eval(f: Function, arg: &Basic) -> PolyResult<Option<Basic>> {
    match f {
        Function::Asin => asin_eval(arg),
        Function::Acos => acos_eval(arg),
        Function::Atan => atan_eval(arg),
        Function::Acot => acot_eval(arg),
        Function::Asec => asec_eval(arg),
        Function::Acsc => acsc_eval(arg),
        Function::Asinh => asinh_eval(arg),
        Function::Acosh => acosh_eval(arg),
        Function::Atanh => atanh_eval(arg),
        Function::Acoth => acoth_eval(arg),
        Function::Asech => asech_eval(arg),
        Function::Acsch => acsch_eval(arg),
        _ => unreachable!("not an inverse function"),
    }
}

fn odd(f: Function, arg: &Basic) -> PolyResult<Basic> {
    neg(&apply(f, &neg(arg)?)?)
}

fn asin_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    match arg.head() {
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::Infinity => return Ok(Some(mul(&[Basic::negative_oo(), i()])?)),
        Head::NegativeInfinity => return Ok(Some(mul(&[Basic::oo(), i()])?)),
        Head::ComplexInfinity => return Ok(Some(Basic::zoo())),
        _ => {}
    }
    if arg.as_rational().is_some() {
        if is_rational_value(arg, 0) {
            return Ok(Some(Basic::zero()));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(half_pi()?));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(neg(&half_pi()?)?));
        }
    }
    if could_extract_minus_sign(arg) {
        return Ok(Some(odd(Function::Asin, arg)?));
    }
    if is_number(arg) {
        if let Some(found) = lookup("asin", asin_table, arg)? {
            return Ok(Some(found));
        }
    }
    if let Some(c) = imaginary_coefficient(arg)? {
        return Ok(Some(mul(&[i(), asinh(&c)?])?));
    }
    if is_zero(arg) {
        return Ok(Some(Basic::zero()));
    }
    if let Some(angle) = function_of(arg, Function::Sin) {
        return Ok(Some(asin_angle(&angle)?));
    }
    if function_of(arg, Function::Cos).is_some() {
        return Ok(Some(sub(&half_pi()?, &acos(arg)?)?));
    }
    Ok(None)
}

fn acos_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    match arg.head() {
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::Infinity => return Ok(Some(mul(&[Basic::oo(), i()])?)),
        Head::NegativeInfinity => return Ok(Some(mul(&[Basic::negative_oo(), i()])?)),
        Head::ComplexInfinity => return Ok(Some(Basic::zoo())),
        _ => {}
    }
    if arg.as_rational().is_some() {
        if is_rational_value(arg, 0) {
            return Ok(Some(half_pi()?));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(Basic::zero()));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(pi()));
        }
    }
    if is_number(arg) {
        if let Some(found) = lookup("asin", asin_table, arg)? {
            return Ok(Some(sub(&half_pi()?, &found)?));
        }
        if let Some(negated) = lookup("asin", asin_table, &neg(arg)?)? {
            return Ok(Some(add(&[half_pi()?, negated])?));
        }
    }
    if imaginary_coefficient(arg)?.is_some() {
        return Ok(Some(sub(&half_pi()?, &asin(arg)?)?));
    }
    let (inner, minus) = negated_node(arg);
    if let Some(angle) = function_of(&inner, Function::Cos) {
        return Ok(Some(acos_angle(&angle, minus)?));
    }
    if function_of(&inner, Function::Sin).is_some() {
        let a: Basic = asin(&inner)?;
        return Ok(Some(if minus {
            add(&[half_pi()?, a])?
        } else {
            sub(&half_pi()?, &a)?
        }));
    }
    Ok(None)
}

fn atan_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    match arg.head() {
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::Infinity => return Ok(Some(half_pi()?)),
        Head::NegativeInfinity => return Ok(Some(neg(&half_pi()?)?)),
        Head::ComplexInfinity => return Err(not_ported("AccumBounds (atan(zoo))")),
        _ => {}
    }
    if arg.as_rational().is_some() {
        if is_rational_value(arg, 0) {
            return Ok(Some(Basic::zero()));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(pi_times(q(1, 4))?));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(pi_times(q(-1, 4))?));
        }
    }
    if could_extract_minus_sign(arg) {
        return Ok(Some(odd(Function::Atan, arg)?));
    }
    if is_number(arg) {
        if let Some(found) = lookup("atan", atan_table, arg)? {
            return Ok(Some(found));
        }
    }
    if let Some(c) = imaginary_coefficient(arg)? {
        return Ok(Some(mul(&[i(), atanh(&c)?])?));
    }
    if is_zero(arg) {
        return Ok(Some(Basic::zero()));
    }
    if let Some(angle) = function_of(arg, Function::Tan) {
        return Ok(Some(into_half_turn(modulo(&angle, &pi())?)?));
    }
    if function_of(arg, Function::Cot).is_some() {
        return Ok(Some(into_half_turn(sub(&half_pi()?, &acot(arg)?)?)?));
    }
    Ok(None)
}

fn acot_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    match arg.head() {
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity => return Ok(Some(Basic::zero())),
        _ => {}
    }
    if arg.as_rational().is_some() {
        if is_rational_value(arg, 0) {
            return Ok(Some(half_pi()?));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(pi_times(q(1, 4))?));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(pi_times(q(-1, 4))?));
        }
    }
    if could_extract_minus_sign(arg) {
        return Ok(Some(odd(Function::Acot, arg)?));
    }
    if is_number(arg) {
        if let Some(found) = lookup("atan", atan_table, arg)? {
            return Ok(Some(into_half_turn(sub(&half_pi()?, &found)?)?));
        }
    }
    if let Some(c) = imaginary_coefficient(arg)? {
        return Ok(Some(mul(&[Basic::negative_one(), i(), acoth(&c)?])?));
    }
    if is_zero(arg) {
        return Ok(Some(half_pi()?));
    }
    if let Some(angle) = function_of(arg, Function::Cot) {
        return Ok(Some(into_half_turn(modulo(&angle, &pi())?)?));
    }
    if function_of(arg, Function::Tan).is_some() {
        return Ok(Some(into_half_turn(sub(&half_pi()?, &atan(arg)?)?)?));
    }
    Ok(None)
}

fn asec_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if is_zero(arg) {
        return Ok(Some(Basic::zoo()));
    }
    if matches!(arg.head(), Head::NaN) {
        return Ok(Some(Basic::nan()));
    }
    if is_rational_value(arg, 1) {
        return Ok(Some(Basic::zero()));
    }
    if is_rational_value(arg, -1) {
        return Ok(Some(pi()));
    }
    if matches!(
        arg.head(),
        Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity
    ) {
        return Ok(Some(half_pi()?));
    }
    if is_number(arg) {
        if let Some(found) = lookup("acsc", acsc_table, arg)? {
            return Ok(Some(sub(&half_pi()?, &found)?));
        }
        if let Some(negated) = lookup("acsc", acsc_table, &neg(arg)?)? {
            return Ok(Some(add(&[half_pi()?, negated])?));
        }
    }
    if is_true(arg, infinite) {
        return Ok(Some(half_pi()?));
    }
    let (inner, minus) = negated_node(arg);
    if let Some(angle) = function_of(&inner, Function::Sec) {
        return Ok(Some(acos_angle(&angle, minus)?));
    }
    if function_of(&inner, Function::Csc).is_some() {
        // SymPy computes pi/2 + acsc(node) for the negated case and
        // discards it: pi/2 - acsc(node) in both cases
        return Ok(Some(sub(&half_pi()?, &acsc(&inner)?)?));
    }
    Ok(None)
}

fn acsc_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if is_zero(arg) {
        return Ok(Some(Basic::zoo()));
    }
    if matches!(arg.head(), Head::NaN) {
        return Ok(Some(Basic::nan()));
    }
    if is_rational_value(arg, 1) {
        return Ok(Some(half_pi()?));
    }
    if is_rational_value(arg, -1) {
        return Ok(Some(neg(&half_pi()?)?));
    }
    if matches!(
        arg.head(),
        Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity
    ) {
        return Ok(Some(Basic::zero()));
    }
    if could_extract_minus_sign(arg) {
        return Ok(Some(odd(Function::Acsc, arg)?));
    }
    if is_true(arg, infinite) {
        return Ok(Some(Basic::zero()));
    }
    if is_number(arg) {
        if let Some(found) = lookup("acsc", acsc_table, arg)? {
            return Ok(Some(found));
        }
    }
    if let Some(angle) = function_of(arg, Function::Csc) {
        return Ok(Some(asin_angle(&angle)?));
    }
    if function_of(arg, Function::Sec).is_some() {
        return Ok(Some(sub(&half_pi()?, &asec(arg)?)?));
    }
    Ok(None)
}

/// A `Number` other than `zoo` (the first branch of the inverse
/// hyperbolic functions).
fn finite_number_class(arg: &Basic) -> bool {
    arg.is_number_class() && !matches!(arg.head(), Head::ComplexInfinity)
}

fn i_pi_times(c: Basic) -> PolyResult<Basic> {
    mul(&[c, pi(), i()])
}

fn asinh_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if finite_number_class(arg) {
        match arg.head() {
            Head::NaN => return Ok(Some(Basic::nan())),
            Head::Infinity => return Ok(Some(Basic::oo())),
            Head::NegativeInfinity => return Ok(Some(Basic::negative_oo())),
            _ => {}
        }
        if is_rational_value(arg, 0) {
            return Ok(Some(Basic::zero()));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(log(&add(&[s(2)?, int(1)])?)?));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(log(&sub(&s(2)?, &int(1))?)?));
        }
        if is_true(arg, negative) {
            return Ok(Some(odd(Function::Asinh, arg)?));
        }
    } else {
        if matches!(arg.head(), Head::ComplexInfinity) {
            return Ok(Some(Basic::zoo()));
        }
        if is_zero(arg) {
            return Ok(Some(Basic::zero()));
        }
        if let Some(c) = imaginary_coefficient(arg)? {
            return Ok(Some(mul(&[i(), asin(&c)?])?));
        }
        if could_extract_minus_sign(arg) {
            return Ok(Some(odd(Function::Asinh, arg)?));
        }
    }
    if arg.is_function(Function::Sinh) && is_number(&arg.args()[0]) && is_true(&arg.args()[0], real) {
        return Ok(Some(arg.args()[0].clone()));
    }
    Ok(None)
}

fn acosh_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if finite_number_class(arg) {
        match arg.head() {
            Head::NaN => return Ok(Some(Basic::nan())),
            Head::Infinity | Head::NegativeInfinity => return Ok(Some(Basic::oo())),
            _ => {}
        }
        if is_rational_value(arg, 0) {
            return Ok(Some(i_pi_times(Basic::half())?));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(Basic::zero()));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(i_pi_times(Basic::one())?));
        }
    }
    if is_number(arg) {
        if let Some(found) = lookup("acosh", acosh_table, arg)? {
            return Ok(Some(if is_true(arg, extended_real) {
                mul(&[found, i()])?
            } else {
                found
            }));
        }
    }
    if matches!(arg.head(), Head::ComplexInfinity) {
        return Ok(Some(Basic::zoo()));
    }
    if *arg == mul(&[i(), Basic::oo()])? {
        return Ok(Some(add(&[Basic::oo(), i_pi_times(Basic::half())?])?));
    }
    if *arg == mul(&[Basic::negative_one(), i(), Basic::oo()])? {
        return Ok(Some(sub(&Basic::oo(), &i_pi_times(Basic::half())?)?));
    }
    if is_zero(arg) {
        return Ok(Some(i_pi_times(Basic::half())?));
    }
    Ok(None)
}

fn atanh_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if finite_number_class(arg) {
        match arg.head() {
            Head::NaN => return Ok(Some(Basic::nan())),
            Head::Infinity => return Ok(Some(mul(&[Basic::negative_one(), i(), atan(arg)?])?)),
            Head::NegativeInfinity => return Ok(Some(mul(&[i(), atan(&neg(arg)?)?])?)),
            _ => {}
        }
        if is_rational_value(arg, 0) {
            return Ok(Some(Basic::zero()));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(Basic::oo()));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(Basic::negative_oo()));
        }
        if is_true(arg, negative) {
            return Ok(Some(odd(Function::Atanh, arg)?));
        }
    } else {
        if matches!(arg.head(), Head::ComplexInfinity) {
            return Err(not_ported("AccumBounds (atanh(zoo))"));
        }
        if let Some(c) = imaginary_coefficient(arg)? {
            return Ok(Some(mul(&[i(), atan(&c)?])?));
        }
        if could_extract_minus_sign(arg) {
            return Ok(Some(odd(Function::Atanh, arg)?));
        }
    }
    if is_zero(arg) {
        return Ok(Some(Basic::zero()));
    }
    if arg.is_function(Function::Tanh) && is_number(&arg.args()[0]) && is_true(&arg.args()[0], real) {
        return Ok(Some(arg.args()[0].clone()));
    }
    Ok(None)
}

fn acoth_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if finite_number_class(arg) {
        match arg.head() {
            Head::NaN => return Ok(Some(Basic::nan())),
            Head::Infinity | Head::NegativeInfinity => return Ok(Some(Basic::zero())),
            _ => {}
        }
        if is_rational_value(arg, 0) {
            return Ok(Some(i_pi_times(Basic::half())?));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(Basic::oo()));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(Basic::negative_oo()));
        }
        if is_true(arg, negative) {
            return Ok(Some(odd(Function::Acoth, arg)?));
        }
    } else {
        if matches!(arg.head(), Head::ComplexInfinity) {
            return Ok(Some(Basic::zero()));
        }
        if let Some(c) = imaginary_coefficient(arg)? {
            return Ok(Some(mul(&[Basic::negative_one(), i(), acot(&c)?])?));
        }
        if could_extract_minus_sign(arg) {
            return Ok(Some(odd(Function::Acoth, arg)?));
        }
    }
    if is_zero(arg) {
        return Ok(Some(i_pi_times(Basic::half())?));
    }
    Ok(None)
}

fn asech_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if finite_number_class(arg) {
        match arg.head() {
            Head::NaN => return Ok(Some(Basic::nan())),
            Head::Infinity | Head::NegativeInfinity => return Ok(Some(i_pi_times(Basic::half())?)),
            _ => {}
        }
        if is_rational_value(arg, 0) {
            return Ok(Some(Basic::oo()));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(Basic::zero()));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(i_pi_times(Basic::one())?));
        }
    }
    if is_number(arg) {
        if let Some(found) = lookup("asech", asech_table, arg)? {
            return Ok(Some(if is_true(arg, extended_real) {
                mul(&[found, i()])?
            } else {
                found
            }));
        }
    }
    if matches!(arg.head(), Head::ComplexInfinity) {
        return Err(not_ported("AccumBounds (asech(zoo))"));
    }
    if is_zero(arg) {
        return Ok(Some(Basic::oo()));
    }
    Ok(None)
}

fn acsch_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if finite_number_class(arg) {
        match arg.head() {
            Head::NaN => return Ok(Some(Basic::nan())),
            Head::Infinity | Head::NegativeInfinity => return Ok(Some(Basic::zero())),
            _ => {}
        }
        if is_rational_value(arg, 0) {
            return Ok(Some(Basic::zoo()));
        }
        if is_rational_value(arg, 1) {
            return Ok(Some(log(&add(&[int(1), s(2)?])?)?));
        }
        if is_rational_value(arg, -1) {
            return Ok(Some(neg(&log(&add(&[int(1), s(2)?])?)?)?));
        }
    }
    if is_number(arg) {
        if let Some(found) = lookup("acsch", acsch_table, arg)? {
            return Ok(Some(mul(&[found, i()])?));
        }
    }
    if matches!(arg.head(), Head::ComplexInfinity) || is_true(arg, infinite) {
        return Ok(Some(Basic::zero()));
    }
    if is_zero(arg) {
        return Ok(Some(Basic::zoo()));
    }
    if could_extract_minus_sign(arg) {
        return Ok(Some(odd(Function::Acsch, arg)?));
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// derivatives, facts, conjugates, values

/// The derivative with respect to the argument (`fdiff`).
pub fn inverse_fdiff(f: Function, x: &Basic) -> PolyResult<Basic> {
    let one: Basic = Basic::one();
    let square: Basic = pow(x, &int(2))?;
    let inverse = |e: Basic| pow(&e, &Basic::negative_one());
    // x**2*sqrt(1 -+ x**(-2))
    let reciprocal_root = |sign: i64| -> PolyResult<Basic> {
        let inner: Basic = add(&[one.clone(), mul(&[int(sign), pow(x, &int(-2))?])?])?;
        mul(&[square.clone(), sqrt(&inner)?])
    };
    match f {
        Function::Asin => inverse(sqrt(&sub(&one, &square)?)?),
        Function::Acos => neg(&inverse(sqrt(&sub(&one, &square)?)?)?),
        Function::Atan => inverse(add(&[one.clone(), square.clone()])?),
        Function::Acot => neg(&inverse(add(&[one.clone(), square.clone()])?)?),
        Function::Asec => inverse(reciprocal_root(-1)?),
        Function::Acsc => neg(&inverse(reciprocal_root(-1)?)?),
        Function::Asinh => inverse(sqrt(&add(&[square.clone(), one.clone()])?)?),
        Function::Acosh => inverse(mul(&[sqrt(&sub(x, &one)?)?, sqrt(&add(&[x.clone(), one.clone()])?)?])?),
        Function::Atanh | Function::Acoth => inverse(sub(&one, &square)?),
        Function::Asech => neg(&inverse(mul(&[x.clone(), sqrt(&sub(&one, &square)?)?])?)?),
        Function::Acsch => neg(&inverse(reciprocal_root(1)?)?),
        _ => unreachable!("not an inverse function"),
    }
}

/// The facts of the node (`_eval_facts`: commutative; the arcsine and
/// the arctangent of a nonzero rational are irrational).
pub fn inverse_facts(f: Function, arg: &Basic) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    if matches!(f, Function::Asin | Function::Atan) {
        let a = facts_of(arg);
        if a.get(&rational) == Some(&true) && a.get(&zero) == Some(&false) {
            facts.set(rational, false);
        }
    }
    facts
}

/// The conjugate by the rule of the class: `acos(conjugate(z))` off the
/// cuts of `acos`, `None` otherwise (`_eval_conjugate`, the default of
/// `DefinedFunction` for the others).
pub fn inverse_conjugate(e: &Basic, f: Function) -> PolyResult<Option<Basic>> {
    if f == Function::Acos {
        let z: &Basic = &e.args()[0];
        let conjugated = || -> PolyResult<Basic> { acos(&super::complexes::conjugate(z)?) };
        if crate::core::expr::is_false(z, extended_real) {
            return Ok(Some(conjugated()?));
        }
        if is_true(z, extended_real)
            && is_true(&add(&[z.clone(), int(1)])?, nonnegative)
            && is_true(&add(&[z.clone(), int(-1)])?, nonpositive)
        {
            return Ok(Some(conjugated()?));
        }
        return Ok(None);
    }
    if is_true(e, extended_real) {
        return Ok(Some(e.clone()));
    }
    if is_true(e, imaginary) {
        return Ok(Some(neg(e)?));
    }
    Ok(None)
}

/// The value at `z` on the principal branch, mpmath's (the formulas by
/// logarithms agree with it on the cuts); `None` at the singular points,
/// also where mpmath gives an infinite value (`atan(I)`, `acot(-I)`).
pub fn inverse_value(f: Function, z: &Complex, prec: usize) -> Option<Complex> {
    use crate::core::evalf::{complex, complex_real, real_int};
    let zero_real = || real_int(&Int::ZERO, prec);
    let c = |n: i64| complex_real(real_int(&Int::from(n), prec), prec);
    let i: Complex = complex(zero_real(), real_int(&Int::ONE, prec));
    let is_zero = |w: &Complex| {
        let (a, b) = parts(w);
        a.repr().significand().is_zero() && b.repr().significand().is_zero()
    };
    let ln = |w: &Complex| -> Option<Complex> {
        if is_zero(w) {
            None
        } else {
            Some(w.ln())
        }
    };
    // the principal square root from the parts (exact on the real axis,
    // so that the products stay on the right side of the cuts)
    let sqrt = |w: &Complex| -> Option<Complex> {
        use crate::core::evalf::Real;
        let (a, b) = parts(w);
        let two: Real = real_int(&Int::from(2u8), prec);
        if b.repr().significand().is_zero() {
            if a >= Real::ZERO {
                return Some(complex(a.sqrt(), zero_real()));
            }
            return Some(complex(zero_real(), (-a).sqrt()));
        }
        let modulus: Real = (&a * &a + &b * &b).sqrt();
        let x: Real = ((&modulus + &a) / &two).sqrt();
        let y: Real = ((&modulus - &a) / &two).sqrt();
        Some(complex(x, if b < Real::ZERO { -y } else { y }))
    };
    let half_pi: Complex = complex_real(crate::core::evalf::Real::pi(prec), prec) / c(2);
    let recip = |w: &Complex| -> Option<Complex> {
        if is_zero(w) {
            None
        } else {
            Some(&c(1) / w)
        }
    };
    // asin z = -i ln(i z + sqrt(1 - z^2))
    let asin_of = |w: &Complex| -> Option<Complex> {
        let root: Complex = sqrt(&(&c(1) - &(w * w)))?;
        Some(&(-&i) * &ln(&(&(&i * w) + &root))?)
    };
    // atan z = i/2 (ln(1 - i z) - ln(1 + i z))
    let atan_of = |w: &Complex| -> Option<Complex> {
        let iw: Complex = &i * w;
        Some(&(&i / &c(2)) * &(&ln(&(&c(1) - &iw))? - &ln(&(&c(1) + &iw))?))
    };
    let asinh_of = |w: &Complex| -> Option<Complex> { ln(&(w + &sqrt(&(&(w * w) + &c(1)))?)) };
    let acosh_of = |w: &Complex| -> Option<Complex> { ln(&(w + &(&sqrt(&(w + &c(1)))? * &sqrt(&(w - &c(1)))?))) };
    let atanh_of = |w: &Complex| -> Option<Complex> { Some(&(&ln(&(&c(1) + w))? - &ln(&(&c(1) - w))?) / &c(2)) };
    let is_unit = |w: &Complex| is_zero(&(w - &c(1))) || is_zero(&(w + &c(1)));
    match f {
        Function::Asin => asin_of(z),
        Function::Acos => Some(&half_pi - &asin_of(z)?),
        Function::Atan => atan_of(z),
        Function::Acot => {
            if is_zero(z) {
                return Some(half_pi);
            }
            atan_of(&recip(z)?)
        }
        Function::Asec => Some(&half_pi - &asin_of(&recip(z)?)?),
        Function::Acsc => asin_of(&recip(z)?),
        Function::Asinh => asinh_of(z),
        Function::Acosh => acosh_of(z),
        Function::Atanh => {
            if is_unit(z) {
                return None;
            }
            atanh_of(z)
        }
        Function::Acoth => {
            if is_unit(z) {
                return None;
            }
            atanh_of(&recip(z)?)
        }
        Function::Asech => acosh_of(&recip(z)?),
        Function::Acsch => asinh_of(&recip(z)?),
        _ => unreachable!("not an inverse function"),
    }
}
