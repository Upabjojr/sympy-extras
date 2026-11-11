//! Port of `sympy_extras/functions/elementary/trigonometric.py`, the direct
//! functions: `sin`, `cos`, `tan`, `cot`, `sec`, `csc` (nodes of
//! [`Head::Function`]; the lower case functions evaluate), with SymPy's
//! exact values at rational multiples of `pi`, the peeling of multiples of
//! `pi/2` off sums, the facts, values, conjugates and real and imaginary
//! parts.
//!
//! The arguments `oo`, `-oo` (accumulation bounds) and the imaginary ones
//! (the hyperbolic functions) are the error `NotImplemented` until those
//! modules are ported, as are the inverse functions, `atan2` and `sinc`.

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py (SymPy 1.14),
// through sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_coeff_mul, as_coefficient, is, is_rational_value, is_true};
use crate::core::function::{expand, expand_default};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::{not_ported, PolyResult};
use crate::numbers::*;

use super::complexes::{as_real_imag, conjugate, RealImag};

use Predicate::*;

fn node(f: Function, x: Basic) -> Basic {
    Basic::new(Head::Function(f), vec![x])
}

fn pi() -> Basic {
    Basic::pi()
}

fn times_pi(c: &Rat) -> PolyResult<Basic> {
    mul(&[Basic::from_rational(c.clone()), pi()])
}

/// `a % b` of fractions (Python's, the sign of `b`).
fn fmod(a: &Rat, b: &Rat) -> Rat {
    a - b * rat_from_int(&div_floor(&numer(&(a / b)), &denom(&(a / b))))
}

fn apply(f: Function, x: &Basic) -> PolyResult<Basic> {
    if evaluate() {
        if let Some(value) = eval(f, x)? {
            return Ok(value);
        }
    }
    Ok(node(f, x.clone()))
}

pub fn sin(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Sin, x)
}

pub fn cos(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Cos, x)
}

pub fn tan(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Tan, x)
}

pub fn cot(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Cot, x)
}

pub fn sec(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Sec, x)
}

pub fn csc(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Csc, x)
}

/// `.eval` of one of the functions of this module.
pub fn eval(f: Function, x: &Basic) -> PolyResult<Option<Basic>> {
    match f {
        Function::Sin => sin_eval(x),
        Function::Cos => cos_eval(x),
        Function::Tan => tan_eval(x),
        Function::Cot => cot_eval(x),
        Function::Sec => reciprocal_eval(Function::Sec, Function::Cos, true, x),
        Function::Csc => reciprocal_eval(Function::Csc, Function::Sin, false, x),
        _ => unreachable!("not a trigonometric function"),
    }
}

/// `c` with `arg = c*I` (`imaginary_coefficient`).
fn imaginary_coefficient(arg: &Basic) -> PolyResult<Option<Basic>> {
    as_coefficient(arg, &Basic::imaginary_unit())
}

/// The denominators whose cosines come from a product formula (`_table2`).
fn table2(q: &Int) -> Option<(i64, i64)> {
    let q: i64 = i64::try_from(q).ok()?;
    match q {
        12 => Some((3, 4)),
        20 => Some((4, 5)),
        30 => Some((5, 6)),
        15 => Some((6, 10)),
        24 => Some((6, 8)),
        40 => Some((8, 10)),
        60 => Some((20, 30)),
        120 => Some((40, 60)),
        _ => None,
    }
}

/// The coefficient of `pi` in a term (`_pi_part`).
fn pi_part(term: &Basic) -> PolyResult<Option<Basic>> {
    if *term == pi() {
        return Ok(Some(Basic::one()));
    }
    if term.is_mul() {
        if let Some(k) = term.args().iter().position(|a| *a == pi()) {
            let mut others: Vec<Basic> = term.args().to_vec();
            others.remove(k);
            return Ok(Some(mul(&others)?));
        }
    }
    Ok(None)
}

/// `(rest, m)` with `arg = rest + m*pi`, `m` a multiple of `1/2`
/// (`peeloff_pi`).
pub fn peeloff_pi(arg: &Basic) -> PolyResult<(Basic, Rat)> {
    let mut coefficient: Rat = Rat::ZERO;
    let mut rest: Vec<Basic> = Vec::new();
    let terms: Vec<Basic> = if arg.is_add() {
        arg.args().to_vec()
    } else {
        vec![arg.clone()]
    };
    for term in terms {
        match pi_part(&term)? {
            Some(k) if k.as_rational().is_some() => coefficient += k.as_rational().expect("a rational").clone(),
            _ => rest.push(term),
        }
    }
    if coefficient.is_zero() {
        return Ok((arg.clone(), Rat::ZERO));
    }
    let m1: Rat = fmod(&coefficient, &rat(1, 2));
    let m2: Rat = &coefficient - &m1;
    rest.push(times_pi(&m1)?);
    Ok((add(&rest)?, m2))
}

/// The number `c` with `arg = c*pi`, normalised into `[0, 2]`
/// (`pi_coeff`).
pub fn pi_coeff(arg: &Basic) -> PolyResult<Option<Basic>> {
    if *arg == pi() {
        return Ok(Some(Basic::one()));
    }
    if arg.as_rational().is_some_and(|v| v.is_zero()) {
        return Ok(Some(Basic::zero()));
    }
    if arg.is_mul() {
        if let Some(cx) = as_coefficient(arg, &pi())? {
            let (c, x) = as_coeff_mul(&cx);
            let Some(cv) = c.as_rational() else {
                return Ok(Some(cx));
            };
            if is_true(&x, integer) {
                let c2: Rat = fmod(cv, &rat(2, 1));
                if c2 == Rat::ONE {
                    return Ok(Some(x));
                }
                if c2.is_zero() {
                    return Ok(Some(if is(&x, even).is_some() {
                        Basic::zero()
                    } else {
                        Basic::int(2)
                    }));
                }
                return Ok(Some(mul(&[Basic::from_rational(c2), x])?));
            }
            return Ok(Some(cx));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::zero()));
    }
    Ok(None)
}

/// The Chebyshev polynomial `T_n` at `x`, expanded.
fn chebyshev_t(n: &Int, x: &Basic) -> PolyResult<Basic> {
    let mut previous: Basic = Basic::one();
    let mut current: Basic = x.clone();
    if n.is_zero() {
        return Ok(previous);
    }
    let steps: usize = usize::try_from(n).map_err(|_| not_ported("a huge Chebyshev polynomial"))? - 1;
    for _ in 0..steps {
        let next: Basic = add(&[
            mul(&[Basic::int(2), x.clone(), current.clone()])?,
            mul(&[Basic::negative_one(), previous])?,
        ])?;
        previous = current;
        current = next;
    }
    expand(&current, true, true, true)
}

fn unsupported_infinite(arg: &Basic) -> PolyResult<()> {
    if matches!(arg.head(), Head::Infinity | Head::NegativeInfinity) {
        return Err(not_ported("accumulation bounds"));
    }
    Ok(())
}

fn is_zero_value(e: &Basic) -> bool {
    e.as_rational().is_some_and(|v| v.is_zero())
}

fn sin_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(arg.head(), Head::NaN | Head::ComplexInfinity) {
        return Ok(Some(Basic::nan()));
    }
    if is_zero_value(arg) {
        return Ok(Some(Basic::zero()));
    }
    unsupported_infinite(arg)?;
    if could_extract_minus_sign(arg) {
        return Ok(Some(mul(&[
            Basic::negative_one(),
            sin(&mul(&[Basic::negative_one(), arg.clone()])?)?,
        ])?));
    }
    if imaginary_coefficient(arg)?.is_some() {
        return Err(not_ported("sinh"));
    }
    if let Some(coefficient) = pi_coeff(arg)? {
        if is_true(&coefficient, integer) {
            return Ok(Some(Basic::zero()));
        }
        if is_true(&mul(&[Basic::int(2), coefficient.clone()])?, integer)
            && crate::core::expr::is_false(&coefficient, even)
        {
            return Ok(Some(pow(
                &Basic::negative_one(),
                &add(&[coefficient, Basic::from_rational(rat(-1, 2))])?,
            )?));
        }
        let Some(c) = coefficient.as_rational() else {
            let narg: Basic = mul(&[coefficient, pi()])?;
            return if narg != *arg { Ok(Some(sin(&narg)?)) } else { Ok(None) };
        };
        let x: Rat = fmod(c, &rat(2, 1));
        if x > Rat::ONE {
            return Ok(Some(mul(&[
                Basic::negative_one(),
                sin(&times_pi(&fmod(&x, &Rat::ONE))?)?,
            ])?));
        }
        if &x * rat(2, 1) > Rat::ONE {
            return Ok(Some(sin(&times_pi(&(Rat::ONE - &x))?)?));
        }
        let narg: Basic = times_pi(&fmod(&(c + rat(3, 2)), &rat(2, 1)))?;
        let result: Basic = cos(&narg)?;
        if !result.is_function(Function::Cos) {
            return Ok(Some(result));
        }
        let angle: Basic = mul(&[coefficient, pi()])?;
        if angle != *arg {
            return Ok(Some(sin(&angle)?));
        }
        return Ok(None);
    }
    if arg.is_add() {
        let (rest, m) = peeloff_pi(arg)?;
        if !m.is_zero() {
            let multiple: Basic = times_pi(&m)?;
            return Ok(Some(add(&[
                mul(&[sin(&multiple)?, cos(&rest)?])?,
                mul(&[cos(&multiple)?, sin(&rest)?])?,
            ])?));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::zero()));
    }
    Ok(None)
}

fn cos_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(arg.head(), Head::NaN | Head::ComplexInfinity) {
        return Ok(Some(Basic::nan()));
    }
    if is_zero_value(arg) {
        return Ok(Some(Basic::one()));
    }
    unsupported_infinite(arg)?;
    if is_true(arg, extended_real) && crate::core::expr::is_false(arg, finite) {
        return Err(not_ported("accumulation bounds"));
    }
    if could_extract_minus_sign(arg) {
        return Ok(Some(cos(&mul(&[Basic::negative_one(), arg.clone()])?)?));
    }
    if imaginary_coefficient(arg)?.is_some() {
        return Err(not_ported("cosh"));
    }
    if let Some(coefficient) = pi_coeff(arg)? {
        if is_true(&coefficient, integer) {
            return Ok(Some(pow(&Basic::negative_one(), &coefficient)?));
        }
        if is_true(&mul(&[Basic::int(2), coefficient.clone()])?, integer)
            && crate::core::expr::is_false(&coefficient, even)
        {
            return Ok(Some(Basic::zero()));
        }
        let Some(c) = coefficient.as_rational().cloned() else {
            let narg: Basic = mul(&[coefficient, pi()])?;
            return if narg != *arg { Ok(Some(cos(&narg)?)) } else { Ok(None) };
        };
        let q: Int = denom(&c);
        let p: Int = mod_floor(&numer(&c), &(&q * int(2)));
        if p > q {
            return Ok(Some(mul(&[Basic::negative_one(), cos(&times_pi(&(&c - Rat::ONE))?)?])?));
        }
        if &p * int(2) > q {
            return Ok(Some(mul(&[Basic::negative_one(), cos(&times_pi(&(Rat::ONE - &c))?)?])?));
        }
        if let Some((a, b)) = table2(&q) {
            let angle_a: Basic = times_pi(&rat_from_parts(p.clone(), int(a)))?;
            let angle_b: Basic = times_pi(&rat_from_parts(p.clone(), int(b)))?;
            let half_pi: Basic = mul(&[Basic::half(), pi()])?;
            let value_a: Basic = cos(&angle_a)?;
            let value_b: Basic = cos(&angle_b)?;
            let complement_a: Basic = cos(&add(&[half_pi.clone(), mul(&[Basic::negative_one(), angle_a])?])?)?;
            let complement_b: Basic = cos(&add(&[half_pi, mul(&[Basic::negative_one(), angle_b])?])?)?;
            if [&value_a, &value_b, &complement_a, &complement_b]
                .iter()
                .any(|v| v.is_function(Function::Cos))
            {
                return Ok(None);
            }
            return Ok(Some(add(&[
                mul(&[value_a, value_b])?,
                mul(&[complement_a, complement_b])?,
            ])?));
        }
        if q > int(12) {
            return Ok(None);
        }
        if q == int(3) {
            return Ok(Some(chebyshev_t(&numer(&c), &Basic::half())?));
        }
        if q == int(5) {
            let x: Basic = mul(&[
                add(&[pow(&Basic::int(5), &Basic::half())?, Basic::one()])?,
                Basic::from_rational(rat(1, 4)),
            ])?;
            return Ok(Some(chebyshev_t(&numer(&c), &x)?));
        }
        if mod_floor(&q, &int(2)).is_zero() {
            let doubled: Basic = cos(&times_pi(&(&c * rat(2, 1)))?)?;
            if doubled.is_function(Function::Cos) {
                return Ok(None);
            }
            let x: Rat = (&c * rat(2, 1) + Rat::ONE) / rat(2, 1);
            let magnitude: Rat = if rat_is_negative(&x) { -x.clone() } else { x.clone() };
            let truncated: Int = div_floor(&numer(&magnitude), &denom(&magnitude));
            let sign: i64 = if mod_floor(&truncated, &int(2)).is_zero() {
                1
            } else {
                -1
            };
            let root: Basic = pow(&mul(&[add(&[Basic::one(), doubled])?, Basic::half()])?, &Basic::half())?;
            return Ok(Some(mul(&[Basic::int(sign), root])?));
        }
        return Ok(None);
    }
    if arg.is_add() {
        let (rest, m) = peeloff_pi(arg)?;
        if !m.is_zero() {
            let multiple: Basic = times_pi(&m)?;
            return Ok(Some(add(&[
                mul(&[cos(&multiple)?, cos(&rest)?])?,
                mul(&[Basic::negative_one(), sin(&multiple)?, sin(&rest)?])?,
            ])?));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::one()));
    }
    Ok(None)
}

/// `tan(n*pi/10)` for `n = 1, ..., 4` (`_tan_table10`).
fn tan_table10(n: i64) -> PolyResult<Basic> {
    let five: Basic = pow(&Basic::int(5), &Basic::half())?;
    let (a, b): (i64, Rat) = match n {
        1 => (1, rat(-2, 5)),
        2 => (5, rat(-2, 1)),
        3 => (1, rat(2, 5)),
        _ => (5, rat(2, 1)),
    };
    pow(
        &add(&[Basic::int(a), mul(&[Basic::from_rational(b), five])?])?,
        &Basic::half(),
    )
}

/// `cos(narg)` and `cos(narg - pi/2)` when neither stays a node.
fn cos_and_shifted(narg: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let c_value: Basic = cos(narg)?;
    let shifted: Basic = add(&[narg.clone(), mul(&[Basic::from_rational(rat(-1, 2)), pi()])?])?;
    let s_value: Basic = cos(&shifted)?;
    if c_value.is_function(Function::Cos) || s_value.is_function(Function::Cos) {
        return Ok(None);
    }
    Ok(Some((c_value, s_value)))
}

fn tan_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(arg.head(), Head::NaN | Head::ComplexInfinity) {
        return Ok(Some(Basic::nan()));
    }
    if is_zero_value(arg) {
        return Ok(Some(Basic::zero()));
    }
    unsupported_infinite(arg)?;
    if could_extract_minus_sign(arg) {
        return Ok(Some(mul(&[
            Basic::negative_one(),
            tan(&mul(&[Basic::negative_one(), arg.clone()])?)?,
        ])?));
    }
    if imaginary_coefficient(arg)?.is_some() {
        return Err(not_ported("tanh"));
    }
    if let Some(coefficient) = pi_coeff(arg)? {
        if is_true(&coefficient, integer) {
            return Ok(Some(Basic::zero()));
        }
        let Some(c) = coefficient.as_rational().cloned() else {
            let narg: Basic = mul(&[coefficient, pi()])?;
            return if narg != *arg { Ok(Some(tan(&narg)?)) } else { Ok(None) };
        };
        let q: Int = denom(&c);
        let p: Int = mod_floor(&numer(&c), &q);
        if q == int(5) || q == int(10) {
            let n: i64 = i64::try_from(&div_floor(&(&p * int(10)), &q)).expect("a small index");
            if n > 5 {
                return Ok(Some(mul(&[Basic::negative_one(), tan_table10(10 - n)?])?));
            }
            return Ok(Some(tan_table10(n)?));
        }
        if mod_floor(&q, &int(2)).is_zero() {
            if let Some((c_value, s_value)) = cos_and_shifted(&times_pi(&(&c * rat(2, 1)))?)? {
                if is_zero_value(&s_value) {
                    return Ok(Some(Basic::zoo()));
                }
                let inverse: Basic = pow(&s_value, &Basic::negative_one())?;
                return Ok(Some(add(&[
                    inverse.clone(),
                    mul(&[Basic::negative_one(), c_value, inverse])?,
                ])?));
            }
        }
        if let Some((a, b)) = table2(&q) {
            let value_a: Basic = tan(&times_pi(&rat_from_parts(p.clone(), int(a)))?)?;
            let value_b: Basic = tan(&times_pi(&rat_from_parts(p.clone(), int(b)))?)?;
            if value_a.is_function(Function::Tan) || value_b.is_function(Function::Tan) {
                return Ok(None);
            }
            let numerator: Basic = add(&[value_a.clone(), mul(&[Basic::negative_one(), value_b.clone()])?])?;
            let denominator: Basic = add(&[Basic::one(), mul(&[value_a, value_b])?])?;
            return Ok(Some(mul(&[numerator, pow(&denominator, &Basic::negative_one())?])?));
        }
        let reduced: Rat = fmod(&(&c + rat(1, 2)), &Rat::ONE) - rat(1, 2);
        let narg: Basic = times_pi(&reduced)?;
        if let Some((c_value, s_value)) = cos_and_shifted(&narg)? {
            if is_zero_value(&c_value) {
                return Ok(Some(Basic::zoo()));
            }
            return Ok(Some(mul(&[s_value, pow(&c_value, &Basic::negative_one())?])?));
        }
        if narg != *arg {
            return Ok(Some(tan(&narg)?));
        }
        return Ok(None);
    }
    if arg.is_add() {
        let (rest, m) = peeloff_pi(arg)?;
        if !m.is_zero() {
            return Ok(Some(if m.is_int() {
                tan(&rest)?
            } else {
                mul(&[Basic::negative_one(), cot(&rest)?])?
            }));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::zero()));
    }
    Ok(None)
}

fn cot_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(arg.head(), Head::NaN | Head::ComplexInfinity) {
        return Ok(Some(Basic::nan()));
    }
    if is_zero_value(arg) {
        return Ok(Some(Basic::zoo()));
    }
    unsupported_infinite(arg)?;
    if could_extract_minus_sign(arg) {
        return Ok(Some(mul(&[
            Basic::negative_one(),
            cot(&mul(&[Basic::negative_one(), arg.clone()])?)?,
        ])?));
    }
    if imaginary_coefficient(arg)?.is_some() {
        return Err(not_ported("coth"));
    }
    if let Some(coefficient) = pi_coeff(arg)? {
        if is_true(&coefficient, integer) {
            return Ok(Some(Basic::zoo()));
        }
        let Some(c) = coefficient.as_rational().cloned() else {
            let narg: Basic = mul(&[coefficient, pi()])?;
            return if narg != *arg { Ok(Some(cot(&narg)?)) } else { Ok(None) };
        };
        let q: Int = denom(&c);
        if q == int(5) || q == int(10) {
            let half_pi: Basic = mul(&[Basic::half(), pi()])?;
            return Ok(Some(tan(&add(&[
                half_pi,
                mul(&[Basic::negative_one(), arg.clone()])?,
            ])?)?));
        }
        if q > int(2) && mod_floor(&q, &int(2)).is_zero() {
            if let Some((c_value, s_value)) = cos_and_shifted(&times_pi(&(&c * rat(2, 1)))?)? {
                let inverse: Basic = pow(&s_value, &Basic::negative_one())?;
                return Ok(Some(add(&[inverse.clone(), mul(&[c_value, inverse])?])?));
            }
        }
        let p: Int = mod_floor(&numer(&c), &q);
        if let Some((a, b)) = table2(&q) {
            let value_a: Basic = cot(&times_pi(&rat_from_parts(p.clone(), int(a)))?)?;
            let value_b: Basic = cot(&times_pi(&rat_from_parts(p.clone(), int(b)))?)?;
            if value_a.is_function(Function::Cot) || value_b.is_function(Function::Cot) {
                return Ok(None);
            }
            let numerator: Basic = add(&[Basic::one(), mul(&[value_a.clone(), value_b.clone()])?])?;
            let denominator: Basic = add(&[value_b, mul(&[Basic::negative_one(), value_a])?])?;
            return Ok(Some(mul(&[numerator, pow(&denominator, &Basic::negative_one())?])?));
        }
        let reduced: Rat = fmod(&(&c + rat(1, 2)), &Rat::ONE) - rat(1, 2);
        let narg: Basic = times_pi(&reduced)?;
        if let Some((c_value, s_value)) = cos_and_shifted(&narg)? {
            if is_zero_value(&s_value) {
                return Ok(Some(Basic::zoo()));
            }
            return Ok(Some(mul(&[c_value, pow(&s_value, &Basic::negative_one())?])?));
        }
        if narg != *arg {
            return Ok(Some(cot(&narg)?));
        }
        return Ok(None);
    }
    if arg.is_add() {
        let (rest, m) = peeloff_pi(arg)?;
        if !m.is_zero() {
            return Ok(Some(if m.is_int() {
                cot(&rest)?
            } else {
                mul(&[Basic::negative_one(), tan(&rest)?])?
            }));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::zoo()));
    }
    Ok(None)
}

/// `1/t` with `sec` for a cosine and `csc` for a sine (`_reciprocal_value`).
fn reciprocal_value(t: &Basic) -> PolyResult<Basic> {
    let negated: Basic = mul(&[Basic::negative_one(), t.clone()])?;
    for (partner, reciprocal) in [(Function::Cos, Function::Sec), (Function::Sin, Function::Csc)] {
        if t.is_function(partner) {
            return Ok(node(reciprocal, t.args()[0].clone()));
        }
        if negated.is_function(partner) {
            return mul(&[Basic::negative_one(), node(reciprocal, negated.args()[0].clone())]);
        }
    }
    pow(t, &Basic::negative_one())
}

/// The evaluation of `sec` and `csc` through that of the partner
/// (`ReciprocalTrigonometricFunction.eval`).
fn reciprocal_eval(f: Function, partner: Function, even_function: bool, arg: &Basic) -> PolyResult<Option<Basic>> {
    if could_extract_minus_sign(arg) {
        let flipped: Basic = apply(f, &mul(&[Basic::negative_one(), arg.clone()])?)?;
        return Ok(Some(if even_function {
            flipped
        } else {
            mul(&[Basic::negative_one(), flipped])?
        }));
    }
    if let Some(coefficient) = pi_coeff(arg)? {
        if let Some(c) = coefficient.as_rational().cloned() {
            if !is_true(&mul(&[Basic::int(2), coefficient])?, integer) {
                let q: Int = denom(&c);
                let p: Int = mod_floor(&numer(&c), &(&q * int(2)));
                if p > q {
                    return Ok(Some(mul(&[
                        Basic::negative_one(),
                        apply(f, &times_pi(&(&c - Rat::ONE))?)?,
                    ])?));
                }
                if &p * int(2) > q {
                    let narg: Basic = times_pi(&(Rat::ONE - &c))?;
                    let value: Basic = apply(f, &narg)?;
                    return Ok(Some(if even_function {
                        mul(&[Basic::negative_one(), value])?
                    } else {
                        value
                    }));
                }
            }
        }
    }
    match eval(partner, arg)? {
        Some(t) => Ok(Some(reciprocal_value(&t)?)),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------
// facts, conjugates, parts

/// The facts shared by the trigonometric functions (`_trig_facts`).
fn trig_facts(arg: &Basic) -> FactList {
    let a = facts_of(arg);
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    if a.get(&zero) == Some(&false) {
        if a.get(&rational) == Some(&true) {
            facts.set(rational, false);
        }
        if a.get(&algebraic) == Some(&true) {
            facts.set(algebraic, false);
        }
    }
    if let Ok(Some(c)) = pi_coeff(arg) {
        if is_true(&c, rational) {
            facts.set(algebraic, true);
        }
    }
    facts
}

/// The facts of a node of this module (`_eval_facts`).
pub fn trigonometric_facts(f: Function, arg: &Basic) -> FactList {
    let mut facts: FactList = trig_facts(arg);
    let a = facts_of(arg);
    let yes = |p: Predicate| a.get(&p) == Some(&true);
    match f {
        Function::Sin | Function::Cos => {
            if yes(extended_real) {
                facts.update(&[(extended_real, true), (finite, true)]);
            }
            if yes(real) {
                facts.set(real, true);
            }
            if yes(extended_real) || yes(complex) {
                facts.set(complex, true);
            }
        }
        Function::Tan => {
            if let Some(v) = a.get(&extended_real) {
                facts.set(extended_real, *v);
            }
            let off_pole: bool = mul(&[arg.clone(), pow(&pi(), &Basic::negative_one()).expect("1/pi")])
                .and_then(|q| add(&[q, Basic::from_rational(rat(-1, 2))]))
                .map(|shifted| crate::core::expr::is_false(&shifted, integer))
                .unwrap_or(false);
            if yes(real) && off_pole {
                facts.update(&[(real, true), (finite, true), (complex, true)]);
            }
            if yes(imaginary) {
                facts.set(finite, true);
            }
        }
        _ => {}
    }
    facts
}

/// The conjugate of a node of this module: the function of the conjugate.
pub fn trigonometric_conjugate(f: Function, arg: &Basic) -> PolyResult<Basic> {
    apply(f, &conjugate(arg)?)
}

/// The parts of the argument (`_argument_parts`), expanded when `deep`.
fn argument_parts(arg: &Basic, deep: bool) -> PolyResult<Option<RealImag>> {
    if is_true(arg, extended_real) {
        return Ok(Some((
            if deep { expand_default(arg)? } else { arg.clone() },
            Basic::zero(),
        )));
    }
    let expanded: Basic = if deep { expand_default(arg)? } else { arg.clone() };
    as_real_imag(&expanded, true, None)
}

/// The real and imaginary parts of a node of this module
/// (`_eval_as_real_imag`); the non-real arguments need the hyperbolic
/// functions.
pub fn trigonometric_real_imag(f: Function, arg: &Basic, deep: bool) -> PolyResult<Option<RealImag>> {
    match f {
        Function::Sec => return as_real_imag(&pow(&cos(arg)?, &Basic::negative_one())?, deep, None),
        Function::Csc => return as_real_imag(&pow(&sin(arg)?, &Basic::negative_one())?, deep, None),
        _ => {}
    }
    let Some((r, i)) = argument_parts(arg, deep)? else {
        return Ok(None);
    };
    if !is_rational_value(&i, 0) {
        return Err(not_ported("cosh and sinh of the imaginary part"));
    }
    // cosh(0) = 1, sinh(0) = 0
    Ok(Some(match f {
        Function::Sin => (sin(&r)?, Basic::zero()),
        Function::Cos => (cos(&r)?, Basic::zero()),
        Function::Tan => (tan(&r)?, Basic::zero()),
        Function::Cot => (cot(&r)?, Basic::zero()),
        _ => unreachable!("not a trigonometric function"),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    #[test]
    fn module_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let at = |c: Rat| times_pi(&c).unwrap();
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        assert_eq!(show(sin(&at(rat(1, 6)))), "1/2");
        assert_eq!(show(cos(&at(rat(1, 4)))), "sqrt(2)/2");
        assert_eq!(show(tan(&at(rat(1, 3)))), "sqrt(3)");
        assert_eq!(show(cos(&at(rat(1, 5)))), "1/4 + sqrt(5)/4");
        assert_eq!(show(sin(&mul(&[Basic::negative_one(), x.clone()]).unwrap())), "-sin(x)");
        assert_eq!(show(cos(&add(&[x.clone(), pi()]).unwrap())), "-cos(x)");
        assert_eq!(show(cot(&at(rat(1, 6)))), "sqrt(3)");
        assert_eq!(show(csc(&at(rat(1, 6)))), "2");
    }
}
