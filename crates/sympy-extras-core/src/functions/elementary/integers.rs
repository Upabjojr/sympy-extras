//! Port of `sympy_extras/functions/elementary/integers.py`: the rounding
//! functions [`Function::Floor`], [`Function::Ceiling`] and
//! [`Function::Frac`]: their evaluation (numbers, number symbols decided
//! numerically at 340 and 680 bits, the bounds from the signs of a real
//! argument, the integral, numerical and symbolic terms of a sum apart,
//! imaginary parts), facts and values. `frac` of an infinity is an
//! `AccumBounds`, not ported.

use crate::core::add::add;
use crate::core::assumptions::{facts_of, is_number, FactList, Facts, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, numeric_value, parts, Complex, Real};
use crate::core::expr::{as_numer_denom, is, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::error::{not_ported, PolyResult};
use crate::numbers::{denom, numer, Int, Rat};

use super::complexes::im;

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

/// The unevaluated node.
fn node(f: Function, arg: &Basic) -> Basic {
    Basic::new(Head::Function(f), vec![arg.clone()])
}

/// `e` as an integer when it is an integral number (`_integer_value`).
fn integer_value(e: &Basic) -> Option<Int> {
    let v: &Rat = e.as_rational()?;
    if denom(v) == Int::ONE {
        Some(numer(v))
    } else {
        None
    }
}

fn floor_of(x: &Real) -> Int {
    x.floor().to_int().value()
}

fn ceil_of(x: &Real) -> Int {
    x.ceil().to_int().value()
}

/// The floor (`direction < 0`) or ceiling of the real and imaginary parts
/// of the number `e`, from its values at 340 and 680 bits; `None` when a
/// part is too close to an integer (`_numerical_integer_part`).
fn numerical_integer_part(e: &Basic, direction: i32) -> Option<(Int, Int)> {
    let known_zero: [bool; 2] = [is_true(e, Predicate::imaginary), is_true(e, Predicate::extended_real)];
    let mut results: Vec<(Int, Int)> = Vec::new();
    for bits in [340usize, 680] {
        let value: Complex = numeric_value(e, bits)?;
        let (re, im_) = parts(&value);
        let mut found: Vec<Int> = Vec::new();
        for (component, zero) in [re, im_].into_iter().zip(known_zero) {
            if zero {
                found.push(Int::ZERO);
                continue;
            }
            let component: Real = component.with_precision(bits).value();
            let nearest: Real = component.round();
            let difference: Real = component.clone() - nearest;
            let distance: Real = if difference < Real::ZERO {
                -difference
            } else {
                difference
            };
            let tolerance: Real = Real::from_parts(dashu_int::IBig::ONE, -((bits / 2) as isize));
            if distance < tolerance {
                return None;
            }
            found.push(if direction < 0 {
                floor_of(&component)
            } else {
                ceil_of(&component)
            });
        }
        results.push((found[0].clone(), found[1].clone()));
    }
    if results[0] == results[1] {
        Some(results.remove(0))
    } else {
        None
    }
}

fn is_lt(a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
    Ok(is(&add(&[a.clone(), neg(b)?])?, Predicate::extended_negative))
}

fn is_le(a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
    Ok(is(&add(&[a.clone(), neg(b)?])?, Predicate::extended_nonpositive))
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

fn is_rounding(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(Function::Floor | Function::Ceiling))
}

/// `_eval_number`: numbers, floors and ceilings, number symbols.
fn round_number(arg: &Basic, direction: i32) -> PolyResult<Option<Basic>> {
    match arg.head() {
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity => return Ok(Some(arg.clone())),
        _ => {}
    }
    if let Some(v) = arg.as_rational() {
        // divmod: the quotient rounded down (the denominator is positive)
        let (p, q) = (numer(v), denom(v));
        let (mut quotient, remainder) = (&p / &q, &p % &q);
        if remainder != Int::ZERO && p < Int::ZERO {
            quotient -= Int::ONE;
        }
        let up: bool = direction > 0 && remainder != Int::ZERO;
        return Ok(Some(Basic::integer(if up { quotient + Int::ONE } else { quotient })));
    }
    if is_rounding(arg) || is_rounding(&neg(arg)?) {
        return Ok(Some(arg.clone()));
    }
    if matches!(
        arg.head(),
        Head::Pi | Head::Exp1 | Head::EulerGamma | Head::Catalan | Head::GoldenRatio | Head::TribonacciConstant
    ) {
        return Ok(numerical_integer_part(arg, direction).map(|(r, _)| Basic::integer(r)));
    }
    Ok(None)
}

/// `_eval_const_number`: the bounds decided from the signs of a real
/// argument.
fn round_constant(arg: &Basic, direction: i32) -> PolyResult<Option<Basic>> {
    if !is_true(arg, Predicate::real) {
        return Ok(None);
    }
    if is_true(arg, Predicate::zero) {
        return Ok(Some(Basic::zero()));
    }
    let positive: bool = is_true(arg, Predicate::positive);
    if !(positive || is_true(arg, Predicate::negative)) {
        return Ok(None);
    }
    let (mut num, mut den) = as_numer_denom(arg)?;
    match is(&den, Predicate::negative) {
        None => return Ok(None),
        Some(true) => {
            num = neg(&num)?;
            den = neg(&den)?;
        }
        Some(false) => {}
    }
    let two_den: Basic = mul(&[int(2), den.clone()])?;
    let minus_den: Basic = neg(&den)?;
    if positive {
        if direction < 0 {
            if is_lt(&num, &den)? == Some(true) {
                return Ok(Some(Basic::zero()));
            }
            if fuzzy_and(&[is_le(&den, &num)?, is_lt(&num, &two_den)?]) == Some(true) {
                return Ok(Some(int(1)));
            }
        } else {
            if is_le(&num, &den)? == Some(true) {
                return Ok(Some(int(1)));
            }
            if fuzzy_and(&[is_lt(&den, &num)?, is_le(&num, &two_den)?]) == Some(true) {
                return Ok(Some(int(2)));
            }
        }
    } else if direction < 0 {
        if is_le(&minus_den, &num)? == Some(true) {
            return Ok(Some(int(-1)));
        }
        if fuzzy_and(&[is_le(&neg(&two_den)?, &num)?, is_lt(&num, &minus_den)?]) == Some(true) {
            return Ok(Some(int(-2)));
        }
    } else {
        if is_lt(&minus_den, &num)? == Some(true) {
            return Ok(Some(Basic::zero()));
        }
        if fuzzy_and(&[is_lt(&neg(&two_den)?, &num)?, is_le(&num, &minus_den)?]) == Some(true) {
            return Ok(Some(int(-1)));
        }
    }
    Ok(None)
}

fn imaginary_like(e: &Basic) -> PolyResult<bool> {
    Ok(is_true(e, Predicate::imaginary) || is_true(&mul(&[Basic::imaginary_unit(), e.clone()])?, Predicate::real))
}

/// `RoundFunction.eval` (`_round_eval`).
fn round_eval(f: Function, arg: &Basic) -> PolyResult<Basic> {
    let direction: i32 = if f == Function::Floor { -1 } else { 1 };
    let i: Basic = Basic::imaginary_unit();
    if let Some(v) = round_number(arg, direction)? {
        return Ok(v);
    }
    if let Some(v) = round_constant(arg, direction)? {
        return Ok(v);
    }
    if is_true(arg, Predicate::integer) || crate::core::expr::is_false(arg, Predicate::finite) {
        return Ok(arg.clone());
    }
    if imaginary_like(arg)? {
        let imaginary: Basic = im(arg)?;
        if !imaginary.has(&i) {
            return mul(&[rounding(f, &imaginary)?, i]);
        }
        return Ok(node(f, arg));
    }
    let mut integral: Basic = Basic::zero();
    let mut numerical: Basic = Basic::zero();
    let mut symbolic: Basic = Basic::zero();
    let terms: Vec<Basic> = if arg.is_add() {
        arg.args().to_vec()
    } else {
        vec![arg.clone()]
    };
    for t in &terms {
        if is_true(t, Predicate::imaginary) {
            let imaginary: Basic = im(t)?;
            match integer_value(&imaginary) {
                None if is_true(&imaginary, Predicate::integer) => {
                    integral = add(&[integral, mul(&[imaginary, i.clone()])?])?;
                    continue;
                }
                Some(k) => {
                    integral = add(&[integral, mul(&[Basic::integer(k), i.clone()])?])?;
                    continue;
                }
                None => {}
            }
        }
        if let Some(k) = integer_value(t) {
            integral = add(&[integral, Basic::integer(k)])?;
        } else if is_true(t, Predicate::integer) {
            integral = add(&[integral, t.clone()])?;
        } else if is_number(t) {
            numerical = add(&[numerical, t.clone()])?;
        } else {
            symbolic = add(&[symbolic, t.clone()])?;
        }
    }
    let zero: Basic = Basic::zero();
    if numerical == zero && symbolic == zero {
        return Ok(integral);
    }
    if numerical != zero
        && (symbolic == zero
            || (is_true(&numerical, Predicate::real) && imaginary_like(&symbolic)?)
            || (is_true(&numerical, Predicate::imaginary) && is_true(&symbolic, Predicate::real)))
    {
        if let Some((r, m)) = numerical_integer_part(&numerical, direction) {
            integral = add(&[integral, Basic::integer(r), mul(&[Basic::integer(m), i.clone()])?])?;
            numerical = Basic::zero();
        }
    }
    symbolic = add(&[symbolic, numerical])?;
    if symbolic == zero {
        return Ok(integral);
    }
    if imaginary_like(&symbolic)? {
        return add(&[integral, mul(&[node(f, &im(&symbolic)?), i])?]);
    }
    if is_rounding(&symbolic) {
        return add(&[integral, symbolic]);
    }
    add(&[integral, node(f, &symbolic)])
}

/// `Frac.eval`.
fn frac_eval(arg: &Basic) -> PolyResult<Basic> {
    let i: Basic = Basic::imaginary_unit();
    let part = |value: &Basic| -> PolyResult<Basic> {
        if matches!(value.head(), Head::Infinity | Head::NegativeInfinity) {
            return Err(not_ported("AccumBounds (frac of an infinity)"));
        }
        if is_true(value, Predicate::integer) {
            return Ok(Basic::zero());
        }
        if is_number(value) {
            if matches!(value.head(), Head::NaN | Head::ComplexInfinity) {
                return Ok(Basic::nan());
            }
            return add(&[value.clone(), neg(&floor(value)?)?]);
        }
        Ok(node(Function::Frac, value))
    };
    let mut real: Basic = Basic::zero();
    let mut imaginary: Basic = Basic::zero();
    let terms: Vec<Basic> = if arg.is_add() {
        arg.args().to_vec()
    } else {
        vec![arg.clone()]
    };
    for t in &terms {
        if imaginary_like(t)? {
            let m: Basic = im(t)?;
            if !m.has(&i) {
                imaginary = add(&[imaginary, m])?;
            } else {
                real = add(&[real, t.clone()])?;
            }
        } else {
            real = add(&[real, t.clone()])?;
        }
    }
    let (real_part, imaginary_part) = (part(&real)?, part(&imaginary)?);
    add(&[real_part, mul(&[i, imaginary_part])?])
}

/// The evaluating function of `Floor`, `Ceiling` or `Frac`.
pub fn rounding(f: Function, arg: &Basic) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, arg));
    }
    match f {
        Function::Frac => frac_eval(arg),
        _ => round_eval(f, arg),
    }
}

/// `floor(x)`, evaluated.
pub fn floor(x: &Basic) -> PolyResult<Basic> {
    rounding(Function::Floor, x)
}

/// `ceiling(x)`, evaluated.
pub fn ceiling(x: &Basic) -> PolyResult<Basic> {
    rounding(Function::Ceiling, x)
}

/// `frac(x)`, evaluated.
pub fn frac(x: &Basic) -> PolyResult<Basic> {
    rounding(Function::Frac, x)
}

fn known(a: &Facts, p: Predicate) -> Option<bool> {
    a.get(&p).copied()
}

/// The facts of the node (`_eval_facts`).
pub fn rounding_facts(f: Function, arg: &Basic) -> FactList {
    use Predicate as P;
    let a: &Facts = facts_of(arg);
    let mut facts: FactList = FactList::new();
    facts.set(P::commutative, true);
    if f == Function::Frac {
        facts.set(P::finite, true);
        facts.set(P::negative, false);
        if let Some(v) = known(a, P::extended_real) {
            facts.set(P::real, v);
        }
        if let Some(v) = known(a, P::imaginary) {
            facts.set(P::imaginary, v);
        }
        if let Some(v) = known(a, P::integer) {
            facts.set(P::integer, v);
        }
        if known(a, P::zero) == Some(true) || known(a, P::integer) == Some(true) {
            facts.set(P::zero, true);
        } else if known(a, P::zero) == Some(false) && known(a, P::integer) == Some(false) {
            facts.set(P::zero, false);
        }
        return facts;
    }
    if let Some(v) = known(a, P::finite) {
        facts.set(P::finite, v);
    }
    if let Some(v) = known(a, P::real) {
        facts.set(P::real, v);
        facts.set(P::integer, v);
    }
    let kept: [Predicate; 2] = if f == Function::Floor {
        [P::negative, P::nonnegative]
    } else {
        [P::positive, P::nonpositive]
    };
    for p in kept {
        if let Some(v) = known(a, p) {
            facts.set(p, v);
        }
    }
    facts
}

/// The value (`numeric`): the floor, ceiling or fractional part of the
/// real and imaginary parts.
pub fn rounding_value(f: Function, z: &Complex) -> Complex {
    let (re, im_) = parts(z);
    match f {
        Function::Floor => complex(re.floor(), im_.floor()),
        Function::Ceiling => complex(re.ceil(), im_.ceil()),
        _ => complex(re.clone() - re.floor(), im_.clone() - im_.floor()),
    }
}
