//! The handlers of the predicates by class of the argument and the last
//! steps of the core's `ask`, ported from
//! `sympy_extras.assumptions.handlers` (itself a typed port of SymPy's
//! `assumptions/handlers/{sets,order,ntheory,common}.py`): [`handler_ask`]
//! answers by the rules of the class of the argument, asking questions
//! about its arguments under the same assumptions; [`final_ask`] tries the
//! handlers, the scaled literal (the frequent case of SymPy's
//! `lra_satask`), then the SAT search of `satask`. The numbers are evaluated by `reliable_complex` at 15 digits.

use std::cell::RefCell;

use crate::core::ask::{ask, q_applied, Truth};
use crate::core::basic::{Basic, Function, Head};
use crate::core::logic::{and_, not_, or_};
use crate::error::{PolyError, PolyResult};
use crate::numbers::Int;

type R = PolyResult<Truth>;

fn qa(name: &str, e: &Basic) -> PolyResult<Basic> {
    q_applied(name, std::slice::from_ref(e))
}

/// The core's `ask` without the global assumptions (`_ask`).
fn ask_in(proposition: &Basic, assumptions: &Basic) -> R {
    ask(proposition, assumptions, Some(&[]))
}

fn askq(name: &str, e: &Basic, assumptions: &Basic) -> R {
    ask_in(&qa(name, e)?, assumptions)
}

/// The core's `ask` with the global assumptions only (`_ask_global`).
fn ask_global(proposition: &Basic) -> R {
    ask(proposition, &crate::core::logic::boolean(true), None)
}

fn is_yes(v: Truth) -> bool {
    v == Some(true)
}

fn fnot(v: Truth) -> Truth {
    v.map(|x| !x)
}

/// `fuzzy_and` of eagerly computed values.
fn fand(values: &[Truth]) -> Truth {
    let mut rv: Truth = Some(true);
    for a in values {
        match a {
            None => {
                if rv == Some(true) {
                    rv = None;
                }
            }
            Some(false) => return Some(false),
            Some(true) => {}
        }
    }
    rv
}

fn f_or(values: &[Truth]) -> Truth {
    fnot(fand(&values.iter().map(|v| fnot(*v)).collect::<Vec<Truth>>()))
}

/// `fuzzy_or` of lazily computed values (a generator in the Python).
fn f_or_lazy(values: impl IntoIterator<Item = R>) -> R {
    let mut rv: Truth = Some(true);
    for a in values {
        match fnot(a?) {
            None => {
                if rv == Some(true) {
                    rv = None;
                }
            }
            Some(false) => return Ok(Some(true)),
            Some(true) => {}
        }
    }
    Ok(fnot(rv))
}

fn sub_one(e: &Basic) -> PolyResult<Basic> {
    crate::core::add::sub(e, &Basic::one())
}

fn add_one(e: &Basic) -> PolyResult<Basic> {
    crate::core::add::add(&[e.clone(), Basic::one()])
}

fn times(n: i64, e: &Basic) -> PolyResult<Basic> {
    crate::core::mul::mul(&[Basic::int(n), e.clone()])
}

fn over_i_pi(x: &Basic) -> PolyResult<Basic> {
    crate::core::mul::div(&crate::core::mul::div(x, &Basic::imaginary_unit())?, &Basic::pi())
}

fn is_number(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}

fn is_rational_node(e: &Basic) -> bool {
    matches!(e.head(), Head::Rational { .. })
}

fn is_integer_node(e: &Basic) -> bool {
    matches!(e.head(), Head::Rational { value, .. } if value.is_int())
}

fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}

/// `x` for `exp(x)` and `E**x`.
fn exp_argument(e: &Basic) -> Option<Basic> {
    if func(e) == Some(Function::Exp) {
        return Some(e.args()[0].clone());
    }
    if e.is_pow() && matches!(e.args()[0].head(), Head::Exp1) {
        return Some(e.args()[1].clone());
    }
    None
}

fn value(e: &Basic) -> Option<(f64, f64)> {
    crate::numeric::reliable_complex(e, 15, &[])
}

fn magnitude(z: (f64, f64)) -> f64 {
    z.0.hypot(z.1)
}

/// Whether a part of the value is zero (`_is_zero_part`).
fn is_zero_part(part: f64, z: (f64, f64)) -> Truth {
    if part == 0.0 {
        return Some(true);
    }
    if part.abs() > 1e-12 * 1f64.max(magnitude(z)) {
        return Some(false);
    }
    None
}

fn sign_of_number(e: &Basic, positive: bool) -> Truth {
    let z = value(e)?;
    if magnitude(z) < 1e-10 {
        return if crate::numeric::proved_zero(e) {
            Some(false)
        } else {
            None
        };
    }
    match is_zero_part(z.1, z) {
        Some(true) => Some(if positive { z.0 > 0.0 } else { z.0 < 0.0 }),
        Some(false) => Some(false),
        None => None,
    }
}

fn real_number(e: &Basic) -> Truth {
    let z = value(e)?;
    is_zero_part(z.1, z)
}

fn imaginary_number(e: &Basic) -> Truth {
    let z = value(e)?;
    is_zero_part(z.0, z)
}

/// Python's `round`: to the nearest integer, ties to even.
fn round_even(x: f64) -> f64 {
    let r: f64 = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `True` when the number is certainly not an integer.
fn not_integer_number(e: &Basic) -> Truth {
    let z = value(e)?;
    if is_zero_part(z.1, z) == Some(false) {
        return Some(true);
    }
    Some((z.0 - round_even(z.0)).abs() > 1e-10 * 1f64.max(magnitude(z)))
}

/// The integer the number equals, proved by its minimal polynomial.
fn proved_integer(e: &Basic) -> PolyResult<Option<Int>> {
    let Some(z) = value(e) else { return Ok(None) };
    if is_zero_part(z.1, z) == Some(false) {
        return Ok(None);
    }
    let r: f64 = round_even(z.0);
    let i: Int = crate::core::evalf::Real::try_from(r)
        .map(|x| x.to_int().value())
        .unwrap_or(Int::ZERO);
    Ok(
        if crate::numeric::proved_zero(&crate::core::add::sub(e, &Basic::integer(i.clone()))?) {
            Some(i)
        } else {
            None
        },
    )
}

fn integer_of_number(e: &Basic) -> R {
    if is_yes(not_integer_number(e)) {
        return Ok(Some(false));
    }
    Ok(if proved_integer(e)?.is_some() { Some(true) } else { None })
}

/// `test_closed_group` (lazy, quick exit).
fn closed_group(e: &Basic, assumptions: &Basic, name: &str) -> R {
    let mut saw_other: bool = false;
    for a in e.args() {
        match askq(name, a, assumptions)? {
            Some(true) => continue,
            None => return Ok(None),
            Some(false) => {
                if saw_other {
                    return Ok(None);
                }
                saw_other = true;
            }
        }
    }
    Ok(Some(!saw_other))
}

fn is_named_constant(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Exp1
            | Head::GoldenRatio
            | Head::ImaginaryUnit
            | Head::Infinity
            | Head::NegativeInfinity
            | Head::Pi
            | Head::TribonacciConstant
    )
}

// -- sets.py

fn integer(e: &Basic, a: &Basic) -> R {
    if is_integer_node(e) {
        return Ok(Some(true));
    }
    if is_named_constant(e) || is_rational_node(e) {
        return Ok(Some(false));
    }
    if e.is_add() {
        if is_number(e) {
            return integer_of_number(e);
        }
        return closed_group(e, a, "integer");
    }
    if e.is_pow() {
        if is_number(e) {
            return integer_of_number(e);
        }
        let (base, exp) = (&e.args()[0], &e.args()[1]);
        if is_yes(fand(&[
            ask_in(&not_(&qa("zero", base)?), a)?,
            askq("finite", base, a)?,
            askq("zero", exp, a)?,
        ])) {
            return Ok(Some(true));
        }
        if is_yes(fand(&[askq("integer", base, a)?, askq("integer", exp, a)?]))
            && is_yes(f_or(&[
                askq("positive", exp, a)?,
                ask_in(&and_(&[qa("nonnegative", exp)?, not_(&qa("zero", base)?)]), a)?,
                askq("zero", &sub_one(base)?, a)?,
                askq("zero", &add_one(base)?, a)?,
            ]))
        {
            return Ok(Some(true));
        }
        return Ok(None);
    }
    if e.is_mul() {
        if is_number(e) {
            return integer_of_number(e);
        }
        let mut output: bool = true;
        for arg in e.args() {
            if !is_yes(askq("integer", arg, a)?) {
                if is_rational_node(arg) {
                    if *arg.as_rational().expect("a rational").denominator() == dashu_int::UBig::from(2u8) {
                        return askq("even", &times(2, e)?, a);
                    }
                    return Ok(None);
                } else if is_yes(askq("irrational", arg, a)?) {
                    if output {
                        output = false;
                    } else {
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            }
        }
        if !output {
            // an integer times an irrational number is 0 when the integer
            // is (SymPy answers False, issue #25)
            return zero_factor(e.args(), a);
        }
        return Ok(Some(output));
    }
    if func(e) == Some(Function::Abs) {
        return Ok(if is_yes(askq("integer", &e.args()[0], a)?) {
            Some(true)
        } else {
            None
        });
    }
    Ok(None)
}

/// Whether one of the factors is zero: `true` when one is, `false` when
/// none is, `None` otherwise (`_zero_factor`).
fn zero_factor(factors: &[Basic], a: &Basic) -> R {
    let mut answers: Vec<Option<bool>> = Vec::new();
    for f in factors {
        answers.push(askq("zero", f, a)?);
    }
    if answers.contains(&Some(true)) {
        return Ok(Some(true));
    }
    if answers.iter().all(|x| *x == Some(false)) {
        return Ok(Some(false));
    }
    Ok(None)
}

fn rational(e: &Basic, a: &Basic) -> R {
    if is_rational_node(e) {
        return Ok(Some(true));
    }
    if is_named_constant(e) {
        return Ok(Some(false));
    }
    if e.is_add() || e.is_mul() {
        if is_number(e) {
            if let Some(z) = value(e) {
                if is_zero_part(z.1, z) == Some(false) {
                    return Ok(Some(false));
                }
            }
        }
        let rational: Option<bool> = closed_group(e, a, "rational")?;
        if rational == Some(false) && e.is_mul() {
            // a rational number times an irrational one is 0 when the
            // rational number is (SymPy answers False, issue #25)
            return zero_factor(e.args(), a);
        }
        return Ok(rational);
    }
    if let Some(x) = exp_argument(e) {
        if is_yes(askq("rational", &x, a)?) {
            return ask_in(&not_(&qa("nonzero", &x)?), a);
        }
        return Ok(None);
    }
    if e.is_pow() {
        let (base, exp) = (&e.args()[0], &e.args()[1]);
        let is_exp_integer: Truth = askq("integer", exp, a)?;
        if is_yes(is_exp_integer) {
            if is_yes(askq("rational", base, a)?) {
                let is_base_zero: Truth = askq("zero", base, a)?;
                if is_base_zero == Some(false) {
                    return Ok(Some(true));
                }
                if is_yes(is_base_zero) && is_yes(ask_global(&qa("positive", exp)?)?) {
                    return Ok(Some(true));
                }
            }
            if askq("algebraic", base, a)? == Some(false) {
                return askq("zero", exp, a);
            }
            if is_yes(askq("irrational", base, a)?)
                && is_yes(ask_global(&q_applied("eq", &[exp.clone(), Basic::int(-1)])?)?)
            {
                return Ok(Some(false));
            }
            return Ok(None);
        }
        if is_yes(askq("rational", exp, a)?) {
            if is_yes(askq("prime", base, a)?) && is_exp_integer == Some(false) {
                return Ok(Some(false));
            }
            if is_yes(ask_global(&qa("zero", base)?)?) && is_yes(ask_global(&qa("positive", exp)?)?) {
                return Ok(Some(true));
            }
            if is_yes(ask_global(&q_applied("eq", &[base.clone(), Basic::one()])?)?) {
                return Ok(Some(true));
            }
        }
        return Ok(None);
    }
    match func(e) {
        Some(Function::Asin | Function::Atan | Function::Cos | Function::Sin | Function::Tan) => {
            let x: &Basic = &e.args()[0];
            if is_yes(askq("rational", x, a)?) {
                return ask_in(&not_(&qa("nonzero", x)?), a);
            }
            Ok(None)
        }
        Some(Function::Acot | Function::Cot) => Ok(if is_yes(askq("rational", &e.args()[0], a)?) {
            Some(false)
        } else {
            None
        }),
        Some(Function::Acos | Function::Log) => {
            let x: &Basic = &e.args()[0];
            if is_yes(askq("rational", x, a)?) {
                return ask_in(&not_(&qa("nonzero", &sub_one(x)?)?), a);
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn irrational(e: &Basic, a: &Basic) -> R {
    let real: Truth = askq("real", e, a)?;
    if is_yes(real) {
        let rational: Truth = askq("rational", e, a)?;
        return Ok(rational.map(|r| !r));
    }
    Ok(real)
}

fn real(e: &Basic, a: &Basic) -> R {
    if matches!(func(e), Some(Function::Abs | Function::Im | Function::Re))
        || matches!(
            e.head(),
            Head::Exp1 | Head::GoldenRatio | Head::Pi | Head::TribonacciConstant
        )
        || is_rational_node(e)
    {
        return Ok(Some(true));
    }
    if matches!(e.head(), Head::ImaginaryUnit | Head::Infinity | Head::NegativeInfinity) {
        return Ok(Some(false));
    }
    if e.is_add() {
        if is_number(e) {
            return Ok(real_number(e));
        }
        return closed_group(e, a, "real");
    }
    if e.is_mul() {
        if is_number(e) {
            return Ok(real_number(e));
        }
        let mut result: bool = true;
        let mut reals: Vec<Basic> = Vec::new();
        for arg in e.args() {
            if is_yes(askq("real", arg, a)?) {
                reals.push(arg.clone());
            } else if is_yes(askq("imaginary", arg, a)?) {
                result = !result;
            } else {
                return Ok(None);
            }
        }
        if !result {
            // SymPy answers False without asking whether a real factor is
            // zero: x*y is 0, real, at x = 0 for a real x and an imaginary y
            // (issue #25)
            return zero_factor(&reals, a);
        }
        return Ok(Some(result));
    }
    if let Some(x) = exp_argument(e) {
        return ask_in(&or_(&[qa("integer", &over_i_pi(&x)?)?, qa("real", &x)?]), a);
    }
    if e.is_pow() {
        return real_pow(e, a);
    }
    match func(e) {
        Some(Function::Cos | Function::Sin) => Ok(if is_yes(askq("real", &e.args()[0], a)?) {
            Some(true)
        } else {
            None
        }),
        // not in SymPy's handlers: atan maps the extended reals onto [-pi/2, pi/2]
        Some(Function::Atan) => Ok(if is_yes(askq("extended_real", &e.args()[0], a)?) {
            Some(true)
        } else {
            None
        }),
        Some(Function::Log) => askq("positive", &e.args()[0], a),
        _ => Ok(None),
    }
}

fn minus_one_power(i: &Basic, exp: &Basic) -> PolyResult<Basic> {
    crate::core::power::pow(&crate::core::power::pow(&Basic::negative_one(), i)?, exp)
}

fn real_pow(e: &Basic, a: &Basic) -> R {
    if is_number(e) {
        return Ok(real_number(e));
    }
    let (base, exp) = (&e.args()[0], &e.args()[1]);
    if let Some(inner) = exp_argument(base) {
        if is_yes(askq("imaginary", &inner, a)?) && is_yes(askq("imaginary", exp, a)?) {
            return Ok(Some(true));
        }
        let i: Basic = over_i_pi(&inner)?;
        if is_yes(askq("integer", &times(2, &i)?, a)?) {
            return askq("real", &minus_one_power(&i, exp)?, a);
        }
        return Ok(None);
    }
    if is_yes(askq("imaginary", base, a)?) && is_yes(askq("integer", exp, a)?) {
        let odd: Truth = askq("odd", exp, a)?;
        return Ok(odd.map(|o| !o));
    }
    if is_yes(askq("imaginary", exp, a)?) {
        let imlog: Truth = askq("imaginary", &crate::functions::elementary::exponential::log(base)?, a)?;
        if imlog.is_some() {
            return Ok(imlog);
        }
    }
    if is_yes(askq("real", base, a)?) && is_yes(askq("real", exp, a)?) {
        let zero: Truth = askq("zero", base, a)?;
        if zero == Some(true) {
            return Ok(if is_yes(askq("positive", exp, a)?) {
                Some(true)
            } else {
                None
            });
        }
        if zero.is_none() {
            return Ok(None);
        }
        if let Some(r) = exp.as_rational() {
            if (r.denominator() % dashu_int::UBig::from(2u8)).is_zero() {
                return askq("positive", base, a);
            }
        }
        if is_yes(askq("integer", exp, a)?) {
            return Ok(Some(true));
        }
        if is_yes(askq("positive", base, a)?) {
            return Ok(Some(true));
        }
    }
    Ok(None)
}

fn extended_real(e: &Basic, a: &Basic) -> R {
    if matches!(e.head(), Head::Infinity | Head::NegativeInfinity) {
        return Ok(Some(true));
    }
    if e.is_add() || e.is_mul() || e.is_pow() {
        return closed_group(e, a, "extended_real");
    }
    Ok(None)
}

fn is_number_symbol(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Pi | Head::Exp1 | Head::EulerGamma | Head::Catalan | Head::GoldenRatio | Head::TribonacciConstant
    )
}

fn complex(e: &Basic, a: &Basic) -> R {
    if matches!(
        func(e),
        Some(
            Function::Abs | Function::Cos | Function::Exp | Function::Im | Function::Log | Function::Re | Function::Sin
        )
    ) || matches!(e.head(), Head::ImaginaryUnit)
        || e.is_number_class()
        || is_number_symbol(e)
    {
        if matches!(e.head(), Head::Infinity | Head::NegativeInfinity | Head::NaN) {
            return Ok(Some(false));
        }
        return Ok(Some(true));
    }
    if e.is_add() || e.is_mul() {
        return closed_group(e, a, "complex");
    }
    if e.is_pow() {
        if matches!(e.args()[0].head(), Head::Exp1) {
            return Ok(Some(true));
        }
        return closed_group(e, a, "complex");
    }
    Ok(None)
}

fn imaginary(e: &Basic, a: &Basic) -> R {
    if matches!(e.head(), Head::ImaginaryUnit) {
        return Ok(Some(true));
    }
    if matches!(e.head(), Head::NaN) {
        return Ok(None);
    }
    if e.is_number_class() || is_number_symbol(e) {
        let Some(z) = value(e) else { return Ok(None) };
        return Ok(fnot(is_zero_part(z.1, z)));
    }
    if e.is_add() {
        if is_number(e) {
            return Ok(imaginary_number(e));
        }
        let mut reals: usize = 0;
        for arg in e.args() {
            if is_yes(askq("imaginary", arg, a)?) {
            } else if is_yes(askq("real", arg, a)?) {
                reals += 1;
            } else {
                return Ok(None);
            }
        }
        if reals == 0 {
            return Ok(Some(true));
        }
        if reals == 1 || reals == e.args().len() {
            return Ok(Some(false));
        }
        return Ok(None);
    }
    if e.is_mul() {
        if is_number(e) {
            return Ok(imaginary_number(e));
        }
        let mut result: bool = false;
        let mut real_factors: Vec<Basic> = Vec::new();
        for arg in e.args() {
            if is_yes(askq("imaginary", arg, a)?) {
                result = !result;
            } else if !is_yes(askq("real", arg, a)?) {
                return Ok(None);
            } else {
                real_factors.push(arg.clone());
            }
        }
        if result {
            // as for Q.real: 0 = x*y at x = 0 is not imaginary (issue #25)
            let zero: Option<bool> = zero_factor(&real_factors, a)?;
            return Ok(zero.map(|z| !z));
        }
        return Ok(Some(result));
    }
    if let Some(x) = exp_argument(e) {
        let t: Basic = over_i_pi(&x)?;
        return ask_in(&and_(&[qa("integer", &times(2, &t)?)?, not_(&qa("integer", &t)?)]), a);
    }
    if e.is_pow() {
        return imaginary_pow(e, a);
    }
    if func(e) == Some(Function::Log) {
        let x: &Basic = &e.args()[0];
        if is_yes(askq("real", x, a)?) {
            return Ok(if is_yes(askq("positive", x, a)?) {
                Some(false)
            } else {
                None
            });
        }
        if let Some(inner) = exp_argument(x) {
            let minus_i: Basic = crate::core::mul::neg(&Basic::imaginary_unit())?;
            if inner == Basic::imaginary_unit() || inner == minus_i {
                return Ok(Some(true));
            }
        }
        if askq("imaginary", x, a)? == Some(false) {
            return Ok(Some(false));
        }
        return Ok(None);
    }
    Ok(None)
}

fn imaginary_pow(e: &Basic, a: &Basic) -> R {
    if is_number(e) {
        return Ok(imaginary_number(e));
    }
    let (base, exp) = (&e.args()[0], &e.args()[1]);
    if let Some(inner) = exp_argument(base) {
        if is_yes(askq("imaginary", &inner, a)?) {
            if is_yes(askq("imaginary", exp, a)?) {
                return Ok(Some(false));
            }
            let i: Basic = over_i_pi(&inner)?;
            if is_yes(askq("integer", &times(2, &i)?, a)?) {
                return askq("imaginary", &minus_one_power(&i, exp)?, a);
            }
        }
    }
    if is_yes(askq("imaginary", base, a)?) && is_yes(askq("integer", exp, a)?) {
        return askq("odd", exp, a);
    }
    if is_yes(askq("imaginary", exp, a)?)
        && askq("imaginary", &crate::functions::elementary::exponential::log(base)?, a)?.is_some()
    {
        return Ok(Some(false));
    }
    if is_yes(ask_in(&and_(&[qa("real", base)?, qa("real", exp)?]), a)?) {
        if is_yes(askq("positive", base, a)?) {
            return Ok(Some(false));
        }
        let rat: Truth = askq("rational", exp, a)?;
        if !is_yes(rat) {
            return Ok(rat);
        }
        if is_yes(askq("integer", exp, a)?) {
            return Ok(Some(false));
        }
        let half: Truth = askq("integer", &times(2, exp)?, a)?;
        if is_yes(half) {
            return askq("negative", base, a);
        }
        return Ok(half);
    }
    Ok(None)
}

// -- order.py

fn negative(e: &Basic, a: &Basic) -> R {
    if func(e) == Some(Function::Abs) || matches!(e.head(), Head::ImaginaryUnit) {
        return Ok(Some(false));
    }
    if let Some(x) = exp_argument(e) {
        return Ok(if is_yes(askq("real", &x, a)?) {
            Some(false)
        } else {
            None
        });
    }
    if is_number(e) {
        return Ok(sign_of_number(e, false));
    }
    if func(e) == Some(Function::Atan) {
        // not in SymPy: atan is odd and increasing, the mirror of the
        // positive handler
        return askq("extended_negative", &e.args()[0], a);
    }
    if e.is_add() {
        let r: Truth = askq("real", e, a)?;
        if r != Some(true) {
            return Ok(r);
        }
        let mut nonpos: usize = 0;
        for arg in e.args() {
            if askq("negative", arg, a)? != Some(true) {
                if askq("positive", arg, a)? == Some(false) {
                    nonpos += 1;
                } else {
                    return Ok(None);
                }
            }
        }
        return Ok(if nonpos < e.args().len() { Some(true) } else { None });
    }
    if e.is_mul() {
        let mut result: Truth = None;
        for arg in e.args() {
            if result.is_none() {
                result = Some(false);
            }
            if is_yes(askq("negative", arg, a)?) {
                result = result.map(|r| !r);
            } else if is_yes(askq("positive", arg, a)?) {
            } else {
                return Ok(None);
            }
        }
        return Ok(result);
    }
    if e.is_pow() {
        let (base, exp) = (&e.args()[0], &e.args()[1]);
        if is_yes(askq("real", base, a)?) {
            if is_yes(askq("positive", base, a)?) && is_yes(askq("real", exp, a)?) {
                return Ok(Some(false));
            }
            if is_yes(askq("even", exp, a)?) {
                return Ok(Some(false));
            }
            if is_yes(askq("odd", exp, a)?) {
                return askq("negative", base, a);
            }
        }
        return Ok(None);
    }
    Ok(None)
}

fn nonnegative(e: &Basic, a: &Basic) -> R {
    if is_number(e) {
        let notnegative: Truth = fnot(sign_of_number(e, false));
        if is_yes(notnegative) {
            return askq("real", e, a);
        }
        return Ok(notnegative);
    }
    Ok(None)
}

fn nonpositive(e: &Basic, a: &Basic) -> R {
    if is_number(e) {
        let notpositive: Truth = fnot(sign_of_number(e, true));
        if is_yes(notpositive) {
            return askq("real", e, a);
        }
        return Ok(notpositive);
    }
    Ok(None)
}

fn nonzero(e: &Basic, a: &Basic) -> R {
    if matches!(e.head(), Head::NaN) {
        return Ok(None);
    }
    if is_number(e) && crate::numeric::proved_zero(e) {
        return Ok(Some(false));
    }
    if e.is_add() {
        let mut all_positive: bool = true;
        for arg in e.args() {
            if !is_yes(askq("positive", arg, a)?) {
                all_positive = false;
                break;
            }
        }
        if all_positive {
            return Ok(Some(true));
        }
        let mut all_negative: bool = true;
        for arg in e.args() {
            if !is_yes(askq("negative", arg, a)?) {
                all_negative = false;
                break;
            }
        }
        return Ok(if all_negative { Some(true) } else { None });
    }
    if e.is_mul() {
        for arg in e.args() {
            let result: Truth = askq("nonzero", arg, a)?;
            if is_yes(result) {
                continue;
            }
            return Ok(result);
        }
        return Ok(Some(true));
    }
    if e.is_pow() {
        return askq("nonzero", &e.args()[0], a);
    }
    if func(e) == Some(Function::Abs) {
        return askq("nonzero", &e.args()[0], a);
    }
    if ask_global(&qa("real", e)?)? == Some(false) {
        return Ok(Some(false));
    }
    if is_number(e) {
        let Some(z) = value(e) else { return Ok(None) };
        return Ok(fnot(fand(&[is_zero_part(z.0, z), is_zero_part(z.1, z)])));
    }
    Ok(None)
}

fn zero(e: &Basic, a: &Basic) -> R {
    if is_number(e) && crate::numeric::proved_zero(e) {
        return Ok(Some(true));
    }
    if e.is_mul() {
        return f_or_lazy(e.args().iter().map(|arg| askq("zero", arg, a)));
    }
    Ok(fand(&[fnot(askq("nonzero", e, a)?), askq("real", e, a)?]))
}

/// Whether `x` is known to be real with `x > 1` or `x < -1`
/// (`_outside_unit_interval`).
fn outside_unit_interval(x: &Basic, a: &Basic) -> PolyResult<bool> {
    Ok(is_yes(askq("positive", &sub_one(x)?, a)?) || is_yes(askq("negative", &add_one(x)?, a)?))
}

fn positive(e: &Basic, a: &Basic) -> R {
    if matches!(e.head(), Head::ImaginaryUnit) {
        return Ok(Some(false));
    }
    if matches!(e.head(), Head::NaN) {
        return Ok(None);
    }
    if let Some(x) = exp_argument(e) {
        if is_yes(askq("real", &x, a)?) {
            return Ok(Some(true));
        }
        if is_yes(askq("imaginary", &x, a)?) {
            let t: Basic = crate::core::mul::div(&x, &crate::core::mul::mul(&[Basic::imaginary_unit(), Basic::pi()])?)?;
            return askq("even", &t, a);
        }
        return Ok(None);
    }
    if is_number(e) {
        return Ok(sign_of_number(e, true));
    }
    if e.is_mul() {
        let mut result: bool = true;
        for arg in e.args() {
            if is_yes(askq("positive", arg, a)?) {
                continue;
            }
            if is_yes(askq("negative", arg, a)?) {
                result = !result;
            } else {
                return Ok(None);
            }
        }
        return Ok(Some(result));
    }
    if e.is_add() {
        let r: Truth = askq("real", e, a)?;
        if r != Some(true) {
            return Ok(r);
        }
        let mut nonneg: usize = 0;
        for arg in e.args() {
            if askq("positive", arg, a)? != Some(true) {
                if askq("negative", arg, a)? == Some(false) {
                    nonneg += 1;
                } else {
                    return Ok(None);
                }
            }
        }
        return Ok(if nonneg < e.args().len() { Some(true) } else { None });
    }
    if e.is_pow() {
        let (base, exp) = (&e.args()[0], &e.args()[1]);
        if is_yes(askq("positive", base, a)?) && is_yes(askq("real", exp, a)?) {
            return Ok(Some(true));
        }
        if is_yes(askq("negative", base, a)?) {
            if is_yes(askq("even", exp, a)?) {
                return Ok(Some(true));
            }
            if is_yes(askq("odd", exp, a)?) {
                return Ok(Some(false));
            }
        }
        return Ok(None);
    }
    match func(e) {
        Some(Function::Log) => {
            let arg: &Basic = &e.args()[0];
            let r: Truth = askq("real", arg, a)?;
            if r != Some(true) {
                return Ok(r);
            }
            if is_yes(askq("positive", &sub_one(arg)?, a)?) {
                return Ok(Some(true));
            }
            if is_yes(askq("nonpositive", &sub_one(arg)?, a)?) {
                // log(1) = 0
                return Ok(Some(false));
            }
            Ok(None)
        }
        Some(Function::Factorial) => {
            let arg: &Basic = &e.args()[0];
            Ok(
                if is_yes(ask_in(&and_(&[qa("integer", arg)?, qa("positive", arg)?]), a)?) {
                    Some(true)
                } else {
                    None
                },
            )
        }
        Some(Function::Abs) => askq("nonzero", e, a),
        // extended_positive, not positive: atan(oo) = pi/2
        Some(Function::Atan) => askq("extended_positive", &e.args()[0], a),
        Some(Function::Asin) => {
            let arg: &Basic = &e.args()[0];
            if is_yes(ask_in(
                &and_(&[qa("positive", arg)?, qa("nonpositive", &sub_one(arg)?)?]),
                a,
            )?) {
                return Ok(Some(true));
            }
            if is_yes(ask_in(
                &and_(&[qa("nonpositive", arg)?, qa("nonnegative", &add_one(arg)?)?]),
                a,
            )?) {
                return Ok(Some(false));
            }
            if outside_unit_interval(arg, a)? {
                // asin of a real number off [-1, 1] is not real
                return Ok(Some(false));
            }
            Ok(None)
        }
        Some(Function::Acos) => {
            // acos is positive on [-1, 1) but acos(1) = 0
            let arg: &Basic = &e.args()[0];
            if is_yes(ask_in(
                &and_(&[qa("negative", &sub_one(arg)?)?, qa("nonnegative", &add_one(arg)?)?]),
                a,
            )?) {
                return Ok(Some(true));
            }
            if is_yes(askq("zero", &sub_one(arg)?, a)?) || outside_unit_interval(arg, a)? {
                return Ok(Some(false));
            }
            Ok(None)
        }
        Some(Function::Acot) => askq("real", &e.args()[0], a),
        _ => Ok(None),
    }
}

// -- ntheory.py

fn prime(e: &Basic, a: &Basic) -> R {
    if is_integer_node(e) {
        return Ok(Some(crate::ntheory::primetest::isprime(
            e.as_rational().expect("an integer").numerator(),
        )));
    }
    if is_rational_node(e) || matches!(e.head(), Head::Infinity | Head::NegativeInfinity | Head::ImaginaryUnit) {
        return Ok(Some(false));
    }
    if matches!(e.head(), Head::NaN) {
        return Ok(None);
    }
    if is_number(e) {
        if is_yes(not_integer_number(e)) {
            return Ok(Some(false));
        }
        return Ok(proved_integer(e)?.map(|i| crate::ntheory::primetest::isprime(&i)));
    }
    if e.is_mul() {
        for arg in e.args() {
            if !is_yes(askq("integer", arg, a)?) {
                return Ok(None);
            }
        }
        for arg in e.args() {
            if is_number(arg) && crate::core::expr::is_true(arg, crate::core::assumptions::Predicate::composite) {
                return Ok(Some(false));
            }
        }
        return Ok(None);
    }
    if e.is_pow() {
        let (base, exp) = (&e.args()[0], &e.args()[1]);
        if is_yes(askq("integer", exp, a)?) && is_yes(askq("integer", base, a)?) {
            let prime_base: Truth = askq("prime", base, a)?;
            if prime_base == Some(false) {
                return Ok(Some(false));
            }
            let is_exp_one: Truth = ask_in(&q_applied("eq", &[exp.clone(), Basic::one()])?, a)?;
            if is_exp_one == Some(false) {
                return Ok(Some(false));
            }
            if prime_base == Some(true) && is_exp_one == Some(true) {
                return Ok(Some(true));
            }
        }
        return Ok(None);
    }
    Ok(None)
}

fn composite(e: &Basic, a: &Basic) -> R {
    let positive_: Truth = askq("positive", e, a)?;
    if is_yes(positive_) {
        let integer_: Truth = askq("integer", e, a)?;
        if is_yes(integer_) {
            let prime_: Truth = askq("prime", e, a)?;
            let Some(prime_) = prime_ else { return Ok(None) };
            let is_one: Truth = ask_in(&q_applied("eq", &[e.clone(), Basic::one()])?, a)?;
            if is_one == Some(true) {
                return Ok(Some(false));
            }
            if is_one.is_none() {
                return Ok(None);
            }
            return Ok(Some(!prime_));
        }
        return Ok(integer_);
    }
    Ok(positive_)
}

fn even(e: &Basic, a: &Basic) -> R {
    if is_integer_node(e) {
        let n: &Int = e.as_rational().expect("an integer").numerator();
        return Ok(Some((n % Int::from(2)).is_zero()));
    }
    if is_rational_node(e) || matches!(e.head(), Head::Infinity | Head::NegativeInfinity | Head::ImaginaryUnit) {
        return Ok(Some(false));
    }
    if matches!(e.head(), Head::NaN) {
        return Ok(None);
    }
    if is_number(e) {
        if is_yes(not_integer_number(e)) {
            return Ok(Some(false));
        }
        return Ok(proved_integer(e)?.map(|i| (i % Int::from(2)).is_zero()));
    }
    if e.is_mul() {
        let (mut even_, mut odd, mut irrational_) = (false, 0usize, false);
        let mut acc: Basic = Basic::one();
        for arg in e.args() {
            if is_yes(askq("integer", arg, a)?) {
                if is_yes(askq("even", arg, a)?) {
                    even_ = true;
                } else if is_yes(askq("odd", arg, a)?) {
                    odd += 1;
                } else if !even_ && acc != Basic::one() {
                    let sum: Basic = crate::core::add::add(&[acc.clone(), arg.clone()])?;
                    if is_yes(askq("odd", &sum, a)?) {
                        even_ = true;
                    }
                }
            } else if is_yes(askq("irrational", arg, a)?) {
                if irrational_ {
                    return Ok(None);
                }
                irrational_ = true;
            } else {
                return Ok(None);
            }
            acc = arg.clone();
        }
        if irrational_ {
            return Ok(Some(false));
        }
        if even_ {
            return Ok(Some(true));
        }
        if odd == e.args().len() {
            return Ok(Some(false));
        }
        return Ok(None);
    }
    if e.is_add() {
        let mut result: bool = true;
        for arg in e.args() {
            if is_yes(askq("even", arg, a)?) {
            } else if is_yes(askq("odd", arg, a)?) {
                result = !result;
            } else {
                return Ok(None);
            }
        }
        return Ok(Some(result));
    }
    if e.is_pow() {
        let (base, exp) = (&e.args()[0], &e.args()[1]);
        if is_yes(askq("integer", exp, a)?) {
            if is_yes(askq("positive", exp, a)?) {
                return askq("even", base, a);
            }
            if is_yes(ask_in(&and_(&[not_(&qa("negative", exp)?), qa("odd", base)?]), a)?) {
                return Ok(Some(false));
            }
            if *base == Basic::negative_one() {
                return Ok(Some(false));
            }
        }
        return Ok(None);
    }
    match func(e) {
        Some(Function::Abs | Function::Re) => {
            let arg: &Basic = &e.args()[0];
            if is_yes(askq("real", arg, a)?) {
                return askq("even", arg, a);
            }
            Ok(None)
        }
        Some(Function::Im) => Ok(if is_yes(askq("real", &e.args()[0], a)?) {
            Some(true)
        } else {
            None
        }),
        _ => Ok(None),
    }
}

fn odd(e: &Basic, a: &Basic) -> R {
    let integer_: Truth = askq("integer", e, a)?;
    if is_yes(integer_) {
        let even_: Truth = askq("even", e, a)?;
        return Ok(even_.map(|v| !v));
    }
    Ok(integer_)
}

type Handler = fn(&Basic, &Basic) -> R;

fn handler_of(name: &str) -> Option<Handler> {
    Some(match name {
        "integer" => integer,
        "rational" => rational,
        "irrational" => irrational,
        "real" => real,
        "extended_real" => extended_real,
        "complex" => complex,
        "imaginary" => imaginary,
        "negative" => negative,
        "nonnegative" => nonnegative,
        "nonpositive" => nonpositive,
        "nonzero" => nonzero,
        "zero" => zero,
        "positive" => positive,
        "prime" => prime,
        "composite" => composite,
        "even" => even,
        "odd" => odd,
        _ => return None,
    })
}

fn sign_flip(name: &str) -> Option<&'static str> {
    Some(match name {
        "positive" => "negative",
        "negative" => "positive",
        "nonnegative" => "nonpositive",
        "nonpositive" => "nonnegative",
        "zero" => "zero",
        "nonzero" => "nonzero",
        _ => return None,
    })
}

fn sign_implies(sign: &str, name: &str) -> Truth {
    let table: &[(&str, bool)] = match sign {
        "positive" => &[
            ("positive", true),
            ("nonnegative", true),
            ("nonzero", true),
            ("negative", false),
            ("nonpositive", false),
            ("zero", false),
        ],
        "negative" => &[
            ("negative", true),
            ("nonpositive", true),
            ("nonzero", true),
            ("positive", false),
            ("nonnegative", false),
            ("zero", false),
        ],
        "zero" => &[
            ("zero", true),
            ("nonnegative", true),
            ("nonpositive", true),
            ("positive", false),
            ("negative", false),
            ("nonzero", false),
        ],
        "nonnegative" => &[("nonnegative", true), ("negative", false)],
        "nonpositive" => &[("nonpositive", true), ("positive", false)],
        "nonzero" => &[("nonzero", true), ("zero", false)],
        _ => &[],
    };
    table.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}

fn defined_applied(e: &Basic) -> Option<(&str, &[Basic])> {
    if !matches!(e.head(), Head::AppliedPredicate) {
        return None;
    }
    match e.args()[0].head() {
        Head::Predicate(d) if d.defined => Some((d.name.as_str(), &e.args()[1..])),
        _ => None,
    }
}

fn is_expr(e: &Basic) -> bool {
    !(crate::core::logic::is_boolean(e) || crate::sets::sets::is_set(e) || matches!(e.head(), Head::Tuple))
}

/// A sign predicate of `e` decided by an assumed one of `f` with `e/f` a
/// nonzero rational number (`_scaled_literal`).
fn scaled_literal(proposition: &Basic, assumptions: &Basic) -> R {
    let Some((name, args)) = defined_applied(proposition) else {
        return Ok(None);
    };
    if sign_flip(name).is_none() || args.len() != 1 {
        return Ok(None);
    }
    let e: &Basic = &args[0];
    if !is_expr(e) || is_number(e) {
        return Ok(None);
    }
    let literals: Vec<Basic> = if matches!(assumptions.head(), Head::And) {
        assumptions.args().to_vec()
    } else {
        vec![assumptions.clone()]
    };
    for literal in &literals {
        let Some((known, fargs)) = defined_applied(literal) else {
            continue;
        };
        if sign_flip(known).is_none() || fargs.len() != 1 {
            continue;
        }
        let f: &Basic = &fargs[0];
        if !is_expr(f) || is_number(f) || f.free_symbols() != e.free_symbols() {
            continue;
        }
        let ratio: Basic = crate::polys::polytools::cancel(&crate::core::mul::div(e, f)?, &[])?;
        let Some(r) = ratio.as_rational() else { continue };
        if r.is_zero() {
            continue;
        }
        let sign: &str = if *r > crate::numbers::Rat::ZERO {
            known
        } else {
            sign_flip(known).expect("a sign")
        };
        if let Some(answer) = sign_implies(sign, name) {
            return Ok(Some(answer));
        }
    }
    Ok(None)
}

/// The deepest nesting of the questions of the handlers.
const MAX_DEPTH: usize = 40;

thread_local! {
    static DEPTH: RefCell<usize> = const { RefCell::new(0) };
}

/// The answer of the handler of the predicate for the class of its
/// argument (`handler_ask`).
pub fn handler_ask(proposition: &Basic, assumptions: &Basic) -> R {
    let Some((name, args)) = defined_applied(proposition) else {
        return Ok(None);
    };
    let Some(handler) = handler_of(name) else {
        return Ok(None);
    };
    if args.len() != 1 || !is_expr(&args[0]) {
        return Ok(None);
    }
    if DEPTH.with(|d| *d.borrow()) >= MAX_DEPTH {
        return Ok(None);
    }
    DEPTH.with(|d| *d.borrow_mut() += 1);
    let result: R = handler(&args[0], assumptions);
    DEPTH.with(|d| *d.borrow_mut() -= 1);
    result
}

/// The last steps of the core's `ask`: the handlers, the scaled literal,
/// then the SAT search (`final_ask`).
pub fn final_ask(proposition: &Basic, assumptions: &Basic) -> R {
    if let Some(answer) = handler_ask(proposition, assumptions)? {
        return Ok(Some(answer));
    }
    if let Some(answer) = scaled_literal(proposition, assumptions)? {
        return Ok(Some(answer));
    }
    let consistent = super::satask::predicates_consistent;
    let can_be_true: bool = consistent(&and_(&[assumptions.clone(), proposition.clone()]))?;
    let can_be_false: bool = consistent(&and_(&[assumptions.clone(), not_(proposition)]))?;
    if can_be_true && can_be_false {
        return Ok(None);
    }
    if can_be_true {
        return Ok(Some(true));
    }
    if can_be_false {
        return Ok(Some(false));
    }
    if matches!(assumptions.head(), Head::BooleanTrue) {
        return Ok(None);
    }
    Err(PolyError::Value("Inconsistent assumptions".to_string()))
}
