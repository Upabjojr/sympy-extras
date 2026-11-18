//! `is_meromorphic`: whether an expression is meromorphic in `x` at `a`
//! (a Laurent series there), ported from the `_eval_is_meromorphic`
//! methods of the core (`Expr`, `AtomicExpr`, `Add`, `Mul`, `Pow`,
//! `Application`) and from `sympy_extras.functions._meromorphic` (the
//! incomplete gamma functions and the Bessel functions with their own
//! rule); the other functions decide by their singularities.

// PORTED FROM SYMPY: sympy/core/expr.py (Expr._eval_is_meromorphic, AtomicExpr._eval_is_meromorphic),
// sympy/core/add.py, sympy/core/mul.py, sympy/core/power.py, sympy/core/function.py
// (Function._eval_is_meromorphic, Function.is_singular), sympy/functions/special/gamma_functions.py
// (lowergamma._eval_is_meromorphic), sympy/functions/special/bessel.py
// (BesselBase._eval_is_meromorphic) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::sub;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::fuzzy::{fuzzy_and, fuzzy_group, fuzzy_not, fuzzy_or};
use crate::core::subs::subs;
use crate::error::{PolyError, PolyResult};

/// The singularities of a function (SymPy's `_singularities`): the
/// points, `zoo` for infinity; `Err(true)` nowhere holomorphic,
/// `Err(false)` unused, `None` unknown.
enum Singularities {
    Points(Vec<Basic>),
    Everywhere,
    Unknown,
}

fn singularities(f: Function) -> PolyResult<Singularities> {
    use Function as F;
    let zoo: Basic = Basic::atom(Head::ComplexInfinity);
    let int = |n: i64| Basic::integer(n.into());
    Ok(match f {
        F::Exp | F::Sin | F::Cos | F::Tan | F::Cot | F::Sec | F::Csc | F::Gamma => Singularities::Points(vec![zoo]),
        F::Log => Singularities::Points(vec![int(0), zoo]),
        F::Asin | F::Acos | F::Asec | F::Acsc => Singularities::Points(vec![int(1), int(-1), int(0), zoo]),
        F::Atan | F::Acot => {
            let i: Basic = Basic::imaginary_unit();
            Singularities::Points(vec![i.clone(), crate::core::mul::neg(&i)?])
        }
        F::Re | F::Im | F::Sign | F::Abs | F::Arg | F::Conjugate => Singularities::Everywhere,
        _ => Singularities::Unknown,
    })
}

/// Whether `a` is a singularity of the function (`Function.is_singular`).
fn is_singular(f: Function, a: &Basic) -> PolyResult<Option<bool>> {
    match singularities(f)? {
        Singularities::Unknown => Ok(None),
        Singularities::Everywhere => Ok(Some(true)),
        Singularities::Points(points) => {
            let mut values: Vec<Option<bool>> = Vec::new();
            for s in points {
                values.push(if matches!(s.head(), Head::ComplexInfinity) {
                    is(a, Predicate::infinite)
                } else {
                    is(&sub(a, &s)?, Predicate::zero)
                });
            }
            Ok(fuzzy_or(values))
        }
    }
}

fn truth(v: Option<bool>) -> bool {
    v == Some(true)
}

// PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:lowergamma._eval_is_meromorphic
fn incomplete_gamma(e: &Basic, x: &Basic, a: &Basic) -> PolyResult<Option<bool>> {
    let (s, z) = (&e.args()[0], &e.args()[1]);
    let args_merom: Option<bool> = fuzzy_and([eval(z, x, a)?, eval(s, x, a)?]);
    if !truth(args_merom) {
        return Ok(args_merom);
    }
    let z0: Basic = subs(z, x, a)?;
    if truth(is(s, Predicate::integer)) {
        return Ok(fuzzy_and([is(s, Predicate::positive), is(&z0, Predicate::finite)]));
    }
    let s0: Basic = subs(s, x, a)?;
    Ok(fuzzy_and([
        is(&s0, Predicate::finite),
        is(&z0, Predicate::finite),
        fuzzy_not(is(&z0, Predicate::zero)),
    ]))
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:BesselBase._eval_is_meromorphic
fn bessel(e: &Basic, x: &Basic, a: &Basic, entire_at_integer_order: bool) -> PolyResult<Option<bool>> {
    let (nu, z) = (&e.args()[0], &e.args()[1]);
    if nu.has(x) {
        return Ok(Some(false));
    }
    if !truth(eval(z, x, a)?) {
        return Ok(None);
    }
    let z0: Basic = subs(z, x, a)?;
    if truth(is(nu, Predicate::integer)) && (entire_at_integer_order || !truth(is(nu, Predicate::zero))) {
        return Ok(fuzzy_not(is(&z0, Predicate::infinite)));
    }
    Ok(fuzzy_not(fuzzy_or([
        is(&z0, Predicate::zero),
        is(&z0, Predicate::infinite),
    ])))
}

// PORTED FROM SYMPY: sympy/core/function.py:Function._eval_is_meromorphic
fn application(e: &Basic, f: Option<Function>, x: &Basic, a: &Basic) -> PolyResult<Option<bool>> {
    let args: &[Basic] = e.args();
    if args.is_empty() {
        return Ok(Some(true));
    }
    if args[1..].iter().any(|t| t.has(x)) {
        return Ok(Some(false));
    }
    let arg: &Basic = &args[0];
    if !truth(eval(arg, x, a)?) {
        return Ok(None);
    }
    match f {
        Some(f) => Ok(fuzzy_not(is_singular(f, &subs(arg, x, a)?)?)),
        None => Ok(None), // an undefined function: no singularities known
    }
}

fn function(e: &Basic, f: Function, x: &Basic, a: &Basic) -> PolyResult<Option<bool>> {
    use Function as F;
    match f {
        F::LowerGamma | F::UpperGamma => incomplete_gamma(e, x, a),
        F::BesselJ | F::BesselI | F::SphericalJ | F::SphericalY | F::SphericalHankel1 | F::SphericalHankel2 => {
            bessel(e, x, a, true)
        }
        F::BesselY | F::BesselK | F::Hankel1 | F::Hankel2 => bessel(e, x, a, false),
        _ => application(e, Some(f), x, a),
    }
}

// PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_is_meromorphic
fn power(e: &Basic, x: &Basic, a: &Basic) -> PolyResult<Option<bool>> {
    // f**g is meromorphic for an integer g and a meromorphic f; E**(log(f)*g)
    // when log(f)*g is meromorphic and finite
    let (base, exponent) = (&e.args()[0], &e.args()[1]);
    let base_merom: Option<bool> = eval(base, x, a)?;
    if matches!(
        exponent.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) {
        return Ok(base_merom);
    }
    let exp_merom: Option<bool> = eval(exponent, x, a)?;
    match base_merom {
        // the singularities of log(f) and g may cancel (g = 1/log(f))
        Some(false) => return Ok(if truth(exp_merom) { Some(false) } else { None }),
        None => return Ok(None),
        Some(true) => {}
    }
    let b: Basic = subs(base, x, a)?;
    let b_zero: Option<bool> = is(&b, Predicate::zero);
    let log_defined: Option<bool> = if truth(b_zero) {
        Some(false)
    } else {
        fuzzy_and([is(&b, Predicate::finite), fuzzy_not(b_zero)])
    };
    match log_defined {
        Some(false) => return Ok(Some(false)), // a zero or a pole of the base
        None => return Ok(None),
        Some(true) => {}
    }
    if !truth(exp_merom) {
        return Ok(exp_merom);
    }
    Ok(is(&subs(exponent, x, a)?, Predicate::finite))
}

/// `_eval_is_meromorphic` of the node.
fn eval(e: &Basic, x: &Basic, a: &Basic) -> PolyResult<Option<bool>> {
    match e.head() {
        // AtomicExpr: a number is meromorphic where it is finite
        Head::ComplexInfinity => Ok(Some(true)),
        Head::Rational { .. } | Head::Infinity | Head::NegativeInfinity | Head::NaN => Ok(is(e, Predicate::finite)),
        _ if e.is_atom() => Ok(Some(true)),
        Head::Add | Head::Mul => {
            // fuzzy_group with quick_exit over a lazy generator: the terms
            // after a None or a second False are not examined
            let mut values: Vec<Option<bool>> = Vec::new();
            let mut falses: usize = 0;
            for t in e.args() {
                let v: Option<bool> = eval(t, x, a)?;
                values.push(v);
                if v == Some(false) {
                    falses += 1;
                }
                if v.is_none() || falses == 2 {
                    break;
                }
            }
            Ok(fuzzy_group(values, true))
        }
        Head::Pow => power(e, x, a),
        Head::Function(f) => function(e, *f, x, a),
        Head::Undefined(_) => application(e, None, x, a),
        _ => {
            // Expr: x itself, or without x
            if e == x || !e.free_symbols().contains(x) {
                return Ok(Some(true));
            }
            Ok(None)
        }
    }
}

/// Whether the expression is meromorphic in `x` at `a`: `Some(true)`,
/// `Some(false)` (a branch point or an essential singularity) or `None`
/// (`is_meromorphic`).
pub fn is_meromorphic(e: &Basic, x: &Basic, a: &Basic) -> PolyResult<Option<bool>> {
    if !x.is_symbol() {
        return Err(PolyError::Type(format!("{} should be of symbol type", x)));
    }
    eval(e, x, a)
}
