//! The singularity (Macaulay bracket) functions, ported from
//! `sympy_extras.functions.special.singularity_functions`:
//! `SingularityFunction(x, a, n)` is `0` left of the offset, `(x - a)**n`
//! right of it for `n >= 0`, `0` away from the offset and `oo` at it for
//! `n = -1, ..., -4`; non-real arguments, imaginary exponents and
//! exponents below `-4` are refused; the derivative lowers the exponent.
//! Not ported: the rewrites into `Heaviside` and `DiracDelta` and the
//! series.

// PORTED FROM SYMPY: sympy/functions/special/singularity_functions.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, parts, Complex, Real};
use crate::core::expr::{is, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::{not_ported, value_error, PolyResult};
use crate::functions::elementary::complexes::im;

fn node(args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(Function::SingularityFunction), args)
}

fn is_small_negative(n: &Basic) -> bool {
    (1..=4).any(|k| *n == Basic::integer((-k).into()))
}

/// `SingularityFunction(x, a, n)`, evaluated.
pub fn singularity_function(x: &Basic, a: &Basic, n: &Basic) -> PolyResult<Basic> {
    let args: Vec<Basic> = vec![x.clone(), a.clone(), n.clone()];
    if !evaluate() {
        return Ok(node(args));
    }
    let shift: Basic = sub(x, a)?;
    if is(&im(&shift)?, Predicate::zero) == Some(false) {
        return Err(value_error(
            "Singularity Functions are defined only for Real Numbers.".to_string(),
        ));
    }
    if is(&im(n)?, Predicate::zero) == Some(false) {
        return Err(value_error(
            "Singularity Functions are not defined for imaginary exponents.".to_string(),
        ));
    }
    if matches!(shift.head(), Head::NaN) || matches!(n.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if is_true(&add(&[n.clone(), Basic::integer(4.into())])?, Predicate::negative) {
        return Err(value_error(
            "Singularity Functions are not defined for exponents less than -4.".to_string(),
        ));
    }
    if is_true(&shift, Predicate::extended_negative) {
        return Ok(Basic::zero());
    }
    if is_true(n, Predicate::nonnegative) {
        if is_true(&shift, Predicate::zero) {
            return pow(&Basic::zero(), n);
        }
        if is_true(&shift, Predicate::extended_nonnegative) {
            return pow(&shift, n);
        }
    }
    if is_small_negative(n) {
        if is_true(&shift, Predicate::negative) || is_true(&shift, Predicate::extended_positive) {
            return Ok(Basic::zero());
        }
        if is_true(&shift, Predicate::zero) {
            return Ok(Basic::oo());
        }
    }
    Ok(node(args))
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn singularity_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let (x, a, n) = (&args[0], &args[1], &args[2]);
    if argindex == 1 {
        let lower: Basic = sub(n, &Basic::one())?;
        if *n == Basic::zero() || (is_small_negative(n) && *n != Basic::integer((-4).into())) {
            return singularity_function(x, a, &lower);
        }
        if is_true(n, Predicate::positive) {
            return mul(&[n.clone(), singularity_function(x, a, &lower)?]);
        }
    }
    let v: &Basic = &args[argindex - 1];
    let elsewhere: bool = args
        .iter()
        .enumerate()
        .any(|(i, w)| i != argindex - 1 && w.free_symbols().contains(v));
    if v.is_symbol() && !elsewhere {
        return Ok(crate::core::function::derivative_node(e, &[(v.clone(), 1)]));
    }
    Err(not_ported("a derivative as a Subs"))
}

/// The facts (`_eval_facts`).
pub fn singularity_facts() -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    facts
}

/// The value (`numeric`).
pub fn singularity_value(values: &[Complex], prec: usize) -> Option<Complex> {
    let (x, a, n) = (&values[0], &values[1], &values[2]);
    let shift: Complex = x - a;
    let (sr, si) = parts(&shift);
    let (nr, ni) = parts(n);
    let zero = || {
        complex(
            Real::ZERO.with_precision(prec).value(),
            Real::ZERO.with_precision(prec).value(),
        )
    };
    if si != Real::ZERO || ni != Real::ZERO {
        return None;
    }
    if sr < Real::ZERO {
        return Some(zero());
    }
    if nr >= Real::ZERO {
        if sr == Real::ZERO {
            if nr == Real::ZERO {
                return Some(complex(
                    Real::ONE.with_precision(prec).value(),
                    Real::ZERO.with_precision(prec).value(),
                ));
            }
            return Some(zero());
        }
        let value: Real = (&nr * &sr.ln()).exp();
        return Some(complex(
            value.with_precision(prec).value(),
            Real::ZERO.with_precision(prec).value(),
        ));
    }
    if sr == Real::ZERO {
        return None;
    }
    Some(zero())
}
