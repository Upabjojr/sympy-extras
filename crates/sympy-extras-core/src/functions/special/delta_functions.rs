//! Port of `sympy_extras/functions/special/delta_functions.py`: the
//! Dirac delta [`Function::DiracDelta`] (`DiracDelta(x)` and its
//! derivatives `DiracDelta(x, k)`) and the Heaviside step
//! [`Function::Heaviside`] (`Heaviside(x, H0)`, `H0 = 1/2` by default and
//! not printed): evaluation, derivatives, facts and values. Not ported, as
//! in Python: the delta of a polynomial expanded, the rewrites.

use crate::core::add::add;
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, parts, Complex, Real};
use crate::core::expr::{as_coeff_mul, is, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::error::{value_error, PolyResult};
use crate::functions::elementary::complexes::im;
use crate::numbers::{denom, numer, Int};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

/// A `ValueError` for an argument known to be non-real
/// (`_refuse_complex`).
fn refuse_complex(arg: &Basic) -> PolyResult<()> {
    let imaginary: Basic = im(arg)?;
    if is(&imaginary, Predicate::zero) == Some(false) {
        return Err(value_error(format!(
            "Function defined only for Real Values. Complex part: {}  found in {} .",
            imaginary, arg
        )));
    }
    Ok(())
}

/// The order `k` of a delta node (`0` with one argument).
pub fn delta_order(e: &Basic) -> Basic {
    e.args().get(1).cloned().unwrap_or_else(Basic::zero)
}

/// `DiracDelta.eval`.
fn dirac_eval(args: &[Basic]) -> PolyResult<Basic> {
    let arg: &Basic = &args[0];
    let k: Basic = args.get(1).cloned().unwrap_or_else(Basic::zero);
    let order: Option<Int> = k.as_rational().filter(|v| denom(v) == Int::ONE).map(numer);
    let Some(order) = order.filter(|n| *n >= Int::ZERO) else {
        return Err(value_error(format!(
            "Error: the second argument of DiracDelta must be a non-negative integer, {} given instead.",
            k
        )));
    };
    if matches!(arg.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if is_true(arg, Predicate::nonzero) {
        return Ok(Basic::zero());
    }
    refuse_complex(arg)?;
    let (coefficient, _) = as_coeff_mul(arg);
    if coefficient.is_number_node() && is_true(&coefficient, Predicate::extended_negative) {
        let negated: Basic = mul(&[Basic::negative_one(), arg.clone()])?;
        let odd: bool = &order % Int::from(2) != Int::ZERO;
        if odd {
            return mul(&[Basic::negative_one(), dirac_delta(&negated, Some(&k))?]);
        }
        return if order == Int::ZERO {
            dirac_delta(&negated, None)
        } else {
            dirac_delta(&negated, Some(&k))
        };
    }
    if order == Int::ZERO && args.len() > 1 {
        return Ok(node(Function::DiracDelta, vec![arg.clone()]));
    }
    Ok(node(Function::DiracDelta, args.to_vec()))
}

/// `Heaviside.eval`.
fn heaviside_eval(args: &[Basic]) -> PolyResult<Basic> {
    let arg: &Basic = &args[0];
    let mut h0: Basic = args.get(1).cloned().unwrap_or_else(Basic::half);
    if matches!(h0.head(), Head::Function(Function::Heaviside)) {
        // SymPy's __new__: a Heaviside as H0 means 1/2
        h0 = Basic::half();
    }
    if is_true(arg, Predicate::extended_negative) {
        return Ok(Basic::zero());
    }
    if is_true(arg, Predicate::extended_positive) {
        return Ok(Basic::one());
    }
    if is_true(arg, Predicate::zero) {
        return Ok(h0);
    }
    if matches!(arg.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    refuse_complex(arg)?;
    Ok(node(Function::Heaviside, vec![arg.clone(), h0]))
}

/// The evaluating function of `DiracDelta` or `Heaviside` with the
/// arguments of the node.
pub fn delta_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        let mut args: Vec<Basic> = args.to_vec();
        if f == Function::Heaviside && args.len() == 1 {
            args.push(Basic::half());
        }
        return Ok(node(f, args));
    }
    match f {
        Function::DiracDelta => dirac_eval(args),
        _ => heaviside_eval(args),
    }
}

/// `DiracDelta(x, k)`, evaluated (`dirac_delta`; `None` for `k = 0` given
/// as the Python integer: one argument).
pub fn dirac_delta(x: &Basic, k: Option<&Basic>) -> PolyResult<Basic> {
    match k {
        None => delta_function(Function::DiracDelta, std::slice::from_ref(x)),
        Some(k) => delta_function(Function::DiracDelta, &[x.clone(), k.clone()]),
    }
}

/// `Heaviside(x, H0)`, evaluated (`heaviside`; `H0 = 1/2` for `None`).
pub fn heaviside(x: &Basic, h0: Option<&Basic>) -> PolyResult<Basic> {
    let h0: Basic = h0.cloned().unwrap_or_else(Basic::half);
    delta_function(Function::Heaviside, &[x.clone(), h0])
}

/// The derivative in the argument (`fdiff`): `DiracDelta(x, k + 1)` and
/// `DiracDelta(x)`.
pub fn delta_fdiff(f: Function, e: &Basic) -> PolyResult<Basic> {
    match f {
        Function::DiracDelta => {
            let k: Basic = add(&[delta_order(e), Basic::one()])?;
            dirac_delta(&e.args()[0], Some(&k))
        }
        _ => dirac_delta(&e.args()[0], None),
    }
}

/// The facts: real and finite (`_REAL_FACTS`).
pub fn delta_facts() -> FactList {
    let mut facts: FactList = FactList::new();
    for (p, v) in [
        (Predicate::commutative, true),
        (Predicate::real, true),
        (Predicate::extended_real, true),
        (Predicate::finite, true),
        (Predicate::complex, true),
        (Predicate::imaginary, false),
    ] {
        facts.set(p, v);
    }
    facts
}

/// The value (`numeric`): the delta is zero off zero; the step is `H0` at
/// zero, `None` off the real line.
pub fn delta_value(f: Function, values: &[Complex]) -> Option<Complex> {
    let (re, imaginary) = parts(&values[0]);
    match f {
        Function::DiracDelta => {
            if re == Real::ZERO && imaginary == Real::ZERO {
                None
            } else {
                Some(complex(Real::ZERO, Real::ZERO))
            }
        }
        _ => {
            if imaginary != Real::ZERO {
                return None;
            }
            if re == Real::ZERO {
                return values.get(1).cloned();
            }
            Some(complex(
                if re > Real::ZERO { Real::ONE } else { Real::ZERO },
                Real::ZERO,
            ))
        }
    }
}

/// `Heaviside(x)` printed without the default `H0 = 1/2` (`pargs`).
pub fn printed_args(e: &Basic) -> Vec<Basic> {
    if matches!(e.head(), Head::Function(Function::Heaviside)) && e.args().len() == 2 && e.args()[1] == Basic::half() {
        return e.args()[..1].to_vec();
    }
    e.args().to_vec()
}
