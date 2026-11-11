//! Port of `sympy_extras/functions`: the function classes on the core.

pub mod elementary;

use crate::core::basic::{Basic, Function};
use crate::error::PolyResult;

/// The partial derivative of the node `e` of the function `f` with
/// respect to its argument (`fdiff`).
pub fn fdiff(f: Function, e: &Basic) -> PolyResult<Basic> {
    use crate::core::add::add;
    use crate::core::mul::mul;
    use crate::core::power::pow;
    use elementary::{hyperbolic as h, trigonometric as t};
    let x: &Basic = &e.args()[0];
    let minus: Basic = Basic::negative_one();
    let square = |b: &Basic| pow(b, &Basic::int(2));
    match f {
        Function::Exp => Ok(e.clone()),
        Function::Log => pow(x, &minus),
        Function::Sin => t::cos(x),
        Function::Cos => mul(&[minus, t::sin(x)?]),
        Function::Tan => add(&[Basic::one(), square(e)?]),
        Function::Cot => add(&[minus.clone(), mul(&[minus, square(e)?])?]),
        Function::Sec => mul(&[t::tan(x)?, e.clone()]),
        Function::Csc => mul(&[minus, t::cot(x)?, e.clone()]),
        Function::Sinh => h::cosh(x),
        Function::Cosh => h::sinh(x),
        Function::Tanh => add(&[Basic::one(), mul(&[minus, square(e)?])?]),
        Function::Coth => mul(&[minus, pow(&h::sinh(x)?, &Basic::int(-2))?]),
        Function::Sech => mul(&[minus, h::tanh(x)?, e.clone()]),
        Function::Csch => mul(&[minus, h::coth(x)?, e.clone()]),
        g @ (Function::Asin
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
        | Function::Acsch) => elementary::inverse::inverse_fdiff(g, x),
        Function::Re | Function::Im | Function::Conjugate | Function::Sign | Function::Abs => {
            unreachable!("these functions have their own derivatives")
        }
    }
}

/// The derivative of the node `e` of `f` with respect to `s` when the
/// function has its own rule (`_eval_derivative`: not holomorphic); `None`
/// for the chain rule with [`fdiff`].
pub fn eval_derivative(f: Function, e: &Basic, s: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::add::add;
    use crate::core::assumptions::Predicate::*;
    use crate::core::expr::is_true;
    use crate::core::function::{derivative, derivative_node};
    use crate::core::mul::mul;
    use crate::core::power::pow;
    use elementary::complexes as c;
    let a: &Basic = &e.args()[0];
    let i: Basic = Basic::imaginary_unit();
    let unevaluated = || derivative_node(e, &[(s.clone(), 1)]);
    Ok(Some(match f {
        Function::Re | Function::Im => {
            let (same, other): (fn(&Basic) -> PolyResult<Basic>, fn(&Basic) -> PolyResult<Basic>) =
                if matches!(f, Function::Re) {
                    (c::re, c::im)
                } else {
                    (c::im, c::re)
                };
            if is_true(s, extended_real) || is_true(a, extended_real) {
                same(&derivative(a, s)?)?
            } else if is_true(s, imaginary) || is_true(a, imaginary) {
                mul(&[Basic::negative_one(), i, other(&derivative(a, s)?)?])?
            } else {
                unevaluated()
            }
        }
        Function::Abs => {
            if is_true(a, extended_real) || is_true(a, imaginary) {
                mul(&[derivative(a, s)?, c::sign(&c::conjugate(a)?)?])?
            } else {
                // 1/Abs(a) written with sign: sign(a)/a
                let (r, m) = (c::re(a)?, c::im(a)?);
                let numerator: Basic = add(&[
                    mul(&[r.clone(), derivative(&r, s)?])?,
                    mul(&[m.clone(), derivative(&m, s)?])?,
                ])?;
                mul(&[numerator, c::sign(a)?, pow(a, &Basic::negative_one())?])?
            }
        }
        Function::Sign => {
            if is_true(a, extended_real) || is_true(a, imaginary) {
                return Err(crate::error::not_ported("DiracDelta"));
            }
            unevaluated()
        }
        Function::Conjugate => {
            if is_true(s, real) {
                c::conjugate(&derivative(a, s)?)?
            } else if is_true(s, imaginary) {
                mul(&[Basic::negative_one(), c::conjugate(&derivative(a, s)?)?])?
            } else {
                unevaluated()
            }
        }
        _ => return Ok(None),
    }))
}

/// The evaluating function of a function head (`cls.apply`).
pub fn apply_function(f: Function, x: &Basic) -> PolyResult<Basic> {
    use elementary::{complexes as c, exponential as e, hyperbolic as h, trigonometric as t};
    match f {
        Function::Exp => e::exp(x),
        Function::Log => e::log(x),
        Function::Re => c::re(x),
        Function::Im => c::im(x),
        Function::Conjugate => c::conjugate(x),
        Function::Sign => c::sign(x),
        Function::Abs => c::abs_(x),
        Function::Sin => t::sin(x),
        Function::Cos => t::cos(x),
        Function::Tan => t::tan(x),
        Function::Cot => t::cot(x),
        Function::Sec => t::sec(x),
        Function::Csc => t::csc(x),
        Function::Sinh => h::sinh(x),
        Function::Cosh => h::cosh(x),
        Function::Tanh => h::tanh(x),
        Function::Coth => h::coth(x),
        Function::Sech => h::sech(x),
        Function::Csch => h::csch(x),
        Function::Asin => elementary::inverse::asin(x),
        Function::Acos => elementary::inverse::acos(x),
        Function::Atan => elementary::inverse::atan(x),
        Function::Acot => elementary::inverse::acot(x),
        Function::Asec => elementary::inverse::asec(x),
        Function::Acsc => elementary::inverse::acsc(x),
        Function::Asinh => elementary::inverse::asinh(x),
        Function::Acosh => elementary::inverse::acosh(x),
        Function::Atanh => elementary::inverse::atanh(x),
        Function::Acoth => elementary::inverse::acoth(x),
        Function::Asech => elementary::inverse::asech(x),
        Function::Acsch => elementary::inverse::acsch(x),
    }
}
