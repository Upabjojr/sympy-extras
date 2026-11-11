//! Port of `sympy_extras/functions`: the function classes on the core.

pub mod elementary;

use crate::core::basic::{Basic, Function};
use crate::error::PolyResult;

/// The evaluating function of a function head (`cls.apply`).
pub fn apply_function(f: Function, x: &Basic) -> PolyResult<Basic> {
    use elementary::{complexes as c, exponential as e, trigonometric as t};
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
    }
}
