//! Port of `sympy_extras/functions`: the function classes on the core.

pub mod combinatorial;
pub mod elementary;
pub mod special;

use crate::core::basic::{Basic, Function};
use crate::error::PolyResult;

/// The partial derivative of the node `e` of the function `f` with
/// respect to its argument (`fdiff`).
/// `argindex` is the index of the argument from 1 (`Function.fdiff`).
pub fn fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
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
        Function::Atan2 => elementary::argument::atan2_fdiff(e, argindex),
        g @ (Function::DiracDelta | Function::Heaviside) => special::delta_functions::delta_fdiff(g, e),
        g @ (Function::Zeta
        | Function::DirichletEta
        | Function::PolyLog
        | Function::LerchPhi
        | Function::RiemannXi
        | Function::Stieltjes) => special::zeta_functions::zeta_fdiff(g, e, argindex),
        g @ (Function::Expint
        | Function::Ei
        | Function::Si
        | Function::Ci
        | Function::Shi
        | Function::Chi
        | Function::LogIntegral
        | Function::OffsetLogIntegral
        | Function::FresnelS
        | Function::FresnelC) => special::error_functions::integral_fdiff(g, e, argindex),
        g @ (Function::Erf
        | Function::Erfc
        | Function::Erfi
        | Function::Erfinv
        | Function::Erfcinv
        | Function::Erf2
        | Function::Erf2inv) => special::error_functions::error_fdiff(g, e, argindex),
        g @ (Function::Beta | Function::BetaInc | Function::BetaIncRegularized) => {
            special::beta_functions::beta_fdiff(g, e, argindex)
        }
        g @ (Function::Catalan
        | Function::Fibonacci
        | Function::Lucas
        | Function::Harmonic
        | Function::Bernoulli
        | Function::Euler
        | Function::Bell
        | Function::Genocchi
        | Function::Partition
        | Function::Totient
        | Function::ReducedTotient
        | Function::DivisorSigma
        | Function::UDivisorSigma
        | Function::PrimeNu
        | Function::PrimeOmega
        | Function::Mobius
        | Function::PrimePi
        | Function::LegendreSymbol
        | Function::JacobiSymbol
        | Function::KroneckerSymbol
        | Function::Tribonacci
        | Function::Motzkin
        | Function::Andre) => combinatorial::numbers::number_fdiff(g, e, argindex),
        Function::Mod | Function::KroneckerDelta | Function::LeviCivita => {
            crate::core::modulo::modulo_fdiff(e, argindex)
        }
        Function::SingularityFunction => special::singularity_functions::singularity_fdiff(e, argindex),
        g @ (Function::LowerGamma | Function::UpperGamma | Function::MultiGamma) => {
            special::gamma_functions::incomplete_fdiff(g, e, argindex)
        }
        g @ (Function::Gamma | Function::LogGamma | Function::PolyGamma | Function::DiGamma | Function::TriGamma) => {
            special::gamma_functions::gamma_fdiff(g, e, argindex)
        }
        Function::Factorial
        | Function::Binomial
        | Function::RisingFactorial
        | Function::FallingFactorial
        | Function::Subfactorial
        | Function::Factorial2 => combinatorial::factorials::combinatorial_derivative(f, e, argindex),
        Function::Re
        | Function::Im
        | Function::Conjugate
        | Function::Sign
        | Function::Abs
        | Function::Arg
        | Function::Max
        | Function::Min
        | Function::Floor
        | Function::Ceiling
        | Function::Frac => {
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
            use special::delta_functions::dirac_delta;
            if is_true(a, extended_real) {
                mul(&[Basic::int(2), derivative(a, s)?, dirac_delta(a, None)?])?
            } else if is_true(a, imaginary) {
                let rotated: Basic = mul(&[Basic::negative_one(), i, a.clone()])?;
                mul(&[Basic::int(2), derivative(a, s)?, dirac_delta(&rotated, None)?])?
            } else {
                unevaluated()
            }
        }
        Function::Arg => elementary::argument::arg_derivative(e, s)?,
        Function::Max | Function::Min => elementary::miscellaneous::minmax_derivative(e, s)?,
        Function::Floor | Function::Ceiling | Function::Frac => unevaluated(),
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
/// `args` are the arguments of the node (two for `atan2`, one otherwise).
pub fn apply_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if f == Function::Atan2 {
        return elementary::argument::atan2(&args[0], &args[1]);
    }
    if matches!(f, Function::Max | Function::Min) {
        return elementary::miscellaneous::minmax(f, args);
    }
    if matches!(f, Function::DiracDelta | Function::Heaviside) {
        return special::delta_functions::delta_function(f, args);
    }
    if special::error_functions::is_integral_function(f) {
        return special::error_functions::integral_function(f, args);
    }
    if special::error_functions::is_error_function(f) {
        return special::error_functions::error_function(f, args);
    }
    if special::beta_functions::is_beta_function(f) {
        return special::beta_functions::beta_function(f, args);
    }
    if combinatorial::numbertheory::is_number_theoretic(f) {
        return combinatorial::numbertheory::number_theoretic(f, args);
    }
    if combinatorial::numbers::is_combinatorial_number(f) {
        return combinatorial::numbers::combinatorial_number(f, args);
    }
    if matches!(f, Function::KroneckerDelta | Function::LeviCivita) {
        return special::tensor_functions::tensor_function(f, args);
    }
    if f == Function::Mod {
        return crate::core::modulo::modulo(&args[0], &args[1]);
    }
    if special::zeta_functions::is_zeta_function(f) {
        return special::zeta_functions::zeta_function(f, args);
    }
    if f == Function::SingularityFunction && args.len() == 3 {
        return special::singularity_functions::singularity_function(&args[0], &args[1], &args[2]);
    }
    if special::gamma_functions::is_incomplete_gamma(f) {
        return special::gamma_functions::incomplete_gamma(f, args);
    }
    if special::gamma_functions::is_gamma_function(f) {
        return special::gamma_functions::gamma_function(f, args);
    }
    if combinatorial::factorials::is_combinatorial(f) {
        return combinatorial::factorials::combinatorial(f, args);
    }
    let x: &Basic = &args[0];
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
        Function::Arg => elementary::argument::arg(x),
        g @ (Function::Floor | Function::Ceiling | Function::Frac) => elementary::integers::rounding(g, x),
        Function::Erf
        | Function::Erfc
        | Function::Erfi
        | Function::Erfinv
        | Function::Erfcinv
        | Function::Erf2
        | Function::Erf2inv => unreachable!("evaluated above with all their arguments"),
        Function::Expint
        | Function::Ei
        | Function::Si
        | Function::Ci
        | Function::Shi
        | Function::Chi
        | Function::LogIntegral
        | Function::OffsetLogIntegral
        | Function::FresnelS
        | Function::FresnelC => unreachable!("evaluated above with all their arguments"),
        Function::Beta | Function::BetaInc | Function::BetaIncRegularized => {
            unreachable!("evaluated above with all their arguments")
        }
        Function::Atan2 | Function::Max | Function::Min | Function::DiracDelta | Function::Heaviside => {
            unreachable!("evaluated above with all their arguments")
        }
        Function::Factorial
        | Function::Binomial
        | Function::RisingFactorial
        | Function::FallingFactorial
        | Function::Subfactorial
        | Function::Factorial2 => unreachable!("evaluated above with all their arguments"),
        Function::Zeta
        | Function::DirichletEta
        | Function::PolyLog
        | Function::LerchPhi
        | Function::RiemannXi
        | Function::Stieltjes
        | Function::Mod
        | Function::KroneckerDelta
        | Function::LeviCivita => unreachable!("evaluated above with all their arguments"),
        Function::Catalan
        | Function::Fibonacci
        | Function::Lucas
        | Function::Harmonic
        | Function::Bernoulli
        | Function::Euler
        | Function::Bell
        | Function::Genocchi
        | Function::Partition
        | Function::Totient
        | Function::ReducedTotient
        | Function::DivisorSigma
        | Function::UDivisorSigma
        | Function::PrimeNu
        | Function::PrimeOmega
        | Function::Mobius
        | Function::PrimePi
        | Function::LegendreSymbol
        | Function::JacobiSymbol
        | Function::KroneckerSymbol
        | Function::Tribonacci
        | Function::Motzkin
        | Function::Andre => unreachable!("evaluated above with all their arguments"),
        Function::Gamma | Function::LogGamma | Function::PolyGamma | Function::DiGamma | Function::TriGamma => {
            unreachable!("evaluated above with all their arguments")
        }
        Function::SingularityFunction => unreachable!("evaluated above with all their arguments"),
        Function::LowerGamma | Function::UpperGamma | Function::MultiGamma => {
            unreachable!("evaluated above with all their arguments")
        }
    }
}
