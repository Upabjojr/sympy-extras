//! Port of `sympy_extras/functions/elementary/hyperbolic.py`, the direct
//! functions: `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch` (nodes of
//! [`Head::Function`]; the lower case functions evaluate): the values at
//! the numbers, the exchange with the trigonometric functions on imaginary
//! arguments, the multiples of `I*pi/2` peeled off sums, the facts, values,
//! conjugates and real and imaginary parts. The inverse functions are not
//! ported yet.

// PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py (SymPy 1.14),
// through sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_coefficient, is_true};
use crate::core::function::expand_default;
use crate::core::mul::{mul, raw_mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::PolyResult;
use crate::numbers::*;

use super::complexes::{as_real_imag, conjugate, RealImag};
use super::trigonometric::{cos, cot, sin, tan};

use Predicate::*;

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

pub fn sinh(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Sinh, x)
}

pub fn cosh(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Cosh, x)
}

pub fn tanh(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Tanh, x)
}

pub fn coth(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Coth, x)
}

pub fn sech(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Sech, x)
}

pub fn csch(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Csch, x)
}

fn minus(e: Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e])
}

fn i() -> Basic {
    Basic::imaginary_unit()
}

/// `(rest, m)` with `arg = rest + m*I*pi`, `m` a multiple of `1/2`, from the
/// first term which is a rational multiple of `I*pi` (`peeloff_ipi`).
pub fn peeloff_ipi(arg: &Basic) -> PolyResult<(Basic, Rat)> {
    let ipi: Basic = mul(&[Basic::pi(), i()])?;
    let mut k: Option<Rat> = None;
    let terms: Vec<Basic> = if arg.is_add() {
        arg.args().to_vec()
    } else {
        vec![arg.clone()]
    };
    for term in &terms {
        if *term == ipi {
            k = Some(Rat::ONE);
            break;
        }
        if term.is_mul() {
            // as_two_terms
            let first: &Basic = &term.args()[0];
            let rest: Basic = raw_mul(term.args()[1..].to_vec());
            if rest == ipi {
                if let Some(v) = first.as_rational() {
                    k = Some(v.clone());
                    break;
                }
            }
        }
    }
    let Some(k) = k else {
        return Ok((arg.clone(), Rat::ZERO));
    };
    let half: Rat = rat(1, 2);
    let m2: Rat = &k - (&k - &half * rat_from_int(&div_floor(&numer(&(&k / &half)), &denom(&(&k / &half)))));
    let shift: Basic = mul(&[Basic::negative_one(), Basic::from_rational(m2.clone()), ipi])?;
    Ok((add(&[arg.clone(), shift])?, m2))
}

/// `.eval` of one of the functions of this module.
pub fn eval(f: Function, arg: &Basic) -> PolyResult<Option<Basic>> {
    match f {
        Function::Sech => return reciprocal_eval(f, Function::Cosh, true, arg),
        Function::Csch => return reciprocal_eval(f, Function::Sinh, false, arg),
        _ => {}
    }
    let odd_function: bool = !matches!(f, Function::Cosh);
    if arg.is_number_class() && !matches!(arg.head(), Head::ComplexInfinity) {
        let value: Option<Basic> = match (arg.head(), f) {
            (Head::NaN, _) => Some(Basic::nan()),
            (Head::Infinity, Function::Sinh) => Some(Basic::oo()),
            (Head::NegativeInfinity, Function::Sinh) => Some(minus(Basic::oo())?),
            (Head::Infinity | Head::NegativeInfinity, Function::Cosh) => Some(Basic::oo()),
            (Head::Infinity, _) => Some(Basic::one()),
            (Head::NegativeInfinity, _) => Some(Basic::negative_one()),
            (Head::Rational { value, .. }, _) if value.is_zero() => Some(match f {
                Function::Cosh => Basic::one(),
                Function::Coth => Basic::zoo(),
                _ => Basic::zero(),
            }),
            _ => None,
        };
        if value.is_some() {
            return Ok(value);
        }
        if is_true(arg, negative) {
            let flipped: Basic = apply(f, &minus(arg.clone())?)?;
            return Ok(Some(if odd_function { minus(flipped)? } else { flipped }));
        }
        return Ok(None);
    }
    if matches!(arg.head(), Head::ComplexInfinity) {
        return Ok(Some(Basic::nan()));
    }
    if let Some(c) = as_coefficient(arg, &i())? {
        return Ok(Some(match f {
            Function::Sinh => mul(&[i(), sin(&c)?])?,
            Function::Cosh => cos(&c)?,
            Function::Tanh => {
                if could_extract_minus_sign(&c) {
                    mul(&[Basic::negative_one(), i(), tan(&minus(c)?)?])?
                } else {
                    mul(&[i(), tan(&c)?])?
                }
            }
            _ => {
                if could_extract_minus_sign(&c) {
                    mul(&[i(), cot(&minus(c)?)?])?
                } else {
                    mul(&[Basic::negative_one(), i(), cot(&c)?])?
                }
            }
        }));
    }
    if could_extract_minus_sign(arg) {
        let flipped: Basic = apply(f, &minus(arg.clone())?)?;
        return Ok(Some(if odd_function { minus(flipped)? } else { flipped }));
    }
    if arg.is_add() {
        let (rest, m) = peeloff_ipi(arg)?;
        if !m.is_zero() {
            let multiple: Basic = mul(&[Basic::from_rational(m.clone()), Basic::pi(), i()])?;
            return Ok(Some(match f {
                Function::Sinh => add(&[
                    mul(&[sinh(&multiple)?, cosh(&rest)?])?,
                    mul(&[cosh(&multiple)?, sinh(&rest)?])?,
                ])?,
                Function::Cosh => add(&[
                    mul(&[cosh(&multiple)?, cosh(&rest)?])?,
                    mul(&[sinh(&multiple)?, sinh(&rest)?])?,
                ])?,
                Function::Tanh => {
                    if m.is_int() {
                        tanh(&rest)?
                    } else {
                        coth(&rest)?
                    }
                }
                _ => {
                    if m.is_int() {
                        coth(&rest)?
                    } else {
                        tanh(&rest)?
                    }
                }
            }));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(match f {
            Function::Cosh => Basic::one(),
            Function::Coth => Basic::zoo(),
            _ => Basic::zero(),
        }));
    }
    Ok(None)
}

/// `sech` and `csch` through `cosh` and `sinh`
/// (`ReciprocalHyperbolicFunction.eval`).
fn reciprocal_eval(f: Function, partner: Function, even_function: bool, arg: &Basic) -> PolyResult<Option<Basic>> {
    if could_extract_minus_sign(arg) {
        let flipped: Basic = apply(f, &minus(arg.clone())?)?;
        return Ok(Some(if even_function { flipped } else { minus(flipped)? }));
    }
    match eval(partner, arg)? {
        Some(t) => Ok(Some(pow(&t, &Basic::negative_one())?)),
        None => Ok(None),
    }
}

/// The facts of a node of this module (`_eval_facts`).
pub fn hyperbolic_facts(f: Function, arg: &Basic) -> FactList {
    let a = facts_of(arg);
    let yes = |p: Predicate| a.get(&p) == Some(&true);
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    let odd_facts = |facts: &mut FactList| {
        if yes(extended_real) {
            facts.set(extended_real, true);
            for p in [positive, negative] {
                if let Some(v) = a.get(&p) {
                    facts.set(p, *v);
                }
            }
        }
        if yes(real) {
            facts.set(real, true);
        }
    };
    match f {
        Function::Sinh => {
            odd_facts(&mut facts);
            if yes(finite) {
                facts.set(finite, true);
            }
        }
        Function::Cosh => {
            if yes(real) || yes(imaginary) {
                facts.set(real, true);
            }
            if yes(real) {
                facts.set(positive, true);
            }
            if yes(finite) {
                facts.set(finite, true);
            }
        }
        Function::Tanh => {
            odd_facts(&mut facts);
            if yes(extended_real) {
                facts.set(finite, true);
            }
            if yes(zero) {
                facts.set(zero, true);
            }
        }
        _ => {}
    }
    facts
}

/// The conjugate of a node of this module: the function of the conjugate.
pub fn hyperbolic_conjugate(f: Function, arg: &Basic) -> PolyResult<Basic> {
    apply(f, &conjugate(arg)?)
}

/// The real and imaginary parts of a node of this module
/// (`_eval_as_real_imag`).
pub fn hyperbolic_real_imag(e: &Basic, f: Function, deep: bool) -> PolyResult<Option<RealImag>> {
    let arg: &Basic = &e.args()[0];
    match f {
        Function::Sech => return as_real_imag(&pow(&cosh(arg)?, &Basic::negative_one())?, deep, None),
        Function::Csch => return as_real_imag(&pow(&sinh(arg)?, &Basic::negative_one())?, deep, None),
        _ => {}
    }
    if is_true(arg, extended_real) {
        return Ok(Some((if deep { expand_default(e)? } else { e.clone() }, Basic::zero())));
    }
    let expanded: Basic = if deep { expand_default(arg)? } else { arg.clone() };
    let Some((r, im_part)) = as_real_imag(&expanded, true, None)? else {
        return Ok(None);
    };
    Ok(Some(match f {
        Function::Sinh => (mul(&[sinh(&r)?, cos(&im_part)?])?, mul(&[cosh(&r)?, sin(&im_part)?])?),
        Function::Cosh => (mul(&[cosh(&r)?, cos(&im_part)?])?, mul(&[sinh(&r)?, sin(&im_part)?])?),
        Function::Tanh => {
            let denominator: Basic = pow(
                &add(&[pow(&sinh(&r)?, &Basic::int(2))?, pow(&cos(&im_part)?, &Basic::int(2))?])?,
                &Basic::negative_one(),
            )?;
            (
                mul(&[sinh(&r)?, cosh(&r)?, denominator.clone()])?,
                mul(&[sin(&im_part)?, cos(&im_part)?, denominator])?,
            )
        }
        _ => {
            let denominator: Basic = pow(
                &add(&[pow(&sinh(&r)?, &Basic::int(2))?, pow(&sin(&im_part)?, &Basic::int(2))?])?,
                &Basic::negative_one(),
            )?;
            (
                mul(&[sinh(&r)?, cosh(&r)?, denominator.clone()])?,
                mul(&[Basic::negative_one(), sin(&im_part)?, cos(&im_part)?, denominator])?,
            )
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    #[test]
    fn examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        let ix: Basic = mul(&[i(), x.clone()]).unwrap();
        assert_eq!(show(sinh(&ix)), "I*sin(x)");
        assert_eq!(show(cosh(&ix)), "cos(x)");
        assert_eq!(show(sin(&ix)), "I*sinh(x)");
        assert_eq!(show(cos(&ix)), "cosh(x)");
        assert_eq!(show(sinh(&minus(x.clone()).unwrap())), "-sinh(x)");
        let ipi: Basic = mul(&[Basic::pi(), i()]).unwrap();
        assert_eq!(show(cosh(&add(&[x.clone(), ipi]).unwrap())), "-cosh(x)");
        assert_eq!(show(tanh(&Basic::oo())), "1");
    }
}
