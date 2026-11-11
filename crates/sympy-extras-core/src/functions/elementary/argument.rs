//! Port of `arg` and `atan2` of `sympy_extras/functions/elementary/
//! complexes.py`: the nodes `Arg(z)` and `Atan2(y, x)` of
//! [`Head::Function`], their evaluation (the argument of an exponential
//! reduced into `(-pi, pi]`, of the sign-normalised factors by the real
//! and imaginary parts; `atan2` by the signs of its arguments, numbers by
//! a logarithm), facts, derivatives, conjugates and values.

// PORTED FROM SYMPY: sympy/functions/elementary/complexes.py (arg),
// sympy/functions/elementary/trigonometric.py (atan2) (SymPy 1.14), through
// sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::assumptions::{facts_of, is_number, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{parts, Complex};
use crate::core::expr::{as_coeff_mul, is_comparable, is_false, is_true};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::PolyResult;

use super::complexes::{as_real_imag, conjugate, im, re, sign};
use super::exponential::log;
use super::inverse::atan;

use Predicate::*;

fn pi() -> Basic {
    Basic::pi()
}

fn minus_one() -> Basic {
    Basic::negative_one()
}

/// `arg(z)`, evaluated.
pub fn arg(z: &Basic) -> PolyResult<Basic> {
    if evaluate() {
        if let Some(value) = arg_eval(z)? {
            return Ok(value);
        }
    }
    Ok(Basic::new(Head::Function(Function::Arg), vec![z.clone()]))
}

/// `atan2(y, x)`, evaluated.
pub fn atan2(y: &Basic, x: &Basic) -> PolyResult<Basic> {
    if evaluate() {
        if let Some(value) = atan2_eval(y, x)? {
            return Ok(value);
        }
    }
    Ok(Basic::new(Head::Function(Function::Atan2), vec![y.clone(), x.clone()]))
}

fn undefined_atoms(e: &Basic) -> Vec<Basic> {
    e.preorder()
        .into_iter()
        .filter(|n| matches!(n.head(), Head::Undefined(_)))
        .collect()
}

/// `Arg.eval`.
fn arg_eval(z: &Basic) -> PolyResult<Option<Basic>> {
    let mut a: Basic = z.clone();
    let mut nested: bool = true;
    for i in 0..3 {
        if a.is_function(Function::Arg) {
            a = a.args()[0].clone();
        } else {
            if i == 2 && is_true(&a, extended_real) {
                return Ok(Some(Basic::nan()));
            }
            nested = false;
            break;
        }
    }
    if nested {
        return Ok(Some(Basic::nan()));
    }
    if z.is_function(Function::Exp) {
        let angle: Basic = im(&z.args()[0])?;
        if is_comparable(&angle) {
            let two_pi: Basic = mul(&[Basic::int(2), pi()])?;
            let mut angle: Basic = super::inverse::modulo(&angle, &two_pi)?;
            if super::inverse::gt(&angle, &pi())? {
                angle = add(&[angle, mul(&[Basic::int(-2), pi()])?])?;
            }
            return Ok(Some(angle));
        }
    }
    let rest: Basic = if !z.args().is_empty() {
        let factored: Basic = crate::core::exprtools::factor_terms(z, false, false, false, true)?;
        let (c, mut rest) = as_coeff_mul(&factored);
        if rest.is_mul() {
            let mut factors: Vec<Basic> = Vec::new();
            for t in rest.args() {
                let s: Basic = sign(t)?;
                let unit: bool = s
                    .as_rational()
                    .is_some_and(|v| *v == crate::numbers::Rat::ONE || *v == -crate::numbers::Rat::ONE);
                factors.push(if unit { s } else { t.clone() });
            }
            rest = mul(&factors)?;
        }
        mul(&[sign(&c)?, rest])?
    } else {
        z.clone()
    };
    if undefined_atoms(&rest)
        .iter()
        .any(|u| crate::core::expr::is(u, extended_positive).is_none())
    {
        return Ok(None);
    }
    let Some((x, y)) = as_real_imag(&rest, true, None)? else {
        return Ok(None);
    };
    let value: Basic = atan2(&y, &x)?;
    if is_number(&value) {
        return Ok(Some(value));
    }
    if rest != *z {
        return Ok(Some(Basic::new(Head::Function(Function::Arg), vec![rest])));
    }
    Ok(None)
}

/// `Atan2.eval`.
fn atan2_eval(y: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let half_pi = || mul(&[pi(), Basic::half()]);
    if matches!(x.head(), Head::NegativeInfinity) {
        if is_true(y, zero) {
            return Ok(Some(pi()));
        }
        // SymPy: 2*pi*Heaviside(re(y)) - pi, evaluated when the sign is known
        let real_y: Basic = re(y)?;
        if is_true(&real_y, extended_positive) {
            return Ok(Some(pi()));
        }
        if is_true(&real_y, extended_negative) {
            return Ok(Some(mul(&[minus_one(), pi()])?));
        }
        if is_true(&real_y, zero) {
            return Ok(Some(Basic::zero()));
        }
        return Ok(None);
    }
    if matches!(x.head(), Head::Infinity) {
        return Ok(Some(Basic::zero()));
    }
    let (mut x, mut y) = (x.clone(), y.clone());
    if is_true(&x, imaginary) && is_true(&y, imaginary) && is_number(&x) && is_number(&y) {
        let (ix, iy) = (im(&x)?, im(&y)?);
        x = ix;
        y = iy;
    }
    let ratio = || mul(&[y.clone(), pow(&x, &minus_one())?]);
    if is_true(&x, extended_real) && is_true(&y, extended_real) {
        if is_true(&x, positive) {
            return Ok(Some(atan(&ratio()?)?));
        }
        if is_true(&x, negative) {
            if is_true(&y, negative) {
                return Ok(Some(add(&[atan(&ratio()?)?, mul(&[minus_one(), pi()])?])?));
            }
            if is_true(&y, nonnegative) {
                return Ok(Some(add(&[atan(&ratio()?)?, pi()])?));
            }
        } else if is_true(&x, zero) {
            if is_true(&y, positive) {
                return Ok(Some(half_pi()?));
            }
            if is_true(&y, negative) {
                return Ok(Some(mul(&[
                    pi(),
                    crate::core::basic::Basic::from_rational(crate::numbers::rat(-1, 2)),
                ])?));
            }
            if is_true(&y, zero) {
                return Ok(Some(Basic::nan()));
            }
        }
    }
    if is_true(&y, zero) {
        if is_true(&x, extended_nonzero) {
            // SymPy: pi*(1 - Heaviside(x)), evaluated when the sign is known
            if is_true(&x, extended_positive) {
                return Ok(Some(Basic::zero()));
            }
            if is_true(&x, extended_negative) {
                return Ok(Some(pi()));
            }
            return Ok(None);
        }
        if is_number(&x) {
            // SymPy: Piecewise((pi, re(x) < 0), (0, Ne(x, 0)), (nan, True)),
            // evaluated when its conditions are decided
            let real_x: Basic = re(&x)?;
            match crate::core::expr::is(&real_x, extended_negative) {
                Some(true) => return Ok(Some(pi())),
                Some(false) => {
                    if is_false(&x, zero) {
                        return Ok(Some(Basic::zero()));
                    }
                    if is_true(&x, zero) {
                        return Ok(Some(Basic::nan()));
                    }
                }
                None => {}
            }
            return Ok(None);
        }
    }
    if is_number(&x) && is_number(&y) {
        let modulus: Basic = pow(
            &add(&[pow(&x, &Basic::int(2))?, pow(&y, &Basic::int(2))?])?,
            &Basic::half(),
        )?;
        let point: Basic = mul(&[
            add(&[x.clone(), mul(&[Basic::imaginary_unit(), y.clone()])?])?,
            pow(&modulus, &minus_one())?,
        ])?;
        return Ok(Some(mul(&[minus_one(), Basic::imaginary_unit(), log(&point)?])?));
    }
    Ok(None)
}

/// The derivative of `Arg(z)` with respect to `s` (`Arg._eval_derivative`).
pub fn arg_derivative(e: &Basic, s: &Basic) -> PolyResult<Basic> {
    use crate::core::function::{derivative, derivative_node};
    let Some((x, y)) = as_real_imag(&e.args()[0], true, None)? else {
        return Ok(derivative_node(e, &[(s.clone(), 1)]));
    };
    let numerator: Basic = add(&[
        mul(&[x.clone(), derivative(&y, s)?])?,
        mul(&[minus_one(), y.clone(), derivative(&x, s)?])?,
    ])?;
    mul(&[
        numerator,
        pow(
            &add(&[pow(&x, &Basic::int(2))?, pow(&y, &Basic::int(2))?])?,
            &minus_one(),
        )?,
    ])
}

/// The derivative of `Atan2(y, x)` in its argument of index `argindex`
/// (1 for `y`, 2 for `x`).
pub fn atan2_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let (y, x) = (&e.args()[0], &e.args()[1]);
    let denominator: Basic = add(&[pow(x, &Basic::int(2))?, pow(y, &Basic::int(2))?])?;
    if argindex == 1 {
        return mul(&[x.clone(), pow(&denominator, &minus_one())?]);
    }
    mul(&[minus_one(), y.clone(), pow(&denominator, &minus_one())?])
}

/// The facts of `Arg` (real, finite) and of `Atan2` (extended real when
/// both arguments are).
pub fn argument_facts(f: Function, args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    if f == Function::Arg {
        for p in [real, extended_real, finite, complex] {
            facts.set(p, true);
        }
        facts.set(imaginary, false);
    } else if args.iter().all(|a| facts_of(a).get(&extended_real) == Some(&true)) {
        facts.set(extended_real, true);
    }
    facts
}

/// The conjugate: `Arg` is real, `atan2(conjugate(y), conjugate(x))`.
pub fn argument_conjugate(e: &Basic, f: Function) -> PolyResult<Option<Basic>> {
    if f == Function::Atan2 {
        return Ok(Some(atan2(&conjugate(&e.args()[0])?, &conjugate(&e.args()[1])?)?));
    }
    if is_true(e, extended_real) {
        return Ok(Some(e.clone()));
    }
    Ok(None)
}

/// The value of `arg(z)` (`None` at zero) and of `atan2(y, x)` (`None`
/// unless both are real and not both zero).
pub fn argument_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    use crate::core::evalf::{complex, complex_real, real_int, Real};
    let is_zero = |r: &Real| r.repr().significand().is_zero();
    let angle = |re_part: Real, im_part: Real| -> Option<Complex> {
        if is_zero(&re_part) && is_zero(&im_part) {
            return None;
        }
        // exact on the real axis (dashu's logarithm may give -pi there)
        if is_zero(&im_part) {
            let angle: Real = if re_part > Real::ZERO {
                real_int(&crate::numbers::Int::ZERO, prec)
            } else {
                Real::pi(prec)
            };
            return Some(complex_real(angle, prec));
        }
        let ln: Complex = complex(re_part, im_part).ln();
        Some(complex_real(parts(&ln).1, prec))
    };
    match f {
        Function::Arg => {
            let (a, b) = parts(&values[0]);
            angle(a, b)
        }
        _ => {
            let ((ya, yb), (xa, xb)) = (parts(&values[0]), parts(&values[1]));
            if !is_zero(&yb) || !is_zero(&xb) {
                return None;
            }
            angle(xa, ya)
        }
    }
}
