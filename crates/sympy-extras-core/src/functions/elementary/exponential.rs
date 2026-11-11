//! Port of `sympy_extras/functions/elementary/exponential.py`: the
//! exponential and the logarithm. `Exp(x)` and `Log(x)` are the nodes
//! ([`Head::Function`]), [`exp`] and [`log`] evaluate (`Exp.eval`,
//! `Log.eval`), with their facts ([`exp_facts`], [`log_facts`]).
//!
//! The log of an algebraic number `r + i*I` with `r != 0` needs the
//! arctangent table and is the error `NotImplemented`. The accumulation
//! bounds, `ExpPolar`, `LambertW` and the series are not ported.

// PORTED FROM SYMPY: sympy/functions/elementary/exponential.py (SymPy 1.14),
// through sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::assumptions::{facts_of, is_number, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{
    as_coeff_mul, as_coefficient, as_independent, factors_of, is_comparable, is_rational_value, is_true,
};
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::{not_ported, PolyResult};
use crate::numbers::*;

use Predicate::*;

/// The node `Exp(x)`, unevaluated.
pub fn exp_node(x: Basic) -> Basic {
    Basic::new(Head::Function(Function::Exp), vec![x])
}

/// The node `Log(x)`, unevaluated.
pub fn log_node(x: Basic) -> Basic {
    Basic::new(Head::Function(Function::Log), vec![x])
}

fn pi_i() -> PolyResult<Basic> {
    mul(&[Basic::pi(), Basic::imaginary_unit()])
}

/// `c` with `arg = c*pi*I` (`_coefficient_of_pi_i`).
fn coefficient_of_pi_i(arg: &Basic) -> PolyResult<Option<Basic>> {
    as_coefficient(arg, &pi_i()?)
}

/// `Exp.eval`: the evaluated form, `None` when the node stays.
pub fn exp_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    match arg.head() {
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::Rational { value, .. } if value.is_zero() => return Ok(Some(Basic::one())),
        Head::Rational { value, .. } if *value == Rat::ONE => return Ok(Some(Basic::e())),
        Head::Infinity => return Ok(Some(Basic::oo())),
        Head::NegativeInfinity => return Ok(Some(Basic::zero())),
        Head::ComplexInfinity => return Ok(Some(Basic::nan())),
        Head::Function(Function::Log) => return Ok(Some(arg.args()[0].clone())),
        _ => {}
    }
    if arg.is_mul() {
        if let Some(coeff) = coefficient_of_pi_i(arg)? {
            let twice: Basic = mul(&[Basic::int(2), coeff.clone()])?;
            if is_true(&twice, integer) {
                if is_true(&coeff, even) {
                    return Ok(Some(Basic::one()));
                }
                if is_true(&coeff, odd) {
                    return Ok(Some(Basic::negative_one()));
                }
                let shifted: Basic = add(&[coeff.clone(), Basic::half()])?;
                if is_true(&shifted, even) {
                    return Ok(Some(mul(&[Basic::negative_one(), Basic::imaginary_unit()])?));
                }
                if is_true(&shifted, odd) {
                    return Ok(Some(Basic::imaginary_unit()));
                }
            } else if let Some(c) = coeff.as_rational() {
                // into (-1, 1]
                let two: Rat = rat(2, 1);
                let mut reduced: Rat = c - &two * rat_from_int(&div_floor(&numer(c), &(denom(c) * int(2))));
                if reduced > Rat::ONE {
                    reduced -= two;
                }
                if reduced != *c {
                    let angle: Basic = mul(&[Basic::from_rational(reduced), Basic::pi(), Basic::imaginary_unit()])?;
                    return Ok(Some(exp(&angle)?));
                }
            }
        }
        // a single log factor: exp(c*log(x)) is x**c
        let (number, terms) = as_coeff_mul(arg);
        if matches!(number.head(), Head::Infinity | Head::NegativeInfinity) {
            if is_number(&terms) {
                use super::complexes::{im, re};
                let direction: Basic = if matches!(number.head(), Head::NegativeInfinity) {
                    mul(&[Basic::negative_one(), terms])?
                } else {
                    terms
                };
                let real_part: Basic = re(&direction)?;
                if is_true(&real_part, zero) && !direction.as_rational().is_some_and(|v| v.is_zero()) {
                    return Ok(Some(Basic::nan()));
                }
                if is_true(&real_part, positive) && !im(&direction)?.as_rational().is_some_and(|v| v.is_zero()) {
                    return Ok(Some(Basic::zoo()));
                }
                if is_true(&real_part, negative) {
                    return Ok(Some(Basic::zero()));
                }
            }
            return Ok(None);
        }
        let mut coefficients: Vec<Basic> = vec![number];
        let mut log_term: Option<Basic> = None;
        for term in factors_of(&terms) {
            if term.is_function(Function::Log) {
                if log_term.is_some() {
                    return Ok(None);
                }
                log_term = Some(term.args()[0].clone());
            } else if is_comparable(&term) {
                coefficients.push(term);
            } else {
                return Ok(None);
            }
        }
        return match log_term {
            Some(b) => Ok(Some(pow(&b, &mul(&coefficients)?)?)),
            None => Ok(None),
        };
    }
    if arg.is_add() {
        let mut out: Vec<Basic> = Vec::new();
        let mut kept: Vec<Basic> = Vec::new();
        let mut changed: bool = false;
        for a in arg.args() {
            if is_rational_value(a, 1) {
                kept.push(a.clone());
                continue;
            }
            let value: Basic = exp(a)?;
            if value.is_function(Function::Exp) {
                if value.args()[0] != *a {
                    kept.push(value.args()[0].clone());
                    changed = true;
                } else {
                    kept.push(a.clone());
                }
            } else {
                out.push(value);
            }
        }
        if !out.is_empty() || changed {
            if kept.is_empty() {
                return Ok(Some(mul(&out)?));
            }
            out.push(exp_node(add(&kept)?));
            return Ok(Some(mul(&out)?));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::one()));
    }
    Ok(None)
}

/// The exponential of `x`, evaluated (`exp`, `Exp.apply`).
pub fn exp(x: &Basic) -> PolyResult<Basic> {
    if evaluate() {
        if let Some(value) = exp_eval(x)? {
            return Ok(value);
        }
    }
    Ok(exp_node(x.clone()))
}

/// `Log.eval`: the evaluated form, `None` when the node stays.
pub fn log_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if let Some(v) = arg.as_rational() {
        if v.is_zero() {
            return Ok(Some(Basic::zoo()));
        }
        if *v == Rat::ONE {
            return Ok(Some(Basic::zero()));
        }
        if numer(v) == Int::ONE {
            return Ok(Some(mul(&[Basic::negative_one(), log(&Basic::integer(denom(v)))?])?));
        }
    }
    match arg.head() {
        Head::Infinity | Head::NegativeInfinity => return Ok(Some(Basic::oo())),
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::Function(Function::Exp) if is_true(&arg.args()[0], extended_real) => {
            return Ok(Some(arg.args()[0].clone()))
        }
        _ => {}
    }
    if is_number(arg) {
        if is_true(arg, negative) {
            let negated: Basic = mul(&[Basic::negative_one(), arg.clone()])?;
            return Ok(Some(add(&[pi_i()?, log(&negated)?])?));
        }
        if matches!(arg.head(), Head::ComplexInfinity) {
            return Ok(Some(Basic::zoo()));
        }
        if matches!(arg.head(), Head::Exp1) {
            return Ok(Some(Basic::one()));
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::zoo()));
    }
    let i: Basic = Basic::imaginary_unit();
    if !arg.is_add() {
        if let Some(coeff) = as_coefficient(arg, &i)? {
            if matches!(coeff.head(), Head::Infinity | Head::NegativeInfinity) {
                return Ok(Some(Basic::oo()));
            }
            if let Some(c) = coeff.as_rational() {
                let half_pi_i: Basic = mul(&[Basic::pi(), i.clone(), Basic::half()])?;
                if !rat_is_negative(c) {
                    return Ok(Some(add(&[half_pi_i, log(&coeff)?])?));
                }
                let minus: Basic = mul(&[Basic::negative_one(), half_pi_i])?;
                return Ok(Some(add(&[minus, log(&Basic::from_rational(-c))?])?));
            }
        }
    }
    if is_number(arg) && is_true(arg, algebraic) {
        // arg = coefficient*(r + i*I) with a positive coefficient
        let (mut coefficient, mut rest) = as_independent(arg, &i, false)?;
        if is_true(&coefficient, negative) {
            coefficient = mul(&[Basic::negative_one(), coefficient])?;
            rest = mul(&[Basic::negative_one(), rest])?;
        }
        if !rest.has(&i) {
            // no imaginary part after expand_mul either
            return Ok(None);
        }
        let rest: Basic = crate::core::function::expand_mul(&rest, true)?;
        let (real_part, imaginary_term) = as_independent(&rest, &i, true)?;
        let imaginary_part: Option<Basic> = as_coefficient(&imaginary_term, &i)?;
        if let Some(imaginary_part) = imaginary_part {
            let not_zero: bool = !(imaginary_part.is_number_class() && is_rational_value(&imaginary_part, 0));
            if is_true(&coefficient, real) && not_zero && is_true(&imaginary_part, real) && is_true(&real_part, real) {
                if !is_true(&real_part, zero) {
                    return Err(not_ported("the arctangent table of log"));
                }
                let half_pi_i: Basic = mul(&[Basic::pi(), i.clone(), Basic::half()])?;
                if is_true(&imaginary_part, positive) {
                    return Ok(Some(add(&[half_pi_i, log(&mul(&[coefficient, imaginary_part])?)?])?));
                }
                if is_true(&imaginary_part, negative) {
                    let minus: Basic = mul(&[Basic::negative_one(), half_pi_i])?;
                    let modulus: Basic = mul(&[coefficient, mul(&[Basic::negative_one(), imaginary_part])?])?;
                    return Ok(Some(add(&[minus, log(&modulus)?])?));
                }
            }
        }
    }
    Ok(None)
}

/// The natural logarithm of `x`, evaluated (`Log.apply`).
pub fn log(x: &Basic) -> PolyResult<Basic> {
    if evaluate() {
        if let Some(value) = log_eval(x)? {
            return Ok(value);
        }
    }
    Ok(log_node(x.clone()))
}

/// The largest `k` with `base**k` dividing `n` (`_multiplicity`).
fn multiplicity(base: &Int, n: &Int) -> i64 {
    let mut k: i64 = 0;
    let mut m: Int = n.clone();
    while mod_floor(&m, base).is_zero() {
        m = div_floor(&m, base);
        k += 1;
    }
    k
}

/// `log(x, b)`: `log(x)/log(b)`, the powers of an integer base taken out of
/// a rational argument.
pub fn log_base(x: &Basic, base: &Basic) -> PolyResult<Basic> {
    if is_rational_value(base, 1) {
        return Ok(if is_rational_value(x, 1) {
            Basic::nan()
        } else {
            Basic::zoo()
        });
    }
    if matches!(base.head(), Head::Exp1) {
        return log(x);
    }
    if let (Some(b), Some(a)) = (base.as_rational(), x.as_rational()) {
        if b.is_int() && numer(b) > Int::ONE && !a.is_zero() {
            let p: Int = numer(b);
            let n: i64 = multiplicity(&p, &abs(&numer(a))) - multiplicity(&p, &denom(a));
            if n != 0 {
                let rest: Basic = mul(&[x.clone(), pow(base, &Basic::int(-n))?])?;
                let quotient: Basic = mul(&[log(&rest)?, pow(&log(base)?, &Basic::negative_one())?])?;
                return add(&[Basic::int(n), quotient]);
            }
        }
    }
    mul(&[log(x)?, pow(&log(base)?, &Basic::negative_one())?])
}

/// The facts of `Exp(arg)` (`Exp._eval_facts`).
pub fn exp_facts(arg: &Basic) -> FactList {
    let a = facts_of(arg);
    let yes = |p: Predicate| a.get(&p) == Some(&true);
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    let minus_oo: bool = matches!(arg.head(), Head::NegativeInfinity);
    facts.set(zero, minus_oo);
    if yes(extended_real) {
        facts.set(extended_real, true);
        if !minus_oo {
            facts.set(extended_positive, true);
        }
    }
    if yes(real) {
        facts.update(&[(real, true), (positive, true)]);
    }
    if yes(finite) {
        facts.update(&[(finite, true), (zero, false)]);
    }
    if yes(complex) {
        facts.set(complex, true);
    }
    if a.get(&zero) == Some(&false) {
        if yes(algebraic) {
            facts.set(algebraic, false);
        }
        if yes(rational) {
            facts.set(rational, false);
        }
    }
    if arg.is_mul() {
        if let Ok(Some(c)) = coefficient_of_pi_i(arg) {
            if is_true(&c, rational) {
                facts.set(algebraic, true);
            }
        }
    }
    facts
}

/// The facts of `Log(arg)` (`Log._eval_facts`).
pub fn log_facts(arg: &Basic) -> FactList {
    let a = facts_of(arg);
    let get = |p: Predicate| a.get(&p).copied();
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    match get(extended_positive) {
        Some(true) => facts.set(extended_real, true),
        Some(false) => facts.set(extended_real, false),
        None => {}
    }
    if get(positive) == Some(true) && get(finite) == Some(true) {
        facts.set(real, true);
    }
    if get(complex) == Some(true) && get(zero) == Some(false) {
        facts.set(complex, true);
    }
    if get(zero) == Some(true) {
        facts.set(finite, false);
    } else if get(finite) == Some(true) && get(zero) == Some(false) {
        facts.set(finite, true);
    }
    let Ok(shifted_node) = add(&[arg.clone(), Basic::negative_one()]) else {
        return facts;
    };
    let shifted = facts_of(&shifted_node);
    for p in [extended_positive, zero, extended_nonnegative] {
        if let Some(v) = shifted.get(&p) {
            facts.set(p, *v);
        }
    }
    match shifted.get(&zero) {
        Some(true) => facts.update(&[(rational, true), (algebraic, true)]),
        Some(false) => {
            if get(rational) == Some(true) {
                facts.set(rational, false);
            }
            if get(algebraic) == Some(true) {
                facts.set(algebraic, false);
            }
        }
        None => {}
    }
    facts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    #[test]
    fn module_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let y: Basic = Symbol::new("y", &[]).0;
        let r: Basic = Symbol::new("r", &[(real, true)]).0;
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        assert_eq!(show(exp(&Basic::zero())), "1");
        assert_eq!(show(exp(&Basic::one())), "E");
        assert_eq!(show(exp(&log(&x).unwrap())), "x");
        assert_eq!(show(exp(&mul(&[Basic::int(2), log(&x).unwrap()]).unwrap())), "x**2");
        assert_eq!(show(exp(&add(&[x.clone(), log(&y).unwrap()]).unwrap())), "y*exp(x)");
        let angle = |c: Rat| mul(&[Basic::from_rational(c), Basic::pi(), Basic::imaginary_unit()]).unwrap();
        assert_eq!(show(exp(&angle(rat(3, 2)))), "-I");
        assert_eq!(show(exp(&angle(rat(7, 3)))), "exp(I*pi/3)");
        assert_eq!(show(log(&Basic::one())), "0");
        assert_eq!(show(log(&Basic::e())), "1");
        assert_eq!(show(log(&Basic::int(-3))), "log(3) + I*pi");
        assert_eq!(
            show(log(&mul(&[Basic::int(3), Basic::imaginary_unit()]).unwrap())),
            "log(3) + I*pi/2"
        );
        assert_eq!(show(log(&Basic::from_rational(rat(1, 7)))), "-log(7)");
        assert_eq!(show(log(&exp(&r).unwrap())), "r");
        assert_eq!(show(log_base(&Basic::int(24), &Basic::int(2))), "log(3)/log(2) + 3");
        assert_eq!(show(log_base(&Basic::from_rational(rat(1, 8)), &Basic::int(2))), "-3");
    }
}
