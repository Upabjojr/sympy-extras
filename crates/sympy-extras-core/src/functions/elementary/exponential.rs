//! Port of `sympy_extras/functions/elementary/exponential.py`: the
//! exponential and the logarithm. `Exp(x)` and `Log(x)` are the nodes
//! ([`Head::Function`]), [`exp`] and [`log`] evaluate (`Exp.eval`,
//! `Log.eval`), with their facts ([`exp_facts`], [`log_facts`]).
//!
//! The accumulation bounds, `ExpPolar`, `LambertW` and the series are not
//! ported.

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
use crate::error::PolyResult;
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
        Head::AccumBounds => {
            // exp.eval (AccumBounds)
            let (lo, hi) = (exp(&arg.args()[0])?, exp(&arg.args()[1])?);
            return Ok(Some(crate::core::accumbounds::accum_bounds(&lo, &hi)?));
        }
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
        Head::Function(Function::ExpPolar) => {
            // log.eval (exp_polar)
            return Ok(Some(crate::functions::elementary::polar::unpolarify(
                &arg.args()[0],
                false,
            )?));
        }
        _ => {}
    }
    if matches!(arg.head(), Head::AccumBounds) {
        // log.eval (AccumBounds)
        let (lo, hi) = (&arg.args()[0], &arg.args()[1]);
        if is_true(lo, positive) {
            return Ok(Some(crate::core::accumbounds::accum_bounds(&log(lo)?, &log(hi)?)?));
        }
        if is_true(lo, zero) {
            return Ok(Some(crate::core::accumbounds::accum_bounds(
                &Basic::negative_oo(),
                &log(hi)?,
            )?));
        }
        return Ok(Some(Basic::nan()));
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
                    return log_of_angle(&coefficient, &rest, &real_part, &imaginary_part);
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

/// `tan(a) -> a` for the angles of the first quadrant with algebraic
/// tangents (`_log_atan_table`), the keys built in the order of the
/// Python's operators.
fn log_atan_table() -> PolyResult<Vec<(Basic, Basic)>> {
    let int = Basic::int;
    let sqrt = |e: &Basic| pow(e, &Basic::half());
    let s = |n: i64| sqrt(&int(n));
    let div = |a: &Basic, b: &Basic| mul(&[a.clone(), pow(b, &Basic::negative_one())?]);
    let sub = |a: &Basic, b: &Basic| add(&[a.clone(), mul(&[Basic::negative_one(), b.clone()])?]);
    let plus = |a: &Basic, b: &Basic| add(&[a.clone(), b.clone()]);
    let pi_q = |p: i64, q: i64| mul(&[Basic::pi(), Basic::from_rational(rat(p, q))]);
    let (s2, s3, s5) = (s(2)?, s(3)?, s(5)?);
    let one: Basic = int(1);
    let two_s5: Basic = mul(&[int(2), s5.clone()])?;
    let two_s5_5: Basic = div(&two_s5, &int(5))?;
    Ok(vec![
        (s3.clone(), pi_q(1, 3)?),
        (one.clone(), pi_q(1, 4)?),
        (sqrt(&sub(&int(5), &two_s5)?)?, pi_q(1, 5)?),
        (
            div(&mul(&[s2.clone(), sqrt(&sub(&int(5), &s5)?)?])?, &plus(&one, &s5)?)?,
            pi_q(1, 5)?,
        ),
        (sqrt(&plus(&int(5), &two_s5)?)?, pi_q(2, 5)?),
        (
            div(&mul(&[s2.clone(), sqrt(&plus(&s5, &int(5))?)?])?, &plus(&int(-1), &s5)?)?,
            pi_q(2, 5)?,
        ),
        (div(&s3, &int(3))?, pi_q(1, 6)?),
        (sub(&s2, &one)?, pi_q(1, 8)?),
        (
            div(&sqrt(&sub(&int(2), &s2)?)?, &sqrt(&plus(&s2, &int(2))?)?)?,
            pi_q(1, 8)?,
        ),
        (plus(&s2, &one)?, pi_q(3, 8)?),
        (
            div(&sqrt(&plus(&s2, &int(2))?)?, &sqrt(&sub(&int(2), &s2)?)?)?,
            pi_q(3, 8)?,
        ),
        (sqrt(&sub(&one, &two_s5_5)?)?, pi_q(1, 10)?),
        (
            div(
                &plus(&mul(&[Basic::negative_one(), s2.clone()])?, &s(10)?)?,
                &mul(&[int(2), sqrt(&plus(&s5, &int(5))?)?])?,
            )?,
            pi_q(1, 10)?,
        ),
        (sqrt(&plus(&one, &two_s5_5)?)?, pi_q(3, 10)?),
        (
            div(&plus(&s2, &s(10)?)?, &mul(&[int(2), sqrt(&sub(&int(5), &s5)?)?])?)?,
            pi_q(3, 10)?,
        ),
        (sub(&int(2), &s3)?, pi_q(1, 12)?),
        (div(&plus(&int(-1), &s3)?, &plus(&one, &s3)?)?, pi_q(1, 12)?),
        (plus(&int(2), &s3)?, pi_q(5, 12)?),
        (div(&plus(&one, &s3)?, &plus(&int(-1), &s3)?)?, pi_q(5, 12)?),
    ])
}

/// `log(coefficient*(r + i*I)) = log(|...|) + I*a` when `i/r` is the
/// tangent of a rational multiple `a` of pi in the table (`_log_of_angle`).
fn log_of_angle(
    coefficient: &Basic,
    rest: &Basic,
    real_part: &Basic,
    imaginary_part: &Basic,
) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::inverse::lookup;
    use crate::polys::polytools::cancel;
    let i: Basic = Basic::imaginary_unit();
    let t: Basic = cancel(
        &mul(&[imaginary_part.clone(), pow(real_part, &Basic::negative_one())?])?,
        &[],
    )?;
    let t1: Basic = cancel(&mul(&[Basic::negative_one(), t.clone()])?, &[])?;
    let modulus = || -> PolyResult<Basic> {
        crate::simplify::ratsimp::ratsimp(&mul(&[coefficient.clone(), super::complexes::abs_(rest)?])?)
    };
    let positive_real: bool = is_true(real_part, Predicate::positive);
    if let Some(angle) = lookup("log_atan", log_atan_table, &t)? {
        let a: Basic = if positive_real {
            angle
        } else {
            add(&[angle, mul(&[Basic::negative_one(), Basic::pi()])?])?
        };
        return Ok(Some(add(&[log(&modulus()?)?, mul(&[i, a])?])?));
    }
    if let Some(angle) = lookup("log_atan", log_atan_table, &t1)? {
        let a: Basic = if positive_real {
            mul(&[Basic::negative_one(), angle])?
        } else {
            add(&[Basic::pi(), mul(&[Basic::negative_one(), angle])?])?
        };
        return Ok(Some(add(&[log(&modulus()?)?, mul(&[i, a])?])?));
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// LambertW

// PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:LambertW (SymPy 1.14), through
// sympy_extras

/// The node `LambertW(x)` or `LambertW(x, k)`, unevaluated.
pub fn lambertw_node(args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(Function::LambertW), args)
}

fn is_zero_truth(e: &Basic) -> Option<bool> {
    crate::core::expr::is(e, zero)
}

/// `LambertW.eval`: the values at the special points; `None` otherwise.
pub fn lambertw_eval(args: &[Basic]) -> PolyResult<Option<Basic>> {
    let x: &Basic = &args[0];
    let k: Basic = if args.len() > 1 { args[1].clone() } else { Basic::zero() };
    if args.len() > 1 && is_zero_truth(&k) == Some(true) {
        return Ok(Some(lambertw(x, None)?));
    }
    let e_inv: Basic = pow(&Basic::e(), &Basic::negative_one())?;
    let minus_one_over_e: Basic = mul(&[Basic::negative_one(), e_inv])?;
    let half = |c: i64| Basic::from_rational(rat(c, 2));
    if is_zero_truth(&k) == Some(true) {
        if is_zero_truth(x) == Some(true) {
            return Ok(Some(Basic::zero()));
        }
        if matches!(x.head(), Head::Exp1) {
            return Ok(Some(Basic::one()));
        }
        if *x == minus_one_over_e {
            return Ok(Some(Basic::negative_one()));
        }
        let log2: Basic = log(&Basic::int(2))?;
        if *x == mul(&[half(-1), log2.clone()])? {
            return Ok(Some(mul(&[Basic::negative_one(), log2])?));
        }
        if *x == mul(&[Basic::int(2), log2.clone()])? {
            return Ok(Some(log2));
        }
        if *x == mul(&[half(-1), Basic::pi()])? {
            return Ok(Some(mul(&[Basic::imaginary_unit(), Basic::pi(), half(1)])?));
        }
        if *x == exp(&add(&[Basic::one(), Basic::e()])?)? {
            return Ok(Some(Basic::e()));
        }
        if matches!(x.head(), Head::Infinity) {
            return Ok(Some(x.clone()));
        }
    }
    if is_zero_truth(&k) == Some(false) && is_zero_truth(x) == Some(true) {
        return Ok(Some(Basic::negative_oo()));
    }
    if k == Basic::negative_one() {
        if *x == mul(&[half(-1), Basic::pi()])? {
            return Ok(Some(mul(&[half(-1), Basic::imaginary_unit(), Basic::pi()])?));
        }
        if *x == minus_one_over_e {
            return Ok(Some(Basic::negative_one()));
        }
        if *x == mul(&[Basic::int(-2), exp(&Basic::int(-2))?])? {
            return Ok(Some(Basic::int(-2)));
        }
    }
    Ok(None)
}

/// `LambertW(x)` or the branch `LambertW(x, k)`, evaluated (`lambertw`).
pub fn lambertw(x: &Basic, k: Option<&Basic>) -> PolyResult<Basic> {
    let args: Vec<Basic> = match k {
        Some(k) => vec![x.clone(), k.clone()],
        None => vec![x.clone()],
    };
    if !evaluate() {
        return Ok(lambertw_node(args));
    }
    Ok(lambertw_eval(&args)?.unwrap_or_else(|| lambertw_node(args)))
}

/// `LambertW.fdiff(1)`: `W/(x (1 + W))`.
pub fn lambertw_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    if argindex != 1 {
        return Err(crate::error::not_ported("the derivative of LambertW in its branch"));
    }
    let w: Basic = e.clone();
    mul(&[
        w.clone(),
        pow(
            &mul(&[e.args()[0].clone(), add(&[Basic::one(), w])?])?,
            &Basic::negative_one(),
        )?,
    ])
}

/// The facts of `LambertW` (`_eval_facts`).
pub fn lambertw_facts(args: &[Basic]) -> FactList {
    let x: &Basic = &args[0];
    let k: Basic = if args.len() > 1 { args[1].clone() } else { Basic::zero() };
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    let ask = |e: &Basic, p: Predicate| crate::core::expr::is(e, p);
    let Ok(e_inv) = pow(&Basic::e(), &Basic::negative_one()) else {
        return facts;
    };
    let Ok(shifted) = add(&[x.clone(), e_inv]) else {
        return facts;
    };
    let Ok(k1) = add(&[k.clone(), Basic::one()]) else {
        return facts;
    };
    let mut is_real: Option<bool> = None;
    if ask(&k, zero) == Some(true) {
        if ask(&shifted, positive) == Some(true) {
            is_real = Some(true);
        } else if ask(&shifted, nonpositive) == Some(true) {
            is_real = Some(false);
        }
    } else if ask(&k1, zero) == Some(true) {
        if ask(x, negative) == Some(true) && ask(&shifted, positive) == Some(true) {
            is_real = Some(true);
        } else if ask(x, nonpositive) == Some(true) || ask(&shifted, nonnegative) == Some(true) {
            is_real = Some(false);
        }
    } else if ask(&k, zero) == Some(false) && ask(&k1, zero) == Some(false) && ask(x, extended_real) == Some(true) {
        is_real = Some(false);
    }
    if let Some(v) = is_real {
        facts.set(extended_real, v);
    }
    let a = facts_of(x);
    if let Some(v) = a.get(&finite) {
        facts.set(finite, *v);
    }
    if args.len() == 1 {
        if let Some(v) = a.get(&zero) {
            facts.set(zero, *v);
        }
    } else {
        let kz: Option<bool> = facts_of(&k).get(&zero).copied();
        let az: Option<bool> = a.get(&zero).copied();
        if az == Some(false) || kz == Some(false) {
            facts.set(zero, false);
        } else if az == Some(true) && kz == Some(true) {
            facts.set(zero, true);
        }
    }
    if a.get(&zero) == Some(&false) && a.get(&algebraic) == Some(&true) {
        facts.set(algebraic, false);
    }
    facts
}

/// mpmath's `lambertw(x, k)` at `prec` bits: Halley's iteration from the
/// series at the branch point near `-1/e`, `x` itself near zero (the
/// principal branch), the asymptotic `L1 - L2 + L2/L1` elsewhere.
fn is_zero_c(z: &crate::core::evalf::Complex) -> bool {
    let (a, b) = crate::core::evalf::parts(z);
    a.repr().significand().is_zero() && b.repr().significand().is_zero()
}

pub fn lambertw_value(values: &[crate::core::evalf::Complex], prec: usize) -> Option<crate::core::evalf::Complex> {
    use crate::core::evalf::{complex, complex_sqrt, parts, real_int, Complex, Real};
    let finite_c = |z: &Complex| {
        let (a, b) = parts(z);
        !a.repr().is_infinite() && !b.repr().is_infinite()
    };
    if values.iter().any(|v| !finite_c(v)) {
        return None;
    }
    let x: Complex = values[0].clone();
    let k: i64 = if values.len() > 1 {
        let (kr, ki) = parts(&values[1]);
        if !ki.repr().significand().is_zero() {
            return None;
        }
        let kv: f64 = kr.to_f64().value();
        if kv.fract() != 0.0 || !kv.is_finite() {
            return None;
        }
        kv as i64
    } else {
        0
    };
    let wp: usize = prec + 40;
    let r = |v: f64| -> Real { Real::try_from(v).expect("finite").with_precision(wp).value() };
    let c = |a: f64, b: f64| -> Complex { complex(r(a), r(b)) };
    let at = |z: &Complex| -> Complex {
        let (a, b) = parts(z);
        complex(a.with_precision(wp).value(), b.with_precision(wp).value())
    };
    let mut x: Complex = at(&x);
    // a real argument evaluated with noise in its imaginary part (-pi/2 as
    // -1.57... - 1e-60 I) is real, as mpmath's mpf is: the side of the cut
    // of log decides the branch
    {
        let (a, b) = parts(&x);
        let (af, bf) = (a.to_f64().value(), b.to_f64().value());
        if bf.abs() <= af.abs().max(1.0) * 2f64.powi(-((prec as i32).min(1000) - 8)) {
            x = complex(a, r(0.0));
        }
    }
    let (xr, xi) = parts(&x);
    let x_is_zero: bool = xr.repr().significand().is_zero() && xi.repr().significand().is_zero();
    if x_is_zero {
        return if k == 0 { Some(c(0.0, 0.0)) } else { None };
    }
    let e: Complex = complex(Real::e(wp), r(0.0));
    let (xf, yf) = (xr.to_f64().value(), xi.to_f64().value());
    let near_branch: bool = ((xf + (-1.0f64).exp()).powi(2) + yf * yf).sqrt() < 0.3;
    let one: Complex = c(1.0, 0.0);
    let two: Complex = c(2.0, 0.0);
    let mut w: Complex = if (k == 0 || (k == -1 && yf == 0.0 && xf < 0.0)) && near_branch {
        // p = sqrt(2 (e x + 1)), w = -1 + p - p**2/3 + 11 p**3/72 (k = 0),
        // with -p on the branch -1
        let mut p: Complex = complex_sqrt(&(&two * &(&(&e * &x) + &one)), wp);
        if k == -1 {
            p = -p;
        }
        let p2: Complex = &p * &p;
        let p3: Complex = &p2 * &p;
        &(&(&p - &one) - &(&p2 / &c(3.0, 0.0))) + &(&p3 * &c(11.0 / 72.0, 0.0))
    } else if k == 0 && (xf * xf + yf * yf).sqrt() < 0.5 {
        x.clone()
    } else if k == 0 && (xf * xf + yf * yf).sqrt() <= 3.0 && ((xf + 1.0).powi(2) + yf * yf).sqrt() > 0.5 {
        // log(1 + x): the asymptotic start has log(x) = 0 at x = 1
        (&x + &one).ln()
    } else if k == -1 && yf == 0.0 && xf < 0.0 && xf > -(-1.0f64).exp() {
        // the real branch -1 on (-1/e, 0): log(-x) - log(-log(-x))
        let l1: Complex = complex(real_int(&Int::from(0), wp), r(0.0)) + (-&x).ln();
        let l2: Complex = (-&l1).ln();
        &(&l1 - &l2) + &(&l2 / &l1)
    } else {
        let two_pi_i_k: Complex = complex(r(0.0), Real::pi(wp) * real_int(&Int::from(2 * k), wp));
        let l1: Complex = &x.ln() + &two_pi_i_k;
        if is_zero_c(&l1) {
            return None;
        }
        let l2: Complex = l1.ln();
        &(&l1 - &l2) + &(&l2 / &l1)
    };
    let tolerance: f64 = 2f64.powi(-((prec as i32).min(1000)));
    for _ in 0..200 {
        {
            let (a, b) = parts(&w);
            if a.repr().is_infinite() || b.repr().is_infinite() || a.repr().exponent() > 40 {
                return None;
            }
        }
        let ew: Complex = w.exp();
        {
            let (a, b) = parts(&ew);
            if a.repr().is_infinite() || b.repr().is_infinite() {
                return None;
            }
        }
        let f: Complex = &(&w * &ew) - &x;
        let wp1: Complex = &w + &one;
        if is_zero_c(&wp1) {
            return None;
        }
        let denominator: Complex = &(&ew * &wp1) - &(&(&(&w + &two) * &f) / &(&two * &wp1));
        if is_zero_c(&denominator) || !finite_c(&denominator) {
            return None;
        }
        let step: Complex = &f / &denominator;
        if !finite_c(&step) {
            return None;
        }
        w = &w - &step;
        let (sr, si) = parts(&step);
        let (wr, wi) = parts(&w);
        if wr.repr().is_infinite() || wi.repr().is_infinite() || sr.repr().is_infinite() || si.repr().is_infinite() {
            return None;
        }
        let size: f64 = (sr.to_f64().value().powi(2) + si.to_f64().value().powi(2)).sqrt();
        let scale: f64 = (wr.to_f64().value().powi(2) + wi.to_f64().value().powi(2))
            .sqrt()
            .max(1e-300);
        if size <= tolerance * scale || size == 0.0 {
            return Some(at(&w));
        }
    }
    None
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
