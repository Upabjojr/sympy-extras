//! Port of `sympy_extras/concrete/dirichlet.py`: Dirichlet series
//! `Sum(f(n)*n**(-s), (n, 1, oo))` summed by pattern matching on the
//! coefficients: `zeta(s)` for `1`, the alternating series, `1/zeta(s)` for
//! `mobius(n)`, `zeta(s)/zeta(2*s)` for its square, `zeta(s - 1)/zeta(s)`
//! for `totient(n)`, `zeta(s)*zeta(s - k)` for `divisor_sigma(n, k)` and the
//! derivatives of `zeta` for the powers of `log(n)`, with the condition of
//! convergence; a sum starting above `1` gets its first terms subtracted.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::basic::{Basic, Function, Head};
use crate::error::PolyResult;

type R = PolyResult<Basic>;

/// The closed form of a Dirichlet series and its condition of
/// convergence (`DirichletSum`).
pub type DirichletSum = (Basic, Basic);

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn zeta(s: &Basic) -> R {
    crate::functions::special::zeta_functions::zeta(s, None)
}
fn re(e: &Basic) -> R {
    crate::functions::elementary::complexes::re(e)
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}

/// `(c, f, s)` with `term == c*f(n)*n**(-s)`, `c` free of `n` (`_split`).
fn split(term: &Basic, n: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    let factors: Vec<Basic> = if term.is_mul() {
        term.args().to_vec()
    } else {
        vec![term.clone()]
    };
    let mut exponent: Basic = Basic::zero();
    let mut constant: Vec<Basic> = Vec::new();
    let mut coefficient: Vec<Basic> = Vec::new();
    for f in factors {
        if !f.has(n) {
            constant.push(f);
        } else if f.is_pow() && f.args()[0] == *n && !f.args()[1].has(n) {
            exponent = add(&[exponent, f.args()[1].clone()])?;
        } else if f == *n {
            exponent = add(&[exponent, int(1)])?;
        } else {
            coefficient.push(f);
        }
    }
    Ok((mul(&constant)?, mul(&coefficient)?, neg(&exponent)?))
}

/// The series of a recognised coefficient pattern (`_closed_form`).
fn closed_form(f: &Basic, n: &Basic, s: &Basic) -> PolyResult<Option<DirichletSum>> {
    use crate::core::assumptions::Predicate;
    let one: Basic = int(1);
    if *f == one {
        return Ok(Some((zeta(s)?, gt(&re(s)?, &one)?)));
    }
    if f.is_pow() && f.args()[0] == Basic::negative_one() {
        let exponent: Basic = crate::core::add::sub(&f.args()[1], n)?;
        if !exponent.has(n) && crate::core::expr::is(&exponent, Predicate::integer) == Some(true) {
            let sign: Basic = if crate::core::expr::is(&exponent, Predicate::even) == Some(true) {
                int(-1)
            } else {
                one.clone()
            };
            // sign*(1 - 2**(1 - s)) first, as SymPy multiplies
            let signed: Basic = mul(&[
                sign,
                add(&[one.clone(), neg(&pow(&int(2), &add(&[one.clone(), neg(s)?])?)?)?])?,
            ])?;
            return Ok(Some((mul(&[signed, zeta(s)?])?, gt(&re(s)?, &Basic::zero())?)));
        }
        return Ok(None);
    }
    let applied = |e: &Basic, g: Function| e.is_function(g) && e.args()[0] == *n;
    if applied(f, Function::Mobius) {
        return Ok(Some((pow(&zeta(s)?, &int(-1))?, gt(&re(s)?, &one)?)));
    }
    if f.is_pow() && applied(&f.args()[0], Function::Mobius) && f.args()[1] == int(2) {
        let value: Basic = mul(&[zeta(s)?, pow(&zeta(&mul(&[int(2), s.clone()])?)?, &int(-1))?])?;
        return Ok(Some((value, gt(&re(s)?, &one)?)));
    }
    if applied(f, Function::Totient) {
        let value: Basic = mul(&[zeta(&add(&[s.clone(), int(-1)])?)?, pow(&zeta(s)?, &int(-1))?])?;
        return Ok(Some((value, gt(&re(s)?, &int(2))?)));
    }
    if applied(f, Function::DivisorSigma) {
        let k: Basic = if f.args().len() > 1 {
            f.args()[1].clone()
        } else {
            one.clone()
        };
        let value: Basic = mul(&[zeta(s)?, zeta(&add(&[s.clone(), neg(&k)?])?)?])?;
        let bound: Basic =
            crate::functions::elementary::miscellaneous::max_(&[one.clone(), add(&[one.clone(), re(&k)?])?])?;
        return Ok(Some((value, gt(&re(s)?, &bound)?)));
    }
    if applied(f, Function::Log) && s.is_symbol() {
        let derivative: Basic = crate::core::function::derivative_node(&zeta(s)?, &[(s.clone(), 1)]);
        return Ok(Some((neg(&derivative)?, gt(&re(s)?, &one)?)));
    }
    if f.is_pow() && applied(&f.args()[0], Function::Log) && s.is_symbol() {
        if let Some(order) = integer_of(&f.args()[1]).filter(|o| *o > 0) {
            let derivative: Basic = crate::core::function::derivative_node(&zeta(s)?, &[(s.clone(), order as u64)]);
            let sign: Basic = int(if order % 2 == 0 { 1 } else { -1 });
            return Ok(Some((mul(&[sign, derivative])?, gt(&re(s)?, &one)?)));
        }
    }
    Ok(None)
}

/// The closed form of `Sum(term, (n, lower, oo))` for a Dirichlet series
/// with recognised coefficients, with its condition of convergence
/// (`dirichlet_series`).
pub fn dirichlet_series(
    term: &Basic,
    n: &Basic,
    lower: &Basic,
    special_values: bool,
) -> PolyResult<Option<DirichletSum>> {
    let (c, f, s) = split(term, n)?;
    let Some((value, condition)) = closed_form(&f, n, &s)? else {
        return Ok(None);
    };
    let mut result: Basic = mul(&[c, value])?;
    if *lower != int(1) {
        let Some(low) = integer_of(lower).filter(|l| *l >= 1) else {
            return Ok(None);
        };
        let head_sum: Basic =
            crate::concrete::expr_with_limits::node(Head::Sum, term, &[(n.clone(), int(1), int(low - 1))])?;
        let head: Basic = crate::concrete::summations::sum_doit(&head_sum, false)?;
        result = add(&[result, neg(&head)?])?;
    }
    let value: Basic = crate::core::function::doit(&result)?;
    if !special_values {
        return Ok(Some((value, condition)));
    }
    let region: Basic = condition.clone();
    let converges = |point: &crate::concrete::karr::SumPoint| -> bool {
        let rule: HashMap<Basic, Basic> = point.iter().cloned().collect();
        match crate::core::function::xreplace_evaluated(&region, &rule) {
            Ok(b) => b != crate::core::logic::false_(),
            Err(_) => true,
        }
    };
    let oo: Basic = Basic::atom(Head::Infinity);
    let at_point = |point: &crate::concrete::karr::SumPoint| -> PolyResult<Option<Basic>> {
        let rule: HashMap<Basic, Basic> = point.iter().cloned().collect();
        let at: Basic = term.xreplace(&rule);
        Ok(Some(crate::concrete::karr::summation(
            &at,
            &[(n.clone(), lower.clone(), oo.clone())],
            &[],
            true,
            true,
        )?))
    };
    let value: Basic = crate::concrete::karr::with_special_values_sum(
        &value,
        term,
        std::slice::from_ref(n),
        &[lower.clone(), oo.clone()],
        &at_point,
        false,
        Some(&converges),
    )?;
    Ok(Some((value, condition)))
}
