//! Port of `sympy_extras/functions/special/beta_functions.py`: the Euler
//! beta [`Function::Beta`] and the incomplete betas
//! [`Function::BetaInc`], [`Function::BetaIncRegularized`]: evaluation
//! (SymPy's rules for numbers: the poles, `beta(x, 1)`, `beta(x, x + 1)`
//! by `catalan`, a negative integer sum of non-integers), derivatives,
//! conjugates, facts and the value of `beta` (mpmath's `gammaprod`); the
//! incomplete betas have no value here (mpmath's `betainc` is not ported).

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::Complex;
use crate::core::expr::{is, is_true};
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::error::{not_ported, PolyResult};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

/// `beta.doit` for two numbers (`_beta_numbers`).
fn beta_numbers(x: &Basic, y: &Basic) -> PolyResult<Option<Basic>> {
    let one: Basic = Basic::one();
    if is_true(y, Predicate::zero) || is_true(x, Predicate::zero) {
        return Ok(Some(Basic::zoo()));
    }
    if *y == one {
        return Ok(Some(div(&one, x)?));
    }
    if *x == one {
        return Ok(Some(div(&one, y)?));
    }
    if *y == add(&[x.clone(), one.clone()])? {
        let c: Basic = crate::functions::combinatorial::numbers::catalan(x)?;
        return Ok(Some(div(&one, &mul(&[x.clone(), y.clone(), c])?)?));
    }
    let s: Basic = add(&[x.clone(), y.clone()])?;
    if is_true(&s, Predicate::integer)
        && is_true(&s, Predicate::negative)
        && crate::core::expr::is_false(x, Predicate::integer)
        && crate::core::expr::is_false(y, Predicate::integer)
    {
        return Ok(Some(Basic::zero()));
    }
    Ok(None)
}

/// The evaluating function of the three heads.
pub fn beta_function(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    match f {
        Function::Beta => {
            if args.len() == 1 {
                return beta(&args[0], Some(&args[0]));
            }
            let (x, y) = (&args[0], &args[1]);
            if x.is_number_node() && y.is_number_node() {
                if let Some(v) = beta_numbers(x, y)? {
                    return Ok(v);
                }
            }
            Ok(node(f, args.to_vec()))
        }
        _ => Ok(node(f, args.to_vec())),
    }
}

pub fn is_beta_function(f: Function) -> bool {
    matches!(f, Function::Beta | Function::BetaInc | Function::BetaIncRegularized)
}

/// `beta(x, y)` (`beta(x, x)` for `None`), evaluated.
pub fn beta(x: &Basic, y: Option<&Basic>) -> PolyResult<Basic> {
    match y {
        None => beta_function(Function::Beta, std::slice::from_ref(x)),
        Some(y) => beta_function(Function::Beta, &[x.clone(), y.clone()]),
    }
}

pub fn betainc(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    beta_function(f, args)
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn beta_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::functions::special::gamma_functions::digamma;
    let args: &[Basic] = e.args();
    let one: Basic = Basic::one();
    match f {
        Function::Beta => {
            let (x, y) = (&args[0], &args[args.len() - 1]);
            let own: &Basic = if argindex == 1 { x } else { y };
            let difference: Basic = sub(&digamma(own)?, &digamma(&add(&[x.clone(), y.clone()])?)?)?;
            mul(&[beta(x, Some(y))?, difference])
        }
        _ => {
            let (a, b) = (&args[0], &args[1]);
            if argindex != 3 && argindex != 4 {
                let v: &Basic = &args[argindex - 1];
                let elsewhere: bool = args
                    .iter()
                    .enumerate()
                    .any(|(i, w)| i != argindex - 1 && w.free_symbols().contains(v));
                if v.is_symbol() && !elsewhere {
                    return Ok(crate::core::function::derivative_node(e, &[(v.clone(), 1)]));
                }
                return Err(not_ported("a derivative as a Subs"));
            }
            let x: &Basic = &args[argindex - 1];
            let mut value: Basic = mul(&[pow(&sub(&one, x)?, &sub(b, &one)?)?, pow(x, &sub(a, &one)?)?])?;
            if argindex == 3 {
                value = mul(&[Basic::negative_one(), value])?;
            }
            if f == Function::BetaIncRegularized {
                return div(&value, &beta(a, Some(b))?);
            }
            Ok(value)
        }
    }
}

/// The conjugate (`_eval_conjugate`).
pub fn beta_conjugate(f: Function, e: &Basic) -> PolyResult<Basic> {
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(crate::functions::elementary::complexes::conjugate(a)?);
    }
    beta_function(f, &args)
}

/// The facts (`_eval_facts`).
pub fn beta_facts(f: Function, args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    let real: Vec<Option<bool>> = args.iter().map(|a| is(a, Predicate::real)).collect();
    if f == Function::Beta {
        if real.iter().all(|v| *v == Some(true)) {
            facts.set(Predicate::real, true);
        } else if real[0] == Some(false) || (real[0] == Some(true) && real[real.len() - 1] == Some(false)) {
            facts.set(Predicate::real, false);
        }
    } else if real.iter().all(|v| *v == Some(true)) {
        facts.set(Predicate::real, true);
    }
    facts
}

/// The value (`numeric`): `beta` by the gamma function; none for the
/// incomplete betas.
pub fn beta_value(f: Function, values: &[Complex], prec: usize) -> Option<Complex> {
    if f != Function::Beta {
        return None;
    }
    let (x, y) = (&values[0], &values[values.len() - 1]);
    crate::functions::combinatorial::factorials::gamma_product(&[x.clone(), y.clone()], &[x + y], prec)
}
