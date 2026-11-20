//! Port of `sympy_extras/integrals/integral.py`: the unevaluated integral
//! of the core, `Integral(f, x)` and `Integral(f, (x, a, b))` (the head
//! `Head::Integral`, arguments `(f, x)` or `(f, x, a, b)`): its free
//! symbols, substitution, derivative (Leibniz's rule), facts and printing.
//! Its `doit` needs the integrators of `indefinite`/`definite` and its
//! value the quadrature of `integrals/_evalf` (not ported).

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Head};
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;

/// `Integral(function, variable)` or, with the bounds, the definite one
/// (the constructor checks the arguments; the node never evaluates).
pub fn integral(function: &Basic, variable: &Basic, bounds: Option<(&Basic, &Basic)>) -> R {
    let mut args: Vec<Basic> = vec![function.clone(), variable.clone()];
    if let Some((a, b)) = bounds {
        args.push(a.clone());
        args.push(b.clone());
    }
    checked(&args)?;
    Ok(Basic::new(Head::Integral, args))
}

fn checked(args: &[Basic]) -> PolyResult<()> {
    let is_expr = |e: &Basic| !crate::core::logic::is_boolean(e) && !matches!(e.head(), Head::Tuple);
    let ok: bool = (args.len() == 2 || args.len() == 4) && args[1].is_symbol() && args.iter().all(is_expr);
    if ok {
        Ok(())
    } else {
        Err(PolyError::Type(format!(
            "invalid arguments of an Integral: ({})",
            args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ")
        )))
    }
}

/// `NonElementaryIntegral(function, variable)` (`integrals/risch/risch.py`).
pub fn non_elementary(function: &Basic, variable: &Basic) -> R {
    let args: Vec<Basic> = vec![function.clone(), variable.clone()];
    checked(&args)?;
    Ok(Basic::new(Head::NonElementaryIntegral, args))
}

/// The node rebuilt from new arguments (`_evaluated`: no evaluation).
pub fn evaluated(args: &[Basic]) -> R {
    checked(args)?;
    Ok(Basic::new(Head::Integral, args.to_vec()))
}

/// `(function, variable, bounds)`.
pub fn parts(e: &Basic) -> (&Basic, &Basic, Option<(&Basic, &Basic)>) {
    let a = e.args();
    (&a[0], &a[1], if a.len() == 4 { Some((&a[2], &a[3])) } else { None })
}

/// The free symbols: an indefinite integral depends on its variable.
pub fn free_symbols(e: &Basic) -> BTreeSet<Basic> {
    let (f, v, bounds) = parts(e);
    let mut out: BTreeSet<Basic> = f.free_symbols();
    match bounds {
        None => {
            out.insert(v.clone());
        }
        Some((a, b)) => {
            out.remove(v);
            out.extend(a.free_symbols());
            out.extend(b.free_symbols());
        }
    }
    out
}

/// The variable of a definite integral is bound.
pub fn bound_symbols(e: &Basic) -> Vec<Basic> {
    let (_, v, bounds) = parts(e);
    if bounds.is_some() {
        vec![v.clone()]
    } else {
        Vec::new()
    }
}

/// `Integral.is_number`: a definite integral without free symbols.
pub fn is_number(e: &Basic) -> bool {
    parts(e).2.is_some() && free_symbols(e).is_empty()
}

/// The facts (`_eval_facts`): the commutativity of the integrand.
pub fn facts(e: &Basic) -> FactList {
    let mut found: FactList = FactList::new();
    if let Some(c) = facts_of(&e.args()[0]).get(&Predicate::commutative) {
        found.set(Predicate::commutative, *c);
    }
    found
}

/// The substitution (`_subs`): the variable is the one of the integration
/// and is never replaced inside the integrand.
pub fn subs_partial(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Option<Basic>> {
    let (f, v, bounds) = parts(e);
    let subs = |x: &Basic| crate::core::subs::subs(x, old, new);
    let nf: Basic = if old == v { f.clone() } else { subs(f)? };
    match bounds {
        None => {
            if old == v || nf == *f {
                return Ok(Some(e.clone()));
            }
            if matches!(e.head(), Head::NonElementaryIntegral) {
                return Ok(Some(non_elementary(&nf, v)?));
            }
            Ok(Some(integral(&nf, v, None)?))
        }
        Some((a, b)) => {
            let (na, nb) = (subs(a)?, subs(b)?);
            if nf == *f && na == *a && nb == *b {
                return Ok(Some(e.clone()));
            }
            Ok(Some(integral(&nf, v, Some((&na, &nb)))?))
        }
    }
}

/// The derivative (`_eval_derivative_node`): the integrand for the
/// variable of an indefinite integral, under the integral sign otherwise,
/// Leibniz's rule for a definite one.
pub fn derivative(e: &Basic, s: &Basic) -> R {
    use crate::core::function::derivative as d;
    let (f, v, bounds) = parts(e);
    match bounds {
        None => {
            if s == v {
                return Ok(f.clone());
            }
            if !f.free_symbols().contains(s) {
                return Ok(Basic::zero());
            }
            integral(&d(f, s)?, v, None)
        }
        Some((lower, upper)) => {
            if !free_symbols(e).contains(s) {
                return Ok(Basic::zero());
            }
            let mut result: Basic = Basic::zero();
            if f.free_symbols().contains(s) && s != v {
                result = integral(&d(f, s)?, v, Some((lower, upper)))?;
            }
            if upper.free_symbols().contains(s) {
                let term: Basic = crate::core::mul::mul(&[crate::core::subs::subs(f, v, upper)?, d(upper, s)?])?;
                result = crate::core::add::add(&[result, term])?;
            }
            if lower.free_symbols().contains(s) {
                let term: Basic = crate::core::mul::mul(&[crate::core::subs::subs(f, v, lower)?, d(lower, s)?])?;
                result = crate::core::add::sub(&result, &term)?;
            }
            Ok(result)
        }
    }
}

/// `Integral(f, x)` or `Integral(f, (x, a, b))` printed.
pub fn print(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let (f, v, bounds) = parts(e);
    match bounds {
        None => format!("Integral({}, {})", doprint(f), doprint(v)),
        Some((a, b)) => format!(
            "Integral({}, ({}, {}, {}))",
            doprint(f),
            doprint(v),
            doprint(a),
            doprint(b)
        ),
    }
}

/// `doit`: the integrators are not ported.
pub fn doit(_e: &Basic) -> R {
    Err(crate::error::not_ported(
        "Integral.doit (integrals/indefinite, integrals/definite)",
    ))
}
