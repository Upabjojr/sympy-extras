//! Port of the evaluation of `sympy_extras/core/add.py`: `add` (SymPy's
//! `Add.flatten`: the nested sums opened, the numbers added, the terms with
//! the same non-numerical part collected, the infinities absorbing the real
//! or finite terms, the terms sorted by the `compare` order) and `sub`.
//!
//! The order terms and the accumulation bounds of the Python function are
//! not in the Rust core yet; a sum with one of them cannot be built here.

use crate::error::PolyResult;

use super::assumptions::Predicate;
use super::basic::{Basic, Head};
use super::expr::{as_coeff_mul, is, is_integer_class, is_rational_value, is_true, is_zero_number};
use super::mul::mul;
use super::numbers::number_add;
use super::parameters::evaluate;

/// `Add(*terms)` without evaluation, one term as itself, none as `0`.
pub fn raw_add(terms: Vec<Basic>) -> Basic {
    match terms.len() {
        0 => Basic::zero(),
        1 => terms.into_iter().next().expect("one term"),
        _ => Basic::new(Head::Add, terms),
    }
}

/// Whether the term `c*rest` is absorbed by the infinite `number`: `oo`
/// absorbs the real terms, `zoo` the finite terms which are known real or
/// known non-real.
fn absorbed_in_sum(number: &Basic, rest: &Basic, c: &Basic) -> bool {
    match number.head() {
        Head::Infinity | Head::NegativeInfinity => c.is_finite_number() && is_true(rest, Predicate::real),
        Head::ComplexInfinity => {
            c.is_finite_number() && is_true(rest, Predicate::finite) && is(rest, Predicate::extended_real).is_some()
        }
        _ => false,
    }
}

/// The evaluated sum of the arguments.
pub fn add(args: &[Basic]) -> PolyResult<Basic> {
    let mut stack: Vec<Basic> = args.to_vec();
    if !evaluate() {
        let mut flat: Vec<Basic> = Vec::new();
        for a in &stack {
            if a.is_add() {
                flat.extend(a.args().iter().cloned());
            } else {
                flat.push(a.clone());
            }
        }
        return Ok(raw_add(flat));
    }
    // SymPy's AssocOp.__new__: the identities dropped, a single argument
    // returned as it is
    let nonidentity: Vec<&Basic> = stack
        .iter()
        .filter(|t| !is_zero_number(t) || !is_integer_class(t))
        .collect();
    if nonidentity.len() == 1 {
        return Ok(nonidentity[0].clone());
    }
    let mut number: Basic = Basic::zero();
    // the non-numerical parts with their coefficients, in order of first
    // occurrence (a Python dict)
    let mut rests: Vec<Basic> = Vec::new();
    let mut coefficients: Vec<Basic> = Vec::new();
    let mut i: usize = 0;
    while i < stack.len() {
        let a: Basic = stack[i].clone();
        i += 1;
        if a.is_add() {
            stack.extend(a.args().iter().cloned());
            continue;
        }
        if a.is_number_node() {
            number = number_add(&number, &a);
            continue;
        }
        let (c, rest) = as_coeff_mul(&a);
        match rests.iter().position(|r| *r == rest) {
            Some(k) => coefficients[k] = number_add(&coefficients[k], &c),
            None => {
                rests.push(rest);
                coefficients.push(c);
            }
        }
    }
    if matches!(number.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    let mut terms: Vec<Basic> = Vec::new();
    let mut needs_pass: bool = false;
    for (rest, c) in rests.iter().zip(&coefficients) {
        if matches!(c.head(), Head::NaN) {
            return Ok(Basic::nan());
        }
        if c.as_rational().is_some_and(|v| v.is_zero()) {
            continue;
        }
        if absorbed_in_sum(&number, rest, c) {
            continue;
        }
        // SymPy's Add.flatten: the terms re-created
        if is_rational_value(c, 1) {
            needs_pass = needs_pass || rest.is_add() || rest.is_number_node();
            terms.push(rest.clone());
            continue;
        }
        let term: Basic = if rest.is_mul() {
            // c in slot 0 of the product's arguments: a nested product is
            // flattened one level
            let mut factors: Vec<Basic> = vec![c.clone()];
            factors.extend(rest.args().iter().cloned());
            Basic::new(Head::Mul, factors)
        } else if rest.is_add() {
            Basic::new(Head::Mul, vec![c.clone(), rest.clone()])
        } else {
            mul(&[c.clone(), rest.clone()])?
        };
        if term.is_add() || term.is_number_node() {
            needs_pass = true;
        }
        terms.push(term);
    }
    if needs_pass {
        let mut again: Vec<Basic> = vec![number];
        again.extend(terms);
        return add(&again);
    }
    terms.sort_by(|a, b| a.compare_key().cmp(b.compare_key()));
    let nonzero: bool = !number.as_rational().is_some_and(|v| v.is_zero());
    if nonzero {
        terms.insert(0, number.clone());
    }
    Ok(match terms.len() {
        0 => number,
        1 => terms.into_iter().next().expect("one term"),
        _ => Basic::new(Head::Add, terms),
    })
}

/// The evaluated difference `a - b`.
pub fn sub(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), mul(&[Basic::negative_one(), b.clone()])?])
}
