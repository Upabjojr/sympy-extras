//! The orthogonal polynomials, part 1, ported from
//! `sympy_extras.functions.special.polynomials`: `legendre`,
//! `chebyshevt`, `chebyshevu`, `hermite`, `laguerre` and `assoc_laguerre`.
//! A numerical order gives the polynomial (from the three-term
//! recurrences, written as SymPy's `Poly.as_expr` writes it), negative
//! orders are reflected; a symbolic order keeps the node but takes out
//! `-x` and evaluates at `0`, `1` and `oo`; the derivatives in `x` and the
//! conjugates.
//!
//! Not ported: the values of a node (a non-integer order: mpmath's
//! hypergeometric continuations), `jacobi`, `gegenbauer`, `assoc_legendre`,
//! `hermite_prob` and the Chebyshev roots (part 2).

// PORTED FROM SYMPY: sympy/functions/special/polynomials.py (legendre, chebyshevt,
// chebyshevu, hermite, laguerre, assoc_laguerre) and the recurrences of
// sympy/polys/orthopolys.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{is, is_true};
use crate::core::function::{expand_with, Hints};
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::{not_ported, value_error, PolyResult};
use crate::numbers::{denom, numer, Int};

type Coefficients = Vec<Basic>;

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn rational(p: i64, q: i64) -> PolyResult<Basic> {
    Basic::rational(Int::from(p), Int::from(q))
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

fn expand(e: &Basic) -> PolyResult<Basic> {
    expand_with(e, true, Hints::DEFAULT)
}

/// `x*p`.
fn shift(p: &Coefficients) -> Coefficients {
    let mut q: Coefficients = vec![Basic::zero()];
    q.extend(p.iter().cloned());
    q
}

/// `sum(c*p)`, the coefficients expanded.
fn combine(terms: &[(Basic, Coefficients)]) -> PolyResult<Coefficients> {
    let size: usize = terms.iter().map(|(_, p)| p.len()).max().unwrap_or(0);
    let mut result: Coefficients = Vec::new();
    for k in 0..size {
        let mut parts: Vec<Basic> = Vec::new();
        for (c, p) in terms {
            if k < p.len() {
                parts.push(mul(&[c.clone(), p[k].clone()])?);
            }
        }
        result.push(expand(&add(&parts)?)?);
    }
    Ok(result)
}

/// `p_n` from `p_0`, `p_1` and `p_{k+1} = step(k, p_k, p_{k-1})`.
fn recurrence(
    n: i64,
    first: Coefficients,
    second: Coefficients,
    step: &dyn Fn(i64, &Coefficients, &Coefficients) -> PolyResult<Coefficients>,
) -> PolyResult<Coefficients> {
    if n == 0 {
        return Ok(first);
    }
    let (mut previous, mut current) = (first, second);
    for k in 1..n {
        let next: Coefficients = step(k, &current, &previous)?;
        previous = std::mem::replace(&mut current, next);
    }
    Ok(current)
}

/// SymPy's `Poly.as_expr`: `sum(c_k*x**k)`.
fn as_expr(coefficients: &Coefficients, x: &Basic) -> PolyResult<Basic> {
    let mut terms: Vec<Basic> = Vec::new();
    for (k, c) in coefficients.iter().enumerate() {
        if !(c.as_rational().is_some_and(|v| v.is_zero()) && c.is_number_node()) {
            terms.push(mul(&[c.clone(), pow(x, &int(k as i64))?])?);
        }
    }
    add(&terms)
}

fn legendre_coefficients(n: i64) -> PolyResult<Coefficients> {
    // (k + 1)*P_{k+1} = (2k + 1)*x*P_k - k*P_{k-1}
    recurrence(n, vec![Basic::one()], vec![Basic::zero(), Basic::one()], &|k, p, q| {
        combine(&[
            (rational(2 * k + 1, k + 1)?, shift(p)),
            (rational(-k, k + 1)?, q.clone()),
        ])
    })
}

fn chebyshev_coefficients(n: i64, second: Coefficients) -> PolyResult<Coefficients> {
    // p_{k+1} = 2*x*p_k - p_{k-1}
    recurrence(n, vec![Basic::one()], second, &|_, p, q| {
        combine(&[(int(2), shift(p)), (Basic::negative_one(), q.clone())])
    })
}

fn hermite_coefficients(n: i64) -> PolyResult<Coefficients> {
    // H_{k+1} = 2*x*H_k - 2*k*H_{k-1}
    recurrence(n, vec![Basic::one()], vec![Basic::zero(), int(2)], &|k, p, q| {
        combine(&[(int(2), shift(p)), (int(-2 * k), q.clone())])
    })
}

fn laguerre_coefficients(n: i64, alpha: &Basic) -> PolyResult<Coefficients> {
    // (k + 1)*L_{k+1} = (2k + 1 + alpha - x)*L_k - (k + alpha)*L_{k-1}
    let second: Coefficients = vec![expand(&add(&[Basic::one(), alpha.clone()])?)?, Basic::negative_one()];
    recurrence(n, vec![Basic::one()], second, &|k, p, q| {
        let inverse: Basic = rational(1, k + 1)?;
        combine(&[
            (
                mul(&[inverse.clone(), add(&[int(2 * k + 1), alpha.clone()])?])?,
                p.clone(),
            ),
            (mul(&[Basic::negative_one(), inverse.clone()])?, shift(p)),
            (
                mul(&[Basic::negative_one(), inverse, add(&[int(k), alpha.clone()])?])?,
                q.clone(),
            ),
        ])
    })
}

/// The order as an integer (`_integer`), refused when huge.
fn integer(n: &Basic) -> PolyResult<Option<i64>> {
    match n.as_rational() {
        Some(v) if n.is_number_node() && denom(v) == Int::ONE => {
            let order: i64 = i64::try_from(&numer(v))
                .ok()
                .filter(|o| o.abs() <= 5000)
                .ok_or_else(|| not_ported("a huge order"))?;
            Ok(Some(order))
        }
        _ => Ok(None),
    }
}

/// The orthogonal polynomials of part 1.
pub fn is_orthogonal(f: Function) -> bool {
    matches!(
        f,
        Function::Legendre
            | Function::ChebyshevT
            | Function::ChebyshevU
            | Function::Hermite
            | Function::Laguerre
            | Function::AssocLaguerre
    )
}

/// Whether the arguments fit the head.
pub fn arity_fits(f: Function, n: usize) -> bool {
    if f == Function::AssocLaguerre {
        n == 3
    } else {
        n == 2
    }
}

pub fn legendre(n: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::Legendre, &[n.clone(), x.clone()])
}

pub fn chebyshevt(n: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::ChebyshevT, &[n.clone(), x.clone()])
}

pub fn chebyshevu(n: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::ChebyshevU, &[n.clone(), x.clone()])
}

pub fn hermite(n: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::Hermite, &[n.clone(), x.clone()])
}

pub fn laguerre(n: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::Laguerre, &[n.clone(), x.clone()])
}

pub fn assoc_laguerre(n: &Basic, alpha: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::AssocLaguerre, &[n.clone(), alpha.clone(), x.clone()])
}

/// `(-1)**n*f(n, -x)` when `x` has a minus sign (`_minus_x`).
fn minus_x(f: Function, n: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    if could_extract_minus_sign(x) {
        let inner: Basic = orthogonal(f, &[n.clone(), neg(x)?])?;
        return Ok(Some(mul(&[pow(&Basic::negative_one(), n)?, inner])?));
    }
    Ok(None)
}

/// The evaluating function of the orthogonal polynomials of part 1.
pub fn orthogonal(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    Ok(eval(f, args)?.unwrap_or_else(|| node(f, args.to_vec())))
}

fn eval(f: Function, args: &[Basic]) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::factorials::binomial;
    use crate::functions::elementary::exponential::exp;
    use crate::functions::elementary::trigonometric::cos;
    use crate::functions::special::gamma_functions::gamma;
    let one: Basic = Basic::one();
    let half: Basic = Basic::half();
    let pi: Basic = Basic::pi();
    let n: &Basic = &args[0];
    let x: &Basic = args.last().expect("an argument");
    if f == Function::Laguerre && is(n, Predicate::integer) == Some(false) {
        return Err(value_error("Error: n should be an integer.".to_string()));
    }
    if f == Function::AssocLaguerre && is_true(&args[1], Predicate::zero) {
        return Ok(Some(laguerre(n, x)?));
    }
    let Some(order) = integer(n)? else {
        if n.is_number_class() {
            return Ok(None);
        }
        let is_zero: bool = is_true(x, Predicate::zero);
        let infinity: bool = matches!(x.head(), Head::Infinity);
        return Ok(match f {
            Function::Legendre => {
                if let Some(r) = minus_x(f, n, x)? {
                    return Ok(Some(r));
                }
                let minus_n: Basic = neg(n)?;
                if could_extract_minus_sign(n) && !could_extract_minus_sign(&sub(&minus_n, &one)?) {
                    return Ok(Some(legendre(&sub(&minus_n, &one)?, x)?));
                }
                if is_zero {
                    let g1: Basic = gamma(&sub(&half, &div(n, &int(2))?)?)?;
                    let g2: Basic = gamma(&add(&[one.clone(), div(n, &int(2))?])?)?;
                    Some(div(&pow(&pi, &half)?, &mul(&[g1, g2])?)?)
                } else if *x == one {
                    Some(one)
                } else if infinity {
                    Some(Basic::oo())
                } else {
                    None
                }
            }
            Function::ChebyshevT => {
                if let Some(r) = minus_x(f, n, x)? {
                    return Ok(Some(r));
                }
                if could_extract_minus_sign(n) {
                    return Ok(Some(chebyshevt(&neg(n)?, x)?));
                }
                if is_zero {
                    Some(cos(&mul(&[half, pi, n.clone()])?)?)
                } else if *x == one {
                    Some(one)
                } else if infinity {
                    Some(Basic::oo())
                } else {
                    None
                }
            }
            Function::ChebyshevU => {
                if let Some(r) = minus_x(f, n, x)? {
                    return Ok(Some(r));
                }
                if could_extract_minus_sign(n) {
                    if *n == Basic::negative_one() {
                        return Ok(Some(Basic::zero()));
                    }
                    let shifted: Basic = sub(&neg(n)?, &int(2))?;
                    if !could_extract_minus_sign(&shifted) {
                        return Ok(Some(neg(&chebyshevu(&shifted, x)?)?));
                    }
                }
                if is_zero {
                    Some(cos(&mul(&[half, pi, n.clone()])?)?)
                } else if *x == one {
                    Some(add(&[one, n.clone()])?)
                } else if infinity {
                    Some(Basic::oo())
                } else {
                    None
                }
            }
            Function::Hermite => {
                if let Some(r) = minus_x(f, n, x)? {
                    return Ok(Some(r));
                }
                if is_zero {
                    let top: Basic = mul(&[pow(&int(2), n)?, pow(&pi, &half)?])?;
                    Some(div(&top, &gamma(&div(&sub(&one, n)?, &int(2))?)?)?)
                } else if infinity {
                    Some(Basic::oo())
                } else {
                    None
                }
            }
            Function::Laguerre => {
                let minus_n: Basic = neg(n)?;
                if could_extract_minus_sign(n) && !could_extract_minus_sign(&sub(&minus_n, &one)?) {
                    return Ok(Some(mul(&[exp(x)?, laguerre(&sub(&minus_n, &one)?, &neg(x)?)?])?));
                }
                if is_zero {
                    Some(one)
                } else if matches!(x.head(), Head::NegativeInfinity) {
                    Some(Basic::oo())
                } else if infinity {
                    Some(mul(&[pow(&Basic::negative_one(), n)?, Basic::oo()])?)
                } else {
                    None
                }
            }
            _ => {
                let alpha: &Basic = &args[1];
                if is_zero {
                    Some(binomial(&add(&[n.clone(), alpha.clone()])?, alpha)?)
                } else if infinity && is_true(n, Predicate::positive) {
                    Some(mul(&[pow(&Basic::negative_one(), n)?, Basic::oo()])?)
                } else if matches!(x.head(), Head::NegativeInfinity) && is_true(n, Predicate::positive) {
                    Some(Basic::oo())
                } else {
                    None
                }
            }
        });
    };
    let refuse = || value_error(format!("The index n must be nonnegative integer (got {})", n));
    Ok(Some(match f {
        Function::Legendre => {
            let order: i64 = if order < 0 { -order - 1 } else { order };
            as_expr(&legendre_coefficients(order)?, x)?
        }
        Function::ChebyshevT => as_expr(&chebyshev_coefficients(order.abs(), vec![Basic::zero(), one])?, x)?,
        Function::ChebyshevU => {
            if order < 0 {
                if order == -1 {
                    return Ok(Some(Basic::zero()));
                }
                neg(&as_expr(
                    &chebyshev_coefficients(-order - 2, vec![Basic::zero(), int(2)])?,
                    x,
                )?)?
            } else {
                as_expr(&chebyshev_coefficients(order, vec![Basic::zero(), int(2)])?, x)?
            }
        }
        Function::Hermite => {
            if order < 0 {
                return Err(refuse());
            }
            as_expr(&hermite_coefficients(order)?, x)?
        }
        Function::Laguerre => {
            if order < 0 {
                mul(&[exp(x)?, laguerre(&int(-order - 1), &neg(x)?)?])?
            } else {
                as_expr(&laguerre_coefficients(order, &Basic::zero())?, x)?
            }
        }
        _ => {
            if order < 0 {
                return Err(refuse());
            }
            as_expr(&laguerre_coefficients(order, &args[1])?, x)?
        }
    }))
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn orthogonal_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let one: Basic = Basic::one();
    let n: &Basic = &args[0];
    let x: &Basic = args.last().expect("an argument");
    if argindex == args.len() {
        let x2m1: Basic = sub(&pow(x, &int(2))?, &one)?;
        return match f {
            Function::Legendre => mul(&[
                div(n, &x2m1)?,
                sub(&mul(&[x.clone(), legendre(n, x)?])?, &legendre(&sub(n, &one)?, x)?)?,
            ]),
            Function::ChebyshevT => mul(&[n.clone(), chebyshevu(&sub(n, &one)?, x)?]),
            Function::ChebyshevU => {
                let n1: Basic = add(&[n.clone(), one.clone()])?;
                let top: Basic = sub(
                    &mul(&[n1.clone(), chebyshevt(&n1, x)?])?,
                    &mul(&[x.clone(), chebyshevu(n, x)?])?,
                )?;
                div(&top, &x2m1)
            }
            Function::Hermite => mul(&[int(2), n.clone(), hermite(&sub(n, &one)?, x)?]),
            Function::Laguerre => neg(&assoc_laguerre(&sub(n, &one)?, &one, x)?),
            _ => neg(&assoc_laguerre(&sub(n, &one)?, &add(&[args[1].clone(), one])?, x)?),
        };
    }
    let v: &Basic = &args[argindex - 1];
    let elsewhere: bool = args
        .iter()
        .enumerate()
        .any(|(k, w)| k != argindex - 1 && w.free_symbols().contains(v));
    if v.is_symbol() && !elsewhere {
        return Ok(crate::core::function::derivative_node(e, &[(v.clone(), 1)]));
    }
    Err(not_ported("a derivative as a Subs"))
}

/// The conjugate (`_eval_conjugate`): `f(n, conjugate(x))`
/// (`assoc_laguerre(n, conjugate(alpha), conjugate(x))`).
pub fn orthogonal_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::conjugate;
    let args: &[Basic] = e.args();
    let mut new: Vec<Basic> = vec![args[0].clone()];
    for a in &args[1..] {
        new.push(conjugate(a)?);
    }
    Ok(Some(orthogonal(f, &new)?))
}

/// The facts (`_eval_facts`).
pub fn orthogonal_facts() -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    facts
}
