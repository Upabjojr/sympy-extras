//! The orthogonal polynomials, part 1, ported from
//! `sympy_extras.functions.special.polynomials`: `legendre`,
//! `chebyshevt`, `chebyshevu`, `hermite`, `laguerre` and `assoc_laguerre`.
//! A numerical order gives the polynomial (from the three-term
//! recurrences, written as SymPy's `Poly.as_expr` writes it), negative
//! orders are reflected; a symbolic order keeps the node but takes out
//! `-x` and evaluates at `0`, `1` and `oo`; the derivatives in `x` and the
//! conjugates.
//!
//!
//! Part 2: `jacobi`, `gegenbauer`, `assoc_legendre`, `hermite_prob` with
//! SymPy's reductions (`jacobi(n, a, a, x)` through `gegenbauer`,
//! `chebyshevt`, `legendre`, `chebyshevu`; `jacobi(n, a, -a, x)` through
//! `assoc_legendre`), `chebyshevt_root`, `chebyshevu_root` and
//! `jacobi_normalized`.
//!
//! Not ported: the values of a node (a non-integer order: mpmath's
//! hypergeometric continuations) and `jacobi` of a symbolic order at `x =
//! 0` (a `hyper`, not yet in the Rust core: `NotImplemented`).

// PORTED FROM SYMPY: sympy/functions/special/polynomials.py (legendre, chebyshevt,
// chebyshevu, hermite, laguerre, assoc_laguerre, jacobi, gegenbauer, assoc_legendre,
// hermite_prob, chebyshevt_root, chebyshevu_root, jacobi_normalized) and the recurrences of
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

/// The orthogonal polynomials (both parts).
pub fn is_orthogonal(f: Function) -> bool {
    matches!(
        f,
        Function::Legendre
            | Function::ChebyshevT
            | Function::ChebyshevU
            | Function::Hermite
            | Function::Laguerre
            | Function::AssocLaguerre
            | Function::Jacobi
            | Function::Gegenbauer
            | Function::AssocLegendre
            | Function::HermiteProb
    )
}

/// Whether the arguments fit the head.
pub fn arity_fits(f: Function, n: usize) -> bool {
    match f {
        Function::Jacobi => n == 4,
        Function::AssocLaguerre | Function::Gegenbauer | Function::AssocLegendre => n == 3,
        _ => n == 2,
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
    match f {
        Function::Jacobi => return jacobi_eval(args),
        Function::Gegenbauer => return gegenbauer_eval(args),
        Function::AssocLegendre => return assoc_legendre_eval(args),
        Function::HermiteProb => return hermite_prob_eval(args),
        _ => {}
    }
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
            Function::Jacobi => {
                let (a, b): (&Basic, &Basic) = (&args[1], &args[2]);
                let inner: Basic = jacobi(
                    &sub(n, &one)?,
                    &add(&[a.clone(), one.clone()])?,
                    &add(&[b.clone(), one.clone()])?,
                    x,
                )?;
                lmul(&[Basic::half(), add(&[a.clone(), b.clone(), n.clone(), one])?, inner])
            }
            Function::Gegenbauer => {
                let a: &Basic = &args[1];
                let inner: Basic = gegenbauer(&sub(n, &one)?, &add(&[a.clone(), one])?, x)?;
                lmul(&[int(2), a.clone(), inner])
            }
            Function::AssocLegendre => {
                let m: &Basic = &args[1];
                let first: Basic = lmul(&[x.clone(), n.clone(), assoc_legendre(n, m, x)?])?;
                let second: Basic = mul(&[add(&[m.clone(), n.clone()])?, assoc_legendre(&sub(n, &one)?, m, x)?])?;
                lmul(&[div(&one, &x2m1)?, sub(&first, &second)?])
            }
            Function::HermiteProb => mul(&[n.clone(), hermite_prob(&sub(n, &one)?, x)?]),
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

/// The conjugate (`_eval_conjugate`): `f(n, conjugate(x))`, the
/// parameters after `n` conjugated too (`assoc_laguerre(n,
/// conjugate(alpha), conjugate(x))`, `jacobi(n, conjugate(a),
/// conjugate(b), conjugate(x))`, ...).
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

// ---------------------------------------------------------------------
// Part 2: jacobi, gegenbauer, assoc_legendre, hermite_prob
// ---------------------------------------------------------------------

pub fn jacobi(n: &Basic, a: &Basic, b: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::Jacobi, &[n.clone(), a.clone(), b.clone(), x.clone()])
}

pub fn gegenbauer(n: &Basic, a: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::Gegenbauer, &[n.clone(), a.clone(), x.clone()])
}

pub fn assoc_legendre(n: &Basic, m: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::AssocLegendre, &[n.clone(), m.clone(), x.clone()])
}

pub fn hermite_prob(n: &Basic, x: &Basic) -> PolyResult<Basic> {
    orthogonal(Function::HermiteProb, &[n.clone(), x.clone()])
}

/// The product evaluated left to right, as Python evaluates SymPy's
/// `a*b/c*d` (`_lmul`).
fn lmul(factors: &[Basic]) -> PolyResult<Basic> {
    let mut result: Basic = factors[0].clone();
    for f in &factors[1..] {
        result = mul(&[result, f.clone()])?;
    }
    Ok(result)
}

fn hermite_prob_coefficients(n: i64) -> PolyResult<Coefficients> {
    // He_{k+1} = x*He_k - k*He_{k-1}
    recurrence(n, vec![Basic::one()], vec![Basic::zero(), Basic::one()], &|k, p, q| {
        combine(&[(Basic::one(), shift(p)), (int(-k), q.clone())])
    })
}

fn gegenbauer_coefficients(n: i64, a: &Basic) -> PolyResult<Coefficients> {
    // k*C_k = 2*(k + a - 1)*x*C_{k-1} - (k + 2*a - 2)*C_{k-2}
    let second: Coefficients = vec![Basic::zero(), expand(&mul(&[int(2), a.clone()])?)?];
    recurrence(n, vec![Basic::one()], second, &|k, p, q| {
        let j: i64 = k + 1;
        combine(&[
            (mul(&[rational(2, j)?, add(&[int(j - 1), a.clone()])?])?, shift(p)),
            (
                mul(&[rational(-1, j)?, add(&[int(j - 2), mul(&[int(2), a.clone()])?])?])?,
                q.clone(),
            ),
        ])
    })
}

/// `binomial(top, k)` as an expanded polynomial in `top`.
fn binomial_polynomial(top: &Basic, k: i64) -> PolyResult<Basic> {
    let mut product: Basic = Basic::one();
    let mut factorial: Int = Int::ONE;
    for i in 0..k {
        product = mul(&[product, sub(top, &int(i))?])?;
        factorial *= Int::from(i + 1);
    }
    expand(&mul(&[Basic::rational(Int::ONE, factorial)?, product])?)
}

/// The coefficients of `((x + shift)/2)**power`.
fn power_coefficients(shift: i64, power: i64) -> PolyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    let mut choose: Int = Int::ONE;
    let two_power: Int = Int::from(2).pow(power as usize);
    for k in 0..=power {
        if k > 0 {
            choose = choose * Int::from(power - k + 1) / Int::from(k);
        }
        let sign: Int = Int::from(shift).pow((power - k) as usize);
        result.push(Basic::rational(choose.clone() * sign, two_power.clone())?);
    }
    Ok(result)
}

/// `P_n^(a, b) = sum(binomial(n + a, n - s)*binomial(n + b, s)
/// *((x - 1)/2)**s*((x + 1)/2)**(n - s))`, the coefficients expanded.
fn jacobi_coefficients(n: i64, a: &Basic, b: &Basic) -> PolyResult<Coefficients> {
    let mut coefficients: Vec<Vec<Basic>> = vec![Vec::new(); (n + 1) as usize];
    for s in 0..=n {
        let weight: Basic = mul(&[
            binomial_polynomial(&add(&[int(n), a.clone()])?, n - s)?,
            binomial_polynomial(&add(&[int(n), b.clone()])?, s)?,
        ])?;
        let minus: Vec<Basic> = power_coefficients(-1, s)?;
        let plus: Vec<Basic> = power_coefficients(1, n - s)?;
        for (i, c) in minus.iter().enumerate() {
            for (j, d) in plus.iter().enumerate() {
                coefficients[i + j].push(mul(&[c.clone(), d.clone(), weight.clone()])?);
            }
        }
    }
    coefficients.iter().map(|terms| expand(&add(terms)?)).collect()
}

fn derivative_coefficients(coefficients: &Coefficients, order: i64) -> PolyResult<Coefficients> {
    let mut result: Coefficients = coefficients.clone();
    for _ in 0..order {
        let mut next: Coefficients = Vec::new();
        for (k, c) in result.iter().enumerate().skip(1) {
            next.push(mul(&[int(k as i64), c.clone()])?);
        }
        if next.is_empty() {
            next.push(Basic::zero());
        }
        result = next;
    }
    Ok(result)
}

fn jacobi_eval(args: &[Basic]) -> PolyResult<Option<Basic>> {
    use crate::functions::combinatorial::factorials::{factorial, rf};
    use crate::functions::special::gamma_functions::gamma;
    let one: Basic = Basic::one();
    let half: Basic = Basic::half();
    let (n, a, b, x) = (&args[0], &args[1], &args[2], &args[3]);
    if a == b {
        if *a == rational(-1, 2)? {
            return Ok(Some(lmul(&[div(&rf(&half, n)?, &factorial(n)?)?, chebyshevt(n, x)?])?));
        }
        if is_true(a, Predicate::zero) {
            return Ok(Some(legendre(n, x)?));
        }
        if *a == half {
            let top: Basic = rf(&rational(3, 2)?, n)?;
            let bottom: Basic = factorial(&add(&[n.clone(), one])?)?;
            return Ok(Some(lmul(&[div(&top, &bottom)?, chebyshevu(n, x)?])?));
        }
        let top: Basic = rf(&add(&[a.clone(), one.clone()])?, n)?;
        let bottom: Basic = rf(&add(&[mul(&[int(2), a.clone()])?, one])?, n)?;
        let inner: Basic = gegenbauer(n, &add(&[a.clone(), half])?, x)?;
        return Ok(Some(lmul(&[div(&top, &bottom)?, inner])?));
    }
    if *b == neg(a)? {
        let half_a: Basic = div(a, &int(2))?;
        let ratio: Basic = div(
            &gamma(&add(&[n.clone(), a.clone(), one.clone()])?)?,
            &gamma(&add(&[n.clone(), one.clone()])?)?,
        )?;
        return Ok(Some(lmul(&[
            ratio,
            pow(&add(&[one.clone(), x.clone()])?, &half_a)?,
            pow(&pow(&sub(&one, x)?, &half_a)?, &Basic::negative_one())?,
            assoc_legendre(n, &neg(a)?, x)?,
        ])?));
    }
    if !n.is_number_class() {
        if could_extract_minus_sign(x) {
            let inner: Basic = jacobi(n, b, a, &neg(x)?)?;
            return Ok(Some(mul(&[pow(&Basic::negative_one(), n)?, inner])?));
        }
        if is_true(x, Predicate::zero) {
            return Err(not_ported("jacobi at 0 of a symbolic order (a hyper)"));
        }
        if *x == one {
            return Ok(Some(div(&rf(&add(&[a.clone(), one])?, n)?, &factorial(n)?)?));
        }
        if matches!(x.head(), Head::Infinity) && is_true(n, Predicate::positive) {
            let total: Basic = add(&[a.clone(), b.clone(), mul(&[int(2), n.clone()])?])?;
            if is_true(&total, Predicate::integer) {
                return Err(value_error("Error. a + b + 2*n should not be an integer.".to_string()));
            }
            let top: Basic = rf(&add(&[a.clone(), b.clone(), n.clone(), one])?, n)?;
            return Ok(Some(mul(&[top, Basic::oo()])?));
        }
        return Ok(None);
    }
    let Some(order) = integer(n)? else {
        return Ok(None);
    };
    if order < 0 {
        return Err(value_error(format!(
            "Cannot generate Jacobi polynomial of degree {}",
            n
        )));
    }
    Ok(Some(as_expr(&jacobi_coefficients(order, a, b)?, x)?))
}

fn gegenbauer_eval(args: &[Basic]) -> PolyResult<Option<Basic>> {
    use crate::core::logic::true_;
    use crate::core::relational::gt;
    use crate::functions::combinatorial::factorials::rf;
    use crate::functions::elementary::complexes::re;
    use crate::functions::elementary::trigonometric::{cos, sec};
    use crate::functions::special::gamma_functions::gamma;
    let one: Basic = Basic::one();
    let half: Basic = Basic::half();
    let pi: Basic = Basic::pi();
    let (n, a, x) = (&args[0], &args[1], &args[2]);
    if is_true(n, Predicate::negative) {
        return Ok(Some(Basic::zero()));
    }
    if *a == half {
        return Ok(Some(legendre(n, x)?));
    }
    if *a == one {
        return Ok(Some(chebyshevu(n, x)?));
    }
    if *a == Basic::negative_one() {
        return Ok(Some(Basic::zero()));
    }
    if !n.is_number_class() {
        let two_a: Basic = mul(&[int(2), a.clone()])?;
        let n1: Basic = add(&[n.clone(), one.clone()])?;
        if *x == Basic::negative_one() {
            if gt(&re(a)?, &half)? == true_() {
                return Ok(Some(Basic::zoo()));
            }
            let top: Basic = lmul(&[
                cos(&mul(&[pi.clone(), add(&[a.clone(), n.clone()])?])?)?,
                sec(&mul(&[pi, a.clone()])?)?,
                gamma(&add(&[two_a.clone(), n.clone()])?)?,
            ])?;
            let bottom: Basic = lmul(&[gamma(&two_a)?, gamma(&n1)?])?;
            return Ok(Some(div(&top, &bottom)?));
        }
        if could_extract_minus_sign(x) {
            let inner: Basic = gegenbauer(n, a, &neg(x)?)?;
            return Ok(Some(mul(&[pow(&Basic::negative_one(), n)?, inner])?));
        }
        if is_true(x, Predicate::zero) {
            let top: Basic = lmul(&[
                pow(&int(2), n)?,
                pow(&pi, &half)?,
                gamma(&add(&[a.clone(), mul(&[half.clone(), n.clone()])?])?)?,
            ])?;
            let bottom: Basic = lmul(&[gamma(&div(&sub(&one, n)?, &int(2))?)?, gamma(&n1)?, gamma(a)?])?;
            return Ok(Some(div(&top, &bottom)?));
        }
        if *x == one {
            let bottom: Basic = mul(&[gamma(&two_a)?, gamma(&n1)?])?;
            return Ok(Some(div(&gamma(&add(&[two_a, n.clone()])?)?, &bottom)?));
        }
        if matches!(x.head(), Head::Infinity) && is_true(n, Predicate::positive) {
            return Ok(Some(mul(&[rf(a, n)?, Basic::oo()])?));
        }
        return Ok(None);
    }
    let Some(order) = integer(n)? else {
        return Ok(None);
    };
    Ok(Some(as_expr(&gegenbauer_coefficients(order, a)?, x)?))
}

fn assoc_legendre_eval(args: &[Basic]) -> PolyResult<Option<Basic>> {
    use crate::core::symbol::Dummy;
    use crate::functions::combinatorial::factorials::factorial;
    use crate::functions::special::gamma_functions::gamma;
    let one: Basic = Basic::one();
    let (n, m, x) = (&args[0], &args[1], &args[2]);
    if could_extract_minus_sign(m) {
        let minus_m: Basic = neg(m)?;
        let ratio: Basic = div(&factorial(&add(&[m.clone(), n.clone()])?)?, &factorial(&sub(n, m)?)?)?;
        return Ok(Some(lmul(&[
            pow(&Basic::negative_one(), &minus_m)?,
            ratio,
            assoc_legendre(n, &minus_m, x)?,
        ])?));
    }
    if *m == Basic::zero() {
        return Ok(Some(legendre(n, x)?));
    }
    if *x == Basic::zero() {
        let top: Basic = mul(&[pow(&int(2), m)?, pow(&Basic::pi(), &Basic::half())?])?;
        let g1: Basic = gamma(&div(&sub(&sub(&one, m)?, n)?, &int(2))?)?;
        let g2: Basic = gamma(&sub(&one, &div(&sub(m, n)?, &int(2))?)?)?;
        return Ok(Some(div(&top, &mul(&[g1, g2])?)?));
    }
    let (Some(nn), Some(mm)) = (integer(n)?, integer(m)?) else {
        return Ok(None);
    };
    if nn < 0 {
        return Err(value_error(format!(
            "assoc_legendre : 1st index must be nonnegative integer (got {})",
            n
        )));
    }
    if mm.abs() > nn {
        return Err(value_error(format!(
            "assoc_legendre : abs('2nd index') must be <= '1st index' (got {}, {})",
            n, m
        )));
    }
    // SymPy builds the function in a dummy and substitutes x
    let order: i64 = mm.abs();
    let dummy: Basic = Dummy::new("x", &[]).0;
    let derivative: Basic = as_expr(&derivative_coefficients(&legendre_coefficients(nn)?, order)?, &dummy)?;
    let built: Basic = lmul(&[
        pow(&Basic::negative_one(), &int(order))?,
        pow(&sub(&one, &pow(&dummy, &int(2))?)?, &rational(order, 2)?)?,
        derivative,
    ])?;
    Ok(Some(crate::core::subs::subs(&built, &dummy, x)?))
}

fn hermite_prob_eval(args: &[Basic]) -> PolyResult<Option<Basic>> {
    use crate::functions::special::gamma_functions::gamma;
    let (n, x) = (&args[0], &args[1]);
    let Some(order) = integer(n)? else {
        if n.is_number_class() {
            return Ok(None);
        }
        if let Some(r) = minus_x(Function::HermiteProb, n, x)? {
            return Ok(Some(r));
        }
        if is_true(x, Predicate::zero) {
            let bottom: Basic = gamma(&div(&sub(&Basic::one(), n)?, &int(2))?)?;
            return Ok(Some(div(&pow(&Basic::pi(), &Basic::half())?, &bottom)?));
        }
        if matches!(x.head(), Head::Infinity) {
            return Ok(Some(Basic::oo()));
        }
        return Ok(None);
    };
    if order < 0 {
        return Ok(None); // SymPy builds a ValueError without raising it: the node stays
    }
    Ok(Some(as_expr(&hermite_prob_coefficients(order)?, x)?))
}

fn check_root_index(n: &Basic, k: &Basic) -> PolyResult<()> {
    use crate::core::logic::true_;
    use crate::core::relational::{le, lt};
    if !(le(&Basic::zero(), k)? == true_() && lt(k, n)? == true_()) {
        return Err(value_error(format!(
            "must have 0 <= k < n, got k = {} and n = {}",
            k, n
        )));
    }
    Ok(())
}

/// The `k`-th root of `chebyshevt(n, x)`, `cos(pi*(2*k + 1)/(2*n))`.
pub fn chebyshevt_root(n: &Basic, k: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::trigonometric::cos;
    check_root_index(n, k)?;
    let top: Basic = mul(&[Basic::pi(), add(&[mul(&[int(2), k.clone()])?, Basic::one()])?])?;
    cos(&div(&top, &mul(&[int(2), n.clone()])?)?)
}

/// The `k`-th root of `chebyshevu(n, x)`, `cos(pi*(k + 1)/(n + 1))`.
pub fn chebyshevu_root(n: &Basic, k: &Basic) -> PolyResult<Basic> {
    use crate::functions::elementary::trigonometric::cos;
    check_root_index(n, k)?;
    let top: Basic = mul(&[Basic::pi(), add(&[k.clone(), Basic::one()])?])?;
    cos(&div(&top, &add(&[n.clone(), Basic::one()])?)?)
}

/// The Jacobi polynomial divided by the square root of its norm.
pub fn jacobi_normalized(n: &Basic, a: &Basic, b: &Basic, x: &Basic) -> PolyResult<Basic> {
    use crate::functions::combinatorial::factorials::factorial;
    use crate::functions::special::gamma_functions::gamma;
    let one: Basic = Basic::one();
    let power: Basic = pow(&int(2), &add(&[a.clone(), b.clone(), one.clone()])?)?;
    let gammas: Basic = mul(&[
        gamma(&add(&[n.clone(), a.clone(), one.clone()])?)?,
        gamma(&add(&[n.clone(), b.clone(), one.clone()])?)?,
    ])?;
    let inner: Basic = div(
        &mul(&[power, gammas])?,
        &add(&[mul(&[int(2), n.clone()])?, a.clone(), b.clone(), one.clone()])?,
    )?;
    let factor: Basic = div(
        &inner,
        &mul(&[factorial(n)?, gamma(&add(&[n.clone(), a.clone(), b.clone(), one])?)?])?,
    )?;
    div(&jacobi(n, a, b, x)?, &pow(&factor, &Basic::half())?)
}
