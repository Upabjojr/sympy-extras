//! Port of parts of `sympy_extras/core/function.py`: the evaluated rebuild
//! of a node (`_evaluated`) and the expansion with the hints `mul` and
//! `multinomial` ([`expand_mul`], [`expand_multinomial`]). The other hints
//! (`power_base`, `power_exp`, `log`, `func`, `trig`, `complex`), the
//! derivatives, `Lambda` and `Subs` are not ported yet.

use crate::error::PolyResult;
use crate::numbers::*;
use crate::simplify::radsimp::fraction;

use super::add::add;
use super::assumptions::Predicate;
use super::basic::{Basic, Function, Head};
use super::expr::{as_base_exp, is_false, is_true, terms_of};
use super::mul::mul;
use super::power::pow;

/// The evaluated node with `args` in place of its arguments (`_evaluated`).
pub fn rebuild_evaluated(e: &Basic, args: Vec<Basic>) -> PolyResult<Basic> {
    match e.head() {
        Head::Add => add(&args),
        Head::Mul => mul(&args),
        Head::Pow => pow(&args[0], &args[1]),
        Head::Function(f) => crate::functions::apply_function(*f, &args[0]),
        _ => Ok(e.rebuild(args)),
    }
}

/// Whether `fraction` puts (part of) the factor in the denominator.
fn denominator_factor(f: &Basic) -> PolyResult<bool> {
    if let Some(v) = f.as_rational() {
        return Ok(denom(v) != Int::ONE);
    }
    if f.is_pow() || f.is_function(Function::Exp) {
        let exponent: Basic = as_base_exp(f)?.1;
        return Ok(is_true(&exponent, Predicate::negative) || exponent.is_mul());
    }
    Ok(false)
}

/// The terms of the product of the sums (`Mul._expandsums`).
fn expandsums(sums: &[Basic]) -> PolyResult<Vec<Basic>> {
    if sums.len() == 1 {
        return Ok(terms_of(&sums[0]));
    }
    let half: usize = sums.len() / 2;
    let (left, right) = (expandsums(&sums[..half])?, expandsums(&sums[half..])?);
    let mut products: Vec<Basic> = Vec::new();
    for a in &left {
        for b in &right {
            products.push(mul(&[a.clone(), b.clone()])?);
        }
    }
    Ok(terms_of(&add(&products)?))
}

/// SymPy's distribution of a product (`Mul._eval_expand_mul`).
fn expand_mul_node(e: &Basic, deep: bool) -> PolyResult<Basic> {
    let mut expr: Basic = e.clone();
    let mut count: usize = 0;
    for f in e.args() {
        if denominator_factor(f)? {
            count += 1;
        }
    }
    if count >= 2 {
        let (mut n, mut d) = fraction(e)?;
        if d.is_mul() {
            if n.is_mul() {
                n = expand_mul_node(&n, deep)?;
            }
            d = expand_mul_node(&d, deep)?;
        }
        expr = mul(&[n, pow(&d, &Basic::negative_one())?])?;
    }
    if !expr.is_mul() {
        return Ok(expr);
    }
    let mut plain: Vec<Basic> = Vec::new();
    let mut sums: Vec<Basic> = Vec::new();
    for f in expr.args() {
        if f.is_add() || is_false(f, Predicate::commutative) {
            sums.push(f.clone());
        } else {
            plain.push(f.clone());
        }
    }
    if !sums.iter().any(Basic::is_add) {
        return Ok(expr);
    }
    let plain_product: Basic = mul(&plain)?;
    let mut result: Vec<Basic> = Vec::new();
    for term in expandsums(&sums)? {
        let mut t: Basic = mul(&[plain_product.clone(), term])?;
        if t.is_mul() && t.args().iter().any(Basic::is_add) && deep {
            t = expand_mul_node(&t, deep)?;
        }
        result.push(t);
    }
    add(&result)
}

/// The tuples of `parts` nonnegative integers with the sum `n`, the first
/// part decreasing (`_compositions`).
fn compositions(n: usize, parts: usize) -> Vec<Vec<usize>> {
    if parts == 1 {
        return vec![vec![n]];
    }
    let mut result: Vec<Vec<usize>> = Vec::new();
    for first in (0..=n).rev() {
        for rest in compositions(n - first, parts - 1) {
            let mut c: Vec<usize> = vec![first];
            c.extend(rest);
            result.push(c);
        }
    }
    result
}

fn factorial(n: usize) -> Int {
    (1..=n).fold(Int::ONE, |acc, k| acc * Int::from(k))
}

/// `base**n` by the multinomial theorem (`_expand_power`).
fn expand_power(base: &Basic, n: usize) -> PolyResult<Basic> {
    let terms: Vec<Basic> = terms_of(base);
    let mut expanded: Vec<Basic> = Vec::new();
    for exponents in compositions(n, terms.len()) {
        let mut coefficient: Int = factorial(n);
        for k in &exponents {
            coefficient /= factorial(*k);
        }
        let mut factors: Vec<Basic> = vec![Basic::integer(coefficient)];
        for (t, k) in terms.iter().zip(&exponents) {
            if *k > 0 {
                factors.push(pow(t, &Basic::int(*k as i64))?);
            }
        }
        expanded.push(mul(&factors)?);
    }
    add(&expanded)
}

fn expand_product(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    let mut terms: Vec<Basic> = Vec::new();
    for s in terms_of(a) {
        for t in terms_of(b) {
            terms.push(mul(&[s.clone(), t])?);
        }
    }
    add(&terms)
}

fn small(n: &Int) -> PolyResult<usize> {
    usize::try_from(n).map_err(|_| crate::error::not_ported("the expansion of a huge power"))
}

/// One pass of the expansion (`_expand` with the hints `mul` and
/// `multinomial` only).
fn expand_pass(e: &Basic, deep: bool, distribute: bool, multinomial: bool) -> PolyResult<Basic> {
    let mut e: Basic = e.clone();
    if deep && !e.args().is_empty() {
        let mut args: Vec<Basic> = Vec::new();
        for a in e.args() {
            args.push(expand_pass(a, deep, distribute, multinomial)?);
        }
        if args.iter().zip(e.args()).any(|(new, old)| new != old) {
            e = rebuild_evaluated(&e, args)?;
        }
    }
    if e.is_mul() && distribute {
        return expand_mul_node(&e, deep);
    }
    if e.is_pow() && multinomial && e.args()[0].is_add() {
        let b: &Basic = &e.args()[0];
        if let Some(x) = e.args()[1].as_rational() {
            let (p, q) = (numer(x), denom(x));
            if q == Int::ONE && !p.is_zero() {
                let power: Basic = expand_power(b, small(&abs(&p))?)?;
                return if is_negative(&p) {
                    pow(&power, &Basic::negative_one())
                } else {
                    Ok(power)
                };
            }
            if q > Int::ONE && p > q {
                let whole: Int = div_floor(&p, &q);
                let rest: Basic = Basic::from_rational(rat_from_parts(&p - &whole * &q, q.clone()));
                return expand_product(&expand_power(b, small(&whole)?)?, &pow(b, &rest)?);
            }
            if q > Int::ONE && -&p > q {
                let whole: Int = div_floor(&-&p, &q);
                let rest: Basic = Basic::from_rational(rat_from_parts(-&p - &whole * &q, q.clone()));
                let product: Basic = expand_product(&expand_power(b, small(&whole)?)?, &pow(b, &rest)?)?;
                return pow(&product, &Basic::negative_one());
            }
        }
    }
    Ok(e)
}

/// `expand` with the hints `mul` and `multinomial` (the passes repeated
/// until nothing changes, as SymPy's `Expr.expand`).
pub fn expand(e: &Basic, deep: bool, mul_hint: bool, multinomial: bool) -> PolyResult<Basic> {
    let mut result: Basic = expand_pass(e, deep, mul_hint, multinomial)?;
    if mul_hint || multinomial {
        loop {
            let was: Basic = result.clone();
            result = expand_pass(&result, deep, mul_hint, multinomial)?;
            if result == was {
                break;
            }
        }
    }
    Ok(result)
}

/// Products distributed over sums only.
pub fn expand_mul(e: &Basic, deep: bool) -> PolyResult<Basic> {
    expand(e, deep, true, false)
}

/// Integer powers of sums expanded only.
pub fn expand_multinomial(e: &Basic, deep: bool) -> PolyResult<Basic> {
    expand(e, deep, false, true)
}

/// Whether a node is one on which the hints `power_base`, `power_exp` or
/// `log` of the default `expand` can act.
fn touched_by_power_or_log_hints(e: &Basic) -> bool {
    e.preorder().iter().any(|n| {
        (n.is_pow() && (n.args()[0].is_mul() || n.args()[1].is_add()))
            || (n.is_function(Function::Exp) && n.args()[0].is_add())
            || n.is_function(Function::Log)
    })
}

/// `e.expand()` with the default hints, when the hints `power_base`,
/// `power_exp` and `log` have nothing to act on (before and after): then it
/// is the expansion by `mul` and `multinomial`; otherwise `NotImplemented`.
pub fn expand_default(e: &Basic) -> PolyResult<Basic> {
    if touched_by_power_or_log_hints(e) {
        return Err(crate::error::not_ported("expand with the power and log hints"));
    }
    let result: Basic = expand(e, true, true, true)?;
    if touched_by_power_or_log_hints(&result) {
        return Err(crate::error::not_ported("expand with the power and log hints"));
    }
    Ok(result)
}
