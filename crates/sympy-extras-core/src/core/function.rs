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
        Head::Derivative => {
            // Derivative._evaluated: diff of the new expression
            let counts: Vec<(Basic, u64)> = derivative_counts(&e.rebuild(args.clone()));
            diff(&args[0], &counts)
        }
        Head::Subs => substituted(&args[0], args[1].args(), args[2].args()),
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

// ---------------------------------------------------------------------
// derivatives (`Derivative`, `Subs`, `diff`)

/// `Tuple(*items)`.
pub fn tuple(items: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, items)
}

/// The unevaluated `Derivative(expr, (s1, n1), ...)`, consecutive counts
/// of the same variable merged (`_merge_counts`).
pub fn derivative_node(expr: &Basic, counts: &[(Basic, u64)]) -> Basic {
    let mut merged: Vec<(Basic, u64)> = Vec::new();
    for (s, n) in counts {
        match merged.last_mut() {
            Some((last, m)) if last == s => *m += n,
            _ => merged.push((s.clone(), *n)),
        }
    }
    let mut args: Vec<Basic> = vec![expr.clone()];
    for (s, n) in merged {
        args.push(tuple(vec![s, Basic::integer(Int::from(n))]));
    }
    Basic::new(Head::Derivative, args)
}

/// The variables and counts of a `Derivative` node.
pub fn derivative_counts(e: &Basic) -> Vec<(Basic, u64)> {
    e.args()[1..]
        .iter()
        .map(|pair| {
            let n: u64 = u64::try_from(pair.args()[1].as_rational().expect("a count").numerator()).expect("a count");
            (pair.args()[0].clone(), n)
        })
        .collect()
}

/// The unevaluated `Subs(expr, variables, point)`.
pub fn subs_node(expr: &Basic, variables: Vec<Basic>, point: Vec<Basic>) -> Basic {
    Basic::new(Head::Subs, vec![expr.clone(), tuple(variables), tuple(point)])
}

/// `expr` with the variables replaced by the points, a `Subs` kept around
/// the variables of a derivative which are not replaced by symbols
/// (`_substituted`).
pub fn substituted(expr: &Basic, variables: &[Basic], point: &[Basic]) -> PolyResult<Basic> {
    let derivatives: Vec<Basic> = expr
        .preorder()
        .into_iter()
        .filter(|n| matches!(n.head(), Head::Derivative))
        .collect();
    let free_symbols = expr.free_symbols();
    let mut blocked: Vec<(Basic, Basic)> = Vec::new();
    let mut free: Vec<(Basic, Basic)> = Vec::new();
    for (v, p) in variables.iter().zip(point) {
        let differentiated: bool = derivatives
            .iter()
            .any(|d| derivative_counts(d).iter().any(|(s, _)| s == v));
        let capturing: bool = !p.is_symbol() || (free_symbols.contains(p) && p != v);
        if differentiated && capturing {
            blocked.push((v.clone(), p.clone()));
        } else {
            free.push((v.clone(), p.clone()));
        }
    }
    let inner: Basic = if free.is_empty() {
        expr.clone()
    } else {
        super::subs::subs_pairs(expr, &free)?
    };
    if blocked.is_empty() {
        return Ok(inner);
    }
    let (vs, ps): (Vec<Basic>, Vec<Basic>) = blocked.into_iter().unzip();
    Ok(subs_node(&inner, vs, ps))
}

fn is_zero_rational(e: &Basic) -> bool {
    e.as_rational().is_some_and(|v| v.is_zero())
}

/// The derivative of `e` with respect to the symbol `s` (`_derivative`).
pub fn derivative(e: &Basic, s: &Basic) -> PolyResult<Basic> {
    if e == s {
        return Ok(Basic::one());
    }
    if !e.free_symbols().contains(s) {
        return Ok(Basic::zero());
    }
    match e.head() {
        Head::Add => {
            let mut terms: Vec<Basic> = Vec::new();
            for t in e.args() {
                terms.push(derivative(t, s)?);
            }
            add(&terms)
        }
        Head::Mul => {
            let mut terms: Vec<Basic> = Vec::new();
            for (i, f) in e.args().iter().enumerate() {
                let d: Basic = derivative(f, s)?;
                if is_zero_rational(&d) {
                    continue;
                }
                let mut factors: Vec<Basic> = e.args()[..i].to_vec();
                factors.push(d);
                factors.extend(e.args()[i + 1..].iter().cloned());
                terms.push(mul(&factors)?);
            }
            add(&terms)
        }
        Head::Pow => {
            let (b, ex) = (&e.args()[0], &e.args()[1]);
            if !ex.free_symbols().contains(s) {
                let inner: Basic = mul(&[ex.clone(), derivative(b, s)?])?;
                return mul(&[inner, e.clone(), pow(b, &Basic::negative_one())?]);
            }
            let log_b: Basic = crate::functions::elementary::exponential::log(b)?;
            let first: Basic = mul(&[derivative(ex, s)?, log_b])?;
            let second: Basic = mul(&[ex.clone(), derivative(b, s)?, pow(b, &Basic::negative_one())?])?;
            mul(&[e.clone(), add(&[first, second])?])
        }
        Head::Function(f) => {
            if let Some(own) = crate::functions::eval_derivative(*f, e, s)? {
                return Ok(own);
            }
            let mut chain: Vec<Basic> = Vec::new();
            let da: Basic = derivative(&e.args()[0], s)?;
            if !is_zero_rational(&da) {
                chain.push(mul(&[crate::functions::fdiff(*f, e)?, da])?);
            }
            add(&chain)
        }
        Head::Undefined(u) => {
            let depending: usize = e.args().iter().filter(|a| a.free_symbols().contains(s)).count();
            let mut terms: Vec<Basic> = Vec::new();
            for (i, a) in e.args().iter().enumerate() {
                if !a.free_symbols().contains(s) {
                    continue;
                }
                if a.is_symbol() && depending <= 1 {
                    terms.push(derivative_node(e, &[(a.clone(), 1)]));
                    continue;
                }
                let xi: Basic = super::Dummy::new(&format!("xi_{}", i + 1), &[]).0;
                let mut args: Vec<Basic> = e.args().to_vec();
                args[i] = xi.clone();
                let inner: Basic = u.apply(args)?;
                let at: Basic = substituted(
                    &derivative_node(&inner, &[(xi.clone(), 1)]),
                    &[xi],
                    std::slice::from_ref(a),
                )?;
                terms.push(mul(&[at, derivative(a, s)?])?);
            }
            add(&terms)
        }
        Head::Derivative => {
            let mut counts: Vec<(Basic, u64)> = derivative_counts(e);
            counts.push((s.clone(), 1));
            Ok(derivative_node(&e.args()[0], &counts))
        }
        Head::Subs => {
            let (expr, variables, point) = (&e.args()[0], e.args()[1].args(), e.args()[2].args());
            let mut terms: Vec<Basic> = Vec::new();
            if expr.free_symbols().contains(s) && !variables.contains(s) {
                terms.push(substituted(&derivative(expr, s)?, variables, point)?);
            }
            for (v, p) in variables.iter().zip(point) {
                let dp: Basic = derivative(p, s)?;
                if is_zero_rational(&dp) {
                    continue;
                }
                terms.push(mul(&[substituted(&derivative(expr, v)?, variables, point)?, dp])?);
            }
            add(&terms)
        }
        _ => Ok(derivative_node(e, &[(s.clone(), 1)])),
    }
}

/// The derivative of `e` with respect to the symbols with their counts
/// (`diff(e, (x, 2), y)`).
pub fn diff(e: &Basic, counts: &[(Basic, u64)]) -> PolyResult<Basic> {
    let mut result: Basic = e.clone();
    for (s, n) in counts {
        for _ in 0..*n {
            result = derivative(&result, s)?;
        }
    }
    Ok(result)
}

/// The tree evaluated bottom up with the evaluating functions (`doit`).
pub fn doit(e: &Basic) -> PolyResult<Basic> {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(doit(a)?);
    }
    rebuild_evaluated(e, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::basic::UndefinedFunction;
    use crate::core::printing::sstr;
    use crate::core::Symbol;
    use std::sync::Arc;

    #[test]
    fn docstring_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        let cube: Basic = pow(&x, &Basic::int(3)).unwrap();
        assert_eq!(show(diff(&cube, &[(x.clone(), 1)])), "3*x**2");
        assert_eq!(
            show(diff(&pow(&x, &Basic::half()).unwrap(), &[(x.clone(), 1)])),
            "1/(2*sqrt(x))"
        );
        let f = Arc::new(UndefinedFunction::new("f", None, &[]));
        let fx: Basic = f.apply(vec![x.clone()]).unwrap();
        assert_eq!(
            show(diff(&mul(&[x.clone(), fx]).unwrap(), &[(x.clone(), 1)])),
            "x*Derivative(f(x), x) + f(x)"
        );
        let fx2: Basic = f.apply(vec![pow(&x, &Basic::int(2)).unwrap()]).unwrap();
        assert_eq!(
            show(diff(&fx2, &[(x.clone(), 1)])),
            "2*x*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, x**2)"
        );
        let two_x: Basic = pow(&Basic::int(2), &x).unwrap();
        assert_eq!(show(diff(&two_x, &[(x.clone(), 1)])), "2**x*log(2)");
    }
}
