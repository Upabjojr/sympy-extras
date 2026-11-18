//! `rewrite`: the nodes of an expression rewritten in terms of a function,
//! ported from `sympy_extras.core.rewriting` (SymPy's `Basic.rewrite` and
//! `Basic._rewrite`): bottom up, each node (of the given pattern
//! functions, all when none) is replaced by its rule for the target, the
//! node rebuilt with its rewritten arguments otherwise. The target is
//! SymPy's name of it (what follows `_eval_rewrite_as_`: `gamma`,
//! `factorial`, `RisingFactorial`, ...). The rules ported are those of the
//! combinatorial functions into `gamma`, `factorial`, `binomial`,
//! `RisingFactorial` and `FallingFactorial`, and those of the elementary
//! functions into `exp`, `cos`, `sin` and `log`
//! (`functions/elementary/rewrites.rs`); another target is not ported.

// PORTED FROM SYMPY: sympy/core/basic.py (Basic.rewrite, Basic._rewrite),
// sympy/functions/combinatorial/factorials.py, sympy/functions/combinatorial/numbers.py,
// sympy/functions/special/gamma_functions.py (the _eval_rewrite_as_<name> methods) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::logic::true_;
use crate::core::mul::{mul, neg};
use crate::core::power::pow;
use crate::error::{not_ported, PolyResult};
use crate::functions::apply_function;

/// The hints of a rewrite (SymPy's keyword arguments): `piecewise` lets
/// the rules of the rising and falling factorials return a `Piecewise`.
#[derive(Debug, Clone, Copy)]
pub struct RewriteHints {
    pub piecewise: bool,
}

impl Default for RewriteHints {
    fn default() -> RewriteHints {
        RewriteHints { piecewise: true }
    }
}

/// A rule: the node and its rewritten arguments to the rewritten node,
/// `None` when it does not apply.
pub type Rule = fn(&Basic, &[Basic], &RewriteHints) -> PolyResult<Option<Basic>>;

/// The targets whose rules are ported.
const PORTED_TARGETS: [&str; 31] = [
    "gamma",
    "factorial",
    "binomial",
    "RisingFactorial",
    "FallingFactorial",
    "exp",
    "cos",
    "sin",
    "log",
    "tan",
    "cot",
    "sec",
    "csc",
    "sincos",
    "sinh",
    "cosh",
    "tanh",
    "coth",
    "csch",
    "sech",
    "asin",
    "acos",
    "atan",
    "acot",
    "asec",
    "acsc",
    "asinh",
    "acosh",
    "atanh",
    "acsch",
    "Abs",
];

fn one() -> Basic {
    Basic::one()
}

fn minus_one() -> Basic {
    Basic::negative_one()
}

fn inv(e: &Basic) -> PolyResult<Basic> {
    pow(e, &minus_one())
}

fn call(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    apply_function(f, args)
}

fn gamma(x: &Basic) -> PolyResult<Basic> {
    call(Function::Gamma, std::slice::from_ref(x))
}

fn factorial(x: &Basic) -> PolyResult<Basic> {
    call(Function::Factorial, std::slice::from_ref(x))
}

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn piecewise(pairs: &[(Basic, Basic)]) -> PolyResult<Basic> {
    crate::functions::elementary::piecewise::piecewise(pairs)
}

fn both_integer(x: &Basic, k: &Basic) -> bool {
    // Python's `x.is_integer and k.is_integer`
    is(x, Predicate::integer) == Some(true) && is(k, Predicate::integer) == Some(true)
}

// the rules into gamma

fn factorial_as_gamma(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    Ok(Some(gamma(&add(&[a[0].clone(), one()])?)?))
}

fn subfactorial_as_gamma(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::exponential::exp;
    let a1: Basic = add(&[a[0].clone(), one()])?;
    let sign: Basic = pow(&minus_one(), &a1)?;
    let phase: Basic = exp(&mul(&[neg(&Basic::imaginary_unit())?, Basic::pi(), a[0].clone()])?)?;
    let lower: Basic = call(Function::LowerGamma, &[a1.clone(), minus_one()])?;
    let first: Basic = mul(&[sign, phase, lower])?;
    Ok(Some(mul(&[add(&[first, gamma(&a1)?])?, exp(&minus_one())?])?))
}

fn factorial2_as_gamma(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    use crate::core::relational::eq;
    let n: &Basic = &a[0];
    let half: Basic = mul(&[n.clone(), Basic::half()])?;
    let parity: Basic = crate::core::modulo::modulo(n, &int(2))?;
    let root: Basic = pow(&mul(&[int(2), inv(&Basic::pi())?])?, &Basic::half())?;
    let cases: Basic = piecewise(&[(one(), eq(&parity, &Basic::zero())?), (root, eq(&parity, &one())?)])?;
    Ok(Some(mul(&[
        pow(&int(2), &half)?,
        gamma(&add(&[half, one()])?)?,
        cases,
    ])?))
}

fn rising_as_gamma(_: &Basic, a: &[Basic], hints: &RewriteHints) -> PolyResult<Option<Basic>> {
    use crate::core::relational::{gt, le};
    let (x, k) = (&a[0], &a[1]);
    let positive: Basic = mul(&[gamma(&add(&[x.clone(), k.clone()])?)?, inv(&gamma(x)?)?])?;
    let other: Basic = mul(&[
        mul(&[pow(&minus_one(), k)?, gamma(&add(&[one(), neg(x)?])?)?])?,
        inv(&gamma(&add(&[neg(k)?, neg(x)?, one()])?)?)?,
    ])?;
    if !hints.piecewise {
        return Ok(Some(if le(x, &Basic::zero())? == true_() {
            other
        } else {
            positive
        }));
    }
    Ok(Some(piecewise(&[
        (positive, gt(x, &Basic::zero())?),
        (other, true_()),
    ])?))
}

fn falling_as_gamma(_: &Basic, a: &[Basic], hints: &RewriteHints) -> PolyResult<Option<Basic>> {
    use crate::core::relational::{ge, lt};
    let (x, k) = (&a[0], &a[1]);
    let nonnegative: Basic = mul(&[
        gamma(&add(&[x.clone(), one()])?)?,
        inv(&gamma(&add(&[x.clone(), neg(k)?, one()])?)?)?,
    ])?;
    let other: Basic = mul(&[
        mul(&[pow(&minus_one(), k)?, gamma(&add(&[k.clone(), neg(x)?])?)?])?,
        inv(&gamma(&neg(x)?)?)?,
    ])?;
    if !hints.piecewise {
        return Ok(Some(if lt(x, &Basic::zero())? == true_() {
            other
        } else {
            nonnegative
        }));
    }
    Ok(Some(piecewise(&[
        (nonnegative, ge(x, &Basic::zero())?),
        (other, true_()),
    ])?))
}

fn binomial_as_gamma(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let (n, k) = (&a[0], &a[1]);
    let den: Basic = mul(&[
        gamma(&add(&[k.clone(), one()])?)?,
        gamma(&add(&[n.clone(), neg(k)?, one()])?)?,
    ])?;
    Ok(Some(mul(&[gamma(&add(&[n.clone(), one()])?)?, inv(&den)?])?))
}

fn catalan_as_gamma(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let n: &Basic = &a[0];
    let half: Basic = Basic::half();
    let den: Basic = mul(&[gamma(&half)?, gamma(&add(&[n.clone(), int(2)])?)?])?;
    Ok(Some(mul(&[
        pow(&int(4), n)?,
        gamma(&add(&[n.clone(), half])?)?,
        inv(&den)?,
    ])?))
}

fn beta_as_gamma(node: &Basic, _: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    // SymPy's beta._eval_expand_func, on the arguments of the node itself
    let (x, y) = (&node.args()[0], &node.args()[1]);
    Ok(Some(mul(&[
        gamma(x)?,
        gamma(y)?,
        inv(&gamma(&add(&[x.clone(), y.clone()])?)?)?,
    ])?))
}

// the rules into factorial, binomial and the rising and falling factorials

fn gamma_as_factorial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    Ok(Some(factorial(&add(&[a[0].clone(), minus_one()])?)?))
}

fn binomial_as_factorial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let (n, k) = (&a[0], &a[1]);
    let den: Basic = mul(&[factorial(k)?, factorial(&add(&[n.clone(), neg(k)?])?)?])?;
    Ok(Some(mul(&[factorial(n)?, inv(&den)?])?))
}

fn rising_as_factorial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    use crate::core::relational::gt;
    let (x, k) = (&a[0], &a[1]);
    if !both_integer(x, k) {
        return Ok(None);
    }
    let positive: Basic = mul(&[
        factorial(&add(&[k.clone(), x.clone(), minus_one()])?)?,
        inv(&factorial(&add(&[x.clone(), minus_one()])?)?)?,
    ])?;
    let other: Basic = mul(&[
        pow(&minus_one(), k)?,
        factorial(&neg(x)?)?,
        inv(&factorial(&add(&[neg(k)?, neg(x)?])?)?)?,
    ])?;
    Ok(Some(piecewise(&[
        (positive, gt(x, &Basic::zero())?),
        (other, true_()),
    ])?))
}

fn falling_as_factorial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    use crate::core::relational::ge;
    let (x, k) = (&a[0], &a[1]);
    if !both_integer(x, k) {
        return Ok(None);
    }
    let nonnegative: Basic = mul(&[factorial(x)?, inv(&factorial(&add(&[neg(k)?, x.clone()])?)?)?])?;
    let other: Basic = mul(&[
        pow(&minus_one(), k)?,
        factorial(&add(&[k.clone(), neg(x)?, minus_one()])?)?,
        inv(&factorial(&add(&[neg(x)?, minus_one()])?)?)?,
    ])?;
    Ok(Some(piecewise(&[
        (nonnegative, ge(x, &Basic::zero())?),
        (other, true_()),
    ])?))
}

fn catalan_as_factorial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let n: &Basic = &a[0];
    let two_n: Basic = mul(&[int(2), n.clone()])?;
    let den: Basic = mul(&[factorial(&add(&[n.clone(), one()])?)?, factorial(n)?])?;
    Ok(Some(mul(&[factorial(&two_n)?, inv(&den)?])?))
}

fn subfactorial_as_factorial(_: &Basic, _: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    Err(not_ported("subfactorial rewritten as factorial (a summation)"))
}

fn rising_as_falling(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let (x, k) = (&a[0], &a[1]);
    let top: Basic = add(&[x.clone(), k.clone(), minus_one()])?;
    Ok(Some(call(Function::FallingFactorial, &[top, k.clone()])?))
}

fn rising_as_binomial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let (x, k) = (&a[0], &a[1]);
    if is(k, Predicate::integer) != Some(true) {
        return Ok(None);
    }
    let top: Basic = add(&[x.clone(), k.clone(), minus_one()])?;
    Ok(Some(mul(&[
        factorial(k)?,
        call(Function::Binomial, &[top, k.clone()])?,
    ])?))
}

fn falling_as_rising(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let (x, k) = (&a[0], &a[1]);
    let bottom: Basic = add(&[x.clone(), neg(k)?, one()])?;
    Ok(Some(call(Function::RisingFactorial, &[bottom, k.clone()])?))
}

fn falling_as_binomial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let (x, k) = (&a[0], &a[1]);
    if is(k, Predicate::integer) != Some(true) {
        return Ok(None);
    }
    Ok(Some(mul(&[
        factorial(k)?,
        call(Function::Binomial, &[x.clone(), k.clone()])?,
    ])?))
}

fn binomial_as_falling(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let (n, k) = (&a[0], &a[1]);
    if is(k, Predicate::integer) != Some(true) {
        return Ok(None);
    }
    let falling: Basic = call(Function::FallingFactorial, &[n.clone(), k.clone()])?;
    Ok(Some(mul(&[falling, inv(&factorial(k)?)?])?))
}

fn catalan_as_binomial(_: &Basic, a: &[Basic], _: &RewriteHints) -> PolyResult<Option<Basic>> {
    let n: &Basic = &a[0];
    let two_n: Basic = mul(&[int(2), n.clone()])?;
    let b: Basic = call(Function::Binomial, &[two_n, n.clone()])?;
    Ok(Some(mul(&[b, inv(&add(&[n.clone(), one()])?)?])?))
}

/// The rule of a function for a ported target, `None` without one.
fn rule_for(f: Function, target: &str) -> Option<Rule> {
    use Function as F;
    let rule: Rule = match (f, target) {
        (F::Factorial, "gamma") => factorial_as_gamma,
        (F::Subfactorial, "gamma") => subfactorial_as_gamma,
        (F::Factorial2, "gamma") => factorial2_as_gamma,
        (F::RisingFactorial, "gamma") => rising_as_gamma,
        (F::FallingFactorial, "gamma") => falling_as_gamma,
        (F::Binomial, "gamma") => binomial_as_gamma,
        (F::Catalan, "gamma") => catalan_as_gamma,
        (F::Beta, "gamma") => beta_as_gamma,
        (F::Gamma, "factorial") => gamma_as_factorial,
        (F::Binomial, "factorial") => binomial_as_factorial,
        (F::RisingFactorial, "factorial") => rising_as_factorial,
        (F::FallingFactorial, "factorial") => falling_as_factorial,
        (F::Catalan, "factorial") => catalan_as_factorial,
        (F::Subfactorial, "factorial") => subfactorial_as_factorial,
        (F::Catalan, "binomial") => catalan_as_binomial,
        (F::RisingFactorial, "binomial") => rising_as_binomial,
        (F::FallingFactorial, "binomial") => falling_as_binomial,
        (F::FallingFactorial, "RisingFactorial") => falling_as_rising,
        (F::RisingFactorial, "FallingFactorial") => rising_as_falling,
        (F::Binomial, "FallingFactorial") => binomial_as_falling,
        _ => return None,
    };
    Some(rule)
}

fn has_pattern(e: &Basic, p: Function) -> bool {
    matches!(e.head(), Head::Function(f) if *f == p) || e.args().iter().any(|a| has_pattern(a, p))
}

/// `e` with its nodes (the applications of `patterns`, all when empty)
/// rewritten in terms of `target`, SymPy's name of it (`rewrite`).
pub fn rewrite(e: &Basic, patterns: &[Function], target: &str, hints: &RewriteHints) -> PolyResult<Basic> {
    if !PORTED_TARGETS.contains(&target) {
        return Err(not_ported(&format!("rewrite in terms of {}", target)));
    }
    let mut kept: Vec<Function> = Vec::new();
    if !patterns.is_empty() {
        kept = patterns.iter().copied().filter(|p| has_pattern(e, *p)).collect();
        if kept.is_empty() {
            return Ok(e.clone());
        }
    }
    rewrite_rec(e, &kept, target, hints)
}

fn rewrite_rec(e: &Basic, patterns: &[Function], target: &str, hints: &RewriteHints) -> PolyResult<Basic> {
    let mut args: Vec<Basic> = Vec::with_capacity(e.args().len());
    for a in e.args() {
        args.push(rewrite_rec(a, patterns, target, hints)?);
    }
    let matched: bool = patterns.is_empty() || matches!(e.head(), Head::Function(f) if patterns.contains(f));
    if matched {
        let rule: Option<Rule> = match e.head() {
            Head::Function(f) => rule_for(*f, target),
            _ => None,
        }
        .or_else(|| crate::functions::elementary::rewrites::rule_for(e, target));
        if let Some(rule) = rule {
            if let Some(rewritten) = rule(e, &args, hints)? {
                return Ok(rewritten);
            }
        }
    }
    if args.is_empty() {
        return Ok(e.clone());
    }
    super::function::rebuild_evaluated(e, args)
}
