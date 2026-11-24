//! The `rewrite` rules of the orthogonal polynomials, ported from
//! `sympy_extras.functions.special._rewrites_polynomials` (SymPy's
//! `_eval_rewrite_as_Sum`, `_eval_rewrite_as_polynomial` and `hermite` and
//! `hermite_prob` in terms of each other), chained into
//! `core::rewriting::rewrite`.

// PORTED FROM SYMPY: sympy/functions/special/polynomials.py (the
// _eval_rewrite_as_Sum, _eval_rewrite_as_polynomial,
// _eval_rewrite_as_hermite_prob and _eval_rewrite_as_hermite methods)
// (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: typed; a table keyed by (head, target); the phase (-1)**m of
// assoc_legendre's sum restored (a SymPy bug).

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::rewriting::{RewriteHints, Rule};
use crate::error::{PolyError, PolyResult};
use crate::functions::combinatorial::factorials::{binomial, factorial, rf};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

/// The targets of these rules (`Sum`, SymPy's old name `polynomial`, and
/// the two Hermite polynomials).
pub const TARGETS: [&str; 4] = ["Sum", "polynomial", "hermite_prob", "hermite"];

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn half() -> Basic {
    Basic::from_rational(crate::numbers::rat(1, 2))
}
fn floor(e: &Basic) -> R {
    crate::functions::elementary::integers::floor(e)
}
fn dummy() -> Basic {
    crate::core::symbol::Dummy::new("k", &[]).0
}
fn sum(kern: &Basic, k: &Basic, upper: &Basic) -> R {
    let limits: Basic = Basic::new(Head::Tuple, vec![k.clone(), Basic::zero(), upper.clone()]);
    crate::concrete::expr_with_limits::sum_(kern, &[limits])
}
fn minus_one_to(k: &Basic) -> R {
    pow(&int(-1), k)
}
fn sqrt2() -> R {
    crate::core::power::pow(&int(2), &half())
}

/// `ValueError` unless `n` may be a nonnegative integer
/// (`_check_nonnegative_integer`).
fn check_nonnegative_integer(n: &Basic) -> PolyResult<()> {
    if is(n, Predicate::negative) == Some(true) || is(n, Predicate::integer) == Some(false) {
        return Err(PolyError::Value(
            "Error: n should be a non-negative integer.".to_string(),
        ));
    }
    Ok(())
}

fn jacobi_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, a, b, x) = (&args[0], &args[1], &args[2], &args[3]);
    check_nonnegative_integer(n)?;
    let k: Basic = dummy();
    let kern: Basic = mul(&[
        rf(&crate::core::mul::neg(n)?, &k)?,
        rf(&add(&[a.clone(), b.clone(), n.clone(), int(1)])?, &k)?,
        rf(&add(&[a.clone(), k.clone(), int(1)])?, &sub(n, &k)?)?,
        pow(&factorial(&k)?, &int(-1))?,
        pow(&div(&sub(&int(1), x)?, &int(2))?, &k)?,
    ])?;
    Ok(Some(mul(&[pow(&factorial(n)?, &int(-1))?, sum(&kern, &k, n)?])?))
}

fn gegenbauer_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, a, x) = (&args[0], &args[1], &args[2]);
    let k: Basic = dummy();
    let n2k: Basic = sub(n, &mul(&[int(2), k.clone()])?)?;
    let kern: Basic = div(
        &mul(&[
            minus_one_to(&k)?,
            rf(a, &sub(n, &k)?)?,
            pow(&mul(&[int(2), x.clone()])?, &n2k)?,
        ])?,
        &mul(&[factorial(&k)?, factorial(&n2k)?])?,
    )?;
    Ok(Some(sum(&kern, &k, &floor(&div(n, &int(2))?)?)?))
}

fn chebyshevt_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, x) = (&args[0], &args[1]);
    let k: Basic = dummy();
    let two_k: Basic = mul(&[int(2), k.clone()])?;
    let kern: Basic = mul(&[
        binomial(n, &two_k)?,
        pow(&sub(&pow(x, &int(2))?, &int(1))?, &k)?,
        pow(x, &sub(n, &two_k)?)?,
    ])?;
    Ok(Some(sum(&kern, &k, &floor(&div(n, &int(2))?)?)?))
}

fn chebyshevu_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, x) = (&args[0], &args[1]);
    let k: Basic = dummy();
    let n2k: Basic = sub(n, &mul(&[int(2), k.clone()])?)?;
    let kern: Basic = div(
        &mul(&[
            minus_one_to(&k)?,
            factorial(&sub(n, &k)?)?,
            pow(&mul(&[int(2), x.clone()])?, &n2k)?,
        ])?,
        &mul(&[factorial(&k)?, factorial(&n2k)?])?,
    )?;
    Ok(Some(sum(&kern, &k, &floor(&div(n, &int(2))?)?)?))
}

fn legendre_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, x) = (&args[0], &args[1]);
    let k: Basic = dummy();
    let kern: Basic = mul(&[
        minus_one_to(&k)?,
        pow(&binomial(n, &k)?, &int(2))?,
        pow(&div(&add(&[int(1), x.clone()])?, &int(2))?, &sub(n, &k)?)?,
        pow(&div(&sub(&int(1), x)?, &int(2))?, &k)?,
    ])?;
    Ok(Some(sum(&kern, &k, n)?))
}

fn assoc_legendre_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, m, x) = (&args[0], &args[1], &args[2]);
    let k: Basic = dummy();
    let two: Basic = int(2);
    let two_k: Basic = mul(&[two.clone(), k.clone()])?;
    let kern: Basic = mul(&[
        div(
            &factorial(&sub(&mul(&[two.clone(), n.clone()])?, &two_k)?)?,
            &mul(&[
                pow(&two, n)?,
                factorial(&sub(n, &k)?)?,
                factorial(&k)?,
                factorial(&sub(&sub(n, &two_k)?, m)?)?,
            ])?,
        )?,
        minus_one_to(&k)?,
        pow(x, &sub(&sub(n, m)?, &two_k)?)?,
    ])?;
    // SymPy's rule omits the Condon-Shortley phase (-1)**m which its
    // assoc_legendre has: restored here
    let upper: Basic = floor(&mul(&[sub(n, m)?, half()])?)?;
    Ok(Some(mul(&[
        minus_one_to(m)?,
        pow(&sub(&int(1), &pow(x, &two)?)?, &div(m, &two)?)?,
        sum(&kern, &k, &upper)?,
    ])?))
}

fn hermite_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, x) = (&args[0], &args[1]);
    let k: Basic = dummy();
    let n2k: Basic = sub(n, &mul(&[int(2), k.clone()])?)?;
    let kern: Basic = mul(&[
        div(&minus_one_to(&k)?, &mul(&[factorial(&k)?, factorial(&n2k)?])?)?,
        pow(&mul(&[int(2), x.clone()])?, &n2k)?,
    ])?;
    Ok(Some(mul(&[
        factorial(n)?,
        sum(&kern, &k, &floor(&div(n, &int(2))?)?)?,
    ])?))
}

fn hermite_as_prob(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, x) = (&args[0], &args[1]);
    let s: Basic = sqrt2()?;
    let p: Basic = crate::functions::special::polynomials::hermite_prob(n, &mul(&[x.clone(), s.clone()])?)?;
    Ok(Some(mul(&[pow(&s, n)?, p])?))
}

fn hermite_prob_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, x) = (&args[0], &args[1]);
    let k: Basic = dummy();
    let n2k: Basic = sub(n, &mul(&[int(2), k.clone()])?)?;
    let kern: Basic = div(
        &mul(&[
            pow(&Basic::from_rational(crate::numbers::rat(-1, 2)), &k)?,
            pow(x, &n2k)?,
        ])?,
        &mul(&[factorial(&k)?, factorial(&n2k)?])?,
    )?;
    Ok(Some(mul(&[
        factorial(n)?,
        sum(&kern, &k, &floor(&div(n, &int(2))?)?)?,
    ])?))
}

fn prob_as_hermite(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    let (n, x) = (&args[0], &args[1]);
    let s: Basic = sqrt2()?;
    let h: Basic = crate::functions::special::polynomials::hermite(n, &div(x, &s)?)?;
    Ok(Some(mul(&[pow(&s, &crate::core::mul::neg(n)?)?, h])?))
}

/// The sum of `laguerre(n, x)` (`_laguerre_sum_of`).
fn laguerre_sum_of(n: &Basic, x: &Basic) -> R {
    if is(n, Predicate::negative) == Some(true) {
        let m: Basic = sub(&crate::core::mul::neg(n)?, &int(1))?;
        return mul(&[
            crate::functions::elementary::exponential::exp(x)?,
            laguerre_sum_of(&m, &crate::core::mul::neg(x)?)?,
        ]);
    }
    if is(n, Predicate::integer) == Some(false) {
        return Err(PolyError::Value("Error: n should be an integer.".to_string()));
    }
    let k: Basic = dummy();
    let kern: Basic = mul(&[
        rf(&crate::core::mul::neg(n)?, &k)?,
        pow(&factorial(&k)?, &int(-2))?,
        pow(x, &k)?,
    ])?;
    sum(&kern, &k, n)
}

fn laguerre_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    Ok(Some(laguerre_sum_of(&args[0], &args[1])?))
}

fn assoc_laguerre_sum(_: &Basic, args: &[Basic], _: &RewriteHints) -> OptR {
    use crate::functions::special::gamma_functions::gamma;
    let (n, alpha, x) = (&args[0], &args[1], &args[2]);
    check_nonnegative_integer(n)?;
    let k: Basic = dummy();
    let kern: Basic = mul(&[
        div(
            &rf(&crate::core::mul::neg(n)?, &k)?,
            &mul(&[gamma(&add(&[k.clone(), alpha.clone(), int(1)])?)?, factorial(&k)?])?,
        )?,
        pow(x, &k)?,
    ])?;
    let front: Basic = div(&gamma(&add(&[n.clone(), alpha.clone(), int(1)])?)?, &factorial(n)?)?;
    Ok(Some(mul(&[front, sum(&kern, &k, n)?])?))
}

/// The rule of the head `f` for the target, `None` without one
/// (`rules()`).
pub fn rule_for(f: Function, target: &str) -> Option<Rule> {
    use Function as F;
    let as_sum: bool = target == "Sum" || target == "polynomial";
    let rule: Rule = match (f, target) {
        (F::Jacobi, _) if as_sum => jacobi_sum,
        (F::Gegenbauer, _) if as_sum => gegenbauer_sum,
        (F::ChebyshevT, _) if as_sum => chebyshevt_sum,
        (F::ChebyshevU, _) if as_sum => chebyshevu_sum,
        (F::Legendre, _) if as_sum => legendre_sum,
        (F::AssocLegendre, _) if as_sum => assoc_legendre_sum,
        (F::Hermite, _) if as_sum => hermite_sum,
        (F::HermiteProb, _) if as_sum => hermite_prob_sum,
        (F::Laguerre, _) if as_sum => laguerre_sum,
        (F::AssocLaguerre, _) if as_sum => assoc_laguerre_sum,
        (F::Hermite, "hermite_prob") => hermite_as_prob,
        (F::HermiteProb, "hermite") => prob_as_hermite,
        _ => return None,
    };
    Some(rule)
}
