//! Port of `sympy_extras/solvers/rootclassification.py`: the number of
//! real roots of a polynomial with parametric coefficients by the
//! discrimination sequence (the leading principal minors of the Hankel
//! matrix of the power sums, Yang's revised sign list): for every sign
//! pattern of the sequence the number of distinct real roots, and the
//! conditions on the parameters.

use crate::core::assumptions::{is_number, Predicate};
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::is;
use crate::core::logic::boolean;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::coefficients as coefficients_of;
use crate::matrices::linalg::{det, normal};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn relation(op: RelOp, a: &Basic) -> Basic {
    Basic::new(Head::Relational(op), vec![a.clone(), int(0)])
}
fn conjunction(v: Vec<Basic>) -> Basic {
    Basic::new(Head::And, v)
}

fn coefficient_list(p: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let (numerator, denominator) = crate::simplify::radsimp::fraction(&normal(p)?)?;
    if denominator.has(x) {
        return Err(value_error(format!("{} is not a polynomial in {}", p, x)));
    }
    let expanded: Basic = expand(&numerator)?;
    if expanded == int(0) {
        Ok(vec![int(0)])
    } else {
        coefficients_of(&expanded, x)
    }
}

fn minors(cs: &[Basic]) -> PolyResult<Vec<Basic>> {
    let n: usize = cs.len() - 1;
    let leading: Basic = cs[0].clone();
    let c: Vec<Basic> = cs
        .iter()
        .map(|a| normal(&d2(a, &leading)?))
        .collect::<PolyResult<_>>()?;
    let mut sums: Vec<Basic> = vec![int(n as i64)];
    for k in 1..(2 * n).saturating_sub(1) {
        let mut total: Vec<Basic> = Vec::new();
        for i in 1..=k.min(n) {
            if i == k {
                total.push(m2(&int(-(k as i64)), &c[i])?);
            } else {
                total.push(m2(&crate::core::mul::neg(&c[i])?, &sums[k - i])?);
            }
        }
        sums.push(normal(&crate::core::add::add(&total)?)?);
    }
    let mut result: Vec<Basic> = Vec::new();
    for k in 1..=n {
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..k {
            for j in 0..k {
                entries.push(sums[i + j].clone());
            }
        }
        let minor: Basic = det(&ExprMatrix::new(k, k, entries)?)?;
        let (numerator, denominator) = crate::simplify::radsimp::fraction(&normal(&minor)?)?;
        result.push(without_even_powers(&expand(&m2(&numerator, &denominator)?)?, &leading)?);
    }
    Ok(result)
}

fn without_even_powers(e: &Basic, leading: &Basic) -> R {
    if is_number(leading) || *e == int(0) {
        return Ok(e.clone());
    }
    let square: Basic = expand(&crate::core::power::pow(leading, &int(2))?)?;
    let mut current: Basic = e.clone();
    for _ in 0..64 {
        let quotient: Basic = normal(&d2(&current, &square)?)?;
        if crate::simplify::radsimp::fraction(&quotient)?.1 != int(1) {
            break;
        }
        current = expand(&quotient)?;
    }
    Ok(current)
}

/// The discrimination sequence of `p` in `x` (`discrimination_sequence`).
pub fn discrimination_sequence(p: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let cs: Vec<Basic> = coefficient_list(p, x)?;
    if cs.len() < 2 {
        return Ok(Vec::new());
    }
    minors(&cs)
}

/// The number of distinct real roots from the signs of the
/// discrimination sequence (`signature_from_signs`).
pub fn signature_from_signs(signs: &[i64]) -> i64 {
    let (mut total, mut previous, mut gap) = (0i64, 1i64, 0i64);
    for sign in signs {
        if *sign == 0 {
            gap += 1;
            continue;
        }
        if gap % 2 == 0 {
            let s: i64 = if (gap / 2) % 2 == 0 { 1 } else { -1 };
            total += s * previous * sign;
        }
        previous = *sign;
        gap = 0;
    }
    total
}

fn sign_of(q: &Basic) -> i64 {
    if is(q, Predicate::positive) == Some(true) {
        1
    } else if is(q, Predicate::negative) == Some(true) {
        -1
    } else {
        0
    }
}

/// `None` for an impossible sign, `true` for a certain one.
fn condition(quantity: &Basic, sign: i64) -> Option<Basic> {
    if is_number(quantity) {
        return if sign_of(quantity) == sign {
            Some(boolean(true))
        } else {
            None
        };
    }
    Some(match sign {
        1 => relation(RelOp::Gt, quantity),
        -1 => relation(RelOp::Lt, quantity),
        _ => relation(RelOp::Eq, quantity),
    })
}

fn simplified(formula: &Basic) -> R {
    match crate::assumptions::resolve::resolve(formula, None, &[], None) {
        Ok(r) => Ok(r),
        Err(PolyError::Value(_)) => Ok(formula.clone()),
        Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => Ok(formula.clone()),
        Err(e) => Err(e),
    }
}

fn push(by_count: &mut Vec<(Option<i64>, Vec<Basic>)>, key: Option<i64>, value: Basic) {
    match by_count.iter_mut().find(|(k, _)| *k == key) {
        Some(entry) => entry.1.push(value),
        None => by_count.push((key, vec![value])),
    }
}

fn all_or_true(v: Vec<Basic>) -> Basic {
    if v.is_empty() {
        boolean(true)
    } else {
        conjunction(v)
    }
}

/// The conditions on the parameters for every number of distinct real
/// roots, `None` for the zero polynomial (`real_root_classification`).
pub fn real_root_classification(p: &Basic, x: &Basic, max_degree: usize) -> PolyResult<Vec<(Basic, Option<i64>)>> {
    let cs: Vec<Basic> = coefficient_list(p, x)?;
    if cs.len() - 1 > max_degree && cs.iter().any(|c| !c.free_symbols().is_empty()) {
        return Err(PolyError::NotImplemented(format!(
            "the degree {} exceeds max_degree = {}: the number of sign patterns is 3**(n - 1)",
            cs.len() - 1,
            max_degree
        )));
    }
    let mut by_count: Vec<(Option<i64>, Vec<Basic>)> = Vec::new();
    let mut vanishing: Vec<Basic> = Vec::new();
    let mut remaining: Vec<Basic> = cs;
    loop {
        let leading: Basic = remaining[0].clone();
        let degree: usize = remaining.len() - 1;
        let numeric: bool = is_number(&leading);
        if numeric && leading == int(0) && degree > 0 {
            remaining.remove(0);
            continue;
        }
        let mut here: Vec<Basic> = vanishing.clone();
        if !numeric {
            here.push(Basic::new(
                Head::Or,
                vec![relation(RelOp::Gt, &leading), relation(RelOp::Lt, &leading)],
            ));
        }
        if degree == 0 {
            if numeric {
                push(
                    &mut by_count,
                    if leading == int(0) { None } else { Some(0) },
                    all_or_true(here),
                );
            } else {
                push(&mut by_count, Some(0), conjunction(here));
                let mut v: Vec<Basic> = vanishing.clone();
                v.push(relation(RelOp::Eq, &leading));
                push(&mut by_count, None, conjunction(v));
            }
            break;
        }
        let ms: Vec<Basic> = minors(&remaining)?;
        let patterns: Vec<Vec<i64>> = if ms.iter().all(is_number) {
            let actual: Vec<i64> = ms.iter().map(sign_of).collect();
            push(
                &mut by_count,
                Some(signature_from_signs(&actual)),
                all_or_true(here.clone()),
            );
            Vec::new()
        } else {
            let mut patterns: Vec<Vec<i64>> = vec![Vec::new()];
            for _ in 0..degree {
                let mut next: Vec<Vec<i64>> = Vec::new();
                for p in &patterns {
                    for s in [1, 0, -1] {
                        let mut q: Vec<i64> = p.clone();
                        q.push(s);
                        next.push(q);
                    }
                }
                patterns = next;
            }
            patterns
        };
        for signs in &patterns {
            let mut conditions: Vec<Basic> = here.clone();
            let mut feasible: bool = true;
            for (minor, sign) in ms.iter().zip(signs.iter()) {
                match condition(minor, *sign) {
                    None => {
                        feasible = false;
                        break;
                    }
                    Some(c) => {
                        if c != boolean(true) {
                            conditions.push(c);
                        }
                    }
                }
            }
            if !feasible {
                continue;
            }
            push(
                &mut by_count,
                Some(signature_from_signs(signs)),
                all_or_true(conditions),
            );
        }
        if numeric {
            break;
        }
        vanishing.push(relation(RelOp::Eq, &leading));
        remaining.remove(0);
    }
    by_count.sort_by_key(|(k, _)| k.unwrap_or(-1));
    let mut results: Vec<(Basic, Option<i64>)> = Vec::new();
    for (key, alternatives) in by_count {
        let formula: Basic = if alternatives.len() == 1 {
            alternatives[0].clone()
        } else {
            Basic::new(Head::Or, alternatives)
        };
        let s: Basic = simplified(&formula)?;
        if s != boolean(false) {
            results.push((s, key));
        }
    }
    Ok(results)
}
