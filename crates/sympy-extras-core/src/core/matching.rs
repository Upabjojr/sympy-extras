//! Pattern matching, ported from `sympy_extras.core.matching` (the
//! algorithm behind `Basic.match`): the bindings of the `Wild` symbols of a
//! pattern for which it equals an expression. The search backtracks: a
//! wild-free pattern must equal the expression; a wild binds when its
//! `exclude` and `properties` allow it; sums and products are matched as
//! multisets, the lone wilds absorbing what is left (the identity when
//! nothing is, unless `exact`); a power matches a non-power as a first
//! power; a sum or product with a single lone wild matches by solving.
//! Structural matches are searched before solved ones. Python's
//! generators are callbacks returning whether to stop. `WildFunction` is
//! not ported.

#![allow(clippy::type_complexity)]

use crate::core::add::{add, sub};
use crate::core::basic::{Basic, Head};
use crate::core::expr::{as_base_exp, as_coeff_mul, factors_of, terms_of};
use crate::core::mul::{div, mul, neg};
use crate::core::power::pow;
use crate::core::sorting::sort_key;
use crate::core::Wild;
use crate::error::PolyResult;

/// The bindings, wild to value, in the order they were made.
pub type Bindings = Vec<(Basic, Basic)>;

/// A continuation: `true` stops the search.
type K<'a> = &'a mut dyn FnMut(Bindings) -> PolyResult<bool>;
type KRest<'a> = &'a mut dyn FnMut(Bindings, Vec<Basic>) -> PolyResult<bool>;

fn bound<'a>(b: &'a Bindings, w: &Basic) -> Option<&'a Basic> {
    b.iter().find(|(k, _)| k == w).map(|(_, v)| v)
}

fn with(b: &Bindings, w: &Basic, value: &Basic) -> Bindings {
    let mut out: Bindings = b.clone();
    out.push((w.clone(), value.clone()));
    out
}

fn is_wild(e: &Basic) -> bool {
    matches!(e.head(), Head::Wild(_))
}

fn accepts(w: &Basic, value: &Basic) -> bool {
    Wild(w.clone()).accepts(value)
}

fn has_wild(pattern: &Basic) -> bool {
    is_wild(pattern) || pattern.args().iter().any(has_wild)
}

fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b) && !crate::sets::sets::is_set(b) && !matches!(b.head(), Head::Tuple)
}

fn is_one_number(e: &Basic) -> bool {
    e.is_number_class() && *e == Basic::one()
}

/// The bindings of the wilds of `pattern` for which it equals `expr`
/// (extending `bindings`), `None` when there are none (`match`).
pub fn match_pattern(
    expr: &Basic,
    pattern: &Basic,
    bindings: &[(Basic, Basic)],
    exact: bool,
) -> PolyResult<Option<Bindings>> {
    let start: Bindings = bindings.to_vec();
    for solve in [false, true] {
        let mut result: Option<Bindings> = None;
        matches(expr, pattern, &start, exact, solve, &mut |found| {
            result = Some(found);
            Ok(true)
        })?;
        if result.is_some() {
            return Ok(result);
        }
    }
    Ok(None)
}

fn matches(expr: &Basic, pattern: &Basic, b: &Bindings, exact: bool, solve: bool, k: K) -> PolyResult<bool> {
    if is_wild(pattern) {
        if let Some(value) = bound(b, pattern) {
            if value == expr {
                return k(b.clone());
            }
        } else if accepts(pattern, expr) {
            return k(with(b, pattern, expr));
        }
        return Ok(false);
    }
    if !has_wild(pattern) {
        if expr == pattern {
            return k(b.clone());
        }
        return Ok(false);
    }
    if is_expr(expr) {
        match pattern.head() {
            Head::Add => return match_assoc(expr, pattern, b, exact, solve, true, k),
            Head::Mul => return match_assoc(expr, pattern, b, exact, solve, false, k),
            Head::Pow => return match_pow(expr, pattern, b, exact, solve, k),
            _ => {}
        }
    }
    if !expr.head().same_head(pattern.head()) || expr.args().len() != pattern.args().len() {
        return Ok(false);
    }
    match_sequence(expr.args(), pattern.args(), b, exact, solve, k)
}

fn match_sequence(
    exprs: &[Basic],
    patterns: &[Basic],
    b: &Bindings,
    exact: bool,
    solve: bool,
    k: K,
) -> PolyResult<bool> {
    if patterns.is_empty() {
        return k(b.clone());
    }
    matches(&exprs[0], &patterns[0], b, exact, solve, &mut |found| {
        match_sequence(&exprs[1..], &patterns[1..], &found, exact, solve, k)
    })
}

fn combine(parts: &[Basic], is_add: bool) -> PolyResult<Basic> {
    if is_add {
        add(parts)
    } else {
        mul(parts)
    }
}

/// Matching of a sum or product pattern against `expr` (`_match_assoc`).
fn match_assoc(
    expr: &Basic,
    pattern: &Basic,
    b: &Bindings,
    exact: bool,
    solve: bool,
    is_add: bool,
    k: K,
) -> PolyResult<bool> {
    if pattern.args().iter().any(|p| is_wild(p) && bound(b, p).is_some()) {
        // the wilds bound already are replaced by their values and the
        // pattern evaluated
        let parts: Vec<Basic> = pattern
            .args()
            .iter()
            .map(|p| {
                if is_wild(p) {
                    bound(b, p).cloned().unwrap_or_else(|| p.clone())
                } else {
                    p.clone()
                }
            })
            .collect();
        let substituted: Basic = combine(&parts, is_add)?;
        return matches(expr, &substituted, b, exact, solve, k);
    }
    if !is_add {
        // PORTED FROM SYMPY: sympy/core/mul.py:Mul._matches_simple: a number
        // times a single factor matches that factor against expr divided by
        // the number
        let (coeff, rest_factors) = as_coeff_mul(pattern);
        let single: Vec<Basic> = factors_of(&rest_factors);
        if coeff.is_number_class() && coeff != Basic::one() && single.len() == 1 {
            let mut first: Option<Bindings> = None;
            matches(&div(expr, &coeff)?, &single[0], b, exact, solve, &mut |found| {
                first = Some(found);
                Ok(true)
            })?;
            if let Some(found) = first {
                // the generator yields this binding only
                return k(found);
            }
        }
    }
    let subject: Vec<Basic> = if is_add { terms_of(expr) } else { factors_of(expr) };
    let identity: Basic = if is_add { Basic::zero() } else { Basic::one() };
    let mut lone: Vec<Basic> = Vec::new();
    let mut others: Vec<Basic> = Vec::new();
    for p in pattern.args() {
        if is_wild(p) {
            match bound(b, p) {
                Some(v) => others.push(v.clone()),
                None => lone.push(p.clone()),
            }
        } else {
            others.push(p.clone());
        }
    }
    let stopped: bool = match_multiset(
        &subject,
        &others,
        b,
        exact,
        solve,
        &identity,
        &mut |found, remaining| {
            let unbound: Vec<Basic> = lone.iter().filter(|w| bound(&found, w).is_none()).cloned().collect();
            if unbound.is_empty() {
                if remaining.is_empty() {
                    return k(found);
                }
                return Ok(false);
            }
            if exact && remaining.is_empty() {
                return Ok(false);
            }
            let mut rest: Vec<Basic> = remaining;
            rest.sort_by_key(sort_key);
            let mut assignments: Vec<(Basic, Basic)> = Vec::new();
            for w in unbound[1..].iter().rev() {
                let value: Basic = rest.pop().unwrap_or_else(|| identity.clone());
                assignments.push((w.clone(), value));
            }
            assignments.push((unbound[0].clone(), combine(&rest, is_add)?));
            let mut result: Bindings = found.clone();
            for (w, value) in assignments {
                if let Some(existing) = bound(&result, &w) {
                    if *existing != value {
                        return Ok(false);
                    }
                }
                if !accepts(&w, &value) || (exact && value == identity) {
                    return Ok(false);
                }
                match result.iter_mut().find(|(key, _)| *key == w) {
                    Some(entry) => entry.1 = value,
                    None => result.push((w, value)),
                }
            }
            k(result)
        },
    )?;
    if stopped || !solve {
        return Ok(stopped);
    }
    // solving: w + rest = expr or w*rest = expr with a single lone wild and
    // wild-free other terms, when the subject is not of the other kind and
    // mentions every symbol of the rest
    let unbound: Vec<Basic> = lone.iter().filter(|w| bound(b, w).is_none()).cloned().collect();
    if unbound.len() != 1 || others.iter().any(has_wild) {
        return Ok(false);
    }
    if (is_add && expr.is_mul()) || (!is_add && expr.is_add()) {
        return Ok(false);
    }
    let rest_value: Basic = combine(&others, is_add)?;
    let expr_free = expr.free_symbols();
    if !(crate::core::assumptions::is_number(expr) || rest_value.free_symbols().iter().all(|s| expr_free.contains(s))) {
        return Ok(false);
    }
    let value: Basic = if is_add {
        sub(expr, &rest_value)?
    } else {
        div(expr, &rest_value)?
    };
    if (expr.is_add() || expr.is_mul()) && !no_more_complex(&value, expr)? {
        return Ok(false);
    }
    if accepts(&unbound[0], &value) && !(exact && value == identity) {
        return k(with(b, &unbound[0], &value));
    }
    Ok(false)
}

/// SymPy's test on a solved binding: the value (its sign dropped) must not
/// have more operations than the subject (`_no_more_complex`).
fn no_more_complex(value: &Basic, expr: &Basic) -> PolyResult<bool> {
    use crate::core::function::count_ops;
    let check: Basic = if crate::core::printing::could_extract_minus_sign(value) {
        neg(value)?
    } else {
        value.clone()
    };
    Ok(count_ops(&check)? <= count_ops(expr)?)
}

/// The ways of matching each pattern term against a distinct subject term
/// (or against the identity, for a term with wilds), with the bindings and
/// the subject terms left over (`_match_multiset`).
fn match_multiset(
    subject: &[Basic],
    patterns: &[Basic],
    b: &Bindings,
    exact: bool,
    solve: bool,
    identity: &Basic,
    k: KRest,
) -> PolyResult<bool> {
    let Some((p, rest)) = patterns.split_last() else {
        return k(b.clone(), subject.to_vec());
    };
    // the pattern terms are tried from the last one, the subject terms of
    // the class of the pattern term first
    let mut order: Vec<usize> = (0..subject.len()).collect();
    order.sort_by_key(|i| subject[*i].class_name() != p.class_name());
    for i in order {
        let mut remaining: Vec<Basic> = subject.to_vec();
        remaining.remove(i);
        let stopped: bool = matches(&subject[i], p, b, exact, solve, &mut |found| {
            match_multiset(&remaining, rest, &found, exact, solve, identity, k)
        })?;
        if stopped {
            return Ok(true);
        }
    }
    if has_wild(p) && !exact {
        return matches(identity, p, b, exact, solve, &mut |found| {
            match_multiset(subject, rest, &found, exact, solve, identity, k)
        });
    }
    Ok(false)
}

fn match_pow(expr: &Basic, pattern: &Basic, b: &Bindings, exact: bool, solve: bool, k: K) -> PolyResult<bool> {
    let (pbase, pexp) = (&pattern.args()[0], &pattern.args()[1]);
    if expr.is_pow() {
        if match_sequence(expr.args(), pattern.args(), b, exact, solve, k)? {
            return Ok(true);
        }
        if !solve {
            return Ok(false);
        }
        // x**4 against a**2: a = x**2
        let (ebase, eexp) = (&expr.args()[0], &expr.args()[1]);
        if let (Some(e1), Some(e2)) = (eexp.as_rational(), pexp.as_rational()) {
            if !has_wild(pexp) && *e2 != crate::numbers::Rat::ZERO {
                let ratio = e1 / e2;
                if ratio.is_int() && ratio != crate::numbers::Rat::ONE {
                    let powered: Basic = pow(ebase, &Basic::from_rational(ratio))?;
                    return matches(&powered, pbase, b, exact, solve, k);
                }
            }
        }
        return Ok(false);
    }
    if is_one_number(expr) {
        // the identity of a product as the zeroth power of the base
        if matches(&Basic::zero(), pexp, b, exact, solve, k)? {
            return Ok(true);
        }
    }
    let (base, exponent) = as_base_exp(expr)?;
    if !is_one_number(&exponent) && match_sequence(&[base, exponent], pattern.args(), b, exact, solve, k)? {
        return Ok(true);
    }
    if match_sequence(&[expr.clone(), Basic::one()], pattern.args(), b, exact, solve, k)? {
        return Ok(true);
    }
    if solve && !has_wild(pexp) {
        if let Some(e) = pexp.as_rational() {
            if *e != crate::numbers::Rat::ZERO {
                let inverse: Basic = Basic::from_rational(crate::numbers::Rat::ONE / e);
                return matches(&pow(expr, &inverse)?, pbase, b, exact, solve, k);
            }
        }
    }
    Ok(false)
}
