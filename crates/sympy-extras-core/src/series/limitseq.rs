//! Port of `sympy_extras/series/limitseq.py` (SymPy's `limit_seq`): the
//! limit of a sequence at infinity by the Stolz–Cesàro theorem
//! (`difference_delta`, `dominant`), separately on the odd and the even
//! indices for an oscillating term.

#![allow(clippy::mutable_key_type)]

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn has_sum(e: &Basic) -> bool {
    e.preorder().iter().any(|n| matches!(n.head(), Head::Sum))
}
fn xreplace(e: &Basic, old: &Basic, new: &Basic) -> R {
    let rule: std::collections::HashMap<Basic, Basic> = [(old.clone(), new.clone())].into_iter().collect();
    crate::core::function::xreplace_evaluated(e, &rule)
}

/// The forward difference `expr(n + step) - expr(n)` with the products,
/// powers and sums simplified (`difference_delta`).
pub fn difference_delta(expr: &Basic, n: &Basic, step: &Basic) -> R {
    if !crate::core::assumptions::is_number(step) || is(step, Predicate::finite) == Some(false) {
        return Err(PolyError::Value("Step should be a finite number.".to_string()));
    }
    if let Some(found) = difference_delta_rule(expr, n, step)? {
        if found != Basic::zero() {
            return Ok(found);
        }
    }
    s2(&subs(expr, n, &a2(n, step)?)?, expr)
}

fn difference_delta_rule(e: &Basic, n: &Basic, h: &Basic) -> OptR {
    if e.is_add() {
        let parts: Vec<Basic> = e
            .args()
            .iter()
            .map(|a| difference_delta(a, n, h))
            .collect::<PolyResult<_>>()?;
        return Ok(Some(crate::core::add::add(&parts)?));
    }
    if e.is_mul() {
        let first: &Basic = &e.args()[0];
        let rest: Basic = crate::core::mul::mul(&e.args()[1..])?;
        let shifted: Basic = subs(first, n, &a2(n, h)?)?;
        let left: Basic = m2(&shifted, &difference_delta(&rest, n, h)?)?;
        let right: Basic = m2(&difference_delta(first, n, h)?, &rest)?;
        return Ok(Some(a2(&left, &right)?));
    }
    if e.is_pow() {
        let (b, x) = (&e.args()[0], &e.args()[1]);
        if x.has(n) && !b.has(n) {
            let shifted: Basic = subs(x, n, &a2(n, h)?)?;
            let ratio: Basic = s2(&pw(b, &s2(&shifted, x)?)?, &Basic::one())?;
            return Ok(Some(m2(&ratio, e)?));
        }
        return Ok(None);
    }
    if matches!(e.head(), Head::Sum) {
        use crate::concrete::expr_with_limits::{limits, sum_};
        let lims = limits(e);
        let (k, _, upper) = lims[lims.len() - 1].clone();
        let new_upper: Basic = subs(&upper, n, &a2(n, h)?)?;
        let tuple = |l: &(Basic, Basic, Basic)| Basic::new(Head::Tuple, vec![l.0.clone(), l.1.clone(), l.2.clone()]);
        let f: Basic = if lims.len() == 1 {
            e.args()[0].clone()
        } else {
            sum_(
                &e.args()[0],
                &lims[..lims.len() - 1].iter().map(tuple).collect::<Vec<_>>(),
            )?
        };
        let last: Basic = Basic::new(Head::Tuple, vec![k, a2(&upper, &Basic::one())?, new_upper]);
        return Ok(Some(crate::core::function::doit(&sum_(&f, &[last])?)?));
    }
    Ok(None)
}

/// The dominant term of a sum of terms, `None` when two are comparable
/// (`dominant`).
pub fn dominant(expr: &Basic, n: &Basic) -> OptR {
    let mut hints = crate::core::function::Hints::DEFAULT;
    hints.func = true;
    let expanded: Basic = crate::core::function::expand_with(expr, true, hints)?;
    let terms: Vec<Basic> = crate::core::expr::terms_of(&expanded);
    let mut term0: Basic = terms[terms.len() - 1].clone();
    let mut comparable: Vec<Basic> = vec![term0.clone()];
    for t in &terms[..terms.len() - 1] {
        let r: Basic = d2(&term0, t)?;
        let mut e: Basic = crate::simplify::gammasimp::gammasimp(&r)?;
        if e == r {
            e = crate::polys::polytools::factor(&r, &[], false, true)?;
        }
        let Some(lim) = limit_seq(&e, Some(n), 5)? else {
            return Ok(None);
        };
        if is(&lim, Predicate::zero) == Some(true) {
            term0 = t.clone();
            comparable = vec![term0.clone()];
        } else if !matches!(lim.head(), Head::Infinity | Head::NegativeInfinity) {
            comparable.push(t.clone());
        }
    }
    if comparable.len() > 1 {
        return Ok(None);
    }
    Ok(Some(term0))
}

fn limit_inf(expr: &Basic, n: &Basic) -> OptR {
    match crate::core::limits::limit(expr, n, &Basic::oo(), "+") {
        Ok(v) => Ok(Some(v)),
        Err(PolyError::Pole(_)) => Ok(None),
        Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => Ok(None),
        Err(e) => Err(e),
    }
}

fn limit_seq_trials(expr: &Basic, n: &Basic, trials: usize) -> OptR {
    let mut expr: Basic = expr.clone();
    for _ in 0..trials {
        crate::timeout::check()?;
        if !has_sum(&expr) {
            if let Some(result) = limit_inf(&expr, n)? {
                return Ok(Some(result));
            }
        }
        let (num, den) = crate::core::expr::as_numer_denom(&expr)?;
        if !den.has(n) || !num.has(n) {
            return limit_inf(&crate::core::function::doit(&expr)?, n);
        }
        let one: Basic = Basic::one();
        let num: Basic = difference_delta(&crate::core::function::expand_default(&num)?, n, &one)?;
        let den: Basic = difference_delta(&crate::core::function::expand_default(&den)?, n, &one)?;
        expr = crate::simplify::gammasimp::gammasimp(&d2(&num, &den)?)?;
        if !has_sum(&expr) {
            if let Some(result) = limit_inf(&expr, n)? {
                return Ok(Some(result));
            }
        }
        let (num, den) = crate::core::expr::as_numer_denom(&expr)?;
        let Some(dominant_num) = dominant(&num, n)? else {
            return Ok(None);
        };
        let Some(dominant_den) = dominant(&den, n)? else {
            return Ok(None);
        };
        expr = crate::simplify::gammasimp::gammasimp(&d2(&dominant_num, &dominant_den)?)?;
    }
    Ok(None)
}

fn fibonacci_as_golden_ratio(e: &Basic) -> R {
    let mut rules: std::collections::HashMap<Basic, Basic> = std::collections::HashMap::new();
    for node in e.preorder() {
        if matches!(node.head(), Head::Function(Function::Fibonacci)) && node.args().len() == 1 {
            let k: &Basic = &node.args()[0];
            let phi: Basic = Basic::atom(Head::GoldenRatio);
            let first: Basic = pw(&phi, k)?;
            let second: Basic = d2(&Basic::one(), &pw(&crate::core::mul::neg(&phi)?, k)?)?;
            let value: Basic = d2(&s2(&first, &second)?, &s2(&m2(&Basic::int(2), &phi)?, &Basic::one())?)?;
            rules.insert(node.clone(), value);
        }
    }
    if rules.is_empty() {
        return Ok(e.clone());
    }
    crate::core::function::xreplace_evaluated(e, &rules)
}

/// The limit of the sequence `expr` at `n -> oo`, `None` when it is not
/// found (`limit_seq`).
pub fn limit_seq(expr: &Basic, n: Option<&Basic>, trials: usize) -> OptR {
    let free: Vec<Basic> = expr
        .free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_)))
        .collect();
    let n: Basic = match n {
        None => {
            if free.len() == 1 {
                free[0].clone()
            } else if free.is_empty() {
                return Ok(Some(expr.clone()));
            } else {
                return Err(PolyError::Value(
                    "Expression has more than one variable. Please specify a variable.".to_string(),
                ));
            }
        }
        Some(n) => {
            if !expr.free_symbols().contains(n) {
                return Ok(Some(expr.clone()));
            }
            n.clone()
        }
    };
    let e: Basic = fibonacci_as_golden_ratio(expr)?;
    let e: Basic = crate::core::rewriting::rewrite(
        &e,
        &[Function::Factorial, Function::Subfactorial],
        "gamma",
        &crate::core::rewriting::RewriteHints::default(),
    )?;
    let dummy = |p: Predicate| crate::core::symbol::Dummy::new("n", &[(p, true), (Predicate::positive, true)]).0;
    let n_: Basic = dummy(Predicate::integer);
    let n1: Basic = dummy(Predicate::odd);
    let n2: Basic = dummy(Predicate::even);
    let oscillating: bool = e
        .preorder()
        .iter()
        .any(|p| p.is_pow() && is(&p.args()[0], Predicate::negative) == Some(true) && p.args()[1].has(&n))
        || e.preorder()
            .iter()
            .any(|p| matches!(p.head(), Head::Function(Function::Cos | Function::Sin)));
    let first: Option<Basic> = if oscillating {
        let first = limit_seq_trials(&xreplace(&e, &n, &n1)?, &n1, trials)?;
        if let Some(f) = &first {
            let second = limit_seq_trials(&xreplace(&e, &n, &n2)?, &n2, trials)?;
            if second.as_ref() != Some(f) {
                if let Some(s) = &second {
                    if crate::core::expr::is_comparable(f) && crate::core::expr::is_comparable(s) {
                        let lo: Basic = crate::functions::elementary::miscellaneous::min_(&[f.clone(), s.clone()])?;
                        let hi: Basic = crate::functions::elementary::miscellaneous::max_(&[f.clone(), s.clone()])?;
                        return Ok(Some(crate::core::accumbounds::accum_bounds(&lo, &hi)?));
                    }
                }
                return Ok(None);
            }
        }
        first
    } else {
        limit_seq_trials(&xreplace(&e, &n, &n_)?, &n_, trials)?
    };
    if first.is_some() {
        return Ok(first);
    }
    if e.is_add() {
        let mut found: Vec<Basic> = Vec::new();
        for term in e.args() {
            let Some(result) = limit_seq(term, Some(&n), trials)? else {
                return Ok(None);
            };
            found.push(result);
        }
        return Ok(Some(crate::core::add::add(&found)?));
    }
    if !has_sum(&e) {
        let absolute: Basic = crate::functions::elementary::complexes::abs_(&xreplace(&e, &n, &n_)?)?;
        if let Some(lim) = limit_seq_trials(&absolute, &n_, trials)? {
            if is(&lim, Predicate::zero) == Some(true) {
                return Ok(Some(Basic::zero()));
            }
        }
    }
    Ok(None)
}
