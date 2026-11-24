//! Port of `sympy_extras/concrete/summations.py`: the evaluation of
//! `Sum(f, (i, a, b))` (`eval_sum`, `sum_doit`, `summation`): direct sums,
//! telescoping, powers by Bernoulli polynomials and harmonic numbers,
//! geometric sums, Gosper's algorithm with the special parameter values,
//! hypergeometric sums through `hyperexpand`, sums of rational functions
//! by residues, the Hurwitz zeta function.

#![allow(clippy::type_complexity, clippy::mutable_key_type)]

use crate::concrete::delta::Limit;
use crate::core::assumptions::{facts_of, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn is_true(b: &Basic) -> bool {
    *b == crate::core::logic::true_()
}
fn true_() -> Basic {
    crate::core::logic::true_()
}
fn is_nan(e: &Basic) -> bool {
    matches!(e.head(), Head::NaN)
}
fn is_oo(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity)
}
fn is_neg_oo(e: &Basic) -> bool {
    matches!(e.head(), Head::NegativeInfinity)
}
fn as_int(e: &Basic) -> Option<i64> {
    match e.head() {
        Head::Rational {
            integer_class: true, ..
        } => i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok(),
        _ => None,
    }
}
fn tuple(l: &Limit) -> Basic {
    Basic::new(Head::Tuple, vec![l.0.clone(), l.1.clone(), l.2.clone()])
}
fn has_head(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn is_piecewise(e: &Basic) -> bool {
    matches!(e.head(), Head::Piecewise)
}
fn pairs(pw_: &Basic) -> Vec<(Basic, Basic)> {
    pw_.args()
        .iter()
        .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
        .collect()
}
fn piecewise(p: &[(Basic, Basic)]) -> R {
    crate::functions::elementary::piecewise::piecewise(p)
}
fn wild(name: &str, exclude: &[Basic]) -> Basic {
    crate::core::symbol::Wild::new(name, exclude.to_vec(), Vec::new(), &[]).0
}
fn matched(e: &Basic, pattern: &Basic) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    crate::core::matching::match_pattern(e, pattern, &[], false)
}
fn get(found: &[(Basic, Basic)], w: &Basic) -> Option<Basic> {
    found.iter().find(|(k, _)| k == w).map(|(_, v)| v.clone())
}
fn is_not_ported(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if m.starts_with(crate::error::NOT_PORTED_PREFIX))
}
fn swallowed(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(_)) && !is_not_ported(e)
}
fn two_terms(e: &Basic) -> PolyResult<(Basic, Basic)> {
    let args: &[Basic] = e.args();
    let rest: Basic = if e.is_add() {
        add_all(&args[1..])?
    } else {
        mul_all(&args[1..])?
    };
    Ok((args[0].clone(), rest))
}
fn truthy(e: &Option<Basic>) -> bool {
    // Python's `if s:` of an expression: not None and not zero
    matches!(e, Some(v) if !(v.is_number_node() && *v == int(0)))
}
fn a_gt_b(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(crate::core::expr::is_comparable(a)
        && crate::core::expr::is_comparable(b)
        && is_true(&crate::core::relational::gt(a, b)?))
}
fn apart(e: &Basic, i: &Basic) -> R {
    match crate::polys::partfrac::apart(e, Some(i), false) {
        Ok(r) => Ok(r),
        Err(PolyError::Polys(..) | PolyError::Value(_)) => Ok(e.clone()),
        Err(err) => Err(err),
    }
}

fn shift_candidates(e: &Basic, m: &Basic) -> PolyResult<Vec<Basic>> {
    match crate::polys::rationalsolve::solveset_rational(e, m)? {
        None => crate::concrete::delta::solve(e, m),
        Some((solutions, excluded)) => Ok(solutions.into_iter().filter(|s| !excluded.contains(s)).collect()),
    }
}

fn telescopic_direct(l: &Basic, r: &Basic, n: i64, limits: &Limit) -> R {
    let (i, a, b) = limits;
    let mut terms: Vec<Basic> = Vec::new();
    for m in 0..n {
        terms.push(a2(&subs(l, i, &a2(a, &int(m))?)?, &subs(r, i, &s2(b, &int(m))?)?)?);
    }
    add_all(&terms)
}

/// The sum of `L + R` when `R` is `-L` shifted by an integer (`telescopic`).
pub fn telescopic(l: &Basic, r: &Basic, limits: &Limit) -> OptR {
    let (i, _, _) = limits;
    if l.is_add() || r.is_add() {
        return Ok(None);
    }
    let k: Basic = wild("k", &[]);
    let mut s: Option<i64> = None;
    if let Some(sol) = matched(&neg(r)?, &subs(l, i, &a2(i, &k)?)?)? {
        if let Some(v) = get(&sol, &k) {
            if let Some(n) = as_int(&v) {
                if a2(&subs(l, i, &a2(i, &v)?)?, r)? == int(0) {
                    s = Some(n);
                }
            }
        }
    }
    if s.is_none() {
        let m: Basic = crate::core::symbol::Dummy::new("m", &[]).0;
        let found = match shift_candidates(&a2(&subs(l, i, &a2(i, &m)?)?, r)?, &m) {
            Ok(f) => f,
            Err(e) if swallowed(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut good: Vec<i64> = Vec::new();
        for si in found {
            if let Some(n) = as_int(&si) {
                let check: Basic = crate::core::function::expand_default(&a2(&subs(l, i, &a2(i, &si)?)?, r)?)?;
                if yes(&check, Predicate::zero) {
                    good.push(n);
                }
            }
        }
        if good.len() != 1 {
            return Ok(None);
        }
        s = Some(good[0]);
    }
    let s: i64 = s.expect("a shift");
    if s < 0 {
        return Ok(Some(telescopic_direct(r, l, -s, limits)?));
    }
    if s > 0 {
        return Ok(Some(telescopic_direct(l, r, s, limits)?));
    }
    Ok(None)
}

/// The sum over one limit, `None` when it is not found (`eval_sum`).
pub fn eval_sum(f: &Basic, limits: &Limit) -> OptR {
    let (i, a, b) = limits;
    if yes(f, Predicate::zero) {
        return Ok(Some(int(0)));
    }
    if !f.free_symbols().contains(i) {
        return Ok(Some(m2(f, &a2(&s2(b, a)?, &int(1))?)?));
    }
    if a == b {
        return Ok(Some(subs(f, i, a)?));
    }
    if a_gt_b(a, b)? {
        // Karr's convention: minus the sum over (i, b + 1, a - 1)
        // (sympy/sympy#31275, V46)
        if s2(a, b)? == int(1) {
            return Ok(Some(int(0)));
        }
        return match eval_sum(f, &(i.clone(), a2(b, &int(1))?, s2(a, &int(1))?))? {
            Some(r) => Ok(Some(neg(&r)?)),
            None => Ok(None),
        };
    }
    if is_piecewise(f) && !pairs(f).iter().any(|(_, c)| c.free_symbols().contains(i)) {
        let mut newargs: Vec<(Basic, Basic)> = Vec::new();
        for (e, c) in pairs(f) {
            let Some(newexpr) = eval_sum(&e, limits)? else {
                return Ok(None);
            };
            newargs.push((newexpr, c));
        }
        return Ok(Some(piecewise(&newargs)?));
    }
    let mut f: Basic = f.clone();
    if has_head(&f, &|e| matches!(e.head(), Head::Function(Function::KroneckerDelta))) {
        f = crate::core::function::replace_where(&f, &|e| matches!(e.head(), Head::Sum), &|e| {
            crate::polys::polytools::factor(e, &[], false, true)
        })?;
        if crate::concrete::delta::has_simple_delta(&f, i)? {
            return Ok(Some(crate::concrete::delta::deltasummation(&f, limits, false)?));
        }
    }
    let dif: Basic = s2(b, a)?;
    let definite: Option<i64> = as_int(&dif);
    if definite.is_some_and(|d| d < 100) {
        return eval_sum_direct(&f, limits);
    }
    if is_piecewise(&f) {
        return Ok(None);
    }
    if let Some(value) = eval_sum_symbolic(&crate::core::function::expand_default(&f)?, limits)? {
        return Ok(Some(merge_point_cases(&value)?));
    }
    if definite.is_some() {
        return eval_sum_direct(&f, limits);
    }
    Ok(None)
}

/// The sum term by term over a definite range (`eval_sum_direct`).
pub fn eval_sum_direct(expr: &Basic, limits: &Limit) -> OptR {
    let (i, a, b) = limits;
    let dif: Basic = s2(b, a)?;
    if expr.is_mul() {
        let (without_i, with_i) = crate::core::expr::as_independent(expr, i, false)?;
        if without_i != int(1) {
            let s = eval_sum_direct(&with_i, limits)?;
            if truthy(&s) {
                let r: Basic = m2(&without_i, &s.expect("a sum"))?;
                if !is_nan(&r) {
                    return Ok(Some(r));
                }
            }
        } else {
            let (l, r) = two_terms(expr)?;
            if !l.has(i) {
                let s = eval_sum_direct(&r, limits)?;
                if truthy(&s) {
                    return Ok(Some(m2(&l, &s.expect("a sum"))?));
                }
            }
            if !r.has(i) {
                let s = eval_sum_direct(&l, limits)?;
                if truthy(&s) {
                    return Ok(Some(m2(&s.expect("a sum"), &r)?));
                }
            }
        }
    }
    let expr: Basic = apart(expr, i)?;
    if expr.is_add() {
        let (without_i, with_i) = crate::core::expr::as_independent(&expr, i, true)?;
        if without_i != int(0) {
            let s = eval_sum_direct(&with_i, limits)?;
            if truthy(&s) {
                let r: Basic = a2(&m2(&without_i, &a2(&dif, &int(1))?)?, &s.expect("a sum"))?;
                if !is_nan(&r) {
                    return Ok(Some(r));
                }
            }
        } else {
            let (l, r) = two_terms(&expr)?;
            let lsum = eval_sum_direct(&l, limits)?;
            let rsum = eval_sum_direct(&r, limits)?;
            if let (Some(ls), Some(rs)) = (lsum, rsum) {
                let total: Basic = a2(&ls, &rs)?;
                if !is_nan(&total) {
                    return Ok(Some(total));
                }
            }
        }
    }
    let n: i64 = as_int(&dif).ok_or_else(|| PolyError::Value("an integer range".to_string()))?;
    let terms: Vec<Basic> = (0..=n)
        .map(|j| subs(&expr, i, &a2(a, &int(j))?))
        .collect::<PolyResult<_>>()?;
    Ok(Some(add_all(&terms)?))
}

/// The sum by the symbolic methods (`eval_sum_symbolic`).
pub fn eval_sum_symbolic(f: &Basic, limits: &Limit) -> OptR {
    let f_orig: Basic = f.clone();
    let (i, a, b) = limits;
    if !f.has(i) {
        return Ok(Some(m2(f, &a2(&s2(b, a)?, &int(1))?)?));
    }
    if f.is_mul() {
        let (without_i, with_i) = crate::core::expr::as_independent(f, i, false)?;
        if without_i != int(1) {
            let s = eval_sum_symbolic(&with_i, limits)?;
            if truthy(&s) {
                let r: Basic = m2(&without_i, &s.expect("a sum"))?;
                if !is_nan(&r) {
                    return Ok(Some(r));
                }
            }
        } else {
            let (l, r) = two_terms(f)?;
            if !l.has(i) {
                let s = eval_sum_symbolic(&r, limits)?;
                if truthy(&s) {
                    return Ok(Some(m2(&l, &s.expect("a sum"))?));
                }
            }
            if !r.has(i) {
                let s = eval_sum_symbolic(&l, limits)?;
                if truthy(&s) {
                    return Ok(Some(m2(&s.expect("a sum"), &r)?));
                }
            }
        }
    }
    let f: Basic = apart(f, i)?;
    if f.is_add() {
        let (l, r) = two_terms(&f)?;
        let lrsum = telescopic(&l, &r, limits)?;
        if truthy(&lrsum) {
            return Ok(lrsum);
        }
        let (without_i, with_i) = crate::core::expr::as_independent(&f, i, true)?;
        if without_i != int(0) {
            let s = eval_sum_symbolic(&with_i, limits)?;
            if truthy(&s) {
                let r: Basic = a2(&m2(&without_i, &a2(&s2(b, a)?, &int(1))?)?, &s.expect("a sum"))?;
                if !is_nan(&r) {
                    return Ok(Some(r));
                }
            }
        } else {
            let lsum = eval_sum_symbolic(&l, limits)?;
            let rsum = eval_sum_symbolic(&r, limits)?;
            if let (Some(ls), Some(rs)) = (lsum, rsum) {
                let total: Basic = a2(&ls, &rs)?;
                if !is_nan(&total) {
                    return Ok(Some(total));
                }
            }
        }
    }
    let nw: Basic = wild("n", &[]);
    if let Some(result) = matched(&f, &pw(i, &nw)?)? {
        if let Some(n) = get(&result, &nw).as_ref().and_then(as_int) {
            if n >= 0 {
                if (is_oo(b) && !is_neg_oo(a)) || (is_neg_oo(a) && !is_oo(b)) {
                    return Ok(Some(Basic::oo()));
                }
                let bern = |x: &Basic| crate::functions::combinatorial::numbers::bernoulli(&int(n + 1), Some(x));
                let value: Basic = d2(&s2(&bern(&a2(b, &int(1))?)?, &bern(a)?)?, &int(n + 1))?;
                return Ok(Some(crate::core::function::expand_default(&value)?));
            }
            if as_int(a).is_some_and(|v| v >= 1) {
                let harmonic = |x: &Basic, m: Option<&Basic>| crate::functions::combinatorial::numbers::harmonic(x, m);
                let a1: Basic = s2(a, &int(1))?;
                if n == -1 {
                    return Ok(Some(s2(&harmonic(b, None)?, &harmonic(&a1, None)?)?));
                }
                let m: Basic = int(n.abs());
                return Ok(Some(s2(&harmonic(b, Some(&m))?, &harmonic(&a1, Some(&m))?)?));
            }
        }
    }
    let infinite_bound = |e: &Basic| has_head(e, &|n| matches!(n.head(), Head::Infinity | Head::NegativeInfinity));
    if !(infinite_bound(a) || infinite_bound(b)) {
        let (c1, c2, c3) = (
            wild("c1", std::slice::from_ref(i)),
            wild("c2", std::slice::from_ref(i)),
            wild("c3", std::slice::from_ref(i)),
        );
        let wexp: Basic = wild("wexp", &[]);
        if let Some(e) = matched(&crate::simplify::powsimp::powsimp_default(&f)?, &pw(&c1, &wexp)?)? {
            let exponent: Basic = get(&e, &wexp).unwrap_or_else(|| int(0));
            let mut rest: Vec<(Basic, Basic)> = e.iter().filter(|(k, _)| *k != wexp).cloned().collect();
            let linear: Basic = a2(&m2(&c2, i)?, &c3)?;
            if let Some(e_exp) = matched(&crate::core::function::expand_default(&exponent)?, &linear)? {
                for (k, v) in e_exp {
                    match rest.iter_mut().find(|(key, _)| *key == k) {
                        Some(entry) => entry.1 = v,
                        None => rest.push((k, v)),
                    }
                }
                let p: Basic = crate::core::subs::subs_pairs(&pw(&c1, &c3)?, &rest)?;
                let q: Basic = crate::core::subs::subs_pairs(&pw(&c1, &c2)?, &rest)?;
                let r: Basic = d2(
                    &m2(&p, &s2(&pw(&q, a)?, &pw(&q, &a2(b, &int(1))?)?)?)?,
                    &s2(&int(1), &q)?,
                )?;
                let l: Basic = m2(&p, &a2(&s2(b, a)?, &int(1))?)?;
                return Ok(Some(piecewise(&[
                    (l, crate::core::relational::eq(&q, &int(1))?),
                    (r, true_()),
                ])?));
            }
        }
        let certificate = crate::concrete::gosper::gosper_term(&f, i)?;
        let gr: Option<Basic> = match &certificate {
            Some(c) => crate::concrete::gosper::gosper_definite(&f, c, i, a, b)?,
            None => None,
        };
        let cases: Vec<(Basic, Basic)> = match (&certificate, &gr) {
            (Some(c), Some(g)) => certificate_cases(&f_orig, c, g, limits)?,
            _ => Vec::new(),
        };
        if let Some(g) = &gr {
            if g.is_mul() || g.is_add() {
                let bound: std::collections::BTreeSet<Basic> =
                    a.free_symbols().into_iter().chain(b.free_symbols()).collect();
                let non_limit: Vec<Basic> = g.free_symbols().into_iter().filter(|s| !bound.contains(s)).collect();
                let den: Basic =
                    crate::simplify::radsimp::fraction(&crate::polys::rationaltools::together(g, false, true)?)?.1;
                let den_free = den.free_symbols();
                let den_sym: Vec<Basic> =
                    crate::core::sorting::ordered(non_limit.into_iter().filter(|s| den_free.contains(s)).collect());
                let mut args: Vec<(Basic, Basic)> = Vec::new();
                for v in den_sym {
                    let sols = match crate::concrete::delta::solve(&den, &v) {
                        Ok(s) => s,
                        Err(e) if swallowed(&e) => continue,
                        Err(e) => return Err(e),
                    };
                    if let Some(first) = sols.first() {
                        let cond: Basic = crate::core::relational::eq(&v, first)?;
                        let inner: Basic =
                            crate::concrete::expr_with_limits::sum_(&subs(&f_orig, &v, first)?, &[tuple(limits)])?;
                        let value: Basic = if matches!(inner.head(), Head::Sum) {
                            sum_doit(&inner, true)?
                        } else {
                            inner
                        };
                        args.push((value, cond));
                    }
                    break;
                }
                args.push((g.clone(), true_()));
                let mut all: Vec<(Basic, Basic)> = cases.clone();
                all.extend(args);
                return Ok(Some(piecewise(&all)?));
            }
            if !is_nan(g) {
                if cases.is_empty() {
                    return Ok(Some(g.clone()));
                }
                let mut all: Vec<(Basic, Basic)> = cases.clone();
                all.push((g.clone(), true_()));
                return Ok(Some(piecewise(&all)?));
            }
        }
    }
    if let Some(h) = eval_sum_hyper(&f_orig, limits)? {
        return Ok(Some(h));
    }
    if let Some(rr) = eval_sum_residue(&f_orig, limits)? {
        return Ok(Some(rr));
    }
    let factored: Basic = crate::polys::polytools::factor(&f_orig, &[], false, true)?;
    if factored != f_orig {
        return eval_sum_symbolic(&factored, limits);
    }
    Ok(None)
}

fn allowed_value(v: &Basic, r: &Basic) -> bool {
    if yes(v, Predicate::integer) && is(r, Predicate::integer) == Some(false) {
        return false;
    }
    if yes(v, Predicate::real) && is(r, Predicate::real) == Some(false) {
        return false;
    }
    let refuted = [
        (Predicate::nonnegative, Predicate::negative),
        (Predicate::positive, Predicate::nonpositive),
        (Predicate::nonpositive, Predicate::positive),
        (Predicate::negative, Predicate::nonnegative),
    ];
    !refuted.iter().any(|(fact, test)| yes(v, *fact) && yes(r, *test))
}

fn certificate_cases(f: &Basic, certificate: &Basic, value: &Basic, limits: &Limit) -> PolyResult<Vec<(Basic, Basic)>> {
    let (i, a, b) = limits;
    let den: Basic =
        crate::simplify::radsimp::fraction(&crate::polys::rationaltools::together(certificate, false, true)?)?.1;
    if den.free_symbols().iter().all(|s| s == i) {
        return Ok(Vec::new());
    }
    let factors = match crate::polys::polytools::factor_list(&den, &[]) {
        Ok((_, fs)) => fs,
        Err(PolyError::Polys(..)) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut cases: Vec<(Basic, Basic)> = Vec::new();
    let mut seen: Vec<(Basic, Basic)> = Vec::new();
    for (fac, _) in factors {
        if fac.has(i) || fac.free_symbols().is_empty() {
            continue;
        }
        for v in crate::core::sorting::ordered(fac.free_symbols().into_iter().collect()) {
            let sols = match crate::concrete::delta::solve(&fac, &v) {
                Ok(s) => s,
                Err(e) if swallowed(&e) => continue,
                Err(e) => return Err(e),
            };
            for r in sols {
                if seen.contains(&(v.clone(), r.clone())) || !allowed_value(&v, &r) {
                    continue;
                }
                seen.push((v.clone(), r.clone()));
                let limit: Basic = Basic::new(Head::Tuple, vec![i.clone(), subs(a, &v, &r)?, subs(b, &v, &r)?]);
                let inner: Basic = crate::concrete::expr_with_limits::sum_(&subs(f, &v, &r)?, &[limit])?;
                let at: Basic = if matches!(inner.head(), Head::Sum) {
                    sum_doit(&inner, true)?
                } else {
                    inner
                };
                if matches!(at.head(), Head::Sum) || has_head(&at, &is_nan) {
                    continue;
                }
                let generic: Basic = subs(value, &v, &r)?;
                if !has_head(&generic, &is_nan) && crate::core::function::expand_default(&s2(&at, &generic)?)? == int(0)
                {
                    continue;
                }
                cases.push((at, crate::core::relational::eq(&v, &r)?));
            }
            break;
        }
    }
    Ok(cases)
}

fn merge_point_cases(value: &Basic) -> R {
    if is_piecewise(value) || !has_head(value, &is_piecewise) {
        return Ok(value.clone());
    }
    let mut points: Vec<(Basic, Basic)> = Vec::new();
    let mut pieces: Vec<Basic> = Vec::new();
    for n in value.preorder() {
        if is_piecewise(&n) && !pieces.contains(&n) {
            pieces.push(n);
        }
    }
    let mut generic_rule: Vec<(Basic, Basic)> = Vec::new();
    for p in &pieces {
        let ps: Vec<(Basic, Basic)> = pairs(p);
        if !is_true(&ps[ps.len() - 1].1) {
            return Ok(value.clone());
        }
        for (e, cond) in &ps[..ps.len() - 1] {
            if !matches!(cond.head(), Head::Relational(crate::core::basic::RelOp::Eq)) {
                return Ok(value.clone());
            }
            let (v, c) = (&cond.args()[0], &cond.args()[1]);
            if !v.is_symbol() || !crate::core::assumptions::is_number(c) || has_head(e, &is_piecewise) {
                return Ok(value.clone());
            }
            if !points.contains(&(v.clone(), c.clone())) {
                points.push((v.clone(), c.clone()));
            }
        }
        generic_rule.push((p.clone(), ps[ps.len() - 1].0.clone()));
    }
    let map: std::collections::HashMap<Basic, Basic> = generic_rule.into_iter().collect();
    let generic: Basic = crate::core::function::xreplace_evaluated(value, &map)?;
    if has_head(&generic, &is_piecewise) {
        return Ok(value.clone());
    }
    let mut cases: Vec<(Basic, Basic)> = Vec::new();
    for (v, c) in points {
        let at: Basic = subs(value, &v, &c)?;
        if has_head(&at, &is_piecewise) || has_head(&at, &is_nan) {
            return Ok(value.clone());
        }
        if crate::core::function::expand_default(&s2(&at, &subs(&generic, &v, &c)?)?)? != int(0) {
            cases.push((at, crate::core::relational::eq(&v, &c)?));
        }
    }
    cases.push((generic, true_()));
    piecewise(&cases)
}

fn coeff_mul(e: &Basic, x: &Basic) -> PolyResult<(Basic, Vec<Basic>)> {
    if !e.is_mul() {
        return Ok(if e.has(x) {
            (int(1), vec![e.clone()])
        } else {
            (e.clone(), Vec::new())
        });
    }
    let free: Vec<Basic> = e.args().iter().filter(|a| !a.has(x)).cloned().collect();
    Ok((mul_all(&free)?, e.args().iter().filter(|a| a.has(x)).cloned().collect()))
}

fn sum_hyper_at(f: &Basic, i: &Basic, a: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    if *a != int(0) {
        return sum_hyper_at(&subs(f, i, &a2(i, a)?)?, i, &int(0));
    }
    if subs(f, i, &int(0))? == int(0) {
        let positive: Basic =
            crate::core::symbol::Dummy::new("i", &[(Predicate::integer, true), (Predicate::positive, true)]).0;
        let simplified: Basic = crate::simplify::simplify::simplify(
            &subs(f, i, &positive)?,
            &crate::simplify::simplify::Options::default(),
        )?;
        if simplified == int(0) {
            return Ok(Some((int(0), true_())));
        }
        return sum_hyper_at(&subs(f, i, &a2(i, &int(1))?)?, i, &int(0));
    }
    let Some(hs) = crate::concrete::gosper::hypersimp(f, i)? else {
        return Ok(None);
    };
    let (numer, denom) = crate::simplify::radsimp::fraction(&crate::polys::polytools::factor(&hs, &[], false, true)?)?;
    let (top, topl) = coeff_mul(&numer, i)?;
    let (bot, botl) = coeff_mul(&denom, i)?;
    let mut ab: [Basic; 2] = [top, bot];
    let mut params: [Vec<Basic>; 2] = [Vec::new(), Vec::new()];
    for (k, facs) in [topl, botl].iter().enumerate() {
        for fac in facs {
            let (mut fac, mut multiplicity) = (fac.clone(), 1i64);
            if fac.is_pow() {
                let Some(m) = as_int(&fac.args()[1]) else {
                    return Ok(None);
                };
                multiplicity = m;
                fac = fac.args()[0].clone();
            }
            let p = match crate::concrete::gosper::UPoly::of(&fac, i) {
                Ok(p) => p,
                Err(PolyError::Value(_)) => return Ok(None),
                Err(e) => return Err(e),
            };
            if p.degree() != 1 {
                return Ok(None);
            }
            let (m, n) = (p.nth(1), p.nth(0));
            ab[k] = m2(&ab[k], &pw(&m, &int(multiplicity))?)?;
            let ratio: Basic = d2(&n, &m)?;
            for _ in 0..multiplicity.max(0) {
                params[k].push(ratio.clone());
            }
        }
    }
    let mut ap: Vec<Basic> = params[0].clone();
    ap.push(int(1));
    let bq: Vec<Basic> = params[1].clone();
    let x: Basic = d2(&ab[0], &ab[1])?;
    let h: Basic = crate::functions::special::hyper::hyper(&ap, &bq, &x)?;
    let statement: Basic = if matches!(h.head(), Head::Hyper) {
        crate::functions::special::hyper::convergence_statement(&h)?
    } else {
        true_()
    };
    let f: Basic = crate::simplify::gammasimp::combsimp_generic(f)?;
    use crate::simplify::hyperexpand::{hyperexpand, Rewrite};
    Ok(Some((
        m2(&subs(&f, i, &int(0))?, &hyperexpand(&h, false, Rewrite::Default, None)?)?,
        statement,
    )))
}

fn count_illegal(e: &Basic) -> usize {
    e.preorder()
        .iter()
        .filter(|n| {
            matches!(
                n.head(),
                Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity | Head::NaN
            )
        })
        .count()
}

fn is_hypergeometric(f: &Basic, i: &Basic) -> PolyResult<Option<bool>> {
    if has_head(f, &is_piecewise) {
        return Ok(None);
    }
    Ok(Some(crate::concrete::gosper::consecutive_ratio(f, i)?.is_some()))
}

/// The sum of a hypergeometric term through `hyperexpand` (`eval_sum_hyper`).
/// The cases of the sum of `f` from `a` to `b` at the values of a symbol of
/// `b` with `b = a - 1, a - 2, a - 3` where the closed form `closed` (valid
/// under `cond`) is shown not to have the value of the sum (it is not a
/// number there, or has another numerical value at sample values of the
/// symbols): 0 for `b = a - 1` and `-(f(b + 1) + ... + f(a - 1))` below
/// (Karr's convention) (`_short_sum_cases`). The closed form is an
/// expansion for generic parameters: `Sum(x**i/factorial(i), (i, 0, n))`
/// was `exp(x)` at `n = -1`, an empty sum.
fn short_sum_cases(
    f: &Basic,
    i: &Basic,
    a: &Basic,
    b: &Basic,
    closed: &Basic,
    cond: &Basic,
) -> PolyResult<Vec<(Basic, Basic)>> {
    use crate::core::function::expand_default as expand;
    let mut cases: Vec<(Basic, Basic)> = Vec::new();
    let mut symbols: Vec<Basic> = b.free_symbols().into_iter().filter(|s| s != i).collect();
    symbols.sort_by_cached_key(crate::core::sorting::sort_key);
    for below in 0..3i64 {
        let length: Basic = expand(&a2(&s2(b, a)?, &int(1 + below))?)?;
        for sym in &symbols {
            let slope: Basic = crate::core::function::derivative(&length, sym)?;
            if slope.as_rational().is_none() || slope == Basic::zero() {
                continue;
            }
            let point: Basic = expand(&s2(sym, &crate::core::mul::div(&length, &slope)?)?)?;
            if point.free_symbols().contains(sym) {
                continue;
            }
            if yes(sym, Predicate::integer) && crate::core::expr::is(&point, Predicate::integer) == Some(false) {
                break;
            }
            let at_where: Basic = crate::core::relational::eq(sym, &point)?;
            let false_b: Basic = Basic::atom(Head::BooleanFalse);
            if at_where == false_b || subs(cond, sym, &point)? == false_b {
                break;
            }
            let mut terms: Vec<Basic> = Vec::new();
            for t in 1..=below {
                terms.push(subs(f, i, &s2(a, &int(t))?)?);
            }
            let value: Basic = neg(&crate::core::add::add(&terms)?)?;
            let at_point: Basic = subs(closed, sym, &point)?;
            if count_illegal(&value) == 0
                && at_point != value
                && expand(&s2(&at_point, &value)?)? != Basic::zero()
                && numerically_different(&at_point, &value)
            {
                cases.push((value, at_where));
            }
            break;
        }
    }
    Ok(cases)
}

/// Whether `u` is not a number (`nan`, an infinity) or the two expressions
/// have different reliable numerical values at a sample point of their
/// symbols (integers for the integer symbols); `false` when no values are
/// found (`_numerically_different`).
fn numerically_different(u: &Basic, v: &Basic) -> bool {
    if count_illegal(u) > 0 {
        return true;
    }
    let mut symbols: Vec<Basic> = u.free_symbols().into_iter().chain(v.free_symbols()).collect();
    symbols.sort_by_cached_key(crate::core::sorting::sort_key);
    symbols.dedup();
    for start in [3i64, 8] {
        let values: Vec<(Basic, Basic)> = symbols
            .iter()
            .enumerate()
            .map(|(index, sym)| {
                let index: i64 = index as i64;
                let value: Basic = if yes(sym, Predicate::integer) {
                    int(start + 2 * index)
                } else {
                    Basic::rational(
                        crate::numbers::Int::from(7 * (start + index) + 2),
                        crate::numbers::Int::from(5),
                    )
                    .unwrap_or_else(|_| int(start))
                };
                (sym.clone(), value)
            })
            .collect();
        let first = crate::numeric::reliable_complex(u, 25, &values);
        let second = crate::numeric::reliable_complex(v, 25, &values);
        if let (Some(x), Some(y)) = (first, second) {
            let size = |z: (f64, f64)| z.0.hypot(z.1);
            let gap: f64 = (x.0 - y.0).hypot(x.1 - y.1);
            if gap > 1e-12 * size(x).max(size(y)).max(1.0) {
                return true;
            }
        }
    }
    false
}

pub fn eval_sum_hyper(f: &Basic, limits: &Limit) -> OptR {
    use crate::core::logic::{and_, false_};
    let (i, a, b) = limits;
    if is_hypergeometric(f, i)? == Some(false) {
        return Ok(None);
    }
    if as_int(&s2(b, a)?).is_some() {
        return Ok(None);
    }
    let old_sum: Basic = crate::concrete::expr_with_limits::sum_(f, &[tuple(limits)])?;
    if !is_oo(b) {
        if is_neg_oo(a) {
            return Ok(match sum_hyper_at(&subs(f, i, &neg(i)?)?, i, &neg(b)?)? {
                Some((r, c)) => Some(piecewise(&[(r, c), (old_sum, true_())])?),
                None => None,
            });
        }
        let had: usize = count_illegal(f);
        let Some((r1, c1)) = sum_hyper_at(f, i, a)? else {
            return Ok(None);
        };
        if count_illegal(&r1) > had {
            return Ok(None);
        }
        let Some((r2, c2)) = sum_hyper_at(f, i, &a2(b, &int(1))?)? else {
            return Ok(None);
        };
        if count_illegal(&r2) > had {
            return Ok(None);
        }
        let cond: Basic = and_(&[c1, c2]);
        if cond == false_() {
            return Ok(None);
        }
        let closed: Basic = s2(&r1, &r2)?;
        let mut pieces: Vec<(Basic, Basic)> = short_sum_cases(f, i, a, b, &closed, &cond)?;
        pieces.push((closed, cond));
        pieces.push((old_sum, true_()));
        return Ok(Some(piecewise(&pieces)?));
    }
    if is_neg_oo(a) {
        let (Some((r1, c1)), Some((r2, c2))) = (
            sum_hyper_at(&subs(f, i, &neg(i)?)?, i, &int(1))?,
            sum_hyper_at(f, i, &int(0))?,
        ) else {
            return Ok(None);
        };
        let cond: Basic = and_(&[c1, c2]);
        if cond == false_() || as_set_is_empty(&cond)? {
            return Ok(None);
        }
        return Ok(Some(piecewise(&[(a2(&r1, &r2)?, cond), (old_sum, true_())])?));
    }
    let Some((r, c)) = sum_hyper_at(f, i, a)? else {
        return Ok(None);
    };
    if c == false_() {
        if crate::core::assumptions::is_number(&r) {
            let positive: Basic =
                crate::core::symbol::Dummy::new("i", &[(Predicate::integer, true), (Predicate::positive, true)]).0;
            let shifted: Basic = subs(f, i, &a2(&positive, a)?)?;
            if yes(&shifted, Predicate::positive) || yes(&shifted, Predicate::zero) {
                return Ok(Some(Basic::oo()));
            }
            if yes(&shifted, Predicate::negative) {
                return Ok(Some(neg(&Basic::oo())?));
            }
        }
        return Ok(None);
    }
    Ok(Some(piecewise(&[(r, c), (old_sum, true_())])?))
}

fn as_set_is_empty(cond: &Basic) -> PolyResult<bool> {
    if *cond == true_() || *cond == crate::core::logic::false_() {
        return Ok(*cond == crate::core::logic::false_());
    }
    let free = cond.free_symbols();
    let symbols: Vec<Basic> = free.iter().filter(|s| s.is_symbol()).cloned().collect();
    if symbols.len() != 1 || free.len() != 1 {
        return Ok(false);
    }
    let name: String = match symbols[0].head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => return Ok(false),
    };
    let plain: Basic = crate::core::symbol::Dummy::new(&name, &[]).0;
    let map: std::collections::HashMap<Basic, Basic> = [(symbols[0].clone(), plain.clone())].into_iter().collect();
    let relaxed: Basic = crate::core::function::xreplace_evaluated(cond, &map)?;
    if !crate::core::logic::is_boolean(&relaxed) {
        return Ok(false);
    }
    let reals: Basic = crate::sets::sets::reals();
    let found = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::assumptions::solve::solve(
            std::slice::from_ref(&relaxed),
            Some(std::slice::from_ref(&plain)),
            &[],
            Some(&reals),
            false,
        )
    })?;
    if let Some(found) = found {
        if !has_head(&found, &|n| matches!(n.head(), Head::ConditionSet)) {
            return Ok(matches!(found.head(), Head::EmptySet));
        }
    }
    Ok(false)
}

fn distinct_roots(denom: &crate::concrete::gosper::UPoly, i: &Basic) -> PolyResult<Vec<Basic>> {
    let rational: bool = denom.c.iter().all(|c| c.is_rational_class());
    let found: Vec<Basic> = if rational {
        crate::polys::complexrootof::all_roots(&denom.as_expr(i)?, Some(i), true)?
    } else {
        let mut sqf = denom.clone();
        if denom.degree() > 2 {
            sqf = denom.quo(&denom.gcd(&denom.diff()?)?)?;
        }
        let found = crate::polys::polyroots::ex_roots(&sqf.as_expr(i)?, i)?;
        if found.len() as i64 != sqf.degree() {
            return Err(PolyError::NotImplemented(format!(
                "sorted roots of {} are not supported",
                denom.as_expr(i)?
            )));
        }
        found
    };
    let mut distinct: Vec<Basic> = Vec::new();
    for r in found {
        if !distinct.contains(&r) {
            distinct.push(r);
        }
    }
    Ok(distinct)
}

fn multiplicity(denom: &crate::concrete::gosper::UPoly, r: &Basic) -> PolyResult<i64> {
    let Head::ComplexRootOf(data) = r.head() else {
        return Err(PolyError::Type("a ComplexRootOf".to_string()));
    };
    let ascending: Vec<Basic> = data
        .coefficients
        .iter()
        .rev()
        .map(|c| Basic::from_rational(c.clone()))
        .collect();
    let minimal = crate::concrete::gosper::UPoly::new(&ascending)?;
    let mut k: i64 = 0;
    let mut p = denom.clone();
    loop {
        let (q, remainder) = p.divmod(&minimal)?;
        if !remainder.is_zero() {
            return Ok(k);
        }
        k += 1;
        p = q;
    }
}

fn algebraic_residue(
    numer: &crate::concrete::gosper::UPoly,
    denom: &crate::concrete::gosper::UPoly,
    kernel: &Basic,
    z: &Basic,
    r: &Basic,
    m: i64,
) -> R {
    if m == 1 {
        return d2(
            &m2(&numer.as_expr(r)?, &subs(kernel, z, r)?)?,
            &denom.diff()?.as_expr(r)?,
        );
    }
    let t: Basic = crate::core::symbol::Dummy::new("t", &[]).0;
    let mut quotient: Vec<Basic> = denom.c.clone();
    for _ in 0..m {
        let top: usize = quotient.len() - 1;
        let mut out: Vec<Basic> = vec![int(0); top];
        let mut carry: Basic = int(0);
        for k in (1..=top).rev() {
            carry = a2(&quotient[k], &m2(&carry, &t)?)?;
            out[k - 1] = carry.clone();
        }
        quotient = out;
    }
    let reduced: Basic = add_all(
        &quotient
            .iter()
            .enumerate()
            .map(|(k, c)| m2(c, &pw(z, &int(k as i64))?))
            .collect::<PolyResult<Vec<_>>>()?,
    )?;
    let mut g: Basic = d2(&m2(&numer.as_expr(z)?, kernel)?, &reduced)?;
    for _ in 0..m - 1 {
        g = crate::core::function::derivative(&g, z)?;
    }
    let value: Basic = d2(
        &subs(&g, z, &t)?,
        &crate::functions::combinatorial::factorials::factorial(&int(m - 1))?,
    )?;
    subs(&value, &t, r)
}

/// The sum of a rational function over the integers by the residues of
/// `cot(pi z)` or `csc(pi z)` (`eval_sum_residue`).
pub fn eval_sum_residue(f: &Basic, limits: &Limit) -> OptR {
    use crate::concrete::gosper::UPoly;
    let (i, a, b) = limits;
    if a_gt_b(a, b)? {
        // Karr's convention, with the sign (see eval_sum)
        if s2(a, b)? == int(1) {
            return Ok(Some(int(0)));
        }
        return match eval_sum_residue(f, &(i.clone(), a2(b, &int(1))?, s2(a, &int(1))?))? {
            Some(r) => Ok(Some(neg(&r)?)),
            None => Ok(None),
        };
    }
    let is_even_function = |numer: &UPoly, denom: &UPoly| -> bool {
        let nm: Vec<usize> = numer
            .c
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != int(0))
            .map(|(k, _)| k)
            .collect();
        let dm: Vec<usize> = denom
            .c
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != int(0))
            .map(|(k, _)| k)
            .collect();
        (nm.iter().all(|k| k % 2 == 0) && dm.iter().all(|k| k % 2 == 0))
            || (nm.iter().all(|k| k % 2 == 1) && dm.iter().all(|k| k % 2 == 1))
    };
    let match_rational = |g: &Basic| -> PolyResult<Option<(UPoly, UPoly)>> {
        let (numer, denom) = crate::core::expr::as_numer_denom(g)?;
        match (UPoly::of(&numer, i), UPoly::of(&denom, i)) {
            (Ok(n), Ok(d)) => Ok(Some((n, d))),
            (Err(PolyError::Value(_)), _) | (_, Err(PolyError::Value(_))) => Ok(None),
            (Err(e), _) | (_, Err(e)) => Err(e),
        }
    };
    let get_poles = |denom: &UPoly| -> PolyResult<Option<(Vec<Basic>, Vec<Basic>)>> {
        let found: Vec<Basic> = distinct_roots(denom, i)?;
        if found.iter().any(|r| is(r, Predicate::integer).is_none()) {
            return Ok(None);
        }
        Ok(Some((
            found.iter().filter(|r| yes(r, Predicate::integer)).cloned().collect(),
            found
                .iter()
                .filter(|r| is(r, Predicate::integer) == Some(false))
                .cloned()
                .collect(),
        )))
    };
    let z: Basic = crate::core::symbol::Dummy::new("z", &[]).0;
    let residues = |numer: &UPoly, denom: &UPoly, alternating: bool, poles: &[Basic]| -> PolyResult<Vec<Basic>> {
        let pz: Basic = m2(&Basic::pi(), &z)?;
        let kernel: Basic = if alternating {
            crate::core::function::rebuild_evaluated(
                &Basic::new(Head::Function(Function::Csc), vec![pz.clone()]),
                vec![pz],
            )?
        } else {
            crate::core::function::rebuild_evaluated(
                &Basic::new(Head::Function(Function::Cot), vec![pz.clone()]),
                vec![pz],
            )?
        };
        let g: Basic = subs(&d2(&numer.as_expr(i)?, &denom.as_expr(i)?)?, i, &z)?;
        let mut out: Vec<Basic> = Vec::new();
        for root in poles {
            if matches!(root.head(), Head::ComplexRootOf(_)) {
                out.push(algebraic_residue(
                    numer,
                    denom,
                    &kernel,
                    &z,
                    root,
                    multiplicity(denom, root)?,
                )?);
            } else {
                out.push(crate::series::residues::residue(&m2(&g, &kernel)?, &z, root)?);
            }
        }
        Ok(out)
    };
    if f.free_symbols().iter().any(|s| s != i) {
        return Ok(None);
    }
    let bound_ok = |e: &Basic| as_int(e).is_some() || is_oo(e) || is_neg_oo(e);
    if !(bound_ok(a) && bound_ok(b)) {
        return Ok(None);
    }
    if !is_neg_oo(a) && !is_oo(b) {
        return Ok(None);
    }
    let mut alternating: bool = false;
    let (mut numer, mut denom) = match match_rational(f)? {
        Some(m) => m,
        None => {
            let Some(m) = match_rational(&d2(f, &pw(&int(-1), i)?)?)? else {
                return Ok(None);
            };
            alternating = true;
            m
        }
    };
    if denom.degree() - numer.degree() < 2 {
        return Ok(None);
    }
    let minus_pi: Basic = neg(&Basic::pi())?;
    if is_neg_oo(a) && is_oo(b) {
        let Some((int_roots, nonint_roots)) = get_poles(&denom)? else {
            return Ok(None);
        };
        if !int_roots.is_empty() {
            return Ok(None);
        }
        return Ok(Some(m2(
            &minus_pi,
            &add_all(&residues(&numer, &denom, alternating, &nonint_roots)?)?,
        )?));
    }
    if !(yes(a, Predicate::finite) && is_oo(b)) {
        return Ok(None);
    }
    if !is_even_function(&numer, &denom) {
        let n: i64 = denom.degree();
        let shift: Basic = d2(&d2(&neg(&denom.nth(n - 1))?, &denom.nth(n))?, &int(n))?;
        if as_int(&shift).is_none() || shift == int(0) {
            return Ok(None);
        }
        numer = numer.shift(&shift)?;
        denom = denom.shift(&shift)?;
        if !is_even_function(&numer, &denom) {
            return Ok(None);
        }
        let g: Basic = if alternating {
            m2(
                &pw(&int(-1), i)?,
                &d2(&m2(&pw(&int(-1), &shift)?, &numer.as_expr(i)?)?, &denom.as_expr(i)?)?,
            )?
        } else {
            d2(&numer.as_expr(i)?, &denom.as_expr(i)?)?
        };
        return eval_sum_residue(&g, &(i.clone(), s2(a, &shift)?, s2(b, &shift)?));
    }
    let Some((int_roots, nonint_roots)) = get_poles(&denom)? else {
        return Ok(None);
    };
    let ints: Vec<i64> = int_roots.iter().filter_map(as_int).collect();
    let a_int: i64 = as_int(a).ok_or_else(|| PolyError::Value("an integer bound".to_string()))?;
    if !ints.is_empty() {
        let (lo, hi) = (*ints.iter().min().expect("ints"), *ints.iter().max().expect("ints"));
        if ints.len() as i64 != hi - lo + 1 || a_int <= hi {
            return Ok(None);
        }
    }
    let mut all_roots: Vec<Basic> = int_roots.clone();
    all_roots.extend(nonint_roots);
    let full_sum: Basic = m2(
        &minus_pi,
        &add_all(&residues(&numer, &denom, alternating, &all_roots)?)?,
    )?;
    if ints.is_empty() {
        let half_sum: Basic = d2(&a2(&full_sum, &subs(f, i, &int(0))?)?, &int(2))?;
        let extraneous_neg: Vec<Basic> = (a_int..0).map(|i0| subs(f, i, &int(i0))).collect::<PolyResult<_>>()?;
        let extraneous_pos: Vec<Basic> = (0..a_int).map(|i0| subs(f, i, &int(i0))).collect::<PolyResult<_>>()?;
        return Ok(Some(s2(
            &a2(&half_sum, &add_all(&extraneous_neg)?)?,
            &add_all(&extraneous_pos)?,
        )?));
    }
    let half_sum: Basic = d2(&full_sum, &int(2))?;
    let hi: i64 = *ints.iter().max().expect("ints");
    let extraneous: Vec<Basic> = (hi + 1..a_int)
        .map(|i0| subs(f, i, &int(i0)))
        .collect::<PolyResult<_>>()?;
    Ok(Some(s2(&half_sum, &add_all(&extraneous)?)?))
}

const INHERITED: [Predicate; 14] = [
    Predicate::extended_nonnegative,
    Predicate::nonnegative,
    Predicate::extended_nonpositive,
    Predicate::nonpositive,
    Predicate::extended_positive,
    Predicate::positive,
    Predicate::extended_negative,
    Predicate::negative,
    Predicate::integer,
    Predicate::rational,
    Predicate::finite,
    Predicate::zero,
    Predicate::real,
    Predicate::extended_real,
];

/// A dummy for the variable with the properties its bounds share
/// (`_dummy_with_inherited_properties`).
pub(crate) fn dummy_with_inherited_properties(limits: &Limit) -> Option<Basic> {
    let (x, a, b) = limits;
    let mut keep: Vec<(Predicate, bool)> = Vec::new();
    let mut added: Vec<(Predicate, bool)> = Vec::new();
    let facts = facts_of(x);
    for p in INHERITED {
        if facts.get(&p) == Some(&true) {
            keep.push((p, true));
        } else if [a, b].iter().all(|e| facts_of(e).get(&p) == Some(&true)) {
            added.push((p, true));
        }
    }
    if added.is_empty() {
        return None;
    }
    keep.extend(added);
    Some(crate::core::symbol::Dummy::new("d", &keep).0)
}

fn eval_zeta_function(s: &Basic, f: &Basic, limits: &Limit) -> OptR {
    let (i, a0, b0) = limits;
    let (mut a, mut b, mut sign): (Basic, Basic, Basic) = (a0.clone(), b0.clone(), int(1));
    if a_gt_b(&a, &b)? {
        // Karr's convention, with the sign (see eval_sum)
        (a, b, sign) = (a2(b0, &int(1))?, s2(a0, &int(1))?, int(-1));
    }
    let (a, b) = (&a, &b);
    if !is_oo(b) {
        return Ok(None);
    }
    let excl = std::slice::from_ref(i);
    let (w, y, z) = (wild("w", excl), wild("y", excl), wild("z", excl));
    let pattern: Basic = pw(&a2(&m2(&w, i)?, &y)?, &neg(&z)?)?;
    let Some(result) = matched(f, &pattern)? else {
        return Ok(None);
    };
    if result.is_empty() {
        return Ok(None);
    }
    let (Some(rw), Some(ry), Some(rz)) = (get(&result, &w), get(&result, &y), get(&result, &z)) else {
        return Ok(None);
    };
    let coeff: Basic = d2(&int(1), &pw(&rw, &rz)?)?;
    let q: Basic = a2(&d2(&ry, &rw)?, a)?;
    let zeta: Basic = crate::functions::special::zeta_functions::zeta(&rz, Some(&q))?;
    let naturals0: Basic = Basic::atom(Head::Naturals0);
    let condition: Basic = crate::core::logic::and_(&[
        crate::core::logic::not_(&crate::sets::sets::contains(&naturals0, &neg(&q)?)?),
        crate::core::relational::gt(&crate::functions::elementary::complexes::re(&rz)?, &int(1))?,
    ]);
    Ok(Some(piecewise(&[
        (m2(&m2(&sign, &coeff)?, &zeta)?, condition),
        (s.clone(), true_()),
    ])?))
}

/// The sum evaluated over its limits in turn (`sum_doit`).
pub fn sum_doit(s: &Basic, deep: bool) -> R {
    use crate::concrete::expr_with_limits::{limits, sum_};
    let mut f: Basic = if deep {
        crate::core::function::doit(&s.args()[0])?
    } else {
        s.args()[0].clone()
    };
    let lims: Vec<Limit> = limits(s);
    let mut reps: Vec<(Basic, Basic)> = Vec::new();
    for xab in &lims {
        if let Some(d) = dummy_with_inherited_properties(xab) {
            reps.push((xab.0.clone(), d));
        }
    }
    if !reps.is_empty() {
        let forward: std::collections::HashMap<Basic, Basic> = reps.iter().cloned().collect();
        let undo: std::collections::HashMap<Basic, Basic> = reps.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
        return Ok(sum_doit(&s.xreplace(&forward), deep)?.xreplace(&undo));
    }
    for (n, (i, a, b)) in lims.iter().enumerate() {
        let (mut a, mut b) = (a.clone(), b.clone());
        let dif: Basic = s2(&b, &a)?;
        if dif == int(-1) {
            return Ok(int(0));
        }
        if yes(&dif, Predicate::integer) && yes(&dif, Predicate::negative) {
            let (na, nb) = (a2(&b, &int(1))?, s2(&a, &int(1))?);
            a = na;
            b = nb;
            f = neg(&f)?;
        }
        match eval_sum(&f, &(i.clone(), a.clone(), b.clone()))? {
            Some(newf) => f = newf,
            None => {
                if f == s.args()[0] {
                    if let Some(z) = eval_zeta_function(s, &f, &(i.clone(), a, b))? {
                        return Ok(z);
                    }
                    return Ok(s.clone());
                }
                let rest: Vec<Basic> = lims[n..].iter().map(tuple).collect();
                return sum_(&f, &rest);
            }
        }
    }
    if deep && !is_piecewise(&f) {
        return crate::core::function::doit(&f);
    }
    Ok(f)
}

/// `Sum(f, *limits)` evaluated (`summation`).
pub fn summation(f: &Basic, limit: &Limit) -> R {
    let s: Basic = crate::concrete::expr_with_limits::sum_(f, &[tuple(limit)])?;
    if !matches!(s.head(), Head::Sum) {
        return Ok(s);
    }
    sum_doit(&s, false)
}
