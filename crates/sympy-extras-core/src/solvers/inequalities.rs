//! Port of `sympy_extras/solvers/inequalities.py` (SymPy's
//! `_solve_inequality`): a relational solved for one variable; linear
//! ones by dividing out the coefficient of known sign (with the conditions
//! at the infinities and at the lost denominators), rational ones with
//! rational coefficients by the CAD, the others by the native
//! transcendental solver.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::{factors_of, is, terms_of};
use crate::error::{PolyError, PolyResult};
use crate::solvers::transcendental::relation;

type R = PolyResult<Basic>;

fn true_() -> Basic {
    crate::core::logic::boolean(true)
}
fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}
fn is_false(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanFalse)
}

fn expanded_terms(e: &Basic, s: &Basic) -> PolyResult<Option<Vec<(Basic, i64)>>> {
    let ex: Basic = crate::core::function::expand_default(e)?;
    let mut found: Vec<(Basic, i64)> = Vec::new();
    for t in terms_of(&ex) {
        let (c, k) = crate::solvers::solvers::as_coeff_exponent(&t, s)?;
        let kv: Option<i64> = if k.is_integer_value() {
            k.as_rational()
                .and_then(|r| i64::try_from(&crate::numbers::numer(r)).ok())
        } else {
            None
        };
        match kv {
            Some(v) if v >= 0 && !c.has(s) => found.push((c, v)),
            _ => return Ok(None),
        }
    }
    Ok(Some(found))
}

fn degree(e: &Basic, s: &Basic) -> PolyResult<Option<i64>> {
    Ok(expanded_terms(e, s)?.map(|ts| ts.iter().map(|(_, k)| *k).max().unwrap_or(0).max(0)))
}

/// The truth of `ie` at `s = value` (`nan` for a type error), `None`
/// when undecided (`_classify`).
fn classify(ie: &Basic, s: &Basic, value: &Basic) -> PolyResult<Option<Basic>> {
    let v: Basic = match crate::core::subs::subs(ie, s, value) {
        Ok(v) => v,
        Err(PolyError::Type(_)) => return Ok(Some(Basic::nan())),
        Err(e) => return Err(e),
    };
    if matches!(v.head(), Head::NaN) {
        return Ok(Some(v));
    }
    if !is_true(&v) && !is_false(&v) {
        return Ok(None);
    }
    Ok(Some(v))
}

fn classify_is(ie: &Basic, s: &Basic, value: &Basic, truth: bool) -> PolyResult<bool> {
    Ok(classify(ie, s, value)?.is_some_and(|v| if truth { is_true(&v) } else { is_false(&v) }))
}

fn rel_parts(ie: &Basic) -> (RelOp, Basic, Basic) {
    let Head::Relational(op) = ie.head() else {
        unreachable!("a relation")
    };
    (*op, ie.args()[0].clone(), ie.args()[1].clone())
}

fn reversed_op(op: RelOp) -> RelOp {
    match op {
        RelOp::Lt => RelOp::Gt,
        RelOp::Le => RelOp::Ge,
        RelOp::Gt => RelOp::Lt,
        RelOp::Ge => RelOp::Le,
        other => other,
    }
}

fn rational_set(ie: &Basic, s: &Basic) -> PolyResult<Option<Basic>> {
    let (op, l, r) = rel_parts(ie);
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(
        &crate::core::add::sub(&l, &r)?,
        false,
        true,
    )?)?;
    for part in [&numerator, &denominator] {
        match expanded_terms(part, s)? {
            Some(ts) if ts.iter().all(|(c, _)| is(c, Predicate::rational) == Some(true)) => {}
            _ => return Ok(None),
        }
    }
    let mut formula: Basic = relation(
        op,
        &crate::core::mul::mul(&[numerator.clone(), denominator.clone()])?,
        &Basic::zero(),
    )?;
    if denominator.has(s) {
        formula = crate::core::logic::and_(&[formula, crate::core::relational::ne(&denominator, &Basic::zero())?]);
    }
    Ok(Some(crate::polys::roots::in_radicals(
        &crate::polys::cad::qe::solution_set(&formula, s, &[], None, true)?,
    )?))
}

fn plain(t: &Basic) -> bool {
    match t.head() {
        Head::EmptySet | Head::FiniteSet | Head::Interval | Head::Reals => true,
        Head::Union => t.args().iter().all(plain),
        _ => false,
    }
}

fn transcendental_set(ie: &Basic, s: &Basic) -> R {
    // the zero-set solver of assumptions/analysis
    let solver = |formula: &Basic, v: &Basic| crate::assumptions::analysis::native_solver_pub(formula, v);
    let solved: Option<Basic> =
        crate::solvers::transcendental::solve_transcendental(ie, s, Some(&solver), None, false)?;
    match solved {
        Some(set) if plain(&set) => Ok(set),
        _ => Err(PolyError::NotImplemented(format!(
            "The inequality {} cannot be solved for {}",
            ie, s
        ))),
    }
}

fn nonlinear(ie: &Basic, s: &Basic) -> R {
    use crate::core::relational as rl;
    let solved: Basic = match rational_set(ie, s)? {
        Some(set) => set,
        None => transcendental_set(ie, s)?,
    };
    let mut rv: Basic = crate::sets::sets::as_relational(&solved, s)?;
    let (oo, moo) = (Basic::oo(), Basic::negative_oo());
    let okoo: bool = classify_is(ie, s, &oo, true)?;
    if okoo && classify_is(&rv, s, &oo, false)? {
        rv = crate::core::subs::subs(&rv, &rl::lt(s, &oo)?, &true_())?;
    }
    let oknoo: bool = classify_is(ie, s, &moo, true)?;
    if oknoo && classify_is(&rv, s, &moo, false)? {
        rv = crate::core::subs::subs(&rv, &rl::lt(&moo, s)?, &true_())?;
        rv = crate::core::subs::subs(&rv, &rl::gt(s, &moo)?, &true_())?;
    }
    if is_true(&rv) {
        rv = if okoo { rl::le(s, &oo)? } else { rl::lt(s, &oo)? };
        if !oknoo {
            rv = crate::core::logic::and_(&[rl::lt(&moo, s)?, rv]);
        }
    }
    Ok(rv)
}

/// The factors of the denominators of every subexpression (`denoms`).
fn denominators(b: &Basic) -> PolyResult<Vec<Basic>> {
    let mut found: Vec<Basic> = Vec::new();
    for p in b.preorder() {
        if matches!(p.head(), Head::Relational(_) | Head::Tuple) || crate::core::logic::is_boolean(&p) {
            continue;
        }
        let d: Basic = crate::simplify::radsimp::denom_of(&p)?;
        if d == Basic::one() {
            continue;
        }
        for f in factors_of(&d) {
            if !found.contains(&f) {
                found.push(f);
            }
        }
    }
    Ok(found)
}

/// `ie` solved for `s` (`_solve_inequality`); `NotImplemented` when it is
/// not solved.
pub fn solve_inequality(ie: &Basic, s: &Basic) -> R {
    use crate::core::relational as rl;
    if !matches!(ie.head(), Head::Relational(_)) || !ie.free_symbols().contains(s) {
        return Ok(ie.clone());
    }
    let mut ie: Basic = ie.clone();
    if ie.args()[1] == *s {
        ie = rl::reversed(&ie);
    }
    let (op, l, r) = rel_parts(&ie);
    if l == *s && !r.free_symbols().contains(s) {
        return Ok(ie);
    }
    let expr: Basic = crate::core::add::sub(&l, &r)?;
    let deg: Option<i64> = degree(&expr, s)?;
    if deg.is_none() || deg.is_some_and(|d| d > 1) {
        return nonlinear(&ie, s);
    }
    let mut conds: Vec<Basic> = Vec::new();
    let rv: Basic = if deg == Some(0) {
        relation(op, &crate::core::function::expand_default(&expr)?, &Basic::zero())?
    } else {
        let mut e: Basic = crate::core::function::expand_default(&expr)?;
        let (b, _) = crate::core::expr::as_independent(&e, s, true)?;
        e = crate::core::add::sub(&e, &b)?;
        let mut rhs: Basic = crate::core::add::sub(&Basic::zero(), &b)?;
        let ef: Basic = crate::core::exprtools::factor_terms(&e, false, false, false, true)?;
        let (mut a, mut e2) = crate::core::expr::as_independent(&ef, s, false)?;
        let neg_unknown: bool = is(&a, Predicate::negative).is_none() && is(&a, Predicate::positive).is_none();
        if is(&a, Predicate::zero) != Some(false) || (neg_unknown && !matches!(op, RelOp::Ne | RelOp::Eq)) {
            e2 = ef.clone();
            a = Basic::one();
        }
        rhs = crate::core::mul::div(&rhs, &a)?;
        let rv: Basic = if is(&a, Predicate::positive) == Some(true) {
            relation(op, &e2, &rhs)?
        } else {
            relation(reversed_op(op), &e2, &rhs)?
        };
        let current: Vec<Basic> = denominators(&rv)?;
        for d in denominators(&ie)? {
            if current.contains(&d) {
                continue;
            }
            let c: Basic = solve_inequality(&rl::eq(&d, &Basic::zero())?, s)?;
            if matches!(c.head(), Head::Relational(RelOp::Eq))
                && c.args()[0] == *s
                && classify_is(&rv, s, &c.args()[1], true)?
            {
                conds.push(crate::core::logic::not_(&c));
            }
        }
        for i in [Basic::negative_oo(), Basic::oo()] {
            if classify_is(&rv, s, &i, true)? && !classify_is(&ie, s, &i, true)? {
                conds.push(if matches!(i.head(), Head::Infinity) {
                    rl::lt(s, &i)?
                } else {
                    rl::lt(&i, s)?
                });
            }
        }
        rv
    };
    conds.push(rv);
    Ok(crate::core::logic::and_(&conds))
}
