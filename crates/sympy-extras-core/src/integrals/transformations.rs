//! Port of `sympy_extras/integrals/transformations.py`: Frullani's
//! theorem for `c*(g(a*x) - g(b*x))/x` over `(0, oo)`, and Glasser's
//! master theorem for `F(x - sum(a_i/(x - b_i)))` over the real line (or
//! the half line for an even `F`), the transformed integral computed by
//! the definite integrator.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::error::PolyResult;
use crate::integrals::conditions::ConditionalValue;

type R = PolyResult<Basic>;
type OptCV = PolyResult<Option<ConditionalValue>>;

fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn neg_oo() -> Basic {
    Basic::atom(Head::NegativeInfinity)
}
fn ask(query: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    crate::assumptions::ask::ask(query, assumptions, None)
}
fn gt0(e: &Basic) -> R {
    crate::core::relational::gt(e, &Basic::zero())
}
fn args_of(e: &Basic, add: bool) -> Vec<Basic> {
    if (add && e.is_add()) || (!add && e.is_mul()) {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn is_application(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(_) | Head::Undefined(_))
}
fn has_head(e: &Basic, f: &dyn Fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| f(n.head()))
}

fn limit_of(f: &Basic, x: &Basic, point: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    let value: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::assumptions::limits::limit(f, x, point, "+", assumptions, None)
    })?;
    let Some(value) = value else {
        return Ok(None);
    };
    let bad: bool = has_head(&value, &|h| {
        matches!(
            h,
            Head::Infinity
                | Head::NegativeInfinity
                | Head::ComplexInfinity
                | Head::NaN
                | Head::Limit
                | Head::AccumBounds
        )
    });
    let free: BTreeSet<Basic> = f.free_symbols();
    if bad || value.free_symbols().iter().any(|s| !free.contains(s)) {
        return Ok(None);
    }
    Ok(Some(value))
}

/// `Integral(f, (x, 0, oo))` by Frullani's theorem (`frullani`).
pub fn frullani(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptCV {
    let terms: Vec<Basic> = args_of(&crate::core::function::expand_method(f)?, true);
    if terms.len() != 2 {
        return Ok(None);
    }
    let reciprocal: Basic = crate::core::power::pow(x, &Basic::negative_one())?;
    let mut parts: Vec<(Basic, Basic, Basic)> = Vec::new();
    for term in &terms {
        let (coefficient, rest) = crate::core::expr::as_independent(term, x, false)?;
        if !rest.is_mul() {
            return Ok(None);
        }
        let factors: &[Basic] = rest.args();
        if factors.len() != 2 || !factors.contains(&reciprocal) {
            return Ok(None);
        }
        let function: &Basic = if factors[1] == reciprocal {
            &factors[0]
        } else {
            &factors[1]
        };
        if !is_application(function) || function.args().len() != 1 {
            return Ok(None);
        }
        let Some((c, e)) = crate::integrals::mellin::monomial(&function.args()[0], x)? else {
            return Ok(None);
        };
        if e != Basic::one() {
            return Ok(None);
        }
        parts.push((
            coefficient,
            c,
            crate::core::function::rebuild_evaluated(function, vec![x.clone()])?,
        ));
    }
    let (c1, mut a, mut g1) = parts[0].clone();
    let (c2, mut b, g2) = parts[1].clone();
    if g1 != g2 || crate::core::add::add(&[c1.clone(), c2])? != Basic::zero() {
        return Ok(None);
    }
    let negative = |e: &Basic| crate::core::expr::is(e, Predicate::negative) == Some(true);
    if negative(&a) && negative(&b) {
        a = crate::core::mul::neg(&a)?;
        b = crate::core::mul::neg(&b)?;
        g1 = crate::core::subs::subs(&g1, x, &crate::core::mul::neg(x)?)?;
    }
    if ask(&gt0(&a)?, assumptions)? == Some(false) || ask(&gt0(&b)?, assumptions)? == Some(false) {
        return Ok(None);
    }
    let Some(at_zero) = limit_of(&g1, x, &Basic::zero(), assumptions)? else {
        return Ok(None);
    };
    let Some(at_infinity) = limit_of(&g1, x, &oo(), assumptions)? else {
        return Ok(None);
    };
    let value: Basic = crate::core::mul::mul(&[
        crate::core::mul::mul(&[c1, crate::core::add::sub(&at_zero, &at_infinity)?])?,
        crate::functions::elementary::exponential::log(&crate::core::mul::div(&b, &a)?)?,
    ])?;
    Ok(Some(ConditionalValue::new(
        value,
        crate::core::logic::and_(&[gt0(&a)?, gt0(&b)?]),
    )))
}

/// `[(a_i, b_i)]` when `u == x - sum(a_i/(x - b_i))` with `a_i > 0`
/// (`_glasser_map`).
fn glasser_map(u: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    let decomposed: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::polys::partfrac::apart(u, Some(x), false)
    })?;
    let Some(decomposed) = decomposed else {
        return Ok(None);
    };
    let mut poles: Vec<(Basic, Basic)> = Vec::new();
    let mut linear: Basic = Basic::zero();
    for term in args_of(&decomposed, true) {
        if term == *x || !term.has(x) {
            linear = crate::core::add::add(&[linear, term])?;
            continue;
        }
        let (coefficient, rest) = crate::core::expr::as_independent(&term, x, false)?;
        if !(rest.is_pow() && rest.args()[1] == Basic::negative_one()) {
            return Ok(None);
        }
        let base: &Basic = &rest.args()[0];
        let slope: Basic = crate::core::function::derivative(base, x)?;
        if slope == Basic::zero() || slope.has(x) {
            return Ok(None);
        }
        let intercept: Basic = crate::core::function::expand_method(&crate::core::add::sub(
            base,
            &crate::core::mul::mul(&[slope.clone(), x.clone()])?,
        )?)?;
        if intercept.has(x) {
            return Ok(None);
        }
        let a_i: Basic = crate::core::mul::neg(&crate::core::mul::div(&coefficient, &slope)?)?;
        let b_i: Basic = crate::core::mul::neg(&crate::core::mul::div(&intercept, &slope)?)?;
        if ask(&gt0(&a_i)?, assumptions)? != Some(true) {
            return Ok(None);
        }
        poles.push((a_i, b_i));
    }
    if crate::core::add::sub(&linear, x)? != Basic::zero() || poles.is_empty() {
        return Ok(None);
    }
    Ok(Some(poles))
}

/// The arguments of functions and bases of powers depending on `x`,
/// largest first (`_inner_arguments`).
fn inner_arguments(f: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let mut found: BTreeSet<Basic> = BTreeSet::new();
    for node in f.atoms(Some(&is_application)) {
        for argument in node.args() {
            if argument.has(x) && argument != x {
                found.insert(argument.clone());
            }
        }
    }
    for node in f.atoms(Some(&|n: &Basic| n.is_pow())) {
        let base: &Basic = &node.args()[0];
        if base.has(x) && base != x {
            found.insert(base.clone());
        }
    }
    let mut keyed: Vec<(i64, String, Basic)> = Vec::new();
    for e in found {
        keyed.push((-crate::core::function::count_ops(&e)?, e.to_string(), e));
    }
    keyed.sort_by(|p, q| (p.0, &p.1).cmp(&(q.0, &q.1)));
    Ok(keyed.into_iter().map(|(_, _, e)| e).collect())
}

/// `Integral(f, (x, a, b))` by Glasser's master theorem (`glasser`).
pub fn glasser(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    let whole_line: bool = *a == neg_oo() && *b == oo();
    let half_line: bool = *a == Basic::zero() && *b == oo();
    if !whole_line && !half_line {
        return Ok(None);
    }
    let y: Basic = crate::integrals::definite::fresh("y", &[(Predicate::real, true)]);
    for v in inner_arguments(f, x)? {
        if glasser_map(&v, x, assumptions)?.is_none() {
            continue;
        }
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        rule.insert(v.clone(), y.clone());
        let replaced: Basic = crate::core::function::xreplace_evaluated(f, &rule)?;
        if replaced.has(x) {
            continue;
        }
        if half_line {
            let mut flip: HashMap<Basic, Basic> = HashMap::new();
            flip.insert(y.clone(), crate::core::mul::neg(&y)?);
            if crate::core::function::xreplace_evaluated(&replaced, &flip)? != replaced {
                continue;
            }
        }
        let lower: Basic = if whole_line { neg_oo() } else { Basic::zero() };
        if let Some(value) = crate::integrals::definite::conditional_integral(
            &replaced,
            &y,
            &lower,
            &oo(),
            assumptions,
            false,
            false,
            false,
        )? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// The first of the identities which applies (`transformation_integral`).
pub fn transformation_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    if *a == Basic::zero() && *b == oo() {
        if let Some(found) = frullani(f, x, assumptions)? {
            return Ok(Some(found));
        }
    }
    glasser(f, x, a, b, assumptions)
}
