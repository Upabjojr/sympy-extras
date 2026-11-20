//! Port of `sympy_extras/assumptions/solve.py`: equations and inequalities
//! solved under assumptions. One unknown: the CAD for polynomial
//! formulas, the transcendental solver, the periodic formulas tile by
//! tile, the native `solveset` (with the rational equations by
//! `solveset_rational`), the isolated real roots; several: integer linear
//! systems, regular chains, the CAD, SymPy's substitution method; with
//! `cases`, the parametric discussions. The solutions respect the flags
//! of the unknowns.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::assumptions::ask::{evaluate, facts_of};
use crate::assumptions::facts::{conjuncts, normalize, to_polynomial, Facts};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::is;
use crate::core::function::xreplace_evaluated;
use crate::core::logic::{and_, boolean, is_boolean};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::sets::conditionset::condition_set;
use crate::sets::sets as st;

type R = PolyResult<Basic>;
type Truth = Option<bool>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}
fn is_false(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanFalse)
}
fn true_() -> Basic {
    boolean(true)
}
fn reals() -> Basic {
    st::reals()
}
fn complexes() -> Basic {
    Basic::atom(Head::Complexes)
}
fn empty() -> Basic {
    Basic::atom(Head::EmptySet)
}
fn rel(b: &Basic) -> Option<RelOp> {
    match b.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}
fn sorted_symbols(s: &BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = s.iter().filter(|x| x.is_symbol()).cloned().collect();
    v.sort_by_key(|x| x.to_string());
    v
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn is_subset_reals(s: &Basic) -> PolyResult<bool> {
    Ok(st::is_subset(s, &reals())? == Some(true))
}

fn as_statement(item: &Basic) -> R {
    if is_boolean(item) {
        return Ok(item.clone());
    }
    if crate::sets::sets::is_set(item) || matches!(item.head(), Head::Tuple) {
        return Err(PolyError::Type(format!(
            "an equation or a Boolean is expected, got {}",
            item
        )));
    }
    crate::core::relational::eq(item, &int(0))
}

/// `x in s`, decided when the set decides it (`_membership`).
fn membership(x: &Basic, s: &Basic) -> R {
    let found: Basic = st::contains(s, x)?;
    if is_true(&found) || is_false(&found) {
        return Ok(found);
    }
    Ok(Basic::new(Head::Contains, vec![x.clone(), s.clone()]))
}

fn unevaluated(s: &Basic) -> bool {
    if matches!(s.head(), Head::Intersection | Head::Complement) {
        return true;
    }
    s.args().iter().filter(|a| st::is_set(a)).any(unevaluated)
}

fn evaluated_set(s: &Basic) -> R {
    if !unevaluated(s) {
        return Ok(s.clone());
    }
    evaluate_operations(s)
}

fn evaluate_operations(s: &Basic) -> R {
    match s.head() {
        Head::Union => {
            let parts: Vec<Basic> = s.args().iter().map(evaluate_operations).collect::<PolyResult<_>>()?;
            st::union(&parts)
        }
        Head::Complement => st::complement(&evaluate_operations(&s.args()[0])?, &evaluate_operations(&s.args()[1])?),
        Head::Intersection => {
            let parts: Vec<Basic> = s.args().iter().map(evaluate_operations).collect::<PolyResult<_>>()?;
            let mut result: Basic = parts[0].clone();
            for other in &parts[1..] {
                result = intersect_two(&result, other)?;
            }
            Ok(result)
        }
        _ => Ok(s.clone()),
    }
}

fn intersect_two(a: &Basic, b: &Basic) -> R {
    let plain: Basic = st::intersection(&[a.clone(), b.clone()])?;
    if !matches!(plain.head(), Head::Intersection) {
        return Ok(plain);
    }
    for (first, second) in [(a, b), (b, a)] {
        if matches!(first.head(), Head::ConditionSet) {
            let base: Basic = intersect_two(&first.args()[2], second)?;
            return condition_set(&first.args()[0], &first.args()[1], &base);
        }
        if matches!(first.head(), Head::FiniteSet) {
            return filtered_elements(first, second, &plain);
        }
        if matches!(first.head(), Head::ImageSet) && matches!(second.head(), Head::Interval) {
            if let Some(found) = image_in(first, second)? {
                return Ok(found);
            }
        }
    }
    Ok(plain)
}

fn membership_truth(e: &Basic, s: &Basic) -> PolyResult<Truth> {
    let found: Basic = st::contains(s, e)?;
    if is_true(&found) {
        return Ok(Some(true));
    }
    if is_false(&found) {
        return Ok(Some(false));
    }
    if matches!(s.head(), Head::FiniteSet) {
        for k in s.args() {
            if same_number(e, k)? {
                return Ok(Some(true));
            }
        }
        return Ok(None);
    }
    if matches!(s.head(), Head::Interval) && crate::core::assumptions::is_number(e) {
        use crate::solvers::isolation::compare_points;
        if clearly_not_real(e)? {
            return Ok(Some(false));
        }
        let (start, end, left_open, right_open) = st::interval_parts(s);
        let finite = |v: &Basic| is(v, Predicate::finite) == Some(true);
        let low: Option<i32> = if finite(&start) {
            compare_points(e, &start)?
        } else {
            Some(1)
        };
        let high: Option<i32> = if finite(&end) {
            compare_points(&end, e)?
        } else {
            Some(1)
        };
        let (Some(low), Some(high)) = (low, high) else {
            return Ok(None);
        };
        if low < 0 || high < 0 || (low == 0 && left_open) || (high == 0 && right_open) {
            return Ok(Some(false));
        }
        return Ok(Some(true));
    }
    Ok(None)
}

fn filtered_elements(elements: &Basic, other: &Basic, unevaluated_: &Basic) -> R {
    let mut kept: Vec<Basic> = Vec::new();
    for e in elements.args() {
        match membership_truth(e, other)? {
            None => return Ok(unevaluated_.clone()),
            Some(true) => kept.push(e.clone()),
            Some(false) => {}
        }
    }
    Ok(if kept.is_empty() {
        empty()
    } else {
        st::finite_set(&kept)
    })
}

const MAX_INTEGERS: i64 = 10000;

fn integer_bounds(low: &Basic, high: &Basic) -> Option<(i64, i64)> {
    let finite = |v: &Basic| is(v, Predicate::finite) == Some(true);
    let lo = if finite(low) {
        crate::numeric::reliable_complex(low, 30, &[])
    } else {
        None
    }?;
    let hi = if finite(high) {
        crate::numeric::reliable_complex(high, 30, &[])
    } else {
        None
    }?;
    let (m, n) = (lo.0.floor() as i64 - 1, hi.0.ceil() as i64 + 1);
    if n - m > MAX_INTEGERS {
        return None;
    }
    Some((m, n))
}

fn image_in(image: &Basic, region: &Basic) -> PolyResult<Option<Basic>> {
    // ImageSet(Tuple(variables), image, base_sets...)
    let variables: &[Basic] = image.args()[0].args();
    let expression: &Basic = &image.args()[1];
    let bases: &[Basic] = &image.args()[2..];
    if variables.len() != 1 || bases.len() != 1 || !matches!(bases[0].head(), Head::Integers) || is_boolean(expression)
    {
        return Ok(None);
    }
    let (start, end, _, _) = st::interval_parts(region);
    let finite = |v: &Basic| is(v, Predicate::finite) == Some(true);
    if !(finite(&start) && finite(&end)) {
        return Ok(None);
    }
    let k: &Basic = &variables[0];
    let a: Basic = crate::core::function::derivative(expression, k)?;
    if a.has(k)
        || !crate::core::assumptions::is_number(&a)
        || is(&a, Predicate::real) != Some(true)
        || is(&a, Predicate::zero) != Some(false)
    {
        return Ok(None);
    }
    let b: Basic = crate::core::subs::subs(expression, k, &int(0))?;
    if !crate::core::assumptions::is_number(&b) || is(&b, Predicate::real) != Some(true) {
        return Ok(None);
    }
    let positive: bool = is(&a, Predicate::positive) == Some(true);
    let negative: bool = is(&a, Predicate::negative) == Some(true);
    if !positive && !negative {
        return Ok(None);
    }
    let mut low: Basic = crate::core::mul::div(&sub(&start, &b)?, &a)?;
    let mut high: Basic = crate::core::mul::div(&sub(&end, &b)?, &a)?;
    if negative {
        std::mem::swap(&mut low, &mut high);
    }
    let Some((m, n)) = integer_bounds(&low, &high) else {
        return Ok(None);
    };
    let mut kept: Vec<Basic> = Vec::new();
    for j in m..=n {
        let point: Basic = crate::core::subs::subs(expression, k, &int(j))?;
        match membership_truth(&point, region)? {
            None => return Ok(None),
            Some(true) => kept.push(point),
            Some(false) => {}
        }
    }
    Ok(Some(if kept.is_empty() {
        empty()
    } else {
        st::finite_set(&kept)
    }))
}

fn intersect(sets: &[Basic]) -> R {
    evaluated_set(&st::intersection(sets)?)
}

fn domain_set(facts: &Facts, symbols: &[Basic], domain: Option<&Basic>) -> PolyResult<Basic> {
    if let Some(d) = domain {
        let named: bool = matches!(
            d.head(),
            Head::Reals | Head::Integers | Head::Naturals | Head::Naturals0 | Head::Rationals | Head::Complexes
        );
        if named || is_subset_reals(d)? {
            return Ok(d.clone());
        }
    }
    if symbols.iter().all(|s| facts.real.contains(s)) {
        return Ok(reals());
    }
    Ok(complexes())
}

fn solveset_(formula: &Basic, x: &Basic, domain: &Basic) -> R {
    match formula.head() {
        Head::BooleanTrue => return Ok(domain.clone()),
        Head::BooleanFalse => return Ok(empty()),
        Head::And => {
            let parts: Vec<Basic> = formula
                .args()
                .iter()
                .map(|a| solveset_(a, x, domain))
                .collect::<PolyResult<_>>()?;
            return intersect(&parts);
        }
        Head::Or => {
            let parts: Vec<Basic> = formula
                .args()
                .iter()
                .map(|a| solveset_(a, x, domain))
                .collect::<PolyResult<_>>()?;
            return st::union(&parts);
        }
        Head::Not => {
            let inner: Basic = solveset_(&formula.args()[0], x, domain)?;
            return evaluated_set(&st::complement(domain, &inner)?);
        }
        Head::Contains => {
            if formula.args()[0] == *x {
                return intersect(&[domain.clone(), formula.args()[1].clone()]);
            }
            return condition_set(x, formula, domain);
        }
        Head::Relational(_) => {
            if !formula.free_symbols().contains(x) {
                return condition_set(x, formula, domain);
            }
            if let Some(r) = rational_solutions(formula, x, domain)? {
                return Ok(r);
            }
            return crate::solvers::solveset::solveset(formula, x, domain);
        }
        _ => {}
    }
    condition_set(x, formula, domain)
}

fn given_assumptions(x: &Basic) -> Vec<(Predicate, bool)> {
    match x.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d
            .given
            .iter()
            .filter(|(p, _)| *p != Predicate::commutative)
            .cloned()
            .collect(),
        _ => vec![],
    }
}

fn rational_solutions(formula: &Basic, x: &Basic, domain: &Basic) -> PolyResult<Option<Basic>> {
    if rel(formula) != Some(RelOp::Eq) || !matches!(domain.head(), Head::Complexes) || !given_assumptions(x).is_empty()
    {
        return Ok(None);
    }
    let f: Basic = sub(&formula.args()[0], &formula.args()[1])?;
    if !low_degree(&f, x)? {
        return Ok(None);
    }
    let found =
        crate::timeout::attempt_within(timeout(), || crate::polys::rationalsolve::solveset_rational(&f, x))?.flatten();
    let Some((solutions, excluded)) = found else {
        return Ok(None);
    };
    let mut result: Basic = st::finite_set(&solutions);
    if !excluded.is_empty() {
        result = st::complement(&result, &st::finite_set(&excluded))?;
        if !matches!(result.head(), Head::FiniteSet | Head::EmptySet) {
            return Ok(None);
        }
    }
    Ok(Some(result))
}

const NATIVE_DEGREE: i64 = 3;

fn low_degree(f: &Basic, x: &Basic) -> PolyResult<bool> {
    let (numerator, _) = crate::core::expr::as_numer_denom(f)?;
    if crate::core::expr::polynomial_like(
        &numerator,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return Ok(false);
    }
    match crate::polys::polytools::degree(&numerator, x) {
        Ok(d) => Ok(d <= NATIVE_DEGREE),
        Err(PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_)) => Ok(false),
        Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => Ok(false),
        Err(e) => Err(e),
    }
}

fn holds_at(formula: &Basic, values: &HashMap<Basic, Basic>, facts: &Facts) -> PolyResult<Truth> {
    let instance: Basic = normalize(&xreplace_evaluated(formula, values)?)?;
    evaluate(&instance, facts)
}

fn tiny(digits: usize) -> Rat {
    let k: usize = digits * 2 / 3;
    Rat::from_parts(Int::ONE, dashu_int::UBig::from(10u8).pow(k))
}

fn clearly_not_real(e: &Basic) -> PolyResult<bool> {
    let s = crate::settings::settings();
    if !s.numerical_checks || is_boolean(e) {
        return Ok(false);
    }
    let Some(value) = crate::numeric::reliable_value(e, s.precision, &[]) else {
        return Ok(false);
    };
    let im: Rat = crate::assumptions::ivarith::real_to_rat(&crate::core::evalf::parts(&value).1);
    let magnitude: Rat = if crate::numbers::rat_is_negative(&im) { -im } else { im };
    Ok(magnitude > tiny(s.precision))
}

fn same_number(first: &Basic, second: &Basic) -> PolyResult<bool> {
    if first == second {
        return Ok(true);
    }
    if is_boolean(first) || is_boolean(second) || !first.free_symbols().is_empty() || !second.free_symbols().is_empty()
    {
        return Ok(false);
    }
    let difference: Basic = sub(first, second)?;
    let simplified: Option<Basic> = crate::timeout::attempt_within(timeout(), || {
        crate::simplify::simplify::simplify(&difference, &crate::simplify::simplify::Options::default())
    })?;
    if let Some(d) = &simplified {
        if *d == int(0) {
            return Ok(true);
        }
        if crate::core::assumptions::is_number(d) && is(d, Predicate::zero) == Some(false) {
            return Ok(false);
        }
    }
    let (Some(apart), Some(size)) = (
        crate::numeric::reliable_complex(&difference, 60, &[]),
        crate::numeric::reliable_complex(first, 60, &[]),
    ) else {
        return Ok(false);
    };
    let gap: f64 = apart.0.hypot(apart.1);
    let scale: f64 = size.0.hypot(size.1).max(1.0);
    if gap > 1e-50 * scale {
        return Ok(false);
    }
    if is(first, Predicate::algebraic) != Some(true) || is(second, Predicate::algebraic) != Some(true) {
        return Ok(false);
    }
    let t: Basic = crate::core::symbol::Dummy::new("t", &[]).0;
    let polynomials = crate::timeout::attempt_within(timeout(), || {
        Ok((
            crate::polys::numberfields::minpoly::minimal_polynomial(first, &t)?,
            crate::polys::numberfields::minpoly::minimal_polynomial(second, &t)?,
        ))
    })?;
    Ok(polynomials.is_some_and(|(p, q)| p == q))
}

/// The real and imaginary parts of `e` when its symbols are all real
/// under the facts, `None` otherwise (`_real_and_imaginary_parts`).
fn real_and_imaginary_parts(e: &Basic, facts: &Facts) -> PolyResult<Option<(Basic, Basic)>> {
    let free = e.free_symbols();
    let symbols: Vec<Basic> = free
        .iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_) | Head::Dummy(_, _)))
        .cloned()
        .collect();
    if symbols.len() != free.len() || symbols.iter().any(|s| !facts.real.contains(s)) {
        return Ok(None);
    }
    let mut forward: HashMap<Basic, Basic> = HashMap::new();
    let mut back: HashMap<Basic, Basic> = HashMap::new();
    for s in &symbols {
        let name: String = match s.head() {
            Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
            _ => s.to_string(),
        };
        let d: Basic = crate::core::symbol::Dummy::new(&name, &[(crate::core::assumptions::Predicate::real, true)]).0;
        forward.insert(s.clone(), d.clone());
        back.insert(d, s.clone());
    }
    let replaced: Basic = e.xreplace(&forward);
    let Some((re_part, im_part)) = crate::functions::elementary::complexes::as_real_imag(&replaced, true, None)? else {
        return Ok(None);
    };
    Ok(Some((re_part.xreplace(&back), im_part.xreplace(&back))))
}

/// Whether `e` is a real solution nowhere: real only where its imaginary
/// part vanishes, and there the condition fails
/// (`_not_real_where_condition_fails`; SymPy's `Q.real(I*y)` is `False`
/// for a real `y`, which forgot `y = 0`, sympy-extras#25).
fn not_real_where_condition_fails(e: &Basic, x: &Basic, condition: &Basic, facts: &Facts) -> PolyResult<bool> {
    if !e.has(&Basic::imaginary_unit()) {
        return Ok(false);
    }
    let Some((re_part, im_part)) = real_and_imaginary_parts(e, facts)? else {
        return Ok(false);
    };
    if im_part == Basic::zero() {
        return Ok(false);
    }
    let rule: HashMap<Basic, Basic> = [(x.clone(), re_part)].into_iter().collect();
    let formula: Basic = and_(&[
        condition.clone(),
        crate::core::relational::eq(&im_part, &Basic::zero())?,
    ]);
    Ok(holds_at(&formula, &rule, facts)? == Some(false))
}

fn filter_finite(elements: &[Basic], x: &Basic, condition: &Basic, facts: &Facts, formula: &Basic, real: bool) -> R {
    let mut kept: Vec<Basic> = Vec::new();
    let mut undecided: Vec<Basic> = Vec::new();
    for e in elements {
        if real && clearly_not_real(e)? {
            continue;
        }
        let rule: HashMap<Basic, Basic> = [(x.clone(), e.clone())].into_iter().collect();
        if !is_true(formula) {
            let at_point: Basic = normalize(&xreplace_evaluated(formula, &rule)?)?;
            if evaluate(&at_point, facts)? == Some(false) || numeric_root(&at_point)? == Some(false) {
                continue;
            }
        }
        let value: Option<bool> = holds_at(condition, &rule, facts)?;
        if value.is_none() && real && not_real_where_condition_fails(e, x, condition, facts)? {
            continue;
        }
        match value {
            Some(true) => {
                let mut duplicate: bool = false;
                for k in &kept {
                    if same_number(e, k)? {
                        duplicate = true;
                        break;
                    }
                }
                if !duplicate {
                    kept.push(e.clone());
                }
            }
            None => undecided.push(e.clone()),
            Some(false) => {}
        }
    }
    let mut result: Basic = if kept.is_empty() {
        empty()
    } else {
        st::finite_set(&kept)
    };
    if !undecided.is_empty() {
        result = st::union(&[result, condition_set(x, condition, &st::finite_set(&undecided))?])?;
    }
    Ok(result)
}

fn numeric_root(instance: &Basic) -> PolyResult<Truth> {
    let s = crate::settings::settings();
    if !s.numerical_checks {
        return Ok(None);
    }
    let equations: Vec<Basic> = conjuncts(instance)
        .into_iter()
        .filter(|c| rel(c) == Some(RelOp::Eq))
        .collect();
    if equations.is_empty() || equations.iter().any(|c| !c.free_symbols().is_empty()) {
        return Ok(None);
    }
    for c in &equations {
        let residual: Basic = sub(&c.args()[0], &c.args()[1])?;
        let Some(value) = crate::numeric::reliable_value(&residual, s.precision, &[]) else {
            return Ok(None);
        };
        let (re, im) = crate::core::evalf::parts(&value);
        let (re, im) = (
            crate::assumptions::ivarith::real_to_rat(&re),
            crate::assumptions::ivarith::real_to_rat(&im),
        );
        // Abs(residual) > 10**-(2*digits/3), with Abs of the float value
        let bound: Rat = tiny(s.precision);
        if &re * &re + &im * &im > &bound * &bound {
            return Ok(Some(false));
        }
    }
    Ok(Some(true))
}

fn restrict(result: &Basic, x: &Basic, condition: &Basic, facts: &Facts, formula: &Basic, real: bool) -> R {
    match result.head() {
        Head::FiniteSet => return filter_finite(result.args(), x, condition, facts, formula, real),
        Head::Union => {
            let parts: Vec<Basic> = result
                .args()
                .iter()
                .map(|a| restrict(a, x, condition, facts, formula, real))
                .collect::<PolyResult<_>>()?;
            return st::union(&parts);
        }
        Head::Intersection => {
            let parts: Vec<Basic> = result
                .args()
                .iter()
                .map(|a| restrict(a, x, condition, facts, formula, real))
                .collect::<PolyResult<_>>()?;
            return intersect(&parts);
        }
        _ => {}
    }
    if is_true(condition) || matches!(result.head(), Head::EmptySet) {
        return Ok(result.clone());
    }
    if condition.free_symbols().iter().all(|s| s == x) && facts.real.contains(x) {
        if let Some(polynomial) = to_polynomial(condition, &facts.real)? {
            return intersect(&[result.clone(), radicals(&solution_set(&polynomial, x)?)?]);
        }
    }
    condition_set(x, condition, result)
}

fn solution_set(formula: &Basic, x: &Basic) -> R {
    crate::polys::cad::qe::solution_set(formula, x, &[], None, true)
}

fn radicals(result: &Basic) -> R {
    crate::polys::roots::in_radicals(result)
}

fn univariate(formula: &Basic, x: &Basic, facts: &Facts, domain: Option<&Basic>) -> R {
    let dom: Basic = domain_set(facts, std::slice::from_ref(x), domain)?;
    let on_x: Basic = and_(
        &facts
            .conjuncts
            .iter()
            .filter(|c| {
                c.free_symbols().contains(x)
                    && !(matches!(c.head(), Head::Contains) && c.args()[0] == *x && c.args()[1] == dom)
            })
            .cloned()
            .collect::<Vec<Basic>>(),
    );
    let others: Basic = and_(
        &facts
            .conjuncts
            .iter()
            .filter(|c| !c.free_symbols().contains(x))
            .cloned()
            .collect::<Vec<Basic>>(),
    );
    let owned: Facts;
    let mut facts: &Facts = facts;
    if !is_true(&others) && !others.free_symbols().is_empty() {
        owned = Facts::new(std::slice::from_ref(&others), None, &[])?;
        facts = &owned;
    }
    let parameters: BTreeSet<Basic> = formula.free_symbols().into_iter().filter(|s| s != x).collect();
    let polynomial_on_x: Option<Basic> = if on_x.free_symbols().iter().all(|s| s == x) {
        to_polynomial(&on_x, &facts.real)?
    } else {
        None
    };
    let polynomial_formula: Option<Basic> = if parameters.is_empty() && facts.real.contains(x) {
        to_polynomial(formula, &facts.real)?
    } else {
        None
    };
    if let (Some(pox), Some(pf)) = (&polynomial_on_x, &polynomial_formula) {
        let real: Basic = radicals(&solution_set(&and_(&[pf.clone(), pox.clone()]), x)?)?;
        if matches!(dom.head(), Head::Reals) {
            return Ok(real);
        }
        return intersect(&[real, dom]);
    }
    if matches!(dom.head(), Head::Reals) {
        if let Some(pox) = &polynomial_on_x {
            let facts_: Facts = facts.clone();
            let solver = |f: &Basic, v: &Basic| -> PolyResult<Option<Basic>> {
                Ok(Some(univariate(f, v, &facts_, Some(&reals()))?))
            };
            let decide = |c: &Basic| -> PolyResult<Option<bool>> { evaluate(&normalize(c)?, &facts_) };
            let transcendental: Option<Basic> = crate::timeout::attempt_within(timeout(), || {
                crate::solvers::transcendental::solve_transcendental(formula, x, Some(&solver), Some(&decide), true)
            })?
            .flatten();
            if let Some(mut t) = transcendental {
                if !is_true(pox) {
                    t = intersect(&[t, radicals(&solution_set(pox, x)?)?])?;
                }
                return restrict(&t, x, &true_(), facts, formula, true);
            }
        }
    }
    let refiner = crate::assumptions::refine::Refiner::new(facts.clone());
    let mut forward: HashMap<Basic, Basic> = HashMap::new();
    let mut back_symbols: HashMap<Basic, Basic> = HashMap::new();
    for parameter in sorted_symbols(&parameters) {
        if let Some(d) = refiner.dummy_for(&parameter)? {
            forward.insert(parameter.clone(), d.clone());
            back_symbols.insert(d, parameter);
        }
    }
    let abstracted: Basic = xreplace_evaluated(formula, &forward)?;
    let mut condition: Basic = on_x.clone();
    let mut solving_domain: Basic = dom.clone();
    if !parameters.is_empty() && !matches!(dom.head(), Head::Complexes) {
        let ordered = |op: Option<RelOp>| matches!(op, Some(RelOp::Gt | RelOp::Lt | RelOp::Ge | RelOp::Le));
        let ordered_condition: bool = conjuncts(&condition).iter().any(|c| ordered(rel(c)));
        let ordered_formula: bool = formula.preorder().iter().any(|a| ordered(rel(a)));
        if matches!(dom.head(), Head::Reals) && ordered_formula {
            solving_domain = reals();
        } else {
            solving_domain = complexes();
            if !(matches!(dom.head(), Head::Reals) && ordered_condition) {
                condition = and_(&[membership(x, &dom)?, condition]);
            }
        }
    }
    let real_dom: bool = is_subset_reals(&dom)?;
    if parameters.is_empty() && real_dom {
        let periodic: Option<Basic> =
            crate::timeout::attempt_within(timeout(), || periodic_(formula, x, facts, &dom))?.flatten();
        if let Some(p) = periodic {
            return restrict(&p, x, &condition, facts, formula, true);
        }
    }
    let result: Basic = solveset_(&abstracted, x, &solving_domain)?;
    let result: Basic = xreplace_evaluated(&result, &back_symbols)?;
    let restricted: Basic = restrict(&result, x, &condition, facts, formula, real_dom)?;
    let has_condition_set: bool = restricted
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::ConditionSet));
    if parameters.is_empty() && has_condition_set && real_dom {
        let isolated: Option<Basic> =
            crate::timeout::attempt_within(timeout(), || isolated_(formula, x, facts, &dom))?.flatten();
        if let Some(i) = isolated {
            return Ok(i);
        }
    }
    Ok(restricted)
}

fn period_of(formula: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let mut period: Option<Basic> = None;
    let mut seen: Vec<Basic> = Vec::new();
    for atom in formula.preorder() {
        if rel(&atom).is_none() || seen.contains(&atom) {
            continue;
        }
        seen.push(atom.clone());
        let difference: Basic = sub(&atom.args()[0], &atom.args()[1])?;
        if !difference.free_symbols().contains(x) {
            continue;
        }
        let value: Option<Basic> =
            crate::timeout::attempt_within(timeout(), || crate::calculus::util::periodicity(&difference, x, false))?
                .flatten();
        let Some(value) = value else { return Ok(None) };
        if is(&value, Predicate::positive) != Some(true) || is(&value, Predicate::finite) != Some(true) {
            return Ok(None);
        }
        if !value.free_symbols().is_empty() {
            return Ok(None);
        }
        period = match period {
            None => Some(value),
            Some(p) => match common_period(&p, &value)? {
                Some(c) => Some(c),
                None => return Ok(None),
            },
        };
    }
    Ok(period)
}

fn common_period(first: &Basic, second: &Basic) -> PolyResult<Option<Basic>> {
    let ratio: Basic = crate::simplify::simplify::simplify(
        &crate::core::mul::div(first, second)?,
        &crate::simplify::simplify::Options::default(),
    )?;
    if !ratio.is_rational_class() {
        return Ok(None);
    }
    let q: Int = Int::from(crate::numbers::denom(ratio.as_rational().expect("rational")));
    Ok(Some(crate::core::mul::mul(&[
        first.clone(),
        Basic::from_rational(Rat::from(q)),
    ])?))
}

fn region_of(x: &Basic, facts: &Facts, dom: &Basic) -> R {
    let mut described: Option<Basic> =
        crate::assumptions::analysis::domain_of(x, &facts.with_reals(std::slice::from_ref(x))?)?;
    if described.is_none() {
        let relations: Vec<Basic> = facts
            .conjuncts
            .iter()
            .filter(|c| rel(c).is_some() && c.free_symbols().iter().all(|s| s == x))
            .cloned()
            .collect();
        let restricted: Facts = Facts::new(&relations, Some(&reals()), std::slice::from_ref(x))?;
        described = crate::assumptions::analysis::domain_of(x, &restricted.with_reals(std::slice::from_ref(x))?)?;
    }
    match described {
        None => Ok(dom.clone()),
        Some(d) => intersect(&[d, dom.clone()]),
    }
}

fn has_rel(formula: &Basic, ops: &[RelOp]) -> bool {
    formula
        .preorder()
        .iter()
        .any(|n| rel(n).is_some_and(|o| ops.contains(&o)))
}

fn periodic_(formula: &Basic, x: &Basic, facts: &Facts, dom: &Basic) -> PolyResult<Option<Basic>> {
    let only_x: bool = formula.free_symbols().len() == 1 && formula.free_symbols().contains(x);
    if !only_x || !has_rel(formula, &[RelOp::Gt, RelOp::Lt, RelOp::Ge, RelOp::Le]) {
        return Ok(None);
    }
    if has_rel(formula, &[RelOp::Eq, RelOp::Ne]) {
        return Ok(None);
    }
    let Some(period) = period_of(formula, x)? else {
        return Ok(None);
    };
    let window: Basic = st::interval(&int(0), &period, false, true)?;
    let Some(base) = base_over_period(formula, x, &period)? else {
        return Ok(None);
    };
    let region: Basic = region_of(x, facts, dom)?;
    if matches!(base.head(), Head::EmptySet) || matches!(region.head(), Head::EmptySet) {
        return Ok(Some(empty()));
    }
    if base == window {
        return Ok(Some(region));
    }
    let Some(tiles) = tiles_of(crate::assumptions::analysis::pieces(&region).as_deref(), &period)? else {
        let m: Basic = crate::core::modulo::modulo(x, &period)?;
        return Ok(Some(condition_set(x, &membership(&m, &base)?, &region)?));
    };
    let mut copies: Vec<Basic> = Vec::new();
    for k in tiles {
        copies.push(translated(&base, &crate::core::mul::mul(&[int(k), period.clone()])?)?);
    }
    Ok(Some(intersect(&[st::union(&copies)?, region])?))
}

fn translated(s: &Basic, offset: &Basic) -> R {
    if *offset == int(0) {
        return Ok(s.clone());
    }
    let shift = |e: &Basic| crate::core::add::add(&[e.clone(), offset.clone()]);
    match s.head() {
        Head::Interval => {
            let (a, b, lo, hi) = st::interval_parts(s);
            st::interval(&shift(&a)?, &shift(&b)?, lo, hi)
        }
        Head::FiniteSet => Ok(st::finite_set(
            &s.args().iter().map(shift).collect::<PolyResult<Vec<Basic>>>()?,
        )),
        Head::Union => {
            let parts: Vec<Basic> = s
                .args()
                .iter()
                .map(|a| translated(a, offset))
                .collect::<PolyResult<_>>()?;
            st::union(&parts)
        }
        Head::EmptySet => Ok(s.clone()),
        _ => {
            let d: Basic = crate::core::symbol::Dummy::new("shift", &[]).0;
            crate::sets::fancysets::image_set(
                &Basic::new(Head::Tuple, vec![d.clone()]),
                &shift(&d)?,
                std::slice::from_ref(s),
            )
        }
    }
}

const TILES: i64 = 64;

fn ordering(point: &Basic) -> f64 {
    match crate::core::evalf::numeric_value(point, crate::numeric::dps_to_prec(20)) {
        Some(v) => {
            let (re, im) = crate::core::evalf::parts(&v);
            if im == crate::core::evalf::Real::ZERO {
                let f: f64 = re.to_f64().value();
                if f.is_finite() {
                    return f;
                }
            }
            if matches!(point.head(), Head::NegativeInfinity) {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        }
        None => {
            if matches!(point.head(), Head::NegativeInfinity) {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        }
    }
}

fn tiles_of(pieces: Option<&[Basic]>, period: &Basic) -> PolyResult<Option<std::ops::RangeInclusive<i64>>> {
    let Some(pieces) = pieces else { return Ok(None) };
    if pieces.is_empty() {
        return Ok(None);
    }
    let starts: Vec<Basic> = pieces.iter().map(|p| st::interval_parts(p).0).collect();
    let ends: Vec<Basic> = pieces.iter().map(|p| st::interval_parts(p).1).collect();
    let pick = |v: &[Basic], max: bool| -> Basic {
        let mut best: &Basic = &v[0];
        for p in v {
            let better: bool = if max {
                ordering(p) > ordering(best)
            } else {
                ordering(p) < ordering(best)
            };
            if better {
                best = p;
            }
        }
        best.clone()
    };
    let (low, high) = (pick(&starts, false), pick(&ends, true));
    if is(&low, Predicate::finite) != Some(true) || is(&high, Predicate::finite) != Some(true) {
        return Ok(None);
    }
    let floor = |e: &Basic| -> PolyResult<Basic> {
        crate::functions::elementary::integers::floor(&crate::core::mul::div(e, period)?)
    };
    let (first, last) = (floor(&low)?, floor(&high)?);
    let as_int = |e: &Basic| -> Option<i64> {
        if e.is_integer_value() {
            i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
        } else {
            None
        }
    };
    let (Some(f), Some(l)) = (as_int(&first), as_int(&last)) else {
        return Ok(None);
    };
    if l - f + 1 > TILES {
        return Ok(None);
    }
    Ok(Some(f..=l))
}

fn base_over_period(formula: &Basic, x: &Basic, period: &Basic) -> PolyResult<Option<Basic>> {
    let window: Basic = st::interval(&int(0), period, false, true)?;
    let facts: Facts = Facts::new(&[], Some(&reals()), std::slice::from_ref(x))?;
    let full: Basic = st::interval(&int(0), period, false, false)?;
    let isolated: Option<Basic> =
        crate::timeout::attempt_within(timeout(), || isolated_(formula, x, &facts, &full))?.flatten();
    if let Some(i) = isolated {
        return Ok(Some(intersect(&[i, window])?));
    }
    let solved: Basic = solveset_(formula, x, &window)?;
    if solved.preorder().iter().any(|n| matches!(n.head(), Head::ConditionSet)) {
        return Ok(None);
    }
    Ok(Some(solved))
}

fn isolated_(formula: &Basic, x: &Basic, facts: &Facts, dom: &Basic) -> PolyResult<Option<Basic>> {
    use crate::solvers::isolation::{compare_points, isolate_real_roots, point_between};
    let only_x: bool = formula.free_symbols().len() == 1 && formula.free_symbols().contains(x);
    let Some(op) = rel(formula) else { return Ok(None) };
    if !only_x {
        return Ok(None);
    }
    let region: Basic = region_of(x, facts, dom)?;
    let Some(pieces) = crate::assumptions::analysis::pieces(&region) else {
        return Ok(None);
    };
    let f: Basic = sub(&formula.args()[0], &formula.args()[1])?;
    let Some(roots) = isolate_real_roots(&f, x, &region, 4)? else {
        return Ok(None);
    };
    if op == RelOp::Eq {
        return Ok(Some(if roots.is_empty() {
            empty()
        } else {
            st::finite_set(&roots)
        }));
    }
    if op == RelOp::Ne {
        return Ok(Some(if roots.is_empty() {
            region
        } else {
            evaluated_set(&st::complement(&region, &st::finite_set(&roots))?)?
        }));
    }
    let strict: bool = matches!(op, RelOp::Gt | RelOp::Lt);
    let wanted: i32 = if matches!(op, RelOp::Gt | RelOp::Ge) { 1 } else { -1 };
    let mut parts: Vec<Basic> = Vec::new();
    for piece in &pieces {
        let (a, b, left_open, right_open) = st::interval_parts(piece);
        let mut inner: Vec<Basic> = Vec::new();
        for r in &roots {
            if compare_points(r, &a)? == Some(1) && compare_points(&b, r)? == Some(1) {
                inner.push(r.clone());
            }
        }
        let mut points: Vec<Basic> = vec![a.clone()];
        points.extend(inner.iter().cloned());
        points.push(b.clone());
        for w in points.windows(2) {
            let Some(sample) = point_between(&w[0], &w[1])? else {
                return Ok(None);
            };
            let Some(s) = crate::assumptions::analysis::certified_sign(&crate::core::subs::subs(&f, x, &sample)?)?
            else {
                return Ok(None);
            };
            if s == wanted {
                parts.push(st::interval(&w[0], &w[1], true, true)?);
            }
        }
        if !strict {
            for r in &inner {
                parts.push(st::finite_set(std::slice::from_ref(r)));
            }
        }
        for (closed, point) in [(!left_open, &a), (!right_open, &b)] {
            if closed && is(point, Predicate::finite) == Some(true) {
                let s: Option<i32> =
                    crate::assumptions::analysis::certified_sign(&crate::core::subs::subs(&f, x, point)?)?;
                if s == Some(wanted) || (s == Some(0) && !strict) {
                    parts.push(st::finite_set(std::slice::from_ref(point)));
                }
            }
        }
    }
    Ok(Some(if parts.is_empty() { empty() } else { st::union(&parts)? }))
}

fn tuple(v: &[Basic]) -> Basic {
    Basic::new(Head::Tuple, v.to_vec())
}

fn multivariate(statements: &[Basic], symbols: &[Basic], facts: &Facts, domain: Option<&Basic>) -> R {
    let mut equations: Vec<Basic> = Vec::new();
    let mut conditions: Vec<Basic> = Vec::new();
    for s in statements {
        for c in conjuncts(&normalize(s)?) {
            if rel(&c) == Some(RelOp::Eq) {
                equations.push(sub(&c.args()[0], &c.args()[1])?);
            } else {
                conditions.push(c);
            }
        }
    }
    let unknown_set: BTreeSet<Basic> = symbols.iter().cloned().collect();
    conditions.extend(
        facts
            .conjuncts
            .iter()
            .filter(|c| c.free_symbols().iter().any(|s| unknown_set.contains(s)))
            .cloned(),
    );
    let condition: Basic = and_(&conditions);
    let dom: Basic = domain_set(facts, symbols, domain)?;
    let real: bool = is_subset_reals(&dom)?;
    let is_reals: bool = matches!(dom.head(), Head::Reals);
    if equations.is_empty() {
        let region: Option<Basic> = if real && is_reals {
            cylindrical(&equations, &conditions, symbols, facts)?
        } else {
            None
        };
        return region.ok_or_else(|| {
            PolyError::NotImplemented(
                "systems of inequalities in several unknowns are solved over the reals (domain=S.Reals, or real unknowns), for polynomial relations with rational coefficients, within the time limit"
                    .to_string(),
            )
        });
    }
    if domain.is_some_and(|d| matches!(d.head(), Head::Integers)) || symbols.iter().all(|s| facts.integer.contains(s)) {
        if let Some(integer) = integer_linear_system(&equations, &conditions, symbols, facts)? {
            return Ok(integer);
        }
    }
    let mut solutions: Option<Basic> = by_regular_chains(&equations, symbols, real, true)?;
    if solutions.is_none() && real && is_reals {
        if let Some(region) = cylindrical(&equations, &conditions, symbols, facts)? {
            return Ok(region);
        }
    }
    let mut unsolved: bool = false;
    let unchecked: bool = solutions.is_none();
    if solutions.is_none() {
        solutions = nonlinsolve(&equations, symbols)?;
        unsolved = solutions.is_none();
    }
    let refuted_now: bool = match &solutions {
        Some(s) if unchecked && matches!(s.head(), Head::PointSet) => refuted(s, &equations, symbols)?,
        _ => false,
    };
    if solutions.is_none() || refuted_now {
        solutions = by_regular_chains(&equations, symbols, real, false)?;
        if solutions.is_none() {
            if unsolved {
                return Err(PolyError::NotImplemented("the system cannot be solved".to_string()));
            }
            let mut all: Vec<Basic> = vec![condition.clone()];
            for e in &equations {
                all.push(crate::core::relational::eq(e, &int(0))?);
            }
            let space: Basic = st::product_set(&vec![dom.clone(); symbols.len()])?;
            return condition_set(&tuple(symbols), &and_(&all), &space);
        }
    }
    let solutions: Basic = solutions.expect("solutions");
    if !matches!(solutions.head(), Head::PointSet) {
        return if is_true(&condition) {
            Ok(solutions)
        } else {
            condition_set(&tuple(symbols), &condition, &solutions)
        };
    }
    let mut kept: Vec<Basic> = Vec::new();
    let mut undecided: Vec<Basic> = Vec::new();
    let complex_dom: bool = matches!(dom.head(), Head::Complexes);
    for point in solutions.args() {
        let values: HashMap<Basic, Basic> = symbols.iter().cloned().zip(point.args().iter().cloned()).collect();
        let membership_: Basic = if complex_dom {
            true_()
        } else {
            and_(
                &point
                    .args()
                    .iter()
                    .map(|v| membership(v, &dom))
                    .collect::<PolyResult<Vec<Basic>>>()?,
            )
        };
        match holds_at(&and_(&[condition.clone(), membership_]), &values, facts)? {
            Some(true) => kept.push(point.clone()),
            None => undecided.push(point.clone()),
            Some(false) => {}
        }
    }
    let mut result: Basic = if kept.is_empty() {
        empty()
    } else {
        st::point_set(&kept)?
    };
    if !undecided.is_empty() {
        let inside: Basic = if complex_dom {
            true_()
        } else {
            and_(
                &symbols
                    .iter()
                    .map(|s| membership(s, &dom))
                    .collect::<PolyResult<Vec<Basic>>>()?,
            )
        };
        let cs: Basic = condition_set(
            &tuple(symbols),
            &and_(&[condition, inside]),
            &st::point_set(&undecided)?,
        )?;
        result = st::union(&[result, cs])?;
    }
    Ok(result)
}

fn nonlinsolve(equations: &[Basic], symbols: &[Basic]) -> PolyResult<Option<Basic>> {
    let found =
        crate::timeout::attempt_within(timeout(), || crate::solvers::solvers::solve_system(equations, symbols))?
            .flatten();
    let Some(found) = found else { return Ok(None) };
    let points: Vec<Basic> = found
        .iter()
        .map(|solution| {
            tuple(
                &symbols
                    .iter()
                    .map(|s| {
                        solution
                            .iter()
                            .find(|(k, _)| k == s)
                            .map(|(_, v)| v.clone())
                            .unwrap_or_else(|| s.clone())
                    })
                    .collect::<Vec<Basic>>(),
            )
        })
        .collect();
    Ok(Some(if points.is_empty() {
        empty()
    } else {
        st::point_set(&points)?
    }))
}

fn informative(c: &Basic) -> bool {
    !(matches!(c.head(), Head::Contains) && matches!(c.args()[1].head(), Head::Reals))
}

fn parametric(
    statements: &[Basic],
    symbols: &[Basic],
    facts: &Facts,
    domain: Option<&Basic>,
) -> PolyResult<Option<Basic>> {
    let mut relations: Vec<Basic> = Vec::new();
    for s in statements {
        relations.extend(conjuncts(&normalize(s)?));
    }
    let mut found: BTreeSet<Basic> = BTreeSet::new();
    for c in &relations {
        found.extend(c.free_symbols());
    }
    let unknown_set: BTreeSet<Basic> = symbols.iter().cloned().collect();
    let parameters: Vec<Basic> = sorted_symbols(&found.difference(&unknown_set).cloned().collect());
    if parameters.is_empty() {
        return Ok(None);
    }
    let unknown: Basic = if symbols.len() == 1 {
        symbols[0].clone()
    } else {
        tuple(symbols)
    };
    let dom: Basic = domain_set(facts, symbols, domain)?;
    let hypotheses: Vec<Basic> = facts.conjuncts.iter().filter(|c| informative(c)).cloned().collect();
    let mut pieces: Vec<Basic> = Vec::new();
    if matches!(dom.head(), Head::Reals) {
        let allowed: BTreeSet<Basic> = parameters.iter().cloned().chain(symbols.iter().cloned()).collect();
        let mut parts: Vec<Basic> = relations.clone();
        parts.extend(
            hypotheses
                .iter()
                .filter(|c| c.free_symbols().iter().all(|s| allowed.contains(s)))
                .cloned(),
        );
        let system: Basic = and_(&parts);
        let discussed = crate::timeout::attempt_within(timeout(), || {
            crate::polys::cad::cylindrical::cylindrical_cases(&system, &parameters, symbols, None, true)
        })?;
        let Some(discussed) = discussed else { return Ok(None) };
        for (condition, solutions) in discussed {
            pieces.push(under(&unknown, &condition, &solutions, facts)?);
        }
        return Ok(Some(if pieces.is_empty() {
            empty()
        } else {
            st::union(&pieces)?
        }));
    }
    if !matches!(dom.head(), Head::Complexes)
        || !relations.iter().all(|c| matches!(rel(c), Some(RelOp::Eq | RelOp::Ne)))
    {
        return Ok(None);
    }
    let mut equations: Vec<Basic> = Vec::new();
    let mut inequations: Vec<Basic> = Vec::new();
    for c in &relations {
        let d: Basic = sub(&c.args()[0], &c.args()[1])?;
        if rel(c) == Some(RelOp::Eq) {
            equations.push(d);
        } else {
            inequations.push(d);
        }
    }
    if equations.is_empty() {
        return Ok(None);
    }
    let found_cases = crate::timeout::attempt_within(timeout(), || {
        crate::solvers::parametric::parametric_cases(&equations, symbols, Some(&parameters), &inequations)
    })?;
    let Some(found_cases) = found_cases else {
        return Ok(None);
    };
    let space: Basic = if symbols.len() == 1 {
        complexes()
    } else {
        st::product_set(&vec![complexes(); symbols.len()])?
    };
    for case in found_cases {
        let solutions: Basic = match &case.solutions {
            None => {
                let eqs: Vec<Basic> = case
                    .equations
                    .iter()
                    .map(|e| crate::core::relational::eq(e, &int(0)))
                    .collect::<PolyResult<_>>()?;
                condition_set(&unknown, &and_(&eqs), &space)?
            }
            Some(s) if symbols.len() == 1 => {
                let values: Vec<Basic> = s
                    .args()
                    .iter()
                    .filter(|p| matches!(p.head(), Head::Tuple))
                    .map(|p| p.args()[0].clone())
                    .collect();
                if values.contains(&symbols[0]) {
                    space.clone()
                } else {
                    st::finite_set(&values)
                }
            }
            Some(s) => s.clone(),
        };
        let mut parts: Vec<Basic> = vec![case.condition.clone()];
        parts.extend(
            hypotheses
                .iter()
                .filter(|c| c.free_symbols().iter().any(|s| unknown_set.contains(s)))
                .cloned(),
        );
        pieces.push(under(&unknown, &and_(&parts), &solutions, facts)?);
    }
    Ok(Some(if pieces.is_empty() {
        empty()
    } else {
        st::union(&pieces)?
    }))
}

fn under(unknown: &Basic, condition: &Basic, solutions: &Basic, facts: &Facts) -> R {
    let holds: Truth = if condition.free_symbols().is_empty() {
        None
    } else {
        evaluate(&normalize(condition)?, facts)?
    };
    if is_true(condition) || holds == Some(true) {
        return Ok(solutions.clone());
    }
    if is_false(condition) || holds == Some(false) {
        return Ok(empty());
    }
    condition_set(unknown, condition, solutions)
}

fn cylindrical(
    equations: &[Basic],
    conditions: &[Basic],
    symbols: &[Basic],
    facts: &Facts,
) -> PolyResult<Option<Basic>> {
    use crate::polys::cad::cylindrical::{cylindrical_formula, cylindrical_set};
    let mut parts: Vec<Basic> = conditions.iter().filter(|c| informative(c)).cloned().collect();
    for e in equations {
        parts.push(crate::core::relational::eq(e, &int(0))?);
    }
    let system: Basic = and_(&parts);
    let unknown_set: BTreeSet<Basic> = symbols.iter().cloned().collect();
    let parameters: Vec<Basic> = sorted_symbols(&system.free_symbols().difference(&unknown_set).cloned().collect());
    let parameter_set: BTreeSet<Basic> = parameters.iter().cloned().collect();
    let hypotheses: Vec<Basic> = facts
        .conjuncts
        .iter()
        .filter(|c| {
            informative(c) && !c.free_symbols().is_empty() && c.free_symbols().iter().all(|s| parameter_set.contains(s))
        })
        .cloned()
        .collect();
    crate::timeout::attempt_within(timeout(), || {
        if parameters.is_empty() {
            return cylindrical_set(&system, symbols, None, true);
        }
        let mut all: Vec<Basic> = vec![system.clone()];
        all.extend(hypotheses.iter().cloned());
        let mut gens: Vec<Basic> = parameters.clone();
        gens.extend(symbols.iter().cloned());
        let formula: Basic = cylindrical_formula(&and_(&all), &gens, &[], None, true)?;
        if is_false(&formula) {
            return Ok(empty());
        }
        condition_set(
            &tuple(symbols),
            &formula,
            &st::product_set(&vec![reals(); symbols.len()])?,
        )
    })
}

fn complex_f64(e: &Basic) -> Option<(f64, f64)> {
    let v = crate::core::evalf::numeric_value(e, crate::numeric::dps_to_prec(15))?;
    let (a, b) = crate::core::evalf::parts(&v);
    Some((a.to_f64().value(), b.to_f64().value()))
}

fn refuted(solutions: &Basic, equations: &[Basic], symbols: &[Basic]) -> PolyResult<bool> {
    let mut rng = crate::random::PyRandom::from_int(&Int::ZERO);
    let found = crate::timeout::attempt_within(timeout(), || -> PolyResult<bool> {
        for point in solutions.args() {
            let values: HashMap<Basic, Basic> = symbols.iter().cloned().zip(point.args().iter().cloned()).collect();
            for e in equations {
                let residual: Basic = xreplace_evaluated(e, &values)?;
                let free: Vec<Basic> = sorted_symbols(&residual.free_symbols());
                let rounds: usize = if free.is_empty() { 1 } else { 2 };
                for _ in 0..rounds {
                    let mut sample: HashMap<Basic, Basic> = HashMap::new();
                    for s in &free {
                        let p: i64 = rng.randint(2, 40);
                        let q: i64 = rng.randint(2, 9);
                        sample.insert(
                            s.clone(),
                            Basic::from_rational(Rat::from(Int::from(p)) / Rat::from(Int::from(q))),
                        );
                    }
                    let expanded: Basic = crate::core::function::expand_method(e)?;
                    let mut scale: f64 = 0.0;
                    for term in crate::core::printing::ordered_terms(&crate::core::expr::terms_of(&expanded)) {
                        let t: Basic = xreplace_evaluated(&xreplace_evaluated(&term, &values)?, &sample)?;
                        let (a, b) = complex_f64(&t).ok_or_else(|| PolyError::Type("not a number".to_string()))?;
                        scale += a.hypot(b);
                    }
                    let r: Basic = xreplace_evaluated(&residual, &sample)?;
                    let (a, b) = complex_f64(&r).ok_or_else(|| PolyError::Type("not a number".to_string()))?;
                    if a.hypot(b) > 1e-8 * (1.0 + scale) {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    });
    match found {
        Ok(v) => Ok(v == Some(true)),
        Err(PolyError::ZeroDivision(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

fn by_regular_chains(equations: &[Basic], symbols: &[Basic], real: bool, isolated: bool) -> PolyResult<Option<Basic>> {
    use crate::core::power::pow;
    let chains = crate::timeout::attempt_within(timeout(), || {
        crate::polys::regularchains::regularchain::triangularize(equations, symbols, &[], false)
    })?;
    let Some(chains) = chains else { return Ok(None) };
    if isolated && chains.iter().any(|c| c.dimension() > 0) {
        return Ok(None);
    }
    let mut points: Vec<Basic> = Vec::new();
    for chain in &chains {
        if chain.dimension() == 0 {
            let found = crate::timeout::attempt_within(timeout(), || chain.solutions(real))?;
            let Some(found) = found else { return Ok(None) };
            for solution in found {
                points.push(tuple(
                    &symbols
                        .iter()
                        .map(|s| {
                            solution
                                .iter()
                                .find(|(k, _)| k == s)
                                .map(|(_, v)| v.clone())
                                .expect("a coordinate")
                        })
                        .collect::<Vec<Basic>>(),
                ));
            }
            continue;
        }
        let mut families: Vec<HashMap<Basic, Basic>> = vec![HashMap::new()];
        for (p, v) in chain.polys()?.iter().zip(chain.main_variables().iter()) {
            let poly = crate::polys::polytools::Poly::new(p, std::slice::from_ref(v))?;
            if poly.degree(0)? > 2 {
                return Ok(None);
            }
            let mut extended: Vec<HashMap<Basic, Basic>> = Vec::new();
            for family in &families {
                let coefficients: Vec<Basic> = poly
                    .all_coeffs()?
                    .iter()
                    .map(|c| xreplace_evaluated(c, family))
                    .collect::<PolyResult<_>>()?;
                let values: Vec<Basic> = if coefficients.len() == 2 {
                    vec![crate::polys::polytools::cancel(
                        &crate::core::mul::div(&crate::core::mul::neg(&coefficients[1])?, &coefficients[0])?,
                        &[],
                    )?]
                } else {
                    let (a, b, c) = (&coefficients[0], &coefficients[1], &coefficients[2]);
                    // sqrt(cancel(b**2 - 4*a*c)); (-b -+ root) / (2*a)
                    let disc: Basic = crate::polys::polytools::cancel(
                        &sub(
                            &pow(b, &int(2))?,
                            &crate::core::mul::mul(&[crate::core::mul::mul(&[int(4), a.clone()])?, c.clone()])?,
                        )?,
                        &[],
                    )?;
                    let root: Basic = pow(
                        &disc,
                        &Basic::from_rational(Rat::from(Int::ONE) / Rat::from(Int::from(2))),
                    )?;
                    let two_a: Basic = crate::core::mul::mul(&[int(2), a.clone()])?;
                    let minus_b: Basic = crate::core::mul::neg(b)?;
                    vec![
                        crate::core::mul::div(&sub(&minus_b, &root)?, &two_a)?,
                        crate::core::mul::div(&crate::core::add::add(&[minus_b, root])?, &two_a)?,
                    ]
                };
                for value in values {
                    let mut longer: HashMap<Basic, Basic> = family.clone();
                    longer.insert(v.clone(), value);
                    extended.push(longer);
                }
            }
            families = extended;
        }
        let found_: Vec<Basic> = families
            .iter()
            .map(|f| {
                tuple(
                    &symbols
                        .iter()
                        .map(|s| f.get(s).cloned().unwrap_or_else(|| s.clone()))
                        .collect::<Vec<Basic>>(),
                )
            })
            .collect();
        let candidates: Basic = st::point_set(&found_)?;
        if matches!(candidates.head(), Head::PointSet) && refuted(&candidates, equations, symbols)? {
            return Ok(None);
        }
        points.extend(found_);
    }
    Ok(Some(st::point_set(&points)?))
}

fn integer_linear_system(
    equations: &[Basic],
    conditions: &[Basic],
    symbols: &[Basic],
    facts: &Facts,
) -> PolyResult<Option<Basic>> {
    use crate::solvers::integers::{linear_diophantine_system, linear_system_pub, minimal_nonnegative_solutions};
    let Some((a, b)) = linear_system_pub(equations, symbols)? else {
        return Ok(None);
    };
    let Some((values, parameters)) = linear_diophantine_system(equations, symbols)? else {
        return Ok(Some(empty()));
    };
    let condition: Basic = and_(
        &conditions
            .iter()
            .filter(|c| {
                !(matches!(c.head(), Head::Contains)
                    && matches!(c.args()[1].head(), Head::Integers)
                    && symbols.contains(&c.args()[0]))
            })
            .cloned()
            .collect::<Vec<Basic>>(),
    );
    if parameters.is_empty() {
        let point: Basic = tuple(&values);
        let rule: HashMap<Basic, Basic> = symbols.iter().cloned().zip(values.iter().cloned()).collect();
        return Ok(Some(match holds_at(&condition, &rule, facts)? {
            Some(false) => empty(),
            Some(true) => st::point_set(&[point])?,
            None => condition_set(&tuple(symbols), &condition, &st::point_set(&[point])?)?,
        }));
    }
    let mut nonnegative: bool = true;
    for s in symbols {
        if evaluate(&crate::core::relational::ge(s, &int(0))?, facts)? != Some(true) {
            nonnegative = false;
            break;
        }
    }
    if nonnegative {
        let (particular, homogeneous) = match minimal_nonnegative_solutions(&a, &b, 10000) {
            Ok(v) => v,
            Err(PolyError::Arithmetic(_)) => (vec![], vec![vec![Int::ZERO]]),
            Err(e) => return Err(e),
        };
        if homogeneous.is_empty() {
            let mut kept: Vec<Basic> = Vec::new();
            for p in particular {
                let point: Vec<Basic> = p.iter().map(|c| Basic::from_rational(Rat::from(c.clone()))).collect();
                let rule: HashMap<Basic, Basic> = symbols.iter().cloned().zip(point.iter().cloned()).collect();
                if holds_at(&condition, &rule, facts)? != Some(false) {
                    kept.push(tuple(&point));
                }
            }
            return Ok(Some(if kept.is_empty() {
                empty()
            } else {
                st::point_set(&kept)?
            }));
        }
    }
    let base: Basic = if parameters.len() > 1 {
        st::product_set(&vec![Basic::atom(Head::Integers); parameters.len()])?
    } else {
        Basic::atom(Head::Integers)
    };
    let solutions: Basic = crate::sets::fancysets::image_set(&tuple(&parameters), &tuple(&values), &[base])?;
    if is_true(&condition) {
        return Ok(Some(solutions));
    }
    Ok(Some(condition_set(&tuple(symbols), &condition, &solutions)?))
}

/// Equations and inequalities solved under assumptions (`solve`).
pub fn solve(
    items: &[Basic],
    symbols: Option<&[Basic]>,
    assumptions: &[Basic],
    domain: Option<&Basic>,
    cases: bool,
) -> R {
    let statements: Vec<Basic> = items.iter().map(as_statement).collect::<PolyResult<_>>()?;
    let mut all_symbols: BTreeSet<Basic> = BTreeSet::new();
    for s in &statements {
        all_symbols.extend(s.free_symbols());
    }
    let unknowns: Vec<Basic> = match symbols {
        None => sorted_symbols(&all_symbols),
        Some(v) => v.to_vec(),
    };
    if unknowns.is_empty() {
        return Err(crate::error::value_error("no unknown to solve for".to_string()));
    }
    let mut given: BTreeSet<Basic> = all_symbols.clone();
    given.extend(unknowns.iter().cloned());
    let facts: Facts = match facts_of(assumptions, domain, &given.into_iter().collect::<Vec<Basic>>()) {
        Ok(f) => f,
        Err(PolyError::Value(_)) => return Ok(empty()),
        Err(e) => return Err(e),
    };
    if cases {
        if let Some(discussed) = parametric(&statements, &unknowns, &facts, domain)? {
            return Ok(discussed);
        }
    }
    if unknowns.len() == 1 {
        let x: &Basic = &unknowns[0];
        let formula: Basic = normalize(&and_(&statements))?;
        let found: Basic = univariate(&formula, x, &facts, domain)?;
        return respecting_flags(&found, &unknowns, &facts);
    }
    let found: Basic = multivariate(&statements, &unknowns, &facts, domain)?;
    respecting_flags(&found, &unknowns, &facts)
}

fn flags(x: &Basic) -> Basic {
    let mut given: Vec<(Predicate, bool)> = given_assumptions(x);
    given.sort_by_key(|(p, _)| p.name().to_string());
    let mut statements: Vec<Basic> = Vec::new();
    for (p, value) in given {
        let name: &str = p.name();
        let Some((_, arity)) = crate::core::ask::PREDICATES_OF_Q.iter().find(|(n, _)| *n == name) else {
            continue;
        };
        if *arity != 1 {
            continue;
        }
        let applied: Basic = crate::core::assume::applied(&crate::core::assume::q(name, 1), std::slice::from_ref(x));
        statements.push(if value {
            applied
        } else {
            Basic::new(Head::Not, vec![applied])
        });
    }
    and_(&statements)
}

fn respecting_flags(result: &Basic, unknowns: &[Basic], facts: &Facts) -> R {
    let all: Vec<Basic> = unknowns.iter().map(flags).collect();
    let f: Basic = and_(&all);
    if is_true(&f) {
        return Ok(result.clone());
    }
    drop_contradicting(result, unknowns, &f, facts)
}

fn contradicts(point: &Basic, unknowns: &[Basic], flags_: &Basic, facts: &Facts) -> PolyResult<bool> {
    if point.args().len() != unknowns.len() {
        return Ok(false);
    }
    let values: HashMap<Basic, Basic> = unknowns.iter().cloned().zip(point.args().iter().cloned()).collect();
    Ok(holds_at(flags_, &values, facts)? == Some(false))
}

fn drop_contradicting(result: &Basic, unknowns: &[Basic], flags_: &Basic, facts: &Facts) -> R {
    match result.head() {
        Head::FiniteSet if unknowns.len() == 1 => {
            let x: &Basic = &unknowns[0];
            let mut kept: Vec<Basic> = Vec::new();
            for e in result.args() {
                let rule: HashMap<Basic, Basic> = [(x.clone(), e.clone())].into_iter().collect();
                if holds_at(flags_, &rule, facts)? != Some(false) {
                    kept.push(e.clone());
                }
            }
            Ok(if kept.len() == result.args().len() {
                result.clone()
            } else {
                st::finite_set(&kept)
            })
        }
        Head::PointSet => {
            let mut points: Vec<Basic> = Vec::new();
            for p in result.args() {
                if !contradicts(p, unknowns, flags_, facts)? {
                    points.push(p.clone());
                }
            }
            if points.len() == result.args().len() {
                Ok(result.clone())
            } else {
                st::point_set(&points)
            }
        }
        Head::Union => {
            let parts: Vec<Basic> = result
                .args()
                .iter()
                .map(|a| drop_contradicting(a, unknowns, flags_, facts))
                .collect::<PolyResult<_>>()?;
            st::union(&parts)
        }
        Head::Intersection => {
            let parts: Vec<Basic> = result
                .args()
                .iter()
                .map(|a| drop_contradicting(a, unknowns, flags_, facts))
                .collect::<PolyResult<_>>()?;
            intersect(&parts)
        }
        Head::Complement => {
            let (a, b) = (&result.args()[0], &result.args()[1]);
            let kept_a: Basic = drop_contradicting(a, unknowns, flags_, facts)?;
            if kept_a.ptr_eq(a) || kept_a == *a {
                Ok(result.clone())
            } else {
                evaluated_set(&st::complement(&kept_a, b)?)
            }
        }
        Head::ConditionSet => {
            let base: Basic = drop_contradicting(&result.args()[2], unknowns, flags_, facts)?;
            if base == result.args()[2] {
                Ok(result.clone())
            } else {
                condition_set(&result.args()[0], &result.args()[1], &base)
            }
        }
        _ => Ok(result.clone()),
    }
}
