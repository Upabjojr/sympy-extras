//! Port of `sympy_extras/sets/sets.py`: the sets of the core as nodes
//! ([`Head::EmptySet`], [`Head::UniversalSet`], [`Head::Reals`],
//! [`Head::Interval`], [`Head::FiniteSet`], [`Head::Union`],
//! [`Head::Intersection`], [`Head::Complement`], [`Head::ProductSet`],
//! [`Head::PointSet`], and the boolean [`Head::Contains`]), their methods
//! (membership, complements, subsets, emptiness, finiteness, infimum and
//! supremum, the condition of membership) and the evaluating functions
//! ([`interval`], [`finite_set`], [`union`], [`intersection`],
//! [`complement`], [`contains`], [`product_set`], [`point_set`]), with the
//! rules of SymPy's handlers.

use crate::core::add::sub;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::evalf::{numeric_value, parts};
use crate::core::expr::{is, is_comparable, is_true};
use crate::core::logic::{and_, boolean, false_, not_, or_, true_, truth};
use crate::core::relational::{eq, ge, le, lt, rel};
use crate::core::sorting::{ordered, sort_key, SortKey};
use crate::error::{not_ported, value_error, PolyError, PolyResult};
use crate::functions::elementary::miscellaneous::{max_, min_};

/// A fuzzy truth value.
pub type Truth = Option<bool>;

/// `fuzzy_and`.
pub fn fuzzy_and(values: impl IntoIterator<Item = Truth>) -> Truth {
    let mut rv: Truth = Some(true);
    for v in values {
        match v {
            None => {
                if rv == Some(true) {
                    rv = None;
                }
            }
            Some(false) => return Some(false),
            Some(true) => {}
        }
    }
    rv
}

/// `fuzzy_not`.
pub fn fuzzy_not(v: Truth) -> Truth {
    v.map(|b| !b)
}

/// `fuzzy_or`.
pub fn fuzzy_or(values: impl IntoIterator<Item = Truth>) -> Truth {
    fuzzy_not(fuzzy_and(values.into_iter().map(fuzzy_not)))
}

/// `_fuzzy_bool` of a condition.
fn fuzzy_bool(b: &Basic) -> Truth {
    truth(b)
}

// ---------------------------------------------------------------------
// the nodes

pub fn empty_set() -> Basic {
    Basic::atom(Head::EmptySet)
}

pub fn universal_set() -> Basic {
    Basic::atom(Head::UniversalSet)
}

pub fn reals() -> Basic {
    Basic::atom(Head::Reals)
}

/// Whether the node is a set.
pub fn is_set(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::EmptySet
            | Head::UniversalSet
            | Head::Reals
            | Head::Interval
            | Head::FiniteSet
            | Head::Union
            | Head::Intersection
            | Head::Complement
            | Head::ProductSet
            | Head::PointSet
            | Head::ConditionSet
    )
}

/// The unevaluated `Interval(start, end, left_open, right_open)`.
pub fn interval_node(start: &Basic, end: &Basic, left_open: bool, right_open: bool) -> Basic {
    Basic::new(
        Head::Interval,
        vec![start.clone(), end.clone(), boolean(left_open), boolean(right_open)],
    )
}

/// `Interval(-oo, oo, True, True)`, equal to `Reals`.
pub fn whole_line() -> Basic {
    interval_node(&Basic::negative_oo(), &Basic::oo(), true, true)
}

/// Whether an interval is the whole line (`_is_whole_line`).
pub fn is_whole_line(e: &Basic) -> bool {
    matches!(e.head(), Head::Interval)
        && matches!(e.args()[0].head(), Head::NegativeInfinity)
        && matches!(e.args()[1].head(), Head::Infinity)
        && truth(&e.args()[2]) == Some(true)
        && truth(&e.args()[3]) == Some(true)
}

/// Whether the node is an interval (`Reals` is one).
fn is_interval(e: &Basic) -> bool {
    matches!(e.head(), Head::Interval | Head::Reals)
}

/// `(start, end, left_open, right_open)` of an interval (or of `Reals`).
pub fn interval_parts(e: &Basic) -> (Basic, Basic, bool, bool) {
    if matches!(e.head(), Head::Reals) {
        return (Basic::negative_oo(), Basic::oo(), true, true);
    }
    let a = e.args();
    (
        a[0].clone(),
        a[1].clone(),
        truth(&a[2]) == Some(true),
        truth(&a[3]) == Some(true),
    )
}

fn finite_node(elements: Vec<Basic>) -> Basic {
    Basic::new(Head::FiniteSet, elements)
}

fn dedupe(items: Vec<Basic>) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for i in items {
        if !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

/// The order of the sets of a union (`Set._infimum_key`): the infimum as a
/// float, `oo` when it is not a comparable number, the canonical key
/// breaking ties.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct InfimumKey(f64, SortKey);

fn infimum_key(s: &Basic) -> InfimumKey {
    let mut value: f64 = f64::INFINITY;
    if is_set(s) {
        let infimum: Basic = inf(s).unwrap_or_else(|_| Basic::oo());
        match infimum.head() {
            Head::Infinity => value = f64::INFINITY,
            Head::NegativeInfinity => value = f64::NEG_INFINITY,
            _ if is_comparable(&infimum) => {
                if let Some(v) = numeric_value(&infimum, 53) {
                    value = parts(&v).0.to_f64().value();
                }
            }
            _ => {}
        }
    }
    InfimumKey(value, sort_key(s))
}

fn sorted_sets(sets: Vec<Basic>) -> Vec<Basic> {
    let mut keyed: Vec<(InfimumKey, Basic)> = sets.into_iter().map(|s| (infimum_key(&s), s)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().map(|(_, s)| s).collect()
}

/// The unevaluated `Union(*sets)`: deduplicated, in the order of the
/// infima.
pub fn union_node(sets: Vec<Basic>) -> Basic {
    Basic::new(Head::Union, sorted_sets(dedupe(sets)))
}

/// The unevaluated `Intersection(*sets)`.
pub fn intersection_node(sets: Vec<Basic>) -> Basic {
    Basic::new(Head::Intersection, sorted_sets(dedupe(sets)))
}

pub fn complement_node(a: &Basic, b: &Basic) -> Basic {
    Basic::new(Head::Complement, vec![a.clone(), b.clone()])
}

pub fn contains_node(x: &Basic, s: &Basic) -> Basic {
    Basic::new(Head::Contains, vec![x.clone(), s.clone()])
}

/// The sets among the arguments (`sets`).
fn sets_of(s: &Basic) -> Vec<Basic> {
    s.args().iter().filter(|a| is_set(a)).cloned().collect()
}

// ---------------------------------------------------------------------
// the methods

/// Whether `x` is an element: `true`, `false`, a condition or an
/// unevaluated `Contains` (`Set.contains`).
pub fn contains(s: &Basic, x: &Basic) -> PolyResult<Basic> {
    match contains_rule(s, x)? {
        Some(c) => Ok(c),
        None => Ok(contains_node(x, s)),
    }
}

fn contains_true(s: &Basic, x: &Basic) -> PolyResult<bool> {
    Ok(truth(&contains(s, x)?) == Some(true))
}

/// `_contains`: `None` when there is no rule.
fn contains_rule(s: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    Ok(match s.head() {
        Head::EmptySet => Some(false_()),
        Head::UniversalSet => Some(true_()),
        Head::Interval | Head::Reals => {
            if matches!(x.head(), Head::NaN) || crate::core::expr::is_false(x, Predicate::real) || x.has(&Basic::zoo())
            {
                return Ok(Some(false_()));
            }
            if is_interval(s) && is_whole_line(&canonical_interval(s)) {
                if let Some(v) = is(x, Predicate::real) {
                    return Ok(Some(boolean(v)));
                }
            }
            Some(as_relational(s, x)?)
        }
        Head::FiniteSet => {
            if s.args().contains(x) {
                return Ok(Some(true_()));
            }
            let mut terms: Vec<Basic> = Vec::new();
            for e in s.args() {
                terms.push(eq(e, x)?);
            }
            Some(or_(&terms))
        }
        Head::Union => {
            let mut terms: Vec<Basic> = Vec::new();
            for t in sets_of(s) {
                terms.push(contains(&t, x)?);
            }
            Some(or_(&terms))
        }
        Head::Intersection => {
            let mut terms: Vec<Basic> = Vec::new();
            for t in sets_of(s) {
                terms.push(contains(&t, x)?);
            }
            Some(and_(&terms))
        }
        Head::Complement => {
            let (a, b) = (&s.args()[0], &s.args()[1]);
            Some(and_(&[contains(a, x)?, not_(&contains(b, x)?)]))
        }
        Head::ProductSet => {
            // the elements are tuples: a symbol may stand for one
            if x.is_symbol() {
                None
            } else {
                Some(false_())
            }
        }
        Head::PointSet => Some(false_()),
        Head::ConditionSet => Some(super::conditionset::condition_set_contains(s, x)?),
        _ => return Err(not_ported("_contains of this set")),
    })
}

fn canonical_interval(s: &Basic) -> Basic {
    if matches!(s.head(), Head::Reals) {
        whole_line()
    } else {
        s.clone()
    }
}

/// The condition that `x` is an element (`as_relational`).
pub fn as_relational(s: &Basic, x: &Basic) -> PolyResult<Basic> {
    Ok(match s.head() {
        Head::EmptySet => false_(),
        Head::UniversalSet => true_(),
        Head::Interval | Head::Reals => {
            let (start, end, lo, ro) = interval_parts(s);
            let right: Basic = if ro { lt(x, &end)? } else { le(x, &end)? };
            let left: Basic = if lo { lt(&start, x)? } else { le(&start, x)? };
            and_(&[left, right])
        }
        Head::FiniteSet => {
            let terms: Vec<Basic> = s
                .args()
                .iter()
                .map(|e| rel(crate::core::basic::RelOp::Eq, x, e))
                .collect();
            or_(&terms)
        }
        Head::Union => {
            let parts_: Vec<Basic> = sets_of(s);
            if parts_.len() == 2 && parts_.iter().all(is_interval) {
                let (a, b) = (&parts_[0], &parts_[1]);
                let (a_start, a_end, a_lo, a_ro) = interval_parts(a);
                let (b_start, b_end, b_lo, b_ro) = interval_parts(b);
                if a_end == b_start && a_ro && b_lo {
                    let mincond: Basic = if a_lo {
                        crate::core::relational::gt(x, &a_start)?
                    } else {
                        ge(x, &a_start)?
                    };
                    let maxcond: Basic = if b_ro { lt(x, &b_end)? } else { le(x, &b_end)? };
                    return Ok(and_(&[rel(crate::core::basic::RelOp::Ne, x, &a_end), mincond, maxcond]));
                }
            }
            let mut terms: Vec<Basic> = Vec::new();
            for t in parts_ {
                terms.push(as_relational(&t, x)?);
            }
            or_(&terms)
        }
        Head::Intersection => {
            let mut terms: Vec<Basic> = Vec::new();
            for t in sets_of(s) {
                terms.push(as_relational(&t, x)?);
            }
            and_(&terms)
        }
        Head::Complement => and_(&[as_relational(&s.args()[0], x)?, not_(&as_relational(&s.args()[1], x)?)]),
        Head::ConditionSet => super::conditionset::condition_set_as_relational(s, x)?,
        _ => return Err(not_ported("as_relational of this set")),
    })
}

/// The infimum (`inf`).
pub fn inf(s: &Basic) -> PolyResult<Basic> {
    match s.head() {
        Head::Interval | Head::Reals => Ok(interval_parts(s).0),
        Head::FiniteSet => min_(s.args()),
        Head::Union => {
            let mut infima: Vec<Basic> = Vec::new();
            for t in sets_of(s) {
                infima.push(inf(&t)?);
            }
            min_(&infima)
        }
        _ => Err(PolyError::NotImplemented(format!("({})._inf", s))),
    }
}

/// The supremum (`sup`).
pub fn sup(s: &Basic) -> PolyResult<Basic> {
    match s.head() {
        Head::Interval | Head::Reals => Ok(interval_parts(s).1),
        Head::FiniteSet => max_(s.args()),
        Head::Union => {
            let mut suprema: Vec<Basic> = Vec::new();
            for t in sets_of(s) {
                suprema.push(sup(&t)?);
            }
            max_(&suprema)
        }
        _ => Err(PolyError::NotImplemented(format!("({})._sup", s))),
    }
}

/// The measure of an interval or a finite set (`measure`).
pub fn measure(s: &Basic) -> PolyResult<Basic> {
    match s.head() {
        Head::Interval | Head::Reals => {
            let (start, end, _, _) = interval_parts(s);
            sub(&end, &start)
        }
        Head::FiniteSet => Ok(Basic::zero()),
        _ => Err(not_ported("the measure of this set")),
    }
}

/// `is_empty`.
pub fn is_empty(s: &Basic) -> PolyResult<Truth> {
    Ok(match s.head() {
        Head::EmptySet => Some(true),
        Head::UniversalSet | Head::Reals => Some(false),
        Head::Interval => {
            let (start, end, lo, ro) = interval_parts(s);
            if lo || ro {
                fuzzy_bool(&ge(&start, &end)?)
            } else {
                fuzzy_bool(&crate::core::relational::gt(&start, &end)?)
            }
        }
        Head::FiniteSet | Head::PointSet => Some(s.args().is_empty()),
        Head::Union => {
            let mut values: Vec<Truth> = Vec::new();
            for t in sets_of(s) {
                values.push(is_empty(&t)?);
            }
            fuzzy_and(values)
        }
        Head::ProductSet => {
            let mut values: Vec<Truth> = Vec::new();
            for t in sets_of(s) {
                values.push(is_empty(&t)?);
            }
            fuzzy_or(values)
        }
        _ => None,
    })
}

/// `is_finite_set`.
pub fn is_finite_set(s: &Basic) -> PolyResult<Truth> {
    Ok(match s.head() {
        Head::EmptySet | Head::FiniteSet | Head::PointSet => Some(true),
        Head::UniversalSet => Some(false),
        Head::Interval | Head::Reals => is(&measure(s)?, Predicate::zero),
        Head::Union => {
            let mut values: Vec<Truth> = Vec::new();
            for t in sets_of(s) {
                values.push(is_finite_set(&t)?);
            }
            fuzzy_and(values)
        }
        Head::Intersection => {
            let mut values: Vec<Truth> = Vec::new();
            for t in sets_of(s) {
                values.push(is_finite_set(&t)?);
            }
            if fuzzy_or(values) == Some(true) {
                Some(true)
            } else {
                None
            }
        }
        Head::Complement => {
            let a_finite: Truth = is_finite_set(&s.args()[0])?;
            if a_finite == Some(true) {
                Some(true)
            } else if a_finite == Some(false) && is_finite_set(&s.args()[1])? == Some(true) {
                Some(false)
            } else {
                None
            }
        }
        Head::ProductSet => {
            let mut values: Vec<Truth> = Vec::new();
            for t in sets_of(s) {
                values.push(is_finite_set(&t)?);
            }
            let all_finite: Truth = fuzzy_and(values);
            fuzzy_or([is_empty(s)?, all_finite])
        }
        _ => None,
    })
}

fn is_comparable_intervals(a: &Basic, b: &Basic) -> bool {
    let (a0, a1, _, _) = interval_parts(a);
    let (b0, b1, _, _) = interval_parts(b);
    [a0, a1, b0, b1].iter().all(is_comparable)
}

/// Whether `a` is a subset of `b` (`is_subset`).
pub fn is_subset(a: &Basic, b: &Basic) -> PolyResult<Truth> {
    if matches!(a.head(), Head::Union) {
        let mut values: Vec<Truth> = Vec::new();
        for t in sets_of(a) {
            values.push(is_subset(&t, b)?);
        }
        return Ok(fuzzy_and(values));
    }
    if a == b {
        return Ok(Some(true));
    }
    let empty: Truth = is_empty(a)?;
    if empty == Some(true) {
        return Ok(Some(true));
    }
    if fuzzy_not(empty) == Some(true) && is_empty(b)? == Some(true) {
        return Ok(Some(false));
    }
    if is_finite_set(a)? == Some(false) && is_finite_set(b)? == Some(true) {
        return Ok(Some(false));
    }
    if matches!(a.head(), Head::FiniteSet) {
        let mut values: Vec<Truth> = Vec::new();
        for e in a.args() {
            let c: Option<Basic> = contains_rule(b, e)?;
            values.push(c.as_ref().and_then(fuzzy_bool));
        }
        if let Some(v) = fuzzy_and(values) {
            return Ok(Some(v));
        }
    }
    if let Some(v) = is_subset_sets(a, b)? {
        return Ok(Some(v));
    }
    if intersection(&[a.clone(), b.clone()])? == *a {
        return Ok(Some(true));
    }
    Ok(None)
}

/// The subset rules between intervals and unions of intervals
/// (`_is_subset_sets`).
fn is_subset_sets(a: &Basic, b: &Basic) -> PolyResult<Truth> {
    if !is_interval(a) {
        return Ok(None);
    }
    let (a_start, a_end, a_lo, a_ro) = interval_parts(a);
    if is_interval(b) {
        let (b_start, b_end, b_lo, b_ro) = interval_parts(b);
        if fuzzy_bool(&lt(&a_start, &b_start)?) == Some(true) {
            return Ok(Some(false));
        }
        if fuzzy_bool(&crate::core::relational::gt(&a_end, &b_end)?) == Some(true) {
            return Ok(Some(false));
        }
        if b_lo && !a_lo && fuzzy_bool(&eq(&a_start, &b_start)?) == Some(true) {
            return Ok(Some(false));
        }
        if b_ro && !a_ro && fuzzy_bool(&eq(&a_end, &b_end)?) == Some(true) {
            return Ok(Some(false));
        }
        return Ok(None);
    }
    if matches!(b.head(), Head::FiniteSet) {
        if fuzzy_not(is(&measure(a)?, Predicate::zero)) == Some(true) {
            return Ok(Some(false));
        }
        return Ok(None);
    }
    if matches!(b.head(), Head::Union) {
        let members: Vec<Basic> = sets_of(b);
        if members
            .iter()
            .all(|s| is_interval(s) || matches!(s.head(), Head::FiniteSet))
        {
            let intervals: Vec<Basic> = members.into_iter().filter(is_interval).collect();
            let mut all_before: bool = true;
            let mut all_after: bool = true;
            for s in &intervals {
                let (s_start, s_end, _, _) = interval_parts(s);
                if fuzzy_bool(&lt(&a_start, &s_start)?) != Some(true) {
                    all_before = false;
                }
                if fuzzy_bool(&crate::core::relational::gt(&a_end, &s_end)?) != Some(true) {
                    all_after = false;
                }
            }
            if all_before {
                return Ok(Some(false));
            }
            if all_after {
                return Ok(Some(false));
            }
            if is_true(&measure(a)?, Predicate::nonzero) {
                let mut all_apart: bool = true;
                for s in &intervals {
                    let (s_start, s_end, _, _) = interval_parts(s);
                    let apart: Truth =
                        fuzzy_or([fuzzy_bool(&le(&s_end, &a_start)?), fuzzy_bool(&ge(&s_start, &a_end)?)]);
                    if apart != Some(true) {
                        all_apart = false;
                    }
                }
                if all_apart {
                    return Ok(Some(false));
                }
            }
        }
        return Ok(None);
    }
    Ok(None)
}

/// `other - s` when a rule of `s` applies (`_complement`).
fn complement_rule(s: &Basic, other: &Basic) -> PolyResult<Option<Basic>> {
    match s.head() {
        Head::EmptySet => return Ok(Some(other.clone())),
        Head::UniversalSet => return Ok(Some(empty_set())),
        Head::Union => {
            let mut parts_: Vec<Basic> = Vec::new();
            for t in sets_of(s) {
                parts_.push(complement(other, &t)?);
            }
            return Ok(Some(intersection(&parts_)?));
        }
        Head::Interval | Head::Reals => {
            if is_interval(other) && is_whole_line(&canonical_interval(other)) {
                let (start, end, lo, ro) = interval_parts(s);
                let a: Basic = interval(&Basic::negative_oo(), &start, true, !lo)?;
                let b: Basic = interval(&end, &Basic::oo(), !ro, true)?;
                return Ok(Some(union(&[a, b])?));
            }
            if matches!(other.head(), Head::FiniteSet) && !other.args().iter().any(crate::core::assumptions::is_number)
            {
                return Ok(None);
            }
        }
        Head::FiniteSet => {
            if is_interval(other) {
                let mut nums: Vec<Basic> = Vec::new();
                let mut syms: Vec<Basic> = Vec::new();
                for m in s.args() {
                    if crate::core::assumptions::is_number(m) && is_true(m, Predicate::real) {
                        nums.push(m.clone());
                    } else if crate::core::expr::is_false(m, Predicate::real) {
                    } else {
                        syms.push(m.clone());
                    }
                }
                if is_whole_line(&canonical_interval(other)) && !nums.is_empty() {
                    let nums: Vec<Basic> = sort_numbers(nums)?;
                    let mut intervals: Vec<Basic> = vec![interval(&Basic::negative_oo(), &nums[0], true, true)?];
                    for w in nums.windows(2) {
                        intervals.push(interval(&w[0], &w[1], true, true)?);
                    }
                    intervals.push(interval(nums.last().expect("a number"), &Basic::oo(), true, true)?);
                    if !syms.is_empty() {
                        return Ok(Some(complement_node(&union_node(intervals), &finite_set(&syms))));
                    }
                    return Ok(Some(union_node(intervals)));
                }
                if nums.is_empty() {
                    if !syms.is_empty() {
                        return Ok(Some(complement_node(other, &finite_set(&syms))));
                    }
                    return Ok(Some(other.clone()));
                }
            } else if matches!(other.head(), Head::FiniteSet) {
                let mut unknown: Vec<Basic> = Vec::new();
                for i in s.args() {
                    if truth(&contains(other, i)?).is_none() {
                        unknown.push(i.clone());
                    }
                }
                let unk: Basic = finite_set(&unknown);
                if unk == *s {
                    return Ok(None);
                }
                let mut not_true: Vec<Basic> = Vec::new();
                for i in other.args() {
                    if truth(&contains(s, i)?) != Some(true) {
                        not_true.push(i.clone());
                    }
                }
                return Ok(Some(complement(&finite_set(&not_true), &unk)?));
            }
        }
        _ => {}
    }
    // Set._complement
    if is_interval(other) {
        if is_interval(s) || matches!(s.head(), Head::FiniteSet) {
            return Ok(Some(intersection(&[other.clone(), complement(&reals(), s)?])?));
        }
        return Ok(None);
    }
    match other.head() {
        Head::Union => {
            let mut parts_: Vec<Basic> = Vec::new();
            for o in sets_of(other) {
                parts_.push(complement(&o, s)?);
            }
            Ok(Some(union(&parts_)?))
        }
        Head::Complement => {
            let inner: Basic = union(&[other.args()[1].clone(), s.clone()])?;
            Ok(Some(complement_node(&other.args()[0], &inner)))
        }
        Head::EmptySet => Ok(Some(empty_set())),
        Head::FiniteSet => {
            let mut false_part: Vec<Basic> = Vec::new();
            let mut unknown: Vec<Basic> = Vec::new();
            for x in other.args() {
                match fuzzy_bool(&contains(s, x)?) {
                    Some(false) => false_part.push(x.clone()),
                    None => unknown.push(x.clone()),
                    Some(true) => {}
                }
            }
            let rest: Basic = if unknown.is_empty() {
                empty_set()
            } else {
                complement_node(&finite_set(&unknown), s)
            };
            Ok(Some(union(&[finite_set(&false_part), rest])?))
        }
        _ => Ok(None),
    }
}

/// The real numbers in increasing order (`nums.sort()`, by `<`).
fn sort_numbers(nums: Vec<Basic>) -> PolyResult<Vec<Basic>> {
    let mut failure: Option<PolyError> = None;
    let mut nums: Vec<Basic> = nums;
    nums.sort_by(|a, b| {
        if failure.is_some() {
            return std::cmp::Ordering::Equal;
        }
        let less = |x: &Basic, y: &Basic| -> PolyResult<Option<bool>> { Ok(truth(&lt(x, y)?)) };
        match less(a, b) {
            Ok(Some(true)) => std::cmp::Ordering::Less,
            Ok(Some(false)) => match less(b, a) {
                Ok(Some(true)) => std::cmp::Ordering::Greater,
                Ok(_) => std::cmp::Ordering::Equal,
                Err(e) => {
                    failure = Some(e);
                    std::cmp::Ordering::Equal
                }
            },
            Ok(None) => {
                failure = Some(PolyError::Type(format!(
                    "cannot determine truth value of {} < {}",
                    a, b
                )));
                std::cmp::Ordering::Equal
            }
            Err(e) => {
                failure = Some(e);
                std::cmp::Ordering::Equal
            }
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(nums),
    }
}

// ---------------------------------------------------------------------
// the evaluating functions

/// The interval, evaluated: empty when `end < start`, `{start}` when
/// degenerate and closed, infinite ends open (`interval`).
pub fn interval(a: &Basic, b: &Basic, left_open: bool, right_open: bool) -> PolyResult<Basic> {
    let (mut lo, mut ro) = (left_open, right_open);
    let difference: Basic = sub(b, a)?;
    let real: Truth = fuzzy_and([
        is(a, Predicate::extended_real),
        is(b, Predicate::extended_real),
        is(&difference, Predicate::extended_real),
    ]);
    if fuzzy_not(real) == Some(true) {
        return Err(value_error("Non-real intervals are not supported".to_string()));
    }
    if fuzzy_bool(&lt(b, a)?) == Some(true) {
        return Ok(empty_set());
    }
    if is_true(&difference, Predicate::negative) {
        return Ok(empty_set());
    }
    if b == a && (lo || ro) {
        return Ok(empty_set());
    }
    if b == a {
        if matches!(a.head(), Head::Infinity | Head::NegativeInfinity) {
            return Ok(empty_set());
        }
        return Ok(finite_set(std::slice::from_ref(b)));
    }
    if matches!(a.head(), Head::NegativeInfinity) {
        lo = true;
    }
    if matches!(b.head(), Head::Infinity) {
        ro = true;
    }
    if matches!(a.head(), Head::Infinity) || matches!(b.head(), Head::NegativeInfinity) {
        return Ok(empty_set());
    }
    Ok(interval_node(a, b, lo, ro))
}

/// The finite set, evaluated: duplicates removed, empty is `EmptySet`
/// (`finite_set`).
pub fn finite_set(elements: &[Basic]) -> Basic {
    if elements.is_empty() {
        return empty_set();
    }
    let mut keyed: Vec<(InfimumKey, Basic)> = dedupe(elements.to_vec())
        .into_iter()
        .map(|e| (infimum_key(&e), e))
        .collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    finite_node(keyed.into_iter().map(|(_, e)| e).collect())
}

/// What a union rule gives: one set, or two (`_union_sets`).
enum Unioned {
    One(Basic),
    Two(Basic, Basic),
}

fn union_sets(a: &Basic, b: &Basic) -> PolyResult<Option<Unioned>> {
    match a.head() {
        Head::EmptySet => return Ok(Some(Unioned::One(b.clone()))),
        Head::UniversalSet => return Ok(Some(Unioned::One(a.clone()))),
        _ => {}
    }
    if is_interval(a) && is_interval(b) {
        if !is_comparable_intervals(a, b) {
            return Ok(None);
        }
        let (a_start, a_end, a_lo, a_ro) = interval_parts(a);
        let (b_start, b_end, b_lo, b_ro) = interval_parts(b);
        let end: Basic = min_(&[a_end.clone(), b_end.clone()])?;
        let start: Basic = max_(&[a_start.clone(), b_start.clone()])?;
        if fuzzy_bool(&lt(&end, &start)?) == Some(true)
            || (end == start && !contains_true(a, &end)? && !contains_true(b, &end)?)
        {
            return Ok(None);
        }
        let start: Basic = min_(&[a_start.clone(), b_start.clone()])?;
        let end: Basic = max_(&[a_end.clone(), b_end.clone()])?;
        let left_open: bool = (a_start != start || a_lo) && (b_start != start || b_lo);
        let right_open: bool = (a_end != end || a_ro) && (b_end != end || b_ro);
        return Ok(Some(Unioned::One(interval(&start, &end, left_open, right_open)?)));
    }
    if is_interval(a) && matches!(b.head(), Head::UniversalSet) {
        return Ok(Some(Unioned::One(universal_set())));
    }
    if is_interval(a) {
        let (a_start, a_end, a_lo, a_ro) = interval_parts(a);
        let open_left: bool = a_lo && contains_true(b, &a_start)? && is_true(&a_start, Predicate::finite);
        let open_right: bool = a_ro && contains_true(b, &a_end)? && is_true(&a_end, Predicate::finite);
        if open_left || open_right {
            let new_a: Basic = interval(
                &a_start,
                &a_end,
                a_lo && !contains_true(b, &a_start)?,
                a_ro && !contains_true(b, &a_end)?,
            )?;
            return Ok(Some(Unioned::Two(new_a, b.clone())));
        }
        return Ok(None);
    }
    if matches!(a.head(), Head::FiniteSet) && matches!(b.head(), Head::FiniteSet) {
        let mut elements: Vec<Basic> = a.args().to_vec();
        elements.extend(b.args().iter().filter(|e| !a.args().contains(e)).cloned());
        return Ok(Some(Unioned::One(finite_set(&elements))));
    }
    if matches!(a.head(), Head::FiniteSet) {
        let mut any: bool = false;
        for x in a.args() {
            if contains_true(b, x)? {
                any = true;
            }
        }
        if any {
            let mut kept: Vec<Basic> = Vec::new();
            for x in a.args() {
                if !contains_true(b, x)? {
                    kept.push(x.clone());
                }
            }
            return Ok(Some(Unioned::Two(finite_set(&kept), b.clone())));
        }
        return Ok(None);
    }
    Ok(None)
}

/// The union, evaluated (`union`, SymPy's `simplify_union`).
pub fn union(sets: &[Basic]) -> PolyResult<Basic> {
    let mut args: Vec<Basic> = Vec::new();
    for s in sets {
        if !is_set(s) {
            return Err(PolyError::Type("Input args to Union must be Sets".to_string()));
        }
        if matches!(s.head(), Head::Union) {
            args.extend(sets_of(s));
        } else {
            args.push(s.clone());
        }
    }
    if args.is_empty() {
        return Ok(empty_set());
    }
    let finite: Vec<Basic> = args
        .iter()
        .filter(|x| matches!(x.head(), Head::FiniteSet))
        .cloned()
        .collect();
    if finite.len() > 1 {
        let elements: Vec<Basic> = finite.iter().flat_map(|f| f.args().iter().cloned()).collect();
        let mut rest: Vec<Basic> = vec![finite_set(&elements)];
        rest.extend(args.iter().filter(|x| !matches!(x.head(), Head::FiniteSet)).cloned());
        args = rest;
    }
    let mut work: Vec<Basic> = dedupe(sorted_sets(args));
    'outer: loop {
        for i in 0..work.len() {
            for j in 0..work.len() {
                if i == j {
                    continue;
                }
                let Some(new) = union_sets(&work[i], &work[j])? else {
                    continue;
                };
                let mut rest: Vec<Basic> = work
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .map(|(_, w)| w.clone())
                    .collect();
                match new {
                    Unioned::One(s) => rest.push(s),
                    Unioned::Two(s, t) => {
                        rest.push(s);
                        rest.push(t);
                    }
                }
                work = dedupe(rest);
                continue 'outer;
            }
        }
        break;
    }
    if work.len() == 1 {
        return Ok(work.remove(0));
    }
    Ok(union_node(work))
}

fn has_float(e: &Basic) -> bool {
    // the core has no Float yet
    let _ = e;
    false
}

/// The intersection of two sets when a rule applies (`intersection_sets`).
fn intersection_sets(a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    if is_interval(a) && is_interval(b) {
        let (a_start, a_end, a_lo, a_ro) = interval_parts(a);
        let (b_start, b_end, b_lo, b_ro) = interval_parts(b);
        if is_whole_line(&canonical_interval(a)) {
            let real_or_infinite =
                |e: &Basic| is_true(e, Predicate::real) || matches!(e.head(), Head::Infinity | Head::NegativeInfinity);
            if real_or_infinite(&a_start) || real_or_infinite(&a_end) {
                return Ok(Some(b.clone()));
            }
        }
        if !is_comparable_intervals(a, b) {
            return Ok(None);
        }
        if fuzzy_bool(&le(&a_start, &b_end)?) == Some(true) && fuzzy_bool(&le(&b_start, &a_end)?) == Some(true) {
            let (start, left_open): (Basic, bool) = if fuzzy_bool(&lt(&a_start, &b_start)?) == Some(true) {
                (b_start.clone(), b_lo)
            } else if fuzzy_bool(&crate::core::relational::gt(&a_start, &b_start)?) == Some(true) {
                (a_start.clone(), a_lo)
            } else {
                let mut start: Basic = a_start.clone();
                if a_start != b_start {
                    if has_float(&b_start) && !has_float(&a_start) {
                        start = b_start.clone();
                    } else if !(has_float(&a_start) && !has_float(&b_start)) {
                        let first: Basic = ordered(vec![a.clone(), b.clone()]).remove(0);
                        start = interval_parts(&first).0;
                    }
                }
                (start, a_lo || b_lo)
            };
            let (end, right_open): (Basic, bool) = if fuzzy_bool(&lt(&a_end, &b_end)?) == Some(true) {
                (a_end.clone(), a_ro)
            } else if fuzzy_bool(&crate::core::relational::gt(&a_end, &b_end)?) == Some(true) {
                (b_end.clone(), b_ro)
            } else {
                let mut end: Basic = a_end.clone();
                if a_end != b_end {
                    if has_float(&b_end) && !has_float(&a_end) {
                        end = b_end.clone();
                    } else if !(has_float(&a_end) && !has_float(&b_end)) {
                        let first: Basic = ordered(vec![a.clone(), b.clone()]).remove(0);
                        end = interval_parts(&first).1;
                    }
                }
                (end, a_ro || b_ro)
            };
            if is_true(&sub(&end, &start)?, Predicate::zero) && (left_open || right_open) {
                return Ok(Some(empty_set()));
            }
            return Ok(Some(interval(&start, &end, left_open, right_open)?));
        }
        return Ok(Some(empty_set()));
    }
    match a.head() {
        Head::EmptySet => return Ok(Some(empty_set())),
        Head::UniversalSet => return Ok(Some(b.clone())),
        _ => {}
    }
    if matches!(a.head(), Head::FiniteSet) && matches!(b.head(), Head::FiniteSet) {
        let kept: Vec<Basic> = a.args().iter().filter(|e| b.args().contains(e)).cloned().collect();
        return Ok(Some(finite_set(&kept)));
    }
    if matches!(a.head(), Head::FiniteSet) {
        let mut kept: Vec<Basic> = Vec::new();
        for el in a.args() {
            let c: Basic = contains(b, el)?;
            match truth(&c) {
                Some(true) => kept.push(el.clone()),
                Some(false) => {}
                None => return Ok(None),
            }
        }
        return Ok(Some(finite_set(&kept)));
    }
    Ok(None)
}

/// `Intersection._handle_finite_sets`.
fn handle_finite_sets(args: &[Basic]) -> PolyResult<Option<Basic>> {
    let fs_args: Vec<&Basic> = args.iter().filter(|s| matches!(s.head(), Head::FiniteSet)).collect();
    let mut others: Vec<Basic> = args
        .iter()
        .filter(|s| !matches!(s.head(), Head::FiniteSet))
        .cloned()
        .collect();
    if fs_args.is_empty() {
        return Ok(None);
    }
    let mut fs_sets: Vec<Vec<Basic>> = fs_args.iter().map(|f| f.args().to_vec()).collect();
    let mut all_elements: Vec<Basic> = Vec::new();
    for fs in &fs_sets {
        for e in fs {
            if !all_elements.contains(e) {
                all_elements.push(e.clone());
            }
        }
    }
    let mut definite: Vec<Basic> = Vec::new();
    for e in &all_elements {
        let mut values: Vec<Truth> = Vec::new();
        for s in args {
            values.push(fuzzy_bool(&contains(s, e)?));
        }
        let inall: Truth = fuzzy_and(values);
        if inall == Some(true) && !definite.contains(e) {
            definite.push(e.clone());
        }
        if inall.is_some() {
            for fs in fs_sets.iter_mut() {
                if let Some(position) = fs.iter().position(|x| x == e) {
                    fs.remove(position);
                }
            }
        }
    }
    let mut fs_elements: Vec<Basic> = Vec::new();
    for fs in &fs_sets {
        for e in fs {
            if !fs_elements.contains(e) {
                fs_elements.push(e.clone());
            }
        }
    }
    let mut fs_symsets: Vec<Basic> = fs_sets.iter().map(|fs| finite_set(fs)).collect();
    'outer: while !fs_elements.is_empty() {
        for k in 0..fs_elements.len() {
            let e: Basic = fs_elements[k].clone();
            let mut values: Vec<Truth> = Vec::new();
            for s in &fs_symsets {
                values.push(fuzzy_bool(&contains(s, &e)?));
            }
            let infs: Truth = fuzzy_and(values);
            if infs == Some(true) && !definite.contains(&e) {
                definite.push(e.clone());
            }
            if infs.is_some() {
                for (n, fs) in fs_sets.iter_mut().enumerate() {
                    if let Some(position) = fs.iter().position(|x| *x == e) {
                        fs.remove(position);
                        fs_symsets[n] = finite_set(fs);
                    }
                }
                fs_elements.remove(k);
                continue 'outer;
            }
        }
        break;
    }
    if !fs_sets.iter().all(|fs| !fs.is_empty()) {
        fs_sets = vec![Vec::new()];
    }
    if !definite.is_empty() {
        for fs in fs_sets.iter_mut() {
            for d in &definite {
                if !fs.contains(d) {
                    fs.push(d.clone());
                }
            }
        }
    }
    if fs_sets.len() == 1 && fs_sets[0].is_empty() {
        return Ok(Some(empty_set()));
    }
    let mut sets: Vec<Basic> = fs_sets.iter().map(|fs| finite_set(fs)).collect();
    let every: Vec<Basic> = fs_sets.iter().flat_map(|fs| fs.iter().cloned()).collect();
    let mut kept_others: Vec<Basic> = Vec::new();
    for o in others.drain(..) {
        let mut all_in: bool = true;
        for e in &every {
            if fuzzy_bool(&contains(&o, e)?) != Some(true) {
                all_in = false;
                break;
            }
        }
        if !all_in {
            kept_others.push(o);
        }
    }
    if !kept_others.is_empty() {
        let rest: Basic = intersection(&kept_others)?;
        if matches!(rest.head(), Head::EmptySet) {
            return Ok(Some(empty_set()));
        }
        if matches!(rest.head(), Head::Intersection) {
            sets.extend(sets_of(&rest));
        } else {
            sets.push(rest);
        }
    }
    if sets.len() == 1 {
        return Ok(Some(sets.remove(0)));
    }
    Ok(Some(intersection_node(sets)))
}

/// The intersection, evaluated (`intersection`, SymPy's
/// `simplify_intersection`).
pub fn intersection(sets: &[Basic]) -> PolyResult<Basic> {
    let mut args: Vec<Basic> = Vec::new();
    for s in sets {
        if !is_set(s) {
            return Err(PolyError::Type("Input args to Intersection must be Sets".to_string()));
        }
        if matches!(s.head(), Head::Intersection) {
            args.extend(sets_of(s));
        } else {
            args.push(s.clone());
        }
    }
    let args: Vec<Basic> = ordered(dedupe(args));
    if args.is_empty() {
        return Ok(universal_set());
    }
    if args.iter().any(|s| matches!(s.head(), Head::EmptySet)) {
        return Ok(empty_set());
    }
    if let Some(rv) = handle_finite_sets(&args)? {
        return Ok(rv);
    }
    for (i, s) in args.iter().enumerate() {
        if matches!(s.head(), Head::Union) {
            let other_sets: Vec<Basic> = args
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i)
                .map(|(_, o)| o.clone())
                .collect();
            if !other_sets.is_empty() {
                let other: Basic = intersection(&other_sets)?;
                let mut parts_: Vec<Basic> = Vec::new();
                for arg in sets_of(s) {
                    parts_.push(intersection(&[arg, other.clone()])?);
                }
                return union(&parts_);
            }
            return union(&sets_of(s));
        }
    }
    for (i, s) in args.iter().enumerate() {
        if matches!(s.head(), Head::Complement) {
            let mut rest: Vec<Basic> = args
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i)
                .map(|(_, o)| o.clone())
                .collect();
            rest.push(s.args()[0].clone());
            return complement(&intersection(&rest)?, &s.args()[1]);
        }
    }
    let mut work: Vec<Basic> = args;
    'outer: loop {
        for i in 0..work.len() {
            for j in 0..work.len() {
                if i == j {
                    continue;
                }
                let Some(new) = intersection_sets(&work[i], &work[j])? else {
                    continue;
                };
                let mut rest: Vec<Basic> = work
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .map(|(_, w)| w.clone())
                    .collect();
                rest.push(new);
                work = dedupe(rest);
                continue 'outer;
            }
        }
        break;
    }
    if work.len() == 1 {
        return Ok(work.remove(0));
    }
    Ok(intersection_node(work))
}

/// `a - b`, evaluated (`complement`, SymPy's `Complement.reduce`).
pub fn complement(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    if matches!(b.head(), Head::UniversalSet) || is_subset(a, b)? == Some(true) {
        return Ok(empty_set());
    }
    if matches!(b.head(), Head::Union) {
        let mut parts_: Vec<Basic> = Vec::new();
        for s in sets_of(b) {
            parts_.push(complement(a, &s)?);
        }
        return intersection(&parts_);
    }
    if let Some(result) = complement_rule(b, a)? {
        return Ok(result);
    }
    Ok(complement_node(a, b))
}

/// The Cartesian product, evaluated (`product_set`).
pub fn product_set(sets: &[Basic]) -> PolyResult<Basic> {
    if sets.is_empty() {
        return Err(value_error(
            "the product of no set is not ported (SymPy gives FiniteSet(()))".to_string(),
        ));
    }
    if sets.iter().any(|s| matches!(s.head(), Head::EmptySet)) {
        return Ok(empty_set());
    }
    Ok(Basic::new(Head::ProductSet, sets.to_vec()))
}

/// The finite set of points (tuples), evaluated (`point_set`).
pub fn point_set(points: &[Basic]) -> PolyResult<Basic> {
    let mut unique: Vec<Basic> = Vec::new();
    for p in points {
        if !matches!(p.head(), Head::Tuple) {
            return Err(PolyError::Type(format!("a point is a tuple, got {}", p)));
        }
        if !unique.contains(p) {
            unique.push(p.clone());
        }
    }
    if unique.is_empty() {
        return Ok(empty_set());
    }
    Ok(Basic::new(Head::PointSet, ordered(unique)))
}

/// The evaluated rebuild of a set node (`_evaluated`).
pub fn rebuild_set(head: &Head, args: &[Basic]) -> PolyResult<Basic> {
    match head {
        Head::EmptySet => Ok(empty_set()),
        Head::UniversalSet => Ok(universal_set()),
        Head::Reals => Ok(reals()),
        Head::Interval => {
            let flag = |b: &Basic| truth(b) == Some(true);
            interval(&args[0], &args[1], flag(&args[2]), flag(&args[3]))
        }
        Head::FiniteSet => Ok(finite_set(args)),
        Head::Union => union(args),
        Head::Intersection => intersection(args),
        Head::Complement => complement(&args[0], &args[1]),
        Head::Contains => contains(&args[1], &args[0]),
        Head::ProductSet => product_set(args),
        Head::PointSet => point_set(args),
        Head::ConditionSet => super::conditionset::condition_set(&args[0], &args[1], &args[2]),
        _ => unreachable!("not a set"),
    }
}

/// `str` of a set node (`_sympystr`); `doprint` prints a subexpression.
pub fn print_set(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    match e.head() {
        Head::EmptySet => "EmptySet".to_string(),
        Head::UniversalSet => "UniversalSet".to_string(),
        Head::Reals => "Reals".to_string(),
        Head::Interval => {
            let (a, b, lo, ro) = interval_parts(e);
            let infinite = |x: &Basic| is_true(x, Predicate::infinite);
            let m: &str =
                if (infinite(&a) && infinite(&b)) || (infinite(&a) && !ro) || (infinite(&b) && !lo) || (!lo && !ro) {
                    ""
                } else if lo && ro {
                    ".open"
                } else if lo {
                    ".Lopen"
                } else {
                    ".Ropen"
                };
            format!("Interval{}({}, {})", m, a, b)
        }
        Head::FiniteSet | Head::PointSet => {
            let mut items: Vec<Basic> = e.args().to_vec();
            items.sort_by_cached_key(sort_key);
            let shown: Vec<String> = items.iter().map(doprint).collect();
            format!("{{{}}}", shown.join(", "))
        }
        Head::Union | Head::Intersection => {
            let mut items: Vec<Basic> = e.args().to_vec();
            items.sort_by_cached_key(sort_key);
            let shown: Vec<String> = items.iter().map(doprint).collect();
            format!("{}({})", e.class_name(), shown.join(", "))
        }
        Head::Complement | Head::Contains | Head::ProductSet | Head::ConditionSet => {
            let shown: Vec<String> = e.args().iter().map(doprint).collect();
            format!("{}({})", e.class_name(), shown.join(", "))
        }
        _ => unreachable!("not a set"),
    }
}
