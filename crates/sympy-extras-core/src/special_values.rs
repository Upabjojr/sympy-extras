//! Port of `sympy_extras/_special_values.py`: the special values of the
//! parameters of a result (the zeros of the bases of negative powers, of
//! the arguments of logarithms, the pole of `zeta`), where the generic
//! formula is undefined, and the cases computed there
//! ([`with_special_values`]): `Piecewise((value_at_point, Eq(p, v)), ...,
//! (generic, True))`.

#![allow(clippy::mutable_key_type, clippy::type_complexity, clippy::too_many_arguments)]

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;

/// A point: the values of some parameters, sorted by the names.
pub type Point = Vec<(Basic, Basic)>;

/// The computation of the value at a point under the assumptions there.
pub type Compute<'a> = &'a dyn Fn(&Point, &[Basic]) -> PolyResult<Option<Basic>>;

const MAX_POINTS: usize = 4;
const CANDIDATE_SECONDS: f64 = 2.0;
const BUDGET_SHARE: f64 = 0.1;
const REMAINING_SHARE: f64 = 0.5;
const MAX_RADICAL_DEGREE: i64 = 2;

struct Deadline {
    end: Option<Instant>,
}

impl Deadline {
    fn new(seconds: Option<f64>) -> Deadline {
        let scale: f64 = crate::settings::settings().time_scale;
        Deadline {
            end: seconds.map(|s| Instant::now() + Duration::from_secs_f64((s * scale).max(0.0))),
        }
    }

    fn left(&self, cap: Option<f64>) -> Option<f64> {
        let Some(end) = self.end else { return cap };
        let now: Instant = Instant::now();
        let secs: f64 = if end > now {
            (end - now).as_secs_f64()
        } else {
            -(now - end).as_secs_f64()
        };
        let left: f64 = secs / crate::settings::settings().time_scale;
        Some(match cap {
            None => left,
            Some(c) => c.min(left),
        })
    }

    fn up(&self) -> bool {
        self.left(None).is_some_and(|l| l <= 0.0)
    }
}

fn name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

fn point_of(mut items: Vec<(Basic, Basic)>) -> Point {
    items.sort_by(|a, b| name(&a.0).cmp(&name(&b.0)).then_with(|| a.0.cmp(&b.0)));
    items
}

fn rule(point: &Point) -> HashMap<Basic, Basic> {
    point.iter().cloned().collect()
}

/// `e` with the parameters replaced by their values (`_replaced`).
pub fn replaced(e: &Basic, point: &Point) -> R {
    crate::core::function::xreplace_evaluated(e, &rule(point))
}

fn symbols(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_)))
        .collect()
}

fn sorted(symbols: &BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.iter().cloned().collect();
    v.sort_by_key(name);
    v
}

fn has_head(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}

fn infinite(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::Infinity | Head::NegativeInfinity))
}

fn nan_or_zoo(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity))
}

fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}

fn is_false(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanFalse)
}

fn polynomial(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
}

/// The coefficients of `e` as a polynomial in `p` (`_coefficients`).
fn coefficients(e: &Basic, p: &Basic) -> PolyResult<Option<Vec<(i64, Basic)>>> {
    let mut gathered: Vec<(i64, Vec<Basic>)> = Vec::new();
    for t in crate::core::expr::terms_of(&crate::core::function::expand_default(e)?) {
        let (c, k) = crate::solvers::solvers::as_coeff_exponent(&t, p)?;
        let Some(k) = as_int(&k).filter(|k| *k >= 0) else {
            return Ok(None);
        };
        if c.has(p) {
            return Ok(None);
        }
        match gathered.iter_mut().find(|(d, _)| *d == k) {
            Some(entry) => entry.1.push(c),
            None => gathered.push((k, vec![c])),
        }
    }
    let mut result: Vec<(i64, Basic)> = Vec::new();
    for (degree, cs) in gathered {
        let coefficient: Basic = crate::core::add::add(&cs)?;
        if coefficient != Basic::zero() {
            result.push((degree, coefficient));
        }
    }
    Ok(Some(result))
}

fn as_int(e: &Basic) -> Option<i64> {
    match e.head() {
        Head::Rational {
            integer_class: true, ..
        } => i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok(),
        _ => None,
    }
}

fn total_degree(e: &Basic, syms: &[Basic]) -> PolyResult<Option<i64>> {
    let mut degree: i64 = 0;
    for t in crate::core::expr::terms_of(&crate::core::function::expand_default(e)?) {
        let mut rest: Basic = t;
        let mut d: i64 = 0;
        for s in syms {
            let (c, k) = crate::solvers::solvers::as_coeff_exponent(&rest, s)?;
            let Some(k) = as_int(&k).filter(|k| *k >= 0) else {
                return Ok(None);
            };
            if c.has(s) {
                return Ok(None);
            }
            d += k;
            rest = c;
        }
        if rest != Basic::zero() {
            degree = degree.max(d);
        }
    }
    Ok(Some(degree))
}

fn not_looked_into(node: &Basic) -> bool {
    matches!(node.head(), Head::Sum | Head::Product | Head::Integral)
}

fn unevaluated(e: &Basic) -> bool {
    has_head(e, &not_looked_into)
}

/// The special points of the parameters in `value` (`special_points`).
pub fn special_points(
    value: &Basic,
    parameters: &[Basic],
    assumptions: &[Basic],
    seconds: Option<f64>,
) -> PolyResult<Vec<Point>> {
    special_points_within(value, parameters, assumptions, &Deadline::new(seconds))
}

fn special_points_within(
    value: &Basic,
    parameters: &[Basic],
    assumptions: &[Basic],
    deadline: &Deadline,
) -> PolyResult<Vec<Point>> {
    let wanted: BTreeSet<Basic> = parameters.iter().cloned().collect();
    points_of(&undefined_parts(value, &wanted)?, parameters, assumptions, deadline)
}

fn point_key(point: &Point) -> String {
    let mut items: Vec<String> = point.iter().map(|(k, v)| format!("({}, {})", k, v)).collect();
    items.sort();
    format!("[{}]", items.join(", "))
}

fn points_of(
    parts: &[Basic],
    parameters: &[Basic],
    assumptions: &[Basic],
    deadline: &Deadline,
) -> PolyResult<Vec<Point>> {
    let wanted: BTreeSet<Basic> = parameters.iter().cloned().collect();
    let mut found: Vec<Point> = Vec::new();
    for part in parts {
        if deadline.up() {
            break;
        }
        for point in zeros(part, &wanted, deadline)? {
            if !found.contains(&point) && point_assumptions_within(&point, assumptions, deadline)?.is_some() {
                found.push(point);
            }
        }
    }
    found.sort_by_key(point_key);
    Ok(found)
}

fn undefined_parts(e: &Basic, parameters: &BTreeSet<Basic>) -> PolyResult<Vec<Basic>> {
    fn walk(node: &Basic, parameters: &BTreeSet<Basic>, parts: &mut Vec<Basic>) -> PolyResult<()> {
        if not_looked_into(node) {
            return Ok(());
        }
        let mut push = |u: Basic| {
            if symbols(&u).iter().any(|s| parameters.contains(s)) && !parts.contains(&u) {
                parts.push(u);
            }
        };
        match node.head() {
            Head::Pow => {
                let exponent: &Basic = &node.args()[1];
                let depends: bool = symbols(exponent).iter().any(|s| parameters.contains(s));
                if is(exponent, Predicate::negative) == Some(true)
                    || (depends && is(exponent, Predicate::nonnegative) != Some(true))
                {
                    push(node.args()[0].clone());
                }
            }
            Head::Function(Function::Log) => push(node.args()[0].clone()),
            Head::Function(Function::Zeta) if node.args().len() == 1 => {
                push(crate::core::add::sub(&node.args()[0], &Basic::one())?)
            }
            _ => {}
        }
        for a in node.args() {
            walk(a, parameters, parts)?;
        }
        Ok(())
    }
    let mut parts: Vec<Basic> = Vec::new();
    walk(e, parameters, &mut parts)?;
    Ok(parts)
}

fn zeros(part: &Basic, parameters: &BTreeSet<Basic>, deadline: &Deadline) -> PolyResult<Vec<Point>> {
    let syms: BTreeSet<Basic> = symbols(part);
    if !syms.iter().any(|s| parameters.contains(s)) {
        return Ok(Vec::new());
    }
    if part.is_symbol() {
        return Ok(vec![vec![(part.clone(), Basic::zero())]]);
    }
    let ordered: Vec<Basic> = sorted(&syms);
    let factors: Vec<Basic> = if syms.iter().all(|s| parameters.contains(s))
        && polynomial(part, &ordered)
        && total_degree(part, &ordered)? == Some(1)
    {
        vec![part.clone()]
    } else {
        let seconds: Option<f64> = deadline.left(Some(CANDIDATE_SECONDS));
        if seconds.is_some_and(|s| s <= 0.0) {
            return Ok(Vec::new());
        }
        let factored =
            crate::timeout::attempt_within(seconds, || crate::polys::polytools::factor(part, &[], false, true))?;
        crate::core::expr::factors_of(factored.as_ref().unwrap_or(part))
    };
    let mut points: Vec<Point> = Vec::new();
    for f in factors {
        let mut g: Basic = f;
        while g.is_pow() && is(&g.args()[1], Predicate::positive) == Some(true) {
            g = g.args()[0].clone();
        }
        let g_syms: Vec<Basic> = sorted(&symbols(&g));
        if g_syms.is_empty() || g_syms.iter().any(|s| !parameters.contains(s)) {
            continue;
        }
        for point in factor_zeros(&g, &g_syms, deadline)? {
            if real_for_real_parameters(&point[0].1)? && !points.contains(&point) {
                points.push(point);
            }
        }
    }
    Ok(points)
}

fn factor_zeros(g: &Basic, syms: &[Basic], deadline: &Deadline) -> PolyResult<Vec<Point>> {
    if polynomial(g, syms) {
        let mut polynomials: Vec<(Basic, Vec<(i64, Basic)>)> = Vec::new();
        for p in syms {
            let Some(cs) = coefficients(g, p)? else {
                return Ok(Vec::new());
            };
            polynomials.push((p.clone(), cs));
        }
        let degree_of = |cs: &[(i64, Basic)]| cs.iter().map(|(d, _)| *d).max().unwrap_or(0);
        let coefficient = |cs: &[(i64, Basic)], d: i64| cs.iter().find(|(k, _)| *k == d).map(|(_, c)| c.clone());
        for constant in [true, false] {
            for (p, cs) in &polynomials {
                if degree_of(cs) == 1 {
                    let c1: Basic = coefficient(cs, 1).expect("a linear coefficient");
                    if !constant || symbols(&c1).is_empty() {
                        let c0: Basic = coefficient(cs, 0).unwrap_or_else(Basic::zero);
                        let value: Basic = crate::core::mul::div(&crate::core::mul::neg(&c0)?, &c1)?;
                        return Ok(vec![vec![(p.clone(), value)]]);
                    }
                }
            }
        }
        if syms.len() != 1 {
            return Ok(Vec::new());
        }
        let (p, cs) = &polynomials[0];
        let seconds: Option<f64> = deadline.left(Some(CANDIDATE_SECONDS));
        if seconds.is_some_and(|s| s <= 0.0) {
            return Ok(Vec::new());
        }
        let found: Vec<Basic> = crate::timeout::attempt_within(seconds, || real_roots(g, p))?.unwrap_or_default();
        if found.is_empty() {
            return Ok(Vec::new());
        }
        let degree: i64 = degree_of(cs);
        return Ok(found
            .into_iter()
            .filter(|r| degree <= MAX_RADICAL_DEGREE || is(r, Predicate::rational) == Some(true))
            .map(|r| vec![(p.clone(), r)])
            .collect());
    }
    for p in syms {
        let seconds: Option<f64> = deadline.left(Some(CANDIDATE_SECONDS));
        if seconds.is_some_and(|s| s <= 0.0) {
            return Ok(Vec::new());
        }
        let solutions = crate::timeout::attempt_within(seconds, || real_solutions(g, p))?;
        if let Some(s) = solutions {
            if matches!(s.head(), Head::FiniteSet) && !s.args().is_empty() && s.args().len() <= MAX_POINTS {
                return Ok(s.args().iter().map(|v| vec![(p.clone(), v.clone())]).collect());
            }
        }
    }
    Ok(Vec::new())
}

fn real_roots(g: &Basic, p: &Basic) -> PolyResult<Vec<Basic>> {
    match crate::polys::polyroots::roots(g, std::slice::from_ref(p), true, true, Some("R")) {
        Ok(found) => Ok(found.iter().map(|(r, _)| r.clone()).collect()),
        Err(PolyError::Polys(..)) => {
            let listed: Vec<Basic> = match crate::polys::polyroots::ex_roots(g, p) {
                Ok(l) => l,
                Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {
                    return Ok(Vec::new())
                }
                Err(e) => return Err(e),
            };
            let mut out: Vec<Basic> = Vec::new();
            for r in listed {
                let (n, d) = crate::core::expr::as_numer_denom(&r)?;
                if is(&n, Predicate::real) == Some(true) && is(&d, Predicate::real) == Some(true) {
                    out.push(r);
                }
            }
            Ok(out)
        }
        Err(e) => Err(e),
    }
}

fn real_solutions(g: &Basic, p: &Basic) -> R {
    let equation: Basic = crate::core::relational::eq(g, &Basic::zero())?;
    if let Some(found) = crate::solvers::transcendental::solve_transcendental(&equation, p, None, None, true)? {
        return Ok(found);
    }
    crate::solvers::solveset::solveset(&equation, p, &crate::sets::sets::reals())
}

fn real_for_real_parameters(v: &Basic) -> PolyResult<bool> {
    let replacement: Point = sorted(&symbols(v))
        .into_iter()
        .map(|s| {
            let d: Basic =
                crate::core::symbol::Dummy::new(&name(&s), &[(Predicate::real, true), (Predicate::nonzero, true)]).0;
            (s, d)
        })
        .collect();
    Ok(is(&replaced(v, &replacement)?, Predicate::extended_real) == Some(true))
}

/// `And(Eq(p, v), ...)` of a point, by the names of the parameters
/// (`point_condition`).
pub fn point_condition(point: &Point) -> R {
    let mut items: Vec<&(Basic, Basic)> = point.iter().collect();
    items.sort_by_key(|(k, _)| name(k));
    let parts: Vec<Basic> = items
        .iter()
        .map(|(p, v)| crate::core::relational::eq(p, v))
        .collect::<PolyResult<_>>()?;
    Ok(crate::core::logic::and_(&parts))
}

/// The assumptions at a point, `None` when they do not hold there
/// (`point_assumptions`).
pub fn point_assumptions(point: &Point, assumptions: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    point_assumptions_within(point, assumptions, &Deadline::new(None))
}

fn point_assumptions_within(
    point: &Point,
    assumptions: &[Basic],
    deadline: &Deadline,
) -> PolyResult<Option<Vec<Basic>>> {
    for (p, v) in point {
        if is_false(&crate::core::relational::eq(p, v)?) {
            return Ok(None);
        }
    }
    let mut substituted: Vec<Basic> = Vec::new();
    for item in assumptions {
        let at: Basic = match replaced(item, point) {
            Ok(a) => a,
            Err(PolyError::Type(_)) => {
                substituted.push(item.clone());
                continue;
            }
            Err(e) => return Err(e),
        };
        if is_false(&at) {
            return Ok(None);
        }
        if !is_true(&at) {
            substituted.push(at);
        }
    }
    let seconds: Option<f64> = deadline.left(Some(CANDIDATE_SECONDS));
    if !substituted.is_empty() && seconds.is_none_or_positive() {
        let formula: Basic = crate::core::logic::and_(&substituted);
        let sat =
            crate::timeout::attempt_within(seconds, || crate::assumptions::sat::satisfiable(&formula, &[], None))?;
        if matches!(sat, Some(crate::assumptions::sat::Sat::Unsatisfiable)) {
            return Ok(None);
        }
    }
    Ok(Some(substituted))
}

trait PositiveOrNone {
    fn is_none_or_positive(&self) -> bool;
}

impl PositiveOrNone for Option<f64> {
    fn is_none_or_positive(&self) -> bool {
        match self {
            None => true,
            Some(s) => *s > 0.0,
        }
    }
}

/// Whether the generic expression is undefined at the point: a `nan`,
/// `zoo` or an infinity which it did not have (`undefined_at`).
pub fn undefined_at(e: &Basic, point: &Point) -> PolyResult<bool> {
    let at: Basic = match replaced(e, point) {
        Ok(a) => a,
        Err(PolyError::Type(_) | PolyError::Value(_) | PolyError::ZeroDivision(_)) => return Ok(true),
        Err(err) => return Err(err),
    };
    if nan_or_zoo(&at) {
        return Ok(true);
    }
    Ok(infinite(&at) && !infinite(e))
}

fn branches(value: &Basic) -> Vec<(Basic, Basic)> {
    if matches!(value.head(), Head::Piecewise) {
        return value
            .args()
            .iter()
            .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
            .collect();
    }
    vec![(value.clone(), crate::core::logic::true_())]
}

#[derive(PartialEq, Clone, Copy)]
enum Failing {
    Fallback,
    Undefined,
}

fn condition_at(condition: &Basic, point: &Point) -> PolyResult<Option<Basic>> {
    match replaced(condition, point) {
        Ok(a) => Ok(Some(a)),
        Err(PolyError::Type(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn taken_undefined(bs: &[(Basic, Basic)], point: &Point) -> PolyResult<Option<Failing>> {
    for (e, condition) in bs {
        let at: Basic = condition_at(condition, point)?.unwrap_or_else(|| condition.clone());
        if is_false(&at) {
            continue;
        }
        let mut failing: Option<Failing> = if unevaluated(e) {
            Some(Failing::Fallback)
        } else if undefined_at(e, point)? {
            Some(Failing::Undefined)
        } else {
            None
        };
        if failing == Some(Failing::Undefined) && !is_true(&at) {
            failing = Some(Failing::Fallback);
        }
        if failing.is_some() || is_true(&at) {
            return Ok(failing);
        }
    }
    Ok(None)
}

fn taken_branch(bs: &[(Basic, Basic)], point: &Point) -> PolyResult<Option<usize>> {
    for (index, (_, condition)) in bs.iter().enumerate() {
        let Some(at) = condition_at(condition, point)? else {
            return Ok(None);
        };
        if is_false(&at) {
            continue;
        }
        return Ok(if is_true(&at) { Some(index) } else { None });
    }
    Ok(None)
}

fn disjuncts(condition: &Basic) -> Vec<Basic> {
    if matches!(condition.head(), Head::Or) {
        return condition.args().to_vec();
    }
    vec![condition.clone()]
}

fn flattened(special: &Basic, point: &Point) -> PolyResult<Vec<(Basic, Basic, usize)>> {
    let mut cases: Vec<(Basic, Basic, usize)> = Vec::new();
    for (e, condition) in branches(special) {
        let inners: Vec<Option<Point>> = disjuncts(&condition)
            .iter()
            .map(condition_point)
            .collect::<PolyResult<_>>()?;
        if inners.iter().all(|i| i.is_some()) {
            for inner in inners.into_iter().flatten() {
                let mut composed: Vec<(Basic, Basic)> = Vec::new();
                for (p, v) in point {
                    composed.push((p.clone(), replaced(v, &inner)?));
                }
                for (p, v) in inner {
                    match composed.iter_mut().find(|(k, _)| *k == p) {
                        Some(entry) => entry.1 = v,
                        None => composed.push((p, v)),
                    }
                }
                let composed: Point = point_of(composed);
                let n: usize = composed.len();
                cases.push((e.clone(), point_condition(&composed)?, n));
            }
        } else {
            let condition: Basic = crate::core::logic::and_(&[point_condition(point)?, condition]);
            cases.push((e, condition, point.len()));
        }
    }
    Ok(cases)
}

/// The points of the parameters named by the `Eq` cases of a result
/// (`equality_points`).
pub fn equality_points(value: &Basic, parameters: &[Basic]) -> PolyResult<Vec<Point>> {
    let mut points: Vec<Point> = Vec::new();
    for (_, condition) in branches(value) {
        for d in disjuncts(&condition) {
            if let Some(point) = condition_point(&d)? {
                if !point.is_empty() && point.iter().all(|(p, _)| parameters.contains(p)) && !points.contains(&point) {
                    points.push(point);
                }
            }
        }
    }
    Ok(points)
}

/// The point of a condition `And(Eq(p, v), ...)` solved for the
/// parameters (`condition_point`); `None` for any other condition.
pub fn condition_point(condition: &Basic) -> PolyResult<Option<Point>> {
    if is_true(condition) {
        return Ok(Some(Vec::new()));
    }
    let parts: Vec<Basic> = if matches!(condition.head(), Head::And) {
        condition.args().to_vec()
    } else {
        vec![condition.clone()]
    };
    let mut point: Vec<(Basic, Basic)> = Vec::new();
    for part in parts {
        if !matches!(part.head(), Head::Relational(RelOp::Eq)) {
            return Ok(None);
        }
        let (lhs, rhs) = (&part.args()[0], &part.args()[1]);
        let entry: (Basic, Basic) = if lhs.is_symbol() && !rhs.has(lhs) {
            (lhs.clone(), rhs.clone())
        } else if rhs.is_symbol() && !lhs.has(rhs) {
            (rhs.clone(), lhs.clone())
        } else {
            return Ok(None);
        };
        match point.iter_mut().find(|(k, _)| *k == entry.0) {
            Some(e) => e.1 = entry.1,
            None => point.push(entry),
        }
    }
    let mut point: Point = point_of(point);
    for _ in 0..point.len() {
        let solved: Point = point.clone();
        let mut next: Vec<(Basic, Basic)> = Vec::new();
        for (p, v) in &solved {
            next.push((p.clone(), replaced(v, &solved)?));
        }
        point = point_of(next);
    }
    let keys: BTreeSet<Basic> = point.iter().map(|(k, _)| k.clone()).collect();
    if point.iter().any(|(_, v)| symbols(v).iter().any(|s| keys.contains(s))) {
        return Ok(None);
    }
    Ok(Some(point))
}

/// Whether the integrand is defined for generic values of the
/// parameters: no `nan`, `zoo`, power of zero or delta without the
/// variables (`defined_problem`).
pub fn defined_problem(integrand: &Basic, variables: &[Basic]) -> bool {
    if nan_or_zoo(integrand) {
        return false;
    }
    for node in integrand.preorder() {
        if node.is_pow() && is(&node.args()[0], Predicate::zero) == Some(true) {
            return false;
        }
        if matches!(node.head(), Head::Function(Function::DiracDelta))
            && !symbols(&node.args()[0]).iter().any(|s| variables.contains(s))
        {
            return false;
        }
    }
    true
}

/// `value` with the cases of the special points of the parameters where
/// it is undefined (or `degenerate` says it degenerates), each computed by
/// `compute` under the assumptions there (`with_special_values`).
pub fn with_special_values(
    value: &Basic,
    parameters: &[Basic],
    assumptions: &[Basic],
    compute: Compute,
    defined: Option<&dyn Fn(&Point) -> bool>,
    sympy_style: bool,
    budget: Option<f64>,
    keep_unevaluated: bool,
    degenerate: Option<&dyn Fn(&Basic) -> PolyResult<Vec<Basic>>>,
) -> R {
    if parameters.is_empty() {
        return Ok(value.clone());
    }
    let mut limit: Option<f64> = budget;
    if limit.is_none() {
        limit = crate::settings::settings().timeout.map(|t| BUDGET_SHARE * t);
    }
    if let Some(enclosing) = crate::timeout::remaining_time() {
        let share: f64 = REMAINING_SHARE * enclosing;
        limit = Some(limit.map_or(share, |l| l.min(share)));
    }
    let deadline: Deadline = Deadline::new(limit);
    let bs: Vec<(Basic, Basic)> = branches(value);
    let mut points: Vec<Point> = Vec::new();
    let mut degenerate_at: Vec<(Point, usize)> = Vec::new();
    for (index, (e, _)) in bs.iter().enumerate() {
        if unevaluated(e) {
            continue;
        }
        for point in special_points_within(e, parameters, assumptions, &deadline)? {
            if !points.contains(&point) {
                points.push(point);
            }
        }
        let seconds: Option<f64> = deadline.left(Some(CANDIDATE_SECONDS));
        let Some(callback) = degenerate else { continue };
        if seconds.is_some_and(|s| s <= 0.0) {
            continue;
        }
        let parts: Vec<Basic> = crate::timeout::attempt_within(seconds, || callback(e))?.unwrap_or_default();
        for point in points_of(&parts, parameters, assumptions, &deadline)? {
            degenerate_at.push((point.clone(), index));
            if !points.contains(&point) {
                points.push(point);
            }
        }
    }
    let mut cases: Vec<(Basic, Basic, usize)> = Vec::new();
    let mut listed: usize = 0;
    for point in &points {
        if listed >= MAX_POINTS || deadline.up() {
            break;
        }
        let at: Option<Vec<Basic>> = point_assumptions_within(point, assumptions, &deadline)?;
        let mut failing: Option<Failing> = if at.is_none() {
            None
        } else {
            taken_undefined(&bs, point)?
        };
        if failing.is_none() && at.is_some() {
            let taken: Option<usize> = taken_branch(&bs, point)?;
            if let Some(t) = taken {
                if degenerate_at.contains(&(point.clone(), t)) {
                    failing = Some(Failing::Undefined);
                }
            }
        }
        let Some(failing) = failing else { continue };
        if let Some(d) = defined {
            if !d(point) {
                continue;
            }
        }
        let remaining: Option<f64> = deadline.left(None);
        if remaining.is_some_and(|r| r <= 0.0) {
            break;
        }
        let at: Vec<Basic> = at.unwrap_or_default();
        let special: Option<Basic> = crate::timeout::attempt_within(remaining, || compute(point, &at))?.flatten();
        let Some(special) = special else { continue };
        if nan_or_zoo(&special) {
            continue;
        }
        if failing == Failing::Fallback && (unevaluated(&special) || infinite(&special)) {
            continue;
        }
        if !keep_unevaluated && unevaluated(&special) {
            continue;
        }
        listed += 1;
        for case in flattened(&special, point)? {
            if cases.iter().all(|other| case.1 != other.1) {
                cases.push(case);
            }
        }
    }
    if cases.is_empty() {
        return Ok(value.clone());
    }
    // a stable sort, the larger points first
    cases.sort_by_key(|c| std::cmp::Reverse(c.2));
    let single: Option<Point> = if sympy_style && cases.len() == 1 && !matches!(value.head(), Head::Piecewise) {
        condition_point(&cases[0].1)?.filter(|p| !p.is_empty())
    } else {
        None
    };
    if let Some(single) = single {
        let mut items: Vec<&(Basic, Basic)> = single.iter().collect();
        items.sort_by_key(|(k, _)| name(k));
        let parts: Vec<Basic> = items
            .iter()
            .map(|(p, v)| crate::core::relational::ne(p, v))
            .collect::<PolyResult<_>>()?;
        let generic: Basic = crate::core::logic::or_(&parts);
        return crate::functions::elementary::piecewise::piecewise(&[
            (value.clone(), generic),
            (cases[0].0.clone(), crate::core::logic::true_()),
        ]);
    }
    let mut pairs: Vec<(Basic, Basic)> = cases.into_iter().map(|(e, c, _)| (e, c)).collect();
    pairs.extend(bs);
    crate::functions::elementary::piecewise::piecewise(&pairs)
}
