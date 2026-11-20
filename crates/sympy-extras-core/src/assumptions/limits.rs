//! Port of `sympy_extras/assumptions/limits.py`: limits (`limit`,
//! `limit_seq`) and series (`series`) under assumptions on the
//! parameters. The parameters are replaced by dummies with the
//! properties the assumptions give (bases of powers by `exp(q)`), the
//! core's limit is taken, and the result is split into cases on the signs
//! it depends on, each case checked at sample points.

#![allow(clippy::mutable_key_type, clippy::too_many_arguments)]

use std::collections::HashMap;

use crate::assumptions::ask::evaluate;
use crate::assumptions::facts::{normalize, Facts};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;

/// The core's limit of an expression in a variable (`Engine`).
pub type Engine<'a> = &'a dyn Fn(&Basic, &Basic) -> R;

const MAX_DEPTH: usize = 3;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    Basic::zero()
}
fn true_() -> Basic {
    crate::core::logic::true_()
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
fn eq(a: &Basic, b: &Basic) -> R {
    crate::core::relational::eq(a, b)
}
fn and_(v: &[Basic]) -> Basic {
    crate::core::logic::and_(v)
}
fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}
fn has_head(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn is_limit(e: &Basic) -> bool {
    matches!(e.head(), Head::Limit)
}
fn has_bounds(e: &Basic) -> bool {
    has_head(e, &crate::core::accumbounds::is_bounds)
}
fn is_infinity(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity | Head::NegativeInfinity)
}
fn not_ported(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if m.starts_with(crate::error::NOT_PORTED_PREFIX))
}
fn reals() -> Basic {
    crate::sets::sets::reals()
}

fn parameters(expr: &Basic, x: &Basic) -> Vec<Basic> {
    let mut v: Vec<Basic> = expr.free_symbols().into_iter().filter(|s| s != x).collect();
    v.sort_by_key(|s| match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    });
    v
}

fn replaced(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    crate::core::function::xreplace_evaluated(e, rule)
}

fn holds(statement: &Basic, facts: &Facts) -> PolyResult<bool> {
    Ok(evaluate(&normalize(statement)?, facts)? == Some(true))
}

fn with(case: &Basic, assumptions: &[Basic]) -> Vec<Basic> {
    let mut extra: Vec<Basic> = vec![case.clone()];
    extra.extend(assumptions.iter().cloned());
    extra
}

fn case_facts(extra: &[Basic], facts: &Facts) -> PolyResult<Facts> {
    Facts::new(extra, Some(&reals()), &facts.real)
}

fn abstract_(expr: &Basic, facts: &Facts, x: &Basic) -> PolyResult<(Basic, HashMap<Basic, Basic>)> {
    let refiner = crate::assumptions::refine::Refiner::new(facts.clone());
    let mut forward: HashMap<Basic, Basic> = HashMap::new();
    let mut back: HashMap<Basic, Basic> = HashMap::new();
    let exp = crate::functions::elementary::exponential::exp;
    let log = crate::functions::elementary::exponential::log;
    for parameter in parameters(expr, x) {
        if holds(&eq(&parameter, &zero())?, facts)? {
            forward.insert(parameter, zero());
            continue;
        }
        if let Some(value) = known_value(&parameter, facts) {
            forward.insert(parameter, value);
            continue;
        }
        if is_base(&parameter, expr, x) {
            let positive = || crate::core::symbol::Dummy::new("q", &[(Predicate::positive, true)]).0;
            if holds(&gt(&parameter, &Basic::one())?, facts)? {
                let q: Basic = positive();
                forward.insert(parameter.clone(), exp(&q)?);
                back.insert(q, log(&parameter)?);
                continue;
            }
            if holds(
                &and_(&[gt(&parameter, &zero())?, lt(&parameter, &Basic::one())?]),
                facts,
            )? {
                let q: Basic = positive();
                forward.insert(parameter.clone(), exp(&crate::core::mul::neg(&q)?)?);
                back.insert(q, crate::core::mul::neg(&log(&parameter)?)?);
                continue;
            }
        }
        if let Some(dummy) = refiner.dummy_for(&parameter)? {
            forward.insert(parameter.clone(), dummy.clone());
            back.insert(dummy, parameter);
        }
    }
    Ok((replaced(expr, &forward)?, back))
}

fn is_base(parameter: &Basic, expr: &Basic, x: &Basic) -> bool {
    expr.preorder()
        .iter()
        .any(|p| p.is_pow() && p.args()[0] == *parameter && p.args()[1].has(x))
}

fn known_value(parameter: &Basic, facts: &Facts) -> Option<Basic> {
    for conjunct in &facts.conjuncts {
        if matches!(conjunct.head(), Head::Relational(RelOp::Eq)) {
            let (lhs, rhs) = (&conjunct.args()[0], &conjunct.args()[1]);
            if lhs == parameter && rhs.free_symbols().is_empty() {
                return Some(rhs.clone());
            }
            if rhs == parameter && lhs.free_symbols().is_empty() {
                return Some(lhs.clone());
            }
        }
    }
    None
}

fn sign_dependency(message: &str, back: &HashMap<Basic, Basic>) -> PolyResult<Option<Basic>> {
    let Some(at) = message.find("sign of ") else {
        return Ok(None);
    };
    let text: &str = message[at + "sign of ".len()..].trim();
    let Ok(mut e) = crate::core::parsing::parse(text, true) else {
        return Ok(None);
    };
    let names: HashMap<Basic, Basic> = back
        .keys()
        .filter(|d| d.is_symbol())
        .map(|d| (crate::core::symbol::Symbol::new(&d.to_string(), &[]).0, d.clone()))
        .collect();
    e = replaced(&e, &names)?;
    if matches!(e.head(), Head::Function(Function::Sign)) {
        e = e.args()[0].clone();
    }
    Ok(Some(replaced(&e, back)?))
}

fn cases(e: &Basic, facts: &Facts, assumptions: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    if is(e, Predicate::extended_real) == Some(false) || has_bounds(e) {
        return Ok(Vec::new());
    }
    let one: Basic = Basic::one();
    let mut candidates: Vec<Basic> = vec![gt(e, &zero())?, eq(e, &zero())?, lt(e, &zero())?];
    if matches!(e.head(), Head::Function(Function::Log)) {
        let u: &Basic = &e.args()[0];
        let minus_one: Basic = int(-1);
        candidates = if holds(&gt(u, &zero())?, facts)? {
            vec![gt(u, &one)?, eq(u, &one)?, lt(u, &one)?]
        } else {
            vec![
                gt(u, &one)?,
                eq(u, &one)?,
                and_(&[gt(u, &zero())?, lt(u, &one)?]),
                eq(u, &zero())?,
                and_(&[gt(u, &minus_one)?, lt(u, &zero())?]),
                eq(u, &minus_one)?,
                lt(u, &minus_one)?,
            ]
        };
    }
    for case in candidates {
        let case_: Basic = normalize(&case)?;
        match evaluate(&case_, facts)? {
            Some(false) => continue,
            Some(true) => return Ok(vec![true_()]),
            None => {}
        }
        if matches!(
            crate::assumptions::sat::satisfiable(&case_, assumptions, None)?,
            crate::assumptions::sat::Sat::Unsatisfiable
        ) {
            continue;
        }
        result.push(case_);
    }
    Ok(result)
}

/// The limit of `expr` as `x -> x0` from `direction` under the
/// assumptions (`limit`): a `Piecewise` on the signs of the parameters
/// it depends on, the unevaluated `Limit` when it is not found.
pub fn limit(expr: &Basic, x: &Basic, x0: &Basic, direction: &str, assumptions: &[Basic], domain: Option<&Basic>) -> R {
    let mut symbols: Vec<Basic> = expr.free_symbols().into_iter().collect();
    symbols.push(x.clone());
    let domain: Basic = domain.cloned().unwrap_or_else(reals);
    let facts: Facts = crate::assumptions::ask::facts_of(assumptions, Some(&domain), &symbols)?;
    limit_under(expr, x, x0, direction, &facts, assumptions, 0, None)
}

fn continuous(e: &Basic, x: &Basic, x0: &Basic, direction: &str) -> R {
    crate::core::limits::limit(e, x, x0, direction)
}

fn sequence(e: &Basic, n: &Basic) -> R {
    if let Some(value) = crate::series::limitseq::limit_seq(e, Some(n), 5)? {
        return Ok(value);
    }
    let value: Basic = crate::core::limits::limit(e, n, &Basic::oo(), "-")?;
    if is_limit(&value) || crate::core::accumbounds::is_bounds(&value) {
        return crate::core::limits::limit_node(e, n, &Basic::oo(), "-");
    }
    Ok(value)
}

fn decide_sign(e: &Basic, facts: &Facts) -> PolyResult<Option<Basic>> {
    if holds(&gt(e, &zero())?, facts)? {
        return Ok(Some(Basic::one()));
    }
    if holds(&lt(e, &zero())?, facts)? {
        return Ok(Some(int(-1)));
    }
    if holds(&eq(e, &zero())?, facts)? {
        return Ok(Some(zero()));
    }
    Ok(None)
}

fn nonpositive_base(expr: &Basic, x: &Basic, x0: &Basic, facts: &Facts) -> PolyResult<Option<Basic>> {
    if !is_infinity(x0) {
        return Ok(None);
    }
    let (coefficient, power) = crate::core::expr::as_independent(expr, x, false)?;
    if !power.is_pow() {
        return Ok(None);
    }
    let (base, exponent) = (&power.args()[0], &power.args()[1]);
    let slope: Basic = crate::core::function::derivative(exponent, x)?;
    let linear: Basic = crate::core::add::sub(exponent, &crate::core::mul::mul(&[slope.clone(), x.clone()])?)?;
    if base.free_symbols().contains(x) || slope.free_symbols().contains(x) || linear != zero() {
        return Ok(None);
    }
    let Some(upwards) = decide_sign(&slope, facts)? else {
        return Ok(None);
    };
    if upwards == zero() {
        return Ok(None);
    }
    let growing: bool = (upwards == Basic::one()) == matches!(x0.head(), Head::Infinity);
    let zoo: Basic = Basic::atom(Head::ComplexInfinity);
    if holds(&eq(base, &zero())?, facts)? {
        return Ok(Some(if growing { zero() } else { zoo }));
    }
    if !holds(&lt(base, &zero())?, facts)? {
        return Ok(None);
    }
    if holds(&eq(base, &int(-1))?, facts)? {
        return Ok(Some(Basic::atom(Head::NaN)));
    }
    let small: Option<bool> = evaluate(&normalize(&and_(&[gt(base, &int(-1))?, lt(base, &zero())?]))?, facts)?;
    let large: Option<bool> = evaluate(&normalize(&lt(base, &int(-1))?)?, facts)?;
    if small != Some(true) && large != Some(true) {
        return Ok(None);
    }
    if (small == Some(true)) == growing {
        return Ok(Some(zero()));
    }
    if coefficient != Basic::one() {
        return Ok(Some(crate::core::mul::mul(&[zoo, coefficient])?));
    }
    Ok(Some(zoo))
}

fn run(engine: Option<Engine>, e: &Basic, x: &Basic, x0: &Basic, direction: &str) -> R {
    match engine {
        Some(f) => f(e, x),
        None => continuous(e, x, x0, direction),
    }
}

fn limit_under(
    expr: &Basic,
    x: &Basic,
    x0: &Basic,
    direction: &str,
    facts: &Facts,
    assumptions: &[Basic],
    depth: usize,
    engine: Option<Engine>,
) -> R {
    crate::timeout::check()?;
    let unevaluated = || crate::core::limits::limit_node(expr, x, x0, direction);
    if engine.is_none() {
        if let Some(special) = nonpositive_base(expr, x, x0, facts)? {
            return Ok(special);
        }
    }
    let (abstracted, back) = abstract_(expr, facts, x)?;
    let value: Basic = match run(engine, &abstracted, x, x0, direction) {
        Ok(v) => v,
        Err(PolyError::NotImplemented(message)) if !message.starts_with(crate::error::NOT_PORTED_PREFIX) => {
            let Some(dependency) = sign_dependency(&message, &back)? else {
                return unevaluated();
            };
            if depth > MAX_DEPTH {
                return unevaluated();
            }
            return split(expr, x, x0, direction, facts, assumptions, &dependency, depth, engine);
        }
        Err(PolyError::Value(_) | PolyError::Type(_) | PolyError::Pole(_)) => return unevaluated(),
        Err(e) => return Err(e),
    };
    let mut result: Basic = replaced(&value, &back)?;
    let undecided_: Option<Basic> = undecided(&result)?;
    if let Some(u) = &undecided_ {
        if depth <= MAX_DEPTH {
            if let Some(decided) = decide_sign(u, facts)? {
                let sign: Basic = Basic::new(Head::Function(Function::Sign), vec![u.clone()]);
                let rule: HashMap<Basic, Basic> = [(sign, decided.clone())].into_iter().collect();
                result = replaced(&result, &rule)?;
                return finish(
                    &result,
                    u,
                    &decided,
                    expr,
                    x,
                    x0,
                    direction,
                    facts,
                    assumptions,
                    depth,
                    engine,
                );
            }
            return split(expr, x, x0, direction, facts, assumptions, u, depth, engine);
        }
        return unevaluated();
    }
    if is_limit(&result) && depth <= MAX_DEPTH {
        for parameter in parameters(expr, x) {
            if cases(&parameter, facts, assumptions)?.len() > 1 {
                return split(expr, x, x0, direction, facts, assumptions, &parameter, depth, engine);
            }
        }
    }
    if depth <= MAX_DEPTH && !is_limit(&result) {
        let engine_: &dyn Fn(&Basic, &Basic) -> R = &|e: &Basic, v: &Basic| run(engine, e, v, x0, direction);
        if let Some(degenerate) = degenerate_parameter(&result, expr, x, facts, assumptions, engine_)? {
            return split(expr, x, x0, direction, facts, assumptions, &degenerate, depth, engine);
        }
    }
    undirected(&result, expr, x, x0, direction)
}

fn engine_value(engine: Engine, instance: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let value = match crate::timeout::attempt_within(crate::settings::settings().timeout, || engine(instance, x)) {
        Ok(v) => v,
        Err(e) if not_ported(&e) => return Err(e),
        Err(PolyError::TimeLimit(m)) => return Err(PolyError::TimeLimit(m)),
        Err(_) => return Ok(None),
    };
    match value {
        Some(v) if !is_limit(&v) && !has_bounds(&v) => Ok(Some(v)),
        _ => Ok(None),
    }
}

fn degenerate_parameter(
    result: &Basic,
    expr: &Basic,
    x: &Basic,
    facts: &Facts,
    assumptions: &[Basic],
    engine: Engine,
) -> PolyResult<Option<Basic>> {
    for parameter in parameters(expr, x) {
        if cases(&parameter, facts, assumptions)?.len() <= 1 {
            continue;
        }
        let rule: HashMap<Basic, Basic> = [(parameter.clone(), zero())].into_iter().collect();
        let generic: Basic = replaced(result, &rule)?;
        let instance: Basic = replaced(expr, &rule)?;
        if has_head(&generic, &|n| matches!(n.head(), Head::ComplexInfinity | Head::NaN)) {
            return Ok(Some(parameter));
        }
        if let Some(value) = engine_value(engine, &instance, x)? {
            if same_limit(&value, &generic)? == Some(false) {
                return Ok(Some(parameter));
            }
        }
    }
    Ok(None)
}

fn same_limit(first: &Basic, second: &Basic) -> PolyResult<Option<bool>> {
    if first == second {
        return Ok(Some(true));
    }
    if is(first, Predicate::infinite) == Some(true) || is(second, Predicate::infinite) == Some(true) {
        return Ok(Some(first == second));
    }
    let difference: Basic = crate::core::add::sub(first, second)?;
    let simplified = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::simplify::simplify::simplify(&difference, &crate::simplify::simplify::Options::default())
    })?;
    let Some(d) = simplified else { return Ok(None) };
    if d == zero() {
        return Ok(Some(true));
    }
    if crate::core::assumptions::is_number(&d) && is(&d, Predicate::zero) == Some(false) {
        return Ok(Some(false));
    }
    Ok(None)
}

fn undirected(result: &Basic, expr: &Basic, x: &Basic, x0: &Basic, direction: &str) -> R {
    let zoo: Basic = Basic::atom(Head::ComplexInfinity);
    if !(is(result, Predicate::infinite) == Some(true) && *result != zoo && !matches!(result.head(), Head::NaN)) {
        return Ok(result.clone());
    }
    let free = expr.free_symbols();
    if free.len() != 1 || !free.contains(x) {
        return Ok(result.clone());
    }
    for sample in approach_samples(x0, direction)? {
        let Some(number) = crate::numeric::reliable_value(expr, 30, &[(x.clone(), sample)]) else {
            return Ok(result.clone());
        };
        let (re, im) = crate::core::evalf::parts(&number);
        let (re, im) = (re.to_f64().value(), im.to_f64().value());
        if !re.is_finite() || !im.is_finite() {
            return Ok(result.clone());
        }
        if im.abs() > 1e-12 * re.abs().max(1.0) {
            return Ok(zoo);
        }
    }
    Ok(result.clone())
}

fn approach_samples(x0: &Basic, direction: &str) -> PolyResult<Vec<Basic>> {
    let r = |p: i64, q: i64| crate::core::mul::div(&int(p), &int(q));
    match x0.head() {
        Head::Infinity => return Ok(vec![int(7), r(31, 2)?, int(101)]),
        Head::NegativeInfinity => return Ok(vec![int(-7), r(-31, 2)?, int(-101)]),
        _ => {}
    }
    if !crate::core::assumptions::is_number(x0) {
        return Ok(Vec::new());
    }
    let mut out: Vec<Basic> = Vec::new();
    for q in [10, 100, 1000] {
        let h: Basic = r(1, q)?;
        let step: Basic = if direction == "+" {
            h
        } else {
            crate::core::mul::neg(&h)?
        };
        out.push(crate::core::add::add(&[x0.clone(), step])?);
    }
    Ok(out)
}

fn sign_case(undecided: &Basic, decided: &Basic) -> R {
    if *decided == Basic::one() {
        return gt(undecided, &zero());
    }
    if *decided == int(-1) {
        return lt(undecided, &zero());
    }
    eq(undecided, &zero())
}

fn finish(
    result: &Basic,
    undecided_: &Basic,
    decided: &Basic,
    expr: &Basic,
    x: &Basic,
    x0: &Basic,
    direction: &str,
    facts: &Facts,
    assumptions: &[Basic],
    depth: usize,
    engine: Option<Engine>,
) -> R {
    if undecided(result)?.is_none() {
        return Ok(result.clone());
    }
    let extra: Vec<Basic> = with(&sign_case(undecided_, decided)?, assumptions);
    let facts2: Facts = case_facts(&extra, facts)?;
    limit_under(expr, x, x0, direction, &facts2, &[and_(&extra)], depth + 1, engine)
}

fn undecided(result: &Basic) -> PolyResult<Option<Basic>> {
    let nodes: Vec<Basic> = result.preorder();
    for n in &nodes {
        if matches!(n.head(), Head::Function(Function::Sign)) {
            return Ok(Some(n.args()[0].clone()));
        }
    }
    for n in &nodes {
        if n.is_pow() && is_infinity(&n.args()[0]) && !n.args()[1].free_symbols().is_empty() {
            return Ok(Some(n.args()[1].clone()));
        }
    }
    for n in &nodes {
        if n.is_mul() && n.args().iter().any(is_infinity) && !n.free_symbols().is_empty() {
            let others: Vec<Basic> = n.args().iter().filter(|a| !is_infinity(a)).cloned().collect();
            if !others.is_empty() {
                return Ok(Some(crate::core::mul::mul(&others)?));
            }
        }
    }
    Ok(None)
}

fn split(
    expr: &Basic,
    x: &Basic,
    x0: &Basic,
    direction: &str,
    facts: &Facts,
    assumptions: &[Basic],
    dependency: &Basic,
    depth: usize,
    engine: Option<Engine>,
) -> R {
    let mut pieces: Vec<(Basic, Basic)> = Vec::new();
    for case in cases(dependency, facts, assumptions)? {
        let extra: Vec<Basic> = with(&case, assumptions);
        let facts2: Facts = case_facts(&extra, facts)?;
        let combined: Vec<Basic> = vec![and_(&extra)];
        if is_true(&case) {
            return limit_under(expr, x, x0, direction, &facts2, &combined, depth + 1, engine);
        }
        let value: Basic = limit_under(expr, x, x0, direction, &facts2, &combined, depth + 1, engine)?;
        let value: Basic = verified_piece(&value, &case, expr, x, x0, direction, &facts2, &combined, depth, engine)?;
        pieces.push((value, case));
    }
    if pieces.is_empty() {
        return crate::core::limits::limit_node(expr, x, x0, direction);
    }
    piecewise(&pieces)
}

fn verified_piece(
    value: &Basic,
    case: &Basic,
    expr: &Basic,
    x: &Basic,
    x0: &Basic,
    direction: &str,
    facts: &Facts,
    assumptions: &[Basic],
    depth: usize,
    engine: Option<Engine>,
) -> R {
    if is_limit(value) || matches!(value.head(), Head::Piecewise) || has_bounds(value) || depth > MAX_DEPTH {
        return Ok(value.clone());
    }
    let params: Vec<Basic> = parameters(expr, x);
    let engine_: &dyn Fn(&Basic, &Basic) -> R = &|e: &Basic, v: &Basic| run(engine, e, v, x0, direction);
    let mut wrong: bool = false;
    for point in points_of(case, &params, 4)? {
        let Some(instance) = engine_value(engine_, &replaced(expr, &point)?, x)? else {
            continue;
        };
        if same_limit(&instance, &replaced(value, &point)?)? == Some(false) {
            wrong = true;
            break;
        }
    }
    if !wrong {
        return Ok(value.clone());
    }
    for (i, first) in params.iter().enumerate() {
        for second in &params[i + 1..] {
            let difference: Basic = crate::core::add::sub(first, second)?;
            if cases(&difference, facts, assumptions)?.len() > 1 {
                return split(
                    expr,
                    x,
                    x0,
                    direction,
                    facts,
                    assumptions,
                    &difference,
                    depth + 1,
                    engine,
                );
            }
        }
    }
    crate::core::limits::limit_node(expr, x, x0, direction)
}

fn points_of(case: &Basic, params: &[Basic], count: usize) -> PolyResult<Vec<HashMap<Basic, Basic>>> {
    let r = |p: i64, q: i64| crate::core::mul::div(&int(p), &int(q));
    let candidates: Vec<Basic> = vec![
        int(2),
        r(1, 2)?,
        int(-1),
        int(3),
        r(-5, 2)?,
        int(0),
        int(1),
        int(5),
        int(-3),
    ];
    let mut found: Vec<HashMap<Basic, Basic>> = Vec::new();
    let k: usize = params.len();
    let total: usize = candidates.len().pow(k as u32);
    for index in 0..total {
        // itertools.product order: the last parameter varies fastest
        let mut choice: Vec<usize> = vec![0; k];
        let mut rest: usize = index;
        for j in (0..k).rev() {
            choice[j] = rest % candidates.len();
            rest /= candidates.len();
        }
        let point: HashMap<Basic, Basic> = params
            .iter()
            .cloned()
            .zip(choice.iter().map(|c| candidates[*c].clone()))
            .collect();
        if is_true(&replaced(case, &point)?) {
            found.push(point);
            if found.len() >= count {
                break;
            }
        }
    }
    Ok(found)
}

fn piecewise(pieces: &[(Basic, Basic)]) -> R {
    let mut flat: Vec<(Basic, Basic)> = Vec::new();
    for (value, condition) in pieces {
        if matches!(value.head(), Head::Piecewise) {
            let mut previous: Vec<Basic> = Vec::new();
            for pair in value.args() {
                let (e, c) = (&pair.args()[0], &pair.args()[1]);
                let mut parts: Vec<Basic> = vec![condition.clone(), c.clone()];
                parts.extend(previous.iter().map(crate::core::logic::not_));
                flat.push((e.clone(), and_(&parts)));
                previous.push(c.clone());
            }
        } else {
            flat.push((value.clone(), condition.clone()));
        }
    }
    let mut merged: Vec<(Basic, Vec<Basic>)> = Vec::new();
    for (value, condition) in flat {
        match merged.iter_mut().find(|(v, _)| *v == value) {
            Some(entry) => entry.1.push(condition),
            None => merged.push((value, vec![condition])),
        }
    }
    if merged.len() == 1 {
        return Ok(merged[0].0.clone());
    }
    let mut pairs: Vec<(Basic, Basic)> = Vec::new();
    for (value, conditions) in merged {
        pairs.push((value, condition(&crate::core::logic::or_(&conditions))?));
    }
    crate::functions::elementary::piecewise::piecewise(&pairs)
}

fn condition(formula: &Basic) -> R {
    let simplified = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::assumptions::resolve::resolve(formula, None, &[], None)
    })?;
    match simplified {
        Some(s) if crate::core::logic::is_boolean(&s) => Ok(s),
        _ => Ok(formula.clone()),
    }
}

/// The limit of the sequence `expr` as `n -> oo` under the assumptions
/// (`limit_seq`), with cases on the signs of the bases of powers.
pub fn limit_seq(expr: &Basic, n: &Basic, assumptions: &[Basic], domain: Option<&Basic>) -> R {
    let mut symbols: Vec<Basic> = expr.free_symbols().into_iter().collect();
    symbols.push(n.clone());
    let domain: Basic = domain.cloned().unwrap_or_else(reals);
    let facts: Facts = crate::assumptions::ask::facts_of(assumptions, Some(&domain), &symbols)?;
    limit_seq_under(expr, n, &facts, assumptions, 0)
}

fn limit_seq_under(expr: &Basic, n: &Basic, facts: &Facts, assumptions: &[Basic], depth: usize) -> R {
    let mut powers: Vec<Basic> = Vec::new();
    for p in expr.preorder() {
        if p.is_pow() && !powers.contains(&p) {
            powers.push(p);
        }
    }
    powers.sort_by_key(|p| p.to_string());
    let mut negative: Vec<Basic> = Vec::new();
    let oo: Basic = Basic::oo();
    for power in powers {
        let (base, exponent) = (&power.args()[0], &power.args()[1]);
        if !(exponent.has(n) && !base.has(n) && !base.free_symbols().is_empty()) {
            continue;
        }
        let s: Option<Basic> = decide_sign(base, facts)?;
        if s.is_none() && depth <= MAX_DEPTH {
            let mut pieces: Vec<(Basic, Basic)> = Vec::new();
            for case in cases(base, facts, assumptions)? {
                let extra: Vec<Basic> = with(&case, assumptions);
                let facts2: Facts = case_facts(&extra, facts)?;
                let value: Basic = limit_seq_under(expr, n, &facts2, &[and_(&extra)], depth + 1)?;
                if is_true(&case) {
                    return Ok(value);
                }
                pieces.push((value, case));
            }
            return if pieces.is_empty() {
                crate::core::limits::limit_node(expr, n, &oo, "-")
            } else {
                piecewise(&pieces)
            };
        }
        if s == Some(int(-1)) {
            negative.push(power);
        }
    }
    let seq: &dyn Fn(&Basic, &Basic) -> R = &sequence;
    if !negative.is_empty() {
        let mut magnitude: Basic = expr.clone();
        for power in &negative {
            let flipped: Basic = crate::core::power::pow(&crate::core::mul::neg(&power.args()[0])?, &power.args()[1])?;
            let rule: HashMap<Basic, Basic> = [(power.clone(), flipped)].into_iter().collect();
            magnitude = replaced(&magnitude, &rule)?;
        }
        let size: Basic = limit_under(&magnitude, n, &oo, "-", facts, assumptions, depth, Some(seq))?;
        let pieces: Vec<(Basic, Basic)> = if matches!(size.head(), Head::Piecewise) {
            size.args()
                .iter()
                .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
                .collect()
        } else {
            vec![(size, true_())]
        };
        let mut out: Vec<(Basic, Basic)> = Vec::new();
        for (v, c) in pieces {
            let value: Basic = if v == zero() {
                zero()
            } else {
                crate::core::limits::limit_node(expr, n, &oo, "-")?
            };
            out.push((value, c));
        }
        return piecewise(&out);
    }
    limit_under(expr, n, &oo, "-", facts, assumptions, depth, Some(seq))
}

/// The series of `expr` in `x` at `x0` under the assumptions (`series`).
pub fn series(expr: &Basic, x: &Basic, x0: &Basic, n: i64, assumptions: &[Basic], domain: Option<&Basic>) -> R {
    let mut symbols: Vec<Basic> = expr.free_symbols().into_iter().collect();
    symbols.push(x.clone());
    let domain: Basic = domain.cloned().unwrap_or_else(reals);
    let facts: Facts = crate::assumptions::ask::facts_of(assumptions, Some(&domain), &symbols)?;
    let (abstracted, back) = abstract_(expr, &facts, x)?;
    let expansion: Basic = crate::core::series::series(&abstracted, Some(x), x0, n, "+", None, &Basic::zero())?;
    replaced(&expansion, &back)
}
