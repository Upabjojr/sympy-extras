//! Port of `sympy_extras/calculus/optimization.py`: exact global optima
//! of rational functions under polynomial constraints by quantifier
//! elimination: the set of values `{t : exists x, f(x) = t and the
//! constraints}`, its infimum or supremum, whether attained and where; the
//! optimum as a function of parameters by cylindrical cases.

use crate::core::basic::{Basic, Head, RelOp};
use crate::core::logic::boolean;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::linalg::normal;
use crate::polys::cad::qe::Quantifier;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rel(op: RelOp, a: &Basic, b: &Basic) -> Basic {
    Basic::new(Head::Relational(op), vec![a.clone(), b.clone()])
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}
fn plain_symbols(e: &Basic) -> Vec<Basic> {
    e.free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_)))
        .collect()
}
fn sorted(mut v: Vec<Basic>) -> Vec<Basic> {
    v.sort_by_cached_key(crate::core::sorting::sort_key);
    v
}
fn shown(v: &[Basic]) -> String {
    format!(
        "[{}]",
        v.iter().map(|s| s.to_string()).collect::<Vec<String>>().join(", ")
    )
}

/// An optimum: the value, whether attained, the points where
/// (`Optimum`).
#[derive(Debug, Clone)]
pub struct Optimum {
    pub value: Basic,
    pub attained: bool,
    pub points: Vec<Vec<(Basic, Basic)>>,
}

impl Optimum {
    pub fn is_finite(&self) -> bool {
        !matches!(self.value.head(), Head::Infinity | Head::NegativeInfinity)
    }
}

fn fresh(used: &[String]) -> Basic {
    let (mut name, mut k) = ("t".to_string(), 0);
    while used.contains(&name) {
        name = format!("t{}", k);
        k += 1;
    }
    symbol(&name)
}

struct Problem {
    variables: Vec<Basic>,
    value: Basic,
    formula: Basic,
}

impl Problem {
    fn new(objective: &Basic, condition: &Basic, variables: Option<&[Basic]>) -> PolyResult<Problem> {
        let mut symbols: Vec<Basic> = plain_symbols(objective);
        for s in plain_symbols(condition) {
            if !symbols.contains(&s) {
                symbols.push(s);
            }
        }
        let chosen: Vec<Basic> = match variables {
            None => sorted(symbols.clone()),
            Some(v) => {
                let extra: Vec<Basic> = symbols.iter().filter(|s| !v.contains(s)).cloned().collect();
                if !extra.is_empty() {
                    return Err(value_error(format!(
                        "the parameters {} are not allowed: every symbol is a variable of the optimisation",
                        shown(&sorted(extra))
                    )));
                }
                v.to_vec()
            }
        };
        let mut used: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
        used.extend(chosen.iter().map(|s| s.to_string()));
        let value: Basic = fresh(&used);
        let (numerator, denominator) = crate::simplify::radsimp::fraction(&normal(objective)?)?;
        let mut parts: Vec<Basic> = vec![rel(
            RelOp::Eq,
            &numerator,
            &crate::core::mul::mul(&[value.clone(), denominator.clone()])?,
        )];
        if !denominator.free_symbols().is_empty() {
            parts.push(rel(RelOp::Ne, &denominator, &int(0)));
        }
        if *condition != boolean(true) {
            parts.push(condition.clone());
        }
        let formula: Basic = if parts.len() == 1 {
            parts[0].clone()
        } else {
            Basic::new(Head::And, parts)
        };
        Ok(Problem {
            variables: chosen,
            value,
            formula,
        })
    }

    fn range(&self) -> R {
        let prefix: Vec<(Quantifier, Basic)> = self.variables.iter().map(|v| (Quantifier::Exists, v.clone())).collect();
        crate::polys::cad::qe::solution_set(&self.formula, &self.value, &prefix, None, true)
    }

    fn points(&self, optimum: &Basic) -> PolyResult<Vec<Vec<(Basic, Basic)>>> {
        let mut gens: Vec<Basic> = vec![self.value.clone()];
        gens.extend(self.variables.iter().cloned());
        let mut found: Vec<Vec<(Basic, Basic)>> = Vec::new();
        for point in crate::polys::cad::qe::sample_points(&self.formula, &gens, None, true)? {
            let at = |v: &Basic| {
                point
                    .iter()
                    .find(|(k, _)| k == v)
                    .map(|(_, x)| x.clone())
                    .expect("a coordinate")
            };
            if crate::assumptions::sat::compare(&at(&self.value), optimum)? == 0 {
                found.push(self.variables.iter().map(|v| (v.clone(), at(v))).collect());
            }
        }
        Ok(found)
    }
}

type Piece = (Basic, bool, Basic, bool);

fn pieces(s: &Basic) -> PolyResult<Vec<Piece>> {
    match s.head() {
        Head::EmptySet => Ok(Vec::new()),
        Head::Interval | Head::Reals => {
            let (start, end, left_open, right_open) = crate::sets::sets::interval_parts(s);
            Ok(vec![(start, !left_open, end, !right_open)])
        }
        Head::FiniteSet => Ok(s.args().iter().map(|e| (e.clone(), true, e.clone(), true)).collect()),
        Head::Union => {
            let mut collected: Vec<Piece> = Vec::new();
            for part in s.args() {
                collected.extend(pieces(part)?);
            }
            Ok(collected)
        }
        _ => Err(PolyError::NotImplemented(format!(
            "the set of values {} is not a union of intervals and points",
            s
        ))),
    }
}

fn less(a: &Basic, b: &Basic) -> PolyResult<bool> {
    if a == b {
        return Ok(false);
    }
    if matches!(a.head(), Head::NegativeInfinity) || matches!(b.head(), Head::Infinity) {
        return Ok(true);
    }
    if matches!(a.head(), Head::Infinity) || matches!(b.head(), Head::NegativeInfinity) {
        return Ok(false);
    }
    Ok(crate::assumptions::sat::compare(a, b)? < 0)
}

fn extreme(s: &Basic, lowest: bool) -> PolyResult<(Basic, bool)> {
    let mut best: Option<(Basic, bool)> = None;
    for (low, low_in, high, high_in) in pieces(s)? {
        let (end, inside) = if lowest { (low, low_in) } else { (high, high_in) };
        let better: bool = match &best {
            None => true,
            Some((b, _)) => {
                if lowest {
                    less(&end, b)?
                } else {
                    less(b, &end)?
                }
            }
        };
        if better {
            best = Some((end, inside));
        } else if let Some((b, _)) = &best {
            if end == *b && inside {
                best = Some((end, true));
            }
        }
    }
    let Some(best) = best else {
        return Err(value_error("the constraints have no solution".to_string()));
    };
    if matches!(best.0.head(), Head::Infinity | Head::NegativeInfinity) {
        return Ok((best.0, false));
    }
    Ok(best)
}

/// The set of values of `f` under the constraints (`value_range`).
pub fn value_range(f: &Basic, constraints: &Basic, variables: Option<&[Basic]>) -> R {
    Problem::new(f, constraints, variables)?.range()
}

fn optimize(f: &Basic, constraints: &Basic, variables: Option<&[Basic]>, lowest: bool) -> PolyResult<Optimum> {
    let problem: Problem = Problem::new(f, constraints, variables)?;
    let (value, attained) = extreme(&problem.range()?, lowest)?;
    let points = if attained { problem.points(&value)? } else { Vec::new() };
    Ok(Optimum {
        value,
        attained,
        points,
    })
}

/// The infimum of `f` under the constraints (`minimize`).
pub fn minimize(f: &Basic, constraints: &Basic, variables: Option<&[Basic]>) -> PolyResult<Optimum> {
    optimize(f, constraints, variables, true)
}

/// The supremum (`maximize`).
pub fn maximize(f: &Basic, constraints: &Basic, variables: Option<&[Basic]>) -> PolyResult<Optimum> {
    optimize(f, constraints, variables, false)
}

fn parametric(
    f: &Basic,
    condition: &Basic,
    variables: Option<&[Basic]>,
    names: &[Basic],
    lowest: bool,
) -> PolyResult<Vec<(Basic, Basic)>> {
    if names.is_empty() {
        return Err(value_error("no parameter: use minimize or maximize".to_string()));
    }
    let mut symbols: Vec<Basic> = plain_symbols(f);
    for s in plain_symbols(condition) {
        if !symbols.contains(&s) {
            symbols.push(s);
        }
    }
    let chosen: Vec<Basic> = match variables {
        None => sorted(symbols.iter().filter(|s| !names.contains(s)).cloned().collect()),
        Some(v) => v.to_vec(),
    };
    if chosen.iter().any(|c| names.contains(c)) || symbols.iter().any(|s| !chosen.contains(s) && !names.contains(s)) {
        return Err(value_error(
            "every symbol is a variable or a parameter, and none is both".to_string(),
        ));
    }
    let (_, denominator) = crate::simplify::radsimp::fraction(&normal(f)?)?;
    if !denominator.is_rational_class() {
        return Err(PolyError::NotImplemented(
            "a parametric optimum of a rational function: the objective must be a polynomial".to_string(),
        ));
    }
    let mut used: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    used.extend(chosen.iter().map(|s| s.to_string()));
    used.extend(names.iter().map(|s| s.to_string()));
    let bound: Basic = fresh(&used);
    let inner: Vec<(Quantifier, Basic)> = chosen.iter().map(|v| (Quantifier::Forall, v.clone())).collect();
    let mut beyond: Vec<(Quantifier, Basic)> = vec![(Quantifier::Forall, bound.clone())];
    beyond.extend(chosen.iter().map(|v| (Quantifier::Exists, v.clone())));
    let mut within: Basic = rel(if lowest { RelOp::Ge } else { RelOp::Le }, f, &bound);
    let mut outside: Basic = rel(if lowest { RelOp::Lt } else { RelOp::Gt }, f, &bound);
    if *condition != boolean(true) {
        within = Basic::new(Head::Implies, vec![condition.clone(), within]);
        outside = Basic::new(Head::And, vec![condition.clone(), outside]);
    }
    let mut results: Vec<(Basic, Basic)> = Vec::new();
    let cases = crate::polys::cad::cylindrical::cylindrical_cases_quantified(
        &within,
        names,
        std::slice::from_ref(&bound),
        None,
        true,
        &inner,
    )?;
    for (case, bounds) in cases {
        if !matches!(bounds.head(), Head::Interval | Head::Reals) {
            return Err(PolyError::NotImplemented(format!(
                "the set of the bounds {} is not an interval",
                bounds
            )));
        }
        let (start, finish, _, _) = crate::sets::sets::interval_parts(&bounds);
        let (end, other) = if lowest { (finish, start) } else { (start, finish) };
        let infinite = |e: &Basic| matches!(e.head(), Head::Infinity | Head::NegativeInfinity);
        if infinite(&end) || !infinite(&other) {
            continue;
        }
        results.push((case, end));
    }
    let unbounded: Basic = crate::polys::cad::cylindrical::cylindrical_formula(&outside, names, &beyond, None, true)?;
    if unbounded != boolean(false) {
        results.push((unbounded, if lowest { Basic::negative_oo() } else { Basic::oo() }));
    }
    Ok(results)
}

/// The minimum as a function of the parameters (`minimize_parametric`).
pub fn minimize_parametric(
    f: &Basic,
    constraints: &Basic,
    variables: Option<&[Basic]>,
    parameters: &[Basic],
) -> PolyResult<Vec<(Basic, Basic)>> {
    parametric(f, constraints, variables, parameters, true)
}

/// The maximum as a function of the parameters (`maximize_parametric`).
pub fn maximize_parametric(
    f: &Basic,
    constraints: &Basic,
    variables: Option<&[Basic]>,
    parameters: &[Basic],
) -> PolyResult<Vec<(Basic, Basic)>> {
    parametric(f, constraints, variables, parameters, false)
}
