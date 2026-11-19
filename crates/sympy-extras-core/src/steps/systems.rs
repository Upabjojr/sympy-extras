//! Port of `sympy_extras/steps/systems.py`: systems of equations solved
//! step by step by substitution and elimination, the single equations by
//! [`super::equations`]. The steps of the equations are substeps of the
//! steps of the systems: a tree of one type holds them as
//! [`SystemStep::Equation`].

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::{value_error, PolyResult};
use crate::rules;

use super::core::{Context, Rule, Step, StepSolver};
use super::equations::{solve_steps, Domain, EquationStep};

type Equation = (Basic, Basic);
type Solution = Vec<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn expand(e: &Basic) -> PolyResult<Basic> {
    crate::core::function::expand_default(e)
}
fn normal(e: &Basic) -> PolyResult<Basic> {
    crate::simplify::normalform::normal_form(e)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Basic> {
    crate::core::subs::subs(e, old, new)
}

/// Equations in unknowns over a domain (`SystemProblem`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SystemProblem {
    pub equations: Vec<Equation>,
    pub variables: Vec<Basic>,
    pub domain: Domain,
}

type B = Box<SystemStep>;

/// A step of the solution of a system (the rule dataclasses), or the step
/// of a single equation inside it.
#[derive(Debug, Clone, PartialEq)]
pub enum SystemStep {
    SingleEquationRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
        step: EquationStep,
    },
    ConsistentRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
    },
    InconsistentRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
    },
    SubstitutionRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
        index: usize,
        variable: Basic,
        isolation: EquationStep,
        expressions: Vec<Basic>,
        remaining: Vec<Basic>,
        cases: Vec<SystemStep>,
        rejected: Vec<Solution>,
    },
    EliminationRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
        first: usize,
        second: usize,
        first_multiplier: Basic,
        second_multiplier: Basic,
        variable: Basic,
        combined: Equation,
        substep: B,
    },
    AlternativeRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
        alternatives: Vec<SystemStep>,
    },
    DontKnowRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
    },
    InProgressRule {
        equations: Vec<Equation>,
        variables: Vec<Basic>,
    },
    /// A step of an equation, a substep of a system step.
    Equation(EquationStep),
}

use SystemStep as S;

fn completed(
    variables: &[Basic],
    remaining: &[Basic],
    partial: &[Basic],
    variable: &Basic,
    expression: &Basic,
) -> PolyResult<Solution> {
    let mut value: Basic = expression.clone();
    for (symbol, number) in remaining.iter().zip(partial.iter()) {
        value = subs(&value, symbol, number)?;
    }
    let mut values: Vec<(Basic, Basic)> = remaining.iter().cloned().zip(partial.iter().cloned()).collect();
    values.push((variable.clone(), normal(&value)?));
    Ok(variables
        .iter()
        .map(|v| {
            values
                .iter()
                .rev()
                .find(|(k, _)| k == v)
                .map(|(_, x)| x.clone())
                .expect("a value")
        })
        .collect())
}

fn satisfies(equations: &[Equation], variables: &[Basic], solution: &[Basic], real: bool) -> PolyResult<Option<bool>> {
    let mut undecided: bool = false;
    for (lhs, rhs) in equations {
        let (mut left, mut right) = (lhs.clone(), rhs.clone());
        for (symbol, value) in variables.iter().zip(solution.iter()) {
            left = subs(&left, symbol, value)?;
            right = subs(&right, symbol, value)?;
        }
        if is(&left, Predicate::finite) == Some(false) || is(&right, Predicate::finite) == Some(false) {
            return Ok(Some(false));
        }
        if real && (is(&left, Predicate::real) == Some(false) || is(&right, Predicate::real) == Some(false)) {
            return Ok(Some(false));
        }
        let difference: Basic = normal(&crate::core::add::sub(&left, &right)?)?;
        if difference == int(0) {
            continue;
        }
        if is(&difference, Predicate::zero) == Some(false)
            || is(&difference, Predicate::positive) == Some(true)
            || is(&difference, Predicate::negative) == Some(true)
        {
            return Ok(Some(false));
        }
        let verdict: Option<bool> = if difference.free_symbols().is_empty() {
            crate::simplify::structure::is_zero(&difference, &[]).unwrap_or(None)
        } else {
            None
        };
        if verdict == Some(false) {
            return Ok(Some(false));
        }
        if verdict.is_none() {
            undecided = true;
        }
    }
    Ok(if undecided { None } else { Some(true) })
}

fn tuple_str(solution: &[Basic]) -> String {
    match solution.len() {
        0 => "()".to_string(),
        1 => format!("({},)", solution[0]),
        _ => format!(
            "({})",
            solution
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        ),
    }
}

impl SystemStep {
    fn system(&self) -> Option<(&Vec<Equation>, &Vec<Basic>)> {
        match self {
            S::SingleEquationRule {
                equations, variables, ..
            }
            | S::ConsistentRule { equations, variables }
            | S::InconsistentRule { equations, variables }
            | S::SubstitutionRule {
                equations, variables, ..
            }
            | S::EliminationRule {
                equations, variables, ..
            }
            | S::AlternativeRule {
                equations, variables, ..
            }
            | S::DontKnowRule { equations, variables }
            | S::InProgressRule { equations, variables } => Some((equations, variables)),
            S::Equation(_) => None,
        }
    }

    /// The solutions, tuples in the order of the unknowns (`solutions`).
    pub fn solutions(&self) -> PolyResult<Vec<Solution>> {
        match self {
            S::SingleEquationRule { step, .. } => {
                let found: Basic = step.solutions()?;
                if matches!(found.head(), Head::EmptySet) {
                    return Ok(Vec::new());
                }
                if !matches!(found.head(), Head::FiniteSet) {
                    return Err(value_error(format!(
                        "the equation has the solutions {}: not finitely many",
                        found
                    )));
                }
                Ok(found.args().iter().map(|v| vec![v.clone()]).collect())
            }
            S::ConsistentRule { .. } => Ok(vec![Vec::new()]),
            S::InconsistentRule { .. } => Ok(Vec::new()),
            S::SubstitutionRule {
                variables,
                variable,
                expressions,
                remaining,
                cases,
                rejected,
                ..
            } => {
                let mut found: Vec<Solution> = Vec::new();
                for (expression, case) in expressions.iter().zip(cases.iter()) {
                    for partial in case.solutions()? {
                        let solution: Solution = completed(variables, remaining, &partial, variable, expression)?;
                        if !found.contains(&solution) && !rejected.contains(&solution) {
                            found.push(solution);
                        }
                    }
                }
                Ok(found)
            }
            S::EliminationRule { substep, .. } => substep.solutions(),
            S::AlternativeRule { alternatives, .. } => alternatives[0].solutions(),
            S::DontKnowRule { .. } | S::InProgressRule { .. } => {
                Err(value_error("the system is not solved".to_string()))
            }
            S::Equation(_) => Err(value_error(
                "the step of an equation has no tuples of solutions".to_string(),
            )),
        }
    }
}

impl Step for SystemStep {
    fn substeps(&self) -> Vec<SystemStep> {
        match self {
            S::SingleEquationRule { step, .. } => vec![S::Equation(step.clone())],
            S::SubstitutionRule { isolation, cases, .. } => {
                let mut v: Vec<SystemStep> = vec![S::Equation(isolation.clone())];
                v.extend(cases.iter().cloned());
                v
            }
            S::EliminationRule { substep, .. } => vec![(**substep).clone()],
            S::AlternativeRule { alternatives, .. } => alternatives.clone(),
            S::Equation(step) => step.substeps().into_iter().map(S::Equation).collect(),
            _ => Vec::new(),
        }
    }

    fn rule(&self) -> String {
        match self {
            S::SingleEquationRule { .. } => "SingleEquationRule".to_string(),
            S::ConsistentRule { .. } => "ConsistentRule".to_string(),
            S::InconsistentRule { .. } => "InconsistentRule".to_string(),
            S::SubstitutionRule { .. } => "SubstitutionRule".to_string(),
            S::EliminationRule { .. } => "EliminationRule".to_string(),
            S::AlternativeRule { .. } => "AlternativeRule".to_string(),
            S::DontKnowRule { .. } => "DontKnowRule".to_string(),
            S::InProgressRule { .. } => "InProgressRule".to_string(),
            S::Equation(step) => step.rule(),
        }
    }

    fn statement(&self) -> String {
        let S::Equation(step) = self else {
            let (equations, _) = self.system().expect("a system step");
            let system: String = equations
                .iter()
                .map(|(l, r)| format!("{} = {}", l, r))
                .collect::<Vec<String>>()
                .join(", ");
            if self.contains_dont_know() {
                return format!("{} -> ?", system);
            }
            let shown: String = match self.solutions() {
                Ok(found) => found.iter().map(|s| tuple_str(s)).collect::<Vec<String>>().join(", "),
                Err(e) => format!("<{:?}>", e),
            };
            return format!(
                "{} -> {{{}}}{}",
                system,
                shown,
                if self.complete() { "" } else { " (not all)" }
            );
        };
        step.statement()
    }

    fn minor(&self) -> bool {
        match self {
            S::Equation(step) => step.minor(),
            _ => false,
        }
    }

    fn is_alternative(&self) -> bool {
        match self {
            S::Equation(step) => step.is_alternative(),
            S::AlternativeRule { .. } => true,
            _ => false,
        }
    }

    fn is_dont_know(&self) -> bool {
        match self {
            S::Equation(step) => step.is_dont_know(),
            S::DontKnowRule { .. } | S::InProgressRule { .. } => true,
            _ => false,
        }
    }

    fn is_partial(&self) -> bool {
        match self {
            S::Equation(step) => step.is_partial(),
            _ => false,
        }
    }

    fn contains_dont_know(&self) -> bool {
        match self {
            S::Equation(step) => step.contains_dont_know(),
            _ => self.is_dont_know() || self.substeps().iter().any(Step::contains_dont_know),
        }
    }

    fn complete(&self) -> bool {
        match self {
            S::Equation(step) => step.complete(),
            S::AlternativeRule { alternatives, .. } => alternatives[0].complete(),
            _ => !self.is_partial() && self.substeps().iter().all(Step::complete),
        }
    }
}

type Solver = StepSolver<SystemProblem, SystemStep>;
type Out = Option<SystemStep>;

fn occurring(equations: &[Equation], variables: &[Basic]) -> Vec<Basic> {
    variables
        .iter()
        .filter(|v| equations.iter().any(|(l, r)| l.has(v) || r.has(v)))
        .cloned()
        .collect()
}

fn linear_coefficient(equation: &Equation, variable: &Basic) -> Option<Basic> {
    let difference: Basic = expand(&crate::core::add::sub(&equation.0, &equation.1).ok()?).ok()?;
    if crate::core::expr::polynomial_like(
        &difference,
        std::slice::from_ref(variable),
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return None;
    }
    let coefficient: Basic = crate::core::function::derivative(&difference, variable).ok()?;
    if !coefficient.free_symbols().is_empty() || coefficient == int(0) {
        return None;
    }
    Some(coefficient)
}

fn trivial(p: &SystemProblem, _: &Context, _: &mut Solver) -> Out {
    match satisfies(&p.equations, &[], &[], p.domain == Domain::Reals).ok()? {
        Some(true) => {
            if p.variables.is_empty() {
                Some(S::ConsistentRule {
                    equations: p.equations.clone(),
                    variables: p.variables.clone(),
                })
            } else {
                None
            }
        }
        Some(false) => Some(S::InconsistentRule {
            equations: p.equations.clone(),
            variables: p.variables.clone(),
        }),
        None => None,
    }
}

fn single(p: &SystemProblem, _: &Context, _: &mut Solver) -> Out {
    let (lhs, rhs) = &p.equations[0];
    let step: EquationStep = solve_steps(lhs, rhs, &p.variables[0], p.domain, false, Some(30));
    if step.contains_dont_know() {
        return None;
    }
    let found: Basic = step.solutions().ok()?;
    if !matches!(found.head(), Head::FiniteSet | Head::EmptySet) {
        return None;
    }
    Some(S::SingleEquationRule {
        equations: p.equations.clone(),
        variables: p.variables.clone(),
        step,
    })
}

fn substitution(p: &SystemProblem, s: &mut Solver, index: usize, variable: &Basic) -> Out {
    let (lhs, rhs) = &p.equations[index];
    if !expand(&crate::core::add::sub(lhs, rhs).ok()?).ok()?.has(variable) {
        return None;
    }
    let isolation: EquationStep = solve_steps(lhs, rhs, variable, p.domain, false, Some(30));
    if isolation.contains_dont_know() {
        return None;
    }
    let found: Basic = isolation.solutions().ok()?;
    if matches!(found.head(), Head::EmptySet) {
        return Some(S::InconsistentRule {
            equations: p.equations.clone(),
            variables: p.variables.clone(),
        });
    }
    if !matches!(found.head(), Head::FiniteSet) {
        return None;
    }
    let expressions: Vec<Basic> = found.args().to_vec();
    let remaining: Vec<Basic> = p.variables.iter().filter(|v| *v != variable).cloned().collect();
    let others: Vec<&Equation> = p
        .equations
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != index)
        .map(|(_, e)| e)
        .collect();
    let mut cases: Vec<SystemStep> = Vec::new();
    for expression in &expressions {
        let mut substituted: Vec<Equation> = Vec::new();
        for (left, right) in &others {
            substituted.push((
                expand(&subs(left, variable, expression).ok()?).ok()?,
                expand(&subs(right, variable, expression).ok()?).ok()?,
            ));
        }
        cases.push(s.solve(&SystemProblem {
            equations: substituted,
            variables: remaining.clone(),
            domain: p.domain,
        }));
    }
    let make = |cases: Vec<SystemStep>, rejected: Vec<Solution>| S::SubstitutionRule {
        equations: p.equations.clone(),
        variables: p.variables.clone(),
        index,
        variable: variable.clone(),
        isolation: isolation.clone(),
        expressions: expressions.clone(),
        remaining: remaining.clone(),
        cases,
        rejected,
    };
    if cases.iter().any(Step::contains_dont_know) {
        return Some(make(cases, Vec::new()));
    }
    let mut rejected: Vec<Solution> = Vec::new();
    for (expression, case) in expressions.iter().zip(cases.iter()) {
        for partial in case.solutions().ok()? {
            let solution: Solution = completed(&p.variables, &remaining, &partial, variable, expression).ok()?;
            if satisfies(&p.equations, &p.variables, &solution, p.domain == Domain::Reals).ok()? == Some(false)
                && !rejected.contains(&solution)
            {
                rejected.push(solution);
            }
        }
    }
    Some(make(cases, rejected))
}

fn elimination(p: &SystemProblem, s: &mut Solver, first: usize, second: usize, variable: &Basic) -> Out {
    let a: Basic = linear_coefficient(&p.equations[first], variable)?;
    let b: Basic = linear_coefficient(&p.equations[second], variable)?;
    let combine = |u: &Basic, v: &Basic| -> Option<Basic> {
        let left: Basic = crate::core::mul::mul(&[b.clone(), u.clone()]).ok()?;
        let right: Basic = crate::core::mul::mul(&[a.clone(), v.clone()]).ok()?;
        expand(&crate::core::add::sub(&left, &right).ok()?).ok()
    };
    let left: Basic = combine(&p.equations[first].0, &p.equations[second].0)?;
    let right: Basic = combine(&p.equations[first].1, &p.equations[second].1)?;
    let combined: Equation = (left, right);
    let equations: Vec<Equation> = p
        .equations
        .iter()
        .enumerate()
        .map(|(k, e)| if k == second { combined.clone() } else { e.clone() })
        .collect();
    let substep: SystemStep = s.solve(&SystemProblem {
        equations,
        variables: p.variables.clone(),
        domain: p.domain,
    });
    Some(S::EliminationRule {
        equations: p.equations.clone(),
        variables: p.variables.clone(),
        first,
        second,
        first_multiplier: b,
        second_multiplier: crate::core::mul::neg(&a).ok()?,
        variable: variable.clone(),
        combined,
        substep: Box::new(substep),
    })
}

fn candidates(p: &SystemProblem) -> Vec<Rule<SystemProblem, SystemStep>> {
    let found: Vec<Basic> = occurring(&p.equations, &p.variables);
    if found.is_empty() {
        return rules![trivial];
    }
    if p.variables.len() == 1 && p.equations.len() == 1 {
        return rules![single];
    }
    if found.len() < p.variables.len() {
        return Vec::new();
    }
    let mut choices: Vec<(i32, usize, usize)> = Vec::new();
    for (index, equation) in p.equations.iter().enumerate() {
        for (position, variable) in p.variables.iter().enumerate() {
            if !(equation.0.has(variable) || equation.1.has(variable)) {
                continue;
            }
            let rank: i32 = match linear_coefficient(equation, variable) {
                None => 2,
                Some(c) if c == int(1) || c == int(-1) => 0,
                Some(_) => 1,
            };
            choices.push((rank, index, position));
        }
    }
    choices.sort();
    let mut out: Vec<Rule<SystemProblem, SystemStep>> = Vec::new();
    for (_, index, position) in choices {
        let variable: Basic = p.variables[position].clone();
        out.push(Box::new(move |q: &SystemProblem, _: &Context, s: &mut Solver| {
            substitution(q, s, index, &variable)
        }));
    }
    for variable in &p.variables {
        let linear: Vec<usize> = p
            .equations
            .iter()
            .enumerate()
            .filter(|(_, e)| linear_coefficient(e, variable).is_some())
            .map(|(k, _)| k)
            .collect();
        if linear.len() >= 2 && p.equations.len() >= 2 {
            let (first, second) = (linear[0], linear[1]);
            let still: Vec<usize> = p
                .equations
                .iter()
                .enumerate()
                .filter(|(k, e)| *k != second && (e.0.has(variable) || e.1.has(variable)))
                .map(|(k, _)| k)
                .collect();
            if still.len() == 1 {
                let v: Basic = variable.clone();
                out.push(Box::new(move |q: &SystemProblem, _: &Context, s: &mut Solver| {
                    elimination(q, s, first, second, &v)
                }));
            }
        }
    }
    out
}

/// The derivation of the solutions of a system (`system_steps`).
pub fn system_steps(
    equations: &[Equation],
    variables: &[Basic],
    domain: Domain,
    branch: bool,
    max_depth: Option<usize>,
) -> PolyResult<SystemStep> {
    let problem = SystemProblem {
        equations: equations.to_vec(),
        variables: variables.to_vec(),
        domain,
    };
    for i in 0..variables.len() {
        if variables[..i].contains(&variables[i]) {
            return Err(value_error("the unknowns are not distinct".to_string()));
        }
    }
    let mut solver: Solver = StepSolver::new(
        candidates,
        |p| S::DontKnowRule {
            equations: p.equations.clone(),
            variables: p.variables.clone(),
        },
        |p| S::InProgressRule {
            equations: p.equations.clone(),
            variables: p.variables.clone(),
        },
        |p, alternatives| S::AlternativeRule {
            equations: p.equations.clone(),
            variables: p.variables.clone(),
            alternatives,
        },
        branch,
        max_depth,
    );
    Ok(solver.solve(&problem))
}

/// The solutions found by the rules, or `None` (`manual_system`).
pub fn manual_system(equations: &[Equation], variables: &[Basic], domain: Domain) -> PolyResult<Option<Vec<Solution>>> {
    let steps: SystemStep = system_steps(equations, variables, domain, false, Some(20))?;
    if steps.contains_dont_know() {
        return Ok(None);
    }
    Ok(Some(steps.solutions()?))
}
