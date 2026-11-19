//! Port of `sympy_extras/steps/inequalities.py`: inequalities in one real
//! unknown solved step by step: linear moves, division (the relation
//! reversed by a negative divisor), the sign chart of a rational function
//! with rational coefficients (its roots and poles by the equations of
//! [`super::equations`]), absolute values by cases.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::{as_independent, is};
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::Rat;
use crate::polys::cad::samplepoints::Real;
use crate::rules;

use super::core::{Context, Rule, Step, StepSolver};
use super::equations::{solve_steps, Domain, EquationStep};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn normal(e: &Basic) -> R {
    crate::simplify::normalform::normal_form(e)
}
fn number(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}

fn reversed(relation: &str) -> &'static str {
    match relation {
        "<" => ">",
        "<=" => ">=",
        ">" => "<",
        _ => "<=",
    }
}

/// `lhs relation rhs` in the variable (`InequalityProblem`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InequalityProblem {
    pub lhs: Basic,
    pub rhs: Basic,
    pub relation: String,
    pub variable: Basic,
}

fn problem(lhs: Basic, rhs: Basic, relation: &str, x: &Basic) -> InequalityProblem {
    InequalityProblem {
        lhs,
        rhs,
        relation: relation.to_string(),
        variable: x.clone(),
    }
}

type B = Box<InequalityStep>;

/// A step of the solution of an inequality, or the step of an equation
/// inside a sign chart.
#[derive(Debug, Clone, PartialEq)]
pub enum InequalityStep {
    SolvedRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
        bound: Basic,
    },
    TrueRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
    },
    FalseRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
    },
    AddToBothSidesRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
        substep: B,
        term: Basic,
    },
    DivideBothSidesRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
        substep: B,
        divisor: Basic,
        flipped: bool,
    },
    SignChartRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
        numerator_factors: Vec<(Basic, i64)>,
        denominator_factors: Vec<(Basic, i64)>,
        points: Vec<Basic>,
        poles: Vec<Basic>,
        signs: Vec<i64>,
        root_steps: Vec<EquationStep>,
    },
    AbsoluteValueRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
        inner: Basic,
        cases: Vec<InequalityStep>,
    },
    AlternativeRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
        alternatives: Vec<InequalityStep>,
    },
    DontKnowRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
    },
    InProgressRule {
        lhs: Basic,
        rhs: Basic,
        relation: String,
        variable: Basic,
    },
    /// A step of an equation, a substep of a sign chart.
    Equation(EquationStep),
}

use InequalityStep as I;

fn open_interval(a: &Basic, b: &Basic, left_open: bool, right_open: bool) -> R {
    crate::sets::sets::interval(a, b, left_open, right_open)
}

impl InequalityStep {
    fn parts(&self) -> Option<(&Basic, &Basic, &String)> {
        match self {
            I::SolvedRule { lhs, rhs, relation, .. }
            | I::TrueRule { lhs, rhs, relation, .. }
            | I::FalseRule { lhs, rhs, relation, .. }
            | I::AddToBothSidesRule { lhs, rhs, relation, .. }
            | I::DivideBothSidesRule { lhs, rhs, relation, .. }
            | I::SignChartRule { lhs, rhs, relation, .. }
            | I::AbsoluteValueRule { lhs, rhs, relation, .. }
            | I::AlternativeRule { lhs, rhs, relation, .. }
            | I::DontKnowRule { lhs, rhs, relation, .. }
            | I::InProgressRule { lhs, rhs, relation, .. } => Some((lhs, rhs, relation)),
            I::Equation(_) => None,
        }
    }

    /// The set of the solutions (`solutions`).
    pub fn solutions(&self) -> R {
        let (oo, moo) = (Basic::oo(), Basic::negative_oo());
        match self {
            I::SolvedRule { relation, bound, .. } => match relation.as_str() {
                "<" => open_interval(&moo, bound, true, true),
                "<=" => open_interval(&moo, bound, true, false),
                ">" => open_interval(bound, &oo, true, true),
                _ => open_interval(bound, &oo, false, true),
            },
            I::TrueRule { .. } => Ok(crate::sets::sets::reals()),
            I::FalseRule { .. } => Ok(crate::sets::sets::empty_set()),
            I::AddToBothSidesRule { substep, .. } | I::DivideBothSidesRule { substep, .. } => substep.solutions(),
            I::SignChartRule {
                relation,
                points,
                poles,
                signs,
                ..
            } => {
                let wanted: i64 = if relation == ">" || relation == ">=" { 1 } else { -1 };
                if points.is_empty() {
                    return Ok(if signs[0] == wanted {
                        crate::sets::sets::reals()
                    } else {
                        crate::sets::sets::empty_set()
                    });
                }
                let mut pieces: Vec<Basic> = Vec::new();
                let mut bounds: Vec<Basic> = vec![moo];
                bounds.extend(points.iter().cloned());
                bounds.push(oo);
                for (position, sign) in signs.iter().enumerate() {
                    if *sign == wanted {
                        pieces.push(open_interval(&bounds[position], &bounds[position + 1], true, true)?);
                    }
                }
                if relation == "<=" || relation == ">=" {
                    let zeros: Vec<Basic> = points.iter().filter(|p| !poles.contains(p)).cloned().collect();
                    if !zeros.is_empty() {
                        pieces.push(crate::sets::sets::finite_set(&zeros));
                    }
                }
                if pieces.is_empty() {
                    Ok(crate::sets::sets::empty_set())
                } else {
                    crate::sets::sets::union(&pieces)
                }
            }
            I::AbsoluteValueRule { cases, .. } => {
                let found: Vec<Basic> = cases.iter().map(|c| c.solutions()).collect::<PolyResult<_>>()?;
                let first: Basic = crate::sets::sets::intersection(&[found[0].clone(), found[1].clone()])?;
                let second: Basic = crate::sets::sets::intersection(&[found[2].clone(), found[3].clone()])?;
                crate::sets::sets::union(&[first, second])
            }
            I::AlternativeRule { alternatives, .. } => alternatives[0].solutions(),
            I::DontKnowRule { lhs, rhs, relation, .. } | I::InProgressRule { lhs, rhs, relation, .. } => Err(
                value_error(format!("the inequality {} {} {} is not solved", lhs, relation, rhs)),
            ),
            I::Equation(step) => step.solutions(),
        }
    }
}

impl Step for InequalityStep {
    fn substeps(&self) -> Vec<InequalityStep> {
        match self {
            I::AddToBothSidesRule { substep, .. } | I::DivideBothSidesRule { substep, .. } => vec![(**substep).clone()],
            I::SignChartRule { root_steps, .. } => root_steps.iter().cloned().map(I::Equation).collect(),
            I::AbsoluteValueRule { cases, .. } => cases.clone(),
            I::AlternativeRule { alternatives, .. } => alternatives.clone(),
            I::Equation(step) => step.substeps().into_iter().map(I::Equation).collect(),
            _ => Vec::new(),
        }
    }

    fn rule(&self) -> String {
        match self {
            I::SolvedRule { .. } => "SolvedRule".to_string(),
            I::TrueRule { .. } => "TrueRule".to_string(),
            I::FalseRule { .. } => "FalseRule".to_string(),
            I::AddToBothSidesRule { .. } => "AddToBothSidesRule".to_string(),
            I::DivideBothSidesRule { .. } => "DivideBothSidesRule".to_string(),
            I::SignChartRule { .. } => "SignChartRule".to_string(),
            I::AbsoluteValueRule { .. } => "AbsoluteValueRule".to_string(),
            I::AlternativeRule { .. } => "AlternativeRule".to_string(),
            I::DontKnowRule { .. } => "DontKnowRule".to_string(),
            I::InProgressRule { .. } => "InProgressRule".to_string(),
            I::Equation(step) => step.rule(),
        }
    }

    fn statement(&self) -> String {
        let Some((lhs, rhs, relation)) = self.parts() else {
            let I::Equation(step) = self else { unreachable!() };
            return step.statement();
        };
        let found: String = if self.contains_dont_know() {
            "?".to_string()
        } else {
            match self.solutions() {
                Ok(s) => s.to_string(),
                Err(e) => format!("<{:?}>", e),
            }
        };
        format!("{} {} {} -> {}", lhs, relation, rhs, found)
    }

    fn minor(&self) -> bool {
        match self {
            I::Equation(step) => step.minor(),
            _ => false,
        }
    }

    fn is_alternative(&self) -> bool {
        match self {
            I::Equation(step) => step.is_alternative(),
            I::AlternativeRule { .. } => true,
            _ => false,
        }
    }

    fn is_dont_know(&self) -> bool {
        match self {
            I::Equation(step) => step.is_dont_know(),
            I::DontKnowRule { .. } | I::InProgressRule { .. } => true,
            _ => false,
        }
    }

    fn is_partial(&self) -> bool {
        match self {
            I::Equation(step) => step.is_partial(),
            _ => false,
        }
    }

    fn contains_dont_know(&self) -> bool {
        match self {
            I::Equation(step) => step.contains_dont_know(),
            _ => self.is_dont_know() || self.substeps().iter().any(Step::contains_dont_know),
        }
    }

    fn complete(&self) -> bool {
        match self {
            I::Equation(step) => step.complete(),
            _ => !self.is_partial() && self.substeps().iter().all(Step::complete),
        }
    }
}

type Solver = StepSolver<InequalityProblem, InequalityStep>;
type Out = Option<InequalityStep>;

fn sign(e: &Basic) -> Option<i64> {
    if *e == int(0) {
        return Some(0);
    }
    if is(e, Predicate::positive) == Some(true) {
        return Some(1);
    }
    if is(e, Predicate::negative) == Some(true) {
        return Some(-1);
    }
    None
}

fn holds(s: i64, relation: &str) -> bool {
    match relation {
        "<" => s < 0,
        "<=" => s <= 0,
        ">" => s > 0,
        _ => s >= 0,
    }
}

fn trivial(p: &InequalityProblem, _: &Context, _: &mut Solver) -> Out {
    let s: i64 = sign(&normal(&crate::core::add::sub(&p.lhs, &p.rhs).ok()?).ok()?)?;
    let (lhs, rhs, relation, variable) = (p.lhs.clone(), p.rhs.clone(), p.relation.clone(), p.variable.clone());
    Some(if holds(s, &p.relation) {
        I::TrueRule {
            lhs,
            rhs,
            relation,
            variable,
        }
    } else {
        I::FalseRule {
            lhs,
            rhs,
            relation,
            variable,
        }
    })
}

fn solved(p: &InequalityProblem, _: &Context, _: &mut Solver) -> Out {
    Some(I::SolvedRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        relation: p.relation.clone(),
        variable: p.variable.clone(),
        bound: p.rhs.clone(),
    })
}

fn split(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let terms: Vec<Basic> = if e.is_add() { e.args().to_vec() } else { vec![e.clone()] };
    let with: Vec<Basic> = terms.iter().filter(|t| t.has(x)).cloned().collect();
    let without: Vec<Basic> = terms.iter().filter(|t| !t.has(x)).cloned().collect();
    Ok((crate::core::add::add(&with)?, crate::core::add::add(&without)?))
}

fn added(p: &InequalityProblem, term: &Basic) -> PolyResult<InequalityProblem> {
    let lhs: Basic = expand(&crate::core::add::add(&[p.lhs.clone(), term.clone()])?)?;
    let rhs: Basic = expand(&crate::core::add::add(&[p.rhs.clone(), term.clone()])?)?;
    Ok(problem(lhs, rhs, &p.relation, &p.variable))
}

fn add_step(p: &InequalityProblem, substep: InequalityStep, term: Basic) -> Out {
    Some(I::AddToBothSidesRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        relation: p.relation.clone(),
        variable: p.variable.clone(),
        substep: Box::new(substep),
        term,
    })
}

fn move_unknowns(p: &InequalityProblem, _: &Context, s: &mut Solver) -> Out {
    let (moving, _) = split(&p.rhs, &p.variable).ok()?;
    if moving == int(0) {
        return None;
    }
    let term: Basic = crate::core::mul::neg(&moving).ok()?;
    let substep = s.solve(&added(p, &term).ok()?);
    add_step(p, substep, term)
}

fn move_constants(p: &InequalityProblem, _: &Context, s: &mut Solver) -> Out {
    let (_, moving) = split(&p.lhs, &p.variable).ok()?;
    if moving == int(0) {
        return None;
    }
    let term: Basic = crate::core::mul::neg(&moving).ok()?;
    let substep = s.solve(&added(p, &term).ok()?);
    add_step(p, substep, term)
}

fn divide(p: &InequalityProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() {
        return None;
    }
    let (coefficient, rest) = as_independent(&p.lhs, x, false).ok()?;
    let sg: Option<i64> = sign(&coefficient);
    if rest != *x || coefficient == int(1) || sg.is_none() || sg == Some(0) || p.rhs.has(x) {
        return None;
    }
    let sg: i64 = sg?;
    let relation: &str = if sg > 0 {
        p.relation.as_str()
    } else {
        reversed(&p.relation)
    };
    let substep = s.solve(&problem(
        x.clone(),
        normal(&crate::core::mul::div(&p.rhs, &coefficient).ok()?).ok()?,
        relation,
        x,
    ));
    Some(I::DivideBothSidesRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        relation: p.relation.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        divisor: coefficient,
        flipped: sg < 0,
    })
}

fn expand_sides(p: &InequalityProblem, _: &Context, s: &mut Solver) -> Out {
    let (lhs, rhs) = (expand(&p.lhs).ok()?, expand(&p.rhs).ok()?);
    if lhs == p.lhs && rhs == p.rhs {
        return None;
    }
    let substep = s.solve(&problem(lhs, rhs, &p.relation, &p.variable));
    add_step(p, substep, int(0))
}

fn standard_form(p: &InequalityProblem, _: &Context, s: &mut Solver) -> Out {
    if p.rhs == int(0) {
        return None;
    }
    let substep = s.solve(&problem(
        crate::core::add::sub(&p.lhs, &p.rhs).ok()?,
        int(0),
        &p.relation,
        &p.variable,
    ));
    add_step(p, substep, crate::core::mul::neg(&p.rhs).ok()?)
}

fn rational_coefficients(e: &Basic, x: &Basic) -> Option<Vec<Rat>> {
    let expanded: Basic = expand(e).ok()?;
    if expanded == int(0)
        || crate::core::expr::polynomial_like(
            &expanded,
            std::slice::from_ref(x),
            crate::core::expr::PolynomialKind::Polynomial,
        ) != Some(true)
    {
        return None;
    }
    crate::matrices::eigen::coefficients(&expanded, x)
        .ok()?
        .iter()
        .map(|c| c.as_rational().cloned())
        .collect()
}

fn factor_roots(cs: &[Rat], x: &Basic) -> PolyResult<Vec<(Real, Basic)>> {
    let algebraic: Vec<Real> = crate::polys::cad::samplepoints::real_roots_of(cs)?;
    let shown: Vec<Basic> = if cs.len() == 2 {
        vec![number(&(-cs[1].clone() / cs[0].clone()))]
    } else if cs.len() == 3 && algebraic.len() == 2 {
        let (a, b, c) = (number(&cs[0]), number(&cs[1]), number(&cs[2]));
        let m2 = |p: &Basic, q: &Basic| crate::core::mul::mul(&[p.clone(), q.clone()]);
        let root: Basic = crate::matrices::eigen::sqrt(&crate::core::add::sub(
            &crate::core::power::pow(&b, &int(2))?,
            &m2(&m2(&int(4), &a)?, &c)?,
        )?)?;
        let two_a: Basic = m2(&int(2), &a)?;
        let minus_b: Basic = crate::core::mul::neg(&b)?;
        let pair: Vec<Basic> = vec![
            crate::core::mul::div(&crate::core::add::sub(&minus_b, &root)?, &two_a)?,
            crate::core::mul::div(&crate::core::add::add(&[minus_b.clone(), root.clone()])?, &two_a)?,
        ];
        if cs[0] > Rat::from(crate::numbers::Int::from(0)) {
            pair
        } else {
            pair.into_iter().rev().collect()
        }
    } else {
        algebraic
            .iter()
            .map(|r| crate::polys::cad::indexedroot::real_root(r, &x.to_string()))
            .collect::<PolyResult<_>>()?
    };
    Ok(algebraic.into_iter().zip(shown).collect())
}

fn sign_chart(p: &InequalityProblem, _: &Context, _: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.rhs != int(0) {
        return None;
    }
    let together: Basic = crate::polys::rationaltools::together(&p.lhs, false, true).ok()?;
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&together).ok()?;
    let (numerator, denominator) = (expand(&numerator).ok()?, expand(&denominator).ok()?);
    rational_coefficients(&numerator, x)?;
    rational_coefficients(&denominator, x)?;
    let listed = |e: &Basic| match crate::polys::polytools::factor_list(e, std::slice::from_ref(x)) {
        Ok(found) => Some(found),
        Err(PolyError::NotImplemented(m)) if m.starts_with("not ported") => None,
        Err(_) => None,
    };
    let (top_constant, top) = listed(&numerator)?;
    let (bottom_constant, bottom) = listed(&denominator)?;
    let (lead, other) = (sign(&top_constant)?, sign(&bottom_constant)?);
    if lead == 0 || other == 0 {
        return None;
    }
    let mut sg: i64 = lead * other;
    let mut exponents: Vec<(Basic, i64)> = Vec::new();
    let mut in_denominator: Vec<Basic> = Vec::new();
    let mut numerator_factors: Vec<(Basic, i64)> = Vec::new();
    let mut denominator_factors: Vec<(Basic, i64)> = Vec::new();
    for (group, below) in [(&top, false), (&bottom, true)] {
        for (factor, multiplicity) in group {
            if !multiplicity.is_integer_value() || !factor.has(x) {
                continue;
            }
            let k: i64 = i64::try_from(&crate::numbers::numer(multiplicity.as_rational()?)).ok()?;
            if below {
                denominator_factors.push((factor.clone(), k));
                if !in_denominator.contains(factor) {
                    in_denominator.push(factor.clone());
                }
            } else {
                numerator_factors.push((factor.clone(), k));
            }
            match exponents.iter_mut().find(|(f, _)| f == factor) {
                Some(entry) => entry.1 += k,
                None => exponents.push((factor.clone(), k)),
            }
        }
    }
    let mut located: Vec<(Real, Basic, i64, bool)> = Vec::new();
    let mut root_steps: Vec<EquationStep> = Vec::new();
    for (factor, exponent) in &exponents {
        let cs: Vec<Rat> = rational_coefficients(factor, x)?;
        if cs[0] < Rat::from(crate::numbers::Int::from(0)) && exponent % 2 != 0 {
            sg = -sg;
        }
        let roots = factor_roots(&cs, x).ok()?;
        if !roots.is_empty() {
            root_steps.push(solve_steps(factor, &int(0), x, Domain::Reals, false, Some(30)));
        }
        for (n, shown) in roots {
            located.push((n, shown, *exponent, in_denominator.contains(factor)));
        }
    }
    let mut failed: bool = false;
    located.sort_by(|a, b| match crate::polys::cad::samplepoints::compare_real(&a.0, &b.0) {
        Ok(c) => c.cmp(&0),
        Err(_) => {
            failed = true;
            std::cmp::Ordering::Equal
        }
    });
    if failed {
        return None;
    }
    let mut signs: Vec<i64> = vec![sg];
    for (_, _, exponent, _) in located.iter().rev() {
        let last: i64 = signs[signs.len() - 1];
        signs.push(if exponent % 2 != 0 { -last } else { last });
    }
    signs.reverse();
    let points: Vec<Basic> = located.iter().map(|l| l.1.clone()).collect();
    let poles: Vec<Basic> = located.iter().filter(|l| l.3).map(|l| l.1.clone()).collect();
    Some(I::SignChartRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        relation: p.relation.clone(),
        variable: x.clone(),
        numerator_factors,
        denominator_factors,
        points,
        poles,
        signs,
        root_steps,
    })
}

fn abs_nodes(e: &Basic, x: &Basic) -> Vec<Basic> {
    e.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Function(Function::Abs))))
        .into_iter()
        .filter(|n| n.has(x))
        .collect()
}

fn absolute_value(p: &InequalityProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let mut nodes: Vec<Basic> = abs_nodes(&p.lhs, x);
    nodes.extend(abs_nodes(&p.rhs, x));
    nodes.sort_by_key(|n| n.to_string());
    let node: Basic = nodes.first()?.clone();
    let inner: Basic = node.args()[0].clone();
    let mut cases: Vec<InequalityStep> = Vec::new();
    for (relation, replacement) in [(">=", inner.clone()), ("<", crate::core::mul::neg(&inner).ok()?)] {
        cases.push(s.solve(&problem(inner.clone(), int(0), relation, x)));
        let lhs: Basic = crate::core::subs::subs(&p.lhs, &node, &replacement).ok()?;
        let rhs: Basic = crate::core::subs::subs(&p.rhs, &node, &replacement).ok()?;
        cases.push(s.solve(&problem(lhs, rhs, &p.relation, x)));
    }
    Some(I::AbsoluteValueRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        relation: p.relation.clone(),
        variable: x.clone(),
        inner,
        cases,
    })
}

fn candidates(p: &InequalityProblem) -> Vec<Rule<InequalityProblem, InequalityStep>> {
    let (lhs, rhs, x) = (&p.lhs, &p.rhs, &p.variable);
    if !lhs.has(x) && !rhs.has(x) {
        return rules![trivial];
    }
    if !abs_nodes(lhs, x).is_empty() || !abs_nodes(rhs, x).is_empty() {
        return rules![absolute_value];
    }
    if lhs == x && !rhs.has(x) {
        return rules![solved];
    }
    let Ok(difference) = crate::core::add::sub(lhs, rhs).and_then(|d| expand(&d)) else {
        return Vec::new();
    };
    if crate::core::expr::polynomial_like(
        &difference,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
    {
        let cs = rational_coefficients(&difference, x);
        if !difference.has(x) {
            return rules![trivial];
        }
        let degree: usize = if difference != int(0) {
            crate::matrices::eigen::coefficients(&difference, x)
                .map(|c| c.len() - 1)
                .unwrap_or(0)
        } else {
            0
        };
        if degree == 1 {
            return rules![expand_sides, move_unknowns, move_constants, divide];
        }
        if cs.is_none() {
            return Vec::new();
        }
    }
    rules![standard_form, sign_chart]
}

fn parsed(inequality: &Basic, x: &Basic) -> PolyResult<InequalityProblem> {
    let relation: &str = match inequality.head() {
        Head::Relational(RelOp::Lt) => "<",
        Head::Relational(RelOp::Le) => "<=",
        Head::Relational(RelOp::Gt) => ">",
        Head::Relational(RelOp::Ge) => ">=",
        Head::Relational(_) => return Err(PolyError::Type(format!("{} is not an inequality", inequality))),
        _ => {
            return Err(PolyError::Type(format!(
                "{} is not an inequality (a comparison which was evaluated?)",
                inequality
            )))
        }
    };
    Ok(problem(
        inequality.args()[0].clone(),
        inequality.args()[1].clone(),
        relation,
        x,
    ))
}

/// The derivation of the solutions of an inequality (`inequality_steps`).
pub fn inequality_steps(
    inequality: &Basic,
    x: &Basic,
    branch: bool,
    max_depth: Option<usize>,
) -> PolyResult<InequalityStep> {
    let mut solver: Solver = StepSolver::new(
        candidates,
        |p| I::DontKnowRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            relation: p.relation.clone(),
            variable: p.variable.clone(),
        },
        |p| I::InProgressRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            relation: p.relation.clone(),
            variable: p.variable.clone(),
        },
        |p, alternatives| I::AlternativeRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            relation: p.relation.clone(),
            variable: p.variable.clone(),
            alternatives,
        },
        branch,
        max_depth,
    );
    Ok(solver.solve(&parsed(inequality, x)?))
}

/// The solutions found by the rules, or `None` (`manual_inequality`).
pub fn manual_inequality(inequality: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let steps: InequalityStep = inequality_steps(inequality, x, false, Some(30))?;
    if steps.contains_dont_know() {
        return Ok(None);
    }
    Ok(Some(steps.solutions()?))
}
