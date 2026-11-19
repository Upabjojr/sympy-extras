//! Port of `sympy_extras/steps/equations.py`: equations in one unknown
//! solved step by step over the complex or the real numbers, as a tree of
//! rules driven by the solver of [`super::core`].

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_independent, is};
use crate::error::{value_error, PolyResult};

use super::core::{Context, Step, StepSolver};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rat(p: i64, q: i64) -> Basic {
    Basic::from_rational(
        crate::numbers::Rat::from(crate::numbers::Int::from(p))
            / crate::numbers::Rat::from(crate::numbers::Int::from(q)),
    )
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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sqrt(a: &Basic) -> R {
    pw(a, &rat(1, 2))
}
fn apply(f: Function, a: &Basic) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), vec![a.clone()]), vec![a.clone()])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn normal(e: &Basic) -> R {
    crate::simplify::normalform::normal_form(e)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn imaginary() -> Basic {
    Basic::atom(Head::ImaginaryUnit)
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn finite(v: &[Basic]) -> Basic {
    crate::sets::sets::finite_set(v)
}
fn empty() -> Basic {
    crate::sets::sets::empty_set()
}
fn union(v: &[Basic]) -> R {
    crate::sets::sets::union(v)
}

/// The domain of the unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    Complexes,
    Reals,
}

impl Domain {
    pub fn set(self) -> Basic {
        match self {
            Domain::Complexes => Basic::atom(Head::Complexes),
            Domain::Reals => crate::sets::sets::reals(),
        }
    }
}

/// `lhs = rhs` in the variable over the domain (`EquationProblem`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EquationProblem {
    pub lhs: Basic,
    pub rhs: Basic,
    pub variable: Basic,
    pub domain: Domain,
}

fn problem(lhs: Basic, rhs: Basic, x: &Basic, domain: Domain) -> EquationProblem {
    EquationProblem {
        lhs,
        rhs,
        variable: x.clone(),
        domain,
    }
}

type B = Box<EquationStep>;

/// A step of the solution of an equation (the rule dataclasses).
#[derive(Debug, Clone, PartialEq)]
pub enum EquationStep {
    SolvedRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        value: Basic,
    },
    IdentityRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        domain: Domain,
    },
    ContradictionRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
    },
    AddToBothSidesRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        term: Basic,
    },
    DivideBothSidesRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        divisor: Basic,
    },
    RewriteRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
    },
    SwapSidesRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
    },
    ZeroProductRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        constant: Basic,
        factors: Vec<Basic>,
        cases: Vec<EquationStep>,
    },
    QuadraticFormulaRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        a: Basic,
        b: Basic,
        c: Basic,
        discriminant: Basic,
        domain: Domain,
    },
    CompleteSquareRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        shift: Basic,
        value: Basic,
    },
    RootRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        base: Basic,
        exponent: i64,
        value: Basic,
        cases: Vec<EquationStep>,
    },
    ClearDenominatorsRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        rejected: Vec<Basic>,
        unverified: Vec<Basic>,
        denominator: Basic,
    },
    SquareBothSidesRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        rejected: Vec<Basic>,
        unverified: Vec<Basic>,
    },
    AbsoluteValueRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        inner: Basic,
        cases: Vec<EquationStep>,
        rejected: Vec<Basic>,
        unverified: Vec<Basic>,
        partial: bool,
    },
    TakeLogarithmRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        base: Basic,
        parameter: Option<Basic>,
    },
    EqualExponentsRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        base: Basic,
        parameter: Option<Basic>,
    },
    ExponentiateRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
    },
    EqualArgumentsRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        rejected: Vec<Basic>,
        unverified: Vec<Basic>,
    },
    CombineLogarithmsRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
        rejected: Vec<Basic>,
        unverified: Vec<Basic>,
    },
    SubstitutionRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        placeholder: Basic,
        expression: Basic,
        substep: B,
        cases: Vec<EquationStep>,
    },
    PeriodicRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        function: String,
        value: Basic,
        parameter: Basic,
        cases: Vec<EquationStep>,
    },
    TangentRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        substep: B,
    },
    CardanoRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        shift: Basic,
        p: Basic,
        q: Basic,
        domain: Domain,
    },
    RealRootsRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        roots: Vec<Basic>,
        missing: usize,
    },
    AlternativeRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
        alternatives: Vec<EquationStep>,
    },
    DontKnowRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
    },
    InProgressRule {
        lhs: Basic,
        rhs: Basic,
        variable: Basic,
    },
}

use EquationStep as E;

fn kept(found: Basic, rejected: &[Basic]) -> Basic {
    if !matches!(found.head(), Head::FiniteSet) {
        return found;
    }
    let remaining: Vec<Basic> = found.args().iter().filter(|e| !rejected.contains(e)).cloned().collect();
    if remaining.is_empty() {
        empty()
    } else {
        finite(&remaining)
    }
}

fn families(found: Basic, parameter: &Basic) -> R {
    if !matches!(found.head(), Head::FiniteSet) {
        if matches!(found.head(), Head::EmptySet) {
            return Ok(found);
        }
        return Err(value_error(format!(
            "the solutions {} of a periodic case are not expressions in {}",
            found, parameter
        )));
    }
    let integers: Basic = Basic::atom(Head::Integers);
    let variables: Basic = Basic::new(Head::Tuple, vec![parameter.clone()]);
    let parts: Vec<Basic> = found
        .args()
        .iter()
        .map(|e| crate::sets::fancysets::image_set(&variables, e, std::slice::from_ref(&integers)))
        .collect::<PolyResult<_>>()?;
    if parts.is_empty() {
        Ok(empty())
    } else {
        union(&parts)
    }
}

fn real_cube_root(value: &Basic) -> R {
    if is(value, Predicate::negative) == Some(true) {
        return neg(&pw(&neg(value)?, &rat(1, 3))?);
    }
    pw(value, &rat(1, 3))
}

impl EquationStep {
    fn sides(&self) -> (&Basic, &Basic) {
        match self {
            E::SolvedRule { lhs, rhs, .. }
            | E::IdentityRule { lhs, rhs, .. }
            | E::ContradictionRule { lhs, rhs, .. }
            | E::AddToBothSidesRule { lhs, rhs, .. }
            | E::DivideBothSidesRule { lhs, rhs, .. }
            | E::RewriteRule { lhs, rhs, .. }
            | E::SwapSidesRule { lhs, rhs, .. }
            | E::ZeroProductRule { lhs, rhs, .. }
            | E::QuadraticFormulaRule { lhs, rhs, .. }
            | E::CompleteSquareRule { lhs, rhs, .. }
            | E::RootRule { lhs, rhs, .. }
            | E::ClearDenominatorsRule { lhs, rhs, .. }
            | E::SquareBothSidesRule { lhs, rhs, .. }
            | E::AbsoluteValueRule { lhs, rhs, .. }
            | E::TakeLogarithmRule { lhs, rhs, .. }
            | E::EqualExponentsRule { lhs, rhs, .. }
            | E::ExponentiateRule { lhs, rhs, .. }
            | E::EqualArgumentsRule { lhs, rhs, .. }
            | E::CombineLogarithmsRule { lhs, rhs, .. }
            | E::SubstitutionRule { lhs, rhs, .. }
            | E::PeriodicRule { lhs, rhs, .. }
            | E::TangentRule { lhs, rhs, .. }
            | E::CardanoRule { lhs, rhs, .. }
            | E::RealRootsRule { lhs, rhs, .. }
            | E::AlternativeRule { lhs, rhs, .. }
            | E::DontKnowRule { lhs, rhs, .. }
            | E::InProgressRule { lhs, rhs, .. } => (lhs, rhs),
        }
    }

    /// The set of the solutions (`solutions`).
    pub fn solutions(&self) -> R {
        match self {
            E::SolvedRule { value, .. } => Ok(finite(std::slice::from_ref(value))),
            E::IdentityRule { domain, .. } => Ok(domain.set()),
            E::ContradictionRule { .. } => Ok(empty()),
            E::AddToBothSidesRule { substep, .. }
            | E::DivideBothSidesRule { substep, .. }
            | E::RewriteRule { substep, .. }
            | E::SwapSidesRule { substep, .. }
            | E::CompleteSquareRule { substep, .. }
            | E::ExponentiateRule { substep, .. }
            | E::TangentRule { substep, .. } => substep.solutions(),
            E::ZeroProductRule { cases, .. } => {
                union(&cases.iter().map(|c| c.solutions()).collect::<PolyResult<Vec<_>>>()?)
            }
            E::RootRule { cases, .. } | E::SubstitutionRule { cases, .. } => {
                if cases.is_empty() {
                    Ok(empty())
                } else {
                    union(&cases.iter().map(|c| c.solutions()).collect::<PolyResult<Vec<_>>>()?)
                }
            }
            E::QuadraticFormulaRule {
                a,
                b,
                discriminant,
                domain,
                ..
            } => {
                if *domain == Domain::Reals && is(discriminant, Predicate::negative) == Some(true) {
                    return Ok(empty());
                }
                let two_a: Basic = m2(&int(2), a)?;
                if *discriminant == int(0) {
                    return Ok(finite(&[d2(&neg(b)?, &two_a)?]));
                }
                let root: Basic = sqrt(discriminant)?;
                Ok(finite(&[
                    d2(&s2(&neg(b)?, &root)?, &two_a)?,
                    d2(&a2(&neg(b)?, &root)?, &two_a)?,
                ]))
            }
            E::ClearDenominatorsRule { substep, rejected, .. }
            | E::SquareBothSidesRule { substep, rejected, .. }
            | E::EqualArgumentsRule { substep, rejected, .. }
            | E::CombineLogarithmsRule { substep, rejected, .. } => Ok(kept(substep.solutions()?, rejected)),
            E::AbsoluteValueRule { cases, rejected, .. } => {
                let found: Basic = union(&cases.iter().map(|c| c.solutions()).collect::<PolyResult<Vec<_>>>()?)?;
                Ok(kept(found, rejected))
            }
            E::TakeLogarithmRule { substep, parameter, .. } | E::EqualExponentsRule { substep, parameter, .. } => {
                let found: Basic = substep.solutions()?;
                match parameter {
                    None => Ok(found),
                    Some(p) => families(found, p),
                }
            }
            E::PeriodicRule { parameter, cases, .. } => {
                let parts: Vec<Basic> = cases
                    .iter()
                    .map(|c| families(c.solutions()?, parameter))
                    .collect::<PolyResult<_>>()?;
                if parts.is_empty() {
                    Ok(empty())
                } else {
                    union(&parts)
                }
            }
            E::CardanoRule {
                shift, p, q, domain, ..
            } => {
                let inner: Basic = sqrt(&a2(&d2(&pw(q, &int(2))?, &int(4))?, &d2(&pw(p, &int(3))?, &int(27))?)?)?;
                let half_q: Basic = d2(&neg(q)?, &int(2))?;
                let (u, v) = (
                    real_cube_root(&a2(&half_q, &inner)?)?,
                    real_cube_root(&s2(&half_q, &inner)?)?,
                );
                let sum_uv: Basic = a2(&u, &v)?;
                let first: Basic = s2(&sum_uv, shift)?;
                if *domain == Domain::Reals {
                    return Ok(finite(&[first]));
                }
                let turn: Basic = d2(&m2(&m2(&sqrt(&int(3))?, &imaginary())?, &s2(&u, &v)?)?, &int(2))?;
                let half: Basic = d2(&neg(&sum_uv)?, &int(2))?;
                Ok(finite(&[
                    first,
                    s2(&a2(&half, &turn)?, shift)?,
                    s2(&s2(&half, &turn)?, shift)?,
                ]))
            }
            E::RealRootsRule { roots, .. } => Ok(if roots.is_empty() { empty() } else { finite(roots) }),
            E::AlternativeRule { alternatives, .. } => alternatives[0].solutions(),
            E::DontKnowRule { lhs, rhs, .. } | E::InProgressRule { lhs, rhs, .. } => {
                Err(value_error(format!("the equation {} = {} is not solved", lhs, rhs)))
            }
        }
    }
}

impl Step for EquationStep {
    fn substeps(&self) -> Vec<EquationStep> {
        match self {
            E::AddToBothSidesRule { substep, .. }
            | E::DivideBothSidesRule { substep, .. }
            | E::RewriteRule { substep, .. }
            | E::SwapSidesRule { substep, .. }
            | E::CompleteSquareRule { substep, .. }
            | E::ClearDenominatorsRule { substep, .. }
            | E::SquareBothSidesRule { substep, .. }
            | E::TakeLogarithmRule { substep, .. }
            | E::EqualExponentsRule { substep, .. }
            | E::ExponentiateRule { substep, .. }
            | E::EqualArgumentsRule { substep, .. }
            | E::CombineLogarithmsRule { substep, .. }
            | E::TangentRule { substep, .. } => vec![(**substep).clone()],
            E::ZeroProductRule { cases, .. }
            | E::RootRule { cases, .. }
            | E::AbsoluteValueRule { cases, .. }
            | E::PeriodicRule { cases, .. } => cases.clone(),
            E::SubstitutionRule { substep, cases, .. } => {
                let mut v: Vec<EquationStep> = vec![(**substep).clone()];
                v.extend(cases.iter().cloned());
                v
            }
            E::AlternativeRule { alternatives, .. } => alternatives.clone(),
            _ => Vec::new(),
        }
    }

    fn rule(&self) -> String {
        let name: &str = match self {
            E::SolvedRule { .. } => "SolvedRule",
            E::IdentityRule { .. } => "IdentityRule",
            E::ContradictionRule { .. } => "ContradictionRule",
            E::AddToBothSidesRule { .. } => "AddToBothSidesRule",
            E::DivideBothSidesRule { .. } => "DivideBothSidesRule",
            E::RewriteRule { .. } => "RewriteRule",
            E::SwapSidesRule { .. } => "SwapSidesRule",
            E::ZeroProductRule { .. } => "ZeroProductRule",
            E::QuadraticFormulaRule { .. } => "QuadraticFormulaRule",
            E::CompleteSquareRule { .. } => "CompleteSquareRule",
            E::RootRule { .. } => "RootRule",
            E::ClearDenominatorsRule { .. } => "ClearDenominatorsRule",
            E::SquareBothSidesRule { .. } => "SquareBothSidesRule",
            E::AbsoluteValueRule { .. } => "AbsoluteValueRule",
            E::TakeLogarithmRule { .. } => "TakeLogarithmRule",
            E::EqualExponentsRule { .. } => "EqualExponentsRule",
            E::ExponentiateRule { .. } => "ExponentiateRule",
            E::EqualArgumentsRule { .. } => "EqualArgumentsRule",
            E::CombineLogarithmsRule { .. } => "CombineLogarithmsRule",
            E::SubstitutionRule { .. } => "SubstitutionRule",
            E::PeriodicRule { .. } => "PeriodicRule",
            E::TangentRule { .. } => "TangentRule",
            E::CardanoRule { .. } => "CardanoRule",
            E::RealRootsRule { .. } => "RealRootsRule",
            E::AlternativeRule { .. } => "AlternativeRule",
            E::DontKnowRule { .. } => "DontKnowRule",
            E::InProgressRule { .. } => "InProgressRule",
        };
        name.to_string()
    }

    fn statement(&self) -> String {
        let (lhs, rhs) = self.sides();
        let found: String = if self.contains_dont_know() {
            "?".to_string()
        } else {
            match self.solutions() {
                Ok(s) => s.to_string(),
                Err(e) => format!("<{:?}>", e),
            }
        };
        format!(
            "{} = {} -> {}{}",
            lhs,
            rhs,
            found,
            if self.complete() { "" } else { " (not all)" }
        )
    }

    fn minor(&self) -> bool {
        matches!(self, E::RewriteRule { .. } | E::SwapSidesRule { .. })
    }

    fn is_alternative(&self) -> bool {
        matches!(self, E::AlternativeRule { .. })
    }

    fn is_dont_know(&self) -> bool {
        matches!(self, E::DontKnowRule { .. } | E::InProgressRule { .. })
    }

    fn is_partial(&self) -> bool {
        match self {
            E::AbsoluteValueRule { partial, .. } => *partial,
            E::RealRootsRule { missing, .. } => *missing > 0,
            _ => false,
        }
    }

    fn complete(&self) -> bool {
        match self {
            E::AlternativeRule { alternatives, .. } => alternatives[0].complete(),
            E::RealRootsRule { missing, .. } => *missing == 0,
            _ => !self.is_partial() && self.substeps().iter().all(Step::complete),
        }
    }
}

type Solver = StepSolver<EquationProblem, EquationStep>;
type Out = Option<EquationStep>;

fn split(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let terms: Vec<Basic> = terms_of(e);
    let with: Vec<Basic> = terms.iter().filter(|t| t.has(x)).cloned().collect();
    let without: Vec<Basic> = terms.iter().filter(|t| !t.has(x)).cloned().collect();
    Ok((crate::core::add::add(&with)?, crate::core::add::add(&without)?))
}

fn polynomial(e: &Basic, x: &Basic) -> Option<Vec<Basic>> {
    let expanded: Basic = expand(e).ok()?;
    if expanded == int(0) {
        return Some(vec![int(0)]);
    }
    if crate::core::expr::polynomial_like(
        &expanded,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return None;
    }
    crate::matrices::eigen::coefficients(&expanded, x).ok()
}

fn denominator(e: &Basic, x: &Basic) -> R {
    let (_, d) = crate::core::expr::as_numer_denom(e)?;
    if d.has(x) {
        expand(&d)
    } else {
        Ok(int(1))
    }
}

fn radicals(e: &Basic, x: &Basic) -> bool {
    let half: Basic = rat(1, 2);
    e.atoms(Some(&|n: &Basic| n.is_pow()))
        .iter()
        .any(|n| n.args()[1] == half && n.args()[0].has(x))
}

fn absolute_values(e: &Basic, x: &Basic) -> bool {
    e.atoms(Some(&|n: &Basic| func(n) == Some(Function::Abs)))
        .iter()
        .any(|n| n.has(x))
}

fn satisfies(p: &EquationProblem, value: &Basic) -> Option<bool> {
    let x: &Basic = &p.variable;
    let (left, right) = (subs(&p.lhs, x, value).ok()?, subs(&p.rhs, x, value).ok()?);
    let bad = |e: &Basic| {
        is(e, Predicate::finite) == Some(false)
            || e.preorder()
                .iter()
                .any(|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity))
    };
    if bad(&left) || bad(&right) {
        return Some(false);
    }
    if p.domain == Domain::Reals
        && (is(&left, Predicate::real) == Some(false) || is(&right, Predicate::real) == Some(false))
    {
        return Some(false);
    }
    let difference: Basic = match normal(&s2(&left, &right).ok()?) {
        Ok(d) => d,
        Err(crate::error::PolyError::ZeroDivision(_)) => return Some(false),
        Err(_) => return None,
    };
    if difference == int(0) || is(&difference, Predicate::zero) == Some(true) {
        return Some(true);
    }
    if is(&difference, Predicate::zero) == Some(false)
        || (crate::core::assumptions::is_number(&difference) && is(&difference, Predicate::real) == Some(false))
    {
        return Some(false);
    }
    if is(&difference, Predicate::positive) == Some(true) || is(&difference, Predicate::negative) == Some(true) {
        return Some(false);
    }
    if !difference.free_symbols().is_empty() {
        return None;
    }
    crate::simplify::structure::is_zero(&difference, &[]).unwrap_or(None)
}

fn trivial(p: &EquationProblem, _: &Context, _: &mut Solver) -> Out {
    let difference: Basic = normal(&s2(&p.lhs, &p.rhs).ok()?).ok()?;
    if difference == int(0) {
        return Some(E::IdentityRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            variable: p.variable.clone(),
            domain: p.domain,
        });
    }
    if !difference.has(&p.variable) {
        return Some(E::ContradictionRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            variable: p.variable.clone(),
        });
    }
    None
}

fn contradiction(p: &EquationProblem) -> Out {
    Some(E::ContradictionRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: p.variable.clone(),
    })
}

fn solved(p: &EquationProblem, _: &Context, _: &mut Solver) -> Out {
    if p.domain == Domain::Reals && is(&p.rhs, Predicate::real) == Some(false) {
        return contradiction(p);
    }
    Some(E::SolvedRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: p.variable.clone(),
        value: p.rhs.clone(),
    })
}

fn rewrite(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let (lhs, rhs) = (expand(&p.lhs).ok()?, expand(&p.rhs).ok()?);
    if lhs == p.lhs && rhs == p.rhs {
        return None;
    }
    let substep = s.solve(&problem(lhs, rhs, &p.variable, p.domain));
    Some(E::RewriteRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: p.variable.clone(),
        substep: Box::new(substep),
    })
}

fn swap(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let substep = s.solve(&problem(p.rhs.clone(), p.lhs.clone(), &p.variable, p.domain));
    Some(E::SwapSidesRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: p.variable.clone(),
        substep: Box::new(substep),
    })
}

fn added(p: &EquationProblem, term: &Basic) -> PolyResult<EquationProblem> {
    Ok(problem(
        expand(&a2(&p.lhs, term)?)?,
        expand(&a2(&p.rhs, term)?)?,
        &p.variable,
        p.domain,
    ))
}

fn add_step(p: &EquationProblem, substep: EquationStep, term: Basic) -> Out {
    Some(E::AddToBothSidesRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: p.variable.clone(),
        substep: Box::new(substep),
        term,
    })
}

fn move_unknowns(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let (moving, _) = split(&p.rhs, &p.variable).ok()?;
    if moving == int(0) {
        return None;
    }
    let term: Basic = neg(&moving).ok()?;
    let substep = s.solve(&added(p, &term).ok()?);
    add_step(p, substep, term)
}

fn move_constants(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let (_, moving) = split(&p.lhs, &p.variable).ok()?;
    if moving == int(0) {
        return None;
    }
    let term: Basic = neg(&moving).ok()?;
    let substep = s.solve(&added(p, &term).ok()?);
    add_step(p, substep, term)
}

fn divide(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() {
        return None;
    }
    let (coefficient, rest) = as_independent(&p.lhs, x, false).ok()?;
    if rest != *x || coefficient == int(1) || coefficient == int(0) || p.rhs.has(x) {
        return None;
    }
    let substep = s.solve(&problem(
        x.clone(),
        normal(&d2(&p.rhs, &coefficient).ok()?).ok()?,
        x,
        p.domain,
    ));
    Some(E::DivideBothSidesRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        divisor: coefficient,
    })
}

fn standard_form(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let lhs: Basic = expand(&s2(&p.lhs, &p.rhs).ok()?).ok()?;
    if p.rhs == int(0) {
        if lhs == p.lhs {
            return None;
        }
        let substep = s.solve(&problem(lhs, int(0), x, p.domain));
        return Some(E::RewriteRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            variable: x.clone(),
            substep: Box::new(substep),
        });
    }
    let substep = s.solve(&problem(lhs, int(0), x, p.domain));
    add_step(p, substep, neg(&p.rhs).ok()?)
}

fn zero_product(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.rhs != int(0) {
        return None;
    }
    let mut symbols: Vec<Basic> = p
        .lhs
        .free_symbols()
        .into_iter()
        .filter(|v| v.is_symbol() && v != x)
        .collect();
    symbols.sort_by_key(|v| v.to_string());
    let mut gens: Vec<Basic> = vec![x.clone()];
    gens.extend(symbols);
    let (constant, listed) = match crate::polys::polytools::factor_list(&p.lhs, &gens) {
        Ok(found) => found,
        Err(crate::error::PolyError::NotImplemented(m)) if m.starts_with("not ported") => return None,
        Err(_) => return None,
    };
    let factors: Vec<Basic> = listed
        .iter()
        .filter(|(f, _)| f.has(x))
        .map(|(f, _)| f.clone())
        .collect();
    let mut outside: Vec<Basic> = vec![constant];
    for (f, k) in &listed {
        if !f.has(x) {
            outside.push(pw(f, k).ok()?);
        }
    }
    let outside: Basic = crate::core::mul::mul(&outside).ok()?;
    if factors.len() < 2 && !listed.iter().any(|(f, k)| f.has(x) && *k != int(1)) {
        return None;
    }
    let cases: Vec<EquationStep> = factors
        .iter()
        .map(|f| s.solve(&problem(f.clone(), int(0), x, p.domain)))
        .collect();
    Some(E::ZeroProductRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        constant: outside,
        factors,
        cases,
    })
}

fn quadratic_formula(p: &EquationProblem, _: &Context, _: &mut Solver) -> Out {
    let cs = polynomial(&p.lhs, &p.variable)?;
    if p.rhs != int(0) || cs.len() != 3 {
        return None;
    }
    let (a, b, c) = (cs[0].clone(), cs[1].clone(), cs[2].clone());
    let discriminant: Basic = normal(&s2(&pw(&b, &int(2)).ok()?, &m2(&m2(&int(4), &a).ok()?, &c).ok()?).ok()?).ok()?;
    Some(E::QuadraticFormulaRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: p.variable.clone(),
        a,
        b,
        c,
        discriminant,
        domain: p.domain,
    })
}

fn complete_square(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let cs = polynomial(&p.lhs, x)?;
    if p.rhs != int(0) || cs.len() != 3 {
        return None;
    }
    let (a, b, c) = (&cs[0], &cs[1], &cs[2]);
    let shift: Basic = normal(&d2(b, &m2(&int(2), a).ok()?).ok()?).ok()?;
    let value: Basic = normal(&s2(&pw(&shift, &int(2)).ok()?, &d2(c, a).ok()?).ok()?).ok()?;
    let substep = s.solve(&problem(
        pw(&a2(x, &shift).ok()?, &int(2)).ok()?,
        value.clone(),
        x,
        p.domain,
    ));
    Some(E::CompleteSquareRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        shift,
        value,
    })
}

fn root(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let (lhs, value) = (&p.lhs, &p.rhs);
    if !(lhs.is_pow() && lhs.args()[1].is_integer_value()) || value.has(x) {
        return None;
    }
    let n: i64 = i64::try_from(&crate::numbers::numer(lhs.args()[1].as_rational()?)).ok()?;
    if n < 2 {
        return None;
    }
    let base: Basic = lhs.args()[0].clone();
    let negative: bool = is(value, Predicate::negative) == Some(true);
    let roots: Vec<Basic> = if *value == int(0) {
        vec![int(0)]
    } else if p.domain == Domain::Complexes {
        let modulus: Basic = if negative {
            pw(&neg(value).ok()?, &rat(1, n)).ok()?
        } else {
            pw(value, &rat(1, n)).ok()?
        };
        let mut found: Vec<Basic> = Vec::new();
        for k in 0..n {
            let angle: Basic = m2(&Basic::pi(), &if negative { rat(2 * k + 1, n) } else { rat(2 * k, n) }).ok()?;
            let rotation: Basic = a2(
                &apply(Function::Cos, &angle).ok()?,
                &m2(&imaginary(), &apply(Function::Sin, &angle).ok()?).ok()?,
            )
            .ok()?;
            let r: Basic = expand(&m2(&modulus, &rotation).ok()?).ok()?;
            if !found.contains(&r) {
                found.push(r);
            }
        }
        found
    } else if n % 2 == 1 {
        vec![if !negative {
            pw(value, &rat(1, n)).ok()?
        } else {
            neg(&pw(&neg(value).ok()?, &rat(1, n)).ok()?).ok()?
        }]
    } else if negative {
        Vec::new()
    } else {
        let principal: Basic = pw(value, &rat(1, n)).ok()?;
        vec![principal.clone(), neg(&principal).ok()?]
    };
    let cases: Vec<EquationStep> = roots
        .iter()
        .map(|r| s.solve(&problem(base.clone(), r.clone(), x, p.domain)))
        .collect();
    Some(E::RootRule {
        lhs: lhs.clone(),
        rhs: value.clone(),
        variable: x.clone(),
        base,
        exponent: n,
        value: value.clone(),
        cases,
    })
}

fn checked(p: &EquationProblem, substep: &EquationStep) -> (Vec<Basic>, Vec<Basic>) {
    if substep.contains_dont_know() {
        return (Vec::new(), Vec::new());
    }
    let Ok(found) = substep.solutions() else {
        return (Vec::new(), Vec::new());
    };
    if !matches!(found.head(), Head::FiniteSet) {
        return (Vec::new(), Vec::new());
    }
    let (mut rejected, mut unverified) = (Vec::new(), Vec::new());
    for candidate in found.args() {
        match satisfies(p, candidate) {
            Some(false) => rejected.push(candidate.clone()),
            None => unverified.push(candidate.clone()),
            Some(true) => {}
        }
    }
    (rejected, unverified)
}

fn clear_denominators(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let (left, right) = (denominator(&p.lhs, x).ok()?, denominator(&p.rhs, x).ok()?);
    if left == int(1) && right == int(1) {
        return None;
    }
    let mut common: Basic = if right == int(1) || normal(&d2(&left, &right).ok()?).ok()? == int(1) {
        left.clone()
    } else if left == int(1) {
        right.clone()
    } else {
        expand(&m2(&left, &right).ok()?).ok()?
    };
    if left != int(1) && right != int(1) && common != left {
        let quotient: Basic = normal(&d2(&left, &right).ok()?).ok()?;
        let (numerator, _) = crate::core::expr::as_numer_denom(&quotient).ok()?;
        common = expand(&m2(&right, &numerator).ok()?).ok()?;
    }
    let lhs: Basic = expand(&normal(&m2(&p.lhs, &common).ok()?).ok()?).ok()?;
    let rhs: Basic = expand(&normal(&m2(&p.rhs, &common).ok()?).ok()?).ok()?;
    let substep = s.solve(&problem(lhs, rhs, x, p.domain));
    let (rejected, unverified) = checked(p, &substep);
    Some(E::ClearDenominatorsRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        rejected,
        unverified,
        denominator: common,
    })
}

fn isolate_by(p: &EquationProblem, s: &mut Solver, kind: fn(&Basic, &Basic) -> bool) -> Out {
    let x: &Basic = &p.variable;
    let terms: Vec<Basic> = terms_of(&p.lhs);
    let chosen: Vec<&Basic> = terms.iter().filter(|t| kind(t, x)).collect();
    if chosen.is_empty() || terms.len() == 1 {
        return None;
    }
    let first: Basic = chosen[0].clone();
    // `t is not chosen[0]`: by identity, the first occurrence only
    let position: usize = terms.iter().position(|t| *t == first).expect("chosen");
    let others: Vec<Basic> = terms
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != position)
        .map(|(_, t)| t.clone())
        .collect();
    let moving: Basic = crate::core::add::add(&others).ok()?;
    let substep = s.solve(&problem(first, expand(&s2(&p.rhs, &moving).ok()?).ok()?, x, p.domain));
    add_step(p, substep, neg(&moving).ok()?)
}

fn isolate_radical(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    isolate_by(p, s, radicals)
}

fn isolate_absolute(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    isolate_by(p, s, absolute_values)
}

fn isolate_kernel(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    isolate_by(p, s, transcendental)
}

fn square(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() {
        return None;
    }
    let (coefficient, rest) = as_independent(&p.lhs, x, false).ok()?;
    if !(rest.is_pow() && rest.args()[1] == rat(1, 2)) || coefficient == int(0) {
        return None;
    }
    let lhs: Basic = expand(&m2(&pw(&coefficient, &int(2)).ok()?, &rest.args()[0]).ok()?).ok()?;
    let rhs: Basic = expand(&pw(&p.rhs, &int(2)).ok()?).ok()?;
    let substep = s.solve(&problem(lhs, rhs, x, p.domain));
    let (rejected, unverified) = checked(p, &substep);
    Some(E::SquareBothSidesRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        rejected,
        unverified,
    })
}

fn absolute_value(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() {
        return None;
    }
    let (coefficient, rest) = as_independent(&p.lhs, x, false).ok()?;
    if func(&rest) != Some(Function::Abs) || coefficient == int(0) {
        return None;
    }
    let inner: Basic = rest.args()[0].clone();
    let value: Basic = normal(&d2(&p.rhs, &coefficient).ok()?).ok()?;
    if is(&value, Predicate::negative) == Some(true) || is(&value, Predicate::real) == Some(false) {
        return contradiction(p);
    }
    let targets: Vec<Basic> = if value == int(0) {
        vec![value.clone()]
    } else {
        vec![value.clone(), neg(&value).ok()?]
    };
    let mut cases: Vec<EquationStep> = Vec::new();
    for target in &targets {
        cases.push(s.solve(&problem(inner.clone(), expand(target).ok()?, x, p.domain)));
    }
    let (mut rejected, mut unverified): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for case in &cases {
        if case.contains_dont_know() {
            continue;
        }
        let Ok(found) = case.solutions() else { continue };
        if !matches!(found.head(), Head::FiniteSet) {
            continue;
        }
        for candidate in found.args() {
            match satisfies(p, candidate) {
                Some(false) if !rejected.contains(candidate) => rejected.push(candidate.clone()),
                None if !unverified.contains(candidate) => unverified.push(candidate.clone()),
                _ => {}
            }
        }
    }
    Some(E::AbsoluteValueRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        inner,
        cases,
        rejected,
        unverified,
        partial: p.domain == Domain::Complexes && value != int(0),
    })
}

fn exponential_parts(e: &Basic, x: &Basic) -> Option<(Basic, Basic)> {
    if func(e) == Some(Function::Exp) {
        let argument: &Basic = &e.args()[0];
        if argument.has(x) {
            return Some((Basic::e(), argument.clone()));
        }
    }
    if e.is_pow() {
        let (base, exponent) = (&e.args()[0], &e.args()[1]);
        if !base.has(x) && exponent.has(x) && is(base, Predicate::positive) == Some(true) && *base != int(1) {
            return Some((base.clone(), exponent.clone()));
        }
    }
    None
}

fn fresh(name: &str, p: &EquationProblem, integer: bool) -> Basic {
    let mut used: Vec<String> = Vec::new();
    for e in [&p.lhs, &p.rhs] {
        for v in e.free_symbols() {
            if v.is_symbol() {
                used.push(v.to_string());
            }
        }
    }
    used.push(p.variable.to_string());
    let (mut candidate, mut k) = (name.to_string(), 0);
    while used.contains(&candidate) {
        k += 1;
        candidate = format!("{}{}", name, k);
    }
    let assumptions: Vec<(Predicate, bool)> = if integer {
        vec![(Predicate::integer, true)]
    } else {
        Vec::new()
    };
    crate::core::symbol::Symbol::new(&candidate, &assumptions).0
}

fn two_pi_i_n_over(n: &Basic, scale: &Basic) -> R {
    d2(&m2(&m2(&m2(&int(2), &Basic::pi())?, &imaginary())?, n)?, scale)
}

fn take_logarithm(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() || p.rhs.has(x) {
        return None;
    }
    let (coefficient, rest) = as_independent(&p.lhs, x, false).ok()?;
    let (base, exponent) = exponential_parts(&rest, x)?;
    if coefficient == int(0) {
        return None;
    }
    let value: Basic = normal(&d2(&p.rhs, &coefficient).ok()?).ok()?;
    let log = |e: &Basic| crate::functions::elementary::exponential::log(e);
    let scale: Basic = if func(&rest) == Some(Function::Exp) {
        int(1)
    } else {
        log(&base).ok()?
    };
    let make = |substep: EquationStep, parameter: Option<Basic>| E::TakeLogarithmRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        base: base.clone(),
        parameter,
    };
    if p.domain == Domain::Complexes {
        if value == int(0) {
            return contradiction(p);
        }
        let n: Basic = fresh("n", p, true);
        let target: Basic = a2(
            &normal(&d2(&log(&value).ok()?, &scale).ok()?).ok()?,
            &two_pi_i_n_over(&n, &scale).ok()?,
        )
        .ok()?;
        let substep = s.solve(&problem(exponent, target, x, p.domain));
        return Some(make(substep, Some(n)));
    }
    if is(&value, Predicate::positive) != Some(true) {
        if is(&value, Predicate::negative) == Some(true) || value == int(0) {
            return contradiction(p);
        }
        return None;
    }
    let substep = s.solve(&problem(
        exponent,
        normal(&d2(&log(&value).ok()?, &scale).ok()?).ok()?,
        x,
        p.domain,
    ));
    Some(make(substep, None))
}

fn equal_exponents(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let (left, right) = (exponential_parts(&p.lhs, x)?, exponential_parts(&p.rhs, x)?);
    if left.0 != right.0 {
        return None;
    }
    let make = |substep: EquationStep, parameter: Option<Basic>| E::EqualExponentsRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        base: left.0.clone(),
        parameter,
    };
    if p.domain == Domain::Complexes {
        let n: Basic = fresh("n", p, true);
        let scale: Basic = if left.0 == Basic::e() {
            int(1)
        } else {
            crate::functions::elementary::exponential::log(&left.0).ok()?
        };
        let substep = s.solve(&problem(
            left.1.clone(),
            a2(&right.1, &two_pi_i_n_over(&n, &scale).ok()?).ok()?,
            x,
            p.domain,
        ));
        return Some(make(substep, Some(n)));
    }
    let substep = s.solve(&problem(left.1.clone(), right.1.clone(), x, p.domain));
    Some(make(substep, None))
}

fn logarithm_argument(e: &Basic, x: &Basic) -> Option<Basic> {
    if func(e) == Some(Function::Log) && e.args()[0].has(x) {
        return Some(e.args()[0].clone());
    }
    None
}

fn exponentiate(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() || p.rhs.has(x) {
        return None;
    }
    let (coefficient, rest) = as_independent(&p.lhs, x, false).ok()?;
    let argument: Basic = logarithm_argument(&rest, x)?;
    if coefficient == int(0) {
        return None;
    }
    let value: Basic = normal(&d2(&p.rhs, &coefficient).ok()?).ok()?;
    if p.domain == Domain::Complexes && is(&value, Predicate::real) != Some(true) {
        return None;
    }
    let substep = s.solve(&problem(
        argument,
        crate::functions::elementary::exponential::exp(&value).ok()?,
        x,
        p.domain,
    ));
    Some(E::ExponentiateRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
    })
}

fn equal_arguments(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let (left, right) = (logarithm_argument(&p.lhs, x)?, logarithm_argument(&p.rhs, x)?);
    let substep = s.solve(&problem(left, right, x, p.domain));
    let (rejected, unverified) = checked(p, &substep);
    Some(E::EqualArgumentsRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        rejected,
        unverified,
    })
}

fn combined(e: &Basic, x: &Basic) -> Option<Basic> {
    let mut product: Basic = int(1);
    let mut others: Vec<Basic> = Vec::new();
    let mut count: usize = 0;
    for term in terms_of(e) {
        let (coefficient, rest) = if func(&term) == Some(Function::Log) {
            (int(1), term.clone())
        } else {
            as_independent(&term, x, false).ok()?
        };
        match logarithm_argument(&rest, x) {
            Some(argument) if coefficient.is_integer_value() => {
                product = m2(&product, &pw(&argument, &coefficient).ok()?).ok()?;
                count += 1;
            }
            _ => others.push(term.clone()),
        }
    }
    if count < 2 {
        return None;
    }
    let mut all: Vec<Basic> = vec![crate::functions::elementary::exponential::log(&product).ok()?];
    all.extend(others);
    crate::core::add::add(&all).ok()
}

fn combine_logarithms(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let (left, right) = (combined(&p.lhs, x), combined(&p.rhs, x));
    if left.is_none() && right.is_none() {
        return None;
    }
    let substep = s.solve(&problem(
        left.unwrap_or_else(|| p.lhs.clone()),
        right.unwrap_or_else(|| p.rhs.clone()),
        x,
        p.domain,
    ));
    let (rejected, unverified) = checked(p, &substep);
    Some(E::CombineLogarithmsRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        rejected,
        unverified,
    })
}

fn primitive_root(n: i64) -> (i64, u32) {
    let bits: u32 = 64 - (n as u64).leading_zeros();
    for k in (2..=bits).rev() {
        let r: i64 = (n as f64).powf(1.0 / k as f64).round() as i64;
        for candidate in [r - 1, r, r + 1] {
            if candidate >= 2 && (candidate as i128).checked_pow(k).is_some_and(|v| v == n as i128) {
                return (candidate, k);
            }
        }
    }
    (n, 1)
}

fn common_base(difference: &Basic, x: &Basic, u: &Basic) -> Option<(Basic, Basic)> {
    let mut nodes: Vec<Basic> = difference.atoms(Some(&|n: &Basic| n.is_pow()));
    nodes.sort_by_key(|n| n.to_string());
    let mut powers: Vec<(Basic, i64)> = Vec::new();
    for node in nodes {
        if node.args()[1].has(x) {
            let base: &Basic = &node.args()[0];
            if base.is_integer_value() {
                let b: i64 = i64::try_from(&crate::numbers::numer(base.as_rational()?)).ok()?;
                if b >= 2 {
                    powers.push((node.clone(), b));
                }
            }
        }
    }
    if powers.len() < 2 {
        return None;
    }
    let first_exponent: Basic = powers[0].0.args()[1].clone();
    if powers.iter().any(|(n, _)| n.args()[1] != first_exponent) {
        return None;
    }
    let roots: Vec<(i64, u32)> = powers.iter().map(|(_, b)| primitive_root(*b)).collect();
    if roots.iter().any(|(r, _)| *r != roots[0].0) {
        return None;
    }
    let root_value: i64 = roots[0].0;
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for ((node, _), (_, k)) in powers.iter().zip(roots.iter()) {
        rule.insert(node.clone(), pw(u, &int(i64::from(*k))).ok()?);
    }
    Some((
        crate::core::function::xreplace_evaluated(difference, &rule).ok()?,
        pw(&int(root_value), &first_exponent).ok()?,
    ))
}

fn substitution(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let difference: Basic = expand(&s2(&p.lhs, &p.rhs).ok()?).ok()?;
    let u: Basic = fresh("u", p, false);
    let mut candidates: Vec<(Basic, Basic)> = Vec::new();
    let mut back: Vec<(Basic, Basic)> = Vec::new();
    let prepared: Basic = crate::simplify::normalform::exponentials_of(std::slice::from_ref(&difference), &mut back)
        .ok()?
        .remove(0);
    if back.len() == 1 {
        let (placeholder, expression) = back[0].clone();
        candidates.push((subs(&prepared, &placeholder, &u).ok()?, expression));
    }
    let kinds = [
        Function::Sin,
        Function::Cos,
        Function::Tan,
        Function::Log,
        Function::Abs,
    ];
    let mut nodes: Vec<Basic> = difference.atoms(Some(&|n: &Basic| matches!(func(n), Some(f) if kinds.contains(&f))));
    nodes.sort_by_key(|n| n.to_string());
    for node in nodes {
        if node.has(x) {
            let rule: HashMap<Basic, Basic> = [(node.clone(), u.clone())].into_iter().collect();
            candidates.push((
                crate::core::function::xreplace_evaluated(&difference, &rule).ok()?,
                node,
            ));
        }
    }
    if let Some(common) = common_base(&difference, x, &u) {
        candidates.push(common);
    }
    for (equation, expression) in candidates {
        let cs = if equation.has(x) {
            None
        } else {
            polynomial(&equation, &u)
        };
        if cs.is_none_or_short() {
            continue;
        }
        let substep = s.solve(&problem(equation.clone(), int(0), &u, p.domain));
        if substep.contains_dont_know() {
            continue;
        }
        let Ok(found) = substep.solutions() else { continue };
        if !matches!(found.head(), Head::FiniteSet | Head::EmptySet) {
            continue;
        }
        let mut cases: Vec<EquationStep> = Vec::new();
        if matches!(found.head(), Head::FiniteSet) {
            for value in found.args() {
                cases.push(s.solve(&problem(expression.clone(), value.clone(), x, p.domain)));
            }
        }
        return Some(E::SubstitutionRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            variable: x.clone(),
            placeholder: u,
            expression,
            substep: Box::new(substep),
            cases,
        });
    }
    None
}

trait Short {
    fn is_none_or_short(&self) -> bool;
}

impl Short for Option<Vec<Basic>> {
    fn is_none_or_short(&self) -> bool {
        self.as_ref().map_or(true, |c| c.len() < 3)
    }
}

fn periodic(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() || p.rhs.has(x) {
        return None;
    }
    let (coefficient, rest) = as_independent(&p.lhs, x, false).ok()?;
    let f: Function = func(&rest)?;
    if !matches!(f, Function::Sin | Function::Cos | Function::Tan) || coefficient == int(0) {
        return None;
    }
    let argument: Basic = rest.args()[0].clone();
    let value: Basic = normal(&d2(&p.rhs, &coefficient).ok()?).ok()?;
    let n: Basic = fresh("n", p, true);
    let pi: Basic = Basic::pi();
    let two_pi_n: Basic = m2(&m2(&int(2), &pi).ok()?, &n).ok()?;
    let (name, angles): (&str, Vec<Basic>) = if f == Function::Tan {
        if value == imaginary() || value == neg(&imaginary()).ok()? {
            return contradiction(p);
        }
        (
            "tan",
            vec![a2(&apply(Function::Atan, &value).ok()?, &m2(&pi, &n).ok()?).ok()?],
        )
    } else {
        if p.domain == Domain::Reals
            && (is(&s2(&value, &int(1)).ok()?, Predicate::positive) == Some(true)
                || is(&a2(&value, &int(1)).ok()?, Predicate::negative) == Some(true))
        {
            return contradiction(p);
        }
        let unit: bool = value == int(1) || value == int(-1);
        if f == Function::Sin {
            let principal: Basic = apply(Function::Asin, &value).ok()?;
            let first: Basic = a2(&principal, &two_pi_n).ok()?;
            if unit {
                ("sin", vec![first])
            } else {
                ("sin", vec![first, a2(&s2(&pi, &principal).ok()?, &two_pi_n).ok()?])
            }
        } else {
            let principal: Basic = apply(Function::Acos, &value).ok()?;
            let first: Basic = a2(&principal, &two_pi_n).ok()?;
            if unit {
                ("cos", vec![first])
            } else {
                ("cos", vec![first, a2(&neg(&principal).ok()?, &two_pi_n).ok()?])
            }
        }
    };
    let mut cases: Vec<EquationStep> = Vec::new();
    for angle in &angles {
        cases.push(s.solve(&problem(argument.clone(), expand(angle).ok()?, x, p.domain)));
    }
    Some(E::PeriodicRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        function: name.to_string(),
        value,
        parameter: n,
        cases,
    })
}

fn tangent(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    if p.lhs.is_add() || p.rhs.is_add() {
        return None;
    }
    let (mut a, mut left) = as_independent(&p.lhs, x, false).ok()?;
    let (mut b, mut right) = as_independent(&p.rhs, x, false).ok()?;
    if func(&left) == Some(Function::Cos) && func(&right) == Some(Function::Sin) {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut left, &mut right);
    }
    if !(func(&left) == Some(Function::Sin) && func(&right) == Some(Function::Cos))
        || left.args() != right.args()
        || a == int(0)
    {
        return None;
    }
    let argument: Basic = left.args()[0].clone();
    let substep = s.solve(&problem(
        apply(Function::Tan, &argument).ok()?,
        normal(&d2(&b, &a).ok()?).ok()?,
        x,
        p.domain,
    ));
    Some(E::TangentRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
    })
}

fn cardano(p: &EquationProblem, _: &Context, _: &mut Solver) -> Out {
    let cs = polynomial(&p.lhs, &p.variable)?;
    if p.rhs != int(0) || cs.len() != 4 || !cs.iter().all(|c| c.is_rational_class()) {
        return None;
    }
    let (a, b, c, d) = (&cs[0], &cs[1], &cs[2], &cs[3]);
    let shift: Basic = d2(b, &m2(&int(3), a).ok()?).ok()?;
    let a2_: Basic = pw(a, &int(2)).ok()?;
    let pp: Basic = normal(
        &s2(
            &d2(c, a).ok()?,
            &d2(&pw(b, &int(2)).ok()?, &m2(&int(3), &a2_).ok()?).ok()?,
        )
        .ok()?,
    )
    .ok()?;
    let q_parts: Basic = a2(
        &s2(
            &d2(
                &m2(&int(2), &pw(b, &int(3)).ok()?).ok()?,
                &m2(&int(27), &pw(a, &int(3)).ok()?).ok()?,
            )
            .ok()?,
            &d2(&m2(b, c).ok()?, &m2(&int(3), &a2_).ok()?).ok()?,
        )
        .ok()?,
        &d2(d, a).ok()?,
    )
    .ok()?;
    let qq: Basic = normal(&q_parts).ok()?;
    let test: Basic = a2(
        &d2(&pw(&qq, &int(2)).ok()?, &int(4)).ok()?,
        &d2(&pw(&pp, &int(3)).ok()?, &int(27)).ok()?,
    )
    .ok()?;
    if is(&test, Predicate::positive) != Some(true) {
        return None;
    }
    Some(E::CardanoRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: p.variable.clone(),
        shift,
        p: pp,
        q: qq,
        domain: p.domain,
    })
}

fn real_roots(p: &EquationProblem, _: &Context, _: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let cs = polynomial(&p.lhs, x)?;
    if p.rhs != int(0) || cs.len() < 4 {
        return None;
    }
    let mut values: Vec<crate::numbers::Rat> = Vec::new();
    for c in &cs {
        values.push(c.as_rational()?.clone());
    }
    let found = crate::polys::cad::samplepoints::real_roots_of(&values).ok()?;
    let roots: Vec<Basic> = found
        .iter()
        .map(|r| crate::polys::cad::indexedroot::real_root(r, &x.to_string()))
        .collect::<PolyResult<_>>()
        .ok()?;
    let missing: usize = if p.domain == Domain::Complexes {
        cs.len() - 1 - roots.len()
    } else {
        0
    };
    Some(E::RealRootsRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        roots,
        missing,
    })
}

fn transcendental(e: &Basic, x: &Basic) -> bool {
    let kinds = [
        Function::Exp,
        Function::Log,
        Function::Sin,
        Function::Cos,
        Function::Tan,
    ];
    if e.atoms(Some(&|n: &Basic| matches!(func(n), Some(f) if kinds.contains(&f))))
        .iter()
        .any(|n| n.has(x))
    {
        return true;
    }
    e.atoms(Some(&|n: &Basic| n.is_pow()))
        .iter()
        .any(|n| !n.args()[0].has(x) && n.args()[1].has(x))
}

fn binomial(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let cs = polynomial(&p.lhs, x)?;
    if p.rhs != int(0) {
        return None;
    }
    let n: i64 = cs.len() as i64 - 1;
    let value: Basic = normal(&d2(&neg(&cs[cs.len() - 1]).ok()?, &cs[0]).ok()?).ok()?;
    let substep = s.solve(&problem(pw(x, &int(n)).ok()?, value, x, p.domain));
    Some(E::DivideBothSidesRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        substep: Box::new(substep),
        divisor: cs[0].clone(),
    })
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

fn power_substitution(p: &EquationProblem, _: &Context, s: &mut Solver) -> Out {
    let x: &Basic = &p.variable;
    let cs = polynomial(&p.lhs, x)?;
    if p.rhs != int(0) {
        return None;
    }
    let degree: i64 = cs.len() as i64 - 1;
    let exponents: Vec<i64> = cs
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != int(0))
        .map(|(position, _)| degree - position as i64)
        .collect();
    let k: i64 = exponents.iter().fold(0, |acc, e| gcd(acc, *e));
    if k < 2 || exponents.len() < 3 {
        return None;
    }
    let u: Basic = fresh("u", p, false);
    let mut terms: Vec<Basic> = Vec::new();
    for (position, c) in cs.iter().enumerate() {
        if *c != int(0) {
            terms.push(m2(c, &pw(&u, &int((degree - position as i64) / k)).ok()?).ok()?);
        }
    }
    let equation: Basic = crate::core::add::add(&terms).ok()?;
    let substep = s.solve(&problem(equation, int(0), &u, p.domain));
    if substep.contains_dont_know() {
        return None;
    }
    let found: Basic = substep.solutions().ok()?;
    if !matches!(found.head(), Head::FiniteSet | Head::EmptySet) {
        return None;
    }
    let power: Basic = pw(x, &int(k)).ok()?;
    let mut cases: Vec<EquationStep> = Vec::new();
    if matches!(found.head(), Head::FiniteSet) {
        for value in found.args() {
            cases.push(s.solve(&problem(power.clone(), value.clone(), x, p.domain)));
        }
    }
    Some(E::SubstitutionRule {
        lhs: p.lhs.clone(),
        rhs: p.rhs.clone(),
        variable: x.clone(),
        placeholder: u,
        expression: power,
        substep: Box::new(substep),
        cases,
    })
}

type Rl = super::core::Rule<EquationProblem, EquationStep>;

fn candidates(p: &EquationProblem) -> Vec<Rl> {
    let (lhs, rhs, x) = (&p.lhs, &p.rhs, &p.variable);
    if !lhs.has(x) && !rhs.has(x) {
        return vec![trivial];
    }
    if lhs == x && !rhs.has(x) {
        return vec![solved];
    }
    let one: Basic = int(1);
    if denominator(lhs, x).map_or(true, |d| d != one) || denominator(rhs, x).map_or(true, |d| d != one) {
        return vec![clear_denominators];
    }
    if absolute_values(lhs, x) || absolute_values(rhs, x) {
        if !absolute_values(lhs, x) {
            return vec![swap];
        }
        return vec![absolute_value, isolate_absolute, substitution];
    }
    if radicals(lhs, x) || radicals(rhs, x) {
        if !radicals(lhs, x) {
            return vec![swap];
        }
        return vec![isolate_radical, square];
    }
    if transcendental(lhs, x) || transcendental(rhs, x) {
        if !transcendental(lhs, x) {
            return vec![swap];
        }
        return vec![
            take_logarithm,
            exponentiate,
            periodic,
            equal_exponents,
            equal_arguments,
            tangent,
            combine_logarithms,
            substitution,
            isolate_kernel,
            rewrite,
        ];
    }
    if lhs.is_pow() && lhs.args()[1].is_integer_value() && !rhs.has(x) {
        let big: bool = lhs.args()[1]
            .as_rational()
            .is_some_and(|r| *r >= crate::numbers::Rat::from(crate::numbers::Int::from(2)));
        if big && polynomial(&lhs.args()[0], x).is_some_and(|c| c.len() == 2) {
            return vec![root];
        }
    }
    let Ok(difference) = s2(lhs, rhs) else {
        return Vec::new();
    };
    let Some(cs) = polynomial(&difference, x) else {
        return Vec::new();
    };
    let degree: i64 = cs.len() as i64 - 1;
    if degree <= 0 {
        return vec![trivial];
    }
    if degree == 1 {
        return vec![rewrite, move_unknowns, move_constants, divide];
    }
    if *rhs != int(0) || expand(lhs).map_or(true, |e| e != *lhs) {
        return vec![standard_form];
    }
    if degree == 2 {
        return vec![zero_product, quadratic_formula, complete_square];
    }
    let binomial_form: bool = cs[1..cs.len() - 1].iter().all(|c| *c == int(0));
    let mut rules: Vec<Rl> = vec![zero_product];
    if binomial_form {
        rules.push(binomial);
    }
    rules.push(power_substitution);
    if degree == 3 {
        rules.push(cardano);
    }
    rules.push(real_roots);
    rules
}

/// The derivation of the solutions of `lhs = rhs` in `x` (`solve_steps`).
pub fn solve_steps(
    lhs: &Basic,
    rhs: &Basic,
    x: &Basic,
    domain: Domain,
    branch: bool,
    max_depth: Option<usize>,
) -> EquationStep {
    let mut solver: Solver = StepSolver::new(
        candidates,
        |p| E::DontKnowRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            variable: p.variable.clone(),
        },
        |p| E::InProgressRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            variable: p.variable.clone(),
        },
        |p, alternatives| E::AlternativeRule {
            lhs: p.lhs.clone(),
            rhs: p.rhs.clone(),
            variable: p.variable.clone(),
            alternatives,
        },
        branch,
        max_depth,
    );
    solver.solve(&problem(lhs.clone(), rhs.clone(), x, domain))
}

/// The solutions found by the rules, or `None` (`manual_solve`).
pub fn manual_solve(lhs: &Basic, rhs: &Basic, x: &Basic, domain: Domain) -> PolyResult<Option<Basic>> {
    let steps: EquationStep = solve_steps(lhs, rhs, x, domain, false, Some(30));
    if steps.contains_dont_know() {
        return Ok(None);
    }
    Ok(Some(steps.solutions()?))
}
