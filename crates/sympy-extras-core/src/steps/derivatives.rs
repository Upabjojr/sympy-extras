//! Derivatives step by step, ported from `sympy_extras.steps.derivatives`:
//! the tree of the rules of differentiation, built by rules which the
//! solver of [`super::core`] drives. Every step has its `result`,
//! assembled from those of its substeps as the rule says, without
//! simplification (Python's operators, one binary evaluation each).
//!
//! `ConstantRule`, `VariableRule`, `PowerRule`, `ExponentialRule`,
//! `FunctionRule` (the table of the elementary functions),
//! `ConstantMultipleRule`, `SumRule`, `ProductRule`, `QuotientRule`,
//! `ChainRule` (the outer function of a placeholder and the inner one),
//! `RewriteRule` (`f**g` as `exp(g*log(f))`, minor), `AlternativeRule`
//! (with `branch`: a quotient also as a product), `DontKnowRule`,
//! `InProgressRule`.

// The design is that of sympy/integrals/manualintegrate.py in SymPy's development branch.

use super::core::{Context, Rule, Step, StepSolver};
use crate::core::add::{add, sub};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_independent, as_numer_denom};
use crate::core::mul::{div, mul, neg};
use crate::core::power::pow;
use crate::error::PolyResult;
use crate::functions::apply_function;

/// The derivative of `expression` with respect to `variable`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DerivativeProblem {
    pub expression: Basic,
    pub variable: Basic,
}

/// A step of a differentiation: the rule applied to `expression`, a
/// function of `variable`, with the fields of the rule.
#[derive(Debug, Clone, PartialEq)]
pub enum DerivativeStep {
    ConstantRule {
        expression: Basic,
        variable: Basic,
    },
    VariableRule {
        expression: Basic,
        variable: Basic,
    },
    /// `variable**exponent` with a constant exponent
    PowerRule {
        expression: Basic,
        variable: Basic,
        exponent: Basic,
    },
    /// `base**variable` with a constant base
    ExponentialRule {
        expression: Basic,
        variable: Basic,
        base: Basic,
    },
    /// an elementary function of the variable itself; `function` its name
    FunctionRule {
        expression: Basic,
        variable: Basic,
        function: &'static str,
    },
    ConstantMultipleRule {
        expression: Basic,
        variable: Basic,
        constant: Basic,
        rest: Basic,
        substep: Box<DerivativeStep>,
    },
    SumRule {
        expression: Basic,
        variable: Basic,
        terms: Vec<DerivativeStep>,
    },
    ProductRule {
        expression: Basic,
        variable: Basic,
        first: Basic,
        second: Basic,
        first_step: Box<DerivativeStep>,
        second_step: Box<DerivativeStep>,
    },
    QuotientRule {
        expression: Basic,
        variable: Basic,
        numerator: Basic,
        denominator: Basic,
        numerator_step: Box<DerivativeStep>,
        denominator_step: Box<DerivativeStep>,
    },
    /// the function of `placeholder` differentiated by `outer_step`, at
    /// `placeholder = inner`
    ChainRule {
        expression: Basic,
        variable: Basic,
        inner: Basic,
        placeholder: Basic,
        outer_step: Box<DerivativeStep>,
        inner_step: Box<DerivativeStep>,
    },
    /// the expression equals `rewritten`, which is differentiated (minor)
    RewriteRule {
        expression: Basic,
        variable: Basic,
        rewritten: Basic,
        substep: Box<DerivativeStep>,
    },
    /// several derivations; the result is that of the first
    AlternativeRule {
        expression: Basic,
        variable: Basic,
        alternatives: Vec<DerivativeStep>,
    },
    /// no rule applies: the derivative is left as it is
    DontKnowRule {
        expression: Basic,
        variable: Basic,
    },
    /// the problem met again while it is being solved
    InProgressRule {
        expression: Basic,
        variable: Basic,
    },
}

use DerivativeStep as D;

fn m2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), b.clone()])
}

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn sqrt(x: &Basic) -> PolyResult<Basic> {
    pow(x, &Basic::half())
}

fn call(f: Function, x: &Basic) -> PolyResult<Basic> {
    apply_function(f, std::slice::from_ref(x))
}

/// The elementary functions of the table, by name.
const NAMES: [(Function, &str); 17] = [
    (Function::Sin, "sin"),
    (Function::Cos, "cos"),
    (Function::Tan, "tan"),
    (Function::Cot, "cot"),
    (Function::Sec, "sec"),
    (Function::Csc, "csc"),
    (Function::Exp, "exp"),
    (Function::Log, "log"),
    (Function::Sinh, "sinh"),
    (Function::Cosh, "cosh"),
    (Function::Tanh, "tanh"),
    (Function::Asin, "asin"),
    (Function::Acos, "acos"),
    (Function::Atan, "atan"),
    (Function::Asinh, "asinh"),
    (Function::Acosh, "acosh"),
    (Function::Atanh, "atanh"),
];

fn name_of(e: &Basic) -> Option<&'static str> {
    match e.head() {
        Head::Function(f) => NAMES.iter().find(|(g, _)| g == f).map(|(_, n)| *n),
        _ => None,
    }
}

/// The derivative of the elementary function `name` at `u` (`_table`).
fn table(name: &str, u: &Basic) -> PolyResult<Option<Basic>> {
    use Function as F;
    let one = Basic::one();
    let square = |f: Function| -> PolyResult<Basic> { pow(&call(f, u)?, &int(2)) };
    let u2 = || pow(u, &int(2));
    Ok(Some(match name {
        "sin" => call(F::Cos, u)?,
        "cos" => neg(&call(F::Sin, u)?)?,
        "tan" => div(&one, &square(F::Cos)?)?,
        "cot" => div(&int(-1), &square(F::Sin)?)?,
        "sec" => div(&call(F::Sin, u)?, &square(F::Cos)?)?,
        "csc" => div(&neg(&call(F::Cos, u)?)?, &square(F::Sin)?)?,
        "exp" => call(F::Exp, u)?,
        "log" => div(&one, u)?,
        "sinh" => call(F::Cosh, u)?,
        "cosh" => call(F::Sinh, u)?,
        "tanh" => div(&one, &square(F::Cosh)?)?,
        "asin" => div(&one, &sqrt(&sub(&one, &u2()?)?)?)?,
        "acos" => div(&int(-1), &sqrt(&sub(&one, &u2()?)?)?)?,
        "atan" => div(&one, &add(&[one.clone(), u2()?])?)?,
        "asinh" => div(&one, &sqrt(&add(&[u2()?, one.clone()])?)?)?,
        "acosh" => div(&one, &sqrt(&sub(&u2()?, &one)?)?)?,
        "atanh" => div(&one, &sub(&one, &u2()?)?)?,
        _ => return Ok(None),
    }))
}

impl DerivativeStep {
    fn head(&self) -> (&Basic, &Basic) {
        match self {
            D::ConstantRule { expression, variable }
            | D::VariableRule { expression, variable }
            | D::PowerRule {
                expression, variable, ..
            }
            | D::ExponentialRule {
                expression, variable, ..
            }
            | D::FunctionRule {
                expression, variable, ..
            }
            | D::ConstantMultipleRule {
                expression, variable, ..
            }
            | D::SumRule {
                expression, variable, ..
            }
            | D::ProductRule {
                expression, variable, ..
            }
            | D::QuotientRule {
                expression, variable, ..
            }
            | D::ChainRule {
                expression, variable, ..
            }
            | D::RewriteRule {
                expression, variable, ..
            }
            | D::AlternativeRule {
                expression, variable, ..
            }
            | D::DontKnowRule { expression, variable }
            | D::InProgressRule { expression, variable } => (expression, variable),
        }
    }

    /// The differentiated expression.
    pub fn expression(&self) -> &Basic {
        self.head().0
    }

    /// The variable of differentiation.
    pub fn variable(&self) -> &Basic {
        self.head().1
    }

    /// The derivative, assembled as the rule says (`result`).
    pub fn result(&self) -> PolyResult<Basic> {
        match self {
            D::ConstantRule { .. } => Ok(Basic::zero()),
            D::VariableRule { .. } => Ok(Basic::one()),
            D::PowerRule { variable, exponent, .. } => m2(exponent, &pow(variable, &sub(exponent, &Basic::one())?)?),
            D::ExponentialRule { expression, base, .. } => {
                m2(expression, &crate::functions::elementary::exponential::log(base)?)
            }
            D::FunctionRule { variable, function, .. } => table(function, variable)?
                .ok_or_else(|| crate::error::value_error(format!("no derivative in the table for {}", function))),
            D::ConstantMultipleRule { constant, substep, .. } => m2(constant, &substep.result()?),
            D::SumRule { terms, .. } => {
                let results: Vec<Basic> = terms.iter().map(|t| t.result()).collect::<PolyResult<_>>()?;
                add(&results)
            }
            D::ProductRule {
                first,
                second,
                first_step,
                second_step,
                ..
            } => add(&[m2(&first_step.result()?, second)?, m2(first, &second_step.result()?)?]),
            D::QuotientRule {
                numerator,
                denominator,
                numerator_step,
                denominator_step,
                ..
            } => {
                let top: Basic = sub(
                    &m2(&numerator_step.result()?, denominator)?,
                    &m2(numerator, &denominator_step.result()?)?,
                )?;
                div(&top, &pow(denominator, &int(2))?)
            }
            D::ChainRule {
                inner,
                placeholder,
                outer_step,
                inner_step,
                ..
            } => {
                let outer: Basic = crate::core::subs::subs(&outer_step.result()?, placeholder, inner)?;
                m2(&outer, &inner_step.result()?)
            }
            D::RewriteRule { substep, .. } => substep.result(),
            D::AlternativeRule { alternatives, .. } => alternatives[0].result(),
            D::DontKnowRule { expression, variable } | D::InProgressRule { expression, variable } => Ok(
                crate::core::function::derivative_node(expression, &[(variable.clone(), 1)]),
            ),
        }
    }
}

impl Step for DerivativeStep {
    fn substeps(&self) -> Vec<DerivativeStep> {
        match self {
            D::ConstantMultipleRule { substep, .. } | D::RewriteRule { substep, .. } => vec![(**substep).clone()],
            D::SumRule { terms, .. } => terms.clone(),
            D::ProductRule {
                first_step,
                second_step,
                ..
            } => vec![(**first_step).clone(), (**second_step).clone()],
            D::QuotientRule {
                numerator_step,
                denominator_step,
                ..
            } => {
                vec![(**numerator_step).clone(), (**denominator_step).clone()]
            }
            D::ChainRule {
                outer_step, inner_step, ..
            } => vec![(**outer_step).clone(), (**inner_step).clone()],
            D::AlternativeRule { alternatives, .. } => alternatives.clone(),
            _ => Vec::new(),
        }
    }

    fn rule(&self) -> String {
        match self {
            D::ConstantRule { .. } => "ConstantRule",
            D::VariableRule { .. } => "VariableRule",
            D::PowerRule { .. } => "PowerRule",
            D::ExponentialRule { .. } => "ExponentialRule",
            D::FunctionRule { .. } => "FunctionRule",
            D::ConstantMultipleRule { .. } => "ConstantMultipleRule",
            D::SumRule { .. } => "SumRule",
            D::ProductRule { .. } => "ProductRule",
            D::QuotientRule { .. } => "QuotientRule",
            D::ChainRule { .. } => "ChainRule",
            D::RewriteRule { .. } => "RewriteRule",
            D::AlternativeRule { .. } => "AlternativeRule",
            D::DontKnowRule { .. } => "DontKnowRule",
            D::InProgressRule { .. } => "InProgressRule",
        }
        .to_string()
    }

    fn statement(&self) -> String {
        let result: String = match self.result() {
            Ok(r) => r.to_string(),
            Err(e) => format!("<{}>", e),
        };
        format!("d/d{}({}) = {}", self.variable(), self.expression(), result)
    }

    fn minor(&self) -> bool {
        matches!(self, D::RewriteRule { .. })
    }

    fn is_alternative(&self) -> bool {
        matches!(self, D::AlternativeRule { .. })
    }

    fn is_dont_know(&self) -> bool {
        matches!(self, D::DontKnowRule { .. } | D::InProgressRule { .. })
    }
}

type Solver = StepSolver<DerivativeProblem, DerivativeStep>;
type Out = Option<DerivativeStep>;

fn problem(expression: Basic, variable: &Basic) -> DerivativeProblem {
    DerivativeProblem {
        expression,
        variable: variable.clone(),
    }
}

// the rules: they call the solver where Python's generators yield

fn constant_rule(p: &DerivativeProblem, _: &Context, _: &mut Solver) -> Out {
    Some(D::ConstantRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
    })
}

fn variable_rule(p: &DerivativeProblem, _: &Context, _: &mut Solver) -> Out {
    Some(D::VariableRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
    })
}

fn power_rule(p: &DerivativeProblem, _: &Context, _: &mut Solver) -> Out {
    p.expression.is_pow().then(|| D::PowerRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        exponent: p.expression.args()[1].clone(),
    })
}

fn exponential_rule(p: &DerivativeProblem, _: &Context, _: &mut Solver) -> Out {
    p.expression.is_pow().then(|| D::ExponentialRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        base: p.expression.args()[0].clone(),
    })
}

fn function_rule(p: &DerivativeProblem, _: &Context, _: &mut Solver) -> Out {
    name_of(&p.expression).map(|function| D::FunctionRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        function,
    })
}

fn constant_multiple_rule(p: &DerivativeProblem, _: &Context, s: &mut Solver) -> Out {
    let (constant, rest) = as_independent(&p.expression, &p.variable, p.expression.is_add()).ok()?;
    let substep = s.solve(&problem(rest.clone(), &p.variable));
    Some(D::ConstantMultipleRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        constant,
        rest,
        substep: Box::new(substep),
    })
}

fn sum_rule(p: &DerivativeProblem, _: &Context, s: &mut Solver) -> Out {
    let mut terms: Vec<DerivativeStep> = Vec::new();
    for term in p.expression.args() {
        terms.push(s.solve(&problem(term.clone(), &p.variable)));
    }
    Some(D::SumRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        terms,
    })
}

fn product_rule(p: &DerivativeProblem, _: &Context, s: &mut Solver) -> Out {
    let factors: &[Basic] = p.expression.args();
    let (first, second) = (factors[0].clone(), mul(&factors[1..]).ok()?);
    let first_step = s.solve(&problem(first.clone(), &p.variable));
    let second_step = s.solve(&problem(second.clone(), &p.variable));
    Some(D::ProductRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        first,
        second,
        first_step: Box::new(first_step),
        second_step: Box::new(second_step),
    })
}

fn quotient_rule(p: &DerivativeProblem, _: &Context, s: &mut Solver) -> Out {
    let (numerator, denominator) = as_numer_denom(&p.expression).ok()?;
    let numerator_step = s.solve(&problem(numerator.clone(), &p.variable));
    let denominator_step = s.solve(&problem(denominator.clone(), &p.variable));
    Some(D::QuotientRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        numerator,
        denominator,
        numerator_step: Box::new(numerator_step),
        denominator_step: Box::new(denominator_step),
    })
}

fn quotient_as_product_rule(p: &DerivativeProblem, _: &Context, s: &mut Solver) -> Out {
    let (numerator, denominator) = as_numer_denom(&p.expression).ok()?;
    let reciprocal: Basic = pow(&denominator, &int(-1)).ok()?;
    let first_step = s.solve(&problem(numerator.clone(), &p.variable));
    let second_step = s.solve(&problem(reciprocal.clone(), &p.variable));
    Some(D::ProductRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        first: numerator,
        second: reciprocal,
        first_step: Box::new(first_step),
        second_step: Box::new(second_step),
    })
}

/// A symbol for the inner function: `u`, or `u1`, ... when the name is
/// taken (`_placeholder`).
fn placeholder(expression: &Basic, variable: &Basic) -> Basic {
    // the names of the symbols (dummies and wilds are symbols in Python)
    let name = |s: &Basic| -> Option<String> {
        match s.head() {
            Head::Symbol(d) | Head::Dummy(d, _) => Some(d.name.clone()),
            Head::Wild(w) => Some(w.symbol.name.clone()),
            _ => None,
        }
    };
    let mut used: Vec<String> = expression.free_symbols().iter().filter_map(name).collect();
    if let Some(n) = name(variable) {
        used.push(n);
    }
    let (mut name, mut k) = ("u".to_string(), 0);
    while used.contains(&name) {
        k += 1;
        name = format!("u{}", k);
    }
    crate::core::Symbol::new(&name, &[]).0
}

fn chain_rule(p: &DerivativeProblem, _: &Context, s: &mut Solver) -> Out {
    let (expression, x) = (&p.expression, &p.variable);
    let u: Basic = placeholder(expression, x);
    let (inner, outer): (Basic, Basic) = if expression.is_pow() {
        let (base, exponent) = (&expression.args()[0], &expression.args()[1]);
        if !exponent.has(x) {
            (base.clone(), pow(&u, exponent).ok()?)
        } else if !base.has(x) {
            (exponent.clone(), pow(base, &u).ok()?)
        } else {
            return None;
        }
    } else if name_of(expression).is_some() && expression.args().len() == 1 {
        let Head::Function(f) = expression.head() else {
            return None;
        };
        // the function at the placeholder, unevaluated
        (
            expression.args()[0].clone(),
            Basic::new(Head::Function(*f), vec![u.clone()]),
        )
    } else {
        return None;
    };
    let outer_step = s.solve(&problem(outer, &u));
    let inner_step = s.solve(&problem(inner.clone(), x));
    Some(D::ChainRule {
        expression: expression.clone(),
        variable: x.clone(),
        inner,
        placeholder: u,
        outer_step: Box::new(outer_step),
        inner_step: Box::new(inner_step),
    })
}

fn general_power_rule(p: &DerivativeProblem, _: &Context, s: &mut Solver) -> Out {
    use crate::functions::elementary::exponential::{exp, log};
    if !p.expression.is_pow() {
        return None;
    }
    let (base, exponent) = (&p.expression.args()[0], &p.expression.args()[1]);
    let rewritten: Basic = exp(&m2(exponent, &log(base).ok()?).ok()?).ok()?;
    let substep = s.solve(&problem(rewritten.clone(), &p.variable));
    Some(D::RewriteRule {
        expression: p.expression.clone(),
        variable: p.variable.clone(),
        rewritten,
        substep: Box::new(substep),
    })
}

type R = Rule<DerivativeProblem, DerivativeStep>;

fn candidates(p: &DerivativeProblem) -> Vec<R> {
    let (expression, x) = (&p.expression, &p.variable);
    if !expression.has(x) {
        return vec![constant_rule];
    }
    if expression == x {
        return vec![variable_rule];
    }
    if expression.is_add() {
        return vec![sum_rule];
    }
    if expression.is_mul() {
        let constant: Option<Basic> = as_independent(expression, x, false).ok().map(|(c, _)| c);
        if constant.is_some_and(|c| c != Basic::one()) {
            return vec![constant_multiple_rule];
        }
        let denominator: Option<Basic> = as_numer_denom(expression).ok().map(|(_, d)| d);
        if denominator.is_some_and(|d| d != Basic::one()) {
            return vec![quotient_rule, quotient_as_product_rule];
        }
        return vec![product_rule];
    }
    if expression.is_pow() {
        let (base, exponent) = (&expression.args()[0], &expression.args()[1]);
        if base == x && !exponent.has(x) {
            return vec![power_rule];
        }
        if exponent == x && !base.has(x) {
            return vec![exponential_rule];
        }
        if base.has(x) && exponent.has(x) {
            return vec![general_power_rule];
        }
        return vec![chain_rule];
    }
    if name_of(expression).is_some() && expression.args().len() == 1 {
        return if expression.args()[0] == *x {
            vec![function_rule]
        } else {
            vec![chain_rule]
        };
    }
    Vec::new()
}

/// The derivation of the derivative of `f` with respect to `x`, as a tree
/// of rules; with `branch` every workable rule at each level, in
/// `AlternativeRule` nodes; `max_depth` limits the depth
/// (`derivative_steps`).
pub fn derivative_steps(f: &Basic, x: &Basic, branch: bool, max_depth: Option<usize>) -> DerivativeStep {
    let mut solver: Solver = StepSolver::new(
        candidates,
        |p| D::DontKnowRule {
            expression: p.expression.clone(),
            variable: p.variable.clone(),
        },
        |p| D::InProgressRule {
            expression: p.expression.clone(),
            variable: p.variable.clone(),
        },
        |p, alternatives| D::AlternativeRule {
            expression: p.expression.clone(),
            variable: p.variable.clone(),
            alternatives,
        },
        branch,
        max_depth,
    );
    solver.solve(&problem(f.clone(), x))
}

/// The derivative found by the rules, as they give it (not simplified);
/// a `Derivative` is left where no rule applies (`manual_derivative`).
pub fn manual_derivative(f: &Basic, x: &Basic) -> PolyResult<Basic> {
    derivative_steps(f, x, false, Some(100)).result()
}
