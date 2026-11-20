//! Port of `sympy_extras/integrals/manualintegrate.py`: integration by
//! the rules a student applies by hand, with the steps.
//!
//! The Python rules are generators which yield the integrals they need and
//! receive their steps back from `IntegralSolver.run`; the driver answers
//! every request at once, so here a rule calls [`IntegralSolver::solve`]
//! where Python writes `yield from _steps(...)`: the same integrals are
//! solved in the same order, with the same cache and cycle cut. An error
//! raised while a requested integral is solved leaves Python's driver
//! (`run`) and so passes every `_guarded` rule up to the top: the solver
//! records it (`escaping`) and the guards let it through.
//!
//! Python's binary operators evaluate at each step (`a*b*c` is
//! `(a*b)*c`): the expressions are built with the binary helpers in the
//! same association.

#![allow(clippy::mutable_key_type)]
#![allow(clippy::too_many_arguments)]
// the branches of the Python LIATE choice are kept apart
#![allow(clippy::if_same_then_else)]

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::numbers::Int;

type R = PolyResult<Basic>;

static COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------
// expressions with Python's binary operators

pub(crate) fn int(n: i64) -> Basic {
    Basic::int(n)
}
pub(crate) fn half() -> Basic {
    Basic::half()
}
pub(crate) fn rational(p: i64, q: i64) -> Basic {
    Basic::from_rational(crate::numbers::rat(p, q))
}
/// `a + b`.
pub(crate) fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
/// `a - b`.
pub(crate) fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
/// `a*b`.
pub(crate) fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
/// `a/b`.
pub(crate) fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
/// `a**b`.
pub(crate) fn p2(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
/// `-a`.
pub(crate) fn ng(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
/// `add(*v)`.
pub(crate) fn add_n(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
/// `mul(*v)`.
pub(crate) fn mul_n(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
pub(crate) fn sqrt(a: &Basic) -> R {
    p2(a, &half())
}
pub(crate) fn apply(f: Function, args: &[Basic]) -> R {
    crate::functions::apply_function(f, args)
}
pub(crate) fn f1(f: Function, a: &Basic) -> R {
    apply(f, std::slice::from_ref(a))
}
pub(crate) fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
pub(crate) fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
pub(crate) fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
pub(crate) fn subs_list(e: &Basic, pairs: &[(Basic, Basic)]) -> R {
    crate::core::subs::subs_pairs(e, pairs)
}
pub(crate) fn subs_map(e: &Basic, pairs: &[(Basic, Basic)]) -> R {
    crate::core::subs::subs_mapping(e, pairs)
}
pub(crate) fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
pub(crate) fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1)])
}
pub(crate) fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
pub(crate) fn piecewise(pairs: &[(Basic, Basic)]) -> R {
    crate::functions::elementary::piecewise::piecewise(pairs)
}
pub(crate) fn true_() -> Basic {
    crate::core::logic::true_()
}
pub(crate) fn ne(a: &Basic, b: &Basic) -> R {
    crate::core::relational::ne(a, b)
}
pub(crate) fn eq(a: &Basic, b: &Basic) -> R {
    crate::core::relational::eq(a, b)
}
pub(crate) fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
pub(crate) fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
pub(crate) fn and2(a: &Basic, b: &Basic) -> Basic {
    crate::core::logic::and_(&[a.clone(), b.clone()])
}

pub(crate) fn is_fn(e: &Basic, fs: &[Function]) -> bool {
    matches!(e.head(), Head::Function(f) if fs.contains(f))
}

pub(crate) fn is_symbol(e: &Basic) -> bool {
    matches!(e.head(), Head::Symbol(_) | Head::Dummy(_, _))
}

pub(crate) fn is_integer_node(e: &Basic) -> bool {
    e.as_rational().is_some_and(|q| q.is_int())
}

pub(crate) fn int_value(e: &Basic) -> Option<i64> {
    e.as_rational()
        .filter(|q| q.is_int())
        .and_then(|q| i64::try_from(crate::numbers::numer(q)).ok())
}

/// A fresh dummy (`_fresh`).
pub(crate) fn fresh(name: &str) -> Basic {
    Dummy::new(
        &format!("{}_manual_{}", name, COUNTER.fetch_add(1, Ordering::Relaxed)),
        &[],
    )
    .0
}

pub(crate) fn is_true_node(b: &Basic) -> bool {
    *b == crate::core::logic::true_()
}

pub(crate) fn is_false_node(b: &Basic) -> bool {
    *b == crate::core::logic::false_()
}

/// Whether `e` is zero, by `simplify` when the assumptions do not say
/// (`_zero`).
pub(crate) fn zero(e: &Basic) -> PolyResult<bool> {
    if e.is_number_node() {
        return Ok(*e == Basic::zero());
    }
    if is(e, Predicate::zero) == Some(true) {
        return Ok(true);
    }
    let options = crate::simplify::simplify::Options::default();
    Ok(crate::simplify::simplify::simplify(e, &options)? == Basic::zero())
}

/// Whether `a == b` up to `cancel` (`_same`).
pub(crate) fn same(a: &Basic, b: &Basic) -> PolyResult<bool> {
    if a == b {
        return Ok(true);
    }
    match cancel(&s2(a, b)?) {
        Ok(v) => Ok(v == Basic::zero()),
        Err(PolyError::NotImplemented(m)) if m.starts_with(crate::error::NOT_PORTED_PREFIX) => {
            Err(PolyError::NotImplemented(m))
        }
        Err(PolyError::NotImplemented(_) | PolyError::Value(_) | PolyError::Type(_) | PolyError::ZeroDivision(_)) => {
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

pub(crate) fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

pub(crate) fn add_args(e: &Basic) -> Vec<Basic> {
    crate::core::expr::terms_of(e)
}

pub(crate) fn as_base_exp(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_base_exp(e)
}

pub(crate) fn as_independent(e: &Basic, x: &Basic, as_add: bool) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_independent(e, x, as_add)
}

/// `as_independent` with SymPy's default `as_Add` (that of the class).
pub(crate) fn as_independent_default(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    as_independent(e, x, e.is_add())
}

pub(crate) fn as_numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}

pub(crate) fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}

pub(crate) fn is_rational_function_in(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Rational) == Some(true)
}

pub(crate) fn is_algebraic_expr(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Algebraic)
        == Some(true)
}

// ---------------------------------------------------------------------
// the rules

pub type RuleRef = Rc<Rule>;

/// A step: the integrand, the variable and what the rule did (`Rule`).
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub integrand: Basic,
    pub variable: Basic,
    pub kind: Kind,
}

/// The rules (the dataclasses of the Python module).
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Constant,
    ConstantTimes {
        constant: Basic,
        other: Basic,
        substep: RuleRef,
    },
    Power {
        base: Basic,
        exp: Basic,
    },
    NestedPow {
        base: Basic,
        exp: Basic,
    },
    Add {
        substeps: Vec<RuleRef>,
    },
    U {
        u_var: Basic,
        u_func: Basic,
        substep: RuleRef,
    },
    Parts {
        u: Basic,
        dv: Basic,
        v_step: RuleRef,
        second_step: Option<RuleRef>,
    },
    CyclicParts {
        parts_rules: Vec<RuleRef>,
        coefficient: Basic,
    },
    Sin,
    Cos,
    SecTan,
    CscCot,
    Sec2,
    Csc2,
    Sinh,
    Cosh,
    Exp {
        base: Basic,
        exp: Basic,
    },
    Reciprocal {
        base: Basic,
    },
    Arcsin,
    Arcsinh,
    ReciprocalSqrtQuadratic {
        a: Basic,
        b: Basic,
        c: Basic,
    },
    SqrtQuadraticDenom {
        a: Basic,
        b: Basic,
        c: Basic,
        coeffs: Vec<Basic>,
    },
    SqrtQuadratic {
        a: Basic,
        b: Basic,
        c: Basic,
    },
    Alternative {
        alternatives: Vec<RuleRef>,
    },
    DontKnow,
    Derivative,
    Rewrite {
        rewritten: Basic,
        substep: RuleRef,
    },
    CompleteSquare {
        rewritten: Basic,
        substep: RuleRef,
    },
    Piecewise {
        subfunctions: Vec<(RuleRef, Basic)>,
    },
    Heaviside {
        harg: Basic,
        ibnd: Basic,
        substep: RuleRef,
    },
    DiracDelta {
        n: Basic,
        a: Basic,
        b: Basic,
    },
    TrigSubstitution {
        theta: Basic,
        func: Basic,
        rewritten: Basic,
        substep: RuleRef,
        restriction: Basic,
    },
    Arctan {
        a: Basic,
        b: Basic,
        c: Basic,
    },
    Jacobi {
        n: Basic,
        a: Basic,
        b: Basic,
    },
    Gegenbauer {
        n: Basic,
        a: Basic,
    },
    ChebyshevT {
        n: Basic,
    },
    ChebyshevU {
        n: Basic,
    },
    Legendre {
        n: Basic,
    },
    Hermite {
        n: Basic,
    },
    Laguerre {
        n: Basic,
    },
    AssocLaguerre {
        n: Basic,
        a: Basic,
    },
    Ci {
        a: Basic,
        b: Basic,
    },
    Chi {
        a: Basic,
        b: Basic,
    },
    Ei {
        a: Basic,
        b: Basic,
    },
    Si {
        a: Basic,
        b: Basic,
    },
    Shi {
        a: Basic,
        b: Basic,
    },
    Li {
        a: Basic,
        b: Basic,
    },
    Erf {
        a: Basic,
        b: Basic,
        c: Basic,
    },
    FresnelC {
        a: Basic,
        b: Basic,
        c: Basic,
    },
    FresnelS {
        a: Basic,
        b: Basic,
        c: Basic,
    },
    Polylog {
        a: Basic,
        b: Basic,
    },
    UpperGamma {
        a: Basic,
        e: Basic,
    },
    EllipticF {
        a: Basic,
        d: Basic,
    },
    EllipticE {
        a: Basic,
        d: Basic,
    },
}

pub(crate) fn rule(integrand: &Basic, variable: &Basic, kind: Kind) -> RuleRef {
    Rc::new(Rule {
        integrand: integrand.clone(),
        variable: variable.clone(),
        kind,
    })
}

impl Rule {
    /// The name of the Python class of the rule.
    pub fn name(&self) -> &'static str {
        match &self.kind {
            Kind::Constant => "ConstantRule",
            Kind::ConstantTimes { .. } => "ConstantTimesRule",
            Kind::Power { .. } => "PowerRule",
            Kind::NestedPow { .. } => "NestedPowRule",
            Kind::Add { .. } => "AddRule",
            Kind::U { .. } => "URule",
            Kind::Parts { .. } => "PartsRule",
            Kind::CyclicParts { .. } => "CyclicPartsRule",
            Kind::Sin => "SinRule",
            Kind::Cos => "CosRule",
            Kind::SecTan => "SecTanRule",
            Kind::CscCot => "CscCotRule",
            Kind::Sec2 => "Sec2Rule",
            Kind::Csc2 => "Csc2Rule",
            Kind::Sinh => "SinhRule",
            Kind::Cosh => "CoshRule",
            Kind::Exp { .. } => "ExpRule",
            Kind::Reciprocal { .. } => "ReciprocalRule",
            Kind::Arcsin => "ArcsinRule",
            Kind::Arcsinh => "ArcsinhRule",
            Kind::ReciprocalSqrtQuadratic { .. } => "ReciprocalSqrtQuadraticRule",
            Kind::SqrtQuadraticDenom { .. } => "SqrtQuadraticDenomRule",
            Kind::SqrtQuadratic { .. } => "SqrtQuadraticRule",
            Kind::Alternative { .. } => "AlternativeRule",
            Kind::DontKnow => "DontKnowRule",
            Kind::Derivative => "DerivativeRule",
            Kind::Rewrite { .. } => "RewriteRule",
            Kind::CompleteSquare { .. } => "CompleteSquareRule",
            Kind::Piecewise { .. } => "PiecewiseRule",
            Kind::Heaviside { .. } => "HeavisideRule",
            Kind::DiracDelta { .. } => "DiracDeltaRule",
            Kind::TrigSubstitution { .. } => "TrigSubstitutionRule",
            Kind::Arctan { .. } => "ArctanRule",
            Kind::Jacobi { .. } => "JacobiRule",
            Kind::Gegenbauer { .. } => "GegenbauerRule",
            Kind::ChebyshevT { .. } => "ChebyshevTRule",
            Kind::ChebyshevU { .. } => "ChebyshevURule",
            Kind::Legendre { .. } => "LegendreRule",
            Kind::Hermite { .. } => "HermiteRule",
            Kind::Laguerre { .. } => "LaguerreRule",
            Kind::AssocLaguerre { .. } => "AssocLaguerreRule",
            Kind::Ci { .. } => "CiRule",
            Kind::Chi { .. } => "ChiRule",
            Kind::Ei { .. } => "EiRule",
            Kind::Si { .. } => "SiRule",
            Kind::Shi { .. } => "ShiRule",
            Kind::Li { .. } => "LiRule",
            Kind::Erf { .. } => "ErfRule",
            Kind::FresnelC { .. } => "FresnelCRule",
            Kind::FresnelS { .. } => "FresnelSRule",
            Kind::Polylog { .. } => "PolylogRule",
            Kind::UpperGamma { .. } => "UpperGammaRule",
            Kind::EllipticF { .. } => "EllipticFRule",
            Kind::EllipticE { .. } => "EllipticERule",
        }
    }

    /// The direct substeps, in the order of the Python fields.
    pub fn substeps(&self) -> Vec<RuleRef> {
        match &self.kind {
            Kind::ConstantTimes { substep, .. }
            | Kind::U { substep, .. }
            | Kind::Rewrite { substep, .. }
            | Kind::CompleteSquare { substep, .. }
            | Kind::Heaviside { substep, .. }
            | Kind::TrigSubstitution { substep, .. } => vec![substep.clone()],
            Kind::Add { substeps } => substeps.clone(),
            Kind::Parts {
                v_step, second_step, ..
            } => {
                let mut v: Vec<RuleRef> = vec![v_step.clone()];
                v.extend(second_step.iter().cloned());
                v
            }
            Kind::CyclicParts { parts_rules, .. } => parts_rules.clone(),
            Kind::Alternative { alternatives } => alternatives.clone(),
            Kind::Piecewise { subfunctions } => subfunctions.iter().map(|(r, _)| r.clone()).collect(),
            _ => Vec::new(),
        }
    }

    /// Whether a step of the tree is a `DontKnowRule`
    /// (`contains_dont_know`).
    pub fn contains_dont_know(&self) -> bool {
        match &self.kind {
            Kind::DontKnow => true,
            _ => self.substeps().iter().any(|s| s.contains_dont_know()),
        }
    }

    /// The antiderivative of the steps (`eval`).
    pub fn eval(&self) -> R {
        let x: &Basic = &self.variable;
        match &self.kind {
            Kind::Constant => m2(&self.integrand, x),
            Kind::ConstantTimes { constant, substep, .. } => m2(constant, &substep.eval()?),
            Kind::Power { base, exp: e } => {
                let e1: Basic = a2(e, &int(1))?;
                piecewise(&[(d2(&p2(base, &e1)?, &e1)?, ne(e, &int(-1))?), (log(base)?, true_())])
            }
            Kind::NestedPow { base, exp: e } => {
                let m: Basic = m2(base, &self.integrand)?;
                piecewise(&[
                    (d2(&m, &a2(e, &int(1))?)?, ne(e, &int(-1))?),
                    (m2(&m, &log(base)?)?, true_()),
                ])
            }
            Kind::Add { substeps } => {
                let values: Vec<Basic> = substeps.iter().map(|s| s.eval()).collect::<PolyResult<_>>()?;
                add_n(&values)
            }
            Kind::U { u_var, u_func, substep } => {
                let mut result: Basic = substep.eval()?;
                if u_func.is_pow() {
                    let (base, e) = as_base_exp(u_func)?;
                    if e == int(-1) {
                        result = subs(&result, &log(u_var)?, &ng(&log(&base)?)?)?;
                    }
                }
                subs(&result, u_var, u_func)
            }
            Kind::Parts {
                u, v_step, second_step, ..
            } => {
                let Some(second) = second_step else {
                    return Err(PolyError::Value(
                        "a part of a cyclic integration by parts has no value of its own".to_string(),
                    ));
                };
                let v: Basic = v_step.eval()?;
                s2(&m2(u, &v)?, &second.eval()?)
            }
            Kind::CyclicParts {
                parts_rules,
                coefficient,
            } => {
                let mut result: Vec<Basic> = Vec::new();
                let mut sign: i64 = 1;
                for r in parts_rules {
                    let Kind::Parts { u, v_step, .. } = &r.kind else {
                        return Err(PolyError::Type("a cyclic integration by parts of parts".to_string()));
                    };
                    result.push(m2(&m2(&int(sign), u)?, &v_step.eval()?)?);
                    sign = -sign;
                }
                d2(&add_n(&result)?, &s2(&int(1), coefficient)?)
            }
            Kind::Sin => ng(&f1(Function::Cos, x)?),
            Kind::Cos => f1(Function::Sin, x),
            Kind::SecTan => f1(Function::Sec, x),
            Kind::CscCot => ng(&f1(Function::Csc, x)?),
            Kind::Sec2 => f1(Function::Tan, x),
            Kind::Csc2 => ng(&f1(Function::Cot, x)?),
            Kind::Sinh => f1(Function::Cosh, x),
            Kind::Cosh => f1(Function::Sinh, x),
            Kind::Exp { base, .. } => d2(&self.integrand, &log(base)?),
            Kind::Reciprocal { base } => log(base),
            Kind::Arcsin => f1(Function::Asin, x),
            Kind::Arcsinh => f1(Function::Asinh, x),
            Kind::ReciprocalSqrtQuadratic { a, b, c } => {
                let q: Basic = a2(&a2(a, &m2(b, x)?)?, &m2(c, &p2(x, &int(2))?)?)?;
                let inner: Basic = a2(
                    &a2(&m2(&m2(&int(2), &sqrt(c)?)?, &sqrt(&q)?)?, b)?,
                    &m2(&m2(&int(2), c)?, x)?,
                )?;
                d2(&log(&inner)?, &sqrt(c)?)
            }
            Kind::SqrtQuadraticDenom { a, b, c, coeffs } => sqrt_quadratic_denom_eval(a, b, c, coeffs, x),
            Kind::SqrtQuadratic { .. } => {
                let Some(step) = sqrt_quadratic_generic(&self.integrand, x)? else {
                    return Err(PolyError::Value(format!("no rule for {}", self.integrand)));
                };
                step.eval()
            }
            Kind::Alternative { alternatives } => alternatives[0].eval(),
            Kind::DontKnow => crate::integrals::integral::integral(&self.integrand, x, None),
            Kind::Derivative => derivative_eval(&self.integrand, x),
            Kind::Rewrite { substep, .. } | Kind::CompleteSquare { substep, .. } => substep.eval(),
            Kind::Piecewise { subfunctions } => {
                let mut pairs: Vec<(Basic, Basic)> = Vec::new();
                for (s, cond) in subfunctions {
                    pairs.push((s.eval()?, cond.clone()));
                }
                piecewise(&pairs)
            }
            Kind::Heaviside { harg, ibnd, substep } => {
                let result: Basic = substep.eval()?;
                let h: Basic = crate::functions::special::delta_functions::heaviside(harg, None)?;
                m2(&h, &s2(&result, &subs(&result, x, ibnd)?)?)
            }
            Kind::DiracDelta { n, a, b } => {
                let arg: Basic = a2(a, &m2(b, x)?)?;
                if *n == Basic::zero() {
                    return d2(&crate::functions::special::delta_functions::heaviside(&arg, None)?, b);
                }
                d2(
                    &crate::functions::special::delta_functions::dirac_delta(&arg, Some(&s2(n, &int(1))?))?,
                    b,
                )
            }
            Kind::TrigSubstitution {
                theta,
                func,
                substep,
                restriction,
                ..
            } => trig_substitution_eval(theta, func, substep, restriction, x),
            Kind::Arctan { a, b, c } => {
                let r: Basic = sqrt(&d2(c, b)?)?;
                m2(&d2(&d2(a, b)?, &r)?, &f1(Function::Atan, &d2(x, &r)?)?)
            }
            Kind::Jacobi { n, a, b } => {
                let nab: Basic = a2(&a2(n, a)?, b)?;
                let first: Basic = d2(
                    &m2(
                        &int(2),
                        &apply(
                            Function::Jacobi,
                            &[a2(n, &int(1))?, s2(a, &int(1))?, s2(b, &int(1))?, x.clone()],
                        )?,
                    )?,
                    &nab,
                )?;
                let third: Basic = a2(
                    &d2(&m2(&a2(&a2(a, b)?, &int(2))?, &p2(x, &int(2))?)?, &int(4))?,
                    &d2(&m2(&s2(a, b)?, x)?, &int(2))?,
                )?;
                piecewise(&[
                    (first, ne(&nab, &int(0))?),
                    (x.clone(), eq(n, &int(0))?),
                    (third, eq(n, &int(1))?),
                ])
            }
            Kind::Gegenbauer { n, a } => {
                let first: Basic = d2(
                    &apply(Function::Gegenbauer, &[a2(n, &int(1))?, s2(a, &int(1))?, x.clone()])?,
                    &m2(&int(2), &s2(a, &int(1))?)?,
                )?;
                let second: Basic = d2(
                    &apply(Function::ChebyshevT, &[a2(n, &int(1))?, x.clone()])?,
                    &a2(n, &int(1))?,
                )?;
                piecewise(&[
                    (first, ne(a, &int(1))?),
                    (second, ne(n, &int(-1))?),
                    (Basic::zero(), true_()),
                ])
            }
            Kind::ChebyshevT { n } => {
                let first: Basic = d2(
                    &s2(
                        &d2(
                            &apply(Function::ChebyshevT, &[a2(n, &int(1))?, x.clone()])?,
                            &a2(n, &int(1))?,
                        )?,
                        &d2(
                            &apply(Function::ChebyshevT, &[s2(n, &int(1))?, x.clone()])?,
                            &s2(n, &int(1))?,
                        )?,
                    )?,
                    &int(2),
                )?;
                let abs_n: Basic = f1(Function::Abs, n)?;
                piecewise(&[(first, ne(&abs_n, &int(1))?), (d2(&p2(x, &int(2))?, &int(2))?, true_())])
            }
            Kind::ChebyshevU { n } => {
                let first: Basic = d2(
                    &apply(Function::ChebyshevT, &[a2(n, &int(1))?, x.clone()])?,
                    &a2(n, &int(1))?,
                )?;
                piecewise(&[(first, ne(n, &int(-1))?), (Basic::zero(), true_())])
            }
            Kind::Legendre { n } => d2(
                &s2(
                    &apply(Function::Legendre, &[a2(n, &int(1))?, x.clone()])?,
                    &apply(Function::Legendre, &[s2(n, &int(1))?, x.clone()])?,
                )?,
                &a2(&m2(&int(2), n)?, &int(1))?,
            ),
            Kind::Hermite { n } => d2(
                &apply(Function::Hermite, &[a2(n, &int(1))?, x.clone()])?,
                &m2(&int(2), &a2(n, &int(1))?)?,
            ),
            Kind::Laguerre { n } => s2(
                &apply(Function::Laguerre, &[n.clone(), x.clone()])?,
                &apply(Function::Laguerre, &[a2(n, &int(1))?, x.clone()])?,
            ),
            Kind::AssocLaguerre { n, a } => ng(&apply(
                Function::AssocLaguerre,
                &[a2(n, &int(1))?, s2(a, &int(1))?, x.clone()],
            )?),
            Kind::Ci { a, b } => {
                let ax: Basic = m2(a, x)?;
                s2(
                    &m2(&f1(Function::Cos, b)?, &f1(Function::Ci, &ax)?)?,
                    &m2(&f1(Function::Sin, b)?, &f1(Function::Si, &ax)?)?,
                )
            }
            Kind::Chi { a, b } => {
                let ax: Basic = m2(a, x)?;
                a2(
                    &m2(&f1(Function::Cosh, b)?, &f1(Function::Chi, &ax)?)?,
                    &m2(&f1(Function::Sinh, b)?, &f1(Function::Shi, &ax)?)?,
                )
            }
            Kind::Ei { a, b } => m2(&exp(b)?, &f1(Function::Ei, &m2(a, x)?)?),
            Kind::Si { a, b } => {
                let ax: Basic = m2(a, x)?;
                a2(
                    &m2(&f1(Function::Sin, b)?, &f1(Function::Ci, &ax)?)?,
                    &m2(&f1(Function::Cos, b)?, &f1(Function::Si, &ax)?)?,
                )
            }
            Kind::Shi { a, b } => {
                let ax: Basic = m2(a, x)?;
                a2(
                    &m2(&f1(Function::Sinh, b)?, &f1(Function::Chi, &ax)?)?,
                    &m2(&f1(Function::Cosh, b)?, &f1(Function::Shi, &ax)?)?,
                )
            }
            Kind::Li { a, b } => d2(&f1(Function::LogIntegral, &a2(&m2(a, x)?, b)?)?, a),
            Kind::Erf { a, b, c } => erf_eval(a, b, c, x),
            Kind::FresnelC { a, b, c } => fresnel_eval(a, b, c, x, true),
            Kind::FresnelS { a, b, c } => fresnel_eval(a, b, c, x, false),
            Kind::Polylog { a, b } => apply(Function::PolyLog, &[a2(b, &int(1))?, m2(a, x)?]),
            Kind::UpperGamma { a, e } => {
                let max: Basic = m2(&ng(a)?, x)?;
                d2(
                    &m2(
                        &m2(&p2(x, e)?, &p2(&max, &ng(e)?)?)?,
                        &apply(Function::UpperGamma, &[a2(e, &int(1))?, max.clone()])?,
                    )?,
                    a,
                )
            }
            Kind::EllipticF { a, d } => d2(&apply(Function::EllipticF, &[x.clone(), d2(d, a)?])?, &sqrt(a)?),
            Kind::EllipticE { a, d } => m2(&apply(Function::EllipticE, &[x.clone(), d2(d, a)?])?, &sqrt(a)?),
        }
    }
}

fn erf_eval(a: &Basic, b: &Basic, c: &Basic, x: &Basic) -> R {
    let pi: Basic = Basic::pi();
    let common = |root: &Basic| -> R {
        m2(
            &d2(&d2(&sqrt(&pi)?, root)?, &int(2))?,
            &exp(&s2(c, &d2(&p2(b, &int(2))?, &m2(&int(4), a)?)?)?)?,
        )
    };
    let erfi_part = || -> R {
        let root: Basic = sqrt(a)?;
        m2(
            &common(&root)?,
            &f1(
                Function::Erfi,
                &d2(&a2(&m2(&m2(&int(2), a)?, x)?, b)?, &m2(&int(2), &root)?)?,
            )?,
        )
    };
    if is(a, Predicate::extended_real) == Some(true) {
        let root: Basic = sqrt(&ng(a)?)?;
        let first: Basic = m2(
            &common(&root)?,
            &f1(
                Function::Erf,
                &d2(&s2(&m2(&m2(&int(-2), a)?, x)?, b)?, &m2(&int(2), &root)?)?,
            )?,
        )?;
        return piecewise(&[(first, lt(a, &int(0))?), (erfi_part()?, true_())]);
    }
    erfi_part()
}

fn fresnel_eval(a: &Basic, b: &Basic, c: &Basic, x: &Basic, cosine: bool) -> R {
    let pi: Basic = Basic::pi();
    let two_a: Basic = m2(&int(2), a)?;
    let lead: Basic = d2(&sqrt(&pi)?, &sqrt(&two_a)?)?;
    let phase: Basic = s2(&d2(&p2(b, &int(2))?, &m2(&int(4), a)?)?, c)?;
    let arg: Basic = d2(&a2(&m2(&two_a, x)?, b)?, &sqrt(&m2(&two_a, &pi)?)?)?;
    let fc: Basic = f1(Function::FresnelC, &arg)?;
    let fs: Basic = f1(Function::FresnelS, &arg)?;
    let (cp, sp) = (f1(Function::Cos, &phase)?, f1(Function::Sin, &phase)?);
    let inner: Basic = if cosine {
        a2(&m2(&cp, &fc)?, &m2(&sp, &fs)?)?
    } else {
        s2(&m2(&cp, &fs)?, &m2(&sp, &fc)?)?
    };
    m2(&lead, &inner)
}

fn derivative_eval(integrand: &Basic, x: &Basic) -> R {
    if !matches!(integrand.head(), Head::Derivative) {
        return Err(PolyError::Type(format!("not a derivative: {}", integrand)));
    }
    let mut specs: Vec<(Basic, u64)> = Vec::new();
    let mut done: bool = false;
    for vc in &integrand.args()[1..] {
        let var: &Basic = &vc.args()[0];
        let mut count: u64 = int_value(&vc.args()[1]).unwrap_or(1) as u64;
        if var == x && !done {
            count -= 1;
            done = true;
        }
        if count > 0 {
            specs.push((var.clone(), count));
        }
    }
    let expr: &Basic = &integrand.args()[0];
    if specs.is_empty() {
        return Ok(expr.clone());
    }
    Ok(crate::core::function::derivative_node(expr, &specs))
}

fn sqrt_quadratic_denom_eval(a: &Basic, b: &Basic, c: &Basic, coeffs: &[Basic], x: &Basic) -> R {
    let mut result_coeffs: Vec<Basic> = Vec::new();
    let mut coeffs: Vec<Basic> = coeffs.to_vec();
    let len: usize = coeffs.len();
    for i in 0..len.saturating_sub(2) {
        let n: i64 = (len - 1 - i) as i64;
        let coeff: Basic = d2(&coeffs[i], &m2(c, &int(n))?)?;
        result_coeffs.push(coeff.clone());
        coeffs[i + 1] = s2(&coeffs[i + 1], &m2(&d2(&m2(&int(2 * n - 1), b)?, &int(2))?, &coeff)?)?;
        coeffs[i + 2] = s2(&coeffs[i + 2], &m2(&m2(&int(n - 1), a)?, &coeff)?)?;
    }
    let (d, e) = (coeffs[len - 1].clone(), coeffs[len - 2].clone());
    let s: Basic = sqrt(&a2(&a2(a, &m2(b, x)?)?, &m2(c, &p2(x, &int(2))?)?)?)?;
    let constant: Basic = s2(&d, &d2(&m2(b, &e)?, &m2(&int(2), c)?)?)?;
    let mut i0: Basic = Basic::zero();
    if constant != Basic::zero() {
        let reciprocal: Basic = d2(&int(1), &s)?;
        let Some(step) = inverse_trig_generic(&reciprocal, x)? else {
            return Err(PolyError::Value(format!(
                "no inverse trigonometric rule for {}",
                reciprocal
            )));
        };
        i0 = m2(&constant, &step.eval()?)?;
    }
    let mut terms: Vec<Basic> = Vec::new();
    for (i, rc) in result_coeffs.iter().enumerate() {
        terms.push(m2(rc, &p2(x, &int((len - 2 - i) as i64))?)?);
    }
    terms.push(d2(&e, c)?);
    a2(&m2(&add_n(&terms)?, &s)?, &i0)
}

fn inverse_composition(node: &Basic) -> R {
    let trig = [Function::Sin, Function::Cos, Function::Tan];
    let inverse = [Function::Asin, Function::Acos, Function::Atan];
    if !is_fn(node, &trig) || !is_fn(&node.args()[0], &inverse) {
        return Ok(node.clone());
    }
    let inner: &Basic = &node.args()[0];
    let z: &Basic = &inner.args()[0];
    let one: Basic = int(1);
    let z2: Basic = p2(z, &int(2))?;
    let values: [Basic; 3] = if is_fn(inner, &[Function::Atan]) {
        let r: Basic = sqrt(&a2(&one, &z2)?)?;
        [d2(z, &r)?, d2(&one, &r)?, z.clone()]
    } else if is_fn(inner, &[Function::Asin]) {
        let r: Basic = sqrt(&s2(&one, &z2)?)?;
        [z.clone(), r.clone(), d2(z, &r)?]
    } else {
        let r: Basic = sqrt(&s2(&one, &z2)?)?;
        [r.clone(), z.clone(), d2(&r, z)?]
    };
    Ok(if is_fn(node, &[Function::Sin]) {
        values[0].clone()
    } else if is_fn(node, &[Function::Cos]) {
        values[1].clone()
    } else {
        values[2].clone()
    })
}

/// `sin(atan(z))` and the like written algebraically
/// (`_inverse_compositions`).
fn inverse_compositions(e: &Basic) -> R {
    let composed = |node: &Basic| -> bool {
        is_fn(node, &[Function::Sin, Function::Cos, Function::Tan])
            && is_fn(&node.args()[0], &[Function::Asin, Function::Acos, Function::Atan])
    };
    crate::core::function::replace_where(e, &composed, &inverse_composition)
}

fn trig_substitution_eval(theta: &Basic, func: &Basic, substep: &RuleRef, restriction: &Basic, x: &Basic) -> R {
    let mut func: Basic = subs(
        func,
        &f1(Function::Sec, theta)?,
        &d2(&int(1), &f1(Function::Cos, theta)?)?,
    )?;
    func = subs(
        &func,
        &f1(Function::Csc, theta)?,
        &d2(&int(1), &f1(Function::Sin, theta)?)?,
    )?;
    func = subs(
        &func,
        &f1(Function::Cot, theta)?,
        &d2(&int(1), &f1(Function::Tan, theta)?)?,
    )?;
    let mut trig_functions: Vec<Basic> = Vec::new();
    for n in func.preorder() {
        if is_fn(&n, &[Function::Sin, Function::Cos, Function::Tan]) && !trig_functions.contains(&n) {
            trig_functions.push(n);
        }
    }
    if trig_functions.len() != 1 {
        return Err(PolyError::Value(format!("not a trigonometric substitution: {}", func)));
    }
    let trig_function: Basic = trig_functions.swap_remove(0);
    let (constant, dependent) = as_independent(&func, theta, false)?;
    let relation: Basic = if dependent == trig_function {
        d2(x, &constant)?
    } else if dependent == d2(&int(1), &trig_function)? {
        d2(&constant, x)?
    } else {
        return Err(PolyError::Value(format!("not a trigonometric substitution: {}", func)));
    };
    let (numer, denom) = crate::simplify::radsimp::fraction(&relation)?;
    let (opposite, adjacent, hypotenuse, inverse) = if is_fn(&trig_function, &[Function::Sin]) {
        (
            numer.clone(),
            sqrt(&s2(&p2(&denom, &int(2))?, &p2(&numer, &int(2))?)?)?,
            denom.clone(),
            f1(Function::Asin, &relation)?,
        )
    } else if is_fn(&trig_function, &[Function::Cos]) {
        (
            sqrt(&s2(&p2(&denom, &int(2))?, &p2(&numer, &int(2))?)?)?,
            numer.clone(),
            denom.clone(),
            f1(Function::Acos, &relation)?,
        )
    } else {
        (
            numer.clone(),
            denom.clone(),
            sqrt(&a2(&p2(&denom, &int(2))?, &p2(&numer, &int(2))?)?)?,
            f1(Function::Atan, &relation)?,
        )
    };
    let substitution: Vec<(Basic, Basic)> = vec![
        (f1(Function::Sin, theta)?, d2(&opposite, &hypotenuse)?),
        (f1(Function::Cos, theta)?, d2(&adjacent, &hypotenuse)?),
        (f1(Function::Tan, theta)?, d2(&opposite, &adjacent)?),
        (theta.clone(), inverse),
    ];
    let hints = crate::core::function::Hints {
        trig: true,
        ..crate::core::function::Hints::DEFAULT
    };
    let result: Basic = subs_list(
        &crate::core::function::expand_with(&substep.eval()?, true, hints)?,
        &substitution,
    )?;
    let simplified: Basic = crate::simplify::trigsimp::trigsimp(
        &inverse_compositions(&result)?,
        crate::simplify::trigsimp::Method::Matching,
    )?;
    piecewise(&[(simplified, restriction.clone())])
}

// ---------------------------------------------------------------------
// the solver

/// An integral to solve (`IntegralInfo`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntegralInfo {
    pub integrand: Basic,
    pub symbol: Basic,
}

impl IntegralInfo {
    pub fn new(integrand: &Basic, symbol: &Basic) -> IntegralInfo {
        IntegralInfo {
            integrand: integrand.clone(),
            symbol: symbol.clone(),
        }
    }
}

/// The state of one integration (`IntegralSolver`): the integrals being
/// solved, the uses of each `u` of an integration by parts, the steps of
/// the integrals solved completely.
pub struct IntegralSolver {
    cache_dummy: Basic,
    active: HashSet<Basic>,
    parts_u: HashMap<Basic, u32>,
    solved: HashMap<IntegralInfo, RuleRef>,
    /// An error raised while a requested integral was solved: it leaves
    /// Python's driver and passes the guards.
    escaping: bool,
}

impl Default for IntegralSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegralSolver {
    pub fn new() -> IntegralSolver {
        IntegralSolver {
            cache_dummy: fresh("z"),
            active: HashSet::new(),
            parts_u: HashMap::new(),
            solved: HashMap::new(),
            escaping: false,
        }
    }

    fn key(&self, e: &Basic, symbol: &Basic) -> Basic {
        let rule: HashMap<Basic, Basic> = [(symbol.clone(), self.cache_dummy.clone())].into_iter().collect();
        e.xreplace(&rule)
    }

    /// Count a use of `u` in an integration by parts; `false` when it was
    /// used too often already (`use_parts_u`).
    pub fn use_parts_u(&mut self, u: &Basic, symbol: &Basic) -> bool {
        let cachekey: Basic = self.key(u, symbol);
        let count: u32 = *self.parts_u.get(&cachekey).unwrap_or(&0);
        if count > 2 {
            return false;
        }
        self.parts_u.insert(cachekey, count + 1);
        true
    }

    /// The steps of `integral` (`solve`).
    pub fn solve(&mut self, integral: &IntegralInfo) -> PolyResult<RuleRef> {
        if let Some(known) = self.solved.get(integral) {
            return Ok(known.clone());
        }
        let cachekey: Basic = self.key(&integral.integrand, &integral.symbol);
        if self.active.contains(&cachekey) {
            return Ok(rule(&integral.integrand, &integral.symbol, Kind::DontKnow));
        }
        self.active.insert(cachekey.clone());
        let found: PolyResult<Option<RuleRef>> = integral_steps_of(integral, self);
        self.active.remove(&cachekey);
        let result: RuleRef = match found {
            Ok(Some(r)) => r,
            Ok(None) => rule(&integral.integrand, &integral.symbol, Kind::DontKnow),
            Err(e) => {
                self.escaping = true;
                return Err(e);
            }
        };
        if !result.contains_dont_know() {
            self.solved.insert(integral.clone(), result.clone());
        }
        Ok(result)
    }

    /// The steps of a subproblem (`yield from _steps(f, x)`).
    pub fn steps(&mut self, integrand: &Basic, symbol: &Basic) -> PolyResult<RuleRef> {
        self.solve(&IntegralInfo::new(integrand, symbol))
    }
}

/// The errors a guarded rule gives up on (`_FAILURES`).
pub(crate) fn is_failure(e: &PolyError) -> bool {
    match e {
        PolyError::NotImplemented(m) => !m.starts_with(crate::error::NOT_PORTED_PREFIX),
        PolyError::ZeroDivision(_)
        | PolyError::Value(_)
        | PolyError::HeuristicGCDFailed(_)
        | PolyError::Arithmetic(_)
        | PolyError::Type(_)
        | PolyError::Polys(_, _) => true,
        _ => false,
    }
}

pub(crate) type Found = PolyResult<Option<RuleRef>>;

/// The rule with its own failures given up on (`_guarded`); the failures
/// of the subproblems go through.
pub(crate) fn guarded(
    strategy: &dyn Fn(&IntegralInfo, &mut IntegralSolver) -> Found,
    integral: &IntegralInfo,
    solver: &mut IntegralSolver,
) -> Found {
    match strategy(integral, solver) {
        Err(e) if !solver.escaping && is_failure(&e) => Ok(None),
        other => other,
    }
}

fn trigsimp(e: &Basic) -> R {
    crate::simplify::trigsimp::trigsimp(e, crate::simplify::trigsimp::Method::Matching)
}

// ---------------------------------------------------------------------
// polynomial patterns (SymPy's wildcards a + b*x + c*x**2)

/// The coefficients of `e` as a polynomial in `x`, the lowest degree
/// first; `None` when it is not one (`_coefficients`).
pub(crate) fn coefficients(e: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    if !is_polynomial(e, x) {
        return Ok(None);
    }
    let mut found: Vec<(usize, Basic)> = Vec::new();
    for term in add_args(&expand(e)?) {
        let (coeff, dependent) = as_independent(&term, x, false)?;
        let k: usize = if dependent == Basic::one() {
            0
        } else if dependent == *x {
            1
        } else if dependent.is_pow()
            && dependent.args()[0] == *x
            && int_value(&dependent.args()[1]).is_some_and(|k| k > 0)
        {
            int_value(&dependent.args()[1]).expect("an integer") as usize
        } else {
            return Ok(None);
        };
        match found.iter_mut().find(|(j, _)| *j == k) {
            Some(entry) => entry.1 = a2(&entry.1, &coeff)?,
            None => found.push((k, a2(&Basic::zero(), &coeff)?)),
        }
    }
    if found.is_empty() {
        return Ok(Some(vec![Basic::zero()]));
    }
    let degree: usize = found.iter().map(|(k, _)| *k).max().expect("a term");
    Ok(Some(
        (0..=degree)
            .map(|k| {
                found
                    .iter()
                    .find(|(j, _)| *j == k)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(Basic::zero)
            })
            .collect(),
    ))
}

/// `(a, b)` with `e = a + b*x` and `b` nonzero (`_linear`).
pub(crate) fn linear(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some(c) = coefficients(e, x)? else { return Ok(None) };
    if c.len() != 2 || is(&c[1], Predicate::zero) == Some(true) {
        return Ok(None);
    }
    Ok(Some((c[0].clone(), c[1].clone())))
}

/// `(a, b, c)` with `e = a + b*x + c*x**2` and `c` nonzero (`_quadratic`).
pub(crate) fn quadratic(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    let Some(c) = coefficients(e, x)? else { return Ok(None) };
    if c.len() != 3 || is(&c[2], Predicate::zero) == Some(true) {
        return Ok(None);
    }
    Ok(Some((c[0].clone(), c[1].clone(), c[2].clone())))
}

/// `a` with `e = a*x` (`_times`).
pub(crate) fn times(x: &Basic, e: &Basic) -> PolyResult<Option<Basic>> {
    let Some(c) = coefficients(e, x)? else { return Ok(None) };
    if c.len() != 2 || c[0] != Basic::zero() {
        return Ok(None);
    }
    Ok(Some(c[1].clone()))
}

// ---------------------------------------------------------------------
// substitutions

/// The derivative with the derivatives of `tan`, `cot`, `sec`, `csc` in
/// SymPy's manual forms (`manual_diff`).
pub fn manual_diff(f: &Basic, symbol: &Basic) -> R {
    if !f.args().is_empty() {
        let arg: &Basic = &f.args()[0];
        if is_fn(f, &[Function::Tan]) {
            return m2(&diff(arg, symbol)?, &p2(&f1(Function::Sec, arg)?, &int(2))?);
        } else if is_fn(f, &[Function::Cot]) {
            return m2(&ng(&diff(arg, symbol)?)?, &p2(&f1(Function::Csc, arg)?, &int(2))?);
        } else if is_fn(f, &[Function::Sec]) {
            return m2(
                &m2(&diff(arg, symbol)?, &f1(Function::Sec, arg)?)?,
                &f1(Function::Tan, arg)?,
            );
        } else if is_fn(f, &[Function::Csc]) {
            return m2(
                &m2(&ng(&diff(arg, symbol)?)?, &f1(Function::Csc, arg)?)?,
                &f1(Function::Cot, arg)?,
            );
        } else if f.is_add() {
            let parts: Vec<Basic> = f
                .args()
                .iter()
                .map(|a| manual_diff(a, symbol))
                .collect::<PolyResult<_>>()?;
            return add_n(&parts);
        } else if f.is_mul() && f.args().len() == 2 && f.args()[0].is_number_node() {
            return m2(&f.args()[0], &manual_diff(&f.args()[1], symbol)?);
        }
    }
    diff(f, symbol)
}

/// The substitutions with `log(x0) -> new` also as `x0 -> exp(new)` and
/// the powers of `x0` through `exp` (`manual_subs`).
pub fn manual_subs(expr: &Basic, sequence: &[(Basic, Basic)]) -> R {
    let mut expr: Basic = expr.clone();
    let mut new_subs: Vec<(Basic, Basic)> = Vec::new();
    for (old, new) in sequence {
        if is_fn(old, &[Function::Log]) {
            let x0: Basic = old.args()[0].clone();
            let is_power = |e: &Basic| e.is_pow() && e.args()[0] == x0;
            let power = |e: &Basic| exp(&m2(&e.args()[1], new)?);
            expr = crate::core::function::replace_where(&expr, &is_power, &power)?;
            new_subs.push((x0, exp(new)?));
        }
    }
    let mut pairs: Vec<(Basic, Basic)> = sequence.to_vec();
    pairs.extend(new_subs);
    subs_list(&expr, &pairs)
}

const INVERSE_TRIG: [Function; 6] = [
    Function::Atan,
    Function::Asin,
    Function::Acos,
    Function::Acot,
    Function::Acsc,
    Function::Asec,
];
const TRIG: [Function; 6] = [
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
];
const HYPERBOLIC: [Function; 6] = [
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];
const ORTHOGONAL: [(Function, usize); 8] = [
    (Function::Jacobi, 3),
    (Function::Gegenbauer, 2),
    (Function::ChebyshevT, 1),
    (Function::ChebyshevU, 1),
    (Function::Legendre, 1),
    (Function::Hermite, 1),
    (Function::Laguerre, 1),
    (Function::AssocLaguerre, 2),
];

fn orthogonal_index(e: &Basic) -> Option<usize> {
    ORTHOGONAL.iter().find(|(f, _)| is_fn(e, &[*f])).map(|(_, i)| *i)
}

fn degree_in(e: &Basic, x: &Basic) -> PolyResult<i64> {
    crate::polys::polytools::degree(e, x)
}

/// The substitutions `u = g(x)` with `f/g'` free of `x` after the change
/// of variable (`find_substitutions`): `(u, constant, new integrand)`.
pub fn find_substitutions(integrand: &Basic, symbol: &Basic, u_var: &Basic) -> PolyResult<Vec<(Basic, Basic, Basic)>> {
    let mut results: Vec<(Basic, Basic, Basic)> = Vec::new();
    let test_subterm = |u: &Basic, u_diff: &Basic| -> PolyResult<Option<(Basic, Basic)>> {
        if *u_diff == Basic::zero() {
            return Ok(None);
        }
        let substituted: Basic = d2(integrand, u_diff)?;
        let substituted: Basic = cancel(&manual_subs(&substituted, &[(u.clone(), u_var.clone())])?)?;
        if substituted.free_symbols().contains(symbol) {
            return Ok(None);
        }
        if is_rational_function_in(integrand, std::slice::from_ref(symbol))
            && is_rational_function_in(&substituted, std::slice::from_ref(u_var))
        {
            let (n0, d0) = as_numer_denom(integrand)?;
            let (n1, d1) = as_numer_denom(&substituted)?;
            let before: i64 = degree_in(&n0, symbol)?.max(degree_in(&d0, symbol)?);
            let after: i64 = degree_in(&n1, u_var)?.max(degree_in(&d1, u_var)?);
            if after > before {
                return Ok(None);
            }
        }
        Ok(Some(as_independent(&substituted, u_var, false)?))
    };
    let exp_subterms = |term: &Basic| -> PolyResult<Vec<Basic>> {
        let mut linear_coeffs: Vec<i64> = Vec::new();
        let mut terms: Vec<Basic> = Vec::new();
        let mut seen: Vec<Basic> = Vec::new();
        for e in term.preorder() {
            if !is_fn(&e, &[Function::Exp]) || seen.contains(&e) {
                continue;
            }
            seen.push(e.clone());
            let arg: &Basic = &e.args()[0];
            if !arg.free_symbols().contains(symbol) {
                continue;
            }
            match times(symbol, arg)? {
                Some(c) if is_integer_node(&c) => linear_coeffs.push(int_value(&c).unwrap_or(0)),
                _ => terms.push(e.clone()),
            }
        }
        if !linear_coeffs.is_empty() {
            let g: i64 = linear_coeffs.iter().fold(0i64, |g, k| gcd(g, *k));
            terms.push(exp(&m2(&int(g), symbol)?)?);
        }
        Ok(terms)
    };
    fn possible_subterms(term: &Basic, symbol: &Basic) -> PolyResult<Vec<Basic>> {
        let mut functions: Vec<Function> = TRIG.to_vec();
        functions.extend(HYPERBOLIC);
        functions.extend(INVERSE_TRIG);
        functions.extend([Function::Exp, Function::Log, Function::Heaviside]);
        if is_fn(term, &functions) {
            return Ok(vec![term.args()[0].clone()]);
        }
        if let Some(index) = orthogonal_index(term) {
            return Ok(vec![term.args()[index].clone()]);
        }
        if term.is_mul() {
            let mut r: Vec<Basic> = Vec::new();
            for u in term.args() {
                r.push(u.clone());
                r.extend(possible_subterms(u, symbol)?);
            }
            return Ok(r);
        }
        if term.is_pow() {
            let mut r: Vec<Basic> = term.args().iter().filter(|a| a.has(symbol)).cloned().collect();
            if let Some(e) = int_value(&term.args()[1]) {
                let e: i64 = e.abs();
                let factors = crate::ntheory::factor_::factorint(&Int::from(e), None);
                for d in factors.keys() {
                    let d: i64 = i64::try_from(d).unwrap_or(0);
                    if 1 < d && d < e {
                        r.push(p2(&term.args()[0], &int(d))?);
                    }
                }
                if term.args()[0].is_add() {
                    r.extend(
                        possible_subterms(&term.args()[0], symbol)?
                            .into_iter()
                            .filter(|t| t.is_pow()),
                    );
                }
            }
            return Ok(r);
        }
        if term.is_add() {
            let mut r: Vec<Basic> = Vec::new();
            for a in term.args() {
                r.push(a.clone());
                r.extend(possible_subterms(a, symbol)?);
            }
            return Ok(r);
        }
        Ok(Vec::new())
    }
    let mut candidates: Vec<Basic> = Vec::new();
    for u in possible_subterms(integrand, symbol)?
        .into_iter()
        .chain(exp_subterms(integrand)?)
    {
        if !candidates.contains(&u) {
            candidates.push(u);
        }
    }
    let at_u: Basic = subs(integrand, symbol, u_var)?;
    for u in candidates {
        if u == *symbol {
            continue;
        }
        let u_diff: Basic = manual_diff(&u, symbol)?;
        if let Some((constant, new_integrand)) = test_subterm(&u, &u_diff)? {
            if new_integrand == at_u {
                continue;
            }
            let substitution = (u, constant, new_integrand);
            if !results.contains(&substitution) {
                results.push(substitution);
            }
        }
    }
    Ok(results)
}

pub(crate) fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

// ---------------------------------------------------------------------
// the combinators

fn first(
    strategies: &[&dyn Fn(&IntegralInfo, &mut IntegralSolver) -> Found],
    integral: &IntegralInfo,
    solver: &mut IntegralSolver,
) -> Found {
    for s in strategies {
        if let Some(found) = guarded(*s, integral, solver)? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

/// A rewriting followed by the steps of the rewritten integrand when it
/// is done (`_rewriter`).
fn rewriter(
    condition: fn(&Basic, &Basic) -> bool,
    rewrite: fn(&Basic, &Basic) -> R,
    integral: &IntegralInfo,
    solver: &mut IntegralSolver,
) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    if condition(integrand, symbol) {
        let rewritten: Basic = rewrite(integrand, symbol)?;
        if rewritten != *integrand {
            let substep: RuleRef = solver.steps(&rewritten, symbol)?;
            if substep.kind != Kind::DontKnow {
                return Ok(Some(rule(integrand, symbol, Kind::Rewrite { rewritten, substep })));
            }
        }
    }
    Ok(None)
}

/// Every strategy tried, the distinct results kept (`_alternatives`).
fn alternatives(
    strategies: &[&dyn Fn(&IntegralInfo, &mut IntegralSolver) -> Found],
    integral: &IntegralInfo,
    solver: &mut IntegralSolver,
) -> Found {
    let mut alts: Vec<RuleRef> = Vec::new();
    for s in strategies {
        if let Some(result) = guarded(*s, integral, solver)? {
            if result.kind != Kind::DontKnow && !alts.contains(&result) {
                alts.push(result);
            }
        }
    }
    if alts.len() == 1 {
        return Ok(alts.pop());
    }
    if !alts.is_empty() {
        let doable: Vec<RuleRef> = alts.iter().filter(|r| !r.contains_dont_know()).cloned().collect();
        let chosen: Vec<RuleRef> = if doable.is_empty() { alts } else { doable };
        return Ok(Some(rule(
            &integral.integrand,
            &integral.symbol,
            Kind::Alternative { alternatives: chosen },
        )));
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// the rules

fn constant_rule(integral: &IntegralInfo) -> Found {
    Ok(Some(rule(&integral.integrand, &integral.symbol, Kind::Constant)))
}

fn power_rule(integral: &IntegralInfo) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let (base, expt) = as_base_exp(integrand)?;
    if !expt.free_symbols().contains(symbol) && is_symbol(&base) {
        if zero(&a2(&expt, &int(1))?)? {
            return Ok(Some(rule(integrand, symbol, Kind::Reciprocal { base })));
        }
        return Ok(Some(rule(integrand, symbol, Kind::Power { base, exp: expt })));
    } else if !base.free_symbols().contains(symbol) && is_symbol(&expt) {
        let exp_rule: RuleRef = rule(
            integrand,
            symbol,
            Kind::Exp {
                base: base.clone(),
                exp: expt,
            },
        );
        let lb: Basic = log(&base)?;
        return Ok(Some(match is(&lb, Predicate::zero) {
            Some(false) => exp_rule,
            Some(true) => rule(&Basic::one(), symbol, Kind::Constant),
            None => rule(
                integrand,
                symbol,
                Kind::Piecewise {
                    subfunctions: vec![
                        (exp_rule, ne(&lb, &int(0))?),
                        (rule(&Basic::one(), symbol, Kind::Constant), true_()),
                    ],
                },
            ),
        }));
    }
    Ok(None)
}

fn exp_rule(integral: &IntegralInfo) -> Found {
    let argument: &Basic = &integral.integrand.args()[0];
    if is_symbol(argument) {
        return Ok(Some(rule(
            &integral.integrand,
            &integral.symbol,
            Kind::Exp {
                base: Basic::atom(Head::Exp1),
                exp: argument.clone(),
            },
        )));
    }
    Ok(None)
}

fn orthogonal_poly_rule(integral: &IntegralInfo) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let Some(var_index) = orthogonal_index(integrand) else {
        return Ok(None);
    };
    let args: &[Basic] = integrand.args();
    if args[var_index] != *symbol || args[..var_index].iter().any(|v| v.has(symbol)) {
        return Ok(None);
    }
    let a = |i: usize| args[i].clone();
    let kind: Kind = match integrand.head() {
        Head::Function(Function::Jacobi) => Kind::Jacobi {
            n: a(0),
            a: a(1),
            b: a(2),
        },
        Head::Function(Function::Gegenbauer) => Kind::Gegenbauer { n: a(0), a: a(1) },
        Head::Function(Function::ChebyshevT) => Kind::ChebyshevT { n: a(0) },
        Head::Function(Function::ChebyshevU) => Kind::ChebyshevU { n: a(0) },
        Head::Function(Function::Legendre) => Kind::Legendre { n: a(0) },
        Head::Function(Function::Hermite) => Kind::Hermite { n: a(0) },
        Head::Function(Function::Laguerre) => Kind::Laguerre { n: a(0) },
        _ => Kind::AssocLaguerre { n: a(0), a: a(1) },
    };
    Ok(Some(rule(integrand, symbol, kind)))
}

fn over_symbol(integrand: &Basic, symbol: &Basic) -> PolyResult<Option<Basic>> {
    if !integrand.is_mul() {
        return Ok(None);
    }
    let mut factors: Vec<Basic> = mul_args(integrand);
    let inverse: Basic = d2(&int(1), symbol)?;
    let Some(position) = factors.iter().position(|f| *f == inverse) else {
        return Ok(None);
    };
    factors.remove(position);
    if factors.len() != 1 {
        return Ok(None);
    }
    Ok(factors.pop())
}

fn special_function_rule(integral: &IntegralInfo) -> Found {
    let (integrand, x) = (&integral.integrand, &integral.symbol);
    if integrand.is_mul() {
        let g: Option<Basic> = over_symbol(integrand, x)?;
        if let Some(g) = &g {
            if is_fn(
                g,
                &[
                    Function::Exp,
                    Function::Cos,
                    Function::Cosh,
                    Function::Sin,
                    Function::Sinh,
                ],
            ) {
                if let Some((b, a)) = linear(&g.args()[0], x)? {
                    let kind: Kind = if is_fn(g, &[Function::Exp]) {
                        Kind::Ei { a, b }
                    } else if is_fn(g, &[Function::Cos]) {
                        Kind::Ci { a, b }
                    } else if is_fn(g, &[Function::Cosh]) {
                        Kind::Chi { a, b }
                    } else if is_fn(g, &[Function::Sin]) {
                        Kind::Si { a, b }
                    } else {
                        Kind::Shi { a, b }
                    };
                    return Ok(Some(rule(integrand, x, kind)));
                }
            }
        }
        let factors: Vec<Basic> = mul_args(integrand);
        if factors.len() == 2 {
            for (power, exponential) in [(&factors[0], &factors[1]), (&factors[1], &factors[0])] {
                if !is_fn(exponential, &[Function::Exp]) {
                    continue;
                }
                let Some(rate) = times(x, &exponential.args()[0])? else {
                    continue;
                };
                if is(&rate, Predicate::zero) != Some(false) {
                    continue;
                }
                let (base, e) = as_base_exp(power)?;
                if base != *x
                    || e.has(x)
                    || (is(&e, Predicate::nonnegative) == Some(true) && is(&e, Predicate::integer) == Some(true))
                {
                    continue;
                }
                return Ok(Some(rule(integrand, x, Kind::UpperGamma { a: rate, e })));
            }
        }
        if let Some(g) = &g {
            if is_fn(g, &[Function::PolyLog]) {
                let order: &Basic = &g.args()[0];
                if let Some(rate) = times(x, &g.args()[1])? {
                    if is(&rate, Predicate::zero) == Some(false) && !order.has(x) {
                        return Ok(Some(rule(
                            integrand,
                            x,
                            Kind::Polylog {
                                a: rate,
                                b: order.clone(),
                            },
                        )));
                    }
                }
            }
        }
        return Ok(None);
    }
    if integrand.is_pow() {
        let (base, e) = as_base_exp(integrand)?;
        if e == int(-1) && is_fn(&base, &[Function::Log]) {
            if let Some((b, a)) = linear(&base.args()[0], x)? {
                return Ok(Some(rule(integrand, x, Kind::Li { a, b })));
            }
        }
        if e == half() || e == rational(-1, 2) {
            if let Some((a, d)) = coefficients_in_sine_square(&base, x)? {
                if a != d {
                    let kind: Kind = if e == half() {
                        Kind::EllipticE { a, d }
                    } else {
                        Kind::EllipticF { a, d }
                    };
                    return Ok(Some(rule(integrand, x, kind)));
                }
            }
        }
        return Ok(None);
    }
    if is_fn(integrand, &[Function::Exp, Function::Sin, Function::Cos]) {
        if let Some((c, b, a)) = quadratic(&integrand.args()[0], x)? {
            let kind: Kind = if is_fn(integrand, &[Function::Exp]) {
                Kind::Erf { a, b, c }
            } else if is_fn(integrand, &[Function::Sin]) {
                Kind::FresnelS { a, b, c }
            } else {
                Kind::FresnelC { a, b, c }
            };
            return Ok(Some(rule(integrand, x, kind)));
        }
    }
    Ok(None)
}

fn coefficients_in_sine_square(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let t: Basic = fresh("s");
    let map: HashMap<Basic, Basic> = [(f1(Function::Sin, x)?, t.clone())].into_iter().collect();
    let replaced: Basic = crate::core::function::xreplace_evaluated(e, &map)?;
    if replaced.has(x) {
        return Ok(None);
    }
    let Some(c) = coefficients(&replaced, &t)? else {
        return Ok(None);
    };
    if c.len() != 3 || c[1] != Basic::zero() {
        return Ok(None);
    }
    let (a, d) = (c[0].clone(), ng(&c[2])?);
    if is(&a, Predicate::zero) != Some(false) || is(&d, Predicate::zero) != Some(false) {
        return Ok(None);
    }
    Ok(Some((a, d)))
}

/// The generic step with the degenerate case as a `Piecewise`
/// (`_add_degenerate_step`).
fn add_degenerate_step(generic_cond: &Basic, generic_step: RuleRef, degenerate_step: Option<RuleRef>) -> RuleRef {
    let Some(degenerate) = degenerate_step else {
        return generic_step;
    };
    let mut subfunctions: Vec<(RuleRef, Basic)> = match &generic_step.kind {
        Kind::Piecewise { subfunctions } => subfunctions
            .iter()
            .map(|(s, c)| (s.clone(), and2(c, generic_cond)))
            .collect(),
        _ => vec![(generic_step.clone(), generic_cond.clone())],
    };
    match &degenerate.kind {
        Kind::Piecewise { subfunctions: more } => subfunctions.extend(more.iter().cloned()),
        _ => subfunctions.push((degenerate.clone(), true_())),
    }
    rule(
        &generic_step.integrand,
        &generic_step.variable,
        Kind::Piecewise { subfunctions },
    )
}

/// `(base, exponent)` of a product of powers of one linear function
/// (`_nested_pow_rule.get_base_exp`); `None` when there is none.
fn get_base_exp(expr: &Basic, x: &Basic, conditions: &mut Vec<Basic>) -> PolyResult<Option<(Basic, Basic)>> {
    if !expr.free_symbols().contains(x) {
        return Ok(Some((Basic::one(), Basic::zero())));
    }
    if expr.is_mul() {
        let terms: Vec<Basic> = as_coeff_mul_terms(expr);
        if terms.is_empty() {
            return Ok(Some((Basic::one(), Basic::zero())));
        }
        let mut results: Vec<(Basic, Basic)> = Vec::new();
        for term in &terms {
            match get_base_exp(term, x, conditions)? {
                Some(r) => results.push(r),
                None => return Ok(None),
            }
        }
        let mut bases: Vec<Basic> = Vec::new();
        for (b, _) in &results {
            if *b != Basic::one() && !bases.contains(b) {
                bases.push(b.clone());
            }
        }
        if bases.len() == 1 {
            let exps: Vec<Basic> = results.into_iter().map(|(_, e)| e).collect();
            return Ok(Some((bases.pop().expect("a base"), add_n(&exps)?)));
        }
        return Ok(None);
    }
    if expr.is_pow() {
        let (b, e) = (&expr.args()[0], &expr.args()[1]);
        if e.free_symbols().contains(x) {
            return Ok(None);
        }
        let Some((base, sub_exp)) = get_base_exp(b, x, conditions)? else {
            return Ok(None);
        };
        return Ok(Some((base, m2(&sub_exp, e)?)));
    }
    if let Some((a, b)) = linear(expr, x)? {
        conditions.push(ne(&b, &int(0))?);
        return Ok(Some((a2(x, &d2(&a, &b)?)?, Basic::one())));
    }
    Ok(None)
}

/// The factors of a product after its rational coefficient
/// (`as_coeff_mul()[1]`).
fn as_coeff_mul_terms(e: &Basic) -> Vec<Basic> {
    let args: &[Basic] = e.args();
    if !args.is_empty() && args[0].as_rational().is_some() {
        args[1..].to_vec()
    } else {
        args.to_vec()
    }
}

fn nested_pow_rule(integral: &IntegralInfo) -> Found {
    let (integrand, x) = (&integral.integrand, &integral.symbol);
    let mut conditions: Vec<Basic> = Vec::new();
    let Some((base, exp_)) = get_base_exp(integrand, x, &mut conditions)? else {
        return Ok(None);
    };
    let generic_cond: Basic = conditions.last().cloned().unwrap_or_else(true_);
    let degenerate: Option<RuleRef> = if !is_true_node(&generic_cond) {
        Some(rule(&subs(integrand, x, &Basic::zero())?, x, Kind::Constant))
    } else {
        None
    };
    let generic: RuleRef = rule(integrand, x, Kind::NestedPow { base, exp: exp_ });
    Ok(Some(add_degenerate_step(&generic_cond, generic, degenerate)))
}

fn make_inverse_trig(
    integrand: &Basic,
    symbol: &Basic,
    arcsin: bool,
    a: &Basic,
    sign_a: i64,
    c: &Basic,
    sign_c: i64,
    h: &Basic,
) -> PolyResult<RuleRef> {
    let u_var: Basic = fresh("u");
    let shifted: Basic = s2(symbol, h)?;
    let rewritten: Basic = d2(
        &int(1),
        &sqrt(&a2(
            &m2(&int(sign_a), a)?,
            &m2(&m2(&int(sign_c), c)?, &p2(&shifted, &int(2))?)?,
        )?)?,
    )?;
    let quadratic_base: Basic = m2(&sqrt(&d2(c, a)?)?, &shifted)?;
    let constant: Basic = d2(&int(1), &sqrt(c)?)?;
    let mut u_func: Option<Basic> = None;
    let mut variable: Basic = symbol.clone();
    if quadratic_base != *symbol {
        u_func = Some(quadratic_base);
        variable = u_var.clone();
    }
    let standard_form: Basic = d2(
        &int(1),
        &sqrt(&a2(&int(sign_a), &m2(&int(sign_c), &p2(&variable, &int(2))?)?)?)?,
    )?;
    let mut substep: RuleRef = rule(
        &standard_form,
        &variable,
        if arcsin { Kind::Arcsin } else { Kind::Arcsinh },
    );
    if constant != int(1) {
        substep = rule(
            &m2(&constant, &standard_form)?,
            symbol,
            Kind::ConstantTimes {
                constant: constant.clone(),
                other: standard_form.clone(),
                substep,
            },
        );
    }
    if let Some(u_func) = u_func {
        substep = rule(&rewritten, symbol, Kind::U { u_var, u_func, substep });
    }
    if *h != Basic::zero() {
        substep = rule(integrand, symbol, Kind::CompleteSquare { rewritten, substep });
    }
    Ok(substep)
}

/// The inverse trigonometric rules for powers of a quadratic
/// (`_inverse_trig`); without a solver, a subproblem is an error.
fn inverse_trig(integral: &IntegralInfo, degenerate: bool, solver: Option<&mut IntegralSolver>) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let (base, exp_) = as_base_exp(integrand)?;
    let Some((a, b, c)) = quadratic(&base, symbol)? else {
        return Ok(None);
    };
    let generic_cond: Basic = ne(&c, &int(0))?;
    let mut degenerate_step: Option<RuleRef> = None;
    if degenerate && !is_true_node(&generic_cond) {
        if is(&b, Predicate::zero) == Some(true) {
            degenerate_step = Some(rule(&p2(&a, &exp_)?, symbol, Kind::Constant));
        } else {
            let Some(solver) = solver else {
                return Err(PolyError::Value(
                    "the generic inverse trigonometric rule asked for a subproblem".to_string(),
                ));
            };
            let linear_integrand: Basic = p2(&a2(&a, &m2(&b, symbol)?)?, &exp_)?;
            degenerate_step = sqrt_linear(&IntegralInfo::new(&linear_integrand, symbol), solver)?;
        }
    }
    if zero(&a2(&m2(&int(2), &exp_)?, &int(1))?)? {
        let h: Basic = d2(&ng(&b)?, &m2(&int(2), &c)?)?;
        let k: Basic = s2(&a, &d2(&p2(&b, &int(2))?, &m2(&int(4), &c)?)?)?;
        let non_square_cond: Basic = ne(&k, &int(0))?;
        let mut square_step: Option<RuleRef> = None;
        let shifted: Basic = s2(symbol, &h)?;
        if !is_true_node(&non_square_cond) {
            let integrand_sq: Basic = d2(&int(1), &sqrt(&m2(&c, &p2(&shifted, &int(2))?)?)?)?;
            square_step = Some(rule(
                &integrand_sq,
                symbol,
                Kind::NestedPow {
                    base: shifted.clone(),
                    exp: int(-1),
                },
            ));
        }
        if is_false_node(&non_square_cond) {
            return Ok(square_step);
        }
        let generic_step: RuleRef = rule(
            integrand,
            symbol,
            Kind::ReciprocalSqrtQuadratic {
                a: a.clone(),
                b: b.clone(),
                c: c.clone(),
            },
        );
        let mut step: RuleRef = add_degenerate_step(&non_square_cond, generic_step.clone(), square_step);
        if is(&k, Predicate::real) == Some(true) && is(&c, Predicate::real) == Some(true) {
            let mut rules: Vec<(RuleRef, Basic)> = Vec::new();
            let zero_: Basic = int(0);
            let candidates: [(bool, i64, Basic); 2] = [
                (true, -1, and2(&gt(&k, &zero_)?, &lt(&c, &zero_)?)),
                (false, 1, and2(&gt(&k, &zero_)?, &gt(&c, &zero_)?)),
            ];
            for (arcsin, sign_c, cond) in candidates {
                let c_: Basic = if arcsin { ng(&c)? } else { c.clone() };
                if is_true_node(&cond) {
                    return Ok(Some(make_inverse_trig(
                        integrand, symbol, arcsin, &k, 1, &c_, sign_c, &h,
                    )?));
                }
                if !is_false_node(&cond) {
                    rules.push((
                        make_inverse_trig(integrand, symbol, arcsin, &k, 1, &c_, sign_c, &h)?,
                        cond,
                    ));
                }
            }
            if !rules.is_empty() {
                if is(&k, Predicate::positive) != Some(true) {
                    rules.push((generic_step.clone(), true_()));
                }
                step = rule(integrand, symbol, Kind::Piecewise { subfunctions: rules });
            } else {
                step = generic_step;
            }
        }
        return Ok(Some(add_degenerate_step(&generic_cond, step, degenerate_step)));
    }
    if exp_ == half() {
        let sqrt_step: RuleRef = rule(integrand, symbol, Kind::SqrtQuadratic { a, b, c });
        return Ok(Some(add_degenerate_step(&generic_cond, sqrt_step, degenerate_step)));
    }
    Ok(None)
}

/// [`inverse_trig`] without the degenerate case (`_inverse_trig_generic`).
fn inverse_trig_generic(integrand: &Basic, symbol: &Basic) -> Found {
    inverse_trig(&IntegralInfo::new(integrand, symbol), false, None)
}

fn inverse_trig_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    inverse_trig(integral, true, Some(solver))
}

fn add_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let mut results: Vec<RuleRef> = Vec::new();
    for g in crate::core::printing::ordered_terms(&add_args(integrand)) {
        results.push(solver.steps(&g, symbol)?);
    }
    Ok(Some(rule(integrand, symbol, Kind::Add { substeps: results })))
}

fn mul_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let (coeff, f) = as_independent_default(integrand, symbol)?;
    if coeff != Basic::one() {
        let next_step: RuleRef = solver.steps(&f, symbol)?;
        return Ok(Some(rule(
            integrand,
            symbol,
            Kind::ConstantTimes {
                constant: coeff,
                other: f,
                substep: next_step,
            },
        )));
    }
    Ok(None)
}

/// A step of an integration by parts (`_Parts`).
#[derive(Clone)]
struct Parts {
    u: Basic,
    dv: Basic,
    v: Basic,
    du: Basic,
    v_step: RuleRef,
}

#[derive(Clone, Copy, PartialEq)]
enum Liate {
    Log,
    InverseTrig,
    Algebraic,
    SinCos,
    Exp,
}

fn pull_out(rule_: Liate, integrand: &Basic, symbol: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let functions: Vec<Function> = match rule_ {
        Liate::Log => vec![Function::Log],
        Liate::InverseTrig => INVERSE_TRIG.to_vec(),
        Liate::SinCos => vec![Function::Sin, Function::Cos],
        Liate::Exp => vec![Function::Exp],
        Liate::Algebraic => {
            let integrand: Basic = crate::polys::rationaltools::together(&cancel(integrand)?, false, true)?;
            let algebraic: Vec<Basic> = if matches!(integrand.head(), Head::Piecewise) || !integrand.is_mul() {
                Vec::new()
            } else {
                integrand
                    .args()
                    .iter()
                    .filter(|a| is_algebraic_expr(a, symbol))
                    .cloned()
                    .collect()
            };
            if !algebraic.is_empty() {
                let u: Basic = mul_n(&algebraic)?;
                let dv: Basic = cancel(&d2(&integrand, &u)?)?;
                return Ok(Some((u, dv)));
            }
            return Ok(None);
        }
    };
    if functions
        .iter()
        .any(|f| integrand.preorder().iter().any(|n| is_fn(n, &[*f])))
    {
        let args: Vec<Basic> = integrand
            .args()
            .iter()
            .filter(|a| is_fn(a, &functions))
            .cloned()
            .collect();
        if !args.is_empty() {
            let u: Basic = mul_n(&args)?;
            let dv: Basic = d2(integrand, &u)?;
            return Ok(Some((u, dv)));
        }
    }
    Ok(None)
}

fn has_function(e: &Basic, fs: &[Function]) -> bool {
    e.preorder().iter().any(|n| is_fn(n, fs))
}

/// The LIATE choice of `u` and the steps of `v` (`_parts_rule`).
fn parts_rule(integrand: &Basic, symbol: &Basic, solver: &mut IntegralSolver) -> PolyResult<Option<Parts>> {
    let liate_rules: [Liate; 5] = [
        Liate::Log,
        Liate::InverseTrig,
        Liate::Algebraic,
        Liate::SinCos,
        Liate::Exp,
    ];
    let dummy: Basic = fresh("temporary");
    let mut integrand: Basic = integrand.clone();
    let mut log_or_inverse: Vec<Function> = vec![Function::Log];
    log_or_inverse.extend(INVERSE_TRIG);
    if is_fn(&integrand, &log_or_inverse) {
        integrand = mul_n(&[dummy.clone(), integrand])?;
    }
    for (index, rule_) in liate_rules.iter().enumerate() {
        let Some((u, dv)) = pull_out(*rule_, &integrand, symbol)? else {
            continue;
        };
        if !u.free_symbols().contains(symbol) && !u.has(&dummy) {
            return Ok(None);
        }
        let u: Basic = subs(&u, &dummy, &Basic::one())?;
        let dv: Basic = subs(&dv, &dummy, &Basic::one())?;
        if *rule_ == Liate::Algebraic && !is_polynomial(&u, symbol) {
            return Ok(None);
        }
        if is_fn(&u, &[Function::Log]) {
            let rec_dv: Basic = d2(&int(1), &dv)?;
            if is_polynomial(&rec_dv, symbol) && degree_in(&rec_dv, symbol)? == 1 {
                return Ok(None);
            }
        }
        if *rule_ == Liate::Algebraic {
            let mut orthogonal_fns: Vec<Function> = ORTHOGONAL.iter().map(|(f, _)| *f).collect();
            orthogonal_fns.dedup();
            if matches!(dv.head(), Head::Derivative) || has_function(&dv, &TRIG) || is_fn(&dv, &orthogonal_fns) {
                let v_step: RuleRef = solver.steps(&dv, symbol)?;
                if v_step.contains_dont_know() {
                    return Ok(None);
                }
                let du: Basic = diff(&u, symbol)?;
                let v: Basic = v_step.eval()?;
                return Ok(Some(Parts { u, dv, v, du, v_step }));
            }
        }
        let mut accept: bool = false;
        if index < 2 {
            accept = true;
        } else if *rule_ == Liate::Algebraic
            && !dv.args().is_empty()
            && dv
                .args()
                .iter()
                .all(|a| is_fn(a, &[Function::Sin, Function::Cos, Function::Exp]))
        {
            accept = true;
        } else {
            for lrule in &liate_rules[index + 1..] {
                if let Some((r0, _)) = pull_out(*lrule, &integrand, symbol)? {
                    if same(&subs(&r0, &dummy, &Basic::one())?, &dv)? {
                        accept = true;
                        break;
                    }
                }
            }
        }
        if accept {
            let du: Basic = diff(&u, symbol)?;
            let options = crate::simplify::simplify::Options::default();
            let v_step: RuleRef = solver.steps(&crate::simplify::simplify::simplify(&dv, &options)?, symbol)?;
            if !v_step.contains_dont_know() {
                let v: Basic = v_step.eval()?;
                return Ok(Some(Parts { u, dv, v, du, v_step }));
            }
        }
    }
    Ok(None)
}

/// The number and the rest of a product (`as_coeff_Mul`).
fn as_coeff_mul(e: &Basic) -> (Basic, Basic) {
    crate::core::expr::as_coeff_mul(e)
}

fn parts(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let symbol: &Basic = &integral.symbol;
    let (constant, integrand) = as_coeff_mul(&integral.integrand);
    let mut result: Option<Parts> = parts_rule(&integrand, symbol, solver)?;
    let mut steps: Vec<Parts> = Vec::new();
    if let Some(r) = result.clone() {
        let (mut u, mut v, mut du) = (r.u.clone(), r.v.clone(), r.du.clone());
        steps.push(r);
        if matches!(v.head(), Head::Integral) {
            return Ok(None);
        }
        if is_fn(
            &u,
            &[
                Function::Sin,
                Function::Cos,
                Function::Exp,
                Function::Sinh,
                Function::Cosh,
            ],
        ) && !solver.use_parts_u(&u, symbol)
        {
            return Ok(None);
        }
        for _ in 0..4 {
            let coefficient: Basic = cancel(&d2(&m2(&v, &du)?, &integrand)?)?;
            if coefficient == Basic::one() {
                break;
            }
            if !coefficient.free_symbols().contains(symbol) {
                let mut parts_rules: Vec<RuleRef> = Vec::new();
                for p in &steps {
                    parts_rules.push(rule(
                        &m2(&p.u, &p.dv)?,
                        symbol,
                        Kind::Parts {
                            u: p.u.clone(),
                            dv: p.dv.clone(),
                            v_step: p.v_step.clone(),
                            second_step: None,
                        },
                    ));
                }
                let sign: Basic = p2(&int(-1), &int(steps.len() as i64))?;
                let mut cyclic: RuleRef = rule(
                    &integrand,
                    symbol,
                    Kind::CyclicParts {
                        parts_rules,
                        coefficient: m2(&sign, &coefficient)?,
                    },
                );
                if constant != Basic::one() {
                    cyclic = rule(
                        &m2(&constant, &integrand)?,
                        symbol,
                        Kind::ConstantTimes {
                            constant: constant.clone(),
                            other: integrand.clone(),
                            substep: cyclic,
                        },
                    );
                }
                return Ok(Some(cyclic));
            }
            let (next_constant, next_integrand) = as_coeff_mul(&m2(&v, &du)?);
            result = parts_rule(&next_integrand, symbol, solver)?;
            match &result {
                Some(r) => {
                    u = m2(&r.u, &next_constant)?;
                    du = m2(&r.du, &next_constant)?;
                    v = r.v.clone();
                    steps.push(Parts {
                        u: u.clone(),
                        dv: r.dv.clone(),
                        v: v.clone(),
                        du: du.clone(),
                        v_step: r.v_step.clone(),
                    });
                }
                None => break,
            }
        }
    }
    fn make_second_step(
        steps: &[Parts],
        integrand: &Basic,
        symbol: &Basic,
        solver: &mut IntegralSolver,
    ) -> PolyResult<RuleRef> {
        if let Some(p) = steps.first() {
            let second: RuleRef = make_second_step(&steps[1..], &m2(&p.v, &p.du)?, symbol, solver)?;
            return Ok(rule(
                integrand,
                symbol,
                Kind::Parts {
                    u: p.u.clone(),
                    dv: p.dv.clone(),
                    v_step: p.v_step.clone(),
                    second_step: Some(second),
                },
            ));
        }
        solver.steps(integrand, symbol)
    }
    if let Some(p) = steps.first() {
        let second: RuleRef = make_second_step(&steps[1..], &m2(&p.v, &p.du)?, symbol, solver)?;
        let mut parts_: RuleRef = rule(
            &integrand,
            symbol,
            Kind::Parts {
                u: p.u.clone(),
                dv: p.dv.clone(),
                v_step: p.v_step.clone(),
                second_step: Some(second),
            },
        );
        if constant != Basic::one() {
            parts_ = rule(
                &m2(&constant, &integrand)?,
                symbol,
                Kind::ConstantTimes {
                    constant: constant.clone(),
                    other: integrand.clone(),
                    substep: parts_,
                },
            );
        }
        return Ok(Some(parts_));
    }
    Ok(None)
}

fn trig_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    if *integrand == f1(Function::Sin, symbol)? {
        return Ok(Some(rule(integrand, symbol, Kind::Sin)));
    }
    if *integrand == f1(Function::Cos, symbol)? {
        return Ok(Some(rule(integrand, symbol, Kind::Cos)));
    }
    if *integrand == p2(&f1(Function::Sec, symbol)?, &int(2))? {
        return Ok(Some(rule(integrand, symbol, Kind::Sec2)));
    }
    if *integrand == p2(&f1(Function::Csc, symbol)?, &int(2))? {
        return Ok(Some(rule(integrand, symbol, Kind::Csc2)));
    }
    let rewritten: Basic = if is_fn(integrand, &[Function::Tan]) {
        let arg: &Basic = &integrand.args()[0];
        d2(&f1(Function::Sin, arg)?, &f1(Function::Cos, arg)?)?
    } else if is_fn(integrand, &[Function::Cot]) {
        let arg: &Basic = &integrand.args()[0];
        d2(&f1(Function::Cos, arg)?, &f1(Function::Sin, arg)?)?
    } else if is_fn(integrand, &[Function::Sec]) {
        let arg: &Basic = &integrand.args()[0];
        let (se, ta) = (f1(Function::Sec, arg)?, f1(Function::Tan, arg)?);
        d2(&a2(&p2(&se, &int(2))?, &m2(&ta, &se)?)?, &a2(&se, &ta)?)?
    } else if is_fn(integrand, &[Function::Csc]) {
        let arg: &Basic = &integrand.args()[0];
        let (cs, co) = (f1(Function::Csc, arg)?, f1(Function::Cot, arg)?);
        d2(&a2(&p2(&cs, &int(2))?, &m2(&co, &cs)?)?, &a2(&cs, &co)?)?
    } else {
        return Ok(None);
    };
    let substep: RuleRef = solver.steps(&rewritten, symbol)?;
    Ok(Some(rule(integrand, symbol, Kind::Rewrite { rewritten, substep })))
}

fn trig_product_rule(integral: &IntegralInfo) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    if *integrand == m2(&f1(Function::Sec, symbol)?, &f1(Function::Tan, symbol)?)? {
        return Ok(Some(rule(integrand, symbol, Kind::SecTan)));
    }
    if *integrand == m2(&f1(Function::Csc, symbol)?, &f1(Function::Cot, symbol)?)? {
        return Ok(Some(rule(integrand, symbol, Kind::CscCot)));
    }
    Ok(None)
}

fn rational_parts(integrand: &Basic, symbol: &Basic) -> PolyResult<Option<(Vec<Basic>, Vec<Basic>)>> {
    if !(integrand.is_mul() || integrand.is_pow()) {
        return Ok(None);
    }
    let (numerator, denominator) = as_numer_denom(integrand)?;
    let (Some(num), Some(den)) = (coefficients(&numerator, symbol)?, coefficients(&denominator, symbol)?) else {
        return Ok(None);
    };
    Ok(Some((num, den)))
}

fn quadratic_denom_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let Some((num, den)) = rational_parts(integrand, symbol)? else {
        return Ok(None);
    };
    if num.len() > 2 || den.len() != 3 {
        return Ok(None);
    }
    if num.len() == 1 && den[1] == Basic::zero() {
        let (a, b, c) = (num[0].clone(), den[2].clone(), den[0].clone());
        let x_2: Basic = p2(symbol, &int(-2))?;
        let mut power: RuleRef = rule(
            &x_2,
            symbol,
            Kind::Power {
                base: symbol.clone(),
                exp: int(-2),
            },
        );
        let ab: Basic = d2(&a, &b)?;
        if ab != int(1) {
            power = rule(
                integrand,
                symbol,
                Kind::ConstantTimes {
                    constant: ab,
                    other: x_2.clone(),
                    substep: power,
                },
            );
        }
        if c == Basic::zero() {
            return Ok(Some(power));
        }
        let general_rule: RuleRef = rule(
            integrand,
            symbol,
            Kind::Arctan {
                a: a.clone(),
                b: b.clone(),
                c: c.clone(),
            },
        );
        if is(&b, Predicate::extended_real) == Some(true) && is(&c, Predicate::extended_real) == Some(true) {
            let positive_cond: Basic = gt(&d2(&c, &b)?, &int(0))?;
            if is_true_node(&positive_cond) {
                return Ok(Some(general_rule));
            }
            let coeff: Basic = d2(&a, &m2(&m2(&int(2), &sqrt(&ng(&c)?)?)?, &sqrt(&b)?)?)?;
            let constant: Basic = sqrt(&d2(&ng(&c)?, &b)?)?;
            let r1: Basic = d2(&int(1), &s2(symbol, &constant)?)?;
            let r2: Basic = d2(&int(1), &a2(symbol, &constant)?)?;
            let log_steps: Vec<RuleRef> = vec![
                rule(
                    &r1,
                    symbol,
                    Kind::Reciprocal {
                        base: s2(symbol, &constant)?,
                    },
                ),
                rule(
                    &ng(&r2)?,
                    symbol,
                    Kind::ConstantTimes {
                        constant: int(-1),
                        other: r2.clone(),
                        substep: rule(
                            &r2,
                            symbol,
                            Kind::Reciprocal {
                                base: a2(symbol, &constant)?,
                            },
                        ),
                    },
                ),
            ];
            let sub: Basic = s2(&r1, &r2)?;
            let mut rewritten: Basic = sub.clone();
            let mut negative_step: RuleRef = rule(&sub, symbol, Kind::Add { substeps: log_steps });
            if coeff != int(1) {
                rewritten = crate::core::mul::raw_mul(vec![coeff.clone(), sub.clone()]);
                negative_step = rule(
                    &rewritten,
                    symbol,
                    Kind::ConstantTimes {
                        constant: coeff,
                        other: sub,
                        substep: negative_step,
                    },
                );
            }
            negative_step = rule(
                integrand,
                symbol,
                Kind::Rewrite {
                    rewritten,
                    substep: negative_step,
                },
            );
            if is_false_node(&positive_cond) {
                return Ok(Some(negative_step));
            }
            return Ok(Some(rule(
                integrand,
                symbol,
                Kind::Piecewise {
                    subfunctions: vec![(general_rule, positive_cond), (negative_step, true_())],
                },
            )));
        }
        return Ok(Some(rule(
            integrand,
            symbol,
            Kind::Piecewise {
                subfunctions: vec![(general_rule, ne(&c, &int(0))?), (power, true_())],
            },
        )));
    }
    if num.len() == 1 {
        let (b, c) = (den[2].clone(), den[1].clone());
        if is(&b, Predicate::zero) == Some(true) {
            return Ok(None);
        }
        let u: Basic = fresh("u");
        let shift: Basic = d2(&c, &m2(&int(2), &b)?)?;
        let u_func: Basic = a2(symbol, &shift)?;
        let integrand2: Basic = subs(integrand, symbol, &s2(&u, &shift)?)?;
        let next_step: RuleRef = solver.steps(&integrand2, &u)?;
        return Ok(Some(rule(
            &integrand2,
            symbol,
            Kind::U {
                u_var: u,
                u_func,
                substep: next_step,
            },
        )));
    }
    let (a, b) = (num[1].clone(), num[0].clone());
    let (c, d, e) = (den[2].clone(), den[1].clone(), den[0].clone());
    if is(&c, Predicate::zero) == Some(true) {
        return Ok(None);
    }
    let denominator: Basic = a2(&a2(&m2(&c, &p2(symbol, &int(2))?)?, &m2(&d, symbol)?)?, &e)?;
    let constant: Basic = d2(&a, &m2(&int(2), &c)?)?;
    let numer1: Basic = a2(&m2(&m2(&int(2), &c)?, symbol)?, &d)?;
    let numer2: Basic = a2(&m2(&ng(&constant)?, &d)?, &b)?;
    let u: Basic = fresh("u");
    let reciprocal: RuleRef = solver.steps(&p2(&u, &int(-1))?, &u)?;
    let mut step1: RuleRef = rule(
        integrand,
        symbol,
        Kind::U {
            u_var: u,
            u_func: denominator.clone(),
            substep: reciprocal,
        },
    );
    let ratio1: Basic = d2(&numer1, &denominator)?;
    if constant != int(1) {
        step1 = rule(
            &d2(&m2(&constant, &numer1)?, &denominator)?,
            symbol,
            Kind::ConstantTimes {
                constant: constant.clone(),
                other: ratio1,
                substep: step1,
            },
        );
    }
    if is(&numer2, Predicate::zero) == Some(true) {
        return Ok(Some(step1));
    }
    let step2: RuleRef = solver.steps(&d2(&numer2, &denominator)?, symbol)?;
    let substeps: RuleRef = rule(
        integrand,
        symbol,
        Kind::Add {
            substeps: vec![step1, step2],
        },
    );
    let rewritten: Basic = a2(
        &d2(&m2(&constant, &numer1)?, &denominator)?,
        &d2(&numer2, &denominator)?,
    )?;
    Ok(Some(rule(
        integrand,
        symbol,
        Kind::Rewrite {
            rewritten,
            substep: substeps,
        },
    )))
}

/// The roots of a linear function made one power of `u`
/// (`_sqrt_linear`).
fn sqrt_linear(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, x) = (&integral.integrand, &integral.symbol);
    let (mut a0, mut b0) = (Basic::zero(), Basic::zero());
    let mut bases: Vec<Basic> = Vec::new();
    let mut qs: Vec<i64> = Vec::new();
    let mut bs: Vec<Basic> = Vec::new();
    let mut seen: Vec<Basic> = Vec::new();
    for pow_ in integrand.preorder() {
        if !pow_.is_pow() || seen.contains(&pow_) {
            continue;
        }
        seen.push(pow_.clone());
        let (base, exp_) = (&pow_.args()[0], &pow_.args()[1]);
        if is_integer_node(exp_) || !base.free_symbols().contains(x) {
            continue;
        }
        let Some(q) = exp_.as_rational() else {
            return Ok(None);
        };
        let Some((a1, b1)) = linear(base, x)? else { continue };
        if m2(&a0, &b1)? != m2(&a1, &b0)? || is(&d2(&b0, &b1)?, Predicate::nonnegative) != Some(true) {
            return Ok(None);
        }
        if b0 == Basic::zero() || is_true_node(&gt(&d2(&b0, &b1)?, &int(1))?) {
            a0 = a1.clone();
            b0 = b1.clone();
        }
        bases.push(base.clone());
        bs.push(b1);
        qs.push(i64::try_from(crate::numbers::denom(q)).unwrap_or(1));
    }
    if b0 == Basic::zero() {
        return Ok(None);
    }
    let q0: i64 = qs.iter().fold(1i64, |l, q| l / gcd(l, *q) * q);
    let u_x: Basic = p2(&a2(&a0, &m2(&b0, x)?)?, &rational(1, q0))?;
    let u: Basic = fresh("u");
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    for ((base, b), q) in bases.iter().zip(&bs).zip(&qs) {
        replacements.push((
            p2(base, &rational(1, *q))?,
            m2(&p2(&d2(b, &b0)?, &rational(1, *q))?, &p2(&u, &rational(q0, *q))?)?,
        ));
    }
    let substituted: Basic = subs(
        &subs_list(integrand, &replacements)?,
        x,
        &d2(&s2(&p2(&u, &int(q0))?, &a0)?, &b0)?,
    )?;
    let new_integrand: Basic = d2(&m2(&m2(&substituted, &p2(&u, &int(q0 - 1))?)?, &int(q0))?, &b0)?;
    let substep: RuleRef = solver.steps(&new_integrand, &u)?;
    if !substep.contains_dont_know() {
        let mut step: RuleRef = rule(
            integrand,
            x,
            Kind::U {
                u_var: u,
                u_func: u_x,
                substep,
            },
        );
        let generic_cond: Basic = ne(&b0, &int(0))?;
        if !is_true_node(&generic_cond) {
            let pairs: Vec<(Basic, Basic)> = bs.iter().map(|b| (b.clone(), Basic::zero())).collect();
            let simplified: Basic = subs_map(integrand, &pairs)?;
            let degenerate_step: RuleRef = solver.steps(&simplified, x)?;
            step = rule(
                integrand,
                x,
                Kind::Piecewise {
                    subfunctions: vec![(step, generic_cond), (degenerate_step, true_())],
                },
            );
        }
        return Ok(Some(step));
    }
    Ok(None)
}

fn sqrt_linear_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    sqrt_linear(integral, solver)
}

/// `f*(a + b*x + c*x**2)**(n/2)` (`_SqrtQuadratic`).
struct SqrtQuadraticMatch {
    a: Basic,
    b: Basic,
    c: Basic,
    f: Basic,
    coefficients: Vec<Basic>,
    n: i64,
}

fn match_sqrt_quadratic(integrand: &Basic, x: &Basic) -> PolyResult<Option<SqrtQuadraticMatch>> {
    for factor in mul_args(integrand) {
        let (base, e) = as_base_exp(&factor)?;
        let Some(q) = e.as_rational() else { continue };
        if q.is_int() || crate::numbers::denom(q) != Int::from(2) {
            continue;
        }
        let Some((a, b, c)) = quadratic(&base, x)? else {
            continue;
        };
        let f: Basic = d2(integrand, &factor)?;
        let Some(coefficients) = coefficients(&f, x)? else {
            continue;
        };
        let n: i64 = i64::try_from(crate::numbers::numer(q)).unwrap_or(0);
        return Ok(Some(SqrtQuadraticMatch {
            a,
            b,
            c,
            f,
            coefficients,
            n,
        }));
    }
    Ok(None)
}

fn polynomial(coefficients: &[Basic], x: &Basic) -> R {
    let mut terms: Vec<Basic> = Vec::new();
    for (k, c) in coefficients.iter().enumerate() {
        terms.push(m2(c, &p2(x, &int(k as i64))?)?);
    }
    add_n(&terms)
}

fn sqrt_quadratic_denom(a: &Basic, b: &Basic, c: &Basic, x: &Basic, numer: &[Basic], integrand: &Basic) -> Found {
    let q: Basic = a2(&a2(a, &m2(b, x)?)?, &m2(c, &p2(x, &int(2))?)?)?;
    let denom: Basic = sqrt(&q)?;
    let mut numer: Vec<Basic> = numer.to_vec();
    while numer.len() > 1 && numer[numer.len() - 1] == Basic::zero() {
        numer.pop();
    }
    let deg: usize = numer.len() - 1;
    if deg <= 1 {
        let (e, d) = if deg == 1 {
            (numer[1].clone(), numer[0].clone())
        } else {
            (Basic::zero(), numer[0].clone())
        };
        let big_a: Basic = d2(&e, &m2(&int(2), c)?)?;
        let big_b: Basic = s2(&d, &m2(&big_a, b)?)?;
        let pre_substitute: Basic = d2(&a2(&m2(&m2(&int(2), c)?, x)?, b)?, &denom)?;
        let mut constant_step: Option<RuleRef> = None;
        let mut linear_step: Option<RuleRef> = None;
        if big_a != Basic::zero() {
            let u: Basic = fresh("u");
            let pow_rule: RuleRef = rule(
                &d2(&int(1), &sqrt(&u)?)?,
                &u,
                Kind::Power {
                    base: u.clone(),
                    exp: ng(&half())?,
                },
            );
            let mut step: RuleRef = rule(
                &pre_substitute,
                x,
                Kind::U {
                    u_var: u,
                    u_func: q.clone(),
                    substep: pow_rule,
                },
            );
            if big_a != int(1) {
                step = rule(
                    &m2(&big_a, &pre_substitute)?,
                    x,
                    Kind::ConstantTimes {
                        constant: big_a.clone(),
                        other: pre_substitute.clone(),
                        substep: step,
                    },
                );
            }
            linear_step = Some(step);
        }
        if big_b != Basic::zero() {
            let reciprocal: Basic = d2(&int(1), &denom)?;
            constant_step = inverse_trig_generic(&reciprocal, x)?;
            if let Some(step) = constant_step.clone() {
                if big_b != int(1) {
                    constant_step = Some(rule(
                        &d2(&big_b, &denom)?,
                        x,
                        Kind::ConstantTimes {
                            constant: big_b.clone(),
                            other: reciprocal,
                            substep: step,
                        },
                    ));
                }
            }
        }
        if let (Some(l), Some(c_)) = (&linear_step, &constant_step) {
            let both: Basic = add_n(&[m2(&big_a, &pre_substitute)?, d2(&big_b, &denom)?])?;
            return Ok(Some(rule(
                integrand,
                x,
                Kind::Rewrite {
                    rewritten: both.clone(),
                    substep: rule(
                        &both,
                        x,
                        Kind::Add {
                            substeps: vec![l.clone(), c_.clone()],
                        },
                    ),
                },
            )));
        }
        return Ok(linear_step.or(constant_step));
    }
    let coeffs: Vec<Basic> = numer.iter().rev().cloned().collect();
    Ok(Some(rule(
        integrand,
        x,
        Kind::SqrtQuadraticDenom {
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            coeffs,
        },
    )))
}

fn sqrt_quadratic_generic_step(m: &SqrtQuadraticMatch, integrand: &Basic, x: &Basic) -> Found {
    let (a, b, c, n) = (&m.a, &m.b, &m.c, m.n);
    let q: Basic = a2(&a2(a, &m2(b, x)?)?, &m2(c, &p2(x, &int(2))?)?)?;
    if n > 0 {
        let Some(numer) = coefficients(
            &m2(&polynomial(&m.coefficients, x)?, &p2(&q, &int((n + 1).div_euclid(2)))?)?,
            x,
        )?
        else {
            return Ok(None);
        };
        let rewritten: Basic = d2(&polynomial(&numer, x)?, &sqrt(&q)?)?;
        let Some(substep) = sqrt_quadratic_denom(a, b, c, x, &numer, &rewritten)? else {
            return Ok(None);
        };
        return Ok(Some(rule(integrand, x, Kind::Rewrite { rewritten, substep })));
    }
    if n == -1 {
        return sqrt_quadratic_denom(a, b, c, x, &m.coefficients, integrand);
    }
    Ok(None)
}

fn sqrt_quadratic_generic(integrand: &Basic, x: &Basic) -> Found {
    let Some(m) = match_sqrt_quadratic(integrand, x)? else {
        return Ok(None);
    };
    sqrt_quadratic_generic_step(&m, integrand, x)
}

fn sqrt_quadratic_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, x) = (&integral.integrand, &integral.symbol);
    let Some(m) = match_sqrt_quadratic(integrand, x)? else {
        return Ok(None);
    };
    let generic_cond: Basic = ne(&m.c, &int(0))?;
    let mut degenerate_step: Option<RuleRef> = None;
    if !is_true_node(&generic_cond) {
        if is(&m.b, Predicate::zero) == Some(true) {
            degenerate_step = Some(solver.steps(&m2(&m.f, &p2(&sqrt(&m.a)?, &int(m.n))?)?, x)?);
        } else {
            let linear_integrand: Basic = m2(&m.f, &p2(&sqrt(&a2(&m.a, &m2(&m.b, x)?)?)?, &int(m.n))?)?;
            degenerate_step = sqrt_linear(&IntegralInfo::new(&linear_integrand, x), solver)?;
        }
    }
    let Some(generic_step) = sqrt_quadratic_generic_step(&m, integrand, x)? else {
        return Ok(None);
    };
    Ok(Some(add_degenerate_step(&generic_cond, generic_step, degenerate_step)))
}

fn hyperbolic_rule(integral: &IntegralInfo) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    if !is_fn(integrand, &HYPERBOLIC) || integrand.args()[0] != *symbol {
        return Ok(None);
    }
    if is_fn(integrand, &[Function::Sinh]) {
        return Ok(Some(rule(integrand, symbol, Kind::Sinh)));
    }
    if is_fn(integrand, &[Function::Cosh]) {
        return Ok(Some(rule(integrand, symbol, Kind::Cosh)));
    }
    let u: Basic = fresh("u");
    let one: Basic = int(1);
    let reciprocal =
        |u: &Basic| -> PolyResult<RuleRef> { Ok(rule(&d2(&one, u)?, u, Kind::Reciprocal { base: u.clone() })) };
    if is_fn(integrand, &[Function::Tanh]) {
        let rewritten: Basic = d2(&f1(Function::Sinh, symbol)?, &f1(Function::Cosh, symbol)?)?;
        let inner: RuleRef = rule(
            &rewritten,
            symbol,
            Kind::U {
                u_var: u.clone(),
                u_func: f1(Function::Cosh, symbol)?,
                substep: reciprocal(&u)?,
            },
        );
        return Ok(Some(rule(
            integrand,
            symbol,
            Kind::Rewrite {
                rewritten,
                substep: inner,
            },
        )));
    }
    if is_fn(integrand, &[Function::Coth]) {
        let rewritten: Basic = d2(&f1(Function::Cosh, symbol)?, &f1(Function::Sinh, symbol)?)?;
        let inner: RuleRef = rule(
            &rewritten,
            symbol,
            Kind::U {
                u_var: u.clone(),
                u_func: f1(Function::Sinh, symbol)?,
                substep: reciprocal(&u)?,
            },
        );
        return Ok(Some(rule(
            integrand,
            symbol,
            Kind::Rewrite {
                rewritten,
                substep: inner,
            },
        )));
    }
    let h: Basic = f1(Function::Tanh, &d2(symbol, &int(2))?)?;
    let h2: Basic = p2(&h, &int(2))?;
    if is_fn(integrand, &[Function::Sech]) {
        let rewritten: Basic = d2(&s2(&one, &h2)?, &a2(&one, &h2)?)?;
        let arctan: RuleRef = rule(
            &d2(&int(2), &a2(&p2(&u, &int(2))?, &one)?)?,
            &u,
            Kind::Arctan {
                a: int(2),
                b: Basic::one(),
                c: Basic::one(),
            },
        );
        let inner: RuleRef = rule(
            &rewritten,
            symbol,
            Kind::U {
                u_var: u.clone(),
                u_func: h.clone(),
                substep: arctan,
            },
        );
        return Ok(Some(rule(
            integrand,
            symbol,
            Kind::Rewrite {
                rewritten,
                substep: inner,
            },
        )));
    }
    let rewritten: Basic = d2(&s2(&one, &h2)?, &m2(&int(2), &h)?)?;
    let inner: RuleRef = rule(
        &rewritten,
        symbol,
        Kind::U {
            u_var: u.clone(),
            u_func: h.clone(),
            substep: reciprocal(&u)?,
        },
    );
    Ok(Some(rule(
        integrand,
        symbol,
        Kind::Rewrite {
            rewritten,
            substep: inner,
        },
    )))
}

/// `sin(a x)**m cos(b x)**n` and the like (`_TrigPowers`).
struct TrigPowers {
    a: Basic,
    b: Basic,
    m: Basic,
    n: Basic,
}

fn match_trig_powers(
    integrand: &Basic,
    symbol: &Basic,
    first_: Function,
    second: Function,
) -> PolyResult<Option<TrigPowers>> {
    let (mut a, mut b, mut m, mut n) = (Basic::zero(), Basic::zero(), Basic::zero(), Basic::zero());
    let (mut seen_first, mut seen_second) = (false, false);
    for factor in mul_args(integrand) {
        let (base, e) = as_base_exp(&factor)?;
        if !is_integer_node(&e) {
            return Ok(None);
        }
        if !is_fn(&base, &[first_, second]) {
            return Ok(None);
        }
        let Some(coefficient) = times(symbol, &base.args()[0])? else {
            return Ok(None);
        };
        if is_fn(&base, &[first_]) {
            if seen_first {
                return Ok(None);
            }
            seen_first = true;
            a = coefficient;
            m = e;
        } else {
            if seen_second {
                return Ok(None);
            }
            seen_second = true;
            b = coefficient;
            n = e;
        }
    }
    Ok(Some(TrigPowers { a, b, m, n }))
}

fn even(e: &Basic) -> bool {
    int_value(e).is_some_and(|k| k.rem_euclid(2) == 0)
}
fn odd(e: &Basic) -> bool {
    int_value(e).is_some_and(|k| k.rem_euclid(2) == 1)
}
fn at_least(e: &Basic, k: i64) -> bool {
    int_value(e).is_some_and(|v| v >= k)
}

fn trig_rewrite(integrand: &Basic, symbol: &Basic, rewritten: Basic, solver: &mut IntegralSolver) -> Found {
    if rewritten != *integrand {
        let substep: RuleRef = solver.steps(&rewritten, symbol)?;
        return Ok(Some(rule(integrand, symbol, Kind::Rewrite { rewritten, substep })));
    }
    Ok(None)
}

fn trig_sincos_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    if !has_function(integrand, &[Function::Sin, Function::Cos]) {
        return Ok(None);
    }
    let Some(tp) = match_trig_powers(integrand, symbol, Function::Sin, Function::Cos)? else {
        return Ok(None);
    };
    let (a, b, m, n) = (&tp.a, &tp.b, &tp.m, &tp.n);
    let one: Basic = int(1);
    let two: Basic = int(2);
    let rewritten: Basic = if even(m) && even(n) && at_least(m, 0) && at_least(n, 0) {
        m2(
            &p2(
                &d2(&s2(&one, &f1(Function::Cos, &m2(&m2(&two, a)?, symbol)?)?)?, &two)?,
                &d2(m, &two)?,
            )?,
            &p2(
                &d2(&a2(&one, &f1(Function::Cos, &m2(&m2(&two, b)?, symbol)?)?)?, &two)?,
                &d2(n, &two)?,
            )?,
        )?
    } else if odd(m) && at_least(m, 3) {
        m2(
            &m2(
                &p2(
                    &s2(&one, &p2(&f1(Function::Cos, &m2(a, symbol)?)?, &two)?)?,
                    &d2(&s2(m, &one)?, &two)?,
                )?,
                &f1(Function::Sin, &m2(a, symbol)?)?,
            )?,
            &p2(&f1(Function::Cos, &m2(b, symbol)?)?, n)?,
        )?
    } else if odd(n) && at_least(n, 3) {
        m2(
            &m2(
                &p2(
                    &s2(&one, &p2(&f1(Function::Sin, &m2(b, symbol)?)?, &two)?)?,
                    &d2(&s2(n, &one)?, &two)?,
                )?,
                &f1(Function::Cos, &m2(b, symbol)?)?,
            )?,
            &p2(&f1(Function::Sin, &m2(a, symbol)?)?, m)?,
        )?
    } else {
        return Ok(None);
    };
    trig_rewrite(integrand, symbol, rewritten, solver)
}

fn trig_tansec_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let symbol: &Basic = &integral.symbol;
    let integrand: Basic = subs_map(
        &integral.integrand,
        &[(d2(&int(1), &f1(Function::Cos, symbol)?)?, f1(Function::Sec, symbol)?)],
    )?;
    if !has_function(&integrand, &[Function::Tan, Function::Sec]) {
        return Ok(None);
    }
    let Some(tp) = match_trig_powers(&integrand, symbol, Function::Tan, Function::Sec)? else {
        return Ok(None);
    };
    let (a, b, m, n) = (&tp.a, &tp.b, &tp.m, &tp.n);
    let (one, two) = (int(1), int(2));
    let rewritten: Basic = if odd(m) {
        m2(
            &m2(
                &p2(
                    &s2(&p2(&f1(Function::Sec, &m2(a, symbol)?)?, &two)?, &one)?,
                    &d2(&s2(m, &one)?, &two)?,
                )?,
                &f1(Function::Tan, &m2(a, symbol)?)?,
            )?,
            &p2(&f1(Function::Sec, &m2(b, symbol)?)?, n)?,
        )?
    } else if even(n) && at_least(n, 4) {
        m2(
            &m2(
                &p2(
                    &a2(&one, &p2(&f1(Function::Tan, &m2(b, symbol)?)?, &two)?)?,
                    &s2(&d2(n, &two)?, &one)?,
                )?,
                &p2(&f1(Function::Sec, &m2(b, symbol)?)?, &two)?,
            )?,
            &p2(&f1(Function::Tan, &m2(a, symbol)?)?, m)?,
        )?
    } else if *m == int(2) && *n == int(0) {
        s2(&p2(&f1(Function::Sec, &m2(a, symbol)?)?, &two)?, &one)?
    } else {
        return Ok(None);
    };
    trig_rewrite(&integrand, symbol, rewritten, solver)
}

fn trig_cotcsc_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let symbol: &Basic = &integral.symbol;
    let (sn, tn, cs) = (
        f1(Function::Sin, symbol)?,
        f1(Function::Tan, symbol)?,
        f1(Function::Cos, symbol)?,
    );
    let integrand: Basic = subs_list(
        &integral.integrand,
        &[
            (d2(&int(1), &sn)?, f1(Function::Csc, symbol)?),
            (d2(&int(1), &tn)?, f1(Function::Cot, symbol)?),
            (d2(&cs, &tn)?, f1(Function::Cot, symbol)?),
        ],
    )?;
    if !has_function(&integrand, &[Function::Cot, Function::Csc]) {
        return Ok(None);
    }
    let Some(tp) = match_trig_powers(&integrand, symbol, Function::Cot, Function::Csc)? else {
        return Ok(None);
    };
    let (a, b, m, n) = (&tp.a, &tp.b, &tp.m, &tp.n);
    let (one, two) = (int(1), int(2));
    let rewritten: Basic = if odd(m) {
        m2(
            &m2(
                &p2(
                    &s2(&p2(&f1(Function::Csc, &m2(a, symbol)?)?, &two)?, &one)?,
                    &d2(&s2(m, &one)?, &two)?,
                )?,
                &f1(Function::Cot, &m2(a, symbol)?)?,
            )?,
            &p2(&f1(Function::Csc, &m2(b, symbol)?)?, n)?,
        )?
    } else if even(n) && at_least(n, 4) {
        m2(
            &m2(
                &p2(
                    &a2(&one, &p2(&f1(Function::Cot, &m2(b, symbol)?)?, &two)?)?,
                    &s2(&d2(n, &two)?, &one)?,
                )?,
                &p2(&f1(Function::Csc, &m2(b, symbol)?)?, &two)?,
            )?,
            &p2(&f1(Function::Cot, &m2(a, symbol)?)?, m)?,
        )?
    } else {
        return Ok(None);
    };
    trig_rewrite(&integrand, symbol, rewritten, solver)
}

fn trig_sindouble_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let double: Basic = f1(Function::Sin, &m2(&int(2), symbol)?)?;
    let mut factors: Vec<Basic> = mul_args(integrand);
    let Some(position) = factors.iter().position(|f| *f == double) else {
        return Ok(None);
    };
    factors.remove(position);
    if mul_n(&factors)?.has(&double) {
        return Ok(None);
    }
    let sin_double: Basic = d2(
        &m2(&m2(&int(2), &f1(Function::Sin, symbol)?)?, &f1(Function::Cos, symbol)?)?,
        &double,
    )?;
    Ok(Some(solver.steps(&m2(integrand, &sin_double)?, symbol)?))
}

fn trig_powers_products_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    first(
        &[
            &trig_sincos_rule,
            &trig_tansec_rule,
            &trig_cotcsc_rule,
            &trig_sindouble_rule,
        ],
        integral,
        solver,
    )
}

fn known_sign(e: &Basic) -> PolyResult<(bool, bool)> {
    let number: bool = crate::core::assumptions::is_number(e);
    let positive: bool = is(e, Predicate::positive) == Some(true) || (number && is_true_node(&gt(e, &int(0))?));
    let negative: bool = is(e, Predicate::negative) == Some(true) || (number && is_true_node(&lt(e, &int(0))?));
    Ok((positive, negative))
}

fn trig_substitution_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let theta: Basic = fresh("theta");
    let mut matches: Vec<(Basic, Basic)> = Vec::new();
    for node in integrand.preorder() {
        if !node.is_add() {
            continue;
        }
        let Some(c) = coefficients(&node, symbol)? else {
            continue;
        };
        if c.len() != 3 || c[1] != Basic::zero() {
            continue;
        }
        let (a, b) = (c[0].clone(), c[2].clone());
        if is(&a, Predicate::zero) == Some(true)
            || is(&b, Predicate::zero) == Some(true)
            || matches.contains(&(a.clone(), b.clone()))
        {
            continue;
        }
        matches.push((a, b));
    }
    for (a, b) in &matches {
        let (a_positive, a_negative) = known_sign(a)?;
        let (b_positive, b_negative) = known_sign(b)?;
        let (x_func, restriction): (Basic, Basic) = if a_positive && b_positive {
            (m2(&d2(&sqrt(a)?, &sqrt(b)?)?, &f1(Function::Tan, &theta)?)?, true_())
        } else if a_positive && b_negative {
            let constant: Basic = d2(&sqrt(a)?, &sqrt(&ng(b)?)?)?;
            (
                m2(&constant, &f1(Function::Sin, &theta)?)?,
                and2(&gt(symbol, &ng(&constant)?)?, &lt(symbol, &constant)?),
            )
        } else if a_negative && b_positive {
            let constant: Basic = d2(&sqrt(&ng(a)?)?, &sqrt(b)?)?;
            (m2(&constant, &f1(Function::Sec, &theta)?)?, gt(symbol, &constant)?)
        } else {
            continue;
        };
        let mut substitutions: Vec<(Basic, Basic)> = Vec::new();
        for f in [
            Function::Sin,
            Function::Cos,
            Function::Tan,
            Function::Sec,
            Function::Csc,
            Function::Cot,
        ] {
            let ft: Basic = f1(f, &theta)?;
            substitutions.push((sqrt(&p2(&ft, &int(2))?)?, ft.clone()));
            substitutions.push((sqrt(&p2(&ft, &int(-2))?)?, d2(&int(1), &ft)?));
        }
        let mut replaced: Basic = trigsimp(&subs(integrand, symbol, &x_func)?)?;
        replaced = manual_subs(&replaced, &substitutions)?;
        if replaced.has(symbol) {
            continue;
        }
        replaced = trigsimp(&m2(&replaced, &manual_diff(&x_func, &theta)?)?)?;
        let map: HashMap<Basic, Basic> = [(d2(&int(1), &f1(Function::Cos, &theta)?)?, f1(Function::Sec, &theta)?)]
            .into_iter()
            .collect();
        replaced = crate::core::function::xreplace_evaluated(&replaced, &map)?;
        let substep: RuleRef = solver.steps(&replaced, &theta)?;
        if !substep.contains_dont_know() {
            return Ok(Some(rule(
                integrand,
                symbol,
                Kind::TrigSubstitution {
                    theta,
                    func: x_func,
                    rewritten: replaced,
                    substep,
                    restriction,
                },
            )));
        }
    }
    Ok(None)
}

fn heaviside_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let factors: Vec<Basic> = mul_args(integrand);
    for (i, factor) in factors.iter().enumerate() {
        if !is_fn(factor, &[Function::Heaviside]) {
            continue;
        }
        let Some((b, m)) = linear(&factor.args()[0], symbol)? else {
            continue;
        };
        let mut rest: Vec<Basic> = factors.clone();
        rest.remove(i);
        let g: Basic = mul_n(&rest)?;
        if g == Basic::zero() {
            return Ok(None);
        }
        let substep: RuleRef = solver.steps(&g, symbol)?;
        return Ok(Some(rule(
            integrand,
            symbol,
            Kind::Heaviside {
                harg: a2(&m2(&m, symbol)?, &b)?,
                ibnd: d2(&ng(&b)?, &m)?,
                substep,
            },
        )));
    }
    Ok(None)
}

fn dirac_delta_rule(integral: &IntegralInfo) -> Found {
    let (integrand, x) = (&integral.integrand, &integral.symbol);
    let n: Basic = if integrand.args().len() == 1 {
        Basic::zero()
    } else {
        integrand.args()[1].clone()
    };
    if !int_value(&n).is_some_and(|k| k >= 0) {
        return Ok(None);
    }
    let Some((a, b)) = linear(&integrand.args()[0], x)? else {
        return Ok(None);
    };
    let generic_cond: Basic = ne(&b, &int(0))?;
    let degenerate: Option<RuleRef> = if !is_true_node(&generic_cond) {
        Some(rule(
            &crate::functions::special::delta_functions::dirac_delta(&a, Some(&n))?,
            x,
            Kind::Constant,
        ))
    } else {
        None
    };
    let generic: RuleRef = rule(integrand, x, Kind::DiracDelta { n, a, b });
    Ok(Some(add_degenerate_step(&generic_cond, generic, degenerate)))
}

fn substitution_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let u_var: Basic = fresh("u");
    let substitutions = find_substitutions(integrand, symbol, &u_var)?;
    let mut ways: Vec<RuleRef> = Vec::new();
    for (u_func, c, substituted) in substitutions {
        let mut subrule: RuleRef = solver.steps(&substituted, &u_var)?;
        if subrule.contains_dont_know() {
            continue;
        }
        if !zero(&s2(&c, &int(1))?)? {
            let (_, denom) = as_numer_denom(&c)?;
            subrule = rule(
                &m2(&c, &substituted)?,
                &u_var,
                Kind::ConstantTimes {
                    constant: c.clone(),
                    other: substituted.clone(),
                    substep: subrule,
                },
            );
            if !denom.free_symbols().is_empty() {
                let mut pieces: Vec<(RuleRef, Basic)> = Vec::new();
                for expr in mul_args(&denom) {
                    if is(&expr, Predicate::zero) != Some(false) {
                        let substep: RuleRef =
                            solver.steps(&manual_subs(integrand, &[(expr.clone(), Basic::zero())])?, symbol)?;
                        pieces.push((substep, eq(&expr, &int(0))?));
                    }
                }
                pieces.push((subrule, true_()));
                subrule = rule(&substituted, symbol, Kind::Piecewise { subfunctions: pieces });
            }
        }
        ways.push(rule(
            integrand,
            symbol,
            Kind::U {
                u_var: u_var.clone(),
                u_func,
                substep: subrule,
            },
        ));
    }
    if ways.len() > 1 {
        return Ok(Some(rule(integrand, symbol, Kind::Alternative { alternatives: ways })));
    }
    Ok(ways.pop())
}

fn apart_(integrand: &Basic, symbol: &Basic) -> R {
    match crate::polys::partfrac::apart(integrand, Some(symbol), false) {
        Ok(v) => Ok(v),
        Err(e) if is_failure(&e) => polynomial_part(integrand, symbol),
        Err(e) => Err(e),
    }
}

fn polynomial_part(integrand: &Basic, symbol: &Basic) -> R {
    let (numerator, denominator) = as_numer_denom(integrand)?;
    let (Some(num), Some(den)) = (coefficients(&numerator, symbol)?, coefficients(&denominator, symbol)?) else {
        return Ok(integrand.clone());
    };
    if den.len() < 2 || num.len() < den.len() || den[den.len() - 1] == Basic::zero() {
        return Ok(integrand.clone());
    }
    let mut remainder: Vec<Basic> = num.clone();
    let mut quotient: Vec<Basic> = vec![Basic::zero(); num.len() - den.len() + 1];
    for k in (0..quotient.len()).rev() {
        let c: Basic = cancel(&d2(&remainder[k + den.len() - 1], &den[den.len() - 1])?)?;
        quotient[k] = c.clone();
        for (j, d) in den.iter().enumerate() {
            remainder[k + j] = cancel(&s2(&remainder[k + j], &m2(&c, d)?)?)?;
        }
    }
    let rest: Basic = polynomial(&remainder[..den.len() - 1], symbol)?;
    a2(&polynomial(&quotient, symbol)?, &d2(&rest, &denominator)?)
}

fn is_rational_(integrand: &Basic, _symbol: &Basic) -> bool {
    let syms: Vec<Basic> = integrand.free_symbols().into_iter().collect();
    is_rational_function_in(integrand, &syms)
}
fn cancel_(integrand: &Basic, _symbol: &Basic) -> R {
    match cancel(integrand) {
        Ok(v) => Ok(v),
        Err(e) if is_failure(&e) => Ok(integrand.clone()),
        Err(e) => Err(e),
    }
}
fn always(_: &Basic, _: &Basic) -> bool {
    true
}
fn expandable(integrand: &Basic, symbol: &Basic) -> bool {
    integrand.is_pow() || integrand.is_mul() || integrand.args().iter().all(|a| a.is_pow() || is_polynomial(a, symbol))
}
fn expanded(integrand: &Basic, _symbol: &Basic) -> R {
    expand(integrand)
}
fn different_trig_arguments(integrand: &Basic, _symbol: &Basic) -> bool {
    let mut arguments: Vec<Basic> = Vec::new();
    for n in integrand.preorder() {
        if is_fn(&n, &TRIG) && !arguments.contains(&n.args()[0]) {
            arguments.push(n.args()[0].clone());
        }
    }
    arguments.len() > 1
}
fn trig_expanded(integrand: &Basic, _symbol: &Basic) -> R {
    let hints = crate::core::function::Hints {
        trig: true,
        ..crate::core::function::Hints::DEFAULT
    };
    crate::core::function::expand_with(integrand, true, hints)
}

fn derivative_rule(integral: &IntegralInfo) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    if !matches!(integrand.head(), Head::Derivative) {
        return Ok(None);
    }
    let diff_variables: Vec<Basic> = integrand.args()[1..].iter().map(|vc| vc.args()[0].clone()).collect();
    let undifferentiated: &Basic = &integrand.args()[0];
    if undifferentiated.free_symbols().contains(symbol) {
        if diff_variables.contains(symbol) {
            return Ok(Some(rule(integrand, symbol, Kind::Derivative)));
        }
        return Ok(Some(rule(integrand, symbol, Kind::DontKnow)));
    }
    Ok(Some(rule(integrand, symbol, Kind::Constant)))
}

fn rewrites_rule(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let (integrand, symbol) = (&integral.integrand, &integral.symbol);
    let sec_form: Basic = d2(&int(1), &f1(Function::Cos, symbol)?)?;
    if *integrand == sec_form {
        let rewritten: Basic = subs(integrand, &sec_form, &f1(Function::Sec, symbol)?)?;
        let substep: RuleRef = solver.steps(&rewritten, symbol)?;
        return Ok(Some(rule(integrand, symbol, Kind::Rewrite { rewritten, substep })));
    }
    Ok(None)
}

/// The class of the integrand for the switch (`_key`).
fn key(integrand: &Basic, symbol: &Basic) -> &'static str {
    if !integrand.free_symbols().contains(symbol) {
        return "Number";
    }
    if is_symbol(integrand) {
        return "Symbol";
    }
    if is_fn(integrand, &TRIG) {
        return "TrigonometricFunction";
    }
    if orthogonal_index(integrand).is_some() {
        return "OrthogonalPolynomial";
    }
    if integrand.is_pow() {
        return "Pow";
    }
    if is_fn(integrand, &[Function::Exp]) {
        return "Exp";
    }
    if integrand.is_add() {
        return "Add";
    }
    if integrand.is_mul() {
        return "Mul";
    }
    if matches!(integrand.head(), Head::Derivative) {
        return "Derivative";
    }
    if is_fn(integrand, &[Function::Heaviside]) {
        return "Heaviside";
    }
    if is_fn(integrand, &[Function::DiracDelta]) {
        return "DiracDelta";
    }
    if is_fn(integrand, &[Function::Log]) {
        return "Log";
    }
    if is_fn(integrand, &INVERSE_TRIG) {
        return "InverseTrigonometricFunction";
    }
    "Other"
}

fn is_mul_or_pow(integral: &IntegralInfo) -> bool {
    matches!(key(&integral.integrand, &integral.symbol), "Mul" | "Pow")
}

fn is_mul_log_or_inverse(integral: &IntegralInfo) -> bool {
    matches!(
        key(&integral.integrand, &integral.symbol),
        "Mul" | "Log" | "InverseTrigonometricFunction"
    )
}

fn atomic(f: fn(&IntegralInfo) -> Found) -> impl Fn(&IntegralInfo, &mut IntegralSolver) -> Found {
    move |integral: &IntegralInfo, _solver: &mut IntegralSolver| f(integral)
}

fn conditioned(
    test: fn(&IntegralInfo) -> bool,
    strategy: &dyn Fn(&IntegralInfo, &mut IntegralSolver) -> Found,
    integral: &IntegralInfo,
    solver: &mut IntegralSolver,
) -> Found {
    if !test(integral) {
        return Ok(None);
    }
    strategy(integral, solver)
}

fn switch(name: &str, integral: &IntegralInfo, solver: &mut IntegralSolver) -> Option<Found> {
    let power = atomic(power_rule);
    Some(match name {
        "Pow" => first(
            &[&power, &inverse_trig_rule, &sqrt_linear_rule, &quadratic_denom_rule],
            integral,
            solver,
        ),
        "Symbol" => guarded(&power, integral, solver),
        "Exp" => guarded(&atomic(exp_rule), integral, solver),
        "Add" => guarded(&add_rule, integral, solver),
        "Mul" => first(
            &[
                &mul_rule,
                &atomic(trig_product_rule),
                &heaviside_rule,
                &quadratic_denom_rule,
                &sqrt_linear_rule,
                &sqrt_quadratic_rule,
            ],
            integral,
            solver,
        ),
        "Derivative" => guarded(&atomic(derivative_rule), integral, solver),
        "TrigonometricFunction" => guarded(&trig_rule, integral, solver),
        "Heaviside" => guarded(&heaviside_rule, integral, solver),
        "DiracDelta" => guarded(&atomic(dirac_delta_rule), integral, solver),
        "OrthogonalPolynomial" => guarded(&atomic(orthogonal_poly_rule), integral, solver),
        "Number" => guarded(&atomic(constant_rule), integral, solver),
        _ => return None,
    })
}

fn general(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    let alternatives_ = |integral: &IntegralInfo, solver: &mut IntegralSolver| -> Found {
        alternatives(
            &[
                &rewrites_rule,
                &substitution_rule,
                &|i: &IntegralInfo, s: &mut IntegralSolver| {
                    conditioned(is_mul_or_pow, &|i, s| rewriter(is_rational_, apart_, i, s), i, s)
                },
                &|i: &IntegralInfo, s: &mut IntegralSolver| {
                    conditioned(is_mul_or_pow, &|i, s| rewriter(always, cancel_, i, s), i, s)
                },
                &|i: &IntegralInfo, s: &mut IntegralSolver| conditioned(is_mul_log_or_inverse, &parts, i, s),
                &|i: &IntegralInfo, s: &mut IntegralSolver| {
                    conditioned(is_mul_or_pow, &|i, s| rewriter(expandable, expanded, i, s), i, s)
                },
                &trig_powers_products_rule,
                &|i: &IntegralInfo, s: &mut IntegralSolver| rewriter(different_trig_arguments, trig_expanded, i, s),
            ],
            integral,
            solver,
        )
    };
    first(
        &[
            &trig_rule,
            &atomic(hyperbolic_rule),
            &alternatives_,
            &|i: &IntegralInfo, s: &mut IntegralSolver| conditioned(is_mul_or_pow, &atomic(nested_pow_rule), i, s),
            &trig_substitution_rule,
        ],
        integral,
        solver,
    )
}

/// The steps of an integral, `None` when no rule applies
/// (`_integral_steps`; the solver turns `None` into a `DontKnowRule`).
fn integral_steps_of(integral: &IntegralInfo, solver: &mut IntegralSolver) -> Found {
    if let Some(found) = guarded(&atomic(special_function_rule), integral, solver)? {
        return Ok(Some(found));
    }
    if let Some(result) = switch(key(&integral.integrand, &integral.symbol), integral, solver) {
        if let Some(found) = result? {
            return Ok(Some(found));
        }
    }
    if let Some(found) = general(integral, solver)? {
        return Ok(Some(found));
    }
    Ok(Some(rule(&integral.integrand, &integral.symbol, Kind::DontKnow)))
}

/// The steps of the integral of `integrand` (`integral_steps`).
pub fn integral_steps(integrand: &Basic, symbol: &Basic) -> PolyResult<RuleRef> {
    IntegralSolver::new().solve(&IntegralInfo::new(integrand, symbol))
}

/// An antiderivative by the rules, with unevaluated integrals for the
/// parts not done (`manualintegrate`).
pub fn manualintegrate(f: &Basic, var: &Basic) -> R {
    let mut result: Basic = integral_steps(f, var)?.eval()?;
    if matches!(result.head(), Head::Piecewise) && result.args().len() == 2 {
        let (first_, second) = (&result.args()[0], &result.args()[1]);
        let cond: &Basic = &first_.args()[1];
        if matches!(cond.head(), Head::Relational(crate::core::basic::RelOp::Eq)) && is_true_node(&second.args()[1]) {
            result = piecewise(&[
                (second.args()[0].clone(), ne(&cond.args()[0], &cond.args()[1])?),
                (first_.args()[0].clone(), true_()),
            ])?;
        }
    }
    Ok(result)
}

/// The tree of the names of the rules, `Name(child, ...)`.
pub fn rule_names(r: &Rule) -> String {
    let children: Vec<String> = r.substeps().iter().map(|c| rule_names(c)).collect();
    if children.is_empty() {
        r.name().to_string()
    } else {
        format!("{}({})", r.name(), children.join(", "))
    }
}
