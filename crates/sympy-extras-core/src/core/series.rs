//! Port of the series machinery of the core: `sympy_extras/core/series.py`
//! (`series`) and the `as_leading_term`, `leadterm`, `nseries`,
//! `_eval_nseries`, `_eval_as_leading_term`, `taylor_term` and `dir`
//! methods of `core/expr.py`, `core/add.py`, `core/mul.py`,
//! `core/power.py`, `core/function.py` and of `Exp` and `Log`
//! (`functions/elementary/exponential.py`).
//!
//! What is not here yet is `NotImplemented` ("not ported"): the series of
//! the trigonometric, hyperbolic and special functions with rules of
//! their own, the asymptotic series, `lseries`, and the limits, which
//! [`limit_lite`] replaces by the value at the point for the expressions
//! continuous there from the right (`NotImplemented` otherwise). The
//! fallbacks of Python which catch `NotImplementedError` never catch a
//! "not ported" error, so that a missing part does not change a result.

#![allow(clippy::mutable_key_type, clippy::too_many_arguments)]

use super::basic::{Basic, Function, Head};
use crate::core::assumptions::Predicate;
use crate::core::expr::{factors_of, is, terms_of};
use crate::core::order::{get_o, getn, order, order_at_zero, remove_o};
use crate::core::symbol::{Dummy, Symbol};
use crate::error::{not_ported, pole_error, value_error, PolyError, PolyResult};
use crate::numbers::*;

type R = PolyResult<Basic>;

fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_method(e)
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn is_zero_number(e: &Basic) -> bool {
    e.is_number_class() && *e == Basic::zero()
}
fn powsimp_exp(e: &Basic) -> R {
    crate::simplify::powsimp::powsimp(
        e,
        true,
        crate::simplify::powsimp::Combine::Exp,
        false,
        &crate::simplify::powsimp::count_ops,
    )
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn hints() -> crate::core::function::Hints {
    crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    }
}
fn expand_logs(e: &Basic, force: bool) -> R {
    crate::core::function::expand_with(
        e,
        true,
        crate::core::function::Hints {
            log: true,
            force,
            ..hints()
        },
    )
}
fn expand_mul(e: &Basic) -> R {
    crate::core::function::expand_mul(e, true)
}
fn mexpand(e: &Basic) -> R {
    expand_mul(&crate::core::function::expand_multinomial(e, true)?)
}

/// Whether `e` is an order term (`is_Order`).
pub fn is_order(e: &Basic) -> bool {
    matches!(e.head(), Head::Order)
}

fn not_implemented(message: &str) -> PolyError {
    PolyError::NotImplemented(message.to_string())
}

/// Whether the error is one of the exceptions a Python `except` clause
/// of the kinds `value`, `not_implemented`, `type_`, `pole` catches; a
/// "not ported" error is never caught.
fn caught(e: &PolyError, value: bool, not_implemented: bool, type_: bool, pole: bool) -> bool {
    match e {
        PolyError::Value(_) => value,
        PolyError::NotImplemented(m) => not_implemented && !m.starts_with("not ported"),
        PolyError::Type(_) => type_,
        PolyError::Pole(_) => pole,
        _ => false,
    }
}

/// `a < b` for numbers (`_less`; `TypeError` when undecided).
pub fn less(a: &Basic, b: &Basic) -> PolyResult<bool> {
    match is(&sub(a, b)?, Predicate::extended_negative) {
        Some(v) => Ok(v),
        None => Err(PolyError::Type(format!(
            "cannot determine truth value of {} < {}",
            a, b
        ))),
    }
}

/// The integer ceiling of a number (`_ceiling_int`).
pub fn ceiling_int(e: &Basic) -> PolyResult<i64> {
    let c: Basic = crate::functions::elementary::integers::ceiling(e)?;
    match c.as_rational() {
        Some(r) if c.is_integer_value() => {
            i64::try_from(&numer(r)).map_err(|_| value_error("an order too large".to_string()))
        }
        _ => Err(PolyError::Type(format!("the order {} is not a number", e))),
    }
}

// -- the limit at a point, for continuous expressions

/// `limit(e, x, point)` from the right (`Expr.limit`): the
/// unevaluated `Limit` when it is not found.
pub fn limit_lite(e: &Basic, x: &Basic, point: &Basic) -> R {
    crate::core::limits::limit(e, x, point, "+")
}

// -- leading terms

/// The leading term of `e` in the symbols, one after the other
/// (`as_leading_term`).
pub fn as_leading_term(e: &Basic, symbols: &[Basic], logx: Option<&Basic>, cdir: &Basic) -> R {
    if symbols.len() > 1 {
        let mut c: Basic = e.clone();
        for v in symbols {
            c = as_leading_term(&c, std::slice::from_ref(v), logx, cdir)?;
        }
        return Ok(c);
    }
    let Some(x) = symbols.first() else { return Ok(e.clone()) };
    if !x.is_symbol() {
        return Err(value_error(format!("expecting a Symbol but got {}", x)));
    }
    if !e.free_symbols().contains(x) {
        return Ok(e.clone());
    }
    let obj: Basic = eval_as_leading_term(e, x, logx, cdir)?;
    powsimp_exp(&obj)
}

fn lt(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    as_leading_term(e, std::slice::from_ref(x), logx, cdir)
}

/// `leadterm`: the leading term as `(c, e)` with the term `c*x**e`.
pub fn leadterm(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut l: Basic = lt(e, x, logx, cdir)?;
    let d: Basic = Dummy::new("logx", &[]).0;
    let logarithm: Basic = log(x)?;
    if l.has(&logarithm) {
        l = subs(&l, &logarithm, &d)?;
    }
    let (c, ex) = crate::solvers::solvers::as_coeff_exponent(&l, x)?;
    if c.free_symbols().contains(x) {
        return Err(value_error(format!(
            "cannot compute leadterm({}, {}). The coefficient should have been free of {} but got {}",
            e, x, x, c
        )));
    }
    Ok((subs(&c, &d, &logarithm)?, ex))
}

/// The function classes whose SymPy counterpart has an
/// `_eval_as_leading_term` of its own (`_SYMPY_OWN_LEADING_TERM`).
const OWN_LEADING_TERM: &[&str] = &[
    "Acos",
    "Acosh",
    "Acot",
    "Acoth",
    "Acsc",
    "Acsch",
    "Arg",
    "Asec",
    "Asech",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "BesselI",
    "BesselJ",
    "BesselK",
    "BesselY",
    "Binomial",
    "Ceiling",
    "ChiNode",
    "CiNode",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "DiGamma",
    "EiNode",
    "Erf",
    "Erfc",
    "Erfi",
    "Exp",
    "Factorial",
    "Floor",
    "Frac",
    "FresnelC",
    "FresnelS",
    "Gamma",
    "LambertW",
    "Log",
    "PolyGamma",
    "Sec",
    "Sech",
    "ShiNode",
    "SiNode",
    "Sin",
    "SingularityFunctionNode",
    "Sinh",
    "Tan",
    "Tanh",
    "TriGamma",
    "Zeta",
    "Eis",
];

/// The function classes with a series rule of their own (the rules of
/// `functions/_series_rules.py` and the `_eval_nseries` methods).
const OWN_NSERIES: &[&str] = &[
    "Abs",
    "Arg",
    "BesselI",
    "BesselJ",
    "BesselK",
    "BesselY",
    "Ceiling",
    "ChiNode",
    "CiNode",
    "EiNode",
    "EllipticE",
    "EllipticK",
    "Expint",
    "Floor",
    "Frac",
    "Gamma",
    "LambertW",
    "LiNode",
    "LogGamma",
    "OffsetLiNode",
    "PolyLog",
    "ShiNode",
    "SiNode",
    "Sign",
    "Sinc",
    "SingularityFunctionNode",
    "Acos",
    "Acosh",
    "Acot",
    "Acoth",
    "Acsc",
    "Acsch",
    "Asec",
    "Asech",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "BesselIs",
    "BesselKs",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "Eis",
    "MinMaxBase",
    "Max",
    "Min",
    "Sec",
    "Sech",
    "Sin",
    "Sinh",
    "Tan",
    "Tanh",
    "AiryAi",
    "AiryBi",
];

/// The classes whose expansion is SymPy's generic one by derivatives or
/// Taylor terms (`generic_nseries`).
const GENERIC_NSERIES: &[&str] = &[
    "Adjoint",
    "AiryAiPrime",
    "AiryBiPrime",
    "Andre",
    "AssocLaguerre",
    "AssocLegendre",
    "Atan2",
    "Bell",
    "Bernoulli",
    "Beta",
    "BetaInc",
    "BetaIncRegularized",
    "Binomial",
    "ChebyshevT",
    "ChebyshevU",
    "Conjugate",
    "DiGamma",
    "DiracDelta",
    "DirichletEta",
    "DivisorSigma",
    "EllipticF",
    "Erf",
    "Erf2",
    "Erf2inv",
    "Erfc",
    "Erfcinv",
    "Erfi",
    "Erfinv",
    "Erfs",
    "Euler",
    "Factorial2",
    "FallingFactorial",
    "Fibonacci",
    "FresnelC",
    "FresnelS",
    "Gegenbauer",
    "Genocchi",
    "Hankel1",
    "Hankel2",
    "Harmonic",
    "Heaviside",
    "Hermite",
    "HermiteProb",
    "Im",
    "Jacobi",
    "JacobiSymbol",
    "KroneckerDeltaNode",
    "KroneckerSymbol",
    "Laguerre",
    "Legendre",
    "LegendreSymbol",
    "LerchPhi",
    "LowerGamma",
    "Lucas",
    "MarcumQ",
    "MathieuC",
    "MathieuCPrime",
    "MathieuS",
    "MathieuSPrime",
    "Mobius",
    "MultiGamma",
    "Partition",
    "PeriodicArgument",
    "PolarLift",
    "PolyGamma",
    "PrimeNu",
    "PrimeOmega",
    "PrimePi",
    "PrincipalBranch",
    "Re",
    "ReducedTotient",
    "Rem",
    "RiemannXi",
    "RisingFactorial",
    "SphericalHankel1",
    "SphericalHankel2",
    "SphericalJ",
    "SphericalY",
    "Stieltjes",
    "Subfactorial",
    "Totient",
    "Transpose",
    "TriGamma",
    "Tribonacci",
    "UDivisorSigma",
    "UpperGamma",
    "YnmNode",
    "Zeta",
    "ZnmNode",
];

/// The classes whose `nargs` is not `{1}` (the others take one argument).
const NOT_ONE_ARGUMENT: &[&str] = &[
    "AppellF1",
    "AssocLaguerre",
    "AssocLegendre",
    "Atan2",
    "Bell",
    "Bernoulli",
    "BesselI",
    "BesselIs",
    "BesselJ",
    "BesselK",
    "BesselKs",
    "BesselY",
    "Beta",
    "BetaInc",
    "BetaIncRegularized",
    "Binomial",
    "ChebyshevT",
    "ChebyshevU",
    "DiracDelta",
    "DirichletEta",
    "DivisorSigma",
    "EllipticE",
    "EllipticF",
    "EllipticPi",
    "Erf2",
    "Erf2inv",
    "Euler",
    "Expint",
    "FallingFactorial",
    "Fibonacci",
    "Gegenbauer",
    "Genocchi",
    "Hankel1",
    "Hankel2",
    "Harmonic",
    "Heaviside",
    "Hermite",
    "HermiteProb",
    "Jacobi",
    "JacobiSymbol",
    "KroneckerDeltaNode",
    "KroneckerSymbol",
    "Laguerre",
    "LambertW",
    "Legendre",
    "LegendreSymbol",
    "LerchPhi",
    "LeviCivitaNode",
    "LowerGamma",
    "MarcumQ",
    "MathieuC",
    "MathieuCPrime",
    "MathieuS",
    "MathieuSPrime",
    "Max",
    "Min",
    "MultiGamma",
    "PeriodicArgument",
    "PolyGamma",
    "PolyLog",
    "PrincipalBranch",
    "Rem",
    "RisingFactorial",
    "SingularityFunctionNode",
    "SphericalHankel1",
    "SphericalHankel2",
    "SphericalJ",
    "SphericalY",
    "Stieltjes",
    "Tribonacci",
    "UDivisorSigma",
    "UpperGamma",
    "YnmNode",
    "Zeta",
    "ZnmNode",
];

fn class_name(e: &Basic) -> String {
    e.class_name().to_string()
}

pub(crate) fn eval_as_leading_term(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    match e.head() {
        Head::Add => add_leading_term(e, x, logx, cdir),
        Head::Mul => {
            let factors: Vec<Basic> = e
                .args()
                .iter()
                .map(|t| lt(t, x, logx, cdir))
                .collect::<PolyResult<_>>()?;
            mul(&factors)
        }
        Head::Pow => pow_leading_term(e, x, logx, cdir),
        Head::Order => Ok(e.clone()),
        Head::Function(Function::Exp) => exp_leading_term(e, x, logx, cdir),
        Head::Function(Function::Log) => log_leading_term(e, x, logx, cdir),
        Head::Function(f) if super::series_trig::is_trig(*f) => super::series_trig::leading_term(e, *f, x, logx, cdir),
        Head::Function(f) if super::series_inverse::is_inverse(*f) => {
            super::series_inverse::leading_term(e, *f, x, logx, cdir)
        }
        Head::Function(_) => {
            let name: String = class_name(e);
            if OWN_LEADING_TERM.contains(&name.as_str()) {
                // the ported rules of the classes without a method of their own
                if let Some(found) = crate::functions::leading_terms::leading_term_rule(e, x, logx, cdir)? {
                    return Ok(found);
                }
            }
            generic_leading_term(e, x, logx)
        }
        Head::Undefined(_) => Ok(e.clone()),
        Head::Subs => {
            let (expr, variables, point) = (&e.args()[0], e.args()[1].args(), e.args()[2].args());
            if let Some(i) = point.iter().position(|p| p == x) {
                return lt(expr, &variables[i], None, &Basic::zero());
            }
            if variables.contains(x) {
                return Ok(e.clone());
            }
            Err(not_ported("the leading term of a Subs"))
        }
        Head::Derivative => {
            // the derivative of the first term of the series with a
            // nonzero derivative
            let counts: Vec<(Basic, u64)> = crate::core::function::derivative_counts(e);
            let mut terms = LSeries::new(&e.args()[0], Some(x), &Basic::zero(), "+", None, &Basic::zero())?;
            let mut d: Basic = Basic::zero();
            while let Some(term) = terms.next_term()? {
                d = crate::core::function::diff(&term, &counts)?;
                if !is_zero_number(&d) {
                    break;
                }
            }
            Ok(d)
        }
        Head::Hyper => hyper_leading_term(e, x, logx, cdir),
        _ if e.is_atom() => Ok(e.clone()),
        _ => Err(not_ported(&format!("the leading term of {}", e.class_name()))),
    }
}

/// `Function._eval_as_leading_term`: the function itself, unless an
/// argument vanishes at `x = 0`.
// PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper._eval_as_leading_term
/// 1 when the argument vanishes, otherwise SymPy's generic rule of
/// functions on the argument (`Hyper._eval_as_leading_term`).
fn hyper_leading_term(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let zero: Basic = Basic::zero();
    let arg: &Basic = &e.args()[2];
    let mut x0: Basic = crate::core::subs::subs(arg, x, &zero)?;
    if matches!(x0.head(), Head::NaN) {
        let negative: bool = is(&crate::functions::elementary::complexes::re(cdir)?, Predicate::negative) == Some(true);
        x0 = crate::core::limits::limit(arg, x, &zero, if negative { "-" } else { "+" })?;
    }
    if x0 == zero {
        return Ok(Basic::one());
    }
    let lead: Basic = as_leading_term(arg, std::slice::from_ref(x), logx, &zero)?;
    let o: Basic = order(&Basic::one(), &[(x.clone(), zero.clone())])?;
    if lead.free_symbols().contains(x) && crate::core::order::contains(&o, &lead)? == Some(true) {
        return Err(PolyError::NotImplemented(
            "hyper has no _eval_as_leading_term routine".to_string(),
        ));
    }
    Ok(e.clone())
}

/// The value of `arg` at `x = 0`; where the substitution gives `nan`
/// (`sin(x)/x`) and the leading term is a finite constant, that constant
/// (`_value_at_zero`).
pub(crate) fn value_at_zero(arg: &Basic, x: &Basic) -> R {
    let x0: Basic = crate::polys::polytools::cancel(&subs(arg, x, &Basic::zero())?, &[])?;
    if matches!(x0.head(), Head::NaN) {
        let leading: Basic = as_leading_term(arg, std::slice::from_ref(x), None, &Basic::zero())?;
        if !leading.has(x) && is(&leading, Predicate::finite) == Some(true) {
            return Ok(leading);
        }
    }
    Ok(x0)
}

pub(crate) fn generic_leading_term_pub(f: &Basic, x: &Basic, logx: Option<&Basic>) -> R {
    generic_leading_term(f, x, logx)
}

fn generic_leading_term(f: &Basic, x: &Basic, logx: Option<&Basic>) -> R {
    let o: Basic = order_at_zero(&Basic::one(), x)?;
    for a in f.args() {
        let l: Basic = lt(a, x, logx, &Basic::zero())?;
        if l.free_symbols().contains(x) && crate::core::order::contains(&o, &l)? == Some(true) {
            return Err(not_implemented(&format!(
                "{} has no _eval_as_leading_term routine",
                f.class_name()
            )));
        }
    }
    Ok(f.clone())
}

fn order_contains(o: &Basic, e: &Basic) -> PolyResult<bool> {
    if !is_order(o) {
        return Err(PolyError::Type(format!("an order term is expected, got {}", o)));
    }
    match crate::core::order::contains(o, e)? {
        Some(v) => Ok(v),
        None => Err(PolyError::Type("contains did not evaluate to a bool".to_string())),
    }
}

fn add_leading_term(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let o: Basic = match get_o(e)? {
        Some(o) => o,
        None => order(&Basic::zero(), &[])?,
    };
    let mut old: Basic = remove_o(e)?;
    if old.preorder().iter().any(|n| matches!(n.head(), Head::Piecewise)) {
        old = crate::functions::elementary::piecewise::piecewise_fold(&old, None)?;
    }
    if e.args()
        .iter()
        .any(|a| matches!(a.head(), Head::Function(Function::Log)))
    {
        old = expand_logs(&old, false)?;
    }
    let expr: Basic = expand_mul(&old)?;
    if !expr.is_add() {
        return lt(&expr, x, logx, cdir);
    }
    let infinite: Vec<Basic> = expr
        .args()
        .iter()
        .filter(|t| yes(t, Predicate::infinite))
        .cloned()
        .collect();
    let dummy_logx: Basic = Dummy::new("logx", &[]).0;
    let logx_: &Basic = logx.unwrap_or(&dummy_logx);
    let leading_terms: Vec<Basic> = expr
        .args()
        .iter()
        .map(|t| lt(t, x, Some(logx_), cdir))
        .collect::<PolyResult<_>>()?;
    let mut smallest: Basic = order(&Basic::zero(), &[])?;
    let mut new_expr: Basic = Basic::zero();
    let decided: PolyResult<()> = (|| {
        for term in &leading_terms {
            let o_term: Basic = order_at_zero_or_free(term, x)?;
            if !order_contains(&smallest, &o_term)? {
                smallest = o_term;
                new_expr = term.clone();
            } else if order_contains(&o_term, &smallest)? {
                new_expr = add(&[new_expr.clone(), term.clone()])?;
            }
        }
        Ok(())
    })();
    if let Err(err) = decided {
        if matches!(err, PolyError::Type(_)) {
            return Ok(expr);
        }
        return Err(err);
    }
    if logx.is_none() {
        new_expr = subs(&new_expr, &dummy_logx, &log(x)?)?;
    }
    let mut is_zero: Option<bool> = is(&new_expr, Predicate::zero);
    if is_zero.is_none() {
        new_expr = cancel(&crate::simplify::trigsimp::trigsimp(
            &new_expr,
            crate::simplify::trigsimp::Method::Matching,
        )?)?;
        is_zero = is(&new_expr, Predicate::zero);
    }
    if is_zero == Some(true) {
        // the terms cancelled: the leading term of the series
        let mut n0: Basic = match getn(&smallest) {
            Ok(Some(v)) => v,
            Ok(None) => Basic::one(),
            Err(err) if caught(&err, false, true, false, false) => Basic::one(),
            Err(err) => return Err(err),
        };
        if n0.preorder().iter().any(|n| n.is_symbol()) {
            n0 = Basic::one();
        }
        let mut res: Basic = order(&Basic::one(), &[])?;
        let mut incr: i64 = 1;
        while is_order(&res) {
            if incr > 64 {
                // every term cancels up to a high order: a sum which
                // vanishes on this side without simplifying to 0
                return Ok(Basic::zero());
            }
            res = eval_nseries(&old, x, ceiling_int(&add(&[n0.clone(), int(incr)])?)?, logx, cdir)?;
            res = crate::simplify::trigsimp::trigsimp(
                &crate::simplify::powsimp::powsimp_default(&cancel(&res)?)?,
                crate::simplify::trigsimp::Method::Matching,
            )?;
            incr *= 2;
        }
        return lt(&res, x, logx, cdir);
    }
    if matches!(new_expr.head(), Head::NaN) {
        let mut parts: Vec<Basic> = infinite;
        parts.push(o);
        return Ok(crate::core::add::raw_add(parts));
    }
    Ok(new_expr)
}

/// `Order(term, x)`.
fn order_at_zero_or_free(term: &Basic, x: &Basic) -> R {
    order_at_zero(term, x)
}

fn is_exp1_power(e: &Basic) -> bool {
    e.is_pow() && matches!(e.args()[0].head(), Head::Exp1)
}

fn pow_leading_term(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let (b, ex) = (&e.args()[0], &e.args()[1]);
    if matches!(b.head(), Head::Exp1) {
        let arg: Basic = lt(ex, x, logx, &Basic::zero())?;
        let mut arg0: Basic = subs(&arg, x, &Basic::zero())?;
        if matches!(arg0.head(), Head::NaN) {
            arg0 = limit_lite(&arg, x, &Basic::zero())?;
        }
        if is(&arg0, Predicate::infinite) == Some(false) {
            return pow(b, &arg0);
        }
        return Err(pole_error(&format!("Cannot expand {} around 0", e)));
    }
    if ex.has(x) {
        return lt(&exp(&mul(&[ex.clone(), log(b)?])?)?, x, logx, cdir);
    }
    let f: Basic = match lt(b, x, logx, cdir) {
        Ok(f) => f,
        Err(PolyError::Pole(_)) => return Ok(e.clone()),
        Err(err) => return Err(err),
    };
    if !yes(ex, Predicate::integer) && yes(&f, Predicate::negative) && !f.has(x) {
        let ndir: Basic = dir(&sub(b, &f)?, x, cdir)?;
        let im_ndir: Basic = crate::functions::elementary::complexes::im(&ndir)?;
        if yes(&im_ndir, Predicate::negative) {
            // on the branch cut: exp(e*(log(f) - 2*pi*I)) == f**e*(-1)**(-2*e)
            let doit: Basic = crate::core::function::doit(&crate::core::power::pow_node(f.clone(), ex.clone()))?;
            return mul(&[doit, pow(&Basic::negative_one(), &mul(&[int(-2), ex.clone()])?)?]);
        }
        if yes(&im_ndir, Predicate::zero) {
            let log_leadterm: Basic = log_leading_term(&log(b)?, x, logx, cdir)?;
            if is(&log_leadterm, Predicate::infinite) == Some(false) {
                return exp(&mul(&[ex.clone(), log_leadterm])?);
            }
        }
    }
    pow(&f, ex)
}

fn exp_leading_term(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: Basic = lt(&cancel(&e.args()[0])?, x, logx, &Basic::zero())?;
    let mut arg0: Basic = subs(&arg, x, &Basic::zero())?;
    if matches!(arg.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if crate::core::accumbounds::is_bounds(&arg0) {
        let re_cdir: Basic = crate::functions::elementary::complexes::re(cdir)?;
        if yes(&re_cdir, Predicate::negative) {
            return exp(&neg(&arg0)?);
        }
        return exp(&arg0);
    }
    if matches!(arg0.head(), Head::NaN) {
        arg0 = limit_lite(&arg, x, &Basic::zero())?;
    }
    if is(&arg0, Predicate::infinite) == Some(false) {
        return exp(&arg0);
    }
    Err(pole_error(&format!("Cannot expand {} around 0", e)))
}

fn positive_dummy(name: &str) -> Basic {
    Dummy::new(name, &[(Predicate::positive, true)]).0
}

fn log_leading_term(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg0: Basic = crate::polys::rationaltools::together(&e.args()[0], false, true)?;
    let t: Basic = positive_dummy("t");
    let cdir: Basic = if is_zero_number(cdir) {
        Basic::one()
    } else {
        cdir.clone()
    };
    let z: Basic = subs(&arg0, x, &mul(&[cdir.clone(), t.clone()])?)?;
    let (mut c, ex) = match leadterm(&z, &t, logx, &Basic::one()) {
        Ok(v) => v,
        Err(PolyError::Value(_)) => return log(&lt(&arg0, x, logx, &cdir)?),
        Err(err) => return Err(err),
    };
    if c.has(&t) {
        c = subs(&c, &t, &div(x, &cdir)?)?;
        if ex != Basic::zero() {
            return Err(pole_error(&format!("Cannot expand {} around 0", e)));
        }
        return log(&c);
    }
    if c == Basic::one() && ex == Basic::zero() {
        return lt(&sub(&arg0, &Basic::one())?, x, logx, &Basic::zero());
    }
    let logx_: Basic = match logx {
        Some(l) => l.clone(),
        None => log(x)?,
    };
    let mut res: Basic = add(&[
        log(&c)?,
        neg(&mul(&[ex.clone(), log(&cdir)?])?)?,
        mul(&[ex.clone(), logx_])?,
    ])?;
    if yes(&c, Predicate::negative) && crate::functions::elementary::complexes::im(&z)? != Basic::zero() {
        res = add(&[res, branch_jump(&z, &t)?])?;
    }
    Ok(res)
}

/// `-2*I*pi*Heaviside(-im(c), 0)` for the first non-real term `c*t**e`
/// among the first five of the series of `z` (the jump of `log` across
/// its branch cut), 0 when they are real (`_branch_jump`).
fn branch_jump(z: &Basic, t: &Basic) -> R {
    let mut terms = LSeries::new(z, Some(t), &Basic::zero(), "+", None, &Basic::zero())?;
    let mut term: Basic = Basic::zero();
    let mut i: usize = 0;
    let mut index: usize = 0;
    while let Some(next) = terms.next_term()? {
        i = index;
        term = next;
        if !yes(&term, Predicate::real) || i == 5 {
            break;
        }
        index += 1;
    }
    if i < 5 {
        let (coeff, _) = crate::solvers::solvers::as_coeff_exponent(&term, t)?;
        let h: Basic = crate::functions::special::delta_functions::heaviside(
            &mul(&[
                Basic::negative_one(),
                crate::functions::elementary::complexes::im(&coeff)?,
            ])?,
            Some(&Basic::zero()),
        )?;
        return mul(&[int(-2), Basic::imaginary_unit(), Basic::pi(), h]);
    }
    Ok(Basic::zero())
}

// -- lseries

/// The terms of `_eval_lseries`: the series of increasing orders, the
/// new terms of each.
struct EvalLSeries {
    e: Basic,
    x: Basic,
    logx: Option<Basic>,
    cdir: Basic,
    k: i64,
    last: Option<Basic>,
    done: bool,
}

impl EvalLSeries {
    fn next_term(&mut self) -> PolyResult<Option<Basic>> {
        if self.done {
            return Ok(None);
        }
        let logx: Option<&Basic> = self.logx.as_ref();
        let Some(e) = self.last.clone() else {
            let mut series: Basic = eval_nseries(&self.e, &self.x, self.k, logx, &self.cdir)?;
            while is_order(&series) {
                self.k += 1;
                series = eval_nseries(&self.e, &self.x, self.k, logx, &self.cdir)?;
            }
            let first: Basic = remove_o(&series)?;
            if is_zero_number(&first) {
                self.done = true;
            }
            self.last = Some(first.clone());
            return Ok(Some(first));
        };
        loop {
            self.k += 1;
            let series: Basic = remove_o(&eval_nseries(&self.e, &self.x, self.k, logx, &self.cdir)?)?;
            if e != series {
                self.last = Some(series.clone());
                return Ok(Some(sub(&series, &e)?));
            }
            if cancel(&sub(&series, &self.e)?)? == Basic::zero() {
                self.done = true;
                return Ok(None);
            }
        }
    }
}

/// The terms of the series one by one (`lseries`): the point moved to
/// zero from the right as `series` does, the sums of `_eval_lseries`
/// split by increasing order (`_yield_lseries`).
pub struct LSeries {
    inner: Option<EvalLSeries>,
    x: Basic,
    /// the substitutions moving the terms back, the innermost last
    back: Vec<Vec<(Basic, Basic)>>,
    pending: std::collections::VecDeque<Basic>,
    /// the expression itself, free of `x`, yielded once
    constant: Option<Basic>,
}

impl LSeries {
    pub fn new(
        e: &Basic,
        x: Option<&Basic>,
        x0: &Basic,
        dir: &str,
        logx: Option<&Basic>,
        cdir: &Basic,
    ) -> PolyResult<LSeries> {
        let x: Basic = match x {
            Some(v) => v.clone(),
            None => {
                let syms = e.free_symbols();
                if syms.len() > 1 {
                    return Err(value_error("x must be given for multivariate functions.".to_string()));
                }
                match syms.into_iter().next() {
                    Some(s) => s,
                    None => {
                        return Ok(LSeries {
                            inner: None,
                            x: e.clone(),
                            back: vec![],
                            pending: Default::default(),
                            constant: Some(e.clone()),
                        })
                    }
                }
            }
        };
        if !e.free_symbols().contains(&x) {
            return Ok(LSeries {
                inner: None,
                x,
                back: vec![],
                pending: Default::default(),
                constant: Some(e.clone()),
            });
        }
        let (_, cdir) = direction(x0, dir, cdir)?;
        let mut e: Basic = e.clone();
        let mut back: Vec<Vec<(Basic, Basic)>> = Vec::new();
        let mut logx: Option<Basic> = logx.cloned();
        let mut cdir: Basic = cdir;
        if !is_zero_number(x0) && yes(x0, Predicate::infinite) {
            e = subs(&e, &x, &div(&cdir, &x)?)?;
            back.push(vec![(x.clone(), div(&cdir, &x)?)]);
            logx = None;
            cdir = Basic::one();
        } else if !is_zero_number(x0) || cdir != Basic::one() {
            e = crate::core::subs::subs_mapping(
                &e,
                &[(x.clone(), add(&[x0.clone(), mul(&[cdir.clone(), x.clone()])?])?)],
            )?;
            back.push(vec![(x.clone(), sub(&div(&x, &cdir)?, &div(x0, &cdir)?)?)]);
            cdir = Basic::one();
        }
        let mut var: Basic = x.clone();
        if is(&x, Predicate::positive).is_none() && is(&x, Predicate::negative).is_none() {
            let xpos: Basic = positive_dummy("x");
            e = subs(&e, &x, &xpos)?;
            back.push(vec![(xpos.clone(), x.clone())]);
            var = xpos;
        }
        let inner = EvalLSeries {
            e: remove_o(&e)?,
            x: var.clone(),
            logx,
            cdir,
            k: 0,
            last: None,
            done: false,
        };
        Ok(LSeries {
            inner: Some(inner),
            x: var,
            back,
            pending: Default::default(),
            constant: None,
        })
    }

    fn moved_back(&self, term: Basic) -> R {
        let mut t: Basic = term;
        for step in self.back.iter().rev() {
            t = crate::core::subs::subs_mapping(&t, step)?;
        }
        Ok(t)
    }

    /// The next term, `None` at the end.
    pub fn next_term(&mut self) -> PolyResult<Option<Basic>> {
        if let Some(c) = self.constant.take() {
            return Ok(Some(c));
        }
        if let Some(t) = self.pending.pop_front() {
            return Ok(Some(self.moved_back(t)?));
        }
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        let Some(si) = inner.next_term()? else {
            self.inner = None;
            return Ok(None);
        };
        if !si.is_add() {
            return Ok(Some(self.moved_back(si)?));
        }
        // _yield_lseries: the terms of the sum by increasing order
        let x: Basic = self.x.clone();
        let mut yielded: Basic = Basic::zero();
        let mut o: Basic = mul(&[order(&si, &[(x.clone(), Basic::zero())])?, x.clone()])?;
        let mut ndid: usize = 0;
        let ndo: usize = si.args().len();
        loop {
            let d: Basic = remove_o(&add(&[sub(&si, &yielded)?, o.clone()])?)?;
            o = mul(&[o, x.clone()])?;
            if is_zero_number(&d) || is_order(&d) {
                continue;
            }
            ndid += if d.is_add() { d.args().len() } else { 1 };
            self.pending.push_back(d.clone());
            if ndid == ndo {
                break;
            }
            yielded = add(&[yielded, d])?;
        }
        let first: Basic = self.pending.pop_front().expect("a term");
        Ok(Some(self.moved_back(first)?))
    }
}

/// The direction from which `e` approaches its value at `x = 0` along
/// `cdir` (`dir`).
pub fn dir(e: &Basic, x: &Basic, cdir: &Basic) -> R {
    if yes(e, Predicate::zero) {
        return Ok(Basic::zero());
    }
    let mut minexp: i64 = 0;
    let mut arg: Basic = e.clone();
    let mut coeff: Basic = Basic::zero();
    while !is_zero_number(&arg) {
        minexp += 1;
        arg = crate::core::function::derivative(&arg, x)?;
        coeff = subs(&arg, x, &Basic::zero())?;
        if matches!(coeff.head(), Head::NaN) {
            coeff = limit_lite(&arg, x, &Basic::zero())?;
        }
        if matches!(coeff.head(), Head::ComplexInfinity) {
            coeff = match leadterm(&arg, x, None, &Basic::zero()) {
                Ok((c, _)) if !c.has(&log(x)?) => c,
                Ok(_) | Err(PolyError::Value(_)) => limit_lite(&arg, x, &Basic::zero())?,
                Err(err) => return Err(err),
            };
        }
        if coeff != Basic::zero() {
            break;
        }
    }
    mul(&[coeff, pow(cdir, &int(minexp))?])
}

// -- nseries

/// `Expr.nseries(x, 0, n, '+', logx, cdir)`: the expression itself when
/// free of `x`.
pub fn nseries(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    if !e.free_symbols().contains(x) {
        return Ok(e.clone());
    }
    eval_nseries(e, x, n, logx, cdir)
}

/// `_eval_nseries`: the terms of the series up to `O(x**n)` at zero from
/// the right.
pub fn eval_nseries(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    match e.head() {
        Head::Add => {
            let terms: Vec<Basic> = e
                .args()
                .iter()
                .map(|t| nseries(t, x, n, logx, cdir))
                .collect::<PolyResult<_>>()?;
            add(&terms)
        }
        Head::Mul => mul_nseries(e, x, n, logx, cdir),
        Head::Pow => pow_nseries(e, x, n, logx, cdir),
        Head::Order => Ok(e.clone()),
        Head::Function(Function::Exp) => exp_nseries(e, x, n, logx, cdir),
        Head::Function(Function::Log) => log_nseries(e, x, n, logx, cdir),
        Head::Function(f) if super::series_trig::is_trig(*f) => super::series_trig::nseries(e, *f, x, n, logx, cdir),
        Head::Function(f) if super::series_inverse::is_inverse(*f) => {
            super::series_inverse::nseries_of(e, *f, x, n, logx, cdir)
        }
        Head::Function(_) => {
            let name: String = class_name(e);
            if OWN_NSERIES.contains(&name.as_str()) {
                if let Some(found) = crate::functions::series_rules::series_rule(e, x, n, logx, cdir)? {
                    return Ok(found);
                }
                if !SERIES_RULES.contains(&name.as_str()) {
                    return Err(not_ported(&format!("the series of {}", name)));
                }
            }
            if GENERIC_NSERIES.contains(&name.as_str()) {
                if let Some(found) = generic_function_nseries(e, x, n, logx)? {
                    return Ok(found);
                }
            }
            application_nseries(e, x, n, logx, cdir, Some(!NOT_ONE_ARGUMENT.contains(&name.as_str())))
        }
        Head::Undefined(u) => {
            let one: Option<bool> = u.nargs.as_ref().map(|v| v.len() == 1 && v[0] == 1);
            application_nseries(e, x, n, logx, cdir, one)
        }
        Head::Derivative => {
            let arg: Basic = nseries(&e.args()[0], x, n, logx, &Basic::zero())?;
            let o: Option<Basic> = get_o(&arg)?;
            let counts: Vec<(Basic, u64)> = crate::core::function::derivative_counts(e);
            let mut terms: Vec<Basic> = terms_of(&remove_o(&arg)?)
                .iter()
                .map(|a| crate::core::function::derivative_node(a, &counts))
                .collect();
            if let Some(o) = o {
                terms.push(div(&o, x)?);
            }
            add(&terms)
        }
        Head::Subs => {
            let (expr, variables, point) = (&e.args()[0], e.args()[1].args(), e.args()[2].args());
            let other: Basic = match point.iter().position(|p| p == x) {
                Some(i) => variables[i].clone(),
                None => x.clone(),
            };
            let arg: Basic = nseries(expr, &other, n, logx, &Basic::zero())?;
            let o: Option<Basic> = get_o(&arg)?;
            let parts: Vec<Basic> = terms_of(&remove_o(&arg)?)
                .iter()
                .map(|a| crate::core::function::subs_node(a, variables.to_vec(), point.to_vec()))
                .collect();
            let mut rv: Basic = add(&parts)?;
            if let Some(o) = o {
                rv = add(&[rv, subs(&o, &other, x)?])?;
            }
            Ok(rv)
        }
        _ if e.is_atom() => Ok(e.clone()),
        _ => Err(not_ported(&format!("the series of {}", e.class_name()))),
    }
}

fn uniquely_named_symbol(name: &str, e: &Basic) -> Basic {
    let names: Vec<String> = e
        .atoms(Some(&|n: &Basic| n.is_symbol()))
        .iter()
        .map(|s| s.to_string())
        .collect();
    if !names.iter().any(|s| s == name) {
        return Symbol::new(name, &[]).0;
    }
    let mut k: usize = 0;
    while names.iter().any(|s| *s == format!("{}{}", name, k)) {
        k += 1;
    }
    Symbol::new(&format!("{}{}", name, k), &[]).0
}

/// The series of a function by its derivatives at zero (the general
/// algorithm, `sin(x + 1)`, `f(x)`).
fn derivative_series(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, limit_nan: bool) -> R {
    let expanded: Basic = expand(f)?;
    if *f != expanded {
        return nseries(&expanded, x, n, logx, &Basic::zero());
    }
    let mut e: Basic = f.clone();
    if e.args().len() == 1 && matches!(e.head(), Head::Function(_) | Head::Undefined(_)) {
        e = crate::core::function::rebuild_evaluated(&e, vec![cancel(&e.args()[0])?])?;
    }
    let term: Basic = subs(&e, x, &Basic::zero())?;
    if is(&term, Predicate::finite) == Some(false) || matches!(term.head(), Head::NaN) {
        return Err(pole_error(&format!("Cannot expand {} around 0", f)));
    }
    let mut result: Basic = term;
    let mut fact: Int = Int::ONE;
    let xi: Basic = uniquely_named_symbol("xi", f);
    e = subs(&e, x, &xi)?;
    for i in 1..n {
        fact *= Int::from(i);
        e = crate::core::function::derivative(&e, &xi)?;
        let mut value: Basic = subs(&e, &xi, &Basic::zero())?;
        if matches!(value.head(), Head::NaN) && limit_nan {
            value = limit_lite(&e, &xi, &Basic::zero())?;
        }
        if is(&value, Predicate::finite) == Some(false) {
            return Err(pole_error(&format!("Cannot expand {} around 0", f)));
        }
        let piece: Basic = mul(&[value, pow(x, &int(i))?, Basic::rational(Int::ONE, fact.clone())?])?;
        result = add(&[result, expand(&piece)?])?;
    }
    add(&[result, order_at_zero(&pow(x, &int(n))?, x)?])
}

/// The series by the Taylor terms of the function in its argument.
fn taylor_series(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> R {
    let arg: Basic = f.args()[0].clone();
    let mut nterms: i64 = n + 2;
    let cf: Option<Basic> = getn(&order_at_zero(&lt(&arg, x, None, &Basic::zero())?, x)?)?;
    if let Some(cf) = cf {
        if cf != Basic::zero() {
            nterms = ceiling_int(&div(&int(n), &cf)?)?;
        }
    }
    let mut terms: Vec<Basic> = Vec::new();
    let mut g: Option<Basic> = None;
    for i in 0..nterms {
        let term: Basic = taylor_term(f, i, &arg, g.as_ref())?;
        let term: Basic = nseries(&term, x, n, logx, &Basic::zero())?;
        terms.push(term.clone());
        g = Some(term);
    }
    add(&[add(&terms)?, order_at_zero(&pow(x, &int(n))?, x)?])
}

/// `_generic_function_nseries`: `None` for an argument going to an
/// infinity.
/// The classes with a ported rule (`functions/series_rules.rs`): when it
/// falls back, the generic expansion follows as in Python.
const SERIES_RULES: &[&str] = &[
    "Eis",
    "LambertW",
    "Abs",
    "Sign",
    "Arg",
    "Gamma",
    "LogGamma",
    "Expint",
    "EiNode",
    "LiNode",
    "OffsetLiNode",
    "SingularityFunctionNode",
    "EllipticK",
    "EllipticE",
    "Floor",
    "Ceiling",
    "Frac",
    "PolyLog",
    "BesselJ",
    "BesselI",
    "BesselY",
    "BesselK",
    "SiNode",
    "CiNode",
    "ShiNode",
    "ChiNode",
];

pub(crate) fn generic_function_nseries_pub(
    f: &Basic,
    x: &Basic,
    n: i64,
    logx: Option<&Basic>,
) -> PolyResult<Option<Basic>> {
    generic_function_nseries(f, x, n, logx)
}

fn generic_function_nseries(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> PolyResult<Option<Basic>> {
    let args: Vec<Basic> = f.args().to_vec();
    let args0: Vec<Basic> = args
        .iter()
        .map(|a| limit_lite(a, x, &Basic::zero()))
        .collect::<PolyResult<_>>()?;
    if args0.iter().any(|a| is(a, Predicate::finite) == Some(false)) {
        return asymptotic_branch(f, &args, &args0, x, n, logx).map(Some);
    }
    let one_argument: bool = !NOT_ONE_ARGUMENT.contains(&class_name(f).as_str());
    if !one_argument || args0[0] != Basic::zero() {
        return Ok(Some(derivative_series(f, x, n, logx, true)?));
    }
    Ok(Some(taylor_series(f, x, n, logx)?))
}

/// The classes whose SymPy counterpart has an `_eval_aseries` of its own
/// (`_SYMPY_OWN_ASERIES`): those without a ported rule are not ported.
const SYMPY_OWN_ASERIES: &[&str] = &[
    "Acot",
    "Atan",
    "BesselI",
    "BesselIs",
    "BesselK",
    "BesselKs",
    "CiNode",
    "DiGamma",
    "EiNode",
    "Eis",
    "Erf",
    "Erfc",
    "Erfi",
    "Erfs",
    "Expint",
    "FresnelC",
    "FresnelS",
    "LogGamma",
    "LowerGamma",
    "PolyGamma",
    "SiNode",
    "TriGamma",
];

/// SymPy's asymptotic branch of `Function._eval_nseries`: an argument
/// going to an infinity faster than logarithmically takes the class's own
/// `_eval_aseries` (`functions/aseries.rs`), SymPy's `PoleError` for a
/// class without one.
fn asymptotic_branch(f: &Basic, args: &[Basic], args0: &[Basic], x: &Basic, n: i64, logx: Option<&Basic>) -> R {
    let mut infinite: bool = false;
    for a in args {
        let leading: Basic = as_leading_term(a, std::slice::from_ref(x), logx, &Basic::zero())?;
        let a0: Basic = limit_lite(&leading, x, &Basic::zero())?;
        if a0.preorder().iter().any(|t| {
            matches!(
                t.head(),
                Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity | Head::NaN
            )
        }) {
            infinite = true;
        }
    }
    if !infinite {
        return Err(not_ported(
            "the series of a function at an argument going to an infinity logarithmically",
        ));
    }
    if let Some(found) = crate::functions::aseries::aseries_rule(f, n, args0, x, logx)? {
        return Ok(found);
    }
    let name: String = class_name(f);
    if SYMPY_OWN_ASERIES.contains(&name.as_str()) {
        return Err(not_ported(&format!("the asymptotic series of {}", name)));
    }
    let listed: Vec<String> = args0.iter().map(|a| a.to_string()).collect();
    Err(crate::error::pole_error(&format!(
        "Asymptotic expansion of {} around [{}] is not implemented.",
        name,
        listed.join(", ")
    )))
}

/// `Application._eval_nseries` (`one_argument`: whether `nargs` is `{1}`,
/// `None` for any number of arguments).
pub(crate) fn application_nseries(
    f: &Basic,
    x: &Basic,
    n: i64,
    logx: Option<&Basic>,
    _cdir: &Basic,
    one_argument: Option<bool>,
) -> R {
    let args0: Vec<Basic> = f
        .args()
        .iter()
        .map(|t| limit_lite(t, x, &Basic::zero()))
        .collect::<PolyResult<_>>()?;
    if args0.iter().any(|t| is(t, Predicate::finite) == Some(false)) {
        return asymptotic_branch(f, f.args(), &args0, x, n, logx);
    }
    let more_than_one: bool = !matches!(f.head(), Head::Function(_)) && f.args().len() > 1;
    if one_argument != Some(true) || !is_zero_number(&args0[0]) || more_than_one {
        return derivative_series(f, x, n, logx, true);
    }
    taylor_series(f, x, n, logx)
}

/// The `n`-th term of the Taylor series of `f` at zero in the argument
/// `a` (`taylor_term`: the classes' own, the derivatives otherwise).
pub fn taylor_term(f: &Basic, n: i64, a: &Basic, previous: Option<&Basic>) -> R {
    match f.head() {
        Head::Function(Function::Exp) => {
            if n < 0 {
                return Ok(Basic::zero());
            }
            if n == 0 {
                return Ok(Basic::one());
            }
            if let Some(p) = previous {
                return mul(&[p.clone(), a.clone(), pow(&int(n), &Basic::negative_one())?]);
            }
            let factorial: Basic = crate::functions::combinatorial::factorials::factorial(&int(n))?;
            mul(&[pow(a, &int(n))?, pow(&factorial, &Basic::negative_one())?])
        }
        Head::Function(Function::Log) => {
            if n < 0 {
                return Ok(Basic::zero());
            }
            if n == 0 {
                return Ok(a.clone());
            }
            if let Some(p) = previous {
                return powsimp_exp(&mul(&[
                    int(-n),
                    p.clone(),
                    a.clone(),
                    pow(&int(n + 1), &Basic::negative_one())?,
                ])?);
            }
            mul(&[
                int(1 - 2 * (n % 2)),
                pow(a, &int(n + 1))?,
                pow(&int(n + 1), &Basic::negative_one())?,
            ])
        }
        Head::Function(g) if super::series_trig::is_trig(*g) => super::series_trig::taylor_term(*g, n, a),
        Head::Function(g) if super::series_inverse::is_inverse(*g) => super::series_inverse::taylor_term(*g, n, a),
        Head::Function(_) if OWN_NSERIES.contains(&class_name(f).as_str()) => {
            Err(not_ported(&format!("the Taylor terms of {}", f.class_name())))
        }
        _ => {
            let d: Basic = Dummy::new("x", &[]).0;
            let mut e: Basic = subs(f, a, &d)?;
            for _ in 0..n {
                e = crate::core::function::derivative(&e, &d)?;
            }
            let factorial: Basic = crate::functions::combinatorial::factorials::factorial(&int(n))?;
            let value: Basic = if a.is_symbol() {
                subs(&subs(&e, &d, a)?, a, &Basic::zero())?
            } else {
                subs(&e, &d, &Basic::zero())?
            };
            mul(&[value, pow(a, &int(n))?, pow(&factorial, &Basic::negative_one())?])
        }
    }
}

fn coeff_exp_in(term: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut lt_: (Basic, Basic) = crate::solvers::solvers::as_coeff_exponent(term, x)?;
    if lt_.0.has(x) {
        lt_ = match leadterm(term, x, None, &Basic::zero()) {
            Ok(v) => v,
            Err(PolyError::Value(_)) => return Ok((term.clone(), Basic::zero())),
            Err(err) => return Err(err),
        };
    }
    Ok(lt_)
}

/// The coefficient and exponent of a term `c*x**e` by its factors (the
/// `coeff_exp` of `Pow._eval_nseries` and `log._eval_nseries`).
fn coeff_exp_by_factors(term: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    // the coefficient multiplied factor by factor, as Python does (a -1
    // meeting a sum distributes over it)
    let mut coeff: Basic = Basic::one();
    let mut exponent: Basic = Basic::zero();
    for factor in factors_of(term) {
        if factor.has(x) {
            let (base, ex) = crate::core::expr::as_base_exp(&factor)?;
            exponent = ex;
            if base != *x {
                return match leadterm(term, x, None, &Basic::zero()) {
                    Ok(v) => Ok(v),
                    Err(PolyError::Value(_)) => Ok((term.clone(), Basic::zero())),
                    Err(err) => Err(err),
                };
            }
        } else {
            coeff = mul(&[coeff, factor])?;
        }
    }
    Ok((coeff, exponent))
}

fn mul_nseries(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    let mut ords: Vec<(Basic, Basic)> = Vec::new();
    let mut nn: Basic = int(n);
    let mut facs: Vec<Basic> = Vec::new();
    let attempt: PolyResult<()> = (|| {
        for t in e.args() {
            let (coeff, ex) = leadterm(t, x, None, &Basic::zero())?;
            if coeff.has(x) {
                return Err(value_error("a coefficient in x".to_string()));
            }
            ords.push((t.clone(), ex));
        }
        let numeric: Vec<Basic> = ords
            .iter()
            .filter(|(_, m)| crate::core::assumptions::is_number(m))
            .map(|(_, m)| m.clone())
            .collect();
        let n0: Basic = add(&numeric)?;
        for (t, m) in &ords {
            let m_: Basic = if crate::core::assumptions::is_number(m) {
                m.clone()
            } else {
                Basic::zero()
            };
            let n1: i64 = ceiling_int(&add(&[nn.clone(), neg(&n0)?, m_])?)?;
            let series: Basic = nseries(t, x, n1, logx, cdir)?;
            if let Some(ns) = getn(&series)? {
                if less(&ns, &int(n1))? {
                    nn = sub(&nn, &sub(&int(n1), &ns)?)?;
                }
            }
            facs.push(series);
        }
        Ok(())
    })();
    if let Err(err) = attempt {
        if !caught(&err, true, true, true, true) {
            return Err(err);
        }
        let numeric: Vec<Basic> = ords
            .iter()
            .filter(|(_, m)| crate::core::assumptions::is_number(m))
            .map(|(_, m)| m.clone())
            .collect();
        let mut n0: Basic = add(&numeric)?;
        if yes(&n0, Predicate::nonnegative) {
            n0 = Basic::zero();
        }
        let k: i64 = ceiling_int(&sub(&nn, &n0)?)?;
        let facs: Vec<Basic> = e
            .args()
            .iter()
            .map(|t| nseries(t, x, k, logx, cdir))
            .collect::<PolyResult<_>>()?;
        let mut fallback: Basic = powsimp_exp(&expand(&mul(&facs)?)?)?;
        if fallback.preorder().iter().any(is_order) {
            fallback = add(&[fallback, order_at_zero(&pow(x, &nn)?, x)?])?;
        }
        return Ok(fallback);
    }
    let mut res: Basic = Basic::zero();
    let ords2: Vec<Vec<Basic>> = facs
        .iter()
        .map(|f| if f.is_add() { f.args().to_vec() } else { vec![f.clone()] })
        .collect();
    let mut indices: Vec<usize> = vec![0; ords2.len()];
    if ords2.iter().all(|v| !v.is_empty()) {
        loop {
            let fac: Vec<Basic> = indices.iter().zip(&ords2).map(|(i, v)| v[*i].clone()).collect();
            let ords3: Vec<(Basic, Basic)> = fac.iter().map(|t| coeff_exp_in(t, x)).collect::<PolyResult<_>>()?;
            let power: Basic = add(&ords3.iter().map(|(_, p)| p.clone()).collect::<Vec<_>>())?;
            // not SymPy's: a term whose exponent is not known to reach the
            // order is kept (the series of exp(x)*x**a for a positive a)
            if !yes(&sub(&power, &nn)?, Predicate::nonnegative) {
                let c: Basic = mul(&ords3.iter().map(|(c, _)| c.clone()).collect::<Vec<_>>())?;
                res = add(&[res, mul(&[c, pow(x, &power)?])?])?;
            }
            // the next element of the product (the last index fastest)
            let mut k: usize = indices.len();
            loop {
                if k == 0 {
                    break;
                }
                k -= 1;
                indices[k] += 1;
                if indices[k] < ords2[k].len() {
                    break;
                }
                indices[k] = 0;
                if k == 0 {
                    k = usize::MAX;
                    break;
                }
            }
            if k == usize::MAX || indices.is_empty() {
                break;
            }
        }
    }
    let polynomial: bool = crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true);
    if polynomial {
        let decided = (|| -> PolyResult<bool> {
            let maxd: Basic = max_degree(e, x)?;
            if !less(&maxd, &nn)? {
                return Ok(true);
            }
            Ok(degree(e, x)? != degree(&res, x)?)
        })();
        match decided {
            Ok(add_order) => {
                if add_order {
                    res = add(&[res, order_at_zero(&pow(x, &nn)?, x)?])?;
                }
                return Ok(res);
            }
            Err(PolyError::Polys(..)) => {}
            Err(err) => return Err(err),
        }
    }
    if res != *e {
        if subs(&sub(e, &res)?, x, &Basic::zero())? == Basic::zero() && less(&Basic::zero(), &nn)? {
            let l: Basic = eval_as_leading_term(e, x, logx, cdir)?;
            if l == Basic::zero() {
                return Ok(res);
            }
        }
        res = add(&[res, order_at_zero(&pow(x, &nn)?, x)?])?;
    }
    Ok(res)
}

fn polynomial_error(message: String) -> PolyError {
    PolyError::Polys(crate::error::PolysErrorKind::PolynomialError, message)
}

/// The degree of `e` in `x` (`_degree`: SymPy's `degree`, term by term
/// for coefficients outside the typed domains).
fn degree(e: &Basic, x: &Basic) -> PolyResult<i64> {
    if let Ok(p) = crate::polys::polytools::Poly::new(e, std::slice::from_ref(x)) {
        if let Ok(d) = p.degree(0) {
            return Ok(d);
        }
    }
    let expanded: Basic = expand(e)?;
    if is_zero_number(&expanded) {
        return Err(polynomial_error(format!("the degree of {} is -oo", e)));
    }
    let mut best: i64 = 0;
    for term in terms_of(&expanded) {
        let dependent: Basic = crate::core::expr::as_independent(&term, x, false)?.1;
        let mut n: i64 = 0;
        for f in factors_of(&dependent) {
            if f == *x {
                n += 1;
            } else if f.is_pow()
                && f.args()[0] == *x
                && f.args()[1].is_integer_value()
                && !rat_is_negative(f.args()[1].as_rational().expect("an integer"))
            {
                n += i64::try_from(&numer(f.args()[1].as_rational().expect("an integer"))).unwrap_or(i64::MAX);
            } else if f != Basic::one() {
                return Err(polynomial_error(format!("{} is not a polynomial in {}", e, x)));
            }
        }
        best = best.max(n);
    }
    Ok(best)
}

/// `_max_degree`.
fn max_degree(e: &Basic, x: &Basic) -> R {
    if e == x {
        return Ok(Basic::one());
    }
    if e.args().is_empty() {
        return Ok(Basic::zero());
    }
    if e.is_add() {
        let mut best: Basic = max_degree(&e.args()[0], x)?;
        for a in &e.args()[1..] {
            let d: Basic = max_degree(a, x)?;
            if less(&best, &d)? {
                best = d;
            }
        }
        return Ok(best);
    }
    if e.is_mul() {
        let parts: Vec<Basic> = e.args().iter().map(|a| max_degree(a, x)).collect::<PolyResult<_>>()?;
        return add(&parts);
    }
    if e.is_pow() {
        return mul(&[max_degree(&e.args()[0], x)?, e.args()[1].clone()]);
    }
    Ok(Basic::zero())
}

fn exp_nseries(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, _cdir: &Basic) -> R {
    let arg: Basic = e.args()[0].clone();
    let arg_series: Basic = eval_nseries(&arg, x, n, logx, &Basic::zero())?;
    if is_order(&arg_series) {
        return add(&[Basic::one(), arg_series]);
    }
    let arg0: Basic = limit_lite(&remove_o(&arg_series)?, x, &Basic::zero())?;
    if matches!(arg0.head(), Head::NegativeInfinity) {
        return order_at_zero(&pow(x, &int(n))?, x);
    }
    if matches!(arg0.head(), Head::Infinity) {
        return Ok(e.clone());
    }
    if yes(&arg0, Predicate::infinite) {
        return Err(pole_error(&format!("Cannot expand {} around 0", e)));
    }
    if arg0
        .args()
        .iter()
        .any(|a| matches!(a.head(), Head::Function(Function::Sign)))
    {
        return Ok(e.clone());
    }
    let t: Basic = Dummy::new("t", &[]).0;
    let mut nterms: i64 = n;
    let cf: Option<Basic> = match lt(&arg, x, logx, &Basic::zero())
        .and_then(|l| order_at_zero(&l, x))
        .and_then(|o| getn(&o))
    {
        Ok(v) => v,
        Err(err) if caught(&err, false, true, false, true) => Some(Basic::zero()),
        Err(err) => return Err(err),
    };
    let positive_cf: bool = match &cf {
        Some(c) => !is_zero_number(c) && less(&Basic::zero(), c)?,
        None => false,
    };
    if positive_cf {
        nterms = ceiling_int(&div(&int(n), cf.as_ref().expect("cf"))?)?;
    }
    // Exp(t)._taylor(t, nterms)
    let exp_t: Basic = Basic::new(Head::Function(Function::Exp), vec![t.clone()]);
    let mut taylor: Vec<Basic> = Vec::new();
    let mut g: Option<Basic> = None;
    for i in 0..nterms {
        let term: Basic = nseries(
            &taylor_term(&exp_t, i, &t, g.as_ref())?,
            &t,
            nterms,
            None,
            &Basic::zero(),
        )?;
        taylor.push(remove_o(&term)?);
        g = Some(term);
    }
    let exp_series: Basic = add(&taylor)?;
    let shift: Basic = sub(&arg_series, &arg0)?;
    let mut r: Basic = mul(&[exp(&arg0)?, subs(&exp_series, &t, &shift)?])?;
    match logx {
        Some(l) => {
            if subs(&r, l, &log(x)?)? == *e {
                return Ok(r);
            }
        }
        None => {
            if r == *e {
                return Ok(r);
            }
        }
    }
    if positive_cf && less(&Basic::one(), cf.as_ref().expect("cf"))? {
        let c: Basic = cf.expect("cf");
        let factor: Basic = pow(x, &mul(&[int(-1), sub(&c, &Basic::one())?, int(n)])?)?;
        r = add(&[r, mul(&[order_at_zero(&pow(&shift, &int(n))?, x)?, factor])?])?;
    } else {
        r = add(&[r, order_at_zero(&pow(&shift, &int(n))?, x)?])?;
    }
    r = expand(&r)?;
    r = powsimp_exp(&r)?;
    // powsimp may introduce unexpanded (-1)**Rational (SymPy's PR #17201)
    let query = |n: &Basic| -> bool {
        n.is_pow()
            && n.args()[0] == Basic::negative_one()
            && n.args()[1].as_rational().is_some_and(|q| {
                let d: Int = denom(q);
                d == Int::from(3) || d == Int::from(4) || d == Int::from(6)
            })
            && !n.args()[1].is_integer_value()
    };
    let value = |n: &Basic| -> R {
        crate::core::function::expand_with(
            n,
            true,
            crate::core::function::Hints {
                complex: true,
                ..hints()
            },
        )
    };
    crate::core::function::replace_where(&r, &query, &value)
}

fn log_nseries(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: Basic = e.args()[0].clone();
    if arg == *x {
        return match logx {
            Some(l) => Ok(l.clone()),
            None => log(x),
        };
    }
    let t: Basic = positive_dummy("t");
    let cdir: Basic = if is_zero_number(cdir) {
        Basic::one()
    } else {
        cdir.clone()
    };
    let z: Basic = subs(&arg, x, &mul(&[cdir.clone(), t.clone()])?)?;
    let wk: Basic = crate::core::symbol::Wild::new("k", vec![], vec![], &[]).0;
    let wl: Basic = crate::core::symbol::Wild::new("l", vec![], vec![], &[]).0;
    let pattern: Basic = mul(&[wk.clone(), pow(&t, &wl)?])?;
    if let Some(bindings) = crate::core::matching::match_pattern(&z, &pattern, &[], false)? {
        let k: Option<&Basic> = bindings.iter().find(|(w, _)| *w == wk).map(|(_, v)| v);
        let l: Option<&Basic> = bindings.iter().find(|(w, _)| *w == wl).map(|(_, v)| v);
        if let (Some(k), Some(l)) = (k, l) {
            if *l != Basic::zero() && !l.has(&t) && !k.has(&t) {
                let logx_: Basic = match logx {
                    Some(v) => v.clone(),
                    None => log(x)?,
                };
                let r: Basic = mul(&[l.clone(), logx_])?;
                return add(&[r, log(k)?, neg(&mul(&[l.clone(), log(&cdir)?])?)?]);
            }
        }
    }
    let mut nn: Basic = int(n);
    let (a, b) = match leadterm(&z, &t, logx, &Basic::one()) {
        Ok(v) => v,
        Err(err) if caught(&err, true, true, false, true) => {
            let mut m: i64 = n;
            let mut s: Basic = eval_nseries(&z, &t, m, logx, &Basic::one())?;
            while is_order(&s) {
                m += 1;
                s = eval_nseries(&z, &t, m, logx, &Basic::one())?;
            }
            nn = int(m);
            let removed: Basic = remove_o(&s)?;
            match leadterm(&removed, &t, None, &Basic::one()) {
                Ok(v) => v,
                Err(PolyError::Value(_)) => (lt(&removed, &t, None, &Basic::one())?, Basic::zero()),
                Err(err) => return Err(err),
            }
        }
        Err(err) => return Err(err),
    };
    let p0: Basic = cancel(&sub(&div(&z, &mul(&[a.clone(), pow(&t, &b)?])?)?, &Basic::one())?)?;
    let mut p: Basic = eval_nseries(&p0, &t, ceiling_int(&nn)?, logx, &Basic::one())?;
    if p.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(Function::Exp)))
    {
        p = crate::simplify::simplify::logcombine(&p, false)?;
    }
    if is_order(&p) {
        if let Some(found) = getn(&p)? {
            nn = found;
        }
    }
    let (_, d) = coeff_exp_by_factors(&p, &t)?;
    let logx_: Basic = match logx {
        Some(v) => v.clone(),
        None => log(x)?,
    };
    if !yes(&d, Predicate::positive) {
        let res: Basic = add(&[
            log(&a)?,
            neg(&mul(&[b.clone(), log(&cdir)?])?)?,
            mul(&[b.clone(), logx_.clone()])?,
        ])?;
        let expr: Basic = expand_logs(e, true)?;
        let log_x: Basic = log(x)?;
        let restored: Basic = if !crate::core::printing::could_extract_minus_sign(&a)
            && crate::core::printing::could_extract_minus_sign(&logx_)
        {
            subs(&res, &neg(&logx_)?, &neg(&log_x)?)?
        } else {
            subs(&res, &logx_, &log_x)?
        };
        if expand_logs(&restored, true)? == expr {
            return Ok(res);
        }
        return add(&[res, order_at_zero(&pow(x, &nn)?, x)?]);
    }
    let mut pterms: Vec<(Basic, Basic)> = Vec::new();
    for term in terms_of(&remove_o(&p)?) {
        let (co1, e1) = coeff_exp_by_factors(&term, &t)?;
        accumulate(&mut pterms, e1, co1)?;
    }
    let mut kk: i64 = 1;
    let mut terms: Vec<(Basic, Basic)> = Vec::new();
    let mut pk: Vec<(Basic, Basic)> = pterms.clone();
    while less(&mul(&[int(kk), d.clone()])?, &nn)? {
        let coeff: Basic = mul(&[
            int(-1),
            pow(&int(-1), &int(kk))?,
            pow(&int(kk), &Basic::negative_one())?,
        ])?;
        for (ex, c) in &pk {
            accumulate(&mut terms, ex.clone(), mul(&[coeff.clone(), c.clone()])?)?;
        }
        pk = product(&pk, &pterms, &nn)?;
        kk += 1;
    }
    let mut res: Basic = add(&[
        log(&a)?,
        neg(&mul(&[b.clone(), log(&cdir)?])?)?,
        mul(&[b.clone(), logx_])?,
    ])?;
    for (ex, c) in &terms {
        res = add(&[res, mul(&[cancel(c)?, pow(&t, ex)?])?])?;
    }
    if yes(&a, Predicate::negative) && crate::functions::elementary::complexes::im(&z)? != Basic::zero() {
        res = add(&[res, branch_jump(&z, &t)?])?;
    }
    res = subs(&res, &t, &div(x, &cdir)?)?;
    add(&[res, order_at_zero(&pow(x, &nn)?, x)?])
}

/// `d[k] = d.get(k, 0) + v` on an insertion-ordered dict.
fn accumulate(d: &mut Vec<(Basic, Basic)>, k: Basic, v: Basic) -> PolyResult<()> {
    match d.iter().position(|(key, _)| *key == k) {
        Some(i) => d[i].1 = add(&[d[i].1.clone(), v])?,
        None => d.push((k, v)),
    }
    Ok(())
}

/// The product of two series as dicts exponent -> coefficient, the
/// exponents below `bound` kept (`product` of the series methods).
fn product(d1: &[(Basic, Basic)], d2: &[(Basic, Basic)], bound: &Basic) -> PolyResult<Vec<(Basic, Basic)>> {
    let mut res: Vec<(Basic, Basic)> = Vec::new();
    for (e1, c1) in d1 {
        for (e2, c2) in d2 {
            let ex: Basic = add(&[e1.clone(), e2.clone()])?;
            if less(&ex, bound)? {
                accumulate(&mut res, ex, mul(&[c1.clone(), c2.clone()])?)?;
            }
        }
    }
    Ok(res)
}

fn has_any(e: &Basic, heads: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(heads)
}

fn pow_nseries(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    if is_exp1_power(e) {
        let e_series: Basic = nseries(&e.args()[1], x, n, logx, &Basic::zero())?;
        if is_order(&e_series) {
            return add(&[Basic::one(), e_series]);
        }
        let e0: Basic = limit_lite(&remove_o(&e_series)?, x, &Basic::zero())?;
        if matches!(e0.head(), Head::NegativeInfinity) {
            return order_at_zero(&pow(x, &int(n))?, x);
        }
        if matches!(e0.head(), Head::Infinity) {
            return Ok(e.clone());
        }
        let t: Basic = sub(&e_series, &e0)?;
        let mut term: Basic = exp(&e0)?;
        let mut exp_series: Basic = term.clone();
        for i in 1..n {
            term = mul(&[term, t.clone(), pow(&int(i), &Basic::negative_one())?])?;
            term = nseries(&term, x, n, logx, &Basic::zero())?;
            exp_series = add(&[exp_series, term.clone()])?;
        }
        exp_series = add(&[exp_series, order_at_zero(&pow(&t, &int(n))?, x)?])?;
        return powsimp_exp(&exp_series);
    }
    // SymPy denests with force=True, which takes every symbol for
    // positive; only the variable of the expansion is positive here (the
    // expansion is at 0+)
    let x_name: String = match x.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => x.to_string(),
    };
    let positive_x: Basic = Dummy::new(&x_name, &[(Predicate::positive, true)]).0;
    let denested: Basic = crate::simplify::powsimp::powdenest(&subs(e, x, &positive_x)?, false, false)?;
    let me: Basic = crate::simplify::trigsimp::trigsimp(
        &subs(&denested, &positive_x, x)?,
        crate::simplify::trigsimp::Method::Matching,
    )?;
    let (mut b, mut ex) = crate::core::expr::as_base_exp(&me)?;
    if has_any(&ex, &|n| {
        matches!(
            n.head(),
            Head::NaN | Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity
        )
    }) {
        return Err(pole_error(""));
    }
    if ex.has(x) {
        return eval_nseries(&exp(&mul(&[ex.clone(), log(&b)?])?)?, x, n, logx, cdir);
    }
    let mut me: Basic = me;
    if let Some(l) = logx {
        if has_any(&b, &|n| matches!(n.head(), Head::Function(Function::Log))) {
            // log(c*x**ex) -> log(c) + ex*logx
            let wc: Basic = crate::core::symbol::Wild::new("c", vec![x.clone()], Vec::new(), &[]).0;
            let wex: Basic = crate::core::symbol::Wild::new("ex", vec![x.clone()], Vec::new(), &[]).0;
            let pattern: Basic = mul(&[wc.clone(), pow(x, &wex)?])?;
            b = crate::core::function::replace_where(
                &b,
                &|n| matches!(n.head(), Head::Function(Function::Log)),
                &|n| {
                    let found = crate::core::matching::match_pattern(&n.args()[0], &pattern, &[], false)?;
                    let Some(found) = found else { return Ok(n.clone()) };
                    let get = |w: &Basic| found.iter().find(|(k, _)| k == w).map(|(_, v)| v.clone());
                    let (Some(c), Some(k)) = (get(&wc), get(&wex)) else {
                        return Ok(n.clone());
                    };
                    add(&[log(&c)?, mul(&[k, l.clone()])?])
                },
            )?;
            me = pow(&b, &ex)?;
        }
    }
    b = remove_o(&b)?;
    let forced: bool = logx.is_some()
        && has_any(&b, &|n| {
            matches!(n.head(), Head::Function(Function::PolyGamma) | Head::EulerGamma)
        });
    let first = if forced {
        Err(value_error(String::new()))
    } else {
        leadterm(&b, x, None, &Basic::zero())
    };
    let m: Basic = match first {
        Ok((_, m)) => m,
        Err(err) if caught(&err, true, true, false, true) => {
            let base: Basic = b.clone();
            b = remove_o(&eval_nseries(&base, x, n.max(2), logx, cdir)?)?;
            if has_any(&b, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity)) {
                return Err(not_implemented(""));
            }
            let mut m: Basic = leadterm(&b, x, None, &Basic::zero())?.1;
            // the series of the base stands for the base from here on: it
            // must be known to x**(n - m*e + m), the relative order of the
            // binomial series
            let needed: Basic = add(&[int(n), neg(&mul(&[m.clone(), ex.clone()])?)?, m.clone()])?;
            if crate::core::assumptions::is_number(&needed)
                && yes(&needed, Predicate::real)
                && less(&int(n.max(2)), &needed)?
            {
                b = remove_o(&eval_nseries(&base, x, ceiling_int(&needed)?, logx, cdir)?)?;
                if has_any(&b, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity)) {
                    return Err(not_implemented(""));
                }
                m = leadterm(&b, x, None, &Basic::zero())?.1;
            }
            m
        }
        Err(err) => return Err(err),
    };
    if has_any(&ex, &|n| matches!(n.head(), Head::Function(Function::Log))) {
        ex = cancel(&crate::simplify::simplify::logcombine(&ex, false)?)?;
    }
    if !(yes(&m, Predicate::zero) || (crate::core::assumptions::is_number(&ex) && yes(&ex, Predicate::real)))
        && me == pow_leading_term(&me, x, logx, cdir)?
    {
        let logarithmic: Basic = exp(&mul(&[ex.clone(), log(&b)?])?)?;
        let via_log: Basic = eval_nseries(&logarithmic, x, n, logx, cdir)?;
        if via_log == logarithmic {
            return Ok(me);
        }
        if is_order(&via_log) && !yes(&sub(&mul(&[m.clone(), ex.clone()])?, &int(n))?, Predicate::nonnegative) {
            // not SymPy's: exp(e*log(b)) with an exponent going to -oo is
            // taken to be smaller than every power, which does not hold for
            // x**a = exp(a*log(x))
            return Ok(me);
        }
        return Ok(via_log);
    }
    let f: Basic = lt(&b, x, logx, &Basic::zero())?;
    let mut g: Basic = cancel(&sub(&mexpand(&b)?, &f)?)?;
    g = div(&g, &f)?;
    if !crate::core::assumptions::is_number(&m) {
        return Err(not_implemented(""));
    }
    let mut maxpow: Basic = sub(&int(n), &mul(&[m.clone(), ex.clone()])?)?;
    if maxpow.preorder().iter().any(|n| n.is_symbol()) {
        maxpow = int(n);
    }
    if yes(&maxpow, Predicate::negative) {
        return order_at_zero(&pow(x, &mul(&[m.clone(), ex.clone()])?)?, x);
    }
    if yes(&g, Predicate::zero) {
        let mut r: Basic = pow(&f, &ex)?;
        if r != me {
            r = add(&[r, order_at_zero(&pow(x, &int(n))?, x)?])?;
        }
        return Ok(r);
    }
    let (c0, mut d) = match leadterm(&g, x, logx, &Basic::zero()) {
        Ok(v) => v,
        Err(err) if caught(&err, true, true, false, false) => {
            let ratio: Basic = div(&g, &pow(x, &maxpow)?)?;
            if limit_lite(&ratio, x, &Basic::zero())? == Basic::zero() {
                let fe: Basic = pow(&f, &ex)?;
                return add(&[fe.clone(), mul(&[ex.clone(), fe, g.clone()])?]);
            }
            return Err(not_implemented(""));
        }
        Err(err) => return Err(err),
    };
    let _ = c0;
    if !yes(&d, Predicate::positive) {
        g = crate::simplify::simplify::simplify(&g, &crate::simplify::simplify::Options::default())?;
        if yes(&g, Predicate::zero) {
            return pow(&f, &ex);
        }
        d = leadterm(&g, x, logx, &Basic::zero())?.1;
        if !yes(&d, Predicate::positive) {
            g = expand(&div(&sub(&b, &f)?, &f)?)?;
            d = leadterm(&g, x, logx, &Basic::zero())?.1;
            if !yes(&d, Predicate::positive) {
                return Err(not_implemented(""));
            }
        }
    }
    let gpoly: Basic = remove_o(&eval_nseries(&g, x, ceiling_int(&maxpow)?, logx, cdir)?)?;
    let mut gterms: Vec<(Basic, Basic)> = Vec::new();
    for term in terms_of(&gpoly) {
        let (co1, e1) = coeff_exp_by_factors(&term, x)?;
        accumulate(&mut gterms, e1, co1)?;
    }
    let mut k: i64 = 1;
    let mut terms: Vec<(Basic, Basic)> = vec![(Basic::zero(), Basic::one())];
    let mut tk: Vec<(Basic, Basic)> = gterms.clone();
    while yes(&sub(&mul(&[int(k), d.clone()])?, &maxpow)?, Predicate::negative) {
        let coeff: Basic = div(
            &crate::functions::combinatorial::factorials::ff(&ex, &int(k))?,
            &crate::functions::combinatorial::factorials::factorial(&int(k))?,
        )?;
        for (e1, c) in &tk {
            accumulate(&mut terms, e1.clone(), mul(&[coeff.clone(), c.clone()])?)?;
        }
        tk = product(&tk, &gterms, &maxpow)?;
        k += 1;
    }
    let (inco, inex) = if !yes(&ex, Predicate::integer) && yes(&m, Predicate::zero) && yes(&f, Predicate::negative) {
        let ndir: Basic = dir(&sub(&b, &f)?, x, cdir)?;
        let im_ndir: Basic = crate::functions::elementary::complexes::im(&ndir)?;
        if yes(&im_ndir, Predicate::negative) {
            coeff_exp_by_factors(
                &mul(&[
                    pow(&f, &ex)?,
                    pow(&Basic::negative_one(), &mul(&[int(-2), ex.clone()])?)?,
                ])?,
                x,
            )?
        } else if yes(&im_ndir, Predicate::zero) {
            coeff_exp_by_factors(&lt(&exp(&mul(&[ex.clone(), log(&b)?])?)?, x, logx, cdir)?, x)?
        } else {
            coeff_exp_by_factors(&pow(&f, &ex)?, x)?
        }
    } else {
        coeff_exp_by_factors(&pow(&f, &ex)?, x)?
    };
    let mut res: Basic = Basic::zero();
    for (e1, c) in &terms {
        let power: Basic = add(&[e1.clone(), inex.clone()])?;
        res = add(&[res, mul(&[c.clone(), inco.clone(), pow(x, &power)?])?])?;
    }
    let exact: bool = yes(&ex, Predicate::integer)
        && yes(&ex, Predicate::positive)
        && yes(&sub(&mul(&[ex.clone(), d.clone()])?, &int(n))?, Predicate::nonpositive)
        && res == mexpand(&me)?;
    if !exact {
        match order_at_zero(&pow(x, &int(n))?, x).and_then(|o| add(&[res.clone(), o])) {
            Ok(v) => res = v,
            Err(err) if caught(&err, false, true, false, false) => {
                return eval_nseries(&exp(&mul(&[ex.clone(), log(&b)?])?)?, x, n, logx, cdir);
            }
            Err(err) => return Err(err),
        }
    }
    Ok(res)
}

// -- series

fn direction(x0: &Basic, dir: &str, cdir: &Basic) -> PolyResult<(String, Basic)> {
    if dir != "+" && dir != "-" {
        return Err(value_error("Dir must be '+' or '-'".to_string()));
    }
    let mut dir: String = dir.to_string();
    let mut cdir: Basic = cdir.clone();
    if !yes(&cdir, Predicate::zero) {
        if yes(&cdir, Predicate::real) {
            dir = if yes(&cdir, Predicate::positive) { "+" } else { "-" }.to_string();
        } else {
            let im_c: Basic = crate::functions::elementary::complexes::im(&cdir)?;
            dir = if yes(&im_c, Predicate::positive) { "+" } else { "-" }.to_string();
        }
    } else if !is_zero_number(x0) && yes(x0, Predicate::infinite) {
        cdir = crate::simplify::simplify::simplify(
            &crate::functions::elementary::complexes::sign(x0)?,
            &crate::simplify::simplify::Options::default(),
        )?;
    } else if dir == "+" {
        cdir = Basic::one();
    } else {
        cdir = Basic::negative_one();
    }
    let a: Basic = crate::functions::elementary::complexes::abs_(&cdir)?;
    Ok((dir, div(&cdir, &a)?))
}

/// The series of `e` at `x = x0` up to `O((x - x0)**n, (x, x0))`
/// (`series`; `x` the only free symbol when `None`).
pub fn series(e: &Basic, x: Option<&Basic>, x0: &Basic, n: i64, dir: &str, logx: Option<&Basic>, cdir: &Basic) -> R {
    let x: Basic = match x {
        Some(v) => v.clone(),
        None => {
            let syms = e.free_symbols();
            if syms.is_empty() {
                return Ok(e.clone());
            }
            if syms.len() > 1 {
                return Err(value_error("x must be given for multivariate functions.".to_string()));
            }
            syms.into_iter().next().expect("a symbol")
        }
    };
    if !e.free_symbols().contains(&x) {
        return Ok(e.clone());
    }
    if n < 0 {
        return Err(value_error("Number of terms should be nonnegative".to_string()));
    }
    let (dir, cdir) = direction(x0, dir, cdir)?;
    if !is_zero_number(x0) && yes(x0, Predicate::infinite) {
        let moved: Basic = subs(e, &x, &div(&cdir, &x)?)?;
        return match series(&moved, Some(&x), &Basic::zero(), n, "+", None, &Basic::one()) {
            Ok(s) => subs(&s, &x, &div(&cdir, &x)?),
            Err(PolyError::Pole(_)) => Err(not_ported("the asymptotic series (aseries)")),
            Err(err) => Err(err),
        };
    }
    if !is_zero_number(x0) || cdir != Basic::one() {
        let moved: Basic =
            crate::core::subs::subs_mapping(e, &[(x.clone(), add(&[x0.clone(), mul(&[cdir.clone(), x.clone()])?])?)])?;
        let s: Basic = series(&moved, Some(&x), &Basic::zero(), n, "+", logx, &Basic::one())?;
        let back: Basic = sub(&div(&x, &cdir)?, &div(x0, &cdir)?)?;
        return crate::core::subs::subs_mapping(&s, &[(x.clone(), back)]);
    }
    let _ = dir;
    if is(&x, Predicate::positive).is_none() && is(&x, Predicate::negative).is_none() {
        let xpos: Basic = positive_dummy("x");
        let rv: Basic = series(&subs(e, &x, &xpos)?, Some(&xpos), x0, n, "+", logx, &cdir)?;
        return subs(&rv, &xpos, &x);
    }
    let mut s1: Basic = eval_nseries(e, &x, n, logx, &cdir)?;
    let mut o: Option<Basic> = get_o(&s1)?;
    if o.as_ref().is_some_and(|v| !is_zero_number(v)) {
        let ngot: Basic = match getn(&s1)? {
            Some(v) => v,
            None => {
                return Err(not_implemented(&format!(
                    "not sure of order of {}",
                    o.as_ref().expect("o")
                )))
            }
        };
        if less(&int(n), &ngot)? {
            let ov: Basic = o.clone().expect("o");
            if n != 0 {
                s1 = add(&[s1, subs(&ov, &x, &pow(&x, &div(&int(n), &ngot)?)?)?])?;
            } else {
                s1 = add(&[s1, order_at_zero(&Basic::one(), &x)?])?;
            }
        } else if less(&ngot, &int(n))? {
            let mut done: bool = false;
            for more in 1..9 {
                s1 = eval_nseries(e, &x, n + more, logx, &cdir)?;
                let newn: Basic = match getn(&s1)? {
                    Some(v) => v,
                    None => return Err(not_implemented(&format!("not sure of order of {}", s1))),
                };
                if newn != ngot {
                    let predicted: Basic = crate::functions::elementary::integers::ceiling(&div(
                        &mul(&[sub(&int(n), &ngot)?, int(more)])?,
                        &sub(&newn, &ngot)?,
                    )?)?;
                    let Some(p) = predicted.as_rational().filter(|_| predicted.is_integer_value()) else {
                        return Err(PolyError::Type(format!(
                            "the number of terms is not an integer: {}",
                            predicted
                        )));
                    };
                    let mut ndo: i64 =
                        n + i64::try_from(&numer(p)).map_err(|_| value_error("too many terms".to_string()))?;
                    s1 = eval_nseries(e, &x, ndo, logx, &cdir)?;
                    loop {
                        let got: Basic = match getn(&s1)? {
                            Some(v) => v,
                            None => return Err(PolyError::Type(format!("the series {} has no order term", s1))),
                        };
                        if !less(&got, &int(n))? {
                            break;
                        }
                        s1 = eval_nseries(e, &x, ndo, logx, &cdir)?;
                        ndo += 1;
                    }
                    done = true;
                    break;
                }
            }
            if !done {
                return Err(value_error(format!("Could not calculate {} terms for {}", n, e)));
            }
            s1 = add(&[s1, order_at_zero(&pow(&x, &int(n))?, &x)?])?;
        }
        o = get_o(&s1)?;
        s1 = remove_o(&s1)?;
    } else if s1.preorder().iter().any(is_order) {
        return Ok(s1);
    } else {
        o = Some(order_at_zero(&pow(&x, &int(n))?, &x)?);
        let s1done: Basic = crate::core::function::doit(&s1)?;
        match add(&[s1done.clone(), o.clone().expect("o")]).and_then(|v| remove_o(&v)) {
            Ok(v) => {
                if v == s1done {
                    o = Some(Basic::zero());
                }
            }
            Err(err) if caught(&err, false, true, false, false) => return Ok(s1),
            Err(err) => return Err(err),
        }
    }
    let rest: Basic = o.unwrap_or_else(Basic::zero);
    match crate::simplify::radsimp::collect(&s1, std::slice::from_ref(&x), None, Some(false)) {
        Ok(c) => add(&[c, rest]),
        Err(err) if caught(&err, false, true, false, false) => add(&[s1, rest]),
        Err(err) => Err(err),
    }
}
