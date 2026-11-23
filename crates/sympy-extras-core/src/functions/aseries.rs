//! The asymptotic series of the special functions at an argument going to
//! an infinity (`_eval_aseries`), ported from `sympy_extras.functions`:
//! Stirling's series of `loggamma`, the series of `polygamma` (`digamma`,
//! `trigamma`), of `lowergamma` in its order, of `besseli`, `besselk` and
//! their helpers `_besseli`, `_besselk`. `None` for a class without a rule
//! of its own; SymPy's `PoleError` where the rule falls back to
//! `Function._eval_aseries`.

use crate::core::basic::{Basic, Function, Head};
use crate::error::{pole_error, PolyError, PolyResult};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
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
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn rational(p: i64, q: i64) -> R {
    d2(&int(p), &int(q))
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn bernoulli(k: i64) -> R {
    crate::functions::combinatorial::numbers::bernoulli(&int(k), None)
}
fn order_in(e: &Basic, x: &Basic) -> R {
    crate::core::order::order(e, &[(x.clone(), Basic::zero())])
}
fn nseries(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> R {
    crate::core::series::eval_nseries(e, x, n, logx, &Basic::zero())
}
fn default(f: &Basic, args0: &[Basic]) -> PolyError {
    let listed: Vec<String> = args0.iter().map(|a| a.to_string()).collect();
    pole_error(&format!(
        "Asymptotic expansion of {} around [{}] is not implemented.",
        f.class_name(),
        listed.join(", ")
    ))
}
fn is_oo(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity)
}

/// The asymptotic series of `f` (`_eval_aseries`), `None` when its class
/// has no rule of its own.
pub fn aseries_rule(f: &Basic, n: i64, args0: &[Basic], x: &Basic, logx: Option<&Basic>) -> PolyResult<Option<Basic>> {
    let Head::Function(g) = f.head() else { return Ok(None) };
    Ok(Some(match g {
        Function::LogGamma => loggamma_aseries(f, n, args0, x, logx)?,
        Function::PolyGamma => polygamma_aseries(f, &f.args()[0], &f.args()[1], n, args0, x, logx)?,
        Function::DiGamma => {
            let args: Vec<Basic> = std::iter::once(Basic::zero()).chain(args0.iter().cloned()).collect();
            polygamma_aseries(f, &Basic::zero(), &f.args()[0], n, &args, x, logx)?
        }
        Function::TriGamma => {
            let args: Vec<Basic> = std::iter::once(Basic::one()).chain(args0.iter().cloned()).collect();
            polygamma_aseries(f, &Basic::one(), &f.args()[0], n, &args, x, logx)?
        }
        Function::LowerGamma => lowergamma_aseries(f, n, args0, x)?,
        Function::Erfs => erfs_aseries(f, n, args0, x, logx)?,
        Function::Erf => erf_aseries(f, n, args0, x)?,
        Function::Erfc => {
            let node: Basic = Basic::new(Head::Function(Function::Erf), vec![f.args()[0].clone()]);
            crate::core::add::sub(&Basic::one(), &erf_aseries(&node, n, args0, x)?)?
        }
        Function::Erfi => erfi_aseries(f, n, args0, x)?,
        Function::Eis => eis_aseries(f, n, args0, x, logx)?,
        Function::Atan | Function::Acot => atan_acot_aseries(f, *g == Function::Acot, n, args0, x, logx)?,
        Function::BesselI | Function::BesselK | Function::BesselIs | Function::BesselKs => {
            bessel_aseries(f, *g, n, args0, x)?
        }
        _ => return Ok(None),
    }))
}

fn factorial2_of(k: i64) -> R {
    crate::functions::apply_function(Function::Factorial2, &[int(k)])
}

/// The asymptotic series of `erf(z)` at an argument going to an infinity
/// faster than logarithmically, the node itself when the leading term of
/// the argument is not found (`_erf_aseries`).
fn erf_aseries(f: &Basic, n: i64, args0: &[Basic], x: &Basic) -> R {
    if matches!(args0[0].head(), Head::Infinity | Head::NegativeInfinity) {
        let z: &Basic = &f.args()[0];
        let ex: Basic = match crate::core::series::leadterm(z, x, None, &Basic::zero()) {
            Ok((_, ex)) => ex,
            Err(PolyError::Value(_)) => return Ok(f.clone()),
            Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {
                return Ok(f.clone())
            }
            Err(e) => return Err(e),
        };
        // as x -> 1/x for the aseries
        let ex: Basic = m2(&int(-1), &ex)?;
        if crate::core::expr::is(&ex, crate::core::assumptions::Predicate::positive) == Some(true) {
            let count: Basic = crate::functions::apply_function(Function::Ceiling, &[d2(&int(n), &ex)?])?;
            let Some(newn) = count
                .as_rational()
                .filter(|q| q.is_int())
                .and_then(|q| i64::try_from(crate::numbers::numer(q)).ok())
            else {
                return Err(PolyError::Type(format!(
                    "the number of terms is not an integer: {}",
                    count
                )));
            };
            let mut terms: Vec<Basic> = Vec::new();
            for k in 0..newn {
                let top: Basic = m2(&pw(&int(-1), &int(k))?, &factorial2_of(2 * k - 1)?)?;
                terms.push(d2(&top, &m2(&pw(z, &int(2 * k + 1))?, &pw(&int(2), &int(k))?)?)?);
            }
            terms.push(order_in(&d2(&Basic::one(), &pw(z, &int(newn))?)?, x)?);
            let gauss: Basic = d2(
                &crate::functions::elementary::exponential::exp(&m2(&int(-1), &pw(z, &int(2))?)?)?,
                &pw(&Basic::pi(), &Basic::half())?,
            )?;
            return crate::core::add::sub(&Basic::one(), &m2(&gauss, &crate::core::add::add(&terms)?)?);
        }
    }
    Err(default(f, args0))
}

// PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfi._eval_aseries
fn erfi_aseries(f: &Basic, n: i64, args0: &[Basic], x: &Basic) -> R {
    if !is_oo(&args0[0]) {
        return Err(default(f, args0));
    }
    let z: &Basic = &f.args()[0];
    let mut terms: Vec<Basic> = Vec::new();
    for k in 0..n {
        terms.push(d2(
            &factorial2_of(2 * k - 1)?,
            &m2(&pw(&int(2), &int(k))?, &pw(z, &int(2 * k + 1))?)?,
        )?);
    }
    terms.push(order_in(&d2(&Basic::one(), &pw(z, &int(n))?)?, x)?);
    let gauss: Basic = d2(
        &crate::functions::elementary::exponential::exp(&pw(z, &int(2))?)?,
        &pw(&Basic::pi(), &Basic::half())?,
    )?;
    a2(
        &m2(&int(-1), &Basic::atom(Head::ImaginaryUnit))?,
        &m2(&gauss, &crate::core::add::add(&terms)?)?,
    )
}

/// The terms expanded and the order added (`_helper_aseries`).
fn helper_aseries(terms: &[Basic], o: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> R {
    a2(&nseries(&crate::core::add::add(terms)?, x, n, logx)?, o)
}

fn factorial_of(k: i64) -> R {
    crate::functions::combinatorial::factorials::factorial(&int(k))
}

// PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_erfs._eval_aseries
fn erfs_aseries(f: &Basic, n: i64, args0: &[Basic], x: &Basic, logx: Option<&Basic>) -> R {
    let point: &Basic = &args0[0];
    let along_i: bool = crate::core::expr::extract_multiplicatively(point, &Basic::atom(Head::ImaginaryUnit))?
        .is_some_and(|v| is_oo(&v));
    if !(is_oo(point) || along_i) {
        return Err(default(f, args0));
    }
    let z: &Basic = &f.args()[0];
    let inverse: Basic = d2(&Basic::one(), z)?;
    let sqrt_pi: Basic = pw(&Basic::pi(), &Basic::half())?;
    let mut terms: Vec<Basic> = Vec::new();
    for k in 0..n {
        let top: Basic = m2(&factorial_of(2 * k)?, &pw(&int(-4), &int(-k))?)?;
        let coefficient: Basic = d2(&top, &m2(&sqrt_pi, &factorial_of(k)?)?)?;
        terms.push(m2(&coefficient, &pw(&inverse, &int(2 * k + 1))?)?);
    }
    let o: Basic = order_in(&d2(&Basic::one(), &pw(z, &int(2 * n + 1))?)?, x)?;
    helper_aseries(&terms, &o, x, n, logx)
}

// PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_eis._eval_aseries
fn eis_aseries(f: &Basic, n: i64, args0: &[Basic], x: &Basic, logx: Option<&Basic>) -> R {
    if !matches!(args0[0].head(), Head::Infinity | Head::NegativeInfinity) {
        return Err(default(f, args0));
    }
    let z: &Basic = &f.args()[0];
    let inverse: Basic = d2(&Basic::one(), z)?;
    let mut terms: Vec<Basic> = Vec::new();
    for k in 0..n {
        terms.push(m2(&factorial_of(k)?, &pw(&inverse, &int(k + 1))?)?);
    }
    let o: Basic = order_in(&d2(&Basic::one(), &pw(z, &int(n + 1))?)?, x)?;
    helper_aseries(&terms, &o, x, n, logx)
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan._eval_aseries, acot._eval_aseries
fn atan_acot_aseries(f: &Basic, is_acot: bool, n: i64, args0: &[Basic], x: &Basic, logx: Option<&Basic>) -> R {
    if !matches!(args0[0].head(), Head::Infinity | Head::NegativeInfinity) {
        return Err(default(f, args0));
    }
    let inverse: Basic = crate::functions::apply_function(Function::Atan, &[pw(&f.args()[0], &int(-1))?])?;
    let e: Basic = if is_acot {
        inverse
    } else {
        crate::core::add::add(&[d2(&Basic::pi(), &int(2))?, m2(&int(-1), &inverse)?])?
    };
    nseries(&e, x, n, logx)
}

// PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_aseries
fn loggamma_aseries(f: &Basic, n: i64, args0: &[Basic], x: &Basic, logx: Option<&Basic>) -> R {
    if !is_oo(&args0[0]) {
        return Err(default(f, args0));
    }
    let z: &Basic = &f.args()[0];
    let half: Basic = rational(1, 2)?;
    let two_pi: Basic = m2(&int(2), &Basic::pi())?;
    let r: Basic = a2(&s2(&m2(&log(z)?, &s2(z, &half)?)?, z)?, &d2(&log(&two_pi)?, &int(2))?)?;
    let mut parts: Vec<Basic> = vec![r];
    for k in 1..n {
        let denominator: Basic = m2(&int(2 * k * (2 * k - 1)), &pw(z, &int(2 * k - 1))?)?;
        parts.push(d2(&bernoulli(2 * k)?, &denominator)?);
    }
    let o: Basic = if n == 0 {
        order_in(&Basic::one(), x)?
    } else {
        order_in(&d2(&Basic::one(), &pw(z, &int(n))?)?, x)?
    };
    a2(&nseries(&crate::core::add::add(&parts)?, x, n, logx)?, &o)
}

// PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:polygamma._eval_aseries
fn polygamma_aseries(
    f: &Basic,
    order_: &Basic,
    z: &Basic,
    n: i64,
    args0: &[Basic],
    x: &Basic,
    logx: Option<&Basic>,
) -> R {
    let big_n: Option<i64> = match order_.head() {
        Head::Rational {
            integer_class: true, ..
        } => i64::try_from(&crate::numbers::numer(order_.as_rational().expect("an integer")))
            .ok()
            .filter(|v| *v >= 0),
        _ => None,
    };
    let (Some(big_n), true) = (big_n, is_oo(&args0[1])) else {
        return Err(default(f, args0));
    };
    if big_n == 0 {
        let mut r: Basic = s2(&log(z)?, &d2(&Basic::one(), &m2(&int(2), z)?)?)?;
        let o: Basic = if n < 2 {
            order_in(&d2(&Basic::one(), z)?, x)?
        } else {
            let m: i64 = (n + 1) / 2;
            let mut terms: Vec<Basic> = Vec::new();
            for k in 1..m {
                terms.push(d2(&bernoulli(2 * k)?, &m2(&int(2 * k), &pw(z, &int(2 * k))?)?)?);
            }
            r = s2(&r, &crate::core::add::add(&terms)?)?;
            order_in(&d2(&Basic::one(), &pw(z, &int(n))?)?, x)?
        };
        return a2(&nseries(&r, x, n, logx)?, &o);
    }
    let nb: Basic = int(big_n);
    let mut fac: Basic = crate::functions::special::gamma_functions::gamma(&nb)?;
    let mut e0: Basic = a2(&fac, &d2(&m2(&nb, &fac)?, &m2(&int(2), z)?)?)?;
    let m: i64 = (n + 1) / 2;
    for k in 1..m {
        let up: Basic = m2(&m2(&fac, &int(2 * k + big_n - 1))?, &int(2 * k + big_n - 2))?;
        fac = d2(&up, &int((2 * k) * (2 * k - 1)))?;
        e0 = a2(&e0, &d2(&m2(&bernoulli(2 * k)?, &fac)?, &pw(z, &int(2 * k))?)?)?;
    }
    let mut o: Basic = order_in(&d2(&Basic::one(), &pw(z, &int(2 * m))?)?, x)?;
    if n == 0 {
        o = order_in(&d2(&Basic::one(), z)?, x)?;
    } else if n == 1 {
        o = order_in(&d2(&Basic::one(), &pw(z, &int(2))?)?, x)?;
    }
    let r: Basic = a2(&e0, &o)?;
    let sign: Basic = m2(&int(-1), &pw(&d2(&int(-1), z)?, &nb)?)?;
    nseries(&m2(&sign, &r)?, x, n, logx)
}

// PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:lowergamma._eval_aseries
fn lowergamma_aseries(f: &Basic, n: i64, args0: &[Basic], x: &Basic) -> R {
    let (s, z) = (&f.args()[0], &f.args()[1]);
    if !(is_oo(&args0[0]) && !z.has(x)) {
        return Err(default(f, args0));
    }
    let coeff: Basic = m2(&pw(z, s)?, &exp(&crate::core::mul::neg(z)?)?)?;
    let mut terms: Vec<Basic> = Vec::new();
    for k in 0..(n - 1).max(0) {
        terms.push(d2(
            &pw(z, &int(k))?,
            &crate::functions::combinatorial::factorials::rf(s, &int(k + 1))?,
        )?);
    }
    let sum_expr: Basic = crate::core::add::add(&terms)?;
    let o: Basic = crate::core::order::order(&m2(&pw(z, s)?, &pw(s, &int(-n))?)?, &[])?;
    a2(&m2(&coeff, &sum_expr)?, &o)
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli._eval_aseries (the sum)
fn bessel_terms(nu: &Basic, z: &Basic, n: i64, x: &Basic, base: i64) -> PolyResult<(Vec<Basic>, Basic)> {
    use crate::functions::combinatorial::factorials::{factorial, rf};
    let half = |numerator: &Basic| -> R {
        if !matches!(numerator.head(), Head::Rational { .. }) {
            return Err(PolyError::Type(format!("invalid input: {}", numerator)));
        }
        d2(numerator, &int(2))
    };
    // (1/2 - nu)_k (DLMF 10.40.1-2; SymPy has (nu - 1/2)_k)
    let lower: Basic = half(&s2(&Basic::one(), &m2(&int(2), nu)?)?)?;
    let upper: Basic = half(&a2(&m2(&int(2), nu)?, &Basic::one())?)?;
    let mut s: Vec<Basic> = Vec::new();
    for k in 0..n {
        let top: Basic = m2(&rf(&lower, &int(k))?, &rf(&upper, &int(k))?)?;
        let bottom: Basic = m2(
            &m2(&pw(&int(base), &int(k))?, &pw(z, &rational(2 * k + 1, 2)?)?)?,
            &factorial(&int(k))?,
        )?;
        s.push(d2(&top, &bottom)?);
    }
    let o: Basic = order_in(&d2(&Basic::one(), &pw(z, &rational(2 * n + 1, 2)?)?)?, x)?;
    Ok((s, o))
}

// PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli._eval_aseries, besselk._eval_aseries,
// _besseli._eval_aseries, _besselk._eval_aseries
fn bessel_aseries(f: &Basic, g: Function, n: i64, args0: &[Basic], x: &Basic) -> R {
    if !matches!(args0[1].head(), Head::Infinity | Head::NegativeInfinity) {
        return Err(default(f, args0));
    }
    let (nu, z) = (&f.args()[0], &f.args()[1]);
    let base: i64 = if matches!(g, Function::BesselI | Function::BesselIs) {
        2
    } else {
        -2
    };
    let (terms, o) = bessel_terms(nu, z, n, x, base)?;
    let half_pi: Basic = pw(&d2(&Basic::pi(), &int(2))?, &rational(1, 2)?)?;
    match g {
        Function::BesselI => {
            let factor: Basic = d2(&exp(z)?, &pw(&m2(&int(2), &Basic::pi())?, &rational(1, 2)?)?)?;
            let mut all: Vec<Basic> = terms;
            all.push(o);
            m2(&factor, &crate::core::add::add(&all)?)
        }
        Function::BesselK => {
            let mut all: Vec<Basic> = terms;
            all.push(o);
            crate::core::mul::mul(&[exp(&crate::core::mul::neg(z)?)?, half_pi, crate::core::add::add(&all)?])
        }
        _ => a2(&m2(&half_pi, &crate::core::add::add(&terms)?)?, &o),
    }
}
