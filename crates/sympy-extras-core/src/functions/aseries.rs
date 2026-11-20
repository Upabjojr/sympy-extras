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
        Function::BesselI | Function::BesselK | Function::BesselIs | Function::BesselKs => {
            bessel_aseries(f, *g, n, args0, x)?
        }
        _ => return Ok(None),
    }))
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
    let lower: Basic = half(&s2(&m2(&int(2), nu)?, &Basic::one())?)?;
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
