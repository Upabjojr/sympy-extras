//! Port of `sympy_extras/functions/_leading_terms.py`: the leading terms of
//! the special functions with an `_eval_as_leading_term` of their own in
//! SymPy (LambertW, factorial, binomial, gamma, polygamma, digamma,
//! trigamma, zeta, erf, erfc, erfi, Ei, Si, Ci, Shi, Chi, fresnels,
//! fresnelc, arg, floor, ceiling, frac, besselj, bessely, besseli, besselk
//! and SingularityFunction), by class name.

// PORTED FROM SYMPY: the _eval_as_leading_term methods of
// sympy/functions/special/error_functions.py, gamma_functions.py,
// zeta_functions.py, bessel.py, singularity_functions.py,
// sympy/functions/combinatorial/factorials.py and
// sympy/functions/elementary/exponential.py (LambertW), complexes.py (arg),
// integers.py (floor, ceiling, frac) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

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
fn sqrt(a: &Basic) -> R {
    pw(a, &Basic::half())
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn pi() -> Basic {
    Basic::pi()
}
fn imaginary() -> Basic {
    Basic::imaginary_unit()
}
fn euler_gamma() -> Basic {
    Basic::atom(Head::EulerGamma)
}
fn subs0(e: &Basic, x: &Basic) -> R {
    crate::core::subs::subs(e, x, &int(0))
}
fn same(f: &Basic, args: Vec<Basic>) -> R {
    crate::core::function::rebuild_evaluated(f, args)
}
fn lead(e: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    crate::core::series::as_leading_term(e, std::slice::from_ref(x), logx, cdir)
}
fn lead0(e: &Basic, x: &Basic) -> R {
    lead(e, x, None, &int(0))
}
fn limit_dir(arg: &Basic, x: &Basic, negative: bool) -> R {
    crate::core::limits::limit(arg, x, &int(0), if negative { "-" } else { "+" })
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn log_term(arg: &Basic, x: &Basic, logx: Option<&Basic>) -> R {
    let (c, ex) = crate::solvers::solvers::as_coeff_exponent(arg, x)?;
    let lx: Basic = match logx {
        Some(l) => l.clone(),
        None => log(x)?,
    };
    a2(&a2(&log(&c)?, &m2(&ex, &lx)?)?, &euler_gamma())
}
fn re_negative(cdir: &Basic) -> PolyResult<bool> {
    Ok(yes(
        &crate::functions::elementary::complexes::re(cdir)?,
        Predicate::negative,
    ))
}
fn not_implemented(message: String) -> PolyError {
    PolyError::NotImplemented(message)
}
fn is_not_ported(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if m.starts_with(crate::error::NOT_PORTED_PREFIX))
}
fn generic(f: &Basic, x: &Basic, logx: Option<&Basic>) -> R {
    crate::core::series::generic_leading_term_pub(f, x, logx)
}
fn arg_of(f: &Basic, i: usize) -> Basic {
    f.args()[i].clone()
}

fn lambertw(f: &Basic, x: &Basic) -> R {
    if f.args().len() == 1 {
        let arg: Basic = arg_of(f, 0);
        let arg0: Basic = crate::polys::polytools::cancel(&subs0(&arg, x)?, &[])?;
        if is(&arg0, Predicate::zero) != Some(true) {
            return same(f, vec![arg0]);
        }
        return lead0(&arg, x);
    }
    Err(not_implemented(format!("as_leading_term({}, {})", f, x)))
}

fn factorial(f: &Basic, x: &Basic) -> R {
    let arg: Basic = lead0(&arg_of(f, 0), x)?;
    let arg0: Basic = subs0(&arg, x)?;
    if yes(&arg0, Predicate::zero) {
        return Ok(int(1));
    }
    if is(&arg0, Predicate::infinite) != Some(true) {
        return same(f, vec![arg]);
    }
    Err(crate::error::pole_error(&format!("Cannot expand {} around 0", f)))
}

fn gamma(f: &Basic, x: &Basic) -> R {
    let arg: Basic = arg_of(f, 0);
    let x0: Basic = subs0(&arg, x)?;
    if yes(&x0, Predicate::integer) && yes(&x0, Predicate::nonpositive) {
        let n: Basic = neg(&x0)?;
        let res: Basic = d2(&pw(&int(-1), &n)?, &same(f, vec![a2(&n, &int(1))?])?)?;
        return d2(&res, &lead0(&a2(&arg, &n)?, x)?);
    }
    if is(&x0, Predicate::infinite) != Some(true) {
        return same(f, vec![x0]);
    }
    Err(crate::error::pole_error(""))
}

fn polygamma(f: &Basic, x: &Basic, logx: Option<&Basic>) -> R {
    let (n, z) = (lead0(&arg_of(f, 0), x)?, lead0(&arg_of(f, 1), x)?);
    let o: Basic = crate::core::order::order_at_zero(&z, x)?;
    if n == int(0)
        && crate::core::order::is_order_at_zero(&o)
        && crate::core::order::contains(&o, &d2(&int(1), x)?)? == Some(true)
    {
        let lx: Basic = match logx {
            Some(l) => l.clone(),
            None => log(x)?,
        };
        let n_o: Basic =
            crate::core::order::getn(&o)?.ok_or_else(|| PolyError::Type("an order without n".to_string()))?;
        return m2(&n_o, &lx);
    }
    same(f, vec![n, z])
}

fn digamma(f: &Basic, x: &Basic, k: i64) -> R {
    lead0(
        &crate::functions::special::gamma_functions::polygamma(&int(k), &arg_of(f, 0))?,
        x,
    )
}

fn zeta(f: &Basic, x: &Basic, logx: Option<&Basic>) -> R {
    let s: Basic = arg_of(f, 0);
    let a: Basic = if f.args().len() == 2 { arg_of(f, 1) } else { int(1) };
    let ex: Basic = match crate::core::series::leadterm(&a, x, None, &int(0)) {
        Ok((_, ex)) => ex,
        Err(e) if matches!(e, PolyError::NotImplemented(_)) && !is_not_ported(&e) => return Ok(f.clone()),
        Err(e) => return Err(e),
    };
    if yes(&ex, Predicate::negative) && is(&s, Predicate::positive) != Some(true) {
        return Err(not_implemented(String::new()));
    }
    generic(f, x, logx)
}

fn erf_like(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic, kind: Function) -> R {
    let arg: Basic = lead(&arg_of(f, 0), x, logx, cdir)?;
    let mut arg0: Basic = subs0(&arg, x)?;
    if kind != Function::Erfi && matches!(arg0.head(), Head::ComplexInfinity) {
        arg0 = limit_dir(&arg, x, *cdir == int(-1))?;
    }
    match kind {
        Function::Erf => {
            if arg.free_symbols().contains(x) && yes(&arg0, Predicate::zero) {
                return d2(&m2(&int(2), &arg)?, &sqrt(&pi())?);
            }
            same(f, vec![arg0])
        }
        Function::Erfc => {
            if yes(&arg0, Predicate::zero) {
                return Ok(int(1));
            }
            same(f, vec![arg0])
        }
        _ => {
            if arg.free_symbols().contains(x) && yes(&arg0, Predicate::zero) {
                return d2(&m2(&int(2), &arg)?, &sqrt(&pi())?);
            }
            if yes(&arg0, Predicate::finite) {
                return same(f, vec![arg0]);
            }
            same(f, vec![arg])
        }
    }
}

fn ei(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let x0: Basic = crate::core::limits::limit(&arg_of(f, 0), x, &int(0), "+")?;
    let arg: Basic = lead(&arg_of(f, 0), x, None, cdir)?;
    let direction: Basic = crate::core::series::dir(&arg, x, cdir)?;
    if yes(&x0, Predicate::zero) {
        let term: Basic = log_term(&arg, x, logx)?;
        let shift: Basic = if re_negative(&direction)? {
            m2(&imaginary(), &pi())?
        } else {
            int(0)
        };
        return s2(&term, &shift);
    }
    generic(f, x, logx)
}

fn odd_integral(f: &Basic, x: &Basic, cdir: &Basic, own_arg: Basic) -> R {
    let arg: Basic = own_arg;
    let mut arg0: Basic = subs0(&arg, x)?;
    if matches!(arg0.head(), Head::NaN) {
        arg0 = limit_dir(&arg, x, re_negative(cdir)?)?;
    }
    if yes(&arg0, Predicate::zero) {
        return Ok(arg);
    }
    if is(&arg0, Predicate::infinite) != Some(true) {
        return same(f, vec![arg0]);
    }
    Ok(f.clone())
}

fn even_integral(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: Basic = lead(&arg_of(f, 0), x, logx, cdir)?;
    let mut arg0: Basic = subs0(&arg, x)?;
    if matches!(arg0.head(), Head::NaN) {
        arg0 = limit_dir(&arg, x, re_negative(cdir)?)?;
    }
    if yes(&arg0, Predicate::zero) {
        return log_term(&arg, x, logx);
    }
    if yes(&arg0, Predicate::finite) {
        return same(f, vec![arg0]);
    }
    Ok(f.clone())
}

fn fresnel(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic, sine: bool) -> R {
    let arg: Basic = lead(&arg_of(f, 0), x, logx, cdir)?;
    let mut arg0: Basic = subs0(&arg, x)?;
    if matches!(arg0.head(), Head::ComplexInfinity) {
        arg0 = limit_dir(&arg, x, re_negative(cdir)?)?;
    }
    if yes(&arg0, Predicate::zero) {
        return if sine {
            d2(&m2(&pi(), &pw(&arg, &int(3))?)?, &int(6))
        } else {
            Ok(arg)
        };
    }
    if matches!(arg0.head(), Head::Infinity | Head::NegativeInfinity) {
        let s: i64 = if matches!(arg0.head(), Head::Infinity) { 1 } else { -1 };
        return a2(&m2(&int(s), &Basic::half())?, &crate::core::order::order_at_zero(x, x)?);
    }
    same(f, vec![arg0])
}

fn arg_rule(f: &Basic, x: &Basic, cdir: &Basic) -> R {
    let t: Basic = crate::core::symbol::Dummy::new("t", &[(Predicate::positive, true)]).0;
    let d: Basic = if *cdir == int(0) { int(1) } else { cdir.clone() };
    let z: Basic = crate::core::subs::subs(&arg_of(f, 0), x, &m2(&d, &t)?)?;
    if yes(&z, Predicate::positive) {
        return Ok(int(0));
    }
    if yes(&z, Predicate::negative) {
        return Ok(pi());
    }
    Err(crate::error::pole_error(&format!("Cannot expand {} around 0", f)))
}

fn rounding(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic, up: bool) -> R {
    let arg: Basic = arg_of(f, 0);
    let mut arg0: Basic = subs0(&arg, x)?;
    let mut r: Basic = subs0(f, x)?;
    if matches!(arg0.head(), Head::NaN | Head::AccumBounds) {
        arg0 = limit_dir(&arg, x, re_negative(cdir)?)?;
        r = same(f, vec![arg0.clone()])?;
    }
    if yes(&arg0, Predicate::finite) {
        if arg0 == r {
            let ndir: Basic = crate::core::series::dir(&arg, x, &if *cdir != int(0) { cdir.clone() } else { int(1) })?;
            if yes(&ndir, Predicate::negative) {
                return if up { Ok(r) } else { s2(&r, &int(1)) };
            }
            if yes(&ndir, Predicate::positive) {
                return if up { a2(&r, &int(1)) } else { Ok(r) };
            }
            return Err(not_implemented(format!("Not sure of sign of {}", ndir)));
        }
        return Ok(r);
    }
    lead(&arg, x, logx, cdir)
}

fn frac(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: Basic = arg_of(f, 0);
    let arg0: Basic = subs0(&arg, x)?;
    let r: Basic = subs0(f, x)?;
    if yes(&arg0, Predicate::finite) {
        if yes(&r, Predicate::zero) {
            let ndir: Basic = crate::core::series::dir(&arg, x, cdir)?;
            if yes(&ndir, Predicate::negative) {
                return Ok(int(1));
            }
            return lead(&s2(&arg, &arg0)?, x, logx, cdir);
        }
        return Ok(r);
    }
    if matches!(
        arg0.head(),
        Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
    ) {
        return crate::core::accumbounds::accum_bounds(&int(0), &int(1));
    }
    lead(&arg, x, logx, cdir)
}

fn bessel_parts(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Basic, Basic, Basic)>> {
    let (nu, z) = (arg_of(f, 0), arg_of(f, 1));
    let lead_: Basic = match lead0(&z, x) {
        Ok(l) => l,
        Err(e) if matches!(e, PolyError::NotImplemented(_)) && !is_not_ported(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let (c, ex) = crate::solvers::solvers::as_coeff_exponent(&lead_, x)?;
    Ok(Some((nu, z, lead_, c, ex)))
}

fn large_positive(c: &Basic, ex: &Basic, cdir: &Basic) -> PolyResult<bool> {
    let d: Basic = if *cdir == int(0) { int(1) } else { cdir.clone() };
    Ok(is(&m2(c, &pw(&d, ex)?)?, Predicate::negative) != Some(true))
}

fn bessel_ji(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic, modified: bool) -> R {
    let Some((nu, z, lead_, c, ex)) = bessel_parts(f, x)? else {
        return Ok(f.clone());
    };
    if yes(&ex, Predicate::positive) {
        let denominator: Basic = m2(
            &pw(&int(2), &nu)?,
            &crate::functions::special::gamma_functions::gamma(&a2(&nu, &int(1))?)?,
        )?;
        return d2(&pw(&lead_, &nu)?, &denominator);
    }
    if yes(&ex, Predicate::negative) {
        if large_positive(&c, &ex, cdir)? {
            if modified {
                return d2(
                    &crate::functions::elementary::exponential::exp(&z)?,
                    &sqrt(&m2(&m2(&int(2), &pi())?, &z)?)?,
                );
            }
            let phase: Basic = s2(&z, &d2(&m2(&pi(), &a2(&m2(&int(2), &nu)?, &int(1))?)?, &int(4))?)?;
            let top: Basic = m2(
                &sqrt(&int(2))?,
                &crate::functions::elementary::trigonometric::cos(&phase)?,
            )?;
            return d2(&top, &sqrt(&m2(&pi(), &z)?)?);
        }
        return Ok(f.clone());
    }
    generic(f, x, logx)
}

fn bessel_y(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    use crate::functions::elementary::trigonometric::{cos, sin};
    let Some((nu, z, _lead, c, ex)) = bessel_parts(f, x)? else {
        return Ok(f.clone());
    };
    if yes(&ex, Predicate::positive) {
        let half_z: Basic = d2(&z, &int(2))?;
        let term_one: Basic = m2(
            &m2(&d2(&int(2), &pi())?, &log(&half_z)?)?,
            &crate::functions::special::bessel::besselj(&nu, &z)?,
        )?;
        let term_two: Basic = if yes(&nu, Predicate::positive) {
            d2(
                &m2(
                    &neg(&pw(&half_z, &neg(&nu)?)?)?,
                    &crate::functions::combinatorial::factorials::factorial(&s2(&nu, &int(1))?)?,
                )?,
                &pi(),
            )?
        } else {
            int(0)
        };
        let digamma_part: Basic = s2(
            &crate::functions::special::gamma_functions::digamma(&a2(&nu, &int(1))?)?,
            &euler_gamma(),
        )?;
        let term_three: Basic = m2(
            &d2(
                &neg(&pw(&half_z, &nu)?)?,
                &m2(&pi(), &crate::functions::combinatorial::factorials::factorial(&nu)?)?,
            )?,
            &digamma_part,
        )?;
        let total: Basic = crate::core::add::add(&[term_one, term_two, term_three])?;
        return lead(&total, x, logx, &int(0));
    }
    if yes(&ex, Predicate::negative) {
        if large_positive(&c, &ex, cdir)? {
            let angle: Basic = a2(&s2(&d2(&m2(&pi(), &nu)?, &int(2))?, &z)?, &d2(&pi(), &int(4))?)?;
            let inner: Basic = a2(
                &neg(&sin(&angle)?)?,
                &d2(&m2(&int(3), &cos(&angle)?)?, &m2(&int(8), &z)?)?,
            )?;
            let value: Basic = m2(&m2(&sqrt(&int(2))?, &inner)?, &sqrt(&d2(&int(1), &z)?)?)?;
            return d2(&value, &sqrt(&pi())?);
        }
        return Ok(f.clone());
    }
    generic(f, x, logx)
}

fn bessel_k(f: &Basic, x: &Basic, logx: Option<&Basic>) -> R {
    let Some((nu, z, lead_, _c, ex)) = bessel_parts(f, x)? else {
        return Ok(f.clone());
    };
    if yes(&ex, Predicate::positive) {
        let term: Basic = if yes(&nu, Predicate::zero) {
            a2(&s2(&neg(&log(&z)?)?, &euler_gamma())?, &log(&int(2))?)?
        } else if yes(&nu, Predicate::nonzero) {
            let a: Basic = crate::functions::elementary::complexes::abs_(&nu)?;
            d2(
                &m2(
                    &crate::functions::special::gamma_functions::gamma(&a)?,
                    &pw(&d2(&z, &int(2))?, &neg(&a)?)?,
                )?,
                &int(2),
            )?
        } else {
            return Err(not_implemented(format!(
                "Cannot proceed without knowing if {} is zero or not.",
                nu
            )));
        };
        return lead(&term, x, logx, &int(0));
    }
    if yes(&ex, Predicate::negative) {
        let value: Basic = m2(
            &sqrt(&pi())?,
            &crate::functions::elementary::exponential::exp(&neg(&lead_)?)?,
        )?;
        return d2(&value, &sqrt(&m2(&int(2), &lead_)?)?);
    }
    same(f, vec![nu, lead_])
}

fn singularity(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let (z, a, n) = (arg_of(f, 0), arg_of(f, 1), arg_of(f, 2));
    let shift: Basic = subs0(&s2(&z, &a)?, x)?;
    if yes(&n, Predicate::negative) {
        return Ok(int(0));
    }
    if yes(&shift, Predicate::zero) {
        // at the singular point the side decides: <d>**n is d**n where d
        // is positive and 0 where it is negative
        let d: Basic = s2(&z, &a)?;
        if !singularity_side_positive(&d, x, cdir)? {
            return Ok(int(0));
        }
        if yes(&n, Predicate::zero) {
            return Ok(int(1));
        }
        return crate::core::series::as_leading_term(&pw(&d, &n)?, std::slice::from_ref(x), logx, cdir);
    }
    if yes(&shift, Predicate::positive) {
        return pw(&shift, &n);
    }
    Ok(int(0))
}

/// Whether `d`, which vanishes at `x = 0`, is positive on the side of
/// `cdir` (`singularity_side_positive`).
pub(crate) fn singularity_side_positive(d: &Basic, x: &Basic, cdir: &Basic) -> PolyResult<bool> {
    let (mut coefficient, exponent) = match crate::core::series::leadterm(d, x, None, &Basic::zero()) {
        Ok(found) => found,
        Err(crate::error::PolyError::Value(_) | crate::error::PolyError::NotImplemented(_)) => {
            return Ok(*cdir != int(-1));
        }
        Err(e) => return Err(e),
    };
    if *cdir == int(-1) {
        coefficient = crate::core::mul::mul(&[coefficient, pw(&int(-1), &exponent)?])?;
    }
    Ok(!yes(&coefficient, Predicate::negative))
}

/// The ported rule of the class of `f`, `None` when it has none
/// (`rules()[type(f).__name__]`).
pub fn leading_term_rule(f: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> PolyResult<Option<Basic>> {
    let name: &str = f.class_name();
    let found: Basic = match name {
        "LambertW" => lambertw(f, x)?,
        "Factorial" => factorial(f, x)?,
        "Binomial" => {
            let rewritten: Basic =
                crate::core::rewriting::rewrite(f, &[], "gamma", &crate::core::rewriting::RewriteHints::default())?;
            crate::core::series::eval_as_leading_term(&rewritten, x, logx, cdir)?
        }
        "Gamma" => gamma(f, x)?,
        "PolyGamma" => polygamma(f, x, logx)?,
        "DiGamma" => digamma(f, x, 0)?,
        "TriGamma" => digamma(f, x, 1)?,
        "Zeta" => zeta(f, x, logx)?,
        "Erf" => erf_like(f, x, logx, cdir, Function::Erf)?,
        "Erfc" => erf_like(f, x, logx, cdir, Function::Erfc)?,
        "Erfi" => erf_like(f, x, logx, cdir, Function::Erfi)?,
        "EiNode" => ei(f, x, logx, cdir)?,
        "SiNode" => odd_integral(f, x, cdir, lead(&arg_of(f, 0), x, logx, cdir)?)?,
        "ShiNode" => odd_integral(f, x, cdir, lead0(&arg_of(f, 0), x)?)?,
        "CiNode" | "ChiNode" => even_integral(f, x, logx, cdir)?,
        "FresnelS" => fresnel(f, x, logx, cdir, true)?,
        "FresnelC" => fresnel(f, x, logx, cdir, false)?,
        "Arg" => arg_rule(f, x, cdir)?,
        "Frac" => frac(f, x, logx, cdir)?,
        "Floor" => rounding(f, x, logx, cdir, false)?,
        "Ceiling" => rounding(f, x, logx, cdir, true)?,
        "BesselJ" => bessel_ji(f, x, logx, cdir, false)?,
        "BesselI" => bessel_ji(f, x, logx, cdir, true)?,
        "BesselY" => bessel_y(f, x, logx, cdir)?,
        "BesselK" => bessel_k(f, x, logx)?,
        "SingularityFunctionNode" => singularity(f, x, logx, cdir)?,
        "Eis" => {
            // PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_eis._eval_as_leading_term
            let z: &Basic = &f.args()[0];
            let x0: Basic = crate::core::limits::limit(z, x, &Basic::zero(), "+")?;
            if crate::core::expr::is(&x0, crate::core::assumptions::Predicate::zero) != Some(true) {
                return Ok(None);
            }
            let rewritten: Basic = crate::functions::special::error_functions::helper_intractable(Function::Eis, z)?;
            crate::core::series::eval_as_leading_term(&rewritten, x, logx, cdir)?
        }
        _ => return Ok(None),
    };
    Ok(Some(found))
}
