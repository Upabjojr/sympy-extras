//! Port of `sympy_extras/functions/_series_rules.py`: the series expansions
//! of the special functions with an `_eval_nseries` of their own in SymPy,
//! by class name. A rule gives `None` where SymPy calls the generic
//! expansion of functions and that one is not available here; the caller
//! then falls back. (`sinc` has no head in the Rust core.)

// PORTED FROM SYMPY: the _eval_nseries methods of LambertW (exponential.py),
// Abs, sign, arg (complexes.py), gamma, loggamma (gamma_functions.py),
// expint, Ei, TrigonometricIntegral, li, Li (error_functions.py),
// SingularityFunction (singularity_functions.py), elliptic_k, elliptic_e
// (elliptic_integrals.py), floor, ceiling, frac (integers.py), polylog
// (zeta_functions.py), besselj, bessely, besseli, besselk (bessel.py) of
// sympy/functions (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
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
fn arg(f: &Basic, i: usize) -> Basic {
    f.args()[i].clone()
}
fn apply(f: Function, args: &[Basic]) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), args.to_vec()), args.to_vec())
}
fn ns(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    crate::core::series::eval_nseries(e, x, n, logx, cdir)
}
fn ns0(e: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> R {
    ns(e, x, n, logx, &int(0))
}
fn generic(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    crate::core::series::generic_function_nseries_pub(f, x, n, logx)
}
fn limit0(e: &Basic, x: &Basic) -> R {
    crate::core::limits::limit(e, x, &int(0), "+")
}
fn subs0(e: &Basic, x: &Basic) -> R {
    crate::core::subs::subs(e, x, &int(0))
}
fn order_xn(x: &Basic, n: i64) -> R {
    crate::core::order::order_at_zero(&pw(x, &int(n))?, x)
}
fn remove_o(e: &Basic) -> R {
    crate::core::order::remove_o(e)
}
fn mexpand(e: &Basic) -> R {
    crate::core::function::expand_mul(&crate::core::function::expand_multinomial(e, true)?, true)
}
fn truncate(e: &Basic, o: &Basic) -> R {
    remove_o(&a2(&mexpand(e)?, o)?)
}
fn is_zero_number(e: &Basic) -> bool {
    e.is_number_node() && *e == int(0)
}
fn re_negative(cdir: &Basic) -> PolyResult<bool> {
    Ok(yes(
        &crate::functions::elementary::complexes::re(cdir)?,
        Predicate::negative,
    ))
}
fn limit_dir(e: &Basic, x: &Basic, cdir: &Basic) -> R {
    crate::core::limits::limit(e, x, &int(0), if re_negative(cdir)? { "-" } else { "+" })
}
fn is_integer_node(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}
fn int_of(e: &Basic) -> PolyResult<i64> {
    if !is_integer_node(e) {
        return Err(PolyError::Type(format!(
            "cannot determine truth value of Relational {} > 0",
            e
        )));
    }
    Ok(i64::try_from(&crate::numbers::numer(e.as_rational().expect("an integer"))).unwrap_or(0))
}
fn ceiling(e: &Basic) -> R {
    crate::functions::elementary::integers::rounding(Function::Ceiling, e)
}
fn factorial(e: &Basic) -> R {
    crate::functions::combinatorial::factorials::factorial(e)
}
fn gamma(e: &Basic) -> R {
    crate::functions::special::gamma_functions::gamma(e)
}
fn digamma(e: &Basic) -> R {
    crate::functions::special::gamma_functions::digamma(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn accum(lo: i64, hi: i64) -> R {
    crate::core::accumbounds::accum_bounds(&int(lo), &int(hi))
}
fn is_not_ported(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if m.starts_with(crate::error::NOT_PORTED_PREFIX))
}
/// `z.leadterm(x)` with `ValueError` and `NotImplementedError` as `None`.
fn leadterm_or_none(z: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    match crate::core::series::leadterm(z, x, None, &int(0)) {
        Ok(found) => Ok(Some(found)),
        Err(e)
            if matches!(e, PolyError::Value(_))
                || (matches!(e, PolyError::NotImplemented(_)) && !is_not_ported(&e)) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

fn lambertw(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    if f.args().len() != 1 {
        return generic(f, x, n, logx);
    }
    let a: Basic = crate::core::series::nseries(&arg(f, 0), x, n, logx, &int(0))?;
    let lt: Basic = crate::core::series::as_leading_term(&a, std::slice::from_ref(x), logx, &int(0))?;
    let lte: Basic = if lt.is_pow() { lt.args()[1].clone() } else { int(1) };
    let top: Basic = ceiling(&d2(&int(n), &lte)?)?;
    let top: i64 = int_of(&top)?;
    let mut s: Basic = int(0);
    if top >= 1 {
        let mut terms: Vec<Basic> = Vec::new();
        for k in 1..top {
            let c: Basic = d2(
                &m2(&pw(&int(-1), &int(k - 1))?, &pw(&int(k), &int(k - 2))?)?,
                &factorial(&int(k - 1))?,
            )?;
            terms.push(m2(&c, &pw(&a, &int(k))?)?);
        }
        s = crate::core::function::expand_multinomial(&add_all(&terms)?, true)?;
    }
    Ok(Some(a2(&s, &order_xn(x, n)?)?))
}

fn abs_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    let mut direction: Basic = crate::core::series::leadterm(&arg(f, 0), x, None, &int(0))?.0;
    if let Some(l) = logx {
        let lx: Basic = log(x)?;
        if direction.has(&lx) {
            direction = crate::core::subs::subs(&direction, &lx, l)?;
        }
    }
    let s: Basic = ns0(&arg(f, 0), x, n, logx)?;
    Ok(Some(crate::core::function::expand_default(&m2(
        &crate::functions::elementary::complexes::sign(&direction)?,
        &s,
    )?)?))
}

fn sign_rule(f: &Basic, x: &Basic, cdir: &Basic) -> OptR {
    let arg0: Basic = arg(f, 0);
    let x0: Basic = subs0(&arg0, x)?;
    if x0 != int(0) {
        return Ok(Some(crate::core::function::rebuild_evaluated(f, vec![x0])?));
    }
    let d: Basic = if *cdir != int(0) {
        crate::core::series::dir(&arg0, x, cdir)?
    } else {
        cdir.clone()
    };
    Ok(Some(int(if re_negative(&d)? { -1 } else { 1 })))
}

fn arg_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> OptR {
    if n <= 0 {
        return Ok(Some(crate::core::order::order(&int(1), &[])?));
    }
    Ok(Some(crate::core::series::eval_as_leading_term(f, x, logx, cdir)?))
}

fn gamma_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    let a: Basic = arg(f, 0);
    let x0: Basic = limit0(&a, x)?;
    if !(is_integer_node(&x0) && int_of(&x0)? <= 0) {
        return generic(f, x, n, logx);
    }
    let t: Basic = s2(&a, &x0)?;
    let rf: Basic = crate::functions::combinatorial::factorials::rf(&a, &a2(&neg(&x0)?, &int(1))?)?;
    Ok(Some(ns0(&d2(&gamma(&a2(&t, &int(1))?)?, &rf)?, x, n, logx)?))
}

fn loggamma_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    if yes(&limit0(&arg(f, 0), x)?, Predicate::zero) {
        return Ok(Some(ns0(&log(&gamma(&arg(f, 0))?)?, x, n, logx)?));
    }
    generic(f, x, n, logx)
}

fn integral(f: Function, z: &Basic) -> R {
    apply(f, std::slice::from_ref(z))
}

fn expint_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    let (nu, z) = (arg(f, 0), arg(f, 1));
    if !nu.has(x) {
        if nu == int(1) {
            return Ok(Some(ns0(
                &s2(&integral(Function::Shi, &z)?, &integral(Function::Chi, &z)?)?,
                x,
                n,
                logx,
            )?));
        }
        if is_integer_node(&nu) && int_of(&nu)? > 1 {
            let k_nu: i64 = int_of(&nu)?;
            let w: Basic = neg(&crate::functions::elementary::polar::unpolarify(&z, false)?)?;
            let polar: Basic = crate::functions::elementary::polar::exp_polar(&m2(&neg(&imaginary())?, &pi())?)?;
            let e1: Basic = s2(
                &neg(&integral(Function::Ei, &m2(&z, &polar)?)?)?,
                &m2(&imaginary(), &pi())?,
            )?;
            let nu1: Basic = s2(&nu, &int(1))?;
            let mut sum: Vec<Basic> = Vec::new();
            for k in 0..k_nu - 1 {
                sum.push(m2(&factorial(&int(k_nu - k - 2))?, &pw(&w, &int(k))?)?);
            }
            let first: Basic = m2(&d2(&pw(&w, &nu1)?, &factorial(&nu1)?)?, &e1)?;
            let second: Basic = m2(
                &d2(&crate::functions::elementary::exponential::exp(&w)?, &factorial(&nu1)?)?,
                &add_all(&sum)?,
            )?;
            return Ok(Some(ns0(&a2(&first, &second)?, x, n, logx)?));
        }
    }
    generic(f, x, n, logx)
}

fn ei_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    let z: Basic = arg(f, 0);
    if yes(&limit0(&z, x)?, Predicate::zero) {
        let polar: bool = z
            .preorder()
            .iter()
            .any(|m| matches!(m.head(), Head::Function(Function::ExpPolar | Function::PolarLift)));
        let negative: bool = yes(&z, Predicate::negative) && !polar;
        let base: Basic = a2(&integral(Function::Shi, &z)?, &integral(Function::Chi, &z)?)?;
        let g: Basic = if negative {
            s2(&base, &m2(&imaginary(), &pi())?)?
        } else {
            base
        };
        return Ok(Some(ns0(&g, x, n, logx)?));
    }
    generic(f, x, n, logx)
}

fn trig_integral(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, trig: Function) -> OptR {
    if subs0(&arg(f, 0), x)? != int(0) {
        return generic(f, x, n, logx);
    }
    let mut baseseries: Basic = ns0(&apply(trig, std::slice::from_ref(x))?, x, n, logx)?;
    let at_zero: bool = apply(trig, &[int(0)])? != int(0);
    if at_zero {
        baseseries = s2(&baseseries, &int(1))?;
    }
    baseseries = crate::core::function::replace_where(&baseseries, &|e| e.is_pow(), &|e| d2(e, &e.args()[1]))?;
    if at_zero {
        baseseries = a2(&a2(&baseseries, &euler_gamma())?, &log(x)?)?;
    }
    Ok(Some(ns0(
        &crate::core::subs::subs(&baseseries, x, &arg(f, 0))?,
        x,
        n,
        logx,
    )?))
}

fn li_rule(f: &Basic, n: i64) -> OptR {
    let z: Basic = arg(f, 0);
    let mut s: Vec<Basic> = Vec::new();
    for k in 1..n {
        s.push(d2(&pw(&log(&z)?, &int(k))?, &m2(&factorial(&int(k))?, &int(k))?)?);
    }
    Ok(Some(a2(&a2(&euler_gamma(), &log(&log(&z)?)?)?, &add_all(&s)?)?))
}

fn offset_li_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> OptR {
    let li = |z: &Basic| integral(Function::LogIntegral, z);
    Ok(Some(ns0(&s2(&li(&arg(f, 0))?, &li(&int(2))?)?, x, n, logx)?))
}

fn singularity_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> OptR {
    let (z, a, p) = (arg(f, 0), arg(f, 1), arg(f, 2));
    let shift: Basic = subs0(&s2(&z, &a)?, x)?;
    if yes(&p, Predicate::negative) {
        return Ok(Some(int(0)));
    }
    if yes(&p, Predicate::zero) && yes(&shift, Predicate::zero) {
        return Ok(Some(int(if *cdir == int(-1) { 0 } else { 1 })));
    }
    if yes(&shift, Predicate::positive) {
        return Ok(Some(ns(&pw(&s2(&z, &a)?, &p)?, x, n, logx, cdir)?));
    }
    Ok(Some(int(0)))
}

fn elliptic_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, first: Basic) -> OptR {
    if f.args().len() != 1 {
        return generic(f, x, n, logx);
    }
    let m: Basic = arg(f, 0);
    let hyp: Basic = crate::functions::special::hyper::hyper(&[first, Basic::half()], &[int(1)], &m)?;
    let g: Basic = m2(&d2(&pi(), &int(2))?, &hyp)?;
    use crate::simplify::hyperexpand::{hyperexpand, Rewrite};
    Ok(Some(hyperexpand(&ns0(&g, x, n, logx)?, false, Rewrite::Default, None)?))
}

fn round_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic, up: bool) -> OptR {
    let a: Basic = arg(f, 0);
    let mut arg0: Basic = subs0(&a, x)?;
    let mut r: Basic = subs0(f, x)?;
    if matches!(arg0.head(), Head::NaN) {
        arg0 = limit_dir(&a, x, cdir)?;
        r = crate::core::function::rebuild_evaluated(f, vec![arg0.clone()])?;
    }
    if yes(&arg0, Predicate::infinite) {
        let s: Basic = ns(&a, x, n, logx, cdir)?;
        if n <= 0 {
            return Ok(Some(a2(
                &s,
                &crate::core::order::order(&int(1), &[(x.clone(), int(0))])?,
            )?));
        }
        return Ok(Some(a2(&s, &if up { accum(0, 1)? } else { accum(-1, 0)? })?));
    }
    if arg0 == r {
        let ndir: Basic = crate::core::series::dir(&a, x, &if *cdir != int(0) { cdir.clone() } else { int(1) })?;
        if yes(&ndir, Predicate::negative) {
            return Ok(Some(if up { r } else { s2(&r, &int(1))? }));
        }
        if yes(&ndir, Predicate::positive) {
            return Ok(Some(if up { a2(&r, &int(1))? } else { r }));
        }
        return Err(PolyError::NotImplemented(format!("Not sure of sign of {}", ndir)));
    }
    Ok(Some(r))
}

fn frac_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> OptR {
    let a: Basic = arg(f, 0);
    let arg0: Basic = subs0(&a, x)?;
    let r: Basic = subs0(f, x)?;
    if yes(&arg0, Predicate::infinite) {
        if n <= 0 {
            return Ok(Some(crate::core::order::order(&int(1), &[(x.clone(), int(0))])?));
        }
        let o: Basic = crate::core::order::order(&pw(x, &int(n))?, &[(x.clone(), int(0))])?;
        return Ok(Some(a2(&accum(0, 1)?, &o)?));
    }
    let res: Basic = ns(&s2(&a, &arg0)?, x, n, logx, cdir)?;
    if yes(&r, Predicate::zero) {
        let ndir: Basic = crate::core::series::dir(&a, x, cdir)?;
        return Ok(Some(a2(
            &res,
            &int(if yes(&ndir, Predicate::negative) { 1 } else { 0 }),
        )?));
    }
    Ok(Some(a2(&res, &r)?))
}

fn polylog_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> OptR {
    let (nu, z) = (arg(f, 0), arg(f, 1));
    let mut z0: Basic = subs0(&z, x)?;
    if matches!(z0.head(), Head::NaN) {
        z0 = limit_dir(&z, x, cdir)?;
    }
    if yes(&z0, Predicate::zero) {
        let Some((_, ex)) = leadterm_or_none(&z, x)? else {
            return Ok(Some(f.clone()));
        };
        if yes(&ex, Predicate::positive) {
            let newn: i64 = int_of(&ceiling(&d2(&int(n), &ex)?)?)?;
            let o: Basic = order_xn(x, n)?;
            let r: Basic = remove_o(&ns(&z, x, n, logx, cdir)?)?;
            if is_zero_number(&r) {
                return Ok(Some(o));
            }
            let mut term: Basic = r.clone();
            let mut terms: Vec<Basic> = vec![term.clone()];
            for k in 2..newn {
                term = m2(&term, &r)?;
                terms.push(d2(&term, &pw(&int(k), &nu)?)?);
            }
            return Ok(Some(a2(&add_all(&terms)?, &o)?));
        }
    }
    generic(f, x, n, logx)
}

fn bessel_head(f: &Basic, x: &Basic, n: i64) -> PolyResult<Option<(Basic, Basic, Basic, i64)>> {
    let (nu, z) = (arg(f, 0), arg(f, 1));
    let Some((_, ex)) = leadterm_or_none(&z, x)? else {
        return Ok(None);
    };
    let newn: i64 = if yes(&ex, Predicate::positive) {
        int_of(&ceiling(&d2(&int(n), &ex)?)?)?
    } else {
        0
    };
    Ok(Some((nu, z, ex, newn)))
}

fn half_series(z: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    remove_o(&ns(&d2(z, &int(2))?, x, n, logx, cdir)?)
}

fn bessel_ji(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic, modified: bool) -> OptR {
    let Some((nu, z, ex, newn)) = bessel_head(f, x, n)? else {
        return Ok(Some(f.clone()));
    };
    if !yes(&ex, Predicate::positive) {
        return generic(f, x, n, logx);
    }
    let o: Basic = order_xn(x, n)?;
    let r: Basic = half_series(&z, x, n, logx, cdir)?;
    if is_zero_number(&r) {
        return Ok(Some(o));
    }
    let t: Basic = remove_o(&a2(&mexpand(&pw(&r, &int(2))?)?, &o)?)?;
    let mut term: Basic = d2(&pw(&r, &nu)?, &gamma(&a2(&nu, &int(1))?)?)?;
    let mut s: Vec<Basic> = vec![term.clone()];
    for k in 1..(newn + 1) / 2 {
        let signed: Basic = if modified { t.clone() } else { neg(&t)? };
        term = m2(&term, &d2(&signed, &m2(&int(k), &a2(&nu, &int(k))?)?)?)?;
        term = truncate(&term, &o)?;
        s.push(term.clone());
    }
    Ok(Some(a2(&add_all(&s)?, &o)?))
}

fn bessel_y(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> OptR {
    let Some((nu, z, ex, newn)) = bessel_head(f, x, n)? else {
        return Ok(Some(f.clone()));
    };
    if !(yes(&ex, Predicate::positive) && yes(&nu, Predicate::integer)) {
        return generic(f, x, n, logx);
    }
    let bn: Basic = crate::functions::special::bessel::besselj(&nu, &z)?;
    let a: Basic = ns(
        &m2(&m2(&d2(&int(2), &pi())?, &log(&d2(&z, &int(2))?)?)?, &bn)?,
        x,
        n,
        logx,
        cdir,
    )?;
    let (mut b, mut c): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    let o: Basic = order_xn(x, n)?;
    let r: Basic = half_series(&z, x, n, logx, cdir)?;
    if is_zero_number(&r) {
        return Ok(Some(o));
    }
    let t: Basic = remove_o(&a2(&mexpand(&pw(&r, &int(2))?)?, &o)?)?;
    let k_nu: i64 = int_of(&nu)?;
    if k_nu > 0 {
        let mut term: Basic = d2(&m2(&pw(&r, &neg(&nu)?)?, &factorial(&s2(&nu, &int(1))?)?)?, &pi())?;
        b.push(term.clone());
        for k in 1..k_nu {
            let denom: i64 = (k_nu - k) * k;
            term = m2(
                &term,
                &if denom == 0 {
                    d2(&t, &int(k))?
                } else {
                    d2(&t, &int(denom))?
                },
            )?;
            term = truncate(&term, &o)?;
            b.push(term.clone());
        }
    }
    let mut p: Basic = d2(&pw(&r, &nu)?, &m2(&pi(), &factorial(&nu)?)?)?;
    c.push(m2(&p, &s2(&digamma(&a2(&nu, &int(1))?)?, &euler_gamma())?)?);
    for k in 1..(newn + 1) / 2 {
        p = m2(&p, &d2(&neg(&t)?, &m2(&int(k), &a2(&int(k), &nu)?)?)?)?;
        p = truncate(&p, &o)?;
        let digammas: Basic = a2(&digamma(&a2(&a2(&int(k), &nu)?, &int(1))?)?, &digamma(&int(k + 1))?)?;
        c.push(m2(&p, &digammas)?);
    }
    Ok(Some(s2(&s2(&a, &add_all(&b)?)?, &add_all(&c)?)?))
}

fn bessel_k(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> OptR {
    let Some((nu, z, ex, newn)) = bessel_head(f, x, n)? else {
        return Ok(Some(f.clone()));
    };
    if !yes(&ex, Predicate::positive) {
        return generic(f, x, n, logx);
    }
    let r: Basic = half_series(&z, x, n, logx, cdir)?;
    if is_zero_number(&r) {
        return Ok(Some(crate::core::order::order_at_zero(
            &a2(&pw(&z, &neg(&nu)?)?, &pw(&z, &nu)?)?,
            x,
        )?));
    }
    let o: Basic = order_xn(x, n)?;
    if yes(&nu, Predicate::integer) {
        let bn: Basic = crate::functions::special::bessel::bessel(Function::BesselI, &[nu.clone(), z.clone()])?;
        let a: Basic = ns(
            &m2(&m2(&pw(&int(-1), &s2(&nu, &int(1))?)?, &log(&d2(&z, &int(2))?)?)?, &bn)?,
            x,
            n,
            logx,
            cdir,
        )?;
        let (mut b, mut c): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
        let t: Basic = mexpand(&pw(&r, &int(2))?)?;
        let k_nu: i64 = int_of(&nu)?;
        if k_nu > 0 {
            let mut term: Basic = d2(&m2(&pw(&r, &neg(&nu)?)?, &factorial(&s2(&nu, &int(1))?)?)?, &int(2))?;
            b.push(term.clone());
            for k in 1..k_nu {
                term = m2(&term, &d2(&t, &int((k - k_nu) * k))?)?;
                term = truncate(&term, &o)?;
                b.push(term.clone());
            }
        }
        let mut p: Basic = d2(
            &m2(&pw(&r, &nu)?, &pw(&int(-1), &nu)?)?,
            &m2(&int(2), &factorial(&nu)?)?,
        )?;
        c.push(m2(&p, &s2(&digamma(&a2(&nu, &int(1))?)?, &euler_gamma())?)?);
        for k in 1..(newn + 1) / 2 {
            p = m2(&p, &d2(&t, &m2(&int(k), &a2(&int(k), &nu)?)?)?)?;
            p = truncate(&p, &o)?;
            let digammas: Basic = a2(&digamma(&a2(&a2(&int(k), &nu)?, &int(1))?)?, &digamma(&int(k + 1))?)?;
            c.push(m2(&p, &digammas)?);
        }
        return Ok(Some(add_all(&[a, add_all(&b)?, add_all(&c)?, o])?));
    }
    if yes(&nu, Predicate::noninteger) {
        let newn_a: i64 = int_of(&ceiling(&d2(&a2(&int(n), &nu)?, &ex)?)?)?;
        let newn_b: i64 = int_of(&ceiling(&d2(&s2(&int(n), &nu)?, &ex)?)?)?;
        let rf = crate::functions::combinatorial::factorials::rf;
        let mut terms: Vec<Basic> = Vec::new();
        for k in 0..(newn_a + 1) / 2 {
            let kk: Basic = int(k);
            let den: Basic = m2(&m2(&int(2), &rf(&s2(&int(1), &nu)?, &kk)?)?, &factorial(&kk)?)?;
            terms.push(mexpand(&d2(
                &m2(&gamma(&nu)?, &pw(&r, &s2(&int(2 * k), &nu)?)?)?,
                &den,
            )?)?);
        }
        let a_part: Basic = add_all(&terms)?;
        let mut terms: Vec<Basic> = Vec::new();
        for k in 0..(newn_b + 1) / 2 {
            let kk: Basic = int(k);
            let den: Basic = m2(&m2(&int(2), &rf(&a2(&nu, &int(1))?, &kk)?)?, &factorial(&kk)?)?;
            terms.push(mexpand(&d2(
                &m2(&gamma(&neg(&nu)?)?, &pw(&r, &a2(&int(2 * k), &nu)?)?)?,
                &den,
            )?)?);
        }
        return Ok(Some(add_all(&[a_part, add_all(&terms)?, o])?));
    }
    Err(PolyError::NotImplemented(
        "besselk expansion is only implemented for real order".to_string(),
    ))
}

/// The ported rule of the class of `f` (`rules()[type(f).__name__]`):
/// `None` when it has none or it falls back.
pub fn series_rule(f: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> OptR {
    match f.class_name() {
        "LambertW" => lambertw(f, x, n, logx),
        "Abs" => abs_rule(f, x, n, logx),
        "Sign" => sign_rule(f, x, cdir),
        "Arg" => arg_rule(f, x, n, logx, cdir),
        "Gamma" => gamma_rule(f, x, n, logx),
        "LogGamma" => loggamma_rule(f, x, n, logx),
        "Expint" => expint_rule(f, x, n, logx),
        "EiNode" => ei_rule(f, x, n, logx),
        "LiNode" => li_rule(f, n),
        "OffsetLiNode" => offset_li_rule(f, x, n, logx),
        "SingularityFunctionNode" => singularity_rule(f, x, n, logx, cdir),
        "EllipticK" => elliptic_rule(f, x, n, logx, Basic::half()),
        "EllipticE" => elliptic_rule(f, x, n, logx, Basic::from_rational(crate::numbers::rat(-1, 2))),
        "Floor" => round_rule(f, x, n, logx, cdir, false),
        "Ceiling" => round_rule(f, x, n, logx, cdir, true),
        "Frac" => frac_rule(f, x, n, logx, cdir),
        "PolyLog" => polylog_rule(f, x, n, logx, cdir),
        "BesselJ" => bessel_ji(f, x, n, logx, cdir, false),
        "BesselI" => bessel_ji(f, x, n, logx, cdir, true),
        "BesselY" => bessel_y(f, x, n, logx, cdir),
        "BesselK" => bessel_k(f, x, n, logx, cdir),
        "SiNode" => trig_integral(f, x, n, logx, Function::Sin),
        "CiNode" => trig_integral(f, x, n, logx, Function::Cos),
        "ShiNode" => trig_integral(f, x, n, logx, Function::Sinh),
        "ChiNode" => trig_integral(f, x, n, logx, Function::Cosh),
        _ => Ok(None),
    }
}
