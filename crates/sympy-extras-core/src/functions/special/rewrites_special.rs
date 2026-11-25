//! Port of the non-Bessel rules of `sympy_extras/functions/special/_rewrites.py`
//! (SymPy's `_eval_rewrite_as_<name>` of `error_functions.py`,
//! `gamma_functions.py`, `zeta_functions.py`, `beta_functions.py`,
//! `elliptic_integrals.py`, `hyper.py` and `marcumq`), keyed by the head
//! and the SymPy name of the target; a rule returns `None` where SymPy
//! leaves the node. The rules of `erfc`, `erfi`, `erf2` and the Fresnel
//! integrals into `erf`, of `Li` and `Ei` into `li` are in
//! `rewrites_bessel.rs`. Not ported: `Si` into `Integral` and `sinc`
//! (no `sinc` in the core).

// PORTED FROM SYMPY: sympy/functions/special/error_functions.py, gamma_functions.py, zeta_functions.py,
// beta_functions.py, elliptic_integrals.py, hyper.py, bessel.py (marcumq) (SymPy 1.14, BSD license; the
// _eval_rewrite_as_<name> methods, through sympy_extras/functions/special/_rewrites.py)

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::rewriting::{rewrite, RewriteHints, Rule};
use crate::error::PolyResult;
use crate::numbers::rat;

type O = PolyResult<Option<Basic>>;
type R = PolyResult<Basic>;

/// The targets of these rules.
pub const TARGETS: [&str; 32] = [
    "erf",
    "erfc",
    "erfi",
    "fresnels",
    "fresnelc",
    "meijerg",
    "hyper",
    "uppergamma",
    "lowergamma",
    "expint",
    "erfinv",
    "erfcinv",
    "Integral",
    "Ei",
    "li",
    "Li",
    "Si",
    "Ci",
    "Shi",
    "Chi",
    "zeta",
    "harmonic",
    "polygamma",
    "lerchphi",
    "polylog",
    "bernoulli",
    "dirichlet_eta",
    "genocchi",
    "besseli",
    "Sum",
    "nonrep",
    "nonrepsmall",
];

fn q(p: i64, d: i64) -> Basic {
    Basic::from_rational(rat(p, d))
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn one() -> Basic {
    Basic::one()
}
fn zero() -> Basic {
    Basic::zero()
}
fn half() -> Basic {
    Basic::half()
}
fn pi() -> Basic {
    Basic::pi()
}
fn imag() -> Basic {
    Basic::imaginary_unit()
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn sqrt(e: &Basic) -> R {
    pow(e, &half())
}
fn call(f: Function, args: &[Basic]) -> R {
    crate::functions::apply_function(f, args)
}
fn log(e: &Basic) -> R {
    call(Function::Log, std::slice::from_ref(e))
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn sin(e: &Basic) -> R {
    call(Function::Sin, std::slice::from_ref(e))
}
fn cos(e: &Basic) -> R {
    call(Function::Cos, std::slice::from_ref(e))
}
fn polar_lift(e: &Basic) -> R {
    crate::functions::elementary::polar::polar_lift(e)
}
fn exp_polar(e: &Basic) -> R {
    crate::functions::elementary::polar::exp_polar(e)
}
fn erf(z: &Basic) -> R {
    call(Function::Erf, std::slice::from_ref(z))
}
fn erfc(z: &Basic) -> R {
    call(Function::Erfc, std::slice::from_ref(z))
}
fn erfi(z: &Basic) -> R {
    call(Function::Erfi, std::slice::from_ref(z))
}
fn fresnels(z: &Basic) -> R {
    call(Function::FresnelS, std::slice::from_ref(z))
}
fn fresnelc(z: &Basic) -> R {
    call(Function::FresnelC, std::slice::from_ref(z))
}
fn expint(nu: &Basic, z: &Basic) -> R {
    call(Function::Expint, &[nu.clone(), z.clone()])
}
/// `E1(z)`, `expint(1, z)`.
fn e1(z: &Basic) -> R {
    expint(&one(), z)
}
fn uppergamma(a: &Basic, z: &Basic) -> R {
    call(Function::UpperGamma, &[a.clone(), z.clone()])
}
fn lowergamma(a: &Basic, z: &Basic) -> R {
    call(Function::LowerGamma, &[a.clone(), z.clone()])
}
fn gamma(z: &Basic) -> R {
    call(Function::Gamma, std::slice::from_ref(z))
}
fn one_arg(f: Function, z: &Basic) -> R {
    call(f, std::slice::from_ref(z))
}
fn hyper(ap: &[Basic], bq: &[Basic], z: &Basic) -> R {
    crate::functions::special::hyper::hyper(ap, bq, z)
}
fn meijerg(an: &[Basic], ap: &[Basic], bm: &[Basic], bq: &[Basic], z: &Basic) -> R {
    crate::functions::special::hyper::meijerg(an, ap, bm, bq, z)
}
fn factorial(n: &Basic) -> R {
    crate::functions::combinatorial::factorials::factorial(n)
}
fn rw(e: &Basic, target: &str, hints: &RewriteHints) -> R {
    rewrite(e, &[], target, hints)
}
/// The value of an `Integer` (Python's `isinstance(e, Integer)`).
fn integer_of(c: &Basic) -> Option<i64> {
    match c.head() {
        Head::Rational {
            integer_class: true, ..
        } => c.as_rational().and_then(|v| i64::try_from(v.numerator()).ok()),
        _ => None,
    }
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

/// SymPy's `Dummy(uniquely_named_symbol(name, args).name)`.
fn named_dummy(name: &str, args: &[Basic]) -> Basic {
    let mut names: Vec<String> = Vec::new();
    for a in args {
        for s in a.atoms(Some(&|n: &Basic| n.is_symbol())) {
            names.push(s.to_string());
        }
    }
    let mut chosen: String = name.to_string();
    if names.contains(&chosen) {
        let mut k: usize = 0;
        while names.iter().any(|s| *s == format!("{}{}", name, k)) {
            k += 1;
        }
        chosen = format!("{}{}", name, k);
    }
    crate::core::symbol::Dummy::new(&chosen, &[]).0
}

/// SymPy's `Integral(f, (t, a, b))`, unevaluated.
fn integral(f: &Basic, t: &Basic, a: &Basic, b: &Basic) -> R {
    crate::integrals::integral::integral(f, t, Some((a, b)))
}

// -- the error functions and the Fresnel integrals

fn sqrt_pi() -> R {
    sqrt(&pi())
}

fn fresnel_form(z: &Basic) -> R {
    let arg: Basic = div(&mul(&[sub(&one(), &imag())?, z.clone()])?, &sqrt_pi()?)?;
    mul(&[
        add(&[one(), imag()])?,
        sub(&fresnelc(&arg)?, &mul(&[imag(), fresnels(&arg)?])?)?,
    ])
}

fn erfi_fresnel_form(z: &Basic) -> R {
    let arg: Basic = div(&mul(&[add(&[one(), imag()])?, z.clone()])?, &sqrt_pi()?)?;
    mul(&[
        sub(&one(), &imag())?,
        sub(&fresnelc(&arg)?, &mul(&[imag(), fresnels(&arg)?])?)?,
    ])
}

fn erf_meijerg(z: &Basic) -> R {
    let g: Basic = meijerg(&[half()], &[], &[zero()], &[q(-1, 2)], &pow(z, &int(2))?)?;
    mul(&[div(z, &sqrt_pi()?)?, g])
}

fn erf_hyper(z: &Basic) -> R {
    let h: Basic = hyper(&[half()], &[q(3, 2)], &neg(&pow(z, &int(2))?)?)?;
    mul(&[int(2), div(z, &sqrt_pi()?)?, h])
}

fn erf_uppergamma(z: &Basic) -> R {
    let z2: Basic = pow(z, &int(2))?;
    mul(&[
        div(&sqrt(&z2)?, z)?,
        sub(&one(), &div(&uppergamma(&half(), &z2)?, &sqrt_pi()?)?)?,
    ])
}

fn erf_expint(z: &Basic) -> R {
    let z2: Basic = pow(z, &int(2))?;
    sub(
        &div(&sqrt(&z2)?, z)?,
        &div(&mul(&[z.clone(), expint(&half(), &z2)?])?, &sqrt_pi()?)?,
    )
}

fn erf_erfc(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&one(), &erfc(&a[0])?)?))
}
fn erf_erfi(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(mul(&[neg(&imag())?, erfi(&mul(&[imag(), a[0].clone()])?)?])?))
}
fn erf_fresnel(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(fresnel_form(&a[0])?))
}
fn erf_meijerg_rule(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(erf_meijerg(&a[0])?))
}
fn erf_hyper_rule(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(erf_hyper(&a[0])?))
}
fn erf_uppergamma_rule(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(erf_uppergamma(&a[0])?))
}
fn erf_expint_rule(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(erf_expint(&a[0])?))
}
fn erfc_erfi(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(add(&[
        one(),
        mul(&[imag(), erfi(&mul(&[imag(), a[0].clone()])?)?])?,
    ])?))
}
fn erfc_fresnel(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&one(), &fresnel_form(&a[0])?)?))
}
fn erfc_meijerg(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&one(), &erf_meijerg(&a[0])?)?))
}
fn erfc_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&one(), &erf_hyper(&a[0])?)?))
}
fn erfc_uppergamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&one(), &erf_uppergamma(&a[0])?)?))
}
fn erfc_expint(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let z2: Basic = pow(z, &int(2))?;
    Ok(Some(add(&[
        sub(&one(), &div(&sqrt(&z2)?, z)?)?,
        div(&mul(&[z.clone(), expint(&half(), &z2)?])?, &sqrt_pi()?)?,
    ])?))
}
fn erfi_erfc(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(
        &mul(&[imag(), erfc(&mul(&[imag(), a[0].clone()])?)?])?,
        &imag(),
    )?))
}
fn erfi_fresnel(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(erfi_fresnel_form(&a[0])?))
}
fn erfi_meijerg(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let g: Basic = meijerg(&[half()], &[], &[zero()], &[q(-1, 2)], &neg(&pow(z, &int(2))?)?)?;
    Ok(Some(mul(&[div(z, &sqrt_pi()?)?, g])?))
}
fn erfi_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let h: Basic = hyper(&[half()], &[q(3, 2)], &pow(z, &int(2))?)?;
    Ok(Some(mul(&[int(2), div(z, &sqrt_pi()?)?, h])?))
}
fn erfi_uppergamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let m: Basic = neg(&pow(z, &int(2))?)?;
    Ok(Some(mul(&[
        div(&sqrt(&m)?, z)?,
        sub(&div(&uppergamma(&half(), &m)?, &sqrt_pi()?)?, &one())?,
    ])?))
}
fn erfi_expint(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    // SymPy's sqrt(-z**2)/z - z*expint(1/2, -z**2)/sqrt(pi) is off by
    // 2*sqrt(-z**2)/z: -I*erf(I*z) with erf's rule
    let z: &Basic = &a[0];
    let m: Basic = neg(&pow(z, &int(2))?)?;
    Ok(Some(sub(
        &neg(&div(&sqrt(&m)?, z)?)?,
        &div(&mul(&[z.clone(), expint(&half(), &m)?])?, &sqrt_pi()?)?,
    )?))
}
/// `erf2(x, y)` as `erf(y) - erf(x)` with both rewritten.
fn erf2_via(target: &str, a: &[Basic], h: &RewriteHints) -> O {
    let (x, y) = (&a[0], &a[1]);
    Ok(Some(sub(&rw(&erf(y)?, target, h)?, &rw(&erf(x)?, target, h)?)?))
}
fn erf2_fresnels(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    erf2_via("fresnels", a, h)
}
fn erf2_fresnelc(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    erf2_via("fresnelc", a, h)
}
fn erf2_meijerg(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    erf2_via("meijerg", a, h)
}
fn erf2_hyper(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    erf2_via("hyper", a, h)
}
fn erf2_expint(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    erf2_via("expint", a, h)
}
fn erf2_uppergamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&erf_uppergamma(&a[1])?, &erf_uppergamma(&a[0])?)?))
}
fn erf2_erfc(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&erfc(&a[0])?, &erfc(&a[1])?)?))
}
fn erf2_erfi(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (x, y) = (&a[0], &a[1]);
    let d: Basic = sub(&erfi(&mul(&[imag(), x.clone()])?)?, &erfi(&mul(&[imag(), y.clone()])?)?)?;
    Ok(Some(mul(&[imag(), d])?))
}
fn erfinv_erfcinv(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(one_arg(Function::Erfcinv, &sub(&one(), &a[0])?)?))
}
fn erfcinv_erfinv(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(one_arg(Function::Erfinv, &sub(&one(), &a[0])?)?))
}

fn fresnel_scales() -> PolyResult<(Basic, Basic)> {
    let s: Basic = sqrt_pi()?;
    Ok((
        mul(&[div(&add(&[one(), imag()])?, &int(2))?, s.clone()])?,
        mul(&[div(&sub(&one(), &imag())?, &int(2))?, s])?,
    ))
}

fn fresnels_erf(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (plus, minus) = fresnel_scales()?;
    let z: &Basic = &a[0];
    let inner: Basic = sub(
        &erf(&mul(&[plus, z.clone()])?)?,
        &mul(&[imag(), erf(&mul(&[minus, z.clone()])?)?])?,
    )?;
    Ok(Some(mul(&[div(&add(&[one(), imag()])?, &int(4))?, inner])?))
}
fn fresnelc_erf(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (plus, minus) = fresnel_scales()?;
    let z: &Basic = &a[0];
    let inner: Basic = add(&[
        erf(&mul(&[plus, z.clone()])?)?,
        mul(&[imag(), erf(&mul(&[minus, z.clone()])?)?])?,
    ])?;
    Ok(Some(mul(&[div(&sub(&one(), &imag())?, &int(4))?, inner])?))
}
/// `-pi**2*z**4/16`.
fn fresnel_argument(z: &Basic) -> R {
    div(&neg(&mul(&[pow(&pi(), &int(2))?, pow(z, &int(4))?])?)?, &int(16))
}
fn fresnels_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let h: Basic = hyper(&[q(3, 4)], &[q(3, 2), q(7, 4)], &fresnel_argument(z)?)?;
    Ok(Some(mul(&[div(&mul(&[pi(), pow(z, &int(3))?])?, &int(6))?, h])?))
}
fn fresnelc_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let h: Basic = hyper(&[q(1, 4)], &[half(), q(5, 4)], &fresnel_argument(z)?)?;
    Ok(Some(mul(&[z.clone(), h])?))
}
fn fresnels_meijerg(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let den: Basic = mul(&[
        sqrt(&int(2))?,
        pow(&pow(z, &int(2))?, &q(3, 4))?,
        pow(&neg(z)?, &q(3, 4))?,
    ])?;
    let g: Basic = meijerg(&[], &[one()], &[q(3, 4)], &[q(1, 4), zero()], &fresnel_argument(z)?)?;
    Ok(Some(mul(&[div(&mul(&[pi(), pow(z, &q(9, 4))?])?, &den)?, g])?))
}
fn fresnelc_meijerg(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let den: Basic = mul(&[
        sqrt(&int(2))?,
        pow(&pow(z, &int(2))?, &q(1, 4))?,
        pow(&neg(z)?, &q(1, 4))?,
    ])?;
    let g: Basic = meijerg(&[], &[one()], &[q(1, 4)], &[q(3, 4), zero()], &fresnel_argument(z)?)?;
    Ok(Some(mul(&[div(&mul(&[pi(), pow(z, &q(3, 4))?])?, &den)?, g])?))
}
fn fresnel_integral(a: &[Basic], trig: fn(&Basic) -> R) -> O {
    let z: &Basic = &a[0];
    let t: Basic = named_dummy("t", std::slice::from_ref(z));
    let f: Basic = trig(&div(&mul(&[pi(), pow(&t, &int(2))?])?, &int(2))?)?;
    Ok(Some(integral(&f, &t, &zero(), z)?))
}
fn fresnels_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    fresnel_integral(a, sin)
}
fn fresnelc_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    fresnel_integral(a, cos)
}

// -- the exponential, logarithmic and trigonometric integrals

fn expint_ei(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (nu, z) = (&a[0], &a[1]);
    let shifted: Basic = mul(&[z.clone(), exp_polar(&mul(&[neg(&imag())?, pi()])?)?])?;
    let e1: Basic = sub(&neg(&one_arg(Function::Ei, &shifted)?)?, &mul(&[imag(), pi()])?)?;
    if *nu == one() {
        return Ok(Some(e1));
    }
    if let Some(m) = integer_of(nu) {
        if m > 1 {
            // DLMF 8.19.7
            let x: Basic = neg(&crate::functions::elementary::polar::unpolarify(z, false)?)?;
            let f: Basic = factorial(&int(m - 1))?;
            let mut terms: Vec<Basic> = Vec::new();
            for k in 0..m - 1 {
                terms.push(mul(&[factorial(&int(m - k - 2))?, pow(&x, &int(k))?])?);
            }
            let first: Basic = mul(&[div(&pow(&x, &sub(nu, &one())?)?, &f)?, e1])?;
            let second: Basic = mul(&[div(&exp(&x)?, &f)?, add(&terms)?])?;
            return Ok(Some(add(&[first, second])?));
        }
    }
    Ok(None)
}
fn ei_expint(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let arg: Basic = mul(&[polar_lift(&int(-1))?, a[0].clone()])?;
    Ok(Some(sub(&neg(&expint(&one(), &arg)?)?, &mul(&[imag(), pi()])?)?))
}
fn ei_si(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let s: Basic = add(&[one_arg(Function::Shi, z)?, one_arg(Function::Chi, z)?])?;
    if yes(z, Predicate::negative) {
        return Ok(Some(sub(&s, &mul(&[imag(), pi()])?)?));
    }
    Ok(Some(s))
}
fn ei_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let t: Basic = named_dummy("t", std::slice::from_ref(z));
    let f: Basic = div(&pow(&Basic::e(), &t)?, &t)?;
    Ok(Some(integral(&f, &t, &Basic::negative_oo(), z)?))
}
fn ei_uppergamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let arg: Basic = mul(&[polar_lift(&int(-1))?, a[0].clone()])?;
    Ok(Some(sub(&neg(&uppergamma(&zero(), &arg)?)?, &mul(&[imag(), pi()])?)?))
}
fn expint_si(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (nu, z) = (&a[0], &a[1]);
    if *nu != one() {
        return Ok(Some(e.clone()));
    }
    Ok(Some(sub(&one_arg(Function::Shi, z)?, &one_arg(Function::Chi, z)?)?))
}
fn expint_integral(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (n, x) = (&e.args()[0], &e.args()[1]);
    let t: Basic = named_dummy("t", a);
    let f: Basic = mul(&[pow(&t, &neg(n)?)?, exp(&neg(&mul(&[t.clone(), x.clone()])?)?)?])?;
    Ok(Some(integral(&f, &t, &one(), &Basic::oo())?))
}
fn expint_uppergamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (nu, z) = (&a[0], &a[1]);
    Ok(Some(mul(&[
        pow(z, &sub(nu, &one())?)?,
        uppergamma(&sub(&one(), nu)?, z)?,
    ])?))
}
fn li_offset(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(add(&[
        one_arg(Function::OffsetLogIntegral, &a[0])?,
        one_arg(Function::LogIntegral, &int(2))?,
    ])?))
}
fn li_ei(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(one_arg(Function::Ei, &log(&a[0])?)?))
}
/// `(log(log(z)) - log(1/log(z)))/2`.
fn half_log(z: &Basic) -> R {
    let l: Basic = log(z)?;
    mul(&[half(), sub(&log(&l)?, &log(&div(&one(), &l)?)?)?])
}
fn li_uppergamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let ml: Basic = neg(&log(z)?)?;
    Ok(Some(add(&[
        neg(&uppergamma(&zero(), &ml)?)?,
        half_log(z)?,
        neg(&log(&ml)?)?,
    ])?))
}
fn li_si(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let l: Basic = log(z)?;
    let il: Basic = mul(&[imag(), l.clone()])?;
    Ok(Some(add(&[
        one_arg(Function::Ci, &il)?,
        neg(&mul(&[imag(), one_arg(Function::Si, &il)?])?)?,
        neg(&mul(&[half(), sub(&log(&div(&one(), &l)?)?, &log(&l)?)?])?)?,
        neg(&log(&il)?)?,
    ])?))
}
fn li_shi(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    // SymPy's Chi(log(z)) - Shi(log(z)) - ... is not li(z): Ei(log(z))
    // with Ei = Chi + Shi and Ei's logarithm
    let z: &Basic = &a[0];
    let l: Basic = log(z)?;
    Ok(Some(add(&[
        one_arg(Function::Chi, &l)?,
        one_arg(Function::Shi, &l)?,
        neg(&mul(&[half(), add(&[log(&div(&one(), &l)?)?, log(&l)?])?])?)?,
    ])?))
}
fn li_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let l: Basic = log(z)?;
    let h: Basic = hyper(&[one(), one()], &[int(2), int(2)], &l)?;
    Ok(Some(add(&[mul(&[l, h])?, half_log(z)?, Basic::euler_gamma()])?))
}
fn li_meijerg(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let l: Basic = log(z)?;
    let ml: Basic = neg(&l)?;
    Ok(Some(add(&[
        neg(&log(&ml)?)?,
        neg(&mul(&[half(), sub(&log(&div(&one(), &l)?)?, &log(&l)?)?])?)?,
        neg(&meijerg(&[], &[one()], &[zero(), zero()], &[], &ml)?)?,
    ])?))
}
/// The `_eval_rewrite_as_expint` of the trigonometric integrals.
fn expint_form(e: &Basic, z: &Basic) -> R {
    let up: Basic = e1(&mul(&[polar_lift(&imag())?, z.clone()])?)?;
    let down: Basic = e1(&mul(&[polar_lift(&neg(&imag())?)?, z.clone()])?)?;
    let pi_i: Basic = mul(&[imag(), pi()])?;
    match e.head() {
        Head::Function(Function::Si) => add(&[div(&pi(), &int(2))?, div(&div(&sub(&up, &down)?, &int(2))?, &imag())?]),
        Head::Function(Function::Ci) => neg(&div(&add(&[up, down])?, &int(2))?),
        Head::Function(Function::Shi) => {
            let flip: Basic = e1(&mul(&[exp_polar(&pi_i)?, z.clone()])?)?;
            sub(&div(&sub(&e1(z)?, &flip)?, &int(2))?, &div(&pi_i, &int(2))?)
        }
        _ => {
            let flip: Basic = e1(&mul(&[exp_polar(&pi_i)?, z.clone()])?)?;
            sub(&neg(&div(&pi_i, &int(2))?)?, &div(&add(&[e1(z)?, flip])?, &int(2))?)
        }
    }
}
fn trig_expint(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(expint_form(e, &a[0])?))
}
fn trig_ei(e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    Ok(Some(rw(&expint_form(e, &a[0])?, "Ei", h)?))
}
fn trig_uppergamma(e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    Ok(Some(rw(&expint_form(e, &a[0])?, "uppergamma", h)?))
}
fn si_integral(_e: &Basic, _a: &[Basic], _h: &RewriteHints) -> O {
    Err(crate::error::not_ported(
        "Si as an Integral of sinc (no sinc in the core)",
    ))
}
fn ci_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &a[0];
    let t: Basic = named_dummy("t", std::slice::from_ref(z));
    let f: Basic = div(&sub(&one(), &cos(&t)?)?, &t)?;
    Ok(Some(sub(
        &add(&[Basic::euler_gamma(), log(z)?])?,
        &integral(&f, &t, &zero(), z)?,
    )?))
}

// -- the incomplete gamma, polygamma and zeta functions

fn lower_upper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&gamma(&a[0])?, &uppergamma(&a[0], &a[1])?)?))
}
fn lower_expint(e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    let s: &Basic = &a[0];
    if yes(s, Predicate::integer) && yes(s, Predicate::nonpositive) {
        return Ok(Some(e.clone()));
    }
    Ok(Some(rw(&rw(e, "uppergamma", h)?, "expint", h)?))
}
fn upper_lower(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(&gamma(&a[0])?, &lowergamma(&a[0], &a[1])?)?))
}
fn upper_expint(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(mul(&[expint(&sub(&one(), &a[0])?, &a[1])?, pow(&a[1], &a[0])?])?))
}
fn zeta(s: &Basic, a: Option<&Basic>) -> R {
    crate::functions::special::zeta_functions::zeta(s, a)
}
fn harmonic(n: &Basic, m: Option<&Basic>) -> R {
    crate::functions::combinatorial::numbers::harmonic(n, m)
}
fn polygamma_zeta(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (n, z) = (&a[0], &a[1]);
    if yes(n, Predicate::integer) && yes(n, Predicate::positive) {
        let n1: Basic = add(&[n.clone(), one()])?;
        return Ok(Some(mul(&[pow(&int(-1), &n1)?, factorial(n)?, zeta(&n1, Some(z))?])?));
    }
    Ok(None)
}
fn polygamma_harmonic(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (n, z) = (&a[0], &a[1]);
    if yes(n, Predicate::integer) {
        let zm: Basic = sub(z, &one())?;
        if yes(n, Predicate::zero) {
            return Ok(Some(sub(&harmonic(&zm, None)?, &Basic::euler_gamma())?));
        }
        let n1: Basic = add(&[n.clone(), one()])?;
        let d: Basic = sub(&zeta(&n1, None)?, &harmonic(&zm, Some(&n1))?)?;
        return Ok(Some(mul(&[pow(&int(-1), &n1)?, factorial(n)?, d])?));
    }
    Ok(None)
}
fn digamma_harmonic(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(sub(
        &harmonic(&sub(&a[0], &one())?, None)?,
        &Basic::euler_gamma(),
    )?))
}
fn digamma_polygamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(call(Function::PolyGamma, &[zero(), a[0].clone()])?))
}
fn trigamma_zeta(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(zeta(&int(2), Some(&a[0]))?))
}
fn trigamma_polygamma(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(call(Function::PolyGamma, &[one(), a[0].clone()])?))
}
fn trigamma_harmonic(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let hm: Basic = harmonic(&sub(&a[0], &one())?, Some(&int(2)))?;
    Ok(Some(add(&[neg(&hm)?, div(&pow(&pi(), &int(2))?, &int(6))?])?))
}
fn has_function(e: &Basic, f: Function) -> bool {
    e.is_function(f) || e.args().iter().any(|a| has_function(a, f))
}
fn lerchphi_via(e: &Basic, target: Function) -> O {
    let res: Option<Basic> = crate::functions::expand_func::lerchphi_rule(e)?;
    match res {
        Some(r) if has_function(&r, target) => Ok(Some(r)),
        _ => Ok(Some(e.clone())),
    }
}
fn lerchphi_zeta(e: &Basic, _a: &[Basic], _h: &RewriteHints) -> O {
    lerchphi_via(e, Function::Zeta)
}
fn lerchphi_polylog(e: &Basic, _a: &[Basic], _h: &RewriteHints) -> O {
    lerchphi_via(e, Function::PolyLog)
}
fn polylog_lerchphi(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let l: Basic = crate::functions::special::zeta_functions::lerchphi(&a[1], &a[0], &one())?;
    Ok(Some(mul(&[a[1].clone(), l])?))
}
fn s_a(a: &[Basic]) -> (Basic, Basic) {
    (a[0].clone(), if a.len() > 1 { a[1].clone() } else { one() })
}
fn zeta_bernoulli(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    use crate::functions::combinatorial::numbers::bernoulli;
    let (s, av) = s_a(a);
    if av == one() && yes(&s, Predicate::integer) && yes(&s, Predicate::nonnegative) && yes(&s, Predicate::even) {
        let c: Basic = pow(&mul(&[int(2), pi(), imag()])?, &s)?;
        return Ok(Some(neg(&div(
            &mul(&[c, bernoulli(&s, None)?])?,
            &mul(&[int(2), factorial(&s)?])?,
        )?)?));
    }
    Ok(Some(div(&bernoulli(&sub(&one(), &s)?, Some(&av))?, &sub(&s, &one())?)?))
}
fn zeta_eta(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (_, av) = s_a(a);
    if av != one() {
        return Ok(Some(e.clone()));
    }
    let s: &Basic = &e.args()[0];
    let eta: Basic = crate::functions::special::zeta_functions::dirichlet_eta(s, None)?;
    Ok(Some(div(&eta, &sub(&one(), &pow(&int(2), &sub(&one(), s)?)?)?)?))
}
fn zeta_lerchphi(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (s, av) = s_a(a);
    Ok(Some(crate::functions::special::zeta_functions::lerchphi(
        &one(),
        &s,
        &av,
    )?))
}
fn eta_zeta(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    use crate::functions::elementary::piecewise::piecewise;
    let (s, av) = s_a(a);
    let two: Basic = int(2);
    let at_one: Basic = crate::core::relational::eq(&s, &one())?;
    if av == one() {
        let rest: Basic = mul(&[sub(&one(), &pow(&two, &sub(&one(), &s)?)?)?, zeta(&s, None)?])?;
        return Ok(Some(piecewise(&[
            (log(&two)?, at_one),
            (rest, crate::core::logic::true_()),
        ])?));
    }
    let digamma = |z: &Basic| one_arg(Function::DiGamma, z);
    let a1: Basic = div(&add(&[av.clone(), one()])?, &two)?;
    let first: Basic = add(&[log(&two)?, neg(&digamma(&av)?)?, digamma(&a1)?])?;
    let rest: Basic = sub(
        &zeta(&s, Some(&av))?,
        &mul(&[pow(&two, &sub(&one(), &s)?)?, zeta(&s, Some(&a1))?])?,
    )?;
    Ok(Some(piecewise(&[
        (first, at_one),
        (rest, crate::core::logic::true_()),
    ])?))
}
fn eta_genocchi(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    use crate::functions::elementary::piecewise::piecewise;
    let (s, av) = s_a(a);
    let two: Basic = int(2);
    let digamma = |z: &Basic| one_arg(Function::DiGamma, z);
    let a1: Basic = div(&add(&[av.clone(), one()])?, &two)?;
    let first: Basic = add(&[log(&two)?, neg(&digamma(&av)?)?, digamma(&a1)?])?;
    let g: Basic = crate::functions::combinatorial::numbers::genocchi(&sub(&one(), &s)?, Some(&av))?;
    let rest: Basic = div(&g, &mul(&[two, sub(&s, &one())?])?)?;
    let at_one: Basic = crate::core::relational::eq(&s, &one())?;
    Ok(Some(piecewise(&[
        (first, at_one),
        (rest, crate::core::logic::true_()),
    ])?))
}
fn xi_zeta(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let s: &Basic = &a[0];
    let half_s: Basic = div(s, &int(2))?;
    let num: Basic = mul(&[s.clone(), sub(s, &one())?, gamma(&half_s)?, zeta(s, None)?])?;
    Ok(Some(div(&num, &mul(&[int(2), pow(&pi(), &half_s)?])?)?))
}

// -- the beta functions, the elliptic integrals, Marcum's Q-function

fn beta_kernel(t: &Basic, x: &Basic, y: &Basic) -> R {
    mul(&[pow(t, &sub(x, &one())?)?, pow(&sub(&one(), t)?, &sub(y, &one())?)?])
}
fn beta_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let x: &Basic = &a[0];
    let y: &Basic = if a.len() > 1 { &a[1] } else { x };
    let t: Basic = named_dummy("t", &[x.clone(), y.clone()]);
    Ok(Some(integral(&beta_kernel(&t, x, y)?, &t, &zero(), &one())?))
}
fn betainc_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let t: Basic = named_dummy("t", a);
    Ok(Some(integral(&beta_kernel(&t, &a[0], &a[1])?, &t, &a[2], &a[3])?))
}
fn betainc_hyper_form(a: &[Basic]) -> R {
    let (av, b, x1, x2) = (&a[0], &a[1], &a[2], &a[3]);
    let up: Vec<Basic> = vec![av.clone(), sub(&one(), b)?];
    let low: Vec<Basic> = vec![add(&[av.clone(), one()])?];
    let d: Basic = sub(
        &mul(&[pow(x2, av)?, hyper(&up, &low, x2)?])?,
        &mul(&[pow(x1, av)?, hyper(&up, &low, x1)?])?,
    )?;
    div(&d, av)
}
fn betainc_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(betainc_hyper_form(a)?))
}
fn regularized_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let t: Basic = named_dummy("t", a);
    let f: Basic = beta_kernel(&t, &a[0], &a[1])?;
    Ok(Some(div(
        &integral(&f, &t, &a[2], &a[3])?,
        &integral(&f, &t, &zero(), &one())?,
    )?))
}
fn regularized_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let beta: Basic = call(Function::Beta, &[a[0].clone(), a[1].clone()])?;
    Ok(Some(div(&betainc_hyper_form(a)?, &beta)?))
}
/// `1 - m*sin(t)**2`.
fn elliptic_kernel(m: &Basic, t: &Basic) -> R {
    sub(&one(), &mul(&[m.clone(), pow(&sin(t)?, &int(2))?])?)
}
fn k_integral(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let t: Basic = named_dummy("t", a);
    let m: &Basic = &e.args()[0];
    let f: Basic = div(&one(), &sqrt(&elliptic_kernel(m, &t)?)?)?;
    Ok(Some(integral(&f, &t, &zero(), &div(&pi(), &int(2))?)?))
}
fn f_integral(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let t: Basic = named_dummy("t", a);
    let (z, m) = (&e.args()[0], &e.args()[1]);
    let f: Basic = div(&one(), &sqrt(&elliptic_kernel(m, &t)?)?)?;
    Ok(Some(integral(&f, &t, &zero(), z)?))
}
fn k_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(mul(&[pi(), half(), hyper(&[half(), half()], &[one()], &a[0])?])?))
}
fn k_meijerg(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let g: Basic = meijerg(&[half(), half()], &[], &[zero()], &[zero()], &neg(&a[0])?)?;
    Ok(Some(div(&g, &int(2))?))
}
fn e_hyper(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    if a.len() != 1 {
        return Ok(None);
    }
    Ok(Some(mul(&[
        div(&pi(), &int(2))?,
        hyper(&[q(-1, 2), half()], &[one()], &a[0])?,
    ])?))
}
fn e_meijerg(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    if a.len() != 1 {
        return Ok(None);
    }
    let g: Basic = meijerg(&[half(), q(3, 2)], &[], &[zero()], &[zero()], &neg(&a[0])?)?;
    Ok(Some(neg(&div(&g, &int(4))?)?))
}
fn e_integral(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let own: &[Basic] = e.args();
    let (z, m): (Basic, Basic) = if own.len() == 1 {
        (div(&pi(), &int(2))?, own[0].clone())
    } else {
        (own[0].clone(), own[1].clone())
    };
    let t: Basic = named_dummy("t", a);
    Ok(Some(integral(&sqrt(&elliptic_kernel(&m, &t)?)?, &t, &zero(), &z)?))
}
fn pi_integral(e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let own: &[Basic] = e.args();
    let (n, m, z): (Basic, Basic, Basic) = if own.len() == 2 {
        (own[0].clone(), own[1].clone(), div(&pi(), &int(2))?)
    } else {
        (own[0].clone(), own[2].clone(), own[1].clone())
    };
    let t: Basic = named_dummy("t", a);
    let den: Basic = mul(&[elliptic_kernel(&n, &t)?, sqrt(&elliptic_kernel(&m, &t)?)?])?;
    Ok(Some(integral(&div(&one(), &den)?, &t, &zero(), &z)?))
}
fn besseli(nu: &Basic, z: &Basic) -> R {
    call(Function::BesselI, &[nu.clone(), z.clone()])
}
fn marcumq_integral(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (m, av, b) = (&a[0], &a[1], &a[2]);
    let x: Basic = crate::core::symbol::Dummy::new("x", &[]).0;
    let g: Basic = exp(&neg(&div(&add(&[pow(&x, &int(2))?, pow(av, &int(2))?])?, &int(2))?)?)?;
    let f: Basic = mul(&[
        pow(&x, m)?,
        g,
        besseli(&sub(m, &one())?, &mul(&[av.clone(), x.clone()])?)?,
    ])?;
    Ok(Some(mul(&[
        pow(av, &sub(&one(), m)?)?,
        integral(&f, &x, b, &Basic::oo())?,
    ])?))
}
fn marcumq_sum(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (m, av, b) = (&a[0], &a[1], &a[2]);
    let k: Basic = crate::core::symbol::Dummy::new("k", &[]).0;
    let g: Basic = exp(&neg(&div(&add(&[pow(av, &int(2))?, pow(b, &int(2))?])?, &int(2))?)?)?;
    let f: Basic = mul(&[pow(&div(av, b)?, &k)?, besseli(&k, &mul(&[av.clone(), b.clone()])?)?])?;
    let lims: Basic = crate::core::function::tuple(vec![k, sub(&one(), m)?, Basic::oo()]);
    Ok(Some(mul(&[g, crate::concrete::expr_with_limits::sum_(&f, &[lims])?])?))
}
fn marcumq_besseli(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    let (m, av, b) = (&a[0], &a[1], &a[2]);
    if av != b {
        return Ok(None);
    }
    let a2: Basic = pow(av, &int(2))?;
    let w: Basic = exp(&neg(&a2)?)?;
    if *m == one() {
        let s: Basic = add(&[one(), mul(&[w, besseli(&zero(), &a2)?])?])?;
        return Ok(Some(div(&s, &int(2))?));
    }
    if let Some(mv) = integer_of(m) {
        if mv >= 2 {
            let mut terms: Vec<Basic> = Vec::new();
            for i in 1..mv {
                terms.push(besseli(&int(i), &a2)?);
            }
            let s: Basic = add(&terms)?;
            return Ok(Some(add(&[
                half(),
                div(&mul(&[w.clone(), besseli(&zero(), &a2)?])?, &int(2))?,
                mul(&[w, s])?,
            ])?));
        }
    }
    Ok(None)
}

// -- hyper as its defining series

/// `hyper._eval_rewrite_as_Sum`: the defining series where it converges,
/// the node elsewhere.
pub fn hyper_sum(e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    use crate::functions::special::hyper::{convergence_statement, hyper_ap, hyper_bq};
    if !matches!(e.head(), Head::Hyper) {
        return Ok(None);
    }
    let rebuilt: Basic = crate::core::function::rebuild_evaluated(e, args.to_vec())?;
    if !matches!(rebuilt.head(), Head::Hyper) {
        return Ok(None);
    }
    let n: Basic = crate::core::symbol::Dummy::new("n", &[(Predicate::integer, true)]).0;
    let rf = |x: &Basic| crate::functions::combinatorial::factorials::rf(x, &n);
    let mut top: Vec<Basic> = Vec::new();
    for a in hyper_ap(&rebuilt) {
        top.push(rf(a)?);
    }
    let mut bottom: Vec<Basic> = Vec::new();
    for b in hyper_bq(&rebuilt) {
        bottom.push(rf(b)?);
    }
    let coeff: Basic = div(&mul(&top)?, &mul(&bottom)?)?;
    let term: Basic = div(&mul(&[coeff, pow(&rebuilt.args()[2], &n)?])?, &factorial(&n)?)?;
    let lims: Basic = crate::core::function::tuple(vec![n.clone(), zero(), Basic::oo()]);
    let series: Basic = crate::concrete::expr_with_limits::sum_(&term, &[lims])?;
    let condition: Basic = convergence_statement(e)?;
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&[
        (series, condition),
        (e.clone(), crate::core::logic::true_()),
    ])?))
}

/// `HyperRep._eval_rewrite_as_nonrep` / `_nonrepsmall`: the closed form of
/// the node itself (as SymPy's).
fn hyperrep_nonrep(e: &Basic, _a: &[Basic], _h: &RewriteHints) -> O {
    match e.head() {
        Head::Function(f) => Ok(Some(crate::functions::special::hyperrep::closed_form(*f, e, false)?)),
        _ => Ok(None),
    }
}
fn hyperrep_nonrepsmall(e: &Basic, _a: &[Basic], _h: &RewriteHints) -> O {
    match e.head() {
        Head::Function(f) => Ok(Some(crate::functions::special::hyperrep::closed_form(*f, e, true)?)),
        _ => Ok(None),
    }
}

/// The rule of the head `f` for `target` (`rules`).
pub fn rule_for(f: Function, target: &str) -> Option<Rule> {
    use Function as F;
    let rule: Rule = match (f, target) {
        (F::Erf, "erfc") => erf_erfc,
        (F::Erf, "erfi") => erf_erfi,
        (F::Erf, "fresnels") | (F::Erf, "fresnelc") => erf_fresnel,
        (F::Erf, "meijerg") => erf_meijerg_rule,
        (F::Erf, "hyper") => erf_hyper_rule,
        (F::Erf, "uppergamma") => erf_uppergamma_rule,
        (F::Erf, "expint") => erf_expint_rule,
        (F::Erfc, "erfi") => erfc_erfi,
        (F::Erfc, "fresnels") | (F::Erfc, "fresnelc") => erfc_fresnel,
        (F::Erfc, "meijerg") => erfc_meijerg,
        (F::Erfc, "hyper") => erfc_hyper,
        (F::Erfc, "uppergamma") => erfc_uppergamma,
        (F::Erfc, "expint") => erfc_expint,
        (F::Erfi, "erfc") => erfi_erfc,
        (F::Erfi, "fresnels") | (F::Erfi, "fresnelc") => erfi_fresnel,
        (F::Erfi, "meijerg") => erfi_meijerg,
        (F::Erfi, "hyper") => erfi_hyper,
        (F::Erfi, "uppergamma") => erfi_uppergamma,
        (F::Erfi, "expint") => erfi_expint,
        (F::Erf2, "erfc") => erf2_erfc,
        (F::Erf2, "erfi") => erf2_erfi,
        (F::Erf2, "fresnels") => erf2_fresnels,
        (F::Erf2, "fresnelc") => erf2_fresnelc,
        (F::Erf2, "meijerg") => erf2_meijerg,
        (F::Erf2, "hyper") => erf2_hyper,
        (F::Erf2, "uppergamma") => erf2_uppergamma,
        (F::Erf2, "expint") => erf2_expint,
        (F::Erfinv, "erfcinv") => erfinv_erfcinv,
        (F::Erfcinv, "erfinv") => erfcinv_erfinv,
        (F::FresnelS, "erf") => fresnels_erf,
        (F::FresnelC, "erf") => fresnelc_erf,
        (F::FresnelS, "hyper") => fresnels_hyper,
        (F::FresnelC, "hyper") => fresnelc_hyper,
        (F::FresnelS, "meijerg") => fresnels_meijerg,
        (F::FresnelC, "meijerg") => fresnelc_meijerg,
        (F::FresnelS, "Integral") => fresnels_integral,
        (F::FresnelC, "Integral") => fresnelc_integral,
        (F::Expint, "Ei") => expint_ei,
        (F::Ei, "expint") => ei_expint,
        (F::Ei, "uppergamma") => ei_uppergamma,
        (F::Ei, "Integral") => ei_integral,
        (F::Ei, "Si") | (F::Ei, "Ci") | (F::Ei, "Chi") | (F::Ei, "Shi") => ei_si,
        (F::Expint, "uppergamma") => expint_uppergamma,
        (F::Expint, "Integral") => expint_integral,
        (F::Expint, "Si") | (F::Expint, "Ci") | (F::Expint, "Chi") | (F::Expint, "Shi") => expint_si,
        (F::LogIntegral, "Li") => li_offset,
        (F::LogIntegral, "Ei") => li_ei,
        (F::LogIntegral, "uppergamma") => li_uppergamma,
        (F::LogIntegral, "Si") | (F::LogIntegral, "Ci") => li_si,
        (F::LogIntegral, "Shi") | (F::LogIntegral, "Chi") => li_shi,
        (F::LogIntegral, "hyper") => li_hyper,
        (F::LogIntegral, "meijerg") => li_meijerg,
        (F::Si | F::Ci | F::Shi | F::Chi, "Ei") => trig_ei,
        (F::Si | F::Ci | F::Shi | F::Chi, "uppergamma") => trig_uppergamma,
        (F::Si | F::Ci | F::Shi | F::Chi, "expint") => trig_expint,
        (F::Si, "Integral") => si_integral,
        (F::Ci, "Integral") => ci_integral,
        (F::LowerGamma, "uppergamma") => lower_upper,
        (F::LowerGamma, "expint") => lower_expint,
        (F::UpperGamma, "lowergamma") => upper_lower,
        (F::UpperGamma, "expint") => upper_expint,
        (F::PolyGamma, "zeta") => polygamma_zeta,
        (F::PolyGamma, "harmonic") => polygamma_harmonic,
        (F::DiGamma, "harmonic") => digamma_harmonic,
        (F::DiGamma, "polygamma") => digamma_polygamma,
        (F::TriGamma, "zeta") => trigamma_zeta,
        (F::TriGamma, "polygamma") => trigamma_polygamma,
        (F::TriGamma, "harmonic") => trigamma_harmonic,
        (F::LerchPhi, "zeta") => lerchphi_zeta,
        (F::LerchPhi, "polylog") => lerchphi_polylog,
        (F::PolyLog, "lerchphi") => polylog_lerchphi,
        (F::Zeta, "bernoulli") => zeta_bernoulli,
        (F::Zeta, "dirichlet_eta") => zeta_eta,
        (F::Zeta, "lerchphi") => zeta_lerchphi,
        (F::DirichletEta, "zeta") => eta_zeta,
        (F::DirichletEta, "genocchi") => eta_genocchi,
        (F::RiemannXi, "zeta") => xi_zeta,
        (F::Beta, "Integral") => beta_integral,
        (F::BetaInc, "Integral") => betainc_integral,
        (F::BetaInc, "hyper") => betainc_hyper,
        (F::BetaIncRegularized, "Integral") => regularized_integral,
        (F::BetaIncRegularized, "hyper") => regularized_hyper,
        (F::EllipticK, "hyper") => k_hyper,
        (F::EllipticK, "meijerg") => k_meijerg,
        (F::EllipticK, "Integral") => k_integral,
        (F::EllipticF, "Integral") => f_integral,
        (F::EllipticE, "hyper") => e_hyper,
        (F::EllipticE, "meijerg") => e_meijerg,
        (F::EllipticE, "Integral") => e_integral,
        (F::EllipticPi, "Integral") => pi_integral,
        (F::MarcumQ, "Integral") => marcumq_integral,
        (F::MarcumQ, "Sum") => marcumq_sum,
        (F::MarcumQ, "besseli") => marcumq_besseli,
        (f, "nonrep") if crate::functions::special::hyperrep::is_hyperrep(f) => hyperrep_nonrep,
        (f, "nonrepsmall") if crate::functions::special::hyperrep::is_hyperrep(f) => hyperrep_nonrepsmall,
        _ => return None,
    };
    Some(rule)
}
