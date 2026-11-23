//! Port of `sympy_extras/functions/special/_rewrites_bessel.py`: the
//! `rewrite` rules of the Bessel, spherical Bessel and Airy functions and
//! of the spherical harmonics (SymPy's `_eval_rewrite_as_<name>` of
//! `bessel.py` and `spherical_harmonics.py`), keyed by the head and the
//! SymPy name of the target. A rule returns `None` where SymPy leaves the
//! node. The Bessel rules of `functions/special/_rewrites.py` are here
//! too. Not ported: the rules through `polar_lift` (`besselj` and
//! `besseli` into each other, `besseli` into `bessely` and `jn`, and the
//! rules composed with them, `bessely` into `besseli` and `besselk` into
//! `besselj`, `bessely` and `yn` for a non-integer order; the core has no
//! polar numbers) and those of `Ynm`
//! into `cos` and `sin` (they need `simplify`).

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function};
use crate::core::expr::{is, is_true};
use crate::core::rewriting::{rewrite, RewriteHints, Rule};
use crate::error::{not_ported, PolyResult};
use crate::numbers::rat;

use super::bessel::bessel;

type O = PolyResult<Option<Basic>>;
type R = PolyResult<Basic>;

fn q(p: i64, d: i64) -> Basic {
    Basic::from_rational(rat(p, d))
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
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
    pow(e, &Basic::half())
}
fn pi() -> Basic {
    Basic::pi()
}
fn i_unit() -> Basic {
    Basic::imaginary_unit()
}
fn b(f: Function, nu: &Basic, z: &Basic) -> R {
    bessel(f, &[nu.clone(), z.clone()])
}
fn half() -> Basic {
    Basic::half()
}
fn third() -> Basic {
    q(1, 3)
}
fn two_thirds() -> Basic {
    q(2, 3)
}

fn rewritten(e: Option<Basic>, target: &str, hints: &RewriteHints) -> O {
    match e {
        None => Ok(None),
        Some(e) => Ok(Some(rewrite(&e, &[], target, hints)?)),
    }
}

fn order_arg(node: &Basic) -> (Basic, Basic) {
    (node.args()[0].clone(), node.args()[1].clone())
}

fn integer_order(args: &[Basic]) -> bool {
    is_true(&args[0], Predicate::integer)
}

// -- the Bessel functions

fn besselj_bessely(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    use crate::functions::elementary::trigonometric::{cot, csc};
    let (nu, z) = (&args[0], &args[1]);
    if is(nu, Predicate::integer) == Some(false) {
        let pinu: Basic = mul(&[pi(), nu.clone()])?;
        return Ok(Some(sub(
            &mul(&[csc(&pinu)?, b(Function::BesselY, &neg(nu)?, z)?])?,
            &mul(&[cot(&pinu)?, b(Function::BesselY, nu, z)?])?,
        )?));
    }
    Ok(None)
}

fn bessely_as_besselj(nu: &Basic, z: &Basic) -> O {
    use crate::functions::elementary::trigonometric::{cos, csc};
    if is(nu, Predicate::integer) == Some(false) {
        let pinu: Basic = mul(&[pi(), nu.clone()])?;
        let inner: Basic = sub(
            &mul(&[cos(&pinu)?, b(Function::BesselJ, nu, z)?])?,
            &b(Function::BesselJ, &neg(nu)?, z)?,
        )?;
        return Ok(Some(mul(&[csc(&pinu)?, inner])?));
    }
    Ok(None)
}

fn besselk_as_besseli(nu: &Basic, z: &Basic) -> O {
    use crate::functions::elementary::trigonometric::csc;
    if is(nu, Predicate::integer) == Some(false) {
        let pinu: Basic = mul(&[pi(), nu.clone()])?;
        let inner: Basic = sub(&b(Function::BesselI, &neg(nu)?, z)?, &b(Function::BesselI, nu, z)?)?;
        return Ok(Some(div(&mul(&[pi(), csc(&pinu)?, inner])?, &int(2))?));
    }
    Ok(None)
}

fn bessely_besseli(e: &Basic, _args: &[Basic], h: &RewriteHints) -> O {
    let (nu, z) = order_arg(e);
    rewritten(bessely_as_besselj(&nu, &z)?, "besseli", h)
}

fn bessely_yn(e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let (nu, z) = (&args[0], &args[1]);
    let factor: Basic = sqrt(&div(&mul(&[int(2), z.clone()])?, &pi())?)?;
    Ok(Some(mul(&[
        factor,
        b(Function::SphericalY, &sub(nu, &half())?, &order_arg(e).1)?,
    ])?))
}

fn besselk_as_bessely(e: &Basic, h: &RewriteHints) -> O {
    let (nu, z) = order_arg(e);
    rewritten(rewritten(besselk_as_besseli(&nu, &z)?, "besselj", h)?, "bessely", h)
}

fn besselk_bessely(e: &Basic, _args: &[Basic], h: &RewriteHints) -> O {
    besselk_as_bessely(e, h)
}

fn besselk_yn(e: &Basic, _args: &[Basic], h: &RewriteHints) -> O {
    rewritten(besselk_as_bessely(e, h)?, "yn", h)
}

// the rules of `functions/special/_rewrites.py` for these heads

fn besselj_jn(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let (nu, z) = (&args[0], &args[1]);
    let factor: Basic = sqrt(&div(&mul(&[int(2), z.clone()])?, &pi())?)?;
    Ok(Some(mul(&[factor, b(Function::SphericalJ, &sub(nu, &half())?, z)?])?))
}

fn bessely_besselj(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    bessely_as_besselj(&args[0], &args[1])
}

fn besselk_besseli(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    besselk_as_besseli(&args[0], &args[1])
}

fn besselk_besselj(_e: &Basic, args: &[Basic], h: &RewriteHints) -> O {
    rewritten(besselk_as_besseli(&args[0], &args[1])?, "besselj", h)
}

fn polar(_e: &Basic, _args: &[Basic], _h: &RewriteHints) -> O {
    Err(not_ported(
        "besseli through polar_lift(I) (the core has no polar numbers)",
    ))
}

// -- the spherical Bessel functions

fn half_pi_over(z: &Basic) -> R {
    sqrt(&div(&pi(), &mul(&[int(2), z.clone()])?)?)
}

fn jn_besselj(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let (nu, z) = (&args[0], &args[1]);
    Ok(Some(mul(&[
        half_pi_over(z)?,
        b(Function::BesselJ, &add(&[nu.clone(), half()])?, z)?,
    ])?))
}

fn jn_bessely(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    Ok(Some(mul(&[
        pow(&int(-1), nu)?,
        half_pi_over(z)?,
        b(Function::BesselY, &sub(&neg(nu)?, &half())?, z)?,
    ])?))
}

fn jn_yn(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    Ok(Some(mul(&[
        pow(&int(-1), nu)?,
        b(Function::SphericalY, &sub(&neg(nu)?, &int(1))?, z)?,
    ])?))
}

fn yn_besselj(e: &Basic, args: &[Basic], h: &RewriteHints) -> O {
    let _ = (e, h);
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    Ok(Some(mul(&[
        pow(&int(-1), &add(&[nu.clone(), int(1)])?)?,
        half_pi_over(z)?,
        b(Function::BesselJ, &sub(&neg(nu)?, &half())?, z)?,
    ])?))
}

fn yn_bessely(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    Ok(Some(mul(&[
        half_pi_over(z)?,
        b(Function::BesselY, &add(&[nu.clone(), half()])?, z)?,
    ])?))
}

fn yn_jn(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    Ok(Some(mul(&[
        pow(&int(-1), &add(&[nu.clone(), int(1)])?)?,
        b(Function::SphericalJ, &sub(&neg(nu)?, &int(1))?, z)?,
    ])?))
}

fn hankel_besselj(args: &[Basic], sign: i64) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    let second: Basic = mul(&[
        int(sign),
        i_unit(),
        pow(&int(-1), &add(&[nu.clone(), int(1)])?)?,
        b(Function::BesselJ, &sub(&neg(nu)?, &half())?, z)?,
    ])?;
    Ok(Some(mul(&[
        half_pi_over(z)?,
        add(&[b(Function::BesselJ, &add(&[nu.clone(), half()])?, z)?, second])?,
    ])?))
}

fn hankel_bessely(args: &[Basic], sign: i64) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    let first: Basic = mul(&[pow(&int(-1), nu)?, b(Function::BesselY, &sub(&neg(nu)?, &half())?, z)?])?;
    let second: Basic = mul(&[
        int(sign),
        i_unit(),
        b(Function::BesselY, &add(&[nu.clone(), half()])?, z)?,
    ])?;
    Ok(Some(mul(&[half_pi_over(z)?, add(&[first, second])?])?))
}

fn hankel_yn(args: &[Basic], sign: i64, h: &RewriteHints) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    let j: Basic = rewrite(&b(Function::SphericalJ, nu, z)?, &[], "yn", h)?;
    Ok(Some(add(&[
        j,
        mul(&[int(sign), i_unit(), b(Function::SphericalY, nu, z)?])?,
    ])?))
}

fn hankel_jn(args: &[Basic], sign: i64, h: &RewriteHints) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    let y: Basic = rewrite(&b(Function::SphericalY, nu, z)?, &[], "jn", h)?;
    Ok(Some(add(&[
        b(Function::SphericalJ, nu, z)?,
        mul(&[int(sign), i_unit(), y])?,
    ])?))
}

fn hankel_hankel(args: &[Basic], f: Function) -> O {
    if !integer_order(args) {
        return Ok(None);
    }
    let (nu, z) = (&args[0], &args[1]);
    Ok(Some(mul(&[half_pi_over(z)?, b(f, &add(&[nu.clone(), half()])?, z)?])?))
}

fn hn1_besselj(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    hankel_besselj(a, 1)
}
fn hn2_besselj(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    hankel_besselj(a, -1)
}
fn hn1_bessely(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    hankel_bessely(a, 1)
}
fn hn2_bessely(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    hankel_bessely(a, -1)
}
fn hn1_yn(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    hankel_yn(a, 1, h)
}
fn hn2_yn(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    hankel_yn(a, -1, h)
}
fn hn1_jn(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    hankel_jn(a, 1, h)
}
fn hn2_jn(_e: &Basic, a: &[Basic], h: &RewriteHints) -> O {
    hankel_jn(a, -1, h)
}
fn hn1_hankel1(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    hankel_hankel(a, Function::Hankel1)
}
fn hn2_hankel2(_e: &Basic, a: &[Basic], _h: &RewriteHints) -> O {
    hankel_hankel(a, Function::Hankel2)
}

// -- the Airy functions

fn negative_re(z: &Basic) -> PolyResult<bool> {
    Ok(is_true(
        &crate::functions::elementary::complexes::re(z)?,
        Predicate::negative,
    ))
}

fn positive_re(z: &Basic) -> PolyResult<bool> {
    Ok(is_true(
        &crate::functions::elementary::complexes::re(z)?,
        Predicate::positive,
    ))
}

fn gamma(x: &Basic) -> R {
    super::gamma_functions::gamma(x)
}

fn hyper1(b: &Basic, z: &Basic) -> R {
    super::hyper::hyper(&[], std::slice::from_ref(b), z)
}

fn cube_over_9(z: &Basic) -> R {
    div(&pow(z, &int(3))?, &int(9))
}

fn airyai_besselj(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let a: Basic = pow(&neg(z)?, &q(3, 2))?;
    if negative_re(z)? {
        let ta: Basic = mul(&[two_thirds(), a])?;
        let s: Basic = add(&[
            b(Function::BesselJ, &neg(&third())?, &ta)?,
            b(Function::BesselJ, &third(), &ta)?,
        ])?;
        return Ok(Some(mul(&[third(), sqrt(&neg(z)?)?, s])?));
    }
    Ok(None)
}

fn airyai_besseli(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let a: Basic = pow(z, &q(3, 2))?;
    let ta: Basic = mul(&[two_thirds(), a.clone()])?;
    let (m, p) = (
        b(Function::BesselI, &neg(&third())?, &ta)?,
        b(Function::BesselI, &third(), &ta)?,
    );
    if positive_re(z)? {
        return Ok(Some(mul(&[third(), sqrt(z)?, sub(&m, &p)?])?));
    }
    let inner: Basic = sub(
        &mul(&[pow(&a, &third())?, m])?,
        &mul(&[z.clone(), pow(&a, &neg(&third())?)?, p])?,
    )?;
    Ok(Some(mul(&[third(), inner])?))
}

fn airyai_hyper(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let pf1: Basic = div(&int(1), &mul(&[pow(&int(3), &two_thirds())?, gamma(&two_thirds())?])?)?;
    let pf2: Basic = div(z, &mul(&[pow(&int(3), &third())?, gamma(&third())?])?)?;
    let w: Basic = cube_over_9(z)?;
    Ok(Some(sub(
        &mul(&[pf1, hyper1(&two_thirds(), &w)?])?,
        &mul(&[pf2, hyper1(&q(4, 3), &w)?])?,
    )?))
}

fn airybi_besselj(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let a: Basic = pow(&neg(z)?, &q(3, 2))?;
    if negative_re(z)? {
        let ta: Basic = mul(&[two_thirds(), a])?;
        let s: Basic = sub(
            &b(Function::BesselJ, &neg(&third())?, &ta)?,
            &b(Function::BesselJ, &third(), &ta)?,
        )?;
        return Ok(Some(mul(&[sqrt(&div(&neg(z)?, &int(3))?)?, s])?));
    }
    Ok(None)
}

fn airybi_besseli(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let a: Basic = pow(z, &q(3, 2))?;
    let ta: Basic = mul(&[two_thirds(), a.clone()])?;
    let (m, p) = (
        b(Function::BesselI, &neg(&third())?, &ta)?,
        b(Function::BesselI, &third(), &ta)?,
    );
    if positive_re(z)? {
        return Ok(Some(mul(&[div(&sqrt(z)?, &sqrt(&int(3))?)?, add(&[m, p])?])?));
    }
    let (bb, c) = (pow(&a, &third())?, pow(&a, &neg(&third())?)?);
    Ok(Some(mul(&[
        sqrt(&third())?,
        add(&[mul(&[bb, m])?, mul(&[z.clone(), c, p])?])?,
    ])?))
}

fn airybi_hyper(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let r6: Basic = pow(&int(3), &q(1, 6))?;
    let pf1: Basic = div(&int(1), &mul(&[r6.clone(), gamma(&two_thirds())?])?)?;
    let pf2: Basic = div(&mul(&[z.clone(), r6])?, &gamma(&third())?)?;
    let w: Basic = cube_over_9(z)?;
    Ok(Some(add(&[
        mul(&[pf1, hyper1(&two_thirds(), &w)?])?,
        mul(&[pf2, hyper1(&q(4, 3), &w)?])?,
    ])?))
}

fn airyaiprime_besselj(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let a: Basic = pow(&neg(z)?, &q(3, 2))?;
    if negative_re(z)? {
        let ta: Basic = mul(&[two_thirds(), a])?;
        let s: Basic = sub(
            &b(Function::BesselJ, &neg(&two_thirds())?, &ta)?,
            &b(Function::BesselJ, &two_thirds(), &ta)?,
        )?;
        return Ok(Some(mul(&[div(z, &int(3))?, s])?));
    }
    Ok(None)
}

fn airyaiprime_besseli(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    if positive_re(z)? {
        let a: Basic = mul(&[two_thirds(), pow(z, &q(3, 2))?])?;
        let s: Basic = sub(
            &b(Function::BesselI, &two_thirds(), &a)?,
            &b(Function::BesselI, &neg(&two_thirds())?, &a)?,
        )?;
        return Ok(Some(mul(&[div(z, &int(3))?, s])?));
    }
    let p: Basic = pow(z, &q(3, 2))?;
    let (bb, c) = (pow(&p, &two_thirds())?, pow(&p, &neg(&two_thirds())?)?);
    let tp: Basic = mul(&[two_thirds(), p])?;
    let inner: Basic = sub(
        &mul(&[pow(z, &int(2))?, c, b(Function::BesselI, &two_thirds(), &tp)?])?,
        &mul(&[bb, b(Function::BesselI, &neg(&two_thirds())?, &tp)?])?,
    )?;
    Ok(Some(mul(&[third(), inner])?))
}

fn airyaiprime_hyper(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let pf1: Basic = div(
        &pow(z, &int(2))?,
        &mul(&[int(2), pow(&int(3), &two_thirds())?, gamma(&two_thirds())?])?,
    )?;
    let pf2: Basic = div(&int(1), &mul(&[pow(&int(3), &third())?, gamma(&third())?])?)?;
    let w: Basic = cube_over_9(z)?;
    Ok(Some(sub(
        &mul(&[pf1, hyper1(&q(5, 3), &w)?])?,
        &mul(&[pf2, hyper1(&third(), &w)?])?,
    )?))
}

fn airybiprime_besselj(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let a: Basic = mul(&[two_thirds(), pow(&neg(z)?, &q(3, 2))?])?;
    if negative_re(z)? {
        let s: Basic = add(&[
            b(Function::BesselJ, &neg(&two_thirds())?, &a)?,
            b(Function::BesselJ, &two_thirds(), &a)?,
        ])?;
        return Ok(Some(mul(&[div(&neg(z)?, &sqrt(&int(3))?)?, s])?));
    }
    Ok(None)
}

fn airybiprime_besseli(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    if positive_re(z)? {
        let a: Basic = mul(&[two_thirds(), pow(z, &q(3, 2))?])?;
        let s: Basic = add(&[
            b(Function::BesselI, &neg(&two_thirds())?, &a)?,
            b(Function::BesselI, &two_thirds(), &a)?,
        ])?;
        return Ok(Some(mul(&[div(z, &sqrt(&int(3))?)?, s])?));
    }
    let p: Basic = pow(z, &q(3, 2))?;
    let (bb, c) = (pow(&p, &two_thirds())?, pow(&p, &neg(&two_thirds())?)?);
    let tp: Basic = mul(&[two_thirds(), p])?;
    let inner: Basic = add(&[
        mul(&[bb, b(Function::BesselI, &neg(&two_thirds())?, &tp)?])?,
        mul(&[pow(z, &int(2))?, c, b(Function::BesselI, &two_thirds(), &tp)?])?,
    ])?;
    Ok(Some(mul(&[sqrt(&third())?, inner])?))
}

fn airybiprime_hyper(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    let r6: Basic = pow(&int(3), &q(1, 6))?;
    let pf1: Basic = div(&pow(z, &int(2))?, &mul(&[int(2), r6.clone(), gamma(&two_thirds())?])?)?;
    let pf2: Basic = div(&r6, &gamma(&third())?)?;
    let w: Basic = cube_over_9(z)?;
    Ok(Some(add(&[
        mul(&[pf1, hyper1(&q(5, 3), &w)?])?,
        mul(&[pf2, hyper1(&third(), &w)?])?,
    ])?))
}

// -- the spherical harmonics

fn ynm_polynomial(e: &Basic, _args: &[Basic], _h: &RewriteHints) -> O {
    let expanded: Basic = crate::functions::expand_func::ynm_rule(e)?.unwrap_or_else(|| e.clone());
    let mut hints = crate::core::function::Hints::DEFAULT;
    hints.func = true;
    Ok(Some(crate::core::function::expand_with(&expanded, true, hints)?))
}

fn ynm_trig(_e: &Basic, _args: &[Basic], _h: &RewriteHints) -> O {
    Err(not_ported("Ynm rewritten as cos or sin (simplify)"))
}

// ---------------------------------------------------------------------
// the error functions in terms of erf and the integrals in terms of li
// (PORTED FROM SYMPY: erfc, erfi, erf2, fresnels, fresnelc ._eval_rewrite_as_erf,
// Li._eval_rewrite_as_li, Ei._eval_rewrite_as_li)

fn call(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    crate::functions::apply_function(f, args)
}
fn imag() -> Basic {
    Basic::atom(crate::core::basic::Head::ImaginaryUnit)
}
fn mul2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}

fn erfc_erf(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(crate::core::add::sub(
        &Basic::one(),
        &call(Function::Erf, &args[..1])?,
    )?))
}

fn erfi_erf(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let inner: Basic = call(Function::Erf, &[mul2(&imag(), &args[0])?])?;
    Ok(Some(mul2(&crate::core::mul::neg(&imag())?, &inner)?))
}

fn erf2_erf(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(crate::core::add::sub(
        &call(Function::Erf, &args[1..2])?,
        &call(Function::Erf, &args[..1])?,
    )?))
}

/// `(1 + I)/2*sqrt(pi)` and `(1 - I)/2*sqrt(pi)`.
fn fresnel_scales() -> PolyResult<(Basic, Basic)> {
    let sqrt_pi: Basic = crate::core::power::pow(&Basic::pi(), &Basic::half())?;
    let half = |s: Basic| -> PolyResult<Basic> { mul2(&crate::core::mul::div(&s, &Basic::int(2))?, &sqrt_pi) };
    Ok((
        half(crate::core::add::add(&[Basic::one(), imag()])?)?,
        half(crate::core::add::sub(&Basic::one(), &imag())?)?,
    ))
}

fn fresnel_erf(args: &[Basic], s: bool) -> O {
    let (plus, minus) = fresnel_scales()?;
    let first: Basic = call(Function::Erf, &[mul2(&plus, &args[0])?])?;
    let second: Basic = mul2(&imag(), &call(Function::Erf, &[mul2(&minus, &args[0])?])?)?;
    let (factor, inner) = if s {
        (
            crate::core::add::add(&[Basic::one(), imag()])?,
            crate::core::add::sub(&first, &second)?,
        )
    } else {
        (
            crate::core::add::sub(&Basic::one(), &imag())?,
            crate::core::add::add(&[first, second])?,
        )
    };
    Ok(Some(mul2(&crate::core::mul::div(&factor, &Basic::int(4))?, &inner)?))
}

fn fresnels_erf(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    fresnel_erf(args, true)
}

fn fresnelc_erf(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    fresnel_erf(args, false)
}

fn offset_li_li(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    Ok(Some(crate::core::add::sub(
        &call(Function::LogIntegral, &args[..1])?,
        &call(Function::LogIntegral, &[Basic::int(2)])?,
    )?))
}

fn ei_li(_e: &Basic, args: &[Basic], _h: &RewriteHints) -> O {
    let z: &Basic = &args[0];
    if z.is_function(Function::Log) {
        return Ok(Some(call(Function::LogIntegral, &z.args()[..1])?));
    }
    if crate::core::expr::is(z, crate::core::assumptions::Predicate::extended_real) != Some(true) {
        // li(exp(z)) is Ei(z) for -pi < im(z) <= pi only
        return Ok(None);
    }
    Ok(Some(call(
        Function::LogIntegral,
        &[crate::functions::elementary::exponential::exp(z)?],
    )?))
}

/// The targets of these rules.
pub const TARGETS: [&str; 10] = [
    "besselj", "bessely", "besseli", "jn", "yn", "hankel1", "hankel2", "hyper", "erf", "li",
];

/// The rule of the head `f` for `target` (`rules`).
pub fn rule_for(f: Function, target: &str) -> Option<Rule> {
    use Function as F;
    let rule: Rule = match (f, target) {
        (F::Erfc, "erf") => erfc_erf,
        (F::Erfi, "erf") => erfi_erf,
        (F::Erf2, "erf") => erf2_erf,
        (F::FresnelS, "erf") => fresnels_erf,
        (F::FresnelC, "erf") => fresnelc_erf,
        (F::OffsetLogIntegral, "li") => offset_li_li,
        (F::Ei, "li") => ei_li,
        (F::BesselJ, "bessely") => besselj_bessely,
        (F::BesselJ, "jn") => besselj_jn,
        (F::BesselJ, "besseli") | (F::BesselI, "besselj") => polar,
        (F::BesselY, "besselj") => bessely_besselj,
        (F::BesselK, "besseli") => besselk_besseli,
        (F::BesselK, "besselj") => besselk_besselj,
        (F::BesselY, "besseli") => bessely_besseli,
        (F::BesselY, "yn") => bessely_yn,
        (F::BesselI, "bessely") | (F::BesselI, "jn") => polar,
        (F::BesselK, "bessely") => besselk_bessely,
        (F::BesselK, "yn") => besselk_yn,
        (F::SphericalJ, "besselj") => jn_besselj,
        (F::SphericalJ, "bessely") => jn_bessely,
        (F::SphericalJ, "yn") => jn_yn,
        (F::SphericalY, "besselj") => yn_besselj,
        (F::SphericalY, "bessely") => yn_bessely,
        (F::SphericalY, "jn") => yn_jn,
        (F::AiryAi, "besselj") => airyai_besselj,
        (F::AiryAi, "besseli") => airyai_besseli,
        (F::AiryAi, "hyper") => airyai_hyper,
        (F::AiryBi, "besselj") => airybi_besselj,
        (F::AiryBi, "besseli") => airybi_besseli,
        (F::AiryBi, "hyper") => airybi_hyper,
        (F::AiryAiPrime, "besselj") => airyaiprime_besselj,
        (F::AiryAiPrime, "besseli") => airyaiprime_besseli,
        (F::AiryAiPrime, "hyper") => airyaiprime_hyper,
        (F::AiryBiPrime, "besselj") => airybiprime_besselj,
        (F::AiryBiPrime, "besseli") => airybiprime_besseli,
        (F::AiryBiPrime, "hyper") => airybiprime_hyper,
        (F::Ynm, "polynomial") => ynm_polynomial,
        (F::Ynm, "cos") | (F::Ynm, "sin") => ynm_trig,
        (F::SphericalHankel1, "besselj") => hn1_besselj,
        (F::SphericalHankel1, "bessely") => hn1_bessely,
        (F::SphericalHankel1, "yn") => hn1_yn,
        (F::SphericalHankel1, "jn") => hn1_jn,
        (F::SphericalHankel1, "hankel1") => hn1_hankel1,
        (F::SphericalHankel2, "besselj") => hn2_besselj,
        (F::SphericalHankel2, "bessely") => hn2_bessely,
        (F::SphericalHankel2, "yn") => hn2_yn,
        (F::SphericalHankel2, "jn") => hn2_jn,
        (F::SphericalHankel2, "hankel2") => hn2_hankel2,
        _ => return None,
    };
    Some(rule)
}
