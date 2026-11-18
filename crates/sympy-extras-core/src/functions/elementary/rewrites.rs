//! The `rewrite` rules of the elementary functions, ported from
//! `sympy_extras.functions.elementary._rewrites` (and the rules of
//! `atan2` and `Ynm` of the other tables) for the targets `exp`, `cos`,
//! `sin`, `log`, `tan`, `cot`, `sec`, `csc`, `sincos`, `sinh`, `cosh`,
//! `tanh`, `coth`, `csch` and `sech`: the trigonometric and hyperbolic
//! functions in terms of exponentials and of each other, the exponentials
//! in terms of `cos`, `sin` and `tanh`, the inverse functions as
//! logarithms. Python's operators are
//! kept as binary operations (`a*b/c` is `(a*b)/c`).

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py, hyperbolic.py, exponential.py
// and sympy/core/power.py (the _eval_rewrite_as_exp, _eval_rewrite_as_cos, _eval_rewrite_as_sin
// and _eval_rewrite_as_log methods), sympy/core/numbers.py (Exp1) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::function::rebuild_evaluated;
use crate::core::mul::{div, mul, neg};
use crate::core::power::pow;
use crate::core::rewriting::{rewrite, RewriteHints, Rule};
use crate::error::{not_ported, PolyResult};
use crate::functions::apply_function;
use crate::functions::elementary::exponential::{exp, log};
use crate::functions::elementary::trigonometric::{cos, sin};

fn i() -> Basic {
    Basic::imaginary_unit()
}

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn half_pi() -> PolyResult<Basic> {
    mul(&[Basic::pi(), Basic::half()])
}

fn inv(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::negative_one())
}

fn m2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), b.clone()])
}

fn a2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), b.clone()])
}

fn sqrt(x: &Basic) -> PolyResult<Basic> {
    pow(x, &Basic::half())
}

/// The unevaluated node `f(x)` (Python's class constructor).
fn raw(f: Function, x: Basic) -> Basic {
    Basic::new(Head::Function(f), vec![x])
}

fn call(f: Function, x: &Basic) -> PolyResult<Basic> {
    apply_function(f, std::slice::from_ref(x))
}

fn is_fn(e: &Basic, f: Function) -> bool {
    matches!(e.head(), Head::Function(g) if *g == f)
}

fn has_fn(e: &Basic, f: Function) -> bool {
    is_fn(e, f) || e.args().iter().any(|a| has_fn(a, f))
}

fn has_symbol(e: &Basic) -> bool {
    matches!(e.head(), Head::Symbol(_) | Head::Dummy(..) | Head::Wild(_)) || e.args().iter().any(has_symbol)
}

fn is_e(e: &Basic) -> bool {
    matches!(e.head(), Head::Exp1)
}

fn trig_or_hyper(e: &Basic) -> bool {
    use Function as F;
    matches!(
        e.head(),
        Head::Function(
            F::Sin
                | F::Cos
                | F::Tan
                | F::Cot
                | F::Sec
                | F::Csc
                | F::Sinh
                | F::Cosh
                | F::Tanh
                | F::Coth
                | F::Sech
                | F::Csch
        )
    )
}

/// SymPy's re-rewrite of a trigonometric or hyperbolic argument.
fn exp_arg(a: &Basic) -> PolyResult<Basic> {
    if trig_or_hyper(a) {
        let rebuilt: Basic = rebuild_evaluated(a, a.args().to_vec())?;
        return rewrite(&rebuilt, &[], "exp", &RewriteHints::default());
    }
    Ok(a.clone())
}

type Out = PolyResult<Option<Basic>>;

// trigonometric, as exp

fn sin_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: Basic = exp_arg(&args[0])?;
    let top: Basic = add(&[exp(&m2(&a, &i())?)?, neg(&exp(&m2(&neg(&a)?, &i())?)?)?])?;
    Ok(Some(mul(&[top, inv(&m2(&int(2), &i())?)?])?))
}

fn cos_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: Basic = exp_arg(&args[0])?;
    let top: Basic = add(&[exp(&m2(&a, &i())?)?, exp(&m2(&neg(&a)?, &i())?)?])?;
    Ok(Some(mul(&[top, Basic::half()])?))
}

fn tan_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: Basic = exp_arg(&args[0])?;
    let (neg_exp, pos_exp) = (exp(&m2(&neg(&a)?, &i())?)?, exp(&m2(&a, &i())?)?);
    Ok(Some(mul(&[
        i(),
        add(&[neg_exp.clone(), neg(&pos_exp)?])?,
        inv(&add(&[neg_exp, pos_exp])?)?,
    ])?))
}

fn cot_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: Basic = exp_arg(&args[0])?;
    let (neg_exp, pos_exp) = (exp(&m2(&neg(&a)?, &i())?)?, exp(&m2(&a, &i())?)?);
    Ok(Some(mul(&[
        i(),
        add(&[pos_exp.clone(), neg_exp.clone()])?,
        inv(&add(&[pos_exp, neg(&neg_exp)?])?)?,
    ])?))
}

/// SymPy's `_rewrite_reciprocal`: `1/t` with `t` the rule of the reciprocal
/// function, unless it leaves that function unchanged.
fn reciprocal(rule: Rule, of: Function, node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    if let Some(r) = rule(node, args, hints)? {
        if r != call(of, &args[0])? {
            return Ok(Some(inv(&r)?));
        }
    }
    Ok(None)
}

fn sec_exp(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal(cos_exp, Function::Cos, node, args, hints)
}

fn csc_exp(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal(sin_exp, Function::Sin, node, args, hints)
}

fn sech_exp(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal(cosh_exp, Function::Cosh, node, args, hints)
}

fn csch_exp(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal(sinh_exp, Function::Sinh, node, args, hints)
}

// trigonometric, as cos

fn shifted_cos(x: &Basic) -> PolyResult<Basic> {
    Ok(raw(Function::Cos, add(&[x.clone(), neg(&half_pi()?)?])?))
}

fn sin_cos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(shifted_cos(&args[0])?))
}

fn tan_cos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(mul(&[shifted_cos(&args[0])?, inv(&cos(&args[0])?)?])?))
}

fn cot_cos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(mul(&[cos(&args[0])?, inv(&shifted_cos(&args[0])?)?])?))
}

fn sec_cos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(inv(&cos(&args[0])?)?))
}

fn csc_cos(_: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    let r: Basic = rewrite(&sin(&args[0])?, &[], "cos", hints)?;
    Ok(Some(inv(&r)?))
}

// hyperbolic

fn sinh_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let top: Basic = add(&[exp(&args[0])?, neg(&exp(&neg(&args[0])?)?)?])?;
    Ok(Some(mul(&[top, Basic::half()])?))
}

fn cosh_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let top: Basic = add(&[exp(&args[0])?, exp(&neg(&args[0])?)?])?;
    Ok(Some(mul(&[top, Basic::half()])?))
}

fn tanh_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let (neg_exp, pos_exp) = (exp(&neg(&args[0])?)?, exp(&args[0])?);
    Ok(Some(mul(&[
        add(&[pos_exp.clone(), neg(&neg_exp)?])?,
        inv(&add(&[pos_exp, neg_exp])?)?,
    ])?))
}

fn coth_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let (neg_exp, pos_exp) = (exp(&neg(&args[0])?)?, exp(&args[0])?);
    Ok(Some(mul(&[
        add(&[pos_exp.clone(), neg_exp.clone()])?,
        inv(&add(&[pos_exp, neg(&neg_exp)?])?)?,
    ])?))
}

fn cosh_cos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(raw(Function::Cos, m2(&i(), &args[0])?)))
}

fn sech_cos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(inv(&raw(Function::Cos, m2(&i(), &args[0])?))?))
}

// exp and powers

fn exp_cos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: Basic = m2(&i(), &args[0])?;
    Ok(Some(add(&[cos(&a)?, m2(&i(), &cos(&a2(&a, &half_pi()?)?)?)?])?))
}

fn pow_exp(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let (base, expo) = (&args[0], &args[1]);
    if is(base, Predicate::zero) == Some(true)
        || has_fn(base, Function::Exp)
        || has_fn(expo, Function::Exp)
        || is_e(base)
    {
        return Ok(Some(pow(base, expo)?));
    }
    if has_symbol(base) {
        // unevaluated without symbols in the exponent: exp(x*log(5)) is x**5
        let e: Basic = m2(&log(base)?, expo)?;
        return Ok(Some(if has_symbol(expo) {
            exp(&e)?
        } else {
            raw(Function::Exp, e)
        }));
    }
    let polar: Basic = add(&[
        log(&call(Function::Abs, base)?)?,
        m2(&i(), &call(Function::Arg, base)?)?,
    ])?;
    Ok(Some(exp(&m2(&polar, expo)?)?))
}

fn pow_cos(node: &Basic, _: &[Basic], hints: &RewriteHints) -> Out {
    if node.is_pow() && is_e(&node.args()[0]) {
        return exp_cos(node, &node.args()[1..2], hints);
    }
    Ok(None)
}

// in terms of sin

fn cos_sin(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(raw(Function::Sin, add(&[args[0].clone(), half_pi()?])?)))
}

fn tan_sin(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    Ok(Some(mul(&[
        int(2),
        pow(&sin(x)?, &int(2))?,
        inv(&sin(&m2(&int(2), x)?)?)?,
    ])?))
}

fn cot_sin(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    Ok(Some(mul(&[
        sin(&m2(&int(2), x)?)?,
        inv(&m2(&int(2), &pow(&sin(x)?, &int(2))?)?)?,
    ])?))
}

fn sec_sin(_: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    let r: Basic = rewrite(&cos(&args[0])?, &[], "sin", hints)?;
    Ok(Some(inv(&r)?))
}

fn csc_sin(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(inv(&sin(&args[0])?)?))
}

fn exp_sin(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: Basic = m2(&i(), &args[0])?;
    Ok(Some(add(&[sin(&a2(&a, &half_pi()?)?)?, neg(&m2(&i(), &sin(&a)?)?)?])?))
}

fn pow_sin(node: &Basic, _: &[Basic], hints: &RewriteHints) -> Out {
    if node.is_pow() && is_e(&node.args()[0]) {
        return exp_sin(node, &node.args()[1..2], hints);
    }
    Ok(None)
}

fn sinh_sin(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(neg(&m2(&i(), &sin(&m2(&i(), &args[0])?)?)?)?))
}

fn csch_sin(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(m2(&i(), &inv(&raw(Function::Sin, m2(&i(), &args[0])?))?)?))
}

fn e_sin(_: &Basic, _: &[Basic], _: &RewriteHints) -> Out {
    let s1: Basic = sin(&a2(&i(), &half_pi()?)?)?;
    Ok(Some(sub(&s1, &m2(&i(), &sin(&i())?)?)?))
}

fn e_cos(_: &Basic, _: &[Basic], _: &RewriteHints) -> Out {
    let c2: Basic = cos(&a2(&i(), &half_pi()?)?)?;
    Ok(Some(a2(&cos(&i())?, &m2(&i(), &c2)?)?))
}

// ---------------------------------------------------------------------
// the trigonometric functions between themselves (tan, cot, sec, csc,
// sincos)

fn f1(f: Function, x: &Basic) -> PolyResult<Basic> {
    call(f, x)
}

fn rw(e: &Basic, target: &str, hints: &RewriteHints) -> PolyResult<Basic> {
    rewrite(e, &[], target, hints)
}

fn one_over(x: &Basic) -> PolyResult<Basic> {
    div(&one(), x)
}

fn half_of(a: &Basic) -> PolyResult<Basic> {
    m2(&Basic::half(), a)
}

fn sin_tan(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let th: Basic = f1(Function::Tan, &half_of(&args[0])?)?;
    Ok(Some(div(&m2(&int(2), &th)?, &a2(&one(), &pow(&th, &int(2))?)?)?))
}

fn sin_sincos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(&m2(&sin(a)?, &cos(a)?)?, &cos(a)?)?))
}

fn not_multiple(a: &Basic, period: &Basic) -> PolyResult<Basic> {
    use crate::core::relational::eq;
    let im_zero: Basic = eq(&f1(Function::Im, a)?, &Basic::zero())?;
    let mod_zero: Basic = eq(&crate::core::modulo::modulo(a, period)?, &Basic::zero())?;
    Ok(crate::core::logic::and_(&[im_zero, mod_zero]))
}

fn sin_cot(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    use crate::core::logic::true_;
    let a: &Basic = &args[0];
    let ch: Basic = f1(Function::Cot, &half_of(a)?)?;
    let value: Basic = div(&m2(&int(2), &ch)?, &a2(&one(), &pow(&ch, &int(2))?)?)?;
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&[
        (Basic::zero(), not_multiple(a, &Basic::pi())?),
        (value, true_()),
    ])?))
}

fn sin_csc(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Csc, &args[0])?)?))
}

fn sin_sec(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&raw(Function::Sec, sub(&args[0], &half_pi()?)?))?))
}

fn cos_tan(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let th: Basic = pow(&f1(Function::Tan, &half_of(&args[0])?)?, &int(2))?;
    Ok(Some(div(&sub(&one(), &th)?, &a2(&one(), &th)?)?))
}

fn cos_sincos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(&m2(&sin(a)?, &cos(a)?)?, &sin(a)?)?))
}

fn cos_cot(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    use crate::core::logic::true_;
    let a: &Basic = &args[0];
    let ch: Basic = pow(&f1(Function::Cot, &half_of(a)?)?, &int(2))?;
    let value: Basic = div(&sub(&ch, &one())?, &a2(&ch, &one())?)?;
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&[
        (one(), not_multiple(a, &m2(&int(2), &Basic::pi())?)?),
        (value, true_()),
    ])?))
}

fn cos_sec(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Sec, &args[0])?)?))
}

fn cos_csc(_: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    Ok(Some(one_over(&rw(&f1(Function::Sec, &args[0])?, "csc", hints)?)?))
}

fn tan_sincos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(div(&sin(&args[0])?, &cos(&args[0])?)?))
}

fn tan_cot(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Cot, &args[0])?)?))
}

/// `tan._eval_rewrite_as_sec`, `_eval_rewrite_as_csc` and those of `cot`.
fn sin_and_cos_in(target: &str, node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    let (s, c) = (rw(&sin(a)?, target, hints)?, rw(&cos(a)?, target, hints)?);
    Ok(Some(if is_fn(node, Function::Tan) {
        div(&s, &c)?
    } else {
        div(&c, &s)?
    }))
}

fn sin_and_cos_in_sec(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    sin_and_cos_in("sec", node, args, hints)
}

fn sin_and_cos_in_csc(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    sin_and_cos_in("csc", node, args, hints)
}

fn cot_sincos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(div(&cos(&args[0])?, &sin(&args[0])?)?))
}

fn cot_tan(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Tan, &args[0])?)?))
}

fn sec_cot(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let ch: Basic = pow(&f1(Function::Cot, &div(&args[0], &int(2))?)?, &int(2))?;
    Ok(Some(div(&a2(&ch, &one())?, &sub(&ch, &one())?)?))
}

fn sec_sincos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(&sin(a)?, &m2(&cos(a)?, &sin(a)?)?)?))
}

fn sec_tan(_: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    Ok(Some(one_over(&rw(&cos(&args[0])?, "tan", hints)?)?))
}

fn sec_csc(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(raw(Function::Csc, sub(&half_pi()?, &args[0])?)))
}

fn csc_sincos(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(&cos(a)?, &m2(&sin(a)?, &cos(a)?)?)?))
}

fn csc_cot(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let ch: Basic = f1(Function::Cot, &div(&args[0], &int(2))?)?;
    Ok(Some(div(&a2(&one(), &pow(&ch, &int(2))?)?, &m2(&int(2), &ch)?)?))
}

fn csc_sec(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(raw(Function::Sec, sub(&half_pi()?, &args[0])?)))
}

fn csc_tan(_: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    Ok(Some(one_over(&rw(&sin(&args[0])?, "tan", hints)?)?))
}

// ---------------------------------------------------------------------
// the hyperbolic functions between themselves and into csc, sec, tan, cot

fn ipi2() -> PolyResult<Basic> {
    div(&m2(&i(), &Basic::pi())?, &int(2))
}

fn minus_i() -> PolyResult<Basic> {
    neg(&i())
}

fn sinh_cosh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(m2(&minus_i()?, &f1(Function::Cosh, &a2(&args[0], &ipi2()?)?)?)?))
}

fn sinh_tanh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let th: Basic = f1(Function::Tanh, &half_of(&args[0])?)?;
    Ok(Some(div(&m2(&int(2), &th)?, &sub(&one(), &pow(&th, &int(2))?)?)?))
}

fn sinh_coth(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let ch: Basic = f1(Function::Coth, &half_of(&args[0])?)?;
    Ok(Some(div(&m2(&int(2), &ch)?, &sub(&pow(&ch, &int(2))?, &one())?)?))
}

fn sinh_csch(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Csch, &args[0])?)?))
}

fn sinh_csc(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(div(&minus_i()?, &f1(Function::Csc, &m2(&i(), &args[0])?)?)?))
}

fn cosh_sec(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&raw(Function::Sec, m2(&i(), &args[0])?))?))
}

fn cosh_sinh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(m2(&minus_i()?, &raw(Function::Sinh, a2(&args[0], &ipi2()?)?))?))
}

fn cosh_tanh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let th: Basic = pow(&f1(Function::Tanh, &half_of(&args[0])?)?, &int(2))?;
    Ok(Some(div(&a2(&one(), &th)?, &sub(&one(), &th)?)?))
}

fn cosh_coth(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let ch: Basic = pow(&f1(Function::Coth, &half_of(&args[0])?)?, &int(2))?;
    Ok(Some(div(&a2(&ch, &one())?, &sub(&ch, &one())?)?))
}

fn cosh_sech(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Sech, &args[0])?)?))
}

fn tanh_tan(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(m2(&minus_i()?, &raw(Function::Tan, m2(&i(), &args[0])?))?))
}

fn tanh_cot(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(div(&minus_i()?, &raw(Function::Cot, m2(&i(), &args[0])?))?))
}

fn tanh_sinh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(
        &m2(&i(), &f1(Function::Sinh, a)?)?,
        &raw(Function::Sinh, sub(&ipi2()?, a)?),
    )?))
}

fn tanh_cosh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(
        &m2(&i(), &raw(Function::Cosh, sub(&ipi2()?, a)?))?,
        &f1(Function::Cosh, a)?,
    )?))
}

fn tanh_coth(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Coth, &args[0])?)?))
}

fn coth_sinh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(
        &m2(&minus_i()?, &raw(Function::Sinh, sub(&ipi2()?, a)?))?,
        &f1(Function::Sinh, a)?,
    )?))
}

fn coth_cosh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    Ok(Some(div(
        &m2(&minus_i()?, &f1(Function::Cosh, a)?)?,
        &raw(Function::Cosh, sub(&ipi2()?, a)?),
    )?))
}

fn coth_tanh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Tanh, &args[0])?)?))
}

fn csch_csc(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(m2(&i(), &raw(Function::Csc, m2(&i(), &args[0])?))?))
}

fn csch_cosh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(div(&i(), &raw(Function::Cosh, a2(&args[0], &ipi2()?)?))?))
}

fn csch_sinh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Sinh, &args[0])?)?))
}

fn sech_sec(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(raw(Function::Sec, m2(&i(), &args[0])?)))
}

fn sech_sinh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(div(&i(), &raw(Function::Sinh, a2(&args[0], &ipi2()?)?))?))
}

fn sech_cosh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(one_over(&f1(Function::Cosh, &args[0])?)?))
}

// PORTED FROM SYMPY: ReciprocalHyperbolicFunction._rewrite_reciprocal
/// The rule of the reciprocal function called on `of(self.args[0])`
/// (`None` when that node is not of the reciprocal class).
fn reciprocal_call(rule: Rule, of: Function, node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    let o: Basic = call(of, &node.args()[0])?;
    if !is_fn(&o, of) {
        return Ok(None);
    }
    if let Some(r) = rule(&o, args, hints)? {
        if r != call(of, &args[0])? {
            return Ok(Some(inv(&r)?));
        }
    }
    Ok(None)
}

fn csch_tanh(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(sinh_tanh, Function::Sinh, node, args, hints)
}

fn csch_coth(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(sinh_coth, Function::Sinh, node, args, hints)
}

fn sech_tanh(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(cosh_tanh, Function::Cosh, node, args, hints)
}

fn sech_coth(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(cosh_coth, Function::Cosh, node, args, hints)
}

fn exp_tanh_of(a: &Basic) -> PolyResult<Basic> {
    let t: Basic = f1(Function::Tanh, &div(a, &int(2))?)?;
    div(&a2(&one(), &t)?, &sub(&one(), &t)?)
}

fn exp_tanh(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(exp_tanh_of(&args[0])?))
}

fn pow_tanh(node: &Basic, _: &[Basic], _: &RewriteHints) -> Out {
    if node.is_pow() && is_e(&node.args()[0]) {
        return Ok(Some(exp_tanh_of(&node.args()[1])?));
    }
    Ok(None)
}

/// The rules of the trigonometric and hyperbolic targets.
fn trig_rule_for(e: &Basic, target: &str) -> Option<Rule> {
    use Function as F;
    if let ("tanh", Head::Pow) = (target, e.head()) {
        return Some(pow_tanh);
    }
    let Head::Function(f) = e.head() else {
        return None;
    };
    let rule: Rule = match (target, *f) {
        ("tan", F::Sin) => sin_tan,
        ("sincos", F::Sin) => sin_sincos,
        ("cot", F::Sin) => sin_cot,
        ("csc", F::Sin) => sin_csc,
        ("sec", F::Sin) => sin_sec,
        ("tan", F::Cos) => cos_tan,
        ("sincos", F::Cos) => cos_sincos,
        ("cot", F::Cos) => cos_cot,
        ("sec", F::Cos) => cos_sec,
        ("csc", F::Cos) => cos_csc,
        ("sincos", F::Tan) => tan_sincos,
        ("cot", F::Tan) => tan_cot,
        ("sec", F::Tan) => sin_and_cos_in_sec,
        ("csc", F::Tan) => sin_and_cos_in_csc,
        ("sincos", F::Cot) => cot_sincos,
        ("tan", F::Cot) => cot_tan,
        ("sec", F::Cot) => sin_and_cos_in_sec,
        ("csc", F::Cot) => sin_and_cos_in_csc,
        ("cot", F::Sec) => sec_cot,
        ("sincos", F::Sec) => sec_sincos,
        ("tan", F::Sec) => sec_tan,
        ("csc", F::Sec) => sec_csc,
        ("sincos", F::Csc) => csc_sincos,
        ("cot", F::Csc) => csc_cot,
        ("sec", F::Csc) => csc_sec,
        ("tan", F::Csc) => csc_tan,
        ("cosh", F::Sinh) => sinh_cosh,
        ("tanh", F::Sinh) => sinh_tanh,
        ("coth", F::Sinh) => sinh_coth,
        ("csch", F::Sinh) => sinh_csch,
        ("csc", F::Sinh) => sinh_csc,
        ("sec", F::Cosh) => cosh_sec,
        ("sinh", F::Cosh) => cosh_sinh,
        ("tanh", F::Cosh) => cosh_tanh,
        ("coth", F::Cosh) => cosh_coth,
        ("sech", F::Cosh) => cosh_sech,
        ("tan", F::Tanh) => tanh_tan,
        ("cot", F::Tanh) => tanh_cot,
        ("sinh", F::Tanh) => tanh_sinh,
        ("cosh", F::Tanh) => tanh_cosh,
        ("coth", F::Tanh) => tanh_coth,
        ("sinh", F::Coth) => coth_sinh,
        ("cosh", F::Coth) => coth_cosh,
        ("tanh", F::Coth) => coth_tanh,
        ("tanh", F::Csch) => csch_tanh,
        ("coth", F::Csch) => csch_coth,
        ("tanh", F::Sech) => sech_tanh,
        ("coth", F::Sech) => sech_coth,
        ("csc", F::Csch) => csch_csc,
        ("cosh", F::Csch) => csch_cosh,
        ("sinh", F::Csch) => csch_sinh,
        ("sec", F::Sech) => sech_sec,
        ("sinh", F::Sech) => sech_sinh,
        ("cosh", F::Sech) => sech_cosh,
        ("tanh", F::Exp) => exp_tanh,
        _ => return None,
    };
    Some(rule)
}

// the inverse functions as logarithms

fn one() -> Basic {
    Basic::one()
}

fn asin_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    let inner: Basic = a2(&m2(&i(), x)?, &sqrt(&sub(&one(), &pow(x, &int(2))?)?)?)?;
    Ok(Some(m2(&neg(&i())?, &log(&inner)?)?))
}

fn acos_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    let inner: Basic = a2(&m2(&i(), x)?, &sqrt(&sub(&one(), &pow(x, &int(2))?)?)?)?;
    Ok(Some(a2(&half_pi()?, &m2(&i(), &log(&inner)?)?)?))
}

fn atan_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let ix: Basic = m2(&i(), &args[0])?;
    let difference: Basic = sub(&log(&sub(&one(), &ix)?)?, &log(&a2(&one(), &ix)?)?)?;
    Ok(Some(m2(&div(&i(), &int(2))?, &difference)?))
}

fn acot_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let i_over: Basic = div(&i(), &args[0])?;
    let difference: Basic = sub(&log(&sub(&one(), &i_over)?)?, &log(&a2(&one(), &i_over)?)?)?;
    Ok(Some(m2(&div(&i(), &int(2))?, &difference)?))
}

fn asec_inner(x: &Basic) -> PolyResult<Basic> {
    let root: Basic = sqrt(&sub(&one(), &div(&one(), &pow(x, &int(2))?)?)?)?;
    log(&a2(&div(&i(), x)?, &root)?)
}

fn asec_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(a2(&half_pi()?, &m2(&i(), &asec_inner(&args[0])?)?)?))
}

fn acsc_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    Ok(Some(m2(&neg(&i())?, &asec_inner(&args[0])?)?))
}

fn asinh_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    Ok(Some(log(&a2(x, &sqrt(&a2(&pow(x, &int(2))?, &one())?)?)?)?))
}

fn acosh_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    let roots: Basic = m2(&sqrt(&a2(x, &one())?)?, &sqrt(&sub(x, &one())?)?)?;
    Ok(Some(log(&a2(x, &roots)?)?))
}

fn atanh_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    let difference: Basic = sub(&log(&a2(&one(), x)?)?, &log(&sub(&one(), x)?)?)?;
    Ok(Some(div(&difference, &int(2))?))
}

fn acoth_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let r: Basic = div(&one(), &args[0])?;
    let difference: Basic = sub(&log(&a2(&one(), &r)?)?, &log(&sub(&one(), &r)?)?)?;
    Ok(Some(div(&difference, &int(2))?))
}

fn asech_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let r: Basic = div(&one(), &args[0])?;
    let roots: Basic = m2(&sqrt(&sub(&r, &one())?)?, &sqrt(&a2(&r, &one())?)?)?;
    Ok(Some(log(&a2(&r, &roots)?)?))
}

fn acsch_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let x: &Basic = &args[0];
    let r: Basic = div(&one(), x)?;
    let root: Basic = sqrt(&a2(&div(&one(), &pow(x, &int(2))?)?, &one())?)?;
    Ok(Some(log(&a2(&r, &root)?)?))
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan2._eval_rewrite_as_log
fn atan2_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let (y, x) = (&args[0], &args[1]);
    let modulus: Basic = sqrt(&add(&[m2(x, x)?, m2(y, y)?])?)?;
    let quotient: Basic = div(&add(&[x.clone(), m2(&i(), y)?])?, &modulus)?;
    Ok(Some(mul(&[neg(&i())?, log(&quotient)?])?))
}

fn ynm_not_ported(_: &Basic, _: &[Basic], _: &RewriteHints) -> Out {
    Err(not_ported("Ynm rewritten as cos or sin (simplify, trigsimp)"))
}

/// The rule of the node for one of the ported targets.
pub fn rule_for(e: &Basic, target: &str) -> Option<Rule> {
    use Function as F;
    if let Some(rule) = trig_rule_for(e, target) {
        return Some(rule);
    }
    if let Some(rule) = super::inverse_rewrites::rule_for(e, target) {
        return Some(rule);
    }
    if let Some(rule) = super::pow_rewrites::rule_for(e, target) {
        return Some(rule);
    }
    let f: Option<Function> = match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    };
    let rule: Rule = match (target, e.head()) {
        ("exp", Head::Pow) => pow_exp,
        ("cos", Head::Pow) => pow_cos,
        ("sin", Head::Pow) => pow_sin,
        ("cos", Head::Exp1) => e_cos,
        ("sin", Head::Exp1) => e_sin,
        _ => match (target, f?) {
            ("exp", F::Sin) => sin_exp,
            ("exp", F::Cos) => cos_exp,
            ("exp", F::Tan) => tan_exp,
            ("exp", F::Cot) => cot_exp,
            ("exp", F::Sec) => sec_exp,
            ("exp", F::Csc) => csc_exp,
            ("exp", F::Sinh) => sinh_exp,
            ("exp", F::Cosh) => cosh_exp,
            ("exp", F::Tanh) => tanh_exp,
            ("exp", F::Coth) => coth_exp,
            ("exp", F::Sech) => sech_exp,
            ("exp", F::Csch) => csch_exp,
            ("cos", F::Sin) => sin_cos,
            ("cos", F::Tan) => tan_cos,
            ("cos", F::Cot) => cot_cos,
            ("cos", F::Sec) => sec_cos,
            ("cos", F::Csc) => csc_cos,
            ("cos", F::Cosh) => cosh_cos,
            ("cos", F::Sech) => sech_cos,
            ("cos", F::Exp) => exp_cos,
            ("sin", F::Cos) => cos_sin,
            ("sin", F::Tan) => tan_sin,
            ("sin", F::Cot) => cot_sin,
            ("sin", F::Sec) => sec_sin,
            ("sin", F::Csc) => csc_sin,
            ("sin", F::Exp) => exp_sin,
            ("sin", F::Sinh) => sinh_sin,
            ("sin", F::Csch) => csch_sin,
            ("cos" | "sin", F::Ynm) => ynm_not_ported,
            ("log", F::Asin) => asin_log,
            ("log", F::Acos) => acos_log,
            ("log", F::Atan) => atan_log,
            ("log", F::Acot) => acot_log,
            ("log", F::Asec) => asec_log,
            ("log", F::Acsc) => acsc_log,
            ("log", F::Asinh) => asinh_log,
            ("log", F::Acosh) => acosh_log,
            ("log", F::Atanh) => atanh_log,
            ("log", F::Acoth) => acoth_log,
            ("log", F::Asech) => asech_log,
            ("log", F::Acsch) => acsch_log,
            ("log", F::Atan2) => atan2_log,
            _ => return None,
        },
    };
    Some(rule)
}
