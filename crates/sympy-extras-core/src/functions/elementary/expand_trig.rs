//! The `expand(trig=True)` rules of the trigonometric and hyperbolic
//! functions (SymPy's `_eval_expand_trig`), ported from
//! `sympy_extras.functions.elementary._expand_trig`: sums of arguments by
//! the addition formulas, integer multiples by the Chebyshev polynomials
//! (sine, cosine), by the real and imaginary parts of `(1 + I*t)**n`
//! (tangent, cotangent) and by binomial sums (hyperbolic tangent and
//! cotangent); the reciprocal functions through their reciprocals.

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py, sympy/functions/elementary/hyperbolic.py
// (the _eval_expand_trig methods; ReciprocalTrigonometricFunction and
// ReciprocalHyperbolicFunction._calculate_reciprocal) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::add::add;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::function::{expand_mul, expand_with, xreplace_evaluated, Hints};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::PolyResult;
use crate::functions::elementary::hyperbolic::{cosh, coth, sinh, tanh};
use crate::functions::elementary::piecewise::as_coeff_mul_rational;
use crate::functions::elementary::trigonometric::{cos, cot, sin, tan};
use crate::numbers::Int;

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

fn inv(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::negative_one())
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

/// SymPy's `Add.as_two_terms`: the first term and the rest.
fn two_terms(a: &Basic) -> PolyResult<(Basic, Basic)> {
    Ok((a.args()[0].clone(), add(&a.args()[1..])?))
}

/// SymPy's `symmetric_poly`: the elementary symmetric polynomial of
/// degree `n`.
fn symmetric_poly(n: usize, gens: &[Basic]) -> PolyResult<Basic> {
    fn go(gens: &[Basic], n: usize, start: usize, current: &mut Vec<Basic>, out: &mut Vec<Basic>) -> PolyResult<()> {
        if current.len() == n {
            out.push(mul(current)?);
            return Ok(());
        }
        for i in start..gens.len() {
            current.push(gens[i].clone());
            go(gens, n, i + 1, current, out)?;
            current.pop();
        }
        Ok(())
    }
    let mut terms: Vec<Basic> = Vec::new();
    go(gens, n, 0, &mut Vec::new(), &mut terms)?;
    add(&terms)
}

/// The integer of an `Integer` node.
fn integer_of(c: &Basic) -> Option<i64> {
    if matches!(
        c.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) {
        c.as_rational().and_then(|v| i64::try_from(v.numerator()).ok())
    } else {
        None
    }
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sin._eval_expand_trig
fn expand_sin(a: &Basic) -> PolyResult<Basic> {
    use crate::functions::special::polynomials::{chebyshevt, chebyshevu};
    if a.is_add() {
        let (x, y) = two_terms(a)?;
        let (sx, sy, cx, cy) = (expand_sin(&x)?, expand_sin(&y)?, expand_cos(&x)?, expand_cos(&y)?);
        return add(&[mul(&[sx, cy])?, mul(&[sy, cx])?]);
    }
    if a.is_mul() {
        let (n, x) = as_coeff_mul_rational(a);
        if let Some(n) = integer_of(&n) {
            if n % 2 != 0 {
                let sign: Basic = pow(&Basic::negative_one(), &int((n - 1).div_euclid(2)))?;
                return mul(&[sign, chebyshevt(&int(n), &sin(&x)?)?]);
            }
            let sign: Basic = pow(&Basic::negative_one(), &int(n.div_euclid(2) - 1))?;
            let product: Basic = mul(&[sign, cos(&x)?, chebyshevu(&int(n - 1), &sin(&x)?)?])?;
            return expand_mul(&product, false);
        }
    }
    sin(a)
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cos._eval_expand_trig
fn expand_cos(a: &Basic) -> PolyResult<Basic> {
    use crate::functions::special::polynomials::chebyshevt;
    if a.is_add() {
        let (x, y) = two_terms(a)?;
        let (sx, sy, cx, cy) = (expand_sin(&x)?, expand_sin(&y)?, expand_cos(&x)?, expand_cos(&y)?);
        return add(&[mul(&[cx, cy])?, neg(&mul(&[sx, sy])?)?]);
    }
    if a.is_mul() {
        let (coeff, terms) = as_coeff_mul_rational(a);
        if integer_of(&coeff).is_some() {
            return chebyshevt(&coeff, &cos(&terms)?);
        }
    }
    cos(a)
}

fn by_dummies(numerator: &Basic, denominator: &Basic, ys: &[Basic], values: &[Basic]) -> PolyResult<Basic> {
    let rule: HashMap<Basic, Basic> = ys.iter().cloned().zip(values.iter().cloned()).collect();
    let q: Basic = mul(&[numerator.clone(), inv(denominator)?])?;
    xreplace_evaluated(&q, &rule)
}

fn real_dummy_power(
    base_of: &dyn Fn(&Basic) -> PolyResult<Basic>,
    n: i64,
    value: &Basic,
    numerator_re: bool,
) -> PolyResult<Basic> {
    use crate::functions::elementary::complexes::{im, re};
    let z: Basic = crate::core::Dummy::new("dummy", &[(Predicate::real, true)]).0;
    let p: Basic = expand_with(&pow(&base_of(&z)?, &int(n))?, true, Hints::DEFAULT)?;
    let q: Basic = if numerator_re {
        mul(&[re(&p)?, inv(&im(&p)?)?])?
    } else {
        mul(&[im(&p)?, inv(&re(&p)?)?])?
    };
    xreplace_evaluated(&q, &HashMap::from([(z, value.clone())]))
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:tan._eval_expand_trig
fn expand_tan(a: &Basic) -> PolyResult<Basic> {
    if a.is_add() {
        let n: usize = a.args().len();
        let tx: Vec<Basic> = a.args().iter().map(expand_tan).collect::<PolyResult<_>>()?;
        let ys: Vec<Basic> = (0..n).map(|_| crate::core::Dummy::new("Y", &[]).0).collect();
        let mut p: [Basic; 2] = [Basic::zero(), Basic::zero()];
        for i in 0..=n {
            let k: usize = 1 - i % 2;
            let sign: i64 = if (i % 4) / 2 == 1 { -1 } else { 1 };
            p[k] = add(&[p[k].clone(), mul(&[symmetric_poly(i, &ys)?, int(sign)])?])?;
        }
        return by_dummies(&p[0], &p[1], &ys, &tx);
    }
    if a.is_mul() {
        let (coeff, terms) = as_coeff_mul_rational(a);
        if let Some(c) = integer_of(&coeff).filter(|c| *c > 1) {
            let base = |z: &Basic| add(&[Basic::one(), mul(&[Basic::imaginary_unit(), z.clone()])?]);
            return real_dummy_power(&base, c, &tan(&terms)?, false);
        }
    }
    tan(a)
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cot._eval_expand_trig
fn expand_cot(a: &Basic) -> PolyResult<Basic> {
    if a.is_add() {
        let n: usize = a.args().len();
        let cx: Vec<Basic> = a.args().iter().map(expand_cot).collect::<PolyResult<_>>()?;
        let ys: Vec<Basic> = (0..n).map(|_| crate::core::Dummy::new("Y", &[]).0).collect();
        let mut p: [Basic; 2] = [Basic::zero(), Basic::zero()];
        for i in (0..=n).rev() {
            let k: usize = (n - i) % 2;
            let sign: i64 = if ((n - i) % 4) / 2 == 1 { -1 } else { 1 };
            p[k] = add(&[p[k].clone(), mul(&[symmetric_poly(i, &ys)?, int(sign)])?])?;
        }
        return by_dummies(&p[0], &p[1], &ys, &cx);
    }
    if a.is_mul() {
        let (coeff, terms) = as_coeff_mul_rational(a);
        if let Some(c) = integer_of(&coeff).filter(|c| *c > 1) {
            let base = |z: &Basic| add(&[z.clone(), Basic::imaginary_unit()]);
            return real_dummy_power(&base, c, &cot(&terms)?, true);
        }
    }
    cot(a)
}

/// `ExpandHints.expand(e, deep)`.
fn expand_by(e: &Basic, deep: bool, hints: &Hints) -> PolyResult<Basic> {
    expand_with(e, deep, Hints { force: false, ..*hints })
}

fn hyperbolic_sum(arg: &Basic, hints: &Hints, cosine: bool) -> PolyResult<Basic> {
    let a: Basic = expand_by(arg, true, hints)?;
    let mut parts: Option<(Basic, Basic)> = None;
    if a.is_add() {
        parts = Some(two_terms(&a)?);
    } else {
        let (coeff, terms) = as_coeff_mul_rational(&a);
        if let Some(c) = integer_of(&coeff) {
            if c != 1 && terms != Basic::one() {
                parts = Some((terms.clone(), mul(&[int(c - 1), terms])?));
            }
        }
    }
    if let Some((x, y)) = parts {
        let e: Basic = if cosine {
            add(&[mul(&[cosh(&x)?, cosh(&y)?])?, mul(&[sinh(&x)?, sinh(&y)?])?])?
        } else {
            add(&[mul(&[sinh(&x)?, cosh(&y)?])?, mul(&[sinh(&y)?, cosh(&x)?])?])?
        };
        return expand_with(
            &e,
            true,
            Hints {
                trig: true,
                ..Hints::DEFAULT
            },
        );
    }
    if cosine {
        cosh(&a)
    } else {
        sinh(&a)
    }
}

// PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:tanh._eval_expand_trig
fn expand_tanh(a: &Basic) -> PolyResult<Basic> {
    use crate::functions::combinatorial::factorials::binomial;
    if a.is_add() {
        let n: usize = a.args().len();
        let tx: Vec<Basic> = a.args().iter().map(expand_tanh).collect::<PolyResult<_>>()?;
        let mut p: [Basic; 2] = [Basic::zero(), Basic::zero()];
        for i in 0..=n {
            p[i % 2] = add(&[p[i % 2].clone(), symmetric_poly(i, &tx)?])?;
        }
        return mul(&[p[1].clone(), inv(&p[0])?]);
    }
    if a.is_mul() {
        let (coeff, terms) = as_coeff_mul_rational(a);
        if let Some(c) = integer_of(&coeff).filter(|c| *c > 1) {
            let t: Basic = tanh(&terms)?;
            let term = |k: i64| -> PolyResult<Basic> { mul(&[binomial(&int(c), &int(k))?, pow(&t, &int(k))?]) };
            let num: Vec<Basic> = (1..=c).step_by(2).map(term).collect::<PolyResult<_>>()?;
            let den: Vec<Basic> = (0..=c).step_by(2).map(term).collect::<PolyResult<_>>()?;
            return mul(&[add(&num)?, inv(&add(&den)?)?]);
        }
    }
    tanh(a)
}

// PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:coth._eval_expand_trig
fn expand_coth(a: &Basic) -> PolyResult<Basic> {
    use crate::functions::combinatorial::factorials::binomial;
    if a.is_add() {
        let cx: Vec<Basic> = a.args().iter().map(expand_coth).collect::<PolyResult<_>>()?;
        let n: usize = a.args().len();
        let mut p: [Vec<Basic>; 2] = [Vec::new(), Vec::new()];
        for i in (0..=n).rev() {
            p[(n - i) % 2].push(symmetric_poly(i, &cx)?);
        }
        return mul(&[add(&p[0])?, inv(&add(&p[1])?)?]);
    }
    if a.is_mul() {
        let (coeff, x) = as_coeff_mul_rational(a);
        if let Some(c) = integer_of(&coeff).filter(|c| *c > 1) {
            // SymPy's Coth(x): the unevaluated node
            let node: Basic = Basic::new(Head::Function(Function::Coth), vec![x]);
            let mut q: [Vec<Basic>; 2] = [Vec::new(), Vec::new()];
            for i in (0..=c).rev() {
                q[((c - i) % 2) as usize].push(mul(&[binomial(&int(c), &int(i))?, pow(&node, &int(i))?])?);
            }
            return mul(&[add(&q[0])?, inv(&add(&q[1])?)?]);
        }
    }
    coth(a)
}

/// The rule of a node (`_eval_expand_trig`), `None` for a node without one.
pub fn expand_trig_node(e: &Basic, hints: &Hints) -> PolyResult<Option<Basic>> {
    let Head::Function(f) = e.head() else {
        return Ok(None);
    };
    let a: &Basic = match e.args().first() {
        Some(a) => a,
        None => return Ok(None),
    };
    // the reciprocal functions: 1/rule(of(arg)) when of(arg) is a node of
    // one of the classes with a rule (SymPy's _calculate_reciprocal applies
    // the rule of `of` to the argument of whichever it is)
    let reciprocal = |of: Basic, rule: &dyn Fn(&Basic) -> PolyResult<Basic>| -> PolyResult<Basic> {
        let own: bool = matches!(
            of.head(),
            Head::Function(
                Function::Sin
                    | Function::Cos
                    | Function::Tan
                    | Function::Cot
                    | Function::Sinh
                    | Function::Cosh
                    | Function::Tanh
                    | Function::Coth
            )
        );
        let expanded: Basic = if own { rule(&of.args()[0])? } else { of };
        inv(&expanded)
    };
    Ok(Some(match f {
        Function::Sin => expand_sin(a)?,
        Function::Cos => expand_cos(a)?,
        Function::Tan => expand_tan(a)?,
        Function::Cot => expand_cot(a)?,
        Function::Sinh => hyperbolic_sum(a, hints, false)?,
        Function::Cosh => hyperbolic_sum(a, hints, true)?,
        Function::Tanh => expand_tanh(a)?,
        Function::Coth => expand_coth(a)?,
        Function::Sec => reciprocal(cos(a)?, &expand_cos)?,
        Function::Csc => reciprocal(sin(a)?, &expand_sin)?,
        Function::Sech => reciprocal(cosh(a)?, &|x: &Basic| hyperbolic_sum(x, hints, true))?,
        Function::Csch => reciprocal(sinh(a)?, &|x: &Basic| hyperbolic_sum(x, hints, false))?,
        _ => return Ok(None),
    }))
}
