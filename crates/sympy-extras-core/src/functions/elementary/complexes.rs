//! Port of `sympy_extras/functions/elementary/complexes.py`, its first
//! part: `re`, `im`, `conjugate`, `sign` and `Abs` ([`Head::Function`]
//! nodes, [`re`], [`im`], [`conjugate`], [`sign`], [`abs_`] evaluate) with
//! the complex expansion they use ([`as_real_imag`], [`conjugate_of`]).
//!
//! What needs the trigonometric functions (the real and imaginary parts of
//! `exp(z)`, of a non-integer power, of `log(z)` through `arg`), the full
//! `expand` (`as_real_imag` of a symbolic power) or `unevaluated_mul` is
//! the error `NotImplemented`. `arg`, `atan2`, the polar functions,
//! `transpose` and `adjoint` are not ported yet.

// PORTED FROM SYMPY: sympy/functions/elementary/complexes.py (SymPy 1.14),
// through sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

// a node caches its keys in `OnceLock`s that the ordering never reads
#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use crate::core::add::add;
use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{
    as_base_exp, as_coefficient, as_numer_denom, is_comparable, is_false, is_rational_value, is_true, terms_of,
};
use crate::core::function::{expand_mul, expand_multinomial};
use crate::core::mul::{mul, neg};
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::{not_ported, PolyResult};
use crate::numbers::*;

use super::exponential::{exp, log};

use Predicate::*;

/// The real and the imaginary part.
pub type RealImag = (Basic, Basic);

fn node(f: Function, x: Basic) -> Basic {
    Basic::new(Head::Function(f), vec![x])
}

fn i() -> Basic {
    Basic::imaginary_unit()
}

fn minus_one() -> Basic {
    Basic::negative_one()
}

/// `e == n` for an integer `n` (Python's `e == 0` on an expression).
fn equals_int(e: &Basic, n: i64) -> bool {
    is_rational_value(e, n)
}

/// The value of a function node of this module (`apply`): the node when
/// nothing evaluates.
fn apply(f: Function, x: &Basic, rule: fn(&Basic) -> PolyResult<Option<Basic>>) -> PolyResult<Basic> {
    if evaluate() {
        if let Some(value) = rule(x)? {
            return Ok(value);
        }
    }
    Ok(node(f, x.clone()))
}

/// The real part of `x`, evaluated.
pub fn re(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Re, x, re_eval)
}

/// The imaginary part of `x`, evaluated.
pub fn im(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Im, x, im_eval)
}

/// The complex conjugate of `x`, evaluated.
pub fn conjugate(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Conjugate, x, conjugate_of)
}

/// The complex sign of `x`, evaluated.
pub fn sign(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Sign, x, sign_eval)
}

/// The absolute value of `x`, evaluated (`abs_`).
pub fn abs_(x: &Basic) -> PolyResult<Basic> {
    apply(Function::Abs, x, abs_eval)
}

// ---------------------------------------------------------------------
// conjugation

/// The conjugate of a function node by the rule of its class
/// (`_eval_conjugate`).
fn function_conjugate(e: &Basic, f: Function) -> PolyResult<Option<Basic>> {
    let arg: &Basic = &e.args()[0];
    match f {
        Function::Exp => Ok(Some(exp(&conjugate(arg)?)?)),
        Function::Re | Function::Im | Function::Abs => Ok(Some(e.clone())),
        Function::Sign => Ok(Some(sign(&conjugate(arg)?)?)),
        Function::Conjugate => Ok(Some(arg.clone())),
        Function::Sin | Function::Cos | Function::Tan | Function::Cot | Function::Sec | Function::Csc => {
            Ok(Some(super::trigonometric::trigonometric_conjugate(f, arg)?))
        }
        Function::Sinh | Function::Cosh | Function::Tanh | Function::Coth | Function::Sech | Function::Csch => {
            Ok(Some(super::hyperbolic::hyperbolic_conjugate(f, arg)?))
        }
        Function::Log => {
            // DefinedFunction._eval_conjugate
            if is_true(e, extended_real) {
                return Ok(Some(e.clone()));
            }
            if is_true(e, imaginary) {
                return Ok(Some(mul(&[minus_one(), e.clone()])?));
            }
            Ok(None)
        }
    }
}

/// The conjugate of `e` when a rule of its class gives it, else `None`
/// (`conjugate_of`).
pub fn conjugate_of(e: &Basic) -> PolyResult<Option<Basic>> {
    match e.head() {
        Head::ImaginaryUnit => return Ok(Some(mul(&[minus_one(), i()])?)),
        Head::ComplexInfinity => return Ok(None),
        _ if e.is_number_class() => return Ok(Some(e.clone())),
        Head::Function(f) => return function_conjugate(e, *f),
        Head::Add | Head::Mul => {
            let mut parts: Vec<Basic> = Vec::new();
            for t in e.args() {
                parts.push(conjugate(t)?);
            }
            return Ok(Some(if e.is_add() { add(&parts)? } else { mul(&parts)? }));
        }
        Head::Pow => {
            let (base, exponent) = (&e.args()[0], &e.args()[1]);
            if is_true(exponent, integer) {
                return Ok(Some(pow(&conjugate(base)?, exponent)?));
            }
            if is_true(base, positive) {
                return Ok(Some(pow(base, &conjugate(exponent)?)?));
            }
            if is_true(e, extended_real) {
                return Ok(Some(e.clone()));
            }
            return Ok(None);
        }
        _ => {}
    }
    if is_true(e, extended_real) {
        return Ok(Some(e.clone()));
    }
    if is_true(e, imaginary) {
        return Ok(Some(mul(&[minus_one(), e.clone()])?));
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// complex expansion

fn binomial(n: usize, k: usize) -> Int {
    let mut c: Int = Int::ONE;
    for j in 0..k {
        c = c * Int::from(n - j) / Int::from(j + 1);
    }
    c
}

/// The parts of `(a + b*I)**n` for real `a`, `b` (`_binomial_parts`).
fn binomial_parts(n: usize, a: &Basic, b: &Basic) -> PolyResult<RealImag> {
    let mut real_terms: Vec<Basic> = Vec::new();
    let mut imaginary_terms: Vec<Basic> = Vec::new();
    for k in 0..=n {
        let term: Basic = mul(&[
            Basic::integer(binomial(n, k)),
            pow(a, &Basic::int((n - k) as i64))?,
            pow(b, &Basic::int(k as i64))?,
        ])?;
        if k % 2 == 0 {
            real_terms.push(mul(&[Basic::int(if k % 4 == 0 { 1 } else { -1 }), term])?);
        } else {
            imaginary_terms.push(mul(&[Basic::int(if k % 4 == 1 { 1 } else { -1 }), term])?);
        }
    }
    Ok((add(&real_terms)?, add(&imaginary_terms)?))
}

/// `Mul.as_real_imag`.
fn mul_real_imag(e: &Basic, deep: bool, ignore: Option<&Basic>) -> PolyResult<Option<RealImag>> {
    let mut other: Vec<Basic> = Vec::new();
    let mut coeff_real: Vec<Basic> = Vec::new();
    let mut coeff_imaginary: Vec<Basic> = Vec::new();
    let mut add_terms: Basic = Basic::one();
    for a in e.args() {
        let Some((r, im_part)) = as_real_imag(a, deep, None)? else {
            return Ok(None);
        };
        if is_true(&im_part, zero) {
            coeff_real.push(r);
        } else if is_true(&r, zero) {
            coeff_imaginary.push(mul(&[im_part, i()])?);
        } else if is_true(a, commutative) {
            let a_conjugate: Option<Basic> = if other.is_empty() { None } else { Some(conjugate(a)?) };
            let found: Option<usize> = other.iter().position(|x| Some(x) == a_conjugate.as_ref());
            match found {
                Some(index) => {
                    let x: Basic = other.remove(index);
                    coeff_real.push(pow(&abs_(&x)?, &Basic::int(2))?);
                }
                None => {
                    if a.is_add() {
                        add_terms = mul(&[add_terms, a.clone()])?;
                    } else {
                        other.push(a.clone());
                    }
                }
            }
        } else {
            other.push(a.clone());
        }
    }
    let m: Basic = mul(&other)?;
    if ignore == Some(&m) {
        return Ok(None);
    }
    let mut imaginary_coefficient: Basic = Basic::zero();
    if coeff_imaginary.len() % 2 == 1 {
        imaginary_coefficient = im(&coeff_imaginary.remove(0))?;
    }
    let mut real_factors: Vec<Basic> = coeff_real;
    real_factors.extend(coeff_imaginary);
    let real_coefficient: Basic = mul(&real_factors)?;
    let (r, im_m) = (
        mul(&[real_coefficient.clone(), re(&m)?])?,
        mul(&[real_coefficient.clone(), im(&m)?])?,
    );
    if equals_int(&add_terms, 1) {
        if equals_int(&m, 1) {
            if is_true(&imaginary_coefficient, zero) {
                return Ok(Some((real_coefficient, Basic::zero())));
            }
            return Ok(Some((Basic::zero(), mul(&[real_coefficient, imaginary_coefficient])?)));
        }
        if equals_int(&imaginary_coefficient, 0) {
            return Ok(Some((r, im_m)));
        }
        return Ok(Some((
            mul(&[neg(&imaginary_coefficient)?, im_m])?,
            mul(&[imaginary_coefficient, r])?,
        )));
    }
    let Some((sum_real, sum_imaginary)) = as_real_imag(&expand_mul(&add_terms, false)?, deep, None)? else {
        return Ok(None);
    };
    let combine = |r: &Basic, i: &Basic| -> PolyResult<RealImag> {
        Ok((
            add(&[
                mul(&[r.clone(), sum_real.clone()])?,
                neg(&mul(&[i.clone(), sum_imaginary.clone()])?)?,
            ])?,
            add(&[
                mul(&[i.clone(), sum_real.clone()])?,
                mul(&[r.clone(), sum_imaginary.clone()])?,
            ])?,
        ))
    };
    if equals_int(&imaginary_coefficient, 0) {
        return Ok(Some(combine(&r, &im_m)?));
    }
    let (r2, i2) = (
        mul(&[neg(&imaginary_coefficient)?, im_m])?,
        mul(&[imaginary_coefficient, r])?,
    );
    // SymPy's order of the imaginary part: r*sum_imaginary + i*sum_real
    Ok(Some((
        add(&[
            mul(&[r2.clone(), sum_real.clone()])?,
            neg(&mul(&[i2.clone(), sum_imaginary.clone()])?)?,
        ])?,
        add(&[mul(&[r2, sum_imaginary])?, mul(&[i2, sum_real])?])?,
    )))
}

/// `Pow.as_real_imag`.
fn pow_real_imag(e: &Basic, deep: bool) -> PolyResult<Option<RealImag>> {
    let (base, exponent) = (&e.args()[0], &e.args()[1]);
    if let Some(x) = exponent.as_rational() {
        let Some((re_e, im_e)) = as_real_imag(base, deep, None)? else {
            return Ok(None);
        };
        if x.is_int() {
            if equals_int(&im_e, 0) {
                return Ok(Some((e.clone(), Basic::zero())));
            }
            let n: Int = numer(x);
            let k: usize = usize::try_from(abs(&n)).map_err(|_| not_ported("a huge power"))?;
            if !is_negative(&n) {
                if re_e.is_number_class() && im_e.is_number_class() {
                    let expanded: Basic = expand_multinomial(&pow(base, exponent)?, true)?;
                    if expanded != *e {
                        return as_real_imag(&expanded, deep, None);
                    }
                }
                return Ok(Some(binomial_parts(k, &re_e, &im_e)?));
            }
            let magnitude: Basic = add(&[pow(&re_e, &Basic::int(2))?, pow(&im_e, &Basic::int(2))?])?;
            let inverse: Basic = pow(&magnitude, &minus_one())?;
            let (a, b) = (mul(&[re_e, inverse.clone()])?, mul(&[neg(&im_e)?, inverse])?);
            if a.is_number_class() && b.is_number_class() {
                let z: Basic = add(&[a.clone(), mul(&[b.clone(), i()])?])?;
                let expanded: Basic = expand_multinomial(&pow(&z, &Basic::integer(-n))?, true)?;
                if expanded != *e {
                    return as_real_imag(&expanded, deep, None);
                }
            }
            return Ok(Some(binomial_parts(k, &a, &b)?));
        }
        if is_true(&im_e, zero) && *x == rat(1, 2) {
            if is_true(&re_e, extended_nonnegative) {
                return Ok(Some((e.clone(), Basic::zero())));
            }
            if is_true(&re_e, extended_nonpositive) {
                return Ok(Some((Basic::zero(), pow(&neg(base)?, exponent)?)));
            }
        }
        return Err(not_ported("atan2, cos and sin of a rational power"));
    }
    if matches!(base.head(), Head::Exp1) {
        return Err(not_ported("cos and sin of a power of E"));
    }
    if deep {
        return Err(not_ported("expand(complex=False) of a power"));
    }
    Ok(Some((re(e)?, im(e)?)))
}

/// The real and imaginary parts of a function node (`_eval_as_real_imag`).
fn function_real_imag(e: &Basic, f: Function, deep: bool, ignore: Option<&Basic>) -> PolyResult<Option<RealImag>> {
    match f {
        Function::Re | Function::Im | Function::Abs => Ok(Some((e.clone(), Basic::zero()))),
        Function::Exp => {
            // exp.as_real_imag: exp(re)*cos(im), exp(re)*sin(im)
            let Some((mut r, mut im_part)) = as_real_imag(&e.args()[0], true, None)? else {
                return Ok(None);
            };
            if deep {
                r = crate::core::function::expand_default(&r)?;
                im_part = crate::core::function::expand_default(&im_part)?;
            }
            let modulus: Basic = exp(&r)?;
            use super::trigonometric::{cos, sin};
            Ok(Some((
                mul(&[modulus.clone(), cos(&im_part)?])?,
                mul(&[modulus, sin(&im_part)?])?,
            )))
        }
        Function::Sin | Function::Cos | Function::Tan | Function::Cot | Function::Sec | Function::Csc => {
            super::trigonometric::trigonometric_real_imag(f, &e.args()[0], deep)
        }
        Function::Sinh | Function::Cosh | Function::Tanh | Function::Coth | Function::Sech | Function::Csch => {
            super::hyperbolic::hyperbolic_real_imag(e, f, deep)
        }
        Function::Log => {
            let z: &Basic = &e.args()[0];
            if deep && !z.is_atom() {
                return Err(not_ported("expand of the argument of log"));
            }
            if abs_(z)? == *z {
                return Ok(Some((e.clone(), Basic::zero())));
            }
            Err(not_ported("arg"))
        }
        Function::Sign | Function::Conjugate => {
            if ignore == Some(e) {
                return Ok(None);
            }
            Ok(Some((re(e)?, im(e)?)))
        }
    }
}

/// The real and imaginary parts of `e` by complex expansion; `None` when
/// they would only be `re(ignore)`, `im(ignore)` (`as_real_imag`).
pub fn as_real_imag(e: &Basic, deep: bool, ignore: Option<&Basic>) -> PolyResult<Option<RealImag>> {
    if e.as_rational().is_some() {
        return Ok(Some((e.clone(), Basic::zero())));
    }
    if is_true(e, extended_real) && !(e.is_add() || e.is_mul() || e.is_pow()) {
        return Ok(Some((e.clone(), Basic::zero())));
    }
    match e.head() {
        Head::Add => {
            let mut real_parts: Vec<Basic> = Vec::new();
            let mut imaginary_parts: Vec<Basic> = Vec::new();
            for t in e.args() {
                let Some((r, im_part)) = as_real_imag(t, deep, None)? else {
                    return Ok(None);
                };
                real_parts.push(r);
                imaginary_parts.push(im_part);
            }
            Ok(Some((add(&real_parts)?, add(&imaginary_parts)?)))
        }
        Head::Mul => mul_real_imag(e, deep, ignore),
        Head::Pow => pow_real_imag(e, deep),
        Head::Function(f) => function_real_imag(e, *f, deep, ignore),
        _ => {
            if ignore == Some(e) {
                return Ok(None);
            }
            Ok(Some((re(e)?, im(e)?)))
        }
    }
}

/// `Expr.as_real_imag`: the parts, `(re(e), im(e))` when nothing expands.
pub fn real_imag_parts(e: &Basic) -> PolyResult<RealImag> {
    match as_real_imag(e, true, None)? {
        Some(parts) => Ok(parts),
        None => Ok((re(e)?, im(e)?)),
    }
}

// ---------------------------------------------------------------------
// re and im

/// `re.eval` and `im.eval` of a sum: the terms whose parts are known taken
/// out (`_split_real_imag`).
fn split_real_imag(arg: &Basic, real_part: bool) -> PolyResult<Option<Basic>> {
    let mut included: Vec<Basic> = Vec::new();
    let mut reverted: Vec<Basic> = Vec::new();
    let mut excluded: Vec<Basic> = Vec::new();
    let terms: Vec<Basic> = terms_of(arg);
    let unit: Basic = i();
    for term in &terms {
        if let Some(coefficient) = as_coefficient(term, &unit)? {
            if !is_true(&coefficient, extended_real) {
                reverted.push(coefficient);
            } else if !real_part {
                excluded.push(coefficient);
            }
        } else if real_part && !term.has(&unit) && is_true(term, extended_real) {
            excluded.push(term.clone());
        } else if real_part || term.has(&unit) || !is_true(term, extended_real) {
            match as_real_imag(term, true, Some(arg))? {
                Some((r, im_part)) => excluded.push(if real_part { r } else { im_part }),
                None => included.push(term.clone()),
            }
        }
    }
    if terms.len() == included.len() {
        return Ok(None);
    }
    let (a, b, c) = (add(&included)?, add(&reverted)?, add(&excluded)?);
    if real_part {
        return Ok(Some(add(&[re(&a)?, mul(&[minus_one(), im(&b)?])?, c])?));
    }
    Ok(Some(add(&[im(&a)?, re(&b)?, c])?))
}

fn rotated_is_real(arg: &Basic) -> PolyResult<bool> {
    Ok(is_true(arg, imaginary) || is_true(&mul(&[i(), arg.clone()])?, extended_real))
}

/// `re.eval`.
pub fn re_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(arg.head(), Head::NaN | Head::ComplexInfinity) {
        return Ok(Some(Basic::nan()));
    }
    if is_true(arg, extended_real) {
        return Ok(Some(arg.clone()));
    }
    if rotated_is_real(arg)? {
        return Ok(Some(Basic::zero()));
    }
    if arg.is_function(Function::Conjugate) {
        return Ok(Some(re(&arg.args()[0])?));
    }
    split_real_imag(arg, true)
}

/// `im.eval`.
pub fn im_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(arg.head(), Head::NaN | Head::ComplexInfinity) {
        return Ok(Some(Basic::nan()));
    }
    if is_true(arg, extended_real) {
        return Ok(Some(Basic::zero()));
    }
    if rotated_is_real(arg)? {
        return Ok(Some(mul(&[minus_one(), i(), arg.clone()])?));
    }
    if arg.is_function(Function::Conjugate) {
        return Ok(Some(mul(&[minus_one(), im(&arg.args()[0])?])?));
    }
    split_real_imag(arg, false)
}

// ---------------------------------------------------------------------
// sign

/// `as_coeff_mul` (rational coefficients only): the coefficient and the
/// tuple of the other factors.
fn as_coeff_mul_tuple(e: &Basic) -> (Basic, Vec<Basic>) {
    if e.is_number_class() {
        if e.as_rational().is_some() {
            return (e.clone(), Vec::new());
        }
        if is_true(e, negative) {
            return (minus_one(), vec![neg(e).expect("the negation of a number")]);
        }
        return (Basic::one(), vec![e.clone()]);
    }
    if e.is_mul() {
        let first: &Basic = &e.args()[0];
        if first.is_number_class() {
            if first.as_rational().is_some() {
                return (first.clone(), e.args()[1..].to_vec());
            }
            if is_true(first, extended_negative) {
                let mut rest: Vec<Basic> = vec![neg(first).expect("the negation of a number")];
                rest.extend(e.args()[1..].iter().cloned());
                return (minus_one(), rest);
            }
        }
        return (Basic::one(), e.args().to_vec());
    }
    (Basic::one(), vec![e.clone()])
}

/// `sign.eval`.
pub fn sign_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if arg.is_mul() {
        let (c, factors) = as_coeff_mul_tuple(arg);
        let mut unknown: Vec<Basic> = Vec::new();
        let mut s: Basic = sign(&c)?;
        for a in &factors {
            if is_true(a, extended_negative) {
                s = mul(&[minus_one(), s])?;
            } else if is_true(a, extended_positive) {
            } else if is_true(a, imaginary) {
                let ai: Basic = im(a)?;
                if is_comparable(&ai) {
                    s = mul(&[s, i()])?;
                    if is_true(&ai, extended_negative) {
                        s = mul(&[minus_one(), s])?;
                    }
                } else {
                    unknown.push(a.clone());
                }
            } else {
                unknown.push(a.clone());
            }
        }
        if equals_int(&c, 1) && unknown.len() == factors.len() {
            return Ok(None);
        }
        if unknown.is_empty() {
            return Ok(Some(s));
        }
        let rest: Basic = if unknown.len() == 1 {
            unknown.remove(0)
        } else {
            Basic::new(Head::Mul, unknown)
        };
        return Ok(Some(mul(&[s, sign(&rest)?])?));
    }
    if matches!(arg.head(), Head::NaN) {
        return Ok(Some(Basic::nan()));
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::zero()));
    }
    if is_true(arg, extended_positive) {
        return Ok(Some(Basic::one()));
    }
    if is_true(arg, extended_negative) {
        return Ok(Some(minus_one()));
    }
    if arg.is_function(Function::Sign) {
        return Ok(Some(arg.clone()));
    }
    if is_true(arg, imaginary) {
        if arg.is_pow() && arg.args()[1].as_rational() == Some(&rat(1, 2)) {
            return Ok(Some(i()));
        }
        let rotated: Basic = mul(&[minus_one(), i(), arg.clone()])?;
        if is_true(&rotated, extended_positive) {
            return Ok(Some(i()));
        }
        if is_true(&rotated, extended_negative) {
            return Ok(Some(mul(&[minus_one(), i()])?));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// Abs

/// `Abs(x)**n` for real `x` (`Abs._eval_power`).
pub fn abs_power(base: &Basic, exponent: &Basic) -> PolyResult<Option<Basic>> {
    let arg: &Basic = &base.args()[0];
    if is_true(arg, extended_real) && is_true(exponent, integer) {
        if is_true(exponent, even) {
            return Ok(Some(pow(arg, exponent)?));
        }
        if let Head::Rational {
            value,
            integer_class: true,
        } = exponent.head()
        {
            if *value != -Rat::ONE {
                return Ok(Some(mul(&[
                    pow(arg, &Basic::from_rational(value - Rat::ONE))?,
                    base.clone(),
                ])?));
            }
        }
    }
    Ok(None)
}

/// The nodes of the function `f` in `e` (`_atoms_of`).
fn atoms_of(e: &Basic, f: Function) -> BTreeSet<Basic> {
    e.preorder().into_iter().filter(|n| n.is_function(f)).collect()
}

/// The free symbols of `e` outside its `Abs` nodes.
fn free_symbols_outside_abs(e: &Basic, found: &mut BTreeSet<Basic>) {
    if e.is_function(Function::Abs) {
        return;
    }
    if e.is_symbol() {
        found.insert(e.clone());
    }
    for a in e.args() {
        free_symbols_outside_abs(a, found);
    }
}

/// `Abs.eval`.
pub fn abs_eval(arg: &Basic) -> PolyResult<Option<Basic>> {
    if arg.is_function(Function::Sign) && is_false(&arg.args()[0], zero) {
        return Ok(Some(Basic::one()));
    }
    if arg.is_function(Function::Conjugate) {
        return Ok(Some(abs_(&arg.args()[0])?));
    }
    if could_extract_minus_sign(arg) {
        return Ok(Some(abs_(&mul(&[minus_one(), arg.clone()])?)?));
    }
    let (n, d) = as_numer_denom(arg)?;
    if !d.free_symbols().is_empty() && n.free_symbols().is_empty() {
        return Ok(Some(mul(&[abs_(&n)?, pow(&abs_(&d)?, &minus_one())?])?));
    }
    if arg.is_mul() {
        let mut known: Vec<Basic> = Vec::new();
        let mut unknown: Vec<Basic> = Vec::new();
        for t in arg.args() {
            if t.is_pow() && is_true(&t.args()[1], integer) && is_true(&t.args()[1], negative) {
                let new_base: Basic = abs_(&t.args()[0])?;
                if new_base.is_function(Function::Abs) {
                    unknown.push(t.clone());
                } else {
                    known.push(pow(&new_base, &t.args()[1])?);
                }
            } else {
                let new_factor: Basic = abs_(t)?;
                if new_factor.is_function(Function::Abs) {
                    unknown.push(t.clone());
                } else {
                    known.push(new_factor);
                }
            }
        }
        let rest: Basic = if unknown.is_empty() {
            Basic::one()
        } else {
            node(Function::Abs, mul(&unknown)?)
        };
        return Ok(Some(mul(&[mul(&known)?, rest])?));
    }
    match arg.head() {
        Head::NaN => return Ok(Some(Basic::nan())),
        Head::ComplexInfinity => return Ok(Some(Basic::oo())),
        _ => {}
    }
    if arg.is_pow() {
        let (base, exponent) = as_base_exp(arg)?;
        if is_true(&base, extended_real) {
            if is_true(&exponent, integer) {
                if is_true(&exponent, even) {
                    return Ok(Some(arg.clone()));
                }
                if equals_int(&base, -1) {
                    return Ok(Some(Basic::one()));
                }
                return Ok(Some(pow(&abs_(&base)?, &exponent)?));
            }
            if is_true(&base, extended_nonnegative) {
                return Ok(Some(pow(&base, &re(&exponent)?)?));
            }
            if is_true(&base, extended_negative) {
                let modulus: Basic = pow(&neg(&base)?, &re(&exponent)?)?;
                let factor: Basic = exp(&mul(&[minus_one(), Basic::pi(), im(&exponent)?])?)?;
                return Ok(Some(mul(&[modulus, factor])?));
            }
            return Ok(None);
        }
        if base.free_symbols().is_empty() {
            if let Some((r, im_part)) = as_real_imag(&log(&base)?, true, None)? {
                let z: Basic = add(&[r, mul(&[i(), im_part])?])?;
                return Ok(Some(exp(&re(&mul(&[exponent, z])?)?)?));
            }
        }
    }
    if arg.is_function(Function::Exp) {
        return Ok(Some(exp(&re(&arg.args()[0])?)?));
    }
    if arg.is_add() && (arg.has(&Basic::oo()) || arg.has(&Basic::negative_oo())) {
        if let Some((r, im_part)) = as_real_imag(arg, true, None)? {
            if is_true(&r, infinite) || is_true(&im_part, infinite) {
                return Ok(Some(Basic::oo()));
            }
        }
    }
    if is_true(arg, zero) {
        return Ok(Some(Basic::zero()));
    }
    if is_true(arg, extended_nonnegative) {
        return Ok(Some(arg.clone()));
    }
    if is_true(arg, extended_nonpositive) {
        return Ok(Some(mul(&[minus_one(), arg.clone()])?));
    }
    if is_true(arg, imaginary) {
        let rotated: Basic = mul(&[minus_one(), i(), arg.clone()])?;
        if is_true(&rotated, extended_nonnegative) {
            return Ok(Some(rotated));
        }
    }
    if is_true(arg, extended_real) {
        return Ok(None);
    }
    let conj: Basic = conjugate(arg)?;
    let before: BTreeSet<Basic> = atoms_of(arg, Function::Conjugate);
    let new_conjugates: Vec<Basic> = atoms_of(&conj, Function::Conjugate)
        .into_iter()
        .filter(|c| !before.contains(c))
        .collect();
    if !new_conjugates.is_empty() && new_conjugates.iter().all(|c| arg.has(&c.args()[0])) {
        return Ok(None);
    }
    if *arg != conj && *arg != mul(&[minus_one(), conj.clone()])? {
        let mut symbols: BTreeSet<Basic> = BTreeSet::new();
        free_symbols_outside_abs(arg, &mut symbols);
        let unknown: Vec<Basic> = symbols
            .into_iter()
            .filter(|s| crate::core::expr::is(s, extended_real).is_none())
            .collect();
        if unknown.is_empty() || !unknown.iter().all(|s| conj.has(&node(Function::Conjugate, s.clone()))) {
            return Ok(Some(pow(
                &expand_mul(&mul(&[arg.clone(), conj])?, true)?,
                &Basic::half(),
            )?));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// facts

/// The facts of a node of this module (`_eval_facts`).
pub fn complexes_facts(f: Function, arg: &Basic) -> FactList {
    let a = facts_of(arg);
    let yes = |p: Predicate| a.get(&p) == Some(&true);
    let mut facts: FactList = FactList::new();
    match f {
        Function::Re | Function::Im => {
            facts.update(&[(commutative, true), (extended_real, true), (imaginary, false)]);
            if yes(finite) {
                facts.update(&[(real, true), (finite, true)]);
            }
        }
        Function::Sign => {
            facts.update(&[(commutative, true), (finite, true), (complex, true)]);
            if yes(extended_real) {
                facts.update(&[
                    (real, true),
                    (extended_real, true),
                    (integer, true),
                    (rational, true),
                    (imaginary, false),
                ]);
            }
            if let Some(v) = a.get(&zero) {
                facts.set(zero, *v);
            }
        }
        Function::Abs => {
            facts.update(&[
                (commutative, true),
                (extended_real, true),
                (imaginary, false),
                (extended_nonnegative, true),
            ]);
            if yes(finite) {
                facts.update(&[(real, true), (finite, true), (nonnegative, true), (complex, true)]);
            }
            if yes(extended_real) {
                for p in [integer, rational] {
                    if let Some(v) = a.get(&p) {
                        facts.set(p, *v);
                    }
                }
            }
            if let Some(v) = a.get(&zero) {
                facts.set(zero, *v);
            }
        }
        Function::Conjugate => {
            if let Some(v) = a.get(&commutative) {
                facts.set(commutative, *v);
            }
        }
        _ => unreachable!("the facts of the other functions are in their modules"),
    }
    facts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    #[test]
    fn examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let r: Basic = Symbol::new("r", &[(real, true)]).0;
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        let two_plus_3i: Basic = add(&[Basic::int(2), mul(&[Basic::int(3), i()]).unwrap()]).unwrap();
        assert_eq!(show(re(&two_plus_3i)), "2");
        assert_eq!(show(im(&two_plus_3i)), "3");
        assert_eq!(show(re(&x)), "re(x)");
        assert_eq!(
            show(im(&add(&[x.clone(), mul(&[Basic::int(2), i()]).unwrap()]).unwrap())),
            "im(x) + 2"
        );
        assert_eq!(show(abs_(&Basic::int(-3))), "3");
        assert_eq!(show(abs_(&two_plus_3i)), "sqrt(13)");
        assert_eq!(show(sign(&mul(&[Basic::int(-2), r.clone()]).unwrap())), "-sign(r)");
        assert_eq!(show(conjugate(&add(&[x.clone(), i()]).unwrap())), "conjugate(x) - I");
        assert_eq!(show(abs_(&x)), "Abs(x)");
    }
}
