//! Port of `sympy_extras/assumptions/intervals.py`: rigorous interval
//! arithmetic on the expressions of the core ([`super::ivarith`], mpmath's
//! `iv` at 53 bits), the enclosure of an expression on a box
//! ([`evaluate`]) and the sign of an enclosure.

use super::ivarith::{Iv, E, PREC};
use crate::core::basic::{Basic, Function, Head};
use crate::numbers::*;

/// A box: for every variable its closed range (`-oo`, `oo` allowed).
pub type Box = Vec<(Basic, (Basic, Basic))>;

fn endpoint(value: &Basic) -> Option<Iv> {
    match value.head() {
        Head::Infinity => Some(Iv::point(E::PInf)),
        Head::NegativeInfinity => Some(Iv::point(E::NInf)),
        _ => evaluate(value, &[]),
    }
}

fn hull(lo: &Basic, hi: &Basic) -> Option<Iv> {
    let (a, b) = (endpoint(lo)?, endpoint(hi)?);
    Some(Iv::new(a.a, b.b))
}

/// An enclosure of the values of `expr` on the box, or `None`
/// (`evaluate`).
pub fn evaluate(expr: &Basic, b: &[(Basic, (Basic, Basic))]) -> Option<Iv> {
    let p: usize = PREC;
    match expr.head() {
        Head::Symbol(_) | Head::Dummy(..) => {
            let (_, (lo, hi)) = b.iter().find(|(s, _)| s == expr)?;
            hull(lo, hi)
        }
        Head::Pi => Some(Iv::pi(p)),
        Head::Exp1 => Some(Iv::e(p)),
        Head::Add => {
            let mut total: Iv = Iv::point(E::zero());
            for a in expr.args() {
                total = total.add(&evaluate(a, b)?, p);
            }
            Some(total)
        }
        Head::Mul => {
            let mut product: Iv = Iv::point(E::one());
            for a in expr.args() {
                product = product.mul(&evaluate(a, b)?, p);
            }
            Some(product)
        }
        Head::Pow => power(&expr.args()[0], &expr.args()[1], b),
        Head::Function(f) => {
            let f: Function = *f;
            if !matches!(
                f,
                Function::Exp
                    | Function::Log
                    | Function::Sin
                    | Function::Cos
                    | Function::Tan
                    | Function::Atan
                    | Function::Sinh
                    | Function::Cosh
                    | Function::Tanh
                    | Function::Abs
            ) {
                return None;
            }
            let v: Iv = evaluate(&expr.args()[0], b)?;
            function(f, &v)
        }
        _ => {
            let r: &Rat = expr.as_rational()?;
            if expr.is_integer_value() {
                return Some(Iv::from_int(&numer(r), p));
            }
            Some(Iv::from_int(&numer(r), p).div(&Iv::from_int(&Int::from(denom(r)), p), p))
        }
    }
}

fn power(base: &Basic, exponent: &Basic, b: &[(Basic, (Basic, Basic))]) -> Option<Iv> {
    let p: usize = PREC;
    let bv: Iv = evaluate(base, b)?;
    let zero: E = E::zero();
    if let Some(r) = exponent.as_rational() {
        if exponent.is_integer_value() {
            let n: i64 = i64::try_from(&numer(r)).ok()?;
            if n >= 0 {
                return Some(bv.pow_int(n, p));
            }
            if bv.a > zero || bv.b < zero {
                return Some(Iv::point(E::one()).div(&bv.pow_int(-n, p), p));
            }
            return None;
        }
        let ratio: Iv = Iv::from_int(&numer(r), p).div(&Iv::from_int(&Int::from(denom(r)), p), p);
        if bv.a > zero {
            return Some(ratio.mul(&bv.log(p)?, p).exp(p));
        }
        let positive: bool = !rat_is_negative(r);
        if bv.a == zero && positive && bv.b > zero {
            let upper: Iv = ratio.mul(&Iv::point(bv.b.clone()).log(p)?, p).exp(p);
            return Some(Iv::new(E::zero(), upper.b));
        }
        if bv.a == zero && bv.b == zero && positive {
            return Some(Iv::point(E::zero()));
        }
        return None;
    }
    let e: Iv = evaluate(exponent, b)?;
    if bv.a <= zero {
        return None;
    }
    Some(e.mul(&bv.log(p)?, p).exp(p))
}

fn function(f: Function, v: &Iv) -> Option<Iv> {
    let p: usize = PREC;
    let two: Iv = Iv::from_int(&Int::from(2u8), p);
    let one: Iv = Iv::from_int(&Int::ONE, p);
    match f {
        Function::Exp => Some(v.exp(p)),
        Function::Log => {
            if v.a > E::zero() {
                v.log(p)
            } else {
                None
            }
        }
        Function::Sin => Some(v.cos_sin(p).1),
        Function::Cos => Some(v.cos_sin(p).0),
        Function::Tan => Some(v.tan(p)),
        Function::Atan => Some(v.atan(p)),
        Function::Sinh => Some(v.exp(p).sub(&v.neg().exp(p), p).div(&two, p)),
        Function::Cosh => Some(v.exp(p).add(&v.neg().exp(p), p).div(&two, p)),
        Function::Tanh => {
            let w: Iv = two.mul(v, p).exp(p);
            Some(w.sub(&one, p).div(&w.add(&one, p), p))
        }
        Function::Abs => Some(v.abs(p)),
        _ => None,
    }
}

/// The sign of every number of the enclosure, `None` when it contains
/// numbers of different signs (`interval_sign`).
pub fn interval_sign(v: &Iv) -> Option<i32> {
    let zero: E = E::zero();
    if v.a > zero {
        return Some(1);
    }
    if v.b < zero {
        return Some(-1);
    }
    if v.a == zero && v.b == zero {
        return Some(0);
    }
    None
}

/// An endpoint as an exact number (`_to_expr`).
pub fn to_expr(value: &E) -> Basic {
    match value {
        E::PInf => Basic::oo(),
        E::NInf => Basic::negative_oo(),
        E::F(r) => Basic::from_rational(r.clone()),
    }
}
