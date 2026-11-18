//! The `rewrite` rules of the inverse trigonometric and hyperbolic
//! functions in terms of each other (the targets `asin`, `acos`, `atan`,
//! `acot`, `asec`, `acsc`, `asinh`, `acosh`, `atanh`, `acsch`), with
//! `atan2` as `atan`, and the rules into `Abs` (`sign`, `Max`, `Min`),
//! ported from `sympy_extras.functions.elementary._rewrites` and
//! `sympy_extras.functions._rewrites_misc`. The expressions are written
//! with [`X`], whose operators are Python's (one binary evaluated
//! operation each), so that the rules read as their Python source.

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py (the _eval_rewrite_as_asin,
// _eval_rewrite_as_acos, _eval_rewrite_as_atan, _eval_rewrite_as_acot, _eval_rewrite_as_asec,
// _eval_rewrite_as_acsc methods; atan2._eval_rewrite_as_atan), sympy/functions/elementary/hyperbolic.py
// (the _eval_rewrite_as_<name> methods of asinh, acosh, atanh, acoth, asech, acsch other than log),
// sympy/functions/elementary/complexes.py (sign._eval_rewrite_as_Abs),
// sympy/functions/elementary/miscellaneous.py (MinMaxBase._eval_rewrite_as_Abs) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::core::basic::{Basic, Function, Head};
use crate::core::rewriting::{rewrite, RewriteHints, Rule};
use crate::error::PolyResult;
use crate::functions::apply_function;

/// An expression under construction: Python's operators on `Expr`, each
/// one evaluated binary operation, the first error kept.
#[derive(Clone)]
pub(super) struct X(pub(super) PolyResult<Basic>);

impl X {
    pub(super) fn of(b: &Basic) -> X {
        X(Ok(b.clone()))
    }

    pub(super) fn int(n: i64) -> X {
        X(Ok(Basic::integer(n.into())))
    }

    pub(super) fn binary(self, other: X, f: fn(&Basic, &Basic) -> PolyResult<Basic>) -> X {
        match (self.0, other.0) {
            (Ok(a), Ok(b)) => X(f(&a, &b)),
            (Err(e), _) | (_, Err(e)) => X(Err(e)),
        }
    }

    /// `self**n` for an integer `n`.
    pub(super) fn pw(self, n: i64) -> X {
        self.binary(X::int(n), crate::core::power::pow)
    }

    pub(super) fn done(self) -> PolyResult<Option<Basic>> {
        self.0.map(Some)
    }
}

fn add2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    crate::core::add::add(&[a.clone(), b.clone()])
}

fn mul2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}

impl Add for X {
    type Output = X;
    fn add(self, other: X) -> X {
        self.binary(other, add2)
    }
}

impl Sub for X {
    type Output = X;
    fn sub(self, other: X) -> X {
        self.binary(other, crate::core::add::sub)
    }
}

impl Mul for X {
    type Output = X;
    fn mul(self, other: X) -> X {
        self.binary(other, mul2)
    }
}

impl Div for X {
    type Output = X;
    fn div(self, other: X) -> X {
        self.binary(other, crate::core::mul::div)
    }
}

impl Neg for X {
    type Output = X;
    fn neg(self) -> X {
        X(self.0.and_then(|a| crate::core::mul::neg(&a)))
    }
}

impl Add<i64> for X {
    type Output = X;
    fn add(self, n: i64) -> X {
        self + X::int(n)
    }
}

impl Sub<i64> for X {
    type Output = X;
    fn sub(self, n: i64) -> X {
        self - X::int(n)
    }
}

impl Mul<i64> for X {
    type Output = X;
    fn mul(self, n: i64) -> X {
        self * X::int(n)
    }
}

impl Div<i64> for X {
    type Output = X;
    fn div(self, n: i64) -> X {
        self / X::int(n)
    }
}

/// `n + x`, `n - x`, `n * x`, `n / x` with a Python int on the left.
pub(super) fn n(v: i64) -> X {
    X::int(v)
}

pub(super) fn call(f: Function, x: X) -> X {
    X(x.0.and_then(|a| apply_function(f, &[a])))
}

/// The unevaluated node `f(x)` (Python's class constructor).
pub(super) fn raw(f: Function, x: X) -> X {
    X(x.0.map(|a| Basic::new(Head::Function(f), vec![a])))
}

pub(super) fn sqrt(x: X) -> X {
    x.binary(X::of(&Basic::half()), crate::core::power::pow)
}

pub(super) fn i() -> X {
    X::of(&Basic::imaginary_unit())
}

pub(super) fn pi() -> X {
    X::of(&Basic::pi())
}

/// `HALF_PI`, `mul(pi, 1/2)`.
fn p2() -> X {
    pi() * X::of(&Basic::half())
}

use Function as F;

fn asin(x: X) -> X {
    call(F::Asin, x)
}
fn acos(x: X) -> X {
    call(F::Acos, x)
}
fn atan(x: X) -> X {
    call(F::Atan, x)
}
fn acot(x: X) -> X {
    call(F::Acot, x)
}
fn asec(x: X) -> X {
    call(F::Asec, x)
}
fn acsc(x: X) -> X {
    call(F::Acsc, x)
}
fn asinh(x: X) -> X {
    call(F::Asinh, x)
}
fn acosh(x: X) -> X {
    call(F::Acosh, x)
}
fn atanh(x: X) -> X {
    call(F::Atanh, x)
}

fn sx2x(x: &X) -> X {
    sqrt(x.clone().pw(2)) / x.clone()
}

fn xs1x2(x: &X) -> X {
    x.clone() * sqrt(n(1) / x.clone().pw(2))
}

type Out = PolyResult<Option<Basic>>;

/// The rule of the inverse functions: `(class, target)` to the image of
/// the (rewritten) argument.
fn inverse(f: Function, target: &str, x: X) -> Option<X> {
    let c = || x.clone();
    Some(match (f, target) {
        (F::Asin, "acos") => p2() - acos(c()),
        (F::Asin, "atan") => n(2) * atan(c() / (n(1) + sqrt(n(1) - c().pw(2)))),
        (F::Asin, "acot") => n(2) * acot((n(1) + sqrt(n(1) - c().pw(2))) / c()),
        (F::Asin, "asec") => p2() - asec(n(1) / c()),
        (F::Asin, "acsc") => acsc(n(1) / c()),
        (F::Acos, "asin") => p2() - asin(c()),
        (F::Acos, "atan") => atan(sqrt(n(1) - c().pw(2)) / c()) + p2() * (n(1) - c() * sqrt(n(1) / c().pw(2))),
        (F::Acos, "acot") => p2() - n(2) * acot((n(1) + sqrt(n(1) - c().pw(2))) / c()),
        (F::Acos, "asec") => asec(n(1) / c()),
        (F::Acos, "acsc") => p2() - acsc(n(1) / c()),
        (F::Atan, "asin") => sx2x(&c()) * (p2() - asin(n(1) / sqrt(n(1) + c().pw(2)))),
        (F::Atan, "acos") => sx2x(&c()) * acos(n(1) / sqrt(n(1) + c().pw(2))),
        (F::Atan, "acot") => acot(n(1) / c()),
        (F::Atan, "asec") => sx2x(&c()) * asec(sqrt(n(1) + c().pw(2))),
        (F::Atan, "acsc") => sx2x(&c()) * (p2() - acsc(sqrt(n(1) + c().pw(2)))),
        (F::Acot, "asin") => xs1x2(&c()) * (p2() - asin(sqrt(-c().pw(2)) / sqrt(-c().pw(2) - 1))),
        (F::Acot, "acos") => xs1x2(&c()) * acos(sqrt(-c().pw(2)) / sqrt(-c().pw(2) - 1)),
        (F::Acot, "atan") => atan(n(1) / c()),
        (F::Acot, "asec") => xs1x2(&c()) * asec(sqrt((n(1) + c().pw(2)) / c().pw(2))),
        (F::Acot, "acsc") => xs1x2(&c()) * (p2() - acsc(sqrt((n(1) + c().pw(2)) / c().pw(2)))),
        (F::Asec, "asin") => p2() - asin(n(1) / c()),
        (F::Asec, "acos") => acos(n(1) / c()),
        (F::Asec, "atan") => p2() * (n(1) - sx2x(&c())) + sx2x(&c()) * atan(sqrt(c().pw(2) - 1)),
        (F::Asec, "acot") => p2() * (n(1) - sx2x(&c())) + sx2x(&c()) * acot(n(1) / sqrt(c().pw(2) - 1)),
        (F::Asec, "acsc") => p2() - acsc(c()),
        (F::Acsc, "asin") => asin(n(1) / c()),
        (F::Acsc, "acos") => p2() - acos(n(1) / c()),
        (F::Acsc, "atan") => sx2x(&c()) * (p2() - atan(sqrt(c().pw(2) - 1))),
        (F::Acsc, "acot") => sx2x(&c()) * (p2() - acot(n(1) / sqrt(c().pw(2) - 1))),
        (F::Acsc, "asec") => p2() - asec(c()),
        (F::Asinh, "atanh") => atanh(c() / sqrt(n(1) + c().pw(2))),
        (F::Asinh, "acosh") => {
            let ix = i() * c();
            i() * (sqrt(n(1) - ix.clone()) / sqrt(ix.clone() - 1) * acosh(ix) - p2())
        }
        (F::Asinh, "asin") => -i() * raw(F::Asin, i() * c()),
        (F::Asinh, "acos") => i() * raw(F::Acos, i() * c()) - i() * pi() / 2,
        (F::Acosh, "acos") => acosh_factor(&c()) * acos(c()),
        (F::Acosh, "asin") => acosh_factor(&c()) * (p2() - asin(c())),
        (F::Acosh, "asinh") => acosh_factor(&c()) * (p2() + i() * raw(F::Asinh, i() * c())),
        (F::Acosh, "atanh") => {
            let (sxm1, s1mx, sx2m1) = (sqrt(c() - 1), sqrt(n(1) - c()), sqrt(c().pw(2) - 1));
            p2() * sxm1.clone() / s1mx * (n(1) - c() * sqrt(n(1) / c().pw(2)))
                + sxm1 * sqrt(c() + 1) / sx2m1.clone() * atanh(sx2m1 / c())
        }
        (F::Atanh, "asinh") => {
            let f = sqrt(n(1) / (c().pw(2) - 1));
            pi() * c() / (n(2) * sqrt(-c().pw(2)))
                - sqrt(-c()) * sqrt(n(1) - c().pw(2)) / sqrt(c()) * f.clone() * asinh(f)
        }
        (F::Acoth, "atanh") => atanh(n(1) / c()),
        (F::Acoth, "asinh") => {
            pi() * i() / 2
                * (sqrt((c() - 1) / c()) * sqrt(c() / (c() - 1)) - sqrt(n(1) + n(1) / c()) * sqrt(c() / (c() + 1)))
                + c() * sqrt(n(1) / c().pw(2)) * asinh(sqrt(n(1) / (c().pw(2) - 1)))
        }
        (F::Asech, "acosh") => acosh(n(1) / c()),
        (F::Asech, "asinh") => asech_factor(&c()) * (i() * raw(F::Asinh, i() / c()) + p2()),
        (F::Asech, "atanh") => {
            i() * pi()
                * (n(1)
                    - sqrt(c()) * sqrt(n(1) / c())
                    - i() / 2 * sqrt(-c()) / sqrt(c())
                    - i() / 2 * sqrt(c().pw(2)) / sqrt(-c().pw(2)))
                + sqrt(n(1) / (c() + 1)) * sqrt(c() + 1) * atanh(sqrt(n(1) - c().pw(2)))
        }
        (F::Asech, "acsch") => asech_factor(&c()) * (p2() - i() * raw(F::Acsch, i() * c())),
        (F::Acsch, "asinh") => asinh(n(1) / c()),
        (F::Acsch, "acosh") => i() * (sqrt(n(1) - i() / c()) / sqrt(i() / c() - 1) * raw(F::Acosh, i() / c()) - p2()),
        (F::Acsch, "atanh") => {
            let x2p1 = c().pw(2) + 1;
            sqrt(-c().pw(2)) / c() * (p2() - sqrt(-x2p1.clone().pw(2)) / x2p1.clone() * atanh(sqrt(x2p1)))
        }
        _ => return None,
    })
}

fn acosh_factor(x: &X) -> X {
    sqrt(x.clone() - 1) / sqrt(n(1) - x.clone())
}

fn asech_factor(x: &X) -> X {
    sqrt(n(1) / x.clone() - 1) / sqrt(n(1) - n(1) / x.clone())
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan2._eval_rewrite_as_atan
fn atan2_atan(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    use crate::core::logic::true_;
    use crate::core::relational::{lt, ne};
    let (y, x) = (&args[0], &args[1]);
    let zero: Basic = Basic::zero();
    let modulus: X = sqrt(X::of(x) * X::of(x) + X::of(y) * X::of(y));
    // the misc helpers: add(x, sqrt(...)) and mul(y, 1/(...)) as n-ary calls
    let value: X = n(2) * atan(X::of(y) / (X::of(x) + modulus));
    let re_x: Basic = apply_function(F::Re, std::slice::from_ref(x))?;
    let pieces: Vec<(Basic, Basic)> = vec![
        (value.0?, ne(y, &zero)?),
        (Basic::pi(), lt(&re_x, &zero)?),
        (zero.clone(), ne(x, &zero)?),
        (Basic::atom(Head::NaN), true_()),
    ];
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&pieces)?))
}

// PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:sign._eval_rewrite_as_Abs
fn sign_abs(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    use crate::core::logic::true_;
    use crate::core::relational::eq;
    let arg: &Basic = &args[0];
    let ratio: Basic = (X::of(arg) * call(F::Abs, X::of(arg)).pw(-1)).0?;
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&[
        (Basic::zero(), eq(arg, &Basic::zero())?),
        (ratio, true_()),
    ])?))
}

// PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:MinMaxBase._eval_rewrite_as_Abs
fn minmax_abs(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    let f: Function = match node.head() {
        Head::Function(f) => *f,
        _ => unreachable!("Max or Min"),
    };
    let rest: Basic = apply_function(f, &args[1..])?;
    let half: X = X::of(&Basic::half());
    let s: X = (X::of(&args[0]) + X::of(&rest)) * half.clone();
    let d: X = call(F::Abs, X::of(&args[0]) - X::of(&rest)) * half;
    let result: Basic = if f == F::Max { (s + d).0? } else { (s - d).0? };
    Ok(Some(rewrite(&result, &[], "Abs", hints)?))
}

macro_rules! inverse_rules {
    ($($name:ident: $f:ident => $target:literal),* $(,)?) => {
        $(
            fn $name(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
                match inverse(F::$f, $target, X::of(&args[0])) {
                    Some(x) => x.done(),
                    None => Ok(None),
                }
            }
        )*
        fn inverse_rule(f: Function, target: &str) -> Option<Rule> {
            $(
                if f == F::$f && target == $target {
                    return Some($name);
                }
            )*
            None
        }
    };
}

inverse_rules! {
    asin_acos: Asin => "acos", asin_atan: Asin => "atan", asin_acot: Asin => "acot", asin_asec: Asin => "asec",
    asin_acsc: Asin => "acsc", acos_asin: Acos => "asin", acos_atan: Acos => "atan", acos_acot: Acos => "acot",
    acos_asec: Acos => "asec", acos_acsc: Acos => "acsc", atan_asin: Atan => "asin", atan_acos: Atan => "acos",
    atan_acot: Atan => "acot", atan_asec: Atan => "asec", atan_acsc: Atan => "acsc", acot_asin: Acot => "asin",
    acot_acos: Acot => "acos", acot_atan: Acot => "atan", acot_asec: Acot => "asec", acot_acsc: Acot => "acsc",
    asec_asin: Asec => "asin", asec_acos: Asec => "acos", asec_atan: Asec => "atan", asec_acot: Asec => "acot",
    asec_acsc: Asec => "acsc", acsc_asin: Acsc => "asin", acsc_acos: Acsc => "acos", acsc_atan: Acsc => "atan",
    acsc_acot: Acsc => "acot", acsc_asec: Acsc => "asec",
    asinh_atanh: Asinh => "atanh", asinh_acosh: Asinh => "acosh", asinh_asin: Asinh => "asin",
    asinh_acos: Asinh => "acos", acosh_acos: Acosh => "acos", acosh_asin: Acosh => "asin",
    acosh_asinh: Acosh => "asinh", acosh_atanh: Acosh => "atanh", atanh_asinh: Atanh => "asinh",
    acoth_atanh: Acoth => "atanh", acoth_asinh: Acoth => "asinh", asech_acosh: Asech => "acosh",
    asech_asinh: Asech => "asinh", asech_atanh: Asech => "atanh", asech_acsch: Asech => "acsch",
    acsch_asinh: Acsch => "asinh", acsch_acosh: Acsch => "acosh", acsch_atanh: Acsch => "atanh",
}

/// The rule of the node for the inverse targets and `Abs`.
pub fn rule_for(e: &Basic, target: &str) -> Option<Rule> {
    let Head::Function(f) = e.head() else {
        return None;
    };
    match (*f, target) {
        (F::Atan2, "atan") => Some(atan2_atan),
        (F::Sign, "Abs") => Some(sign_abs),
        (F::Max | F::Min, "Abs") => Some(minmax_abs),
        _ => inverse_rule(*f, target),
    }
}
