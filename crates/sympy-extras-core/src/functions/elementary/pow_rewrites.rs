//! The `rewrite` rules into `sqrt`, `pow` and `Pow`, ported from
//! `sympy_extras.functions.elementary._rewrites` (and the `sqrt` rules of
//! `fibonacci`, `lucas` and `tribonacci` of the combinatorial table): the
//! cosines of rational multiples of `pi` in square roots (the Fermat
//! prime denominators 3, 5, 17 and 257, the powers of two by halving, the
//! other denominators by partial fractions), the other trigonometric
//! functions through `cos`, the trigonometric functions of a logarithm as
//! powers, `exp(I*pi*c)` in square roots and `exp(c*log(b))` as `b**c`.

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py (cos._eval_rewrite_as_sqrt, the
// _eval_rewrite_as_pow, _eval_rewrite_as_sqrt and _eval_rewrite_as_Pow methods of sin, cos, tan,
// cot, sec, csc), sympy/functions/elementary/_trigonometric_special.py (cos_table, migcdex,
// ipartfrac, fermat_coords), sympy/functions/elementary/exponential.py (exp._eval_rewrite_as_sqrt,
// _eval_rewrite_as_Pow), sympy/core/power.py (Pow._eval_rewrite_as_sqrt),
// sympy/functions/combinatorial/numbers.py (fibonacci, lucas, tribonacci _eval_rewrite_as_sqrt)
// (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use super::inverse_rewrites::{call, i, n, pi, sqrt, X};
use crate::core::basic::{Basic, Function, Head};
use crate::core::rewriting::{rewrite, RewriteHints, Rule};
use crate::error::PolyResult;
use crate::numbers::Int;
use Function as F;

type Out = PolyResult<Option<Basic>>;

fn rw(e: &Basic, target: &str, hints: &RewriteHints) -> PolyResult<Basic> {
    rewrite(e, &[], target, hints)
}

fn is_fn(e: &Basic, f: Function) -> bool {
    matches!(e.head(), Head::Function(g) if *g == f)
}

fn has_fn(e: &Basic, f: Function) -> bool {
    is_fn(e, f) || e.args().iter().any(|a| has_fn(a, f))
}

fn q(p: i64, d: i64) -> X {
    X(Basic::rational(p.into(), d.into()))
}

fn pwx(base: X, e: X) -> X {
    base.binary(e, crate::core::power::pow)
}

// PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:cos_table
fn cos_value(d: i64) -> X {
    match d {
        3 => q(1, 2),
        5 => (sqrt(n(5)) + 1) / 4,
        17 => {
            let (s17, s2) = (sqrt(n(17)), sqrt(n(2)));
            let inner = s2.clone()
                * (n(-8) * sqrt(n(17) + s17.clone()) - (n(1) - s17.clone()) * sqrt(n(17) - s17.clone()))
                + n(6) * s17.clone()
                + 34;
            sqrt((n(15) + s17.clone()) / 32 + s2 * (sqrt(n(17) - s17) + sqrt(inner)) / 32)
        }
        _ => cos_257(),
    }
}

fn cos_257() -> X {
    let f1 = |a: X, b: X| -> (X, X) {
        let root = sqrt(a.clone().pw(2) + b);
        ((a.clone() + root.clone()) / 2, (a - root) / 2)
    };
    let f2 = |a: X, b: X| -> X { (a.clone() - sqrt(a.pw(2) + b)) / 2 };
    let (t1, t2) = f1(n(-1), n(256));
    let (z1, z3) = f1(t1.clone(), n(64));
    let (z2, z4) = f1(t2.clone(), n(64));
    let (y1, y5) = f1(z1.clone(), n(4) * (n(5) + t1.clone() + n(2) * z1.clone()));
    let (y6, y2) = f1(z2.clone(), n(4) * (n(5) + t2.clone() + n(2) * z2.clone()));
    let (y3, y7) = f1(z3.clone(), n(4) * (n(5) + t1.clone() + n(2) * z3.clone()));
    let (y8, y4) = f1(z4.clone(), n(4) * (n(5) + t2.clone() + n(2) * z4.clone()));
    let g = |a: &X, b: &X, c: &X, d: &X| n(-4) * (a.clone() + b.clone() + c.clone() + n(2) * d.clone());
    let (x1, x9) = f1(y1.clone(), g(&t1, &y1, &y3, &y6));
    let (x2, x10) = f1(y2.clone(), g(&t2, &y2, &y4, &y7));
    let (x3, x11) = f1(y3.clone(), g(&t1, &y3, &y5, &y8));
    let (x4, x12) = f1(y4.clone(), g(&t2, &y4, &y6, &y1));
    let (x5, x13) = f1(y5.clone(), g(&t1, &y5, &y7, &y2));
    let (x6, x14) = f1(y6.clone(), g(&t2, &y6, &y8, &y3));
    let (x15, x7) = f1(y7.clone(), g(&t1, &y7, &y1, &y4));
    let (x8, x16) = f1(y8.clone(), g(&t2, &y8, &y2, &y5));
    let h = |a: &X, b: &X, c: &X, d: &X| n(-4) * (a.clone() + b.clone() + c.clone() + d.clone());
    let v1 = f2(x1.clone(), h(&x1, &x2, &x3, &x6));
    let v2 = f2(x2.clone(), h(&x2, &x3, &x4, &x7));
    let v3 = f2(x8.clone(), h(&x8, &x9, &x10, &x13));
    let v4 = f2(x9.clone(), h(&x9, &x10, &x11, &x14));
    let v5 = f2(x10.clone(), h(&x10, &x11, &x12, &x15));
    let v6 = f2(x16.clone(), h(&x16, &x1, &x2, &x5));
    let u1 = -f2(-v1, n(-4) * (v2 + v3));
    let u2 = -f2(-v4, n(-4) * (v5 + v6));
    let w1 = n(-2) * f2(-u1, n(-4) * u2);
    sqrt(sqrt(n(2)) * sqrt(w1 + 4) / 8 + q(1, 2))
}

// PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:migcdex
fn migcdex(xs: &[Int]) -> (Vec<Int>, Int) {
    use crate::core::intfunc::igcdex;
    match xs.len() {
        0 => (Vec::new(), Int::ZERO),
        1 => (vec![Int::ONE], xs[0].clone()),
        2 => {
            let (u, v, h2) = igcdex(&xs[0], &xs[1]);
            (vec![u, v], h2)
        }
        _ => {
            let (y, g) = migcdex(&xs[1..]);
            let (u, v, h2) = igcdex(&xs[0], &g);
            let mut out: Vec<Int> = vec![u];
            out.extend(y.iter().map(|w| &v * w));
            (out, h2)
        }
    }
}

// PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:ipartfrac
fn ipartfrac(denoms: &[Int]) -> Vec<Int> {
    if denoms.is_empty() {
        return Vec::new();
    }
    let mut denom: Int = Int::ONE;
    for d in denoms {
        denom *= d;
    }
    let quotients: Vec<Int> = denoms.iter().map(|d| &denom / d).collect();
    migcdex(&quotients).0
}

// PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:fermat_coords
fn fermat_coords(m: &Int) -> Option<Vec<Int>> {
    let mut m: Int = m.clone();
    let mut primes: Vec<Int> = Vec::new();
    for p in [3, 5, 17, 257, 65537] {
        let p: Int = Int::from(p);
        if (&m % &p).is_zero() {
            m = &m / &p;
            primes.push(p);
            if m.is_one() {
                return Some(primes);
            }
        }
    }
    None
}

// PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cos._eval_rewrite_as_sqrt
fn cos_sqrt(_: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    use crate::core::function::{expand_default, expand_with, Hints};
    let Some(c) = crate::functions::elementary::trigonometric::pi_coeff(&args[0])? else {
        return Ok(None);
    };
    let Some(value) = c.as_rational().cloned() else {
        return Ok(None);
    };
    if matches!(
        c.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) || value.is_int()
    {
        return Ok(None);
    }
    let p: Int = value.numerator().clone();
    let d: Int = Int::from(value.denominator().clone());
    if let Ok(small) = i64::try_from(&d) {
        if [3, 5, 17, 257].contains(&small) {
            let rv: Basic =
                crate::functions::special::polynomials::chebyshevt(&Basic::integer(p), &cos_value(small).0?)?;
            return Ok(Some(if small < 257 { expand_default(&rv)? } else { rv }));
        }
    }
    if (&d % Int::from(2)).is_zero() {
        // the factors of 2 removed recursively
        let pico2: X = X::of(&c) * 2;
        let nval: Basic = rw(&call(F::Cos, pico2.clone() * pi()).0?, "sqrt", hints)?;
        let half: Basic = ((pico2 + 1) / 2).0?;
        let half_value = half.as_rational().expect("a rational").clone();
        let floor: Int = half_value.floor();
        let sign_cos: i64 = if (&floor % Int::from(2)).is_zero() { 1 } else { -1 };
        return (n(sign_cos) * sqrt((n(1) + X::of(&nval)) / 2)).done();
    }
    let fc: Option<Vec<Int>> = fermat_coords(&d);
    let denoms: Vec<Int> = match &fc {
        Some(f) => f.clone(),
        None => crate::ntheory::factor_::factorint(&d, None)
            .into_iter()
            .map(|(b, k)| b.pow(k as usize))
            .collect(),
    };
    let apart: Vec<Int> = ipartfrac(&denoms);
    let zs: Vec<Basic> = (0..denoms.len())
        .map(|k| crate::core::Symbol::new(&format!("z{}", k), &[]).0)
        .collect();
    let mut pairs: Vec<(Basic, Basic)> = Vec::new();
    for ((z, m), den) in zs.iter().zip(&apart).zip(&denoms) {
        let r: Basic = Basic::rational(&p * m, den.clone())?;
        pairs.push((z.clone(), crate::core::mul::mul(&[r, Basic::pi()])?));
    }
    let sum: Basic = crate::core::add::add(&zs)?;
    let trig_only = Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: true,
        func: false,
    };
    let expanded: Basic = expand_with(&Basic::new(Head::Function(F::Cos), vec![sum]), false, trig_only)?;
    let expanded: Basic = crate::core::subs::subs_pairs(&expanded, &pairs)?;
    if fc.as_ref().map_or(true, |f| f.len() == 1) {
        return Ok(Some(expanded));
    }
    Ok(Some(rw(&expanded, "sqrt", hints)?))
}

/// SymPy's `self.rewrite(cos).rewrite(target)` (`None` when a cosine
/// remains, with `check`: the rules of tan and cot).
fn through_cos(target: &str, check: bool, node: &Basic, hints: &RewriteHints) -> Out {
    let y: Basic = rw(&rw(node, "cos", hints)?, target, hints)?;
    if check && has_fn(&y, F::Cos) {
        return Ok(None);
    }
    Ok(Some(y))
}

fn through_cos_sqrt(node: &Basic, _: &[Basic], hints: &RewriteHints) -> Out {
    through_cos("sqrt", false, node, hints)
}

fn through_cos_sqrt_checked(node: &Basic, _: &[Basic], hints: &RewriteHints) -> Out {
    through_cos("sqrt", true, node, hints)
}

fn through_cos_pow(node: &Basic, _: &[Basic], hints: &RewriteHints) -> Out {
    through_cos("pow", false, node, hints)
}

fn through_cos_pow_checked(node: &Basic, _: &[Basic], hints: &RewriteHints) -> Out {
    through_cos("pow", true, node, hints)
}

/// `x` when the argument is `log(x)`.
fn of_log(arg: &Basic) -> Option<X> {
    if is_fn(arg, F::Log) && arg.args().len() == 1 {
        return Some(X::of(&arg.args()[0]));
    }
    None
}

fn sin_pow_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let Some(x) = of_log(&args[0]) else {
        return Ok(None);
    };
    (i() * pwx(x.clone(), -i()) / 2 - i() * pwx(x, i()) / 2).done()
}

fn cos_pow_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let Some(x) = of_log(&args[0]) else {
        return Ok(None);
    };
    (pwx(x.clone(), i()) / 2 + pwx(x, -i()) / 2).done()
}

fn tan_pow_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let Some(x) = of_log(&args[0]) else {
        return Ok(None);
    };
    let (a, b) = (pwx(x.clone(), -i()), pwx(x, i()));
    (i() * (a.clone() - b.clone()) / (a + b)).done()
}

fn cot_pow_log(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let Some(x) = of_log(&args[0]) else {
        return Ok(None);
    };
    let (a, b) = (pwx(x.clone(), -i()), pwx(x, i()));
    (-i() * (a.clone() + b.clone()) / (a - b)).done()
}

// PORTED FROM SYMPY: ReciprocalTrigonometricFunction._rewrite_reciprocal
fn reciprocal_call(rule: Rule, of: Function, node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    let o: Basic = crate::functions::apply_function(of, &node.args()[..1])?;
    if !is_fn(&o, of) {
        return Ok(None);
    }
    if let Some(r) = rule(&o, args, hints)? {
        if r != crate::functions::apply_function(of, &args[..1])? {
            return Ok(Some(crate::core::power::pow(&r, &Basic::negative_one())?));
        }
    }
    Ok(None)
}

fn sec_sqrt(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(cos_sqrt, F::Cos, node, args, hints)
}

fn csc_sqrt(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(through_cos_sqrt, F::Sin, node, args, hints)
}

fn csc_pow(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(through_cos_pow, F::Sin, node, args, hints)
}

fn sec_pow_log(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(cos_pow_log, F::Cos, node, args, hints)
}

fn csc_pow_log(node: &Basic, args: &[Basic], hints: &RewriteHints) -> Out {
    reciprocal_call(sin_pow_log, F::Sin, node, args, hints)
}

/// SymPy's `exp._eval_rewrite_as_sqrt` on the exponent: `exp(I*pi*c)` as
/// `cos(pi*c) + I*sin(pi*c)` when both evaluate.
fn exp_sqrt_of(e: &Basic) -> Out {
    if !e.is_mul() {
        return Ok(None);
    }
    let mut factors: Vec<Basic> = e.args().to_vec();
    let (pi_b, i_b) = (Basic::pi(), Basic::imaginary_unit());
    let (Some(ip), Some(_)) = (
        factors.iter().position(|f| *f == pi_b),
        factors.iter().position(|f| *f == i_b),
    ) else {
        return Ok(None);
    };
    factors.remove(ip);
    let ii = factors.iter().position(|f| *f == i_b).expect("I among the factors");
    factors.remove(ii);
    let coeff: Basic = crate::core::mul::mul(&factors)?;
    if coeff != Basic::zero() && crate::core::assumptions::is_number(&coeff) {
        let arg: X = pi() * X::of(&coeff);
        let (cosine, sine) = (call(F::Cos, arg.clone()).0?, call(F::Sin, arg).0?);
        if !is_fn(&cosine, F::Cos) && !is_fn(&sine, F::Sin) {
            return (X::of(&cosine) + i() * X::of(&sine)).done();
        }
    }
    Ok(None)
}

fn exp_sqrt(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    exp_sqrt_of(&args[0])
}

fn pow_sqrt(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    if !matches!(args[0].head(), Head::Exp1) {
        return Ok(None);
    }
    exp_sqrt_of(&args[1])
}

fn exp_pow(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let a: &Basic = &args[0];
    if a.is_mul() {
        if let Some(l) = a.args().iter().find(|f| is_fn(f, F::Log) && f.args().len() == 1) {
            let exponent: Basic = crate::core::expr::coeff(a, l)?;
            return Ok(Some(crate::core::power::pow(&l.args()[0], &exponent)?));
        }
    }
    Ok(None)
}

// PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:fibonacci._eval_rewrite_as_sqrt
fn fibonacci_sqrt(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    if args.len() != 1 {
        return Ok(None);
    }
    let m: X = X::of(&args[0]);
    let five: X = sqrt(n(5));
    let difference = pwx(n(1) + five.clone(), m.clone()) - pwx(-five.clone() + 1, m.clone());
    (pwx(n(2), -m) * five * difference / 5).done()
}

// PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:lucas._eval_rewrite_as_sqrt
fn lucas_sqrt(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    let m: X = X::of(&args[0]);
    let five: X = sqrt(n(5));
    (pwx(n(2), -m.clone()) * (pwx(n(1) + five.clone(), m.clone()) + pwx(-five + 1, m))).done()
}

// PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:tribonacci._eval_rewrite_as_sqrt
fn tribonacci_sqrt(_: &Basic, args: &[Basic], _: &RewriteHints) -> Out {
    if args.len() != 1 {
        return Ok(None);
    }
    let m: X = X::of(&args[0]);
    let cbrt = |x: X| pwx(x, q(1, 3));
    let w: X = (n(-1) + i() * sqrt(n(3))) / 2;
    let r33: X = sqrt(n(33));
    let (c1, c2) = (cbrt(n(19) + n(3) * r33.clone()), cbrt(n(19) - n(3) * r33));
    let a: X = (n(1) + c1.clone() + c2.clone()) / 3;
    let b: X = (n(1) + w.clone() * c1.clone() + w.clone().pw(2) * c2.clone()) / 3;
    let c: X = (n(1) + w.clone().pw(2) * c1 + w * c2) / 3;
    let up = |x: &X| pwx(x.clone(), m.clone() + 1);
    let term = |x: &X, y: &X, z: &X| up(x) / ((x.clone() - y.clone()) * (x.clone() - z.clone()));
    (term(&a, &b, &c) + term(&b, &a, &c) + term(&c, &a, &b)).done()
}

/// The rule of the node for the targets `sqrt`, `pow` and `Pow`.
pub fn rule_for(e: &Basic, target: &str) -> Option<Rule> {
    match (target, e.head()) {
        ("sqrt", Head::Pow) => return Some(pow_sqrt),
        (_, Head::Function(_)) => {}
        _ => return None,
    }
    let Head::Function(f) = e.head() else {
        return None;
    };
    let rule: Rule = match (target, *f) {
        ("sqrt", F::Cos) | ("pow", F::Cos) => cos_sqrt,
        ("sqrt", F::Sin) => through_cos_sqrt,
        ("sqrt", F::Tan | F::Cot) => through_cos_sqrt_checked,
        ("sqrt", F::Sec) | ("pow", F::Sec) => sec_sqrt,
        ("sqrt", F::Csc) => csc_sqrt,
        ("sqrt", F::Exp) => exp_sqrt,
        ("sqrt", F::Fibonacci) => fibonacci_sqrt,
        ("sqrt", F::Lucas) => lucas_sqrt,
        ("sqrt", F::Tribonacci) => tribonacci_sqrt,
        ("pow", F::Sin) => through_cos_pow,
        ("pow", F::Tan | F::Cot) => through_cos_pow_checked,
        ("pow", F::Csc) => csc_pow,
        ("Pow", F::Sin) => sin_pow_log,
        ("Pow", F::Cos) => cos_pow_log,
        ("Pow", F::Tan) => tan_pow_log,
        ("Pow", F::Cot) => cot_pow_log,
        ("Pow", F::Sec) => sec_pow_log,
        ("Pow", F::Csc) => csc_pow_log,
        ("Pow", F::Exp) => exp_pow,
        _ => return None,
    };
    Some(rule)
}
