//! Port of `sympy_extras/simplify/fu.py` (itself ported from SymPy's
//! `sympy.simplify.fu`): the trigonometric transformation rules `TR0` to
//! `TR22`, `TRmorrie` and `TRpower` of Fu, Zhong and Zeng, the helpers
//! `trig_split`, `as_f_sign_1`, `process_common_addends` and
//! `hyper_as_trig`, the greedy strategies and `fu`. Each rule works bottom
//! up. Python's dictionaries are insertion-ordered vectors here; where the
//! Python iterates a set, the order of first occurrence is used.

#![allow(clippy::type_complexity, clippy::too_many_arguments, clippy::needless_range_loop, clippy::mutable_key_type, clippy::redundant_guards)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{
    as_base_exp, as_coeff_mul, as_numer_denom, as_powers_dict, factors_of, is, is_integer_class, is_true,
};
use crate::core::exprtools::ExprFactors;
use crate::core::sorting::ordered;
use crate::error::{PolyError, PolyResult};
use crate::functions::elementary::hyperbolic::{cosh, coth, csch, sech, sinh, tanh};
use crate::functions::elementary::trigonometric::{cos, cot, csc, sec, sin, tan};
use crate::numbers::*;

type R = PolyResult<Basic>;
/// A rule on expressions.
pub type Rule = fn(&Basic) -> R;

fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn mul2(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), b.clone()])
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn add2(a: &Basic, b: &Basic) -> R {
    add(&[a.clone(), b.clone()])
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn inv(e: &Basic) -> R {
    pow(e, &Basic::negative_one())
}
fn neg(e: &Basic) -> R {
    crate::core::mul::neg(e)
}
fn half(e: &Basic) -> R {
    mul2(e, &Basic::half())
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn arg(e: &Basic) -> Basic {
    e.args()[0].clone()
}
fn pi() -> Basic {
    Basic::pi()
}
fn sqrt(e: &Basic) -> R {
    pow(e, &Basic::half())
}
fn root2() -> R {
    sqrt(&int(2))
}
fn root3() -> R {
    sqrt(&int(3))
}
fn inv_root3() -> R {
    inv(&root3()?)
}
fn is_zero_number(e: &Basic) -> bool {
    e.is_number_class() && *e == Basic::zero()
}
fn integer_value(e: &Basic) -> Option<i64> {
    if !is_integer_class(e) {
        return None;
    }
    e.as_rational().and_then(|v| i64::try_from(&numer(v)).ok())
}
fn is_non_expr(e: &Basic) -> bool {
    matches!(e.head(), Head::Tuple) || crate::core::logic::is_boolean(e)
}

fn function_of(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn is_f(e: &Basic, f: Function) -> bool {
    function_of(e) == Some(f)
}
const TRIG: [Function; 6] = [
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
];
const HYPER: [Function; 6] = [
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];
fn is_trig(e: &Basic) -> bool {
    function_of(e).is_some_and(|f| TRIG.contains(&f))
}
fn is_hyper(e: &Basic) -> bool {
    function_of(e).is_some_and(|f| HYPER.contains(&f))
}
fn is_sin_or_cos(e: &Basic) -> bool {
    is_f(e, Function::Sin) || is_f(e, Function::Cos)
}

/// The evaluating function of a trigonometric class.
fn rebuild(f: Function, x: &Basic) -> R {
    match f {
        Function::Sin => sin(x),
        Function::Cos => cos(x),
        Function::Tan => tan(x),
        Function::Cot => cot(x),
        Function::Sec => sec(x),
        Function::Csc => csc(x),
        _ => Err(PolyError::Type(format!("not a trigonometric function: {:?}", f))),
    }
}

/// The unevaluated node `F(x)` (Python's class constructor).
fn node(f: Function, x: &Basic) -> Basic {
    Basic::new(Head::Function(f), vec![x.clone()])
}

fn has_function(e: &Basic, fs: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| function_of(n).is_some_and(|f| fs.contains(&f)))
}

fn bottom(rv: &Basic, f: &dyn Fn(&Basic) -> R) -> R {
    let g = |b: &Basic| -> R {
        if is_non_expr(b) {
            Ok(b.clone())
        } else {
            f(b)
        }
    };
    crate::core::traversal::bottom_up(rv, &g, false)
}

/// `replace` with a predicate and a callable, bottom up; a replaced
/// subtree is not examined again.
fn replace(e: &Basic, query: &dyn Fn(&Basic) -> PolyResult<bool>, value: &dyn Fn(&Basic) -> R) -> R {
    let mut n: Basic = e.clone();
    if !n.args().is_empty() {
        let args: Vec<Basic> = n
            .args()
            .iter()
            .map(|a| replace(a, query, value))
            .collect::<PolyResult<_>>()?;
        if args.as_slice() != n.args() {
            n = crate::core::function::rebuild_evaluated(&n, args)?;
        }
    }
    if query(&n)? {
        return value(&n);
    }
    Ok(n)
}

fn factor(e: &Basic) -> R {
    crate::polys::polytools::factor(e, &[], false, true)
}

fn factor_terms(e: &Basic) -> R {
    crate::core::exprtools::factor_terms(e, false, false, false, true)
}

fn lt_true(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(matches!(crate::core::relational::lt(a, b)?.head(), Head::BooleanTrue))
}

fn gt_true(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(matches!(crate::core::relational::gt(a, b)?.head(), Head::BooleanTrue))
}

fn opt_get<'a, K: PartialEq, V>(v: &'a [(K, V)], k: &K) -> Option<&'a V> {
    v.iter().find(|(a, _)| a == k).map(|(_, b)| b)
}

fn opt_set<K: PartialEq, V>(v: &mut Vec<(K, V)>, k: K, value: V) {
    match v.iter().position(|(a, _)| *a == k) {
        Some(i) => v[i].1 = value,
        None => v.push((k, value)),
    }
}

// -- TR0 to TR3

/// `rv` normalised, factored and expanded (`TR0`).
pub fn tr0(rv: &Basic) -> R {
    crate::core::function::expand_default(&factor(&super::radsimp::normal(rv)?)?)
}

/// `sec` and `csc` as `1/cos` and `1/sin` (`TR1`).
pub fn tr1(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if is_f(e, Function::Sec) {
            return inv(&cos(&arg(e))?);
        }
        if is_f(e, Function::Csc) {
            return inv(&sin(&arg(e))?);
        }
        Ok(e.clone())
    })
}

/// `tan` and `cot` as ratios of `sin` and `cos` (`TR2`).
pub fn tr2(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if is_f(e, Function::Tan) {
            return mul2(&sin(&arg(e))?, &inv(&cos(&arg(e))?)?);
        }
        if is_f(e, Function::Cot) {
            return mul2(&cos(&arg(e))?, &inv(&sin(&arg(e))?)?);
        }
        Ok(e.clone())
    })
}

fn tr2i_ok(k: &Basic, e: &Basic, half_: bool) -> bool {
    (is_true(e, Predicate::integer) || is_true(k, Predicate::positive))
        && (is_sin_or_cos(k)
            || (half_
                && k.is_add()
                && k.args().len() >= 2
                && k.args().iter().any(|a| {
                    factors_of(a)
                        .iter()
                        .any(|ai| is_f(ai, Function::Cos) || (ai.is_pow() && is_f(&ai.args()[0], Function::Cos)))
                })))
}

fn tr2i_factorize(d: &mut Vec<(Basic, Basic)>, ddone: &mut Vec<(Basic, Basic)>, half_: bool) -> PolyResult<()> {
    let mut newk: Vec<(Basic, Basic)> = Vec::new();
    for (k, _) in d.iter() {
        if k.is_add() && k.args().len() > 1 {
            let knew: Basic = if half_ { factor(k)? } else { factor_terms(k)? };
            if knew != *k {
                newk.push((k.clone(), knew));
            }
        }
    }
    if newk.is_empty() {
        return Ok(());
    }
    for (k, _) in &newk {
        d.retain(|(a, _)| a != k);
    }
    let product: Basic = mul(&newk.iter().map(|(_, n)| n.clone()).collect::<Vec<Basic>>())?;
    for (k, value) in as_powers_dict(&product)? {
        let v: Basic = match opt_get(d, &k) {
            Some(old) => add2(old, &value)?,
            None => value,
        };
        if tr2i_ok(&k, &v, half_) {
            opt_set(d, k, v);
        } else {
            ddone.push((k, v));
        }
    }
    Ok(())
}

/// Ratios of `sin` and `cos` as `tan`; with `half`, `1 + cos` in the
/// denominator as `tan` of the half angle (`TR2i`).
pub fn tr2i(rv: &Basic, half_: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !e.is_mul() {
            return Ok(e.clone());
        }
        let (n0, d0) = as_numer_denom(e)?;
        if n0.args().is_empty() || d0.args().is_empty() {
            return Ok(e.clone());
        }
        let mut n: Vec<(Basic, Basic)> = as_powers_dict(&n0)?;
        let mut ndone: Vec<(Basic, Basic)> = n.iter().filter(|(k, v)| !tr2i_ok(k, v, half_)).cloned().collect();
        n.retain(|(k, v)| tr2i_ok(k, v, half_));
        if n.is_empty() {
            return Ok(e.clone());
        }
        let mut d: Vec<(Basic, Basic)> = as_powers_dict(&d0)?;
        let mut ddone: Vec<(Basic, Basic)> = d.iter().filter(|(k, v)| !tr2i_ok(k, v, half_)).cloned().collect();
        d.retain(|(k, v)| tr2i_ok(k, v, half_));
        if d.is_empty() {
            return Ok(e.clone());
        }
        tr2i_factorize(&mut n, &mut ndone, half_)?;
        tr2i_factorize(&mut d, &mut ddone, half_)?;
        let mut nn: Vec<(Basic, Option<Basic>)> = n.into_iter().map(|(k, v)| (k, Some(v))).collect();
        let mut dd: Vec<(Basic, Option<Basic>)> = d.into_iter().map(|(k, v)| (k, Some(v))).collect();
        let dd_get = |dd: &Vec<(Basic, Option<Basic>)>, k: &Basic| -> Option<Option<Basic>> { opt_get(dd, k).cloned() };
        let mut t: Vec<Basic> = Vec::new();
        for idx in 0..nn.len() {
            let k: Basic = nn[idx].0.clone();
            let Some(nk) = nn[idx].1.clone() else { continue };
            if is_f(&k, Function::Sin) {
                let a: Basic = node(Function::Cos, &arg(&k));
                if dd_get(&dd, &a) == Some(Some(nk.clone())) {
                    t.push(pow(&tan(&arg(&k))?, &nk)?);
                    nn[idx].1 = None;
                    opt_set(&mut dd, a, None);
                } else if half_ {
                    let a1: Basic = add2(&Basic::one(), &a)?;
                    if dd_get(&dd, &a1) == Some(Some(nk.clone())) {
                        t.push(pow(&tan(&half(&arg(&k))?)?, &nk)?);
                        nn[idx].1 = None;
                        opt_set(&mut dd, a1, None);
                    }
                }
            } else if is_f(&k, Function::Cos) {
                let a: Basic = node(Function::Sin, &arg(&k));
                if dd_get(&dd, &a) == Some(Some(nk.clone())) {
                    t.push(pow(&tan(&arg(&k))?, &neg(&nk)?)?);
                    nn[idx].1 = None;
                    opt_set(&mut dd, a, None);
                }
            } else if half_ && k.is_add() && k.args()[0] == Basic::one() && is_f(&k.args()[1], Function::Cos) {
                let a: Basic = node(Function::Sin, &arg(&k.args()[1]));
                if let Some(Some(da)) = dd_get(&dd, &a) {
                    if da == nk && (is_true(&da, Predicate::integer) || is_true(&a, Predicate::positive)) {
                        t.push(pow(&tan(&half(&arg(&a))?)?, &neg(&nk)?)?);
                        nn[idx].1 = None;
                        opt_set(&mut dd, a, None);
                    }
                }
            }
        }
        if t.is_empty() {
            return Ok(e.clone());
        }
        let powers = |v: &[(Basic, Option<Basic>)]| -> PolyResult<Vec<Basic>> {
            v.iter()
                .filter_map(|(b, ex)| ex.as_ref().filter(|x| **x != Basic::zero()).map(|x| pow(b, x)))
                .collect()
        };
        let mut numerator_parts: Vec<Basic> = t;
        numerator_parts.extend(powers(&nn)?);
        let numerator: Basic = mul(&numerator_parts)?;
        let denominator: Basic = mul(&powers(&dd)?)?;
        let result: Basic = mul2(&numerator, &inv(&denominator)?)?;
        let nd: Vec<Basic> = ndone.iter().map(|(b, ex)| pow(b, ex)).collect::<PolyResult<_>>()?;
        let dn: Vec<Basic> = ddone.iter().map(|(b, ex)| pow(b, ex)).collect::<PolyResult<_>>()?;
        mul(&[result, mul(&nd)?, inv(&mul(&dn)?)?])
    })
}

/// Arguments reduced: the sign out and `pi/4 < x < pi/2` mapped to the
/// co-function of `pi/2 - x` (`TR3`).
pub fn tr3(rv: &Basic) -> R {
    let numeric_mul = |n: &Basic| -> PolyResult<bool> { Ok(n.is_mul() && crate::core::assumptions::is_number(n)) };
    let remul = |n: &Basic| -> R { mul(n.args()) };
    let inner = |x: &Basic| -> R { replace(x, &numeric_mul, &remul) };
    let e0: Basic = replace(rv, &|x: &Basic| Ok(is_trig(x)), &inner)?;
    bottom(&e0, &|e: &Basic| -> R {
        let Some(f) = function_of(e).filter(|f| TRIG.contains(f)) else {
            return Ok(e.clone());
        };
        let e: Basic = rebuild(f, &super::signsimp::signsimp(&arg(e), true)?)?;
        let Some(f) = function_of(&e).filter(|f| TRIG.contains(f)) else {
            return Ok(e);
        };
        let a: Basic = arg(&e);
        let quarter: Basic = mul2(&pi(), &Basic::from_rational(rat(1, 4)))?;
        let halfpi: Basic = mul2(&pi(), &Basic::half())?;
        let first = is(&crate::core::add::sub(&a, &quarter)?, Predicate::positive);
        let second = is(&crate::core::add::sub(&halfpi, &a)?, Predicate::positive);
        if first == Some(true) && second == Some(true) {
            let co: Function = match f {
                Function::Cos => Function::Sin,
                Function::Sin => Function::Cos,
                Function::Tan => Function::Cot,
                Function::Cot => Function::Tan,
                Function::Sec => Function::Csc,
                _ => Function::Sec,
            };
            return rebuild(co, &crate::core::add::sub(&halfpi, &a)?);
        }
        Ok(e)
    })
}

// -- TR5 to TR9

fn tr56(rv: &Basic, f: Function, g: fn(&Basic) -> R, h: &dyn Fn(&Basic) -> R, max: i64, pow_: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_pow() && is_f(&e.args()[0], f)) {
            return Ok(e.clone());
        }
        let ex: &Basic = &e.args()[1];
        if !is_true(ex, Predicate::real) {
            return Ok(e.clone());
        }
        if lt_true(ex, &Basic::zero())? || gt_true(ex, &int(max))? {
            return Ok(e.clone());
        }
        if *ex == Basic::one() {
            return Ok(e.clone());
        }
        let a: Basic = arg(&e.args()[0]);
        let squared = || -> R { h(&pow(&g(&a)?, &int(2))?) };
        if *ex == int(2) {
            return squared();
        }
        let Some(p) = integer_value(ex) else {
            return Ok(e.clone());
        };
        if p.rem_euclid(2) == 1 {
            return mul2(&rebuild(f, &a)?, &pow(&squared()?, &int(p.div_euclid(2)))?);
        }
        let k: i64 = if p == 4 {
            2
        } else if !pow_ {
            p.div_euclid(2)
        } else {
            if crate::ntheory::factor_::perfect_power(&Int::from(p)).is_none() {
                return Ok(e.clone());
            }
            p.div_euclid(2)
        };
        pow(&squared()?, &int(k))
    })
}

fn one_minus(x: &Basic) -> R {
    crate::core::add::sub(&Basic::one(), x)
}

/// `sin(x)**n` for even `n <= max` through `1 - cos(x)**2` (`TR5`).
pub fn tr5(rv: &Basic, max: i64, pow_: bool) -> R {
    tr56(rv, Function::Sin, cos, &one_minus, max, pow_)
}

/// `cos(x)**n` for even `n <= max` through `1 - sin(x)**2` (`TR6`).
pub fn tr6(rv: &Basic, max: i64, pow_: bool) -> R {
    tr56(rv, Function::Cos, sin, &one_minus, max, pow_)
}

/// `cos(x)**2` as `(1 + cos(2*x))/2` (`TR7`).
pub fn tr7(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_pow() && is_f(&e.args()[0], Function::Cos) && e.args()[1] == int(2)) {
            return Ok(e.clone());
        }
        half(&add2(&Basic::one(), &cos(&mul2(&int(2), &arg(&e.args()[0]))?)?)?)
    })
}

/// Products and powers of `sin` and `cos` as sums (`TR8`).
pub fn tr8(rv: &Basic, first: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        let power_case: bool = e.is_pow()
            && is_sin_or_cos(&e.args()[0])
            && (is_true(&e.args()[1], Predicate::integer) || is_true(&e.args()[0], Predicate::positive));
        if !(e.is_mul() || power_case) {
            return Ok(e.clone());
        }
        if first {
            let (n0, d0) = as_numer_denom(e)?;
            let (n, d) = (
                crate::core::function::expand_mul(&n0, true)?,
                crate::core::function::expand_mul(&d0, true)?,
            );
            let (newn, newd) = (tr8(&n, false)?, tr8(&d, false)?);
            let mut e: Basic = e.clone();
            if newn != n || newd != d {
                e = crate::core::exprtools::gcd_terms(&mul2(&newn, &inv(&newd)?)?, false, true, true)?;
                if e.is_mul() && e.args()[0].is_rational_class() && e.args().len() == 2 && e.args()[1].is_add() {
                    let (c, rest) = as_coeff_mul(&e);
                    e = mul2(&c, &rest)?;
                }
            }
            return Ok(e);
        }
        let (mut cs, mut ss, mut others): (Vec<Basic>, Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new(), Vec::new());
        for a in factors_of(e) {
            if is_f(&a, Function::Cos) {
                cs.push(arg(&a));
            } else if is_f(&a, Function::Sin) {
                ss.push(arg(&a));
            } else if a.is_pow() && integer_value(&a.args()[1]).is_some_and(|p| p > 0) && is_sin_or_cos(&a.args()[0]) {
                let p: usize = integer_value(&a.args()[1]).expect("an integer") as usize;
                let target = if is_f(&a.args()[0], Function::Cos) {
                    &mut cs
                } else {
                    &mut ss
                };
                target.extend(std::iter::repeat(arg(&a.args()[0])).take(p));
            } else {
                others.push(a);
            }
        }
        if !((!cs.is_empty() && !ss.is_empty()) || cs.len() > 1 || ss.len() > 1) {
            return Ok(e.clone());
        }
        let mut args: Vec<Basic> = others;
        for _ in 0..cs.len().min(ss.len()) {
            let (a1, a2) = (ss.pop().expect("a sine"), cs.pop().expect("a cosine"));
            args.push(half(&add2(
                &sin(&add2(&a1, &a2)?)?,
                &sin(&crate::core::add::sub(&a1, &a2)?)?,
            )?)?);
        }
        while cs.len() > 1 {
            let (a1, a2) = (cs.pop().expect("a cosine"), cs.pop().expect("a cosine"));
            args.push(half(&add2(
                &cos(&add2(&a1, &a2)?)?,
                &cos(&crate::core::add::sub(&a1, &a2)?)?,
            )?)?);
        }
        if let Some(c) = cs.pop() {
            args.push(cos(&c)?);
        }
        while ss.len() > 1 {
            let (a1, a2) = (ss.pop().expect("a sine"), ss.pop().expect("a sine"));
            args.push(half(&add2(
                &neg(&cos(&add2(&a1, &a2)?)?)?,
                &cos(&crate::core::add::sub(&a1, &a2)?)?,
            )?)?);
        }
        if let Some(s) = ss.pop() {
            args.push(sin(&s)?);
        }
        tr8(&crate::core::function::expand_mul(&mul(&args)?, true)?, true)
    })
}

/// `do` applied to the groups of terms with the same absolute coefficient
/// (and the same `key2`) (`process_common_addends`).
pub fn process_common_addends(
    rv: &Basic,
    do_: &dyn Fn(&Basic) -> R,
    key2: Option<&dyn Fn(&Basic) -> R>,
    key1: bool,
) -> R {
    let mut absc: Vec<((Basic, Basic), Vec<Basic>)> = Vec::new();
    if key1 {
        for a0 in rv.args() {
            let (mut c, mut a) = as_coeff_mul(a0);
            if is_true(&c, Predicate::negative) {
                c = neg(&c)?;
                a = neg(&a)?;
            }
            let k2: Basic = match key2 {
                Some(f) => f(&a)?,
                None => Basic::one(),
            };
            let key = (c, k2);
            match absc.iter().position(|(k, _)| *k == key) {
                Some(i) => absc[i].1.push(a),
                None => absc.push((key, vec![a])),
            }
        }
    } else if let Some(f) = key2 {
        for a0 in rv.args() {
            let key = (Basic::one(), f(a0)?);
            match absc.iter().position(|(k, _)| *k == key) {
                Some(i) => absc[i].1.push(a0.clone()),
                None => absc.push((key, vec![a0.clone()])),
            }
        }
    } else {
        return Err(crate::error::value_error("must have at least one key".to_string()));
    }
    let mut args: Vec<Basic> = Vec::new();
    let mut hit: bool = false;
    for ((coefficient, _), v) in absc {
        if v.len() > 1 {
            let mut group: Basic = crate::core::add::raw_add(v);
            let new: Basic = do_(&group)?;
            if new != group {
                group = new;
                hit = true;
            }
            args.push(mul2(&coefficient, &group)?);
        } else {
            args.push(mul2(&coefficient, &v[0])?);
        }
    }
    if hit {
        return add(&args);
    }
    Ok(rv.clone())
}

type CosSin = (Option<Basic>, Option<Basic>, Option<Basic>);

fn pow_cos_sin(a: &Basic, two: bool) -> PolyResult<Option<CosSin>> {
    let (mut c, mut s): (Option<Basic>, Option<Basic>) = (None, None);
    let mut co: Basic = Basic::one();
    let half_power = |b: &Basic| b.is_pow() && b.args()[1] == Basic::half();
    if a.is_mul() {
        let (number, rest) = as_coeff_mul(a);
        co = number;
        if rest.args().len() > 2 || !two {
            return Ok(None);
        }
        let mut args: Vec<Basic> = if rest.is_mul() {
            rest.args().to_vec()
        } else {
            vec![rest.clone()]
        };
        let first: Basic = args.remove(0);
        if is_f(&first, Function::Cos) {
            c = Some(first);
        } else if is_f(&first, Function::Sin) {
            s = Some(first);
        } else if half_power(&first) {
            co = mul2(&co, &first)?;
        } else {
            return Ok(None);
        }
        if let Some(b) = args.first() {
            if is_f(b, Function::Cos) {
                if c.is_some() {
                    s = Some(b.clone());
                } else {
                    c = Some(b.clone());
                }
            } else if is_f(b, Function::Sin) {
                if s.is_some() {
                    c = Some(b.clone());
                } else {
                    s = Some(b.clone());
                }
            } else if half_power(b) {
                co = mul2(&co, b)?;
            } else {
                return Ok(None);
            }
        }
        return Ok(Some((if co != Basic::one() { Some(co) } else { None }, c, s)));
    }
    if is_f(a, Function::Cos) {
        c = Some(a.clone());
    } else if is_f(a, Function::Sin) {
        s = Some(a.clone());
    }
    if c.is_none() && s.is_none() {
        return Ok(None);
    }
    Ok(Some((if co != Basic::one() { Some(co) } else { None }, c, s)))
}

/// `(gcd, n1, n2, a, b, iscos)` of a split of `a0 + b0` (`trig_split`).
pub type Split = (Basic, i64, i64, Basic, Basic, bool);

fn minus_one_factors() -> ExprFactors {
    ExprFactors::new(vec![(Basic::negative_one(), Basic::one())])
}

/// `a0 + b0` as `gcd*(n1*f(a) + n2*f(b))` with `f` both `cos` (`iscos`) or
/// both `sin`; with `two`, also products of a `cos` and a `sin` and the
/// radical coefficients of `sin(x + pi/4)` and its kin (`trig_split`).
pub fn trig_split(a0: &Basic, b0: &Basic, two: bool) -> PolyResult<Option<Split>> {
    let (fa, fb) = (ExprFactors::of(a0)?, ExprFactors::of(b0)?);
    let (mut ua, mut ub) = fa.normal(&fb)?;
    let gcd: Basic = fa.gcd(&fb)?.as_expr()?;
    let (mut n1, mut n2): (i64, i64) = (1, 1);
    if ua.contains(&Basic::negative_one()) {
        ua = ua.quo(&minus_one_factors())?;
        n1 = -n1;
    } else if ub.contains(&Basic::negative_one()) {
        ub = ub.quo(&minus_one_factors())?;
        n2 = -n2;
    }
    let (a, b) = (ua.as_expr()?, ub.as_expr()?);
    let Some((mut coa, mut ca, mut sa)) = pow_cos_sin(&a, two)? else {
        return Ok(None);
    };
    let Some((mut cob, mut cb, mut sb)) = pow_cos_sin(&b, two)? else {
        return Ok(None);
    };
    if (ca.is_none() && cb.is_some()) || ca.as_ref().is_some_and(|x| is_f(x, Function::Sin)) {
        std::mem::swap(&mut coa, &mut cob);
        std::mem::swap(&mut ca, &mut cb);
        std::mem::swap(&mut sa, &mut sb);
        std::mem::swap(&mut n1, &mut n2);
    }
    if !two {
        let c = ca.clone().or(sa.clone());
        let s = cb.clone().or(sb.clone());
        let (Some(c), Some(s)) = (c, s) else { return Ok(None) };
        if function_of(&c) != function_of(&s) {
            return Ok(None);
        }
        return Ok(Some((gcd, n1, n2, arg(&c), arg(&s), is_f(&c, Function::Cos))));
    }
    if coa.is_none() && cob.is_none() {
        if let (Some(ca), Some(cb), Some(sa), Some(sb)) = (&ca, &cb, &sa, &sb) {
            if (function_of(ca) == function_of(sa)) != (function_of(cb) == function_of(sb)) {
                return Ok(None);
            }
            let args: [&[Basic]; 2] = [ca.args(), sa.args()];
            if ![cb, sb].iter().all(|i| args.contains(&i.args())) {
                return Ok(None);
            }
            return Ok(Some((
                gcd,
                n1,
                n2,
                arg(ca),
                arg(sa),
                function_of(ca) == function_of(sa),
            )));
        }
    }
    if (ca.is_some() && sa.is_some())
        || (cb.is_some() && sb.is_some())
        || ((ca.is_none() && sa.is_none()) || (cb.is_none() && sb.is_none()))
    {
        return Ok(None);
    }
    let c: Basic = ca.or(sa).expect("a function");
    let s: Basic = cb.or(sb).expect("a function");
    if c.args() != s.args() {
        return Ok(None);
    }
    let coa: Basic = coa.unwrap_or_else(Basic::one);
    let cob: Basic = cob.unwrap_or_else(Basic::one);
    if coa == cob {
        return Ok(Some((
            mul2(&gcd, &root2()?)?,
            n1,
            n2,
            arg(&c),
            mul2(&pi(), &Basic::from_rational(rat(1, 4)))?,
            false,
        )));
    }
    let ratio: Basic = mul2(&coa, &inv(&cob)?)?;
    if ratio == root3()? {
        return Ok(Some((
            mul(&[gcd, int(2), cob])?,
            n1,
            n2,
            arg(&c),
            mul2(&pi(), &Basic::from_rational(rat(1, 3)))?,
            false,
        )));
    }
    if ratio == inv_root3()? {
        return Ok(Some((
            mul(&[gcd, int(2), coa])?,
            n1,
            n2,
            arg(&c),
            mul2(&pi(), &Basic::from_rational(rat(1, 6)))?,
            false,
        )));
    }
    Ok(None)
}

/// `(g, f, s)` with `e = g*(f + s)` and `s` equal to `1` or `-1`, for a
/// sum of two terms (`as_f_sign_1`).
pub fn as_f_sign_1(e: &Basic) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    if !e.is_add() || e.args().len() != 2 {
        return Ok(None);
    }
    let (mut a, mut b) = (e.args()[0].clone(), e.args()[1].clone());
    if a == Basic::negative_one() || a == Basic::one() {
        let mut g: Basic = Basic::one();
        if b.is_mul() && b.args()[0].is_number_class() && is_true(&b.args()[0], Predicate::negative) {
            a = neg(&a)?;
            b = neg(&b)?;
            g = Basic::negative_one();
        }
        return Ok(Some((g, b, a)));
    }
    let (fa, fb) = (ExprFactors::of(&a)?, ExprFactors::of(&b)?);
    let (mut ua, mut ub) = fa.normal(&fb)?;
    let mut gcd: Basic = fa.gcd(&fb)?.as_expr()?;
    let (mut n1, mut n2): (i64, i64);
    if ua.contains(&Basic::negative_one()) {
        ua = ua.quo(&minus_one_factors())?;
        (n1, n2) = (-1, 1);
    } else if ub.contains(&Basic::negative_one()) {
        ub = ub.quo(&minus_one_factors())?;
        (n1, n2) = (1, -1);
    } else {
        (n1, n2) = (1, 1);
    }
    let (mut a, mut b) = (ua.as_expr()?, ub.as_expr()?);
    if a == Basic::one() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut n1, &mut n2);
    }
    if n1 == -1 {
        gcd = neg(&gcd)?;
        n2 = -n2;
    }
    if b == Basic::one() {
        return Ok(Some((gcd, a, int(n2))));
    }
    Ok(None)
}

/// The pairwise search over the terms of a sum (TR9, TR10i).
fn pair_search(e: &Basic, do_: &dyn Fn(&Basic) -> R) -> R {
    if !e.is_add() {
        return Ok(e.clone());
    }
    let mut args: Vec<Option<Basic>> = ordered(e.args().to_vec()).into_iter().map(Some).collect();
    let mut hit: bool = false;
    for i in 0..args.len() {
        let Some(ai) = args[i].clone() else { continue };
        for j in i + 1..args.len() {
            let Some(aj) = args[j].clone() else { continue };
            let was: Basic = add2(&ai, &aj)?;
            let new: Basic = do_(&was)?;
            if new != was {
                args[i] = Some(new);
                args[j] = None;
                hit = true;
                break;
            }
        }
    }
    let mut e: Basic = e.clone();
    if hit {
        e = add(&args.into_iter().flatten().collect::<Vec<Basic>>())?;
        if e.is_add() {
            e = do_(&e)?;
        }
    }
    Ok(e)
}

fn tr9_do(e: &Basic) -> R {
    if !e.is_add() {
        return Ok(e.clone());
    }
    let args: Vec<Basic> = ordered(e.args().to_vec());
    if args.len() != 2 {
        return pair_search(e, &tr9_do);
    }
    let Some((gcd, n1, n2, mut a, mut b, iscos)) = trig_split(&args[0], &args[1], false)? else {
        return Ok(e.clone());
    };
    let sums = |a: &Basic, b: &Basic| -> PolyResult<(Basic, Basic)> {
        Ok((half(&add2(a, b)?)?, half(&crate::core::add::sub(a, b)?)?))
    };
    let (mut s_ab, mut d_ab) = sums(&a, &b)?;
    if iscos {
        if n1 == n2 {
            return mul(&[gcd, int(n1), int(2), cos(&s_ab)?, cos(&d_ab)?]);
        }
        if n1 < 0 {
            std::mem::swap(&mut a, &mut b);
            (s_ab, d_ab) = sums(&a, &b)?;
        }
        return mul(&[int(-2), gcd, sin(&s_ab)?, sin(&d_ab)?]);
    }
    if n1 == n2 {
        return mul(&[gcd, int(n1), int(2), sin(&s_ab)?, cos(&d_ab)?]);
    }
    if n1 < 0 {
        std::mem::swap(&mut a, &mut b);
        (s_ab, d_ab) = sums(&a, &b)?;
    }
    mul(&[int(2), gcd, cos(&s_ab)?, sin(&d_ab)?])
}

/// Sums of `cos` or of `sin` as products (`TR9`).
pub fn tr9(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !e.is_add() {
            return Ok(e.clone());
        }
        process_common_addends(e, &tr9_do, None, true)
    })
}

// -- Osborne's rule

fn osborne(e: &Basic, d: &Basic) -> R {
    bottom(e, &|rv: &Basic| -> R {
        let Some(f) = function_of(rv).filter(|_| is_hyper(rv)) else {
            return Ok(rv.clone());
        };
        let a0: Basic = arg(rv);
        let a: Basic = if a0.is_add() {
            crate::core::add::raw_add(a0.args().iter().map(|t| mul2(t, d)).collect::<PolyResult<_>>()?)
        } else {
            mul2(&a0, d)?
        };
        let i: Basic = Basic::imaginary_unit();
        match f {
            Function::Sinh => mul2(&i, &sin(&a)?),
            Function::Cosh => cos(&a),
            Function::Tanh => mul2(&i, &tan(&a)?),
            Function::Coth => mul2(&cot(&a)?, &inv(&i)?),
            Function::Sech => sec(&a),
            _ => mul2(&csc(&a)?, &inv(&i)?),
        }
    })
}

fn osbornei(e: &Basic, d: &Basic) -> R {
    bottom(e, &|rv: &Basic| -> R {
        let Some(f) = function_of(rv).filter(|_| is_trig(rv)) else {
            return Ok(rv.clone());
        };
        let (constant, x) = crate::core::expr::as_independent(&arg(rv), d, true)?;
        let rule: HashMap<Basic, Basic> = HashMap::from([(d.clone(), Basic::one())]);
        let i: Basic = Basic::imaginary_unit();
        let a: Basic = add2(
            &crate::core::function::xreplace_evaluated(&x, &rule)?,
            &mul2(&constant, &i)?,
        )?;
        match f {
            Function::Sin => mul2(&sinh(&a)?, &inv(&i)?),
            Function::Cos => cosh(&a),
            Function::Tan => mul2(&tanh(&a)?, &inv(&i)?),
            Function::Cot => mul2(&coth(&a)?, &i),
            Function::Sec => sech(&a),
            _ => mul2(&csch(&a)?, &i),
        }
    })
}

/// `rv` with the hyperbolic functions written as trigonometric ones (the
/// trigonometric functions masked), and the data of the inverse map
/// ([`HyperAsTrig::inverse`]) (`hyper_as_trig`).
pub struct HyperAsTrig {
    pub expr: Basic,
    back: HashMap<Basic, Basic>,
    d: Basic,
}

impl HyperAsTrig {
    /// The inverse map: back to hyperbolic functions.
    pub fn inverse(&self, x: &Basic) -> R {
        let unmasked: Basic = crate::core::function::xreplace_evaluated(&osbornei(x, &self.d)?, &self.back)?;
        super::radsimp::collect(
            &super::signsimp::signsimp(&unmasked, true)?,
            &[Basic::imaginary_unit()],
            None,
            Some(false),
        )
    }
}

/// See [`HyperAsTrig`].
pub fn hyper_as_trig(rv: &Basic) -> PolyResult<HyperAsTrig> {
    let trigs: Vec<Basic> = rv.atoms(Some(&|n: &Basic| is_trig(n)));
    let reps: Vec<(Basic, Basic)> = trigs
        .iter()
        .map(|t| (t.clone(), crate::core::symbol::Dummy::new("Dummy", &[]).0))
        .collect();
    let forward: HashMap<Basic, Basic> = reps.iter().cloned().collect();
    let masked: Basic = crate::core::function::xreplace_evaluated(rv, &forward)?;
    let back: HashMap<Basic, Basic> = reps.into_iter().map(|(k, v)| (v, k)).collect();
    let d: Basic = crate::core::symbol::Dummy::new("Dummy", &[]).0;
    Ok(HyperAsTrig {
        expr: osborne(&masked, &d)?,
        back,
        d,
    })
}

// -- TR4, TR10 to TR13

/// The values at the special angles evaluated (`TR4`).
pub fn tr4(rv: &Basic) -> R {
    let query = |x: &Basic| -> PolyResult<bool> {
        if !is_trig(x) {
            return Ok(false);
        }
        let r: Basic = mul2(&arg(x), &inv(&pi())?)?;
        Ok(r.is_rational_class()
            && r.as_rational()
                .is_some_and(|v| [1, 2, 3, 4, 6].iter().any(|q| denom(v) == Int::from(*q))))
    };
    let value = |x: &Basic| -> R {
        let a: Basic = arg(x);
        let evaluated: Basic = if a.args().is_empty() {
            a
        } else {
            crate::core::function::rebuild_evaluated(&a, a.args().to_vec())?
        };
        rebuild(function_of(x).expect("a function"), &evaluated)
    };
    replace(rv, &query, &value)
}

fn split_last(a: &Basic, first: bool) -> (Basic, Basic) {
    let mut args: Vec<Basic> = if first {
        ordered(a.args().to_vec())
    } else {
        a.args().to_vec()
    };
    let last: Basic = args.pop().expect("terms");
    let rest: Basic = if args.len() > 1 {
        crate::core::add::raw_add(args)
    } else {
        args.remove(0)
    };
    (last, rest)
}

/// `sin` and `cos` of sums expanded (`TR10`).
pub fn tr10(rv: &Basic, first: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !is_sin_or_cos(e) {
            return Ok(e.clone());
        }
        let x: Basic = arg(e);
        if !x.is_add() {
            return Ok(e.clone());
        }
        let (a, b) = split_last(&x, first);
        let is_sin: bool = is_f(e, Function::Sin);
        if b.is_add() {
            if is_sin {
                return add2(
                    &mul2(&sin(&a)?, &tr10(&cos(&b)?, false)?)?,
                    &mul2(&cos(&a)?, &tr10(&sin(&b)?, false)?)?,
                );
            }
            return crate::core::add::sub(
                &mul2(&cos(&a)?, &tr10(&cos(&b)?, false)?)?,
                &mul2(&sin(&a)?, &tr10(&sin(&b)?, false)?)?,
            );
        }
        if is_sin {
            return add2(&mul2(&sin(&a)?, &cos(&b)?)?, &mul2(&cos(&a)?, &sin(&b)?)?);
        }
        crate::core::add::sub(&mul2(&cos(&a)?, &cos(&b)?)?, &mul2(&sin(&a)?, &sin(&b)?)?)
    })
}

fn tr10i_do(e: &Basic) -> R {
    if !e.is_add() {
        return Ok(e.clone());
    }
    if e.args().len() != 2 {
        return pair_search(e, &tr10i_do);
    }
    let args: Vec<Basic> = ordered(e.args().to_vec());
    let Some((gcd, n1, n2, a, b, same)) = trig_split(&args[0], &args[1], true)? else {
        return Ok(e.clone());
    };
    let gcd: Basic = mul2(&int(n1), &gcd)?;
    if same {
        if n1 == n2 {
            return mul2(&gcd, &cos(&crate::core::add::sub(&a, &b)?)?);
        }
        return mul2(&gcd, &cos(&add2(&a, &b)?)?);
    }
    if n1 == n2 {
        return mul2(&gcd, &sin(&add2(&a, &b)?)?);
    }
    mul2(&gcd, &sin(&crate::core::add::sub(&b, &a)?)?)
}

/// Sums of products of `sin` and `cos` as `sin`/`cos` of sums (`TR10i`).
pub fn tr10i(rv: &Basic) -> R {
    bottom(rv, &|e0: &Basic| -> R {
        if !e0.is_add() {
            return Ok(e0.clone());
        }
        let key2 = |x: &Basic| -> R { Ok(Basic::new(Head::Tuple, ordered(x.free_symbols().into_iter().collect()))) };
        let mut e: Basic = process_common_addends(e0, &tr10i_do, Some(&key2), true)?;
        while e.is_add() {
            let mut byrad: Vec<(Basic, Vec<Option<Basic>>)> = Vec::new();
            for a in e.args() {
                let mut key: Basic = Basic::one();
                if a.is_mul() {
                    if let Some(ai) = a
                        .args()
                        .iter()
                        .find(|ai| ai.is_pow() && ai.args()[1] == Basic::half() && is_integer_class(&ai.args()[0]))
                    {
                        key = ai.clone();
                    }
                }
                match byrad.iter().position(|(k, _)| *k == key) {
                    Some(i) => byrad[i].1.push(Some(a.clone())),
                    None => byrad.push((key, vec![Some(a.clone())])),
                }
            }
            let mut args: Vec<Basic> = Vec::new();
            let keys: Vec<Basic> = byrad.iter().map(|(k, _)| k.clone()).collect();
            for a in &keys {
                for b in [mul2(&root3()?, a)?, inv_root3()?] {
                    let Some(bi) = byrad.iter().position(|(k, _)| *k == b) else {
                        continue;
                    };
                    let ai: usize = byrad.iter().position(|(k, _)| k == a).expect("a key");
                    for i in 0..byrad[ai].1.len() {
                        let Some(item) = byrad[ai].1[i].clone() else { continue };
                        for j in 0..byrad[bi].1.len() {
                            let Some(other) = byrad[bi].1[j].clone() else { continue };
                            let was: Basic = add2(&item, &other)?;
                            let new: Basic = tr10i_do(&was)?;
                            if new != was {
                                args.push(new);
                                byrad[ai].1[i] = None;
                                byrad[bi].1[j] = None;
                                break;
                            }
                        }
                    }
                }
            }
            if !args.is_empty() {
                for (_, v) in &byrad {
                    args.push(add(&v.iter().flatten().cloned().collect::<Vec<Basic>>())?);
                }
                e = add(&args)?;
            } else {
                e = tr10i_do(&e)?;
                break;
            }
        }
        Ok(e)
    })
}

/// Double angles reduced; with `base`, only the functions of `2*base`
/// (`TR11`).
pub fn tr11(rv: &Basic, base: Option<&Basic>) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !is_sin_or_cos(e) {
            return Ok(e.clone());
        }
        if let Some(base) = base {
            let f: Function = function_of(e).expect("a function");
            let mut t: Basic = rebuild(f, &mul2(base, &int(2))?)?;
            let mut co: Basic = Basic::one();
            if t.is_mul() {
                let (number, rest) = as_coeff_mul(&t);
                co = number;
                t = rest;
            }
            if !is_sin_or_cos(&t) {
                return Ok(e.clone());
            }
            if arg(e) == arg(&t) {
                let (c, s) = (cos(base)?, sin(base)?);
                if is_f(e, Function::Cos) {
                    return mul2(
                        &crate::core::add::sub(&pow(&c, &int(2))?, &pow(&s, &int(2))?)?,
                        &inv(&co)?,
                    );
                }
                return mul(&[int(2), c, s, inv(&co)?]);
            }
            return Ok(e.clone());
        }
        if !arg(e).is_number_class() {
            let (c, m) = super::powsimp::as_coeff_mul(&arg(e), true)?;
            if let Some(v) = c.as_rational().filter(|_| c.is_rational_class()) {
                let (p, q) = (numer(v), denom(v));
                if (&p % &Int::from(2)).is_zero() {
                    let factor_q: Basic = if q.is_one() {
                        Basic::one()
                    } else {
                        Basic::from_rational(rat_from_parts(Int::ONE, q))
                    };
                    let a: Basic = mul(&[Basic::integer(div_floor(&p, &Int::from(2))), m, factor_q])?;
                    let (cc, ss) = (tr11(&cos(&a)?, None)?, tr11(&sin(&a)?, None)?);
                    if is_f(e, Function::Sin) {
                        return mul(&[int(2), ss, cc]);
                    }
                    return crate::core::add::sub(&pow(&cc, &int(2))?, &pow(&ss, &int(2))?);
                }
            }
        }
        Ok(e.clone())
    })
}

/// TR11 on the half angles which appear in the other part of a fraction
/// (`_TR11`).
pub fn tr11_half(rv: &Basic) -> R {
    fn sincos_args(flat: &Basic) -> PolyResult<(Vec<Basic>, Vec<Basic>)> {
        let (mut coss, mut sins): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
        for fi in factors_of(flat) {
            let (b, ex) = as_base_exp(&fi)?;
            if integer_value(&ex).is_some_and(|p| p > 0) && is_sin_or_cos(&b) {
                let target = if is_f(&b, Function::Cos) { &mut coss } else { &mut sins };
                if !target.contains(&arg(&b)) {
                    target.push(arg(&b));
                }
            }
        }
        Ok((coss, sins))
    }
    fn handle(e: &Basic, num: &(Vec<Basic>, Vec<Basic>), den: &mut (Vec<Basic>, Vec<Basic>)) -> R {
        let mut e: Basic = e.clone();
        for narg in num.1.clone() {
            let half_arg: Basic = half(&narg)?;
            let in_cos: bool = den.0.contains(&half_arg);
            let in_sin: bool = den.1.contains(&half_arg);
            if !in_cos && !in_sin {
                continue;
            }
            e = tr11(&e, Some(&half_arg))?;
            if in_cos {
                den.0.retain(|x| *x != half_arg);
            } else {
                den.1.retain(|x| *x != half_arg);
            }
        }
        Ok(e)
    }
    bottom(rv, &|e: &Basic| -> R {
        let (n, d) = as_numer_denom(e)?;
        let mut num_args = sincos_args(&n)?;
        let mut den_args = sincos_args(&d)?;
        let e: Basic = handle(e, &num_args, &mut den_args)?;
        handle(&e, &den_args, &mut num_args)
    })
}

/// `tan` of sums expanded (`TR12`).
pub fn tr12(rv: &Basic, first: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !is_f(e, Function::Tan) {
            return Ok(e.clone());
        }
        let x: Basic = arg(e);
        if !x.is_add() {
            return Ok(e.clone());
        }
        let (a, b) = split_last(&x, first);
        let tb: Basic = if b.is_add() { tr12(&tan(&b)?, false)? } else { tan(&b)? };
        let ta: Basic = tan(&a)?;
        mul2(
            &add2(&ta, &tb)?,
            &inv(&crate::core::add::sub(&Basic::one(), &mul2(&ta, &tb)?)?)?,
        )
    })
}

fn ok_d(di: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    if let Some((g, f, s)) = as_f_sign_1(di)? {
        if s == Basic::negative_one()
            && f.is_mul()
            && f.args().len() == 2
            && f.args().iter().all(|fi| is_f(fi, Function::Tan))
        {
            return Ok(Some((g, f)));
        }
    }
    Ok(None)
}

fn ok_n(ni: &Basic) -> Option<(Basic, Basic)> {
    if ni.is_add() && ni.args().len() == 2 && is_f(&ni.args()[0], Function::Tan) && is_f(&ni.args()[1], Function::Tan) {
        return Some((ni.args()[0].clone(), ni.args()[1].clone()));
    }
    None
}

fn sum_of_args(t: &Basic) -> R {
    add(&t.args().iter().map(arg).collect::<Vec<Basic>>())
}

/// `(tan(a) + tan(b))/(1 - tan(a)*tan(b))` as `tan(a + b)` (`TR12i`).
pub fn tr12i(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_add() || e.is_mul() || e.is_pow()) {
            return Ok(e.clone());
        }
        let (n, d) = as_numer_denom(e)?;
        if d.args().is_empty() || n.args().is_empty() {
            return Ok(e.clone());
        }
        let mut dok: Vec<(Basic, Basic)> = Vec::new();
        let mut d_args: Vec<Basic> = factors_of(&d);
        let mut i: usize = 0;
        while i < d_args.len() {
            let di: Basic = d_args[i].clone();
            if let Some((g, t)) = ok_d(&di)? {
                opt_set(&mut dok, sum_of_args(&t)?, Basic::one());
                d_args[i] = g;
            } else if di.is_add() {
                let fd: Basic = factor(&di)?;
                if fd.is_mul() {
                    d_args.extend(fd.args().iter().cloned());
                    d_args[i] = Basic::one();
                }
            } else if di.is_pow()
                && (is_true(&di.args()[1], Predicate::integer) || is_true(&di.args()[0], Predicate::positive))
            {
                if let Some((g, t)) = ok_d(&di.args()[0])? {
                    opt_set(&mut dok, sum_of_args(&t)?, di.args()[1].clone());
                    d_args[i] = pow(&g, &di.args()[1])?;
                } else {
                    let fd: Basic = factor(&di)?;
                    if fd.is_mul() {
                        d_args.extend(fd.args().iter().cloned());
                        d_args[i] = Basic::one();
                    }
                }
            }
            i += 1;
        }
        if dok.is_empty() {
            return Ok(e.clone());
        }
        let mut n_args: Vec<Basic> = factors_of(&factor_terms(&n)?);
        let mut hit: bool = false;
        let mut i: usize = 0;
        while i < n_args.len() {
            let ni: Basic = n_args[i].clone();
            let mut m: Option<(Basic, Basic)> = ok_n(&ni);
            if m.is_none() {
                m = ok_n(&neg(&ni)?);
                if m.is_some() {
                    n_args[i] = Basic::negative_one();
                } else if ni.is_add() {
                    let fnn: Basic = factor(&ni)?;
                    if fnn.is_mul() {
                        n_args.extend(fnn.args().iter().cloned());
                        n_args[i] = Basic::one();
                    }
                    i += 1;
                    continue;
                } else if ni.is_pow()
                    && (is_true(&ni.args()[1], Predicate::integer) || is_true(&ni.args()[0], Predicate::positive))
                {
                    m = ok_n(&ni.args()[0]);
                    if m.is_some() {
                        n_args[i] = Basic::one();
                    } else {
                        let fnn: Basic = factor(&ni)?;
                        if fnn.is_mul() {
                            n_args.extend(fnn.args().iter().cloned());
                            n_args[i] = Basic::one();
                        }
                        i += 1;
                        continue;
                    }
                } else {
                    i += 1;
                    continue;
                }
            } else {
                n_args[i] = Basic::one();
            }
            hit = true;
            let (t1, t2) = m.expect("a pair");
            let s: Basic = add2(&arg(&t1), &arg(&t2))?;
            let Some(ed) = opt_get(&dok, &s).cloned() else {
                return Err(PolyError::Index(format!("no denominator for {}", s)));
            };
            if let Some(newed) = crate::core::expr::extract_additively(&ed, &Basic::one())? {
                if !is_zero_number(&newed) {
                    opt_set(&mut dok, s.clone(), newed);
                } else {
                    dok.retain(|(k, _)| *k != s);
                }
            }
            n_args[i] = mul2(&n_args[i], &neg(&tan(&s)?)?)?;
            i += 1;
        }
        if hit {
            let mut parts: Vec<Basic> = Vec::new();
            for (k, ex) in &dok {
                let ks: Vec<Basic> = if k.is_add() { k.args().to_vec() } else { vec![k.clone()] };
                let tans: Basic = add(&ks.iter().map(tan).collect::<PolyResult<Vec<Basic>>>()?)?;
                parts.push(pow(&add2(&tans, &Basic::negative_one())?, ex)?);
            }
            let denominator: Basic = mul(&parts)?;
            return mul(&[mul(&n_args)?, inv(&mul(&d_args)?)?, inv(&denominator)?]);
        }
        Ok(e.clone())
    })
}

/// Products of `tan` (`cot`) as sums through `tan(a + b)` (`TR13`).
pub fn tr13(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !e.is_mul() {
            return Ok(e.clone());
        }
        let (mut ts, mut cs, mut args): (Vec<Basic>, Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new(), Vec::new());
        for a in factors_of(e) {
            if is_f(&a, Function::Tan) {
                ts.push(arg(&a));
            } else if is_f(&a, Function::Cot) {
                cs.push(arg(&a));
            } else {
                args.push(a);
            }
        }
        if ts.len() < 2 && cs.len() < 2 {
            return Ok(e.clone());
        }
        while ts.len() > 1 {
            let (t1, t2) = (ts.pop().expect("a tangent"), ts.pop().expect("a tangent"));
            let s12: Basic = tan(&add2(&t1, &t2)?)?;
            let inner: Basic = add2(&mul2(&tan(&t1)?, &inv(&s12)?)?, &mul2(&tan(&t2)?, &inv(&s12)?)?)?;
            args.push(crate::core::add::sub(&Basic::one(), &inner)?);
        }
        if let Some(t) = ts.pop() {
            args.push(tan(&t)?);
        }
        while cs.len() > 1 {
            let (t1, t2) = (cs.pop().expect("a cotangent"), cs.pop().expect("a cotangent"));
            let c12: Basic = cot(&add2(&t1, &t2)?)?;
            args.push(add(&[Basic::one(), mul2(&cot(&t1)?, &c12)?, mul2(&cot(&t2)?, &c12)?])?);
        }
        if let Some(c) = cs.pop() {
            args.push(cot(&c)?);
        }
        mul(&args)
    })
}

/// A float of a number for sorting (`_value`); `0` for the others.
fn value_of(n: &Basic) -> f64 {
    if let Some(v) = n.as_rational().filter(|_| n.is_rational_class()) {
        return crate::core::evalf::real_rat(v, 64).to_f64().value();
    }
    match crate::numeric::reliable_complex(n, 15, &[]) {
        Some((re, im)) if im == 0.0 => re,
        _ => 0.0,
    }
}

fn morrie(e: &Basic, first: bool) -> R {
    if !e.is_mul() {
        return Ok(e.clone());
    }
    if first {
        let (n, d) = as_numer_denom(e)?;
        return mul2(&morrie(&n, false)?, &inv(&morrie(&d, false)?)?);
    }
    let mut args: Vec<(Basic, Vec<Basic>)> = Vec::new();
    let mut coss: Vec<(Basic, i64)> = Vec::new();
    let mut other: Vec<Basic> = Vec::new();
    for c in e.args() {
        let (b, ex) = as_base_exp(c)?;
        match integer_value(&ex) {
            Some(p) if is_f(&b, Function::Cos) => {
                let (co, a) = as_coeff_mul(&arg(&b));
                match args.iter().position(|(k, _)| *k == a) {
                    Some(i) => args[i].1.push(co),
                    None => args.push((a, vec![co])),
                }
                opt_set(&mut coss, b, p);
            }
            _ => other.push(c.clone()),
        }
    }
    let mut new: Vec<Basic> = Vec::new();
    for idx in 0..args.len() {
        let a: Basic = args[idx].0.clone();
        let mut cl: Vec<Basic> = std::mem::take(&mut args[idx].1);
        let mut keyed: Vec<(f64, Basic)> = cl.into_iter().map(|c| (value_of(&c), c)).collect();
        keyed.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        cl = keyed.into_iter().map(|(_, c)| c).collect();
        while !cl.is_empty() {
            let mut k: u32 = 0;
            let ci: Basic = cl[0].clone();
            let mut cc: Basic = ci.clone();
            while cl.contains(&cc) {
                k += 1;
                cc = mul2(&cc, &int(2))?;
            }
            if k > 1 {
                let two_k: Basic = Basic::integer(Int::ONE << (k as usize));
                let newarg: Basic = mul(&[
                    sin(&mul(&[two_k.clone(), ci.clone(), a.clone()])?)?,
                    inv(&two_k)?,
                    inv(&sin(&mul2(&ci, &a)?)?)?,
                ])?;
                let mut take: Option<i64> = None;
                let mut ccs: Vec<Basic> = Vec::new();
                for _ in 0..k {
                    cc = mul2(&cc, &Basic::half())?;
                    let key: Basic = node(Function::Cos, &mul2(&a, &cc)?);
                    ccs.push(cc.clone());
                    let value: i64 = *opt_get(&coss, &key).ok_or_else(|| PolyError::Index(format!("{}", key)))?;
                    let current: i64 = match take {
                        Some(t) if t != 0 => t,
                        _ => value,
                    };
                    take = Some(value.min(current));
                }
                for _ in 0..k {
                    let cc: Basic = ccs.pop().expect("a coefficient");
                    let key: Basic = node(Function::Cos, &mul2(&a, &cc)?);
                    let i: usize = coss.iter().position(|(kk, _)| *kk == key).expect("present");
                    coss[i].1 -= take.unwrap_or(0);
                    if coss[i].1 == 0 {
                        if let Some(pos) = cl.iter().position(|x| *x == cc) {
                            cl.remove(pos);
                        }
                    }
                }
                new.push(pow(&newarg, &int(take.unwrap_or(0)))?);
            } else {
                let first_c: Basic = cl.remove(0);
                let b: Basic = cos(&mul2(&first_c, &a)?)?;
                let exponent: i64 = *opt_get(&coss, &b).ok_or_else(|| PolyError::Index(format!("{}", b)))?;
                other.push(pow(&b, &int(exponent))?);
            }
        }
        args[idx].1 = cl;
    }
    if !new.is_empty() {
        let mut parts: Vec<Basic> = new;
        parts.extend(other);
        for (a, ks) in &args {
            for k in ks {
                parts.push(node(Function::Cos, &mul2(k, a)?));
            }
        }
        return mul(&parts);
    }
    Ok(e.clone())
}

/// Morrie's law (`TRmorrie`).
pub fn tr_morrie(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| morrie(e, true))
}

#[derive(Clone)]
struct Item {
    g: Basic,
    e: Basic,
    f: Basic,
    si: Basic,
    a: Basic,
}

/// Factored powers of `sin`/`cos` identities (`TR14`).
pub fn tr14(rv: &Basic, first: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !e.is_mul() {
            return Ok(e.clone());
        }
        if first {
            let (n, d) = as_numer_denom(e)?;
            if d != Basic::one() {
                let (newn, newd) = (tr14(&n, false)?, tr14(&d, false)?);
                if newn != n || newd != d {
                    return mul2(&newn, &inv(&newd)?);
                }
                return Ok(e.clone());
            }
        }
        let mut other: Vec<Basic> = Vec::new();
        let mut process: Vec<Item> = Vec::new();
        for a0 in e.args() {
            let (a, ex): (Basic, Basic) = if a0.is_pow() {
                let (b, ex) = as_base_exp(a0)?;
                if !(is_true(&ex, Predicate::integer) || is_true(&b, Predicate::positive)) {
                    other.push(a0.clone());
                    continue;
                }
                (b, ex)
            } else {
                (a0.clone(), Basic::one())
            };
            let m = as_f_sign_1(&a)?;
            let Some((g, fn_, si)) = m.filter(|m| is_sin_or_cos(&m.1)) else {
                other.push(if ex == Basic::one() { a } else { pow(&a, &ex)? });
                continue;
            };
            process.push(Item {
                g,
                e: ex,
                f: fn_,
                si,
                a,
            });
        }
        let mut wrapped: Vec<(Basic, Item)> = Vec::new();
        for item in process {
            let key: Basic = Basic::new(
                Head::Tuple,
                vec![
                    item.g.clone(),
                    crate::core::logic::boolean(item.e.is_number_class()),
                    item.e.clone(),
                    item.f.clone(),
                    item.si.clone(),
                    item.a.clone(),
                ],
            );
            opt_set(&mut wrapped, key, item);
        }
        let order: Vec<Basic> = ordered(wrapped.iter().map(|(k, _)| k.clone()).collect());
        let mut queue: std::collections::VecDeque<Item> = order
            .iter()
            .map(|k| opt_get(&wrapped, k).expect("an item").clone())
            .collect();
        let nother: usize = other.len();
        while let Some(a) = queue.pop_front() {
            if let Some(b) = queue.front().cloned() {
                if a.e.is_number_class() && b.e.is_number_class() {
                    if a.f == b.f && a.si != b.si {
                        queue.pop_front();
                        let take: Basic = if value_of(&a.e) <= value_of(&b.e) {
                            a.e.clone()
                        } else {
                            b.e.clone()
                        };
                        if b.e != take {
                            queue.push_front(Item {
                                e: crate::core::add::sub(&b.e, &take)?,
                                ..b.clone()
                            });
                        } else if a.e != take {
                            queue.push_front(Item {
                                e: crate::core::add::sub(&a.e, &take)?,
                                ..a.clone()
                            });
                        }
                        let t: Basic = if is_f(&a.f, Function::Cos) {
                            sin(&arg(&a.f))?
                        } else {
                            cos(&arg(&a.f))?
                        };
                        other.push(pow(&mul(&[neg(&a.g)?, b.g.clone(), pow(&t, &int(2))?])?, &take)?);
                        continue;
                    }
                } else if a.e == b.e && a.f == b.f && a.si != b.si {
                    queue.pop_front();
                    let t: Basic = if is_f(&a.f, Function::Cos) {
                        sin(&arg(&a.f))?
                    } else {
                        cos(&arg(&a.f))?
                    };
                    other.push(pow(&mul(&[neg(&a.g)?, b.g.clone(), pow(&t, &int(2))?])?, &a.e)?);
                    continue;
                }
            }
            other.push(pow(&a.a, &a.e)?);
        }
        if other.len() != nother {
            return mul(&other);
        }
        Ok(e.clone())
    })
}

fn one_plus(x: &Basic) -> R {
    add2(&Basic::one(), x)
}

fn minus_one(x: &Basic) -> R {
    add2(x, &Basic::negative_one())
}

/// Negative even powers of `sin` through `1 + cot**2` (`TR15`).
pub fn tr15(rv: &Basic, max: i64, pow_: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_pow() && is_f(&e.args()[0], Function::Sin)) {
            return Ok(e.clone());
        }
        if let Some(p) = integer_value(&e.args()[1]).filter(|p| p.rem_euclid(2) == 1) {
            return mul2(&tr15(&pow(&e.args()[0], &int(p + 1))?, max, pow_)?, &inv(&e.args()[0])?);
        }
        let ia: Basic = inv(e)?;
        let a: Basic = tr56(&ia, Function::Sin, cot, &one_plus, max, pow_)?;
        Ok(if a != ia { a } else { e.clone() })
    })
}

/// Negative even powers of `cos` through `1 + tan**2` (`TR16`; an odd
/// power goes through `TR15`, as in SymPy).
pub fn tr16(rv: &Basic, max: i64, pow_: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_pow() && is_f(&e.args()[0], Function::Cos)) {
            return Ok(e.clone());
        }
        if let Some(p) = integer_value(&e.args()[1]).filter(|p| p.rem_euclid(2) == 1) {
            return mul2(&tr15(&pow(&e.args()[0], &int(p + 1))?, max, pow_)?, &inv(&e.args()[0])?);
        }
        let ia: Basic = inv(e)?;
        let a: Basic = tr56(&ia, Function::Cos, tan, &one_plus, max, pow_)?;
        Ok(if a != ia { a } else { e.clone() })
    })
}

/// Negative powers of `tan`, `sin`, `cos` as `cot`, `csc`, `sec`
/// (`TR111`).
pub fn tr111(rv: &Basic) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_pow()
            && (is_true(&e.args()[0], Predicate::positive)
                || (is_true(&e.args()[1], Predicate::integer) && is_true(&e.args()[1], Predicate::negative))))
        {
            return Ok(e.clone());
        }
        let (b, ex) = (&e.args()[0], &e.args()[1]);
        if is_f(b, Function::Tan) {
            return pow(&cot(&arg(b))?, &neg(ex)?);
        }
        if is_f(b, Function::Sin) {
            return pow(&csc(&arg(b))?, &neg(ex)?);
        }
        if is_f(b, Function::Cos) {
            return pow(&sec(&arg(b))?, &neg(ex)?);
        }
        Ok(e.clone())
    })
}

/// Even powers of `tan`/`cot` through `sec**2 - 1`/`csc**2 - 1` (`TR22`).
pub fn tr22(rv: &Basic, max: i64, pow_: bool) -> R {
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_pow() && (is_f(&e.args()[0], Function::Cot) || is_f(&e.args()[0], Function::Tan))) {
            return Ok(e.clone());
        }
        let e: Basic = tr56(e, Function::Tan, sec, &minus_one, max, pow_)?;
        tr56(&e, Function::Cot, csc, &minus_one, max, pow_)
    })
}

/// Positive integer powers of `sin`/`cos` as sums of multiple angles
/// (`TRpower`).
pub fn tr_power(rv: &Basic) -> R {
    use crate::functions::combinatorial::factorials::binomial;
    bottom(rv, &|e: &Basic| -> R {
        if !(e.is_pow() && is_sin_or_cos(&e.args()[0])) {
            return Ok(e.clone());
        }
        let (b, n0) = as_base_exp(e)?;
        let Some(n) = integer_value(&n0).filter(|n| *n > 0) else {
            return Ok(e.clone());
        };
        let x: Basic = arg(&b);
        let scale: Basic = pow(&int(2), &int(1 - n))?;
        let bin = |k: i64| binomial(&int(n), &int(k));
        let sign = |k: i64| if k.rem_euclid(2) == 0 { 1 } else { -1 };
        let is_cos: bool = is_f(&b, Function::Cos);
        let mut result: Basic = if n % 2 == 1 && is_cos {
            let terms: Vec<Basic> = (0..(n + 1) / 2)
                .map(|k| mul2(&bin(k)?, &cos(&mul2(&int(n - 2 * k), &x)?)?))
                .collect::<PolyResult<_>>()?;
            mul2(&scale, &add(&terms)?)?
        } else if n % 2 == 1 {
            let terms: Vec<Basic> = (0..(n + 1) / 2)
                .map(|k| mul(&[bin(k)?, int(sign(k)), sin(&mul2(&int(n - 2 * k), &x)?)?]))
                .collect::<PolyResult<_>>()?;
            mul(&[scale, int(sign((n - 1) / 2)), add(&terms)?])?
        } else if is_cos {
            let terms: Vec<Basic> = (0..n / 2)
                .map(|k| mul2(&bin(k)?, &cos(&mul2(&int(n - 2 * k), &x)?)?))
                .collect::<PolyResult<_>>()?;
            mul2(&scale, &add(&terms)?)?
        } else {
            let terms: Vec<Basic> = (0..n / 2)
                .map(|k| mul(&[bin(k)?, int(sign(k)), cos(&mul2(&int(n - 2 * k), &x)?)?]))
                .collect::<PolyResult<_>>()?;
            mul(&[scale, int(sign(n / 2)), add(&terms)?])?
        };
        if n % 2 == 0 {
            result = add2(&result, &mul2(&pow(&int(2), &int(-n))?, &bin(n / 2)?)?)?;
        }
        Ok(result)
    })
}

/// The number of trigonometric functions (`L`).
pub fn l_count(rv: &Basic) -> i64 {
    rv.preorder().iter().filter(|n| is_trig(n)).count() as i64
}

/// `TR8` of the expanded `TRpower` (`sincos_to_sum`).
pub fn sincos_to_sum(e: &Basic) -> R {
    if !has_function(e, &[Function::Sin, Function::Cos]) {
        return Ok(e.clone());
    }
    tr8(&crate::core::function::expand_mul(&tr_power(e)?, true)?, true)
}

// -- the strategies

/// A rule tree: a rule, a chain of trees, or the best of alternatives
/// (`Chain`, `Minimize`).
pub enum Tree {
    Rule(Rule),
    Chain(Vec<Tree>),
    Minimize(Vec<Tree>),
}

/// The measure `fu` minimises.
pub type Measure = fn(&Basic) -> PolyResult<(i64, i64)>;

/// The tree applied greedily (`greedy`).
pub fn apply_greedy(tree: &Tree, objective: Measure, e: &Basic) -> R {
    match tree {
        Tree::Rule(r) => r(e),
        Tree::Chain(rules) => {
            let mut e: Basic = e.clone();
            for r in rules {
                e = apply_greedy(r, objective, &e)?;
            }
            Ok(e)
        }
        Tree::Minimize(rules) => {
            let mut best: Option<((i64, i64), Basic)> = None;
            for r in rules {
                let candidate: Basic = apply_greedy(r, objective, e)?;
                let key = objective(&candidate)?;
                if best.as_ref().is_none_or_lt(&key) {
                    best = Some((key, candidate));
                }
            }
            Ok(best.expect("alternatives").1)
        }
    }
}

trait FirstMinimum {
    fn is_none_or_lt(&self, key: &(i64, i64)) -> bool;
}

impl FirstMinimum for Option<&((i64, i64), Basic)> {
    fn is_none_or_lt(&self, key: &(i64, i64)) -> bool {
        match self {
            None => true,
            Some((k, _)) => key < k,
        }
    }
}

fn identity(e: &Basic) -> R {
    Ok(e.clone())
}
fn r_tr5(e: &Basic) -> R {
    tr5(e, 4, false)
}
fn r_tr6(e: &Basic) -> R {
    tr6(e, 4, false)
}
fn r_tr8(e: &Basic) -> R {
    tr8(e, true)
}
fn r_tr10(e: &Basic) -> R {
    tr10(e, true)
}
fn r_tr11(e: &Basic) -> R {
    tr11(e, None)
}
fn r_tr12(e: &Basic) -> R {
    tr12(e, true)
}

fn ctr1() -> Tree {
    Tree::Minimize(vec![
        Tree::Chain(vec![Tree::Rule(r_tr5), Tree::Rule(tr0)]),
        Tree::Chain(vec![Tree::Rule(r_tr6), Tree::Rule(tr0)]),
        Tree::Rule(identity),
    ])
}

fn ctr2() -> Tree {
    Tree::Chain(vec![
        Tree::Rule(r_tr11),
        Tree::Minimize(vec![
            Tree::Chain(vec![Tree::Rule(r_tr5), Tree::Rule(tr0)]),
            Tree::Chain(vec![Tree::Rule(r_tr6), Tree::Rule(tr0)]),
            Tree::Rule(tr0),
        ]),
    ])
}

fn ctr3() -> Tree {
    Tree::Minimize(vec![
        Tree::Chain(vec![Tree::Rule(tr_morrie), Tree::Rule(r_tr8), Tree::Rule(tr0)]),
        Tree::Chain(vec![
            Tree::Rule(tr_morrie),
            Tree::Rule(r_tr8),
            Tree::Rule(tr10i),
            Tree::Rule(tr0),
        ]),
        Tree::Rule(identity),
    ])
}

fn ctr4() -> Tree {
    Tree::Minimize(vec![
        Tree::Chain(vec![Tree::Rule(tr4), Tree::Rule(tr10i)]),
        Tree::Rule(identity),
    ])
}

fn rl1() -> Tree {
    Tree::Chain(vec![
        Tree::Rule(tr4),
        Tree::Rule(tr3),
        Tree::Rule(tr4),
        Tree::Rule(r_tr12),
        Tree::Rule(tr4),
        Tree::Rule(tr13),
        Tree::Rule(tr4),
        Tree::Rule(tr0),
    ])
}

fn rl2() -> Tree {
    Tree::Minimize(vec![
        Tree::Chain(vec![
            Tree::Rule(tr4),
            Tree::Rule(tr3),
            Tree::Rule(r_tr10),
            Tree::Rule(tr4),
            Tree::Rule(tr3),
            Tree::Rule(r_tr11),
        ]),
        Tree::Chain(vec![
            Tree::Rule(r_tr5),
            Tree::Rule(tr7),
            Tree::Rule(r_tr11),
            Tree::Rule(tr4),
        ]),
        Tree::Chain(vec![
            ctr3(),
            ctr1(),
            Tree::Rule(tr9),
            ctr2(),
            Tree::Rule(tr4),
            Tree::Rule(tr9),
            Tree::Rule(tr9),
            ctr4(),
        ]),
        Tree::Rule(identity),
    ])
}

/// The number of trigonometric functions, then of operations
/// (`_fu_measure`).
pub fn fu_measure(e: &Basic) -> PolyResult<(i64, i64)> {
    Ok((l_count(e), crate::core::function::count_ops(e)?))
}

fn first_min(candidates: Vec<Basic>, measure: Measure) -> R {
    let mut best: Option<((i64, i64), Basic)> = None;
    for c in candidates {
        let key = measure(&c)?;
        if best.as_ref().is_none_or_lt(&key) {
            best = Some((key, c));
        }
    }
    Ok(best.expect("candidates").1)
}

/// The trigonometric simplification of Fu, Zhong and Zeng (`fu`).
pub fn fu(rv: &Basic, measure: Measure) -> R {
    let was: Basic = rv.clone();
    let mut e: Basic = tr1(&was)?;
    if has_function(&e, &[Function::Tan, Function::Cot]) {
        let e1: Basic = apply_greedy(&rl1(), measure, &e)?;
        if measure(&e1)? < measure(&e)? {
            e = e1;
        }
        if has_function(&e, &[Function::Tan, Function::Cot]) {
            e = tr2(&e)?;
        }
    }
    if has_function(&e, &[Function::Sin, Function::Cos]) {
        let e1: Basic = apply_greedy(&rl2(), measure, &e)?;
        let e2: Basic = tr8(&tr_morrie(&e1)?, true)?;
        e = first_min(vec![was, e, e1, e2], measure)?;
    }
    first_min(vec![tr2i(&e, false)?, e], measure)
}
