//! Port of the second part of `sympy_extras/functions/elementary/_series.py`:
//! the Taylor terms, series and leading terms of the inverse
//! trigonometric and hyperbolic functions (SymPy's `trigonometric.py` and
//! `hyperbolic.py`): the branch points by the rewrites in logarithms or by
//! the series in `t = sqrt(arg - a)`, the points on the branch cuts by the
//! direction `dir` of the argument. The operators of Python (`a - b`,
//! `-a`, `a*b`, `a/b`) are mirrored one at a time, since the order of the
//! two-argument products decides the printed forms.

#![allow(clippy::too_many_arguments)]

use super::basic::{Basic, Function, Head};
use super::series::{application_nseries, as_leading_term, dir, eval_as_leading_term, eval_nseries, nseries};
use crate::core::assumptions::Predicate;
use crate::core::expr::is;
use crate::core::order::{order, order_at_zero, remove_o};
use crate::error::PolyResult;

type R = PolyResult<Basic>;
type F = Function;

fn add(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn mul(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn i() -> Basic {
    Basic::imaginary_unit()
}
fn pi() -> Basic {
    Basic::pi()
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn sqrt(e: &Basic) -> R {
    pow(e, &Basic::half())
}
fn half_pi() -> R {
    mul(&pi(), &pow(&int(2), &Basic::negative_one())?)
}
fn factorial(n: i64) -> R {
    crate::functions::combinatorial::factorials::factorial(&int(n))
}
fn rf_half(k: i64) -> R {
    crate::functions::combinatorial::factorials::rf(&Basic::half(), &int(k))
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn im(e: &Basic) -> R {
    crate::functions::elementary::complexes::im(e)
}
fn re(e: &Basic) -> R {
    crate::functions::elementary::complexes::re(e)
}
fn lt(e: &Basic, x: &Basic) -> R {
    as_leading_term(e, std::slice::from_ref(x), None, &Basic::zero())
}
fn rebuilt(f: F, arg: &Basic) -> R {
    crate::functions::apply_function(f, std::slice::from_ref(arg))
}
fn is_number(e: &Basic, value: i64) -> bool {
    e.is_number_class() && *e == int(value)
}
fn rewrite_log(e: &Basic) -> R {
    crate::core::rewriting::rewrite(e, &[], "log", &crate::core::rewriting::RewriteHints::default())
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_method(e)
}
fn ndir(arg: &Basic, x: &Basic, cdir: &Basic) -> R {
    let c: Basic = if cdir.is_number_class() && *cdir == Basic::zero() {
        Basic::one()
    } else {
        cdir.clone()
    };
    dir(arg, x, &c)
}
fn log_lt(node: &Basic, x: &Basic, logx: Option<&Basic>, cdir: &Basic, expanded: bool) -> R {
    let l: Basic = eval_as_leading_term(&rewrite_log(node)?, x, logx, cdir)?;
    if expanded {
        expand(&l)
    } else {
        Ok(l)
    }
}
fn log_ns(node: &Basic, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    eval_nseries(&rewrite_log(node)?, x, n, logx, cdir)
}
fn generic_ns(node: &Basic, x: &Basic, n: i64, logx: Option<&Basic>) -> R {
    application_nseries(node, x, n, logx, &Basic::zero(), Some(true))
}
fn x0_of(arg: &Basic, x: &Basic) -> R {
    crate::polys::polytools::cancel(&subs(arg, x, &Basic::zero())?, &[])
}

/// Whether the class is one of this module's.
pub fn is_inverse(f: F) -> bool {
    matches!(
        f,
        F::Asin
            | F::Acos
            | F::Atan
            | F::Acot
            | F::Asec
            | F::Acsc
            | F::Asinh
            | F::Acosh
            | F::Atanh
            | F::Acoth
            | F::Asech
            | F::Acsch
    )
}

/// `R` and `F` of the Taylor terms of asec, acsc, asech, acsch.
fn half_rf_term(n: i64) -> PolyResult<(Basic, Basic)> {
    let k: i64 = n / 2;
    let r: Basic = mul(&rf_half(k)?, &int(n))?;
    let fk: Basic = factorial(k)?;
    let fkv: crate::numbers::Int = crate::numbers::numer(fk.as_rational().expect("an integer factorial"));
    let n_ = crate::numbers::Int::from(n);
    let two = crate::numbers::Int::from(2u8);
    let value = ((fkv * n_.clone()) / two.clone() * n_) / two;
    Ok((r, Basic::integer(value)))
}

/// The `n`-th Taylor term (the `taylor_term` of the classes).
pub fn taylor_term(f: F, n: i64, x: &Basic) -> R {
    let odd_zero = n < 0 || n % 2 == 0;
    let even_zero = n < 0 || n % 2 == 1;
    let xn = || pow(x, &int(n));
    match f {
        F::Asin | F::Acos => {
            if f == F::Acos && n == 0 {
                return half_pi();
            }
            if odd_zero {
                return Ok(Basic::zero());
            }
            let k: i64 = (n - 1) / 2;
            let t: Basic = div(&mul(&rf_half(k)?, &xn()?)?, &mul(&factorial(k)?, &int(n))?)?;
            if f == F::Acos {
                mul(&Basic::negative_one(), &t)
            } else {
                Ok(t)
            }
        }
        F::Atan => {
            if odd_zero {
                return Ok(Basic::zero());
            }
            let s: i64 = if ((n - 1) / 2) % 2 == 0 { 1 } else { -1 };
            div(&mul(&int(s), &xn()?)?, &int(n))
        }
        F::Acot => {
            if n == 0 {
                return half_pi();
            }
            if odd_zero {
                return Ok(Basic::zero());
            }
            let s: i64 = if ((n + 1) / 2) % 2 == 0 { 1 } else { -1 };
            div(&mul(&int(s), &xn()?)?, &int(n))
        }
        F::Asec => {
            if n == 0 {
                return mul(&i(), &log(&div(&int(2), x)?)?);
            }
            if even_zero {
                return Ok(Basic::zero());
            }
            let (r, fc) = half_rf_term(n)?;
            div(&mul(&div(&mul(&neg(&i())?, &r)?, &fc)?, &xn()?)?, &int(4))
        }
        F::Acsc => {
            if n == 0 {
                return add(&sub(&half_pi()?, &mul(&i(), &log(&int(2))?)?)?, &mul(&i(), &log(x)?)?);
            }
            if even_zero {
                return Ok(Basic::zero());
            }
            let (r, fc) = half_rf_term(n)?;
            div(&mul(&div(&mul(&i(), &r)?, &fc)?, &xn()?)?, &int(4))
        }
        F::Asinh => {
            if odd_zero {
                return Ok(Basic::zero());
            }
            let k: i64 = (n - 1) / 2;
            let s: i64 = if k % 2 == 0 { 1 } else { -1 };
            div(
                &mul(&div(&mul(&int(s), &rf_half(k)?)?, &factorial(k)?)?, &xn()?)?,
                &int(n),
            )
        }
        F::Acosh => {
            if n == 0 {
                return mul(&i(), &half_pi()?);
            }
            if odd_zero {
                return Ok(Basic::zero());
            }
            let k: i64 = (n - 1) / 2;
            div(
                &mul(&mul(&div(&neg(&rf_half(k)?)?, &factorial(k)?)?, &i())?, &xn()?)?,
                &int(n),
            )
        }
        F::Atanh | F::Acoth => {
            if f == F::Acoth && n == 0 {
                return mul(&neg(&i())?, &half_pi()?);
            }
            if odd_zero {
                return Ok(Basic::zero());
            }
            div(&xn()?, &int(n))
        }
        F::Asech | F::Acsch => {
            if n == 0 {
                return log(&div(&int(2), x)?);
            }
            if even_zero {
                return Ok(Basic::zero());
            }
            let (r, fc) = half_rf_term(n)?;
            let s: i64 = if f == F::Asech || (n / 2 + 1) % 2 == 1 { -1 } else { 1 };
            div(&mul(&div(&mul(&int(s), &r)?, &fc)?, &xn()?)?, &int(4))
        }
        _ => unreachable!("not an inverse function"),
    }
}

/// The series at a branch point (`branch_point_nseries`): `at(t)` the
/// argument in `t`, `arg1` the vanishing quantity, `not_meromorphic` the
/// value when it has no Laurent series (`None`: not checked).
fn branch_point_nseries(
    _node: &Basic,
    f: F,
    x: &Basic,
    n: i64,
    logx: Option<&Basic>,
    at: &dyn Fn(&Basic) -> R,
    arg1: &Basic,
    not_meromorphic: Option<Basic>,
) -> R {
    let t: Basic = crate::core::symbol::Dummy::new("t", &[(Predicate::positive, true)]).0;
    let ser: Basic = nseries(&rewrite_log(&rebuilt(f, &at(&t)?)?)?, &t, 2 * n, None, &Basic::zero())?;
    let fl: Basic = lt(arg1, x)?;
    let g: Basic = div(&sub(arg1, &fl)?, &fl)?;
    if let Some(value) = not_meromorphic {
        if crate::core::meromorphic::is_meromorphic(&g, x, &Basic::zero())? != Some(true) {
            return if n == 0 {
                order(&Basic::one(), &[])
            } else {
                add(&value, &order(&sqrt(x)?, &[])?)
            };
        }
    }
    let res1: Basic = eval_nseries(&sqrt(&add(&Basic::one(), &g)?)?, x, n, logx, &Basic::zero())?;
    let res: Basic = expand(&mul(&remove_o(&res1)?, &sqrt(&fl)?)?)?;
    let body: Basic = crate::simplify::powsimp::powsimp_default(&expand(&subs(&remove_o(&ser)?, &t, &res)?)?)?;
    add(&body, &order_at_zero(&pow(x, &int(n))?, x)?)
}

/// The series of the functions (`_eval_nseries`).
pub fn nseries_of(node: &Basic, f: F, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: Basic = node.args()[0].clone();
    let arg0: Basic = subs(&arg, x, &Basic::zero())?;
    let zoo = |e: &Basic| matches!(e.head(), Head::ComplexInfinity);
    match f {
        F::Asin | F::Acos => {
            let is_acos: bool = f == F::Acos;
            for point in [1i64, -1] {
                if !is_number(&arg0, point) {
                    continue;
                }
                let at = |t: &Basic| add(&int(point), &mul(&int(-point), &pow(t, &int(2))?)?);
                let arg1: Basic = add(&Basic::one(), &mul(&int(-point), &arg)?)?;
                let value: Basic = if is_acos {
                    if point == 1 {
                        Basic::zero()
                    } else {
                        pi()
                    }
                } else {
                    mul(&int(point), &half_pi()?)?
                };
                return branch_point_nseries(node, f, x, n, logx, &at, &arg1, Some(value));
            }
            let res: Basic = generic_ns(node, x, n, logx)?;
            if zoo(&arg0) {
                return Ok(res);
            }
            if yes(
                &add(&Basic::one(), &mul(&Basic::negative_one(), &pow(&arg0, &int(2))?)?)?,
                Predicate::negative,
            ) {
                let nd: Basic = ndir(&arg, x, cdir)?;
                let imd: Basic = im(&nd)?;
                if yes(&imd, Predicate::negative) {
                    if yes(&arg0, Predicate::negative) {
                        return if is_acos {
                            add(&mul(&int(2), &pi())?, &mul(&Basic::negative_one(), &res)?)
                        } else {
                            add(
                                &mul(&Basic::negative_one(), &pi())?,
                                &mul(&Basic::negative_one(), &res)?,
                            )
                        };
                    }
                } else if yes(&imd, Predicate::positive) {
                    if yes(&arg0, Predicate::positive) {
                        return if is_acos {
                            mul(&Basic::negative_one(), &res)
                        } else {
                            add(&pi(), &mul(&Basic::negative_one(), &res)?)
                        };
                    }
                } else {
                    return log_ns(node, x, n, logx, cdir);
                }
            }
            Ok(res)
        }
        F::Atan | F::Acot => {
            if arg0 == i() || arg0 == neg(&i())? {
                return log_ns(node, x, n, logx, cdir);
            }
            let res: Basic = generic_ns(node, x, n, logx)?;
            if f == F::Atan {
                let nd: Basic = ndir(&arg, x, cdir)?;
                if zoo(&arg0) {
                    if super::series::less(&Basic::zero(), &re(&nd)?)? {
                        return add(&res, &mul(&Basic::negative_one(), &pi())?);
                    }
                    return Ok(res);
                }
                if yes(&add(&Basic::one(), &pow(&arg0, &int(2))?)?, Predicate::negative) {
                    let red: Basic = re(&nd)?;
                    if yes(&red, Predicate::negative) {
                        if yes(&im(&arg0)?, Predicate::positive) {
                            return add(&res, &mul(&Basic::negative_one(), &pi())?);
                        }
                    } else if yes(&red, Predicate::positive) {
                        if yes(&im(&arg0)?, Predicate::negative) {
                            return add(&res, &pi());
                        }
                    } else {
                        return log_ns(node, x, n, logx, cdir);
                    }
                }
                return Ok(res);
            }
            if zoo(&arg0) {
                return Ok(res);
            }
            let nd: Basic = ndir(&arg, x, cdir)?;
            if yes(&arg0, Predicate::zero) {
                if super::series::less(&re(&nd)?, &Basic::zero())? {
                    return add(&res, &mul(&Basic::negative_one(), &pi())?);
                }
                return Ok(res);
            }
            if yes(&arg0, Predicate::imaginary) && yes(&add(&Basic::one(), &pow(&arg0, &int(2))?)?, Predicate::positive)
            {
                let red: Basic = re(&nd)?;
                if yes(&red, Predicate::positive) {
                    if yes(&im(&arg0)?, Predicate::positive) {
                        return add(&res, &pi());
                    }
                } else if yes(&red, Predicate::negative) {
                    if yes(&im(&arg0)?, Predicate::negative) {
                        return add(&res, &mul(&Basic::negative_one(), &pi())?);
                    }
                } else {
                    return log_ns(node, x, n, logx, cdir);
                }
            }
            Ok(res)
        }
        F::Asec | F::Acsc => {
            if is_number(&arg0, 1) {
                let at = |t: &Basic| add(&Basic::one(), &pow(t, &int(2))?);
                return branch_point_nseries(node, f, x, n, logx, &at, &sub(&arg, &Basic::one())?, None);
            }
            if is_number(&arg0, -1) {
                let at = |t: &Basic| sub(&Basic::negative_one(), &pow(t, &int(2))?);
                return branch_point_nseries(node, f, x, n, logx, &at, &sub(&Basic::negative_one(), &arg)?, None);
            }
            let res: Basic = generic_ns(node, x, n, logx)?;
            if zoo(&arg0) {
                return Ok(res);
            }
            if yes(&arg0, Predicate::real) && yes(&sub(&Basic::one(), &pow(&arg0, &int(2))?)?, Predicate::positive) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                if yes(&imd, Predicate::negative) {
                    if yes(&arg0, Predicate::positive) {
                        return if f == F::Asec { neg(&res) } else { sub(&pi(), &res) };
                    }
                } else if yes(&imd, Predicate::positive) {
                    if yes(&arg0, Predicate::negative) {
                        return if f == F::Asec {
                            sub(&mul(&int(2), &pi())?, &res)
                        } else {
                            sub(&neg(&pi())?, &res)
                        };
                    }
                } else {
                    return log_ns(node, x, n, logx, cdir);
                }
            }
            Ok(res)
        }
        F::Asinh => {
            if arg0 == i() || arg0 == neg(&i())? {
                return log_ns(node, x, n, logx, cdir);
            }
            let res: Basic = generic_ns(node, x, n, logx)?;
            if zoo(&arg0) {
                return Ok(res);
            }
            if yes(&add(&Basic::one(), &pow(&arg0, &int(2))?)?, Predicate::negative) {
                let red: Basic = re(&ndir(&arg, x, cdir)?)?;
                let ipi: Basic = mul(&i(), &pi())?;
                if yes(&red, Predicate::positive) {
                    if yes(&im(&arg0)?, Predicate::negative) {
                        return sub(&neg(&res)?, &ipi);
                    }
                } else if yes(&red, Predicate::negative) {
                    if yes(&im(&arg0)?, Predicate::positive) {
                        return add(&neg(&res)?, &ipi);
                    }
                } else {
                    return log_ns(node, x, n, logx, cdir);
                }
            }
            Ok(res)
        }
        F::Acosh | F::Atanh | F::Acoth => {
            if arg0 == Basic::one() || arg0 == Basic::negative_one() {
                return log_ns(node, x, n, logx, cdir);
            }
            let res: Basic = generic_ns(node, x, n, logx)?;
            if zoo(&arg0) {
                return Ok(res);
            }
            let ipi: Basic = mul(&i(), &pi())?;
            match f {
                F::Acosh => {
                    if yes(&sub(&arg0, &Basic::one())?, Predicate::negative) {
                        let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                        if yes(&imd, Predicate::negative) {
                            if yes(&add(&arg0, &Basic::one())?, Predicate::negative) {
                                return sub(&res, &mul(&mul(&int(2), &i())?, &pi())?);
                            }
                            return neg(&res);
                        } else if !yes(&imd, Predicate::positive) {
                            return log_ns(node, x, n, logx, cdir);
                        }
                    }
                }
                F::Atanh => {
                    if yes(&sub(&Basic::one(), &pow(&arg0, &int(2))?)?, Predicate::negative) {
                        let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                        if yes(&imd, Predicate::negative) {
                            if yes(&arg0, Predicate::negative) {
                                return sub(&res, &ipi);
                            }
                        } else if yes(&imd, Predicate::positive) {
                            if yes(&arg0, Predicate::positive) {
                                return add(&res, &ipi);
                            }
                        } else {
                            return log_ns(node, x, n, logx, cdir);
                        }
                    }
                }
                _ => {
                    if yes(&arg0, Predicate::real)
                        && yes(&sub(&Basic::one(), &pow(&arg0, &int(2))?)?, Predicate::positive)
                    {
                        let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                        if yes(&imd, Predicate::negative) {
                            if yes(&arg0, Predicate::positive) {
                                return add(&res, &ipi);
                            }
                        } else if yes(&imd, Predicate::positive) {
                            if yes(&arg0, Predicate::negative) {
                                return sub(&res, &ipi);
                            }
                        } else {
                            return log_ns(node, x, n, logx, cdir);
                        }
                    }
                }
            }
            Ok(res)
        }
        F::Asech => {
            if is_number(&arg0, 1) {
                let at = |t: &Basic| sub(&Basic::one(), &pow(t, &int(2))?);
                return branch_point_nseries(
                    node,
                    f,
                    x,
                    n,
                    logx,
                    &at,
                    &sub(&Basic::one(), &arg)?,
                    Some(Basic::zero()),
                );
            }
            if is_number(&arg0, -1) {
                let at = |t: &Basic| add(&Basic::negative_one(), &pow(t, &int(2))?);
                return branch_point_nseries(
                    node,
                    f,
                    x,
                    n,
                    logx,
                    &at,
                    &add(&Basic::one(), &arg)?,
                    Some(mul(&i(), &pi())?),
                );
            }
            let res: Basic = generic_ns(node, x, n, logx)?;
            if zoo(&arg0) {
                return Ok(res);
            }
            if yes(&arg0, Predicate::negative) || yes(&sub(&Basic::one(), &arg0)?, Predicate::negative) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                if yes(&imd, Predicate::positive) {
                    if yes(&arg0, Predicate::positive) || yes(&add(&arg0, &Basic::one())?, Predicate::negative) {
                        return neg(&res);
                    }
                    return sub(&res, &mul(&mul(&int(2), &i())?, &pi())?);
                } else if !yes(&imd, Predicate::negative) {
                    return log_ns(node, x, n, logx, cdir);
                }
            }
            Ok(res)
        }
        F::Acsch => {
            if arg0 == i() {
                let at = |t: &Basic| add(&i(), &pow(t, &int(2))?);
                return branch_point_nseries(
                    node,
                    f,
                    x,
                    n,
                    logx,
                    &at,
                    &add(&neg(&i())?, &arg)?,
                    Some(mul(&neg(&i())?, &half_pi()?)?),
                );
            }
            if arg0 == neg(&i())? {
                let at = |t: &Basic| add(&neg(&i())?, &pow(t, &int(2))?);
                return branch_point_nseries(
                    node,
                    f,
                    x,
                    n,
                    logx,
                    &at,
                    &add(&i(), &arg)?,
                    Some(mul(&i(), &half_pi()?)?),
                );
            }
            let res: Basic = generic_ns(node, x, n, logx)?;
            if zoo(&arg0) {
                return Ok(res);
            }
            if yes(&arg0, Predicate::imaginary) && yes(&add(&Basic::one(), &pow(&arg0, &int(2))?)?, Predicate::positive)
            {
                let red: Basic = re(&ndir(&arg, x, cdir)?)?;
                let ipi: Basic = mul(&i(), &pi())?;
                if yes(&red, Predicate::positive) {
                    if yes(&im(&arg0)?, Predicate::positive) {
                        return sub(&neg(&res)?, &ipi);
                    }
                } else if yes(&red, Predicate::negative) {
                    if yes(&im(&arg0)?, Predicate::negative) {
                        return add(&neg(&res)?, &ipi);
                    }
                } else {
                    return log_ns(node, x, n, logx, cdir);
                }
            }
            Ok(res)
        }
        _ => unreachable!("not an inverse function"),
    }
}

fn nan_leading_term(node: &Basic, f: F, x: &Basic) -> R {
    let e: Basic = rebuilt(f, &lt(&node.args()[0], x)?)?;
    Ok(if yes(&e, Predicate::finite) { e } else { node.clone() })
}

/// The leading terms (`_eval_as_leading_term`).
pub fn leading_term(node: &Basic, f: F, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: Basic = node.args()[0].clone();
    let x0: Basic = x0_of(&arg, x)?;
    let nan: bool = matches!(x0.head(), Head::NaN);
    let zoo: bool = matches!(x0.head(), Head::ComplexInfinity);
    let fx0 = || rebuilt(f, &x0);
    let one: Basic = Basic::one();
    let m1: Basic = Basic::negative_one();
    let in_ = |vals: &[Basic]| vals.contains(&x0);
    match f {
        F::Asin | F::Acos => {
            let is_acos: bool = f == F::Acos;
            if nan {
                return rebuilt(f, &lt(&arg, x)?);
            }
            if !is_acos && yes(&x0, Predicate::zero) {
                return lt(&arg, x);
            }
            if is_acos && is_number(&x0, 1) {
                return mul(&sqrt(&int(2))?, &sqrt(&lt(&add(&one, &mul(&m1, &arg)?)?, x)?)?);
            }
            let branch: Vec<Basic> = if is_acos {
                vec![m1.clone(), Basic::zoo()]
            } else {
                vec![m1.clone(), one.clone(), Basic::zoo()]
            };
            if in_(&branch) {
                return log_lt(node, x, logx, cdir, !is_acos);
            }
            if yes(&add(&one, &mul(&m1, &pow(&x0, &int(2))?)?)?, Predicate::negative) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                if yes(&imd, Predicate::negative) {
                    if yes(&x0, Predicate::negative) {
                        return if is_acos {
                            add(&mul(&int(2), &pi())?, &mul(&m1, &fx0()?)?)
                        } else {
                            add(&mul(&m1, &pi())?, &mul(&m1, &fx0()?)?)
                        };
                    }
                } else if yes(&imd, Predicate::positive) {
                    if yes(&x0, Predicate::positive) {
                        return if is_acos {
                            mul(&m1, &fx0()?)
                        } else {
                            add(&pi(), &mul(&m1, &fx0()?)?)
                        };
                    }
                } else {
                    return log_lt(node, x, logx, cdir, true);
                }
            }
            fx0()
        }
        F::Atan => {
            if nan {
                return rebuilt(f, &lt(&arg, x)?);
            }
            if yes(&x0, Predicate::zero) {
                return lt(&arg, x);
            }
            if in_(&[mul(&m1, &i())?, i(), Basic::zoo()]) {
                return log_lt(node, x, logx, cdir, true);
            }
            if yes(&add(&one, &pow(&x0, &int(2))?)?, Predicate::negative) {
                let red: Basic = re(&ndir(&arg, x, cdir)?)?;
                if yes(&red, Predicate::negative) {
                    if yes(&im(&x0)?, Predicate::positive) {
                        return add(&fx0()?, &mul(&m1, &pi())?);
                    }
                } else if yes(&red, Predicate::positive) {
                    if yes(&im(&x0)?, Predicate::negative) {
                        return add(&fx0()?, &pi());
                    }
                } else {
                    return log_lt(node, x, logx, cdir, true);
                }
            }
            fx0()
        }
        F::Acot => {
            if nan {
                return rebuilt(f, &lt(&arg, x)?);
            }
            if zoo {
                return lt(&pow(&arg, &m1)?, x);
            }
            if in_(&[mul(&m1, &i())?, i(), Basic::zero()]) {
                return log_lt(node, x, logx, cdir, true);
            }
            if yes(&x0, Predicate::imaginary) && yes(&add(&one, &pow(&x0, &int(2))?)?, Predicate::positive) {
                let red: Basic = re(&ndir(&arg, x, cdir)?)?;
                if yes(&red, Predicate::positive) {
                    if yes(&im(&x0)?, Predicate::positive) {
                        return add(&fx0()?, &pi());
                    }
                } else if yes(&red, Predicate::negative) {
                    if yes(&im(&x0)?, Predicate::negative) {
                        return add(&fx0()?, &mul(&m1, &pi())?);
                    }
                } else {
                    return log_lt(node, x, logx, cdir, true);
                }
            }
            fx0()
        }
        F::Asec | F::Acsc => {
            if nan {
                return rebuilt(f, &lt(&arg, x)?);
            }
            if f == F::Asec {
                if is_number(&x0, 1) {
                    return mul(&sqrt(&int(2))?, &sqrt(&lt(&sub(&arg, &one)?, x)?)?);
                }
                if in_(&[m1.clone(), Basic::zero()]) {
                    return log_lt(node, x, logx, cdir, false);
                }
            } else {
                if in_(&[m1.clone(), one.clone(), Basic::zero()]) {
                    return log_lt(node, x, logx, cdir, true);
                }
                if zoo {
                    return lt(&div(&one, &arg)?, x);
                }
            }
            if yes(&x0, Predicate::real) && yes(&sub(&one, &pow(&x0, &int(2))?)?, Predicate::positive) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                if yes(&imd, Predicate::negative) {
                    if yes(&x0, Predicate::positive) {
                        return if f == F::Asec {
                            neg(&fx0()?)
                        } else {
                            sub(&pi(), &fx0()?)
                        };
                    }
                } else if yes(&imd, Predicate::positive) {
                    if yes(&x0, Predicate::negative) {
                        return if f == F::Asec {
                            sub(&mul(&int(2), &pi())?, &fx0()?)
                        } else {
                            sub(&neg(&pi())?, &fx0()?)
                        };
                    }
                } else {
                    return log_lt(node, x, logx, cdir, true);
                }
            }
            fx0()
        }
        F::Asinh => {
            if yes(&x0, Predicate::zero) {
                return lt(&arg, x);
            }
            if nan {
                return nan_leading_term(node, f, x);
            }
            if in_(&[neg(&i())?, i(), Basic::zoo()]) {
                return log_lt(node, x, logx, cdir, false);
            }
            if yes(&add(&one, &pow(&x0, &int(2))?)?, Predicate::negative) {
                let red: Basic = re(&ndir(&arg, x, cdir)?)?;
                let ipi: Basic = mul(&i(), &pi())?;
                if yes(&red, Predicate::positive) {
                    if yes(&im(&x0)?, Predicate::negative) {
                        return sub(&neg(&fx0()?)?, &ipi);
                    }
                } else if yes(&red, Predicate::negative) {
                    if yes(&im(&x0)?, Predicate::positive) {
                        return add(&neg(&fx0()?)?, &ipi);
                    }
                } else {
                    return log_lt(node, x, logx, cdir, false);
                }
            }
            fx0()
        }
        F::Acosh => {
            if in_(&[m1.clone(), Basic::zero(), one.clone(), Basic::zoo()]) {
                return log_lt(node, x, logx, cdir, false);
            }
            if nan {
                return nan_leading_term(node, f, x);
            }
            if yes(&sub(&x0, &one)?, Predicate::negative) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                if yes(&imd, Predicate::negative) {
                    if yes(&add(&x0, &one)?, Predicate::negative) {
                        return sub(&fx0()?, &mul(&mul(&int(2), &i())?, &pi())?);
                    }
                    return neg(&fx0()?);
                } else if !yes(&imd, Predicate::positive) {
                    return log_lt(node, x, logx, cdir, false);
                }
            }
            fx0()
        }
        F::Atanh => {
            if yes(&x0, Predicate::zero) {
                return lt(&arg, x);
            }
            if nan {
                return nan_leading_term(node, f, x);
            }
            if in_(&[m1.clone(), one.clone(), Basic::zoo()]) {
                return log_lt(node, x, logx, cdir, false);
            }
            if yes(&sub(&one, &pow(&x0, &int(2))?)?, Predicate::negative) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                let ipi: Basic = mul(&i(), &pi())?;
                if yes(&imd, Predicate::negative) {
                    if yes(&x0, Predicate::negative) {
                        return sub(&fx0()?, &ipi);
                    }
                } else if yes(&imd, Predicate::positive) {
                    if yes(&x0, Predicate::positive) {
                        return add(&fx0()?, &ipi);
                    }
                } else {
                    return log_lt(node, x, logx, cdir, false);
                }
            }
            fx0()
        }
        F::Acoth => {
            if zoo {
                return lt(&div(&one, &arg)?, x);
            }
            if nan {
                return nan_leading_term(node, f, x);
            }
            if in_(&[m1.clone(), one.clone(), Basic::zero()]) {
                return log_lt(node, x, logx, cdir, false);
            }
            if yes(&x0, Predicate::real) && yes(&sub(&one, &pow(&x0, &int(2))?)?, Predicate::positive) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                let ipi: Basic = mul(&i(), &pi())?;
                if yes(&imd, Predicate::negative) {
                    if yes(&x0, Predicate::positive) {
                        return add(&fx0()?, &ipi);
                    }
                } else if yes(&imd, Predicate::positive) {
                    if yes(&x0, Predicate::negative) {
                        return sub(&fx0()?, &ipi);
                    }
                } else {
                    return log_lt(node, x, logx, cdir, false);
                }
            }
            fx0()
        }
        F::Asech => {
            if in_(&[m1.clone(), Basic::zero(), one.clone(), Basic::zoo()]) {
                return log_lt(node, x, logx, cdir, false);
            }
            if nan {
                return nan_leading_term(node, f, x);
            }
            if yes(&x0, Predicate::negative) || yes(&sub(&one, &x0)?, Predicate::negative) {
                let imd: Basic = im(&ndir(&arg, x, cdir)?)?;
                if yes(&imd, Predicate::positive) {
                    if yes(&x0, Predicate::positive) || yes(&add(&x0, &one)?, Predicate::negative) {
                        return neg(&fx0()?);
                    }
                    return sub(&fx0()?, &mul(&mul(&int(2), &i())?, &pi())?);
                } else if !yes(&imd, Predicate::negative) {
                    return log_lt(node, x, logx, cdir, false);
                }
            }
            fx0()
        }
        F::Acsch => {
            if in_(&[neg(&i())?, i(), Basic::zero()]) {
                return log_lt(node, x, logx, cdir, false);
            }
            if nan {
                return nan_leading_term(node, f, x);
            }
            if zoo {
                return lt(&div(&one, &arg)?, x);
            }
            if yes(&x0, Predicate::imaginary) && yes(&add(&one, &pow(&x0, &int(2))?)?, Predicate::positive) {
                let red: Basic = re(&ndir(&arg, x, cdir)?)?;
                let ipi: Basic = mul(&i(), &pi())?;
                if yes(&red, Predicate::positive) {
                    if yes(&im(&x0)?, Predicate::positive) {
                        return sub(&neg(&fx0()?)?, &ipi);
                    }
                } else if yes(&red, Predicate::negative) {
                    if yes(&im(&x0)?, Predicate::negative) {
                        return add(&neg(&fx0()?)?, &ipi);
                    }
                } else {
                    return log_lt(node, x, logx, cdir, false);
                }
            }
            fx0()
        }
        _ => unreachable!("not an inverse function"),
    }
}
