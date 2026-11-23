//! Port of the first part of `sympy_extras/functions/elementary/_series.py`:
//! the Taylor terms, series and leading terms of `sin`, `cos`, `tan`,
//! `cot`, `sec`, `csc` and of their hyperbolic counterparts, as the methods
//! of the classes call them (SymPy's `trigonometric.py` and
//! `hyperbolic.py`). A leading term at an infinite argument is an
//! accumulation bound.

use super::basic::{Basic, Function, Head};
use super::series::{application_nseries, as_leading_term, eval_as_leading_term, eval_nseries, limit_lite};
use crate::core::assumptions::Predicate;
use crate::core::expr::is;
use crate::error::{pole_error, PolyResult};
use crate::numbers::*;

type R = PolyResult<Basic>;

fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn big(n: Int) -> Basic {
    Basic::integer(n)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn factorial(n: i64) -> R {
    crate::functions::combinatorial::factorials::factorial(&int(n))
}
fn bernoulli(n: i64) -> R {
    crate::functions::combinatorial::numbers::bernoulli(&int(n), None)
}
fn euler(n: i64) -> R {
    crate::functions::combinatorial::numbers::euler(&int(n), None)
}
fn two_pow(n: i64) -> Int {
    Int::from(2u8).pow(n as usize)
}
fn sign_pow(k: i64) -> Int {
    if k % 2 == 0 {
        Int::ONE
    } else {
        -Int::ONE
    }
}

/// Whether the class is one of this module's.
pub fn is_trig(f: Function) -> bool {
    matches!(
        f,
        Function::Sin
            | Function::Cos
            | Function::Tan
            | Function::Cot
            | Function::Sec
            | Function::Csc
            | Function::Sinh
            | Function::Cosh
            | Function::Tanh
            | Function::Coth
            | Function::Sech
            | Function::Csch
    )
}

/// The `n`-th Taylor term at zero of the function in the argument `x`
/// (the `taylor_term` of the classes).
pub fn taylor_term(f: Function, n: i64, x: &Basic) -> R {
    let xn = || pow(x, &int(n));
    match f {
        Function::Sin => {
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            div(&mul(&[big(sign_pow(n / 2)), xn()?])?, &factorial(n)?)
        }
        Function::Cos => {
            if n < 0 || n % 2 == 1 {
                return Ok(Basic::zero());
            }
            div(&mul(&[big(sign_pow(n / 2)), xn()?])?, &factorial(n)?)
        }
        Function::Tan => {
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            let (a, b) = ((n - 1) / 2, two_pow(n + 1));
            let c: Int = sign_pow(a) * b.clone() * (b - Int::ONE);
            mul(&[big(c), div(&bernoulli(n + 1)?, &factorial(n + 1)?)?, xn()?])
        }
        Function::Cot => {
            if n == 0 {
                return pow(x, &Basic::negative_one());
            }
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            let c: Int = sign_pow((n + 1) / 2) * two_pow(n + 1);
            mul(&[big(c), div(&bernoulli(n + 1)?, &factorial(n + 1)?)?, xn()?])
        }
        Function::Sec => {
            if n < 0 || n % 2 == 1 {
                return Ok(Basic::zero());
            }
            let k: i64 = n / 2;
            mul(&[
                big(sign_pow(k)),
                div(&euler(2 * k)?, &factorial(2 * k)?)?,
                pow(x, &int(2 * k))?,
            ])
        }
        Function::Csc => {
            if n == 0 {
                return pow(x, &Basic::negative_one());
            }
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            let k: i64 = n / 2 + 1;
            let c: Int = sign_pow(k - 1) * Int::from(2u8) * (two_pow(2 * k - 1) - Int::ONE);
            div(
                &mul(&[big(c), bernoulli(2 * k)?, pow(x, &int(2 * k - 1))?])?,
                &factorial(2 * k)?,
            )
        }
        Function::Sinh => {
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            div(&xn()?, &factorial(n)?)
        }
        Function::Cosh => {
            if n < 0 || n % 2 == 1 {
                return Ok(Basic::zero());
            }
            div(&xn()?, &factorial(n)?)
        }
        Function::Tanh => {
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            let a: Int = two_pow(n + 1);
            mul(&[
                big(a.clone() * (a - Int::ONE)),
                div(&bernoulli(n + 1)?, &factorial(n + 1)?)?,
                xn()?,
            ])
        }
        Function::Coth => {
            if n == 0 {
                return pow(x, &Basic::negative_one());
            }
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            mul(&[big(two_pow(n + 1)), div(&bernoulli(n + 1)?, &factorial(n + 1)?)?, xn()?])
        }
        Function::Sech => {
            if n < 0 || n % 2 == 1 {
                return Ok(Basic::zero());
            }
            mul(&[div(&euler(n)?, &factorial(n)?)?, xn()?])
        }
        Function::Csch => {
            if n == 0 {
                return pow(x, &Basic::negative_one());
            }
            if n < 0 || n % 2 == 0 {
                return Ok(Basic::zero());
            }
            let c: Int = Int::from(2u8) * (Int::ONE - two_pow(n));
            mul(&[big(c), div(&bernoulli(n + 1)?, &factorial(n + 1)?)?, xn()?])
        }
        _ => unreachable!("not a trigonometric function"),
    }
}

fn rebuilt(f: Function, arg: &Basic) -> R {
    crate::functions::apply_function(f, std::slice::from_ref(arg))
}

fn rewrite_cos(e: &Basic) -> R {
    crate::core::rewriting::rewrite(e, &[], "cos", &crate::core::rewriting::RewriteHints::default())
}

fn rewrite_tan(e: &Basic) -> R {
    crate::core::rewriting::rewrite(e, &[], "tan", &crate::core::rewriting::RewriteHints::default())
}

/// The series of the functions (`_eval_nseries`).
pub fn nseries(e: &Basic, f: Function, x: &Basic, n: i64, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: &Basic = &e.args()[0];
    match f {
        Function::Sin | Function::Cos => {
            // sincos_nseries
            let mut a: Basic = arg.clone();
            if let Some(l) = logx {
                a = subs(&a, &crate::functions::elementary::exponential::log(x)?, l)?;
            }
            let at0: Basic = subs(&a, x, &Basic::zero())?;
            if at0
                .preorder()
                .iter()
                .any(|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity))
            {
                return Err(pole_error(&format!("Cannot expand {} around 0", e)));
            }
            application_nseries(e, x, n, logx, cdir, Some(true))
        }
        Function::Tan => {
            let i: Basic = div(&mul(&[limit_lite(arg, x, &Basic::zero())?, int(2)])?, &Basic::pi())?;
            if i != Basic::zero() && i.is_integer_value() {
                return eval_nseries(&rewrite_cos(e)?, x, n, logx, &Basic::zero());
            }
            application_nseries(e, x, n, logx, &Basic::zero(), Some(true))
        }
        Function::Cot => {
            let i: Basic = div(&limit_lite(arg, x, &Basic::zero())?, &Basic::pi())?;
            if i != Basic::zero() && i.is_integer_value() {
                return eval_nseries(&rewrite_cos(e)?, x, n, logx, &Basic::zero());
            }
            eval_nseries(&rewrite_tan(e)?, x, n, logx, &Basic::zero())
        }
        Function::Sec => eval_nseries(
            &pow(&rebuilt(Function::Cos, arg)?, &Basic::negative_one())?,
            x,
            n,
            logx,
            &Basic::zero(),
        ),
        Function::Csc => eval_nseries(
            &pow(&rebuilt(Function::Sin, arg)?, &Basic::negative_one())?,
            x,
            n,
            logx,
            &Basic::zero(),
        ),
        _ => application_nseries(e, x, n, logx, cdir, Some(true)),
    }
}

fn half_pi() -> R {
    div(&Basic::pi(), &int(2))
}

/// The leading terms (`_eval_as_leading_term`).
pub fn leading_term(e: &Basic, f: Function, x: &Basic, logx: Option<&Basic>, cdir: &Basic) -> R {
    let arg: &Basic = &e.args()[0];
    let pi: Basic = Basic::pi();
    let neg_one: Basic = Basic::negative_one();
    match f {
        Function::Sin => trig_leading_term(
            e,
            f,
            x,
            cdir,
            &div(&subs0(arg, x)?, &pi)?,
            &|k| mul(&[k.clone(), pi.clone()]),
            &|k, lt| mul(&[pow(&neg_one, k)?, lt.clone()]),
        ),
        Function::Csc => trig_leading_term(
            e,
            f,
            x,
            cdir,
            &div(&subs0(arg, x)?, &pi)?,
            &|k| mul(&[k.clone(), pi.clone()]),
            &|k, lt| div(&pow(&neg_one, k)?, lt),
        ),
        Function::Cos | Function::Sec => {
            let k: Basic = div(&add(&[subs0(arg, x)?, half_pi()?])?, &pi)?;
            let shift = |k: &Basic| add(&[mul(&[k.clone(), pi.clone()])?, crate::core::mul::neg(&half_pi()?)?]);
            if f == Function::Cos {
                trig_leading_term(e, f, x, cdir, &k, &shift, &|k, lt| {
                    mul(&[pow(&neg_one, k)?, lt.clone()])
                })
            } else {
                trig_leading_term(e, f, x, cdir, &k, &shift, &|k, lt| div(&pow(&neg_one, k)?, lt))
            }
        }
        Function::Tan | Function::Cot => {
            let k: Basic = div(&mul(&[int(2), subs0(arg, x)?])?, &pi)?;
            let shift = |k: &Basic| mul(&[k.clone(), half_pi()?]);
            if f == Function::Tan {
                trig_leading_term(e, f, x, cdir, &k, &shift, &|k, lt| {
                    if yes(k, Predicate::even) {
                        Ok(lt.clone())
                    } else {
                        crate::core::mul::neg(&pow(lt, &Basic::negative_one())?)
                    }
                })
            } else {
                trig_leading_term(e, f, x, cdir, &k, &shift, &|k, lt| {
                    if yes(k, Predicate::even) {
                        pow(lt, &Basic::negative_one())
                    } else {
                        crate::core::mul::neg(lt)
                    }
                })
            }
        }
        Function::Sinh | Function::Cosh => {
            let a: Basic = as_leading_term(arg, std::slice::from_ref(x), logx, cdir)?;
            let mut a0: Basic = subs(&a, x, &Basic::zero())?;
            if matches!(a0.head(), Head::NaN) {
                let d: &str = if yes(cdir, Predicate::negative) { "-" } else { "+" };
                a0 = crate::core::limits::limit(&a, x, &Basic::zero(), d)?;
            }
            if yes(&a0, Predicate::zero) {
                return Ok(if f == Function::Sinh { a } else { Basic::one() });
            }
            if yes(&a0, Predicate::finite) {
                return rebuilt(f, &a0);
            }
            Ok(e.clone())
        }
        Function::Tanh | Function::Coth => {
            let a: Basic = as_leading_term(arg, std::slice::from_ref(x), None, &Basic::zero())?;
            let o: Basic = crate::core::order::order_at_zero(&Basic::one(), x)?;
            if a.free_symbols().contains(x) && crate::core::order::contains(&o, &a)? == Some(true) {
                return if f == Function::Coth {
                    pow(&a, &Basic::negative_one())
                } else {
                    Ok(a)
                };
            }
            rebuilt(f, &a)
        }
        Function::Sech => eval_as_leading_term(
            &pow(&rebuilt(Function::Cosh, arg)?, &Basic::negative_one())?,
            x,
            logx,
            cdir,
        ),
        Function::Csch => eval_as_leading_term(
            &pow(&rebuilt(Function::Sinh, arg)?, &Basic::negative_one())?,
            x,
            logx,
            cdir,
        ),
        _ => unreachable!("not a trigonometric function"),
    }
}

/// `arg.subs(x, 0).cancel()`.
fn subs0(arg: &Basic, x: &Basic) -> R {
    crate::core::series::value_at_zero(arg, x)
}

/// `trig_leading_term`: `k` the multiple of the period at the value of
/// the argument, `shift(k)` the point, `at_multiple(k, lt)` the leading
/// term from the one of `arg - shift(k)`.
fn trig_leading_term(
    e: &Basic,
    f: Function,
    x: &Basic,
    cdir: &Basic,
    k: &Basic,
    shift: &dyn Fn(&Basic) -> R,
    at_multiple: &dyn Fn(&Basic, &Basic) -> R,
) -> R {
    let arg: &Basic = &e.args()[0];
    let mut x0: Basic = subs0(arg, x)?;
    if yes(k, Predicate::integer) {
        let l: Basic = as_leading_term(
            &crate::core::add::sub(arg, &shift(k)?)?,
            std::slice::from_ref(x),
            None,
            &Basic::zero(),
        )?;
        return at_multiple(k, &l);
    }
    if matches!(x0.head(), Head::ComplexInfinity) {
        let d: &str = if yes(&crate::functions::elementary::complexes::re(cdir)?, Predicate::negative) {
            "-"
        } else {
            "+"
        };
        x0 = crate::core::limits::limit(arg, x, &Basic::zero(), d)?;
    }
    if matches!(x0.head(), Head::Infinity | Head::NegativeInfinity) {
        let (lo, hi) = if matches!(f, Function::Sin | Function::Cos) {
            (Basic::negative_one(), Basic::one())
        } else {
            (Basic::negative_oo(), Basic::oo())
        };
        return crate::core::accumbounds::accum_bounds(&lo, &hi);
    }
    if yes(&x0, Predicate::finite) {
        return rebuilt(f, &x0);
    }
    Ok(e.clone())
}
