//! Port of `sympy_extras/core/limits.py` and `sympy_extras/core/gruntz.py`
//! (SymPy's `limits.py` and `gruntz.py`): the node `Limit(e, z, z0, dir)`
//! (`Head::Limit`, the arguments `(e, z, z0, Symbol(dir))`), its `doit`
//! (the leading term of a meromorphic expression, of the series at the
//! point, then Gruntz's algorithm, then the heuristics), [`limit`], and
//! Gruntz's algorithm: the most rapidly varying subexpressions (`mrv`),
//! their rewriting in one `w` going to zero and the leading term of the
//! series in `w`.
//!
//! The tractable rewriting of the special functions (`gamma`, `erf`,
//! `Ei`, the Bessel functions, ...) waits for their ports: an expression
//! with one of them in the variable is `NotImplemented`. The caches of
//! Python (`cacheit`) are not needed for the results.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use super::basic::{Basic, Function, Head};
use super::series::{as_leading_term, eval_nseries, is_order, leadterm};
use crate::core::assumptions::Predicate;
use crate::core::expr::{factors_of, is};
use crate::core::symbol::{Dummy, Symbol};
use crate::error::{not_ported, pole_error, value_error, PolyError, PolyResult};
use crate::numbers::*;

type R = PolyResult<Basic>;

fn add2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn mul2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn is_number(e: &Basic, value: i64) -> bool {
    e.is_number_class() && *e == Basic::int(value)
}
fn oo() -> Basic {
    Basic::oo()
}
fn sign_fn(e: &Basic) -> R {
    crate::functions::elementary::complexes::sign(e)
}
fn caught(e: &PolyError, value: bool, not_implemented: bool, pole: bool) -> bool {
    match e {
        PolyError::Value(_) => value,
        PolyError::NotImplemented(m) => not_implemented && !m.starts_with("not ported"),
        PolyError::Pole(_) => pole,
        _ => false,
    }
}
fn has_any(e: &Basic, pred: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(pred)
}
fn has_infinities(e: &Basic) -> bool {
    has_any(e, &|n| {
        matches!(
            n.head(),
            Head::NaN | Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity
        )
    })
}
fn is_limit(e: &Basic) -> bool {
    matches!(e.head(), Head::Limit)
}
fn sym(name: &str) -> Basic {
    Symbol::new(name, &[]).0
}

// -- the node

/// The parts `(e, z, z0, dir)` of a `Limit`.
pub fn parts(e: &Basic) -> (Basic, Basic, Basic, String) {
    let a = e.args();
    (a[0].clone(), a[1].clone(), a[2].clone(), a[3].to_string())
}

/// The node `Limit(e, z, z0, dir)` (`Limit.__init__`): the direction `-`
/// at `oo` and `I*oo`, `+` at `-oo` and `-I*oo`.
pub fn limit_node(e: &Basic, z: &Basic, z0: &Basic, dir: &str) -> R {
    let mut dir: String = dir.to_string();
    let i_oo: Basic = mul2(&Basic::imaginary_unit(), &oo())?;
    let i_moo: Basic = mul2(&Basic::imaginary_unit(), &Basic::negative_oo())?;
    if matches!(z0.head(), Head::Infinity) || *z0 == i_oo {
        dir = "-".to_string();
    } else if matches!(z0.head(), Head::NegativeInfinity) || *z0 == i_moo {
        dir = "+".to_string();
    }
    if z0.has(z) {
        return Err(PolyError::NotImplemented(format!(
            "Limits approaching a variable point are not supported ({} -> {})",
            z, z0
        )));
    }
    if dir != "+" && dir != "-" && dir != "+-" {
        return Err(value_error(format!(
            "direction must be one of '+', '-' or '+-', not {}",
            dir
        )));
    }
    Ok(Basic::new(
        Head::Limit,
        vec![e.clone(), z.clone(), z0.clone(), sym(&dir)],
    ))
}

/// The node rebuilt from new arguments (`Limit._evaluated`: not evaluated).
pub fn evaluated(args: &[Basic]) -> R {
    if !args[1].is_symbol() {
        return Err(PolyError::Type("a limit of an expression is expected".to_string()));
    }
    limit_node(&args[0], &args[1], &args[2], &args[3].to_string())
}

/// The free symbols: those of the expression but the variable, and of
/// the point.
pub fn free_symbols(e: &Basic) -> BTreeSet<Basic> {
    let (expr, z, z0, _) = parts(e);
    let mut s: BTreeSet<Basic> = expr.free_symbols();
    s.remove(&z);
    s.extend(z0.free_symbols());
    s
}

/// `Limit(e, z, z0, dir='+')` as Python prints it.
pub fn print(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let (expr, z, z0, dir) = parts(e);
    format!(
        "Limit({}, {}, {}, dir='{}')",
        doprint(&expr),
        doprint(&z),
        doprint(&z0),
        dir
    )
}

/// The limit of `e` as `z` goes to `z0` from `dir` (`+`, `-`, `+-`); the
/// unevaluated `Limit` when it is not found (`limit`).
pub fn limit(e: &Basic, z: &Basic, z0: &Basic, dir: &str) -> R {
    doit(&limit_node(e, z, z0, dir)?, false)
}

fn int_of(e: &Basic) -> PolyResult<Int> {
    match e.as_rational() {
        Some(r) => {
            let (p, q) = (numer(r), Int::from(denom(r)));
            // Python's int(p/q): truncation towards zero
            Ok(&p / &q)
        }
        None => Err(PolyError::Type(format!("an integer is expected, got {}", e))),
    }
}

fn positive_number(e: &Basic) -> PolyResult<bool> {
    is(e, Predicate::extended_positive)
        .ok_or_else(|| PolyError::Type(format!("cannot determine truth value of {} > 0", e)))
}

fn pow_heuristics(e: &Basic, z: &Basic, z0: &Basic) -> PolyResult<Option<Basic>> {
    if !e.is_pow() {
        return Ok(None);
    }
    let (b1, e1) = (&e.args()[0], &e.args()[1]);
    if !b1.has(z) {
        return Ok(Some(exp(&limit(&mul2(e1, &log(b1)?)?, z, z0, "+")?)?));
    }
    let ex_lim: Basic = limit(e1, z, z0, "+")?;
    let base_lim: Basic = limit(b1, z, z0, "+")?;
    if is_number(&base_lim, 1) && matches!(ex_lim.head(), Head::Infinity | Head::NegativeInfinity) {
        return Ok(Some(exp(&limit(&mul2(e1, &sub(b1, &Basic::one())?)?, z, z0, "+")?)?));
    }
    if matches!(base_lim.head(), Head::NegativeInfinity) && matches!(ex_lim.head(), Head::Infinity) {
        return Ok(Some(Basic::zoo()));
    }
    Ok(None)
}

/// `Limit.doit`: the limit, the node itself when it is not found.
pub fn doit(node: &Basic, deep: bool) -> R {
    let (mut e, z, mut z0, mut dir) = parts(node);
    if dir == "+-" {
        let r: Basic = limit(&e, &z, &z0, "+")?;
        let left: Basic = limit(&e, &z, &z0, "-")?;
        if is_limit(&r) && is_limit(&left) && r.args()[0] == left.args()[0] {
            return Ok(node.clone());
        }
        if r == left {
            return Ok(left);
        }
        if yes(&r, Predicate::infinite) && yes(&left, Predicate::infinite) {
            return Ok(Basic::zoo());
        }
        return Err(value_error(format!(
            "The limit does not exist since left hand limit = {} and right hand limit = {}",
            left, r
        )));
    }
    if matches!(z0.head(), Head::ComplexInfinity) {
        return Err(PolyError::NotImplemented(
            "Limits at complex infinity are not implemented".to_string(),
        ));
    }
    if yes(&z0, Predicate::infinite) {
        let mut cdir_inf: Basic = sign_fn(&z0)?;
        cdir_inf = div(&cdir_inf, &crate::functions::elementary::complexes::abs_(&cdir_inf)?)?;
        e = subs(&e, &z, &mul2(&cdir_inf, &z)?)?;
        dir = "-".to_string();
        z0 = oo();
    }
    if deep {
        e = crate::core::function::doit(&e)?;
        z0 = crate::core::function::doit(&z0)?;
    }
    if e == z {
        return Ok(z0);
    }
    if !e.has(&z) {
        return Ok(e);
    }
    if matches!(z0.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if has_infinities(&e) {
        return Ok(node.clone());
    }
    if is_order(&e) {
        let (expr, limits) = crate::core::order::parts(&e);
        return crate::core::order::order(&limit(&expr, &z, &z0, "+")?, &limits);
    }
    let mut cdir: Basic = match dir.as_str() {
        "+" => Basic::one(),
        "-" => Basic::negative_one(),
        _ => Basic::zero(),
    };
    e = set_signs(&e, &z, &z0, &dir)?;
    if crate::core::meromorphic::is_meromorphic(&e, &z, &z0)? == Some(true) {
        let mut mz: Basic = z.clone();
        let newe: Basic = if matches!(z0.head(), Head::Infinity) {
            cdir = neg(&cdir)?;
            subs(&e, &z, &div(&Basic::one(), &z)?)?
        } else {
            mz = shifted_variable(&z, &z0, &dir);
            subs(&e, &z, &add2(&mz, &z0)?)?
        };
        match leadterm(&newe, &mz, None, &cdir) {
            Err(PolyError::Value(_)) => {}
            Err(err) => return Err(err),
            Ok((coeff, ex)) => {
                if positive_number(&ex)? {
                    return Ok(Basic::zero());
                }
                if is_number(&ex, 0) {
                    return Ok(coeff);
                }
                if is_number(&cdir, 1) || (int_of(&ex)? % Int::from(2u8)) == Int::ZERO {
                    return mul2(&oo(), &sign_fn(&coeff)?);
                }
                if is_number(&cdir, -1) {
                    return mul2(&neg(&oo())?, &sign_fn(&coeff)?);
                }
                return Ok(Basic::zoo());
            }
        }
    }
    let (newe, newz): (Basic, Basic) = if matches!(z0.head(), Head::Infinity) {
        if e.is_mul() {
            e = crate::core::exprtools::factor_terms(&e, false, true, false, false)?;
        }
        let mut assumptions: Vec<(Predicate, bool)> = Vec::new();
        for p in [Predicate::positive, Predicate::negative, Predicate::real] {
            if let Some(v) = is(&z, p) {
                assumptions.push((p, v));
            }
        }
        let dummy: Basic = Dummy::new("z", &assumptions).0;
        cdir = neg(&cdir)?;
        (subs(&e, &z, &div(&Basic::one(), &dummy)?)?, dummy)
    } else {
        let newz: Basic = shifted_variable(&z, &z0, &dir);
        (subs(&e, &z, &add2(&newz, &z0)?)?, newz)
    };
    match leadterm(&newe, &newz, None, &cdir) {
        Err(err) if caught(&err, true, true, true) => {
            e = crate::simplify::powsimp::powsimp_default(&e)?;
            if e.is_pow() {
                if let Some(found) = pow_heuristics(&e, &z, &z0)? {
                    return Ok(found);
                }
            }
            match as_leading_term(&newe, std::slice::from_ref(&newz), None, &cdir) {
                Ok(coeff) => {
                    let has_exp = has_any(&coeff, &|n| {
                        matches!(n.head(), Head::Function(Function::Exp) | Head::Exp1)
                    });
                    if coeff != newe && has_exp {
                        let d: &str = if yes(
                            &crate::functions::elementary::complexes::re(&cdir)?,
                            Predicate::negative,
                        ) {
                            "-"
                        } else {
                            "+"
                        };
                        return gruntz(&coeff, &newz, &Basic::zero(), d);
                    }
                }
                Err(err) if caught(&err, true, true, true) => {}
                Err(err) => return Err(err),
            }
        }
        Err(err) => return Err(err),
        Ok((coeff, ex)) => {
            if crate::core::accumbounds::is_bounds(&coeff) && is_number(&ex, 0) {
                return Ok(coeff);
            }
            if has_infinities(&coeff) {
                return Ok(node.clone());
            }
            if !coeff.has(&newz) {
                if yes(&ex, Predicate::positive) {
                    return Ok(Basic::zero());
                }
                if is_number(&ex, 0) {
                    return Ok(coeff);
                }
                if yes(&ex, Predicate::negative) {
                    if is_number(&cdir, 1) {
                        return mul2(&oo(), &sign_fn(&coeff)?);
                    }
                    if is_number(&cdir, -1) {
                        let s: Basic = mul2(
                            &mul2(&neg(&oo())?, &sign_fn(&coeff)?)?,
                            &pow(&Basic::negative_one(), &add2(&Basic::one(), &ex)?)?,
                        )?;
                        return Ok(s);
                    }
                    return Ok(Basic::zoo());
                }
                return Err(PolyError::NotImplemented(format!("Not sure of sign of {}", ex)));
            }
        }
    }
    if yes(&z0, Predicate::extended_nonnegative) {
        e = crate::core::rewriting::rewrite(
            &e,
            &[Function::Factorial],
            "gamma",
            &crate::core::rewriting::RewriteHints::default(),
        )?;
    }
    let result: PolyResult<Basic> = gruntz(&e, &z, &z0, &dir).and_then(|r| {
        if matches!(r.head(), Head::NaN) {
            Err(pole_error(""))
        } else {
            Ok(r)
        }
    });
    match result {
        Ok(r) => Ok(r),
        Err(PolyError::Pole(_)) | Err(PolyError::Value(_)) => match heuristics(&e, &z, &z0, &dir)? {
            Some(found) => Ok(found),
            None => Ok(node.clone()),
        },
        Err(err) => Err(err),
    }
}

/// `Abs`, `sign`, `arg` and `Heaviside` of arguments of known sign at the
/// point replaced (`set_signs`).
fn set_signs(expr: &Basic, z: &Basic, z0: &Basic, dir: &str) -> R {
    if expr.args().is_empty() {
        return Ok(expr.clone());
    }
    let mut expr: Basic = expr.clone();
    let newargs: Vec<Basic> = expr
        .args()
        .iter()
        .map(|a| set_signs(a, z, z0, dir))
        .collect::<PolyResult<_>>()?;
    if newargs.as_slice() != expr.args() {
        expr = crate::core::function::rebuild_evaluated(&expr, newargs)?;
    }
    let kind = match expr.head() {
        Head::Function(Function::Abs) => 0,
        Head::Function(Function::Sign) => 1,
        Head::Function(Function::Arg) => 2,
        Head::Function(Function::Heaviside) => 3,
        _ => return Ok(expr),
    };
    let inner: Basic = expr.args()[0].clone();
    let mut sig: Basic = match limit(&inner, z, z0, dir) {
        Ok(v) => v,
        Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => return Ok(expr),
        Err(err) => return Err(err),
    };
    if yes(&sig, Predicate::zero) {
        sig = match limit(&div(&Basic::one(), &inner)?, z, z0, dir) {
            Ok(v) => v,
            Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => return Ok(expr),
            Err(err) => return Err(err),
        };
    }
    if yes(&sig, Predicate::extended_real) {
        if yes(&sig, Predicate::extended_negative) {
            return match kind {
                0 => neg(&inner),
                1 => Ok(Basic::negative_one()),
                3 => Ok(Basic::zero()),
                _ => Ok(Basic::pi()),
            };
        }
        if yes(&sig, Predicate::extended_positive) {
            return match kind {
                0 => Ok(inner),
                1 | 3 => Ok(Basic::one()),
                _ => Ok(Basic::zero()),
            };
        }
    }
    Ok(expr)
}

/// The limit from the limits of the arguments (`heuristics`; `None` when
/// they do not decide it).
pub fn heuristics(e: &Basic, z: &Basic, z0: &Basic, dir: &str) -> PolyResult<Option<Basic>> {
    let mut rv: Option<Basic> = None;
    if matches!(z0.head(), Head::Infinity) {
        let found: Basic = limit(&subs(e, z, &div(&Basic::one(), z)?)?, z, &Basic::zero(), "+")?;
        if is_limit(&found) {
            return Ok(None);
        }
        rv = Some(found);
    } else if e.is_mul() || e.is_add() || e.is_pow() || matches!(e.head(), Head::Function(_)) {
        let mut r: Vec<Basic> = Vec::new();
        for a in e.args() {
            let lim: Basic = limit(a, z, z0, dir)?;
            if has_any(&lim, &|n| matches!(n.head(), Head::Infinity)) && is(&lim, Predicate::finite).is_none() {
                if e.is_add() {
                    let mut m: Basic = crate::core::exprtools::factor_terms(e, false, true, false, false)?;
                    if !m.is_mul() {
                        m = crate::polys::rationaltools::together(&m, false, true)?;
                    }
                    if !m.is_mul() {
                        m = crate::polys::polytools::factor(e, &[], false, true)?;
                    }
                    if m.is_mul() {
                        return heuristics(&m, z, z0, dir);
                    }
                    return Ok(None);
                }
                return Ok(None);
            }
            if is_limit(&lim) || matches!(lim.head(), Head::NaN) {
                return Ok(None);
            }
            r.push(lim);
        }
        if !r.is_empty() {
            let mut value: Basic = crate::core::function::rebuild_evaluated(e, r.clone())?;
            if matches!(value.head(), Head::NaN) && e.is_mul() && r.iter().any(crate::core::accumbounds::is_bounds) {
                // the bounds apart, the limit of the rest simplified
                let mut r2: Vec<Basic> = Vec::new();
                let mut e2: Vec<Basic> = Vec::new();
                for (ii, rval) in r.iter().enumerate() {
                    if crate::core::accumbounds::is_bounds(rval) {
                        r2.push(rval.clone());
                    } else {
                        e2.push(e.args()[ii].clone());
                    }
                }
                if !e2.is_empty() {
                    let e3: Basic = crate::simplify::simplify::simplify(
                        &crate::core::mul::mul(&e2)?,
                        &crate::simplify::simplify::Options::default(),
                    )?;
                    let lim3: Basic = limit(&e3, z, z0, dir)?;
                    value = mul2(&lim3, &crate::core::mul::mul(&r2)?)?;
                }
            }
            if matches!(value.head(), Head::NaN) {
                let rat_e: Basic = match crate::simplify::ratsimp::ratsimp(e) {
                    Ok(v) => v,
                    Err(PolyError::Polys(..)) => return Ok(None),
                    Err(err) => return Err(err),
                };
                if matches!(rat_e.head(), Head::NaN) || rat_e == *e {
                    return Ok(None);
                }
                return Ok(Some(limit(&rat_e, z, z0, dir)?));
            }
            rv = Some(value);
        }
    }
    Ok(rv)
}

// -- the tractable form

fn tractable_hyperbolic(f: Function, arg: &Basic) -> PolyResult<Option<Basic>> {
    let pos: Basic = exp(arg)?;
    let negv: Basic = exp(&neg(arg)?)?;
    let two: Basic = Basic::int(2);
    Ok(match f {
        Function::Sinh => Some(div(&sub(&pos, &negv)?, &two)?),
        Function::Cosh => Some(div(&add2(&pos, &negv)?, &two)?),
        Function::Tanh => Some(div(&sub(&pos, &negv)?, &add2(&pos, &negv)?)?),
        Function::Coth => Some(div(&add2(&pos, &negv)?, &sub(&pos, &negv)?)?),
        Function::Sech | Function::Csch => {
            let rf: Function = if f == Function::Sech {
                Function::Cosh
            } else {
                Function::Sinh
            };
            let reciprocal: Basic = Basic::new(Head::Function(rf), vec![arg.clone()]);
            match tractable_hyperbolic(rf, arg)? {
                Some(t) if t != crate::core::function::doit(&reciprocal)? => Some(div(&Basic::one(), &t)?),
                _ => None,
            }
        }
        _ => None,
    })
}

/// `e.rewrite('tractable', deep=True)` (or `'intractable'`) without the
/// limit variable (`rewrite_tractable(e, None)`).
pub fn rewrite_tractable(e: &Basic, intractable: bool) -> R {
    rewrite_tractable_at(e, None, intractable)
}

fn gamma_rewrite(e: &Basic) -> R {
    crate::core::rewriting::rewrite(e, &[], "gamma", &crate::core::rewriting::RewriteHints::default())
}

/// SymPy's `_eval_rewrite_as_tractable` of the gamma function, the
/// factorials written with it, `harmonic`, `uppergamma`, `besseli`,
/// `besselk` and `fibonacci` (`args` already rewritten): `None` for the
/// other nodes and where SymPy's rule gives none (`_tractable_special`,
/// `_tractable_error_and_bessel`).
fn tractable_special(e: &Basic, f: Function, args: &[Basic], limitvar: Option<&Basic>) -> PolyResult<Option<Basic>> {
    use crate::functions::special::gamma_functions::{gamma, loggamma, lowergamma, polygamma};
    let exp = crate::functions::elementary::exponential::exp;
    let tractable = |b: &Basic, var: Option<&Basic>| rewrite_tractable_at(b, var, false);
    Ok(match f {
        // PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma._eval_rewrite_as_tractable
        Function::Gamma => Some(exp(&loggamma(&args[0])?)?),
        // PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:RisingFactorial._eval_rewrite_as_tractable,
        // FallingFactorial._eval_rewrite_as_tractable
        Function::RisingFactorial | Function::FallingFactorial => {
            let (x, k) = (&args[0], &args[1]);
            let rising: bool = f == Function::RisingFactorial;
            if let Some(var) = limitvar {
                let k_lim: Basic = subs(k, var, &Basic::oo())?;
                let one: Basic = Basic::one();
                if matches!(k_lim.head(), Head::Infinity) {
                    if rising {
                        return Ok(Some(div(&tractable(&gamma(&add2(x, k)?)?, limitvar)?, &gamma(x)?)?));
                    }
                    let top: Basic = crate::core::mul::mul(&[
                        pow(&Basic::negative_one(), k)?,
                        tractable(&gamma(&sub(k, x)?)?, limitvar)?,
                    ])?;
                    return Ok(Some(div(&top, &gamma(&neg(x)?)?)?));
                }
                if matches!(k_lim.head(), Head::NegativeInfinity) {
                    if rising {
                        let top: Basic =
                            crate::core::mul::mul(&[pow(&Basic::negative_one(), k)?, gamma(&sub(&one, x)?)?])?;
                        let bottom: Basic = tractable(&gamma(&add2(&sub(&neg(k)?, x)?, &one)?)?, limitvar)?;
                        return Ok(Some(div(&top, &bottom)?));
                    }
                    let bottom: Basic = tractable(&gamma(&add2(&sub(x, k)?, &one)?)?, limitvar)?;
                    return Ok(Some(div(&gamma(&add2(x, &one)?)?, &bottom)?));
                }
            }
            let node: Basic = crate::core::function::rebuild_evaluated(e, args.to_vec())?;
            Some(tractable(&gamma_rewrite(&node)?, limitvar)?)
        }
        // PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial._eval_rewrite_as_tractable
        Function::Binomial => {
            let node: Basic = crate::core::function::rebuild_evaluated(e, args.to_vec())?;
            Some(tractable(&gamma_rewrite(&node)?, None)?)
        }
        // PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_rewrite_as_tractable
        Function::Harmonic => {
            let n: &Basic = &args[0];
            let m: Basic = if args.len() > 1 { args[1].clone() } else { Basic::one() };
            let one: Basic = Basic::one();
            if m == one {
                let value: Basic = add2(
                    &polygamma(&Basic::zero(), &add2(n, &one)?)?,
                    &Basic::atom(Head::EulerGamma),
                )?;
                Some(tractable(&value, limitvar)?)
            } else if yes(&m, Predicate::integer) && yes(&m, Predicate::positive) {
                let m1: Basic = sub(&m, &one)?;
                let difference: Basic = sub(&polygamma(&m1, &one)?, &polygamma(&m1, &add2(n, &one)?)?)?;
                let value: Basic = div(
                    &crate::core::mul::mul(&[pow(&Basic::negative_one(), &m)?, difference])?,
                    &gamma(&m)?,
                )?;
                Some(tractable(&value, limitvar)?)
            } else {
                None
            }
        }
        // PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:uppergamma._eval_rewrite_as_tractable
        // PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf._eval_rewrite_as_tractable
        Function::Erf => {
            use crate::functions::special::error_functions::erfs;
            let z: &Basic = &args[0];
            let gauss: Basic = exp(&neg(&pow(z, &Basic::int(2))?)?)?;
            if let Some(var) = limitvar {
                let at: Basic = limit(z, var, &Basic::oo(), "+")?;
                if matches!(at.head(), Head::NegativeInfinity) {
                    return Ok(Some(add2(
                        &Basic::negative_one(),
                        &crate::core::mul::mul(&[erfs(&neg(z)?)?, gauss])?,
                    )?));
                }
            }
            Some(sub(&Basic::one(), &crate::core::mul::mul(&[erfs(z)?, gauss])?)?)
        }
        // PORTED FROM SYMPY: erfc._eval_rewrite_as_tractable, erfi._eval_rewrite_as_tractable
        // (self.rewrite(erf).rewrite("tractable", deep=True))
        Function::Erfc | Function::Erfi => {
            let as_erf: Basic =
                crate::core::rewriting::rewrite(e, &[], "erf", &crate::core::rewriting::RewriteHints::default())?;
            Some(tractable(&as_erf, limitvar)?)
        }
        // PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei._eval_rewrite_as_tractable
        Function::Ei => Some(crate::core::mul::mul(&[
            exp(&args[0])?,
            crate::functions::special::error_functions::eis(&args[0])?,
        ])?),
        // PORTED FROM SYMPY: sympy/functions/special/error_functions.py:li._eval_rewrite_as_tractable
        Function::LogIntegral => Some(crate::core::mul::mul(&[
            args[0].clone(),
            crate::functions::special::error_functions::eis(&log(&args[0])?)?,
        ])?),
        // PORTED FROM SYMPY: Li._eval_rewrite_as_tractable (self.rewrite(li).rewrite("tractable", deep=True))
        Function::OffsetLogIntegral => {
            let as_li: Basic =
                crate::core::rewriting::rewrite(e, &[], "li", &crate::core::rewriting::RewriteHints::default())?;
            Some(tractable(&as_li, None)?)
        }
        // not SymPy's (sympy-extras#25): for a positive integer order and a
        // positive argument, E_1(z) = -exp(-z)*eis(-z) and E_n(z) =
        // (exp(-z) - z*E_(n-1)(z))/(n - 1)
        Function::Expint => {
            let (nu, z) = (&args[0], &args[1]);
            match nu
                .as_rational()
                .filter(|q| q.is_int())
                .and_then(|q| i64::try_from(crate::numbers::numer(q)).ok())
            {
                Some(order) if order >= 1 && is(z, Predicate::positive) == Some(true) => {
                    let mut factor: Basic = neg(&crate::functions::special::error_functions::eis(&neg(z)?)?)?;
                    for k in 2..=order {
                        factor = div(
                            &sub(&Basic::one(), &crate::core::mul::mul(&[z.clone(), factor])?)?,
                            &Basic::int(k - 1),
                        )?;
                    }
                    Some(crate::core::mul::mul(&[exp(&neg(z)?)?, factor])?)
                }
                _ => None,
            }
        }
        Function::UpperGamma => Some(sub(&exp(&loggamma(&args[0])?)?, &lowergamma(&args[0], &args[1])?)?),
        // PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli._eval_rewrite_as_tractable,
        // besselk._eval_rewrite_as_tractable
        Function::BesselI | Function::BesselK => {
            let (nu, z) = (&args[0], &args[1]);
            if !yes(z, Predicate::extended_real) {
                return Ok(None);
            }
            let (helper, sign) = if f == Function::BesselI {
                (Function::BesselIs, z.clone())
            } else {
                (Function::BesselKs, neg(z)?)
            };
            let scaled: Basic = crate::functions::apply_function(helper, &[nu.clone(), z.clone()])?;
            Some(crate::core::mul::mul(&[exp(&sign)?, scaled])?)
        }
        // PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:fibonacci._eval_rewrite_as_tractable
        Function::Fibonacci if args.len() == 1 => {
            let n: &Basic = &args[0];
            let phi: Basic = Basic::atom(Head::GoldenRatio);
            let phi_n: Basic = pow(&phi, n)?;
            let cosine: Basic =
                crate::functions::elementary::trigonometric::cos(&crate::core::mul::mul(&[Basic::pi(), n.clone()])?)?;
            let top: Basic = sub(&phi_n, &div(&cosine, &phi_n)?)?;
            Some(div(&top, &pow(&Basic::int(5), &div(&Basic::one(), &Basic::int(2))?)?)?)
        }
        _ => None,
    })
}

/// SymPy's `_eval_rewrite_as_intractable` of `loggamma`, `_besseli` and
/// `_besselk` (`args` already rewritten).
fn intractable_special(f: Function, args: &[Basic]) -> PolyResult<Option<Basic>> {
    let exp = crate::functions::elementary::exponential::exp;
    Ok(match f {
        // PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_rewrite_as_intractable
        Function::LogGamma => Some(log(&crate::functions::special::gamma_functions::gamma(&args[0])?)?),
        Function::Erfs | Function::Eis => Some(crate::functions::special::error_functions::helper_intractable(
            f, &args[0],
        )?),
        Function::BesselIs => {
            let value: Basic = crate::functions::apply_function(Function::BesselI, args)?;
            Some(crate::core::mul::mul(&[exp(&neg(&args[1])?)?, value])?)
        }
        Function::BesselKs => {
            let value: Basic = crate::functions::apply_function(Function::BesselK, args)?;
            Some(crate::core::mul::mul(&[exp(&args[1])?, value])?)
        }
        _ => None,
    })
}

/// `e.rewrite('tractable', deep=True, limitvar=...)` (or `'intractable'`):
/// the hyperbolic functions in exponentials, the inverse trigonometric and
/// hyperbolic functions in logarithms, the gamma function as
/// `exp(loggamma)` and the factorials, harmonic numbers, Bessel functions
/// through it (`rewrite_tractable`).
pub fn rewrite_tractable_at(e: &Basic, limitvar: Option<&Basic>, intractable: bool) -> R {
    let args: Vec<Basic> = e
        .args()
        .iter()
        .map(|a| rewrite_tractable_at(a, limitvar, intractable))
        .collect::<PolyResult<_>>()?;
    if let Head::Function(f) = e.head() {
        if !intractable {
            if matches!(
                f,
                Function::Sinh | Function::Cosh | Function::Tanh | Function::Coth | Function::Sech | Function::Csch
            ) {
                if let Some(r) = tractable_hyperbolic(*f, &args[0])? {
                    return Ok(r);
                }
            } else if super::series_inverse::is_inverse(*f) {
                let node: Basic = crate::core::function::rebuild_evaluated(e, args.clone())?;
                return crate::core::rewriting::rewrite(
                    &node,
                    &[],
                    "log",
                    &crate::core::rewriting::RewriteHints::default(),
                );
            }
        }
        if !intractable {
            if let Some(r) = tractable_special(e, *f, &args, limitvar)? {
                return Ok(r);
            }
        } else if let Some(r) = intractable_special(*f, &args)? {
            return Ok(r);
        }
    }
    if args.is_empty() {
        return Ok(e.clone());
    }
    crate::core::function::rebuild_evaluated(e, args)
}

// -- Gruntz's algorithm

fn is_exp(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(Function::Exp)) || (e.is_pow() && matches!(e.args()[0].head(), Head::Exp1))
}

fn exponent(e: &Basic) -> R {
    if matches!(e.head(), Head::Function(Function::Exp)) {
        return Ok(e.args()[0].clone());
    }
    if e.is_pow() {
        return Ok(e.args()[1].clone());
    }
    Err(PolyError::Type(format!("an exponential is expected, got {}", e)))
}

/// `'<'`, `'='` or `'>'`: the growth of `a` against `b` (`compare`).
fn compare(a: &Basic, b: &Basic, x: &Basic) -> PolyResult<char> {
    let la: Basic = if is_exp(a) { exponent(a)? } else { log(a)? };
    let lb: Basic = if is_exp(b) { exponent(b)? } else { log(b)? };
    let c: Basic = limitinf(&div(&la, &lb)?, x)?;
    if is_number(&c, 0) {
        return Ok('<');
    }
    if yes(&c, Predicate::infinite) {
        return Ok('>');
    }
    Ok('=')
}

/// The mrv set: the most rapidly varying subexpressions with their
/// dummies (insertion order), and the rewrites of the dummies
/// (`SubsSet`).
#[derive(Clone, Debug, Default)]
struct SubsSet {
    items: Vec<(Basic, Basic)>,
    rewrites: Vec<(Basic, Basic)>,
}

impl SubsSet {
    fn get(&mut self, key: &Basic) -> Basic {
        if let Some((_, v)) = self.items.iter().find(|(k, _)| k == key) {
            return v.clone();
        }
        let d: Basic = Dummy::new("Dummy", &[]).0;
        self.items.push((key.clone(), d.clone()));
        d
    }
    fn set(&mut self, key: Basic, var: Basic) {
        match self.items.iter().position(|(k, _)| *k == key) {
            Some(i) => self.items[i].1 = var,
            None => self.items.push((key, var)),
        }
    }
    fn contains(&self, key: &Basic) -> bool {
        self.items.iter().any(|(k, _)| k == key)
    }
    fn set_rewrite(&mut self, var: Basic, r: Basic) {
        match self.rewrites.iter().position(|(k, _)| *k == var) {
            Some(i) => self.rewrites[i].1 = r,
            None => self.rewrites.push((var, r)),
        }
    }
    fn rewrite_of(&self, var: &Basic) -> Option<&Basic> {
        self.rewrites.iter().find(|(k, _)| k == var).map(|(_, r)| r)
    }
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    fn do_subs(&self, e: &Basic) -> Basic {
        let mut e: Basic = e.clone();
        for (expr, var) in &self.items {
            let rule: HashMap<Basic, Basic> = [(var.clone(), expr.clone())].into_iter().collect();
            e = e.xreplace(&rule);
        }
        e
    }
    fn meets(&self, s2: &SubsSet) -> bool {
        self.items.iter().any(|(k, _)| s2.contains(k))
    }
    fn union(&self, s2: &SubsSet, exps: Option<Basic>) -> (SubsSet, Option<Basic>) {
        let mut res: SubsSet = self.clone();
        let mut exps: Option<Basic> = exps;
        let mut tr: HashMap<Basic, Basic> = HashMap::new();
        for (expr, var) in &s2.items {
            if self.contains(expr) {
                let mine: Basic = res.get(expr);
                if let Some(e) = exps.as_ref() {
                    let rule: HashMap<Basic, Basic> = [(var.clone(), mine.clone())].into_iter().collect();
                    exps = Some(e.xreplace(&rule));
                }
                tr.insert(var.clone(), mine);
            } else {
                res.items.push((expr.clone(), var.clone()));
            }
        }
        for (var, rewr) in &s2.rewrites {
            res.set_rewrite(var.clone(), rewr.xreplace(&tr));
        }
        (res, exps)
    }
}

fn rebuild_assoc(e: &Basic, args: &[Basic]) -> R {
    if e.is_add() {
        crate::core::add::add(args)
    } else {
        crate::core::mul::mul(args)
    }
}

fn as_two_terms(e: &Basic) -> (Basic, Basic) {
    let rest: Vec<Basic> = e.args()[1..].to_vec();
    let tail: Basic = if e.is_add() {
        if rest.len() == 1 {
            rest[0].clone()
        } else {
            crate::core::add::raw_add(rest)
        }
    } else if rest.len() == 1 {
        rest[0].clone()
    } else {
        crate::core::mul::raw_mul(rest)
    };
    (e.args()[0].clone(), tail)
}

/// The mrv set of `e` and `e` with its elements replaced by their
/// dummies (`mrv`).
fn mrv(e: &Basic, x: &Basic) -> PolyResult<(SubsSet, Basic)> {
    let e: Basic = crate::simplify::powsimp::powsimp(
        e,
        true,
        crate::simplify::powsimp::Combine::Exp,
        false,
        &crate::simplify::powsimp::count_ops,
    )?;
    if !e.has(x) {
        return Ok((SubsSet::default(), e));
    }
    if e == *x {
        let mut s = SubsSet::default();
        let v: Basic = s.get(x);
        return Ok((s, v));
    }
    if e.is_mul() || e.is_add() {
        let (i, d) = crate::core::expr::as_independent(&e, x, e.is_add())?;
        if (e.is_add() && !d.is_add()) || (e.is_mul() && !d.is_mul()) {
            let (s, expr) = mrv(&d, x)?;
            return Ok((s, rebuild_assoc(&e, &[i, expr])?));
        }
        let (a, b) = as_two_terms(&d);
        let (s1, e1) = mrv(&a, x)?;
        let (s2, e2) = mrv(&b, x)?;
        return mrv_max1(&s1, &s2, &rebuild_assoc(&e, &[i, e1, e2])?, x);
    }
    if e.is_pow() && !is_exp(&e) {
        let mut ep: Basic = Basic::one();
        let mut b1: Basic = e.clone();
        let mut cur: Basic = e.clone();
        while cur.is_pow() {
            b1 = cur.args()[0].clone();
            ep = mul2(&ep, &cur.args()[1])?;
            cur = b1.clone();
        }
        if is_number(&b1, 1) {
            return Ok((SubsSet::default(), b1));
        }
        if ep.has(x) {
            return mrv(&exp(&mul2(&ep, &log(&b1)?)?)?, x);
        }
        let (s, expr) = mrv(&b1, x)?;
        return Ok((s, pow(&expr, &ep)?));
    }
    if matches!(e.head(), Head::Function(Function::Log)) {
        let (s, expr) = mrv(&e.args()[0], x)?;
        return Ok((s, log(&expr)?));
    }
    if is_exp(&e) {
        let ex: Basic = exponent(&e)?;
        if matches!(ex.head(), Head::Function(Function::Log)) {
            return mrv(&ex.args()[0], x);
        }
        let li: Basic = limitinf(&ex, x)?;
        if factors_of(&li).iter().any(|f| yes(f, Predicate::infinite)) {
            let mut s1 = SubsSet::default();
            let e1v: Basic = s1.get(&e);
            let (s2, e2) = mrv(&ex, x)?;
            let (mut su, _) = s1.union(&s2, None);
            let ee2: Basic = exp(&e2)?;
            su.set_rewrite(e1v.clone(), ee2.clone());
            return mrv_max3(&s1, &e1v, &s2, &ee2, &su, &e1v, x);
        }
        let (s, expr) = mrv(&ex, x)?;
        return Ok((s, exp(&expr)?));
    }
    if matches!(e.head(), Head::Undefined(_)) {
        return Err(value_error(
            "MRV set computation for UndefinedFunction is not allowed".to_string(),
        ));
    }
    if matches!(e.head(), Head::Function(_)) {
        let found: Vec<(SubsSet, Basic)> = e.args().iter().map(|a| mrv(a, x)).collect::<PolyResult<_>>()?;
        let nonempty: Vec<&SubsSet> = found.iter().filter(|(s, _)| !s.is_empty()).map(|(s, _)| s).collect();
        if nonempty.len() != 1 {
            return Err(PolyError::NotImplemented(
                "MRV set computation for functions in several variables not implemented.".to_string(),
            ));
        }
        let s: SubsSet = nonempty[0].clone();
        let ss = SubsSet::default();
        let args: Vec<Basic> = found.iter().map(|(_, r)| ss.do_subs(r)).collect();
        return Ok((s, crate::core::function::rebuild_evaluated(&e, args)?));
    }
    if matches!(e.head(), Head::Hyper | Head::MeijerG | Head::Piecewise) {
        // a function of SymPy which is not an application of the core: its
        // arguments which are not expressions (the parameter tuples) are
        // free of x and kept
        let is_expression =
            |a: &Basic| !(crate::core::logic::is_boolean(a) || matches!(a.head(), Head::Tuple | Head::ExprCondPair));
        let mut found: Vec<(SubsSet, Basic)> = Vec::new();
        for argument in e.args() {
            if is_expression(argument) {
                found.push(mrv(argument, x)?);
            } else if argument.has(x) {
                return Err(PolyError::NotImplemented(format!(
                    "Don't know how to calculate the mrv of '{}'",
                    e
                )));
            } else {
                found.push((SubsSet::default(), argument.clone()));
            }
        }
        let nonempty: Vec<&SubsSet> = found.iter().filter(|(s, _)| !s.is_empty()).map(|(s, _)| s).collect();
        if nonempty.len() != 1 {
            return Err(PolyError::NotImplemented(
                "MRV set computation for functions in several variables not implemented.".to_string(),
            ));
        }
        let s: SubsSet = nonempty[0].clone();
        let ss = SubsSet::default();
        let args: Vec<Basic> = found
            .iter()
            .map(|(_, r)| if is_expression(r) { ss.do_subs(r) } else { r.clone() })
            .collect();
        return Ok((s, crate::core::function::rebuild_evaluated(&e, args)?));
    }
    if matches!(e.head(), Head::Derivative) {
        return Err(PolyError::NotImplemented(
            "MRV set computation for derivatives not implemented yet.".to_string(),
        ));
    }
    Err(not_ported(&format!("the mrv of {}", e)))
}

fn mrv_max3(
    f: &SubsSet,
    expsf: &Basic,
    g: &SubsSet,
    expsg: &Basic,
    union: &SubsSet,
    expsboth: &Basic,
    x: &Basic,
) -> PolyResult<(SubsSet, Basic)> {
    if f.is_empty() {
        return Ok((g.clone(), expsg.clone()));
    }
    if g.is_empty() {
        return Ok((f.clone(), expsf.clone()));
    }
    if f.meets(g) {
        return Ok((union.clone(), expsboth.clone()));
    }
    match compare(&f.items[0].0, &g.items[0].0, x)? {
        '>' => Ok((f.clone(), expsf.clone())),
        '<' => Ok((g.clone(), expsg.clone())),
        _ => Ok((union.clone(), expsboth.clone())),
    }
}

fn mrv_max1(f: &SubsSet, g: &SubsSet, exps: &Basic, x: &Basic) -> PolyResult<(SubsSet, Basic)> {
    let (u, b) = f.union(g, Some(exps.clone()));
    let b: Basic = b.ok_or_else(|| value_error("the union lost the expression".to_string()))?;
    mrv_max3(f, &g.do_subs(exps), g, &f.do_subs(exps), &u, &b, x)
}

/// The sign of `e` for `x` large enough (`sign`).
fn gsign(e: &Basic, x: &Basic) -> R {
    if yes(e, Predicate::positive) {
        return Ok(Basic::one());
    }
    if yes(e, Predicate::negative) {
        return Ok(Basic::negative_one());
    }
    if yes(e, Predicate::zero) {
        return Ok(Basic::zero());
    }
    if !e.has(x) {
        return sign_fn(&crate::simplify::simplify::logcombine(e, false)?);
    }
    if e == x {
        return Ok(Basic::one());
    }
    if e.is_mul() {
        let (a, b) = as_two_terms(e);
        let sa: Basic = gsign(&a, x)?;
        if is_number(&sa, 0) {
            return Ok(Basic::zero());
        }
        return mul2(&sa, &gsign(&b, x)?);
    }
    if matches!(e.head(), Head::Function(Function::Exp)) {
        return Ok(Basic::one());
    }
    if e.is_pow() {
        if is_exp(e) {
            return Ok(Basic::one());
        }
        let s: Basic = gsign(&e.args()[0], x)?;
        if is_number(&s, 1) {
            return Ok(Basic::one());
        }
        if e.args()[1].is_integer_value() {
            return pow(&s, &e.args()[1]);
        }
    } else if matches!(e.head(), Head::Function(Function::Log)) && yes(&e.args()[0], Predicate::positive) {
        return gsign(&sub(&e.args()[0], &Basic::one())?, x);
    }
    let (c0, _) = mrv_leadterm(e, x)?;
    gsign(&c0, x)
}

/// The limit of `e` as `x` goes to `oo` (`limitinf`).
fn limitinf(e: &Basic, x: &Basic) -> R {
    let old: Basic = e.clone();
    if !e.has(x) {
        return Ok(e.clone());
    }
    let mut e: Basic = e.clone();
    let mut x: Basic = x.clone();
    if has_any(&e, &is_order) {
        e = crate::core::order::remove_o(&crate::core::function::expand_method(&e)?)?;
    }
    if !yes(&x, Predicate::positive) || yes(&x, Predicate::integer) {
        let p: Basic = Dummy::new("p", &[(Predicate::positive, true)]).0;
        e = subs(&e, &x, &p)?;
        x = p;
    }
    let rewritten: Basic = rewrite_tractable_at(&e, Some(&x), false)?;
    e = crate::simplify::powsimp::powdenest(&rewritten, false, false)?;
    let (c0, e0) = if crate::core::accumbounds::is_bounds(&e) {
        let lo = mrv_leadterm(&e.args()[0], &x)?;
        let hi = mrv_leadterm(&e.args()[1], &x)?;
        if lo != hi {
            return Err(PolyError::NotImplemented(String::new()));
        }
        lo
    } else {
        mrv_leadterm(&e, &x)?
    };
    let sig: Basic = gsign(&e0, &x)?;
    if is_number(&sig, 1) {
        return Ok(Basic::zero());
    }
    if is_number(&sig, -1) {
        let i: Basic = Basic::imaginary_unit();
        let a: Basic = crate::core::symbol::Wild::new("a", vec![i.clone()], vec![], &[]).0;
        if crate::core::matching::match_pattern(&c0, &mul2(&i, &a)?, &[], false)?.is_some() {
            return mul2(&c0, &oo());
        }
        let s: Basic = gsign(&c0, &x)?;
        if is_number(&s, 0) {
            return Err(value_error("Leading term should not be 0".to_string()));
        }
        return mul2(&s, &oo());
    }
    if is_number(&sig, 0) {
        let c: Basic = if c0 == old {
            crate::polys::polytools::cancel(&c0, &[])?
        } else {
            c0
        };
        return limitinf(&c, &x);
    }
    Err(value_error(format!("{} could not be evaluated", sig)))
}

fn moveup2(s: &SubsSet, x: &Basic) -> PolyResult<SubsSet> {
    let up: HashMap<Basic, Basic> = [(x.clone(), exp(x)?)].into_iter().collect();
    let mut r = SubsSet::default();
    for (expr, var) in &s.items {
        r.set(expr.xreplace(&up), var.clone());
    }
    for (var, expr) in &s.rewrites {
        r.set_rewrite(var.clone(), expr.xreplace(&up));
    }
    Ok(r)
}

/// `(c0, e0)`: the leading term `c0*w**e0` of `e` in the most rapidly
/// varying `w` going to zero (`mrv_leadterm`).
/// The variable `h` of `z = z0 + h`, which goes to `0` from `dir`
/// (`_shifted_variable`): `z` itself when its assumptions say nothing of
/// its sign, otherwise a dummy of the sign of the direction (not in SymPy:
/// the shifted `z` kept the assumptions of `z`, sympy-extras#25).
fn shifted_variable(z: &Basic, z0: &Basic, dir: &str) -> Basic {
    let signs = [
        Predicate::positive,
        Predicate::negative,
        Predicate::nonnegative,
        Predicate::nonpositive,
    ];
    if signs.iter().all(|p| is(z, *p).is_none()) {
        return z.clone();
    }
    if is(z, Predicate::extended_real) == Some(true)
        && is(z0, Predicate::extended_real) == Some(true)
        && (dir == "+" || dir == "-")
    {
        let p: Predicate = if dir == "+" {
            Predicate::positive
        } else {
            Predicate::negative
        };
        return Dummy::new("h", &[(p, true)]).0;
    }
    Dummy::new("h", &[]).0
}

fn mrv_leadterm(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    if !e.has(x) {
        return Ok((e.clone(), Basic::zero()));
    }
    let (mut omega, mut exps) = mrv(e, x)?;
    if omega.is_empty() {
        return Ok((exps, Basic::zero()));
    }
    if omega.contains(x) {
        omega = moveup2(&omega, x)?;
        let up: HashMap<Basic, Basic> = [(x.clone(), exp(x)?)].into_iter().collect();
        exps = exps.xreplace(&up);
    }
    let w: Basic = Dummy::new("w", &[(Predicate::positive, true)]).0;
    let (f0, logw) = rewrite(&exps, &omega, x, &w)?;
    let query = |t: &Basic| t.is_pow() && t.has(x);
    let value = |t: &Basic| -> R { exp(&mul2(&log(&t.args()[0])?, &t.args()[1])?) };
    let f: Basic = crate::core::function::replace_where(&f0, &query, &value)?;
    let lt: (Basic, Basic) = match leadterm(&f, &w, Some(&logw), &Basic::zero()) {
        Ok(v) => v,
        Err(err) if caught(&err, true, true, true) => {
            let n0: i64 = 1;
            let mut series: Basic = crate::core::order::order(&Basic::one(), &[])?;
            let mut incr: i64 = 1;
            while is_order(&series) {
                if n0 + incr > 128 {
                    // SymPy's loop has no end: the expansion of a function
                    // which is exponentially small in w (expint(2, 1/w)) is
                    // O(w**n) for every n
                    return Err(PolyError::Pole(format!(
                        "the expansion of {} in {} vanishes to every order",
                        f, w
                    )));
                }
                series = eval_nseries(&f, &w, n0 + incr, Some(&logw), &Basic::zero())?;
                incr *= 2;
            }
            let series: Basic = crate::core::order::remove_o(&crate::core::function::expand_method(&series)?)?;
            match leadterm(&series, &w, Some(&logw), &Basic::zero()) {
                Ok(v) => v,
                Err(err) if caught(&err, true, true, true) => {
                    let mut l = crate::solvers::solvers::as_coeff_exponent(&f, &w)?;
                    if l.0.has(&w) {
                        let (fb, fe) = crate::core::expr::as_base_exp(&f)?;
                        let base = crate::solvers::solvers::as_coeff_exponent(&fb, &w)?;
                        l = (pow(&base.0, &fe)?, mul2(&base.1, &fe)?);
                    }
                    l
                }
                Err(err) => return Err(err),
            }
        }
        Err(err) => return Err(err),
    };
    Ok((subs(&lt.0, &log(&w)?, &logw)?, lt.1))
}

fn heights(omega: &[(Basic, Basic)], rewrites: &SubsSet) -> Vec<(Basic, i64)> {
    let before: Vec<(Basic, Vec<Basic>)> = omega
        .iter()
        .map(|(_, v)| {
            let mut b: Vec<Basic> = Vec::new();
            if let Some(r) = rewrites.rewrite_of(v) {
                for (_, v2) in omega {
                    if r.has(v2) {
                        b.push(v2.clone());
                    }
                }
            }
            (v.clone(), b)
        })
        .collect();
    fn ht(v: &Basic, before: &[(Basic, Vec<Basic>)]) -> i64 {
        let deps: &Vec<Basic> = &before.iter().find(|(k, _)| k == v).expect("an element").1;
        deps.iter().map(|u| ht(u, before)).sum::<i64>() + 1
    }
    omega.iter().map(|(_, v)| (v.clone(), ht(v, &before))).collect()
}

/// `e` rewritten in terms of the most rapidly varying `wsym`, with the
/// value of `log(wsym)` (`rewrite`).
fn rewrite(e: &Basic, omega_set: &SubsSet, x: &Basic, wsym: &Basic) -> PolyResult<(Basic, Basic)> {
    if omega_set.is_empty() {
        return Err(value_error("Length cannot be 0".to_string()));
    }
    if omega_set.items.iter().any(|(t, _)| !is_exp(t)) {
        return Err(value_error("Value should be exp".to_string()));
    }
    let mut omega: Vec<(Basic, Basic)> = omega_set.items.clone();
    let hs: Vec<(Basic, i64)> = heights(&omega, omega_set);
    let height = |v: &Basic| hs.iter().find(|(k, _)| k == v).expect("a height").1;
    omega.sort_by_key(|a| std::cmp::Reverse(height(&a.1)));
    let mut sig: Basic = Basic::zero();
    let mut g: Basic = omega[0].0.clone();
    for (gg, _) in &omega {
        g = gg.clone();
        sig = gsign(&exponent(gg)?, x)?;
        if !is_number(&sig, 1) && !is_number(&sig, -1) && !has_any(&sig, &crate::core::accumbounds::is_bounds) {
            return Err(PolyError::NotImplemented(format!(
                "Result depends on the sign of {}",
                sig
            )));
        }
    }
    let wsym: Basic = if is_number(&sig, 1) {
        div(&Basic::one(), wsym)?
    } else {
        wsym.clone()
    };
    let g_exp: Basic = exponent(&g)?;
    let mut o2: Vec<(Basic, Basic)> = Vec::new();
    let mut denominators: Vec<Int> = Vec::new();
    for (f, var) in &omega {
        let c: Basic = limitinf(&div(&exponent(f)?, &g_exp)?, x)?;
        if let Some(r) = c.as_rational() {
            denominators.push(Int::from(denom(r)));
        }
        let mut arg: Basic = exponent(f)?;
        if let Some(r) = omega_set.rewrite_of(var) {
            if !is_exp(r) {
                return Err(value_error("Value should be exp".to_string()));
            }
            arg = r.args()[0].clone();
        }
        o2.push((
            var.clone(),
            mul2(&exp(&sub(&arg, &mul2(&c, &g_exp)?)?)?, &pow(&wsym, &c)?)?,
        ));
    }
    let mut f: Basic = crate::simplify::powsimp::powsimp(
        e,
        true,
        crate::simplify::powsimp::Combine::Exp,
        false,
        &crate::simplify::powsimp::count_ops,
    )?;
    for (a, b) in &o2 {
        let rule: HashMap<Basic, Basic> = [(a.clone(), b.clone())].into_iter().collect();
        f = f.xreplace(&rule);
    }
    for (_, var) in &omega {
        if f.has(var) {
            return Err(value_error(format!("the rewrite left {}", var)));
        }
    }
    let mut logw: Basic = g_exp.clone();
    if is_number(&sig, 1) {
        logw = neg(&logw)?;
    }
    let mut all: Vec<Int> = vec![Int::ONE];
    all.extend(denominators);
    let exponent_v: Int = crate::core::intfunc::ilcm(&all);
    let ex: Basic = Basic::integer(exponent_v);
    f = crate::core::subs::subs_mapping(&f, &[(wsym.clone(), pow(&wsym, &ex)?)])?;
    logw = div(&logw, &ex)?;
    let normal = |t: &Basic| -> R { crate::simplify::radsimp::normal(t) };
    let normalised: Basic = crate::core::traversal::bottom_up(&f, &normal, false)?;
    Ok((normalised, logw))
}

/// The limit of `e` as `z` goes to `z0` from `dir` by Gruntz's algorithm
/// (`gruntz`).
pub fn gruntz(e: &Basic, z: &Basic, z0: &Basic, dir: &str) -> R {
    if !z.is_symbol() {
        return Err(PolyError::NotImplemented(
            "Second argument must be a Symbol".to_string(),
        ));
    }
    let i: Basic = Basic::imaginary_unit();
    let e0: Basic = if matches!(z0.head(), Head::Infinity) || *z0 == mul2(&i, &oo())? {
        e.clone()
    } else if matches!(z0.head(), Head::NegativeInfinity) || *z0 == mul2(&neg(&i)?, &oo())? {
        subs(e, z, &neg(z)?)?
    } else if dir == "-" {
        subs(e, z, &sub(z0, &div(&Basic::one(), z)?)?)?
    } else if dir == "+" {
        subs(e, z, &add2(z0, &div(&Basic::one(), z)?)?)?
    } else {
        return Err(PolyError::NotImplemented("dir must be '+' or '-'".to_string()));
    };
    let r: Basic = limitinf(&e0, z)?;
    rewrite_tractable(&r, true)
}
