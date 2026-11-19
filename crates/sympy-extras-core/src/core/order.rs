//! Port of `sympy_extras/core/order.py` (SymPy's `Order`): the node
//! `O(expr, (x, point), ...)`, `Head::Order` with the arguments
//! `(expr, Tuple(x, point), ...)`; its evaluated constructor [`order`]
//! (the leading term in the variables, at zero after moving the point),
//! [`contains`], the substitution of a variable, and the order terms in
//! sums, products and powers (`Add.flatten`, `Mul.flatten`,
//! `Order._eval_power`).
//!
//! The limits of `contains` are taken by [`crate::core::series::limit_lite`]
//! until `core/limits.py` is ported: a ratio which is not `c*x**k` there
//! is `NotImplemented`.

use super::basic::{Basic, Head};
use super::series::{as_leading_term, is_order};
use crate::core::assumptions::Predicate;
use crate::core::expr::{factors_of, is};
use crate::core::symbol::Dummy;
use crate::error::{not_ported, value_error, PolyError, PolyResult};
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
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn tuple(v: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, v)
}
fn is_zero_number(e: &Basic) -> bool {
    e.is_number_class() && *e == Basic::zero()
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

/// The raw node `Order(expr, *limits)`; without limits, the free symbols
/// of `expr` (sorted) at zero (`Order.__init__`).
pub fn node(expr: &Basic, limits: &[(Basic, Basic)]) -> Basic {
    let mut args: Vec<Basic> = vec![expr.clone()];
    if limits.is_empty() {
        let mut symbols: Vec<Basic> = expr.free_symbols().into_iter().collect();
        symbols.sort_by_cached_key(crate::core::sorting::sort_key);
        args.extend(symbols.into_iter().map(|s| tuple(vec![s, Basic::zero()])));
    } else {
        args.extend(limits.iter().map(|(s, p)| tuple(vec![s.clone(), p.clone()])));
    }
    Basic::new(Head::Order, args)
}

/// The expression and the `(variable, point)` pairs of an order term.
pub fn parts(e: &Basic) -> (Basic, Vec<(Basic, Basic)>) {
    let limits: Vec<(Basic, Basic)> = e.args()[1..]
        .iter()
        .map(|t| (t.args()[0].clone(), t.args()[1].clone()))
        .collect();
    (e.args()[0].clone(), limits)
}

/// `O(e)` as Python prints it.
pub fn print(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let (expr, limits) = parts(e);
    if limits.iter().all(|(_, p)| is_zero_number(p)) {
        if limits.len() <= 1 {
            return format!("O({})", doprint(&expr));
        }
        let vars: Vec<String> = limits.iter().map(|(s, _)| doprint(s)).collect();
        return format!("O({}, {})", doprint(&expr), vars.join(", "));
    }
    let vars: Vec<String> = limits
        .iter()
        .map(|(s, p)| format!("({}, {})", doprint(s), doprint(p)))
        .collect();
    format!("O({}, {})", doprint(&expr), vars.join(", "))
}

/// The node rebuilt from new arguments, evaluated (`Order._evaluated`).
pub fn evaluated(args: &[Basic]) -> R {
    let limits: Vec<(Basic, Basic)> = args[1..]
        .iter()
        .map(|t| {
            if !matches!(t.head(), Head::Tuple) || t.args().len() != 2 {
                return Err(PolyError::Type(format!("a tuple is expected, got {}", t)));
            }
            Ok((t.args()[0].clone(), t.args()[1].clone()))
        })
        .collect::<PolyResult<_>>()?;
    order(&args[0], &limits)
}

/// The evaluated order term, as SymPy's `Order` constructor gives it
/// (`order`): the variables at zero when none is given.
pub fn order(expr: &Basic, given: &[(Basic, Basic)]) -> R {
    for (s, _) in given {
        if !s.is_symbol() {
            return Err(PolyError::Type(format!("Variables are not symbols, got {}", s)));
        }
    }
    order_inner(expr, given, !given.is_empty())
}

/// `O(e, x)`: the order term in one variable at zero.
pub fn order_at_zero(expr: &Basic, x: &Basic) -> R {
    order(expr, &[(x.clone(), Basic::zero())])
}

fn is_infinity_times_i(p: &Basic, negative: bool) -> bool {
    p.is_mul()
        && p.args().len() == 2
        && matches!(
            (p.args()[0].head(), negative),
            (Head::Infinity, false) | (Head::NegativeInfinity, true)
        )
        && matches!(p.args()[1].head(), Head::ImaginaryUnit)
}

/// `as_independent(*symbols, as_Add=False)[1]`: the factors depending on
/// one of the symbols.
fn dependent_part(e: &Basic, symbols: &[Basic]) -> R {
    let depends = |f: &Basic| symbols.iter().any(|s| f.has(s));
    if e.is_mul() {
        let dep: Vec<Basic> = e.args().iter().filter(|f| depends(f)).cloned().collect();
        return mul(&dep);
    }
    Ok(if depends(e) { e.clone() } else { Basic::one() })
}

fn expand_power_base(e: &Basic) -> R {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: true,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}

fn expand_log(e: &Basic) -> R {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: true,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}

fn order_expr(e: &Basic) -> R {
    if !is_order(e) {
        return Err(PolyError::Type(format!("an order term is expected, got {}", e)));
    }
    Ok(e.args()[0].clone())
}

fn order_inner(expr: &Basic, given: &[(Basic, Basic)], explicit: bool) -> R {
    let mut expr: Basic = expr.clone();
    let (mut variables, mut point): (Vec<Basic>, Vec<Basic>) = if !explicit {
        if is_order(&expr) {
            parts(&expr).1.into_iter().unzip()
        } else {
            let mut v: Vec<Basic> = expr.free_symbols().into_iter().collect();
            v.sort_by_cached_key(crate::core::sorting::sort_key);
            let n: usize = v.len();
            (v, vec![Basic::zero(); n])
        }
    } else {
        given.iter().cloned().unzip()
    };
    for (i, v) in variables.iter().enumerate() {
        if variables[..i].contains(v) {
            return Err(value_error(format!(
                "Variables are supposed to be unique symbols, got {:?}",
                variables
            )));
        }
    }
    if is_order(&expr) {
        let expr_vp: Vec<(Basic, Basic)> = parts(&expr).1;
        let mut new_vp: Vec<(Basic, Basic)> = expr_vp.clone();
        for (v, p) in variables.iter().zip(&point) {
            match new_vp.iter().find(|(k, _)| k == v) {
                Some((_, q)) => {
                    if p != q {
                        return Err(not_ported("Mixing Order at different points is not supported."));
                    }
                }
                None => new_vp.push((v.clone(), p.clone())),
            }
        }
        if new_vp.len() == expr_vp.len() {
            return Ok(expr);
        }
        (variables, point) = new_vp.into_iter().unzip();
    }
    if matches!(expr.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if variables.iter().any(|x| point.iter().any(|p| p.has(x))) {
        return Err(value_error(format!("Got {:?} as a point.", point)));
    }
    if !variables.is_empty() {
        if point.iter().any(|p| *p != point[0]) {
            return Err(not_ported(
                "Multivariable orders at different points are not supported.",
            ));
        }
        let p0: Basic = point[0].clone();
        let mut s: Vec<(Basic, Basic)> = Vec::new();
        let mut rs: Vec<(Basic, Basic)> = Vec::new();
        let mut dummies: Vec<Basic> = Vec::new();
        let positive_infinity: bool = matches!(p0.head(), Head::Infinity) || is_infinity_times_i(&p0, false);
        let negative_infinity: bool = matches!(p0.head(), Head::NegativeInfinity) || is_infinity_times_i(&p0, true);
        for k in &variables {
            if !(positive_infinity || negative_infinity || !is_zero_number(&p0)) {
                break;
            }
            let d: Basic = Dummy::new("Dummy", &[]).0;
            if positive_infinity {
                s.push((k.clone(), pow(&d, &Basic::negative_one())?));
                rs.push((d.clone(), pow(k, &Basic::negative_one())?));
            } else if negative_infinity {
                s.push((k.clone(), crate::core::mul::neg(&pow(&d, &Basic::negative_one())?)?));
                rs.push((d.clone(), crate::core::mul::neg(&pow(k, &Basic::negative_one())?)?));
            } else {
                s.push((k.clone(), add(&[d.clone(), p0.clone()])?));
                rs.push((d.clone(), sub(k, &p0)?));
            }
            dummies.push(d);
        }
        let ps: Vec<Basic> = if s.is_empty() {
            point.clone()
        } else {
            vec![Basic::zero(); point.len()]
        };
        if !s.is_empty() {
            expr = crate::core::subs::subs_mapping(&expr, &s)?;
        }
        if expr.is_add() {
            expr = crate::polys::polytools::factor(&expr, &[], false, true)?;
        }
        let args: Vec<Basic> = if s.is_empty() {
            variables.clone()
        } else {
            dummies.clone()
        };
        if variables.len() > 1 {
            // SymPy expands (x*(x + y)).as_leading_term(x, y) is x*y, the
            // wrong order term
            expr = crate::core::function::expand_method(&expr)?;
        }
        let pts: Vec<(Basic, Basic)> = args.iter().cloned().zip(ps.iter().cloned()).collect();
        let mut old_expr: Option<Basic> = None;
        while old_expr.as_ref() != Some(&expr) {
            old_expr = Some(expr.clone());
            if expr.is_add() {
                let lst = extract_leading_order(&expr, &args, None)?;
                let leading: Vec<Basic> = lst.iter().map(|(_, f)| f.args()[0].clone()).collect();
                expr = add(&leading)?;
            } else if !is_zero_number(&expr) {
                match as_leading_term(&expr, &args, None, &Basic::zero()) {
                    Ok(lt) => expr = lt,
                    Err(PolyError::Pole(_)) => {
                        let applications = |e: &Basic| matches!(e.head(), Head::Function(_) | Head::Undefined(_));
                        if applications(&expr) || expr.args().iter().all(applications) {
                            // an expression of functions only cannot be
                            // simplified further
                        } else {
                            let mut orders: Vec<Basic> = Vec::new();
                            for a in expr.args() {
                                let lt: Basic = match as_leading_term(a, &args, None, &Basic::zero()) {
                                    Ok(v) => v,
                                    Err(PolyError::Pole(_)) => a.clone(),
                                    Err(e) => return Err(e),
                                };
                                orders.push(if !args.contains(&lt) {
                                    order(&lt, &[])?
                                } else {
                                    order(&lt, &pts)?
                                });
                            }
                            if expr.is_add() {
                                let mut new_expr: Basic = order(&add(&orders)?, &pts)?;
                                if new_expr.is_add() {
                                    let inner: Vec<Basic> =
                                        new_expr.args().iter().map(order_expr).collect::<PolyResult<_>>()?;
                                    new_expr = order(&add(&inner)?, &pts)?;
                                }
                                expr = order_expr(&new_expr)?;
                            } else if expr.is_mul() {
                                let inner: Vec<Basic> = orders.iter().map(order_expr).collect::<PolyResult<_>>()?;
                                expr = mul(&inner)?;
                            } else if expr.is_pow() {
                                let log_base: Basic = crate::functions::elementary::exponential::log(&expr.args()[0])?;
                                expr = crate::functions::elementary::exponential::exp(&mul(&[
                                    expr.args()[1].clone(),
                                    log_base,
                                ])?)?;
                            }
                        }
                    }
                    Err(e) => return Err(e),
                }
                // a symbol with zero=True
                expr = if yes(&expr, Predicate::zero) {
                    Basic::zero()
                } else {
                    dependent_part(&expr, &args)?
                };
                expr = expand_power_base(&expr)?;
                expr = expand_log(&expr)?;
                if args.len() == 1 {
                    expr = combine_exponents(&expr, &args[0])?;
                }
            }
        }
        if !rs.is_empty() {
            expr = crate::core::subs::subs_mapping(&expr, &rs)?;
        }
    }
    if is_order(&expr) {
        expr = expr.args()[0].clone();
    }
    if !variables.iter().any(|v| expr.has(v)) && !yes(&expr, Predicate::zero) {
        expr = Basic::one();
    }
    let vp: Vec<(Basic, Basic)> = variables.iter().cloned().zip(point.iter().cloned()).collect();
    variables.sort_by_cached_key(crate::core::sorting::sort_key);
    if variables.is_empty() {
        return Ok(Basic::new(Head::Order, vec![expr]));
    }
    let limits: Vec<(Basic, Basic)> = variables
        .iter()
        .map(|v| {
            (
                v.clone(),
                vp.iter().find(|(k, _)| k == v).expect("a variable").1.clone(),
            )
        })
        .collect();
    Ok(node(&expr, &limits))
}

/// The factors `x**q`, `(-x)**q`, `(x**r)**q` and `(-x**r)**q` with real
/// exponents made powers of `x` (`_combine_exponents`).
fn combine_exponents(expr: &Basic, x: &Basic) -> R {
    let mut margs: Vec<Basic> = factors_of(&dependent_part(expr, std::slice::from_ref(x))?);
    let minus_x: Basic = crate::core::mul::neg(x)?;
    let is_x = |b: &Basic| *b == *x || *b == minus_x;
    for t in margs.iter_mut() {
        if !t.is_pow() {
            continue;
        }
        let (b, q) = (t.args()[0].clone(), t.args()[1].clone());
        if is_x(&b) && yes(&q, Predicate::real) && !q.has(x) {
            *t = pow(x, &q)?;
        } else if b.is_pow() && !b.args()[1].has(x) {
            let (b2, r) = (&b.args()[0], &b.args()[1]);
            if is_x(b2) && yes(r, Predicate::real) {
                *t = pow(x, &mul(&[r.clone(), q.clone()])?)?;
            }
        } else if b.is_mul() && b.args()[0] == Basic::negative_one() {
            let nb: Basic = crate::core::mul::neg(&b)?;
            if nb.is_pow() && !nb.args()[1].has(x) {
                let (b2, r) = (&nb.args()[0], &nb.args()[1]);
                if is_x(b2) && yes(r, Predicate::real) {
                    *t = pow(x, &mul(&[r.clone(), q.clone()])?)?;
                }
            }
        }
    }
    mul(&margs)
}

/// The terms of the leading order of a sum with their order terms
/// (`Add.extract_leading_order`).
pub fn extract_leading_order(e: &Basic, symbols: &[Basic], point: Option<&[Basic]>) -> PolyResult<Vec<(Basic, Basic)>> {
    let pts: Vec<Basic> = match point {
        Some(p) if !p.is_empty() => p.to_vec(),
        _ => vec![Basic::zero(); symbols.len()],
    };
    let limits: Vec<(Basic, Basic)> = symbols.iter().cloned().zip(pts).collect();
    let mut lst: Vec<(Basic, Basic)> = Vec::new();
    for ef in e.args() {
        let found: Basic = order(ef, &limits)?;
        if !is_order(&found) {
            return Err(PolyError::Type(format!("the order of {} is {}", ef, found)));
        }
        let mut absorbed: bool = false;
        for (_, o) in &lst {
            if contains(o, &found)? == Some(true) && *o != found {
                absorbed = true;
                break;
            }
        }
        if absorbed {
            continue;
        }
        let mut new_lst: Vec<(Basic, Basic)> = vec![(ef.clone(), found.clone())];
        for (t, o) in &lst {
            if contains(&found, o)? == Some(true) && *o != found {
                continue;
            }
            new_lst.push((t.clone(), o.clone()));
        }
        lst = new_lst;
    }
    Ok(lst)
}

// -- contains

/// `Limit(ratio, s, point) != 0` for the ratios `c*s**k` (the others wait
/// for the limits: `NotImplemented`).
fn limit_is_nonzero(ratio: &Basic, s: &Basic, point: &Basic) -> PolyResult<Option<bool>> {
    if !ratio.has(s) {
        return Ok(Some(!is_zero_number(ratio)));
    }
    // the ratios c*s**k*log(s)**m (c free of s): the power of s decides,
    // the power of log(s) when there is none
    let mut k: Rat = Rat::ZERO;
    let mut m: Rat = Rat::ZERO;
    let log_s: Basic = crate::functions::elementary::exponential::log(s)?;
    for f in factors_of(ratio) {
        if !f.has(s) {
            continue;
        }
        let (b, e) = crate::core::expr::as_base_exp(&f)?;
        let Some(r) = e.as_rational().cloned() else {
            return limit_fallback(ratio, s, point);
        };
        if b == *s {
            k += r;
        } else if b == log_s {
            m += r;
        } else {
            return limit_fallback(ratio, s, point);
        }
    }
    let at_infinity: bool = matches!(point.head(), Head::Infinity);
    if !at_infinity && !is_zero_number(point) {
        return limit_fallback(ratio, s, point);
    }
    // at zero the vanishing direction is k > 0 (log(s) -> -oo grows), at
    // infinity k < 0
    let deciding: Rat = if !k.is_zero() { k.clone() } else { m.clone() };
    if deciding.is_zero() {
        return Ok(Some(true));
    }
    let vanishes: bool = if at_infinity {
        rat_is_negative(&deciding)
    } else if !k.is_zero() {
        !rat_is_negative(&k)
    } else {
        rat_is_negative(&m)
    };
    Ok(Some(!vanishes))
}

fn limit_fallback(ratio: &Basic, s: &Basic, point: &Basic) -> PolyResult<Option<bool>> {
    let value: Basic = crate::core::series::limit_lite(ratio, s, point)?;
    Ok(Some(!is_zero_number(&value)))
}

fn power_rule(o: &Basic, other_expr: &Basic, point: &Basic) -> PolyResult<Option<bool>> {
    let (mine, limits) = parts(o);
    let symbol: &Basic = &limits[0].0;
    let other: Basic = dependent_part(other_expr, std::slice::from_ref(symbol))?;
    if other.is_pow() && other.args()[0] == *symbol && mine.is_pow() && mine.args()[0] == *symbol {
        let difference: Basic = sub(&mine.args()[1], &other.args()[1])?;
        let mut rv: Option<bool> = None;
        if yes(point, Predicate::zero) {
            rv = is(&difference, Predicate::nonpositive);
        }
        if yes(point, Predicate::infinite) {
            rv = is(&difference, Predicate::nonnegative);
        }
        if rv.is_some() {
            return Ok(rv);
        }
    }
    Ok(None)
}

/// `True` when `expr` belongs to the order term `o`, `False` when `o`
/// belongs to `expr`, `None` when undecided (`Order.contains`).
pub fn contains(o: &Basic, expr: &Basic) -> PolyResult<Option<bool>> {
    if yes(expr, Predicate::zero) {
        return Ok(Some(true));
    }
    if matches!(expr.head(), Head::NaN) {
        return Ok(Some(false));
    }
    let (mine, limits) = parts(o);
    let point: Basic = limits.first().map(|(_, p)| p.clone()).unwrap_or_else(Basic::zero);
    let variables: Vec<Basic> = limits.iter().map(|(s, _)| s.clone()).collect();
    if is_order(expr) {
        let (theirs, their_limits) = parts(expr);
        if their_limits.iter().any(|(_, p)| *p != point) || limits.iter().any(|(_, p)| *p != point) {
            return Ok(None);
        }
        if theirs == mine {
            // O(1) + O(1), O(1) + O(1, x), ...
            return Ok(Some(expr.args()[1..].iter().all(|t| o.args()[1..].contains(t))));
        }
        if theirs.is_add() {
            for t in theirs.args() {
                if contains(o, t)? != Some(true) {
                    return Ok(Some(false));
                }
            }
            return Ok(Some(true));
        }
        if mine.is_add() && yes(&point, Predicate::zero) {
            for t in mine.args() {
                let ot: Basic = order_inner(t, &limits, true)?;
                if is_order(&ot) && contains(&ot, expr)? == Some(true) {
                    return Ok(Some(true));
                }
            }
            return Ok(Some(false));
        }
        let their_variables: Vec<Basic> = their_limits.iter().map(|(s, _)| s.clone()).collect();
        let common: Vec<Basic> = if !variables.is_empty() && !their_variables.is_empty() {
            variables
                .iter()
                .filter(|s| their_variables.contains(s))
                .cloned()
                .collect()
        } else if !variables.is_empty() {
            variables.clone()
        } else {
            their_variables.clone()
        };
        if common.is_empty() {
            return Ok(None);
        }
        if mine.is_pow() && variables.len() == 1 && variables == their_variables {
            if let Some(rv) = power_rule(o, &theirs, &point)? {
                return Ok(Some(rv));
            }
        }
        let ratio: Basic = crate::simplify::powsimp::powsimp(
            &mul(&[mine.clone(), pow(&theirs, &Basic::negative_one())?])?,
            true,
            crate::simplify::powsimp::Combine::Exp,
            false,
            &crate::simplify::powsimp::count_ops,
        )?;
        let mut r: Option<bool> = None;
        for s in &common {
            let l: Option<bool> = limit_is_nonzero(&ratio, s, &point)?;
            match r {
                None => r = l,
                Some(v) if Some(v) != l => return Ok(None),
                _ => {}
            }
        }
        return Ok(r);
    }
    if mine.is_pow() && variables.len() == 1 {
        if let Some(rv) = power_rule(o, expr, &point)? {
            return Ok(Some(rv));
        }
    }
    let obj: Basic = order_inner(expr, &limits, true)?;
    if !is_order(&obj) {
        return Ok(None);
    }
    contains(o, &obj)
}

// -- substitution

/// `Order._subs` for the substitution of a variable; `None` for the
/// generic substitution of the arguments.
pub fn subs(o: &Basic, old: &Basic, new: &Basic) -> PolyResult<Option<Basic>> {
    if o == old {
        return Ok(Some(new.clone()));
    }
    if !old.is_symbol() {
        return Ok(None);
    }
    let (expr, limits) = parts(o);
    let Some(i) = limits.iter().position(|(s, _)| s == old) else {
        return Ok(None);
    };
    let newexpr: Basic = crate::core::subs::subs(&expr, old, new)?;
    let mut newvars: Vec<Basic> = limits.iter().map(|(s, _)| s.clone()).collect();
    let mut newpt: Vec<Basic> = limits.iter().map(|(_, p)| p.clone()).collect();
    if new.is_symbol() {
        newvars[i] = new.clone();
    } else {
        let syms = new.free_symbols();
        if syms.len() == 1 || syms.contains(old) {
            let var: Basic = if syms.contains(old) {
                limits[i].0.clone()
            } else {
                syms.iter().next().expect("a symbol").clone()
            };
            if let Some(no) = get_o(new)? {
                if is_order(&no) {
                    let op: Basic = parts(&no).1[0].1.clone();
                    if crate::core::series::limit_lite(&no.args()[0], &var, &op)? == limits[i].1 {
                        return Ok(Some(order(&newexpr, &[(var, op)])?));
                    }
                }
            }
            let mut p: Basic = crate::core::subs::subs(new, &var, &limits[i].1)?;
            if p != limits[i].1 {
                let d: Basic = Dummy::new("Dummy", &[]).0;
                let equation: Basic = sub(old, &crate::core::subs::subs(new, &var, &d)?)?;
                let Some(solution) = solve_for(&equation, &d)? else {
                    return Ok(None);
                };
                p = crate::core::series::limit_lite(&solution, old, &limits[i].1)?;
            }
            newvars[i] = var;
            newpt[i] = p;
        } else if !syms.contains(old) {
            newvars.remove(i);
            newpt.remove(i);
        } else {
            return Ok(None);
        }
    }
    let pairs: Vec<(Basic, Basic)> = newvars.into_iter().zip(newpt).collect();
    Ok(Some(order(&newexpr, &pairs)?))
}

/// A solution of `equation == 0` for `d` (`_solve_for`): directly when
/// it is linear in `d`, the first one of SymPy's `solveset` for a
/// rational equation; the other equations wait for `solveset`.
fn solve_for(equation: &Basic, d: &Basic) -> PolyResult<Option<Basic>> {
    let slope: Basic = crate::core::function::derivative(equation, d)?;
    if !slope.has(d) && is(&slope, Predicate::zero) == Some(false) {
        let rest: Basic = crate::core::subs::subs(equation, d, &Basic::zero())?;
        return Ok(Some(mul(&[
            Basic::negative_one(),
            rest,
            pow(&slope, &Basic::negative_one())?,
        ])?));
    }
    if let Some((solutions, _)) = crate::polys::rationalsolve::solveset_rational(equation, d)? {
        if let Some(first) = solutions.first() {
            return Ok(Some(first.clone()));
        }
    }
    Err(not_ported(
        "the substitution of an order variable moving the point (solveset)",
    ))
}

// -- order terms in sums, products and powers

/// `removeO`: the sum without its order terms.
pub fn remove_o(e: &Basic) -> R {
    if is_order(e) {
        return Ok(Basic::zero());
    }
    if !e.is_add() {
        return Ok(e.clone());
    }
    let kept: Vec<Basic> = e.args().iter().filter(|t| !is_order(t)).cloned().collect();
    add(&kept)
}

/// `getO`: the order term of a sum, `None` when there is none.
pub fn get_o(e: &Basic) -> PolyResult<Option<Basic>> {
    if is_order(e) {
        return Ok(Some(e.clone()));
    }
    if !e.is_add() {
        return Ok(None);
    }
    let orders: Vec<Basic> = e.args().iter().filter(|t| is_order(t)).cloned().collect();
    if orders.is_empty() {
        return Ok(None);
    }
    Ok(Some(add(&orders)?))
}

/// `getn`: the order of the order term (`2` for `O(x**2)`), `None` when
/// there is none.
pub fn getn(e: &Basic) -> PolyResult<Option<Basic>> {
    let Some(o) = get_o(e)? else { return Ok(None) };
    if is_order(&o) {
        let oe: &Basic = &o.args()[0];
        if *oe == Basic::one() {
            return Ok(Some(Basic::zero()));
        }
        if oe.is_symbol() {
            return Ok(Some(Basic::one()));
        }
        if oe.is_pow() {
            return Ok(Some(oe.args()[1].clone()));
        }
        if oe.is_mul() {
            for oi in oe.args() {
                if oi.is_symbol() {
                    return Ok(Some(Basic::one()));
                }
                if oi.is_pow() {
                    let syms: Vec<Basic> = oi.atoms(Some(&|n: &Basic| n.is_symbol()));
                    if syms.len() == 1 {
                        let v: Basic = syms[0].clone();
                        let positive: Basic =
                            crate::core::subs::subs(oi, &v, &Dummy::new("x", &[(Predicate::positive, true)]).0)?;
                        if positive.is_pow() && positive.args()[0].is_symbol() {
                            if let Some(r) = positive.args()[1].as_rational() {
                                return Ok(Some(Basic::from_rational(rat_abs(r))));
                            }
                        }
                    }
                }
            }
        }
    }
    Err(not_ported(&format!("not sure of order of {}", o)))
}

fn rat_abs(r: &Rat) -> Rat {
    if rat_is_negative(r) {
        -r.clone()
    } else {
        r.clone()
    }
}

/// A sum with order terms (`Add.flatten`, `_with_orders`): the order
/// terms contained in others dropped, then the terms contained in an
/// order term.
pub fn sum_with_orders(stack: &[Basic]) -> R {
    let mut flat: Vec<Basic> = Vec::new();
    for a in stack {
        if a.is_add() {
            flat.extend(a.args().iter().cloned());
        } else {
            flat.push(a.clone());
        }
    }
    let mut order_factors: Vec<Basic> = Vec::new();
    let mut others: Vec<Basic> = Vec::new();
    for o in flat {
        if is_order(&o) {
            if yes(&o.args()[0], Predicate::zero) {
                continue;
            }
            let mut inside: bool = false;
            for o1 in &order_factors {
                if contains(o1, &o)? == Some(true) {
                    inside = true;
                    break;
                }
            }
            if inside {
                continue;
            }
            let mut kept: Vec<Basic> = vec![o.clone()];
            for o1 in &order_factors {
                if contains(&o, o1)? != Some(true) {
                    kept.push(o1.clone());
                }
            }
            order_factors = kept;
        } else {
            others.push(o);
        }
    }
    let rest: Basic = add(&others)?;
    if matches!(rest.head(), Head::NaN) {
        return Ok(rest);
    }
    let mut coeff: Basic = Basic::zero();
    let mut newseq: Vec<Basic> = Vec::new();
    let rest_terms: Vec<Basic> = if rest.is_add() {
        rest.args().to_vec()
    } else {
        vec![rest.clone()]
    };
    for t in rest_terms {
        if t.is_number_class() {
            coeff = t;
        } else {
            newseq.push(t);
        }
    }
    let mut terms: Vec<Basic> = Vec::new();
    for t in newseq {
        let mut inside: bool = false;
        for o in &order_factors {
            if contains(o, &t)? == Some(true) {
                inside = true;
                break;
            }
        }
        if !inside {
            terms.push(t);
        }
    }
    terms.extend(order_factors.iter().cloned());
    for o in &order_factors {
        if contains(o, &coeff)? == Some(true) {
            coeff = Basic::zero();
            break;
        }
    }
    terms.sort_by(|a, b| a.compare_key().cmp(b.compare_key()));
    if !is_zero_number(&coeff) {
        terms.insert(0, coeff);
    }
    Ok(crate::core::add::raw_add(terms))
}

fn as_expr_variables(
    o: &Basic,
    order_symbols: Option<Vec<(Basic, Basic)>>,
) -> PolyResult<(Basic, Vec<(Basic, Basic)>)> {
    let (expr, limits) = parts(o);
    let Some(order_symbols) = order_symbols else {
        return Ok((expr, limits));
    };
    let point: Vec<Basic> = limits.iter().map(|(_, p)| p.clone()).collect();
    if !order_symbols.iter().all(|(_, p)| *p == order_symbols[0].1) && !point.iter().all(|p| *p == point[0]) {
        return Err(not_ported(&format!(
            "Order at points other than 0 or oo not supported, got {:?} as a point.",
            point
        )));
    }
    if !order_symbols.is_empty() && !point.is_empty() && order_symbols[0].1 != point[0] {
        return Err(not_ported("Multiplying Order at different points is not supported."));
    }
    let mut merged: Vec<(Basic, Basic)> = order_symbols.clone();
    for (s, p) in limits {
        if !merged.iter().any(|(k, _)| *k == s) {
            merged.push((s, p));
        }
    }
    merged.sort_by_cached_key(|(s, _)| crate::core::sorting::sort_key(s));
    Ok((expr, merged))
}

/// A product with order factors (`Mul.flatten`, `_with_orders`): the
/// product of their expressions in the order term of their merged
/// variables.
pub fn product_with_orders(factors: &[Basic]) -> R {
    let mut order_symbols: Option<Vec<(Basic, Basic)>> = None;
    let mut plain: Vec<Basic> = Vec::new();
    let mut flat: Vec<Basic> = Vec::new();
    for f in factors {
        if f.is_mul() {
            flat.extend(f.args().iter().cloned());
        } else {
            flat.push(f.clone());
        }
    }
    for f in flat {
        if is_order(&f) {
            let (e, symbols) = as_expr_variables(&f, order_symbols.take())?;
            order_symbols = Some(symbols);
            plain.push(e);
        } else {
            plain.push(f);
        }
    }
    let product: Basic = mul(&plain)?;
    match order_symbols {
        None => Ok(product),
        Some(symbols) => order(&product, &symbols),
    }
}

/// A power with an order term (`Order._eval_power`): `O(f)**e` is
/// `O(f**e)` for a nonnegative number `e`, `O(f)**O(1)` is `O(f)`; the
/// others stay unevaluated.
pub fn order_power(base: &Basic, exp: &Basic) -> R {
    if exp.is_number_class() && is_zero_number(exp) {
        return Ok(Basic::one());
    }
    if is_order(base) {
        let (expr, limits) = parts(base);
        if exp.is_number_class() && yes(exp, Predicate::nonnegative) {
            return order(&pow(&expr, exp)?, &limits);
        }
        if *exp == order(&Basic::one(), &[])? {
            return Ok(base.clone());
        }
    }
    Ok(crate::core::power::pow_node(base.clone(), exp.clone()))
}

/// Whether `e` is an order term with the variable `x` at zero.
pub fn is_order_at_zero(e: &Basic) -> bool {
    is_order(e) && parts(e).1.iter().all(|(_, p)| is_zero_number(p))
}
