//! `refine`, ported from `sympy_extras.core.refine` (SymPy's
//! `assumptions/refine.py` and the `_eval_refine` of the core classes): the
//! tree refined bottom up, each node by the rule of its class (a power of
//! a negated base with an integer exponent, `exp` of rational multiples of
//! `pi*I`, a Boolean decided by `ask`) or by a handler (`Abs`, `Pow`,
//! `atan2`, `re`, `im`, `arg`, `sign`), the result refined again.

use crate::core::add::{add, sub};
use crate::core::ask::{ask, q_applied, Truth};
use crate::core::basic::{Basic, Function, Head};
use crate::core::logic::{and_, boolean};
use crate::core::mul::{div, mul, neg};
use crate::core::power::pow;
use crate::error::PolyResult;
use crate::functions::elementary::complexes::{abs_, sign};

fn qa(name: &str, e: &Basic) -> PolyResult<Basic> {
    q_applied(name, std::slice::from_ref(e))
}

fn ask_in(prop: &Basic, assumptions: &Basic) -> PolyResult<Truth> {
    ask(prop, assumptions, None)
}

fn askq(name: &str, e: &Basic, assumptions: &Basic) -> PolyResult<bool> {
    Ok(ask_in(&qa(name, e)?, assumptions)? == Some(true))
}

/// `ask` without the local assumptions (the global ones only).
fn askg(name: &str, e: &Basic) -> PolyResult<bool> {
    Ok(ask_in(&qa(name, e)?, &boolean(true))? == Some(true))
}

fn both(a: &str, x: &Basic, b: &str, y: &Basic, assumptions: &Basic) -> PolyResult<bool> {
    Ok(ask_in(&and_(&[qa(a, x)?, qa(b, y)?]), assumptions)? == Some(true))
}

fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}

fn is_expr(e: &Basic) -> bool {
    !(crate::core::logic::is_boolean(e) || crate::sets::sets::is_set(e) || matches!(e.head(), Head::Tuple))
}

/// `expr` simplified under the assumptions and the global ones (`refine`).
pub fn refine(expr: &Basic, assumptions: &Basic) -> PolyResult<Basic> {
    refine_inner(expr, assumptions)
}

fn refine_inner(expr: &Basic, assumptions: &Basic) -> PolyResult<Basic> {
    let mut expr: Basic = expr.clone();
    if !expr.args().is_empty() {
        let args: Vec<Basic> = expr
            .args()
            .iter()
            .map(|a| refine_inner(a, assumptions))
            .collect::<PolyResult<_>>()?;
        expr = crate::core::function::rebuild_evaluated(&expr, args)?;
    }
    if let Some(own) = eval_refine(&expr, assumptions)? {
        return Ok(own);
    }
    if !is_expr(&expr) {
        return Ok(expr);
    }
    let new_expr: Option<Basic> = match handler(&expr, assumptions)? {
        Handled::NoHandler => return Ok(expr),
        Handled::Result(r) => r,
    };
    let Some(new_expr) = new_expr else { return Ok(expr) };
    if new_expr == expr {
        return Ok(expr);
    }
    if !is_expr(&new_expr) {
        return Ok(new_expr);
    }
    refine_inner(&new_expr, assumptions)
}

enum Handled {
    NoHandler,
    Result(Option<Basic>),
}

/// The `_eval_refine` of SymPy's classes.
fn eval_refine(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    if expr.is_symbol() {
        return Ok(Some(expr.clone()));
    }
    if func(expr) == Some(Function::Exp) {
        let arg: &Basic = &expr.args()[0];
        if arg.is_mul() {
            let i_oo: Basic = mul(&[Basic::imaginary_unit(), Basic::oo()])?;
            if *arg == i_oo || *arg == neg(&i_oo)? {
                return Ok(Some(Basic::nan()));
            }
            let pi_i: Basic = mul(&[Basic::pi(), Basic::imaginary_unit()])?;
            if let Some(coeff) = crate::core::expr::as_coefficient(arg, &pi_i)? {
                if coeff != Basic::zero() && askg("integer", &mul(&[Basic::int(2), coeff.clone()])?)? {
                    if askg("even", &coeff)? {
                        return Ok(Some(Basic::one()));
                    }
                    if askg("odd", &coeff)? {
                        return Ok(Some(Basic::negative_one()));
                    }
                    let shifted: Basic = add(&[coeff.clone(), Basic::half()])?;
                    if askg("even", &shifted)? {
                        return Ok(Some(neg(&Basic::imaginary_unit())?));
                    }
                    if askg("odd", &shifted)? {
                        return Ok(Some(Basic::imaginary_unit()));
                    }
                }
            }
        }
        return Ok(None);
    }
    if expr.is_pow() {
        let (b, e) = crate::core::expr::as_base_exp(expr)?;
        if askq("integer", &e, assumptions)? && crate::core::printing::could_extract_minus_sign(&b) {
            if askq("even", &e, assumptions)? {
                return Ok(Some(pow(&neg(&b)?, &e)?));
            }
            if askq("odd", &e, assumptions)? {
                return Ok(Some(neg(&pow(&neg(&b)?, &e)?)?));
            }
        }
        return Ok(None);
    }
    if crate::core::logic::is_boolean(expr) {
        return Ok(match ask_in(expr, assumptions)? {
            Some(true) => Some(boolean(true)),
            Some(false) => Some(boolean(false)),
            None => None,
        });
    }
    Ok(None)
}

fn handler(expr: &Basic, assumptions: &Basic) -> PolyResult<Handled> {
    if expr.is_pow() {
        return Ok(Handled::Result(refine_pow(expr, assumptions)?));
    }
    Ok(Handled::Result(match func(expr) {
        Some(Function::Abs) => refine_abs(expr, assumptions)?,
        Some(Function::Atan2) => refine_atan2(expr, assumptions)?,
        Some(Function::Re) => refine_re(expr, assumptions)?,
        Some(Function::Im) => refine_im(expr, assumptions)?,
        Some(Function::Arg) => refine_arg(expr, assumptions)?,
        Some(Function::Sign) => refine_sign(expr, assumptions)?,
        _ => return Ok(Handled::NoHandler),
    }))
}

fn refine_abs(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let arg: &Basic = &expr.args()[0];
    if !is_expr(arg) {
        return Ok(None);
    }
    if askq("real", arg, assumptions)? && ask_in(&qa("negative", arg)?, assumptions)?.map(|v| !v) == Some(true) {
        return Ok(Some(arg.clone()));
    }
    if askq("negative", arg, assumptions)? {
        return Ok(Some(neg(arg)?));
    }
    if arg.is_mul() {
        let mut non_abs: Vec<Basic> = Vec::new();
        let mut in_abs: Vec<Basic> = Vec::new();
        for a in arg.args() {
            if !is_expr(a) {
                continue;
            }
            let i: Basic = refine_inner(&abs_(a)?, assumptions)?;
            if func(&i) == Some(Function::Abs) {
                in_abs.push(i.args()[0].clone());
            } else if is_expr(&i) {
                non_abs.push(i);
            }
        }
        return Ok(Some(mul(&[mul(&non_abs)?, abs_(&mul(&in_abs)?)?])?));
    }
    Ok(None)
}

fn add_terms(e: &Basic) -> (Basic, Vec<Basic>) {
    let (coeff, rest) = crate::core::expr::as_coeff_add(e);
    let terms: Vec<Basic> = if rest.is_add() {
        rest.args().to_vec()
    } else if rest == Basic::zero() {
        Vec::new()
    } else {
        vec![rest]
    };
    (coeff, terms)
}

fn refine_pow(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let (base, exp) = (&expr.args()[0], &expr.args()[1]);
    if func(base) == Some(Function::Abs)
        && askq("real", &base.args()[0], assumptions)?
        && askq("even", exp, assumptions)?
    {
        return Ok(Some(pow(&base.args()[0], exp)?));
    }
    if askq("real", base, assumptions)? {
        if crate::core::assumptions::is_number(base) {
            if askq("even", exp, assumptions)? {
                return Ok(Some(pow(&abs_(base)?, exp)?));
            }
            if askq("odd", exp, assumptions)? {
                return Ok(Some(mul(&[sign(base)?, pow(&abs_(base)?, exp)?])?));
            }
        }
        if exp.as_rational().is_some() && base.is_pow() {
            let inner: Basic = mul(&[base.args()[1].clone(), exp.clone()])?;
            return Ok(Some(pow(&abs_(&base.args()[0])?, &inner)?));
        }
        if *base == Basic::negative_one() && exp.is_add() {
            let old: Basic = expr.clone();
            let (coeff, terms_all) = add_terms(exp);
            let mut terms: Vec<Basic> = Vec::new();
            for t in terms_all {
                if !terms.contains(&t) {
                    terms.push(t);
                }
            }
            let initial: usize = terms.len();
            let mut even_terms: Vec<Basic> = Vec::new();
            for t in &terms {
                if askq("even", t, assumptions)? {
                    even_terms.push(t.clone());
                }
            }
            let mut odd_terms: Vec<Basic> = Vec::new();
            for t in &terms {
                if !even_terms.contains(t) && askq("odd", t, assumptions)? {
                    odd_terms.push(t.clone());
                }
            }
            terms.retain(|t| !even_terms.contains(t) && !odd_terms.contains(t));
            let two: Basic = Basic::int(2);
            let new_coeff: Basic = if odd_terms.len() % 2 == 1 {
                crate::core::modulo::modulo(&add(&[coeff.clone(), Basic::one()])?, &two)?
            } else {
                crate::core::modulo::modulo(&coeff, &two)?
            };
            let mut current: Basic = expr.clone();
            if new_coeff != coeff || terms.len() < initial {
                let mut all: Vec<Basic> = terms.clone();
                all.push(new_coeff);
                current = pow(base, &add(&all)?)?;
            }
            if !current.is_pow() {
                return Ok(Some(current));
            }
            let mut e2: Basic = mul(&[two.clone(), current.args()[1].clone()])?;
            if askq("even", &e2, assumptions)? && crate::core::printing::could_extract_minus_sign(&e2) {
                e2 = mul(&[e2, current.args()[0].clone()])?;
            }
            if e2.is_add() {
                // Add.as_two_terms: the first term and the sum of the others
                let mut i: Basic = e2.args()[0].clone();
                let rest: &[Basic] = &e2.args()[1..];
                let p: Basic = if rest.len() == 1 {
                    rest[0].clone()
                } else {
                    Basic::new(Head::Add, rest.to_vec())
                };
                if p.is_pow() && p.args()[0] == Basic::negative_one() && askq("integer", &p.args()[1], assumptions)? {
                    i = div(&add(&[i, Basic::one()])?, &two)?;
                    let cb: &Basic = &current.args()[0];
                    let pe: &Basic = &p.args()[1];
                    if askq("even", &i, assumptions)? {
                        return Ok(Some(pow(cb, pe)?));
                    }
                    if askq("odd", &i, assumptions)? {
                        return Ok(Some(pow(cb, &add(&[pe.clone(), Basic::one()])?)?));
                    }
                    return Ok(Some(pow(cb, &add(&[pe.clone(), i])?)?));
                }
            }
            if old != current {
                return Ok(Some(current));
            }
        }
    }
    Ok(None)
}

fn refine_atan2(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let (y, x) = (&expr.args()[0], &expr.args()[1]);
    if !is_expr(y) || !is_expr(x) {
        return Ok(None);
    }
    let atan_yx = || -> PolyResult<Basic> { crate::functions::elementary::inverse::atan(&div(y, x)?) };
    let pi: Basic = Basic::pi();
    if both("real", y, "positive", x, assumptions)? {
        return Ok(Some(atan_yx()?));
    }
    if both("negative", y, "negative", x, assumptions)? {
        return Ok(Some(sub(&atan_yx()?, &pi)?));
    }
    if both("positive", y, "negative", x, assumptions)? {
        return Ok(Some(add(&[atan_yx()?, pi])?));
    }
    if both("zero", y, "negative", x, assumptions)? {
        return Ok(Some(pi));
    }
    if both("positive", y, "zero", x, assumptions)? {
        return Ok(Some(div(&pi, &Basic::int(2))?));
    }
    if both("negative", y, "zero", x, assumptions)? {
        return Ok(Some(neg(&div(&pi, &Basic::int(2))?)?));
    }
    if both("zero", y, "zero", x, assumptions)? {
        return Ok(Some(Basic::nan()));
    }
    Ok(Some(expr.clone()))
}

fn refine_reim(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let expanded: Basic =
        crate::core::function::expand_with_hints(expr, Some(&crate::core::function::Hints::DEFAULT), Some(true))?;
    if expanded != *expr {
        let refined: Basic = refine_inner(&expanded, assumptions)?;
        if refined != expanded {
            return Ok(Some(refined));
        }
    }
    Ok(None)
}

fn refine_re(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let arg: &Basic = &expr.args()[0];
    if !is_expr(arg) {
        return Ok(None);
    }
    if askq("real", arg, assumptions)? {
        return Ok(Some(arg.clone()));
    }
    if askq("imaginary", arg, assumptions)? {
        return Ok(Some(Basic::zero()));
    }
    refine_reim(expr, assumptions)
}

fn refine_im(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let arg: &Basic = &expr.args()[0];
    if !is_expr(arg) {
        return Ok(None);
    }
    if askq("real", arg, assumptions)? {
        return Ok(Some(Basic::zero()));
    }
    if askq("imaginary", arg, assumptions)? {
        return Ok(Some(mul(&[neg(&Basic::imaginary_unit())?, arg.clone()])?));
    }
    refine_reim(expr, assumptions)
}

fn refine_arg(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let rg: &Basic = &expr.args()[0];
    if !is_expr(rg) {
        return Ok(None);
    }
    if askq("positive", rg, assumptions)? {
        return Ok(Some(Basic::zero()));
    }
    if askq("negative", rg, assumptions)? {
        return Ok(Some(Basic::pi()));
    }
    Ok(None)
}

fn refine_sign(expr: &Basic, assumptions: &Basic) -> PolyResult<Option<Basic>> {
    let arg: &Basic = &expr.args()[0];
    if !is_expr(arg) {
        return Ok(None);
    }
    if askq("zero", arg, assumptions)? {
        return Ok(Some(Basic::zero()));
    }
    if askg("real", arg)? {
        if askq("positive", arg, assumptions)? {
            return Ok(Some(Basic::one()));
        }
        if askq("negative", arg, assumptions)? {
            return Ok(Some(Basic::negative_one()));
        }
    }
    if askg("imaginary", arg)? {
        if let Some((_, arg_im)) = crate::functions::elementary::complexes::as_real_imag(arg, true, None)? {
            if askq("positive", &arg_im, assumptions)? {
                return Ok(Some(Basic::imaginary_unit()));
            }
            if askq("negative", &arg_im, assumptions)? {
                return Ok(Some(neg(&Basic::imaginary_unit())?));
            }
        }
    }
    Ok(Some(expr.clone()))
}
