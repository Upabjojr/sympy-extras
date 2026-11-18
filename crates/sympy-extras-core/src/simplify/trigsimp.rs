//! Port of `sympy_extras/simplify/trigsimp.py` (itself ported from
//! SymPy's `sympy.simplify.trigsimp`): `futrig`, the greedy tree of the
//! `fu` rules bottom up (the hyperbolic functions through their
//! trigonometric forms), `trigsimp` with the methods `matching` and `fu`
//! (the others are not ported, as in the Python), and `exptrigsimp`.

use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_coeff_mul, as_powers_dict, factors_of, is_false};
use crate::error::{not_ported, PolyResult};

use super::fu::{
    apply_greedy, first_min, fu, fu_measure, has_function, hyper_as_trig, l_count, rule, tr1, tr10, tr10i, tr11, tr111,
    tr11_half, tr12, tr14, tr15, tr16, tr2, tr22, tr2i, tr3, tr5, tr6, tr8, tr_morrie, Tree, HYPER, TRIG,
};

type R = PolyResult<Basic>;

fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}

fn is_non_expr(e: &Basic) -> bool {
    matches!(e.head(), Head::Tuple) || crate::core::logic::is_boolean(e)
}

/// Whether the tree is made of expressions (`_is_Expr`).
fn is_expr_tree(e: &Basic) -> bool {
    if matches!(e.head(), Head::Derivative) {
        return is_expr_tree(&e.args()[0]);
    }
    if is_non_expr(e) {
        return false;
    }
    e.args().iter().all(is_expr_tree)
}

/// `func` applied to `e`, or to the arguments satisfying `cond` when `e`
/// has non-expression arguments (`_eapply`).
fn eapply(func: &dyn Fn(&Basic) -> R, e: &Basic, cond: Option<&dyn Fn(&Basic) -> bool>) -> R {
    if is_expr_tree(e) || e.args().is_empty() {
        return func(e);
    }
    let args: Vec<Basic> = e
        .args()
        .iter()
        .map(|ei| {
            if !is_non_expr(ei) && cond.map_or(true, |c| c(ei)) {
                eapply(func, ei, None)
            } else {
                Ok(ei.clone())
            }
        })
        .collect::<PolyResult<_>>()?;
    crate::core::function::rebuild_evaluated(e, args)
}

fn has_trig(x: &Basic) -> bool {
    has_function(x, &TRIG)
}

/// The measure of `futrig` (`_lops`).
fn lops(x: &Basic) -> PolyResult<Vec<i64>> {
    Ok(vec![
        l_count(x),
        crate::core::function::count_ops(x)?,
        x.count_nodes() as i64,
        x.args().len() as i64,
        i64::from(x.is_add()),
    ])
}

fn factor(x: &Basic) -> R {
    crate::polys::polytools::factor(x, &[], false, true)
}

fn mexpand(x: &Basic) -> R {
    super::sqrtdenest::mexpand(x)
}

fn expand_mul(x: &Basic) -> R {
    crate::core::function::expand_mul(x, true)
}

fn factor_terms(x: &Basic) -> R {
    crate::core::exprtools::factor_terms(x, false, false, false, true)
}

fn identity(x: &Basic) -> R {
    Ok(x.clone())
}

fn trigs(f: fn(&Basic) -> R) -> impl Fn(&Basic) -> R + 'static {
    move |x: &Basic| eapply(&f, x, Some(&has_trig))
}

fn build_tree() -> Tree {
    Tree::Minimize(vec![
        rule(identity),
        Tree::Chain(vec![
            rule(tr3),
            rule(tr1),
            rule(|x: &Basic| tr12(x, true)),
            rule(trigs(factor)),
            rule(tr2),
            Tree::Minimize(vec![rule(identity), rule(trigs(mexpand))]),
            rule(|x: &Basic| tr2i(x, false)),
            rule(trigs(|i: &Basic| factor(&super::radsimp::normal(i)?))),
            rule(|x: &Basic| super::fu::tr14(x, true)),
            rule(|x: &Basic| tr5(x, 4, false)),
            rule(|x: &Basic| tr10(x, true)),
            rule(|x: &Basic| tr11(x, None)),
            rule(tr11_half),
            rule(|x: &Basic| tr6(x, 4, false)),
            rule(trigs(factor)),
            rule(|x: &Basic| tr14(x, true)),
            Tree::Minimize(vec![rule(identity), rule(trigs(mexpand))]),
            rule(tr10i),
            rule(tr_morrie),
            Tree::Minimize(vec![rule(identity), rule(|x: &Basic| tr8(x, true))]),
            Tree::Minimize(vec![rule(identity), rule(|x: &Basic| tr2i(&tr2(x)?, false))]),
            Tree::Minimize(vec![
                rule(|x: &Basic| eapply(&expand_mul, &tr5(x, 4, false)?, Some(&has_trig))),
                rule(|x: &Basic| eapply(&expand_mul, &tr15(x, 4, false)?, Some(&has_trig))),
            ]),
            Tree::Minimize(vec![
                rule(|x: &Basic| eapply(&expand_mul, &tr6(x, 4, false)?, Some(&has_trig))),
                rule(|x: &Basic| eapply(&expand_mul, &tr16(x, 4, false)?, Some(&has_trig))),
            ]),
            rule(tr111),
            Tree::Minimize(vec![rule(identity), rule(|x: &Basic| tr2i(x, false))]),
            Tree::Minimize(vec![
                rule(identity),
                rule(|x: &Basic| eapply(&expand_mul, &tr22(x, 4, false)?, Some(&has_trig))),
            ]),
            rule(tr1),
            rule(tr2),
            rule(|x: &Basic| tr2i(x, false)),
            Tree::Minimize(vec![
                rule(identity),
                rule(|x: &Basic| eapply(&factor_terms, &tr12(x, true)?, Some(&has_trig))),
            ]),
        ]),
    ])
}

thread_local! {
    static TREE: Tree = build_tree();
}

fn futrig_node(e: &Basic) -> R {
    if is_non_expr(e) || !has_trig(e) {
        return Ok(e.clone());
    }
    let mut e: Basic = e.clone();
    let mut coeff: Option<Basic> = None;
    if e.is_mul() {
        let (independent, dependent): (Vec<Basic>, Vec<Basic>) = factors_of(&e).into_iter().partition(|f| !has_trig(f));
        coeff = Some(mul(&independent)?);
        e = mul(&dependent)?;
    }
    e = TREE.with(|tree| apply_greedy(tree, lops, &e))?;
    match coeff {
        Some(c) => mul(&[c, e]),
        None => Ok(e),
    }
}

/// `_futrig` bottom up, and on the hyperbolic functions through their
/// trigonometric forms with `hyper` (`futrig`).
pub fn futrig(e: &Basic, hyper: bool) -> R {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let old: Basic = e.clone();
    let mut expr: Basic = crate::core::traversal::bottom_up(e, &futrig_node, false)?;
    if hyper && has_function(&expr, &HYPER) {
        let h = hyper_as_trig(&expr)?;
        expr = h.inverse(&crate::core::traversal::bottom_up(&h.expr, &futrig_node, false)?)?;
    }
    if expr != old && expr.is_mul() && expr.args()[0].is_rational_class() {
        let (c, rest) = as_coeff_mul(&expr);
        expr = mul(&[c, rest])?;
    }
    Ok(expr)
}

/// The method of `trigsimp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Matching,
    Fu,
    Groebner,
    Combined,
    Old,
}

/// The expression simplified with trigonometric identities (`trigsimp`):
/// `Matching` is [`futrig`], `Fu` is [`fu`]; the others are not ported.
pub fn trigsimp(e: &Basic, method: Method) -> R {
    match method {
        Method::Matching => futrig(e, true),
        Method::Fu => fu(e, fu_measure),
        other => Err(not_ported(&format!("trigsimp with method={:?}", other))),
    }
}

fn count(x: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(x)
}

fn ops_measure(x: &Basic) -> PolyResult<Vec<i64>> {
    Ok(vec![count(x)?])
}

/// `(sign, x)` with `e = sign*exp(x)` (`signlog`).
fn signlog(x: &Basic, sign: i64) -> PolyResult<Option<(i64, Basic)>> {
    if matches!(x.head(), Head::Exp1) {
        return Ok(Some((sign, Basic::one())));
    }
    if x.is_function(Function::Exp) {
        return Ok(Some((sign, x.args()[0].clone())));
    }
    if x.is_pow() && matches!(x.args()[0].head(), Head::Exp1) {
        return Ok(Some((sign, x.args()[1].clone())));
    }
    if sign == 1 {
        return signlog(&crate::core::mul::neg(x)?, -1);
    }
    Ok(None)
}

fn get_or_zero(d: &[(Basic, Basic)], k: &Basic) -> Basic {
    d.iter()
        .find(|(a, _)| a == k)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(Basic::zero)
}

fn set(d: &mut Vec<(Basic, Basic)>, k: Basic, v: Basic) {
    match d.iter().position(|(a, _)| *a == k) {
        Some(i) => d[i].1 = v,
        None => d.push((k, v)),
    }
}

fn exp_products(b: &Basic) -> R {
    use crate::functions::elementary::hyperbolic::{cosh, sinh, tanh};
    if !b.is_mul() {
        return Ok(b.clone());
    }
    let all: Vec<Basic> = factors_of(b);
    let commutative: Vec<Basic> = all
        .iter()
        .filter(|g| !is_false(g, crate::core::assumptions::Predicate::commutative))
        .cloned()
        .collect();
    let noncommutative: Vec<Basic> = all
        .iter()
        .filter(|g| is_false(g, crate::core::assumptions::Predicate::commutative))
        .cloned()
        .collect();
    if noncommutative.len() > 1 {
        let mut parts: Vec<Basic> = vec![exp_products(&mul(&commutative)?)?];
        parts.extend(noncommutative);
        return mul(&parts);
    }
    let rvd: Vec<(Basic, Basic)> = as_powers_dict(b)?;
    let mut newd: Vec<(Basic, Basic)> = rvd.clone();
    let e_const: Basic = Basic::e();
    let mut ee: Basic = get_or_zero(&rvd, &e_const);
    let sub = crate::core::add::sub;
    let half: Basic = Basic::half();
    for (k, m) in &rvd {
        if !(k.is_add() && k.args().len() == 2) {
            continue;
        }
        let c: Basic = k.args()[0].clone();
        let Some((sign, x)) = signlog(
            &mul(&[
                k.args()[1].clone(),
                crate::core::power::pow(&c, &Basic::negative_one())?,
            ])?,
            1,
        )?
        else {
            continue;
        };
        if crate::core::expr::is_integer_class(&x) && x == Basic::zero() {
            continue;
        }
        let v: Basic = sub(&get_or_zero(&newd, k), m)?;
        set(&mut newd, k.clone(), v);
        let signed_exp: Basic = mul(&[Basic::int(sign), crate::core::power::pow(&e_const, &x)?])?;
        let one_minus: Basic = sub(&Basic::one(), &signed_exp)?;
        if ee == mul(&[Basic::negative_one(), x.clone(), m.clone(), half.clone()])? {
            let v: Basic = sub(&get_or_zero(&newd, &e_const), &ee)?;
            set(&mut newd, e_const.clone(), v);
            ee = Basic::zero();
            let hx: Basic = mul(&[x.clone(), half.clone()])?;
            let key: Basic = if sign == 1 {
                mul(&[Basic::int(2), c.clone(), cosh(&hx)?])?
            } else {
                mul(&[Basic::int(-2), c.clone(), sinh(&hx)?])?
            };
            let v: Basic = crate::core::add::add(&[get_or_zero(&newd, &key), m.clone()])?;
            set(&mut newd, key, v);
        } else if get_or_zero(&newd, &one_minus) == crate::core::mul::neg(m)? {
            newd.retain(|(a, _)| *a != one_minus);
            let hx: Basic = mul(&[x.clone(), half.clone()])?;
            let key: Basic = if sign == 1 {
                mul(&[
                    Basic::negative_one(),
                    c.clone(),
                    crate::core::power::pow(&tanh(&hx)?, &Basic::negative_one())?,
                ])?
            } else {
                mul(&[Basic::negative_one(), c.clone(), tanh(&hx)?])?
            };
            let v: Basic = crate::core::add::add(&[get_or_zero(&newd, &key), m.clone()])?;
            set(&mut newd, key, v);
        } else {
            let key: Basic = crate::core::add::add(&[Basic::one(), signed_exp])?;
            let v: Basic = crate::core::add::add(&[get_or_zero(&newd, &key), m.clone()])?;
            set(&mut newd, key, v);
            let w: Basic = crate::core::add::add(&[get_or_zero(&newd, &c), m.clone()])?;
            set(&mut newd, c, w);
        }
    }
    let powers: Vec<Basic> = newd
        .iter()
        .map(|(k, v)| crate::core::power::pow(k, v))
        .collect::<PolyResult<_>>()?;
    mul(&powers)
}

/// Exponential and trigonometric forms chosen by the number of
/// operations; `cosh`/`sinh`/`tanh` recognised in products with
/// `1 +- exp(x)` (`exptrigsimp`).
pub fn exptrigsimp(e0: &Basic) -> R {
    let hints = crate::core::rewriting::RewriteHints::default();
    let mut both: Vec<Function> = TRIG.to_vec();
    both.extend(HYPER);
    let exp_trig = |b: &Basic| -> R {
        if is_non_expr(b) {
            return Ok(b.clone());
        }
        let mut choices: Vec<Basic> = vec![b.clone()];
        if has_function(b, &both) {
            choices.push(crate::core::rewriting::rewrite(b, &[], "exp", &hints)?);
        }
        choices.push(crate::core::rewriting::rewrite(b, &[], "cos", &hints)?);
        first_min(choices, ops_measure)
    };
    let mut newexpr: Basic = crate::core::traversal::bottom_up(e0, &exp_trig, false)?;
    newexpr = crate::core::traversal::bottom_up(
        &newexpr,
        &|b: &Basic| if is_non_expr(b) { Ok(b.clone()) } else { exp_products(b) },
        false,
    )?;
    if has_function(&newexpr, &HYPER) {
        let h = hyper_as_trig(&newexpr)?;
        newexpr = h.inverse(&tr2i(&h.expr, false)?)?;
    }
    if has_function(&newexpr, &TRIG) {
        newexpr = tr2i(&newexpr, false)?;
    }
    let i: Basic = Basic::imaginary_unit();
    if !(newexpr.has(&i) && !e0.has(&i)) {
        return Ok(newexpr);
    }
    Ok(e0.clone())
}
