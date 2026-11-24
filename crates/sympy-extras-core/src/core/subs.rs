//! Port of the substitution of `sympy_extras/core`: `Basic.subs` and
//! `_subs` (`basic.py`), the replacement of sums and products among the
//! terms and factors (`Add._subs_partial`, `Mul._subs_partial`), SymPy's
//! rules for powers (`power_subs` of `power.py`, with `Symbol._subs` and
//! `exp._eval_subs`), and the bound variables of `Derivative` and `Subs`
//! (`function.py`).

use std::collections::HashSet;

use crate::error::PolyResult;
use crate::functions::elementary::exponential::{exp, log, log_base};

use super::add::add;
use super::assumptions::Predicate;
use super::basic::{Basic, Function, Head};
use super::expr::{as_base_exp, as_coeff_mul_tuple, is_integer_class, is_rational_value, is_true};
use super::function::{rebuild_evaluated, subs_node};
use super::mul::mul;
use super::power::pow;
use super::sorting::sort_key;

/// The tree with `old` replaced by `new`, the rebuilt nodes evaluated
/// (`e.subs(old, new)`).
pub fn subs(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Basic> {
    subs_node_rec(e, old, new)
}

/// The substitutions applied in sequence (`e.subs([(old, new), ...])`).
pub fn subs_pairs(e: &Basic, pairs: &[(Basic, Basic)]) -> PolyResult<Basic> {
    let mut result: Basic = e.clone();
    for (old, new) in pairs {
        result = subs(&result, old, new)?;
    }
    Ok(result)
}

/// The substitutions of a mapping, in SymPy's order: the keys with more
/// nodes first, then by the sort key (`e.subs({old: new, ...})`).
pub fn subs_mapping(e: &Basic, pairs: &[(Basic, Basic)]) -> PolyResult<Basic> {
    let mut ordered: Vec<(Basic, Basic)> = pairs.to_vec();
    ordered.sort_by_cached_key(|(old, _)| (std::cmp::Reverse(old.count_nodes()), sort_key(old)));
    subs_pairs(e, &ordered)
}

/// `_subs`.
fn subs_node_rec(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Basic> {
    if e == old {
        // the node itself when it equals new too: Python's new is then
        // (usually) the same object, and the parents are not rebuilt
        if e == new {
            return Ok(e.clone());
        }
        return Ok(new.clone());
    }
    if e.is_symbol() {
        // Symbol._subs: x is x**1 for a power to replace
        if old.is_pow() {
            if let Some(replaced) = power_subs(e, &Basic::one(), old, new, None)? {
                return Ok(replaced);
            }
        }
        return Ok(e.clone());
    }
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    if matches!(e.head(), Head::RootSum) {
        return crate::polys::rootoftools::subs(e, old, new);
    }
    if matches!(e.head(), Head::Order) {
        if let Some(r) = crate::core::order::subs(e, old, new)? {
            return Ok(r);
        }
    }
    if matches!(e.head(), Head::TranscendentalRoot) {
        // the variable is bound: a substitution leaves the root alone
        return Ok(if e == old { new.clone() } else { e.clone() });
    }
    if matches!(e.head(), Head::IndexedRoot) {
        // the variable is bound: a substitution of it leaves the root alone
        let (f, t, k) = crate::polys::cad::indexedroot::parts(e);
        if old == t {
            return Ok(e.clone());
        }
        let replaced: Basic = subs_node_rec(f, old, new)?;
        if replaced.ptr_eq(f) {
            return Ok(e.clone());
        }
        return crate::polys::cad::indexedroot::indexed_root(&replaced, t, k);
    }
    if matches!(e.head(), Head::ComplexRootOf(_)) {
        // a number: no substitution of its bound variable changes it
        return Ok(e.clone());
    }
    if crate::assumptions::quantifiers::is_quantifier(e) {
        return crate::assumptions::quantifiers::quantifier_subs(e, old, new, &|f: &Basic| subs_node_rec(f, old, new));
    }
    if matches!(e.head(), Head::ConditionSet) {
        // ConditionSet._subs: the condition first; a parameter value which
        // makes it false gives the empty set, even when it makes the base
        // set invalid (SymPy rebuilds the base set first and raises, issue
        // #25); a bound symbol is not replaced by a value
        let (sym, condition, base) = (&e.args()[0], &e.args()[1], &e.args()[2]);
        let bound: Vec<Basic> = if matches!(sym.head(), Head::Tuple) {
            sym.args().to_vec()
        } else {
            vec![sym.clone()]
        };
        if bound.contains(old) && !matches!(new.head(), Head::Symbol(_) | Head::Dummy(_, _)) {
            return Ok(e.clone());
        }
        let condition: Basic = subs_node_rec(condition, old, new)?;
        if condition == crate::core::logic::false_() {
            return Ok(crate::sets::sets::empty_set());
        }
        let args: Vec<Basic> = vec![subs_node_rec(sym, old, new)?, condition, subs_node_rec(base, old, new)?];
        return rebuild_evaluated(e, args);
    }
    if let Some(replaced) = subs_partial(e, old, new)? {
        return Ok(replaced);
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(subs_node_rec(a, old, new)?);
    }
    if args.iter().zip(e.args()).all(|(a, b)| a.ptr_eq(b)) {
        return Ok(e.clone());
    }
    rebuild_evaluated(e, args)
}

/// The rule of the class of `e`, `None` for the substitution in the
/// arguments (`_subs_partial`).
fn subs_partial(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Option<Basic>> {
    match e.head() {
        Head::Add | Head::Mul => {
            if old.head().is_atom() || std::mem::discriminant(old.head()) != std::mem::discriminant(e.head()) {
                return Ok(None);
            }
            let mut remaining: Vec<Basic> = e.args().to_vec();
            for term in old.args() {
                match remaining.iter().position(|t| t == term) {
                    Some(k) => {
                        remaining.remove(k);
                    }
                    None => return Ok(None),
                }
            }
            let mut parts: Vec<Basic> = vec![new.clone()];
            for t in &remaining {
                parts.push(subs_node_rec(t, old, new)?);
            }
            Ok(Some(if e.is_add() { add(&parts)? } else { mul(&parts)? }))
        }
        Head::Pow => power_subs(&e.args()[0], &e.args()[1], old, new, None),
        Head::Function(Function::Exp) => {
            let mut target: Basic = old.clone();
            if old.is_pow() {
                let rewritten: Basic = exp(&mul(&[old.args()[1].clone(), log(&old.args()[0])?])?)?;
                if !rewritten.is_function(Function::Exp) {
                    return Ok(None);
                }
                target = rewritten;
            }
            if target.is_function(Function::Exp) || matches!(target.head(), Head::Exp1) {
                return power_subs(&Basic::e(), &e.args()[0], &target, new, None);
            }
            Ok(None)
        }
        Head::Derivative => derivative_subs(e, old, new),
        Head::Subs => subs_subs(e, old, new),
        Head::Sum | Head::Product => crate::concrete::expr_with_limits::subs_partial(e, old, new),
        Head::Integral | Head::NonElementaryIntegral => crate::integrals::integral::subs_partial(e, old, new),
        _ => Ok(None),
    }
}

/// `_diff_wrt`: a symbol or an applied undefined function.
fn diff_wrt(e: &Basic) -> bool {
    e.is_symbol() || matches!(e.head(), Head::Undefined(_))
}

/// `Derivative._subs_partial`: a variable replaced by what is not a symbol,
/// a captured free symbol, or a renaming onto a symbol of the expression
/// make a `Subs`.
fn derivative_subs(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Option<Basic>> {
    let orders: Vec<(Basic, Basic)> = crate::core::function::derivative_orders(e);
    let wrt: Vec<Basic> = orders.iter().map(|(v, _)| v.clone()).collect();
    let expr: &Basic = &e.args()[0];
    if old.is_symbol() && wrt.contains(old) && !diff_wrt(new) {
        return Ok(Some(subs_node(e, vec![old.clone()], vec![new.clone()])));
    }
    let new_expr: Basic = subs_node_rec(expr, old, new)?;
    if new_expr != *expr && old.is_symbol() {
        let forbidden: HashSet<Basic> = expr.free_symbols().into_iter().filter(|s| wrt.contains(s)).collect();
        let old_free = old.free_symbols();
        if new
            .free_symbols()
            .iter()
            .any(|s| !old_free.contains(s) && forbidden.contains(s))
        {
            return Ok(Some(subs_node(e, vec![old.clone()], vec![new.clone()])));
        }
    }
    let mut renamed: Vec<(Basic, Basic)> = Vec::new();
    for v in &wrt {
        renamed.push((v.clone(), subs_node_rec(v, old, new)?));
    }
    if renamed.iter().any(|(v, nv)| nv != v) && old.is_symbol() {
        let atoms: HashSet<Basic> = expr.preorder().into_iter().filter(|n| n.is_symbol()).collect();
        if renamed.iter().any(|(v, nv)| nv != v && atoms.contains(nv)) {
            return Ok(Some(subs_node(e, vec![old.clone()], vec![new.clone()])));
        }
    }
    if !renamed.iter().all(|(_, nv)| nv.is_symbol()) {
        return Ok(None);
    }
    // (a symbolic order is substituted too: Derivative(f(x), (x,
    // n)).subs(n, 2) is Derivative(f(x), (x, 2)))
    let mut new_orders: Vec<(Basic, Basic)> = Vec::new();
    for ((_, nv), (_, n)) in renamed.into_iter().zip(&orders) {
        new_orders.push((nv, subs_node_rec(n, old, new)?));
    }
    Ok(Some(crate::core::function::derivative_node_orders(
        &new_expr,
        &new_orders,
    )?))
}

/// `Subs._subs_partial`: the bound variables respected.
fn subs_subs(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Option<Basic>> {
    let (expr, variables, point) = (&e.args()[0], e.args()[1].args().to_vec(), e.args()[2].args().to_vec());
    let mut pt: Vec<Basic> = point.clone();
    if old.is_symbol() {
        if let Some(i) = variables.iter().position(|v| v == old) {
            if new.is_symbol() && !e.args().iter().any(|a| a.has(new)) {
                let mut rule = std::collections::HashMap::new();
                rule.insert(old.clone(), new.clone());
                return Ok(Some(e.xreplace(&rule)));
            }
            for p in pt.iter_mut().skip(i) {
                *p = subs_node_rec(p, old, new)?;
            }
            return Ok(Some(subs_node(expr, variables, pt)));
        }
    }
    let mut v: Vec<Basic> = Vec::new();
    for var in &variables {
        v.push(subs_node_rec(var, old, new)?);
    }
    if v != variables {
        if !old.is_symbol() {
            return Ok(None);
        }
        let mut vs: Vec<Basic> = variables;
        vs.push(old.clone());
        pt.push(new.clone());
        return Ok(Some(subs_node(expr, vs, pt)));
    }
    let mut variables: Vec<Basic> = variables;
    let mut body: Basic = expr.clone();
    let free = new.free_symbols();
    let captured: Vec<Basic> = variables.iter().filter(|v| free.contains(*v)).cloned().collect();
    if !captured.is_empty() && body.has(old) {
        // the new expression has a bound variable, which would capture it
        // (Subs(x, y, 2).subs(x, y) is not Subs(y, y, 2); sympy/sympy#31275,
        // V59): the bound variables are renamed first
        let mut renaming = std::collections::HashMap::new();
        for var in &captured {
            let name: String = match var.head() {
                Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
                _ => var.to_string(),
            };
            renaming.insert(var.clone(), crate::core::symbol::Dummy::new(&name, &[]).0);
        }
        variables = variables
            .iter()
            .map(|v| renaming.get(v).cloned().unwrap_or_else(|| v.clone()))
            .collect();
        body = body.xreplace(&renaming);
    }
    let new_expr: Basic = subs_node_rec(&body, old, new)?;
    let mut points: Vec<Basic> = Vec::new();
    for p in &pt {
        points.push(subs_node_rec(p, old, new)?);
    }
    Ok(Some(subs_node(&new_expr, variables, points)))
}

// ---------------------------------------------------------------------
// powers

/// Whether a node of the tree is a symbol (`e.has(Symbol)`).
fn has_symbol(e: &Basic) -> bool {
    e.preorder().iter().any(Basic::is_symbol)
}

/// SymPy's `e.as_independent(Symbol, as_Add=False)` (`_independent_of_symbols`).
fn independent_of_symbols(e: &Basic) -> PolyResult<(Basic, Basic)> {
    if e.is_mul() {
        let free: Vec<Basic> = e.args().iter().filter(|f| !has_symbol(f)).cloned().collect();
        let dependent: Vec<Basic> = e.args().iter().filter(|f| has_symbol(f)).cloned().collect();
        return Ok((mul(&free)?, mul(&dependent)?));
    }
    if has_symbol(e) {
        return Ok((Basic::one(), e.clone()));
    }
    Ok((e.clone(), Basic::one()))
}

/// SymPy's `_check` of `Pow._eval_subs` (`_combines`).
fn combines(coeff1: &Basic, coeff2: &Basic, old: &Basic) -> PolyResult<(bool, Basic)> {
    let p: Basic = mul(&[coeff1.clone(), pow(coeff2, &Basic::negative_one())?])?;
    if is_integer_class(&p) {
        return Ok((true, p));
    }
    let (b, e) = as_base_exp(old)?;
    let ok: bool = (is_true(&b, Predicate::positive) && is_true(&e, Predicate::real))
        || (is_true(&b, Predicate::nonnegative) && is_true(&e, Predicate::nonnegative));
    Ok((ok, p))
}

/// `(base, exponent)` of a power or of `exp(a)` (`_power_view`).
fn power_view(e: &Basic) -> Option<(Basic, Basic)> {
    if e.is_pow() {
        return Some((e.args()[0].clone(), e.args()[1].clone()));
    }
    if e.is_function(Function::Exp) {
        return Some((Basic::e(), e.args()[0].clone()));
    }
    None
}

/// `(base**exponent).subs(old, new)` with SymPy's rules for powers
/// (`power_subs`); `None` for the plain substitution in the arguments.
pub fn power_subs(
    base: &Basic,
    exponent: &Basic,
    old: &Basic,
    new: &Basic,
    old_is_power: Option<bool>,
) -> PolyResult<Option<Basic>> {
    if old == base {
        return Ok(Some(pow(new, &subs_node_rec(exponent, old, new)?)?));
    }
    let view: Option<(Basic, Basic)> = if old_is_power != Some(false) {
        power_view(old)
    } else {
        None
    };
    if let Some((ob, oe)) = &view {
        if old.is_pow() || matches!(base.head(), Head::Exp1) {
            if exponent == oe {
                let l: Basic = log_base(base, ob)?;
                if l.is_number_class() {
                    return Ok(Some(pow(new, &l)?));
                }
            }
            if base == ob {
                if !exponent.is_add() {
                    let (c1, t1) = independent_of_symbols(exponent)?;
                    let (c2, t2) = independent_of_symbols(oe)?;
                    if t1 == t2 {
                        let (ok, p) = combines(&c1, &c2, old)?;
                        if ok {
                            return Ok(Some(pow(new, &p)?));
                        }
                    }
                } else {
                    let (c2n, t2n) = as_coeff_mul_tuple(oe);
                    let mut new_l: Vec<Basic> = Vec::new();
                    let mut others: Vec<Basic> = Vec::new();
                    for a in exponent.args() {
                        let newa: Basic = subs_node_rec(a, old, new)?;
                        let (c1n, t1n) = as_coeff_mul_tuple(&newa);
                        if t1n == t2n {
                            let (ok, p) = combines(&c1n, &c2n, old)?;
                            if ok {
                                new_l.push(pow(new, &p)?);
                                continue;
                            }
                        }
                        others.push(newa);
                    }
                    if !new_l.is_empty() {
                        let expo: Basic = add(&others)?;
                        new_l.push(if is_integer_class(&expo) && is_rational_value(&expo, 1) {
                            base.clone()
                        } else {
                            Basic::new(Head::Pow, vec![base.clone(), expo])
                        });
                        return Ok(Some(mul(&new_l)?));
                    }
                }
            }
        }
    }
    if let Some((ob, oe)) = power_view(old) {
        let exp_like: bool = old.is_function(Function::Exp) || matches!(ob.head(), Head::Exp1);
        if exp_like && is_true(exponent, Predicate::extended_real) && is_true(base, Predicate::positive) {
            let (c1, t1) = independent_of_symbols(&oe)?;
            let (c2, t2) = independent_of_symbols(&mul(&[exponent.clone(), log(base)?])?)?;
            if t1 == t2 {
                let (ok, p) = combines(&c1, &c2, old)?;
                if ok {
                    return Ok(Some(pow(new, &p)?));
                }
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    #[test]
    fn docstring_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let y: Basic = Symbol::new("y", &[]).0;
        let z: Basic = Symbol::new("z", &[]).0;
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        let s: Basic = add(&[x.clone(), y.clone()]).unwrap();
        assert_eq!(show(subs(&s, &x, &Basic::one())), "y + 1");
        let x4: Basic = pow(&x, &Basic::int(4)).unwrap();
        assert_eq!(show(subs(&x4, &pow(&x, &Basic::int(2)).unwrap(), &y)), "y**2");
        let three_x_plus_one: Basic = add(&[mul(&[Basic::int(3), x.clone()]).unwrap(), Basic::one()]).unwrap();
        assert_eq!(
            show(subs(&exp(&three_x_plus_one).unwrap(), &exp(&x).unwrap(), &y)),
            "E*y**3"
        );
        let xyz: Basic = add(&[x.clone(), y.clone(), z.clone()]).unwrap();
        assert_eq!(show(subs(&xyz, &s, &z)), "2*z");
        assert_eq!(show(subs(&pow(&x, &Basic::negative_one()).unwrap(), &x, &y)), "1/y");
    }
}
