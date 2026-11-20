//! Port of `sympy_extras/concrete/expr_with_limits.py`: `Sum(f, (i, a, b),
//! ...)` and `Product(f, (i, a, b), ...)` (the heads `Head::Sum`,
//! `Head::Product`, the function then a `Tuple(i, a, b)` per limit),
//! unevaluated, with SymPy's constructors `sum_` and `product_` (nested
//! nodes of the same kind merged, the `Piecewise` depending on the
//! variables folded); the bound variables (`free_symbols`, substitution)
//! and the printing. `doit` needs `concrete/summations.py` and
//! `products.py` (not ported).

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::core::assumptions::{facts_of, FactList, Predicate};
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;

/// The limits `(variable, lower, upper)` of a node.
pub fn limits(e: &Basic) -> Vec<(Basic, Basic, Basic)> {
    e.args()[1..]
        .iter()
        .map(|l| (l.args()[0].clone(), l.args()[1].clone(), l.args()[2].clone()))
        .collect()
}

/// Whether a node is a `Sum` or a `Product`.
pub fn is_expr_with_limits(e: &Basic) -> bool {
    matches!(e.head(), Head::Sum | Head::Product)
}

fn check_limit(l: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    if !matches!(l.head(), Head::Tuple) || l.args().len() != 3 {
        return Err(PolyError::Value(format!(
            "a limit is (variable, lower, upper), not {}",
            l
        )));
    }
    let (v, a, b) = (&l.args()[0], &l.args()[1], &l.args()[2]);
    if !v.is_symbol() {
        return Err(PolyError::Type(format!(
            "the variable of a limit is a symbol, not {}",
            v
        )));
    }
    if crate::core::logic::is_boolean(a)
        || crate::core::logic::is_boolean(b)
        || matches!(a.head(), Head::Tuple)
        || matches!(b.head(), Head::Tuple)
    {
        return Err(PolyError::Type("the bounds of a limit are expressions".to_string()));
    }
    Ok((v.clone(), a.clone(), b.clone()))
}

/// The node itself (`cls(function, *limits)`).
pub fn node(head: Head, function: &Basic, lims: &[(Basic, Basic, Basic)]) -> PolyResult<Basic> {
    if lims.is_empty() {
        let name: &str = if matches!(head, Head::Sum) { "Sum" } else { "Product" };
        return Err(PolyError::Value(format!("{} requires limits", name)));
    }
    let mut args: Vec<Basic> = vec![function.clone()];
    for (v, a, b) in lims {
        args.push(Basic::new(Head::Tuple, vec![v.clone(), a.clone(), b.clone()]));
    }
    Ok(Basic::new(head, args))
}

/// `ExprWithLimits._evaluated`: the node of the new arguments.
pub fn evaluated(head: Head, args: &[Basic]) -> R {
    let lims: Vec<(Basic, Basic, Basic)> = args[1..].iter().map(check_limit).collect::<PolyResult<_>>()?;
    node(head, &args[0], &lims)
}

/// SymPy's constructor (`_common_new`).
fn common_new(head: Head, function: &Basic, given: &[Basic]) -> R {
    let mut lims: Vec<(Basic, Basic, Basic)> = given.iter().map(check_limit).collect::<PolyResult<_>>()?;
    let mut f: Basic = function.clone();
    if matches!(f.head(), Head::NaN) {
        return Ok(f);
    }
    let is_sum: bool = matches!(head, Head::Sum);
    while (is_sum && matches!(f.head(), Head::Sum)) || (!is_sum && matches!(f.head(), Head::Product)) {
        let mut inner: Vec<(Basic, Basic, Basic)> = limits(&f);
        inner.extend(lims);
        lims = inner;
        f = f.args()[0].clone();
    }
    let variables: Vec<Basic> = lims.iter().map(|l| l.0.clone()).collect();
    let is_piecewise = |n: &Basic| matches!(n.head(), Head::Piecewise);
    if f.preorder().iter().any(is_piecewise) {
        let mut reps: Vec<(Basic, Basic)> = Vec::new();
        for p in f.preorder() {
            if is_piecewise(&p) && !variables.iter().any(|v| p.has(v)) && !reps.iter().any(|(k, _)| *k == p) {
                reps.push((p.clone(), crate::core::symbol::Dummy::new("Dummy", &[]).0));
            }
        }
        let forward: HashMap<Basic, Basic> = reps.iter().cloned().collect();
        let g: Basic = crate::functions::elementary::piecewise::piecewise_fold(&f.xreplace(&forward), None)?;
        let back: HashMap<Basic, Basic> = reps.into_iter().map(|(k, v)| (v, k)).collect();
        f = g.xreplace(&back);
    }
    node(head, &f, &lims)
}

/// `Sum(function, *limits)` as SymPy's constructor builds it (`sum_`).
pub fn sum_(function: &Basic, lims: &[Basic]) -> R {
    common_new(Head::Sum, function, lims)
}

/// `Product(function, *limits)` as SymPy's constructor builds it (`product_`).
pub fn product_(function: &Basic, lims: &[Basic]) -> R {
    common_new(Head::Product, function, lims)
}

/// The free symbols: those of the function but the variables, and those
/// of the bounds (`free_symbols`).
pub fn free_symbols(e: &Basic) -> BTreeSet<Basic> {
    let mut symbols: BTreeSet<Basic> = e.args()[0].free_symbols();
    for (v, a, b) in limits(e) {
        symbols.remove(&v);
        symbols.extend(a.free_symbols());
        symbols.extend(b.free_symbols());
    }
    symbols
}

/// The variables (`bound_symbols`).
pub fn bound_symbols(e: &Basic) -> Vec<Basic> {
    limits(e).into_iter().map(|l| l.0).collect()
}

/// Whether every bound is finite, `None` when not known (`has_finite_limits`).
pub fn has_finite_limits(e: &Basic) -> Option<bool> {
    let mut none: bool = false;
    for (v, a, b) in limits(e) {
        if is(&a, Predicate::infinite) == Some(true) || is(&b, Predicate::infinite) == Some(true) {
            return Some(false);
        }
        if (is(&a, Predicate::infinite).is_none() || is(&b, Predicate::infinite).is_none())
            && is(&v, Predicate::infinite).is_none()
        {
            none = true;
        }
    }
    if none {
        None
    } else {
        Some(true)
    }
}

/// The facts (`_eval_facts`): the commutativity of the function, zero for
/// a sum of zero.
pub fn facts(e: &Basic) -> FactList {
    let mut found: FactList = FactList::new();
    if let Some(c) = facts_of(&e.args()[0]).get(&Predicate::commutative) {
        found.set(Predicate::commutative, *c);
    }
    if matches!(e.head(), Head::Sum) && is(&e.args()[0], Predicate::zero) == Some(true) {
        found.set(Predicate::zero, true);
    }
    found
}

/// `Sum(f, (i, a, b))` printed.
pub fn print(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let name: &str = if matches!(e.head(), Head::Sum) {
        "Sum"
    } else {
        "Product"
    };
    let lims: Vec<String> = limits(e)
        .iter()
        .map(|(v, a, b)| format!("({}, {}, {})", doprint(v), doprint(a), doprint(b)))
        .collect();
    format!("{}({}, {})", name, doprint(&e.args()[0]), lims.join(", "))
}

/// The substitution with the variables bound (`_subs_partial`).
pub fn subs_partial(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Option<Basic>> {
    let mut func: Basic = e.args()[0].clone();
    let mut lims: Vec<(Basic, Basic, Basic)> = limits(e);
    lims.reverse();
    let old_symbols: BTreeSet<Basic> = old.free_symbols();
    let free: BTreeSet<Basic> = free_symbols(e);
    let subs = |x: &Basic| crate::core::subs::subs(x, old, new);
    if !old.is_symbol() || old_symbols.iter().any(|s| free.contains(s)) {
        let mut sub_into_func: bool = true;
        for lim in lims.iter_mut() {
            *lim = (lim.0.clone(), subs(&lim.1)?, subs(&lim.2)?);
            if lim.0.free_symbols().iter().any(|s| old_symbols.contains(s)) {
                sub_into_func = false;
                break;
            }
        }
        if sub_into_func {
            func = subs(&func)?;
        }
    } else {
        for lim in lims.iter_mut() {
            *lim = (lim.0.clone(), subs(&lim.1)?, subs(&lim.2)?);
            if *old == lim.0 {
                break;
            }
        }
    }
    lims.reverse();
    Ok(Some(node(e.head().clone(), &func, &lims)?))
}
