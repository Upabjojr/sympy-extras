//! Traversals of the tree and rebuilding helpers, ported from
//! `sympy_extras.core.traversal`: [`postorder_traversal`], [`iterargs`],
//! [`iterfreeargs`], [`bottom_up`], [`use_at`] and the bound symbols of a
//! node ([`bound_symbols`]).

use std::collections::{BTreeSet, VecDeque};

use super::basic::{Basic, Head};
use crate::error::PolyResult;

/// The symbols bound by a node (`bound_symbols`): the variables of a
/// `Subs`, of a quantifier, of an `IndexedRoot`; empty otherwise.
pub fn bound_symbols(e: &Basic) -> Vec<Basic> {
    match e.head() {
        Head::Subs | Head::ForAll | Head::Exists => {
            let i: usize = if matches!(e.head(), Head::Subs) { 1 } else { 0 };
            e.args()[i].args().to_vec()
        }
        Head::IndexedRoot | Head::TranscendentalRoot => vec![e.args()[1].clone()],
        Head::Sum | Head::Product => crate::concrete::expr_with_limits::bound_symbols(e),
        Head::Integral => crate::integrals::integral::bound_symbols(e),
        Head::RootSum => {
            let (_, var, _, t) = crate::polys::rootoftools::parts(e);
            if var == t {
                vec![t.clone()]
            } else {
                vec![t.clone(), var.clone()]
            }
        }
        _ => Vec::new(),
    }
}

/// The nodes of the tree, children before parents.
pub fn postorder_traversal(node: &Basic) -> Vec<Basic> {
    fn go(node: &Basic, out: &mut Vec<Basic>) {
        for a in node.args() {
            go(a, out);
        }
        out.push(node.clone());
    }
    let mut out: Vec<Basic> = Vec::new();
    go(node, &mut out);
    out
}

/// The nodes of the tree in breadth first order.
pub fn iterargs(node: &Basic) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    let mut queue: VecDeque<Basic> = VecDeque::from([node.clone()]);
    while let Some(current) = queue.pop_front() {
        queue.extend(current.args().iter().cloned());
        out.push(current);
    }
    out
}

/// The nodes in breadth first order, skipping the arguments which contain
/// a symbol bound by their parent.
pub fn iterfreeargs(node: &Basic) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    let mut queue: VecDeque<Basic> = VecDeque::from([node.clone()]);
    while let Some(current) = queue.pop_front() {
        let bound: Vec<Basic> = bound_symbols(&current);
        for a in current.args() {
            if !bound.is_empty() {
                let free: BTreeSet<Basic> = a.free_symbols();
                let atoms: Vec<Basic> = a.atoms(None);
                if bound.iter().any(|s| free.contains(s) || atoms.contains(s)) {
                    continue;
                }
            }
            queue.push_back(a.clone());
        }
        out.push(current);
    }
    out
}

/// `f` applied to every node of the tree, children first (the atoms only
/// when `atoms`); the rebuilt nodes are evaluated (`bottom_up`).
pub fn bottom_up(rv: &Basic, f: &dyn Fn(&Basic) -> PolyResult<Basic>, atoms: bool) -> PolyResult<Basic> {
    crate::timeout::check()?;
    if rv.args().is_empty() {
        return if atoms { f(rv) } else { Ok(rv.clone()) };
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in rv.args() {
        args.push(bottom_up(a, f, atoms)?);
    }
    // rebuilt when the arguments are not equal (not merely not identical)
    let rebuilt: Basic = if args.iter().zip(rv.args()).any(|(a, b)| !a.ptr_eq(b) && a != b) {
        super::function::rebuild_evaluated(rv, args)?
    } else {
        rv.clone()
    };
    f(&rebuilt)
}

/// `func` applied to the nodes at depth `level` (the root at level 0); the
/// rebuilt nodes are evaluated (`use`).
pub fn use_at(expr: &Basic, func: &dyn Fn(&Basic) -> PolyResult<Basic>, level: usize) -> PolyResult<Basic> {
    if level == 0 {
        return func(expr);
    }
    if expr.args().is_empty() {
        return Ok(expr.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in expr.args() {
        args.push(use_at(a, func, level - 1)?);
    }
    // Python compares by identity, and these classes build new argument
    // objects (Tuples, Integers) at every access of `args`: always rebuilt
    let recomputed: bool = matches!(
        expr.head(),
        Head::Subs
            | Head::Derivative
            | Head::ForAll
            | Head::Exists
            | Head::Hyper
            | Head::MeijerG
            | Head::ComplexRootOf(_)
    );
    if !recomputed && args.iter().zip(expr.args()).all(|(a, b)| a.ptr_eq(b)) {
        return Ok(expr.clone());
    }
    super::function::rebuild_evaluated(expr, args)
}
