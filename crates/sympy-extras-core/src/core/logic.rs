//! Port of `sympy_extras/core/logic.py`: the boolean nodes of the core
//! ([`Head::BooleanTrue`], [`Head::BooleanFalse`], [`Head::And`],
//! [`Head::Or`], [`Head::Not`], [`Head::Xor`], [`Head::Implies`],
//! [`Head::Equivalent`], [`Head::Ite`]) and the evaluating functions
//! ([`and_`], [`or_`], [`not_`], [`xor`], [`implies`], [`equivalent`],
//! [`ite`], [`nand`], [`nor`], [`xnor`]). Any node may be an argument (a
//! symbol is a boolean variable).

use std::collections::HashSet;

use super::basic::{Basic, Head};
use super::sorting::{ordered, sort_key};

pub fn true_() -> Basic {
    Basic::atom(Head::BooleanTrue)
}

pub fn false_() -> Basic {
    Basic::atom(Head::BooleanFalse)
}

/// `true` or `false` of a bool.
pub fn boolean(value: bool) -> Basic {
    if value {
        true_()
    } else {
        false_()
    }
}

/// `Some(value)` for `true` and `false`.
pub fn truth(e: &Basic) -> Option<bool> {
    match e.head() {
        Head::BooleanTrue => Some(true),
        Head::BooleanFalse => Some(false),
        _ => None,
    }
}

/// Whether the node is a boolean (a connective, a relation, a truth value).
pub fn is_boolean(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::BooleanTrue
            | Head::BooleanFalse
            | Head::And
            | Head::Or
            | Head::Not
            | Head::Xor
            | Head::Implies
            | Head::Equivalent
            | Head::Ite
            | Head::Relational(_)
            | Head::Contains
            | Head::ForAll
            | Head::Exists
            | Head::Predicate(_)
            | Head::AppliedPredicate
    )
}

fn flatten(head: &Head, args: &[Basic]) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    for a in args {
        if std::mem::discriminant(a.head()) == std::mem::discriminant(head) {
            result.extend(flatten(head, a.args()));
        } else {
            result.push(a.clone());
        }
    }
    result
}

fn lattice(head: Head, identity: bool, args: &[Basic]) -> Basic {
    let mut kept: Vec<Basic> = Vec::new();
    let mut seen: HashSet<Basic> = HashSet::new();
    for a in flatten(&head, args) {
        match truth(&a) {
            Some(v) if v == identity => continue,
            Some(_) => return boolean(!identity),
            None => {}
        }
        if seen.contains(&a) {
            continue;
        }
        seen.insert(a.clone());
        kept.push(a);
    }
    for a in &kept {
        // a relation and its negation give the zero (x > 0 and x <= 0)
        if let Head::Relational(op) = a.head() {
            let negated: Basic = Basic::new(Head::Relational(op.negated()), a.args().to_vec());
            if seen.contains(&negated) {
                return boolean(!identity);
            }
        }
    }
    match kept.len() {
        0 => boolean(identity),
        1 => kept.remove(0),
        _ => Basic::new(head, ordered(kept)),
    }
}

/// The evaluated conjunction (`and_`).
pub fn and_(args: &[Basic]) -> Basic {
    lattice(Head::And, true, args)
}

/// The evaluated disjunction (`or_`).
pub fn or_(args: &[Basic]) -> Basic {
    lattice(Head::Or, false, args)
}

/// The evaluated negation (`not_`): a truth value flipped, a relation
/// negated, a double negation cancelled.
pub fn not_(a: &Basic) -> Basic {
    if let Some(v) = truth(a) {
        return boolean(!v);
    }
    match a.head() {
        Head::Not => a.args()[0].clone(),
        Head::Relational(op) => Basic::new(Head::Relational(op.negated()), a.args().to_vec()),
        _ => Basic::new(Head::Not, vec![a.clone()]),
    }
}

/// The evaluated exclusive disjunction (`xor`).
pub fn xor(args: &[Basic]) -> Basic {
    let mut negate: bool = false;
    let mut kept: Vec<Basic> = Vec::new();
    for a in flatten(&Head::Xor, args) {
        if let Some(v) = truth(&a) {
            negate ^= v;
            continue;
        }
        if let Some(position) = kept.iter().position(|k| *k == a) {
            kept.remove(position);
            continue;
        }
        kept.push(a);
    }
    if kept.is_empty() {
        return boolean(negate);
    }
    let result: Basic = if kept.len() == 1 {
        kept.remove(0)
    } else {
        kept.sort_by_cached_key(sort_key);
        Basic::new(Head::Xor, kept)
    };
    if negate {
        not_(&result)
    } else {
        result
    }
}

/// The evaluated implication (`implies`).
pub fn implies(a: &Basic, b: &Basic) -> Basic {
    if let Some(v) = truth(a) {
        return if v { b.clone() } else { true_() };
    }
    if let Some(v) = truth(b) {
        return if v { true_() } else { not_(a) };
    }
    if a == b {
        return true_();
    }
    Basic::new(Head::Implies, vec![a.clone(), b.clone()])
}

/// The evaluated equivalence (`equivalent`).
pub fn equivalent(args: &[Basic]) -> Basic {
    let mut items: Vec<Basic> = Vec::new();
    for a in args {
        if !items.contains(a) {
            items.push(a.clone());
        }
    }
    if items.iter().any(|a| truth(a) == Some(true)) {
        let rest: Vec<Basic> = items.into_iter().filter(|a| truth(a) != Some(true)).collect();
        return and_(&rest);
    }
    if items.iter().any(|a| truth(a) == Some(false)) {
        let rest: Vec<Basic> = items.iter().filter(|a| truth(a) != Some(false)).map(not_).collect();
        return and_(&rest);
    }
    if items.len() <= 1 {
        return true_();
    }
    items.sort_by_cached_key(sort_key);
    Basic::new(Head::Equivalent, items)
}

/// The symbols of a boolean tree used as truth values (`binary_symbols`).
pub fn binary_symbols(e: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    fn walk(e: &Basic, found: &mut Vec<Basic>) {
        match e.head() {
            Head::Symbol(_) | Head::Dummy(..) => {
                if !found.contains(e) {
                    found.push(e.clone());
                }
            }
            Head::Relational(_) => {}
            Head::ForAll | Head::Exists => {
                let bound: &[Basic] = e.args()[0].args();
                for s in binary_symbols(&e.args()[1]) {
                    if !bound.contains(&s) && !found.contains(&s) {
                        found.push(s);
                    }
                }
            }
            _ if is_boolean(e) => {
                for a in e.args() {
                    if is_boolean(a) || a.is_symbol() || matches!(a.head(), Head::Dummy(..)) {
                        walk(a, found);
                    }
                }
            }
            _ => {}
        }
    }
    walk(e, &mut found);
    found
}

// PORTED FROM SYMPY: sympy/logic/boolalg.py:ITE.__new__, ITE.eval (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
/// The evaluated `ITE(c, a, b)` (`ite`), by SymPy's rules: the branch
/// chosen by a known condition, the branch when both are equal, the
/// condition (or its negation) when the branches are `true` and `false`;
/// an equation `Eq(p, e)` between a symbol `p` used as a truth value in a
/// branch and something else is false.
pub fn ite(c: &Basic, a: &Basic, b: &Basic) -> Basic {
    let mut c: Basic = c.clone();
    if let Head::Relational(op @ (super::basic::RelOp::Eq | super::basic::RelOp::Ne)) = c.head() {
        let mut binary: Vec<Basic> = binary_symbols(a);
        for s in binary_symbols(b) {
            if !binary.contains(&s) {
                binary.push(s);
            }
        }
        let mut sides: Vec<Basic> = Vec::new();
        for side in c.args() {
            if !sides.contains(side) {
                sides.push(side.clone());
            }
        }
        if sides.iter().filter(|s| !binary.contains(s)).count() == 1 {
            c = if matches!(op, super::basic::RelOp::Ne) {
                true_()
            } else {
                false_()
            };
        }
    }
    if let Some(v) = truth(&c) {
        return if v { a.clone() } else { b.clone() };
    }
    if a == b {
        return a.clone();
    }
    if truth(a) == Some(true) && truth(b) == Some(false) {
        return c;
    }
    if truth(a) == Some(false) && truth(b) == Some(true) {
        return not_(&c);
    }
    Basic::new(Head::Ite, vec![c, a.clone(), b.clone()])
}

pub fn nand(args: &[Basic]) -> Basic {
    not_(&and_(args))
}

pub fn nor(args: &[Basic]) -> Basic {
    not_(&or_(args))
}

pub fn xnor(args: &[Basic]) -> Basic {
    not_(&xor(args))
}
