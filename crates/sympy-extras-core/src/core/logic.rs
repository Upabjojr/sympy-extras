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

/// The evaluated `ITE(c, a, b)` (`ite`).
pub fn ite(c: &Basic, a: &Basic, b: &Basic) -> Basic {
    if let Some(v) = truth(c) {
        return if v { a.clone() } else { b.clone() };
    }
    Basic::new(Head::Ite, vec![c.clone(), a.clone(), b.clone()])
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
