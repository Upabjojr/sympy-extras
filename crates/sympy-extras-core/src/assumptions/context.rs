//! The global assumptions of `sympy_extras.assumptions` (Mathematica's
//! `$Assumptions`) and [`assuming`], ported from
//! `sympy_extras.assumptions.context`: an ordered collection, per thread,
//! whose `assuming` also pushes the equivalent predicates of `Q` to the
//! core's global assumptions ([`crate::core::assume`]) for the duration of
//! the scope.

use std::cell::RefCell;

use crate::core::basic::Basic;
use crate::core::logic::and_;
use crate::error::PolyResult;

use super::facts::{conjuncts, Facts};

thread_local! {
    static GLOBAL: RefCell<Vec<Basic>> = const { RefCell::new(Vec::new()) };
}

/// The assumptions in force everywhere, in the order they were added.
pub fn global_assumptions() -> Vec<Basic> {
    GLOBAL.with(|g| g.borrow().clone())
}

/// Add assumptions (once each).
pub fn add(assumptions: &[Basic]) {
    GLOBAL.with(|g| {
        let mut g = g.borrow_mut();
        for a in assumptions {
            if !g.contains(a) {
                g.push(a.clone());
            }
        }
    })
}

/// Remove assumptions; missing ones are ignored.
pub fn remove(assumptions: &[Basic]) {
    GLOBAL.with(|g| g.borrow_mut().retain(|b| !assumptions.contains(b)))
}

/// Remove all the assumptions.
pub fn clear() {
    GLOBAL.with(|g| g.borrow_mut().clear())
}

/// The conjunction of the global assumptions (`as_boolean`).
pub fn as_boolean() -> Basic {
    and_(&global_assumptions())
}

/// The scope of [`assuming`].
pub struct Assuming {
    added: Vec<Basic>,
    pushed: Vec<Basic>,
}

impl Drop for Assuming {
    fn drop(&mut self) {
        remove(&self.added);
        let _ = crate::core::assume::remove_global(&self.pushed);
    }
}

/// Add the assumptions to the global ones, and their predicates of `Q` to
/// the core's, until the guard is dropped (`with assuming(...)`).
pub fn assuming(assumptions: &[Basic]) -> PolyResult<Assuming> {
    let predicates: Vec<Basic> = conjuncts(&Facts::new(assumptions, None, &[])?.predicates);
    let present: Vec<Basic> = global_assumptions();
    let mut added: Vec<Basic> = Vec::new();
    for a in assumptions {
        if !present.contains(a) && !added.contains(a) {
            added.push(a.clone());
        }
    }
    add(&added);
    let core_present: Vec<Basic> = crate::core::assume::global_assumptions();
    let mut pushed: Vec<Basic> = Vec::new();
    for p in predicates {
        if !core_present.contains(&p) && !pushed.contains(&p) {
            pushed.push(p);
        }
    }
    crate::core::assume::add_global(&pushed);
    Ok(Assuming { added, pushed })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped() {
        let x: Basic = crate::core::symbol::Symbol::new("x", &[]).0;
        let positive: Basic = crate::core::relational::gt(&x, &Basic::one()).unwrap();
        {
            let _scope = assuming(std::slice::from_ref(&positive)).unwrap();
            assert_eq!(global_assumptions(), vec![positive.clone()]);
            assert!(!crate::core::assume::global_assumptions().is_empty());
            let q: Basic = crate::core::ask::q_applied("positive", std::slice::from_ref(&x)).unwrap();
            assert_eq!(crate::core::ask::ask_simple(&q).unwrap(), Some(true));
        }
        assert!(global_assumptions().is_empty());
        assert!(crate::core::assume::global_assumptions().is_empty());
    }
}
