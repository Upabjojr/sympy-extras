//! Predicates and assumptions, ported from `sympy_extras.core.assume`:
//! [`Head::Predicate`] (`Q.positive`, or a predicate of the user, distinct
//! from the one of the same name in `Q`), [`Head::AppliedPredicate`]
//! (`Q.positive(x)`, a proposition which never evaluates by itself), the
//! handlers registered on a predicate (the Python registers them for
//! classes; here for a test of the node), and the global assumptions with
//! [`assuming`].
//!
//! The registry and the global assumptions are per thread (the Python's
//! are global to the interpreter).

use std::cell::RefCell;
use std::sync::Arc;

use crate::core::basic::{Basic, Head};
use crate::error::{PolyError, PolyResult};

/// The data of a predicate: its name, its number of arguments, and
/// whether it is one of `Q` (equal on the name and that flag, as the
/// Python's `__eq__`).
#[derive(Debug, Clone)]
pub struct PredicateData {
    pub name: String,
    pub arity: usize,
    pub defined: bool,
}

impl PartialEq for PredicateData {
    fn eq(&self, other: &PredicateData) -> bool {
        self.name == other.name && self.defined == other.defined
    }
}

impl Eq for PredicateData {}

impl std::hash::Hash for PredicateData {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.defined.hash(state);
    }
}

/// SymPy's class name of a predicate (`PositivePredicate`, `_class_name`).
pub fn class_name_of(name: &str) -> String {
    let mut out: String = String::new();
    for part in name.split('_') {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(&chars.as_str().to_lowercase());
        }
    }
    out.push_str("Predicate");
    out
}

/// A predicate of the user (`Predicate(name, arity)`).
pub fn predicate(name: &str, arity: usize) -> Basic {
    Basic::atom(Head::Predicate(Arc::new(PredicateData {
        name: name.to_string(),
        arity,
        defined: false,
    })))
}

/// A predicate of `Q` (`Predicate._known`): `q("positive")` is `Q.positive`.
pub fn q(name: &str, arity: usize) -> Basic {
    Basic::atom(Head::Predicate(Arc::new(PredicateData {
        name: name.to_string(),
        arity,
        defined: true,
    })))
}

/// The data of a predicate node.
pub fn predicate_data(p: &Basic) -> Option<&Arc<PredicateData>> {
    match p.head() {
        Head::Predicate(d) => Some(d),
        _ => None,
    }
}

/// The predicate applied to the arguments (`predicate(*args)`); `Type` for
/// a wrong number of arguments or something which is no predicate.
pub fn apply(p: &Basic, args: &[Basic]) -> PolyResult<Basic> {
    let data = predicate_data(p).ok_or_else(|| PolyError::Type(format!("a predicate is expected, got {}", p)))?;
    if args.len() != data.arity {
        return Err(PolyError::Type(format!(
            "{} takes {} argument(s), {} given",
            p,
            data.arity,
            args.len()
        )));
    }
    Ok(applied(p, args))
}

/// `AppliedPredicate(predicate, *args)` without the check of the arity
/// (the Python's constructor).
pub fn applied(p: &Basic, args: &[Basic]) -> Basic {
    let mut all: Vec<Basic> = vec![p.clone()];
    all.extend(args.iter().cloned());
    Basic::new(Head::AppliedPredicate, all)
}

/// The rebuilt node of new arguments (`AppliedPredicate._evaluated`).
pub fn rebuild_applied(args: &[Basic]) -> PolyResult<Basic> {
    if args.is_empty() || predicate_data(&args[0]).is_none() {
        return Err(PolyError::Type(format!(
            "a predicate is expected, got {}",
            args.first().map(|a| a.to_string()).unwrap_or_default()
        )));
    }
    Ok(applied(&args[0], &args[1..]))
}

/// The predicate of an applied predicate (`function`).
pub fn function_of(e: &Basic) -> Option<&Basic> {
    if matches!(e.head(), Head::AppliedPredicate) {
        e.args().first()
    } else {
        None
    }
}

/// The arguments of an applied predicate (`arguments`).
pub fn arguments_of(e: &Basic) -> &[Basic] {
    if matches!(e.head(), Head::AppliedPredicate) {
        &e.args()[1..]
    } else {
        &[]
    }
}

/// A handler: `handler(expr, assumptions)` answers `Some(true)`,
/// `Some(false)` or `None`.
pub type Handler = fn(&Basic, &Basic) -> Option<bool>;
/// Which nodes a handler is for (the classes of the Python's `register`).
pub type Matcher = fn(&Basic) -> bool;

thread_local! {
    static HANDLERS: RefCell<Vec<(PredicateData, Matcher, Handler)>> = const { RefCell::new(Vec::new()) };
    static GLOBAL: RefCell<Vec<Basic>> = const { RefCell::new(Vec::new()) };
}

/// Register a handler on a predicate for the nodes of `matcher`
/// (`Predicate.register`).
pub fn register(p: &Basic, matcher: Matcher, handler: Handler) -> PolyResult<()> {
    let data = predicate_data(p).ok_or_else(|| PolyError::Type(format!("a predicate is expected, got {}", p)))?;
    HANDLERS.with(|h| h.borrow_mut().push(((**data).clone(), matcher, handler)));
    Ok(())
}

/// The answer of the registered handlers for `expr`, the first which is
/// not `None` (`Predicate.eval`).
pub fn eval(p: &Basic, expr: &Basic, assumptions: &Basic) -> Option<bool> {
    let data = predicate_data(p)?;
    let found: Vec<(Matcher, Handler)> = HANDLERS.with(|h| {
        h.borrow()
            .iter()
            .filter(|(d, _, _)| d == &**data)
            .map(|(_, m, f)| (*m, *f))
            .collect()
    });
    for (matcher, handler) in found {
        if matcher(expr) {
            if let Some(answer) = handler(expr, assumptions) {
                return Some(answer);
            }
        }
    }
    None
}

/// The global assumptions (`global_assumptions`), sorted by their
/// strings as the Python iterates them.
pub fn global_assumptions() -> Vec<Basic> {
    let mut items: Vec<Basic> = GLOBAL.with(|g| g.borrow().clone());
    items.sort_by_cached_key(|a| a.to_string());
    items
}

/// Add assumptions to the global ones (`global_assumptions.add`).
pub fn add_global(assumptions: &[Basic]) {
    GLOBAL.with(|g| {
        let mut g = g.borrow_mut();
        for a in assumptions {
            if !g.contains(a) {
                g.push(a.clone());
            }
        }
    })
}

/// Remove assumptions (`global_assumptions.remove`; `Value` for one which
/// is absent, Python's `KeyError`).
pub fn remove_global(assumptions: &[Basic]) -> PolyResult<()> {
    GLOBAL.with(|g| {
        let mut g = g.borrow_mut();
        for a in assumptions {
            match g.iter().position(|b| b == a) {
                Some(i) => {
                    g.remove(i);
                }
                None => return Err(PolyError::Value(format!("{} is not assumed", a))),
            }
        }
        Ok(())
    })
}

/// Remove every global assumption.
pub fn clear_global() {
    GLOBAL.with(|g| g.borrow_mut().clear());
}

/// The scope of [`assuming`]: the added assumptions are removed when it is
/// dropped.
pub struct Assuming {
    added: Vec<Basic>,
}

impl Drop for Assuming {
    fn drop(&mut self) {
        let _ = remove_global(&self.added);
    }
}

/// Add the assumptions to the global ones until the guard is dropped
/// (`with assuming(...)`).
pub fn assuming(assumptions: &[Basic]) -> Assuming {
    let present: Vec<Basic> = GLOBAL.with(|g| g.borrow().clone());
    let mut added: Vec<Basic> = Vec::new();
    for a in assumptions {
        if !present.contains(a) && !added.contains(a) {
            added.push(a.clone());
        }
    }
    add_global(&added);
    Assuming { added }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicates_and_context() {
        let x: Basic = crate::core::symbol::Symbol::new("x", &[]).0;
        let positive: Basic = q("positive", 1);
        let mine: Basic = predicate("positive", 1);
        assert_ne!(positive, mine);
        assert_eq!(positive, q("positive", 1));
        let p: Basic = apply(&positive, std::slice::from_ref(&x)).unwrap();
        assert_eq!(p.to_string(), "Q.positive(x)");
        assert!(apply(&positive, &[]).is_err());
        assert_eq!(class_name_of("extended_positive"), "ExtendedPositivePredicate");
        {
            let _scope = assuming(std::slice::from_ref(&p));
            assert_eq!(global_assumptions(), vec![p.clone()]);
        }
        assert!(global_assumptions().is_empty());
        fn symbols(e: &Basic) -> bool {
            e.is_symbol()
        }
        fn yes(_: &Basic, _: &Basic) -> Option<bool> {
            Some(true)
        }
        register(&mine, symbols, yes).unwrap();
        assert_eq!(eval(&mine, &x, &crate::core::logic::boolean(true)), Some(true));
        assert_eq!(eval(&positive, &x, &crate::core::logic::boolean(true)), None);
    }
}
