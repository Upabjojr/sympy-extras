//! Port of the classes of `sympy_extras/core/symbol.py`: `Symbol`, `Dummy`
//! and `Wild`, as atoms of the tree, and the typed views [`Symbol`],
//! [`Dummy`], [`Wild`].
//!
//! A symbol is identified by its name and the closure of its assumptions
//! (`Symbol('p', positive=True, real=True) == Symbol('p', positive=True)`);
//! a dummy only equals itself; a wild also carries the subtrees it may not
//! contain and predicates its values must satisfy (compared by identity, as
//! Python compares the functions).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::assumptions::{closure, Facts, Predicate};
use super::basic::{Basic, Head};
use super::sorting::CompareItem;

/// The data of a symbol: its name, the assumptions as given and their
/// closure (with `commutative = true` by default).
#[derive(Debug, Clone)]
pub struct SymbolData {
    pub name: String,
    pub given: Vec<(Predicate, bool)>,
    pub assumptions: Facts,
}

impl SymbolData {
    pub fn new(name: &str, assumptions: &[(Predicate, bool)]) -> SymbolData {
        let mut given: Vec<(Predicate, bool)> = assumptions.to_vec();
        given.sort();
        given.dedup_by(|a, b| a.0 == b.0);
        let mut closed: Facts = closure(&given);
        closed.entry(Predicate::commutative).or_insert(true);
        SymbolData {
            name: name.to_string(),
            given,
            assumptions: closed,
        }
    }

    /// SymPy's content of a symbol: its name and the closure of its
    /// assumptions, so that the symbols with assumptions sort after the
    /// plain ones.
    pub fn compare_content(&self) -> Vec<CompareItem> {
        let mut content: Vec<CompareItem> = vec![CompareItem::Str(self.name.clone())];
        for (p, v) in &self.assumptions {
            content.push(CompareItem::Fact(p.name().to_string(), *v));
        }
        content
    }
}

/// A predicate a value must satisfy to be bound to a wild.
pub type WildProperty = Arc<dyn Fn(&Basic) -> bool + Send + Sync>;

/// The data of a wild.
#[derive(Clone)]
pub struct WildData {
    pub symbol: SymbolData,
    pub exclude: Vec<Basic>,
    pub properties: Vec<WildProperty>,
}

impl std::fmt::Debug for WildData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WildData")
            .field("symbol", &self.symbol)
            .field("exclude", &self.exclude)
            .field("properties", &self.properties.len())
            .finish()
    }
}

impl WildData {
    pub fn same(&self, other: &WildData) -> bool {
        self.symbol.name == other.symbol.name
            && self.exclude == other.exclude
            && self.properties.len() == other.properties.len()
            && self
                .properties
                .iter()
                .zip(&other.properties)
                .all(|(a, b)| Arc::ptr_eq(a, b))
            && self.symbol.assumptions == other.symbol.assumptions
    }
}

/// A symbol (a node whose head is `Symbol`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol(pub Basic);

impl Symbol {
    /// `Symbol(name, **assumptions)`.
    pub fn new(name: &str, assumptions: &[(Predicate, bool)]) -> Symbol {
        Symbol(Basic::atom(Head::Symbol(Arc::new(SymbolData::new(name, assumptions)))))
    }

    pub fn name(&self) -> &str {
        match self.0.head() {
            Head::Symbol(s) => &s.name,
            _ => unreachable!("a Symbol node"),
        }
    }

    /// The assumptions as given to the constructor.
    pub fn given_assumptions(&self) -> &[(Predicate, bool)] {
        match self.0.head() {
            Head::Symbol(s) => &s.given,
            _ => unreachable!("a Symbol node"),
        }
    }

    pub fn as_basic(&self) -> &Basic {
        &self.0
    }
}

/// The index of the next dummy.
static DUMMY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A dummy symbol, equal only to itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dummy(pub Basic);

impl Dummy {
    pub fn new(name: &str, assumptions: &[(Predicate, bool)]) -> Dummy {
        let index: u64 = DUMMY_COUNTER.fetch_add(1, Ordering::Relaxed);
        Dummy(Basic::atom(Head::Dummy(
            Arc::new(SymbolData::new(name, assumptions)),
            index,
        )))
    }

    pub fn dummy_index(&self) -> u64 {
        match self.0.head() {
            Head::Dummy(_, index) => *index,
            _ => unreachable!("a Dummy node"),
        }
    }
}

/// A symbol which matches any subtree, except those containing one of
/// `exclude` or failing one of the `properties`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Wild(pub Basic);

impl Wild {
    pub fn new(
        name: &str,
        exclude: Vec<Basic>,
        properties: Vec<WildProperty>,
        assumptions: &[(Predicate, bool)],
    ) -> Wild {
        let symbol: SymbolData = SymbolData::new(name, assumptions);
        Wild(Basic::atom(Head::Wild(Arc::new(WildData {
            symbol,
            exclude,
            properties,
        }))))
    }

    /// Whether `value` may be bound to this wild.
    pub fn accepts(&self, value: &Basic) -> bool {
        match self.0.head() {
            Head::Wild(w) => !w.exclude.iter().any(|e| value.has(e)) && w.properties.iter().all(|p| p(value)),
            _ => unreachable!("a Wild node"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::assumptions::Predicate::*;
    use std::collections::HashMap;

    #[test]
    fn doctests() {
        let x = Symbol::new("x", &[]);
        assert_eq!(x.name(), "x");
        let p = Symbol::new("p", &[(positive, true)]);
        assert_eq!(p.0.symbol_facts().unwrap().get(&real), Some(&true));
        assert_eq!(Symbol::new("x", &[]), x);
        assert_ne!(Symbol::new("p", &[]), p);
        assert_eq!(Symbol::new("p", &[(positive, true), (real, true)]), p);
        let d = Dummy::new("x", &[]);
        assert_ne!(Dummy::new("x", &[]), d);
        assert_eq!(d, d.clone());
        assert_eq!(d.0.to_string(), "_x");
        let k = Wild::new("k", vec![], vec![Arc::new(|e: &Basic| e.is_integer_value())], &[]);
        assert_eq!(k.0.to_string(), "k_");
        assert!(k.accepts(&Basic::int(3)) && !k.accepts(&x.0));
        let a = Wild::new("a", vec![x.0.clone()], vec![], &[]);
        assert!(!a.accepts(&Basic::new(Head::Add, vec![x.0.clone(), Basic::one()])));
        // the compare order: numbers, symbols, symbols with assumptions, powers, products, sums
        let y = Symbol::new("y", &[]);
        let n = Symbol::new("n", &[(integer, true)]);
        let x2 = Basic::new(Head::Pow, vec![x.0.clone(), Basic::int(2)]);
        let xy = Basic::new(Head::Mul, vec![x.0.clone(), y.0.clone()]);
        let x1 = Basic::new(Head::Add, vec![Basic::one(), x.0.clone()]);
        let mut items: Vec<Basic> = vec![
            n.0.clone(),
            y.0.clone(),
            x2.clone(),
            Basic::int(2),
            xy.clone(),
            x1.clone(),
            x.0.clone(),
        ];
        items.sort_by(|a, b| a.compare_key().cmp(b.compare_key()));
        let shown: Vec<String> = items.iter().map(|e| e.to_string()).collect();
        assert_eq!(shown, vec!["2", "x", "y", "n", "x**2", "x*y", "x + 1"]);
        assert_eq!(
            (
                x.0.compare(&y.0),
                y.0.compare(&x.0),
                x.0.compare(&x.0),
                n.0.compare(&y.0)
            ),
            (-1, 1, 0, 1)
        );
        // xreplace is structural
        let mut rules: HashMap<Basic, Basic> = HashMap::new();
        rules.insert(x.0.clone(), y.0.clone());
        assert_eq!(x1.xreplace(&rules).to_string(), "y + 1");
        assert_eq!(xy.free_symbols().len(), 2);
        assert!(x2.has(&Basic::int(2)) && !x2.has(&y.0));
        assert_eq!(x1.count_nodes(), 3);
        assert_eq!(x1.atoms(None).len(), 2);
    }
}
