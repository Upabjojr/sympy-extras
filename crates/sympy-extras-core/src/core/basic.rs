//! Port of the structure of `sympy_extras/core/basic.py`: the node of the
//! expression tree.
//!
//! A node is a [`Head`] (the class, with the data of an atom) and a vector
//! of arguments. Nodes are immutable and shared ([`Basic`] is an `Arc`); two
//! nodes are equal exactly when their heads and their arguments are equal,
//! and the hash is derived from the same data. Nothing is evaluated when a
//! node is built: `Add(x, x)` keeps its two arguments.
//!
//! The traversals and rebuilds of Python's `Basic` (`args`, `func`, `atoms`,
//! `free_symbols`, `has`, `xreplace`, `count`, the `compare` order) are
//! here; `subs`, `doit`, `match`, `replace` need the evaluating functions
//! and come with them.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use crate::numbers::*;

use super::assumptions::Facts;
use super::sorting::{compare_key, CompareItem, CompareKey};
use super::symbol::{SymbolData, WildData};

/// The class of a node, with the data of an atom.
#[derive(Debug, Clone)]
pub enum Head {
    /// `Integer` and `Rational` (Python's two classes; `integer_class` says
    /// which one built the node, the value decides everything else).
    Rational {
        value: Rat,
        integer_class: bool,
    },
    Infinity,
    NegativeInfinity,
    ComplexInfinity,
    NaN,
    Pi,
    Exp1,
    EulerGamma,
    Catalan,
    GoldenRatio,
    TribonacciConstant,
    ImaginaryUnit,
    Symbol(Arc<SymbolData>),
    /// A dummy with its index: equal only to itself.
    Dummy(Arc<SymbolData>, u64),
    Wild(Arc<WildData>),
    Add,
    Mul,
    Pow,
}

impl Head {
    /// The name of the Python class (`type(e).__name__`).
    pub fn class_name(&self) -> &'static str {
        match self {
            Head::Rational {
                integer_class: true, ..
            } => "Integer",
            Head::Rational {
                integer_class: false, ..
            } => "Rational",
            Head::Infinity => "Infinity",
            Head::NegativeInfinity => "NegativeInfinity",
            Head::ComplexInfinity => "ComplexInfinity",
            Head::NaN => "NaN",
            Head::Pi => "Pi",
            Head::Exp1 => "Exp1",
            Head::EulerGamma => "EulerGammaType",
            Head::Catalan => "CatalanType",
            Head::GoldenRatio => "GoldenRatioType",
            Head::TribonacciConstant => "TribonacciConstantType",
            Head::ImaginaryUnit => "ImaginaryUnit",
            Head::Symbol(_) => "Symbol",
            Head::Dummy(_, _) => "Dummy",
            Head::Wild(_) => "Wild",
            Head::Add => "Add",
            Head::Mul => "Mul",
            Head::Pow => "Pow",
        }
    }

    /// The name of the class in SymPy's `compare` order (`_compare_name`).
    fn compare_name(&self) -> &'static str {
        match self {
            Head::Rational { value, .. } => {
                if value.is_int() {
                    if value.is_zero() {
                        "Zero"
                    } else if value.is_one() {
                        "One"
                    } else if *value == -Rat::ONE {
                        "NegativeOne"
                    } else {
                        "Integer"
                    }
                } else if *value == rat(1, 2) {
                    "Half"
                } else {
                    "Rational"
                }
            }
            Head::EulerGamma => "EulerGamma",
            Head::Catalan => "Catalan",
            Head::GoldenRatio => "GoldenRatio",
            Head::TribonacciConstant => "TribonacciConstant",
            other => other.class_name(),
        }
    }

    /// Whether the class is an atom (no arguments).
    pub fn is_atom(&self) -> bool {
        !matches!(self, Head::Add | Head::Mul | Head::Pow)
    }

    fn same(&self, other: &Head) -> bool {
        match (self, other) {
            // a Rational equals the Integer of the same value
            (Head::Rational { value: a, .. }, Head::Rational { value: b, .. }) => a == b,
            (Head::Symbol(a), Head::Symbol(b)) => a.name == b.name && a.assumptions == b.assumptions,
            (Head::Dummy(a, i), Head::Dummy(b, j)) => i == j && a.name == b.name,
            (Head::Wild(a), Head::Wild(b)) => a.same(b),
            (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b),
        }
    }

    fn hash_into<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Head::Rational { value, .. } => {
                value.numerator().hash(state);
                value.denominator().hash(state);
            }
            Head::Symbol(s) => {
                s.name.hash(state);
                s.assumptions.hash(state);
            }
            Head::Dummy(s, index) => {
                s.name.hash(state);
                index.hash(state);
            }
            Head::Wild(w) => {
                w.symbol.name.hash(state);
                w.symbol.assumptions.hash(state);
            }
            _ => {}
        }
    }
}

/// A node: its head and its arguments, with the hash and the `compare` key
/// computed once.
pub struct Node {
    head: Head,
    args: Vec<Basic>,
    hash: u64,
    compare_key: OnceLock<CompareKey>,
}

/// A shared immutable node of the expression tree (Python's `Basic`).
#[derive(Clone)]
pub struct Basic(Arc<Node>);

impl Basic {
    /// The node of `head` with `args`, unevaluated.
    pub fn new(head: Head, args: Vec<Basic>) -> Basic {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        head.hash_into(&mut hasher);
        for a in &args {
            a.0.hash.hash(&mut hasher);
        }
        let hash: u64 = hasher.finish();
        Basic(Arc::new(Node {
            head,
            args,
            hash,
            compare_key: OnceLock::new(),
        }))
    }

    /// An atom.
    pub fn atom(head: Head) -> Basic {
        Basic::new(head, Vec::new())
    }

    pub fn head(&self) -> &Head {
        &self.0.head
    }

    /// The arguments.
    pub fn args(&self) -> &[Basic] {
        &self.0.args
    }

    /// The node with `args` in place of its arguments, unevaluated
    /// (`self.func(*args)`).
    pub fn rebuild(&self, args: Vec<Basic>) -> Basic {
        if args.is_empty() && self.0.args.is_empty() {
            return self.clone();
        }
        Basic::new(self.0.head.clone(), args)
    }

    /// The name of the Python class.
    pub fn class_name(&self) -> &'static str {
        self.0.head.class_name()
    }

    pub fn is_atom(&self) -> bool {
        self.0.args.is_empty() && self.0.head.is_atom()
    }

    pub fn ptr_eq(&self, other: &Basic) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    // -- the class flags of SymPy

    pub fn is_number_class(&self) -> bool {
        matches!(
            self.0.head,
            Head::Rational { .. } | Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity | Head::NaN
        )
    }

    pub fn is_rational_class(&self) -> bool {
        matches!(self.0.head, Head::Rational { .. })
    }

    /// Whether the node is an integer (`Rational(2, 1)` is one).
    pub fn is_integer_value(&self) -> bool {
        matches!(&self.0.head, Head::Rational { value, .. } if value.is_int())
    }

    pub fn is_symbol(&self) -> bool {
        matches!(self.0.head, Head::Symbol(_) | Head::Dummy(_, _) | Head::Wild(_))
    }

    pub fn is_add(&self) -> bool {
        matches!(self.0.head, Head::Add)
    }

    pub fn is_mul(&self) -> bool {
        matches!(self.0.head, Head::Mul)
    }

    pub fn is_pow(&self) -> bool {
        matches!(self.0.head, Head::Pow)
    }

    /// The value of a rational.
    pub fn as_rational(&self) -> Option<&Rat> {
        match &self.0.head {
            Head::Rational { value, .. } => Some(value),
            _ => None,
        }
    }

    /// The assumptions of a symbol (with their consequences).
    pub fn symbol_facts(&self) -> Option<&Facts> {
        match &self.0.head {
            Head::Symbol(s) | Head::Dummy(s, _) => Some(&s.assumptions),
            Head::Wild(w) => Some(&w.symbol.assumptions),
            _ => None,
        }
    }

    // -- ordering

    fn compare_content(&self) -> Vec<CompareItem> {
        match &self.0.head {
            Head::Rational { value, .. } => {
                vec![
                    CompareItem::Int(value.numerator().clone()),
                    CompareItem::Int(denom(value)),
                ]
            }
            Head::Symbol(s) => s.compare_content(),
            Head::Dummy(s, index) => {
                let mut content: Vec<CompareItem> = s.compare_content();
                content.push(CompareItem::Int(Int::from(*index)));
                content
            }
            Head::Wild(w) => {
                // SymPy adds the exclusions and the properties
                let mut content: Vec<CompareItem> = w.symbol.compare_content();
                content.push(CompareItem::Int(Int::ZERO));
                content.push(CompareItem::Int(Int::ZERO));
                content
            }
            _ => self
                .0
                .args
                .iter()
                .map(|a| CompareItem::Key(a.compare_key().clone()))
                .collect(),
        }
    }

    /// The key of SymPy's `compare` order, computed once per node.
    pub fn compare_key(&self) -> &CompareKey {
        self.0
            .compare_key
            .get_or_init(|| compare_key(self.0.head.compare_name(), self.compare_content()))
    }

    /// `-1`, `0` or `1` as `self` sorts before, with, or after `other` in the
    /// order of the arguments of sums and products.
    pub fn compare(&self, other: &Basic) -> i32 {
        if self.ptr_eq(other) {
            return 0;
        }
        match self.compare_key().cmp(other.compare_key()) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }

    // -- traversal

    /// The nodes in preorder (the node, then its arguments in order).
    pub fn preorder(&self) -> Vec<Basic> {
        let mut result: Vec<Basic> = Vec::new();
        let mut stack: Vec<Basic> = vec![self.clone()];
        while let Some(node) = stack.pop() {
            for a in node.args().iter().rev() {
                stack.push(a.clone());
            }
            result.push(node);
        }
        result
    }

    /// The number of nodes of the tree.
    pub fn count_nodes(&self) -> usize {
        1 + self.0.args.iter().map(|a| a.count_nodes()).sum::<usize>()
    }

    /// The atoms of the tree (`atoms()`), or the nodes satisfying `filter`.
    pub fn atoms(&self, filter: Option<&dyn Fn(&Basic) -> bool>) -> Vec<Basic> {
        let mut seen: std::collections::HashSet<Basic> = std::collections::HashSet::new();
        let mut result: Vec<Basic> = Vec::new();
        for node in self.preorder() {
            let keep: bool = match filter {
                Some(f) => f(&node),
                None => node.args().is_empty(),
            };
            if keep && seen.insert(node.clone()) {
                result.push(node);
            }
        }
        result
    }

    /// The symbols of the tree (`free_symbols`; nothing binds symbols yet).
    pub fn free_symbols(&self) -> BTreeSet<Basic> {
        self.preorder().into_iter().filter(|n| n.is_symbol()).collect()
    }

    /// Whether the tree contains `pattern` as a subtree.
    pub fn has(&self, pattern: &Basic) -> bool {
        self.preorder().iter().any(|n| n == pattern)
    }

    /// The tree with the subtrees in `rules` replaced, purely structurally
    /// (the rebuilt nodes are not evaluated); the largest subtrees first.
    pub fn xreplace(&self, rules: &HashMap<Basic, Basic>) -> Basic {
        if let Some(value) = rules.get(self) {
            return value.clone();
        }
        if self.0.args.is_empty() {
            return self.clone();
        }
        let args: Vec<Basic> = self.0.args.iter().map(|a| a.xreplace(rules)).collect();
        if args.iter().zip(&self.0.args).all(|(a, b)| a.ptr_eq(b)) {
            return self.clone();
        }
        self.rebuild(args)
    }
}

impl PartialEq for Basic {
    fn eq(&self, other: &Basic) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        self.0.hash == other.0.hash && self.0.head.same(&other.0.head) && self.0.args == other.0.args
    }
}

impl Eq for Basic {}

impl Hash for Basic {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash.hash(state);
    }
}

/// The `compare` order (for sets of nodes; equal nodes compare equal).
impl PartialOrd for Basic {
    fn partial_cmp(&self, other: &Basic) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Basic {
    fn cmp(&self, other: &Basic) -> std::cmp::Ordering {
        if self == other {
            return std::cmp::Ordering::Equal;
        }
        self.compare_key()
            .cmp(other.compare_key())
            .then_with(|| self.0.hash.cmp(&other.0.hash))
    }
}

impl fmt::Debug for Basic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The printed form of the atoms (Python's `str`); a compound node prints
/// as `Class(arg, ...)` until the printer is ported.
impl fmt::Display for Basic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0.head {
            Head::Rational { value, .. } => {
                if value.is_int() {
                    write!(f, "{}", value.numerator())
                } else {
                    write!(f, "{}/{}", value.numerator(), denom(value))
                }
            }
            Head::Infinity => write!(f, "oo"),
            Head::NegativeInfinity => write!(f, "-oo"),
            Head::ComplexInfinity => write!(f, "zoo"),
            Head::NaN => write!(f, "nan"),
            Head::Pi => write!(f, "pi"),
            Head::Exp1 => write!(f, "E"),
            Head::EulerGamma => write!(f, "EulerGamma"),
            Head::Catalan => write!(f, "Catalan"),
            Head::GoldenRatio => write!(f, "GoldenRatio"),
            Head::TribonacciConstant => write!(f, "TribonacciConstant"),
            Head::ImaginaryUnit => write!(f, "I"),
            Head::Symbol(s) => write!(f, "{}", s.name),
            Head::Dummy(s, _) => write!(f, "_{}", s.name),
            Head::Wild(w) => write!(f, "{}_", w.symbol.name),
            _ => {
                let args: Vec<String> = self.0.args.iter().map(|a| a.to_string()).collect();
                write!(f, "{}({})", self.class_name(), args.join(", "))
            }
        }
    }
}
