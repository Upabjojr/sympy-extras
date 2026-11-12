//! Port of `sympy_extras/core/sorting.py`: the `compare` order of SymPy's
//! `Basic.compare`, the order of the arguments of sums and products.
//!
//! The key of a node is `((rank of the class, class name), length of the
//! content, the content)`, compared lexicographically; the content holds
//! the keys of the arguments, the name and the assumptions of a symbol, the
//! numerator and the denominator of a rational, ...

use crate::numbers::Int;

/// An element of the content of a node in the `compare` order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompareItem {
    /// The key of an argument.
    Key(CompareKey),
    /// A name.
    Str(String),
    /// An integer (a numerator, a denominator, the index of a dummy).
    Int(Int),
    /// An assumption of a symbol `(name, value)`.
    Fact(String, bool),
}

/// The key of the `compare` order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompareKey {
    pub rank: usize,
    pub name: String,
    pub length: usize,
    pub content: Vec<CompareItem>,
}

/// The classes which SymPy's `compare` orders first, in this order; every
/// other class follows, by name.
const CLASS_ORDER: [&str; 30] = [
    "Zero",
    "One",
    "Half",
    "Infinity",
    "NaN",
    "NegativeOne",
    "NegativeInfinity",
    "Integer",
    "Rational",
    "Float",
    "Exp1",
    "Pi",
    "ImaginaryUnit",
    "Symbol",
    "Wild",
    "Pow",
    "Mul",
    "Add",
    "Derivative",
    "Integral",
    "Abs",
    "factorial",
    "Lambda",
    "Order",
    "Equality",
    "Unequality",
    "StrictGreaterThan",
    "StrictLessThan",
    "GreaterThan",
    "LessThan",
];

/// The place of a class (by its SymPy name) in the `compare` order.
pub fn class_rank(name: &str) -> (usize, String) {
    let rank: usize = CLASS_ORDER.iter().position(|n| *n == name).unwrap_or(CLASS_ORDER.len());
    (rank, name.to_string())
}

/// The key of a class name and a content.
pub fn compare_key(name: &str, content: Vec<CompareItem>) -> CompareKey {
    let (rank, name) = class_rank(name);
    CompareKey {
        rank,
        name,
        length: content.len(),
        content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctest() {
        assert!(class_rank("Symbol") < class_rank("Pow"));
        assert!(class_rank("Pow") < class_rank("Add"));
        assert!(class_rank("Add") < class_rank("Dummy"));
        assert!(class_rank("Dummy") < class_rank("sin"));
    }
}

// ---------------------------------------------------------------------
// the canonical key of `default_sort_key` (`Basic.sort_key`)

/// `(group, subgroup, class name)`: the class part of a sort key (SymPy's
/// `class_key`).
pub type ClassKey = (i64, i64, String);

/// The arguments part of a sort key: their keys, the name of an atom, or
/// the name and index of a dummy (`ArgsKey`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArgsItems {
    Keys(Vec<SortKey>),
    Name(String),
    Dummy(String, u64),
}

/// `(class key, (number of arguments, arguments), exponent key,
/// coefficient)`; the exponent key is empty (`None`) for a number, which
/// orders it first as Python's `()`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SortKey {
    pub class: ClassKey,
    pub args: (usize, ArgsItems),
    pub exp: Option<Box<SortKey>>,
    pub coefficient: crate::numbers::Rat,
}

use super::basic::{Basic, Head};
use super::expr::{as_base_exp, as_coeff_mul, factors_of, is_true};
use crate::numbers::{rat, Rat};

/// The class key of a node (`class_key`).
pub fn class_key(e: &Basic) -> ClassKey {
    let name: String = e.class_name().to_string();
    match e.head() {
        _ if e.is_number_node() => (1, 0, "Number".to_string()),
        Head::Symbol(_) => (2, 0, "Symbol".to_string()),
        Head::Dummy(..) => (2, 0, "Dummy".to_string()),
        Head::Wild(_) => (2, 0, "Wild".to_string()),
        Head::Add => (3, 1, name),
        Head::Mul => (3, 0, name),
        Head::Pow => (3, 2, name),
        Head::Function(f) => (4, f.rank(), f.name().to_string()),
        Head::Undefined(u) => (4, 0, u.name.clone()),
        Head::Relational(op) => (5, 0, op.sympy_name().to_string()),
        Head::AccumBounds => (5, 0, "AccumulationBounds".to_string()),
        Head::Piecewise => (4, 10000, "Piecewise".to_string()),
        Head::ExprCondPair => (3, 1, "ExprCondPair".to_string()),
        _ if super::logic::is_boolean(e) || crate::sets::sets::is_set(e) => (5, 0, name),
        _ if e.is_atom() => (2, 0, name),
        _ => (5, 0, name),
    }
}

/// The value which orders the numbers (`sort_value`: `0` for the
/// infinities and `nan`).
fn sort_value(e: &Basic) -> Rat {
    e.as_rational().cloned().unwrap_or(Rat::ZERO)
}

/// The arguments part of the key of an atom (`_sort_atom_args`).
fn sort_atom_args(e: &Basic) -> (usize, ArgsItems) {
    match e.head() {
        Head::Dummy(data, index) => (2, ArgsItems::Dummy(data.name.clone(), *index)),
        _ => (1, ArgsItems::Name(e.to_string())),
    }
}

fn one_key() -> SortKey {
    SortKey {
        class: (1, 0, "Number".to_string()),
        args: (0, ArgsItems::Keys(Vec::new())),
        exp: None,
        coefficient: rat(1, 1),
    }
}

/// The canonical key of an expression (`default_sort_key`).
pub fn sort_key(e: &Basic) -> SortKey {
    if crate::sets::sets::is_set(e) {
        // Basic._compute_sort_key (a set is not an expression)
        return SortKey {
            class: class_key(e),
            args: (e.args().len(), ArgsItems::Keys(e.args().iter().map(sort_key).collect())),
            exp: None,
            coefficient: rat(1, 1),
        };
    }
    if super::logic::is_boolean(e) {
        // Basic._compute_sort_key (a boolean is not an expression); true
        // and false: their names (BooleanAtom)
        let args: (usize, ArgsItems) = if e.is_atom() {
            (1, ArgsItems::Name(e.to_string()))
        } else {
            (e.args().len(), ArgsItems::Keys(e.args().iter().map(sort_key).collect()))
        };
        return SortKey {
            class: class_key(e),
            args,
            exp: None,
            coefficient: rat(1, 1),
        };
    }
    if e.is_number_node() {
        return SortKey {
            class: class_key(e),
            args: (0, ArgsItems::Keys(Vec::new())),
            exp: None,
            coefficient: sort_value(e),
        };
    }
    if matches!(e.head(), Head::Tuple) {
        // Basic._compute_sort_key (a Tuple is not an expression)
        return SortKey {
            class: class_key(e),
            args: (e.args().len(), ArgsItems::Keys(e.args().iter().map(sort_key).collect())),
            exp: None,
            coefficient: rat(1, 1),
        };
    }
    if e.is_atom() {
        return SortKey {
            class: class_key(e),
            args: sort_atom_args(e),
            exp: Some(Box::new(one_key())),
            coefficient: rat(1, 1),
        };
    }
    let (coeff, rest) = as_coeff_mul(e);
    let (base, exp): (Basic, Option<Basic>) = if rest.is_pow() {
        let (b, x) = as_base_exp(&rest).expect("the decomposition of a power");
        (b, Some(x))
    } else {
        (rest, None)
    };
    let args: (usize, ArgsItems) = if base.is_atom() {
        sort_atom_args(&base)
    } else {
        let ordered: Vec<Basic> = if base.is_add() {
            super::printing::ordered_terms(base.args())
        } else if base.is_mul() {
            as_ordered_factors(&base)
        } else {
            base.args().to_vec()
        };
        (ordered.len(), ArgsItems::Keys(ordered.iter().map(sort_key).collect()))
    };
    SortKey {
        class: class_key(&base),
        args,
        exp: Some(Box::new(match exp {
            None => one_key(),
            Some(x) => sort_key(&x),
        })),
        coefficient: sort_value(&coeff),
    }
}

/// The factors of a product in canonical order, a negative coefficient
/// other than `-1` split into `-1` and its absolute value
/// (`as_ordered_factors`).
pub fn as_ordered_factors(e: &Basic) -> Vec<Basic> {
    let mut factors: Vec<Basic> = factors_of(e);
    if factors.len() > 1
        && factors[0].is_number_node()
        && is_true(&factors[0], super::assumptions::Predicate::negative)
        && factors[0] != Basic::negative_one()
    {
        let positive: Basic = super::numbers::number_mul(&factors[0], &Basic::negative_one());
        factors.splice(0..1, [Basic::negative_one(), positive]);
    }
    factors.sort_by_cached_key(sort_key);
    factors
}

/// SymPy's `ordered` (`core.sorting.ordered`): by the number of nodes, then
/// the canonical key.
pub fn ordered(items: Vec<Basic>) -> Vec<Basic> {
    let mut items: Vec<Basic> = items;
    items.sort_by_cached_key(|e| (e.count_nodes(), sort_key(e)));
    items
}
