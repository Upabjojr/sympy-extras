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
