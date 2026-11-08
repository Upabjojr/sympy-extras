//! Port of the rules of `sympy_extras/core/assumptions.py`: the predicates
//! of the assumptions system, their implications and the closure of a set
//! of facts (the structural rules for compound expressions come with the
//! evaluating functions).
//!
//! The predicates are the enum [`Predicate`] (Python's strings), declared in
//! alphabetical order of their names so that sorting them sorts the names,
//! as the Python code sorts the facts.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// A predicate of the assumptions system.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Predicate {
    algebraic,
    antihermitian,
    commutative,
    complex,
    composite,
    even,
    extended_negative,
    extended_nonnegative,
    extended_nonpositive,
    extended_nonzero,
    extended_positive,
    extended_real,
    finite,
    hermitian,
    imaginary,
    infinite,
    integer,
    irrational,
    negative,
    noninteger,
    nonnegative,
    nonpositive,
    nonzero,
    odd,
    polar,
    positive,
    prime,
    rational,
    real,
    transcendental,
    zero,
}

use Predicate::*;

/// The facts of a node: predicate -> truth value, sorted by name.
pub type Facts = BTreeMap<Predicate, bool>;

/// The predicates in Python's order (`PREDICATES`).
pub const PREDICATES: [Predicate; 31] = [
    commutative,
    complex,
    real,
    extended_real,
    imaginary,
    integer,
    noninteger,
    rational,
    irrational,
    algebraic,
    transcendental,
    positive,
    negative,
    nonnegative,
    nonpositive,
    zero,
    nonzero,
    extended_positive,
    extended_negative,
    extended_nonnegative,
    extended_nonpositive,
    extended_nonzero,
    even,
    odd,
    prime,
    composite,
    finite,
    infinite,
    hermitian,
    antihermitian,
    polar,
];

impl Predicate {
    /// The name (Python's string).
    pub fn name(self) -> &'static str {
        match self {
            algebraic => "algebraic",
            antihermitian => "antihermitian",
            commutative => "commutative",
            complex => "complex",
            composite => "composite",
            even => "even",
            extended_negative => "extended_negative",
            extended_nonnegative => "extended_nonnegative",
            extended_nonpositive => "extended_nonpositive",
            extended_nonzero => "extended_nonzero",
            extended_positive => "extended_positive",
            extended_real => "extended_real",
            finite => "finite",
            hermitian => "hermitian",
            imaginary => "imaginary",
            infinite => "infinite",
            integer => "integer",
            irrational => "irrational",
            negative => "negative",
            noninteger => "noninteger",
            nonnegative => "nonnegative",
            nonpositive => "nonpositive",
            nonzero => "nonzero",
            odd => "odd",
            polar => "polar",
            positive => "positive",
            prime => "prime",
            rational => "rational",
            real => "real",
            transcendental => "transcendental",
            zero => "zero",
        }
    }

    /// The predicate of a name (`None` for an unknown assumption).
    pub fn from_name(name: &str) -> Option<Predicate> {
        PREDICATES.iter().copied().find(|p| p.name() == name)
    }
}

/// Implications `(name, value) -> (name, value)`.
const RULES: &[(Predicate, bool, Predicate, bool)] = &[
    (integer, true, rational, true),
    (integer, true, noninteger, false),
    (rational, true, real, true),
    (rational, true, algebraic, true),
    (irrational, true, real, true),
    (irrational, true, rational, false),
    (real, true, complex, true),
    (real, true, extended_real, true),
    (real, true, imaginary, false),
    (real, true, finite, true),
    (real, true, hermitian, true),
    (imaginary, true, complex, true),
    (imaginary, true, real, false),
    (imaginary, true, extended_real, false),
    (imaginary, true, antihermitian, true),
    (complex, true, finite, true),
    (complex, true, commutative, true),
    (extended_real, true, commutative, true),
    (algebraic, true, complex, true),
    (algebraic, true, transcendental, false),
    (transcendental, true, complex, true),
    (infinite, true, finite, false),
    (finite, false, infinite, true),
    (positive, true, real, true),
    (positive, true, nonnegative, true),
    (positive, true, nonzero, true),
    (positive, true, negative, false),
    (positive, true, nonpositive, false),
    (positive, true, extended_positive, true),
    (negative, true, real, true),
    (negative, true, nonpositive, true),
    (negative, true, nonzero, true),
    (negative, true, positive, false),
    (negative, true, nonnegative, false),
    (negative, true, extended_negative, true),
    (nonnegative, true, real, true),
    (nonnegative, true, negative, false),
    (nonnegative, true, extended_nonnegative, true),
    (nonpositive, true, real, true),
    (nonpositive, true, positive, false),
    (nonpositive, true, extended_nonpositive, true),
    (zero, true, integer, true),
    (zero, true, even, true),
    (zero, true, nonnegative, true),
    (zero, true, nonpositive, true),
    (zero, true, nonzero, false),
    (zero, true, extended_nonzero, false),
    (zero, true, positive, false),
    (zero, true, negative, false),
    (nonzero, true, real, true),
    (nonzero, true, zero, false),
    (nonzero, true, extended_nonzero, true),
    (extended_positive, true, extended_nonnegative, true),
    (extended_positive, true, extended_nonzero, true),
    (extended_positive, true, extended_negative, false),
    (extended_positive, true, extended_nonpositive, false),
    (extended_negative, true, extended_nonpositive, true),
    (extended_negative, true, extended_nonzero, true),
    (extended_negative, true, extended_positive, false),
    (extended_negative, true, extended_nonnegative, false),
    (extended_nonnegative, true, extended_real, true),
    (extended_nonnegative, true, extended_negative, false),
    (extended_nonpositive, true, extended_real, true),
    (extended_nonpositive, true, extended_positive, false),
    (extended_nonzero, true, extended_real, true),
    (extended_nonzero, true, zero, false),
    (noninteger, true, extended_real, true),
    (noninteger, true, integer, false),
    (even, true, integer, true),
    (even, true, odd, false),
    (odd, true, integer, true),
    (odd, true, even, false),
    (prime, true, integer, true),
    (prime, true, positive, true),
    (prime, true, composite, false),
    (composite, true, integer, true),
    (composite, true, positive, true),
    (composite, true, prime, false),
];

/// Implications with two premises: `(a, va, b, vb) -> (c, vc)`.
const PAIR_RULES: &[(Predicate, bool, Predicate, bool, Predicate, bool)] = &[
    (nonnegative, true, nonzero, true, positive, true),
    (nonpositive, true, nonzero, true, negative, true),
    (nonnegative, true, nonpositive, true, zero, true),
    (real, true, negative, false, nonnegative, true),
    (real, true, positive, false, nonpositive, true),
    (real, true, zero, false, nonzero, true),
    (real, true, nonzero, false, zero, true),
    (real, true, rational, false, irrational, true),
    (integer, true, even, false, odd, true),
    (integer, true, odd, false, even, true),
    (
        extended_nonnegative,
        true,
        extended_nonzero,
        true,
        extended_positive,
        true,
    ),
    (
        extended_nonpositive,
        true,
        extended_nonzero,
        true,
        extended_negative,
        true,
    ),
    (extended_nonnegative, true, extended_nonpositive, true, zero, true),
    (
        extended_real,
        true,
        extended_negative,
        false,
        extended_nonnegative,
        true,
    ),
    (
        extended_real,
        true,
        extended_positive,
        false,
        extended_nonpositive,
        true,
    ),
    (extended_real, true, zero, false, extended_nonzero, true),
    (extended_real, true, extended_nonzero, false, zero, true),
    (extended_real, true, integer, false, noninteger, true),
    (extended_real, true, finite, true, real, true),
    (extended_positive, true, finite, true, positive, true),
    (extended_negative, true, finite, true, negative, true),
    (extended_nonnegative, true, finite, true, nonnegative, true),
    (extended_nonpositive, true, finite, true, nonpositive, true),
    (extended_nonzero, true, finite, true, nonzero, true),
    (complex, true, algebraic, false, transcendental, true),
];

/// Every fact which follows from the single fact by the implications and
/// their contrapositives (the fact itself first), in Python's order.
fn implied_from(name: Predicate, value: bool) -> Vec<(Predicate, bool)> {
    let mut found: Vec<(Predicate, bool)> = vec![(name, value)];
    let mut pending: Vec<(Predicate, bool)> = vec![(name, value)];
    let has = |found: &Vec<(Predicate, bool)>, p: Predicate| found.iter().any(|(q, _)| *q == p);
    while let Some((a, va)) = pending.pop() {
        for &(p, vp, q, vq) in RULES {
            if (p, vp) == (a, va) && !has(&found, q) {
                found.push((q, vq));
                pending.push((q, vq));
            }
            if (q, !vq) == (a, va) && !has(&found, p) {
                found.push((p, !vp));
                pending.push((p, !vp));
            }
        }
    }
    found
}

/// The consequences of each single fact, computed once.
fn implied(name: Predicate, value: bool) -> &'static [(Predicate, bool)] {
    static TABLE: OnceLock<BTreeMap<(Predicate, bool), Vec<(Predicate, bool)>>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = BTreeMap::new();
        for p in PREDICATES {
            for v in [true, false] {
                table.insert((p, v), implied_from(p, v));
            }
        }
        table
    });
    &table[&(name, value)]
}

/// The facts completed by the implications and their contrapositives; the
/// given facts are processed in their order, the first value of a fact
/// winning (so that a contradiction among them stays visible).
pub fn closure(facts: &[(Predicate, bool)]) -> Facts {
    let mut known: Facts = Facts::new();
    for &(p, v) in facts {
        known.insert(p, v);
    }
    let mut pending: Vec<(Predicate, bool)> = facts.to_vec();
    while !pending.is_empty() {
        for &(name, value) in &pending {
            if known.get(&name) == Some(&value) {
                for &(b, vb) in implied(name, value) {
                    known.entry(b).or_insert(vb);
                }
            }
        }
        pending.clear();
        for &(a, va, b, vb, c, vc) in PAIR_RULES {
            if !known.contains_key(&c) && known.get(&a) == Some(&va) && known.get(&b) == Some(&vb) {
                known.insert(c, vc);
                pending.push((c, vc));
            }
        }
    }
    known
}

/// Whether no implication is contradicted by the facts.
pub fn is_consistent(facts: &[(Predicate, bool)]) -> bool {
    let known: Facts = closure(facts);
    for &(a, va, b, vb) in RULES {
        if known.get(&a) == Some(&va) && known.get(&b) == Some(&!vb) {
            return false;
        }
    }
    for &(a, va, b, vb, c, vc) in PAIR_RULES {
        if known.get(&a) == Some(&va) && known.get(&b) == Some(&vb) && known.get(&c) == Some(&!vc) {
            return false;
        }
    }
    true
}

/// The order in which facts are tried as generators of a closed set.
const GENERATOR_ORDER: [Predicate; 31] = [
    positive,
    negative,
    zero,
    prime,
    composite,
    even,
    odd,
    integer,
    rational,
    irrational,
    algebraic,
    transcendental,
    imaginary,
    nonzero,
    nonnegative,
    nonpositive,
    real,
    extended_positive,
    extended_negative,
    extended_nonzero,
    extended_nonnegative,
    extended_nonpositive,
    noninteger,
    extended_real,
    infinite,
    finite,
    complex,
    hermitian,
    antihermitian,
    polar,
    commutative,
];

/// A small set of facts whose closure is the closure of `facts`,
/// `commutative = true` (the default) left out; in the order of the
/// generators.
pub fn minimal_generators(facts: &[(Predicate, bool)]) -> Vec<(Predicate, bool)> {
    let mut target: Facts = closure(facts);
    if target.get(&commutative) == Some(&true) {
        target.remove(&commutative);
    }
    let mut generators: Vec<(Predicate, bool)> = Vec::new();
    for name in GENERATOR_ORDER {
        for value in [true, false] {
            if target.get(&name) == Some(&value) && closure(&generators).get(&name) != Some(&value) {
                generators.push((name, value));
            }
        }
    }
    for (&name, &value) in &target {
        if closure(&generators).get(&name) != Some(&value) {
            generators.push((name, value));
        }
    }
    generators
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctests() {
        let c = closure(&[(positive, true)]);
        assert_eq!(
            (c[&real], c[&nonzero], c[&negative], c[&zero], c[&extended_positive]),
            (true, true, false, false, true)
        );
        assert!(closure(&[(integer, true), (even, false)])[&odd]);
        assert!(is_consistent(&[(positive, true), (integer, true)]));
        assert!(!is_consistent(&[(positive, true), (zero, true)]));
        assert!(!is_consistent(&[(even, true), (odd, true)]));
        assert!(!is_consistent(&[(real, true), (rational, false), (irrational, false)]));
        assert_eq!(
            minimal_generators(
                &closure(&[(positive, true), (integer, true)])
                    .into_iter()
                    .collect::<Vec<_>>()
            ),
            vec![(positive, true), (integer, true)]
        );
        let names: Vec<&str> = PREDICATES.iter().map(|p| p.name()).collect();
        let mut sorted: Vec<&str> = names.clone();
        sorted.sort();
        let mut by_enum: Vec<Predicate> = PREDICATES.to_vec();
        by_enum.sort();
        assert_eq!(by_enum.iter().map(|p| p.name()).collect::<Vec<_>>(), sorted);
    }
}
