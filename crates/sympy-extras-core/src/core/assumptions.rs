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

// ---------------------------------------------------------------------
// the structural rules: the facts of a node from those of its arguments
// (ported from the version of sympy_extras/core/assumptions.py at 2d17798)

use super::basic::{Basic, Head};
use super::evalf::{numeric_value, parts, real_int, Real};
use crate::numbers::{bit, denom, is_negative, numer, Int, Rat};

/// The facts of a node before their closure, in the order they are set
/// (Python's `dict`: setting a known fact keeps its place).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FactList(Vec<(Predicate, bool)>);

impl FactList {
    pub fn new() -> FactList {
        FactList(Vec::new())
    }

    pub fn get(&self, p: Predicate) -> Option<bool> {
        self.0.iter().find(|(q, _)| *q == p).map(|(_, v)| *v)
    }

    pub fn contains(&self, p: Predicate) -> bool {
        self.get(p).is_some()
    }

    /// `facts[p] = v`.
    pub fn set(&mut self, p: Predicate, v: bool) {
        match self.0.iter_mut().find(|(q, _)| *q == p) {
            Some(entry) => entry.1 = v,
            None => self.0.push((p, v)),
        }
    }

    /// `facts.setdefault(p, v)`.
    pub fn set_default(&mut self, p: Predicate, v: bool) {
        if !self.contains(p) {
            self.0.push((p, v));
        }
    }

    pub fn remove(&mut self, p: Predicate) {
        self.0.retain(|(q, _)| *q != p);
    }

    /// `facts.update(...)`.
    pub fn update(&mut self, items: &[(Predicate, bool)]) {
        for &(p, v) in items {
            self.set(p, v);
        }
    }

    pub fn items(&self) -> &[(Predicate, bool)] {
        &self.0
    }
}

/// `facts.get(name) is True` on closed facts.
fn yes(f: &Facts, p: Predicate) -> bool {
    f.get(&p) == Some(&true)
}

/// `facts.get(name) is False` on closed facts.
fn no(f: &Facts, p: Predicate) -> bool {
    f.get(&p) == Some(&false)
}

fn number_facts(n: &Basic) -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(commutative, true);
    match n.head() {
        Head::Rational { value, .. } => {
            facts.update(&[(rational, true), (finite, true), (imaginary, false), (algebraic, true)]);
            let p: Int = numer(value);
            facts.set(integer, value.is_int());
            facts.set(zero, p.is_zero());
            facts.set(positive, p > Int::ZERO);
            facts.set(negative, is_negative(&p));
            facts.set(irrational, false);
            if value.is_int() {
                facts.set(even, !bit(&p, 0));
                // Python's n.p % 2 == 1 (floor modulo): odd for every odd integer
                facts.set(odd, bit(&p, 0));
                let is_prime: bool = crate::ntheory::isprime(&p);
                facts.set(prime, is_prime);
                facts.set(composite, p > Int::ONE && !is_prime);
            } else {
                facts.set(prime, false);
                facts.set(composite, false);
            }
        }
        Head::Infinity => facts.update(&[
            (extended_real, true),
            (real, false),
            (complex, false),
            (finite, false),
            (infinite, true),
            (extended_positive, true),
            (zero, false),
            (negative, false),
            (integer, false),
            (rational, false),
            (imaginary, false),
            (algebraic, false),
            (transcendental, false),
        ]),
        Head::NegativeInfinity => facts.update(&[
            (extended_real, true),
            (real, false),
            (complex, false),
            (finite, false),
            (infinite, true),
            (extended_negative, true),
            (zero, false),
            (positive, false),
            (integer, false),
            (rational, false),
            (imaginary, false),
            (algebraic, false),
            (transcendental, false),
        ]),
        Head::ComplexInfinity => facts.update(&[
            (real, false),
            (extended_real, false),
            (complex, false),
            (finite, false),
            (infinite, true),
            (zero, false),
            (imaginary, false),
            (integer, false),
            (rational, false),
            (algebraic, false),
            (transcendental, false),
            (noninteger, false),
        ]),
        _ => {} // nan: nothing is known
    }
    facts
}

/// The facts of a number or a named constant (`pi`, `E`, `I`, ...), empty
/// for anything else.
pub fn constant_facts(e: &Basic) -> FactList {
    if e.is_number_node() {
        return number_facts(e);
    }
    let mut facts: FactList = FactList::new();
    match e.head() {
        Head::Pi | Head::Exp1 => facts.update(&[
            (commutative, true),
            (real, true),
            (positive, true),
            (irrational, true),
            (integer, false),
            (prime, false),
            (transcendental, true),
            (composite, false),
        ]),
        Head::GoldenRatio | Head::TribonacciConstant => facts.update(&[
            (commutative, true),
            (real, true),
            (positive, true),
            (irrational, true),
            (integer, false),
            (prime, false),
            (algebraic, true),
            (composite, false),
        ]),
        Head::EulerGamma | Head::Catalan => facts.update(&[(commutative, true), (real, true), (positive, true)]),
        Head::ImaginaryUnit => facts.update(&[
            (commutative, true),
            (imaginary, true),
            (real, false),
            (finite, true),
            (zero, false),
            (integer, false),
            (rational, false),
            (extended_real, false),
            (algebraic, true),
        ]),
        _ => {}
    }
    facts
}

/// Whether `e` is built from numbers and constants only.
pub fn is_number(e: &Basic) -> bool {
    // an undefined function, a derivative, and a node with a Tuple argument
    // (not an expression: Subs) are not numbers
    if e.is_symbol()
        || matches!(e.head(), Head::Undefined(_) | Head::Derivative | Head::Tuple)
        || crate::sets::sets::is_set(e)
        || matches!(e.head(), Head::AccumBounds)
    {
        return false;
    }
    if matches!(e.head(), Head::IndexedRoot) {
        // _is_number_node: no free symbols (the variable is bound)
        return e.free_symbols().is_empty();
    }
    e.args().iter().all(is_number)
}

/// The sign of a sum from the signs of its terms (`extended` selects the
/// extended predicates).
fn sign_facts(sub: &[&Facts], extended: bool, facts: &mut FactList) {
    let (pos, neg, nonneg, nonpos) = if extended {
        (
            extended_positive,
            extended_negative,
            extended_nonnegative,
            extended_nonpositive,
        )
    } else {
        (positive, negative, nonnegative, nonpositive)
    };
    if sub.iter().all(|s| yes(s, pos)) {
        facts.set(pos, true);
    } else if sub.iter().all(|s| yes(s, nonneg)) {
        facts.set(nonneg, true);
        if sub.iter().any(|s| yes(s, pos)) {
            facts.set(pos, true);
        }
    }
    if sub.iter().all(|s| yes(s, neg)) {
        facts.set(neg, true);
    } else if sub.iter().all(|s| yes(s, nonpos)) {
        facts.set(nonpos, true);
        if sub.iter().any(|s| yes(s, neg)) {
            facts.set(neg, true);
        }
    }
}

/// The facts of a sum from the facts of its terms.
pub fn add_facts(args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    let sub: Vec<&Facts> = args.iter().map(facts_of).collect();
    // SymPy's Add._eval_is_infinite: every term decided, exactly one
    // infinite (two might cancel)
    if sub.iter().all(|s| s.contains_key(&infinite)) && sub.iter().filter(|s| yes(s, infinite)).count() == 1 {
        facts.update(&[(infinite, true), (finite, false)]);
    }
    for name in [
        commutative,
        finite,
        integer,
        rational,
        real,
        complex,
        extended_real,
        hermitian,
        antihermitian,
        algebraic,
    ] {
        if sub.iter().all(|s| yes(s, name)) {
            facts.set(name, true);
        } else if name == commutative && sub.iter().any(|s| no(s, name)) {
            facts.set(name, false);
        }
    }
    for name in [rational, integer, algebraic] {
        let domain: Predicate = if name == algebraic { complex } else { real };
        let falses: usize = sub.iter().filter(|s| no(s, name)).count();
        let trues: usize = sub.iter().filter(|s| yes(s, name)).count();
        if sub.iter().all(|s| yes(s, domain)) && falses == 1 && trues == sub.len() - 1 {
            facts.set(name, false);
        }
    }
    if facts.get(algebraic) == Some(false) && sub.iter().filter(|s| yes(s, transcendental)).count() == 1 {
        facts.set(transcendental, true);
    }
    if facts.get(integer) == Some(false)
        && sub.iter().all(|s| yes(s, extended_real))
        && sub.iter().filter(|s| yes(s, noninteger)).count() == 1
    {
        facts.set(noninteger, true);
    }
    sign_facts(&sub, false, &mut facts);
    sign_facts(&sub, true, &mut facts);
    if sub.iter().all(|s| yes(s, zero)) {
        facts.set(zero, true);
    }
    if facts.get(integer) == Some(true) && sub.iter().all(|s| yes(s, even) || yes(s, odd)) {
        let odds: usize = sub.iter().filter(|s| yes(s, odd)).count();
        facts.set(odd, odds % 2 == 1);
        facts.set(even, odds % 2 == 0);
    }
    // a real part and a nonzero imaginary part: not real, not zero
    let imaginary_terms: Vec<&&Facts> = sub.iter().filter(|s| yes(s, imaginary)).collect();
    if !imaginary_terms.is_empty() && sub.iter().all(|s| yes(s, real) || yes(s, imaginary)) {
        if imaginary_terms.len() == sub.len() {
            facts.set(imaginary, true);
        } else if imaginary_terms.len() == 1 && no(imaginary_terms[0], zero) {
            facts.update(&[(real, false), (zero, false), (extended_real, false)]);
            if sub.iter().all(|s| yes(s, finite)) {
                facts.set(finite, true);
            }
            if sub.iter().any(|s| yes(s, real) && no(s, zero)) {
                facts.set(imaginary, false);
            }
        }
    }
    // a real part and a single term which is not real: not real, not zero
    let non_real: Vec<usize> = (0..sub.len())
        .filter(|&i| no(sub[i], real) && yes(sub[i], complex))
        .collect();
    if non_real.len() == 1 && (0..sub.len()).all(|i| yes(sub[i], real) || i == non_real[0]) {
        facts.update(&[(real, false), (extended_real, false), (zero, false)]);
    }
    if !facts.contains(positive) && !facts.contains(negative) && sub.iter().all(|s| yes(s, real)) {
        let numbers: Vec<Basic> = args.iter().filter(|a| is_number(a)).cloned().collect();
        let others: Vec<&Facts> = args
            .iter()
            .zip(&sub)
            .filter(|(a, _)| !is_number(a))
            .map(|(_, s)| *s)
            .collect();
        let sign: Option<i32> = if numbers.is_empty() {
            None
        } else {
            numeric_sign(&numbers)
        };
        match sign {
            Some(s) if others.is_empty() => facts.update(&[(positive, s > 0), (negative, s < 0), (zero, false)]),
            Some(s) if s > 0 && others.iter().all(|o| yes(o, nonnegative)) => {
                facts.update(&[(positive, true), (zero, false)])
            }
            Some(s) if s < 0 && others.iter().all(|o| yes(o, nonpositive)) => {
                facts.update(&[(negative, true), (zero, false)])
            }
            _ => {}
        }
    }
    // SymPy's _monotonic_sign for an integer term and a rational constant:
    // k + c >= 1 + c for a positive integer k
    if args.len() == 2 && !facts.contains(positive) && !facts.contains(negative) {
        let constants: Vec<usize> = (0..2).filter(|&i| args[i].is_rational_class()).collect();
        if constants.len() == 1 {
            let c: Rat = args[constants[0]].as_rational().expect("a rational").clone();
            let other: &Facts = if constants[0] == 0 { sub[1] } else { sub[0] };
            if yes(other, integer) {
                let low: Option<Rat> = if yes(other, positive) {
                    Some(Rat::ONE)
                } else if yes(other, nonnegative) {
                    Some(Rat::ZERO)
                } else {
                    None
                };
                let high: Option<Rat> = if yes(other, negative) {
                    Some(-Rat::ONE)
                } else if yes(other, nonpositive) {
                    Some(Rat::ZERO)
                } else {
                    None
                };
                if let Some(low) = low {
                    if &low + &c >= Rat::ZERO {
                        facts.update(&[(nonnegative, true), (negative, false), (real, true)]);
                        if &low + &c > Rat::ZERO {
                            facts.update(&[(positive, true), (zero, false), (nonpositive, false)]);
                        }
                    }
                }
                if let Some(high) = high {
                    if &high + &c <= Rat::ZERO {
                        facts.update(&[(nonpositive, true), (positive, false), (real, true)]);
                        if &high + &c < Rat::ZERO {
                            facts.update(&[(negative, true), (zero, false), (nonnegative, false)]);
                        }
                    }
                }
            }
        }
    }
    // the real terms are numbers: the sum is not imaginary (SymPy answers
    // None when a symbol is among them)
    if facts.get(imaginary) == Some(false)
        && !args
            .iter()
            .zip(&sub)
            .filter(|(_, s)| yes(s, real))
            .all(|(a, _)| is_number(a))
    {
        facts.remove(imaginary);
    }
    facts
}

fn abs_real(x: &Real) -> Real {
    if x < &Real::ZERO {
        -x.clone()
    } else {
        x.clone()
    }
}

fn sign_of(x: &Real) -> i32 {
    if x > &Real::ZERO {
        1
    } else if x < &Real::ZERO {
        -1
    } else {
        0
    }
}

/// `scale * 2**(-(prec - 20))`, the size below which a value is noise.
fn noise(scale: &Real, prec: usize) -> Real {
    scale / real_int(&Int::from(2u8).pow(prec - 20), prec + 20)
}

/// The facts of a numerical expression found by evaluating it at two
/// precisions: real or not, its sign, purely imaginary; nothing when the
/// value is too small to be trusted.
pub fn numeric_facts(e: &Basic) -> FactList {
    let mut reals: Vec<i32> = Vec::new();
    let mut imaginaries: Vec<i32> = Vec::new();
    for prec in [53usize, 160] {
        let value = match numeric_value(e, prec) {
            Some(v) => v,
            None => return FactList::new(),
        };
        let (re, im) = parts(&value);
        let (are, aim) = (abs_real(&re), abs_real(&im));
        let scale: Real = if are > aim { are.clone() } else { aim.clone() };
        if scale == Real::ZERO {
            return FactList::new();
        }
        let threshold: Real = noise(&scale, prec);
        reals.push(if are <= threshold { 0 } else { sign_of(&re) });
        imaginaries.push(if aim <= threshold { 0 } else { sign_of(&im) });
    }
    let mut facts: FactList = FactList::new();
    if reals[0] != reals[1] || imaginaries[0] != imaginaries[1] {
        return facts;
    }
    if imaginaries[0] != 0 {
        facts.update(&[(real, false), (zero, false)]);
        if reals[0] == 0 {
            facts.set(imaginary, true);
        }
    } else if reals[0] != 0 {
        facts.update(&[
            (real, true),
            (zero, false),
            (positive, reals[0] > 0),
            (negative, reals[0] < 0),
        ]);
    }
    facts
}

/// The sign of a sum of real numbers found numerically at two precisions,
/// `None` when the sum is too small to be trusted.
fn numeric_sign(args: &[Basic]) -> Option<i32> {
    let mut signs: Vec<i32> = Vec::new();
    for prec in [53usize, 160] {
        let mut total: Real = Real::ZERO;
        let mut scale: Real = Real::ZERO;
        for a in args {
            let (re, im) = parts(&numeric_value(a, prec)?);
            if im != Real::ZERO {
                return None;
            }
            let magnitude: Real = abs_real(&re);
            if magnitude > scale {
                scale = magnitude;
            }
            total = &total + &re;
        }
        if scale == Real::ZERO || abs_real(&total) <= noise(&scale, prec) {
            return None;
        }
        signs.push(sign_of(&total));
    }
    if signs[0] == signs[1] {
        Some(signs[0])
    } else {
        None
    }
}

/// The numerical term and the rest of a canonical sum, structurally.
fn split_constant(e: &Basic) -> (Rat, Vec<Basic>) {
    if let Some(r) = e.as_rational() {
        return (r.clone(), Vec::new());
    }
    if e.is_add() {
        let mut constant: Rat = Rat::ZERO;
        let mut rest: Vec<Basic> = Vec::new();
        for a in e.args() {
            match a.as_rational() {
                Some(r) => constant = &constant + r,
                None => rest.push(a.clone()),
            }
        }
        return (constant, rest);
    }
    (Rat::ZERO, vec![e.clone()])
}

/// Whether two factors differ by an odd integer (`n*(n - 1)` is even).
///
/// Python evaluates `g - f`; for evaluated (canonical) factors the
/// difference is a number exactly when their non-numerical terms agree, so
/// the test is structural here until `add` is ported.
fn consecutive_factors(args: &[Basic]) -> bool {
    for (i, f) in args.iter().enumerate() {
        for g in &args[i + 1..] {
            let (cf, rf) = split_constant(f);
            let (cg, rg) = split_constant(g);
            if rf == rg {
                let d: Rat = cg - cf;
                if d.is_int() && bit(&numer(&d), 0) {
                    return true;
                }
            }
        }
    }
    false
}

/// The facts of a product from the facts of its factors.
pub fn mul_facts(args: &[Basic]) -> FactList {
    let mut facts: FactList = FactList::new();
    let sub: Vec<&Facts> = args.iter().map(facts_of).collect();
    for name in [
        commutative,
        finite,
        integer,
        rational,
        real,
        complex,
        extended_real,
        algebraic,
    ] {
        if sub.iter().all(|s| yes(s, name)) {
            facts.set(name, true);
        } else if name == commutative && sub.iter().any(|s| no(s, name)) {
            facts.set(name, false);
        }
    }
    if sub.iter().any(|s| yes(s, zero)) && sub.iter().all(|s| !no(s, finite)) {
        facts.set(zero, true);
    }
    if sub.iter().all(|s| yes(s, nonzero)) {
        facts.set(nonzero, true);
    }
    if sub.iter().all(|s| yes(s, extended_nonzero)) {
        facts.set(extended_nonzero, true);
    }
    if sub.iter().all(|s| no(s, zero)) {
        facts.set(zero, false);
        // SymPy's Mul._eval_is_infinite: an infinite factor and no factor
        // which may be zero make an infinite product
        if sub.iter().any(|s| yes(s, infinite)) {
            facts.update(&[(infinite, true), (finite, false)]);
        }
    }
    for extended in [false, true] {
        let (pos, neg, nonneg, nonpos, re) = if extended {
            (
                extended_positive,
                extended_negative,
                extended_nonnegative,
                extended_nonpositive,
                extended_real,
            )
        } else {
            (positive, negative, nonnegative, nonpositive, real)
        };
        if sub.iter().all(|s| yes(s, re)) {
            if sub.iter().all(|s| yes(s, pos) || yes(s, neg)) {
                let negatives: usize = sub.iter().filter(|s| yes(s, neg)).count();
                facts.set(pos, negatives % 2 == 0);
                facts.set(neg, negatives % 2 == 1);
            } else if sub.iter().all(|s| yes(s, nonneg) || yes(s, nonpos)) {
                let negatives: usize = sub.iter().filter(|s| yes(s, nonpos) && !yes(s, nonneg)).count();
                facts.set(if negatives % 2 == 0 { nonneg } else { nonpos }, true);
            }
        }
    }
    if facts.get(integer) == Some(true) {
        if sub.iter().any(|s| yes(s, even)) || consecutive_factors(args) {
            facts.set(even, true);
        } else if sub.iter().all(|s| yes(s, odd)) {
            facts.set(odd, true);
        }
    }
    let irrationals: Vec<usize> = (0..sub.len()).filter(|&i| yes(sub[i], irrational)).collect();
    if irrationals.len() == 1
        && (0..sub.len()).all(|i| i == irrationals[0] || (yes(sub[i], rational) && no(sub[i], zero)))
    {
        facts.update(&[(irrational, true), (rational, false)]);
    } else if args.len() == 2 && args[0].as_rational() == Some(&crate::numbers::rat(1, 2)) {
        // SymPy: an even number halved is an integer, an odd one is not
        if yes(sub[1], even) {
            facts.set(integer, true);
        } else if yes(sub[1], odd) {
            facts.update(&[(integer, false), (rational, true)]);
        }
    }
    let imaginary_factors: usize = sub.iter().filter(|s| yes(s, imaginary)).count();
    if imaginary_factors == 1
        && sub.iter().all(|s| yes(s, real) || yes(s, imaginary))
        && sub.iter().all(|s| no(s, zero))
    {
        facts.update(&[(imaginary, true), (real, false), (zero, false), (extended_real, false)]);
        if sub.iter().all(|s| yes(s, finite)) {
            facts.set(finite, true);
        }
    }
    let transcendentals: usize = sub.iter().filter(|s| yes(s, transcendental)).count();
    if transcendentals == 1
        && sub
            .iter()
            .filter(|s| !yes(s, transcendental))
            .all(|s| yes(s, algebraic) && no(s, zero))
    {
        facts.update(&[(transcendental, true), (algebraic, false)]);
    }
    // SymPy: a product is polar when it has a polar factor and every factor
    // is polar or positive, and not polar otherwise
    facts.set(
        polar,
        sub.iter().any(|s| yes(s, polar)) && sub.iter().all(|s| yes(s, polar) || yes(s, positive)),
    );
    if !facts.contains(real) && args.iter().all(is_number) {
        let product: Basic = Basic::new(Head::Mul, args.to_vec());
        for &(name, known) in numeric_facts(&product).items() {
            facts.set_default(name, known);
        }
    }
    // nonzero real factors and a single factor which is not real: not real
    let non_real: Vec<usize> = (0..sub.len())
        .filter(|&i| no(sub[i], real) && yes(sub[i], complex))
        .collect();
    if non_real.len() == 1 && (0..sub.len()).all(|i| (yes(sub[i], real) && no(sub[i], zero)) || i == non_real[0]) {
        facts.update(&[(real, false), (extended_real, false)]);
        if no(sub[non_real[0]], zero) {
            facts.set(zero, false);
        }
        if no(sub[non_real[0]], imaginary) {
            facts.set(imaginary, false);
        }
    }
    facts
}

/// The facts of a power from the facts of its base and exponent.
pub fn pow_facts(base: &Basic, exp: &Basic) -> FactList {
    let mut facts: FactList = FactList::new();
    let (b, e) = (facts_of(base), facts_of(exp));
    if yes(b, commutative) && yes(e, commutative) {
        facts.set(commutative, true);
    } else if no(b, commutative) || no(e, commutative) {
        facts.set(commutative, false);
    }
    let exp_rational: Option<&Rat> = exp.as_rational();
    if yes(b, integer) && yes(e, integer) && yes(e, nonnegative) {
        facts.set(integer, true);
    }
    if yes(b, rational) && yes(e, integer) && (yes(e, nonnegative) || yes(b, nonzero)) {
        facts.set(rational, true);
    }
    if yes(b, real) && yes(e, integer) && (yes(e, nonnegative) || yes(b, nonzero)) {
        facts.set(real, true);
    }
    if yes(b, extended_real) && yes(e, integer) && yes(e, positive) {
        facts.set(extended_real, true);
    }
    if yes(b, positive) && yes(e, real) {
        facts.update(&[(positive, true), (real, true)]);
    }
    if yes(b, extended_positive) && yes(e, positive) {
        facts.set(extended_positive, true);
    }
    if yes(b, extended_positive) && yes(e, real) {
        facts.set(extended_nonnegative, true);
    }
    if yes(b, nonnegative) && yes(e, positive) {
        facts.update(&[(nonnegative, true), (real, true)]);
    }
    if yes(b, real) && yes(b, nonzero) && yes(e, even) {
        facts.set(positive, true);
    }
    if yes(b, real) && yes(e, even) && (yes(e, nonnegative) || yes(b, nonzero)) {
        facts.set(nonnegative, true); // x**(-2) may be zoo
    }
    if yes(b, negative) && yes(e, odd) {
        facts.set(negative, true);
    }
    // SymPy's Pow._eval_is_extended_real (real base and exponent)
    if yes(b, extended_real) && yes(e, extended_real) {
        if yes(b, extended_positive)
            || (yes(b, extended_nonnegative) && yes(e, extended_nonnegative))
            || (yes(e, integer) && (yes(b, extended_nonzero) || yes(e, nonnegative)))
        {
            facts.set_default(extended_real, true);
        } else if yes(b, extended_negative) && exp_rational.is_some() {
            facts.set_default(extended_real, false);
        }
    }
    // SymPy's Pow._eval_is_extended_negative
    let mut negative_value: Option<bool> = None;
    if exp_rational == Some(&crate::numbers::rat(1, 2)) && (yes(b, complex) || yes(b, extended_real)) {
        // only for a base known to be a number: sqrt(a) of a plain symbol is
        // not known to be non-negative (SymPy's is_extended_negative is None)
        negative_value = Some(false);
    } else if yes(b, extended_negative) {
        if yes(e, odd) && yes(b, finite) {
            negative_value = Some(true);
        } else if yes(e, even) {
            negative_value = Some(false);
        }
    } else if yes(b, extended_positive) || yes(b, zero) {
        // Python's two branches, an extended positive and a zero base
        if yes(e, extended_real) {
            negative_value = Some(false);
        }
    } else if yes(b, extended_nonnegative) {
        if yes(e, extended_nonnegative) {
            negative_value = Some(false);
        }
    } else if (yes(b, extended_nonpositive) || yes(b, extended_real)) && yes(e, even) {
        negative_value = Some(false);
    }
    if let Some(v) = negative_value {
        facts.set_default(extended_negative, v);
    }
    if yes(b, zero) && yes(e, positive) {
        facts.set(zero, true);
    }
    if yes(b, nonzero) && yes(e, real) && yes(e, finite) {
        facts.set(zero, false);
    }
    if yes(b, finite) && yes(e, negative) && yes(e, finite) {
        facts.set(zero, false); // 1/(p - 4) is never zero
    }
    if yes(b, real) && yes(e, integer) {
        facts.set(imaginary, false);
    }
    let unit_integer: bool = exp_rational.is_some_and(|r| r.is_int() && crate::numbers::abs(&numer(r)).is_one());
    if no(b, real) && yes(b, complex) && unit_integer {
        facts.update(&[(real, false), (extended_real, false)]); // 1/(pi + I)
    }
    if !facts.contains(real) && is_number(base) && is_number(exp) {
        let power: Basic = Basic::new(Head::Pow, vec![base.clone(), exp.clone()]);
        for &(name, known) in numeric_facts(&power).items() {
            facts.set_default(name, known);
        }
    }
    if yes(b, finite) && yes(e, integer) && yes(e, nonnegative) {
        facts.set(finite, true);
    }
    if yes(b, finite) && yes(e, positive) {
        facts.set(finite, true); // sqrt(log(n))
    }
    if yes(b, positive) && yes(b, finite) && yes(e, finite) {
        facts.set(finite, true); // p**I
    }
    if yes(b, finite) && no(b, zero) && yes(e, real) && yes(e, finite) {
        facts.set(finite, true); // 1/sqrt(p + I)
    }
    if let Some(r) = exp_rational {
        if yes(b, algebraic) && (!is_negative(&numer(r)) || no(b, zero)) {
            facts.set(algebraic, true);
        }
        let unit_numerator: bool = crate::numbers::abs(&numer(r)).is_one();
        if no(b, real) && yes(b, complex) && unit_numerator && denom(r) > Int::ONE {
            facts.update(&[(real, false), (extended_real, false)]); // a root of a non-real number is not real
        }
        if yes(b, transcendental) && !numer(r).is_zero() {
            facts.update(&[(transcendental, true), (algebraic, false)]);
        }
    }
    if yes(b, polar) {
        facts.set(polar, true);
    }
    if let (Some(rb), Some(re)) = (base.as_rational(), exp_rational) {
        // a non-integer power of a positive rational which is not a perfect
        // power is irrational (pow keeps the node when it is not)
        if *rb > Rat::ZERO && !re.is_int() && super::power::rational_power_is_kept(rb, re) {
            facts.update(&[(irrational, true), (real, true)]);
        }
    }
    facts
}

/// The facts of the node before their closure (`_eval_facts`).
fn eval_facts(e: &Basic) -> FactList {
    match e.head() {
        Head::Symbol(s) | Head::Dummy(s, _) => {
            let mut facts: FactList = FactList(s.given.clone());
            facts.set_default(commutative, true);
            facts
        }
        Head::Wild(w) => {
            let mut facts: FactList = FactList(w.symbol.given.clone());
            facts.set_default(commutative, true);
            facts
        }
        Head::Add => add_facts(e.args()),
        Head::Mul => mul_facts(e.args()),
        Head::Pow => pow_facts(&e.args()[0], &e.args()[1]),
        Head::Function(f) => function_facts(e, *f),
        Head::Undefined(u) => {
            // AppliedUndef._eval_facts: the given assumptions, commutative
            // from the arguments; then facts_of's rule of the functions
            let mut facts: FactList = FactList(u.assumptions.clone());
            let args: Vec<Option<bool>> = e
                .args()
                .iter()
                .map(|a| facts_of(a).get(&commutative).copied())
                .collect();
            if args.iter().all(|v| *v == Some(true)) {
                facts.set_default(commutative, true);
            } else if args.contains(&Some(false)) {
                facts.set_default(commutative, false);
            }
            if facts.get(commutative) == Some(true) && e.args().iter().any(|a| no(facts_of(a), commutative)) {
                facts.set(commutative, false);
            }
            facts
        }
        Head::Derivative | Head::Subs => {
            let mut facts: FactList = FactList::new();
            if let Some(v) = facts_of(&e.args()[0]).get(&commutative) {
                facts.set(commutative, *v);
            }
            facts
        }
        Head::Tuple => FactList::new(),
        Head::IndexedRoot => crate::polys::cad::indexedroot::indexed_root_facts(e),
        // a boolean is not an expression: no facts
        Head::AccumBounds => super::accumbounds::bounds_facts(e),
        _ if super::logic::is_boolean(e) || crate::sets::sets::is_set(e) => FactList::new(),
        _ => constant_facts(e),
    }
}

/// The facts of an applied function: its class's, then (as Python's
/// `facts_of`) not commutative with a noncommutative argument, and the sign
/// of a real numerical value found by evaluating it.
fn function_facts(e: &Basic, f: super::basic::Function) -> FactList {
    use crate::functions::elementary::exponential::{exp_facts, log_facts};
    let mut facts: FactList = match f {
        super::basic::Function::Exp => exp_facts(&e.args()[0]),
        super::basic::Function::Log => log_facts(&e.args()[0]),
        super::basic::Function::Sin
        | super::basic::Function::Cos
        | super::basic::Function::Tan
        | super::basic::Function::Cot
        | super::basic::Function::Sec
        | super::basic::Function::Csc => {
            crate::functions::elementary::trigonometric::trigonometric_facts(f, &e.args()[0])
        }
        super::basic::Function::Sinh
        | super::basic::Function::Cosh
        | super::basic::Function::Tanh
        | super::basic::Function::Coth
        | super::basic::Function::Sech
        | super::basic::Function::Csch => crate::functions::elementary::hyperbolic::hyperbolic_facts(f, &e.args()[0]),
        f if crate::functions::elementary::inverse::is_inverse(f) => {
            crate::functions::elementary::inverse::inverse_facts(f, &e.args()[0])
        }
        g @ (super::basic::Function::Floor | super::basic::Function::Ceiling | super::basic::Function::Frac) => {
            crate::functions::elementary::integers::rounding_facts(g, &e.args()[0])
        }
        g @ (super::basic::Function::Factorial
        | super::basic::Function::Binomial
        | super::basic::Function::RisingFactorial
        | super::basic::Function::FallingFactorial
        | super::basic::Function::Subfactorial
        | super::basic::Function::Factorial2) => {
            crate::functions::combinatorial::factorials::combinatorial_facts(g, e.args())
        }
        super::basic::Function::DiracDelta | super::basic::Function::Heaviside => {
            crate::functions::special::delta_functions::delta_facts()
        }
        g @ (super::basic::Function::Max | super::basic::Function::Min) => {
            crate::functions::elementary::miscellaneous::minmax_facts(g, e.args())
        }
        g @ (super::basic::Function::Arg | super::basic::Function::Atan2) => {
            crate::functions::elementary::argument::argument_facts(g, e.args())
        }
        other => crate::functions::elementary::complexes::complexes_facts(other, &e.args()[0]),
    };
    if facts.get(commutative) == Some(true) && e.args().iter().any(|a| no(facts_of(a), commutative)) {
        facts.set(commutative, false);
    }
    if ![positive, negative, zero].iter().any(|p| facts.contains(*p)) && is_number(e) {
        let found: FactList = numeric_facts(e);
        if found.get(real) == Some(true) {
            for (name, known) in found.0.iter() {
                facts.set_default(*name, *known);
            }
        }
    }
    facts
}

/// The closed facts of `e`, computed once and cached on the node.
pub fn facts_of(e: &Basic) -> &Facts {
    e.cached_facts(|| closure(eval_facts(e).items()))
}

/// `True`, `False` or `None` for the predicate on `e`.
pub fn ask_structural(e: &Basic, name: Predicate) -> Option<bool> {
    facts_of(e).get(&name).copied()
}

#[cfg(test)]
mod structural_tests {
    use super::*;
    use crate::core::Symbol;

    fn add(args: Vec<Basic>) -> Basic {
        Basic::new(Head::Add, args)
    }

    fn mul(args: Vec<Basic>) -> Basic {
        Basic::new(Head::Mul, args)
    }

    fn pow(b: Basic, e: Basic) -> Basic {
        Basic::new(Head::Pow, vec![b, e])
    }

    #[test]
    fn doctests() {
        let p = Symbol::new("p", &[(positive, true)]).0;
        let n = Symbol::new("n", &[(integer, true)]).0;
        let x = Symbol::new("x", &[]).0;
        let p1 = add(vec![Basic::one(), p.clone()]);
        assert_eq!(
            (ask_structural(&p1, positive), ask_structural(&p1, zero)),
            (Some(true), Some(false))
        );
        // n**2 + 1 > 0, 2*n even, n of unknown sign
        let n2 = add(vec![Basic::one(), pow(n.clone(), Basic::int(2))]);
        assert_eq!(ask_structural(&n2, positive), Some(true));
        assert_eq!(ask_structural(&mul(vec![Basic::int(2), n.clone()]), even), Some(true));
        assert_eq!(ask_structural(&n, positive), None);
        // is_number
        assert!(is_number(&add(vec![Basic::int(2), Basic::pi()])));
        assert!(is_number(&pow(Basic::int(2), Basic::half())));
        assert!(!is_number(&add(vec![Basic::one(), x.clone()])));
        // numeric facts: pi - 3 > 0, (-1 - I)**-2 is imaginary
        let pi3 = add(vec![Basic::int(-3), Basic::pi()]);
        assert_eq!(ask_structural(&pi3, positive), Some(true));
        let z = pow(
            add(vec![Basic::int(-1), mul(vec![Basic::int(-1), Basic::imaginary_unit()])]),
            Basic::int(-2),
        );
        let f = numeric_facts(&z);
        assert_eq!(
            (f.get(real), f.get(zero), f.get(imaginary)),
            (Some(false), Some(false), Some(true))
        );
        // sqrt(2) is irrational, sqrt(4) is not decided by the node
        assert_eq!(
            ask_structural(&pow(Basic::int(2), Basic::half()), irrational),
            Some(true)
        );
        assert_eq!(ask_structural(&pow(Basic::int(4), Basic::half()), irrational), None);
        // sqrt(x) of a plain symbol is not known to be non-negative, sqrt(2) is
        assert_eq!(ask_structural(&pow(x.clone(), Basic::half()), extended_negative), None);
        assert_eq!(
            ask_structural(&pow(Basic::int(2), Basic::half()), extended_negative),
            Some(false)
        );
        // n*(n + 1) is even
        let consecutive = mul(vec![n.clone(), add(vec![Basic::one(), n.clone()])]);
        assert_eq!(ask_structural(&consecutive, even), Some(true));
        // I*p is imaginary, x has nothing
        assert_eq!(
            ask_structural(&mul(vec![Basic::imaginary_unit(), p.clone()]), imaginary),
            Some(true)
        );
        assert_eq!(ask_structural(&x, real), None);
        assert_eq!(ask_structural(&Basic::oo(), extended_positive), Some(true));
        assert_eq!(ask_structural(&Basic::int(7), prime), Some(true));
        assert_eq!(ask_structural(&Basic::int(-3), odd), Some(true));
    }
}
