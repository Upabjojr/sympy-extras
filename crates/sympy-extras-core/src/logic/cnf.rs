//! Conjunctive normal forms of the Booleans of the core, encoded with
//! integers for a SAT solver, ported from `sympy_extras.logic.cnf`: the
//! formula in negation normal form, the conjunctions distributed over the
//! disjunctions without auxiliary variables (SymPy's `CNF.to_CNF`), the
//! clauses and literals without duplicates in the order first met, a
//! tautological clause kept; an optional expansion replaces atoms by
//! equivalent formulas.

// PORTED FROM SYMPY: sympy/assumptions/cnf.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::core::basic::{Basic, Head};

/// A clause: literals, `+i` for the atom number `i` (from 1), `-i` for its
/// negation.
type Clause = BTreeSet<i64>;

/// A replacement of an atom by an equivalent formula, `None` to keep it.
pub type Expansion<'a> = &'a dyn Fn(&Basic) -> Option<Basic>;

/// A conjunction of clauses over numbered atoms (`Clauses`): `atoms[i - 1]`
/// is the atom of the literals `i` and `-i`; each clause sorted by atom,
/// the negative literal first; an unsatisfiable conjunction is the single
/// empty clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Clauses {
    pub atoms: Vec<Basic>,
    pub clauses: Vec<Vec<i64>>,
}

impl Clauses {
    /// Whether a clause is empty (the conjunction is unsatisfiable).
    pub fn is_false(&self) -> bool {
        self.clauses.iter().any(|c| c.is_empty())
    }
}

fn dedup(clauses: Vec<Clause>) -> Vec<Clause> {
    let mut seen: HashSet<Clause> = HashSet::new();
    clauses.into_iter().filter(|c| seen.insert(c.clone())).collect()
}

/// The clauses of a conjunction.
fn conj(parts: Vec<Vec<Clause>>) -> Vec<Clause> {
    dedup(parts.into_iter().flatten().collect())
}

/// The clauses of a disjunction, by distribution (SymPy's `CNF._or`).
fn disj(parts: Vec<Vec<Clause>>) -> Vec<Clause> {
    let mut result: Vec<Clause> = vec![Clause::new()];
    for p in parts {
        let mut next: Vec<Clause> = Vec::new();
        for a in &result {
            for b in &p {
                next.push(a.union(b).copied().collect());
            }
        }
        result = dedup(next);
    }
    result
}

/// The `k`-element subsets of `0..n`, in `itertools.combinations` order.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn go(n: usize, k: usize, start: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            go(n, k, i + 1, current, out);
            current.pop();
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    go(n, k, 0, &mut Vec::new(), &mut out);
    out
}

/// The numbering of the atoms met while a formula is converted.
struct Encoder<'a> {
    atoms: Vec<Basic>,
    index: HashMap<Basic, i64>,
    expand: Option<Expansion<'a>>,
}

impl Encoder<'_> {
    fn variable(&mut self, atom: &Basic) -> i64 {
        if let Some(found) = self.index.get(atom) {
            return *found;
        }
        self.atoms.push(atom.clone());
        let n: i64 = self.atoms.len() as i64;
        self.index.insert(atom.clone(), n);
        n
    }

    // PORTED FROM SYMPY: sympy/assumptions/cnf.py:to_NNF, distribute_AND_over_OR
    /// The clauses of `f` (`positive`) or of `~f`.
    fn clauses(&mut self, f: &Basic, positive: bool) -> Vec<Clause> {
        let args: Vec<Basic> = f.args().to_vec();
        match f.head() {
            Head::BooleanTrue => {
                return if positive { Vec::new() } else { vec![Clause::new()] };
            }
            Head::BooleanFalse => {
                return if positive { vec![Clause::new()] } else { Vec::new() };
            }
            Head::Not => return self.clauses(&args[0], !positive),
            Head::And | Head::Or => {
                let parts: Vec<Vec<Clause>> = args.iter().map(|a| self.clauses(a, positive)).collect();
                let conjunctive: bool = matches!(f.head(), Head::And) == positive;
                return if conjunctive { conj(parts) } else { disj(parts) };
            }
            Head::Implies => {
                let (a, b) = (&args[0], &args[1]);
                if positive {
                    let parts = vec![self.clauses(a, false), self.clauses(b, true)];
                    return disj(parts);
                }
                let parts = vec![self.clauses(a, true), self.clauses(b, false)];
                return conj(parts);
            }
            Head::Equivalent => {
                // a chain of implications closed by the first argument
                let n: usize = args.len();
                let mut outer: Vec<Vec<Clause>> = Vec::new();
                for i in 0..n {
                    let (a, b) = (&args[i], &args[(i + 1) % n]);
                    if positive {
                        let parts = vec![self.clauses(a, false), self.clauses(b, true)];
                        outer.push(disj(parts));
                    } else {
                        let parts = vec![self.clauses(a, true), self.clauses(b, false)];
                        outer.push(conj(parts));
                    }
                }
                return if positive { conj(outer) } else { disj(outer) };
            }
            Head::Xor => {
                // one clause excluding each assignment with an even number
                // of true arguments: those in `neg`
                let n: usize = args.len();
                let mut evens: Vec<Vec<usize>> = Vec::new();
                let mut i: usize = 0;
                while i <= n {
                    evens.extend(combinations(n, i));
                    i += 2;
                }
                let mut outer: Vec<Vec<Clause>> = Vec::new();
                for neg in &evens {
                    let mut parts: Vec<Vec<Clause>> = Vec::new();
                    for (j, s) in args.iter().enumerate() {
                        let polarity: bool = if positive { !neg.contains(&j) } else { neg.contains(&j) };
                        parts.push(self.clauses(s, polarity));
                    }
                    outer.push(if positive { disj(parts) } else { conj(parts) });
                }
                return if positive { conj(outer) } else { disj(outer) };
            }
            Head::Ite => {
                let (c, t, e) = (&args[0], &args[1], &args[2]);
                if positive {
                    let first = vec![self.clauses(c, false), self.clauses(t, true)];
                    let first = disj(first);
                    let second = vec![self.clauses(c, true), self.clauses(e, true)];
                    let second = disj(second);
                    return conj(vec![first, second]);
                }
                let first = vec![self.clauses(c, true), self.clauses(t, false)];
                let first = conj(first);
                let second = vec![self.clauses(c, false), self.clauses(e, false)];
                let second = conj(second);
                return disj(vec![first, second]);
            }
            _ => {}
        }
        if let Some(expand) = self.expand {
            if let Some(expanded) = expand(f) {
                return self.clauses(&expanded, positive);
            }
        }
        let v: i64 = self.variable(f);
        vec![Clause::from([if positive { v } else { -v }])]
    }
}

fn sorted_clause(c: &Clause) -> Vec<i64> {
    let mut v: Vec<i64> = c.iter().copied().collect();
    v.sort_by_key(|lit| (lit.abs(), *lit));
    v
}

/// Renumber the atoms in the order in which the clauses use them, dropping
/// those which no clause uses.
fn compact(atoms: &[Basic], clauses: &[Clause]) -> Clauses {
    if clauses.iter().any(|c| c.is_empty()) {
        return Clauses {
            atoms: Vec::new(),
            clauses: vec![Vec::new()],
        };
    }
    let mut order: HashMap<i64, i64> = HashMap::new();
    let mut new_atoms: Vec<Basic> = Vec::new();
    for c in clauses {
        for lit in sorted_clause(c) {
            if !order.contains_key(&lit.abs()) {
                order.insert(lit.abs(), order.len() as i64 + 1);
                new_atoms.push(atoms[(lit.abs() - 1) as usize].clone());
            }
        }
    }
    let renumbered: Vec<Vec<i64>> = clauses
        .iter()
        .map(|c| {
            let mut v: Vec<i64> = c
                .iter()
                .map(|lit| {
                    if *lit > 0 {
                        order[&lit.abs()]
                    } else {
                        -order[&lit.abs()]
                    }
                })
                .collect();
            v.sort_by_key(|lit| (lit.abs(), *lit));
            v
        })
        .collect();
    Clauses {
        atoms: new_atoms,
        clauses: renumbered,
    }
}

/// The conjunctive normal form of a formula: its atoms and its clauses
/// (`to_clauses`).
pub fn to_clauses(formula: &Basic, expand: Option<Expansion>) -> Clauses {
    let mut encoder: Encoder = Encoder {
        atoms: Vec::new(),
        index: HashMap::new(),
        expand,
    };
    let clauses: Vec<Clause> = encoder.clauses(formula, true);
    compact(&encoder.atoms, &clauses)
}

/// The conjunction of conjunctive normal forms, on one numbering of their
/// atoms (`conjunction`).
pub fn conjunction(parts: &[Clauses]) -> Clauses {
    let mut atoms: Vec<Basic> = Vec::new();
    let mut index: HashMap<Basic, i64> = HashMap::new();
    let mut clauses: Vec<Clause> = Vec::new();
    for part in parts {
        let mut numbers: Vec<i64> = Vec::new();
        for atom in &part.atoms {
            let found: i64 = match index.get(atom) {
                Some(f) => *f,
                None => {
                    atoms.push(atom.clone());
                    let n: i64 = atoms.len() as i64;
                    index.insert(atom.clone(), n);
                    n
                }
            };
            numbers.push(found);
        }
        for c in &part.clauses {
            clauses.push(
                c.iter()
                    .map(|lit| {
                        let v: i64 = numbers[(lit.abs() - 1) as usize];
                        if *lit > 0 {
                            v
                        } else {
                            -v
                        }
                    })
                    .collect(),
            );
        }
    }
    compact(&atoms, &dedup(clauses))
}
