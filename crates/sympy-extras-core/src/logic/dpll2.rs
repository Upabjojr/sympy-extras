//! A SAT solver, ported from `sympy_extras.logic.dpll2`: the DPLL
//! algorithm with watched literals and the VSIDS heuristic (SymPy's
//! default settings: no clause learning, no theory), enumerating every
//! model of a conjunction of clauses; the models come in a deterministic
//! order.

// PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py, sympy/logic/inference.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashSet};

use crate::core::basic::Basic;

use super::cnf::Clauses;

/// A decision level: its decision literal (`0` for the base level), the
/// literals it set and whether the decision was flipped.
struct Level {
    decision: i64,
    var_settings: HashSet<i64>,
    flipped: bool,
}

impl Level {
    fn new(decision: i64, flipped: bool) -> Level {
        Level {
            decision,
            var_settings: HashSet::new(),
            flipped,
        }
    }
}

/// The state of a DPLL search over clauses on the variables `1, ...,
/// variables` (`SATSolver`).
pub struct SatSolver {
    var_settings: HashSet<i64>,
    is_unsatisfied: bool,
    unit_prop_queue: Vec<i64>,
    sentinels: BTreeMap<i64, BTreeSet<usize>>,
    variable_set: Vec<bool>,
    clauses: Vec<Vec<i64>>,
    lit_heap: BinaryHeap<Reverse<(i64, i64)>>,
    lit_scores: BTreeMap<i64, i64>,
    levels: Vec<Level>,
}

/// A model: the variables (from 1) to their values, in increasing order.
pub type Model = Vec<(usize, bool)>;

impl SatSolver {
    pub fn new(clauses: &[Vec<i64>], variables: usize) -> SatSolver {
        let mut solver = SatSolver {
            var_settings: HashSet::new(),
            is_unsatisfied: false,
            unit_prop_queue: Vec::new(),
            sentinels: BTreeMap::new(),
            variable_set: vec![false; variables + 1],
            clauses: clauses.to_vec(),
            lit_heap: BinaryHeap::new(),
            lit_scores: BTreeMap::new(),
            levels: vec![Level::new(0, false)],
        };
        // SATSolver._initialize_clauses
        let mut occurrence_count: BTreeMap<i64, i64> = BTreeMap::new();
        for i in 0..solver.clauses.len() {
            let clause: Vec<i64> = solver.clauses[i].clone();
            if clause.len() == 1 {
                solver.unit_prop_queue.push(clause[0]);
                continue;
            }
            solver.sentinels.entry(clause[0]).or_default().insert(i);
            solver.sentinels.entry(clause[clause.len() - 1]).or_default().insert(i);
            for lit in &clause {
                *occurrence_count.entry(*lit).or_default() += 1;
            }
        }
        // SATSolver._vsids_init
        for var in 1..=variables as i64 {
            for lit in [var, -var] {
                let score: i64 = -occurrence_count.get(&lit).copied().unwrap_or(0);
                solver.lit_scores.insert(lit, score);
                solver.lit_heap.push(Reverse((score, lit)));
            }
        }
        solver
    }

    fn current_level(&mut self) -> &mut Level {
        self.levels.last_mut().expect("a level")
    }

    // PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._find_model
    /// The models (at most `limit` of them).
    pub fn find_models(&mut self, limit: Option<usize>) -> Vec<Model> {
        let mut models: Vec<Model> = Vec::new();
        let mut flip_var: bool = false;
        self.simplify();
        if self.is_unsatisfied {
            return models;
        }
        loop {
            let lit: i64;
            if flip_var {
                flip_var = false;
                lit = self.current_level().decision;
            } else {
                lit = self.vsids_calculate();
                if lit == 0 {
                    let mut settings: Vec<i64> = self.var_settings.iter().copied().collect();
                    settings.sort_by_key(|v| v.abs());
                    models.push(settings.iter().map(|v| (v.unsigned_abs() as usize, *v > 0)).collect());
                    if limit.is_some_and(|l| models.len() >= l) {
                        return models;
                    }
                    while self.current_level().flipped {
                        self.undo();
                    }
                    if self.levels.len() == 1 {
                        return models;
                    }
                    let flip_lit: i64 = -self.current_level().decision;
                    self.undo();
                    self.levels.push(Level::new(flip_lit, true));
                    flip_var = true;
                    continue;
                }
                self.levels.push(Level::new(lit, false));
            }
            self.assign_literal(lit);
            self.simplify();
            if self.is_unsatisfied {
                self.is_unsatisfied = false;
                while self.current_level().flipped {
                    self.undo();
                    if self.levels.len() == 1 {
                        return models;
                    }
                }
                let flip_lit: i64 = -self.current_level().decision;
                self.undo();
                self.levels.push(Level::new(flip_lit, true));
                flip_var = true;
            }
        }
    }

    fn clause_sat(&self, cls: usize) -> bool {
        self.clauses[cls].iter().any(|lit| self.var_settings.contains(lit))
    }

    fn is_sentinel(&self, lit: i64, cls: usize) -> bool {
        self.sentinels.get(&lit).is_some_and(|s| s.contains(&cls))
    }

    // PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._assign_literal
    fn assign_literal(&mut self, lit: i64) {
        self.var_settings.insert(lit);
        self.current_level().var_settings.insert(lit);
        self.variable_set[lit.unsigned_abs() as usize] = true;
        let watching: Vec<usize> = self.sentinels.entry(-lit).or_default().iter().copied().collect();
        for cls in watching {
            if self.clause_sat(cls) {
                continue;
            }
            let mut other_sentinel: i64 = 0;
            let clause: Vec<i64> = self.clauses[cls].clone();
            for newlit in clause {
                if newlit != -lit {
                    if self.is_sentinel(newlit, cls) {
                        other_sentinel = newlit;
                    } else if !self.variable_set[newlit.unsigned_abs() as usize] {
                        self.sentinels.get_mut(&-lit).expect("watched").remove(&cls);
                        self.sentinels.entry(newlit).or_default().insert(cls);
                        other_sentinel = 0;
                        break;
                    }
                }
            }
            if other_sentinel != 0 {
                self.unit_prop_queue.push(other_sentinel);
            }
        }
    }

    // PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._undo
    fn undo(&mut self) {
        let mut settings: Vec<i64> = self.current_level().var_settings.iter().copied().collect();
        settings.sort_by_key(|v| v.abs());
        for lit in settings {
            self.var_settings.remove(&lit);
            self.vsids_lit_unset(lit);
            self.variable_set[lit.unsigned_abs() as usize] = false;
        }
        self.levels.pop();
    }

    fn simplify(&mut self) {
        // pure literal elimination is disabled in SymPy's solver
        while self.unit_prop() {}
    }

    // PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._unit_prop
    fn unit_prop(&mut self) -> bool {
        let result: bool = !self.unit_prop_queue.is_empty();
        while let Some(next_lit) = self.unit_prop_queue.pop() {
            if self.var_settings.contains(&-next_lit) {
                self.is_unsatisfied = true;
                self.unit_prop_queue.clear();
                return false;
            }
            self.assign_literal(next_lit);
        }
        result
    }

    // PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._vsids_calculate
    fn vsids_calculate(&mut self) -> i64 {
        loop {
            let Some(Reverse((_, lit))) = self.lit_heap.peek().copied() else {
                return 0;
            };
            if !self.variable_set[lit.unsigned_abs() as usize] {
                self.lit_heap.pop();
                return lit;
            }
            self.lit_heap.pop();
        }
    }

    fn vsids_lit_unset(&mut self, lit: i64) {
        let var: i64 = lit.abs();
        self.lit_heap.push(Reverse((self.lit_scores[&var], var)));
        self.lit_heap.push(Reverse((self.lit_scores[&-var], -var)));
    }
}

/// The models of a conjunctive normal form (at most `limit`), each
/// assigning every atom, as SymPy's `satisfiable(..., all_models=True)`
/// with the `dpll2` solver (`propositional_models`).
pub fn propositional_models(cnf: &Clauses, limit: Option<usize>) -> Vec<Vec<(Basic, bool)>> {
    if cnf.is_false() {
        return Vec::new();
    }
    let mut solver = SatSolver::new(&cnf.clauses, cnf.atoms.len());
    solver
        .find_models(limit)
        .into_iter()
        .map(|model| {
            model
                .into_iter()
                .map(|(v, value)| (cnf.atoms[v - 1].clone(), value))
                .collect()
        })
        .collect()
}

/// Whether a conjunctive normal form has a model.
pub fn is_satisfiable(cnf: &Clauses) -> bool {
    !propositional_models(cnf, Some(1)).is_empty()
}
