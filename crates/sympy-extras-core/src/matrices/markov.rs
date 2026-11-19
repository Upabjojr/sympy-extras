//! Port of `sympy_extras/matrices/markov.py`: finite Markov chains with
//! exact (possibly symbolic) transition probabilities: communicating
//! classes (Tarjan), recurrence and periods, stationary distributions,
//! the fundamental matrix, absorption and hitting probabilities, mean
//! hitting and return times, the limiting matrix.

#![allow(clippy::needless_range_loop)]

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::linalg::{inverse, linear_solve, normal, nullspace, square};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn sum(v: &[Basic]) -> R {
    crate::core::add::add(v)
}

/// A communicating class (`CommunicatingClass`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommunicatingClass {
    pub states: Vec<usize>,
    pub recurrent: bool,
    pub period: u64,
}

impl CommunicatingClass {
    pub fn repr(&self) -> String {
        let states: Vec<String> = self.states.iter().map(|s| s.to_string()).collect();
        format!(
            "CommunicatingClass([{}], {}, period {})",
            states.join(", "),
            if self.recurrent { "recurrent" } else { "transient" },
            self.period
        )
    }
}

fn components(successors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n: usize = successors.len();
    let mut index: Vec<i64> = vec![-1; n];
    let mut low: Vec<i64> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut found: Vec<Vec<usize>> = Vec::new();
    let mut counter: i64 = 0;
    for root in 0..n {
        if index[root] != -1 {
            continue;
        }
        let mut work: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = counter;
        low[root] = counter;
        counter += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&(node, position)) = work.last() {
            if position < successors[node].len() {
                let last: usize = work.len() - 1;
                work[last] = (node, position + 1);
                let target: usize = successors[node][position];
                if index[target] == -1 {
                    index[target] = counter;
                    low[target] = counter;
                    counter += 1;
                    stack.push(target);
                    on_stack[target] = true;
                    work.push((target, 0));
                } else if on_stack[target] {
                    low[node] = low[node].min(index[target]);
                }
            } else {
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    low[parent] = low[parent].min(low[node]);
                }
                if low[node] == index[node] {
                    let mut component: Vec<usize> = Vec::new();
                    loop {
                        let member: usize = stack.pop().expect("a member");
                        on_stack[member] = false;
                        component.push(member);
                        if member == node {
                            break;
                        }
                    }
                    component.sort();
                    found.push(component);
                }
            }
        }
    }
    found
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn period(states: &[usize], successors: &[Vec<usize>]) -> u64 {
    let mut level: Vec<(usize, i64)> = vec![(states[0], 0)];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::from([states[0]]);
    let mut result: u64 = 0;
    let level_of = |level: &[(usize, i64)], s: usize| level.iter().find(|(k, _)| *k == s).map(|(_, v)| *v);
    while let Some(node) = queue.pop_front() {
        for &target in &successors[node] {
            if !states.contains(&target) {
                continue;
            }
            let ln: i64 = level_of(&level, node).expect("a level");
            match level_of(&level, target) {
                None => {
                    level.push((target, ln + 1));
                    queue.push_back(target);
                }
                Some(lt) => result = gcd(result, (ln + 1 - lt).unsigned_abs()),
            }
        }
    }
    result
}

/// A finite Markov chain (`MarkovChain`).
#[derive(Debug, Clone)]
pub struct MarkovChain {
    pub p: ExprMatrix,
    pub size: usize,
    successors: Vec<Vec<usize>>,
    classes: Vec<CommunicatingClass>,
}

impl MarkovChain {
    pub fn new(p: &ExprMatrix) -> PolyResult<MarkovChain> {
        let n: usize = square(p)?;
        let matrix: ExprMatrix = p.applyfunc(&normal)?;
        for i in 0..n {
            let total: Basic = sum(matrix.row(i))?;
            if normal(&crate::core::add::sub(&total, &int(1))?)? != int(0) {
                return Err(value_error(format!(
                    "the row {} of the transition matrix has the sum {}, not 1",
                    i,
                    normal(&total)?
                )));
            }
        }
        let successors: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|j| *matrix.at(i, *j) != int(0)).collect())
            .collect();
        let mut found: Vec<CommunicatingClass> = Vec::new();
        for component in components(&successors) {
            let closed: bool = component
                .iter()
                .all(|s| successors[*s].iter().all(|t| component.contains(t)));
            let p_: u64 = period(&component, &successors);
            found.push(CommunicatingClass {
                states: component,
                recurrent: closed,
                period: p_,
            });
        }
        found.sort_by(|a, b| (!a.recurrent, &a.states).cmp(&(!b.recurrent, &b.states)));
        Ok(MarkovChain {
            p: matrix,
            size: n,
            successors,
            classes: found,
        })
    }

    pub fn classes(&self) -> Vec<CommunicatingClass> {
        self.classes.clone()
    }

    pub fn recurrent_classes(&self) -> Vec<CommunicatingClass> {
        self.classes.iter().filter(|c| c.recurrent).cloned().collect()
    }

    pub fn transient_states(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .classes
            .iter()
            .filter(|c| !c.recurrent)
            .flat_map(|c| c.states.iter().cloned())
            .collect();
        v.sort();
        v
    }

    pub fn recurrent_states(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .classes
            .iter()
            .filter(|c| c.recurrent)
            .flat_map(|c| c.states.iter().cloned())
            .collect();
        v.sort();
        v
    }

    pub fn is_irreducible(&self) -> bool {
        self.classes.len() == 1
    }

    pub fn is_absorbing(&self) -> bool {
        self.classes.iter().filter(|c| c.recurrent).all(|c| c.states.len() == 1)
    }

    pub fn is_aperiodic(&self) -> bool {
        self.classes.iter().filter(|c| c.recurrent).all(|c| c.period == 1)
    }

    pub fn is_ergodic(&self) -> bool {
        self.is_irreducible() && self.is_aperiodic()
    }

    pub fn period(&self) -> PolyResult<u64> {
        if !self.is_irreducible() {
            return Err(value_error("the chain is not irreducible: see classes()".to_string()));
        }
        Ok(self.classes[0].period)
    }

    fn class_distribution(&self, recurrent: &CommunicatingClass) -> PolyResult<Vec<Basic>> {
        let states: &[usize] = &recurrent.states;
        let k: usize = states.len();
        let mut entries: Vec<Basic> = Vec::new();
        for i in states {
            for j in states {
                entries.push(self.p.at(*i, *j).clone());
            }
        }
        let block: ExprMatrix = ExprMatrix::new(k, k, entries)?;
        let kernel: Vec<ExprMatrix> = nullspace(&block.transpose().sub(&ExprMatrix::eye(k))?.applyfunc(&normal)?)?;
        if kernel.len() != 1 {
            let shown: Vec<String> = states.iter().map(|s| s.to_string()).collect();
            return Err(PolyError::Arithmetic(format!(
                "the recurrent class [{}] has {} invariant vectors",
                shown.join(", "),
                kernel.len()
            )));
        }
        let vector: Vec<Basic> = (0..k).map(|i| kernel[0].at(i, 0).clone()).collect();
        let total: Basic = sum(&vector)?;
        let mut distribution: Vec<Basic> = vec![int(0); self.size];
        for (state, value) in states.iter().zip(vector.iter()) {
            distribution[*state] = normal(&crate::core::mul::div(value, &total)?)?;
        }
        Ok(distribution)
    }

    pub fn stationary_distributions(&self) -> PolyResult<Vec<ExprMatrix>> {
        self.recurrent_classes()
            .iter()
            .map(|c| ExprMatrix::new(1, self.size, self.class_distribution(c)?))
            .collect()
    }

    pub fn stationary_distribution(&self) -> PolyResult<ExprMatrix> {
        let mut distributions: Vec<ExprMatrix> = self.stationary_distributions()?;
        if distributions.len() != 1 {
            return Err(value_error(format!(
                "the chain has {} recurrent classes: see stationary_distributions()",
                distributions.len()
            )));
        }
        Ok(distributions.remove(0))
    }

    pub fn fundamental_matrix(&self) -> PolyResult<ExprMatrix> {
        let transient: Vec<usize> = self.transient_states();
        let k: usize = transient.len();
        if k == 0 {
            return Ok(ExprMatrix::zeros(0, 0));
        }
        let mut entries: Vec<Basic> = Vec::new();
        for i in &transient {
            for j in &transient {
                entries.push(self.p.at(*i, *j).clone());
            }
        }
        let q: ExprMatrix = ExprMatrix::new(k, k, entries)?;
        inverse(&ExprMatrix::eye(k).sub(&q)?.applyfunc(&normal)?)
    }

    pub fn expected_steps(&self) -> PolyResult<ExprMatrix> {
        let n: ExprMatrix = self.fundamental_matrix()?;
        Ok(ExprMatrix::column(
            (0..n.rows)
                .map(|i| normal(&sum(n.row(i))?))
                .collect::<PolyResult<_>>()?,
        ))
    }

    pub fn absorption_probabilities(&self) -> PolyResult<ExprMatrix> {
        let transient: Vec<usize> = self.transient_states();
        let recurrent: Vec<CommunicatingClass> = self.recurrent_classes();
        let n: ExprMatrix = self.fundamental_matrix()?;
        let mut entries: Vec<Basic> = Vec::new();
        for row in 0..transient.len() {
            for target in &recurrent {
                let mut total: Vec<Basic> = Vec::new();
                for (position, state) in transient.iter().enumerate() {
                    let step: Basic = sum(&target
                        .states
                        .iter()
                        .map(|j| self.p.at(*state, *j).clone())
                        .collect::<Vec<Basic>>())?;
                    if step != int(0) && *n.at(row, position) != int(0) {
                        total.push(crate::core::mul::mul(&[n.at(row, position).clone(), step])?);
                    }
                }
                entries.push(if total.is_empty() {
                    int(0)
                } else {
                    normal(&sum(&total)?)?
                });
            }
        }
        ExprMatrix::new(transient.len(), recurrent.len(), entries)
    }

    fn can_reach(&self, targets: &[usize]) -> Vec<usize> {
        let mut reached: Vec<usize> = targets.to_vec();
        let mut changed: bool = true;
        while changed {
            changed = false;
            for state in 0..self.size {
                if !reached.contains(&state) && self.successors[state].iter().any(|t| reached.contains(t)) {
                    reached.push(state);
                    changed = true;
                }
            }
        }
        reached
    }

    fn targets(&self, targets: &[i64]) -> PolyResult<Vec<usize>> {
        let mut chosen: Vec<i64> = targets.to_vec();
        chosen.sort();
        chosen.dedup();
        if chosen.is_empty() || chosen[0] < 0 || chosen[chosen.len() - 1] >= self.size as i64 {
            return Err(value_error(format!(
                "the target states are among 0, ..., {}",
                self.size as i64 - 1
            )));
        }
        Ok(chosen.into_iter().map(|s| s as usize).collect())
    }

    fn unit_minus(&self, states: &[usize]) -> PolyResult<ExprMatrix> {
        let mut entries: Vec<Basic> = Vec::new();
        for i in states {
            for j in states {
                entries.push(crate::core::add::sub(
                    &int(if i == j { 1 } else { 0 }),
                    self.p.at(*i, *j),
                )?);
            }
        }
        ExprMatrix::new(states.len(), states.len(), entries)
    }

    pub fn hitting_probabilities(&self, targets: &[i64]) -> PolyResult<ExprMatrix> {
        let chosen: Vec<usize> = self.targets(targets)?;
        let able: Vec<usize> = self.can_reach(&chosen);
        let unknown: Vec<usize> = (0..self.size)
            .filter(|s| able.contains(s) && !chosen.contains(s))
            .collect();
        let mut result: Vec<Basic> = (0..self.size)
            .map(|s| int(if chosen.contains(&s) { 1 } else { 0 }))
            .collect();
        if !unknown.is_empty() {
            let a: ExprMatrix = self.unit_minus(&unknown)?;
            let rhs: Vec<Basic> = unknown
                .iter()
                .map(|i| sum(&chosen.iter().map(|j| self.p.at(*i, *j).clone()).collect::<Vec<Basic>>()))
                .collect::<PolyResult<_>>()?;
            let solution = linear_solve(&a.applyfunc(&normal)?, &ExprMatrix::column(rhs).applyfunc(&normal)?)?;
            let solution = match solution {
                Some(s) if s.is_unique() => s,
                _ => {
                    return Err(PolyError::Arithmetic(
                        "the hitting probabilities are not determined".to_string(),
                    ))
                }
            };
            for (position, state) in unknown.iter().enumerate() {
                result[*state] = solution.particular.at(position, 0).clone();
            }
        }
        Ok(ExprMatrix::column(result))
    }

    pub fn mean_hitting_times(&self, targets: &[i64]) -> PolyResult<ExprMatrix> {
        let chosen: Vec<usize> = self.targets(targets)?;
        let chosen_i: Vec<i64> = chosen.iter().map(|s| *s as i64).collect();
        let probabilities: ExprMatrix = self.hitting_probabilities(&chosen_i)?;
        let mut certain: Vec<usize> = Vec::new();
        for state in 0..self.size {
            if !chosen.contains(&state)
                && normal(&crate::core::add::sub(probabilities.at(state, 0), &int(1))?)? == int(0)
            {
                certain.push(state);
            }
        }
        let mut result: Vec<Basic> = (0..self.size)
            .map(|s| if chosen.contains(&s) { int(0) } else { Basic::oo() })
            .collect();
        if !certain.is_empty() {
            let a: ExprMatrix = self.unit_minus(&certain)?;
            let solution = linear_solve(&a.applyfunc(&normal)?, &ExprMatrix::column(vec![int(1); certain.len()]))?;
            let solution = match solution {
                Some(s) if s.is_unique() => s,
                _ => {
                    return Err(PolyError::Arithmetic(
                        "the mean hitting times are not determined".to_string(),
                    ))
                }
            };
            for (position, state) in certain.iter().enumerate() {
                result[*state] = solution.particular.at(position, 0).clone();
            }
        }
        Ok(ExprMatrix::column(result))
    }

    pub fn mean_return_time(&self, state: i64) -> R {
        for recurrent in self.recurrent_classes() {
            if state >= 0 && recurrent.states.contains(&(state as usize)) {
                let d: Vec<Basic> = self.class_distribution(&recurrent)?;
                return normal(&crate::core::mul::div(&int(1), &d[state as usize])?);
            }
        }
        if !(0 <= state && state < self.size as i64) {
            return Err(value_error(format!("no state {}", state)));
        }
        Ok(Basic::oo())
    }

    pub fn limiting_matrix(&self) -> PolyResult<ExprMatrix> {
        let recurrent: Vec<CommunicatingClass> = self.recurrent_classes();
        let distributions: Vec<Vec<Basic>> = recurrent
            .iter()
            .map(|c| self.class_distribution(c))
            .collect::<PolyResult<_>>()?;
        let transient: Vec<usize> = self.transient_states();
        let absorption: ExprMatrix = self.absorption_probabilities()?;
        let mut rows: Vec<Vec<Basic>> = vec![vec![int(0); self.size]; self.size];
        for (distribution, group) in distributions.iter().zip(recurrent.iter()) {
            for state in &group.states {
                rows[*state] = distribution.clone();
            }
        }
        for (position, state) in transient.iter().enumerate() {
            let mut row: Vec<Basic> = Vec::new();
            for j in 0..self.size {
                let terms: Vec<Basic> = (0..recurrent.len())
                    .map(|c| crate::core::mul::mul(&[absorption.at(position, c).clone(), distributions[c][j].clone()]))
                    .collect::<PolyResult<_>>()?;
                row.push(normal(&sum(&terms)?)?);
            }
            rows[*state] = row;
        }
        ExprMatrix::new(self.size, self.size, rows.into_iter().flatten().collect())
    }

    pub fn distribution_after(&self, initial: &[Basic], steps: i64) -> PolyResult<ExprMatrix> {
        if initial.len() != self.size || steps < 0 {
            return Err(value_error(format!(
                "an initial distribution on {} states and a number of steps are needed",
                self.size
            )));
        }
        let mut current: ExprMatrix = ExprMatrix::new(1, self.size, initial.to_vec())?;
        for _ in 0..steps {
            current = current.matmul(&self.p)?.applyfunc(&normal)?;
        }
        Ok(current)
    }
}
