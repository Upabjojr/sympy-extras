//! The engine of the step-by-step solutions, ported from
//! `sympy_extras.steps.core`: rules, steps as values, and the solver which
//! drives them.
//!
//! * A **problem** is a small immutable, hashable value.
//! * A **step** records one operation (a type implementing [`Step`]): the
//!   rule is its identifier, its fields what the rule needs and the steps
//!   of its subproblems, so that a derivation is a *tree of steps*.
//! * A **rule** is a function of the problem; where Python's rule (a
//!   generator) yields a subproblem and receives its solved step, the Rust
//!   rule calls [`StepSolver::solve`] on the solver it is given. It
//!   returns its step, or `None` when it does not apply.
//! * The [`StepSolver`] owns the state of one run: the subproblems on the
//!   current path (a rule leading back to a problem being solved gets the
//!   `in_progress` step), the steps already found, the limit on the depth.
//!
//! With `branch` every candidate which works is kept in an alternative
//! step; otherwise the first rule which solves the problem completely.
//! [`format_steps`] prints a tree, the `minor` steps skipped unless asked
//! for.

// The design is that of sympy/integrals/manualintegrate.py (the IntegrationSolver and its
// generator rules) in SymPy's development branch, made generic over the kind of problem.

use std::collections::HashMap;
use std::hash::Hash;

/// One operation of a derivation.
pub trait Step: Clone + PartialEq {
    /// The steps of the subproblems, in order.
    fn substeps(&self) -> Vec<Self>;

    /// The identifier of the rule (the name of the Python class).
    fn rule(&self) -> String;

    /// What the step establishes, as a formula without words.
    fn statement(&self) -> String;

    /// A step of little interest (arithmetic, a rewriting), hidden by
    /// default when the tree is printed.
    fn minor(&self) -> bool {
        false
    }

    /// Whether the step holds several derivations of the same problem
    /// (`AlternativeStep`): its substeps are the alternatives.
    fn is_alternative(&self) -> bool {
        false
    }

    /// Whether the step itself stands for work left undone (`DontKnow`
    /// and the step of a problem in progress).
    fn is_dont_know(&self) -> bool {
        false
    }

    /// Whether the step itself is right but partial (its own part of
    /// `complete`).
    fn is_partial(&self) -> bool {
        false
    }

    /// Whether some part of the derivation was left undone.
    fn contains_dont_know(&self) -> bool {
        self.is_dont_know() || self.substeps().iter().any(Step::contains_dont_know)
    }

    /// Whether the result is the whole answer: a partial step answers
    /// `false`, and so does every step built on it.
    fn complete(&self) -> bool {
        !self.is_partial() && self.substeps().iter().all(Step::complete)
    }
}

/// What a rule may know about the run: whether alternatives are
/// collected, and the depth of the problem in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context {
    pub branch: bool,
    pub depth: usize,
}

/// A rule: the problem, the context and the solver of its subproblems to
/// its step, `None` when it does not apply.
pub type Rule<P, S> = Box<dyn Fn(&P, &Context, &mut StepSolver<P, S>) -> Option<S>>;

/// The candidate rules, from plain functions or closures (`rules![a, b]`).
#[macro_export]
macro_rules! rules {
    ($($r:expr),* $(,)?) => {
        vec![$(Box::new($r) as $crate::steps::core::Rule<_, _>),*]
    };
}

/// Solves problems of type `P` into steps of type `S` (`StepSolver`).
pub struct StepSolver<P, S> {
    /// the rules to try on a problem, in order of preference
    pub candidates: fn(&P) -> Vec<Rule<P, S>>,
    /// the step of a problem no rule solves (or beyond `max_depth`)
    pub dont_know: fn(&P) -> S,
    /// the step of a problem met again while it is being solved: it must
    /// answer `contains_dont_know()` with `true`
    pub in_progress: fn(&P) -> S,
    /// the step holding several derivations of a problem
    pub alternative: fn(&P, Vec<S>) -> S,
    /// collect every workable rule (`true`) or keep the first
    pub branch: bool,
    /// the limit on the depth of nested subproblems (`None`: no limit)
    pub max_depth: Option<usize>,
    active: Vec<P>,
    solved: HashMap<P, S>,
    starved: bool,
}

impl<P: Clone + Eq + Hash, S: Step> StepSolver<P, S> {
    pub fn new(
        candidates: fn(&P) -> Vec<Rule<P, S>>,
        dont_know: fn(&P) -> S,
        in_progress: fn(&P) -> S,
        alternative: fn(&P, Vec<S>) -> S,
        branch: bool,
        max_depth: Option<usize>,
    ) -> StepSolver<P, S> {
        StepSolver {
            candidates,
            dont_know,
            in_progress,
            alternative,
            branch,
            max_depth,
            active: Vec::new(),
            solved: HashMap::new(),
            starved: false,
        }
    }

    /// The step tree of a problem (possibly with parts left undone: see
    /// [`Step::contains_dont_know`]).
    pub fn solve(&mut self, problem: &P) -> S {
        if let Some(known) = self.solved.get(problem) {
            return known.clone();
        }
        if self.max_depth.is_some_and(|m| self.active.len() >= m) {
            self.starved = true;
            return (self.dont_know)(problem);
        }
        if self.active.contains(problem) {
            // the attempt leads around in a loop: the problem is still being worked on
            return (self.in_progress)(problem);
        }
        let outer: bool = if self.active.is_empty() { false } else { self.starved };
        self.starved = false;
        self.active.push(problem.clone());
        let result: S = self.apply(problem);
        if !self.starved {
            // a result found without running out of depth does not depend on where it was asked
            self.solved.insert(problem.clone(), result.clone());
        }
        self.active.pop();
        self.starved = outer || self.starved;
        result
    }

    fn apply(&mut self, problem: &P) -> S {
        let context = Context {
            branch: self.branch,
            depth: self.active.len() - 1,
        };
        let mut complete: Vec<S> = Vec::new();
        let mut some: Option<S> = None; // solved, but not the whole answer
        let mut partial: Option<S> = None;
        for rule in (self.candidates)(problem) {
            let Some(result) = rule(problem, &context, self) else {
                continue;
            };
            if result.contains_dont_know() {
                if partial.is_none() {
                    partial = Some(result);
                }
                continue;
            }
            if !result.complete() {
                // kept only when no rule gives the whole answer
                if some.is_none() {
                    some = Some(result);
                }
                continue;
            }
            if !self.branch {
                return result;
            }
            if !complete.contains(&result) {
                complete.push(result);
            }
        }
        if complete.len() == 1 {
            return complete.pop().expect("one step");
        }
        if !complete.is_empty() {
            return (self.alternative)(problem, complete);
        }
        if let Some(s) = some {
            return s;
        }
        partial.unwrap_or_else(|| (self.dont_know)(problem))
    }
}

/// The number of steps of a tree, the alternatives all counted
/// (`count_steps`).
pub fn count_steps<S: Step>(step: &S) -> usize {
    1 + step.substeps().iter().map(count_steps).sum::<usize>()
}

/// The tree as text (`format_steps`): one line `Rule: statement` for each
/// step, the substeps indented below it, down to `depth` levels (all for
/// `None`); the `minor` steps are left out, their substeps taking their
/// place, unless `minor`; the alternatives are numbered.
pub fn format_steps<S: Step>(step: &S, depth: Option<usize>, minor: bool, indent: &str) -> String {
    fn walk<S: Step>(
        current: &S,
        level: usize,
        prefix: &str,
        depth: Option<usize>,
        minor: bool,
        indent: &str,
        lines: &mut Vec<String>,
    ) {
        if depth.is_some_and(|d| level > d) {
            return;
        }
        let children: Vec<S> = current.substeps();
        let shown: bool = minor || !current.minor() || children.is_empty();
        if shown {
            lines.push(format!(
                "{}{}{}: {}",
                indent.repeat(level),
                prefix,
                current.rule(),
                current.statement()
            ));
        }
        let below: usize = if shown { level + 1 } else { level };
        if current.is_alternative() {
            for (number, alternative) in children.iter().enumerate() {
                walk(
                    alternative,
                    below,
                    &format!("[{}] ", number + 1),
                    depth,
                    minor,
                    indent,
                    lines,
                );
            }
        } else {
            for child in &children {
                walk(
                    child,
                    below,
                    if shown { "" } else { prefix },
                    depth,
                    minor,
                    indent,
                    lines,
                );
            }
        }
    }
    let mut lines: Vec<String> = Vec::new();
    walk(step, 0, "", depth, minor, indent, &mut lines);
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy domain: the Fibonacci numbers by their recurrence, with a
    /// rule leading in a loop (n from n + 1 - 1) and a minor step.
    #[derive(Debug, Clone, PartialEq)]
    enum Fib {
        Base(u64),
        Sum(u64, Vec<Fib>),
        Shift(u64, Box<Fib>),
        Alternatives(u64, Vec<Fib>),
        DontKnow(u64),
        InProgress(u64),
    }

    impl Step for Fib {
        fn substeps(&self) -> Vec<Fib> {
            match self {
                Fib::Sum(_, s) | Fib::Alternatives(_, s) => s.clone(),
                Fib::Shift(_, s) => vec![(**s).clone()],
                _ => Vec::new(),
            }
        }
        fn rule(&self) -> String {
            match self {
                Fib::Base(_) => "Base",
                Fib::Sum(..) => "Sum",
                Fib::Shift(..) => "Shift",
                Fib::Alternatives(..) => "Alternatives",
                Fib::DontKnow(_) => "DontKnow",
                Fib::InProgress(_) => "InProgress",
            }
            .to_string()
        }
        fn statement(&self) -> String {
            let n = match self {
                Fib::Base(n) | Fib::Sum(n, _) | Fib::Shift(n, _) | Fib::Alternatives(n, _) => n,
                Fib::DontKnow(n) | Fib::InProgress(n) => n,
            };
            format!("F({})", n)
        }
        fn minor(&self) -> bool {
            matches!(self, Fib::Shift(..))
        }
        fn is_alternative(&self) -> bool {
            matches!(self, Fib::Alternatives(..))
        }
        fn is_dont_know(&self) -> bool {
            matches!(self, Fib::DontKnow(_) | Fib::InProgress(_))
        }
    }

    fn base(n: &u64, _: &Context, _: &mut StepSolver<u64, Fib>) -> Option<Fib> {
        (*n < 2).then_some(Fib::Base(*n))
    }

    fn sum(n: &u64, _: &Context, s: &mut StepSolver<u64, Fib>) -> Option<Fib> {
        if *n < 2 {
            return None;
        }
        let a = s.solve(&(n - 1));
        let b = s.solve(&(n - 2));
        Some(Fib::Sum(*n, vec![a, b]))
    }

    fn shift(n: &u64, _: &Context, s: &mut StepSolver<u64, Fib>) -> Option<Fib> {
        // a loop: the problem asks for itself
        Some(Fib::Shift(*n, Box::new(s.solve(n))))
    }

    fn candidates(_: &u64) -> Vec<Rule<u64, Fib>> {
        crate::rules![shift, base, sum]
    }

    fn solver(branch: bool, max_depth: Option<usize>) -> StepSolver<u64, Fib> {
        StepSolver::new(
            candidates,
            |n| Fib::DontKnow(*n),
            |n| Fib::InProgress(*n),
            |n, s| Fib::Alternatives(*n, s),
            branch,
            max_depth,
        )
    }

    #[test]
    fn loops_are_cut_and_results_shared() {
        let mut s = solver(false, None);
        let step = s.solve(&4);
        assert!(!step.contains_dont_know());
        assert_eq!(step.rule(), "Sum");
        // F(4) = F(3) + F(2), F(3) = F(2) + F(1): F(2) solved once and shared
        assert_eq!(count_steps(&step), 9);
        assert_eq!(
            format_steps(&step, Some(1), false, "  "),
            "Sum: F(4)\n  Sum: F(3)\n  Sum: F(2)"
        );
    }

    #[test]
    fn depth_limit_leaves_undone() {
        let mut s = solver(false, Some(2));
        let step = s.solve(&5);
        assert!(step.contains_dont_know());
    }

    #[test]
    fn minor_steps_hidden() {
        let shifted = Fib::Shift(3, Box::new(Fib::Base(1)));
        assert_eq!(format_steps(&shifted, None, false, "  "), "Base: F(1)");
        assert_eq!(format_steps(&shifted, None, true, "  "), "Shift: F(3)\n  Base: F(1)");
        let alternatives = Fib::Alternatives(1, vec![Fib::Base(1), shifted]);
        assert_eq!(
            format_steps(&alternatives, None, false, "-"),
            "Alternatives: F(1)\n-[1] Base: F(1)\n-[2] Base: F(1)"
        );
    }
}
