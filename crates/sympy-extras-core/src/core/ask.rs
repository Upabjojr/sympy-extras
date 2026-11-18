//! `Q` and `ask`, ported from `sympy_extras.core.ask`: questions about
//! expressions under assumptions. The assumptions (with the global ones)
//! are read as facts about subexpressions, closed under the implications
//! of the old assumptions and checked for consistency (`Value` otherwise);
//! the proposition is answered by the old assumptions after replacing every
//! subject by a dummy carrying its facts, a relation through the
//! relational functions, the connectives by three-valued logic, the
//! handlers of a predicate first. Then the cases of disjunctive
//! assumptions, the refutation, and the last steps of SymPy's `ask`
//! (`assumptions.handlers.final_ask`), reached through a hook which that
//! module registers ([`set_final_ask`]).

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::HashMap;

use crate::core::assumptions::{closure, facts_of, is_consistent, minimal_generators, Facts, Predicate};
use crate::core::basic::{Basic, Head};
use crate::core::logic::{and_, boolean, not_, or_};
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};

/// A three-valued answer.
pub type Truth = Option<bool>;

/// The predicates of `Q` with their arities.
pub const PREDICATES_OF_Q: &[(&str, usize)] = &[
    ("algebraic", 1),
    ("antihermitian", 1),
    ("commutative", 1),
    ("complex", 1),
    ("complex_elements", 1),
    ("composite", 1),
    ("diagonal", 1),
    ("eq", 2),
    ("even", 1),
    ("extended_negative", 1),
    ("extended_nonnegative", 1),
    ("extended_nonpositive", 1),
    ("extended_nonzero", 1),
    ("extended_positive", 1),
    ("extended_real", 1),
    ("finite", 1),
    ("fullrank", 1),
    ("ge", 2),
    ("gt", 2),
    ("hermitian", 1),
    ("imaginary", 1),
    ("infinite", 1),
    ("integer", 1),
    ("integer_elements", 1),
    ("invertible", 1),
    ("irrational", 1),
    ("is_true", 1),
    ("le", 2),
    ("lower_triangular", 1),
    ("lt", 2),
    ("ne", 2),
    ("negative", 1),
    ("negative_infinite", 1),
    ("noninteger", 1),
    ("nonnegative", 1),
    ("nonpositive", 1),
    ("nonzero", 1),
    ("normal", 1),
    ("odd", 1),
    ("orthogonal", 1),
    ("positive", 1),
    ("positive_definite", 1),
    ("positive_infinite", 1),
    ("prime", 1),
    ("rational", 1),
    ("real", 1),
    ("real_elements", 1),
    ("singular", 1),
    ("square", 1),
    ("symmetric", 1),
    ("transcendental", 1),
    ("triangular", 1),
    ("unit_triangular", 1),
    ("unitary", 1),
    ("upper_triangular", 1),
    ("zero", 1),
];

/// `Q.name` (`Value` for a name which is not one of `Q`).
pub fn q(name: &str) -> PolyResult<Basic> {
    PREDICATES_OF_Q
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(n, arity)| crate::core::assume::q(n, *arity))
        .ok_or_else(|| PolyError::Value(format!("Q has no predicate {}", name)))
}

/// `Q.name(*args)`.
pub fn q_applied(name: &str, args: &[Basic]) -> PolyResult<Basic> {
    crate::core::assume::apply(&q(name)?, args)
}

type FinalAsk = fn(&Basic, &Basic) -> PolyResult<Truth>;

thread_local! {
    static FINAL_ASK: RefCell<Option<FinalAsk>> = const { RefCell::new(None) };
}

/// Register the last steps of `ask` (the handlers of the classes and the
/// SAT search of `assumptions.handlers`).
pub fn set_final_ask(f: FinalAsk) {
    FINAL_ASK.with(|h| *h.borrow_mut() = Some(f));
}

fn defined_name(p: &Basic) -> Option<&str> {
    match p.head() {
        Head::Predicate(d) if d.defined => Some(d.name.as_str()),
        _ => None,
    }
}

fn applied_parts(e: &Basic) -> Option<(&Basic, &[Basic])> {
    if matches!(e.head(), Head::AppliedPredicate) {
        Some((&e.args()[0], &e.args()[1..]))
    } else {
        None
    }
}

fn is_expr(e: &Basic) -> bool {
    !(crate::core::logic::is_boolean(e) || crate::sets::sets::is_set(e) || matches!(e.head(), Head::Tuple))
}

fn relation(name: &str, a: &Basic, b: &Basic) -> Option<PolyResult<Basic>> {
    use crate::core::relational as r;
    Some(match name {
        "eq" => r::eq(a, b),
        "ne" => r::ne(a, b),
        "gt" => r::gt(a, b),
        "ge" => r::ge(a, b),
        "lt" => r::lt(a, b),
        "le" => r::le(a, b),
        _ => return None,
    })
}

fn truth(b: &Basic) -> Truth {
    match b.head() {
        Head::BooleanTrue => Some(true),
        Head::BooleanFalse => Some(false),
        _ => None,
    }
}

fn negated(v: Truth) -> Truth {
    v.map(|x| !x)
}

fn all(values: &[Truth]) -> Truth {
    if values.contains(&Some(false)) {
        return Some(false);
    }
    if values.iter().all(|v| *v == Some(true)) {
        return Some(true);
    }
    None
}

fn any(values: &[Truth]) -> Truth {
    if values.contains(&Some(true)) {
        return Some(true);
    }
    if values.iter().all(|v| *v == Some(false)) {
        return Some(false);
    }
    None
}

fn conjuncts(e: &Basic) -> Vec<Basic> {
    if matches!(e.head(), Head::And) {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

fn inconsistent(message: String) -> PolyError {
    PolyError::Value(message)
}

/// The assumptions read as facts about subjects, and the evaluation of
/// propositions under them (`_Facts`).
pub struct FactsOf {
    assumptions: Basic,
    pub literals: Vec<Basic>,
    facts: Vec<(Basic, Facts)>,
    dummies: HashMap<Basic, Basic>,
}

impl FactsOf {
    pub fn new(assumptions: &Basic, proposition: &Basic) -> PolyResult<FactsOf> {
        let mut f: FactsOf = FactsOf {
            assumptions: assumptions.clone(),
            literals: Vec::new(),
            facts: Vec::new(),
            dummies: HashMap::new(),
        };
        for conjunct in conjuncts(assumptions) {
            f.add(&conjunct, true)?;
        }
        let facts: Vec<(Basic, Facts)> = f.facts.clone();
        for (subject, given) in &facts {
            let own: Facts = match subject.head() {
                Head::Symbol(d) | Head::Dummy(d, _) => d.assumptions.clone(),
                _ => Facts::new(),
            };
            let mut merged: Facts = own.clone();
            for (k, v) in given {
                merged.insert(*k, *v);
            }
            let merged_list: Vec<(Predicate, bool)> = merged.iter().map(|(k, v)| (*k, *v)).collect();
            if proposition.has(subject)
                && (given.iter().any(|(k, v)| own.get(k) == Some(&!*v)) || !is_consistent(&merged_list))
            {
                return Err(inconsistent(format!("inconsistent assumptions {}", f.about(subject))));
            }
            if is_expr(subject) {
                let closed: Facts = closure(&merged_list);
                let closed_list: Vec<(Predicate, bool)> = closed.iter().map(|(k, v)| (*k, *v)).collect();
                let name: String = match subject.head() {
                    Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
                    _ => "x".to_string(),
                };
                f.dummies
                    .insert(subject.clone(), Dummy::new(&name, &minimal_generators(&closed_list)).0);
            }
        }
        Ok(f)
    }

    fn add(&mut self, literal: &Basic, value: bool) -> PolyResult<()> {
        let stored: Basic = if value { literal.clone() } else { not_(literal) };
        if !self.literals.contains(&stored) {
            self.literals.push(stored);
        }
        match literal.head() {
            Head::BooleanTrue | Head::BooleanFalse => {
                if (matches!(literal.head(), Head::BooleanTrue)) != value {
                    return Err(inconsistent("Inconsistent assumptions".to_string()));
                }
                return Ok(());
            }
            Head::Not => return self.add(&literal.args()[0], !value),
            _ => {}
        }
        if let Some((p, args)) = applied_parts(literal) {
            if let Some(name) = defined_name(p) {
                if name == "is_true" {
                    if crate::core::logic::is_boolean(&args[0]) {
                        let inner: Basic = args[0].clone();
                        self.add(&inner, value)?;
                    }
                } else if let (Some(pred), 1) = (Predicate::from_name(name), args.len()) {
                    self.fact(&args[0], pred, value)?;
                } else if (name == "positive_infinite" || name == "negative_infinite") && value {
                    self.fact(&args[0], Predicate::infinite, true)?;
                    let sign: Predicate = if name == "positive_infinite" {
                        Predicate::extended_positive
                    } else {
                        Predicate::extended_negative
                    };
                    self.fact(&args[0], sign, true)?;
                }
            }
        }
        Ok(())
    }

    fn fact(&mut self, subject: &Basic, name: Predicate, value: bool) -> PolyResult<()> {
        let position: usize = match self.facts.iter().position(|(s, _)| s == subject) {
            Some(i) => i,
            None => {
                self.facts.push((subject.clone(), Facts::new()));
                self.facts.len() - 1
            }
        };
        if self.facts[position].1.get(&name) == Some(&!value) {
            return Err(inconsistent(format!(
                "inconsistent assumptions {}",
                self.about(subject)
            )));
        }
        self.facts[position].1.insert(name, value);
        Ok(())
    }

    /// The conjunction of the literals about `subject`.
    fn about(&self, subject: &Basic) -> Basic {
        let mut found: Vec<Basic> = self.literals.iter().filter(|l| l.has(subject)).cloned().collect();
        found.sort_by_cached_key(|l| l.to_string());
        and_(&found)
    }

    fn substituted(&self, e: &Basic) -> Basic {
        if self.dummies.is_empty() {
            e.clone()
        } else {
            e.xreplace(&self.dummies)
        }
    }

    pub fn evaluate(&self, prop: &Basic) -> PolyResult<Truth> {
        if self.literals.contains(prop) {
            return Ok(Some(true));
        }
        if self.literals.contains(&not_(prop)) {
            return Ok(Some(false));
        }
        let args: &[Basic] = prop.args();
        Ok(match prop.head() {
            Head::BooleanTrue => Some(true),
            Head::BooleanFalse => Some(false),
            Head::And => all(&args
                .iter()
                .map(|a| self.evaluate(a))
                .collect::<PolyResult<Vec<Truth>>>()?),
            Head::Or => any(&args
                .iter()
                .map(|a| self.evaluate(a))
                .collect::<PolyResult<Vec<Truth>>>()?),
            Head::Not => negated(self.evaluate(&args[0])?),
            Head::Implies => any(&[negated(self.evaluate(&args[0])?), self.evaluate(&args[1])?]),
            Head::Xor => {
                let values: Vec<Truth> = args.iter().map(|a| self.evaluate(a)).collect::<PolyResult<_>>()?;
                if values.iter().any(|v| v.is_none()) {
                    None
                } else {
                    Some(values.iter().filter(|v| **v == Some(true)).count() % 2 == 1)
                }
            }
            Head::Equivalent => {
                let values: Vec<Truth> = args.iter().map(|a| self.evaluate(a)).collect::<PolyResult<_>>()?;
                if values.iter().any(|v| v.is_none()) {
                    None
                } else {
                    Some(values.iter().all(|v| *v == Some(true)) || values.iter().all(|v| *v == Some(false)))
                }
            }
            Head::Ite => match self.evaluate(&args[0])? {
                None => None,
                Some(c) => self.evaluate(if c { &args[1] } else { &args[2] })?,
            },
            Head::AppliedPredicate => self.predicate(prop)?,
            Head::Relational(_) => {
                let rebuilt: Basic = crate::core::function::rebuild_evaluated(
                    prop,
                    vec![self.substituted(&args[0]), self.substituted(&args[1])],
                )?;
                truth(&rebuilt)
            }
            _ => None,
        })
    }

    fn predicate(&self, prop: &Basic) -> PolyResult<Truth> {
        let (p, args) = applied_parts(prop).expect("an applied predicate");
        if args.len() == 1 {
            if let Some(answer) = crate::core::assume::eval(p, &args[0], &self.assumptions) {
                return Ok(Some(answer));
            }
        }
        let Some(name) = defined_name(p) else { return Ok(None) };
        if name == "is_true" {
            return if crate::core::logic::is_boolean(&args[0]) {
                self.evaluate(&args[0])
            } else {
                Ok(None)
            };
        }
        if args.len() == 2 {
            if let (true, true) = (is_expr(&args[0]), is_expr(&args[1])) {
                if let Some(result) = relation(name, &self.substituted(&args[0]), &self.substituted(&args[1])) {
                    return Ok(truth(&result?));
                }
            }
            if relation(name, &args[0], &args[1]).is_some() {
                return Ok(None);
            }
        }
        let arg: &Basic = &args[0];
        if !is_expr(arg) {
            return Ok(None);
        }
        let subject: Basic = self.substituted(arg);
        let facts: &Facts = facts_of(&subject);
        if let Some(pred) = Predicate::from_name(name) {
            return Ok(facts.get(&pred).copied());
        }
        if name == "positive_infinite" || name == "negative_infinite" {
            let sign: Predicate = if name == "positive_infinite" {
                Predicate::extended_positive
            } else {
                Predicate::extended_negative
            };
            let (infinite, signed) = (facts.get(&Predicate::infinite).copied(), facts.get(&sign).copied());
            if infinite == Some(true) && signed == Some(true) {
                return Ok(Some(true));
            }
            if infinite == Some(false) || signed == Some(false) {
                return Ok(Some(false));
            }
        }
        Ok(None)
    }
}

fn relational_form(b: &Basic) -> PolyResult<Basic> {
    if matches!(b.head(), Head::Not) && crate::core::logic::is_boolean(&b.args()[0]) {
        return Ok(not_(&relational_form(&b.args()[0])?));
    }
    if let Some((p, args)) = applied_parts(b) {
        if let Some(name) = defined_name(p) {
            if args.len() == 2 && is_expr(&args[0]) && is_expr(&args[1]) {
                if let Some(result) = relation(name, &args[0], &args[1]) {
                    return result;
                }
            }
        }
    }
    Ok(b.clone())
}

/// Whether a conjunction of literals is contradictory (`_inconsistent`).
fn is_inconsistent(conjunction: &Basic) -> PolyResult<bool> {
    if matches!(conjunction.head(), Head::BooleanFalse) {
        return Ok(true);
    }
    let facts: FactsOf = match FactsOf::new(conjunction, conjunction) {
        Ok(f) => f,
        Err(PolyError::Value(_)) => return Ok(true),
        Err(e) => return Err(e),
    };
    let mut literals: Vec<Basic> = Vec::new();
    for l in &facts.literals {
        let r: Basic = relational_form(l)?;
        if !literals.contains(&r) {
            literals.push(r);
        }
    }
    for l in &literals {
        if literals.contains(&relational_form(&not_(l))?) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The largest number of cases tried.
const MAX_CASES: usize = 16;

/// The disjunctive normal form as lists of literals, `None` past `limit`
/// cases (`_dnf`).
fn dnf(b: &Basic, limit: usize, positive: bool) -> Option<Vec<Vec<Basic>>> {
    match b.head() {
        Head::BooleanTrue => return Some(if positive { vec![Vec::new()] } else { Vec::new() }),
        Head::BooleanFalse => return Some(if positive { Vec::new() } else { vec![Vec::new()] }),
        Head::Not => return dnf(&b.args()[0], limit, !positive),
        _ => {}
    }
    let parts: Vec<Basic> = b
        .args()
        .iter()
        .filter(|a| crate::core::logic::is_boolean(a))
        .cloned()
        .collect();
    let all_boolean: bool = parts.len() == b.args().len();
    match b.head() {
        Head::And | Head::Or if all_boolean => {
            let conjunctive: bool = matches!(b.head(), Head::And) == positive;
            if conjunctive {
                let mut out: Vec<Vec<Basic>> = vec![Vec::new()];
                for a in &parts {
                    let cases: Vec<Vec<Basic>> = dnf(a, limit, positive)?;
                    out = out
                        .iter()
                        .flat_map(|l| cases.iter().map(move |r| [l.clone(), r.clone()].concat()))
                        .collect();
                    if out.len() > limit {
                        return None;
                    }
                }
                Some(out)
            } else {
                let mut union: Vec<Vec<Basic>> = Vec::new();
                for a in &parts {
                    union.extend(dnf(a, limit, positive)?);
                    if union.len() > limit {
                        return None;
                    }
                }
                Some(union)
            }
        }
        Head::Implies if parts.len() == 2 => dnf(&or_(&[not_(&parts[0]), parts[1].clone()]), limit, positive),
        Head::Equivalent if parts.len() == 2 => dnf(
            &or_(&[
                and_(&[parts[0].clone(), parts[1].clone()]),
                and_(&[not_(&parts[0]), not_(&parts[1])]),
            ]),
            limit,
            positive,
        ),
        Head::Xor if parts.len() == 2 => dnf(
            &or_(&[
                and_(&[parts[0].clone(), not_(&parts[1])]),
                and_(&[not_(&parts[0]), parts[1].clone()]),
            ]),
            limit,
            positive,
        ),
        Head::Implies | Head::Equivalent | Head::Xor | Head::And | Head::Or => None,
        _ => Some(vec![vec![if positive { b.clone() } else { not_(b) }]]),
    }
}

fn ask_by_refutation(prop: &Basic, assumptions: &Basic) -> PolyResult<Truth> {
    match dnf(assumptions, MAX_CASES, true) {
        None => return Ok(None),
        Some(alone) => {
            let mut all_inconsistent: bool = true;
            for case in &alone {
                if !is_inconsistent(&and_(case))? {
                    all_inconsistent = false;
                    break;
                }
            }
            if all_inconsistent {
                return Ok(None);
            }
        }
    }
    for (value, formula) in [
        (true, and_(&[assumptions.clone(), not_(prop)])),
        (false, and_(&[assumptions.clone(), prop.clone()])),
    ] {
        if let Some(cases) = dnf(&formula, MAX_CASES, true) {
            let mut all_inconsistent: bool = true;
            for case in &cases {
                if !is_inconsistent(&and_(case))? {
                    all_inconsistent = false;
                    break;
                }
            }
            if all_inconsistent {
                return Ok(Some(value));
            }
        }
    }
    Ok(None)
}

fn ask_by_cases(prop: &Basic, assumptions: &Basic) -> PolyResult<Truth> {
    if !assumptions.preorder().iter().any(|b| {
        matches!(
            b.head(),
            Head::Or | Head::Xor | Head::Implies | Head::Equivalent | Head::Not
        )
    }) {
        return Ok(None);
    }
    let Some(cases) = dnf(assumptions, MAX_CASES, true) else {
        return Ok(None);
    };
    if cases.len() < 2 {
        return Ok(None);
    }
    let mut answers: Vec<Truth> = Vec::new();
    for case in &cases {
        let conjunction: Basic = and_(case);
        if matches!(conjunction.head(), Head::BooleanFalse) {
            continue;
        }
        match FactsOf::new(&conjunction, prop) {
            Ok(f) => answers.push(f.evaluate(prop)?),
            Err(PolyError::Value(_)) => continue,
            Err(e) => return Err(e),
        }
    }
    if let Some(first) = answers.first() {
        if first.is_some() && answers.iter().all(|a| a == first) {
            return Ok(*first);
        }
    }
    Ok(None)
}

fn has_user_predicate(b: &Basic) -> bool {
    b.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Predicate(d) if !d.defined))
}

fn final_ask(prop: &Basic, assumptions: &Basic) -> PolyResult<Truth> {
    if assumptions.is_symbol() {
        return Err(PolyError::Type(
            "assumptions must be a valid logical expression".to_string(),
        ));
    }
    match FINAL_ASK.with(|h| *h.borrow()) {
        Some(f) => f(prop, assumptions),
        None => Ok(None),
    }
}

fn argument(value: &Basic, role: &str) -> PolyResult<Basic> {
    if crate::core::logic::is_boolean(value) || (role == "assumptions" && value.is_symbol()) {
        return Ok(value.clone());
    }
    Err(PolyError::Type(format!("{} must be a valid logical expression", role)))
}

/// `True`, `False` or `None`: whether the proposition holds under the
/// assumptions and the context (the global assumptions by default)
/// (`ask`).
pub fn ask(proposition: &Basic, assumptions: &Basic, context: Option<&[Basic]>) -> PolyResult<Truth> {
    let prop: Basic = argument(proposition, "proposition")?;
    let given: Basic = argument(assumptions, "assumptions")?;
    let mut all_parts: Vec<Basic> = vec![given];
    match context {
        Some(c) => all_parts.extend(c.iter().cloned()),
        None => all_parts.extend(crate::core::assume::global_assumptions()),
    }
    let everything: Basic = and_(&all_parts);
    if matches!(everything.head(), Head::BooleanFalse) {
        return Err(PolyError::Value("Inconsistent assumptions".to_string()));
    }
    let facts: FactsOf = FactsOf::new(&everything, &prop)?;
    let result: Truth = facts.evaluate(&prop)?;
    if result.is_some() || has_user_predicate(&prop) || has_user_predicate(&everything) {
        return Ok(result);
    }
    if let Some(split) = ask_by_cases(&prop, &everything)? {
        return Ok(Some(split));
    }
    if let Some(refuted) = ask_by_refutation(&prop, &everything)? {
        return Ok(Some(refuted));
    }
    final_ask(&prop, &everything)
}

/// `ask(proposition)` with no assumptions but the global ones.
pub fn ask_simple(proposition: &Basic) -> PolyResult<Truth> {
    ask(proposition, &boolean(true), None)
}
