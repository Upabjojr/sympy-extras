//! Assumptions written as ordinary mathematical statements, ported from
//! `sympy_extras.assumptions.facts`: relations, memberships in sets
//! (`element`, `Contains`) and predicates of `Q`, normalized
//! ([`normalize`]: memberships of intervals, finite sets and combinations
//! of sets as relations, relations canonical with their symbolic
//! denominators cleared), translated to predicates of `Q`
//! ([`to_predicates`]) and to polynomial relations between real variables
//! for the CAD ([`to_polynomial`]), collected by [`Facts`]. As in
//! Mathematica, an inequality says that both sides are real.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use crate::core::ask::{ask, q_applied};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::expr::is_true;
use crate::core::logic::{and_, boolean, not_, or_};
use crate::core::relational::{eq, ge, gt, le, lt, ne};
use crate::error::{PolyError, PolyResult};

fn q1(name: &str, e: &Basic) -> PolyResult<Basic> {
    q_applied(name, std::slice::from_ref(e))
}

fn contains_node(x: &Basic, domain: &Basic) -> Basic {
    Basic::new(Head::Contains, vec![x.clone(), domain.clone()])
}

fn is_truth(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue | Head::BooleanFalse)
}

/// Membership of `x` in `domain`, evaluated when known (`element`).
pub fn element(x: &Basic, domain: &Basic) -> PolyResult<Basic> {
    if !crate::sets::sets::is_set(domain) {
        return Err(PolyError::Type(format!("expected a set, got {}", domain)));
    }
    let result: Basic = crate::sets::sets::contains(domain, x)?;
    if is_truth(&result) {
        return Ok(result);
    }
    Ok(contains_node(x, domain))
}

fn is_named_set(s: &Basic) -> bool {
    matches!(
        s.head(),
        Head::Naturals | Head::Naturals0 | Head::Integers | Head::Rationals | Head::Reals | Head::Complexes
    )
}

pub(crate) fn is_compound(f: &Basic) -> bool {
    matches!(
        f.head(),
        Head::And | Head::Or | Head::Not | Head::Implies | Head::Equivalent | Head::Xor | Head::Ite
    )
}

fn rebuilt(formula: &Basic, args: Vec<Basic>) -> PolyResult<Basic> {
    crate::core::function::rebuild_evaluated(formula, args)
}

fn rel_op(e: &Basic) -> Option<RelOp> {
    match e.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

fn sorted_symbols(symbols: impl IntoIterator<Item = Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().collect();
    v.sort_by_key(symbol_name);
    v.dedup();
    v
}

/// The relation with a symbolic polynomial denominator cleared, with
/// `den != 0` alongside (`_cleared_denominator`).
fn cleared_denominator(atom: &Basic) -> PolyResult<Option<Basic>> {
    let op: RelOp = rel_op(atom).expect("a relation");
    let difference: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
    let (numerator, denominator) =
        crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&difference, false, true)?)?;
    if denominator.free_symbols().is_empty() {
        return Ok(None);
    }
    let symbols: BTreeSet<Basic> = denominator
        .free_symbols()
        .into_iter()
        .chain(numerator.free_symbols())
        .collect();
    if !(super::bounds::is_polynomial_in(&denominator, &symbols)
        && super::bounds::is_polynomial_in(&numerator, &symbols))
    {
        return Ok(None);
    }
    let zero: Basic = Basic::zero();
    let nonzero: Basic = ne(&denominator, &zero)?;
    let product: Basic = crate::core::mul::mul(&[numerator.clone(), denominator.clone()])?;
    Ok(Some(match op {
        RelOp::Eq => and_(&[eq(&numerator, &zero)?, nonzero]),
        RelOp::Ne => and_(&[ne(&numerator, &zero)?, nonzero]),
        RelOp::Gt => gt(&product, &zero)?,
        RelOp::Lt => lt(&product, &zero)?,
        RelOp::Ge => and_(&[ge(&product, &zero)?, nonzero]),
        RelOp::Le => and_(&[le(&product, &zero)?, nonzero]),
    }))
}

/// The normal form of an assumption or query (`normalize`).
pub fn normalize(f: &Basic) -> PolyResult<Basic> {
    if is_truth(f) {
        return Ok(f.clone());
    }
    if matches!(f.head(), Head::Contains) {
        let (member, domain) = (&f.args()[0], &f.args()[1]);
        if !crate::sets::sets::is_set(domain) {
            return Err(PolyError::Type("Contains takes an expression and a set".to_string()));
        }
        return normalize_contains(member, domain);
    }
    if matches!(f.head(), Head::Ite) {
        let (c, a, b) = (
            normalize(&f.args()[0])?,
            normalize(&f.args()[1])?,
            normalize(&f.args()[2])?,
        );
        return Ok(or_(&[and_(&[c.clone(), a]), and_(&[not_(&c), b])]));
    }
    if is_compound(f) {
        let args: Vec<Basic> = f.args().iter().map(normalize).collect::<PolyResult<_>>()?;
        return rebuilt(f, args);
    }
    if rel_op(f).is_some() {
        if let Some(cleared) = cleared_denominator(f)? {
            return normalize(&cleared);
        }
        let canonical: Basic = crate::core::relational::canonical(f)?;
        if let Some(op) = rel_op(&canonical) {
            let rhs: &Basic = &canonical.args()[1];
            if matches!(rhs.head(), Head::Infinity) && matches!(op, RelOp::Lt | RelOp::Le) {
                return Ok(boolean(true));
            }
            if matches!(rhs.head(), Head::NegativeInfinity) && matches!(op, RelOp::Gt | RelOp::Ge) {
                return Ok(boolean(true));
            }
        }
        return Ok(canonical);
    }
    Ok(f.clone())
}

fn normalize_contains(x: &Basic, domain: &Basic) -> PolyResult<Basic> {
    if is_named_set(domain) {
        return Ok(contains_node(x, domain));
    }
    match domain.head() {
        Head::Interval => normalize(&crate::sets::sets::as_relational(domain, x)?),
        Head::FiniteSet => Ok(or_(&domain
            .args()
            .iter()
            .map(|v| eq(x, v))
            .collect::<PolyResult<Vec<Basic>>>()?)),
        Head::Union => Ok(or_(&domain
            .args()
            .iter()
            .map(|s| normalize_contains(x, s))
            .collect::<PolyResult<Vec<Basic>>>()?)),
        Head::Intersection => Ok(and_(
            &domain
                .args()
                .iter()
                .map(|s| normalize_contains(x, s))
                .collect::<PolyResult<Vec<Basic>>>()?,
        )),
        Head::Complement => Ok(and_(&[
            normalize_contains(x, &domain.args()[0])?,
            not_(&normalize_contains(x, &domain.args()[1])?),
        ])),
        Head::EmptySet => Ok(boolean(false)),
        Head::UniversalSet => Ok(boolean(true)),
        _ => {
            let result: Basic = crate::sets::sets::contains(domain, x)?;
            if is_truth(&result) {
                return Ok(result);
            }
            Ok(contains_node(x, domain))
        }
    }
}

/// The top level conjuncts of a normalized formula (`conjuncts`).
pub fn conjuncts(formula: &Basic) -> Vec<Basic> {
    if matches!(formula.head(), Head::And) {
        return formula.args().to_vec();
    }
    if matches!(formula.head(), Head::BooleanTrue) {
        return Vec::new();
    }
    vec![formula.clone()]
}

fn set_predicate(x: &Basic, domain: &Basic) -> PolyResult<Option<Basic>> {
    Ok(Some(match domain.head() {
        Head::Naturals0 => and_(&[q1("integer", x)?, q1("nonnegative", x)?]),
        Head::Naturals => and_(&[q1("integer", x)?, q1("positive", x)?]),
        Head::Integers => q1("integer", x)?,
        Head::Rationals => q1("rational", x)?,
        Head::Reals => q1("real", x)?,
        Head::Complexes => q1("complex", x)?,
        _ => return Ok(None),
    }))
}

fn is_expr(e: &Basic) -> bool {
    !(crate::core::logic::is_boolean(e) || crate::sets::sets::is_set(e) || matches!(e.head(), Head::Tuple))
}

/// The predicate of `Q` equivalent to an atom (`_predicate_of_atom`).
pub(crate) fn predicate_of_atom(atom: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(atom.head(), Head::AppliedPredicate) {
        return Ok(Some(atom.clone()));
    }
    if matches!(atom.head(), Head::Contains) {
        let (x, domain) = (&atom.args()[0], &atom.args()[1]);
        if !is_expr(x) {
            return Ok(None);
        }
        return set_predicate(x, domain);
    }
    if let Some(op) = rel_op(atom) {
        let d: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
        return Ok(Some(match op {
            RelOp::Gt => q1("positive", &d)?,
            RelOp::Lt => q1("negative", &d)?,
            RelOp::Ge => q1("nonnegative", &d)?,
            RelOp::Le => q1("nonpositive", &d)?,
            RelOp::Eq => q1("zero", &d)?,
            // Python's Not(...) constructor: not evaluated
            RelOp::Ne => Basic::new(Head::Not, vec![q1("zero", &d)?]),
        }));
    }
    Ok(None)
}

/// A normalized formula as predicates of `Q`, or `None` (`to_predicates`).
pub fn to_predicates(formula: &Basic) -> PolyResult<Option<Basic>> {
    if is_truth(formula) {
        return Ok(Some(formula.clone()));
    }
    if is_compound(formula) {
        let mut args: Vec<Basic> = Vec::new();
        for arg in formula.args() {
            match to_predicates(arg)? {
                Some(p) => args.push(p),
                None => return Ok(None),
            }
        }
        return Ok(Some(rebuilt(formula, args)?));
    }
    predicate_of_atom(formula)
}

/// The truth of a comparison of real numbers (`_holds`).
fn holds(b: &Basic) -> PolyResult<bool> {
    match b.head() {
        Head::BooleanTrue => Ok(true),
        Head::BooleanFalse => Ok(false),
        _ => Err(PolyError::Type(format!(
            "cannot determine truth value of Relational: {}",
            b
        ))),
    }
}

fn is_number(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}

/// The predicates implied by a top level atom beyond its equivalent one
/// (`_consequences`).
/// The variables an inequality makes real (`_real_variables`): its free
/// symbols, except those met only inside `re`, `im`, `Abs`, `arg` and
/// `conjugate` (the sides of `re(z)*im(z) > pi/2` are real, `z` is not,
/// sympy-extras#74).
fn real_variables(atom: &Basic) -> Vec<Basic> {
    use crate::core::basic::Function;
    let mut found: Vec<Basic> = Vec::new();
    let mut stack: Vec<Basic> = vec![atom.args()[0].clone(), atom.args()[1].clone()];
    while let Some(node) = stack.pop() {
        if matches!(node.head(), Head::Symbol(_) | Head::Dummy(_, _)) {
            found.push(node);
        } else if !matches!(
            node.head(),
            Head::Function(Function::Re | Function::Im | Function::Abs | Function::Arg | Function::Conjugate)
        ) {
            stack.extend(node.args().iter().cloned());
        }
    }
    let free = atom.free_symbols();
    sorted_symbols(found.into_iter().filter(|s| free.contains(s)))
}

pub(crate) fn consequences(atom: &Basic) -> PolyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    let Some(mut op) = rel_op(atom) else { return Ok(result) };
    let (mut lhs, mut rhs) = (atom.args()[0].clone(), atom.args()[1].clone());
    if matches!(op, RelOp::Gt | RelOp::Lt | RelOp::Ge | RelOp::Le) {
        for s in real_variables(atom) {
            result.push(q1("real", &s)?);
        }
    }
    if is_number(&lhs) && !is_number(&rhs) {
        std::mem::swap(&mut lhs, &mut rhs);
        op = rel_op(&crate::core::relational::reversed(atom)).expect("a relation");
    }
    let zero: Basic = Basic::zero();
    if is_number(&rhs) && !is_number(&lhs) && is_true(&rhs, Predicate::extended_real) {
        let c: &Basic = &rhs;
        match op {
            RelOp::Gt => {
                if holds(&ge(c, &zero)?)? {
                    result.push(q1("positive", &lhs)?);
                }
            }
            RelOp::Ge => {
                if holds(&gt(c, &zero)?)? {
                    result.push(q1("positive", &lhs)?);
                } else if holds(&eq(c, &zero)?)? {
                    result.push(q1("nonnegative", &lhs)?);
                } else {
                    result.push(q1("real", &lhs)?);
                }
            }
            RelOp::Lt => {
                if holds(&le(c, &zero)?)? {
                    result.push(q1("negative", &lhs)?);
                }
            }
            RelOp::Le => {
                if holds(&lt(c, &zero)?)? {
                    result.push(q1("negative", &lhs)?);
                } else if holds(&eq(c, &zero)?)? {
                    result.push(q1("nonpositive", &lhs)?);
                } else {
                    result.push(q1("real", &lhs)?);
                }
            }
            RelOp::Eq => {
                if is_true(c, Predicate::integer) {
                    result.push(q1("integer", &lhs)?);
                } else if is_true(c, Predicate::rational) {
                    result.push(q1("rational", &lhs)?);
                } else {
                    result.push(q1("real", &lhs)?);
                }
                if holds(&gt(c, &zero)?)? {
                    result.push(q1("positive", &lhs)?);
                } else if holds(&lt(c, &zero)?)? {
                    result.push(q1("negative", &lhs)?);
                } else {
                    result.push(q1("zero", &lhs)?);
                }
            }
            RelOp::Ne => {}
        }
    }
    Ok(result)
}

/// Whether `expr` is a polynomial with rational coefficients in `symbols`
/// (`_is_polynomial`).
fn is_rational_polynomial(expr: &Basic, symbols: &[Basic]) -> bool {
    let free = expr.free_symbols();
    if !free.iter().all(|s| symbols.contains(s)) {
        return false;
    }
    if free.is_empty() {
        return is_true(expr, Predicate::rational);
    }
    let gens: Vec<Basic> = sorted_symbols(free);
    match crate::polys::polytools::Poly::new(expr, &gens) {
        Ok(p) => p.domain.params.is_empty(),
        Err(_) => false,
    }
}

fn sign_relation(name: &str, x: &Basic) -> PolyResult<Option<Basic>> {
    let zero: Basic = Basic::zero();
    Ok(Some(match name {
        "positive" => gt(x, &zero)?,
        "negative" => lt(x, &zero)?,
        "nonnegative" => ge(x, &zero)?,
        "nonpositive" => le(x, &zero)?,
        "zero" => eq(x, &zero)?,
        "nonzero" => ne(x, &zero)?,
        _ => return Ok(None),
    }))
}

/// The polynomial relation equivalent to an atom between the real
/// variables, or `None` (`_relational_of_atom`).
fn relational_of_atom(atom: &Basic, real: &[Basic]) -> PolyResult<Option<Basic>> {
    if rel_op(atom).is_some() {
        let d: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
        return Ok(if is_rational_polynomial(&d, real) {
            Some(atom.clone())
        } else {
            None
        });
    }
    if matches!(atom.head(), Head::Contains) {
        let (x, domain) = (&atom.args()[0], &atom.args()[1]);
        if matches!(domain.head(), Head::Reals) && is_expr(x) && is_rational_polynomial(x, real) {
            return Ok(Some(boolean(true)));
        }
        return Ok(None);
    }
    if matches!(atom.head(), Head::AppliedPredicate) {
        if atom.args().len() != 2 {
            return Ok(None);
        }
        let x: &Basic = &atom.args()[1];
        if !is_expr(x) || !is_rational_polynomial(x, real) {
            return Ok(None);
        }
        let Head::Predicate(d) = atom.args()[0].head() else {
            return Ok(None);
        };
        if !d.defined {
            return Ok(None);
        }
        if matches!(d.name.as_str(), "real" | "complex" | "extended_real" | "finite") {
            return Ok(Some(boolean(true)));
        }
        return sign_relation(&d.name, x);
    }
    Ok(None)
}

/// A normalized formula as polynomial relations between the real
/// variables, or `None` (`to_polynomial`).
pub fn to_polynomial(formula: &Basic, real: &[Basic]) -> PolyResult<Option<Basic>> {
    if is_truth(formula) {
        return Ok(Some(formula.clone()));
    }
    if is_compound(formula) {
        let mut args: Vec<Basic> = Vec::new();
        for arg in formula.args() {
            match to_polynomial(arg, real)? {
                Some(p) => args.push(p),
                None => return Ok(None),
            }
        }
        return Ok(Some(rebuilt(formula, args)?));
    }
    relational_of_atom(formula, real)
}

/// The assumptions in force, translated for the backends (`Facts`).
#[derive(Debug, Clone)]
pub struct Facts {
    pub domain: Option<Basic>,
    pub formula: Basic,
    pub conjuncts: Vec<Basic>,
    pub predicates: Basic,
    /// The variables known to be real, sorted by name.
    pub real: Vec<Basic>,
    /// The variables known to be integers, sorted by name.
    pub integer: Vec<Basic>,
    pub polynomial: Basic,
}

impl Facts {
    pub fn new(assumptions: &[Basic], domain: Option<&Basic>, symbols: &[Basic]) -> PolyResult<Facts> {
        let mut items: Vec<Basic> = assumptions.to_vec();
        let mut variables_given: BTreeSet<Basic> = symbols.iter().cloned().collect();
        if let Some(d) = domain {
            if !crate::sets::sets::is_set(d) {
                return Err(PolyError::Type(format!("domain must be a set, got {}", d)));
            }
            for item in &items {
                variables_given.extend(item.free_symbols());
            }
            for s in sorted_symbols(variables_given.iter().cloned()) {
                items.push(contains_node(&s, d));
            }
        }
        let formula: Basic = normalize(&and_(&items))?;
        if matches!(formula.head(), Head::BooleanFalse) {
            let shown: Vec<String> = items.iter().map(|i| i.to_string()).collect();
            return Err(PolyError::Value(format!(
                "the assumptions are contradictory: [{}]",
                shown.join(", ")
            )));
        }
        let conjuncts_: Vec<Basic> = conjuncts(&formula);
        let mut predicates: Vec<Basic> = Vec::new();
        for c in &conjuncts_ {
            if let Some(p) = to_predicates(c)? {
                predicates.push(p);
            }
            if !is_compound(c) {
                predicates.extend(consequences(c)?);
            }
        }
        let predicates: Basic = and_(&predicates);
        let known = |p: &Basic| -> bool { matches!(ask(p, &predicates, None), Ok(Some(true))) };
        let mut real: BTreeSet<Basic> = BTreeSet::new();
        let mut integer: BTreeSet<Basic> = BTreeSet::new();
        for s in formula
            .free_symbols()
            .into_iter()
            .chain(variables_given.iter().cloned())
        {
            // is_real, not is_extended_real: a symbol only extended real
            // may be oo (sympy-extras#74, 5)
            if is_true(&s, Predicate::real) || known(&q1("real", &s)?) {
                real.insert(s.clone());
            }
            if is_true(&s, Predicate::integer) || known(&q1("integer", &s)?) {
                integer.insert(s.clone());
            }
        }
        for c in &conjuncts_ {
            if matches!(rel_op(c), Some(RelOp::Gt | RelOp::Lt | RelOp::Ge | RelOp::Le)) {
                real.extend(real_variables(c));
            }
        }
        let real: Vec<Basic> = sorted_symbols(real);
        let integer: Vec<Basic> = sorted_symbols(integer);
        let mut polynomial: Vec<Basic> = Vec::new();
        for c in &conjuncts_ {
            if let Some(p) = to_polynomial(c, &real)? {
                if !matches!(p.head(), Head::BooleanTrue) {
                    polynomial.push(p);
                }
            }
        }
        Ok(Facts {
            domain: domain.cloned(),
            formula,
            conjuncts: conjuncts_,
            predicates,
            real,
            integer,
            polynomial: and_(&polynomial),
        })
    }

    /// Whether the normalized formula can be handled by the CAD.
    pub fn is_polynomial(&self, formula: &Basic) -> PolyResult<bool> {
        Ok(to_polynomial(formula, &self.real)?.is_some())
    }

    /// The same facts with `symbols` also assumed real (`with_reals`).
    pub fn with_reals(&self, symbols: &[Basic]) -> PolyResult<Facts> {
        let extra: Vec<Basic> = symbols.iter().filter(|s| !self.real.contains(s)).cloned().collect();
        if extra.is_empty() {
            return Ok(self.clone());
        }
        let mut items: Vec<Basic> = self.conjuncts.clone();
        for s in &extra {
            items.push(contains_node(s, &Basic::atom(Head::Reals)));
        }
        Facts::new(&items, None, &[])
    }
}
