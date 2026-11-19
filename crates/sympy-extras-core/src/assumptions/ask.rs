//! Port of `sympy_extras/assumptions/ask.py`: the truth value of a statement
//! under assumptions. The predicates go to the core's `ask`, polynomial
//! formulas to virtual substitution or the cylindrical algebraic
//! decomposition, the other relations to calculus and interval
//! arithmetic (`assumptions/analysis.rs`) and to polynomial bounds of the
//! functions (`assumptions/bounds.rs`).
//!
//! Not ported yet (they raise "not ported" where the Python goes on): the
//! zero test of the structure theorem (`simplify/structure.py`), the
//! sample points under assumptions (`integrals/conditions.py`). The time limits
//! of the Python (`attempt` with seconds) are not ported.

#![allow(clippy::mutable_key_type)]

use std::cell::Cell;
use std::collections::BTreeSet;

use crate::assumptions::facts::{element, normalize, predicate_of_atom, to_polynomial, Facts};
use crate::assumptions::quantifiers::{Prefix, QuantifierKind};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::core::logic::{and_, not_, or_};
use crate::error::{not_ported, PolyError, PolyResult};

/// `True`, `False` or `None`.
pub type Truth = Option<bool>;

fn symbols(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect()
}

fn sorted_symbols(s: BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = s.into_iter().collect();
    v.sort_by_key(|s| s.to_string());
    v
}

/// An applied function as SymPy's `Function` sees it: `Max` and `Min` are
/// not, `Piecewise` is (`_is_function`).
fn is_function(e: &Basic) -> bool {
    match e.head() {
        Head::Function(f) => !matches!(f, Function::Max | Function::Min),
        Head::Undefined(_) | Head::Piecewise | Head::Hyper | Head::MeijerG => true,
        _ => false,
    }
}

pub(crate) fn has_function(e: &Basic) -> bool {
    e.preorder().iter().any(is_function)
}

/// Whether `e` has a power whose exponent is not known to be an integer
/// (`_has_fractional_power`).
pub(crate) fn has_fractional_power(e: &Basic) -> bool {
    e.preorder()
        .iter()
        .any(|p| p.is_pow() && is(&p.args()[1], Predicate::integer) != Some(true))
}

/// The facts from the global assumptions and the explicit ones (`_facts`).
pub fn facts_of(assumptions: &[Basic], domain: Option<&Basic>, symbols: &[Basic]) -> PolyResult<Facts> {
    let mut items: Vec<Basic> = crate::assumptions::context::global_assumptions();
    items.extend(assumptions.iter().cloned());
    let plain: Vec<Basic> = symbols.iter().filter(|s| s.is_symbol()).cloned().collect();
    Facts::new(&items, domain, &plain)
}

fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}
fn is_false(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanFalse)
}
fn is_atom_bool(b: &Basic) -> bool {
    is_true(b) || is_false(b)
}

fn cad_ask(formula: &Basic, facts: &Facts) -> PolyResult<Truth> {
    let Some(poly) = to_polynomial(formula, &facts.real)? else {
        return Ok(None);
    };
    if is_atom_bool(&poly) {
        return Ok(Some(is_true(&poly)));
    }
    let premise: Basic = facts.polynomial.clone();
    let mut all: BTreeSet<Basic> = symbols(&poly);
    all.extend(symbols(&premise));
    let gens: Vec<Basic> = sorted_symbols(all);
    if gens.is_empty() {
        return Ok(Some(is_true(&poly)));
    }
    if let Some(linear) = linear_ask(&poly, &premise, &gens)? {
        return Ok(Some(linear));
    }
    let (_, tables) = crate::polys::cad::qe::truth_tables(&[premise, poly], &gens, None, false)?;
    let relevant: Vec<bool> = tables[0]
        .iter()
        .zip(tables[1].iter())
        .filter(|(h, _)| **h)
        .map(|(_, v)| *v)
        .collect();
    if relevant.is_empty() {
        return Ok(None);
    }
    if relevant.iter().all(|v| *v) {
        return Ok(Some(true));
    }
    if !relevant.iter().any(|v| *v) {
        return Ok(Some(false));
    }
    Ok(None)
}

fn linear_ask(poly: &Basic, premise: &Basic, gens: &[Basic]) -> PolyResult<Truth> {
    use crate::polys::virtual_substitution::{is_linear_in, linear_quantifier_elimination};
    let combined: Basic = and_(&[premise.clone(), poly.clone()]);
    for g in gens {
        if !is_linear_in(&combined, g)? {
            return Ok(None);
        }
    }
    let prefix: Prefix = gens.iter().map(|g| (QuantifierKind::Exists, g.clone())).collect();
    let (counterexamples, rest) = linear_quantifier_elimination(&and_(&[premise.clone(), not_(poly)]), &prefix)?;
    let (examples, rest_) = linear_quantifier_elimination(&combined, &prefix)?;
    if !rest.is_empty() || !rest_.is_empty() || !is_atom_bool(&counterexamples) || !is_atom_bool(&examples) {
        return Ok(None);
    }
    if is_false(&examples) && is_false(&counterexamples) {
        return Ok(None);
    }
    if is_false(&counterexamples) {
        return Ok(Some(true));
    }
    if is_false(&examples) {
        return Ok(Some(false));
    }
    Ok(None)
}

/// The core's `ask` of a predicate, `None` when it gives up
/// (`_predicate_ask`).
fn predicate_ask(predicate: &Basic, facts: &Facts) -> PolyResult<Truth> {
    match crate::core::ask::ask(predicate, &facts.predicates, None) {
        Ok(t) => Ok(t),
        Err(PolyError::Value(_) | PolyError::Type(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn rel_op(atom: &Basic) -> Option<RelOp> {
    match atom.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}

/// The truth of an atom (`_evaluate_atom`).
pub fn evaluate_atom(atom: &Basic, facts: &Facts) -> PolyResult<Truth> {
    if is_atom_bool(atom) {
        return Ok(Some(is_true(atom)));
    }
    if matches!(rel_op(atom), Some(RelOp::Gt | RelOp::Lt | RelOp::Ge | RelOp::Le)) {
        let d: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
        let real: Basic = crate::core::assume::applied(&crate::core::assume::q("real", 1), &[d]);
        if predicate_ask(&real, facts)? == Some(false) {
            return Ok(Some(false));
        }
    }
    if let Some(member) = real_membership(atom) {
        if let Some(v) = realness(&member, facts)? {
            return Ok(Some(v));
        }
        if fragile(&member) {
            return Ok(None);
        }
    }
    if let Some(predicate) = predicate_of_atom(atom)? {
        if let Some(v) = predicate_ask(&predicate, facts)? {
            return Ok(Some(v));
        }
    }
    if rel_op(atom).is_some() && atom.free_symbols().is_empty() {
        let done: Basic = crate::core::function::doit(atom)?;
        if is_atom_bool(&done) {
            return Ok(Some(is_true(&done)));
        }
    }
    let mut value: Truth = cad_ask(atom, facts)?;
    if value.is_none() && rel_op(atom).is_some() {
        value = structure_ask(atom, facts)?;
    }
    if value.is_none() && rel_op(atom).is_some() {
        value = crate::assumptions::analysis::decide_relational(atom, facts)?;
    }
    if value.is_none() {
        value = bounded_cad_ask(atom, facts)?;
    }
    Ok(value)
}

thread_local! {
    /// The depth of the structure theorem inside `ask` (`_structure_depth`).
    static STRUCTURE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

fn structure_ask(atom: &Basic, facts: &Facts) -> PolyResult<Truth> {
    if STRUCTURE_DEPTH.with(|d| d.get()) > 0 {
        return Ok(None);
    }
    let difference: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
    if !(has_function(&difference)
        || has_fractional_power(&difference)
        || difference.has(&Basic::pi())
        || difference.has(&Basic::e()))
    {
        return Ok(None);
    }
    if !difference.free_symbols().is_empty() && clearly_not_zero(&difference, facts)? {
        return Ok(None);
    }
    Err(not_ported("the zero test of simplify/structure (ElementaryTower)"))
}

fn clearly_not_zero(difference: &Basic, facts: &Facts) -> PolyResult<bool> {
    let syms: Vec<Basic> = sorted_symbols(symbols(difference));
    if !facts.conjuncts.is_empty() {
        return Err(not_ported("integrals/conditions sample_values"));
    }
    let values: Vec<(Basic, Basic)> = syms
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let r = crate::numbers::Rat::from_parts(
                crate::numbers::Int::from(3 + 2 * i as i64),
                dashu_int::UBig::from(7 + i as u64),
            );
            (s.clone(), Basic::from_rational(r))
        })
        .collect();
    Ok(crate::numeric::reliable_complex(difference, 20, &values).is_some_and(|(a, b)| a.hypot(b) > 1e-12))
}

fn real_membership(atom: &Basic) -> Option<Basic> {
    if matches!(atom.head(), Head::Contains) {
        let (member, s) = (&atom.args()[0], &atom.args()[1]);
        if matches!(s.head(), Head::Reals) && is_expr(member) {
            return Some(member.clone());
        }
    }
    if matches!(atom.head(), Head::AppliedPredicate) {
        let p = &atom.args()[0];
        if crate::core::assume::predicate_data(p).is_some_and(|d| d.name == "real")
            && atom.args().len() > 1
            && is_expr(&atom.args()[1])
        {
            return Some(atom.args()[1].clone());
        }
    }
    None
}

fn is_expr(e: &Basic) -> bool {
    !(crate::core::logic::is_boolean(e)
        || crate::sets::sets::is_set(e)
        || matches!(e.head(), Head::Tuple | Head::ExprCondPair))
}

fn fragile(e: &Basic) -> bool {
    has_fractional_power(e)
        || e.preorder().iter().any(|n| {
            matches!(
                n.head(),
                Head::Function(Function::Log | Function::Asin | Function::Acos | Function::Acosh | Function::Atanh)
            )
        })
}

fn function_of(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}

/// Whether `e` is a real number, by its structure and the facts about the
/// signs of its parts (`_realness`).
fn realness(e: &Basic, facts: &Facts) -> PolyResult<Truth> {
    use crate::core::relational as rl;
    let ev = |f: &Basic| evaluate(f, facts);
    let int = |n: i64| Basic::int(n);
    if e.is_symbol() {
        return Ok(
            if facts.real.contains(e) || is(e, Predicate::extended_real) == Some(true) {
                Some(true)
            } else {
                None
            },
        );
    }
    if crate::core::assumptions::is_number(e) {
        return Ok(is(e, Predicate::extended_real));
    }
    if e.is_add() || e.is_mul() {
        for a in e.args() {
            if realness(a, facts)? != Some(true) {
                return Ok(None);
            }
        }
        return Ok(Some(true));
    }
    if e.is_pow() {
        let (base, exponent) = (&e.args()[0], &e.args()[1]);
        let base_real: Truth = realness(base, facts)?;
        if is(exponent, Predicate::integer) == Some(true) {
            return Ok(base_real);
        }
        if base_real != Some(true) {
            return Ok(None);
        }
        let rational: bool = exponent.is_rational_class();
        if rational || realness(exponent, facts)? == Some(true) {
            if ev(&rl::ge(base, &int(0))?)? == Some(true) {
                return Ok(Some(true));
            }
            if rational && ev(&rl::lt(base, &int(0))?)? == Some(true) {
                return Ok(Some(false));
            }
        }
        return Ok(None);
    }
    let Some(f) = function_of(e) else { return Ok(None) };
    let argument: &Basic = &e.args()[0];
    let decide = |holds: Basic, fails: Basic| -> PolyResult<Truth> {
        if ev(&holds)? == Some(true) {
            return Ok(Some(true));
        }
        if ev(&fails)? == Some(true) {
            return Ok(Some(false));
        }
        Ok(None)
    };
    match f {
        Function::Log => decide(rl::gt(argument, &int(0))?, rl::lt(argument, &int(0))?),
        Function::Asin | Function::Acos => decide(
            and_(&[rl::ge(argument, &int(-1))?, rl::le(argument, &int(1))?]),
            or_(&[rl::gt(argument, &int(1))?, rl::lt(argument, &int(-1))?]),
        ),
        Function::Acosh => decide(rl::ge(argument, &int(1))?, rl::lt(argument, &int(1))?),
        Function::Atanh => decide(
            and_(&[rl::gt(argument, &int(-1))?, rl::lt(argument, &int(1))?]),
            or_(&[rl::gt(argument, &int(1))?, rl::lt(argument, &int(-1))?]),
        ),
        Function::Abs => Ok(if is(argument, Predicate::finite) != Some(false) {
            Some(true)
        } else {
            None
        }),
        Function::Exp
        | Function::Sin
        | Function::Cos
        | Function::Tan
        | Function::Atan
        | Function::Sinh
        | Function::Cosh
        | Function::Tanh
        | Function::Asinh
        | Function::Floor
        | Function::Ceiling => Ok(if realness(argument, facts)? == Some(true) {
            Some(true)
        } else {
            None
        }),
        _ => Ok(None),
    }
}

fn bounded_cad_ask(formula: &Basic, facts: &Facts) -> PolyResult<Truth> {
    if !has_function(formula) {
        return Ok(None);
    }
    let Some(abstraction) = crate::assumptions::bounds::polynomial_abstraction(&normalize(formula)?, &facts.real, 3)?
    else {
        return Ok(None);
    };
    for requirement in &abstraction.requirements {
        if cad_ask(&normalize(requirement)?, facts)? != Some(true) {
            return Ok(None);
        }
    }
    let mut items: Vec<Basic> = facts.conjuncts.clone();
    items.extend(abstraction.constraints.iter().cloned());
    for t in &abstraction.variables {
        items.push(element(t, &crate::sets::sets::reals())?);
    }
    let extended: Facts = match Facts::new(&items, None, &[]) {
        Ok(f) => f,
        Err(PolyError::Value(_) | PolyError::Type(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut all: BTreeSet<Basic> = symbols(&extended.polynomial);
    all.extend(symbols(&abstraction.formula));
    if all.len() > 3 {
        return Ok(None);
    }
    let bounded: Basic = normalize(&abstraction.formula)?;
    Ok(crate::timeout::attempt(|| cad_ask(&bounded, &extended))?.flatten())
}

/// The three-valued evaluation of a normalized formula (`_evaluate`).
pub fn evaluate(formula: &Basic, facts: &Facts) -> PolyResult<Truth> {
    if is_atom_bool(formula) {
        return Ok(Some(is_true(formula)));
    }
    if matches!(formula.head(), Head::Not) {
        return Ok(evaluate(&formula.args()[0], facts)?.map(|v| !v));
    }
    if matches!(
        formula.head(),
        Head::And | Head::Or | Head::Implies | Head::Equivalent | Head::Xor | Head::Ite
    ) {
        let values: Vec<Truth> = formula
            .args()
            .iter()
            .map(|a| evaluate(a, facts))
            .collect::<PolyResult<_>>()?;
        let mut value: Truth = combine(formula, &values);
        if value.is_none() {
            value = cad_ask(formula, facts)?;
        }
        if value.is_none() {
            value = bounded_cad_ask(formula, facts)?;
        }
        return Ok(value);
    }
    evaluate_atom(formula, facts)
}

fn combine(formula: &Basic, values: &[Truth]) -> Truth {
    match formula.head() {
        Head::And => {
            if values.contains(&Some(false)) {
                Some(false)
            } else if values.iter().all(|v| *v == Some(true)) {
                Some(true)
            } else {
                None
            }
        }
        Head::Or => {
            if values.contains(&Some(true)) {
                Some(true)
            } else if values.iter().all(|v| *v == Some(false)) {
                Some(false)
            } else {
                None
            }
        }
        Head::Implies => match (values[0], values[1]) {
            (Some(false), _) | (_, Some(true)) => Some(true),
            (Some(true), Some(false)) => Some(false),
            _ => None,
        },
        Head::Equivalent => {
            if values.iter().any(|v| v.is_none()) {
                None
            } else {
                Some(values.iter().all(|v| *v == values[0]))
            }
        }
        Head::Xor => {
            if values.iter().any(|v| v.is_none()) {
                None
            } else {
                Some(values.iter().filter(|v| **v == Some(true)).count() % 2 == 1)
            }
        }
        Head::Ite => match values[0] {
            Some(true) => values[1],
            Some(false) => values[2],
            None => {
                if values[1].is_some() && values[1] == values[2] {
                    values[1]
                } else {
                    None
                }
            }
        },
        _ => None,
    }
}

/// The truth value of a statement under assumptions (`ask`): `True` if it
/// follows from them, `False` if its negation does, `None` otherwise.
pub fn ask(query: &Basic, assumptions: &[Basic], domain: Option<&Basic>) -> PolyResult<Truth> {
    let mut q: Basic = query.clone();
    if q.preorder().iter().any(crate::assumptions::quantifiers::is_quantifier) {
        q = crate::assumptions::resolve::resolve(&q, domain, assumptions, None)?;
    }
    q = normalize(&q)?;
    let syms: Vec<Basic> = q.free_symbols().into_iter().collect();
    let facts: Facts = facts_of(assumptions, domain, &syms)?;
    evaluate(&q, &facts)
}
