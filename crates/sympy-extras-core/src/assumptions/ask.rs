//! Port of `sympy_extras/assumptions/ask.py`: the truth value of a statement
//! under assumptions. The predicates go to the core's `ask`, polynomial
//! formulas to virtual substitution or the cylindrical algebraic
//! decomposition, the other relations to calculus and interval
//! arithmetic (`assumptions/analysis.rs`) and to polynomial bounds of the
//! functions (`assumptions/bounds.rs`).
//!
//! The time limits of the Python are those of `crate::timeout`.

#![allow(clippy::mutable_key_type)]

use std::cell::Cell;
use std::collections::BTreeSet;

use crate::assumptions::facts::{element, normalize, predicate_of_atom, to_polynomial, Facts};
use crate::assumptions::quantifiers::{Prefix, QuantifierKind};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::core::logic::{and_, not_, or_};
use crate::error::{PolyError, PolyResult};

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
        let mut v: Truth = realness(&member, facts)?;
        if v.is_none() && real_by_parts(&member, facts)? == Some(false) {
            // the imaginary part is not zero: Q.real(z) under
            // re(z)*im(z) > pi/2 (sympy-extras#74)
            v = Some(false);
        }
        if let Some(v) = v {
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
    if value.is_none() && matches!(rel_op(atom), Some(RelOp::Eq | RelOp::Ne)) {
        // complex variables: the real and the imaginary parts apart
        let difference: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
        if let Some(zero) = zero_by_parts(&difference, facts)? {
            value = Some(if rel_op(atom) == Some(RelOp::Eq) { zero } else { !zero });
        }
    }
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

// ---------------------------------------------------------------------
// complex variables by their real and imaginary parts

/// The real and the imaginary parts of an expression of complex
/// variables, written in real variables, with the facts translated
/// (`_ComplexParts`).
struct ComplexParts {
    re: Basic,
    im: Basic,
    facts: Facts,
}

fn unevaluated(f: Function, z: &Basic) -> Basic {
    Basic::new(Head::Function(f), vec![z.clone()])
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

/// Whether `re(z)` and `im(z)` are finite real numbers (`_finite_parts`).
fn finite_parts(z: &Basic, facts: &Facts) -> bool {
    if is(z, Predicate::finite) == Some(true) {
        return true;
    }
    let mut found: Vec<Basic> = Vec::new();
    for c in &facts.conjuncts {
        if matches!(rel_op(c), Some(RelOp::Gt | RelOp::Lt | RelOp::Ge | RelOp::Le)) {
            for node in c.preorder() {
                if matches!(function_of(&node), Some(Function::Re | Function::Im | Function::Abs))
                    && node.args()[0] == *z
                    && !found.contains(&node)
                {
                    found.push(node);
                }
            }
        }
    }
    found.contains(&unevaluated(Function::Abs, z))
        || (found.contains(&unevaluated(Function::Re, z)) && found.contains(&unevaluated(Function::Im, z)))
}

fn xreplace_with(e: &Basic, rule: &[(Basic, Basic)]) -> PolyResult<Basic> {
    let map: std::collections::HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

/// The parts of `e` in real variables, or `None` when `e` has no complex
/// variable, one whose parts may be infinite, or something which is not
/// translated (`_complex_parts`).
fn complex_parts(e: &Basic, facts: &Facts) -> PolyResult<Option<ComplexParts>> {
    use crate::core::relational::{eq, ge, gt, le, lt};
    let all: BTreeSet<Basic> = symbols(e);
    let complex_: Vec<Basic> = sorted_symbols(
        all.iter()
            .filter(|s| !facts.real.contains(s) && is(s, Predicate::real) != Some(true))
            .cloned()
            .collect(),
    );
    if complex_.is_empty() || !complex_.iter().all(|z| finite_parts(z, facts)) {
        return Ok(None);
    }
    let (re_part, im_part) = match crate::functions::elementary::complexes::as_real_imag(e, true, None)? {
        Some(parts) => parts,
        None => (
            crate::functions::elementary::complexes::re(e)?,
            crate::functions::elementary::complexes::im(e)?,
        ),
    };
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for s in sorted_symbols(all.iter().filter(|s| !complex_.contains(s)).cloned().collect()) {
        rule.push((unevaluated(Function::Re, &s), s.clone()));
        rule.push((unevaluated(Function::Im, &s), Basic::zero()));
    }
    let mut variables: Vec<Basic> = Vec::new();
    let mut constraints: Vec<Basic> = Vec::new();
    let mut nodes: Vec<Basic> = Vec::new();
    for c in [re_part.clone(), im_part.clone()].iter().chain(facts.conjuncts.iter()) {
        for n in c.preorder() {
            if !nodes.contains(&n) {
                nodes.push(n);
            }
        }
    }
    let real = |name: String| crate::core::symbol::Dummy::new(&name, &[(Predicate::real, true)]).0;
    for z in &complex_ {
        let name: String = symbol_name(z);
        let (x, y) = (real(format!("re_{}", name)), real(format!("im_{}", name)));
        rule.push((unevaluated(Function::Re, z), x.clone()));
        rule.push((unevaluated(Function::Im, z), y.clone()));
        variables.push(x.clone());
        variables.push(y.clone());
        if nodes.contains(&unevaluated(Function::Abs, z)) {
            let r: Basic = real(format!("abs_{}", name));
            rule.push((unevaluated(Function::Abs, z), r.clone()));
            variables.push(r.clone());
            let two: Basic = Basic::int(2);
            let sum: Basic =
                crate::core::add::add(&[crate::core::power::pow(&x, &two)?, crate::core::power::pow(&y, &two)?])?;
            constraints.push(ge(&r, &Basic::zero())?);
            constraints.push(eq(&crate::core::power::pow(&r, &two)?, &sum)?);
        }
    }
    let p: Basic = real("pi".to_string());
    let mut angles: Vec<Basic> = Vec::new();
    let mut requirements: Vec<Basic> = Vec::new();
    let mut ordered_nodes: Vec<Basic> = nodes.clone();
    ordered_nodes.sort_by_cached_key(|n| n.to_string());
    for node in &ordered_nodes {
        if function_of(node) == Some(Function::Arg) && is(&node.args()[0], Predicate::zero) == Some(false) {
            let t: Basic = real("t".to_string());
            rule.push((node.clone(), t.clone()));
            angles.push(t);
        } else if function_of(node) == Some(Function::Atan2) {
            let (b, a) = (&node.args()[0], &node.args()[1]);
            let (b_, a_) = (xreplace_with(b, &rule)?, xreplace_with(a, &rule)?);
            let free: BTreeSet<Basic> = a_.free_symbols().union(&b_.free_symbols()).cloned().collect();
            if complex_.iter().any(|z| free.contains(z)) {
                continue;
            }
            let t: Basic = real("t".to_string());
            rule.push((node.clone(), t.clone()));
            angles.push(t);
            // atan2(0, 0) is not an angle
            let two: Basic = Basic::int(2);
            requirements.push(crate::core::add::add(&[
                crate::core::power::pow(&a_, &two)?,
                crate::core::power::pow(&b_, &two)?,
            ])?);
        }
    }
    if !angles.is_empty() || nodes.contains(&Basic::pi()) {
        rule.push((Basic::pi(), p.clone()));
        variables.push(p.clone());
        constraints.push(gt(&p, &Basic::from_rational(crate::numbers::rat(314159, 100000)))?);
        constraints.push(lt(&p, &Basic::from_rational(crate::numbers::rat(314160, 100000)))?);
    }
    for angle in &angles {
        variables.push(angle.clone());
        constraints.push(gt(angle, &crate::core::mul::neg(&p)?)?);
        constraints.push(le(angle, &p)?);
    }
    let re_new: Basic = xreplace_with(&re_part, &rule)?;
    let im_new: Basic = xreplace_with(&im_part, &rule)?;
    let free: BTreeSet<Basic> = re_new.free_symbols().union(&im_new.free_symbols()).cloned().collect();
    if complex_.iter().any(|z| free.contains(z)) {
        return Ok(None);
    }
    let mut kept: Vec<Basic> = Vec::new();
    for c in &facts.conjuncts {
        let translated: Basic = xreplace_with(c, &rule)?;
        let f = translated.free_symbols();
        if !complex_.iter().any(|z| f.contains(z)) {
            kept.push(translated);
        }
    }
    kept.extend(constraints);
    let reals: Basic = crate::sets::sets::reals();
    for v in &variables {
        kept.push(element(v, &reals)?);
    }
    let extended: Facts = match Facts::new(&kept, None, &[]) {
        Ok(f) => f,
        Err(PolyError::Value(_) | PolyError::Type(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    for q in &requirements {
        if !nonzero(q, &extended)? {
            return Ok(None);
        }
    }
    Ok(Some(ComplexParts {
        re: re_new,
        im: im_new,
        facts: extended,
    }))
}

/// Whether a difference of expressions of complex variables is zero, by
/// its real and imaginary parts (`_zero_by_parts`).
fn zero_by_parts(difference: &Basic, facts: &Facts) -> PolyResult<Truth> {
    let Some(parts) = complex_parts(difference, facts)? else {
        return Ok(None);
    };
    let im_zero: Truth = zero_part(&parts.im, &parts.facts)?;
    if im_zero == Some(false) {
        return Ok(Some(false));
    }
    let re_zero: Truth = zero_part(&parts.re, &parts.facts)?;
    if re_zero == Some(false) {
        return Ok(Some(false));
    }
    if im_zero == Some(true) && re_zero == Some(true) {
        return Ok(Some(true));
    }
    Ok(None)
}

/// `false` when the imaginary part of `e` is not zero on the region of the
/// facts (`_real_by_parts`).
fn real_by_parts(e: &Basic, facts: &Facts) -> PolyResult<Truth> {
    let Some(parts) = complex_parts(e, facts)? else {
        return Ok(None);
    };
    if zero_part(&parts.im, &parts.facts)? == Some(false) {
        return Ok(Some(false));
    }
    Ok(None)
}

/// The budget of a part which is not polynomial (`_PARTS_SECONDS`).
const PARTS_SECONDS: f64 = 2.0;

/// Whether a part in real variables is zero on the region of the facts
/// (`_zero_part`).
fn zero_part(part: &Basic, facts: &Facts) -> PolyResult<Truth> {
    let formula: Basic = normalize(&crate::core::relational::eq(part, &Basic::zero())?)?;
    if facts.is_polynomial(&formula)? {
        return cad_ask(&formula, facts);
    }
    Ok(crate::timeout::attempt_within(Some(PARTS_SECONDS), || evaluate(&formula, facts))?.flatten())
}

/// The budget of the structure theorem inside `ask` (`_STRUCTURE_SECONDS`).
const STRUCTURE_SECONDS: f64 = 6.0;

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
    let conjuncts: Vec<Basic> = facts.conjuncts.clone();
    STRUCTURE_DEPTH.with(|d| d.set(d.get() + 1));
    let zero = crate::timeout::attempt_within(Some(STRUCTURE_SECONDS), || structure_zero(&difference, &conjuncts));
    STRUCTURE_DEPTH.with(|d| d.set(d.get() - 1));
    let Some(zero) = zero?.flatten() else { return Ok(None) };
    let op: Option<RelOp> = rel_op(atom);
    match op {
        Some(RelOp::Eq) => return Ok(Some(zero)),
        Some(RelOp::Ne) => return Ok(Some(!zero)),
        _ => {}
    }
    if !zero {
        return Ok(None);
    }
    if matches!(op, Some(RelOp::Gt | RelOp::Lt)) {
        return Ok(Some(false));
    }
    Ok(if realness(&atom.args()[0], facts)? == Some(true) {
        Some(true)
    } else {
        None
    })
}

/// The structure theorem's verdict on the region of the conjuncts
/// (`_structure_zero`).
fn structure_zero(difference: &Basic, conjuncts: &[Basic]) -> PolyResult<Option<bool>> {
    use crate::simplify::structure::{is_not_elementary, Tower};
    let mut tower: Tower = Tower::new(conjuncts);
    let value = match tower.element(difference) {
        Ok(v) => v,
        Err(e) if is_not_elementary(&e) || matches!(e, PolyError::ZeroDivision(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if tower.vanishes(&value)? {
        return Ok(Some(true));
    }
    if difference.free_symbols().is_empty() && tower.certifies(&value)? {
        return Ok(Some(false));
    }
    Ok(None)
}

fn clearly_not_zero(difference: &Basic, facts: &Facts) -> PolyResult<bool> {
    let syms: Vec<Basic> = sorted_symbols(symbols(difference));
    let mut values: Vec<(Basic, Basic)> = syms
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
    if !facts.conjuncts.is_empty() {
        let mut rng = crate::random::PyRandom::from_string(&difference.to_string());
        match crate::integrals::conditions::sample_values(&syms, &facts.conjuncts, &mut rng)? {
            Some(v) => values = v,
            None => return Ok(false),
        }
    }
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
        return Ok(if facts.real.contains(e) || is(e, Predicate::real) == Some(true) {
            Some(true)
        } else {
            None
        });
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
            // a non-real base can have a real power (I**2), and 1/x is zoo
            // at x = 0
            if base_real != Some(true) {
                return Ok(None);
            }
            if is(exponent, Predicate::nonnegative) == Some(true) || nonzero(base, facts)? {
                return Ok(Some(true));
            }
            return Ok(None);
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
        Function::Abs => {
            if is(argument, Predicate::finite) == Some(true) || realness(argument, facts)? == Some(true) {
                return Ok(Some(true));
            }
            let finite: Basic =
                crate::core::assume::applied(&crate::core::assume::q("finite", 1), std::slice::from_ref(argument));
            Ok(if predicate_ask(&finite, facts)? == Some(true) {
                Some(true)
            } else {
                None
            })
        }
        Function::Tan => {
            let cosine: Basic = crate::functions::apply_function(Function::Cos, std::slice::from_ref(argument))?;
            if realness(argument, facts)? == Some(true) && nonzero(&cosine, facts)? {
                return Ok(Some(true));
            }
            Ok(None)
        }
        Function::Exp
        | Function::Sin
        | Function::Cos
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

/// Whether `e` is known to be nonzero under the facts (`_nonzero`).
fn nonzero(e: &Basic, facts: &Facts) -> PolyResult<bool> {
    if is(e, Predicate::zero) == Some(false) {
        return Ok(true);
    }
    Ok(evaluate(&normalize(&crate::core::relational::ne(e, &Basic::int(0))?)?, facts)? == Some(true))
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
    let limit: Option<f64> = crate::settings::settings().timeout.map(|t| t.min(5.0));
    Ok(crate::timeout::attempt_within(limit, || cad_ask(&bounded, &extended))?.flatten())
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
            // the predicates of the whole formula: an atom undecided alone
            // decides nothing in the combination of the atoms, but
            // element(x, Reals) | ~element(x, Reals) is Q.real(x) |
            // ~Q.real(x), and Eq(x, 0) | Eq(y, 0) follows from Q.zero(x*y)
            // (sympy-extras#74, 2 and 3)
            if let Some(predicates) = crate::assumptions::facts::to_predicates(formula)? {
                value = predicate_ask(&predicates, facts)?;
            }
        }
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
