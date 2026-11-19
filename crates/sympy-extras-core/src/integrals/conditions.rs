//! Port of `sympy_extras/integrals/conditions.py`: values which hold under
//! a condition on the parameters (`ConditionalValue`), the conditions
//! decided against the assumptions, and the numerical comparisons of the
//! integrators: the sample points of the region of the assumptions
//! (`sample_values`), `numerically_equal` and `numerical_verdict`, with the
//! structure theorem as a second opinion.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::assumptions::ask::ask;
use crate::assumptions::facts::{element, Facts};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::core::function::xreplace_evaluated;
use crate::core::logic::{and_, boolean, or_};
use crate::core::relational as rl;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::random::PyRandom;

type R = PolyResult<Basic>;
/// Values of symbols.
pub type Values = Vec<(Basic, Basic)>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rational(p: i64, q: u64) -> Basic {
    Basic::from_rational(Rat::from_parts(Int::from(p), dashu_int::UBig::from(q)))
}
fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}
fn is_false(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanFalse)
}
fn minus(e: &Basic) -> R {
    crate::core::mul::mul(&[int(-1), e.clone()])
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), minus(b)?])
}

/// A value with the condition under which it holds (`ConditionalValue`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConditionalValue {
    pub value: Basic,
    pub condition: Basic,
}

impl ConditionalValue {
    pub fn new(value: Basic, condition: Basic) -> ConditionalValue {
        ConditionalValue { value, condition }
    }

    /// The value multiplied by `factor` (`scaled`).
    pub fn scaled(&self, factor: &Basic) -> PolyResult<ConditionalValue> {
        Ok(ConditionalValue::new(
            crate::core::mul::mul(&[factor.clone(), self.value.clone()])?,
            self.condition.clone(),
        ))
    }

    /// The sum, holding when both conditions do (`add`).
    pub fn add(&self, other: &ConditionalValue) -> PolyResult<ConditionalValue> {
        Ok(ConditionalValue::new(
            crate::core::add::add(&[self.value.clone(), other.value.clone()])?,
            and_(&[self.condition.clone(), other.condition.clone()]),
        ))
    }

    /// The same value under `condition` as well (`with_condition`).
    pub fn with_condition(&self, condition: &Basic) -> ConditionalValue {
        ConditionalValue::new(self.value.clone(), and_(&[self.condition.clone(), condition.clone()]))
    }

    /// The `(value, condition)` pairs of a value which is a `Piecewise` of
    /// cases whose disjunction is the condition (`cases`).
    pub fn cases(&self) -> Option<Vec<(Basic, Basic)>> {
        if !matches!(self.value.head(), Head::Piecewise) {
            return None;
        }
        let pairs: Vec<(Basic, Basic)> = self
            .value
            .args()
            .iter()
            .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
            .collect();
        let conditions: Vec<Basic> = pairs.iter().map(|(_, c)| c.clone()).collect();
        if pairs.iter().any(|(_, c)| is_true(c)) || self.condition != or_(&conditions) {
            return None;
        }
        Some(pairs)
    }

    /// The value where the condition holds, `otherwise` elsewhere
    /// (`as_piecewise`).
    pub fn as_piecewise(&self, otherwise: &Basic) -> R {
        use crate::functions::elementary::piecewise::piecewise;
        if is_true(&self.condition) {
            return Ok(self.value.clone());
        }
        if let Some(mut branches) = self.cases() {
            branches.push((otherwise.clone(), boolean(true)));
            return piecewise(&branches);
        }
        piecewise(&[
            (self.value.clone(), self.condition.clone()),
            (otherwise.clone(), boolean(true)),
        ])
    }

    /// `repr`.
    pub fn repr(&self) -> String {
        if is_true(&self.condition) {
            format!("ConditionalValue({})", self.value)
        } else {
            format!("ConditionalValue({}, {})", self.value, self.condition)
        }
    }
}

fn conjuncts(condition: &Basic) -> Vec<Basic> {
    if matches!(condition.head(), Head::And) {
        condition.args().to_vec()
    } else {
        vec![condition.clone()]
    }
}

/// `condition` without the conjuncts the assumptions settle; `None` when
/// it is refuted (`decide`).
pub fn decide(condition: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    if is_true(condition) {
        return Ok(Some(boolean(true)));
    }
    if is_false(condition) {
        return Ok(None);
    }
    let mut remaining: Vec<Basic> = Vec::new();
    for part in conjuncts(condition) {
        let part_: Basic = settled(&part, assumptions)?;
        match ask(&part_, assumptions, None)? {
            Some(false) => return Ok(None),
            None => remaining.push(part_),
            Some(true) => {}
        }
    }
    Ok(Some(and_(&remaining)))
}

fn settled(part: &Basic, assumptions: &[Basic]) -> R {
    if matches!(part.head(), Head::Or) {
        let alternatives: Vec<Basic> = part
            .args()
            .iter()
            .map(|a| settled(a, assumptions))
            .collect::<PolyResult<_>>()?;
        if alternatives.iter().any(is_true) {
            return Ok(boolean(true));
        }
        return Ok(or_(&alternatives));
    }
    if matches!(part.head(), Head::And) {
        let parts: Vec<Basic> = part
            .args()
            .iter()
            .map(|a| settled(a, assumptions))
            .collect::<PolyResult<_>>()?;
        let mut kept: Vec<Basic> = Vec::new();
        for c in parts {
            let verdict: Option<bool> = if is_true(&c) {
                Some(true)
            } else {
                ask(&c, assumptions, None)?
            };
            if verdict != Some(true) {
                kept.push(c);
            }
        }
        return Ok(if kept.is_empty() { boolean(true) } else { and_(&kept) });
    }
    canonical(&real_form(part, assumptions)?)
}

fn inequality(b: &Basic) -> Option<RelOp> {
    match b.head() {
        Head::Relational(op) if matches!(op, RelOp::Lt | RelOp::Le | RelOp::Gt | RelOp::Ge) => Some(*op),
        _ => None,
    }
}

fn relation(op: RelOp, lhs: &Basic, rhs: &Basic) -> R {
    match op {
        RelOp::Lt => rl::lt(lhs, rhs),
        RelOp::Le => rl::le(lhs, rhs),
        RelOp::Gt => rl::gt(lhs, rhs),
        RelOp::Ge => rl::ge(lhs, rhs),
        _ => Err(PolyError::Type(format!("expected an inequality, got {:?}", op))),
    }
}

/// The parts of a sum independent of the symbols and the rest
/// (`as_independent(*symbols, as_Add=True)`).
fn independent_terms(e: &Basic, symbols: &[Basic]) -> PolyResult<(Basic, Basic)> {
    let depends = |t: &Basic| t.free_symbols().iter().any(|s| symbols.contains(s));
    if !e.is_add() {
        return Ok(if depends(e) {
            (int(0), e.clone())
        } else {
            (e.clone(), int(0))
        });
    }
    let free: Vec<Basic> = e.args().iter().filter(|a| !depends(a)).cloned().collect();
    let dependent: Vec<Basic> = e.args().iter().filter(|a| depends(a)).cloned().collect();
    Ok((crate::core::add::add(&free)?, crate::core::add::add(&dependent)?))
}

/// An inequality with the constant of its left side moved to the right
/// (`canonical`).
pub fn canonical(condition: &Basic) -> R {
    if let Some(op) = inequality(condition) {
        let lhs: &Basic = &condition.args()[0];
        if lhs.is_add() {
            let syms: Vec<Basic> = lhs.free_symbols().into_iter().collect();
            let (constant, rest) = independent_terms(lhs, &syms)?;
            if constant != int(0) {
                return relation(op, &rest, &sub(&condition.args()[1], &constant)?);
            }
        }
    }
    Ok(condition.clone())
}

fn is_real(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    Ok(is(e, Predicate::extended_real) == Some(true)
        || ask(&element(e, &crate::sets::sets::reals())?, assumptions, None)? == Some(true))
}

fn has_re_im(e: &Basic) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(Function::Re | Function::Im)))
}

fn real_part(e: &Basic, assumptions: &[Basic]) -> R {
    use crate::functions::elementary::complexes::re;
    let part: Basic = re(e)?;
    if !has_re_im(&part) {
        return Ok(part);
    }
    let mut syms: Vec<Basic> = e.free_symbols().into_iter().collect();
    syms.sort_by_key(|s| s.to_string());
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    let mut back: HashMap<Basic, Basic> = HashMap::new();
    for p in syms {
        let real: bool = is(&p, Predicate::extended_real) == Some(true)
            || ask(&element(&p, &crate::sets::sets::reals())?, assumptions, None)? == Some(true);
        if real {
            let name: String = match p.head() {
                Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
                _ => p.to_string(),
            };
            let d: Basic = crate::core::symbol::Dummy::new(&name, &[(Predicate::real, true)]).0;
            replacement.insert(p.clone(), d.clone());
            back.insert(d, p);
        }
    }
    if replacement.is_empty() {
        return Ok(part);
    }
    let substituted: Basic = xreplace_evaluated(e, &replacement)?;
    xreplace_evaluated(&re(&substituted)?, &back)
}

/// A condition `Abs(arg(e)) < c` rewritten for a real `e` (`real_form`).
pub fn real_form(condition: &Basic, assumptions: &[Basic]) -> R {
    let op: Option<RelOp> = inequality(condition);
    let lhs: &Basic = match condition.args().first() {
        Some(l) => l,
        None => return Ok(condition.clone()),
    };
    if !(matches!(op, Some(RelOp::Lt | RelOp::Le))
        && matches!(lhs.head(), Head::Function(Function::Abs))
        && matches!(lhs.args()[0].head(), Head::Function(Function::Arg)))
    {
        return Ok(condition.clone());
    }
    let e: Basic = lhs.args()[0].args()[0].clone();
    let bound: Basic = condition.args()[1].clone();
    let pi: Basic = Basic::pi();
    let bound_positive: bool = is(&bound, Predicate::positive) == Some(true);
    if bound_positive && positive(&e, assumptions)? {
        return Ok(boolean(true));
    }
    let excess: Basic = sub(&bound, &pi)?;
    if bound_positive && (is(&excess, Predicate::positive) == Some(true) || (bound == pi && op == Some(RelOp::Le))) {
        return Ok(boolean(true));
    }
    if bound == pi && e.is_pow() && e.args()[1] == int(2) && !is_real(&e.args()[0], assumptions)? {
        let rp: Basic = real_part(&e.args()[0], assumptions)?;
        if !has_re_im(&rp)
            && (ask(&rl::gt(&rp, &int(0))?, assumptions, None)? == Some(true)
                || ask(&rl::lt(&rp, &int(0))?, assumptions, None)? == Some(true))
        {
            return Ok(boolean(true));
        }
    }
    if bound == crate::core::mul::mul(&[pi.clone(), rational(1, 2)])? && !is_real(&e, assumptions)? {
        let rp: Basic = real_part(&e, assumptions)?;
        if !has_re_im(&rp) {
            match ask(&rl::gt(&rp, &int(0))?, assumptions, None)? {
                Some(true) => return Ok(boolean(true)),
                Some(false) => return Ok(boolean(false)),
                None => {}
            }
        }
    }
    if is_real(&e, assumptions)? {
        if bound_positive && is(&excess, Predicate::positive) == Some(true) {
            return Ok(boolean(true));
        }
        if bound_positive && is(&excess, Predicate::nonpositive) == Some(true) {
            return rl::gt(&e, &int(0));
        }
    }
    Ok(condition.clone())
}

/// Whether `e` is positive under the assumptions, factor by factor
/// (`positive`).
pub fn positive(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if e.is_mul() {
        for f in e.args() {
            if !positive(f, assumptions)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if e.is_pow() {
        if positive(&e.args()[0], assumptions)? {
            return Ok(true);
        }
        return Ok(is(&e.args()[1], Predicate::even) == Some(true) && is_real(&e.args()[0], assumptions)?);
    }
    if matches!(e.head(), Head::Function(Function::Abs)) {
        return Ok(true);
    }
    if is(e, Predicate::positive) == Some(true) {
        return Ok(true);
    }
    if is(e, Predicate::extended_real) == Some(false) {
        return Ok(false);
    }
    Ok(ask(&rl::gt(e, &int(0))?, assumptions, None)? == Some(true))
}

fn name_of(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

/// Each symbol mapped to a plain dummy of the same name (every symbol
/// carries `commutative`, so every one is flagged) (`plain_symbols`).
pub fn plain_symbols(symbols: &[Basic]) -> Vec<(Basic, Basic)> {
    symbols
        .iter()
        .map(|s| (s.clone(), crate::core::symbol::Dummy::new(&name_of(s), &[]).0))
        .collect()
}

fn plain_of(plain: &[(Basic, Basic)], s: &Basic) -> Basic {
    plain
        .iter()
        .find(|(t, _)| t == s)
        .map(|(_, p)| p.clone())
        .unwrap_or_else(|| s.clone())
}

/// The flags of the symbols as statements on their plain counterparts
/// (`symbol_facts`).
pub fn symbol_facts(symbols: &[Basic], plain: &[(Basic, Basic)]) -> PolyResult<Vec<Basic>> {
    use Predicate as P;
    let mut facts: Vec<Basic> = Vec::new();
    for s in symbols {
        let p: Basic = plain_of(plain, s);
        let yes = |k: P| is(s, k) == Some(true);
        if yes(P::positive) {
            facts.push(rl::gt(&p, &int(0))?);
        } else if yes(P::negative) {
            facts.push(rl::lt(&p, &int(0))?);
        } else if yes(P::nonnegative) {
            facts.push(rl::ge(&p, &int(0))?);
        } else if yes(P::nonpositive) {
            facts.push(rl::le(&p, &int(0))?);
        } else if yes(P::nonzero) {
            facts.push(rl::ne(&p, &int(0))?);
        }
        if yes(P::integer) {
            facts.push(element(&p, &Basic::atom(Head::Integers))?);
        } else if yes(P::real) {
            facts.push(element(&p, &crate::sets::sets::reals())?);
        }
    }
    Ok(facts)
}

/// The assumptions on the plain symbols with the facts of the flags
/// (`with_symbol_facts`).
pub fn with_symbol_facts(assumptions: &[Basic], symbols: &[Basic], plain: &[(Basic, Basic)]) -> PolyResult<Vec<Basic>> {
    let mut facts: Vec<Basic> = symbol_facts(symbols, plain)?;
    let rule: HashMap<Basic, Basic> = plain.iter().cloned().collect();
    for a in assumptions {
        let replaced: Basic = a.xreplace(&rule);
        facts.push(if crate::core::logic::is_boolean(&replaced) {
            replaced
        } else {
            a.clone()
        });
    }
    Ok(facts)
}

fn random_value(rng: &mut PyRandom) -> Basic {
    rational(rng.randint(3, 19), 4)
}

fn instances(formula: &Basic, variables: &[Basic]) -> PolyResult<Option<Vec<Values>>> {
    crate::assumptions::sat::find_instance(formula, variables, Some(&crate::sets::sets::reals()), &[], 1)
}

/// Values of the symbols satisfying the assumptions and their flags
/// (`sample_values`); `None` when there is no instance.
pub fn sample_values(symbols: &[Basic], assumptions: &[Basic], rng: &mut PyRandom) -> PolyResult<Option<Values>> {
    if symbols.is_empty() {
        return Ok(Some(vec![]));
    }
    let plain: Vec<(Basic, Basic)> = plain_symbols(symbols);
    let variables: Vec<Basic> = symbols.iter().map(|s| plain_of(&plain, s)).collect();
    let mut facts: Facts = Facts::new(&with_symbol_facts(assumptions, symbols, &plain)?, None, &variables)?;
    if is_true(&facts.formula) {
        return Ok(Some(symbols.iter().map(|s| (s.clone(), random_value(rng))).collect()));
    }
    let mut found: Option<Vec<Values>> = instances(&facts.formula, &variables)?;
    if found.as_ref().map_or(true, |v| v.is_empty()) {
        let rationalised: Vec<Basic> = assumptions.iter().map(rational_relation).collect::<PolyResult<_>>()?;
        if rationalised == assumptions {
            return Ok(None);
        }
        facts = Facts::new(&with_symbol_facts(&rationalised, symbols, &plain)?, None, &variables)?;
        found = instances(&facts.formula, &variables)?;
        if found.as_ref().map_or(true, |v| v.is_empty()) {
            return Ok(None);
        }
    }
    let witness: Values = found.and_then(|v| v.into_iter().next()).unwrap_or_default();
    let mut values: Values = Vec::new();
    for s in symbols {
        let p: Basic = plain_of(&plain, s);
        let v: Basic = match witness.iter().find(|(t, _)| *t == p) {
            Some((_, v)) => v.clone(),
            None => random_value(rng),
        };
        values.push((s.clone(), v));
    }
    for direction in [1i64, -1] {
        for _ in 0..4 {
            let mut moved: Values = Vec::new();
            for (s, v) in &values {
                let step: Basic = rational(direction * rng.randint(1, 9), 8);
                moved.push((s.clone(), crate::core::add::add(&[v.clone(), step])?));
            }
            let rule: HashMap<Basic, Basic> = moved.iter().map(|(s, v)| (plain_of(&plain, s), v.clone())).collect();
            let instance: Basic = xreplace_evaluated(&facts.formula, &rule)?;
            if crate::core::logic::is_boolean(&instance) && ask(&instance, &[], None)? == Some(true) {
                return Ok(Some(moved));
            }
        }
    }
    Ok(Some(values))
}

/// `str(evalf(c, 20))` read back as a fraction: the value rounded to 20
/// significant digits.
fn decimal_20(c: &Basic) -> Option<Rat> {
    let x: Rat = crate::assumptions::analysis::evalf_rational(c, 20)?;
    if x == Rat::ZERO {
        return Some(x);
    }
    let ten = |k: i64| -> Rat {
        let p: Rat = Rat::from(Int::from(10u8).pow(k.unsigned_abs() as usize));
        if k >= 0 {
            p
        } else {
            Rat::ONE / p
        }
    };
    let magnitude: Rat = if crate::numbers::rat_is_negative(&x) {
        -x.clone()
    } else {
        x.clone()
    };
    let mut e: i64 = 0;
    while magnitude >= ten(e) {
        e += 1;
    }
    while magnitude < ten(e - 1) {
        e -= 1;
    }
    let scaled: Rat = &x * ten(20 - e);
    // the nearest integer, ties to even
    let (n, d) = (
        crate::numbers::numer(&scaled),
        Int::from(crate::numbers::denom(&scaled)),
    );
    let two: Int = Int::from(2u8);
    let mut q: Int = &n / &d;
    let mut r: Int = &n - &q * &d;
    if r < Int::ZERO {
        q -= Int::ONE;
        r += &d;
    }
    let twice: Int = &r * &two;
    if twice > d || (twice == d && (&q % &two) != Int::ZERO) {
        q += Int::ONE;
    }
    Some(Rat::from(q) / ten(20 - e))
}

/// An inequality with an irrational constant term replaced by a rational
/// one on the safe side (`_rational_relation`).
fn rational_relation(rel: &Basic) -> R {
    let Some(op) = inequality(rel) else {
        return Ok(rel.clone());
    };
    let expression: Basic = sub(&rel.args()[0], &rel.args()[1])?;
    let mut variables: Vec<Basic> = expression.free_symbols().into_iter().collect();
    variables.sort_by_key(|s| s.to_string());
    let (constant, rest) = independent_terms(&expression, &variables)?;
    if is(&constant, Predicate::rational) != Some(false) || is(&constant, Predicate::real) != Some(true) {
        return Ok(rel.clone());
    }
    match crate::polys::polytools::Poly::new(&rest, &variables) {
        Ok(p) => {
            if !p.coeffs()?.iter().all(|c| is(c, Predicate::rational) == Some(true)) {
                return Ok(rel.clone());
            }
        }
        Err(PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_)) => return Ok(rel.clone()),
        Err(e) => return Err(e),
    }
    let Some(approximation) = decimal_20(&constant) else {
        return Ok(rel.clone());
    };
    let margin: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(10u8).pow(12));
    let shifted: Rat = if matches!(op, RelOp::Gt | RelOp::Ge) {
        approximation - margin
    } else {
        approximation + margin
    };
    relation(op, &rest, &Basic::from_rational(-shifted))
}

fn samples_verdict(
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    condition: &Basic,
    samples: usize,
    patient: bool,
) -> PolyResult<Option<bool>> {
    let mut symbols: Vec<Basic> = a.free_symbols().into_iter().chain(b.free_symbols()).collect();
    symbols.sort_by_key(|s| s.to_string());
    symbols.dedup();
    let mut rng: PyRandom = PyRandom::from_string(&format!("({}, {})", a, b));
    let mut verdict: Option<bool> = Some(true);
    let rounds: usize = if symbols.is_empty() { 1 } else { samples };
    for _ in 0..rounds {
        let values: Option<Values> = sample_under(&symbols, assumptions, condition, &mut rng)?;
        let outcome: Option<bool> = match &values {
            None => None,
            Some(v) => agree_at(a, b, v),
        };
        if outcome == Some(false) {
            return Ok(Some(false));
        }
        if outcome.is_none() {
            if !patient || values.is_none() {
                return Ok(None);
            }
            verdict = None;
        }
    }
    Ok(verdict)
}

fn sample_under(
    symbols: &[Basic],
    assumptions: &[Basic],
    condition: &Basic,
    rng: &mut PyRandom,
) -> PolyResult<Option<Values>> {
    if is_true(condition) {
        return sample_values(symbols, assumptions, rng);
    }
    if is_false(condition) {
        return Ok(None);
    }
    let mut together: Vec<Basic> = vec![condition.clone()];
    together.extend(assumptions.iter().cloned());
    if let Some(v) = sample_values(symbols, &together, rng)? {
        return Ok(Some(v));
    }
    for _ in 0..4 {
        let Some(values) = sample_values(symbols, assumptions, rng)? else {
            return Ok(None);
        };
        let rule: HashMap<Basic, Basic> = values.iter().cloned().collect();
        if is_true(&xreplace_evaluated(condition, &rule)?) {
            return Ok(Some(values));
        }
    }
    Ok(None)
}

/// Whether `a` and `b` have the same value at the sample (`_agree_at`).
fn agree_at(a: &Basic, b: &Basic, values: &Values) -> Option<bool> {
    use crate::core::evalf::{parts, Real};
    let left = crate::numeric::reliable_value(a, 30, values)?;
    let right = crate::numeric::reliable_value(b, 30, values)?;
    let prec: usize = 30 * 4 + 20;
    let at = |x: &Real| -> Real { x.clone().with_precision(prec).value() };
    let ((lr, li), (rr, ri)) = (parts(&left), parts(&right));
    let (lr, li, rr, ri) = (at(&lr), at(&li), at(&rr), at(&ri));
    let modulus = |x: &Real, y: &Real| -> Real { (x * x + y * y).sqrt() };
    let gap: Real = modulus(&(&lr - &rr), &(&li - &ri));
    let one: Real = crate::core::evalf::real_rat(&Rat::ONE, prec);
    let scale: Real = one + modulus(&lr, &li) + modulus(&rr, &ri);
    let bound: Real = scale / crate::core::evalf::real_rat(&Rat::from(Int::from(10u8).pow(12)), prec);
    Some(gap <= bound)
}

/// Whether `a` and `b` agree at `samples` points of the region of the
/// assumptions (`numerically_equal`).
pub fn numerically_equal(a: &Basic, b: &Basic, assumptions: &[Basic], samples: usize) -> PolyResult<bool> {
    if samples_verdict(a, b, assumptions, &boolean(true), samples, false)? != Some(true) {
        return Ok(false);
    }
    Ok(!provably_different(a, b, assumptions)?)
}

/// Whether the structure theorem proves `a - b` not identically zero
/// (`_provably_different`).
fn provably_different(a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    use crate::simplify::structure::{is_not_elementary, Tower};
    let difference: Basic = sub(a, b)?;
    if difference == int(0) || crate::core::function::count_ops(&difference)? > 120 {
        return Ok(false);
    }
    let decided = crate::timeout::attempt_within(Some(1.0), || {
        let mut tower: Tower = Tower::new(assumptions);
        let value = match tower.element(&difference) {
            Ok(v) => v,
            Err(e) if is_not_elementary(&e) || matches!(e, PolyError::ZeroDivision(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        Ok(!tower.vanishes(&value)? && tower.certifies(&value)?)
    })?;
    Ok(decided == Some(true))
}

/// The verdict of a numerical comparison (`numerical_verdict`): `False`
/// when the values differ at a sample, `True` when they agree at every
/// one, `None` when nothing could be checked.
pub fn numerical_verdict(
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    condition: &Basic,
    samples: usize,
) -> PolyResult<Option<bool>> {
    samples_verdict(a, b, assumptions, condition, samples, true)
}

/// The sum of the values, holding when every condition does (`combine`).
pub fn combine(values: &[ConditionalValue]) -> PolyResult<ConditionalValue> {
    let mut total: ConditionalValue = ConditionalValue::new(int(0), boolean(true));
    for v in values {
        total = total.add(v)?;
    }
    Ok(total)
}
