//! Port of `sympy_extras/assumptions/refine.py`, the refinement part: an
//! expression refined under assumptions (the counterpart of Mathematica's
//! `Refine`). The symbols and the subexpressions whose sign or nature
//! matters are decided with `ask` and replaced by dummies carrying the
//! assumptions of the core, so that its evaluation and its `refine` apply;
//! relations, absolute values, signs, maxima and minima, `atan2`, powers
//! and logarithms of polynomials with factors of known sign and the
//! conditions of `Piecewise` are rewritten here. `simplify` (with the
//! Boolean simplification) is not ported yet.

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use crate::assumptions::ask::{evaluate, evaluate_atom, facts_of, Truth};
use crate::assumptions::facts::{element, normalize, Facts};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::core::function::{rebuild_evaluated, replace_where, xreplace_evaluated};
use crate::core::logic::{boolean, is_boolean, not_};
use crate::core::relational as rl;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;

const MAX_PASSES: usize = 4;

/// The functions whose arguments are abstracted (`_ARGUMENTS`).
const ARGUMENTS: &[Function] = &[
    Function::Abs,
    Function::Sign,
    Function::Log,
    Function::Arg,
    Function::Re,
    Function::Im,
    Function::Conjugate,
    Function::Floor,
    Function::Ceiling,
    Function::Frac,
    Function::Atan2,
    Function::Gamma,
    Function::LogGamma,
    Function::Factorial,
    Function::Asin,
    Function::Acos,
];

fn size(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

fn symbols(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect()
}

fn sorted_symbols(e: &Basic) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols(e).into_iter().collect();
    v.sort_by_key(|s| s.to_string());
    v
}

fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}

fn int(n: i64) -> Basic {
    Basic::int(n)
}

fn is_expr(e: &Basic) -> bool {
    !(is_boolean(e) || crate::sets::sets::is_set(e) || matches!(e.head(), Head::Tuple | Head::ExprCondPair))
}

/// Whether the error is one the Python catches as `ValueError`,
/// `TypeError` or `NotImplementedError` (not a part not ported).
fn caught(e: &PolyError, not_implemented: bool) -> bool {
    match e {
        PolyError::Value(_) | PolyError::Type(_) => true,
        PolyError::NotImplemented(m) => not_implemented && !m.starts_with("not ported"),
        _ => false,
    }
}

/// The refinement under fixed facts with a cache of the decided
/// statements (`_Refiner`).
pub struct Refiner {
    pub facts: Facts,
    cache: RefCell<HashMap<Basic, Truth>>,
    dummies: RefCell<Vec<(Basic, Basic)>>,
}

type Properties = Vec<(Predicate, bool)>;

impl Refiner {
    pub fn new(facts: Facts) -> Refiner {
        Refiner {
            facts,
            cache: RefCell::new(HashMap::new()),
            dummies: RefCell::new(Vec::new()),
        }
    }

    /// The three-valued truth of a statement (`holds`).
    pub fn holds(&self, statement: &Basic) -> PolyResult<Truth> {
        let formula: Basic = normalize(statement)?;
        if let Some(v) = self.cache.borrow().get(&formula) {
            return Ok(*v);
        }
        let v: Truth = evaluate(&formula, &self.facts)?;
        self.cache.borrow_mut().insert(formula, v);
        Ok(v)
    }

    fn true_(&self, statement: &Basic) -> PolyResult<bool> {
        Ok(self.holds(statement)? == Some(true))
    }

    fn is_real(&self, f: &Basic) -> PolyResult<bool> {
        if f.is_symbol() {
            return Ok(self.facts.real.contains(f));
        }
        self.true_(&element(f, &crate::sets::sets::reals())?)
    }

    fn q(&self, name: &str, f: &Basic) -> Basic {
        crate::core::assume::applied(&crate::core::assume::q(name, 1), std::slice::from_ref(f))
    }

    /// The assumption keys of the core which follow from the facts
    /// (`properties`).
    fn properties(&self, f: &Basic) -> PolyResult<Properties> {
        use Predicate as P;
        let mut props: Properties = Vec::new();
        if !self.is_real(f)? {
            return Ok(props);
        }
        props.push((P::real, true));
        if self.true_(&rl::gt(f, &int(0))?)? {
            props.push((P::positive, true));
        } else if self.true_(&rl::lt(f, &int(0))?)? {
            props.push((P::negative, true));
        } else {
            let nonnegative: bool = self.true_(&rl::ge(f, &int(0))?)?;
            let nonpositive: bool = self.true_(&rl::le(f, &int(0))?)?;
            let mut zero: bool = false;
            if nonnegative && nonpositive {
                props.push((P::zero, true));
                zero = true;
            } else if nonnegative {
                props.push((P::nonnegative, true));
            } else if nonpositive {
                props.push((P::nonpositive, true));
            }
            if !zero && self.true_(&rl::ne(f, &int(0))?)? {
                props.push((P::nonzero, true));
            }
        }
        let integer: bool = f.is_symbol() && self.facts.integer.contains(f);
        if integer || self.true_(&element(f, &Basic::atom(Head::Integers))?)? {
            props.push((P::integer, true));
            let mut parity: Option<i64> = self.residue(f, 2)?;
            if parity.is_none() {
                parity = if self.true_(&self.q("even", f))? {
                    Some(0)
                } else if self.true_(&self.q("odd", f))? {
                    Some(1)
                } else {
                    None
                };
            }
            match parity {
                Some(0) => props.push((P::even, true)),
                Some(1) => props.push((P::odd, true)),
                _ => {}
            }
            if f.is_symbol() && self.true_(&self.q("prime", f))? {
                props.push((P::prime, true));
            }
        } else if self.true_(&element(f, &Basic::atom(Head::Rationals))?)? {
            props.push((P::rational, true));
        }
        Ok(props)
    }

    /// The residue of an integer polynomial in integer variables modulo
    /// `m` when it is the same at every point (`residue`).
    fn residue(&self, f: &Basic, m: i64) -> PolyResult<Option<i64>> {
        let syms: Vec<Basic> = sorted_symbols(f);
        if syms.is_empty() || !syms.iter().all(|s| self.facts.integer.contains(s)) {
            return Ok(None);
        }
        let poly = match crate::polys::polytools::Poly::new(f, &syms) {
            Ok(p) => p,
            Err(PolyError::Polys(..) | PolyError::Value(_)) => return Ok(None),
            Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => return Ok(None),
            Err(e) => return Err(e),
        };
        if !poly.coeffs()?.iter().all(|c| c.is_integer_value()) {
            return Ok(None);
        }
        if syms.len() as i64 * m > 64 {
            return Ok(None);
        }
        let polynomial: Basic = poly.as_expr()?;
        let mut found: Option<i64> = None;
        let k: usize = syms.len();
        let total: i64 = m.pow(k as u32);
        for index in 0..total {
            // the points of product(range(m), repeat=k), the last coordinate fastest
            let mut point: Vec<i64> = vec![0; k];
            let mut rest: i64 = index;
            for j in (0..k).rev() {
                point[j] = rest % m;
                rest /= m;
            }
            let at: HashMap<Basic, Basic> = syms.iter().cloned().zip(point.iter().map(|v| int(*v))).collect();
            let value: Basic = xreplace_evaluated(&polynomial, &at)?;
            let n: i64 = value
                .as_rational()
                .and_then(|r| i64::try_from(&crate::numbers::numer(r)).ok())
                .ok_or_else(|| PolyError::Type(format!("an integer is expected, got {}", value)))?;
            let v: i64 = n.rem_euclid(m);
            match found {
                None => found = Some(v),
                Some(w) if w != v => return Ok(None),
                _ => {}
            }
        }
        Ok(found)
    }

    fn dummy(&self, f: &Basic, props: &Properties) -> Option<Basic> {
        if let Some((_, d)) = self.dummies.borrow().iter().find(|(g, _)| g == f) {
            return Some(d.clone());
        }
        let mut merged: Vec<(Predicate, bool)> = Vec::new();
        let mut name: String = "e".to_string();
        if let Head::Symbol(data) | Head::Dummy(data, _) = f.head() {
            name = data.name.clone();
            merged.extend(data.assumptions.iter().map(|(p, v)| (*p, *v)));
        }
        for (p, v) in props {
            merged.retain(|(q, _)| q != p);
            merged.push((*p, *v));
        }
        if !crate::core::assumptions::is_consistent(&merged) {
            return None;
        }
        let d: Basic = crate::core::symbol::Dummy::new(&name, &merged).0;
        self.dummies.borrow_mut().push((f.clone(), d.clone()));
        Some(d)
    }

    fn candidates(&self, expr: &Basic) -> Vec<Basic> {
        fn add(e: &Basic, found: &mut Vec<Basic>, seen: &mut BTreeSet<Basic>) {
            if is_expr(e) && !e.is_atom() && !seen.contains(e) {
                seen.insert(e.clone());
                found.push(e.clone());
                if e.is_mul() {
                    for factor in e.args() {
                        add(factor, found, seen);
                    }
                }
            }
        }
        let mut found: Vec<Basic> = Vec::new();
        let mut seen: BTreeSet<Basic> = BTreeSet::new();
        for node in expr.preorder() {
            if node.is_pow() {
                let (base, exponent) = (&node.args()[0], &node.args()[1]);
                if !exponent.is_integer_value() {
                    add(base, &mut found, &mut seen);
                }
                if base.is_number_class() && is(base, Predicate::positive) != Some(true) {
                    add(exponent, &mut found, &mut seen);
                }
            } else if func(&node).is_some_and(|f| ARGUMENTS.contains(&f)) {
                for a in node.args() {
                    add(a, &mut found, &mut seen);
                }
            }
        }
        found
    }

    /// `expr` with its symbols and relevant subexpressions replaced by
    /// dummies, and the substitutions undoing it (`abstract`).
    fn abstract_(&self, expr: &Basic) -> PolyResult<(Basic, HashMap<Basic, Basic>, HashMap<Basic, Basic>)> {
        let mut syms: HashMap<Basic, Basic> = HashMap::new();
        for s in sorted_symbols(expr) {
            let props: Properties = self.properties(&s)?;
            if !props.is_empty() {
                if let Some(d) = self.dummy(&s, &props) {
                    syms.insert(s, d);
                }
            }
        }
        let mut sub: Vec<(Basic, Basic)> = Vec::new();
        for f in self.candidates(expr) {
            let props: Properties = self.properties(&f)?;
            if props.is_empty() {
                continue;
            }
            let g: Basic = xreplace_evaluated(&f, &syms)?;
            if props.iter().all(|(p, v)| is(&g, *p) == Some(*v)) {
                continue;
            }
            if let Some(d) = self.dummy(&f, &props) {
                sub.push((f, d));
            }
        }
        let mut abstracted: Basic = xreplace_evaluated(expr, &syms)?;
        let mut back_sub: HashMap<Basic, Basic> = HashMap::new();
        if !sub.is_empty() {
            let rule: HashMap<Basic, Basic> = sub.iter().cloned().collect();
            abstracted = xreplace_evaluated(&xreplace_evaluated(expr, &rule)?, &syms)?;
            for (f, d) in &sub {
                back_sub.insert(d.clone(), xreplace_evaluated(f, &syms)?);
            }
        }
        let back_symbols: HashMap<Basic, Basic> = syms.iter().map(|(s, d)| (d.clone(), s.clone())).collect();
        Ok((abstracted, back_sub, back_symbols))
    }

    fn through(&self, expr: &Basic, operation: &dyn Fn(&Basic) -> R) -> R {
        let (abstracted, back_sub, back_symbols) = self.abstract_(expr)?;
        let mut result: Basic = operation(&abstracted)?;
        if !back_sub.is_empty() {
            result = xreplace_evaluated(&result, &back_sub)?;
        }
        xreplace_evaluated(&result, &back_symbols)
    }

    fn core_refine(&self, expr: &Basic) -> R {
        self.through(expr, &|e: &Basic| {
            let rule: HashMap<Basic, Basic> = self
                .dummies
                .borrow()
                .iter()
                .filter(|(s, _)| s.is_symbol())
                .cloned()
                .collect();
            let predicates: Basic = xreplace_evaluated(&self.facts.predicates, &rule)?;
            refine_around_booleans(e, &predicates)
        })
    }

    fn polynomial_factors(&self, f: &Basic) -> PolyResult<Option<(Basic, Vec<(Basic, i64)>)>> {
        let syms: BTreeSet<Basic> = symbols(f);
        if syms.is_empty() || !syms.iter().all(|s| self.facts.real.contains(s)) {
            return Ok(None);
        }
        let gens: Vec<Basic> = sorted_symbols(f);
        if crate::core::expr::polynomial_like(f, &gens, crate::core::expr::PolynomialKind::Polynomial) != Some(true) {
            return Ok(None);
        }
        match crate::polys::polytools::factor_list(f, &gens) {
            Ok((c, factors)) => {
                let mut out: Vec<(Basic, i64)> = Vec::new();
                for (g, m) in factors {
                    let k: i64 = m
                        .as_rational()
                        .and_then(|r| i64::try_from(&crate::numbers::numer(r)).ok())
                        .unwrap_or(1);
                    out.push((g, k));
                }
                Ok(Some((c, out)))
            }
            Err(PolyError::Polys(..) | PolyError::Value(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn better(&self, new: Basic, old: &Basic) -> R {
        Ok(if size(&new)? <= size(old)? { new } else { old.clone() })
    }

    fn refine_abs(&self, e: &Basic) -> R {
        use crate::core::power::pow;
        let f: &Basic = &e.args()[0];
        if self.true_(&rl::ge(f, &int(0))?)? {
            return Ok(f.clone());
        }
        if self.true_(&rl::le(f, &int(0))?)? {
            return crate::core::mul::neg(f);
        }
        let Some((c, factors)) = self.polynomial_factors(f)? else {
            return Ok(e.clone());
        };
        let mut parts: Vec<Basic> = vec![crate::functions::elementary::complexes::abs_(&c)?];
        for (g, m) in factors {
            let mm: Basic = int(m);
            if self.true_(&rl::ge(&g, &int(0))?)? {
                parts.push(pow(&g, &mm)?);
            } else if self.true_(&rl::le(&g, &int(0))?)? {
                parts.push(pow(&crate::core::mul::neg(&g)?, &mm)?);
            } else if m % 2 == 0 {
                parts.push(pow(&g, &mm)?);
            } else {
                parts.push(pow(&crate::functions::elementary::complexes::abs_(&g)?, &mm)?);
            }
        }
        self.better(crate::core::mul::mul(&parts)?, e)
    }

    fn refine_sign(&self, e: &Basic) -> R {
        let f: &Basic = &e.args()[0];
        if self.true_(&rl::gt(f, &int(0))?)? {
            return Ok(int(1));
        }
        if self.true_(&rl::lt(f, &int(0))?)? {
            return Ok(int(-1));
        }
        if self.true_(&rl::eq(f, &int(0))?)? {
            return Ok(int(0));
        }
        Ok(e.clone())
    }

    fn refine_minmax(&self, e: &Basic) -> R {
        let is_max: bool = func(e) == Some(Function::Max);
        let args: &[Basic] = e.args();
        let mut dominated: BTreeSet<usize> = BTreeSet::new();
        for (i, a) in args.iter().enumerate() {
            for (j, b) in args.iter().enumerate() {
                if i == j || dominated.contains(&j) {
                    continue;
                }
                let better: Basic = if is_max { rl::ge(b, a)? } else { rl::le(b, a)? };
                if self.true_(&better)? {
                    dominated.insert(i);
                    break;
                }
            }
        }
        if dominated.is_empty() {
            return Ok(e.clone());
        }
        let remaining: Vec<Basic> = args
            .iter()
            .enumerate()
            .filter(|(i, _)| !dominated.contains(i))
            .map(|(_, a)| a.clone())
            .collect();
        use crate::functions::elementary::miscellaneous::{max_, min_};
        if is_max {
            max_(&remaining)
        } else {
            min_(&remaining)
        }
    }

    fn refine_floor(&self, e: &Basic) -> R {
        let f: &Basic = &e.args()[0];
        if self.true_(&element(f, &Basic::atom(Head::Integers))?)? {
            return Ok(if func(e) == Some(Function::Frac) {
                int(0)
            } else {
                f.clone()
            });
        }
        Ok(e.clone())
    }

    fn refine_real_part(&self, e: &Basic) -> R {
        let f: &Basic = &e.args()[0];
        if self.is_real(f)? {
            return Ok(if func(e) == Some(Function::Im) {
                int(0)
            } else {
                f.clone()
            });
        }
        Ok(e.clone())
    }

    fn refine_arg(&self, e: &Basic) -> R {
        let f: &Basic = &e.args()[0];
        if self.true_(&rl::gt(f, &int(0))?)? {
            return Ok(int(0));
        }
        if self.true_(&rl::lt(f, &int(0))?)? {
            return Ok(Basic::pi());
        }
        Ok(e.clone())
    }

    fn refine_atan2(&self, e: &Basic) -> R {
        use crate::core::add::{add, sub};
        let (y, x) = (&e.args()[0], &e.args()[1]);
        let at = || crate::functions::elementary::inverse::atan(&crate::core::mul::div(y, x)?);
        if self.true_(&rl::gt(x, &int(0))?)? {
            return at();
        }
        if self.true_(&rl::lt(x, &int(0))?)? {
            if self.true_(&rl::ge(y, &int(0))?)? {
                return add(&[at()?, Basic::pi()]);
            }
            if self.true_(&rl::lt(y, &int(0))?)? {
                return sub(&at()?, &Basic::pi());
            }
        }
        if self.true_(&rl::eq(x, &int(0))?)? {
            let half: Basic = crate::core::mul::div(&Basic::pi(), &int(2))?;
            if self.true_(&rl::gt(y, &int(0))?)? {
                return Ok(half);
            }
            if self.true_(&rl::lt(y, &int(0))?)? {
                return crate::core::mul::neg(&half);
            }
        }
        Ok(e.clone())
    }

    fn refine_pow(&self, e: &Basic) -> R {
        use crate::core::mul::{mul, neg};
        use crate::core::power::pow;
        let (base, exponent) = (&e.args()[0], &e.args()[1]);
        if exponent.is_integer_value() {
            return Ok(e.clone());
        }
        let Some((mut c, factors)) = self.polynomial_factors(base)? else {
            return Ok(e.clone());
        };
        let mut known: Vec<Basic> = Vec::new();
        let mut unknown: Vec<Basic> = Vec::new();
        for (g, m) in factors {
            let me: Basic = mul(&[int(m), exponent.clone()])?;
            if self.true_(&rl::ge(&g, &int(0))?)? {
                known.push(pow(&g, &me)?);
            } else if self.true_(&rl::le(&g, &int(0))?)? {
                known.push(pow(&neg(&g)?, &me)?);
                if m % 2 == 1 {
                    c = neg(&c)?;
                }
            } else if m % 2 == 0 {
                known.push(pow(&crate::functions::elementary::complexes::abs_(&g)?, &me)?);
            } else {
                unknown.push(pow(&g, &int(m))?);
            }
        }
        if known.is_empty() {
            return Ok(e.clone());
        }
        if is(&c, Predicate::negative) == Some(true) {
            unknown.push(c);
        } else {
            known.push(pow(&c, exponent)?);
        }
        if !unknown.is_empty() {
            known.push(pow(&mul(&unknown)?, exponent)?);
        }
        self.better(mul(&known)?, e)
    }

    fn refine_log(&self, e: &Basic) -> R {
        use crate::functions::elementary::exponential::log;
        let f: &Basic = &e.args()[0];
        if f.is_pow() {
            let (base, exponent) = (&f.args()[0], &f.args()[1]);
            if self.true_(&rl::gt(base, &int(0))?)? && self.is_real(exponent)? {
                return crate::core::mul::mul(&[exponent.clone(), log(base)?]);
            }
            if self.true_(&rl::lt(base, &int(0))?)?
                && exponent.is_integer_value()
                && is(exponent, Predicate::even) == Some(true)
            {
                return crate::core::mul::mul(&[exponent.clone(), log(&crate::core::mul::neg(base)?)?]);
            }
            return Ok(e.clone());
        }
        let Some((c, factors)) = self.polynomial_factors(f)? else {
            return Ok(e.clone());
        };
        if is(&c, Predicate::positive) != Some(true) {
            return Ok(e.clone());
        }
        for (g, _) in &factors {
            if !self.true_(&rl::gt(g, &int(0))?)? {
                return Ok(e.clone());
            }
        }
        let mut total: Basic = int(0);
        for (g, m) in &factors {
            total = crate::core::add::add(&[total, crate::core::mul::mul(&[int(*m), log(g)?])?])?;
        }
        if c != int(1) {
            total = crate::core::add::add(&[total, log(&c)?])?;
        }
        self.better(total, e)
    }

    fn refine_mod(&self, e: &Basic) -> R {
        let (f, m) = (&e.args()[0], &e.args()[1]);
        let Some(mv) = m
            .as_rational()
            .filter(|_| m.is_integer_value())
            .and_then(|r| i64::try_from(&crate::numbers::numer(r)).ok())
        else {
            return Ok(e.clone());
        };
        if mv <= 0 {
            return Ok(e.clone());
        }
        Ok(match self.residue(f, mv)? {
            Some(r) => int(r),
            None => e.clone(),
        })
    }

    fn refine_piecewise(&self, e: &Basic) -> R {
        let mut pairs: Vec<(Basic, Basic)> = Vec::new();
        let mut excluded: Vec<Basic> = Vec::new();
        for branch in e.args() {
            let (expr, cond) = (&branch.args()[0], &branch.args()[1]);
            let value: Truth = self.holds(cond)?;
            if value == Some(false) {
                excluded.push(cond.clone());
                continue;
            }
            let mut extra: Vec<Basic> = excluded.iter().map(not_).collect();
            if value.is_none() {
                extra.push(cond.clone());
            }
            let mut items: Vec<Basic> = self.facts.conjuncts.clone();
            items.extend(extra);
            let refined: Basic = match Facts::new(&items, None, &[]) {
                Ok(facts) => Refiner::new(facts).refine_expr(expr)?,
                Err(err) if caught(&err, false) => self.refine_expr(expr)?,
                Err(err) => return Err(err),
            };
            excluded.push(cond.clone());
            pairs.push((
                refined,
                if value == Some(true) {
                    boolean(true)
                } else {
                    cond.clone()
                },
            ));
            if value == Some(true) {
                break;
            }
        }
        if pairs.is_empty() {
            return Ok(Basic::nan());
        }
        crate::functions::elementary::piecewise::piecewise(&pairs)
    }

    fn refine_relational(&self, e: &Basic) -> R {
        Ok(match self.holds(e)? {
            None => e.clone(),
            Some(v) => boolean(v),
        })
    }

    fn handler(&self, e: &Basic) -> R {
        match e.head() {
            Head::Relational(_) => self.refine_relational(e),
            Head::Pow => self.refine_pow(e),
            Head::Piecewise => self.refine_piecewise(e),
            Head::Function(f) => match f {
                Function::Abs => self.refine_abs(e),
                Function::Sign => self.refine_sign(e),
                Function::Max | Function::Min => self.refine_minmax(e),
                Function::Floor | Function::Ceiling | Function::Frac => self.refine_floor(e),
                Function::Re | Function::Im | Function::Conjugate => self.refine_real_part(e),
                Function::Arg => self.refine_arg(e),
                Function::Atan2 => self.refine_atan2(e),
                Function::Log => self.refine_log(e),
                Function::Mod => self.refine_mod(e),
                _ => Ok(e.clone()),
            },
            _ => Ok(e.clone()),
        }
    }

    fn handlers_pass(&self, expr: &Basic) -> R {
        // the classes of `_HANDLERS`, one replace each, in order
        type Query = fn(&Basic) -> bool;
        let classes: [Query; 12] = [
            |e| matches!(e.head(), Head::Relational(_)),
            |e| e.is_pow(),
            |e| func(e) == Some(Function::Abs),
            |e| func(e) == Some(Function::Sign),
            |e| matches!(func(e), Some(Function::Max | Function::Min)),
            |e| matches!(func(e), Some(Function::Floor | Function::Ceiling | Function::Frac)),
            |e| matches!(func(e), Some(Function::Re | Function::Im | Function::Conjugate)),
            |e| func(e) == Some(Function::Arg),
            |e| func(e) == Some(Function::Atan2),
            |e| func(e) == Some(Function::Log),
            |e| matches!(e.head(), Head::Piecewise),
            |e| func(e) == Some(Function::Mod),
        ];
        let mut e: Basic = expr.clone();
        for query in classes {
            e = replace_where(&e, &query, &|n: &Basic| self.handler(n))?;
        }
        Ok(e)
    }

    /// A non-Boolean expression refined to a fixpoint (`refine_expr`).
    pub fn refine_expr(&self, expr: &Basic) -> R {
        let mut e: Basic = expr.clone();
        for _ in 0..MAX_PASSES {
            let mut new: Basic = self.handlers_pass(&e)?;
            new = self.core_refine(&new)?;
            if new == e {
                break;
            }
            e = new;
        }
        Ok(e)
    }

    /// The atoms of a Boolean formula decided by the facts replaced
    /// (`refine_boolean`).
    pub fn refine_boolean(&self, formula: &Basic) -> R {
        let formula: Basic = normalize(formula)?;
        if let Some(v) = self.holds(&formula)? {
            return Ok(boolean(v));
        }
        let mut decided: HashMap<Basic, Basic> = HashMap::new();
        for atom in formula.preorder() {
            if !matches!(
                atom.head(),
                Head::Relational(_) | Head::Contains | Head::AppliedPredicate
            ) || decided.contains_key(&atom)
            {
                continue;
            }
            if let Some(v) = evaluate_atom(&atom, &self.facts)? {
                decided.insert(atom.clone(), boolean(v));
            }
        }
        if decided.is_empty() {
            return Ok(formula);
        }
        xreplace_evaluated(&formula, &decided)
    }
}

fn has_boolean(e: &Basic) -> bool {
    e.preorder().iter().any(is_boolean)
}

fn refine_around_booleans(expr: &Basic, predicates: &Basic) -> R {
    if is_boolean(expr) {
        return Ok(expr.clone());
    }
    if has_boolean(expr) {
        let args: Vec<Basic> = expr
            .args()
            .iter()
            .map(|a| refine_around_booleans(a, predicates))
            .collect::<PolyResult<_>>()?;
        return rebuild_evaluated(expr, args);
    }
    match crate::core::refine::refine(expr, predicates) {
        Ok(v) => Ok(v),
        Err(e) if caught(&e, true) => Ok(expr.clone()),
        Err(e) => Err(e),
    }
}

/// The atoms of a formula decided by the facts replaced
/// (`_refine_boolean`, used by `resolve`).
pub fn refine_boolean_under(formula: &Basic, facts: &Facts) -> R {
    Refiner::new(facts.clone()).refine_boolean(formula)
}

/// An expression refined under assumptions (`refine`).
pub fn refine(expr: &Basic, assumptions: &[Basic], domain: Option<&Basic>) -> R {
    let syms: Vec<Basic> = expr.free_symbols().into_iter().collect();
    let facts: Facts = facts_of(assumptions, domain, &syms)?;
    let refiner: Refiner = Refiner::new(facts);
    if is_boolean(expr) {
        let mut formula: Basic = expr.clone();
        if formula
            .preorder()
            .iter()
            .any(crate::assumptions::quantifiers::is_quantifier)
        {
            formula = crate::assumptions::resolve::resolve(&formula, domain, assumptions, None)?;
            if matches!(formula.head(), Head::BooleanTrue | Head::BooleanFalse) {
                return Ok(formula);
            }
        }
        return refiner.refine_boolean(&formula);
    }
    refiner.refine_expr(expr)
}

// ---------------------------------------------------------------------
// the simplification under assumptions

// PORTED FROM SYMPY: sympy/polys/matrices/linsolve.py:_lin_eq2dict
/// The constant term and the coefficients of the symbols of a linear
/// expression, without expanding it; `None` when it is not linear
/// (`_lin_eq2dict`, `_NonlinearError`).
pub(crate) fn lin_eq2dict(a: &Basic, symset: &[Basic]) -> PolyResult<Option<(Basic, Vec<(Basic, Basic)>)>> {
    use crate::core::mul::mul;
    if a.is_symbol() && symset.contains(a) {
        return Ok(Some((int(0), vec![(a.clone(), int(1))])));
    }
    if a.is_add() {
        let mut terms: Vec<(Basic, Vec<Basic>)> = Vec::new();
        let mut coeffs: Vec<Basic> = Vec::new();
        for ai in a.args() {
            let Some((ci, ti)) = lin_eq2dict(ai, symset)? else {
                return Ok(None);
            };
            coeffs.push(ci);
            for (m, c) in ti {
                match terms.iter_mut().find(|(s, _)| *s == m) {
                    Some((_, v)) => v.push(c),
                    None => terms.push((m, vec![c])),
                }
            }
        }
        let mut out: Vec<(Basic, Basic)> = Vec::new();
        for (s, cs) in terms {
            out.push((s, crate::core::add::add(&cs)?));
        }
        return Ok(Some((crate::core::add::add(&coeffs)?, out)));
    }
    if a.is_mul() {
        let mut terms: Option<Vec<(Basic, Basic)>> = None;
        let mut terms_coeff: Basic = int(1);
        let mut factors: Vec<Basic> = Vec::new();
        for ai in a.args() {
            let Some((ci, ti)) = lin_eq2dict(ai, symset)? else {
                return Ok(None);
            };
            if ti.is_empty() {
                factors.push(ci);
            } else if terms.is_none() {
                terms = Some(ti);
                terms_coeff = ci;
            } else {
                return Ok(None);
            }
        }
        let coeff: Basic = mul(&factors)?;
        let Some(terms) = terms else {
            return Ok(Some((coeff, vec![])));
        };
        let mut out: Vec<(Basic, Basic)> = Vec::new();
        for (s, c) in terms {
            out.push((s, mul(&[coeff.clone(), c])?));
        }
        return Ok(Some((mul(&[coeff, terms_coeff])?, out)));
    }
    if !a.free_symbols().iter().any(|s| symset.contains(s)) {
        return Ok(Some((a.clone(), vec![])));
    }
    Ok(None)
}

/// The coefficients of the symbols, then the constant term
/// (`_linear_coeffs`); `None` when not linear.
fn linear_coeffs(e: &Basic, syms: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let Some((c, d)) = lin_eq2dict(e, syms)? else {
        return Ok(None);
    };
    let mut out: Vec<Basic> = syms
        .iter()
        .map(|s| {
            d.iter()
                .find(|(t, _)| t == s)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| int(0))
        })
        .collect();
    out.push(c);
    Ok(Some(out))
}

/// The size of a formula (`_measure`).
pub(crate) fn measure(b: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(b)
}

fn is_relational(b: &Basic) -> bool {
    matches!(b.head(), Head::Relational(_))
}

// PORTED FROM SYMPY: sympy/core/relational.py:Relational._eval_simplify
/// SymPy's `simplify` of a relation (`simplify_relational`): the sides
/// simplified, decided when the difference is a comparable number, the
/// canonical form solved for its only symbol or with coprime
/// coefficients; kept when shorter than `ratio` times the relation.
pub fn simplify_relational(r0: &Basic, ratio: f64, inverse: bool, doit: bool) -> R {
    use crate::core::add::add;
    use crate::core::mul::mul;
    use crate::core::power::pow;
    let zero: Basic = int(0);
    let minus = |e: &Basic| mul(&[int(-1), e.clone()]);
    let inv = |e: &Basic| pow(e, &int(-1));
    let built = |rel: &Basic, l: Basic, r: Basic| rebuild_evaluated(rel, vec![l, r]);
    let options = crate::simplify::simplify::Options {
        ratio,
        inverse,
        doit,
        ..Default::default()
    };
    let lhs: Basic = crate::simplify::simplify::simplify(&r0.args()[0], &options)?;
    let rhs: Basic = crate::simplify::simplify::simplify(&r0.args()[1], &options)?;
    let mut r: Basic = built(r0, lhs, rhs)?;
    if is_relational(&r) {
        let dif: Basic = add(&[r.args()[0].clone(), minus(&r.args()[1])?])?;
        if crate::core::expr::is_comparable(&dif) {
            // SymPy compares dif.n(2) with 0; the core has no floats: the
            // sign of the reliable value decides the relation the same way
            if let Some(value) = crate::numeric::reliable_value(&dif, 15, &[]) {
                let (re, im) = crate::core::evalf::parts(&value);
                let re: crate::numbers::Rat = crate::assumptions::ivarith::real_to_rat(&re);
                let im: crate::numbers::Rat = crate::assumptions::ivarith::real_to_rat(&im);
                if re != crate::numbers::Rat::ZERO || im != crate::numbers::Rat::ZERO {
                    let s: i64 = if crate::numbers::rat_is_negative(&re) { -1 } else { 1 };
                    r = built(&r, int(s), zero.clone())?;
                }
            }
        } else if crate::core::expr::equals(&dif, &zero)? {
            r = built(&r, zero.clone(), zero.clone())?;
        }
    }
    if is_relational(&r) {
        let rel: Basic = rl::canonical(&r)?;
        r = rel.clone();
        if is_relational(&rel) {
            let free: Vec<Basic> = rel
                .free_symbols()
                .into_iter()
                .filter(|s| is(s, Predicate::real) != Some(false))
                .collect();
            let dif: Basic = add(&[rel.args()[0].clone(), minus(&rel.args()[1])?])?;
            if free.len() == 1 {
                let x: &Basic = &free[0];
                match linear_coeffs(&dif, std::slice::from_ref(x))? {
                    Some(mb) => {
                        let (m, b) = (&mb[0], &mb[1]);
                        if is(m, Predicate::zero) == Some(false) {
                            let solved: Basic = mul(&[int(-1), b.clone(), inv(m)?])?;
                            r = if is(m, Predicate::negative) == Some(true) {
                                built(&rel, solved, x.clone())?
                            } else {
                                built(&rel, x.clone(), solved)?
                            };
                        } else {
                            r = built(&rel, b.clone(), zero.clone())?;
                        }
                    }
                    None => {
                        let attempt = || -> PolyResult<Basic> {
                            let p = crate::polys::polytools::Poly::new(&dif, std::slice::from_ref(x))?;
                            let mut c: Vec<Basic> = p.all_coeffs()?;
                            let n: usize = c.len() - 1;
                            let constant: Basic = c[n].clone();
                            c[n] = zero.clone();
                            let scale: Basic = crate::polys::polytools::gcd_list(&c, &[])?;
                            let mut terms: Vec<Basic> = Vec::new();
                            for (i, ci) in c.iter().enumerate() {
                                let scaled: Basic = mul(&[ci.clone(), inv(&scale)?])?;
                                terms.push(mul(&[scaled, pow(x, &int((n - i) as i64))?])?);
                            }
                            built(&rel, add(&terms)?, mul(&[int(-1), constant, inv(&scale)?])?)
                        };
                        match attempt() {
                            Ok(v) => r = v,
                            Err(PolyError::Polys(..)) => {}
                            Err(e) => return Err(e),
                        }
                    }
                }
            } else if free.len() >= 2 {
                let syms: Vec<Basic> = crate::core::sorting::ordered(free.clone())
                    .into_iter()
                    .filter(|s| s.is_symbol())
                    .collect();
                if let Some(mut coefficients) = linear_coeffs(&dif, &syms)? {
                    let constant: Basic = coefficients.pop().expect("the constant");
                    let scale: Basic = crate::polys::polytools::gcd_list(&coefficients, &[])?;
                    if is(&scale, Predicate::zero) == Some(false) {
                        let mut nzm: Vec<(Basic, Basic)> = Vec::new();
                        for (mi, s) in coefficients.iter().zip(syms.iter()) {
                            let scaled: Basic = mul(&[mi.clone(), inv(&scale)?])?;
                            if scaled != zero {
                                nzm.push((scaled, s.clone()));
                            }
                        }
                        let products = |items: &[(Basic, Basic)]| -> PolyResult<Vec<Basic>> {
                            items.iter().map(|(i, j)| mul(&[i.clone(), j.clone()])).collect()
                        };
                        if constant != zero {
                            r = built(&rel, add(&products(&nzm)?)?, mul(&[int(-1), constant, inv(&scale)?])?)?;
                        } else {
                            let first: Basic = mul(&[nzm[0].0.clone(), nzm[0].1.clone()])?;
                            r = built(&rel, first, mul(&[int(-1), add(&products(&nzm[1..])?)?])?)?;
                        }
                    } else {
                        r = built(&rel, constant, zero.clone())?;
                    }
                }
            }
        }
    }
    if is_relational(&r) {
        r = rl::canonical(&r)?;
    }
    if (measure(&r)? as f64) < ratio * measure(r0)? as f64 {
        return Ok(r);
    }
    Ok(r0.clone())
}

fn simplify_boolean_under(
    formula: &Basic,
    refiner: &Refiner,
    assumptions: &[Basic],
    domain: Option<&Basic>,
    ratio: f64,
    inverse: bool,
    doit: bool,
) -> R {
    let mut formula: Basic = formula.clone();
    if formula
        .preorder()
        .iter()
        .any(crate::assumptions::quantifiers::is_quantifier)
    {
        formula = crate::assumptions::resolve::resolve(&formula, domain, assumptions, None)?;
    }
    formula = refiner.refine_boolean(&formula)?;
    if matches!(formula.head(), Head::BooleanTrue | Head::BooleanFalse) {
        return Ok(formula);
    }
    if refiner.facts.is_polynomial(&formula)? && symbols(&formula).iter().all(|s| refiner.facts.real.contains(s)) {
        match crate::assumptions::resolve::resolve(&formula, domain, assumptions, None) {
            Ok(v) => return Ok(v),
            Err(PolyError::Value(_)) => {}
            Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => {}
            Err(e) => return Err(e),
        }
    }
    let relation = |r: &Basic| simplify_relational(r, ratio, inverse, doit);
    let simplified: Basic = refiner.through(&formula, &|e: &Basic| {
        if is_relational(e) {
            return relation(e);
        }
        if e.args().is_empty() {
            return Ok(e.clone());
        }
        crate::logic::boolsimp::simplify_boolean(e, &relation, &measure, ratio, false)
    })?;
    refiner.refine_boolean(&simplified)
}

/// An expression simplified under assumptions (`simplify`; Mathematica's
/// `Simplify[expr, assum]`), with SymPy's default `ratio` 1.7.
pub fn simplify(
    expr: &Basic,
    assumptions: &[Basic],
    domain: Option<&Basic>,
    ratio: f64,
    inverse: bool,
    doit: bool,
) -> R {
    let syms: Vec<Basic> = expr.free_symbols().into_iter().collect();
    let facts: Facts = facts_of(assumptions, domain, &syms)?;
    let refiner: Refiner = Refiner::new(facts);
    if is_boolean(expr) {
        return simplify_boolean_under(expr, &refiner, assumptions, domain, ratio, inverse, doit);
    }
    let options = crate::simplify::simplify::Options {
        ratio,
        inverse,
        doit,
        ..Default::default()
    };
    let refined: Basic = refiner.refine_expr(expr)?;
    let mut simplified: Basic = refiner.through(&refined, &|e: &Basic| {
        if !is_expr(e) {
            return Ok(e.clone());
        }
        crate::simplify::simplify::simplify(e, &options)
    })?;
    simplified = refiner.refine_expr(&simplified)?;
    let best: Basic = if size(&simplified)? <= size(&refined)? {
        simplified
    } else {
        refined
    };
    if is_expr(&best) {
        return proved_simplification(&best);
    }
    Ok(best)
}

/// `_proved_simplification`: the structure theorem, not ported yet.
fn proved_simplification(expr: &Basic) -> R {
    if !(crate::assumptions::ask::has_function(expr)
        || crate::assumptions::ask::has_fractional_power(expr)
        || expr.has(&Basic::pi()))
    {
        return Ok(expr.clone());
    }
    Err(crate::error::not_ported(
        "the proofs of simplify/structure (ElementaryTower)",
    ))
}
