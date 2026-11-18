//! Port of `sympy_extras/simplify/simplify.py`, so far [`logcombine`] and
//! [`posify`] (what `powsimp` needs); `nsimplify` needs `Float`, which the
//! Rust core does not have.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{is, is_true};
use crate::core::sorting::ordered;
use crate::core::symbol::Dummy;
use crate::error::PolyResult;
use crate::functions::elementary::exponential::{log, log_node};

fn is_log(e: &Basic) -> bool {
    e.is_function(Function::Log)
}

fn arg(e: &Basic) -> Basic {
    e.args()[0].clone()
}

fn mul(v: &[Basic]) -> PolyResult<Basic> {
    crate::core::mul::mul(v)
}

fn pow(b: &Basic, e: &Basic) -> PolyResult<Basic> {
    crate::core::power::pow(b, e)
}

fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

/// An insertion-ordered map, as Python's `dict`.
pub(crate) struct OrderedMap<K, V>(pub(crate) Vec<(K, V)>);

impl<K: PartialEq + Clone, V> OrderedMap<K, V> {
    pub(crate) fn new() -> Self {
        OrderedMap(Vec::new())
    }
    pub(crate) fn entry(&mut self, k: &K, default: impl FnOnce() -> V) -> &mut V {
        let i: usize = match self.0.iter().position(|(key, _)| key == k) {
            Some(i) => i,
            None => {
                self.0.push((k.clone(), default()));
                self.0.len() - 1
            }
        };
        &mut self.0[i].1
    }
    pub(crate) fn set(&mut self, k: K, v: V) {
        match self.0.iter().position(|(key, _)| *key == k) {
            Some(i) => self.0[i].1 = v,
            None => self.0.push((k, v)),
        }
    }
    pub(crate) fn contains(&self, k: &K) -> bool {
        self.0.iter().any(|(key, _)| key == k)
    }
    pub(crate) fn get(&self, k: &K) -> Option<&V> {
        self.0.iter().find(|(key, _)| key == k).map(|(_, v)| v)
    }
    pub(crate) fn pop(&mut self, k: &K) -> Option<V> {
        let i: usize = self.0.iter().position(|(key, _)| key == k)?;
        Some(self.0.remove(i).1)
    }
}

type LogTerm = (Vec<Basic>, Basic);

/// The logarithms of a sum or a product combined, bottom up
/// (`logcombine`).
pub fn logcombine(expr: &Basic, force: bool) -> PolyResult<Basic> {
    let f = |node: &Basic| -> PolyResult<Basic> { combine_node(node, force) };
    crate::core::traversal::bottom_up(expr, &f, false)
}

fn gooda(a: &Basic, force: bool) -> bool {
    let minus_one: bool = a.is_integer_value() && a.as_rational().is_some_and(|v| *v == crate::numbers::rat(-1, 1));
    !minus_one && (is_true(a, Predicate::extended_real) || (force && is(a, Predicate::extended_real) != Some(false)))
}

fn goodlog(l: &Basic, force: bool) -> bool {
    let a: Basic = arg(l);
    is_true(&a, Predicate::positive) || (force && is(&a, Predicate::nonpositive) != Some(false))
}

fn combine_node(node: &Basic, force: bool) -> PolyResult<Basic> {
    if !node.is_add() && !node.is_mul() {
        return Ok(node.clone());
    }
    let mut other: Vec<Basic> = Vec::new();
    let mut logs: Vec<(Vec<Basic>, Vec<Basic>, Vec<Basic>)> = Vec::new();
    let mut log1: OrderedMap<Vec<Basic>, Vec<LogTerm>> = OrderedMap::new();
    let terms: Vec<Basic> = if node.is_add() {
        node.args().to_vec()
    } else {
        vec![node.clone()]
    };
    for a in terms {
        if is_log(&a) && goodlog(&a, force) {
            log1.entry(&vec![], Vec::new).push((vec![], a));
        } else if !a.is_mul() {
            other.push(a);
        } else {
            let (mut ot, mut co, mut lo): (Vec<Basic>, Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new(), Vec::new());
            for ai in a.args() {
                if ai.as_rational().is_some_and(crate::numbers::rat_is_negative) {
                    ot.push(Basic::negative_one());
                    co.push(crate::core::mul::neg(ai)?);
                } else if is_log(ai) && goodlog(ai, force) {
                    lo.push(ai.clone());
                } else if gooda(ai, force) {
                    co.push(ai.clone());
                } else {
                    ot.push(ai.clone());
                }
            }
            if lo.len() > 1 {
                logs.push((ot, co, lo));
            } else if let Some(l) = lo.pop() {
                log1.entry(&ot, Vec::new).push((co, l));
            } else {
                other.push(a);
            }
        }
    }
    if other.len() == 1 && is_log(&other[0]) {
        let l: Basic = other.pop().expect("one term");
        log1.entry(&vec![], Vec::new).push((vec![], l));
    }
    if logs.is_empty() && log1.0.is_empty() {
        return Ok(node.clone());
    }
    for (o, e, l0) in logs {
        let mut l: std::collections::VecDeque<Basic> = ordered(l0).into();
        let first: Basic = l.pop_front().expect("two logarithms");
        let mut combined: Basic = log(&pow(&arg(&first), &mul(&e)?)?)?;
        while let Some(li) = l.pop_front() {
            combined = log(&pow(&arg(&li), &combined)?)?;
        }
        let c: Basic = mul(&o)?;
        if is_log(&combined) {
            log1.entry(&vec![c], Vec::new).push((vec![], combined));
        } else {
            other.push(mul(&[c, combined])?);
        }
    }
    let mut joined: OrderedMap<Basic, Basic> = OrderedMap::new();
    for (key, terms) in std::mem::take(&mut log1.0) {
        let powers: Vec<Basic> = terms
            .iter()
            .map(|(cs, lg)| pow(&arg(lg), &mul(cs)?))
            .collect::<PolyResult<_>>()?;
        let inner: Basic = logcombine(&mul(&powers)?, force)?;
        joined.set(mul(&key)?, log_node(inner));
    }
    let keys: Vec<Basic> = ordered(joined.0.iter().map(|(k, _)| k.clone()).collect());
    for k in keys {
        if !joined.contains(&k) {
            continue;
        }
        let minus_k: Basic = crate::core::mul::neg(&k)?;
        if joined.contains(&minus_k) {
            let (mut num, mut den) = (k.clone(), minus_k);
            if ops(&num)? > ops(&den)? {
                std::mem::swap(&mut num, &mut den);
            }
            let numerator: Basic = joined.pop(&num).expect("present");
            let denominator: Basic = joined.pop(&den).expect("present");
            let ratio: Basic = mul(&[arg(&numerator), pow(&arg(&denominator), &Basic::negative_one())?])?;
            other.push(mul(&[num, log_node(ratio)])?);
        } else {
            let v: Basic = joined.pop(&k).expect("present");
            other.push(mul(&[k, v])?);
        }
    }
    crate::core::add::add(&other)
}

/// `eq` with the symbols of unknown sign replaced by positive dummies of
/// the same name, and the mapping back (`posify`).
pub fn posify(eq: &Basic) -> PolyResult<(Basic, HashMap<Basic, Basic>)> {
    let mut reps: Vec<(Basic, Basic)> = Vec::new();
    for s in eq.free_symbols() {
        let data = match s.head() {
            Head::Symbol(d) | Head::Dummy(d, _) => (**d).clone(),
            Head::Wild(w) => w.symbol.clone(),
            _ => continue,
        };
        if is(&s, Predicate::positive).is_some() {
            continue;
        }
        let mut assumptions: Vec<(Predicate, bool)> = data.assumptions.iter().map(|(p, v)| (*p, *v)).collect();
        assumptions.retain(|(p, _)| *p != Predicate::positive);
        assumptions.push((Predicate::positive, true));
        reps.push((s.clone(), Dummy::new(&data.name, &assumptions).0));
    }
    let replaced: Basic = crate::core::subs::subs_mapping(eq, &reps)?;
    Ok((replaced, reps.into_iter().map(|(s, r)| (r, s)).collect()))
}
