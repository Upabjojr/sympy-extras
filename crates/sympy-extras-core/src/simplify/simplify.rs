//! Port of `sympy_extras/simplify/simplify.py`: [`simplify`],
//! [`logcombine`], [`posify`], [`collect_abs`], [`inversecombine`] and
//! [`kroneckersimp`]; `nsimplify` needs `Float`, which the Rust core does
//! not have, and `nc_simplify` and the simplification of sums and
//! products are not ported; [`besselsimp`] is.

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

// -- collect_abs, inversecombine, kroneckersimp

fn is_non_expr(e: &Basic) -> bool {
    matches!(e.head(), Head::Tuple) || crate::core::logic::is_boolean(e)
}

/// `replace` with a predicate and a callable, bottom up; a replaced
/// subtree is not examined again.
fn replace(
    e: &Basic,
    query: &dyn Fn(&Basic) -> bool,
    value: &dyn Fn(&Basic) -> PolyResult<Basic>,
) -> PolyResult<Basic> {
    let mut n: Basic = e.clone();
    if !n.args().is_empty() {
        let args: Vec<Basic> = n
            .args()
            .iter()
            .map(|a| replace(a, query, value))
            .collect::<PolyResult<_>>()?;
        if args.as_slice() != n.args() {
            n = crate::core::function::rebuild_evaluated(&n, args)?;
        }
    }
    if query(&n) {
        return value(&n);
    }
    Ok(n)
}

fn abs_node(x: Basic) -> Basic {
    Basic::new(Head::Function(Function::Abs), vec![x])
}

fn collect_abs_node(e: &Basic) -> PolyResult<Basic> {
    let factors: Vec<Basic> = if e.is_mul() {
        crate::core::expr::factors_of(e)
    } else {
        vec![e.clone()]
    };
    let c: Vec<Basic> = factors
        .iter()
        .filter(|f| !crate::core::expr::is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    let nc: Vec<Basic> = factors
        .iter()
        .filter(|f| crate::core::expr::is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    let (mut a, mut o): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for i in c {
        if i.is_function(Function::Abs) {
            a.push(arg(&i));
        } else if i.is_pow() && i.args()[0].is_function(Function::Abs) && is_true(&i.args()[1], Predicate::real) {
            a.push(pow(&arg(&i.args()[0]), &i.args()[1])?);
        } else {
            o.push(i);
        }
    }
    if a.len() < 2
        && !a
            .iter()
            .any(|i| i.is_pow() && is_true(&i.args()[1], Predicate::negative))
    {
        return Ok(e.clone());
    }
    let absarg: Basic = mul(&a)?;
    let mut big_a: Basic = crate::functions::elementary::complexes::abs_(&absarg)?;
    let has_abs: bool = big_a.preorder().iter().any(|n| n.is_function(Function::Abs));
    if has_abs && !big_a.is_function(Function::Abs) {
        big_a = abs_node(absarg);
    }
    let mut parts: Vec<Basic> = vec![big_a];
    parts.extend(o);
    parts.extend(nc);
    mul(&parts)
}

/// The arguments of several `Abs` of a product joined under one
/// (`collect_abs`).
pub fn collect_abs(e: &Basic) -> PolyResult<Basic> {
    let e: Basic = replace(e, &|x: &Basic| x.is_mul(), &collect_abs_node)?;
    replace(&e, &|x: &Basic| x.is_pow(), &collect_abs_node)
}

fn inverse_of(f: Function) -> Option<Function> {
    use Function as F;
    Some(match f {
        F::Exp => F::Log,
        F::Log => F::Exp,
        F::Tan => F::Atan,
        F::Cot => F::Acot,
        F::Asin => F::Sin,
        F::Acos => F::Cos,
        F::Atan => F::Tan,
        F::Acot => F::Cot,
        F::Asec => F::Sec,
        F::Acsc => F::Csc,
        F::Sinh => F::Asinh,
        F::Tanh => F::Atanh,
        F::Coth => F::Acoth,
        F::Asinh => F::Sinh,
        F::Acosh => F::Cosh,
        F::Atanh => F::Tanh,
        F::Acoth => F::Coth,
        F::Asech => F::Sech,
        F::Acsch => F::Csch,
        F::Erf => F::Erfinv,
        F::Erfc => F::Erfcinv,
        F::Erfinv => F::Erf,
        F::Erfcinv => F::Erfc,
        _ => return None,
    })
}

fn function_of(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}

/// Functions of their inverses cancelled (`inversecombine`).
pub fn inversecombine(e: &Basic) -> PolyResult<Basic> {
    let exponent = |b: &Basic| -> Option<Basic> {
        if b.is_function(Function::Exp) {
            return Some(arg(b));
        }
        if b.is_pow() && matches!(b.args()[0].head(), Head::Exp1) {
            return Some(b.args()[1].clone());
        }
        None
    };
    let f = |b: &Basic| -> PolyResult<Basic> {
        let mut rv: Basic = b.clone();
        if rv.is_function(Function::Log) {
            if let Some(ex) = exponent(&rv.args()[0]) {
                rv = ex;
            }
        } else if let Some(inv) = function_of(&rv).and_then(inverse_of) {
            if rv.args().len() == 1 && rv.args()[0].args().len() == 1 && rv.args()[0].is_function(inv) {
                rv = rv.args()[0].args()[0].clone();
            }
        }
        if rv.is_pow() && matches!(rv.args()[0].head(), Head::Exp1) && rv.args()[1].is_function(Function::Log) {
            rv = rv.args()[1].args()[0].clone();
        } else if rv.is_function(Function::Exp) && rv.args()[0].is_function(Function::Log) {
            rv = rv.args()[0].args()[0].clone();
        }
        Ok(rv)
    };
    crate::core::traversal::bottom_up(e, &f, false)
}

/// Products of Kronecker deltas which cannot both be 1 set to 0
/// (`kroneckersimp`); with a `Piecewise`, its rewriting as Kronecker
/// deltas is not ported.
pub fn kroneckersimp(e: &Basic) -> PolyResult<Basic> {
    let is_delta = |n: &Basic| n.is_function(Function::KroneckerDelta);
    if !e.preorder().iter().any(is_delta) {
        return Ok(e.clone());
    }
    if e.preorder().iter().any(|n| matches!(n.head(), Head::Piecewise)) {
        return Err(crate::error::not_ported(
            "kroneckersimp of a Piecewise (rewrite as KroneckerDelta)",
        ));
    }
    let truth = |a: &Basic, b: &Basic| -> PolyResult<Option<bool>> {
        Ok(match crate::core::relational::eq(a, b)?.head() {
            Head::BooleanTrue => Some(true),
            Head::BooleanFalse => Some(false),
            _ => None,
        })
    };
    let args_cancel = |a1: &[Basic], a2: &[Basic]| -> PolyResult<bool> {
        for i1 in 0..2 {
            for i2 in 0..2 {
                if truth(&a1[i1], &a2[i2])? == Some(true) && truth(&a1[(i1 + 1) % 2], &a2[(i2 + 1) % 2])? == Some(false)
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    };
    let cancel_mul = |b: &Basic| -> PolyResult<Basic> {
        let deltas: Vec<&Basic> = b.args().iter().filter(|a| is_delta(a)).collect();
        for i in 0..deltas.len() {
            for j in i + 1..deltas.len() {
                if args_cancel(deltas[i].args(), deltas[j].args())? {
                    return mul(&[Basic::zero(), b.clone()]);
                }
            }
        }
        Ok(b.clone())
    };
    let mut e: Basic = e.clone();
    let mut previous: Option<Basic> = None;
    while previous.as_ref() != Some(&e) {
        previous = Some(e.clone());
        e = replace(&e, &|x: &Basic| x.is_mul(), &cancel_mul)?;
    }
    Ok(e)
}

// -- simplify

/// The measure of `simplify` (`count_ops` by default).
pub type Measure = fn(&Basic) -> PolyResult<i64>;

/// The options of `simplify`.
#[derive(Clone, Copy)]
pub struct Options {
    pub ratio: f64,
    pub measure: Measure,
    pub inverse: bool,
    pub doit: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            ratio: 1.7,
            measure: ops,
            inverse: false,
            doit: true,
        }
    }
}

fn first_min(choices: Vec<Basic>, measure: Measure) -> PolyResult<Basic> {
    let mut best: Option<(i64, Basic)> = None;
    for c in choices {
        let k: i64 = measure(&c)?;
        if best.as_ref().map_or(true, |(b, _)| k < *b) {
            best = Some((k, c));
        }
    }
    Ok(best.expect("choices").1)
}

fn shorter(choices: Vec<Basic>, measure: Measure) -> PolyResult<Basic> {
    if choices.iter().all(|c| *c == choices[0]) {
        return Ok(choices[0].clone());
    }
    first_min(choices, measure)
}

fn has_head(e: &Basic, test: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(test)
}

fn is_combinatorial(f: Function) -> bool {
    use Function as F;
    matches!(
        f,
        F::Factorial | F::Factorial2 | F::Subfactorial | F::RisingFactorial | F::FallingFactorial | F::Binomial
    )
}

fn expand_with(e: &Basic, power_exp: bool, log: bool) -> PolyResult<Basic> {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp,
        log,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}

fn eval_simplify(e: &Basic, o: &Options) -> PolyResult<Option<Basic>> {
    match e.head() {
        Head::Hyper => {
            // hyper._eval_simplify
            use super::hyperexpand::{hyperexpand, Rewrite};
            return Ok(Some(hyperexpand(e, false, Rewrite::Default, None)?));
        }
        Head::Piecewise => {
            return Ok(Some(crate::functions::elementary::piecewise::piecewise_simplify(
                e, o.doit,
            )?))
        }
        Head::Function(f) if crate::functions::special::bessel::is_bessel(*f) => return Ok(Some(besselsimp(e)?)),
        Head::Function(Function::Log) => {
            let mut new: Basic = log(&simplify(&arg(e), o)?)?;
            if o.inverse {
                new = inversecombine(&new)?;
            }
            new = expand_with(&new, false, true)?;
            return Ok(Some(first_min(vec![new, e.clone()], o.measure)?));
        }
        Head::Function(Function::Sign) => {
            let inner: Basic = crate::core::exprtools::factor_terms(&arg(e), false, false, false, true)?;
            return Ok(Some(crate::functions::elementary::complexes::sign(&inner)?));
        }
        Head::Function(f) if is_combinatorial(*f) => {
            let new: Basic = super::gammasimp::combsimp(e)?;
            return Ok(Some(
                if ((o.measure)(&new)? as f64) <= o.ratio * ((o.measure)(e)? as f64) {
                    new
                } else {
                    e.clone()
                },
            ));
        }
        _ => {}
    }
    Ok(None)
}

fn simplified_arg(a: &Basic, o: &Options) -> PolyResult<Basic> {
    if matches!(a.head(), Head::Tuple) {
        let args: Vec<Basic> = a
            .args()
            .iter()
            .map(|x| simplified_arg(x, o))
            .collect::<PolyResult<_>>()?;
        return Ok(Basic::new(Head::Tuple, args));
    }
    if crate::core::logic::is_boolean(a) {
        // Basic.simplify of a Boolean: the native simplify_boolean, the
        // arguments simplified when it raises NotImplementedError
        if a.args().is_empty() {
            return Ok(a.clone());
        }
        match crate::logic::boolsimp::simplify_boolean_default(a, 1.7) {
            Ok(v) => return Ok(v),
            Err(crate::error::PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {}
            Err(e) => return Err(e),
        }
        let args: Vec<Basic> = a
            .args()
            .iter()
            .map(|x| simplified_arg(x, &Options::default()))
            .collect::<PolyResult<_>>()?;
        if args.iter().zip(a.args()).all(|(new, old)| new == old) {
            return Ok(a.clone());
        }
        return crate::core::function::rebuild_evaluated(a, args);
    }
    if matches!(a.head(), Head::ExprCondPair) {
        // the expression and the condition of a piece
        return Ok(crate::functions::elementary::piecewise::expr_cond_pair(
            &simplify(&a.args()[0], o)?,
            &crate::functions::elementary::piecewise::simplify_condition(&a.args()[1])?,
        ));
    }
    simplify(a, o)
}

/// Whether folding `rewritten` (`e` with `sign(u)` as a `Piecewise` with a
/// case `0` at `u = 0`) gives a value to `e` at a point where it has none
/// (sympy/sympy#31275, V66) (`_zero_case_at_a_pole`).
fn zero_case_at_a_pole(e: &Basic, rewritten: &Basic) -> PolyResult<bool> {
    let folded: Basic = crate::functions::elementary::piecewise::piecewise_fold(rewritten, None)?;
    for node in folded.preorder() {
        if !matches!(node.head(), Head::Piecewise) {
            continue;
        }
        for pair in node.args() {
            let cond: &Basic = &pair.args()[1];
            if !matches!(cond.head(), Head::Relational(crate::core::basic::RelOp::Eq)) {
                continue;
            }
            let (lhs, rhs) = (&cond.args()[0], &cond.args()[1]);
            if !(lhs.is_symbol() && crate::core::assumptions::is_number(rhs)) {
                continue;
            }
            let value: Basic = crate::core::subs::subs(e, lhs, rhs)?;
            let bad: bool = value.preorder().iter().any(|n| {
                matches!(
                    n.head(),
                    Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
                )
            });
            if bad {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The expression simplified: the shortest of SymPy's heuristics by the
/// measure, the input when every result is more than `ratio` times longer
/// (`simplify`). Not ported: the simplification of `Piecewise` conditions,
/// `hyper`/`meijerg` (`hyperexpand`), the Bessel functions (`besselsimp`)
/// and noncommutative products (`nc_simplify`); the core has no `Sum`,
/// `Product` or `Float`.
pub fn simplify(expr: &Basic, o: &Options) -> PolyResult<Basic> {
    use crate::core::function::expand_mul;
    let measure: Measure = o.measure;
    let done = |e: &Basic| -> PolyResult<Basic> {
        let rv: Basic = if o.doit {
            crate::core::function::doit(e)?
        } else {
            e.clone()
        };
        let collected: Basic = collect_abs(&rv)?;
        shorter(vec![rv, collected], measure)
    };
    let e: Basic = expr.clone();
    if is_true(&e, Predicate::zero) {
        return Ok(if e.is_number_class() { e } else { Basic::zero() });
    }
    if let Some(hook) = eval_simplify(&e, o)? {
        return Ok(hook);
    }
    let original: Basic = collect_abs(&super::signsimp::signsimp(&e, true)?)?;
    let mut e: Basic = original.clone();
    if e.args().is_empty() {
        return Ok(e);
    }
    if o.inverse && has_head(&e, &|n| matches!(n.head(), Head::Function(_) | Head::Undefined(..))) {
        e = inversecombine(&e)?;
        if e.args().is_empty() {
            return Ok(e);
        }
    }
    let handled = |x: &Basic| x.is_add() || x.is_mul() || x.is_pow() || x.is_function(Function::Exp);
    let query = |x: &Basic| !is_non_expr(x) && !x.args().is_empty() && !handled(x);
    let value = |x: &Basic| -> PolyResult<Basic> {
        let args: Vec<Basic> = x
            .args()
            .iter()
            .map(|a| simplified_arg(a, o))
            .collect::<PolyResult<_>>()?;
        crate::core::function::rebuild_evaluated(x, args)
    };
    e = replace(&e, &query, &value)?;
    if !handled(&e) {
        return done(&e);
    }
    if crate::core::expr::is_false(&e, Predicate::commutative) {
        return Err(crate::error::not_ported(
            "simplify of a noncommutative product (nc_simplify)",
        ));
    }
    e = crate::core::traversal::bottom_up(
        &e,
        &|w: &Basic| {
            if is_non_expr(w) {
                Ok(w.clone())
            } else {
                super::radsimp::normal(w)
            }
        },
        false,
    )?;
    let (c, p) = crate::core::exprtools::as_content_primitive(&super::powsimp::powsimp_default(&e)?, false, true)?;
    e = mul(&[c, p])?;
    let cancel = |x: &Basic| crate::polys::polytools::cancel(x, &[]);
    let together = |x: &Basic| crate::polys::rationaltools::together(x, true, true);
    let canceled: Basic = cancel(&e)?;
    let expr1: Basic = shorter(
        vec![canceled.clone(), cancel(&super::sqrtdenest::mexpand(&canceled)?)?],
        measure,
    )?;
    let expr2: Basic = shorter(vec![together(&e)?, together(&expr1)?], measure)?;
    e = if o.ratio == f64::INFINITY {
        expr2
    } else {
        shorter(vec![expr2, expr1, e], measure)?
    };
    e = crate::core::exprtools::factor_terms(&e, false, false, false, false)?;
    let has_f = |x: &Basic, f: Function| has_head(x, &|n| n.is_function(f));
    if has_f(&e, Function::Sign) {
        let rewritten: Basic =
            crate::core::rewriting::rewrite(&e, &[], "Abs", &crate::core::rewriting::RewriteHints::default())?;
        if !zero_case_at_a_pole(&e, &rewritten)? {
            e = rewritten;
        }
    }
    let has_pw = |x: &Basic| has_head(x, &|n| matches!(n.head(), Head::Piecewise));
    if has_pw(&e) {
        use crate::functions::elementary::piecewise::{piecewise_fold, piecewise_simplify};
        e = done(&piecewise_fold(&e, Some(true))?)?;
        if has_pw(&e) {
            e = piecewise_fold(&e, Some(true))?;
            if has_f(&e, Function::KroneckerDelta) {
                e = kroneckersimp(&e)?;
            }
            if has_pw(&e) {
                // SymPy's piecewise_simplify: the folded expression is a Piecewise
                if matches!(e.head(), Head::Piecewise) {
                    e = piecewise_simplify(&e, false)?;
                }
                if has_pw(&e) {
                    let factored: Basic = crate::core::exprtools::factor_terms(&e, false, false, false, true)?;
                    return shorter(vec![e, factored], measure);
                }
            }
        }
    }
    if has_head(&e, &|n| {
        matches!(
            n.head(),
            Head::Hyper | Head::MeijerG | Head::Function(Function::ExpPolar | Function::PolarLift)
        )
    }) {
        e = super::hyperexpand::hyperexpand(&e, false, super::hyperexpand::Rewrite::Default, None)?;
    }
    if has_f(&e, Function::KroneckerDelta) {
        e = kroneckersimp(&e)?;
    }
    if has_head(
        &e,
        &|n| matches!(n.head(), Head::Function(f) if crate::functions::special::bessel::is_bessel(*f)),
    ) {
        e = besselsimp(&e)?;
    }
    let mut trig_hyper: Vec<Function> = super::fu::TRIG.to_vec();
    trig_hyper.extend(super::fu::HYPER);
    if super::fu::has_function(&e, &trig_hyper) {
        e = super::trigsimp::trigsimp(&e, super::trigsimp::Method::Matching)?;
    }
    if has_f(&e, Function::Log) {
        e = shorter(vec![expand_with(&e, false, true)?, logcombine(&e, false)?], measure)?;
    }
    if has_head(
        &e,
        &|n| matches!(n.head(), Head::Function(f) if is_combinatorial(*f) || *f == Function::Gamma),
    ) {
        e = super::gammasimp::combsimp(&e)?;
    }
    use super::powsimp::{powsimp, Combine};
    let count = |x: &Basic| crate::core::function::count_ops(x);
    let mut short: Basic = shorter(
        vec![
            powsimp(&e, true, Combine::Exp, false, &count)?,
            super::powsimp::powsimp_default(&e)?,
            e.clone(),
        ],
        measure,
    )?;
    short = shorter(vec![short.clone(), cancel(&short)?], measure)?;
    short = shorter(
        vec![
            short.clone(),
            crate::core::exprtools::factor_terms(&short, false, false, false, true)?,
            expand_with(&expand_mul(&short, true)?, true, false)?,
        ],
        measure,
    )?;
    let mut with_exp: Vec<Function> = trig_hyper.clone();
    with_exp.push(Function::Exp);
    if super::fu::has_function(&short, &with_exp) {
        short = super::trigsimp::exptrigsimp(&short)?;
    }
    let hollow = |x: &Basic| {
        x.is_mul()
            && x.args().len() == 2
            && x.args()[0].is_number_class()
            && x.args()[1].is_add()
            && !crate::core::expr::is_false(x, Predicate::commutative)
    };
    e = replace(&short, &hollow, &|x: &Basic| mul(x.args()))?;
    let (numer, denom) = crate::core::expr::as_numer_denom(&e)?;
    if denom.is_add() {
        let (n, d) = super::radsimp::fraction(&super::radsimp::radsimp(
            &pow(&denom, &Basic::negative_one())?,
            false,
            1,
        )?)?;
        if n != Basic::one() {
            e = mul(&[
                crate::core::function::expand_default(&mul(&[numer, n])?)?,
                pow(&d, &Basic::negative_one())?,
            ])?;
        }
    }
    if crate::core::printing::could_extract_minus_sign(&e) {
        let (n, d) = super::radsimp::fraction(&e)?;
        if d != Basic::zero() {
            let flipped: Basic = mul(&[
                crate::core::mul::neg(&n)?,
                pow(&crate::core::mul::neg(&d)?, &Basic::negative_one())?,
            ])?;
            e = super::signsimp::signsimp(&flipped, true)?;
        }
    }
    if (measure(&e)? as f64) > o.ratio * (measure(&original)? as f64) {
        e = original;
    }
    done(&e)
}

// ---------------------------------------------------------------------
// besselsimp

type BesselMaker<'a> = dyn Fn(&Basic, &Basic) -> PolyResult<Basic> + 'a;

fn bessel(f: Function) -> impl Fn(&Basic, &Basic) -> PolyResult<Basic> {
    move |nu: &Basic, z: &Basic| crate::functions::apply_function(f, &[nu.clone(), z.clone()])
}

/// `to(nu, z)` when a factor of `z` is one of `factors`, `fro(nu, z)`
/// otherwise (besselsimp's `replacer`).
fn replacer(fro: Function, to: &BesselMaker<'_>, factors: &[Basic], nu: &Basic, z: &Basic) -> PolyResult<Basic> {
    let zfactors: Vec<Basic> = if z.is_mul() { z.args().to_vec() } else { vec![z.clone()] };
    if zfactors.iter().any(|f| factors.contains(f)) {
        return to(nu, z);
    }
    bessel(fro)(nu, z)
}

fn replace_bessel(t: &Basic, f: Function, g: &BesselMaker<'_>) -> PolyResult<Basic> {
    crate::core::function::replace_where(t, &|n| n.is_function(f), &|n| g(&n.args()[0], &n.args()[1]))
}

/// Bessel functions simplified: the factors `I` and `-1` of the arguments
/// taken out, the half-integer orders written with trigonometric
/// functions, the integer orders expanded, and the recurrence used to
/// remove the lowest orders; the result factored when it changed
/// (`besselsimp`).
pub fn besselsimp(expr: &Basic) -> PolyResult<Basic> {
    use crate::core::mul::{div, mul};
    use crate::functions::elementary::polar::{exp_polar, unpolarify};
    let m2 = |a: &Basic, b: &Basic| mul(&[a.clone(), b.clone()]);
    let a2 = |a: &Basic, b: &Basic| crate::core::add::add(&[a.clone(), b.clone()]);
    let i: Basic = Basic::atom(Head::ImaginaryUnit);
    let pi: Basic = Basic::pi();
    let ipi: Basic = m2(&i, &pi)?;
    let mipi: Basic = m2(&crate::core::mul::neg(&i)?, &pi)?;
    let hints = crate::core::function::Hints {
        log: false,
        func: true,
        ..crate::core::function::Hints::DEFAULT
    };
    let rewrite = |e: &Basic, target: &str| {
        crate::core::rewriting::rewrite(e, &[], target, &crate::core::rewriting::RewriteHints::default())
    };
    let orig: Basic = expr.clone();
    let mut e: Basic = expr.clone();
    let ifactors: Vec<Basic> = vec![
        i.clone(),
        exp_polar(&div(&ipi, &Basic::int(2))?)?,
        exp_polar(&div(&mipi, &Basic::int(2))?)?,
    ];
    let j_to_i = |nu: &Basic, z: &Basic| rewrite(&bessel(Function::BesselJ)(nu, z)?, "besseli");
    let i_to_j = |nu: &Basic, z: &Basic| rewrite(&bessel(Function::BesselI)(nu, z)?, "besselj");
    e = replace_bessel(&e, Function::BesselJ, &|nu: &Basic, z: &Basic| {
        replacer(Function::BesselJ, &j_to_i, &ifactors, nu, z)
    })?;
    e = replace_bessel(&e, Function::BesselI, &|nu: &Basic, z: &Basic| {
        replacer(Function::BesselI, &i_to_j, &ifactors, nu, z)
    })?;
    let minusfactors: Vec<Basic> = vec![Basic::int(-1), exp_polar(&ipi)?];
    let tominus = |f: Function| {
        let ipi = ipi.clone();
        let mipi = mipi.clone();
        move |nu: &Basic, z: &Basic| -> PolyResult<Basic> {
            let factor: Basic = crate::functions::elementary::exponential::exp(&m2(&ipi, nu)?)?;
            m2(&factor, &bessel(f)(nu, &m2(&exp_polar(&mipi)?, z)?)?)
        }
    };
    let j_minus = tominus(Function::BesselJ);
    let i_minus = tominus(Function::BesselI);
    e = replace_bessel(&e, Function::BesselJ, &|nu: &Basic, z: &Basic| {
        replacer(Function::BesselJ, &j_minus, &minusfactors, nu, z)
    })?;
    e = replace_bessel(&e, Function::BesselI, &|nu: &Basic, z: &Basic| {
        replacer(Function::BesselI, &i_minus, &minusfactors, nu, z)
    })?;
    let z0: Basic = Dummy::new("z", &[]).0;
    let expander = |f: Function| {
        let z0 = z0.clone();
        move |nu: &Basic, z: &Basic| -> PolyResult<Basic> {
            if crate::core::modulo::modulo(nu, &Basic::one())? == Basic::half() {
                let mut rewritten: Basic = rewrite(&bessel(f)(nu, &z0)?, "besselj")?;
                rewritten = rewrite(&rewritten, "jn")?;
                rewritten = crate::core::function::expand_with(&rewritten, true, hints)?;
                let simpler: Basic = simplify(
                    &super::trigsimp::trigsimp(&unpolarify(&rewritten, false)?, super::trigsimp::Method::Matching)?,
                    &Options::default(),
                )?;
                return crate::core::subs::subs(&simpler, &z0, z);
            }
            if nu
                .as_rational()
                .is_some_and(|q| q.is_int() && *q > crate::numbers::rat(1, 1))
            {
                return crate::core::function::expand_with(&bessel(f)(nu, z)?, true, hints);
            }
            bessel(f)(nu, z)
        }
    };
    for f in [
        Function::BesselJ,
        Function::BesselY,
        Function::BesselI,
        Function::BesselK,
    ] {
        e = replace_bessel(&e, f, &expander(f))?;
    }
    let use_recursion = |f: Function, t: Basic| -> PolyResult<Basic> {
        let mut t: Basic = t;
        loop {
            let mut found: Vec<Basic> = Vec::new();
            for n in t.preorder() {
                if n.is_function(f) && !found.contains(&n) {
                    found.push(n);
                }
            }
            let keys: Vec<Basic> = found
                .iter()
                .map(|b| crate::functions::elementary::complexes::re(&b.args()[0]))
                .collect::<PolyResult<_>>()?;
            if found.len() > 1 && !keys.iter().all(crate::core::expr::is_comparable) {
                return Ok(t);
            }
            let mut ordered_found: Vec<(f64, Basic)> = Vec::new();
            for (k, b) in keys.iter().zip(found) {
                let value: f64 = crate::core::evalf::numeric_value(k, 64)
                    .map(|v| crate::core::evalf::parts(&v).0.to_f64().value())
                    .unwrap_or(0.0);
                ordered_found.push((value, b));
            }
            if ordered_found.len() > 1 {
                ordered_found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            let mut changed: bool = false;
            for (_, ba) in &ordered_found {
                let (a, x) = (&ba.args()[0], &ba.args()[1]);
                let bap1: Basic = bessel(f)(&a2(a, &Basic::one())?, x)?;
                let bap2: Basic = bessel(f)(&a2(a, &Basic::int(2))?, x)?;
                if t.has(&bap1) && t.has(&bap2) {
                    let two_a1: Basic = m2(&Basic::int(2), &a2(a, &Basic::one())?)?;
                    let new: Basic = crate::core::add::sub(&m2(&div(&two_a1, x)?, &bap1)?, &bap2)?;
                    t = crate::core::subs::subs(&t, ba, &new)?;
                    changed = true;
                    break;
                }
            }
            if !changed {
                return Ok(t);
            }
        }
    };
    let has = |e: &Basic, f: Function| e.preorder().iter().any(|n| n.is_function(f));
    if has(&e, Function::BesselJ) {
        e = use_recursion(Function::BesselJ, e)?;
    }
    if has(&e, Function::BesselY) {
        e = use_recursion(Function::BesselY, e)?;
    }
    if e != orig {
        e = crate::polys::polytools::factor(&e, &[], false, true)?;
    }
    Ok(e)
}
