//! Port of `Max` and `Min` of `sympy_extras/functions/elementary/
//! miscellaneous.py` (SymPy's `MinMaxBase`): the n-ary nodes
//! [`Function::Max`] and [`Function::Min`] and their evaluation (the
//! nested nodes flattened, the identity dropped, the zero absorbing, the
//! arguments collapsed against the numbers and the opposite nodes, common
//! arguments factored, the arguments of known order removed), facts and
//! values. The derivative needs `Heaviside` (not ported) unless no argument
//! depends on the variable.

use crate::core::add::{add, sub};
use crate::core::assumptions::{facts_of, is_number, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{is, is_comparable};
use crate::core::parameters::evaluate;
use crate::core::sorting::sort_key;
use crate::error::{value_error, PolyResult};

use Predicate::*;

fn is_max(f: Function) -> bool {
    f == Function::Max
}

fn opposite(f: Function) -> Function {
    if is_max(f) {
        Function::Min
    } else {
        Function::Max
    }
}

/// The absorbing argument (`oo` for `Max`).
fn absorbing(f: Function) -> Basic {
    if is_max(f) {
        Basic::oo()
    } else {
        Basic::negative_oo()
    }
}

/// The neutral argument (`-oo` for `Max`).
fn identity(f: Function) -> Basic {
    if is_max(f) {
        Basic::negative_oo()
    } else {
        Basic::oo()
    }
}

/// `x >= y` when decided (`_ge`).
fn ge(x: &Basic, y: &Basic) -> PolyResult<Option<bool>> {
    if x == y {
        return Ok(Some(true));
    }
    let difference: Basic = sub(x, y)?;
    if matches!(difference.head(), Head::NaN) {
        return Ok(None);
    }
    Ok(is(&difference, extended_nonnegative))
}

fn unique(args: Vec<Basic>) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for a in args {
        if !out.contains(&a) {
            out.push(a);
        }
    }
    out
}

/// `cls(*args, evaluate=False)` without repetitions, sorted; the identity
/// for none, the argument for one (`_node`).
fn node(f: Function, args: Vec<Basic>) -> Basic {
    let mut items: Vec<Basic> = unique(args);
    items.sort_by_cached_key(sort_key);
    match items.len() {
        0 => identity(f),
        1 => items.remove(0),
        _ => Basic::new(Head::Function(f), items),
    }
}

fn function_of(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(g @ (Function::Max | Function::Min)) => Some(*g),
        _ => None,
    }
}

/// The arguments flattened, the identity dropped; `None` for the zero
/// (`_new_args_filter`, `ShortCircuit`).
fn filtered(f: Function, args: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let mut out: Vec<Basic> = Vec::new();
    for arg in args {
        if crate::core::expr::is_false(arg, extended_real) || (is_number(arg) && !is_comparable(arg)) {
            return Err(value_error(format!("The argument '{}' is not comparable.", arg)));
        }
        if *arg == absorbing(f) {
            return Ok(None);
        }
        if *arg == identity(f) {
            continue;
        }
        if function_of(arg) == Some(f) {
            out.extend(arg.args().iter().cloned());
        } else {
            out.push(arg.clone());
        }
    }
    Ok(Some(out))
}

/// The `Min` and `Max` nodes reached through `Min` and `Max` nodes (`_walk`).
fn walk(e: &Basic, out: &mut Vec<Basic>) {
    if function_of(e).is_some() {
        out.push(e.clone());
        for a in e.args() {
            walk(a, out);
        }
    }
}

fn collapse(f: Function, given: Vec<Basic>) -> PolyResult<Vec<Basic>> {
    if given.is_empty() {
        return Ok(Vec::new());
    }
    let mut args: Vec<Basic> = given;
    args.sort_by_cached_key(sort_key);
    let mut other: Function = opposite(f);
    if is_number(&args[0]) {
        let mut mins: Vec<Basic> = Vec::new();
        let mut maxs: Vec<Basic> = Vec::new();
        for a in &args {
            let mut nodes: Vec<Basic> = Vec::new();
            walk(a, &mut nodes);
            for n in nodes {
                if is_comparable(&n.args()[0]) {
                    if function_of(&n) == Some(Function::Max) {
                        maxs.push(n);
                    } else {
                        mins.push(n);
                    }
                }
            }
        }
        let mut small: Basic = identity(Function::Min);
        for v in &mins {
            let first: &Basic = &v.args()[0];
            if is_number(first) && ge(first, &small)? == Some(false) {
                small = first.clone();
            }
        }
        let mut big: Basic = identity(Function::Max);
        for v in &maxs {
            let first: &Basic = &v.args()[0];
            if is_number(first) && ge(&big, first)? == Some(false) {
                big = first.clone();
            }
        }
        for a in &args {
            if !is_number(a) {
                break;
            }
            if !is_max(f) && ge(a, &small)? == Some(false) {
                small = a.clone();
            } else if is_max(f) && ge(&big, a)? == Some(false) {
                big = a.clone();
            }
        }
        let mut threshold: Option<Basic> = None;
        if !is_max(f) {
            if small != identity(Function::Min) {
                other = Function::Max;
                threshold = Some(small);
            }
        } else if big != identity(Function::Max) {
            other = Function::Min;
            threshold = Some(big);
        }
        if let Some(t) = threshold {
            for slot in args.iter_mut() {
                if function_of(slot) == Some(other) {
                    let first: Basic = slot.args()[0].clone();
                    let beyond: bool = if other == Function::Max {
                        ge(&t, &first)? == Some(false)
                    } else {
                        ge(&first, &t)? == Some(false)
                    };
                    if beyond {
                        *slot = identity(f);
                    }
                }
            }
        }
    }
    fn reduce(f: Function, ai: &Basic, a: &Basic) -> Basic {
        let Some(g) = function_of(ai) else { return ai.clone() };
        let sub_args: &[Basic] = ai.args();
        if !sub_args.contains(a) {
            return node(g, sub_args.iter().map(|b| reduce(f, b, a)).collect());
        }
        if g == f {
            return node(
                g,
                sub_args.iter().filter(|b| *b != a).map(|b| reduce(f, b, a)).collect(),
            );
        }
        a.clone()
    }
    for i in 0..args.len() {
        let a: Basic = args[i].clone();
        for slot in args.iter_mut().skip(i + 1) {
            *slot = reduce(f, slot, &a);
        }
    }
    if args.len() > 1 {
        args = factor_minmax(f, args, other);
    }
    Ok(args)
}

fn factor_minmax(f: Function, args: Vec<Basic>, other: Function) -> Vec<Basic> {
    let other_args: Vec<&Basic> = args.iter().filter(|a| function_of(a) == Some(other)).collect();
    if other_args.is_empty() {
        return args;
    }
    let remaining: Vec<Basic> = args.iter().filter(|a| function_of(a) != Some(other)).cloned().collect();
    let arg_sets: Vec<Vec<Basic>> = other_args.iter().map(|a| unique(a.args().to_vec())).collect();
    let common: Vec<Basic> = arg_sets[0]
        .iter()
        .filter(|x| arg_sets.iter().all(|s| s.contains(x)))
        .cloned()
        .collect();
    if common.is_empty() {
        return args;
    }
    let mut new_other_args: Vec<Basic> = common.clone();
    new_other_args.sort_by_cached_key(sort_key);
    let differences: Vec<Vec<Basic>> = arg_sets
        .iter()
        .map(|s| s.iter().filter(|x| !common.contains(x)).cloned().collect())
        .collect();
    if differences.iter().all(|d| !d.is_empty()) {
        new_other_args.push(node(f, differences.into_iter().map(|d| node(other, d)).collect()));
    }
    let mut result: Vec<Basic> = remaining;
    result.push(node(other, new_other_args));
    result
}

/// What `_is_connected` finds: `Equal` for equal arguments, `Greater`
/// when `x >= y` is decided true (`Max`), `Less` when false (`Min`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Connection {
    None,
    Equal,
    Of(Function),
}

fn is_connected(x: &Basic, y: &Basic) -> PolyResult<Connection> {
    let (mut x, mut y) = (x.clone(), y.clone());
    for _ in 0..2 {
        if x == y {
            return Ok(Connection::Equal);
        }
        let (mut t, mut f) = (Function::Max, Function::Min);
        for op in ['>', '<'] {
            for _ in 0..2 {
                if crate::core::expr::is_false(&x, extended_real) || crate::core::expr::is_false(&y, extended_real) {
                    return Ok(Connection::None);
                }
                let v: Option<bool> = if op == '>' { ge(&x, &y)? } else { ge(&y, &x)? };
                if let Some(v) = v {
                    return Ok(Connection::Of(if v { t } else { f }));
                }
                std::mem::swap(&mut t, &mut f);
                std::mem::swap(&mut x, &mut y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        x = crate::core::exprtools::factor_terms(&sub(&x, &y)?, false, false, false, true)?;
        y = Basic::zero();
    }
    Ok(Connection::None)
}

fn find_localzeros(f: Function, values: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut zeros: Vec<Basic> = Vec::new();
    for v in values {
        let mut is_new: bool = true;
        for z in zeros.clone() {
            if v.ptr_eq(&z) {
                is_new = false;
                continue;
            }
            let connection: Connection = is_connected(v, &z)?;
            if connection != Connection::None {
                is_new = false;
                if connection == Connection::Equal || connection == Connection::Of(f) {
                    if let Some(position) = zeros.iter().position(|w| *w == z) {
                        zeros.remove(position);
                    }
                    if !zeros.contains(v) {
                        zeros.push(v.clone());
                    }
                }
            }
        }
        if is_new && !zeros.contains(v) {
            zeros.push(v.clone());
        }
    }
    Ok(zeros)
}

/// `MinMaxBase.eval`.
fn eval(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    let Some(given) = filtered(f, args)? else {
        return Ok(absorbing(f));
    };
    let collapsed: Vec<Basic> = collapse(f, unique(given))?;
    let zeros: Vec<Basic> = find_localzeros(f, &collapsed)?;
    Ok(node(f, zeros))
}

fn apply(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if evaluate() {
        return eval(f, args);
    }
    Ok(Basic::new(Head::Function(f), args.to_vec()))
}

/// The maximum of the arguments, evaluated (`max_`).
pub fn max_(args: &[Basic]) -> PolyResult<Basic> {
    apply(Function::Max, args)
}

/// The minimum of the arguments, evaluated (`min_`).
pub fn min_(args: &[Basic]) -> PolyResult<Basic> {
    apply(Function::Min, args)
}

/// The evaluating function of `Max` or `Min`.
pub fn minmax(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    apply(f, args)
}

/// The facts the arguments agree on (`_eval_facts`).
pub fn minmax_facts(f: Function, args: &[Basic]) -> FactList {
    const AGREEING: [Predicate; 26] = [
        algebraic,
        antihermitian,
        commutative,
        complex,
        composite,
        even,
        finite,
        hermitian,
        imaginary,
        infinite,
        integer,
        irrational,
        noninteger,
        nonpositive,
        nonzero,
        odd,
        polar,
        prime,
        rational,
        real,
        extended_real,
        transcendental,
        zero,
        negative,
        nonnegative,
        positive,
    ];
    let subs: Vec<&crate::core::assumptions::Facts> = args.iter().map(facts_of).collect();
    let mut facts: FactList = FactList::new();
    for p in AGREEING {
        let values: Vec<Option<bool>> = subs.iter().map(|s| s.get(&p).copied()).collect();
        if values.iter().all(|v| *v == Some(true)) {
            facts.set(p, true);
        } else if values.iter().all(|v| *v == Some(false)) {
            facts.set(p, false);
        }
    }
    if subs.iter().all(|s| s.get(&finite) == Some(&true)) {
        let (any_names, all_names): (&[Predicate], &[Predicate]) = if is_max(f) {
            (&[positive, nonnegative], &[negative])
        } else {
            (&[negative], &[positive, nonnegative])
        };
        for p in any_names {
            if subs.iter().any(|s| s.get(p) == Some(&true)) {
                facts.set(*p, true);
            }
        }
        for p in all_names {
            if subs.iter().any(|s| s.get(p) == Some(&false)) {
                facts.set(*p, false);
            }
        }
    }
    facts
}

/// The derivative (`_eval_derivative`): zero when no argument depends on
/// `s`, otherwise it needs `Heaviside` (not ported).
pub fn minmax_derivative(e: &Basic, s: &Basic) -> PolyResult<Basic> {
    let mut depends: bool = false;
    for a in e.args() {
        let da: Basic = crate::core::function::derivative(a, s)?;
        if !crate::core::expr::is_true(&da, Predicate::zero) {
            depends = true;
        }
    }
    if depends {
        return Err(crate::error::not_ported("Heaviside (the derivative of Max and Min)"));
    }
    add(&[])
}

/// The value: the largest (smallest) of the real values (`numeric`).
pub fn minmax_value(f: Function, values: &[crate::core::evalf::Complex]) -> Option<crate::core::evalf::Complex> {
    use crate::core::evalf::{parts, Real};
    let mut best: Option<(Real, usize)> = None;
    for (i, v) in values.iter().enumerate() {
        let (a, b) = parts(v);
        if !b.repr().significand().is_zero() {
            return None;
        }
        best = match best {
            None => Some((a, i)),
            Some((m, j)) => {
                let better: bool = if is_max(f) { a > m } else { a < m };
                if better {
                    Some((a, i))
                } else {
                    Some((m, j))
                }
            }
        };
    }
    best.map(|(_, i)| values[i].clone())
}
