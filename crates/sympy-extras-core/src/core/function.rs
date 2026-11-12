//! Port of parts of `sympy_extras/core/function.py`: the evaluated rebuild
//! of a node (`_evaluated`) and the expansion with the hints `mul` and
//! `multinomial` ([`expand_mul`], [`expand_multinomial`]). The other hints
//! (`power_base`, `power_exp`, `log`, `func`, `trig`, `complex`), the
//! derivatives, `Lambda` and `Subs` are not ported yet.

use crate::error::PolyResult;
use crate::numbers::*;
use crate::simplify::radsimp::fraction;

use super::add::add;
use super::assumptions::Predicate;
use super::basic::{Basic, Function, Head};
use super::expr::{as_base_exp, is_false, is_true, terms_of};
use super::mul::mul;
use super::power::pow;

/// The evaluated node with `args` in place of its arguments (`_evaluated`).
pub fn rebuild_evaluated(e: &Basic, args: Vec<Basic>) -> PolyResult<Basic> {
    match e.head() {
        Head::Add => add(&args),
        Head::Mul => mul(&args),
        Head::Pow => pow(&args[0], &args[1]),
        Head::Function(f) => crate::functions::apply_function(*f, &args),
        Head::Derivative => {
            // Derivative._evaluated: diff of the new expression
            let counts: Vec<(Basic, u64)> = derivative_counts(&e.rebuild(args.clone()));
            diff(&args[0], &counts)
        }
        Head::Subs => substituted(&args[0], args[1].args(), args[2].args()),
        Head::And => Ok(super::logic::and_(&args)),
        Head::Or => Ok(super::logic::or_(&args)),
        Head::Not => Ok(super::logic::not_(&args[0])),
        Head::Xor => Ok(super::logic::xor(&args)),
        Head::Implies => Ok(super::logic::implies(&args[0], &args[1])),
        Head::Equivalent => Ok(super::logic::equivalent(&args)),
        Head::Ite => Ok(super::logic::ite(&args[0], &args[1], &args[2])),
        Head::Relational(op) => super::relational::evaluate_relation(*op, &args[0], &args[1]),
        Head::AccumBounds => super::accumbounds::accum_bounds(&args[0], &args[1]),
        Head::Piecewise => crate::functions::elementary::piecewise::piecewise_of_pairs(args),
        Head::ExprCondPair => crate::functions::elementary::piecewise::checked_pair(&args[0], &args[1]),
        h @ (Head::EmptySet
        | Head::UniversalSet
        | Head::Reals
        | Head::Interval
        | Head::FiniteSet
        | Head::Union
        | Head::Intersection
        | Head::Complement
        | Head::Contains
        | Head::ProductSet
        | Head::PointSet
        | Head::ConditionSet) => crate::sets::sets::rebuild_set(h, &args),
        Head::IndexedRoot => {
            let k: usize = args[2]
                .as_rational()
                .and_then(|v| usize::try_from(&crate::numbers::numer(v)).ok())
                .unwrap_or(0);
            crate::polys::cad::indexedroot::indexed_root(&args[0], &args[1], k)
        }
        _ => Ok(e.rebuild(args)),
    }
}

/// Whether `fraction` puts (part of) the factor in the denominator.
fn denominator_factor(f: &Basic) -> PolyResult<bool> {
    if let Some(v) = f.as_rational() {
        return Ok(denom(v) != Int::ONE);
    }
    if f.is_pow() || f.is_function(Function::Exp) {
        let exponent: Basic = as_base_exp(f)?.1;
        return Ok(is_true(&exponent, Predicate::negative) || exponent.is_mul());
    }
    Ok(false)
}

/// The terms of the product of the sums (`Mul._expandsums`).
fn expandsums(sums: &[Basic]) -> PolyResult<Vec<Basic>> {
    if sums.len() == 1 {
        return Ok(terms_of(&sums[0]));
    }
    let half: usize = sums.len() / 2;
    let (left, right) = (expandsums(&sums[..half])?, expandsums(&sums[half..])?);
    let mut products: Vec<Basic> = Vec::new();
    for a in &left {
        for b in &right {
            products.push(mul(&[a.clone(), b.clone()])?);
        }
    }
    Ok(terms_of(&add(&products)?))
}

/// SymPy's distribution of a product (`Mul._eval_expand_mul`).
fn expand_mul_node(e: &Basic, deep: bool) -> PolyResult<Basic> {
    let mut expr: Basic = e.clone();
    let mut count: usize = 0;
    for f in e.args() {
        if denominator_factor(f)? {
            count += 1;
        }
    }
    if count >= 2 {
        let (mut n, mut d) = fraction(e)?;
        if d.is_mul() {
            if n.is_mul() {
                n = expand_mul_node(&n, deep)?;
            }
            d = expand_mul_node(&d, deep)?;
        }
        expr = mul(&[n, pow(&d, &Basic::negative_one())?])?;
    }
    if !expr.is_mul() {
        return Ok(expr);
    }
    let mut plain: Vec<Basic> = Vec::new();
    let mut sums: Vec<Basic> = Vec::new();
    for f in expr.args() {
        if f.is_add() || is_false(f, Predicate::commutative) {
            sums.push(f.clone());
        } else {
            plain.push(f.clone());
        }
    }
    if !sums.iter().any(Basic::is_add) {
        return Ok(expr);
    }
    let plain_product: Basic = mul(&plain)?;
    let mut result: Vec<Basic> = Vec::new();
    for term in expandsums(&sums)? {
        let mut t: Basic = mul(&[plain_product.clone(), term])?;
        if t.is_mul() && t.args().iter().any(Basic::is_add) && deep {
            t = expand_mul_node(&t, deep)?;
        }
        result.push(t);
    }
    add(&result)
}

/// The tuples of `parts` nonnegative integers with the sum `n`, the first
/// part decreasing (`_compositions`).
fn compositions(n: usize, parts: usize) -> Vec<Vec<usize>> {
    if parts == 1 {
        return vec![vec![n]];
    }
    let mut result: Vec<Vec<usize>> = Vec::new();
    for first in (0..=n).rev() {
        for rest in compositions(n - first, parts - 1) {
            let mut c: Vec<usize> = vec![first];
            c.extend(rest);
            result.push(c);
        }
    }
    result
}

fn factorial(n: usize) -> Int {
    (1..=n).fold(Int::ONE, |acc, k| acc * Int::from(k))
}

/// `base**n` by the multinomial theorem (`_expand_power`).
fn expand_power(base: &Basic, n: usize) -> PolyResult<Basic> {
    let terms: Vec<Basic> = terms_of(base);
    let mut expanded: Vec<Basic> = Vec::new();
    for exponents in compositions(n, terms.len()) {
        let mut coefficient: Int = factorial(n);
        for k in &exponents {
            coefficient /= factorial(*k);
        }
        let mut factors: Vec<Basic> = vec![Basic::integer(coefficient)];
        for (t, k) in terms.iter().zip(&exponents) {
            if *k > 0 {
                factors.push(pow(t, &Basic::int(*k as i64))?);
            }
        }
        expanded.push(mul(&factors)?);
    }
    add(&expanded)
}

fn expand_product(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    let mut terms: Vec<Basic> = Vec::new();
    for s in terms_of(a) {
        for t in terms_of(b) {
            terms.push(mul(&[s.clone(), t])?);
        }
    }
    add(&terms)
}

fn small(n: &Int) -> PolyResult<usize> {
    usize::try_from(n).map_err(|_| crate::error::not_ported("the expansion of a huge power"))
}

/// `_xreplace_evaluated`: the keys of `rule` replaced by their values,
/// the rebuilt nodes evaluated (the largest subtrees first).
pub fn xreplace_evaluated(e: &Basic, rule: &std::collections::HashMap<Basic, Basic>) -> PolyResult<Basic> {
    if let Some(v) = rule.get(e) {
        return Ok(v.clone());
    }
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(xreplace_evaluated(a, rule)?);
    }
    if args.iter().zip(e.args()).all(|(a, b)| a.ptr_eq(b)) {
        return Ok(e.clone());
    }
    rebuild_evaluated(e, args)
}

/// `replace(query, value)` with a predicate on nodes: bottom up, the
/// rebuilt nodes evaluated, a replaced subtree not examined again.
pub fn replace_where(
    e: &Basic,
    query: &dyn Fn(&Basic) -> bool,
    value: &dyn Fn(&Basic) -> PolyResult<Basic>,
) -> PolyResult<Basic> {
    let mut node: Basic = e.clone();
    if !node.args().is_empty() {
        let mut args: Vec<Basic> = Vec::new();
        for a in node.args() {
            args.push(replace_where(a, query, value)?);
        }
        if args.iter().zip(node.args()).any(|(a, b)| !a.ptr_eq(b) && a != b) {
            node = rebuild_evaluated(&node, args)?;
        }
    }
    if query(&node) {
        return value(&node);
    }
    Ok(node)
}

/// The hints of an expansion (`expand`'s keywords; `func` and `trig` are
/// not ported).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hints {
    pub mul: bool,
    pub multinomial: bool,
    pub power_base: bool,
    pub power_exp: bool,
    pub log: bool,
    pub force: bool,
    /// Every node as `re + I*im` (SymPy's `complex`), before the others.
    pub complex: bool,
}

impl Hints {
    /// The default hints of `expand()`.
    pub const DEFAULT: Hints = Hints {
        mul: true,
        multinomial: true,
        power_base: true,
        power_exp: true,
        log: true,
        force: false,
        complex: false,
    };
    const NONE: Hints = Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
    };
}

/// Whether the terms of a sum are all nonnegative or all nonpositive
/// (`_all_nonneg_or_nonpos`).
fn all_nonneg_or_nonpos(e: &Basic) -> bool {
    use super::assumptions::Predicate;
    use super::expr::is_true;
    let (mut nonnegative, mut nonpositive) = (false, false);
    for a in e.args() {
        if is_true(a, Predicate::nonnegative) {
            if nonpositive {
                return false;
            }
            nonnegative = true;
        } else if is_true(a, Predicate::nonpositive) {
            if nonnegative {
                return false;
            }
            nonpositive = true;
        } else {
            return false;
        }
    }
    true
}

/// `(a*b)**e` split into `a**e*b**e` for the factors which allow it
/// (`_expand_power_base_node`); `None` when the base is not a product.
fn expand_power_base_node(e: &Basic, force: bool) -> PolyResult<Option<Basic>> {
    use super::assumptions::Predicate;
    use super::expr::{is, is_false, is_integer_class, is_rational_value, is_true};
    use super::mul::neg;
    let (b, ex) = (&e.args()[0], &e.args()[1]);
    if !b.is_mul() {
        return Ok(None);
    }
    let cargs: Vec<Basic> = b
        .args()
        .iter()
        .filter(|f| !is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    let mut nc: Vec<Basic> = b
        .args()
        .iter()
        .filter(|f| is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    if !nc.is_empty() {
        let mut expanded: Vec<Basic> = Vec::new();
        for f in &nc {
            expanded.push(if f.is_pow() {
                expand_power_base_node(f, force)?.unwrap_or_else(|| f.clone())
            } else {
                f.clone()
            });
        }
        nc = expanded;
        if is_integer_class(ex) {
            let n: i64 = i64::try_from(ex.as_rational().expect("an integer").numerator())
                .map_err(|_| crate::error::not_ported("a huge power"))?;
            let mut rv: Basic = if n > 0 {
                let repeated: Vec<Basic> = (0..n).flat_map(|_| nc.iter().cloned()).collect();
                Basic::new(Head::Mul, repeated)
            } else {
                let mut inverses: Vec<Basic> = Vec::new();
                for f in nc.iter().rev() {
                    inverses.push(pow(f, &Basic::negative_one())?);
                }
                let repeated: Vec<Basic> = (0..-n).flat_map(|_| inverses.iter().cloned()).collect();
                Basic::new(Head::Mul, repeated)
            };
            if !cargs.is_empty() {
                rv = mul(&[rv, pow(&mul(&cargs)?, ex)?])?;
            }
            return Ok(Some(rv));
        }
        if cargs.is_empty() {
            return Ok(Some(Basic::new(Head::Pow, vec![Basic::new(Head::Mul, nc), ex.clone()])));
        }
        nc = vec![Basic::new(Head::Mul, nc)];
    }
    let mut other: Vec<Basic> = cargs
        .iter()
        .filter(|f| is_false(f, Predicate::extended_real))
        .cloned()
        .collect();
    let maybe_real: Vec<Basic> = cargs
        .iter()
        .filter(|f| !is_false(f, Predicate::extended_real))
        .cloned()
        .collect();
    let mut nonneg: Vec<Basic> = Vec::new();
    let mut negative: Vec<Basic> = Vec::new();
    let mut imag: usize = 0;
    for f in maybe_real {
        if matches!(f.head(), Head::ImaginaryUnit) {
            imag += 1;
            continue;
        }
        match is(&f, Predicate::polar) {
            Some(true) => nonneg.push(f),
            None => match is(&f, Predicate::extended_nonnegative) {
                Some(true) => nonneg.push(f),
                Some(false) => negative.push(f),
                None => other.push(f),
            },
            Some(false) => other.push(f),
        }
    }
    if imag > 0 {
        let i: usize = imag % 4;
        if i == 1 {
            other.push(Basic::imaginary_unit());
        } else if i == 2 || i == 3 {
            if let Some(last) = negative.pop() {
                let nonn: Basic = neg(&last)?;
                if !(is_integer_class(&nonn) && is_rational_value(&nonn, 1)) {
                    nonneg.push(nonn);
                }
            } else {
                negative.push(Basic::negative_one());
            }
            if i == 3 {
                other.push(Basic::imaginary_unit());
            }
        }
    }
    let cargs: Vec<Basic>;
    if force || is_true(ex, Predicate::integer) {
        let mut all: Vec<Basic> = nonneg;
        all.extend(negative);
        all.extend(other);
        cargs = all;
        other = nc;
    } else {
        if negative.len() > 1 {
            let mut o: Basic = Basic::one();
            if other.is_empty() && negative[0].is_number_class() {
                o = mul(&[o, negative.remove(0)])?;
            }
            if negative.len() % 2 == 1 {
                o = neg(&o)?;
            }
            for n in &negative {
                nonneg.push(neg(n)?);
            }
            if !(is_integer_class(&o) && is_rational_value(&o, 1)) {
                other.push(o);
            }
        } else if !negative.is_empty() && !other.is_empty() {
            if negative[0].is_number_class() && !(is_integer_class(&negative[0]) && is_rational_value(&negative[0], -1))
            {
                other.push(Basic::negative_one());
                nonneg.push(neg(&negative[0])?);
            } else {
                other.extend(negative);
            }
        } else {
            other.extend(negative);
        }
        other.extend(nc);
        cargs = nonneg;
    }
    let mut evaluated: Vec<Basic> = Vec::new();
    let mut kept: Vec<Basic> = Vec::new();
    if !cargs.is_empty() {
        let mut rest: Vec<Basic> = Vec::new();
        if ex.as_rational().is_some() {
            for f in cargs {
                if f.is_pow() && f.args()[1].as_rational().is_some() && super::assumptions::is_number(&f.args()[0]) {
                    evaluated.push(pow(&doit(&f)?, ex)?);
                } else {
                    rest.push(f);
                }
            }
        } else {
            rest = cargs;
        }
        kept = rest;
    }
    if !other.is_empty() {
        kept.push(mul(&other)?);
    }
    if evaluated.is_empty() && kept.len() == 1 {
        return Ok(Some(Basic::new(Head::Pow, vec![kept.remove(0), ex.clone()])));
    }
    let mut factors: Vec<Basic> = evaluated;
    for f in &kept {
        factors.push(pow(f, ex)?);
    }
    Ok(Some(mul(&factors)?))
}

/// `unpolarify` on the core, which has no polar numbers: the tree
/// evaluated again until it does not change.
pub(crate) fn unpolarify(e: &Basic) -> PolyResult<Basic> {
    let mut current: Basic = e.clone();
    loop {
        let next: Basic = doit(&current)?;
        if next == current {
            return Ok(current);
        }
        current = next;
    }
}

/// The logarithm split by the hint `log` (`_expand_log_node`; `factor`
/// is that of `expand_log`).
pub fn expand_log_node(node: &Basic, force: bool, factor: bool) -> PolyResult<Basic> {
    use super::assumptions::Predicate;
    use super::expr::{as_base_exp, is_integer_class, is_true};
    use crate::functions::elementary::exponential::log;
    let arg: &Basic = &node.args()[0];
    if is_integer_class(arg) {
        let value: Int = arg.as_rational().expect("an integer").numerator().clone();
        let mut logarg: Option<Basic> = None;
        let mut coeff: Basic = Basic::one();
        let mut base: Int = value.clone();
        if value > Int::ONE {
            if let Some((b, k)) = crate::ntheory::factor_::perfect_power(&value) {
                base = b;
                coeff = Basic::integer(Int::from(k));
                logarg = Some(log(&Basic::integer(base.clone()))?);
            }
        }
        if factor {
            let primes = if base > Int::ONE {
                crate::ntheory::factor_::factorint(&base, None)
            } else {
                Default::default()
            };
            if !primes.contains_key(&base) {
                let mut terms: Vec<Basic> = Vec::new();
                for (p, k) in primes {
                    terms.push(mul(&[Basic::integer(Int::from(k)), log(&Basic::integer(p))?])?);
                }
                logarg = Some(add(&terms)?);
            }
        }
        if let Some(l) = logarg {
            return mul(&[coeff, l]);
        }
    } else if let Some(v) = arg.as_rational() {
        return add(&[
            log(&Basic::integer(numer(v)))?,
            mul(&[Basic::negative_one(), log(&Basic::integer(denom(v)))?])?,
        ]);
    } else if arg.is_mul() {
        let mut terms: Vec<Basic> = Vec::new();
        let mut nonpos: Vec<Basic> = Vec::new();
        for x in arg.args() {
            if force || is_true(x, Predicate::positive) || is_true(x, Predicate::polar) {
                let a: Basic = log(x)?;
                terms.push(if a.is_function(Function::Log) {
                    expand_log_node(&a, force, factor)?
                } else {
                    a
                });
            } else if is_true(x, Predicate::negative) {
                terms.push(log(&mul(&[Basic::negative_one(), x.clone()])?)?);
                nonpos.push(Basic::negative_one());
            } else {
                nonpos.push(x.clone());
            }
        }
        return add(&[add(&terms)?, log(&mul(&nonpos)?)?]);
    } else if arg.is_pow() || arg.is_function(Function::Exp) {
        let (b, ex) = as_base_exp(arg)?;
        let in_range = || -> PolyResult<bool> {
            Ok(is_true(&add(&[ex.clone(), Basic::one()])?, Predicate::positive)
                && is_true(&add(&[ex.clone(), Basic::negative_one()])?, Predicate::nonpositive))
        };
        if force
            || (is_true(&ex, Predicate::extended_real) && (is_true(&b, Predicate::positive) || in_range()?))
            || is_true(&b, Predicate::polar)
        {
            let a: Basic = log(&b)?;
            if a.is_function(Function::Log) {
                return mul(&[unpolarify(&ex)?, expand_log_node(&a, force, factor)?]);
            }
            return mul(&[unpolarify(&ex)?, a]);
        }
    }
    log(arg)
}

/// One pass of the expansion with the hints (`_expand`).
fn expand_pass(e: &Basic, deep: bool, hints: Hints) -> PolyResult<Basic> {
    let mut e: Basic = e.clone();
    if deep && !e.args().is_empty() && !matches!(e.head(), Head::Derivative) {
        let mut args: Vec<Basic> = Vec::new();
        for a in e.args() {
            args.push(expand_pass(a, deep, hints)?);
        }
        if args.iter().zip(e.args()).any(|(new, old)| new != old) {
            e = rebuild_evaluated(&e, args)?;
        }
    }
    if e.is_mul() && hints.mul {
        return expand_mul_node(&e, deep);
    }
    if e.is_pow() {
        let (b, ex) = (e.args()[0].clone(), e.args()[1].clone());
        if hints.multinomial && b.is_add() {
            if let Some(x) = ex.as_rational() {
                let (p, q) = (numer(x), denom(x));
                if q == Int::ONE && !p.is_zero() {
                    let power: Basic = expand_power(&b, small(&abs(&p))?)?;
                    return if is_negative(&p) {
                        pow(&power, &Basic::negative_one())
                    } else {
                        Ok(power)
                    };
                }
                if q > Int::ONE && p > q {
                    let whole: Int = div_floor(&p, &q);
                    let rest: Basic = Basic::from_rational(rat_from_parts(&p - &whole * &q, q.clone()));
                    return expand_product(&expand_power(&b, small(&whole)?)?, &pow(&b, &rest)?);
                }
                if q > Int::ONE && -&p > q {
                    let whole: Int = div_floor(&-&p, &q);
                    let rest: Basic = Basic::from_rational(rat_from_parts(-&p - &whole * &q, q.clone()));
                    let product: Basic = expand_product(&expand_power(&b, small(&whole)?)?, &pow(&b, &rest)?)?;
                    return pow(&product, &Basic::negative_one());
                }
            }
        }
        if hints.power_base && b.is_mul() {
            if let Some(split) = expand_power_base_node(&e, hints.force)? {
                // the power_exp hint is a later pass of SymPy over the whole
                // tree: the new powers are split too
                if hints.power_exp {
                    let again = Hints {
                        power_exp: true,
                        force: hints.force,
                        ..Hints::NONE
                    };
                    return expand_pass(&split, true, again);
                }
                return Ok(split);
            }
        }
        if hints.power_exp
            && ex.is_add()
            && (hints.force
                || super::expr::is_false(&b, super::assumptions::Predicate::zero)
                || all_nonneg_or_nonpos(&ex))
            && !super::expr::is_false(&ex, super::assumptions::Predicate::commutative)
        {
            let mut factors: Vec<Basic> = Vec::new();
            for t in ex.args() {
                factors.push(pow(&b, t)?);
            }
            return mul(&factors);
        }
    }
    if hints.power_exp && e.is_function(Function::Exp) && e.args()[0].is_add() {
        let mut factors: Vec<Basic> = Vec::new();
        for t in e.args()[0].args() {
            factors.push(pow(&Basic::e(), t)?);
        }
        return mul(&factors);
    }
    if hints.log && e.is_function(Function::Log) {
        return expand_log_node(&e, hints.force, false);
    }
    Ok(e)
}

/// `expand` with the given hints: the log hint first over the whole tree
/// (with `mul` or `multinomial`), then all the hints, then the passes of
/// `mul`, `multinomial` and `log` repeated until nothing changes, as
/// SymPy's `Expr.expand`.
pub fn expand_with(e: &Basic, deep: bool, hints: Hints) -> PolyResult<Basic> {
    let mut e: Basic = e.clone();
    let mut hints: Hints = hints;
    if hints.complex {
        // SymPy applies its hints in sorted order: complex before log,
        // multinomial, mul and the power hints (ExpandHints carries no force)
        e = expand_complex(
            &e,
            deep,
            &Hints {
                complex: true,
                force: false,
                ..hints
            },
        )?;
        hints.complex = false;
    }
    if hints.log && (hints.mul || hints.multinomial) {
        e = expand_pass(
            &e,
            deep,
            Hints {
                log: true,
                force: hints.force,
                ..Hints::NONE
            },
        )?;
    }
    let mut result: Basic = expand_pass(&e, deep, hints)?;
    if hints.mul || hints.multinomial || hints.log {
        let repeated = Hints {
            power_exp: false,
            power_base: false,
            ..hints
        };
        loop {
            let was: Basic = result.clone();
            result = expand_pass(&result, deep, repeated)?;
            if result == was {
                break;
            }
        }
    }
    Ok(result)
}

/// `expand` with the hints `mul` and `multinomial` only.
pub fn expand(e: &Basic, deep: bool, mul_hint: bool, multinomial: bool) -> PolyResult<Basic> {
    expand_with(
        e,
        deep,
        Hints {
            mul: mul_hint,
            multinomial,
            ..Hints::NONE
        },
    )
}

/// Products distributed over sums only.
pub fn expand_mul(e: &Basic, deep: bool) -> PolyResult<Basic> {
    expand(e, deep, true, false)
}

/// Integer powers of sums expanded only.
pub fn expand_multinomial(e: &Basic, deep: bool) -> PolyResult<Basic> {
    expand(e, deep, false, true)
}

/// `expand(e)` with the default hints of the function `expand`
/// (logarithms split).
pub fn expand_default(e: &Basic) -> PolyResult<Basic> {
    expand_with(e, true, Hints::DEFAULT)
}

/// `e.expand()`, the method: its defaults leave the logarithms (`log =
/// False`), unlike the function `expand`.
pub fn expand_method(e: &Basic) -> PolyResult<Basic> {
    expand_with(
        e,
        true,
        Hints {
            log: false,
            ..Hints::DEFAULT
        },
    )
}

/// `e.expand(deep, **hints)` inside `as_real_imag` (`expand_with_hints`):
/// the hints of the `expand` call (`complex` replaced when given), the
/// defaults of the method without a call.
pub fn expand_with_hints(e: &Basic, hints: Option<&Hints>, complex: Option<bool>) -> PolyResult<Basic> {
    match hints {
        None => expand_method(e),
        Some(h) => {
            let mut h: Hints = *h;
            if let Some(c) = complex {
                h.complex = c;
            }
            h.force = false;
            expand_with(e, true, h)
        }
    }
}

/// Every node (its arguments first) as `re + I*im` (`_expand_complex`).
fn expand_complex(e: &Basic, deep: bool, hints: &Hints) -> PolyResult<Basic> {
    let mut e: Basic = e.clone();
    if deep && !e.args().is_empty() {
        let mut args: Vec<Basic> = Vec::new();
        for a in e.args() {
            args.push(expand_complex(a, deep, hints)?);
        }
        if args
            .iter()
            .zip(e.args())
            .any(|(new, old)| !new.ptr_eq(old) && new != old)
        {
            e = rebuild_evaluated(&e, args)?;
        }
    }
    use crate::functions::elementary::complexes::{as_real_imag_with, im, re};
    let (real, imaginary) = match as_real_imag_with(&e, true, None, Some(hints))? {
        Some(parts) => parts,
        None => (re(&e)?, im(&e)?),
    };
    let new: Basic = add(&[real, mul(&[Basic::imaginary_unit(), imaginary])?])?;
    if new != e {
        return Ok(new);
    }
    Ok(e)
}

// ---------------------------------------------------------------------
// derivatives (`Derivative`, `Subs`, `diff`)

/// `Tuple(*items)`.
pub fn tuple(items: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, items)
}

/// The unevaluated `Derivative(expr, (s1, n1), ...)`, consecutive counts
/// of the same variable merged (`_merge_counts`).
pub fn derivative_node(expr: &Basic, counts: &[(Basic, u64)]) -> Basic {
    let mut merged: Vec<(Basic, u64)> = Vec::new();
    for (s, n) in counts {
        match merged.last_mut() {
            Some((last, m)) if last == s => *m += n,
            _ => merged.push((s.clone(), *n)),
        }
    }
    let mut args: Vec<Basic> = vec![expr.clone()];
    for (s, n) in merged {
        args.push(tuple(vec![s, Basic::integer(Int::from(n))]));
    }
    Basic::new(Head::Derivative, args)
}

/// The variables and counts of a `Derivative` node.
pub fn derivative_counts(e: &Basic) -> Vec<(Basic, u64)> {
    e.args()[1..]
        .iter()
        .map(|pair| {
            let n: u64 = u64::try_from(pair.args()[1].as_rational().expect("a count").numerator()).expect("a count");
            (pair.args()[0].clone(), n)
        })
        .collect()
}

/// The unevaluated `Subs(expr, variables, point)`.
pub fn subs_node(expr: &Basic, variables: Vec<Basic>, point: Vec<Basic>) -> Basic {
    Basic::new(Head::Subs, vec![expr.clone(), tuple(variables), tuple(point)])
}

/// `expr` with the variables replaced by the points, a `Subs` kept around
/// the variables of a derivative which are not replaced by symbols
/// (`_substituted`).
pub fn substituted(expr: &Basic, variables: &[Basic], point: &[Basic]) -> PolyResult<Basic> {
    let derivatives: Vec<Basic> = expr
        .preorder()
        .into_iter()
        .filter(|n| matches!(n.head(), Head::Derivative))
        .collect();
    let free_symbols = expr.free_symbols();
    let mut blocked: Vec<(Basic, Basic)> = Vec::new();
    let mut free: Vec<(Basic, Basic)> = Vec::new();
    for (v, p) in variables.iter().zip(point) {
        let differentiated: bool = derivatives
            .iter()
            .any(|d| derivative_counts(d).iter().any(|(s, _)| s == v));
        let capturing: bool = !p.is_symbol() || (free_symbols.contains(p) && p != v);
        if differentiated && capturing {
            blocked.push((v.clone(), p.clone()));
        } else {
            free.push((v.clone(), p.clone()));
        }
    }
    let inner: Basic = if free.is_empty() {
        expr.clone()
    } else {
        super::subs::subs_pairs(expr, &free)?
    };
    if blocked.is_empty() {
        return Ok(inner);
    }
    let (vs, ps): (Vec<Basic>, Vec<Basic>) = blocked.into_iter().unzip();
    Ok(subs_node(&inner, vs, ps))
}

fn is_zero_rational(e: &Basic) -> bool {
    e.as_rational().is_some_and(|v| v.is_zero())
}

/// The derivative of `e` with respect to the symbol `s` (`_derivative`).
pub fn derivative(e: &Basic, s: &Basic) -> PolyResult<Basic> {
    if e == s {
        return Ok(Basic::one());
    }
    if !e.free_symbols().contains(s) {
        return Ok(Basic::zero());
    }
    match e.head() {
        Head::Add => {
            let mut terms: Vec<Basic> = Vec::new();
            for t in e.args() {
                terms.push(derivative(t, s)?);
            }
            add(&terms)
        }
        Head::Mul => {
            let mut terms: Vec<Basic> = Vec::new();
            for (i, f) in e.args().iter().enumerate() {
                let d: Basic = derivative(f, s)?;
                if is_zero_rational(&d) {
                    continue;
                }
                let mut factors: Vec<Basic> = e.args()[..i].to_vec();
                factors.push(d);
                factors.extend(e.args()[i + 1..].iter().cloned());
                terms.push(mul(&factors)?);
            }
            add(&terms)
        }
        Head::Pow => {
            let (b, ex) = (&e.args()[0], &e.args()[1]);
            if !ex.free_symbols().contains(s) {
                let inner: Basic = mul(&[ex.clone(), derivative(b, s)?])?;
                return mul(&[inner, e.clone(), pow(b, &Basic::negative_one())?]);
            }
            let log_b: Basic = crate::functions::elementary::exponential::log(b)?;
            let first: Basic = mul(&[derivative(ex, s)?, log_b])?;
            let second: Basic = mul(&[ex.clone(), derivative(b, s)?, pow(b, &Basic::negative_one())?])?;
            mul(&[e.clone(), add(&[first, second])?])
        }
        Head::Function(f) => {
            if let Some(own) = crate::functions::eval_derivative(*f, e, s)? {
                return Ok(own);
            }
            let mut chain: Vec<Basic> = Vec::new();
            for (i, a) in e.args().iter().enumerate() {
                let da: Basic = derivative(a, s)?;
                if !is_zero_rational(&da) {
                    chain.push(mul(&[crate::functions::fdiff(*f, e, i + 1)?, da])?);
                }
            }
            add(&chain)
        }
        Head::Undefined(u) => {
            let depending: usize = e.args().iter().filter(|a| a.free_symbols().contains(s)).count();
            let mut terms: Vec<Basic> = Vec::new();
            for (i, a) in e.args().iter().enumerate() {
                if !a.free_symbols().contains(s) {
                    continue;
                }
                if a.is_symbol() && depending <= 1 {
                    terms.push(derivative_node(e, &[(a.clone(), 1)]));
                    continue;
                }
                let xi: Basic = super::Dummy::new(&format!("xi_{}", i + 1), &[]).0;
                let mut args: Vec<Basic> = e.args().to_vec();
                args[i] = xi.clone();
                let inner: Basic = u.apply(args)?;
                let at: Basic = substituted(
                    &derivative_node(&inner, &[(xi.clone(), 1)]),
                    &[xi],
                    std::slice::from_ref(a),
                )?;
                terms.push(mul(&[at, derivative(a, s)?])?);
            }
            add(&terms)
        }
        Head::Piecewise => crate::functions::elementary::piecewise::map_expressions(e, &|a: &Basic| derivative(a, s)),
        Head::Derivative => {
            let mut counts: Vec<(Basic, u64)> = derivative_counts(e);
            counts.push((s.clone(), 1));
            Ok(derivative_node(&e.args()[0], &counts))
        }
        Head::Subs => {
            let (expr, variables, point) = (&e.args()[0], e.args()[1].args(), e.args()[2].args());
            let mut terms: Vec<Basic> = Vec::new();
            if expr.free_symbols().contains(s) && !variables.contains(s) {
                terms.push(substituted(&derivative(expr, s)?, variables, point)?);
            }
            for (v, p) in variables.iter().zip(point) {
                let dp: Basic = derivative(p, s)?;
                if is_zero_rational(&dp) {
                    continue;
                }
                terms.push(mul(&[substituted(&derivative(expr, v)?, variables, point)?, dp])?);
            }
            add(&terms)
        }
        _ => Ok(derivative_node(e, &[(s.clone(), 1)])),
    }
}

/// The derivative of `e` with respect to the symbols with their counts
/// (`diff(e, (x, 2), y)`).
pub fn diff(e: &Basic, counts: &[(Basic, u64)]) -> PolyResult<Basic> {
    let mut result: Basic = e.clone();
    for (s, n) in counts {
        for _ in 0..*n {
            result = derivative(&result, s)?;
        }
    }
    Ok(result)
}

/// The tree evaluated bottom up with the evaluating functions (`doit`).
pub fn doit(e: &Basic) -> PolyResult<Basic> {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(doit(a)?);
    }
    rebuild_evaluated(e, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::basic::UndefinedFunction;
    use crate::core::printing::sstr;
    use crate::core::Symbol;
    use std::sync::Arc;

    #[test]
    fn docstring_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        let cube: Basic = pow(&x, &Basic::int(3)).unwrap();
        assert_eq!(show(diff(&cube, &[(x.clone(), 1)])), "3*x**2");
        assert_eq!(
            show(diff(&pow(&x, &Basic::half()).unwrap(), &[(x.clone(), 1)])),
            "1/(2*sqrt(x))"
        );
        let f = Arc::new(UndefinedFunction::new("f", None, &[]));
        let fx: Basic = f.apply(vec![x.clone()]).unwrap();
        assert_eq!(
            show(diff(&mul(&[x.clone(), fx]).unwrap(), &[(x.clone(), 1)])),
            "x*Derivative(f(x), x) + f(x)"
        );
        let fx2: Basic = f.apply(vec![pow(&x, &Basic::int(2)).unwrap()]).unwrap();
        assert_eq!(
            show(diff(&fx2, &[(x.clone(), 1)])),
            "2*x*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, x**2)"
        );
        let two_x: Basic = pow(&Basic::int(2), &x).unwrap();
        assert_eq!(show(diff(&two_x, &[(x.clone(), 1)])), "2**x*log(2)");
    }
}
