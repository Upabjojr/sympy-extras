//! Port of the evaluation of `sympy_extras/core/mul.py`: `mul` (SymPy's
//! `Mul.flatten`: the nested products opened, the numbers multiplied, the
//! powers of the same base collected, the rational powers of numbers paired
//! and grouped, the infinities absorbing the factors of known sign, a number
//! distributed over a single sum), `div`, `neg` and `keep_coeff`.
//!
//! The order terms, the accumulation bounds and the products with an
//! `exp(...)` factor of the Python function are not in the Rust core yet.

use crate::error::PolyResult;
use crate::numbers::*;

use super::add::add;
use super::assumptions::Predicate;
use super::basic::{Basic, Head};
use super::expr::{as_base_exp, as_coeff_mul, is, is_false, is_integer_class, is_rational_value, is_true};
use super::numbers::{number_add, number_mul};
use super::parameters::{distribute, evaluate};
use super::power::{minus_one_pow, pow};

/// The most passes of flattening of one product (`_MAX_PASSES`).
const MAX_PASSES: usize = 8;

/// `Mul(*factors)` without evaluation, one factor as itself, none as `1`.
pub fn raw_mul(factors: Vec<Basic>) -> Basic {
    match factors.len() {
        0 => Basic::one(),
        1 => factors.into_iter().next().expect("one factor"),
        _ => Basic::new(Head::Mul, factors),
    }
}

/// The factors with the nested products opened, in order.
fn flattened(args: &[Basic]) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    for a in args {
        if a.is_mul() {
            result.extend(flattened(a.args()));
        } else {
            result.push(a.clone());
        }
    }
    result
}

fn is_one(n: &Basic) -> bool {
    is_rational_value(n, 1)
}

fn is_integer_one(e: &Basic) -> bool {
    is_integer_class(e) && is_one(e)
}

/// The evaluated product of the arguments.
pub fn mul(args: &[Basic]) -> PolyResult<Basic> {
    let mut factors: Vec<Basic> = args.to_vec();
    if !evaluate() {
        return Ok(raw_mul(flattened(&factors)));
    }
    // SymPy's AssocOp.__new__: the identities dropped, a single argument
    // returned as it is
    let nonidentity: Vec<&Basic> = factors.iter().filter(|f| !is_integer_one(f)).collect();
    if nonidentity.len() == 1 {
        return Ok(nonidentity[0].clone());
    }
    if flattened(&factors).iter().any(super::accumbounds::is_bounds) {
        return super::accumbounds::product_with_bounds(&factors);
    }
    if let Some(result) = two_argument_rule(&factors)? {
        return Ok(result);
    }
    for _ in 0..MAX_PASSES {
        let (result, again) = mul_pass(&factors)?;
        match again {
            None => return Ok(result),
            Some(next) => factors = next,
        }
    }
    Ok(raw_mul(factors))
}

/// SymPy's rule for exactly two factors (the factors `1` dropped first), a
/// nonzero rational `a` and `b = r*s` with `s` a sum: `a*r*s` stays an
/// undistributed product when `r != 1`, and `a` is distributed over `s`
/// when `r = 1` (`Mul.flatten`, the 2-arg hack).
fn two_argument_rule(factors: &[Basic]) -> PolyResult<Option<Basic>> {
    let rest: Vec<&Basic> = factors.iter().filter(|f| !is_integer_one(f)).collect();
    if rest.len() != 2 {
        return Ok(None);
    }
    let (mut a, mut b) = (rest[0], rest[1]);
    if b.as_rational().is_some() {
        std::mem::swap(&mut a, &mut b);
    }
    match a.as_rational() {
        Some(v) if !v.is_zero() => {}
        _ => return Ok(None),
    }
    let (r, s) = as_coeff_mul(b);
    if !s.is_add() {
        return Ok(None);
    }
    if !is_one(&r) {
        let ar: Basic = number_mul(a, &r);
        return Ok(Some(if is_one(&ar) {
            s
        } else {
            Basic::new(Head::Mul, vec![ar, s])
        }));
    }
    if distribute() && !is_false(&s, Predicate::commutative) {
        let mut terms: Vec<Basic> = Vec::new();
        for t in s.args() {
            terms.push(keep_coeff(a, t)?);
        }
        return Ok(Some(add(&terms)?));
    }
    Ok(None)
}

/// A map in the order of insertion (a Python dict).
struct Ordered<K, V> {
    keys: Vec<K>,
    values: Vec<V>,
}

impl<K: PartialEq, V> Ordered<K, V> {
    fn new() -> Self {
        Ordered {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn position(&self, key: &K) -> Option<usize> {
        self.keys.iter().position(|k| k == key)
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.position(key).map(|i| &self.values[i])
    }

    fn entry(&mut self, key: K, default: impl FnOnce() -> V) -> &mut V {
        let i: usize = match self.position(&key) {
            Some(i) => i,
            None => {
                self.keys.push(key);
                self.values.push(default());
                self.keys.len() - 1
            }
        };
        &mut self.values[i]
    }
}

/// The commutative factors during flattening, keyed by base and by the
/// non-numerical part of the exponent: the numerical coefficients of the
/// exponents, and the factor itself.
type Collected = Ordered<(Basic, Basic), (Vec<Basic>, Basic)>;

/// One pass of flattening and collecting; the second component is the list
/// of factors to flatten again when a rebuilt factor is a product or a
/// number.
fn mul_pass(args: &[Basic]) -> PolyResult<(Basic, Option<Vec<Basic>>)> {
    let mut coefficient: Basic = Basic::one();
    let mut collected: Collected = Ordered::new();
    let mut noncommutative: Vec<Basic> = Vec::new();
    for a in flattened(args) {
        if a.is_number_node() {
            coefficient = number_mul(&coefficient, &a);
            continue;
        }
        if is_false(&a, Predicate::commutative) {
            noncommutative.push(a);
            continue;
        }
        let (base, exponent) = as_base_exp(&a)?;
        let (c, rest) = as_coeff_mul(&exponent);
        let key = (base, rest);
        if let Some(i) = collected.position(&key) {
            collected.values[i].0.push(c);
        } else {
            collected.keys.push(key);
            collected.values.push((vec![c], a));
        }
    }
    if matches!(coefficient.head(), Head::NaN) {
        return Ok((Basic::nan(), None));
    }
    // the commutative factors rebuilt: positive numerical bases with the
    // same exponent are multiplied together
    let mut rebuilt: Vec<Basic> = Vec::new();
    let mut numerical: Ordered<Basic, Vec<Rat>> = Ordered::new();
    let mut rebased: bool = false;
    for ((base, rest), (cs, factor)) in collected.keys.iter().zip(&collected.values) {
        let mut total: Basic = cs[0].clone();
        for c in &cs[1..] {
            total = number_add(&total, c);
        }
        let exponent: Basic = if is_one(&total) {
            rest.clone()
        } else {
            mul(&[total, rest.clone()])?
        };
        if let Some(b) = base.as_rational() {
            if groupable(b, &exponent, factor) {
                let (mut b, mut x) = (b.clone(), exponent.clone());
                if !rat_is_negative(&b) && !b.is_zero() {
                    if let Some(xv) = x.as_rational() {
                        if rat_is_negative(xv) {
                            b = recip(&b);
                            x = Basic::from_rational(-xv);
                        }
                    }
                }
                numerical.entry(x, Vec::new).push(b);
                continue;
            }
        }
        if cs.len() == 1 {
            // SymPy's _gather rebuilds b**(c*t) with c*t evaluated
            if factor.is_pow() && as_base_exp(factor)?.1 != exponent {
                rebuilt.push(pow(base, &exponent)?);
            } else if matches!(base.head(), Head::Exp1) && !factor.args().is_empty() {
                rebuilt.push(crate::functions::elementary::exponential::exp(&exponent)?);
            } else {
                rebuilt.push(factor.clone());
            }
            continue;
        }
        let combined: Basic = pow(base, &exponent)?;
        if as_base_exp(&combined)?.0 != *base {
            rebased = true;
        }
        rebuilt.push(combined);
    }
    if let Some((pairs, rest)) = pair_radicals(&numerical)? {
        numerical = rest;
        rebased = true; // the products of the pairs are flattened again
        rebuilt.extend(pairs);
    }
    for (exponent, bases) in numerical.keys.iter().zip(&numerical.values) {
        if bases.len() == 1 {
            rebuilt.push(single_power(&collected, &bases[0], exponent)?);
            continue;
        }
        // the bases with the same exponent multiplied
        let negatives: usize = bases.iter().filter(|b| rat_is_negative(b)).count();
        let mut product: Rat = Rat::ONE;
        for b in bases {
            product *= if rat_is_negative(b) { -b } else { b.clone() };
        }
        let product_node: Basic = Basic::from_rational(product.clone());
        if negatives == 0 {
            rebuilt.push(pow(&product_node, exponent)?);
        } else if negatives == 1 {
            rebuilt.push(pow(&Basic::from_rational(-product), exponent)?);
        } else if !exponent.is_number_node() {
            // an integer symbolic exponent: (-1)**(k*m) is 1 for an even k
            let signed: Rat = if negatives % 2 == 0 { product } else { -product };
            rebuilt.push(pow(&Basic::from_rational(signed), exponent)?);
        } else {
            let Some(x) = exponent.as_rational() else {
                return Err(crate::error::value_error("a rational exponent is expected".to_string()));
            };
            rebuilt.push(minus_one_pow(&(rat(negatives as i64, 1) * x)));
            rebuilt.push(pow(&product_node, exponent)?);
        }
    }
    // adjacent noncommutative powers of the same base are merged
    let mut merged: Vec<Basic> = Vec::new();
    for f in noncommutative {
        if let Some(last) = merged.last() {
            let (b1, e1) = as_base_exp(last)?;
            let (b2, e2) = as_base_exp(&f)?;
            if b1 == b2 && as_coeff_mul(&e1).1 == as_coeff_mul(&e2).1 {
                let combined: Basic = pow(&b1, &add(&[e1, e2])?)?;
                let n: usize = merged.len();
                merged[n - 1] = combined;
                continue;
            }
        }
        merged.push(f);
    }
    let again: bool = rebuilt.iter().chain(&merged).any(|f| f.is_mul() || f.is_number_node());
    if again || rebased {
        let mut next: Vec<Basic> = vec![coefficient];
        next.extend(rebuilt);
        next.extend(merged);
        return Ok((Basic::zero(), Some(next)));
    }
    match coefficient.head() {
        Head::Infinity | Head::NegativeInfinity => {
            // oo absorbs the factors of known sign
            let mut kept: Vec<Basic> = Vec::new();
            for f in rebuilt {
                if is_true(&f, Predicate::positive) {
                    continue;
                }
                if is_true(&f, Predicate::negative) {
                    coefficient = number_mul(&coefficient, &Basic::negative_one());
                    continue;
                }
                kept.push(f);
            }
            rebuilt = kept;
        }
        Head::ComplexInfinity => {
            // zoo absorbs the nonzero factors known real or known non-real
            rebuilt.retain(|f| !(is_false(f, Predicate::zero) && is(f, Predicate::extended_real).is_some()));
        }
        _ => {}
    }
    rebuilt.sort_by(|a, b| a.compare_key().cmp(b.compare_key()));
    if rebuilt.is_empty() && merged.is_empty() {
        return Ok((coefficient, None));
    }
    if coefficient.as_rational().is_some_and(|v| v.is_zero()) {
        // SymPy's Mul.flatten: zero times a factor known to be infinite is nan
        if rebuilt.iter().chain(&merged).any(|f| is_false(f, Predicate::finite)) {
            return Ok((Basic::nan(), None));
        }
        // SymPy distributes a rational coefficient over a single sum first
        let others: Vec<&Basic> = rebuilt.iter().chain(&merged).collect();
        if others.len() == 1
            && others[0].is_add()
            && distribute()
            && others[0].args().iter().any(|t| {
                matches!(
                    t.head(),
                    Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity
                ) || is_false(t, Predicate::finite)
            })
        {
            return Ok((Basic::nan(), None));
        }
        return Ok((Basic::zero(), None));
    }
    let one: bool = is_one(&coefficient);
    if one && rebuilt.len() + merged.len() == 1 {
        let only: Basic = rebuilt.into_iter().chain(merged).next().expect("one factor");
        return Ok((only, None));
    }
    if !one && rebuilt.len() == 1 && merged.is_empty() && coefficient.is_finite_number() && distribute() {
        let only: &Basic = &rebuilt[0];
        if only.is_add() {
            let mut terms: Vec<Basic> = Vec::new();
            for t in only.args() {
                terms.push(mul(&[coefficient.clone(), t.clone()])?);
            }
            return Ok((add(&terms)?, None));
        }
    }
    let mut result: Vec<Basic> = if one { Vec::new() } else { vec![coefficient] };
    result.extend(rebuilt);
    result.extend(merged);
    Ok((Basic::new(Head::Mul, result), None))
}

/// SymPy's combination of rational powers of positive integers with
/// different exponents through the gcd of their bases (`sqrt(2)*6**(1/4)`
/// is `2**(3/4)*3**(1/4)`): the new factors and the groups left alone, or
/// `None` when nothing combines (`Mul.flatten`, the gcd pairing of
/// `num_rat`).
// the loop rewrites `num_rat[j]` as Python's does
#[allow(clippy::type_complexity, clippy::needless_range_loop)]
fn pair_radicals(numerical: &Ordered<Basic, Vec<Rat>>) -> PolyResult<Option<(Vec<Basic>, Ordered<Basic, Vec<Rat>>)>> {
    let candidates: Vec<usize> = (0..numerical.keys.len())
        .filter(|&i| {
            numerical.keys[i].as_rational().is_some_and(|e| denom(e) > Int::ONE)
                && numerical.values[i]
                    .iter()
                    .all(|b| !rat_is_negative(b) && !b.is_zero() && b.is_int())
        })
        .collect();
    if candidates.len() < 2 {
        return Ok(None);
    }
    let mut num_rat: Vec<(Int, Rat)> = Vec::new();
    for &i in &candidates {
        let mut product: Int = Int::ONE;
        for b in &numerical.values[i] {
            product *= numer(b);
        }
        num_rat.push((product, numerical.keys[i].as_rational().expect("a rational").clone()));
    }
    let mut changed: bool = false;
    let mut factors: Vec<Basic> = Vec::new();
    let mut pnew: Ordered<Rat, Vec<Int>> = Ordered::new();
    let mut i: usize = 0;
    while i < num_rat.len() {
        let (mut bi, ei) = num_rat[i].clone();
        if bi == Int::ONE {
            i += 1;
            continue;
        }
        let mut grow: Vec<(Int, Rat)> = Vec::new();
        for j in i + 1..num_rat.len() {
            let (bj, ej) = num_rat[j].clone();
            let g: Int = gcd(&bi, &bj);
            if g != Int::ONE {
                changed = true;
                let mut es: Rat = &ei + &ej;
                if es.is_int() {
                    factors.push(pow(&Basic::integer(g.clone()), &Basic::integer(numer(&es)))?);
                } else {
                    let (n, d) = (numer(&es), denom(&es));
                    if n > d {
                        let (e_i, ep) = div_mod_floor(&n, &d);
                        factors.push(pow(&Basic::integer(g.clone()), &Basic::integer(e_i))?);
                        es = rat_from_parts(ep, d);
                    }
                    grow.push((g.clone(), es));
                }
                num_rat[j] = (div_floor(&bj, &g), ej);
                bi = div_floor(&bi, &g);
                if bi == Int::ONE {
                    break;
                }
            }
        }
        if bi != Int::ONE {
            pnew.entry(ei.clone(), Vec::new).push(bi);
        }
        num_rat.extend(grow);
        i += 1;
    }
    if !changed {
        return Ok(None);
    }
    let mut rest: Ordered<Basic, Vec<Rat>> = Ordered::new();
    for k in 0..numerical.keys.len() {
        if !candidates.contains(&k) {
            rest.keys.push(numerical.keys[k].clone());
            rest.values.push(numerical.values[k].clone());
        }
    }
    for (ef, bs) in pnew.keys.iter().zip(&pnew.values) {
        let mut base_product: Int = Int::ONE;
        for b in bs {
            base_product *= b;
        }
        factors.push(pow(&Basic::integer(base_product), &Basic::from_rational(ef.clone()))?);
    }
    Ok(Some((factors, rest)))
}

/// Whether the numerical `base**exponent` may be multiplied with the other
/// numerical bases of the same exponent.
fn groupable(base: &Rat, exponent: &Basic, factor: &Basic) -> bool {
    if !rat_is_negative(base) && !base.is_zero() {
        if let Some(e) = exponent.as_rational() {
            return denom(e) > Int::ONE;
        }
        return !exponent.is_number_node();
    }
    if rat_is_negative(base) && factor.is_pow() {
        if !exponent.is_number_node() && is_true(exponent, Predicate::integer) {
            return true; // SymPy: (-1)**m*2**m is (-2)**m for an integer m
        }
        return exponent
            .as_rational()
            .is_some_and(|e| denom(e) > int(2) && !rat_is_negative(e) && !e.is_zero());
    }
    false
}

/// `base**exponent` for a numerical base alone in its exponent group: the
/// factor as given when its exponent did not change.
fn single_power(collected: &Collected, base: &Rat, exponent: &Basic) -> PolyResult<Basic> {
    let base_node: Basic = Basic::from_rational(base.clone());
    let key = (base_node.clone(), as_coeff_mul(exponent).1);
    if let Some((cs, factor)) = collected.get(&key) {
        if cs.len() == 1 && factor.is_pow() && factor.args()[0] == base_node && factor.args()[1] == *exponent {
            return Ok(factor.clone());
        }
    }
    pow(&base_node, exponent)
}

/// The evaluated quotient `a / b`.
pub fn div(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    // the operators of AccumBounds (its _op_priority is the higher)
    if evaluate() && super::accumbounds::is_bounds(a) {
        return super::accumbounds::ab_truediv(a, b);
    }
    if evaluate() && super::accumbounds::is_bounds(b) {
        return super::accumbounds::ab_rtruediv(b, a);
    }
    if is_one(a) {
        return pow(b, &Basic::negative_one());
    }
    mul(&[a.clone(), pow(b, &Basic::negative_one())?])
}

/// The evaluated opposite `-a`.
pub fn neg(a: &Basic) -> PolyResult<Basic> {
    if evaluate() && super::accumbounds::is_bounds(a) {
        // AccumulationBounds.__neg__
        return super::accumbounds::ab_neg(a);
    }
    if a.is_number_node() {
        return mul(&[Basic::negative_one(), a.clone()]);
    }
    if a.is_add() {
        // SymPy's Add.__neg__ distributes
        let mut terms: Vec<Basic> = Vec::new();
        for t in a.args() {
            terms.push(neg(t)?);
        }
        return add(&terms);
    }
    if a.is_mul() {
        // SymPy's Mul.__neg__: no flattening
        let factors: &[Basic] = a.args();
        let first: &Basic = &factors[0];
        if first.is_number_node() {
            let negated: Basic = mul(&[Basic::negative_one(), first.clone()])?;
            if is_rational_value(&negated, 1) && negated.as_rational().is_some_and(|r| denom(r) == Int::ONE) {
                return Ok(raw_mul(factors[1..].to_vec()));
            }
            let mut result: Vec<Basic> = vec![negated];
            result.extend(factors[1..].iter().cloned());
            return Ok(Basic::new(Head::Mul, result));
        }
        let mut result: Vec<Basic> = vec![Basic::negative_one()];
        result.extend(factors.iter().cloned());
        return Ok(Basic::new(Head::Mul, result));
    }
    // SymPy's Expr.__neg__ (Mul._from_args)
    Ok(Basic::new(Head::Mul, vec![Basic::negative_one(), a.clone()]))
}

/// `coeff*factors` without distributing a number over a sum (`2*(x + y)`),
/// the number put first in a product (SymPy's `_keep_coeff`).
pub fn keep_coeff(coeff: &Basic, factors: &Basic) -> PolyResult<Basic> {
    keep_coeff_with(coeff, factors, true, false)
}

/// `_keep_coeff` with its options: with `clear=False` a fraction is
/// distributed over a sum when that gives a term an integer coefficient;
/// with `sign` a coefficient `-1` stays in front of a sum.
pub fn keep_coeff_with(coeff: &Basic, factors: &Basic, clear: bool, sign: bool) -> PolyResult<Basic> {
    let (mut coeff, mut factors) = (coeff.clone(), factors.clone());
    if !coeff.is_number_node() {
        if factors.is_number_node() {
            std::mem::swap(&mut coeff, &mut factors);
        } else {
            return mul(&[coeff, factors]);
        }
    }
    if is_rational_value(&factors, 1) {
        return Ok(coeff);
    }
    if is_rational_value(&coeff, 1) {
        return Ok(factors);
    }
    if is_rational_value(&coeff, -1) && !sign {
        return mul(&[Basic::negative_one(), factors]);
    }
    if factors.is_add() {
        if !clear && coeff.as_rational().is_some_and(|v| !v.is_int()) {
            let mut scaled: Vec<(Basic, Basic)> = Vec::new();
            for t in factors.args() {
                let (c, m) = as_coeff_mul(t);
                scaled.push((keep_coeff(&c, &coeff)?, m));
            }
            if scaled.iter().any(|(c, _)| super::expr::is_integer_class(c)) {
                let terms: Vec<Basic> = scaled
                    .into_iter()
                    .map(|(c, m)| {
                        if is_rational_value(&c, 1) {
                            m
                        } else {
                            Basic::new(Head::Mul, vec![c, m])
                        }
                    })
                    .collect();
                return Ok(Basic::new(Head::Add, terms));
            }
        }
        return Ok(Basic::new(Head::Mul, vec![coeff, factors]));
    }
    if factors.is_mul() {
        let mut margs: Vec<Basic> = factors.args().to_vec();
        if margs[0].is_number_node() {
            let product: Basic = mul(&[margs[0].clone(), coeff])?;
            if is_rational_value(&product, 1) {
                margs.remove(0);
            } else {
                margs[0] = product;
            }
        } else {
            margs.insert(0, coeff);
        }
        return Ok(if margs.len() != 1 {
            Basic::new(Head::Mul, margs)
        } else {
            margs.remove(0)
        });
    }
    let m: Basic = mul(&[coeff.clone(), factors.clone()])?;
    if m.is_number_node() && !factors.is_number_node() {
        return Ok(Basic::new(Head::Mul, vec![coeff, factors]));
    }
    Ok(m)
}
