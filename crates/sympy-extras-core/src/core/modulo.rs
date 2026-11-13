//! Port of `sympy_extras/core/mod.py`: the modulo [`Function::Mod`] and
//! its evaluating function [`modulo`] (SymPy's rules: numbers reduced with
//! the sign of `q`, `Mod(p, p)` and `Mod(0, q)` zero, an integer modulo 1
//! zero, a common factor taken out, the rational terms of `p` reduced, a
//! negative integer coefficient reduced, nested moduli collapsed, the
//! product of two integers differing by an odd integer even). `Mod` has no
//! numerical value of its own here, as in Python (where SymPy gives it).

use crate::error::{zero_division, PolyResult};
use crate::numbers::{denom, numer, Int, Rat};

use super::add::add;
use super::assumptions::{facts_of, FactList, Predicate};
use super::basic::{Basic, Function, Head};
use super::expr::{as_base_exp, as_coeff_mul, is_true};
use super::mul::{div, mul};
use super::parameters::evaluate;
use super::power::pow;

fn node(p: &Basic, q: &Basic) -> Basic {
    Basic::new(Head::Function(Function::Mod), vec![p.clone(), q.clone()])
}

fn number(r: &Rat) -> PolyResult<Basic> {
    if denom(r) == Int::ONE {
        return Ok(Basic::integer(numer(r)));
    }
    Basic::rational(numer(r), denom(r))
}

fn floor_rat(r: &Rat) -> Int {
    let (p, q) = (numer(r), denom(r));
    let mut f: Int = &p / &q;
    if &p % &q != Int::ZERO && p < Int::ZERO {
        f -= Int::ONE;
    }
    f
}

/// `a - b*floor(a/b)` (Python's `%` on fractions).
fn rat_mod(a: &Rat, b: &Rat) -> Rat {
    a - b * Rat::from(floor_rat(&(a / b)))
}

fn gcd(a: &Int, b: &Int) -> Int {
    let (mut a, mut b) = (
        if *a < Int::ZERO { -a.clone() } else { a.clone() },
        if *b < Int::ZERO { -b.clone() } else { b.clone() },
    );
    while b != Int::ZERO {
        let r: Int = &a % &b;
        a = b;
        b = r;
    }
    a
}

fn lcm(a: &Int, b: &Int) -> Int {
    if *a == Int::ZERO || *b == Int::ZERO {
        return Int::ZERO;
    }
    let g: Int = gcd(a, b);
    let p: Int = a * b / g;
    if p < Int::ZERO {
        -p
    } else {
        p
    }
}

fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

fn factors_of(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// The numerical coefficient as a rational (`1` for a float or a number
/// with no exact value) (`_coefficient`).
fn coefficient(e: &Basic) -> Rat {
    let (c, _) = as_coeff_mul(e);
    c.as_rational().cloned().unwrap_or(Rat::ONE)
}

fn content(terms: &[Basic]) -> Rat {
    let mut result: Rat = Rat::ZERO;
    for t in terms {
        let c: Rat = coefficient(t);
        if c == Rat::ZERO {
            return Rat::ONE;
        }
        result = if result != Rat::ZERO {
            Rat::from(gcd(&numer(&result), &numer(&c))) / Rat::from(lcm(&denom(&result), &denom(&c)))
        } else if c < Rat::ZERO {
            -c
        } else {
            c
        };
    }
    if result == Rat::ZERO {
        Rat::ONE
    } else {
        result
    }
}

/// The positive integer power of `base` dividing `e` (`_exponent_of`).
fn exponent_of(base: &Basic, e: &Basic) -> PolyResult<Int> {
    if e.is_add() {
        return Ok(Int::ZERO);
    }
    for f in factors_of(e) {
        let (b, exponent) = as_base_exp(&f)?;
        if b == *base {
            if let Some(v) = exponent.as_rational() {
                if denom(v) == Int::ONE && numer(v) > Int::ZERO {
                    return Ok(numer(v));
                }
            }
        }
    }
    Ok(Int::ZERO)
}

/// A common factor of `a` and `b`, `None` when it is 1 (`_common_factor`).
fn common_factor(a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    let ca: Rat = if a.is_add() {
        content(&terms_of(a))
    } else {
        coefficient(a)
    };
    let cb: Rat = if b.is_add() {
        content(&terms_of(b))
    } else {
        coefficient(b)
    };
    let common: Rat = if denom(&ca) == Int::ONE && denom(&cb) == Int::ONE {
        Rat::from(gcd(&numer(&ca), &numer(&cb)))
    } else {
        Rat::ONE
    };
    let mut factors: Vec<Basic> = Vec::new();
    if !a.is_add() {
        for f in factors_of(a) {
            if f.is_number_node() {
                continue;
            }
            let (base, exponent) = as_base_exp(&f)?;
            let Some(v) = exponent.as_rational() else { continue };
            if !(denom(v) == Int::ONE && numer(v) > Int::ZERO) {
                continue;
            }
            let other: Int = exponent_of(&base, b)?;
            if other > Int::ZERO {
                let k: Int = if numer(v) < other { numer(v) } else { other };
                factors.push(pow(&base, &Basic::integer(k))?);
            }
        }
    }
    if common == Rat::ONE && factors.is_empty() {
        return Ok(None);
    }
    if common == Rat::ONE {
        return Ok(Some(mul(&factors)?));
    }
    let mut items: Vec<Basic> = vec![number(&common)?];
    items.extend(factors);
    Ok(Some(mul(&items)?))
}

fn divide(e: &Basic, factor: &Basic) -> PolyResult<Basic> {
    if e.is_add() {
        let mut terms: Vec<Basic> = Vec::new();
        for t in e.args() {
            terms.push(div(t, factor)?);
        }
        return add(&terms);
    }
    div(e, factor)
}

/// The rational terms of `a` reduced modulo a rational `b`, a negative
/// integer coefficient of an integer `a` reduced (`_reduce_terms`).
fn reduce_terms(a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    let Some(vb) = b.as_rational().cloned() else {
        if a.is_add() {
            let mut kept: Vec<Basic> = Vec::new();
            for t in a.args() {
                if !is_true(&div(t, b)?, Predicate::integer) {
                    kept.push(t.clone());
                }
            }
            if kept.len() < a.args().len() {
                return Ok(Some(modulo(&add(&kept)?, b)?));
            }
        }
        return Ok(None);
    };
    let integral_b: bool = denom(&vb) == Int::ONE;
    if a.is_add() {
        let mut changed: bool = false;
        let mut new_terms: Vec<Basic> = Vec::new();
        for t in a.args() {
            if let Some(vt) = t.as_rational() {
                let reduced: Rat = rat_mod(vt, &vb);
                if reduced != *vt {
                    changed = true;
                }
                new_terms.push(number(&reduced)?);
            } else if integral_b
                && is_true(t, Predicate::integer)
                && denom(&coefficient(t)) == Int::ONE
                && rat_mod(&coefficient(t), &vb) == Rat::ZERO
            {
                changed = true;
            } else {
                new_terms.push(t.clone());
            }
        }
        if changed {
            return Ok(Some(modulo(&add(&new_terms)?, b)?));
        }
        return Ok(None);
    }
    if integral_b && is_true(a, Predicate::integer) {
        let (c, rest) = as_coeff_mul(a);
        if let Some(vc) = c.as_rational() {
            if denom(vc) == Int::ONE && numer(vc) < Int::ZERO {
                let reduced: Rat = rat_mod(vc, &vb);
                if reduced < Rat::ZERO {
                    // a negative modulus keeps the coefficient negative: Python
                    // applies this rule again without end
                    return Err(crate::error::PolyError::Recursion(
                        "maximum recursion depth exceeded".to_string(),
                    ));
                }
                return Ok(Some(modulo(&mul(&[number(&reduced)?, rest])?, b)?));
            }
        }
    }
    Ok(None)
}

/// A product of integers with two factors differing by an odd integer
/// (`_has_consecutive_factors`).
fn has_consecutive_factors(a: &Basic) -> PolyResult<bool> {
    if !a.is_mul() {
        return Ok(false);
    }
    let factors: Vec<Basic> = factors_of(a);
    if factors.iter().any(|f| !is_true(f, Predicate::integer)) {
        return Ok(false);
    }
    for (i, f) in factors.iter().enumerate() {
        for g in &factors[i + 1..] {
            let d: Basic = super::add::sub(g, f)?;
            if let Some(v) = d.as_rational() {
                if denom(v) == Int::ONE && (&numer(v) % Int::from(2)) != Int::ZERO {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// The evaluated `p` modulo `q` (`mod`).
pub fn modulo(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(a, b));
    }
    let exact = |e: &Basic| e.as_rational().cloned();
    if b.is_number_node() && exact(b).is_some_and(|v| v == Rat::ZERO) {
        return Err(zero_division("Modulo by zero"));
    }
    if a.is_number_node() && b.is_number_node() {
        return match (exact(a), exact(b)) {
            (Some(va), Some(vb)) => number(&rat_mod(&va, &vb)),
            _ => Ok(Basic::nan()),
        };
    }
    if (a.is_number_node() && exact(a).is_none()) || (b.is_number_node() && exact(b).is_none()) {
        return Ok(Basic::nan());
    }
    if a == b {
        return Ok(Basic::zero());
    }
    if a.is_number_node() && exact(a).is_some_and(|v| v == Rat::ZERO) {
        return Ok(Basic::zero());
    }
    if exact(b).is_some_and(|v| v == Rat::ONE) && is_true(a, Predicate::integer) {
        return Ok(Basic::zero());
    }
    if matches!(a.head(), Head::Function(Function::Mod)) {
        let (inner_p, inner_q) = (&a.args()[0], &a.args()[1]);
        if let (Some(vi), Some(vb)) = (exact(inner_q), exact(b)) {
            let ratio: Rat = &vi / &vb;
            if denom(&ratio) == Int::ONE {
                return modulo(inner_p, b);
            }
            if numer(&ratio) == Int::ONE || numer(&ratio) == -Int::ONE {
                return Ok(a.clone());
            }
        }
    }
    if let Some(common) = common_factor(a, b)? {
        return mul(&[common.clone(), modulo(&divide(a, &common)?, &divide(b, &common)?)?]);
    }
    if let Some(reduced) = reduce_terms(a, b)? {
        return Ok(reduced);
    }
    if exact(b).is_some_and(|v| v == Rat::from(Int::from(2))) && has_consecutive_factors(a)? {
        return Ok(Basic::zero());
    }
    Ok(node(a, b))
}

/// The facts (`_eval_facts`).
pub fn modulo_facts(args: &[Basic]) -> FactList {
    let (p, q) = (facts_of(&args[0]), facts_of(&args[1]));
    let yes = |f: &super::assumptions::Facts, k: Predicate| f.get(&k) == Some(&true);
    let mut facts: FactList = FactList::new();
    if yes(p, Predicate::commutative) && yes(q, Predicate::commutative) {
        facts.set(Predicate::commutative, true);
    }
    if yes(p, Predicate::integer) && yes(q, Predicate::integer) {
        facts.set(Predicate::integer, true);
    }
    if yes(q, Predicate::real) {
        facts.set(Predicate::real, true);
    }
    if yes(q, Predicate::positive) {
        facts.set(Predicate::nonnegative, true);
    }
    if yes(q, Predicate::negative) {
        facts.set(Predicate::nonpositive, true);
    }
    if yes(p, Predicate::finite) && yes(q, Predicate::finite) {
        facts.set(Predicate::finite, true);
    }
    facts
}

/// The derivative in the argument `argindex` (`Application.fdiff`):
/// `Derivative(Mod(...), x)` for a symbol argument found in no other.
pub fn modulo_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let a: &Basic = &args[argindex - 1];
    let elsewhere: bool = args
        .iter()
        .enumerate()
        .any(|(i, v)| i != argindex - 1 && v.free_symbols().contains(a));
    if a.is_symbol() && !elsewhere {
        return Ok(super::function::derivative_node(e, &[(a.clone(), 1)]));
    }
    Err(crate::error::not_ported("a derivative of Mod as a Subs"))
}
