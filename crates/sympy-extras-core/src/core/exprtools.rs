//! Port of `sympy_extras/core/exprtools.py`, its main part: the terms as
//! `coeff * numer / denom` ([`Factors`], [`Term`], SymPy's), [`gcd_terms`]
//! (a sum over a common denominator with the common factors taken out),
//! [`factor_terms`] (the same in every node), [`decompose_power`] and the
//! `as_content_primitive` of the classes of the core.
//!
//! `factor_nc`, the `Factors` with symbolic exponents of `fu`
//! (`ExprFactors`), the `radical` option (`_common_radicals`) and the sums
//! of `concrete` are not ported yet.

use std::collections::HashMap;

use crate::error::{not_ported, PolyError, PolyResult};
use crate::numbers::*;

use super::add::add;
use super::assumptions::Predicate;
use super::basic::{Basic, Head};
use super::expr::{
    as_base_exp, as_coeff_add, as_coeff_mul, as_coeff_mul_tuple, as_numer_denom, extract_multiplicatively, factors_of,
    is_false, is_integer_class, is_rational_value, primitive,
};
use super::function::rebuild_evaluated;
use super::mul::{keep_coeff, keep_coeff_with, mul};
use super::power::pow;
use super::Dummy;

fn small(n: &Int) -> PolyResult<i64> {
    i64::try_from(n).map_err(|_| not_ported("a huge exponent in exprtools"))
}

/// `(base, n)` with `e = base**n`, `n` an integer (`decompose_power`).
pub fn decompose_power(e: &Basic) -> PolyResult<(Basic, i64)> {
    let (mut base, exponent) = as_base_exp(e)?;
    if exponent.is_number_class() {
        if let Some(x) = exponent.as_rational() {
            if denom(x) != Int::ONE {
                base = pow(&base, &Basic::from_rational(rat_from_parts(Int::ONE, denom(x))))?;
            }
            return Ok((base, small(&numer(x))?));
        }
        return Ok((e.clone(), 1));
    }
    let (c, tail) = as_coeff_mul(&exponent);
    if let Some(v) = c.as_rational() {
        if *v == -Rat::ONE {
            return Ok((pow(&base, &tail)?, -1));
        }
        if *v != Rat::ONE {
            let root: Basic = keep_coeff(&Basic::from_rational(rat_from_parts(Int::ONE, denom(v))), &tail)?;
            return Ok((pow(&base, &root)?, small(&numer(v))?));
        }
    }
    Ok((e.clone(), 1))
}

fn lcm(a: &Int, b: &Int) -> Int {
    if a.is_zero() || b.is_zero() {
        return Int::ZERO;
    }
    abs(&(a / &gcd(a, b) * b))
}

/// SymPy's `Rational.gcd`; `1` for other numbers (`_number_gcd`).
fn number_gcd(a: &Basic, b: &Basic) -> Basic {
    match (a.as_rational(), b.as_rational()) {
        (Some(x), Some(y)) => {
            Basic::from_rational(rat_from_parts(gcd(&numer(x), &numer(y)), lcm(&denom(x), &denom(y))))
        }
        _ => Basic::one(),
    }
}

fn number_lcm(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    if let (Some(x), Some(y)) = (a.as_rational(), b.as_rational()) {
        return Ok(Basic::from_rational(rat_from_parts(
            lcm(&numer(x), &numer(y)),
            gcd(&denom(x), &denom(y)),
        )));
    }
    let product: Basic = mul(&[a.clone(), b.clone()])?;
    Ok(if product.is_number_class() {
        product
    } else {
        Basic::one()
    })
}

fn as_number(e: Basic) -> Basic {
    if e.is_number_class() {
        e
    } else {
        Basic::one()
    }
}

// ---------------------------------------------------------------------
// Factors: an insertion-ordered map (a Python dict)

/// A product of bases with nonzero integer exponents (`Factors`; `I**2` is
/// folded into `-1`).
#[derive(Debug, Clone, PartialEq)]
pub struct Factors {
    pub factors: Vec<(Basic, i64)>,
}

fn position(map: &[(Basic, i64)], key: &Basic) -> Option<usize> {
    map.iter().position(|(k, _)| k == key)
}

fn get(map: &[(Basic, i64)], key: &Basic) -> Option<i64> {
    position(map, key).map(|i| map[i].1)
}

fn set(map: &mut Vec<(Basic, i64)>, key: Basic, value: i64) {
    match position(map, &key) {
        Some(i) => map[i].1 = value,
        None => map.push((key, value)),
    }
}

fn remove(map: &mut Vec<(Basic, i64)>, key: &Basic) {
    if let Some(i) = position(map, key) {
        map.remove(i);
    }
}

impl Factors {
    pub fn new(factors: Vec<(Basic, i64)>) -> Factors {
        let mut result: Vec<(Basic, i64)> = Vec::new();
        for (base, n) in factors {
            if n != 0 {
                set(&mut result, base, n);
            }
        }
        let i: Basic = Basic::imaginary_unit();
        let unit: i64 = get(&result, &i).unwrap_or(0).rem_euclid(4);
        remove(&mut result, &i);
        let minus: Basic = Basic::negative_one();
        if unit == 2 || unit == 3 {
            let count: i64 = get(&result, &minus).unwrap_or(0) + 1;
            set(&mut result, minus.clone(), count);
        }
        if unit == 1 || unit == 3 {
            set(&mut result, i, 1);
        }
        match get(&result, &minus) {
            Some(count) if count % 2 == 0 => remove(&mut result, &minus),
            Some(_) => set(&mut result, minus, 1),
            None => {}
        }
        Factors { factors: result }
    }

    pub fn is_one(&self) -> bool {
        self.factors.is_empty()
    }

    pub fn as_expr(&self) -> PolyResult<Basic> {
        let mut args: Vec<Basic> = Vec::new();
        for (factor, n) in &self.factors {
            if *n != 1 {
                let (b, e) = as_base_exp(factor)?;
                args.push(pow(&b, &keep_coeff(&Basic::int(*n), &e)?)?);
            } else {
                args.push(factor.clone());
            }
        }
        mul(&args)
    }

    pub fn mul(&self, other: &Factors) -> Factors {
        let mut factors: Vec<(Basic, i64)> = self.factors.clone();
        for (factor, n) in &other.factors {
            let total: i64 = get(&factors, factor).unwrap_or(0) + n;
            set(&mut factors, factor.clone(), total);
        }
        Factors::new(factors)
    }

    /// `self` and `other` with their common factors cancelled.
    pub fn normal(&self, other: &Factors) -> (Factors, Factors) {
        let (mut mine, mut theirs) = (self.factors.clone(), other.factors.clone());
        for (factor, n) in &self.factors {
            let Some(m) = get(&other.factors, factor) else {
                continue;
            };
            let d: i64 = n - m;
            if d > 0 {
                set(&mut mine, factor.clone(), d);
                remove(&mut theirs, factor);
            } else {
                remove(&mut mine, factor);
                if d != 0 {
                    set(&mut theirs, factor.clone(), -d);
                } else {
                    remove(&mut theirs, factor);
                }
            }
        }
        (Factors::new(mine), Factors::new(theirs))
    }

    pub fn quo(&self, other: &Factors) -> Factors {
        let mut quotient: Vec<(Basic, i64)> = self.factors.clone();
        for (factor, n) in &other.factors {
            if let Some(q) = get(&quotient, factor) {
                let d: i64 = q - n;
                if d <= 0 {
                    remove(&mut quotient, factor);
                } else {
                    set(&mut quotient, factor.clone(), d);
                }
            }
        }
        Factors::new(quotient)
    }

    pub fn gcd(&self, other: &Factors) -> Factors {
        Factors::new(
            self.factors
                .iter()
                .filter_map(|(f, n)| get(&other.factors, f).map(|m| (f.clone(), (*n).min(m))))
                .collect(),
        )
    }

    pub fn lcm(&self, other: &Factors) -> Factors {
        let mut factors: Vec<(Basic, i64)> = self.factors.clone();
        for (factor, n) in &other.factors {
            let value: i64 = match get(&factors, factor) {
                Some(m) => (*n).max(m),
                None => *n,
            };
            set(&mut factors, factor.clone(), value);
        }
        Factors::new(factors)
    }
}

/// A commutative term as `coeff * numer / denom` (`Term`).
#[derive(Debug, Clone)]
pub struct Term {
    pub coeff: Basic,
    pub numer: Factors,
    pub denom: Factors,
}

impl Term {
    pub fn of(term: &Basic) -> PolyResult<Term> {
        if is_false(term, Predicate::commutative) {
            return Err(PolyError::Value("commutative expression expected".to_string()));
        }
        let (mut coeff, factors) = as_coeff_mul_tuple(term);
        let mut numer: Vec<(Basic, i64)> = Vec::new();
        let mut denominator: Vec<(Basic, i64)> = Vec::new();
        for factor in factors {
            let (mut base, n) = decompose_power(&factor)?;
            if base.is_add() {
                let (content, rest) = primitive(&base)?;
                base = rest;
                let scaled: Basic = mul(&[coeff.clone(), pow(&content, &Basic::int(n))?])?;
                if scaled.is_number_class() {
                    coeff = scaled;
                }
            }
            if n > 0 {
                let total: i64 = get(&numer, &base).unwrap_or(0) + n;
                set(&mut numer, base, total);
            } else {
                let total: i64 = get(&denominator, &base).unwrap_or(0) - n;
                set(&mut denominator, base, total);
            }
        }
        Ok(Term {
            coeff,
            numer: Factors::new(numer),
            denom: Factors::new(denominator),
        })
    }

    pub fn as_expr(&self) -> PolyResult<Basic> {
        mul(&[
            self.coeff.clone(),
            self.numer.as_expr()?,
            pow(&self.denom.as_expr()?, &Basic::negative_one())?,
        ])
    }

    pub fn mul(&self, other: &Term) -> PolyResult<Term> {
        let coeff: Basic = mul(&[self.coeff.clone(), other.coeff.clone()])?;
        let (numer, denominator) = self.numer.mul(&other.numer).normal(&self.denom.mul(&other.denom));
        Ok(Term {
            coeff: as_number(coeff),
            numer,
            denom: denominator,
        })
    }

    pub fn inv(&self) -> PolyResult<Term> {
        let inverse: Basic = pow(&self.coeff, &Basic::negative_one())?;
        Ok(Term {
            coeff: as_number(inverse),
            numer: self.denom.clone(),
            denom: self.numer.clone(),
        })
    }

    pub fn quo(&self, other: &Term) -> PolyResult<Term> {
        self.mul(&other.inv()?)
    }

    pub fn gcd(&self, other: &Term) -> Term {
        Term {
            coeff: number_gcd(&self.coeff, &other.coeff),
            numer: self.numer.gcd(&other.numer),
            denom: self.denom.gcd(&other.denom),
        }
    }

    pub fn lcm(&self, other: &Term) -> PolyResult<Term> {
        Ok(Term {
            coeff: number_lcm(&self.coeff, &other.coeff)?,
            numer: self.numer.lcm(&other.numer),
            denom: self.denom.lcm(&other.denom),
        })
    }
}

// ---------------------------------------------------------------------
// gcd_terms

/// `(content, numerator, denominator)` of a sum of terms (`_gcd_terms`).
fn gcd_terms_parts(terms: &[Basic], isprimitive: bool, fraction: bool) -> PolyResult<(Basic, Basic, Basic)> {
    let mut parsed: Vec<Term> = Vec::new();
    for t in terms {
        if !(t.is_number_class() && is_rational_value(t, 0)) {
            parsed.push(Term::of(t)?);
        }
    }
    if parsed.is_empty() {
        return Ok((Basic::zero(), Basic::zero(), Basic::one()));
    }
    let (mut content, mut numer, denominator): (Basic, Basic, Basic);
    if parsed.len() == 1 {
        content = parsed[0].coeff.clone();
        numer = parsed[0].numer.as_expr()?;
        denominator = parsed[0].denom.as_expr()?;
    } else {
        let mut common: Term = parsed[0].clone();
        for term in &parsed[1..] {
            common = common.gcd(term);
        }
        let mut quotients: Vec<Term> = Vec::new();
        for term in &parsed {
            quotients.push(term.quo(&common)?);
        }
        let mut numers: Vec<Basic> = Vec::new();
        if fraction {
            let mut lcm_denom: Factors = quotients[0].denom.clone();
            for term in &quotients[1..] {
                lcm_denom = lcm_denom.lcm(&term.denom);
            }
            for term in &quotients {
                numers.push(mul(&[
                    term.coeff.clone(),
                    term.numer.mul(&lcm_denom.quo(&term.denom)).as_expr()?,
                ])?);
            }
            denominator = lcm_denom.as_expr()?;
        } else {
            for term in &quotients {
                numers.push(term.as_expr()?);
            }
            denominator = Basic::one();
        }
        content = common.as_expr()?;
        numer = add(&numers)?;
    }
    if !isprimitive && numer.is_add() {
        let (c, rest) = primitive(&numer)?;
        numer = rest;
        content = mul(&[content, c])?;
    }
    Ok((content, numer, denominator))
}

/// The noncommutative parts of the terms replaced by dummies (`_mask`).
fn mask(terms: &[Basic]) -> PolyResult<(Vec<Basic>, HashMap<Basic, Basic>)> {
    let mut masked: Vec<Basic> = Vec::new();
    let mut replacements: HashMap<Basic, Basic> = HashMap::new();
    for t in terms {
        if !is_false(t, Predicate::commutative) {
            masked.push(t.clone());
            continue;
        }
        let parts: Vec<Basic> = factors_of(t);
        let mut c: Vec<Basic> = parts
            .iter()
            .filter(|f| !is_false(f, Predicate::commutative))
            .cloned()
            .collect();
        let nc: Vec<Basic> = parts
            .iter()
            .filter(|f| is_false(f, Predicate::commutative))
            .cloned()
            .collect();
        let d: Basic = Dummy::new("Dummy", &[]).0;
        replacements.insert(d.clone(), mul(&nc)?);
        c.push(d);
        masked.push(mul(&c)?);
    }
    Ok((masked, replacements))
}

/// `gcd_terms` of a sum (`_gcd_terms_sum`).
fn gcd_terms_sum(terms: &[Basic], isprimitive: bool, clear: bool, fraction: bool) -> PolyResult<Basic> {
    let (masked, replacements) = mask(terms)?;
    let (content, mut numer, denominator) = gcd_terms_parts(&masked, isprimitive, fraction)?;
    if !replacements.is_empty() {
        numer = numer.xreplace(&replacements);
    }
    let (mut coeff, factors) = as_coeff_mul(&content);
    if !clear {
        let (c, rest) = as_coeff_mul(&coeff);
        if !is_integer_class(&c) && numer.is_add() {
            let (n, d) = as_numer_denom(&c)?;
            let divided: Basic = mul(&[numer.clone(), pow(&d, &Basic::negative_one())?])?;
            if divided.is_add() && divided.args().iter().any(|a| is_integer_class(&as_coeff_mul(a).0)) {
                numer = divided;
                coeff = as_number(mul(&[n, rest])?);
            }
        }
    }
    let rest: Basic = mul(&[factors, numer, pow(&denominator, &Basic::negative_one())?])?;
    keep_coeff_with(&coeff, &rest, clear, false)
}

/// The terms put over a common denominator with their content and their
/// common factors taken out (`gcd_terms`).
pub fn gcd_terms(e: &Basic, isprimitive: bool, clear: bool, fraction: bool) -> PolyResult<Basic> {
    if e.is_add() {
        return gcd_terms_sum(e.args(), isprimitive, clear, fraction);
    }
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    if e.is_mul() {
        let (c, factors) = as_coeff_mul_tuple(e);
        let mut parts: Vec<Basic> = Vec::new();
        for f in &factors {
            parts.push(gcd_terms(f, isprimitive, clear, fraction)?);
        }
        return keep_coeff_with(&c, &mul(&parts)?, clear, false);
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(gcd_terms(a, isprimitive, clear, fraction)?);
    }
    rebuild_evaluated(e, args)
}

// ---------------------------------------------------------------------
// as_content_primitive

/// SymPy's rule for `clear=False`: a fractional content is taken out of a
/// sum only when every term has a fractional coefficient (`_clearable`).
fn clearable(e: &Basic) -> bool {
    e.args()
        .iter()
        .all(|a| as_coeff_mul(a).0.as_rational().is_some_and(|v| !v.is_int()))
}

/// A positive rational content and the rest (`as_content_primitive`).
pub fn as_content_primitive(e: &Basic, radical: bool, clear: bool) -> PolyResult<(Basic, Basic)> {
    match e.head() {
        Head::Rational { value, .. } => {
            if value.is_zero() {
                return Ok((Basic::one(), e.clone()));
            }
            let magnitude: Rat = if rat_is_negative(value) {
                -value.clone()
            } else {
                value.clone()
            };
            // Python's Rational(|p|, q): the class Rational even for q = 1
            let content: Basic = Basic::atom(Head::Rational {
                value: magnitude,
                integer_class: false,
            });
            Ok((content, Basic::int(if rat_is_negative(value) { -1 } else { 1 })))
        }
        Head::Add => add_content_primitive(e, radical, clear),
        Head::Mul => {
            let mut content: Rat = Rat::ONE;
            let mut primitives: Vec<Basic> = Vec::new();
            for a in e.args() {
                let (c, p) = as_content_primitive(a, radical, clear)?;
                if let Some(v) = c.as_rational() {
                    content *= v.clone();
                }
                if !(is_integer_class(&p) && is_rational_value(&p, 1)) {
                    primitives.push(p);
                }
            }
            Ok((Basic::from_rational(content), mul(&primitives)?))
        }
        Head::Pow => pow_content_primitive(e, radical, clear),
        _ => Ok((Basic::one(), e.clone())),
    }
}

fn add_content_primitive(e: &Basic, radical: bool, clear: bool) -> PolyResult<(Basic, Basic)> {
    let mut rebuilt: Vec<Basic> = Vec::new();
    for t in e.args() {
        let (c, p) = as_content_primitive(t, radical, clear)?;
        rebuilt.push(mul(&[c, p])?);
    }
    if rebuilt.as_slice() != e.args() {
        let total: Basic = add(&rebuilt)?;
        if total.is_add() && total != *e {
            return as_content_primitive(&total, radical, clear);
        }
    }
    if radical {
        return Err(not_ported("_common_radicals"));
    }
    let (c, p) = primitive(e)?;
    let cv: Rat = c.as_rational().expect("a rational content").clone();
    if clear || cv.is_int() || clearable(e) {
        return Ok((c, p));
    }
    if numer(&cv) == Int::ONE {
        return Ok((Basic::one(), e.clone()));
    }
    let inverse: Basic = Basic::from_rational(rat_from_parts(Int::ONE, numer(&cv)));
    let mut terms: Vec<Basic> = Vec::new();
    for t in e.args() {
        terms.push(mul(&[inverse.clone(), t.clone()])?);
    }
    Ok((Basic::integer(numer(&cv)), add(&terms)?))
}

fn pow_content_primitive(e: &Basic, radical: bool, clear: bool) -> PolyResult<(Basic, Basic)> {
    let (b0, e0) = as_base_exp(e)?;
    let (bc, bp) = as_content_primitive(&b0, radical, clear)?;
    let b: Basic = keep_coeff(&bc, &bp)?;
    let (ce, pe) = as_content_primitive(&e0, radical, clear)?;
    if let Some(bv) = b.as_rational() {
        let (h, t) = as_coeff_add(&pe);
        if let (Some(hv), Some(cev)) = (h.as_rational(), ce.as_rational()) {
            if !bv.is_zero() {
                let ceh: Rat = cev * hv;
                let mut c: Basic = pow(&b, &Basic::from_rational(ceh.clone()))?;
                let mut r: Basic = Basic::zero();
                if c.as_rational().is_none() {
                    let (p, q) = (numer(&ceh), denom(&ceh));
                    c = pow(&b, &Basic::integer(div_floor(&p, &q)))?;
                    r = Basic::integer(mod_floor(&p, &q));
                }
                if c.as_rational().is_some() {
                    let shift: Basic = mul(&[
                        r,
                        pow(&ce, &Basic::negative_one())?,
                        Basic::from_rational(rat_from_parts(Int::ONE, denom(&ceh))),
                    ])?;
                    return Ok((c, pow(&b, &keep_coeff(&ce, &add(&[t, shift])?)?)?));
                }
            }
        }
    }
    let exponent: Basic = keep_coeff(&ce, &pe)?;
    if exponent.as_rational().is_some() && b.is_mul() {
        let (h2, t2) = as_content_primitive(&b, radical, clear)?;
        let (c2, m2) = as_coeff_mul(&pow(&h2, &exponent)?);
        let (m3, me) = as_base_exp(&m2)?;
        if c2.is_number_class() && ((is_integer_class(&m3) && is_rational_value(&m3, 1)) || me == exponent) {
            return Ok((c2, pow(&keep_coeff(&m3, &t2)?, &exponent)?));
        }
    }
    Ok((Basic::one(), pow(&b, &exponent)?))
}

// ---------------------------------------------------------------------
// factor_terms

/// The common factors of the terms taken out, in every node
/// (`factor_terms`).
pub fn factor_terms(e: &Basic, radical: bool, clear: bool, fraction: bool, sign: bool) -> PolyResult<Basic> {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    if e.is_pow() || matches!(e.head(), Head::Function(_)) {
        let mut new: Vec<Basic> = Vec::new();
        for a in e.args() {
            new.push(factor_terms(a, radical, clear, fraction, sign)?);
        }
        return if new.as_slice() == e.args() {
            Ok(e.clone())
        } else {
            rebuild_evaluated(e, new)
        };
    }
    let (mut content, mut p) = as_content_primitive(e, radical, clear)?;
    if p.is_add() {
        let mut terms: Vec<Basic> = Vec::new();
        for a in p.args() {
            terms.push(factor_terms(a, radical, clear, fraction, sign)?);
        }
        let mut all_negative: bool = true;
        for t in &terms {
            if extract_multiplicatively(&as_coeff_mul(t).0, &Basic::negative_one())?.is_none() {
                all_negative = false;
                break;
            }
        }
        if all_negative {
            content = as_number(mul(&[content, Basic::negative_one()])?);
            let mut negated: Vec<Basic> = Vec::new();
            for t in terms {
                negated.push(mul(&[Basic::negative_one(), t])?);
            }
            terms = negated;
        }
        let mut special: HashMap<Basic, Basic> = HashMap::new();
        for t in terms.iter_mut() {
            let (_, exponent) = as_base_exp(t)?;
            if exponent.is_mul() && exponent != mul(exponent.args())? {
                let d: Basic = Dummy::new("Dummy", &[]).0;
                special.insert(d.clone(), t.clone());
                *t = d;
            }
        }
        let sum: Basic = if terms.len() > 1 {
            Basic::new(Head::Add, terms)
        } else {
            terms.remove(0)
        };
        let factored: Basic = gcd_terms(&sum, true, clear, fraction)?;
        p = if special.is_empty() {
            factored
        } else {
            factored.xreplace(&special)
        };
    } else if !p.args().is_empty() {
        let mut new: Vec<Basic> = Vec::new();
        for a in p.args() {
            new.push(factor_terms(a, radical, clear, fraction, sign)?);
        }
        p = rebuild_evaluated(&p, new)?;
    }
    keep_coeff_with(&content, &p, clear, sign)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    #[test]
    fn docstring_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let y: Basic = Symbol::new("y", &[]).0;
        let show = |e: PolyResult<Basic>| sstr(&e.unwrap());
        let two_x_four_y: Basic = add(&[
            mul(&[Basic::int(2), x.clone()]).unwrap(),
            mul(&[Basic::int(4), y.clone()]).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            show(factor_terms(&two_x_four_y, false, false, false, true)),
            "2*(x + 2*y)"
        );
        let xy_x: Basic = add(&[mul(&[x.clone(), y.clone()]).unwrap(), x.clone()]).unwrap();
        assert_eq!(show(factor_terms(&xy_x, false, false, false, true)), "x*(y + 1)");
        let minus: Basic = add(&[
            mul(&[Basic::negative_one(), x.clone()]).unwrap(),
            mul(&[Basic::negative_one(), y.clone()]).unwrap(),
        ])
        .unwrap();
        assert_eq!(show(factor_terms(&minus, false, false, false, true)), "-(x + y)");
        let quarter: Basic = add(&[
            mul(&[Basic::half(), x.clone()]).unwrap(),
            mul(&[Basic::from_rational(rat(1, 4)), y.clone()]).unwrap(),
        ])
        .unwrap();
        assert_eq!(show(gcd_terms(&quarter, false, true, true)), "(2*x + y)/4");
        assert_eq!(
            decompose_power(&pow(&x, &Basic::from_rational(rat(3, 2))).unwrap())
                .unwrap()
                .1,
            3
        );
    }
}
