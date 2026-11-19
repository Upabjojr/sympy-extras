//! Port of `sympy_extras/simplify/normalform.py`: a normal form of
//! expressions in radicals, `I`, trigonometric and hyperbolic functions of
//! the same arguments and exponentials of rational multiples: every
//! algebraic dependency becomes a relation (`rho**q = n d**(q - 1)`,
//! `c**2 + s**2 = 1`, `c**2 - s**2 = 1`, `I**2 = -1`, the exponentials of
//! one argument powers of one placeholder), the expression a quotient of
//! polynomials reduced modulo the relations; the smaller of the forms with
//! the cosines or the sines eliminated first.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{zero_division, PolyResult};
use crate::matrices::linalg::{collect_kernels, fraction_default, IntegerLift, Lift, PolynomialLift, ZPoly};
use crate::numbers::{denom, numer, Int, Rat};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::integerring::IntegerRing;
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn integer(n: &Int) -> Basic {
    Basic::from_rational(Rat::from(n.clone()))
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn apply(f: Function, a: &Basic) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), vec![a.clone()]), vec![a.clone()])
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn xreplace(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn insert(rule: &mut Vec<(Basic, Basic)>, k: Basic, v: Basic) {
    match rule.iter_mut().find(|(a, _)| *a == k) {
        Some(entry) => entry.1 = v,
        None => rule.push((k, v)),
    }
}

fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut found: Vec<Basic> = Vec::new();
    collect_kernels(e, &mut found);
    if found.is_empty() {
        let (n, d) = IntegerLift.fraction(e)?;
        let (p, q) = IntegerLift.cancel(&n, &d)?;
        return Ok((integer(&p), integer(&q)));
    }
    let lift: PolynomialLift = PolynomialLift::new(&found);
    let (n, d) = lift.fraction(e)?;
    let (n, d) = lift.cancel(&n, &d)?;
    Ok((lift.expr(&n)?, lift.expr(&d)?))
}

/// The placeholders of the algebraic quantities (`_Abstraction`).
struct Abstraction {
    roots: Vec<((Basic, i64), (Basic, Basic))>,
    root_order: Vec<Basic>,
    cosines: Vec<((bool, Basic), Basic)>,
    sines: Vec<((bool, Basic), Basic)>,
    relations: Vec<Basic>,
    back: Vec<(Basic, Basic)>,
    imaginary: bool,
}

impl Abstraction {
    fn new() -> Abstraction {
        Abstraction {
            roots: Vec::new(),
            root_order: Vec::new(),
            cosines: Vec::new(),
            sines: Vec::new(),
            relations: Vec::new(),
            back: Vec::new(),
            imaginary: false,
        }
    }

    fn pair(&mut self, hyperbolic: bool, argument: &Basic) -> PolyResult<(Basic, Basic)> {
        let key: (bool, Basic) = (hyperbolic, argument.clone());
        if !self.sines.iter().any(|(k, _)| *k == key) {
            let (s, c) = (dummy("s"), dummy("c"));
            self.sines.push((key.clone(), s.clone()));
            self.cosines.push((key.clone(), c.clone()));
            let (c2, s2_) = (pw(&c, &int(2))?, pw(&s, &int(2))?);
            if hyperbolic {
                self.back.push((s.clone(), apply(Function::Sinh, argument)?));
                self.back.push((c.clone(), apply(Function::Cosh, argument)?));
                self.relations.push(s2(&s2(&c2, &s2_)?, &int(1))?);
            } else {
                self.back.push((s.clone(), apply(Function::Sin, argument)?));
                self.back.push((c.clone(), apply(Function::Cos, argument)?));
                self.relations.push(s2(&a2(&c2, &s2_)?, &int(1))?);
            }
        }
        let s: Basic = self.sines.iter().find(|(k, _)| *k == key).expect("a sine").1.clone();
        let c: Basic = self
            .cosines
            .iter()
            .find(|(k, _)| *k == key)
            .expect("a cosine")
            .1
            .clone();
        Ok((s, c))
    }

    fn restore(&self, e: &Basic) -> R {
        let mut e: Basic = e.clone();
        for (placeholder, original) in self.back.iter().rev() {
            if e.has(placeholder) {
                e = crate::core::subs::subs(&e, placeholder, original)?;
            }
        }
        Ok(e)
    }

    fn abstract_(&mut self, e: &Basic) -> R {
        if e.is_rational_class() || e.is_symbol() {
            return Ok(e.clone());
        }
        if matches!(e.head(), Head::ImaginaryUnit) {
            self.imaginary = true;
            return Ok(e.clone());
        }
        if e.is_add() {
            let parts: Vec<Basic> = e.args().iter().map(|a| self.abstract_(a)).collect::<PolyResult<_>>()?;
            return crate::core::add::add(&parts);
        }
        if e.is_mul() {
            let parts: Vec<Basic> = e.args().iter().map(|a| self.abstract_(a)).collect::<PolyResult<_>>()?;
            return crate::core::mul::mul(&parts);
        }
        if e.is_pow() {
            let (base, exponent) = (&e.args()[0], &e.args()[1]);
            if exponent.is_integer_value() {
                return pw(&self.abstract_(base)?, exponent);
            }
            if let Some(r) = exponent.as_rational() {
                let p: i64 = i64::try_from(&numer(r)).expect("a small exponent");
                let q: i64 = i64::try_from(&denom(r)).expect("a small exponent");
                let key: (Basic, i64) = (base.clone(), q);
                if !self.roots.iter().any(|(k, _)| *k == key) {
                    let abstracted: Basic = self.abstract_(base)?;
                    let (numerator, denominator) = numer_denom(&abstracted)?;
                    let rho: Basic = dummy("rho");
                    self.roots.push((key.clone(), (rho.clone(), denominator.clone())));
                    self.root_order.push(rho.clone());
                    self.relations.push(s2(
                        &pw(&rho, &int(q))?,
                        &m2(&numerator, &pw(&denominator, &int(q - 1))?)?,
                    )?);
                    let restored: Basic = self.restore(&denominator)?;
                    let root: Basic = pw(base, &d2(&int(1), &int(q))?)?;
                    self.back.push((rho.clone(), m2(&restored, &root)?));
                }
                let (rho, denominator) = self.roots.iter().find(|(k, _)| *k == key).expect("a root").1.clone();
                return d2(&pw(&rho, &int(p))?, &pw(&denominator, &int(p))?);
            }
            return Ok(e.clone());
        }
        let trig = [
            Function::Sin,
            Function::Cos,
            Function::Tan,
            Function::Cot,
            Function::Sec,
            Function::Csc,
        ];
        let hyper = [
            Function::Sinh,
            Function::Cosh,
            Function::Tanh,
            Function::Coth,
            Function::Sech,
            Function::Csch,
        ];
        if let Some(f) = func(e) {
            let position = trig
                .iter()
                .position(|g| *g == f)
                .map(|i| (false, i))
                .or_else(|| hyper.iter().position(|g| *g == f).map(|i| (true, i)));
            if let Some((hyperbolic, i)) = position {
                let argument: Basic = e.args()[0].clone();
                let (s, c) = self.pair(hyperbolic, &argument)?;
                return match i {
                    0 => Ok(s),
                    1 => Ok(c),
                    2 => d2(&s, &c),
                    3 => d2(&c, &s),
                    4 => d2(&int(1), &c),
                    _ => d2(&int(1), &s),
                };
            }
        }
        Ok(e.clone())
    }
}

fn of_function(e: &Basic, fs: &[Function]) -> Vec<Basic> {
    e.atoms(Some(
        &|n: &Basic| matches!(n.head(), Head::Function(f) if fs.contains(f)),
    ))
}

fn logarithms_split(expression: &Basic) -> R {
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for node in of_function(expression, &[Function::Log]) {
        let argument: &Basic = &node.args()[0];
        let (coefficient, rest) = crate::core::expr::as_coeff_mul(argument);
        let Some(c) = coefficient.as_rational() else { continue };
        if numer(c) <= Int::from(0) {
            continue;
        }
        let mut terms: Vec<Basic> = Vec::new();
        for (prime, exponent) in crate::ntheory::factor_::factorint(&numer(c), None) {
            terms.push(m2(
                &int(exponent as i64),
                &crate::functions::elementary::exponential::log(&integer(&prime))?,
            )?);
        }
        for (prime, exponent) in crate::ntheory::factor_::factorint(&denom(c), None) {
            terms.push(m2(
                &int(-(exponent as i64)),
                &crate::functions::elementary::exponential::log(&integer(&prime))?,
            )?);
        }
        let factors: Vec<Basic> = if rest.is_mul() {
            rest.args().to_vec()
        } else if rest == int(1) {
            Vec::new()
        } else {
            vec![rest.clone()]
        };
        let mut valid: bool = true;
        for factor in &factors {
            let (base, power) = if factor.is_pow() {
                (factor.args()[0].clone(), factor.args()[1].clone())
            } else {
                (factor.clone(), int(1))
            };
            if !(base.is_symbol() && is(&base, Predicate::positive) == Some(true) && power.is_rational_class()) {
                valid = false;
                break;
            }
            terms.push(m2(&power, &crate::functions::elementary::exponential::log(&base)?)?);
        }
        if !valid {
            continue;
        }
        let replacement: Basic = if terms.is_empty() {
            int(0)
        } else {
            crate::core::add::add(&terms)?
        };
        if replacement != node {
            insert(&mut rule, node.clone(), replacement);
        }
    }
    xreplace(expression, &rule)
}

fn hyperbolic_as_exponentials(expression: &Basic) -> R {
    let mut bases: Vec<Basic> = Vec::new();
    for node in of_function(expression, &[Function::Exp]) {
        let (coefficient, rest) = crate::core::expr::as_coeff_mul(&node.args()[0]);
        if coefficient.is_rational_class() && !bases.contains(&rest) {
            bases.push(rest);
        }
    }
    if bases.is_empty() {
        return Ok(expression.clone());
    }
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    let hyper = [
        Function::Sinh,
        Function::Cosh,
        Function::Tanh,
        Function::Coth,
        Function::Sech,
        Function::Csch,
    ];
    for node in of_function(expression, &hyper) {
        let argument: &Basic = &node.args()[0];
        let (coefficient, rest) = crate::core::expr::as_coeff_mul(argument);
        if !coefficient.is_rational_class() || !bases.contains(&rest) {
            continue;
        }
        let up: Basic = crate::functions::elementary::exponential::exp(argument)?;
        let down: Basic = crate::functions::elementary::exponential::exp(&crate::core::mul::neg(argument)?)?;
        let value: Basic = match func(&node).expect("a function") {
            Function::Sinh => d2(&s2(&up, &down)?, &int(2))?,
            Function::Cosh => d2(&a2(&up, &down)?, &int(2))?,
            Function::Tanh => d2(&s2(&up, &down)?, &a2(&up, &down)?)?,
            Function::Coth => d2(&a2(&up, &down)?, &s2(&up, &down)?)?,
            Function::Sech => d2(&int(2), &a2(&up, &down)?)?,
            _ => d2(&int(2), &s2(&up, &down)?)?,
        };
        insert(&mut rule, node.clone(), value);
    }
    xreplace(expression, &rule)
}

fn gcd(a: &Int, b: &Int) -> Int {
    ZZ_GCD.gcd(a, b)
}
const ZZ_GCD: IntegerRing = crate::polys::domains::integerring::ZZ;

fn exponentials(expression: &Basic, back: &mut Vec<(Basic, Basic)>) -> R {
    let mut groups: Vec<(Basic, Vec<(Basic, Rat)>)> = Vec::new();
    for node in of_function(expression, &[Function::Exp]) {
        let (coefficient, rest) = crate::core::expr::as_coeff_mul(&node.args()[0]);
        if let Some(c) = coefficient.as_rational() {
            match groups.iter_mut().find(|(r, _)| *r == rest) {
                Some(entry) => entry.1.push((node.clone(), c.clone())),
                None => groups.push((rest, vec![(node.clone(), c.clone())])),
            }
        }
    }
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for (rest, nodes) in &groups {
        let (mut numerator, mut denominator) = (Int::from(0), Int::from(1));
        for (_, r) in nodes {
            numerator = gcd(&(numerator.clone() * denom(r)), &(numer(r) * denominator.clone()));
            denominator *= denom(r);
        }
        let step: Rat = Rat::from(numerator) / Rat::from(denominator);
        let placeholder: Basic = dummy("E");
        back.push((
            placeholder.clone(),
            crate::functions::elementary::exponential::exp(&m2(&Basic::from_rational(step.clone()), rest)?)?,
        ));
        for (node, r) in nodes {
            let power: Rat = r.clone() / step.clone();
            insert(&mut rule, node.clone(), pw(&placeholder, &integer(&numer(&power)))?);
        }
    }
    xreplace(expression, &rule)
}

/// A polynomial lift reducing modulo the relations (`_QuotientLift`).
struct QuotientLift {
    inner: PolynomialLift,
    divisors: Vec<ZPoly>,
}

impl Lift for QuotientLift {
    type K = SparsePolynomialRing<IntegerRing>;
    fn domain(&self) -> &SparsePolynomialRing<IntegerRing> {
        &self.inner.ring
    }
    fn exact(&self) -> bool {
        self.inner.exact
    }
    fn atom(&self, e: &Basic) -> PolyResult<ZPoly> {
        self.inner.atom(e)
    }
    fn expr(&self, a: &ZPoly) -> R {
        self.inner.expr(a)
    }
    fn cancel(&self, n: &ZPoly, d: &ZPoly) -> PolyResult<(ZPoly, ZPoly)> {
        self.inner.cancel(n, d)
    }
    fn fraction(&self, e: &Basic) -> PolyResult<(ZPoly, ZPoly)> {
        let (mut n, mut d) = fraction_default(self, e)?;
        if !self.divisors.is_empty() && (e.is_add() || e.is_mul() || e.is_pow()) {
            n = n.rem(&self.divisors)?;
            d = d.rem(&self.divisors)?;
            if d.is_zero() {
                return Err(zero_division(&format!("division by zero in {}", e)));
            }
        }
        Ok((n, d))
    }
}

fn lift_of(abstraction: &Abstraction, found: &[Basic], sines_first: bool) -> PolyResult<QuotientLift> {
    let mut special: Vec<Basic> = abstraction.root_order.iter().rev().cloned().collect();
    if abstraction.imaginary {
        special.push(Basic::atom(Head::ImaginaryUnit));
    }
    let (first, second) = if sines_first {
        (&abstraction.sines, &abstraction.cosines)
    } else {
        (&abstraction.cosines, &abstraction.sines)
    };
    special.extend(first.iter().map(|(_, v)| v.clone()));
    special.extend(second.iter().map(|(_, v)| v.clone()));
    let mut kernels: Vec<Basic> = special.clone();
    kernels.extend(found.iter().filter(|k| !special.contains(k)).cloned());
    let mut lift = QuotientLift {
        inner: PolynomialLift::new(&kernels),
        divisors: Vec::new(),
    };
    let mut divisors: Vec<ZPoly> = Vec::new();
    for relation in &abstraction.relations {
        divisors.push(lift.fraction(relation)?.0);
    }
    if abstraction.imaginary {
        let unit: ZPoly = lift.inner.index[&Basic::atom(Head::ImaginaryUnit)].clone();
        divisors.push(unit.mul(&unit).add(&lift.inner.ring.ring.one()));
    }
    lift.divisors = divisors;
    Ok(lift)
}

fn measure(numerator: &ZPoly, denominator: &ZPoly) -> (usize, usize) {
    let mut used: Vec<usize> = Vec::new();
    for f in [numerator, denominator] {
        for monomial in f.terms.keys() {
            for (i, exponent) in monomial.iter().enumerate() {
                if *exponent != 0 && !used.contains(&i) {
                    used.push(i);
                }
            }
        }
    }
    (numerator.terms.len() + denominator.terms.len(), used.len())
}

fn moved(f: &ZPoly, source: &QuotientLift, target: &QuotientLift) -> PolyResult<ZPoly> {
    let positions: Vec<usize> = target
        .inner
        .kernels
        .iter()
        .map(|k| source.inner.kernels.iter().position(|s| s == k).expect("a kernel"))
        .collect();
    let mut terms = crate::polys::rings::Terms::new();
    for (monomial, coefficient) in &f.terms {
        terms.insert(positions.iter().map(|i| monomial[*i]).collect(), coefficient.clone());
    }
    target.inner.ring.ring.from_dict(terms).rem(&target.divisors)
}

/// The normal form of `e` (`normal_form`).
pub fn normal_form(e: &Basic) -> R {
    let mut abstraction: Abstraction = Abstraction::new();
    let prepared: Basic = exponentials(
        &hyperbolic_as_exponentials(&logarithms_split(e)?)?,
        &mut abstraction.back,
    )?;
    let abstracted: Basic = abstraction.abstract_(&prepared)?;
    let mut found: Vec<Basic> = Vec::new();
    collect_kernels(&abstracted, &mut found);
    for relation in &abstraction.relations {
        collect_kernels(relation, &mut found);
    }
    if found.is_empty() {
        let (n, d) = IntegerLift.fraction(&abstracted)?;
        return IntegerLift.ratio(&n, &d);
    }
    let lift: QuotientLift = lift_of(&abstraction, &found, false)?;
    let (n0, d0) = lift.fraction(&abstracted)?;
    let (numerator, denominator) = lift.cancel(&n0, &d0)?;
    let mut best: Basic = lift.ratio(&numerator, &denominator)?;
    if !abstraction.sines.is_empty() {
        let other: QuotientLift = lift_of(&abstraction, &found, true)?;
        let (p, q) = (moved(&numerator, &lift, &other)?, moved(&denominator, &lift, &other)?);
        if q.is_zero() {
            return Err(zero_division(&format!("division by zero in {}", e)));
        }
        let (p, q) = other.cancel(&p, &q)?;
        if measure(&p, &q) < measure(&numerator, &denominator) {
            best = other.ratio(&p, &q)?;
        }
    }
    abstraction.restore(&best)
}
