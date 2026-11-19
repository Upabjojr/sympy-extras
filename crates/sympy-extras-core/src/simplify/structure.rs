//! Port of `sympy_extras/simplify/structure.py`: a canonical form and a zero
//! test for elementary expressions by the Risch–Rosenlicht structure
//! theorem. An expression is written as a rational function of generators
//! proven algebraically independent (variables, exponentials, logarithms,
//! and roots with their relations) over a number field which grows as
//! algebraic numbers are met; it vanishes identically exactly when its
//! numerator does.
//!
//! mpmath's `pslq` and `polyroots` are ported in `identification.rs`; the
//! numerical checks of the branches use the evaluation of the core.

#![allow(clippy::mutable_key_type, clippy::needless_range_loop, clippy::unnecessary_unwrap)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::assumptions::ask::ask;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::domains::{AlgebraicField, Domain};
use crate::polys::numberfields::qqbar::Algebraic;
use crate::polys::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

type R = PolyResult<Basic>;
type TP = PolyElement<AlgebraicField>;
type Coef = Vec<Rat>;

const FIELD_DEGREE_LIMIT: usize = 32;
const ASK_SECONDS: f64 = 5.0;
const PLACEHOLDER: &str = "_";

/// The kind of a generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Variable,
    Exponential,
    Logarithm,
    Root,
}

// ---------------------------------------------------------------------
// errors

fn not_elementary(message: String) -> PolyError {
    PolyError::Value(format!("NotElementary: {}", message))
}
fn not_in_field(message: String) -> PolyError {
    PolyError::Value(format!("NotInField: {}", message))
}
/// Whether the error is `NotElementary`.
pub fn is_not_elementary(e: &PolyError) -> bool {
    matches!(e, PolyError::Value(m) if m.starts_with("NotElementary: "))
}
fn is_not_in_field(e: &PolyError) -> bool {
    matches!(e, PolyError::Value(m) if m.starts_with("NotInField: "))
}
fn is_arithmetic(e: &PolyError) -> bool {
    matches!(e, PolyError::Arithmetic(_) | PolyError::ZeroDivision(_))
}
fn zero_division(m: &str) -> PolyError {
    PolyError::ZeroDivision(m.to_string())
}

// ---------------------------------------------------------------------
// numbers

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn number(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}
fn rat(n: i64) -> Rat {
    Rat::from(Int::from(n))
}
fn frac(p: i64, q: i64) -> Rat {
    rat(p) / rat(q)
}
fn numer(r: &Rat) -> Int {
    crate::numbers::numer(r)
}
fn denom(r: &Rat) -> Int {
    Int::from(crate::numbers::denom(r))
}
fn small(n: &Int) -> i64 {
    i64::try_from(n).expect("a small integer")
}
fn is_rational(e: &Basic) -> bool {
    e.is_rational_class()
}
fn rational_of_expr(e: &Basic) -> PolyResult<Rat> {
    e.as_rational()
        .cloned()
        .ok_or_else(|| PolyError::Type(format!("a rational number is expected, got {}", e)))
}
fn gcd_i(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
fn lcm_i(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd_i(a, b) * b
    }
}
fn lcm_all(values: impl IntoIterator<Item = i64>) -> i64 {
    values.into_iter().fold(1, lcm_i)
}

fn is_function(e: &Basic, f: Function) -> bool {
    matches!(e.head(), Head::Function(g) if *g == f)
}

/// The algebraic number of an expression built from rationals, `I`, the
/// golden ratio, sums, products and rational powers (`_algebraic`).
fn algebraic(e: &Basic) -> PolyResult<Algebraic> {
    if let (true, Some(r)) = (is_rational(e), e.as_rational()) {
        return Ok(Algebraic::from_rational(r));
    }
    match e.head() {
        Head::ImaginaryUnit => return Ok(Algebraic::imaginary_unit()),
        Head::GoldenRatio => {
            let five: Algebraic = Algebraic::from_rational(&rat(5)).nth_root(2)?;
            return Algebraic::from_rational(&rat(1))
                .add(&five)?
                .div(&Algebraic::from_rational(&rat(2)));
        }
        Head::Add => {
            let mut total: Algebraic = Algebraic::from_rational(&Rat::ZERO);
            for t in e.args() {
                total = total.add(&algebraic(t)?)?;
            }
            return Ok(total);
        }
        Head::Mul => {
            let mut product: Algebraic = Algebraic::from_rational(&Rat::ONE);
            for f in e.args() {
                product = product.mul(&algebraic(f)?)?;
            }
            return Ok(product);
        }
        Head::Pow if is_rational(&e.args()[1]) => {
            let mut base: Algebraic = algebraic(&e.args()[0])?;
            let r: Rat = rational_of_expr(&e.args()[1])?;
            let (p, q) = (small(&numer(&r)), small(&denom(&r)));
            if q != 1 {
                if base.is_zero() {
                    return Err(not_in_field("a root of zero".to_string()));
                }
                base = base.nth_root(q as usize)?;
            }
            if p < 0 {
                if base.is_zero() {
                    return Err(not_in_field("a negative power of zero".to_string()));
                }
                return base.inverse()?.pow(-p);
            }
            return base.pow(p);
        }
        _ => {}
    }
    Err(not_in_field(format!(
        "{} is not an algebraic number this module can hold",
        e
    )))
}

/// `(b, r)` for `b**r` with rationals (`I` is `(-1, 1/2)`) (`_rational_power`).
fn rational_power(e: &Basic) -> Option<(Rat, Rat)> {
    if matches!(e.head(), Head::ImaginaryUnit) {
        return Some((rat(-1), frac(1, 2)));
    }
    if e.is_pow() && is_rational(&e.args()[0]) && is_rational(&e.args()[1]) && e.args()[0] != int(0) {
        return Some((e.args()[0].as_rational()?.clone(), e.args()[1].as_rational()?.clone()));
    }
    None
}

fn floor_div_i(a: i64, b: i64) -> i64 {
    let t: i64 = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        t - 1
    } else {
        t
    }
}

fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut a, mut b) = (a, b);
    let (mut u0, mut v0, mut u1, mut v1) = (1i64, 0i64, 0i64, 1i64);
    while b != 0 {
        let q: i64 = floor_div_i(a, b);
        (a, b) = (b, a - q * b);
        (u0, u1) = (u1, u0 - q * u1);
        (v0, v1) = (v1, v0 - q * v1);
    }
    if a < 0 {
        (-a, -u0, -v0)
    } else {
        (a, u0, v0)
    }
}

/// The reduced row echelon form over the rationals and its pivots (`_rref`).
fn rref(rows: &[Vec<Rat>], width: usize) -> (Vec<Vec<Rat>>, Vec<usize>) {
    let mut work: Vec<Vec<Rat>> = rows.to_vec();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r: usize = 0;
    for column in 0..width {
        if r == work.len() {
            break;
        }
        let Some(pivot) = (r..work.len()).find(|&i| work[i][column] != Rat::ZERO) else {
            continue;
        };
        work.swap(r, pivot);
        let inverse: Rat = Rat::ONE / &work[r][column];
        work[r] = work[r].iter().map(|v| v * &inverse).collect();
        for i in 0..work.len() {
            if i != r && work[i][column] != Rat::ZERO {
                let factor: Rat = work[i][column].clone();
                let row: Vec<Rat> = work[r].clone();
                work[i] = work[i].iter().zip(row.iter()).map(|(a, b)| a - &factor * b).collect();
            }
        }
        pivots.push(column);
        r += 1;
    }
    (work, pivots)
}

// ---------------------------------------------------------------------
// elements and generators

/// An element of the tower: a fraction of polynomials in the generators
/// (`Element`).
#[derive(Debug, Clone)]
pub struct Element {
    pub num: TP,
    pub den: TP,
}

/// A generator of the tower (`Generator`).
#[derive(Debug, Clone)]
pub struct Generator {
    pub kind: Kind,
    pub name: String,
    pub expression: Basic,
    pub argument: Option<Element>,
    pub certified: bool,
    pub index: i64,
    pub source: i64,
    pub radicand: Option<Element>,
    pub scale: Option<Element>,
    pub eliminated: Option<Element>,
}

impl Generator {
    fn new(kind: Kind, name: String, expression: Basic, argument: Option<Element>, certified: bool) -> Generator {
        Generator {
            kind,
            name,
            expression,
            argument,
            certified,
            index: 1,
            source: -1,
            radicand: None,
            scale: None,
            eliminated: None,
        }
    }
}

/// A pair `(z, e)`, `e = exp(z)`, of a generator (`_Pair`).
#[derive(Debug, Clone)]
struct Pair {
    z: Element,
    e: Element,
    generator: usize,
    logarithmic: bool,
    master: i64,
}

/// The elementary tower (`ElementaryTower`).
pub struct Tower {
    pub assumptions: Vec<Basic>,
    pub extensions: Vec<Basic>,
    numbers: Vec<Algebraic>,
    coordinates: Vec<Coef>,
    theta: Option<Algebraic>,
    theta_expression: Basic,
    degrees: Vec<usize>,
    to_basis: Vec<Vec<Rat>>,
    basis_expressions: Vec<Basic>,
    pub domain: AlgebraicField,
    embeddings: Vec<(AlgebraicField, Coef)>,
    pub generators: Vec<Generator>,
    ring: Arc<PolyRing<AlgebraicField>>,
    pairs: Vec<Pair>,
    variables: Vec<(Basic, usize)>,
    cache: HashMap<Basic, Element>,
    derivatives: HashMap<(usize, usize), Element>,
    prime_logs: BTreeMap<Int, usize>,
    minus_one: i64,
    kummer: Option<bool>,
    signs: HashMap<Basic, i32>,
}

fn rationals() -> AlgebraicField {
    AlgebraicField::new(vec![Rat::ONE, Rat::ZERO], int(0)).expect("x is monic")
}

fn ring_of(domain: &AlgebraicField, names: Vec<String>) -> Arc<PolyRing<AlgebraicField>> {
    PolyRing::new(domain.clone(), names, MonomialOrder::Lex).expect("a ring")
}

fn same_ring(a: &Arc<PolyRing<AlgebraicField>>, b: &Arc<PolyRing<AlgebraicField>>) -> bool {
    Arc::ptr_eq(a, b) || (a.names == b.names && a.domain == b.domain)
}

impl Tower {
    pub fn new(assumptions: &[Basic]) -> Tower {
        let domain: AlgebraicField = rationals();
        let ring = ring_of(&domain, vec![PLACEHOLDER.to_string()]);
        Tower {
            assumptions: assumptions.to_vec(),
            extensions: Vec::new(),
            numbers: Vec::new(),
            coordinates: Vec::new(),
            theta: None,
            theta_expression: int(0),
            degrees: Vec::new(),
            to_basis: Vec::new(),
            basis_expressions: Vec::new(),
            domain,
            embeddings: Vec::new(),
            generators: Vec::new(),
            ring,
            pairs: Vec::new(),
            variables: Vec::new(),
            cache: HashMap::new(),
            derivatives: HashMap::new(),
            prime_logs: BTreeMap::new(),
            minus_one: -1,
            kummer: None,
            signs: HashMap::new(),
        }
    }

    fn names(&self) -> Vec<String> {
        if self.generators.is_empty() {
            vec![PLACEHOLDER.to_string()]
        } else {
            self.generators.iter().map(|g| g.name.clone()).collect()
        }
    }

    fn rebuild(&mut self) {
        self.ring = ring_of(&self.domain, self.names());
        self.kummer = None;
    }

    fn count(ring: &PolyRing<AlgebraicField>) -> usize {
        if ring.names == [PLACEHOLDER.to_string()] {
            0
        } else {
            ring.ngens
        }
    }

    fn is_rationals(&self) -> bool {
        self.domain.degree == 1
    }

    /// `c` (over an older field) in the current field, its generator
    /// written `theta` (`_embed`).
    fn embed(&self, c: &Coef, theta: &Coef) -> Coef {
        let k: &AlgebraicField = &self.domain;
        let mut result: Coef = k.zero();
        for coefficient in c {
            result = k.add(&k.mul(&result, theta), &k.from_rational(coefficient));
        }
        result
    }

    /// A polynomial of an older ring in the current one (`_poly`).
    fn poly(&self, p: &TP) -> TP {
        if same_ring(&p.ring, &self.ring) {
            return p.clone();
        }
        let old: usize = Tower::count(&p.ring);
        let width: usize = self.generators.len().max(1);
        let theta: Option<Coef> = if p.ring.domain != self.domain {
            self.embeddings
                .iter()
                .find(|(f, _)| *f == p.ring.domain)
                .map(|(_, c)| c.clone())
        } else {
            None
        };
        let mut terms: Terms<Coef> = Terms::new();
        for (m, c) in &p.terms {
            let mut monomial: Monomial = m[..old].to_vec();
            monomial.resize(width, 0);
            let value: Coef = match &theta {
                Some(t) => self.embed(c, t),
                None => c.clone(),
            };
            terms.insert(monomial, value);
        }
        self.ring.from_dict(terms)
    }

    fn lift(&self, a: &Element) -> Element {
        if same_ring(&a.num.ring, &self.ring) && same_ring(&a.den.ring, &self.ring) {
            return a.clone();
        }
        Element {
            num: self.poly(&a.num),
            den: self.poly(&a.den),
        }
    }

    fn cancel(&self, num: &TP, den: &TP) -> PolyResult<(TP, TP)> {
        if self.is_rationals() {
            use crate::polys::domains::{SparsePolynomialRing, QQ};
            let r = SparsePolynomialRing::sympy(QQ, self.ring.names.clone());
            let convert = |p: &TP| {
                let terms: Terms<Rat> = p.terms.iter().map(|(m, c)| (m.clone(), c[0].clone())).collect();
                r.ring.from_dict(terms)
            };
            let (p, q) = r.cancel(&convert(num), &convert(den));
            let back = |p: &PolyElement<crate::polys::domains::RationalField>| {
                let terms: Terms<Coef> = p.terms.iter().map(|(m, c)| (m.clone(), vec![c.clone()])).collect();
                self.ring.from_dict(terms)
            };
            return Ok((back(&p), back(&q)));
        }
        let polynomials =
            crate::polys::algebraicfactor::AlgebraicPolynomialRing::new(self.domain.clone(), self.ring.names.clone())?;
        let h: TP = polynomials.gcd(num, den);
        if h.is_ground() {
            return Ok((num.clone(), den.clone()));
        }
        Ok((num.exquo(&h)?, den.exquo(&h)?))
    }

    fn fraction(&self, num: TP, den: TP) -> PolyResult<Element> {
        if den.is_zero() {
            return Err(zero_division("division by the zero of the tower"));
        }
        if num.is_zero() {
            return Ok(Element {
                num: self.ring.zero(),
                den: self.ring.one(),
            });
        }
        let (mut p, mut q) = self.cancel(&num, &den)?;
        let lead: Coef = q.LC();
        if !self.domain.is_one(&lead) {
            let inverse: Coef = self.domain.invert(&lead)?;
            p = p.mul_ground(&inverse);
            q = q.mul_ground(&inverse);
        }
        Ok(Element { num: p, den: q })
    }

    fn combine(&self, a: &Element, b: &Element, sign: i32) -> PolyResult<Element> {
        let (a, b) = (self.lift(a), self.lift(b));
        if a.den == b.den {
            let n: TP = if sign > 0 { a.num.add(&b.num) } else { a.num.sub(&b.num) };
            return self.fraction(n, a.den.clone());
        }
        let (x, y) = (a.num.mul(&b.den), b.num.mul(&a.den));
        self.fraction(if sign > 0 { x.add(&y) } else { x.sub(&y) }, a.den.mul(&b.den))
    }

    pub fn add(&self, a: &Element, b: &Element) -> PolyResult<Element> {
        self.combine(a, b, 1)
    }
    pub fn sub(&self, a: &Element, b: &Element) -> PolyResult<Element> {
        self.combine(a, b, -1)
    }
    pub fn mul(&self, a: &Element, b: &Element) -> PolyResult<Element> {
        let (a, b) = (self.lift(a), self.lift(b));
        self.fraction(a.num.mul(&b.num), a.den.mul(&b.den))
    }
    pub fn div(&self, a: &Element, b: &Element) -> PolyResult<Element> {
        let (a, b) = (self.lift(a), self.lift(b));
        if b.num.is_zero() {
            return Err(zero_division("division by the zero of the tower"));
        }
        self.fraction(a.num.mul(&b.den), a.den.mul(&b.num))
    }
    pub fn neg(&self, a: &Element) -> Element {
        Element {
            num: a.num.neg(),
            den: a.den.clone(),
        }
    }
    pub fn pow(&self, a: &Element, n: i64) -> PolyResult<Element> {
        if n < 0 {
            if a.num.is_zero() {
                return Err(zero_division("a negative power of the zero of the tower"));
            }
            return Ok(Element {
                num: a.den.pow((-n) as u64),
                den: a.num.pow((-n) as u64),
            });
        }
        Ok(Element {
            num: a.num.pow(n as u64),
            den: a.den.pow(n as u64),
        })
    }
    /// `a` times a rational (`times`).
    pub fn times(&self, a: &Element, r: &Rat) -> PolyResult<Element> {
        let a: Element = self.lift(a);
        let k: &AlgebraicField = &self.domain;
        self.fraction(
            a.num.mul_ground(&k.from_integer(&numer(r))),
            a.den.mul_ground(&k.from_integer(&denom(r))),
        )
    }

    /// The rational `value` (`constant`).
    pub fn constant(&self, value: &Rat) -> Element {
        let k: &AlgebraicField = &self.domain;
        Element {
            num: self.ring.ground(k.from_integer(&numer(value))),
            den: self.ring.ground(k.from_integer(&denom(value))),
        }
    }
    pub fn zero(&self) -> Element {
        Element {
            num: self.ring.zero(),
            den: self.ring.one(),
        }
    }
    pub fn one(&self) -> Element {
        Element {
            num: self.ring.one(),
            den: self.ring.one(),
        }
    }
    fn generator_element(&self, position: usize) -> Element {
        Element {
            num: self.ring.gens()[position].clone(),
            den: self.ring.one(),
        }
    }
    fn poly_element(&self, p: TP) -> Element {
        Element {
            num: p,
            den: self.ring.one(),
        }
    }
    fn add_generator(&mut self, g: Generator) -> usize {
        self.generators.push(g);
        self.rebuild();
        self.generators.len() - 1
    }
    fn new_name(&self) -> String {
        format!("g{}", self.generators.len())
    }

    /// The powers of the roots reduced by their relations (`_reduce`).
    fn reduce(&self, p: &TP) -> TP {
        let mut p: TP = self.poly(p);
        let mut changed: bool = true;
        while changed {
            changed = false;
            for position in (0..self.generators.len()).rev() {
                let g: &Generator = &self.generators[position];
                if g.kind != Kind::Root || g.eliminated.is_some() || g.radicand.is_none() {
                    continue;
                }
                if p.degree_in(position) < g.index {
                    continue;
                }
                let radicand: TP = self.lift(g.radicand.as_ref().expect("a radicand")).num;
                let mut result: TP = self.ring.zero();
                for (monomial, coefficient) in &p.terms {
                    let e: i64 = monomial[position] as i64;
                    let mut rest: Monomial = monomial.clone();
                    rest[position] = (e % g.index) as u32;
                    let mut term: TP = self.ring.term(rest, coefficient.clone());
                    if e >= g.index {
                        term = term.mul(&radicand.pow((e / g.index) as u64));
                    }
                    result = result.add(&term);
                }
                p = result;
                changed = true;
            }
        }
        p
    }

    fn substituted(&self, p: &TP, position: usize, value: &Element) -> PolyResult<Element> {
        let mut parts: BTreeMap<u32, TP> = BTreeMap::new();
        let mut order: Vec<u32> = Vec::new();
        for (monomial, coefficient) in &p.terms {
            let e: u32 = monomial[position];
            let mut rest: Monomial = monomial.clone();
            rest[position] = 0;
            let term: TP = self.ring.term(rest, coefficient.clone());
            if !parts.contains_key(&e) {
                order.push(e);
            }
            let entry = parts.entry(e).or_insert_with(|| self.ring.zero());
            *entry = entry.add(&term);
        }
        let mut total: Element = self.zero();
        for e in order {
            let part: TP = parts[&e].clone();
            total = self.add(
                &total,
                &self.mul(&self.poly_element(part), &self.pow(value, e as i64)?)?,
            )?;
        }
        Ok(total)
    }

    /// The normal form: eliminated generators substituted, roots reduced
    /// (`normal`).
    pub fn normal(&self, a: &Element) -> PolyResult<Element> {
        let mut a: Element = self.lift(a);
        for position in 0..self.generators.len() {
            let Some(eliminated) = self.generators[position].eliminated.clone() else {
                continue;
            };
            if a.num.degree_in(position) > 0 || a.den.degree_in(position) > 0 {
                let value: Element = self.lift(&eliminated);
                let n: Element = self.substituted(&a.num, position, &value)?;
                let d: Element = self.substituted(&a.den, position, &value)?;
                a = self.lift(&self.div(&n, &d)?);
            }
        }
        let (num, den) = (self.reduce(&a.num), self.reduce(&a.den));
        if den.is_zero() {
            return Err(zero_division("division by the zero of the tower"));
        }
        self.fraction(num, den)
    }

    /// Whether `a` is zero (`vanishes`).
    pub fn vanishes(&self, a: &Element) -> PolyResult<bool> {
        Ok(self.normal(a)?.num.is_zero())
    }
}

// ---------------------------------------------------------------------
// derivations, linear systems, the number field

impl Tower {
    fn variable_positions(&self) -> Vec<usize> {
        (0..self.generators.len())
            .filter(|&i| self.generators[i].kind == Kind::Variable)
            .collect()
    }

    fn generator_derivative(&mut self, position: usize, variable: usize) -> PolyResult<Element> {
        if let Some(v) = self.derivatives.get(&(position, variable)) {
            return Ok(v.clone());
        }
        let g: Generator = self.generators[position].clone();
        let value: Element = match g.kind {
            Kind::Variable => {
                if position == variable {
                    self.one()
                } else {
                    self.zero()
                }
            }
            Kind::Exponential if g.argument.is_some() => {
                let d: Element = self.derivative(g.argument.as_ref().expect("argument"), variable)?;
                self.mul(&d, &self.generator_element(position))?
            }
            Kind::Logarithm if g.argument.is_some() => {
                let arg: Element = g.argument.clone().expect("argument");
                let d: Element = self.derivative(&arg, variable)?;
                self.div(&d, &arg)?
            }
            Kind::Root if g.radicand.is_some() => {
                let radicand: Element = g.radicand.clone().expect("radicand");
                let d: Element = self.derivative(&radicand, variable)?;
                let top: Element = self.mul(&d, &self.generator_element(position))?;
                self.div(&top, &self.times(&radicand, &rat(g.index))?)?
            }
            _ => self.zero(),
        };
        self.derivatives.insert((position, variable), value.clone());
        Ok(value)
    }

    fn polynomial_derivative(&mut self, p: &TP, variable: usize) -> PolyResult<Element> {
        let mut total: Element = self.zero();
        for position in 0..self.generators.len() {
            if p.degree_in(position) <= 0 {
                continue;
            }
            let partial: TP = self.poly(p).diff(position);
            if partial.is_zero() {
                continue;
            }
            let dg: Element = self.generator_derivative(position, variable)?;
            total = self.add(&total, &self.mul(&self.poly_element(partial), &dg)?)?;
        }
        Ok(total)
    }

    /// The derivative with respect to the variable generator `variable`
    /// (`derivative`).
    pub fn derivative(&mut self, a: &Element, variable: usize) -> PolyResult<Element> {
        let a: Element = self.normal(a)?;
        let n: Element = self.poly_element(a.num.clone());
        let d: Element = self.poly_element(a.den.clone());
        let dn: Element = self.polynomial_derivative(&a.num, variable)?;
        let dd: Element = self.polynomial_derivative(&a.den, variable)?;
        if self.lift(&dd).num.is_zero() {
            return self.div(&dn, &d);
        }
        let top: Element = self.sub(&self.mul(&dn, &d)?, &self.mul(&n, &dd)?)?;
        self.div(&top, &self.mul(&d, &d)?)
    }

    fn formal_polynomial(&mut self, p: &TP, position: usize) -> PolyResult<Element> {
        let p: TP = self.poly(p);
        let mut total: Element = self.poly_element(p.diff(position));
        for j in 0..self.generators.len() {
            let g: Generator = self.generators[j].clone();
            if g.kind != Kind::Root || g.eliminated.is_some() || g.radicand.is_none() || p.degree_in(j) <= 0 {
                continue;
            }
            let radicand: Element = g.radicand.clone().expect("radicand");
            let inner: Element = self.formal_derivative(&radicand, position)?;
            if self.lift(&inner).num.is_zero() {
                continue;
            }
            let t: Element = self.mul(
                &self.mul(&self.poly_element(p.diff(j)), &inner)?,
                &self.generator_element(j),
            )?;
            total = self.add(&total, &self.div(&t, &self.times(&radicand, &rat(g.index))?)?)?;
        }
        Ok(total)
    }

    /// The derivative in the generator `position`, the others constant
    /// but for the roots of radicands in it (`_formal_derivative`).
    fn formal_derivative(&mut self, a: &Element, position: usize) -> PolyResult<Element> {
        let a: Element = self.normal(a)?;
        let n: Element = self.poly_element(a.num.clone());
        let d: Element = self.poly_element(a.den.clone());
        let fn_: Element = self.formal_polynomial(&a.num, position)?;
        let fd: Element = self.formal_polynomial(&a.den, position)?;
        let top: Element = self.sub(&self.mul(&fn_, &d)?, &self.mul(&n, &fd)?)?;
        self.div(&top, &self.mul(&d, &d)?)
    }

    /// Whether every derivation of the variables kills `a` (`is_constant`).
    pub fn is_constant(&mut self, a: &Element) -> PolyResult<bool> {
        for k in self.variable_positions() {
            let d: Element = self.derivative(a, k)?;
            if !self.vanishes(&d)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn field_degree(&self) -> usize {
        self.domain.degree
    }

    fn components(&self, c: &Coef) -> Vec<Rat> {
        let degree: usize = self.field_degree();
        let mut out: Vec<Rat> = vec![Rat::ZERO; degree.saturating_sub(c.len())];
        out.extend(c.iter().cloned());
        out
    }

    /// The rational solution `r` of `target = sum r_i basis_i` for every
    /// equation, `None` when there is none (`_solve`).
    fn solve(&self, equations: &[(Element, Vec<Element>)]) -> PolyResult<Option<Vec<Rat>>> {
        if equations.is_empty() {
            return Ok(Some(vec![]));
        }
        let unknowns: usize = equations[0].1.len();
        let width: usize = self.field_degree();
        let mut rows: Vec<Vec<Rat>> = Vec::new();
        for (target, basis) in equations {
            let mut members: Vec<Element> = vec![self.normal(target)?];
            for b in basis {
                members.push(self.normal(b)?);
            }
            let mut common: TP = self.ring.one();
            for m in &members {
                common = self.lcm(&common, &m.den)?;
            }
            let mut numerators: Vec<TP> = Vec::new();
            for m in &members {
                numerators.push(self.reduce(&m.num.mul(&common.exquo(&self.poly(&m.den))?)));
            }
            // a set of monomials: the order of the rows does not change the
            // solution of the system
            let mut monomials: Vec<Monomial> = Vec::new();
            for p in &numerators {
                for m in p.terms.keys() {
                    if !monomials.contains(m) {
                        monomials.push(m.clone());
                    }
                }
            }
            for monomial in monomials {
                let columns: Vec<Vec<Rat>> = numerators
                    .iter()
                    .map(|p| match p.terms.get(&monomial) {
                        Some(c) => self.components(c),
                        None => vec![Rat::ZERO; width],
                    })
                    .collect();
                for k in 0..width {
                    let mut row: Vec<Rat> = (0..unknowns).map(|i| columns[i + 1][k].clone()).collect();
                    row.push(columns[0][k].clone());
                    rows.push(row);
                }
            }
        }
        rows.retain(|row| row.iter().any(|v| *v != Rat::ZERO));
        if rows.is_empty() {
            return Ok(Some(vec![Rat::ZERO; unknowns]));
        }
        if unknowns == 0 {
            return Ok(None);
        }
        let (table, pivots) = rref(&rows, unknowns + 1);
        if pivots.contains(&unknowns) {
            return Ok(None);
        }
        let mut solution: Vec<Rat> = vec![Rat::ZERO; unknowns];
        for (row, pivot) in pivots.iter().enumerate() {
            solution[*pivot] = table[row][unknowns].clone();
        }
        Ok(Some(solution))
    }

    fn lcm(&self, a: &TP, b: &TP) -> PolyResult<TP> {
        let (a, b) = (self.poly(a), self.poly(b));
        if a.is_ground() {
            return Ok(b);
        }
        if b.is_ground() || a == b {
            return Ok(a);
        }
        let (_, cofactor) = self.cancel(&a, &b)?;
        Ok(a.mul(&cofactor))
    }

    /// An algebraic number of the field, which grows to hold it (`_number`).
    fn number(&mut self, value: &Basic) -> PolyResult<Element> {
        if is_rational(value) {
            return Ok(self.constant(&rational_of_expr(value)?));
        }
        let converted: Coef = match self.field_element(value) {
            Ok(c) => c,
            Err(e) if is_not_in_field(&e) || is_arithmetic(&e) => {
                let n: Algebraic = match algebraic(value) {
                    Ok(n) => n,
                    Err(e) if is_not_in_field(&e) || is_arithmetic(&e) => {
                        return Err(not_elementary(format!(
                            "{} is not an algebraic number this module can hold",
                            value
                        )))
                    }
                    Err(e) => return Err(e),
                };
                if n.is_rational() {
                    return Ok(self.constant(&n.rational_value()?));
                }
                let degree: usize = n.degree();
                if degree * self.field_degree() > FIELD_DEGREE_LIMIT {
                    return Err(not_elementary(format!(
                        "the number field of {} would have degree up to {}",
                        value,
                        degree * self.field_degree()
                    )));
                }
                let (_, core) = crate::core::expr::as_coeff_mul(value);
                let before: usize = self.field_degree();
                let result: PolyResult<Coef> = (|| {
                    self.extend(&core)?;
                    self.field_element(value)
                })();
                if self.field_degree() != before {
                    self.rebuild();
                    self.derivatives.clear();
                }
                match result {
                    Ok(c) => c,
                    Err(e) if is_not_in_field(&e) || is_arithmetic(&e) => {
                        return Err(not_elementary(format!(
                            "{} is not an algebraic number this module can hold",
                            value
                        )))
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(e) => return Err(e),
        };
        Ok(Element {
            num: self.ring.ground(converted),
            den: self.ring.one(),
        })
    }

    /// The field extended by an algebraic number (`_extend`).
    fn extend(&mut self, extension: &Basic) -> PolyResult<()> {
        let n: Algebraic = algebraic(extension)?;
        if n.is_rational() {
            return Err(not_in_field(format!("{} is rational", extension)));
        }
        let (theta, old, new): (Algebraic, Vec<Rat>, Vec<Rat>) = match &self.theta {
            None => (n.clone(), vec![Rat::ONE, Rat::ZERO], vec![Rat::ONE, Rat::ZERO]),
            Some(t) => crate::polys::numberfields::qqbar::primitive_element(t, &n)?,
        };
        if theta.degree() == self.field_degree() {
            return Err(not_in_field(format!("{} is in the field already", extension)));
        }
        let minpoly: Vec<Int> = theta.minpoly();
        let leading: Rat = Rat::from(minpoly[0].clone());
        let monic: Vec<Rat> = minpoly.iter().map(|c| Rat::from(c.clone()) / &leading).collect();
        let mut expression: Basic = extension.clone();
        if self.theta.is_some() {
            let k0: AlgebraicField = AlgebraicField::new(monic.clone(), int(0))?;
            let ratio: Coef = k0.quo(&k0.sub(&k0.alpha(), &k0.reduce(old.clone())), &k0.reduce(new.clone()))?;
            let c: Rat = ratio.first().cloned().unwrap_or(Rat::ZERO);
            expression = crate::core::add::add(&[
                self.theta_expression.clone(),
                crate::core::mul::mul(&[number(&c), extension.clone()])?,
            ])?;
        }
        let k: AlgebraicField = AlgebraicField::new(monic, expression.clone())?;
        let previous: AlgebraicField = self.domain.clone();
        let embedding: Coef = k.reduce(old);
        self.domain = k.clone();
        let embeddings: Vec<(AlgebraicField, Coef)> = self
            .embeddings
            .iter()
            .map(|(f, rep)| (f.clone(), self.embed(rep, &embedding)))
            .collect();
        self.embeddings = embeddings;
        self.embeddings.push((previous.clone(), embedding.clone()));
        let mut coordinates: Vec<Coef> = self.coordinates.iter().map(|c| self.embed(c, &embedding)).collect();
        coordinates.push(k.reduce(new));
        self.coordinates = coordinates;
        self.extensions.push(extension.clone());
        self.numbers.push(n);
        self.degrees.push(k.degree / previous.degree);
        self.theta = Some(theta);
        self.theta_expression = expression;
        self.monomial_basis()
    }

    fn monomial_basis(&mut self) -> PolyResult<()> {
        let k: AlgebraicField = self.domain.clone();
        let n: usize = k.degree;
        let mut exponents: Vec<Vec<usize>> = vec![vec![]];
        for &d in &self.degrees {
            let mut next: Vec<Vec<usize>> = Vec::new();
            for e in &exponents {
                for j in 0..d {
                    let mut f: Vec<usize> = e.clone();
                    f.push(j);
                    next.push(f);
                }
            }
            exponents = next;
        }
        let mut rows: Vec<Vec<Rat>> = Vec::new();
        let mut expressions: Vec<Basic> = Vec::new();
        for e in &exponents {
            let mut c: Coef = k.one();
            let mut factors: Vec<Basic> = Vec::new();
            for ((kk, coordinates), extension) in e.iter().zip(self.coordinates.iter()).zip(self.extensions.iter()) {
                if *kk > 0 {
                    c = k.mul(&c, &k.pow(coordinates, *kk as u64));
                    factors.push(crate::core::power::pow(extension, &int(*kk as i64))?);
                }
            }
            rows.push(self.components(&c));
            expressions.push(if factors.is_empty() {
                int(1)
            } else {
                crate::core::mul::mul(&factors)?
            });
        }
        let width: usize = rows.len();
        let augmented: Vec<Vec<Rat>> = (0..n)
            .map(|i| {
                let mut row: Vec<Rat> = (0..width).map(|j| rows[j][i].clone()).collect();
                row.extend((0..n).map(|kk| if i == kk { Rat::ONE } else { Rat::ZERO }));
                row
            })
            .collect();
        let (reduced, pivots) = rref(&augmented, width);
        if pivots != (0..width).collect::<Vec<usize>>() {
            return Err(not_in_field(
                "the products of the extensions are not a basis".to_string(),
            ));
        }
        self.to_basis = reduced[..width].iter().map(|row| row[width..].to_vec()).collect();
        self.basis_expressions = expressions;
        Ok(())
    }

    /// `value` as an element of the current field (`_field_element`).
    fn field_element(&self, value: &Basic) -> PolyResult<Coef> {
        let k: &AlgebraicField = &self.domain;
        if is_rational(value) {
            return Ok(k.from_rational(&rational_of_expr(value)?));
        }
        for (extension, coordinates) in self.extensions.iter().zip(self.coordinates.iter()) {
            if value == extension {
                return Ok(coordinates.clone());
            }
        }
        let structural: PolyResult<Option<Coef>> = (|| {
            if value.is_add() {
                let mut total: Coef = k.zero();
                for t in value.args() {
                    total = k.add(&total, &self.field_element(t)?);
                }
                return Ok(Some(total));
            }
            if value.is_mul() {
                let mut product: Coef = k.one();
                for f in value.args() {
                    product = k.mul(&product, &self.field_element(f)?);
                }
                return Ok(Some(product));
            }
            if value.is_pow() && value.args()[1].is_integer_value() {
                let base: Coef = self.field_element(&value.args()[0])?;
                let nn: i64 = small(&numer(&rational_of_expr(&value.args()[1])?));
                if nn < 0 {
                    if base.is_empty() {
                        return Err(not_in_field("a negative power of zero".to_string()));
                    }
                    return Ok(Some(k.pow(&k.invert(&base)?, (-nn) as u64)));
                }
                return Ok(Some(k.pow(&base, nn as u64)));
            }
            Ok(None)
        })();
        match structural {
            Ok(Some(c)) => return Ok(c),
            Ok(None) => {}
            Err(e) if is_not_in_field(&e) => {}
            Err(e) => return Err(e),
        }
        let Some(theta) = self.theta.clone() else {
            return Err(not_in_field(format!("{} is not rational", value)));
        };
        if let Some(power) = self.radical_element(value)? {
            return Ok(power);
        }
        let n: Algebraic = algebraic(value)?;
        if n.is_rational() {
            return Ok(k.from_rational(&n.rational_value()?));
        }
        let minpoly: Vec<Int> = n.minpoly();
        let leading: Rat = Rat::from(minpoly[0].clone());
        let f: Vec<Coef> = minpoly
            .iter()
            .map(|c| k.from_rational(&(Rat::from(c.clone()) / &leading)))
            .collect();
        let (_, factors) = crate::polys::factortools::dup_ext_factor(&f, k)?;
        for (g, _) in factors {
            if g.len() == 2 {
                let root: Coef = k.neg(&k.quo(&g[1], &g[0])?);
                if theta.polynomial_value(&root)?.equals(&n)? {
                    return Ok(root);
                }
            }
        }
        Err(not_in_field(format!("{} is not in {}", value, k.name())))
    }

    fn radical_element(&self, value: &Basic) -> PolyResult<Option<Coef>> {
        let Some((base, r)) = rational_power(value) else {
            return Ok(None);
        };
        if let Some(direct) = self.power_of_extensions(&base, &r)? {
            return Ok(Some(direct));
        }
        let k: &AlgebraicField = &self.domain;
        let mut result: Coef = k.one();
        let mut parts: Vec<(Rat, Rat)> = if crate::numbers::rat_is_negative(&base) {
            vec![(rat(-1), r.clone())]
        } else {
            vec![]
        };
        let absn: Int = {
            let n: Int = numer(&base);
            if n < Int::ZERO {
                -n
            } else {
                n
            }
        };
        for (n, sign) in [(absn, 1i64), (denom(&base), -1i64)] {
            for (prime, exponent) in crate::ntheory::factor_::factorint(&n, None) {
                parts.push((Rat::from(prime), &r * rat(exponent as i64 * sign)));
            }
        }
        if parts.len() < 2 {
            return Ok(None);
        }
        for (b, t) in parts {
            let found: Option<Coef> = if denom(&t) > Int::ONE {
                self.power_of_extensions(&b, &t)?
            } else if t >= Rat::ZERO {
                Some(k.pow(&k.from_rational(&b), small(&numer(&t)) as u64))
            } else {
                Some(k.pow(&k.invert(&k.from_rational(&b))?, (-small(&numer(&t))) as u64))
            };
            let Some(found) = found else { return Ok(None) };
            result = k.mul(&result, &found);
        }
        Ok(Some(result))
    }

    fn power_of_extensions(&self, base: &Rat, r: &Rat) -> PolyResult<Option<Coef>> {
        let mut same: Vec<(Rat, Coef)> = Vec::new();
        for (extension, coordinates) in self.extensions.iter().zip(self.coordinates.iter()) {
            if let Some((b, e)) = rational_power(extension) {
                if b == *base {
                    same.push((e, coordinates.clone()));
                }
            }
        }
        if same.is_empty() {
            return Ok(None);
        }
        let l: i64 = lcm_all(same.iter().map(|(e, _)| small(&denom(e))).chain([small(&denom(r))]));
        let steps: Vec<i64> = same.iter().map(|(e, _)| small(&numer(&(e * rat(l))))).collect();
        let mut g: i64 = steps[0];
        let mut coefficients: Vec<i64> = vec![0; steps.len()];
        coefficients[0] = 1;
        for i in 1..steps.len() {
            let (d, u, v) = extended_gcd(g, steps[i]);
            coefficients = coefficients.iter().map(|c| c * u).collect();
            coefficients[i] = v;
            g = d;
        }
        let target: i64 = small(&numer(&(r * rat(l))));
        if target.rem_euclid(g.abs()) != 0 {
            return Ok(None);
        }
        let k: &AlgebraicField = &self.domain;
        let mut result: Coef = k.one();
        for (n, (_, coordinates)) in coefficients.iter().zip(same.iter()) {
            let n: i64 = n * floor_div_i(target, g);
            if n != 0 {
                let p: Coef = if n > 0 {
                    k.pow(coordinates, n as u64)
                } else {
                    k.pow(&k.invert(coordinates)?, (-n) as u64)
                };
                result = k.mul(&result, &p);
            }
        }
        Ok(Some(result))
    }

    fn coefficient_expr(&self, c: &Coef) -> R {
        if c.len() <= 1 {
            return Ok(match c.first() {
                Some(v) => number(v),
                None => int(0),
            });
        }
        let vector: Vec<Rat> = self.components(c);
        let mut terms: Vec<Basic> = Vec::new();
        for (row, expression) in self.to_basis.iter().zip(self.basis_expressions.iter()) {
            let mut y: Rat = Rat::ZERO;
            for (a, b) in row.iter().zip(vector.iter()) {
                y += a * b;
            }
            if y != Rat::ZERO {
                terms.push(crate::core::mul::mul(&[number(&y), expression.clone()])?);
            }
        }
        if terms.is_empty() {
            Ok(int(0))
        } else {
            crate::core::add::add(&terms)
        }
    }

    /// The number which `a` is, when it is one (`as_number`).
    pub fn as_number(&self, a: &Element) -> PolyResult<Option<Basic>> {
        let a: Element = self.normal(a)?;
        if !a.num.is_ground() || !a.den.is_ground() {
            return Ok(None);
        }
        Ok(Some(crate::core::mul::div(
            &self.coefficient_expr(&a.num.LC())?,
            &self.coefficient_expr(&a.den.LC())?,
        )?))
    }
}

// ---------------------------------------------------------------------
// signs, exponentials, roots

fn rel_gt(a: &Basic) -> R {
    crate::core::relational::gt(a, &int(0))
}
fn rel_lt(a: &Basic) -> R {
    crate::core::relational::lt(a, &int(0))
}

impl Tower {
    fn ask_(&self, query: &Basic, more: &[Basic]) -> PolyResult<Option<bool>> {
        let mut facts: Vec<Basic> = self.assumptions.clone();
        facts.extend(more.iter().cloned());
        let limit: f64 = match crate::settings::settings().timeout {
            None => ASK_SECONDS,
            Some(t) => ASK_SECONDS.min(t),
        };
        Ok(crate::timeout::attempt_within(Some(limit), || ask(query, &facts, None))?.flatten())
    }

    fn positive(&mut self, a: &Element) -> PolyResult<bool> {
        let e: Basic = self.to_expr(a)?;
        Ok(self.sign(&e)? > 0)
    }

    fn negative(&mut self, a: &Element) -> PolyResult<bool> {
        let e: Basic = self.to_expr(a)?;
        Ok(self.sign(&e)? < 0)
    }

    fn sign(&mut self, value: &Basic) -> PolyResult<i32> {
        if let Some(s) = self.signs.get(value) {
            return Ok(*s);
        }
        let mut sign: i32 = 0;
        if is(value, Predicate::positive) == Some(true) {
            sign = 1;
        } else if is(value, Predicate::negative) == Some(true) {
            sign = -1;
        } else if !value.free_symbols().is_empty()
            && is(value, Predicate::extended_real) != Some(false)
            && !value.has(&Basic::atom(Head::ImaginaryUnit))
        {
            if self.ask_(&rel_gt(value)?, &[])? == Some(true) {
                sign = 1;
            } else if self.ask_(&rel_lt(value)?, &[])? == Some(true) {
                sign = -1;
            }
        }
        self.signs.insert(value.clone(), sign);
        Ok(sign)
    }

    fn real(&mut self, a: &Element) -> PolyResult<bool> {
        let value: Basic = self.to_expr(a)?;
        if is(&value, Predicate::extended_real) == Some(true) {
            return Ok(true);
        }
        if value.free_symbols().is_empty() {
            return Ok(false);
        }
        let member: Basic = crate::assumptions::facts::element(&value, &crate::sets::sets::reals())?;
        Ok(self.ask_(&member, &[])? == Some(true))
    }

    fn split_pairs(&mut self) -> PolyResult<(Vec<usize>, Vec<usize>)> {
        let mut moving: Vec<usize> = Vec::new();
        let mut still: Vec<usize> = Vec::new();
        for i in 0..self.pairs.len() {
            let z: Element = self.pairs[i].z.clone();
            if self.is_constant(&z)? {
                still.push(i);
            } else {
                moving.push(i);
            }
        }
        Ok((moving, still))
    }

    fn pair_zs(&self, indices: &[usize]) -> Vec<Element> {
        indices.iter().map(|&i| self.pairs[i].z.clone()).collect()
    }

    /// `exp(a)` in the tower (`exponential`).
    pub fn exponential(&mut self, a: &Element) -> PolyResult<Element> {
        let a: Element = self.normal(a)?;
        if a.num.is_zero() {
            return Ok(self.one());
        }
        let exact: Option<Vec<Rat>> = if self.pairs.is_empty() {
            None
        } else {
            let all: Vec<usize> = (0..self.pairs.len()).collect();
            self.solve(&[(a.clone(), self.pair_zs(&all))])?
        };
        if let Some(exact) = exact {
            let mut result: Element = self.one();
            for (i, r) in exact.iter().enumerate() {
                if *r != Rat::ZERO {
                    let p: Element = self.pair_power(i, r)?;
                    result = self.mul(&result, &p)?;
                }
            }
            return self.normal(&result);
        }
        let variables: Vec<usize> = self.variable_positions();
        let (moving, still) = self.split_pairs()?;
        let mut factors: Vec<(usize, Rat)> = Vec::new();
        let mut constant: Element = a.clone();
        if !self.is_constant(&a)? {
            let found: Option<Vec<Rat>> = if moving.is_empty() {
                None
            } else {
                let mut equations: Vec<(Element, Vec<Element>)> = Vec::new();
                for &k in &variables {
                    let target: Element = self.derivative(&a, k)?;
                    let mut basis: Vec<Element> = Vec::new();
                    for &i in &moving {
                        let z: Element = self.pairs[i].z.clone();
                        basis.push(self.derivative(&z, k)?);
                    }
                    equations.push((target, basis));
                }
                self.solve(&equations)?
            };
            let Some(found) = found else {
                return self.new_exponential(&a, true);
            };
            for (&i, r) in moving.iter().zip(found.iter()) {
                if *r != Rat::ZERO {
                    factors.push((i, r.clone()));
                    let z: Element = self.pairs[i].z.clone();
                    constant = self.sub(&constant, &self.times(&z, r)?)?;
                }
            }
            constant = self.normal(&constant)?;
            if !self.is_constant(&constant)? {
                return self.new_exponential(&a, true);
            }
        }
        let mut result: Element = self.one();
        if !constant.num.is_zero() {
            let found: Option<Vec<Rat>> = if still.is_empty() {
                None
            } else {
                self.solve(&[(constant.clone(), self.pair_zs(&still))])?
            };
            match found {
                None => {
                    let certified: bool = self.free_of_functions(&constant)?;
                    result = self.new_exponential(&constant, certified)?;
                }
                Some(found) => {
                    for (&j, s) in still.iter().zip(found.iter()) {
                        if *s != Rat::ZERO {
                            factors.push((j, s.clone()));
                        }
                    }
                }
            }
        }
        for (i, r) in factors {
            let p: Element = self.pair_power(i, &r)?;
            result = self.mul(&result, &p)?;
        }
        self.normal(&result)
    }

    fn free_of_functions(&mut self, a: &Element) -> PolyResult<bool> {
        let a: Element = self.normal(a)?;
        for position in 0..self.generators.len() {
            if a.num.degree_in(position) > 0 || a.den.degree_in(position) > 0 {
                let g: Element = self.generator_element(position);
                if !self.is_constant(&g)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn sound(&self) -> bool {
        self.generators.iter().all(|g| g.certified)
    }

    fn new_exponential(&mut self, a: &Element, certified: bool) -> PolyResult<Element> {
        let certified: bool = certified && self.sound();
        let expression: Basic = crate::functions::elementary::exponential::exp(&self.to_expr(a)?)?;
        let name: String = self.new_name();
        let position: usize = self.add_generator(Generator::new(
            Kind::Exponential,
            name,
            expression,
            Some(a.clone()),
            certified,
        ));
        let theta: Element = self.generator_element(position);
        self.pairs.push(Pair {
            z: a.clone(),
            e: theta.clone(),
            generator: position,
            logarithmic: false,
            master: -1,
        });
        Ok(theta)
    }

    fn pair_power(&mut self, i: usize, r: &Rat) -> PolyResult<Element> {
        let (p, q) = (small(&numer(r)), small(&denom(r)));
        if q == 1 {
            let e: Element = self.pairs[i].e.clone();
            return self.pow(&e, p);
        }
        let root: Element = self.root(i, q)?;
        self.pow(&root, p)
    }

    /// `exp(z_i/q)` (`_root`).
    fn root(&mut self, i: usize, q: i64) -> PolyResult<Element> {
        let pair: Pair = self.pairs[i].clone();
        let nb: Option<Basic> = self.as_number(&pair.e)?;
        if let (Some(nb), true) = (&nb, pair.logarithmic) {
            let v: Basic = crate::core::power::pow(nb, &number(&frac(1, q)))?;
            return self.number(&v);
        }
        let index: i64 = if pair.master >= 0 {
            let master: Generator = self.generators[pair.master as usize].clone();
            if master.index % q == 0 {
                if let Some(scale) = &master.scale {
                    let g: Element = self.generator_element(pair.master as usize);
                    let base: Element = self.div(&g, scale)?;
                    return self.pow(&base, master.index / q);
                }
            }
            lcm_i(master.index, q)
        } else {
            q
        };
        let e: Element = self.normal(&pair.e)?;
        let n: Element = self.poly_element(e.num.clone());
        let d: Element = self.poly_element(e.den.clone());
        let expression: Basic = if pair.logarithmic {
            crate::core::mul::mul(&[
                self.to_expr(&d)?,
                crate::core::power::pow(&self.to_expr(&e)?, &number(&frac(1, index)))?,
            ])?
        } else {
            crate::core::mul::mul(&[
                self.to_expr(&d)?,
                crate::functions::elementary::exponential::exp(&crate::core::mul::div(
                    &self.to_expr(&pair.z)?,
                    &int(index),
                )?)?,
            ])?
        };
        let name: String = self.new_name();
        let mut generator: Generator = Generator::new(Kind::Root, name, expression, None, true);
        generator.index = index;
        generator.source = i as i64;
        generator.radicand = Some(self.mul(&n, &self.pow(&d, index - 1)?)?);
        generator.scale = Some(d.clone());
        let previous: i64 = pair.master;
        let position: usize = self.add_generator(generator);
        self.pairs[i].master = position as i64;
        let rho: Element = self.div(&self.generator_element(position), &d)?;
        if previous >= 0 {
            let old: Generator = self.generators[previous as usize].clone();
            if let Some(scale) = &old.scale {
                let eliminated: Element = self.mul(scale, &self.pow(&rho, index / old.index)?)?;
                self.generators[previous as usize].eliminated = Some(eliminated);
            }
        }
        self.derivatives.clear();
        self.pow(&rho, index / q)
    }
}

// ---------------------------------------------------------------------
// logarithms

impl Tower {
    /// `log(v)` in the tower (`logarithm`).
    pub fn logarithm(&mut self, v: &Element) -> PolyResult<Element> {
        let v: Element = self.normal(v)?;
        if v.num.is_zero() {
            return Err(not_elementary("the logarithm of zero".to_string()));
        }
        if self.is_number(&v)? {
            return self.number_logarithm(&v);
        }
        let (positive, rest) = self.positive_part(&v)?;
        if positive.is_empty() {
            return self.structured_logarithm(&v);
        }
        let mut total: Element = self.zero();
        for (factor, power) in positive {
            let piece: Element = if self.is_number(&factor)? {
                self.number_logarithm(&factor)?
            } else {
                self.structured_logarithm(&factor)?
            };
            total = self.add(&total, &self.times(&piece, &rat(power))?)?;
        }
        if !self.vanishes(&self.sub(&rest, &self.one())?)? {
            let piece: Element = if self.is_number(&rest)? {
                self.number_logarithm(&rest)?
            } else {
                self.structured_logarithm(&rest)?
            };
            total = self.add(&total, &piece)?;
        }
        self.normal(&total)
    }

    fn factor_list(&self, p: &TP) -> PolyResult<(Coef, Vec<(TP, usize)>)> {
        let p: TP = self.poly(p);
        if self.is_rationals() {
            use crate::polys::domains::QQ;
            let ring = PolyRing::new(QQ, self.ring.names.clone(), MonomialOrder::Lex)?;
            let terms: Terms<Rat> = p.terms.iter().map(|(m, c)| (m.clone(), c[0].clone())).collect();
            let (coefficient, factors) = crate::polys::polytools::factor_all(&ring.from_dict(terms))?;
            let converted: Vec<(TP, usize)> = factors
                .into_iter()
                .map(|(f, k)| {
                    let t: Terms<Coef> = f
                        .terms
                        .iter()
                        .map(|(m, c)| (m.clone(), vec![Rat::from(c.clone())]))
                        .collect();
                    (self.ring.from_dict(t), k)
                })
                .collect();
            return Ok((self.domain.from_rational(&coefficient), converted));
        }
        crate::polys::algebraicfactor::ext_factor_list(&p)
    }

    fn ground_element(&self, c: Coef) -> Element {
        Element {
            num: self.ring.ground(c),
            den: self.ring.one(),
        }
    }

    /// The factors of known sign of `v` and the rest (`_positive_part`).
    fn positive_part(&mut self, v: &Element) -> PolyResult<(Vec<(Element, i64)>, Element)> {
        let roots: Vec<usize> = (0..self.generators.len())
            .filter(|&i| self.generators[i].kind == Kind::Root)
            .collect();
        if roots.iter().any(|&x| v.num.degree_in(x) > 0 || v.den.degree_in(x) > 0) {
            return Ok((vec![], v.clone()));
        }
        let mut pieces: Vec<(Element, i64)> = Vec::new();
        let mut content: Element = self.one();
        for (polynomial, sign) in [(v.num.clone(), 1i64), (v.den.clone(), -1i64)] {
            let (coefficient, factors) = self.factor_list(&polynomial)?;
            content = self.mul(&content, &self.pow(&self.ground_element(coefficient), sign)?)?;
            for (f, multiplicity) in factors {
                pieces.push((self.poly_element(f), sign * multiplicity as i64));
            }
        }
        let content_is_one: bool = self.vanishes(&self.sub(&content, &self.one())?)?;
        if pieces.len() == 1 && pieces[0].1 == 1 && content_is_one {
            return Ok((vec![], v.clone()));
        }
        let mut positive: Vec<(Element, i64)> = Vec::new();
        let mut rest: Element = self.one();
        if self.is_number(&content)? && self.positive_number(&content)? {
            if !content_is_one {
                positive.push((content.clone(), 1));
            }
        } else if self.is_number(&content)? && self.negative_number(&content)? {
            positive.push((self.neg(&content), 1));
            rest = self.neg(&rest);
        } else {
            rest = self.mul(&rest, &content)?;
        }
        for (factor, power) in pieces {
            if self.positive(&factor)? {
                positive.push((factor, power));
            } else if self.negative(&factor)? {
                positive.push((self.neg(&factor), power));
                if power % 2 != 0 {
                    rest = self.neg(&rest);
                }
            } else {
                rest = self.mul(&rest, &self.pow(&factor, power)?)?;
            }
        }
        if positive.len() == 1 && positive[0].1 == 1 && self.vanishes(&self.sub(&rest, &self.one())?)? {
            return Ok((vec![], v.clone()));
        }
        let rest: Element = self.normal(&rest)?;
        Ok((positive, rest))
    }

    fn structured_logarithm(&mut self, v: &Element) -> PolyResult<Element> {
        let v: Element = self.normal(v)?;
        if self.is_number(&v)? {
            return self.number_logarithm(&v);
        }
        if let Some(known) = self.known_logarithm(&v)? {
            return Ok(known);
        }
        let variables: Vec<usize> = self.variable_positions();
        let (moving, _) = self.split_pairs()?;
        if self.is_constant(&v)? {
            if self.positive(&v)? {
                if let Some(found) = self.positive_constant_logarithm(&v)? {
                    return Ok(found);
                }
            }
            return self.new_logarithm(&v, false);
        }
        let found: Option<Vec<Rat>> = if moving.is_empty() {
            None
        } else {
            let mut equations: Vec<(Element, Vec<Element>)> = Vec::new();
            for &k in &variables {
                let dv: Element = self.derivative(&v, k)?;
                let target: Element = self.div(&dv, &v)?;
                let mut basis: Vec<Element> = Vec::new();
                for &i in &moving {
                    let z: Element = self.pairs[i].z.clone();
                    basis.push(self.derivative(&z, k)?);
                }
                equations.push((target, basis));
            }
            self.solve(&equations)?
        };
        let Some(found) = found else {
            return self.new_logarithm(&v, true);
        };
        let used: Vec<(usize, Rat)> = moving
            .iter()
            .zip(found.iter())
            .filter(|(_, r)| **r != Rat::ZERO)
            .map(|(i, r)| (*i, r.clone()))
            .collect();
        let mut all_real: bool = true;
        if self.positive(&v)? {
            for (i, _) in &used {
                if !self.pair_is_real(*i)? {
                    all_real = false;
                    break;
                }
            }
            if all_real {
                let scale: i64 = lcm_all(used.iter().map(|(_, r)| small(&denom(r))));
                let mut m: Element = self.pow(&v, scale)?;
                let mut total: Element = self.zero();
                for (i, r) in &used {
                    let e: Element = self.pairs[*i].e.clone();
                    let z: Element = self.pairs[*i].z.clone();
                    m = self.mul(&m, &self.pow(&e, -small(&numer(&(r * rat(scale)))))?)?;
                    total = self.add(&total, &self.times(&z, r)?)?;
                }
                m = self.normal(&m)?;
                if self.is_constant(&m)? {
                    if let Some(rest) = self.positive_constant_logarithm(&m)? {
                        let t: Element = self.add(&total, &self.times(&rest, &frac(1, scale))?)?;
                        return self.normal(&t);
                    }
                }
            }
        }
        if self.negative(&v)? {
            let minus: Element = self.minus_one_logarithm();
            let inner: Element = self.structured_logarithm(&self.neg(&v))?;
            return self.normal(&self.add(&minus, &inner)?);
        }
        if let Some(pinned) = self.pinned_logarithm(&v, &used)? {
            return Ok(pinned);
        }
        self.new_logarithm(&v, false)
    }

    fn known_logarithm(&mut self, v: &Element) -> PolyResult<Option<Element>> {
        for j in 0..self.pairs.len() {
            let pair: Pair = self.pairs[j].clone();
            if pair.logarithmic && self.vanishes(&self.sub(&pair.e, v)?)? {
                return Ok(Some(pair.z));
            }
        }
        for position in 0..self.generators.len() {
            let g: Generator = self.generators[position].clone();
            if g.kind != Kind::Root || g.eliminated.is_some() || g.scale.is_none() || g.source < 0 {
                continue;
            }
            let source: Pair = self.pairs[g.source as usize].clone();
            if !source.logarithmic {
                continue;
            }
            let rho: Element = self.div(&self.generator_element(position), g.scale.as_ref().expect("scale"))?;
            let mut power: Element = self.one();
            for k in 1..g.index {
                power = self.mul(&power, &rho)?;
                if self.vanishes(&self.sub(&power, v)?)? {
                    return Ok(Some(self.times(&source.z, &frac(k, g.index))?));
                }
                if self.vanishes(&self.sub(&self.mul(&power, v)?, &self.one())?)? {
                    return Ok(Some(self.times(&source.z, &frac(-k, g.index))?));
                }
            }
        }
        Ok(None)
    }

    fn pinned_logarithm(&mut self, v: &Element, used: &[(usize, Rat)]) -> PolyResult<Option<Element>> {
        use crate::core::relational as rl;
        if !self.convex_region()? {
            return Ok(None);
        }
        let mut arguments: Vec<Element> = vec![v.clone()];
        for (i, _) in used {
            if self.pairs[*i].logarithmic {
                arguments.push(self.pairs[*i].e.clone());
            }
        }
        let mut found: std::collections::BTreeSet<Basic> = std::collections::BTreeSet::new();
        for w in &arguments {
            found.extend(self.to_expr(w)?.free_symbols());
        }
        let mut symbols: Vec<Basic> = found.into_iter().collect();
        symbols.sort_by_key(|s| s.to_string());
        for s in &symbols {
            let var: Element = self.variable(s);
            if !self.real(&var)? {
                return Ok(None);
            }
        }
        let mut meanings: Vec<Basic> = Vec::new();
        let mut facts: Vec<Basic> = Vec::new();
        let mut stand_ins: Vec<Basic> = Vec::new();
        for gi in 0..self.generators.len() {
            let g: Generator = self.generators[gi].clone();
            let mut meaning: Basic = g.expression.clone();
            if g.kind == Kind::Root && g.eliminated.is_none() && g.scale.is_some() && g.source >= 0 {
                let source: Pair = self.pairs[g.source as usize].clone();
                if source.logarithmic && self.positive(&source.e)? {
                    let rho: Basic = crate::core::symbol::Dummy::new("rho", &[(Predicate::positive, true)]).0;
                    stand_ins.push(rho.clone());
                    meaning = crate::core::mul::mul(&[self.to_expr(g.scale.as_ref().expect("scale"))?, rho.clone()])?;
                    facts.push(rel_gt(&rho)?);
                    facts.push(rl::eq(
                        &crate::core::power::pow(&rho, &int(g.index))?,
                        &self.to_expr(&source.e)?,
                    )?);
                }
            }
            meanings.push(meaning);
        }
        let mut reals: HashMap<Basic, Basic> = HashMap::new();
        let mut back: HashMap<Basic, Basic> = HashMap::new();
        for s in &symbols {
            let (name, mut given): (String, Vec<(Predicate, bool)>) = match s.head() {
                Head::Symbol(d) | Head::Dummy(d, _) => {
                    (d.name.clone(), d.assumptions.iter().map(|(p, v)| (*p, *v)).collect())
                }
                _ => (s.to_string(), vec![]),
            };
            given.retain(|(p, _)| *p != Predicate::real);
            given.push((Predicate::real, true));
            let d: Basic = crate::core::symbol::Dummy::new(&name, &given).0;
            reals.insert(s.clone(), d.clone());
            back.insert(d, s.clone());
        }
        let parts = |value: &Basic| -> PolyResult<Option<(Basic, Basic)>> {
            let replaced: Basic = value.xreplace(&reals);
            let Some((re, im)) = crate::functions::elementary::complexes::as_real_imag(&replaced, true, None)? else {
                return Ok(None);
            };
            Ok(Some((re.xreplace(&back), im.xreplace(&back))))
        };
        let mut gens: Vec<Basic> = symbols.clone();
        gens.extend(stand_ins.iter().cloned());
        let rational_in = |e: &Basic| {
            crate::core::expr::polynomial_like(e, &gens, crate::core::expr::PolynomialKind::Rational) == Some(true)
        };
        let mut checked: Vec<Element> = arguments.clone();
        for (i, _) in used {
            checked.push(self.pairs[*i].z.clone());
        }
        for w in &checked {
            let wn: Element = self.normal(w)?;
            let denominator: Basic = self.polynomial_expr(&wn.den, &meanings)?;
            if !denominator.free_symbols().is_empty() {
                if !rational_in(&denominator) {
                    return Ok(None);
                }
                let Some((re, im)) = parts(&denominator)? else {
                    return Ok(None);
                };
                let q: Basic = crate::core::logic::or_(&[rl::ne(&re, &int(0))?, rl::ne(&im, &int(0))?]);
                if self.ask_(&q, &facts)? != Some(true) {
                    return Ok(None);
                }
            }
        }
        for w in &arguments {
            let w: Element = self.normal(w)?;
            let value: Basic = crate::core::mul::div(
                &self.polynomial_expr(&w.num, &meanings)?,
                &self.polynomial_expr(&w.den, &meanings)?,
            )?;
            if !rational_in(&value) {
                return Ok(None);
            }
            let Some((re, im)) = parts(&value)? else {
                return Ok(None);
            };
            if im == int(0) {
                let e: Basic = self.to_expr(&w)?;
                if self.sign(&e)? == 0 {
                    return Ok(None);
                }
            } else {
                let q: Basic = crate::core::logic::or_(&[rl::ne(&im, &int(0))?, rel_gt(&re)?]);
                if self.ask_(&q, &facts)? != Some(true) {
                    return Ok(None);
                }
            }
        }
        let scale: i64 = lcm_all(used.iter().map(|(_, r)| small(&denom(r))));
        let mut m: Element = self.pow(v, scale)?;
        let mut total: Element = self.zero();
        for (i, r) in used {
            let e: Element = self.pairs[*i].e.clone();
            let z: Element = self.pairs[*i].z.clone();
            m = self.mul(&m, &self.pow(&e, -small(&numer(&(r * rat(scale)))))?)?;
            total = self.add(&total, &self.times(&z, r)?)?;
        }
        m = self.normal(&m)?;
        if !self.is_number(&m)? {
            return Ok(None);
        }
        let principal: Element = self.number_logarithm(&m)?;
        let mut values: Option<Vec<(Basic, Basic)>> = Some(vec![]);
        if !symbols.is_empty() {
            let mut rng = crate::random::PyRandom::from_string(&self.to_expr(v)?.to_string());
            values = crate::integrals::conditions::sample_values(&symbols, &self.assumptions, &mut rng)?;
        }
        let Some(values) = values else { return Ok(None) };
        let point: HashMap<Basic, Basic> = values.into_iter().collect();
        use crate::core::add::add;
        use crate::core::mul::{mul, neg};
        let gap: Basic = add(&[
            mul(&[
                int(scale),
                add(&[
                    crate::functions::elementary::exponential::log(&self.to_expr(v)?)?,
                    neg(&self.to_expr(&total)?)?,
                ])?,
            ])?,
            neg(&self.to_expr(&principal)?)?,
        ])?;
        let measured: Option<(f64, f64)> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            Ok(evalf_complex(&gap.xreplace(&point), 40))
        })?
        .flatten();
        let Some((re, im)) = measured else { return Ok(None) };
        let turns: f64 = im / (2.0 * std::f64::consts::PI);
        if re.abs() > 1e-12 || (turns - turns.round()).abs() > 1e-12 {
            return Ok(None);
        }
        let mut constant: Element = principal;
        let k: i64 = turns.round() as i64;
        if k != 0 {
            let minus: Element = self.minus_one_logarithm();
            constant = self.add(&constant, &self.times(&minus, &rat(2 * k))?)?;
        }
        let t: Element = self.add(&total, &self.times(&constant, &frac(1, scale))?)?;
        Ok(Some(self.normal(&t)?))
    }

    fn convex_region(&self) -> PolyResult<bool> {
        let mut parts: Vec<Basic> = Vec::new();
        let mut pending: Vec<Basic> = self.assumptions.clone();
        while let Some(part) = pending.pop() {
            if matches!(part.head(), Head::And) {
                pending.extend(part.args().iter().cloned());
            } else {
                parts.push(part);
            }
        }
        for part in parts {
            if matches!(part.head(), Head::BooleanTrue) {
                continue;
            }
            if !matches!(
                part.head(),
                Head::Relational(
                    crate::core::basic::RelOp::Gt
                        | crate::core::basic::RelOp::Ge
                        | crate::core::basic::RelOp::Lt
                        | crate::core::basic::RelOp::Le
                )
            ) {
                return Ok(false);
            }
            let difference: Basic = crate::core::add::sub(&part.args()[0], &part.args()[1])?;
            let mut symbols: Vec<Basic> = difference.free_symbols().into_iter().collect();
            symbols.sort_by_key(|s| s.to_string());
            if !symbols.is_empty() {
                match crate::polys::polytools::Poly::new(&difference, &symbols) {
                    Ok(p) => {
                        if p.total_degree() > 1 {
                            return Ok(false);
                        }
                    }
                    Err(PolyError::Polys(..) | PolyError::Value(_)) => return Ok(false),
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(true)
    }

    fn pair_is_real(&mut self, i: usize) -> PolyResult<bool> {
        let pair: Pair = self.pairs[i].clone();
        if pair.logarithmic {
            self.positive(&pair.e)
        } else {
            self.real(&pair.z)
        }
    }

    fn new_logarithm(&mut self, v: &Element, certified: bool) -> PolyResult<Element> {
        let certified: bool = certified && self.sound();
        let expression: Basic = crate::functions::elementary::exponential::log(&self.to_expr(v)?)?;
        let name: String = self.new_name();
        let position: usize = self.add_generator(Generator::new(
            Kind::Logarithm,
            name,
            expression,
            Some(v.clone()),
            certified,
        ));
        let l: Element = self.generator_element(position);
        self.pairs.push(Pair {
            z: l.clone(),
            e: v.clone(),
            generator: position,
            logarithmic: true,
            master: -1,
        });
        Ok(l)
    }

    fn minus_one_logarithm(&mut self) -> Element {
        if self.minus_one < 0 {
            let v: Element = self.constant(&rat(-1));
            let name: String = self.new_name();
            let expression: Basic =
                crate::core::mul::mul(&[Basic::atom(Head::ImaginaryUnit), Basic::pi()]).expect("I*pi");
            let position: usize =
                self.add_generator(Generator::new(Kind::Logarithm, name, expression, Some(v.clone()), true));
            self.pairs.push(Pair {
                z: self.generator_element(position),
                e: v,
                generator: position,
                logarithmic: true,
                master: -1,
            });
            self.minus_one = position as i64;
        }
        self.generator_element(self.minus_one as usize)
    }

    fn prime_logarithm(&mut self, p: &Int) -> PolyResult<Element> {
        if !self.prime_logs.contains_key(p) {
            let v: Element = self.constant(&Rat::from(p.clone()));
            let name: String = self.new_name();
            let expression: Basic =
                crate::functions::elementary::exponential::log(&Basic::from_rational(Rat::from(p.clone())))?;
            let position: usize =
                self.add_generator(Generator::new(Kind::Logarithm, name, expression, Some(v.clone()), true));
            self.pairs.push(Pair {
                z: self.generator_element(position),
                e: v,
                generator: position,
                logarithmic: true,
                master: -1,
            });
            self.prime_logs.insert(p.clone(), position);
        }
        Ok(self.generator_element(self.prime_logs[p]))
    }

    fn number_logarithm(&mut self, a: &Element) -> PolyResult<Element> {
        if let Some(q) = self.rational_of(a)? {
            if crate::numbers::rat_is_negative(&q) {
                let minus: Element = self.minus_one_logarithm();
                let rest: Element = self.number_logarithm(&self.constant(&-q))?;
                return self.normal(&self.add(&minus, &rest)?);
            }
            let mut total: Element = self.zero();
            for (prime, exponent) in crate::ntheory::factor_::factorint(&numer(&q), None) {
                let l: Element = self.prime_logarithm(&prime)?;
                total = self.add(&total, &self.times(&l, &rat(exponent as i64))?)?;
            }
            for (prime, exponent) in crate::ntheory::factor_::factorint(&denom(&q), None) {
                let l: Element = self.prime_logarithm(&prime)?;
                total = self.sub(&total, &self.times(&l, &rat(exponent as i64))?)?;
            }
            return self.normal(&total);
        }
        let a: Element = self.normal(a)?;
        let nb: Algebraic = self.algebraic_value(&a)?;
        let mut power: Element = self.one();
        for k in 1..=(2 * nb.degree() as i64) {
            power = self.normal(&self.mul(&power, &a)?)?;
            if let Some(r) = self.rational_of(&power)? {
                let negative: bool = crate::numbers::rat_is_negative(&r);
                let absr: Rat = if negative { -r.clone() } else { r.clone() };
                let real: Element = self.number_logarithm(&self.constant(&absr))?;
                let (re, im) = nb.to_complex()?;
                let angle: f64 = k as f64 * im.atan2(re) - if negative { std::f64::consts::PI } else { 0.0 };
                let turns: i64 = python_round(angle / (2.0 * std::f64::consts::PI));
                let extra: i64 = 2 * turns + if negative { 1 } else { 0 };
                let total: Element = if extra != 0 {
                    let minus: Element = self.minus_one_logarithm();
                    self.add(&real, &self.times(&minus, &rat(extra))?)?
                } else {
                    real
                };
                return self.normal(&self.times(&total, &frac(1, k))?);
            }
        }
        if let Some(related) = self.related_logarithm(&a, &nb)? {
            return Ok(related);
        }
        let mut certified: bool = moduli_differ(&nb.minpoly());
        if certified {
            for g in self.generators.clone() {
                if g.kind == Kind::Logarithm {
                    if let Some(arg) = &g.argument {
                        if self.is_number(arg)? && self.rational_of(arg)?.is_none() {
                            certified = false;
                            break;
                        }
                    }
                }
            }
        }
        self.new_logarithm(&a, certified)
    }

    fn related_logarithm(&mut self, a: &Element, number_a: &Algebraic) -> PolyResult<Option<Element>> {
        let mut candidates: Vec<(usize, Algebraic)> = Vec::new();
        let minus_pair: i64 = if self.minus_one >= 0 {
            self.pair_of(self.minus_one as usize)
        } else {
            -1
        };
        for j in 0..self.pairs.len() {
            let pair: Pair = self.pairs[j].clone();
            if pair.logarithmic && j as i64 != minus_pair && self.is_number(&pair.e)? {
                candidates.push((j, self.algebraic_value(&pair.e)?));
            }
        }
        let minimal: Vec<Int> = number_a.minpoly();
        let norm: Rat = Rat::from(minimal[minimal.len() - 1].clone()) / Rat::from(minimal[0].clone());
        let mut primes: std::collections::BTreeSet<Int> = std::collections::BTreeSet::new();
        let absn: Int = {
            let n: Int = numer(&norm);
            if n < Int::ZERO {
                -n
            } else {
                n
            }
        };
        primes.extend(crate::ntheory::factor_::factorint(&absn, None).into_keys());
        primes.extend(crate::ntheory::factor_::factorint(&denom(&norm), None).into_keys());
        for prime in primes {
            if prime > Int::ONE && !self.prime_logs.contains_key(&prime) && candidates.len() < 12 {
                self.prime_logarithm(&prime)?;
                let j: i64 = self.pair_of(self.prime_logs[&prime]);
                candidates.push((j as usize, Algebraic::from_rational(&Rat::from(prime.clone()))));
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        // the PSLQ of mpmath at 80 digits
        let prec: usize = 269;
        use crate::core::evalf::{real_rat, Real};
        let t: Real = real_rat(&rat(3), prec).sqrt() + real_rat(&frac(1, 7), prec);
        let coordinate = |n: &Algebraic| -> PolyResult<Real> {
            let z = approximation(n, 90, prec)?;
            let l = z.ln();
            let (re, im) = crate::core::evalf::parts(&l);
            Ok(re + &t * im)
        };
        let mut vector: Vec<Real> = vec![coordinate(number_a)?];
        for (_, n) in &candidates {
            vector.push(coordinate(n)?);
        }
        vector.push(Real::pi(prec) * real_rat(&rat(2), prec) * &t);
        let Some(relation) = crate::identification::pslq(&vector, prec, 10_000, 100_000) else {
            return Ok(None);
        };
        if relation[0] == 0 {
            return Ok(None);
        }
        let n0: i64 = relation[0];
        let mut product: Element = self.pow(a, n0)?;
        for ((j, _), n) in candidates.iter().zip(relation[1..relation.len() - 1].iter()) {
            let e: Element = self.pairs[*j].e.clone();
            product = self.mul(&product, &self.pow(&e, *n)?)?;
        }
        if !self.vanishes(&self.sub(&product, &self.one())?)? {
            return Ok(None);
        }
        let mut total: Element = self.zero();
        for ((j, _), n) in candidates.iter().zip(relation[1..relation.len() - 1].iter()) {
            let z: Element = self.pairs[*j].z.clone();
            total = self.sub(&total, &self.times(&z, &rat(*n))?)?;
        }
        let measured_prec: usize = crate::numeric::dps_to_prec(40);
        let za = approximation(number_a, 45, measured_prec)?.ln();
        let Some(tv) = crate::core::evalf::numeric_value(&self.to_expr(&total)?, measured_prec) else {
            return Ok(None);
        };
        let (zr, zi) = crate::core::evalf::parts(&za);
        let (tr, ti) = crate::core::evalf::parts(&tv);
        let re: f64 = zr.to_f64().value() * n0 as f64 - tr.to_f64().value();
        let im: f64 = zi.to_f64().value() * n0 as f64 - ti.to_f64().value();
        let turns: f64 = im / (2.0 * std::f64::consts::PI);
        if re.abs() > 1e-12 || (turns - turns.round()).abs() > 1e-9 {
            return Ok(None);
        }
        let k: i64 = python_round(turns);
        if k != 0 {
            let minus: Element = self.minus_one_logarithm();
            total = self.add(&total, &self.times(&minus, &rat(2 * k))?)?;
        }
        Ok(Some(self.normal(&self.times(&total, &frac(1, n0))?)?))
    }

    fn pair_of(&self, generator: usize) -> i64 {
        self.pairs
            .iter()
            .position(|p| p.generator == generator)
            .map(|j| j as i64)
            .unwrap_or(-1)
    }

    fn is_number(&self, a: &Element) -> PolyResult<bool> {
        let a: Element = self.normal(a)?;
        Ok(a.num.is_ground() && a.den.is_ground())
    }

    fn rational_of(&self, a: &Element) -> PolyResult<Option<Rat>> {
        let a: Element = self.normal(a)?;
        if !a.num.is_ground() || !a.den.is_ground() {
            return Ok(None);
        }
        let c: Coef = if a.num.is_zero() {
            vec![]
        } else {
            self.domain.mul(&a.num.LC(), &self.domain.invert(&a.den.LC())?)
        };
        if c.len() > 1 {
            return Ok(None);
        }
        Ok(Some(c.first().cloned().unwrap_or(Rat::ZERO)))
    }

    fn positive_number(&self, a: &Element) -> PolyResult<bool> {
        let n: Algebraic = self.algebraic_value(a)?;
        Ok(n.is_real() && n.sign()? > 0)
    }

    fn negative_number(&self, a: &Element) -> PolyResult<bool> {
        let n: Algebraic = self.algebraic_value(a)?;
        Ok(n.is_real() && n.sign()? < 0)
    }

    fn algebraic_value(&self, a: &Element) -> PolyResult<Algebraic> {
        let a: Element = self.normal(a)?;
        if !a.num.is_ground() || !a.den.is_ground() {
            return Err(PolyError::Value("not a number of the field".to_string()));
        }
        let c: Coef = if a.num.is_zero() {
            vec![]
        } else {
            self.domain.mul(&a.num.LC(), &self.domain.invert(&a.den.LC())?)
        };
        match &self.theta {
            Some(theta) if c.len() > 1 => theta.polynomial_value(&c),
            _ => Ok(Algebraic::from_rational(&c.first().cloned().unwrap_or(Rat::ZERO))),
        }
    }

    fn positive_constant_logarithm(&mut self, m: &Element) -> PolyResult<Option<Element>> {
        if self.is_number(m)? {
            return Ok(if self.positive_number(m)? {
                Some(self.number_logarithm(m)?)
            } else {
                None
            });
        }
        let (_, still) = self.split_pairs()?;
        let independent: Vec<usize> = still
            .iter()
            .map(|&j| self.pairs[j].generator)
            .filter(|&g| matches!(self.generators[g].kind, Kind::Exponential | Kind::Logarithm))
            .collect();
        if independent.is_empty() {
            return Ok(None);
        }
        let mut equations: Vec<(Element, Vec<Element>)> = Vec::new();
        for &g in &independent {
            let dm: Element = self.formal_derivative(m, g)?;
            let target: Element = self.div(&dm, m)?;
            let mut basis: Vec<Element> = Vec::new();
            for &j in &still {
                let e: Element = self.pairs[j].e.clone();
                let de: Element = self.formal_derivative(&e, g)?;
                basis.push(self.div(&de, &e)?);
            }
            equations.push((target, basis));
        }
        let Some(found) = self.solve(&equations)? else {
            return Ok(None);
        };
        let used: Vec<(usize, Rat)> = still
            .iter()
            .zip(found.iter())
            .filter(|(_, s)| **s != Rat::ZERO)
            .map(|(j, s)| (*j, s.clone()))
            .collect();
        for (j, _) in &used {
            if !self.pair_is_real(*j)? {
                return Ok(None);
            }
        }
        let scale: i64 = lcm_all(used.iter().map(|(_, s)| small(&denom(s))));
        let mut rest: Element = self.pow(m, scale)?;
        let mut total: Element = self.zero();
        for (j, s) in &used {
            let e: Element = self.pairs[*j].e.clone();
            let z: Element = self.pairs[*j].z.clone();
            rest = self.mul(&rest, &self.pow(&e, -small(&numer(&(s * rat(scale)))))?)?;
            total = self.add(&total, &self.times(&z, s)?)?;
        }
        rest = self.normal(&rest)?;
        if !self.is_number(&rest)? || !self.positive_number(&rest)? {
            return Ok(None);
        }
        let l: Element = self.number_logarithm(&rest)?;
        let t: Element = self.add(&total, &self.times(&l, &frac(1, scale))?)?;
        Ok(Some(self.normal(&t)?))
    }
}

/// Python's `round` of a float to an integer (ties to even).
fn python_round(x: f64) -> i64 {
    let r: f64 = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

/// `complex(e.evalf(digits))`.
fn evalf_complex(e: &Basic, digits: usize) -> Option<(f64, f64)> {
    let v = crate::core::evalf::numeric_value(e, crate::numeric::dps_to_prec(digits))?;
    let (a, b) = crate::core::evalf::parts(&v);
    Some((a.to_f64().value(), b.to_f64().value()))
}

/// The centre of an enclosure of the number of width `10**-digits`, at
/// `prec` bits (`_approximation`).
fn approximation(n: &Algebraic, digits: usize, prec: usize) -> PolyResult<crate::core::evalf::Complex> {
    use crate::core::evalf::{complex, real_rat};
    let eps: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(10u8).pow(digits));
    let (a, b, c, d) = n.enclosure(&eps)?;
    let two: Rat = rat(2);
    Ok(complex(
        real_rat(&((a + b) / &two), prec),
        real_rat(&((c + d) / &two), prec),
    ))
}

/// Whether the roots of the minimal polynomial have different moduli
/// (`_moduli_differ`, the roots of mpmath's `polyroots` at 30 digits).
fn moduli_differ(minimal: &[Int]) -> bool {
    let Some(roots) = crate::identification::polyroots(minimal, crate::numeric::dps_to_prec(30) + 60, 200) else {
        return false;
    };
    let mut moduli: Vec<f64> = roots
        .iter()
        .map(|z| {
            let (a, b) = crate::core::evalf::parts(z);
            (&a * &a + &b * &b).sqrt().to_f64().value()
        })
        .collect();
    moduli.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match (moduli.first(), moduli.last()) {
        (Some(lo), Some(hi)) => hi - lo > 1e-12 * (1.0 + hi),
        _ => false,
    }
}

// ---------------------------------------------------------------------
// expressions in and out, certification

const EXPONENTIAL_FORMS: &[Function] = &[
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];
const LOGARITHMIC_FORMS: &[Function] = &[
    Function::Asin,
    Function::Acos,
    Function::Atan,
    Function::Acot,
    Function::Asinh,
    Function::Acosh,
    Function::Atanh,
    Function::Acoth,
];

fn has_any(e: &Basic, forms: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(f) if forms.contains(f)))
}

impl Tower {
    /// The generator of a variable (`variable`).
    pub fn variable(&mut self, symbol: &Basic) -> Element {
        if let Some((_, p)) = self.variables.iter().find(|(s, _)| s == symbol) {
            return self.generator_element(*p);
        }
        let name: String = self.new_name();
        let position: usize = self.add_generator(Generator::new(Kind::Variable, name, symbol.clone(), None, true));
        self.variables.push((symbol.clone(), position));
        self.derivatives.clear();
        self.generator_element(position)
    }

    /// An expression as an element of the tower (`element`);
    /// `NotElementary` for what it cannot hold.
    pub fn element(&mut self, expression: &Basic) -> PolyResult<Element> {
        if let Some(v) = self.cache.get(expression) {
            return Ok(v.clone());
        }
        let value: Element = self.element_(expression)?;
        self.cache.insert(expression.clone(), value.clone());
        Ok(value)
    }

    fn element_(&mut self, e: &Basic) -> PolyResult<Element> {
        if is_rational(e) {
            return Ok(self.constant(&rational_of_expr(e)?));
        }
        if e.is_symbol() {
            return Ok(self.variable(e));
        }
        if matches!(e.head(), Head::Pi) {
            let minus: Element = self.minus_one_logarithm();
            let minus_i: Element =
                self.number(&crate::core::mul::mul(&[int(-1), Basic::atom(Head::ImaginaryUnit)])?)?;
            return self.normal(&self.mul(&minus, &minus_i)?);
        }
        if matches!(e.head(), Head::Exp1) {
            let one: Element = self.one();
            return self.exponential(&one);
        }
        let algebraic_number: bool = crate::core::assumptions::is_number(e)
            && is(e, Predicate::algebraic) == Some(true)
            && !e
                .preorder()
                .iter()
                .any(|n| matches!(n.head(), Head::Pi | Head::Function(Function::Exp | Function::Log)));
        if matches!(e.head(), Head::ImaginaryUnit) || algebraic_number {
            return self.number(e);
        }
        if e.is_add() {
            let mut total: Element = self.zero();
            for t in e.args() {
                let v: Element = self.element(t)?;
                total = self.add(&total, &v)?;
            }
            return Ok(total);
        }
        if e.is_mul() {
            let mut product: Element = self.one();
            for f in e.args() {
                let v: Element = self.element(f)?;
                product = self.mul(&product, &v)?;
            }
            return Ok(product);
        }
        if is_function(e, Function::Exp) {
            let v: Element = self.element(&e.args()[0])?;
            return self.exponential(&v);
        }
        if is_function(e, Function::Log) && e.args().len() == 1 {
            let v: Element = self.element(&e.args()[0])?;
            return self.logarithm(&v);
        }
        if e.is_pow() {
            let (base, exponent) = (&e.args()[0], &e.args()[1]);
            if exponent.is_integer_value() {
                let b: Element = self.element(base)?;
                return self.pow(&b, small(&numer(&rational_of_expr(exponent)?)));
            }
            if matches!(base.head(), Head::Exp1) {
                let v: Element = self.element(exponent)?;
                return self.exponential(&v);
            }
            let b: Element = self.element(base)?;
            if self.vanishes(&b)? {
                return Err(not_elementary(format!("a power of zero: {}", e)));
            }
            let x: Element = self.element(exponent)?;
            let l: Element = self.logarithm(&b)?;
            return self.exponential(&self.mul(&x, &l)?);
        }
        if let Head::Function(f) = e.head() {
            let hints = crate::core::rewriting::RewriteHints::default();
            if EXPONENTIAL_FORMS.contains(f) {
                let rewritten: Basic = crate::core::rewriting::rewrite(e, &[], "exp", &hints)?;
                if !has_any(&rewritten, EXPONENTIAL_FORMS) {
                    return self.element(&rewritten);
                }
            }
            if LOGARITHMIC_FORMS.contains(f) {
                let rewritten: Basic = crate::core::rewriting::rewrite(e, &[], "log", &hints)?;
                if !has_any(&rewritten, LOGARITHMIC_FORMS) {
                    return self.element(&rewritten);
                }
            }
        }
        Err(not_elementary(format!(
            "not an elementary expression this module knows: {}",
            e
        )))
    }

    fn polynomial_expr(&self, p: &TP, meanings: &[Basic]) -> R {
        let p: TP = self.poly(p);
        let mut terms: Vec<Basic> = Vec::new();
        for (monomial, coefficient) in &p.terms {
            let mut factors: Vec<Basic> = vec![self.coefficient_expr(coefficient)?];
            for (meaning, e) in meanings.iter().zip(monomial.iter()) {
                if *e > 0 {
                    factors.push(crate::core::power::pow(meaning, &int(*e as i64))?);
                }
            }
            terms.push(crate::core::mul::mul(&factors)?);
        }
        if terms.is_empty() {
            Ok(int(0))
        } else {
            crate::core::add::add(&terms)
        }
    }

    /// The element as an expression (`to_expr`).
    pub fn to_expr(&self, a: &Element) -> R {
        let a: Element = self.normal(a)?;
        let meanings: Vec<Basic> = self.generators.iter().map(|g| g.expression.clone()).collect();
        crate::core::mul::div(
            &self.polynomial_expr(&a.num, &meanings)?,
            &self.polynomial_expr(&a.den, &meanings)?,
        )
    }

    /// Kummer theory: the roots have the full degree (`_roots_certified`).
    fn roots_certified(&mut self) -> PolyResult<bool> {
        if let Some(k) = self.kummer {
            return Ok(k);
        }
        let roots: Vec<Generator> = self
            .generators
            .iter()
            .filter(|g| g.kind == Kind::Root && g.eliminated.is_none() && g.radicand.is_some())
            .cloned()
            .collect();
        self.kummer = Some(true);
        if roots.is_empty() {
            return Ok(true);
        }
        let root_positions: Vec<usize> = (0..self.generators.len())
            .filter(|&i| self.generators[i].kind == Kind::Root)
            .collect();
        let mut factors: Vec<TP> = Vec::new();
        let mut vectors: Vec<BTreeMap<usize, i64>> = Vec::new();
        for g in &roots {
            let radicand: TP = self.normal(g.radicand.as_ref().expect("radicand"))?.num;
            if root_positions.iter().any(|&x| radicand.degree_in(x) > 0) {
                self.kummer = Some(false);
                return Ok(false);
            }
            let mut vector: BTreeMap<usize, i64> = BTreeMap::new();
            for (factor, multiplicity) in self.factor_list(&radicand)?.1 {
                let monic: TP = factor.monic()?;
                let index: usize = match factors.iter().position(|f| *f == monic) {
                    Some(i) => i,
                    None => {
                        factors.push(monic);
                        factors.len() - 1
                    }
                };
                vector.insert(index, multiplicity as i64);
            }
            vectors.push(vector);
        }
        if factors.is_empty() {
            self.kummer = Some(false);
            return Ok(false);
        }
        let n: i64 = lcm_all(roots.iter().map(|g| g.index));
        let width: usize = factors.len();
        let mut rows: Vec<Vec<Int>> = roots
            .iter()
            .zip(vectors.iter())
            .map(|(g, vector)| {
                (0..width)
                    .map(|j| Int::from((n / g.index) * vector.get(&j).copied().unwrap_or(0)))
                    .collect()
            })
            .collect();
        for i in 0..width {
            rows.push(
                (0..width)
                    .map(|j| if i == j { Int::from(n) } else { Int::ZERO })
                    .collect(),
            );
        }
        let invariants: Vec<Int> = crate::polys::lattice::smith_normal_form(&rows)?;
        let mut index: Int = Int::ONE;
        for k in 0..width {
            index *= invariants.get(k).cloned().unwrap_or(Int::ZERO);
        }
        let order: Int = if index != Int::ZERO {
            Int::from(n).pow(width) / &index
        } else {
            Int::ZERO
        };
        let mut expected: Int = Int::ONE;
        for g in &roots {
            expected *= Int::from(g.index);
        }
        let k: bool = order == expected;
        self.kummer = Some(k);
        Ok(k)
    }

    /// Whether every generator of `a` is certified (`certifies`).
    pub fn certifies(&mut self, a: &Element) -> PolyResult<bool> {
        let a: Element = self.normal(a)?;
        for position in 0..self.generators.len() {
            if !self.generators[position].certified && (a.num.degree_in(position) > 0 || a.den.degree_in(position) > 0)
            {
                return Ok(false);
            }
        }
        self.roots_certified()
    }

    /// Whether the whole tower is certified (`certified`).
    pub fn certified(&mut self) -> PolyResult<bool> {
        Ok(self.generators.iter().all(|g| g.certified) && self.roots_certified()?)
    }
}

/// The canonical form of an expression (`canonical_form`).
pub fn canonical_form(expression: &Basic, assumptions: &[Basic]) -> R {
    let mut tower: Tower = Tower::new(assumptions);
    let value: Element = tower.element(expression)?;
    tower.to_expr(&value)
}

/// Whether an elementary expression vanishes identically on the region of
/// the assumptions (`is_zero`): `None` when it is not decided.
pub fn is_zero(expression: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    let mut tower: Tower = Tower::new(assumptions);
    let found: Option<Option<bool>> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        let value: Element = match tower.element(expression) {
            Ok(v) => v,
            Err(e) if is_not_elementary(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        if tower.vanishes(&value)? {
            return Ok(Some(true));
        }
        Ok(if tower.certifies(&value)? { Some(false) } else { None })
    })?;
    if let Some(Some(v)) = found {
        return Ok(Some(v));
    }
    Ok(if witness(expression, assumptions)? {
        Some(false)
    } else {
        None
    })
}

/// Whether two expressions are equal (`equal`).
pub fn equal(a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    is_zero(&crate::core::add::sub(a, b)?, assumptions)
}

fn witness(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if !crate::settings::settings().numerical_checks {
        return Ok(false);
    }
    let mut symbols: Vec<Basic> = e.free_symbols().into_iter().collect();
    symbols.sort_by_key(|s| s.to_string());
    let mut mentioned: std::collections::BTreeSet<Basic> = std::collections::BTreeSet::new();
    for f in assumptions {
        mentioned.extend(f.free_symbols());
    }
    let plain = |s: &Basic| match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.assumptions.keys().all(|p| *p == Predicate::commutative),
        _ => false,
    };
    let unconstrained: Vec<Basic> = symbols
        .iter()
        .filter(|s| !mentioned.contains(*s) && plain(s))
        .cloned()
        .collect();
    let constrained: Vec<Basic> = symbols
        .iter()
        .filter(|s| !unconstrained.contains(*s))
        .cloned()
        .collect();
    let mut rng = crate::random::PyRandom::from_string(&e.to_string());
    for attempt_number in 0..6 {
        let mut values: Vec<(Basic, Basic)> = if constrained.is_empty() {
            vec![]
        } else {
            match crate::integrals::conditions::sample_values(&constrained, assumptions, &mut rng)? {
                Some(v) => v,
                None => return Ok(false),
            }
        };
        for s in &unconstrained {
            let real: Basic = number(&frac(rng.randint(-300, 300), 100));
            let value: Basic = if attempt_number >= 3 {
                let imaginary: Basic = number(&frac(rng.randint(-1000, 1000), 100));
                crate::core::add::add(&[
                    real,
                    crate::core::mul::mul(&[Basic::atom(Head::ImaginaryUnit), imaginary])?,
                ])?
            } else {
                real
            };
            values.push((s.clone(), value));
        }
        for s in &constrained {
            let current: Basic = values
                .iter()
                .find(|(t, _)| t == s)
                .map(|(_, v)| v.clone())
                .expect("a value");
            if is(s, Predicate::nonnegative).is_none()
                && is(s, Predicate::nonpositive).is_none()
                && is(s, Predicate::integer).is_none()
                && is(&current, Predicate::extended_real) == Some(true)
                && rng.random() < 0.5
            {
                let mut flipped: Vec<(Basic, Basic)> = values.clone();
                let negated: Basic = crate::core::mul::neg(&current)?;
                for entry in flipped.iter_mut() {
                    if entry.0 == *s {
                        entry.1 = negated.clone();
                    }
                }
                let substitution: HashMap<Basic, Basic> = flipped.iter().cloned().collect();
                let mut all_true: bool = true;
                for f in assumptions {
                    if !matches!(
                        crate::core::function::xreplace_evaluated(f, &substitution)?.head(),
                        Head::BooleanTrue
                    ) {
                        all_true = false;
                        break;
                    }
                }
                if all_true {
                    values = flipped;
                }
            }
        }
        let point: Vec<(Basic, Basic)> = values;
        let found: Option<(f64, f64)> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            Ok(crate::numeric::reliable_complex(e, 40, &point))
        })?
        .flatten();
        let Some((re, im)) = found else { continue };
        if re.hypot(im) > 1e-25 {
            return Ok(true);
        }
    }
    Ok(false)
}
