//! Port of `sympy_extras/solvers/recurr.py`: linear recurrences with
//! polynomial coefficients, their polynomial, hypergeometric (Petkovšek)
//! and d'Alembertian (Abramov–Petkovšek) solutions, the Gosper–Petkovšek
//! form and the general solution of an equation written with `y(n + i)`.
//!
//! The coefficients are dense polynomials in `n` over a field: `QQ`, the
//! rational functions `QQ(a, b, ...)` of the parameters, a number field
//! `QQ(alpha)`, or a simple algebraic extension of one of these. Python's
//! generic `_Field[T]`, whose element type nests one more list with every
//! adjoined root, is the one type [`Field`] here, whose elements
//! [`Elem`] carry their variant: the depth of the extensions is only known
//! at run time.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::error::{PolyError, PolyResult, NOT_PORTED_PREFIX};
use crate::numbers::{Int, Rat};
use crate::polys::densearith::{
    dup_add, dup_mul, dup_mul_ground, dup_neg, dup_pow, dup_quo, dup_quo_ground, dup_rem, dup_sub,
};
use crate::polys::densebasic::{dup_LC, dup_degree, dup_from_dict, dup_one, dup_strip};
use crate::polys::densetools::{dup_monic, dup_shift};
use crate::polys::domains::algebraicfield::{AlgebraicElement, AlgebraicField};
use crate::polys::domains::domain::{Domain, SparseRow};
use crate::polys::domains::rationalfield::{RationalField, QQ};
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;
use crate::polys::euclidtools::{dup_gcd, dup_invert, dup_lcm};
use crate::polys::polytools::Poly;

type R = PolyResult<Basic>;

/// A solution with the first index from which it is defined (`_Solution`).
pub type Solution = (Basic, i64);

/// The names of the summation indices, by depth of the nested sums.
const INDEX_NAMES: [&str; 5] = ["k", "j", "i", "l", "m"];

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn doit(e: &Basic) -> R {
    crate::core::function::doit(e)
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}
/// The variable of the polynomials handed to the native factorisation and
/// roots (`_X`).
fn rx() -> Basic {
    symbol("_rx")
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
fn integer_of(e: &Basic) -> Option<i64> {
    if !e.is_number_class() {
        return None;
    }
    let r: &Rat = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn is_integer_node(e: &Basic) -> bool {
    matches!(e.head(), Head::Rational { value, .. } if value.is_int())
}
fn is_rational_node(e: &Basic) -> bool {
    matches!(e.head(), Head::Rational { .. })
}
/// A `NotImplementedError` of the Python algorithm, not a part of the core
/// which is not ported.
fn genuine_not_implemented(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if !m.starts_with(NOT_PORTED_PREFIX))
}
fn not_implemented(message: String) -> PolyError {
    PolyError::NotImplemented(message)
}
fn has_function(e: &Basic, functions: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|node| functions.iter().any(|f| node.is_function(*f)))
}
fn has_sum_or_product(e: &Basic) -> bool {
    e.preorder()
        .iter()
        .any(|node| matches!(node.head(), Head::Sum | Head::Product))
}
fn has_sum(e: &Basic) -> bool {
    e.preorder().iter().any(|node| matches!(node.head(), Head::Sum))
}
fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b)
        && !crate::sets::sets::is_set(b)
        && !matches!(b.head(), Head::Tuple | Head::ExprCondPair)
}

// ---------------------------------------------------------------------------
// the fields of coefficients

/// An element of a [`Field`].
#[derive(Clone, Debug, PartialEq)]
pub enum Elem {
    /// An element of `QQ`.
    Q(Rat),
    /// An element of a number field `QQ(alpha)`.
    A(AlgebraicElement),
    /// An element of `QQ(a, b, ...)`.
    P(SparseFracElement<RationalField>),
    /// An element of an extension: a polynomial in the root over the base.
    E(Vec<Elem>),
}

/// The simple algebraic extension `K[t]/(m(t))` of a field which is not
/// `QQ` (`_ExtensionDomain`): elements are dense polynomials in `t` over
/// `K` of degree below that of `m`.
#[derive(Debug)]
pub struct Extension {
    pub base: Field,
    pub minpoly: Vec<Elem>,
    pub degree: usize,
    pub alpha_expr: Basic,
    pub name: String,
}

/// A field of coefficients (`_Field` with its domain).
#[derive(Clone, Debug)]
pub enum Field {
    Rational,
    Number(AlgebraicField),
    Parametric(SparseFractionField<RationalField>),
    Extension(Arc<Extension>),
}

impl PartialEq for Field {
    fn eq(&self, other: &Field) -> bool {
        match (self, other) {
            (Field::Rational, Field::Rational) => true,
            (Field::Number(a), Field::Number(b)) => a == b,
            (Field::Parametric(a), Field::Parametric(b)) => a == b,
            // the extension domains compare by identity, as in Python
            (Field::Extension(a), Field::Extension(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

fn q(a: &Elem) -> &Rat {
    match a {
        Elem::Q(r) => r,
        _ => unreachable!("an element of QQ"),
    }
}
fn av(a: &Elem) -> &AlgebraicElement {
    match a {
        Elem::A(r) => r,
        _ => unreachable!("an element of a number field"),
    }
}
fn pv(a: &Elem) -> &SparseFracElement<RationalField> {
    match a {
        Elem::P(r) => r,
        _ => unreachable!("an element of a field of rational functions"),
    }
}
fn ev(a: &Elem) -> &Vec<Elem> {
    match a {
        Elem::E(r) => r,
        _ => unreachable!("an element of an algebraic extension"),
    }
}
fn qs(f: &[Elem]) -> Vec<Rat> {
    f.iter().map(|c| q(c).clone()).collect()
}

impl Extension {
    fn reduce(&self, a: &[Elem]) -> Vec<Elem> {
        let a: Vec<Elem> = dup_strip(a, &self.base);
        if dup_degree(&a) < self.degree as isize {
            return a;
        }
        dup_rem(&a, &self.minpoly, &self.base).expect("a monic modulus")
    }
    fn one(&self) -> Vec<Elem> {
        vec![self.base.one()]
    }
    fn lift(&self, c: Elem) -> Vec<Elem> {
        if self.base.is_zero(&c) {
            Vec::new()
        } else {
            vec![c]
        }
    }
    fn alpha(&self) -> Vec<Elem> {
        self.reduce(&[self.base.one(), self.base.zero()])
    }
    fn mul(&self, a: &[Elem], b: &[Elem]) -> Vec<Elem> {
        self.reduce(&dup_mul(a, b, &self.base))
    }
    fn invert(&self, a: &[Elem]) -> PolyResult<Vec<Elem>> {
        if a.is_empty() {
            return Err(PolyError::ZeroDivision("0 has no inverse".to_string()));
        }
        dup_invert(a, &self.minpoly, &self.base)
    }
    fn pow_signed(&self, a: &[Elem], k: i64) -> PolyResult<Vec<Elem>> {
        let base: Vec<Elem> = if k < 0 { self.invert(a)? } else { a.to_vec() };
        let mut result: Vec<Elem> = self.one();
        for _ in 0..k.unsigned_abs() {
            result = self.mul(&result, &base);
        }
        Ok(result)
    }
    fn to_expr(&self, a: &[Elem]) -> R {
        let top: usize = a.len().saturating_sub(1);
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in a.iter().enumerate() {
            if !self.base.is_zero(c) {
                terms.push(mul(&[
                    self.base.to_expr(c)?,
                    pow(&self.alpha_expr, &int((top - i) as i64))?,
                ])?);
            }
        }
        add(&terms)
    }
    /// An expression made of `alpha_expr` and elements of the base field
    /// by sums, products and integer powers.
    fn from_expr(&self, e: &Basic) -> PolyResult<Vec<Elem>> {
        if *e == self.alpha_expr {
            return Ok(self.alpha());
        }
        if e.is_add() {
            let mut total: Vec<Elem> = Vec::new();
            for a in e.args() {
                total = dup_add(&total, &self.from_expr(a)?, &self.base);
            }
            return Ok(total);
        }
        if e.has(&self.alpha_expr) {
            if e.is_pow() && is_integer_node(&e.args()[1]) {
                let base: Vec<Elem> = self.from_expr(&e.args()[0])?;
                let k: i64 =
                    integer_of(&e.args()[1]).ok_or_else(|| PolyError::Value("an exponent too large".to_string()))?;
                return self.pow_signed(&base, k);
            }
            let mut product: Vec<Elem> = self.one();
            for factor in factors_of(e) {
                if factor == *e {
                    return Err(PolyError::Value(format!("{} is not an element of {}", e, self.name)));
                }
                product = self.mul(&product, &self.from_expr(&factor)?);
            }
            return Ok(product);
        }
        Ok(self.lift(self.base.from_expr(e)?))
    }
}

impl Domain for Field {
    type Elem = Elem;

    fn name(&self) -> String {
        match self {
            Field::Rational => QQ.name(),
            Field::Number(k) => k.name(),
            Field::Parametric(k) => k.name(),
            Field::Extension(x) => x.name.clone(),
        }
    }
    fn is_field(&self) -> bool {
        true
    }
    fn is_ordered(&self) -> bool {
        match self {
            Field::Rational => true,
            Field::Number(k) => k.is_ordered(),
            Field::Parametric(k) => k.is_ordered(),
            Field::Extension(_) => false,
        }
    }
    fn characteristic(&self) -> Int {
        Int::ZERO
    }
    fn zero(&self) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.zero()),
            Field::Number(k) => Elem::A(k.zero()),
            Field::Parametric(k) => Elem::P(k.zero()),
            Field::Extension(_) => Elem::E(Vec::new()),
        }
    }
    fn one(&self) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.one()),
            Field::Number(k) => Elem::A(k.one()),
            Field::Parametric(k) => Elem::P(k.one()),
            Field::Extension(x) => Elem::E(x.one()),
        }
    }
    fn to_expr(&self, a: &Elem) -> R {
        match self {
            Field::Rational => QQ.to_expr(q(a)),
            Field::Number(k) => k.to_expr(av(a)),
            Field::Parametric(k) => k.to_expr(pv(a)),
            Field::Extension(x) => x.to_expr(ev(a)),
        }
    }
    fn from_expr(&self, e: &Basic) -> PolyResult<Elem> {
        Ok(match self {
            Field::Rational => Elem::Q(QQ.from_expr(e)?),
            Field::Number(k) => Elem::A(k.from_expr(e)?),
            Field::Parametric(k) => Elem::P(k.from_expr(e)?),
            Field::Extension(x) => Elem::E(x.from_expr(e)?),
        })
    }
    fn is_zero(&self, a: &Elem) -> bool {
        match self {
            Field::Rational => QQ.is_zero(q(a)),
            Field::Number(k) => k.is_zero(av(a)),
            Field::Parametric(k) => k.is_zero(pv(a)),
            Field::Extension(_) => ev(a).is_empty(),
        }
    }
    fn is_one(&self, a: &Elem) -> bool {
        match self {
            Field::Rational => QQ.is_one(q(a)),
            Field::Number(k) => k.is_one(av(a)),
            Field::Parametric(k) => k.is_one(pv(a)),
            Field::Extension(x) => {
                let a: &Vec<Elem> = ev(a);
                a.len() == 1 && x.base.is_one(&a[0])
            }
        }
    }
    fn is_negative(&self, a: &Elem) -> bool {
        match self {
            Field::Rational => QQ.is_negative(q(a)),
            Field::Number(k) => k.is_negative(av(a)),
            Field::Parametric(k) => k.is_negative(pv(a)),
            Field::Extension(_) => false,
        }
    }
    fn add(&self, a: &Elem, b: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.add(q(a), q(b))),
            Field::Number(k) => Elem::A(k.add(av(a), av(b))),
            Field::Parametric(k) => Elem::P(k.add(pv(a), pv(b))),
            Field::Extension(x) => Elem::E(dup_add(ev(a), ev(b), &x.base)),
        }
    }
    fn sub(&self, a: &Elem, b: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.sub(q(a), q(b))),
            Field::Number(k) => Elem::A(k.sub(av(a), av(b))),
            Field::Parametric(k) => Elem::P(k.sub(pv(a), pv(b))),
            Field::Extension(x) => Elem::E(dup_sub(ev(a), ev(b), &x.base)),
        }
    }
    fn neg(&self, a: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.neg(q(a))),
            Field::Number(k) => Elem::A(k.neg(av(a))),
            Field::Parametric(k) => Elem::P(k.neg(pv(a))),
            Field::Extension(x) => Elem::E(dup_neg(ev(a), &x.base)),
        }
    }
    fn mul(&self, a: &Elem, b: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.mul(q(a), q(b))),
            Field::Number(k) => Elem::A(k.mul(av(a), av(b))),
            Field::Parametric(k) => Elem::P(k.mul(pv(a), pv(b))),
            Field::Extension(x) => Elem::E(x.mul(ev(a), ev(b))),
        }
    }
    fn pow(&self, a: &Elem, n: u64) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.pow(q(a), n)),
            Field::Number(k) => Elem::A(k.pow(av(a), n)),
            Field::Parametric(k) => Elem::P(k.pow(pv(a), n)),
            Field::Extension(x) => {
                let mut result: Vec<Elem> = x.one();
                for _ in 0..n {
                    result = x.mul(&result, ev(a));
                }
                Elem::E(result)
            }
        }
    }
    fn exquo(&self, a: &Elem, b: &Elem) -> PolyResult<Elem> {
        Ok(match self {
            Field::Rational => Elem::Q(QQ.exquo(q(a), q(b))?),
            Field::Number(k) => Elem::A(k.exquo(av(a), av(b))?),
            Field::Parametric(k) => Elem::P(k.exquo(pv(a), pv(b))?),
            Field::Extension(x) => Elem::E(x.mul(ev(a), &x.invert(ev(b))?)),
        })
    }
    fn quo(&self, a: &Elem, b: &Elem) -> PolyResult<Elem> {
        Ok(match self {
            Field::Rational => Elem::Q(QQ.quo(q(a), q(b))?),
            Field::Number(k) => Elem::A(k.quo(av(a), av(b))?),
            Field::Parametric(k) => Elem::P(k.quo(pv(a), pv(b))?),
            Field::Extension(x) => Elem::E(x.mul(ev(a), &x.invert(ev(b))?)),
        })
    }
    fn rem(&self, a: &Elem, b: &Elem) -> PolyResult<Elem> {
        Ok(match self {
            Field::Rational => Elem::Q(QQ.rem(q(a), q(b))?),
            Field::Number(k) => Elem::A(k.rem(av(a), av(b))?),
            Field::Parametric(k) => Elem::P(k.rem(pv(a), pv(b))?),
            Field::Extension(x) => {
                if ev(b).is_empty() {
                    return Err(PolyError::ZeroDivision(format!("division by zero in {}", x.name)));
                }
                Elem::E(Vec::new())
            }
        })
    }
    fn invert(&self, a: &Elem) -> PolyResult<Elem> {
        Ok(match self {
            Field::Rational => Elem::Q(QQ.invert(q(a))?),
            Field::Number(k) => Elem::A(k.invert(av(a))?),
            Field::Parametric(k) => Elem::P(k.invert(pv(a))?),
            Field::Extension(x) => Elem::E(x.invert(ev(a))?),
        })
    }
    fn gcd(&self, a: &Elem, b: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.gcd(q(a), q(b))),
            Field::Number(k) => Elem::A(k.gcd(av(a), av(b))),
            Field::Parametric(k) => Elem::P(k.gcd(pv(a), pv(b))),
            Field::Extension(x) => {
                if ev(a).is_empty() && ev(b).is_empty() {
                    Elem::E(Vec::new())
                } else {
                    Elem::E(x.one())
                }
            }
        }
    }
    fn lcm(&self, a: &Elem, b: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.lcm(q(a), q(b))),
            Field::Number(k) => Elem::A(k.lcm(av(a), av(b))),
            Field::Parametric(k) => Elem::P(k.lcm(pv(a), pv(b))),
            Field::Extension(x) => {
                if ev(a).is_empty() || ev(b).is_empty() {
                    Elem::E(Vec::new())
                } else {
                    Elem::E(x.one())
                }
            }
        }
    }
    fn gcdex(&self, a: &Elem, b: &Elem) -> PolyResult<(Elem, Elem, Elem)> {
        Ok(match self {
            Field::Rational => {
                let (s, t, h) = QQ.gcdex(q(a), q(b))?;
                (Elem::Q(s), Elem::Q(t), Elem::Q(h))
            }
            Field::Number(k) => {
                let (s, t, h) = k.gcdex(av(a), av(b))?;
                (Elem::A(s), Elem::A(t), Elem::A(h))
            }
            Field::Parametric(k) => {
                let (s, t, h) = k.gcdex(pv(a), pv(b))?;
                (Elem::P(s), Elem::P(t), Elem::P(h))
            }
            Field::Extension(x) => {
                let empty = || Elem::E(Vec::new());
                if !ev(a).is_empty() {
                    (Elem::E(x.invert(ev(a))?), empty(), Elem::E(x.one()))
                } else if !ev(b).is_empty() {
                    (empty(), Elem::E(x.invert(ev(b))?), Elem::E(x.one()))
                } else {
                    (empty(), empty(), empty())
                }
            }
        })
    }
    fn canonical_unit(&self, a: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.canonical_unit(q(a))),
            Field::Number(k) => Elem::A(k.canonical_unit(av(a))),
            Field::Parametric(k) => Elem::P(k.canonical_unit(pv(a))),
            Field::Extension(x) => {
                if ev(a).is_empty() {
                    Elem::E(x.one())
                } else {
                    Elem::E(x.invert(ev(a)).expect("a nonzero element"))
                }
            }
        }
    }
    fn sparse_echelon(&self, rows: &[SparseRow<Elem>]) -> Vec<SparseRow<Elem>> {
        match self {
            // QQ's fraction-free echelon form
            Field::Rational => {
                let converted: Vec<SparseRow<Rat>> = rows
                    .iter()
                    .map(|row| row.iter().map(|(j, c)| (*j, q(c).clone())).collect())
                    .collect();
                QQ.sparse_echelon(&converted)
                    .into_iter()
                    .map(|row| row.into_iter().map(|(j, c)| (j, Elem::Q(c))).collect())
                    .collect()
            }
            _ => self.try_sparse_echelon(rows).expect("a nonzero element of a field"),
        }
    }
    fn from_int(&self, n: i64) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.from_int(n)),
            Field::Number(k) => Elem::A(k.from_int(n)),
            Field::Parametric(k) => Elem::P(k.from_int(n)),
            Field::Extension(x) => Elem::E(x.lift(x.base.from_int(n))),
        }
    }
    fn from_integer(&self, n: &Int) -> Elem {
        match self {
            Field::Rational => Elem::Q(QQ.from_integer(n)),
            Field::Number(k) => Elem::A(k.from_integer(n)),
            Field::Parametric(k) => Elem::P(k.from_integer(n)),
            Field::Extension(x) => Elem::E(x.lift(x.base.from_integer(n))),
        }
    }
}

type Dense = Vec<Elem>;

// ---------------------------------------------------------------------------
// dense polynomials in n over a field

/// The coefficient of `n**j`.
fn coeff(f: &[Elem], j: i64, k: &Field) -> Elem {
    let i: i64 = f.len() as i64 - 1 - j;
    if 0 <= i && i < f.len() as i64 {
        f[i as usize].clone()
    } else {
        k.zero()
    }
}

/// `f(n + h)`.
fn shift(f: &[Elem], h: i64, k: &Field) -> Dense {
    if h == 0 || f.is_empty() {
        return f.to_vec();
    }
    dup_shift(f, &k.from_int(h), k)
}

/// `prod_{j = lo}^{hi - 1} p(n + j)`.
fn products(p: &[Elem], lo: i64, hi: i64, k: &Field) -> Dense {
    let mut result: Dense = dup_one(k);
    for j in lo..hi {
        result = dup_mul(&result, &shift(p, j, k), k);
    }
    result
}

fn monic(f: &[Elem], k: &Field) -> Dense {
    dup_monic(f, k).expect("monic over a field")
}

fn quo(f: &[Elem], g: &[Elem], k: &Field) -> Dense {
    dup_quo(f, g, k).expect("a division over a field")
}

/// The rational function `num/den` in lowest terms, `den` monic.
fn cancelled(num: &[Elem], den: &[Elem], k: &Field) -> (Dense, Dense) {
    if num.is_empty() {
        return (Vec::new(), dup_one(k));
    }
    let g: Dense = dup_gcd(num, den, k);
    let (num, den) = (quo(num, &g, k), quo(den, &g, k));
    let lead: Elem = dup_LC(&den, k);
    (
        dup_quo_ground(&num, &lead, k).expect("a nonzero leading coefficient"),
        monic(&den, k),
    )
}

/// `binomial(e, j)` as a polynomial in `e`.
fn binomial_polynomial(j: i64, k: &Field) -> Dense {
    let mut result: Dense = dup_one(k);
    for i in 0..j {
        let factor: Dense = if i != 0 {
            vec![k.one(), k.neg(&k.from_int(i))]
        } else {
            vec![k.one(), k.zero()]
        };
        result = dup_quo_ground(&dup_mul(&result, &factor, k), &k.from_int(i + 1), k).expect("a nonzero integer");
    }
    result
}

// ---------------------------------------------------------------------------
// the operations of the solvers on a field (`_Field`)

impl Field {
    /// An element as an element of the extensions of the field (`embed`).
    fn embed(&self, c: &Elem) -> Elem {
        match self {
            Field::Rational => Elem::A(if self.is_zero(c) {
                Vec::new()
            } else {
                vec![q(c).clone()]
            }),
            _ => Elem::E(if self.is_zero(c) { Vec::new() } else { vec![c.clone()] }),
        }
    }

    /// Rational polynomials whose common integer roots are the integer
    /// roots of `f` (`components`).
    fn components(&self, f: &[Elem]) -> PolyResult<Vec<Vec<Rat>>> {
        match self {
            Field::Rational => Ok(vec![qs(f)]),
            Field::Number(k) => {
                let m: usize = k.degree;
                let mut columns: Vec<Vec<Rat>> = vec![Vec::new(); m];
                for c in f {
                    let c: &Vec<Rat> = av(c);
                    let mut padded: Vec<Rat> = vec![Rat::ZERO; m - c.len()];
                    padded.extend(c.iter().cloned());
                    for j in 0..m {
                        columns[j].push(padded[j].clone());
                    }
                }
                Ok(columns.iter().map(|column| dup_strip(column, &QQ)).collect())
            }
            Field::Parametric(field) => {
                let ring: &SparsePolynomialRing<RationalField> = &field.ring;
                let mut common = ring.one();
                for c in f {
                    common = ring.lcm(&common, &pv(c).denom);
                }
                let top: usize = f.len().saturating_sub(1);
                let mut groups: Vec<(Vec<u32>, BTreeMap<usize, Rat>)> = Vec::new();
                for (i, c) in f.iter().enumerate() {
                    if field.is_zero(pv(c)) {
                        continue;
                    }
                    let c: &SparseFracElement<RationalField> = pv(c);
                    let numerator = ring.mul(&c.numer, &ring.exquo(&common, &c.denom)?);
                    for (monomial, value) in numerator.terms.iter() {
                        match groups.iter_mut().find(|(m, _)| m == monomial) {
                            Some((_, g)) => {
                                g.insert(top - i, value.clone());
                            }
                            None => {
                                let mut g: BTreeMap<usize, Rat> = BTreeMap::new();
                                g.insert(top - i, value.clone());
                                groups.push((monomial.clone(), g));
                            }
                        }
                    }
                }
                Ok(groups.iter().map(|(_, g)| dup_from_dict(g, &QQ)).collect())
            }
            Field::Extension(x) => {
                let m: usize = x.degree;
                let mut columns: Vec<Dense> = vec![Vec::new(); m];
                let zero: Elem = x.base.zero();
                for c in f {
                    let c: &Vec<Elem> = ev(c);
                    let mut padded: Vec<Elem> = vec![zero.clone(); m - c.len()];
                    padded.extend(c.iter().cloned());
                    for j in 0..m {
                        columns[j].push(padded[j].clone());
                    }
                }
                let mut result: Vec<Vec<Rat>> = Vec::new();
                for column in columns {
                    let stripped: Dense = dup_strip(&column, &x.base);
                    if !stripped.is_empty() {
                        result.extend(x.base.components(&stripped)?);
                    }
                }
                Ok(result)
            }
        }
    }

    /// The irreducible factors of `f` (of positive degree) with their
    /// multiplicities; `NotImplementedError` where there is no
    /// factorisation (`factor_dense`).
    fn factor_dense(&self, f: &[Elem]) -> PolyResult<Vec<(Dense, usize)>> {
        match self {
            // through Poly: SymPy's order of the factors
            Field::Rational | Field::Parametric(_) => native_factors(self, f),
            Field::Number(k) => {
                let converted: Vec<AlgebraicElement> = f.iter().map(|c| av(c).clone()).collect();
                let (_, factors) = k.factor_dense(&converted)?;
                Ok(factors
                    .into_iter()
                    .map(|(g, m)| (g.into_iter().map(Elem::A).collect(), m))
                    .collect())
            }
            Field::Extension(x) => Err(not_implemented(format!("no factorisation over {}", x.name))),
        }
    }

    /// The monic irreducible factors of `f` with their multiplicities.
    fn factor(&self, f: &[Elem]) -> PolyResult<Vec<(Dense, usize)>> {
        if dup_degree(f) <= 0 {
            return Ok(Vec::new());
        }
        Ok(self
            .factor_dense(f)?
            .into_iter()
            .filter(|(g, _)| dup_degree(g) > 0)
            .map(|(g, m)| (monic(&g, self), m))
            .collect())
    }

    /// The field with `root`, a root of the monic irreducible `f`,
    /// adjoined (`extend`).
    fn extend(&self, f: &[Elem], root: &Basic) -> PolyResult<Field> {
        match self {
            Field::Rational => Ok(Field::Number(AlgebraicField::new(qs(&monic(f, self)), root.clone())?)),
            _ => {
                let minpoly: Dense = monic(f, self);
                let degree: usize = dup_degree(&minpoly) as usize;
                Ok(Field::Extension(Arc::new(Extension {
                    base: self.clone(),
                    minpoly,
                    degree,
                    alpha_expr: root.clone(),
                    name: format!("{}<{}>", self.name(), root),
                })))
            }
        }
    }

    /// The polynomial as an expression in `x`.
    fn expression(&self, f: &[Elem], x: &Basic) -> R {
        let top: usize = f.len().saturating_sub(1);
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in f.iter().enumerate() {
            if !self.is_zero(c) {
                terms.push(mul(&[self.to_expr(c)?, pow(x, &int((top - i) as i64))?])?);
            }
        }
        add(&terms)
    }

    /// A polynomial expression in `x` as a dense polynomial; `ValueError`
    /// when it is not one over the field.
    fn dense(&self, e: &Basic, x: &Basic) -> PolyResult<Dense> {
        let mut terms: BTreeMap<usize, Elem> = BTreeMap::new();
        for term in terms_of(&expand(e)?) {
            if term == Basic::zero() {
                continue;
            }
            let (coefficient, dependent) = crate::core::expr::as_independent(&term, x, false)?;
            let j: usize = if dependent == Basic::one() {
                0
            } else if dependent == *x {
                1
            } else if dependent.is_pow()
                && dependent.args()[0] == *x
                && is_integer_node(&dependent.args()[1])
                && integer_of(&dependent.args()[1]).is_some_and(|p| p > 0)
            {
                integer_of(&dependent.args()[1]).unwrap_or(0) as usize
            } else {
                return Err(PolyError::Value(format!("{} is not a polynomial in {}", e, x)));
            };
            let value: Elem = self.from_expr(&coefficient)?;
            let total: Elem = match terms.get(&j) {
                Some(previous) => self.add(previous, &value),
                None => value,
            };
            terms.insert(j, total);
        }
        let nonzero: BTreeMap<usize, Elem> = terms.into_iter().filter(|(_, c)| !self.is_zero(c)).collect();
        Ok(dup_from_dict(&nonzero, self))
    }

    /// A rational function of `x` as a numerator and a monic denominator
    /// in lowest terms.
    fn rational(&self, e: &Basic, x: &Basic) -> PolyResult<(Dense, Dense)> {
        let together: Basic = crate::polys::rationaltools::together(e, false, true)?;
        let (numerator, denominator) = crate::core::expr::as_numer_denom(&together)?;
        let (num, den) = (self.dense(&numerator, x)?, self.dense(&denominator, x)?);
        if den.is_empty() {
            return Err(PolyError::Value(format!("{} has a zero denominator", e)));
        }
        Ok(cancelled(&num, &den, self))
    }
}

/// The factorisation by [`Poly`] (over `QQ` and `QQ[a, ...]`, the
/// denominators cleared first).
fn native_factors(field: &Field, f: &[Elem]) -> PolyResult<Vec<(Dense, usize)>> {
    let x: Basic = rx();
    let together: Basic = crate::polys::rationaltools::together(&field.expression(f, &x)?, false, true)?;
    let (numerator, _) = crate::core::expr::as_numer_denom(&together)?;
    let factors = match Poly::new(&numerator, std::slice::from_ref(&x)).and_then(|p| p.factor_list()) {
        Ok((_, factors)) => factors,
        Err(PolyError::Polys(_, message)) => return Err(not_implemented(message)),
        Err(e) => return Err(e),
    };
    let mut result: Vec<(Dense, usize)> = Vec::new();
    for (g, m) in factors {
        if g.degree(0)? > 0 {
            result.push((field.dense(&g.as_expr()?, &x)?, m));
        }
    }
    Ok(result)
}

/// The nonnegative integer roots of the nonzero polynomial `f`.
fn integer_roots(field: &Field, f: &[Elem]) -> PolyResult<Vec<i64>> {
    let mut g: Vec<Rat> = Vec::new();
    for component in field.components(f)? {
        if !component.is_empty() {
            g = if g.is_empty() {
                component
            } else {
                dup_gcd(&g, &component, &QQ)
            };
        }
    }
    if dup_degree(&g) <= 0 {
        return Ok(Vec::new());
    }
    let mut found: BTreeSet<i64> = BTreeSet::new();
    for (h, _) in QQ.factor_dense(&g)?.1 {
        if dup_degree(&h) == 1 {
            let r: Rat = -h[1].clone() / h[0].clone();
            if r.is_int() && r >= Rat::ZERO {
                if let Ok(v) = i64::try_from(&crate::numbers::numer(&r)) {
                    found.insert(v);
                }
            }
        }
    }
    Ok(found.into_iter().collect())
}

// ---------------------------------------------------------------------------
// the context: a field of coefficients with the renaming of its parameters

/// A field of coefficients with the names of its parameters
/// (`_FieldContext`).
struct Context {
    field: Field,
    /// the names inside the field to the symbols of the problem
    restore: Vec<(Basic, Basic)>,
    renaming: Vec<(Basic, Basic)>,
}

/// The field of the coefficients of `exprs` (rational functions of `n`):
/// `QQ`, `QQ(parameters)` or `QQ(alpha)` for one radical `alpha` (or `I`)
/// (`_context`).
fn context(exprs: &[Basic], n: &Basic) -> PolyResult<Context> {
    let mut symbols: Vec<Basic> = Vec::new();
    for e in exprs {
        for s in e.free_symbols() {
            if s != *n && !symbols.contains(&s) {
                symbols.push(s);
            }
        }
    }
    symbols.sort_by_cached_key(crate::core::sorting::sort_key);
    let mut radicals: Vec<(Basic, Int)> = Vec::new();
    let mut imaginary: bool = false;
    let i_unit: Basic = Basic::atom(Head::ImaginaryUnit);
    for e in exprs {
        for atom in e.preorder() {
            if atom.is_pow()
                && is_rational_node(&atom.args()[0])
                && is_rational_node(&atom.args()[1])
                && !is_integer_node(&atom.args()[1])
            {
                let exponent: &Rat = atom.args()[1].as_rational().expect("a rational exponent");
                let entry: (Basic, Int) = (atom.args()[0].clone(), crate::numbers::denom(exponent));
                if !radicals.contains(&entry) {
                    radicals.push(entry);
                }
            }
        }
        if e.has(&i_unit) {
            imaginary = true;
        }
    }
    if !radicals.is_empty() || imaginary {
        if !symbols.is_empty() {
            return Err(not_implemented(
                "coefficients with parameters and algebraic numbers are not supported".to_string(),
            ));
        }
        let mut bases: Vec<Basic> = Vec::new();
        for (b, _) in &radicals {
            if !bases.contains(b) {
                bases.push(b.clone());
            }
        }
        if (imaginary && !radicals.is_empty()) || bases.len() > 1 {
            return Err(not_implemented(
                "coefficients with several algebraic numbers are not supported".to_string(),
            ));
        }
        let alpha: Basic = if imaginary {
            i_unit
        } else {
            let mut qd: Int = Int::ONE;
            for (_, d) in &radicals {
                qd = &qd * d / crate::numbers::gcd(&qd, d);
            }
            pow(&bases[0], &Basic::rational(Int::ONE, qd)?)?
        };
        let minpoly: Vec<Rat> = crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(&alpha)?
            .into_iter()
            .map(Rat::from)
            .collect();
        let field: AlgebraicField = AlgebraicField::new(dup_monic(&minpoly, &QQ)?, alpha)?;
        return Ok(Context {
            field: Field::Number(field),
            restore: Vec::new(),
            renaming: Vec::new(),
        });
    }
    if symbols.is_empty() {
        return Ok(Context {
            field: Field::Rational,
            restore: Vec::new(),
            renaming: Vec::new(),
        });
    }
    let names: Vec<String> = (0..symbols.len()).map(|i| format!("_rr{}", i)).collect();
    let restore: Vec<(Basic, Basic)> = names
        .iter()
        .zip(&symbols)
        .map(|(name, s)| (symbol(name), s.clone()))
        .collect();
    let renaming: Vec<(Basic, Basic)> = restore.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
    let field = SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names));
    Ok(Context {
        field: Field::Parametric(field),
        restore,
        renaming,
    })
}

impl Context {
    /// `e` with the parameters named as inside the field.
    fn rename(&self, e: &Basic) -> R {
        if self.renaming.is_empty() {
            return Ok(e.clone());
        }
        crate::core::subs::subs_mapping(e, &self.renaming)
    }

    /// `e` with the parameters named as in the problem.
    fn restored(&self, e: &Basic) -> R {
        if self.restore.is_empty() {
            return Ok(e.clone());
        }
        crate::core::subs::subs_mapping(e, &self.restore)
    }

    /// The recurrence with these coefficients (renamed).
    fn operator(&self, coefficients: &[Basic], n: &Basic, normalize: bool) -> PolyResult<Operator> {
        let renamed: Vec<Basic> = coefficients.iter().map(|c| self.rename(c)).collect::<PolyResult<_>>()?;
        let polys: Vec<Dense> = coefficient_polys(&self.field, &renamed, n)?;
        Ok(Operator {
            field: self.field.clone(),
            polys: if normalize {
                normalized(&self.field, &polys)
            } else {
                polys
            },
            n: n.clone(),
        })
    }

    /// [`gosper_form`] of `r` (renamed).
    fn gosper(&self, r: &Basic, n: &Basic) -> PolyResult<(Basic, Poly, Poly, Poly)> {
        let field: &Field = &self.field;
        let (p, qd) = field.rational(&self.rename(r)?, n)?;
        if p.is_empty() {
            return Err(PolyError::Value("the ratio is zero".to_string()));
        }
        let (z, a, b, c) = gosper(field, &p, &qd)?;
        Ok((
            self.restored(&field.to_expr(&z)?)?,
            monic_poly(&self.restored(&field.expression(&a, n)?)?, n)?,
            monic_poly(&self.restored(&field.expression(&b, n)?)?, n)?,
            monic_poly(&self.restored(&field.expression(&c, n)?)?, n)?,
        ))
    }
}

/// A recurrence over a field (`_OperatorOver`).
struct Operator {
    field: Field,
    polys: Vec<Dense>,
    n: Basic,
}

impl Operator {
    /// The order (`-1` for the zero operator).
    fn order(&self) -> i64 {
        self.polys.len() as i64 - 1
    }

    fn polynomial_solutions(&self) -> PolyResult<Vec<Basic>> {
        polynomial_solutions(&self.field, &self.polys)?
            .iter()
            .map(|c| self.field.expression(c, &self.n))
            .collect()
    }

    fn hypergeometric_solutions(&self, avoid: &BTreeSet<Basic>) -> PolyResult<Vec<Solution>> {
        Ok(hypergeometric_records(&self.field, &self.polys, &self.n, 0, avoid)?
            .into_iter()
            .map(|h| (h.expr, h.start))
            .collect())
    }

    fn dalembertian_solutions(&self, avoid: &BTreeSet<Basic>) -> PolyResult<Vec<Solution>> {
        dalembertian_records(&self.field, &self.polys, &self.n, 0, avoid)
    }

    fn coefficients(&self) -> PolyResult<Vec<Basic>> {
        self.polys.iter().map(|p| self.field.expression(p, &self.n)).collect()
    }

    fn checked(&self, solution: &Basic, rhs: &Basic, start: i64, count: i64) -> PolyResult<bool> {
        checked(&self.coefficients()?, &self.n, solution, rhs, start, count)
    }

    fn particular(&self, f: &Basic, rho: &Basic, avoid: &BTreeSet<Basic>) -> PolyResult<Option<Solution>> {
        let (field, n) = (&self.field, &self.n);
        let (num, den) = match field.rational(rho, n) {
            Ok(fraction) => fraction,
            Err(PolyError::Value(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        let d: usize = self.polys.len() - 1;
        let mut extended: Vec<Dense> = Vec::new();
        for i in 0..d + 2 {
            let previous: Dense = if (1..=d + 1).contains(&i) {
                dup_mul(&den, &shift(&self.polys[i - 1], 1, field), field)
            } else {
                Vec::new()
            };
            let current: Dense = if i <= d {
                dup_mul(&num, &self.polys[i], field)
            } else {
                Vec::new()
            };
            extended.push(dup_sub(&previous, &current, field));
        }
        while extended.last().is_some_and(|p| p.is_empty()) {
            extended.pop();
        }
        let operator: Vec<Dense> = normalized(field, &extended);
        let coefficients: Vec<Basic> = self.coefficients()?;
        for (u, start) in dalembertian_records(field, &operator, n, 0, avoid)? {
            for at in start..start + 4 {
                let mut terms: Vec<Basic> = Vec::new();
                for (i, c) in coefficients.iter().enumerate() {
                    terms.push(mul(&[value(c, n, at)?, value(&u, n, at + i as i64)?])?);
                }
                let total: Basic = add(&terms)?;
                let f_value: Basic = value(f, n, at)?;
                if f_value == Basic::zero() || not_finite(&total) || not_finite(&f_value) {
                    continue;
                }
                let kappa: Basic = normal(&div(&total, &f_value)?)?;
                if is_zero(&kappa)? {
                    break;
                }
                let candidate: Basic = div(&u, &kappa)?;
                if checked(&coefficients, n, &candidate, f, start, 6)? {
                    return Ok(Some((candidate, start)));
                }
                break;
            }
        }
        Ok(None)
    }
}

fn not_finite(e: &Basic) -> bool {
    crate::core::expr::is(e, Predicate::finite) == Some(false)
}

/// The monic polynomial `e` as a `Poly` over the field, as SymPy's
/// `monic` gives it (scaled first: the native `monic` keeps `ZZ` when the
/// leading coefficient is already one).
fn monic_poly(e: &Basic, n: &Basic) -> PolyResult<Poly> {
    Poly::new(&mul(&[int(2), e.clone()])?, std::slice::from_ref(n))?.monic()
}

/// The coefficients as polynomials in `n` over the field, the zero
/// leading ones dropped; rational functions of `n` are brought to a common
/// denominator first.
fn coefficient_polys(field: &Field, exprs: &[Basic], n: &Basic) -> PolyResult<Vec<Dense>> {
    let mut fractions: Vec<(Dense, Dense)> = Vec::new();
    for e in exprs {
        match field.rational(e, n) {
            Ok(fraction) => fractions.push(fraction),
            Err(PolyError::Value(_)) => {
                let shown: Vec<String> = exprs.iter().map(|e| e.to_string()).collect();
                return Err(PolyError::Value(format!(
                    "the coefficients must be rational functions of {}: [{}]",
                    n,
                    shown.join(", ")
                )));
            }
            Err(e) => return Err(e),
        }
    }
    let mut common: Dense = dup_one(field);
    for (_, den) in &fractions {
        common = dup_lcm(&common, den, field);
    }
    let mut polys: Vec<Dense> = fractions
        .iter()
        .map(|(num, den)| dup_mul(num, &quo(&common, den, field), field))
        .collect();
    while polys.last().is_some_and(|p| p.is_empty()) {
        polys.pop();
    }
    Ok(polys)
}

/// The recurrence with its trailing coefficient nonzero: when
/// `p_0 = ... = p_{s-1} = 0` the equation is one in `y(n + s), ...`, the
/// same recurrence at the index `n + s`.
fn normalized(field: &Field, polys: &[Dense]) -> Vec<Dense> {
    let s: usize = polys.iter().take_while(|p| p.is_empty()).count();
    let rest: &[Dense] = &polys[s..];
    if s == 0 {
        return rest.to_vec();
    }
    rest.iter().map(|p| shift(p, -(s as i64), field)).collect()
}

// ---------------------------------------------------------------------------
// polynomial solutions

/// The largest possible degree of a polynomial solution of the
/// recurrence, or `None` when there is none (`_degree_bound`).
fn degree_bound(field: &Field, polys: &[Dense]) -> PolyResult<Option<i64>> {
    let big_d: i64 = polys.iter().map(|p| dup_degree(p) as i64).max().unwrap_or(-1);
    let order: i64 = polys.len() as i64 - 1;
    for k in 0..big_d + order + 2 {
        let mut total: Dense = Vec::new();
        for j in 0..k + 1 {
            let m: i64 = big_d - k + j;
            if m < 0 {
                continue;
            }
            let mut s: Elem = field.zero();
            for (i, p) in polys.iter().enumerate() {
                let power: Int = Int::from(i as i64).pow(j as usize);
                s = field.add(&s, &field.mul(&coeff(p, m, field), &field.from_integer(&power)));
            }
            if !field.is_zero(&s) {
                total = dup_add(
                    &total,
                    &dup_mul_ground(&binomial_polynomial(j, field), &s, field),
                    field,
                );
            }
        }
        if !total.is_empty() {
            let candidates: Vec<i64> = integer_roots(field, &total)?;
            return Ok(candidates.iter().max().copied());
        }
    }
    Ok(None) // the zero operator
}

/// `L[n**j] = sum p_i(n) (n + i)**j`.
fn image_of_power(field: &Field, polys: &[Dense], j: usize) -> Dense {
    let mut result: Dense = Vec::new();
    for (i, p) in polys.iter().enumerate() {
        let linear: Dense = if i != 0 {
            vec![field.one(), field.from_int(i as i64)]
        } else {
            vec![field.one(), field.zero()]
        };
        let power: Dense = dup_pow(&linear, j, field);
        result = dup_add(&result, &dup_mul(p, &power, field), field);
    }
    result
}

/// A basis of the polynomial solutions of `L y = 0`.
fn polynomial_solutions(field: &Field, polys: &[Dense]) -> PolyResult<Vec<Dense>> {
    let Some(bound) = degree_bound(field, polys)? else {
        return Ok(Vec::new());
    };
    // the images L[n**j] as polynomials: the columns of the linear system
    let images: Vec<Dense> = (0..=bound as usize).map(|j| image_of_power(field, polys, j)).collect();
    let height: i64 = images
        .iter()
        .filter(|p| !p.is_empty())
        .map(|p| dup_degree(p) as i64)
        .chain(std::iter::once(0))
        .max()
        .unwrap_or(0)
        + 1;
    let rows: Vec<Vec<Elem>> = (0..height)
        .map(|e| images.iter().map(|image| coeff(image, e, field)).collect())
        .collect();
    Ok(field
        .nullspace(&rows, bound as usize + 1)?
        .into_iter()
        .map(|v| {
            let reversed: Vec<Elem> = v.into_iter().rev().collect();
            dup_strip(&reversed, field)
        })
        .collect())
}

/// A basis of the polynomial solutions of
/// `sum coefficients[i] * y(n + i) = 0` (`rsolve_polynomial`).
pub fn rsolve_polynomial(coefficients: &[Basic], n: &Basic) -> PolyResult<Vec<Basic>> {
    let context: Context = context(coefficients, n)?;
    let operator: Operator = context.operator(coefficients, n, true)?;
    if operator.order() < 0 {
        return Err(PolyError::Value("the recurrence is trivial".to_string()));
    }
    operator
        .polynomial_solutions()?
        .iter()
        .map(|e| context.restored(e))
        .collect()
}

// ---------------------------------------------------------------------------
// the Gosper–Petkovšek form

/// The Gosper–Petkovšek form `(z, a, b, c)` of a rational function:
/// `r = z * a(n)/b(n) * c(n + 1)/c(n)` with `a`, `b`, `c` monic and
/// `gcd(a(n), b(n + h)) = 1` for every nonnegative integer `h`
/// (`gosper_form`).
pub fn gosper_form(r: &Basic, n: &Basic) -> PolyResult<(Basic, Poly, Poly, Poly)> {
    context(std::slice::from_ref(r), n)?.gosper(r, n)
}

fn gosper(field: &Field, p: &[Elem], qd: &[Elem]) -> PolyResult<(Elem, Dense, Dense, Dense)> {
    let z: Elem = field.quo(&dup_LC(p, field), &dup_LC(qd, field))?;
    let mut a: Dense = monic(p, field);
    let mut b: Dense = monic(qd, field);
    let mut c: Dense = dup_one(field);
    for root in shift_coincidences(field, &a, &b)? {
        let g: Dense = dup_gcd(&a, &shift(&b, root, field), field);
        if dup_degree(&g) <= 0 {
            continue;
        }
        a = monic(&quo(&a, &g, field), field);
        b = monic(&quo(&b, &shift(&g, -root, field), field), field);
        for i in 1..=root {
            c = dup_mul(&c, &shift(&g, -i, field), field);
        }
    }
    let c: Dense = monic(&c, field);
    Ok((z, a, b, c))
}

/// The monic polynomial `q` with `num/den = q(n + 1)/q(n)`, or `None` when
/// the rational function is not such a quotient (`_shift_quotient`).
fn shift_quotient(field: &Field, num: &[Elem], den: &[Elem]) -> PolyResult<Option<Dense>> {
    let solutions: Vec<Dense> = polynomial_solutions(field, &normalized(field, &[dup_neg(num, field), den.to_vec()]))?;
    if solutions.len() != 1 {
        return Ok(None);
    }
    Ok(Some(monic(&solutions[0], field)))
}

// ---------------------------------------------------------------------------
// hypergeometric solutions

/// The nonnegative integers `h` with `gcd(a(n), b(n + h)) != 1`, read off
/// the second coefficients of the irreducible factors of the same degree
/// (`_shift_coincidences`).
fn shift_coincidences(field: &Field, a: &[Elem], b: &[Elem]) -> PolyResult<Vec<i64>> {
    if dup_degree(a) <= 0 || dup_degree(b) <= 0 {
        return Ok(Vec::new());
    }
    let mut found: BTreeSet<i64> = BTreeSet::new();
    let factors_b: Vec<Dense> = field.factor(b)?.into_iter().map(|(g, _)| g).collect();
    for (f, _) in field.factor(a)? {
        let m: i64 = dup_degree(&f) as i64;
        for g in &factors_b {
            if dup_degree(g) as i64 != m {
                continue;
            }
            let difference: Elem = field.sub(&coeff(&f, m - 1, field), &coeff(g, m - 1, field));
            let h: Basic = field.to_expr(&field.quo(&difference, &field.from_int(m))?)?;
            if is_integer_node(&h) {
                if let Some(hp) = integer_of(&h).filter(|v| *v >= 0) {
                    if shift(g, hp, field) == f {
                        found.insert(hp);
                    }
                }
            }
        }
    }
    Ok(found.into_iter().collect())
}

/// The pairs `(a, b)` of monic divisors of `p0` and `pd` with
/// `gcd(a(n), b(n + h)) = 1` for every nonnegative integer `h`
/// (`_divisor_pairs`).
fn divisor_pairs(field: &Field, p0: &[Elem], pd: &[Elem]) -> PolyResult<Vec<(Dense, Dense)>> {
    let factors_a: Vec<(Dense, usize)> = field.factor(p0)?;
    let factors_b: Vec<(Dense, usize)> = field.factor(pd)?;
    let mut incompatible: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (i, (f, _)) in factors_a.iter().enumerate() {
        for (j, (g, _)) in factors_b.iter().enumerate() {
            if !shift_coincidences(field, f, g)?.is_empty() {
                incompatible.insert((i, j));
            }
        }
    }
    let divisors = |factors: &[(Dense, usize)]| -> Vec<(Dense, BTreeSet<usize>)> {
        let mut found: Vec<(Dense, BTreeSet<usize>)> = vec![(dup_one(field), BTreeSet::new())];
        for (index, (f, multiplicity)) in factors.iter().enumerate() {
            let mut next: Vec<(Dense, BTreeSet<usize>)> = Vec::new();
            for (d, used) in &found {
                for e in 0..=*multiplicity {
                    let mut used: BTreeSet<usize> = used.clone();
                    if e != 0 {
                        used.insert(index);
                    }
                    next.push((dup_mul(d, &dup_pow(f, e, field), field), used));
                }
            }
            found = next;
        }
        found
    };
    let mut pairs: Vec<(Dense, Dense)> = Vec::new();
    let divisors_b: Vec<(Dense, BTreeSet<usize>)> = divisors(&factors_b);
    for (a, used_a) in divisors(&factors_a) {
        for (b, used_b) in &divisors_b {
            if used_a
                .iter()
                .any(|i| used_b.iter().any(|j| incompatible.contains(&(*i, *j))))
            {
                continue;
            }
            pairs.push((a.clone(), b.clone()));
        }
    }
    Ok(pairs)
}

/// A summation index for the given depth which is none of the symbols of
/// the problem (`_index`).
fn index(depth: usize, avoid: &BTreeSet<Basic>) -> Basic {
    let mut names: Vec<String> = INDEX_NAMES.iter().map(|s| s.to_string()).collect();
    for i in 1..100 {
        for c in INDEX_NAMES {
            names.push(format!("{}{}", c, i));
        }
    }
    let taken: BTreeSet<String> = avoid.iter().filter(|s| s.is_symbol()).map(|s| s.to_string()).collect();
    let free: Vec<&String> = names.iter().filter(|name| !taken.contains(*name)).collect();
    crate::core::symbol::Symbol::new(free[depth], &[(Predicate::integer, true)]).0
}

/// The first index from which `prod a(k)/b(k)` has no vanishing factor
/// (`_start_of`).
fn start_of(field: &Field, a: &[Elem], b: &[Elem]) -> PolyResult<i64> {
    let product: Dense = dup_mul(a, b, field);
    let roots: Vec<i64> = if dup_degree(&product) > 0 {
        integer_roots(field, &product)?
    } else {
        Vec::new()
    };
    Ok(roots.last().map(|r| r + 1).unwrap_or(0))
}

/// `prod_{k = start}^{n - 1} p(k)` with the linear factors written as
/// rising factorials (`_product_expression`).
fn product_expression(field: &Field, p: &[Elem], n: &Basic, start: i64, k: &Basic) -> R {
    let mut content: Basic = if p.is_empty() {
        Basic::zero()
    } else {
        field.to_expr(&dup_LC(p, field))?
    };
    let mut pieces: Vec<Basic> = Vec::new();
    let span: Basic = sub(n, &int(start))?;
    for (f, multiplicity) in field.factor(p)? {
        let coefficients: Vec<Basic> = f.iter().map(|c| field.to_expr(c)).collect::<PolyResult<_>>()?;
        let mut lead: Basic = Basic::one();
        if coefficients.iter().all(is_rational_node) {
            let mut scale: Int = Int::ONE;
            for c in &coefficients {
                if let Some(r) = c.as_rational() {
                    let d: Int = crate::numbers::denom(r);
                    scale = &scale * &d / crate::numbers::gcd(&scale, &d);
                }
            }
            lead = Basic::integer(scale);
        }
        content = div(&content, &pow(&lead, &int(multiplicity as i64))?)?;
        let piece: Basic = if dup_degree(&f) == 1 {
            let root: Basic = neg(&coefficients[1])?;
            mul(&[
                pow(&lead, &span)?,
                crate::functions::combinatorial::factorials::rf(&sub(&int(start), &root)?, &span)?,
            ])?
        } else {
            let body: Basic = subs(&mul(&[lead.clone(), field.expression(&f, n)?])?, n, k)?;
            crate::concrete::expr_with_limits::product_(
                &body,
                &[Basic::new(Head::Tuple, vec![k.clone(), int(start), sub(n, &int(1))?])],
            )?
        };
        pieces.push(pow(&piece, &int(multiplicity as i64))?);
    }
    let mut factors: Vec<Basic> = vec![pow(&content, &span)?];
    factors.extend(pieces);
    mul(&factors)
}

/// The quotients of rising factorials cancelled (`gammasimp`), when that
/// keeps the expression free of gamma functions (`_simplified`).
fn simplified(expr: &Basic) -> R {
    let mut candidate: Basic = crate::simplify::gammasimp::gammasimp(expr)?;
    if has_function(&candidate, &[Function::Gamma])
        || candidate.preorder().iter().any(|e| matches!(e.head(), Head::Piecewise))
    {
        candidate = expr.clone();
    }
    crate::core::function::replace_where(
        &candidate,
        &|e: &Basic| e.is_function(Function::RisingFactorial),
        &factorial_form,
    )
}

/// `RisingFactorial(a, m)` as `(a + m - 1)!/(a - 1)!` for a positive
/// integer `a` (`_factorial_form`).
fn factorial_form(e: &Basic) -> R {
    if e.is_function(Function::RisingFactorial) {
        let (a, m) = (&e.args()[0], &e.args()[1]);
        if is_integer_node(a) && integer_of(a).is_some_and(|v| v >= 1) && is_expr(m) {
            use crate::functions::combinatorial::factorials::factorial;
            return div(
                &factorial(&add(&[a.clone(), m.clone(), int(-1)])?)?,
                &factorial(&add(&[a.clone(), int(-1)])?)?,
            );
        }
    }
    Ok(e.clone())
}

/// Whether the error is one of those `_closed_form` catches.
fn closed_form_failure(e: &PolyError) -> bool {
    matches!(
        e,
        PolyError::Value(_)
            | PolyError::Type(_)
            | PolyError::Polys(..)
            | PolyError::Arithmetic(_)
            | PolyError::ZeroDivision(_)
            | PolyError::Pole(_)
    ) || genuine_not_implemented(e)
}

/// The sums evaluated by the summation when that gives an expression
/// without sums, products or special functions (`_closed_form`).
fn closed_form(expr: &Basic) -> R {
    if !has_sum(expr) {
        return Ok(expr.clone());
    }
    let sums: Vec<Basic> = expr.atoms(Some(&|e: &Basic| matches!(e.head(), Head::Sum)));
    if sums.iter().any(|s| has_sum(&s.args()[0])) {
        return Ok(expr.clone()); // the symbolic summation of nested sums took minutes
    }
    let from_one: Basic =
        crate::core::function::replace_where(expr, &|e: &Basic| matches!(e.head(), Head::Sum), &from_one)?;
    let mut forms: Vec<Basic> = vec![expr.clone()];
    if is_expr(&from_one) {
        forms.push(from_one);
    }
    let special: [Function; 7] = [
        Function::Gamma,
        Function::UpperGamma,
        Function::LowerGamma,
        Function::PolyLog,
        Function::LerchPhi,
        Function::Zeta,
        Function::RisingFactorial,
    ];
    for form in forms {
        let candidate: Basic = match doit(&form) {
            Ok(c) => c,
            Err(e) if closed_form_failure(&e) => continue,
            Err(e) => return Err(e),
        };
        let unwanted: bool = candidate
            .preorder()
            .iter()
            .any(|e| matches!(e.head(), Head::Sum | Head::Product | Head::Hyper | Head::Piecewise))
            || has_function(&candidate, &special[..6]);
        if !unwanted {
            return Ok(candidate);
        }
    }
    Ok(expr.clone())
}

/// `Sum(f(k), (k, 0, m))` as `Sum(f(k - 1), (k, 1, m + 1))`, the form in
/// which the summation recognises the harmonic numbers (`_from_one`).
fn from_one(s: &Basic) -> R {
    if matches!(s.head(), Head::Sum) {
        let lims = crate::concrete::expr_with_limits::limits(s);
        if lims.len() == 1 && lims[0].1 == Basic::zero() {
            let (k, _, hi) = &lims[0];
            let body: Basic = subs(&s.args()[0], k, &sub(k, &int(1))?)?;
            return crate::concrete::expr_with_limits::sum_(
                &body,
                &[Basic::new(
                    Head::Tuple,
                    vec![k.clone(), int(1), add(&[hi.clone(), int(1)])?],
                )],
            );
        }
    }
    Ok(s.clone())
}

/// A hypergeometric solution with the first index from which it is
/// defined, the operator over its field and its ratio `num/den`
/// (`_HyperOver`).
struct Hyper {
    expr: Basic,
    start: i64,
    field: Field,
    polys: Vec<Dense>,
    num: Dense,
    den: Dense,
}

impl Hyper {
    /// The d'Alembertian solutions of the reduced operator;
    /// `NotImplementedError` when it lives over a field without a
    /// factorisation.
    fn reduced_solutions(&self, n: &Basic, depth: usize, avoid: &BTreeSet<Basic>) -> PolyResult<Vec<Solution>> {
        let reduced: Vec<Dense> = reduced_operator(&self.field, &self.polys, &self.num, &self.den);
        dalembertian_records(&self.field, &reduced, n, depth, avoid)
    }
}

/// A similarity class of hypergeometric terms (`_Class`).
struct Class {
    key: (usize, usize, usize),
    num: Dense,
    den: Dense,
    start: i64,
    products: Basic,
    polynomials: Vec<Dense>,
}

/// A pair `(a, b)` of the search with the operator `P_i` it gives
/// (`_Candidate`).
struct Candidate {
    key: (usize, usize),
    a: Dense,
    b: Dense,
    p: Vec<Dense>,
    start: i64,
    products: Basic,
}

/// The candidates sharing a root `z` (`_RootGroup`).
struct RootGroup {
    expr: Basic,
    value: Option<Elem>,
    factor: Dense,
    candidates: Vec<Candidate>,
}

/// The hypergeometric solutions (Petkovšek), in the order in which their
/// classes are found (`_hypergeometric_records`).
fn hypergeometric_records(
    field: &Field,
    polys: &[Dense],
    n: &Basic,
    depth: usize,
    avoid: &BTreeSet<Basic>,
) -> PolyResult<Vec<Hyper>> {
    let d: usize = polys.len() - 1;
    if d == 0 {
        return Ok(Vec::new());
    }
    let k: Basic = index(depth, avoid);
    let mut groups: Vec<RootGroup> = Vec::new();
    let pairs: Vec<(Dense, Dense)> = divisor_pairs(field, &polys[0], &shift(&polys[d], 1 - d as i64, field))?;
    for (pair_index, (a, b)) in pairs.into_iter().enumerate() {
        let p: Vec<Dense> = (0..=d)
            .map(|i| {
                dup_mul(
                    &dup_mul(&polys[i], &products(&a, 0, i as i64, field), field),
                    &products(&b, i as i64, d as i64, field),
                    field,
                )
            })
            .collect();
        let m: isize = p.iter().map(|x| dup_degree(x)).max().unwrap_or(-1);
        let leading_terms: BTreeMap<usize, Elem> = p
            .iter()
            .enumerate()
            .filter(|(_, x)| dup_degree(x) == m)
            .map(|(i, x)| (i, dup_LC(x, field)))
            .collect();
        let leading: Dense = dup_from_dict(&leading_terms, field);
        let start: i64 = start_of(field, &a, &b)?;
        let mut products_expr: Option<Basic> = None;
        for (root_index, (expr, value, factor)) in nonzero_roots(field, &leading)?.into_iter().enumerate() {
            if products_expr.is_none() {
                products_expr = Some(div(
                    &product_expression(field, &a, n, start, &k)?,
                    &product_expression(field, &b, n, start, &k)?,
                )?);
            }
            let position: usize = match groups.iter().position(|g| g.expr == expr) {
                Some(position) => position,
                None => {
                    groups.push(RootGroup {
                        expr: expr.clone(),
                        value,
                        factor,
                        candidates: Vec::new(),
                    });
                    groups.len() - 1
                }
            };
            groups[position].candidates.push(Candidate {
                key: (pair_index, root_index),
                a: a.clone(),
                b: b.clone(),
                p: p.clone(),
                start,
                products: products_expr.clone().expect("computed above"),
            });
        }
    }
    let mut found: Vec<((usize, usize, usize), Hyper)> = Vec::new();
    for group in &groups {
        if let Some(value) = &group.value {
            found.extend(classes_at(field, field, &|c: &Elem| c.clone(), polys, value, group, n)?);
        } else {
            let extended: Field = field.extend(&group.factor, &group.expr)?;
            let z: Elem = extended.from_expr(&group.expr)?;
            found.extend(classes_at(
                field,
                &extended,
                &|c: &Elem| field.embed(c),
                polys,
                &z,
                group,
                n,
            )?);
        }
    }
    found.sort_by_key(|item| item.0);
    Ok(found.into_iter().map(|(_, record)| record).collect())
}

/// The hypergeometric solutions `c(n) z**n prod a/b` for the root `z` of
/// `g` (`f` itself, or `f` with `z` adjoined), grouped into similarity
/// classes, with the keys of their classes (`_classes_at`).
fn classes_at(
    _f: &Field,
    g: &Field,
    embed: &dyn Fn(&Elem) -> Elem,
    polys: &[Dense],
    z: &Elem,
    group: &RootGroup,
    n: &Basic,
) -> PolyResult<Vec<((usize, usize, usize), Hyper)>> {
    let mapped = |p: &[Elem]| -> Dense {
        let image: Vec<Elem> = p.iter().map(embed).collect();
        dup_strip(&image, g)
    };
    let operator: Vec<Dense> = polys.iter().map(|p| mapped(p)).collect();
    let mut classes: Vec<Class> = Vec::new();
    for candidate in &group.candidates {
        let (a, b) = (mapped(&candidate.a), mapped(&candidate.b));
        let scaled: Vec<Dense> = candidate
            .p
            .iter()
            .enumerate()
            .map(|(i, p)| dup_mul_ground(&mapped(p), &g.pow(z, i as u64), g))
            .collect();
        let aux: Vec<Dense> = normalized(g, &scaled);
        if aux.is_empty() {
            continue;
        }
        let (core_num, core_den) = cancelled(&dup_mul_ground(&a, z, g), &b, g);
        for (c_index, c) in polynomial_solutions(g, &aux)?.into_iter().enumerate() {
            let (num, den) = cancelled(&dup_mul(&core_num, &shift(&c, 1, g), g), &dup_mul(&core_den, &c, g), g);
            let key = (candidate.key.0, candidate.key.1, c_index);
            classify(g, &mut classes, key, &num, &den, &core_num, &core_den, &c, candidate)?;
        }
    }
    let mut records: Vec<((usize, usize, usize), Hyper)> = Vec::new();
    for cls in &classes {
        for member in polynomial_basis(g, &cls.polynomials) {
            let (num, den) = cancelled(
                &dup_mul(&cls.num, &shift(&member, 1, g), g),
                &dup_mul(&cls.den, &member, g),
                g,
            );
            let expr: Basic = simplified(&mul(&[
                g.expression(&member, n)?,
                pow(&group.expr, n)?,
                cls.products.clone(),
            ])?)?;
            records.push((
                cls.key,
                Hyper {
                    expr,
                    start: cls.start,
                    field: g.clone(),
                    polys: operator.clone(),
                    num,
                    den,
                },
            ));
        }
    }
    Ok(records)
}

/// Put the term of ratio `num/den` (the core `core_num/core_den` times
/// `c(n + 1)/c(n)`) into its similarity class (`_classify`).
#[allow(clippy::too_many_arguments)]
fn classify(
    g: &Field,
    classes: &mut Vec<Class>,
    key: (usize, usize, usize),
    num: &[Elem],
    den: &[Elem],
    core_num: &[Elem],
    core_den: &[Elem],
    c: &[Elem],
    candidate: &Candidate,
) -> PolyResult<()> {
    for cls in classes.iter_mut() {
        let (q_num, q_den) = cancelled(&dup_mul(num, &cls.den, g), &dup_mul(den, &cls.num, g), g);
        if let Some(quotient) = shift_quotient(g, &q_num, &q_den)? {
            cls.polynomials.push(quotient);
            return Ok(());
        }
    }
    classes.push(Class {
        key,
        num: core_num.to_vec(),
        den: core_den.to_vec(),
        start: candidate.start,
        products: candidate.products.clone(),
        polynomials: vec![monic(c, g)],
    });
    Ok(())
}

/// A basis of the span of the polynomials (the rows of the reduced
/// echelon form of their coefficients) (`_polynomial_basis`).
fn polynomial_basis(g: &Field, polynomials: &[Dense]) -> Vec<Dense> {
    if polynomials.is_empty() {
        return Vec::new();
    }
    let degree: i64 = polynomials.iter().map(|p| dup_degree(p) as i64).max().unwrap_or(0);
    let rows: Vec<SparseRow<Elem>> = polynomials
        .iter()
        .map(|p| {
            (0..=degree)
                .filter_map(|j| {
                    let c: Elem = coeff(p, j, g);
                    if g.is_zero(&c) {
                        None
                    } else {
                        Some((j as usize, c))
                    }
                })
                .collect::<SparseRow<Elem>>()
        })
        .filter(|row| !row.is_empty())
        .collect();
    g.sparse_echelon(&rows)
        .iter()
        .map(|row| dup_from_dict(row, g))
        .collect()
}

/// The roots of a polynomial found by SymPy's `roots` (ported), in its
/// order; none when it fails.
fn listed_roots(field: &Field, f: &[Elem]) -> PolyResult<Vec<Basic>> {
    let x: Basic = rx();
    let expression: Basic = field.expression(f, &x)?;
    match crate::polys::polyroots::roots(&expression, std::slice::from_ref(&x), true, true, None) {
        Ok(found) => Ok(found.keys().cloned().collect()),
        Err(PolyError::Polys(..)) | Err(PolyError::Value(_)) => Ok(Vec::new()),
        Err(e) if genuine_not_implemented(&e) => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

type RootEntry = (Basic, Option<Elem>, Dense);

/// The nonzero roots of a polynomial in `z`: the expression of each, with
/// its value when it is in the field, or the irreducible factor of which
/// it is a root otherwise; in the order of SymPy's `roots` when it finds
/// them all (`_nonzero_roots`).
fn nonzero_roots(field: &Field, leading: &[Elem]) -> PolyResult<Vec<RootEntry>> {
    let mut entries: Vec<RootEntry> = Vec::new();
    for (f, _) in field.factor(leading)? {
        if dup_degree(&f) == 1 {
            let value: Elem = field.neg(&f[1]);
            if !field.is_zero(&value) {
                entries.push((field.to_expr(&value)?, Some(value), f));
            }
        } else {
            for root in irreducible_roots(field, &f)? {
                entries.push((root, None, f.clone()));
            }
        }
    }
    let order: Vec<Basic> = listed_roots(field, leading)?;
    let position = |e: &Basic| order.iter().position(|o| o == e).unwrap_or(order.len());
    entries.sort_by_key(|entry| position(&entry.0));
    Ok(entries)
}

/// The roots of the monic irreducible `f` of degree two or more, as
/// expressions: SymPy's `roots` (ported), the formula of the quadratic
/// otherwise (`_irreducible_roots`).
fn irreducible_roots(field: &Field, f: &[Elem]) -> PolyResult<Vec<Basic>> {
    let values: Vec<Basic> = listed_roots(field, f)?;
    if !values.is_empty() || dup_degree(f) != 2 {
        return Ok(values);
    }
    let (p, qc) = (&f[1], &f[2]);
    let discriminant: Basic = field.to_expr(&field.sub(&field.mul(p, p), &field.mul(&field.from_int(4), qc)))?;
    let half: Basic = div(&neg(&field.to_expr(p)?)?, &int(2))?;
    let r: Basic = div(&pow(&discriminant, &Basic::half())?, &int(2))?;
    Ok(vec![sub(&half, &r)?, add(&[half, r])?])
}

/// The symbols the summation indices must not be (`_avoided`).
fn avoided(exprs: &[Basic], n: &Basic) -> BTreeSet<Basic> {
    let mut result: BTreeSet<Basic> = BTreeSet::new();
    result.insert(n.clone());
    for e in exprs {
        result.extend(e.free_symbols());
    }
    result
}

/// A basis of the hypergeometric solutions of
/// `sum coefficients[i] * y(n + i) = 0` (`rsolve_hypergeometric`).
pub fn rsolve_hypergeometric(coefficients: &[Basic], n: &Basic) -> PolyResult<Vec<Basic>> {
    let context: Context = context(coefficients, n)?;
    let operator: Operator = context.operator(coefficients, n, true)?;
    if operator.order() < 0 {
        return Err(PolyError::Value("the recurrence is trivial".to_string()));
    }
    let avoid: BTreeSet<Basic> = avoided(coefficients, n);
    let mut result: Vec<Basic> = Vec::new();
    for (expr, start) in operator.hypergeometric_solutions(&avoid)? {
        if operator.checked(&expr, &Basic::zero(), start, 6)? {
            result.push(context.restored(&expr)?);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// d'Alembertian solutions

fn add_fractions(a: &(Dense, Dense), b: &(Dense, Dense), field: &Field) -> (Dense, Dense) {
    cancelled(
        &dup_add(&dup_mul(&a.0, &b.1, field), &dup_mul(&b.0, &a.1, field), field),
        &dup_mul(&a.1, &b.1, field),
        field,
    )
}

/// The operator `L'` of order `d - 1` with `L[h sum z] = h L'[z]` for the
/// hypergeometric solution `h` of ratio `num/den` (`_reduced_operator`).
fn reduced_operator(field: &Field, polys: &[Dense], num: &[Elem], den: &[Elem]) -> Vec<Dense> {
    let d: usize = polys.len() - 1;
    let one: Dense = dup_one(field);
    let mut rho: Vec<(Dense, Dense)> = vec![(one.clone(), one.clone())];
    for i in 1..=d {
        let previous: &(Dense, Dense) = rho.last().expect("nonempty");
        let next = cancelled(
            &dup_mul(&previous.0, &shift(num, i as i64 - 1, field), field),
            &dup_mul(&previous.1, &shift(den, i as i64 - 1, field), field),
            field,
        );
        rho.push(next);
    }
    let mut fractions: Vec<(Dense, Dense)> = Vec::new();
    for j in 0..d {
        let mut total: (Dense, Dense) = (Vec::new(), one.clone());
        for i in j + 1..=d {
            total = add_fractions(&total, &(dup_mul(&polys[i], &rho[i].0, field), rho[i].1.clone()), field);
        }
        fractions.push(total);
    }
    let mut common: Dense = one;
    for (_, qd) in &fractions {
        common = dup_lcm(&common, qd, field);
    }
    let mut reduced: Vec<Dense> = fractions
        .iter()
        .map(|(p, qd)| dup_mul(p, &quo(&common, qd, field), field))
        .collect();
    while reduced.last().is_some_and(|p| p.is_empty()) {
        reduced.pop();
    }
    normalized(field, &reduced)
}

fn dalembertian_records(
    field: &Field,
    polys: &[Dense],
    n: &Basic,
    depth: usize,
    avoid: &BTreeSet<Basic>,
) -> PolyResult<Vec<Solution>> {
    let hyper: Vec<Hyper> = hypergeometric_records(field, polys, n, depth, avoid)?;
    if hyper.is_empty() {
        return Ok(Vec::new());
    }
    let mut records: Vec<Solution> = hyper.iter().map(|h| (h.expr.clone(), h.start)).collect();
    let order: usize = polys.len() - 1;
    if order <= 1 || hyper.len() >= order {
        return Ok(records);
    }
    // the hypergeometric solutions come first; the sums h sum z add the
    // directions the hypergeometric ones do not span (by the first solution
    // whose reduced operator is over a field with a factorisation)
    let k: Basic = index(depth, avoid);
    for h in &hyper {
        let reduced: Vec<Solution> = match h.reduced_solutions(n, depth + 1, avoid) {
            Ok(reduced) => reduced,
            Err(e) if genuine_not_implemented(&e) => continue,
            Err(e) => return Err(e),
        };
        for (z, z_start) in reduced {
            let lower: i64 = h.start.max(z_start);
            let sum: Basic = crate::concrete::expr_with_limits::sum_(
                &subs(&z, n, &k)?,
                &[Basic::new(Head::Tuple, vec![k.clone(), int(lower), sub(n, &int(1))?])],
            )?;
            records.push((mul(&[h.expr.clone(), sum])?, lower));
        }
        break;
    }
    independent(field, &records, polys, n)
}

/// A maximal independent subset of the solutions, the earlier ones
/// preferred (`_independent`).
fn independent(field: &Field, records: &[Solution], polys: &[Dense], n: &Basic) -> PolyResult<Vec<Solution>> {
    let d: usize = polys.len() - 1;
    let leading: &Dense = &polys[d];
    let roots: Vec<i64> = if dup_degree(leading) > 0 {
        integer_roots(field, leading)?
    } else {
        Vec::new()
    };
    let start: i64 = records
        .iter()
        .map(|r| r.1)
        .chain(roots)
        .chain(std::iter::once(-1))
        .max()
        .unwrap_or(-1)
        + 1;
    let mut chosen: Vec<Solution> = Vec::new();
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for record in records {
        if chosen.len() >= d {
            break;
        }
        let values: Vec<Basic> = (start..start + d as i64)
            .map(|at| value(&record.0, n, at))
            .collect::<PolyResult<_>>()?;
        if values.iter().any(not_finite) {
            continue;
        }
        let mut extended: Vec<Vec<Basic>> = rows.clone();
        extended.push(values.clone());
        if rank(&extended)? > rows.len() {
            rows.push(values);
            chosen.push(record.clone());
        }
    }
    Ok(chosen)
}

/// A basis of the d'Alembertian solutions of
/// `sum coefficients[i] * y(n + i) = 0`: the nested indefinite sums of
/// hypergeometric terms (`rsolve_dalembertian`).
pub fn rsolve_dalembertian(coefficients: &[Basic], n: &Basic) -> PolyResult<Vec<Basic>> {
    let context: Context = context(coefficients, n)?;
    let operator: Operator = context.operator(coefficients, n, true)?;
    if operator.order() < 0 {
        return Err(PolyError::Value("the recurrence is trivial".to_string()));
    }
    let avoid: BTreeSet<Basic> = avoided(coefficients, n);
    let mut result: Vec<Basic> = Vec::new();
    for (expr, start) in operator.dalembertian_solutions(&avoid)? {
        if operator.checked(&expr, &Basic::zero(), start, 6)? {
            result.push(context.restored(&closed_form(&expr)?)?);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// verification and linear algebra on expressions

/// `expr` at `n = at`, the sums and products with integer limits written
/// out (`_value`).
fn value(expr: &Basic, n: &Basic, at: i64) -> R {
    evaluated(&subs(expr, n, &int(at))?)
}

fn evaluated(expr: &Basic) -> R {
    if !has_sum_or_product(expr) {
        if has_function(expr, &[Function::RisingFactorial, Function::Factorial]) {
            return doit(expr);
        }
        return Ok(expr.clone());
    }
    if matches!(expr.head(), Head::Sum | Head::Product) {
        let is_sum: bool = matches!(expr.head(), Head::Sum);
        let mut function: Basic = expr.args()[0].clone();
        for (index, lo, hi) in crate::concrete::expr_with_limits::limits(expr) {
            let (Some(lo), Some(hi)) = (
                integer_of(&lo).filter(|_| is_integer_node(&lo)),
                integer_of(&hi).filter(|_| is_integer_node(&hi)),
            ) else {
                return doit(expr);
            };
            let values: Vec<Basic> = (lo..=hi)
                .map(|i| evaluated(&subs(&function, &index, &int(i))?))
                .collect::<PolyResult<_>>()?;
            function = if is_sum { add(&values)? } else { mul(&values)? };
        }
        return Ok(function);
    }
    if expr.args().is_empty() {
        return Ok(expr.clone());
    }
    let args: Vec<Basic> = expr
        .args()
        .iter()
        .map(|a| if is_expr(a) { evaluated(a) } else { Ok(a.clone()) })
        .collect::<PolyResult<_>>()?;
    let rebuilt: Basic = crate::core::function::rebuild_evaluated(expr, args)?;
    if !is_expr(&rebuilt) {
        return Err(PolyError::Type(format!("{} is not an expression", rebuilt)));
    }
    doit(&rebuilt)
}

/// `e` expanded and cancelled (`_normal`).
fn normal(e: &Basic) -> R {
    let expanded: Basic = expand(e)?;
    match crate::polys::polytools::cancel(&expanded, &[]) {
        Ok(c) => Ok(c),
        Err(PolyError::Polys(..)) | Err(PolyError::Value(_)) | Err(PolyError::Type(_)) => Ok(expanded),
        Err(err) if genuine_not_implemented(&err) => Ok(expanded),
        Err(err) => Err(err),
    }
}

fn soft_failure(e: &PolyError) -> bool {
    matches!(e, PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_)) || genuine_not_implemented(e)
}

/// Whether `value` is zero: exactly for the rational functions and the
/// algebraic numbers (by their minimal polynomial), numerically at 30
/// digits for the other numbers (`_is_zero`).
fn is_zero(value: &Basic) -> PolyResult<bool> {
    if *value == Basic::zero() {
        return Ok(true);
    }
    let normal_form: Basic = normal(value)?;
    if normal_form == Basic::zero() {
        return Ok(true);
    }
    if !normal_form.free_symbols().is_empty() {
        let options = crate::simplify::simplify::Options::default();
        return match crate::simplify::simplify::simplify(&normal_form, &options) {
            Ok(s) => Ok(s == Basic::zero()),
            Err(e) if soft_failure(&e) => Ok(false),
            Err(e) => Err(e),
        };
    }
    if is_rational_node(&normal_form) {
        return Ok(false);
    }
    // a number which is not small at 120 digits is not zero: the minimal
    // polynomial is only asked to prove the zeros
    let approximation = crate::core::evalf::numeric_value(&normal_form, 120);
    if let Some(a) = &approximation {
        let (re_part, im_part) = crate::core::evalf::parts(a);
        if re_part.to_f64().value().hypot(im_part.to_f64().value()) > 1e-20 {
            return Ok(false);
        }
    }
    match crate::polys::numberfields::minpoly::minimal_polynomial_coeffs(&normal_form) {
        Ok(coefficients) => return Ok(coefficients == vec![Int::ONE, Int::ZERO]),
        Err(e) if soft_failure(&e) => {}
        Err(e) => return Err(e),
    }
    Ok(approximation.is_some())
}

/// The reduced row echelon form of a matrix of expressions and its pivot
/// columns, the pivots decided by [`is_zero`] (`_row_reduce`).
pub(crate) fn row_reduce(rows: &[Vec<Basic>]) -> PolyResult<(Vec<Vec<Basic>>, Vec<usize>)> {
    let mut matrix: Vec<Vec<Basic>> = rows
        .iter()
        .map(|row| row.iter().map(normal).collect::<PolyResult<_>>())
        .collect::<PolyResult<_>>()?;
    let mut pivots: Vec<usize> = Vec::new();
    let columns: usize = matrix.first().map(|r| r.len()).unwrap_or(0);
    let mut r: usize = 0;
    for c in 0..columns {
        if r == matrix.len() {
            break;
        }
        let mut p: Option<usize> = None;
        for (i, row) in matrix.iter().enumerate().skip(r) {
            if !is_zero(&row[c])? {
                p = Some(i);
                break;
            }
        }
        let Some(p) = p else {
            continue;
        };
        matrix.swap(r, p);
        let pivot: Basic = matrix[r][c].clone();
        matrix[r] = matrix[r]
            .iter()
            .enumerate()
            .map(|(j, e)| {
                if j == c {
                    Ok(Basic::one())
                } else {
                    normal(&div(e, &pivot)?)
                }
            })
            .collect::<PolyResult<_>>()?;
        for i in 0..matrix.len() {
            if i != r && !is_zero(&matrix[i][c])? {
                let factor: Basic = matrix[i][c].clone();
                let pivot_row: Vec<Basic> = matrix[r].clone();
                matrix[i] = matrix[i]
                    .iter()
                    .enumerate()
                    .map(|(j, e)| {
                        if j == c {
                            Ok(Basic::zero())
                        } else {
                            normal(&sub(e, &mul(&[factor.clone(), pivot_row[j].clone()])?)?)
                        }
                    })
                    .collect::<PolyResult<_>>()?;
            }
        }
        pivots.push(c);
        r += 1;
    }
    Ok((matrix, pivots))
}

fn rank(rows: &[Vec<Basic>]) -> PolyResult<usize> {
    Ok(row_reduce(rows)?.1.len())
}

/// A solution of `rows * x = rhs` (the free unknowns zero), `None` when
/// there is none (`_solve_linear`).
fn solve_linear(rows: &[Vec<Basic>], rhs: &[Basic], columns: usize) -> PolyResult<Option<Vec<Basic>>> {
    if rows.is_empty() {
        return Ok(Some(vec![Basic::zero(); columns]));
    }
    let augmented: Vec<Vec<Basic>> = rows
        .iter()
        .zip(rhs)
        .map(|(row, v)| {
            let mut row: Vec<Basic> = row.clone();
            row.push(v.clone());
            row
        })
        .collect();
    let (reduced, pivots) = row_reduce(&augmented)?;
    if pivots.contains(&columns) {
        return Ok(None);
    }
    let mut solution: Vec<Basic> = vec![Basic::zero(); columns];
    for (row, column) in pivots.iter().enumerate() {
        solution[*column] = reduced[row][columns].clone();
    }
    Ok(Some(solution))
}

/// Whether `solution` satisfies `sum coefficients[i] y(n + i) = rhs` at
/// `n = start, ..., start + count - 1`, exactly (`is_recurrence_solution`).
pub fn is_recurrence_solution(
    coefficients: &[Basic],
    n: &Basic,
    solution: &Basic,
    rhs: &Basic,
    start: i64,
    count: i64,
) -> PolyResult<bool> {
    let context: Context = context(coefficients, n)?;
    let operator: Operator = context.operator(coefficients, n, false)?;
    operator.checked(&context.rename(solution)?, &context.rename(rhs)?, start, count)
}

fn checked(
    coefficients: &[Basic],
    n: &Basic,
    solution: &Basic,
    rhs: &Basic,
    start: i64,
    count: i64,
) -> PolyResult<bool> {
    for at in start..start + count {
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in coefficients.iter().enumerate() {
            terms.push(mul(&[value(c, n, at)?, value(solution, n, at + i as i64)?])?);
        }
        let total: Basic = sub(&add(&terms)?, &value(rhs, n, at)?)?;
        if not_finite(&total) || !is_zero(&total)? {
            return Ok(false);
        }
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// equations

fn same_function(t: &Basic, y: &Basic) -> bool {
    match (t.head(), y.head()) {
        (Head::Undefined(a), Head::Undefined(b)) => a == b,
        _ => false,
    }
}

fn applications(e: &Basic, y: &Basic) -> Vec<Basic> {
    e.atoms(Some(&|t: &Basic| same_function(t, y)))
}

/// `(coefficients, rhs, n)` of a linear recurrence written with
/// `y(n + i)`, the index shifted so that the least shift is zero
/// (`_parse`).
fn parse(equation: &Basic, y: &Basic) -> PolyResult<(Vec<Basic>, Basic, Basic)> {
    let argument: Option<&Basic> = if y.args().len() == 1 { Some(&y.args()[0]) } else { None };
    let Some(n) = argument.filter(|a| a.is_symbol()).cloned() else {
        return Err(PolyError::Value(format!("{} is not a function of one symbol", y)));
    };
    let expr: Basic = if matches!(equation.head(), Head::Relational(crate::core::basic::RelOp::Eq)) {
        sub(&equation.args()[0], &equation.args()[1])?
    } else {
        equation.clone()
    };
    let terms: Vec<Basic> = applications(&expr, y);
    if terms.is_empty() {
        return Err(PolyError::Value(format!("no {} in the equation", y)));
    }
    let mut shifts: Vec<i64> = Vec::new();
    for t in &terms {
        let offset: Option<Basic> = if t.args().len() == 1 && is_expr(&t.args()[0]) {
            Some(sub(&t.args()[0], &n)?)
        } else {
            None
        };
        match offset.as_ref().filter(|o| is_integer_node(o)).and_then(integer_of) {
            Some(v) => shifts.push(v),
            None => return Err(PolyError::Value(format!("{} is not a shift of {} by an integer", t, y))),
        }
    }
    let least: i64 = *shifts.iter().min().expect("nonempty");
    let shifted: Basic = subs(&expr, &n, &sub(&n, &int(least))?)?;
    let order: i64 = shifts.iter().max().expect("nonempty") - least;
    let together: Basic = crate::polys::rationaltools::together(&shifted, false, true)?;
    let (numerator, _) = crate::core::expr::as_numer_denom(&together)?;
    let (mut coefficients, mut rhs) = linear_parts(&numerator, y, &n, order)?;
    let rational = |c: &Basic| {
        crate::core::expr::polynomial_like(c, std::slice::from_ref(&n), crate::core::expr::PolynomialKind::Rational)
            == Some(true)
    };
    if !coefficients.iter().all(rational) {
        // the denominator of the right-hand side is no polynomial in n
        // (factorials): the equation is not multiplied by it
        (coefficients, rhs) = linear_parts(&shifted, y, &n, order)?;
    }
    Ok((coefficients, rhs, n))
}

/// `(coefficients, rhs)` of `expr = sum coefficients[i] y(n + i) - rhs`
/// (`_linear_parts`).
fn linear_parts(expr: &Basic, y: &Basic, n: &Basic, order: i64) -> PolyResult<(Vec<Basic>, Basic)> {
    let Head::Undefined(function) = y.head() else {
        return Err(PolyError::Value(format!("{} is not an undefined function", y)));
    };
    let unknowns: Vec<Basic> = (0..=order)
        .map(|i| crate::core::symbol::Dummy::new(&format!("y{}", i), &[]).0)
        .collect();
    let mut mapping: Vec<(Basic, Basic)> = Vec::new();
    for (i, u) in unknowns.iter().enumerate() {
        let point: Basic = add(&[n.clone(), int(i as i64)])?;
        mapping.push((function.apply(vec![point])?, u.clone()));
    }
    let expr: Basic = expand(&crate::core::subs::subs_mapping(expr, &mapping)?)?;
    if !applications(&expr, y).is_empty() {
        return Err(PolyError::Value(format!("the equation is not linear in {}", y)));
    }
    let mut coefficients: Vec<Basic> = vec![Basic::zero(); order as usize + 1];
    let mut rhs: Basic = Basic::zero();
    for term in terms_of(&expr) {
        let (mut independent, mut dependent): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
        for factor in factors_of(&term) {
            if unknowns.iter().any(|u| factor.has(u)) {
                dependent.push(factor);
            } else {
                independent.push(factor);
            }
        }
        let coefficient: Basic = mul(&independent)?;
        let dependent: Basic = mul(&dependent)?;
        if dependent == Basic::one() {
            rhs = sub(&rhs, &coefficient)?;
        } else if let Some(i) = unknowns.iter().position(|u| *u == dependent) {
            coefficients[i] = add(&[coefficients[i].clone(), coefficient])?;
        } else {
            return Err(PolyError::Value(format!("the equation is not linear in {}", y)));
        }
    }
    Ok((coefficients, rhs))
}

/// `f(n + 1)/f(n)` when it is a rational function of `n`
/// (`_hypergeometric_ratio`).
fn hypergeometric_ratio(f: &Basic, n: &Basic) -> PolyResult<Option<Basic>> {
    if *f == Basic::zero() {
        return Ok(None);
    }
    let Some(ratio) = crate::concrete::gosper::hypersimp(f, n)? else {
        return Ok(None);
    };
    let ratio: Basic = crate::polys::polytools::cancel(&ratio, &[])?;
    let rational: bool = crate::core::expr::polynomial_like(
        &ratio,
        std::slice::from_ref(n),
        crate::core::expr::PolynomialKind::Rational,
    ) == Some(true);
    if rational && !has_sum_or_product(&ratio) {
        return Ok(Some(ratio));
    }
    Ok(None)
}

/// The general solution of a linear recurrence with polynomial (or
/// rational) coefficients, written with `y(n + i)`, as a d'Alembertian
/// expression with the constants `C0, C1, ...`; `None` when the
/// d'Alembertian solutions do not span the solutions, or when the
/// right-hand side is not a sum of hypergeometric terms with a
/// d'Alembertian particular solution. With `init`, the values
/// `[(y(0), ...), (y(1), ...)]` determine the constants (`rsolve_linear`).
pub fn rsolve_linear(equation: &Basic, y: &Basic, init: Option<&[(Basic, Basic)]>) -> PolyResult<Option<Basic>> {
    let (coefficients, rhs, n) = parse(equation, y)?;
    let terms: Vec<Basic> = if rhs == Basic::zero() {
        Vec::new()
    } else {
        terms_of(&expand(&rhs)?)
    };
    let ratios: Vec<Option<Basic>> = terms
        .iter()
        .map(|t| hypergeometric_ratio(t, &n))
        .collect::<PolyResult<_>>()?;
    let mut exprs: Vec<Basic> = coefficients.clone();
    exprs.extend(ratios.iter().flatten().cloned());
    let context: Context = context(&exprs, &n)?;
    let operator: Operator = context.operator(&coefficients, &n, true)?;
    if operator.order() < 0 {
        return Err(PolyError::Value("the recurrence is trivial".to_string()));
    }
    let order: usize = operator.order() as usize;
    let mut with_rhs: Vec<Basic> = coefficients.clone();
    with_rhs.push(rhs.clone());
    let avoid: BTreeSet<Basic> = avoided(&with_rhs, &n);
    let mut homogeneous: Vec<Basic> = Vec::new();
    for (expr, start) in operator.dalembertian_solutions(&avoid)? {
        if operator.checked(&expr, &Basic::zero(), start, 6)? {
            homogeneous.push(closed_form(&expr)?);
        }
    }
    if homogeneous.len() < order {
        return Ok(None);
    }
    let mut particular: Basic = Basic::zero();
    for (term, ratio) in terms.iter().zip(&ratios) {
        let Some(ratio) = ratio else {
            return Ok(None);
        };
        let Some(found) = operator.particular(&context.rename(term)?, &context.rename(ratio)?, &avoid)? else {
            return Ok(None);
        };
        particular = add(&[particular, closed_form(&found.0)?])?;
    }
    let constants: Vec<Basic> = (0..order).map(|i| symbol(&format!("C{}", i))).collect();
    let mut parts: Vec<Basic> = vec![particular];
    for (c, h) in constants.iter().zip(&homogeneous) {
        parts.push(mul(&[c.clone(), h.clone()])?);
    }
    let general: Basic = context.restored(&add(&parts)?)?;
    let Some(init) = init else {
        return Ok(Some(general));
    };
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    let mut values: Vec<Basic> = Vec::new();
    let zeros: Vec<(Basic, Basic)> = constants.iter().map(|c| (c.clone(), Basic::zero())).collect();
    for (point, given) in init {
        let at: Option<i64> = if point.args().len() == 1 && is_integer_node(&point.args()[0]) {
            integer_of(&point.args()[0])
        } else {
            None
        };
        let Some(at) = at else {
            return Err(PolyError::Value(format!(
                "the initial values must be given at integers: {}",
                point
            )));
        };
        if given.has(&n) {
            // a value depending on the index is not a constant of the
            // solution: substituting it gives a sequence which does not
            // satisfy the recurrence
            return Err(PolyError::Value(format!(
                "the initial value of {} depends on {}: {}",
                point, n, given
            )));
        }
        let evaluated_general: Basic = value(&general, &n, at)?;
        rows.push(
            constants
                .iter()
                .map(|c| crate::core::function::diff(&evaluated_general, &[(c.clone(), 1)]))
                .collect::<PolyResult<_>>()?,
        );
        values.push(sub(
            given,
            &crate::core::subs::subs_mapping(&evaluated_general, &zeros)?,
        )?);
    }
    let Some(solution) = solve_linear(&rows, &values, constants.len())? else {
        return Err(PolyError::Value(
            "the initial values are inconsistent with the recurrence".to_string(),
        ));
    };
    let assignment: Vec<(Basic, Basic)> = constants.into_iter().zip(solution).collect();
    Ok(Some(crate::core::subs::subs_mapping(&general, &assignment)?))
}
