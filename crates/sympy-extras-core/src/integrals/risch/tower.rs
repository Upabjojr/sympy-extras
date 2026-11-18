//! The arithmetic of the differential fields of the Risch algorithm,
//! ported from `sympy_extras.integrals.risch.tower`. This part:
//! [`RationalFunction`], an element of `C(x, t1, ..., tn)`: a fraction of
//! polynomials over `ZZ` in its generators (the symbols, the kernels such
//! as `exp(x)` or `pi`, and the algebraic constants `I` and the square
//! roots of primes), in canonical form: the powers of the algebraic
//! generators reduced, the denominator free of them (multiplied by its
//! conjugates), numerator and denominator coprime over `ZZ`, the leading
//! coefficient of the denominator positive, the generators in SymPy's
//! order. Other constants raise `UnsupportedConstant` (a
//! `NotImplementedError`).

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::basic::{Basic, Head};
use crate::error::{zero_division, PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::domains::{IntegerRing, SparsePolynomialRing, ZZ};
use crate::polys::rings::{Monomial, PolyElement, PolyRing, Terms};

/// A polynomial over `ZZ` in the generators.
pub type ZPoly = PolyElement<IntegerRing>;

/// An unsupported constant (`UnsupportedConstant`, a
/// `NotImplementedError`).
pub fn unsupported(message: String) -> PolyError {
    PolyError::NotImplemented(message)
}

thread_local! {
    static RINGS: RefCell<HashMap<usize, SparsePolynomialRing<IntegerRing>>> = RefCell::new(HashMap::new());
}

/// `ZZ[g0, ..., g(n-1)]`, one variable for the constants (`_ring`).
pub fn ring(n: usize) -> SparsePolynomialRing<IntegerRing> {
    let width: usize = n.max(1);
    RINGS.with(|rings| {
        rings
            .borrow_mut()
            .entry(width)
            .or_insert_with(|| SparsePolynomialRing::sympy(ZZ, (0..width).map(|i| format!("g{}", i)).collect()))
            .clone()
    })
}

fn poly_ring(n: usize) -> Arc<PolyRing<IntegerRing>> {
    ring(n).ring
}

type GenKey = (i64, String, Int, i64);

fn key(g: &Basic) -> GenKey {
    crate::polys::polytools::gen_key(g)
}

/// The square of an algebraic generator: `-1` for `I`, `p` for `sqrt(p)`
/// (`_square`).
pub fn square(g: &Basic) -> Option<Int> {
    if matches!(g.head(), Head::ImaginaryUnit) {
        return Some(Int::NEG_ONE);
    }
    if g.is_pow() {
        let (base, exponent) = (&g.args()[0], &g.args()[1]);
        if matches!(
            base.head(),
            Head::Rational {
                integer_class: true,
                ..
            }
        ) && exponent.as_rational() == Some(&Rat::from_parts_signed(Int::ONE, Int::from(2)))
        {
            return base.as_rational().map(|v| v.numerator().clone());
        }
    }
    None
}

/// The algebraic generators of the generators: `(index, square)`
/// (`_algebraic`).
fn algebraic(gens: &[Basic]) -> Vec<(usize, Int)> {
    gens.iter()
        .enumerate()
        .filter_map(|(i, g)| square(g).map(|s| (i, s)))
        .collect()
}

/// The union of two lists of generators, sorted by their keys (`_merge`).
fn merge(a: &[Basic], b: &[Basic]) -> Vec<Basic> {
    if a == b || b.is_empty() {
        return a.to_vec();
    }
    if a.is_empty() {
        return b.to_vec();
    }
    let extra: Vec<Basic> = b.iter().filter(|g| !a.contains(g)).cloned().collect();
    if extra.is_empty() {
        return a.to_vec();
    }
    let mut all: Vec<Basic> = a.to_vec();
    all.extend(extra);
    all.sort_by_cached_key(key);
    all
}

/// `f` in the generators `old` as a polynomial in `new` (`_remap`).
fn remap(f: &ZPoly, old: &[Basic], new: &[Basic]) -> ZPoly {
    let r = poly_ring(new.len());
    if old == new && Arc::ptr_eq(&f.ring, &r) {
        return f.clone();
    }
    let width: usize = new.len().max(1);
    if old.is_empty() {
        let mut terms: Terms<Int> = Terms::new();
        for c in f.terms.values() {
            terms.insert(vec![0; width], c.clone());
        }
        return r.from_dict(terms);
    }
    let positions: Vec<usize> = old
        .iter()
        .map(|g| new.iter().position(|h| h == g).expect("a superset"))
        .collect();
    let mut terms: Terms<Int> = Terms::new();
    for (m, c) in &f.terms {
        let mut e: Monomial = vec![0; width];
        for (i, k) in positions.iter().zip(m) {
            e[*i] = *k;
        }
        terms.insert(e, c.clone());
    }
    r.from_dict(terms)
}

/// The powers of the algebraic generators reduced, `g**2 = square`
/// (`_reduce`).
fn reduce(f: &ZPoly, algebraic: &[(usize, Int)]) -> ZPoly {
    if algebraic.is_empty() || f.terms.is_empty() {
        return f.clone();
    }
    if !f.terms.keys().any(|m| algebraic.iter().any(|(i, _)| m[*i] > 1)) {
        return f.clone();
    }
    let mut terms: Terms<Int> = Terms::new();
    for (m, c) in &f.terms {
        let mut e: Monomial = m.clone();
        let mut c: Int = c.clone();
        for (i, s) in algebraic {
            if e[*i] > 1 {
                c *= s.pow((e[*i] / 2) as usize);
                e[*i] %= 2;
            }
        }
        let value: Int = terms.get(&e).cloned().unwrap_or(Int::ZERO) + c;
        if value.is_zero() {
            terms.remove(&e);
        } else {
            terms.insert(e, value);
        }
    }
    f.ring.from_dict(terms)
}

/// `f` with the `i`-th generator negated (`_conjugate`).
fn conjugate(f: &ZPoly, i: usize) -> ZPoly {
    let terms: Terms<Int> = f
        .terms
        .iter()
        .map(|(m, c)| (m.clone(), if m[i] % 2 == 1 { -c.clone() } else { c.clone() }))
        .collect();
    f.ring.from_dict(terms)
}

fn used(f: &ZPoly, n: usize) -> Vec<bool> {
    let mut used: Vec<bool> = vec![false; n];
    for m in f.terms.keys() {
        for i in 0..n {
            if m[i] > 0 {
                used[i] = true;
            }
        }
    }
    used
}

fn project(f: &ZPoly, keep: &[usize], r: &Arc<PolyRing<IntegerRing>>) -> ZPoly {
    let width: usize = keep.len().max(1);
    if keep.is_empty() {
        let mut terms: Terms<Int> = Terms::new();
        for c in f.terms.values() {
            terms.insert(vec![0; width], c.clone());
        }
        return r.from_dict(terms);
    }
    r.from_dict(
        f.terms
            .iter()
            .map(|(m, c)| (keep.iter().map(|i| m[*i]).collect(), c.clone()))
            .collect(),
    )
}

fn content(f: &ZPoly) -> Int {
    let mut g: Int = Int::ZERO;
    for c in f.terms.values() {
        g = crate::core::intfunc::igcd(&[g, c.clone()]);
        if g.is_one() {
            break;
        }
    }
    g
}

/// An element `numer/denom` of the field of rational functions in its
/// generators over `QQ(I, sqrt(2), sqrt(3), ...)`, in canonical form
/// (`RationalFunction`).
#[derive(Debug, Clone)]
pub struct RationalFunction {
    pub gens: Vec<Basic>,
    pub numer: ZPoly,
    pub denom: ZPoly,
}

impl PartialEq for RationalFunction {
    fn eq(&self, other: &RationalFunction) -> bool {
        self.gens == other.gens && self.numer.terms == other.numer.terms && self.denom.terms == other.denom.terms
    }
}

impl Eq for RationalFunction {}

impl std::hash::Hash for RationalFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.gens.hash(state);
        self.numer.terms.hash(state);
        self.denom.terms.hash(state);
    }
}

impl std::fmt::Display for RationalFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_expr() {
            Ok(e) => write!(f, "{}", e),
            Err(_) => write!(f, "RationalFunction(?)"),
        }
    }
}

fn is_one_ground(p: &ZPoly) -> bool {
    p.is_ground() && p.LC().is_one()
}

impl RationalFunction {
    pub fn zero() -> RationalFunction {
        let r = poly_ring(0);
        RationalFunction {
            gens: Vec::new(),
            numer: r.zero(),
            denom: r.one(),
        }
    }

    pub fn one() -> RationalFunction {
        let r = poly_ring(0);
        RationalFunction {
            gens: Vec::new(),
            numer: r.one(),
            denom: r.one(),
        }
    }

    /// `I` as an element.
    pub fn imaginary_unit() -> RationalFunction {
        let r = poly_ring(1);
        RationalFunction {
            gens: vec![Basic::imaginary_unit()],
            numer: r.from_dict(Terms::from([(vec![1], Int::ONE)])),
            denom: r.one(),
        }
    }

    /// The canonical form of `numer/denom` (`_make`).
    pub fn make(gens: Vec<Basic>, numer: ZPoly, denom: ZPoly, cancel: bool) -> PolyResult<RationalFunction> {
        let sparse = ring(gens.len());
        if numer.terms.is_empty() {
            return Ok(RationalFunction::zero());
        }
        if denom.terms.is_empty() {
            return Err(zero_division("division by zero in a field of rational functions"));
        }
        let (mut numer, mut denom, mut cancel) = (numer, denom, cancel);
        let alg: Vec<(usize, Int)> = algebraic(&gens);
        if !alg.is_empty() {
            numer = reduce(&numer, &alg);
            denom = reduce(&denom, &alg);
            for (i, _) in &alg {
                if denom.terms.keys().any(|m| m[*i] > 0) {
                    let c: ZPoly = conjugate(&denom, *i);
                    numer = reduce(&numer.mul(&c), &alg);
                    denom = reduce(&denom.mul(&c), &alg);
                }
            }
            if numer.terms.is_empty() {
                return Ok(RationalFunction::zero());
            }
            cancel = true;
        }
        if cancel {
            if denom.terms.len() == 1 && denom.is_ground() {
                let d: Int = denom.LC();
                let mut g: Int = crate::core::intfunc::igcd(&[content(&numer), d.clone()]);
                if d < Int::ZERO {
                    g = -g;
                }
                if !g.is_one() {
                    numer = numer.exquo_ground(&g)?;
                    denom = denom.exquo_ground(&g)?;
                }
            } else {
                (numer, denom) = sparse.cancel(&numer, &denom);
            }
        }
        // prune the generators which do not appear
        let n: usize = gens.len();
        if n > 0 {
            let (used_n, used_d) = (used(&numer, n), used(&denom, n));
            let keep: Vec<usize> = (0..n).filter(|i| used_n[*i] || used_d[*i]).collect();
            if keep.len() < n {
                let s = poly_ring(keep.len());
                let new_gens: Vec<Basic> = keep.iter().map(|i| gens[*i].clone()).collect();
                return Ok(RationalFunction {
                    gens: new_gens,
                    numer: project(&numer, &keep, &s),
                    denom: project(&denom, &keep, &s),
                });
            }
        }
        Ok(RationalFunction { gens, numer, denom })
    }

    pub fn from_int(n: &Int) -> RationalFunction {
        if n.is_zero() {
            return RationalFunction::zero();
        }
        let r = poly_ring(0);
        RationalFunction {
            gens: Vec::new(),
            numer: r.ground(n.clone()),
            denom: r.one(),
        }
    }

    pub fn from_fraction(q: &Rat) -> RationalFunction {
        if q.is_zero() {
            return RationalFunction::zero();
        }
        let r = poly_ring(0);
        RationalFunction {
            gens: Vec::new(),
            numer: r.ground(q.numerator().clone()),
            denom: r.ground(Int::from(q.denominator().clone())),
        }
    }

    /// The generator `g` (a symbol or a kernel) as an element
    /// (`generator`).
    pub fn generator(g: &Basic) -> RationalFunction {
        if square(g).is_some()
            && g.is_pow()
            && matches!(
                g.args()[0].head(),
                Head::Rational {
                    integer_class: true,
                    ..
                }
            )
        {
            return sqrt_integer(g.args()[0].as_rational().expect("an integer").numerator());
        }
        let r = poly_ring(1);
        RationalFunction {
            gens: vec![g.clone()],
            numer: r.from_dict(Terms::from([(vec![1], Int::ONE)])),
            denom: r.one(),
        }
    }

    /// The element of a core expression (`from_expr`).
    pub fn from_expr(e: &Basic) -> PolyResult<RationalFunction> {
        from_expr(e)
    }

    pub fn is_zero(&self) -> bool {
        self.numer.terms.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.gens.is_empty() && self.numer == self.denom
    }

    /// Whether the element is a constant of `QQ(I, sqrt(2), ...)`.
    pub fn is_number(&self) -> bool {
        self.gens.iter().all(|g| square(g).is_some())
    }

    pub fn is_rational(&self) -> bool {
        self.gens.is_empty()
    }

    pub fn is_integer(&self) -> bool {
        self.gens.is_empty() && self.denom.LC().is_one()
    }

    pub fn as_fraction(&self) -> Option<Rat> {
        if !self.gens.is_empty() {
            return None;
        }
        if self.numer.terms.is_empty() {
            return Some(Rat::ZERO);
        }
        Some(Rat::from_parts_signed(self.numer.LC(), self.denom.LC()))
    }

    pub fn as_int(&self) -> Option<Int> {
        self.as_fraction().filter(|q| q.is_int()).map(|q| q.numerator().clone())
    }

    pub fn has(&self, gens: &[Basic]) -> bool {
        gens.iter().any(|g| self.gens.contains(g))
    }

    pub fn numerator(&self) -> PolyResult<RationalFunction> {
        RationalFunction::make(
            self.gens.clone(),
            self.numer.clone(),
            poly_ring(self.gens.len()).one(),
            false,
        )
    }

    pub fn denominator(&self) -> PolyResult<RationalFunction> {
        RationalFunction::make(
            self.gens.clone(),
            self.denom.clone(),
            poly_ring(self.gens.len()).one(),
            false,
        )
    }

    /// Whether the denominator is free of `g`.
    pub fn is_polynomial_in(&self, g: &Basic) -> bool {
        match self.gens.iter().position(|h| h == g) {
            None => true,
            Some(i) => !self.denom.terms.keys().any(|m| m[i] > 0),
        }
    }

    /// The degree of the numerator in `g` (`-1` for zero).
    pub fn degree(&self, g: &Basic) -> i64 {
        if self.numer.terms.is_empty() {
            return -1;
        }
        match self.gens.iter().position(|h| h == g) {
            None => 0,
            Some(i) => self.numer.terms.keys().map(|m| m[i] as i64).max().unwrap_or(0),
        }
    }

    /// Whether the leading coefficient of the numerator is negative.
    pub fn could_extract_minus_sign(&self) -> bool {
        !self.numer.terms.is_empty() && self.numer.LC() < Int::ZERO
    }

    fn unified(&self, other: &RationalFunction) -> (Vec<Basic>, ZPoly, ZPoly, ZPoly, ZPoly) {
        let gens: Vec<Basic> = merge(&self.gens, &other.gens);
        (
            gens.clone(),
            remap(&self.numer, &self.gens, &gens),
            remap(&self.denom, &self.gens, &gens),
            remap(&other.numer, &other.gens, &gens),
            remap(&other.denom, &other.gens, &gens),
        )
    }

    pub fn add(&self, b: &RationalFunction) -> PolyResult<RationalFunction> {
        if b.numer.terms.is_empty() {
            return Ok(self.clone());
        }
        if self.numer.terms.is_empty() {
            return Ok(b.clone());
        }
        let (gens, an, ad, bn, bd) = self.unified(b);
        if ad == bd {
            return RationalFunction::make(gens, an.add(&bn), ad, true);
        }
        RationalFunction::make(gens, an.mul(&bd).add(&bn.mul(&ad)), ad.mul(&bd), true)
    }

    pub fn neg(&self) -> RationalFunction {
        if self.numer.terms.is_empty() {
            return self.clone();
        }
        RationalFunction {
            gens: self.gens.clone(),
            numer: self.numer.neg(),
            denom: self.denom.clone(),
        }
    }

    pub fn sub(&self, b: &RationalFunction) -> PolyResult<RationalFunction> {
        self.add(&b.neg())
    }

    pub fn mul(&self, b: &RationalFunction) -> PolyResult<RationalFunction> {
        if b.numer.terms.is_empty() || self.numer.terms.is_empty() {
            return Ok(RationalFunction::zero());
        }
        if b.is_one() {
            return Ok(self.clone());
        }
        if self.is_one() {
            return Ok(b.clone());
        }
        let (gens, an, ad, bn, bd) = self.unified(b);
        let has_algebraic: bool = !algebraic(&gens).is_empty();
        if !has_algebraic && is_one_ground(&ad) && is_one_ground(&bd) {
            return RationalFunction::make(gens, an.mul(&bn), ad, false);
        }
        RationalFunction::make(gens, an.mul(&bn), ad.mul(&bd), true)
    }

    pub fn invert(&self) -> PolyResult<RationalFunction> {
        if self.numer.terms.is_empty() {
            return Err(zero_division("division by zero in a field of rational functions"));
        }
        RationalFunction::make(self.gens.clone(), self.denom.clone(), self.numer.clone(), true)
    }

    pub fn div(&self, b: &RationalFunction) -> PolyResult<RationalFunction> {
        self.mul(&b.invert()?)
    }

    pub fn pow(&self, n: i64) -> PolyResult<RationalFunction> {
        if n < 0 {
            return self.invert()?.pow(-n);
        }
        if n == 0 {
            return Ok(RationalFunction::one());
        }
        if n == 1 {
            return Ok(self.clone());
        }
        let mut result: RationalFunction = RationalFunction::one();
        let mut base: RationalFunction = self.clone();
        let mut n: i64 = n;
        while n > 0 {
            if n & 1 == 1 {
                result = result.mul(&base)?;
            }
            n >>= 1;
            if n > 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    /// The derivative with respect to the generator `g` (`diff`).
    pub fn diff(&self, g: &Basic) -> PolyResult<RationalFunction> {
        let Some(i) = self.gens.iter().position(|h| h == g) else {
            return Ok(RationalFunction::zero());
        };
        let (n, d) = (&self.numer, &self.denom);
        let (dn, dd) = (n.diff(i), d.diff(i));
        if dd.terms.is_empty() {
            return RationalFunction::make(self.gens.clone(), dn, d.clone(), true);
        }
        RationalFunction::make(self.gens.clone(), dn.mul(d).sub(&n.mul(&dd)), d.mul(d), true)
    }

    /// The element with `g` replaced by `value` (`subs`).
    pub fn subs(&self, g: &Basic, value: &RationalFunction) -> PolyResult<RationalFunction> {
        if !self.gens.contains(g) {
            return Ok(self.clone());
        }
        evaluate(&self.numer, &self.gens, g, value)?.div(&evaluate(&self.denom, &self.gens, g, value)?)
    }

    /// The substitutions one after the other (`xreplace`).
    pub fn xreplace(&self, rule: &[(Basic, RationalFunction)]) -> PolyResult<RationalFunction> {
        let mut result: RationalFunction = self.clone();
        for (g, v) in rule {
            result = result.subs(g, v)?;
        }
        Ok(result)
    }

    /// The element with `I` replaced by `-I` (`conjugate_i`).
    pub fn conjugate_i(&self) -> PolyResult<RationalFunction> {
        self.subs(&Basic::imaginary_unit(), &RationalFunction::imaginary_unit().neg())
    }

    /// The element as a core expression `numer/denom` (`as_expr`).
    pub fn as_expr(&self) -> PolyResult<Basic> {
        let n: Basic = poly_expr(&self.numer, &self.gens)?;
        if is_one_ground(&self.denom) {
            return Ok(n);
        }
        crate::core::mul::mul(&[
            n,
            crate::core::power::pow(&poly_expr(&self.denom, &self.gens)?, &Basic::integer(Int::NEG_ONE))?,
        ])
    }
}

fn poly_expr(f: &ZPoly, gens: &[Basic]) -> PolyResult<Basic> {
    let mut terms: Vec<Basic> = Vec::new();
    for (m, c) in &f.terms {
        let mut factors: Vec<Basic> = vec![Basic::integer(c.clone())];
        for (g, k) in gens.iter().zip(m) {
            if *k == 1 {
                factors.push(g.clone());
            } else if *k > 1 {
                factors.push(crate::core::power::pow(g, &Basic::integer(Int::from(*k)))?);
            }
        }
        terms.push(crate::core::mul::mul(&factors)?);
    }
    crate::core::add::add(&terms)
}

/// The polynomial `f` with `g` replaced by `value` (`_evaluate`).
fn evaluate(f: &ZPoly, gens: &[Basic], g: &Basic, value: &RationalFunction) -> PolyResult<RationalFunction> {
    let i: usize = gens.iter().position(|h| h == g).expect("a generator");
    let mut grouped: std::collections::BTreeMap<u32, Terms<Int>> = std::collections::BTreeMap::new();
    for (m, c) in &f.terms {
        let mut e: Monomial = m.clone();
        let k: u32 = e[i];
        e[i] = 0;
        grouped.entry(k).or_default().insert(e, c.clone());
    }
    let r = poly_ring(gens.len());
    let mut result: RationalFunction = RationalFunction::zero();
    for (k, terms) in grouped.into_iter().rev() {
        let coefficient: RationalFunction = RationalFunction::make(gens.to_vec(), r.from_dict(terms), r.one(), true)?;
        result = result.add(&coefficient.mul(&value.pow(k as i64)?)?)?;
    }
    Ok(result)
}

/// `sqrt(n)` for an integer `n`: the product of the square roots of its
/// prime factors, `I` for the sign (`_sqrt_integer`).
fn sqrt_integer(n: &Int) -> RationalFunction {
    if n.is_zero() {
        return RationalFunction::zero();
    }
    let mut result: RationalFunction = if *n < Int::ZERO {
        RationalFunction::imaginary_unit()
    } else {
        RationalFunction::one()
    };
    let magnitude: Int = if *n < Int::ZERO { -n.clone() } else { n.clone() };
    for (p, k) in crate::ntheory::factor_::factorint(&magnitude, None) {
        if k / 2 > 0 {
            result = result
                .mul(&RationalFunction::from_int(&p.pow((k / 2) as usize)))
                .expect("a product");
        }
        if k % 2 == 1 {
            let g: Basic = Basic::new(Head::Pow, vec![Basic::integer(p.clone()), Basic::half()]);
            let r = poly_ring(1);
            let root = RationalFunction {
                gens: vec![g],
                numer: r.from_dict(Terms::from([(vec![1], Int::ONE)])),
                denom: r.one(),
            };
            result = result.mul(&root).expect("a product");
        }
    }
    result
}

fn from_expr(e: &Basic) -> PolyResult<RationalFunction> {
    match e.head() {
        Head::Rational {
            integer_class: true,
            value,
        } => return Ok(RationalFunction::from_int(value.numerator())),
        Head::Rational { value, .. } => return Ok(RationalFunction::from_fraction(value)),
        Head::ImaginaryUnit => return Ok(RationalFunction::imaginary_unit()),
        _ if e.is_number_class() => {
            return Err(unsupported(format!(
                "the constant {} is not supported by the Risch algorithm",
                e
            )));
        }
        Head::Add => {
            let parts: Vec<RationalFunction> = e.args().iter().map(from_expr).collect::<PolyResult<_>>()?;
            return sum(&parts);
        }
        Head::Mul => {
            let mut result: RationalFunction = RationalFunction::one();
            for a in e.args() {
                result = result.mul(&from_expr(a)?)?;
            }
            return Ok(result);
        }
        Head::Pow => {
            let (base, exponent) = (&e.args()[0], &e.args()[1]);
            if let Some(r) = exponent.as_rational() {
                let p: i64 = i64::try_from(r.numerator()).map_err(|_| unsupported("a large exponent".to_string()))?;
                if r.is_int() {
                    return from_expr(base)?.pow(p);
                }
                if let (Some(b), true) = (base.as_rational(), *r.denominator() == dashu_int::UBig::from(2u8)) {
                    let root: RationalFunction = if b.is_int() {
                        sqrt_integer(b.numerator())
                    } else {
                        sqrt_integer(b.numerator()).div(&sqrt_integer(&Int::from(b.denominator().clone())))?
                    };
                    return root.pow(p);
                }
                return Err(unsupported(format!(
                    "the algebraic {} is not supported by the Risch algorithm",
                    e
                )));
            }
        }
        _ => {}
    }
    if e.is_pow() || matches!(e.head(), Head::Function(crate::core::basic::Function::Exp)) {
        // SymPy's generators: exp(2*x) is exp(x)**2, x**(-y) is (x**y)**-1
        let (kernel, k) = crate::core::exprtools::decompose_power(e)?;
        if k != 1 {
            return from_expr(&kernel)?.pow(k);
        }
    }
    Ok(RationalFunction::generator(e))
}

/// A sum: the polynomial parts added directly, one common denominator for
/// the others (`_sum`).
fn sum(parts: &[RationalFunction]) -> PolyResult<RationalFunction> {
    let (mut polynomial, mut fractions): (Vec<&RationalFunction>, Vec<&RationalFunction>) = (Vec::new(), Vec::new());
    for p in parts {
        if is_one_ground(&p.denom) && algebraic(&p.gens).is_empty() {
            polynomial.push(p);
        } else {
            fractions.push(p);
        }
    }
    let mut result: RationalFunction = RationalFunction::zero();
    if !polynomial.is_empty() {
        let mut gens: Vec<Basic> = Vec::new();
        for p in &polynomial {
            gens = merge(&gens, &p.gens);
        }
        let r = poly_ring(gens.len());
        let mut total: ZPoly = r.zero();
        for p in &polynomial {
            total = total.add(&remap(&p.numer, &p.gens, &gens));
        }
        result = RationalFunction::make(gens, total, r.one(), false)?;
    }
    for p in fractions {
        result = result.add(p)?;
    }
    Ok(result)
}

/// The field of [`RationalFunction`] elements as a domain of the dense
/// polynomial algorithms (`TowerDomain`, the instance `TOWER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TowerDomain;

/// The field of the tower (`TOWER`).
pub const TOWER: TowerDomain = TowerDomain;

fn infallible(r: PolyResult<RationalFunction>) -> RationalFunction {
    r.expect("an operation of the field which cannot fail")
}

impl crate::polys::domains::Domain for TowerDomain {
    type Elem = RationalFunction;

    fn name(&self) -> String {
        "K".to_string()
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        false
    }

    fn characteristic(&self) -> Int {
        Int::ZERO
    }

    fn zero(&self) -> RationalFunction {
        RationalFunction::zero()
    }

    fn one(&self) -> RationalFunction {
        RationalFunction::one()
    }

    fn to_expr(&self, a: &RationalFunction) -> PolyResult<Basic> {
        a.as_expr()
    }

    fn from_expr(&self, e: &Basic) -> PolyResult<RationalFunction> {
        from_expr(e)
    }

    fn is_zero(&self, a: &RationalFunction) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &RationalFunction) -> bool {
        a.is_one()
    }

    fn is_negative(&self, a: &RationalFunction) -> bool {
        a.could_extract_minus_sign()
    }

    fn is_unit(&self, a: &RationalFunction) -> bool {
        !a.is_zero()
    }

    fn add(&self, a: &RationalFunction, b: &RationalFunction) -> RationalFunction {
        infallible(a.add(b))
    }

    fn sub(&self, a: &RationalFunction, b: &RationalFunction) -> RationalFunction {
        infallible(a.sub(b))
    }

    fn neg(&self, a: &RationalFunction) -> RationalFunction {
        a.neg()
    }

    fn mul(&self, a: &RationalFunction, b: &RationalFunction) -> RationalFunction {
        infallible(a.mul(b))
    }

    fn pow(&self, a: &RationalFunction, n: u64) -> RationalFunction {
        infallible(a.pow(n as i64))
    }

    fn exquo(&self, a: &RationalFunction, b: &RationalFunction) -> PolyResult<RationalFunction> {
        a.div(b)
    }

    fn quo(&self, a: &RationalFunction, b: &RationalFunction) -> PolyResult<RationalFunction> {
        a.div(b)
    }

    fn rem(&self, _a: &RationalFunction, b: &RationalFunction) -> PolyResult<RationalFunction> {
        if b.is_zero() {
            return Err(zero_division("division by zero"));
        }
        Ok(RationalFunction::zero())
    }

    fn invert(&self, a: &RationalFunction) -> PolyResult<RationalFunction> {
        a.invert()
    }

    fn gcd(&self, a: &RationalFunction, b: &RationalFunction) -> RationalFunction {
        if !a.is_zero() || !b.is_zero() {
            RationalFunction::one()
        } else {
            RationalFunction::zero()
        }
    }

    fn lcm(&self, a: &RationalFunction, b: &RationalFunction) -> RationalFunction {
        if !a.is_zero() && !b.is_zero() {
            RationalFunction::one()
        } else {
            RationalFunction::zero()
        }
    }

    fn gcdex(
        &self,
        a: &RationalFunction,
        b: &RationalFunction,
    ) -> PolyResult<(RationalFunction, RationalFunction, RationalFunction)> {
        if !a.is_zero() {
            return Ok((a.invert()?, RationalFunction::zero(), RationalFunction::one()));
        }
        if !b.is_zero() {
            return Ok((RationalFunction::zero(), b.invert()?, RationalFunction::one()));
        }
        Ok((
            RationalFunction::zero(),
            RationalFunction::zero(),
            RationalFunction::zero(),
        ))
    }

    fn canonical_unit(&self, a: &RationalFunction) -> RationalFunction {
        if a.is_zero() {
            RationalFunction::one()
        } else {
            infallible(a.invert())
        }
    }

    fn size(&self, a: &RationalFunction) -> u64 {
        (a.numer.terms.len() + a.denom.terms.len()) as u64
    }

    fn from_int(&self, n: i64) -> RationalFunction {
        RationalFunction::from_int(&Int::from(n))
    }

    fn from_integer(&self, n: &Int) -> RationalFunction {
        RationalFunction::from_int(n)
    }

    fn to_str(&self, a: &RationalFunction) -> String {
        a.to_string()
    }
}

/// A dense polynomial over the field, the highest degree first.
pub type Dup = Vec<RationalFunction>;

fn polynomial_error(message: String) -> PolyError {
    crate::error::polys_error(crate::error::PolysErrorKind::PolynomialError, message)
}

/// A polynomial in `gen` with coefficients free of `gen` (an element of
/// `k[t]`), dense, the highest degree first (`TowerPoly`): the part of
/// SymPy's `Poly` the Risch algorithm uses.
#[derive(Debug, Clone)]
pub struct TowerPoly {
    pub rep: Dup,
    pub gen: Basic,
}

impl PartialEq for TowerPoly {
    fn eq(&self, other: &TowerPoly) -> bool {
        if self.gen == other.gen {
            return self.rep == other.rep;
        }
        self.rep.len() <= 1 && other.rep.len() <= 1 && self.rep == other.rep
    }
}

impl std::fmt::Display for TowerPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_expr() {
            Ok(e) => write!(f, "TowerPoly({}, {})", e, self.gen),
            Err(_) => write!(f, "TowerPoly(?, {})", self.gen),
        }
    }
}

impl TowerPoly {
    pub fn new(rep: Dup, gen: &Basic) -> TowerPoly {
        TowerPoly {
            rep: crate::polys::densebasic::dup_strip(&rep, &TOWER),
            gen: gen.clone(),
        }
    }

    fn with(&self, rep: Dup) -> TowerPoly {
        TowerPoly::new(rep, &self.gen)
    }

    /// `a` as a polynomial in `gen`; `None` when its denominator depends on
    /// `gen` (`from_element`).
    pub fn from_element(a: &RationalFunction, gen: &Basic) -> PolyResult<Option<TowerPoly>> {
        let Some(i) = a.gens.iter().position(|g| g == gen) else {
            return Ok(Some(TowerPoly::new(
                if a.is_zero() { Vec::new() } else { vec![a.clone()] },
                gen,
            )));
        };
        if a.denom.terms.keys().any(|m| m[i] > 0) {
            return Ok(None);
        }
        let mut grouped: HashMap<u32, Terms<Int>> = HashMap::new();
        for (m, c) in &a.numer.terms {
            let mut e: Monomial = m.clone();
            let k: u32 = e[i];
            e[i] = 0;
            grouped.entry(k).or_default().insert(e, c.clone());
        }
        let n: u32 = grouped.keys().copied().max().unwrap_or(0);
        let r = poly_ring(a.gens.len());
        let mut rep: Dup = Vec::new();
        for k in (0..=n).rev() {
            match grouped.remove(&k) {
                None => rep.push(RationalFunction::zero()),
                Some(terms) => rep.push(RationalFunction::make(
                    a.gens.clone(),
                    r.from_dict(terms),
                    a.denom.clone(),
                    true,
                )?),
            }
        }
        Ok(Some(TowerPoly::new(rep, gen)))
    }

    pub fn from_element_strict(a: &RationalFunction, gen: &Basic) -> PolyResult<TowerPoly> {
        TowerPoly::from_element(a, gen)?
            .ok_or_else(|| polynomial_error(format!("{} is not a polynomial in {}", a, gen)))
    }

    /// The polynomial in `gen` of a core expression (`PolynomialError` when
    /// it is not one).
    pub fn from_expr(e: &Basic, gen: &Basic) -> PolyResult<TowerPoly> {
        let a: RationalFunction = from_expr(e)?;
        if a.gens.iter().any(|g| g != gen && g.has(gen)) {
            return Err(polynomial_error(format!(
                "{} contains an element of the set of generators",
                e
            )));
        }
        TowerPoly::from_element_strict(&a, gen)
    }

    pub fn from_list(coefficients: &[RationalFunction], gen: &Basic) -> TowerPoly {
        TowerPoly::new(coefficients.to_vec(), gen)
    }

    pub fn zero(gen: &Basic) -> TowerPoly {
        TowerPoly::new(Vec::new(), gen)
    }

    pub fn one(gen: &Basic) -> TowerPoly {
        TowerPoly::new(vec![RationalFunction::one()], gen)
    }

    /// The constant polynomial `c` (an element free of `gen`).
    pub fn ground(c: &RationalFunction, gen: &Basic) -> PolyResult<TowerPoly> {
        if c.has(std::slice::from_ref(gen)) {
            return Err(polynomial_error(format!("{} is not a constant in {}", c, gen)));
        }
        Ok(TowerPoly::new(vec![c.clone()], gen))
    }

    /// `c*gen**n`.
    pub fn monomial(gen: &Basic, n: usize, c: &RationalFunction) -> TowerPoly {
        if c.is_zero() {
            return TowerPoly::zero(gen);
        }
        let mut rep: Dup = vec![c.clone()];
        rep.extend(std::iter::repeat(RationalFunction::zero()).take(n));
        TowerPoly::new(rep, gen)
    }

    /// The polynomial as a core expression `sum c_k gen**k`.
    pub fn as_expr(&self) -> PolyResult<Basic> {
        let n: usize = self.rep.len().saturating_sub(1);
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in self.rep.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let k: usize = n - i;
            let power: Basic = match k {
                0 => Basic::one(),
                1 => self.gen.clone(),
                _ => crate::core::power::pow(&self.gen, &Basic::integer(Int::from(k)))?,
            };
            terms.push(crate::core::mul::mul(&[c.as_expr()?, power])?);
        }
        crate::core::add::add(&terms)
    }

    /// The polynomial as an element of the field.
    pub fn as_element(&self) -> PolyResult<RationalFunction> {
        match self.rep.len() {
            0 => Ok(RationalFunction::zero()),
            1 => Ok(self.rep[0].clone()),
            _ => horner(&self.rep, &self.gen),
        }
    }

    /// The polynomial in another generator; `None` when it is not one.
    pub fn as_poly(&self, gen: &Basic) -> PolyResult<Option<TowerPoly>> {
        if *gen == self.gen {
            return Ok(Some(self.clone()));
        }
        TowerPoly::from_element(&self.as_element()?, gen)
    }

    pub fn as_poly_strict(&self, gen: &Basic) -> PolyResult<TowerPoly> {
        self.as_poly(gen)?
            .ok_or_else(|| polynomial_error(format!("{} is not a polynomial in {}", self, gen)))
    }

    /// The degree in `gen` (by default the generator; `-1` for zero).
    pub fn degree(&self, gen: Option<&Basic>) -> i64 {
        match gen {
            Some(g) if *g != self.gen => {
                if self.rep.is_empty() {
                    return -1;
                }
                self.rep
                    .iter()
                    .filter(|c| !c.is_zero())
                    .map(|c| c.degree(g))
                    .max()
                    .unwrap_or(-1)
            }
            _ => self.rep.len() as i64 - 1,
        }
    }

    #[allow(non_snake_case)]
    pub fn LC(&self) -> RationalFunction {
        self.rep.first().cloned().unwrap_or_else(RationalFunction::zero)
    }

    #[allow(non_snake_case)]
    pub fn TC(&self) -> RationalFunction {
        self.rep.last().cloned().unwrap_or_else(RationalFunction::zero)
    }

    /// The coefficient of `gen**n`.
    pub fn nth(&self, n: i64) -> RationalFunction {
        let k: i64 = self.rep.len() as i64 - 1 - n;
        if n < 0 || k < 0 {
            return RationalFunction::zero();
        }
        self.rep[k as usize].clone()
    }

    pub fn all_coeffs(&self) -> Dup {
        if self.rep.is_empty() {
            vec![RationalFunction::zero()]
        } else {
            self.rep.clone()
        }
    }

    /// The nonzero coefficients, the highest degree first.
    pub fn coeffs(&self) -> Dup {
        let found: Dup = self.rep.iter().filter(|c| !c.is_zero()).cloned().collect();
        if found.is_empty() {
            vec![RationalFunction::zero()]
        } else {
            found
        }
    }

    /// The `(degree, coefficient)` pairs of the nonzero terms.
    pub fn terms(&self) -> Vec<(usize, RationalFunction)> {
        let n: usize = self.rep.len().saturating_sub(1);
        self.rep
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| (n - i, c.clone()))
            .collect()
    }

    /// The term of the lowest degree.
    #[allow(non_snake_case)]
    pub fn ET(&self) -> (usize, RationalFunction) {
        let n: usize = self.rep.len().saturating_sub(1);
        for i in (0..self.rep.len()).rev() {
            if !self.rep[i].is_zero() {
                return (n - i, self.rep[i].clone());
            }
        }
        (0, RationalFunction::zero())
    }

    pub fn is_zero(&self) -> bool {
        self.rep.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.rep.len() == 1 && self.rep[0].is_one()
    }

    pub fn is_ground(&self) -> bool {
        self.rep.len() <= 1
    }

    pub fn is_monic(&self) -> bool {
        !self.rep.is_empty() && self.rep[0].is_one()
    }

    pub fn is_monomial(&self) -> bool {
        self.rep.iter().filter(|c| !c.is_zero()).count() <= 1
    }

    /// Whether the polynomial depends on one of `gens`.
    pub fn has(&self, gens: &[Basic]) -> bool {
        if gens.contains(&self.gen) && self.rep.len() > 1 {
            return true;
        }
        self.rep.iter().any(|c| c.has(gens))
    }

    /// Equality with an element of the field (`__eq__` with a
    /// `RationalFunction` or an `int`).
    pub fn eq_element(&self, a: &RationalFunction) -> bool {
        self.rep.len() <= 1 && self.LC() == *a
    }

    /// The representations of `self` and `other` in one generator
    /// (`_binary`).
    fn binary(&self, other: &TowerPoly) -> PolyResult<(Dup, Dup, Basic)> {
        let gen: &Basic = &self.gen;
        if other.gen == *gen {
            return Ok((self.rep.clone(), other.rep.clone(), gen.clone()));
        }
        if other.rep.len() <= 1 && !(other.rep.first().is_some_and(|c| c.has(std::slice::from_ref(gen)))) {
            return Ok((self.rep.clone(), other.rep.clone(), gen.clone()));
        }
        if self.rep.len() <= 1
            && !(self
                .rep
                .first()
                .is_some_and(|c| c.has(std::slice::from_ref(&other.gen))))
        {
            return Ok((self.rep.clone(), other.rep.clone(), other.gen.clone()));
        }
        match other.as_poly(gen)? {
            None => Err(polynomial_error(format!("polynomials in {} and {}", gen, other.gen))),
            Some(converted) => Ok((self.rep.clone(), converted.rep, gen.clone())),
        }
    }

    /// `_binary` with an element of the field.
    fn binary_element(&self, a: &RationalFunction) -> PolyResult<(Dup, Dup, Basic)> {
        let gen: &Basic = &self.gen;
        if a.is_zero() {
            return Ok((self.rep.clone(), Vec::new(), gen.clone()));
        }
        if a.has(std::slice::from_ref(gen)) {
            return Ok((
                self.rep.clone(),
                TowerPoly::from_element_strict(a, gen)?.rep,
                gen.clone(),
            ));
        }
        Ok((self.rep.clone(), vec![a.clone()], gen.clone()))
    }

    pub fn neg(&self) -> TowerPoly {
        self.with(crate::polys::densearith::dup_neg(&self.rep, &TOWER))
    }

    pub fn add(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_add(&f, &g, &TOWER), &gen))
    }

    pub fn add_element(&self, a: &RationalFunction) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary_element(a)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_add(&f, &g, &TOWER), &gen))
    }

    pub fn sub(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_sub(&f, &g, &TOWER), &gen))
    }

    pub fn sub_element(&self, a: &RationalFunction) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary_element(a)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_sub(&f, &g, &TOWER), &gen))
    }

    pub fn mul(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_mul(&f, &g, &TOWER), &gen))
    }

    pub fn mul_element(&self, a: &RationalFunction) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary_element(a)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_mul(&f, &g, &TOWER), &gen))
    }

    pub fn pow(&self, n: usize) -> TowerPoly {
        self.with(crate::polys::densearith::dup_pow(&self.rep, n, &TOWER))
    }

    pub fn mul_ground(&self, c: &RationalFunction) -> TowerPoly {
        self.with(crate::polys::densearith::dup_mul_ground(&self.rep, c, &TOWER))
    }

    pub fn quo_ground(&self, c: &RationalFunction) -> PolyResult<TowerPoly> {
        Ok(self.with(crate::polys::densearith::dup_mul_ground(
            &self.rep,
            &c.invert()?,
            &TOWER,
        )))
    }

    pub fn exquo_ground(&self, c: &RationalFunction) -> PolyResult<TowerPoly> {
        self.quo_ground(c)
    }

    pub fn div(&self, other: &TowerPoly) -> PolyResult<(TowerPoly, TowerPoly)> {
        let (f, g, gen) = self.binary(other)?;
        let (q, r) = crate::polys::densearith::dup_div(&f, &g, &TOWER)?;
        Ok((TowerPoly::new(q, &gen), TowerPoly::new(r, &gen)))
    }

    pub fn quo(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_quo(&f, &g, &TOWER)?, &gen))
    }

    pub fn rem(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(crate::polys::densearith::dup_rem(&f, &g, &TOWER)?, &gen))
    }

    /// The exact quotient (an error when the division is not exact).
    pub fn exquo(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(
            crate::polys::densearith::dup_exquo(&f, &g, &TOWER)?,
            &gen,
        ))
    }

    /// The monic gcd (`0` for two zeros).
    pub fn gcd(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(crate::polys::euclidtools::dup_gcd(&f, &g, &TOWER), &gen))
    }

    pub fn lcm(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(crate::polys::euclidtools::dup_lcm(&f, &g, &TOWER), &gen))
    }

    pub fn monic(&self) -> PolyResult<TowerPoly> {
        if self.rep.is_empty() {
            return Ok(self.clone());
        }
        Ok(self.with(crate::polys::densetools::dup_monic(&self.rep, &TOWER)?))
    }

    /// The derivative with respect to the generator, or to another
    /// generator of the coefficients.
    pub fn diff(&self, gen: Option<&Basic>) -> PolyResult<TowerPoly> {
        match gen {
            Some(g) if *g != self.gen => Ok(self.with(self.rep.iter().map(|c| c.diff(g)).collect::<PolyResult<_>>()?)),
            _ => Ok(self.with(crate::polys::densetools::dup_diff(&self.rep, 1, &TOWER))),
        }
    }

    /// The value at `gen = value`.
    pub fn eval(&self, value: &RationalFunction) -> RationalFunction {
        crate::polys::densetools::dup_eval(&self.rep, value, &TOWER)
    }

    /// The polynomial with `gen` (a generator of the coefficients, or the
    /// generator itself) replaced by `value`.
    pub fn subs(&self, gen: &Basic, value: &RationalFunction) -> PolyResult<TowerPoly> {
        let own: &[Basic] = std::slice::from_ref(&self.gen);
        if *gen == self.gen {
            if !value.has(own) {
                return TowerPoly::ground(&self.eval(value), &self.gen);
            }
            return TowerPoly::from_element_strict(&horner_at(&self.rep, value)?, &self.gen);
        }
        if value.has(own) {
            return TowerPoly::from_element_strict(&self.as_element()?.subs(gen, value)?, &self.gen);
        }
        Ok(self.with(self.rep.iter().map(|c| c.subs(gen, value)).collect::<PolyResult<_>>()?))
    }

    /// `(p, q)` with `p/q == self/other` in lowest terms (SymPy's
    /// `Poly.cancel(include=True)` over a field).
    pub fn cancel(&self, other: &TowerPoly) -> PolyResult<(TowerPoly, TowerPoly)> {
        let (f, g, gen) = self.binary(other)?;
        if g.is_empty() {
            return Err(zero_division("cancel by zero"));
        }
        if f.is_empty() {
            return Ok((TowerPoly::zero(&gen), TowerPoly::one(&gen)));
        }
        let h: Dup = crate::polys::euclidtools::dup_gcd(&f, &g, &TOWER);
        let p: Dup = crate::polys::densearith::dup_quo(&f, &h, &TOWER)?;
        let q: Dup = crate::polys::densearith::dup_quo(&g, &h, &TOWER)?;
        let (p, q) = integral_pair(&p, &q)?;
        Ok((TowerPoly::new(p, &gen), TowerPoly::new(q, &gen)))
    }

    pub fn resultant(&self, other: &TowerPoly) -> PolyResult<RationalFunction> {
        let (f, g, _) = self.binary(other)?;
        Ok(crate::polys::euclidtools::dup_resultant(&f, &g, &TOWER))
    }

    /// The resultant and the subresultant PRS (SymPy's
    /// `resultant(includePRS=True)`).
    pub fn resultant_prs(&self, other: &TowerPoly) -> PolyResult<(RationalFunction, Vec<TowerPoly>)> {
        let (f, g, gen) = self.binary(other)?;
        if f.is_empty() || g.is_empty() {
            return Ok((RationalFunction::zero(), Vec::new()));
        }
        let (prs, scalars) = crate::polys::euclidtools::dup_inner_subresultants(&f, &g, &TOWER);
        let last: &Dup = prs.last().expect("a sequence");
        let res: RationalFunction = if crate::polys::densebasic::dup_degree(last) > 0 {
            RationalFunction::zero()
        } else {
            scalars.last().cloned().unwrap_or_else(RationalFunction::zero)
        };
        Ok((res, prs.into_iter().map(|p| TowerPoly::new(p, &gen)).collect()))
    }

    pub fn sqf_list(&self) -> (RationalFunction, Vec<(TowerPoly, usize)>) {
        let (c, factors) = crate::polys::sqfreetools::dup_sqf_list(&self.rep, &TOWER);
        (c, factors.into_iter().map(|(f, k)| (self.with(f), k)).collect())
    }

    /// The square-free factors, the content in the first one (SymPy's
    /// `sqf_list_include`).
    pub fn sqf_list_include(&self) -> Vec<(TowerPoly, usize)> {
        let (c, mut factors) = self.sqf_list();
        if factors.is_empty() {
            return vec![(self.with(if c.is_zero() { Vec::new() } else { vec![c] }), 1)];
        }
        if c.is_one() {
            return factors;
        }
        if factors[0].1 == 1 {
            factors[0].0 = factors[0].0.mul_ground(&c);
            return factors;
        }
        let mut result: Vec<(TowerPoly, usize)> = vec![(self.with(vec![c]), 1)];
        result.extend(factors);
        result
    }

    pub fn half_gcdex(&self, other: &TowerPoly) -> PolyResult<(TowerPoly, TowerPoly)> {
        let (f, g, gen) = self.binary(other)?;
        let (s, h) = crate::polys::euclidtools::dup_half_gcdex(&f, &g, &TOWER)?;
        Ok((TowerPoly::new(s, &gen), TowerPoly::new(h, &gen)))
    }

    pub fn gcdex(&self, other: &TowerPoly) -> PolyResult<(TowerPoly, TowerPoly, TowerPoly)> {
        let (f, g, gen) = self.binary(other)?;
        let (s, t, h) = crate::polys::euclidtools::dup_gcdex(&f, &g, &TOWER)?;
        Ok((
            TowerPoly::new(s, &gen),
            TowerPoly::new(t, &gen),
            TowerPoly::new(h, &gen),
        ))
    }

    /// The inverse modulo `other` (an error when there is none).
    pub fn invert(&self, other: &TowerPoly) -> PolyResult<TowerPoly> {
        let (f, g, gen) = self.binary(other)?;
        Ok(TowerPoly::new(
            crate::polys::euclidtools::dup_invert(&f, &g, &TOWER)?,
            &gen,
        ))
    }

    /// The polynomial of `gen + a`.
    pub fn compose_shift(&self, a: &RationalFunction) -> TowerPoly {
        self.with(crate::polys::densetools::dup_shift(&self.rep, a, &TOWER))
    }

    pub fn coefficients_have(&self, gens: &[Basic]) -> bool {
        self.rep.iter().any(|c| c.has(gens))
    }
}

/// The lcm of two polynomials (elements with the denominator 1).
fn polynomial_lcm(a: &RationalFunction, b: &RationalFunction) -> PolyResult<RationalFunction> {
    use crate::polys::domains::Domain;
    if a == b {
        return Ok(a.clone());
    }
    let gens: Vec<Basic> = merge(&a.gens, &b.gens);
    let r = ring(gens.len());
    let m: ZPoly = r.lcm(&remap(&a.numer, &a.gens, &gens), &remap(&b.numer, &b.gens, &gens));
    let one: ZPoly = r.ring.one();
    RationalFunction::make(gens, m, one, false)
}

/// The gcd of two polynomials (elements with the denominator 1).
fn polynomial_gcd(a: &RationalFunction, b: &RationalFunction) -> PolyResult<RationalFunction> {
    use crate::polys::domains::Domain;
    if a.is_zero() {
        return Ok(b.clone());
    }
    if b.is_zero() || a == b {
        return Ok(a.clone());
    }
    let gens: Vec<Basic> = merge(&a.gens, &b.gens);
    let r = ring(gens.len());
    let g: ZPoly = r.gcd(&remap(&a.numer, &a.gens, &gens), &remap(&b.numer, &b.gens, &gens));
    let one: ZPoly = r.ring.one();
    RationalFunction::make(gens, g, one, false)
}

/// SymPy's form of `p/q` of `Poly.cancel(include=True)` over a fraction
/// field: the denominators of the coefficients cleared, the common content
/// taken out, the leading coefficient of `q` positive (`integral_pair`).
pub fn integral_pair(p: &[RationalFunction], q: &[RationalFunction]) -> PolyResult<(Dup, Dup)> {
    use crate::polys::densearith::{dup_mul_ground, dup_neg};
    let mut denominator: RationalFunction = RationalFunction::one();
    for c in p.iter().chain(q) {
        if !c.is_zero() && !is_one_ground(&c.denom) {
            denominator = polynomial_lcm(&denominator, &c.denominator()?)?;
        }
    }
    let (mut p, mut q): (Dup, Dup) = (p.to_vec(), q.to_vec());
    if !denominator.is_one() {
        p = dup_mul_ground(&p, &denominator, &TOWER);
        q = dup_mul_ground(&q, &denominator, &TOWER);
    }
    let mut content: RationalFunction = RationalFunction::zero();
    for c in p.iter().chain(&q) {
        if !c.is_zero() {
            content = polynomial_gcd(&content, &c.numerator()?)?;
            if content.is_one() {
                break;
            }
        }
    }
    if !content.is_zero() && !content.is_one() {
        let inverse: RationalFunction = content.invert()?;
        p = dup_mul_ground(&p, &inverse, &TOWER);
        q = dup_mul_ground(&q, &inverse, &TOWER);
    }
    if q.first().is_some_and(|c| c.could_extract_minus_sign()) {
        p = dup_neg(&p, &TOWER);
        q = dup_neg(&q, &TOWER);
    }
    Ok((p, q))
}

/// `sum c_k gen**k` in the field, through a common denominator
/// (`_horner`).
fn horner(rep: &[RationalFunction], gen: &Basic) -> PolyResult<RationalFunction> {
    use crate::polys::domains::Domain;
    let mut gens: Vec<Basic> = vec![gen.clone()];
    for c in rep {
        gens = merge(&gens, &c.gens);
    }
    let r = ring(gens.len());
    let i: usize = gens.iter().position(|g| g == gen).expect("the generator");
    let denominators: Vec<ZPoly> = rep
        .iter()
        .filter(|c| !c.is_zero())
        .map(|c| remap(&c.denom, &c.gens, &gens))
        .collect();
    let mut common: ZPoly = denominators[0].clone();
    for d in &denominators[1..] {
        if *d != common {
            common = r.lcm(&common, d);
        }
    }
    let mut total: ZPoly = r.ring.zero();
    let n: usize = rep.len() - 1;
    let width: usize = gens.len().max(1);
    for (j, c) in rep.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let k: usize = n - j;
        let mut numer: ZPoly = remap(&c.numer, &c.gens, &gens);
        let denom: ZPoly = remap(&c.denom, &c.gens, &gens);
        if denom != common {
            numer = numer.mul(&common.exquo(&denom)?);
        }
        let shift: Vec<u32> = (0..width).map(|idx| if idx == i { k as u32 } else { 0 }).collect();
        total = total.add(&numer.mul_monom(&shift));
    }
    RationalFunction::make(gens, total, common, false)
}

fn horner_at(rep: &[RationalFunction], value: &RationalFunction) -> PolyResult<RationalFunction> {
    let mut result: RationalFunction = RationalFunction::zero();
    for c in rep {
        result = result.mul(value)?.add(c)?;
    }
    Ok(result)
}
