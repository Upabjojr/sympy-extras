//! Port of `sympy_extras/polys/numberfields/qqbar.py`: the field of
//! algebraic numbers, exact arithmetic on the roots of polynomials with
//! rational coefficients.
//!
//! An [`Algebraic`] is its minimal polynomial over `QQ` (primitive integer
//! coefficients, positive leading coefficient) and a region isolating it
//! among the roots: a rational point, an open interval for a real number, a
//! rectangle otherwise. A sum, product or power is a root of a resultant
//! (`Res_y(p(x - y), q(y))`, `Res_y(y**d p(x/y), q(y))`,
//! `Res_y(p(y), x - y**n)`), factored over `ZZ`; the factor and the root are
//! selected by interval enclosures and exact root counts in boxes (winding
//! numbers, [`super::super::complexroots`]). Equality, the order of the real
//! numbers and signs are decided, never estimated.
//!
//! As in Python, the isolating region is refined in place whenever a
//! smaller enclosure is asked for (behind a mutex here, so that the methods
//! take `&self`); the values and the decisions do not depend on it.
//!
//! References
//! ==========
//!
//! 1. R. Loos, *Computing in algebraic extensions*, in Computer Algebra:
//!    Symbolic and Algebraic Computation, Springer, 1982, 173-187.
//! 2. H. Cohen, *A Course in Computational Algebraic Number Theory*,
//!    Springer, 1993, Section 4.2.

use std::sync::Mutex;

use crate::core::symbol::Symbol;
use crate::error::{arithmetic_error, value_error, zero_division, PolyError, PolyResult};
use crate::numbers::*;

use super::super::complexroots::{dup_isolate_complex_roots_sqf, refine, winding_number, Rectangle};
use super::super::densearith::dup_mul;
use super::super::domains::{AlgebraicElement, AlgebraicField, Domain, PolynomialRing, ZZ};
use super::super::euclidtools::{dup_gcd, dup_resultant};
use super::super::factortools::dup_zz_factor;
use super::super::rootisolation::{dup_isolate_real_roots_sqf, dup_refine_real_root, Interval};

/// A box `[re_min, re_max] x [im_min, im_max]`, possibly flat.
pub type Box4 = (Rat, Rat, Rat, Rat);

/// An element of `ZZ[x][y]`: a polynomial in `y` over `ZZ[x]`.
pub(crate) type BiPoly = Vec<Vec<Int>>;

pub(crate) fn zx() -> PolynomialRing<super::super::domains::IntegerRing> {
    PolynomialRing::new(ZZ, "x")
}

fn gcd_abs(a: &Int, b: &Int) -> Int {
    abs(&gcd(a, b))
}

/// Without common factor, the leading coefficient positive.
fn normalised(f: &[Int]) -> Vec<Int> {
    let mut content: Int = Int::ZERO;
    for c in f {
        content = gcd_abs(&content, c);
    }
    if content.is_zero() {
        return Vec::new();
    }
    if is_negative(&f[0]) {
        content = -content;
    }
    f.iter().map(|c| div_floor(c, &content)).collect()
}

/// `m(y)` as a polynomial in `y` over `ZZ[x]`.
pub(crate) fn in_y(m: &[Int]) -> BiPoly {
    m.iter()
        .map(|c| if c.is_zero() { Vec::new() } else { vec![c.clone()] })
        .collect()
}

fn add_constant(f: BiPoly, c: Vec<Int>) -> BiPoly {
    if f.is_empty() {
        return vec![c];
    }
    let mut f: BiPoly = f;
    let last: Vec<Int> = f.pop().expect("a coefficient");
    f.push(zx().add(&last, &c));
    f
}

/// `Res_y(m1(x - y), m2(y))` (`minpoly._sum_resultant`).
pub fn sum_resultant(m1: &[Int], m2: &[Int]) -> Vec<Int> {
    let ring = zx();
    let x_minus_y: BiPoly = vec![vec![int(-1)], vec![Int::ONE, Int::ZERO]];
    let mut shifted: BiPoly = Vec::new();
    for c in m1 {
        // Horner in the ring ZZ[x][y]
        shifted = dup_mul(&shifted, &x_minus_y, &ring);
        if !c.is_zero() {
            shifted = add_constant(shifted, vec![c.clone()]);
        }
    }
    dup_resultant(&shifted, &in_y(m2), &ring)
}

/// `Res_y(y**d m1(x/y), m2(y))`, both numbers nonzero
/// (`minpoly._product_resultant`).
pub fn product_resultant(m1: &[Int], m2: &[Int]) -> Vec<Int> {
    let d: usize = m1.len() - 1;
    let scaled: BiPoly = (0..=d)
        .map(|k| {
            let c: &Int = &m1[d - k];
            if c.is_zero() {
                Vec::new()
            } else {
                let mut v: Vec<Int> = vec![c.clone()];
                v.extend(vec![Int::ZERO; k]);
                v
            }
        })
        .collect();
    dup_resultant(&scaled, &in_y(m2), &zx())
}

fn rmin(values: &[Rat]) -> Rat {
    values.iter().min().expect("values").clone()
}

fn rmax(values: &[Rat]) -> Rat {
    values.iter().max().expect("values").clone()
}

/// The product of the intervals `[a, b]` and `[c, d]`.
fn interval_product(a: &Rat, b: &Rat, c: &Rat, d: &Rat) -> (Rat, Rat) {
    let products: [Rat; 4] = [a * c, a * d, b * c, b * d];
    (rmin(&products), rmax(&products))
}

fn box_sum(p: &Box4, q: &Box4) -> Box4 {
    (&p.0 + &q.0, &p.1 + &q.1, &p.2 + &q.2, &p.3 + &q.3)
}

fn box_product(p: &Box4, q: &Box4) -> Box4 {
    let rr = interval_product(&p.0, &p.1, &q.0, &q.1);
    let ii = interval_product(&p.2, &p.3, &q.2, &q.3);
    let ri = interval_product(&p.0, &p.1, &q.2, &q.3);
    let ir = interval_product(&p.2, &p.3, &q.0, &q.1);
    (&rr.0 - &ii.1, &rr.1 - &ii.0, &ri.0 + &ir.0, &ri.1 + &ir.1)
}

/// The minimum of `t**2` on `[low, high]`.
fn smallest_square(low: &Rat, high: &Rat) -> Rat {
    if *low <= Rat::ZERO && Rat::ZERO <= *high {
        return Rat::ZERO;
    }
    std::cmp::min(low * low, high * high)
}

/// A box containing the inverses of the points of `p`; `None` when `p`
/// contains zero.
fn box_inverse(p: &Box4) -> Option<Box4> {
    let norm_low: Rat = smallest_square(&p.0, &p.1) + smallest_square(&p.2, &p.3);
    if norm_low.is_zero() {
        return None;
    }
    let norm_high: Rat = std::cmp::max(&p.0 * &p.0, &p.1 * &p.1) + std::cmp::max(&p.2 * &p.2, &p.3 * &p.3);
    // (x - i y)/(x**2 + y**2)
    let re: [Rat; 4] = [&p.0 / &norm_low, &p.0 / &norm_high, &p.1 / &norm_low, &p.1 / &norm_high];
    let im: [Rat; 4] = [
        -&p.2 / &norm_low,
        -&p.2 / &norm_high,
        -&p.3 / &norm_low,
        -&p.3 / &norm_high,
    ];
    Some((rmin(&re), rmax(&re), rmin(&im), rmax(&im)))
}

fn disjoint(p: &Box4, q: &Box4) -> bool {
    p.1 < q.0 || q.1 < p.0 || p.3 < q.2 || q.3 < p.2
}

/// The isolating region: an interval of a real number, a rectangle
/// otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Region {
    Interval(Interval),
    Rectangle(Rectangle),
}

/// An algebraic number: its minimal polynomial and an isolating region.
#[derive(Debug)]
pub struct Algebraic {
    minpoly: Vec<Int>,
    rational_minpoly: Vec<Rat>,
    region: Mutex<Region>,
}

impl Clone for Algebraic {
    fn clone(&self) -> Algebraic {
        Algebraic {
            minpoly: self.minpoly.clone(),
            rational_minpoly: self.rational_minpoly.clone(),
            region: Mutex::new(self.region()),
        }
    }
}

/// Python's `'%.10g' % x`.
pub fn format_g10(x: f64) -> String {
    const PRECISION: i32 = 10;
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    let scientific: String = format!("{:.*e}", (PRECISION - 1) as usize, x);
    let (mantissa, exponent) = scientific.split_once('e').expect("an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if !(-4..PRECISION).contains(&exponent) {
        let sign: char = if exponent < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", strip(mantissa), sign, exponent.abs())
    } else {
        strip(&format!("{:.*}", (PRECISION - 1 - exponent) as usize, x))
    }
}

impl Algebraic {
    fn new(minpoly: Vec<Int>, region: Region) -> Algebraic {
        let rational_minpoly: Vec<Rat> = minpoly.iter().map(rat_from_int).collect();
        Algebraic {
            minpoly,
            rational_minpoly,
            region: Mutex::new(region),
        }
    }

    fn region(&self) -> Region {
        self.region.lock().expect("an unpoisoned region").clone()
    }

    fn set_region(&self, region: Region) {
        *self.region.lock().expect("an unpoisoned region") = region;
    }

    // -- constructors

    /// The rational number `q`.
    pub fn from_rational(q: &Rat) -> Algebraic {
        Algebraic::new(vec![denom(q), -numer(q)], Region::Interval((q.clone(), q.clone())))
    }

    /// The root `i` of `x**2 + 1` in the upper half-plane.
    pub fn imaginary_unit() -> Algebraic {
        Algebraic::new(
            vec![Int::ONE, Int::ZERO, Int::ONE],
            Region::Rectangle(Rectangle::new(rat(-1, 1), rat(1, 1), rat(1, 2), rat(2, 1)).expect("a rectangle")),
        )
    }

    /// From a minimal polynomial and an isolating interval (a real number).
    pub fn real(minpoly: Vec<Int>, interval: Interval) -> Algebraic {
        Algebraic::new(minpoly, Region::Interval(interval))
    }

    /// From a minimal polynomial and an isolating rectangle.
    pub fn complex(minpoly: Vec<Int>, rectangle: Rectangle) -> Algebraic {
        Algebraic::new(minpoly, Region::Rectangle(rectangle))
    }

    // -- description

    pub fn minpoly(&self) -> Vec<Int> {
        self.minpoly.clone()
    }

    pub fn degree(&self) -> usize {
        self.minpoly.len() - 1
    }

    pub fn is_real(&self) -> bool {
        matches!(self.region(), Region::Interval(_))
    }

    pub fn is_rational(&self) -> bool {
        self.minpoly.len() == 2
    }

    pub fn is_zero(&self) -> bool {
        self.minpoly == [Int::ONE, Int::ZERO]
    }

    /// The value of a rational number (`ValueError` otherwise).
    pub fn rational_value(&self) -> PolyResult<Rat> {
        if !self.is_rational() {
            return Err(value_error("not a rational number".to_string()));
        }
        Ok(rat_from_parts(-&self.minpoly[1], self.minpoly[0].clone()))
    }

    /// A box containing the number, with sides shorter than `eps` (the
    /// isolating region is refined and kept).
    fn box_of(&self, eps: &Rat) -> PolyResult<Box4> {
        match self.region() {
            Region::Interval((low, high)) => {
                let (low, high) = if &high - &low >= *eps {
                    let refined: Interval = dup_refine_real_root(&self.rational_minpoly, low, high, eps);
                    self.set_region(Region::Interval(refined.clone()));
                    refined
                } else {
                    (low, high)
                };
                Ok((low, high, Rat::ZERO, Rat::ZERO))
            }
            Region::Rectangle(rectangle) => {
                let rectangle: Rectangle = if rectangle.width() >= *eps || rectangle.height() >= *eps {
                    let refined: Rectangle = refine(&self.minpoly, rectangle, Some(eps))?;
                    self.set_region(Region::Rectangle(refined.clone()));
                    refined
                } else {
                    rectangle
                };
                Ok((rectangle.re_min, rectangle.re_max, rectangle.im_min, rectangle.im_max))
            }
        }
    }

    /// The current isolating interval of a real number (`TypeError` in
    /// Python for a number which is not real: `Value` here).
    pub fn real_interval(&self) -> PolyResult<Interval> {
        match self.region() {
            Region::Interval(interval) => Ok(interval),
            Region::Rectangle(_) => Err(value_error("not a real number".to_string())),
        }
    }

    /// Halve the isolating interval of a real number.
    pub fn refine_real(&self) -> PolyResult<()> {
        let (low, high) = self.real_interval()?;
        if low != high {
            let width: Rat = &high - &low;
            self.set_region(Region::Interval(dup_refine_real_root(
                &self.rational_minpoly,
                low,
                high,
                &width,
            )));
        }
        Ok(())
    }

    /// A box with rational corners and sides shorter than `eps` containing
    /// the number (`ValueError` unless `eps > 0`).
    pub fn enclosure(&self, eps: &Rat) -> PolyResult<Box4> {
        if *eps <= Rat::ZERO {
            return Err(value_error("eps must be positive".to_string()));
        }
        self.box_of(eps)
    }

    /// The real and imaginary parts as floats (Python's `complex()`).
    pub fn to_complex(&self) -> PolyResult<(f64, f64)> {
        let b: Box4 = self.box_of(&rat_from_parts(Int::ONE, Int::from(10u8).pow(18)))?;
        let two: Rat = rat(2, 1);
        let re: f64 = ((&b.0 + &b.1) / &two).to_f64().value();
        let im: f64 = ((&b.2 + &b.3) / &two).to_f64().value();
        Ok((re, im))
    }

    /// Python's `repr`: `Algebraic([1, 0, -2], 1.414213562)`.
    pub fn repr(&self) -> PolyResult<String> {
        let shown: String = if self.is_rational() {
            self.rational_value()?.to_string()
        } else {
            let (re, im) = self.to_complex()?;
            if self.is_real() {
                format_g10(re)
            } else {
                format!(
                    "{} {} {}*I",
                    format_g10(re),
                    if im > 0.0 { '+' } else { '-' },
                    format_g10(im.abs())
                )
            }
        };
        let coefficients: Vec<String> = self.minpoly.iter().map(|c| c.to_string()).collect();
        Ok(format!("Algebraic([{}], {})", coefficients.join(", "), shown))
    }

    // -- unary operations

    pub fn neg(&self) -> Algebraic {
        let n: usize = self.degree();
        let flipped: Vec<Int> = self
            .minpoly
            .iter()
            .enumerate()
            .map(|(k, c)| if (n - k) % 2 == 0 { c.clone() } else { -c })
            .collect();
        let minpoly: Vec<Int> = normalised(&flipped);
        match self.region() {
            Region::Interval((low, high)) => Algebraic::new(minpoly, Region::Interval((-high, -low))),
            Region::Rectangle(r) => Algebraic::new(
                minpoly,
                Region::Rectangle(Rectangle::new(-&r.re_max, -&r.re_min, -&r.im_max, -&r.im_min).expect("a rectangle")),
            ),
        }
    }

    /// The complex conjugate.
    pub fn conjugate(&self) -> Algebraic {
        match self.region() {
            Region::Interval(_) => self.clone(),
            Region::Rectangle(r) => Algebraic::new(self.minpoly.clone(), Region::Rectangle(r.conjugate())),
        }
    }

    /// `1/self` (`ZeroDivisionError` for zero).
    pub fn inverse(&self) -> PolyResult<Algebraic> {
        if self.is_zero() {
            return Err(zero_division("the inverse of zero"));
        }
        if self.is_rational() {
            return Ok(Algebraic::from_rational(&recip(&self.rational_value()?)));
        }
        let mut reversed: Vec<Int> = self.minpoly.clone();
        reversed.reverse();
        let reversed: Vec<Int> = normalised(&reversed);
        select(&[reversed], |eps| Ok(box_inverse(&self.box_of(eps)?)))
    }

    // -- binary operations

    pub fn add(&self, b: &Algebraic) -> PolyResult<Algebraic> {
        if self.is_rational() && b.is_rational() {
            return Ok(Algebraic::from_rational(
                &(self.rational_value()? + b.rational_value()?),
            ));
        }
        let resultant: Vec<Int> = sum_resultant(&self.minpoly, &b.minpoly);
        select(&irreducible_factors(&resultant)?, |eps| {
            Ok(Some(box_sum(&self.box_of(eps)?, &b.box_of(eps)?)))
        })
    }

    pub fn sub(&self, b: &Algebraic) -> PolyResult<Algebraic> {
        self.add(&b.neg())
    }

    pub fn mul(&self, b: &Algebraic) -> PolyResult<Algebraic> {
        if self.is_zero() || b.is_zero() {
            return Ok(Algebraic::from_rational(&Rat::ZERO));
        }
        if std::ptr::eq(self, b) {
            return self.pow(2);
        }
        if self.is_rational() && b.is_rational() {
            return Ok(Algebraic::from_rational(
                &(self.rational_value()? * b.rational_value()?),
            ));
        }
        let resultant: Vec<Int> = product_resultant(&self.minpoly, &b.minpoly);
        select(&irreducible_factors(&resultant)?, |eps| {
            Ok(Some(box_product(&self.box_of(eps)?, &b.box_of(eps)?)))
        })
    }

    pub fn div(&self, b: &Algebraic) -> PolyResult<Algebraic> {
        self.mul(&b.inverse()?)
    }

    pub fn pow(&self, exponent: i64) -> PolyResult<Algebraic> {
        if exponent < 0 {
            return self.inverse()?.pow(-exponent);
        }
        if exponent == 0 {
            return Ok(Algebraic::from_rational(&Rat::ONE));
        }
        if exponent == 1 {
            return Ok(self.clone());
        }
        if self.is_rational() {
            let v: Rat = self.rational_value()?;
            let e: usize = exponent as usize;
            return Ok(Algebraic::from_rational(&rat_from_parts(
                numer(&v).pow(e),
                denom(&v).pow(e),
            )));
        }
        // Res_y(m(y), x - y**n) vanishes at the n-th power and has the
        // degree of m (a product would square the degree)
        let n: usize = exponent as usize;
        let mut target: BiPoly = vec![vec![int(-1)]];
        target.extend(vec![Vec::new(); n - 1]);
        target.push(vec![Int::ONE, Int::ZERO]);
        let resultant: Vec<Int> = dup_resultant(&in_y(&self.minpoly), &target, &zx());
        select(&irreducible_factors(&resultant)?, |eps| {
            let base: Box4 = self.box_of(eps)?;
            let mut result: Box4 = base.clone();
            for _ in 0..n - 1 {
                result = box_product(&result, &base);
            }
            Ok(Some(result))
        })
    }

    /// `g(self)` for the polynomial `g` with these rational coefficients
    /// (highest degree first), through one resultant
    /// `Res_y(m(y), D x - G(y))`.
    pub fn polynomial_value(&self, coefficients: &[Rat]) -> PolyResult<Algebraic> {
        let start: usize = coefficients
            .iter()
            .position(|c| !c.is_zero())
            .unwrap_or(coefficients.len());
        let g: Vec<Rat> = coefficients[start..].to_vec();
        if g.len() <= 1 || self.is_rational() {
            let point: Rat = if self.is_rational() {
                self.rational_value()?
            } else {
                Rat::ZERO
            };
            let mut value: Rat = Rat::ZERO;
            for c in &g {
                value = value * &point + c;
            }
            return Ok(Algebraic::from_rational(&value));
        }
        let mut common: Int = Int::ONE;
        for c in &g {
            common = lcm(&common, &denom(c));
        }
        let scale: Rat = rat_from_int(&common);
        let integers: Vec<Int> = g.iter().map(|c| numer(&(c * &scale))).collect();
        let mut target: BiPoly = integers[..integers.len() - 1]
            .iter()
            .map(|c| if c.is_zero() { Vec::new() } else { vec![-c] })
            .collect();
        target.push(vec![common.clone(), -&integers[integers.len() - 1]]);
        let resultant: Vec<Int> = dup_resultant(&in_y(&self.minpoly), &target, &zx());
        select(&irreducible_factors(&resultant)?, |eps| {
            let base: Box4 = self.box_of(eps)?;
            let mut result: Box4 = (g[0].clone(), g[0].clone(), Rat::ZERO, Rat::ZERO);
            for c in &g[1..] {
                result = box_product(&result, &base);
                result = (&result.0 + c, &result.1 + c, result.2, result.3);
            }
            Ok(Some(result))
        })
    }

    /// The principal `n`-th root: the root of `z**n = self` whose argument
    /// is in `(-pi/n, pi/n]` (`ValueError` for `n < 1`).
    pub fn nth_root(&self, n: usize) -> PolyResult<Algebraic> {
        if n < 1 {
            return Err(value_error("the index of the root must be positive".to_string()));
        }
        if n == 1 || self.is_zero() {
            return Ok(self.clone());
        }
        let mut expanded: Vec<Int> = Vec::new();
        for c in &self.minpoly[..self.minpoly.len() - 1] {
            expanded.push(c.clone());
            expanded.extend(vec![Int::ZERO; n - 1]);
        }
        expanded.push(self.minpoly[self.minpoly.len() - 1].clone());
        let (real_roots, other_roots) = roots(&expanded)?;
        let mut candidates: Vec<Algebraic> = real_roots;
        candidates.extend(other_roots);
        let mut eps: Rat = rat(1, 4);
        while candidates.len() > n {
            let target: Box4 = self.box_of(&eps)?;
            let mut kept: Vec<Algebraic> = Vec::new();
            for candidate in candidates {
                let base: Box4 = candidate.box_of(&eps)?;
                let mut result: Box4 = base.clone();
                for _ in 0..n - 1 {
                    result = box_product(&result, &base);
                }
                if !disjoint(&result, &target) {
                    kept.push(candidate);
                }
            }
            candidates = kept;
            eps /= rat(16, 1);
        }
        if candidates.len() != n {
            return Err(arithmetic_error(format!(
                "{} roots of index {} were found",
                candidates.len(),
                n
            )));
        }
        let mut best: Algebraic = candidates[0].clone();
        for candidate in &candidates[1..] {
            best = larger_real_part(&best, candidate)?;
        }
        Ok(best)
    }

    /// The principal square root.
    pub fn sqrt(&self) -> PolyResult<Algebraic> {
        self.nth_root(2)
    }

    /// `(z + conjugate(z))/2`.
    pub fn real_part(&self) -> PolyResult<Algebraic> {
        if self.is_real() {
            return Ok(self.clone());
        }
        self.add(&self.conjugate())?.mul(&Algebraic::from_rational(&rat(1, 2)))
    }

    /// `(z - conjugate(z))/(2 i)`, a real number.
    pub fn imaginary_part(&self) -> PolyResult<Algebraic> {
        if self.is_real() {
            return Ok(Algebraic::from_rational(&Rat::ZERO));
        }
        self.sub(&self.conjugate())?
            .mul(&Algebraic::from_rational(&rat(1, 2)))?
            .mul(&Algebraic::imaginary_unit().neg())
    }

    // -- decisions

    /// Whether the two numbers are equal.
    pub fn equals(&self, other: &Algebraic) -> PolyResult<bool> {
        if self.minpoly != other.minpoly || self.is_real() != other.is_real() {
            return Ok(false);
        }
        if self.is_rational() {
            return Ok(true);
        }
        let mut eps: Rat = rat(1, 4);
        loop {
            let (p, q) = (self.box_of(&eps)?, other.box_of(&eps)?);
            if disjoint(&p, &q) {
                return Ok(false);
            }
            let hull: Box4 = (
                std::cmp::min(p.0.clone(), q.0.clone()),
                std::cmp::max(p.1.clone(), q.1.clone()),
                std::cmp::min(p.2.clone(), q.2.clone()),
                std::cmp::max(p.3.clone(), q.3.clone()),
            );
            if roots_in(&self.minpoly, &hull, &eps)? == Some(1) {
                return Ok(true);
            }
            eps /= rat(16, 1);
        }
    }

    /// Whether the number is the rational `q`.
    pub fn equals_rational(&self, q: &Rat) -> PolyResult<bool> {
        Ok(self.is_rational() && self.rational_value()? == *q)
    }

    /// The sign of a real number.
    pub fn sign(&self) -> PolyResult<i32> {
        self.real_interval()?;
        if self.is_zero() {
            return Ok(0);
        }
        let mut eps: Rat = rat(1, 4);
        loop {
            let (low, high, _, _) = self.box_of(&eps)?;
            if low > Rat::ZERO || (low.is_zero() && high > Rat::ZERO) {
                return Ok(1);
            }
            if high < Rat::ZERO || (high.is_zero() && low < Rat::ZERO) {
                return Ok(-1);
            }
            eps /= rat(16, 1);
        }
    }

    /// `-1`, `0` or `1`; `Value` (Python's `TypeError`) unless both are real.
    pub fn compare(&self, b: &Algebraic) -> PolyResult<i32> {
        if !(self.is_real() && b.is_real()) {
            return Err(value_error("complex numbers are not ordered".to_string()));
        }
        if self.equals(b)? {
            return Ok(0);
        }
        let mut eps: Rat = rat(1, 4);
        loop {
            let (p, q) = (self.box_of(&eps)?, b.box_of(&eps)?);
            if p.1 < q.0 {
                return Ok(-1);
            }
            if q.1 < p.0 {
                return Ok(1);
            }
            if p.1 == q.0 && p.0 < q.1 {
                // open intervals (or a point and an interval) which touch
                return Ok(-1);
            }
            if q.1 == p.0 && q.0 < p.1 {
                return Ok(1);
            }
            eps /= rat(16, 1);
        }
    }
}

/// The one of the two numbers with the larger real part; when the real
/// parts are equal, the one with the larger imaginary part.
fn larger_real_part(a: &Algebraic, b: &Algebraic) -> PolyResult<Algebraic> {
    let mut eps: Rat = rat(1, 4);
    let limit: Rat = rat_from_parts(Int::ONE, Int::ONE << 64usize);
    while eps > limit {
        let (p, q) = (a.box_of(&eps)?, b.box_of(&eps)?);
        if p.0 > q.1 {
            return Ok(a.clone());
        }
        if q.0 > p.1 {
            return Ok(b.clone());
        }
        eps /= rat(16, 1);
    }
    // close real parts: conjugate numbers have the same one; otherwise they
    // are compared exactly
    if !a.is_real() && !b.is_real() && a.conjugate().equals(b)? {
        return Ok(if above(a)? { a.clone() } else { b.clone() });
    }
    let (ra, rb) = (a.real_part()?, b.real_part()?);
    if !ra.equals(&rb)? {
        return Ok(if ra.compare(&rb)? > 0 { a.clone() } else { b.clone() });
    }
    Ok(if a.imaginary_part()?.compare(&b.imaginary_part()?)? > 0 {
        a.clone()
    } else {
        b.clone()
    })
}

/// The distinct irreducible factors of positive degree.
fn irreducible_factors(f: &[Int]) -> PolyResult<Vec<Vec<Int>>> {
    let (_, factors) = dup_zz_factor(f, &ZZ)?;
    Ok(factors.iter().map(|(g, _)| normalised(g)).collect())
}

/// The number of roots of `g` in the box widened by `pad` (or a little more
/// when a root lies on its boundary); `None` when every widening tried has a
/// root on its boundary.
fn roots_in(g: &[Int], b: &Box4, pad: &Rat) -> PolyResult<Option<i64>> {
    for k in 0..4i64 {
        let margin: Rat = pad * rat(4 + k, 4);
        let rectangle: Rectangle = Rectangle::new(&b.0 - &margin, &b.1 + &margin, &b.2 - &margin, &b.3 + &margin)?;
        if let Some(count) = winding_number(g, &rectangle)? {
            return Ok(Some(count));
        }
    }
    Ok(None)
}

/// The root of one of the irreducible `factors` which the boxes
/// `enclosure(eps)` contain for every `eps`.
fn select<F>(factors: &[Vec<Int>], enclosure: F) -> PolyResult<Algebraic>
where
    F: Fn(&Rat) -> PolyResult<Option<Box4>>,
{
    let mut eps: Rat = rat(1, 4);
    let sixteen: Rat = rat(16, 1);
    loop {
        let Some(b) = enclosure(&eps)? else {
            eps = &eps / &sixteen;
            continue;
        };
        let mut found: Vec<(&Vec<Int>, Option<i64>)> = Vec::new();
        for g in factors {
            let count: Option<i64> = roots_in(g, &b, &eps)?;
            if count != Some(0) {
                found.push((g, count));
            }
        }
        if found.len() != 1 || found[0].1 != Some(1) {
            eps = &eps / &sixteen;
            continue;
        }
        let g: &Vec<Int> = found[0].0;
        if g.len() == 2 {
            return Ok(Algebraic::from_rational(&rat_from_parts(-&g[1], g[0].clone())));
        }
        if &b.2 - &eps > Rat::ZERO || &b.3 + &eps < Rat::ZERO {
            // away from the real axis; the rectangle which was counted
            let mut chosen: Option<Rectangle> = None;
            for k in 0..4i64 {
                let margin: Rat = &eps * rat(4 + k, 4);
                let rectangle: Rectangle =
                    Rectangle::new(&b.0 - &margin, &b.1 + &margin, &b.2 - &margin, &b.3 + &margin)?;
                if winding_number(g, &rectangle)? == Some(1)
                    && (rectangle.im_min > Rat::ZERO || rectangle.im_max < Rat::ZERO)
                {
                    chosen = Some(rectangle);
                    break;
                }
            }
            if let Some(rectangle) = chosen {
                return Ok(Algebraic::complex(g.clone(), rectangle));
            }
            eps = &eps / &sixteen;
            continue;
        }
        // the box meets the real axis: real exactly when the symmetric box
        // still has one root
        let height: Rat = std::cmp::max(rat_abs(&b.2), rat_abs(&b.3)) + &eps * rat(2, 1);
        let symmetric: Box4 = (b.0.clone(), b.1.clone(), -&height, height.clone());
        if roots_in(g, &symmetric, &eps)? == Some(1) {
            for k in 0..4i64 {
                let margin: Rat = &eps * rat(4 + k, 4);
                let rectangle: Rectangle =
                    Rectangle::new(&b.0 - &margin, &b.1 + &margin, -&height - &margin, &height + &margin)?;
                if winding_number(g, &rectangle)? == Some(1) {
                    return Ok(Algebraic::real(g.clone(), (rectangle.re_min, rectangle.re_max)));
                }
            }
        }
        eps = &eps / &sixteen;
    }
}

fn rat_abs(v: &Rat) -> Rat {
    if *v < Rat::ZERO {
        -v
    } else {
        v.clone()
    }
}

/// The real roots and the other roots, in no particular order.
fn roots(f: &[Int]) -> PolyResult<(Vec<Algebraic>, Vec<Algebraic>)> {
    if f.iter().all(|c| c.is_zero()) {
        return Err(value_error("the zero polynomial".to_string()));
    }
    let mut real: Vec<Algebraic> = Vec::new();
    let mut others: Vec<Algebraic> = Vec::new();
    for g in irreducible_factors(f)? {
        if g.len() == 2 {
            real.push(Algebraic::from_rational(&rat_from_parts(-&g[1], g[0].clone())));
            continue;
        }
        let rational: Vec<Rat> = g.iter().map(rat_from_int).collect();
        for interval in dup_isolate_real_roots_sqf(&rational, None) {
            real.push(Algebraic::real(g.clone(), interval));
        }
        for rectangle in dup_isolate_complex_roots_sqf(&rational, None)? {
            others.push(Algebraic::complex(g.clone(), rectangle.conjugate()));
            others.push(Algebraic::complex(g.clone(), rectangle));
        }
    }
    Ok((real, others))
}

/// Whether a number which is not real is above the real axis.
fn above(a: &Algebraic) -> PolyResult<bool> {
    let mut eps: Rat = rat(1, 4);
    loop {
        let b: Box4 = a.box_of(&eps)?;
        if b.2 > Rat::ZERO {
            return Ok(true);
        }
        if b.3 < Rat::ZERO {
            return Ok(false);
        }
        eps /= rat(16, 1);
    }
}

/// The sorting key of the other roots: by real part, then by imaginary
/// part (`_ComplexKey.__lt__`).
fn complex_less(a: &Algebraic, b: &Algebraic) -> PolyResult<bool> {
    let mut eps: Rat = rat(1, 4);
    let floor: Rat = rat_from_parts(Int::ONE, Int::ONE << 200usize);
    let close: Rat = rat_from_parts(Int::ONE, Int::ONE << 40usize);
    while eps > floor {
        let (p, q) = (a.box_of(&eps)?, b.box_of(&eps)?);
        if p.1 < q.0 {
            return Ok(true);
        }
        if q.1 < p.0 {
            return Ok(false);
        }
        eps /= rat(16, 1);
        if eps < close {
            // the real parts may be equal: those of conjugate numbers are,
            // the others are compared exactly
            if !a.is_real() && !b.is_real() && a.conjugate().equals(b)? {
                return Ok(!above(a)?);
            }
            let (ra, rb) = (a.real_part()?, b.real_part()?);
            if !ra.equals(&rb)? {
                return Ok(ra.compare(&rb)? < 0);
            }
            return Ok(a.imaginary_part()?.compare(&b.imaginary_part()?)? < 0);
        }
    }
    Ok(false)
}

/// A stable insertion sort with a fallible strict order.
fn sort_by_less<F>(items: Vec<Algebraic>, less: F) -> PolyResult<Vec<Algebraic>>
where
    F: Fn(&Algebraic, &Algebraic) -> PolyResult<bool>,
{
    let mut sorted: Vec<Algebraic> = Vec::new();
    for item in items {
        let mut position: usize = sorted.len();
        while position > 0 && less(&item, &sorted[position - 1])? {
            position -= 1;
        }
        sorted.insert(position, item);
    }
    Ok(sorted)
}

/// The distinct complex roots of a nonzero polynomial with integer
/// coefficients (highest degree first): the real roots in increasing order,
/// then the others by real part, the root below the real axis before its
/// conjugate.
pub fn algebraic_roots(f: &[Int]) -> PolyResult<Vec<Algebraic>> {
    let (real, others) = roots(f)?;
    let mut result: Vec<Algebraic> = sort_by_less(real, |a, b| Ok(a.compare(b)? < 0))?;
    result.extend(sort_by_less(others, complex_less)?);
    Ok(result)
}

/// A primitive element of the field generated by two algebraic numbers:
/// `(theta, p, q)` with `theta = a + c b` for a small integer `c`, and
/// polynomials with rational coefficients (highest degree first) such that
/// `a = p(theta)` and `b = q(theta)`.
pub fn primitive_element(a: &Algebraic, b: &Algebraic) -> PolyResult<(Algebraic, Vec<Rat>, Vec<Rat>)> {
    if b.is_rational() {
        return Ok((a.clone(), vec![Rat::ONE, Rat::ZERO], vec![b.rational_value()?]));
    }
    if a.is_rational() {
        return Ok((b.clone(), vec![a.rational_value()?], vec![Rat::ONE, Rat::ZERO]));
    }
    let ma: Vec<Rat> = a.minpoly.iter().map(rat_from_int).collect();
    let mb: Vec<Rat> = b.minpoly.iter().map(rat_from_int).collect();
    let theta_symbol = Symbol::new("theta", &[]).0;
    for c in 1..=(4 * a.degree() * b.degree() + 1) as i64 {
        let theta: Algebraic = a.add(&Algebraic::from_rational(&rat(c, 1)).mul(b)?)?;
        let lc: Rat = rat_from_int(&theta.minpoly[0]);
        let k: AlgebraicField = AlgebraicField::new(
            theta.minpoly.iter().map(|v| rat_from_int(v) / &lc).collect(),
            theta_symbol.clone(),
        )?;
        // m_a(theta - c x) as a polynomial in x over QQ(theta), by Horner
        let mut shifted: Vec<AlgebraicElement> = Vec::new();
        let argument: [AlgebraicElement; 2] = [vec![rat(-c, 1)], k.alpha()];
        for coefficient in &ma {
            let mut product: Vec<AlgebraicElement> = if shifted.is_empty() {
                Vec::new()
            } else {
                vec![k.zero(); shifted.len() + 1]
            };
            for (i, u) in shifted.iter().enumerate() {
                product[i] = k.add(&product[i], &k.mul(u, &argument[0]));
                product[i + 1] = k.add(&product[i + 1], &k.mul(u, &argument[1]));
            }
            if product.is_empty() {
                if !coefficient.is_zero() {
                    product = vec![k.from_rational(coefficient)];
                }
            } else {
                let last: usize = product.len() - 1;
                product[last] = k.add(&product[last], &k.from_rational(coefficient));
            }
            shifted = product;
        }
        let other: Vec<AlgebraicElement> = mb.iter().map(|v| k.from_rational(v)).collect();
        let common: Vec<AlgebraicElement> = dup_gcd(&shifted, &other, &k);
        if common.len() != 2 {
            continue;
        }
        let q: AlgebraicElement = k.neg(&k.quo(&common[1], &common[0])?);
        let p: AlgebraicElement = k.sub(&k.alpha(), &k.mul(&k.from_int(c), &q));
        return Ok((theta, p, q));
    }
    Err(arithmetic_error(
        "no primitive element of the form a + c b was found".to_string(),
    ))
}

/// `PolyError` for the comparison of numbers which are not both real.
pub fn is_unordered(e: &PolyError) -> bool {
    matches!(e, PolyError::Value(m) if m == "complex numbers are not ordered")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: &[i64]) -> Vec<Int> {
        v.iter().map(|c| int(*c)).collect()
    }

    #[test]
    fn docstring_examples() {
        let roots2 = algebraic_roots(&z(&[1, 0, -2])).unwrap();
        let roots3 = algebraic_roots(&z(&[1, 0, -3])).unwrap();
        let (sqrt2, sqrt3) = (&roots2[1], &roots3[1]);
        let s = sqrt2.add(sqrt3).unwrap();
        assert_eq!(s.repr().unwrap(), "Algebraic([1, 0, -10, 0, 1], 3.14626437)");
        let lhs = s.mul(&s).unwrap();
        let rhs = Algebraic::from_rational(&rat(5, 1))
            .add(
                &Algebraic::from_rational(&rat(2, 1))
                    .mul(sqrt2)
                    .unwrap()
                    .mul(sqrt3)
                    .unwrap(),
            )
            .unwrap();
        assert!(lhs.equals(&rhs).unwrap());
        let two = sqrt2.mul(sqrt2).unwrap();
        assert_eq!(two.minpoly(), z(&[1, -2]));
        assert!(sqrt3.sub(sqrt2).unwrap().inverse().unwrap().equals(&s).unwrap());
        let i = Algebraic::imaginary_unit();
        let one_i = Algebraic::from_rational(&Rat::ONE).add(&i).unwrap();
        assert_eq!(one_i.pow(8).unwrap().repr().unwrap(), "Algebraic([1, -16], 16)");
        let cube = algebraic_roots(&z(&[1, 0, 0, -2])).unwrap();
        let shown: Vec<String> = cube.iter().map(|r| r.repr().unwrap()).collect();
        assert_eq!(
            shown,
            vec![
                "Algebraic([1, 0, 0, -2], 1.25992105)",
                "Algebraic([1, 0, 0, -2], -0.6299605249 - 1.091123636*I)",
                "Algebraic([1, 0, 0, -2], -0.6299605249 + 1.091123636*I)",
            ]
        );
        assert!(cube[0].add(&cube[1]).unwrap().add(&cube[2]).unwrap().is_zero());
        assert!(cube[1].mul(&cube[2]).unwrap().is_real());
        assert!(cube[1].conjugate().equals(&cube[2]).unwrap());
        let eight = Algebraic::from_rational(&rat(-8, 1)).nth_root(3).unwrap();
        assert_eq!(eight.repr().unwrap(), "Algebraic([1, -2, 4], 1 + 1.732050808*I)");
        let (theta, p, q) = primitive_element(sqrt2, sqrt3).unwrap();
        assert_eq!(theta.repr().unwrap(), "Algebraic([1, 0, -10, 0, 1], 3.14626437)");
        assert_eq!(p, vec![rat(1, 2), rat(0, 1), rat(-9, 2), rat(0, 1)]);
        assert_eq!(q, vec![rat(-1, 2), rat(0, 1), rat(11, 2), rat(0, 1)]);
        assert!(theta.polynomial_value(&p).unwrap().equals(sqrt2).unwrap());
        assert_eq!(format_g10(1.0e-7), "1e-07");
        assert_eq!(format_g10(12345678901.0), "1.23456789e+10");
    }
}
