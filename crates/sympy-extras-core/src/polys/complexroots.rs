//! Port of `sympy_extras/polys/complexroots.py`: isolation and counting of
//! the complex roots of dense polynomials with rational coefficients, by
//! exact winding numbers.
//!
//! The number of roots inside a rectangle with no root on its boundary is
//! the winding number of `f` along the boundary, `-1/2` the sum over the
//! edges of the Cauchy indices of `v/u` (`c f(z_0 + t w) = u(t) + i v(t)`),
//! counted exactly by Sturm sequences of integer polynomials; a root on an
//! edge is detected and the rectangle refused. Isolation bisects a
//! rectangle above a root-free strip around the real axis; refinement
//! accepts the small square proposed by Newton's iteration when Rouché's
//! theorem (or the winding number) proves it. The same Cauchy index on a
//! line counts the roots on one side (Routh–Hurwitz); a disc is the image
//! of a half-plane (Schur–Cohn).
//!
//! References
//! ==========
//!
//! 1. H. S. Wilf, *A global bisection algorithm for computing the zeros of
//!    polynomials in the complex plane*, Journal of the ACM 25 (1978),
//!    415-420.
//! 2. S. Basu, R. Pollack, M.-F. Roy, *Algorithms in Real Algebraic
//!    Geometry*, 2nd ed., Springer, 2006, Section 2.2.2.
//! 3. P. Henrici, *Applied and Computational Complex Analysis*, vol. 1,
//!    Wiley, 1974, Section 6.11.

use std::fmt;

use crate::error::{arithmetic_error, value_error, PolyError, PolyResult};
use crate::numbers::*;

use super::densebasic::dup_degree;
use super::densetools::dup_clear_denoms;
use super::domains::QQ;
use super::rootisolation::{dup_count_real_roots, dup_fujiwara_bound, dup_isolate_real_roots_sqf, Interval};
use super::sqfreetools::{dup_sqf_list, dup_sqf_p};

/// The constants `c = a + b i` tried so that `Re(c f)` does not vanish at
/// the corners: pairwise non-parallel, and a corner excludes one direction,
/// so that one of five suits four corners.
const DIRECTIONS: [(i64, i64); 5] = [(1, 0), (1, 1), (1, -1), (1, 2), (2, 1)];

/// The rectangle `[re_min, re_max] x [im_min, im_max]` of the complex plane,
/// with rational corners.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rectangle {
    pub re_min: Rat,
    pub re_max: Rat,
    pub im_min: Rat,
    pub im_max: Rat,
}

impl Rectangle {
    /// `ValueError` for an empty rectangle.
    pub fn new(re_min: Rat, re_max: Rat, im_min: Rat, im_max: Rat) -> PolyResult<Rectangle> {
        if !(re_min < re_max && im_min < im_max) {
            return Err(value_error("an empty rectangle".to_string()));
        }
        Ok(Rectangle {
            re_min,
            re_max,
            im_min,
            im_max,
        })
    }

    /// A rectangle known not to be empty.
    fn of(re_min: Rat, re_max: Rat, im_min: Rat, im_max: Rat) -> Rectangle {
        Rectangle::new(re_min, re_max, im_min, im_max).expect("a non-empty rectangle")
    }

    pub fn width(&self) -> Rat {
        &self.re_max - &self.re_min
    }

    pub fn height(&self) -> Rat {
        &self.im_max - &self.im_min
    }

    /// The real and imaginary parts of the centre.
    pub fn center(&self) -> (Rat, Rat) {
        let two: Rat = rat(2, 1);
        (
            (&self.re_min + &self.re_max) / &two,
            (&self.im_min + &self.im_max) / &two,
        )
    }

    /// The mirror image in the real axis.
    pub fn conjugate(&self) -> Rectangle {
        Rectangle::of(self.re_min.clone(), self.re_max.clone(), -&self.im_max, -&self.im_min)
    }

    /// Whether the point `re + im i` is in the closed rectangle.
    pub fn contains(&self, re: &Rat, im: &Rat) -> bool {
        self.re_min <= *re && *re <= self.re_max && self.im_min <= *im && *im <= self.im_max
    }
}

impl fmt::Display for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rectangle({}, {}, {}, {})",
            self.re_min, self.re_max, self.im_min, self.im_max
        )
    }
}

// ---------------------------------------------------------------------------
// integer polynomials (highest degree first)

fn strip(f: &[Int]) -> Vec<Int> {
    let start: usize = f.iter().position(|c| !c.is_zero()).unwrap_or(f.len());
    f[start..].to_vec()
}

/// `f` divided by the (positive) gcd of its coefficients.
fn primitive(f: Vec<Int>) -> Vec<Int> {
    let mut g: Int = Int::ZERO;
    for c in &f {
        g = gcd(&g, c);
        if g == Int::ONE {
            return f;
        }
    }
    if g > Int::ONE {
        f.iter().map(|c| div_floor(c, &g)).collect()
    } else {
        f
    }
}

/// A positive multiple of `-rem(a, b)`, primitive.
fn negated_remainder(a: &[Int], b: &[Int]) -> Vec<Int> {
    let (da, db) = (a.len() - 1, b.len() - 1);
    if da < db {
        return a.iter().map(|c| -c).collect();
    }
    let lc: &Int = &b[0];
    let scale: Int = abs(lc).pow(da - db + 1);
    let mut r: Vec<Int> = a.iter().map(|c| &scale * c).collect();
    for i in 0..=da - db {
        let q: Int = div_floor(&r[i], lc);
        if !q.is_zero() {
            for j in 1..=db {
                let t: Int = &q * &b[j];
                r[i + j] -= t;
            }
        }
    }
    let rest: Vec<Int> = strip(&r[da - db + 1..]).iter().map(|c| -c).collect();
    primitive(rest)
}

/// `u, v, -rem(u, v), ...` up to positive factors, without the final zero:
/// the last polynomial is the gcd.
fn signed_remainder_sequence(u: &[Int], v: &[Int]) -> Vec<Vec<Int>> {
    let mut sequence: Vec<Vec<Int>> = vec![primitive(u.to_vec()), primitive(v.to_vec())];
    while !sequence[sequence.len() - 1].is_empty() {
        let n: usize = sequence.len();
        let next: Vec<Int> = negated_remainder(&sequence[n - 2], &sequence[n - 1]);
        sequence.push(next);
    }
    sequence.pop();
    sequence
}

fn sum(g: &[Int]) -> Int {
    let mut total: Int = Int::ZERO;
    for c in g {
        total += c;
    }
    total
}

/// The sign variations of the values at 1 (the sums of the coefficients) or
/// at 0 (the constant terms).
fn variations(sequence: &[Vec<Int>], at_one: bool) -> i64 {
    let mut count: i64 = 0;
    let mut previous: i32 = 0;
    for g in sequence {
        let value: Int = if at_one { sum(g) } else { g[g.len() - 1].clone() };
        if value.is_zero() {
            continue;
        }
        let s: i32 = if value > Int::ZERO { 1 } else { -1 };
        if previous != 0 && s != previous {
            count += 1;
        }
        previous = s;
    }
    count
}

fn derivative(g: &[Int]) -> Vec<Int> {
    let n: usize = g.len() - 1;
    g[..n].iter().enumerate().map(|(i, c)| c * Int::from(n - i)).collect()
}

/// Whether `g` has a root in `[0, 1]`.
fn has_root_in_unit_interval(g: &[Int]) -> bool {
    if g.len() <= 1 {
        return false;
    }
    if g[g.len() - 1].is_zero() || sum(g).is_zero() {
        return true;
    }
    let sequence: Vec<Vec<Int>> = signed_remainder_sequence(g, &derivative(g));
    variations(&sequence, false) != variations(&sequence, true)
}

/// The real and imaginary parts of `sum a_k scale**k Z**(n - k)` at
/// `Z = z0 + t w` (Gaussian integers), as polynomials in `t`, lowest degree
/// first: `scale**n f((z0 + t w)/scale)` by Horner's rule.
fn on_segment(a: &[Int], scale: &Int, z0: &(Int, Int), w: &(Int, Int)) -> (Vec<Int>, Vec<Int>) {
    let (x0, y0) = z0;
    let (wx, wy) = w;
    let mut re: Vec<Int> = vec![a[0].clone()];
    let mut im: Vec<Int> = vec![Int::ZERO];
    let mut power: Int = Int::ONE;
    for c in &a[1..] {
        power *= scale;
        let size: usize = re.len();
        let mut new_re: Vec<Int> = vec![Int::ZERO; size + 1];
        let mut new_im: Vec<Int> = vec![Int::ZERO; size + 1];
        for j in 0..size {
            let (r, s) = (&re[j], &im[j]);
            new_re[j] += r * x0 - s * y0;
            new_im[j] += r * y0 + s * x0;
            new_re[j + 1] += r * wx - s * wy;
            new_im[j + 1] += r * wy + s * wx;
        }
        new_re[0] += c * &power;
        re = new_re;
        im = new_im;
    }
    (re, im)
}

/// `r * scale` for a rational whose denominator divides `scale`.
fn scaled(r: &Rat, scale: &Int) -> Int {
    numer(&(r * rat_from_int(scale)))
}

/// The number of roots of the integer polynomial `a` inside the rectangle;
/// `None` when a root lies on the boundary.
pub(crate) fn winding_number(a: &[Int], rectangle: &Rectangle) -> PolyResult<Option<i64>> {
    let (x0, x1, y0, y1) = (
        &rectangle.re_min,
        &rectangle.re_max,
        &rectangle.im_min,
        &rectangle.im_max,
    );
    let scale: Int = lcm(&lcm(&denom(x0), &denom(x1)), &lcm(&denom(y0), &denom(y1)));
    let (xa, xb, ya, yb) = (
        scaled(x0, &scale),
        scaled(x1, &scale),
        scaled(y0, &scale),
        scaled(y1, &scale),
    );
    // counter-clockwise from the lower left corner
    let segments: [((Int, Int), (Int, Int)); 4] = [
        ((xa.clone(), ya.clone()), (&xb - &xa, Int::ZERO)),
        ((xb.clone(), ya.clone()), (Int::ZERO, &yb - &ya)),
        ((xb.clone(), yb.clone()), (&xa - &xb, Int::ZERO)),
        ((xa.clone(), yb.clone()), (Int::ZERO, &ya - &yb)),
    ];
    let edges: Vec<(Vec<Int>, Vec<Int>)> = segments.iter().map(|(z0, w)| on_segment(a, &scale, z0, w)).collect();
    let direction: Option<(Int, Int)> = DIRECTIONS
        .iter()
        .map(|(ca, cb)| (Int::from(*ca), Int::from(*cb)))
        .find(|(ca, cb)| edges.iter().all(|(re, im)| !(ca * &re[0] - cb * &im[0]).is_zero()));
    let Some((ca, cb)) = direction else {
        // a corner is a root (four corners exclude four directions at most)
        return Ok(None);
    };
    let mut total: i64 = 0;
    for (re, im) in &edges {
        let mut u: Vec<Int> = re.iter().zip(im).map(|(r, s)| &ca * r - &cb * s).collect();
        let mut v: Vec<Int> = re.iter().zip(im).map(|(r, s)| &ca * s + &cb * r).collect();
        u.reverse();
        v.reverse();
        let (u, v) = (strip(&u), strip(&v));
        if v.is_empty() {
            // c f is real and without zero at the ends; its zeros on the
            // edge are those of u
            if has_root_in_unit_interval(&u) {
                return Ok(None);
            }
            continue;
        }
        let sequence: Vec<Vec<Int>> = signed_remainder_sequence(&u, &v);
        if has_root_in_unit_interval(&sequence[sequence.len() - 1]) {
            return Ok(None);
        }
        total += variations(&sequence, false) - variations(&sequence, true);
    }
    if total % 2 != 0 {
        return Err(arithmetic_error("odd total Cauchy index on a closed path".to_string()));
    }
    Ok(Some(-total / 2))
}

fn integer_coefficients(f: &[Rat]) -> PolyResult<Vec<Int>> {
    if dup_degree(f) < 0 {
        return Err(value_error("the zero polynomial".to_string()));
    }
    let (_, cleared) = dup_clear_denoms(f, &QQ);
    Ok(cleared.iter().map(numer).collect())
}

/// The number of complex roots of `f`, with multiplicities, in the interior
/// of the rectangle; `ValueError` when a root lies on the boundary.
pub fn dup_count_complex_roots(f: &[Rat], rectangle: &Rectangle) -> PolyResult<i64> {
    match winding_number(&integer_coefficients(f)?, rectangle)? {
        Some(count) => Ok(count),
        None => Err(value_error(format!("a root lies on the boundary of {}", rectangle))),
    }
}

/// Positions of a cut along a side, as fractions of its length: the middle
/// first, then `2 count` others in the middle half.
fn cuts(count: i64) -> Vec<Rat> {
    let half: Rat = rat(1, 2);
    let mut result: Vec<Rat> = vec![half.clone()];
    for j in 1..=count {
        let step: Rat = rat(j, 4 * (count + 1));
        result.push(&half + &step);
        result.push(&half - &step);
    }
    result
}

/// The rectangle (with `count` roots inside, none on the boundary) cut along
/// its longer side into two with no root on the cut, and their numbers of
/// roots.
fn split(a: &[Int], rectangle: &Rectangle, count: i64) -> PolyResult<(Rectangle, i64, Rectangle, i64)> {
    let (x0, x1, y0, y1) = (
        &rectangle.re_min,
        &rectangle.re_max,
        &rectangle.im_min,
        &rectangle.im_max,
    );
    for position in cuts(count) {
        let (first, second) = if rectangle.width() >= rectangle.height() {
            let middle: Rat = x0 + &position * (x1 - x0);
            (
                Rectangle::of(x0.clone(), middle.clone(), y0.clone(), y1.clone()),
                Rectangle::of(middle, x1.clone(), y0.clone(), y1.clone()),
            )
        } else {
            let middle: Rat = y0 + &position * (y1 - y0);
            (
                Rectangle::of(x0.clone(), x1.clone(), y0.clone(), middle.clone()),
                Rectangle::of(x0.clone(), x1.clone(), middle, y1.clone()),
            )
        };
        if let Some(inside) = winding_number(a, &first)? {
            return Ok((first, inside, second, count - inside));
        }
    }
    Err(arithmetic_error(format!("no cut of {} avoids the roots", rectangle)))
}

fn pow2(k: usize) -> Int {
    Int::ONE << k
}

/// Python's `int()` of a rational: truncated towards zero.
fn truncate(r: &Rat) -> Int {
    div_rem_trunc(&numer(r), &denom(r)).0
}

/// Newton's iteration from the centre of the rectangle in fixed point
/// arithmetic: the last iterate `(X + Y i)/2**precision`.
fn newton(a: &[Int], rectangle: &Rectangle, precision: usize) -> (Int, Int) {
    let n: usize = a.len() - 1;
    let (re, im) = rectangle.center();
    let unit: Rat = rat_from_int(&pow2(precision));
    let mut x: Int = truncate(&(re * &unit));
    let mut y: Int = truncate(&(im * &unit));
    let two: Int = Int::from(2u8);
    for _ in 0..precision + 30 {
        // G(Z) = 2**(precision n) f(Z/2**precision) and G'(Z) by Horner
        let (mut gr, mut gi, mut dr, mut di): (Int, Int, Int, Int) = (a[0].clone(), Int::ZERO, Int::ZERO, Int::ZERO);
        for (k, c) in a.iter().enumerate().skip(1).take(n) {
            let (ndr, ndi) = (&dr * &x - &di * &y + &gr, &dr * &y + &di * &x + &gi);
            dr = ndr;
            di = ndi;
            let (ngr, ngi) = (&gr * &x - &gi * &y + c * pow2(precision * k), &gr * &y + &gi * &x);
            gr = ngr;
            gi = ngi;
        }
        let norm: Int = &dr * &dr + &di * &di;
        if norm.is_zero() {
            break;
        }
        // the step G/G', rounded
        let step_x: Int = div_floor(&(&two * (&gr * &dr + &gi * &di) + &norm), &(&two * &norm));
        let step_y: Int = div_floor(&(&two * (&gi * &dr - &gr * &di) + &norm), &(&two * &norm));
        if step_x.is_zero() && step_y.is_zero() {
            break;
        }
        x -= step_x;
        y -= step_y;
    }
    (x, y)
}

/// A sufficient test (Rouché) that the square of half-side `4/2**precision`
/// centred at `(X + Y i)/2**precision` has exactly one root inside and none
/// on its boundary.
fn one_root_near(a: &[Int], x: &Int, y: &Int, precision: usize) -> bool {
    let n: usize = a.len() - 1;
    let mut re: Vec<Int> = a.iter().enumerate().map(|(k, c)| c * pow2(precision * k)).collect();
    let mut im: Vec<Int> = vec![Int::ZERO; n + 1];
    // Taylor shift by repeated synthetic division: B_k ends at index n - k
    for i in 0..n {
        for j in 1..=n - i {
            let (r, t) = (re[j - 1].clone(), im[j - 1].clone());
            re[j] += &r * x - &t * y;
            im[j] += &r * y + &t * x;
        }
    }
    let lower: Int = std::cmp::max(abs(&re[n - 1]), abs(&im[n - 1]));
    for radius in [4i64, 6] {
        let radius: Int = Int::from(radius);
        let mut rest: Int = abs(&re[n]) + abs(&im[n]);
        let mut power: Int = radius.clone();
        for k in 2..=n {
            power *= &radius;
            rest += (abs(&re[n - k]) + abs(&im[n - k])) * &power;
        }
        if &lower * &radius <= rest {
            return false;
        }
    }
    true
}

/// A rectangle with one root shrunk until its sides are below `eps`.
pub(crate) fn refine(a: &[Int], rectangle: Rectangle, eps: Option<&Rat>) -> PolyResult<Rectangle> {
    let Some(eps) = eps else {
        return Ok(rectangle);
    };
    let mut precision: usize = 8;
    let sixteenth: Rat = eps / rat(16, 1);
    while rat_from_parts(Int::ONE, pow2(precision)) > sixteenth {
        precision += 1;
    }
    let unit: Rat = rat_from_parts(Int::ONE, pow2(precision));
    let four: Rat = &unit * rat(4, 1);
    let mut rectangle: Rectangle = rectangle;
    while rectangle.width() >= *eps || rectangle.height() >= *eps {
        let (x_int, y_int) = newton(a, &rectangle, precision);
        let x: Rat = rat_from_int(&x_int) * &unit;
        let y: Rat = rat_from_int(&y_int) * &unit;
        if rectangle.re_min < &x - &four
            && &x + &four < rectangle.re_max
            && rectangle.im_min < &y - &four
            && &y + &four < rectangle.im_max
        {
            let square: Rectangle = Rectangle::of(&x - &four, &x + &four, &y - &four, &y + &four);
            if one_root_near(a, &x_int, &y_int, precision) || winding_number(a, &square)? == Some(1) {
                return Ok(square);
            }
        }
        let (first, inside, second, _) = split(a, &rectangle, 1)?;
        rectangle = if inside != 0 { first } else { second };
    }
    Ok(rectangle)
}

/// A rectangle inside the given one, with sides shorter than `eps`,
/// containing the same root: `rectangle` must contain exactly one root of
/// `f` in its interior and none on its boundary (`ValueError` otherwise).
pub fn dup_refine_complex_root(f: &[Rat], rectangle: &Rectangle, eps: &Rat) -> PolyResult<Rectangle> {
    if *eps <= Rat::ZERO {
        return Err(value_error("eps must be positive".to_string()));
    }
    let a: Vec<Int> = integer_coefficients(f)?;
    if winding_number(&a, rectangle)? != Some(1) {
        return Err(value_error(format!("{} does not isolate a root", rectangle)));
    }
    refine(&a, rectangle.clone(), Some(eps))
}

fn sort_by_lower_left(rectangles: &mut [Rectangle]) {
    rectangles.sort_by(|r, s| (&r.re_min, &r.im_min).cmp(&(&s.re_min, &s.im_min)));
}

/// Isolating rectangles of the roots with positive imaginary part of a
/// square-free `f` with rational coefficients, sorted by their lower left
/// corners; with `eps` their sides are shorter than `eps`.
pub fn dup_isolate_complex_roots_sqf(f: &[Rat], eps: Option<&Rat>) -> PolyResult<Vec<Rectangle>> {
    let n: isize = dup_degree(f);
    if n < 1 {
        return Ok(Vec::new());
    }
    if !dup_sqf_p(f, &QQ) {
        return Err(value_error("the polynomial is not square-free".to_string()));
    }
    if let Some(e) = eps {
        if *e <= Rat::ZERO {
            return Err(value_error("eps must be positive".to_string()));
        }
    }
    let pairs: i64 = (n as i64 - dup_count_real_roots(f, None, None)) / 2;
    if pairs == 0 {
        return Ok(Vec::new());
    }
    let a: Vec<Int> = integer_coefficients(f)?;
    let bound: Rat = rat_from_int(&(truncate(&dup_fujiwara_bound(f)) + Int::ONE));
    // a strip around the real axis without roots off the axis
    let mut low: Rat = bound.clone();
    let whole: Rectangle = loop {
        low /= rat(2, 1);
        let whole: Rectangle = Rectangle::of(-&bound, bound.clone(), low.clone(), bound.clone());
        if winding_number(&a, &whole)? == Some(pairs) {
            break whole;
        }
    };
    let mut pending: Vec<(Rectangle, i64)> = vec![(whole, pairs)];
    let mut isolated: Vec<Rectangle> = Vec::new();
    while let Some((rectangle, count)) = pending.pop() {
        if count == 1 {
            isolated.push(refine(&a, rectangle, eps)?);
            continue;
        }
        let (first, inside, second, outside) = split(&a, &rectangle, count)?;
        if inside != 0 {
            pending.push((first, inside));
        }
        if outside != 0 {
            pending.push((second, outside));
        }
    }
    sort_by_lower_left(&mut isolated);
    Ok(isolated)
}

/// The real roots of a square-free `f` by isolating intervals and the roots
/// above the real axis by isolating rectangles.
pub fn dup_isolate_all_roots_sqf(f: &[Rat], eps: Option<&Rat>) -> PolyResult<(Vec<Interval>, Vec<Rectangle>)> {
    Ok((
        dup_isolate_real_roots_sqf(f, eps),
        dup_isolate_complex_roots_sqf(f, eps)?,
    ))
}

/// Whether the closed rectangles meet.
fn overlap(r: &Rectangle, s: &Rectangle) -> bool {
    !(r.re_max < s.re_min || s.re_max < r.re_min || r.im_max < s.im_min || s.im_max < r.im_min)
}

/// Isolating rectangles of the roots with positive imaginary part of any
/// `f` with rational coefficients, with their multiplicities: the
/// square-free factors one by one, the rectangles of different factors
/// refined until they are disjoint.
pub fn dup_isolate_complex_roots(f: &[Rat], eps: Option<&Rat>) -> PolyResult<Vec<(Rectangle, usize)>> {
    if dup_degree(f) < 1 {
        return Ok(Vec::new());
    }
    let (_, factors) = dup_sqf_list(f, &QQ);
    // the rectangle, the multiplicity and the index of the factor
    let mut found: Vec<(Rectangle, usize, usize)> = Vec::new();
    let mut coefficients: Vec<Vec<Int>> = Vec::new();
    for (g, multiplicity) in &factors {
        if dup_degree(g) < 1 {
            continue;
        }
        let index: usize = coefficients.len();
        coefficients.push(integer_coefficients(g)?);
        for rectangle in dup_isolate_complex_roots_sqf(g, eps)? {
            found.push((rectangle, *multiplicity, index));
        }
    }
    loop {
        let mut clash: bool = false;
        for i in 0..found.len() {
            for j in i + 1..found.len() {
                let (r, s) = (found[i].0.clone(), found[j].0.clone());
                if found[i].2 != found[j].2 && overlap(&r, &s) {
                    clash = true;
                    let size: Rat = [r.width(), r.height(), s.width(), s.height()]
                        .into_iter()
                        .min()
                        .expect("four sides")
                        / rat(2, 1);
                    found[i].0 = refine(&coefficients[found[i].2], r, Some(&size))?;
                    found[j].0 = refine(&coefficients[found[j].2], s, Some(&size))?;
                }
            }
        }
        if !clash {
            break;
        }
    }
    let mut result: Vec<(Rectangle, usize)> = found.into_iter().map(|(r, k, _)| (r, k)).collect();
    result.sort_by(|x, y| (&x.0.re_min, &x.0.im_min).cmp(&(&y.0.re_min, &y.0.im_min)));
    Ok(result)
}

// ---------------------------------------------------------------------------
// half-planes and discs

/// The sign variations of the sequence at `+oo` (`sign = 1`) or `-oo`.
fn variations_at_infinity(sequence: &[Vec<Int>], sign: i32) -> i64 {
    let mut count: i64 = 0;
    let mut previous: i32 = 0;
    for g in sequence {
        let mut s: i32 = if g[0] > Int::ZERO { 1 } else { -1 };
        if sign < 0 && (g.len() - 1) % 2 == 1 {
            s = -s;
        }
        if previous != 0 && s != previous {
            count += 1;
        }
        previous = s;
    }
    count
}

fn has_real_root(g: &[Int]) -> bool {
    if g.len() <= 1 {
        return false;
    }
    let sequence: Vec<Vec<Int>> = signed_remainder_sequence(g, &derivative(g));
    variations_at_infinity(&sequence, -1) != variations_at_infinity(&sequence, 1)
}

/// The number of roots with positive real part of the polynomial with
/// Gaussian integer coefficients `re + i im` (highest degree first, a
/// nonzero leading coefficient); `None` when a root lies on the imaginary
/// axis.
fn right_half_plane(re: &[Int], im: &[Int]) -> Option<i64> {
    let n: usize = re.len() - 1;
    if n == 0 {
        return Some(0);
    }
    // the leading coefficient made real and positive
    let (lr, li) = (re[0].clone(), im[0].clone());
    let re2: Vec<Int> = re.iter().zip(im).map(|(r, s)| r * &lr + s * &li).collect();
    let im2: Vec<Int> = re.iter().zip(im).map(|(r, s)| s * &lr - r * &li).collect();
    // g(i t) = u(t) + i v(t)
    let mut u: Vec<Int> = Vec::new();
    let mut v: Vec<Int> = Vec::new();
    for k in 0..=n {
        let (cr, ci) = (&re2[k], &im2[k]);
        match (n - k) % 4 {
            0 => {
                u.push(cr.clone());
                v.push(ci.clone());
            }
            1 => {
                u.push(-ci);
                v.push(cr.clone());
            }
            2 => {
                u.push(-cr);
                v.push(-ci);
            }
            _ => {
                u.push(ci.clone());
                v.push(-cr);
            }
        }
    }
    if n % 2 == 1 {
        // times -i: the leading term goes to the real part
        let negated: Vec<Int> = u.iter().map(|c| -c).collect();
        u = v;
        v = negated;
    }
    let (u, v) = (strip(&u), strip(&v));
    if v.is_empty() {
        return if has_real_root(&u) || n % 2 == 1 {
            None
        } else {
            Some((n / 2) as i64)
        };
    }
    let sequence: Vec<Vec<Int>> = signed_remainder_sequence(&u, &v);
    if has_real_root(&sequence[sequence.len() - 1]) {
        return None;
    }
    let index: i64 = variations_at_infinity(&sequence, -1) - variations_at_infinity(&sequence, 1);
    Some((n as i64 + index).div_euclid(2))
}

/// The coefficients of `denominator**n a((numerator + w)/denominator)` in
/// `w`.
fn taylor_shift(a: &[Int], numerator: &Int, denominator: &Int) -> Vec<Int> {
    let n: usize = a.len() - 1;
    let mut result: Vec<Int> = a.iter().enumerate().map(|(k, c)| c * denominator.pow(k)).collect();
    for i in 0..n {
        for j in 1..=n - i {
            let t: Int = &result[j - 1] * numerator;
            result[j] += t;
        }
    }
    result
}

/// The number of complex roots of `f`, with multiplicities, whose real part
/// is larger than `a`; `ValueError` when a root has the real part `a`.
pub fn dup_count_roots_right_of(f: &[Rat], a: &Rat) -> PolyResult<i64> {
    let coefficients: Vec<Int> = integer_coefficients(f)?;
    let n: usize = coefficients.len() - 1;
    // denominator**n f(a + w/denominator) as a polynomial in w
    let shifted: Vec<Int> = taylor_shift(&coefficients, &numer(a), &denom(a));
    match right_half_plane(&shifted, &vec![Int::ZERO; n + 1]) {
        Some(count) => Ok(count),
        None => Err(value_error(format!("a root has the real part {}", a))),
    }
}

/// The number of complex roots of `f`, with multiplicities, in the open disc
/// of the given centre `(re, im)` and radius; `ValueError` when a root lies
/// on the circle or the radius is not positive.
pub fn dup_count_roots_in_disc(f: &[Rat], center: &(Rat, Rat), radius: &Rat) -> PolyResult<i64> {
    if *radius <= Rat::ZERO {
        return Err(value_error("the radius must be positive".to_string()));
    }
    let coefficients: Vec<Int> = integer_coefficients(f)?;
    let n: usize = coefficients.len() - 1;
    let (cx, cy) = center;
    let scale: Int = lcm(&lcm(&denom(cx), &denom(cy)), &denom(radius));
    let (x, y, r) = (scaled(cx, &scale), scaled(cy, &scale), scaled(radius, &scale));
    // z = c + radius (w - 1)/(w + 1): scale**n (w + 1)**n f(z) =
    // sum a_k N**(n - k) (scale (w + 1))**k, N = (X + i Y)(w + 1) + R (w - 1)
    let numerator_re: [Int; 2] = [&x + &r, &x - &r];
    let numerator_im: [Int; 2] = [y.clone(), y.clone()];
    let mut total_re: Vec<Int> = vec![Int::ZERO; n + 1];
    let mut total_im: Vec<Int> = vec![Int::ZERO; n + 1];
    let mut power_re: Vec<Int> = vec![Int::ONE];
    let mut power_im: Vec<Int> = vec![Int::ZERO];
    for j in 0..=n {
        let k: usize = n - j;
        // (scale (w + 1))**k
        let mut binomial: Vec<Int> = vec![Int::ONE];
        for _ in 0..k {
            let mut next: Vec<Int> = binomial.clone();
            next.push(Int::ZERO);
            for (i, b) in binomial.iter().enumerate() {
                next[i + 1] += b;
            }
            binomial = next;
        }
        let factor: Int = &coefficients[k] * scale.pow(k);
        if !factor.is_zero() {
            for p_index in 0..power_re.len() {
                for (b_index, weight) in binomial.iter().enumerate() {
                    total_re[p_index + b_index] += &factor * weight * &power_re[p_index];
                    total_im[p_index + b_index] += &factor * weight * &power_im[p_index];
                }
            }
        }
        if j < n {
            let mut next_re: Vec<Int> = vec![Int::ZERO; power_re.len() + 1];
            let mut next_im: Vec<Int> = vec![Int::ZERO; power_re.len() + 1];
            for p_index in 0..power_re.len() {
                for q_index in 0..2 {
                    next_re[p_index + q_index] +=
                        &power_re[p_index] * &numerator_re[q_index] - &power_im[p_index] * &numerator_im[q_index];
                    next_im[p_index + q_index] +=
                        &power_re[p_index] * &numerator_im[q_index] + &power_im[p_index] * &numerator_re[q_index];
                }
            }
            power_re = next_re;
            power_im = next_im;
        }
    }
    if total_re[0].is_zero() && total_im[0].is_zero() {
        // the leading coefficient is scale**n f(c + radius)
        return Err(value_error("a root lies on the circle".to_string()));
    }
    match right_half_plane(&total_re, &total_im) {
        Some(count) => Ok(count),
        None => Err(value_error("a root lies on the circle".to_string())),
    }
}

/// Whether every root of `f` has a negative real part (Routh–Hurwitz, by a
/// Cauchy index).
pub fn dup_hurwitz_stable_p(f: &[Rat]) -> PolyResult<bool> {
    let coefficients: Vec<Int> = integer_coefficients(f)?;
    Ok(right_half_plane(&coefficients, &vec![Int::ZERO; f.len()]) == Some(0))
}

/// Whether every root of `f` is in the open unit disc (Schur–Cohn).
pub fn dup_schur_stable_p(f: &[Rat]) -> PolyResult<bool> {
    match dup_count_roots_in_disc(f, &(Rat::ZERO, Rat::ZERO), &Rat::ONE) {
        Ok(count) => Ok(count == dup_degree(f) as i64),
        Err(PolyError::Value(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: &[i64]) -> Vec<Rat> {
        v.iter().map(|c| rat(*c, 1)).collect()
    }

    #[test]
    fn docstring_examples() {
        let f = q(&[1, 0, 0, -1]);
        let shown: Vec<String> = dup_isolate_complex_roots_sqf(&f, None)
            .unwrap()
            .iter()
            .map(|r| r.to_string())
            .collect();
        assert_eq!(shown, vec!["Rectangle(-3, 3, 3/4, 3)"]);
        let shown: Vec<String> = dup_isolate_complex_roots_sqf(&f, Some(&rat(1, 10)))
            .unwrap()
            .iter()
            .map(|r| r.to_string())
            .collect();
        assert_eq!(shown, vec!["Rectangle(-33/64, -31/64, 109/128, 113/128)"]);
        let square = Rectangle::new(rat(-2, 1), rat(2, 1), rat(-2, 1), rat(2, 1)).unwrap();
        assert_eq!(dup_count_complex_roots(&f, &square).unwrap(), 3);
        let left = Rectangle::new(rat(-2, 1), rat(0, 1), rat(-2, 1), rat(2, 1)).unwrap();
        assert_eq!(dup_count_complex_roots(&f, &left).unwrap(), 2);
        let i2 = q(&[1, 0, 1]);
        let r = Rectangle::new(rat(-1, 1), rat(2, 1), rat(0, 1), rat(3, 1)).unwrap();
        assert_eq!(
            dup_refine_complex_root(&i2, &r, &rat(1, 4)).unwrap().to_string(),
            "Rectangle(-1/64, 1/64, 63/64, 65/64)"
        );
        assert_eq!(
            (
                dup_count_roots_right_of(&f, &Rat::ZERO).unwrap(),
                dup_count_roots_right_of(&f, &rat(-1, 1)).unwrap(),
                dup_count_roots_right_of(&f, &rat(2, 1)).unwrap()
            ),
            (1, 3, 0)
        );
        let g = q(&[2, 0, 0, 0, 1]);
        let origin = (Rat::ZERO, Rat::ZERO);
        assert_eq!(dup_count_roots_in_disc(&g, &origin, &Rat::ONE).unwrap(), 4);
        assert_eq!(dup_count_roots_in_disc(&g, &origin, &rat(1, 2)).unwrap(), 0);
        assert_eq!(
            dup_count_roots_in_disc(&g, &(Rat::ONE, Rat::ONE), &Rat::ONE).unwrap(),
            1
        );
        assert!(dup_hurwitz_stable_p(&q(&[1, 3, 3, 1])).unwrap() && !dup_hurwitz_stable_p(&q(&[1, 1, 1, 6])).unwrap());
        assert!(dup_schur_stable_p(&q(&[4, 0, 1])).unwrap() && !dup_schur_stable_p(&q(&[1, 0, 4])).unwrap());
        let multiple = dup_isolate_complex_roots(&q(&[1, 0, 3, 0, 3, 0, 1]), None).unwrap();
        assert_eq!(multiple.len(), 1);
        assert_eq!(
            (multiple[0].0.to_string(), multiple[0].1),
            ("Rectangle(-3, 3, 3/4, 3)".to_string(), 3)
        );
    }
}
