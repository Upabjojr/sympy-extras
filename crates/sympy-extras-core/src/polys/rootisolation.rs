//! Port of `sympy_extras/polys/rootisolation.py`: real root isolation and
//! counting for dense polynomials with rational coefficients.
//!
//! **Descartes' rule with bisection** (Collins and Akritas; Rouillier and
//! Zimmermann): for a square-free `f`, the positive roots are brought into
//! `(0, 1)` by a root bound (the smaller of Cauchy's and Fujiwara's), and an
//! interval `(c/2**k, (c + 1)/2**k)` carries the polynomial `p` whose roots
//! in `(0, 1)` are the roots of `f` in the interval. The number `v` of sign
//! variations of `(x + 1)**n p(1/(x + 1))` bounds the number of roots of
//! `p` in `(0, 1)` and has the same parity: `v = 0` means no root, `v = 1`
//! exactly one; otherwise the interval is bisected, the midpoint being
//! checked for an exact root. Each interval returned contains one root, as
//! an open interval with rational endpoints or as a point (`a == b`) for a
//! rational root. **Sturm sequences** count the roots in an interval, and
//! the isolating intervals are refined by bisection.
//!
//! References
//! ==========
//!
//! 1. G. E. Collins, A. G. Akritas, *Polynomial real root isolation using
//!    Descartes' rule of signs*, SYMSAC '76, 272-275.
//! 2. F. Rouillier, P. Zimmermann, *Efficient isolation of polynomial's real
//!    roots*, Journal of Computational and Applied Mathematics 162 (2004),
//!    33-50.
//! 3. S. Basu, R. Pollack, M.-F. Roy, *Algorithms in Real Algebraic
//!    Geometry*, 2nd ed., Springer, 2006, Section 2.2 and Chapter 10.

use crate::numbers::*;

use super::densearith::{dup_div, dup_mul_ground, dup_neg};
use super::densebasic::{dup_degree, dup_reverse, dup_strip};
use super::densetools::{dup_diff, dup_eval, dup_mirror, dup_monic_field, dup_scale, dup_shift, dup_sign_variations};
use super::domains::{Domain, QQ};
use super::sqfreetools::{dup_sqf_list, dup_sqf_part};

/// An isolating interval: `(a, b)` with the root in the open interval when
/// `a < b`, the root itself when `a == b`.
pub type Interval = (Rat, Rat);

/// `1 + max |a_i/a_n|`: every root has absolute value below it.
pub fn dup_cauchy_bound(f: &[Rat]) -> Rat {
    if dup_degree(f) < 1 {
        return Rat::ONE;
    }
    let lc: &Rat = &f[0];
    let best: Rat = f[1..]
        .iter()
        .map(|c| QQ.abs(&(c / lc)))
        .max()
        .expect("a positive degree");
    Rat::ONE + best
}

/// The least integer `m >= 1` with `m**k >= q` (`q >= 0`).
fn ceil_root(q: &Rat, k: usize) -> Int {
    if *q <= Rat::ONE {
        return Int::ONE;
    }
    let reaches = |m: &Int| -> bool { Rat::from(m.pow(k)) >= *q };
    let (mut low, mut high): (Int, Int) = (Int::ONE, Int::from(2u8));
    while !reaches(&high) {
        high = &high * Int::from(2u8);
    }
    while &high - &low > Int::ONE {
        let mid: Int = div_floor(&(&low + &high), &Int::from(2u8));
        if reaches(&mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
    high
}

/// Fujiwara's bound, `2 max(|a_{n-1}/a_n|, |a_{n-2}/a_n|**(1/2), ...,
/// |a_0/(2 a_n)|**(1/n))`, with each root rounded up to an integer.
pub fn dup_fujiwara_bound(f: &[Rat]) -> Rat {
    let n: isize = dup_degree(f);
    if n < 1 {
        return Rat::ONE;
    }
    let n: usize = n as usize;
    let lc: &Rat = &f[0];
    let mut best: Int = Int::ZERO;
    for (k, coefficient) in f.iter().enumerate().take(n + 1).skip(1) {
        let mut c: Rat = QQ.abs(&(coefficient / lc));
        if k == n {
            c /= Rat::from(Int::from(2u8));
        }
        if !c.is_zero() {
            best = best.max(ceil_root(&c, k));
        }
    }
    Rat::from(Int::from(2u8) * best.max(Int::ONE))
}

/// The sign of `f(a)`.
pub fn dup_sign_at(f: &[Rat], a: &Rat) -> i32 {
    QQ.sign(&dup_eval(f, a, &QQ))
}

/// `f` divided by the absolute value of its leading coefficient.
pub fn dup_monic_signed(f: &[Rat]) -> Vec<Rat> {
    if f.is_empty() {
        return Vec::new();
    }
    dup_mul_ground(f, &recip(&QQ.abs(&f[0])), &QQ)
}

/// The Sturm sequence `f, f', -rem(f, f'), ...` (each polynomial made monic
/// up to its sign).
pub fn dup_sturm(f: &[Rat]) -> Vec<Vec<Rat>> {
    let mut sequence: Vec<Vec<Rat>> = vec![dup_monic_signed(f), dup_monic_signed(&dup_diff(f, 1, &QQ))];
    while !sequence[sequence.len() - 1].is_empty() {
        let n: usize = sequence.len();
        let remainder: Vec<Rat> = dup_div(&sequence[n - 2], &sequence[n - 1], &QQ)
            .expect("a nonzero divisor")
            .1;
        sequence.push(dup_monic_signed(&dup_neg(&remainder, &QQ)));
    }
    sequence.pop();
    sequence
}

/// Where the sign variations are counted.
#[derive(Debug, Clone, Copy)]
enum At<'a> {
    Point(&'a Rat),
    MinusInfinity,
    PlusInfinity,
}

fn variations_at(sequence: &[Vec<Rat>], at: At<'_>) -> i64 {
    let mut signs: Vec<i32> = Vec::new();
    for g in sequence {
        let s: i32 = match at {
            At::Point(a) => dup_sign_at(g, a),
            At::MinusInfinity | At::PlusInfinity => {
                let mut s: i32 = if g.is_empty() { 0 } else { QQ.sign(&g[0]) };
                if matches!(at, At::MinusInfinity) && dup_degree(g) % 2 != 0 {
                    s = -s;
                }
                s
            }
        };
        if s != 0 {
            signs.push(s);
        }
    }
    signs.windows(2).filter(|w| w[0] != w[1]).count() as i64
}

/// The number of distinct real roots in `(inf, sup]` (the whole line when
/// the bounds are `None`), by Sturm's theorem on the square-free part.
pub fn dup_count_real_roots(f: &[Rat], inf: Option<&Rat>, sup: Option<&Rat>) -> i64 {
    if dup_degree(f) < 1 {
        return 0;
    }
    let g: Vec<Rat> = dup_sqf_part(f, &QQ);
    let sequence: Vec<Vec<Rat>> = dup_sturm(&g);
    let low: At<'_> = match inf {
        Some(a) => At::Point(a),
        None => At::MinusInfinity,
    };
    let high: At<'_> = match sup {
        Some(b) => At::Point(b),
        None => At::PlusInfinity,
    };
    variations_at(&sequence, low) - variations_at(&sequence, high)
}

/// The sign variations of `(x + 1)**n p(1/(x + 1))`: a bound on the roots
/// of `p` in `(0, 1)` with the same parity.
fn descartes_bound(p: &[Rat]) -> usize {
    dup_sign_variations(&dup_shift(&dup_reverse(p, &QQ), &Rat::ONE, &QQ), &QQ)
}

/// A node `(c, k, exact)` of the bisection tree: the interval `(c/2**k, (c +
/// 1)/2**k)` of the unit interval, `exact` when the root is its left end.
type Node = (Int, usize, bool);

/// The roots of `p` in `(0, 1)` as the nodes of the bisection tree.
fn isolate_in_unit_interval(p: &[Rat], c: &Int, k: usize, found: &mut Vec<Node>) {
    let v: usize = descartes_bound(p);
    if v == 0 {
        return;
    }
    if v == 1 {
        found.push((c.clone(), k, false));
        return;
    }
    // the left half: the roots of p(x/2) in (0, 1)
    let left: Vec<Rat> = dup_scale(p, &rat(1, 2), &QQ);
    let twice: Int = Int::from(2u8) * c;
    isolate_in_unit_interval(&left, &twice, k + 1, found);
    let mut right: Vec<Rat> = dup_shift(&left, &Rat::ONE, &QQ);
    let next: Int = &twice + Int::ONE;
    if right.last().is_some_and(|t| t.is_zero()) {
        // the midpoint is a root: strip it and go on with the right half
        found.push((next.clone(), k + 1, true));
        right.pop();
        right = dup_strip(&right, &QQ);
    }
    isolate_in_unit_interval(&right, &next, k + 1, found);
}

/// Isolating intervals of the real roots of a square-free `f`, in
/// increasing order, each open interval (rational endpoints) containing one
/// root or a point for a rational root; refined below the width `eps` when
/// given.
pub fn dup_isolate_real_roots_sqf(f: &[Rat], eps: Option<&Rat>) -> Vec<Interval> {
    if dup_degree(f) < 1 {
        return Vec::new();
    }
    let mut f: Vec<Rat> = dup_monic_field(f, &QQ);
    let mut intervals: Vec<Interval> = Vec::new();
    let whole: Vec<Rat> = f.clone();
    // the root 0
    if f.last().is_some_and(|t| t.is_zero()) {
        intervals.push((Rat::ZERO, Rat::ZERO));
        f.pop();
        f = dup_strip(&f, &QQ);
    }
    // Fujiwara's bound is not strict: a root at the bound itself is taken
    // out first
    let bound: Rat = dup_cauchy_bound(&f).min(dup_fujiwara_bound(&f));
    for sign in [-1i32, 1] {
        let g: Vec<Rat> = if sign < 0 { dup_mirror(&f, &QQ) } else { f.clone() };
        let signed_bound: Rat = if sign < 0 { -&bound } else { bound.clone() };
        if dup_sign_at(&g, &bound) == 0 {
            intervals.push((signed_bound.clone(), signed_bound.clone()));
        }
        // the positive roots of g into (0, 1): g(bound x)
        let scaled: Vec<Rat> = dup_scale(&g, &bound, &QQ);
        let mut found: Vec<Node> = Vec::new();
        isolate_in_unit_interval(&scaled, &Int::ZERO, 0, &mut found);
        for (c, k, exact) in found {
            let power: Int = Int::from(2u8).pow(k);
            let a: Rat = rat_from_parts(c.clone(), power.clone()) * &bound;
            let b: Rat = rat_from_parts(c + Int::ONE, power) * &bound;
            let mut interval: Interval = if exact { (a.clone(), a) } else { (a, b) };
            if sign < 0 {
                interval = (-&interval.1, -&interval.0);
            }
            intervals.push(interval);
        }
    }
    // an endpoint of an open interval may be another root (found exactly at
    // a midpoint, or 0): shrink such intervals
    let mut sequence: Vec<Vec<Rat>> = Vec::new();
    for interval in intervals.iter_mut() {
        let (a, b) = interval.clone();
        if a != b && (dup_sign_at(&whole, &a) == 0 || dup_sign_at(&whole, &b) == 0) {
            if sequence.is_empty() {
                sequence = dup_sturm(&whole);
            }
            *interval = endpoints_off_roots(&whole, &sequence, a, b);
        }
    }
    intervals.sort();
    if let Some(eps) = eps {
        intervals = intervals
            .into_iter()
            .map(|(a, b)| dup_refine_real_root(&f, a, b, eps))
            .collect();
    }
    intervals
}

/// The isolating interval `(a, b)` of a root of the square-free `f`
/// bisected (with Sturm counts) until neither endpoint is a root of `f`.
fn endpoints_off_roots(f: &[Rat], sequence: &[Vec<Rat>], a: Rat, b: Rat) -> Interval {
    let (mut a, mut b): (Rat, Rat) = (a, b);
    while dup_sign_at(f, &a) == 0 || dup_sign_at(f, &b) == 0 {
        let m: Rat = (&a + &b) / Rat::from(Int::from(2u8));
        if dup_sign_at(f, &m) == 0 {
            return (m.clone(), m);
        }
        // the number of roots in (a, m]
        if variations_at(sequence, At::Point(&a)) - variations_at(sequence, At::Point(&m)) == 1 {
            b = m;
        } else {
            a = m;
        }
    }
    (a, b)
}

/// The isolating interval `(a, b)` of a root of `f` bisected until its width
/// is below `eps` (a point when the root is found exactly).
pub fn dup_refine_real_root(f: &[Rat], a: Rat, b: Rat, eps: &Rat) -> Interval {
    if a == b {
        return (a, b);
    }
    let sa: i32 = dup_sign_at(f, &a);
    if sa == 0 {
        return (a.clone(), a);
    }
    let (mut a, mut b): (Rat, Rat) = (a, b);
    while &(&b - &a) >= eps {
        let m: Rat = (&a + &b) / Rat::from(Int::from(2u8));
        let sm: i32 = dup_sign_at(f, &m);
        if sm == 0 {
            return (m.clone(), m);
        }
        if sm == sa {
            a = m;
        } else {
            b = m;
        }
    }
    (a, b)
}

/// Isolating intervals of the real roots of `f` with their multiplicities,
/// in increasing order.
pub fn dup_isolate_real_roots(f: &[Rat], eps: Option<&Rat>) -> Vec<(Interval, usize)> {
    if dup_degree(f) < 1 {
        return Vec::new();
    }
    let (_, factors) = dup_sqf_list(f, &QQ);
    let mut result: Vec<(Interval, usize)> = Vec::new();
    for (g, k) in &factors {
        for interval in dup_isolate_real_roots_sqf(g, eps) {
            result.push((interval, *k));
        }
    }
    result.sort_by(|x, y| x.0.cmp(&y.0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::densebasic::testing::qq;

    #[test]
    fn doctests() {
        let f: Vec<Rat> = qq(&[1, 0, -2]);
        assert_eq!(
            dup_isolate_real_roots(&f, None),
            vec![((rat(-2, 1), rat(0, 1)), 1), ((rat(0, 1), rat(2, 1)), 1)]
        );
        assert_eq!(dup_count_real_roots(&f, None, None), 2);
        assert_eq!(dup_count_real_roots(&f, Some(&rat(0, 1)), Some(&rat(3, 1))), 1);
        assert_eq!(
            dup_refine_real_root(&f, rat(0, 1), rat(2, 1), &rat(1, 100)),
            (rat(181, 128), rat(91, 64))
        );
        assert_eq!(
            dup_isolate_real_roots(&qq(&[1, -3, 3, -1]), None),
            vec![((rat(0, 1), rat(2, 1)), 3)]
        );
    }

    #[test]
    fn rational_roots_are_points() {
        // (x - 1)(x + 1/2)(x - 3) x
        let f: Vec<Rat> = vec![rat(1, 1), rat(-7, 2), rat(1, 1), rat(3, 2), rat(0, 1)];
        let roots: Vec<(Interval, usize)> = dup_isolate_real_roots(&f, None);
        assert_eq!(roots.len(), 4);
        for ((a, b), _) in &roots {
            assert!(a <= b);
            assert_eq!(dup_count_real_roots(&f, Some(&(a - rat(1, 1000000))), Some(b)), 1);
        }
        assert_eq!(roots[1].0, (rat(0, 1), rat(0, 1)));
    }
}
