//! Port of mpmath's tanh-sinh quadrature (`mpmath/calculus/quadrature.py`:
//! `QuadratureRule.summation`, `transform_nodes`, `estimate_error`,
//! `TanhSinh.calc_nodes`, `TanhSinh.sum_next`, `quad`), the rule of SymPy's
//! `evalf_integral`. The values are pairs of reals at the working
//! precision of mpmath (`prec + 20` during the summation, `prec + 40` for
//! the standard nodes).

// PORTED FROM SYMPY: mpmath/calculus/quadrature.py (mpmath 1.3, BSD license)

use dashu_base::{BitTest, UnsignedAbs};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::core::evalf::{real_int, Real};
use crate::numbers::Int;

/// A point of the subdivision: finite or one of the infinities.
#[derive(Clone, Debug, PartialEq)]
pub enum Point {
    NegInf,
    Finite(Real),
    PosInf,
}

impl Point {
    fn rank(&self) -> i8 {
        match self {
            Point::NegInf => -1,
            Point::Finite(_) => 0,
            Point::PosInf => 1,
        }
    }

    /// The order of the extended reals.
    pub fn less(&self, other: &Point) -> bool {
        match (self, other) {
            (Point::Finite(a), Point::Finite(b)) => a < b,
            _ => self.rank() < other.rank(),
        }
    }

    pub fn is_finite(&self) -> bool {
        matches!(self, Point::Finite(_))
    }
}

/// A complex value as its real and imaginary parts.
pub type Value = (Real, Real);

fn at(x: &Real, p: usize) -> Real {
    x.clone().with_precision(p).value()
}

fn int(n: i64, p: usize) -> Real {
    real_int(&Int::from(n), p)
}

fn two_pow(k: isize, p: usize) -> Real {
    // 2**k exactly (a power of two is exact at any precision)
    let one: Real = int(1, p);
    if k >= 0 {
        one * int(1i64 << k.min(62), p) * pow2_rest(k - k.min(62), p)
    } else {
        let d: isize = -k;
        one / (int(1i64 << d.min(62), p) * pow2_rest(d - d.min(62), p))
    }
}

fn pow2_rest(k: isize, p: usize) -> Real {
    let mut r: Real = int(1, p);
    let big: Real = int(1i64 << 62, p);
    let mut left: isize = k;
    while left >= 62 {
        r *= &big;
        left -= 62;
    }
    r * int(1i64 << left, p)
}

fn abs_real(r: &Real) -> Real {
    if *r < Real::ZERO {
        -r.clone()
    } else {
        r.clone()
    }
}

/// `log10(|x|)` as a float, `-inf` at zero (mpmath's `log(0, 10)`).
pub fn log10_abs(x: &Real) -> f64 {
    let r = x.repr();
    let sig = r.significand();
    if sig.is_zero() {
        return f64::NEG_INFINITY;
    }
    let bits: usize = sig.unsigned_abs().bit_len();
    // the top 53 bits of the significand as a float times 2**(exponent + shift)
    let shift: usize = bits.saturating_sub(53);
    let top: f64 = (sig.unsigned_abs() >> shift).to_f64().value();
    (top.log2() + shift as f64 + r.exponent() as f64) * std::f64::consts::LOG10_2
}

/// `|re + i im|`.
pub fn modulus(v: &Value) -> Real {
    if v.1.repr().significand().is_zero() {
        return abs_real(&v.0);
    }
    if v.0.repr().significand().is_zero() {
        return abs_real(&v.1);
    }
    (&v.0 * &v.0 + &v.1 * &v.1).sqrt()
}

thread_local! {
    static STANDARD: RefCell<HashMap<(usize, usize), Vec<(Real, Real)>>> = RefCell::new(HashMap::new());
}

/// `TanhSinh.calc_nodes` at `prec + 40` bits (the standard nodes on
/// `[-1, 1]` of the degree).
fn calc_nodes(degree: usize, prec: usize) -> Vec<(Real, Real)> {
    if let Some(found) = STANDARD.with(|c| c.borrow().get(&(degree, prec)).cloned()) {
        return found;
    }
    let p: usize = prec + 40;
    let mut nodes: Vec<(Real, Real)> = Vec::new();
    let tol: Real = two_pow(-(prec as isize) - 10, p);
    let pi: Real = Real::pi(p);
    let pi4: Real = &pi / int(4, p);
    let t0: Real = two_pow(-(degree as isize), p);
    let h: Real = if degree == 1 {
        nodes.push((int(0, p), &pi / int(2, p)));
        t0.clone()
    } else {
        &t0 * int(2, p)
    };
    let expt0: Real = t0.exp();
    let mut a: Real = &pi4 * &expt0;
    let mut b: Real = &pi4 / &expt0;
    let udelta: Real = h.exp();
    let urdelta: Real = int(1, p) / &udelta;
    let one: Real = int(1, p);
    let two: Real = int(2, p);
    for _ in 0..=(20usize << degree) {
        let c: Real = (&a - &b).exp();
        let d: Real = &one / &c;
        let co: Real = (&c + &d) / &two;
        let si: Real = (&c - &d) / &two;
        let x: Real = &si / &co;
        let w: Real = (&a + &b) / (&co * &co);
        if abs_real(&(&x - &one)) <= tol {
            break;
        }
        nodes.push((x.clone(), w.clone()));
        nodes.push((-x, w));
        a = &a * &udelta;
        b = &b * &urdelta;
    }
    STANDARD.with(|c| c.borrow_mut().insert((degree, prec), nodes.clone()));
    nodes
}

/// `transform_nodes` at `prec + 20` bits: the nodes of `[-1, 1]` moved
/// onto `(a, b)` (an infinite end by `x -> a - 1 + 2/(x + 1)`).
fn transform_nodes(nodes: &[(Real, Real)], a: &Point, b: &Point, p: usize) -> Vec<(Real, Real)> {
    let one: Real = int(1, p);
    let half: Real = &one / int(2, p);
    let r = |x: Real| at(&x, p);
    match (a, b) {
        (Point::Finite(fa), Point::Finite(fb)) => {
            if *fa == -one.clone() && *fb == one {
                return nodes.iter().map(|(x, w)| (r(x.clone()), r(w.clone()))).collect();
            }
            let c: Real = r((fb - fa) / int(2, p));
            let d: Real = r((fb + fa) / int(2, p));
            nodes.iter().map(|(x, w)| (r(&d + &c * x), r(&c * w))).collect()
        }
        (Point::NegInf, Point::PosInf) => nodes
            .iter()
            .map(|(x, w)| {
                let px1: Real = r(&one - x * x);
                let spx1: Real = r(&one / px1.sqrt());
                (r(x * &spx1), r(w * &spx1 / &px1))
            })
            .collect(),
        (Point::NegInf, Point::Finite(fb)) => {
            let b1: Real = r(fb + &one);
            nodes
                .iter()
                .map(|(x, w)| {
                    let u: Real = r(int(2, p) / (x + &one));
                    (r(&b1 - &u), r(w * &half * &u * &u))
                })
                .collect()
        }
        (Point::Finite(fa), Point::PosInf) => {
            let a1: Real = r(fa - &one);
            nodes
                .iter()
                .map(|(x, w)| {
                    let u: Real = r(int(2, p) / (x + &one));
                    (r(&a1 + &u), r(w * &half * &u * &u))
                })
                .collect()
        }
        _ => transform_nodes(nodes, b, a, p)
            .into_iter()
            .map(|(x, w)| (x, -w))
            .collect(),
    }
}

/// `guess_degree`.
fn guess_degree(prec: usize) -> usize {
    (4.0 + (prec as f64 / 30.0).log2().max(0.0)) as usize + 2
}

/// Python's `max` of floats (the first of the largest, by `>`).
fn py_max(items: &[f64]) -> f64 {
    let mut best: f64 = items[0];
    for v in &items[1..] {
        if *v > best {
            best = *v;
        }
    }
    best
}

/// `estimate_error`.
fn estimate_error(results: &[Value], prec: usize, epsilon: &Real, p: usize) -> Real {
    let diff = |u: &Value, v: &Value| modulus(&(&u.0 - &v.0, &u.1 - &v.1));
    let n: usize = results.len();
    if n == 2 {
        return diff(&results[0], &results[1]);
    }
    if results[n - 1] == results[n - 2] && results[n - 2] == results[n - 3] {
        return int(0, p);
    }
    let d1: f64 = log10_abs(&diff(&results[n - 1], &results[n - 2]));
    let d2: f64 = log10_abs(&diff(&results[n - 1], &results[n - 3]));
    if d1.is_infinite() && d2.is_infinite() {
        // mpmath: inf/-inf is nan, and Python's max keeps it: min(0, nan) is 0
        let _ = epsilon;
        return int(1, p);
    }
    let d3: f64 = -(prec as f64);
    let first: f64 = d1 * d1 / d2;
    let mut d4: f64 = py_max(&[first, 2.0 * d1, d3]);
    // min(0, d4): 0 unless d4 < 0 (nan is not)
    d4 = if d4 < 0.0 { d4 } else { 0.0 };
    let k: i64 = d4.trunc() as i64;
    if k >= 0 {
        int(10i64.pow(k as u32), p)
    } else {
        let mut ten: Real = int(1, p);
        for _ in 0..(-k) {
            ten *= int(10, p);
        }
        int(1, p) / ten
    }
}

/// `TanhSinh.sum_next` with mpmath's exact `fdot`, rounded once.
fn sum_next(
    f: &mut dyn FnMut(&Real) -> Option<Value>,
    nodes: &[(Real, Real)],
    degree: usize,
    previous: Option<&Value>,
    p: usize,
) -> Option<Value> {
    let h: Real = two_pow(-(degree as isize), p);
    let wide: usize = 2 * p + 64;
    let (mut sr, mut si): (Real, Real) = match previous {
        Some(v) => {
            let twice: Real = &h * int(2, p);
            (at(&(&v.0 / &twice), wide), at(&(&v.1 / &twice), wide))
        }
        None => (int(0, wide), int(0, wide)),
    };
    let (mut dr, mut di): (Real, Real) = (int(0, wide), int(0, wide));
    for (x, w) in nodes {
        let (vr, vi) = f(x)?;
        let w2: Real = at(w, wide);
        dr += &w2 * at(&vr, wide);
        di += &w2 * at(&vi, wide);
    }
    sr = at(&(at(&sr, p) + at(&dr, p)), p);
    si = at(&(at(&si, p) + at(&di, p)), p);
    Some((at(&(&h * &sr), p), at(&(&h * &si), p)))
}

/// `quad(f, points, method='tanh-sinh', error=True)` at `prec` bits: the
/// integral over the consecutive intervals and the estimated error;
/// `None` when the integrand has no value at a node.
pub fn quad(f: &mut dyn FnMut(&Real) -> Option<Value>, points: &[Point], prec: usize) -> Option<(Value, Real)> {
    let p: usize = prec + 20;
    let epsilon: Real = two_pow(-(prec as isize) - 2, p);
    let max_degree: usize = guess_degree(prec);
    let mut total: Value = (int(0, p), int(0, p));
    let mut total_err: Real = int(0, p);
    for window in points.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if a == b {
            continue;
        }
        let both: bool = matches!((a, b), (Point::NegInf, Point::PosInf));
        let (a, b) = if both {
            (Point::Finite(int(0, p)), Point::PosInf)
        } else {
            (a.clone(), b.clone())
        };
        let mut g = |x: &Real| -> Option<Value> {
            if both {
                let (u, v) = (f(&-x.clone())?, f(x)?);
                return Some((&u.0 + &v.0, &u.1 + &v.1));
            }
            f(x)
        };
        let mut results: Vec<Value> = Vec::new();
        let mut err: Real = int(0, p);
        for degree in 1..=max_degree {
            let nodes: Vec<(Real, Real)> = transform_nodes(&calc_nodes(degree, prec), &a, &b, p);
            let result: Value = sum_next(&mut g, &nodes, degree, results.last(), p)?;
            results.push(result);
            if degree > 1 {
                err = estimate_error(&results, prec, &epsilon, p);
                if err <= epsilon {
                    break;
                }
            }
        }
        let last: &Value = results.last().expect("a degree");
        total = (at(&(&total.0 + &last.0), p), at(&(&total.1 + &last.1), p));
        total_err = at(&(&total_err + &err), p);
    }
    Some(((at(&total.0, prec), at(&total.1, prec)), total_err))
}
