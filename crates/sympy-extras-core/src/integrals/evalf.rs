//! Port of `sympy_extras/integrals/_evalf.py` (SymPy's `evalf_integral`):
//! the value of a definite integral by tanh-sinh quadrature
//! ([`crate::integrals::quadrature`]) on a subdivision of the range with
//! the kinks of the integrand, the working precision raised until the
//! accuracy is reached and two values agree. The integrand is evaluated
//! by `numeric_value` with its variable bound to the node
//! ([`bound_value`]), which also gives the branches of a `Piecewise` and
//! the inner integrals (the role of `_numeric_function`).

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;

use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::evalf::{complex, numeric_value, parts, real_int, Complex, Real};
use crate::integrals::quadrature::{modulus, quad, Point, Value};
use crate::numbers::{Int, Rat};

thread_local! {
    static BINDINGS: RefCell<Vec<(Basic, Complex)>> = const { RefCell::new(Vec::new()) };
}

/// The value bound to a symbol by an enclosing quadrature.
pub fn bound_value(s: &Basic) -> Option<Complex> {
    BINDINGS.with(|b| b.borrow().iter().rev().find(|(k, _)| k == s).map(|(_, v)| v.clone()))
}

/// Whether `e` depends on a variable bound by a quadrature (a `Piecewise`
/// of the integrand gets its numerical rule).
pub fn binding_active(e: &Basic) -> bool {
    BINDINGS.with(|b| {
        let bound = b.borrow();
        !bound.is_empty() && e.free_symbols().iter().any(|s| bound.iter().any(|(k, _)| k == s))
    })
}

/// `e` at `x = z` (`_numeric_function(e, x)(z)`).
pub fn value_at(e: &Basic, x: &Basic, z: Complex, prec: usize) -> Option<Complex> {
    BINDINGS.with(|b| b.borrow_mut().push((x.clone(), z)));
    let found: Option<Complex> = numeric_value(e, prec);
    BINDINGS.with(|b| b.borrow_mut().pop());
    found
}

const MINUS_INF: i64 = -1_000_000_000;

/// SymPy's `fastlog`: the exponent plus the bit count.
fn fastlog(x: &Real) -> i64 {
    use dashu_base::{BitTest, UnsignedAbs};
    let r = x.repr();
    if r.significand().is_zero() {
        return MINUS_INF;
    }
    r.exponent() as i64 + r.significand().unsigned_abs().bit_len() as i64
}

fn real(n: i64, p: usize) -> Real {
    real_int(&Int::from(n), p)
}

fn is_zero(r: &Real) -> bool {
    r.repr().significand().is_zero()
}

/// The exact rational of a binary float.
pub fn exact(r: &Real) -> Rat {
    let rep = r.repr();
    let sig: Int = rep.significand().clone();
    let e: isize = rep.exponent();
    if e >= 0 {
        Rat::from(sig << (e as usize))
    } else {
        Rat::from_parts(sig, dashu_int::UBig::ONE << ((-e) as usize))
    }
}

fn bound(e: &Basic, prec: usize) -> Option<Point> {
    match e.head() {
        Head::Infinity => return Some(Point::PosInf),
        Head::NegativeInfinity => return Some(Point::NegInf),
        _ => {}
    }
    let (re, im) = parts(&numeric_value(e, prec)?);
    if !is_zero(&im) {
        return None;
    }
    Some(Point::Finite(re))
}

fn finite(p: &Point) -> &Real {
    match p {
        Point::Finite(r) => r,
        _ => panic!("a finite point"),
    }
}

fn subdivision(lo: &Point, hi: &Point, p: usize) -> Vec<Point> {
    let f = |n: i64| Point::Finite(real(n, p));
    match (lo, hi) {
        (Point::NegInf, Point::PosInf) => {
            let mut out: Vec<Point> = vec![Point::NegInf];
            out.extend([-100, -10, -1, 0, 1, 10, 100].into_iter().map(f));
            out.push(Point::PosInf);
            out
        }
        (Point::Finite(a), Point::PosInf | Point::NegInf) => {
            let sign: i64 = if matches!(hi, Point::PosInf) { 1 } else { -1 };
            let mut out: Vec<Point> = vec![lo.clone()];
            out.extend(
                [1, 10, 100, 1000]
                    .into_iter()
                    .map(|d| Point::Finite(a + real(sign * d, p))),
            );
            out.push(hi.clone());
            out
        }
        (Point::PosInf | Point::NegInf, _) => {
            let mut out: Vec<Point> = subdivision(hi, lo, p);
            out.reverse();
            out
        }
        (Point::Finite(a), Point::Finite(b)) => {
            let span: Real = if b > a { b - a } else { a - b };
            let ceil: i64 = span.ceil().to_int().value().try_into().unwrap_or(i64::MAX);
            let pieces: i64 = ceil.min(100);
            if pieces <= 2 {
                return vec![lo.clone(), hi.clone()];
            }
            let mut out: Vec<Point> = (0..pieces)
                .map(|k| Point::Finite(a + (b - a) * real(k, p) / real(pieces, p)))
                .collect();
            out.push(hi.clone());
            out
        }
    }
}

fn kinks(f: &Basic, x: &Basic, prec: usize, lo: &Point, hi: &Point) -> Vec<Real> {
    let mut arguments: Vec<Basic> = Vec::new();
    let mut steps: Vec<Basic> = Vec::new();
    for node in f.preorder() {
        match node.head() {
            Head::Function(Function::Floor | Function::Ceiling) => steps.push(node.args()[0].clone()),
            Head::Function(Function::Abs | Function::Sign | Function::Heaviside) => {
                arguments.push(node.args()[0].clone())
            }
            Head::Function(Function::Max | Function::Min) => {
                let items: &[Basic] = node.args();
                for w in items.windows(2) {
                    if let Ok(d) = crate::core::add::sub(&w[0], &w[1]) {
                        arguments.push(d);
                    }
                }
            }
            Head::Piecewise => {
                for c in node.preorder() {
                    if matches!(c.head(), Head::Relational(_)) {
                        if let Ok(d) = crate::core::add::sub(&c.args()[0], &c.args()[1]) {
                            arguments.push(d);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let mut points: Vec<Real> = Vec::new();
    let tagged = arguments
        .into_iter()
        .map(|u| (u, false))
        .chain(steps.into_iter().map(|u| (u, true)));
    for (u, step) in tagged {
        if !u.has(x) {
            continue;
        }
        let Ok(slope) = crate::core::function::derivative(&u, x) else {
            continue;
        };
        if slope.has(x) || !crate::core::assumptions::is_number(&slope) {
            continue;
        }
        let Ok(rest) = crate::core::subs::subs(&u, x, &Basic::zero()) else {
            continue;
        };
        let (Some(a), Some(b)) = (numeric_value(&slope, prec), numeric_value(&rest, prec)) else {
            continue;
        };
        let ((ar, ai), (br, bi)) = (parts(&a), parts(&b));
        if is_zero(&ar) && is_zero(&ai) || !is_zero(&ai) || !is_zero(&bi) {
            continue;
        }
        if !step {
            points.push(-(&br / &ar));
            continue;
        }
        if !(lo.is_finite() && hi.is_finite()) {
            continue;
        }
        let (e0, e1) = (&ar * finite(lo) + &br, &ar * finite(hi) + &br);
        let (e0, e1) = if e1 < e0 { (e1, e0) } else { (e0, e1) };
        if &e1 - &e0 > real(400, prec) {
            continue;
        }
        let first: i64 = e0.ceil().to_int().value().try_into().unwrap_or(0);
        let last: i64 = e1.floor().to_int().value().try_into().unwrap_or(-1);
        for n in first..=last {
            points.push((real(n, prec) - &br) / &ar);
        }
    }
    points
}

/// `do_integral`: the value at `prec` bits and its accuracy in bits (`-1`
/// for a part found zero), `None` when the quadrature fails.
pub fn do_integral(f: &Basic, x: &Basic, lower: &Basic, upper: &Basic, prec: usize) -> Option<(Value, i64)> {
    if lower == upper {
        return Some(((real(0, prec), real(0, prec)), prec as i64));
    }
    let wp: usize = prec + 5;
    let lo: Point = bound(lower, prec + 15)?;
    let hi: Point = bound(upper, prec + 15)?;
    let largest: RefCell<[i64; 2]> = RefCell::new([MINUS_INF, MINUS_INF]);
    let mut h = |t: &Real| -> Option<Value> {
        let z: Complex = complex(t.clone(), real(0, wp + 20));
        let (re, im) = parts(&value_at(f, x, z, wp + 20)?);
        let mut l = largest.borrow_mut();
        l[0] = l[0].max(fastlog(&re));
        l[1] = l[1].max(fastlog(&im));
        Some((re, im))
    };
    let mut nodes: Vec<Point> = subdivision(&lo, &hi, wp);
    let (small, big) = if hi.less(&lo) { (&hi, &lo) } else { (&lo, &hi) };
    let extra: Vec<Real> = kinks(f, x, prec + 15, &lo, &hi)
        .into_iter()
        .filter(|k| small.less(&Point::Finite(k.clone())) && Point::Finite(k.clone()).less(big))
        .filter(|k| !nodes.contains(&Point::Finite(k.clone())))
        .collect();
    if !extra.is_empty() {
        nodes.extend(extra.into_iter().map(Point::Finite));
        nodes.sort_by(|a, b| {
            if a.less(b) {
                std::cmp::Ordering::Less
            } else if b.less(a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        if hi.less(&lo) {
            nodes.reverse();
        }
    }
    let (value, error) = quad(&mut h, &nodes, wp)?;
    if !(value.0.repr().is_finite() && value.1.repr().is_finite()) {
        return None;
    }
    let quadrature_error: i64 = fastlog(&error);
    let l = *largest.borrow();
    let mut accuracies: Vec<i64> = Vec::new();
    for (part, big) in [(&value.0, l[0]), (&value.1, l[1])] {
        if big == MINUS_INF {
            continue;
        }
        if is_zero(part) {
            accuracies.push(-1);
        } else {
            accuracies.push(-(big - fastlog(part) - prec as i64).max(quadrature_error));
        }
    }
    let accuracy: i64 = accuracies.into_iter().min().unwrap_or(prec as i64);
    Some((value, accuracy))
}

fn agree(a: &Value, b: &Value, prec: usize) -> bool {
    let p: usize = prec + 20;
    let d: Value = (&a.0 - &b.0, &a.1 - &b.1);
    let tiny: Real = real(1, p) / crate::core::evalf::real_int(&(Int::ONE << prec), p);
    let size: Real = modulus(b);
    let scale: Real = if size > tiny { size } else { tiny.clone() };
    modulus(&d) <= &tiny * &scale
}

/// `evalf_integral`: the value of `Integral(f, (x, lower, upper))` at
/// `prec` bits, the working precision raised up to `max(2 prec + 40,
/// 333)`; `None` when it is not reached.
pub fn evalf_integral(f: &Basic, x: &Basic, lower: &Basic, upper: &Basic, prec: usize) -> Option<Complex> {
    let mut workprec: usize = prec;
    let maxprec: usize = (2 * prec + 40).max(333);
    let mut i: u32 = 0;
    let mut previous: Option<Value> = None;
    loop {
        crate::timeout::check().ok()?;
        let (result, accuracy) = do_integral(f, x, lower, upper, workprec)?;
        if accuracy >= prec as i64 {
            if let Some(p) = &previous {
                if agree(p, &result, prec) {
                    return Some(complex(result.0, result.1));
                }
            }
            previous = Some(result);
            if workprec >= maxprec {
                return None;
            }
            workprec = (workprec + (prec / 2).max(20)).min(maxprec);
            continue;
        }
        previous = None;
        if workprec >= maxprec {
            return None;
        }
        if accuracy == -1 {
            workprec *= 2;
        } else {
            workprec += prec.max(1usize << i.min(40));
        }
        workprec = workprec.min(maxprec);
        i += 1;
    }
}

/// The value of an `Integral` node in `numeric_value`: the quadrature of a
/// closed definite integral; an inner integral of an integrand at the
/// bound value of the outer variable (`_numeric_integral`).
pub fn integral_value(e: &Basic, prec: usize) -> Option<Complex> {
    let (f, u, bounds) = crate::integrals::integral::parts(e);
    let (a, b) = bounds?;
    let free = crate::integrals::integral::free_symbols(e);
    if free.is_empty() {
        return evalf_integral(f, u, a, b, prec);
    }
    // the free symbols bound by enclosing quadratures: their exact values
    let mut g: Basic = f.clone();
    for s in &free {
        let z: Complex = bound_value(s)?;
        let (re, im) = parts(&z);
        let mut v: Basic = Basic::from_rational(exact(&re));
        if !is_zero(&im) {
            let imag: Basic =
                crate::core::mul::mul(&[Basic::imaginary_unit(), Basic::from_rational(exact(&im))]).ok()?;
            v = crate::core::add::add(&[v, imag]).ok()?;
        }
        g = crate::core::subs::subs(&g, s, &v).ok()?;
    }
    let (ar, ai) = parts(&numeric_value(a, prec)?);
    let (br, bi) = parts(&numeric_value(b, prec)?);
    if !is_zero(&ai) || !is_zero(&bi) {
        return None;
    }
    let mut inner = |t: &Real| -> Option<Value> {
        let z: Complex = complex(t.clone(), real(0, prec + 20));
        Some(parts(&value_at(&g, u, z, prec + 20)?))
    };
    let (v, _) = quad(&mut inner, &[Point::Finite(ar), Point::Finite(br)], prec)?;
    Some(complex(v.0, v.1))
}

/// The value of a `Piecewise` at the bound variable (`_numeric_piecewise`):
/// the first branch whose condition holds.
pub fn piecewise_value(e: &Basic, prec: usize) -> Option<Complex> {
    for pair in e.args() {
        if condition(&pair.args()[1], prec)? {
            return numeric_value(&pair.args()[0], prec);
        }
    }
    None
}

fn condition(c: &Basic, prec: usize) -> Option<bool> {
    match c.head() {
        Head::BooleanTrue => Some(true),
        Head::BooleanFalse => Some(false),
        Head::Relational(op) => {
            let (u, v) = (numeric_value(&c.args()[0], prec)?, numeric_value(&c.args()[1], prec)?);
            match op {
                RelOp::Eq => Some(u == v),
                RelOp::Ne => Some(u != v),
                _ => {
                    let ((p, pi), (q, qi)) = (parts(&u), parts(&v));
                    if !is_zero(&pi) || !is_zero(&qi) {
                        return None;
                    }
                    Some(match op {
                        RelOp::Lt => p < q,
                        RelOp::Le => p <= q,
                        RelOp::Gt => p > q,
                        _ => p >= q,
                    })
                }
            }
        }
        Head::And => {
            for a in c.args() {
                if !condition(a, prec)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        Head::Or => {
            for a in c.args() {
                if condition(a, prec)? {
                    return Some(true);
                }
            }
            Some(false)
        }
        Head::Not => Some(!condition(&c.args()[0], prec)?),
        _ => None,
    }
}
