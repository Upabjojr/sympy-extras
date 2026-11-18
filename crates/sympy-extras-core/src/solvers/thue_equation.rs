//! Thue equations `F(x, y) = m`, ported from
//! `sympy_extras.solvers.thue_equation`: all the integer solutions, by
//! Baker's method with the reduction of Tzanakis and de Weger (the units of
//! `Z[alpha]` with their saturation by Friedman's bound, the elements of a
//! given norm modulo units by a Minkowski lattice, the Baker–Wüstholz
//! bound, de Weger's LLL reduction, and the enumeration).
//!
//! The numerical parts in the Python use mpmath at 60 decimal places and
//! Python floats; here the multiprecision values are `dashu` floats at the
//! same binary precision (the conjugates from an exact isolation of the
//! roots refined by Newton's method instead of `polyroots`), and the
//! Python floats are `f64` with the same operations in the same order.
//! The fundamental units found may differ from Python's by the choice
//! between units of equal heights (the docstring of the Python allows both
//! answers); the solutions do not.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeSet, HashMap};

use crate::core::basic::Basic;
use crate::core::evalf::{complex, parts, real_int, real_rat, Complex, Real};
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::{rat_from_parts, Int, Rat};
use crate::polys::densebasic::dup_strip;
use crate::polys::domains::ZZ;
use crate::polys::euclidtools::dup_resultant;
use crate::polys::factortools::dup_zz_factor;

/// The working precision, in decimal places, of the numerical parts.
pub const PRECISION: usize = 60;

/// An element of `Z[alpha]` by its integer coordinates in `1, alpha, ...,
/// alpha**(n-1)`.
pub type OrderElement = Vec<Int>;

fn not_implemented(message: String) -> PolyError {
    PolyError::NotImplemented(message)
}

/// mpmath's binary precision for `dps` decimal places.
fn bits(dps: usize) -> usize {
    (((dps + 1) as f64) * std::f64::consts::LOG2_10).round() as usize
}

// ---------------------------------------------------------------------------
// the numbers

/// The multiprecision numbers at one precision (mpmath's `mp` context).
#[derive(Debug, Clone, Copy)]
struct Mp {
    prec: usize,
}

impl Mp {
    fn int(&self, n: &Int) -> Real {
        real_int(n, self.prec)
    }

    fn small(&self, n: i64) -> Real {
        real_int(&Int::from(n), self.prec)
    }

    fn rat(&self, r: &Rat) -> Real {
        real_rat(r, self.prec).with_precision(self.prec).value()
    }

    fn float(&self, x: f64) -> Real {
        Real::try_from(x)
            .expect("a finite float")
            .with_precision(self.prec)
            .value()
    }

    fn zero(&self) -> Real {
        self.small(0)
    }

    fn one(&self) -> Real {
        self.small(1)
    }

    fn c(&self, re: Real) -> Complex {
        complex(re, self.zero())
    }

    fn cint(&self, n: &Int) -> Complex {
        self.c(self.int(n))
    }

    fn pi(&self) -> Real {
        Real::pi(self.prec)
    }

    /// `10**(-k)`.
    fn tenth_power(&self, k: usize) -> Real {
        self.one() / self.int(&Int::from(10u8).pow(k))
    }
}

/// `ln`, `exp` and `sqrt` at the precision of the argument: dashu gives
/// the exact results (`ln(1) = 0`, `exp(0) = 1`) an unlimited precision,
/// on which the next transcendental operation fails.
trait AtPrecision: Sized {
    fn lnp(&self) -> Self;
    fn expp(&self) -> Self;
    fn sqrtp(&self) -> Self;
}

fn kept(value: Real, precision: usize) -> Real {
    if value.precision() == 0 {
        value.with_precision(precision.max(64)).value()
    } else {
        value
    }
}

impl AtPrecision for f64 {
    fn lnp(&self) -> f64 {
        self.ln()
    }

    fn expp(&self) -> f64 {
        self.exp()
    }

    fn sqrtp(&self) -> f64 {
        self.sqrt()
    }
}

impl AtPrecision for Real {
    fn lnp(&self) -> Real {
        kept(self.ln(), self.precision())
    }

    fn expp(&self) -> Real {
        kept(self.exp(), self.precision())
    }

    fn sqrtp(&self) -> Real {
        kept(self.sqrt(), self.precision())
    }
}

impl AtPrecision for Complex {
    fn lnp(&self) -> Complex {
        let (a, b) = parts(self);
        let precision: usize = a.precision().max(b.precision());
        let (x, y) = parts(&self.ln());
        complex(kept(x, precision), kept(y, precision))
    }

    fn expp(&self) -> Complex {
        let (a, b) = parts(self);
        let precision: usize = a.precision().max(b.precision());
        let (x, y) = parts(&self.exp());
        complex(kept(x, precision), kept(y, precision))
    }

    fn sqrtp(&self) -> Complex {
        let (a, b) = parts(self);
        let precision: usize = a.precision().max(b.precision());
        let (x, y) = parts(&self.sqrt());
        complex(kept(x, precision), kept(y, precision))
    }
}

fn re(z: &Complex) -> Real {
    parts(z).0
}

fn im(z: &Complex) -> Real {
    parts(z).1
}

fn conj(z: &Complex) -> Complex {
    let (a, b) = parts(z);
    complex(a, -b)
}

fn rabs(x: &Real) -> Real {
    if *x < Real::ZERO {
        -x.clone()
    } else {
        x.clone()
    }
}

fn cabs(z: &Complex) -> Real {
    let (a, b) = parts(z);
    if b == Real::ZERO {
        return rabs(&a);
    }
    if a == Real::ZERO {
        return rabs(&b);
    }
    (&a * &a + &b * &b).sqrtp()
}

fn f(x: &Real) -> f64 {
    x.to_f64().value()
}

fn nint(x: &Real) -> Int {
    x.round().to_int().value()
}

fn abs_int(n: &Int) -> Int {
    if *n < Int::ZERO {
        -n.clone()
    } else {
        n.clone()
    }
}

fn int_f64(n: &Int) -> f64 {
    n.to_f64().value()
}

fn small_int(n: &Int) -> i64 {
    i64::try_from(n).expect("a machine integer")
}

/// The argument in `(-pi, pi]` (exact on the real axis).
fn carg(z: &Complex, mp: Mp) -> Real {
    let (a, b) = parts(z);
    if b == Real::ZERO {
        return if a >= Real::ZERO { mp.zero() } else { mp.pi() };
    }
    im(&z.lnp())
}

/// The complex logarithm.
fn clog(z: &Complex, mp: Mp) -> Complex {
    complex(cabs(z).lnp(), carg(z, mp))
}

/// `exp(i theta)`.
fn cis(theta: &Real, mp: Mp) -> Complex {
    complex(mp.zero(), theta.clone()).expp()
}

/// `|det|`-free determinant of a square real matrix (Gaussian elimination
/// with partial pivoting, mpmath's `det`).
fn det(mut rows: Vec<Vec<Real>>, mp: Mp) -> Real {
    let n: usize = rows.len();
    let mut result: Real = mp.one();
    for c in 0..n {
        let p: usize = (c..n)
            .max_by(|a, b| rabs(&rows[*a][c]).partial_cmp(&rabs(&rows[*b][c])).unwrap())
            .unwrap();
        if rows[p][c] == Real::ZERO {
            return mp.zero();
        }
        if p != c {
            rows.swap(p, c);
            result = -result;
        }
        let pivot: Real = rows[c][c].clone();
        result *= &pivot;
        for r in c + 1..n {
            if rows[r][c] != Real::ZERO {
                let q: Real = &rows[r][c] / &pivot;
                for k in c..n {
                    let v: Real = &rows[r][k] - &q * &rows[c][k];
                    rows[r][k] = v;
                }
            }
        }
    }
    result
}

/// The inverse of a square real matrix (Gauss–Jordan), `None` when
/// singular.
fn inverse(rows: &[Vec<Real>], mp: Mp) -> Option<Vec<Vec<Real>>> {
    let n: usize = rows.len();
    let mut m: Vec<Vec<Real>> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r.iter()
                .cloned()
                .chain((0..n).map(|j| if i == j { mp.one() } else { mp.zero() }))
                .collect()
        })
        .collect();
    for c in 0..n {
        let p: usize = (c..n)
            .max_by(|a, b| rabs(&m[*a][c]).partial_cmp(&rabs(&m[*b][c])).unwrap())
            .unwrap();
        if m[p][c] == Real::ZERO {
            return None;
        }
        m.swap(p, c);
        let pivot: Real = m[c][c].clone();
        m[c] = m[c].iter().map(|v| v / &pivot).collect();
        for r in 0..n {
            if r != c && m[r][c] != Real::ZERO {
                let q: Real = m[r][c].clone();
                let row: Vec<Real> = m[r].iter().zip(&m[c]).map(|(a, b)| a - &q * b).collect();
                m[r] = row;
            }
        }
    }
    Some(m.into_iter().map(|r| r[n..].to_vec()).collect())
}

/// The solution of a square complex system (mpmath's `lu_solve`), `None`
/// when it is singular.
fn complex_solve(mut a: Vec<Vec<Complex>>, mut b: Vec<Complex>) -> Option<Vec<Complex>> {
    let n: usize = a.len();
    for c in 0..n {
        let p: usize = (c..n)
            .max_by(|x, y| cabs(&a[*x][c]).partial_cmp(&cabs(&a[*y][c])).unwrap())
            .unwrap();
        if cabs(&a[p][c]) == Real::ZERO {
            return None;
        }
        a.swap(p, c);
        b.swap(p, c);
        let pivot: Complex = a[c][c].clone();
        for r in c + 1..n {
            let q: Complex = &a[r][c] / &pivot;
            for k in c..n {
                let v: Complex = &a[r][k] - &q * &a[c][k];
                a[r][k] = v;
            }
            let v: Complex = &b[r] - &q * &b[c];
            b[r] = v;
        }
    }
    let mut x: Vec<Complex> = b.clone();
    for i in (0..n).rev() {
        let mut s: Complex = b[i].clone();
        for k in i + 1..n {
            s = &s - &a[i][k] * &x[k];
        }
        x[i] = &s / &a[i][i];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// the roots

fn horner_real(g: &[Int], x: &Real, mp: Mp) -> (Real, Real) {
    let mut value: Real = mp.zero();
    let mut derivative: Real = mp.zero();
    for c in g {
        derivative = &derivative * x + &value;
        value = &value * x + mp.int(c);
    }
    (value, derivative)
}

fn horner_complex(g: &[Int], z: &Complex, mp: Mp) -> (Complex, Complex) {
    let mut value: Complex = mp.c(mp.zero());
    let mut derivative: Complex = mp.c(mp.zero());
    for c in g {
        derivative = &derivative * z + &value;
        value = &value * z + mp.cint(c);
    }
    (value, derivative)
}

/// The roots of the square-free integer polynomial `g` at `prec` bits, in
/// mpmath's order: the real ones by value, then the others by `(|Im|,
/// Re)` with the conjugates side by side.
fn polynomial_roots(g: &[Int], mp: Mp) -> PolyResult<(Vec<Complex>, usize)> {
    let fq: Vec<Rat> = g.iter().map(|c| Rat::from(c.clone())).collect();
    let eps: Rat = rat_from_parts(Int::ONE, Int::ONE << 64);
    let (intervals, rectangles) = crate::polys::complexroots::dup_isolate_all_roots_sqf(&fq, Some(&eps))?;
    let work: Mp = Mp { prec: mp.prec + 32 };
    let tolerance: Real = work.one() / work.int(&(Int::ONE << (mp.prec + 8)));
    let mut reals: Vec<Real> = Vec::new();
    for (a, b) in &intervals {
        let mut x: Real = work.rat(&((a + b) / Rat::from(Int::from(2))));
        if a == b {
            reals.push(x.with_precision(mp.prec).value());
            continue;
        }
        for _ in 0..200 {
            let (v, d) = horner_real(g, &x, work);
            if d == Real::ZERO {
                break;
            }
            let step: Real = &v / &d;
            x = &x - &step;
            if rabs(&step) <= &tolerance * (work.one() + rabs(&x)) {
                break;
            }
        }
        reals.push(x.with_precision(mp.prec).value());
    }
    let mut others: Vec<Complex> = Vec::new();
    for r in &rectangles {
        let two: Rat = Rat::from(Int::from(2));
        let mut z: Complex = complex(
            work.rat(&((&r.re_min + &r.re_max) / &two)),
            work.rat(&((&r.im_min + &r.im_max) / &two)),
        );
        for _ in 0..200 {
            let (v, d) = horner_complex(g, &z, work);
            if cabs(&d) == Real::ZERO {
                break;
            }
            let step: Complex = &v / &d;
            z = &z - &step;
            if cabs(&step) <= &tolerance * (work.one() + cabs(&z)) {
                break;
            }
        }
        let (a, b) = parts(&z);
        let z: Complex = complex(a.with_precision(mp.prec).value(), b.with_precision(mp.prec).value());
        others.push(conj(&z));
        others.push(z);
    }
    reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // (|Im|, Re), the pair members kept together (negative imaginary part first)
    let mut paired: Vec<(Complex, Complex)> = others.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect();
    paired.sort_by(|p, q| {
        let key = |z: &Complex| (rabs(&im(z)), re(z));
        key(&p.1).partial_cmp(&key(&q.1)).unwrap()
    });
    let count: usize = reals.len();
    let mut roots: Vec<Complex> = reals.into_iter().map(|x| mp.c(x)).collect();
    for (a, b) in paired {
        roots.push(a);
        roots.push(b);
    }
    Ok((roots, count))
}

// ---------------------------------------------------------------------------
// arithmetic in Z[alpha]

/// The order `Z[alpha]` for a monic integer polynomial `g`: exact
/// arithmetic on coordinate vectors and numerical conjugates (`_Order`).
#[derive(Debug, Clone)]
pub struct Order {
    /// `[1, c_{n-1}, ..., c_0]`.
    pub g: Vec<Int>,
    pub n: usize,
    /// In decimal places.
    pub precision: usize,
    mp: Mp,
    roots: Vec<Complex>,
    pub real_indices: Vec<usize>,
    pub pairs: Vec<(usize, usize)>,
    pub r1: usize,
    pub r2: usize,
    pub rank: usize,
    powers: Vec<Vec<Int>>,
}

impl Order {
    pub fn new(g: &[Int], precision: usize) -> PolyResult<Order> {
        let n: usize = g.len() - 1;
        let mut order: Order = Order {
            g: g.to_vec(),
            n,
            precision,
            mp: Mp { prec: bits(precision) },
            roots: Vec::new(),
            real_indices: Vec::new(),
            pairs: Vec::new(),
            r1: 0,
            r2: 0,
            rank: 0,
            powers: Vec::new(),
        };
        order.set_precision(precision)?;
        let mp: Mp = order.mp;
        let threshold: Real = mp.rat(&rat_from_parts(Int::ONE, Int::from(10u8).pow(10)));
        let mut taken: BTreeSet<usize> = BTreeSet::new();
        for i in 0..n {
            if order.real_indices.contains(&i) || taken.contains(&i) {
                continue;
            }
            for j in 0..n {
                if j != i
                    && !taken.contains(&j)
                    && !order.real_indices.contains(&j)
                    && cabs(&(&order.roots[j] - &conj(&order.roots[i]))) < threshold
                {
                    order.pairs.push((i, j));
                    taken.insert(i);
                    taken.insert(j);
                    break;
                }
            }
        }
        order.r1 = order.real_indices.len();
        order.r2 = order.pairs.len();
        order.rank = order.r1 + order.r2 - 1;
        let mut current: Vec<Int> = vec![Int::ZERO; n];
        current[0] = Int::ONE;
        for _ in 0..2 * n {
            order.powers.push(current.clone());
            current = order.times_alpha(&current);
        }
        Ok(order)
    }

    /// Recompute the roots at the given precision (in decimal digits),
    /// keeping their order (`set_precision`).
    pub fn set_precision(&mut self, precision: usize) -> PolyResult<()> {
        self.precision = precision;
        self.mp = Mp { prec: bits(precision) };
        let (computed, count) = polynomial_roots(&self.g, self.mp)?;
        if self.roots.is_empty() {
            self.roots = computed;
            self.real_indices = (0..count).collect();
        } else {
            let mut remaining: Vec<Complex> = computed;
            let mut ordered: Vec<Complex> = Vec::new();
            for old in &self.roots {
                let k: usize = (0..remaining.len())
                    .min_by(|a, b| {
                        cabs(&(&remaining[*a] - old))
                            .partial_cmp(&cabs(&(&remaining[*b] - old)))
                            .unwrap()
                    })
                    .unwrap();
                ordered.push(remaining.remove(k));
            }
            self.roots = ordered;
        }
        for i in self.real_indices.clone() {
            self.roots[i] = self.mp.c(re(&self.roots[i]));
        }
        Ok(())
    }

    fn times_alpha(&self, v: &[Int]) -> Vec<Int> {
        let n: usize = self.n;
        let mut shifted: Vec<Int> = std::iter::once(Int::ZERO).chain(v[..n - 1].iter().cloned()).collect();
        let top: &Int = &v[n - 1];
        for i in 0..n {
            shifted[i] -= top * &self.g[n - i];
        }
        shifted
    }

    pub fn one(&self) -> OrderElement {
        let mut e: OrderElement = vec![Int::ZERO; self.n];
        e[0] = Int::ONE;
        e
    }

    pub fn multiply(&self, a: &[Int], b: &[Int]) -> OrderElement {
        let n: usize = self.n;
        let mut product: Vec<Int> = vec![Int::ZERO; 2 * n - 1];
        for (i, ai) in a.iter().enumerate() {
            if ai.is_zero() {
                continue;
            }
            for (j, bj) in b.iter().enumerate() {
                product[i + j] += ai * bj;
            }
        }
        let mut result: Vec<Int> = vec![Int::ZERO; n];
        for (k, coefficient) in product.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            for i in 0..n {
                result[i] += coefficient * &self.powers[k][i];
            }
        }
        result
    }

    pub fn power(&self, a: &[Int], e: i64) -> PolyResult<OrderElement> {
        let mut result: OrderElement = self.one();
        let mut base: OrderElement = a.to_vec();
        let mut e: i64 = e;
        if e < 0 {
            base = self.inverse_unit(a)?;
            e = -e;
        }
        while e > 0 {
            if e & 1 == 1 {
                result = self.multiply(&result, &base);
            }
            base = self.multiply(&base, &base);
            e >>= 1;
        }
        Ok(result)
    }

    /// The norm, as the resultant with `g`.
    pub fn norm(&self, a: &[Int]) -> Int {
        let element: Vec<Int> = dup_strip(&a.iter().rev().cloned().collect::<Vec<Int>>(), &ZZ);
        if element.is_empty() {
            return Int::ZERO;
        }
        dup_resultant(&self.g, &element, &ZZ)
    }

    pub fn conjugates(&self, a: &[Int]) -> Vec<Complex> {
        let mp: Mp = self.mp;
        self.roots
            .iter()
            .map(|root| {
                let mut total: Complex = mp.c(mp.zero());
                let mut power: Complex = mp.c(mp.one());
                for c in a {
                    total = &total + &(&mp.cint(c) * &power);
                    power = &power * root;
                }
                total
            })
            .collect()
    }

    /// The element with the given conjugates, when its coordinates are
    /// integers (`from_conjugates`).
    pub fn from_conjugates(&self, values: &[Complex]) -> Option<OrderElement> {
        let mp: Mp = self.mp;
        let n: usize = self.n;
        let v: Vec<Vec<Complex>> = self
            .roots
            .iter()
            .map(|root| {
                let mut row: Vec<Complex> = Vec::new();
                let mut power: Complex = mp.c(mp.one());
                for _ in 0..n {
                    row.push(power.clone());
                    power = &power * root;
                }
                row
            })
            .collect();
        let solution: Vec<Complex> = complex_solve(v, values.to_vec())?;
        let mut coordinates: OrderElement = Vec::new();
        for value in &solution {
            let (a, b) = parts(value);
            let rounded: Int = nint(&a);
            if f(&rabs(&b)) > 1e-6 || f(&rabs(&(&a - &mp.int(&rounded)))) > 1e-6 {
                return None;
            }
            coordinates.push(rounded);
        }
        let check: Vec<Complex> = self.conjugates(&coordinates);
        if check
            .iter()
            .zip(values)
            .any(|(u, w)| f(&cabs(&(u - w))) > 1e-6 * (1.0 + f(&cabs(w))))
        {
            return None;
        }
        Some(coordinates)
    }

    /// The inverse of a unit.
    pub fn inverse_unit(&self, u: &[Int]) -> PolyResult<OrderElement> {
        let one: Complex = self.mp.c(self.mp.one());
        let inverted: Vec<Complex> = self.conjugates(u).iter().map(|c| &one / c).collect();
        self.from_conjugates(&inverted)
            .ok_or_else(|| value_error(format!("{:?} is not a unit", u)))
    }

    /// `b/a` when it lies in the order.
    pub fn divides(&self, a: &[Int], b: &[Int]) -> Option<OrderElement> {
        let (ca, cb) = (self.conjugates(a), self.conjugates(b));
        let small: Real = self.mp.tenth_power(self.precision / 2);
        if ca.iter().any(|c| cabs(c) < small) {
            return None;
        }
        let quotients: Vec<Complex> = ca.iter().zip(&cb).map(|(u, v)| v / u).collect();
        self.from_conjugates(&quotients)
    }

    fn log_vector(&self, a: &[Int], indices: &[usize]) -> Vec<Real> {
        let c: Vec<Complex> = self.conjugates(a);
        indices.iter().map(|i| cabs(&c[*i]).lnp()).collect()
    }

    /// The absolute logarithmic Weil height of an algebraic integer.
    pub fn height(&self, a: &[Int]) -> f64 {
        let mut total: Real = self.mp.zero();
        for c in self.conjugates(a) {
            let l: Real = cabs(&c).lnp();
            if l > Real::ZERO {
                total += l;
            }
        }
        f(&(total / self.mp.small(self.n as i64)))
    }
}

// ---------------------------------------------------------------------------
// units

fn independent_indices(order: &Order) -> Vec<usize> {
    let mut indices: Vec<usize> = order.real_indices.clone();
    indices.extend(order.pairs.iter().map(|(i, _)| *i));
    indices.pop();
    indices
}

fn regulator(order: &Order, units: &[OrderElement]) -> Real {
    let indices: Vec<usize> = independent_indices(order);
    let rows: Vec<Vec<Real>> = units
        .iter()
        .map(|u| {
            let c: Vec<Complex> = order.conjugates(u);
            indices
                .iter()
                .map(|i| {
                    let l: Real = cabs(&c[*i]).lnp();
                    if order.real_indices.contains(i) {
                        l
                    } else {
                        order.mp.small(2) * l
                    }
                })
                .collect()
        })
        .collect();
    rabs(&det(rows, order.mp))
}

/// Units with coordinates in `[-bound, bound]` (a floating point filter on
/// the product of the conjugates, the exact norm afterwards).
fn small_units(order: &Order, bound: i64, limit: usize) -> Vec<OrderElement> {
    let mut found: Vec<OrderElement> = Vec::new();
    let n: usize = order.n;
    let roots: Vec<(f64, f64)> = order.roots.iter().map(|r| (f(&re(r)), f(&im(r)))).collect();
    let mut count: usize = 0;
    let mut coordinates: Vec<i64> = vec![-bound; n];
    loop {
        count += 1;
        if count > limit {
            break;
        }
        if coordinates.iter().any(|c| *c != 0) {
            let mut product: f64 = 1.0;
            for (rr, ri) in &roots {
                let (mut vr, mut vi) = (0.0f64, 0.0f64);
                for c in coordinates.iter().rev() {
                    let (nr, ni) = (vr * rr - vi * ri, vr * ri + vi * rr);
                    vr = nr + *c as f64;
                    vi = ni;
                }
                product *= vr.hypot(vi);
            }
            if (product - 1.0).abs() <= 1e-6 {
                let element: OrderElement = coordinates.iter().map(|c| Int::from(*c)).collect();
                if abs_int(&order.norm(&element)) == Int::ONE {
                    found.push(element);
                }
            }
        }
        // the next point of the cartesian product, the last coordinate fastest
        let mut k: usize = n;
        loop {
            if k == 0 {
                return found;
            }
            k -= 1;
            if coordinates[k] < bound {
                coordinates[k] += 1;
                break;
            }
            coordinates[k] = -bound;
        }
    }
    found
}

/// All the integer combinations `c` of the row vectors with `||sum c_i
/// rows_i|| <= radius` (Fincke–Pohst enumeration on the Gram–Schmidt
/// orthogonalisation, in floats); a nonzero `budget` bounds the number of
/// vectors (`NotImplementedError` beyond it).
pub fn short_vectors(rows: &[Vec<Int>], radius: f64, budget: i64) -> PolyResult<Vec<Vec<i64>>> {
    let m: usize = rows.len();
    let b: Vec<Vec<f64>> = rows.iter().map(|r| r.iter().map(int_f64).collect()).collect();
    let mut star: Vec<Vec<f64>> = Vec::new();
    let mut mu: Vec<Vec<f64>> = vec![vec![0.0; m]; m];
    let mut norms: Vec<f64> = Vec::new();
    for i in 0..m {
        let mut v: Vec<f64> = b[i].clone();
        for j in 0..i {
            mu[i][j] = if norms[j] != 0.0 {
                let mut s: f64 = 0.0;
                for (a, c) in b[i].iter().zip(&star[j]) {
                    s += a * c;
                }
                s / norms[j]
            } else {
                0.0
            };
            v = v.iter().zip(&star[j]).map(|(a, c)| a - mu[i][j] * c).collect();
        }
        let mut s: f64 = 0.0;
        for a in &v {
            s += a * a;
        }
        star.push(v);
        norms.push(s);
    }
    struct Search<'a> {
        m: usize,
        mu: &'a [Vec<f64>],
        norms: &'a [f64],
        coefficients: Vec<i64>,
        results: Vec<Vec<i64>>,
        budget: i64,
    }
    fn search(s: &mut Search, i: usize, remaining: f64) -> PolyResult<()> {
        if s.norms[i] <= 0.0 {
            return Ok(());
        }
        let mut total: f64 = 0.0;
        for j in i + 1..s.m {
            total += s.coefficients[j] as f64 * s.mu[j][i];
        }
        let center: f64 = -total;
        let span: f64 = (remaining.max(0.0) / s.norms[i]).sqrtp();
        let (low, high) = ((center - span).ceil() as i64, (center + span).floor() as i64);
        for c in low..=high {
            s.coefficients[i] = c;
            let used: f64 = (c as f64 - center).powi(2) * s.norms[i];
            if used > remaining * (1.0 + 1e-9) {
                continue;
            }
            if i == 0 {
                s.results.push(s.coefficients.clone());
                if s.budget != 0 && s.results.len() as i64 > s.budget {
                    return Err(not_implemented(format!(
                        "more than {} lattice points to examine; the enumeration of the elements of a given norm \
                         is linear in that norm (see the module documentation)",
                        s.budget
                    )));
                }
            } else {
                search(s, i - 1, remaining - used)?;
            }
        }
        s.coefficients[i] = 0;
        Ok(())
    }
    if m == 0 {
        return Ok(Vec::new());
    }
    let mut s: Search = Search {
        m,
        mu: &mu,
        norms: &norms,
        coefficients: vec![0; m],
        results: Vec::new(),
        budget,
    };
    search(&mut s, m - 1, radius * radius * (1.0 + 1e-9))?;
    Ok(s.results)
}

fn minkowski_rows(order: &Order) -> Vec<Vec<Real>> {
    (0..order.n)
        .map(|j| {
            let mut element: OrderElement = vec![Int::ZERO; order.n];
            element[j] = Int::ONE;
            let c: Vec<Complex> = order.conjugates(&element);
            let mut row: Vec<Real> = order.real_indices.iter().map(|i| re(&c[*i])).collect();
            for (i, _) in &order.pairs {
                row.push(re(&c[*i]));
                row.push(im(&c[*i]));
            }
            row
        })
        .collect()
}

/// The largest number of lattice points examined when looking for the
/// elements of a given norm.
const ENUMERATION_BUDGET: i64 = 400000;

/// An LLL reduced basis of the lattice spanned by the integer row vectors
/// (`lll`, `delta = 3/4`).
pub fn lll(rows: &[Vec<Int>]) -> PolyResult<Vec<Vec<Int>>> {
    crate::polys::lattice::lll(rows, &crate::polys::lattice::default_delta())
}

fn cartesian(ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
    let mut result: Vec<Vec<i64>> = vec![Vec::new()];
    for (low, high) in ranges {
        let mut next: Vec<Vec<i64>> = Vec::new();
        for prefix in &result {
            for v in *low..=*high {
                let mut longer: Vec<i64> = prefix.clone();
                longer.push(v);
                next.push(longer);
            }
        }
        result = next;
    }
    result
}

fn elements_by_lattice(order: &Order, norm: &Int, limits: &[f64]) -> PolyResult<Vec<OrderElement>> {
    let mp: Mp = order.mp;
    let embedding: Vec<Vec<Real>> = minkowski_rows(order);
    let weights: Vec<i64> = std::iter::repeat(1)
        .take(order.r1)
        .chain(std::iter::repeat(2).take(order.r2))
        .collect();
    let places: usize = order.r1 + order.r2;
    if places <= 1 {
        return Ok(Vec::new());
    }
    let free: usize = places - 1;
    let n: usize = order.n;
    let abs_norm: Int = abs_int(norm);
    let size: Real = (mp.int(&abs_norm).lnp() / mp.small(n as i64)).expp();
    let mut found: BTreeSet<OrderElement> = BTreeSet::new();
    let scale: Int = Int::from(10u8).pow(12);
    let (budget, mut examined) = (ENUMERATION_BUDGET, 0i64);
    let ranges: Vec<(i64, i64)> = limits[..free]
        .iter()
        .map(|l| (-(l.ceil() as i64), l.ceil() as i64))
        .collect();
    for grid in cartesian(&ranges) {
        let mut logs: Vec<f64> = grid.iter().map(|v| *v as f64).collect();
        let mut total: f64 = 0.0;
        for (w, l) in weights[..weights.len() - 1].iter().zip(&logs) {
            total += *w as f64 * l;
        }
        let last: f64 = -total / weights[weights.len() - 1] as f64;
        if last.abs() > limits[limits.len() - 1] + 1.0 {
            continue;
        }
        logs.push(last);
        let mut factors: Vec<Real> = Vec::new();
        for i in 0..order.r1 {
            factors.push(mp.float(-logs[i]).expp() / &size);
        }
        for k in 0..order.r2 {
            let v: Real = mp.float(-logs[order.r1 + k]).expp() / &size;
            factors.push(v.clone());
            factors.push(v);
        }
        let scale_real: Real = mp.int(&scale);
        let rows: Vec<Vec<Int>> = embedding
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&factors)
                    .map(|(v, w)| nint(&(v * w * &scale_real)))
                    .collect()
            })
            .collect();
        let augmented: Vec<Vec<Int>> = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .cloned()
                    .chain((0..n).map(|j| if i == j { scale.clone() } else { Int::ZERO }))
                    .collect()
            })
            .collect();
        let reduced: Vec<Vec<Int>> = lll(&augmented)?;
        let coordinates_of: Vec<Vec<Int>> = reduced
            .iter()
            .map(|row| {
                row[row.len() - n..]
                    .iter()
                    .map(|v| crate::numbers::div_floor(v, &scale))
                    .collect()
            })
            .collect();
        let radius: f64 = (n as f64).sqrtp() * 1.5f64.expp() * 1e12;
        let lattice: Vec<Vec<Int>> = reduced.iter().map(|row| row[..row.len() - n].to_vec()).collect();
        let vectors: Vec<Vec<i64>> = short_vectors(&lattice, radius, budget - examined)?;
        examined += vectors.len() as i64;
        for combination in vectors {
            let mut coordinates: OrderElement = vec![Int::ZERO; n];
            for (c, vector) in combination.iter().zip(&coordinates_of) {
                if *c != 0 {
                    for i in 0..n {
                        coordinates[i] += Int::from(*c) * &vector[i];
                    }
                }
            }
            if coordinates.iter().all(|v| v.is_zero()) || found.contains(&coordinates) {
                continue;
            }
            let mut product: Real = mp.one();
            for c in order.conjugates(&coordinates) {
                product *= cabs(&c);
            }
            let target: f64 = int_f64(&abs_norm);
            if f(&rabs(&(product - mp.int(&abs_norm)))) > 1e-6 * target {
                continue;
            }
            let exact: Int = order.norm(&coordinates);
            if exact == *norm || (abs_norm == Int::ONE && abs_int(&exact) == Int::ONE) {
                found.insert(coordinates);
            }
        }
    }
    Ok(found.into_iter().collect())
}

fn units_by_lattice(order: &Order, l: i64) -> PolyResult<Vec<OrderElement>> {
    elements_by_lattice(order, &Int::ONE, &vec![l as f64; order.r1 + order.r2])
}

/// The roots of unity among the candidates, as the group they generate.
fn torsion(order: &Order, candidates: &[OrderElement]) -> Vec<OrderElement> {
    let roots: Vec<&OrderElement> = candidates
        .iter()
        .filter(|u| order.conjugates(u).iter().all(|c| (f(&cabs(c)) - 1.0).abs() < 1e-10))
        .collect();
    let mut group: BTreeSet<OrderElement> = BTreeSet::new();
    group.insert(order.one());
    for u in roots {
        let mut current: OrderElement = u.clone();
        for _ in 0..60 {
            if group.contains(&current) {
                break;
            }
            group.insert(current.clone());
            current = order.multiply(&current, u);
        }
    }
    group.into_iter().collect()
}

fn sorted_by_height(order: &Order, units: &[OrderElement]) -> Vec<OrderElement> {
    let mut keyed: Vec<(f64, OrderElement)> = units.iter().map(|u| (order.height(u), u.clone())).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    keyed.into_iter().map(|(_, u)| u).collect()
}

fn independent_system(order: &Order, units: &[OrderElement]) -> Option<Vec<OrderElement>> {
    let indices: Vec<usize> = independent_indices(order);
    let mp: Mp = order.mp;
    let mut chosen: Vec<OrderElement> = Vec::new();
    let mut rows: Vec<Vec<Real>> = Vec::new();
    for u in sorted_by_height(order, units) {
        let row: Vec<Real> = order.log_vector(&u, &indices);
        if row.iter().all(|v| f(&rabs(v)) < 1e-12) {
            continue;
        }
        let mut trial: Vec<Vec<Real>> = rows.clone();
        trial.push(row.clone());
        let gram: Vec<Vec<Real>> = trial
            .iter()
            .map(|a| {
                trial
                    .iter()
                    .map(|b| {
                        let mut s: Real = mp.zero();
                        for (x, y) in a.iter().zip(b) {
                            s += x * y;
                        }
                        s
                    })
                    .collect()
            })
            .collect();
        if f(&rabs(&det(gram, mp))) > 1e-12 {
            rows.push(row);
            chosen.push(u);
            if chosen.len() == order.rank {
                return Some(chosen);
            }
        }
    }
    None
}

fn reduce_system(order: &Order, units: &[OrderElement]) -> PolyResult<Vec<OrderElement>> {
    let indices: Vec<usize> = independent_indices(order);
    let scale: Real = order.mp.int(&Int::from(10u8).pow(12));
    let r: usize = units.len();
    let augmented: Vec<Vec<Int>> = units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            order
                .log_vector(u, &indices)
                .iter()
                .map(|v| nint(&(v * &scale)))
                .chain((0..r).map(|j| if i == j { Int::ONE } else { Int::ZERO }))
                .collect()
        })
        .collect();
    let reduced: Vec<Vec<Int>> = lll(&augmented)?;
    let mut result: Vec<OrderElement> = Vec::new();
    for i in 0..r {
        let mut element: OrderElement = order.one();
        for j in 0..r {
            let e: &Int = &reduced[i][indices.len() + j];
            if !e.is_zero() {
                element = order.multiply(&element, &order.power(&units[j], small_int(e))?);
            }
        }
        result.push(element);
    }
    Ok(result)
}

/// A unit `v` with `v**p == u`.
fn pth_root(order: &Order, u: &[Int], p: u64) -> PolyResult<Option<OrderElement>> {
    let mp: Mp = order.mp;
    let c: Vec<Complex> = order.conjugates(u);
    let inverse_p: Real = mp.one() / mp.small(p as i64);
    let mut branches: Vec<Vec<Complex>> = Vec::new();
    for i in 0..order.n {
        if order.real_indices.contains(&i) {
            let value: Real = re(&c[i]);
            let root: Real = (rabs(&value).lnp() * &inverse_p).expp();
            branches.push(if value > Real::ZERO {
                vec![mp.c(root)]
            } else if p % 2 == 1 {
                vec![mp.c(-root)]
            } else {
                Vec::new()
            });
        } else {
            let principal: Complex = (clog(&c[i], mp) * mp.c(inverse_p.clone())).expp();
            branches.push(
                (0..p)
                    .map(|k| {
                        let theta: Real = mp.small(2) * mp.pi() * mp.small(k as i64) / mp.small(p as i64);
                        &principal * &cis(&theta, mp)
                    })
                    .collect(),
            );
        }
    }
    if branches.iter().any(|b| b.is_empty()) {
        return Ok(None);
    }
    let first_of_pair: HashMap<usize, usize> = order.pairs.iter().map(|(i, j)| (*i, *j)).collect();
    let second_of_pair: HashMap<usize, usize> = order.pairs.iter().map(|(i, j)| (*j, *i)).collect();
    let free: Vec<usize> = (0..order.n).filter(|i| !second_of_pair.contains_key(i)).collect();
    let ranges: Vec<(i64, i64)> = free.iter().map(|i| (0, branches[*i].len() as i64 - 1)).collect();
    for choice in cartesian(&ranges) {
        let mut values: Vec<Complex> = vec![mp.c(mp.zero()); order.n];
        for (i, k) in free.iter().zip(&choice) {
            values[*i] = branches[*i][*k as usize].clone();
            if let Some(j) = first_of_pair.get(i) {
                values[*j] = conj(&branches[*i][*k as usize]);
            }
        }
        if let Some(candidate) = order.from_conjugates(&values) {
            if order.power(&candidate, p as i64)? == u {
                return Ok(Some(candidate));
            }
        }
    }
    Ok(None)
}

fn saturate(order: &Order, units: Vec<OrderElement>, torsion: &[OrderElement]) -> PolyResult<Vec<OrderElement>> {
    let mut units: Vec<OrderElement> = units;
    let friedman: Real = order.mp.rat(&rat_from_parts(Int::from(2052), Int::from(10000)));
    loop {
        let index_bound: Int = (regulator(order, &units) / &friedman).floor().to_int().value();
        let mut enlarged: bool = false;
        for p in crate::ntheory::generate::primerange(&Int::from(2), &(index_bound + Int::ONE)) {
            let p: u64 = u64::try_from(&p).expect("a small prime");
            if (p as f64).powi(units.len() as i32) > 2000000.0 {
                return Err(not_implemented(format!(
                    "the saturation of the unit group needs the prime {}",
                    p
                )));
            }
            let ranges: Vec<(i64, i64)> = vec![(0, p as i64 - 1); units.len()];
            for exponents in cartesian(&ranges) {
                let Some(first) = exponents.iter().position(|e| *e != 0) else {
                    continue;
                };
                if exponents[first] != 1 {
                    continue;
                }
                let mut base: OrderElement = order.one();
                for (u, e) in units.iter().zip(&exponents) {
                    if *e != 0 {
                        base = order.multiply(&base, &order.power(u, *e)?);
                    }
                }
                for zeta in torsion {
                    if let Some(root) = pth_root(order, &order.multiply(&base, zeta), p)? {
                        units[first] = root;
                        enlarged = true;
                        break;
                    }
                }
                if enlarged {
                    break;
                }
            }
            if enlarged {
                break;
            }
        }
        if !enlarged {
            return reduce_system(order, &units);
        }
    }
}

/// The element or its negative, with a positive first nonzero coordinate.
pub fn normalized(element: &[Int]) -> OrderElement {
    for c in element {
        if !c.is_zero() {
            return if *c > Int::ZERO {
                element.to_vec()
            } else {
                element.iter().map(|v| -v.clone()).collect()
            };
        }
    }
    element.to_vec()
}

fn units_of(order: &Order, bound: i64) -> PolyResult<(Vec<OrderElement>, Vec<OrderElement>)> {
    if order.rank == 0 {
        let candidates: Vec<OrderElement> = small_units(order, 2, 4000000);
        return Ok((Vec::new(), torsion(order, &candidates)));
    }
    let mut candidates: Vec<OrderElement> = small_units(order, 3, 4000000);
    let roots_of_unity: Vec<OrderElement> = torsion(order, &candidates);
    let mut system: Option<Vec<OrderElement>> = independent_system(order, &candidates);
    let mut l: i64 = 4;
    while system.is_none() && l <= bound {
        candidates.extend(units_by_lattice(order, l)?);
        system = independent_system(order, &candidates);
        l *= 2;
    }
    let Some(system) = system else {
        return Err(not_implemented(format!(
            "no system of {} independent units with logarithms up to {}",
            order.rank, bound
        )));
    };
    let reduced: Vec<OrderElement> = reduce_system(order, &system)?;
    let units: Vec<OrderElement> = saturate(order, reduced, &roots_of_unity)?
        .iter()
        .map(|u| normalized(u))
        .collect();
    Ok((units, roots_of_unity))
}

/// A fundamental system of units and the roots of unity of `Z[alpha]`,
/// `alpha` a root of the monic integer polynomial `g` (coefficients from
/// the leading one), as coordinate vectors (`units_of_order`).
pub fn units_of_order(g: &[Int], bound: i64) -> PolyResult<(Vec<OrderElement>, Vec<OrderElement>)> {
    units_of(&Order::new(g, PRECISION)?, bound)
}

/// Representatives of the elements of `Z[alpha]` of the given norm modulo
/// the units, one per class of associates (`elements_of_norm`).
pub fn elements_of_norm(g: &[Int], norm: &Int, units: &[OrderElement], limit: usize) -> PolyResult<Vec<OrderElement>> {
    elements_of_norm_in(&Order::new(g, PRECISION)?, norm, units, limit)
}

fn elements_of_norm_in(
    order: &Order,
    norm: &Int,
    units: &[OrderElement],
    limit: usize,
) -> PolyResult<Vec<OrderElement>> {
    let mp: Mp = order.mp;
    let n: usize = order.n;
    let mut bounds: Vec<Real> = vec![mp.zero(); n];
    for u in units {
        let c: Vec<Complex> = order.conjugates(u);
        for i in 0..n {
            bounds[i] += rabs(&cabs(&c[i]).lnp()) / mp.small(2);
        }
    }
    let mut limits: Vec<f64> = order.real_indices.iter().map(|i| f(&bounds[*i])).collect();
    limits.extend(order.pairs.iter().map(|(i, _)| f(&bounds[*i])));
    let mut total: u128 = 1;
    for l in &limits[..limits.len() - 1] {
        total = total.saturating_mul(2 * l.ceil() as u128 + 1);
    }
    if total > limit as u128 {
        return Err(not_implemented(format!(
            "too many lattice boxes ({}) for the elements of norm {}",
            total, norm
        )));
    }
    let mut found: Vec<OrderElement> = elements_by_lattice(order, norm, &limits)?;
    found.sort_by(|a, b| {
        let size = |e: &OrderElement| e.iter().map(abs_int).fold(Int::ZERO, |s, c| s + c);
        size(a).cmp(&size(b)).then_with(|| a.cmp(b))
    });
    let mut representatives: Vec<OrderElement> = Vec::new();
    for element in found {
        if representatives.iter().any(|other| associates(order, &element, other)) {
            continue;
        }
        representatives.push(normalized(&element));
    }
    Ok(representatives)
}

fn associates(order: &Order, a: &[Int], b: &[Int]) -> bool {
    match order.divides(a, b) {
        Some(q) => abs_int(&order.norm(&q)) == Int::ONE,
        None => false,
    }
}

// ---------------------------------------------------------------------------
// the linear form, Baker's bound and the reduction

fn factorial(n: usize) -> Int {
    (1..=n).fold(Int::ONE, |p, k| p * Int::from(k))
}

/// The constant `C` with `log|Lambda| >= -C log B` (Baker and Wüstholz).
fn baker_wustholz(k: usize, d: usize, heights: &[f64]) -> f64 {
    let exact: Int = Int::from(18) * factorial(k + 1) * Int::from(k).pow(k + 1) * Int::from(32 * d).pow(k + 2);
    let mut c: f64 = int_f64(&exact) * ((2 * k * d) as f64).lnp();
    for h in heights {
        c *= h;
    }
    c
}

/// The solution of a square rational system (`_solve_rational`).
fn solve_rational(a: &[Vec<Rat>], b: &[Rat]) -> Option<Vec<Rat>> {
    let n: usize = a.len();
    let mut m: Vec<Vec<Rat>> = a
        .iter()
        .zip(b)
        .map(|(row, v)| row.iter().cloned().chain(std::iter::once(v.clone())).collect())
        .collect();
    for col in 0..n {
        let pivot: usize = (col..n).find(|r| !m[*r][col].is_zero())?;
        m.swap(col, pivot);
        let p: Rat = m[col][col].clone();
        for r in 0..n {
            if r != col && !m[r][col].is_zero() {
                let q: Rat = &m[r][col] / &p;
                let row: Vec<Rat> = m[r].iter().zip(&m[col]).map(|(x, c)| x - &q * c).collect();
                m[r] = row;
            }
        }
    }
    Some((0..n).map(|i| &m[i][n] / &m[i][i]).collect())
}

/// Python's `round` of a fraction: to the nearest integer, ties to even.
fn round_half_even(v: &Rat) -> Int {
    let (p, q) = (v.numerator().clone(), Int::from(v.denominator().clone()));
    let floor: Int = crate::numbers::div_floor(&p, &q);
    let twice_rest: Int = (&p - &floor * &q) * Int::from(2);
    if twice_rest > q || (twice_rest == q && (&floor % Int::from(2)) != Int::ZERO) {
        floor + Int::ONE
    } else {
        floor
    }
}

/// de Weger's reduction: a new bound on `B`, `None` when the lattice is
/// not large enough (`_lll_reduce`).
fn lll_reduce(deltas: &[Real], delta0: &Real, b0: f64, k1: f64, k2: f64, mp: Mp) -> PolyResult<Option<f64>> {
    let s: usize = deltas.len();
    for factor in [100.0f64, 1e4, 1e6] {
        let base: Real = mp.float(b0) * mp.float(factor);
        let c: Real = (0..s).fold(mp.one(), |p, _| p * &base);
        let mut rows: Vec<Vec<Int>> = Vec::new();
        for i in 0..s - 1 {
            let mut row: Vec<Int> = (0..s - 1).map(|j| if j == i { Int::ONE } else { Int::ZERO }).collect();
            row.push(nint(&(&c * &deltas[i])));
            rows.push(row);
        }
        let mut last: Vec<Int> = vec![Int::ZERO; s - 1];
        last.push(nint(&(&c * &deltas[s - 1])));
        rows.push(last);
        let mut y: Vec<Rat> = vec![Rat::ZERO; s - 1];
        y.push(Rat::from(-nint(&(&c * delta0))));
        let reduced: Vec<Vec<Int>> = lll(&rows)?;
        if reduced.len() != s {
            continue;
        }
        let a: Vec<Vec<Rat>> = (0..s)
            .map(|i| (0..s).map(|j| Rat::from(reduced[j][i].clone())).collect())
            .collect();
        let Some(sigma) = solve_rational(&a, &y) else { continue };
        let distances: Vec<Rat> = sigma
            .iter()
            .map(|v| {
                let d: Rat = v - Rat::from(round_half_even(v));
                if d < Rat::ZERO {
                    -d
                } else {
                    d
                }
            })
            .collect();
        let Some(i0) = distances.iter().rposition(|d| !d.is_zero()) else {
            continue;
        };
        let mut squares: Real = mp.zero();
        for v in &reduced[0] {
            let x: Real = mp.int(v);
            squares += &x * &x;
        }
        let length: Real = squares.sqrtp();
        let distance: &Rat = &distances[i0];
        let two_power: Real = (-(mp.small(s as i64 - 1) / mp.small(2)) * mp.small(2).lnp()).expp();
        let d: Real =
            two_power * mp.int(distance.numerator()) / mp.int(&Int::from(distance.denominator().clone())) * length;
        let b0r: Real = mp.float(b0);
        let inside: Real = &d * &d - mp.small(s as i64 - 1) * &b0r * &b0r;
        if inside <= Real::ZERO {
            continue;
        }
        let shift: Real = (mp.small(s as i64) * &b0r + mp.one()) / mp.small(2);
        let big_s: Real = inside.sqrtp() - shift;
        if big_s <= Real::ZERO {
            continue;
        }
        let bound: Real = ((&c * mp.float(k1)).lnp() - big_s.lnp()) / mp.float(k2);
        return Ok(Some(f(&bound)));
    }
    Ok(None)
}

fn form_coefficients(
    order: &Order,
    units: &[OrderElement],
    mu: &[Int],
    i0: usize,
    j: usize,
    k: usize,
    complex_pair: bool,
) -> (Vec<Real>, Real) {
    let mp: Mp = order.mp;
    let roots: &[Complex] = &order.roots;
    let cmu: Vec<Complex> = order.conjugates(mu);
    let gamma0: Complex = &(&(&roots[k] - &roots[i0]) * &cmu[j]) / &(&(&roots[j] - &roots[i0]) * &cmu[k]);
    let gammas: Vec<Complex> = units
        .iter()
        .map(|u| {
            let c: Vec<Complex> = order.conjugates(u);
            &c[j] / &c[k]
        })
        .collect();
    if complex_pair {
        let mut coefficients: Vec<Real> = gammas.iter().map(|g| carg(g, mp)).collect();
        coefficients.push(mp.small(2) * mp.pi());
        return (coefficients, carg(&gamma0, mp));
    }
    (gammas.iter().map(|g| cabs(g).lnp()).collect(), cabs(&gamma0).lnp())
}

fn independent_indices_excluding(order: &Order, i0: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = order.real_indices.iter().copied().filter(|i| *i != i0).collect();
    indices.extend(order.pairs.iter().map(|(i, _)| *i));
    indices
}

/// A linear form of the reduction: `(i0, mu, K1, K2, (j, k), complex_pair)`.
struct Form {
    i0: usize,
    mu: OrderElement,
    k1: f64,
    k2: f64,
    j: usize,
    k: usize,
    complex_pair: bool,
}

/// The data of `F(x, y) = m` prepared for the resolution (`ThueEquation`).
#[derive(Debug, Clone)]
pub struct ThueEquation {
    pub coefficients: Vec<Int>,
    pub m: Int,
    pub n: usize,
    pub a0: Int,
    pub g: Vec<Int>,
    pub order: Order,
    pub norm: Int,
    pub units: Vec<OrderElement>,
    pub torsion: Vec<OrderElement>,
    pub representatives: Vec<OrderElement>,
}

impl ThueEquation {
    pub fn new(coefficients: &[Int], m: &Int, bound: i64) -> PolyResult<ThueEquation> {
        let n: usize = coefficients.len() - 1;
        let a0: Int = coefficients[0].clone();
        if a0.is_zero() || n < 3 {
            return Err(value_error(
                "a form of degree at least 3 with a nonzero leading coefficient is expected".to_string(),
            ));
        }
        let mut g: Vec<Int> = vec![Int::ONE];
        for i in 1..=n {
            g.push(&coefficients[i] * a0.pow(i - 1));
        }
        let order: Order = Order::new(&g, PRECISION)?;
        let norm: Int = a0.pow(n - 1) * m;
        let (units, torsion) = units_of(&order, bound)?;
        let representatives: Vec<OrderElement> = elements_of_norm_in(&order, &norm, &units, 100000)?;
        Ok(ThueEquation {
            coefficients: coefficients.to_vec(),
            m: m.clone(),
            n,
            a0,
            g,
            order,
            norm,
            units,
            torsion,
            representatives,
        })
    }

    pub fn evaluate(&self, x: &Int, y: &Int) -> Int {
        let mut total: Int = Int::ZERO;
        for (i, c) in self.coefficients.iter().enumerate() {
            total += c * x.pow(self.n - i) * y.pow(i);
        }
        total
    }

    fn solution_from_element(&self, beta: &[Int]) -> Option<(Int, Int)> {
        if beta[2..].iter().any(|c| !c.is_zero()) {
            return None;
        }
        if !(&beta[0] % &self.a0).is_zero() {
            return None;
        }
        let (x, y) = (crate::numbers::div_floor(&beta[0], &self.a0), -beta[1].clone());
        if self.evaluate(&x, &y) == self.m {
            Some((x, y))
        } else {
            None
        }
    }

    /// The solutions with `|y| <= Y`, by the integer roots of `F(X, y) - m`.
    pub fn small_solutions(&self, bound: i64) -> PolyResult<Vec<(Int, Int)>> {
        let mut found: Vec<(Int, Int)> = Vec::new();
        for y in -bound..=bound {
            let yi: Int = Int::from(y);
            let mut polynomial: Vec<Int> = self
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * yi.pow(i))
                .collect();
            let last: usize = polynomial.len() - 1;
            polynomial[last] -= &self.m;
            let polynomial: Vec<Int> = dup_strip(&polynomial, &ZZ);
            if polynomial.len() < 2 {
                continue;
            }
            let (_, factors) = dup_zz_factor(&polynomial, &ZZ)?;
            let mut roots: Vec<Int> = factors
                .iter()
                .filter(|(f, _)| f.len() == 2 && f[0] == Int::ONE)
                .map(|(f, _)| -f[1].clone())
                .collect();
            roots.sort();
            found.extend(roots.into_iter().map(|r| (r, yi.clone())));
        }
        Ok(found)
    }

    /// All the integer solutions.
    pub fn solve(&mut self) -> PolyResult<Vec<(Int, Int)>> {
        let base: usize = self.order.precision;
        let result = self.solve_inner();
        if self.order.precision != base {
            self.order.set_precision(base)?;
        }
        result
    }

    fn solve_inner(&mut self) -> PolyResult<Vec<(Int, Int)>> {
        let n: usize = self.n;
        let mp: Mp = self.order.mp;
        let roots: Vec<Complex> = self.order.roots.clone();
        let n0: Int = abs_int(&self.norm);
        let mut solutions: BTreeSet<(Int, Int)> = BTreeSet::new();
        let nth = |x: Real| -> Real { (x.lnp() / mp.small(n as i64)).expp() };
        if self.order.r1 == 0 {
            let mut product: Real = mp.one();
            for r in &roots {
                product *= rabs(&im(r));
            }
            let y: i64 = small_int(&nth(mp.int(&n0) / product).floor().to_int().value()) + 1;
            let found: BTreeSet<(Int, Int)> = self.small_solutions(y)?.into_iter().collect();
            return Ok(found.into_iter().collect());
        }
        let r: usize = self.order.rank;
        if r == 0 {
            return Err(not_implemented("unit rank zero".to_string()));
        }
        let size: Real = nth(mp.int(&n0));
        let mut y1: Real = mp.one();
        let imaginary: Vec<Real> = roots
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.order.real_indices.contains(i))
            .map(|(_, r)| rabs(&im(r)))
            .collect();
        if let Some(smallest) = imaginary.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            let v: Real = &size / smallest;
            if v > y1 {
                y1 = v;
            }
        }
        for &i0 in &self.order.real_indices {
            let d: Real = (0..n)
                .filter(|j| *j != i0)
                .map(|j| cabs(&(&roots[j] - &roots[i0])))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let v: Real = mp.small(2) * &size / d;
            if v > y1 {
                y1 = v;
            }
        }
        let mut bound_b: f64 = 0.0;
        let mut forms: Vec<Form> = Vec::new();
        let identity: OrderElement = self.order.one();
        for &i0 in &self.order.real_indices.clone() {
            let order: &Order = &self.order;
            let others: Vec<usize> = (0..n).filter(|j| *j != i0).collect();
            let distance: HashMap<usize, Real> =
                others.iter().map(|j| (*j, cabs(&(&roots[*j] - &roots[i0])))).collect();
            let d: Real = others
                .iter()
                .map(|j| distance[j].clone())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let mut c1: Real = mp.int(&n0) * mp.int(&(Int::ONE << (n - 1)));
            for j in &others {
                c1 /= &distance[j];
            }
            let mut candidates: Vec<(usize, usize, bool)> = order.pairs.iter().map(|(a, b)| (*a, *b, true)).collect();
            for a in 0..others.len() {
                for b in 0..others.len() {
                    if a != b {
                        candidates.push((others[a], others[b], false));
                    }
                }
            }
            let first: &OrderElement = self.representatives.first().unwrap_or(&identity);
            let tiny: Real = mp.tenth_power(order.precision / 3);
            let mut chosen: Option<(usize, usize, bool)> = None;
            for (j_, k_, pair_) in candidates {
                let (coefficients, _) = form_coefficients(order, &self.units, first, i0, j_, k_, pair_);
                if coefficients.iter().all(|c| rabs(c) > tiny) {
                    chosen = Some((j_, k_, pair_));
                    break;
                }
            }
            let Some((j, k, complex_pair)) = chosen else {
                return Err(not_implemented(format!(
                    "no linear form with nonzero coefficients for the root {}",
                    i0
                )));
            };
            let c6: Real = mp.small(2) * cabs(&(&roots[j] - &roots[k])) * &c1 / (&distance[&j] * &distance[&k]);
            let v: Real = nth(mp.small(2) * &c6);
            if v > y1 {
                y1 = v;
            }
            let jj: Vec<usize> = independent_indices_excluding(order, i0);
            let matrix: Vec<Vec<Real>> = jj
                .iter()
                .map(|idx| {
                    self.units
                        .iter()
                        .map(|u| cabs(&order.conjugates(u)[*idx]).lnp())
                        .collect()
                })
                .collect();
            let minv: Vec<Vec<Real>> =
                inverse(&matrix, mp).ok_or_else(|| value_error("a singular matrix of logarithms".to_string()))?;
            let c4: Real = (0..r)
                .map(|a| (0..r).fold(mp.zero(), |s, b| s + rabs(&minv[a][b])))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let c8: Real = jj
                .iter()
                .map(|idx| {
                    let first: Real = (cabs(&roots[i0]) + &d / mp.small(2) + cabs(&roots[*idx])).lnp();
                    let second: Real = rabs(&(&distance[idx] / mp.small(2)).lnp());
                    if first >= second {
                        first
                    } else {
                        second
                    }
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let h_alpha: f64 = {
                let mut alpha: OrderElement = vec![Int::ZERO; n];
                alpha[1] = Int::ONE;
                order.height(&alpha)
            };
            let degree: usize = (1..=n).product();
            for mu in &self.representatives {
                let cmu: Vec<Complex> = order.conjugates(mu);
                let largest: Real = jj
                    .iter()
                    .map(|idx| rabs(&cabs(&cmu[*idx]).lnp()))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let c3: Real = &c4 * (&c8 + largest);
                let gamma0: Complex = &(&(&roots[k] - &roots[i0]) * &cmu[j]) / &(&(&roots[j] - &roots[i0]) * &cmu[k]);
                let gammas: Vec<Complex> = self
                    .units
                    .iter()
                    .map(|u| {
                        let c: Vec<Complex> = order.conjugates(u);
                        &c[j] / &c[k]
                    })
                    .collect();
                let degree_f: f64 = degree as f64;
                let h0: f64 = 2.0 * (2.0 * h_alpha + 2f64.lnp()) + 2.0 * order.height(mu);
                let mut heights: Vec<f64> = vec![h0.max(f(&cabs(&clog(&gamma0, mp))) / degree_f).max(1.0 / degree_f)];
                for (u, gl) in self.units.iter().zip(&gammas) {
                    heights.push(
                        (2.0 * order.height(u))
                            .max(f(&cabs(&clog(gl, mp))) / degree_f)
                            .max(1.0 / degree_f),
                    );
                }
                let (kk, extra) = if complex_pair {
                    heights.push(0.0f64.max(std::f64::consts::PI / degree_f).max(1.0 / degree_f));
                    (r + 2, true)
                } else {
                    (r + 1, false)
                };
                let big_c: f64 = baker_wustholz(kk, degree, &heights);
                let (c3f, c4f) = (f(&c3), f(&c4));
                let log_2c6: f64 = f(&(mp.small(2) * &c6).lnp());
                let mut l: f64 = 1e40;
                for _ in 0..200 {
                    let mut bp: f64 = c3f + c4f * l;
                    if extra {
                        bp = r as f64 * bp + 2.0;
                    }
                    let new: f64 = (log_2c6 + big_c * 1.0f64.max(bp.max(3.0).lnp())) / n as f64;
                    if (new - l).abs() < 1e-6 * 1.0f64.max(l) {
                        l = new;
                        break;
                    }
                    l = new;
                }
                let mut b0: f64 = c3f + c4f * l;
                if extra {
                    b0 = r as f64 * b0 + 2.0;
                }
                let nr: Real = mp.small(n as i64);
                let (k1, k2) = if extra {
                    let exponent: Real = &nr * (&c3 / &c4 + mp.small(2) / (mp.small(r as i64) * &c4));
                    (
                        f(&(mp.small(2) * &c6 * exponent.expp())),
                        f(&(&nr / (mp.small(r as i64) * &c4))),
                    )
                } else {
                    (f(&(mp.small(2) * &c6 * (&nr * &c3 / &c4).expp())), f(&(&nr / &c4)))
                };
                forms.push(Form {
                    i0,
                    mu: mu.clone(),
                    k1,
                    k2,
                    j,
                    k,
                    complex_pair,
                });
                bound_b = bound_b.max(b0);
            }
        }
        // the reduction, on every form, from the common initial bound
        let mut current: f64 = bound_b;
        let base_precision: usize = self.order.precision;
        for _ in 0..12 {
            let mut best: f64 = 0.0;
            let digits: usize = ((r + 2) as f64 * (current.max(10.0).lnp() / 10f64.lnp() + 6.0)) as usize + 40;
            self.order.set_precision(base_precision.max(digits))?;
            for form in &forms {
                let (deltas, delta0) = form_coefficients(
                    &self.order,
                    &self.units,
                    &form.mu,
                    form.i0,
                    form.j,
                    form.k,
                    form.complex_pair,
                );
                let reduced: f64 =
                    lll_reduce(&deltas, &delta0, current, form.k1, form.k2, self.order.mp)?.unwrap_or(current);
                best = best.max(reduced);
            }
            if best >= current * 0.9 {
                current = best;
                break;
            }
            current = best;
        }
        self.order.set_precision(base_precision)?;
        let b_final: i64 = current.ceil() as i64 + 1;
        if b_final > 200 {
            return Err(not_implemented(format!(
                "the reduction of the exponent bound stopped at {}",
                b_final
            )));
        }
        let small_bound: i64 = small_int(&y1.floor().to_int().value()) + 1;
        for solution in self.small_solutions(small_bound)? {
            solutions.insert(solution);
        }
        let order: &Order = &self.order;
        let mut powers: HashMap<(usize, i64), OrderElement> = HashMap::new();
        for (index, u) in self.units.iter().enumerate() {
            for e in -b_final..=b_final {
                if e != 0 {
                    powers.insert((index, e), order.power(u, e)?);
                }
            }
        }
        let ranges: Vec<(i64, i64)> = vec![(-b_final, b_final); r];
        for mu in &self.representatives {
            for exponents in cartesian(&ranges) {
                let mut element: OrderElement = mu.clone();
                for (index, e) in exponents.iter().enumerate() {
                    if *e != 0 {
                        element = order.multiply(&element, &powers[&(index, *e)]);
                    }
                }
                for zeta in &self.torsion {
                    let beta: OrderElement = order.multiply(&element, zeta);
                    if let Some(found) = self.solution_from_element(&beta) {
                        solutions.insert(found);
                    }
                }
            }
        }
        Ok(solutions.into_iter().collect())
    }
}

/// All the integer solutions of the Thue equation `F(x, y) = m` for an
/// irreducible binary form of degree at least 3 in the generators `x` and
/// `y` (`thue`).
pub fn thue(form: &Basic, m: &Int, x: &Basic, y: &Basic, bound: i64) -> PolyResult<Vec<(Int, Int)>> {
    let integer_error = || value_error("integer coefficients are expected".to_string());
    let poly = crate::polys::polytools::Poly::new(form, &[x.clone(), y.clone()]).map_err(|_| integer_error())?;
    let n: i64 = poly.total_degree();
    let terms = poly.terms()?;
    if terms
        .iter()
        .any(|(monomial, _)| monomial.iter().map(|e| *e as i64).sum::<i64>() != n)
        || n < 3
    {
        return Err(value_error(
            "a homogeneous form of degree at least 3 is expected".to_string(),
        ));
    }
    let n: usize = n as usize;
    let mut coefficient_of: HashMap<(u32, u32), Int> = HashMap::new();
    for (monomial, c) in &terms {
        let value: &Rat = c.as_rational().ok_or_else(integer_error)?;
        if !value.is_int() {
            return Err(integer_error());
        }
        coefficient_of.insert((monomial[0], monomial[1]), value.numerator().clone());
    }
    let coefficients: Vec<Int> = (0..=n)
        .map(|i| {
            coefficient_of
                .get(&((n - i) as u32, i as u32))
                .cloned()
                .unwrap_or(Int::ZERO)
        })
        .collect();
    let (_, factors) = dup_zz_factor(&dup_strip(&coefficients, &ZZ), &ZZ)?;
    let irreducible: bool = factors.len() == 1 && factors[0].1 == 1;
    if !irreducible || coefficients[0].is_zero() {
        return Err(value_error(
            "an irreducible form with a nonzero leading coefficient is expected".to_string(),
        ));
    }
    if m.is_zero() {
        return Ok(vec![(Int::ZERO, Int::ZERO)]);
    }
    ThueEquation::new(&coefficients, m, bound)?.solve()
}
