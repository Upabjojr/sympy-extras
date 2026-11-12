//! Port of `sympy_extras/polys/cad/samplepoints.py`: exact real algebraic
//! sample points for cylindrical algebraic decomposition. A real algebraic
//! number is an [`Algebraic`] of `qqbar` (shared, [`Real`], so that the
//! refinements of its isolating interval are seen by every holder, as the
//! Python's objects are), and a [`SamplePoint`] is a real algebraic
//! `theta` with the coordinates as elements of `QQ(theta)`. Signs and
//! comparisons are decided by interval arithmetic, refining the interval
//! of `theta`; the field is extended by a primitive element when a new
//! algebraic coordinate is added.

use std::cell::RefCell;
use std::sync::Arc;

use crate::core::basic::Basic;
use crate::core::symbol::Symbol;
use crate::error::{polys_error, value_error, PolyResult, PolysErrorKind};
use crate::numbers::*;

use super::super::densearith::{dup_div, dup_neg};
use super::super::densebasic::{dup_degree, dup_strip};
use super::super::densetools::{dup_diff, dup_eval};
use super::super::domains::{AlgebraicElement, AlgebraicField, Domain, QQ, ZZ};
use super::super::euclidtools::dup_gcd;
use super::super::factortools::{dup_ext_norm, dup_zz_factor};
use super::super::numberfields::qqbar::{sum_resultant, Algebraic};
use super::super::polytools::Poly;
use super::super::rings::Monomial;
use super::super::rootisolation::{dup_count_real_roots, dup_isolate_real_roots_sqf};
use super::super::sqfreetools::{dup_sqf_list, dup_sqf_p};

/// A sign: -1, 0 or 1.
pub type Sign = i32;

/// A shared real algebraic number (`RealAlgebraic`).
pub type Real = Arc<Algebraic>;

fn polynomial_error(message: String) -> crate::error::PolyError {
    polys_error(PolysErrorKind::PolynomialError, message)
}

/// The real algebraic number of a rational.
pub fn rational(q: &Rat) -> Real {
    Arc::new(Algebraic::from_rational(q))
}

fn real_checked(r: &Real) -> PolyResult<Real> {
    if !r.is_real() {
        return Err(value_error(format!("{} is not real", r.repr()?)));
    }
    Ok(r.clone())
}

fn bounds(r: &Algebraic) -> PolyResult<(Rat, Rat)> {
    r.real_interval()
}

/// `f` primitive with a positive leading coefficient.
fn normalised(f: &[Int]) -> Vec<Int> {
    let mut g: Int = Int::ZERO;
    for c in f {
        g = abs(&gcd(&g, c));
    }
    if f.first().is_some_and(is_negative) {
        g = -g;
    }
    if g.is_zero() || g == Int::ONE {
        return f.to_vec();
    }
    f.iter().map(|c| div_floor(c, &g)).collect()
}

/// The irreducible factors of positive degree, normalised, with their
/// multiplicities.
fn irreducible_factors(f: &[Int]) -> PolyResult<Vec<(Vec<Int>, usize)>> {
    let (_, factors) = dup_zz_factor(f, &ZZ)?;
    Ok(factors
        .into_iter()
        .filter(|(g, _)| g.len() > 1)
        .map(|(g, k)| (normalised(&g), k))
        .collect())
}

fn real_roots_of_irreducible(g: &[Int]) -> Vec<Real> {
    if g.len() == 2 {
        return vec![rational(&rat_from_parts(-g[1].clone(), g[0].clone()))];
    }
    let rational_g: Vec<Rat> = g.iter().map(rat_from_int).collect();
    dup_isolate_real_roots_sqf(&rational_g, None)
        .into_iter()
        .map(|interval| Arc::new(Algebraic::real(g.to_vec(), interval)))
        .collect()
}

fn integer_dense(f: &[Rat]) -> Vec<Int> {
    let mut d: Int = Int::ONE;
    for c in f {
        let den: Int = denom(c);
        d = &d * &den / abs(&gcd(&d, &den));
    }
    let scale: Rat = rat_from_int(&d);
    f.iter().map(|c| numer(&(c * &scale))).collect()
}

/// The exact comparison of two real algebraic numbers (`compare_real`).
pub fn compare_real(a: &Real, b: &Real) -> PolyResult<Sign> {
    let (x, y) = (real_checked(a)?, real_checked(b)?);
    if Arc::ptr_eq(&x, &y) {
        return Ok(0);
    }
    let same: bool = x.minpoly() == y.minpoly() && x.degree() > 1;
    let rational_poly: Vec<Rat> = if same {
        x.minpoly().iter().map(rat_from_int).collect()
    } else {
        Vec::new()
    };
    loop {
        let (alo, ahi) = bounds(&x)?;
        let (blo, bhi) = bounds(&y)?;
        if ahi < blo || (ahi == blo && (alo < ahi || blo < bhi)) {
            return Ok(-1);
        }
        if bhi < alo || (bhi == alo && (blo < bhi || alo < ahi)) {
            return Ok(1);
        }
        if alo == ahi && blo == bhi {
            return Ok(0);
        }
        let (lo, hi) = (alo.clone().min(blo.clone()), ahi.clone().max(bhi.clone()));
        if same && dup_count_real_roots(&rational_poly, Some(&lo), Some(&hi)) == 1 {
            return Ok(0);
        }
        if &ahi - &alo >= &bhi - &blo {
            x.refine_real()?;
        } else {
            y.refine_real()?;
        }
    }
}

/// The positions of `values` in increasing order, equal numbers grouped
/// (`_order`; the stable merge sort of Python's `sorted`).
fn order(values: &[Real]) -> PolyResult<Vec<Vec<usize>>> {
    let bounds_list: RefCell<Vec<(Rat, Rat)>> =
        RefCell::new(values.iter().map(|v| bounds(v)).collect::<PolyResult<_>>()?);
    let compare = |i: usize, j: usize| -> PolyResult<Sign> {
        {
            let b = bounds_list.borrow();
            let ((_, ahi), (blo, _)) = (&b[i], &b[j]);
            if ahi < blo {
                return Ok(-1);
            }
            let ((alo, _), (_, bhi)) = (&b[i], &b[j]);
            if bhi < alo {
                return Ok(1);
            }
        }
        let c: Sign = compare_real(&values[i], &values[j])?;
        let mut b = bounds_list.borrow_mut();
        b[i] = bounds(&values[i])?;
        b[j] = bounds(&values[j])?;
        Ok(c)
    };
    let mut positions: Vec<usize> = (0..values.len()).collect();
    let mut failure: Option<crate::error::PolyError> = None;
    positions.sort_by(|&i, &j| {
        if failure.is_some() {
            return std::cmp::Ordering::Equal;
        }
        match compare(i, j) {
            Ok(c) => c.cmp(&0),
            Err(e) => {
                failure = Some(e);
                std::cmp::Ordering::Equal
            }
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in positions {
        let joins: bool = match groups.last() {
            Some(g) => compare(*g.last().expect("a member"), i)? == 0,
            None => false,
        };
        if joins {
            groups.last_mut().expect("a group").push(i);
        } else {
            groups.push(vec![i]);
        }
    }
    Ok(groups)
}

fn dense_real_roots(f: &[Rat]) -> PolyResult<Vec<(Real, usize)>> {
    let g: Vec<Rat> = dup_strip(f, &QQ);
    if dup_degree(&g) < 1 {
        return Ok(Vec::new());
    }
    let mut values: Vec<Real> = Vec::new();
    let mut multiplicities: Vec<usize> = Vec::new();
    for (h, m) in irreducible_factors(&integer_dense(&g))? {
        for r in real_roots_of_irreducible(&h) {
            values.push(r);
            multiplicities.push(m);
        }
    }
    Ok(order(&values)?
        .into_iter()
        .map(|group| (values[group[0]].clone(), group.iter().map(|i| multiplicities[*i]).sum()))
        .collect())
}

/// The distinct real roots in increasing order of the polynomial with the
/// rational coefficients `f` (highest degree first).
pub fn real_roots_of(f: &[Rat]) -> PolyResult<Vec<Real>> {
    Ok(dense_real_roots(f)?.into_iter().map(|(r, _)| r).collect())
}

fn univariate(f: &Poly) -> PolyResult<Vec<Rat>> {
    if f.gens.len() != 1 {
        return Err(polynomial_error(format!(
            "a univariate polynomial expected, got {}",
            f.to_str()?
        )));
    }
    let n: usize = f.degree(0)?.max(0) as usize;
    let mut dense: Vec<Rat> = vec![Rat::ZERO; n + 1];
    for (m, c) in &f.rep.terms {
        dense[n - m[0] as usize] = c.clone();
    }
    Ok(dense)
}

/// The distinct real roots of the univariate `f` over `QQ` with their
/// multiplicities (`rational_real_roots`).
pub fn rational_real_roots(f: &Poly) -> PolyResult<Vec<(Real, usize)>> {
    dense_real_roots(&univariate(f)?)
}

fn floor_rat(q: &Rat) -> Int {
    div_floor(&numer(q), &denom(q))
}

/// `floor(2**k a)` exactly (`floor_scaled`).
pub fn floor_scaled(a: &Real, k: i64) -> PolyResult<Int> {
    let r: Real = real_checked(a)?;
    let scale: Rat = if k >= 0 {
        rat_from_int(&(Int::ONE << k as usize))
    } else {
        rat_from_parts(Int::ONE, Int::ONE << (-k) as usize)
    };
    loop {
        let (lo, hi) = bounds(&r)?;
        let (flo, fhi) = (floor_rat(&(&lo * &scale)), floor_rat(&(&hi * &scale)));
        if flo == fhi {
            return Ok(flo);
        }
        r.refine_real()?;
    }
}

/// The rational with the smallest denominator in the interval `(lo, hi)`
/// (`hi = None` for infinity; the ends included unless open).
pub fn simplest_between(lo: &Rat, hi: Option<&Rat>, lo_open: bool, hi_open: bool) -> PolyResult<Rat> {
    let zero: Rat = Rat::ZERO;
    let Some(hi) = hi else {
        if *lo < zero || (*lo == zero && !lo_open) {
            return Ok(zero);
        }
        let n: Int = floor_rat(lo);
        let exact: bool = rat_from_int(&n) == *lo;
        return Ok(rat_from_int(&if exact && !lo_open { n } else { n + Int::ONE }));
    };
    if lo > hi || (lo == hi && (lo_open || hi_open)) {
        return Err(value_error("empty interval".to_string()));
    }
    if (*lo < zero || (*lo == zero && !lo_open)) && (*hi > zero || (*hi == zero && !hi_open)) {
        return Ok(zero);
    }
    if *hi < zero || (*hi == zero && hi_open) {
        return Ok(-simplest_between(&-hi.clone(), Some(&-lo.clone()), hi_open, lo_open)?);
    }
    let n: Int = floor_rat(lo);
    let n_rat: Rat = rat_from_int(&n);
    let c: Int = if n_rat == *lo && !lo_open {
        n.clone()
    } else {
        &n + Int::ONE
    };
    let c_rat: Rat = rat_from_int(&c);
    if c_rat < *hi || (c_rat == *hi && !hi_open) {
        return Ok(c_rat);
    }
    let one: Rat = Rat::ONE;
    let inner_hi: Option<Rat> = if *lo > n_rat { Some(&one / &(lo - &n_rat)) } else { None };
    let inner: Rat = simplest_between(&(&one / &(hi - &n_rat)), inner_hi.as_ref(), hi_open, lo_open)?;
    Ok(n_rat + one / inner)
}

fn dyadic_bounds(r: &Real, k: i64) -> PolyResult<(Rat, Rat, bool, bool)> {
    if r.is_rational() {
        let c: Rat = r.rational_value()?;
        return Ok((c.clone(), c, true, true));
    }
    let scale: Rat = rat_from_int(&(Int::ONE << k as usize));
    let lo: Rat = rat_from_int(&floor_scaled(r, k)?) / &scale;
    let hi: Rat = &lo + Rat::ONE / &scale;
    Ok((lo, hi, false, false))
}

/// A simple rational strictly between `a < b` (`rational_between`).
pub fn rational_between(a: &Real, b: &Real) -> PolyResult<Rat> {
    let (x, y) = (real_checked(a)?, real_checked(b)?);
    if compare_real(&x, &y)? >= 0 {
        return Err(value_error(format!("{} is not smaller than {}", a.repr()?, b.repr()?)));
    }
    let mut k: i64 = 0;
    loop {
        let (_, hi, _, hi_open) = dyadic_bounds(&x, k)?;
        let (lo, _, lo_open, _) = dyadic_bounds(&y, k)?;
        if hi < lo || (hi == lo && !(hi_open || lo_open)) {
            return simplest_between(&hi, Some(&lo), hi_open, lo_open);
        }
        k += 1;
    }
}

/// A simple rational smaller than `a` (`rational_below`).
pub fn rational_below(a: &Real) -> PolyResult<Rat> {
    let r: Real = real_checked(a)?;
    if compare_real(&r, &rational(&Rat::ZERO))? > 0 {
        return Ok(Rat::ZERO);
    }
    let mut fl: Int = floor_scaled(&r, 0)?;
    if r.is_rational() && rat_from_int(&fl) == r.rational_value()? {
        fl -= Int::ONE;
    }
    Ok(rat_from_int(&fl))
}

/// A simple rational greater than `a` (`rational_above`).
pub fn rational_above(a: &Real) -> PolyResult<Rat> {
    let r: Real = real_checked(a)?;
    if compare_real(&r, &rational(&Rat::ZERO))? < 0 {
        return Ok(Rat::ZERO);
    }
    Ok(rat_from_int(&(floor_scaled(&r, 0)? + Int::ONE)))
}

/// The sign of the element `a` of `QQ(theta)` by interval arithmetic
/// (`_sign_in_field`).
fn sign_in_field(theta: &Algebraic, a: &[Rat]) -> PolyResult<Sign> {
    if a.is_empty() {
        return Ok(0);
    }
    loop {
        let (lo, hi) = bounds(theta)?;
        let (mut vlo, mut vhi): (Rat, Rat) = (Rat::ZERO, Rat::ZERO);
        for c in a {
            let products: [Rat; 4] = [&vlo * &lo, &vlo * &hi, &vhi * &lo, &vhi * &hi];
            let min: Rat = products.iter().min().expect("products").clone();
            let max: Rat = products.iter().max().expect("products").clone();
            vlo = min + c;
            vhi = max + c;
        }
        if vlo > Rat::ZERO {
            return Ok(1);
        }
        if vhi < Rat::ZERO {
            return Ok(-1);
        }
        theta.refine_real()?;
    }
}

fn field_of(theta: &Algebraic) -> PolyResult<AlgebraicField> {
    let minpoly: Vec<Int> = theta.minpoly();
    let lc: Int = minpoly[0].clone();
    AlgebraicField::new(
        minpoly.iter().map(|c| rat_from_parts(c.clone(), lc.clone())).collect(),
        Symbol::new("theta", &[]).0,
    )
}

/// The primitive element `gamma = theta + k*beta` of `QQ(theta, beta)`,
/// its field, and `theta`, `beta` in it (`_join`).
fn join(theta: &Real, beta: &Real) -> PolyResult<(Real, AlgebraicField, AlgebraicElement, AlgebraicElement)> {
    let (m1, m2) = (theta.minpoly(), beta.minpoly());
    let mut k: i64 = 1;
    let big_n: Vec<Int> = loop {
        let scaled: Vec<Int> = m2.iter().enumerate().map(|(i, c)| c * Int::from(k).pow(i)).collect();
        let n: Vec<Int> = sum_resultant(&m1, &scaled);
        if dup_sqf_p(&n, &ZZ) {
            break n;
        }
        k += 1;
    };
    let mut candidates: Vec<Real> = Vec::new();
    for (g, _) in irreducible_factors(&big_n)? {
        candidates.extend(real_roots_of_irreducible(&g));
    }
    let kk: Rat = rat_from_int(&Int::from(k));
    let alive: Vec<Real> = loop {
        let (tlo, thi) = bounds(theta)?;
        let (blo, bhi) = bounds(beta)?;
        let (lo, hi) = (&tlo + &kk * &blo, &thi + &kk * &bhi);
        let mut alive: Vec<Real> = Vec::new();
        for r in &candidates {
            let (rlo, rhi) = bounds(r)?;
            if !(rhi < lo || hi < rlo) {
                alive.push(r.clone());
            }
        }
        if alive.len() == 1 {
            break alive;
        }
        theta.refine_real()?;
        beta.refine_real()?;
        for r in &alive {
            r.refine_real()?;
        }
    };
    let gamma: Real = alive[0].clone();
    if gamma.is_rational() {
        return Err(polynomial_error("unexpected rational primitive element".to_string()));
    }
    let field: AlgebraicField = field_of(&gamma)?;
    let from_int = |c: &Int| field.from_rational(&rat_from_int(c));
    // beta is the only common root of m2(t) and m1(gamma - k*t)
    let u: [AlgebraicElement; 2] = [from_int(&Int::from(-k)), field.alpha()];
    let mut f: Vec<AlgebraicElement> = Vec::new();
    for c in &m1 {
        let mut product: Vec<AlgebraicElement> = if f.is_empty() {
            Vec::new()
        } else {
            vec![field.zero(); f.len() + 1]
        };
        for (i, v) in f.iter().enumerate() {
            product[i] = field.add(&product[i], &field.mul(v, &u[0]));
            product[i + 1] = field.add(&product[i + 1], &field.mul(v, &u[1]));
        }
        if product.is_empty() {
            product = if c.is_zero() { Vec::new() } else { vec![from_int(c)] };
        } else {
            let last: usize = product.len() - 1;
            product[last] = field.add(&product[last], &from_int(c));
        }
        f = dup_strip(&product, &field);
    }
    let m2_field: Vec<AlgebraicElement> = m2.iter().map(from_int).collect();
    let g: Vec<AlgebraicElement> = dup_gcd(&m2_field, &f, &field);
    if dup_degree(&g) != 1 {
        return Err(polynomial_error("failed to compute a primitive element".to_string()));
    }
    let beta_k: AlgebraicElement = field.neg(&field.quo(&g[1], &g[0])?);
    let theta_k: AlgebraicElement = field.sub(&field.alpha(), &field.mul(&from_int(&Int::from(k)), &beta_k));
    Ok((gamma, field, theta_k, beta_k))
}

/// Refine the pairwise distinct `candidates` until their intervals are
/// disjoint (`_separate`).
fn separate(candidates: &[Real]) -> PolyResult<()> {
    loop {
        let b: Vec<(Rat, Rat)> = candidates.iter().map(|r| bounds(r)).collect::<PolyResult<_>>()?;
        let mut crowded: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for i in 0..candidates.len() {
            for j in i + 1..candidates.len() {
                if !(b[i].1 < b[j].0 || b[j].1 < b[i].0) {
                    crowded.insert(i);
                    crowded.insert(j);
                }
            }
        }
        if crowded.is_empty() {
            return Ok(());
        }
        for i in crowded {
            candidates[i].refine_real()?;
        }
    }
}

fn sturm<K: Domain>(f: &[K::Elem], k: &K) -> PolyResult<Vec<Vec<K::Elem>>> {
    let mut sequence: Vec<Vec<K::Elem>> = vec![f.to_vec(), dup_diff(f, 1, k)];
    while !sequence.last().expect("a member").is_empty() {
        let n: usize = sequence.len();
        let (_, r) = dup_div(&sequence[n - 2], &sequence[n - 1], k)?;
        sequence.push(dup_neg(&r, k));
    }
    sequence.pop();
    Ok(sequence)
}

fn variations(
    sturm: &[Vec<AlgebraicElement>],
    value: &Rat,
    theta: &Algebraic,
    field: &AlgebraicField,
) -> PolyResult<usize> {
    let point: AlgebraicElement = field.from_rational(value);
    let mut nonzero: Vec<Sign> = Vec::new();
    for f in sturm {
        let s: Sign = sign_in_field(theta, &dup_eval(f, &point, field))?;
        if s != 0 {
            nonzero.push(s);
        }
    }
    Ok(nonzero.windows(2).filter(|w| w[0] != w[1]).count())
}

/// The distinct real roots of `f` over `QQ(theta)` with their
/// multiplicities, in increasing order (`_real_roots_over`).
fn real_roots_over(theta: &Real, field: &AlgebraicField, f: &[AlgebraicElement]) -> PolyResult<Vec<(Real, usize)>> {
    let mut found: Vec<(Real, usize)> = Vec::new();
    for (g, multiplicity) in dup_sqf_list(f, field).1 {
        if dup_degree(&g) < 1 {
            continue;
        }
        let candidates: Vec<Real> = dense_real_roots(&dup_ext_norm(&g, field))?
            .into_iter()
            .map(|(r, _)| r)
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let sequence: Vec<Vec<AlgebraicElement>> = sturm(&g, field)?;
        separate(&candidates)?;
        let mut total: usize = 0;
        for r in &candidates {
            if r.is_rational() {
                if dup_eval(&g, &field.from_rational(&r.rational_value()?), field).is_empty() {
                    found.push((r.clone(), multiplicity));
                    total += 1;
                }
                continue;
            }
            let (lo, hi) = bounds(r)?;
            let number: i64 =
                variations(&sequence, &lo, theta, field)? as i64 - variations(&sequence, &hi, theta, field)? as i64;
            if number == 1 {
                found.push((r.clone(), multiplicity));
                total += 1;
            } else if number != 0 {
                return Err(polynomial_error(format!(
                    "{} roots of a factor in the isolating interval of {}",
                    number,
                    r.repr()?
                )));
            }
        }
        let below: Rat = bounds(&candidates[0])?.0 - Rat::ONE;
        let above: Rat = bounds(candidates.last().expect("a candidate"))?.1 + Rat::ONE;
        let expected: i64 =
            variations(&sequence, &below, theta, field)? as i64 - variations(&sequence, &above, theta, field)? as i64;
        if total as i64 != expected {
            return Err(polynomial_error(
                "the real roots of a factor are not among those of its norm".to_string(),
            ));
        }
    }
    let roots: Vec<Real> = found.iter().map(|(r, _)| r.clone()).collect();
    Ok(order(&roots)?
        .into_iter()
        .map(|group| found[group[0]].clone())
        .collect())
}

/// A polynomial with the coordinates of a sample point put for all its
/// variables but the last (`Specialization`).
#[derive(Debug, Clone)]
pub struct Specialization {
    /// The degree in the last variable at the point, `-1` for a
    /// polynomial vanishing identically there.
    pub degree: i64,
    /// The sign of the leading coefficient at the point.
    pub leading_sign: Sign,
    /// The distinct real roots in increasing order with their
    /// multiplicities.
    pub roots: Vec<(Real, usize)>,
}

impl Specialization {
    /// The sign between the roots of index `position - 1` and `position`.
    pub fn sign_before(&self, position: usize) -> PolyResult<Sign> {
        if position > self.roots.len() {
            return Err(value_error(format!("there are {} roots", self.roots.len())));
        }
        let odd: bool = self.roots[position..].iter().map(|(_, m)| m).sum::<usize>() % 2 == 1;
        Ok(if odd { -self.leading_sign } else { self.leading_sign })
    }
}

/// The polynomial with the rational `terms` at the point `values` of `k`:
/// with `last` a dense polynomial in the variable after them, otherwise a
/// list holding the value (`_evaluate`).
#[allow(clippy::needless_range_loop)]
fn evaluate<K: Domain>(
    terms: &[(Monomial, Rat)],
    values: &[K::Elem],
    k: &K,
    last: bool,
    convert: &dyn Fn(&Rat) -> K::Elem,
) -> Vec<K::Elem> {
    let n: usize = values.len();
    let mut powers: Vec<std::collections::HashMap<u32, K::Elem>> = values
        .iter()
        .map(|v| [(0u32, k.one()), (1u32, v.clone())].into_iter().collect())
        .collect();
    fn power<K: Domain>(
        powers: &mut [std::collections::HashMap<u32, K::Elem>],
        values: &[K::Elem],
        k: &K,
        i: usize,
        e: u32,
    ) -> K::Elem {
        if let Some(found) = powers[i].get(&e) {
            return found.clone();
        }
        let previous: K::Elem = power(powers, values, k, i, e - 1);
        let found: K::Elem = k.mul(&previous, &values[i]);
        powers[i].insert(e, found.clone());
        found
    }
    let mut collected: std::collections::BTreeMap<u32, K::Elem> = std::collections::BTreeMap::new();
    for (m, c) in terms {
        let mut value: K::Elem = convert(c);
        for i in 0..n {
            if m[i] != 0 {
                let p: K::Elem = power(&mut powers, values, k, i, m[i]);
                value = k.mul(&value, &p);
            }
        }
        let e: u32 = if last { m[n] } else { 0 };
        let total: K::Elem = match collected.remove(&e) {
            None => value,
            Some(previous) => k.add(&previous, &value),
        };
        collected.insert(e, total);
    }
    let Some(&top) = collected.keys().next_back() else {
        return Vec::new();
    };
    let dense: Vec<K::Elem> = (0..=top)
        .map(|i| collected.get(&(top - i)).cloned().unwrap_or_else(|| k.zero()))
        .collect();
    dup_strip(&dense, k)
}

/// A point with real algebraic coordinates (`SamplePoint`).
#[derive(Debug, Clone)]
pub struct SamplePoint {
    pub field: Option<Arc<AlgebraicField>>,
    pub theta: Option<Real>,
    pub coords: Vec<AlgebraicElement>,
    pub values: Vec<Real>,
    specializations: Arc<std::sync::Mutex<Vec<(Poly, Specialization)>>>,
}

impl Default for SamplePoint {
    fn default() -> Self {
        SamplePoint::origin()
    }
}

impl SamplePoint {
    /// The point with no coordinates.
    pub fn origin() -> SamplePoint {
        SamplePoint::with(None, None, Vec::new(), Vec::new())
    }

    fn with(
        field: Option<Arc<AlgebraicField>>,
        theta: Option<Real>,
        coords: Vec<AlgebraicElement>,
        values: Vec<Real>,
    ) -> SamplePoint {
        SamplePoint {
            field,
            theta,
            coords,
            values,
            specializations: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// `repr(point)`.
    pub fn repr(&self) -> PolyResult<String> {
        let shown: Vec<String> = self.values.iter().map(|v| v.repr()).collect::<PolyResult<_>>()?;
        Ok(format!("SamplePoint({})", shown.join(", ")))
    }

    /// The degree of the field of the coordinates.
    pub fn degree(&self) -> usize {
        self.theta.as_ref().map(|t| t.degree()).unwrap_or(1)
    }

    /// The point with the coordinate `root` appended.
    pub fn extend(&self, root: &Real) -> PolyResult<SamplePoint> {
        let r: Real = real_checked(root)?;
        let mut values: Vec<Real> = self.values.clone();
        values.push(r.clone());
        if !r.is_rational() {
            if let (Some(theta), Some(field)) = (&self.theta, &self.field) {
                if Arc::ptr_eq(&r, theta) || (r.minpoly() == theta.minpoly() && compare_real(&r, theta)? == 0) {
                    let mut coords: Vec<AlgebraicElement> = self.coords.clone();
                    coords.push(field.alpha());
                    return Ok(SamplePoint::with(
                        self.field.clone(),
                        self.theta.clone(),
                        coords,
                        values,
                    ));
                }
                let (gamma, k, theta_k, beta_k) = join(theta, &r)?;
                let mut coords: Vec<AlgebraicElement> = Vec::new();
                for coord in &self.coords {
                    if coord.is_empty() {
                        coords.push(k.zero());
                    } else {
                        let lifted: Vec<AlgebraicElement> = coord.iter().map(|c| k.from_rational(c)).collect();
                        coords.push(dup_eval(&lifted, &theta_k, &k));
                    }
                }
                coords.push(beta_k);
                return Ok(SamplePoint::with(Some(Arc::new(k)), Some(gamma), coords, values));
            }
            let k: AlgebraicField = field_of(&r)?;
            let mut coords: Vec<AlgebraicElement> = self.coords.clone();
            coords.push(k.alpha());
            return Ok(SamplePoint::with(Some(Arc::new(k)), Some(r), coords, values));
        }
        let q: Rat = r.rational_value()?;
        let mut coords: Vec<AlgebraicElement> = self.coords.clone();
        coords.push(if q.is_zero() { Vec::new() } else { vec![q] });
        Ok(SamplePoint::with(
            self.field.clone(),
            self.theta.clone(),
            coords,
            values,
        ))
    }

    /// The generator `theta` as an element of the field.
    pub fn field_alpha(&self) -> PolyResult<AlgebraicElement> {
        match &self.field {
            None => Err(value_error("the coordinates are rational".to_string())),
            Some(f) => Ok(f.alpha()),
        }
    }

    fn rational_coords(&self) -> Vec<Rat> {
        self.coords
            .iter()
            .map(|c| c.first().cloned().unwrap_or(Rat::ZERO))
            .collect()
    }

    fn check(&self, poly: &Basic, gens: &[Basic], extra: usize) -> PolyResult<Poly> {
        if gens.len() != self.coords.len() + extra {
            return Err(value_error(format!(
                "expected {} generators",
                self.coords.len() + extra
            )));
        }
        Poly::new(poly, gens)
    }

    fn terms(f: &Poly) -> Vec<(Monomial, Rat)> {
        f.rep.terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect()
    }

    /// The exact sign of `poly` in `gens` at the point.
    pub fn sign(&self, poly: &Basic, gens: &[Basic]) -> PolyResult<Sign> {
        if gens.is_empty() {
            if !self.coords.is_empty() {
                return Err(value_error(format!("expected {} generators", self.coords.len())));
            }
            let Some(v) = poly.as_rational() else {
                return Err(value_error(format!("a rational number expected, got {}", poly)));
            };
            return Ok(rat_sign(v));
        }
        let f: Poly = self.check(poly, gens, 0)?;
        match (&self.field, &self.theta) {
            (Some(field), Some(theta)) => {
                let element = evaluate(&Self::terms(&f), &self.coords, field.as_ref(), false, &|c| {
                    field.from_rational(c)
                });
                if element.is_empty() {
                    return Ok(0);
                }
                sign_in_field(theta, &element[0])
            }
            _ => {
                let value = evaluate(&Self::terms(&f), &self.rational_coords(), &QQ, false, &|c| c.clone());
                Ok(value.first().map(rat_sign).unwrap_or(0))
            }
        }
    }

    /// `poly` in the last generator of `gens` at the point: its degree,
    /// leading sign and real roots (`specialization`, kept).
    pub fn specialization(&self, poly: &Basic, gens: &[Basic]) -> PolyResult<Specialization> {
        let f: Poly = self.check(poly, gens, 1)?;
        if let Some((_, found)) = self
            .specializations
            .lock()
            .expect("an unpoisoned cache")
            .iter()
            .find(|(g, _)| *g == f)
        {
            return Ok(found.clone());
        }
        let result: Specialization = match (&self.field, &self.theta) {
            (Some(field), Some(theta)) => {
                let dense = evaluate(&Self::terms(&f), &self.coords, field.as_ref(), true, &|c| {
                    field.from_rational(c)
                });
                if dense.is_empty() {
                    Specialization {
                        degree: -1,
                        leading_sign: 0,
                        roots: Vec::new(),
                    }
                } else {
                    let pairs = if dense.len() > 1 {
                        real_roots_over(theta, field, &dense)?
                    } else {
                        Vec::new()
                    };
                    Specialization {
                        degree: dense.len() as i64 - 1,
                        leading_sign: sign_in_field(theta, &dense[0])?,
                        roots: pairs,
                    }
                }
            }
            _ => {
                let rational = evaluate(&Self::terms(&f), &self.rational_coords(), &QQ, true, &|c| c.clone());
                if rational.is_empty() {
                    Specialization {
                        degree: -1,
                        leading_sign: 0,
                        roots: Vec::new(),
                    }
                } else {
                    let pairs = if rational.len() > 1 {
                        dense_real_roots(&rational)?
                    } else {
                        Vec::new()
                    };
                    Specialization {
                        degree: rational.len() as i64 - 1,
                        leading_sign: rat_sign(&rational[0]),
                        roots: pairs,
                    }
                }
            }
        };
        self.specializations
            .lock()
            .expect("an unpoisoned cache")
            .push((f, result.clone()));
        Ok(result)
    }

    /// The sorted distinct real roots of `poly` in the last generator at
    /// the point; `None` when it vanishes identically there.
    pub fn real_roots(&self, poly: &Basic, gens: &[Basic]) -> PolyResult<Option<Vec<Real>>> {
        let found: Specialization = self.specialization(poly, gens)?;
        if found.degree < 0 {
            return Ok(None);
        }
        Ok(Some(found.roots.into_iter().map(|(r, _)| r).collect()))
    }
}

fn rat_sign(v: &Rat) -> Sign {
    if v.is_zero() {
        0
    } else if rat_is_negative(v) {
        -1
    } else {
        1
    }
}
