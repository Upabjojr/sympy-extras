//! Port of `sympy_extras/solvers/irregular.py`: the exponential parts of
//! the hyperexponential solutions at the irregular finite singular points
//! of a linear differential operator, from the local Newton polygons.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeMap;

use crate::core::basic::Basic;
use crate::error::PolyResult;
use crate::numbers::Rat;
use crate::polys::domains::algebraicfield::{AlgebraicElement, AlgebraicField};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::rationalfield::QQ;
use crate::solvers::kovacic::{failure, replaced};
use crate::solvers::linear_ode::{normal, order_at, places, root_multiplicities, LinearOperator, UPoly};

type R = PolyResult<Basic>;

/// How many terms of an exponential part are followed at one point
/// (`MAX_DEPTH`).
const MAX_DEPTH: usize = 6;

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
fn sum(terms: &[Basic]) -> R {
    if terms.is_empty() {
        Ok(Basic::zero())
    } else {
        add(terms)
    }
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}

/// A finite singular point: the irreducible monic `q`, one root `alpha`
/// (a rational number, or a symbol standing for a root of `q`) and the
/// field `K(alpha)` (`Place`).
#[derive(Clone, Debug)]
pub struct Place {
    pub q: UPoly,
    pub x: Basic,
    pub degree: i64,
    pub alpha: Basic,
}

impl Place {
    pub fn new(q: &UPoly, x: &Basic) -> PolyResult<Place> {
        let monic: UPoly = q.monic()?;
        let degree: i64 = q.degree();
        let alpha: Basic = if degree == 1 {
            neg(&monic.coeff(0))?
        } else {
            dummy("alpha")
        };
        Ok(Place {
            q: monic,
            x: x.clone(),
            degree,
            alpha,
        })
    }

    /// `p(alpha)` as an expression in `alpha` of degree less than that of
    /// `q` (`value`).
    pub fn value(&self, p: &UPoly) -> R {
        if self.degree == 1 {
            return normal(&replaced(&p.as_expr()?, &[(self.x.clone(), self.alpha.clone())])?);
        }
        let reduced: UPoly = p.rem(&self.q)?;
        replaced(&reduced.as_expr()?, &[(self.x.clone(), self.alpha.clone())])
    }

    /// `K(alpha)` as a number field, `None` when `q` does not have rational
    /// coefficients (`field`).
    pub fn field(&self) -> PolyResult<Option<AlgebraicField>> {
        let mut minpoly: Vec<Rat> = Vec::new();
        for c in self.q.all_coeffs() {
            match QQ.from_expr(&c) {
                Ok(v) => minpoly.push(v),
                Err(e) if failure(&e) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
        match AlgebraicField::new(minpoly, self.alpha.clone()) {
            Ok(field) => Ok(Some(field)),
            Err(e) if failure(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// The coefficient of `(x - alpha)**order` in `q**order p` at `alpha`
/// (`_leading_value`).
fn leading_value(place: &Place, p: &UPoly, order: i64) -> R {
    let derivative: UPoly = place.q.diff()?;
    let mut value: Basic = place.value(p)?;
    if order != 0 {
        value = mul(&[value, pow(&place.value(&derivative)?, &int(order))?])?;
    }
    if place.degree == 1 {
        return Ok(value);
    }
    let in_x: Basic = replaced(&value, &[(place.alpha.clone(), place.x.clone())])?;
    place.value(&UPoly::of(&in_x, &place.x)?)
}

/// The lower convex hull of points with increasing abscissae
/// (`_lower_hull`).
fn lower_hull(points: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut hull: Vec<(i64, i64)> = Vec::new();
    for &point in points {
        while hull.len() >= 2 {
            let ((i1, v1), (i2, v2)) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            let (i3, v3) = point;
            // drop the middle point when it lies above the chord
            if (v2 - v1) * (i3 - i1) >= (v3 - v1) * (i2 - i1) {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(point);
    }
    hull
}

/// The roots of the characteristic polynomial: all of them at a rational
/// point, those in `K(alpha)` otherwise; and whether no root was missed
/// (`_roots_in_field`).
fn roots_in_field(characteristic: &UPoly, place: &Place) -> PolyResult<(Vec<Basic>, bool)> {
    if place.degree == 1 {
        let found: Vec<(Basic, usize)> = root_multiplicities(characteristic)?;
        let complete: bool = found.iter().map(|(_, m)| *m as i64).sum::<i64>() == characteristic.degree();
        return Ok((
            found
                .into_iter()
                .map(|(r, _)| r)
                .filter(|r| *r != Basic::zero())
                .collect(),
            complete,
        ));
    }
    let Some(field) = place.field()? else {
        return Ok((Vec::new(), false));
    };
    let mut dense: Vec<AlgebraicElement> = Vec::new();
    for c in characteristic.all_coeffs() {
        match expand(&c).and_then(|e| field.from_expr(&e)) {
            Ok(v) => dense.push(v),
            Err(e) if failure(&e) => return Ok((Vec::new(), false)),
            Err(e) => return Err(e),
        }
    }
    let (_, factors) = field.factor_dense(&dense)?;
    let mut result: Vec<Basic> = Vec::new();
    let mut complete: bool = true;
    for (f, _) in factors {
        if f.len() != 2 {
            complete = false;
            continue;
        }
        let r: Basic = field.to_expr(&field.neg(&field.exquo(&f[1], &f[0])?))?;
        if r != Basic::zero() {
            result.push(r);
        }
    }
    Ok((result, complete))
}

/// `sum_sigma sigma(r)/(x - sigma alpha)**m` over the conjugates of
/// `alpha`, as a rational function in `K(x)` (`conjugate_sum`).
pub fn conjugate_sum(place: &Place, r: &Basic, m: i64) -> R {
    let x: &Basic = &place.x;
    if place.degree == 1 {
        return div(r, &pow(&sub(x, &place.alpha)?, &int(m))?);
    }
    let c: Basic = dummy("c");
    let q_c: UPoly = UPoly::of(&replaced(&place.q.as_expr()?, &[(x.clone(), c.clone())])?, &c)?;
    // q(x)/(x - c) as a polynomial in x with coefficients in K[c]
    let coefficients: Vec<Basic> = place.q.all_coeffs();
    let mut cofactor: Vec<Basic> = Vec::new();
    let mut carry: Basic = Basic::zero();
    for a in &coefficients[..coefficients.len() - 1] {
        carry = add(&[mul(&[carry, c.clone()])?, a.clone()])?;
        cofactor.push(carry.clone());
    }
    let mut cofactor_terms: Vec<Basic> = Vec::new();
    for (i, a) in cofactor.iter().enumerate() {
        cofactor_terms.push(mul(&[a.clone(), pow(x, &int((cofactor.len() - 1 - i) as i64))?])?);
    }
    let cofactor_expr: Basic = sum(&cofactor_terms)?;
    let r_c: Basic = replaced(r, &[(place.alpha.clone(), c.clone())])?;
    let numerator: UPoly = UPoly::of(&expand(&pow(&cofactor_expr, &int(m))?)?, x)?;
    // the power sums of the roots of q, for the traces of the powers of c
    let power_sums: Vec<Basic> = power_sums(&place.q, (place.degree - 1) as usize)?;
    let mut traced: BTreeMap<i64, Basic> = BTreeMap::new();
    for (k, a) in &numerator.terms {
        let term: UPoly = UPoly::of(&expand(&mul(&[a.clone(), r_c.clone()])?)?, &c)?.rem(&q_c)?;
        let mut parts: Vec<Basic> = Vec::new();
        for (j, value) in &term.terms {
            parts.push(mul(&[value.clone(), power_sums[*j as usize].clone()])?);
        }
        traced.insert(*k, sum(&parts)?);
    }
    div(&UPoly::new(traced, x).as_expr()?, &pow(&place.q.as_expr()?, &int(m))?)
}

/// `p_0, ..., p_count`, the power sums of the roots of the monic `q`
/// (Newton's identities) (`_power_sums`).
fn power_sums(q: &UPoly, count: usize) -> PolyResult<Vec<Basic>> {
    let coefficients: Vec<Basic> = q.all_coeffs();
    let n: usize = q.degree() as usize;
    let mut sums: Vec<Basic> = vec![int(n as i64)];
    for k in 1..=count {
        let mut total: Basic = Basic::zero();
        for i in 1..=k.min(n) {
            total = sub(&total, &mul(&[coefficients[i].clone(), sums[k - i].clone()])?)?;
        }
        if k <= n {
            total = sub(&total, &mul(&[int(k as i64), coefficients[k].clone()])?)?;
        }
        sums.push(total);
    }
    Ok(sums)
}

/// A candidate leading term of an exponential part, by its integral `F`
/// (`_Term`).
struct Term {
    integral: Basic,
    m: i64,
}

/// The candidate leading terms of the exponential part at the place, and
/// whether every candidate was found (`_leading_terms`).
fn leading_terms(l: &LinearOperator, place: &Place, below: Option<i64>) -> PolyResult<(Vec<Term>, bool)> {
    let x: &Basic = &l.x;
    let mut data: Vec<(i64, i64, UPoly)> = Vec::new();
    for (i, c) in l.coefficients.iter().enumerate() {
        if *c != Basic::zero() {
            let (order, cofactor) = order_at(&UPoly::of(c, x)?, &place.q)?;
            data.push((i as i64, order, cofactor));
        }
    }
    if data.len() < 2 {
        return Ok((Vec::new(), true));
    }
    let points: Vec<(i64, i64)> = data.iter().map(|(i, order, _)| (*i, *order)).collect();
    let hull: Vec<(i64, i64)> = lower_hull(&points);
    let mut candidates: Vec<Term> = Vec::new();
    let mut complete: bool = true;
    let t: Basic = dummy("t");
    for pair in hull.windows(2) {
        let ((i1, v1), (i2, v2)) = (pair[0], pair[1]);
        // v_i - m i constant on the edge: m = (v2 - v1)/(i2 - i1)
        let slope: Rat = Rat::from(v2 - v1) / Rat::from(i2 - i1);
        if slope < Rat::from(2) || !slope.is_int() {
            continue;
        }
        let m: i64 = i64::try_from(&crate::numbers::numer(&slope)).unwrap_or(0);
        if below.is_some_and(|b| m >= b) {
            continue;
        }
        let mut terms: Vec<Basic> = Vec::new();
        for (i, order, cofactor) in &data {
            if Rat::from(order - v1) == slope.clone() * Rat::from(i - i1) {
                terms.push(mul(&[leading_value(place, cofactor, *order)?, pow(&t, &int(*i))?])?);
            }
        }
        let characteristic: UPoly = UPoly::of(&sum(&terms)?, &t)?;
        let (found, found_complete) = roots_in_field(&characteristic, place)?;
        complete = complete && found_complete;
        for r in found {
            // the integral of r/(x - alpha)**m is -r/((m - 1) (x - alpha)**(m - 1))
            let integral: Basic = conjugate_sum(place, &div(&neg(&r)?, &int(m - 1))?, m - 1)?;
            candidates.push(Term {
                integral: normal(&integral)?,
                m,
            });
        }
    }
    Ok((candidates, complete))
}

/// The candidate integrals `F` of the exponential parts at the place, and
/// whether they are all the possible parts (`local_exponential_parts`).
pub fn local_exponential_parts(
    l: &LinearOperator,
    place: &Place,
    below: Option<i64>,
    depth: usize,
) -> PolyResult<(Vec<Basic>, bool)> {
    let mut parts: Vec<Basic> = vec![Basic::zero()];
    let (leading, mut complete) = leading_terms(l, place, below)?;
    if depth >= MAX_DEPTH {
        return Ok((parts, complete && leading.is_empty()));
    }
    for term in leading {
        let m_operator: LinearOperator =
            l.transformed(&crate::functions::elementary::exponential::exp(&term.integral)?)?;
        let (lower, lower_complete) = local_exponential_parts(&m_operator, place, Some(term.m), depth + 1)?;
        complete = complete && lower_complete;
        for part in lower {
            let candidate: Basic = add(&[term.integral.clone(), part])?;
            if !parts.contains(&candidate) {
                parts.push(candidate);
            }
        }
    }
    Ok((parts, complete))
}

/// The irreducible factors of the leading coefficient at whose roots the
/// operator is not regular singular, monic (`irregular_places`).
pub fn irregular_places(l: &LinearOperator) -> PolyResult<Vec<UPoly>> {
    let x: &Basic = &l.x;
    let n: usize = l.order();
    let mut result: Vec<UPoly> = Vec::new();
    for (q, multiplicity) in places(l.leading_coefficient(), x)? {
        for (i, c) in l.coefficients[..n].iter().enumerate() {
            if *c == Basic::zero() {
                continue;
            }
            let (order, _) = order_at(&UPoly::of(c, x)?, &q)?;
            if order < multiplicity - (n - i) as i64 {
                result.push(q.monic()?);
                break;
            }
        }
    }
    Ok(result)
}

/// The candidate integrals `F` of the exponential parts at all the
/// irregular finite singular points, and whether the list is complete
/// (`finite_exponential_parts`).
pub fn finite_exponential_parts(l: &LinearOperator, max_combinations: usize) -> PolyResult<(Vec<Basic>, bool)> {
    let found: Vec<Place> = irregular_places(l)?
        .iter()
        .map(|q| Place::new(q, &l.x))
        .collect::<PolyResult<_>>()?;
    let mut complete: bool = true;
    let mut local: Vec<Vec<Basic>> = Vec::new();
    for place in &found {
        let (parts, parts_complete) = local_exponential_parts(l, place, None, 0)?;
        complete = complete && parts_complete;
        local.push(parts);
    }
    let mut combinations: Vec<Vec<Basic>> = vec![Vec::new()];
    for parts in &local {
        let mut next: Vec<Vec<Basic>> = Vec::new();
        for c in &combinations {
            for p in parts {
                let mut n: Vec<Basic> = c.clone();
                n.push(p.clone());
                next.push(n);
            }
        }
        combinations = next;
    }
    if combinations.len() > max_combinations {
        combinations.truncate(max_combinations);
        complete = false;
    }
    let mut result: Vec<Basic> = Vec::new();
    for combination in combinations {
        let mut total: Basic = Basic::zero();
        for part in combination {
            total = add(&[total, part])?;
        }
        if !result.contains(&total) {
            result.push(total);
        }
    }
    Ok((result, complete))
}
