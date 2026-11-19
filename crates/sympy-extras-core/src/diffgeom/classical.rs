//! Port of `sympy_extras/diffgeom/classical.py`: the classical
//! differential geometry of parametrised curves (speed, tangent,
//! curvature, torsion, Frenet frame) and surfaces (fundamental forms,
//! normal, Gaussian, mean and principal curvatures), the square roots
//! taken of the squarefree parts so that the results stay exact.

use crate::core::assumptions::Predicate;
use crate::core::basic::Basic;
use crate::core::expr::is;
use crate::error::{value_error, PolyResult};
use crate::matrices::linalg::collect_kernels;
use crate::simplify::normalform::{normal_form, numer_denom};

type R = PolyResult<Basic>;
type Vector = Vec<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sqrt(a: &Basic) -> R {
    crate::matrices::eigen::sqrt(a)
}
fn diff(e: &Basic, v: &Basic) -> R {
    crate::core::function::derivative(e, v)
}

fn dot(a: &[Basic], b: &[Basic]) -> R {
    let terms: Vec<Basic> = a
        .iter()
        .zip(b.iter())
        .map(|(p, q)| m2(p, q))
        .collect::<PolyResult<_>>()?;
    crate::core::add::add(&terms)
}

fn cross(a: &[Basic], b: &[Basic]) -> PolyResult<Vector> {
    Ok(vec![
        s2(&m2(&a[1], &b[2])?, &m2(&a[2], &b[1])?)?,
        s2(&m2(&a[2], &b[0])?, &m2(&a[0], &b[2])?)?,
        s2(&m2(&a[0], &b[1])?, &m2(&a[1], &b[0])?)?,
    ])
}

fn triple(a: &[Basic], b: &[Basic], c: &[Basic]) -> R {
    dot(a, &cross(b, c)?)
}

fn positive_polynomial(f: &Basic, symbols: &[Basic]) -> PolyResult<bool> {
    if is(f, Predicate::nonnegative) == Some(true) {
        return Ok(true);
    }
    let terms = crate::polys::polytools::Poly::new(f, symbols)?.terms()?;
    Ok(terms.iter().all(|(monomial, coefficient)| {
        monomial.iter().all(|k| k % 2 == 0) && is(coefficient, Predicate::positive) == Some(true)
    }))
}

fn polynomial_in_symbols(e: &Basic) -> bool {
    let mut found: Vec<Basic> = Vec::new();
    collect_kernels(e, &mut found);
    found.iter().all(|k| k.is_symbol())
}

fn root_of_factors(e: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut symbols: Vec<Basic> = e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect();
    symbols.sort_by_key(|s| s.to_string());
    if symbols.is_empty() {
        return Ok((int(1), e.clone()));
    }
    let (content, factors) = crate::polys::polytools::factor_list(e, &symbols)?;
    let mut outside: Vec<Basic> = Vec::new();
    let mut inside: Vec<Basic> = vec![content];
    for (factor, exponent) in factors {
        if !exponent.is_integer_value() {
            return Ok((int(1), e.clone()));
        }
        let k: i64 = i64::try_from(&crate::numbers::numer(exponent.as_rational().expect("an integer"))).expect("small");
        if k / 2 != 0 {
            let base: Basic = if positive_polynomial(&factor, &symbols)? {
                factor.clone()
            } else {
                crate::functions::elementary::complexes::abs_(&factor)?
            };
            outside.push(pw(&base, &int(k / 2))?);
        }
        if k % 2 != 0 {
            inside.push(factor);
        }
    }
    Ok((crate::core::mul::mul(&outside)?, crate::core::mul::mul(&inside)?))
}

fn split_sqrt(e: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    let reduced: Basic = normal_form(e)?;
    if reduced == int(0) {
        return Ok((int(0), int(1), int(1)));
    }
    let (numerator, denominator) = numer_denom(&reduced)?;
    if !(polynomial_in_symbols(&numerator) && polynomial_in_symbols(&denominator)) {
        return Ok((int(1), numerator, denominator));
    }
    let (top_out, top_in) = root_of_factors(&numerator)?;
    let (bottom_out, bottom_in) = root_of_factors(&denominator)?;
    Ok((d2(&top_out, &bottom_out)?, top_in, bottom_in))
}

fn root(e: &Basic) -> R {
    let (outside, top, bottom) = split_sqrt(e)?;
    d2(&m2(&outside, &sqrt(&top)?)?, &sqrt(&bottom)?)
}

fn over_sqrt(e: &Basic, radicand: &Basic, power: i64) -> R {
    let (outside, top, bottom) = split_sqrt(radicand)?;
    let half: Basic = int(power / 2);
    let rational: Basic = normal_form(&d2(
        &m2(e, &pw(&bottom, &half)?)?,
        &m2(&pw(&outside, &int(power))?, &pw(&top, &half)?)?,
    )?)?;
    d2(&m2(&rational, &sqrt(&bottom)?)?, &sqrt(&top)?)
}

/// A parametrised curve in the plane or in space (`Curve`).
#[derive(Debug, Clone)]
pub struct Curve {
    pub components: Vec<Basic>,
    pub parameter: Basic,
    pub dimension: usize,
}

impl Curve {
    pub fn new(components: &[Basic], parameter: &Basic) -> PolyResult<Curve> {
        if !matches!(components.len(), 2 | 3) {
            return Err(value_error(
                "a curve in the plane or in space has two or three coordinates".to_string(),
            ));
        }
        Ok(Curve {
            components: components.to_vec(),
            parameter: parameter.clone(),
            dimension: components.len(),
        })
    }

    /// The derivative of the given order (`derivative`).
    pub fn derivative(&self, order: usize) -> PolyResult<Vector> {
        if order == 0 {
            return Ok(self.components.clone());
        }
        self.components
            .iter()
            .map(|c| {
                let mut d: Basic = c.clone();
                for _ in 0..order {
                    d = diff(&d, &self.parameter)?;
                }
                normal_form(&d)
            })
            .collect()
    }

    fn space(&self, order: usize) -> PolyResult<Vector> {
        let mut vector: Vector = self.derivative(order)?;
        while vector.len() < 3 {
            vector.push(int(0));
        }
        Ok(vector)
    }

    pub fn speed(&self) -> R {
        let velocity: Vector = self.derivative(1)?;
        root(&dot(&velocity, &velocity)?)
    }

    pub fn tangent(&self) -> PolyResult<Vector> {
        let velocity: Vector = self.derivative(1)?;
        let speed2: Basic = dot(&velocity, &velocity)?;
        velocity.iter().map(|c| over_sqrt(c, &speed2, 1)).collect()
    }

    pub fn curvature(&self) -> R {
        let (first, second) = (self.space(1)?, self.space(2)?);
        let product: Vector = cross(&first, &second)?;
        root(&d2(&dot(&product, &product)?, &pw(&dot(&first, &first)?, &int(3))?)?)
    }

    pub fn signed_curvature(&self) -> R {
        if self.dimension != 2 {
            return Err(value_error("the signed curvature is that of a plane curve".to_string()));
        }
        let (d1, d2_) = (self.derivative(1)?, self.derivative(2)?);
        let (x1, y1, x2, y2) = (&d1[0], &d1[1], &d2_[0], &d2_[1]);
        over_sqrt(
            &s2(&m2(x1, y2)?, &m2(y1, x2)?)?,
            &a2(&pw(x1, &int(2))?, &pw(y1, &int(2))?)?,
            3,
        )
    }

    pub fn torsion(&self) -> R {
        let (first, second, third) = (self.space(1)?, self.space(2)?, self.space(3)?);
        let product: Vector = cross(&first, &second)?;
        let norm2: Basic = normal_form(&dot(&product, &product)?)?;
        if norm2 == int(0) {
            return Err(value_error(
                "the curvature vanishes: the torsion is not defined".to_string(),
            ));
        }
        normal_form(&d2(&dot(&product, &third)?, &norm2)?)
    }

    #[allow(clippy::type_complexity)]
    pub fn frenet_frame(&self) -> PolyResult<(Vector, Vector, Vector)> {
        let (first, second) = (self.space(1)?, self.space(2)?);
        let product: Vector = cross(&first, &second)?;
        let norm2: Basic = normal_form(&dot(&product, &product)?)?;
        let speed2: Basic = dot(&first, &first)?;
        if norm2 == int(0) {
            return Err(value_error(
                "the curvature vanishes: the Frenet frame is not defined".to_string(),
            ));
        }
        let tangent: Vector = first
            .iter()
            .map(|c| over_sqrt(c, &speed2, 1))
            .collect::<PolyResult<_>>()?;
        let binormal: Vector = product
            .iter()
            .map(|c| over_sqrt(c, &norm2, 1))
            .collect::<PolyResult<_>>()?;
        let both: Basic = m2(&norm2, &speed2)?;
        let normal: Vector = cross(&product, &first)?
            .iter()
            .map(|c| over_sqrt(c, &both, 1))
            .collect::<PolyResult<_>>()?;
        Ok((tangent, normal, binormal))
    }
}

/// A parametrised surface in space (`Surface`).
#[derive(Debug, Clone)]
pub struct Surface {
    pub components: Vec<Basic>,
    pub u: Basic,
    pub v: Basic,
    ru: Vector,
    rv: Vector,
}

impl Surface {
    pub fn new(components: &[Basic], u: &Basic, v: &Basic) -> PolyResult<Surface> {
        if components.len() != 3 {
            return Err(value_error("a surface in space has three coordinates".to_string()));
        }
        if u == v {
            return Err(value_error("the parameters are not distinct".to_string()));
        }
        let ru: Vector = components.iter().map(|c| diff(c, u)).collect::<PolyResult<_>>()?;
        let rv: Vector = components.iter().map(|c| diff(c, v)).collect::<PolyResult<_>>()?;
        Ok(Surface {
            components: components.to_vec(),
            u: u.clone(),
            v: v.clone(),
            ru,
            rv,
        })
    }

    pub fn first_fundamental_form(&self) -> PolyResult<(Basic, Basic, Basic)> {
        Ok((
            normal_form(&dot(&self.ru, &self.ru)?)?,
            normal_form(&dot(&self.ru, &self.rv)?)?,
            normal_form(&dot(&self.rv, &self.rv)?)?,
        ))
    }

    pub fn metric_determinant(&self) -> R {
        let (e, f, g) = self.first_fundamental_form()?;
        normal_form(&s2(&m2(&e, &g)?, &pw(&f, &int(2))?)?)
    }

    pub fn area_element(&self) -> R {
        root(&self.metric_determinant()?)
    }

    fn regular(&self) -> R {
        let w: Basic = self.metric_determinant()?;
        if w == int(0) {
            return Err(value_error("the parametrisation is not regular".to_string()));
        }
        Ok(w)
    }

    pub fn normal(&self) -> PolyResult<Vector> {
        let w: Basic = self.regular()?;
        cross(&self.ru, &self.rv)?.iter().map(|c| over_sqrt(c, &w, 1)).collect()
    }

    fn second_determinants(&self) -> PolyResult<(Basic, Basic, Basic)> {
        let ruu: Vector = self.ru.iter().map(|c| diff(c, &self.u)).collect::<PolyResult<_>>()?;
        let ruv: Vector = self.ru.iter().map(|c| diff(c, &self.v)).collect::<PolyResult<_>>()?;
        let rvv: Vector = self.rv.iter().map(|c| diff(c, &self.v)).collect::<PolyResult<_>>()?;
        Ok((
            normal_form(&triple(&ruu, &self.ru, &self.rv)?)?,
            normal_form(&triple(&ruv, &self.ru, &self.rv)?)?,
            normal_form(&triple(&rvv, &self.ru, &self.rv)?)?,
        ))
    }

    pub fn second_fundamental_form(&self) -> PolyResult<(Basic, Basic, Basic)> {
        let w: Basic = self.regular()?;
        let (l, m, n) = self.second_determinants()?;
        Ok((over_sqrt(&l, &w, 1)?, over_sqrt(&m, &w, 1)?, over_sqrt(&n, &w, 1)?))
    }

    pub fn gaussian_curvature(&self) -> R {
        let w: Basic = self.regular()?;
        let (l, m, n) = self.second_determinants()?;
        normal_form(&d2(&s2(&m2(&l, &n)?, &pw(&m, &int(2))?)?, &pw(&w, &int(2))?)?)
    }

    pub fn mean_curvature(&self) -> R {
        let (e, f, g) = self.first_fundamental_form()?;
        let w: Basic = self.regular()?;
        let (l, m, n) = self.second_determinants()?;
        let numerator: Basic = a2(&s2(&m2(&e, &n)?, &m2(&m2(&int(2), &f)?, &m)?)?, &m2(&g, &l)?)?;
        over_sqrt(&d2(&numerator, &int(2))?, &w, 3)
    }

    pub fn principal_curvatures(&self) -> PolyResult<(Basic, Basic)> {
        let (h, k) = (self.mean_curvature()?, self.gaussian_curvature()?);
        let r: Basic = root(&normal_form(&s2(&pw(&h, &int(2))?, &k)?)?)?;
        Ok((normal_form(&s2(&h, &r)?)?, normal_form(&a2(&h, &r)?)?))
    }
}
