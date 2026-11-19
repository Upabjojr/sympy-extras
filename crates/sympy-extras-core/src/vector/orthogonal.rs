//! Port of `sympy_extras/vector/orthogonal.py`: differential operators in
//! orthogonal curvilinear coordinates from their scale factors (gradient,
//! divergence, curl, Laplacian, vector Laplacian, volume element), the
//! scale factors from a transformation to Cartesian coordinates.

use crate::core::basic::Basic;
use crate::error::{value_error, PolyResult};

type R = PolyResult<Basic>;
/// The simplification applied to every result.
pub type Simplifier = fn(&Basic) -> R;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn diff(e: &Basic, v: &Basic) -> R {
    crate::core::function::derivative(e, v)
}

/// The default simplifier, `normal_form`.
pub fn default_simplifier(e: &Basic) -> R {
    crate::simplify::normalform::normal_form(e)
}

/// Orthogonal coordinates with their scale factors
/// (`OrthogonalCoordinates`).
#[derive(Debug, Clone)]
pub struct OrthogonalCoordinates {
    pub coordinates: Vec<Basic>,
    pub scale_factors: Vec<Basic>,
    pub dimension: usize,
    pub simplifier: Simplifier,
}

impl OrthogonalCoordinates {
    pub fn new(
        coordinates: &[Basic],
        scale_factors: &[Basic],
        simplifier: Option<Simplifier>,
    ) -> PolyResult<OrthogonalCoordinates> {
        if coordinates.len() != scale_factors.len() {
            return Err(value_error(format!(
                "{} scale factors for {} coordinates",
                scale_factors.len(),
                coordinates.len()
            )));
        }
        for i in 0..coordinates.len() {
            if coordinates[..i].contains(&coordinates[i]) {
                return Err(value_error("the coordinates are not distinct".to_string()));
            }
        }
        Ok(OrthogonalCoordinates {
            coordinates: coordinates.to_vec(),
            scale_factors: scale_factors.to_vec(),
            dimension: coordinates.len(),
            simplifier: simplifier.unwrap_or(default_simplifier),
        })
    }

    /// The coordinates of `x = f(q)`, checked orthogonal
    /// (`from_transformation`).
    pub fn from_transformation(
        coordinates: &[Basic],
        cartesian_coordinates: &[Basic],
        simplifier: Option<Simplifier>,
    ) -> PolyResult<OrthogonalCoordinates> {
        let tidy: Simplifier = simplifier.unwrap_or(default_simplifier);
        let tangents: Vec<Vec<Basic>> = coordinates
            .iter()
            .map(|q| {
                cartesian_coordinates
                    .iter()
                    .map(|f| diff(f, q))
                    .collect::<PolyResult<_>>()
            })
            .collect::<PolyResult<_>>()?;
        let n: usize = coordinates.len();
        for i in 0..n {
            for j in 0..i {
                let products: Vec<Basic> = tangents[i]
                    .iter()
                    .zip(tangents[j].iter())
                    .map(|(a, b)| m2(a, b))
                    .collect::<PolyResult<_>>()?;
                if tidy(&crate::core::add::add(&products)?)? != int(0) {
                    return Err(value_error(format!(
                        "the coordinates {} and {} are not orthogonal",
                        coordinates[j], coordinates[i]
                    )));
                }
            }
        }
        let mut factors: Vec<Basic> = Vec::new();
        for tangent in &tangents {
            let squares: Vec<Basic> = tangent.iter().map(|a| m2(a, a)).collect::<PolyResult<_>>()?;
            factors.push(crate::matrices::eigen::sqrt(&tidy(&crate::core::add::add(&squares)?)?)?);
        }
        OrthogonalCoordinates::new(coordinates, &factors, simplifier)
    }

    pub fn repr(&self) -> String {
        let show = |v: &[Basic]| v.iter().map(|e| e.to_string()).collect::<Vec<String>>().join(", ");
        format!(
            "OrthogonalCoordinates([{}], [{}])",
            show(&self.coordinates),
            show(&self.scale_factors)
        )
    }

    fn vector(&self, components: &[Basic]) -> PolyResult<Vec<Basic>> {
        if components.len() != self.dimension {
            return Err(value_error(format!(
                "{} components in dimension {}",
                components.len(),
                self.dimension
            )));
        }
        Ok(components.to_vec())
    }

    fn volume(&self) -> R {
        crate::core::mul::mul(&self.scale_factors)
    }

    /// The product of the scale factors (`volume_element`).
    pub fn volume_element(&self) -> R {
        if self.scale_factors.is_empty() {
            return Ok(int(1));
        }
        (self.simplifier)(&self.volume()?)
    }

    pub fn gradient(&self, f: &Basic) -> PolyResult<Vec<Basic>> {
        self.coordinates
            .iter()
            .zip(self.scale_factors.iter())
            .map(|(q, h)| (self.simplifier)(&d2(&diff(f, q)?, h)?))
            .collect()
    }

    pub fn divergence(&self, field: &[Basic]) -> R {
        let f: Vec<Basic> = self.vector(field)?;
        let volume: Basic = self.volume()?;
        let mut terms: Vec<Basic> = Vec::new();
        for ((q, h), component) in self.coordinates.iter().zip(self.scale_factors.iter()).zip(f.iter()) {
            terms.push(diff(&m2(&d2(&volume, h)?, component)?, q)?);
        }
        (self.simplifier)(&d2(&crate::core::add::add(&terms)?, &volume)?)
    }

    pub fn curl(&self, field: &[Basic]) -> PolyResult<Vec<Basic>> {
        if self.dimension != 3 {
            return Err(value_error("the curl is defined in dimension three".to_string()));
        }
        let f: Vec<Basic> = self.vector(field)?;
        let (q, h) = (&self.coordinates, &self.scale_factors);
        let mut result: Vec<Basic> = Vec::new();
        for i in 0..3 {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            let difference: Basic =
                crate::core::add::sub(&diff(&m2(&h[k], &f[k])?, &q[j])?, &diff(&m2(&h[j], &f[j])?, &q[k])?)?;
            result.push((self.simplifier)(&d2(&difference, &m2(&h[j], &h[k])?)?)?);
        }
        Ok(result)
    }

    pub fn laplacian(&self, f: &Basic) -> R {
        let volume: Basic = self.volume()?;
        let mut terms: Vec<Basic> = Vec::new();
        for (q, h) in self.coordinates.iter().zip(self.scale_factors.iter()) {
            let inner: Basic = m2(&d2(&volume, &crate::core::power::pow(h, &int(2))?)?, &diff(f, q)?)?;
            terms.push(diff(&inner, q)?);
        }
        (self.simplifier)(&d2(&crate::core::add::add(&terms)?, &volume)?)
    }

    /// `grad div F - curl curl F` (`vector_laplacian`).
    pub fn vector_laplacian(&self, field: &[Basic]) -> PolyResult<Vec<Basic>> {
        let first: Vec<Basic> = self.gradient(&self.divergence(field)?)?;
        let second: Vec<Basic> = self.curl(&self.curl(field)?)?;
        first
            .iter()
            .zip(second.iter())
            .map(|(a, b)| (self.simplifier)(&crate::core::add::sub(a, b)?))
            .collect()
    }
}

pub fn cartesian(x: &Basic, y: &Basic, z: &Basic) -> PolyResult<OrthogonalCoordinates> {
    OrthogonalCoordinates::new(&[x.clone(), y.clone(), z.clone()], &[int(1), int(1), int(1)], None)
}

pub fn cylindrical(rho: &Basic, phi: &Basic, z: &Basic) -> PolyResult<OrthogonalCoordinates> {
    OrthogonalCoordinates::new(
        &[rho.clone(), phi.clone(), z.clone()],
        &[int(1), rho.clone(), int(1)],
        None,
    )
}

pub fn spherical(r: &Basic, theta: &Basic, phi: &Basic) -> PolyResult<OrthogonalCoordinates> {
    let s: Basic = crate::core::function::rebuild_evaluated(
        &Basic::new(
            crate::core::basic::Head::Function(crate::core::basic::Function::Sin),
            vec![theta.clone()],
        ),
        vec![theta.clone()],
    )?;
    OrthogonalCoordinates::new(
        &[r.clone(), theta.clone(), phi.clone()],
        &[int(1), r.clone(), m2(r, &s)?],
        None,
    )
}
