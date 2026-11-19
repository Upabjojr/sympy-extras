//! Port of `sympy_extras/series/approximants.py`: Padé approximants and
//! rational interpolation by the extended Euclidean algorithm stopped at
//! a degree (`r = t g mod modulus`, `deg r <= m`), series reversion by
//! Lagrange inversion; in the field of the coefficients of
//! `matrices/eigen.rs`, the leading coefficients certified nonzero.

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::{
    coefficients as coefficients_of, Field, FunctionEntries, GaussianEntries, RationalEntries,
};
use crate::matrices::linalg::{complex_lift, kernels as kernels_of, normal, PolynomialLift};
use crate::polys::densearith::{dup_add, dup_div, dup_exquo, dup_mul, dup_mul_ground, dup_sub};
use crate::polys::densetools::dup_eval;
use crate::polys::domains::domain::Domain;
use crate::polys::euclidtools::dup_gcd;

type R = PolyResult<Basic>;
type Pair = (Vec<Basic>, Vec<Basic>);

fn int(n: i64) -> Basic {
    Basic::int(n)
}

struct Worker<F: Field> {
    field: F,
    exact: bool,
}

type Elem<F> = <<F as Field>::D as Domain>::Elem;

impl<F: Field> Worker<F> {
    fn nonzero(&self, f: Vec<Elem<F>>) -> PolyResult<Vec<Elem<F>>> {
        if !f.is_empty() && !self.exact {
            let e: Basic = self.field.expr(&f[0])?;
            let nonzero_rational: bool = e.is_rational_class() && e != int(0);
            if !nonzero_rational && crate::simplify::structure::is_zero(&e, &[])? != Some(false) {
                return Err(PolyError::NotImplemented(format!(
                    "the coefficient {} is not proved nonzero: simplify the coefficients first",
                    e
                )));
            }
        }
        Ok(f)
    }

    fn dense(&self, ascending: &[Basic]) -> PolyResult<Vec<Elem<F>>> {
        let k: &F::D = self.field.domain();
        let mut f: Vec<Elem<F>> = ascending
            .iter()
            .rev()
            .map(|c| self.field.element(c))
            .collect::<PolyResult<_>>()?;
        while !f.is_empty() && k.is_zero(&f[0]) {
            f.remove(0);
        }
        Ok(f)
    }

    fn ascending(&self, f: &[Elem<F>]) -> PolyResult<Vec<Basic>> {
        f.iter().rev().map(|c| self.field.expr(c)).collect()
    }

    #[allow(clippy::type_complexity)]
    fn euclid(
        &self,
        modulus: Vec<Elem<F>>,
        g: Vec<Elem<F>>,
        bound: usize,
    ) -> PolyResult<(Vec<Elem<F>>, Vec<Elem<F>>, Vec<Elem<F>>)> {
        let k: &F::D = self.field.domain();
        let (mut r0, mut r1) = (modulus, g);
        let mut t0: Vec<Elem<F>> = Vec::new();
        let mut t1: Vec<Elem<F>> = vec![k.one()];
        while r1.len() as i64 - 1 > bound as i64 {
            let (quotient, remainder) = dup_div(&r0, &self.nonzero(r1.clone())?, k)?;
            r0 = std::mem::replace(&mut r1, remainder);
            let next: Vec<Elem<F>> = dup_sub(&t0, &dup_mul(&quotient, &t1, k), k);
            t0 = std::mem::replace(&mut t1, next);
        }
        let cofactor: Vec<Elem<F>> = t1.clone();
        if !r1.is_empty() {
            let common: Vec<Elem<F>> = dup_gcd(&self.nonzero(r1.clone())?, &t1, k);
            if common.len() > 1 {
                r1 = dup_exquo(&r1, &common, k)?;
                t1 = dup_exquo(&t1, &common, k)?;
            }
        } else {
            t1 = vec![k.one()];
        }
        Ok((r1, t1, cofactor))
    }

    fn normalised(&self, mut r: Vec<Elem<F>>, mut t: Vec<Elem<F>>) -> PolyResult<Pair> {
        let k: &F::D = self.field.domain();
        let last = t[t.len() - 1].clone();
        let scale = if !k.is_zero(&last) { last } else { t[0].clone() };
        if !k.is_one(&scale) {
            let inverse = k.invert(&scale)?;
            r = dup_mul_ground(&r, &inverse, k);
            t = dup_mul_ground(&t, &inverse, k);
        }
        Ok((self.ascending(&r)?, self.ascending(&t)?))
    }

    fn pade(&self, coefficients: &[Basic], m: usize, n: usize, strict: bool) -> PolyResult<Pair> {
        let k: &F::D = self.field.domain();
        let total: usize = m + n + 1;
        let mut modulus: Vec<Elem<F>> = vec![k.one()];
        modulus.extend(std::iter::repeat(k.zero()).take(total));
        let (r, t, cofactor) = self.euclid(modulus, self.dense(&coefficients[..total.min(coefficients.len())])?, m)?;
        if strict && k.is_zero(&cofactor[cofactor.len() - 1]) {
            return Err(value_error(format!(
                "the [{}/{}] Pade approximant does not exist",
                m, n
            )));
        }
        self.normalised(r, t)
    }

    fn interpolate(&self, nodes: &[Basic], values: &[Basic], degree: usize) -> PolyResult<Option<Pair>> {
        let k: &F::D = self.field.domain();
        let points: Vec<Elem<F>> = nodes.iter().map(|e| self.field.element(e)).collect::<PolyResult<_>>()?;
        let ordinates: Vec<Elem<F>> = values
            .iter()
            .map(|e| self.field.element(e))
            .collect::<PolyResult<_>>()?;
        for i in 0..points.len() {
            if points[..i].contains(&points[i]) {
                return Err(value_error("the points of interpolation are not distinct".to_string()));
            }
        }
        let mut modulus: Vec<Elem<F>> = vec![k.one()];
        let mut g: Vec<Elem<F>> = Vec::new();
        for (point, value) in points.iter().zip(ordinates.iter()) {
            let difference = k.sub(value, &dup_eval(&g, point, k));
            if !k.is_zero(&difference) {
                let weight = k.exquo(&difference, &dup_eval(&modulus, point, k))?;
                g = dup_add(&g, &dup_mul_ground(&modulus, &weight, k), k);
            }
            modulus = dup_mul(&modulus, &[k.one(), k.neg(point)], k);
        }
        let (r, t, _) = self.euclid(modulus, g, degree)?;
        for (point, value) in points.iter().zip(ordinates.iter()) {
            let denominator = dup_eval(&t, point, k);
            if k.is_zero(&denominator) || k.mul(&denominator, value) != dup_eval(&r, point, k) {
                return Ok(None);
            }
        }
        Ok(Some(self.normalised(r, t)?))
    }

    fn revert(&self, coefficients: &[Basic]) -> PolyResult<Vec<Basic>> {
        let k: &F::D = self.field.domain();
        let n: usize = coefficients.len();
        let series: Vec<Elem<F>> = coefficients
            .iter()
            .map(|c| self.field.element(c))
            .collect::<PolyResult<_>>()?;
        if n < 2 || k.is_zero(&series[1]) {
            return Err(value_error(
                "the series has no inverse: the coefficient of the first power is zero".to_string(),
            ));
        }
        self.nonzero(vec![series[1].clone()])?;
        let product = |f: &[Elem<F>], g: &[Elem<F>]| -> Vec<Elem<F>> {
            let mut out: Vec<Elem<F>> = vec![k.zero(); n - 1];
            for (i, a) in f.iter().enumerate() {
                if k.is_zero(a) {
                    continue;
                }
                for j in 0..(n - 1).saturating_sub(i) {
                    if !k.is_zero(&g[j]) {
                        out[i + j] = k.add(&out[i + j], &k.mul(a, &g[j]));
                    }
                }
            }
            out
        };
        let inverse = k.invert(&series[1])?;
        let mut phi: Vec<Elem<F>> = Vec::new();
        for kk in 0..n - 1 {
            let mut total = if kk == 0 { k.one() } else { k.zero() };
            for j in 1..=kk {
                total = k.sub(&total, &k.mul(&series[j + 1], &phi[kk - j]));
            }
            phi.push(k.mul(&total, &inverse));
        }
        let mut result: Vec<Basic> = vec![int(0)];
        let mut power: Vec<Elem<F>> = phi.clone();
        for kk in 1..n {
            result.push(self.field.expr(&k.exquo(&power[kk - 1], &k.from_int(kk as i64))?)?);
            if kk + 1 < n {
                power = product(&power, &phi);
            }
        }
        Ok(result)
    }
}

/// The operations of the approximation over the field of the coefficients.
enum Approximation {
    Gaussian(Worker<GaussianEntries>),
    Function(Worker<FunctionEntries>),
    Rational(Worker<RationalEntries>),
}

macro_rules! each {
    ($a:expr, |$w:ident| $body:expr) => {
        match $a {
            Approximation::Gaussian($w) => $body,
            Approximation::Function($w) => $body,
            Approximation::Rational($w) => $body,
        }
    };
}

fn approximation(coefficients: &[Basic]) -> PolyResult<Approximation> {
    let kernels: Vec<Basic> = kernels_of(&[&ExprMatrix::new(1, coefficients.len(), coefficients.to_vec())?]);
    if let Some(g) = complex_lift(&kernels)? {
        return Ok(Approximation::Gaussian(Worker {
            field: GaussianEntries::new(g),
            exact: true,
        }));
    }
    if !kernels.is_empty() {
        let lift: PolynomialLift = PolynomialLift::new(&kernels);
        let exact: bool = lift.exact;
        return Ok(Approximation::Function(Worker {
            field: FunctionEntries::new(lift),
            exact,
        }));
    }
    Ok(Approximation::Rational(Worker {
        field: RationalEntries::new(),
        exact: true,
    }))
}

fn polynomial(ascending: &[Basic], h: &Basic) -> R {
    let terms: Vec<Basic> = ascending
        .iter()
        .enumerate()
        .map(|(k, c)| crate::core::mul::mul(&[c.clone(), crate::core::power::pow(h, &int(k as i64))?]))
        .collect::<PolyResult<_>>()?;
    crate::core::add::add(&terms)
}

fn quotient(numerator: &[Basic], denominator: &[Basic], h: &Basic) -> R {
    normal(&crate::core::mul::div(
        &polynomial(numerator, h)?,
        &polynomial(denominator, h)?,
    )?)
}

/// The first `n` Taylor coefficients of `f` at `x0` (`taylor_coefficients`).
pub fn taylor_coefficients(f: &Basic, x: &Basic, n: i64, x0: &Basic) -> PolyResult<Vec<Basic>> {
    if n <= 0 {
        return Ok(Vec::new());
    }
    let shifted: Basic = if *x0 != int(0) {
        crate::core::subs::subs(f, x, &crate::core::add::add(&[x.clone(), x0.clone()])?)?
    } else {
        f.clone()
    };
    let expansion: Basic = crate::core::series::series(&shifted, Some(x), &int(0), n, "+", None, &int(0))?;
    let truncated: Basic = crate::core::function::expand_default(&crate::core::order::remove_o(&expansion)?)?;
    let cs: Vec<Basic> = if truncated == int(0) {
        vec![int(0)]
    } else {
        coefficients_of(&truncated, x)?
    };
    if cs.iter().any(|c| c.has(x)) {
        return Err(value_error(format!(
            "the expansion of {} at {} is not a power series",
            f, x0
        )));
    }
    let mut ascending: Vec<Basic> = cs.into_iter().rev().take(n as usize).collect();
    while (ascending.len() as i64) < n {
        ascending.push(int(0));
    }
    Ok(ascending)
}

/// The `[m/n]` Padé approximant from the Taylor coefficients
/// (`pade_from_coefficients`).
pub fn pade_from_coefficients(coefficients: &[Basic], x: &Basic, m: i64, n: i64, strict: bool) -> R {
    if m < 0 || n < 0 {
        return Err(value_error(
            "the degrees of a Pade approximant are not negative".to_string(),
        ));
    }
    if (coefficients.len() as i64) < m + n + 1 {
        return Err(value_error(format!(
            "{} coefficients are needed for the [{}/{}] approximant",
            m + n + 1,
            m,
            n
        )));
    }
    let (numerator, denominator) = each!(approximation(coefficients)?, |w| w.pade(
        coefficients,
        m as usize,
        n as usize,
        strict
    ))?;
    quotient(&numerator, &denominator, x)
}

/// The `[m/n]` Padé approximant of `f` at `x0` (`pade`).
pub fn pade(f: &Basic, x: &Basic, m: i64, n: i64, x0: &Basic, strict: bool) -> R {
    if m < 0 || n < 0 {
        return Err(value_error(
            "the degrees of a Pade approximant are not negative".to_string(),
        ));
    }
    let coefficients: Vec<Basic> = taylor_coefficients(f, x, m + n + 1, x0)?;
    let (numerator, denominator) = each!(approximation(&coefficients)?, |w| w.pade(
        &coefficients,
        m as usize,
        n as usize,
        strict
    ))?;
    quotient(&numerator, &denominator, &crate::core::add::sub(x, x0)?)
}

/// The rational function of numerator degree `degree` through the points
/// (`rational_interpolate`); `None` when there is none.
pub fn rational_interpolate(points: &[(Basic, Basic)], x: &Basic, degree: i64) -> PolyResult<Option<Basic>> {
    let count: i64 = points.len() as i64;
    if !(0 <= degree && degree < count.max(1)) {
        return Err(value_error(format!(
            "the degree of the numerator is between 0 and {}",
            count - 1
        )));
    }
    let nodes: Vec<Basic> = points.iter().map(|(p, _)| p.clone()).collect();
    let values: Vec<Basic> = points.iter().map(|(_, v)| v.clone()).collect();
    let mut all: Vec<Basic> = nodes.clone();
    all.extend(values.iter().cloned());
    let found = each!(approximation(&all)?, |w| w.interpolate(
        &nodes,
        &values,
        degree as usize
    ))?;
    match found {
        None => Ok(None),
        Some((p, q)) => Ok(Some(quotient(&p, &q, x)?)),
    }
}

/// The coefficients of the inverse series (`revert_coefficients`).
pub fn revert_coefficients(coefficients: &[Basic]) -> PolyResult<Vec<Basic>> {
    each!(approximation(coefficients)?, |w| w.revert(coefficients))
}

/// The inverse series of `f` at `x0` in `y` (`series_reversion`).
pub fn series_reversion(f: &Basic, x: &Basic, y: &Basic, n: i64, x0: &Basic) -> R {
    let coefficients: Vec<Basic> = taylor_coefficients(f, x, n, x0)?;
    if n < 2 {
        return Ok(x0.clone());
    }
    let reverted: Vec<Basic> = each!(approximation(&coefficients)?, |w| w.revert(&coefficients))?;
    crate::core::add::add(&[
        x0.clone(),
        polynomial(&reverted, &crate::core::add::sub(y, &coefficients[0])?)?,
    ])
}
