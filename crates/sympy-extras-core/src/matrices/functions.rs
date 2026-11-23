//! Port of `sympy_extras/matrices/functions.py`: functions of a matrix by
//! its spectral components. For every irreducible factor `g` of the
//! characteristic polynomial with multiplicity `m` the projector is a
//! polynomial in `M` over `K[t]/(g)` (the inverse of the cofactor modulo
//! `(x - t)**m`), and the components `(M - t)**j P / j!`; `f(M)` is the
//! sum over the roots of `f^(j)(t)` times the components.

use crate::core::basic::{Basic, Head};
use crate::error::{PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::domainmatrix::DomainMatrix;
use crate::matrices::eigen::{
    characteristic_factors, coefficients, column, extension, fresh_symbol, roots_in_radicals, shifted, sqrt, Field,
    FunctionEntries, GaussianEntries, RationalEntries,
};
use crate::matrices::linalg::{complex_lift, PolynomialLift};
use crate::numbers::Rat;
use crate::polys::densearith::{dup_div, dup_mul, dup_pow, dup_rem};
use crate::polys::domains::domain::Domain;
use crate::polys::euclidtools::dup_invert;

type R = PolyResult<Basic>;

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
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}

/// The components of an irreducible factor, polynomials in its root
/// (`MatrixComponents`).
#[derive(Debug, Clone)]
pub struct MatrixComponents {
    pub factor: Basic,
    pub symbol: Basic,
    pub components: Vec<ExprMatrix>,
}

impl MatrixComponents {
    pub fn values(&self) -> PolyResult<Option<Vec<Basic>>> {
        roots_in_radicals(&self.factor, &self.symbol)
    }
}

fn components_with<F: Field>(
    field: &F,
    m: &ExprMatrix,
    factors: &[(Basic, usize)],
    unknown: &Basic,
    x: &Basic,
) -> PolyResult<Vec<MatrixComponents>> {
    let k: &F::D = field.domain();
    let n: usize = m.rows;
    let mut characteristic = vec![k.one()];
    for (factor, multiplicity) in factors {
        let cs: Vec<_> = coefficients(factor, unknown)?
            .iter()
            .map(|c| field.element(c))
            .collect::<PolyResult<_>>()?;
        let leading = k.invert(&cs[0])?;
        let g: Vec<_> = cs.iter().map(|c| k.mul(c, &leading)).collect();
        characteristic = dup_mul(&characteristic, &dup_pow(&g, *multiplicity, k), k);
    }
    let mut result: Vec<MatrixComponents> = Vec::new();
    for (factor, multiplicity) in factors {
        let f = extension(field, factor, unknown)?;
        let lifted: Vec<_> = characteristic.iter().map(|c| f.from_ground(c)).collect();
        let power = dup_pow(&[f.one(), f.neg(&f.generator())], *multiplicity, &f);
        let cofactor = dup_div(&lifted, &power, &f)?.0;
        let projector_polynomial = if cofactor.len() == 1 {
            vec![f.one()]
        } else {
            let inverse = dup_invert(&dup_rem(&cofactor, &power, &f)?, &power, &f)?;
            dup_rem(&dup_mul(&inverse, &cofactor, &f), &lifted, &f)?
        };
        let mut rows = Vec::new();
        for i in 0..n {
            rows.push(
                m.row(i)
                    .iter()
                    .map(|e| Ok(f.from_ground(&field.element(e)?)))
                    .collect::<PolyResult<Vec<_>>>()?,
            );
        }
        let a = DomainMatrix::new(rows, f.clone(), Some(n))?;
        let identity = DomainMatrix::eye(n, f.clone());
        let mut projector = DomainMatrix::zeros(n, n, f.clone());
        for c in &projector_polynomial {
            projector = projector.mul(&a)?.add(&identity.scale(c))?;
        }
        let sh = shifted(field, &f, m)?;
        let mut components: Vec<ExprMatrix> = Vec::new();
        let mut current = projector;
        for j in 0..*multiplicity {
            if j > 0 {
                current = sh.mul(&current)?.scale(&f.invert(&f.from_int(j as i64))?);
            }
            let mut columns: Vec<ExprMatrix> = Vec::new();
            for kk in 0..n {
                let v: Vec<_> = (0..n).map(|i| current.rows[i][kk].clone()).collect();
                columns.push(column(field, &v, x)?);
            }
            let mut entries: Vec<Basic> = Vec::new();
            for i in 0..n {
                for c in &columns {
                    entries.push(c.at(i, 0).clone());
                }
            }
            components.push(ExprMatrix::new(n, n, entries)?);
        }
        result.push(MatrixComponents {
            factor: crate::core::subs::subs(factor, unknown, x)?,
            symbol: x.clone(),
            components,
        });
    }
    Ok(result)
}

/// The spectral components of `M` (`matrix_components`).
pub fn matrix_components(m: &ExprMatrix, x: &Basic) -> PolyResult<Vec<MatrixComponents>> {
    let unknown: Basic = fresh_symbol(m);
    let (kernels, factors) = characteristic_factors(m, &unknown)?;
    if let Some(g) = complex_lift(&kernels)? {
        return components_with(&GaussianEntries::new(g), m, &factors, &unknown, x);
    }
    if !kernels.is_empty() {
        return components_with(
            &FunctionEntries::new(PolynomialLift::new(&kernels)),
            m,
            &factors,
            &unknown,
            x,
        );
    }
    components_with(&RationalEntries::new(), m, &factors, &unknown, x)
}

fn explicit_values(group: &MatrixComponents) -> PolyResult<Vec<Basic>> {
    match group.values()? {
        Some(v) => Ok(v),
        None => Err(PolyError::NotImplemented(format!(
            "the roots of {} are not expressed in radicals: see matrix_components",
            group.factor
        ))),
    }
}

fn symbol_names(e: &Basic) -> Vec<String> {
    e.free_symbols()
        .iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_)))
        .map(|s| s.to_string())
        .collect()
}

/// The sum over the eigenvalues of `derivative(j, eigenvalue)` times the
/// component matrices; a zero component asks nothing of the function,
/// and a derivative which is needed and not finite is a `ValueError`
/// (`_spectral_sum`).
fn spectral_sum(
    m: &ExprMatrix,
    sym: &Basic,
    derivative: &mut dyn FnMut(usize, &Basic) -> PolyResult<Basic>,
) -> PolyResult<ExprMatrix> {
    let n: usize = m.rows;
    let mut total: Vec<Basic> = vec![int(0); n * n];
    for group in matrix_components(m, sym)? {
        let values: Vec<Basic> = explicit_values(&group)?;
        for (j, component) in group.components.iter().enumerate() {
            if component.entries.iter().all(|e| *e == int(0)) {
                continue;
            }
            for value in &values {
                let scalar: Basic = derivative(j, value)?;
                let bad: bool = scalar.preorder().iter().any(|t| {
                    matches!(
                        t.head(),
                        Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
                    )
                });
                if bad {
                    return Err(PolyError::Value(format!(
                        "the function is not defined on the spectrum of the matrix: its derivative of order {} at the eigenvalue {} is {}",
                        j, value, scalar
                    )));
                }
                let matrix: ExprMatrix = component.subs(&[(sym.clone(), value.clone())])?;
                total = total
                    .iter()
                    .zip(matrix.entries.iter())
                    .map(|(a, b)| a2(a, &m2(&scalar, b)?))
                    .collect::<PolyResult<_>>()?;
            }
        }
    }
    ExprMatrix::new(
        n,
        n,
        total
            .iter()
            .map(crate::core::function::expand_method)
            .collect::<PolyResult<_>>()?,
    )
}

/// `f(M)` for an expression `f` in `x` (`matrix_function`).
pub fn matrix_function(m: &ExprMatrix, function: &Basic, x: &Basic) -> PolyResult<ExprMatrix> {
    let mut sym: Basic = fresh_symbol(m);
    let used: Vec<String> = symbol_names(function);
    while used.contains(&sym.to_string()) {
        sym = symbol(&format!("{}_", sym));
    }
    let mut derivatives: Vec<Basic> = vec![function.clone()];
    let mut derivative = |j: usize, value: &Basic| -> PolyResult<Basic> {
        while derivatives.len() <= j {
            let next: Basic = crate::core::function::derivative(derivatives.last().expect("a derivative"), x)?;
            derivatives.push(next);
        }
        crate::core::subs::subs(&derivatives[j], x, value)
    };
    spectral_sum(m, &sym, &mut derivative)
}

fn real_exponential(group: &MatrixComponents, t: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let cs: Vec<Basic> = coefficients(&group.factor, &group.symbol)?;
    if cs.len() != 3 || !cs.iter().all(|c| c.is_rational_class()) {
        return Ok(None);
    }
    let r = |c: &Basic| c.as_rational().expect("rational").clone();
    let lead: Rat = r(&cs[0]);
    let p: Rat = r(&cs[1]) / lead.clone();
    let q: Rat = r(&cs[2]) / lead;
    let discriminant: Rat = q - p.clone() * p.clone() / Rat::from(crate::numbers::Int::from(4));
    if discriminant <= Rat::from(crate::numbers::Int::from(0)) {
        return Ok(None);
    }
    let alpha: Basic = Basic::from_rational(-p / Rat::from(crate::numbers::Int::from(2)));
    let beta: Basic = sqrt(&Basic::from_rational(discriminant))?;
    let size: usize = group.components[0].entries.len();
    let mut total: Vec<Basic> = vec![int(0); size];
    let bt: Basic = m2(&beta, t)?;
    let (cosine, sine) = (
        crate::functions::elementary::trigonometric::cos(&bt)?,
        crate::functions::elementary::trigonometric::sin(&bt)?,
    );
    for (j, component) in group.components.iter().enumerate() {
        for (index, entry) in component.entries.iter().enumerate() {
            let linear: Vec<Basic> = if entry.has(&group.symbol) {
                coefficients(entry, &group.symbol)?
            } else {
                vec![entry.clone()]
            };
            if linear.len() > 2 {
                return Err(PolyError::Arithmetic(format!(
                    "a component which is not reduced: {}",
                    entry
                )));
            }
            let (c1, c0) = if linear.len() == 2 {
                (linear[0].clone(), linear[1].clone())
            } else {
                (int(0), linear[0].clone())
            };
            let real: Basic = a2(&c0, &m2(&c1, &alpha)?)?;
            let imaginary: Basic = m2(&c1, &beta)?;
            let wave: Basic = s2(&m2(&real, &cosine)?, &m2(&imaginary, &sine)?)?;
            total[index] = a2(&total[index], &m2(&m2(&int(2), &pw(t, &int(j as i64))?)?, &wave)?)?;
        }
    }
    let growth: Basic = crate::functions::elementary::exponential::exp(&m2(&alpha, t)?)?;
    Ok(Some(total.iter().map(|e| m2(&growth, e)).collect::<PolyResult<_>>()?))
}

/// `exp(M*t)` (`matrix_exp`).
pub fn matrix_exp(m: &ExprMatrix, time: &Basic) -> PolyResult<ExprMatrix> {
    let n: usize = m.rows;
    let mut sym: Basic = fresh_symbol(m);
    let used: Vec<String> = symbol_names(time);
    while used.contains(&sym.to_string()) {
        sym = symbol(&format!("{}_", sym));
    }
    let mut total: Vec<Basic> = vec![int(0); n * n];
    let numbers: bool = m.entries.iter().all(|e| e.is_rational_class());
    for group in matrix_components(m, &sym)? {
        let real = if numbers { real_exponential(&group, time)? } else { None };
        if let Some(real) = real {
            total = total
                .iter()
                .zip(real.iter())
                .map(|(a, b)| a2(a, b))
                .collect::<PolyResult<_>>()?;
            continue;
        }
        for value in explicit_values(&group)? {
            let growth: Basic = crate::functions::elementary::exponential::exp(&m2(time, &value)?)?;
            for (j, component) in group.components.iter().enumerate() {
                let scalar: Basic = m2(&pw(time, &int(j as i64))?, &growth)?;
                let matrix: ExprMatrix = component.subs(&[(sym.clone(), value.clone())])?;
                total = total
                    .iter()
                    .zip(matrix.entries.iter())
                    .map(|(a, b)| a2(a, &m2(&scalar, b)?))
                    .collect::<PolyResult<_>>()?;
            }
        }
    }
    ExprMatrix::new(
        n,
        n,
        total
            .iter()
            .map(|e| crate::core::add::add(&[crate::core::function::expand_method(e)?]))
            .collect::<PolyResult<_>>()?,
    )
}

/// `M**n`: the product itself for an integer `n` (of the inverse when it
/// is negative), else a closed form in `n` (`matrix_power`).
pub fn matrix_power(m: &ExprMatrix, exponent: &Basic) -> PolyResult<ExprMatrix> {
    let size: usize = crate::matrices::linalg::square(m)?;
    if let Some(k) = exponent.as_rational().filter(|r| r.is_int()) {
        let mut k: crate::numbers::Int = crate::numbers::numer(k);
        let mut base: ExprMatrix = if k >= crate::numbers::Int::ZERO {
            m.clone()
        } else {
            crate::matrices::linalg::inverse(m)?
        };
        let mut result: ExprMatrix = ExprMatrix::eye(size);
        let normal = crate::matrices::linalg::normal;
        let two: crate::numbers::Int = crate::numbers::Int::from(2);
        k = crate::numbers::abs(&k);
        while k != crate::numbers::Int::ZERO {
            if &k % &two == crate::numbers::Int::from(1) {
                result = result.matmul(&base)?.applyfunc(&normal)?;
            }
            k /= two.clone();
            if k != crate::numbers::Int::ZERO {
                base = base.matmul(&base)?.applyfunc(&normal)?;
            }
        }
        return Ok(result);
    }
    let mut sym: Basic = fresh_symbol(m);
    let used: Vec<String> = symbol_names(exponent);
    while used.contains(&sym.to_string()) {
        sym = symbol(&format!("{}_", sym));
    }
    let mut derivative = |j: usize, value: &Basic| -> PolyResult<Basic> {
        // the derivative of order j of x**n: n (n - 1) ... (n - j + 1) x**(n - j)
        let mut falling: Basic = int(1);
        for i in 0..j {
            falling = m2(&falling, &crate::core::add::sub(exponent, &int(i as i64))?)?;
        }
        m2(&falling, &pw(value, &crate::core::add::sub(exponent, &int(j as i64))?)?)
    };
    spectral_sum(m, &sym, &mut derivative)
}
