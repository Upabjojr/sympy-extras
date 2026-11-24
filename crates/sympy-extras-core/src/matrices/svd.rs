//! Port of `sympy_extras/matrices/svd.py`: singular values (the square
//! roots of the eigenvalues of `M^T M`) and the singular value
//! decomposition of a rational matrix, the right singular vectors
//! orthogonalised in the extension `QQ[t]/(factor)` and the real roots of
//! cubics in trigonometric form.

use crate::core::basic::Basic;
use crate::error::{PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::{
    characteristic_factors, column, eigenvalues, extension, field_nullspace, fresh_symbol, shifted, sqrt, Ext, ExtElem,
    Field, RationalEntries,
};
use crate::numbers::{Int, Rat};
use crate::polys::domains::domain::Domain;

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
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn q(n: i64) -> Rat {
    Rat::from(Int::from(n))
}
fn number(r: Rat) -> Basic {
    Basic::from_rational(r)
}

/// `M^* M`, the star being the conjugate transpose (`_gram`).
fn gram(m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let product: ExprMatrix = crate::matrices::linalg::conjugate_transpose(m)?.matmul(m)?;
    if product.has(&Basic::atom(crate::core::basic::Head::ImaginaryUnit)) {
        return product.applyfunc(&crate::matrices::linalg::normal);
    }
    Ok(product)
}

/// The real part of the value, for the order: the roots of a polynomial
/// with real roots are written through complex numbers (`_size`).
fn size(e: &Basic) -> PolyResult<f64> {
    let Some(z) = crate::core::evalf::numeric_value(e, 53) else {
        return Err(PolyError::Type(format!("Cannot convert expression to complex: {}", e)));
    };
    Ok(crate::core::evalf::parts(&z).0.to_f64().value())
}

/// The singular values, decreasing when they are numbers
/// (`singular_values`).
pub fn singular_values(m: &ExprMatrix) -> PolyResult<Vec<Basic>> {
    let mut values: Vec<Basic> = Vec::new();
    // the smaller Gram matrix: min(rows, columns) singular values (A^* A
    // and A A^* have the same nonzero eigenvalues)
    let g: ExprMatrix = if m.cols > m.rows {
        gram(&crate::matrices::linalg::conjugate_transpose(m)?)?
    } else {
        gram(m)?
    };
    if g.entries.iter().all(|e| e.is_rational_class()) {
        // the real forms of the roots (a cubic with three real roots)
        let unknown: Basic = crate::matrices::eigen::fresh_symbol(&g);
        for (factor, multiplicity) in crate::matrices::eigen::characteristic_factors(&g, &unknown)?.1 {
            for root in real_roots(&factor, &unknown)? {
                let value: Basic = sqrt(&root)?;
                for _ in 0..multiplicity {
                    values.push(value.clone());
                }
            }
        }
    } else {
        for (value, multiplicity) in eigenvalues(&g)? {
            let root: Basic = sqrt(&value)?;
            for _ in 0..multiplicity {
                values.push(root.clone());
            }
        }
    }
    if values.iter().all(|v| v.free_symbols().is_empty()) {
        let mut keyed: Vec<(f64, Basic)> = values
            .iter()
            .map(|v| Ok((size(v)?, v.clone())))
            .collect::<PolyResult<_>>()?;
        keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        values = keyed.into_iter().map(|(_, v)| v).collect();
    }
    Ok(values)
}

fn orthogonal<F: Field>(f: &Ext<F>, vectors: Vec<Vec<ExtElem<F>>>) -> PolyResult<Vec<(Vec<ExtElem<F>>, ExtElem<F>)>> {
    let dot = |u: &[ExtElem<F>], v: &[ExtElem<F>]| -> ExtElem<F> {
        let mut total = f.zero();
        for (a, b) in u.iter().zip(v.iter()) {
            if !a.is_empty() && !b.is_empty() {
                total = f.add(&total, &f.mul(a, b));
            }
        }
        total
    };
    let mut done: Vec<(Vec<ExtElem<F>>, ExtElem<F>)> = Vec::new();
    for vector in vectors {
        let mut current = vector;
        for (other, norm) in &done {
            let factor = f.mul(&dot(&current, other), &f.invert(norm)?);
            if !factor.is_empty() {
                current = current
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| f.sub(a, &f.mul(&factor, b)))
                    .collect();
            }
        }
        let norm = dot(&current, &current);
        done.push((current, norm));
    }
    Ok(done)
}

fn real_roots(factor: &Basic, unknown: &Basic) -> PolyResult<Vec<Basic>> {
    let polynomial = crate::polys::polytools::Poly::new(factor, std::slice::from_ref(unknown))?;
    let cs: Vec<Basic> = polynomial.all_coeffs()?;
    let degree: i64 = polynomial.degree(0)?;
    if degree == 3 && cs.iter().all(|c| c.is_rational_class()) {
        let v: Vec<Rat> = cs.iter().map(|c| c.as_rational().expect("rational").clone()).collect();
        let (a, b, c, d) = (v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone());
        if a == q(0) {
            return Err(PolyError::ZeroDivision("Fraction(0, 0)".to_string()));
        }
        let p: Rat = (q(3) * a.clone() * c.clone() - b.clone() * b.clone()) / (q(3) * a.clone() * a.clone());
        let qq: Rat = (q(2) * b.clone() * b.clone() * b.clone() - q(9) * a.clone() * b.clone() * c.clone()
            + q(27) * a.clone() * a.clone() * d)
            / (q(27) * a.clone() * a.clone() * a.clone());
        if p == q(0) {
            return Err(PolyError::ZeroDivision("Fraction division by zero".to_string()));
        }
        let shift: Basic = number(-b / (q(3) * a));
        let radius: Basic = m2(&int(2), &sqrt(&number(-p.clone() / q(3)))?)?;
        let inner: Basic = m2(&number(q(3) * qq / (q(2) * p.clone())), &sqrt(&number(q(-3) / p))?)?;
        let angle: Basic = d2(&crate::functions::elementary::inverse::acos(&inner)?, &int(3))?;
        let two_pi: Basic = m2(&int(2), &Basic::pi())?;
        let mut out: Vec<Basic> = Vec::new();
        for k in 0..3 {
            let turn: Basic = d2(&m2(&two_pi, &int(k))?, &int(3))?;
            out.push(a2(
                &shift,
                &m2(
                    &radius,
                    &crate::functions::elementary::trigonometric::cos(&s2(&angle, &turn)?)?,
                )?,
            )?);
        }
        return Ok(out);
    }
    let found = crate::polys::polyroots::roots(factor, std::slice::from_ref(unknown), true, true, None)?;
    let total: usize = found.iter().map(|(_, k)| *k).sum();
    if total as i64 != degree {
        return Err(PolyError::NotImplemented(format!(
            "the roots of {} are not expressed in radicals",
            factor
        )));
    }
    Ok(found.iter().map(|(r, _)| r.clone()).collect())
}

fn singular_vectors<F: Field>(field: &F, g: &ExprMatrix) -> PolyResult<Vec<(Basic, ExprMatrix)>> {
    let unknown: Basic = fresh_symbol(g);
    let (_, factors) = characteristic_factors(g, &unknown)?;
    let mut pairs: Vec<(Basic, ExprMatrix)> = Vec::new();
    for (factor, _) in &factors {
        if crate::core::subs::subs(factor, &unknown, &int(0))? == int(0) {
            continue;
        }
        let f = extension(field, factor, &unknown)?;
        let basis = orthogonal::<F>(&f, field_nullspace(&shifted(field, &f, g)?)?)?;
        for value in real_roots(factor, &unknown)? {
            for (vector, norm) in &basis {
                let col: ExprMatrix = column(field, vector, &unknown)?.subs(&[(unknown.clone(), value.clone())])?;
                let length: Basic = sqrt(&crate::core::subs::subs(
                    column(field, std::slice::from_ref(norm), &unknown)?.at(0, 0),
                    &unknown,
                    &value,
                )?)?;
                pairs.push((value.clone(), col.div(&length)?));
            }
        }
    }
    Ok(pairs)
}

/// `M = U*S*V^T` for a rational matrix (`svd`).
pub fn svd(m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix, ExprMatrix)> {
    if !m.entries.iter().all(|e| e.is_rational_class()) {
        return Err(PolyError::NotImplemented(
            "the singular value decomposition of a matrix whose entries are not rational numbers".to_string(),
        ));
    }
    let pairs = singular_vectors(&RationalEntries::new(), &gram(m)?)?;
    let mut keyed: Vec<(f64, (Basic, ExprMatrix))> = pairs
        .into_iter()
        .map(|p| Ok((size(&p.0)?, p)))
        .collect::<PolyResult<_>>()?;
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let pairs: Vec<(Basic, ExprMatrix)> = keyed.into_iter().map(|(_, p)| p).collect();
    let r: usize = pairs.len();
    let sigma: Vec<Basic> = pairs.iter().map(|(v, _)| sqrt(v)).collect::<PolyResult<_>>()?;
    let left: Vec<ExprMatrix> = pairs
        .iter()
        .zip(sigma.iter())
        .map(|((_, v), s)| m.matmul(v)?.div(s))
        .collect::<PolyResult<_>>()?;
    let mut u: Vec<Basic> = Vec::new();
    for i in 0..m.rows {
        for l in &left {
            u.push(l.at(i, 0).clone());
        }
    }
    let mut v: Vec<Basic> = Vec::new();
    for i in 0..m.cols {
        for p in &pairs {
            v.push(p.1.at(i, 0).clone());
        }
    }
    let s: Vec<Basic> = (0..r * r)
        .map(|k| if k / r == k % r { sigma[k / r].clone() } else { int(0) })
        .collect();
    Ok((
        ExprMatrix::new(m.rows, r, u)?,
        ExprMatrix::new(r, r, s)?,
        ExprMatrix::new(m.cols, r, v)?,
    ))
}
