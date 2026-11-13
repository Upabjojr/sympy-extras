//! `sympy_extras_rs.polys.domains.algebraicfield.AlgebraicField` and the
//! number-field functions of `factortools` (`dup_ext_norm`,
//! `dup_ext_sqf_norm`, `dup_ext_factor`). The elements are lists of
//! fractions, as in Python. The generator is given as any object: the core
//! expression of the Python module, whose `str` names the root (the
//! conversions through expressions are not available).

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::core::symbol::Symbol;
use sympy_extras_core::numbers::Rat;
use sympy_extras_core::polys::domains::{AlgebraicElement, AlgebraicField, Domain, RationalField};
use sympy_extras_core::polys::factortools as ft;

use crate::convert::{err, out_int, out_poly, tuple, Bridge};

fn rationals(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Rat>> {
    crate::convert::poly(&RationalField, ob)
}

fn out_rationals<'py>(py: Python<'py>, f: &[Rat]) -> PyResult<Bound<'py, PyAny>> {
    out_poly(&RationalField, py, f)
}

/// The elements of a number field as lists of fractions (highest degree
/// first), reduced on the way in.
impl Bridge for AlgebraicField {
    fn extract(&self, ob: &Bound<'_, PyAny>) -> PyResult<AlgebraicElement> {
        Ok(self.reduce(rationals(ob)?))
    }

    fn to_py<'py>(&self, py: Python<'py>, e: &AlgebraicElement) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, e)
    }
}

/// `AlgebraicField(minpoly, alpha_expr)`.
#[pyclass(
    name = "AlgebraicField",
    module = "sympy_extras_rs.polys.domains.algebraicfield",
    frozen
)]
pub struct PyAlgebraicField {
    pub inner: AlgebraicField,
}

impl PyAlgebraicField {
    fn element(&self, ob: &Bound<'_, PyAny>) -> PyResult<AlgebraicElement> {
        rationals(ob)
    }

    fn elements(&self, ob: &Bound<'_, PyAny>) -> PyResult<Vec<AlgebraicElement>> {
        let mut result: Vec<AlgebraicElement> = Vec::new();
        for item in ob.try_iter()? {
            result.push(self.element(&item?)?);
        }
        Ok(result)
    }

    fn out_elements<'py>(&self, py: Python<'py>, f: &[AlgebraicElement]) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Bound<'py, PyAny>> = f.iter().map(|a| out_rationals(py, a)).collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into_any())
    }
}

#[pymethods]
impl PyAlgebraicField {
    #[new]
    fn new(minpoly: &Bound<'_, PyAny>, alpha_expr: &Bound<'_, PyAny>) -> PyResult<PyAlgebraicField> {
        let alpha: Symbol = Symbol::new(&alpha_expr.str()?.to_string(), &[]);
        Ok(PyAlgebraicField {
            inner: AlgebraicField::new(rationals(minpoly)?, alpha.0).map_err(err)?,
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        self.inner.name()
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree
    }

    #[getter]
    fn minpoly<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.minpoly)
    }

    #[getter]
    fn is_field(&self) -> bool {
        true
    }

    #[getter]
    fn zero<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.zero())
    }

    #[getter]
    fn one<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.one())
    }

    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.alpha())
    }

    fn from_int<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.from_integer(&crate::convert::extract_int(n)?))
    }

    fn add<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.add(&self.element(a)?, &self.element(b)?))
    }

    fn sub<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.sub(&self.element(a)?, &self.element(b)?))
    }

    fn mul<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.mul(&self.element(a)?, &self.element(b)?))
    }

    fn pow<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, n: i64) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.pow_signed(&self.element(a)?, n).map_err(err)?)
    }

    fn exquo<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(
            py,
            &self.inner.exquo(&self.element(a)?, &self.element(b)?).map_err(err)?,
        )
    }

    fn invert<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.invert(&self.element(a)?).map_err(err)?)
    }

    fn norm<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        RationalField.to_py(py, &self.inner.norm(&self.element(a)?))
    }

    fn characteristic_polynomial<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.characteristic_polynomial(&self.element(a)?))
    }

    fn minimal_polynomial<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.minimal_polynomial(&self.element(a)?))
    }

    fn factor_dense<'py>(&self, py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        dup_ext_factor(py, f, self)
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn dup_ext_norm<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyAlgebraicField) -> PyResult<Bound<'py, PyAny>> {
    out_rationals(py, &ft::dup_ext_norm(&K.elements(f)?, &K.inner))
}

#[pyfunction]
#[allow(non_snake_case)]
fn dup_ext_sqf_norm<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyAlgebraicField) -> PyResult<Bound<'py, PyAny>> {
    let (s, g, norm) = ft::dup_ext_sqf_norm(&K.elements(f)?, &K.inner);
    tuple(
        py,
        vec![
            out_int(py, &s.into())?,
            K.out_elements(py, &g)?,
            out_rationals(py, &norm)?,
        ],
    )
}

#[pyfunction]
#[allow(non_snake_case)]
fn dup_ext_factor<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyAlgebraicField) -> PyResult<Bound<'py, PyAny>> {
    let (lc, factors) = ft::dup_ext_factor(&K.elements(f)?, &K.inner).map_err(err)?;
    let items: Vec<Bound<'py, PyAny>> = factors
        .iter()
        .map(|(g, k)| tuple(py, vec![K.out_elements(py, g)?, k.into_pyobject(py)?.into_any()]))
        .collect::<PyResult<_>>()?;
    tuple(py, vec![out_rationals(py, &lc)?, PyList::new(py, items)?.into_any()])
}

pub fn register_algebraicfield(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAlgebraicField>()?;
    Ok(())
}

pub fn register_ext_factortools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dup_ext_norm, m)?)?;
    m.add_function(wrap_pyfunction!(dup_ext_sqf_norm, m)?)?;
    m.add_function(wrap_pyfunction!(dup_ext_factor, m)?)?;
    Ok(())
}

// -- algebraicfactor

fn ext_elem(f: &crate::rings::PyPolyElement) -> PyResult<sympy_extras_core::polys::algebraicfactor::ExtPoly> {
    match &f.inner {
        crate::rings::ElemKind::Algebraic(p) => Ok(p.clone()),
        other => {
            let ring = crate::rings::domain_name(&crate::rings::ring_of_kind(other));
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "a polynomial over an algebraic field is expected, got one over {}",
                ring
            )))
        }
    }
}

fn ext_out(f: sympy_extras_core::polys::algebraicfactor::ExtPoly) -> crate::rings::PyPolyElement {
    crate::rings::PyPolyElement {
        inner: crate::rings::ElemKind::Algebraic(f),
    }
}

/// The ring `K[x_1, ..., x_n]` over a number field, as a domain.
#[pyclass(
    name = "AlgebraicPolynomialRing",
    module = "sympy_extras_rs.polys.algebraicfactor",
    frozen
)]
pub struct PyAlgebraicPolynomialRing {
    inner: sympy_extras_core::polys::algebraicfactor::AlgebraicPolynomialRing,
}

#[pymethods]
impl PyAlgebraicPolynomialRing {
    #[new]
    fn new(ground: &PyAlgebraicField, names: Vec<String>) -> PyResult<PyAlgebraicPolynomialRing> {
        Ok(PyAlgebraicPolynomialRing {
            inner: sympy_extras_core::polys::algebraicfactor::AlgebraicPolynomialRing::new(ground.inner.clone(), names)
                .map_err(err)?,
        })
    }

    #[getter]
    fn ring(&self) -> crate::rings::PyPolyRing {
        crate::rings::PyPolyRing {
            inner: crate::rings::RingKind::Algebraic(std::sync::Arc::clone(&self.inner.ring)),
        }
    }

    fn gcd(
        &self,
        a: &crate::rings::PyPolyElement,
        b: &crate::rings::PyPolyElement,
    ) -> PyResult<crate::rings::PyPolyElement> {
        Ok(ext_out(self.inner.gcd(&ext_elem(a)?, &ext_elem(b)?)))
    }
}

#[pyfunction]
fn ext_gcd(f: &crate::rings::PyPolyElement, g: &crate::rings::PyPolyElement) -> PyResult<crate::rings::PyPolyElement> {
    Ok(ext_out(
        sympy_extras_core::polys::algebraicfactor::ext_gcd(&ext_elem(f)?, &ext_elem(g)?).map_err(err)?,
    ))
}

#[pyfunction]
fn ext_factor_list<'py>(py: Python<'py>, f: &crate::rings::PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    let (lc, factors) = sympy_extras_core::polys::algebraicfactor::ext_factor_list(&ext_elem(f)?).map_err(err)?;
    let items: Vec<Bound<'py, PyAny>> = factors
        .into_iter()
        .map(|(g, k)| {
            tuple(
                py,
                vec![
                    Bound::new(py, ext_out(g))?.into_any(),
                    crate::convert::out_usize(py, k)?,
                ],
            )
        })
        .collect::<PyResult<_>>()?;
    tuple(py, vec![out_rationals(py, &lc)?, PyList::new(py, items)?.into_any()])
}

pub fn register_algebraicfactor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAlgebraicPolynomialRing>()?;
    m.add_function(wrap_pyfunction!(ext_gcd, m)?)?;
    m.add_function(wrap_pyfunction!(ext_factor_list, m)?)?;
    Ok(())
}
