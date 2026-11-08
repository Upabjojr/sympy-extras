//! `sympy_extras_rs.polys.domains.sparsepolynomialring`: the polynomial
//! rings as domains, with the methods of the Python classes. One Python
//! class holds any of them; `IntegerPolynomialRing`,
//! `RationalPolynomialRing` and `IteratedPolynomialRing` are constructors.

#![allow(non_snake_case)]

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sympy_extras_core::polys::domains::{
    Domain, FiniteField, IntegerRing, IteratedPolynomialRing, RationalField, SparsePolynomialRing,
};
use sympy_extras_core::polys::rings::PolyElement;

use crate::convert::{err, tuple, Bridge};
use crate::domains::{DomainKind, PyDomain};
use crate::rings::{ElemKind, PyPolyElement, PyPolyRing, RingKind};

#[derive(Debug, Clone)]
pub enum AnyRing {
    SZ(SparsePolynomialRing<IntegerRing>),
    SQ(SparsePolynomialRing<RationalField>),
    SG(SparsePolynomialRing<FiniteField>),
    IZ(IteratedPolynomialRing<IntegerRing>),
    IQ(IteratedPolynomialRing<RationalField>),
    IG(IteratedPolynomialRing<FiniteField>),
}

/// A polynomial ring as a domain.
#[pyclass(
    name = "SparsePolynomialRing",
    module = "sympy_extras_rs.polys.domains.sparsepolynomialring",
    frozen
)]
#[derive(Debug, Clone)]
pub struct PySparseRing {
    pub inner: AnyRing,
}

fn names_of(ob: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    ob.extract::<Vec<String>>()
}

fn element_zz(f: &PyPolyElement) -> PyResult<PolyElement<IntegerRing>> {
    match &f.inner {
        ElemKind::ZZ(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("a polynomial over ZZ is expected")),
    }
}

fn element_qq(f: &PyPolyElement) -> PyResult<PolyElement<RationalField>> {
    match &f.inner {
        ElemKind::QQ(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("a polynomial over QQ is expected")),
    }
}

fn element_gf(f: &PyPolyElement) -> PyResult<PolyElement<FiniteField>> {
    match &f.inner {
        ElemKind::GF(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("a polynomial over GF(p) is expected")),
    }
}

fn wrap(inner: ElemKind) -> PyPolyElement {
    PyPolyElement { inner }
}

/// Runs `$body` with `$r` the ring, `$get` the conversion of a Python
/// element and `$put` the wrapping of a result.
macro_rules! with_any {
    ($ring:expr, ($r:ident, $get:ident, $put:ident) => $body:expr) => {
        match $ring {
            AnyRing::SZ($r) => {
                let $get = element_zz;
                let $put = |f| wrap(ElemKind::ZZ(f));
                $body
            }
            AnyRing::SQ($r) => {
                let $get = element_qq;
                let $put = |f| wrap(ElemKind::QQ(f));
                $body
            }
            AnyRing::SG($r) => {
                let $get = element_gf;
                let $put = |f| wrap(ElemKind::GF(f));
                $body
            }
            AnyRing::IZ($r) => {
                let $get = element_zz;
                let $put = |f| wrap(ElemKind::ZZ(f));
                $body
            }
            AnyRing::IQ($r) => {
                let $get = element_qq;
                let $put = |f| wrap(ElemKind::QQ(f));
                $body
            }
            AnyRing::IG($r) => {
                let $get = element_gf;
                let $put = |f| wrap(ElemKind::GF(f));
                $body
            }
        }
    };
}

#[pymethods]
impl PySparseRing {
    /// `SparsePolynomialRing(ground, names)`: the subresultant gcd.
    #[new]
    fn new(ground: &PyDomain, names: &Bound<'_, PyAny>) -> PyResult<PySparseRing> {
        let names: Vec<String> = names_of(names)?;
        if names.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a polynomial ring needs at least one variable",
            ));
        }
        let inner: AnyRing = match ground.kind {
            DomainKind::ZZ => AnyRing::SZ(SparsePolynomialRing::new(IntegerRing, names)),
            DomainKind::QQ => AnyRing::SQ(SparsePolynomialRing::new(RationalField, names)),
            DomainKind::GF(p) => AnyRing::SG(SparsePolynomialRing::new(FiniteField::new(p).map_err(err)?, names)),
            _ => return Err(PyTypeError::new_err("the ground must be ZZ, QQ or GF(p)")),
        };
        Ok(PySparseRing { inner })
    }

    #[getter]
    fn name(&self) -> String {
        with_any!(&self.inner, (r, _g, _p) => r.name())
    }

    fn __repr__(&self) -> String {
        self.name()
    }

    #[getter]
    fn ring(&self) -> PyPolyRing {
        let inner: RingKind = match &self.inner {
            AnyRing::SZ(r) => RingKind::ZZ(r.ring.clone()),
            AnyRing::SQ(r) => RingKind::QQ(r.ring.clone()),
            AnyRing::SG(r) => RingKind::GF(r.ring.clone()),
            AnyRing::IZ(r) => RingKind::ZZ(r.ring().clone()),
            AnyRing::IQ(r) => RingKind::QQ(r.ring().clone()),
            AnyRing::IG(r) => RingKind::GF(r.ring().clone()),
        };
        PyPolyRing { inner }
    }

    #[getter]
    fn zero(&self) -> PyPolyElement {
        with_any!(&self.inner, (r, _g, put) => put(r.zero()))
    }

    #[getter]
    fn one(&self) -> PyPolyElement {
        with_any!(&self.inner, (r, _g, put) => put(r.one()))
    }

    fn is_negative(&self, a: &PyPolyElement) -> PyResult<bool> {
        with_any!(&self.inner, (r, get, _p) => Ok(r.is_negative(&get(a)?)))
    }

    fn is_unit(&self, a: &PyPolyElement) -> PyResult<bool> {
        with_any!(&self.inner, (r, get, _p) => Ok(r.is_unit(&get(a)?)))
    }

    fn is_one(&self, a: &PyPolyElement) -> PyResult<bool> {
        with_any!(&self.inner, (r, get, _p) => Ok(r.is_one(&get(a)?)))
    }

    fn gcd(&self, a: &PyPolyElement, b: &PyPolyElement) -> PyResult<PyPolyElement> {
        with_any!(&self.inner, (r, get, put) => Ok(put(r.gcd(&get(a)?, &get(b)?))))
    }

    fn lcm(&self, a: &PyPolyElement, b: &PyPolyElement) -> PyResult<PyPolyElement> {
        with_any!(&self.inner, (r, get, put) => Ok(put(r.lcm(&get(a)?, &get(b)?))))
    }

    fn exquo(&self, a: &PyPolyElement, b: &PyPolyElement) -> PyResult<PyPolyElement> {
        with_any!(&self.inner, (r, get, put) => Ok(put(r.exquo(&get(a)?, &get(b)?).map_err(err)?)))
    }

    fn invert(&self, a: &PyPolyElement) -> PyResult<PyPolyElement> {
        with_any!(&self.inner, (r, get, put) => Ok(put(r.invert(&get(a)?).map_err(err)?)))
    }

    fn canonical_unit(&self, a: &PyPolyElement) -> PyResult<PyPolyElement> {
        with_any!(&self.inner, (r, get, put) => Ok(put(r.canonical_unit(&get(a)?))))
    }

    fn mul(&self, a: &PyPolyElement, b: &PyPolyElement) -> PyResult<PyPolyElement> {
        with_any!(&self.inner, (r, get, put) => Ok(put(r.mul(&get(a)?, &get(b)?))))
    }

    /// The content and the primitive part (`SparsePolynomialRing` only).
    fn primitive<'py>(&self, py: Python<'py>, a: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyRing::SZ(r) => {
                let (c, p) = r.primitive(&element_zz(a)?);
                tuple(
                    py,
                    vec![
                        r.ground.to_py(py, &c)?,
                        Bound::new(py, wrap(ElemKind::ZZ(p)))?.into_any(),
                    ],
                )
            }
            AnyRing::SQ(r) => {
                let (c, p) = r.primitive(&element_qq(a)?);
                tuple(
                    py,
                    vec![
                        r.ground.to_py(py, &c)?,
                        Bound::new(py, wrap(ElemKind::QQ(p)))?.into_any(),
                    ],
                )
            }
            AnyRing::SG(r) => {
                let (c, p) = r.primitive(&element_gf(a)?);
                tuple(
                    py,
                    vec![
                        r.ground.to_py(py, &c)?,
                        Bound::new(py, wrap(ElemKind::GF(p)))?.into_any(),
                    ],
                )
            }
            _ => Err(PyTypeError::new_err("primitive of an iterated ring")),
        }
    }

    fn content<'py>(&self, py: Python<'py>, a: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
        self.primitive(py, a)?.get_item(0)
    }

    fn ground_LC(&self, a: &PyPolyElement) -> PyResult<PyPolyElement> {
        match &self.inner {
            AnyRing::IZ(r) => Ok(wrap(ElemKind::ZZ(r.ground_LC(&element_zz(a)?)))),
            AnyRing::IQ(r) => Ok(wrap(ElemKind::QQ(r.ground_LC(&element_qq(a)?)))),
            AnyRing::IG(r) => Ok(wrap(ElemKind::GF(r.ground_LC(&element_gf(a)?)))),
            _ => Err(PyTypeError::new_err("ground_LC of a ring which is not iterated")),
        }
    }

    fn ground_content(&self, a: &PyPolyElement) -> PyResult<PyPolyElement> {
        match &self.inner {
            AnyRing::IZ(r) => Ok(wrap(ElemKind::ZZ(r.ground_content(&element_zz(a)?)))),
            AnyRing::IQ(r) => Ok(wrap(ElemKind::QQ(r.ground_content(&element_qq(a)?)))),
            AnyRing::IG(r) => Ok(wrap(ElemKind::GF(r.ground_content(&element_gf(a)?)))),
            _ => Err(PyTypeError::new_err("ground_content of a ring which is not iterated")),
        }
    }
}

#[pyfunction]
#[pyo3(name = "IntegerPolynomialRing")]
fn integer_polynomial_ring(names: &Bound<'_, PyAny>) -> PyResult<PySparseRing> {
    Ok(PySparseRing {
        inner: AnyRing::SZ(SparsePolynomialRing::sympy(IntegerRing, names_of(names)?)),
    })
}

#[pyfunction]
#[pyo3(name = "RationalPolynomialRing")]
fn rational_polynomial_ring(names: &Bound<'_, PyAny>) -> PyResult<PySparseRing> {
    Ok(PySparseRing {
        inner: AnyRing::SQ(SparsePolynomialRing::sympy(RationalField, names_of(names)?)),
    })
}

#[pyfunction]
#[pyo3(name = "IteratedPolynomialRing")]
fn iterated_polynomial_ring(params: &PySparseRing, gen_names: &Bound<'_, PyAny>) -> PyResult<PySparseRing> {
    let gen_names: Vec<String> = names_of(gen_names)?;
    let inner: AnyRing = match &params.inner {
        AnyRing::SZ(p) => AnyRing::IZ(sympy_extras_core::polys::domains::IteratedPolynomialRing::new(
            p.clone(),
            gen_names,
        )),
        AnyRing::SQ(p) => AnyRing::IQ(sympy_extras_core::polys::domains::IteratedPolynomialRing::new(
            p.clone(),
            gen_names,
        )),
        AnyRing::SG(p) => AnyRing::IG(sympy_extras_core::polys::domains::IteratedPolynomialRing::new(
            p.clone(),
            gen_names,
        )),
        _ => return Err(PyTypeError::new_err("the parameters must be a SparsePolynomialRing")),
    };
    Ok(PySparseRing { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparseRing>()?;
    m.add_function(wrap_pyfunction!(integer_polynomial_ring, m)?)?;
    m.add_function(wrap_pyfunction!(rational_polynomial_ring, m)?)?;
    m.add_function(wrap_pyfunction!(iterated_polynomial_ring, m)?)?;
    Ok(())
}

// -- multisqf

use sympy_extras_core::polys::multisqf::{
    sqf_list as core_sqf_list, IntegerSqfDomain, ParameterSqfDomain, RationalSqfDomain, SqfDomain,
};

#[derive(Debug, Clone)]
enum SqfKind {
    Integer(IntegerSqfDomain),
    Rational(RationalSqfDomain),
    ParameterZZ(ParameterSqfDomain<IntegerRing>),
    ParameterQQ(ParameterSqfDomain<RationalField>),
}

/// `multisqf.SqfDomain`: one Python class for the three flavours.
#[pyclass(name = "SqfDomain", module = "sympy_extras_rs.polys.multisqf", frozen)]
#[derive(Debug, Clone)]
pub struct PySqfDomain {
    inner: SqfKind,
}

#[pymethods]
impl PySqfDomain {
    #[getter]
    fn ring(&self) -> PyPolyRing {
        let inner: RingKind = match &self.inner {
            SqfKind::Integer(d) => RingKind::ZZ(d.ring().clone()),
            SqfKind::Rational(d) => RingKind::QQ(d.ring().clone()),
            SqfKind::ParameterZZ(d) => RingKind::ZZ(d.ring().clone()),
            SqfKind::ParameterQQ(d) => RingKind::QQ(d.ring().clone()),
        };
        PyPolyRing { inner }
    }

    #[getter]
    fn n_gens(&self) -> usize {
        match &self.inner {
            SqfKind::Integer(d) => d.n_gens(),
            SqfKind::Rational(d) => d.n_gens(),
            SqfKind::ParameterZZ(d) => d.n_gens(),
            SqfKind::ParameterQQ(d) => d.n_gens(),
        }
    }
}

#[pyfunction]
#[pyo3(name = "IntegerSqfDomain")]
fn integer_sqf_domain(names: Vec<String>) -> PySqfDomain {
    PySqfDomain {
        inner: SqfKind::Integer(IntegerSqfDomain::new(names)),
    }
}

#[pyfunction]
#[pyo3(name = "RationalSqfDomain")]
fn rational_sqf_domain(names: Vec<String>) -> PySqfDomain {
    PySqfDomain {
        inner: SqfKind::Rational(RationalSqfDomain::new(names)),
    }
}

#[pyfunction]
#[pyo3(name = "ParameterSqfDomain")]
fn parameter_sqf_domain(params: &PySparseRing, gen_names: Vec<String>) -> PyResult<PySqfDomain> {
    let inner: SqfKind = match &params.inner {
        AnyRing::SZ(p) => SqfKind::ParameterZZ(ParameterSqfDomain::new(p.clone(), gen_names)),
        AnyRing::SQ(p) => SqfKind::ParameterQQ(ParameterSqfDomain::new(p.clone(), gen_names)),
        _ => {
            return Err(PyTypeError::new_err(
                "the parameters must be a polynomial ring over ZZ or QQ",
            ))
        }
    };
    Ok(PySqfDomain { inner })
}

fn out_sqf<'py, K: Domain>(
    py: Python<'py>,
    result: (PolyElement<K>, Vec<(PolyElement<K>, usize)>),
    put: fn(PolyElement<K>) -> PyPolyElement,
) -> PyResult<Bound<'py, PyAny>> {
    let (coeff, factors) = result;
    let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
    for (g, k) in factors {
        items.push(tuple(
            py,
            vec![Bound::new(py, put(g))?.into_any(), k.into_pyobject(py)?.into_any()],
        )?);
    }
    tuple(
        py,
        vec![
            Bound::new(py, put(coeff))?.into_any(),
            pyo3::types::PyList::new(py, items)?.into_any(),
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (f, D, all = false))]
fn sqf_list<'py>(py: Python<'py>, f: &PyPolyElement, D: &PySqfDomain, all: bool) -> PyResult<Bound<'py, PyAny>> {
    match &D.inner {
        SqfKind::Integer(d) => out_sqf(py, core_sqf_list(&element_zz(f)?, d, all), |g| wrap(ElemKind::ZZ(g))),
        SqfKind::Rational(d) => out_sqf(py, core_sqf_list(&element_qq(f)?, d, all), |g| wrap(ElemKind::QQ(g))),
        SqfKind::ParameterZZ(d) => out_sqf(py, core_sqf_list(&element_zz(f)?, d, all), |g| wrap(ElemKind::ZZ(g))),
        SqfKind::ParameterQQ(d) => out_sqf(py, core_sqf_list(&element_qq(f)?, d, all), |g| wrap(ElemKind::QQ(g))),
    }
}

pub fn register_multisqf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySqfDomain>()?;
    m.add_function(wrap_pyfunction!(integer_sqf_domain, m)?)?;
    m.add_function(wrap_pyfunction!(rational_sqf_domain, m)?)?;
    m.add_function(wrap_pyfunction!(parameter_sqf_domain, m)?)?;
    m.add_function(wrap_pyfunction!(sqf_list, m)?)?;
    Ok(())
}
