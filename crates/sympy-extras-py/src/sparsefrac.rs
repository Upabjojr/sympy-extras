//! `sympy_extras_rs.polys.domains.sparsefractionfield`: the fields of
//! rational functions `K(y1, ..., yk)` over the sparse polynomial rings of
//! `ZZ` and `QQ`, with the methods of the Python class. The elements have
//! `numer` and `denom`, polynomials of `sparsepolynomialring`.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::polys::domains::{Domain, IntegerRing, RationalField, SparseFracElement, SparseFractionField};
use sympy_extras_core::polys::rings::PolyElement;

use crate::convert::err;
use crate::rings::{ElemKind, PyPolyElement};
use crate::sparse::{AnyRing, PySparseRing};

#[derive(Debug, Clone)]
enum AnyField {
    Z(SparseFractionField<IntegerRing>),
    Q(SparseFractionField<RationalField>),
}

#[derive(Debug, Clone)]
enum AnyFrac {
    Z(SparseFracElement<IntegerRing>),
    Q(SparseFracElement<RationalField>),
}

/// `numer/denom` in lowest terms.
#[pyclass(
    name = "SparseFracElement",
    module = "sympy_extras_rs.polys.domains.sparsefractionfield",
    frozen
)]
pub struct PySparseFrac {
    inner: AnyFrac,
}

#[pymethods]
impl PySparseFrac {
    #[getter]
    fn numer(&self) -> PyPolyElement {
        match &self.inner {
            AnyFrac::Z(a) => PyPolyElement {
                inner: ElemKind::ZZ(a.numer.clone()),
            },
            AnyFrac::Q(a) => PyPolyElement {
                inner: ElemKind::QQ(a.numer.clone()),
            },
        }
    }

    #[getter]
    fn denom(&self) -> PyPolyElement {
        match &self.inner {
            AnyFrac::Z(a) => PyPolyElement {
                inner: ElemKind::ZZ(a.denom.clone()),
            },
            AnyFrac::Q(a) => PyPolyElement {
                inner: ElemKind::QQ(a.denom.clone()),
            },
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            AnyFrac::Z(a) => format!("SparseFracElement({}, {})", a.numer, a.denom),
            AnyFrac::Q(a) => format!("SparseFracElement({}, {})", a.numer, a.denom),
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match (other.downcast::<PySparseFrac>(), &self.inner) {
            (Ok(o), AnyFrac::Z(a)) => matches!(&o.get().inner, AnyFrac::Z(b) if a == b),
            (Ok(o), AnyFrac::Q(a)) => matches!(&o.get().inner, AnyFrac::Q(b) if a == b),
            _ => false,
        }
    }
}

/// `SparseFractionField(ring)`.
#[pyclass(
    name = "SparseFractionField",
    module = "sympy_extras_rs.polys.domains.sparsefractionfield",
    frozen
)]
pub struct PySparseFracField {
    inner: AnyField,
}

fn poly_zz(f: &PyPolyElement) -> PyResult<PolyElement<IntegerRing>> {
    match &f.inner {
        ElemKind::ZZ(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("a polynomial over ZZ is expected")),
    }
}

fn poly_qq(f: &PyPolyElement) -> PyResult<PolyElement<RationalField>> {
    match &f.inner {
        ElemKind::QQ(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("a polynomial over QQ is expected")),
    }
}

fn frac_z(a: &PySparseFrac) -> PyResult<SparseFracElement<IntegerRing>> {
    match &a.inner {
        AnyFrac::Z(a) => Ok(a.clone()),
        _ => Err(PyTypeError::new_err("an element of a field over ZZ is expected")),
    }
}

fn frac_q(a: &PySparseFrac) -> PyResult<SparseFracElement<RationalField>> {
    match &a.inner {
        AnyFrac::Q(a) => Ok(a.clone()),
        _ => Err(PyTypeError::new_err("an element of a field over QQ is expected")),
    }
}

/// Runs `$body` with `$f` the field, `$get` the conversion of an element
/// and `$put` the wrapping of a result.
macro_rules! with_field {
    ($field:expr, ($f:ident, $get:ident, $put:ident) => $body:expr) => {
        match $field {
            AnyField::Z($f) => {
                let $get = frac_z;
                let $put = |a| PySparseFrac { inner: AnyFrac::Z(a) };
                $body
            }
            AnyField::Q($f) => {
                let $get = frac_q;
                let $put = |a| PySparseFrac { inner: AnyFrac::Q(a) };
                $body
            }
        }
    };
}

#[pymethods]
impl PySparseFracField {
    #[new]
    fn new(ring: &PySparseRing) -> PyResult<PySparseFracField> {
        let inner: AnyField = match &ring.inner {
            AnyRing::SZ(r) => AnyField::Z(SparseFractionField::new(r.clone())),
            AnyRing::SQ(r) => AnyField::Q(SparseFractionField::new(r.clone())),
            _ => {
                return Err(PyTypeError::new_err(
                    "the ring must be a sparse polynomial ring over ZZ or QQ",
                ))
            }
        };
        Ok(PySparseFracField { inner })
    }

    #[getter]
    fn name(&self) -> String {
        with_field!(&self.inner, (f, _g, _p) => f.name())
    }

    fn __repr__(&self) -> String {
        self.name()
    }

    #[getter]
    fn is_field(&self) -> bool {
        true
    }

    #[getter]
    fn zero(&self) -> PySparseFrac {
        with_field!(&self.inner, (f, _g, put) => put(f.zero()))
    }

    #[getter]
    fn one(&self) -> PySparseFrac {
        with_field!(&self.inner, (f, _g, put) => put(f.one()))
    }

    #[getter]
    fn gens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<PySparseFrac> =
            with_field!(&self.inner, (f, _g, put) => f.gens().into_iter().map(put).collect());
        let objects: Vec<Bound<'py, PyAny>> = items
            .into_iter()
            .map(|a| Ok(Bound::new(py, a)?.into_any()))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, objects)?.into_any())
    }

    /// `new(numer, denom)`: the fraction in lowest terms.
    #[pyo3(name = "new")]
    fn new_element(&self, numer: &PyPolyElement, denom: &PyPolyElement) -> PyResult<PySparseFrac> {
        match &self.inner {
            AnyField::Z(f) => Ok(PySparseFrac {
                inner: AnyFrac::Z(f.new_element(&poly_zz(numer)?, &poly_zz(denom)?).map_err(err)?),
            }),
            AnyField::Q(f) => Ok(PySparseFrac {
                inner: AnyFrac::Q(f.new_element(&poly_qq(numer)?, &poly_qq(denom)?).map_err(err)?),
            }),
        }
    }

    fn from_ring(&self, a: &PyPolyElement) -> PyResult<PySparseFrac> {
        match &self.inner {
            AnyField::Z(f) => Ok(PySparseFrac {
                inner: AnyFrac::Z(f.from_ring(&poly_zz(a)?)),
            }),
            AnyField::Q(f) => Ok(PySparseFrac {
                inner: AnyFrac::Q(f.from_ring(&poly_qq(a)?)),
            }),
        }
    }

    fn from_int(&self, n: i64) -> PySparseFrac {
        with_field!(&self.inner, (f, _g, put) => put(f.from_int(n)))
    }

    fn is_zero(&self, a: &PySparseFrac) -> PyResult<bool> {
        with_field!(&self.inner, (f, get, _p) => Ok(f.is_zero(&get(a)?)))
    }

    fn is_one(&self, a: &PySparseFrac) -> PyResult<bool> {
        with_field!(&self.inner, (f, get, _p) => Ok(f.is_one(&get(a)?)))
    }

    fn is_negative(&self, a: &PySparseFrac) -> PyResult<bool> {
        with_field!(&self.inner, (f, get, _p) => Ok(f.is_negative(&get(a)?)))
    }

    fn add(&self, a: &PySparseFrac, b: &PySparseFrac) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.add(&get(a)?, &get(b)?))))
    }

    fn sub(&self, a: &PySparseFrac, b: &PySparseFrac) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.sub(&get(a)?, &get(b)?))))
    }

    fn neg(&self, a: &PySparseFrac) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.neg(&get(a)?))))
    }

    fn mul(&self, a: &PySparseFrac, b: &PySparseFrac) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.mul(&get(a)?, &get(b)?))))
    }

    fn exquo(&self, a: &PySparseFrac, b: &PySparseFrac) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.exquo(&get(a)?, &get(b)?).map_err(err)?)))
    }

    fn invert(&self, a: &PySparseFrac) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.invert(&get(a)?).map_err(err)?)))
    }

    fn pow(&self, a: &PySparseFrac, n: i64) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.pow_signed(&get(a)?, n).map_err(err)?)))
    }

    fn canonical_unit(&self, a: &PySparseFrac) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.canonical_unit(&get(a)?))))
    }

    fn diff(&self, a: &PySparseFrac, i: usize) -> PyResult<PySparseFrac> {
        with_field!(&self.inner, (f, get, put) => Ok(put(f.diff(&get(a)?, i))))
    }

    fn size(&self, a: &PySparseFrac) -> PyResult<u64> {
        with_field!(&self.inner, (f, get, _p) => Ok(f.size(&get(a)?)))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparseFracField>()?;
    m.add_class::<PySparseFrac>()?;
    Ok(())
}
