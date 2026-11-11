//! `sympy_extras_rs.polys.numberfields.qqbar`: the class `Algebraic` with
//! the operators and decisions of the Python class, `algebraic_roots` and
//! `primitive_element`.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::{Int, Rat};
use sympy_extras_core::polys::domains::RationalField;
use sympy_extras_core::polys::numberfields::qqbar::{self as qb, Algebraic};
use sympy_extras_core::polys::rings::{PolyElement, PolyRing};
use sympy_extras_core::polys::zerodim as zd;

use crate::rings::{ElemKind, PyPolyElement, PyPolyRing, RingKind};

use crate::convert::{err, extract_int, out_int, tuple, Bridge};

/// An algebraic number: its minimal polynomial and an isolating region.
#[pyclass(name = "Algebraic", module = "sympy_extras_rs.polys.numberfields.qqbar", frozen)]
pub struct PyAlgebraic {
    pub inner: Algebraic,
}

fn wrap(inner: Algebraic) -> PyAlgebraic {
    PyAlgebraic { inner }
}

/// The Python `TypeError` of the decisions on numbers which are not real.
fn real_error(e: sympy_extras_core::error::PolyError) -> PyErr {
    match &e {
        sympy_extras_core::error::PolyError::Value(m)
            if m == "not a real number" || m == "complex numbers are not ordered" =>
        {
            PyTypeError::new_err(m.clone())
        }
        _ => err(e),
    }
}

/// An operand: an `Algebraic`, or an integer or a fraction.
enum Operand<'a> {
    Same(&'a Algebraic),
    Owned(Box<Algebraic>),
}

impl Operand<'_> {
    fn get(&self) -> &Algebraic {
        match self {
            Operand::Same(a) => a,
            Operand::Owned(a) => a,
        }
    }
}

fn operand<'a>(other: &'a Bound<'_, PyAny>) -> PyResult<Option<Operand<'a>>> {
    if let Ok(a) = other.downcast::<PyAlgebraic>() {
        let reference: &'a PyAlgebraic = a.get();
        return Ok(Some(Operand::Same(&reference.inner)));
    }
    match RationalField.extract(other) {
        Ok(q) => Ok(Some(Operand::Owned(Box::new(Algebraic::from_rational(&q))))),
        Err(_) => Ok(None),
    }
}

fn rationals(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Rat>> {
    let mut values: Vec<Rat> = Vec::new();
    for item in ob.try_iter()? {
        values.push(RationalField.extract(&item?)?);
    }
    Ok(values)
}

fn out_rationals<'py>(py: Python<'py>, values: &[Rat]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = values
        .iter()
        .map(|v| RationalField.to_py(py, v))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

fn out_box<'py>(py: Python<'py>, b: &qb::Box4) -> PyResult<Bound<'py, PyAny>> {
    tuple(
        py,
        vec![
            RationalField.to_py(py, &b.0)?,
            RationalField.to_py(py, &b.1)?,
            RationalField.to_py(py, &b.2)?,
            RationalField.to_py(py, &b.3)?,
        ],
    )
}

#[pymethods]
impl PyAlgebraic {
    #[staticmethod]
    fn from_rational(q: &Bound<'_, PyAny>) -> PyResult<PyAlgebraic> {
        Ok(wrap(Algebraic::from_rational(&RationalField.extract(q)?)))
    }

    #[staticmethod]
    fn imaginary_unit() -> PyAlgebraic {
        wrap(Algebraic::imaginary_unit())
    }

    #[getter]
    fn minpoly<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Bound<'py, PyAny>> = self
            .inner
            .minpoly()
            .iter()
            .map(|c: &Int| out_int(py, c))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into_any())
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[getter]
    fn is_real(&self) -> bool {
        self.inner.is_real()
    }

    #[getter]
    fn is_rational(&self) -> bool {
        self.inner.is_rational()
    }

    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn rational_value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        RationalField.to_py(py, &self.inner.rational_value().map_err(err)?)
    }

    fn real_interval<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (a, b) = self.inner.real_interval().map_err(real_error)?;
        tuple(py, vec![RationalField.to_py(py, &a)?, RationalField.to_py(py, &b)?])
    }

    fn refine_real(&self) -> PyResult<()> {
        self.inner.refine_real().map_err(real_error)
    }

    fn enclosure<'py>(&self, py: Python<'py>, eps: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_box(py, &self.inner.enclosure(&RationalField.extract(eps)?).map_err(err)?)
    }

    fn __complex__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (re, im) = self.inner.to_complex().map_err(err)?;
        Ok(pyo3::types::PyComplex::from_doubles(py, re, im).into_any())
    }

    fn __float__(&self) -> PyResult<f64> {
        if !self.inner.is_real() {
            return Err(PyTypeError::new_err("not a real number"));
        }
        Ok(self.inner.to_complex().map_err(err)?.0)
    }

    fn __repr__(&self) -> PyResult<String> {
        self.inner.repr().map_err(err)
    }

    fn __neg__(&self) -> PyAlgebraic {
        wrap(self.inner.neg())
    }

    fn __pos__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn conjugate(&self) -> PyAlgebraic {
        wrap(self.inner.conjugate())
    }

    fn inverse(&self) -> PyResult<PyAlgebraic> {
        Ok(wrap(self.inner.inverse().map_err(err)?))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = other.py();
        match operand(other)? {
            Some(b) => Ok(Py::new(py, wrap(self.inner.add(b.get()).map_err(err)?))?.into_any()),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = other.py();
        match operand(other)? {
            Some(b) => Ok(Py::new(py, wrap(self.inner.sub(b.get()).map_err(err)?))?.into_any()),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = other.py();
        match operand(other)? {
            Some(b) => Ok(Py::new(py, wrap(self.inner.neg().add(b.get()).map_err(err)?))?.into_any()),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __mul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = other.py();
        let this: &Algebraic = &slf.get().inner;
        if other.is(slf) {
            // `b is self`: the power, of the degree of the minimal polynomial
            if this.is_zero() {
                return Ok(Py::new(py, wrap(Algebraic::from_rational(&Rat::ZERO)))?.into_any());
            }
            return Ok(Py::new(py, wrap(this.pow(2).map_err(err)?))?.into_any());
        }
        match operand(other)? {
            Some(b) => Ok(Py::new(py, wrap(this.mul(b.get()).map_err(err)?))?.into_any()),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        PyAlgebraic::__mul__(slf, other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = other.py();
        match operand(other)? {
            Some(b) => Ok(Py::new(py, wrap(self.inner.div(b.get()).map_err(err)?))?.into_any()),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = other.py();
        match operand(other)? {
            Some(b) => Ok(Py::new(py, wrap(self.inner.inverse().map_err(err)?.mul(b.get()).map_err(err)?))?.into_any()),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __pow__(&self, exponent: i64, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyAlgebraic> {
        Ok(wrap(self.inner.pow(exponent).map_err(err)?))
    }

    fn polynomial_value(&self, coefficients: &Bound<'_, PyAny>) -> PyResult<PyAlgebraic> {
        Ok(wrap(
            self.inner.polynomial_value(&rationals(coefficients)?).map_err(err)?,
        ))
    }

    fn nth_root(&self, n: i64) -> PyResult<PyAlgebraic> {
        if n < 1 {
            return Err(PyValueError::new_err("the index of the root must be positive"));
        }
        Ok(wrap(self.inner.nth_root(n as usize).map_err(err)?))
    }

    fn sqrt(&self) -> PyResult<PyAlgebraic> {
        Ok(wrap(self.inner.sqrt().map_err(err)?))
    }

    fn real_part(&self) -> PyResult<PyAlgebraic> {
        Ok(wrap(self.inner.real_part().map_err(err)?))
    }

    fn imaginary_part(&self) -> PyResult<PyAlgebraic> {
        Ok(wrap(self.inner.imaginary_part().map_err(err)?))
    }

    fn sign(&self) -> PyResult<i32> {
        self.inner.sign().map_err(real_error)
    }

    fn __hash__(&self) -> isize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.inner.minpoly(), self.inner.is_real()).hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyObject> {
        let py = other.py();
        let Some(b) = operand(other)? else {
            return Ok(py.NotImplemented());
        };
        let value: bool = match op {
            CompareOp::Eq => self.inner.equals(b.get()).map_err(err)?,
            CompareOp::Ne => !self.inner.equals(b.get()).map_err(err)?,
            _ => {
                let c: i32 = self.inner.compare(b.get()).map_err(real_error)?;
                match op {
                    CompareOp::Lt => c < 0,
                    CompareOp::Le => c <= 0,
                    CompareOp::Gt => c > 0,
                    _ => c >= 0,
                }
            }
        };
        Ok(value.into_pyobject(py)?.to_owned().into_any().unbind())
    }
}

#[pyfunction]
fn algebraic_roots<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut coefficients: Vec<Int> = Vec::new();
    for item in f.try_iter()? {
        coefficients.push(extract_int(&item?)?);
    }
    let roots: Vec<Algebraic> = qb::algebraic_roots(&coefficients).map_err(err)?;
    let items: Vec<Bound<'py, PyAny>> = roots
        .into_iter()
        .map(|r| Ok(Bound::new(py, wrap(r))?.into_any()))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

#[pyfunction]
fn primitive_element<'py>(py: Python<'py>, a: &PyAlgebraic, b: &PyAlgebraic) -> PyResult<Bound<'py, PyAny>> {
    let (theta, p, q) = qb::primitive_element(&a.inner, &b.inner).map_err(err)?;
    tuple(
        py,
        vec![
            Bound::new(py, wrap(theta))?.into_any(),
            out_rationals(py, &p)?,
            out_rationals(py, &q)?,
        ],
    )
}

/// `recognize_algebraic(re, im=0, digits=15, max_degree=6)`.
#[pyfunction]
#[pyo3(signature = (re, im = None, digits = 15, max_degree = 6))]
fn recognize_algebraic(
    re: &Bound<'_, PyAny>,
    im: Option<&Bound<'_, PyAny>>,
    digits: i64,
    max_degree: i64,
) -> PyResult<Option<PyAlgebraic>> {
    if digits < 1 || max_degree < 1 {
        return Err(PyValueError::new_err("digits and max_degree must be positive"));
    }
    let im: Rat = match im {
        Some(v) => RationalField.extract(v)?,
        None => Rat::ZERO,
    };
    let found = sympy_extras_core::polys::numberfields::recognize::recognize_algebraic(
        &RationalField.extract(re)?,
        &im,
        digits as usize,
        max_degree as usize,
    )
    .map_err(err)?;
    Ok(found.map(wrap))
}

pub fn register_recognize(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recognize_algebraic, m)?)?;
    Ok(())
}

fn qq_elements(ob: &Bound<'_, PyAny>) -> PyResult<Vec<PolyElement<sympy_extras_core::polys::domains::RationalField>>> {
    let mut result = Vec::new();
    for item in ob.try_iter()? {
        match &item?.downcast::<PyPolyElement>()?.get().inner {
            ElemKind::QQ(f) => result.push(f.clone()),
            _ => return Err(PyTypeError::new_err("polynomials over QQ are expected")),
        }
    }
    Ok(result)
}

fn qq_ring(ring: &PyPolyRing) -> PyResult<std::sync::Arc<PolyRing<sympy_extras_core::polys::domains::RationalField>>> {
    match &ring.inner {
        RingKind::QQ(r) => Ok(std::sync::Arc::clone(r)),
        _ => Err(PyTypeError::new_err("a ring over QQ is expected")),
    }
}

#[pyfunction]
fn univariate_in_ideal<'py>(
    py: Python<'py>,
    basis: &Bound<'py, PyAny>,
    ring: &PyPolyRing,
    index: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let values = zd::univariate_in_ideal(&qq_elements(basis)?, &qq_ring(ring)?, index).map_err(err)?;
    out_rationals(py, &values)
}

#[pyfunction]
fn solve_zero_dimensional<'py>(
    py: Python<'py>,
    polys: &Bound<'py, PyAny>,
    ring: &PyPolyRing,
) -> PyResult<Bound<'py, PyAny>> {
    let solutions = zd::solve_zero_dimensional(&qq_elements(polys)?, &qq_ring(ring)?).map_err(err)?;
    let points: Vec<Bound<'py, PyAny>> = solutions
        .into_iter()
        .map(|point| {
            let values: Vec<Bound<'py, PyAny>> = point
                .into_iter()
                .map(|a| Ok(Bound::new(py, wrap(a))?.into_any()))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, values)?.into_any())
        })
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, points)?.into_any())
}

pub fn register_zerodim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(univariate_in_ideal, m)?)?;
    m.add_function(wrap_pyfunction!(solve_zero_dimensional, m)?)?;
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAlgebraic>()?;
    m.add_function(wrap_pyfunction!(algebraic_roots, m)?)?;
    m.add_function(wrap_pyfunction!(primitive_element, m)?)?;
    Ok(())
}
