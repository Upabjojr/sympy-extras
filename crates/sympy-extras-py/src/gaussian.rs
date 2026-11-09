//! `sympy_extras_rs.polys.domains.gaussiandomains`: the elements
//! `GaussianInteger` and `GaussianRational` (equal to any object with the
//! same `x` and `y`, such as the Python classes) and the domains `ZZ_I`,
//! `QQ_I` (kinds of `Domain`, so that every dense binding runs over them).

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sympy_extras_core::numbers::{Int, Rat};
use sympy_extras_core::polys::domains::{
    GaussianInteger, GaussianIntegerRing, GaussianRational, GaussianRationalField, RationalField,
};

use crate::convert::{extract_int, out_int, Bridge};
use crate::domains::{DomainKind, PyDomain};

/// `x + y*I` with integer parts.
#[pyclass(name = "GaussianInteger", module = "sympy_extras_rs.polys.domains.gaussiandomains", frozen)]
pub struct PyGaussianInteger {
    #[pyo3(get)]
    x: Py<PyAny>,
    #[pyo3(get)]
    y: Py<PyAny>,
}

/// `x + y*I` with rational parts.
#[pyclass(name = "GaussianRational", module = "sympy_extras_rs.polys.domains.gaussiandomains", frozen)]
pub struct PyGaussianRational {
    #[pyo3(get)]
    x: Py<PyAny>,
    #[pyo3(get)]
    y: Py<PyAny>,
}

fn same_parts(py: Python<'_>, x: &Py<PyAny>, y: &Py<PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let (Ok(ox), Ok(oy)) = (other.getattr("x"), other.getattr("y")) else {
        return Ok(false);
    };
    Ok(x.bind(py).eq(ox)? && y.bind(py).eq(oy)?)
}

#[pymethods]
impl PyGaussianInteger {
    #[new]
    #[pyo3(signature = (x, y = None))]
    fn new(py: Python<'_>, x: &Bound<'_, PyAny>, y: Option<&Bound<'_, PyAny>>) -> PyResult<PyGaussianInteger> {
        let y: Int = match y {
            Some(v) => extract_int(v)?,
            None => Int::ZERO,
        };
        Ok(PyGaussianInteger {
            x: out_int(py, &extract_int(x)?)?.unbind(),
            y: out_int(py, &y)?.unbind(),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("GaussianInteger({}, {})", self.x.bind(py).str()?, self.y.bind(py).str()?))
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        same_parts(py, &self.x, &self.y, other)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        (self.x.bind(py), self.y.bind(py)).into_pyobject(py)?.hash()
    }

    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.x.bind(py).is_truthy()? || self.y.bind(py).is_truthy()?)
    }
}

#[pymethods]
impl PyGaussianRational {
    #[new]
    #[pyo3(signature = (x, y = None))]
    fn new(py: Python<'_>, x: &Bound<'_, PyAny>, y: Option<&Bound<'_, PyAny>>) -> PyResult<PyGaussianRational> {
        let y: Rat = match y {
            Some(v) => RationalField.extract(v)?,
            None => Rat::ZERO,
        };
        Ok(PyGaussianRational {
            x: RationalField.to_py(py, &RationalField.extract(x)?)?.unbind(),
            y: RationalField.to_py(py, &y)?.unbind(),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("GaussianRational({}, {})", self.x.bind(py).str()?, self.y.bind(py).str()?))
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        same_parts(py, &self.x, &self.y, other)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        (self.x.bind(py), self.y.bind(py)).into_pyobject(py)?.hash()
    }

    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.x.bind(py).is_truthy()? || self.y.bind(py).is_truthy()?)
    }
}

/// An element is any object with the parts `x` and `y`, or an integer.
impl Bridge for GaussianIntegerRing {
    fn extract(&self, ob: &Bound<'_, PyAny>) -> PyResult<GaussianInteger> {
        if let (Ok(x), Ok(y)) = (ob.getattr("x"), ob.getattr("y")) {
            return Ok(GaussianInteger::new(extract_int(&x)?, extract_int(&y)?));
        }
        Ok(GaussianInteger::new(extract_int(ob)?, Int::ZERO))
    }

    fn to_py<'py>(&self, py: Python<'py>, e: &GaussianInteger) -> PyResult<Bound<'py, PyAny>> {
        let element = PyGaussianInteger {
            x: out_int(py, &e.x)?.unbind(),
            y: out_int(py, &e.y)?.unbind(),
        };
        Ok(Bound::new(py, element)?.into_any())
    }
}

impl Bridge for GaussianRationalField {
    fn extract(&self, ob: &Bound<'_, PyAny>) -> PyResult<GaussianRational> {
        if let (Ok(x), Ok(y)) = (ob.getattr("x"), ob.getattr("y")) {
            return Ok(GaussianRational::new(RationalField.extract(&x)?, RationalField.extract(&y)?));
        }
        Ok(GaussianRational::new(RationalField.extract(ob)?, Rat::ZERO))
    }

    fn to_py<'py>(&self, py: Python<'py>, e: &GaussianRational) -> PyResult<Bound<'py, PyAny>> {
        let element = PyGaussianRational {
            x: RationalField.to_py(py, &e.x)?.unbind(),
            y: RationalField.to_py(py, &e.y)?.unbind(),
        };
        Ok(Bound::new(py, element)?.into_any())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianInteger>()?;
    m.add_class::<PyGaussianRational>()?;
    m.add("ZZ_I", PyDomain::base(DomainKind::GaussianZZ))?;
    m.add("QQ_I", PyDomain::base(DomainKind::GaussianQQ))?;
    Ok(())
}
