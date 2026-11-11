//! `sympy_extras_rs.solvers.simplex`: exact linear and integer programming
//! and Fourier–Motzkin elimination, with the result classes of the Python
//! module (`LinearProgramResult`, `IntegerProgramResult`, `LinearSystem`).

// the Python names of the matrices (`A`, `b`)
#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::{Int, Rat};
use sympy_extras_core::polys::domains::RationalField;
use sympy_extras_core::solvers::simplex as sx;

use crate::convert::{err, out_int, Bridge};

fn rationals(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Rat>> {
    let mut values: Vec<Rat> = Vec::new();
    for item in ob.try_iter()? {
        values.push(RationalField.extract(&item?)?);
    }
    Ok(values)
}

fn matrix(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Rat>>> {
    let mut rows: Vec<Vec<Rat>> = Vec::new();
    for item in ob.try_iter()? {
        rows.push(rationals(&item?)?);
    }
    Ok(rows)
}

fn flags(ob: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<bool>>> {
    match ob {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            let mut values: Vec<bool> = Vec::new();
            for item in o.try_iter()? {
                values.push(item?.is_truthy()?);
            }
            Ok(Some(values))
        }
    }
}

fn out_rational<'py>(py: Python<'py>, v: &Rat) -> PyResult<Bound<'py, PyAny>> {
    RationalField.to_py(py, v)
}

fn out_rationals<'py>(py: Python<'py>, values: &[Rat]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = values.iter().map(|v| out_rational(py, v)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

fn out_optional<'py>(py: Python<'py>, values: &Option<Vec<Rat>>) -> PyResult<Option<Bound<'py, PyAny>>> {
    values.as_ref().map(|v| out_rationals(py, v)).transpose()
}

/// Python's `str` of a `Fraction` (`11`, `14/5`) or `None`.
fn value_str(value: &Option<Rat>) -> String {
    match value {
        None => "None".to_string(),
        Some(v) => v.to_string(),
    }
}

/// The outcome of `linear_program`.
#[pyclass(name = "LinearProgramResult", module = "sympy_extras_rs.solvers.simplex", frozen)]
pub struct PyLinearProgramResult {
    inner: sx::LinearProgramResult,
}

#[pymethods]
impl PyLinearProgramResult {
    #[getter]
    fn status(&self) -> String {
        self.inner.status.to_string()
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner.value.as_ref().map(|v| out_rational(py, v)).transpose()
    }

    #[getter]
    fn point<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        out_optional(py, &self.inner.point)
    }

    #[getter]
    fn dual<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        out_optional(py, &self.inner.dual)
    }

    #[getter]
    fn ray<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        out_optional(py, &self.inner.ray)
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearProgramResult('{}', value={})",
            self.inner.status,
            value_str(&self.inner.value)
        )
    }
}

/// The outcome of `integer_program`.
#[pyclass(name = "IntegerProgramResult", module = "sympy_extras_rs.solvers.simplex", frozen)]
pub struct PyIntegerProgramResult {
    inner: sx::IntegerProgramResult,
}

#[pymethods]
impl PyIntegerProgramResult {
    #[getter]
    fn status(&self) -> String {
        self.inner.status.to_string()
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner.value.as_ref().map(|v| out_rational(py, v)).transpose()
    }

    #[getter]
    fn point<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner.point {
            None => Ok(None),
            Some(p) => {
                let items: Vec<Bound<'py, PyAny>> = p.iter().map(|v: &Int| out_int(py, v)).collect::<PyResult<_>>()?;
                Ok(Some(PyList::new(py, items)?.into_any()))
            }
        }
    }

    fn __repr__(&self) -> String {
        let point: String = match &self.inner.point {
            None => "None".to_string(),
            Some(p) => format!(
                "[{}]",
                p.iter().map(|v| v.to_string()).collect::<Vec<String>>().join(", ")
            ),
        };
        format!(
            "IntegerProgramResult('{}', value={}, point={})",
            self.inner.status,
            value_str(&self.inner.value),
            point
        )
    }
}

/// The inequalities `A_i x <= b_i`, strict on the rows marked in `strict`.
#[pyclass(name = "LinearSystem", module = "sympy_extras_rs.solvers.simplex", frozen)]
pub struct PyLinearSystem {
    inner: sx::LinearSystem,
}

#[pymethods]
impl PyLinearSystem {
    #[getter]
    fn A<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rows: Vec<Bound<'py, PyAny>> = self
            .inner
            .a
            .iter()
            .map(|r| out_rationals(py, r))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, rows)?.into_any())
    }

    #[getter]
    fn b<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rationals(py, &self.inner.b)
    }

    #[getter]
    fn strict(&self) -> Vec<bool> {
        self.inner.strict.clone()
    }

    fn contains(&self, point: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.contains(&rationals(point)?))
    }

    fn is_feasible(&self) -> PyResult<bool> {
        self.inner.is_feasible().map_err(err)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "LinearSystem({}, {}, {})",
            self.A(py)?.repr()?,
            self.b(py)?.repr()?,
            PyList::new(py, &self.inner.strict)?.repr()?
        ))
    }
}

#[pyfunction]
fn linear_program(c: &Bound<'_, PyAny>, A: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyLinearProgramResult> {
    Ok(PyLinearProgramResult {
        inner: sx::linear_program(&rationals(c)?, &matrix(A)?, &rationals(b)?).map_err(err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (A, b, strict = None))]
fn linear_feasible<'py>(
    py: Python<'py>,
    A: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    strict: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let strict: Option<Vec<bool>> = flags(strict)?;
    let point = sx::linear_feasible(&matrix(A)?, &rationals(b)?, strict.as_deref()).map_err(err)?;
    out_optional(py, &point)
}

#[pyfunction]
fn integer_program(
    c: &Bound<'_, PyAny>,
    A: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<PyIntegerProgramResult> {
    Ok(PyIntegerProgramResult {
        inner: sx::integer_program(&rationals(c)?, &matrix(A)?, &rationals(b)?).map_err(err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (A, b, variables, strict = None))]
fn eliminate_variables(
    A: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    variables: &Bound<'_, PyAny>,
    strict: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyLinearSystem> {
    let mut indices: Vec<usize> = Vec::new();
    for item in variables.try_iter()? {
        let j: i64 = item?.extract::<i64>()?;
        if j < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "no unknown number {}",
                j
            )));
        }
        indices.push(j as usize);
    }
    let strict: Option<Vec<bool>> = flags(strict)?;
    Ok(PyLinearSystem {
        inner: sx::eliminate_variables(&matrix(A)?, &rationals(b)?, &indices, strict.as_deref()).map_err(err)?,
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearProgramResult>()?;
    m.add_class::<PyIntegerProgramResult>()?;
    m.add_class::<PyLinearSystem>()?;
    m.add_function(wrap_pyfunction!(linear_program, m)?)?;
    m.add_function(wrap_pyfunction!(linear_feasible, m)?)?;
    m.add_function(wrap_pyfunction!(integer_program, m)?)?;
    m.add_function(wrap_pyfunction!(eliminate_variables, m)?)?;
    Ok(())
}
