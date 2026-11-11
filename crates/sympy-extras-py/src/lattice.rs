//! `sympy_extras_rs.polys.lattice`: LLL reduction, the Hermite and Smith
//! normal forms and the integer linear systems, with the classes
//! `LatticeReduction` and `IntegerSolutions` of the Python module.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::{Int, Rat};
use sympy_extras_core::polys::domains::RationalField;
use sympy_extras_core::polys::lattice::{self as la, Vector};

use crate::convert::{err, extract_int, out_int, Bridge};

fn vector(ob: &Bound<'_, PyAny>) -> PyResult<Vector> {
    let mut values: Vector = Vec::new();
    for item in ob.try_iter()? {
        values.push(extract_int(&item?)?);
    }
    Ok(values)
}

fn vectors(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Vector>> {
    let mut rows: Vec<Vector> = Vec::new();
    for item in ob.try_iter()? {
        rows.push(vector(&item?)?);
    }
    Ok(rows)
}

fn out_vector<'py>(py: Python<'py>, v: &[Int]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = v.iter().map(|c| out_int(py, c)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

fn out_vectors<'py>(py: Python<'py>, rows: &[Vector]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = rows.iter().map(|r| out_vector(py, r)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

fn delta_from(delta: Option<&Bound<'_, PyAny>>) -> PyResult<Rat> {
    match delta {
        Some(d) => RationalField.extract(d),
        None => Ok(la::default_delta()),
    }
}

/// An LLL reduced basis with its Gram determinants.
#[pyclass(name = "LatticeReduction", module = "sympy_extras_rs.polys.lattice", frozen)]
pub struct PyLatticeReduction {
    inner: la::LatticeReduction,
}

#[pymethods]
impl PyLatticeReduction {
    #[getter]
    fn basis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_vectors(py, &self.inner.basis)
    }

    #[getter]
    fn gram<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_vector(py, &self.inner.gram)
    }

    fn norms_squared<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Bound<'py, PyAny>> = self
            .inner
            .norms_squared()
            .iter()
            .map(|r| RationalField.to_py(py, r))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into_any())
    }

    fn short_part<'py>(&self, py: Python<'py>, bound_squared: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_vectors(py, &self.inner.short_part(&RationalField.extract(bound_squared)?))
    }
}

/// The integer solutions of a linear system.
#[pyclass(name = "IntegerSolutions", module = "sympy_extras_rs.polys.lattice", frozen)]
pub struct PyIntegerSolutions {
    inner: la::IntegerSolutions,
}

#[pymethods]
impl PyIntegerSolutions {
    #[getter]
    fn particular<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_vector(py, &self.inner.particular)
    }

    #[getter]
    fn kernel<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_vectors(py, &self.inner.kernel)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IntegerSolutions({}, {})",
            self.particular(py)?.repr()?,
            self.kernel(py)?.repr()?
        ))
    }
}

#[pyfunction]
fn hermite_normal_form<'py>(py: Python<'py>, rows: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_vectors(py, &la::hermite_normal_form(&vectors(rows)?).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (rows, delta = None))]
fn lll_reduction(rows: &Bound<'_, PyAny>, delta: Option<&Bound<'_, PyAny>>) -> PyResult<PyLatticeReduction> {
    Ok(PyLatticeReduction {
        inner: la::lll_reduction(&vectors(rows)?, &delta_from(delta)?).map_err(err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (rows, delta = None))]
fn lll<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
    delta: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    out_vectors(py, &la::lll(&vectors(rows)?, &delta_from(delta)?).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (rows, delta = None))]
fn is_lll_reduced(rows: &Bound<'_, PyAny>, delta: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    Ok(la::is_lll_reduced(&vectors(rows)?, &delta_from(delta)?))
}

#[pyfunction]
fn integer_kernel<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>, unknowns: usize) -> PyResult<Bound<'py, PyAny>> {
    out_vectors(py, &la::integer_kernel(&vectors(matrix)?, unknowns).map_err(err)?)
}

#[pyfunction]
fn solve_integer_system(
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    unknowns: usize,
) -> PyResult<Option<PyIntegerSolutions>> {
    let solutions = la::solve_integer_system(&vectors(matrix)?, &vector(rhs)?, unknowns).map_err(err)?;
    Ok(solutions.map(|inner| PyIntegerSolutions { inner }))
}

#[pyfunction]
fn smith_normal_form<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_vector(py, &la::smith_normal_form(&vectors(matrix)?).map_err(err)?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLatticeReduction>()?;
    m.add_class::<PyIntegerSolutions>()?;
    m.add_function(wrap_pyfunction!(hermite_normal_form, m)?)?;
    m.add_function(wrap_pyfunction!(lll_reduction, m)?)?;
    m.add_function(wrap_pyfunction!(lll, m)?)?;
    m.add_function(wrap_pyfunction!(is_lll_reduced, m)?)?;
    m.add_function(wrap_pyfunction!(integer_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(solve_integer_system, m)?)?;
    m.add_function(wrap_pyfunction!(smith_normal_form, m)?)?;
    Ok(())
}
