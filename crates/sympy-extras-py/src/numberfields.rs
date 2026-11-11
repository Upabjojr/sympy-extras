//! `sympy_extras_rs.polys.numberfields.basis`: integral bases and field
//! discriminants by the Round 2 algorithm.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::Int;
use sympy_extras_core::polys::domains::RationalField;
use sympy_extras_core::polys::numberfields::basis as nb;

use crate::convert::{err, extract_int, out_int, Bridge};

fn ints(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Int>> {
    let mut values: Vec<Int> = Vec::new();
    for item in ob.try_iter()? {
        values.push(extract_int(&item?)?);
    }
    Ok(values)
}

#[pyfunction]
fn polynomial_discriminant<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_int(py, &nb::polynomial_discriminant(&ints(f)?).map_err(err)?)
}

#[pyfunction]
fn field_discriminant<'py>(py: Python<'py>, minpoly: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_int(py, &nb::field_discriminant(&ints(minpoly)?).map_err(err)?)
}

#[pyfunction]
fn integral_basis<'py>(py: Python<'py>, minpoly: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let basis = nb::integral_basis(&ints(minpoly)?).map_err(err)?;
    let rows: Vec<Bound<'py, PyAny>> = basis
        .iter()
        .map(|row| {
            let items: Vec<Bound<'py, PyAny>> = row
                .iter()
                .map(|c| RationalField.to_py(py, c))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, items)?.into_any())
        })
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, rows)?.into_any())
}

pub fn register_basis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(polynomial_discriminant, m)?)?;
    m.add_function(wrap_pyfunction!(field_discriminant, m)?)?;
    m.add_function(wrap_pyfunction!(integral_basis, m)?)?;
    Ok(())
}
