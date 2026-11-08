//! `sympy_extras_rs.polys.polyerrors`: the exceptions of the algorithms
//! which are not `ValueError` or `ZeroDivisionError`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyModule;

create_exception!(
    polyerrors,
    BasePolynomialError,
    PyException,
    "The base of the polynomial errors."
);
create_exception!(
    polyerrors,
    HeuristicGCDFailed,
    BasePolynomialError,
    "No evaluation point gave the gcd."
);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py: Python<'_> = m.py();
    m.add("BasePolynomialError", py.get_type::<BasePolynomialError>())?;
    m.add("HeuristicGCDFailed", py.get_type::<HeuristicGCDFailed>())?;
    Ok(())
}
