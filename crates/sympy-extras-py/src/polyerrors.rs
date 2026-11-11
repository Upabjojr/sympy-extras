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

create_exception!(
    polyerrors,
    PolynomialError,
    BasePolynomialError,
    "polys.polyerrors.PolynomialError"
);
create_exception!(
    polyerrors,
    GeneratorsError,
    BasePolynomialError,
    "polys.polyerrors.GeneratorsError"
);
create_exception!(
    polyerrors,
    GeneratorsNeeded,
    GeneratorsError,
    "polys.polyerrors.GeneratorsNeeded"
);
create_exception!(
    polyerrors,
    CoercionFailed,
    BasePolynomialError,
    "polys.polyerrors.CoercionFailed"
);
create_exception!(
    polyerrors,
    ExactQuotientFailed,
    BasePolynomialError,
    "polys.polyerrors.ExactQuotientFailed"
);
create_exception!(
    polyerrors,
    OperationNotSupported,
    BasePolynomialError,
    "polys.polyerrors.OperationNotSupported"
);
create_exception!(
    polyerrors,
    NotInvertible,
    BasePolynomialError,
    "polys.polyerrors.NotInvertible"
);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py: Python<'_> = m.py();
    m.add("BasePolynomialError", py.get_type::<BasePolynomialError>())?;
    m.add("HeuristicGCDFailed", py.get_type::<HeuristicGCDFailed>())?;
    m.add("PolynomialError", py.get_type::<PolynomialError>())?;
    m.add("GeneratorsError", py.get_type::<GeneratorsError>())?;
    m.add("GeneratorsNeeded", py.get_type::<GeneratorsNeeded>())?;
    m.add("CoercionFailed", py.get_type::<CoercionFailed>())?;
    m.add("ExactQuotientFailed", py.get_type::<ExactQuotientFailed>())?;
    m.add("OperationNotSupported", py.get_type::<OperationNotSupported>())?;
    m.add("NotInvertible", py.get_type::<NotInvertible>())?;

    Ok(())
}

/// The Python exception of a `polys.polyerrors` kind.
pub fn polys_error(kind: sympy_extras_core::error::PolysErrorKind, message: String) -> PyErr {
    use sympy_extras_core::error::PolysErrorKind as K;
    match kind {
        K::PolynomialError => PolynomialError::new_err(message),
        K::GeneratorsError => GeneratorsError::new_err(message),
        K::GeneratorsNeeded => GeneratorsNeeded::new_err(message),
        K::CoercionFailed => CoercionFailed::new_err(message),
        K::ExactQuotientFailed => ExactQuotientFailed::new_err(message),
        K::OperationNotSupported => OperationNotSupported::new_err(message),
        K::NotInvertible => NotInvertible::new_err(message),
    }
}
