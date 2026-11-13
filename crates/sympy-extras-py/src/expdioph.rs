//! `sympy_extras_rs.solvers.exponential_diophantine`: Pillai's equation and
//! the integer zeros of exponential polynomials.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::{Int, Rat};
use sympy_extras_core::polys::domains::RationalField;
use sympy_extras_core::solvers::exponential_diophantine as ed;

use crate::convert::{err, extract_int, tuple, Bridge};

/// `pillai(a, b, c, r=1, s=1)`.
#[pyfunction]
#[pyo3(signature = (a, b, c, r = None, s = None))]
fn pillai<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    c: &Bound<'py, PyAny>,
    r: Option<&Bound<'py, PyAny>>,
    s: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let r: Int = match r {
        Some(v) => extract_int(v)?,
        None => Int::ONE,
    };
    let s: Int = match s {
        Some(v) => extract_int(v)?,
        None => Int::ONE,
    };
    let solutions = ed::pillai(&extract_int(a)?, &extract_int(b)?, &extract_int(c)?, &r, &s).map_err(err)?;
    let items: Vec<Bound<'py, PyAny>> = solutions
        .iter()
        .map(|(x, y)| {
            tuple(
                py,
                vec![x.into_pyobject(py)?.into_any(), y.into_pyobject(py)?.into_any()],
            )
        })
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

#[pyfunction]
fn exponential_polynomial_roots(terms: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let mut parsed: Vec<ed::ExponentialTerm> = Vec::new();
    for item in terms.try_iter()? {
        let item = item?;
        let mut coefficients: Vec<Rat> = Vec::new();
        for c in item.get_item(0)?.try_iter()? {
            coefficients.push(RationalField.extract(&c?)?);
        }
        parsed.push((coefficients, RationalField.extract(&item.get_item(1)?)?));
    }
    ed::exponential_polynomial_roots(&parsed).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pillai, m)?)?;
    m.add_function(wrap_pyfunction!(exponential_polynomial_roots, m)?)?;
    Ok(())
}
