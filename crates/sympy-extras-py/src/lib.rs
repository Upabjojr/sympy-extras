//! PyO3 bindings: the extension module `sympy_extras_rs`, distributed on
//! PyPI as `sympy-extras-rs`, separately from the pure Python
//! `sympy-extras`.
//!
//! The submodules mirror the Python package (`sympy_extras_rs.polys.
//! densearith` has the functions of `sympy_extras.polys.densearith`, with
//! the same arguments and results), so that the same tests run against
//! both implementations. Each submodule is registered in `sys.modules`, so
//! that `import sympy_extras_rs.polys.densearith` works.

// the Python names (`from_int` is a method of the domain)
#![allow(clippy::wrong_self_convention)]

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod convert;
mod dense;
mod domains;

/// A submodule `parent.name`, registered as `full` in `sys.modules`.
fn submodule<'py>(parent: &Bound<'py, PyModule>, name: &str, full: &str) -> PyResult<Bound<'py, PyModule>> {
    let py: Python<'py> = parent.py();
    let m: Bound<'py, PyModule> = PyModule::new(py, name)?;
    m.setattr("__name__", full)?;
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item(full, &m)?;
    Ok(m)
}

#[pymodule]
fn sympy_extras_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", sympy_extras_core::VERSION)?;
    let polys: Bound<'_, PyModule> = submodule(m, "polys", "sympy_extras_rs.polys")?;
    domains::register(&submodule(&polys, "domains", "sympy_extras_rs.polys.domains")?)?;
    dense::register_densebasic(&submodule(&polys, "densebasic", "sympy_extras_rs.polys.densebasic")?)?;
    dense::register_densearith(&submodule(&polys, "densearith", "sympy_extras_rs.polys.densearith")?)?;
    dense::register_densetools(&submodule(&polys, "densetools", "sympy_extras_rs.polys.densetools")?)?;
    dense::register_euclidtools(&submodule(&polys, "euclidtools", "sympy_extras_rs.polys.euclidtools")?)?;
    dense::register_sqfreetools(&submodule(&polys, "sqfreetools", "sympy_extras_rs.polys.sqfreetools")?)?;
    Ok(())
}
