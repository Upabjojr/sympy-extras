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

mod algorithms;
mod convert;
mod dense;
mod domains;
mod factor;
mod ntheory;
mod polyerrors;
mod rings;
mod sparse;

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
    let ntheory_module: Bound<'_, PyModule> = submodule(m, "ntheory", "sympy_extras_rs.ntheory")?;
    ntheory::register(
        &ntheory_module,
        &submodule(&ntheory_module, "primetest", "sympy_extras_rs.ntheory.primetest")?,
        &submodule(&ntheory_module, "generate", "sympy_extras_rs.ntheory.generate")?,
        &submodule(&ntheory_module, "factor_", "sympy_extras_rs.ntheory.factor_")?,
    )?;
    let polys: Bound<'_, PyModule> = submodule(m, "polys", "sympy_extras_rs.polys")?;
    let domains_module: Bound<'_, PyModule> = submodule(&polys, "domains", "sympy_extras_rs.polys.domains")?;
    domains::register(&domains_module)?;
    domains::register_polynomialring(&submodule(
        &domains_module,
        "polynomialring",
        "sympy_extras_rs.polys.domains.polynomialring",
    )?)?;
    domains::register_fractionfield(&submodule(
        &domains_module,
        "fractionfield",
        "sympy_extras_rs.polys.domains.fractionfield",
    )?)?;
    sparse::register(&submodule(
        &domains_module,
        "sparsepolynomialring",
        "sympy_extras_rs.polys.domains.sparsepolynomialring",
    )?)?;
    dense::register_densebasic(&submodule(&polys, "densebasic", "sympy_extras_rs.polys.densebasic")?)?;
    dense::register_densearith(&submodule(&polys, "densearith", "sympy_extras_rs.polys.densearith")?)?;
    dense::register_densetools(&submodule(&polys, "densetools", "sympy_extras_rs.polys.densetools")?)?;
    dense::register_euclidtools(&submodule(&polys, "euclidtools", "sympy_extras_rs.polys.euclidtools")?)?;
    dense::register_sqfreetools(&submodule(&polys, "sqfreetools", "sympy_extras_rs.polys.sqfreetools")?)?;
    factor::register_galoistools(&submodule(&polys, "galoistools", "sympy_extras_rs.polys.galoistools")?)?;
    factor::register_factortools(&submodule(&polys, "factortools", "sympy_extras_rs.polys.factortools")?)?;
    polyerrors::register(&submodule(&polys, "polyerrors", "sympy_extras_rs.polys.polyerrors")?)?;
    rings::register_rings(&submodule(&polys, "rings", "sympy_extras_rs.polys.rings")?)?;
    algorithms::register_rootisolation(&submodule(
        &polys,
        "rootisolation",
        "sympy_extras_rs.polys.rootisolation",
    )?)?;
    algorithms::register_stronggroebner(&submodule(
        &polys,
        "stronggroebner",
        "sympy_extras_rs.polys.stronggroebner",
    )?)?;
    sparse::register_multisqf(&submodule(&polys, "multisqf", "sympy_extras_rs.polys.multisqf")?)?;
    algorithms::register_f4(&submodule(&polys, "f4", "sympy_extras_rs.polys.f4")?)?;
    rings::register_heuristicgcd(&submodule(
        &polys,
        "heuristicgcd",
        "sympy_extras_rs.polys.heuristicgcd",
    )?)?;
    Ok(())
}
