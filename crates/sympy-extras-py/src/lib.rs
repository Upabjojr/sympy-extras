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

mod algebraic;
mod algorithms;
mod complexroots;
mod convert;
mod dense;
mod domains;
mod expdioph;
mod factor;
mod galois;
mod gaussian;
mod lattice;
mod ntheory;
mod numberfields;
mod polyerrors;
mod qqbar;
mod rings;
mod simplex;
mod sparse;
mod sparsefrac;

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
    let solvers: Bound<'_, PyModule> = submodule(m, "solvers", "sympy_extras_rs.solvers")?;
    simplex::register(&submodule(&solvers, "simplex", "sympy_extras_rs.solvers.simplex")?)?;
    expdioph::register(&submodule(
        &solvers,
        "exponential_diophantine",
        "sympy_extras_rs.solvers.exponential_diophantine",
    )?)?;
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
    galois::register_galoisfield(&submodule(
        &domains_module,
        "galoisfield",
        "sympy_extras_rs.polys.domains.galoisfield",
    )?)?;
    gaussian::register(&submodule(
        &domains_module,
        "gaussiandomains",
        "sympy_extras_rs.polys.domains.gaussiandomains",
    )?)?;
    algebraic::register_algebraicfield(&submodule(
        &domains_module,
        "algebraicfield",
        "sympy_extras_rs.polys.domains.algebraicfield",
    )?)?;
    sparse::register(&submodule(
        &domains_module,
        "sparsepolynomialring",
        "sympy_extras_rs.polys.domains.sparsepolynomialring",
    )?)?;
    sparsefrac::register(&submodule(
        &domains_module,
        "sparsefractionfield",
        "sympy_extras_rs.polys.domains.sparsefractionfield",
    )?)?;
    dense::register_densebasic(&submodule(&polys, "densebasic", "sympy_extras_rs.polys.densebasic")?)?;
    dense::register_densearith(&submodule(&polys, "densearith", "sympy_extras_rs.polys.densearith")?)?;
    dense::register_densetools(&submodule(&polys, "densetools", "sympy_extras_rs.polys.densetools")?)?;
    dense::register_euclidtools(&submodule(&polys, "euclidtools", "sympy_extras_rs.polys.euclidtools")?)?;
    dense::register_sqfreetools(&submodule(&polys, "sqfreetools", "sympy_extras_rs.polys.sqfreetools")?)?;
    factor::register_galoistools(&submodule(&polys, "galoistools", "sympy_extras_rs.polys.galoistools")?)?;
    let factortools_module: Bound<'_, PyModule> =
        submodule(&polys, "factortools", "sympy_extras_rs.polys.factortools")?;
    factor::register_factortools(&factortools_module)?;
    algebraic::register_ext_factortools(&factortools_module)?;
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
    algorithms::register_multifactor(&submodule(&polys, "multifactor", "sympy_extras_rs.polys.multifactor")?)?;
    lattice::register(&submodule(&polys, "lattice", "sympy_extras_rs.polys.lattice")?)?;
    complexroots::register(&submodule(
        &polys,
        "complexroots",
        "sympy_extras_rs.polys.complexroots",
    )?)?;
    let numberfields_module: Bound<'_, PyModule> =
        submodule(&polys, "numberfields", "sympy_extras_rs.polys.numberfields")?;
    numberfields::register_basis(&submodule(
        &numberfields_module,
        "basis",
        "sympy_extras_rs.polys.numberfields.basis",
    )?)?;
    qqbar::register(&submodule(
        &numberfields_module,
        "qqbar",
        "sympy_extras_rs.polys.numberfields.qqbar",
    )?)?;
    qqbar::register_recognize(&submodule(
        &numberfields_module,
        "recognize",
        "sympy_extras_rs.polys.numberfields.recognize",
    )?)?;
    qqbar::register_zerodim(&submodule(&polys, "zerodim", "sympy_extras_rs.polys.zerodim")?)?;
    algebraic::register_algebraicfactor(&submodule(
        &polys,
        "algebraicfactor",
        "sympy_extras_rs.polys.algebraicfactor",
    )?)?;
    galois::register_galoisfactor(&submodule(
        &polys,
        "galoisfactor",
        "sympy_extras_rs.polys.galoisfactor",
    )?)?;
    algorithms::register_gaussianfactor(&submodule(
        &polys,
        "gaussianfactor",
        "sympy_extras_rs.polys.gaussianfactor",
    )?)?;
    algorithms::register_f4(&submodule(&polys, "f4", "sympy_extras_rs.polys.f4")?)?;
    algorithms::register_buchberger(&submodule(&polys, "buchberger", "sympy_extras_rs.polys.buchberger")?)?;
    algorithms::register_fglm(&submodule(&polys, "fglm", "sympy_extras_rs.polys.fglm")?)?;
    dense::register_subresultants(&submodule(
        &polys,
        "subresultants",
        "sympy_extras_rs.polys.subresultants",
    )?)?;
    algorithms::register_modulargroebner(&submodule(
        &polys,
        "modulargroebner",
        "sympy_extras_rs.polys.modulargroebner",
    )?)?;
    algorithms::register_berlekampmassey(&submodule(
        &polys,
        "berlekampmassey",
        "sympy_extras_rs.polys.berlekampmassey",
    )?)?;
    algorithms::register_bigatti(&submodule(&polys, "bigatti", "sympy_extras_rs.polys.bigatti")?)?;
    algorithms::register_hilbert(&submodule(&polys, "hilbert", "sympy_extras_rs.polys.hilbert")?)?;
    algorithms::register_groebnerwalk(&submodule(
        &polys,
        "groebnerwalk",
        "sympy_extras_rs.polys.groebnerwalk",
    )?)?;
    rings::register_heuristicgcd(&submodule(
        &polys,
        "heuristicgcd",
        "sympy_extras_rs.polys.heuristicgcd",
    )?)?;
    Ok(())
}
