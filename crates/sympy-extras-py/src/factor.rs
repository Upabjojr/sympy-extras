//! `sympy_extras_rs.polys.galoistools` and `sympy_extras_rs.polys.
//! factortools`: the functions of the Python modules with the same names
//! and arguments.
//!
//! The functions which take a `random.Random` seed the Rust generator from
//! it (`rng.getrandbits(64)`), so the arguments are those of Python.

// the Python names (the argument `K`)
#![allow(non_snake_case)]

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::Int;
use sympy_extras_core::polys::domains::{FiniteField, IntegerRing, RationalField};
use sympy_extras_core::polys::{factortools as ft, galoistools as gt};
use sympy_extras_core::random::Rng;

use crate::convert::{err, extract_int, out_elem, out_factors, out_int, out_poly, poly, tuple, Bridge};
use crate::domains::{DomainKind, PyDomain};

fn field(K: &PyDomain) -> PyResult<FiniteField> {
    match K.kind {
        DomainKind::GF(p) => FiniteField::new(p).map_err(err),
        _ => Err(PyTypeError::new_err(format!("{:?} is not a finite field", K.kind))),
    }
}

fn integers(K: &PyDomain) -> PyResult<IntegerRing> {
    match K.kind {
        DomainKind::ZZ => Ok(IntegerRing),
        _ => Err(PyTypeError::new_err("the domain must be ZZ")),
    }
}

fn rationals(K: &PyDomain) -> PyResult<RationalField> {
    match K.kind {
        DomainKind::QQ => Ok(RationalField),
        _ => Err(PyTypeError::new_err("the domain must be QQ")),
    }
}

/// The Rust generator seeded from a Python `random.Random`.
fn rng_from(rng: &Bound<'_, PyAny>) -> PyResult<Rng> {
    Ok(Rng::new(rng.call_method1("getrandbits", (64,))?.extract::<u64>()?))
}

fn int_polys<'py>(py: Python<'py>, fs: &[Vec<Int>]) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(fs.len());
    for f in fs {
        items.push(out_poly(&IntegerRing, py, f)?);
    }
    Ok(PyList::new(py, items)?.into_any())
}

fn field_polys<'py>(k: &FiniteField, py: Python<'py>, fs: &[Vec<u64>]) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(fs.len());
    for f in fs {
        items.push(out_poly(k, py, f)?);
    }
    Ok(PyList::new(py, items)?.into_any())
}

fn int_poly(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Int>> {
    poly(&IntegerRing, ob)
}

fn int_poly_list(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Int>>> {
    let mut result: Vec<Vec<Int>> = Vec::new();
    for item in ob.try_iter()? {
        result.push(int_poly(&item?)?);
    }
    Ok(result)
}

// -- galoistools

#[pyfunction]
fn gf_pow_mod<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
    g: &Bound<'py, PyAny>,
    K: &PyDomain,
) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    let n: Int = extract_int(n)?;
    if sympy_extras_core::numbers::is_negative(&n) {
        return Err(pyo3::exceptions::PyValueError::new_err("negative exponent"));
    }
    out_poly(&k, py, &gt::gf_pow_mod(&poly(&k, f)?, &n, &poly(&k, g)?, &k))
}

#[pyfunction]
fn gf_frobenius_monomial_base<'py>(
    py: Python<'py>,
    g: &Bound<'py, PyAny>,
    K: &PyDomain,
) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    field_polys(&k, py, &gt::gf_frobenius_monomial_base(&poly(&k, g)?, &k))
}

#[pyfunction]
fn gf_frobenius_map<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    g: &Bound<'py, PyAny>,
    base: &Bound<'py, PyAny>,
    K: &PyDomain,
) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    let mut rows: Vec<Vec<u64>> = Vec::new();
    for item in base.try_iter()? {
        rows.push(poly(&k, &item?)?);
    }
    out_poly(&k, py, &gt::gf_frobenius_map(&poly(&k, f)?, &poly(&k, g)?, &rows, &k))
}

#[pyfunction]
fn gf_ddf<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    out_factors(&k, py, &gt::gf_ddf(&poly(&k, f)?, &k))
}

#[pyfunction]
fn gf_edf<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    d: usize,
    K: &PyDomain,
    rng: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    let mut generator: Rng = rng_from(rng)?;
    field_polys(&k, py, &gt::gf_edf(&poly(&k, f)?, d, &k, &mut generator))
}

#[pyfunction]
fn gf_factor_sqf<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    let (lc, factors) = gt::gf_factor_sqf(&poly(&k, f)?, &k);
    tuple(py, vec![out_elem(&k, py, &lc)?, field_polys(&k, py, &factors)?])
}

#[pyfunction]
fn gf_factor<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    let (lc, factors) = gt::gf_factor(&poly(&k, f)?, &k);
    tuple(py, vec![out_elem(&k, py, &lc)?, out_factors(&k, py, &factors)?])
}

#[pyfunction]
fn gf_irreducible_p(f: &Bound<'_, PyAny>, K: &PyDomain) -> PyResult<bool> {
    let k: FiniteField = field(K)?;
    Ok(gt::gf_irreducible_p(&poly(&k, f)?, &k))
}

#[pyfunction]
fn gf_random<'py>(py: Python<'py>, n: usize, K: &PyDomain, rng: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    let mut generator: Rng = rng_from(rng)?;
    out_poly(&k, py, &gt::gf_random(n, &k, &mut generator))
}

#[pyfunction]
fn gf_irreducible<'py>(
    py: Python<'py>,
    n: usize,
    K: &PyDomain,
    rng: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let k: FiniteField = field(K)?;
    let mut generator: Rng = rng_from(rng)?;
    out_poly(&k, py, &gt::gf_irreducible(n, &k, &mut generator))
}

// -- factortools

#[pyfunction]
fn dup_zz_mignotte_bound<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    let k: IntegerRing = integers(K)?;
    let f: Vec<Int> = int_poly(f)?;
    if f.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the zero polynomial has no Mignotte bound",
        ));
    }
    out_int(py, &ft::dup_zz_mignotte_bound(&f, &k))
}

#[pyfunction]
fn dup_zz_hensel_step<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyAny>,
    f: &Bound<'py, PyAny>,
    g: &Bound<'py, PyAny>,
    h: &Bound<'py, PyAny>,
    s: &Bound<'py, PyAny>,
    t: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (g1, h1, s1, t1) = ft::dup_zz_hensel_step(
        &extract_int(m)?,
        &int_poly(f)?,
        &int_poly(g)?,
        &int_poly(h)?,
        &int_poly(s)?,
        &int_poly(t)?,
    );
    let z: &IntegerRing = &IntegerRing;
    tuple(
        py,
        vec![
            out_poly(z, py, &g1)?,
            out_poly(z, py, &h1)?,
            out_poly(z, py, &s1)?,
            out_poly(z, py, &t1)?,
        ],
    )
}

#[pyfunction]
fn dup_zz_hensel_lift<'py>(
    py: Python<'py>,
    p: u64,
    f: &Bound<'py, PyAny>,
    factors: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (lifted, m) =
        ft::dup_zz_hensel_lift(p, &int_poly(f)?, &int_poly_list(factors)?, &extract_int(target)?).map_err(err)?;
    tuple(py, vec![int_polys(py, &lifted)?, out_int(py, &m)?])
}

#[pyfunction]
fn dup_zz_zassenhaus<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    int_polys(py, &ft::dup_zz_zassenhaus(&int_poly(f)?).map_err(err)?)
}

#[pyfunction]
fn dup_zz_cyclotomic_poly<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("n must be positive"));
    }
    out_poly(&IntegerRing, py, &ft::dup_zz_cyclotomic_poly(n))
}

#[pyfunction]
fn dup_zz_cyclotomic_factor<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    match ft::dup_zz_cyclotomic_factor(&int_poly(f)?) {
        Some(factors) => Ok(Some(int_polys(py, &factors)?)),
        None => Ok(None),
    }
}

#[pyfunction]
fn dup_zz_factor_sqf<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    let k: IntegerRing = integers(K)?;
    let (content, factors) = ft::dup_zz_factor_sqf(&int_poly(f)?, &k).map_err(err)?;
    tuple(py, vec![out_int(py, &content)?, int_polys(py, &factors)?])
}

#[pyfunction]
fn dup_zz_factor<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    let k: IntegerRing = integers(K)?;
    let (content, factors) = ft::dup_zz_factor(&int_poly(f)?, &k).map_err(err)?;
    tuple(py, vec![out_int(py, &content)?, out_factors(&k, py, &factors)?])
}

#[pyfunction]
fn dup_qq_factor<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    let k: RationalField = rationals(K)?;
    let (c, factors) = ft::dup_qq_factor(&poly(&k, f)?, &k).map_err(err)?;
    tuple(py, vec![k.to_py(py, &c)?, out_factors(&k, py, &factors)?])
}

#[pyfunction]
fn dup_zz_irreducible_p(f: &Bound<'_, PyAny>, K: &PyDomain) -> PyResult<bool> {
    let k: IntegerRing = integers(K)?;
    ft::dup_zz_irreducible_p(&int_poly(f)?, &k).map_err(err)
}

pub fn register_galoistools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gf_pow_mod, m)?)?;
    m.add_function(wrap_pyfunction!(gf_frobenius_monomial_base, m)?)?;
    m.add_function(wrap_pyfunction!(gf_frobenius_map, m)?)?;
    m.add_function(wrap_pyfunction!(gf_ddf, m)?)?;
    m.add_function(wrap_pyfunction!(gf_edf, m)?)?;
    m.add_function(wrap_pyfunction!(gf_factor_sqf, m)?)?;
    m.add_function(wrap_pyfunction!(gf_factor, m)?)?;
    m.add_function(wrap_pyfunction!(gf_irreducible_p, m)?)?;
    m.add_function(wrap_pyfunction!(gf_random, m)?)?;
    m.add_function(wrap_pyfunction!(gf_irreducible, m)?)?;
    Ok(())
}

pub fn register_factortools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dup_zz_mignotte_bound, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_hensel_step, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_hensel_lift, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_zassenhaus, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_cyclotomic_poly, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_cyclotomic_factor, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_factor_sqf, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_factor, m)?)?;
    m.add_function(wrap_pyfunction!(dup_qq_factor, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zz_irreducible_p, m)?)?;
    Ok(())
}
