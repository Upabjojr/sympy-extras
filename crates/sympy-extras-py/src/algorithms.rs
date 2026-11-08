//! `sympy_extras_rs.polys.rootisolation` and `sympy_extras_rs.polys.f4`:
//! the functions of the Python modules with the same names and arguments.

#![allow(non_snake_case)]

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use sympy_extras_core::numbers::Rat;
use sympy_extras_core::polys::domains::{Domain, RationalField, SparseRow};
use sympy_extras_core::polys::rings::PolyElement;
use sympy_extras_core::polys::{f4 as groebner, rootisolation as ri};

use crate::convert::{err, out_polys, out_usize, poly, tuple, Bridge};
use crate::domains::PyDomain;
use crate::rings::{ElemKind, PyPolyElement, PyPolyRing, RingKind};
use crate::with_domain;

fn rational(ob: &Bound<'_, PyAny>) -> PyResult<Rat> {
    RationalField.extract(ob)
}

fn optional_rational(ob: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Rat>> {
    match ob {
        Some(value) if !value.is_none() => Ok(Some(rational(value)?)),
        _ => Ok(None),
    }
}

fn rational_poly(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Rat>> {
    poly(&RationalField, ob)
}

fn out_interval<'py>(py: Python<'py>, interval: &(Rat, Rat)) -> PyResult<Bound<'py, PyAny>> {
    tuple(
        py,
        vec![
            RationalField.to_py(py, &interval.0)?,
            RationalField.to_py(py, &interval.1)?,
        ],
    )
}

fn out_intervals<'py>(py: Python<'py>, intervals: &[(Rat, Rat)]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = intervals.iter().map(|i| out_interval(py, i)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

pub fn out_roots<'py>(py: Python<'py>, roots: &[((Rat, Rat), usize)]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = roots
        .iter()
        .map(|(interval, k)| tuple(py, vec![out_interval(py, interval)?, out_usize(py, *k)?]))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

// -- rootisolation

#[pyfunction]
fn dup_cauchy_bound<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    RationalField.to_py(py, &ri::dup_cauchy_bound(&rational_poly(f)?))
}

#[pyfunction]
fn dup_fujiwara_bound<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    RationalField.to_py(py, &ri::dup_fujiwara_bound(&rational_poly(f)?))
}

#[pyfunction]
fn dup_sign_at(f: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<i32> {
    Ok(ri::dup_sign_at(&rational_poly(f)?, &rational(a)?))
}

#[pyfunction]
fn dup_monic_signed<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    crate::convert::out_poly(&RationalField, py, &ri::dup_monic_signed(&rational_poly(f)?))
}

#[pyfunction]
fn dup_sturm<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_polys(&RationalField, py, &ri::dup_sturm(&rational_poly(f)?))
}

#[pyfunction]
#[pyo3(signature = (f, inf = None, sup = None))]
fn dup_count_real_roots(
    f: &Bound<'_, PyAny>,
    inf: Option<&Bound<'_, PyAny>>,
    sup: Option<&Bound<'_, PyAny>>,
) -> PyResult<i64> {
    let (inf, sup) = (optional_rational(inf)?, optional_rational(sup)?);
    Ok(ri::dup_count_real_roots(&rational_poly(f)?, inf.as_ref(), sup.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (f, eps = None))]
fn dup_isolate_real_roots_sqf<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    eps: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let eps: Option<Rat> = optional_rational(eps)?;
    out_intervals(py, &ri::dup_isolate_real_roots_sqf(&rational_poly(f)?, eps.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (f, eps = None))]
fn dup_isolate_real_roots<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    eps: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let eps: Option<Rat> = optional_rational(eps)?;
    out_roots(py, &ri::dup_isolate_real_roots(&rational_poly(f)?, eps.as_ref()))
}

#[pyfunction]
fn dup_refine_real_root<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    eps: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let interval = ri::dup_refine_real_root(&rational_poly(f)?, rational(a)?, rational(b)?, &rational(eps)?);
    out_interval(py, &interval)
}

// -- f4

macro_rules! same_kind {
    ($items:expr, $variant:ident) => {{
        let mut out = Vec::new();
        for item in $items {
            match item {
                ElemKind::$variant(f) => out.push(f.clone()),
                _ => return Err(PyTypeError::new_err("polynomials of different rings")),
            }
        }
        out
    }};
}

fn elements(ob: &Bound<'_, PyAny>) -> PyResult<Vec<ElemKind>> {
    let mut result: Vec<ElemKind> = Vec::new();
    for item in ob.try_iter()? {
        result.push(item?.downcast::<PyPolyElement>()?.get().inner.clone());
    }
    Ok(result)
}

fn out_elements<'py>(py: Python<'py>, items: Vec<ElemKind>) -> PyResult<Bound<'py, PyAny>> {
    let objects: Vec<Bound<'py, PyAny>> = items
        .into_iter()
        .map(|inner| Ok(Bound::new(py, PyPolyElement { inner })?.into_any()))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objects)?.into_any())
}

fn wrap<K: Domain>(fs: Vec<PolyElement<K>>, kind: fn(PolyElement<K>) -> ElemKind) -> Vec<ElemKind> {
    fs.into_iter().map(kind).collect()
}

#[pyfunction]
fn f4<'py>(py: Python<'py>, polys: &Bound<'py, PyAny>, ring: &PyPolyRing) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<ElemKind> = elements(polys)?;
    let result: Vec<ElemKind> = match &ring.inner {
        // the domain is checked first, as in Python
        RingKind::ZZ(r) => wrap(groebner::f4(&[], r).map_err(err)?, ElemKind::ZZ),
        RingKind::QQ(r) => wrap(groebner::f4(&same_kind!(&items, QQ), r).map_err(err)?, ElemKind::QQ),
        RingKind::GF(r) => wrap(groebner::f4(&same_kind!(&items, GF), r).map_err(err)?, ElemKind::GF),
    };
    out_elements(py, result)
}

#[pyfunction]
fn reduced_basis<'py>(py: Python<'py>, basis: &Bound<'py, PyAny>, ring: &PyPolyRing) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<ElemKind> = elements(basis)?;
    let result: Vec<ElemKind> = match &ring.inner {
        RingKind::ZZ(_) => return Err(PyTypeError::new_err("reduced_basis needs a field")),
        RingKind::QQ(r) => wrap(groebner::reduced_basis(&same_kind!(&items, QQ), r), ElemKind::QQ),
        RingKind::GF(r) => wrap(groebner::reduced_basis(&same_kind!(&items, GF), r), ElemKind::GF),
    };
    out_elements(py, result)
}

#[pyfunction]
fn is_graded(ring: &PyPolyRing) -> bool {
    match &ring.inner {
        RingKind::ZZ(r) => groebner::is_graded(r),
        RingKind::QQ(r) => groebner::is_graded(r),
        RingKind::GF(r) => groebner::is_graded(r),
    }
}

fn sparse_rows<K: Bridge>(k: &K, ob: &Bound<'_, PyAny>) -> PyResult<Vec<SparseRow<K::Elem>>> {
    let mut rows: Vec<SparseRow<K::Elem>> = Vec::new();
    for item in ob.try_iter()? {
        let item = item?;
        let dict: &Bound<'_, PyDict> = item.downcast::<PyDict>()?;
        let mut row: SparseRow<K::Elem> = SparseRow::new();
        for (c, v) in dict.iter() {
            row.insert(c.extract::<usize>()?, k.extract(&v)?);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn out_sparse_rows<'py, K: Bridge>(k: &K, py: Python<'py>, rows: &[SparseRow<K::Elem>]) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
    for row in rows {
        let dict: Bound<'py, PyDict> = PyDict::new(py);
        for (c, v) in row {
            dict.set_item(*c, k.to_py(py, v)?)?;
        }
        items.push(dict.into_any());
    }
    Ok(PyList::new(py, items)?.into_any())
}

/// `row_echelon(rows, K)` over a field.
#[pyfunction]
fn row_echelon<'py>(py: Python<'py>, rows: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    if !K.is_base() {
        return Err(PyTypeError::new_err("row_echelon over QQ or GF(p)"));
    }
    with_domain!(K, k => {
        if !k.is_field() {
            return Err(PyTypeError::new_err("row_echelon needs a field"));
        }
        out_sparse_rows(k, py, &groebner::row_echelon(&sparse_rows(k, rows)?, k))
    })
}

pub fn register_rootisolation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dup_cauchy_bound, m)?)?;
    m.add_function(wrap_pyfunction!(dup_fujiwara_bound, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sign_at, m)?)?;
    m.add_function(wrap_pyfunction!(dup_monic_signed, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sturm, m)?)?;
    m.add_function(wrap_pyfunction!(dup_count_real_roots, m)?)?;
    m.add_function(wrap_pyfunction!(dup_isolate_real_roots_sqf, m)?)?;
    m.add_function(wrap_pyfunction!(dup_isolate_real_roots, m)?)?;
    m.add_function(wrap_pyfunction!(dup_refine_real_root, m)?)?;
    Ok(())
}

pub fn register_f4(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SELECTION", groebner::SELECTION)?;
    m.add_function(wrap_pyfunction!(f4, m)?)?;
    m.add_function(wrap_pyfunction!(reduced_basis, m)?)?;
    m.add_function(wrap_pyfunction!(is_graded, m)?)?;
    m.add_function(wrap_pyfunction!(row_echelon, m)?)?;
    Ok(())
}
