//! `sympy_extras_rs.polys.rootisolation` and `sympy_extras_rs.polys.f4`:
//! the functions of the Python modules with the same names and arguments.

#![allow(non_snake_case)]

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use sympy_extras_core::numbers::Int;
use sympy_extras_core::numbers::Rat;
use sympy_extras_core::polys::domains::{Domain, RationalField, SparseRow};
use sympy_extras_core::polys::domains::{GaussianRational, GaussianRationalField, IntegerRing};
use sympy_extras_core::polys::groebnerwalk::{self as walk, WalkLift};
use sympy_extras_core::polys::modulargroebner as mg;
use sympy_extras_core::polys::rings::PolyElement;
use sympy_extras_core::polys::rings::{Monomial, MonomialOrder};
use sympy_extras_core::polys::{
    berlekampmassey as bm, bigatti, f4 as groebner, fglm as fglm_mod, gaussianfactor as gf, hilbert, multifactor as mf,
    rootisolation as ri, stronggroebner as sg,
};

use crate::convert::{err, extract_int, out_int, out_poly, out_polys, out_usize, poly, tuple, Bridge};
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
        RingKind::GaussianQQ(r) => wrap(
            groebner::f4(&same_kind!(&items, GaussianQQ), r).map_err(err)?,
            ElemKind::GaussianQQ,
        ),
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
        RingKind::GaussianQQ(r) => wrap(
            groebner::reduced_basis(&same_kind!(&items, GaussianQQ), r),
            ElemKind::GaussianQQ,
        ),
    };
    out_elements(py, result)
}

#[pyfunction]
fn is_graded(ring: &PyPolyRing) -> bool {
    match &ring.inner {
        RingKind::ZZ(r) => groebner::is_graded(r),
        RingKind::QQ(r) => groebner::is_graded(r),
        RingKind::GF(r) => groebner::is_graded(r),
        RingKind::GaussianQQ(r) => groebner::is_graded(r),
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

// -- stronggroebner

fn zz_elem(f: &PyPolyElement) -> PyResult<PolyElement<IntegerRing>> {
    match &f.inner {
        ElemKind::ZZ(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("polynomials over ZZ are expected")),
    }
}

fn zz_elems(ob: &Bound<'_, PyAny>) -> PyResult<Vec<PolyElement<IntegerRing>>> {
    let items: Vec<ElemKind> = elements(ob)?;
    Ok(same_kind!(&items, ZZ))
}

fn zz_out<'py>(py: Python<'py>, f: PolyElement<IntegerRing>) -> PyResult<Bound<'py, PyAny>> {
    Ok(Bound::new(py, PyPolyElement { inner: ElemKind::ZZ(f) })?.into_any())
}

/// `strong_groebner(polys, ring)`: the generators are polynomials over `ZZ`
/// or integers.
#[pyfunction]
fn strong_groebner<'py>(py: Python<'py>, polys: &Bound<'py, PyAny>, ring: &PyPolyRing) -> PyResult<Bound<'py, PyAny>> {
    let r = match &ring.inner {
        RingKind::ZZ(r) => r,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "strong Gröbner bases are computed over ZZ, the ring is over {}",
                crate::rings::domain_name(&ring.inner)
            )))
        }
    };
    let mut generators: Vec<PolyElement<IntegerRing>> = Vec::new();
    for item in polys.try_iter()? {
        let item = item?;
        if let Ok(f) = item.downcast::<PyPolyElement>() {
            generators.push(zz_elem(f.get())?);
        } else {
            generators.push(r.ground(crate::convert::extract_int(&item)?));
        }
    }
    let basis = sg::strong_groebner(&generators, r).map_err(err)?;
    out_elements(py, basis.into_iter().map(ElemKind::ZZ).collect())
}

#[pyfunction]
fn strong_reduce<'py>(py: Python<'py>, f: &PyPolyElement, basis: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    zz_out(py, sg::strong_reduce(&zz_elem(f)?, &zz_elems(basis)?))
}

#[pyfunction]
fn is_strong_groebner(basis: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(sg::is_strong_groebner(&zz_elems(basis)?))
}

#[pyfunction]
fn s_polynomial_zz<'py>(py: Python<'py>, f: &PyPolyElement, g: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    zz_out(py, sg::s_polynomial_zz(&zz_elem(f)?, &zz_elem(g)?))
}

#[pyfunction]
fn g_polynomial_zz<'py>(py: Python<'py>, f: &PyPolyElement, g: &PyPolyElement) -> PyResult<Option<Bound<'py, PyAny>>> {
    match sg::g_polynomial_zz(&zz_elem(f)?, &zz_elem(g)?) {
        Some(h) => Ok(Some(zz_out(py, h)?)),
        None => Ok(None),
    }
}

pub fn register_stronggroebner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(strong_groebner, m)?)?;
    m.add_function(wrap_pyfunction!(strong_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(is_strong_groebner, m)?)?;
    m.add_function(wrap_pyfunction!(s_polynomial_zz, m)?)?;
    m.add_function(wrap_pyfunction!(g_polynomial_zz, m)?)?;
    Ok(())
}

// -- multifactor

fn out_int_factors<'py>(
    py: Python<'py>,
    factors: Vec<(PolyElement<IntegerRing>, usize)>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
    for (g, k) in factors {
        items.push(tuple(py, vec![zz_out(py, g)?, out_usize(py, k)?])?);
    }
    Ok(PyList::new(py, items)?.into_any())
}

#[pyfunction]
fn zz_factor<'py>(py: Python<'py>, f: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    let (c, factors) = mf::zz_factor(&zz_elem(f)?).map_err(err)?;
    tuple(
        py,
        vec![crate::convert::out_int(py, &c)?, out_int_factors(py, factors)?],
    )
}

/// `zz_wang(f, mod=None, seed=0)`.
#[pyfunction]
#[pyo3(signature = (f, r#mod = None, seed = 0))]
fn zz_wang<'py>(py: Python<'py>, f: &PyPolyElement, r#mod: Option<i64>, seed: u64) -> PyResult<Bound<'py, PyAny>> {
    let factors = mf::zz_wang(&zz_elem(f)?, r#mod, seed).map_err(err)?;
    out_elements(py, factors.into_iter().map(ElemKind::ZZ).collect())
}

#[pyfunction]
fn sqf_part<'py>(py: Python<'py>, f: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    zz_out(py, mf::sqf_part(&zz_elem(f)?).map_err(err)?)
}

#[pyfunction]
fn trial_division<'py>(py: Python<'py>, f: &PyPolyElement, factors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_int_factors(py, mf::trial_division(&zz_elem(f)?, &zz_elems(factors)?).map_err(err)?)
}

fn dense_key_to_py<'py>(py: Python<'py>, key: &mf::DenseKey) -> PyResult<Bound<'py, PyAny>> {
    match key {
        mf::DenseKey::Int(n) => crate::convert::out_int(py, n),
        mf::DenseKey::Tuple(items) => {
            let objects: Vec<Bound<'py, PyAny>> =
                items.iter().map(|k| dense_key_to_py(py, k)).collect::<PyResult<_>>()?;
            tuple(py, objects)
        }
    }
}

#[pyfunction]
fn factor_sort_key<'py>(py: Python<'py>, f: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    dense_key_to_py(py, &mf::factor_sort_key(&zz_elem(f)?))
}

pub fn register_multifactor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zz_factor, m)?)?;
    m.add_function(wrap_pyfunction!(zz_wang, m)?)?;
    m.add_function(wrap_pyfunction!(sqf_part, m)?)?;
    m.add_function(wrap_pyfunction!(trial_division, m)?)?;
    m.add_function(wrap_pyfunction!(factor_sort_key, m)?)?;
    Ok(())
}

// -- fglm

#[pyfunction]
fn is_zero_dimensional(basis: &Bound<'_, PyAny>, ring: &PyPolyRing) -> PyResult<bool> {
    let items: Vec<ElemKind> = elements(basis)?;
    Ok(match &ring.inner {
        RingKind::ZZ(r) => fglm_mod::is_zero_dimensional(&same_kind!(&items, ZZ), r),
        RingKind::QQ(r) => fglm_mod::is_zero_dimensional(&same_kind!(&items, QQ), r),
        RingKind::GF(r) => fglm_mod::is_zero_dimensional(&same_kind!(&items, GF), r),
        RingKind::GaussianQQ(r) => fglm_mod::is_zero_dimensional(&same_kind!(&items, GaussianQQ), r),
    })
}

#[pyfunction]
fn fglm<'py>(
    py: Python<'py>,
    basis: &Bound<'py, PyAny>,
    ring: &PyPolyRing,
    order: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<ElemKind> = elements(basis)?;
    let order: MonomialOrder = crate::rings::order_from(order)?;
    let result: Vec<ElemKind> = match &ring.inner {
        RingKind::ZZ(r) => wrap(
            fglm_mod::fglm(&same_kind!(&items, ZZ), r, &order).map_err(err)?,
            ElemKind::ZZ,
        ),
        RingKind::QQ(r) => wrap(
            fglm_mod::fglm(&same_kind!(&items, QQ), r, &order).map_err(err)?,
            ElemKind::QQ,
        ),
        RingKind::GF(r) => wrap(
            fglm_mod::fglm(&same_kind!(&items, GF), r, &order).map_err(err)?,
            ElemKind::GF,
        ),
        RingKind::GaussianQQ(r) => wrap(
            fglm_mod::fglm(&same_kind!(&items, GaussianQQ), r, &order).map_err(err)?,
            ElemKind::GaussianQQ,
        ),
    };
    out_elements(py, result)
}

#[pyfunction]
fn groebner_by_fglm<'py>(py: Python<'py>, polys: &Bound<'py, PyAny>, ring: &PyPolyRing) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<ElemKind> = elements(polys)?;
    let result: Vec<ElemKind> = match &ring.inner {
        RingKind::ZZ(r) => wrap(
            fglm_mod::groebner_by_fglm(&same_kind!(&items, ZZ), r).map_err(err)?,
            ElemKind::ZZ,
        ),
        RingKind::QQ(r) => wrap(
            fglm_mod::groebner_by_fglm(&same_kind!(&items, QQ), r).map_err(err)?,
            ElemKind::QQ,
        ),
        RingKind::GF(r) => wrap(
            fglm_mod::groebner_by_fglm(&same_kind!(&items, GF), r).map_err(err)?,
            ElemKind::GF,
        ),
        RingKind::GaussianQQ(r) => wrap(
            fglm_mod::groebner_by_fglm(&same_kind!(&items, GaussianQQ), r).map_err(err)?,
            ElemKind::GaussianQQ,
        ),
    };
    out_elements(py, result)
}

pub fn register_fglm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_zero_dimensional, m)?)?;
    m.add_function(wrap_pyfunction!(fglm, m)?)?;
    m.add_function(wrap_pyfunction!(groebner_by_fglm, m)?)?;
    Ok(())
}

// -- berlekampmassey

#[pyfunction]
fn berlekamp_massey<'py>(py: Python<'py>, sequence: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    with_domain!(K, k => out_poly(k, py, &bm::berlekamp_massey(&poly(k, sequence)?, k).map_err(err)?))
}

#[pyfunction]
fn satisfies_recurrence(sequence: &Bound<'_, PyAny>, polynomial: &Bound<'_, PyAny>, K: &PyDomain) -> PyResult<bool> {
    with_domain!(K, k => Ok(bm::satisfies_recurrence(&poly(k, sequence)?, &poly(k, polynomial)?, k)))
}

#[pyfunction]
fn extend_sequence<'py>(
    py: Python<'py>,
    sequence: &Bound<'py, PyAny>,
    K: &PyDomain,
    count: usize,
) -> PyResult<Bound<'py, PyAny>> {
    with_domain!(K, k => out_poly(k, py, &bm::extend_sequence(&poly(k, sequence)?, k, count).map_err(err)?))
}

pub fn register_berlekampmassey(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(berlekamp_massey, m)?)?;
    m.add_function(wrap_pyfunction!(satisfies_recurrence, m)?)?;
    m.add_function(wrap_pyfunction!(extend_sequence, m)?)?;
    Ok(())
}

// -- gaussianfactor

fn qq_i_elem(f: &PyPolyElement) -> PyResult<PolyElement<GaussianRationalField>> {
    match &f.inner {
        ElemKind::GaussianQQ(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("polynomials over QQ_I are expected")),
    }
}

fn gaussian_result<'py>(py: Python<'py>, result: (GaussianRational, gf::GaussFactors)) -> PyResult<Bound<'py, PyAny>> {
    let (c, factors) = result;
    let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
    for (g, k) in factors {
        let g: Bound<'py, PyAny> = Bound::new(
            py,
            PyPolyElement {
                inner: ElemKind::GaussianQQ(g),
            },
        )?
        .into_any();
        items.push(tuple(py, vec![g, out_usize(py, k)?])?);
    }
    tuple(
        py,
        vec![GaussianRationalField.to_py(py, &c)?, PyList::new(py, items)?.into_any()],
    )
}

fn gauss_key_to_py<'py>(py: Python<'py>, key: &gf::GaussKey) -> PyResult<Bound<'py, PyAny>> {
    match key {
        gf::GaussKey::Coefficient(y, x) => tuple(py, vec![RationalField.to_py(py, y)?, RationalField.to_py(py, x)?]),
        gf::GaussKey::Tuple(items) => {
            let objects: Vec<Bound<'py, PyAny>> =
                items.iter().map(|k| gauss_key_to_py(py, k)).collect::<PyResult<_>>()?;
            tuple(py, objects)
        }
    }
}

#[pyfunction]
fn gaussian_ext_factor<'py>(py: Python<'py>, f: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    gaussian_result(py, gf::gaussian_ext_factor(&qq_i_elem(f)?).map_err(err)?)
}

#[pyfunction]
fn gaussian_factor_list<'py>(py: Python<'py>, f: &PyPolyElement, integral: bool) -> PyResult<Bound<'py, PyAny>> {
    gaussian_result(py, gf::gaussian_factor_list(&qq_i_elem(f)?, integral).map_err(err)?)
}

/// `gaussian_sort_key((f, k))`.
#[pyfunction]
fn gaussian_sort_key<'py>(py: Python<'py>, item: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let f: PolyElement<GaussianRationalField> = qq_i_elem(item.get_item(0)?.downcast::<PyPolyElement>()?.get())?;
    let k: usize = item.get_item(1)?.extract::<usize>()?;
    let (length, k, key) = gf::gaussian_sort_key(&(f, k));
    tuple(
        py,
        vec![
            length.into_pyobject(py)?.into_any(),
            out_usize(py, k)?,
            gauss_key_to_py(py, &key)?,
        ],
    )
}

pub fn register_gaussianfactor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussian_ext_factor, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_factor_list, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_sort_key, m)?)?;
    Ok(())
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

// -- hilbert

fn monomials(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Monomial>> {
    let mut result: Vec<Monomial> = Vec::new();
    for item in ob.try_iter()? {
        result.push(item?.extract::<Vec<u32>>()?);
    }
    Ok(result)
}

fn out_monomials<'py>(py: Python<'py>, items: &[Monomial]) -> PyResult<Bound<'py, PyAny>> {
    let tuples: Vec<Bound<'py, PyAny>> = items
        .iter()
        .map(|m| Ok(pyo3::types::PyTuple::new(py, m)?.into_any()))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, tuples)?.into_any())
}

#[pyfunction]
fn minimal_generators<'py>(py: Python<'py>, monoms: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_monomials(py, &hilbert::minimal_generators(&monomials(monoms)?))
}

/// The numerator as a dense list over `ZZ` (Python: a `Poly` in `t`).
#[pyfunction]
fn hilbert_numerator<'py>(py: Python<'py>, monoms: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_poly(&IntegerRing, py, &hilbert::hilbert_numerator(&monomials(monoms)?))
}

/// `(numerator, exponent)` of `numerator/(1 - t)**exponent` (Python: the
/// expression in `t`).
#[pyfunction]
fn hilbert_series<'py>(py: Python<'py>, monoms: &Bound<'py, PyAny>, variables: usize) -> PyResult<Bound<'py, PyAny>> {
    let series: hilbert::HilbertSeries = hilbert::hilbert_series(&monomials(monoms)?, variables);
    tuple(
        py,
        vec![
            out_poly(&IntegerRing, py, &series.numerator)?,
            series.exponent.into_pyobject(py)?.into_any(),
        ],
    )
}

/// The coefficients lowest degree first, without trailing zeros.
#[pyfunction]
fn hilbert_numerator_coefficients<'py>(py: Python<'py>, monoms: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_poly(
        &IntegerRing,
        py,
        &bigatti::hilbert_numerator_coefficients(&monomials(monoms)?),
    )
}

pub fn register_bigatti(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimal_generators, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert_numerator_coefficients, m)?)?;
    Ok(())
}

pub fn register_hilbert(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimal_generators, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert_numerator, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert_series, m)?)?;
    Ok(())
}

// -- groebnerwalk

fn weight_vector(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Rat>> {
    let mut result: Vec<Rat> = Vec::new();
    for item in ob.try_iter()? {
        result.push(rational(&item?)?);
    }
    Ok(result)
}

#[pyfunction]
fn initial_form(f: &PyPolyElement, weights: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
    let w: Vec<Rat> = weight_vector(weights)?;
    let inner: ElemKind = match &f.inner {
        ElemKind::ZZ(p) => ElemKind::ZZ(walk::initial_form(p, &w)),
        ElemKind::QQ(p) => ElemKind::QQ(walk::initial_form(p, &w)),
        ElemKind::GF(p) => ElemKind::GF(walk::initial_form(p, &w)),
        ElemKind::GaussianQQ(p) => ElemKind::GaussianQQ(walk::initial_form(p, &w)),
    };
    Ok(PyPolyElement { inner })
}

fn representations<'py, K: Domain>(
    py: Python<'py>,
    result: (Vec<PolyElement<K>>, Vec<Vec<PolyElement<K>>>),
    kind: fn(PolyElement<K>) -> ElemKind,
) -> PyResult<Bound<'py, PyAny>> {
    let (h, c) = result;
    let rows: Vec<Bound<'py, PyAny>> = c
        .into_iter()
        .map(|row| out_elements(py, wrap(row, kind)))
        .collect::<PyResult<_>>()?;
    tuple(
        py,
        vec![out_elements(py, wrap(h, kind))?, PyList::new(py, rows)?.into_any()],
    )
}

#[pyfunction]
fn extended_groebner<'py>(py: Python<'py>, F: &Bound<'py, PyAny>, ring: &PyPolyRing) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<ElemKind> = elements(F)?;
    match &ring.inner {
        RingKind::ZZ(r) => representations(
            py,
            walk::extended_groebner(&same_kind!(&items, ZZ), r).map_err(err)?,
            ElemKind::ZZ,
        ),
        RingKind::QQ(r) => representations(
            py,
            walk::extended_groebner(&same_kind!(&items, QQ), r).map_err(err)?,
            ElemKind::QQ,
        ),
        RingKind::GF(r) => representations(
            py,
            walk::extended_groebner(&same_kind!(&items, GF), r).map_err(err)?,
            ElemKind::GF,
        ),
        RingKind::GaussianQQ(r) => representations(
            py,
            walk::extended_groebner(&same_kind!(&items, GaussianQQ), r).map_err(err)?,
            ElemKind::GaussianQQ,
        ),
    }
}

#[pyfunction]
fn is_groebner(G: &Bound<'_, PyAny>, ring: &PyPolyRing) -> PyResult<bool> {
    let items: Vec<ElemKind> = elements(G)?;
    match &ring.inner {
        RingKind::ZZ(r) => walk::is_groebner(&same_kind!(&items, ZZ), r).map_err(err),
        RingKind::QQ(r) => walk::is_groebner(&same_kind!(&items, QQ), r).map_err(err),
        RingKind::GF(r) => walk::is_groebner(&same_kind!(&items, GF), r).map_err(err),
        RingKind::GaussianQQ(r) => walk::is_groebner(&same_kind!(&items, GaussianQQ), r).map_err(err),
    }
}

/// `groebner_walk(G, ring, target, check=False, lift='normal_form')`; the
/// domain must be a field (SymPy also walks over `ZZ` through `QQ`). A
/// failed check raises `RuntimeError`, as in Python.
#[pyfunction]
#[pyo3(signature = (G, ring, target, check = false, lift = "normal_form"))]
fn groebner_walk<'py>(
    py: Python<'py>,
    G: &Bound<'py, PyAny>,
    ring: &PyPolyRing,
    target: &Bound<'py, PyAny>,
    check: bool,
    lift: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let target: MonomialOrder = crate::rings::order_from(target)?;
    let lift: WalkLift = match lift {
        "normal_form" => WalkLift::NormalForm,
        "cofactors" => WalkLift::Cofactors,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lift must be 'normal_form' or 'cofactors'",
            ))
        }
    };
    let items: Vec<ElemKind> = elements(G)?;
    let failed = |e: sympy_extras_core::error::PolyError| -> PyErr {
        let message: String = e.to_string();
        if message.contains("not a Groebner basis") {
            pyo3::exceptions::PyRuntimeError::new_err(message)
        } else {
            err(e)
        }
    };
    let result: Vec<ElemKind> = match &ring.inner {
        RingKind::ZZ(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the Groebner walk needs a field",
            ))
        }
        RingKind::QQ(r) => wrap(
            walk::groebner_walk(&same_kind!(&items, QQ), r, &target, check, lift).map_err(failed)?,
            ElemKind::QQ,
        ),
        RingKind::GF(r) => wrap(
            walk::groebner_walk(&same_kind!(&items, GF), r, &target, check, lift).map_err(failed)?,
            ElemKind::GF,
        ),
        RingKind::GaussianQQ(r) => wrap(
            walk::groebner_walk(&same_kind!(&items, GaussianQQ), r, &target, check, lift).map_err(failed)?,
            ElemKind::GaussianQQ,
        ),
    };
    out_elements(py, result)
}

pub fn register_groebnerwalk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initial_form, m)?)?;
    m.add_function(wrap_pyfunction!(extended_groebner, m)?)?;
    m.add_function(wrap_pyfunction!(is_groebner, m)?)?;
    m.add_function(wrap_pyfunction!(groebner_walk, m)?)?;
    Ok(())
}

// -- modulargroebner

#[pyfunction]
#[pyo3(signature = (a, m, bound = None))]
fn rational_reconstruction<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    m: &Bound<'py, PyAny>,
    bound: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let bound: Option<Int> = match bound {
        Some(b) if !b.is_none() => Some(extract_int(b)?),
        _ => None,
    };
    match mg::rational_reconstruction(&extract_int(a)?, &extract_int(m)?, bound.as_ref()).map_err(err)? {
        Some((n, d)) => Ok(Some(tuple(py, vec![out_int(py, &n)?, out_int(py, &d)?])?)),
        None => Ok(None),
    }
}

#[pyfunction]
fn chinese_remainder<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    m: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    p: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let r: Int =
        mg::chinese_remainder(&extract_int(a)?, &extract_int(m)?, &extract_int(b)?, &extract_int(p)?).map_err(err)?;
    out_int(py, &r)
}

/// The iterator of [`mg::default_primes`].
#[pyclass(name = "DefaultPrimes", module = "sympy_extras_rs.polys.modulargroebner")]
struct PyDefaultPrimes {
    inner: mg::DefaultPrimes,
}

#[pymethods]
impl PyDefaultPrimes {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(mut slf: PyRefMut<'py, Self>, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match slf.inner.next() {
            Some(p) => Ok(Some(out_int(py, &p)?)),
            None => Ok(None),
        }
    }
}

#[pyfunction]
fn default_primes() -> PyDefaultPrimes {
    PyDefaultPrimes {
        inner: mg::default_primes(),
    }
}

/// `ModularTrace`: what `modular_groebner` did.
#[pyclass(name = "ModularTrace", module = "sympy_extras_rs.polys.modulargroebner")]
#[derive(Default)]
struct PyModularTrace {
    inner: mg::ModularTrace,
}

fn int_list<'py>(py: Python<'py>, items: &[Int]) -> PyResult<Bound<'py, PyAny>> {
    let objects: Vec<Bound<'py, PyAny>> = items.iter().map(|n| out_int(py, n)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objects)?.into_any())
}

#[pymethods]
impl PyModularTrace {
    #[new]
    fn new() -> PyModularTrace {
        PyModularTrace::default()
    }

    #[getter]
    fn homogenized(&self) -> bool {
        self.inner.homogenized
    }

    #[getter]
    fn used<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        int_list(py, &self.inner.used)
    }

    #[getter]
    fn skipped<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        int_list(py, &self.inner.skipped)
    }

    #[getter]
    fn unlucky<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        int_list(py, &self.inner.unlucky)
    }

    #[getter]
    fn lucky<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        int_list(py, &self.inner.lucky)
    }

    #[getter]
    fn failed_verifications(&self) -> usize {
        self.inner.failed_verifications
    }

    fn __repr__(&self) -> String {
        let show =
            |v: &[Int]| -> String { format!("[{}]", v.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")) };
        format!(
            "ModularTrace(used={}, skipped={}, unlucky={}, lucky={}, failed_verifications={})",
            show(&self.inner.used),
            show(&self.inner.skipped),
            show(&self.inner.unlucky),
            show(&self.inner.lucky),
            self.inner.failed_verifications
        )
    }
}

/// The integers of a Python iterator; the first error stops it and is kept.
struct IntIterator<'py> {
    iterator: Bound<'py, pyo3::types::PyIterator>,
    error: Option<PyErr>,
}

impl Iterator for IntIterator<'_> {
    type Item = Int;

    fn next(&mut self) -> Option<Int> {
        if self.error.is_some() {
            return None;
        }
        match self.iterator.next()? {
            Ok(item) => match extract_int(&item) {
                Ok(n) => Some(n),
                Err(e) => {
                    self.error = Some(e);
                    None
                }
            },
            Err(e) => {
                self.error = Some(e);
                None
            }
        }
    }
}

/// `modular_groebner(polys, ring, primes=None, trace=None)` over `QQ` and
/// `ZZ`; for `GF(p)` the basis of `f4`, as Python falls back on SymPy's.
#[pyfunction]
#[pyo3(signature = (polys, ring, primes = None, trace = None))]
fn modular_groebner<'py>(
    py: Python<'py>,
    polys: &Bound<'py, PyAny>,
    ring: &PyPolyRing,
    primes: Option<&Bound<'py, PyAny>>,
    trace: Option<&Bound<'py, PyModularTrace>>,
) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<ElemKind> = elements(polys)?;
    let mut given: Option<IntIterator<'py>> = match primes {
        Some(p) if !p.is_none() => Some(IntIterator {
            iterator: p.try_iter()?,
            error: None,
        }),
        _ => None,
    };
    let mut record: mg::ModularTrace = mg::ModularTrace::default();
    let result: PyResult<Vec<ElemKind>> = {
        let primes_arg: Option<&mut dyn Iterator<Item = Int>> =
            given.as_mut().map(|g| g as &mut dyn Iterator<Item = Int>);
        match &ring.inner {
            RingKind::QQ(r) => mg::modular_groebner(&same_kind!(&items, QQ), r, primes_arg, Some(&mut record))
                .map(|g| wrap(g, ElemKind::QQ))
                .map_err(err),
            RingKind::ZZ(r) => mg::modular_groebner_zz(&same_kind!(&items, ZZ), r, primes_arg, Some(&mut record))
                .map(|g| wrap(g, ElemKind::ZZ))
                .map_err(err),
            RingKind::GF(r) => groebner::f4(&same_kind!(&items, GF), r)
                .map(|g| wrap(g, ElemKind::GF))
                .map_err(err),
            RingKind::GaussianQQ(_) => Err(PyTypeError::new_err("modular Groebner bases over QQ or ZZ")),
        }
    };
    if let Some(e) = given.and_then(|g| g.error) {
        return Err(e);
    }
    if let Some(t) = trace {
        t.borrow_mut().inner = record;
    }
    out_elements(py, result?)
}

pub fn register_modulargroebner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("GUARD_BITS", mg::GUARD_BITS)?;
    m.add_class::<PyModularTrace>()?;
    m.add_class::<PyDefaultPrimes>()?;
    m.add_function(wrap_pyfunction!(rational_reconstruction, m)?)?;
    m.add_function(wrap_pyfunction!(chinese_remainder, m)?)?;
    m.add_function(wrap_pyfunction!(default_primes, m)?)?;
    m.add_function(wrap_pyfunction!(modular_groebner, m)?)?;
    Ok(())
}
