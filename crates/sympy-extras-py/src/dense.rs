//! `sympy_extras_rs.polys.{densebasic, densearith, densetools,
//! euclidtools, sqfreetools}`: the functions of the Python modules with the
//! same names and arguments (the domain last, as `K`).

// the Python names (`dup_LC`, the argument `K`)
#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sympy_extras_core::polys::domains::RationalField;
use sympy_extras_core::polys::{
    densearith as da, densebasic as db, densetools as dt, euclidtools as et, sqfreetools as sq,
};

use crate::convert::{
    err, out_dict, out_elem, out_factors, out_poly, out_polys, out_usize, poly, terms, tuple, Bridge,
};
use crate::domains::PyDomain;
use crate::with_domain;

/// The conversion of an argument of a binding by its kind.
macro_rules! arg {
    ($k:ident, $a:ident, poly) => {
        crate::convert::poly($k, $a)?
    };
    ($k:ident, $a:ident, elem) => {
        $k.extract($a)?
    };
    ($k:ident, $a:ident, usize) => {
        $a.extract::<usize>()?
    };
    ($k:ident, $a:ident, terms) => {
        terms($k, $a)?
    };
    ($k:ident, $a:ident, ints) => {{
        let mut values: Vec<sympy_extras_core::numbers::Int> = Vec::new();
        for item in $a.try_iter()? {
            values.push(crate::convert::extract_int(&item?)?);
        }
        values
    }};
}

/// A binding `name(args..., K)` whose body runs with `k` the Rust domain
/// and the arguments converted by their kinds.
macro_rules! binding {
    ($name:ident, ($($a:ident : $t:ident),*), |$py:ident, $k:ident| $body:expr) => {
        #[pyfunction]
        #[allow(non_snake_case, unused_variables)]
        pub fn $name<'py>($py: Python<'py>, $($a: &Bound<'py, PyAny>,)* K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
            with_domain!(K, $k => { $(let $a = arg!($k, $a, $t);)* $body })
        }
    };
    (ordered $name:ident, ($($a:ident : $t:ident),*), |$py:ident, $k:ident| $body:expr) => {
        #[pyfunction]
        #[allow(non_snake_case, unused_variables)]
        pub fn $name<'py>($py: Python<'py>, $($a: &Bound<'py, PyAny>,)* K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
            K.require_ordered()?;
            with_domain!(K, $k => { $(let $a = arg!($k, $a, $t);)* $body })
        }
    };
}

fn pair<'py, K: Bridge>(k: &K, py: Python<'py>, value: (Vec<K::Elem>, Vec<K::Elem>)) -> PyResult<Bound<'py, PyAny>> {
    tuple(py, vec![out_poly(k, py, &value.0)?, out_poly(k, py, &value.1)?])
}

fn elem_poly<'py, K: Bridge>(k: &K, py: Python<'py>, value: (K::Elem, Vec<K::Elem>)) -> PyResult<Bound<'py, PyAny>> {
    tuple(py, vec![out_elem(k, py, &value.0)?, out_poly(k, py, &value.1)?])
}

// -- densebasic

binding!(dup_strip, (f: poly), |py, k| out_poly(k, py, &db::dup_strip(&f, k)));
binding!(dup_zero, (), |py, k| out_poly(k, py, &db::dup_zero(k)));
binding!(dup_one, (), |py, k| out_poly(k, py, &db::dup_one(k)));
binding!(dup_constant, (c: elem), |py, k| out_poly(k, py, &db::dup_constant(&c, k)));
binding!(dup_monomial, (c: elem, n: usize), |py, k| out_poly(k, py, &db::dup_monomial(&c, n, k)));
binding!(dup_is_one, (f: poly), |py, k| Ok(db::dup_is_one(&f, k).into_pyobject(py)?.to_owned().into_any()));
binding!(dup_LC, (f: poly), |py, k| out_elem(k, py, &db::dup_LC(&f, k)));
binding!(dup_TC, (f: poly), |py, k| out_elem(k, py, &db::dup_TC(&f, k)));
binding!(dup_normal, (f: poly), |py, k| out_poly(k, py, &db::dup_normal(&f, k)));
binding!(dup_from_list, (f: ints), |py, k| out_poly(k, py, &db::dup_from_list(&f, k)));
binding!(dup_from_dict, (f: terms), |py, k| out_poly(k, py, &db::dup_from_dict(&f, k)));
binding!(dup_to_dict, (f: poly), |py, k| out_dict(k, py, &db::dup_to_dict(&f, k)));
binding!(dup_terms, (f: poly), |py, k| {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
    for (e, c) in db::dup_terms(&f, k) {
        items.push(tuple(py, vec![out_usize(py, e)?, out_elem(k, py, &c)?])?);
    }
    Ok(pyo3::types::PyList::new(py, items)?.into_any())
});
binding!(dup_reverse, (f: poly), |py, k| out_poly(k, py, &db::dup_reverse(&f, k)));
binding!(dup_slice, (f: poly, m: usize, n: usize), |py, k| out_poly(k, py, &db::dup_slice(&f, m, n, k)));

/// `dup_convert(f, K0, K1)`.
#[pyfunction]
#[allow(non_snake_case)]
pub fn dup_convert<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    K0: &PyDomain,
    K1: &PyDomain,
) -> PyResult<Bound<'py, PyAny>> {
    crate::with_base_domain!(K0, k0 => {
        let f = poly(k0, f)?;
        crate::with_base_domain!(K1, k1 => out_poly(k1, py, &db::dup_convert(&f, k0, k1).map_err(err)?))
    })
}

/// The functions of `densebasic` which take no domain.
#[pyfunction]
pub fn dup_degree(f: &Bound<'_, PyAny>) -> PyResult<isize> {
    Ok(f.len()? as isize - 1)
}

#[pyfunction]
pub fn dup_is_zero(f: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(f.len()? == 0)
}

#[pyfunction]
pub fn dup_is_constant(f: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(f.len()? <= 1)
}

#[pyfunction]
pub fn dup_copy<'py>(f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(pyo3::types::PyList::new(f.py(), f.try_iter()?.collect::<PyResult<Vec<Bound<'py, PyAny>>>>()?)?.into_any())
}

// -- densearith

binding!(dup_neg, (f: poly), |py, k| out_poly(k, py, &da::dup_neg(&f, k)));
binding!(dup_add, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_add(&f, &g, k)));
binding!(dup_sub, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_sub(&f, &g, k)));
binding!(dup_mul, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_mul(&f, &g, k)));
binding!(dup_sqr, (f: poly), |py, k| out_poly(k, py, &da::dup_sqr(&f, k)));
binding!(dup_pow, (f: poly, n: usize), |py, k| out_poly(k, py, &da::dup_pow(&f, n, k)));
binding!(dup_add_term, (f: poly, c: elem, n: usize), |py, k| out_poly(k, py, &da::dup_add_term(&f, &c, n, k)));
binding!(dup_sub_term, (f: poly, c: elem, n: usize), |py, k| out_poly(k, py, &da::dup_sub_term(&f, &c, n, k)));
binding!(dup_mul_term, (f: poly, c: elem, n: usize), |py, k| out_poly(k, py, &da::dup_mul_term(&f, &c, n, k)));
binding!(dup_add_ground, (f: poly, c: elem), |py, k| out_poly(k, py, &da::dup_add_ground(&f, &c, k)));
binding!(dup_sub_ground, (f: poly, c: elem), |py, k| out_poly(k, py, &da::dup_sub_ground(&f, &c, k)));
binding!(dup_mul_ground, (f: poly, c: elem), |py, k| out_poly(k, py, &da::dup_mul_ground(&f, &c, k)));
binding!(dup_quo_ground, (f: poly, c: elem), |py, k| out_poly(k, py, &da::dup_quo_ground(&f, &c, k).map_err(err)?));
binding!(dup_exquo_ground, (f: poly, c: elem), |py, k| out_poly(k, py, &da::dup_exquo_ground(&f, &c, k).map_err(err)?));
binding!(dup_lshift, (f: poly, n: usize), |py, k| out_poly(k, py, &da::dup_lshift(&f, n, k)));
binding!(dup_rshift, (f: poly, n: usize), |py, k| out_poly(k, py, &da::dup_rshift(&f, n, k)));
binding!(dup_div, (f: poly, g: poly), |py, k| pair(k, py, da::dup_div(&f, &g, k).map_err(err)?));
binding!(dup_quo, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_quo(&f, &g, k).map_err(err)?));
binding!(dup_rem, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_rem(&f, &g, k).map_err(err)?));
binding!(dup_exquo, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_exquo(&f, &g, k).map_err(err)?));
binding!(dup_pdiv, (f: poly, g: poly), |py, k| pair(k, py, da::dup_pdiv(&f, &g, k).map_err(err)?));
binding!(dup_prem, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_prem(&f, &g, k).map_err(err)?));
binding!(dup_pquo, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_pquo(&f, &g, k).map_err(err)?));
binding!(dup_pexquo, (f: poly, g: poly), |py, k| out_poly(k, py, &da::dup_pexquo(&f, &g, k).map_err(err)?));
binding!(ordered dup_max_norm, (f: poly), |py, k| out_elem(k, py, &da::dup_max_norm(&f, k)));
binding!(ordered dup_l1_norm, (f: poly), |py, k| out_elem(k, py, &da::dup_l1_norm(&f, k)));

// -- densetools

binding!(dup_diff, (f: poly, m: usize), |py, k| out_poly(k, py, &dt::dup_diff(&f, m, k)));
binding!(dup_eval, (f: poly, a: elem), |py, k| out_elem(k, py, &dt::dup_eval(&f, &a, k)));
binding!(dup_shift, (f: poly, a: elem), |py, k| out_poly(k, py, &dt::dup_shift(&f, &a, k)));
binding!(dup_taylor_shift, (f: poly, a: elem), |py, k| out_poly(k, py, &dt::dup_taylor_shift(&f, &a, k)));
binding!(dup_compose, (f: poly, g: poly), |py, k| out_poly(k, py, &dt::dup_compose(&f, &g, k)));
binding!(dup_mirror, (f: poly), |py, k| out_poly(k, py, &dt::dup_mirror(&f, k)));
binding!(dup_scale, (f: poly, a: elem), |py, k| out_poly(k, py, &dt::dup_scale(&f, &a, k)));
binding!(dup_content, (f: poly), |py, k| out_elem(k, py, &dt::dup_content(&f, k)));
binding!(dup_primitive, (f: poly), |py, k| elem_poly(k, py, dt::dup_primitive(&f, k)));
binding!(dup_decompose_content, (f: poly), |py, k| elem_poly(k, py, dt::dup_decompose_content(&f, k)));
binding!(dup_monic, (f: poly), |py, k| out_poly(k, py, &dt::dup_monic(&f, k).map_err(err)?));
binding!(dup_trunc, (f: poly, p: elem), |py, k| out_poly(k, py, &dt::dup_trunc(&f, &p, k).map_err(err)?));
binding!(dup_integrate, (f: poly, m: usize), |py, k| out_poly(k, py, &dt::dup_integrate(&f, m, k).map_err(err)?));
binding!(ordered dup_sign_variations, (f: poly), |py, k| out_usize(py, dt::dup_sign_variations(&f, k)));

/// `dup_clear_denoms(f, K)` over `QQ`.
#[pyfunction]
#[allow(non_snake_case)]
pub fn dup_clear_denoms<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyDomain) -> PyResult<Bound<'py, PyAny>> {
    if K.kind != crate::domains::DomainKind::QQ {
        return Err(pyo3::exceptions::PyTypeError::new_err("dup_clear_denoms needs QQ"));
    }
    let k: &RationalField = &RationalField;
    let (common, g) = dt::dup_clear_denoms(&poly(k, f)?, k);
    tuple(py, vec![crate::convert::out_int(py, &common)?, out_poly(k, py, &g)?])
}

// -- euclidtools

binding!(dup_euclid, (f: poly, g: poly), |py, k| out_poly(k, py, &et::dup_euclid(&f, &g, k).map_err(err)?));
binding!(dup_prs, (f: poly, g: poly), |py, k| out_polys(k, py, &et::dup_prs(&f, &g, k).map_err(err)?));
binding!(dup_subresultants, (f: poly, g: poly), |py, k| out_polys(k, py, &et::dup_subresultants(&f, &g, k)));
binding!(dup_gcd, (f: poly, g: poly), |py, k| out_poly(k, py, &et::dup_gcd(&f, &g, k)));
binding!(dup_lcm, (f: poly, g: poly), |py, k| out_poly(k, py, &et::dup_lcm(&f, &g, k)));
binding!(dup_gcdex, (f: poly, g: poly), |py, k| {
    let (s, t, h) = et::dup_gcdex(&f, &g, k).map_err(err)?;
    tuple(py, vec![out_poly(k, py, &s)?, out_poly(k, py, &t)?, out_poly(k, py, &h)?])
});
binding!(dup_half_gcdex, (f: poly, g: poly), |py, k| pair(k, py, et::dup_half_gcdex(&f, &g, k).map_err(err)?));
binding!(dup_invert, (f: poly, g: poly), |py, k| out_poly(k, py, &et::dup_invert(&f, &g, k).map_err(err)?));
binding!(dup_resultant, (f: poly, g: poly), |py, k| out_elem(k, py, &et::dup_resultant(&f, &g, k)));
binding!(dup_prs_resultant, (f: poly, g: poly), |py, k| out_elem(k, py, &et::dup_prs_resultant(&f, &g, k)));
binding!(dup_inner_subresultants, (f: poly, g: poly), |py, k| {
    let (r, s) = et::dup_inner_subresultants(&f, &g, k);
    let mut scalars: Vec<Bound<'py, PyAny>> = Vec::new();
    for c in &s {
        scalars.push(out_elem(k, py, c)?);
    }
    tuple(py, vec![out_polys(k, py, &r)?, pyo3::types::PyList::new(py, scalars)?.into_any()])
});
binding!(dup_discriminant, (f: poly), |py, k| out_elem(k, py, &et::dup_discriminant(&f, k)));

/// `dup_qq_gcd(f, g)` (no domain argument: always `QQ`).
#[pyfunction]
pub fn dup_qq_gcd<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, g: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let k: &RationalField = &RationalField;
    out_poly(k, py, &et::dup_qq_gcd(&poly(k, f)?, &poly(k, g)?))
}

// -- sqfreetools

binding!(dup_sqf_norm_content, (f: poly), |py, k| elem_poly(k, py, sq::dup_sqf_norm_content(&f, k)));
binding!(dup_sqf_list, (f: poly), |py, k| {
    let (content, factors) = sq::dup_sqf_list(&f, k);
    tuple(py, vec![out_elem(k, py, &content)?, out_factors(k, py, &factors)?])
});
binding!(dup_sqf_part, (f: poly), |py, k| out_poly(k, py, &sq::dup_sqf_part(&f, k)));
binding!(dup_sqf_p, (f: poly), |py, k| Ok(sq::dup_sqf_p(&f, k).into_pyobject(py)?.to_owned().into_any()));

/// `KARATSUBA_THRESHOLD` and the functions of each module.
pub fn register_densebasic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dup_strip, m)?)?;
    m.add_function(wrap_pyfunction!(dup_degree, m)?)?;
    m.add_function(wrap_pyfunction!(dup_LC, m)?)?;
    m.add_function(wrap_pyfunction!(dup_TC, m)?)?;
    m.add_function(wrap_pyfunction!(dup_normal, m)?)?;
    m.add_function(wrap_pyfunction!(dup_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(dup_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(dup_from_list, m)?)?;
    m.add_function(wrap_pyfunction!(dup_convert, m)?)?;
    m.add_function(wrap_pyfunction!(dup_zero, m)?)?;
    m.add_function(wrap_pyfunction!(dup_one, m)?)?;
    m.add_function(wrap_pyfunction!(dup_constant, m)?)?;
    m.add_function(wrap_pyfunction!(dup_is_zero, m)?)?;
    m.add_function(wrap_pyfunction!(dup_is_one, m)?)?;
    m.add_function(wrap_pyfunction!(dup_is_constant, m)?)?;
    m.add_function(wrap_pyfunction!(dup_monomial, m)?)?;
    m.add_function(wrap_pyfunction!(dup_reverse, m)?)?;
    m.add_function(wrap_pyfunction!(dup_copy, m)?)?;
    m.add_function(wrap_pyfunction!(dup_slice, m)?)?;
    m.add_function(wrap_pyfunction!(dup_terms, m)?)?;
    Ok(())
}

pub fn register_densearith(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("KARATSUBA_THRESHOLD", da::KARATSUBA_THRESHOLD)?;
    m.add_function(wrap_pyfunction!(dup_add, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sub, m)?)?;
    m.add_function(wrap_pyfunction!(dup_neg, m)?)?;
    m.add_function(wrap_pyfunction!(dup_mul, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sqr, m)?)?;
    m.add_function(wrap_pyfunction!(dup_pow, m)?)?;
    m.add_function(wrap_pyfunction!(dup_add_term, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sub_term, m)?)?;
    m.add_function(wrap_pyfunction!(dup_mul_term, m)?)?;
    m.add_function(wrap_pyfunction!(dup_add_ground, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sub_ground, m)?)?;
    m.add_function(wrap_pyfunction!(dup_mul_ground, m)?)?;
    m.add_function(wrap_pyfunction!(dup_quo_ground, m)?)?;
    m.add_function(wrap_pyfunction!(dup_exquo_ground, m)?)?;
    m.add_function(wrap_pyfunction!(dup_div, m)?)?;
    m.add_function(wrap_pyfunction!(dup_quo, m)?)?;
    m.add_function(wrap_pyfunction!(dup_rem, m)?)?;
    m.add_function(wrap_pyfunction!(dup_exquo, m)?)?;
    m.add_function(wrap_pyfunction!(dup_pdiv, m)?)?;
    m.add_function(wrap_pyfunction!(dup_prem, m)?)?;
    m.add_function(wrap_pyfunction!(dup_pquo, m)?)?;
    m.add_function(wrap_pyfunction!(dup_pexquo, m)?)?;
    m.add_function(wrap_pyfunction!(dup_lshift, m)?)?;
    m.add_function(wrap_pyfunction!(dup_rshift, m)?)?;
    m.add_function(wrap_pyfunction!(dup_max_norm, m)?)?;
    m.add_function(wrap_pyfunction!(dup_l1_norm, m)?)?;
    Ok(())
}

pub fn register_densetools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dup_diff, m)?)?;
    m.add_function(wrap_pyfunction!(dup_eval, m)?)?;
    m.add_function(wrap_pyfunction!(dup_shift, m)?)?;
    m.add_function(wrap_pyfunction!(dup_compose, m)?)?;
    m.add_function(wrap_pyfunction!(dup_content, m)?)?;
    m.add_function(wrap_pyfunction!(dup_primitive, m)?)?;
    m.add_function(wrap_pyfunction!(dup_monic, m)?)?;
    m.add_function(wrap_pyfunction!(dup_clear_denoms, m)?)?;
    m.add_function(wrap_pyfunction!(dup_trunc, m)?)?;
    m.add_function(wrap_pyfunction!(dup_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(dup_mirror, m)?)?;
    m.add_function(wrap_pyfunction!(dup_scale, m)?)?;
    m.add_function(wrap_pyfunction!(dup_taylor_shift, m)?)?;
    m.add_function(wrap_pyfunction!(dup_decompose_content, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sign_variations, m)?)?;
    Ok(())
}

pub fn register_euclidtools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dup_gcd, m)?)?;
    m.add_function(wrap_pyfunction!(dup_qq_gcd, m)?)?;
    m.add_function(wrap_pyfunction!(dup_lcm, m)?)?;
    m.add_function(wrap_pyfunction!(dup_gcdex, m)?)?;
    m.add_function(wrap_pyfunction!(dup_half_gcdex, m)?)?;
    m.add_function(wrap_pyfunction!(dup_invert, m)?)?;
    m.add_function(wrap_pyfunction!(dup_subresultants, m)?)?;
    m.add_function(wrap_pyfunction!(dup_resultant, m)?)?;
    m.add_function(wrap_pyfunction!(dup_prs_resultant, m)?)?;
    m.add_function(wrap_pyfunction!(dup_inner_subresultants, m)?)?;
    m.add_function(wrap_pyfunction!(dup_discriminant, m)?)?;
    m.add_function(wrap_pyfunction!(dup_euclid, m)?)?;
    m.add_function(wrap_pyfunction!(dup_prs, m)?)?;
    Ok(())
}

pub fn register_sqfreetools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dup_sqf_list, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sqf_part, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sqf_p, m)?)?;
    m.add_function(wrap_pyfunction!(dup_sqf_norm_content, m)?)?;
    Ok(())
}
