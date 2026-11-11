//! `sympy_extras_rs.polys.complexroots`: the class `Rectangle` and the
//! complex root counting and isolation of the Python module.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::Rat;
use sympy_extras_core::polys::complexroots as cr;
use sympy_extras_core::polys::domains::RationalField;

use crate::convert::{err, out_usize, tuple, Bridge};

fn rational(ob: &Bound<'_, PyAny>) -> PyResult<Rat> {
    RationalField.extract(ob)
}

fn out_rational<'py>(py: Python<'py>, v: &Rat) -> PyResult<Bound<'py, PyAny>> {
    RationalField.to_py(py, v)
}

fn poly(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Rat>> {
    let mut values: Vec<Rat> = Vec::new();
    for item in ob.try_iter()? {
        values.push(rational(&item?)?);
    }
    Ok(values)
}

fn optional_rational(ob: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Rat>> {
    match ob {
        Some(o) if !o.is_none() => Ok(Some(rational(o)?)),
        _ => Ok(None),
    }
}

/// The rectangle `[re_min, re_max] x [im_min, im_max]`.
#[pyclass(name = "Rectangle", module = "sympy_extras_rs.polys.complexroots", frozen, eq, hash)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyRectangle {
    inner: cr::Rectangle,
}

fn rectangle_list<'py>(py: Python<'py>, items: Vec<cr::Rectangle>) -> PyResult<Bound<'py, PyAny>> {
    let objects: Vec<Bound<'py, PyAny>> = items
        .into_iter()
        .map(|inner| Ok(Bound::new(py, PyRectangle { inner })?.into_any()))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objects)?.into_any())
}

#[pymethods]
impl PyRectangle {
    #[new]
    fn new(
        re_min: &Bound<'_, PyAny>,
        re_max: &Bound<'_, PyAny>,
        im_min: &Bound<'_, PyAny>,
        im_max: &Bound<'_, PyAny>,
    ) -> PyResult<PyRectangle> {
        Ok(PyRectangle {
            inner: cr::Rectangle::new(
                rational(re_min)?,
                rational(re_max)?,
                rational(im_min)?,
                rational(im_max)?,
            )
            .map_err(err)?,
        })
    }

    #[getter]
    fn re_min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rational(py, &self.inner.re_min)
    }

    #[getter]
    fn re_max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rational(py, &self.inner.re_max)
    }

    #[getter]
    fn im_min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rational(py, &self.inner.im_min)
    }

    #[getter]
    fn im_max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rational(py, &self.inner.im_max)
    }

    #[getter]
    fn width<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rational(py, &self.inner.width())
    }

    #[getter]
    fn height<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_rational(py, &self.inner.height())
    }

    #[getter]
    fn center<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (re, im) = self.inner.center();
        tuple(py, vec![out_rational(py, &re)?, out_rational(py, &im)?])
    }

    fn corners<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tuple(
            py,
            vec![self.re_min(py)?, self.re_max(py)?, self.im_min(py)?, self.im_max(py)?],
        )
    }

    fn conjugate(&self) -> PyRectangle {
        PyRectangle {
            inner: self.inner.conjugate(),
        }
    }

    fn contains(&self, re: &Bound<'_, PyAny>, im: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.contains(&rational(re)?, &rational(im)?))
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

#[pyfunction]
fn dup_count_complex_roots(f: &Bound<'_, PyAny>, rectangle: &PyRectangle) -> PyResult<i64> {
    cr::dup_count_complex_roots(&poly(f)?, &rectangle.inner).map_err(err)
}

#[pyfunction]
fn dup_refine_complex_root(
    f: &Bound<'_, PyAny>,
    rectangle: &PyRectangle,
    eps: &Bound<'_, PyAny>,
) -> PyResult<PyRectangle> {
    Ok(PyRectangle {
        inner: cr::dup_refine_complex_root(&poly(f)?, &rectangle.inner, &rational(eps)?).map_err(err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (f, eps = None))]
fn dup_isolate_complex_roots_sqf<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    eps: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let eps: Option<Rat> = optional_rational(eps)?;
    rectangle_list(
        py,
        cr::dup_isolate_complex_roots_sqf(&poly(f)?, eps.as_ref()).map_err(err)?,
    )
}

#[pyfunction]
#[pyo3(signature = (f, eps = None))]
fn dup_isolate_all_roots_sqf<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    eps: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let eps: Option<Rat> = optional_rational(eps)?;
    let (real, upper) = cr::dup_isolate_all_roots_sqf(&poly(f)?, eps.as_ref()).map_err(err)?;
    let intervals: Vec<Bound<'py, PyAny>> = real
        .iter()
        .map(|(a, b)| tuple(py, vec![out_rational(py, a)?, out_rational(py, b)?]))
        .collect::<PyResult<_>>()?;
    tuple(
        py,
        vec![PyList::new(py, intervals)?.into_any(), rectangle_list(py, upper)?],
    )
}

#[pyfunction]
#[pyo3(signature = (f, eps = None))]
fn dup_isolate_complex_roots<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    eps: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let eps: Option<Rat> = optional_rational(eps)?;
    let found = cr::dup_isolate_complex_roots(&poly(f)?, eps.as_ref()).map_err(err)?;
    let items: Vec<Bound<'py, PyAny>> = found
        .into_iter()
        .map(|(inner, k)| {
            tuple(
                py,
                vec![Bound::new(py, PyRectangle { inner })?.into_any(), out_usize(py, k)?],
            )
        })
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (f, a = None))]
fn dup_count_roots_right_of(f: &Bound<'_, PyAny>, a: Option<&Bound<'_, PyAny>>) -> PyResult<i64> {
    let a: Rat = optional_rational(a)?.unwrap_or(Rat::ZERO);
    cr::dup_count_roots_right_of(&poly(f)?, &a).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (f, center = None, radius = None))]
fn dup_count_roots_in_disc(
    f: &Bound<'_, PyAny>,
    center: Option<&Bound<'_, PyAny>>,
    radius: Option<&Bound<'_, PyAny>>,
) -> PyResult<i64> {
    let center: (Rat, Rat) = match center {
        Some(c) if !c.is_none() => (rational(&c.get_item(0)?)?, rational(&c.get_item(1)?)?),
        _ => (Rat::ZERO, Rat::ZERO),
    };
    let radius: Rat = optional_rational(radius)?.unwrap_or(Rat::ONE);
    cr::dup_count_roots_in_disc(&poly(f)?, &center, &radius).map_err(err)
}

#[pyfunction]
fn dup_hurwitz_stable_p(f: &Bound<'_, PyAny>) -> PyResult<bool> {
    cr::dup_hurwitz_stable_p(&poly(f)?).map_err(err)
}

#[pyfunction]
fn dup_schur_stable_p(f: &Bound<'_, PyAny>) -> PyResult<bool> {
    cr::dup_schur_stable_p(&poly(f)?).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRectangle>()?;
    m.add_function(wrap_pyfunction!(dup_count_complex_roots, m)?)?;
    m.add_function(wrap_pyfunction!(dup_refine_complex_root, m)?)?;
    m.add_function(wrap_pyfunction!(dup_isolate_complex_roots_sqf, m)?)?;
    m.add_function(wrap_pyfunction!(dup_isolate_all_roots_sqf, m)?)?;
    m.add_function(wrap_pyfunction!(dup_isolate_complex_roots, m)?)?;
    m.add_function(wrap_pyfunction!(dup_count_roots_right_of, m)?)?;
    m.add_function(wrap_pyfunction!(dup_count_roots_in_disc, m)?)?;
    m.add_function(wrap_pyfunction!(dup_hurwitz_stable_p, m)?)?;
    m.add_function(wrap_pyfunction!(dup_schur_stable_p, m)?)?;
    Ok(())
}
