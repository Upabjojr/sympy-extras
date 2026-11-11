//! `sympy_extras_rs.polys.domains.galoisfield` and
//! `sympy_extras_rs.polys.galoisfactor`: the class `GaloisField` with the
//! methods of the Python domain (elements are lists of residues, highest
//! degree first), and the factorisation over it.

// the Python name of the domain argument (`K`)
#![allow(non_snake_case)]

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::Int;
use sympy_extras_core::polys::domains::{Domain, GaloisElement, GaloisField};
use sympy_extras_core::polys::galoisfactor as gq;

use crate::convert::{err, extract_int, out_int, out_usize, tuple};

/// The field with `p**n` elements.
#[pyclass(
    name = "GaloisField",
    module = "sympy_extras_rs.polys.domains.galoisfield",
    frozen,
    eq
)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyGaloisField {
    pub inner: GaloisField,
}

fn element(ob: &Bound<'_, PyAny>) -> PyResult<GaloisElement> {
    let mut values: GaloisElement = Vec::new();
    for item in ob.try_iter()? {
        let c: Int = extract_int(&item?)?;
        values.push(u64::try_from(c).map_err(|_| PyTypeError::new_err("residues in [0, p) are expected"))?);
    }
    Ok(values)
}

fn out_element<'py>(py: Python<'py>, a: &GaloisElement) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyList::new(py, a)?.into_any())
}

fn poly(ob: &Bound<'_, PyAny>) -> PyResult<Vec<GaloisElement>> {
    let mut values: Vec<GaloisElement> = Vec::new();
    for item in ob.try_iter()? {
        values.push(element(&item?)?);
    }
    Ok(values)
}

fn out_poly<'py>(py: Python<'py>, f: &[GaloisElement]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = f.iter().map(|c| out_element(py, c)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

fn out_factors<'py>(py: Python<'py>, factors: &gq::GaloisFactors) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = factors
        .iter()
        .map(|(g, k)| tuple(py, vec![out_poly(py, g)?, out_usize(py, *k)?]))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

#[pymethods]
impl PyGaloisField {
    #[new]
    #[pyo3(signature = (p, n, modulus = None))]
    fn new(p: u64, n: usize, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<PyGaloisField> {
        let modulus: Option<Vec<Int>> = match modulus {
            Some(m) => {
                let mut values: Vec<Int> = Vec::new();
                for item in m.try_iter()? {
                    values.push(extract_int(&item?)?);
                }
                Some(values)
            }
            None => None,
        };
        Ok(PyGaloisField {
            inner: GaloisField::new(p, n, modulus.as_deref()).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        self.inner.name()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ("GaloisField", self.inner.p, &self.inner.modulus).hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    #[getter]
    fn p(&self) -> u64 {
        self.inner.p
    }

    #[getter]
    fn characteristic(&self) -> u64 {
        self.inner.p
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree
    }

    #[getter]
    fn order<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_int(py, &self.inner.order)
    }

    #[getter]
    fn modulus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.modulus)
    }

    #[getter]
    fn is_field(&self) -> bool {
        true
    }

    #[getter]
    fn is_ordered(&self) -> bool {
        false
    }

    #[getter]
    fn zero<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.zero())
    }

    #[getter]
    fn one<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.one())
    }

    #[getter]
    fn generator<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.generator())
    }

    fn is_zero(&self, a: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.is_zero(&element(a)?))
    }

    fn is_one(&self, a: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.is_one(&element(a)?))
    }

    fn is_unit(&self, a: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.is_unit(&element(a)?))
    }

    fn add<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.add(&element(a)?, &element(b)?))
    }

    fn sub<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.sub(&element(a)?, &element(b)?))
    }

    fn neg<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.neg(&element(a)?))
    }

    fn mul<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.mul(&element(a)?, &element(b)?))
    }

    fn pow<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, n: i64) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.pow_signed(&element(a)?, n).map_err(err)?)
    }

    fn exquo<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.exquo(&element(a)?, &element(b)?).map_err(err)?)
    }

    fn quo<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.quo(&element(a)?, &element(b)?).map_err(err)?)
    }

    fn rem<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.rem(&element(a)?, &element(b)?).map_err(err)?)
    }

    fn invert<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.invert(&element(a)?).map_err(err)?)
    }

    fn gcd<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.gcd(&element(a)?, &element(b)?))
    }

    fn lcm<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.lcm(&element(a)?, &element(b)?))
    }

    fn gcdex<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let (s, t, g) = self.inner.gcdex(&element(a)?, &element(b)?).map_err(err)?;
        tuple(
            py,
            vec![out_element(py, &s)?, out_element(py, &t)?, out_element(py, &g)?],
        )
    }

    fn canonical_unit<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.canonical_unit(&element(a)?))
    }

    fn frobenius<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.frobenius(&element(a)?))
    }

    fn pth_root<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.pth_root(&element(a)?))
    }

    fn trace(&self, a: &Bound<'_, PyAny>) -> PyResult<u64> {
        Ok(self.inner.trace(&element(a)?))
    }

    fn from_int<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        out_element(py, &self.inner.from_integer(&extract_int(n)?))
    }

    fn to_str(&self, a: &Bound<'_, PyAny>) -> PyResult<String> {
        Ok(self.inner.to_str(&element(a)?))
    }

    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Bound<'py, PyAny>> = self
            .inner
            .elements()
            .map(|a| out_element(py, &a))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into_any())
    }
}

#[pyfunction]
fn gfq_pow_mod<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
    g: &Bound<'py, PyAny>,
    K: &PyGaloisField,
) -> PyResult<Bound<'py, PyAny>> {
    out_poly(
        py,
        &gq::gfq_pow_mod(&poly(f)?, &extract_int(n)?, &poly(g)?, &K.inner).map_err(err)?,
    )
}

#[pyfunction]
fn gfq_sqf_list<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyGaloisField) -> PyResult<Bound<'py, PyAny>> {
    let (lc, factors) = gq::gfq_sqf_list(&poly(f)?, &K.inner).map_err(err)?;
    tuple(py, vec![out_element(py, &lc)?, out_factors(py, &factors)?])
}

#[pyfunction]
fn gfq_ddf<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyGaloisField) -> PyResult<Bound<'py, PyAny>> {
    out_factors(py, &gq::gfq_ddf(&poly(f)?, &K.inner).map_err(err)?)
}

#[pyfunction]
fn gfq_factor<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, K: &PyGaloisField) -> PyResult<Bound<'py, PyAny>> {
    let (lc, factors) = gq::gfq_factor(&poly(f)?, &K.inner).map_err(err)?;
    tuple(py, vec![out_element(py, &lc)?, out_factors(py, &factors)?])
}

#[pyfunction]
fn gfq_irreducible_p(f: &Bound<'_, PyAny>, K: &PyGaloisField) -> PyResult<bool> {
    gq::gfq_irreducible_p(&poly(f)?, &K.inner).map_err(err)
}

pub fn register_galoisfield(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaloisField>()?;
    Ok(())
}

pub fn register_galoisfactor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gfq_pow_mod, m)?)?;
    m.add_function(wrap_pyfunction!(gfq_sqf_list, m)?)?;
    m.add_function(wrap_pyfunction!(gfq_ddf, m)?)?;
    m.add_function(wrap_pyfunction!(gfq_factor, m)?)?;
    m.add_function(wrap_pyfunction!(gfq_irreducible_p, m)?)?;
    Ok(())
}
