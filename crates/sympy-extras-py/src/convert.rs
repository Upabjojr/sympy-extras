//! Conversions between Python values and the elements of the Rust domains:
//! `int` for `ZZ` and `GF(p)`, `fractions.Fraction` for `QQ`, `list` for a
//! dense polynomial.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};

use sympy_extras_core::error::PolyError;
use sympy_extras_core::numbers::{denom, numer, rat_from_parts, Int, Rat};
use sympy_extras_core::polys::domains::{Domain, FiniteField, IntegerRing, RationalField};

/// The Python exception of an error of the core.
pub fn err(e: PolyError) -> PyErr {
    match e {
        PolyError::ZeroDivision(message) => PyZeroDivisionError::new_err(message),
        PolyError::Value(message) => PyValueError::new_err(message),
    }
}

/// A domain whose elements convert to and from Python values.
pub trait Bridge: Domain {
    fn extract(&self, ob: &Bound<'_, PyAny>) -> PyResult<Self::Elem>;
    fn to_py<'py>(&self, py: Python<'py>, e: &Self::Elem) -> PyResult<Bound<'py, PyAny>>;
}

impl Bridge for IntegerRing {
    fn extract(&self, ob: &Bound<'_, PyAny>) -> PyResult<Int> {
        extract_int(ob)
    }

    fn to_py<'py>(&self, py: Python<'py>, e: &Int) -> PyResult<Bound<'py, PyAny>> {
        out_int(py, e)
    }
}

static FRACTION: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

fn fraction_class(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    let class: &Py<PyAny> = FRACTION.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("fractions")?.getattr("Fraction")?.unbind())
    })?;
    Ok(class.bind(py))
}

impl Bridge for RationalField {
    fn extract(&self, ob: &Bound<'_, PyAny>) -> PyResult<Rat> {
        if ob.is_instance_of::<PyInt>() {
            return Ok(Rat::from(extract_int(ob)?));
        }
        let numerator: Int = extract_int(&ob.getattr("numerator")?)?;
        let denominator: Int = extract_int(&ob.getattr("denominator")?)?;
        if denominator.is_zero() {
            return Err(PyZeroDivisionError::new_err("zero denominator"));
        }
        Ok(rat_from_parts(numerator, denominator))
    }

    fn to_py<'py>(&self, py: Python<'py>, e: &Rat) -> PyResult<Bound<'py, PyAny>> {
        let numerator: Bound<'py, PyAny> = out_int(py, &numer(e))?;
        let denominator: Bound<'py, PyAny> = out_int(py, &denom(e))?;
        fraction_class(py)?.call1((numerator, denominator))
    }
}

impl Bridge for FiniteField {
    fn extract(&self, ob: &Bound<'_, PyAny>) -> PyResult<u64> {
        ob.extract::<u64>()
    }

    fn to_py<'py>(&self, py: Python<'py>, e: &u64) -> PyResult<Bound<'py, PyAny>> {
        Ok(e.into_pyobject(py)?.into_any())
    }
}

/// A sequence of coefficients (highest degree first).
pub fn poly<K: Bridge>(k: &K, ob: &Bound<'_, PyAny>) -> PyResult<Vec<K::Elem>> {
    let mut result: Vec<K::Elem> = Vec::new();
    for item in ob.try_iter()? {
        result.push(k.extract(&item?)?);
    }
    Ok(result)
}

/// A dict `{exponent: coefficient}`.
pub fn terms<K: Bridge>(k: &K, ob: &Bound<'_, PyAny>) -> PyResult<BTreeMap<usize, K::Elem>> {
    let dict: &Bound<'_, PyDict> = ob.downcast::<PyDict>()?;
    let mut result: BTreeMap<usize, K::Elem> = BTreeMap::new();
    for (key, value) in dict.iter() {
        result.insert(key.extract::<usize>()?, k.extract(&value)?);
    }
    Ok(result)
}

pub fn out_elem<'py, K: Bridge>(k: &K, py: Python<'py>, e: &K::Elem) -> PyResult<Bound<'py, PyAny>> {
    k.to_py(py, e)
}

pub fn out_poly<'py, K: Bridge>(k: &K, py: Python<'py>, f: &[K::Elem]) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(f.len());
    for c in f {
        items.push(k.to_py(py, c)?);
    }
    Ok(PyList::new(py, items)?.into_any())
}

pub fn out_polys<'py, K: Bridge>(k: &K, py: Python<'py>, fs: &[Vec<K::Elem>]) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(fs.len());
    for f in fs {
        items.push(out_poly(k, py, f)?);
    }
    Ok(PyList::new(py, items)?.into_any())
}

/// `[(polynomial, multiplicity), ...]`.
pub fn out_factors<'py, K: Bridge>(
    k: &K,
    py: Python<'py>,
    fs: &[(Vec<K::Elem>, usize)],
) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(fs.len());
    for (f, m) in fs {
        items.push(tuple(py, vec![out_poly(k, py, f)?, out_usize(py, *m)?])?);
    }
    Ok(PyList::new(py, items)?.into_any())
}

pub fn out_dict<'py, K: Bridge>(k: &K, py: Python<'py>, f: &BTreeMap<usize, K::Elem>) -> PyResult<Bound<'py, PyAny>> {
    let dict: Bound<'py, PyDict> = PyDict::new(py);
    // highest exponent first, as the Python dict is built
    for (e, c) in f.iter().rev() {
        dict.set_item(*e, k.to_py(py, c)?)?;
    }
    Ok(dict.into_any())
}

pub fn out_usize(py: Python<'_>, n: usize) -> PyResult<Bound<'_, PyAny>> {
    Ok(n.into_pyobject(py)?.into_any())
}

/// A Python `int` as an [`Int`] (through PyO3's `num-bigint` conversion and
/// the two's complement bytes).
pub fn extract_int(ob: &Bound<'_, PyAny>) -> PyResult<Int> {
    let n: BigInt = ob.extract::<BigInt>()?;
    Ok(Int::from_le_bytes(&n.to_signed_bytes_le()))
}

/// An [`Int`] as a Python `int`.
pub fn out_int<'py>(py: Python<'py>, n: &Int) -> PyResult<Bound<'py, PyAny>> {
    let big: BigInt = BigInt::from_signed_bytes_le(&n.to_le_bytes());
    Ok(big.into_pyobject(py)?.into_any())
}

pub fn tuple<'py>(py: Python<'py>, items: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyTuple::new(py, items)?.into_any())
}
