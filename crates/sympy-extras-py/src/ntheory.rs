//! `sympy_extras_rs.ntheory` and its submodules `primetest`, `generate`,
//! `factor_`: the functions of the Python modules with the same names.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use sympy_extras_core::ntheory::{factor_ as fa, generate as ge, primetest as pt};
use sympy_extras_core::numbers::Int;

use crate::convert::{err, extract_int, out_int};

fn int_list<'py>(py: Python<'py>, values: &[Int]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = values.iter().map(|v| out_int(py, v)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

#[pyfunction]
fn isprime(n: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(pt::isprime(&extract_int(n)?))
}

#[pyfunction]
fn is_strong_probable_prime(n: &Bound<'_, PyAny>, base: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(pt::is_strong_probable_prime(&extract_int(n)?, &extract_int(base)?))
}

#[pyfunction]
fn is_strong_lucas_probable_prime(n: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(pt::is_strong_lucas_probable_prime(&extract_int(n)?))
}

#[pyfunction]
fn jacobi_symbol(a: &Bound<'_, PyAny>, n: &Bound<'_, PyAny>) -> PyResult<i32> {
    pt::jacobi_symbol(&extract_int(a)?, &extract_int(n)?).map_err(err)
}

#[pyfunction]
fn nextprime<'py>(py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_int(py, &ge::nextprime(&extract_int(n)?))
}

#[pyfunction]
fn prevprime<'py>(py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    out_int(py, &ge::prevprime(&extract_int(n)?).map_err(err)?)
}

/// The primes in `[a, b)` (a list; Python's is a generator).
#[pyfunction]
fn primerange<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let primes: Vec<Int> = ge::primerange(&extract_int(a)?, &extract_int(b)?).collect();
    int_list(py, &primes)?.try_iter().map(|i| i.into_any())
}

#[pyfunction]
#[pyo3(signature = (n, limit = None))]
fn factorint<'py>(
    py: Python<'py>,
    n: &Bound<'py, PyAny>,
    limit: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let limit: Option<Int> = match limit {
        Some(l) if !l.is_none() => Some(extract_int(l)?),
        _ => None,
    };
    let dict: Bound<'py, PyDict> = PyDict::new(py);
    for (p, e) in fa::factorint(&extract_int(n)?, limit.as_ref()) {
        dict.set_item(out_int(py, &p)?, e)?;
    }
    Ok(dict.into_any())
}

#[pyfunction]
#[pyo3(signature = (n, seed = None))]
fn pollard_brent<'py>(
    py: Python<'py>,
    n: &Bound<'py, PyAny>,
    seed: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let seed: Int = match seed {
        Some(s) => extract_int(s)?,
        None => Int::ONE,
    };
    out_int(py, &fa::pollard_brent(&extract_int(n)?, &seed))
}

#[pyfunction]
fn perfect_power<'py>(py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    match fa::perfect_power(&extract_int(n)?) {
        Some((b, e)) => Ok(Some(crate::convert::tuple(
            py,
            vec![out_int(py, &b)?, e.into_pyobject(py)?.into_any()],
        )?)),
        None => Ok(None),
    }
}

#[pyfunction]
fn is_carmichael(n: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(fa::is_carmichael(&extract_int(n)?))
}

#[pyfunction]
fn find_carmichael_numbers_in_range<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    int_list(
        py,
        &fa::find_carmichael_numbers_in_range(&extract_int(x)?, &extract_int(y)?).map_err(err)?,
    )
}

#[pyfunction]
fn find_first_n_carmichaels(py: Python<'_>, n: usize) -> PyResult<Bound<'_, PyAny>> {
    int_list(py, &fa::find_first_n_carmichaels(n))
}

fn register_primetest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(isprime, m)?)?;
    m.add_function(wrap_pyfunction!(is_strong_probable_prime, m)?)?;
    m.add_function(wrap_pyfunction!(is_strong_lucas_probable_prime, m)?)?;
    m.add_function(wrap_pyfunction!(jacobi_symbol, m)?)?;
    m.add("SMALL_PRIMES", pt::small_primes().to_vec())?;
    Ok(())
}

fn register_generate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nextprime, m)?)?;
    m.add_function(wrap_pyfunction!(prevprime, m)?)?;
    m.add_function(wrap_pyfunction!(primerange, m)?)?;
    Ok(())
}

fn register_factor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factorint, m)?)?;
    m.add_function(wrap_pyfunction!(pollard_brent, m)?)?;
    m.add_function(wrap_pyfunction!(perfect_power, m)?)?;
    m.add_function(wrap_pyfunction!(is_carmichael, m)?)?;
    m.add_function(wrap_pyfunction!(find_carmichael_numbers_in_range, m)?)?;
    m.add_function(wrap_pyfunction!(find_first_n_carmichaels, m)?)?;
    Ok(())
}

/// `sympy_extras_rs.ntheory` with the public names of the package and the
/// three submodules.
pub fn register(
    ntheory: &Bound<'_, PyModule>,
    primetest: &Bound<'_, PyModule>,
    generate: &Bound<'_, PyModule>,
    factor: &Bound<'_, PyModule>,
) -> PyResult<()> {
    register_primetest(primetest)?;
    register_generate(generate)?;
    register_factor(factor)?;
    for name in ["isprime", "jacobi_symbol"] {
        ntheory.add(name, primetest.getattr(name)?)?;
    }
    for name in ["nextprime", "prevprime", "primerange"] {
        ntheory.add(name, generate.getattr(name)?)?;
    }
    for name in [
        "factorint",
        "find_carmichael_numbers_in_range",
        "find_first_n_carmichaels",
        "is_carmichael",
        "perfect_power",
        "pollard_brent",
    ] {
        ntheory.add(name, factor.getattr(name)?)?;
    }
    Ok(())
}
