//! `sympy_extras_rs.polys.regularchains.recursive` and `.operations`: the
//! recursive view of the polynomials over `ZZ` and the operations on
//! squarefree regular chains (`Chain`, `Decomposer`).

use std::sync::Arc;

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use sympy_extras_core::numbers::Int;
use sympy_extras_core::polys::regularchains::operations::{self as ops, Chain};
use sympy_extras_core::polys::regularchains::recursive::{self as rc, ZPoly};

use crate::convert::{err, out_int, tuple};
use crate::rings::{ElemKind, PyPolyElement, PyPolyRing, RingKind};

fn zpoly(f: &PyPolyElement) -> PyResult<ZPoly> {
    match &f.inner {
        ElemKind::ZZ(p) => Ok(p.clone()),
        _ => Err(PyTypeError::new_err("polynomials over ZZ are expected")),
    }
}

fn zpolys(ob: &Bound<'_, PyAny>) -> PyResult<Vec<ZPoly>> {
    let mut result: Vec<ZPoly> = Vec::new();
    for item in ob.try_iter()? {
        result.push(zpoly(item?.downcast::<PyPolyElement>()?.get())?);
    }
    Ok(result)
}

fn out(py: Python<'_>, p: ZPoly) -> PyResult<Bound<'_, PyAny>> {
    Ok(Bound::new(py, PyPolyElement { inner: ElemKind::ZZ(p) })?.into_any())
}

fn out_list<'py>(py: Python<'py>, polys: Vec<ZPoly>) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = polys.into_iter().map(|p| out(py, p)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

// -- recursive

#[pyfunction]
fn main_variable(p: &PyPolyElement) -> PyResult<usize> {
    Ok(rc::main_variable(&zpoly(p)?))
}

#[pyfunction]
fn degree(p: &PyPolyElement, i: usize) -> PyResult<i64> {
    Ok(rc::degree(&zpoly(p)?, i))
}

#[pyfunction]
fn coefficient<'py>(py: Python<'py>, p: &PyPolyElement, i: usize, d: i64) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::coefficient(&zpoly(p)?, i, d))
}

#[pyfunction]
fn initial<'py>(py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::initial(&zpoly(p)?))
}

#[pyfunction]
fn tail<'py>(py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::tail(&zpoly(p)?))
}

#[pyfunction]
fn normalize<'py>(py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::normalize(&zpoly(p)?))
}

#[pyfunction]
fn exact_quotient<'py>(py: Python<'py>, p: &PyPolyElement, q: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::exact_quotient(&zpoly(p)?, &zpoly(q)?).map_err(err)?)
}

#[pyfunction]
fn primitive_part<'py>(py: Python<'py>, p: &PyPolyElement, i: usize) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::primitive_part(&zpoly(p)?, i).map_err(err)?)
}

#[pyfunction]
fn pseudo_divide<'py>(py: Python<'py>, f: &PyPolyElement, g: &PyPolyElement, i: usize) -> PyResult<Bound<'py, PyAny>> {
    let (m, q, r) = rc::pseudo_divide(&zpoly(f)?, &zpoly(g)?, i).map_err(err)?;
    tuple(py, vec![out(py, m)?, out(py, q)?, out(py, r)?])
}

#[pyfunction]
fn pseudo_remainder<'py>(
    py: Python<'py>,
    f: &PyPolyElement,
    g: &PyPolyElement,
    i: usize,
) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::pseudo_remainder(&zpoly(f)?, &zpoly(g)?, i).map_err(err)?)
}

#[pyfunction]
fn prem<'py>(py: Python<'py>, f: &PyPolyElement, g: &PyPolyElement, i: usize) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::prem(&zpoly(f)?, &zpoly(g)?, i).map_err(err)?)
}

#[pyfunction]
fn subresultants<'py>(py: Python<'py>, f: &PyPolyElement, g: &PyPolyElement, i: usize) -> PyResult<Bound<'py, PyAny>> {
    out_list(py, rc::subresultants(&zpoly(f)?, &zpoly(g)?, i).map_err(err)?)
}

#[pyfunction]
fn subresultant_chain<'py>(
    py: Python<'py>,
    f: &PyPolyElement,
    g: &PyPolyElement,
    i: usize,
) -> PyResult<Bound<'py, PyAny>> {
    out_list(py, rc::subresultant_chain(&zpoly(f)?, &zpoly(g)?, i).map_err(err)?)
}

#[pyfunction]
fn distinct_factors<'py>(py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out_list(py, rc::distinct_factors(&zpoly(p)?).map_err(err)?)
}

#[pyfunction]
fn gcd<'py>(py: Python<'py>, p: &PyPolyElement, q: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::gcd(&zpoly(p)?, &zpoly(q)?))
}

#[pyfunction]
fn cofactors<'py>(py: Python<'py>, p: &PyPolyElement, q: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    let (h, a, b) = rc::cofactors(&zpoly(p)?, &zpoly(q)?).map_err(err)?;
    tuple(py, vec![out(py, h)?, out(py, a)?, out(py, b)?])
}

#[pyfunction]
fn sqf_part<'py>(py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out(py, rc::sqf_part(&zpoly(p)?).map_err(err)?)
}

#[pyfunction]
fn max_norm<'py>(py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    out_int(py, &rc::max_norm(&zpoly(p)?))
}

#[pyfunction]
fn sort_key<'py>(py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    let (length, terms) = rc::sort_key(&zpoly(p)?);
    let items: Vec<Bound<'py, PyAny>> = terms
        .iter()
        .map(|(m, c): &(Vec<u32>, Int)| tuple(py, vec![pyo3::types::PyTuple::new(py, m)?.into_any(), out_int(py, c)?]))
        .collect::<PyResult<_>>()?;
    tuple(
        py,
        vec![length.into_pyobject(py)?.into_any(), PyList::new(py, items)?.into_any()],
    )
}

pub fn register_recursive(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(main_variable, m)?)?;
    m.add_function(wrap_pyfunction!(degree, m)?)?;
    m.add_function(wrap_pyfunction!(coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(initial, m)?)?;
    m.add_function(wrap_pyfunction!(tail, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(exact_quotient, m)?)?;
    m.add_function(wrap_pyfunction!(primitive_part, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_divide, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_remainder, m)?)?;
    m.add_function(wrap_pyfunction!(prem, m)?)?;
    m.add_function(wrap_pyfunction!(subresultants, m)?)?;
    m.add_function(wrap_pyfunction!(subresultant_chain, m)?)?;
    m.add_function(wrap_pyfunction!(distinct_factors, m)?)?;
    m.add_function(wrap_pyfunction!(gcd, m)?)?;
    m.add_function(wrap_pyfunction!(cofactors, m)?)?;
    m.add_function(wrap_pyfunction!(sqf_part, m)?)?;
    m.add_function(wrap_pyfunction!(max_norm, m)?)?;
    m.add_function(wrap_pyfunction!(sort_key, m)?)?;
    Ok(())
}

// -- operations

/// A triangular set: at most one polynomial per main variable.
#[pyclass(name = "Chain", module = "sympy_extras_rs.polys.regularchains.operations", frozen)]
#[derive(Clone)]
pub struct PyChain {
    inner: Chain,
}

fn chain_of(inner: Chain) -> PyChain {
    PyChain { inner }
}

fn chains<'py>(py: Python<'py>, items: Vec<Chain>) -> PyResult<Bound<'py, PyAny>> {
    let objects: Vec<Bound<'py, PyAny>> = items
        .into_iter()
        .map(|c| Ok(Bound::new(py, chain_of(c))?.into_any()))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objects)?.into_any())
}

fn regularized<'py>(py: Python<'py>, items: Vec<ops::Regularized>) -> PyResult<Bound<'py, PyAny>> {
    let objects: Vec<Bound<'py, PyAny>> = items
        .into_iter()
        .map(|(c, zero)| {
            tuple(
                py,
                vec![
                    Bound::new(py, chain_of(c))?.into_any(),
                    zero.into_pyobject(py)?.to_owned().into_any(),
                ],
            )
        })
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objects)?.into_any())
}

#[pymethods]
impl PyChain {
    #[new]
    fn new(polys: &Bound<'_, PyAny>) -> PyResult<PyChain> {
        let mut values: Vec<Option<ZPoly>> = Vec::new();
        for item in polys.try_iter()? {
            let item = item?;
            values.push(if item.is_none() {
                None
            } else {
                Some(zpoly(item.downcast::<PyPolyElement>()?.get())?)
            });
        }
        Ok(chain_of(Chain::new(values)))
    }

    #[getter]
    fn polys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Bound<'py, PyAny>> = self
            .inner
            .polys
            .iter()
            .map(|p| match p {
                Some(p) => out(py, p.clone()),
                None => Ok(py.None().into_bound(py)),
            })
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyTuple::new(py, items)?.into_any())
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height
    }

    fn get<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.get(i) {
            Some(p) => out(py, p.clone()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn below(&self, i: usize) -> PyChain {
        chain_of(self.inner.below(i))
    }

    fn above<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        out_list(py, self.inner.above(i))
    }

    fn with_poly(&self, i: usize, p: &PyPolyElement) -> PyResult<PyChain> {
        Ok(chain_of(self.inner.with_poly(i, zpoly(p)?)))
    }

    fn union(&self, polys: &Bound<'_, PyAny>) -> PyResult<PyChain> {
        Ok(chain_of(self.inner.union(&zpolys(polys)?)))
    }

    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        out_list(py, self.inner.elements())
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyObject> {
        let py = other.py();
        let Ok(o) = other.downcast::<PyChain>() else {
            return Ok(py.NotImplemented());
        };
        let equal: bool = self.inner == o.get().inner;
        match op {
            CompareOp::Eq => Ok(equal.into_pyobject(py)?.to_owned().into_any().unbind()),
            CompareOp::Ne => Ok((!equal).into_pyobject(py)?.to_owned().into_any().unbind()),
            _ => Ok(py.NotImplemented()),
        }
    }
}

/// The operations on the squarefree regular chains of a ring over `ZZ`,
/// with their caches.
#[pyclass(name = "Decomposer", module = "sympy_extras_rs.polys.regularchains.operations")]
pub struct PyDecomposer {
    inner: ops::Decomposer,
}

#[pymethods]
impl PyDecomposer {
    #[new]
    #[pyo3(signature = (ring, bound = None))]
    fn new(ring: &PyPolyRing, bound: Option<usize>) -> PyResult<PyDecomposer> {
        match &ring.inner {
            RingKind::ZZ(r) => Ok(PyDecomposer {
                inner: ops::Decomposer::new(Arc::clone(r), bound),
            }),
            _ => Err(PyTypeError::new_err("a ring over ZZ is expected")),
        }
    }

    #[getter]
    fn empty(&self) -> PyChain {
        chain_of(self.inner.empty.clone())
    }

    #[getter]
    fn bound(&self) -> Option<usize> {
        self.inner.bound
    }

    fn kept(&self, chain: &PyChain) -> bool {
        self.inner.kept(&chain.inner)
    }

    fn factors<'py>(&mut self, py: Python<'py>, p: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
        out_list(py, self.inner.factors(&zpoly(p)?).map_err(err)?)
    }

    fn reduce<'py>(&self, py: Python<'py>, p: &PyPolyElement, chain: &PyChain) -> PyResult<Bound<'py, PyAny>> {
        out(py, self.inner.reduce(&zpoly(p)?, &chain.inner).map_err(err)?)
    }

    fn regularize<'py>(&mut self, py: Python<'py>, p: &PyPolyElement, chain: &PyChain) -> PyResult<Bound<'py, PyAny>> {
        regularized(py, self.inner.regularize(&zpoly(p)?, &chain.inner).map_err(err)?)
    }

    fn regular_gcd<'py>(
        &mut self,
        py: Python<'py>,
        p: &PyPolyElement,
        q: &PyPolyElement,
        v: usize,
        chain: &PyChain,
    ) -> PyResult<Bound<'py, PyAny>> {
        let found = self
            .inner
            .regular_gcd(&zpoly(p)?, &zpoly(q)?, v, &chain.inner)
            .map_err(err)?;
        let items: Vec<Bound<'py, PyAny>> = found
            .into_iter()
            .map(|(g, c, lost)| {
                let g: Bound<'py, PyAny> = match g {
                    Some(g) => out(py, g)?,
                    None => py.None().into_bound(py),
                };
                tuple(py, vec![g, Bound::new(py, chain_of(c))?.into_any(), out(py, lost)?])
            })
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into_any())
    }

    fn add<'py>(&mut self, py: Python<'py>, chain: &PyChain, f: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
        chains(py, self.inner.add(&chain.inner, &zpoly(f)?).map_err(err)?)
    }

    fn extend<'py>(
        &mut self,
        py: Python<'py>,
        chain: &PyChain,
        polys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        chains(py, self.inner.extend(&chain.inner, &zpolys(polys)?).map_err(err)?)
    }

    fn intersect<'py>(&mut self, py: Python<'py>, p: &PyPolyElement, chain: &PyChain) -> PyResult<Bound<'py, PyAny>> {
        chains(py, self.inner.intersect(&zpoly(p)?, &chain.inner).map_err(err)?)
    }

    fn is_regular(&mut self, p: &PyPolyElement, chain: &PyChain) -> PyResult<bool> {
        self.inner.is_regular(&zpoly(p)?, &chain.inner).map_err(err)
    }

    #[pyo3(signature = (first, second, closures = false))]
    fn inside(&mut self, first: &PyChain, second: &PyChain, closures: bool) -> PyResult<bool> {
        self.inner.inside(&first.inner, &second.inner, closures).map_err(err)
    }

    #[pyo3(signature = (chain_list, closures = false))]
    fn irredundant<'py>(
        &mut self,
        py: Python<'py>,
        chain_list: &Bound<'py, PyAny>,
        closures: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut values: Vec<Chain> = Vec::new();
        for item in chain_list.try_iter()? {
            values.push(item?.downcast::<PyChain>()?.get().inner.clone());
        }
        chains(py, self.inner.irredundant(&values, closures).map_err(err)?)
    }

    #[pyo3(signature = (equations, inequations = None))]
    fn triangularize<'py>(
        &mut self,
        py: Python<'py>,
        equations: &Bound<'py, PyAny>,
        inequations: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inequations: Vec<ZPoly> = match inequations {
            Some(i) => zpolys(i)?,
            None => Vec::new(),
        };
        chains(
            py,
            self.inner
                .triangularize(&zpolys(equations)?, &inequations)
                .map_err(err)?,
        )
    }
}

pub fn register_operations(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChain>()?;
    m.add_class::<PyDecomposer>()?;
    Ok(())
}
