//! `sympy_extras_rs.polys.rings` and `sympy_extras_rs.polys.heuristicgcd`:
//! `PolyRing`, `PolyElement` and the orders, with the attributes and methods
//! of the Python classes. A Python class cannot be generic, so the ring and
//! its elements hold one of the three domains.

// the Python names (`LM`, `LC`, the argument `K`)
#![allow(non_snake_case)]

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use sympy_extras_core::numbers::Int;
use sympy_extras_core::polys::domains::{FiniteField, IntegerRing, RationalField};
use sympy_extras_core::polys::heuristicgcd as hg;
use sympy_extras_core::polys::rings::{self, Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

use crate::convert::{err, extract_int, tuple, Bridge};
use crate::domains::{DomainKind, PyDomain};

/// A monomial order.
#[pyclass(name = "MonomialOrder", module = "sympy_extras_rs.polys.rings", frozen, eq, hash)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyOrder {
    pub order: MonomialOrder,
}

fn order_from(ob: &Bound<'_, PyAny>) -> PyResult<MonomialOrder> {
    if let Ok(order) = ob.downcast::<PyOrder>() {
        return Ok(order.get().order.clone());
    }
    match ob.extract::<String>()?.as_str() {
        "lex" => Ok(MonomialOrder::Lex),
        "grlex" => Ok(MonomialOrder::Grlex),
        "grevlex" => Ok(MonomialOrder::Grevlex),
        other => Err(PyValueError::new_err(format!("unknown monomial order {}", other))),
    }
}

#[pymethods]
impl PyOrder {
    #[getter]
    fn name(&self) -> String {
        self.order.to_string()
    }

    fn __repr__(&self) -> String {
        self.order.to_string()
    }

    fn is_graded(&self, ngens: usize) -> bool {
        self.order.is_graded(ngens)
    }

    /// Compares two monomials: `-1`, `0` or `1`.
    fn compare(&self, a: Vec<u32>, b: Vec<u32>) -> i32 {
        match self.order.cmp(&a, &b) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }

    /// `orderings.WeightOrder(weights, tail)`.
    #[staticmethod]
    #[pyo3(signature = (weights, tail = None))]
    fn weight(weights: &Bound<'_, PyAny>, tail: Option<&Bound<'_, PyAny>>) -> PyResult<PyOrder> {
        let mut values: Vec<Int> = Vec::new();
        for w in weights.try_iter()? {
            values.push(extract_int(&w?)?);
        }
        let tail: MonomialOrder = match tail {
            Some(t) => order_from(t)?,
            None => MonomialOrder::Lex,
        };
        Ok(PyOrder {
            order: MonomialOrder::weight(values, tail).map_err(err)?,
        })
    }

    /// `orderings.BlockOrder(blocks)`: pairs `(order, size)`.
    #[staticmethod]
    fn block(blocks: &Bound<'_, PyAny>) -> PyResult<PyOrder> {
        let mut values: Vec<(MonomialOrder, usize)> = Vec::new();
        for item in blocks.try_iter()? {
            let item = item?;
            values.push((order_from(&item.get_item(0)?)?, item.get_item(1)?.extract::<usize>()?));
        }
        Ok(PyOrder {
            order: MonomialOrder::block(values).map_err(err)?,
        })
    }
}

#[derive(Debug, Clone)]
pub enum RingKind {
    ZZ(Arc<PolyRing<IntegerRing>>),
    QQ(Arc<PolyRing<RationalField>>),
    GF(Arc<PolyRing<FiniteField>>),
}

#[derive(Debug, Clone)]
pub enum ElemKind {
    ZZ(PolyElement<IntegerRing>),
    QQ(PolyElement<RationalField>),
    GF(PolyElement<FiniteField>),
}

/// Runs `$body` with `$f` the element of its domain; the result keeps the
/// kind.
macro_rules! map_elem {
    ($e:expr, $f:ident => $body:expr) => {
        match $e {
            ElemKind::ZZ($f) => ElemKind::ZZ($body),
            ElemKind::QQ($f) => ElemKind::QQ($body),
            ElemKind::GF($f) => ElemKind::GF($body),
        }
    };
}

/// Runs `$body` with `$f` the element of its domain.
macro_rules! with_elem {
    ($e:expr, $f:ident => $body:expr) => {
        match $e {
            ElemKind::ZZ($f) => $body,
            ElemKind::QQ($f) => $body,
            ElemKind::GF($f) => $body,
        }
    };
}

/// Runs `$body` on two elements of the same domain (`TypeError`
/// otherwise); the result keeps the kind.
macro_rules! binary_elem {
    ($a:expr, $b:expr, ($f:ident, $g:ident) => $body:expr) => {
        match ($a, $b) {
            (ElemKind::ZZ($f), ElemKind::ZZ($g)) => Ok(ElemKind::ZZ($body)),
            (ElemKind::QQ($f), ElemKind::QQ($g)) => Ok(ElemKind::QQ($body)),
            (ElemKind::GF($f), ElemKind::GF($g)) => Ok(ElemKind::GF($body)),
            _ => Err(PyTypeError::new_err("polynomials of different rings")),
        }
    };
}

macro_rules! with_ring {
    ($r:expr, $ring:ident => $body:expr) => {
        match $r {
            RingKind::ZZ($ring) => $body,
            RingKind::QQ($ring) => $body,
            RingKind::GF($ring) => $body,
        }
    };
}

/// A polynomial ring.
#[pyclass(name = "PolyRing", module = "sympy_extras_rs.polys.rings", frozen)]
#[derive(Debug, Clone)]
pub struct PyPolyRing {
    pub inner: RingKind,
}

/// A polynomial.
#[pyclass(name = "PolyElement", module = "sympy_extras_rs.polys.rings", frozen)]
#[derive(Debug, Clone)]
pub struct PyPolyElement {
    pub inner: ElemKind,
}

fn elem(inner: ElemKind) -> PyPolyElement {
    PyPolyElement { inner }
}

fn monomial_to_py<'py>(py: Python<'py>, m: &[u32]) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyTuple::new(py, m)?.into_any())
}

fn terms_from_py<K: Bridge>(k: &K, ob: &Bound<'_, PyAny>) -> PyResult<Terms<K::Elem>> {
    let dict: &Bound<'_, PyDict> = ob.downcast::<PyDict>()?;
    let mut terms: Terms<K::Elem> = Terms::new();
    for (m, c) in dict.iter() {
        terms.insert(m.extract::<Monomial>()?, k.extract(&c)?);
    }
    Ok(terms)
}

/// The terms as a dict in decreasing order of the monomials.
fn terms_to_py<'py, K: Bridge>(py: Python<'py>, f: &PolyElement<K>) -> PyResult<Bound<'py, PyDict>> {
    let dict: Bound<'py, PyDict> = PyDict::new(py);
    for (m, c) in f.ordered_terms() {
        dict.set_item(monomial_to_py(py, &m)?, f.ring.domain.to_py(py, &c)?)?;
    }
    Ok(dict)
}

fn ring_of(e: &ElemKind) -> RingKind {
    match e {
        ElemKind::ZZ(f) => RingKind::ZZ(Arc::clone(&f.ring)),
        ElemKind::QQ(f) => RingKind::QQ(Arc::clone(&f.ring)),
        ElemKind::GF(f) => RingKind::GF(Arc::clone(&f.ring)),
    }
}

fn domain_of(r: &RingKind) -> PyDomain {
    match r {
        RingKind::ZZ(_) => PyDomain::base(DomainKind::ZZ),
        RingKind::QQ(_) => PyDomain::base(DomainKind::QQ),
        RingKind::GF(ring) => PyDomain::base(DomainKind::GF(ring.domain.p())),
    }
}

fn elements(ob: &Bound<'_, PyAny>) -> PyResult<Vec<ElemKind>> {
    let mut result: Vec<ElemKind> = Vec::new();
    for item in ob.try_iter()? {
        result.push(item?.downcast::<PyPolyElement>()?.get().inner.clone());
    }
    Ok(result)
}

/// The elements of one kind (`TypeError` for a mixture).
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

fn elem_list<'py>(py: Python<'py>, items: Vec<ElemKind>) -> PyResult<Bound<'py, PyAny>> {
    let objects: Vec<Bound<'py, PyAny>> = items
        .into_iter()
        .map(|e| Ok(Bound::new(py, elem(e))?.into_any()))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objects)?.into_any())
}

#[pymethods]
impl PyPolyRing {
    #[new]
    #[pyo3(signature = (domain, symbols, order = None))]
    fn new(domain: &PyDomain, symbols: Vec<String>, order: Option<&Bound<'_, PyAny>>) -> PyResult<PyPolyRing> {
        let order: MonomialOrder = match order {
            Some(o) => order_from(o)?,
            None => MonomialOrder::Lex,
        };
        let inner: RingKind = match domain.kind {
            DomainKind::ZZ => RingKind::ZZ(PolyRing::new(IntegerRing, symbols, order).map_err(err)?),
            DomainKind::QQ => RingKind::QQ(PolyRing::new(RationalField, symbols, order).map_err(err)?),
            DomainKind::GF(p) => {
                RingKind::GF(PolyRing::new(FiniteField::new(p).map_err(err)?, symbols, order).map_err(err)?)
            }
            _ => return Err(PyTypeError::new_err("the domain must be ZZ, QQ or GF(p)")),
        };
        Ok(PyPolyRing { inner })
    }

    #[getter]
    fn domain(&self) -> PyDomain {
        domain_of(&self.inner)
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        with_ring!(&self.inner, r => r.names.clone())
    }

    #[getter]
    fn ngens(&self) -> usize {
        with_ring!(&self.inner, r => r.ngens)
    }

    #[getter]
    fn order(&self) -> PyOrder {
        with_ring!(&self.inner, r => PyOrder { order: r.order.clone() })
    }

    #[getter]
    fn zero_monom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_ring!(&self.inner, r => monomial_to_py(py, &r.zero_monom))
    }

    fn __repr__(&self) -> String {
        with_ring!(&self.inner, r => r.name())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.downcast::<PyPolyRing>() {
            Ok(o) => match (&self.inner, &o.get().inner) {
                (RingKind::ZZ(a), RingKind::ZZ(b)) => a == b,
                (RingKind::QQ(a), RingKind::QQ(b)) => a == b,
                (RingKind::GF(a), RingKind::GF(b)) => a == b,
                _ => false,
            },
            Err(_) => false,
        }
    }

    #[getter]
    fn zero(&self) -> PyPolyElement {
        elem(match &self.inner {
            RingKind::ZZ(r) => ElemKind::ZZ(r.zero()),
            RingKind::QQ(r) => ElemKind::QQ(r.zero()),
            RingKind::GF(r) => ElemKind::GF(r.zero()),
        })
    }

    #[getter]
    fn one(&self) -> PyPolyElement {
        elem(match &self.inner {
            RingKind::ZZ(r) => ElemKind::ZZ(r.one()),
            RingKind::QQ(r) => ElemKind::QQ(r.one()),
            RingKind::GF(r) => ElemKind::GF(r.one()),
        })
    }

    #[getter]
    fn gens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<ElemKind> = match &self.inner {
            RingKind::ZZ(r) => r.gens().into_iter().map(ElemKind::ZZ).collect(),
            RingKind::QQ(r) => r.gens().into_iter().map(ElemKind::QQ).collect(),
            RingKind::GF(r) => r.gens().into_iter().map(ElemKind::GF).collect(),
        };
        elem_list(py, items)
    }

    fn term(&self, monomial: Monomial, coefficient: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
        Ok(elem(match &self.inner {
            RingKind::ZZ(r) => ElemKind::ZZ(r.term(monomial, r.domain.extract(coefficient)?)),
            RingKind::QQ(r) => ElemKind::QQ(r.term(monomial, r.domain.extract(coefficient)?)),
            RingKind::GF(r) => ElemKind::GF(r.term(monomial, r.domain.extract(coefficient)?)),
        }))
    }

    fn ground(&self, coefficient: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
        let zero: Monomial = with_ring!(&self.inner, r => r.zero_monom.clone());
        self.term(zero, coefficient)
    }

    fn from_int(&self, n: i64) -> PyPolyElement {
        elem(match &self.inner {
            RingKind::ZZ(r) => ElemKind::ZZ(r.from_int(n)),
            RingKind::QQ(r) => ElemKind::QQ(r.from_int(n)),
            RingKind::GF(r) => ElemKind::GF(r.from_int(n)),
        })
    }

    fn from_dict(&self, terms: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
        Ok(elem(match &self.inner {
            RingKind::ZZ(r) => ElemKind::ZZ(r.from_dict(terms_from_py(&r.domain, terms)?)),
            RingKind::QQ(r) => ElemKind::QQ(r.from_dict(terms_from_py(&r.domain, terms)?)),
            RingKind::GF(r) => ElemKind::GF(r.from_dict(terms_from_py(&r.domain, terms)?)),
        }))
    }

    fn with_order(&self, order: &Bound<'_, PyAny>) -> PyResult<PyPolyRing> {
        let order: MonomialOrder = order_from(order)?;
        Ok(PyPolyRing {
            inner: match &self.inner {
                RingKind::ZZ(r) => RingKind::ZZ(r.with_order(order)),
                RingKind::QQ(r) => RingKind::QQ(r.with_order(order)),
                RingKind::GF(r) => RingKind::GF(r.with_order(order)),
            },
        })
    }
}

#[pymethods]
impl PyPolyElement {
    #[getter]
    fn ring(&self) -> PyPolyRing {
        PyPolyRing {
            inner: ring_of(&self.inner),
        }
    }

    #[getter]
    fn terms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        with_elem!(&self.inner, f => terms_to_py(py, f))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let terms: String = self.terms(py)?.repr()?.extract::<String>()?;
        let ring: String = with_elem!(&self.inner, f => f.ring.name());
        Ok(format!("PolyElement({}, {})", terms, ring))
    }

    fn __str__(&self) -> String {
        with_elem!(&self.inner, f => f.to_string())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.downcast::<PyPolyElement>() {
            Ok(o) => match (&self.inner, &o.get().inner) {
                (ElemKind::ZZ(a), ElemKind::ZZ(b)) => a == b,
                (ElemKind::QQ(a), ElemKind::QQ(b)) => a == b,
                (ElemKind::GF(a), ElemKind::GF(b)) => a == b,
                _ => false,
            },
            Err(_) => false,
        }
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        let items = self.terms(py)?.call_method0("items")?;
        let frozen = py.import("builtins")?.getattr("frozenset")?.call1((items,))?;
        let key = PyTuple::new(py, [frozen.into_any(), self.__str__().into_pyobject(py)?.into_any()])?;
        key.hash()
    }

    fn __bool__(&self) -> bool {
        with_elem!(&self.inner, f => !f.is_zero())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.terms(py)?.call_method0("items")?.try_iter().map(|i| i.into_any())
    }

    fn copy(&self) -> PyPolyElement {
        self.clone()
    }

    fn monomials<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ms: Vec<Monomial> = with_elem!(&self.inner, f => f.monomials());
        let items: Vec<Bound<'py, PyAny>> = ms.iter().map(|m| monomial_to_py(py, m)).collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into_any())
    }

    #[getter]
    fn LM<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_elem!(&self.inner, f => monomial_to_py(py, &f.LM()))
    }

    #[getter]
    fn LC<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_elem!(&self.inner, f => f.ring.domain.to_py(py, &f.LC()))
    }

    #[getter]
    fn LT<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_elem!(&self.inner, f => {
            let (m, c) = f.LT();
            tuple(py, vec![monomial_to_py(py, &m)?, f.ring.domain.to_py(py, &c)?])
        })
    }

    fn degree(&self) -> i64 {
        with_elem!(&self.inner, f => f.degree())
    }

    fn degree_in(&self, i: usize) -> i64 {
        with_elem!(&self.inner, f => f.degree_in(i))
    }

    #[getter]
    fn is_zero(&self) -> bool {
        with_elem!(&self.inner, f => f.is_zero())
    }

    #[getter]
    fn is_ground(&self) -> bool {
        with_elem!(&self.inner, f => f.is_ground())
    }

    #[getter]
    fn is_monic(&self) -> bool {
        with_elem!(&self.inner, f => f.is_monic())
    }

    fn coeff<'py>(&self, py: Python<'py>, monomial: Monomial) -> PyResult<Bound<'py, PyAny>> {
        with_elem!(&self.inner, f => f.ring.domain.to_py(py, &f.coeff(&monomial)))
    }

    fn __neg__(&self) -> PyPolyElement {
        elem(map_elem!(&self.inner, f => f.neg()))
    }

    fn __add__(&self, other: &PyPolyElement) -> PyResult<PyPolyElement> {
        Ok(elem(binary_elem!(&self.inner, &other.inner, (f, g) => f.add(g))?))
    }

    fn __sub__(&self, other: &PyPolyElement) -> PyResult<PyPolyElement> {
        Ok(elem(binary_elem!(&self.inner, &other.inner, (f, g) => f.sub(g))?))
    }

    fn __mul__(&self, other: &PyPolyElement) -> PyResult<PyPolyElement> {
        Ok(elem(binary_elem!(&self.inner, &other.inner, (f, g) => f.mul(g))?))
    }

    fn __pow__(&self, n: i64, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyPolyElement> {
        if n < 0 {
            return Err(PyValueError::new_err("negative exponent"));
        }
        Ok(elem(map_elem!(&self.inner, f => f.pow(n as u64))))
    }

    fn mul_term(&self, monomial: Monomial, coefficient: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
        Ok(elem(
            map_elem!(&self.inner, f => f.mul_term(&monomial, &f.ring.domain.extract(coefficient)?)),
        ))
    }

    fn mul_ground(&self, coefficient: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
        Ok(elem(
            map_elem!(&self.inner, f => f.mul_ground(&f.ring.domain.extract(coefficient)?)),
        ))
    }

    fn mul_monom(&self, monomial: Monomial) -> PyPolyElement {
        elem(map_elem!(&self.inner, f => f.mul_monom(&monomial)))
    }

    fn exquo_ground(&self, coefficient: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
        Ok(elem(
            map_elem!(&self.inner, f => f.exquo_ground(&f.ring.domain.extract(coefficient)?).map_err(err)?),
        ))
    }

    fn monic(&self) -> PyResult<PyPolyElement> {
        Ok(elem(map_elem!(&self.inner, f => f.monic().map_err(err)?)))
    }

    fn primitive<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            ElemKind::ZZ(f) => {
                let (c, p) = f.primitive();
                tuple(
                    py,
                    vec![
                        f.ring.domain.to_py(py, &c)?,
                        Bound::new(py, elem(ElemKind::ZZ(p)))?.into_any(),
                    ],
                )
            }
            ElemKind::QQ(f) => {
                let (c, p) = f.primitive();
                tuple(
                    py,
                    vec![
                        f.ring.domain.to_py(py, &c)?,
                        Bound::new(py, elem(ElemKind::QQ(p)))?.into_any(),
                    ],
                )
            }
            ElemKind::GF(f) => {
                let (c, p) = f.primitive();
                tuple(
                    py,
                    vec![
                        f.ring.domain.to_py(py, &c)?,
                        Bound::new(py, elem(ElemKind::GF(p)))?.into_any(),
                    ],
                )
            }
        }
    }

    fn div<'py>(&self, py: Python<'py>, divisors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let divisors: Vec<ElemKind> = elements(divisors)?;
        let (quotients, remainder): (Vec<ElemKind>, ElemKind) = match &self.inner {
            ElemKind::ZZ(f) => {
                let (q, r) = f.div(&same_kind!(&divisors, ZZ)).map_err(err)?;
                (q.into_iter().map(ElemKind::ZZ).collect(), ElemKind::ZZ(r))
            }
            ElemKind::QQ(f) => {
                let (q, r) = f.div(&same_kind!(&divisors, QQ)).map_err(err)?;
                (q.into_iter().map(ElemKind::QQ).collect(), ElemKind::QQ(r))
            }
            ElemKind::GF(f) => {
                let (q, r) = f.div(&same_kind!(&divisors, GF)).map_err(err)?;
                (q.into_iter().map(ElemKind::GF).collect(), ElemKind::GF(r))
            }
        };
        tuple(
            py,
            vec![elem_list(py, quotients)?, Bound::new(py, elem(remainder))?.into_any()],
        )
    }

    fn rem<'py>(&self, py: Python<'py>, divisors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.div(py, divisors)?.get_item(1)
    }

    fn exquo(&self, other: &PyPolyElement) -> PyResult<PyPolyElement> {
        let result: ElemKind = binary_elem!(&self.inner, &other.inner, (f, g) => f.exquo(g).map_err(err)?)?;
        Ok(elem(result))
    }

    fn diff(&self, i: usize) -> PyPolyElement {
        elem(map_elem!(&self.inner, f => f.diff(i)))
    }

    fn evaluate(&self, i: usize, a: &Bound<'_, PyAny>) -> PyResult<PyPolyElement> {
        Ok(elem(
            map_elem!(&self.inner, f => f.evaluate(i, &f.ring.domain.extract(a)?)),
        ))
    }
}

fn integer_elem(f: &PyPolyElement) -> PyResult<PolyElement<IntegerRing>> {
    match &f.inner {
        ElemKind::ZZ(f) => Ok(f.clone()),
        _ => Err(PyTypeError::new_err("polynomials over ZZ are expected")),
    }
}

fn zz<'py>(py: Python<'py>, f: PolyElement<IntegerRing>) -> PyResult<Bound<'py, PyAny>> {
    Ok(Bound::new(py, elem(ElemKind::ZZ(f)))?.into_any())
}

/// `heugcd(f, g)`: `(h, f/h, g/h)` over `ZZ`.
#[pyfunction]
fn heugcd<'py>(py: Python<'py>, f: &PyPolyElement, g: &PyPolyElement) -> PyResult<Bound<'py, PyAny>> {
    let (h, cff, cfg) = hg::heugcd(&integer_elem(f)?, &integer_elem(g)?).map_err(err)?;
    tuple(py, vec![zz(py, h)?, zz(py, cff)?, zz(py, cfg)?])
}

#[pyfunction]
fn deflate<'py>(py: Python<'py>, polys: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<ElemKind> = elements(polys)?;
    let integers: Vec<PolyElement<IntegerRing>> = same_kind!(&items, ZZ);
    if integers.is_empty() {
        return Err(PyValueError::new_err("deflate needs at least one polynomial"));
    }
    let (steps, deflated) = hg::deflate(&integers);
    tuple(
        py,
        vec![
            monomial_to_py(py, &steps)?,
            elem_list(py, deflated.into_iter().map(ElemKind::ZZ).collect())?,
        ],
    )
}

#[pyfunction]
fn inflate(f: &PyPolyElement, steps: Monomial) -> PyResult<PyPolyElement> {
    Ok(elem(ElemKind::ZZ(hg::inflate(&integer_elem(f)?, &steps))))
}

#[pyfunction]
fn monomial_mul<'py>(py: Python<'py>, a: Monomial, b: Monomial) -> PyResult<Bound<'py, PyAny>> {
    monomial_to_py(py, &rings::monomial_mul(&a, &b))
}

#[pyfunction]
fn monomial_div<'py>(py: Python<'py>, a: Monomial, b: Monomial) -> PyResult<Bound<'py, PyAny>> {
    monomial_to_py(py, &rings::monomial_div(&a, &b).map_err(err)?)
}

#[pyfunction]
fn monomial_divides(a: Monomial, b: Monomial) -> bool {
    rings::monomial_divides(&a, &b)
}

#[pyfunction]
fn monomial_lcm<'py>(py: Python<'py>, a: Monomial, b: Monomial) -> PyResult<Bound<'py, PyAny>> {
    monomial_to_py(py, &rings::monomial_lcm(&a, &b))
}

#[pyfunction]
fn monomial_gcd<'py>(py: Python<'py>, a: Monomial, b: Monomial) -> PyResult<Bound<'py, PyAny>> {
    monomial_to_py(py, &rings::monomial_gcd(&a, &b))
}

pub fn register_rings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrder>()?;
    m.add_class::<PyPolyRing>()?;
    m.add_class::<PyPolyElement>()?;
    m.add(
        "lex",
        PyOrder {
            order: MonomialOrder::Lex,
        },
    )?;
    m.add(
        "grlex",
        PyOrder {
            order: MonomialOrder::Grlex,
        },
    )?;
    m.add(
        "grevlex",
        PyOrder {
            order: MonomialOrder::Grevlex,
        },
    )?;
    m.add_function(wrap_pyfunction!(monomial_mul, m)?)?;
    m.add_function(wrap_pyfunction!(monomial_div, m)?)?;
    m.add_function(wrap_pyfunction!(monomial_divides, m)?)?;
    m.add_function(wrap_pyfunction!(monomial_lcm, m)?)?;
    m.add_function(wrap_pyfunction!(monomial_gcd, m)?)?;
    Ok(())
}

pub fn register_heuristicgcd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("HEU_GCD_MAX", hg::HEU_GCD_MAX)?;
    m.add_function(wrap_pyfunction!(heugcd, m)?)?;
    m.add_function(wrap_pyfunction!(deflate, m)?)?;
    m.add_function(wrap_pyfunction!(inflate, m)?)?;
    Ok(())
}
