//! `sympy_extras_rs.polys.domains`: the domains `ZZ`, `QQ` and `GF(p)` as
//! one Python class, with the methods of the Python domains.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sympy_extras_core::polys::domains::{Domain, FiniteField, IntegerRing, RationalField};

use crate::convert::{err, out_elem, tuple, Bridge};

/// Which domain a Python `Domain` object stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainKind {
    ZZ,
    QQ,
    GF(u64),
}

/// A coefficient domain.
#[pyclass(name = "Domain", module = "sympy_extras_rs.polys.domains", frozen, eq, hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyDomain {
    pub kind: DomainKind,
}

/// Runs `$body` with `$k` bound to the Rust domain of a `PyDomain`.
#[macro_export]
macro_rules! with_domain {
    ($dom:expr, $k:ident => $body:expr) => {
        match $dom.kind {
            $crate::domains::DomainKind::ZZ => {
                let $k = &sympy_extras_core::polys::domains::IntegerRing;
                $body
            }
            $crate::domains::DomainKind::QQ => {
                let $k = &sympy_extras_core::polys::domains::RationalField;
                $body
            }
            $crate::domains::DomainKind::GF(p) => {
                let field = sympy_extras_core::polys::domains::FiniteField::new(p).map_err($crate::convert::err)?;
                let $k = &field;
                $body
            }
        }
    };
}

fn rust_name(kind: DomainKind) -> String {
    match kind {
        DomainKind::ZZ => IntegerRing.name(),
        DomainKind::QQ => RationalField.name(),
        DomainKind::GF(p) => format!("GF({})", p),
    }
}

#[pymethods]
impl PyDomain {
    #[getter]
    fn name(&self) -> String {
        rust_name(self.kind)
    }

    #[getter]
    fn is_field(&self) -> PyResult<bool> {
        with_domain!(self, k => Ok(k.is_field()))
    }

    #[getter]
    fn is_ordered(&self) -> PyResult<bool> {
        with_domain!(self, k => Ok(k.is_ordered()))
    }

    #[getter]
    fn characteristic(&self) -> PyResult<u64> {
        with_domain!(self, k => Ok(k.characteristic()))
    }

    /// The modulus of `GF(p)`.
    #[getter]
    fn p(&self) -> PyResult<u64> {
        match self.kind {
            DomainKind::GF(p) => Ok(p),
            _ => Err(pyo3::exceptions::PyAttributeError::new_err("p")),
        }
    }

    #[getter]
    fn zero<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.zero()))
    }

    #[getter]
    fn one<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.one()))
    }

    fn __repr__(&self) -> String {
        rust_name(self.kind)
    }

    fn __str__(&self) -> String {
        rust_name(self.kind)
    }

    fn is_zero(&self, a: &Bound<'_, PyAny>) -> PyResult<bool> {
        with_domain!(self, k => Ok(k.is_zero(&k.extract(a)?)))
    }

    fn is_one(&self, a: &Bound<'_, PyAny>) -> PyResult<bool> {
        with_domain!(self, k => Ok(k.is_one(&k.extract(a)?)))
    }

    fn is_negative(&self, a: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.ordered()?;
        with_domain!(self, k => Ok(k.is_negative(&k.extract(a)?)))
    }

    fn is_unit(&self, a: &Bound<'_, PyAny>) -> PyResult<bool> {
        with_domain!(self, k => Ok(k.is_unit(&k.extract(a)?)))
    }

    fn add<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.add(&k.extract(a)?, &k.extract(b)?)))
    }

    fn sub<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.sub(&k.extract(a)?, &k.extract(b)?)))
    }

    fn neg<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.neg(&k.extract(a)?)))
    }

    fn mul<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.mul(&k.extract(a)?, &k.extract(b)?)))
    }

    fn pow<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, n: i64) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.pow_signed(&k.extract(a)?, n).map_err(err)?))
    }

    fn exquo<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.exquo(&k.extract(a)?, &k.extract(b)?).map_err(err)?))
    }

    fn quo<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.quo(&k.extract(a)?, &k.extract(b)?).map_err(err)?))
    }

    fn rem<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.rem(&k.extract(a)?, &k.extract(b)?).map_err(err)?))
    }

    fn divmod<'py>(
        &self,
        py: Python<'py>,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => {
            let (q, r) = k.divmod(&k.extract(a)?, &k.extract(b)?).map_err(err)?;
            tuple(py, vec![out_elem(k, py, &q)?, out_elem(k, py, &r)?])
        })
    }

    fn invert<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.invert(&k.extract(a)?).map_err(err)?))
    }

    fn gcd<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.gcd(&k.extract(a)?, &k.extract(b)?)))
    }

    fn lcm<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.lcm(&k.extract(a)?, &k.extract(b)?)))
    }

    fn gcdex<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => {
            let (s, t, g) = k.gcdex(&k.extract(a)?, &k.extract(b)?).map_err(err)?;
            tuple(py, vec![out_elem(k, py, &s)?, out_elem(k, py, &t)?, out_elem(k, py, &g)?])
        })
    }

    fn abs<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.ordered()?;
        with_domain!(self, k => out_elem(k, py, &k.abs(&k.extract(a)?)))
    }

    fn sign(&self, a: &Bound<'_, PyAny>) -> PyResult<i32> {
        self.ordered()?;
        with_domain!(self, k => Ok(k.sign(&k.extract(a)?)))
    }

    fn canonical_unit<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => out_elem(k, py, &k.canonical_unit(&k.extract(a)?)))
    }

    fn size(&self, a: &Bound<'_, PyAny>) -> PyResult<u64> {
        with_domain!(self, k => Ok(k.size(&k.extract(a)?)))
    }

    fn from_int<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let n = crate::convert::extract_int(n)?;
        with_domain!(self, k => out_elem(k, py, &k.from_integer(&n)))
    }

    fn numer<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self.kind {
            DomainKind::QQ => crate::convert::out_int(py, &RationalField.numer(&RationalField.extract(a)?)),
            _ => Err(pyo3::exceptions::PyAttributeError::new_err("numer")),
        }
    }

    fn denom<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self.kind {
            DomainKind::QQ => crate::convert::out_int(py, &RationalField.denom(&RationalField.extract(a)?)),
            _ => Err(pyo3::exceptions::PyAttributeError::new_err("denom")),
        }
    }

    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.kind {
            DomainKind::GF(p) => {
                let field: FiniteField = FiniteField::new(p).map_err(err)?;
                py.import("builtins")?
                    .getattr("iter")?
                    .call1((py.import("builtins")?.getattr("range")?.call1((field.p(),))?,))
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "{} is not finite",
                rust_name(self.kind)
            ))),
        }
    }
}

impl PyDomain {
    /// `NotImplementedError` for an unordered domain (a panic in the core).
    fn ordered(&self) -> PyResult<()> {
        with_domain!(self, k => {
            if k.is_ordered() {
                Ok(())
            } else {
                Err(pyo3::exceptions::PyNotImplementedError::new_err(format!("{} is not ordered", k.name())))
            }
        })
    }

    /// `NotImplementedError` unless the domain is ordered; used by the
    /// functions of the ordered domains.
    pub fn require_ordered(&self) -> PyResult<()> {
        self.ordered()
    }
}

/// The prime field with `p` elements.
#[pyfunction]
#[pyo3(name = "GF")]
fn gf(p: u64) -> PyResult<PyDomain> {
    FiniteField::new(p).map_err(err)?;
    Ok(PyDomain {
        kind: DomainKind::GF(p),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDomain>()?;
    m.add("ZZ", PyDomain { kind: DomainKind::ZZ })?;
    m.add("QQ", PyDomain { kind: DomainKind::QQ })?;
    m.add_function(wrap_pyfunction!(gf, m)?)?;
    m.add("FiniteField", m.getattr("GF")?)?;
    Ok(())
}
