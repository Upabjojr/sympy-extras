//! `sympy_extras_rs.polys.domains`: the domains `ZZ`, `QQ` and `GF(p)`, and
//! the polynomial rings `K[x]` and the fields of fractions `K(x)` over them
//! (`polynomialring.PolynomialRing`, `fractionfield.FractionField`), as one
//! Python class with the methods of the Python domains.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sympy_extras_core::polys::domains::{Domain, FiniteField, RationalField};

use crate::convert::{err, out_elem, tuple, Bridge};

/// Which domain a Python `Domain` object stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainKind {
    ZZ,
    QQ,
    GF(u64),
    /// `ZZ[x]`, `QQ[x]`, `GF(p)[x]` (the variable is `PyDomain::variable`)
    PolyZZ,
    PolyQQ,
    PolyGF(u64),
    /// `ZZ(x)`, `QQ(x)`, `GF(p)(x)`
    FracZZ,
    FracQQ,
    FracGF(u64),
}

/// A coefficient domain.
#[pyclass(name = "Domain", module = "sympy_extras_rs.polys.domains", frozen, eq, hash)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyDomain {
    pub kind: DomainKind,
    /// The variable of `K[x]` and `K(x)` (empty otherwise).
    pub variable: String,
}

impl PyDomain {
    /// `ZZ`, `QQ` or `GF(p)`.
    pub fn base(kind: DomainKind) -> PyDomain {
        PyDomain {
            kind,
            variable: String::new(),
        }
    }

    /// Whether this is `ZZ`, `QQ` or `GF(p)`.
    pub fn is_base(&self) -> bool {
        matches!(self.kind, DomainKind::ZZ | DomainKind::QQ | DomainKind::GF(_))
    }
}

/// Runs `$body` with `$k` bound to the Rust domain of a `PyDomain`.
#[macro_export]
macro_rules! with_domain {
    ($dom:expr, $k:ident => $body:expr) => {{
        use sympy_extras_core::polys::domains::{
            FiniteField as _F, FractionField as _Frac, IntegerRing as _Z, PolynomialRing as _P, RationalField as _Q,
        };
        let dom: &$crate::domains::PyDomain = &$dom;
        match dom.kind {
            $crate::domains::DomainKind::ZZ => {
                let $k = &_Z;
                $body
            }
            $crate::domains::DomainKind::QQ => {
                let $k = &_Q;
                $body
            }
            $crate::domains::DomainKind::GF(p) => {
                let field = _F::new(p).map_err($crate::convert::err)?;
                let $k = &field;
                $body
            }
            $crate::domains::DomainKind::PolyZZ => {
                let ring = _P::new(_Z, &dom.variable);
                let $k = &ring;
                $body
            }
            $crate::domains::DomainKind::PolyQQ => {
                let ring = _P::new(_Q, &dom.variable);
                let $k = &ring;
                $body
            }
            $crate::domains::DomainKind::PolyGF(p) => {
                let ring = _P::new(_F::new(p).map_err($crate::convert::err)?, &dom.variable);
                let $k = &ring;
                $body
            }
            $crate::domains::DomainKind::FracZZ => {
                let field = _Frac::new(_P::new(_Z, &dom.variable));
                let $k = &field;
                $body
            }
            $crate::domains::DomainKind::FracQQ => {
                let field = _Frac::new(_P::new(_Q, &dom.variable));
                let $k = &field;
                $body
            }
            $crate::domains::DomainKind::FracGF(p) => {
                let field = _Frac::new(_P::new(_F::new(p).map_err($crate::convert::err)?, &dom.variable));
                let $k = &field;
                $body
            }
        }
    }};
}

/// [`with_domain!`] for `ZZ`, `QQ` and `GF(p)` only (`TypeError` for the
/// polynomial domains): the functions which need conversions between any
/// two of the domains.
#[macro_export]
macro_rules! with_base_domain {
    ($dom:expr, $k:ident => $body:expr) => {{
        let dom: &$crate::domains::PyDomain = &$dom;
        match dom.kind {
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
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "only ZZ, QQ and GF(p) are supported here",
            )),
        }
    }};
}

fn rust_name(dom: &PyDomain) -> String {
    let name = || -> PyResult<String> { with_domain!(dom, k => Ok(k.name())) };
    name().unwrap_or_else(|_| format!("{:?}", dom.kind))
}

#[pymethods]
impl PyDomain {
    #[getter]
    fn name(&self) -> String {
        rust_name(self)
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
    fn characteristic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_domain!(self, k => crate::convert::out_int(py, &k.characteristic()))
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
        rust_name(self)
    }

    fn __str__(&self) -> String {
        rust_name(self)
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

    fn factor_dense<'py>(&self, py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if !self.is_base() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "no factorisation over {}",
                rust_name(self)
            )));
        }
        with_domain!(self, k => {
            let (c, factors) = k.factor_dense(&crate::convert::poly(k, f)?).map_err(err)?;
            tuple(py, vec![out_elem(k, py, &c)?, crate::convert::out_factors(k, py, &factors)?])
        })
    }

    #[pyo3(signature = (f, eps = None))]
    fn isolate_real_roots_dense<'py>(
        &self,
        py: Python<'py>,
        f: &Bound<'py, PyAny>,
        eps: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let eps = match eps {
            Some(e) if !e.is_none() => Some(RationalField.extract(e)?),
            _ => None,
        };
        match self.kind {
            DomainKind::ZZ => crate::algorithms::out_roots(
                py,
                &sympy_extras_core::polys::domains::IntegerRing.isolate_real_roots_dense(
                    &crate::convert::poly(&sympy_extras_core::polys::domains::IntegerRing, f)?,
                    eps.as_ref(),
                ),
            ),
            DomainKind::QQ => crate::algorithms::out_roots(
                py,
                &RationalField.isolate_real_roots_dense(&crate::convert::poly(&RationalField, f)?, eps.as_ref()),
            ),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "no real root isolation over {}",
                rust_name(self)
            ))),
        }
    }

    fn count_real_roots_dense(
        &self,
        f: &Bound<'_, PyAny>,
        inf: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
    ) -> PyResult<i64> {
        let bound = |b: &Bound<'_, PyAny>| -> PyResult<Option<sympy_extras_core::numbers::Rat>> {
            if b.is_none() {
                Ok(None)
            } else {
                Ok(Some(RationalField.extract(b)?))
            }
        };
        let (inf, sup) = (bound(inf)?, bound(sup)?);
        match self.kind {
            DomainKind::ZZ => Ok(sympy_extras_core::polys::domains::IntegerRing.count_real_roots_dense(
                &crate::convert::poly(&sympy_extras_core::polys::domains::IntegerRing, f)?,
                inf.as_ref(),
                sup.as_ref(),
            )),
            DomainKind::QQ => Ok(RationalField.count_real_roots_dense(
                &crate::convert::poly(&RationalField, f)?,
                inf.as_ref(),
                sup.as_ref(),
            )),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "no real root counting over {}",
                rust_name(self)
            ))),
        }
    }

    /// `FractionField.new(numer, denom)`: the reduced fraction.
    #[allow(clippy::new_ret_no_self)]
    fn new<'py>(
        &self,
        py: Python<'py>,
        numer: &Bound<'py, PyAny>,
        denom: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !matches!(
            self.kind,
            DomainKind::FracZZ | DomainKind::FracQQ | DomainKind::FracGF(_)
        ) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "new is a method of the fields of fractions",
            ));
        }
        let raw = Bound::new(
            py,
            crate::convert::PyFracElement::from_parts(numer.clone().unbind(), denom.clone().unbind()),
        )?;
        with_domain!(self, k => out_elem(k, py, &k.extract(raw.as_any())?))
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
                rust_name(self)
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
    Ok(PyDomain::base(DomainKind::GF(p)))
}

/// `polynomialring.PolynomialRing(ground, variable)` over `ZZ`, `QQ` or
/// `GF(p)`.
#[pyfunction]
#[pyo3(name = "PolynomialRing")]
fn polynomial_ring(ground: &PyDomain, variable: String) -> PyResult<PyDomain> {
    let kind: DomainKind = match ground.kind {
        DomainKind::ZZ => DomainKind::PolyZZ,
        DomainKind::QQ => DomainKind::PolyQQ,
        DomainKind::GF(p) => DomainKind::PolyGF(p),
        _ => {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "the ground must be ZZ, QQ or GF(p)",
            ))
        }
    };
    Ok(PyDomain { kind, variable })
}

/// `fractionfield.FractionField(ring)` of a `PolynomialRing`.
#[pyfunction]
#[pyo3(name = "FractionField")]
fn fraction_field(ring: &PyDomain) -> PyResult<PyDomain> {
    let kind: DomainKind = match ring.kind {
        DomainKind::PolyZZ => DomainKind::FracZZ,
        DomainKind::PolyQQ => DomainKind::FracQQ,
        DomainKind::PolyGF(p) => DomainKind::FracGF(p),
        _ => {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "a PolynomialRing over ZZ, QQ or GF(p) is expected",
            ))
        }
    };
    Ok(PyDomain {
        kind,
        variable: ring.variable.clone(),
    })
}

pub fn register_polynomialring(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(polynomial_ring, m)?)?;
    Ok(())
}

pub fn register_fractionfield(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fraction_field, m)?)?;
    m.add_class::<crate::convert::PyFracElement>()?;
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDomain>()?;
    m.add("ZZ", PyDomain::base(DomainKind::ZZ))?;
    m.add("QQ", PyDomain::base(DomainKind::QQ))?;
    m.add_function(wrap_pyfunction!(gf, m)?)?;
    m.add("FiniteField", m.getattr("GF")?)?;
    Ok(())
}
