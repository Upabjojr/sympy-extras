//! `sympy_extras_rs.core`: the bridge between the expressions of the Python
//! core (`sympy_extras.core`) and the nodes of the Rust core, and the
//! functions of the Rust core on it.
//!
//! An expression of the Python core is read by its class name and its
//! arguments (numbers by value, symbols by name and given assumptions) and
//! rebuilt on the way back with the unevaluating classes of the Python core
//! (`Add(*args)` keeps its arguments). A Python `Dummy` and its Rust node are
//! kept in a registry, in both directions, so that a dummy stays itself over
//! round trips. The classes the Rust core does not have yet are a
//! `TypeError`.

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use sympy_extras_core::core::assumptions::{facts_of, is_number, Predicate};
use sympy_extras_core::core::evalf::{numeric_value, parts, Real};
use sympy_extras_core::core::{self as rust_core, Basic, Dummy, Function, Head, Symbol};
use sympy_extras_core::functions::elementary::{complexes, exponential, hyperbolic, trigonometric};
#[allow(unused_imports)]
use sympy_extras_core::numbers::rat;
use sympy_extras_core::numbers::{denom, rat_from_parts, Int, Rat};

use crate::convert::{extract_int, out_int};

/// The dummies met: the Python `dummy_index` to the Rust node, and the Rust
/// index to the Python object.
struct Dummies {
    from_python: HashMap<i64, Basic>,
    to_python: HashMap<u64, Py<PyAny>>,
}

static DUMMIES: Mutex<Option<Dummies>> = Mutex::new(None);

fn with_dummies<T>(f: impl FnOnce(&mut Dummies) -> T) -> T {
    let mut guard = DUMMIES.lock().expect("an unpoisoned registry");
    let dummies: &mut Dummies = guard.get_or_insert_with(|| Dummies {
        from_python: HashMap::new(),
        to_python: HashMap::new(),
    });
    f(dummies)
}

fn core_module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    py.import(format!("sympy_extras.core.{}", name))
}

fn given_assumptions(ob: &Bound<'_, PyAny>) -> PyResult<Vec<(Predicate, bool)>> {
    let mut result: Vec<(Predicate, bool)> = Vec::new();
    let given: Bound<'_, PyAny> = ob.getattr("given_assumptions")?;
    let items: Bound<'_, PyAny> = if given.is_callable() { given.call0()? } else { given };
    let dict: &Bound<'_, PyDict> = items.downcast::<PyDict>()?;
    for (k, v) in dict.iter() {
        let name: String = k.extract()?;
        let Some(p) = Predicate::from_name(&name) else {
            return Err(PyTypeError::new_err(format!("unknown assumption {}", name)));
        };
        result.push((p, v.is_truthy()?));
    }
    Ok(result)
}

/// The Rust node of a Python core expression.
pub fn from_python(ob: &Bound<'_, PyAny>) -> PyResult<Basic> {
    let class: String = ob.get_type().name()?.to_string();
    let args = || -> PyResult<Vec<Basic>> {
        let mut result: Vec<Basic> = Vec::new();
        for a in ob.getattr("args")?.try_iter()? {
            result.push(from_python(&a?)?);
        }
        Ok(result)
    };
    Ok(match class.as_str() {
        "Integer" | "Zero" | "One" | "NegativeOne" => Basic::integer(extract_int(&ob.getattr("p")?)?),
        "Rational" | "Half" => {
            // the class Rational even with denominator one (Rational(3, 1))
            let p: Int = extract_int(&ob.getattr("p")?)?;
            let q: Int = extract_int(&ob.getattr("q")?)?;
            Basic::rational(p, q).map_err(crate::convert::err)?
        }
        "Infinity" => Basic::oo(),
        "NegativeInfinity" => Basic::negative_oo(),
        "ComplexInfinity" => Basic::zoo(),
        "NaN" => Basic::nan(),
        "Pi" => Basic::pi(),
        "Exp1" => Basic::e(),
        "EulerGammaType" | "EulerGamma" => Basic::euler_gamma(),
        "CatalanType" | "Catalan" => Basic::catalan(),
        "GoldenRatioType" | "GoldenRatio" => Basic::golden_ratio(),
        "TribonacciConstantType" | "TribonacciConstant" => Basic::tribonacci_constant(),
        "ImaginaryUnit" => Basic::imaginary_unit(),
        "Symbol" => Symbol::new(&ob.getattr("name")?.extract::<String>()?, &given_assumptions(ob)?).0,
        "Dummy" => {
            let index: i64 = ob.getattr("dummy_index")?.extract()?;
            if let Some(found) = with_dummies(|d| d.from_python.get(&index).cloned()) {
                return Ok(found);
            }
            let node: Dummy = Dummy::new(&ob.getattr("name")?.extract::<String>()?, &given_assumptions(ob)?);
            let rust_index: u64 = node.dummy_index();
            let object: Py<PyAny> = ob.clone().unbind();
            with_dummies(|d| {
                d.from_python.insert(index, node.0.clone());
                d.to_python.insert(rust_index, object);
            });
            node.0
        }
        "Add" => Basic::new(Head::Add, args()?),
        "Mul" => Basic::new(Head::Mul, args()?),
        "Pow" => Basic::new(Head::Pow, args()?),
        "Exp" => Basic::new(Head::Function(Function::Exp), args()?),
        "Log" => Basic::new(Head::Function(Function::Log), args()?),
        "Re" => Basic::new(Head::Function(Function::Re), args()?),
        "Im" => Basic::new(Head::Function(Function::Im), args()?),
        "Conjugate" => Basic::new(Head::Function(Function::Conjugate), args()?),
        "Sign" => Basic::new(Head::Function(Function::Sign), args()?),
        "Abs" => Basic::new(Head::Function(Function::Abs), args()?),
        "Sin" => Basic::new(Head::Function(Function::Sin), args()?),
        "Cos" => Basic::new(Head::Function(Function::Cos), args()?),
        "Tan" => Basic::new(Head::Function(Function::Tan), args()?),
        "Cot" => Basic::new(Head::Function(Function::Cot), args()?),
        "Sec" => Basic::new(Head::Function(Function::Sec), args()?),
        "Csc" => Basic::new(Head::Function(Function::Csc), args()?),
        "Sinh" => Basic::new(Head::Function(Function::Sinh), args()?),
        "Cosh" => Basic::new(Head::Function(Function::Cosh), args()?),
        "Tanh" => Basic::new(Head::Function(Function::Tanh), args()?),
        "Coth" => Basic::new(Head::Function(Function::Coth), args()?),
        "Sech" => Basic::new(Head::Function(Function::Sech), args()?),
        "Csch" => Basic::new(Head::Function(Function::Csch), args()?),
        other => return Err(PyTypeError::new_err(format!("the Rust core has no {} yet", other))),
    })
}

fn kwargs<'py>(py: Python<'py>, assumptions: &[(Predicate, bool)]) -> PyResult<Bound<'py, PyDict>> {
    let d: Bound<'py, PyDict> = PyDict::new(py);
    for (p, v) in assumptions {
        d.set_item(p.name(), *v)?;
    }
    Ok(d)
}

/// The Python core expression of a Rust node.
pub fn to_python<'py>(py: Python<'py>, e: &Basic) -> PyResult<Bound<'py, PyAny>> {
    let numbers = core_module(py, "numbers")?;
    let class = |module: &str, name: &str| -> PyResult<Bound<'py, PyAny>> { core_module(py, module)?.getattr(name) };
    let rebuilt = |name: &str, module: &str| -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
        class(module, name)?.call1(PyTuple::new(py, items)?)
    };
    Ok(match e.head() {
        Head::Rational { value, integer_class } => {
            if *integer_class {
                numbers.getattr("Integer")?.call1((out_int(py, value.numerator())?,))?
            } else {
                numbers
                    .getattr("Rational")?
                    .call1((out_int(py, value.numerator())?, out_int(py, &denom(value))?))?
            }
        }
        Head::Infinity => numbers.getattr("oo")?,
        Head::NegativeInfinity => numbers.getattr("oo")?.call_method0("__neg__")?,
        Head::ComplexInfinity => numbers.getattr("zoo")?,
        Head::NaN => numbers.getattr("nan")?,
        Head::Pi => numbers.getattr("pi")?,
        Head::Exp1 => numbers.getattr("E")?,
        Head::EulerGamma => numbers.getattr("EulerGamma")?,
        Head::Catalan => numbers.getattr("Catalan")?,
        Head::GoldenRatio => numbers.getattr("GoldenRatio")?,
        Head::TribonacciConstant => numbers.getattr("TribonacciConstant")?,
        Head::ImaginaryUnit => numbers.getattr("I")?,
        Head::Symbol(data) => {
            let given: Vec<(Predicate, bool)> = Symbol(e.clone()).given_assumptions().to_vec();
            class("symbol", "Symbol")?.call((data.name.clone(),), Some(&kwargs(py, &given)?))?
        }
        Head::Dummy(_, index) => match with_dummies(|d| d.to_python.get(index).map(|o| o.clone_ref(py))) {
            Some(object) => object.into_bound(py),
            None => return Err(PyTypeError::new_err("a dummy which did not come from Python")),
        },
        Head::Wild(_) => return Err(PyTypeError::new_err("wilds are not converted")),
        Head::Add => rebuilt("Add", "add")?,
        Head::Mul => rebuilt("Mul", "mul")?,
        Head::Pow => rebuilt("Pow", "power")?,
        Head::Function(f) => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            let module: &str = match f {
                Function::Exp | Function::Log => "exponential",
                Function::Sin | Function::Cos | Function::Tan | Function::Cot | Function::Sec | Function::Csc => {
                    "trigonometric"
                }
                Function::Sinh | Function::Cosh | Function::Tanh | Function::Coth | Function::Sech | Function::Csch => {
                    "hyperbolic"
                }
                _ => "complexes",
            };
            py.import(format!("sympy_extras.functions.elementary.{}", module))?
                .getattr(f.class_name())?
                .call1(PyTuple::new(py, items)?)?
        }
    })
}

/// The round trip `to_python(from_python(e))`.
#[pyfunction]
fn roundtrip<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, &from_python(e)?)
}

/// The facts of `facts_of`, as a dict of predicate names.
#[pyfunction]
fn facts<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let node: Basic = from_python(e)?;
    let d: Bound<'py, PyDict> = PyDict::new(py);
    for (p, v) in facts_of(&node) {
        d.set_item(p.name(), *v)?;
    }
    Ok(d)
}

#[pyfunction(name = "is_number")]
fn is_number_py(e: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_number(&from_python(e)?))
}

/// An exact rational of a binary float.
fn real_to_fraction<'py>(py: Python<'py>, x: &Real) -> PyResult<Bound<'py, PyAny>> {
    let repr = x.repr();
    let significand: Int = repr.significand().clone();
    let exponent: isize = repr.exponent();
    let value: Rat = if exponent >= 0 {
        Rat::from(significand * (Int::ONE << exponent as usize))
    } else {
        rat_from_parts(significand, Int::ONE << (-exponent) as usize)
    };
    py.import("fractions")?
        .getattr("Fraction")?
        .call1((out_int(py, value.numerator())?, out_int(py, &denom(&value))?))
}

/// `numeric_value(e, prec)` as `(re, im)` exact fractions of the binary
/// floats, or `None`.
#[pyfunction(name = "numeric_value")]
fn numeric_value_py<'py>(py: Python<'py>, e: &Bound<'py, PyAny>, prec: usize) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(z) = numeric_value(&from_python(e)?, prec) else {
        return Ok(None);
    };
    let (re, im) = parts(&z);
    Ok(Some(
        PyTuple::new(py, [real_to_fraction(py, &re)?, real_to_fraction(py, &im)?])?.into_any(),
    ))
}

/// `a.compare(b)`.
#[pyfunction]
fn compare(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<i32> {
    Ok(from_python(a)?.compare(&from_python(b)?))
}

#[pyfunction]
fn count_nodes(e: &Bound<'_, PyAny>) -> PyResult<usize> {
    Ok(from_python(e)?.count_nodes())
}

/// The free symbols, in the order of the Rust set (sort them to compare).
#[pyfunction]
fn free_symbols<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = from_python(e)?
        .free_symbols()
        .iter()
        .map(|s| to_python(py, s))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

/// Equality of the two Rust nodes (Python's structural `==`).
#[pyfunction]
fn equal(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(from_python(a)? == from_python(b)?)
}

fn nodes(args: &Bound<'_, PyTuple>) -> PyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    for a in args.iter() {
        result.push(from_python(&a)?);
    }
    Ok(result)
}

/// `add(*args)` of the Rust core.
#[pyfunction(signature = (*args))]
fn add<'py>(py: Python<'py>, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, &rust_core::add::add(&nodes(args)?).map_err(crate::convert::err)?)
}

/// `mul(*args)` of the Rust core.
#[pyfunction(signature = (*args))]
fn mul<'py>(py: Python<'py>, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, &rust_core::mul::mul(&nodes(args)?).map_err(crate::convert::err)?)
}

#[pyfunction]
fn pow<'py>(py: Python<'py>, b: &Bound<'py, PyAny>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::power::pow(&from_python(b)?, &from_python(e)?).map_err(crate::convert::err)?,
    )
}

#[pyfunction]
fn sub<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::add::sub(&from_python(a)?, &from_python(b)?).map_err(crate::convert::err)?,
    )
}

#[pyfunction]
fn div<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::mul::div(&from_python(a)?, &from_python(b)?).map_err(crate::convert::err)?,
    )
}

#[pyfunction]
fn neg<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, &rust_core::mul::neg(&from_python(a)?).map_err(crate::convert::err)?)
}

/// `str(e)` printed by the Rust core.
#[pyfunction]
fn sstr(e: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(rust_core::printing::sstr(&from_python(e)?))
}

#[pyfunction]
fn exp<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, &exponential::exp(&from_python(x)?).map_err(crate::convert::err)?)
}

#[pyfunction]
#[pyo3(signature = (x, base=None))]
fn log<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, base: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let value = match base {
        None => exponential::log(&from_python(x)?),
        Some(b) => exponential::log_base(&from_python(x)?, &from_python(b)?),
    };
    to_python(py, &value.map_err(crate::convert::err)?)
}

fn pair<'py>(py: Python<'py>, value: PyResult<(Basic, Basic)>) -> PyResult<Bound<'py, PyTuple>> {
    let (a, b) = value?;
    PyTuple::new(py, [to_python(py, &a)?, to_python(py, &b)?])
}

#[pyfunction]
fn as_numer_denom<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    pair(
        py,
        rust_core::expr::as_numer_denom(&from_python(e)?).map_err(crate::convert::err),
    )
}

#[pyfunction]
fn fraction<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    pair(
        py,
        sympy_extras_core::simplify::radsimp::fraction(&from_python(e)?).map_err(crate::convert::err),
    )
}

#[pyfunction]
#[pyo3(signature = (e, deep=true))]
fn expand_mul<'py>(py: Python<'py>, e: &Bound<'py, PyAny>, deep: bool) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::function::expand_mul(&from_python(e)?, deep).map_err(crate::convert::err)?,
    )
}

#[pyfunction]
#[pyo3(signature = (e, deep=true))]
fn expand_multinomial<'py>(py: Python<'py>, e: &Bound<'py, PyAny>, deep: bool) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::function::expand_multinomial(&from_python(e)?, deep).map_err(crate::convert::err)?,
    )
}

macro_rules! unary {
    ($name:ident, $path:path) => {
        #[pyfunction]
        fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            to_python(py, &$path(&from_python(x)?).map_err(crate::convert::err)?)
        }
    };
}

unary!(re, complexes::re);
unary!(im, complexes::im);
unary!(conjugate, complexes::conjugate);
unary!(sign, complexes::sign);
unary!(abs_, complexes::abs_);
unary!(sin, trigonometric::sin);
unary!(cos, trigonometric::cos);
unary!(tan, trigonometric::tan);
unary!(cot, trigonometric::cot);
unary!(sec, trigonometric::sec);
unary!(csc, trigonometric::csc);
unary!(sinh, hyperbolic::sinh);
unary!(cosh, hyperbolic::cosh);
unary!(tanh, hyperbolic::tanh);
unary!(coth, hyperbolic::coth);
unary!(sech, hyperbolic::sech);
unary!(csch, hyperbolic::csch);

#[pyfunction]
fn as_real_imag<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    pair(
        py,
        complexes::real_imag_parts(&from_python(e)?).map_err(crate::convert::err),
    )
}

fn terms_to_py<'py, K: crate::convert::Bridge>(
    py: Python<'py>,
    k: &K,
    terms: &sympy_extras_core::polys::rings::Terms<K::Elem>,
) -> PyResult<Bound<'py, PyDict>> {
    let d: Bound<'py, PyDict> = PyDict::new(py);
    for (m, c) in terms {
        d.set_item(PyTuple::new(py, m.iter().copied())?, k.to_py(py, c)?)?;
    }
    Ok(d)
}

fn terms_from_py<K: crate::convert::Bridge>(
    k: &K,
    d: &Bound<'_, PyDict>,
) -> PyResult<sympy_extras_core::polys::rings::Terms<K::Elem>> {
    let mut terms = sympy_extras_core::polys::rings::Terms::new();
    for (m, c) in d.iter() {
        let monomial: Vec<u32> = m.extract()?;
        terms.insert(monomial, k.extract(&c)?);
    }
    Ok(terms)
}

/// `polyutils.dict_from_expr` over `ZZ`, `QQ`, `ZZ_I`, `QQ_I` or `GF(p)`
/// (the domain given by its name, or by the prime `p`).
#[pyfunction]
fn dict_from_expr<'py>(
    py: Python<'py>,
    expr: &Bound<'py, PyAny>,
    gens: Vec<String>,
    domain: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::polys::domains as d;
    use sympy_extras_core::polys::polyutils::dict_from_expr as convert;
    let e: Basic = from_python(expr)?;
    let e_ = crate::convert::err;
    if let Ok(p) = domain.extract::<u64>() {
        let k = d::finitefield::FiniteField::new(p).map_err(crate::convert::err)?;
        return terms_to_py(py, &k, &convert(&e, &gens, &k).map_err(e_)?);
    }
    let name: String = domain.extract()?;
    match name.as_str() {
        "ZZ" => terms_to_py(
            py,
            &d::integerring::ZZ,
            &convert(&e, &gens, &d::integerring::ZZ).map_err(e_)?,
        ),
        "QQ" => terms_to_py(
            py,
            &d::rationalfield::QQ,
            &convert(&e, &gens, &d::rationalfield::QQ).map_err(e_)?,
        ),
        "ZZ_I" => {
            let k = d::gaussiandomains::GaussianIntegerRing;
            terms_to_py(py, &k, &convert(&e, &gens, &k).map_err(e_)?)
        }
        "QQ_I" => {
            let k = d::gaussiandomains::GaussianRationalField;
            terms_to_py(py, &k, &convert(&e, &gens, &k).map_err(e_)?)
        }
        other => Err(PyTypeError::new_err(format!("no domain {}", other))),
    }
}

/// `polyutils.expr_from_dict` over the same domains.
#[pyfunction]
fn expr_from_dict<'py>(
    py: Python<'py>,
    terms: &Bound<'py, PyDict>,
    gens: Vec<String>,
    domain: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    use sympy_extras_core::polys::domains as d;
    use sympy_extras_core::polys::polyutils::expr_from_dict as convert;
    let e_ = crate::convert::err;
    let result: Basic = if let Ok(p) = domain.extract::<u64>() {
        let k = d::finitefield::FiniteField::new(p).map_err(crate::convert::err)?;
        convert(&terms_from_py(&k, terms)?, &gens, &k).map_err(e_)?
    } else {
        let name: String = domain.extract()?;
        match name.as_str() {
            "ZZ" => convert(&terms_from_py(&d::integerring::ZZ, terms)?, &gens, &d::integerring::ZZ).map_err(e_)?,
            "QQ" => convert(
                &terms_from_py(&d::rationalfield::QQ, terms)?,
                &gens,
                &d::rationalfield::QQ,
            )
            .map_err(e_)?,
            "ZZ_I" => {
                let k = d::gaussiandomains::GaussianIntegerRing;
                convert(&terms_from_py(&k, terms)?, &gens, &k).map_err(e_)?
            }
            "QQ_I" => {
                let k = d::gaussiandomains::GaussianRationalField;
                convert(&terms_from_py(&k, terms)?, &gens, &k).map_err(e_)?
            }
            other => return Err(PyTypeError::new_err(format!("no domain {}", other))),
        }
    };
    to_python(py, &result)
}

#[pyfunction]
#[pyo3(signature = (e, isprimitive=false, clear=true, fraction=true))]
fn gcd_terms<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    isprimitive: bool,
    clear: bool,
    fraction: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let value = rust_core::exprtools::gcd_terms(&from_python(e)?, isprimitive, clear, fraction);
    to_python(py, &value.map_err(crate::convert::err)?)
}

#[pyfunction]
#[pyo3(signature = (e, radical=false, clear=false, fraction=false, sign=true))]
fn factor_terms<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    radical: bool,
    clear: bool,
    fraction: bool,
    sign: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let value = rust_core::exprtools::factor_terms(&from_python(e)?, radical, clear, fraction, sign);
    to_python(py, &value.map_err(crate::convert::err)?)
}

#[pyfunction]
#[pyo3(signature = (e, radical=false, clear=true))]
fn as_content_primitive<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    radical: bool,
    clear: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    pair(
        py,
        rust_core::exprtools::as_content_primitive(&from_python(e)?, radical, clear).map_err(crate::convert::err),
    )
}

#[pyfunction]
fn decompose_power<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, i64)> {
    let (b, n) = rust_core::exprtools::decompose_power(&from_python(e)?).map_err(crate::convert::err)?;
    Ok((to_python(py, &b)?, n))
}

#[pyfunction]
fn number_pow<'py>(py: Python<'py>, b: &Bound<'py, PyAny>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::power::number_pow(&from_python(b)?, &from_python(e)?).map_err(crate::convert::err)?,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(facts, m)?)?;
    m.add_function(wrap_pyfunction!(is_number_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_function(wrap_pyfunction!(count_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(free_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(pow, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(number_pow, m)?)?;
    m.add_function(wrap_pyfunction!(sstr, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(as_numer_denom, m)?)?;
    m.add_function(wrap_pyfunction!(dict_from_expr, m)?)?;
    m.add_function(wrap_pyfunction!(gcd_terms, m)?)?;
    m.add_function(wrap_pyfunction!(factor_terms, m)?)?;
    m.add_function(wrap_pyfunction!(as_content_primitive, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_power, m)?)?;
    m.add_function(wrap_pyfunction!(expr_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(re, m)?)?;
    m.add_function(wrap_pyfunction!(im, m)?)?;
    m.add_function(wrap_pyfunction!(conjugate, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(abs_, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(cot, m)?)?;
    m.add_function(wrap_pyfunction!(sec, m)?)?;
    m.add_function(wrap_pyfunction!(csc, m)?)?;
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(coth, m)?)?;
    m.add_function(wrap_pyfunction!(sech, m)?)?;
    m.add_function(wrap_pyfunction!(csch, m)?)?;
    m.add_function(wrap_pyfunction!(as_real_imag, m)?)?;
    m.add_function(wrap_pyfunction!(fraction, m)?)?;
    m.add_function(wrap_pyfunction!(expand_mul, m)?)?;
    m.add_function(wrap_pyfunction!(expand_multinomial, m)?)?;
    Ok(())
}
