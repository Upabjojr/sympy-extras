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
        "AppliedUndef" => {
            let function: Bound<'_, PyAny> = ob.getattr("function")?;
            let name: String = function.getattr("name")?.extract()?;
            let nargs_ob: Bound<'_, PyAny> = function.getattr("nargs")?;
            let nargs: Option<Vec<usize>> = if nargs_ob.is_none() {
                None
            } else {
                let mut n: Vec<usize> = Vec::new();
                for k in nargs_ob.try_iter()? {
                    n.push(k?.extract()?);
                }
                Some(n)
            };
            let mut assumptions: Vec<(Predicate, bool)> = Vec::new();
            let given: Bound<'_, PyAny> = function.getattr("assumptions")?;
            for (k, v) in given.downcast::<PyDict>()?.iter() {
                let key: String = k.extract()?;
                let Some(p) = Predicate::from_name(&key) else {
                    return Err(PyTypeError::new_err(format!("unknown assumption {}", key)));
                };
                assumptions.push((p, v.extract()?));
            }
            let u = std::sync::Arc::new(sympy_extras_core::core::basic::UndefinedFunction::new(
                &name,
                nargs,
                &assumptions,
            ));
            Basic::new(Head::Undefined(u), args()?)
        }
        "Tuple" => Basic::new(Head::Tuple, args()?),
        "Derivative" => Basic::new(Head::Derivative, args()?),
        "Subs" => Basic::new(Head::Subs, args()?),
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
        "Arg" => Basic::new(Head::Function(Function::Arg), args()?),
        "Atan2" => Basic::new(Head::Function(Function::Atan2), args()?),
        "Asin" => Basic::new(Head::Function(Function::Asin), args()?),
        "Acos" => Basic::new(Head::Function(Function::Acos), args()?),
        "Atan" => Basic::new(Head::Function(Function::Atan), args()?),
        "Acot" => Basic::new(Head::Function(Function::Acot), args()?),
        "Asec" => Basic::new(Head::Function(Function::Asec), args()?),
        "Acsc" => Basic::new(Head::Function(Function::Acsc), args()?),
        "Asinh" => Basic::new(Head::Function(Function::Asinh), args()?),
        "Acosh" => Basic::new(Head::Function(Function::Acosh), args()?),
        "Atanh" => Basic::new(Head::Function(Function::Atanh), args()?),
        "Acoth" => Basic::new(Head::Function(Function::Acoth), args()?),
        "Asech" => Basic::new(Head::Function(Function::Asech), args()?),
        "Acsch" => Basic::new(Head::Function(Function::Acsch), args()?),
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
        Head::Dummy(data, index) => match with_dummies(|d| d.to_python.get(index).map(|o| o.clone_ref(py))) {
            Some(object) => object.into_bound(py),
            None => {
                // a dummy made in Rust: a new Python dummy, registered both ways
                let given: Vec<(Predicate, bool)> = data.given.clone();
                let object: Bound<'py, PyAny> =
                    class("symbol", "Dummy")?.call((data.name.clone(),), Some(&kwargs(py, &given)?))?;
                let python_index: i64 = object.getattr("dummy_index")?.extract()?;
                let unbound: Py<PyAny> = object.clone().unbind();
                with_dummies(|d| {
                    d.from_python.insert(python_index, e.clone());
                    d.to_python.insert(*index, unbound);
                });
                object
            }
        },
        Head::Wild(_) => return Err(PyTypeError::new_err("wilds are not converted")),
        Head::Add => rebuilt("Add", "add")?,
        Head::Mul => rebuilt("Mul", "mul")?,
        Head::Pow => rebuilt("Pow", "power")?,
        Head::Undefined(u) => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            let given: Vec<(Predicate, bool)> = u.assumptions.clone();
            let nargs: Bound<'py, PyAny> = match &u.nargs {
                None => py.None().into_bound(py),
                Some(n) => PyTuple::new(py, n.iter().copied())?.into_any(),
            };
            let function: Bound<'py, PyAny> =
                class("function", "Function")?.call((u.name.clone(), nargs), Some(&kwargs(py, &given)?))?;
            function.call1(PyTuple::new(py, items)?)?
        }
        Head::Tuple => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            class("containers", "Tuple")?.call1(PyTuple::new(py, items)?)?
        }
        Head::Derivative => rebuilt("Derivative", "function")?,
        Head::Subs => rebuilt("Subs", "function")?,
        Head::Function(f) => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            let module: &str = match f {
                Function::Exp | Function::Log => "exponential",
                Function::Sin | Function::Cos | Function::Tan | Function::Cot | Function::Sec | Function::Csc => {
                    "trigonometric"
                }
                f if sympy_extras_core::functions::elementary::inverse::is_inverse_trigonometric(*f) => "trigonometric",
                f if sympy_extras_core::functions::elementary::inverse::is_inverse(*f) => "hyperbolic",
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
unary!(arg, sympy_extras_core::functions::elementary::argument::arg);

#[pyfunction]
fn atan2<'py>(py: Python<'py>, y: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let value = sympy_extras_core::functions::elementary::argument::atan2(&from_python(y)?, &from_python(x)?);
    to_python(py, &value.map_err(crate::convert::err)?)
}
unary!(asin, sympy_extras_core::functions::elementary::inverse::asin);
unary!(acos, sympy_extras_core::functions::elementary::inverse::acos);
unary!(atan, sympy_extras_core::functions::elementary::inverse::atan);
unary!(acot, sympy_extras_core::functions::elementary::inverse::acot);
unary!(asec, sympy_extras_core::functions::elementary::inverse::asec);
unary!(acsc, sympy_extras_core::functions::elementary::inverse::acsc);
unary!(asinh, sympy_extras_core::functions::elementary::inverse::asinh);
unary!(acosh, sympy_extras_core::functions::elementary::inverse::acosh);
unary!(atanh, sympy_extras_core::functions::elementary::inverse::atanh);
unary!(acoth, sympy_extras_core::functions::elementary::inverse::acoth);
unary!(asech, sympy_extras_core::functions::elementary::inverse::asech);
unary!(acsch, sympy_extras_core::functions::elementary::inverse::acsch);

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
#[pyo3(signature = (e, deep=false, fraction=true))]
fn together<'py>(py: Python<'py>, e: &Bound<'py, PyAny>, deep: bool, fraction: bool) -> PyResult<Bound<'py, PyAny>> {
    let value = sympy_extras_core::polys::rationaltools::together(&from_python(e)?, deep, fraction);
    to_python(py, &value.map_err(crate::convert::err)?)
}

/// `diff(e, *spec)`: symbols, a symbol and a count, or `(symbol, n)`.
#[pyfunction(signature = (e, *spec))]
fn diff<'py>(py: Python<'py>, e: &Bound<'py, PyAny>, spec: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    let mut counts: Vec<(Basic, u64)> = Vec::new();
    for item in spec.iter() {
        if let Ok(n) = item.extract::<u64>() {
            let Some(last) = counts.last_mut() else {
                return Err(PyTypeError::new_err("a count must follow a symbol"));
            };
            last.1 = last.1 - 1 + n;
        } else if let Ok(t) = item.downcast::<PyTuple>() {
            counts.push((from_python(&t.get_item(0)?)?, t.get_item(1)?.extract()?));
        } else {
            counts.push((from_python(&item)?, 1));
        }
    }
    to_python(
        py,
        &rust_core::function::diff(&from_python(e)?, &counts).map_err(crate::convert::err)?,
    )
}

/// `e.subs(old, new)`, `e.subs([(old, new), ...])` or `e.subs({old: new})`.
#[pyfunction(signature = (e, *args))]
fn subs<'py>(py: Python<'py>, e: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    use sympy_extras_core::core::subs as s;
    let expr: Basic = from_python(e)?;
    let result = if args.len() == 2 {
        s::subs(
            &expr,
            &from_python(&args.get_item(0)?)?,
            &from_python(&args.get_item(1)?)?,
        )
    } else if args.len() == 1 {
        let rules: Bound<'py, PyAny> = args.get_item(0)?;
        let mut pairs: Vec<(Basic, Basic)> = Vec::new();
        if let Ok(d) = rules.downcast::<PyDict>() {
            for (k, v) in d.iter() {
                pairs.push((from_python(&k)?, from_python(&v)?));
            }
            s::subs_mapping(&expr, &pairs)
        } else {
            for item in rules.try_iter()? {
                let item = item?;
                pairs.push((from_python(&item.get_item(0)?)?, from_python(&item.get_item(1)?)?));
            }
            s::subs_pairs(&expr, &pairs)
        }
    } else {
        return Err(PyTypeError::new_err("subs takes (old, new), a mapping or pairs"));
    };
    to_python(py, &result.map_err(crate::convert::err)?)
}

fn expressions(items: &Bound<'_, PyAny>) -> PyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    for item in items.try_iter()? {
        result.push(from_python(&item?)?);
    }
    Ok(result)
}

fn expression_list<'py>(py: Python<'py>, items: &[Basic]) -> PyResult<Bound<'py, PyList>> {
    let converted: Vec<Bound<'py, PyAny>> = items.iter().map(|e| to_python(py, e)).collect::<PyResult<_>>()?;
    PyList::new(py, converted)
}

/// `linsolve.solve_linear_unique`.
#[pyfunction]
fn solve_linear_unique<'py>(
    py: Python<'py>,
    equations: &Bound<'py, PyAny>,
    unknowns: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    use sympy_extras_core::solvers::linsolve as l;
    let found =
        l::solve_linear_unique(&expressions(equations)?, &expressions(unknowns)?).map_err(crate::convert::err)?;
    found.map(|values| expression_list(py, &values)).transpose()
}

/// `linsolve.linear_root`.
#[pyfunction]
fn linear_root<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    use sympy_extras_core::solvers::linsolve as l;
    let found = l::linear_root(&from_python(e)?, &from_python(v)?).map_err(crate::convert::err)?;
    found.map(|root| to_python(py, &root)).transpose()
}

/// `linsolve.solve_one`.
#[pyfunction]
fn solve_one<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    use sympy_extras_core::solvers::linsolve as l;
    let found = l::solve_one(&from_python(e)?, &from_python(v)?).map_err(crate::convert::err)?;
    found.map(|roots| expression_list(py, &roots)).transpose()
}

/// `linsolve.numeric_linear_solution`.
#[pyfunction]
fn numeric_linear_solution<'py>(
    py: Python<'py>,
    equations: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    use sympy_extras_core::solvers::linsolve as l;
    let found = l::numeric_linear_solution(&expressions(equations)?).map_err(crate::convert::err)?;
    let Some(pairs) = found else {
        return Ok(None);
    };
    let d: Bound<'py, PyDict> = PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(to_python(py, &k)?, to_python(py, &v)?)?;
    }
    Ok(Some(d))
}

/// An equation as an expression: `lhs - rhs` of an `Equality`.
fn equation(ob: &Bound<'_, PyAny>) -> PyResult<Basic> {
    let class: String = ob.get_type().name()?.to_string();
    if class == "Equality" || class == "Eq" {
        return from_python(&ob.getattr("lhs")?.call_method1("__sub__", (ob.getattr("rhs")?,))?);
    }
    from_python(ob)
}

fn differential_ring(
    functions: &Bound<'_, PyAny>,
    variables: Option<&Bound<'_, PyAny>>,
    ranking: Option<&Bound<'_, PyAny>>,
    parameters: &Bound<'_, PyAny>,
) -> PyResult<sympy_extras_core::polys::differential::ring::DifferentialRing> {
    let functions: Vec<Basic> = expressions(functions)?;
    let variables: Option<Vec<Basic>> = variables.map(expressions).transpose()?;
    let ranking: Option<Vec<Vec<Basic>>> = match ranking {
        None => None,
        Some(r) => {
            let mut blocks: Vec<Vec<Basic>> = Vec::new();
            for block in r.try_iter()? {
                blocks.push(expressions(&block?)?);
            }
            Some(blocks)
        }
    };
    sympy_extras_core::polys::differential::ring::DifferentialRing::new(
        &functions,
        variables.as_deref(),
        ranking.as_deref(),
        &expressions(parameters)?,
    )
    .map_err(crate::convert::err)
}

fn jet_monomial<'py>(py: Python<'py>, m: &[((i64, Vec<u32>), u32)]) -> PyResult<Bound<'py, PyTuple>> {
    let mut items: Vec<Bound<'py, PyTuple>> = Vec::new();
    for ((a, j), e) in m {
        let jet: Bound<'py, PyTuple> = PyTuple::new(
            py,
            [
                a.into_pyobject(py)?.into_any(),
                PyTuple::new(py, j.iter().copied())?.into_any(),
            ],
        )?;
        items.push(PyTuple::new(py, [jet.into_any(), e.into_pyobject(py)?.into_any()])?);
    }
    PyTuple::new(py, items)
}

/// `DifferentialRing(functions, variables, ranking, parameters).terms(eq)`
/// as `[(monomial, coefficient expression)]`.
#[pyfunction]
#[pyo3(signature = (functions, eq, variables=None, ranking=None, parameters=None))]
fn differential_terms<'py>(
    py: Python<'py>,
    functions: &Bound<'py, PyAny>,
    eq: &Bound<'py, PyAny>,
    variables: Option<&Bound<'py, PyAny>>,
    ranking: Option<&Bound<'py, PyAny>>,
    parameters: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let empty = PyTuple::empty(py).into_any();
    let ring = differential_ring(functions, variables, ranking, parameters.unwrap_or(&empty))?;
    let terms = ring.terms(&equation(eq)?).map_err(crate::convert::err)?;
    let mut items: Vec<Bound<'py, PyTuple>> = Vec::new();
    for (m, c) in terms.iter() {
        let c_expr: Basic = ring.coefficient_expr(c).map_err(crate::convert::err)?;
        items.push(PyTuple::new(
            py,
            [jet_monomial(py, m)?.into_any(), to_python(py, &c_expr)?],
        )?);
    }
    PyList::new(py, items)
}

/// `...integral(ring.terms(eq))` as `[(monomial, integer coefficient expression)]`.
#[pyfunction]
#[pyo3(signature = (functions, eq, variables=None, ranking=None, parameters=None))]
fn differential_integral<'py>(
    py: Python<'py>,
    functions: &Bound<'py, PyAny>,
    eq: &Bound<'py, PyAny>,
    variables: Option<&Bound<'py, PyAny>>,
    ranking: Option<&Bound<'py, PyAny>>,
    parameters: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let empty = PyTuple::empty(py).into_any();
    let ring = differential_ring(functions, variables, ranking, parameters.unwrap_or(&empty))?;
    let terms = ring.integral(&ring.terms(&equation(eq)?).map_err(crate::convert::err)?);
    let mut items: Vec<Bound<'py, PyTuple>> = Vec::new();
    for (m, c) in terms.iter() {
        let c_expr: Basic = ring.integer_expr(c, None).map_err(crate::convert::err)?;
        items.push(PyTuple::new(
            py,
            [jet_monomial(py, m)?.into_any(), to_python(py, &c_expr)?],
        )?);
    }
    PyList::new(py, items)
}

/// `...leader(eq)` and the expression of the terms (`from_terms`).
#[pyfunction]
#[pyo3(signature = (functions, eq, variables=None, ranking=None, parameters=None))]
fn differential_leader<'py>(
    py: Python<'py>,
    functions: &Bound<'py, PyAny>,
    eq: &Bound<'py, PyAny>,
    variables: Option<&Bound<'py, PyAny>>,
    ranking: Option<&Bound<'py, PyAny>>,
    parameters: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let empty = PyTuple::empty(py).into_any();
    let ring = differential_ring(functions, variables, ranking, parameters.unwrap_or(&empty))?;
    let e: Basic = equation(eq)?;
    let leader: Basic = ring.leader(&e).map_err(crate::convert::err)?;
    let back: Basic = ring
        .from_terms(&ring.terms(&e).map_err(crate::convert::err)?)
        .map_err(crate::convert::err)?;
    Ok((to_python(py, &leader)?, to_python(py, &back)?))
}

/// The operations of `DifferentialPolynomial.from_expr(ring, eq)` as a
/// dict of expressions: `to_expr`, `leader`, `degree`, `initial`,
/// `separant`, `derivative_<i>`, and the pseudo-, partial and full
/// remainders by the polynomials of `chain` (whose leaders must differ).
#[pyfunction]
#[pyo3(signature = (functions, eq, chain, variables=None, ranking=None, parameters=None))]
fn differential_polynomial_ops<'py>(
    py: Python<'py>,
    functions: &Bound<'py, PyAny>,
    eq: &Bound<'py, PyAny>,
    chain: &Bound<'py, PyAny>,
    variables: Option<&Bound<'py, PyAny>>,
    ranking: Option<&Bound<'py, PyAny>>,
    parameters: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::polys::differential::polynomial::DifferentialPolynomial as P;
    let e_ = crate::convert::err;
    let empty = PyTuple::empty(py).into_any();
    let ring = std::rc::Rc::new(differential_ring(
        functions,
        variables,
        ranking,
        parameters.unwrap_or(&empty),
    )?);
    let p: P = P::from_expr(ring.clone(), &equation(eq)?).map_err(e_)?;
    let mut members: Vec<P> = Vec::new();
    for c in chain.try_iter()? {
        members.push(P::from_expr(ring.clone(), &equation(&c?)?).map_err(e_)?);
    }
    let d: Bound<'py, PyDict> = PyDict::new(py);
    d.set_item("to_expr", to_python(py, &p.to_expr().map_err(e_)?)?)?;
    if !p.is_ground() {
        d.set_item("leader", to_python(py, &p.leader_expr().map_err(e_)?)?)?;
        d.set_item("degree", p.degree().map_err(e_)?)?;
        d.set_item(
            "initial",
            to_python(py, &p.initial().map_err(e_)?.to_expr().map_err(e_)?)?,
        )?;
        d.set_item(
            "separant",
            to_python(py, &p.separant().map_err(e_)?.to_expr().map_err(e_)?)?,
        )?;
    }
    for i in 0..ring.n() {
        d.set_item(
            format!("derivative_{}", i),
            to_python(py, &p.derivative(i).to_expr().map_err(e_)?)?,
        )?;
    }
    if let Some(first) = members.first() {
        if !first.is_ground() {
            let jet = first.leader().map_err(e_)?;
            d.set_item(
                "pseudo_remainder",
                to_python(py, &p.pseudo_remainder(first, &jet).to_expr().map_err(e_)?)?,
            )?;
        }
    }
    if members.iter().all(|m| !m.is_ground()) && !members.is_empty() {
        d.set_item(
            "partial_remainder",
            to_python(py, &p.partial_remainder(&members).map_err(e_)?.to_expr().map_err(e_)?)?,
        )?;
        d.set_item(
            "remainder",
            to_python(py, &p.remainder(&members).map_err(e_)?.to_expr().map_err(e_)?)?,
        )?;
    }
    Ok(d)
}

/// The queries of `janet_basis(equations, functions, variables, ranking)`
/// as a dict: `equations`, `leaders`, `multiplicative_variables`,
/// `is_consistent`, and for a consistent basis `dimension`,
/// `hilbert_function` (orders up to `order`), `hilbert_series` (in `t`),
/// `hilbert_polynomial` (in `s`), `parametric_derivatives`, `reduce` and
/// `contains` of the `probes`, `series_solution` (absent when it needs the
/// hints of `expand` which are not ported).
#[pyfunction]
#[pyo3(signature = (equations, functions, t, s, probes, order, variables=None, ranking=None))]
#[allow(clippy::too_many_arguments)]
fn janet_basis_ops<'py>(
    py: Python<'py>,
    equations: &Bound<'py, PyAny>,
    functions: &Bound<'py, PyAny>,
    t: &Bound<'py, PyAny>,
    s: &Bound<'py, PyAny>,
    probes: &Bound<'py, PyAny>,
    order: u32,
    variables: Option<&Bound<'py, PyAny>>,
    ranking: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::polys::differential::janet::janet_basis;
    let e_ = crate::convert::err;
    let mut eqs: Vec<Basic> = Vec::new();
    for e in equations.try_iter()? {
        eqs.push(equation(&e?)?);
    }
    let variables: Option<Vec<Basic>> = variables.map(expressions).transpose()?;
    let ranking: Option<Vec<Vec<Basic>>> = match ranking {
        None => None,
        Some(r) => {
            let mut blocks: Vec<Vec<Basic>> = Vec::new();
            for block in r.try_iter()? {
                blocks.push(expressions(&block?)?);
            }
            Some(blocks)
        }
    };
    let basis = janet_basis(&eqs, &expressions(functions)?, variables.as_deref(), ranking.as_deref()).map_err(e_)?;
    let d: Bound<'py, PyDict> = PyDict::new(py);
    d.set_item("equations", expression_list(py, &basis.equations().map_err(e_)?)?)?;
    d.set_item("leaders", expression_list(py, &basis.leaders())?)?;
    let mv: Vec<Bound<'py, PyTuple>> = basis
        .multiplicative_variables()
        .iter()
        .map(|vs| PyTuple::new(py, vs.iter().map(|v| to_python(py, v)).collect::<PyResult<Vec<_>>>()?))
        .collect::<PyResult<_>>()?;
    d.set_item("multiplicative_variables", PyList::new(py, mv)?)?;
    d.set_item("is_consistent", basis.is_consistent())?;
    if basis.is_consistent() {
        d.set_item("dimension", to_python(py, &basis.dimension().map_err(e_)?)?)?;
        let mut hf: Vec<Bound<'py, PyAny>> = Vec::new();
        for q in 0..=order {
            hf.push(crate::convert::out_int(py, &basis.hilbert_function(q).map_err(e_)?)?);
        }
        d.set_item("hilbert_function", PyList::new(py, hf)?)?;
        d.set_item(
            "hilbert_series",
            to_python(py, &basis.hilbert_series(&from_python(t)?).map_err(e_)?)?,
        )?;
        d.set_item(
            "hilbert_polynomial",
            to_python(py, &basis.hilbert_polynomial(&from_python(s)?).map_err(e_)?)?,
        )?;
        d.set_item(
            "parametric_derivatives",
            expression_list(py, &basis.parametric_derivatives(order).map_err(e_)?)?,
        )?;
        let mut reduced: Vec<Basic> = Vec::new();
        let mut contained: Vec<bool> = Vec::new();
        for p in probes.try_iter()? {
            let p: Basic = equation(&p?)?;
            reduced.push(basis.reduce(&p).map_err(e_)?);
            contained.push(basis.contains(&p).map_err(e_)?);
        }
        d.set_item("reduce", expression_list(py, &reduced)?)?;
        d.set_item("contains", contained)?;
        match basis.series_solution(order, None, "C") {
            Ok(series) => {
                let out: Bound<'py, PyDict> = PyDict::new(py);
                for (f, e) in series {
                    out.set_item(to_python(py, &f)?, to_python(py, &e)?)?;
                }
                d.set_item("series_solution", out)?;
            }
            Err(sympy_extras_core::error::PolyError::NotImplemented(_)) => {}
            Err(sympy_extras_core::error::PolyError::Value(message)) => d.set_item("series_solution_error", message)?,
            Err(other) => return Err(e_(other)),
        }
    }
    Ok(d)
}

/// `functionfields.integral_basis(F, x, y)`.
#[pyfunction]
fn integral_basis<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    use sympy_extras_core::polys::functionfields::integral_basis as basis;
    let found = basis(&from_python(f)?, &from_python(x)?, &from_python(y)?).map_err(crate::convert::err)?;
    expression_list(py, &found)
}

/// The bases of `function_field(F, x, y, QQ)` as expressions:
/// `(integral, local at infinity, normal, pole orders of the normal one)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn function_field_bases<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>, Vec<i64>)> {
    use sympy_extras_core::polys::domains::rationalfield::QQ;
    use sympy_extras_core::polys::functionfields::function_field;
    let e_ = crate::convert::err;
    let field = function_field(&from_python(f)?, &from_python(x)?, &from_python(y)?, QQ).map_err(e_)?;
    let shown = |basis: &[sympy_extras_core::polys::functionfields::Element<_>]| -> PyResult<Vec<Basic>> {
        basis.iter().map(|w| field.as_expr(w).map_err(e_)).collect()
    };
    let integral = shown(&field.integral_basis().map_err(e_)?)?;
    let local = shown(&field.local_basis_at_infinity().map_err(e_)?)?;
    let normal_basis = field.normal_basis().map_err(e_)?;
    let normal = shown(&normal_basis)?;
    let orders = field.pole_orders_at_infinity(&normal_basis).map_err(e_)?;
    Ok((
        expression_list(py, &integral)?,
        expression_list(py, &local)?,
        expression_list(py, &normal)?,
        orders,
    ))
}

fn pisigma_report<'py, Cs: sympy_extras_core::concrete::pisigma::Constants>(
    py: Python<'py>,
    base: sympy_extras_core::concrete::pisigma::PiSigmaField<Cs>,
    extensions: &Bound<'py, PyAny>,
    a: &Bound<'py, PyAny>,
    fs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::concrete::pisigma::{Kind, PiSigmaElement};
    let e_ = crate::convert::err;
    let mut field = base;
    for item in extensions.try_iter()? {
        let item = item?;
        let kind: String = item.get_item(0)?.extract()?;
        let symbol: Basic = from_python(&item.get_item(1)?)?;
        let value: PiSigmaElement = field.from_expr(&from_python(&item.get_item(2)?)?).map_err(e_)?;
        // the sequence when the Rust core has its functions, else the generator itself
        let sequence: Basic = from_python(&item.get_item(3)?).unwrap_or_else(|_| symbol.clone());
        let kind = if kind == "pi" { Kind::Pi } else { Kind::Sigma };
        field = field
            .extend(kind, &value, &sequence, None, Some(&symbol))
            .map_err(e_)?
            .0;
    }
    let shown = |f: &PiSigmaElement| field.to_expr(f, false, false).map_err(e_);
    let a: PiSigmaElement = field.from_expr(&from_python(a)?).map_err(e_)?;
    let mut elements: Vec<PiSigmaElement> = Vec::new();
    for f in fs.try_iter()? {
        elements.push(field.from_expr(&from_python(&f?)?).map_err(e_)?);
    }
    let d: Bound<'py, PyDict> = PyDict::new(py);
    let mut shifted: Vec<Basic> = Vec::new();
    let mut levels: Vec<i64> = Vec::new();
    let mut telescoped: Vec<Option<Bound<'py, PyAny>>> = Vec::new();
    for f in &elements {
        shifted.push(shown(&field.sigma(f, 1).map_err(e_)?)?);
        shifted.push(shown(&field.sigma(f, -2).map_err(e_)?)?);
        levels.push(field.level_of(f));
        telescoped.push(match field.telescope(f).map_err(e_)? {
            Some(g) => Some(to_python(py, &shown(&g)?)?),
            None => None,
        });
    }
    d.set_item("sigma", expression_list(py, &shifted)?)?;
    d.set_item("level_of", levels)?;
    d.set_item("telescope", telescoped)?;
    match field.solve(&a, &elements, None).map_err(e_)? {
        None => d.set_item("solve", py.None())?,
        Some(sols) => {
            let mut out: Vec<Bound<'py, PyTuple>> = Vec::new();
            for (c, g) in sols {
                let mut constants: Vec<Basic> = Vec::new();
                for ci in &c {
                    constants.push(field.constant_to_expr(ci).map_err(e_)?);
                }
                out.push(PyTuple::new(
                    py,
                    [expression_list(py, &constants)?.into_any(), to_python(py, &shown(&g)?)?],
                )?);
            }
            d.set_item("solve", PyList::new(py, out)?)?;
        }
    }
    Ok(d)
}

/// The ΠΣ-field over `QQ(params)` (`QQ` without parameters) with the
/// `extensions` `(kind, symbol, value, sequence)` as built in Python, and
/// for the elements `fs`: their shifts by `1` and `-2`, levels, telescoped
/// sums, and the basis of `solve(a, fs)` (all with `substitute=False`).
#[pyfunction]
fn pisigma_report_ops<'py>(
    py: Python<'py>,
    k: &Bound<'py, PyAny>,
    params: &Bound<'py, PyAny>,
    extensions: &Bound<'py, PyAny>,
    a: &Bound<'py, PyAny>,
    fs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::concrete::pisigma::{parametric_pisigma_field, rational_pisigma_field};
    let e_ = crate::convert::err;
    let k: Basic = from_python(k)?;
    let params: Vec<Basic> = expressions(params)?;
    if params.is_empty() {
        pisigma_report(py, rational_pisigma_field(&k).map_err(e_)?, extensions, a, fs)
    } else {
        pisigma_report(
            py,
            parametric_pisigma_field(&k, &params).map_err(e_)?,
            extensions,
            a,
            fs,
        )
    }
}

/// `expand(e, deep, mul, multinomial, power_base, power_exp, log, force)`.
#[pyfunction]
#[pyo3(signature = (e, deep=true, mul=true, multinomial=true, power_base=true, power_exp=true, log=true, force=false, complex=false))]
#[allow(clippy::too_many_arguments)]
fn expand<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    deep: bool,
    mul: bool,
    multinomial: bool,
    power_base: bool,
    power_exp: bool,
    log: bool,
    force: bool,
    complex: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use rust_core::function::{expand_with, Hints};
    let hints = Hints {
        mul,
        multinomial,
        power_base,
        power_exp,
        log,
        force,
        complex,
    };
    to_python(
        py,
        &expand_with(&from_python(e)?, deep, hints).map_err(crate::convert::err)?,
    )
}

/// The operations of `Poly(e, *gens)` and `Poly(other, *gens)` as
/// strings (`str` of each result, or the error).
#[pyfunction]
fn poly_report<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    other: &Bound<'py, PyAny>,
    gens: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::polys::polytools::Poly;
    let gens: Vec<Basic> = gens.iter().map(from_python).collect::<PyResult<_>>()?;
    let (e, other) = (from_python(e)?, from_python(other)?);
    let out = PyDict::new(py);
    let shown = |r: Result<String, sympy_extras_core::error::PolyError>| -> String {
        match r {
            Ok(s) => s,
            Err(err) => format!(
                "ERROR {}",
                crate::convert::err(err)
                    .get_type(py)
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ),
        }
    };
    let f = match Poly::new(&e, &gens) {
        Ok(f) => f,
        Err(err) => {
            out.set_item("poly", shown(Err(err)))?;
            return Ok(out);
        }
    };
    out.set_item("poly", shown(f.to_str()))?;
    out.set_item("gens", f.gens.iter().map(|g| g.to_string()).collect::<Vec<_>>())?;
    out.set_item("as_expr", shown(f.as_expr().map(|x| x.to_string())))?;
    out.set_item(
        "degrees",
        (0..f.gens.len()).map(|i| f.degree(i).unwrap_or(-2)).collect::<Vec<_>>(),
    )?;
    out.set_item("total_degree", f.total_degree())?;
    out.set_item(
        "terms",
        shown(f.terms().map(|t| {
            format!(
                "{:?}",
                t.iter().map(|(m, c)| (m.clone(), c.to_string())).collect::<Vec<_>>()
            )
        })),
    )?;
    out.set_item("pow", shown(f.pow(2).to_str()))?;
    out.set_item("diff", shown(f.diff(0).to_str()))?;
    use sympy_extras_core::polys::polytools::PolyOrExpr;
    let either = |r: Result<PolyOrExpr, sympy_extras_core::error::PolyError>| match r {
        Ok(PolyOrExpr::Poly(p)) => p.to_str(),
        Ok(PolyOrExpr::Expr(x)) => Ok(x.to_string()),
        Err(e) => Err(e),
    };
    let listed = |r: Result<(Basic, Vec<(Poly, usize)>), sympy_extras_core::error::PolyError>| {
        r.and_then(|(c, fs)| {
            let mut parts: Vec<String> = vec![c.to_string()];
            for (g, k) in fs {
                parts.push(format!("({}, {})", g.to_str()?, k));
            }
            Ok(parts.join(" "))
        })
    };
    out.set_item("monic", shown(f.monic().and_then(|h| h.to_str())))?;
    out.set_item("content", shown(f.content().map(|c| c.to_string())))?;
    out.set_item(
        "primitive",
        shown(f.primitive().and_then(|(c, h)| Ok(format!("{} {}", c, h.to_str()?)))),
    )?;
    out.set_item("eval2", shown(either(f.eval(0, &Basic::int(2)))))?;
    out.set_item("discriminant", shown(either(f.discriminant())))?;
    out.set_item("sqf_list", shown(listed(f.sqf_list())))?;
    out.set_item("factor_list", shown(listed(f.factor_list())))?;
    let g = match Poly::new(&other, &f.gens) {
        Ok(g) => g,
        Err(err) => {
            out.set_item("other", shown(Err(err)))?;
            return Ok(out);
        }
    };
    out.set_item("other", shown(g.to_str()))?;
    out.set_item("add", shown(f.add(&g).and_then(|h| h.to_str())))?;
    out.set_item("sub", shown(f.sub(&g).and_then(|h| h.to_str())))?;
    out.set_item("mul", shown(f.mul(&g).and_then(|h| h.to_str())))?;
    out.set_item("gcd", shown(f.gcd(&g).and_then(|h| h.to_str())))?;
    out.set_item("lcm", shown(f.lcm(&g).and_then(|h| h.to_str())))?;
    out.set_item(
        "cofactors",
        shown(
            f.cofactors(&g)
                .and_then(|(a, b, c)| Ok(format!("{} {} {}", a.to_str()?, b.to_str()?, c.to_str()?))),
        ),
    )?;
    out.set_item(
        "cancel",
        shown(
            f.cancel(&g)
                .and_then(|(c, p, q)| Ok(format!("{} {} {}", c, p.to_str()?, q.to_str()?))),
        ),
    )?;
    out.set_item(
        "div",
        shown(
            f.div(&g)
                .and_then(|(q, r)| Ok(format!("{} {}", q.to_str()?, r.to_str()?))),
        ),
    )?;
    out.set_item("resultant", shown(either(f.resultant(&g))))?;
    out.set_item("eval_other", shown(either(f.eval(0, &other))))?;
    out.set_item(
        "exquo",
        shown(f.mul(&g).and_then(|h| h.exquo(&g)).and_then(|h| h.to_str())),
    )?;
    Ok(out)
}

/// The functions of `polytools` on expressions and `signsimp`:
/// `polytools_call(name, args, gens)`.
#[pyfunction]
fn polytools_call<'py>(
    py: Python<'py>,
    name: &str,
    args: Vec<Bound<'py, PyAny>>,
    gens: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    use sympy_extras_core::polys::polytools as pt;
    let args: Vec<Basic> = args.iter().map(from_python).collect::<PyResult<_>>()?;
    let gens: Vec<Basic> = gens.iter().map(from_python).collect::<PyResult<_>>()?;
    let e = crate::convert::err;
    let one = |r: Result<Basic, sympy_extras_core::error::PolyError>| -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &r.map_err(e)?)
    };
    match name {
        "signsimp" => one(sympy_extras_core::simplify::signsimp::signsimp(&args[0], true)),
        "signsimp_unevaluated" => one(sympy_extras_core::simplify::signsimp::signsimp(&args[0], false)),
        "cancel" => one(pt::cancel(&args[0], &gens)),
        "invert" => one(pt::invert(&args[0], &args[1])),
        "gcd" => one(pt::gcd_expr(&args[0], &args[1], &gens)),
        "lcm" => one(pt::lcm_expr(&args[0], &args[1], &gens)),
        "gcd_list" => one(pt::gcd_list(&args, &gens)),
        "lcm_list" => one(pt::lcm_list(&args, &gens)),
        "resultant" => one(pt::resultant_expr(&args[0], &args[1], &gens)),
        "discriminant" => one(pt::discriminant_expr(&args[0], &gens)),
        "LC" => one(pt::LC(&args[0], &gens)),
        "degree" => {
            let d = if gens.is_empty() {
                pt::degree_index(&args[0], 0)
            } else {
                pt::degree(&args[0], &gens[0])
            };
            Ok(d.map_err(e)?.into_pyobject(py)?.into_any())
        }
        "cancel_fraction" => {
            let (p, q) = pt::cancel_fraction(&args[0], &args[1]).map_err(e)?;
            Ok(PyTuple::new(py, [to_python(py, &p)?, to_python(py, &q)?])?.into_any())
        }
        "factor" => one(pt::factor(&args[0], &gens, false, true)),
        "factor_deep" => one(pt::factor(&args[0], &gens, true, true)),
        "factor_nofraction" => one(pt::factor(&args[0], &gens, false, false)),
        "sqf" => one(pt::sqf(&args[0], &gens, true)),
        "minimal_polynomial" => one(sympy_extras_core::polys::numberfields::minpoly::minimal_polynomial(
            &args[0], &gens[0],
        )),
        "ratsimp" => one(sympy_extras_core::simplify::ratsimp::ratsimp(&args[0])),
        "apart" => one(sympy_extras_core::polys::partfrac::apart(&args[0], gens.first(), false)),
        "apart_full" => one(sympy_extras_core::polys::partfrac::apart(&args[0], gens.first(), true)),
        "factor_list" | "sqf_list" | "factor_list_fraction" | "sqf_list_fraction" => {
            let factors = |py: Python<'py>, fs: Vec<(Basic, Basic)>| -> PyResult<Bound<'py, PyAny>> {
                let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
                for (g, k) in fs {
                    items.push(PyTuple::new(py, [to_python(py, &g)?, to_python(py, &k)?])?.into_any());
                }
                Ok(PyList::new(py, items)?.into_any())
            };
            let method = if name.starts_with("sqf") {
                pt::FactorMethod::Sqf
            } else {
                pt::FactorMethod::Factor
            };
            if name.ends_with("fraction") {
                let (c, fp, fq) = pt::factor_list_fraction(&args[0], &gens, method).map_err(e)?;
                return Ok(PyTuple::new(py, [to_python(py, &c)?, factors(py, fp)?, factors(py, fq)?])?.into_any());
            }
            let (c, fs) = if method == pt::FactorMethod::Sqf {
                pt::sqf_list(&args[0], &gens)
            } else {
                pt::factor_list(&args[0], &gens)
            }
            .map_err(e)?;
            Ok(PyTuple::new(py, [to_python(py, &c)?, factors(py, fs)?])?.into_any())
        }
        _ => Err(PyTypeError::new_err(format!("unknown polytools function {}", name))),
    }
}

/// `roots(f, *gens, cubics, quartics, multiple, filter)`: a dict of the
/// roots with their multiplicities, or the list with `multiple`.
#[pyfunction]
#[pyo3(signature = (f, gens, cubics=true, quartics=true, multiple=false, filter=None))]
fn roots<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    gens: Vec<Bound<'py, PyAny>>,
    cubics: bool,
    quartics: bool,
    multiple: bool,
    filter: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    use sympy_extras_core::polys::polyroots;
    let gens: Vec<Basic> = gens.iter().map(from_python).collect::<PyResult<_>>()?;
    let f: Basic = from_python(f)?;
    let e = crate::convert::err;
    if multiple {
        let listed = polyroots::roots_list(&f, &gens, cubics, quartics, filter).map_err(e)?;
        let items: Vec<Bound<'py, PyAny>> = listed.iter().map(|r| to_python(py, r)).collect::<PyResult<_>>()?;
        return Ok(PyList::new(py, items)?.into_any());
    }
    let found = polyroots::roots(&f, &gens, cubics, quartics, filter).map_err(e)?;
    let out = PyDict::new(py);
    for (r, m) in found.iter() {
        out.set_item(to_python(py, r)?, *m)?;
    }
    Ok(out.into_any())
}

/// The expressions sorted by the canonical key (`sort_key`).
#[pyfunction]
fn sorted_by_key<'py>(py: Python<'py>, items: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyList>> {
    let mut items: Vec<Basic> = items.iter().map(from_python).collect::<PyResult<_>>()?;
    items.sort_by_cached_key(rust_core::sorting::sort_key);
    let out: Vec<Bound<'py, PyAny>> = items.iter().map(|e| to_python(py, e)).collect::<PyResult<_>>()?;
    PyList::new(py, out)
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
    m.add_function(wrap_pyfunction!(together, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(subs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear_unique, m)?)?;
    m.add_function(wrap_pyfunction!(linear_root, m)?)?;
    m.add_function(wrap_pyfunction!(solve_one, m)?)?;
    m.add_function(wrap_pyfunction!(numeric_linear_solution, m)?)?;
    m.add_function(wrap_pyfunction!(differential_terms, m)?)?;
    m.add_function(wrap_pyfunction!(differential_integral, m)?)?;
    m.add_function(wrap_pyfunction!(differential_leader, m)?)?;
    m.add_function(wrap_pyfunction!(differential_polynomial_ops, m)?)?;
    m.add_function(wrap_pyfunction!(janet_basis_ops, m)?)?;
    m.add_function(wrap_pyfunction!(integral_basis, m)?)?;
    m.add_function(wrap_pyfunction!(function_field_bases, m)?)?;
    m.add_function(wrap_pyfunction!(pisigma_report_ops, m)?)?;
    m.add_function(wrap_pyfunction!(expand, m)?)?;
    m.add_function(wrap_pyfunction!(poly_report, m)?)?;
    m.add_function(wrap_pyfunction!(polytools_call, m)?)?;
    m.add_function(wrap_pyfunction!(roots, m)?)?;
    m.add_function(wrap_pyfunction!(sorted_by_key, m)?)?;
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
    m.add_function(wrap_pyfunction!(arg, m)?)?;
    m.add_function(wrap_pyfunction!(atan2, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;
    m.add_function(wrap_pyfunction!(acot, m)?)?;
    m.add_function(wrap_pyfunction!(asec, m)?)?;
    m.add_function(wrap_pyfunction!(acsc, m)?)?;
    m.add_function(wrap_pyfunction!(asinh, m)?)?;
    m.add_function(wrap_pyfunction!(acosh, m)?)?;
    m.add_function(wrap_pyfunction!(atanh, m)?)?;
    m.add_function(wrap_pyfunction!(acoth, m)?)?;
    m.add_function(wrap_pyfunction!(asech, m)?)?;
    m.add_function(wrap_pyfunction!(acsch, m)?)?;

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
