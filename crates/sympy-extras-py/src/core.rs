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
use sympy_extras_core::core::basic::RelOp;
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
        "CatalanType" => Basic::catalan(),
        "GoldenRatioType" | "GoldenRatio" => Basic::golden_ratio(),
        "TribonacciConstantType" | "TribonacciConstant" => Basic::tribonacci_constant(),
        "ImaginaryUnit" => Basic::imaginary_unit(),
        "Symbol" => Symbol::new(&ob.getattr("name")?.extract::<String>()?, &given_assumptions(ob)?).0,
        "Wild" => {
            if ob.getattr("properties")?.len()? > 0 {
                return Err(PyTypeError::new_err("the properties of a wild are not converted"));
            }
            let mut exclude: Vec<Basic> = Vec::new();
            for item in ob.getattr("exclude")?.try_iter()? {
                exclude.push(from_python(&item?)?);
            }
            let name: String = ob.getattr("name")?.extract()?;
            sympy_extras_core::core::Wild::new(&name, exclude, Vec::new(), &given_assumptions(ob)?).0
        }
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
        "BooleanTrue" => sympy_extras_core::core::logic::true_(),
        "BooleanFalse" => sympy_extras_core::core::logic::false_(),
        "bool" => sympy_extras_core::core::logic::boolean(ob.extract::<bool>()?),
        "And" => Basic::new(Head::And, args()?),
        "Or" => Basic::new(Head::Or, args()?),
        "Not" => Basic::new(Head::Not, args()?),
        "Xor" => Basic::new(Head::Xor, args()?),
        "Implies" => Basic::new(Head::Implies, args()?),
        "Equivalent" => Basic::new(Head::Equivalent, args()?),
        "ITE" => Basic::new(Head::Ite, args()?),
        "Eq" | "Equality" => Basic::new(Head::Relational(RelOp::Eq), args()?),
        "Ne" | "Unequality" => Basic::new(Head::Relational(RelOp::Ne), args()?),
        "Lt" | "StrictLessThan" => Basic::new(Head::Relational(RelOp::Lt), args()?),
        "Le" | "LessThan" => Basic::new(Head::Relational(RelOp::Le), args()?),
        "Gt" | "StrictGreaterThan" => Basic::new(Head::Relational(RelOp::Gt), args()?),
        "Ge" | "GreaterThan" => Basic::new(Head::Relational(RelOp::Ge), args()?),
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
        "IndexedRoot" => Basic::new(Head::IndexedRoot, args()?),
        "EmptySetClass" => sympy_extras_core::sets::sets::empty_set(),
        "UniversalSetClass" => sympy_extras_core::sets::sets::universal_set(),
        "RealsClass" => sympy_extras_core::sets::sets::reals(),
        "Interval" => Basic::new(Head::Interval, args()?),
        "FiniteSet" => Basic::new(Head::FiniteSet, args()?),
        "Union" => Basic::new(Head::Union, args()?),
        "Intersection" => Basic::new(Head::Intersection, args()?),
        "Complement" => Basic::new(Head::Complement, args()?),
        "Contains" => Basic::new(Head::Contains, args()?),
        "ProductSet" => Basic::new(Head::ProductSet, args()?),
        "PointSet" => Basic::new(Head::PointSet, args()?),
        "SetOfSets" => Basic::new(Head::SetOfSets, args()?),
        "PowerSet" => Basic::new(Head::PowerSet, args()?),
        "NaturalsClass" => Basic::atom(Head::Naturals),
        "Naturals0Class" => Basic::atom(Head::Naturals0),
        "IntegersClass" => Basic::atom(Head::Integers),
        "RationalsClass" => Basic::atom(Head::Rationals),
        "ComplexesClass" => Basic::atom(Head::Complexes),
        "ImageSet" => Basic::new(Head::ImageSet, args()?),
        "Order" => Basic::new(Head::Order, args()?),
        "RootSum" => {
            // RootSum(f, Lambda(var, body), t): the Lambda flattened
            let fun = ob.getattr("fun")?;
            let var: Basic = from_python(&fun.getattr("variables")?.get_item(0)?)?;
            let body: Basic = from_python(&fun.getattr("expr")?)?;
            let f: Basic = from_python(&ob.getattr("expr")?)?;
            let t: Basic = from_python(&ob.getattr("variable")?)?;
            sympy_extras_core::polys::rootoftools::node(&f, &var, &body, &t)
        }
        "ComplexRootOf" => {
            let a: Vec<Basic> = args()?;
            sympy_extras_core::polys::complexrootof::complex_root_of(&a[0], &a[1]).map_err(crate::convert::err)?
        }
        "ConditionSet" => Basic::new(Head::ConditionSet, args()?),
        "AccumBounds" | "AccumulationBounds" => Basic::new(Head::AccumBounds, args()?),
        "Piecewise" => Basic::new(Head::Piecewise, args()?),
        "ExprCondPair" => Basic::new(Head::ExprCondPair, args()?),
        "Exp" => Basic::new(Head::Function(Function::Exp), args()?),
        "Log" => Basic::new(Head::Function(Function::Log), args()?),
        "LambertW" => Basic::new(Head::Function(Function::LambertW), args()?),
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
        "Max" => Basic::new(Head::Function(Function::Max), args()?),
        "Floor" => Basic::new(Head::Function(Function::Floor), args()?),
        "DiracDelta" => Basic::new(Head::Function(Function::DiracDelta), args()?),
        "Factorial" => Basic::new(Head::Function(Function::Factorial), args()?),
        "Binomial" => Basic::new(Head::Function(Function::Binomial), args()?),
        "RisingFactorial" => Basic::new(Head::Function(Function::RisingFactorial), args()?),
        "FallingFactorial" => Basic::new(Head::Function(Function::FallingFactorial), args()?),
        "Subfactorial" => Basic::new(Head::Function(Function::Subfactorial), args()?),
        "Factorial2" => Basic::new(Head::Function(Function::Factorial2), args()?),
        "Gamma" => Basic::new(Head::Function(Function::Gamma), args()?),
        "LogGamma" => Basic::new(Head::Function(Function::LogGamma), args()?),
        "PolyGamma" => Basic::new(Head::Function(Function::PolyGamma), args()?),
        "SingularityFunctionNode" => Basic::new(Head::Function(Function::SingularityFunction), args()?),
        "YnmNode" => Basic::new(Head::Function(Function::Ynm), args()?),
        "Legendre" => Basic::new(Head::Function(Function::Legendre), args()?),
        "ChebyshevT" => Basic::new(Head::Function(Function::ChebyshevT), args()?),
        "ChebyshevU" => Basic::new(Head::Function(Function::ChebyshevU), args()?),
        "Hermite" => Basic::new(Head::Function(Function::Hermite), args()?),
        "Laguerre" => Basic::new(Head::Function(Function::Laguerre), args()?),
        "AssocLaguerre" => Basic::new(Head::Function(Function::AssocLaguerre), args()?),
        "Jacobi" => Basic::new(Head::Function(Function::Jacobi), args()?),
        "Gegenbauer" => Basic::new(Head::Function(Function::Gegenbauer), args()?),
        "AssocLegendre" => Basic::new(Head::Function(Function::AssocLegendre), args()?),
        "HermiteProb" => Basic::new(Head::Function(Function::HermiteProb), args()?),
        "AppellF1" => Basic::new(Head::Function(Function::AppellF1), args()?),
        "BesselJ" => Basic::new(Head::Function(Function::BesselJ), args()?),
        "BesselY" => Basic::new(Head::Function(Function::BesselY), args()?),
        "BesselI" => Basic::new(Head::Function(Function::BesselI), args()?),
        "BesselK" => Basic::new(Head::Function(Function::BesselK), args()?),
        "Hankel1" => Basic::new(Head::Function(Function::Hankel1), args()?),
        "Hankel2" => Basic::new(Head::Function(Function::Hankel2), args()?),
        "BesselIs" => Basic::new(Head::Function(Function::BesselIs), args()?),
        "BesselKs" => Basic::new(Head::Function(Function::BesselKs), args()?),
        "SphericalJ" => Basic::new(Head::Function(Function::SphericalJ), args()?),
        "SphericalY" => Basic::new(Head::Function(Function::SphericalY), args()?),
        "SphericalHankel1" => Basic::new(Head::Function(Function::SphericalHankel1), args()?),
        "SphericalHankel2" => Basic::new(Head::Function(Function::SphericalHankel2), args()?),
        "AiryAi" => Basic::new(Head::Function(Function::AiryAi), args()?),
        "AiryBi" => Basic::new(Head::Function(Function::AiryBi), args()?),
        "AiryAiPrime" => Basic::new(Head::Function(Function::AiryAiPrime), args()?),
        "AiryBiPrime" => Basic::new(Head::Function(Function::AiryBiPrime), args()?),
        "MarcumQ" => Basic::new(Head::Function(Function::MarcumQ), args()?),
        "HyperRep_power1" => Basic::new(Head::Function(Function::HyperRepPower1), args()?),
        "HyperRep_power2" => Basic::new(Head::Function(Function::HyperRepPower2), args()?),
        "HyperRep_log1" => Basic::new(Head::Function(Function::HyperRepLog1), args()?),
        "HyperRep_atanh" => Basic::new(Head::Function(Function::HyperRepAtanh), args()?),
        "HyperRep_asin1" => Basic::new(Head::Function(Function::HyperRepAsin1), args()?),
        "HyperRep_asin2" => Basic::new(Head::Function(Function::HyperRepAsin2), args()?),
        "HyperRep_sqrts1" => Basic::new(Head::Function(Function::HyperRepSqrts1), args()?),
        "HyperRep_sqrts2" => Basic::new(Head::Function(Function::HyperRepSqrts2), args()?),
        "HyperRep_log2" => Basic::new(Head::Function(Function::HyperRepLog2), args()?),
        "HyperRep_cosasin" => Basic::new(Head::Function(Function::HyperRepCosasin), args()?),
        "HyperRep_sinasin" => Basic::new(Head::Function(Function::HyperRepSinasin), args()?),
        "Hyper" => Basic::new(Head::Hyper, args()?),
        "ForAll" => Basic::new(Head::ForAll, args()?),
        "Predicate" => {
            let name: String = ob.getattr("name")?.extract()?;
            let arity: usize = ob.getattr("_arity")?.extract()?;
            let defined: bool = ob.getattr("_defined")?.extract()?;
            if defined {
                sympy_extras_core::core::assume::q(&name, arity)
            } else {
                sympy_extras_core::core::assume::predicate(&name, arity)
            }
        }
        "AppliedPredicate" => Basic::new(Head::AppliedPredicate, args()?),
        "Exists" => Basic::new(Head::Exists, args()?),
        "MeijerG" => Basic::new(Head::MeijerG, args()?),
        "EllipticK" => Basic::new(Head::Function(Function::EllipticK), args()?),
        "EllipticF" => Basic::new(Head::Function(Function::EllipticF), args()?),
        "EllipticE" => Basic::new(Head::Function(Function::EllipticE), args()?),
        "EllipticPi" => Basic::new(Head::Function(Function::EllipticPi), args()?),
        "MathieuS" => Basic::new(Head::Function(Function::MathieuS), args()?),
        "MathieuC" => Basic::new(Head::Function(Function::MathieuC), args()?),
        "MathieuSPrime" => Basic::new(Head::Function(Function::MathieuSPrime), args()?),
        "MathieuCPrime" => Basic::new(Head::Function(Function::MathieuCPrime), args()?),
        "ZnmNode" => Basic::new(Head::Function(Function::Znm), args()?),
        "LowerGamma" => Basic::new(Head::Function(Function::LowerGamma), args()?),
        "UpperGamma" => Basic::new(Head::Function(Function::UpperGamma), args()?),
        "MultiGamma" => Basic::new(Head::Function(Function::MultiGamma), args()?),
        "DiGamma" => Basic::new(Head::Function(Function::DiGamma), args()?),
        "TriGamma" => Basic::new(Head::Function(Function::TriGamma), args()?),
        "Zeta" => Basic::new(Head::Function(Function::Zeta), args()?),
        "DirichletEta" => Basic::new(Head::Function(Function::DirichletEta), args()?),
        "PolyLog" => Basic::new(Head::Function(Function::PolyLog), args()?),
        "LerchPhi" => Basic::new(Head::Function(Function::LerchPhi), args()?),
        "RiemannXi" => Basic::new(Head::Function(Function::RiemannXi), args()?),
        "Stieltjes" => Basic::new(Head::Function(Function::Stieltjes), args()?),
        "Mod" => Basic::new(Head::Function(Function::Mod), args()?),
        "KroneckerDeltaNode" => Basic::new(Head::Function(Function::KroneckerDelta), args()?),
        "LeviCivitaNode" => Basic::new(Head::Function(Function::LeviCivita), args()?),
        "Catalan" => Basic::new(Head::Function(Function::Catalan), args()?),
        "Fibonacci" => Basic::new(Head::Function(Function::Fibonacci), args()?),
        "Lucas" => Basic::new(Head::Function(Function::Lucas), args()?),
        "Harmonic" => Basic::new(Head::Function(Function::Harmonic), args()?),
        "Bernoulli" => Basic::new(Head::Function(Function::Bernoulli), args()?),
        "Euler" => Basic::new(Head::Function(Function::Euler), args()?),
        "Bell" => Basic::new(Head::Function(Function::Bell), args()?),
        "Genocchi" => Basic::new(Head::Function(Function::Genocchi), args()?),
        "Partition" => Basic::new(Head::Function(Function::Partition), args()?),
        "Totient" => Basic::new(Head::Function(Function::Totient), args()?),
        "ReducedTotient" => Basic::new(Head::Function(Function::ReducedTotient), args()?),
        "DivisorSigma" => Basic::new(Head::Function(Function::DivisorSigma), args()?),
        "UDivisorSigma" => Basic::new(Head::Function(Function::UDivisorSigma), args()?),
        "PrimeNu" => Basic::new(Head::Function(Function::PrimeNu), args()?),
        "PrimeOmega" => Basic::new(Head::Function(Function::PrimeOmega), args()?),
        "Mobius" => Basic::new(Head::Function(Function::Mobius), args()?),
        "PrimePi" => Basic::new(Head::Function(Function::PrimePi), args()?),
        "LegendreSymbol" => Basic::new(Head::Function(Function::LegendreSymbol), args()?),
        "JacobiSymbol" => Basic::new(Head::Function(Function::JacobiSymbol), args()?),
        "KroneckerSymbol" => Basic::new(Head::Function(Function::KroneckerSymbol), args()?),
        "Tribonacci" => Basic::new(Head::Function(Function::Tribonacci), args()?),
        "Motzkin" => Basic::new(Head::Function(Function::Motzkin), args()?),
        "Andre" => Basic::new(Head::Function(Function::Andre), args()?),
        "Beta" => Basic::new(Head::Function(Function::Beta), args()?),
        "BetaInc" => Basic::new(Head::Function(Function::BetaInc), args()?),
        "BetaIncRegularized" => Basic::new(Head::Function(Function::BetaIncRegularized), args()?),
        "Erf" => Basic::new(Head::Function(Function::Erf), args()?),
        "Erfc" => Basic::new(Head::Function(Function::Erfc), args()?),
        "Erfi" => Basic::new(Head::Function(Function::Erfi), args()?),
        "Erfinv" => Basic::new(Head::Function(Function::Erfinv), args()?),
        "Erfcinv" => Basic::new(Head::Function(Function::Erfcinv), args()?),
        "Erf2" => Basic::new(Head::Function(Function::Erf2), args()?),
        "Erf2inv" => Basic::new(Head::Function(Function::Erf2inv), args()?),
        "Expint" => Basic::new(Head::Function(Function::Expint), args()?),
        "EiNode" => Basic::new(Head::Function(Function::Ei), args()?),
        "SiNode" => Basic::new(Head::Function(Function::Si), args()?),
        "CiNode" => Basic::new(Head::Function(Function::Ci), args()?),
        "ShiNode" => Basic::new(Head::Function(Function::Shi), args()?),
        "ChiNode" => Basic::new(Head::Function(Function::Chi), args()?),
        "LiNode" => Basic::new(Head::Function(Function::LogIntegral), args()?),
        "OffsetLiNode" => Basic::new(Head::Function(Function::OffsetLogIntegral), args()?),
        "FresnelS" => Basic::new(Head::Function(Function::FresnelS), args()?),
        "FresnelC" => Basic::new(Head::Function(Function::FresnelC), args()?),
        "Heaviside" => Basic::new(Head::Function(Function::Heaviside), args()?),
        "Ceiling" => Basic::new(Head::Function(Function::Ceiling), args()?),
        "Frac" => Basic::new(Head::Function(Function::Frac), args()?),
        "Min" => Basic::new(Head::Function(Function::Min), args()?),
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
        Head::Wild(data) => {
            if !data.properties.is_empty() {
                return Err(PyTypeError::new_err("the properties of a wild are not converted"));
            }
            let exclude: Vec<Bound<'py, PyAny>> =
                data.exclude.iter().map(|x| to_python(py, x)).collect::<PyResult<_>>()?;
            let kw = kwargs(py, &data.symbol.given)?;
            kw.set_item("exclude", PyTuple::new(py, exclude)?)?;
            class("symbol", "Wild")?.call((data.symbol.name.clone(),), Some(&kw))?
        }
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
        Head::BooleanTrue => class("logic", "true")?,
        Head::BooleanFalse => class("logic", "false")?,
        Head::And | Head::Or | Head::Not | Head::Xor | Head::Implies | Head::Equivalent | Head::Ite => {
            rebuilt(e.class_name(), "logic")?
        }
        Head::Relational(op) => rebuilt(op.class_name(), "relational")?,
        Head::EmptySet | Head::UniversalSet | Head::Reals => {
            let name: &str = match e.head() {
                Head::EmptySet => "EmptySet",
                Head::UniversalSet => "UniversalSet",
                _ => "Reals",
            };
            py.import("sympy_extras.sets.sets")?.getattr(name)?
        }
        Head::Interval
        | Head::FiniteSet
        | Head::Union
        | Head::Intersection
        | Head::Complement
        | Head::Contains
        | Head::ProductSet
        | Head::PointSet => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.sets.sets")?
                .getattr(e.class_name())?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::AccumBounds => rebuilt("AccumBounds", "accumbounds")?,
        Head::Piecewise | Head::ExprCondPair => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            let name: &str = if matches!(e.head(), Head::Piecewise) {
                "Piecewise"
            } else {
                "ExprCondPair"
            };
            py.import("sympy_extras.functions.elementary.piecewise")?
                .getattr(name)?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::Naturals | Head::Naturals0 | Head::Integers | Head::Rationals | Head::Complexes => py
            .import("sympy_extras.sets.fancysets")?
            .getattr(sympy_extras_core::sets::fancysets::set_name(e))?,
        Head::RootSum => {
            let (f, var, body, t) = sympy_extras_core::polys::rootoftools::parts(e);
            let lambda = py
                .import("sympy_extras.core.function")?
                .getattr("Lambda")?
                .call1((to_python(py, var)?, to_python(py, body)?))?;
            py.import("sympy_extras.polys.rootoftools")?
                .getattr("RootSum")?
                .call1((to_python(py, f)?, lambda, to_python(py, t)?))?
        }
        Head::Order => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.core.order")?
                .getattr("Order")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::ImageSet => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.sets.fancysets")?
                .getattr("ImageSet")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::ComplexRootOf(_) => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.polys.complexrootof")?
                .getattr("ComplexRootOf")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::SetOfSets => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.sets.sets")?
                .getattr("SetOfSets")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::PowerSet => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.sets.powerset")?
                .getattr("PowerSet")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::Predicate(p) => {
            let assume = core_module(py, "assume")?;
            if p.defined {
                let keys = core_module(py, "ask")?.getattr("Q")?;
                match keys.getattr(p.name.as_str()) {
                    Ok(found) => found,
                    Err(_) => assume
                        .getattr("Predicate")?
                        .getattr("_known")?
                        .call1((p.name.as_str(), p.arity))?,
                }
            } else {
                assume.getattr("Predicate")?.call1((p.name.as_str(), p.arity))?
            }
        }
        Head::AppliedPredicate => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            core_module(py, "assume")?
                .getattr("AppliedPredicate")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::ForAll | Head::Exists => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.assumptions.quantifiers")?
                .getattr(e.class_name())?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::Hyper => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.functions.special.hyper")?
                .getattr("Hyper")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::MeijerG => {
            let (first, second) = (&e.args()[0], &e.args()[1]);
            let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
            for t in [
                &first.args()[0],
                &first.args()[1],
                &second.args()[0],
                &second.args()[1],
                &e.args()[2],
            ] {
                items.push(to_python(py, t)?);
            }
            py.import("sympy_extras.functions.special.hyper")?
                .getattr("MeijerG")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::ConditionSet => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.sets.conditionset")?
                .getattr("ConditionSet")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::IndexedRoot => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.polys.cad.indexedroot")?
                .getattr("IndexedRoot")?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::Function(Function::Mod) => rebuilt("Mod", "mod")?,
        Head::Function(f @ (Function::KroneckerDelta | Function::LeviCivita)) => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            py.import("sympy_extras.functions.special.tensor_functions")?
                .getattr(f.class_name())?
                .call1(PyTuple::new(py, items)?)?
        }
        Head::Function(f) => {
            let items: Vec<Bound<'py, PyAny>> = e.args().iter().map(|a| to_python(py, a)).collect::<PyResult<_>>()?;
            let module: &str = match f {
                Function::Exp | Function::Log | Function::LambertW => "exponential",
                Function::Sin | Function::Cos | Function::Tan | Function::Cot | Function::Sec | Function::Csc => {
                    "trigonometric"
                }
                Function::Max | Function::Min => "miscellaneous",
                Function::Floor | Function::Ceiling | Function::Frac => "integers",
                Function::Catalan
                | Function::Fibonacci
                | Function::Lucas
                | Function::Harmonic
                | Function::Bernoulli
                | Function::Euler
                | Function::Bell
                | Function::Genocchi
                | Function::Partition
                | Function::Totient
                | Function::ReducedTotient
                | Function::DivisorSigma
                | Function::UDivisorSigma
                | Function::PrimeNu
                | Function::PrimeOmega
                | Function::Mobius
                | Function::PrimePi
                | Function::LegendreSymbol
                | Function::JacobiSymbol
                | Function::KroneckerSymbol
                | Function::Tribonacci
                | Function::Motzkin
                | Function::Andre => "numbers",
                Function::Beta | Function::BetaInc | Function::BetaIncRegularized => "beta_functions",
                Function::Erf
                | Function::Erfc
                | Function::Erfi
                | Function::Erfinv
                | Function::Erfcinv
                | Function::Erf2
                | Function::Erf2inv => "error_functions",
                Function::Expint
                | Function::Ei
                | Function::Si
                | Function::Ci
                | Function::Shi
                | Function::Chi
                | Function::LogIntegral
                | Function::OffsetLogIntegral
                | Function::FresnelS
                | Function::FresnelC => "error_functions",
                Function::SingularityFunction => "singularity_functions",
                Function::Ynm | Function::Znm => "spherical_harmonics",
                Function::Legendre
                | Function::ChebyshevT
                | Function::ChebyshevU
                | Function::Hermite
                | Function::Laguerre
                | Function::AssocLaguerre
                | Function::Jacobi
                | Function::Gegenbauer
                | Function::AssocLegendre
                | Function::HermiteProb => "polynomials",
                Function::AppellF1 => "hyper",
                f if sympy_extras_core::functions::special::hyperrep::is_hyperrep(*f) => "hyperrep",
                f if sympy_extras_core::functions::special::bessel::is_bessel(*f)
                    || sympy_extras_core::functions::special::bessel::is_airy(*f)
                    || *f == Function::MarcumQ =>
                {
                    "bessel"
                }
                Function::EllipticK | Function::EllipticF | Function::EllipticE | Function::EllipticPi => {
                    "elliptic_integrals"
                }
                Function::MathieuS | Function::MathieuC | Function::MathieuSPrime | Function::MathieuCPrime => {
                    "mathieu_functions"
                }
                Function::DiracDelta | Function::Heaviside => "delta_functions",
                Function::Zeta
                | Function::DirichletEta
                | Function::PolyLog
                | Function::LerchPhi
                | Function::RiemannXi
                | Function::Stieltjes => "zeta_functions",
                Function::LowerGamma | Function::UpperGamma | Function::MultiGamma => "gamma_functions",
                Function::Gamma | Function::LogGamma | Function::PolyGamma | Function::DiGamma | Function::TriGamma => {
                    "gamma_functions"
                }
                Function::Factorial
                | Function::Binomial
                | Function::RisingFactorial
                | Function::FallingFactorial
                | Function::Subfactorial
                | Function::Factorial2 => "factorials",
                f if sympy_extras_core::functions::elementary::inverse::is_inverse_trigonometric(*f) => "trigonometric",
                f if sympy_extras_core::functions::elementary::inverse::is_inverse(*f) => "hyperbolic",
                Function::Sinh | Function::Cosh | Function::Tanh | Function::Coth | Function::Sech | Function::Csch => {
                    "hyperbolic"
                }
                _ => "complexes",
            };
            let package: &str = if matches!(f, Function::DiracDelta | Function::Heaviside)
                || sympy_extras_core::functions::special::gamma_functions::is_gamma_function(*f)
                || sympy_extras_core::functions::special::gamma_functions::is_incomplete_gamma(*f)
                || *f == Function::SingularityFunction
                || sympy_extras_core::functions::special::spherical_harmonics::is_spherical_harmonic(*f)
                || sympy_extras_core::functions::special::mathieu_functions::is_mathieu(*f)
                || sympy_extras_core::functions::special::polynomials::is_orthogonal(*f)
                || *f == Function::AppellF1
                || sympy_extras_core::functions::special::hyperrep::is_hyperrep(*f)
                || sympy_extras_core::functions::special::bessel::is_bessel(*f)
                || sympy_extras_core::functions::special::bessel::is_airy(*f)
                || *f == Function::MarcumQ
                || sympy_extras_core::functions::special::elliptic_integrals::is_elliptic(*f)
                || sympy_extras_core::functions::special::zeta_functions::is_zeta_function(*f)
                || sympy_extras_core::functions::special::beta_functions::is_beta_function(*f)
                || sympy_extras_core::functions::special::error_functions::is_error_function(*f)
                || sympy_extras_core::functions::special::error_functions::is_integral_function(*f)
            {
                "special"
            } else if sympy_extras_core::functions::combinatorial::factorials::is_combinatorial(*f)
                || sympy_extras_core::functions::combinatorial::numbers::is_combinatorial_number(*f)
                || sympy_extras_core::functions::combinatorial::numbertheory::is_number_theoretic(*f)
            {
                "combinatorial"
            } else {
                "elementary"
            };
            py.import(format!("sympy_extras.functions.{}.{}", package, module))?
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
        &rust_core::add::sub_operator(&from_python(a)?, &from_python(b)?).map_err(crate::convert::err)?,
    )
}

#[pyfunction]
fn div<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::mul::div_operator(&from_python(a)?, &from_python(b)?).map_err(crate::convert::err)?,
    )
}

#[pyfunction]
fn neg<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_python(
        py,
        &rust_core::mul::neg_operator(&from_python(a)?).map_err(crate::convert::err)?,
    )
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
unary!(floor, sympy_extras_core::functions::elementary::integers::floor);

/// `dirac_delta(x, k)` (`k` omitted: one argument).
#[pyfunction]
#[pyo3(signature = (x, k=None))]
fn dirac_delta<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    k: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let k: Option<Basic> = k.as_ref().map(from_python).transpose()?;
    let value = sympy_extras_core::functions::special::delta_functions::dirac_delta(&from_python(x)?, k.as_ref())
        .map_err(crate::convert::err)?;
    to_python(py, &value)
}

/// `heaviside(x, H0)` (`H0` omitted: 1/2).
#[pyfunction]
#[pyo3(signature = (x, h0=None))]
fn heaviside<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    h0: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let h0: Option<Basic> = h0.as_ref().map(from_python).transpose()?;
    let value = sympy_extras_core::functions::special::delta_functions::heaviside(&from_python(x)?, h0.as_ref())
        .map_err(crate::convert::err)?;
    to_python(py, &value)
}
unary!(ceiling, sympy_extras_core::functions::elementary::integers::ceiling);
unary!(frac, sympy_extras_core::functions::elementary::integers::frac);
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
#[pyo3(signature = (e, deep=true, mul=true, multinomial=true, power_base=true, power_exp=true, log=true, force=false, complex=false, trig=false, func=false))]
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
    trig: bool,
    func: bool,
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
        trig,
        func,
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
        "solveset_rational" => {
            match sympy_extras_core::polys::rationalsolve::solveset_rational(&args[0], &gens[0]).map_err(e)? {
                None => Ok(py.None().into_bound(py)),
                Some((solutions, excluded)) => {
                    let a: Vec<Bound<'py, PyAny>> =
                        solutions.iter().map(|s| to_python(py, s)).collect::<PyResult<_>>()?;
                    let b: Vec<Bound<'py, PyAny>> =
                        excluded.iter().map(|s| to_python(py, s)).collect::<PyResult<_>>()?;
                    Ok(PyTuple::new(py, [PyList::new(py, a)?.into_any(), PyList::new(py, b)?.into_any()])?.into_any())
                }
            }
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

/// The CAD projection factor sets of `polys` in `gens`, each `Poly`
/// printed; with `extra`, the augmented sets of those.
#[pyfunction]
#[pyo3(signature = (polys, gens, method="mccallum", equational=None, extra=None))]
fn cad_projection_sets<'py>(
    polys: Vec<Bound<'py, PyAny>>,
    gens: Vec<Bound<'py, PyAny>>,
    method: &str,
    equational: Option<Bound<'py, PyAny>>,
    extra: Option<Vec<Bound<'py, PyAny>>>,
) -> PyResult<Vec<Vec<String>>> {
    use sympy_extras_core::polys::cad::projection as pr;
    let e = crate::convert::err;
    let polys: Vec<Basic> = polys.iter().map(from_python).collect::<PyResult<_>>()?;
    let gens: Vec<Basic> = gens.iter().map(from_python).collect::<PyResult<_>>()?;
    let method: pr::Method = pr::Method::from_name(method).map_err(e)?;
    let equational: Option<Basic> = equational.as_ref().map(from_python).transpose()?;
    let mut sets = pr::projection_sets(&polys, &gens, method, equational.as_ref()).map_err(e)?;
    if let Some(extra) = extra {
        let extra: Vec<Basic> = extra.iter().map(from_python).collect::<PyResult<_>>()?;
        sets = pr::augmented_projection_sets(&sets, &gens, &extra, method).map_err(e)?;
    }
    sets.iter()
        .map(|level| level.iter().map(|p| p.to_str().map_err(e)).collect())
        .collect()
}

/// A report of a CAD sample point (`samplepoints`): the point built from
/// `coordinates` (each `(dense, index)`: the root of that index of the
/// polynomial with the rational coefficients `dense`, given as
/// `(numerator, denominator)` pairs), the signs of `signs` (in all the
/// generators) and the specializations of `lifts` (in one more), and the
/// standalone functions on the coordinates.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn samplepoint_report<'py>(
    py: Python<'py>,
    coordinates: Vec<(Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>, usize)>,
    signs: Vec<Bound<'py, PyAny>>,
    lifts: Vec<Bound<'py, PyAny>>,
    gens: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::polys::cad::samplepoints as sp;
    let e = crate::convert::err;
    let gens: Vec<Basic> = gens.iter().map(from_python).collect::<PyResult<_>>()?;
    let out = PyDict::new(py);
    let mut point = sp::SamplePoint::origin();
    let mut roots: Vec<sp::Real> = Vec::new();
    for (dense, index) in coordinates {
        let mut f: Vec<Rat> = Vec::new();
        for (p, q) in dense {
            f.push(rat_from_parts(extract_int(&p)?, extract_int(&q)?));
        }
        let found = sp::real_roots_of(&f).map_err(e)?;
        let reprs: Vec<String> = found.iter().map(|r| r.repr()).collect::<Result<_, _>>().map_err(e)?;
        out.set_item(format!("roots{}", roots.len()), reprs)?;
        let r = found[index % found.len().max(1)].clone();
        roots.push(r.clone());
        point = point.extend(&r).map_err(e)?;
    }
    out.set_item("point", point.repr().map_err(e)?)?;
    out.set_item("degree", point.degree())?;
    let n = point.len();
    let mut sign_values: Vec<String> = Vec::new();
    for f in &signs {
        sign_values.push(match point.sign(&from_python(f)?, &gens[..n]) {
            Ok(s) => s.to_string(),
            Err(err) => format!("ERROR {}", crate::convert::err(err).get_type(py).name()?),
        });
    }
    out.set_item("signs", sign_values)?;
    let mut lift_values: Vec<String> = Vec::new();
    for f in &lifts {
        lift_values.push(match point.specialization(&from_python(f)?, &gens[..n + 1]) {
            Ok(s) => {
                let mut parts: Vec<String> = Vec::new();
                for (r, m) in &s.roots {
                    parts.push(format!("({}, {})", r.repr().map_err(e)?, m));
                }
                let mut before: Vec<String> = Vec::new();
                for i in 0..=s.roots.len() {
                    before.push(s.sign_before(i).map_err(e)?.to_string());
                }
                format!(
                    "{} {} [{}] [{}]",
                    s.degree,
                    s.leading_sign,
                    parts.join(", "),
                    before.join(", ")
                )
            }
            Err(err) => format!("ERROR {}", crate::convert::err(err).get_type(py).name()?),
        });
    }
    out.set_item("lifts", lift_values)?;
    let fraction = |q: Rat| format!("{}", q);
    let mut functions: Vec<String> = Vec::new();
    for r in &roots {
        functions.push(format!(
            "{} {} {} {} {}",
            sp::floor_scaled(r, 0).map_err(e)?,
            sp::floor_scaled(r, 5).map_err(e)?,
            fraction(sp::rational_below(r).map_err(e)?),
            fraction(sp::rational_above(r).map_err(e)?),
            sp::compare_real(r, &roots[0]).map_err(e)?
        ));
    }
    for (a, b) in roots.iter().zip(roots.iter().skip(1)) {
        functions.push(match sp::compare_real(a, b).map_err(e)? {
            -1 => fraction(sp::rational_between(a, b).map_err(e)?),
            1 => fraction(sp::rational_between(b, a).map_err(e)?),
            _ => "equal".to_string(),
        });
    }
    out.set_item("functions", functions)?;
    Ok(out)
}

/// The functions of `polys.cad.cylindrical` (`cylindrical_formula`,
/// `cylindrical_set`, `cylindrical_cases`; for the cases `gens` are the
/// parameters and `unknowns` the unknowns).
#[pyfunction]
#[pyo3(signature = (name, formula, gens, quantifiers=Vec::new(), unknowns=Vec::new(), method=None, partial=true))]
#[allow(clippy::too_many_arguments)]
fn cylindrical_call<'py>(
    py: Python<'py>,
    name: &str,
    formula: Bound<'py, PyAny>,
    gens: Vec<Bound<'py, PyAny>>,
    quantifiers: Vec<(String, Bound<'py, PyAny>)>,
    unknowns: Vec<Bound<'py, PyAny>>,
    method: Option<&str>,
    partial: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use sympy_extras_core::polys::cad::{cylindrical as cy, projection::Method, qe};
    let e = crate::convert::err;
    let method: Option<Method> = method.map(Method::from_name).transpose().map_err(e)?;
    let mut prefix: qe::QuantifierPrefix = Vec::new();
    for (kind, variables) in &quantifiers {
        let kind: qe::Quantifier = qe::Quantifier::from_name(kind).map_err(e)?;
        let class: String = variables.get_type().name()?.to_string();
        if class == "list" || class == "tuple" {
            for v in variables.try_iter()? {
                prefix.push((kind, from_python(&v?)?));
            }
        } else {
            prefix.push((kind, from_python(variables)?));
        }
    }
    let formula: Basic = from_python(&formula)?;
    let gens: Vec<Basic> = gens.iter().map(from_python).collect::<PyResult<_>>()?;
    let unknowns: Vec<Basic> = unknowns.iter().map(from_python).collect::<PyResult<_>>()?;
    Ok(match name {
        "cylindrical_formula" => to_python(
            py,
            &cy::cylindrical_formula(&formula, &gens, &prefix, method, partial).map_err(e)?,
        )?,
        "cylindrical_set" => to_python(py, &cy::cylindrical_set(&formula, &gens, method, partial).map_err(e)?)?,
        "cylindrical_cases" => {
            let cases = cy::cylindrical_cases(&formula, &gens, &unknowns, method, partial).map_err(e)?;
            let out = pyo3::types::PyList::empty(py);
            for (c, s) in cases {
                out.append(PyTuple::new(py, [to_python(py, &c)?, to_python(py, &s)?])?)?;
            }
            out.into_any()
        }
        _ => return Err(PyTypeError::new_err(format!("unknown cylindrical function {}", name))),
    })
}

/// The functions of `polys.cad.qe` (`quantifier_elimination`, `decide`,
/// `sample_points`, `solution_set`, `truth_tables`): `formula` is a list of
/// formulas for `truth_tables`, `free` the free variables (the generators
/// for `sample_points` and `truth_tables`, `[x]` for `solution_set`).
#[pyfunction]
#[pyo3(signature = (name, formula, quantifiers, free=None, method=None, partial=true))]
fn qe_call<'py>(
    py: Python<'py>,
    name: &str,
    formula: Bound<'py, PyAny>,
    quantifiers: Vec<(String, Bound<'py, PyAny>)>,
    free: Option<Vec<Bound<'py, PyAny>>>,
    method: Option<&str>,
    partial: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use sympy_extras_core::polys::cad::{projection::Method, qe};
    let e = crate::convert::err;
    let method: Option<Method> = method.map(Method::from_name).transpose().map_err(e)?;
    let mut prefix: qe::QuantifierPrefix = Vec::new();
    for (kind, variables) in &quantifiers {
        let kind: qe::Quantifier = qe::Quantifier::from_name(kind).map_err(e)?;
        let class: String = variables.get_type().name()?.to_string();
        if class == "list" || class == "tuple" {
            for v in variables.try_iter()? {
                prefix.push((kind, from_python(&v?)?));
            }
        } else {
            prefix.push((kind, from_python(variables)?));
        }
    }
    let free: Option<Vec<Basic>> = free
        .map(|f| f.iter().map(from_python).collect::<PyResult<_>>())
        .transpose()?;
    if name == "truth_tables" {
        let mut formulas: Vec<Basic> = Vec::new();
        for f in formula.try_iter()? {
            formulas.push(from_python(&f?)?);
        }
        let (cells, tables) = qe::truth_tables(&formulas, &free.unwrap_or_default(), method, partial).map_err(e)?;
        let shown: Vec<String> = cells.iter().map(|c| c.repr()).collect::<Result<_, _>>().map_err(e)?;
        return Ok(PyTuple::new(
            py,
            [
                shown.into_pyobject(py)?.into_any(),
                tables.into_pyobject(py)?.into_any(),
            ],
        )?
        .into_any());
    }
    let formula: Basic = from_python(&formula)?;
    Ok(match name {
        "quantifier_elimination" => to_python(
            py,
            &qe::quantifier_elimination(&formula, &prefix, free.as_deref(), method, partial).map_err(e)?,
        )?,
        "decide" => pyo3::types::PyBool::new(py, qe::decide(&formula, &prefix, method, partial).map_err(e)?)
            .to_owned()
            .into_any(),
        "solution_set" => {
            let x: Basic = free
                .and_then(|f| f.into_iter().next())
                .ok_or_else(|| PyTypeError::new_err("no variable"))?;
            to_python(
                py,
                &qe::solution_set(&formula, &x, &prefix, method, partial).map_err(e)?,
            )?
        }
        "sample_points" => {
            let points = qe::sample_points(&formula, &free.unwrap_or_default(), method, partial).map_err(e)?;
            let out = pyo3::types::PyList::empty(py);
            for point in points {
                let d = PyDict::new(py);
                for (v, c) in point {
                    d.set_item(to_python(py, &v)?, to_python(py, &c)?)?;
                }
                out.append(d)?;
            }
            out.into_any()
        }
        _ => return Err(PyTypeError::new_err(format!("unknown qe function {}", name))),
    })
}

/// The cylindrical algebraic decomposition of `polys`: its repr, the
/// method used, the projection sets and every top cell's repr and signs.
#[pyfunction]
#[pyo3(signature = (polys, gens, method=None))]
fn cad_report<'py>(
    py: Python<'py>,
    polys: Vec<Bound<'py, PyAny>>,
    gens: Vec<Bound<'py, PyAny>>,
    method: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use sympy_extras_core::polys::cad::{lifting, projection::Method};
    let e = crate::convert::err;
    let polys: Vec<Basic> = polys.iter().map(from_python).collect::<PyResult<_>>()?;
    let gens: Vec<Basic> = gens.iter().map(from_python).collect::<PyResult<_>>()?;
    let method: Option<Method> = method.map(Method::from_name).transpose().map_err(e)?;
    let cad = lifting::cylindrical_algebraic_decomposition(&polys, &gens, method).map_err(e)?;
    let out = PyDict::new(py);
    out.set_item("repr", cad.repr())?;
    out.set_item(
        "method",
        if cad.method == Method::McCallum {
            "mccallum"
        } else {
            "hong"
        },
    )?;
    let projection: Vec<Vec<String>> = cad
        .projection
        .iter()
        .map(|l| l.iter().map(|p| p.to_str().map_err(e)).collect())
        .collect::<PyResult<_>>()?;
    out.set_item("projection", projection)?;
    let mut cells: Vec<String> = Vec::new();
    for cell in cad.cells() {
        let signs: Vec<String> = cell
            .signs
            .lock()
            .expect("signs")
            .iter()
            .map(|s| s.to_string())
            .collect();
        cells.push(format!("{} {}", cell.repr().map_err(e)?, signs.join(",")));
    }
    out.set_item("cells", cells)?;
    let sizes: Vec<usize> = cad.levels().iter().map(|l| l.len()).collect();
    out.set_item("sizes", sizes)?;
    Ok(out)
}

/// `indexed_root(f, t, k)` of `polys.cad.indexedroot`.
#[pyfunction]
fn indexed_root<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    t: &Bound<'py, PyAny>,
    k: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let value = sympy_extras_core::polys::cad::indexedroot::indexed_root(&from_python(f)?, &from_python(t)?, k);
    to_python(py, &value.map_err(crate::convert::err)?)
}

/// The functions of `core.logic` and `core.relational` by name.
#[pyfunction]
fn logic_call<'py>(py: Python<'py>, name: &str, args: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    use sympy_extras_core::core::{logic as lg, relational as rl};
    use sympy_extras_core::sets::sets as st;
    let a: Vec<Basic> = args.iter().map(from_python).collect::<PyResult<_>>()?;
    let e = crate::convert::err;
    let value: Basic = match name {
        "and_" => lg::and_(&a),
        "or_" => lg::or_(&a),
        "not_" => lg::not_(&a[0]),
        "xor" => lg::xor(&a),
        "implies" => lg::implies(&a[0], &a[1]),
        "equivalent" => lg::equivalent(&a),
        "ite" => lg::ite(&a[0], &a[1], &a[2]),
        "Nand" => lg::nand(&a),
        "Nor" => lg::nor(&a),
        "Xnor" => lg::xnor(&a),
        "eq" => rl::eq(&a[0], &a[1]).map_err(e)?,
        "ne" => rl::ne(&a[0], &a[1]).map_err(e)?,
        "lt" => rl::lt(&a[0], &a[1]).map_err(e)?,
        "le" => rl::le(&a[0], &a[1]).map_err(e)?,
        "gt" => rl::gt(&a[0], &a[1]).map_err(e)?,
        "ge" => rl::ge(&a[0], &a[1]).map_err(e)?,
        "max_" => sympy_extras_core::functions::elementary::miscellaneous::max_(&a).map_err(e)?,
        "min_" => sympy_extras_core::functions::elementary::miscellaneous::min_(&a).map_err(e)?,
        "interval" => {
            let flag = |b: &Basic| sympy_extras_core::core::logic::truth(b) == Some(true);
            st::interval(&a[0], &a[1], a.get(2).is_some_and(flag), a.get(3).is_some_and(flag)).map_err(e)?
        }
        "finite_set" => st::finite_set(&a),
        "condition_set" => sympy_extras_core::sets::conditionset::condition_set(&a[0], &a[1], &a[2]).map_err(e)?,
        "union" => st::union(&a).map_err(e)?,
        "intersection" => st::intersection(&a).map_err(e)?,
        "complement" => st::complement(&a[0], &a[1]).map_err(e)?,
        "contains" => st::contains(&a[1], &a[0]).map_err(e)?,
        "as_relational" => st::as_relational(&a[0], &a[1]).map_err(e)?,
        "inf" => st::inf(&a[0]).map_err(e)?,
        "sup" => st::sup(&a[0]).map_err(e)?,
        "is_subset" | "is_empty" | "is_finite_set" => {
            let v: Option<bool> = match name {
                "is_subset" => st::is_subset(&a[0], &a[1]),
                "is_empty" => st::is_empty(&a[0]),
                _ => st::is_finite_set(&a[0]),
            }
            .map_err(e)?;
            return Ok(match v {
                Some(b) => pyo3::types::PyBool::new(py, b).to_owned().into_any(),
                None => py.None().into_bound(py),
            });
        }
        "accum_bounds" => sympy_extras_core::core::accumbounds::accum_bounds(&a[0], &a[1]).map_err(e)?,
        "ab_abs" => {
            if sympy_extras_core::core::accumbounds::is_bounds(&a[0]) {
                sympy_extras_core::core::accumbounds::ab_abs(&a[0]).map_err(e)?
            } else {
                sympy_extras_core::functions::elementary::complexes::abs_(&a[0]).map_err(e)?
            }
        }
        "ab_union" | "ab_intersection" => {
            use sympy_extras_core::core::accumbounds as ab;
            let found = if name == "ab_union" {
                ab::union(&a[0], &a[1])
            } else {
                ab::intersection(&a[0], &a[1])
            };
            match found.map_err(e)? {
                Some(v) => v,
                None => return Ok(py.None().into_bound(py)),
            }
        }
        "ab_contains" => {
            let v: bool = sympy_extras_core::core::accumbounds::contains(&a[0], &a[1]).map_err(e)?;
            return Ok(pyo3::types::PyBool::new(py, v).to_owned().into_any());
        }
        "mod" => sympy_extras_core::core::modulo::modulo(&a[0], &a[1]).map_err(e)?,
        "beta" => sympy_extras_core::functions::special::beta_functions::beta(&a[0], a.get(1)).map_err(e)?,
        "expint" | "Ei" | "Si" | "Ci" | "Shi" | "Chi" | "li" | "Li" | "fresnels" | "fresnelc" => {
            use sympy_extras_core::core::basic::Function as F;
            let f: F = match name {
                "expint" => F::Expint,
                "Ei" => F::Ei,
                "Si" => F::Si,
                "Ci" => F::Ci,
                "Shi" => F::Shi,
                "Chi" => F::Chi,
                "li" => F::LogIntegral,
                "Li" => F::OffsetLogIntegral,
                "fresnels" => F::FresnelS,
                _ => F::FresnelC,
            };
            sympy_extras_core::functions::special::error_functions::integral_function(f, &a).map_err(e)?
        }
        "erf" | "erfc" | "erfi" | "erfinv" | "erfcinv" | "erf2" | "erf2inv" => {
            use sympy_extras_core::core::basic::Function as F;
            let f: F = match name {
                "erf" => F::Erf,
                "erfc" => F::Erfc,
                "erfi" => F::Erfi,
                "erfinv" => F::Erfinv,
                "erfcinv" => F::Erfcinv,
                "erf2" => F::Erf2,
                _ => F::Erf2inv,
            };
            sympy_extras_core::functions::special::error_functions::error_function(f, &a).map_err(e)?
        }
        "betainc" | "betainc_regularized" => {
            use sympy_extras_core::core::basic::Function as F;
            let f: F = if name == "betainc" {
                F::BetaInc
            } else {
                F::BetaIncRegularized
            };
            sympy_extras_core::functions::special::beta_functions::betainc(f, &a).map_err(e)?
        }
        n if [
            "totient",
            "reduced_totient",
            "divisor_sigma",
            "udivisor_sigma",
            "primenu",
            "primeomega",
            "mobius",
            "primepi",
            "legendre_symbol",
            "jacobi_symbol",
            "kronecker_symbol",
            "tribonacci",
            "motzkin",
            "andre",
        ]
        .contains(&n) =>
        {
            use sympy_extras_core::core::basic::Function as F;
            let f: F = match n {
                "totient" => F::Totient,
                "reduced_totient" => F::ReducedTotient,
                "divisor_sigma" => F::DivisorSigma,
                "udivisor_sigma" => F::UDivisorSigma,
                "primenu" => F::PrimeNu,
                "primeomega" => F::PrimeOmega,
                "mobius" => F::Mobius,
                "primepi" => F::PrimePi,
                "legendre_symbol" => F::LegendreSymbol,
                "jacobi_symbol" => F::JacobiSymbol,
                "kronecker_symbol" => F::KroneckerSymbol,
                "tribonacci" => F::Tribonacci,
                "motzkin" => F::Motzkin,
                _ => F::Andre,
            };
            sympy_extras_core::functions::combinatorial::numbertheory::number_theoretic(f, &a).map_err(e)?
        }
        "catalan" | "fibonacci" | "lucas" | "harmonic" | "bernoulli" | "euler" | "bell" | "genocchi" | "partition" => {
            use sympy_extras_core::functions::combinatorial::numbers as nb;
            match name {
                "catalan" => nb::catalan(&a[0]),
                "fibonacci" => nb::fibonacci(&a[0], a.get(1)),
                "lucas" => nb::lucas(&a[0]),
                "harmonic" => nb::harmonic(&a[0], a.get(1)),
                "bernoulli" => nb::bernoulli(&a[0], a.get(1)),
                "euler" => nb::euler(&a[0], a.get(1)),
                "bell" => nb::bell(&a[0], a.get(1)),
                "genocchi" => nb::genocchi(&a[0], a.get(1)),
                _ => nb::partition(&a[0]),
            }
            .map_err(e)?
        }
        "KroneckerDelta" => {
            sympy_extras_core::functions::special::tensor_functions::kronecker_delta(&a[0], &a[1]).map_err(e)?
        }
        "LeviCivita" => sympy_extras_core::functions::special::tensor_functions::levi_civita(&a).map_err(e)?,
        "zeta" | "dirichlet_eta" | "polylog" | "lerchphi" | "riemann_xi" | "stieltjes" => {
            use sympy_extras_core::functions::special::zeta_functions as zf;
            match name {
                "zeta" => zf::zeta(&a[0], a.get(1)),
                "dirichlet_eta" => zf::dirichlet_eta(&a[0], a.get(1)),
                "polylog" => zf::polylog(&a[0], &a[1]),
                "lerchphi" => zf::lerchphi(&a[0], &a[1], &a[2]),
                "riemann_xi" => zf::riemann_xi(&a[0]),
                _ => zf::stieltjes(&a[0], a.get(1)),
            }
            .map_err(e)?
        }
        "bspline_basis" | "bspline_basis_set" => {
            use sympy_extras_core::functions::special::bsplines as bs;
            let small = |b: &Basic| -> PyResult<i64> {
                b.as_rational()
                    .filter(|v| sympy_extras_core::numbers::denom(v) == sympy_extras_core::numbers::Int::ONE)
                    .and_then(|v| i64::try_from(&sympy_extras_core::numbers::numer(v)).ok())
                    .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("an integer is expected"))
            };
            let d: i64 = small(&a[0])?;
            let knots: Vec<Basic> = a[1].args().to_vec();
            if name == "bspline_basis" {
                bs::bspline_basis(d, &knots, small(&a[2])?, &a[3]).map_err(e)?
            } else {
                Basic::new(Head::Tuple, bs::bspline_basis_set(d, &knots, &a[2]).map_err(e)?)
            }
        }
        "piecewise" => {
            sympy_extras_core::functions::elementary::piecewise::piecewise_of_pairs(a.to_vec()).map_err(e)?
        }
        "piecewise_fold" | "piecewise_fold_none" | "piecewise_fold_false" => {
            let evaluate: Option<bool> = match name {
                "piecewise_fold" => Some(true),
                "piecewise_fold_none" => None,
                _ => Some(false),
            };
            sympy_extras_core::functions::elementary::piecewise::piecewise_fold(&a[0], evaluate).map_err(e)?
        }
        "elliptic_k" | "elliptic_f" | "elliptic_e" | "elliptic_pi" => {
            use sympy_extras_core::core::basic::Function as F;
            use sympy_extras_core::functions::special::elliptic_integrals as ei;
            let f: F = match name {
                "elliptic_k" => F::EllipticK,
                "elliptic_f" => F::EllipticF,
                "elliptic_e" => F::EllipticE,
                _ => F::EllipticPi,
            };
            if !ei::arity_fits(f, a.len()) {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong number of arguments"));
            }
            ei::elliptic(f, &a).map_err(e)?
        }
        "hyper" | "meijerg" | "appellf1" | "hyper_radius" | "hyper_eta" | "hyper_convergence" | "meijerg_nu"
        | "meijerg_delta" | "meijerg_period" => {
            use sympy_extras_core::functions::special::hyper as hy;
            let wrong = || pyo3::exceptions::PyTypeError::new_err("wrong arguments");
            let tuple_of = |b: &Basic| -> PyResult<Vec<Basic>> {
                if matches!(b.head(), Head::Tuple) {
                    Ok(b.args().to_vec())
                } else {
                    Err(wrong())
                }
            };
            match (name, a.len()) {
                ("hyper", 3) => hy::hyper(&tuple_of(&a[0])?, &tuple_of(&a[1])?, &a[2]).map_err(e)?,
                ("meijerg", 5) => hy::meijerg(
                    &tuple_of(&a[0])?,
                    &tuple_of(&a[1])?,
                    &tuple_of(&a[2])?,
                    &tuple_of(&a[3])?,
                    &a[4],
                )
                .map_err(e)?,
                ("appellf1", 6) => hy::appellf1(&a).map_err(e)?,
                ("hyper_radius", 1) if matches!(a[0].head(), Head::Hyper) => {
                    hy::radius_of_convergence(&a[0]).map_err(e)?
                }
                ("hyper_eta", 1) if matches!(a[0].head(), Head::Hyper) => hy::hyper_eta(&a[0]).map_err(e)?,
                ("hyper_convergence", 1) if matches!(a[0].head(), Head::Hyper) => {
                    hy::convergence_statement(&a[0]).map_err(e)?
                }
                ("meijerg_nu", 1) if matches!(a[0].head(), Head::MeijerG) => hy::meijerg_nu(&a[0]).map_err(e)?,
                ("meijerg_delta", 1) if matches!(a[0].head(), Head::MeijerG) => hy::meijerg_delta(&a[0]).map_err(e)?,
                ("meijerg_period", 1) if matches!(a[0].head(), Head::MeijerG) => {
                    hy::meijerg_get_period(&a[0]).map_err(e)?
                }
                _ => return Err(wrong()),
            }
        }
        "xor_evaluated"
        | "and_evaluated"
        | "is_anf"
        | "to_anf"
        | "distribute_xor_over_and"
        | "anf_coeffs"
        | "bool_monomial"
        | "ANFform" => {
            use sympy_extras_core::logic::anf;
            let wrong = || pyo3::exceptions::PyTypeError::new_err("wrong arguments");
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let flag = |b: &Basic| sympy_extras_core::core::logic::truth(b) == Some(true);
            let digits = |b: &Basic| -> PyResult<Vec<u8>> {
                b.args()
                    .iter()
                    .map(|v| {
                        v.as_rational()
                            .and_then(|r| u8::try_from(r.numerator()).ok())
                            .ok_or_else(wrong)
                    })
                    .collect()
            };
            let small = |v: u8| Basic::integer((v as i64).into());
            match (name, a.len()) {
                ("xor_evaluated", n) if n >= 1 => anf::xor_evaluated(&a[1..], flag(&a[0])).map_err(e)?,
                ("and_evaluated", _) => anf::and_evaluated(&a).map_err(e)?,
                ("is_anf", 1) => sympy_extras_core::core::logic::boolean(anf::is_anf(&a[0])),
                ("to_anf", 2) => anf::to_anf(&a[0], flag(&a[1])).map_err(e)?,
                ("distribute_xor_over_and", 1) => anf::distribute_xor_over_and(&a[0]).map_err(e)?,
                ("anf_coeffs", 1) => tuple(
                    anf::anf_coeffs(&digits(&a[0])?)
                        .map_err(e)?
                        .into_iter()
                        .map(small)
                        .collect(),
                ),
                ("bool_monomial", 2) => {
                    if matches!(a[0].head(), Head::Tuple) {
                        anf::bool_monomial_digits(&digits(&a[0])?, a[1].args()).map_err(e)?
                    } else {
                        let k: i64 = a[0]
                            .as_rational()
                            .and_then(|v| i64::try_from(v.numerator()).ok())
                            .ok_or_else(wrong)?;
                        anf::bool_monomial(k, a[1].args()).map_err(e)?
                    }
                }
                ("ANFform", 2) => anf::anf_form(a[0].args(), &digits(&a[1])?).map_err(e)?,
                _ => return Err(wrong()),
            }
        }
        "multiquadratic" => {
            // [radicands Tuple, operation symbol, element, (element | index)];
            // an element over QQ: a Tuple of (mask, coefficient) pairs
            use sympy_extras_core::polys::domains::multiquadratic::{MultiquadraticElement, MultiquadraticField};
            use sympy_extras_core::polys::domains::{Domain, QQ};
            let wrong = || pyo3::exceptions::PyTypeError::new_err("wrong arguments");
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            if a.len() < 3 {
                return Err(wrong());
            }
            let small = |b: &Basic| -> PyResult<i64> {
                b.as_rational()
                    .and_then(|v| i64::try_from(v.numerator()).ok())
                    .ok_or_else(wrong)
            };
            let radicands: Vec<i64> = a[0].args().iter().map(small).collect::<PyResult<_>>()?;
            let field = MultiquadraticField::new(QQ, radicands).map_err(e)?;
            let element = |b: &Basic| -> PyResult<MultiquadraticElement<sympy_extras_core::numbers::Rat>> {
                let mut m = std::collections::BTreeMap::new();
                for pair in b.args() {
                    let mask: u64 = small(&pair.args()[0])? as u64;
                    let c = pair.args()[1].as_rational().cloned().ok_or_else(wrong)?;
                    m.insert(mask, c);
                }
                Ok(m)
            };
            let encode = |m: &MultiquadraticElement<sympy_extras_core::numbers::Rat>| -> Basic {
                tuple(
                    m.iter()
                        .map(|(k, c)| {
                            tuple(vec![
                                Basic::integer((*k as i64).into()),
                                Basic::from_rational(c.clone()),
                            ])
                        })
                        .collect(),
                )
            };
            let x = element(&a[2])?;
            let label = |t: String| sympy_extras_core::core::Symbol::new(&t, &[]).0;
            match (a[1].to_string().as_str(), a.len()) {
                ("to_str", 3) => label(field.to_str(&x)),
                ("name", 3) => label(field.name()),
                ("size", 3) => Basic::integer((field.size(&x) as i64).into()),
                ("invert", 3) => encode(&field.invert(&x).map_err(e)?),
                ("neg", 3) => encode(&field.neg(&x)),
                ("add", 4) => encode(&field.add(&x, &element(&a[3])?)),
                ("sub", 4) => encode(&field.sub(&x, &element(&a[3])?)),
                ("mul", 4) => encode(&field.mul(&x, &element(&a[3])?)),
                ("exquo", 4) => encode(&field.exquo(&x, &element(&a[3])?).map_err(e)?),
                ("conjugate", 4) => encode(&field.conjugate(&x, small(&a[3])? as usize)),
                _ => return Err(wrong()),
            }
        }
        "domain_matrix" => {
            // [domain symbol ZZ/QQ, operation symbol, matrix rows, (second matrix | power)]
            use sympy_extras_core::matrices::domainmatrix::DomainMatrix;
            use sympy_extras_core::polys::domains::{QQ, ZZ};
            let wrong = || pyo3::exceptions::PyTypeError::new_err("wrong arguments");
            if a.len() < 3 {
                return Err(wrong());
            }
            let op: String = a[1].to_string();
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let rats = |m: &Basic| -> PyResult<Vec<Vec<sympy_extras_core::numbers::Rat>>> {
                m.args()
                    .iter()
                    .map(|r| {
                        r.args()
                            .iter()
                            .map(|v| v.as_rational().cloned().ok_or_else(wrong))
                            .collect()
                    })
                    .collect()
            };
            macro_rules! run {
                ($k:expr, $to:expr, $from:expr) => {{
                    let k = $k;
                    let to_elem = $to;
                    let from_elem = $from;
                    let build = |m: &Basic| -> PyResult<DomainMatrix<_>> {
                        let rows = rats(m)?
                            .into_iter()
                            .map(|r| r.iter().map(&to_elem).collect::<PyResult<Vec<_>>>())
                            .collect::<PyResult<Vec<_>>>()?;
                        let width: Option<usize> = if rows.is_empty() { Some(0) } else { None };
                        DomainMatrix::new(rows, k.clone(), width).map_err(e)
                    };
                    let encode = |m: &DomainMatrix<_>| -> Basic {
                        tuple(
                            m.rows
                                .iter()
                                .map(|r| tuple(r.iter().map(&from_elem).collect()))
                                .collect(),
                        )
                    };
                    let ints = |v: &[usize]| tuple(v.iter().map(|i| Basic::integer((*i as i64).into())).collect());
                    let m = build(&a[2])?;
                    match op.as_str() {
                        "det" => from_elem(&m.det().map_err(e)?),
                        "rank" => Basic::integer((m.rank().map_err(e)? as i64).into()),
                        "rref_den" => {
                            let (r, d, p) = m.rref_den().map_err(e)?;
                            tuple(vec![encode(&r), from_elem(&d), ints(&p)])
                        }
                        "rref" => {
                            let (r, p) = m.rref().map_err(e)?;
                            tuple(vec![encode(&r), ints(&p)])
                        }
                        "nullspace" => tuple(
                            m.nullspace()
                                .map_err(e)?
                                .iter()
                                .map(|v| tuple(v.iter().map(&from_elem).collect()))
                                .collect(),
                        ),
                        "inv_den" => {
                            let (n, d) = m.inv_den().map_err(e)?;
                            tuple(vec![encode(&n), from_elem(&d)])
                        }
                        "inv" => encode(&m.inv().map_err(e)?),
                        "charpoly" => tuple(m.charpoly().map_err(e)?.iter().map(&from_elem).collect()),
                        "adjugate" => encode(&m.adjugate().map_err(e)?),
                        "transpose" => encode(&m.transpose()),
                        "repr" => sympy_extras_core::core::Symbol::new(&m.to_str(), &[]).0,
                        "solve_den" if a.len() == 4 => match m.solve_den(&build(&a[3])?).map_err(e)? {
                            None => tuple(Vec::new()),
                            Some((x, d)) => tuple(vec![encode(&x), from_elem(&d)]),
                        },
                        "mul" if a.len() == 4 => encode(&m.mul(&build(&a[3])?).map_err(e)?),
                        "add" if a.len() == 4 => encode(&m.add(&build(&a[3])?).map_err(e)?),
                        "pow" if a.len() == 4 => {
                            let n: i64 = a[3]
                                .as_rational()
                                .and_then(|v| i64::try_from(v.numerator()).ok())
                                .ok_or_else(wrong)?;
                            encode(&m.pow(n).map_err(e)?)
                        }
                        _ => return Err(wrong()),
                    }
                }};
            }
            match a[0].to_string().as_str() {
                "ZZ" => run!(
                    ZZ,
                    |v: &sympy_extras_core::numbers::Rat| -> PyResult<sympy_extras_core::numbers::Int> {
                        if v.is_int() {
                            Ok(v.numerator().clone())
                        } else {
                            Err(wrong())
                        }
                    },
                    |v: &sympy_extras_core::numbers::Int| Basic::integer(v.clone())
                ),
                "QQ" => run!(
                    QQ,
                    |v: &sympy_extras_core::numbers::Rat| -> PyResult<sympy_extras_core::numbers::Rat> {
                        Ok(v.clone())
                    },
                    |v: &sympy_extras_core::numbers::Rat| Basic::from_rational(v.clone())
                ),
                _ => return Err(wrong()),
            }
        }
        "polynomial_abstraction" => {
            use sympy_extras_core::assumptions::bounds as bd;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            if a.len() != 3 || !matches!(a[1].head(), Head::Tuple) {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments"));
            }
            let max: usize = a[2]
                .as_rational()
                .and_then(|v| usize::try_from(v.numerator()).ok())
                .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("max_variables"))?;
            match bd::polynomial_abstraction(&a[0], a[1].args(), max).map_err(e)? {
                None => tuple(Vec::new()),
                // the variables first: the bridge creates the Python dummies in
                // the order it meets them, which keeps their relative order
                Some(ab) => tuple(vec![
                    tuple(ab.variables),
                    ab.formula,
                    tuple(ab.constraints),
                    tuple(ab.requirements),
                ]),
            }
        }
        "igcd" | "ilcm" | "igcdex" | "mod_inverse" | "integer_nthroot" | "integer_log" | "num_digits" | "trailing" => {
            use sympy_extras_core::core::intfunc as inf;
            let mut ints: Vec<sympy_extras_core::numbers::Int> = Vec::new();
            for x in &a {
                match x.as_rational() {
                    Some(v) if v.is_int() => ints.push(v.numerator().clone()),
                    _ => return Err(pyo3::exceptions::PyTypeError::new_err("integers are expected")),
                }
            }
            let num = |v: sympy_extras_core::numbers::Int| Basic::integer(v);
            let boolean = sympy_extras_core::core::logic::boolean;
            match name {
                "igcd" => num(inf::igcd(&ints)),
                "ilcm" => num(inf::ilcm(&ints)),
                "igcdex" => {
                    let (x, y, g) = inf::igcdex(&ints[0], &ints[1]);
                    Basic::new(Head::Tuple, vec![num(x), num(y), num(g)])
                }
                "mod_inverse" => num(inf::mod_inverse(&ints[0], &ints[1]).map_err(e)?),
                "integer_nthroot" | "integer_log" => {
                    let (r, exact) = if name == "integer_nthroot" {
                        inf::integer_nthroot(&ints[0], &ints[1]).map_err(e)?
                    } else {
                        inf::integer_log(&ints[0], &ints[1]).map_err(e)?
                    };
                    Basic::new(Head::Tuple, vec![num(r), boolean(exact)])
                }
                "num_digits" => num(inf::num_digits(&ints[0], &ints[1])),
                _ => num(inf::trailing(&ints[0])),
            }
        }
        "match" => {
            // expr, pattern, exact (0 or 1), the bindings as Tuple(wild, value); the
            // bindings found as Tuple(Tuple(wild, value), ...) or false
            if a.len() < 3 {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments"));
            }
            let start: Vec<(Basic, Basic)> = a[3..]
                .iter()
                .map(|t| (t.args()[0].clone(), t.args()[1].clone()))
                .collect();
            match sympy_extras_core::core::matching::match_pattern(&a[0], &a[1], &start, a[2] != Basic::zero())
                .map_err(e)?
            {
                Some(found) => Basic::new(
                    Head::Tuple,
                    found
                        .into_iter()
                        .map(|(w, v)| Basic::new(Head::Tuple, vec![w, v]))
                        .collect(),
                ),
                None => sympy_extras_core::core::logic::false_(),
            }
        }
        "is_meromorphic" => {
            // True, False or Integer(0) for None
            match sympy_extras_core::core::meromorphic::is_meromorphic(&a[0], &a[1], &a[2]).map_err(e)? {
                Some(v) => sympy_extras_core::core::logic::boolean(v),
                None => Basic::zero(),
            }
        }
        "facts" => {
            // op (facts, normalize, to_predicates, to_polynomial, element), arguments ->
            // facts: Tuple(assumptions), domain or Tuple(), Tuple(symbols) -> Tuple(formula,
            //   Tuple(conjuncts), predicates, Tuple(real), Tuple(integer), polynomial);
            // normalize, to_predicates: formula; to_polynomial: formula, Tuple(real);
            // element: x, domain. None as Tuple()
            use sympy_extras_core::assumptions::facts as fc;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let none = || tuple(Vec::new());
            match a[0].to_string().as_str() {
                "facts" => {
                    let domain: Option<&Basic> = if matches!(a[2].head(), Head::Tuple) {
                        None
                    } else {
                        Some(&a[2])
                    };
                    let f = fc::Facts::new(a[1].args(), domain, a[3].args()).map_err(e)?;
                    tuple(vec![
                        f.formula.clone(),
                        tuple(f.conjuncts.clone()),
                        f.predicates.clone(),
                        tuple(f.real.clone()),
                        tuple(f.integer.clone()),
                        f.polynomial.clone(),
                    ])
                }
                "normalize" => fc::normalize(&a[1]).map_err(e)?,
                "to_predicates" => fc::to_predicates(&a[1]).map_err(e)?.unwrap_or_else(none),
                "to_polynomial" => fc::to_polynomial(&a[1], a[2].args()).map_err(e)?.unwrap_or_else(none),
                "element" => fc::element(&a[1], &a[2]).map_err(e)?,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "solve" => {
            // op: equation(f, x), lambert(f, x) -> Tuple of solutions, or the symbol NONE for None;
            // invert(f, x) -> Tuple(r, g); system(Tuple(equations), Tuple(unknowns)) -> Tuple of
            // solutions, each a Tuple of Tuple(unknown, value), or NONE
            use sympy_extras_core::solvers::{bivariate, solvers as sv};
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let none = || sympy_extras_core::core::symbol::Symbol::new("NONE", &[]).0;
            match a[0].to_string().as_str() {
                "equation" => sv::solve_equation(&a[1], &a[2])
                    .map_err(e)?
                    .map(tuple)
                    .unwrap_or_else(none),
                "lambert" => bivariate::solve_lambert(&a[1], &a[2])
                    .map_err(e)?
                    .map(tuple)
                    .unwrap_or_else(none),
                "invert" => {
                    let (r, g) = sv::invert(&a[1], &a[2], &Basic::zero()).map_err(e)?;
                    tuple(vec![r, g])
                }
                "system" => match sv::solve_system(a[1].args(), a[2].args()).map_err(e)? {
                    None => none(),
                    Some(solutions) => tuple(
                        solutions
                            .into_iter()
                            .map(|s| tuple(s.into_iter().map(|(k, v)| tuple(vec![k, v])).collect()))
                            .collect(),
                    ),
                },
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "lie" => {
            // op: symmetries(eqs, funcs, degree, basis, superposition) -> Tuple of Tuple(Tuple(xi),
            // Tuple(eta), Symbol(generator)); check(eqs, funcs, Tuple(xi + eta)) -> bool;
            // condition(eqs, funcs, Tuple(xi + eta)) -> Tuple; system(eqs, funcs) -> Tuple(Tuple(eqs),
            // Tuple(unknowns), Tuple(base)); janet(eqs, funcs) -> dimension; leading(eqs, funcs) ->
            // Tuple of Tuple(jet, value); characteristic/commutator/prolong on symmetries
            use sympy_extras_core::solvers::lie;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let small = |b: &Basic| -> PyResult<u32> {
                b.to_string()
                    .parse::<u32>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("a small integer"))
            };
            let funcs = a[2].args();
            match a[0].to_string().as_str() {
                "symmetries" => tuple(
                    lie::symmetries(
                        a[1].args(),
                        funcs,
                        small(&a[3])?,
                        a[4].args(),
                        a[5].to_string() == "True",
                    )
                    .map_err(e)?
                    .into_iter()
                    .map(|s| {
                        let g = sympy_extras_core::core::symbol::Symbol::new(&s.generator(), &[]).0;
                        tuple(vec![tuple(s.xi), tuple(s.eta), g])
                    })
                    .collect(),
                ),
                "check" => sympy_extras_core::core::logic::boolean(
                    lie::check_symmetry(a[1].args(), funcs, a[3].args()).map_err(e)?,
                ),
                "condition" => tuple(lie::infinitesimal_condition(a[1].args(), funcs, a[3].args()).map_err(e)?),
                "system" => {
                    let (sys, unknowns, base) = lie::determining_system(a[1].args(), funcs).map_err(e)?;
                    tuple(vec![tuple(sys), tuple(unknowns), tuple(base)])
                }
                "janet" => lie::symmetry_janet_basis(a[1].args(), funcs)
                    .map_err(e)?
                    .dimension()
                    .map_err(e)?,
                "leading" => {
                    let jets = lie::JetSpace::new(funcs).map_err(e)?;
                    let eqs: Vec<Basic> = a[1]
                        .args()
                        .iter()
                        .map(|q| jets.to_jet(q))
                        .collect::<Result<_, _>>()
                        .map_err(e)?;
                    tuple(
                        lie::leading_derivatives(&jets, &eqs)
                            .map_err(e)?
                            .into_iter()
                            .map(|(k, v)| tuple(vec![k, v]))
                            .collect(),
                    )
                }
                "symmetry" => {
                    // symmetry(_, funcs, Tuple(xi + eta), what, [other / order / expr])
                    let jets = lie::JetSpace::new(funcs).map_err(e)?;
                    let p = jets.p();
                    let s =
                        lie::Symmetry::new(&jets, a[3].args()[..p].to_vec(), a[3].args()[p..].to_vec()).map_err(e)?;
                    match a[4].to_string().as_str() {
                        "characteristic" => tuple(s.characteristic().map_err(e)?),
                        "commutator" => {
                            let o = lie::Symmetry::new(&jets, a[5].args()[..p].to_vec(), a[5].args()[p..].to_vec())
                                .map_err(e)?;
                            let c = s.commutator(&o).map_err(e)?;
                            tuple(vec![tuple(c.xi), tuple(c.eta)])
                        }
                        "apply" => s.apply(&jets.to_jet(&a[5]).map_err(e)?).map_err(e)?,
                        "generator" => sympy_extras_core::core::symbol::Symbol::new(&s.generator(), &[]).0,
                        other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
                    }
                }
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "series" => {
            // op: order(e, Tuple(Tuple(x, p), ...)); contains(o, e) -> bool or the empty Tuple;
            // leading(e, Tuple(symbols)); leadterm(e, x) -> Tuple(c, e); nseries(e, x, n);
            // series(e, x, x0, n, Symbol(dir)); getn(e) -> n or the empty Tuple;
            // extract(e, Tuple(symbols)) -> Tuple of Tuple(term, order); subs(e, old, new);
            // add(Tuple(terms)), mul(Tuple(factors)), pow(b, e) evaluated; dir(e, x, cdir)
            use sympy_extras_core::core::{order as od, series as sr};
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let small = |b: &Basic| -> PyResult<i64> {
                b.to_string()
                    .parse::<i64>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("a small integer"))
            };
            match a[0].to_string().as_str() {
                "order" => {
                    let limits: Vec<(Basic, Basic)> = a[2]
                        .args()
                        .iter()
                        .map(|t| (t.args()[0].clone(), t.args()[1].clone()))
                        .collect();
                    od::order(&a[1], &limits).map_err(e)?
                }
                "contains" => match od::contains(&a[1], &a[2]).map_err(e)? {
                    Some(v) => sympy_extras_core::core::logic::boolean(v),
                    None => tuple(vec![]),
                },
                "leading" => sr::as_leading_term(&a[1], a[2].args(), None, &Basic::zero()).map_err(e)?,
                "leadterm" => {
                    let (c, ex) = sr::leadterm(&a[1], &a[2], None, &Basic::zero()).map_err(e)?;
                    tuple(vec![c, ex])
                }
                "nseries" => sr::nseries(&a[1], &a[2], small(&a[3])?, None, &Basic::zero()).map_err(e)?,
                "series" => sr::series(
                    &a[1],
                    Some(&a[2]),
                    &a[3],
                    small(&a[4])?,
                    &a[5].to_string(),
                    None,
                    &Basic::zero(),
                )
                .map_err(e)?,
                "getn" => od::getn(&a[1]).map_err(e)?.unwrap_or_else(|| tuple(vec![])),
                "extract" => tuple(
                    od::extract_leading_order(&a[1], a[2].args(), None)
                        .map_err(e)?
                        .into_iter()
                        .map(|(t, o)| tuple(vec![t, o]))
                        .collect(),
                ),
                "subs" => sympy_extras_core::core::subs::subs(&a[1], &a[2], &a[3]).map_err(e)?,
                "add" => sympy_extras_core::core::add::add(a[1].args()).map_err(e)?,
                "mul" => sympy_extras_core::core::mul::mul(a[1].args()).map_err(e)?,
                "pow" => sympy_extras_core::core::power::pow(&a[1], &a[2]).map_err(e)?,
                "dir" => sr::dir(&a[1], &a[2], &a[3]).map_err(e)?,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "is" => {
            // e, Symbol(predicate name) -> true, false or the empty Tuple for None (`e.is_<name>`)
            let Some(p) = sympy_extras_core::core::assumptions::Predicate::from_name(&a[1].to_string()) else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown predicate {}",
                    a[1]
                )));
            };
            match sympy_extras_core::core::expr::is(&a[0], p) {
                Some(v) => sympy_extras_core::core::logic::boolean(v),
                None => Basic::new(Head::Tuple, vec![]),
            }
        }
        "ratint" => {
            // op: ratint(f, x, symbol, real), ratint_pair(p, q, x, symbol, real) (real: true, false or
            // the symbol None), ratpart(f, g, x), polynomial_part(f, g, x), logpart(f, g, x, t),
            // log_to_atan(f, g, x), log_to_real(h, q, x, t, complex_only) -> empty Tuple for None
            use sympy_extras_core::integrals::risch::rationaltools as rt;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let real = |b: &Basic| match b.head() {
                Head::BooleanTrue => Some(true),
                Head::BooleanFalse => Some(false),
                _ => None,
            };
            match a[0].to_string().as_str() {
                "ratint" => rt::ratint(&rt::Integrand::of(&a[1]).map_err(e)?, &a[2], &a[3], real(&a[4])).map_err(e)?,
                "ratint_pair" => {
                    rt::ratint(&rt::Integrand::pair(&a[1], &a[2]), &a[3], &a[4], real(&a[5])).map_err(e)?
                }
                "ratpart" => {
                    let (x, y) = rt::ratint_ratpart(&a[1], &a[2], &a[3]).map_err(e)?;
                    tuple(vec![x, y])
                }
                "polynomial_part" => {
                    let (x, y) = rt::polynomial_part(&a[1], &a[2], &a[3]).map_err(e)?;
                    tuple(vec![x, y])
                }
                "logpart" => {
                    let pairs = rt::ratint_logpart(&a[1], &a[2], &a[3], Some(&a[4])).map_err(e)?;
                    tuple(pairs.into_iter().map(|(x, y)| tuple(vec![x, y])).collect())
                }
                "log_to_atan" => rt::log_to_atan(&a[1], &a[2], &a[3]).map_err(e)?,
                "log_to_real" => rt::log_to_real(&a[1], &a[2], &a[3], &a[4], matches!(a[5].head(), Head::BooleanTrue))
                    .map_err(e)?
                    .unwrap_or_else(|| tuple(vec![])),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "root_sum" => {
            // op: root_sum(f, var, body (or empty Tuples for the identity), t (or empty Tuple), auto,
            // quadratic), doit(node), diff(node, s), value(node, digits) -> Tuple(re, im) as rationals
            use sympy_extras_core::polys::rootoftools as rs;
            let given = |b: &Basic| !matches!(b.head(), Head::Tuple);
            let flag = |b: Option<&Basic>, d: bool| match b.map(|x| x.head()) {
                Some(Head::BooleanTrue) => true,
                Some(Head::BooleanFalse) => false,
                _ => d,
            };
            match a[0].to_string().as_str() {
                "root_sum" => {
                    let fun = if given(&a[2]) { Some((&a[2], &a[3])) } else { None };
                    let t = if given(&a[4]) { Some(&a[4]) } else { None };
                    rs::root_sum(&a[1], fun, t, flag(a.get(5), true), flag(a.get(6), false)).map_err(e)?
                }
                "doit" => sympy_extras_core::core::function::doit(&a[1]).map_err(e)?,
                "diff" => sympy_extras_core::core::function::derivative(&a[1], &a[2]).map_err(e)?,
                "is_number" => {
                    sympy_extras_core::core::logic::boolean(sympy_extras_core::core::assumptions::is_number(&a[1]))
                }
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "periodicity" => {
            // op: periodicity(f, symbol, check) -> period or empty Tuple; lcim(*numbers) -> lcm or empty Tuple
            use sympy_extras_core::calculus::util as cu;
            let none = || Basic::new(Head::Tuple, vec![]);
            match a[0].to_string().as_str() {
                "periodicity" => cu::periodicity(
                    &a[1],
                    &a[2],
                    matches!(a.get(3).map(|b| b.head()), Some(Head::BooleanTrue)),
                )
                .map_err(e)?
                .unwrap_or_else(none),
                "lcim" => cu::lcim(&a[1..]).map_err(e)?.unwrap_or_else(none),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "simplify" => {
            // op (simplify, collect_abs, inversecombine, kroneckersimp), e; simplify: ratio (a rational or
            // the symbol oo), inverse, doit
            use sympy_extras_core::simplify::simplify as sm;
            match a[0].to_string().as_str() {
                "simplify" => {
                    let ratio: f64 = match a.get(2) {
                        Some(r) if matches!(r.head(), Head::Infinity) => f64::INFINITY,
                        Some(r) => r
                            .as_rational()
                            .map(|v| sympy_extras_core::core::evalf::real_rat(v, 64).to_f64().value())
                            .unwrap_or(1.7),
                        None => 1.7,
                    };
                    let flag = |i: usize, default: bool| match a.get(i).map(|b| b.head()) {
                        Some(Head::BooleanTrue) => true,
                        Some(Head::BooleanFalse) => false,
                        _ => default,
                    };
                    let o = sm::Options {
                        ratio,
                        inverse: flag(3, false),
                        doit: flag(4, true),
                        ..sm::Options::default()
                    };
                    sm::simplify(&a[1], &o).map_err(e)?
                }
                "collect_abs" => sm::collect_abs(&a[1]).map_err(e)?,
                "inversecombine" => sm::inversecombine(&a[1]).map_err(e)?,
                "kroneckersimp" => sm::kroneckersimp(&a[1]).map_err(e)?,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "settings" => {
            // no argument: the current settings; with the changes (numerical_checks, precision, timeout,
            // no_timeout, groebner_algorithm, time_scale; an empty Tuple for None): the settings inside
            // configure and after it
            use sympy_extras_core::settings as st;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let shown = |s: &st::Settings| {
                tuple(vec![
                    sympy_extras_core::core::logic::boolean(s.numerical_checks),
                    Basic::int(s.precision as i64),
                    match s.timeout {
                        Some(t) => sympy_extras_core::core::symbol::Symbol::new(&format!("{}", t), &[]).0,
                        None => tuple(vec![]),
                    },
                    sympy_extras_core::core::symbol::Symbol::new(&format!("{}", s.time_scale), &[]).0,
                    sympy_extras_core::core::logic::boolean(s.modular_groebner),
                    sympy_extras_core::core::symbol::Symbol::new(&format!("{}", s.groebner_direct_time), &[]).0,
                    sympy_extras_core::core::symbol::Symbol::new(s.groebner_algorithm.name(), &[]).0,
                ])
            };
            if a.is_empty() {
                shown(&st::settings())
            } else {
                let given = |i: usize| -> Option<&Basic> {
                    if matches!(a[i].head(), Head::Tuple) {
                        None
                    } else {
                        Some(&a[i])
                    }
                };
                let float = |b: &Basic| b.to_string().parse::<f64>().unwrap_or(f64::NAN);
                let changes = st::Changes {
                    numerical_checks: given(0).map(|b| matches!(b.head(), Head::BooleanTrue)),
                    precision: given(1).map(|b| b.to_string().parse::<usize>().unwrap_or(30)),
                    timeout: given(2).map(float),
                    no_timeout: matches!(a[3].head(), Head::BooleanTrue),
                    groebner_algorithm: given(4).map(|b| b.to_string()),
                    time_scale: given(5).map(float),
                    ..st::Changes::default()
                };
                let inside = match st::configure(&changes) {
                    Ok(_guard) => shown(&st::settings()),
                    Err(err) => tuple(vec![
                        sympy_extras_core::core::symbol::Symbol::new(&format!("ERROR {}", err), &[]).0,
                    ]),
                };
                tuple(vec![inside, shown(&st::settings())])
            }
        }
        "trigsimp" => {
            // op: futrig(e, hyper), trigsimp(e, method name), exptrigsimp(e)
            use sympy_extras_core::simplify::trigsimp as t;
            match a[0].to_string().as_str() {
                "futrig" => {
                    t::futrig(&a[1], !matches!(a.get(2).map(|b| b.head()), Some(Head::BooleanFalse))).map_err(e)?
                }
                "trigsimp" => {
                    let method = match a[2].to_string().as_str() {
                        "matching" => t::Method::Matching,
                        "fu" => t::Method::Fu,
                        "groebner" => t::Method::Groebner,
                        "combined" => t::Method::Combined,
                        _ => t::Method::Old,
                    };
                    t::trigsimp(&a[1], method).map_err(e)?
                }
                "exptrigsimp" => t::exptrigsimp(&a[1]).map_err(e)?,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "fu" => {
            // op (a TR rule, fu, L, sincos_to_sum, trig_split, as_f_sign_1, hyper_as_trig) and its arguments
            use sympy_extras_core::simplify::fu as f;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let none = || tuple(vec![]);
            let x = &a[1];
            match a[0].to_string().as_str() {
                "TR0" => f::tr0(x).map_err(e)?,
                "TR1" => f::tr1(x).map_err(e)?,
                "TR2" => f::tr2(x).map_err(e)?,
                "TR2i" => f::tr2i(x, false).map_err(e)?,
                "TR2i_half" => f::tr2i(x, true).map_err(e)?,
                "TR3" => f::tr3(x).map_err(e)?,
                "TR4" => f::tr4(x).map_err(e)?,
                "TR5" => f::tr5(x, 4, false).map_err(e)?,
                "TR6" => f::tr6(x, 4, false).map_err(e)?,
                "TR6_pow" => f::tr6(x, 6, true).map_err(e)?,
                "TR7" => f::tr7(x).map_err(e)?,
                "TR8" => f::tr8(x, true).map_err(e)?,
                "TR9" => f::tr9(x).map_err(e)?,
                "TR10" => f::tr10(x, true).map_err(e)?,
                "TR10i" => f::tr10i(x).map_err(e)?,
                "TR11" => f::tr11(x, None).map_err(e)?,
                "TR11_base" => f::tr11(x, Some(&a[2])).map_err(e)?,
                "_TR11" => f::tr11_half(x).map_err(e)?,
                "TR12" => f::tr12(x, true).map_err(e)?,
                "TR12i" => f::tr12i(x).map_err(e)?,
                "TR13" => f::tr13(x).map_err(e)?,
                "TR14" => f::tr14(x, true).map_err(e)?,
                "TR15" => f::tr15(x, 4, false).map_err(e)?,
                "TR16" => f::tr16(x, 4, false).map_err(e)?,
                "TR111" => f::tr111(x).map_err(e)?,
                "TR22" => f::tr22(x, 4, false).map_err(e)?,
                "TRmorrie" => f::tr_morrie(x).map_err(e)?,
                "TRpower" => f::tr_power(x).map_err(e)?,
                "L" => Basic::int(f::l_count(x)),
                "sincos_to_sum" => f::sincos_to_sum(x).map_err(e)?,
                "fu" => f::fu(x, f::fu_measure).map_err(e)?,
                "trig_split" => match f::trig_split(x, &a[2], matches!(a[3].head(), Head::BooleanTrue)).map_err(e)? {
                    None => none(),
                    Some((g, n1, n2, p, q, c)) => tuple(vec![
                        g,
                        Basic::int(n1),
                        Basic::int(n2),
                        p,
                        q,
                        sympy_extras_core::core::logic::boolean(c),
                    ]),
                },
                "as_f_sign_1" => match f::as_f_sign_1(x).map_err(e)? {
                    None => none(),
                    Some((g, p, s)) => tuple(vec![g, p, s]),
                },
                "hyper_as_trig" => {
                    let h = f::hyper_as_trig(x).map_err(e)?;
                    let back = h.inverse(&h.expr).map_err(e)?;
                    tuple(vec![h.expr.clone(), back])
                }
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "collect" => {
            // op (collect, collect_dict, rcollect), e, Tuple(syms), exact (true, false or the symbol None)
            use sympy_extras_core::simplify::radsimp as rs;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let exact = match a.get(3).map(|b| b.head()) {
                Some(Head::BooleanTrue) => Some(true),
                Some(Head::BooleanFalse) => Some(false),
                _ => None,
            };
            match a[0].to_string().as_str() {
                "collect" => rs::collect(&a[1], a[2].args(), None, exact).map_err(e)?,
                "collect_dict" => {
                    let d = rs::collect_dict(&a[1], a[2].args(), None, exact).map_err(e)?;
                    tuple(d.into_iter().map(|(k, v)| tuple(vec![k, v])).collect())
                }
                "rcollect" => rs::rcollect(&a[1], a[2].args()).map_err(e)?,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "rc_solutions" => {
            // Tuple(polys), Tuple(variables), real -> Tuple of points, a point a Tuple of Tuple(v, value)
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let points = sympy_extras_core::polys::regularchains::solutions::exact_solutions(
                a[0].args(),
                a[1].args(),
                matches!(a[2].head(), Head::BooleanTrue),
                None,
            )
            .map_err(e)?;
            tuple(
                points
                    .into_iter()
                    .map(|p| tuple(p.into_iter().map(|(k, v)| tuple(vec![k, v])).collect()))
                    .collect(),
            )
        }
        "ex_domain" => {
            // op, rationalized flag, arguments: add, sub, mul, neg, exquo, invert, pow(a, n), lcm,
            // gcdex -> Tuple, is_zero, is_negative, normal, canonical_unit
            use sympy_extras_core::polys::domains::{Domain, ExpressionDomain};
            let k = ExpressionDomain {
                rationalized: matches!(a[1].head(), Head::BooleanTrue),
            };
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let b = |v: bool| sympy_extras_core::core::logic::boolean(v);
            match a[0].to_string().as_str() {
                "add" => k.add(&a[2], &a[3]),
                "sub" => k.sub(&a[2], &a[3]),
                "mul" => k.mul(&a[2], &a[3]),
                "neg" => k.neg(&a[2]),
                "exquo" => k.exquo(&a[2], &a[3]).map_err(e)?,
                "invert" => k.invert(&a[2]).map_err(e)?,
                "pow" => k.pow(
                    &a[2],
                    a[3].as_rational()
                        .and_then(|v| u64::try_from(v.numerator()).ok())
                        .unwrap_or(0),
                ),
                "lcm" => k.lcm(&a[2], &a[3]),
                "gcdex" => {
                    let (s1, t1, g) = k.gcdex(&a[2], &a[3]).map_err(e)?;
                    tuple(vec![s1, t1, g])
                }
                "is_zero" => b(k.is_zero(&a[2])),
                "is_negative" => b(k.is_negative(&a[2])),
                "normal" => k.normal(&a[2]).map_err(e)?,
                "canonical_unit" => k.canonical_unit(&a[2]),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "radsimp" => {
            // op: radsimp(e, symbolic, max_terms), sqrtdenest(e, max_iter), split_surds(e),
            // rad_rationalize(num, den), unevaluated_mul(*args), sqrt_depth(e), is_algebraic(e)
            use sympy_extras_core::simplify::{radsimp as rs, sqrtdenest as sd};
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let flag = |b: &Basic| matches!(b.head(), Head::BooleanTrue);
            let small = |b: &Basic| {
                b.as_rational()
                    .and_then(|v| usize::try_from(v.numerator()).ok())
                    .unwrap_or(4)
            };
            match a[0].to_string().as_str() {
                "radsimp" => rs::radsimp(&a[1], flag(&a[2]), small(&a[3])).map_err(e)?,
                "sqrtdenest" => sd::sqrtdenest(&a[1], small(&a[2])).map_err(e)?,
                "split_surds" => {
                    let (g, x, y) = rs::split_surds(&a[1]).map_err(e)?;
                    tuple(vec![g, x, y])
                }
                "rad_rationalize" => {
                    let (n, d) = rs::rad_rationalize(&a[1], &a[2]).map_err(e)?;
                    tuple(vec![n, d])
                }
                "unevaluated_mul" => rs::unevaluated_mul(&a[1..]).map_err(e)?,
                "sqrt_depth" => Basic::int(sd::sqrt_depth(&a[1]) as i64),
                "is_algebraic" => sympy_extras_core::core::logic::boolean(sd::is_algebraic(&a[1])),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "powsimp" => {
            // op: powsimp(e, deep, combine, force), powdenest(e, force, polar), logcombine(e, force),
            // posify(e) -> Tuple(e', Tuple(Tuple(dummy, symbol), ...)), multiplicity(p, n)
            use sympy_extras_core::simplify::powsimp as ps;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let flag = |b: &Basic| matches!(b.head(), Head::BooleanTrue);
            match a[0].to_string().as_str() {
                "powsimp" => {
                    let combine = match a[3].to_string().as_str() {
                        "exp" => ps::Combine::Exp,
                        "base" => ps::Combine::Base,
                        _ => ps::Combine::All,
                    };
                    let count = |x: &Basic| sympy_extras_core::core::function::count_ops(x);
                    ps::powsimp(&a[1], flag(&a[2]), combine, flag(&a[4]), &count).map_err(e)?
                }
                "powdenest" => ps::powdenest(&a[1], flag(&a[2]), flag(&a[3])).map_err(e)?,
                "logcombine" => sympy_extras_core::simplify::simplify::logcombine(&a[1], flag(&a[2])).map_err(e)?,
                "posify" => {
                    let (replaced, back) = sympy_extras_core::simplify::simplify::posify(&a[1]).map_err(e)?;
                    let mut pairs: Vec<Basic> = back.into_iter().map(|(d, s)| tuple(vec![d, s])).collect();
                    pairs.sort_by_key(|p| p.args()[1].to_string());
                    tuple(vec![replaced, tuple(pairs)])
                }
                "multiplicity" => {
                    let (Some(p), Some(n)) = (a[1].as_rational(), a[2].as_rational()) else {
                        return Err(pyo3::exceptions::PyTypeError::new_err("rationals expected"));
                    };
                    Basic::int(ps::multiplicity(p, n).map_err(e)?)
                }
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "int_linear" => {
            // op (hnf, diophantine, hilbert, minimal, cooper, presburger, is_presburger) and its
            // arguments; matrices as tuples of rows; None as an empty Tuple
            use sympy_extras_core::assumptions::quantifiers::QuantifierKind;
            use sympy_extras_core::solvers::integers as zi;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let int_of = |b: &Basic| b.as_rational().map(|v| v.numerator().clone()).unwrap_or_default();
            let vector_of = |b: &Basic| -> zi::IntVector { b.args().iter().map(int_of).collect() };
            let matrix_of = |b: &Basic| -> zi::IntMatrix { b.args().iter().map(vector_of).collect() };
            let shown_vector = |v: &zi::IntVector| tuple(v.iter().map(|c| Basic::integer(c.clone())).collect());
            let shown_matrix = |m: &zi::IntMatrix| tuple(m.iter().map(shown_vector).collect());
            match a[0].to_string().as_str() {
                "hnf" => {
                    let (h, u) = zi::hermite_normal_form_with_transform(&matrix_of(&a[1]));
                    tuple(vec![shown_matrix(&h), shown_matrix(&u)])
                }
                "diophantine" => match zi::linear_diophantine_system(a[1].args(), a[2].args()).map_err(e)? {
                    None => tuple(vec![]),
                    Some((values, parameters)) => tuple(vec![tuple(values), tuple(parameters)]),
                },
                "hilbert" => {
                    // the completion beyond its limit is Python's RuntimeError
                    let basis = zi::hilbert_basis(&matrix_of(&a[1]), int_of(&a[2]).try_into().unwrap_or(10000))
                        .map_err(|err| match err {
                            sympy_extras_core::error::PolyError::Arithmetic(m) => {
                                pyo3::exceptions::PyRuntimeError::new_err(m)
                            }
                            other => e(other),
                        })?;
                    shown_matrix(&basis)
                }
                "minimal" => {
                    let (p, h) =
                        zi::minimal_nonnegative_solutions(&matrix_of(&a[1]), &vector_of(&a[2]), 10000).map_err(e)?;
                    tuple(vec![shown_matrix(&p), shown_matrix(&h)])
                }
                "cooper" => zi::cooper(&a[1], &a[2]).map_err(e)?,
                "presburger" => {
                    let prefix: Vec<(QuantifierKind, Basic)> = a[2]
                        .args()
                        .iter()
                        .map(|q| {
                            let k = if q.args()[0].to_string() == "exists" {
                                QuantifierKind::Exists
                            } else {
                                QuantifierKind::ForAll
                            };
                            (k, q.args()[1].clone())
                        })
                        .collect();
                    zi::presburger_quantifier_elimination(&a[1], &prefix).map_err(e)?
                }
                "is_presburger" => {
                    sympy_extras_core::core::logic::boolean(zi::is_presburger(&a[1], a[2].args()).map_err(e)?)
                }
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "virtual_sub" => {
            // op (linear, quadratic, nnf, with_sides, linear_qe, qe), formula, [x | Tuple of
            // Tuple(exists/forall, variable)] -> formula, or Tuple(formula, Tuple(remaining prefix))
            use sympy_extras_core::assumptions::quantifiers::QuantifierKind;
            use sympy_extras_core::polys::virtual_substitution as vs;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let prefix_of = |b: &Basic| -> Vec<(QuantifierKind, Basic)> {
                b.args()
                    .iter()
                    .map(|q| {
                        let k = if q.args()[0].to_string() == "exists" {
                            QuantifierKind::Exists
                        } else {
                            QuantifierKind::ForAll
                        };
                        (k, q.args()[1].clone())
                    })
                    .collect()
            };
            let shown = |p: &[(QuantifierKind, Basic)]| -> Basic {
                tuple(
                    p.iter()
                        .map(|(k, v)| {
                            let name = if *k == QuantifierKind::Exists {
                                "exists"
                            } else {
                                "forall"
                            };
                            tuple(vec![
                                sympy_extras_core::core::symbol::Symbol::new(name, &[]).0,
                                v.clone(),
                            ])
                        })
                        .collect(),
                )
            };
            match a[0].to_string().as_str() {
                "linear" => vs::eliminate_linear(&a[1], &a[2]).map_err(e)?,
                "quadratic" => vs::eliminate_quadratic(&a[1], &a[2]).map_err(e)?,
                "nnf" => vs::to_nnf(&a[1]),
                "with_sides" => vs::with_sides(&a[1]).map_err(e)?,
                "linear_qe" => {
                    let (f, rest) = vs::linear_quantifier_elimination(&a[1], &prefix_of(&a[2])).map_err(e)?;
                    tuple(vec![f, shown(&rest)])
                }
                "qe" => {
                    let (f, rest) = vs::virtual_substitution_elimination(&a[1], &prefix_of(&a[2]), 12, 2).map_err(e)?;
                    tuple(vec![f, shown(&rest)])
                }
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "refine" => sympy_extras_core::core::refine::refine(&a[0], &a[1]).map_err(e)?,
        "satask" => sympy_extras_core::core::logic::boolean(
            sympy_extras_core::assumptions::satask::predicates_consistent(&a[0]).map_err(e)?,
        ),
        "ask" => {
            // proposition, assumptions, [Tuple(context)] -> True, False, or Tuple() for None
            let context: Option<Vec<Basic>> = a.get(2).map(|c| c.args().to_vec());
            match sympy_extras_core::core::ask::ask(&a[0], &a[1], context.as_deref()).map_err(e)? {
                Some(v) => sympy_extras_core::core::logic::boolean(v),
                None => Basic::new(Head::Tuple, Vec::new()),
            }
        }
        "reliable" => {
            // op (form, value, complex, proved_zero), e, digits, Tuple of Tuple(symbol, value) ->
            // form: the expression or Tuple() for None; value: Tuple(re, im) as exact rationals of the
            // binary floats or Tuple(); complex: Tuple(re, im) of the f64 as rationals or Tuple();
            // proved_zero: a boolean
            use sympy_extras_core::numeric as nm;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let digits: usize = a.get(2).map(|d| d.to_string().parse().unwrap_or(20)).unwrap_or(20);
            let values: Vec<(Basic, Basic)> = a
                .get(3)
                .map(|t| {
                    t.args()
                        .iter()
                        .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
                        .collect()
                })
                .unwrap_or_default();
            let exact = |x: &sympy_extras_core::core::evalf::Real| -> Basic {
                let r = x.repr();
                let significand: sympy_extras_core::numbers::Int = r.significand().clone();
                let exponent: isize = r.exponent();
                let value: sympy_extras_core::numbers::Rat = if exponent >= 0 {
                    sympy_extras_core::numbers::Rat::from(significand << exponent as usize)
                } else {
                    sympy_extras_core::numbers::rat_from_parts(
                        significand,
                        sympy_extras_core::numbers::Int::ONE << (-exponent) as usize,
                    )
                };
                Basic::from_rational(value)
            };
            match a[0].to_string().as_str() {
                "form" => nm::reliable_form(&a[1], digits).unwrap_or_else(|| tuple(Vec::new())),
                "value" => match nm::reliable_value(&a[1], digits, &values) {
                    Some(z) => {
                        let (re, im) = sympy_extras_core::core::evalf::parts(&z);
                        tuple(vec![exact(&re), exact(&im)])
                    }
                    None => tuple(Vec::new()),
                },
                "complex" => match nm::reliable_complex(&a[1], digits, &values) {
                    Some((x, y)) => {
                        let f = |v: f64| -> Basic {
                            exact(&sympy_extras_core::core::evalf::Real::try_from(v).expect("a finite float"))
                        };
                        tuple(vec![f(x), f(y)])
                    }
                    None => tuple(Vec::new()),
                },
                "proved_zero" => sympy_extras_core::core::logic::boolean(nm::proved_zero(&a[1])),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "kind" => Basic::new(
            Head::Symbol(std::sync::Arc::new(sympy_extras_core::core::symbol::SymbolData::new(
                &sympy_extras_core::core::kind::kind_of(&a[0]).to_string(),
                &[],
            ))),
            vec![],
        ),
        "parse" => {
            // the text as the name of a symbol, evaluate (a boolean) -> the expression
            let text: String = match a[0].head() {
                Head::Symbol(d) => d.name.clone(),
                _ => a[0].to_string(),
            };
            let evaluate: bool = a
                .get(1)
                .map_or(true, |b| sympy_extras_core::core::logic::truth(b) == Some(true));
            sympy_extras_core::core::parsing::parse(&text, evaluate).map_err(e)?
        }
        "rosenfeld" => {
            // Tuple(equations), Tuple(functions), Tuple(variables) (empty: the default),
            // Tuple(blocks) (empty: orderly), Tuple(inequations), Tuple(probes) ->
            // Tuple(Tuple of Tuple(equations, inequations, leaders, saturation), Tuple(contains of
            // each probe), Tuple(remainder of each probe by each component))
            use sympy_extras_core::polys::differential::rosenfeld_groebner::rosenfeld_groebner;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let variables: Option<&[Basic]> = if a[2].args().is_empty() {
                None
            } else {
                Some(a[2].args())
            };
            let blocks: Vec<Vec<Basic>> = a[3].args().iter().map(|b| b.args().to_vec()).collect();
            let ranking: Option<&[Vec<Basic>]> = if blocks.is_empty() { None } else { Some(&blocks) };
            let ideal = rosenfeld_groebner(a[0].args(), a[1].args(), variables, ranking, a[4].args()).map_err(e)?;
            let mut components: Vec<Basic> = Vec::new();
            for c in &ideal.components {
                components.push(tuple(vec![
                    tuple(c.equations().map_err(e)?),
                    tuple(c.inequations().map_err(e)?),
                    tuple(c.leaders().map_err(e)?),
                    tuple(c.saturation().map_err(e)?),
                ]));
            }
            let mut contains: Vec<Basic> = Vec::new();
            let mut remainders: Vec<Basic> = Vec::new();
            for probe in a[5].args() {
                contains.push(sympy_extras_core::core::logic::boolean(
                    ideal.contains(probe).map_err(e)?,
                ));
                let mut row: Vec<Basic> = Vec::new();
                for c in &ideal.components {
                    row.push(c.remainder(probe).map_err(e)?);
                }
                remainders.push(tuple(row));
            }
            tuple(vec![tuple(components), tuple(contains), tuple(remainders)])
        }
        "complex_reduction" => {
            // cgs: op, Tuple(polys), Tuple(variables), Tuple(parameters) -> Tuple of
            //   Tuple(Tuple(equations), Tuple(nonzero), Tuple(basis));
            // exists: op, Tuple(equations), Tuple(inequations), Tuple(variables) -> formula;
            // qe: op, formula, Tuple of Tuple(exists/forall, variable) -> formula;
            // reduce: op, formula -> formula
            use sympy_extras_core::polys::comprehensive as cp;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            match a[0].to_string().as_str() {
                "cgs" => {
                    let branches =
                        cp::comprehensive_groebner_system(a[1].args(), a[2].args(), a[3].args()).map_err(e)?;
                    tuple(
                        branches
                            .into_iter()
                            .map(|b| tuple(vec![tuple(b.equations), tuple(b.nonzero), tuple(b.basis)]))
                            .collect(),
                    )
                }
                "exists" => cp::exists_complex(a[1].args(), a[2].args(), a[3].args()).map_err(e)?,
                "qe" => {
                    let prefix: Vec<(cp::Quantifier, Basic)> = a[2]
                        .args()
                        .iter()
                        .map(|q| {
                            let kind = if q.args()[0].to_string() == "exists" {
                                cp::Quantifier::Exists
                            } else {
                                cp::Quantifier::Forall
                            };
                            (kind, q.args()[1].clone())
                        })
                        .collect();
                    cp::complex_quantifier_elimination(&a[1], &prefix).map_err(e)?
                }
                "reduce" => cp::reduce_complex(&a[1]).map_err(e)?,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", other))),
            }
        }
        "rur" => {
            // Tuple(polys), Tuple(gens), t -> Tuple(form, polynomial, characteristic, denominator,
            // Tuple(numerators), Tuple(polynomials), Tuple(solutions), Tuple(real solutions)), each
            // solution a Tuple of the values of the variables
            use sympy_extras_core::polys::hermite::System;
            use sympy_extras_core::polys::rur::{rur, to_expr};
            let r = rur(&System::Polys(a[0].args(), a[1].args()), a.get(2)).map_err(e)?;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let polys = |v: &[sympy_extras_core::polys::rur::QPoly]| -> PyResult<Basic> {
                Ok(tuple(v.iter().map(to_expr).collect::<Result<_, _>>().map_err(e)?))
            };
            let points = |real: bool| -> PyResult<Basic> {
                let found = r.solutions(real).map_err(e)?;
                Ok(tuple(
                    found
                        .into_iter()
                        .map(|p| tuple(p.into_iter().map(|(_, v)| v).collect()))
                        .collect(),
                ))
            };
            tuple(vec![
                r.form.clone(),
                to_expr(&r.polynomial).map_err(e)?,
                to_expr(&r.characteristic).map_err(e)?,
                to_expr(&r.denominator).map_err(e)?,
                polys(&r.numerators)?,
                polys(&r.polynomials().map_err(e)?)?,
                points(false)?,
                points(true)?,
            ])
        }
        "hermite_form" => {
            // op (a symbol), Tuple(polys), Tuple(gens), extra... ; decide: the formula as the
            // extra argument, None as an empty Tuple
            use sympy_extras_core::polys::hermite as hm;
            use sympy_extras_core::polys::ideals::PolyLike;
            let op: String = a[0].to_string();
            let (polys, gens) = (a[1].args(), a[2].args());
            let system = hm::System::Polys(polys, gens);
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let rational = |r: &sympy_extras_core::numbers::Rat| Basic::from_rational(r.clone());
            match op.as_str() {
                "count_real" => Basic::int(hm::count_real_solutions(&system).map_err(e)?),
                "count_complex" => Basic::int(hm::count_complex_solutions(&system).map_err(e)?),
                "tarski" => Basic::int(hm::tarski_query(&system, Some(&PolyLike::Expr(&a[3]))).map_err(e)?),
                "sign_counts" => {
                    let c = hm::real_sign_counts(&system, &PolyLike::Expr(&a[3])).map_err(e)?;
                    tuple(vec![Basic::int(c.zero), Basic::int(c.positive), Basic::int(c.negative)])
                }
                "sign_determination" => {
                    let likes: Vec<PolyLike<_>> = a[3].args().iter().map(PolyLike::Expr).collect();
                    let counts = hm::sign_determination(&system, &likes).map_err(e)?;
                    tuple(
                        counts
                            .into_iter()
                            .map(|(s, c)| {
                                tuple(vec![
                                    tuple(s.into_iter().map(|v| Basic::int(v as i64)).collect()),
                                    Basic::int(c),
                                ])
                            })
                            .collect(),
                    )
                }
                "matrix" => {
                    let g = a.get(3).map(PolyLike::Expr);
                    let rows = hm::hermite_matrix(&system, g.as_ref()).map_err(e)?;
                    tuple(rows.iter().map(|r| tuple(r.iter().map(rational).collect())).collect())
                }
                "decide" => match hm::decide_zero_dimensional(&a[3], gens, 64).map_err(e)? {
                    Some(v) => sympy_extras_core::core::logic::boolean(v),
                    None => tuple(Vec::new()),
                },
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", op))),
            }
        }
        "ideal" => {
            // op (a symbol), Tuple(gens), Tuple(symbols), order (a symbol), modulus (0 for QQ),
            // extra arguments... -> the result of ideal_operation
            use sympy_extras_core::polys::ideals::ideal_operation;
            use sympy_extras_core::polys::rings::MonomialOrder;
            let op: String = a[0].to_string();
            let order: MonomialOrder = MonomialOrder::named(&a[3].to_string()).map_err(e)?;
            let modulus: u64 = a[4]
                .to_string()
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("a modulus"))?;
            let (gens, symbols, extra) = (a[1].args(), a[2].args(), &a[5..]);
            if modulus == 0 {
                ideal_operation(&op, gens, symbols, sympy_extras_core::polys::domains::QQ, order, extra).map_err(e)?
            } else {
                let k = sympy_extras_core::polys::domains::GF(modulus).map_err(e)?;
                ideal_operation(&op, gens, symbols, k, order, extra).map_err(e)?
            }
        }
        "thue" | "thue_units" | "thue_norm" | "thue_short" => {
            // thue: F, m, x, y -> Tuple of (x, y); thue_units: Tuple(g) -> Tuple(Tuple(units),
            // Tuple(torsion)); thue_norm: Tuple(g), norm, Tuple(units) -> Tuple(elements);
            // thue_short: Tuple(rows), radius (a rational) -> Tuple(combinations)
            use sympy_extras_core::numbers::Int;
            use sympy_extras_core::solvers::thue_equation as th;
            let int = |b: &Basic| -> PyResult<Int> {
                match b.as_rational() {
                    Some(v) if v.is_int() => Ok(v.numerator().clone()),
                    _ => Err(pyo3::exceptions::PyValueError::new_err("an integer")),
                }
            };
            let ints = |b: &Basic| -> PyResult<Vec<Int>> { b.args().iter().map(int).collect() };
            let elements = |v: &[Vec<Int>]| -> Basic {
                Basic::new(
                    Head::Tuple,
                    v.iter()
                        .map(|e| Basic::new(Head::Tuple, e.iter().map(|c| Basic::integer(c.clone())).collect()))
                        .collect(),
                )
            };
            match name {
                "thue" => {
                    let found = th::thue(&a[0], &int(&a[1])?, &a[2], &a[3], 64).map_err(e)?;
                    elements(&found.into_iter().map(|(x, y)| vec![x, y]).collect::<Vec<_>>())
                }
                "thue_units" => {
                    let (units, torsion) = th::units_of_order(&ints(&a[0])?, 64).map_err(e)?;
                    Basic::new(Head::Tuple, vec![elements(&units), elements(&torsion)])
                }
                "thue_norm" => {
                    let units: Vec<Vec<Int>> = a[2].args().iter().map(ints).collect::<PyResult<_>>()?;
                    elements(&th::elements_of_norm(&ints(&a[0])?, &int(&a[1])?, &units, 100000).map_err(e)?)
                }
                _ => {
                    let rows: Vec<Vec<Int>> = a[0].args().iter().map(ints).collect::<PyResult<_>>()?;
                    let radius: f64 = a[1].to_string().parse::<f64>().unwrap_or_else(|_| {
                        let v = a[1].as_rational().cloned().unwrap_or_default();
                        sympy_extras_core::numbers::numer(&v).to_f64().value()
                            / sympy_extras_core::numbers::denom(&v).to_f64().value()
                    });
                    let found = th::short_vectors(&rows, radius, 0).map_err(e)?;
                    elements(
                        &found
                            .into_iter()
                            .map(|v| v.into_iter().map(Int::from).collect())
                            .collect::<Vec<_>>(),
                    )
                }
            }
        }
        "field_matrix" => {
            // op (a symbol), rows, cols, entries..., [rows2, cols2, entries2...] -> a Tuple of
            // Tuple(rows, cols, Tuple(entries)) (rref: the pivots last, rank: an Integer)
            use sympy_extras_core::integrals::risch::polymatrix::FieldMatrix as M;
            use sympy_extras_core::integrals::risch::tower::RationalFunction as R;
            let op: String = a[0].to_string();
            let size = |b: &Basic| -> PyResult<usize> {
                b.to_string()
                    .parse()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("a size"))
            };
            let read = |start: usize| -> PyResult<(M, usize)> {
                let (m, n) = (size(&a[start])?, size(&a[start + 1])?);
                let entries: Vec<R> = a[start + 2..start + 2 + m * n]
                    .iter()
                    .map(R::from_expr)
                    .collect::<Result<_, _>>()
                    .map_err(e)?;
                Ok((M::new(m, n, entries).map_err(e)?, start + 2 + m * n))
            };
            let shown = |m: &M| -> PyResult<Basic> {
                let entries: Vec<Basic> = m
                    .flat()
                    .iter()
                    .map(|r| r.as_expr())
                    .collect::<Result<_, _>>()
                    .map_err(e)?;
                Ok(Basic::new(
                    Head::Tuple,
                    vec![
                        Basic::int(m.rows as i64),
                        Basic::int(m.cols as i64),
                        Basic::new(Head::Tuple, entries),
                    ],
                ))
            };
            let (first, next) = read(1)?;
            let second = || read(next).map(|(m, _)| m);
            let out: Vec<Basic> = match op.as_str() {
                "rref" => {
                    let (r, pivots) = first.rref().map_err(e)?;
                    vec![
                        shown(&r)?,
                        Basic::new(Head::Tuple, pivots.iter().map(|p| Basic::int(*p as i64)).collect()),
                    ]
                }
                "nullspace" => first
                    .nullspace()
                    .map_err(e)?
                    .iter()
                    .map(shown)
                    .collect::<PyResult<_>>()?,
                "rank" => vec![Basic::int(first.rank().map_err(e)? as i64)],
                "transpose" => vec![shown(&first.transpose())?],
                "mul" => vec![shown(&first.mul(&second()?).map_err(e)?)?],
                "add" => vec![shown(&first.add(&second()?).map_err(e)?)?],
                "sub" => vec![shown(&first.sub(&second()?).map_err(e)?)?],
                "row_join" => vec![shown(&first.row_join(&second()?).map_err(e)?)?],
                "col_join" => vec![shown(&first.col_join(&second()?).map_err(e)?)?],
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", op))),
            };
            Basic::new(Head::Tuple, out)
        }
        "tower_rf" => {
            // op (a symbol), elements... -> Tuple(expr, Tuple(gens))
            use sympy_extras_core::integrals::risch::tower::RationalFunction as R;
            let op: String = a[0].to_string();
            let x = |i: usize| R::from_expr(&a[i]);
            let r: R = match op.as_str() {
                "from_expr" => x(1).map_err(e)?,
                "add" => x(1).and_then(|p| p.add(&x(2)?)).map_err(e)?,
                "sub" => x(1).and_then(|p| p.sub(&x(2)?)).map_err(e)?,
                "mul" => x(1).and_then(|p| p.mul(&x(2)?)).map_err(e)?,
                "div" => x(1).and_then(|p| p.div(&x(2)?)).map_err(e)?,
                "pow" => {
                    let n: i64 = a[2]
                        .to_string()
                        .parse()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("an integer"))?;
                    x(1).and_then(|p| p.pow(n)).map_err(e)?
                }
                "diff" => x(1).and_then(|p| p.diff(&a[2])).map_err(e)?,
                "subs" => x(1).and_then(|p| p.subs(&a[2], &x(3)?)).map_err(e)?,
                "conjugate_i" => x(1).and_then(|p| p.conjugate_i()).map_err(e)?,
                "numerator" => x(1).and_then(|p| p.numerator()).map_err(e)?,
                "denominator" => x(1).and_then(|p| p.denominator()).map_err(e)?,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown op {}", op))),
            };
            Basic::new(
                Head::Tuple,
                vec![r.as_expr().map_err(e)?, Basic::new(Head::Tuple, r.gens.clone())],
            )
        }
        "tower_poly" => {
            // op (a symbol), polynomial expressions..., the generator last (subs, as_poly and
            // compose_shift: further arguments after it) -> a Tuple of results
            use sympy_extras_core::integrals::risch::tower::{RationalFunction as R, TowerPoly as T};
            let op: String = a[0].to_string();
            let t: Basic = a[match op.as_str() {
                "from_expr" | "monic" | "sqf_list_include" | "diff" | "subs" | "as_poly" | "compose_shift"
                | "degree" | "as_element" => 2,
                _ => 3,
            }]
            .clone();
            let p = |i: usize| T::from_expr(&a[i], &t);
            let px = |q: &T| q.as_expr();
            let tup = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let out: Vec<Basic> = (|| -> sympy_extras_core::error::PolyResult<Vec<Basic>> {
                Ok(match op.as_str() {
                    "from_expr" => vec![px(&p(1)?)?],
                    "monic" => vec![px(&p(1)?.monic()?)?],
                    "degree" => vec![Basic::int(p(1)?.degree(a.get(3)))],
                    "as_element" => vec![p(1)?.as_element()?.as_expr()?],
                    "diff" => vec![px(&p(1)?.diff(a.get(3))?)?],
                    "subs" => vec![px(&p(1)?.subs(&a[3], &R::from_expr(&a[4])?)?)?],
                    "as_poly" => match p(1)?.as_poly(&a[3])? {
                        Some(q) => vec![px(&q)?, q.gen.clone()],
                        None => vec![],
                    },
                    "compose_shift" => vec![px(&p(1)?.compose_shift(&R::from_expr(&a[3])?))?],
                    "sqf_list_include" => p(1)?
                        .sqf_list_include()
                        .iter()
                        .map(|(f, k)| Ok(tup(vec![px(f)?, Basic::int(*k as i64)])))
                        .collect::<sympy_extras_core::error::PolyResult<_>>()?,
                    "gcd" => vec![px(&p(1)?.gcd(&p(2)?)?)?],
                    "lcm" => vec![px(&p(1)?.lcm(&p(2)?)?)?],
                    "exquo" => vec![px(&p(1)?.exquo(&p(2)?)?)?],
                    "invert" => vec![px(&p(1)?.invert(&p(2)?)?)?],
                    "div" => {
                        let (q, r) = p(1)?.div(&p(2)?)?;
                        vec![px(&q)?, px(&r)?]
                    }
                    "cancel" => {
                        let (q, r) = p(1)?.cancel(&p(2)?)?;
                        vec![px(&q)?, px(&r)?]
                    }
                    "half_gcdex" => {
                        let (s, h) = p(1)?.half_gcdex(&p(2)?)?;
                        vec![px(&s)?, px(&h)?]
                    }
                    "gcdex" => {
                        let (s, u, h) = p(1)?.gcdex(&p(2)?)?;
                        vec![px(&s)?, px(&u)?, px(&h)?]
                    }
                    "resultant" => vec![p(1)?.resultant(&p(2)?)?.as_expr()?],
                    "resultant_prs" => {
                        let (res, prs) = p(1)?.resultant_prs(&p(2)?)?;
                        let mut v: Vec<Basic> = vec![res.as_expr()?];
                        for q in &prs {
                            v.push(px(q)?);
                        }
                        v
                    }
                    _ => return Err(sympy_extras_core::error::value_error(format!("unknown op {}", op))),
                })
            })()
            .map_err(e)?;
            tup(out)
        }
        "curve_integrate" => {
            // f, x -> Tuple(elementary part, remainder, verdict: True / False / Integer(0) for None)
            use sympy_extras_core::integrals::curves::{integrate_on_curve, AlgebraicIntegrand};
            let integrand = AlgebraicIntegrand::new(&a[0], &a[1]).map_err(e)?;
            let result = integrate_on_curve(&integrand, true).map_err(e)?;
            let verdict = match result.nonelementary {
                Some(v) => sympy_extras_core::core::logic::boolean(v),
                None => Basic::zero(),
            };
            Basic::new(
                Head::Tuple,
                vec![
                    result.elementary_part(None).map_err(e)?,
                    result.remainder_expr(None).map_err(e)?,
                    verdict,
                ],
            )
        }
        "curve_hermite" => {
            // f, x -> Tuple(N, elementary part, remainder, verdict: True / False / Integer(0) for None)
            use sympy_extras_core::integrals::curves::{hermite_part, AlgebraicIntegrand};
            let integrand = AlgebraicIntegrand::new(&a[0], &a[1]).map_err(e)?;
            let part = hermite_part(&integrand).map_err(e)?;
            let verdict = match part.nonelementary {
                Some(v) => sympy_extras_core::core::logic::boolean(v),
                None => Basic::zero(),
            };
            Basic::new(
                Head::Tuple,
                vec![
                    Basic::integer(part.transformed.n.into()),
                    part.elementary_part().map_err(e)?,
                    part.remainder_expr().map_err(e)?,
                    verdict,
                ],
            )
        }
        "curve_integrand" | "curve_transformed" => {
            // curve_integrand: f, x -> Tuple(curve, element_expr, polynomial_to_expr(element_expr), theta,
            //   Tuple(weights), y); curve_transformed: f, x, N -> Tuple(curve in z and y,
            //   to_x_expr of the transformed integrand, z, y)
            use sympy_extras_core::integrals::curves::{AlgebraicIntegrand, Transformed};
            let t = |items: Vec<Basic>| Basic::new(Head::Tuple, items);
            let integrand = AlgebraicIntegrand::new(&a[0], &a[1]).map_err(e)?;
            if name == "curve_integrand" {
                let element = integrand.element_expr().map_err(e)?;
                let radicals = integrand.polynomial_to_expr(&element).map_err(e)?;
                t(vec![
                    integrand.curve.clone(),
                    element,
                    radicals,
                    integrand.theta.clone(),
                    t(integrand.weights.iter().map(|w| Basic::integer((*w).into())).collect()),
                    integrand.y.clone(),
                ])
            } else {
                let n: i64 = a[2]
                    .as_rational()
                    .and_then(|v| i64::try_from(v.numerator()).ok())
                    .unwrap_or(0);
                let transformed = Transformed::new(&integrand, n).map_err(e)?;
                let back = transformed
                    .to_x_expr(&transformed.field, &transformed.integrand_element, &Basic::zero())
                    .map_err(e)?;
                t(vec![
                    transformed.field.original().map_err(e)?,
                    back,
                    transformed.z.clone(),
                    integrand.y.clone(),
                ])
            }
        }
        "pyrandom_randint" => {
            // seed (an Integer, or a Symbol whose name is the string seed), a, b, count
            use sympy_extras_core::random::PyRandom;
            let mut rng: PyRandom = match a[0].as_rational() {
                Some(v) => PyRandom::from_int(v.numerator()),
                None => PyRandom::from_string(&a[0].to_string()),
            };
            let int = |b: &Basic| {
                b.as_rational()
                    .and_then(|v| i64::try_from(v.numerator()).ok())
                    .unwrap_or(0)
            };
            let (lo, hi, count) = (int(&a[1]), int(&a[2]), int(&a[3]));
            Basic::new(
                Head::Tuple,
                (0..count).map(|_| Basic::integer(rng.randint(lo, hi).into())).collect(),
            )
        }
        "puiseux_expansions" => {
            // F, x, y, point, order, Tuple(domain minpoly), domain alpha -> Tuple(Tuple(ramification,
            // conjugates, series as expr, Tuple(field minpoly), field alpha, order or false, Tuple(scale)))
            use sympy_extras_core::numbers::Rat;
            use sympy_extras_core::polys::domains::AlgebraicField;
            use sympy_extras_core::polys::puiseux as px;
            if a.len() != 7 {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments"));
            }
            let t = |items: Vec<Basic>| Basic::new(Head::Tuple, items);
            let minpoly: Vec<Rat> = a[5]
                .args()
                .iter()
                .map(|c| {
                    c.as_rational()
                        .cloned()
                        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("rationals"))
                })
                .collect::<PyResult<_>>()?;
            let domain: AlgebraicField = if minpoly.len() <= 2 {
                px::rationals()
            } else {
                AlgebraicField::new(minpoly, a[6].clone()).map_err(e)?
            };
            let order: i64 = a[4]
                .as_rational()
                .and_then(|v| i64::try_from(v.numerator()).ok())
                .unwrap_or(4);
            let found = px::puiseux_expansions(&a[0], &a[1], &a[2], &a[3], order, &domain).map_err(e)?;
            let number = |v: &[Rat]| t(v.iter().map(|q| Basic::from_rational(q.clone())).collect());
            let mut out: Vec<Basic> = Vec::new();
            for b in found {
                out.push(t(vec![
                    Basic::integer(b.ramification.into()),
                    Basic::integer((b.conjugates as i64).into()),
                    b.as_expr(None).map_err(e)?,
                    number(&b.field.minpoly),
                    b.field.alpha_expr.clone(),
                    b.order()
                        .map_or_else(sympy_extras_core::core::logic::false_, |o| Basic::integer(o.into())),
                    number(&b.scale),
                ]));
            }
            t(out)
        }
        "puiseux_field_from_polynomial" | "puiseux_field_of" | "puiseux_adjoin_root" | "puiseux_factor_over" => {
            // a field travels as (Tuple(minpoly), alpha), a number as Tuple(coordinates)
            use sympy_extras_core::numbers::Rat;
            use sympy_extras_core::polys::domains::AlgebraicField;
            use sympy_extras_core::polys::puiseux as px;
            let t = |items: Vec<Basic>| Basic::new(Head::Tuple, items);
            let rats = |b: &Basic| -> PyResult<Vec<Rat>> {
                b.args()
                    .iter()
                    .map(|c| {
                        c.as_rational()
                            .cloned()
                            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("rationals"))
                    })
                    .collect()
            };
            let number = |v: &[Rat]| t(v.iter().map(|q| Basic::from_rational(q.clone())).collect());
            let field = |k: &AlgebraicField| t(vec![number(&k.minpoly), k.alpha_expr.clone()]);
            let field_from = |m: &Basic, alpha: &Basic| -> PyResult<AlgebraicField> {
                let minpoly = rats(m)?;
                if minpoly.len() <= 2 {
                    return Ok(px::rationals());
                }
                AlgebraicField::new(minpoly, alpha.clone()).map_err(e)
            };
            match name {
                "puiseux_field_from_polynomial" => field(&px::field_from_polynomial(&rats(&a[0])?).map_err(e)?),
                "puiseux_field_of" => {
                    let (k, c) = px::field_of(&a[0]).map_err(e)?;
                    t(vec![field(&k), number(&c)])
                }
                "puiseux_adjoin_root" => {
                    let k = field_from(&a[0], &a[1])?;
                    let psi: Vec<Vec<Rat>> = a[2].args().iter().map(&rats).collect::<PyResult<_>>()?;
                    let (l, into, u) = px::adjoin_root(&k, &psi).map_err(e)?;
                    let image = into
                        .image
                        .as_ref()
                        .map_or_else(sympy_extras_core::core::logic::false_, |v| number(v));
                    let back = px::in_subfield(&into.apply(&px::generator(&k)), &l, &into)
                        .map_or_else(sympy_extras_core::core::logic::false_, |v| number(&v));
                    t(vec![
                        field(&l),
                        image,
                        number(&u),
                        number(&px::minimal_polynomial_of(&u, &l)),
                        back,
                    ])
                }
                _ => {
                    let k = field_from(&a[0], &a[1])?;
                    let f: Vec<Vec<Rat>> = a[2].args().iter().map(&rats).collect::<PyResult<_>>()?;
                    let factors = px::factor_over(&k, &f).map_err(e)?;
                    t(factors
                        .into_iter()
                        .map(|(g, m)| {
                            t(vec![
                                t(g.iter().map(|c| number(c)).collect()),
                                Basic::integer((m as i64).into()),
                            ])
                        })
                        .collect())
                }
            }
        }
        "pi_groups" | "pi_exponents" | "solve_for" | "dimension_matrix" | "dimension_of" => {
            // the quantities: Tuple(Tuple(Symbol(name), Tuple(Tuple(Symbol(base), exponent), ...)), ...)
            use sympy_extras_core::numbers::Rat;
            use sympy_extras_core::physics::dimensional as dm;
            let t = |items: Vec<Basic>| Basic::new(Head::Tuple, items);
            let sym = |n: &str| sympy_extras_core::core::Symbol::new(n, &[]).0;
            let pairs = |b: &Basic| -> PyResult<Vec<(String, Rat)>> {
                b.args()
                    .iter()
                    .map(|p| {
                        let value = p.args()[1]
                            .as_rational()
                            .cloned()
                            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("a rational exponent"))?;
                        Ok((p.args()[0].to_string(), value))
                    })
                    .collect()
            };
            let quantities_of = |b: &Basic| -> PyResult<dm::Quantities> {
                b.args()
                    .iter()
                    .map(|q| Ok((q.args()[0].to_string(), pairs(&q.args()[1])?)))
                    .collect()
            };
            let ints = |v: &[sympy_extras_core::numbers::Int]| t(v.iter().map(|x| Basic::integer(x.clone())).collect());
            match name {
                "pi_groups" => t(dm::pi_groups(&quantities_of(&a[0])?).map_err(e)?),
                "pi_exponents" => t(dm::pi_exponents(&quantities_of(&a[0])?)
                    .map_err(e)?
                    .iter()
                    .map(|v| ints(v))
                    .collect()),
                "solve_for" => dm::solve_for(&a[0].to_string(), &quantities_of(&a[1])?).map_err(e)?,
                "dimension_matrix" => {
                    let (bases, rows) = dm::dimension_matrix(&quantities_of(&a[0])?);
                    t(vec![
                        t(bases.iter().map(|b| sym(b)).collect()),
                        t(rows.iter().map(|r| ints(r)).collect()),
                    ])
                }
                _ => {
                    let result = dm::dimension_of(&pairs(&a[0])?, &quantities_of(&a[1])?).map_err(e)?;
                    t(result
                        .into_iter()
                        .map(|(b, x)| t(vec![sym(&b), Basic::from_rational(x)]))
                        .collect())
                }
            }
        }
        "triangularize" | "parametric_cases" => {
            // triangularize: Tuple(equations), Tuple(symbols), Tuple(inequations), kalkbrener (0/1)
            //   -> Tuple(Tuple(Tuple(polys), Tuple(main variables), Tuple(initials)), ...)
            // parametric_cases: Tuple(equations), Tuple(unknowns), Tuple(parameters) or false,
            //   Tuple(inequations) -> Tuple(Tuple(condition, solutions or false, Tuple(equations)), ...)
            if a.len() != 4 {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments"));
            }
            let t = |items: Vec<Basic>| Basic::new(Head::Tuple, items);
            if name == "triangularize" {
                let chains = sympy_extras_core::polys::regularchains::regularchain::triangularize(
                    a[0].args(),
                    a[1].args(),
                    a[2].args(),
                    a[3] != Basic::zero(),
                )
                .map_err(e)?;
                let mut out: Vec<Basic> = Vec::new();
                for c in chains {
                    out.push(t(vec![
                        t(c.polys().map_err(e)?),
                        t(c.main_variables()),
                        t(c.initials().map_err(e)?),
                    ]));
                }
                t(out)
            } else {
                let parameters: Option<&[Basic]> = if matches!(a[2].head(), Head::Tuple) {
                    Some(a[2].args())
                } else {
                    None
                };
                let cases = sympy_extras_core::solvers::parametric::parametric_cases(
                    a[0].args(),
                    a[1].args(),
                    parameters,
                    a[3].args(),
                )
                .map_err(e)?;
                let mut out: Vec<Basic> = Vec::new();
                for c in cases {
                    out.push(t(vec![
                        c.condition,
                        c.solutions.unwrap_or_else(sympy_extras_core::core::logic::false_),
                        t(c.equations),
                    ]));
                }
                t(out)
            }
        }
        "derivative_steps" => {
            // f, x, branch (0 or 1), depth (-1 for all), minor (0 or 1): the formatted
            // tree, the result and the number of steps
            use sympy_extras_core::steps::core::{count_steps, format_steps};
            if a.len() != 5 {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments"));
            }
            let step =
                sympy_extras_core::steps::derivatives::derivative_steps(&a[0], &a[1], a[2] != Basic::zero(), Some(100));
            let depth: Option<usize> = a[3].as_rational().and_then(|v| usize::try_from(v.numerator()).ok());
            let text: String = format_steps(&step, depth, a[4] != Basic::zero(), "  ");
            let label = sympy_extras_core::core::Symbol::new(&text, &[]).0;
            let count = Basic::integer((count_steps(&step) as i64).into());
            Basic::new(Head::Tuple, vec![label, step.result().map_err(e)?, count])
        }
        "gammasimp" | "combsimp" => {
            use sympy_extras_core::simplify::gammasimp as gs;
            if name == "gammasimp" {
                gs::gammasimp(&a[0]).map_err(e)?
            } else {
                gs::combsimp(&a[0]).map_err(e)?
            }
        }
        "rewrite" => {
            // e, Symbol(target name), piecewise (0 or 1), sample nodes of the patterns
            use sympy_extras_core::core::rewriting::{rewrite, RewriteHints};
            if a.len() < 3 {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments"));
            }
            let mut patterns: Vec<Function> = Vec::new();
            for node in &a[3..] {
                match node.head() {
                    Head::Function(f) => patterns.push(*f),
                    _ => return Err(pyo3::exceptions::PyTypeError::new_err("a pattern is a function node")),
                }
            }
            let hints = RewriteHints {
                piecewise: a[2] != Basic::zero(),
            };
            rewrite(&a[0], &patterns, &a[1].to_string(), &hints).map_err(e)?
        }
        "count_ops" | "count_ops_visual" => {
            use sympy_extras_core::core::function as fc;
            if name == "count_ops" {
                Basic::integer(fc::count_ops(&a[0]).map_err(e)?.into())
            } else {
                fc::count_ops_visual(&a[0]).map_err(e)?
            }
        }
        "postorder_traversal" | "iterargs" | "iterfreeargs" | "fuzzy_group" | "bottom_up_x2" | "use_x2" => {
            use sympy_extras_core::core::{fuzzy, traversal as tv};
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let flag = |b: &Basic| sympy_extras_core::core::logic::truth(b) == Some(true);
            // the function of the tests: the symbol x replaced by 2
            let x2 = |n: &Basic| -> sympy_extras_core::error::PolyResult<Basic> {
                Ok(if n.is_symbol() && n.to_string() == "x" {
                    Basic::integer(2i64.into())
                } else {
                    n.clone()
                })
            };
            match (name, a.len()) {
                ("postorder_traversal", 1) => tuple(tv::postorder_traversal(&a[0])),
                ("iterargs", 1) => tuple(tv::iterargs(&a[0])),
                ("iterfreeargs", 1) => tuple(tv::iterfreeargs(&a[0])),
                ("fuzzy_group", 2) => {
                    let values: Vec<Option<bool>> =
                        a[0].args().iter().map(sympy_extras_core::core::logic::truth).collect();
                    match fuzzy::fuzzy_group(values, flag(&a[1])) {
                        Some(v) => sympy_extras_core::core::logic::boolean(v),
                        None => Basic::integer(0i64.into()),
                    }
                }
                ("bottom_up_x2", 2) => tv::bottom_up(&a[0], &x2, flag(&a[1])).map_err(e)?,
                ("use_x2", 2) => {
                    let level: usize = a[1]
                        .as_rational()
                        .and_then(|v| usize::try_from(v.numerator()).ok())
                        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("a level"))?;
                    tv::use_at(&a[0], &x2, level).map_err(e)?
                }
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments")),
            }
        }
        "rootof" | "all_roots" | "real_roots" | "root_count" | "real_roots_indexed" | "crootof_address"
        | "complex_root_of" => {
            use sympy_extras_core::polys::complexrootof as cr;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let flag = |b: &Basic| sympy_extras_core::core::logic::truth(b) == Some(true);
            // the variable: a symbol, or 0 for none
            let gen = |b: &Basic| if b.is_symbol() { Some(b.clone()) } else { None };
            let int = |n: usize| Basic::integer((n as i64).into());
            match (name, a.len()) {
                ("rootof", 4) => {
                    let k: i64 = a[1]
                        .as_rational()
                        .filter(|v| v.is_int())
                        .and_then(|v| i64::try_from(v.numerator()).ok())
                        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("expected an integer root index"))?;
                    cr::rootof(&a[0], k, gen(&a[2]).as_ref(), flag(&a[3])).map_err(e)?
                }
                ("all_roots", 3) => tuple(cr::all_roots(&a[0], gen(&a[1]).as_ref(), flag(&a[2])).map_err(e)?),
                ("real_roots", 3) => tuple(cr::real_roots(&a[0], gen(&a[1]).as_ref(), flag(&a[2])).map_err(e)?),
                ("root_count", 2) => {
                    let (r, n) = cr::root_count(&a[0], gen(&a[1]).as_ref()).map_err(e)?;
                    tuple(vec![int(r), int(n)])
                }
                ("real_roots_indexed", 1) => cr::real_roots_indexed(&a[0]).map_err(e)?,
                ("complex_root_of", 2) => cr::complex_root_of(&a[0], &a[1]).map_err(e)?,
                ("crootof_address", 1) if matches!(a[0].head(), Head::ComplexRootOf(_)) => {
                    let (f, i) = cr::address(&a[0]).map_err(e)?;
                    tuple(vec![tuple(f.into_iter().map(Basic::integer).collect()), int(i)])
                }
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments")),
            }
        }
        "named_set" | "image_set" => {
            use sympy_extras_core::sets::fancysets as fs;
            match (name, a.len()) {
                ("named_set", 1) => match a[0].to_string().as_str() {
                    "Naturals" => Basic::atom(Head::Naturals),
                    "Naturals0" => Basic::atom(Head::Naturals0),
                    "Integers" => Basic::atom(Head::Integers),
                    "Rationals" => Basic::atom(Head::Rationals),
                    _ => Basic::atom(Head::Complexes),
                },
                ("image_set", n) if n >= 2 => fs::image_set(&a[0], &a[1], &a[2..]).map_err(e)?,
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments")),
            }
        }
        "set_of_sets" | "power_set" | "powerset" | "power_set_elements" | "finite_elements" => {
            use sympy_extras_core::sets::sets as st;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            match (name, a.len()) {
                ("set_of_sets", _) => st::set_of_sets(&a).map_err(e)?,
                ("power_set", 1) => st::power_set(&a[0]).map_err(e)?,
                ("powerset", 1) => st::powerset(&a[0]).map_err(e)?,
                ("power_set_elements", 1) if matches!(a[0].head(), Head::PowerSet) => {
                    tuple(st::power_set_elements(&a[0]).map_err(e)?)
                }
                ("finite_elements", 1) => match st::finite_elements(&a[0]) {
                    Some(v) => tuple(v),
                    None => sympy_extras_core::core::logic::false_(),
                },
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments")),
            }
        }
        "forall" | "exists" | "prenex" => {
            use sympy_extras_core::assumptions::quantifiers as qf;
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let vars = |b: &Basic| -> Vec<Basic> {
                if matches!(b.head(), Head::Tuple) {
                    b.args().to_vec()
                } else {
                    vec![b.clone()]
                }
            };
            match (name, a.len()) {
                ("forall", 2) => qf::forall(&vars(&a[0]), &a[1]).map_err(e)?,
                ("exists", 2) => qf::exists(&vars(&a[0]), &a[1]).map_err(e)?,
                ("prenex", 1) => {
                    let (prefix, matrix) = qf::prenex(&a[0]).map_err(e)?;
                    let pairs: Vec<Basic> = prefix
                        .into_iter()
                        .map(|(k, v)| tuple(vec![sympy_extras_core::core::Symbol::new(k.name(), &[]).0, v]))
                        .collect();
                    tuple(vec![tuple(pairs), matrix])
                }
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments")),
            }
        }
        "to_clauses" | "clauses_conjunction" | "propositional_models" | "is_satisfiable" => {
            use sympy_extras_core::logic::{cnf, dpll2};
            let tuple = |v: Vec<Basic>| Basic::new(Head::Tuple, v);
            let encode = |c: &cnf::Clauses| -> Basic {
                tuple(vec![
                    tuple(c.atoms.clone()),
                    tuple(
                        c.clauses
                            .iter()
                            .map(|cl| tuple(cl.iter().map(|l| Basic::integer((*l).into())).collect()))
                            .collect(),
                    ),
                ])
            };
            let clauses: Vec<cnf::Clauses> = a.iter().map(|f| cnf::to_clauses(f, None)).collect();
            match (name, a.len()) {
                ("to_clauses", 1) => encode(&clauses[0]),
                ("clauses_conjunction", _) => encode(&cnf::conjunction(&clauses)),
                ("propositional_models", 1) => tuple(
                    dpll2::propositional_models(&clauses[0], None)
                        .into_iter()
                        .map(|m| {
                            tuple(
                                m.into_iter()
                                    .map(|(atom, v)| tuple(vec![atom, sympy_extras_core::core::logic::boolean(v)]))
                                    .collect(),
                            )
                        })
                        .collect(),
                ),
                ("is_satisfiable", 1) => sympy_extras_core::core::logic::boolean(dpll2::is_satisfiable(&clauses[0])),
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments")),
            }
        }
        "simplify_logic"
        | "SOPform"
        | "POSform"
        | "is_literal"
        | "is_cnf"
        | "is_dnf"
        | "is_nnf"
        | "to_nnf"
        | "eliminate_implications"
        | "distribute_and_over_or"
        | "distribute_or_over_and"
        | "to_cnf"
        | "to_dnf" => {
            use sympy_extras_core::logic::boolalg as ba;
            let wrong = || pyo3::exceptions::PyTypeError::new_err("wrong arguments");
            // flags are passed as the Booleans true/false, the form as the
            // symbols cnf/dnf (anything else: none), no dontcare as 0
            let flag = |b: &Basic| sympy_extras_core::core::logic::truth(b) == Some(true);
            let terms = |b: &Basic| -> PyResult<Vec<ba::Term>> {
                b.args()
                    .iter()
                    .map(|t| {
                        t.args()
                            .iter()
                            .map(|v| {
                                v.as_rational()
                                    .and_then(|r| u8::try_from(r.numerator()).ok())
                                    .ok_or_else(wrong)
                            })
                            .collect()
                    })
                    .collect()
            };
            let boolean = |v: bool| sympy_extras_core::core::logic::boolean(v);
            match (name, a.len()) {
                ("simplify_logic", 4) => {
                    let form: Option<ba::Form> = match a[1].to_string().as_str() {
                        "cnf" => Some(ba::Form::Cnf),
                        "dnf" => Some(ba::Form::Dnf),
                        _ => None,
                    };
                    let dontcare: Option<&Basic> = if a[3].is_number_node() { None } else { Some(&a[3]) };
                    ba::simplify_logic(&a[0], form, flag(&a[2]), dontcare).map_err(e)?
                }
                ("SOPform", 3) => ba::sop_form(a[0].args(), &terms(&a[1])?, &terms(&a[2])?).map_err(e)?,
                ("POSform", 3) => ba::pos_form(a[0].args(), &terms(&a[1])?, &terms(&a[2])?).map_err(e)?,
                ("is_literal", 1) => boolean(ba::is_literal(&a[0])),
                ("is_cnf", 1) => boolean(ba::is_cnf(&a[0])),
                ("is_dnf", 1) => boolean(ba::is_dnf(&a[0])),
                ("is_nnf", 2) => boolean(ba::is_nnf(&a[0], flag(&a[1]))),
                ("to_nnf", 2) => ba::to_nnf(&a[0], flag(&a[1])).map_err(e)?,
                ("eliminate_implications", 1) => ba::eliminate_implications(&a[0]).map_err(e)?,
                ("distribute_and_over_or", 1) => ba::distribute_and_over_or(&a[0]),
                ("distribute_or_over_and", 1) => ba::distribute_or_over_and(&a[0]),
                ("to_cnf", 3) => ba::to_cnf(&a[0], flag(&a[1]), flag(&a[2])).map_err(e)?,
                ("to_dnf", 3) => ba::to_dnf(&a[0], flag(&a[1]), flag(&a[2])).map_err(e)?,
                _ => return Err(wrong()),
            }
        }
        "hyperrep" | "rewrite_nonrep" | "rewrite_nonrepsmall" => {
            use sympy_extras_core::core::basic::Function as F;
            use sympy_extras_core::functions::special::hyperrep as hr;
            match name {
                "rewrite_nonrep" if a.len() == 1 => hr::rewrite_nonrep(&a[0], false).map_err(e)?,
                "rewrite_nonrepsmall" if a.len() == 1 => hr::rewrite_nonrep(&a[0], true).map_err(e)?,
                "hyperrep" if !a.is_empty() => {
                    // the head by its Python name, then the arguments
                    let label: String = a[0].to_string();
                    let f: F = [
                        F::HyperRepPower1,
                        F::HyperRepPower2,
                        F::HyperRepLog1,
                        F::HyperRepAtanh,
                        F::HyperRepAsin1,
                        F::HyperRepAsin2,
                        F::HyperRepSqrts1,
                        F::HyperRepSqrts2,
                        F::HyperRepLog2,
                        F::HyperRepCosasin,
                        F::HyperRepSinasin,
                    ]
                    .into_iter()
                    .find(|f| f.name() == label)
                    .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("unknown HyperRep"))?;
                    if a.len() - 1 != hr::hyperrep_arity(f) {
                        return Err(pyo3::exceptions::PyTypeError::new_err("wrong number of arguments"));
                    }
                    hr::hyperrep(f, &a[1..]).map_err(e)?
                }
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong arguments")),
            }
        }
        "airyai" | "airybi" | "airyaiprime" | "airybiprime" | "marcumq" => {
            use sympy_extras_core::core::basic::Function as F;
            use sympy_extras_core::functions::special::bessel as bs;
            match (name, a.len()) {
                ("marcumq", 3) => bs::marcumq(&a).map_err(e)?,
                ("airyai", 1) => bs::airy(F::AiryAi, &a[0]).map_err(e)?,
                ("airybi", 1) => bs::airy(F::AiryBi, &a[0]).map_err(e)?,
                ("airyaiprime", 1) => bs::airy(F::AiryAiPrime, &a[0]).map_err(e)?,
                ("airybiprime", 1) => bs::airy(F::AiryBiPrime, &a[0]).map_err(e)?,
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong number of arguments")),
            }
        }
        "besselj" | "bessely" | "besseli" | "besselk" | "hankel1" | "hankel2" | "besselis" | "besselks" | "jn"
        | "yn" | "hn1" | "hn2" => {
            use sympy_extras_core::core::basic::Function as F;
            let f: F = match name {
                "besselj" => F::BesselJ,
                "bessely" => F::BesselY,
                "besseli" => F::BesselI,
                "besselk" => F::BesselK,
                "hankel1" => F::Hankel1,
                "hankel2" => F::Hankel2,
                "besselis" => F::BesselIs,
                "besselks" => F::BesselKs,
                "jn" => F::SphericalJ,
                "yn" => F::SphericalY,
                "hn1" => F::SphericalHankel1,
                _ => F::SphericalHankel2,
            };
            if a.len() != 2 {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong number of arguments"));
            }
            sympy_extras_core::functions::special::bessel::bessel(f, &a).map_err(e)?
        }
        "chebyshevt_root" | "chebyshevu_root" | "jacobi_normalized" => {
            use sympy_extras_core::functions::special::polynomials as op;
            match (name, a.len()) {
                ("chebyshevt_root", 2) => op::chebyshevt_root(&a[0], &a[1]).map_err(e)?,
                ("chebyshevu_root", 2) => op::chebyshevu_root(&a[0], &a[1]).map_err(e)?,
                ("jacobi_normalized", 4) => op::jacobi_normalized(&a[0], &a[1], &a[2], &a[3]).map_err(e)?,
                _ => return Err(pyo3::exceptions::PyTypeError::new_err("wrong number of arguments")),
            }
        }
        "legendre" | "chebyshevt" | "chebyshevu" | "hermite" | "laguerre" | "assoc_laguerre" | "jacobi"
        | "gegenbauer" | "assoc_legendre" | "hermite_prob" => {
            use sympy_extras_core::core::basic::Function as F;
            use sympy_extras_core::functions::special::polynomials as op;
            let f: F = match name {
                "legendre" => F::Legendre,
                "chebyshevt" => F::ChebyshevT,
                "chebyshevu" => F::ChebyshevU,
                "hermite" => F::Hermite,
                "laguerre" => F::Laguerre,
                "assoc_laguerre" => F::AssocLaguerre,
                "jacobi" => F::Jacobi,
                "gegenbauer" => F::Gegenbauer,
                "assoc_legendre" => F::AssocLegendre,
                _ => F::HermiteProb,
            };
            if !op::arity_fits(f, a.len()) {
                return Err(pyo3::exceptions::PyTypeError::new_err("wrong number of arguments"));
            }
            op::orthogonal(f, &a).map_err(e)?
        }
        "mathieus" | "mathieuc" | "mathieusprime" | "mathieucprime" => {
            use sympy_extras_core::core::basic::Function as F;
            let f: F = match name {
                "mathieus" => F::MathieuS,
                "mathieuc" => F::MathieuC,
                "mathieusprime" => F::MathieuSPrime,
                _ => F::MathieuCPrime,
            };
            sympy_extras_core::functions::special::mathieu_functions::mathieu(f, &a).map_err(e)?
        }
        "Ynm" | "Znm" | "Ynm_c" => {
            use sympy_extras_core::functions::special::spherical_harmonics as sh;
            match name {
                "Ynm" => sh::ynm(&a[0], &a[1], &a[2], &a[3]),
                "Znm" => sh::znm(&a[0], &a[1], &a[2], &a[3]),
                _ => sh::ynm_c(&a[0], &a[1], &a[2], &a[3]),
            }
            .map_err(e)?
        }
        "SingularityFunction" => {
            sympy_extras_core::functions::special::singularity_functions::singularity_function(&a[0], &a[1], &a[2])
                .map_err(e)?
        }
        "lowergamma" | "uppergamma" | "multigamma" => {
            use sympy_extras_core::functions::special::gamma_functions as gf;
            match name {
                "lowergamma" => gf::lowergamma(&a[0], &a[1]),
                "uppergamma" => gf::uppergamma(&a[0], &a[1]),
                _ => gf::multigamma(&a[0], &a[1]),
            }
            .map_err(e)?
        }
        "gamma" | "loggamma" | "polygamma" | "digamma" | "trigamma" => {
            use sympy_extras_core::functions::special::gamma_functions as gf;
            match name {
                "gamma" => gf::gamma(&a[0]),
                "loggamma" => gf::loggamma(&a[0]),
                "polygamma" => gf::polygamma(&a[0], &a[1]),
                "digamma" => gf::digamma(&a[0]),
                _ => gf::trigamma(&a[0]),
            }
            .map_err(e)?
        }
        "factorial" | "binomial" | "rf" | "ff" | "subfactorial" | "factorial2" => {
            use sympy_extras_core::functions::combinatorial::factorials as fc;
            match name {
                "factorial" => fc::factorial(&a[0]),
                "binomial" => fc::binomial(&a[0], &a[1]),
                "rf" => fc::rf(&a[0], &a[1]),
                "ff" => fc::ff(&a[0], &a[1]),
                "subfactorial" => fc::subfactorial(&a[0]),
                _ => fc::factorial2(&a[0]),
            }
            .map_err(e)?
        }
        "reversed" => rl::reversed(&a[0]),
        "negated" => rl::negated(&a[0]),
        "reversedsign" => rl::reversedsign(&a[0]).map_err(e)?,
        "canonical" => rl::canonical(&a[0]).map_err(e)?,
        _ => return Err(PyTypeError::new_err(format!("unknown logic function {}", name))),
    };
    to_python(py, &value)
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
    m.add_function(wrap_pyfunction!(cad_projection_sets, m)?)?;
    m.add_function(wrap_pyfunction!(samplepoint_report, m)?)?;
    m.add_function(wrap_pyfunction!(cad_report, m)?)?;
    m.add_function(wrap_pyfunction!(qe_call, m)?)?;
    m.add_function(wrap_pyfunction!(cylindrical_call, m)?)?;
    m.add_function(wrap_pyfunction!(indexed_root, m)?)?;
    m.add_function(wrap_pyfunction!(logic_call, m)?)?;
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
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(dirac_delta, m)?)?;
    m.add_function(wrap_pyfunction!(heaviside, m)?)?;
    m.add_function(wrap_pyfunction!(ceiling, m)?)?;
    m.add_function(wrap_pyfunction!(frac, m)?)?;
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
