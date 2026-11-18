//! The parser of `sympify` for strings, ported from
//! `sympy_extras.core.parsing`: arithmetic expressions in Python's syntax
//! as SymPy's `parse_expr` reads them with its standard transformations
//! (integers, the constants, the functions by SymPy's names, symbols and
//! undefined functions; `+ - * / **` with `**` right associative and
//! binding tighter than a unary sign; parentheses and calls).
//!
//! Decimal numbers need `Float`, which the Rust core does not have
//! (`NotImplemented`); the plain callables of `sympy_extras.functions`
//! which are no function head (`Piecewise`, `hyper`, `meijerg`, ...) are
//! `NotImplemented` too, except `sqrt`, `cbrt`, `root`, `ln` and the
//! two-argument `log`. Outside the grammar: `ParseUnsupported`, a `Value`
//! error.

use std::sync::Arc;

use crate::core::basic::{Basic, Function, Head, UndefinedFunction};
use crate::core::symbol::Symbol;
use crate::error::{PolyError, PolyResult};
use crate::numbers::Int;

/// The function heads with SymPy's names.
const FUNCTIONS: &[(Function, &str)] = &[
    (Function::Exp, "exp"),
    (Function::Log, "log"),
    (Function::Re, "re"),
    (Function::Im, "im"),
    (Function::Conjugate, "conjugate"),
    (Function::Sign, "sign"),
    (Function::Abs, "Abs"),
    (Function::Sin, "sin"),
    (Function::Cos, "cos"),
    (Function::Tan, "tan"),
    (Function::Cot, "cot"),
    (Function::Sec, "sec"),
    (Function::Csc, "csc"),
    (Function::Sinh, "sinh"),
    (Function::Cosh, "cosh"),
    (Function::Tanh, "tanh"),
    (Function::Coth, "coth"),
    (Function::Sech, "sech"),
    (Function::Csch, "csch"),
    (Function::Asin, "asin"),
    (Function::Acos, "acos"),
    (Function::Atan, "atan"),
    (Function::Acot, "acot"),
    (Function::Asec, "asec"),
    (Function::Acsc, "acsc"),
    (Function::Asinh, "asinh"),
    (Function::Acosh, "acosh"),
    (Function::Atanh, "atanh"),
    (Function::Acoth, "acoth"),
    (Function::Asech, "asech"),
    (Function::Acsch, "acsch"),
    (Function::Arg, "arg"),
    (Function::Atan2, "atan2"),
    (Function::Max, "Max"),
    (Function::Min, "Min"),
    (Function::Floor, "floor"),
    (Function::Ceiling, "ceiling"),
    (Function::Frac, "frac"),
    (Function::DiracDelta, "DiracDelta"),
    (Function::Heaviside, "Heaviside"),
    (Function::Factorial, "factorial"),
    (Function::Binomial, "binomial"),
    (Function::RisingFactorial, "RisingFactorial"),
    (Function::FallingFactorial, "FallingFactorial"),
    (Function::Subfactorial, "subfactorial"),
    (Function::Factorial2, "factorial2"),
    (Function::Gamma, "gamma"),
    (Function::LogGamma, "loggamma"),
    (Function::PolyGamma, "polygamma"),
    (Function::SingularityFunction, "SingularityFunction"),
    (Function::Ynm, "Ynm"),
    (Function::Legendre, "legendre"),
    (Function::ChebyshevT, "chebyshevt"),
    (Function::ChebyshevU, "chebyshevu"),
    (Function::Hermite, "hermite"),
    (Function::Laguerre, "laguerre"),
    (Function::AssocLaguerre, "assoc_laguerre"),
    (Function::Jacobi, "jacobi"),
    (Function::Gegenbauer, "gegenbauer"),
    (Function::AssocLegendre, "assoc_legendre"),
    (Function::HermiteProb, "hermite_prob"),
    (Function::AppellF1, "appellf1"),
    (Function::BesselJ, "besselj"),
    (Function::BesselY, "bessely"),
    (Function::BesselI, "besseli"),
    (Function::BesselK, "besselk"),
    (Function::Hankel1, "hankel1"),
    (Function::Hankel2, "hankel2"),
    (Function::BesselIs, "_besseli"),
    (Function::BesselKs, "_besselk"),
    (Function::SphericalJ, "jn"),
    (Function::SphericalY, "yn"),
    (Function::SphericalHankel1, "hn1"),
    (Function::SphericalHankel2, "hn2"),
    (Function::AiryAi, "airyai"),
    (Function::AiryBi, "airybi"),
    (Function::AiryAiPrime, "airyaiprime"),
    (Function::AiryBiPrime, "airybiprime"),
    (Function::MarcumQ, "marcumq"),
    (Function::EllipticK, "elliptic_k"),
    (Function::EllipticF, "elliptic_f"),
    (Function::EllipticE, "elliptic_e"),
    (Function::EllipticPi, "elliptic_pi"),
    (Function::MathieuS, "mathieus"),
    (Function::MathieuC, "mathieuc"),
    (Function::MathieuSPrime, "mathieusprime"),
    (Function::MathieuCPrime, "mathieucprime"),
    (Function::Znm, "Znm"),
    (Function::LowerGamma, "lowergamma"),
    (Function::UpperGamma, "uppergamma"),
    (Function::MultiGamma, "multigamma"),
    (Function::DiGamma, "digamma"),
    (Function::TriGamma, "trigamma"),
    (Function::Zeta, "zeta"),
    (Function::DirichletEta, "dirichlet_eta"),
    (Function::PolyLog, "polylog"),
    (Function::LerchPhi, "lerchphi"),
    (Function::RiemannXi, "riemann_xi"),
    (Function::Stieltjes, "stieltjes"),
    (Function::KroneckerDelta, "KroneckerDelta"),
    (Function::LeviCivita, "LeviCivita"),
    (Function::Catalan, "catalan"),
    (Function::Fibonacci, "fibonacci"),
    (Function::Lucas, "lucas"),
    (Function::Harmonic, "harmonic"),
    (Function::Bernoulli, "bernoulli"),
    (Function::Euler, "euler"),
    (Function::Bell, "bell"),
    (Function::Genocchi, "genocchi"),
    (Function::Partition, "partition"),
    (Function::Totient, "totient"),
    (Function::ReducedTotient, "reduced_totient"),
    (Function::DivisorSigma, "divisor_sigma"),
    (Function::UDivisorSigma, "udivisor_sigma"),
    (Function::PrimeNu, "primenu"),
    (Function::PrimeOmega, "primeomega"),
    (Function::Mobius, "mobius"),
    (Function::PrimePi, "primepi"),
    (Function::LegendreSymbol, "legendre_symbol"),
    (Function::JacobiSymbol, "jacobi_symbol"),
    (Function::KroneckerSymbol, "kronecker_symbol"),
    (Function::Tribonacci, "tribonacci"),
    (Function::Motzkin, "motzkin"),
    (Function::Andre, "andre"),
    (Function::Beta, "beta"),
    (Function::BetaInc, "betainc"),
    (Function::BetaIncRegularized, "betainc_regularized"),
    (Function::Erf, "erf"),
    (Function::Erfc, "erfc"),
    (Function::Erfi, "erfi"),
    (Function::Erfinv, "erfinv"),
    (Function::Erfcinv, "erfcinv"),
    (Function::Erf2, "erf2"),
    (Function::Erf2inv, "erf2inv"),
    (Function::Expint, "expint"),
    (Function::Ei, "Ei"),
    (Function::Si, "Si"),
    (Function::Ci, "Ci"),
    (Function::Shi, "Shi"),
    (Function::Chi, "Chi"),
    (Function::LogIntegral, "li"),
    (Function::OffsetLogIntegral, "Li"),
    (Function::FresnelS, "fresnels"),
    (Function::FresnelC, "fresnelc"),
];

/// The function head of a SymPy name.
pub fn function_named(name: &str) -> Option<Function> {
    FUNCTIONS.iter().find(|(_, n)| *n == name).map(|(f, _)| *f)
}

fn unsupported(message: String) -> PolyError {
    PolyError::Value(format!("ParseUnsupported: {}", message))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Number(String),
    Name(String),
    Op(String),
    End,
}

fn tokens(text: &str) -> PolyResult<Vec<Token>> {
    let chars: Vec<char> = text.trim_end().chars().collect();
    let mut out: Vec<Token> = Vec::new();
    let mut i: usize = 0;
    let digits = |chars: &[char], mut k: usize| -> usize {
        while k < chars.len() && chars[k].is_ascii_digit() {
            k += 1;
        }
        k
    };
    // an exponent [eE][-+]?\d+ at k, its end or None
    let exponent = |chars: &[char], k: usize| -> Option<usize> {
        if k < chars.len() && (chars[k] == 'e' || chars[k] == 'E') {
            let mut m: usize = k + 1;
            if m < chars.len() && (chars[m] == '-' || chars[m] == '+') {
                m += 1;
            }
            let end: usize = digits(chars, m);
            if end > m {
                return Some(end);
            }
        }
        None
    };
    while i < chars.len() {
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        let c: char = chars[i];
        if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start: usize = i;
            let mut k: usize = digits(&chars, i);
            if k < chars.len() && chars[k] == '.' {
                k = digits(&chars, k + 1);
                if let Some(e) = exponent(&chars, k) {
                    k = e;
                }
            } else if let Some(e) = exponent(&chars, k) {
                k = e;
            }
            out.push(Token::Number(chars[start..k].iter().collect()));
            i = k;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start: usize = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Token::Name(chars[start..i].iter().collect()));
        } else if c == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            out.push(Token::Op("**".to_string()));
            i += 2;
        } else if "-+*/(),".contains(c) {
            out.push(Token::Op(c.to_string()));
            i += 1;
        } else {
            return Err(unsupported(format!("cannot parse {:?} at {}", text, i)));
        }
    }
    out.push(Token::End);
    Ok(out)
}

struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    evaluate: bool,
    text: &'a str,
}

fn op(s: &str) -> Token {
    Token::Op(s.to_string())
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &Token {
        &self.tokens[self.position]
    }

    fn take(&mut self) -> Token {
        let t: Token = self.tokens[self.position].clone();
        self.position += 1;
        t
    }

    fn expect(&mut self, s: &str) -> PolyResult<()> {
        if self.take() != op(s) {
            return Err(unsupported(format!("expected {:?} in {:?}", s, self.text)));
        }
        Ok(())
    }

    fn parse(&mut self) -> PolyResult<Basic> {
        let e: Basic = self.sum()?;
        if *self.peek() != Token::End {
            return Err(unsupported(format!("unexpected {:?} in {:?}", self.peek(), self.text)));
        }
        Ok(e)
    }

    fn sum(&mut self) -> PolyResult<Basic> {
        let mut e: Basic = self.product()?;
        while *self.peek() == op("+") || *self.peek() == op("-") {
            let plus: bool = self.take() == op("+");
            let right: Basic = self.product()?;
            let right: Basic = if plus { right } else { self.negate(&right)? };
            e = self.add(&e, &right)?;
        }
        Ok(e)
    }

    fn product(&mut self) -> PolyResult<Basic> {
        let mut e: Basic = self.unary()?;
        while *self.peek() == op("*") || *self.peek() == op("/") {
            let times: bool = self.take() == op("*");
            let right: Basic = self.unary()?;
            let right: Basic = if times {
                right
            } else {
                self.pow(&right, &Basic::int(-1))?
            };
            e = self.mul(&e, &right)?;
        }
        Ok(e)
    }

    fn unary(&mut self) -> PolyResult<Basic> {
        if *self.peek() == op("-") {
            self.take();
            let a: Basic = self.unary()?;
            return self.negate(&a);
        }
        if *self.peek() == op("+") {
            self.take();
            return self.unary();
        }
        self.power()
    }

    fn power(&mut self) -> PolyResult<Basic> {
        let base: Basic = self.call()?;
        if *self.peek() == op("**") {
            self.take();
            let exponent: Basic = self.unary()?;
            return self.pow(&base, &exponent);
        }
        Ok(base)
    }

    fn call(&mut self) -> PolyResult<Basic> {
        match self.take() {
            Token::Number(text) => number(&text),
            Token::Op(o) if o == "(" => {
                let e: Basic = self.sum()?;
                self.expect(")")?;
                Ok(e)
            }
            Token::Name(name) => {
                if *self.peek() == op("(") {
                    self.take();
                    let mut args: Vec<Basic> = Vec::new();
                    if *self.peek() != op(")") {
                        args.push(self.sum()?);
                        while *self.peek() == op(",") {
                            self.take();
                            args.push(self.sum()?);
                        }
                    }
                    self.expect(")")?;
                    return self.function(&name, args);
                }
                Ok(named(&name))
            }
            other => Err(unsupported(format!("unexpected {:?} in {:?}", other, self.text))),
        }
    }

    fn function(&self, name: &str, args: Vec<Basic>) -> PolyResult<Basic> {
        if name == "Rational" || name == "Integer" {
            let bound: usize = if name == "Rational" { 2 } else { 1 };
            let integers: Option<Vec<Int>> = args
                .iter()
                .map(|a| a.as_rational().filter(|v| v.is_int()).map(|v| v.numerator().clone()))
                .collect();
            let Some(values) = integers.filter(|v| !v.is_empty() && v.len() <= bound) else {
                return Err(unsupported(format!("{} of {:?} in {:?}", name, args, self.text)));
            };
            let q: Int = values.get(1).cloned().unwrap_or(Int::ONE);
            return crate::core::mul::div(&Basic::integer(values[0].clone()), &Basic::integer(q));
        }
        let arity = |n: usize| -> PolyResult<()> {
            if args.len() == n {
                Ok(())
            } else {
                Err(PolyError::Value(format!(
                    "{}() takes {} arguments ({} given)",
                    name,
                    n,
                    args.len()
                )))
            }
        };
        // Python's order: the class itself without evaluation, then the plain
        // callables (also without evaluation), then the class's `apply`
        if !self.evaluate && !(name == "log" && args.len() == 2) {
            if let Some(f) = function_named(name) {
                if !args.is_empty() {
                    return Ok(Basic::new(Head::Function(f), args));
                }
            }
        }
        {
            match name {
                "sqrt" => {
                    arity(1)?;
                    return crate::core::power::pow(&args[0], &Basic::half());
                }
                "cbrt" => {
                    arity(1)?;
                    return crate::core::power::pow(&args[0], &crate::core::mul::div(&Basic::one(), &Basic::int(3))?);
                }
                "root" => {
                    if args.len() != 2 && args.len() != 3 {
                        return Err(PolyError::Value(format!(
                            "root() takes 2 or 3 arguments ({} given)",
                            args.len()
                        )));
                    }
                    let exponent: Basic = crate::core::mul::div(&Basic::one(), &args[1])?;
                    let principal: Basic = crate::core::power::pow(&args[0], &exponent)?;
                    if args.len() == 3 && args[2] != Basic::zero() {
                        let twice: Basic = crate::core::mul::mul(&[Basic::int(2), args[2].clone()])?;
                        let unit: Basic =
                            crate::core::power::pow(&Basic::negative_one(), &crate::core::mul::div(&twice, &args[1])?)?;
                        return crate::core::mul::mul(&[principal, unit]);
                    }
                    return Ok(principal);
                }
                "log" | "ln" if args.len() == 2 => {
                    return crate::functions::elementary::exponential::log_base(&args[0], &args[1])
                }
                "ln" => {
                    arity(1)?;
                    return crate::functions::elementary::exponential::log(&args[0]);
                }
                _ => {}
            }
        }
        if let Some(f) = function_named(name) {
            if args.is_empty() {
                return Err(PolyError::Value(format!("{}() takes arguments", name)));
            }
            return crate::functions::apply_function(f, &args);
        }
        if PLAIN.contains(&name) {
            return Err(PolyError::NotImplemented(format!(
                "the function {} in the Rust parser",
                name
            )));
        }
        Arc::new(UndefinedFunction::new(name, None, &[])).apply(args)
    }

    fn add(&self, a: &Basic, b: &Basic) -> PolyResult<Basic> {
        if self.evaluate {
            crate::core::add::add(&[a.clone(), b.clone()])
        } else {
            Ok(Basic::new(Head::Add, vec![a.clone(), b.clone()]))
        }
    }

    fn mul(&self, a: &Basic, b: &Basic) -> PolyResult<Basic> {
        if self.evaluate {
            crate::core::mul::mul(&[a.clone(), b.clone()])
        } else {
            Ok(Basic::new(Head::Mul, vec![a.clone(), b.clone()]))
        }
    }

    fn negate(&self, a: &Basic) -> PolyResult<Basic> {
        if !self.evaluate && a.is_number_class() {
            return crate::core::mul::neg(a);
        }
        self.mul(&Basic::negative_one(), a)
    }

    fn pow(&self, a: &Basic, b: &Basic) -> PolyResult<Basic> {
        if self.evaluate {
            crate::core::power::pow(a, b)
        } else {
            Ok(Basic::new(Head::Pow, vec![a.clone(), b.clone()]))
        }
    }
}

/// The names of `sympy_extras.functions` (plain callables and function
/// classes) without a function head in the Rust core (not ported in the
/// parser).
const PLAIN: &[&str] = &[
    "Acos",
    "Acosh",
    "Acot",
    "Acoth",
    "Acsc",
    "Acsch",
    "Adjoint",
    "AiryAi",
    "AiryAiPrime",
    "AiryBi",
    "AiryBiPrime",
    "Andre",
    "AppellF1",
    "Arg",
    "Asec",
    "Asech",
    "Asin",
    "Asinh",
    "AssocLaguerre",
    "AssocLegendre",
    "Atan",
    "Atan2",
    "Atanh",
    "Bell",
    "Bernoulli",
    "BesselI",
    "BesselJ",
    "BesselK",
    "BesselY",
    "Beta",
    "BetaInc",
    "BetaIncRegularized",
    "Binomial",
    "Catalan",
    "Ceiling",
    "ChebyshevT",
    "ChebyshevU",
    "ChiNode",
    "CiNode",
    "Conjugate",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "DiGamma",
    "DirichletEta",
    "DivisorSigma",
    "E1",
    "EiNode",
    "Eijk",
    "EllipticE",
    "EllipticF",
    "EllipticK",
    "EllipticPi",
    "Erf",
    "Erf2",
    "Erf2inv",
    "Erfc",
    "Erfcinv",
    "Erfi",
    "Erfinv",
    "Euler",
    "Exp",
    "ExpPolar",
    "Expint",
    "ExprCondPair",
    "Factorial",
    "Factorial2",
    "Fibonacci",
    "Floor",
    "Frac",
    "FresnelC",
    "FresnelS",
    "Gamma",
    "Gegenbauer",
    "Genocchi",
    "Hankel1",
    "Hankel2",
    "Harmonic",
    "Hermite",
    "HermiteProb",
    "Hyper",
    "Id",
    "Im",
    "Jacobi",
    "JacobiSymbol",
    "KroneckerDeltaNode",
    "KroneckerSymbol",
    "Laguerre",
    "LambertW",
    "Legendre",
    "LegendreSymbol",
    "LerchPhi",
    "LeviCivitaNode",
    "LiNode",
    "Log",
    "LogGamma",
    "LowerGamma",
    "Lucas",
    "MarcumQ",
    "MathieuC",
    "MathieuCPrime",
    "MathieuS",
    "MathieuSPrime",
    "MeijerG",
    "Mobius",
    "Motzkin",
    "MultiGamma",
    "OffsetLiNode",
    "Partition",
    "PeriodicArgument",
    "Piecewise",
    "PolarLift",
    "PolyGamma",
    "PolyLog",
    "PrimeNu",
    "PrimeOmega",
    "PrimePi",
    "PrincipalBranch",
    "Re",
    "ReducedTotient",
    "Rem",
    "RiemannXi",
    "Sec",
    "Sech",
    "ShiNode",
    "SiNode",
    "Sign",
    "Sin",
    "Sinc",
    "SingularityFunctionNode",
    "Sinh",
    "SphericalHankel1",
    "SphericalHankel2",
    "SphericalJ",
    "SphericalY",
    "Stieltjes",
    "Subfactorial",
    "Tan",
    "Tanh",
    "Totient",
    "Transpose",
    "TriGamma",
    "Tribonacci",
    "UDivisorSigma",
    "UpperGamma",
    "YnmNode",
    "Ynm_c",
    "Zeta",
    "ZnmNode",
    "_eis",
    "_erfs",
    "abs_",
    "adjoint",
    "bspline_basis",
    "bspline_basis_set",
    "chebyshevt_root",
    "chebyshevu_root",
    "dirac_delta",
    "exp_polar",
    "ff",
    "heaviside",
    "hyper",
    "identity_function",
    "jacobi_normalized",
    "jn_zeros",
    "lambertw",
    "max_",
    "meijerg",
    "min_",
    "periodic_argument",
    "piecewise",
    "piecewise_fold",
    "polar_lift",
    "polarify",
    "principal_branch",
    "real_root",
    "rem",
    "rf",
    "sinc",
    "transpose",
    "unbranched_argument",
    "unpolarify",
];

fn number(text: &str) -> PolyResult<Basic> {
    if text.chars().all(|c| c.is_ascii_digit()) {
        return Ok(Basic::integer(
            text.parse::<Int>().map_err(|_| unsupported(text.to_string()))?,
        ));
    }
    Err(PolyError::NotImplemented(format!(
        "the Float {} (no Float in the Rust core)",
        text
    )))
}

fn named(text: &str) -> Basic {
    match text {
        "pi" => Basic::pi(),
        "E" => Basic::e(),
        "I" => Basic::imaginary_unit(),
        "oo" => Basic::oo(),
        "zoo" => Basic::zoo(),
        "nan" => Basic::nan(),
        "EulerGamma" => Basic::euler_gamma(),
        "Catalan" => Basic::catalan(),
        "GoldenRatio" => Basic::golden_ratio(),
        "TribonacciConstant" => Basic::tribonacci_constant(),
        _ => Symbol::new(text, &[]).0,
    }
}

/// The expression of `text` (`parse`): `ParseUnsupported` (a `Value`
/// error) outside the grammar.
pub fn parse(text: &str, evaluate: bool) -> PolyResult<Basic> {
    Parser {
        tokens: tokens(text)?,
        position: 0,
        evaluate,
        text,
    }
    .parse()
}
