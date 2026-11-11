//! The errors of the algorithms: the Python exceptions which depend on the
//! data.

use std::fmt;

/// An error of a computation, named after the Python exception it stands
/// for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolyError {
    /// `ZeroDivisionError`: a division by zero.
    ZeroDivision(String),
    /// `ValueError`: a division which is not exact, an element which is not
    /// a unit, a conversion without an exact image, an invalid argument.
    Value(String),
    /// `polyerrors.HeuristicGCDFailed`: no evaluation point gave the gcd.
    HeuristicGCDFailed(String),
    /// `ArithmeticError`: an internal invariant of an exact computation
    /// failed (a bug, not a property of the data).
    Arithmetic(String),
    /// `NotImplementedError`: a case which needs a part of the Python core
    /// which is not ported yet (a function class, `Float`, `Order`).
    NotImplemented(String),
    /// One of the exceptions of `polys.polyerrors` (`PolynomialError`,
    /// `GeneratorsNeeded`, `CoercionFailed`, ...).
    Polys(PolysErrorKind, String),
}

/// The classes of `polys.polyerrors` which the ported code raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolysErrorKind {
    PolynomialError,
    GeneratorsError,
    GeneratorsNeeded,
    CoercionFailed,
    ExactQuotientFailed,
    OperationNotSupported,
}

impl PolysErrorKind {
    /// The name of the Python class.
    pub fn name(self) -> &'static str {
        match self {
            PolysErrorKind::PolynomialError => "PolynomialError",
            PolysErrorKind::GeneratorsError => "GeneratorsError",
            PolysErrorKind::GeneratorsNeeded => "GeneratorsNeeded",
            PolysErrorKind::CoercionFailed => "CoercionFailed",
            PolysErrorKind::ExactQuotientFailed => "ExactQuotientFailed",
            PolysErrorKind::OperationNotSupported => "OperationNotSupported",
        }
    }
}

/// The exception `kind(message)` of `polys.polyerrors`.
pub fn polys_error(kind: PolysErrorKind, message: String) -> PolyError {
    PolyError::Polys(kind, message)
}

/// The result of a computation which may fail.
pub type PolyResult<T> = Result<T, PolyError>;

impl fmt::Display for PolyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolyError::ZeroDivision(message) => write!(f, "{}", message),
            PolyError::Value(message) => write!(f, "{}", message),
            PolyError::HeuristicGCDFailed(message) => write!(f, "{}", message),
            PolyError::Arithmetic(message) => write!(f, "{}", message),
            PolyError::Polys(_, message) => write!(f, "{}", message),
            PolyError::NotImplemented(message) => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for PolyError {}

/// `ZeroDivisionError(message)`.
pub fn zero_division(message: &str) -> PolyError {
    PolyError::ZeroDivision(message.to_string())
}

/// `ValueError(message)`.
pub fn value_error(message: String) -> PolyError {
    PolyError::Value(message)
}

/// `ArithmeticError(message)`.
pub fn arithmetic_error(message: String) -> PolyError {
    PolyError::Arithmetic(message)
}

/// `NotImplementedError(message)`: a part of the Python core not ported yet.
pub fn not_ported(message: &str) -> PolyError {
    PolyError::NotImplemented(format!("not ported to the Rust core yet: {}", message))
}
