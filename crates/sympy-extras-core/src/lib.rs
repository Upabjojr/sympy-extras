//! Pure Rust port of `sympy_extras`.
//!
//! The modules mirror the Python package one to one
//! (`sympy_extras.polys.densearith` is [`polys::densearith`], with the same
//! function names). This crate has no Python dependency; the PyO3 bindings
//! live in the `sympy-extras-py` crate (Python module `sympy_extras_rs`).
//!
//! # Conventions of the port
//!
//! * A Python `int` that is a number of the computation is an [`Int`]; a
//!   `Fraction` is a [`Rat`] (both from `dashu`, chosen in [`numbers`]);
//!   a degree, an index
//!   or a count is a `usize` (an `isize` where Python uses `-1` for the
//!   zero polynomial).
//! * A Python exception that depends on the data (`ZeroDivisionError`,
//!   `ValueError`: a division which is not exact, a non-invertible
//!   element) is an `Err` of [`error::PolyError`]; a `NotImplementedError`
//!   for an operation the domain does not have (`is_negative` in
//!   `GF(p)`) is a panic, a programming error as it is in Python.
//! * Arguments that Python rejects when negative (orders of
//!   differentiation, exponents of polynomials) are `usize`.
//!
//! [`Int`]: numbers::Int
//! [`Rat`]: numbers::Rat

// the names and signatures follow the Python package (`from_int` takes the
// domain, results are tuples of polynomials)
#![allow(clippy::wrong_self_convention, clippy::type_complexity)]

pub mod core;
pub mod error;
pub mod functions;
pub mod ntheory;
pub mod numbers;
pub mod orderedmap;
pub mod polys;
pub mod random;
pub mod simplify;
pub mod solvers;

/// Version of the crate, kept equal to `sympy_extras.__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;
    #[allow(unused_imports)]
    use crate::numbers::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
