//! `sympy_extras._timeout` without the clock: [`attempt`] runs a
//! computation and gives up on the errors the Python gives up on
//! (`NotImplementedError`, `ValueError`, `TypeError`, `RecursionError`,
//! the polynomial errors and `ArithmeticError`, here every [`PolyError`]
//! but `Index`), returning `None`.
//!
//! The Python's time limits (an alarm signal checked by the interpreter)
//! are not ported: the Rust computations run to their end. The rest of the
//! module prepares SymPy (its tables, its `manualintegrate`), which the
//! Rust code does not use.

use crate::error::{PolyError, PolyResult};

/// The value of `f()`, or `None` when it gives up (`attempt`; no time
/// limit).
pub fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    match f() {
        Ok(v) => Ok(Some(v)),
        Err(PolyError::Index(message)) => Err(PolyError::Index(message)),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gives_up() {
        assert_eq!(attempt(|| Ok(3)).unwrap(), Some(3));
        let failed: PolyResult<i32> = Err(PolyError::ZeroDivision("1/0".to_string()));
        assert_eq!(attempt(|| failed).unwrap(), None);
        let index: PolyResult<i32> = Err(PolyError::Index("out".to_string()));
        assert!(attempt(|| index).is_err());
    }
}
