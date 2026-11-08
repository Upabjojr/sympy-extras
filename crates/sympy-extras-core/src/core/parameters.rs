//! Port of `sympy_extras/core/parameters.py`: the global parameters of the
//! evaluation, with scoped switches.
//!
//! `evaluate()` tells the evaluating functions (`add`, `mul`, `pow`, ...)
//! whether to evaluate; [`evaluate_guard`]`(false)` turns them into the
//! constructors until the guard is dropped (Python's `with evaluate(False)`).
//! [`distribute_guard`]`(false)` keeps a number in front of a sum
//! (`2*(x + y)` stays a product); `divide_error` makes a division by zero an
//! error instead of `zoo` (`seterr`).
//!
//! Python's `global_parameters` is one object of the interpreter; here the
//! switches are per thread, the scope of a `with` block.

use std::cell::Cell;

thread_local! {
    static EVALUATE: Cell<bool> = const { Cell::new(true) };
    static DISTRIBUTE: Cell<bool> = const { Cell::new(true) };
    static DIVIDE_ERROR: Cell<bool> = const { Cell::new(false) };
}

/// Whether the evaluating functions evaluate.
pub fn evaluate() -> bool {
    EVALUATE.with(|c| c.get())
}

/// Whether a number in front of a sum is distributed.
pub fn distribute() -> bool {
    DISTRIBUTE.with(|c| c.get())
}

/// Whether a division by zero is an error.
pub fn divide_error() -> bool {
    DIVIDE_ERROR.with(|c| c.get())
}

/// `seterr(divide=flag)`.
pub fn set_divide_error(flag: bool) {
    DIVIDE_ERROR.with(|c| c.set(flag));
}

/// Restores a switch when dropped.
pub struct Guard {
    previous: bool,
    restore: fn(bool),
}

impl Drop for Guard {
    fn drop(&mut self) {
        (self.restore)(self.previous);
    }
}

/// `with evaluate(flag):` until the guard is dropped.
pub fn evaluate_guard(flag: bool) -> Guard {
    let previous: bool = EVALUATE.with(|c| c.replace(flag));
    Guard {
        previous,
        restore: |v| EVALUATE.with(|c| c.set(v)),
    }
}

/// `with distribute(flag):` until the guard is dropped.
pub fn distribute_guard(flag: bool) -> Guard {
    let previous: bool = DISTRIBUTE.with(|c| c.replace(flag));
    Guard {
        previous,
        restore: |v| DISTRIBUTE.with(|c| c.set(v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_nest_and_restore() {
        assert!(evaluate() && distribute());
        {
            let _outer = evaluate_guard(false);
            assert!(!evaluate());
            {
                let _inner = evaluate_guard(true);
                assert!(evaluate());
            }
            assert!(!evaluate());
            let _d = distribute_guard(false);
            assert!(!distribute());
        }
        assert!(evaluate() && distribute());
    }
}
