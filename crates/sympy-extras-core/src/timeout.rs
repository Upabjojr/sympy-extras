//! `sympy_extras._timeout`: [`attempt`] runs a computation and gives up
//! on the errors the Python gives up on (`NotImplementedError`,
//! `ValueError`, `TypeError`, `RecursionError`, the polynomial errors and
//! `ArithmeticError`, here every [`PolyError`] but `Index` and
//! `TimeLimit`), returning `None`; [`attempt_within`] does so within a
//! time limit too (`attempt(f, seconds)`).
//!
//! The limits are cooperative: a computation which may run long calls
//! [`check`] in its loops (the lifting of the cylindrical algebraic
//! decomposition, its projection), which fails with
//! [`PolyError::TimeLimit`] once a limit in force has expired. As in the
//! Python an enclosing limit which expires is not the time of the inner
//! computation being up: its error goes on to the `attempt` which set it.
//! The limits are multiplied by the `time_scale` of the settings. The rest
//! of the Python module prepares SymPy, which the Rust code does not use.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use crate::error::{PolyError, PolyResult};

thread_local! {
    /// The deadlines in force, the outermost first.
    static DEADLINES: RefCell<Vec<Instant>> = const { RefCell::new(Vec::new()) };
}

/// `TimeLimitExceeded` when a limit in force has expired (the check of
/// the Python's alarm).
pub fn check() -> PolyResult<()> {
    let now: Instant = Instant::now();
    if DEADLINES.with(|d| d.borrow().iter().any(|t| now >= *t)) {
        return Err(PolyError::TimeLimit("the time limit was exceeded".to_string()));
    }
    Ok(())
}

/// The seconds left of the innermost limit in force, in the units of the
/// settings (divided by `time_scale`; a little rather than nothing once
/// it has expired), `None` without a limit (`remaining_time`).
pub fn remaining_time() -> Option<f64> {
    let end: Instant = DEADLINES.with(|d| d.borrow().last().copied())?;
    let left: f64 = end.saturating_duration_since(Instant::now()).as_secs_f64().max(1e-6);
    Some(left / crate::settings::settings().time_scale)
}

/// The payload of the unwinding of [`check_infallible`].
struct TimeLimitUnwind;

/// [`check`] for the routines which cannot fail (the dense resultants):
/// the expiry unwinds to the [`attempt_within`] which set the limit, as
/// the Python's alarm interrupts any computation. `resume_unwind` runs no
/// panic hook; nothing is printed.
pub fn check_infallible() {
    let any_deadline: bool = DEADLINES.with(|d| !d.borrow().is_empty());
    if any_deadline && check().is_err() {
        std::panic::resume_unwind(Box::new(TimeLimitUnwind));
    }
}

/// The errors Python's `attempt` takes as giving up; a part of the port
/// which is missing (`not_ported`) is no Python error and goes through, so
/// that it is reported instead of a different answer.
fn gives_up(e: &PolyError) -> bool {
    if let PolyError::NotImplemented(m) = e {
        if m.starts_with(crate::error::NOT_PORTED_PREFIX) {
            return false;
        }
    }
    !matches!(e, PolyError::Index(_) | PolyError::TimeLimit(_))
}

/// The value of `f()`, or `None` when it gives up (`attempt` without a
/// time limit).
pub fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    match f() {
        Ok(v) => Ok(Some(v)),
        Err(e) if gives_up(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// The value of `f()` computed within `seconds` (scaled by the settings'
/// `time_scale`), or `None` when the time is up or it gives up
/// (`attempt(f, seconds)`); `seconds` `None` is no limit.
pub fn attempt_within<T>(seconds: Option<f64>, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    let Some(s) = seconds else { return attempt(f) };
    let scaled: f64 = (s * crate::settings::settings().time_scale).max(0.0);
    let deadline: Instant = Instant::now() + Duration::from_secs_f64(scaled);
    let depth: usize = DEADLINES.with(|d| {
        let mut d = d.borrow_mut();
        d.push(deadline);
        d.len() - 1
    });
    let result: PolyResult<T> = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) if payload.is::<TimeLimitUnwind>() => {
            Err(PolyError::TimeLimit("the time limit was exceeded".to_string()))
        }
        Err(payload) => {
            DEADLINES.with(|d| d.borrow_mut().truncate(depth));
            std::panic::resume_unwind(payload)
        }
    };
    let enclosing_expired: bool = DEADLINES.with(|d| {
        let mut d = d.borrow_mut();
        d.truncate(depth);
        let now: Instant = Instant::now();
        d.iter().any(|t| now >= *t)
    });
    match result {
        Ok(v) => Ok(Some(v)),
        Err(PolyError::TimeLimit(m)) => {
            if !enclosing_expired && Instant::now() >= deadline {
                Ok(None)
            } else {
                Err(PolyError::TimeLimit(m))
            }
        }
        Err(e) if gives_up(&e) => Ok(None),
        Err(e) => Err(e),
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

    #[test]
    fn time_limits() {
        let slow = || -> PolyResult<i32> {
            loop {
                check()?;
            }
        };
        assert_eq!(attempt_within(Some(0.05), slow).unwrap(), None);
        // the inner limit expires first: the outer computation goes on
        let outer = attempt_within(Some(5.0), || Ok((attempt_within(Some(0.05), slow)?, "went on"))).unwrap();
        assert_eq!(outer, Some((None, "went on")));
        // an expiry inside a routine which cannot fail unwinds to the attempt
        let infallible = || -> PolyResult<i32> {
            loop {
                check_infallible();
            }
        };
        assert_eq!(attempt_within(Some(0.05), infallible).unwrap(), None);
        // the outer limit expires: the inner attempt does not swallow it
        let inner_long = attempt_within(Some(0.05), || attempt_within(Some(5.0), slow)).unwrap();
        assert_eq!(inner_long, None);
    }
}
