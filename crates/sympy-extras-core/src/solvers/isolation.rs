//! Port of `sympy_extras/solvers/isolation.py` (the first part): the
//! comparison of real points and a rational point between two of them,
//! for exact constants and infinities (the root objects
//! `TranscendentalRoot` come with the root isolation).

use crate::core::basic::Basic;
use crate::error::PolyResult;

/// The sign of `u - v` for exact real constants and infinities
/// (`compare_points`).
pub fn compare_points(u: &Basic, v: &Basic) -> PolyResult<Option<i32>> {
    crate::assumptions::analysis::certified_compare(u, v)
}

/// A rational number strictly between `p < q` (`point_between`).
pub fn point_between(p: &Basic, q: &Basic) -> PolyResult<Option<Basic>> {
    // the ends of constants do not move: separated at once or never
    if crate::assumptions::analysis::certified_compare(p, q)? != Some(-1) {
        return Ok(None);
    }
    let middle: Option<Basic> = crate::assumptions::analysis::between(p, q)?;
    Ok(middle.filter(|m| m.is_rational_class()))
}
