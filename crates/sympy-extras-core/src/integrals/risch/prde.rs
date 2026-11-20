//! Port of `sympy_extras/integrals/risch/prde.py` (the parametric Risch
//! differential equation and the structure theorems): not ported yet,
//! every entry point reports it.

use super::risch::DifferentialExtension;
use super::tower::TowerPoly;
use crate::core::basic::Basic;
use crate::error::PolyResult;

/// `(ans, u, const)` with `Dg/g = Du + const`-style decomposition of `a/d`
/// as a derivative in the tower (`is_deriv_k`).
pub fn is_deriv_k(
    _a: &TowerPoly,
    _d: &TowerPoly,
    _de: &DifferentialExtension,
) -> PolyResult<Option<(Vec<(Basic, Basic)>, Basic, Basic)>> {
    Err(crate::error::not_ported("integrals/risch/prde (is_deriv_k)"))
}

/// `(ans, u, n, const)` for a logarithmic derivative of a radical
/// (`is_log_deriv_k_t_radical`).
pub fn is_log_deriv_k_t_radical(
    _a: &TowerPoly,
    _d: &TowerPoly,
    _de: &DifferentialExtension,
) -> PolyResult<Option<(Vec<(Basic, Basic)>, Basic, i64, Basic)>> {
    Err(crate::error::not_ported(
        "integrals/risch/prde (is_log_deriv_k_t_radical)",
    ))
}

/// The tangent analogue (`is_log_deriv_k_t_radical_tan`).
pub fn is_log_deriv_k_t_radical_tan(
    _a: &TowerPoly,
    _d: &TowerPoly,
    _de: &DifferentialExtension,
) -> PolyResult<Option<(Vec<(Basic, Basic)>, Basic, i64, Basic)>> {
    Err(crate::error::not_ported(
        "integrals/risch/prde (is_log_deriv_k_t_radical_tan)",
    ))
}

/// The arc-tangent analogue (`is_deriv_k_atan`).
pub fn is_deriv_k_atan(
    _a: &TowerPoly,
    _d: &TowerPoly,
    _de: &DifferentialExtension,
) -> PolyResult<Option<(Vec<(Basic, Basic)>, Basic)>> {
    Err(crate::error::not_ported("integrals/risch/prde (is_deriv_k_atan)"))
}

/// `((ba, bd), c)` with `f = Db + sum c_i w_i` (`limited_integrate`).
pub fn limited_integrate(
    _fa: &TowerPoly,
    _fd: &TowerPoly,
    _g: &[(TowerPoly, TowerPoly)],
    _de: &mut DifferentialExtension,
) -> PolyResult<Option<((TowerPoly, TowerPoly), Vec<super::tower::RationalFunction>)>> {
    Err(crate::error::not_ported("integrals/risch/prde (limited_integrate)"))
}
