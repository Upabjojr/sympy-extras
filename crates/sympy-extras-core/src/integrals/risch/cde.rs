//! Port of `sympy_extras/integrals/risch/cde.py` (the coupled
//! differential system): not ported yet.

use super::risch::DifferentialExtension;
use super::tower::TowerPoly;
use crate::error::PolyResult;

type Fraction2 = (TowerPoly, TowerPoly);

/// A solution `((y1a, y1d), (y2a, y2d))` of the coupled system
/// (`coupled_DE_system`).
pub fn coupled_de_system(
    _f1: &Fraction2,
    _f2: &Fraction2,
    _g1: &Fraction2,
    _g2: &Fraction2,
    _de: &mut DifferentialExtension,
) -> PolyResult<(Fraction2, Fraction2)> {
    Err(crate::error::not_ported("integrals/risch/cde (coupled_DE_system)"))
}
