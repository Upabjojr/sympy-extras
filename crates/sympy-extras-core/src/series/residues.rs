//! Port of `sympy_extras/series/residues.py` (SymPy's `residue`): the
//! coefficient of `1/(x - x0)` in the Laurent series.

use crate::core::assumptions::Predicate;
use crate::core::basic::Basic;
use crate::core::expr::{is, terms_of};
use crate::error::{PolyError, PolyResult};

fn coeff_mul(e: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    if e.is_mul() {
        let free: Vec<Basic> = e.args().iter().filter(|f| !f.has(x)).cloned().collect();
        let dep: Vec<Basic> = e.args().iter().filter(|f| f.has(x)).cloned().collect();
        return Ok((crate::core::mul::mul(&free)?, crate::core::mul::mul(&dep)?));
    }
    if e.has(x) {
        return Ok((Basic::one(), e.clone()));
    }
    Ok((e.clone(), Basic::one()))
}

/// The residue of `expr` at `x = x0` (`residue`; `NotImplemented` for a
/// series of another form).
pub fn residue(expr: &Basic, x: &Basic, x0: &Basic) -> PolyResult<Basic> {
    let mut e: Basic = expr.clone();
    if *x0 != Basic::zero() {
        e = crate::core::subs::subs(&e, x, &crate::core::add::add(&[x.clone(), x0.clone()])?)?;
    }
    let mut s: Basic = e.clone();
    for n in [0i64, 1, 2, 4, 8, 16, 32] {
        s = crate::core::series::nseries(&e, x, n, None, &Basic::zero())?;
        let order: Option<Basic> = crate::core::order::getn(&s)?;
        let has_order: bool = s.preorder().iter().any(crate::core::series::is_order);
        if !has_order
            || order
                .as_ref()
                .is_some_and(|o| is(o, Predicate::nonnegative) == Some(true))
        {
            break;
        }
    }
    let collected: Basic = crate::simplify::radsimp::collect(
        &crate::core::order::remove_o(&s)?,
        std::slice::from_ref(x),
        None,
        Some(false),
    )?;
    let inverse: Basic = crate::core::power::pow(x, &Basic::negative_one())?;
    let mut res: Basic = Basic::zero();
    for arg in terms_of(&collected) {
        let (c, m) = coeff_mul(&arg, x)?;
        let integer_power: bool = m.is_pow() && m.args()[1].is_integer_value() && m.args()[1].is_number_class();
        if !(m == Basic::one() || m == *x || integer_power) {
            return Err(PolyError::NotImplemented(format!("term of unexpected form: {}", m)));
        }
        if m == inverse {
            res = crate::core::add::add(&[res, c])?;
        }
    }
    Ok(res)
}
