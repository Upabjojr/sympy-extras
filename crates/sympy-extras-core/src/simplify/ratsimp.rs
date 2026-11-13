//! Port of `sympy_extras/simplify/ratsimp.py`: [`ratsimp`], a rational
//! function as a polynomial plus a proper fraction (the numerator divided
//! by the denominator in the lexicographic order of the generators, the
//! remainder over the denominator cancelled). The domains `ZZ_I`, `RR`
//! and `EX`, which the Python hands to SymPy, are `NotImplemented`.

// PORTED FROM SYMPY: sympy/simplify/ratsimp.py (ratsimp), sympy/polys/rings.py
// (PolyElement.div, one divisor over a field) (SymPy 1.14), through
// sympy_extras; BSD 3-clause license, see sympy_extras/LICENSE-SymPy.

use std::collections::BTreeMap;

use crate::core::add::add;
use crate::core::basic::Basic;
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::{not_ported, PolyError, PolyResult, PolysErrorKind};
use crate::numbers::*;
use crate::polys::polytools::{cancel, joint_rings};
use crate::polys::rings::Monomial;

type Terms = BTreeMap<Monomial, Rat>;

/// The quotient and the remainder of `f` by `g` in the lexicographic order
/// (`PolyElement.div` with one divisor over a field).
fn divide(f: &Terms, g: &Terms) -> (Terms, Terms) {
    let (lg, cg) = g
        .iter()
        .next_back()
        .map(|(m, c)| (m.clone(), c.clone()))
        .expect("a nonzero divisor");
    let mut p: Terms = f.clone();
    let mut q: Terms = Terms::new();
    let mut r: Terms = Terms::new();
    while let Some((lp, cp)) = p.iter().next_back().map(|(m, c)| (m.clone(), c.clone())) {
        if lp.iter().zip(&lg).all(|(a, b)| a >= b) {
            let m: Monomial = lp.iter().zip(&lg).map(|(a, b)| a - b).collect();
            let c: Rat = &cp / &cg;
            let total: Rat = q.remove(&m).unwrap_or(Rat::ZERO) + &c;
            if !total.is_zero() {
                q.insert(m.clone(), total);
            }
            for (mg, coefficient) in g {
                let k: Monomial = m.iter().zip(mg).map(|(a, b)| a + b).collect();
                let value: Rat = p.remove(&k).unwrap_or(Rat::ZERO) - &c * coefficient;
                if !value.is_zero() {
                    p.insert(k, value);
                }
            }
        } else {
            let total: Rat = r.remove(&lp).unwrap_or(Rat::ZERO) + &cp;
            r.insert(lp.clone(), total);
            p.remove(&lp);
        }
    }
    (q, r)
}

fn expr_of(terms: &Terms, gens: &[Basic]) -> PolyResult<Basic> {
    let mut parts: Vec<Basic> = Vec::new();
    for (m, c) in terms.iter().rev() {
        let mut factors: Vec<Basic> = vec![Basic::from_rational(c.clone())];
        for (g, k) in gens.iter().zip(m) {
            if *k > 0 {
                factors.push(pow(g, &Basic::int(i64::from(*k)))?);
            }
        }
        parts.push(mul(&factors)?);
    }
    add(&parts)
}

/// The cancelled expression as the quotient of its numerator by its
/// denominator plus the cancelled remainder over the denominator.
pub fn ratsimp(e: &Basic) -> PolyResult<Basic> {
    let (f, g) = crate::core::expr::as_numer_denom(&cancel(e, &[])?)?;
    let (gens, reps) = match joint_rings(&[f.clone(), g.clone()], false) {
        Ok(r) => r,
        Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
            return Err(not_ported("ratsimp over ZZ_I, RR or EX"))
        }
        Err(other) => return Err(other),
    };
    let inverse: Basic = pow(&g, &Basic::negative_one())?;
    if gens.is_empty() {
        return mul(&[f, inverse]);
    }
    let (q, r) = divide(&reps[0], &reps[1]);
    add(&[
        expr_of(&q, &gens)?,
        cancel(&mul(&[expr_of(&r, &gens)?, inverse])?, &[])?,
    ])
}
