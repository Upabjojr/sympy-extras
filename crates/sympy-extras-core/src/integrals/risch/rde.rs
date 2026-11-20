//! Port of `sympy_extras/integrals/risch/rde.py` (the Risch differential
//! equation): the rational roots of the coefficients so far; the rest
//! follows.

use std::collections::BTreeMap;

use super::tower::{RationalFunction, TowerPoly};
use crate::numbers::{Int, Rat};

/// The rational roots of `p` with their multiplicities for every value of
/// the parameters of its coefficients (`_rational_roots`).
pub fn rational_roots(p: &TowerPoly) -> crate::error::PolyResult<Vec<RationalFunction>> {
    use crate::polys::domains::integerring::ZZ;
    let a: RationalFunction = p.as_element()?;
    let gen = &p.gen;
    if a.is_zero() || !a.gens.contains(gen) {
        return Ok(Vec::new());
    }
    let i: usize = a.gens.iter().position(|g| g == gen).expect("a generator");
    let mut components: Vec<(Vec<u32>, BTreeMap<u32, Int>)> = Vec::new();
    for (m, c) in a.numer.terms.iter() {
        let key: Vec<u32> = m.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, e)| *e).collect();
        match components.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => {
                entry.1.insert(m[i], c.clone());
            }
            None => {
                let mut terms: BTreeMap<u32, Int> = BTreeMap::new();
                terms.insert(m[i], c.clone());
                components.push((key, terms));
            }
        }
    }
    let mut g: Vec<Int> = Vec::new();
    for (_, terms) in &components {
        let n: u32 = *terms.keys().next_back().expect("a term");
        let dense: Vec<Int> = (0..=n)
            .rev()
            .map(|k| terms.get(&k).cloned().unwrap_or(Int::ZERO))
            .collect();
        g = if g.is_empty() {
            dense
        } else {
            crate::polys::euclidtools::dup_gcd(&g, &dense, &ZZ)
        };
        if g.len() <= 1 {
            return Ok(Vec::new());
        }
    }
    if g.len() <= 1 {
        return Ok(Vec::new());
    }
    let (_, factors) = crate::polys::factortools::dup_zz_factor(&g, &ZZ)?;
    let mut roots: Vec<RationalFunction> = Vec::new();
    for (f, k) in factors {
        if f.len() == 2 {
            let root: RationalFunction =
                RationalFunction::from_fraction(&(Rat::from(-f[1].clone()) / Rat::from(f[0].clone())));
            for _ in 0..k {
                roots.push(root.clone());
            }
        }
    }
    Ok(roots)
}

/// A solution `(ya, yd)` of `Dy + f y = g` (`rischDE`); the error of
/// `NonElementaryIntegralException` when there is none.
pub fn risch_de(
    _fa: &TowerPoly,
    _fd: &TowerPoly,
    _ga: &TowerPoly,
    _gd: &TowerPoly,
    _de: &mut super::risch::DifferentialExtension,
) -> crate::error::PolyResult<(TowerPoly, TowerPoly)> {
    Err(crate::error::not_ported("integrals/risch/rde (rischDE)"))
}

/// The order of `a` at `p` (`order_at`); `None` for `a == 0` (`oo`).
pub fn order_at(
    _a: &TowerPoly,
    _p: &TowerPoly,
    _t: &crate::core::basic::Basic,
) -> crate::error::PolyResult<Option<i64>> {
    Err(crate::error::not_ported("integrals/risch/rde (order_at)"))
}
