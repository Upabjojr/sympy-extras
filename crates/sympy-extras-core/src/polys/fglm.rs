//! Port of `sympy_extras/polys/fglm.py`: change of monomial order for
//! zero-dimensional ideals, the FGLM algorithm on the typed polynomial rings.
//!
//! The monomials are visited in increasing order for the new order, starting
//! from 1; the normal form (for the old basis) of each comes from that of
//! the monomial it is a multiple of; when it is a linear combination of the
//! normal forms of the monomials kept so far, the relation is an element of
//! the new basis, otherwise the monomial is kept and its multiples by the
//! variables become candidates. Multiples of the leading monomials found are
//! skipped. [`groebner_by_fglm`] computes a basis for any order: F4 for
//! `grevlex`, then the conversion.
//!
//! References
//! ==========
//!
//! 1. J.-C. Faugère, P. Gianni, D. Lazard, T. Mora, *Efficient computation of
//!    zero-dimensional Gröbner bases by change of ordering*, Journal of
//!    Symbolic Computation 16 (1993), 329-344.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{value_error, PolyResult};

use super::domains::Domain;
use super::f4::f4;
use super::rings::{monomial_divides, Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

/// Whether the ideal of a Gröbner basis has dimension zero: a power of every
/// variable is a leading monomial, or the ideal is the whole ring.
pub fn is_zero_dimensional<K: Domain>(basis: &[PolyElement<K>], ring: &PolyRing<K>) -> bool {
    let mut pure: Vec<bool> = vec![false; ring.ngens];
    for g in basis {
        if g.is_zero() {
            continue;
        }
        let leading: Monomial = g.LM();
        let degree: u32 = leading.iter().sum();
        if degree == 0 {
            return true;
        }
        for (i, e) in leading.iter().enumerate() {
            if *e == degree {
                pure[i] = true;
            }
        }
    }
    pure.iter().all(|p| *p)
}

/// A row of the echelon form: a normal form with coefficient one at its
/// `pivot` monomial, and its expression `sum combination[j] NF(kept[j])` in
/// the normal forms of the monomials kept.
struct Row<T> {
    pivot: Monomial,
    vector: Terms<T>,
    combination: BTreeMap<usize, T>,
}

/// `terms[key] += value`, the entry removed when it becomes zero.
fn accumulate<K: Domain, M: Ord + Clone>(terms: &mut BTreeMap<M, K::Elem>, key: &M, value: K::Elem, k: &K) {
    let total: K::Elem = match terms.get(key) {
        Some(previous) => k.add(previous, &value),
        None => value,
    };
    if k.is_zero(&total) {
        terms.remove(key);
    } else {
        terms.insert(key.clone(), total);
    }
}

/// The reduced Gröbner basis for `order` of the zero-dimensional ideal whose
/// reduced Gröbner basis for the order of `ring` is `basis` (over a field);
/// in the ring with the new order, sorted by decreasing leading monomial.
/// `ValueError` when the ideal is not zero-dimensional.
///
/// The pivot of a new row is its largest monomial for the order of `ring`
/// (Python takes the first key of the dict, which the eliminations reorder):
/// the reduced basis does not depend on it.
pub fn fglm<K: Domain>(
    basis: &[PolyElement<K>],
    ring: &Arc<PolyRing<K>>,
    order: &MonomialOrder,
) -> PolyResult<Vec<PolyElement<K>>> {
    let k: &K = &ring.domain;
    let target: Arc<PolyRing<K>> = ring.with_order(order.clone());
    let generators: Vec<PolyElement<K>> = basis.iter().filter(|g| !g.is_zero()).cloned().collect();
    if !is_zero_dimensional(&generators, ring) {
        return Err(value_error("the ideal is not zero-dimensional".to_string()));
    }
    let n: usize = ring.ngens;
    let units: Vec<Monomial> = (0..n).map(|i| (0..n).map(|t| u32::from(t == i)).collect()).collect();
    let origin: Monomial = vec![0; n];
    let mut kept: Vec<Monomial> = Vec::new();
    let mut rows: Vec<Row<K::Elem>> = Vec::new();
    let mut leading: Vec<Monomial> = Vec::new();
    let mut result: Vec<PolyElement<K>> = Vec::new();
    // the candidates, with the normal form each one is computed from
    let mut candidates: BTreeMap<Monomial, PolyElement<K>> = BTreeMap::new();
    candidates.insert(origin, ring.one().rem(&generators)?);
    while !candidates.is_empty() {
        let monomial: Monomial = candidates
            .keys()
            .min_by(|a, b| order.cmp(a, b))
            .expect("a candidate")
            .clone();
        let normal_form: PolyElement<K> = candidates.remove(&monomial).expect("a candidate");
        if leading.iter().any(|lm| monomial_divides(lm, &monomial)) {
            continue;
        }
        let mut vector: Terms<K::Elem> = normal_form.terms.clone();
        let mut combination: BTreeMap<usize, K::Elem> = BTreeMap::new();
        for row in &rows {
            let Some(c) = vector.get(&row.pivot).cloned() else {
                continue;
            };
            for (m, v) in &row.vector {
                accumulate(&mut vector, m, k.neg(&k.mul(&c, v)), k);
            }
            for (j, v) in &row.combination {
                accumulate(&mut combination, j, k.mul(&c, v), k);
            }
        }
        if vector.is_empty() {
            // NF(monomial) = sum combination[j] NF(kept[j]): a basis element
            let mut terms: Terms<K::Elem> = Terms::new();
            terms.insert(monomial.clone(), k.one());
            for (j, v) in &combination {
                terms.insert(kept[*j].clone(), k.neg(v));
            }
            result.push(target.from_dict(terms));
            leading.push(monomial);
            continue;
        }
        let index: usize = kept.len();
        kept.push(monomial.clone());
        let pivot: Monomial = vector
            .keys()
            .max_by(|a, b| ring.order.cmp(a, b))
            .expect("a nonzero vector")
            .clone();
        let inverse: K::Elem = k.invert(&vector[&pivot])?;
        let mut scaled: BTreeMap<usize, K::Elem> = combination
            .iter()
            .map(|(j, v)| (*j, k.neg(&k.mul(v, &inverse))))
            .collect();
        scaled.insert(index, inverse.clone());
        rows.push(Row {
            pivot,
            vector: vector.iter().map(|(m, v)| (m.clone(), k.mul(v, &inverse))).collect(),
            combination: scaled,
        });
        for unit in &units {
            let multiple: Monomial = monomial.iter().zip(unit).map(|(a, e)| a + e).collect();
            if !candidates.contains_key(&multiple) && !leading.iter().any(|lm| monomial_divides(lm, &multiple)) {
                let form: PolyElement<K> = normal_form.mul_monom(unit).rem(&generators)?;
                candidates.insert(multiple, form);
            }
        }
    }
    result.sort_by(|a, b| order.cmp(&b.LM(), &a.LM()));
    Ok(result)
}

/// The reduced Gröbner basis for the order of `ring` of a zero-dimensional
/// ideal over a field: F4 for `grevlex`, then the change of order.
pub fn groebner_by_fglm<K: Domain>(
    polys: &[PolyElement<K>],
    ring: &Arc<PolyRing<K>>,
) -> PolyResult<Vec<PolyElement<K>>> {
    let graded: Arc<PolyRing<K>> = ring.with_order(MonomialOrder::Grevlex);
    let converted: Vec<PolyElement<K>> = polys.iter().map(|f| graded.from_dict(f.terms.clone())).collect();
    fglm(&f4(&converted, &graded)?, &graded, &ring.order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::{FiniteField, QQ};

    #[test]
    fn docstring_examples() {
        let r = PolyRing::new(QQ, vec!["x".into(), "y".into()], MonomialOrder::Grevlex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let basis = f4(
            &[
                x.mul(x).add(&y.mul(y)).sub(&r.from_int(5)),
                x.mul(y).sub(&r.from_int(2)),
            ],
            &r,
        )
        .unwrap();
        let converted = fglm(&basis, &r, &MonomialOrder::Lex).unwrap();
        let shown: Vec<String> = converted.iter().map(|g| g.to_string()).collect();
        assert_eq!(shown, vec!["x + 1/2*y**3 - 5/2*y", "y**4 - 5*y**2 + 4"]);
        let gf = PolyRing::new(
            FiniteField::new(7).unwrap(),
            vec!["x".into(), "y".into()],
            MonomialOrder::Grevlex,
        )
        .unwrap();
        let h = gf.gens();
        let (x, y) = (&h[0], &h[1]);
        let converted = fglm(&[y.mul(y).sub(x), x.mul(x).sub(&gf.one())], &gf, &MonomialOrder::Lex).unwrap();
        let shown: Vec<String> = converted.iter().map(|g| g.to_string()).collect();
        assert_eq!(shown, vec!["x + 6*y**2", "y**4 + 6"]);
        assert!(fglm(&[x.mul(y)], &gf, &MonomialOrder::Lex).is_err());
    }
}
