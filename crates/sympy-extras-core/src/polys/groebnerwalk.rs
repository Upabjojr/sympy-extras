//! Port of `sympy_extras/polys/groebnerwalk.py`: the Gröbner walk, the
//! conversion of Gröbner bases between monomial orders for any ideal (FGLM
//! only applies to zero-dimensional ones).
//!
//! The source and target orders are represented by weight vectors `omega`
//! and `tau`, and the basis is transformed along the segment from `omega`
//! to `tau`. At every point where the ordering of the monomials of the
//! basis changes, the *initial forms* of the basis elements (their terms of
//! maximal weight) are computed, a Gröbner basis of the ideal they generate
//! is computed, and it is lifted to a Gröbner basis of the whole ideal for
//! the new weight: by normal forms modulo the previous basis (Amrhein,
//! Gloor, Küchlin: `h - NF_G(h)` has initial form `h`), or through the
//! representations of [`extended_groebner`].
//!
//! The Python module computes the bases of the initial forms with SymPy's
//! Buchberger algorithm; here they are computed by [`super::f4`], which
//! gives the same reduced basis. The domain must be a field.
//!
//! References
//! ==========
//!
//! 1. S. Collart, M. Kalkbrener, D. Mall, *Converting bases with the
//!    Gröbner walk*, J. Symbolic Computation 24 (1997) 465-469.
//! 2. B. Amrhein, O. Gloor, W. Küchlin, *On the walk*, Theoretical Computer
//!    Science 187 (1997) 179-202.

use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::domains::Domain;
use super::f4::{f4, reduced_basis};
use super::rings::{monomial_div, monomial_lcm, monomial_mul, Monomial, MonomialOrder, PolyElement, PolyRing};

/// A weight vector with rational entries.
pub type Weights = Vec<Rat>;

/// How the Gröbner bases of the initial forms are lifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkLift {
    /// `'normal_form'`: `h - NF_G(h)`, normal form for the previous order.
    NormalForm,
    /// `'cofactors'`: through the representations of [`extended_groebner`].
    Cofactors,
}

fn total_degree(m: &[u32]) -> u64 {
    m.iter().map(|e| *e as u64).sum()
}

fn dot(weights: &[Rat], m: &[i64]) -> Rat {
    let mut total: Rat = Rat::ZERO;
    for (w, e) in weights.iter().zip(m) {
        total += w * Rat::from(*e);
    }
    total
}

fn weight_of_monomial(weights: &[Rat], m: &[u32]) -> Rat {
    let exponents: Vec<i64> = m.iter().map(|e| *e as i64).collect();
    dot(weights, &exponents)
}

/// A weight vector in the closure of the Gröbner cone of `basis` for the
/// order: the degree vector for the graded orders, decreasing powers for
/// `lex`.
fn weight_of<K: Domain>(order: &MonomialOrder, nvars: usize, basis: &[PolyElement<K>]) -> PolyResult<Weights> {
    match order {
        MonomialOrder::Weight { weights, .. } => Ok(weights.iter().map(rat_from_int).collect()),
        MonomialOrder::Grlex | MonomialOrder::Grevlex => Ok(vec![Rat::ONE; nvars]),
        MonomialOrder::Lex => {
            let mut top: u64 = 1;
            for g in basis {
                // SymPy's monoms() of a zero polynomial is empty: no degree
                if let Some(d) = g.terms.keys().map(|m| total_degree(m)).max() {
                    top = top.max(d);
                }
            }
            let d: Rat = Rat::from(top + 1);
            Ok((0..nvars).map(|i| pow_rat(&d, (nvars - 1 - i) as u64)).collect())
        }
        other => Err(value_error(format!(
            "no weight vector is known for the order {}",
            other
        ))),
    }
}

fn pow_rat(a: &Rat, n: u64) -> Rat {
    let mut result: Rat = Rat::ONE;
    for _ in 0..n {
        result *= a;
    }
    result
}

/// The weight vector whose refinement by `order` is `order` itself.
fn target_weight(order: &MonomialOrder, nvars: usize) -> PolyResult<Weights> {
    match order {
        MonomialOrder::Weight { weights, .. } => Ok(weights.iter().map(rat_from_int).collect()),
        MonomialOrder::Grlex | MonomialOrder::Grevlex => Ok(vec![Rat::ONE; nvars]),
        MonomialOrder::Lex => Ok((0..nvars).map(|i| if i == 0 { Rat::ONE } else { Rat::ZERO }).collect()),
        other => Err(value_error(format!(
            "no weight vector is known for the order {}",
            other
        ))),
    }
}

/// A rational weight vector scaled to integers.
fn integral(weights: &[Rat]) -> Vec<Int> {
    let mut den: Int = Int::ONE;
    for w in weights {
        den = lcm(&den, &denom(w));
    }
    weights.iter().map(|w| numer(&(w * rat_from_int(&den)))).collect()
}

/// The terms of `f` of maximal weight for the weight vector.
pub fn initial_form<K: Domain>(f: &PolyElement<K>, weights: &[Rat]) -> PolyElement<K> {
    let mut best: Option<Rat> = None;
    let mut terms: Vec<(Monomial, K::Elem)> = Vec::new();
    for (monom, coeff) in f.ordered_terms() {
        let w: Rat = weight_of_monomial(weights, &monom);
        match &best {
            Some(b) if w < *b => {}
            Some(b) if w == *b => terms.push((monom, coeff)),
            _ => {
                best = Some(w);
                terms = vec![(monom, coeff)];
            }
        }
    }
    f.ring.from_dict(terms.into_iter().collect())
}

/// The S-polynomial of `f` and `g` with its representation.
fn spoly_cofactors<K: Domain>(
    f: &PolyElement<K>,
    g: &PolyElement<K>,
    cf: &[PolyElement<K>],
    cg: &[PolyElement<K>],
    ring: &Arc<PolyRing<K>>,
) -> PolyResult<(PolyElement<K>, Vec<PolyElement<K>>)> {
    let k: &K = &ring.domain;
    let lcm_monomial: Monomial = monomial_lcm(&f.LM(), &g.LM());
    let mf: PolyElement<K> = ring.term(monomial_div(&lcm_monomial, &f.LM())?, k.exquo(&k.one(), &f.LC())?);
    let mg: PolyElement<K> = ring.term(monomial_div(&lcm_monomial, &g.LM())?, k.exquo(&k.one(), &g.LC())?);
    let s: PolyElement<K> = mf.mul(f).sub(&mg.mul(g));
    let cofactors: Vec<PolyElement<K>> = cf.iter().zip(cg).map(|(a, b)| mf.mul(a).sub(&mg.mul(b))).collect();
    Ok((s, cofactors))
}

/// A Gröbner basis `H` of the ideal generated by `polys` for the order of
/// `ring`, with the representations `C`: `H[i] = sum_j C[i][j] polys[j]`.
/// The Buchberger algorithm without the optimisations of [`f4`], for small
/// inputs such as the initial forms of the walk.
#[allow(clippy::type_complexity)]
pub fn extended_groebner<K: Domain>(
    polys: &[PolyElement<K>],
    ring: &Arc<PolyRing<K>>,
) -> PolyResult<(Vec<PolyElement<K>>, Vec<Vec<PolyElement<K>>>)> {
    let f: Vec<PolyElement<K>> = polys.iter().map(|p| convert(p, ring)).collect();
    let n: usize = f.len();
    let mut basis: Vec<PolyElement<K>> = Vec::new();
    let mut representations: Vec<Vec<PolyElement<K>>> = Vec::new();
    for (i, p) in f.iter().enumerate() {
        if !p.is_zero() {
            basis.push(p.clone());
            representations.push((0..n).map(|j| if j == i { ring.one() } else { ring.zero() }).collect());
        }
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..basis.len() {
        for j in 0..i {
            pairs.push((i, j));
        }
    }
    while let Some((i, j)) = pairs.pop() {
        let (lm_i, lm_j) = (basis[i].LM(), basis[j].LM());
        if monomial_lcm(&lm_i, &lm_j) == monomial_mul(&lm_i, &lm_j) {
            continue; // the product criterion
        }
        let (s, mut cofactors) = spoly_cofactors(&basis[i], &basis[j], &representations[i], &representations[j], ring)?;
        if s.is_zero() {
            continue;
        }
        let (quotients, r) = s.div(&basis)?;
        if r.is_zero() {
            continue;
        }
        for (q, c) in quotients.iter().zip(&representations) {
            if !q.is_zero() {
                cofactors = cofactors.iter().zip(c).map(|(a, b)| a.sub(&q.mul(b))).collect();
            }
        }
        basis.push(r);
        representations.push(cofactors);
        let last: usize = basis.len() - 1;
        pairs.extend((0..last).map(|l| (last, l)));
    }
    Ok((basis, representations))
}

/// `f` in `ring` (the same variables, another order).
fn convert<K: Domain>(f: &PolyElement<K>, ring: &Arc<PolyRing<K>>) -> PolyElement<K> {
    ring.from_dict(f.terms.clone())
}

/// The point of the segment from `weights` to `target` where the ordering
/// of the monomials of `basis` first changes (`target` if it never does).
fn next_weight<K: Domain>(basis: &[PolyElement<K>], weights: &[Rat], target: &[Rat]) -> Weights {
    let mut tmin: Rat = Rat::ONE;
    for g in basis {
        let lm: Monomial = g.LM();
        for m in g.monomials() {
            if m == lm {
                continue;
            }
            let v: Vec<i64> = lm.iter().zip(&m).map(|(a, b)| *a as i64 - *b as i64).collect();
            let wv: Rat = dot(weights, &v);
            let tv: Rat = dot(target, &v);
            if tv < Rat::ZERO && Rat::ZERO < wv {
                let t: Rat = &wv / (&wv - &tv);
                if t < tmin {
                    tmin = t;
                }
            }
        }
    }
    weights.iter().zip(target).map(|(w, t)| w + &tmin * (t - w)).collect()
}

/// Whether `basis` is a Gröbner basis (every S-polynomial reduces to zero),
/// SymPy's `is_groebner`.
pub fn is_groebner<K: Domain>(basis: &[PolyElement<K>], ring: &Arc<PolyRing<K>>) -> PolyResult<bool> {
    let k: &K = &ring.domain;
    for i in 0..basis.len() {
        for j in (i + 1)..basis.len() {
            let (f, g) = (&basis[i], &basis[j]);
            let l: Monomial = monomial_lcm(&f.LM(), &g.LM());
            let s: PolyElement<K> = f
                .mul_term(&monomial_div(&l, &f.LM())?, &k.exquo(&k.one(), &f.LC())?)
                .sub(&g.mul_term(&monomial_div(&l, &g.LM())?, &k.exquo(&k.one(), &g.LC())?));
            if !s.rem(basis)?.is_zero() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Convert the reduced Gröbner basis `basis` of `ring` (whose order is the
/// source order) to the reduced Gröbner basis for the `target` order, in
/// the ring with that order, sorted by decreasing leading monomial.
///
/// `check` verifies every lifted basis (`Value` if one is not a Gröbner
/// basis: Python's `RuntimeError`); `Value` for a domain which is not a
/// field or an order without a known weight vector.
pub fn groebner_walk<K: Domain>(
    basis: &[PolyElement<K>],
    ring: &Arc<PolyRing<K>>,
    target: &MonomialOrder,
    check: bool,
    lift: WalkLift,
) -> PolyResult<Vec<PolyElement<K>>> {
    let n: usize = ring.ngens;
    let mut g: Vec<PolyElement<K>> = basis.iter().map(|p| convert(p, ring)).collect();
    if g.is_empty() || g.iter().all(|p| p.is_zero()) {
        let r: Arc<PolyRing<K>> = ring.with_order(target.clone());
        return Ok(g.iter().map(|p| convert(p, &r)).collect());
    }
    let mut weights: Weights = weight_of(&ring.order, n, &g)?;
    let tau: Weights = target_weight(target, n)?;
    let mut current: Arc<PolyRing<K>> = Arc::clone(ring);
    loop {
        let new_order: MonomialOrder = MonomialOrder::weight(integral(&weights), target.clone())?;
        let new_ring: Arc<PolyRing<K>> = ring.with_order(new_order);
        let initial: Vec<PolyElement<K>> = g
            .iter()
            .map(|p| convert(&initial_form(p, &weights), &new_ring))
            .collect();
        let mut lifted: Vec<PolyElement<K>> = Vec::new();
        match lift {
            WalkLift::Cofactors => {
                let (_, representations) = extended_groebner(&initial, &new_ring)?;
                let g_new: Vec<PolyElement<K>> = g.iter().map(|p| convert(p, &new_ring)).collect();
                for cofactors in &representations {
                    let mut h: PolyElement<K> = new_ring.zero();
                    for (c, p) in cofactors.iter().zip(&g_new) {
                        if !c.is_zero() {
                            h = h.add(&c.mul(p));
                        }
                    }
                    lifted.push(h);
                }
            }
            WalkLift::NormalForm => {
                for h in f4(&initial, &new_ring)? {
                    let h_old: PolyElement<K> = convert(&h, &current);
                    let normal_form: PolyElement<K> = h_old.rem(&g)?;
                    lifted.push(convert(&h_old.sub(&normal_form), &new_ring));
                }
            }
        }
        g = reduced_basis(&lifted, &new_ring);
        current = new_ring;
        if check && !is_groebner(&g, &current)? {
            return Err(value_error("the lifted basis is not a Groebner basis".to_string()));
        }
        if weights == tau {
            break;
        }
        weights = next_weight(&g, &weights, &tau);
    }
    let r: Arc<PolyRing<K>> = ring.with_order(target.clone());
    let converted: Vec<PolyElement<K>> = g.iter().map(|p| convert(p, &r)).collect();
    Ok(reduced_basis(&converted, &r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polys::domains::{RationalField, QQ};

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    fn strings<K: Domain>(basis: &[PolyElement<K>]) -> Vec<String> {
        basis.iter().map(|g| g.to_string()).collect()
    }

    #[test]
    fn doctests() {
        let r: Arc<PolyRing<RationalField>> =
            PolyRing::new(QQ, names(&["x", "y", "z"]), MonomialOrder::Grevlex).unwrap();
        let v = r.gens();
        let (x, y, z) = (&v[0], &v[1], &v[2]);
        let ideal: Vec<PolyElement<RationalField>> = vec![x.mul(z).sub(&y.pow(2)), x.pow(2).sub(&y.mul(z))];
        let g = f4(&ideal, &r).unwrap();
        for lift in [WalkLift::NormalForm, WalkLift::Cofactors] {
            let walked = groebner_walk(&g, &r, &MonomialOrder::Lex, true, lift).unwrap();
            assert_eq!(
                strings(&walked),
                vec!["x**2 - y*z", "x*y**2 - y*z**2", "x*z - y**2", "y**4 - y*z**3"]
            );
            let t = r.with_order(MonomialOrder::Lex);
            let direct = f4(&ideal.iter().map(|p| convert(p, &t)).collect::<Vec<_>>(), &t).unwrap();
            assert_eq!(walked, direct);
            let back = groebner_walk(&walked, &t, &MonomialOrder::Grevlex, false, lift).unwrap();
            assert_eq!(back, g);
        }
        // extended_groebner keeps the representations
        let l = PolyRing::new(QQ, names(&["x", "y"]), MonomialOrder::Lex).unwrap();
        let w = l.gens();
        let f = vec![w[0].pow(2).sub(&w[1]), w[0].mul(&w[1]).sub(&l.one())];
        let (h, c) = extended_groebner(&f, &l).unwrap();
        for (hi, ci) in h.iter().zip(&c) {
            let mut total = l.zero();
            for (cij, fj) in ci.iter().zip(&f) {
                total = total.add(&cij.mul(fj));
            }
            assert_eq!(&total, hi);
        }
        assert!(is_groebner(&f4(&f, &l).unwrap(), &l).unwrap());
        assert_eq!(
            strings(&[initial_form(&f[0], &[Rat::ONE, Rat::from(2u8)])]),
            vec!["x**2 - y"]
        );
        assert_eq!(strings(&[initial_form(&f[0], &[Rat::ONE, Rat::from(3u8)])]), vec!["-y"]);
    }
}
