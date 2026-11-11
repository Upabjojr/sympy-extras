//! Port of `sympy_extras/polys/buchberger.py`: reduced Gröbner bases by
//! Buchberger's algorithm on the typed rings over any field (SymPy's
//! `_buchberger`: GROEBNERNEWS2 of Becker and Weispfenning with the criteria
//! of Gebauer and Möller and the normal selection strategy).
//!
//! The Python code iterates over sets of indices and of pairs; here they are
//! ordered sets, which changes only the order of the steps between pairs
//! with the same lcm: the reduced basis returned is unique.
//!
//! References
//! ==========
//!
//! 1. T. Becker, V. Weispfenning, *Gröbner Bases: A Computational Approach
//!    to Commutative Algebra*, Springer, 1993, page 232.
//! 2. R. Gebauer, H. M. Möller, *On an installation of Buchberger's
//!    algorithm*, J. Symbolic Computation 6 (1988) 275-286.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/buchberger.py):
// sympy/polys/groebnertools.py (_buchberger, spoly) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: strict typing on the typed PolyRing over a field, ordered sets.
// ---------------------------------------------------------------------

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::error::{value_error, PolyResult};

use super::domains::Domain;
use super::rings::{
    monomial_div, monomial_divides, monomial_lcm, monomial_mul, Monomial, PolyElement, PolyRing, Terms,
};

/// A critical pair: the indices of its two polynomials.
type Pair = (usize, usize);

/// The remainder of the division of `f` by `divisors` (in this order, over
/// a field).
fn remainder<K: Domain>(f: &PolyElement<K>, divisors: &[&PolyElement<K>]) -> PolyResult<PolyElement<K>> {
    let ring: &Arc<PolyRing<K>> = &f.ring;
    let k: &K = &ring.domain;
    let mut leading: Vec<(Monomial, K::Elem, &Terms<K::Elem>)> = Vec::new();
    for g in divisors {
        if !g.is_zero() {
            let lm: Monomial = g.LM();
            let inverse: K::Elem = k.invert(&g.terms[&lm])?;
            leading.push((lm, inverse, &g.terms));
        }
    }
    let mut p: Terms<K::Elem> = f.terms.clone();
    let mut rest: Terms<K::Elem> = Terms::new();
    while !p.is_empty() {
        let m: Monomial = p.keys().max_by(|a, b| ring.order.cmp(a, b)).expect("a term").clone();
        let c: K::Elem = p[&m].clone();
        let divisor = leading.iter().find(|(lm, _, _)| monomial_divides(lm, &m));
        match divisor {
            Some((lm, inverse, terms)) => {
                let shift: Monomial = monomial_div(&m, lm)?;
                let factor: K::Elem = k.mul(&c, inverse);
                for (gm, gc) in terms.iter() {
                    let t: Monomial = monomial_mul(gm, &shift);
                    let product: K::Elem = k.mul(&factor, gc);
                    let value: K::Elem = match p.get(&t) {
                        None => k.neg(&product),
                        Some(old) => k.sub(old, &product),
                    };
                    if k.is_zero(&value) {
                        p.remove(&t);
                    } else {
                        p.insert(t, value);
                    }
                }
            }
            None => {
                rest.insert(m.clone(), c);
                p.remove(&m);
            }
        }
    }
    Ok(ring.from_dict(rest))
}

/// `lcm/LM(p1)*p1 - lcm/LM(p2)*p2` with `lcm` the lcm of the leading
/// monomials: the S-polynomial of monic `p1` and `p2`.
pub fn s_polynomial<K: Domain>(p1: &PolyElement<K>, p2: &PolyElement<K>) -> PolyResult<PolyElement<K>> {
    let lcm: Monomial = monomial_lcm(&p1.LM(), &p2.LM());
    Ok(p1
        .mul_monom(&monomial_div(&lcm, &p1.LM())?)
        .sub(&p2.mul_monom(&monomial_div(&lcm, &p2.LM())?)))
}

/// The state of the algorithm: the polynomials met, by index.
struct State<K: Domain> {
    f: Vec<PolyElement<K>>,
}

impl<K: Domain> State<K> {
    fn lm(&self, i: usize) -> Monomial {
        self.f[i].LM()
    }

    /// The index of the monic remainder of `g` by `f[j]`, `j` in `indices`,
    /// appended to `f` if new; `None` if it is zero.
    fn normal(&mut self, g: &PolyElement<K>, indices: &[usize]) -> PolyResult<Option<usize>> {
        let divisors: Vec<&PolyElement<K>> = indices.iter().map(|&j| &self.f[j]).collect();
        let h: PolyElement<K> = remainder(g, &divisors)?;
        if h.is_zero() {
            return Ok(None);
        }
        let h: PolyElement<K> = h.monic()?;
        if let Some(found) = self.f.iter().position(|q| *q == h) {
            return Ok(Some(found));
        }
        self.f.push(h);
        Ok(Some(self.f.len() - 1))
    }

    /// [BW] page 230: the new pairs `(h, g)`, `g` in `G`, the old pairs, the
    /// basis.
    fn update(&self, g_set: BTreeSet<usize>, b_set: BTreeSet<Pair>, ih: usize) -> (BTreeSet<usize>, BTreeSet<Pair>) {
        let mh: Monomial = self.lm(ih);
        let mut c_set: BTreeSet<usize> = g_set.clone();
        let mut d_set: BTreeSet<Pair> = BTreeSet::new();
        while let Some(ig) = c_set.pop_first() {
            let mg: Monomial = self.lm(ig);
            let lcm_hg: Monomial = monomial_lcm(&mh, &mg);
            // LCM(LM(h), LM(p)) divides LCM(LM(h), LM(g))
            let lcm_divides = |ip: usize| -> bool { monomial_divides(&monomial_lcm(&mh, &self.lm(ip)), &lcm_hg) };
            // HT(h) and HT(g) disjoint: mh*mg == LCMhg
            if monomial_mul(&mh, &mg) == lcm_hg
                || (!c_set.iter().any(|&ipx| lcm_divides(ipx)) && !d_set.iter().any(|pr| lcm_divides(pr.1)))
            {
                d_set.insert((ih, ig));
            }
        }
        let mut e_set: BTreeSet<Pair> = BTreeSet::new();
        while let Some((ih_, ig)) = d_set.pop_first() {
            let mg: Monomial = self.lm(ig);
            if monomial_mul(&mh, &mg) != monomial_lcm(&mh, &mg) {
                e_set.insert((ih_, ig));
            }
        }
        let mut b_new: BTreeSet<Pair> = BTreeSet::new();
        for (ig1, ig2) in b_set {
            let (mg1, mg2) = (self.lm(ig1), self.lm(ig2));
            let lcm12: Monomial = monomial_lcm(&mg1, &mg2);
            // HT(h) does not divide lcm(HT(g1), HT(g2))
            if !monomial_divides(&mh, &lcm12) || monomial_lcm(&mg1, &mh) == lcm12 || monomial_lcm(&mg2, &mh) == lcm12 {
                b_new.insert((ig1, ig2));
            }
        }
        b_new.extend(e_set);
        let mut g_new: BTreeSet<usize> = g_set
            .into_iter()
            .filter(|&ig| !monomial_divides(&mh, &self.lm(ig)))
            .collect();
        g_new.insert(ih);
        (g_new, b_new)
    }
}

/// The reduced Gröbner basis of the ideal generated by `polys` in `ring`
/// (whose domain must be a field, `ValueError` otherwise) for the order of
/// the ring: monic, sorted by decreasing leading monomial.
pub fn buchberger<K: Domain>(polys: &[PolyElement<K>], ring: &Arc<PolyRing<K>>) -> PolyResult<Vec<PolyElement<K>>> {
    if !ring.domain.is_field() {
        return Err(value_error(format!(
            "Buchberger's algorithm needs a field as the coefficient domain, got {}",
            ring.domain.name()
        )));
    }
    let order = &ring.order;
    let f: Vec<PolyElement<K>> = polys
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| {
            if p.ring == *ring {
                p.clone()
            } else {
                ring.from_dict(p.terms.clone())
            }
        })
        .collect();
    if f.is_empty() {
        return Ok(Vec::new());
    }
    // replace f with a reduced list of initial polynomials; see [BW] page 203
    let mut f1: Vec<PolyElement<K>> = f;
    let f: Vec<PolyElement<K>> = loop {
        let current: Vec<PolyElement<K>> = f1;
        let mut next: Vec<PolyElement<K>> = Vec::new();
        for i in 0..current.len() {
            let divisors: Vec<&PolyElement<K>> = current[..i].iter().collect();
            let r: PolyElement<K> = remainder(&current[i], &divisors)?;
            if !r.is_zero() {
                next.push(r.monic()?);
            }
        }
        if current == next {
            break current;
        }
        f1 = next;
    };
    let mut state: State<K> = State { f };
    let mut pending: BTreeSet<usize> = (0..state.f.len()).collect();
    let mut g_set: BTreeSet<usize> = BTreeSet::new();
    let mut pairs: BTreeSet<Pair> = BTreeSet::new();
    // algorithm GROEBNERNEWS2 in [BW] page 232
    while !pending.is_empty() {
        // the polynomial with the smallest leading monomial
        let ih: usize = *pending
            .iter()
            .min_by(|a, b| order.cmp(&state.lm(**a), &state.lm(**b)))
            .expect("a pending polynomial");
        pending.remove(&ih);
        (g_set, pairs) = state.update(g_set, pairs, ih);
    }
    while !pairs.is_empty() {
        // normal selection strategy: the pair with the smallest lcm
        let pair: Pair = *pairs
            .iter()
            .min_by(|a, b| {
                let la: Monomial = monomial_lcm(&state.lm(a.0), &state.lm(a.1));
                let lb: Monomial = monomial_lcm(&state.lm(b.0), &state.lm(b.1));
                order.cmp(&la, &lb)
            })
            .expect("a pair");
        pairs.remove(&pair);
        let s: PolyElement<K> = s_polynomial(&state.f[pair.0], &state.f[pair.1])?;
        // ordering the divisors is on average more efficient [Cox] page 111
        let mut g1: Vec<usize> = g_set.iter().copied().collect();
        g1.sort_by(|a, b| order.cmp(&state.lm(*a), &state.lm(*b)));
        if let Some(ih) = state.normal(&s, &g1)? {
            (g_set, pairs) = state.update(g_set, pairs, ih);
        }
    }
    // G is a Gröbner basis: reduce it
    let mut reduced: BTreeSet<usize> = BTreeSet::new();
    for &ig in &g_set {
        let others: Vec<usize> = g_set.iter().copied().filter(|&j| j != ig).collect();
        let g: PolyElement<K> = state.f[ig].clone();
        if let Some(ih) = state.normal(&g, &others)? {
            reduced.insert(ih);
        }
    }
    let mut result: Vec<PolyElement<K>> = reduced.into_iter().map(|i| state.f[i].clone()).collect();
    result.sort_by(|a, b| order.cmp(&b.LM(), &a.LM()));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers::rat;
    use crate::polys::domains::{FiniteField, QQ};
    use crate::polys::f4::f4;
    use crate::polys::rings::MonomialOrder;

    #[test]
    fn docstring_examples() {
        let r = PolyRing::new(QQ, vec!["x".into(), "y".into()], MonomialOrder::Grevlex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let basis = buchberger(
            &[x.mul(x).add(&y.mul(y)).sub(&r.one()), x.mul(y).sub(&r.from_int(2))],
            &r,
        )
        .unwrap();
        let shown: Vec<String> = basis.iter().map(|g| g.to_string()).collect();
        assert_eq!(shown, vec!["y**3 + 2*x - y", "x**2 + y**2 - 1", "x*y - 2"]);
        let gf = PolyRing::new(
            FiniteField::new(7).unwrap(),
            vec!["x".into(), "y".into()],
            MonomialOrder::Lex,
        )
        .unwrap();
        let h = gf.gens();
        let (x, y) = (&h[0], &h[1]);
        let basis = buchberger(&[x.mul(x).sub(y), y.mul(y).mul_ground(&3).sub(x)], &gf).unwrap();
        let shown: Vec<String> = basis.iter().map(|g| g.to_string()).collect();
        assert_eq!(shown, vec!["x + 4*y**2", "y**4 + 3*y"]);
        let l = PolyRing::new(QQ, vec!["x".into(), "y".into(), "z".into()], MonomialOrder::Lex).unwrap();
        let v = l.gens();
        let (x, y, z) = (&v[0], &v[1], &v[2]);
        let system = vec![
            x.mul(x).add(&y.mul(y)).add(&z.mul(z)).sub(&l.one()),
            x.mul(y).sub(&z.mul_ground(&rat(1, 3))),
            x.add(y).sub(&z.mul_ground(&rat(2, 1))),
        ];
        assert_eq!(buchberger(&system, &l).unwrap(), f4(&system, &l).unwrap());
        assert_eq!(buchberger(&[l.zero()], &l).unwrap(), vec![]);
        let s = s_polynomial(&x.mul(x).sub(y), &x.mul(y).sub(&l.one())).unwrap();
        assert_eq!(s.to_string(), "x - y**2");
    }
}
