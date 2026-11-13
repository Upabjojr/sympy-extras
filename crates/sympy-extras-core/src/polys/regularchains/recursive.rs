//! Port of `sympy_extras/polys/regularchains/recursive.py`: polynomials of
//! `ZZ[x_1, ..., x_n]` seen as univariate ones in their main variable (the
//! greatest variable occurring, the first generator being the greatest), with
//! lazy pseudo-divisions, SymPy's `prem` and subresultant remainder
//! sequence, the subresultant chain by the structure theorem (Ducos), and
//! the factors over `ZZ` (Wang's algorithm).

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/regularchains/recursive.py):
// sympy/polys/rings.py (PolyElement.prem, PolyElement.subresultants,
// PolyElement.coeff_wrt, PolyElement.sort_key) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: typed functions on the PolyElement over ZZ, the variable given by
// its index.
// ---------------------------------------------------------------------

use std::sync::Arc;

use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::super::domains::{Domain, IntegerRing, SparsePolynomialRing, ZZ};
use super::super::multifactor::{sqf_part as wang_sqf_part, zz_factor};
use super::super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

/// A polynomial over the integers.
pub type ZPoly = PolyElement<IntegerRing>;

/// SymPy's sort key of a polynomial: its length, then its terms in
/// decreasing order.
pub type SortKey = (usize, Vec<(Monomial, Int)>);

/// The gcd domain of the ring (`IntegerPolynomialRing`, SymPy's gcd).
pub fn integer_ring(ring: &PolyRing<IntegerRing>) -> SparsePolynomialRing<IntegerRing> {
    SparsePolynomialRing::sympy(ZZ, ring.names.clone())
}

/// The gcd of SymPy's `PolyElement.gcd` (positive leading coefficient).
pub fn gcd(p: &ZPoly, q: &ZPoly) -> ZPoly {
    let r: SparsePolynomialRing<IntegerRing> = integer_ring(&p.ring);
    let result: ZPoly = r.gcd(&r.ring.from_dict(p.terms.clone()), &r.ring.from_dict(q.terms.clone()));
    p.ring.from_dict(result.terms)
}

/// `(h, p/h, q/h)` for the gcd `h` (SymPy's `PolyElement.cofactors`).
pub fn cofactors(p: &ZPoly, q: &ZPoly) -> PolyResult<(ZPoly, ZPoly, ZPoly)> {
    let h: ZPoly = gcd(p, q);
    if h.is_zero() {
        return Ok((h, p.ring.zero(), p.ring.zero()));
    }
    let (a, b) = (exact_quotient(p, &h)?, exact_quotient(q, &h)?);
    Ok((h, a, b))
}

/// The greatest absolute value of a coefficient.
pub fn max_norm(p: &ZPoly) -> Int {
    p.terms.values().map(abs).max().unwrap_or(Int::ZERO)
}

/// SymPy's `PolyElement.sort_key`.
pub fn sort_key(p: &ZPoly) -> SortKey {
    let mut terms: Vec<(Monomial, Int)> = p.terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect();
    terms.sort();
    terms.reverse();
    (p.terms.len(), terms)
}

/// The index of the greatest variable of `p`, the number of generators for
/// a constant.
pub fn main_variable(p: &ZPoly) -> usize {
    let n: usize = p.ring.ngens;
    let mut level: usize = n;
    for monom in p.terms.keys() {
        if let Some(i) = (0..level).find(|&i| monom[i] != 0) {
            level = i;
        }
        if level == 0 {
            break;
        }
    }
    level
}

/// The degree of `p` in the variable of index `i` (`-1` for zero).
pub fn degree(p: &ZPoly, i: usize) -> i64 {
    p.terms.keys().map(|m| i64::from(m[i])).max().unwrap_or(-1)
}

/// The coefficient of the `d`-th power of the variable `i`.
pub fn coefficient(p: &ZPoly, i: usize, d: i64) -> ZPoly {
    let terms: Terms<Int> = p
        .terms
        .iter()
        .filter(|(m, _)| i64::from(m[i]) == d)
        .map(|(m, c)| {
            let mut key: Monomial = m.clone();
            key[i] = 0;
            (key, c.clone())
        })
        .collect();
    p.ring.from_dict(terms)
}

/// The leading coefficient of `p` in its main variable.
pub fn initial(p: &ZPoly) -> ZPoly {
    let i: usize = main_variable(p);
    if i == p.ring.ngens {
        return p.clone();
    }
    coefficient(p, i, degree(p, i))
}

/// `p` times the `k`-th power of the variable `i`.
fn power(p: &ZPoly, i: usize, k: i64) -> ZPoly {
    if k == 0 {
        return p.clone();
    }
    let mut monom: Monomial = vec![0; p.ring.ngens];
    monom[i] = k as u32;
    p.mul_monom(&monom)
}

fn does_not_divide(q: &ZPoly, p: &ZPoly) -> crate::error::PolyError {
    value_error(format!("{} does not divide {}", q, p))
}

/// `p/q` for `q` dividing `p` (`ValueError` otherwise, `ZeroDivisionError`
/// for zero).
pub fn exact_quotient(p: &ZPoly, q: &ZPoly) -> PolyResult<ZPoly> {
    if q.is_zero() {
        return Err(zero_division("division by zero"));
    }
    if q.terms.len() == 1 {
        let (m, c) = q.terms.iter().next().expect("one term");
        let mut terms: Terms<Int> = Terms::new();
        for (n, d) in &p.terms {
            let (quotient, remainder) = div_mod_floor(d, c);
            if !remainder.is_zero() || n.iter().zip(m).any(|(a, b)| a < b) {
                return Err(does_not_divide(q, p));
            }
            terms.insert(n.iter().zip(m).map(|(a, b)| a - b).collect(), quotient);
        }
        return Ok(p.ring.from_dict(terms));
    }
    exquo(p, q)
}

/// The exact division by a polynomial of several terms, in the lex order.
fn exquo(p: &ZPoly, q: &ZPoly) -> PolyResult<ZPoly> {
    let (lm, lc) = q
        .terms
        .iter()
        .next_back()
        .map(|(m, c)| (m.clone(), c.clone()))
        .expect("terms");
    let rest: Vec<(Monomial, Int)> = q
        .terms
        .iter()
        .filter(|(m, _)| **m != lm)
        .map(|(m, c)| (m.clone(), c.clone()))
        .collect();
    let mut r: Terms<Int> = p.terms.clone();
    let mut quotient: Terms<Int> = Terms::new();
    while let Some((m, c)) = r.iter().next_back().map(|(m, c)| (m.clone(), c.clone())) {
        if m.iter().zip(&lm).any(|(a, b)| a < b) {
            return Err(does_not_divide(q, p));
        }
        let (k, remainder) = div_mod_floor(&c, &lc);
        if !remainder.is_zero() {
            return Err(does_not_divide(q, p));
        }
        let e: Monomial = m.iter().zip(&lm).map(|(a, b)| a - b).collect();
        r.remove(&m);
        for (n, d) in &rest {
            let t: Monomial = n.iter().zip(&e).map(|(a, b)| a + b).collect();
            let value: Int = r.get(&t).cloned().unwrap_or(Int::ZERO) - &k * d;
            if value.is_zero() {
                r.remove(&t);
            } else {
                r.insert(t, value);
            }
        }
        quotient.insert(e, k);
    }
    Ok(p.ring.from_dict(quotient))
}

/// `p` without its term of highest degree in its main variable.
pub fn tail(p: &ZPoly) -> ZPoly {
    let i: usize = main_variable(p);
    if i == p.ring.ngens {
        return p.ring.zero();
    }
    let d: i64 = degree(p, i);
    p.ring.from_dict(
        p.terms
            .iter()
            .filter(|(m, _)| i64::from(m[i]) != d)
            .map(|(m, c)| (m.clone(), c.clone()))
            .collect(),
    )
}

/// The primitive part of `p` over the integers, with a positive leading
/// coefficient.
pub fn normalize(p: &ZPoly) -> ZPoly {
    if p.is_zero() {
        return p.clone();
    }
    let mut content: Int = Int::ZERO;
    for c in p.terms.values() {
        content = crate::numbers::gcd(&content, c);
        if content == Int::ONE {
            break;
        }
    }
    if is_negative(p.terms.values().next_back().expect("terms")) {
        content = -content;
    }
    if content == Int::ONE {
        return p.clone();
    }
    p.ring.from_dict(
        p.terms
            .iter()
            .map(|(m, c)| (m.clone(), div_floor(c, &content)))
            .collect(),
    )
}

/// `p` divided by the gcd of its coefficients as a polynomial in the
/// variable `i`, normalized.
pub fn primitive_part(p: &ZPoly, i: usize) -> PolyResult<ZPoly> {
    if p.is_zero() {
        return Ok(p.clone());
    }
    let d: i64 = degree(p, i);
    let mut content: ZPoly = p.ring.zero();
    for k in (0..=d).rev() {
        let c: ZPoly = coefficient(p, i, k);
        if !c.is_zero() {
            content = gcd(&content, &c);
            if content.is_ground() {
                break;
            }
        }
    }
    let p: ZPoly = if content.is_ground() {
        p.clone()
    } else {
        exact_quotient(p, &content)?
    };
    Ok(normalize(&p))
}

/// The lazy pseudo-division of `f` by `g` in the variable `i`: `(m, q, r)`
/// with `m f == q g + r`, the degree of `r` lower than that of `g`, and `m`
/// a product of divisors of the leading coefficient of `g`.
pub fn pseudo_divide(f: &ZPoly, g: &ZPoly, i: usize) -> PolyResult<(ZPoly, ZPoly, ZPoly)> {
    let ring: &Arc<PolyRing<IntegerRing>> = &f.ring;
    let dg: i64 = degree(g, i);
    if dg < 0 {
        return Err(zero_division("pseudo-division by zero"));
    }
    let lg: ZPoly = coefficient(g, i, dg);
    let mut m: ZPoly = ring.one();
    let mut q: ZPoly = ring.zero();
    let mut r: ZPoly = f.clone();
    while !r.is_zero() {
        let dr: i64 = degree(&r, i);
        if dr < dg {
            break;
        }
        let lr: ZPoly = coefficient(&r, i, dr);
        let (_, a, b) = cofactors(&lr, &lg)?;
        m = m.mul(&b);
        q = q.mul(&b).add(&power(&a, i, dr - dg));
        r = r.mul(&b).sub(&power(&a.mul(g), i, dr - dg));
    }
    Ok((m, q, r))
}

/// The remainder of the lazy pseudo-division of `f` by `g`.
pub fn pseudo_remainder(f: &ZPoly, g: &ZPoly, i: usize) -> PolyResult<ZPoly> {
    let dg: i64 = degree(g, i);
    if dg < 0 {
        return Err(zero_division("pseudo-division by zero"));
    }
    let lg: ZPoly = coefficient(g, i, dg);
    let mut r: ZPoly = f.clone();
    while !r.is_zero() {
        let dr: i64 = degree(&r, i);
        if dr < dg {
            break;
        }
        let lr: ZPoly = coefficient(&r, i, dr);
        let (_, a, b) = cofactors(&lr, &lg)?;
        r = r.mul(&b).sub(&power(&a.mul(g), i, dr - dg));
    }
    Ok(r)
}

/// The pseudo-remainder of `f` by `g` in the variable `i`: `m f = q g + r`
/// with `m` the leading coefficient of `g` to the power `deg f - deg g + 1`
/// (SymPy's `PolyElement.prem`).
pub fn prem(f: &ZPoly, g: &ZPoly, i: usize) -> PolyResult<ZPoly> {
    let (df, dg) = (degree(f, i), degree(g, i));
    if dg < 0 {
        return Err(zero_division("polynomial division"));
    }
    let (mut r, mut dr) = (f.clone(), df);
    if df < dg {
        return Ok(r);
    }
    let mut big_n: i64 = df - dg + 1;
    let lc_g: ZPoly = coefficient(g, i, dg);
    loop {
        let lc_r: ZPoly = coefficient(&r, i, dr);
        let j: i64 = dr - dg;
        big_n -= 1;
        r = r.mul(&lc_g).sub(&power(&g.mul(&lc_r), i, j));
        dr = degree(&r, i);
        if dr < dg {
            break;
        }
    }
    Ok(r.mul(&lc_g.pow(big_n as u64)))
}

/// The subresultant polynomial remainder sequence of `f` and `g` in the
/// variable `i`, starting with `f` and `g` (the one of greater degree first)
/// (SymPy's `PolyElement.subresultants`).
pub fn subresultants(f: &ZPoly, g: &ZPoly, i: usize) -> PolyResult<Vec<ZPoly>> {
    let ring: &Arc<PolyRing<IntegerRing>> = &f.ring;
    let (mut f, mut g) = (f.clone(), g.clone());
    let (mut n, mut m) = (degree(&f, i), degree(&g, i));
    if n < m {
        std::mem::swap(&mut f, &mut g);
        std::mem::swap(&mut n, &mut m);
    }
    if f.is_zero() {
        return Ok(vec![ring.zero(), ring.zero()]);
    }
    if g.is_zero() {
        return Ok(vec![f, ring.one()]);
    }
    let mut result: Vec<ZPoly> = vec![f.clone(), g.clone()];
    let mut d: i64 = n - m;
    let mut h: ZPoly = prem(&f, &g, i)?;
    if d % 2 == 0 {
        h = h.neg();
    }
    let mut lc: ZPoly = coefficient(&g, i, m);
    let mut c: ZPoly = lc.pow(d as u64).neg();
    while !h.is_zero() {
        let k: i64 = degree(&h, i);
        result.push(h.clone());
        f = g;
        g = h;
        d = m - k;
        m = k;
        let b: ZPoly = lc.neg().mul(&c.pow(d as u64));
        h = exact_quotient(&prem(&f, &g, i)?, &b)?;
        lc = coefficient(&g, i, k);
        c = if d > 1 {
            exact_quotient(&lc.neg().pow(d as u64), &c.pow((d - 1) as u64))?
        } else {
            lc.neg()
        };
    }
    Ok(result)
}

/// The subresultants `S_0, ..., S_{n-1}` of `f` and `g` in the variable `i`,
/// up to signs, where `n = deg g <= deg f` (`ValueError` otherwise).
pub fn subresultant_chain(f: &ZPoly, g: &ZPoly, i: usize) -> PolyResult<Vec<ZPoly>> {
    let ring: &Arc<PolyRing<IntegerRing>> = &f.ring;
    let n: i64 = degree(g, i);
    if degree(f, i) < n || n < 1 {
        return Err(value_error("the degrees must satisfy deg f >= deg g >= 1".to_string()));
    }
    let mut chain: Vec<ZPoly> = vec![ring.zero(); n as usize];
    let remainders: Vec<ZPoly> = subresultants(f, g, i)?;
    let mut s: ZPoly = coefficient(g, i, n).pow((degree(f, i) - n) as u64);
    let mut before: i64 = n;
    for r in &remainders[2..] {
        if r.is_zero() {
            break;
        }
        let d: i64 = degree(r, i);
        chain[(before - 1) as usize] = r.clone();
        let delta: i64 = before - d;
        let mut r: ZPoly = r.clone();
        if delta > 1 {
            r = exact_quotient(
                &coefficient(&r, i, d).pow((delta - 1) as u64).mul(&r),
                &s.pow((delta - 1) as u64),
            )?;
            chain[d as usize] = r.clone();
        }
        s = coefficient(&r, i, d);
        before = d;
    }
    Ok(chain)
}

/// The square-free part, normalized.
pub fn sqf_part(p: &ZPoly) -> PolyResult<ZPoly> {
    Ok(normalize(&wang_sqf_part(p)?))
}

/// The irreducible factors of a polynomial which is not constant (SymPy's
/// `dmp_factor_list` over `ZZ`: the monomial content, then Wang's algorithm
/// in the variables which occur).
fn factor_list(p: &ZPoly) -> PolyResult<Vec<ZPoly>> {
    let ring: &Arc<PolyRing<IntegerRing>> = &p.ring;
    let n: usize = ring.ngens;
    let exponents: Vec<u32> = (0..n)
        .map(|k| p.terms.keys().map(|m| m[k]).min().expect("terms"))
        .collect();
    let shifted: Terms<Int> = p
        .terms
        .iter()
        .map(|(m, c)| (m.iter().zip(&exponents).map(|(e, s)| e - s).collect(), c.clone()))
        .collect();
    let mut factors: Vec<ZPoly> = Vec::new();
    for (k, e) in exponents.iter().enumerate() {
        if *e != 0 {
            factors.push(ring.term((0..n).map(|j| u32::from(j == k)).collect(), Int::ONE));
        }
    }
    let used: Vec<usize> = (0..n).filter(|&k| shifted.keys().any(|m| m[k] != 0)).collect();
    if used.is_empty() {
        return Ok(factors);
    }
    let sub: Arc<PolyRing<IntegerRing>> = PolyRing::new(
        ZZ,
        used.iter().map(|&k| ring.names[k].clone()).collect(),
        MonomialOrder::Lex,
    )?;
    let g: ZPoly = sub.from_dict(
        shifted
            .iter()
            .map(|(m, c)| (used.iter().map(|&k| m[k]).collect(), c.clone()))
            .collect(),
    );
    for (h, _) in zz_factor(&g)?.1 {
        let mut terms: Terms<Int> = Terms::new();
        for (m, value) in &h.terms {
            let mut full: Monomial = vec![0; n];
            for (position, e) in used.iter().zip(m) {
                full[*position] = *e;
            }
            terms.insert(full, value.clone());
        }
        factors.push(ring.from_dict(terms));
    }
    Ok(factors)
}

/// The irreducible factors of `p` over the integers which are not constant,
/// without their multiplicities, sorted.
pub fn distinct_factors(p: &ZPoly) -> PolyResult<Vec<ZPoly>> {
    if p.is_ground() {
        return Ok(Vec::new());
    }
    let mut factors: Vec<ZPoly> = factor_list(p)?
        .iter()
        .filter(|q| !q.is_ground())
        .map(normalize)
        .collect();
    factors.sort_by_cached_key(sort_key);
    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let r = PolyRing::new(ZZ, vec!["x".into(), "y".into()], MonomialOrder::Lex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        // the docstring example: (y + 1) x**2 + y x + 3
        let p = y.add(&r.one()).mul(&x.mul(x)).add(&y.mul(x)).add(&r.from_int(3));
        assert_eq!(main_variable(&p), 0);
        assert_eq!(main_variable(&y.add(&r.one())), 1);
        assert_eq!(
            (initial(&p).to_string(), tail(&p).to_string()),
            ("y + 1".to_string(), "x*y + 3".to_string())
        );
        let f = x.mul(x).add(&y.mul(y));
        let (m, q, rem) = pseudo_divide(&f, &x.mul(y).add(&r.one()), 0).unwrap();
        assert_eq!(m.mul(&f), q.mul(&x.mul(y).add(&r.one())).add(&rem));
        let chain = subresultant_chain(&x.mul(x).sub(y), &x.mul(x).sub(&y.mul(y)).add(x), 0).unwrap();
        assert_eq!(chain.len(), 2);
        let factors = distinct_factors(&x.mul(x).sub(&y.mul(y)).mul(x)).unwrap();
        assert_eq!(factors.len(), 3);
    }
}
