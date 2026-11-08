//! Port of `sympy_extras/polys/stronggroebner.py`: strong Gröbner bases
//! over the integers.
//!
//! A *strong* Gröbner basis `G` of an ideal `I` of `ZZ[x]` is a finite
//! subset of `I` such that for every nonzero `f` in `I` some `g` in `G` has
//! a leading term dividing that of `f`, the coefficient included. Then every
//! `f` in `I` reduces to zero by `G`, and only those.
//!
//! The algorithm is Buchberger's with two kinds of polynomials for a pair
//! `f, g` with leading terms `a m` and `b n`, `L = lcm(m, n)`: the
//! **S-polynomial** `(c/a) (L/m) f - (c/b) (L/n) g` with `c = lcm(a, b)`,
//! and the **G-polynomial** `u (L/m) f + v (L/n) g` with `u a + v b = gcd(a,
//! b)` (only when neither of `a`, `b` divides the other). Both are reduced
//! with Euclidean division of the coefficients (`0 <= r < |d|`); the nonzero
//! remainders join the basis, which is finally made reduced (positive
//! leading coefficients, no leading term dividing another, every term
//! reduced by the others).
//!
//! References
//! ==========
//!
//! 1. A. Kandri-Rody, D. Kapur, *Computing a Gröbner basis of a polynomial
//!    ideal over a Euclidean domain*, Journal of Symbolic Computation 6
//!    (1988) 37-57.
//! 2. T. Becker, V. Weispfenning, *Gröbner Bases*, Springer (1993), chapter
//!    10.
//! 3. D. Lichtblau, *Effective computation of strong Gröbner bases over
//!    Euclidean domains*, Illinois Journal of Mathematics 56 (2012) 177-194.

use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::domains::{Domain, IntegerRing, ZZ};
use super::rings::{monomial_div, monomial_divides, monomial_lcm, Monomial, MonomialOrder, PolyElement, PolyRing};

type IntPoly = PolyElement<IntegerRing>;

fn check_ring(ring: &PolyRing<IntegerRing>) -> PolyResult<()> {
    if ring.domain != ZZ {
        return Err(value_error(format!(
            "strong Gröbner bases are computed over ZZ, the ring is over {}",
            ring.domain.name()
        )));
    }
    Ok(())
}

fn quotient(lcm: &[u32], m: &[u32]) -> Monomial {
    monomial_div(lcm, m).expect("the lcm is a multiple")
}

/// `q` with `c = q d + r` and `0 <= r < |d|`.
fn euclidean_quotient(c: &Int, d: &Int) -> Int {
    if is_negative(d) {
        -div_floor(c, &-d)
    } else {
        div_floor(c, d)
    }
}

/// The normal form of `f` modulo the basis over `ZZ`: every term `c t` with
/// `lm(g) | t` for some `g` of the basis is replaced by its remainder `(c mod
/// |lc(g)|) t`, until no term can be reduced. When the basis is a strong
/// Gröbner basis, the result is zero exactly for the elements of the ideal.
pub fn strong_reduce(f: &IntPoly, basis: &[IntPoly]) -> IntPoly {
    let ring: &Arc<PolyRing<IntegerRing>> = &f.ring;
    let leading: Vec<Option<(Monomial, Int)>> = basis
        .iter()
        .map(|g| if g.is_zero() { None } else { Some(g.LT()) })
        .collect();
    let mut remainder: IntPoly = ring.zero();
    let mut current: IntPoly = f.clone();
    while !current.is_zero() {
        let (monomial, coefficient) = current.LT();
        let mut reduced: bool = false;
        for (g, lead) in basis.iter().zip(&leading) {
            let (g_monomial, g_coefficient) = match lead {
                Some(lt) => lt,
                None => continue,
            };
            if !monomial_divides(g_monomial, &monomial) {
                continue;
            }
            let q: Int = euclidean_quotient(&coefficient, g_coefficient);
            if q.is_zero() {
                continue;
            }
            let multiplier: Monomial = quotient(&monomial, g_monomial);
            current = current.sub(&g.mul_monom(&multiplier).mul_ground(&q));
            reduced = true;
            break;
        }
        if !reduced {
            let term: IntPoly = ring.term(monomial, coefficient);
            remainder = remainder.add(&term);
            current = current.sub(&term);
        }
    }
    remainder
}

/// The S-polynomial over `ZZ`: the leading terms are brought to their least
/// common multiple, coefficients included, and subtracted.
pub fn s_polynomial_zz(f: &IntPoly, g: &IntPoly) -> IntPoly {
    let ((m, a), (n, b)) = (f.LT(), g.LT());
    let lcm_monomial: Monomial = monomial_lcm(&m, &n);
    let lcm_coefficient: Int = lcm(&a, &b);
    let left: IntPoly = f
        .mul_monom(&quotient(&lcm_monomial, &m))
        .mul_ground(&div_floor(&lcm_coefficient, &a));
    let right: IntPoly = g
        .mul_monom(&quotient(&lcm_monomial, &n))
        .mul_ground(&div_floor(&lcm_coefficient, &b));
    left.sub(&right)
}

/// The G-polynomial over `ZZ`: `u (L/m) f + v (L/n) g` with `u a + v b =
/// gcd(a, b)` for the leading terms `a m` and `b n`; `None` when one leading
/// coefficient divides the other.
pub fn g_polynomial_zz(f: &IntPoly, g: &IntPoly) -> Option<IntPoly> {
    let ((m, a), (n, b)) = (f.LT(), g.LT());
    if mod_floor(&a, &b).is_zero() || mod_floor(&b, &a).is_zero() {
        return None;
    }
    let (u, v, _) = ZZ.gcdex(&a, &b).expect("the extended Euclidean algorithm over ZZ");
    let lcm_monomial: Monomial = monomial_lcm(&m, &n);
    let left: IntPoly = f.mul_monom(&quotient(&lcm_monomial, &m)).mul_ground(&u);
    let right: IntPoly = g.mul_monom(&quotient(&lcm_monomial, &n)).mul_ground(&v);
    Some(left.add(&right))
}

fn positive(f: IntPoly) -> IntPoly {
    if is_negative(&f.LC()) {
        f.neg()
    } else {
        f
    }
}

/// The reduced strong Gröbner basis over `ZZ` of the ideal generated by
/// `polys` in `ring` (whose domain must be `ZZ`), sorted by decreasing
/// leading monomial; every element of the ideal, and only those, reduces to
/// zero by it.
pub fn strong_groebner(polys: &[IntPoly], ring: &Arc<PolyRing<IntegerRing>>) -> PolyResult<Vec<IntPoly>> {
    check_ring(ring)?;
    let mut basis: Vec<IntPoly> = polys.iter().filter(|f| !f.is_zero()).cloned().map(positive).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..basis.len() {
        for j in 0..i {
            pairs.push((i, j));
        }
    }
    while let Some((i, j)) = pairs.pop() {
        let (f, g) = (basis[i].clone(), basis[j].clone());
        let mut candidates: Vec<IntPoly> = vec![s_polynomial_zz(&f, &g)];
        if let Some(gp) = g_polynomial_zz(&f, &g) {
            candidates.push(gp);
        }
        for candidate in candidates {
            let remainder: IntPoly = strong_reduce(&candidate, &basis);
            if !remainder.is_zero() {
                let n: usize = basis.len();
                pairs.extend((0..n).map(|k| (n, k)));
                basis.push(positive(remainder));
            }
        }
    }
    Ok(reduced(basis))
}

/// The reduced strong Gröbner basis of a strong Gröbner basis.
fn reduced(basis: Vec<IntPoly>) -> Vec<IntPoly> {
    if basis.is_empty() {
        return Vec::new();
    }
    let order: MonomialOrder = basis[0].ring.order.clone();
    // minimal: drop the elements whose leading term another one divides
    let mut ordered: Vec<IntPoly> = basis.into_iter().filter(|f| !f.is_zero()).collect();
    ordered.sort_by(|f, g| {
        order
            .cmp(&f.LM(), &g.LM())
            .then_with(|| abs(&f.LC()).cmp(&abs(&g.LC())))
    });
    let mut kept: Vec<IntPoly> = Vec::new();
    for f in ordered {
        let (monomial, coefficient) = f.LT();
        if kept.iter().any(|g| {
            let (gm, gc) = g.LT();
            monomial_divides(&gm, &monomial) && mod_floor(&coefficient, &gc).is_zero()
        }) {
            continue;
        }
        kept.push(f);
    }
    // reduced: every element reduced by the others
    let mut result: Vec<IntPoly> = Vec::new();
    for i in 0..kept.len() {
        let others: Vec<IntPoly> = kept
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, g)| g.clone())
            .collect();
        result.push(positive(strong_reduce(&kept[i], &others)));
    }
    result.sort_by(|f, g| order.cmp(&g.LM(), &f.LM()));
    result
}

/// Whether the polynomials, over `ZZ`, form a strong Gröbner basis of the
/// ideal they generate: every S-polynomial and G-polynomial of two of them
/// reduces to zero.
pub fn is_strong_groebner(basis: &[IntPoly]) -> bool {
    let polys: Vec<IntPoly> = basis.iter().filter(|f| !f.is_zero()).cloned().collect();
    for i in 0..polys.len() {
        for j in 0..i {
            let (f, g) = (&polys[i], &polys[j]);
            if !strong_reduce(&s_polynomial_zz(f, g), &polys).is_zero() {
                return false;
            }
            if let Some(gp) = g_polynomial_zz(f, g) {
                if !strong_reduce(&gp, &polys).is_zero() {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(basis: &[IntPoly]) -> Vec<String> {
        basis.iter().map(|g| g.to_string()).collect()
    }

    #[test]
    fn doctests() {
        let r = PolyRing::new(ZZ, vec!["x".to_string(), "y".to_string()], MonomialOrder::Lex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let basis = vec![x.mul_ground(&int(2)), y.mul_ground(&int(3)), x.mul(y)];
        assert_eq!(
            strong_reduce(&x.mul(y).mul_ground(&int(7)).add(&x.mul_ground(&int(5))), &basis).to_string(),
            "x"
        );
        assert_eq!(
            strong_reduce(&x.mul_ground(&int(6)).add(&y.mul_ground(&int(3))), &basis).to_string(),
            "0"
        );
        let f = x.mul(x).mul_ground(&int(2)).add(y);
        let h = x.mul(y).mul_ground(&int(3)).add(&r.one());
        assert_eq!(s_polynomial_zz(&f, &h).to_string(), "-2*x + 3*y**2");
        assert_eq!(
            g_polynomial_zz(&x.mul_ground(&int(2)), &y.mul_ground(&int(3)))
                .unwrap()
                .to_string(),
            "x*y"
        );
        assert!(g_polynomial_zz(&x.mul_ground(&int(2)), &y.mul_ground(&int(4))).is_none());
        let two_x = x.mul_ground(&int(2));
        let three_y = y.mul_ground(&int(3));
        assert_eq!(
            strings(&strong_groebner(&[two_x.clone(), three_y.clone()], &r).unwrap()),
            vec!["x*y", "2*x", "3*y"]
        );
        assert_eq!(
            strings(&strong_groebner(&[two_x.clone(), x.mul_ground(&int(3))], &r).unwrap()),
            vec!["x"]
        );
        assert_eq!(
            strings(&strong_groebner(&[r.from_int(4), two_x.add(&r.one())], &r).unwrap()),
            vec!["1"]
        );
        let big_g = strong_groebner(&[f.clone(), h.clone()], &r).unwrap();
        assert_eq!(strings(&big_g), vec!["x*y + 3*y**3 + 1", "2*x - 3*y**2", "9*y**3 + 2"]);
        assert_eq!(strong_reduce(&x.mul(&f).sub(&y.mul(&h)), &big_g).to_string(), "0");
        assert_eq!(strong_reduce(x, &big_g).to_string(), "x");
        assert!(!is_strong_groebner(&[two_x.clone(), three_y.clone()]));
        assert!(is_strong_groebner(&[two_x, three_y, x.mul(y)]));
    }
}
