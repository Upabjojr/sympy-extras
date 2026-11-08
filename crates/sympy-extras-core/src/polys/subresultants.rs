//! Port of `sympy_extras/polys/subresultants.py`: the principal subresultant
//! coefficients, as Hong's projection operator for the cylindrical
//! algebraic decomposition needs them.
//!
//! The Python module works on SymPy's `dup_*`/`dmp_*` representation. Here
//! a multivariate polynomial is a univariate one over a tower of
//! [`PolynomialRing`](super::domains::PolynomialRing)s, so `dmp_psc(f, g, u, K)`
//! is [`dup_psc`] over `K[x_2, ..., x_n]`; `psc` on SymPy's `PolyElement`
//! has no counterpart.

use super::densebasic::dup_degree;
use super::domains::Domain;
use super::euclidtools::dup_inner_subresultants;

/// The principal subresultant coefficients `[psc_0, ..., psc_m]` of `f`
/// and `g` (`m` the smaller degree): `psc_j` is the coefficient of `x**j`
/// in the `j`-th subresultant, zero when it is defective, so `psc_0` is the
/// resultant. Empty when `f` or `g` is zero.
pub fn dup_psc<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() || g.is_empty() {
        return Vec::new();
    }
    let (r, s) = dup_inner_subresultants(f, g, k);
    if r.is_empty() {
        return Vec::new();
    }
    let m: usize = dup_degree(f).min(dup_degree(g)) as usize;
    let mut psc: Vec<K::Elem> = vec![k.zero(); m + 1];
    for (ri, si) in r.iter().skip(1).zip(s.iter().skip(1)) {
        psc[dup_degree(ri) as usize] = si.clone();
    }
    psc
}

#[cfg(test)]
mod tests {
    use super::super::densebasic::testing::zz;
    use super::*;
    use crate::numbers::*;
    use crate::polys::domains::{PolynomialRing, ZZ};

    #[test]
    fn doctests() {
        let ints = |v: &[i64]| -> Vec<Int> { v.iter().map(|c| Int::from(*c)).collect() };
        assert_eq!(dup_psc(&zz(&[1, 0, 1]), &zz(&[1, 0, -1]), &ZZ), ints(&[4, 0, 1]));
        assert_eq!(dup_psc(&zz(&[1, 0, -1, 0]), &zz(&[1, 0, -1]), &ZZ), ints(&[0, 0, 1]));
        assert!(dup_psc(&zz(&[]), &zz(&[1]), &ZZ).is_empty());
        // dmp_psc(x**2*y + x, x + y) over ZZ[y]: [y**3 - y, 1]
        let k = PolynomialRing::new(ZZ, "y");
        let f: Vec<Vec<Int>> = vec![ints(&[1, 0]), ints(&[1]), vec![]];
        let g: Vec<Vec<Int>> = vec![ints(&[1]), ints(&[1, 0])];
        assert_eq!(dup_psc(&f, &g, &k), vec![ints(&[1, 0, -1, 0]), ints(&[1])]);
    }
}
