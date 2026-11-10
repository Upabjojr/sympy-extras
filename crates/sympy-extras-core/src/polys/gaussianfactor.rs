//! Port of `sympy_extras/polys/gaussianfactor.py`: factorisation of
//! multivariate polynomials over the Gaussian integers `ZZ_I` and the
//! Gaussian rationals `QQ_I` by Trager's algorithm over `Q(i)`.
//!
//! For a polynomial `f` over `Q(i)`, made monic (its leading coefficient in
//! the lexicographic order divided out) and square-free, a shift
//! `g(X) = f(x_1 - s_1 i, ..., x_n - s_n i)` is chosen whose norm
//! `N(g) = g * conjugate(g) = A**2 + B**2` (`g = A + iB`, `A`, `B` over `Q`)
//! is square-free: the shifts `(0, ..., 0)`, then one variable at a time
//! (`1` in a single position, the variables of lowest degree first), then
//! `(j, ..., j)` for `j = 1, 2, ...`, in SymPy's order. The norm is factored
//! over `Q` (Wang's algorithm, [`super::multifactor`]); each of its
//! irreducible factors `h` gives the irreducible factor `gcd(h, g)` of `g`
//! over `Q(i)`, shifted back, and the multiplicities of the factors of the
//! original polynomial come from trial division. Over `ZZ_I` the
//! denominators and the Gaussian contents of the factors are moved into the
//! coefficient as SymPy does, and the factors are sorted by SymPy's
//! `_sort_factors` (the length of the dense representation, the
//! multiplicity, the dense coefficients, Gaussian numbers compared by their
//! imaginary parts first).
//!
//! [`factor_all`] is the port of `polytools._factor_all` (the factorisation
//! over `QQ` in all the variables of the ring) which the Python module
//! imports; the rest of `polytools` waits for the core.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/gaussianfactor.py):
// sympy/polys/factortools.py (dmp_zz_i_factor, dmp_qq_i_factor,
// dmp_ext_factor, dmp_trial_division, dmp_factor_list),
// sympy/polys/sqfreetools.py (dmp_sqf_norm, _dmp_sqf_norm_shifts,
// dmp_sqf_p, dmp_sqf_part), sympy/polys/polyutils.py (_sort_factors)
// (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: sparse polynomials instead of dense nested lists, the field
// QQ<I> only (the norm computed as A**2 + B**2 instead of a resultant),
// the gcd over ZZ_I with the denominators cleared, strict typing.
// ---------------------------------------------------------------------

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::domains::{
    ConvertFrom, Domain, GaussianInteger, GaussianIntegerRing, GaussianRational, GaussianRationalField, IntegerRing,
    RationalField, SparsePolynomialRing, QQ, QQ_I, ZZ, ZZ_I,
};
use super::multifactor as mf;
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

/// A polynomial over `QQ_I`.
pub type GaussPoly = PolyElement<GaussianRationalField>;

/// The factors with their multiplicities.
pub type GaussFactors = Vec<(GaussPoly, usize)>;

/// The dense nested representation SymPy sorts the factors by, the
/// coefficients `(y, x)` (`GaussKey`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GaussKey {
    Coefficient(Rat, Rat),
    Tuple(Vec<GaussKey>),
}

/// SymPy's key of a factor with its multiplicity (`GaussSortKey`).
pub type GaussSortKey = (i64, usize, GaussKey);

fn gaussian(x: Rat, y: Rat) -> GaussianRational {
    GaussianRational::new(x, y)
}

fn convert_to_zz_i(c: &GaussianRational) -> PolyResult<GaussianInteger> {
    ZZ_I.convert(c, &QQ_I)
}

fn convert_to_qq_i(c: &GaussianInteger) -> GaussianRational {
    QQ_I.convert(c, &ZZ_I)
        .expect("a Gaussian integer is a Gaussian rational")
}

fn zz_i_ring(names: &[String]) -> SparsePolynomialRing<GaussianIntegerRing> {
    SparsePolynomialRing::new(ZZ_I, names.to_vec())
}

/// `f` times the lcm of the denominators of its coefficients (SymPy's
/// `dmp_clear_denoms` over `QQ_I`), over `ZZ_I`.
fn cleared(
    f: &GaussPoly,
    z: &SparsePolynomialRing<GaussianIntegerRing>,
) -> PolyResult<PolyElement<GaussianIntegerRing>> {
    let mut common: GaussianInteger = ZZ_I.one();
    for c in f.terms.values() {
        common = ZZ_I.lcm(&common, &ZZ_I.denom_of(c));
    }
    let k: GaussianRational = convert_to_qq_i(&common);
    let mut terms: Terms<GaussianInteger> = Terms::new();
    for (m, c) in &f.terms {
        terms.insert(m.clone(), convert_to_zz_i(&QQ_I.mul(c, &k))?);
    }
    Ok(z.ring.from_dict(terms))
}

/// The monic gcd over `QQ<I>` (SymPy's `dmp_inner_gcd` over an algebraic
/// field returns it monic), computed over `ZZ_I`.
fn poly_gcd(f: &GaussPoly, g: &GaussPoly) -> PolyResult<GaussPoly> {
    if f.is_zero() {
        return g.monic();
    }
    if g.is_zero() {
        return f.monic();
    }
    let z: SparsePolynomialRing<GaussianIntegerRing> = zz_i_ring(&f.ring.names);
    let h: PolyElement<GaussianIntegerRing> = z.gcd(&cleared(f, &z)?, &cleared(g, &z)?);
    let terms: Terms<GaussianRational> = h.terms.iter().map(|(m, c)| (m.clone(), convert_to_qq_i(c))).collect();
    f.ring.from_dict(terms).monic()
}

/// The square-free part of `f`, monic (`dmp_sqf_part`).
fn sqf_part(f: &GaussPoly) -> PolyResult<GaussPoly> {
    if f.is_zero() {
        return Ok(f.clone());
    }
    let mut g: GaussPoly = f.clone();
    for i in 0..f.ring.ngens {
        g = poly_gcd(&g, &f.diff(i))?;
        if g.is_ground() {
            // the gcd of the remaining derivatives is 1 too
            return f.monic();
        }
    }
    f.exquo(&g)?.monic()
}

/// The binomial coefficient `comb(n, k)`.
fn comb(n: u32, k: u32) -> Int {
    let mut result: Int = Int::ONE;
    for i in 0..k {
        result = result * Int::from(n - i) / Int::from(i + 1);
    }
    result
}

/// `f(x_1 + a_1, ..., x_n + a_n)` (`dmp_shift`).
fn shift(f: &GaussPoly, a: &[GaussianRational]) -> GaussPoly {
    let mut terms: Terms<GaussianRational> = f.terms.clone();
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        let mut shifted: Terms<GaussianRational> = Terms::new();
        for (m, c) in &terms {
            let e: u32 = m[i];
            for k in 0..=e {
                let coefficient: GaussianRational = QQ_I.mul(
                    c,
                    &QQ_I.mul(&QQ_I.from_integer(&comb(e, k)), &QQ_I.pow(ai, u64::from(e - k))),
                );
                let mut monomial: Monomial = m.clone();
                monomial[i] = k;
                let total: GaussianRational = match shifted.get(&monomial) {
                    Some(previous) => QQ_I.add(previous, &coefficient),
                    None => coefficient,
                };
                shifted.insert(monomial, total);
            }
        }
        terms = shifted.into_iter().filter(|(_, c)| !c.is_zero()).collect();
    }
    f.ring.from_dict(terms)
}

/// The norm `f * conjugate(f) = A**2 + B**2` over `QQ` (SymPy's `dmp_norm`:
/// the resultant with `y**2 + 1` of `A + y*B`).
fn norm(f: &GaussPoly) -> PolyResult<PolyElement<RationalField>> {
    let ring: Arc<PolyRing<RationalField>> = PolyRing::new(QQ, f.ring.names.clone(), MonomialOrder::Lex)?;
    let a: PolyElement<RationalField> = ring.from_dict(f.terms.iter().map(|(m, c)| (m.clone(), c.x.clone())).collect());
    let b: PolyElement<RationalField> = ring.from_dict(f.terms.iter().map(|(m, c)| (m.clone(), c.y.clone())).collect());
    Ok(a.mul(&a).add(&b.mul(&b)))
}

/// Whether `r` is square-free over `QQ` (`dmp_sqf_p`).
fn sqf_p(r: &PolyElement<RationalField>) -> bool {
    if r.is_zero() {
        return true;
    }
    let q: SparsePolynomialRing<RationalField> = SparsePolynomialRing::sympy(QQ, r.ring.names.clone());
    for i in 0..r.ring.ngens {
        let d: PolyElement<RationalField> = r.diff(i);
        if d.is_zero() {
            continue;
        }
        if q.gcd(r, &d).degree_in(i) != 0 {
            return false;
        }
    }
    true
}

/// `f(X - s*I)`.
fn shifted_by(f: &GaussPoly, s: &[i64]) -> GaussPoly {
    let a: Vec<GaussianRational> = s.iter().map(|si| gaussian(Rat::ZERO, rat(-si, 1))).collect();
    shift(f, &a)
}

/// `(s, g, r)`: `g = f(X - s*I)` with the square-free norm `r`, the
/// candidate shifts in SymPy's order (`dmp_sqf_norm`,
/// `_dmp_sqf_norm_shifts`).
fn sqf_norm(f: &GaussPoly) -> PolyResult<(Vec<i64>, GaussPoly, PolyElement<RationalField>)> {
    let n: usize = f.ring.ngens;
    let s0: Vec<i64> = vec![0; n];
    let r: PolyElement<RationalField> = norm(f)?;
    if sqf_p(&r) {
        return Ok((s0, f.clone(), r));
    }
    let mut by_degree: Vec<(i64, usize)> = (0..n).map(|i| (f.degree_in(i), i)).collect();
    by_degree.sort();
    for (di, i) in by_degree {
        if di <= 0 {
            continue;
        }
        let mut s1: Vec<i64> = s0.clone();
        s1[i] = 1;
        let g: GaussPoly = shifted_by(f, &s1);
        let r: PolyElement<RationalField> = norm(&g)?;
        if sqf_p(&r) {
            return Ok((s1, g, r));
        }
    }
    let mut j: i64 = 0;
    loop {
        j += 1;
        let s: Vec<i64> = vec![j; n];
        let g: GaussPoly = shifted_by(f, &s);
        let r: PolyElement<RationalField> = norm(&g)?;
        if sqf_p(&r) {
            return Ok((s, g, r));
        }
    }
}

/// The multiplicities of the factors in `f` (`dmp_trial_division`), sorted.
fn trial_division(f: &GaussPoly, factors: &[GaussPoly]) -> PolyResult<GaussFactors> {
    let mut result: GaussFactors = Vec::new();
    let mut f: GaussPoly = f.clone();
    for factor in factors {
        let mut k: usize = 0;
        loop {
            let (q, r) = f.div(std::slice::from_ref(factor))?;
            if !r.is_zero() {
                break;
            }
            f = q.into_iter().next().expect("one quotient");
            k += 1;
        }
        if k == 0 {
            return Err(value_error("trial division failed".to_string()));
        }
        result.push((factor.clone(), k));
    }
    Ok(sorted(result))
}

fn sorted(mut factors: GaussFactors) -> GaussFactors {
    factors.sort_by_cached_key(gaussian_sort_key);
    factors
}

/// The coefficient and the integer primitive factors (positive leading
/// coefficients), sorted, of `f` over `QQ` in all the variables of its ring
/// (SymPy's `dup_factor_list`/`dmp_factor_list` over `ZZ` or `QQ`; the
/// port of `polytools._factor_all`); `ValueError` for zero.
pub fn factor_all(f: &PolyElement<RationalField>) -> PolyResult<(Rat, mf::Factors)> {
    if f.is_zero() {
        return Err(value_error("min() arg is an empty sequence".to_string()));
    }
    let names: Vec<String> = f.ring.names.clone();
    let n: usize = names.len();
    let exponents: Vec<u32> = (0..n)
        .map(|i| f.terms.keys().map(|m| m[i]).min().expect("a nonzero polynomial"))
        .collect();
    let shifted: Terms<Rat> = f
        .terms
        .iter()
        .map(|(m, c)| (m.iter().zip(&exponents).map(|(e, s)| e - s).collect(), c.clone()))
        .collect();
    let mut content: Rat = Rat::ZERO;
    for c in shifted.values() {
        let (cn, cd) = (numer(&content), denom(&content));
        let (vn, vd) = (numer(c), denom(c));
        content = rat_from_parts(gcd(&cn, &vn), &cd * &vd / gcd(&cd, &vd));
    }
    let ints: Terms<Int> = shifted
        .iter()
        .map(|(m, c)| (m.clone(), numer(&(c / &content))))
        .collect();
    let ring: Arc<PolyRing<IntegerRing>> = PolyRing::new(ZZ, names.clone(), MonomialOrder::Lex)?;
    let (coefficient, mut factors): (Int, mf::Factors) = if n == 1 {
        // dup_zz_factor, sorted below as the Python sorts it
        mf::zz_factor(&ring.from_dict(ints))?
    } else {
        let used: Vec<usize> = (0..n).filter(|&i| ints.keys().any(|m| m[i] != 0)).collect();
        if used.is_empty() {
            let zero: Monomial = vec![0; n];
            (ints.get(&zero).cloned().unwrap_or(Int::ZERO), Vec::new())
        } else {
            let sub: Arc<PolyRing<IntegerRing>> =
                PolyRing::new(ZZ, used.iter().map(|&i| names[i].clone()).collect(), MonomialOrder::Lex)?;
            let g: PolyElement<IntegerRing> = sub.from_dict(
                ints.iter()
                    .map(|(m, c)| (used.iter().map(|&i| m[i]).collect(), c.clone()))
                    .collect(),
            );
            let (coefficient, sub_factors) = mf::zz_factor(&g)?;
            let mut factors: mf::Factors = Vec::new();
            for (h, k) in sub_factors {
                let mut terms: Terms<Int> = Terms::new();
                for (m, value) in &h.terms {
                    let mut full: Monomial = vec![0; n];
                    for (position, e) in used.iter().zip(m) {
                        full[*position] = *e;
                    }
                    terms.insert(full, value.clone());
                }
                factors.push((ring.from_dict(terms), k));
            }
            (coefficient, factors)
        }
    };
    for i in (0..n).rev() {
        if exponents[i] != 0 {
            let monomial: Monomial = (0..n).map(|k| u32::from(k == i)).collect();
            factors.insert(0, (ring.term(monomial, Int::ONE), exponents[i] as usize));
        }
    }
    // the key of the univariate branch (degree, multiplicity, dense
    // coefficients) is that of `_sorted` in one variable
    Ok((rat_from_int(&coefficient) * content, mf::sorted(factors)))
}

/// The leading coefficient and the monic irreducible factors over `QQ<I>`
/// with their multiplicities (Trager's algorithm, `dmp_ext_factor`).
pub fn gaussian_ext_factor(f: &GaussPoly) -> PolyResult<(GaussianRational, GaussFactors)> {
    if f.is_zero() {
        return Ok((QQ_I.zero(), Vec::new()));
    }
    let lc: GaussianRational = f.LC();
    let f: GaussPoly = f.monic()?;
    if f.is_ground() {
        return Ok((lc, Vec::new()));
    }
    let original: GaussPoly = f.clone();
    let f: GaussPoly = sqf_part(&f)?;
    let (s, mut g, r) = sqf_norm(&f)?;
    let (_, rational_factors) = factor_all(&r)?;
    let factors: Vec<GaussPoly> = if rational_factors.len() == 1 {
        vec![f]
    } else {
        let a: Vec<GaussianRational> = s.iter().map(|si| gaussian(Rat::ZERO, rat(*si, 1))).collect();
        let mut factors: Vec<GaussPoly> = Vec::new();
        for (h, _k) in &rational_factors {
            let hk: GaussPoly = f.ring.from_dict(
                h.terms
                    .iter()
                    .map(|(m, c)| (m.clone(), gaussian(rat_from_int(c), Rat::ZERO)))
                    .collect(),
            );
            let hg: GaussPoly = poly_gcd(&hk, &g)?;
            g = g.exquo(&hg)?;
            factors.push(shift(&hg, &a));
        }
        factors
    };
    Ok((lc, trial_division(&original, &factors)?))
}

/// The dense nested representation from the variable `i` on, the
/// coefficients `(y, x)` as SymPy compares Gaussian numbers.
fn dense_key(terms: &Terms<GaussianRational>, i: usize, n: usize) -> GaussKey {
    let mut by_degree: BTreeMap<u32, Terms<GaussianRational>> = BTreeMap::new();
    for (m, c) in terms {
        by_degree.entry(m[i]).or_default().insert(m.clone(), c.clone());
    }
    let top: u32 = match by_degree.keys().next_back() {
        Some(top) => *top,
        None => return GaussKey::Tuple(Vec::new()),
    };
    let empty: Terms<GaussianRational> = Terms::new();
    if i == n - 1 {
        return GaussKey::Tuple(
            (0..=top)
                .rev()
                .map(|d| match by_degree.get(&d).and_then(|level| level.values().next()) {
                    Some(c) => GaussKey::Coefficient(c.y.clone(), c.x.clone()),
                    None => GaussKey::Coefficient(Rat::ZERO, Rat::ZERO),
                })
                .collect(),
        );
    }
    GaussKey::Tuple(
        (0..=top)
            .rev()
            .map(|d| dense_key(by_degree.get(&d).unwrap_or(&empty), i + 1, n))
            .collect(),
    )
}

/// SymPy's key of a factor with its multiplicity (`_sort_factors` with
/// `multiple=True`): the length of the dense representation (the degree in
/// the first variable plus one), the multiplicity, then the dense
/// coefficients.
pub fn gaussian_sort_key(item: &(GaussPoly, usize)) -> GaussSortKey {
    let (f, k) = item;
    let length: i64 = if f.is_zero() { 0 } else { f.degree_in(0) + 1 };
    (length, *k, dense_key(&f.terms, 0, f.ring.ngens))
}

/// SymPy's `dmp_ground_content` over `ZZ_I`.
fn content(f: &PolyElement<GaussianIntegerRing>) -> GaussianInteger {
    let mut content: GaussianInteger = ZZ_I.zero();
    for c in f.terms.values() {
        content = ZZ_I.gcd(&content, c);
        if content == ZZ_I.one() {
            break;
        }
    }
    content
}

/// The coefficient and the irreducible factors of `f` over `ZZ_I`
/// (`integral`, the coefficients Gaussian integers) or `QQ_I`, as SymPy's
/// `dmp_factor_list` returns them: the monomial of the variables split off,
/// over `ZZ_I` the factors primitive Gaussian integer polynomials, over
/// `QQ_I` monic; sorted (`dmp_zz_i_factor`, `dmp_qq_i_factor`).
pub fn gaussian_factor_list(f: &GaussPoly, integral: bool) -> PolyResult<(GaussianRational, GaussFactors)> {
    let ring: Arc<PolyRing<GaussianRationalField>> = Arc::clone(&f.ring);
    let n: usize = ring.ngens;
    if f.is_zero() {
        return Ok((QQ_I.zero(), Vec::new()));
    }
    // dmp_terms_gcd
    let j: Vec<u32> = (0..n)
        .map(|i| f.terms.keys().map(|m| m[i]).min().expect("a nonzero polynomial"))
        .collect();
    let mut f: GaussPoly = ring.from_dict(
        f.terms
            .iter()
            .map(|(m, c)| (m.iter().zip(&j).map(|(e, ji)| e - ji).collect(), c.clone()))
            .collect(),
    );
    let mut cont: GaussianRational = QQ_I.one();
    let z: SparsePolynomialRing<GaussianIntegerRing> = zz_i_ring(&ring.names);
    if integral {
        // dmp_ground_primitive over ZZ_I
        let mut terms: Terms<GaussianInteger> = Terms::new();
        for (m, c) in &f.terms {
            terms.insert(m.clone(), convert_to_zz_i(c)?);
        }
        cont = convert_to_qq_i(&content(&z.ring.from_dict(terms)));
        let mut divided: Terms<GaussianRational> = Terms::new();
        for (m, c) in &f.terms {
            divided.insert(m.clone(), QQ_I.exquo(c, &cont)?);
        }
        f = ring.from_dict(divided);
    }
    // dmp_qq_i_factor: over QQ<I> (the content over the field QQ_I only
    // changes the leading coefficient, which is that of f in the end)
    let (mut coeff, mut factors) = gaussian_ext_factor(&f)?;
    if integral {
        // dmp_zz_i_factor: the denominators and the contents over ZZ_I of
        // the factors moved into the coefficient
        let mut new_factors: GaussFactors = Vec::new();
        for (fac, k) in factors {
            let mut denominator: GaussianInteger = ZZ_I.one();
            for c in fac.terms.values() {
                denominator = ZZ_I.lcm(&denominator, &ZZ_I.denom_of(c));
            }
            let d: GaussianRational = convert_to_qq_i(&denominator);
            let mut terms: Terms<GaussianInteger> = Terms::new();
            for (m, c) in &fac.terms {
                terms.insert(m.clone(), convert_to_zz_i(&QQ_I.mul(c, &d))?);
            }
            let numerator: PolyElement<GaussianIntegerRing> = z.ring.from_dict(terms);
            let fac_content: GaussianInteger = content(&numerator);
            let mut primitive: Terms<GaussianRational> = Terms::new();
            for (m, c) in &numerator.terms {
                primitive.insert(m.clone(), convert_to_qq_i(&ZZ_I.exquo(c, &fac_content)?));
            }
            let power: u64 = k as u64;
            coeff = QQ_I.exquo(
                &QQ_I.mul(&coeff, &QQ_I.pow(&convert_to_qq_i(&fac_content), power)),
                &QQ_I.pow(&d, power),
            )?;
            new_factors.push((ring.from_dict(primitive), k));
        }
        factors = new_factors;
    }
    for i in (0..n).rev() {
        if j[i] != 0 {
            let monomial: Monomial = (0..n).map(|t| u32::from(t == i)).collect();
            factors.insert(0, (ring.term(monomial, QQ_I.one()), j[i] as usize));
        }
    }
    Ok((QQ_I.mul(&coeff, &cont), sorted(factors)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(names: &[&str]) -> Arc<PolyRing<GaussianRationalField>> {
        PolyRing::new(QQ_I, names.iter().map(|s| s.to_string()).collect(), MonomialOrder::Lex).unwrap()
    }

    #[test]
    fn sum_of_squares() {
        let r = ring(&["x", "y"]);
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let f = x.mul(x).add(&y.mul(y));
        let (lc, factors) = gaussian_ext_factor(&f).unwrap();
        assert_eq!(lc, QQ_I.one());
        let shown: Vec<(String, usize)> = factors.iter().map(|(g, k)| (g.to_string(), *k)).collect();
        assert_eq!(shown.len(), 2);
        assert!(factors.iter().all(|(g, _)| g.degree() == 1));
    }

    #[test]
    fn repeated_rational_factor() {
        let r = ring(&["x", "y"]);
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let d = x.sub(y);
        let f = x.mul(x).add(&y.mul(y)).mul(&d).mul(&d);
        let (_, factors) = gaussian_ext_factor(&f).unwrap();
        let multiplicities: Vec<usize> = factors.iter().map(|(_, k)| *k).collect();
        assert_eq!(multiplicities, vec![1, 1, 2]);
        assert_eq!(factors[2].0, d);
    }
}
