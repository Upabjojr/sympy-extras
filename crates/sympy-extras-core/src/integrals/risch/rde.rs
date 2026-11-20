//! Port of `sympy_extras/integrals/risch/rde.py`: the Risch differential
//! equation `Dy + f y = g` over the tower (Bronstein, chapter 6): the weak
//! normaliser, the normal and special denominators, the degree bounds,
//! `SPDE`, the non-cancellation and cancellation cases and `rischDE`.

#![allow(non_snake_case, clippy::type_complexity)]

use std::collections::BTreeMap;

use super::risch::{
    derivation, derivation_element, frac_in, frac_in_pair, gcdex_diophantine, in_t, non_elementary,
    recognize_log_derivative, splitfactor, unique_dummy, Case, DifferentialExtension,
};
use super::tower::{RationalFunction, TowerPoly};
use crate::core::basic::Basic;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

/// A degree bound, `OO` for an infinite one (`Degree`).
pub type Degree = i64;
/// The infinite degree (`OO`).
pub const OO: Degree = 1 << 62;

/// Whether the bound is infinite (`is_oo`).
pub fn is_oo(n: Degree) -> bool {
    n >= OO >> 1
}

/// The bound, which must be finite (`finite`).
pub fn finite(n: Degree) -> PolyResult<i64> {
    if is_oo(n) {
        return Err(PolyError::Value("a finite degree bound is needed, got oo".to_string()));
    }
    Ok(n)
}

fn rf_int(n: i64) -> RationalFunction {
    RationalFunction::from_int(&Int::from(n))
}

fn t_poly(de: &DifferentialExtension) -> TowerPoly {
    TowerPoly::monomial(&de.t, 1, &RationalFunction::one())
}

fn t2_1(de: &DifferentialExtension) -> TowerPoly {
    TowerPoly::from_list(
        &[
            RationalFunction::one(),
            RationalFunction::zero(),
            RationalFunction::one(),
        ],
        &de.t,
    )
}

/// With the level of the extension one lower (`DecrementLevel`).
pub(crate) fn below<T>(
    de: &mut DifferentialExtension,
    f: impl FnOnce(&mut DifferentialExtension) -> PolyResult<T>,
) -> PolyResult<T> {
    de.decrement_level()?;
    let result: PolyResult<T> = f(de);
    de.increment_level()?;
    result
}

/// The order of `a` at `p` in `t` (`order_at`), `OO` for `a == 0`.
pub fn order_at(a: &TowerPoly, p: &TowerPoly, t: &Basic) -> PolyResult<Degree> {
    if a.is_zero() {
        return Ok(OO);
    }
    let a: TowerPoly = a.as_poly_strict(t)?;
    let p: TowerPoly = p.as_poly_strict(t)?;
    if p == TowerPoly::monomial(t, 1, &RationalFunction::one()) {
        return Ok(a.ET().0 as i64);
    }
    let mut power_list: Vec<(TowerPoly, i64)> = Vec::new();
    let mut p1: TowerPoly = p.clone();
    let mut r: TowerPoly = a.rem(&p1)?;
    let mut tracks_power: i64 = 1;
    while r.is_zero() {
        crate::timeout::check()?;
        power_list.push((p1.clone(), tracks_power));
        p1 = p1.mul(&p1)?;
        tracks_power *= 2;
        r = a.rem(&p1)?;
    }
    let mut n: i64 = 0;
    let mut product: TowerPoly = TowerPoly::one(t);
    while let Some((f, k)) = power_list.pop() {
        let productf: TowerPoly = product.mul(&f)?;
        if a.rem(&productf)?.is_zero() {
            n += k;
            product = productf;
        }
    }
    Ok(n)
}

/// The order of `a/d` at infinity (`order_at_oo`).
pub fn order_at_oo(a: &TowerPoly, d: &TowerPoly, t: &Basic) -> Degree {
    if a.is_zero() {
        return OO;
    }
    d.degree(Some(t)) - a.degree(Some(t))
}

/// The rational roots of `p` with their multiplicities for every value of
/// the parameters of its coefficients (`_rational_roots`).
pub fn rational_roots(p: &TowerPoly) -> PolyResult<Vec<RationalFunction>> {
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

/// The weak normaliser (`weak_normalizer`): `(q, (sn, sd))` with
/// `f - Dq/q = sn/sd` weakly normalised.
pub fn weak_normalizer(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, (TowerPoly, TowerPoly))> {
    let z: Basic = unique_dummy("z");
    let (a, d) = (in_t(a, de)?, in_t(d, de)?);
    let (dn, _ds) = splitfactor(&d, de, false)?;
    let g: TowerPoly = dn.gcd(&dn.diff(Some(&de.t))?)?;
    let d_sqf_part: TowerPoly = dn.quo(&g)?;
    let d1: TowerPoly = d_sqf_part.quo(&d_sqf_part.gcd(&g)?)?;
    let (_a1, _b) = gcdex_diophantine(&d.quo(&d1)?, &d1, &a)?;
    let zf: RationalFunction = RationalFunction::generator(&z);
    let r: TowerPoly = TowerPoly::from_element_strict(
        &a.sub(&derivation(&d1, de, false)?.mul_ground(&zf))?.resultant(&d1)?,
        &z,
    )?;
    if !r.has(std::slice::from_ref(&z)) {
        return Ok((TowerPoly::one(&de.t), (a, d)));
    }
    let mut n_int: Vec<i64> = Vec::new();
    for root in rational_roots(&r)? {
        if let Some(n) = root.as_int() {
            if n > Int::ZERO {
                let k: i64 = i64::try_from(&n).map_err(|_| PolyError::Value("a root too large".to_string()))?;
                if !n_int.contains(&k) {
                    n_int.push(k);
                }
            }
        }
    }
    n_int.sort();
    let mut q: TowerPoly = TowerPoly::one(&de.t);
    for n in n_int {
        let factor: TowerPoly = a.sub(&derivation(&d1, de, false)?.mul_ground(&rf_int(n)))?.gcd(&d1)?;
        q = q.mul(&factor.pow(n as usize))?;
    }
    let dq: TowerPoly = derivation(&q, de, false)?;
    let sn: TowerPoly = q.mul(&a)?.sub(&d.mul(&dq)?)?;
    let sd: TowerPoly = q.mul(&d)?;
    let (sn, sd) = sn.cancel(&sd)?;
    Ok((q, (sn, sd)))
}

/// The normal part of the denominator (`normal_denom`).
pub fn normal_denom(
    fa: &TowerPoly,
    fd: &TowerPoly,
    ga: &TowerPoly,
    gd: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, (TowerPoly, TowerPoly), (TowerPoly, TowerPoly), TowerPoly)> {
    let (fa, fd, ga, gd) = (in_t(fa, de)?, in_t(fd, de)?, in_t(ga, de)?, in_t(gd, de)?);
    let (dn, _ds) = splitfactor(&fd, de, false)?;
    let (en, _es) = splitfactor(&gd, de, false)?;
    let p: TowerPoly = dn.gcd(&en)?;
    let h: TowerPoly = en.gcd(&en.diff(Some(&de.t))?)?.quo(&p.gcd(&p.diff(Some(&de.t))?)?)?;
    let a: TowerPoly = dn.mul(&h)?;
    let c: TowerPoly = a.mul(&h)?;
    if !c.div(&en)?.1.is_zero() {
        return Err(non_elementary("en does not divide dn*h**2 in normal_denom()."));
    }
    let ca: TowerPoly = c.mul(&ga)?;
    let (ca, cd) = ca.cancel(&gd)?;
    let ba: TowerPoly = a.mul(&fa)?.sub(&dn.mul(&derivation(&h, de, false)?)?.mul(&fd)?)?;
    let (ba, bd) = ba.cancel(&fd)?;
    Ok((a, (ba, bd), (ca, cd), h))
}

pub(crate) fn special_denom_cancel_bound(
    a: &TowerPoly,
    ba: &TowerPoly,
    bd: &TowerPoly,
    n: Degree,
    de: &mut DifferentialExtension,
    case: Case,
) -> PolyResult<Degree> {
    let mut n: Degree = n;
    let zero: RationalFunction = RationalFunction::zero();
    if case == Case::Exp {
        let dcoeff: TowerPoly = de.d.quo(&t_poly(de))?;
        let alpha: RationalFunction = ba.eval(&zero).neg().div(&bd.eval(&zero))?.div(&a.eval(&zero))?;
        let found = below(de, |de| {
            let (alphaa, alphad) = frac_in(&alpha, &de.t, false)?;
            let (etaa, etad) = frac_in_pair(&dcoeff, &TowerPoly::one(&dcoeff.gen), &de.t, false)?;
            super::prde::parametric_log_deriv(&alphaa, &alphad, &etaa, &etad, de)
        })?;
        if let Some((q, m, _z)) = found {
            if q == 1 {
                n = n.min(m);
            }
        }
    } else if case == Case::Tan {
        let dcoeff: TowerPoly = de.d.quo(&t2_1(de))?;
        let (betaa, alphaa, alphad) = super::prde::real_imag(&ba.neg(), &bd.mul(a)?, &de.t)?;
        let betad: TowerPoly = alphad.clone();
        let alpha: RationalFunction = alphaa.as_element()?.div(&alphad.as_element()?)?;
        let beta: RationalFunction = betaa.as_element()?.div(&betad.as_element()?)?;
        let found = below(de, |de| {
            let (alphaa, alphad) = frac_in(&alpha, &de.t, false)?;
            let (betaa, betad) = frac_in(&beta, &de.t, false)?;
            let (etaa, etad) = frac_in_pair(&dcoeff, &TowerPoly::one(&dcoeff.gen), &de.t, false)?;
            if recognize_log_derivative(&betaa.mul_ground(&rf_int(2)), &betad, de)? {
                return super::prde::parametric_log_deriv(&alphaa, &alphad, &etaa, &etad, de);
            }
            Ok(None)
        })?;
        if let Some((q, m, _z)) = found {
            if q == 1 && m % 2 == 0 {
                n = n.min(m / 2);
            }
        }
    }
    Ok(n)
}

/// The special part of the denominator (`special_denom`): `(A, B, C, h)`.
pub fn special_denom(
    a: &TowerPoly,
    ba: &TowerPoly,
    bd: &TowerPoly,
    ca: &TowerPoly,
    cd: &TowerPoly,
    de: &mut DifferentialExtension,
    case: Option<Case>,
) -> PolyResult<(TowerPoly, TowerPoly, TowerPoly, TowerPoly)> {
    let case: Case = case.unwrap_or(de.case);
    let p: TowerPoly = match case {
        Case::Exp => t_poly(de),
        Case::Tan => t2_1(de),
        Case::Primitive | Case::Base => {
            let big_b: TowerPoly = ba.quo(bd)?;
            let big_c: TowerPoly = ca.quo(cd)?;
            return Ok((a.clone(), big_b, big_c, TowerPoly::one(&de.t)));
        }
        _ => {
            return Err(PolyError::NotImplemented(format!(
                "The {} case is not implemented in special_denom().",
                case.name()
            )));
        }
    };
    let t: Basic = de.t.clone();
    let nb: Degree = order_at(ba, &p, &t)? - order_at(bd, &p, &t)?;
    let nc: Degree = order_at(ca, &p, &t)? - order_at(cd, &p, &t)?;
    let mut n: Degree = 0.min(nc - 0.min(nb));
    if nb == 0 {
        n = special_denom_cancel_bound(a, ba, bd, n, de, case)?;
    }
    let big_n: Degree = 0.max(-nb).max(n - nc);
    let p_n: TowerPoly = p.pow(big_n as usize);
    let pn: TowerPoly = p.pow((-n) as usize);
    let big_a: TowerPoly = a.mul(&p_n)?;
    let big_b: TowerPoly = ba.mul(&p_n.quo(bd)?)?.add(
        &a.mul(&derivation(&p, de, false)?.quo(&p)?)?
            .mul(&p_n)?
            .mul_ground(&rf_int(n)),
    )?;
    let big_c: TowerPoly = ca.mul(&p_n)?.mul(&pn)?.quo(cd)?;
    Ok((big_a, big_b, big_c, pn))
}

fn length_one(m_list: &[RationalFunction]) -> PolyResult<Option<i64>> {
    if m_list.len() != 1 {
        return Err(PolyError::Value("Length of m should be 1".to_string()));
    }
    Ok(m_list[0].as_int().and_then(|m| i64::try_from(&m).ok()))
}

/// A bound of the degree of the polynomial solutions (`bound_degree`).
pub fn bound_degree(
    a: &TowerPoly,
    b: &TowerPoly,
    c: &[TowerPoly],
    de: &mut DifferentialExtension,
    case: Option<Case>,
) -> PolyResult<Degree> {
    let case: Case = case.unwrap_or(de.case);
    let (a, b) = (in_t(a, de)?, in_t(b, de)?);
    let t: Basic = de.t.clone();
    let da: i64 = a.degree(Some(&t));
    let db: i64 = b.degree(Some(&t));
    let dc: i64 = c.iter().map(|i| i.degree(Some(&t))).max().unwrap_or(-1);
    let alpha: RationalFunction = b.LC().neg().div(&a.LC())?;
    let alpha_int: Option<i64> = alpha.as_int().and_then(|v| i64::try_from(&v).ok());
    let n: i64 = match case {
        Case::Base => {
            let mut n: i64 = 0.max(dc - db.max(da - 1));
            if db == da - 1 {
                if let Some(ai) = alpha_int {
                    n = 0.max(ai).max(dc - db);
                }
            }
            n
        }
        Case::Primitive => {
            let mut n: i64 = if db > da { 0.max(dc - db) } else { 0.max(dc - da + 1) };
            let lower: Basic = de.T[de.T.len() - 2 + (de.level + 1) as usize].clone();
            let (etaa, etad) = frac_in_pair(&de.d, &TowerPoly::one(&de.d.gen), &lower, false)?;
            let t1: Basic = de.t.clone();
            let found: Option<i64> = below(de, |de| {
                let (alphaa, alphad) = frac_in(&alpha, &de.t, false)?;
                if db == da - 1 {
                    let found =
                        match super::prde::limited_integrate(&alphaa, &alphad, &[(etaa.clone(), etad.clone())], de) {
                            Ok(v) => v,
                            Err(e) if super::risch::is_non_elementary(&e) => None,
                            Err(e) => return Err(e),
                        };
                    if let Some((_, m_list)) = found {
                        return length_one(&m_list);
                    }
                } else if db == da {
                    if let Some((aa, z)) = super::prde::is_log_deriv_k_t_radical_in_field(&alphaa, &alphad, de, None)? {
                        if aa == 1 {
                            let dz: RationalFunction = derivation_element(&z, de, false)?;
                            let beta: RationalFunction = a
                                .mul_ground(&dz)
                                .add(&b.mul_ground(&z))?
                                .as_poly_strict(&t1)?
                                .LC()
                                .neg()
                                .div(&z.mul(&a.LC())?)?;
                            let (betaa, betad) = frac_in(&beta, &de.t, false)?;
                            let found = match super::prde::limited_integrate(
                                &betaa,
                                &betad,
                                &[(etaa.clone(), etad.clone())],
                                de,
                            ) {
                                Ok(v) => v,
                                Err(e) if super::risch::is_non_elementary(&e) => None,
                                Err(e) => return Err(e),
                            };
                            if let Some((_, m_list)) = found {
                                return length_one(&m_list);
                            }
                        }
                    }
                }
                Ok(None)
            })?;
            if let Some(m) = found {
                n = n.max(m);
            }
            n
        }
        Case::Exp => {
            let mut n: i64 = 0.max(dc - db.max(da));
            if da == db {
                let lower: Basic = de.T[de.T.len() - 2 + (de.level + 1) as usize].clone();
                let eta: TowerPoly = de.d.quo(&t_poly(de))?;
                let (etaa, etad) = frac_in_pair(&eta, &TowerPoly::one(&eta.gen), &lower, false)?;
                let found = below(de, |de| {
                    let (alphaa, alphad) = frac_in(&alpha, &de.t, false)?;
                    super::prde::parametric_log_deriv(&alphaa, &alphad, &etaa, &etad, de)
                })?;
                if let Some((a_found, m_found, _)) = found {
                    if a_found == 1 {
                        n = n.max(m_found);
                    }
                }
            }
            n
        }
        Case::Tan | Case::OtherNonlinear => {
            let delta: i64 = de.d.degree(Some(&t));
            let lam: RationalFunction = de.d.LC();
            let alpha_int: Option<i64> = alpha.div(&lam)?.as_int().and_then(|v| i64::try_from(&v).ok());
            let mut n: i64 = 0.max(dc - (da + delta - 1).max(db));
            if db == da + delta - 1 {
                if let Some(ai) = alpha_int {
                    n = 0.max(ai).max(dc - db);
                }
            }
            n
        }
        Case::OtherLinear => {
            return Err(PolyError::NotImplemented(
                "The other_linear case is not implemented in bound_degree().".to_string(),
            ));
        }
    };
    Ok(n)
}

/// Rothstein's special polynomial differential equation algorithm
/// (`spde`): `(B, C, m, alpha, beta)`.
pub fn spde(
    a: &TowerPoly,
    b: &TowerPoly,
    c: &TowerPoly,
    n: Degree,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly, Degree, TowerPoly, TowerPoly)> {
    let (mut a, mut b, mut c) = (in_t(a, de)?, in_t(b, de)?, in_t(c, de)?);
    let mut n: Degree = n;
    let zero: TowerPoly = TowerPoly::zero(&de.t);
    let mut alpha: TowerPoly = TowerPoly::one(&de.t);
    let mut beta: TowerPoly = TowerPoly::zero(&de.t);
    let t: Basic = de.t.clone();
    let mut oo_passes: i64 = 4 * (a.degree(Some(&t)) + b.degree(Some(&t)) + c.degree(Some(&t)) + 4);
    loop {
        crate::timeout::check()?;
        if c.is_zero() {
            return Ok((zero.clone(), zero.clone(), 0, zero, beta));
        }
        if n < 0 {
            return Err(non_elementary("n became negative in spde()."));
        }
        let g: TowerPoly = a.gcd(&b)?;
        if !c.rem(&g)?.is_zero() {
            return Err(non_elementary("gcd(a, b) does not divide c in spde()."));
        }
        a = a.quo(&g)?;
        b = b.quo(&g)?;
        c = c.quo(&g)?;
        if a.degree(Some(&t)) == 0 {
            b = b.quo(&a)?;
            c = c.quo(&a)?;
            return Ok((b, c, n, alpha, beta));
        }
        if is_oo(n) {
            oo_passes -= 1;
            if oo_passes < 0 {
                return Err(PolyError::NotImplemented(
                    "spde() ran out of passes with an infinite degree bound; cannot decide solvability.".to_string(),
                ));
            }
        }
        let (r, z) = gcdex_diophantine(&b, &a, &c)?;
        b = b.add(&derivation(&a, de, false)?)?;
        c = z.sub(&derivation(&r, de, false)?)?;
        n -= a.degree(Some(&t));
        beta = beta.add(&alpha.mul(&r)?)?;
        alpha = alpha.mul(&a)?;
    }
}

/// `Dq + b q = c` with `deg b` large (`no_cancel_b_large`).
pub fn no_cancel_b_large(b: &TowerPoly, c: &TowerPoly, n: Degree, de: &DifferentialExtension) -> PolyResult<TowerPoly> {
    let (b, mut c) = (in_t(b, de)?, in_t(c, de)?);
    let mut q: TowerPoly = TowerPoly::zero(&de.t);
    let mut n: Degree = n;
    let t: Basic = de.t.clone();
    while !c.is_zero() {
        crate::timeout::check()?;
        let m: i64 = c.degree(Some(&t)) - b.degree(Some(&t));
        if !(0 <= m && m <= n) {
            return Err(non_elementary(
                "deg(c) - deg(b) is not between 0 and n in no_cancel_b_large().",
            ));
        }
        let p: TowerPoly = TowerPoly::monomial(&t, m as usize, &c.LC().div(&b.LC())?);
        q = q.add(&p)?;
        n = m - 1;
        c = c.sub(&derivation(&p, de, false)?)?.sub(&b.mul(&p)?)?;
    }
    Ok(q)
}

/// The outcomes of the non-cancellation cases: a solution, or a
/// solution so far with the equation left.
pub enum NoCancel {
    Done(TowerPoly),
    Small(TowerPoly, TowerPoly, TowerPoly),
    Equal(TowerPoly, Degree, TowerPoly),
}

/// `Dq + b q = c` with `deg b` small (`no_cancel_b_small`).
pub fn no_cancel_b_small(b: &TowerPoly, c: &TowerPoly, n: Degree, de: &DifferentialExtension) -> PolyResult<NoCancel> {
    let (b, mut c) = (in_t(b, de)?, in_t(c, de)?);
    let mut q: TowerPoly = TowerPoly::zero(&de.t);
    let mut n: Degree = n;
    let t: Basic = de.t.clone();
    while !c.is_zero() {
        crate::timeout::check()?;
        let m: i64 = if n == 0 {
            0
        } else {
            c.degree(Some(&t)) - de.d.degree(Some(&t)) + 1
        };
        if !(0 <= m && m <= n) {
            return Err(non_elementary("m is not between 0 and n in no_cancel_b_small()."));
        }
        let p: TowerPoly = if m > 0 {
            TowerPoly::monomial(&t, m as usize, &c.LC().div(&de.d.LC().mul(&rf_int(m))?)?)
        } else {
            if b.degree(Some(&t)) != c.degree(Some(&t)) {
                return Err(non_elementary(
                    "deg(b) != deg(c) in the m == 0 case of no_cancel_b_small().",
                ));
            }
            if b.degree(Some(&t)) == 0 {
                return Ok(NoCancel::Small(q, b, c));
            }
            TowerPoly::ground(&c.LC().div(&b.LC())?, &t)?
        };
        q = q.add(&p)?;
        n = m - 1;
        c = c.sub(&derivation(&p, de, false)?)?.sub(&b.mul(&p)?)?;
    }
    Ok(NoCancel::Done(q))
}

/// `Dq + b q = c` with `deg b == deg Dt - 1` (`no_cancel_equal`).
pub fn no_cancel_equal(b: &TowerPoly, c: &TowerPoly, n: Degree, de: &DifferentialExtension) -> PolyResult<NoCancel> {
    let (b, mut c) = (in_t(b, de)?, in_t(c, de)?);
    let mut q: TowerPoly = TowerPoly::zero(&de.t);
    let mut n: Degree = n;
    let t: Basic = de.t.clone();
    let lc: Option<i64> = b
        .LC()
        .neg()
        .div(&de.d.LC())?
        .as_int()
        .and_then(|v| i64::try_from(&v).ok());
    let big_m: i64 = match lc {
        Some(l) if l > 0 && l <= n => l,
        _ => -1,
    };
    while !c.is_zero() {
        crate::timeout::check()?;
        let m: i64 = big_m.max(c.degree(Some(&t)) - de.d.degree(Some(&t)) + 1);
        if !(0 <= m && m <= n) {
            return Err(non_elementary("m is not between 0 and n in no_cancel_equal()."));
        }
        let u: RationalFunction = de.d.LC().mul(&rf_int(m))?.add(&b.LC())?;
        if u.is_zero() {
            return Ok(NoCancel::Equal(q, m, c));
        }
        let p: TowerPoly = if m > 0 {
            TowerPoly::monomial(&t, m as usize, &c.LC().div(&u)?)
        } else {
            if c.degree(Some(&t)) != de.d.degree(Some(&t)) - 1 {
                return Err(non_elementary(
                    "deg(c) != deg(D) - 1 in the m == 0 case of no_cancel_equal().",
                ));
            }
            TowerPoly::ground(&c.LC().div(&b.LC())?, &t)?
        };
        q = q.add(&p)?;
        n = m - 1;
        c = c.sub(&derivation(&p, de, false)?)?.sub(&b.mul(&p)?)?;
    }
    Ok(NoCancel::Done(q))
}

/// The primitive cancellation case (`cancel_primitive`).
pub fn cancel_primitive(
    b: &TowerPoly,
    c: &TowerPoly,
    n: Degree,
    de: &mut DifferentialExtension,
) -> PolyResult<TowerPoly> {
    let (b, mut c) = (in_t(b, de)?, in_t(c, de)?);
    let mut n: Degree = n;
    let (ba, bd, found) = below(de, |de| {
        let (ba, bd) = frac_in_pair(&b, &TowerPoly::one(&b.gen), &de.t, false)?;
        let found = super::prde::is_log_deriv_k_t_radical_in_field(&ba, &bd, de, None)?;
        Ok((ba, bd, found))
    })?;
    let t: Basic = de.t.clone();
    if let Some((m, z)) = found {
        if m == 1 {
            let (zca, zcd) = frac_in(&z.mul(&c.as_element()?)?, &t, true)?;
            let Some((pa, pd)) = super::prde::is_deriv_in_field(&zca, &zcd, de)? else {
                return Err(non_elementary(
                    "z*c is not the derivative of an element of k(t) in cancel_primitive().",
                ));
            };
            let Some(found) = TowerPoly::from_element(&pa.as_element()?.div(&pd.as_element()?.mul(&z)?)?, &t)? else {
                return Err(non_elementary("p/z is not in k[t] in cancel_primitive()."));
            };
            if found.degree(Some(&t)) > n {
                return Err(non_elementary("deg(p/z) > n in cancel_primitive()."));
            }
            return Ok(found);
        }
    }
    if c.is_zero() {
        return Ok(c);
    }
    if n < c.degree(Some(&t)) {
        return Err(non_elementary("n < deg(c) in cancel_primitive()."));
    }
    let mut q: TowerPoly = TowerPoly::zero(&t);
    while !c.is_zero() {
        crate::timeout::check()?;
        let m: i64 = c.degree(Some(&t));
        if n < m {
            return Err(non_elementary("n < deg(c) in cancel_primitive()."));
        }
        let lc: RationalFunction = c.LC();
        let (sa, sd) = below(de, |de| {
            let (a2a, a2d) = frac_in(&lc, &de.t, false)?;
            risch_de(&ba, &bd, &a2a, &a2d, de)
        })?;
        let stm: TowerPoly = TowerPoly::monomial(&t, m as usize, &sa.as_element()?.div(&sd.as_element()?)?);
        q = q.add(&stm)?;
        n = m - 1;
        c = c.sub(&b.mul(&stm)?.add(&derivation(&stm, de, false)?)?)?;
    }
    Ok(q)
}

/// The hyperexponential cancellation case (`cancel_exp`).
pub fn cancel_exp(b: &TowerPoly, c: &TowerPoly, n: Degree, de: &mut DifferentialExtension) -> PolyResult<TowerPoly> {
    let (b, mut c) = (in_t(b, de)?, in_t(c, de)?);
    let mut n: Degree = n;
    let eta: RationalFunction = de.d.quo(&t_poly(de))?.as_element()?;
    let (etaa, etad, found) = below(de, |de| {
        let (etaa, etad) = frac_in(&eta, &de.t, false)?;
        let (ba, bd) = frac_in_pair(&b, &TowerPoly::one(&b.gen), &de.t, false)?;
        let found = super::prde::parametric_log_deriv(&ba, &bd, &etaa, &etad, de)?;
        Ok((etaa, etad, found))
    })?;
    let t: Basic = de.t.clone();
    let tf: RationalFunction = RationalFunction::generator(&t);
    if let Some((a, m, z)) = found {
        if a == 1 {
            let (pa, pd) = frac_in(&z.mul(&tf.pow(m)?)?.mul(&c.as_element()?)?, &t, true)?;
            let Some((va, vd)) = super::prde::is_deriv_in_field(&pa, &pd, de)? else {
                return Err(non_elementary(
                    "z*t**m*c is not the derivative of an element of k(t) in cancel_exp().",
                ));
            };
            let mut p: RationalFunction = va.as_element()?.div(&vd.as_element()?)?;
            if m > 0 {
                p = p.sub(&p.subs(&t, &RationalFunction::zero())?)?;
            }
            let Some(mut q0) = TowerPoly::from_element(&p.div(&z.mul(&tf.pow(m)?)?)?, &t)? else {
                return Err(non_elementary("p/(z*t**m) is not in k[t] in cancel_exp()."));
            };
            if q0.degree(Some(&t)) > n {
                if m < 0 && q0.degree(Some(&t)) == -m && !z.mul(&q0.nth(-m))?.has(&de.T) {
                    q0 = q0.sub(&TowerPoly::monomial(&t, (-m) as usize, &q0.nth(-m)))?;
                }
                if q0.degree(Some(&t)) > n {
                    return Err(non_elementary("deg(p/(z*t**m)) > n in cancel_exp()."));
                }
            }
            return Ok(q0);
        }
    }
    if c.is_zero() {
        return Ok(c);
    }
    if n < c.degree(Some(&t)) {
        return Err(non_elementary("n < deg(c) in cancel_exp()."));
    }
    let mut q: TowerPoly = TowerPoly::zero(&t);
    while !c.is_zero() {
        crate::timeout::check()?;
        let m: i64 = c.degree(Some(&t));
        if n < m {
            return Err(non_elementary("n < deg(c) in cancel_exp()."));
        }
        let a1: RationalFunction = b.as_element()?;
        let lc: RationalFunction = c.LC();
        let (sa, sd) = below(de, |de| {
            let (a1a, a1d) = frac_in(&a1, &de.t, false)?;
            let a1a: TowerPoly = a1a.mul(&etad)?.add(&etaa.mul(&a1d)?.mul_ground(&rf_int(m)))?;
            let a1d: TowerPoly = a1d.mul(&etad)?;
            let (a2a, a2d) = frac_in(&lc, &de.t, false)?;
            risch_de(&a1a, &a1d, &a2a, &a2d, de)
        })?;
        let stm: TowerPoly = TowerPoly::monomial(&t, m as usize, &sa.as_element()?.div(&sd.as_element()?)?);
        q = q.add(&stm)?;
        n = m - 1;
        c = c.sub(&b.mul(&stm)?.add(&derivation(&stm, de, false)?)?)?;
    }
    Ok(q)
}

/// The hypertangent cancellation case (`cancel_tan`).
pub fn cancel_tan(b0: &TowerPoly, c: &TowerPoly, n: Degree, de: &mut DifferentialExtension) -> PolyResult<TowerPoly> {
    let (b0, c) = (in_t(b0, de)?, in_t(c, de)?);
    if is_oo(n) {
        return Err(PolyError::Value(
            "n must be a finite integer in cancel_tan().".to_string(),
        ));
    }
    if n < 0 {
        if c.is_zero() {
            return Ok(c);
        }
        return Err(non_elementary("n < 0 and c != 0 in cancel_tan()."));
    }
    let t: Basic = de.t.clone();
    if n == 0 {
        if c.degree(Some(&t)) > 0 {
            return Err(non_elementary("deg(c) > 0 == n in cancel_tan()."));
        }
        let c0: RationalFunction = c.nth(0);
        if b0.is_zero() {
            let qe: RationalFunction = below(de, |de| {
                let (ca, cd) = frac_in(&c0, &de.t, false)?;
                let Some((va, vd)) = super::prde::is_deriv_in_field(&ca, &cd, de)? else {
                    return Err(non_elementary(
                        "c is not the derivative of an element of k in cancel_tan().",
                    ));
                };
                va.as_element()?.div(&vd.as_element()?)
            })?;
            return TowerPoly::from_element_strict(&qe, &t);
        }
        let b0e: RationalFunction = b0.as_element()?;
        let s: RationalFunction = below(de, |de| {
            let (b0a, b0d) = frac_in(&b0e, &de.t, false)?;
            let (ca, cd) = frac_in(&c0, &de.t, false)?;
            let (sa, sd) = risch_de(&b0a, &b0d, &ca, &cd, de)?;
            sa.as_element()?.div(&sd.as_element()?)
        })?;
        return TowerPoly::from_element_strict(&s, &t);
    }
    let eta: RationalFunction = de.d.exquo(&t2_1(de))?.as_element()?;
    let p: TowerPoly = t2_1(de);
    let (_, rem) = c.div(&p)?;
    let (c0, c1) = (rem.nth(0), rem.nth(1));
    let b0e: RationalFunction = b0.as_element()?;
    let (ue, ve) = below(de, |de| {
        let f1 = frac_in(&b0e, &de.t, false)?;
        let f2 = frac_in(&eta.neg().mul(&rf_int(n))?, &de.t, false)?;
        let g1 = frac_in(&c0, &de.t, false)?;
        let g2 = frac_in(&c1, &de.t, false)?;
        let (u, v) = super::cde::coupled_de_system(&f1, &f2, &g1, &g2, de)?;
        Ok((
            u.0.as_element()?.div(&u.1.as_element()?)?,
            v.0.as_element()?.div(&v.1.as_element()?)?,
        ))
    })?;
    let tf: RationalFunction = RationalFunction::generator(&t);
    let r: TowerPoly = TowerPoly::from_element_strict(&ue.add(&ve.mul(&tf)?)?, &t)?;
    let b: TowerPoly = TowerPoly::from_element_strict(&b0e.sub(&eta.mul(&tf)?.mul(&rf_int(n))?)?, &t)?;
    let cnew: TowerPoly = c.sub(&derivation(&r, de, false)?)?.sub(&b.mul(&r)?)?.exquo(&p)?;
    let h: TowerPoly = cancel_tan(&b0, &cnew, n - 2, de)?;
    p.mul(&h)?.add(&r)
}

pub(crate) fn no_cancel_equal_applies(b: &TowerPoly, n: Degree, de: &DifferentialExtension) -> PolyResult<bool> {
    let b: TowerPoly = in_t(b, de)?;
    let ratio: RationalFunction = b.LC().neg().div(&de.d.LC())?;
    if ratio.is_number() {
        let r: Option<i64> = ratio.as_int().and_then(|v| i64::try_from(&v).ok());
        return Ok(!matches!(r, Some(r) if r > 0) || Some(n) != r);
    }
    let tower: &[Basic] = &de.T;
    let inside = |g: &Basic| g.free_symbols().iter().all(|s| tower.contains(s));
    if ratio.gens.iter().all(inside) {
        return Ok(true);
    }
    if !ratio.denominator()?.has(tower) {
        let positions: Vec<usize> = ratio
            .gens
            .iter()
            .enumerate()
            .filter(|(_, g)| tower.contains(g))
            .map(|(i, _)| i)
            .collect();
        let parameters: Vec<usize> = ratio
            .gens
            .iter()
            .enumerate()
            .filter(|(_, g)| !inside(g))
            .map(|(i, _)| i)
            .collect();
        let mut grouped: Vec<(Vec<u32>, Vec<Vec<u32>>)> = Vec::new();
        for m in ratio.numer.terms.keys() {
            let key: Vec<u32> = positions.iter().map(|i| m[*i]).collect();
            let rest: Vec<u32> = parameters.iter().map(|i| m[*i]).collect();
            match grouped.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1.push(rest),
                None => grouped.push((key, vec![rest])),
            }
        }
        for (tower_part, rest) in grouped {
            if tower_part.iter().any(|e| *e != 0) && rest.iter().all(|r| r.iter().all(|e| *e == 0)) {
                return Ok(true);
            }
        }
    }
    Err(PolyError::Type(
        "-lc(b)/lc(Dt) must be a number or an element of k that stays nonconstant under every specialization of the parameters in solve_poly_rde()."
            .to_string(),
    ))
}

/// The polynomial solution of `Dq + b q = c` of degree at most `n`
/// (`solve_poly_rde`).
pub fn solve_poly_rde(
    b: &TowerPoly,
    c: &TowerPoly,
    n: Degree,
    de: &mut DifferentialExtension,
) -> PolyResult<TowerPoly> {
    let (b, c) = (in_t(b, de)?, in_t(c, de)?);
    let t: Basic = de.t.clone();
    let dd: i64 = de.d.degree(Some(&t));
    let bd: i64 = b.degree(Some(&t));
    if !b.is_zero() && (de.case == Case::Base || bd > 0.max(dd - 1)) {
        return no_cancel_b_large(&b, &c, n, de);
    }
    if (b.is_zero() || bd < dd - 1) && (de.case == Case::Base || dd >= 2) {
        return match no_cancel_b_small(&b, &c, n, de)? {
            NoCancel::Done(q) => Ok(q),
            NoCancel::Small(h, b0, c0) => {
                if de.case == Case::Base {
                    return h.add(&TowerPoly::ground(&c0.as_element()?.div(&b0.as_element()?)?, &t)?);
                }
                let (b0e, c0e) = (b0.as_element()?, c0.as_element()?);
                let y: RationalFunction = below(de, |de| {
                    let (ba, bd) = frac_in(&b0e, &de.t, false)?;
                    let (ca, cd) = frac_in(&c0e, &de.t, false)?;
                    let (ya, yd) = risch_de(&ba, &bd, &ca, &cd, de)?;
                    ya.as_element()?.div(&yd.as_element()?)
                })?;
                h.add(&TowerPoly::from_element_strict(&y, &t)?)
            }
            _ => unreachable!(),
        };
    }
    if dd >= 2 && bd == dd - 1 && no_cancel_equal_applies(&b, n, de)? {
        return match no_cancel_equal(&b, &c, n, de)? {
            NoCancel::Equal(h, m, big_c) => h.add(&solve_poly_rde(&b, &big_c, m, de)?),
            NoCancel::Done(q) => Ok(q),
            NoCancel::Small(..) => unreachable!(),
        };
    }
    if b.is_zero() {
        let Some((va, vd)) = super::prde::is_deriv_in_field(&c, &TowerPoly::one(&t), de)? else {
            return Err(non_elementary(
                "c is not the derivative of an element of k(t) in solve_poly_rde().",
            ));
        };
        let Some(q) = TowerPoly::from_element(&va.as_element()?.div(&vd.as_element()?)?, &t)? else {
            return Err(non_elementary(
                "The antiderivative of c is not in k[t] in solve_poly_rde().",
            ));
        };
        if q.degree(Some(&t)) > n {
            return Err(non_elementary(
                "The antiderivative of c has degree > n in solve_poly_rde().",
            ));
        }
        return Ok(q);
    }
    match de.case {
        Case::Exp => cancel_exp(&b, &c, n, de),
        Case::Primitive => cancel_primitive(&b, &c, n, de),
        Case::Tan => {
            let eta: RationalFunction = de.d.exquo(&t2_1(de))?.as_element()?;
            if !is_oo(n) && bd <= 1 && b.nth(1).add(&eta.mul(&rf_int(n))?)?.is_zero() {
                return cancel_tan(&TowerPoly::ground(&b.nth(0), &t)?, &c, n, de);
            }
            Err(PolyError::NotImplemented(
                "The tangent cancellation case requires b == b0 - n*eta*t with b0 in k in solve_poly_rde()."
                    .to_string(),
            ))
        }
        other => Err(PolyError::NotImplemented(format!(
            "Other Poly (P)RDE cancellation cases are not yet implemented ({}).",
            other.name()
        ))),
    }
}

/// A solution `(ya, yd)` of `Dy + f y = g` in the field (`rischDE`); the
/// error of `NonElementaryIntegralException` when there is none.
pub fn risch_de(
    fa: &TowerPoly,
    fd: &TowerPoly,
    ga: &TowerPoly,
    gd: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly)> {
    let (fa, fd, ga, gd) = (in_t(fa, de)?, in_t(fd, de)?, in_t(ga, de)?, in_t(gd, de)?);
    let (q, (fa, fd)) = weak_normalizer(&fa, &fd, de)?;
    let (ga, gd) = ga.mul(&q)?.cancel(&gd)?;
    let (a, (ba, bd), (ca, cd), hn) = normal_denom(&fa, &fd, &ga, &gd, de)?;
    let (big_a, big_b, big_c, hs) = special_denom(&a, &ba, &bd, &ca, &cd, de, None)?;
    let n: Degree = match bound_degree(&big_a, &big_b, std::slice::from_ref(&big_c), de, None) {
        Ok(n) => n,
        Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => OO,
        Err(e) => return Err(e),
    };
    let (big_b, big_c, m, alpha, beta) = spde(&big_a, &big_b, &big_c, n, de)?;
    let y: TowerPoly = if big_c.is_zero() {
        big_c
    } else {
        solve_poly_rde(&big_b, &big_c, m, de)?
    };
    Ok((alpha.mul(&y)?.add(&beta)?, q.mul(&hn)?.mul(&hs)?))
}
