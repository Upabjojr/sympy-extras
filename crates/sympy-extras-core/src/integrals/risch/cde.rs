//! Port of `sympy_extras/integrals/risch/cde.py`: the coupled
//! differential system `Dy1 + f1 y1 - f2 y2 = g1`, `Dy2 + f2 y1 + f1 y2 =
//! g2` through the Risch differential equation over `k(sqrt(-1))`, its
//! parametric version, and Bronstein's cancellation cases of chapter 8.

#![allow(non_snake_case, clippy::type_complexity, clippy::too_many_arguments)]

use super::polymatrix::FieldMatrix;
use super::rde::{below, is_oo, risch_de, Degree};
use super::risch::{derivation, frac_in, in_t, non_elementary, DifferentialExtension};
use super::tower::{RationalFunction, TowerPoly};
use crate::core::basic::Basic;
use crate::error::{PolyError, PolyResult};
use crate::numbers::Int;

type Fraction2 = (TowerPoly, TowerPoly);

fn imaginary() -> PolyResult<RationalFunction> {
    RationalFunction::from_expr(&Basic::imaginary_unit())
}

fn el(f: &Fraction2) -> PolyResult<RationalFunction> {
    f.0.as_element()?.div(&f.1.as_element()?)
}

fn rf_int(n: i64) -> RationalFunction {
    RationalFunction::from_int(&Int::from(n))
}

/// The real and imaginary parts of an element of `k(sqrt(-1))`
/// (`real_imag_element`).
pub fn real_imag_element(e: &RationalFunction) -> PolyResult<(RationalFunction, RationalFunction)> {
    let conj: RationalFunction = e.conjugate_i()?;
    let two: RationalFunction = rf_int(2);
    Ok((e.add(&conj)?.div(&two)?, e.sub(&conj)?.div(&imaginary()?.mul(&two)?)?))
}

fn complex_of(a: &Fraction2, b: &Fraction2, t: &Basic) -> PolyResult<Fraction2> {
    frac_in(&el(a)?.add(&imaginary()?.mul(&el(b)?)?)?, t, true)
}

/// A solution of the coupled system (`coupled_DE_system`).
pub fn coupled_de_system(
    f1: &Fraction2,
    f2: &Fraction2,
    g1: &Fraction2,
    g2: &Fraction2,
    de: &mut DifferentialExtension,
) -> PolyResult<(Fraction2, Fraction2)> {
    let (fa, fd) = complex_of(f1, f2, &de.t)?;
    let (ga, gd) = complex_of(g1, g2, &de.t)?;
    let (ya, yd) = risch_de(&fa, &fd, &ga, &gd, de)?;
    let (y1, y2) = real_imag_element(&ya.as_element()?.div(&yd.as_element()?)?)?;
    Ok((frac_in(&y1, &de.t, true)?, frac_in(&y2, &de.t, true)?))
}

/// The parametric coupled system (`param_coupled_DE_system`).
pub fn param_coupled_de_system(
    f1: &Fraction2,
    f2: &Fraction2,
    g: &[(Fraction2, Fraction2)],
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<(Fraction2, Fraction2)>, FieldMatrix)> {
    let (fa, fd) = complex_of(f1, f2, &de.t)?;
    let gc: Vec<Fraction2> = g
        .iter()
        .map(|(g1, g2)| complex_of(g1, g2, &de.t))
        .collect::<PolyResult<_>>()?;
    let (h, a) = super::prde::param_risch_de(&fa, &fd, &gc, de)?;
    let m: usize = g.len();
    let rc: usize = h.len();
    let parts: Vec<(RationalFunction, RationalFunction)> = h
        .iter()
        .map(|(ya, yd)| real_imag_element(&ya.as_element()?.div(&yd.as_element()?)?))
        .collect::<PolyResult<_>>()?;
    let mut out: Vec<(Fraction2, Fraction2)> = Vec::new();
    for (y1, y2) in &parts {
        out.push((frac_in(y1, &de.t, true)?, frac_in(y2, &de.t, true)?));
    }
    for (y1, y2) in &parts {
        out.push((frac_in(&y2.neg(), &de.t, true)?, frac_in(y1, &de.t, true)?));
    }
    let mut entries: Vec<RationalFunction> = Vec::new();
    let rows: usize = 2 * a.rows;
    let cols: usize = m + 2 * rc;
    for i in 0..rows {
        for j in 0..cols {
            let value: RationalFunction = if i < a.rows {
                if j < m + rc {
                    real_imag_element(a.entry(i as i64, j as i64)?)?.0
                } else {
                    real_imag_element(a.entry(i as i64, (j - rc) as i64)?)?.1.neg()
                }
            } else {
                let i2: usize = i - a.rows;
                if j < m + rc {
                    real_imag_element(a.entry(i2 as i64, j as i64)?)?.1
                } else {
                    real_imag_element(a.entry(i2 as i64, (j - rc) as i64)?)?.0
                }
            };
            entries.push(value);
        }
    }
    Ok((out, FieldMatrix::new(rows, cols, entries)?))
}

fn pair(p: &TowerPoly, t: &Basic) -> PolyResult<Fraction2> {
    frac_in(&p.as_element()?, t, false)
}

fn step_coupled(
    b1: &TowerPoly,
    b2: &TowerPoly,
    c1: &mut TowerPoly,
    c2: &mut TowerPoly,
    s1: &Fraction2,
    s2: &Fraction2,
    m: i64,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly)> {
    let s1tm: TowerPoly = TowerPoly::monomial(&de.t, m as usize, &el(s1)?);
    let s2tm: TowerPoly = TowerPoly::monomial(&de.t, m as usize, &el(s2)?);
    *c1 = c1.sub(
        &derivation(&s1tm, de, false)?
            .add(&b1.mul(&s1tm)?)?
            .sub(&b2.mul(&s2tm)?)?,
    )?;
    *c2 = c2.sub(
        &derivation(&s2tm, de, false)?
            .add(&b2.mul(&s1tm)?)?
            .add(&b1.mul(&s2tm)?)?,
    )?;
    Ok((s1tm, s2tm))
}

/// The primitive cancellation case of the coupled system
/// (`coupled_DE_cancel_prim`).
pub fn coupled_de_cancel_prim(
    b1: &TowerPoly,
    b2: &TowerPoly,
    c1: &TowerPoly,
    c2: &TowerPoly,
    de: &mut DifferentialExtension,
    n: Degree,
) -> PolyResult<(TowerPoly, TowerPoly)> {
    let (b1, b2, mut c1, mut c2) = (in_t(b1, de)?, in_t(b2, de)?, in_t(c1, de)?, in_t(c2, de)?);
    let mut n: Degree = n;
    let b: RationalFunction = b1.as_element()?.add(&imaginary()?.mul(&b2.as_element()?)?)?;
    let found = below(de, |de| {
        let (ba, bd) = frac_in(&b, &de.t, true)?;
        super::prde::is_log_deriv_k_t_radical_in_field(&ba, &bd, de, None)
    })?;
    let t: Basic = de.t.clone();
    if let Some((m, z)) = found {
        if m == 1 {
            let (z1, z2) = real_imag_element(&z)?;
            let (ce1, ce2) = (c1.as_element()?, c2.as_element()?);
            let mut p: Vec<RationalFunction> = Vec::new();
            for rhs in [z1.mul(&ce1)?.sub(&z2.mul(&ce2)?)?, z2.mul(&ce1)?.add(&z1.mul(&ce2)?)?] {
                let (pa, pd) = frac_in(&rhs, &t, true)?;
                let Some((va, vd)) = super::prde::is_deriv_in_field(&pa, &pd, de)? else {
                    return Err(non_elementary(
                        "z*c is not the derivative of an element of k(t) in coupled_DE_cancel_prim().",
                    ));
                };
                p.push(va.as_element()?.div(&vd.as_element()?)?);
            }
            let zz: RationalFunction = z1.mul(&z1)?.add(&z2.mul(&z2)?)?;
            let q1 = TowerPoly::from_element(&z1.mul(&p[0])?.add(&z2.mul(&p[1])?)?.div(&zz)?, &t)?;
            let q2 = TowerPoly::from_element(&z1.mul(&p[1])?.sub(&z2.mul(&p[0])?)?.div(&zz)?, &t)?;
            let (Some(q1), Some(q2)) = (q1, q2) else {
                return Err(non_elementary("p/z is not in k[t] in coupled_DE_cancel_prim()."));
            };
            if q1.degree(Some(&t)) > n || q2.degree(Some(&t)) > n {
                return Err(non_elementary("deg(p/z) > n in coupled_DE_cancel_prim()."));
            }
            return Ok((q1, q2));
        }
    }
    if c1.is_zero() && c2.is_zero() {
        return Ok((c1, c2));
    }
    if n < c1.degree(Some(&t)).max(c2.degree(Some(&t))) {
        return Err(non_elementary("n < deg(c) in coupled_DE_cancel_prim()."));
    }
    let (mut r1, mut r2) = (TowerPoly::zero(&t), TowerPoly::zero(&t));
    while !(c1.is_zero() && c2.is_zero()) {
        crate::timeout::check()?;
        let m: i64 = c1.degree(Some(&t)).max(c2.degree(Some(&t)));
        if n < m {
            return Err(non_elementary("n < deg(c) in coupled_DE_cancel_prim()."));
        }
        let (c1m, c2m) = (c1.nth(m), c2.nth(m));
        let (s1, s2) = below(de, |de| {
            let f1 = pair(&b1, &de.t)?;
            let f2 = pair(&b2, &de.t)?;
            let g1 = frac_in(&c1m, &de.t, false)?;
            let g2 = frac_in(&c2m, &de.t, false)?;
            coupled_de_system(&f1, &f2, &g1, &g2, de)
        })?;
        let (s1tm, s2tm) = step_coupled(&b1, &b2, &mut c1, &mut c2, &s1, &s2, m, de)?;
        r1 = r1.add(&s1tm)?;
        r2 = r2.add(&s2tm)?;
        n = m - 1;
    }
    Ok((r1, r2))
}

/// The hyperexponential cancellation case of the coupled system
/// (`coupled_DE_cancel_exp`).
pub fn coupled_de_cancel_exp(
    b1: &TowerPoly,
    b2: &TowerPoly,
    c1: &TowerPoly,
    c2: &TowerPoly,
    de: &mut DifferentialExtension,
    n: Degree,
) -> PolyResult<(TowerPoly, TowerPoly)> {
    let (b1, b2, mut c1, mut c2) = (in_t(b1, de)?, in_t(b2, de)?, in_t(c1, de)?, in_t(c2, de)?);
    let mut n: Degree = n;
    let t: Basic = de.t.clone();
    let eta: RationalFunction =
        de.d.quo(&TowerPoly::monomial(&t, 1, &RationalFunction::one()))?
            .as_element()?;
    let b: RationalFunction = b1.as_element()?.add(&imaginary()?.mul(&b2.as_element()?)?)?;
    let (etaa, etad, found) = below(de, |de| {
        let (etaa, etad) = frac_in(&eta, &de.t, false)?;
        let (ba, bd) = frac_in(&b, &de.t, true)?;
        let found = super::prde::parametric_log_deriv(&ba, &bd, &etaa, &etad, de)?;
        Ok((etaa, etad, found))
    })?;
    let tf: RationalFunction = RationalFunction::generator(&t);
    if let Some((a, m, z)) = found {
        if a == 1 {
            let (z1, z2) = real_imag_element(&z)?;
            let (ce1, ce2) = (c1.as_element()?, c2.as_element()?);
            let tm: RationalFunction = tf.pow(m)?;
            let mut p: Vec<RationalFunction> = Vec::new();
            for rhs in [
                z1.mul(&ce1)?.sub(&z2.mul(&ce2)?)?.mul(&tm)?,
                z2.mul(&ce1)?.add(&z1.mul(&ce2)?)?.mul(&tm)?,
            ] {
                let (pa, pd) = frac_in(&rhs, &t, true)?;
                let Some((va, vd)) = super::prde::is_deriv_in_field(&pa, &pd, de)? else {
                    return Err(non_elementary(
                        "z*t**m*c is not the derivative of an element of k(t) in coupled_DE_cancel_exp().",
                    ));
                };
                let mut pj: RationalFunction = va.as_element()?.div(&vd.as_element()?)?;
                if m > 0 {
                    pj = pj.sub(&pj.subs(&t, &RationalFunction::zero())?)?;
                }
                p.push(pj);
            }
            let zz: RationalFunction = z1.mul(&z1)?.add(&z2.mul(&z2)?)?.mul(&tm)?;
            let f1 = TowerPoly::from_element(&z1.mul(&p[0])?.add(&z2.mul(&p[1])?)?.div(&zz)?, &t)?;
            let f2 = TowerPoly::from_element(&z1.mul(&p[1])?.sub(&z2.mul(&p[0])?)?.div(&zz)?, &t)?;
            let (Some(mut q1), Some(mut q2)) = (f1, f2) else {
                return Err(non_elementary("p/(z*t**m) is not in k[t] in coupled_DE_cancel_exp()."));
            };
            if q1.degree(Some(&t)).max(q2.degree(Some(&t))) > n {
                if m < 0 && q1.degree(Some(&t)).max(q2.degree(Some(&t))) == -m {
                    let cc: RationalFunction = z.mul(&q1.nth(-m).add(&imaginary()?.mul(&q2.nth(-m))?)?)?;
                    if !cc.has(&de.T) {
                        q1 = q1.sub(&TowerPoly::monomial(&t, (-m) as usize, &q1.nth(-m)))?;
                        q2 = q2.sub(&TowerPoly::monomial(&t, (-m) as usize, &q2.nth(-m)))?;
                    }
                }
                if q1.degree(Some(&t)).max(q2.degree(Some(&t))) > n {
                    return Err(non_elementary("deg(p/(z*t**m)) > n in coupled_DE_cancel_exp()."));
                }
            }
            return Ok((q1, q2));
        }
    }
    if c1.is_zero() && c2.is_zero() {
        return Ok((c1, c2));
    }
    if n < c1.degree(Some(&t)).max(c2.degree(Some(&t))) {
        return Err(non_elementary("n < deg(c) in coupled_DE_cancel_exp()."));
    }
    let (mut r1, mut r2) = (TowerPoly::zero(&t), TowerPoly::zero(&t));
    while !(c1.is_zero() && c2.is_zero()) {
        crate::timeout::check()?;
        let m: i64 = c1.degree(Some(&t)).max(c2.degree(Some(&t)));
        if n < m {
            return Err(non_elementary("n < deg(c) in coupled_DE_cancel_exp()."));
        }
        let (c1m, c2m) = (c1.nth(m), c2.nth(m));
        let b1e: RationalFunction = b1.as_element()?;
        let (s1, s2) = below(de, |de| {
            let (b1a, b1d) = frac_in(&b1e, &de.t, false)?;
            let b1a: TowerPoly = b1a.mul(&etad)?.add(&etaa.mul(&b1d)?.mul_ground(&rf_int(m)))?;
            let b1d: TowerPoly = b1d.mul(&etad)?;
            let f2 = pair(&b2, &de.t)?;
            let g1 = frac_in(&c1m, &de.t, false)?;
            let g2 = frac_in(&c2m, &de.t, false)?;
            coupled_de_system(&(b1a, b1d), &f2, &g1, &g2, de)
        })?;
        let (s1tm, s2tm) = step_coupled(&b1, &b2, &mut c1, &mut c2, &s1, &s2, m, de)?;
        r1 = r1.add(&s1tm)?;
        r2 = r2.add(&s2tm)?;
        n = m - 1;
    }
    Ok((r1, r2))
}

/// The hypertangent cancellation case of the coupled system
/// (`coupled_DE_cancel_tan`).
pub fn coupled_de_cancel_tan(
    b0: &TowerPoly,
    b2: &TowerPoly,
    c1: &TowerPoly,
    c2: &TowerPoly,
    de: &mut DifferentialExtension,
    n: Degree,
) -> PolyResult<(TowerPoly, TowerPoly)> {
    if is_oo(n) {
        return Err(PolyError::Value(
            "n must be a finite integer in coupled_DE_cancel_tan().".to_string(),
        ));
    }
    let (b0, b2, c1, c2) = (in_t(b0, de)?, in_t(b2, de)?, in_t(c1, de)?, in_t(c2, de)?);
    let t: Basic = de.t.clone();
    if n < 0 {
        if c1.is_zero() && c2.is_zero() {
            return Ok((TowerPoly::zero(&t), TowerPoly::zero(&t)));
        }
        return Err(non_elementary("n < 0 and c != 0 in coupled_DE_cancel_tan()."));
    }
    if n == 0 {
        if c1.degree(Some(&t)) > 0 || c2.degree(Some(&t)) > 0 {
            return Err(non_elementary("deg(c) > 0 == n in coupled_DE_cancel_tan()."));
        }
        let (c10, c20) = (c1.nth(0), c2.nth(0));
        let (e1, e2) = below(de, |de| {
            let f1 = pair(&b0, &de.t)?;
            let f2 = pair(&b2, &de.t)?;
            let g1 = frac_in(&c10, &de.t, false)?;
            let g2 = frac_in(&c20, &de.t, false)?;
            let (y1, y2) = coupled_de_system(&f1, &f2, &g1, &g2, de)?;
            Ok((el(&y1)?, el(&y2)?))
        })?;
        return Ok((
            TowerPoly::from_element_strict(&e1, &t)?,
            TowerPoly::from_element_strict(&e2, &t)?,
        ));
    }
    let p2: TowerPoly = TowerPoly::from_list(
        &[
            RationalFunction::one(),
            RationalFunction::zero(),
            RationalFunction::one(),
        ],
        &t,
    );
    let eta: RationalFunction = de.d.exquo(&p2)?.as_element()?;
    let one: TowerPoly = TowerPoly::one(&t);
    let (a1, bb1, _) = super::prde::real_imag(&c1, &one, &t)?;
    let (a2, bb2, _) = super::prde::real_imag(&c2, &one, &t)?;
    let z1: RationalFunction = a1.as_element()?.sub(&bb2.as_element()?)?;
    let z2: RationalFunction = bb1.as_element()?.add(&a2.as_element()?)?;
    let b2e: RationalFunction = b2.as_element()?.sub(&eta.mul(&rf_int(n))?)?;
    let (s1, s2) = below(de, |de| {
        let f1 = pair(&b0, &de.t)?;
        let f2 = frac_in(&b2e, &de.t, false)?;
        let g1 = frac_in(&z1, &de.t, false)?;
        let g2 = frac_in(&z2, &de.t, false)?;
        let (f1s, f2s) = coupled_de_system(&f1, &f2, &g1, &g2, de)?;
        Ok((el(&f1s)?, el(&f2s)?))
    })?;
    let tf: RationalFunction = RationalFunction::generator(&t);
    let p = |e: &RationalFunction| TowerPoly::from_element_strict(e, &t);
    let en: RationalFunction = eta.mul(&rf_int(n))?;
    let n1: TowerPoly = c1.sub(&p(&z1)?)?.add(&p(&en.mul(&s1.mul(&tf)?.add(&s2)?)?)?)?;
    let n2: TowerPoly = c2.sub(&p(&z2)?)?.add(&p(&en.mul(&s2.mul(&tf)?.sub(&s1)?)?)?)?;
    let tp: TowerPoly = TowerPoly::monomial(&t, 1, &RationalFunction::one());
    let d1: TowerPoly = n1.mul(&tp)?.sub(&n2)?.exquo(&p2)?;
    let d2: TowerPoly = n2.mul(&tp)?.add(&n1)?.exquo(&p2)?;
    let (h1, h2) = coupled_de_cancel_tan(&b0, &b2.add(&p(&eta)?)?, &d1, &d2, de, n - 1)?;
    Ok((
        h1.mul(&tp)?.add(&h2)?.add(&p(&s1)?)?,
        h2.mul(&tp)?.sub(&h1)?.add(&p(&s2)?)?,
    ))
}
