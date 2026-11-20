//! Port of `sympy_extras/integrals/risch/prde.py`: the parametric Risch
//! differential equation (Bronstein, chapter 7), limited integration, the
//! parametric logarithmic derivative problem and the structure theorems
//! (`is_deriv_k`, `is_log_deriv_k_t_radical` and their tangent analogues).

#![allow(non_snake_case, clippy::type_complexity, clippy::too_many_arguments)]

use super::polymatrix::FieldMatrix;
use super::rde::{
    below, bound_degree, finite, is_oo, no_cancel_equal_applies, order_at, order_at_oo, rational_roots,
    special_denom_cancel_bound, weak_normalizer, Degree,
};
use super::risch::{
    derivation, derivation_element, frac_in, frac_in_pair, gcdex_diophantine, in_t, residue_derivation_element,
    residue_reduce, splitfactor, unique_dummy, Case, DifferentialExtension, Extension,
};
use super::tower::{RationalFunction, TowerPoly};
use crate::core::basic::Basic;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

/// A fraction of polynomials `(numerator, denominator)`.
pub type Fraction2 = (TowerPoly, TowerPoly);
/// Pairs `(term, coefficient)`.
pub type Terms = Vec<(Basic, Basic)>;

fn zero() -> RationalFunction {
    RationalFunction::zero()
}
fn one() -> RationalFunction {
    RationalFunction::one()
}
fn rf_int(n: i64) -> RationalFunction {
    RationalFunction::from_int(&Int::from(n))
}
fn el(a: &TowerPoly, d: &TowerPoly) -> PolyResult<RationalFunction> {
    a.as_element()?.div(&d.as_element()?)
}
fn poly(e: &RationalFunction, de: &DifferentialExtension) -> PolyResult<TowerPoly> {
    TowerPoly::from_element_strict(e, &de.t)
}
fn number(q: &Rat) -> Basic {
    Basic::from_rational(q.clone())
}
fn lcm_polys(polys: &[TowerPoly], start: &TowerPoly) -> PolyResult<TowerPoly> {
    let mut acc: TowerPoly = start.clone();
    for p in polys {
        acc = acc.lcm(p)?;
    }
    Ok(acc)
}
fn ilcm(a: &Int, b: &Int) -> Int {
    use dashu_base::Gcd;
    if *a == Int::ZERO || *b == Int::ZERO {
        return Int::ZERO;
    }
    let g: Int = Int::from(a.gcd(b));
    let v: Int = (a / &g) * b;
    if v < Int::ZERO {
        -v
    } else {
        v
    }
}
fn denominator(r: &Rat) -> Int {
    crate::numbers::denom(r)
}
fn numerator(r: &Rat) -> Int {
    crate::numbers::numer(r)
}
fn to_i64(n: &Int) -> PolyResult<i64> {
    i64::try_from(n).map_err(|_| PolyError::Value("an integer too large".to_string()))
}
fn t2_1(t: &Basic) -> TowerPoly {
    TowerPoly::from_list(&[one(), zero(), one()], t)
}
fn monomial_t(t: &Basic) -> TowerPoly {
    TowerPoly::monomial(t, 1, &one())
}
fn caught_not_implemented(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if !m.starts_with(crate::error::NOT_PORTED_PREFIX))
}

/// The normal part of the denominator, parametric (`prde_normal_denom`).
pub fn prde_normal_denom(
    fa: &TowerPoly,
    fd: &TowerPoly,
    g: &[Fraction2],
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, Fraction2, Vec<Fraction2>, TowerPoly)> {
    let (fa, fd) = (in_t(fa, de)?, in_t(fd, de)?);
    let (dn, _ds) = splitfactor(&fd, de, false)?;
    let gds: Vec<TowerPoly> = g.iter().map(|(_, d)| in_t(d, de)).collect::<PolyResult<_>>()?;
    let gd: TowerPoly = lcm_polys(&gds, &TowerPoly::one(&de.t))?;
    let (en, _es) = splitfactor(&gd, de, false)?;
    let p: TowerPoly = dn.gcd(&en)?;
    let h: TowerPoly = en.gcd(&en.diff(Some(&de.t))?)?.quo(&p.gcd(&p.diff(Some(&de.t))?)?)?;
    let a: TowerPoly = dn.mul(&h)?;
    let c: TowerPoly = a.mul(&h)?;
    let ba: TowerPoly = a.mul(&fa)?.sub(&dn.mul(&derivation(&h, de, false)?)?.mul(&fd)?)?;
    let (ba, bd) = ba.cancel(&fd)?;
    let mut out: Vec<Fraction2> = Vec::new();
    for (ga, gd_) in g {
        out.push(c.mul(&in_t(ga, de)?)?.cancel(&in_t(gd_, de)?)?);
    }
    Ok((a, (ba, bd), out, h))
}

/// The real and imaginary parts of `ba/bd` at `gen = sqrt(-1)`
/// (`real_imag`): `(real, imag, denominator)`.
pub fn real_imag(ba: &TowerPoly, bd: &TowerPoly, gen: &Basic) -> PolyResult<(TowerPoly, TowerPoly, TowerPoly)> {
    let parts = |p: &TowerPoly| -> PolyResult<(RationalFunction, RationalFunction)> {
        let (mut real, mut imag) = (zero(), zero());
        for (k, c) in p.as_poly_strict(gen)?.terms() {
            match k % 4 {
                0 => real = real.add(&c)?,
                2 => real = real.sub(&c)?,
                1 => imag = imag.add(&c)?,
                _ => imag = imag.sub(&c)?,
            }
        }
        Ok((real, imag))
    };
    let (bd_real, bd_imag) = parts(bd)?;
    let (ba_real, ba_imag) = parts(ba)?;
    let real_part: TowerPoly = TowerPoly::ground(&ba_real.mul(&bd_real)?.add(&ba_imag.mul(&bd_imag)?)?, gen)?;
    let imag_part: TowerPoly = TowerPoly::ground(&ba_imag.mul(&bd_real)?.sub(&ba_real.mul(&bd_imag)?)?, gen)?;
    let denom: TowerPoly = TowerPoly::ground(&bd_real.mul(&bd_real)?.add(&bd_imag.mul(&bd_imag)?)?, gen)?;
    Ok((real_part, imag_part, denom))
}

/// The special part of the denominator, parametric (`prde_special_denom`).
pub fn prde_special_denom(
    a: &TowerPoly,
    ba: &TowerPoly,
    bd: &TowerPoly,
    g: &[Fraction2],
    de: &mut DifferentialExtension,
    case: Option<Case>,
) -> PolyResult<(TowerPoly, TowerPoly, Vec<Fraction2>, TowerPoly)> {
    let case: Case = case.unwrap_or(de.case);
    let p: TowerPoly = match case {
        Case::Exp => monomial_t(&de.t),
        Case::Tan => t2_1(&de.t),
        Case::Primitive | Case::Base => {
            return Ok((a.clone(), ba.quo(bd)?, g.to_vec(), TowerPoly::one(&de.t)));
        }
        _ => {
            return Err(PolyError::NotImplemented(format!(
                "The {} case is not implemented in prde_special_denom().",
                case.name()
            )))
        }
    };
    let t: Basic = de.t.clone();
    let nb: Degree = order_at(ba, &p, &t)? - order_at(bd, &p, &t)?;
    let mut nc: Degree = super::rde::OO;
    for (ga, gd) in g {
        nc = nc.min(order_at(ga, &p, &t)? - order_at(gd, &p, &t)?);
    }
    let mut n: Degree = 0.min(nc - 0.min(nb));
    if nb == 0 {
        n = special_denom_cancel_bound(a, ba, bd, n, de, case)?;
    }
    let big_n: Degree = 0.max(-nb);
    let p_n: TowerPoly = p.pow(big_n as usize);
    let pn: TowerPoly = p.pow((-n) as usize);
    let big_a: TowerPoly = a.mul(&p_n)?;
    let big_b: TowerPoly = ba.mul(&p_n.quo(bd)?)?.add(
        &a.mul(&derivation(&p, de, false)?.quo(&p)?)?
            .mul(&p_n)?
            .mul_ground(&rf_int(n)),
    )?;
    let mut out: Vec<Fraction2> = Vec::new();
    for (ga, gd) in g {
        out.push(in_t(ga, de)?.mul(&p_n)?.mul(&pn)?.cancel(&in_t(gd, de)?)?);
    }
    Ok((big_a, big_b, out, pn))
}

/// The linear constraints on the constants for the parametric equation
/// (`prde_linear_constraints`).
pub fn prde_linear_constraints(
    g: &[Fraction2],
    de: &DifferentialExtension,
) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    let m: usize = g.len();
    let gds: Vec<TowerPoly> = g.iter().map(|(_, d)| in_t(d, de)).collect::<PolyResult<_>>()?;
    let d: TowerPoly = lcm_polys(&gds[1..], &gds[0])?;
    let mut q: Vec<(TowerPoly, TowerPoly)> = Vec::new();
    for (ga, gd) in g {
        q.push(in_t(ga, de)?.mul(&d.quo(&in_t(gd, de)?)?)?.div(&d)?);
    }
    let big_m: FieldMatrix = if !q.iter().all(|(_, r)| r.is_zero()) {
        let n: i64 = q.iter().map(|(_, r)| r.degree(Some(&de.t))).max().unwrap_or(0);
        FieldMatrix::from_function((n + 1) as usize, m, |i, j| q[j].1.nth(i as i64))
    } else {
        FieldMatrix::new(0, m, Vec::new())?
    };
    Ok((q.into_iter().map(|(a, _)| a).collect(), big_m))
}

/// The quotients by `d` and the linear constraints making the remainders
/// zero (`poly_linear_constraints`).
pub fn poly_linear_constraints(p: &[TowerPoly], d: &TowerPoly) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    let m: usize = p.len();
    let divs: Vec<(TowerPoly, TowerPoly)> = p.iter().map(|pi| pi.div(d)).collect::<PolyResult<_>>()?;
    let q: Vec<TowerPoly> = divs.iter().map(|(a, _)| a.clone()).collect();
    let r: Vec<TowerPoly> = divs.into_iter().map(|(_, b)| b).collect();
    let big_m: FieldMatrix = if !r.iter().all(|ri| ri.is_zero()) {
        let n: i64 = r.iter().map(|ri| ri.degree(None)).max().unwrap_or(0);
        FieldMatrix::from_function((n + 1) as usize, m, |i, j| r[j].nth(i as i64))
    } else {
        FieldMatrix::new(0, m, Vec::new())?
    };
    Ok((q, big_m))
}

/// The system `A x = u` made one over the constants (`constant_system`).
pub fn constant_system(
    a: &FieldMatrix,
    u: &FieldMatrix,
    de: &DifferentialExtension,
) -> PolyResult<(FieldMatrix, FieldMatrix)> {
    if a.is_empty() {
        return Ok((a.clone(), u.clone()));
    }
    let (au, _) = a.row_join(u)?.rref()?;
    let rows: Vec<usize> = (0..au.rows).collect();
    let mut big_a: FieldMatrix = au.submatrix(&rows, &(0..au.cols - 1).collect::<Vec<_>>());
    let mut big_u: FieldMatrix = au.submatrix(&rows, &[au.cols - 1]);
    let d = |x: &RationalFunction| derivation_element(x, de, false);
    let passes: usize = (au.rows + 1) * (au.cols + 1) * (de.T.len() + 1) + 10;
    let mut done: bool = false;
    for _ in 0..passes {
        crate::timeout::check()?;
        let mut found: Option<(usize, usize)> = None;
        'search: for j in 0..big_a.cols {
            for i in 0..big_a.rows {
                if big_a.entry(i as i64, j as i64)?.has(&de.T) {
                    found = Some((i, j));
                    break 'search;
                }
            }
        }
        let Some((i, j)) = found else {
            done = true;
            break;
        };
        let ri: FieldMatrix = big_a.row(i);
        let daij: RationalFunction = d(big_a.entry(i as i64, j as i64)?)?;
        let rm1: FieldMatrix = ri.applyfunc(|e| d(e)?.div(&daij))?;
        let um1: RationalFunction = d(big_u.entry(i as i64, 0)?)?.div(&daij)?;
        let aj: FieldMatrix = big_a.col(j);
        big_a = big_a.sub(&aj.mul(&rm1)?)?;
        big_u = big_u.sub(&aj.scale(&um1)?)?;
        big_a = big_a.col_join(&rm1)?;
        big_u = big_u.col_join(&FieldMatrix::column(&[um1]))?;
    }
    if !done {
        return Err(PolyError::Value(
            "constant_system() did not reach a constant system.".to_string(),
        ));
    }
    Ok((big_a, big_u))
}

/// Rothstein's SPDE, parametric (`prde_spde`).
pub fn prde_spde(
    a: &TowerPoly,
    b: &TowerPoly,
    q: &[TowerPoly],
    n: Degree,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly, Vec<TowerPoly>, Vec<TowerPoly>, Degree)> {
    let (a, b) = (in_t(a, de)?, in_t(b, de)?);
    let mut r: Vec<TowerPoly> = Vec::new();
    let mut z: Vec<TowerPoly> = Vec::new();
    for qi in q {
        let (ri, zi) = gcdex_diophantine(&b, &a, &in_t(qi, de)?)?;
        r.push(ri);
        z.push(zi);
    }
    let big_b: TowerPoly = b.add(&derivation(&a, de, false)?)?;
    let mut qq: Vec<TowerPoly> = Vec::new();
    for (ri, zi) in r.iter().zip(&z) {
        qq.push(zi.sub(&derivation(ri, de, false)?)?);
    }
    let n1: Degree = n - a.degree(Some(&de.t));
    Ok((a, big_b, qq, r, n1))
}

fn constraints(q: &[TowerPoly], de: &DifferentialExtension) -> PolyResult<FieldMatrix> {
    let m: usize = q.len();
    let dc: i64 = if q.iter().all(|qi| qi.is_zero()) {
        -1
    } else {
        q.iter().map(|qi| qi.degree(Some(&de.t))).max().unwrap_or(-1)
    };
    let big_m: FieldMatrix = FieldMatrix::from_function((dc + 1) as usize, m, |i, j| q[j].nth(i as i64));
    Ok(constant_system(&big_m, &FieldMatrix::zeros((dc + 1) as usize, 1), de)?.0)
}

fn with_identity(a: FieldMatrix, m: usize) -> PolyResult<FieldMatrix> {
    let c: FieldMatrix = FieldMatrix::eye(m);
    a.row_join(&FieldMatrix::zeros(a.rows, m))?
        .col_join(&c.row_join(&c.neg())?)
}

/// `deg b` large, parametric (`prde_no_cancel_b_large`).
pub fn prde_no_cancel_b_large(
    b: &TowerPoly,
    q: &[TowerPoly],
    n: Degree,
    de: &DifferentialExtension,
) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    let b: TowerPoly = in_t(b, de)?;
    let mut q: Vec<TowerPoly> = q.iter().map(|qi| in_t(qi, de)).collect::<PolyResult<_>>()?;
    let db: i64 = b.degree(Some(&de.t));
    let m: usize = q.len();
    let mut h: Vec<TowerPoly> = vec![TowerPoly::zero(&de.t); m];
    for big_n in (0..=finite(n)?).rev() {
        for i in 0..m {
            let si: RationalFunction = q[i].nth(big_n + db).div(&b.LC())?;
            let sitn: TowerPoly = TowerPoly::monomial(&de.t, big_n as usize, &si);
            h[i] = h[i].add(&sitn)?;
            q[i] = q[i].sub(&derivation(&sitn, de, false)?)?.sub(&b.mul(&sitn)?)?;
        }
    }
    let a: FieldMatrix = constraints(&q, de)?;
    Ok((h, with_identity(a, m)?))
}

/// `deg b` small, parametric (`prde_no_cancel_b_small`).
pub fn prde_no_cancel_b_small(
    b: &TowerPoly,
    q: &[TowerPoly],
    n: Degree,
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    let b: TowerPoly = in_t(b, de)?;
    let mut q: Vec<TowerPoly> = q.iter().map(|qi| in_t(qi, de)).collect::<PolyResult<_>>()?;
    let m: usize = q.len();
    let t: Basic = de.t.clone();
    let mut h: Vec<TowerPoly> = vec![TowerPoly::zero(&t); m];
    for big_n in (1..=finite(n)?).rev() {
        for i in 0..m {
            let si: RationalFunction = q[i]
                .nth(big_n + de.d.degree(Some(&t)) - 1)
                .div(&de.d.LC().mul(&rf_int(big_n))?)?;
            let sitn: TowerPoly = TowerPoly::monomial(&t, big_n as usize, &si);
            h[i] = h[i].add(&sitn)?;
            q[i] = q[i].sub(&derivation(&sitn, de, false)?)?.sub(&b.mul(&sitn)?)?;
        }
    }
    if b.degree(Some(&t)) > 0 {
        for i in 0..m {
            let si_p: TowerPoly = TowerPoly::ground(&q[i].nth(b.degree(Some(&t))).div(&b.LC())?, &t)?;
            h[i] = h[i].add(&si_p)?;
            q[i] = q[i].sub(&derivation(&si_p, de, false)?)?.sub(&b.mul(&si_p)?)?;
        }
        let a: FieldMatrix = constraints(&q, de)?;
        return Ok((h, with_identity(a, m)?));
    }
    let (f_polys, big_b): (Vec<TowerPoly>, FieldMatrix) = if de.case != Case::Base {
        let (f, big_b) = below(de, |de| {
            let t0: Basic = de.t.clone();
            let (ba, bd) = frac_in_pair(&b, &TowerPoly::one(&b.gen), &t0, false)?;
            let q0: Vec<Fraction2> = q
                .iter()
                .map(|qi| frac_in(&qi.TC(), &t0, false))
                .collect::<PolyResult<_>>()?;
            param_risch_de(&ba, &bd, &q0, de)
        })?;
        let polys: Vec<TowerPoly> = f
            .iter()
            .map(|(fa, fd)| TowerPoly::from_element_strict(&el(fa, fd)?, &t))
            .collect::<PolyResult<_>>()?;
        (polys, big_b)
    } else {
        let mut row: Vec<RationalFunction> = q.iter().map(|qi| qi.TC()).collect();
        row.push(zero());
        (vec![TowerPoly::one(&t)], FieldMatrix::row_vector(&row))
    };
    let d: i64 = q.iter().map(|qi| qi.degree(Some(&t))).max().unwrap_or(-1);
    let a: FieldMatrix = if d > 0 {
        let big_m: FieldMatrix = FieldMatrix::from_function(d as usize, m, |i, j| q[j].nth(i as i64 + 1));
        constant_system(&big_m, &FieldMatrix::zeros(d as usize, 1), de)?.0
    } else {
        FieldMatrix::new(0, m, Vec::new())?
    };
    let r: usize = f_polys.len();
    let im: FieldMatrix = FieldMatrix::eye(m);
    let a: FieldMatrix = a.row_join(&FieldMatrix::zeros(a.rows, r + m))?;
    let big_b: FieldMatrix = big_b.row_join(&FieldMatrix::zeros(big_b.rows, m))?;
    let c: FieldMatrix = im.row_join(&FieldMatrix::zeros(m, r))?.row_join(&im.neg())?;
    let mut all: Vec<TowerPoly> = f_polys;
    all.extend(h);
    Ok((all, a.col_join(&big_b)?.col_join(&c)?))
}

/// `deg b == deg Dt - 1`, parametric (`prde_no_cancel_b_equal`).
pub fn prde_no_cancel_b_equal(
    b: &TowerPoly,
    q: &[TowerPoly],
    n: Degree,
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    let b: TowerPoly = in_t(b, de)?;
    let m: usize = q.len();
    let mut q: Vec<TowerPoly> = q.iter().map(|qi| in_t(qi, de)).collect::<PolyResult<_>>()?;
    let t: Basic = de.t.clone();
    let delta: i64 = de.d.degree(Some(&t));
    let lam: RationalFunction = de.d.LC();
    let mc: Option<i64> = b.LC().neg().div(&lam)?.as_int().and_then(|v| i64::try_from(&v).ok());
    let big_m: i64 = match mc {
        Some(v) if v > 0 && v <= n => v,
        _ => -1,
    };
    let mut h: Vec<TowerPoly> = vec![TowerPoly::zero(&t); m];
    let mut big_n: i64 = finite(n)?;
    while big_n > big_m {
        let u: RationalFunction = lam.mul(&rf_int(big_n))?.add(&b.LC())?;
        for i in 0..m {
            let si: RationalFunction = q[i].nth(big_n + delta - 1).div(&u)?;
            let sitn: TowerPoly = TowerPoly::monomial(&t, big_n as usize, &si);
            h[i] = h[i].add(&sitn)?;
            q[i] = q[i].sub(&derivation(&sitn, de, false)?)?.sub(&b.mul(&sitn)?)?;
        }
        big_n -= 1;
    }
    if big_m == -1 {
        let a: FieldMatrix = constraints(&q, de)?;
        return Ok((h, with_identity(a, m)?));
    }
    let (f, big_b) = param_poly_risch_de(&TowerPoly::one(&t), &b, &q, big_m, de)?;
    let r: usize = f.len();
    let bc: FieldMatrix = FieldMatrix::from_function(big_b.rows, m, |i, j| {
        big_b.entry(i as i64, j as i64).cloned().unwrap_or_else(|_| zero())
    });
    let bd: FieldMatrix = FieldMatrix::from_function(big_b.rows, r, |i, j| {
        big_b
            .entry(i as i64, (m + j) as i64)
            .cloned()
            .unwrap_or_else(|_| zero())
    });
    let top: FieldMatrix = FieldMatrix::eye(m)
        .row_join(&FieldMatrix::eye(m).neg())?
        .row_join(&FieldMatrix::zeros(m, r))?;
    let bottom: FieldMatrix = bc.row_join(&FieldMatrix::zeros(big_b.rows, m))?.row_join(&bd)?;
    let mut all: Vec<TowerPoly> = h;
    all.extend(f);
    Ok((all, top.col_join(&bottom)?))
}

/// The Liouvillian cancellation case, parametric (`prde_cancel_liouvillian`).
pub fn prde_cancel_liouvillian(
    b: &TowerPoly,
    q: &[TowerPoly],
    n: Degree,
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    if de.case != Case::Primitive && de.case != Case::Exp {
        return Err(PolyError::Value(format!(
            "case must be 'primitive' or 'exp', not {}.",
            de.case.name()
        )));
    }
    let b: TowerPoly = in_t(b, de)?;
    let mut q: Vec<TowerPoly> = q.iter().map(|qi| in_t(qi, de)).collect::<PolyResult<_>>()?;
    let mut h: Vec<TowerPoly> = Vec::new();
    let t: Basic = de.t.clone();
    let eta: RationalFunction = if de.case == Case::Exp {
        de.d.quo(&monomial_t(&t))?.as_element()?
    } else {
        zero()
    };
    let (mut ba, mut bd) = below(de, |de| frac_in_pair(&b, &TowerPoly::one(&b.gen), &de.t, false))?;
    let mut big_m: FieldMatrix = FieldMatrix::eye(0);
    let nn: i64 = finite(n)?;
    for i in (0..=nn).rev() {
        crate::timeout::check()?;
        if de.case == Case::Exp {
            let be: RationalFunction = b.as_element()?.add(&eta.mul(&rf_int(i))?)?;
            let (x, y) = below(de, |de| frac_in(&be, &de.t, false))?;
            ba = x;
            bd = y;
        }
        let coefficients: Vec<RationalFunction> = q.iter().map(|qq| qq.nth(i)).collect();
        let (fi_fractions, ai) = below(de, |de| {
            let qy: Vec<Fraction2> = coefficients
                .iter()
                .map(|c| frac_in(c, &de.t, false))
                .collect::<PolyResult<_>>()?;
            param_risch_de(&ba, &bd, &qy, de)
        })?;
        let fi: Vec<TowerPoly> = fi_fractions
            .iter()
            .map(|(fa, fd)| TowerPoly::from_element_strict(&el(fa, fd)?, &t))
            .collect::<PolyResult<_>>()?;
        let ri: usize = fi.len();
        big_m = if i == nn {
            ai
        } else {
            ai.col_join(&big_m.row_join(&FieldMatrix::zeros(big_m.rows, ri))?)?
        };
        let mut fi_new: Vec<TowerPoly> = Vec::new();
        for f in fi {
            let hji: TowerPoly = f.mul(&TowerPoly::monomial(&t, i as usize, &one()))?;
            fi_new.push(derivation(&hji, de, false)?.add(&b.mul(&hji)?)?.neg());
            h.push(hji);
        }
        q.extend(fi_new);
    }
    Ok((h, big_m))
}

/// The hypertangent cancellation case, parametric (`prde_cancel_tan`).
pub fn prde_cancel_tan(
    b0: &TowerPoly,
    q: &[TowerPoly],
    n: Degree,
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    let m: usize = q.len();
    if is_oo(n) {
        return Err(PolyError::Value(
            "n must be a finite integer in prde_cancel_tan().".to_string(),
        ));
    }
    let b0: TowerPoly = in_t(b0, de)?;
    let q: Vec<TowerPoly> = q.iter().map(|qi| in_t(qi, de)).collect::<PolyResult<_>>()?;
    let t: Basic = de.t.clone();
    if n < 0 {
        if q.iter().all(|qi| qi.is_zero()) {
            return Ok((Vec::new(), FieldMatrix::zeros(1, m)));
        }
        let big_n: i64 = q.iter().map(|qi| qi.degree(Some(&t))).max().unwrap_or(0);
        let big_m: FieldMatrix = FieldMatrix::from_function((big_n + 1) as usize, m, |i, j| q[j].nth(i as i64));
        let (a, _) = constant_system(&big_m, &FieldMatrix::zeros(big_m.rows, 1), de)?;
        return Ok((Vec::new(), a));
    }
    if n == 0 {
        let b0e: RationalFunction = b0.as_element()?;
        let coefficients: Vec<RationalFunction> = q.iter().map(|qi| qi.nth(0)).collect();
        let (f, a0) = below(de, |de| {
            let (b0a, b0d) = frac_in(&b0e, &de.t, false)?;
            let qy: Vec<Fraction2> = coefficients
                .iter()
                .map(|c| frac_in(c, &de.t, false))
                .collect::<PolyResult<_>>()?;
            param_risch_de(&b0a, &b0d, &qy, de)
        })?;
        let h: Vec<TowerPoly> = f
            .iter()
            .map(|(fa, fd)| TowerPoly::from_element_strict(&el(fa, fd)?, &t))
            .collect::<PolyResult<_>>()?;
        let r: usize = h.len();
        let mut a: FieldMatrix = a0;
        let big_n: i64 = q.iter().map(|qi| qi.degree(Some(&t))).max().unwrap_or(0);
        if big_n > 0 {
            let big_m: FieldMatrix = FieldMatrix::from_function(big_n as usize, m, |i, j| q[j].nth(i as i64 + 1));
            let (ac, _) = constant_system(&big_m, &FieldMatrix::zeros(big_m.rows, 1), de)?;
            a = a.col_join(&ac.row_join(&FieldMatrix::zeros(ac.rows, r))?)?;
        }
        return Ok((h, a));
    }
    let eta: RationalFunction = de.d.exquo(&t2_1(&t))?.as_element()?;
    let p: TowerPoly = t2_1(&t);
    let tf: RationalFunction = RationalFunction::generator(&t);
    let b0e: RationalFunction = b0.as_element()?;
    let b: TowerPoly = poly(&b0e.sub(&eta.mul(&tf)?.mul(&rf_int(n))?)?, de)?;
    let divs: Vec<(TowerPoly, TowerPoly)> = q.iter().map(|qi| qi.div(&p)).collect::<PolyResult<_>>()?;
    let rems: Vec<(RationalFunction, RationalFunction)> = divs.iter().map(|(_, r)| (r.nth(0), r.nth(1))).collect();
    let (re, an) = below(de, |de| {
        let g: Vec<(Fraction2, Fraction2)> = rems
            .iter()
            .map(|(r0, r1)| Ok((frac_in(r0, &de.t, false)?, frac_in(r1, &de.t, false)?)))
            .collect::<PolyResult<_>>()?;
        let f1 = frac_in(&b0e, &de.t, false)?;
        let f2 = frac_in(&eta.neg().mul(&rf_int(n))?, &de.t, false)?;
        let (r, an) = super::cde::param_coupled_de_system(&f1, &f2, &g, de)?;
        let re: Vec<(RationalFunction, RationalFunction)> = r
            .iter()
            .map(|((ua, ud), (va, vd))| Ok((el(ua, ud)?, el(va, vd)?)))
            .collect::<PolyResult<_>>()?;
        Ok((re, an))
    })?;
    let rp: Vec<TowerPoly> = re
        .iter()
        .map(|(ue, ve)| poly(&ue.add(&ve.mul(&tf)?)?, de))
        .collect::<PolyResult<_>>()?;
    let mut qnew: Vec<TowerPoly> = divs.iter().map(|(quo, _)| quo.clone()).collect();
    for rj in &rp {
        qnew.push(derivation(rj, de, false)?.add(&b.mul(rj)?)?.div(&p)?.0.neg());
    }
    let (h2, m2) = prde_cancel_tan(&b0, &qnew, n - 2, de)?;
    let mut h: Vec<TowerPoly> = rp;
    for hh in &h2 {
        h.push(p.mul(hh)?);
    }
    let big_m: FieldMatrix = an.row_join(&FieldMatrix::zeros(an.rows, h2.len()))?.col_join(&m2)?;
    Ok((h, big_m))
}

/// The polynomial solutions of the parametric equation
/// (`param_poly_rischDE`): `(h, A)` with the solutions `sum d_j h_j` for
/// the constants with `A (c, d) = 0`.
pub fn param_poly_risch_de(
    a: &TowerPoly,
    b: &TowerPoly,
    q: &[TowerPoly],
    n: Degree,
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<TowerPoly>, FieldMatrix)> {
    let (mut a, mut b) = (in_t(a, de)?, in_t(b, de)?);
    let mut q: Vec<TowerPoly> = q.iter().map(|qi| in_t(qi, de)).collect::<PolyResult<_>>()?;
    let m: usize = q.len();
    let t: Basic = de.t.clone();
    let mut n: Degree = n;
    if n < 0 {
        if q.iter().all(|qi| qi.is_zero()) {
            return Ok((Vec::new(), FieldMatrix::zeros(1, m)));
        }
        let big_n: i64 = q.iter().map(|qi| qi.degree(Some(&t))).max().unwrap_or(0);
        let big_m: FieldMatrix = FieldMatrix::from_function((big_n + 1) as usize, m, |i, j| q[j].nth(i as i64));
        let (a, _) = constant_system(&big_m, &FieldMatrix::zeros(big_m.rows, 1), de)?;
        return Ok((Vec::new(), a));
    }
    if a.is_ground() {
        let lc: RationalFunction = a.LC();
        b = b.exquo_ground(&lc)?;
        q = q.iter().map(|qi| qi.exquo_ground(&lc)).collect::<PolyResult<_>>()?;
        let dd: i64 = de.d.degree(None);
        if !b.is_zero() && (de.case == Case::Base || b.degree(None) > 0.max(dd - 1)) {
            return prde_no_cancel_b_large(&b, &q, n, de);
        }
        if (b.is_zero() || b.degree(None) < dd - 1) && (de.case == Case::Base || dd >= 2) {
            return prde_no_cancel_b_small(&b, &q, n, de);
        }
        if dd >= 2 && b.degree(None) == dd - 1 && no_cancel_equal_applies(&b, n, de)? {
            return prde_no_cancel_b_equal(&b, &q, n, de);
        }
        return match de.case {
            Case::Primitive | Case::Exp => prde_cancel_liouvillian(&b, &q, n, de),
            Case::Tan => {
                let eta: RationalFunction = de.d.exquo(&t2_1(&t))?.as_element()?;
                if !is_oo(n) && b.degree(Some(&t)) <= 1 && b.nth(1).add(&eta.mul(&rf_int(n))?)?.is_zero() {
                    return prde_cancel_tan(&TowerPoly::ground(&b.nth(0), &t)?, &q, n, de);
                }
                Err(PolyError::NotImplemented(
                    "The tangent cancellation case requires b == b0 - n*eta*t with b0 in k in param_poly_rischDE()."
                        .to_string(),
                ))
            }
            _ => Err(PolyError::NotImplemented(
                "nonlinear cases have not yet been implemented".to_string(),
            )),
        };
    }
    let mut alpha: TowerPoly = TowerPoly::one(&t);
    let mut beta: Vec<TowerPoly> = vec![TowerPoly::zero(&t); m];
    let mut d: TowerPoly = TowerPoly::one(&t);
    while n >= 0 {
        crate::timeout::check()?;
        let (a2, b2, q2, r, n2) = prde_spde(&a, &b, &q, n, de)?;
        a = a2;
        b = b2;
        q = q2;
        n = n2;
        beta = beta
            .iter()
            .zip(&r)
            .map(|(bi, ri)| bi.add(&alpha.mul(ri)?))
            .collect::<PolyResult<_>>()?;
        alpha = alpha.mul(&a)?;
        d = a.gcd(&b)?;
        if !d.is_ground() {
            break;
        }
    }
    let (qq, big_m) = poly_linear_constraints(&q, &d)?;
    let (a_sys, _) = constant_system(&big_m, &FieldMatrix::zeros(big_m.rows, 1), de)?;
    let v: Vec<FieldMatrix> = a_sys.nullspace()?;
    if v.is_empty() {
        return Ok((Vec::new(), FieldMatrix::eye(m)));
    }
    let mqq: FieldMatrix =
        FieldMatrix::row_vector(&qq.iter().map(|qi| qi.as_element()).collect::<PolyResult<Vec<_>>>()?);
    let r: Vec<TowerPoly> = v
        .iter()
        .map(|vj| poly(mqq.mul(vj)?.element(0), de))
        .collect::<PolyResult<_>>()?;
    let mbeta: FieldMatrix =
        FieldMatrix::row_vector(&beta.iter().map(|bi| bi.as_element()).collect::<PolyResult<Vec<_>>>()?);
    let f: Vec<TowerPoly> = v
        .iter()
        .map(|vj| poly(mbeta.mul(vj)?.element(0), de))
        .collect::<PolyResult<_>>()?;
    let (g, big_b) = param_poly_risch_de(&a.quo(&d)?, &b.quo(&d)?, &r, n, de)?;
    let mut h: Vec<TowerPoly> = f;
    for gk in &g {
        h.push(alpha.mul(gk)?);
    }
    let mut a_out: FieldMatrix = FieldMatrix::eye(m).neg();
    for vj in &v {
        a_out = a_out.row_join(vj)?;
    }
    a_out = a_out.row_join(&FieldMatrix::zeros(m, g.len()))?;
    a_out = a_out.col_join(&FieldMatrix::zeros(big_b.rows, m).row_join(&big_b)?)?;
    Ok((h, a_out))
}

fn prde_normalized_solve(
    big_a: &TowerPoly,
    big_b: &TowerPoly,
    g: &[Fraction2],
    gamma: &TowerPoly,
    de: &mut DifferentialExtension,
    n: Option<Degree>,
) -> PolyResult<(Vec<Fraction2>, FieldMatrix)> {
    let m: usize = g.len();
    let (big_a, big_b, gamma) = (in_t(big_a, de)?, in_t(big_b, de)?, in_t(gamma, de)?);
    let gg: TowerPoly = big_a.gcd(&big_b)?;
    let (a, b) = (big_a.quo(&gg)?, big_b.quo(&gg)?);
    let mut gs: Vec<Fraction2> = Vec::new();
    for (gia, gid) in g {
        gs.push(in_t(gia, de)?.cancel(&in_t(gid, de)?.mul(&gg)?)?);
    }
    let (q, big_m) = prde_linear_constraints(&gs, de)?;
    let (big_m, _) = constant_system(&big_m, &FieldMatrix::zeros(big_m.rows, 1), de)?;
    let v: Vec<FieldMatrix> = big_m.nullspace()?;
    if v.is_empty() {
        return Ok((Vec::new(), FieldMatrix::eye(m)));
    }
    let mq: FieldMatrix = FieldMatrix::row_vector(&q.iter().map(|qi| qi.as_element()).collect::<PolyResult<Vec<_>>>()?);
    let r: Vec<TowerPoly> = v
        .iter()
        .map(|vj| poly(mq.mul(vj)?.element(0), de))
        .collect::<PolyResult<_>>()?;
    let n: Degree = match n {
        Some(n) => n,
        None => bound_degree(&a, &b, &r, de, None)?,
    };
    let (h, relation_b) = param_poly_risch_de(&a, &b, &r, n, de)?;
    let mut relation: FieldMatrix = FieldMatrix::eye(m).neg();
    for vj in &v {
        relation = relation.row_join(vj)?;
    }
    relation = relation.row_join(&FieldMatrix::zeros(m, h.len()))?;
    relation = relation.col_join(&FieldMatrix::zeros(relation_b.rows, m).row_join(&relation_b)?)?;
    let w: Vec<FieldMatrix> = relation.nullspace()?;
    let vlen: usize = h.len();
    let mut items: Vec<RationalFunction> = Vec::new();
    for wl in &w {
        let entries: &[RationalFunction] = wl.flat();
        items.extend(entries[..m].iter().cloned());
        if vlen > 0 {
            items.extend(entries[entries.len() - vlen..].iter().cloned());
        }
    }
    let big_m: FieldMatrix = FieldMatrix::new(w.len(), m + vlen, items)?;
    let nsp: Vec<FieldMatrix> = big_m.nullspace()?;
    let rows: Vec<Vec<RationalFunction>> = nsp.iter().map(|ni| ni.flat().to_vec()).collect();
    let c: FieldMatrix = FieldMatrix::from_rows(&rows, m + vlen)?;
    let mut out: Vec<Fraction2> = Vec::new();
    for hk in &h {
        out.push(hk.cancel(&gamma)?);
    }
    Ok((out, c))
}

/// The parametric Risch differential equation (`param_rischDE`).
pub fn param_risch_de(
    fa: &TowerPoly,
    fd: &TowerPoly,
    g: &[Fraction2],
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<Fraction2>, FieldMatrix)> {
    let (fa, fd) = (in_t(fa, de)?, in_t(fd, de)?);
    let (q, (fa, fd)) = weak_normalizer(&fa, &fd, de)?;
    let mut gamma: TowerPoly = q.clone();
    let mut g2: Vec<Fraction2> = Vec::new();
    for (ga, gd) in g {
        g2.push(q.mul(&in_t(ga, de)?)?.cancel(&in_t(gd, de)?)?);
    }
    let (a, (ba, bd), g3, hn) = prde_normal_denom(&fa, &fd, &g2, de)?;
    gamma = gamma.mul(&hn)?;
    let (big_a, big_b, g4, hs) = prde_special_denom(&a, &ba, &bd, &g3, de, None)?;
    gamma = gamma.mul(&hs)?;
    prde_normalized_solve(&big_a, &big_b, &g4, &gamma, de, None)
}

/// The reduction of limited integration (`limited_integrate_reduce`).
pub fn limited_integrate_reduce(
    fa: &TowerPoly,
    fd: &TowerPoly,
    g: &[Fraction2],
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly, TowerPoly, Degree, Fraction2, Vec<Fraction2>)> {
    let (fa, fd) = (in_t(fa, de)?, in_t(fd, de)?);
    let g: Vec<Fraction2> = g
        .iter()
        .map(|(a, d)| Ok((in_t(a, de)?, in_t(d, de)?)))
        .collect::<PolyResult<_>>()?;
    let (dn, ds) = splitfactor(&fd, de, false)?;
    let e: Vec<(TowerPoly, TowerPoly)> = g
        .iter()
        .map(|(_, gd)| splitfactor(gd, de, false))
        .collect::<PolyResult<_>>()?;
    let en: Vec<TowerPoly> = e.iter().map(|(x, _)| x.clone()).collect();
    let es: Vec<TowerPoly> = e.iter().map(|(_, s)| s.clone()).collect();
    let c: TowerPoly = lcm_polys(&en, &dn)?;
    let hn: TowerPoly = c.gcd(&c.diff(Some(&de.t))?)?;
    if !matches!(de.case, Case::Base | Case::Primitive | Case::Exp | Case::Tan) {
        return Err(PolyError::NotImplemented(String::new()));
    }
    let hs: TowerPoly = lcm_polys(&es, &ds)?;
    let a: TowerPoly = hn.mul(&hs)?;
    let b: TowerPoly = derivation(&hn, de, false)?
        .neg()
        .sub(&hn.mul(&derivation(&hs, de, false)?)?.quo(&hs)?)?;
    let mut mu: Degree = order_at_oo(&fa, &fd, &de.t);
    for (ga, gd) in &g {
        mu = mu.min(order_at_oo(ga, gd, &de.t));
    }
    let big_n: Degree = hn.degree(Some(&de.t)) + hs.degree(Some(&de.t)) + 0.max(1 - de.d.degree(Some(&de.t)) - mu);
    let mut v: Vec<Fraction2> = Vec::new();
    for (ga, gd) in &g {
        v.push(a.neg().mul(&hn)?.mul(ga)?.cancel(gd)?);
    }
    let first: Fraction2 = a.mul(&hn)?.mul(&fa)?.cancel(&fd)?;
    Ok((hn, b, a, big_n, first, v))
}

/// Limited integration (`limited_integrate`): `((ya, yd), c)` with
/// `f = Dy + sum c_i w_i`, `None` when there is none.
pub fn limited_integrate(
    fa: &TowerPoly,
    fd: &TowerPoly,
    g: &[Fraction2],
    de: &mut DifferentialExtension,
) -> PolyResult<Option<(Fraction2, Vec<RationalFunction>)>> {
    let (fa, fd) = (in_t(fa, de)?, in_t(fd, de)?);
    let (fa, fd) = (fa.mul_ground(&fd.LC().invert()?), fd.monic()?);
    let (big_a, b, h, big_n, gg, v) = limited_integrate_reduce(&fa, &fd, g, de)?;
    let mut q: Vec<Fraction2> = vec![gg];
    q.extend(v);
    let (hs, c) = prde_normalized_solve(&big_a, &b, &q, &h, de, Some(big_n))?;
    let w: Vec<FieldMatrix> = c.nullspace()?.into_iter().filter(|w| !w.element(0).is_zero()).collect();
    if w.is_empty() {
        return Ok(None);
    }
    let w0: Vec<RationalFunction> = w[0].div(w[0].element(0))?.flat().to_vec();
    let r: usize = hs.len();
    let m: usize = w0.len() - r - 1;
    let constants: Vec<RationalFunction> = w0[1..m + 1].to_vec();
    let mut y: RationalFunction = zero();
    for i in 0..r {
        y = y.add(&w0[m + 1 + i].mul(&el(&hs[i].0, &hs[i].1)?)?)?;
    }
    let (ya, yd) = frac_in(&y, &de.t, false)?;
    let lc: RationalFunction = yd.LC().invert()?;
    Ok(Some(((ya.mul_ground(&lc), yd.mul_ground(&lc)), constants)))
}

/// `y` with `Dy = f` in the field, `None` when there is none
/// (`is_deriv_in_field`).
pub fn is_deriv_in_field(
    fa: &TowerPoly,
    fd: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<Option<Fraction2>> {
    Ok(limited_integrate(fa, fd, &[], de)?.map(|(y, _)| y))
}

fn solve_c1(ps: &[RationalFunction], qs: &[RationalFunction]) -> PolyResult<Option<RationalFunction>> {
    let mut c: Option<RationalFunction> = None;
    for (p, q) in ps.iter().zip(qs) {
        if !q.is_zero() {
            c = Some(p.div(q)?);
            break;
        }
    }
    let Some(c) = c else { return Ok(None) };
    for (p, q) in ps.iter().zip(qs) {
        if !p.sub(&c.mul(q)?)?.is_zero() {
            return Ok(None);
        }
    }
    Ok(Some(c))
}

fn rational_solution(c: Option<RationalFunction>) -> PolyResult<Option<(i64, i64)>> {
    let Some(c) = c else { return Ok(None) };
    let Some(q) = c.as_fraction() else { return Ok(None) };
    Ok(Some((to_i64(&numerator(&q))?, to_i64(&denominator(&q))?)))
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i64(b, a % b)
    }
}

fn heu_tail(
    fa: &TowerPoly,
    fd: &TowerPoly,
    wa: &TowerPoly,
    wd: &TowerPoly,
    big_m: i64,
    big_n: i64,
    de: &mut DifferentialExtension,
) -> PolyResult<Option<(i64, i64, RationalFunction)>> {
    let nfmwa: TowerPoly = fa
        .mul(wd)?
        .mul_ground(&rf_int(big_n))
        .sub(&wa.mul(fd)?.mul_ground(&rf_int(big_m)))?;
    let nfmwd: TowerPoly = fd.mul(wd)?;
    let Some((q, v)) = is_log_deriv_k_t_radical_in_field(&nfmwa, &nfmwd, de, None)? else {
        return Ok(None);
    };
    if q == 0 || v.is_zero() {
        return Ok(None);
    }
    Ok(Some((q * big_n, q * big_m, v)))
}

fn structure_needed() -> PolyError {
    PolyError::NotImplemented(
        "parametric_log_deriv_heu() heuristic failed: the full parametric logarithmic derivative problem (using the structure theorems) is not yet implemented."
            .to_string(),
    )
}

/// The heuristic of the parametric logarithmic derivative problem
/// (`parametric_log_deriv_heu`): `(n, m, v)` with `n f = Dv/v + m w`.
pub fn parametric_log_deriv_heu(
    fa: &TowerPoly,
    fd: &TowerPoly,
    wa: &TowerPoly,
    wd: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<Option<(i64, i64, RationalFunction)>> {
    let (fa, fd, wa, wd) = (in_t(fa, de)?, in_t(fd, de)?, in_t(wa, de)?, in_t(wd, de)?);
    let f: Option<Rat> = el(&fa, &fd)?.as_fraction();
    let w: Option<Rat> = el(&wa, &wd)?.as_fraction();
    if let (Some(f), Some(w)) = (&f, &w) {
        if *f == Rat::ZERO && *w == Rat::ZERO {
            return Ok(Some((1, 0, one())));
        }
        let x: i64 = to_i64(&(numerator(f) * denominator(w)))?;
        let y: i64 = to_i64(&(numerator(w) * denominator(f)))?;
        let g: i64 = gcd_i64(x, y);
        let (x, y) = if g == 0 { (x, y) } else { (x / g, y / g) };
        let (mut n, mut m) = (y, x);
        if n == 0 {
            return Ok(None);
        }
        if n < 0 {
            n = -n;
            m = -m;
        }
        return Ok(Some((n, m, one())));
    }
    let (p, _a) = fa.div(&fd)?;
    let (q, _b) = wa.div(&wd)?;
    let t: Basic = de.t.clone();
    let big_b: i64 = 0.max(de.d.degree(Some(&t)) - 1);
    let big_c: i64 = p.degree(Some(&t)).max(q.degree(Some(&t)));
    if q.degree(Some(&t)) > big_b {
        let ps: Vec<RationalFunction> = (big_b + 1..=big_c).map(|i| p.nth(i)).collect();
        let qs: Vec<RationalFunction> = (big_b + 1..=big_c).map(|i| q.nth(i)).collect();
        let Some((big_m, big_n)) = rational_solution(solve_c1(&ps, &qs)?)? else {
            return Ok(None);
        };
        return heu_tail(&fa, &fd, &wa, &wd, big_m, big_n, de);
    }
    if p.degree(Some(&t)) > big_b {
        return Ok(None);
    }
    let l: TowerPoly = fd.monic()?.lcm(&wd.monic()?)?;
    let (ln, ls) = splitfactor(&l, de, false)?;
    let z: TowerPoly = ls.mul(&ln.gcd(&ln.diff(Some(&t))?)?)?;
    if z.degree(Some(&t)) < 1 {
        if de.case == Case::Base {
            if q.is_zero() {
                if !p.is_zero() {
                    return Ok(None);
                }
                return Err(structure_needed());
            }
            let (mi, ni): (i64, i64) = if p.is_zero() {
                (0, 1)
            } else {
                if p.mul_ground(&q.LC()) != q.mul_ground(&p.LC()) {
                    return Ok(None);
                }
                let Some(cc) = p.LC().div(&q.LC())?.as_fraction() else {
                    return Ok(None);
                };
                (to_i64(&numerator(&cc))?, to_i64(&denominator(&cc))?)
            };
            return heu_tail(&fa, &fd, &wa, &wd, mi, ni, de);
        }
        return Err(structure_needed());
    }
    let (_u1, r1) = fa.mul(&l.quo(&fd)?)?.div(&z)?;
    let (_u2, r2) = wa.mul(&l.quo(&wd)?)?.div(&z)?;
    let ps: Vec<RationalFunction> = (0..z.degree(Some(&t))).map(|i| r1.nth(i)).collect();
    let qs: Vec<RationalFunction> = (0..z.degree(Some(&t))).map(|i| r2.nth(i)).collect();
    let Some((big_m, big_n)) = rational_solution(solve_c1(&ps, &qs)?)? else {
        return Ok(None);
    };
    heu_tail(&fa, &fd, &wa, &wd, big_m, big_n, de)
}

fn term_element(e: &Option<Basic>) -> PolyResult<RationalFunction> {
    match e {
        Some(e) => RationalFunction::from_expr(e),
        None => Err(PolyError::NotImplemented(
            "The structure theorems need the arguments of the monomials of the tower (DE.extargs).".to_string(),
        )),
    }
}

fn term_expr(e: &Option<Basic>) -> PolyResult<Basic> {
    match e {
        Some(e) => Ok(e.clone()),
        None => Err(PolyError::NotImplemented(
            "The structure theorems need the arguments of the monomials of the tower (DE.extargs).".to_string(),
        )),
    }
}

fn solve_linear(a: &FieldMatrix, u: &FieldMatrix) -> PolyResult<Option<Vec<RationalFunction>>> {
    let (r, pivots) = a.row_join(u)?.rref()?;
    let n: usize = a.cols;
    if pivots.contains(&n) {
        return Ok(None);
    }
    let mut xs: Vec<RationalFunction> = vec![zero(); n];
    for (k, c) in pivots.iter().enumerate() {
        xs[*c] = r.entry(k as i64, n as i64)?.clone();
    }
    Ok(Some(xs))
}

fn e_part(de: &DifferentialExtension, indices: &[usize]) -> PolyResult<Vec<RationalFunction>> {
    indices
        .iter()
        .map(|i| de.D[*i].quo(&monomial_t(&de.T[*i]))?.as_element())
        .collect()
}

fn l_part(de: &DifferentialExtension, indices: &[usize]) -> PolyResult<Vec<RationalFunction>> {
    indices.iter().map(|i| de.D[*i].as_element()).collect()
}

fn t_part(de: &DifferentialExtension, indices: &[usize]) -> PolyResult<Vec<RationalFunction>> {
    indices
        .iter()
        .map(|i| de.D[*i].quo(&t2_1(&de.T[*i]))?.as_element())
        .collect()
}

/// The structure theorems for the parametric logarithmic derivative
/// problem (`parametric_log_deriv_structure`).
pub fn parametric_log_deriv_structure(
    fa: &TowerPoly,
    fd: &TowerPoly,
    wa: &TowerPoly,
    wd: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<Option<(i64, i64, RationalFunction)>> {
    let (e_idx, l_idx, _, _) = structure_tower(de, &[fa, fd, wa, wd], false)?;
    let ep: Vec<RationalFunction> = e_part(de, &e_idx)?;
    let lp: Vec<RationalFunction> = l_part(de, &l_idx)?;
    let w: RationalFunction = el(wa, wd)?;
    let f: RationalFunction = el(fa, fd)?;
    let mut row: Vec<RationalFunction> = vec![w.clone()];
    row.extend(ep.iter().cloned());
    row.extend(lp.iter().cloned());
    let (a, u) = constant_system(
        &FieldMatrix::row_vector(&row),
        &FieldMatrix::column(std::slice::from_ref(&f)),
        de,
    )?;
    let rational = |m: &FieldMatrix| m.flat().iter().all(|i| i.is_rational());
    if !(rational(&a) && rational(&u)) {
        let constant = |m: &FieldMatrix| -> PolyResult<bool> {
            for i in m.flat() {
                if !derivation_element(i, de, false)?.is_zero() {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        if !constant(&a)? || !constant(&u)? {
            return Ok(None);
        }
        return Err(PolyError::NotImplemented(
            "Cannot work with non-rational coefficients in this case.".to_string(),
        ));
    }
    let Some(found) = solve_linear(&a, &u)? else {
        return Ok(None);
    };
    let mut fractions: Vec<Rat> = Vec::new();
    for xi in found {
        match xi.as_fraction() {
            Some(q) => fractions.push(q),
            None => {
                return Err(PolyError::NotImplemented(
                    "Cannot work with non-rational coefficients in this case.".to_string(),
                ))
            }
        }
    }
    let mut n: Int = Int::ONE;
    for q in &fractions {
        n = ilcm(&n, &denominator(q));
    }
    let xs: Vec<i64> = fractions
        .iter()
        .map(|q| to_i64(&numerator(&(q * Rat::from(n.clone())))))
        .collect::<PolyResult<_>>()?;
    let m: i64 = xs[0];
    let mut terms: Vec<RationalFunction> = e_idx.iter().map(|i| RationalFunction::generator(&de.T[*i])).collect();
    for i in &l_idx {
        terms.push(term_element(&de.extargs[*i - 1])?);
    }
    let mut v: RationalFunction = one();
    for (term, k) in terms.iter().zip(&xs[1..]) {
        v = v.mul(&term.pow(*k)?)?;
    }
    let mut n: i64 = to_i64(&n)?;
    if n == 0 {
        return Ok(None);
    }
    let mut check: RationalFunction = zero();
    for (part, k) in ep.iter().chain(&lp).zip(&xs[1..]) {
        check = check.add(&part.mul(&rf_int(*k))?)?;
    }
    if !f.mul(&rf_int(n))?.sub(&w.mul(&rf_int(m))?)?.sub(&check)?.is_zero() {
        return Ok(None);
    }
    let mut m: i64 = m;
    if n < 0 {
        n = -n;
        m = -m;
        v = v.invert()?;
    }
    Ok(Some((n, m, v)))
}

/// The parametric logarithmic derivative problem (`parametric_log_deriv`).
pub fn parametric_log_deriv(
    fa: &TowerPoly,
    fd: &TowerPoly,
    wa: &TowerPoly,
    wd: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<Option<(i64, i64, RationalFunction)>> {
    match parametric_log_deriv_heu(fa, fd, wa, wd, de) {
        Ok(v) => Ok(v),
        Err(e) if caught_not_implemented(&e) => {
            let found = parametric_log_deriv_structure(fa, fd, wa, wd, de)?;
            if found.is_none() {
                return Err(PolyError::NotImplemented(
                    "parametric_log_deriv() could not decide: the heuristic failed and f - (m/n)*w is not a combination of logarithmic derivatives from the current tower for any m/n in QQ."
                        .to_string(),
                ));
            }
            Ok(found)
        }
        Err(e) => Err(e),
    }
}

fn structure_system_solve(
    lhs: &FieldMatrix,
    rhs: &FieldMatrix,
    de: &DifferentialExtension,
) -> PolyResult<Option<Vec<RationalFunction>>> {
    let (a, u) = constant_system(lhs, rhs, de)?;
    if a.is_empty() {
        return Ok(None);
    }
    for i in u.flat() {
        if !derivation_element(i, de, false)?.is_zero() {
            return Ok(None);
        }
    }
    let Some(xs) = solve_linear(&a, &u)? else {
        return Ok(None);
    };
    for i in 0..lhs.rows {
        let mut total: RationalFunction = zero();
        for (j, x) in xs.iter().enumerate().take(lhs.cols) {
            total = total.add(&lhs.entry(i as i64, j as i64)?.mul(x)?)?;
        }
        if !total.sub(rhs.entry(i as i64, 0)?)?.is_zero() {
            return Err(PolyError::NotImplemented(
                "The candidate solution of the structure system failed verification.".to_string(),
            ));
        }
    }
    Ok(Some(xs))
}

fn tower_has_i(de: &DifferentialExtension, polys: &[&TowerPoly]) -> bool {
    let i_list: [Basic; 1] = [Basic::imaginary_unit()];
    let top: usize = (de.T.len() as i64 + de.level) as usize;
    polys.iter().any(|p| p.has(&i_list)) || de.D[..=top].iter().any(|d| d.has(&i_list))
}

fn structure_tower(
    de: &DifferentialExtension,
    polys: &[&TowerPoly],
    real: bool,
) -> PolyResult<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)> {
    if de.exts.len() != de.D.len() - 1 {
        return Err(PolyError::NotImplemented(
            "The structure theorems need every monomial of the tower labeled as 'exp', 'log', 'tan' or 'atan' in DE.exts; nonelementary extensions are not supported."
                .to_string(),
        ));
    }
    let top: usize = (de.T.len() as i64 + de.level) as usize;
    let keep = |v: Vec<usize>| -> Vec<usize> { v.into_iter().filter(|i| *i <= top).collect() };
    let e: Vec<usize> = keep(de.indices(Extension::Exp));
    let l: Vec<usize> = keep(de.indices(Extension::Log));
    let t: Vec<usize> = keep(de.indices(Extension::Tan));
    let a: Vec<usize> = keep(de.indices(Extension::Atan));
    if (real || !t.is_empty() || !a.is_empty()) && tower_has_i(de, polys) {
        return Err(PolyError::NotImplemented(
            "The real version of the structure theorems (Section 9.3 of Bronstein's book), needed for towers with hypertangent or arc-tangent monomials, requires sqrt(-1) not in the field."
                .to_string(),
        ));
    }
    Ok((e, l, t, a))
}

fn has_symbols(e: &RationalFunction) -> bool {
    e.gens.iter().any(|g| !g.free_symbols().is_empty())
}

fn structure_solve(
    parts: &[RationalFunction],
    rhs: &RationalFunction,
    de: &DifferentialExtension,
) -> PolyResult<Option<Vec<Rat>>> {
    if rhs.is_zero() {
        return Ok(Some(vec![Rat::ZERO; parts.len()]));
    }
    let Some(u) = structure_system_solve(
        &FieldMatrix::row_vector(parts),
        &FieldMatrix::column(std::slice::from_ref(rhs)),
        de,
    )?
    else {
        return Ok(None);
    };
    let mut result: Vec<Rat> = Vec::new();
    for i in &u {
        match i.as_fraction() {
            Some(q) => result.push(q),
            None => {
                if u.iter().all(|j| j.is_rational() || has_symbols(j)) {
                    return Ok(None);
                }
                return Err(PolyError::NotImplemented(
                    "Cannot work with non-rational coefficients in this case.".to_string(),
                ));
            }
        }
    }
    Ok(Some(result))
}

fn df_over_f(fa: &TowerPoly, fd: &TowerPoly, de: &DifferentialExtension) -> PolyResult<RationalFunction> {
    let dfa: TowerPoly = fd
        .mul(&derivation(fa, de, false)?)?
        .sub(&fa.mul(&derivation(fd, de, false)?)?)?;
    dfa.as_element()?.div(&fd.mul(fa)?.as_element()?)
}

fn df(fa: &TowerPoly, fd: &TowerPoly, de: &DifferentialExtension) -> PolyResult<RationalFunction> {
    let dfa: TowerPoly = fd
        .mul(&derivation(fa, de, false)?)?
        .sub(&fa.mul(&derivation(fd, de, false)?)?)?;
    dfa.as_element()?.div(&fd.pow(2).as_element()?)
}

fn powers_of_factors(e: &Basic, j: &Rat) -> PolyResult<Basic> {
    let jj: Basic = number(j);
    let (n, d) = crate::core::expr::as_numer_denom(e)?;
    let mut parts: Vec<Basic> = Vec::new();
    for (f, sign) in [(n, 1i64), (d, -1i64)] {
        let (coeff, terms) = crate::polys::polytools::sqf_list(&f, &[])?;
        parts.push(crate::core::power::pow(
            &coeff,
            &crate::core::mul::mul(&[Basic::int(sign), jj.clone()])?,
        )?);
        for (b, k) in terms {
            parts.push(crate::core::power::pow(
                &b,
                &crate::core::mul::mul(&[Basic::int(sign), k, jj.clone()])?,
            )?);
        }
    }
    crate::core::mul::mul(&parts)
}

fn cancel_expr(e: &Basic) -> PolyResult<Basic> {
    match RationalFunction::from_expr(e) {
        Ok(r) => r.as_expr(),
        Err(err) if caught_not_implemented(&err) => crate::polys::polytools::cancel(e, &[]),
        Err(err) => Err(err),
    }
}

/// `f` as `Du/u` for a product `u` of powers of the exponential and
/// logarithmic generators, up to a constant (`is_deriv_k`): `(ans, u, const)`.
pub fn is_deriv_k(
    fa: &TowerPoly,
    fd: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<Option<(Terms, Basic, Basic)>> {
    let dff: RationalFunction = df_over_f(fa, fd, de)?;
    let (e_idx, l_idx, _, _) = structure_tower(de, &[fa, fd], false)?;
    let mut parts: Vec<RationalFunction> = e_part(de, &e_idx)?;
    parts.extend(l_part(de, &l_idx)?);
    let Some(u) = structure_solve(&parts, &dff, de)? else {
        return Ok(None);
    };
    let mut terms: Vec<Basic> = e_idx
        .iter()
        .map(|i| term_expr(&de.extargs[*i - 1]))
        .collect::<PolyResult<_>>()?;
    terms.extend(l_idx.iter().map(|i| de.T[*i].clone()));
    let ans: Terms = terms.into_iter().zip(u.iter().map(number)).collect();
    let mut summands: Vec<Basic> = Vec::new();
    for (i, j) in &ans {
        summands.push(crate::core::mul::mul(&[i.clone(), j.clone()])?);
    }
    let result: Basic = crate::core::add::add(&summands)?;
    let mut argterms: Vec<Basic> = e_idx.iter().map(|i| de.T[*i].clone()).collect();
    for i in &l_idx {
        argterms.push(term_expr(&de.extargs[*i - 1])?);
    }
    let mut l: Vec<Basic> = Vec::new();
    for (i, j) in argterms.iter().zip(&u) {
        l.push(powers_of_factors(i, j)?);
    }
    let f: Basic = el(fa, fd)?.as_expr()?;
    let constant: Basic = cancel_expr(&crate::core::mul::mul(&[
        f,
        crate::core::power::pow(&crate::core::mul::mul(&l)?, &Basic::int(-1))?,
    ])?)?;
    Ok(Some((ans, result, constant)))
}

/// `f` as `Du/u` with `u` a radical of a product of the generators, up to
/// a constant (`is_log_deriv_k_t_radical`): `(ans, u, n, const)`.
pub fn is_log_deriv_k_t_radical(
    fa: &TowerPoly,
    fd: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<Option<(Terms, Basic, i64, Basic)>> {
    let dff: RationalFunction = df(fa, fd, de)?;
    let (e_idx, l_idx, _, _) = structure_tower(de, &[fa, fd], false)?;
    let mut parts: Vec<RationalFunction> = e_part(de, &e_idx)?;
    parts.extend(l_part(de, &l_idx)?);
    let Some(u) = structure_solve(&parts, &dff, de)? else {
        return Ok(None);
    };
    let mut n: Int = Int::ONE;
    for q in &u {
        n = ilcm(&n, &denominator(q));
    }
    let un: Vec<Rat> = u.iter().map(|q| q * Rat::from(n.clone())).collect();
    let mut terms: Vec<Basic> = e_idx.iter().map(|i| de.T[*i].clone()).collect();
    for i in &l_idx {
        terms.push(term_expr(&de.extargs[*i - 1])?);
    }
    let ans: Terms = terms.into_iter().zip(un.iter().map(number)).collect();
    let mut factors: Vec<Basic> = Vec::new();
    for (i, j) in &ans {
        factors.push(crate::core::power::pow(i, j)?);
    }
    let result: Basic = crate::core::mul::mul(&factors)?;
    let mut argterms: Vec<RationalFunction> = e_idx
        .iter()
        .map(|i| term_element(&de.extargs[*i - 1]))
        .collect::<PolyResult<_>>()?;
    argterms.extend(l_idx.iter().map(|i| RationalFunction::generator(&de.T[*i])));
    let mut constant: RationalFunction = el(fa, fd)?;
    for (i, j) in argterms.iter().zip(&u) {
        constant = constant.sub(&i.mul(&RationalFunction::from_fraction(j))?)?;
    }
    Ok(Some((ans, result, to_i64(&n)?, constant.as_expr()?)))
}

/// `f` as `D atan(u)`-like combination of the tangent and arc-tangent
/// generators (`is_deriv_k_atan`): `(ans, u)`.
pub fn is_deriv_k_atan(
    fa: &TowerPoly,
    fd: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<Option<(Terms, Basic)>> {
    let dfa: TowerPoly = fd
        .mul(&derivation(fa, de, false)?)?
        .sub(&fa.mul(&derivation(fd, de, false)?)?)?;
    let dff: RationalFunction = dfa.as_element()?.div(&fa.pow(2).add(&fd.pow(2))?.as_element()?)?;
    let (_, _, t_idx, a_idx) = structure_tower(de, &[fa, fd], true)?;
    let mut parts: Vec<RationalFunction> = t_part(de, &t_idx)?;
    parts.extend(l_part(de, &a_idx)?);
    let Some(u) = structure_solve(&parts, &dff, de)? else {
        return Ok(None);
    };
    let mut terms: Vec<Basic> = t_idx
        .iter()
        .map(|i| term_expr(&de.extargs[*i - 1]))
        .collect::<PolyResult<_>>()?;
    terms.extend(a_idx.iter().map(|i| de.T[*i].clone()));
    let ans: Terms = terms.into_iter().zip(u.iter().map(number)).collect();
    let mut summands: Vec<Basic> = Vec::new();
    for (i, j) in &ans {
        summands.push(crate::core::mul::mul(&[i.clone(), j.clone()])?);
    }
    Ok(Some((ans, crate::core::add::add(&summands)?)))
}

/// The tangent analogue of [`is_log_deriv_k_t_radical`]
/// (`is_log_deriv_k_t_radical_tan`).
pub fn is_log_deriv_k_t_radical_tan(
    fa: &TowerPoly,
    fd: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<Option<(Terms, Basic, i64, Basic)>> {
    let dff: RationalFunction = df(fa, fd, de)?;
    let (_, _, t_idx, a_idx) = structure_tower(de, &[fa, fd], true)?;
    let mut parts: Vec<RationalFunction> = t_part(de, &t_idx)?;
    parts.extend(l_part(de, &a_idx)?);
    let Some(u) = structure_solve(&parts, &dff, de)? else {
        return Ok(None);
    };
    let mut n: Int = Int::ONE;
    for q in &u {
        n = ilcm(&n, &denominator(q));
    }
    let un: Vec<Rat> = u.iter().map(|q| q * Rat::from(n.clone())).collect();
    let mut terms: Vec<Basic> = t_idx
        .iter()
        .map(|i| term_expr(&de.extargs[*i - 1]))
        .collect::<PolyResult<_>>()?;
    terms.extend(a_idx.iter().map(|i| de.T[*i].clone()));
    let ans: Terms = terms.into_iter().zip(un.iter().map(number)).collect();
    let mut summands: Vec<Basic> = Vec::new();
    for (i, j) in &ans {
        summands.push(crate::core::mul::mul(&[i.clone(), j.clone()])?);
    }
    let result: Basic = crate::core::add::add(&summands)?;
    let mut elements: Vec<RationalFunction> = t_idx
        .iter()
        .map(|i| term_element(&de.extargs[*i - 1]))
        .collect::<PolyResult<_>>()?;
    elements.extend(a_idx.iter().map(|i| RationalFunction::generator(&de.T[*i])));
    let mut result_el: RationalFunction = zero();
    for (e, j) in elements.iter().zip(&un) {
        result_el = result_el.add(&e.mul(&RationalFunction::from_fraction(j))?)?;
    }
    let nb: RationalFunction = RationalFunction::from_int(&n);
    let constant: RationalFunction = el(fa, fd)?.sub(&result_el.div(&nb)?)?;
    Ok(Some((ans, result, to_i64(&n)?, constant.as_expr()?)))
}

/// `(n, u)` with `n f = Du/u`, `u` in the field
/// (`is_log_deriv_k_t_radical_in_field`).
pub fn is_log_deriv_k_t_radical_in_field(
    fa: &TowerPoly,
    fd: &TowerPoly,
    de: &mut DifferentialExtension,
    case: Option<Case>,
) -> PolyResult<Option<(i64, RationalFunction)>> {
    let (fa, fd) = (in_t(fa, de)?, in_t(fd, de)?);
    let (fa, fd) = fa.cancel(&fd)?;
    let z: Basic = unique_dummy("z");
    let (h, b) = residue_reduce(&fa, &fd, de, Some(&z), true)?;
    if !b {
        return Ok(None);
    }
    let mut roots: Vec<(TowerPoly, Vec<RationalFunction>)> = Vec::new();
    for (i, _) in &h {
        roots.push((i.clone(), rational_roots(i)?));
    }
    if !roots.iter().all(|(i, j)| j.len() as i64 == i.degree(None)) {
        return Ok(None);
    }
    let mut residueterms: Vec<(TowerPoly, Rat)> = Vec::new();
    for (j, (_, residues)) in roots.iter().enumerate() {
        for i in residues {
            let Some(q) = i.as_fraction() else { return Ok(None) };
            residueterms.push((h[j].1.subs(&z, i)?, q));
        }
    }
    let pe: RationalFunction = el(&fa, &fd)?.sub(&residue_derivation_element(&h, de, &z)?)?;
    let Some(p) = TowerPoly::from_element(&pe, &de.t)? else {
        return Ok(None);
    };
    let t: Basic = de.t.clone();
    if p.degree(Some(&t)) >= 1.max(de.d.degree(Some(&t))) {
        return Ok(None);
    }
    let case: Case = case.unwrap_or(de.case);
    let (n, u): (i64, RationalFunction) = match case {
        Case::Exp => {
            let (wa, wd) = de.d.cancel(&monomial_t(&t))?;
            let found = below(de, |de| {
                let (pa, pd) = frac_in_pair(&p, &TowerPoly::one(&p.gen), &de.t, true)?;
                let (wa, wd) = frac_in_pair(&wa, &wd, &de.t, false)?;
                parametric_log_deriv(&pa, &pd, &wa, &wd, de)
            })?;
            let Some((n, e, u_found)) = found else { return Ok(None) };
            (n, u_found.mul(&RationalFunction::generator(&t).pow(e)?)?)
        }
        Case::Primitive => {
            let radical = below(de, |de| {
                let (pa, pd) = frac_in_pair(&p, &TowerPoly::one(&p.gen), &de.t, false)?;
                is_log_deriv_k_t_radical_in_field(&pa, &pd, de, None)
            })?;
            let Some(r) = radical else { return Ok(None) };
            r
        }
        Case::Base => {
            if fd.gcd(&fd.diff(None)?)?.degree(None) > 0 || fa.degree(None) >= fd.degree(None) {
                return Ok(None);
            }
            let mut n: Int = Int::ONE;
            for (_, q) in &residueterms {
                n = ilcm(&n, &denominator(q));
            }
            let mut u: RationalFunction = one();
            for (poly_, mult) in &residueterms {
                let k: i64 = to_i64(&numerator(&(mult * Rat::from(n.clone()))))?;
                u = u.mul(&poly_.as_element()?.pow(k)?)?;
            }
            return Ok(Some((to_i64(&n)?, u)));
        }
        Case::Tan => {
            if tower_has_i(de, &[&fa, &fd]) {
                return Err(PolyError::NotImplemented(
                    "The hypertangent case of is_log_deriv_k_t_radical_in_field() requires sqrt(-1) not in k(t)."
                        .to_string(),
                ));
            }
            let eta: RationalFunction = de.d.quo(&t2_1(&t))?.as_element()?;
            let (a, bb) = (p.nth(0), p.nth(1));
            let Some(ratio) = bb.div(&eta.mul(&rf_int(2))?)?.as_fraction() else {
                return Ok(None);
            };
            let (na, v): (i64, RationalFunction) = if a.is_zero() {
                (1, one())
            } else {
                let radical = below(de, |de| {
                    let (aa, ad) = frac_in(&a, &de.t, false)?;
                    is_log_deriv_k_t_radical_in_field(&aa, &ad, de, None)
                })?;
                let Some(r) = radical else { return Ok(None) };
                r
            };
            let n: i64 = to_i64(&ilcm(&Int::from(na), &denominator(&ratio)))?;
            let tf: RationalFunction = RationalFunction::generator(&t);
            let power: i64 = to_i64(&numerator(&(ratio.clone() * Rat::from(Int::from(n)))))?;
            (n, v.pow(n / na)?.mul(&tf.pow(2)?.add(&one())?.pow(power)?)?)
        }
        other => {
            return Err(PolyError::Value(format!(
                "The {} case is not supported in this function.",
                other.name()
            )))
        }
    };
    let mut common: Int = Int::from(n);
    for (_, j) in &residueterms {
        common = ilcm(&common, &denominator(j));
    }
    let m: Int = &common / Int::from(n);
    if common != Int::from(n) * &m {
        return Err(PolyError::Value("Inexact division".to_string()));
    }
    let mut result: RationalFunction = u.pow(to_i64(&m)?)?;
    for (poly_, mult) in &residueterms {
        let k: i64 = to_i64(&numerator(&(mult * Rat::from(common.clone()))))?;
        result = result.mul(&poly_.as_element()?.pow(k)?)?;
    }
    Ok(Some((to_i64(&common)?, result)))
}
