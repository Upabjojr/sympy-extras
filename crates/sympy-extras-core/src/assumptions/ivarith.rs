//! Interval arithmetic with outward rounding, as mpmath's `iv` context
//! (`mpmath/libmp/libmpi.py`, `mpmath/ctx_iv.py`) computes it at a given
//! precision (53 bits by default): the endpoints are binary floating
//! point numbers (exact dyadic rationals here, with the two infinities),
//! the sums, products and quotients are the exact results rounded down
//! (lower endpoint) or up (upper endpoint) to the precision, as mpmath's
//! correctly rounded `mpf` operations give them. The transcendental
//! functions (`exp`, `log`, `atan`, `cos`/`sin`, the constants) are
//! evaluated with many guard bits and rounded the same way, which gives
//! mpmath's endpoints except at values within `2**-100` of a rounding
//! boundary; `cos`/`sin` follow `mpi_cos_sin` (the values at the endpoints
//! truncated to `prec + 20` bits, the quadrants, the perturbation).

use crate::core::evalf::{real_rat, Real};
use crate::numbers::*;
use dashu_base::{BitTest, UnsignedAbs};
use dashu_int::UBig;

/// The precision of mpmath's default context.
pub const PREC: usize = 53;

/// The guard bits of the transcendental functions.
const GUARD: usize = 96;

/// An endpoint: a dyadic rational or an infinity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum E {
    NInf,
    F(Rat),
    PInf,
}

impl PartialOrd for E {
    fn partial_cmp(&self, other: &E) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for E {
    fn cmp(&self, other: &E) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match (self, other) {
            (E::NInf, E::NInf) | (E::PInf, E::PInf) => Equal,
            (E::NInf, _) | (_, E::PInf) => Less,
            (_, E::NInf) | (E::PInf, _) => Greater,
            (E::F(a), E::F(b)) => a.cmp(b),
        }
    }
}

/// The rounding modes of mpmath (`round_floor`, `round_ceiling`,
/// `round_down`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rnd {
    Floor,
    Ceil,
    Down,
}

fn sign_of(r: &Rat) -> i32 {
    if r.is_zero() {
        0
    } else if rat_is_negative(r) {
        -1
    } else {
        1
    }
}

impl E {
    pub fn zero() -> E {
        E::F(Rat::ZERO)
    }
    pub fn one() -> E {
        E::F(Rat::ONE)
    }
    pub fn int(n: i64) -> E {
        E::F(Rat::from(n))
    }
    /// `mpf_sign`.
    pub fn sign(&self) -> i32 {
        match self {
            E::NInf => -1,
            E::PInf => 1,
            E::F(r) => sign_of(r),
        }
    }
    fn neg(&self) -> E {
        match self {
            E::NInf => E::PInf,
            E::PInf => E::NInf,
            E::F(r) => E::F(-r.clone()),
        }
    }
    pub fn is_zero(&self) -> bool {
        matches!(self, E::F(r) if r.is_zero())
    }
}

fn pow2(e: isize) -> Rat {
    if e >= 0 {
        Rat::from(Int::ONE << (e as usize))
    } else {
        Rat::from_parts(Int::ONE, UBig::ONE << ((-e) as usize))
    }
}

/// The floor of log2 of a positive rational.
fn ilog2(a: &Rat) -> isize {
    let n: usize = numer(a).clone().unsigned_abs().bit_len();
    let d: usize = denom(a).bit_len();
    let mut e: isize = n as isize - d as isize;
    // 2**e <= a < 2**(e + 1)
    while pow2(e) > *a {
        e -= 1;
    }
    while pow2(e + 1) <= *a {
        e += 1;
    }
    e
}

/// `r` rounded to `prec` bits of mantissa.
pub fn round(r: &Rat, prec: usize, rnd: Rnd) -> Rat {
    if r.is_zero() {
        return Rat::ZERO;
    }
    let negative: bool = rat_is_negative(r);
    let a: Rat = if negative { -r.clone() } else { r.clone() };
    let e: isize = ilog2(&a);
    let shift: isize = prec as isize - 1 - e;
    let scaled: Rat = &a * &pow2(shift);
    let (q, rem) = (numer(&scaled), denom(&scaled));
    let floor_m: Int = q.clone() / Int::from(rem.clone());
    let exact: bool = Rat::from(floor_m.clone()) == scaled;
    // the magnitude rounded up or down
    let up: bool = match rnd {
        Rnd::Down => false,
        Rnd::Floor => negative,
        Rnd::Ceil => !negative,
    };
    let m: Int = if up && !exact { floor_m + Int::ONE } else { floor_m };
    let v: Rat = Rat::from(m) * pow2(-shift);
    if negative {
        -v
    } else {
        v
    }
}

/// The largest binary exponent of a finite endpoint: beyond it an
/// endpoint rounded outwards is an infinity, inwards the bound itself (a
/// sound enclosure, which keeps the exact rationals small; mpmath's
/// exponents are unbounded).
const MAX_EXPONENT: isize = 1 << 20;

/// `r` rounded to `prec` bits as an endpoint, clamped to the exponent
/// range.
fn rnd_e(r: &Rat, prec: usize, rnd: Rnd) -> E {
    if r.is_zero() {
        return E::zero();
    }
    let negative: bool = rat_is_negative(r);
    let a: Rat = if negative { -r.clone() } else { r.clone() };
    let e: isize = ilog2(&a);
    let up: bool = match rnd {
        Rnd::Down => false,
        Rnd::Floor => negative,
        Rnd::Ceil => !negative,
    };
    if e > MAX_EXPONENT {
        return match (negative, up) {
            (false, true) => E::PInf,
            (true, true) => E::NInf,
            (false, false) => E::F(pow2(MAX_EXPONENT)),
            (true, false) => E::F(-pow2(MAX_EXPONENT)),
        };
    }
    if e < -MAX_EXPONENT {
        return match (negative, up) {
            (_, false) => E::zero(),
            (false, true) => E::F(pow2(-MAX_EXPONENT)),
            (true, true) => E::F(-pow2(-MAX_EXPONENT)),
        };
    }
    E::F(round(r, prec, rnd))
}

fn round_e(e: E, prec: usize, rnd: Rnd) -> E {
    match e {
        E::F(r) => rnd_e(&r, prec, rnd),
        other => other,
    }
}

/// `mpf_add` (`None` for `nan`).
fn f_add(x: &E, y: &E, prec: usize, rnd: Rnd) -> Option<E> {
    match (x, y) {
        (E::PInf, E::NInf) | (E::NInf, E::PInf) => None,
        (E::PInf, _) | (_, E::PInf) => Some(E::PInf),
        (E::NInf, _) | (_, E::NInf) => Some(E::NInf),
        (E::F(a), E::F(b)) => Some(rnd_e(&(a + b), prec, rnd)),
    }
}

fn f_mul_exact(x: &E, y: &E) -> Option<E> {
    match (x, y) {
        (E::F(a), E::F(b)) => Some(E::F(a * b)),
        _ => {
            let s: i32 = x.sign() * y.sign();
            if s == 0 {
                None
            } else if s > 0 {
                Some(E::PInf)
            } else {
                Some(E::NInf)
            }
        }
    }
}

fn f_mul(x: &E, y: &E, prec: usize, rnd: Rnd) -> Option<E> {
    f_mul_exact(x, y).map(|v| round_e(v, prec, rnd))
}

fn f_div(x: &E, y: &E, prec: usize, rnd: Rnd) -> Option<E> {
    match (x, y) {
        (E::F(a), E::F(b)) => {
            if b.is_zero() {
                None
            } else {
                Some(rnd_e(&(a / b), prec, rnd))
            }
        }
        (E::F(_), _) => Some(E::zero()),
        (_, E::F(b)) => {
            let s: i32 = x.sign() * sign_of(b);
            if s == 0 {
                None
            } else if s > 0 {
                Some(E::PInf)
            } else {
                Some(E::NInf)
            }
        }
        _ => None,
    }
}

fn f_pow_int(x: &E, n: u64, prec: usize, rnd: Rnd) -> E {
    match x {
        E::F(r)
            if !r.is_zero() && {
                let a: Rat = if rat_is_negative(r) { -r.clone() } else { r.clone() };
                ilog2(&a).unsigned_abs().saturating_mul(n as usize) > (MAX_EXPONENT as usize) * 2
            } =>
        {
            // far beyond the exponent range: the clamped bound
            let negative: bool = rat_is_negative(r) && n % 2 == 1;
            let a: Rat = if rat_is_negative(r) { -r.clone() } else { r.clone() };
            let big: bool = ilog2(&a) > 0;
            let m: Rat = if big {
                pow2(2 * MAX_EXPONENT)
            } else {
                pow2(-2 * MAX_EXPONENT)
            };
            rnd_e(&if negative { -m } else { m }, prec, rnd)
        }
        E::F(r) => {
            let mut p: Rat = Rat::ONE;
            for _ in 0..n {
                p = &p * r;
            }
            rnd_e(&p, prec, rnd)
        }
        E::PInf => E::PInf,
        E::NInf => {
            if n % 2 == 0 {
                E::PInf
            } else {
                E::NInf
            }
        }
    }
}

fn to_real(r: &Rat, prec: usize) -> Real {
    real_rat(r, prec)
}

/// A float as an exact rational.
pub fn real_to_rat(v: &Real) -> Rat {
    let repr = v.repr();
    let m: Int = repr.significand().clone();
    let e: isize = repr.exponent();
    Rat::from(m) * pow2(e)
}

fn rounded_real(v: &Real, prec: usize, rnd: Rnd) -> E {
    rnd_e(&real_to_rat(v), prec, rnd)
}

fn f_exp(x: &E, prec: usize, rnd: Rnd) -> E {
    match x {
        E::NInf => E::zero(),
        E::PInf => E::PInf,
        E::F(r) if r.is_zero() => E::one(),
        E::F(r) if r.clone() > Rat::from(MAX_EXPONENT as i64) || r.clone() < Rat::from(-(MAX_EXPONENT as i64)) => {
            // beyond the exponent range: the clamped bounds
            let huge: Rat = if rat_is_negative(r) {
                pow2(-MAX_EXPONENT - 1)
            } else {
                pow2(MAX_EXPONENT + 1)
            };
            rnd_e(&huge, prec, rnd)
        }
        E::F(r) => rounded_real(&to_real(r, prec + GUARD).exp(), prec, rnd),
    }
}

fn f_log(x: &E, prec: usize, rnd: Rnd) -> Option<E> {
    match x {
        E::NInf => None,
        E::PInf => Some(E::PInf),
        E::F(r) if r.is_zero() => Some(E::NInf),
        E::F(r) if rat_is_negative(r) => None,
        E::F(r) if *r == Rat::ONE => Some(E::zero()),
        E::F(r) => Some(rounded_real(&to_real(r, prec + GUARD).ln(), prec, rnd)),
    }
}

fn pi_real(prec: usize) -> Real {
    Real::pi(prec)
}

fn f_pi(prec: usize, rnd: Rnd) -> E {
    rounded_real(&pi_real(prec + GUARD), prec, rnd)
}

fn f_e(prec: usize, rnd: Rnd) -> E {
    rounded_real(&to_real(&Rat::ONE, prec + GUARD).exp(), prec, rnd)
}

fn negative_rnd(rnd: Rnd) -> Rnd {
    match rnd {
        Rnd::Floor => Rnd::Ceil,
        Rnd::Ceil => Rnd::Floor,
        Rnd::Down => Rnd::Down,
    }
}

/// `mpf_atan2(y, 1, prec, rnd)` as mpmath computes it: the arctangent
/// truncated to `prec + 4` bits, then rounded (a double rounding which
/// can leave the endpoint on the wrong side of the true value: mpmath's
/// `iv.atan2` is not a rigorous enclosure, mirrored here).
fn f_atan(x: &E, prec: usize, rnd: Rnd) -> E {
    match x {
        E::F(r) if r.is_zero() => E::zero(),
        E::PInf => match f_pi(prec, rnd) {
            E::F(p) => E::F(p / Rat::from(2)),
            other => other,
        },
        E::NInf => match f_pi(prec, negative_rnd(rnd)) {
            E::F(p) => E::F(-(p / Rat::from(2))),
            other => other,
        },
        E::F(r) if rat_is_negative(r) => f_atan(&E::F(-r.clone()), prec, negative_rnd(rnd)).neg(),
        E::F(r) => {
            let t: Rat = round(&real_to_rat(&to_real(r, prec + GUARD).atan()), prec + 4, Rnd::Down);
            E::F(round(&t, prec, rnd))
        }
    }
}

/// An interval `[a, b]` (mpmath's `ivmpf`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iv {
    pub a: E,
    pub b: E,
}

fn nan_to(v: Option<E>, fallback: E) -> E {
    v.unwrap_or(fallback)
}

impl Iv {
    pub fn new(a: E, b: E) -> Iv {
        Iv { a, b }
    }
    pub fn point(e: E) -> Iv {
        Iv { a: e.clone(), b: e }
    }
    /// An integer (`convert`: `from_int` rounded down and up).
    pub fn from_int(n: &Int, prec: usize) -> Iv {
        let r: Rat = Rat::from(n.clone());
        Iv {
            a: E::F(round(&r, prec, Rnd::Floor)),
            b: E::F(round(&r, prec, Rnd::Ceil)),
        }
    }
    pub fn pi(prec: usize) -> Iv {
        Iv {
            a: f_pi(prec, Rnd::Floor),
            b: f_pi(prec, Rnd::Ceil),
        }
    }
    pub fn e(prec: usize) -> Iv {
        Iv {
            a: f_e(prec, Rnd::Floor),
            b: f_e(prec, Rnd::Ceil),
        }
    }

    /// `mpi_add`.
    pub fn add(&self, t: &Iv, prec: usize) -> Iv {
        let a: E = nan_to(f_add(&self.a, &t.a, prec, Rnd::Floor), E::NInf);
        let b: E = nan_to(f_add(&self.b, &t.b, prec, Rnd::Ceil), E::PInf);
        Iv { a, b }
    }
    /// `mpi_sub`.
    pub fn sub(&self, t: &Iv, prec: usize) -> Iv {
        let a: E = nan_to(f_add(&self.a, &t.b.neg(), prec, Rnd::Floor), E::NInf);
        let b: E = nan_to(f_add(&self.b, &t.a.neg(), prec, Rnd::Ceil), E::PInf);
        Iv { a, b }
    }
    /// `mpi_neg`.
    pub fn neg(&self) -> Iv {
        Iv {
            a: self.b.neg(),
            b: self.a.neg(),
        }
    }
    /// `mpi_abs`.
    pub fn abs(&self, prec: usize) -> Iv {
        let (sas, sbs) = (self.a.sign(), self.b.sign());
        if sas >= 0 {
            Iv {
                a: round_e(self.a.clone(), prec, Rnd::Floor),
                b: round_e(self.b.clone(), prec, Rnd::Ceil),
            }
        } else if sbs >= 0 {
            let negsa: E = self.a.neg();
            let b: E = if negsa < self.b {
                round_e(self.b.clone(), prec, Rnd::Ceil)
            } else {
                round_e(negsa, prec, Rnd::Ceil)
            };
            Iv { a: E::zero(), b }
        } else {
            Iv {
                a: round_e(self.b.neg(), prec, Rnd::Floor),
                b: round_e(self.a.neg(), prec, Rnd::Ceil),
            }
        }
    }
    /// `mpi_mul`.
    pub fn mul(&self, t: &Iv, prec: usize) -> Iv {
        let (sa, sb, ta, tb) = (&self.a, &self.b, &t.a, &t.b);
        let (sas, sbs, tas, tbs) = (sa.sign(), sb.sign(), ta.sign(), tb.sign());
        if sas == 0 && sbs == 0 {
            if *ta == E::NInf || *tb == E::PInf {
                return Iv { a: E::NInf, b: E::PInf };
            }
            return Iv {
                a: E::zero(),
                b: E::zero(),
            };
        }
        if tas == 0 && tbs == 0 {
            if *sa == E::NInf || *sb == E::PInf {
                return Iv { a: E::NInf, b: E::PInf };
            }
            return Iv {
                a: E::zero(),
                b: E::zero(),
            };
        }
        let fl = |x: &E, y: &E, nan: E| nan_to(f_mul(x, y, prec, Rnd::Floor), nan);
        let ce = |x: &E, y: &E, nan: E| nan_to(f_mul(x, y, prec, Rnd::Ceil), nan);
        if sas >= 0 {
            if tas >= 0 {
                Iv {
                    a: fl(sa, ta, E::zero()),
                    b: ce(sb, tb, E::PInf),
                }
            } else if tbs <= 0 {
                Iv {
                    a: fl(sb, ta, E::NInf),
                    b: ce(sa, tb, E::zero()),
                }
            } else {
                Iv {
                    a: fl(sb, ta, E::NInf),
                    b: ce(sb, tb, E::PInf),
                }
            }
        } else if sbs <= 0 {
            if tas >= 0 {
                Iv {
                    a: fl(sa, tb, E::NInf),
                    b: ce(sb, ta, E::zero()),
                }
            } else if tbs <= 0 {
                Iv {
                    a: fl(sb, tb, E::zero()),
                    b: ce(sa, ta, E::PInf),
                }
            } else {
                Iv {
                    a: fl(sa, tb, E::NInf),
                    b: ce(sa, ta, E::PInf),
                }
            }
        } else {
            let cases: Vec<Option<E>> = vec![
                f_mul_exact(sa, ta),
                f_mul_exact(sa, tb),
                f_mul_exact(sb, ta),
                f_mul_exact(sb, tb),
            ];
            if cases.iter().any(|c| c.is_none()) {
                return Iv { a: E::NInf, b: E::PInf };
            }
            let vals: Vec<E> = cases.into_iter().flatten().collect();
            let lo: E = vals.iter().min().expect("four").clone();
            let hi: E = vals.iter().max().expect("four").clone();
            Iv {
                a: round_e(lo, prec, Rnd::Floor),
                b: round_e(hi, prec, Rnd::Ceil),
            }
        }
    }
    /// `mpi_square`.
    fn square(&self, prec: usize) -> Iv {
        let (sa, sb) = (&self.a, &self.b);
        let zero: E = E::zero();
        if *sa >= zero {
            Iv {
                a: nan_to(f_mul(sa, sa, prec, Rnd::Floor), E::zero()),
                b: nan_to(f_mul(sb, sb, prec, Rnd::Ceil), E::PInf),
            }
        } else if *sb <= zero {
            Iv {
                a: nan_to(f_mul(sb, sb, prec, Rnd::Floor), E::zero()),
                b: nan_to(f_mul(sa, sa, prec, Rnd::Ceil), E::PInf),
            }
        } else {
            let nsa: E = sa.neg();
            let big: E = if nsa > *sb { nsa } else { sb.clone() };
            Iv {
                a: E::zero(),
                b: nan_to(f_mul(&big, &big, prec, Rnd::Ceil), E::PInf),
            }
        }
    }
    /// `mpi_div`.
    pub fn div(&self, t: &Iv, prec: usize) -> Iv {
        let (sa, sb, ta, tb) = (&self.a, &self.b, &t.a, &t.b);
        let (sas, sbs, tas, tbs) = (sa.sign(), sb.sign(), ta.sign(), tb.sign());
        if sas == 0 && sbs == 0 {
            if (tas < 0 && tbs > 0) || (tas == 0 || tbs == 0) {
                return Iv { a: E::NInf, b: E::PInf };
            }
            return Iv {
                a: E::zero(),
                b: E::zero(),
            };
        }
        if tas < 0 && tbs > 0 {
            return Iv { a: E::NInf, b: E::PInf };
        }
        if tas < 0 {
            return self.neg().div(&t.neg(), prec);
        }
        let fl = |x: &E, y: &E, nan: E| nan_to(f_div(x, y, prec, Rnd::Floor), nan);
        let ce = |x: &E, y: &E, nan: E| nan_to(f_div(x, y, prec, Rnd::Ceil), nan);
        if tas == 0 {
            if sas < 0 && sbs > 0 {
                return Iv { a: E::NInf, b: E::PInf };
            }
            if tas == tbs {
                return Iv { a: E::NInf, b: E::PInf };
            }
            let mut a: E = E::NInf;
            let mut b: E = E::PInf;
            if sas >= 0 {
                a = fl(sa, tb, E::NInf);
                b = E::PInf;
            }
            if sbs <= 0 {
                a = E::NInf;
                b = ce(sb, tb, E::PInf);
            }
            return Iv { a, b };
        }
        if sas >= 0 {
            Iv {
                a: fl(sa, tb, E::zero()),
                b: ce(sb, ta, E::PInf),
            }
        } else if sbs <= 0 {
            Iv {
                a: fl(sa, ta, E::NInf),
                b: ce(sb, tb, E::zero()),
            }
        } else {
            Iv {
                a: fl(sa, ta, E::NInf),
                b: ce(sb, ta, E::PInf),
            }
        }
    }
    /// `mpi_pow_int`.
    pub fn pow_int(&self, n: i64, prec: usize) -> Iv {
        if n < 0 {
            return Iv::point(E::one()).div(&self.pow_int(-n, prec + 20), prec);
        }
        if n == 0 {
            return Iv::point(E::one());
        }
        if n == 1 {
            return self.clone();
        }
        if n == 2 {
            return self.square(prec);
        }
        let n_u: u64 = n as u64;
        let (sa, sb) = (&self.a, &self.b);
        if n % 2 == 1 {
            return Iv {
                a: f_pow_int(sa, n_u, prec, Rnd::Floor),
                b: f_pow_int(sb, n_u, prec, Rnd::Ceil),
            };
        }
        let (sas, sbs) = (sa.sign(), sb.sign());
        if sas >= 0 {
            Iv {
                a: f_pow_int(sa, n_u, prec, Rnd::Floor),
                b: f_pow_int(sb, n_u, prec, Rnd::Ceil),
            }
        } else if sbs <= 0 {
            Iv {
                a: f_pow_int(sb, n_u, prec, Rnd::Floor),
                b: f_pow_int(sa, n_u, prec, Rnd::Ceil),
            }
        } else {
            let nsa: E = sa.neg();
            let b: E = if nsa >= *sb {
                f_pow_int(&nsa, n_u, prec, Rnd::Ceil)
            } else {
                f_pow_int(sb, n_u, prec, Rnd::Ceil)
            };
            Iv { a: E::zero(), b }
        }
    }
    /// `mpi_exp`.
    pub fn exp(&self, prec: usize) -> Iv {
        Iv {
            a: f_exp(&self.a, prec, Rnd::Floor),
            b: f_exp(&self.b, prec, Rnd::Ceil),
        }
    }
    /// `mpi_log` (`None` where mpmath raises or gives a complex result).
    pub fn log(&self, prec: usize) -> Option<Iv> {
        Some(Iv {
            a: f_log(&self.a, prec, Rnd::Floor)?,
            b: f_log(&self.b, prec, Rnd::Ceil)?,
        })
    }
    /// `mpi_atan2(self, [1, 1])` (the right half-plane).
    pub fn atan(&self, prec: usize) -> Iv {
        if self.a.is_zero() && self.b.is_zero() {
            return Iv::point(E::zero());
        }
        Iv {
            a: f_atan(&self.a, prec, Rnd::Floor),
            b: f_atan(&self.b, prec, Rnd::Ceil),
        }
    }
    /// `mpi_cos_sin`.
    pub fn cos_sin(&self, prec: usize) -> (Iv, Iv) {
        let (a, b) = (&self.a, &self.b);
        if a.is_zero() && b.is_zero() {
            return (Iv::point(E::one()), Iv::point(E::zero()));
        }
        let full = || Iv {
            a: E::int(-1),
            b: E::one(),
        };
        if matches!(a, E::NInf | E::PInf) || matches!(b, E::NInf | E::PInf) {
            return (full(), full());
        }
        let wp: usize = prec + 20;
        let (ca0, sa0, na) = cos_sin_quadrant(a, wp);
        let (cb0, sb0, nb) = cos_sin_quadrant(b, wp);
        let (mut ca, mut cb) = if ca0 <= cb0 { (ca0, cb0) } else { (cb0, ca0) };
        let (mut sa, mut sb) = if sa0 <= sb0 { (sa0, sb0) } else { (sb0, sa0) };
        if na != nb {
            if nb - na >= 4 {
                return (full(), full());
            }
            if na.div_euclid(4) != nb.div_euclid(4) {
                cb = Rat::ONE;
            }
            if (na - 2).div_euclid(4) != (nb - 2).div_euclid(4) {
                ca = -Rat::ONE;
            }
            if (na - 1).div_euclid(4) != (nb - 1).div_euclid(4) {
                sb = Rat::ONE;
            }
            if (na - 3).div_euclid(4) != (nb - 3).div_euclid(4) {
                sa = -Rat::ONE;
            }
        }
        // the perturbation forcing the interval rounding
        let one_wp: Rat = pow2(wp as isize);
        let ten: Rat = pow2(10);
        let more: Rat = (&one_wp + &ten) * pow2(-(wp as isize));
        let less: Rat = (&one_wp - &ten) * pow2(-(wp as isize));
        let finalize = |v: &Rat, rnd: Rnd| -> E {
            let negative: bool = rat_is_negative(v);
            let p: &Rat = if negative == (rnd == Rnd::Floor) { &more } else { &less };
            let w: Rat = round(&(v * p), prec, rnd);
            let abs: Rat = if rat_is_negative(&w) { -w.clone() } else { w.clone() };
            if abs >= Rat::ONE {
                return if rat_is_negative(&w) { E::int(-1) } else { E::one() };
            }
            E::F(w)
        };
        (
            Iv {
                a: finalize(&ca, Rnd::Floor),
                b: finalize(&cb, Rnd::Ceil),
            },
            Iv {
                a: finalize(&sa, Rnd::Floor),
                b: finalize(&sb, Rnd::Ceil),
            },
        )
    }
    /// `mpi_tan`: `sin/cos` at `prec + 20`.
    pub fn tan(&self, prec: usize) -> Iv {
        let (c, s) = self.cos_sin(prec + 20);
        s.div(&c, prec)
    }
}

/// `cos_sin_quadrant`: `cos(x)`, `sin(x)` truncated to `wp` bits and the
/// quadrant `n` of `x` (`x` finite).
fn cos_sin_quadrant(x: &E, wp: usize) -> (Rat, Rat, i64) {
    let E::F(r) = x else { unreachable!("a finite endpoint") };
    if r.is_zero() {
        return (Rat::ONE, Rat::ZERO, 0);
    }
    let bits: usize = wp + GUARD + ilog2(&if rat_is_negative(r) { -r.clone() } else { r.clone() }).max(0) as usize;
    let v: Real = to_real(r, bits);
    let (s, c) = (v.sin(), v.cos());
    let cr: Rat = round(&real_to_rat(&c), wp, Rnd::Down);
    let sr: Rat = round(&real_to_rat(&s), wp, Rnd::Down);
    // the quadrant of |x| (mod_pi2), negated for a negative x
    let a: Rat = if rat_is_negative(r) { -r.clone() } else { r.clone() };
    let half_pi: Real = pi_real(bits) / to_real(&Rat::from(2), bits);
    let q: Rat = &a / &real_to_rat(&half_pi);
    let mut n: i64 = i64::try_from(&(numer(&q) / Int::from(denom(&q)))).unwrap_or(i64::MAX);
    if rat_is_negative(r) {
        n = -1 - n;
    }
    (cr, sr, n)
}
