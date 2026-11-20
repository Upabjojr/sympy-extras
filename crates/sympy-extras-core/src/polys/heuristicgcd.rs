//! Port of `sympy_extras/polys/heuristicgcd.py`: the heuristic gcd of
//! multivariate polynomials over `ZZ` (Char, Geddes and Gonnet): the first
//! variable evaluated at a large integer, the gcd of the images computed
//! recursively, the gcd interpolated back from its balanced digits in that
//! integer and checked by division.

// ---------------------------------------------------------------------
// PORTED FROM SYMPY (through sympy_extras/polys/heuristicgcd.py):
// sympy/polys/heuristicgcd.py (heugcd, _gcd_interpolate),
// sympy/polys/rings.py (PolyElement.deflate, inflate) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: typed; the evaluation drops the variable into an explicit
// smaller ring; failures are PolyError::HeuristicGCDFailed.
// ---------------------------------------------------------------------

use std::sync::Arc;

use crate::error::{PolyError, PolyResult};
use crate::numbers::*;

use super::domains::{IntegerRing, ZZ};
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

pub const HEU_GCD_MAX: usize = 6;

type IntPoly = PolyElement<IntegerRing>;

/// `(h, f/h, g/h)`.
pub type Cofactors = (IntPoly, IntPoly, IntPoly);

fn max_norm(f: &IntPoly) -> Int {
    f.terms.values().map(abs).max().unwrap_or(Int::ZERO)
}

fn content(f: &IntPoly) -> Int {
    let mut result: Int = Int::ZERO;
    for c in f.terms.values() {
        result = gcd(&result, c);
    }
    result
}

/// The coefficients floor-divided by `c` (Python's `//`), the zeros dropped.
fn quo_ground(f: &IntPoly, c: &Int) -> IntPoly {
    f.ring
        .from_dict(f.terms.iter().map(|(m, v)| (m.clone(), div_floor(v, c))).collect())
}

/// `f / h` when `h` divides `f` exactly over `ZZ`, else `None`.
fn divides(f: &IntPoly, h: &IntPoly) -> Option<IntPoly> {
    if h.terms.is_empty() {
        return None;
    }
    match f.div(std::slice::from_ref(h)) {
        Ok((mut quotients, remainder)) => {
            if remainder.terms.is_empty() {
                Some(quotients.remove(0))
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

fn evaluate_first(f: &IntPoly, x: &Int, rest: &Arc<PolyRing<IntegerRing>>) -> IntPoly {
    let mut terms: Terms<Int> = Terms::new();
    for (m, c) in &f.terms {
        let key: Monomial = m[1..].to_vec();
        let value: Int = c * x.pow(m[0] as usize);
        let total: Int = match terms.get(&key) {
            Some(existing) => existing + value,
            None => value,
        };
        terms.insert(key, total);
    }
    rest.from_dict(terms)
}

fn positive(f: IntPoly) -> IntPoly {
    if !f.terms.is_empty() && is_negative(&f.LC()) {
        f.neg()
    } else {
        f
    }
}

/// The polynomial in `ring` whose coefficients of the first variable are
/// the balanced digits of `h` (a polynomial in the other ones) in base
/// `x`, made positive.
fn interpolate(h: &IntPoly, x: &Int, ring: &Arc<PolyRing<IntegerRing>>) -> IntPoly {
    let mut terms: Terms<Int> = Terms::new();
    for (m, c) in &h.terms {
        for (i, g) in balanced_digits(c, x).into_iter().enumerate() {
            if !g.is_zero() {
                let mut key: Monomial = vec![i as u32];
                key.extend_from_slice(m);
                terms.insert(key, g);
            }
        }
    }
    positive(ring.from_dict(terms))
}

/// The balanced digits of `c` in base `x`, the lowest first, as the loop
/// `g = balanced(c, x); c = (c - g) // x` gives them, by splitting at
/// `x**(2**k)` (`_balanced_digits`; the loop is quadratic in the size of
/// `c`, issue #25). The digits lie in `[low, low + x - 1]` with
/// `low = x//2 - x + 1`, so the numbers with `n` digits are the `x**n`
/// consecutive integers from `low*(x**n - 1)/(x - 1)`.
fn balanced_digits(c: &Int, x: &Int) -> Vec<Int> {
    if c.is_zero() {
        return Vec::new();
    }
    let low: Int = div_floor(x, &Int::from(2u8)) - x + Int::ONE;
    let mut powers: Vec<Int> = vec![x.clone()];
    let mut starts: Vec<Int> = vec![low];
    let inside = |v: &Int, k: usize, powers: &[Int], starts: &[Int]| -> bool {
        starts[k] <= *v && *v <= &starts[k] + &powers[k] - Int::ONE
    };
    while !inside(c, powers.len() - 1, &powers, &starts) {
        let big_x: Int = powers[powers.len() - 1].clone();
        let last: Int = starts[starts.len() - 1].clone();
        starts.push(&last * &big_x + &last);
        powers.push(&big_x * &big_x);
    }
    fn split(v: Int, k: usize, powers: &[Int], starts: &[Int], digits: &mut Vec<Int>) {
        if k == 0 {
            digits.push(v);
            return;
        }
        let (big_x, l) = (&powers[k - 1], &starts[k - 1]);
        let lower: Int = mod_floor(&(&v - l), big_x) + l;
        let upper: Int = div_floor(&(&v - &lower), big_x);
        split(lower, k - 1, powers, starts, digits);
        split(upper, k - 1, powers, starts, digits);
    }
    let mut digits: Vec<Int> = Vec::new();
    split(c.clone(), powers.len() - 1, &powers, &starts, &mut digits);
    while digits.last().is_some_and(|d| d.is_zero()) {
        digits.pop();
    }
    digits
}

/// The value of a univariate polynomial at `x`, the halves of the
/// coefficients evaluated recursively and joined with `x**(2**k)`
/// (`_horner`; Horner's scheme is quadratic on values of hundreds of
/// thousands of digits, issue #25).
fn horner(f: &IntPoly, x: &Int) -> Int {
    let n: usize = f.terms.keys().map(|m| m[0] as usize).max().unwrap_or(0) + 1;
    let mut dense: Vec<Int> = vec![Int::ZERO; n];
    for (m, c) in &f.terms {
        dense[m[0] as usize] = c.clone();
    }
    evaluate_dense(&dense, &square_powers(x, n))
}

/// `[x, x**2, x**4, ...]` up to the first `x**(2**k)` with `2**k >= n`
/// (`_square_powers`).
fn square_powers(x: &Int, n: usize) -> Vec<Int> {
    let mut powers: Vec<Int> = vec![x.clone()];
    while (1usize << (powers.len() - 1)) < n {
        let last: Int = powers[powers.len() - 1].clone();
        powers.push(&last * &last);
    }
    powers
}

/// `sum(dense[i] * x**i)` with `powers[k] = x**(2**k)` (`_evaluate_dense`).
fn evaluate_dense(dense: &[Int], powers: &[Int]) -> Int {
    let n: usize = dense.len();
    if n <= 16 {
        let mut result: Int = Int::ZERO;
        for c in dense.iter().rev() {
            result = result * &powers[0] + c;
        }
        return result;
    }
    let k: usize = (usize::BITS - (n - 1).leading_zeros()) as usize - 1;
    let half: usize = 1 << k;
    evaluate_dense(&dense[..half], powers) + &powers[k] * evaluate_dense(&dense[half..], powers)
}

fn interpolate_integer(h: &Int, x: &Int, ring: &Arc<PolyRing<IntegerRing>>) -> IntPoly {
    let mut terms: Terms<Int> = Terms::new();
    for (i, g) in balanced_digits(h, x).into_iter().enumerate() {
        if !g.is_zero() {
            terms.insert(vec![i as u32], g);
        }
    }
    positive(ring.from_dict(terms))
}

fn primitive_part(f: &IntPoly) -> IntPoly {
    let c: Int = content(f);
    if c.is_zero() || c.is_one() {
        f.clone()
    } else {
        quo_ground(f, &c)
    }
}

/// `(h, f/h, g/h)` with `h` the gcd of two nonzero polynomials over `ZZ`
/// (positive leading coefficient); `HeuristicGCDFailed` when the
/// evaluation points tried give no answer.
pub fn heugcd(f: &IntPoly, g: &IntPoly) -> PolyResult<Cofactors> {
    let ring: Arc<PolyRing<IntegerRing>> = Arc::clone(&f.ring);
    let common: Int = gcd(&content(f), &content(g));
    let (f, g): (IntPoly, IntPoly) = if common > Int::ONE {
        (quo_ground(f, &common), quo_ground(g, &common))
    } else {
        (f.clone(), g.clone())
    };
    let (f_norm, g_norm) = (max_norm(&f), max_norm(&g));
    let b: Int = Int::from(2u8) * (&f_norm).min(&g_norm) + Int::from(29u8);
    let ratio: Int = div_floor(&f_norm, &abs(&f.LC())).min(div_floor(&g_norm, &abs(&g.LC())));
    let mut x: Int = (&b)
        .min(&(Int::from(99u8) * isqrt(&b)))
        .clone()
        .max(Int::from(2u8) * ratio + Int::from(4u8));
    let rest: Option<Arc<PolyRing<IntegerRing>>> = if ring.ngens > 1 {
        Some(PolyRing::new(ZZ, ring.names[1..].to_vec(), MonomialOrder::Lex)?)
    } else {
        None
    };
    for _ in 0..HEU_GCD_MAX {
        match &rest {
            None => {
                let (ff, gg) = (horner(&f, &x), horner(&g, &x));
                if !ff.is_zero() && !gg.is_zero() {
                    let hi: Int = gcd(&ff, &gg);
                    let h: IntPoly = primitive_part(&interpolate_integer(&hi, &x, &ring));
                    if let Some(found) = check(&f, &g, &h, &common) {
                        return Ok(found);
                    }
                    let cff: IntPoly = interpolate_integer(&div_floor(&ff, &hi), &x, &ring);
                    if let Some(found) = check_cofactor(&f, &g, &cff, &common, true) {
                        return Ok(found);
                    }
                    let cfg: IntPoly = interpolate_integer(&div_floor(&gg, &hi), &x, &ring);
                    if let Some(found) = check_cofactor(&g, &f, &cfg, &common, false) {
                        return Ok(found);
                    }
                }
            }
            Some(rest) => {
                let (fe, ge) = (evaluate_first(&f, &x, rest), evaluate_first(&g, &x, rest));
                if !fe.terms.is_empty() && !ge.terms.is_empty() {
                    let (he, cffe, cfge) = heugcd(&fe, &ge)?;
                    let h: IntPoly = primitive_part(&interpolate(&he, &x, &ring));
                    if let Some(found) = check(&f, &g, &h, &common) {
                        return Ok(found);
                    }
                    if let Some(found) = check_cofactor(&f, &g, &interpolate(&cffe, &x, &ring), &common, true) {
                        return Ok(found);
                    }
                    if let Some(found) = check_cofactor(&g, &f, &interpolate(&cfge, &x, &ring), &common, false) {
                        return Ok(found);
                    }
                }
            }
        }
        x = div_floor(&(Int::from(73794u32) * &x * isqrt(&isqrt(&x))), &Int::from(27011u32));
    }
    Err(PolyError::HeuristicGCDFailed("no luck".to_string()))
}

fn check(f: &IntPoly, g: &IntPoly, h: &IntPoly, common: &Int) -> Option<Cofactors> {
    let cff: IntPoly = divides(f, h)?;
    let cfg: IntPoly = divides(g, h)?;
    Some((h.mul_ground(common), cff, cfg))
}

/// `h = a / cofactor` checked against `b`; the cofactors in the order `(f,
/// g)`.
fn check_cofactor(a: &IntPoly, b: &IntPoly, cofactor: &IntPoly, common: &Int, first: bool) -> Option<Cofactors> {
    let h: IntPoly = divides(a, cofactor)?;
    let other: IntPoly = divides(b, &h)?;
    if first {
        Some((h.mul_ground(common), cofactor.clone(), other))
    } else {
        Some((h.mul_ground(common), other, cofactor.clone()))
    }
}

/// The gcd `J` of the exponents of each variable (`1` for the variables
/// which do not occur) and the polynomials in `x_i**J_i`.
pub fn deflate(polys: &[IntPoly]) -> (Monomial, Vec<IntPoly>) {
    let n: usize = polys[0].ring.ngens;
    let mut j: Vec<u32> = vec![0; n];
    for p in polys {
        for m in p.terms.keys() {
            for (i, e) in m.iter().enumerate() {
                j[i] = gcd_u32(j[i], *e);
            }
        }
    }
    let steps: Monomial = j.iter().map(|s| if *s == 0 { 1 } else { *s }).collect();
    if steps.iter().all(|s| *s == 1) {
        return (steps, polys.to_vec());
    }
    let deflated: Vec<IntPoly> = polys
        .iter()
        .map(|p| {
            p.ring.from_dict(
                p.terms
                    .iter()
                    .map(|(m, c)| (m.iter().zip(&steps).map(|(e, s)| e / s).collect(), c.clone()))
                    .collect(),
            )
        })
        .collect();
    (steps, deflated)
}

pub fn inflate(f: &IntPoly, steps: &[u32]) -> IntPoly {
    f.ring.from_dict(
        f.terms
            .iter()
            .map(|(m, c)| (m.iter().zip(steps).map(|(e, s)| e * s).collect(), c.clone()))
            .collect(),
    )
}

fn gcd_u32(a: u32, b: u32) -> u32 {
    let (mut a, mut b): (u32, u32) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctest() {
        let r = PolyRing::new(ZZ, vec!["x".to_string(), "y".to_string()], MonomialOrder::Lex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let f = x.add(y).mul(&x.mul_ground(&int(2)).sub(y));
        let h = x.add(y).mul(&x.add(&r.from_int(3)));
        let (gcd, cff, cfg) = heugcd(&f, &h).unwrap();
        assert_eq!(
            (gcd.to_string(), cff.to_string(), cfg.to_string()),
            ("x + y".to_string(), "2*x - y".to_string(), "x + 3".to_string())
        );
    }

    #[test]
    fn univariate_and_deflation() {
        let r = PolyRing::new(ZZ, vec!["x".to_string()], MonomialOrder::Lex).unwrap();
        let x = &r.gens()[0];
        let f = x.pow(4).sub(&r.one()).mul_ground(&int(6));
        let g = x.pow(2).sub(&r.one()).mul_ground(&int(4));
        let (h, cff, cfg) = heugcd(&f, &g).unwrap();
        assert_eq!(h.to_string(), "2*x**2 - 2");
        assert_eq!(cff.to_string(), "3*x**2 + 3");
        assert_eq!(cfg.to_string(), "2");
        let (steps, deflated) = deflate(&[f.clone(), g.clone()]);
        assert_eq!(steps, vec![2]);
        assert_eq!(deflated[1].to_string(), "4*x - 4");
        assert_eq!(inflate(&deflated[0], &steps), f);
    }
}
