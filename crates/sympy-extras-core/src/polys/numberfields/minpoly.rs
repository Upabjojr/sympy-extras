//! Port of `sympy_extras/polys/numberfields/minpoly.py`: the minimal
//! polynomials of algebraic numbers over the rationals by composition. The
//! minimal polynomial of every subexpression is computed first; sums and
//! products combine them by resultants ([`sum_resultant`],
//! [`product_resultant`] of `qqbar`), a rational power by
//! `Res_y(m(y), x**d - y**n)`; the factor vanishing at the number is chosen
//! numerically, the precision doubled until one factor is far smaller than
//! the others. The atoms: the rationals, `I`, the golden ratio, the
//! tribonacci constant, and `cos`, `sin`, `tan`, `exp(I*pi*c)` at rational
//! multiples `c` of `pi`. Only numbers (no `AlgebraicNumber` in the core).

// PORTED FROM SYMPY: sympy/polys/numberfields/minpoly.py (_minpoly_tan and
// the composition by resultants) (SymPy 1.14), through sympy_extras; BSD
// 3-clause license, see sympy_extras/LICENSE-SymPy.

use crate::core::add::add;
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex_real, numeric_value, parts, real_int, Complex, Real};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::{not_ported, polys_error, zero_division, PolyResult, PolysErrorKind};
use crate::numbers::*;

use super::super::densearith::{dup_mul, dup_sub};
use super::super::densetools::dup_primitive;
use super::super::domains::ZZ;
use super::super::euclidtools::dup_resultant;
use super::super::factortools::dup_zz_factor;
use super::qqbar::{in_y, product_resultant, sum_resultant, zx, BiPoly};

/// The precisions (bits) tried when choosing the vanishing factor.
const FIRST_PRECISION: usize = 64;
const LAST_PRECISION: usize = 1 << 14;

fn not_algebraic(e: &Basic) -> crate::error::PolyError {
    polys_error(
        PolysErrorKind::NotAlgebraic,
        format!("{} does not seem to be an algebraic element", e),
    )
}

fn ints(values: &[i64]) -> Vec<Int> {
    values.iter().map(|v| Int::from(*v)).collect()
}

/// The minimal polynomial of the algebraic number `e` in `x`, primitive
/// with a positive leading coefficient (`minimal_polynomial`).
pub fn minimal_polynomial(e: &Basic, x: &Basic) -> PolyResult<Basic> {
    let coefficients: Vec<Int> = minimal_polynomial_coeffs(e)?;
    let n: usize = coefficients.len() - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in coefficients.iter().enumerate() {
        if !c.is_zero() {
            terms.push(mul(&[Basic::integer(c.clone()), pow(x, &Basic::int((n - i) as i64))?])?);
        }
    }
    add(&terms)
}

/// The coefficients of [`minimal_polynomial`], highest degree first.
pub fn minimal_polynomial_coeffs(e: &Basic) -> PolyResult<Vec<Int>> {
    if !e.free_symbols().is_empty() {
        return Err(not_ported("minimal polynomials with symbolic coefficients"));
    }
    Ok(normalised(&minpoly(e)?))
}

fn normalised(f: &[Int]) -> Vec<Int> {
    let (_, p) = dup_primitive(f, &ZZ);
    if p.first().is_some_and(is_negative) {
        return p.iter().map(|c| -c).collect();
    }
    p
}

fn is_linear_x(m: &[Int]) -> bool {
    m.len() == 2 && m[0] == Int::ONE && m[1].is_zero()
}

/// A multiple of the minimal polynomial irreducible over `ZZ` (`_minpoly`).
fn minpoly(e: &Basic) -> PolyResult<Vec<Int>> {
    if let Some(v) = e.as_rational() {
        return Ok(vec![denom(v), -numer(v)]);
    }
    match e.head() {
        Head::ImaginaryUnit => return Ok(ints(&[1, 0, 1])),
        Head::GoldenRatio => return Ok(ints(&[1, -1, -1])),
        Head::TribonacciConstant => return Ok(ints(&[1, -1, -1, -1])),
        Head::Add => return fold(e.args(), true),
        Head::Mul => return fold(e.args(), false),
        Head::Pow => return power(e),
        _ => {}
    }
    if e.is_function(Function::Cos) {
        let (p, q) = rational_multiple(e, &e.args()[0], false)?;
        return choose_factor(&chebyshev_cos(&p, &q)?, e);
    }
    if e.is_function(Function::Sin) {
        // sin(c pi) = cos((1/2 - c) pi)
        let (p, q) = rational_multiple(e, &e.args()[0], false)?;
        let c: Rat = rat(1, 2) - rat_from_parts(p, q);
        return choose_factor(&chebyshev_cos(&numer(&c), &denom(&c))?, e);
    }
    if e.is_function(Function::Tan) {
        let (p, q) = rational_multiple(e, &e.args()[0], false)?;
        return choose_factor(&tangent(&p, &q)?, e);
    }
    if e.is_function(Function::Exp) {
        // exp(i pi p/q) is a root of x**q - (-1)**p
        let (p, q) = rational_multiple(e, &e.args()[0], true)?;
        let q: usize = usize::try_from(&q).map_err(|_| not_ported("a huge denominator"))?;
        let mut f: Vec<Int> = vec![Int::ONE];
        f.extend(vec![Int::ZERO; q - 1]);
        f.push(if (&p % Int::from(2u8)).is_zero() {
            -Int::ONE
        } else {
            Int::ONE
        });
        return choose_factor(&f, e);
    }
    Err(not_algebraic(e))
}

/// `(p, q)` with the argument `p/q*pi` (`p/q*I*pi` when `imaginary`).
fn rational_multiple(e: &Basic, argument: &Basic, imaginary: bool) -> PolyResult<(Int, Int)> {
    let (c, rest) = crate::core::expr::as_coeff_mul(argument);
    let want: Basic = if imaginary {
        mul(&[Basic::imaginary_unit(), Basic::pi()])?
    } else {
        Basic::pi()
    };
    if let Some(v) = c.as_rational() {
        if rest == want {
            return Ok((numer(v), denom(v)));
        }
    }
    Err(not_algebraic(e))
}

/// `T_q(x) - (-1)**p`, of which `cos(p pi/q)` is a root.
fn chebyshev_cos(p: &Int, q: &Int) -> PolyResult<Vec<Int>> {
    let q: usize = usize::try_from(q).map_err(|_| not_ported("a huge denominator"))?;
    let (mut previous, mut current): (Vec<Int>, Vec<Int>) = (ints(&[1]), ints(&[1, 0]));
    if q == 0 {
        current = previous.clone();
    }
    for _ in 0..q.saturating_sub(1) {
        let next: Vec<Int> = dup_sub(&dup_mul(&ints(&[2, 0]), &current, &ZZ), &previous, &ZZ);
        previous = std::mem::replace(&mut current, next);
    }
    let sign: i64 = if (p % Int::from(2u8)).is_zero() { 1 } else { -1 };
    Ok(dup_sub(&current, &ints(&[sign]), &ZZ))
}

/// A polynomial with the root `tan(p pi/q)`: the real or imaginary part of
/// `(1 + i x)**n` with `n` the denominator of `2 p/q` (`_minpoly_tan`).
fn tangent(p: &Int, q: &Int) -> PolyResult<Vec<Int>> {
    let c: Rat = rat_from_parts(Int::from(2u8) * p, q.clone());
    let n_int: Int = denom(&c);
    let n: usize = usize::try_from(&n_int).map_err(|_| not_ported("a huge denominator"))?;
    let num: Int = numer(&c);
    let mut a: Int = if (&num % Int::from(2u8)).is_zero() {
        n_int.clone()
    } else {
        Int::ONE
    };
    let mut coefficients: Vec<Int> = vec![Int::ZERO; n + 1];
    let start: usize = if ((&num + Int::ONE) % Int::from(2u8)).is_zero() {
        0
    } else {
        1
    };
    let mut k: usize = start;
    while k <= n {
        coefficients[k] = a.clone();
        let (kk, nn) = (Int::from(k as i64), Int::from(n as i64));
        a = div_floor(
            &-(&a * (&nn - &kk - Int::ONE) * (&nn - &kk)),
            &((&kk + Int::ONE) * (&kk + Int::from(2u8))),
        );
        k += 2;
    }
    let mut f: Vec<Int> = coefficients.into_iter().rev().collect();
    while f.first().is_some_and(|c| c.is_zero()) {
        f.remove(0);
    }
    Ok(f)
}

/// The minimal polynomial of a sum or product, one term at a time (`_fold`).
fn fold(terms: &[Basic], is_sum: bool) -> PolyResult<Vec<Int>> {
    let mut current: Basic = terms[0].clone();
    let mut m: Vec<Int> = normalised(&minpoly(&current)?);
    for t in &terms[1..] {
        let mt: Vec<Int> = normalised(&minpoly(t)?);
        current = if is_sum {
            add(&[current, t.clone()])?
        } else {
            mul(&[current, t.clone()])?
        };
        if !is_sum && (is_linear_x(&m) || is_linear_x(&mt)) {
            m = ints(&[1, 0]);
            continue;
        }
        let r: Vec<Int> = if is_sum {
            sum_resultant(&m, &mt)
        } else {
            product_resultant(&m, &mt)
        };
        if m.len() == 2 || mt.len() == 2 {
            // one of them rational: the resultant is the other polynomial
            // shifted or scaled, so irreducible
            m = normalised(&r);
        } else {
            m = normalised(&choose_factor(&r, &current)?);
        }
    }
    Ok(m)
}

fn power(e: &Basic) -> PolyResult<Vec<Int>> {
    let (base, exponent) = (&e.args()[0], &e.args()[1]);
    let Some(x) = exponent.as_rational() else {
        return Err(not_algebraic(e));
    };
    let mut m: Vec<Int> = normalised(&minpoly(base)?);
    let (mut n, d) = (numer(x), denom(x));
    if is_negative(&n) {
        if is_linear_x(&m) {
            return Err(zero_division(&format!("{} is zero", base)));
        }
        m.reverse();
        while m.first().is_some_and(|c| c.is_zero()) {
            m.remove(0);
        }
        n = -n;
    }
    if n == Int::ONE && d == Int::ONE {
        return Ok(m);
    }
    let n: usize = usize::try_from(&n).map_err(|_| not_ported("a huge exponent"))?;
    let d: usize = usize::try_from(&d).map_err(|_| not_ported("a huge exponent"))?;
    // Res_y(m(y), x**d - y**n): x**d - y**n in y over ZZ[x]
    let mut target: BiPoly = vec![vec![-Int::ONE]];
    target.extend((0..n - 1).map(|_| Vec::new()));
    let mut top: Vec<Int> = vec![Int::ONE];
    top.extend(vec![Int::ZERO; d]);
    target.push(top);
    let r: Vec<Int> = dup_resultant(&in_y(&m), &target, &zx());
    choose_factor(&r, e)
}

/// The irreducible factor of `f` vanishing at `v` (`_choose_factor`).
fn choose_factor(f: &[Int], v: &Basic) -> PolyResult<Vec<Int>> {
    let (_, factors) = dup_zz_factor(f, &ZZ)?;
    let candidates: Vec<Vec<Int>> = factors.into_iter().map(|(g, _)| g).collect();
    if candidates.len() == 1 {
        return Ok(candidates.into_iter().next().expect("a factor"));
    }
    let mut precision: usize = FIRST_PRECISION;
    while precision <= LAST_PRECISION {
        let Some(value) = numeric_value(v, precision + 16) else {
            return Err(polys_error(
                PolysErrorKind::NotAlgebraic,
                format!("{} has no numerical value", v),
            ));
        };
        let mut sizes: Vec<(Real, usize)> = Vec::new();
        for (i, g) in candidates.iter().enumerate() {
            let mut s: Complex = complex_real(real_int(&Int::ZERO, precision), precision);
            for c in g {
                s = &(&s * &value) + &complex_real(real_int(c, precision), precision);
            }
            let (a, b) = parts(&s);
            sizes.push(((&a * &a + &b * &b).sqrt(), i));
        }
        sizes.sort_by(|x, y| {
            x.0.partial_cmp(&y.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.1.cmp(&y.1))
        });
        let (a, i) = (&sizes[0].0, sizes[0].1);
        let b: &Real = &sizes[1].0;
        if *b > a * &real_int(&Int::from(1_000_000u32), precision) {
            return Ok(candidates[i].clone());
        }
        precision *= 2;
    }
    Err(not_ported("multiple candidates for the minimal polynomial"))
}
