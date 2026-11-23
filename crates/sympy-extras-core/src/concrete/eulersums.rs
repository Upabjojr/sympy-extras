//! Port of `sympy_extras/concrete/eulersums.py`: series of polygamma
//! functions and harmonic numbers summed through the Euler sums `S(p, q) =
//! Sum(harmonic(n, p)/n**q, (n, 1, oo))` in zeta values (Euler's formula
//! for `p = 1`, the symmetry for `p = q`, Flajolet–Salvy for an odd weight,
//! the known sums of weight six), and the integral representation of
//! `Sum(polygamma(0, n)/n**s)` for a symbolic `s`.

use crate::core::basic::{Basic, Function};
use crate::error::PolyResult;

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rational(p: i64, q: i64) -> R {
    Basic::rational(p.into(), q.into())
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_method(e)
}
fn zeta_of(s: &Basic) -> R {
    crate::functions::special::zeta_functions::zeta(s, None)
}
fn zeta_k(k: i64) -> R {
    zeta_of(&int(k))
}
fn binomial(n: i64, k: i64) -> R {
    crate::functions::combinatorial::factorials::binomial(&int(n), &int(k))
}
fn harmonic(n: i64, m: Option<i64>) -> R {
    match m {
        None => crate::functions::combinatorial::numbers::harmonic(&int(n), None),
        Some(m) => crate::functions::combinatorial::numbers::harmonic(&int(n), Some(&int(m))),
    }
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn factors_of(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// The Euler sum `S(p, q)` in zeta values, or `None` when no evaluation is
/// known (`euler_sum`).
pub fn euler_sum(p: i64, q: i64) -> R2 {
    if p < 1 || q < 2 {
        return Ok(None);
    }
    if p == 1 {
        let mut total: Basic = mul(&[add(&[int(1), rational(q, 2)?])?, zeta_k(q + 1)?])?;
        let mut products: Vec<Basic> = Vec::new();
        for k in 1..q - 1 {
            products.push(mul(&[zeta_k(k + 1)?, zeta_k(q - k)?])?);
        }
        total = sub(&total, &mul(&[rational(1, 2)?, add(&products)?])?)?;
        return Ok(Some(expand(&total)?));
    }
    if p == q {
        let total: Basic = div(&add(&[pow(&zeta_k(p)?, &int(2))?, zeta_k(2 * p)?])?, &int(2))?;
        return Ok(Some(expand(&total)?));
    }
    let w: i64 = p + q;
    if w % 2 == 1 {
        return Ok(Some(odd_weight(p, q)?));
    }
    if q == 1 {
        return Ok(None);
    }
    let s24 = || -> R { sub(&pow(&zeta_k(3)?, &int(2))?, &div(&zeta_k(6)?, &int(3))?) };
    if (p, q) == (2, 4) {
        return Ok(Some(expand(&s24()?)?));
    }
    if (p, q) == (4, 2) {
        // from the symmetry with S(2, 4)
        let total: Basic = sub(&add(&[mul(&[zeta_k(2)?, zeta_k(4)?])?, zeta_k(6)?])?, &s24()?)?;
        return Ok(Some(expand(&total)?));
    }
    Ok(None)
}

/// Flajolet–Salvy, Theorem 3.1, for an odd weight `p + q`
/// (`_odd_weight`).
fn odd_weight(p: i64, q: i64) -> R {
    let w: i64 = p + q;
    let sign: i64 = if p % 2 == 0 { 1 } else { -1 };
    let mut total: Basic = mul(&[
        zeta_k(w)?,
        sub(
            &sub(&rational(1, 2)?, &mul(&[rational(sign, 2)?, binomial(w - 1, p)?])?)?,
            &mul(&[rational(sign, 2)?, binomial(w - 1, q)?])?,
        )?,
    ])?;
    if p % 2 == 1 && p > 1 {
        total = add(&[total, mul(&[zeta_k(p)?, zeta_k(q)?])?])?;
    }
    for k in 1..=q / 2 {
        total = add(&[
            total,
            mul(&[
                int(sign),
                binomial(w - 2 * k - 1, p - 1)?,
                zeta_k(2 * k)?,
                zeta_k(w - 2 * k)?,
            ])?,
        ])?;
    }
    for k in 1..=p / 2 {
        total = add(&[
            total,
            mul(&[
                int(sign),
                binomial(w - 2 * k - 1, q - 1)?,
                zeta_k(2 * k)?,
                zeta_k(w - 2 * k)?,
            ])?,
        ])?;
    }
    expand(&total)
}

/// `(c, f, q)` with `term == c*f*n**(-q)`, `c` free of `n` and `f` a
/// single harmonic number or polygamma factor (`_split`).
fn split(term: &Basic, n: &Basic) -> PolyResult<Option<(Basic, Basic, i64)>> {
    let mut constant: Vec<Basic> = Vec::new();
    let mut special: Vec<Basic> = Vec::new();
    let mut q: i64 = 0;
    for f in factors_of(term) {
        if !f.has(n) {
            constant.push(f);
        } else if f.is_pow() && f.args()[0] == *n && integer_of(&f.args()[1]).is_some() {
            q -= integer_of(&f.args()[1]).unwrap_or(0);
        } else if f == *n {
            q -= 1;
        } else {
            special.push(f);
        }
    }
    if special.len() != 1 {
        return Ok(None);
    }
    Ok(Some((mul(&constant)?, special.remove(0), q)))
}

/// `(p, shift)` with `f == harmonic(n + shift, p)`, a polygamma
/// `polygamma(m, n + a)` giving `(m + 1, a - 1)` (`_harmonic_data`).
fn harmonic_data(f: &Basic, n: &Basic) -> PolyResult<Option<(i64, i64)>> {
    if f.is_function(Function::Harmonic) {
        let argument: &Basic = &f.args()[0];
        let p: Option<i64> = if f.args().len() > 1 {
            integer_of(&f.args()[1])
        } else {
            Some(1)
        };
        let shift: Option<i64> = integer_of(&sub(argument, n)?);
        return Ok(match (p, shift) {
            (Some(p), Some(shift)) => Some((p, shift)),
            _ => None,
        });
    }
    if f.is_function(Function::PolyGamma) {
        let m: Option<i64> = integer_of(&f.args()[0]);
        let shift: Option<i64> = integer_of(&sub(&f.args()[1], n)?);
        return Ok(match (m, shift) {
            (Some(m), Some(shift)) if m >= 0 => Some((m + 1, shift - 1)),
            _ => None,
        });
    }
    Ok(None)
}

/// `f` as `scale*harmonic(n + shift, p) + constant` (`_to_harmonic`).
fn to_harmonic(f: &Basic, n: &Basic) -> PolyResult<Option<(Basic, i64, i64, Basic)>> {
    let Some((p, shift)) = harmonic_data(f, n)? else {
        return Ok(None);
    };
    if f.is_function(Function::PolyGamma) {
        let m: i64 = p - 1;
        if m == 0 {
            // psi(n + a) = H_{n + a - 1} - gamma
            return Ok(Some((int(1), shift, 1, crate::core::mul::neg(&Basic::euler_gamma())?)));
        }
        let scale: Basic = mul(&[
            pow(&int(-1), &int(m))?,
            crate::functions::combinatorial::factorials::factorial(&int(m))?,
        ])?;
        let constant: Basic = crate::core::mul::neg(&mul(&[scale.clone(), zeta_k(p)?])?)?;
        return Ok(Some((scale, shift, p, constant)));
    }
    Ok(Some((int(1), shift, p, Basic::zero())))
}

/// `Sum(expr, (k, lower, oo))` for a rational function with poles at
/// integers through partial fractions (`_rational_tail`).
fn rational_tail(expr: &Basic, k: &Basic, lower: i64) -> R2 {
    let decomposed: Basic = crate::polys::partfrac::apart(expr, Some(k), false)?;
    let terms: Vec<Basic> = if decomposed.is_add() {
        decomposed.args().to_vec()
    } else {
        vec![decomposed]
    };
    let mut total: Basic = Basic::zero();
    let mut simple: Basic = Basic::zero();
    for term in terms {
        let (c, rest) = crate::core::expr::as_independent(&term, k, false)?;
        if !rest.has(k) {
            return Ok(None);
        }
        let (base, m): (Basic, i64) = match integer_of(if rest.is_pow() { &rest.args()[1] } else { &rest }) {
            Some(e) if rest.is_pow() && e < 0 => (rest.args()[0].clone(), -e),
            _ => {
                let inverse: Basic = div(&int(1), &rest)?;
                if crate::core::expr::polynomial_like(
                    &inverse,
                    std::slice::from_ref(k),
                    crate::core::expr::PolynomialKind::Polynomial,
                ) == Some(true)
                {
                    (inverse, 1)
                } else {
                    return Ok(None);
                }
            }
        };
        let Some(a) = integer_of(&sub(&base, k)?) else {
            return Ok(None);
        };
        if lower + a < 1 {
            return Ok(None);
        }
        if m == 1 {
            simple = add(&[simple, c.clone()])?;
            total = sub(&total, &mul(&[c, harmonic(lower + a - 1, None)?])?)?;
        } else {
            total = add(&[total, mul(&[c, sub(&zeta_k(m)?, &harmonic(lower + a - 1, Some(m))?)?])?])?;
        }
    }
    if simple != Basic::zero() {
        return Ok(None);
    }
    Ok(Some(expand(&total)?))
}

/// `Sum(term, (n, lower, oo))` for a summand `c*f(n)/n**q` with `f` a
/// harmonic number or a polygamma function, in zeta values
/// (`polygamma_series`).
pub fn polygamma_series(term: &Basic, n: &Basic, lower: &Basic) -> R2 {
    let Some(l) = integer_of(lower).filter(|l| *l >= 1) else {
        return Ok(None);
    };
    let Some((c, f, q)) = split(term, n)? else {
        return Ok(None);
    };
    let converted = to_harmonic(&f, n)?;
    let Some((scale, shift, p, constant)) = converted.filter(|_| q >= 2) else {
        return Ok(None);
    };
    if l + shift < 0 {
        // a harmonic number with a negative index (a pole of polygamma)
        return Ok(None);
    }
    let Some(base) = euler_sum(p, q)? else {
        return Ok(None);
    };
    let k: Basic = crate::core::symbol::Dummy::new("k", &[]).0;
    let mut head: Vec<Basic> = Vec::new();
    for j in 1..l {
        head.push(div(&harmonic(j, Some(p))?, &pow(&int(j), &int(q))?)?);
    }
    let mut total: Basic = sub(&base, &add(&head)?)?;
    let offsets: Vec<i64> = if shift > 0 {
        (1..=shift).collect()
    } else {
        (0..-shift).map(|j| -j).collect()
    };
    for offset in offsets {
        let summand: Basic = div(
            &int(1),
            &mul(&[pow(&add(&[k.clone(), int(offset)])?, &int(p))?, pow(&k, &int(q))?])?,
        )?;
        let Some(extra) = rational_tail(&summand, &k, l)? else {
            return Ok(None);
        };
        total = if shift > 0 {
            add(&[total, extra])?
        } else {
            sub(&total, &extra)?
        };
    }
    let mut powers: Vec<Basic> = Vec::new();
    for j in 1..l {
        powers.push(pow(&int(j), &int(-q))?);
    }
    total = add(&[
        mul(&[scale, total])?,
        mul(&[constant, sub(&zeta_k(q)?, &add(&powers)?)?])?,
    ])?;
    Ok(Some(expand(&mul(&[c, total])?)?))
}

/// `Sum(polygamma(0, n)/n**s, (n, 1, oo))` through the integral
/// representation of `psi`, for a symbolic `s`
/// (`polygamma_integral_representation`).
pub fn polygamma_integral_representation(term: &Basic, n: &Basic) -> R2 {
    let mut exponent: Option<Basic> = None;
    let mut found: bool = false;
    let mut constant: Vec<Basic> = Vec::new();
    for f in factors_of(term) {
        if f.is_function(Function::PolyGamma) && f.args()[0] == Basic::zero() && f.args()[1] == *n {
            found = true;
        } else if f.is_pow() && f.args()[0] == *n && !f.args()[1].has(n) {
            exponent = Some(crate::core::mul::neg(&f.args()[1])?);
        } else if !f.has(n) {
            constant.push(f);
        } else {
            return Ok(None);
        }
    }
    let Some(s) = exponent.filter(|_| found) else {
        return Ok(None);
    };
    let t: Basic = crate::core::symbol::Symbol::new("t", &[]).0;
    let integrand: Basic = div(
        &sub(
            &zeta_of(&s)?,
            &div(&crate::functions::special::zeta_functions::polylog(&s, &t)?, &t)?,
        )?,
        &sub(&int(1), &t)?,
    )?;
    let integral: Basic = crate::integrals::integral::integral(&integrand, &t, Some((&Basic::zero(), &Basic::one())))?;
    let gamma_term: Basic = crate::core::mul::neg(&mul(&[Basic::euler_gamma(), zeta_of(&s)?])?)?;
    Ok(Some(mul(&[mul(&constant)?, add(&[gamma_term, integral])?])?))
}
