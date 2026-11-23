//! Port of `sympy_extras/solvers/special.py`: the solutions of second
//! order linear equations with rational coefficients in Bessel, Whittaker
//! and Gauss hypergeometric functions, from the normal form `z'' = r z`.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::Basic;
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::solvers::kovacic::{failure, normal_form, replaced};
use crate::solvers::linear_ode::{normal, root_multiplicities, LinearOperator, UPoly};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn sqrt(e: &Basic) -> R {
    pow(e, &Basic::half())
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn hyper(ap: &[Basic], bq: &[Basic], z: &Basic) -> R {
    crate::functions::special::hyper::hyper(ap, bq, z)
}
fn quarter() -> Basic {
    Basic::rational(Int::ONE, Int::from(4)).expect("1/4")
}
fn rat(r: Rat) -> Basic {
    Basic::from_rational(r)
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn upoly(e: &Basic, x: &Basic) -> PolyResult<Option<UPoly>> {
    match UPoly::of(e, x) {
        Ok(p) => Ok(Some(p)),
        Err(PolyError::Value(_)) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Whittaker's function `M_{kappa, mu}(t)` (`whittaker_m`).
pub fn whittaker_m(kappa: &Basic, mu: &Basic, t: &Basic) -> R {
    let half: Basic = Basic::half();
    mul(&[
        exp(&div(&neg(t)?, &int(2))?)?,
        pow(t, &add(&[mu.clone(), half.clone()])?)?,
        hyper(
            &[add(&[sub(mu, kappa)?, half])?],
            &[add(&[int(1), mul(&[int(2), mu.clone()])?])?],
            t,
        )?,
    ])
}

/// The coefficients of `(x - c)**e` in `r` when `r` is a Laurent
/// polynomial in `x - c` (`_shifted_terms`).
fn shifted_terms(r: &Basic, c: &Basic, x: &Basic) -> PolyResult<Option<BTreeMap<i64, Basic>>> {
    let s: Basic = dummy("s");
    let shifted: Basic = normal(&replaced(r, &[(x.clone(), add(&[c.clone(), s.clone()])?)])?)?;
    let (num, den) = numer_denom(&shifted)?;
    let (Some(den_poly), Some(num_poly)) = (upoly(&den, &s)?, upoly(&num, &s)?) else {
        return Ok(None);
    };
    if den_poly.terms.len() != 1 {
        return Ok(None);
    }
    let j: i64 = den_poly.degree();
    let lead: Basic = den_poly.lc();
    let mut result: BTreeMap<i64, Basic> = BTreeMap::new();
    for (e, coefficient) in num_poly.terms.iter().rev() {
        result.insert(e - j, div(coefficient, &lead)?);
    }
    Ok(Some(result))
}

/// Shifts `c` around which `r` may be a Laurent polynomial
/// (`_candidates`).
fn candidates(r: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let (num, den) = numer_denom(&normal(r)?)?;
    let (Some(den_poly), Some(num_poly)) = (upoly(&den, x)?, upoly(&num, x)?) else {
        return Ok(Vec::new());
    };
    if den_poly.degree() > 0 {
        let found = root_multiplicities(&den_poly)?;
        if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != den_poly.degree() || found.len() != 1 {
            return Ok(Vec::new());
        }
        return Ok(found.into_iter().map(|(r, _)| r).collect());
    }
    if num_poly.is_zero() {
        // r = 0: no singular point to centre a form on (sympy-extras#42)
        return Ok(Vec::new());
    }
    let m: i64 = num_poly.degree();
    if m == 0 {
        return Ok(vec![Basic::zero()]);
    }
    let coefficients: Vec<Basic> = num_poly.all_coeffs();
    let centre: Basic = div(&neg(&coefficients[1])?, &mul(&[int(m), coefficients[0].clone()])?)?;
    Ok(if m > 1 {
        vec![centre]
    } else {
        vec![centre, Basic::zero()]
    })
}

/// A normal-form equation matched to a family (`_Family`).
struct Family {
    k: Rat,
    a: Basic,
    parameters: Vec<(&'static str, Basic)>,
}

impl Family {
    fn parameter(&self, name: &str) -> Basic {
        self.parameters
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.clone())
            .expect("a parameter")
    }
}

fn match_bessel(terms: &BTreeMap<i64, Basic>) -> PolyResult<Option<Family>> {
    let others: Vec<(i64, Basic)> = terms
        .iter()
        .filter(|(e, _)| **e != -2)
        .map(|(e, v)| (*e, v.clone()))
        .collect();
    if others.len() != 1 {
        return Ok(None);
    }
    let (m, b) = others[0].clone();
    let k: Rat = Rat::from(m + 2) / Rat::from(2);
    if k == Rat::ZERO {
        return Ok(None);
    }
    let a_term: Basic = terms.get(&-2).cloned().unwrap_or_else(Basic::zero);
    let a: Basic = div(&sqrt(&neg(&b)?)?, &rat(k.clone()))?;
    let nu2: Basic = div(&add(&[a_term, quarter()])?, &pow(&rat(k.clone()), &int(2))?)?;
    Ok(Some(Family {
        k,
        a,
        parameters: vec![("nu2", nu2), ("B", b)],
    }))
}

fn match_whittaker(terms: &BTreeMap<i64, Basic>) -> PolyResult<Option<Family>> {
    let others: Vec<i64> = terms.keys().filter(|e| **e != -2).copied().collect();
    if others.len() != 2 {
        return Ok(None);
    }
    let (e1, e2) = (others[0], others[1]);
    if e2 != 2 * e1 + 2 {
        return Ok(None);
    }
    let k: Rat = Rat::from(e1 + 2);
    if k == Rat::ZERO {
        return Ok(None);
    }
    let (c, d, e) = (
        terms[&e2].clone(),
        terms[&e1].clone(),
        terms.get(&-2).cloned().unwrap_or_else(Basic::zero),
    );
    let k_expr: Basic = rat(k.clone());
    let a: Basic = div(&mul(&[int(2), sqrt(&c)?])?, &k_expr)?;
    let kappa: Basic = div(&neg(&d)?, &mul(&[a.clone(), pow(&k_expr, &int(2))?])?)?;
    let mu2: Basic = div(&add(&[e, quarter()])?, &pow(&k_expr, &int(2))?)?;
    Ok(Some(Family {
        k,
        a,
        parameters: vec![("kappa", kappa), ("mu2", mu2)],
    }))
}

fn prefactor(p: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let Some(integral) = crate::integrals::indefinite::closed_antiderivative(p, x, false, false, None)? else {
        return Ok(None);
    };
    Ok(Some(exp(&div(&neg(&integral)?, &int(2))?)?))
}

/// `sqrt(value2)` with perfect squares recognised (`_sqrt_parameter`).
fn sqrt_parameter(value2: &Basic) -> R {
    let factored: Basic = match crate::polys::polytools::factor(value2, &[], false, true) {
        Ok(f) => f,
        Err(e) if failure(&e) => value2.clone(),
        Err(e) => return Err(e),
    };
    if crate::core::assumptions::is_number(&factored) {
        return sqrt(&factored);
    }
    let mut symbols: Vec<Basic> = factored.free_symbols().into_iter().filter(|s| s.is_symbol()).collect();
    symbols.sort_by_key(|s| s.to_string());
    if crate::core::expr::polynomial_like(&factored, &symbols, crate::core::expr::PolynomialKind::Polynomial)
        == Some(true)
    {
        let (constant, factors) = match crate::polys::polytools::sqf_list(&factored, &symbols) {
            Ok(found) => found,
            Err(e) if failure(&e) => return sqrt(&factored),
            Err(e) => return Err(e),
        };
        let even = |m: &Basic| {
            m.as_rational()
                .filter(|r| r.is_int())
                .is_some_and(|r| crate::numbers::numer(r) % Int::from(2) == Int::ZERO)
        };
        if factors.iter().all(|(_, m)| even(m)) {
            let mut root: Basic = sqrt(&constant)?;
            for (f, m) in &factors {
                if let Some(r) = m.as_rational().filter(|r| r.is_int()) {
                    root = mul(&[root, pow(f, &rat(r.clone() / Rat::from(2)))?])?;
                }
            }
            return Ok(root);
        }
    }
    sqrt(&factored)
}

fn without_constant(e: &Basic, x: &Basic) -> R {
    let (constant, rest) = crate::core::expr::as_independent(e, x, false)?;
    Ok(if constant != Basic::zero() && rest != Basic::zero() {
        rest
    } else {
        e.clone()
    })
}

/// A light simplification (`_tidy`).
fn tidy(e: &Basic) -> R {
    let measure = |b: &Basic| crate::core::function::count_ops(b);
    match crate::simplify::powsimp::powsimp(e, false, crate::simplify::powsimp::Combine::Exp, false, &measure) {
        Ok(v) => Ok(v),
        Err(err) if failure(&err) => Ok(e.clone()),
        Err(err) => Err(err),
    }
}

/// Solutions of `z'' = r z` in Bessel functions, or `None`
/// (`bessel_solutions`).
pub fn bessel_solutions(r: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    use crate::functions::special::bessel::{besseli, besselj, besselk, bessely};
    for c in candidates(r, x)? {
        let Some(terms) = shifted_terms(r, &c, x)? else {
            continue;
        };
        let Some(family) = match_bessel(&terms)? else {
            continue;
        };
        let (k, a) = (rat(family.k.clone()), family.a.clone());
        let nu: Basic = sqrt_parameter(&family.parameter("nu2"))?;
        let u: Basic = sub(x, &c)?;
        let b: Basic = family.parameter("B");
        let (first, second) = if is(&b, Predicate::positive) == Some(true) {
            // a is imaginary: modified Bessel functions of sqrt(B)/k u**k
            let t: Basic = mul(&[div(&sqrt(&b)?, &k)?, pow(&u, &k)?])?;
            (besseli(&nu, &t)?, besselk(&nu, &t)?)
        } else {
            let t: Basic = mul(&[a, pow(&u, &k)?])?;
            let first: Basic = besselj(&nu, &t)?;
            let second: Basic = if is(&nu, Predicate::integer) == Some(false) {
                besselj(&neg(&nu)?, &t)?
            } else {
                bessely(&nu, &t)?
            };
            (first, second)
        };
        return Ok(Some(vec![mul(&[sqrt(&u)?, first])?, mul(&[sqrt(&u)?, second])?]));
    }
    Ok(None)
}

/// Solutions of `z'' = r z` in Whittaker functions, or `None`
/// (`whittaker_solutions`).
pub fn whittaker_solutions(r: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    for c in candidates(r, x)? {
        let Some(terms) = shifted_terms(r, &c, x)? else {
            continue;
        };
        let Some(family) = match_whittaker(&terms)? else {
            continue;
        };
        let (k, a) = (rat(family.k.clone()), family.a.clone());
        let kappa: Basic = family.parameter("kappa");
        let mu: Basic = sqrt_parameter(&family.parameter("mu2"))?;
        let u: Basic = sub(x, &c)?;
        let t: Basic = mul(&[a, pow(&u, &k)?])?;
        // w = z(t) phi'**(-1/2), phi' = a k u**(k-1): the constant is dropped
        let factor: Basic = pow(&u, &div(&sub(&int(1), &k)?, &int(2))?)?;
        let mut solutions: Vec<Basic> = vec![mul(&[factor.clone(), whittaker_m(&kappa, &mu, &t)?])?];
        if is(&mul(&[int(2), mu.clone()])?, Predicate::integer) != Some(true) {
            solutions.push(mul(&[factor, whittaker_m(&kappa, &neg(&mu)?, &t)?])?);
        }
        return Ok(Some(solutions));
    }
    Ok(None)
}

/// `phi` sending `c1, c2, c3` to `0, 1, oo`, its inverse and the symbol
/// (`_moebius`).
fn moebius(c1: &Basic, c2: &Basic, c3: Option<&Basic>, x: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    let z: Basic = dummy("z");
    let (phi, inverse) = match c3 {
        None => (
            div(&sub(x, c1)?, &sub(c2, c1)?)?,
            add(&[c1.clone(), mul(&[sub(c2, c1)?, z.clone()])?])?,
        ),
        Some(c3) => (
            div(&mul(&[sub(x, c1)?, sub(c2, c3)?])?, &mul(&[sub(x, c3)?, sub(c2, c1)?])?)?,
            div(
                &sub(
                    &mul(&[c1.clone(), sub(c2, c3)?])?,
                    &mul(&[z.clone(), c3.clone(), sub(c2, c1)?])?,
                )?,
                &sub(&sub(c2, c3)?, &mul(&[z.clone(), sub(c2, c1)?])?)?,
            )?,
        ),
    };
    Ok((phi, inverse, z))
}

/// Solutions of `z'' = r z` through Gauss's hypergeometric function, or
/// `None` (`hypergeometric_solutions`).
pub fn hypergeometric_solutions(r: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let (num, den) = numer_denom(&normal(r)?)?;
    let (Some(den_poly), Some(num_poly)) = (upoly(&den, x)?, upoly(&num, x)?) else {
        return Ok(None);
    };
    let found = root_multiplicities(&den_poly)?;
    if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != den_poly.degree() || found.iter().any(|(_, m)| *m > 2) {
        return Ok(None);
    }
    let poles: Vec<Basic> = found.into_iter().map(|(r, _)| r).collect();
    let order_at_infinity: i64 = den_poly.degree() - num_poly.degree();
    let (c1, c2, c3) = if poles.len() == 2 && order_at_infinity >= 2 {
        (poles[0].clone(), poles[1].clone(), None)
    } else if poles.len() == 3 && order_at_infinity >= 4 {
        // infinity is not singular: phi'**2 R(phi) with phi Moebius
        (poles[0].clone(), poles[1].clone(), Some(poles[2].clone()))
    } else {
        return Ok(None);
    };
    let (phi, inverse, z) = moebius(&c1, &c2, c3.as_ref(), x)?;
    let dphi: Basic = crate::core::function::diff(&phi, &[(x.clone(), 1)])?;
    let big_r: Basic = normal(&replaced(&div(r, &pow(&dphi, &int(2))?)?, &[(x.clone(), inverse)])?)?;
    let z_minus_1: Basic = sub(&z, &int(1))?;
    let a0: Basic = replaced(
        &normal(&mul(&[big_r.clone(), pow(&z, &int(2))?])?)?,
        &[(z.clone(), Basic::zero())],
    )?;
    let a1: Basic = replaced(
        &normal(&mul(&[big_r.clone(), pow(&z_minus_1, &int(2))?])?)?,
        &[(z.clone(), Basic::one())],
    )?;
    let lam2: Basic = add(&[int(1), mul(&[int(4), a0.clone()])?])?;
    let mu2: Basic = add(&[int(1), mul(&[int(4), a1.clone()])?])?;
    let remainder: Basic = normal(&sub(
        &sub(&big_r, &div(&a0, &pow(&z, &int(2))?)?)?,
        &div(&a1, &pow(&z_minus_1, &int(2))?)?,
    )?)?;
    let k: Basic = normal(&mul(&[remainder, z.clone(), z_minus_1])?)?;
    if k.has(&z) {
        return Ok(None);
    }
    let nu2: Basic = add(&[lam2.clone(), mu2.clone(), int(-1), mul(&[int(4), k])?])?;
    let (lam, mu, nu) = (sqrt_parameter(&lam2)?, sqrt_parameter(&mu2)?, sqrt_parameter(&nu2)?);
    let half = |e: Basic| div(&e, &int(2));
    let alpha: Basic = half(add(&[int(1), lam.clone()])?)?;
    let alpha_: Basic = half(sub(&int(1), &lam)?)?;
    let beta: Basic = half(add(&[int(1), mu.clone()])?)?;
    let a_1: Basic = half(add(&[int(1), lam.clone(), mu.clone(), nu.clone()])?)?;
    let b_1: Basic = half(sub(&add(&[int(1), lam.clone(), mu.clone()])?, &nu)?)?;
    let c_: Basic = add(&[int(1), lam.clone()])?;
    let one_minus_phi: Basic = sub(&int(1), &phi)?;
    let w: Basic = mul(&[
        pow(&phi, &alpha)?,
        pow(&one_minus_phi, &beta)?,
        hyper(&[a_1, b_1], &[c_], &phi)?,
    ])?;
    let root: Basic = sqrt(&dphi)?;
    let mut solutions: Vec<Basic> = vec![without_constant(&div(&w, &root)?, x)?];
    if is(&lam, Predicate::integer) != Some(true) {
        let a_2: Basic = half(add(&[sub(&int(1), &lam)?, mu.clone(), nu.clone()])?)?;
        let b_2: Basic = half(sub(&add(&[sub(&int(1), &lam)?, mu.clone()])?, &nu)?)?;
        let w2: Basic = mul(&[
            pow(&phi, &alpha_)?,
            pow(&one_minus_phi, &beta)?,
            hyper(&[a_2, b_2], &[sub(&int(1), &lam)?], &phi)?,
        ])?;
        solutions.push(without_constant(&div(&w2, &root)?, x)?);
    }
    Ok(Some(solutions))
}

/// Solutions of a second order linear equation with rational coefficients
/// in Bessel, Whittaker or hypergeometric functions, or `None`
/// (`special_solutions`).
pub fn special_solutions(equation: &Basic, f: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let l: LinearOperator = LinearOperator::from_equation(equation, f)?;
    let x: Basic = match f.args().first() {
        Some(x) if x.is_symbol() && l.order() == 2 => x.clone(),
        _ => return Ok(None),
    };
    let (a2, a1, a0) = (&l.coefficients[2], &l.coefficients[1], &l.coefficients[0]);
    let p: Basic = normal(&div(a1, a2)?)?;
    let q: Basic = normal(&div(a0, a2)?)?;
    let r: Basic = normal_form(&p, &q, &x)?;
    let Some(factor) = prefactor(&p, &x)? else {
        return Ok(None);
    };
    let methods: [fn(&Basic, &Basic) -> PolyResult<Option<Vec<Basic>>>; 3] =
        [bessel_solutions, whittaker_solutions, hypergeometric_solutions];
    for method in methods {
        if let Some(found) = method(&r, &x)?.filter(|v| !v.is_empty()) {
            return Ok(Some(
                found
                    .iter()
                    .map(|z| without_constant(&tidy(&mul(&[factor.clone(), z.clone()])?)?, &x))
                    .collect::<PolyResult<_>>()?,
            ));
        }
    }
    Ok(None)
}
