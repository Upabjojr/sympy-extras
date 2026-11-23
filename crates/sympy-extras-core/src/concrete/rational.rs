//! Port of `sympy_extras/concrete/rational.py`: Abramov's decomposition of
//! a rational function `f = g(k + 1) - g(k) + h` with `h` of minimal
//! denominator (the shift classes of the irreducible factors of the
//! denominator), and the definite sums of rational functions: the
//! rational part in closed form, the remainder summed by the summation of
//! the core (polygamma functions) or left as a `Sum`.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::error::{PolyError, PolyResult};
use crate::polys::polytools::Poly;

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
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_method(e)
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn poly(e: &Basic, k: &Basic) -> PolyResult<Poly> {
    Poly::new(e, std::slice::from_ref(k))
}
fn shifted(e: &Basic, k: &Basic, by: i64) -> R {
    subs(e, k, &add(&[k.clone(), int(by)])?)
}

/// `f = g(k + 1) - g(k) + h` with `h` of minimal denominator, immutable
/// (`RationalDecomposition`).
#[derive(Clone, Debug, PartialEq)]
pub struct RationalDecomposition {
    pub g: Basic,
    pub h: Basic,
}

impl RationalDecomposition {
    /// Whether `f` has a rational sum (`summable`).
    pub fn summable(&self) -> bool {
        self.h == Basic::zero()
    }
}

/// `j` with `q2(k) == q1(k + j)` for monic polynomials of the same degree
/// (`_shift`).
fn shift_of(q1: &Poly, q2: &Poly, k: &Basic) -> PolyResult<Option<i64>> {
    let d: i64 = q1.degree(0)?;
    if d != q2.degree(0)? || d == 0 {
        return Ok(None);
    }
    let (c1, c2) = (q1.all_coeffs()?[1].clone(), q2.all_coeffs()?[1].clone());
    // the coefficient of k**(d-1) of q1(k + j) is c1 + d*j
    let j: Basic = cancel(&mul(&[sub(&c2, &c1)?, pow(&int(d), &int(-1))?])?)?;
    let Some(j) = integer_of(&j) else {
        return Ok(None);
    };
    if expand(&sub(&shifted(&q1.as_expr()?, k, j)?, &q2.as_expr()?)?)? == Basic::zero() {
        return Ok(Some(j));
    }
    Ok(None)
}

fn not_rational(e: &Basic, k: &Basic) -> PolyError {
    PolyError::Value(format!("{} is not a rational function of {}", e, k))
}

/// Abramov's decomposition of a rational function of `k`
/// (`abramov_decomposition`).
pub fn abramov_decomposition(f: &Basic, k: &Basic) -> PolyResult<RationalDecomposition> {
    let expression: Basic = cancel(f)?;
    if crate::core::expr::polynomial_like(
        &expression,
        std::slice::from_ref(k),
        crate::core::expr::PolynomialKind::Rational,
    ) != Some(true)
    {
        return Err(not_rational(&expression, k));
    }
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&expression)?;
    let (num, den) = match (poly(&numerator, k), poly(&denominator, k)) {
        (Ok(n), Ok(d)) => (n, d),
        (Err(PolyError::Polys(..)), _) | (_, Err(PolyError::Polys(..))) => return Err(not_rational(&expression, k)),
        (Err(e), _) | (_, Err(e)) => return Err(e),
    };
    let (quotient, remainder) = num.div(&den)?;
    let mut g: Basic = Basic::zero();
    // the polynomial part: sum_{i=0}^{k-1} quotient(i)
    if !quotient.is_zero() {
        let i: Basic = crate::core::symbol::Dummy::new("i", &[(Predicate::integer, true)]).0;
        let summand: Basic = subs(&quotient.as_expr()?, k, &i)?;
        let summed: Basic =
            crate::concrete::summations::summation(&summand, &(i, Basic::zero(), add(&[k.clone(), int(-1)])?))?;
        if matches!(summed.head(), Head::Sum) {
            return Err(PolyError::Value("the polynomial part could not be summed".to_string()));
        }
        g = add(&[g, summed])?;
    }
    if remainder.is_zero() {
        return Ok(RationalDecomposition {
            g: cancel(&g)?,
            h: Basic::zero(),
        });
    }
    // the shift classes of the irreducible factors of the denominator
    let (_, factors) = crate::polys::polytools::factor_list(&den.as_expr()?, std::slice::from_ref(k))?;
    let mut classes: Vec<Vec<(Poly, i64)>> = Vec::new();
    for (factor, _) in factors {
        let factor_poly: Poly = poly(&factor, k)?.monic()?;
        let mut placed: bool = false;
        for group in classes.iter_mut() {
            if let Some(j) = shift_of(&group[0].0, &factor_poly, k)? {
                group.push((factor_poly.clone(), j));
                placed = true;
                break;
            }
        }
        if !placed {
            classes.push(vec![(factor_poly, 0)]);
        }
    }
    // partial fractions of the proper part
    let proper: Basic = mul(&[remainder.as_expr()?, pow(&den.as_expr()?, &int(-1))?])?;
    let mut h_terms: Vec<Basic> = Vec::new();
    for (c, q, m) in partial_terms(&proper, k)? {
        let q_poly: Poly = poly(&q, k)?.monic()?;
        let scale: Basic = poly(&q, k)?.LC()?;
        let c_: Basic = mul(&[c, pow(&scale, &int(-m))?])?;
        let q_expr: Basic = q_poly.as_expr()?;
        // the class and the shift of this factor
        let mut found: Option<(usize, i64)> = None;
        'search: for (gi, group) in classes.iter().enumerate() {
            for (member, j) in group {
                if expand(&sub(&member.as_expr()?, &q_expr)?)? == Basic::zero() {
                    found = Some((gi, *j));
                    break 'search;
                }
            }
        }
        let Some((gi, shift)) = found else {
            h_terms.push(mul(&[c_, pow(&q_expr, &int(-m))?])?);
            continue;
        };
        let group: &Vec<(Poly, i64)> = &classes[gi];
        let representative: &Poly = &group[0].0;
        // the leftmost member of the class is the target denominator
        let least: i64 = group.iter().map(|(_, j)| *j).min().unwrap_or(0);
        let offset: i64 = shift - least;
        let target: Basic = shifted(&representative.as_expr()?, k, least)?;
        if offset == 0 {
            h_terms.push(mul(&[c_, pow(&target, &int(-m))?])?);
            continue;
        }
        // c(k)/q(k + offset)^m = c(k - offset)/q(k)^m
        //     + Delta(sum_{i=0}^{offset-1} c(k - offset + i)/q(k + i)^m)
        h_terms.push(mul(&[shifted(&c_, k, -offset)?, pow(&target, &int(-m))?])?);
        let mut parts: Vec<Basic> = vec![g];
        for i in 0..offset {
            parts.push(mul(&[
                shifted(&c_, k, i - offset)?,
                pow(&shifted(&target, k, i)?, &int(-m))?,
            ])?);
        }
        g = add(&parts)?;
    }
    let h: Basic = if h_terms.is_empty() {
        Basic::zero()
    } else {
        cancel(&add(&h_terms)?)?
    };
    Ok(RationalDecomposition { g: cancel(&g)?, h })
}

/// `(c(k), q(k), m)` for the terms `c/q**m` of the partial fraction
/// expansion of a proper rational function (`_partial_terms`).
fn partial_terms(proper: &Basic, k: &Basic) -> PolyResult<Vec<(Basic, Basic, i64)>> {
    let mut result: Vec<(Basic, Basic, i64)> = Vec::new();
    let expanded: Basic = crate::polys::partfrac::apart(proper, Some(k), false)?;
    let terms: Vec<Basic> = if expanded.is_add() {
        expanded.args().to_vec()
    } else {
        vec![expanded]
    };
    for term in terms {
        let mut c: Basic = int(1);
        let mut q: Option<Basic> = None;
        let mut m: i64 = 1;
        let factors: Vec<Basic> = if term.is_mul() {
            term.args().to_vec()
        } else {
            vec![term]
        };
        for factor in factors {
            let negative_power: Option<i64> = if factor.is_pow() && factor.args()[0].has(k) {
                integer_of(&factor.args()[1]).filter(|e| *e < 0)
            } else {
                None
            };
            if let Some(e) = negative_power {
                q = Some(factor.args()[0].clone());
                m = -e;
            } else if factor.has(k)
                && crate::core::expr::polynomial_like(
                    &factor,
                    std::slice::from_ref(k),
                    crate::core::expr::PolynomialKind::Polynomial,
                ) != Some(true)
            {
                let (inner_numerator, inner_denominator) = crate::core::expr::as_numer_denom(&factor)?;
                q = Some(inner_denominator);
                c = mul(&[c, inner_numerator])?;
            } else {
                c = mul(&[c, factor])?;
            }
        }
        if let Some(q) = q {
            result.push((c, q, m));
        }
    }
    Ok(result)
}

/// `sum_{k=lo}^{hi} f` for a rational function (`rational_sum`).
pub fn rational_sum(f: &Basic, k: &Basic, lo: &Basic, hi: &Basic, special_values: bool) -> R {
    if !k.is_symbol() {
        return Err(PolyError::Type(format!(
            "the index of summation must be a symbol, not {}",
            k
        )));
    }
    let decomposition: RationalDecomposition = abramov_decomposition(f, k)?;
    let (g, h) = (&decomposition.g, &decomposition.h);
    let mut total: Basic = sub(&subs(g, k, &add(&[hi.clone(), int(1)])?)?, &subs(g, k, lo)?)?;
    if *h != Basic::zero() {
        total = add(&[
            total,
            crate::concrete::summations::summation(h, &(k.clone(), lo.clone(), hi.clone()))?,
        ])?;
    }
    let symbols: Vec<Basic> = total.free_symbols().into_iter().collect();
    let value: Basic =
        if crate::core::expr::polynomial_like(&total, &symbols, crate::core::expr::PolynomialKind::Rational)
            == Some(true)
        {
            crate::polys::polytools::factor(&cancel(&total)?, &[], false, true)?
        } else {
            total
        };
    if !special_values {
        return Ok(value);
    }
    let at_point = |point: &crate::concrete::karr::SumPoint| -> PolyResult<Option<Basic>> {
        let rule: HashMap<Basic, Basic> = point.iter().cloned().collect();
        Ok(Some(rational_sum(&f.xreplace(&rule), k, lo, hi, true)?))
    };
    crate::concrete::karr::with_special_values_sum(
        &value,
        f,
        std::slice::from_ref(k),
        &[lo.clone(), hi.clone()],
        &at_point,
        true,
        None,
    )
}
