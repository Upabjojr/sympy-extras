//! Port of `sympy_extras/integrals/deltafunctions.py`: antiderivatives of
//! expressions with Dirac delta functions. The integral of
//! `f(x)*DiracDelta(x - a)` is `f(a)*Heaviside(x - a)`; a delta of a
//! polynomial argument is first written by its simple roots, and the
//! derivatives of the delta are integrated by parts. The terms without a
//! delta go to the verified methods of [`crate::integrals::indefinite`].

#![allow(clippy::mutable_key_type)]

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;

fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn p2(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn is_delta(e: &Basic) -> bool {
    e.is_function(Function::DiracDelta)
}
fn has_delta(e: &Basic) -> bool {
    e.preorder().iter().any(is_delta)
}
fn dirac_delta(x: &Basic, k: Option<&Basic>) -> R {
    crate::functions::special::delta_functions::dirac_delta(x, k)
}
fn heaviside(x: &Basic) -> R {
    crate::functions::special::delta_functions::heaviside(x, None)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn is_polynomial_error(e: &PolyError) -> bool {
    matches!(e, PolyError::Polys(..) | PolyError::HeuristicGCDFailed(_))
}

/// `e.as_poly(x)`: `None` when it is not a polynomial in `x`.
fn as_poly(e: &Basic, x: &Basic) -> PolyResult<Option<crate::polys::polytools::Poly>> {
    use crate::error::PolysErrorKind;
    match crate::polys::polytools::Poly::new(e, std::slice::from_ref(x)) {
        Ok(p) => Ok(Some(p)),
        Err(PolyError::Polys(PolysErrorKind::PolynomialError | PolysErrorKind::GeneratorsNeeded, _)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn order(d: &Basic) -> Basic {
    d.args().get(1).cloned().unwrap_or_else(Basic::zero)
}

/// Whether the argument of `d` is linear in `x` (`is_simple`).
pub fn is_simple(d: &Basic, x: &Basic) -> PolyResult<bool> {
    Ok(match as_poly(&d.args()[0], x)? {
        Some(p) => p.degree(0)? == 1,
        None => false,
    })
}

/// `d` written by the simple real roots of its argument,
/// `sum(DiracDelta(x - r)/|g'(r)|)`; `d` itself when its argument has no
/// `x`, a complex or multiple root, or `d` is a derivative
/// (`expand_diracdelta`).
pub fn expand_diracdelta(d: &Basic, x: &Basic) -> R {
    let g: &Basic = &d.args()[0];
    if !g.has(x) || order(d) != Basic::zero() {
        return Ok(d.clone());
    }
    let found = match crate::polys::polyroots::roots(g, std::slice::from_ref(x), true, true, None) {
        Ok(v) => v,
        Err(e) if is_polynomial_error(&e) || matches!(e, PolyError::Value(_) | PolyError::Type(_)) => {
            return Ok(d.clone())
        }
        Err(e) => return Err(e),
    };
    if found.is_empty() {
        return Ok(d.clone());
    }
    let derivative: Basic =
        crate::functions::elementary::complexes::abs_(&crate::core::function::diff(g, &[(x.clone(), 1)])?)?;
    let mut result: Basic = Basic::zero();
    for (r, m) in found.iter() {
        if is(r, Predicate::real) == Some(false) || *m != 1 {
            return Ok(d.clone());
        }
        let term: Basic = d2(
            &dirac_delta(&s2(x, r)?, None)?,
            &crate::core::subs::subs(&derivative, x, r)?,
        )?;
        result = a2(&result, &term)?;
    }
    Ok(result)
}

/// The factors of the product `node` with a simple delta in front:
/// `(delta, rest)`; `(None, expanded)` when there is no simple delta but
/// the deltas could be expanded, `(None, None)` when nothing changes
/// (`change_mul`).
pub fn change_mul(node: &Basic, x: &Basic) -> PolyResult<(Option<Basic>, Option<Basic>)> {
    let mut new_args: Vec<Basic> = Vec::new();
    let mut dirac: Option<Basic> = None;
    let mut sorted_args: Vec<Basic> = mul_args(node);
    sorted_args.sort_by_cached_key(crate::core::sorting::sort_key);
    for arg in &sorted_args {
        let factor: Basic = if arg.is_pow() && is_delta(&arg.args()[0]) {
            new_args.push(p2(&arg.args()[0], &s2(&arg.args()[1], &Basic::one())?)?);
            arg.args()[0].clone()
        } else {
            arg.clone()
        };
        if dirac.is_none() && is_delta(&factor) && is_simple(&factor, x)? {
            dirac = Some(factor);
        } else {
            new_args.push(factor);
        }
    }
    let Some(dirac) = dirac else {
        let mut expanded_args: Vec<Basic> = Vec::new();
        for arg in &sorted_args {
            if is_delta(arg) {
                expanded_args.push(expand_diracdelta(arg, x)?);
            } else if arg.is_pow() && is_delta(&arg.args()[0]) {
                expanded_args.push(p2(&expand_diracdelta(&arg.args()[0], x)?, &arg.args()[1])?);
            } else {
                expanded_args.push(arg.clone());
            }
        }
        if expanded_args != sorted_args {
            return Ok((None, Some(expand(&crate::core::mul::mul(&expanded_args)?)?)));
        }
        return Ok((None, None));
    };
    Ok((Some(dirac), Some(crate::core::mul::mul(&new_args)?)))
}

/// An antiderivative of `g`: its terms with a delta by [`deltaintegrate`],
/// the others by the verified methods (`_integrate`).
fn integrate(g: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let mut parts: Vec<Basic> = Vec::new();
    for term in crate::core::expr::terms_of(g) {
        let (coeff, rest) = crate::core::expr::as_independent(&term, x, false)?;
        if !rest.has(x) {
            parts.push(m2(&term, x)?);
            continue;
        }
        let found: Option<Basic> = if has_delta(&rest) {
            deltaintegrate(&rest, x)?
        } else {
            crate::integrals::indefinite::verified_antiderivative(&rest, x, &[], None)?.map(|v| v.0)
        };
        let Some(found) = found else {
            return Ok(None);
        };
        parts.push(m2(&coeff, &found)?);
    }
    Ok(Some(crate::core::add::add(&parts)?))
}

fn solve_linear(e: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let Some(p) = as_poly(e, x)? else {
        return Ok(None);
    };
    if p.degree(0)? != 1 {
        return Ok(None);
    }
    let c: Vec<Basic> = p.all_coeffs()?;
    Ok(Some(d2(&crate::core::mul::neg(&c[1])?, &c[0])?))
}

/// An antiderivative of `f`, which has delta functions; `None` when `f`
/// has no delta or a delta which cannot be simplified (`deltaintegrate`).
pub fn deltaintegrate(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    if !has_delta(f) {
        return Ok(None);
    }
    if is_delta(f) {
        let h: Basic = expand_diracdelta(f, x)?;
        if h == *f {
            if is_simple(f, x)? {
                if order(f) == Basic::zero() {
                    return Ok(Some(heaviside(&f.args()[0])?));
                }
                let Some(p) = as_poly(&f.args()[0], x)? else {
                    return Ok(None);
                };
                let lower: Basic = s2(&order(f), &Basic::one())?;
                return Ok(Some(d2(&dirac_delta(&f.args()[0], Some(&lower))?, &p.LC()?)?));
            }
            return Ok(None);
        }
        return integrate(&h, x);
    }
    if f.is_mul() || f.is_pow() {
        let g: Basic = expand(f)?;
        if *f != g {
            return integrate(&g, x);
        }
        let (deltaterm, rest_mult) = change_mul(f, x)?;
        let Some(mut deltaterm) = deltaterm else {
            if let Some(r) = rest_mult {
                return integrate(&r, x);
            }
            return Ok(None);
        };
        let mut rest: Basic = rest_mult.unwrap_or_else(Basic::one);
        let expanded: Basic = expand_diracdelta(&deltaterm, x)?;
        if expanded.is_mul() {
            // factors taken out by the expansion
            let (inner, rest_2) = change_mul(&expanded, x)?;
            let (Some(inner), Some(rest_2)) = (inner, rest_2) else {
                return Ok(None);
            };
            deltaterm = inner;
            rest = m2(&rest, &rest_2)?;
        } else if is_delta(&expanded) {
            deltaterm = expanded;
        } else {
            return Ok(None);
        }
        let Some(point) = solve_linear(&deltaterm.args()[0], x)? else {
            return Ok(None);
        };
        let ord: Basic = order(&deltaterm);
        let Some(q) = ord.as_rational().filter(|q| q.is_int()) else {
            return Ok(None);
        };
        let Ok(mut n) = i64::try_from(crate::numbers::numer(q)) else {
            return Ok(None);
        };
        // the largest hyperreal term left by the integrations by parts
        let mut m: i64 = 0;
        while n >= 0 {
            let r: Basic = if n != 0 {
                let derivative: Basic = crate::core::function::diff(&rest, &[(x.clone(), n as u64)])?;
                m2(
                    &p2(&Basic::int(-1), &Basic::int(n))?,
                    &crate::core::subs::subs(&derivative, x, &point)?,
                )?
            } else {
                crate::core::subs::subs(&rest, x, &point)?
            };
            if is(&r, Predicate::zero) == Some(true) {
                n -= 1;
                m += 1;
            } else {
                if m == 0 {
                    return Ok(Some(m2(&r, &heaviside(&s2(x, &point)?)?)?));
                }
                return Ok(Some(m2(&r, &dirac_delta(&s2(x, &point)?, Some(&Basic::int(m - 1)))?)?));
            }
        }
        // every term vanished: the derivative of the delta integrates to 0
        return Ok(Some(Basic::zero()));
    }
    Ok(None)
}
