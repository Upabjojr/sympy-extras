//! Port of `sympy_extras/solvers/abel.py`: Abel equations of the first
//! kind `y' = f3 y**3 + f2 y**2 + f1 y + f0`: their invariants, the
//! equivalence with representatives of integrable classes, the AIR class
//! through its inverse Riccati equation, and the numerical check that an
//! implicit relation defines solutions.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::core::basic::{Basic, Head, RelOp, UndefinedFunction};
use crate::core::evalf::{Complex, Real};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::solvers::kovacic::{failure, replaced};
use crate::solvers::linear_ode::{normal, UPoly};

type R = PolyResult<Basic>;

/// The coefficients `(f3, f2, f1, f0)` (`Coefficients`).
pub type Coefficients = (Basic, Basic, Basic, Basic);

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
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1)])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn symbol(name: &str) -> Basic {
    crate::core::symbol::Symbol::new(name, &[]).0
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn rat(p: i64, q: i64) -> Basic {
    Basic::rational(Int::from(p), Int::from(q)).expect("a nonzero denominator")
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn is_rational_function(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}
fn eq(a: &Basic, b: &Basic) -> Basic {
    crate::core::relational::rel(RelOp::Eq, a, b)
}
fn is_eq(e: &Basic) -> bool {
    matches!(e.head(), Head::Relational(RelOp::Eq))
}

/// The coefficients of `e` as a polynomial in `v` from the leading one
/// down, each in its normal form, `None` when `e` is no polynomial in `v`
/// (`first_order._all_coeffs`).
pub(crate) fn all_coeffs(e: &Basic, v: &Basic, normal_form: bool) -> PolyResult<Option<Vec<Basic>>> {
    let terms = match crate::solvers::kovacic::coefficients(e, v) {
        Ok(t) => t,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(err) => return Err(err),
    };
    let mut normalized: std::collections::BTreeMap<i64, Basic> = std::collections::BTreeMap::new();
    for (k, c) in terms {
        let c: Basic = if normal_form { normal(&c)? } else { c };
        if c != Basic::zero() {
            normalized.insert(k, c);
        }
    }
    let Some(&top) = normalized.keys().next_back() else {
        return Ok(Some(Vec::new()));
    };
    Ok(Some(
        (0..=top)
            .rev()
            .map(|k| normalized.get(&k).cloned().unwrap_or_else(Basic::zero))
            .collect(),
    ))
}

/// `e` with the keys of `rule` replaced, evaluated, whatever the result
/// (`_xreplace_evaluated`).
pub(crate) fn replace_any(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    crate::core::function::xreplace_evaluated(e, &rule.iter().cloned().collect())
}

/// The degree of a list of [`all_coeffs`] (`_degree`).
pub(crate) fn degree(coefficients: &[Basic]) -> i64 {
    coefficients.len() as i64 - 1
}

/// The relative and absolute invariants (`AbelInvariants`).
#[derive(Clone, Debug)]
pub struct AbelInvariants {
    pub s3: Basic,
    pub s5: Basic,
    pub i1: Basic,
    pub i2: Basic,
}

/// `y = P(x) u(xi(x)) + Q(x)` (`AbelTransformation`).
#[derive(Clone, Debug)]
pub struct AbelTransformation {
    pub xi: Basic,
    pub p: Basic,
    pub q: Basic,
    pub x: Basic,
    pub t: Basic,
}

/// The native `factor` (`_factor`).
fn factor(e: &Basic) -> R {
    match crate::polys::polytools::factor(e, &[], false, true) {
        Ok(f) => Ok(f),
        Err(err) if failure(&err) => Ok(e.clone()),
        Err(err) => Err(err),
    }
}

/// `((f3, f2, f1, f0), x)` of an equation of the first kind
/// (`abel_coefficients`).
pub fn abel_coefficients(equation: &Basic, f: &Basic) -> PolyResult<(Coefficients, Basic)> {
    let (big_f, x, u, p) = crate::solvers::first_order::derivative_polynomial(equation, f)?;
    let Some(in_p) = all_coeffs(&big_f, &p, true)?.filter(|c| degree(c) == 1) else {
        return Err(PolyError::Value(format!(
            "{} is not linear in the derivative",
            equation
        )));
    };
    let (a1, a0) = (&in_p[0], &in_p[1]);
    let rhs: Basic = normal(&neg(&div(a0, a1)?)?)?;
    if rhs.has(&u) && a1.has(&u) {
        return Err(PolyError::Value(format!(
            "{} is not an Abel equation of the first kind",
            equation
        )));
    }
    let Some(cubic) = all_coeffs(&expand(&rhs)?, &u, true)? else {
        return Err(PolyError::Value(format!("{} is not polynomial in {}", equation, f)));
    };
    if degree(&cubic) != 3 {
        return Err(PolyError::Value(format!(
            "{} is not an Abel equation of the first kind",
            equation
        )));
    }
    Ok((
        (cubic[0].clone(), cubic[1].clone(), cubic[2].clone(), cubic[3].clone()),
        x,
    ))
}

/// `(g1, g0)` of the normal form (`_normal_form_data`).
fn normal_form_data(c: &Coefficients, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let (f3, f2, f1, f0) = c;
    let k: Basic = normal(&neg(&div(f2, &mul(&[int(3), f3.clone()])?)?)?)?;
    let g1: Basic = normal(&add(&[
        mul(&[int(3), f3.clone(), pow(&k, &int(2))?])?,
        mul(&[int(2), f2.clone(), k.clone()])?,
        f1.clone(),
    ])?)?;
    let g0: Basic = normal(&add(&[
        mul(&[f3.clone(), pow(&k, &int(3))?])?,
        mul(&[f2.clone(), pow(&k, &int(2))?])?,
        mul(&[f1.clone(), k.clone()])?,
        f0.clone(),
        neg(&diff(&k, x)?)?,
    ])?)?;
    Ok((g1, g0))
}

/// The invariants of `y' = f3 y**3 + f2 y**2 + f1 y + f0`
/// (`abel_invariants`).
pub fn abel_invariants(c: &Coefficients, x: &Basic) -> PolyResult<AbelInvariants> {
    let (f3, _, _, _) = c;
    let (g1, g0) = normal_form_data(c, x)?;
    let s3: Basic = normal(&mul(&[g0.clone(), pow(f3, &int(2))?])?)?;
    let big_t: Basic = normal(&sub(
        &diff(&div(&g0, f3)?, x)?,
        &div(&mul(&[int(3), g1.clone(), g0.clone()])?, f3)?,
    )?)?;
    let s5: Basic = normal(&mul(&[pow(f3, &int(4))?, big_t.clone()])?)?;
    let i1: Basic = if g0 != Basic::zero() {
        factor(&normal(&div(
            &mul(&[pow(f3, &int(2))?, pow(&big_t, &int(3))?])?,
            &pow(&g0, &int(5))?,
        )?)?)?
    } else {
        Basic::zero()
    };
    let i2: Basic = if big_t != Basic::zero() {
        let numerator: Basic = sub(
            &sub(&diff(&big_t, x)?, &mul(&[int(5), g1, big_t.clone()])?)?,
            &div(&mul(&[big_t.clone(), diff(f3, x)?])?, f3)?,
        )?;
        factor(&normal(&div(
            &mul(&[numerator, g0])?,
            &mul(&[f3.clone(), pow(&big_t, &int(2))?])?,
        )?)?)?
    } else {
        Basic::zero()
    };
    Ok(AbelInvariants { s3, s5, i1, i2 })
}

/// Whether the invariant `I1` is constant, or `None` when this cannot be
/// decided (`constant_invariant`).
pub fn constant_invariant(c: &Coefficients, x: &Basic) -> PolyResult<Option<bool>> {
    let i1: Basic = abel_invariants(c, x)?.i1;
    if !i1.has(x) {
        return Ok(Some(true));
    }
    if is_rational_function(&i1, x) {
        return Ok(Some(false));
    }
    Ok(attempt(|| simplify(&diff(&i1, x)?))?.map(|d| d == Basic::zero()))
}

/// A transformation taking the target equation to the representative
/// (`abel_equivalence`).
pub fn abel_equivalence(
    target: &Coefficients,
    x: &Basic,
    representative: &Coefficients,
    t: &Basic,
) -> PolyResult<Option<AbelTransformation>> {
    let invariants: AbelInvariants = abel_invariants(target, x)?;
    let reference: AbelInvariants = abel_invariants(representative, t)?;
    if !invariants.i1.has(x) || !reference.i1.has(t) {
        return Ok(None);
    }
    for xi in xi_candidates(&invariants, &reference, x, t)? {
        if let Some(transformation) = match_transformation(target, x, representative, t, &xi)? {
            return Ok(Some(transformation));
        }
    }
    Ok(None)
}

/// The functions `xi(x)` with `J1(xi) = I1(x)` and `J2(xi) = I2(x)`
/// (`_xi_candidates`).
fn xi_candidates(
    invariants: &AbelInvariants,
    reference: &AbelInvariants,
    x: &Basic,
    t: &Basic,
) -> PolyResult<Vec<Basic>> {
    let xi_symbol: Basic = dummy("xi");
    let numerator = |e: Basic| -> R {
        Ok(crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&e, false, true)?)?.0)
    };
    let first: Basic = numerator(sub(
        &replaced(&reference.i1, &[(t.clone(), xi_symbol.clone())])?,
        &invariants.i1,
    )?)?;
    let second: Basic = numerator(sub(
        &replaced(&reference.i2, &[(t.clone(), xi_symbol.clone())])?,
        &invariants.i2,
    )?)?;
    let common: UPoly = match UPoly::of(&first, &xi_symbol).and_then(|p| p.gcd(&UPoly::of(&second, &xi_symbol)?)) {
        Ok(c) => c,
        Err(e) if failure(&e) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    if common.degree() < 1 {
        return Ok(Vec::new());
    }
    let mut candidates: Vec<Basic> = Vec::new();
    if common.degree() == 1 {
        let coefficients: Vec<Basic> = common.all_coeffs();
        candidates.push(normal(&neg(&div(&coefficients[1], &coefficients[0])?)?)?);
    } else {
        let polynomial: Basic = common.as_expr()?;
        let found: Vec<Basic> = attempt(|| crate::solvers::solvers::solve_equation(&polynomial, &xi_symbol))?
            .flatten()
            .unwrap_or_default();
        for c in found {
            if c.free_symbols().iter().all(|s| s == x) {
                candidates.push(c);
            }
        }
    }
    Ok(candidates)
}

/// `P` and `Q` for a known `xi` (`_match`).
fn match_transformation(
    target: &Coefficients,
    x: &Basic,
    representative: &Coefficients,
    t: &Basic,
    xi: &Basic,
) -> PolyResult<Option<AbelTransformation>> {
    let (f3, f2, f1, f0) = target;
    let at = |c: &Basic| replaced(c, &[(t.clone(), xi.clone())]);
    let (r3, r2, r1, r0) = (
        at(&representative.0)?,
        at(&representative.1)?,
        at(&representative.2)?,
        at(&representative.3)?,
    );
    let dxi: Basic = diff(xi, x)?;
    let square: Basic = normal(&div(&mul(&[r3, dxi.clone()])?, f3)?)?;
    // the square root taken for positive x
    let positive: Basic =
        crate::core::symbol::Dummy::new("x", &[(crate::core::assumptions::Predicate::positive, true)]).0;
    let root: Basic = replaced(
        &pow(&replaced(&square, &[(x.clone(), positive.clone())])?, &Basic::half())?,
        &[(positive, x.clone())],
    )?;
    for sign in [1i64, -1] {
        let p: Basic = simplified(&mul(&[int(sign), root.clone()])?, x)?;
        let q: Basic = normal(&div(
            &sub(&div(&mul(&[r2.clone(), dxi.clone()])?, &p)?, f2)?,
            &mul(&[int(3), f3.clone()])?,
        )?)?;
        let first: Basic = sub(
            &div(
                &sub(
                    &add(&[
                        mul(&[int(3), f3.clone(), pow(&q, &int(2))?])?,
                        mul(&[int(2), f2.clone(), q.clone()])?,
                        f1.clone(),
                    ])?,
                    &div(&diff(&p, x)?, &p)?,
                )?,
                &dxi,
            )?,
            &r1,
        )?;
        if !is_zero(&first, x)? {
            continue;
        }
        let second: Basic = sub(
            &div(
                &sub(
                    &add(&[
                        mul(&[f3.clone(), pow(&q, &int(3))?])?,
                        mul(&[f2.clone(), pow(&q, &int(2))?])?,
                        mul(&[f1.clone(), q.clone()])?,
                        f0.clone(),
                    ])?,
                    &diff(&q, x)?,
                )?,
                &mul(&[dxi.clone(), p.clone()])?,
            )?,
            &r0,
        )?;
        if !is_zero(&second, x)? {
            continue;
        }
        return Ok(Some(AbelTransformation {
            xi: xi.clone(),
            p,
            q,
            x: x.clone(),
            t: t.clone(),
        }));
    }
    Ok(None)
}

/// `e` simplified (`_simplified`).
fn simplified(e: &Basic, x: &Basic) -> R {
    if is_rational_function(e, x) {
        let n: Basic = normal(e)?;
        if is_rational_function(&n, x) {
            return factor(&n);
        }
    }
    simplify(e)
}

/// Whether `e` simplifies to zero (`_is_zero`).
fn is_zero(e: &Basic, _x: &Basic) -> PolyResult<bool> {
    let n: Basic = normal(e)?;
    if n == Basic::zero() {
        return Ok(true);
    }
    let symbols: Vec<Basic> = n.free_symbols().into_iter().collect();
    if crate::core::expr::polynomial_like(&n, &symbols, crate::core::expr::PolynomialKind::Rational) == Some(true) {
        return Ok(false);
    }
    Ok(simplify(e)? == Basic::zero())
}

/// A polynomial or rational particular solution `Q(x)`
/// (`particular_solution`).
pub fn particular_solution(c: &Coefficients, x: &Basic, max_degree: usize) -> PolyResult<Option<Basic>> {
    let (f3, f2, f1, f0) = c;
    if *f0 == Basic::zero() {
        return Ok(Some(Basic::zero()));
    }
    let unknowns: Vec<Basic> = (0..=max_degree).map(|i| symbol(&format!("a{}", i))).collect();
    let denominators: Vec<Basic> = (1..=max_degree).map(|i| symbol(&format!("b{}", i))).collect();
    let residual_of = |q: &Basic| -> R {
        sub(
            &diff(q, x)?,
            &add(&[
                mul(&[f3.clone(), pow(q, &int(3))?])?,
                mul(&[f2.clone(), pow(q, &int(2))?])?,
                mul(&[f1.clone(), q.clone()])?,
                f0.clone(),
            ])?,
        )
    };
    for shape in 0..=max_degree {
        let mut numerator_terms: Vec<Basic> = Vec::new();
        for (i, a) in unknowns.iter().enumerate() {
            numerator_terms.push(mul(&[a.clone(), pow(x, &int(i as i64))?])?);
        }
        let mut denominator_terms: Vec<Basic> = vec![Basic::one()];
        for i in 1..=shape {
            denominator_terms.push(mul(&[denominators[i - 1].clone(), pow(x, &int(i as i64))?])?);
        }
        let numerator: Basic = add(&numerator_terms)?;
        let denominator: Basic = add(&denominator_terms)?;
        let q: Basic = div(&numerator, &denominator)?;
        let residual: Basic = expand(&normal(&mul(&[residual_of(&q)?, pow(&denominator, &int(4))?])?)?)?;
        let Some(equations) = all_coeffs(&residual, x, true)? else {
            continue;
        };
        let mut variables: Vec<Basic> = unknowns.clone();
        variables.extend(denominators[..shape].iter().cloned());
        let solutions = attempt(|| crate::solvers::solvers::solve_system(&equations, &variables))?.flatten();
        let Some(solutions) = solutions.filter(|s| !s.is_empty()) else {
            continue;
        };
        for solution in solutions {
            let mut candidate: Basic = replaced(&q, &solution)?;
            let extra: Vec<(Basic, Basic)> = candidate
                .free_symbols()
                .into_iter()
                .filter(|s| s != x)
                .map(|s| (s, Basic::one()))
                .collect();
            if !extra.is_empty() {
                candidate = replaced(&candidate, &extra)?;
            }
            if normal(&residual_of(&candidate)?)? == Basic::zero() {
                return Ok(Some(normal(&candidate)?));
            }
        }
    }
    Ok(None)
}

fn integrate(e: &Basic, v: &Basic) -> PolyResult<Option<Basic>> {
    crate::integrals::indefinite::closed_antiderivative(e, v, true, false, None)
}

fn quadratic(e: &Basic, x: &Basic) -> PolyResult<bool> {
    Ok(all_coeffs(e, x, true)?.is_some_and(|c| degree(&c) <= 2))
}

// ---------------------------------------------------------------------------
// the numerical check of a relation

/// The digits of the numerical check (`_DIGITS`).
const DIGITS: usize = 30;

fn real_of(z: &Real) -> Rat {
    let repr = z.repr();
    let significand: Int = repr.significand().clone();
    let exponent: isize = repr.exponent();
    if exponent >= 0 {
        Rat::from(significand * (Int::ONE << exponent as usize))
    } else {
        crate::numbers::rat_from_parts(significand, Int::ONE << (-exponent) as usize)
    }
}

fn complex_of(z: &Complex) -> PolyResult<Basic> {
    let (re_part, im_part) = crate::core::evalf::parts(z);
    let (re_value, im_value) = (
        Basic::from_rational(real_of(&re_part)),
        Basic::from_rational(real_of(&im_part)),
    );
    if im_value == Basic::zero() {
        return Ok(re_value);
    }
    add(&[re_value, mul(&[im_value, Basic::atom(Head::ImaginaryUnit)])?])
}

fn evaluate(e: &Basic, y: &Basic, at: &Complex, prec: usize) -> PolyResult<Option<Complex>> {
    let value: Basic = complex_of(at)?;
    let substituted: Basic =
        match crate::core::function::xreplace_evaluated(e, &[(y.clone(), value)].into_iter().collect()) {
            Ok(s) => s,
            Err(err) if failure(&err) => return Ok(None),
            Err(err) => return Err(err),
        };
    Ok(crate::core::evalf::numeric_value(&substituted, prec))
}

fn size(z: &Complex) -> f64 {
    let (a, b) = crate::core::evalf::parts(z);
    a.to_f64().value().hypot(b.to_f64().value())
}

/// A real root of `e(Y) = 0` near `guess` by mpmath's secant method at 30
/// digits (`_real_root`).
fn real_root(e: &Basic, y: &Basic, guess: &Rat) -> PolyResult<Option<Rat>> {
    let prec: usize = crate::numeric::dps_to_prec(DIGITS) + 20;
    let value = |at: &Complex| -> PolyResult<Option<Complex>> { evaluate(e, y, at, prec) };
    let x0: Complex = crate::core::evalf::complex_real(crate::core::evalf::real_rat(guess, prec), prec);
    // tol = eps * 2**10 at the working precision of findroot
    let tol: f64 = 2f64.powi(-(prec as i32) + 1 + 10);
    let Some(f0_start) = value(&x0)? else {
        return Ok(None);
    };
    if size(&f0_start) == 0.0 {
        return Ok(Some(guess.clone()));
    }
    let quarter: Complex =
        crate::core::evalf::complex_real(crate::core::evalf::real_rat(&crate::numbers::rat(1, 4), prec), prec);
    let (mut a, mut b): (Complex, Complex) = (x0.clone(), &x0 + &quarter);
    let mut fa: Complex = f0_start;
    let mut found: Option<Complex> = None;
    for step in 1..=30 {
        let Some(fb) = value(&b)? else {
            return Ok(None);
        };
        let l: Complex = &b - &a;
        if size(&l) == 0.0 {
            break;
        }
        let s: Complex = (&fb - &fa) / &l;
        if size(&s) == 0.0 {
            break;
        }
        let next: Complex = &b - &(&fb / &s);
        a = b;
        b = next;
        fa = fb;
        found = Some(b.clone());
        if size(&l) < tol * size(&b).max(1.0) || step >= 30 {
            break;
        }
    }
    let Some(root) = found else {
        return Ok(None);
    };
    let Some(at_root) = value(&root)? else {
        return Ok(None);
    };
    if size(&at_root).powi(2) > tol {
        return Ok(None);
    }
    let (re_part, im_part) = crate::core::evalf::parts(&root);
    if !im_part.repr().significand().is_zero() {
        return Ok(None);
    }
    Ok(Some(real_of(&re_part)))
}

/// Whether `relation = 0` defines solutions of the Abel equation
/// (`_defines_solution`).
pub(crate) fn defines_solution(
    relation: &Basic,
    c: &Coefficients,
    x: &Basic,
    f: &Basic,
    c1: &Basic,
) -> PolyResult<bool> {
    let mut rng = crate::random::PyRandom::from_int(&Int::ZERO);
    let mut randint = |a: i64, b: i64| a + rng.randbelow((b - a + 1) as u64) as i64;
    let y: Basic = crate::core::symbol::Dummy::new("Y", &[(crate::core::assumptions::Predicate::real, true)]).0;
    let big_f: Basic = replaced(relation, &[(f.clone(), y.clone())])?;
    if !big_f.has(&y) {
        // no curve y(x) at all
        return Ok(false);
    }
    let (f_x, f_y) = (diff(&big_f, x)?, diff(&big_f, &y)?);
    let (f3, f2, f1, f0) = c;
    let implied: Basic = neg(&div(&f_x, &f_y)?)?;
    let rhs: Basic = add(&[
        mul(&[f3.clone(), pow(&y, &int(3))?])?,
        mul(&[f2.clone(), pow(&y, &int(2))?])?,
        mul(&[f1.clone(), y.clone()])?,
        f0.clone(),
    ])?;
    let prec: usize = crate::numeric::dps_to_prec(DIGITS);
    for _ in 0..8 {
        let (p0, q0) = (randint(1, 9), randint(1, 4));
        let (p1, q1) = (randint(-5, 5), randint(1, 3));
        let fixed: Vec<(Basic, Basic)> = vec![(x.clone(), rat(p0, q0)), (c1.clone(), rat(p1, q1))];
        let on_line: Basic = replaced(&big_f, &fixed)?;
        if !on_line.has(&y) {
            continue;
        }
        for guess in [
            crate::numbers::rat(1, 3),
            crate::numbers::rat(-1, 2),
            crate::numbers::rat(2, 1),
            crate::numbers::rat(-3, 1),
        ] {
            let Some(root) = real_root(&on_line, &y, &guess)? else {
                continue;
            };
            let point: Complex = crate::core::evalf::complex_real(crate::core::evalf::real_rat(&root, prec), prec);
            let left = evaluate(&replaced(&implied, &fixed)?, &y, &point, prec)?;
            let right = evaluate(&replaced(&rhs, &fixed)?, &y, &point, prec)?;
            let (Some(left), Some(right)) = (left, right) else {
                continue;
            };
            let real_only = |z: &Complex| crate::core::evalf::parts(z).1.repr().significand().is_zero();
            if !real_only(&left) || !real_only(&right) {
                continue;
            }
            if !size(&left).is_finite() || !size(&right).is_finite() {
                continue;
            }
            return Ok(size(&(&left - &right)) <= 1e-12 * size(&right).max(1.0));
        }
    }
    // no point could be placed on a curve: nothing was shown either way
    Ok(true)
}

/// An implicit solution of an equation of the AIR class through its
/// inverse Riccati equation, or `None` (`air_solution`).
pub fn air_solution(c: &Coefficients, x: &Basic, f: &Basic, q: Option<&Basic>) -> PolyResult<Option<Basic>> {
    let (f3, f2, f1, _) = c;
    let q: Basic = match q {
        Some(q) => q.clone(),
        None => match particular_solution(c, x, 2)? {
            Some(q) => q,
            None => return Ok(None),
        },
    };
    let Some(exponent) = integrate(
        &normal(&add(&[
            mul(&[int(3), f3.clone(), pow(&q, &int(2))?])?,
            mul(&[int(2), f2.clone(), q.clone()])?,
            f1.clone(),
        ])?)?,
        x,
    )?
    else {
        return Ok(None);
    };
    let p: Basic = simplify(&crate::functions::elementary::exponential::exp(&exponent)?)?;
    let g1: Basic = normal(&neg(&pow(&mul(&[f3.clone(), pow(&p, &int(2))?])?, &int(-1))?)?)?;
    if !quadratic(&g1, x)? {
        return Ok(None);
    }
    let r2: Basic = normal(&mul(&[
        p.clone(),
        add(&[mul(&[int(3), f3.clone(), q.clone()])?, f2.clone()])?,
    ])?)?;
    let Some(primitive) = integrate(&r2, x)? else {
        return Ok(None);
    };
    let g0: Basic = normal(&neg(&mul(&[g1.clone(), primitive])?)?)?;
    if !quadratic(&g0, x)? {
        return Ok(None);
    }
    // z = 1/u - g0/g1 solves dx/dz = g1(x) z + g0(x)
    let z: Basic = dummy("z");
    let big_x: Basic = Arc::new(UndefinedFunction::new("X", None, &[])).apply(vec![z.clone()])?;
    let riccati: Basic = eq(
        &diff(&big_x, &z)?,
        &add(&[
            mul(&[replaced(&g1, &[(x.clone(), big_x.clone())])?, z.clone()])?,
            replaced(&g0, &[(x.clone(), big_x.clone())])?,
        ])?,
    );
    let mut solved: Option<Basic> =
        attempt(|| crate::solvers::first_order::riccati_ode(&riccati, &big_x, true))?.flatten();
    if solved.is_none() {
        let found: Option<Vec<Basic>> = attempt(|| crate::solvers::dsolve::dsolve(&riccati, &big_x))?;
        solved = found.filter(|v| v.len() == 1).map(|mut v| v.remove(0));
    }
    let Some(solved) = solved.filter(is_eq) else {
        return Ok(None);
    };
    let z_value: Basic = sub(&div(&p, &sub(f, &q)?)?, &div(&g0, &g1)?)?;
    let difference: Basic = sub(&solved.args()[0], &solved.args()[1])?;
    let mut relation: Basic = replaced(&replaced(&difference, &[(big_x, x.clone())])?, &[(z, z_value)])?;
    let c1: Basic = symbol("C1");
    let mut constants: Vec<Basic> = relation
        .free_symbols()
        .into_iter()
        .filter(|s| s.to_string().starts_with('C'))
        .collect();
    constants.sort_by_key(|s| s.to_string());
    if constants.len() == 1 && constants[0] != c1 {
        relation = replaced(&relation, &[(constants[0].clone(), c1.clone())])?;
    }
    if !defines_solution(&relation, c, x, f, &c1)? {
        // the transformation back can collapse to a relation with no x
        return Ok(None);
    }
    if !relation.free_symbols().contains(&c1) {
        return Ok(Some(eq(&relation, &Basic::zero())));
    }
    // the Riccati solution is linear fractional in C1: isolate it
    let numerator: Basic =
        crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&relation, false, true)?)?.0;
    let Some(linear) = all_coeffs(&numerator, &c1, false)?.filter(|l| degree(l) == 1) else {
        return Ok(Some(eq(&relation, &Basic::zero())));
    };
    Ok(Some(eq(&div(&neg(&linear[1])?, &linear[0])?, &c1)))
}

/// The representatives of integrable classes (`REPRESENTATIVES`).
pub fn representatives() -> PolyResult<Vec<(Coefficients, (Basic, Basic))>> {
    let t: Basic = symbol("t");
    let inv = |e: &Basic, k: i64| pow(e, &int(-k));
    Ok(vec![
        // dx/dz = z + x**2: Airy functions
        (
            (int(-1), mul(&[int(-2), t.clone()])?, Basic::zero(), Basic::zero()),
            (Basic::one(), pow(&t, &int(2))?),
        ),
        // dx/dz = x z + 1: error functions
        (
            (neg(&inv(&t, 1)?)?, inv(&t, 2)?, Basic::zero(), Basic::zero()),
            (t.clone(), Basic::one()),
        ),
        // dx/dz = x**2 z + 1
        (
            (
                neg(&inv(&t, 2)?)?,
                mul(&[int(2), inv(&t, 3)?])?,
                Basic::zero(),
                Basic::zero(),
            ),
            (pow(&t, &int(2))?, Basic::one()),
        ),
    ])
}

type Solved = Mutex<HashMap<Vec<Basic>, Basic>>;

/// The solutions of the representatives, memoised (`_SOLVED`).
fn solved_cache() -> &'static Solved {
    static CACHE: OnceLock<Solved> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn representative_solution(representative: &Coefficients, u: &Basic) -> PolyResult<Option<Basic>> {
    let key: Vec<Basic> = vec![
        representative.0.clone(),
        representative.1.clone(),
        representative.2.clone(),
        representative.3.clone(),
    ];
    if let Some(found) = solved_cache().lock().expect("the cache").get(&key) {
        return Ok(Some(found.clone()));
    }
    let t: Basic = symbol("t");
    let Some(solution) = attempt(|| air_solution(representative, &t, u, Some(&Basic::zero())))?.flatten() else {
        return Ok(None);
    };
    solved_cache().lock().expect("the cache").insert(key, solution.clone());
    Ok(Some(solution))
}

/// An implicit solution of an Abel equation of the first kind with a
/// non-constant invariant (`abel_by_invariants`).
pub fn abel_by_invariants(equation: &Basic, f: &Basic) -> PolyResult<Option<Basic>> {
    let (coefficients, x) = abel_coefficients(equation, f)?;
    if let Some(direct) = attempt(|| air_solution(&coefficients, &x, f, None))?.flatten() {
        return Ok(Some(direct));
    }
    let t: Basic = symbol("t");
    for (representative, _) in representatives()? {
        let Some(transformation) = attempt(|| abel_equivalence(&coefficients, &x, &representative, &t))?.flatten()
        else {
            continue;
        };
        // the representative's solution pulled back through
        // u = (y - Q)/P, t = xi(x)
        let u: Basic = Arc::new(UndefinedFunction::new("U", None, &[])).apply(vec![t.clone()])?;
        let Some(solution) = representative_solution(&representative, &u)?.filter(is_eq) else {
            continue;
        };
        let u_value: Basic = div(&sub(f, &transformation.q)?, &transformation.p)?;
        let pulled: Basic = replace_any(
            &replace_any(&solution, &[(u, u_value)])?,
            &[(t.clone(), transformation.xi.clone())],
        )?;
        if !is_eq(&pulled) {
            continue;
        }
        let relation: Basic = sub(&pulled.args()[0], &pulled.args()[1])?;
        if !defines_solution(&relation, &coefficients, &x, f, &symbol("C1"))? {
            return Ok(None);
        }
        return Ok(Some(eq(&pulled.args()[0], &pulled.args()[1])));
    }
    Ok(None)
}
