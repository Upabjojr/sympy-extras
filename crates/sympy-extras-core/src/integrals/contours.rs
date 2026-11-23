//! Port of `sympy_extras/integrals/contours.py`: three more contours for
//! the residue theorem.
//!
//! 1. The rectangle `0 < Im z < 2*pi/c` for `x**n*exp(k*x)*R(exp(c*x))`
//!    over the real line: a triangular system for the integrals
//!    `A_m = Integral(x**m*exp(k*x)*R(exp(c*x)))`, `m <= n` (or `1 <= m <=
//!    n + 1` when `exp(2*pi*I*k/c) = 1`).
//! 2. The sector of angle `2*pi/n` for `x**a/(b + c*x**n)` over `(0, oo)`,
//!    symbolic `a` and `n` allowed.
//! 3. The indented real line for `R(x)*sin(k*x)` (or `cos`) with simple
//!    real poles at which the integrand is regular.
//!
//! The poles, residues and their placement are those of
//! [`crate::integrals::residues`].

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::integrals::conditions::ConditionalValue;
use crate::integrals::residues::{
    fresh, is_bad, part, poles, rational_function, replace, replace1, residue, tidy, Fraction, Locator, Pole,
    Polynomial,
};

type R = PolyResult<Basic>;
type OptCV = PolyResult<Option<ConditionalValue>>;

/// The hyperbolic functions (SymPy's `HyperbolicFunction`).
const HYPERBOLIC: [Function; 6] = [
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn i_unit() -> Basic {
    Basic::imaginary_unit()
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
fn minus(e: &Basic) -> R {
    mul(&[int(-1), e.clone()])
}
fn quotient(p: &Basic, q: &Basic) -> R {
    mul(&[p.clone(), pow(q, &int(-1))?])
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
fn and_(args: &[Basic]) -> Basic {
    crate::core::logic::and_(args)
}
fn is_true(b: &Basic) -> bool {
    *b == crate::core::logic::true_()
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn budget() -> Option<f64> {
    crate::settings::settings().timeout
}
fn has_function(e: &Basic, fs: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(f) if fs.contains(f)))
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn is_number(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}
fn args_of(e: &Basic, sum: bool) -> Vec<Basic> {
    if (sum && e.is_add()) || (!sum && e.is_mul()) {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn monomial(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    crate::integrals::mellin::monomial(e, x)
}

// ---------------------------------------------------------------------------
// 1. The rectangle

/// `x**n*exp(k*x)*R(u)` with `u = exp(c*x)` (`_Exponential`).
struct Exponential {
    n: i64,
    k: Basic,
    c: Basic,
    fraction: Fraction,
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `f` written as `x**n*exp(k*x)*R(exp(c*x))`, the hyperbolic functions
/// rewritten first (`_exponential_form`).
fn exponential_form(f: &Basic, x: &Basic) -> PolyResult<Option<Exponential>> {
    let g: Basic = if has_function(f, &HYPERBOLIC) {
        crate::core::rewriting::rewrite(f, &[], "exp", &crate::core::rewriting::RewriteHints::default())?
    } else {
        f.clone()
    };
    let mut n: i64 = 0;
    let mut rest: Vec<Basic> = Vec::new();
    for e in args_of(&g, false) {
        if e == *x {
            n += 1;
        } else if e.is_pow() && e.args()[0] == *x && integer_of(&e.args()[1]).is_some_and(|k| k > 0) {
            n += integer_of(&e.args()[1]).unwrap_or(0);
        } else {
            rest.push(e);
        }
    }
    let g: Basic = mul(&rest)?;
    let exponentials: Vec<Basic> = g
        .atoms(Some(&|e: &Basic| e.is_function(Function::Exp)))
        .into_iter()
        .filter(|e| e.args()[0].has(x))
        .collect();
    let mut coefficients: Vec<(i64, i64)> = Vec::new();
    let mut symbolic: usize = 0;
    for node in &exponentials {
        let Some((coefficient, power)) = monomial(&node.args()[0], x)? else {
            return Ok(None);
        };
        if power != Basic::one() {
            return Ok(None);
        }
        match coefficient.as_rational() {
            Some(r) => coefficients.push((
                i64::try_from(&crate::numbers::numer(r)).map_err(|_| crate::error::not_ported("a large exponent"))?,
                i64::try_from(&crate::numbers::denom(r)).map_err(|_| crate::error::not_ported("a large exponent"))?,
            )),
            None => symbolic += 1,
        }
    }
    if coefficients.is_empty() && symbolic == 0 {
        return Ok(None);
    }
    let (mut numerator, mut denominator): (i64, i64) = (0, 1);
    for (p, q) in &coefficients {
        numerator = gcd_i64(numerator, *p);
        denominator = denominator / gcd_i64(denominator, *q) * q;
    }
    let c: Basic = if coefficients.is_empty() {
        Basic::one()
    } else {
        Basic::rational(numerator.into(), denominator.into())?
    };
    let u: Basic = fresh("u", false);
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    let mut k: Basic = Basic::zero();
    for node in &exponentials {
        let Some((coefficient, _)) = monomial(&node.args()[0], x)? else {
            return Ok(None);
        };
        if coefficient.as_rational().is_some() {
            let ratio: Basic = quotient(&coefficient, &c)?;
            if integer_of(&ratio).is_none() {
                return Ok(None);
            }
            replacement.insert(node.clone(), pow(&u, &ratio)?);
        } else {
            if symbolic > 1 || k != Basic::zero() {
                return Ok(None);
            }
            // exp(a*x) stays as the factor exp(k*x)
            k = coefficient;
            replacement.insert(node.clone(), Basic::one());
        }
    }
    let h: Basic = replace(&g, &replacement)?;
    if h.has(x) {
        return Ok(None);
    }
    let Some(fraction) = rational_function(&h, &u)? else {
        return Ok(None);
    };
    Ok(Some(Exponential { n, k, c, fraction }))
}

fn order_at_zero(p: &Polynomial) -> i64 {
    for (i, c) in p.coefficients.iter().rev().enumerate() {
        if *c != Basic::zero() {
            return i as i64;
        }
    }
    0
}

/// The point of the strip `0 < Im z < 2*pi/c` where `exp(c*z)` is the
/// pole of `R`, and whether it lies on the real axis (`_strip_pole`).
fn strip_pole(p: &Pole, c: &Basic, locator: &mut Locator) -> PolyResult<Option<(Basic, bool)>> {
    let u: &Basic = &p.point;
    if *u == Basic::zero() {
        return Ok(None);
    }
    let negative_branch = || -> R { quotient(&add(&[log(&minus(u)?)?, mul(&[i_unit(), Basic::pi()])?])?, c) };
    if let Some(r) = u.as_rational() {
        if *r > crate::numbers::Rat::from(0) {
            return Ok(Some((quotient(&log(u)?, c)?, true)));
        }
        return Ok(Some((negative_branch()?, false)));
    }
    let Some(s) = locator.imaginary_sign(p)? else {
        return Ok(None);
    };
    if s == 0 {
        let real: Basic = crate::functions::elementary::complexes::re(u)?;
        return match locator.decide(&gt(&real, &Basic::zero())?)? {
            None => Ok(None),
            Some(true) => Ok(Some((quotient(&log(u)?, c)?, true))),
            Some(false) => Ok(Some((negative_branch()?, false))),
        };
    }
    let argument: Basic = crate::functions::elementary::argument::arg(u)?;
    let theta: Basic = if s > 0 {
        argument
    } else {
        add(&[argument, mul(&[int(2), Basic::pi()])?])?
    };
    let modulus: Basic = crate::functions::elementary::complexes::abs_(u)?;
    Ok(Some((
        quotient(&add(&[log(&modulus)?, mul(&[i_unit(), theta])?])?, c)?,
        false,
    )))
}

/// The residue of `z**m*exp(k*z)*R(exp(c*z))` at `point`
/// (`_strip_residue`).
fn strip_residue(form: &Exponential, p: &Pole, point: &Basic, m: i64, z: &Basic) -> PolyResult<Option<Basic>> {
    let u: &Basic = &p.point;
    if p.multiplicity == 1 {
        let numerator: Basic = form.fraction.numerator.eval(u)?;
        let slope: Basic = form.fraction.denominator.diff()?.eval(u)?;
        return Ok(Some(mul(&[
            pow(point, &int(m))?,
            exp(&mul(&[form.k.clone(), point.clone()])?)?,
            numerator,
            pow(&mul(&[form.c.clone(), u.clone(), slope])?, &int(-1))?,
        ])?));
    }
    let expression: Basic = mul(&[
        pow(z, &int(m))?,
        exp(&mul(&[form.k.clone(), z.clone()])?)?,
        replace1(
            &form.fraction.as_expr()?,
            &form.fraction.x,
            &exp(&mul(&[form.c.clone(), z.clone()])?)?,
        )?,
    ])?;
    let value: Option<Basic> =
        crate::timeout::attempt_within(budget(), || crate::series::residues::residue(&expression, z, point))?;
    Ok(value.filter(|v| !is_bad(v)))
}

fn known_zero(e: &Basic) -> bool {
    is(e, Predicate::zero) == Some(true)
}

/// The solution of the square linear system by Gaussian elimination with
/// the first pivot not zero (SymPy's `LUsolve`); `None` for a singular
/// system (`_solve`).
fn solve(system: Vec<Vec<Basic>>, rhs: Vec<Basic>) -> PolyResult<Option<Vec<Basic>>> {
    let n: usize = system.len();
    let mut rows: Vec<Vec<Basic>> = system
        .into_iter()
        .zip(rhs)
        .map(|(mut r, b)| {
            r.push(b);
            r
        })
        .collect();
    for j in 0..n {
        let Some(pivot) = (j..n).find(|&i| !known_zero(&rows[i][j])) else {
            return Ok(None);
        };
        rows.swap(j, pivot);
        for i in j + 1..n {
            if known_zero(&rows[i][j]) {
                continue;
            }
            let ratio: Basic = quotient(&rows[i][j], &rows[j][j])?;
            let mut updated: Vec<Basic> = Vec::new();
            for (target, source) in rows[i].iter().zip(&rows[j]) {
                updated.push(add(&[target.clone(), minus(&mul(&[ratio.clone(), source.clone()])?)?])?);
            }
            rows[i] = updated;
        }
    }
    let mut solution: Vec<Basic> = vec![Basic::zero(); n];
    for j in (0..n).rev() {
        let mut known: Vec<Basic> = Vec::new();
        for (entry, value) in rows[j][j + 1..n].iter().zip(&solution[j + 1..n]) {
            known.push(mul(&[entry.clone(), value.clone()])?);
        }
        solution[j] = quotient(&add(&[rows[j][n].clone(), minus(&add(&known)?)?])?, &rows[j][j])?;
    }
    Ok(Some(solution))
}

/// `Integral(x**n*exp(k*x)*R(exp(c*x)), (x, -oo, oo))` by the rectangular
/// contour (`rectangular_integral`).
pub fn rectangular_integral(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptCV {
    let Some(form) = exponential_form(f, x)? else {
        return Ok(None);
    };
    let (p, q) = (&form.fraction.numerator, &form.fraction.denominator);
    let (degree_p, degree_q, order_p, order_q) = (p.degree(), q.degree(), order_at_zero(p), order_at_zero(q));
    let (c, k) = (&form.c, &form.k);
    let mut conditions: Vec<Basic> = Vec::new();
    let lower: Basic = mul(&[c.clone(), int(order_q - order_p)])?;
    let upper: Basic = mul(&[c.clone(), int(degree_q - degree_p)])?;
    if is_number(k) {
        if !(is_true(&lt(&lower, k)?) && is_true(&lt(k, &upper)?)) {
            return Ok(None);
        }
    } else {
        conditions.push(gt(k, &lower)?);
        conditions.push(lt(k, &upper)?);
    }
    let h: Basic = quotient(&mul(&[int(2), Basic::pi()])?, c)?;
    let factor: Basic = exp(&mul(&[i_unit(), k.clone(), h.clone()])?)?;
    let unit: Basic = quotient(k, c)?;
    let shifted: bool = match is(&unit, Predicate::integer) {
        Some(b) => b,
        None => {
            // k/c lies strictly between two consecutive integers when the
            // window of decay is one unit wide: no integer inside
            let one_wide: bool = add(&[upper.clone(), minus(&lower)?])? == *c
                && is(&quotient(&lower, c)?, Predicate::integer) == Some(true);
            if !one_wide {
                let sine: Basic =
                    crate::functions::elementary::trigonometric::sin(&mul(&[Basic::pi(), unit.clone()])?)?;
                conditions.push(crate::core::relational::ne(&sine, &Basic::zero())?);
            }
            false
        }
    };
    let Some(found) = poles(&form.fraction.denominator)? else {
        return Ok(None);
    };
    let mut locator: Locator = Locator::new(assumptions);
    let z: Basic = fresh("z", false);
    let n: i64 = form.n;
    let orders: Vec<i64> = if shifted {
        (1..n + 2).collect()
    } else {
        (0..n + 1).collect()
    };
    let mut system: Vec<Vec<Basic>> = Vec::new();
    let mut rhs: Vec<Basic> = Vec::new();
    let ih: Basic = mul(&[i_unit(), h.clone()])?;
    for &m in &orders {
        let mut total: Basic = Basic::zero();
        for p in &found {
            let Some((point, on_axis)) = strip_pole(p, c, &mut locator)? else {
                return Ok(None);
            };
            if on_axis {
                // a pole of R on the positive axis is a pole of F on the
                // real line, unless z**m cancels a simple pole at 0
                if point != Basic::zero() || p.multiplicity != 1 || m == 0 {
                    return Ok(None);
                }
                continue;
            }
            let Some(value) = strip_residue(&form, p, &point, m, &z)? else {
                return Ok(None);
            };
            total = add(&[total, value])?;
        }
        // the coefficients of A_0, ..., A_n in
        // A_m - factor*sum_j binomial(m, j)*(I*h)**(m - j)*A_j
        let mut row: Vec<Basic> = Vec::new();
        for j in 0..=n {
            let mut entry: Basic = Basic::zero();
            if j <= m.min(n) {
                entry = minus(&mul(&[
                    factor.clone(),
                    crate::functions::combinatorial::factorials::binomial(&int(m), &int(j))?,
                    pow(&ih, &int(m - j))?,
                ])?)?;
            }
            if j == m && !shifted {
                entry = add(&[entry, Basic::one()])?;
            } else if j == m && shifted {
                // the coefficient of A_m is 1 - factor = 0
                entry = Basic::zero();
            }
            row.push(entry);
        }
        system.push(row);
        rhs.push(mul(&[int(2), Basic::pi(), i_unit(), total])?);
    }
    let Some(solution) = solve(system, rhs)? else {
        return Ok(None);
    };
    let mut value: Basic = solution[n as usize].clone();
    if is_bad(&value) {
        return Ok(None);
    }
    value = tidy(&value, assumptions)?;
    if value.has(&i_unit()) {
        // the integral of a real integrand is real
        if let Some(real) = part(&value, true, assumptions)? {
            let simpler: Option<Basic> = crate::timeout::attempt_within(budget(), || {
                crate::simplify::trigsimp::trigsimp(&simplify(&real)?, crate::simplify::trigsimp::Method::Matching)
            })?;
            value = simpler.unwrap_or(real);
        }
    }
    Ok(Some(ConditionalValue::new(value, and_(&conditions))))
}

// ---------------------------------------------------------------------------
// 2. The sector

/// `(c, n)` when `e == c*x**n` with `c` and `n` free of `x`
/// (`_symbolic_monomial`).
fn symbolic_monomial(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let (coefficient, rest) = crate::core::expr::as_independent(e, x, false)?;
    if rest == *x {
        return Ok(Some((coefficient, Basic::one())));
    }
    if rest.is_pow() && rest.args()[0] == *x && !rest.args()[1].has(x) {
        return Ok(Some((coefficient, rest.args()[1].clone())));
    }
    Ok(None)
}

/// `Integral(x**a/(b + c*x**n), (x, 0, oo))` by the sector of angle
/// `2*pi/n` (`sector_integral`).
pub fn sector_integral(f: &Basic, x: &Basic) -> OptCV {
    let mut alpha: Basic = Basic::zero();
    let mut power: Option<(Basic, Basic, Basic)> = None;
    let mut constant: Basic = Basic::one();
    for e in args_of(f, false) {
        if !e.has(x) {
            constant = mul(&[constant, e])?;
        } else if e == *x {
            alpha = add(&[alpha, Basic::one()])?;
        } else if e.is_pow() && e.args()[0] == *x && !e.args()[1].has(x) {
            alpha = add(&[alpha, e.args()[1].clone()])?;
        } else if e.is_pow() && e.args()[1] == Basic::negative_one() && e.args()[0].is_add() && power.is_none() {
            let (b, rest) = crate::core::expr::as_independent(&e.args()[0], x, true)?;
            let found = symbolic_monomial(&rest, x)?;
            let Some((c, n)) = found.filter(|_| b != Basic::zero()) else {
                return Ok(None);
            };
            power = Some((b, c, n));
        } else {
            return Ok(None);
        }
    }
    let Some((b, c, n)) = power else {
        return Ok(None);
    };
    let mut conditions: Vec<Basic> = Vec::new();
    for e in [&b, &c, &n] {
        if is(e, Predicate::positive) == Some(true) {
            continue;
        }
        if is_number(e) {
            return Ok(None);
        }
        conditions.push(gt(e, &Basic::zero())?);
    }
    conditions.push(gt(&alpha, &int(-1))?);
    conditions.push(lt(&alpha, &add(&[n.clone(), int(-1)])?)?);
    let exponent: Basic = quotient(&add(&[alpha, Basic::one()])?, &n)?;
    let sine: Basic = crate::functions::elementary::trigonometric::sin(&mul(&[Basic::pi(), exponent.clone()])?)?;
    let value: Basic = mul(&[
        constant,
        pow(&quotient(&b, &c)?, &exponent)?,
        Basic::pi(),
        pow(&mul(&[b.clone(), n.clone(), sine])?, &int(-1))?,
    ])?;
    Ok(Some(ConditionalValue::new(simplify(&value)?, and_(&conditions))))
}

// ---------------------------------------------------------------------------
// 3. The indented real line

/// `Integral(R(x)*sin(k*x), (x, -oo, oo))` (or `cos`) for a rational `R`
/// with simple real poles at which the integrand is regular, by the
/// indented contour; over `(0, oo)` for an even integrand
/// (`indented_integral`).
pub fn indented_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    let half: bool = *a == Basic::zero() && *b == oo();
    if !half && !(*a == Basic::atom(Head::NegativeInfinity) && *b == oo()) {
        return Ok(None);
    }
    let mut trig: Option<(Basic, bool)> = None;
    let mut rest: Vec<Basic> = Vec::new();
    for e in args_of(f, false) {
        let sine: bool = e.is_function(Function::Sin);
        if (sine || e.is_function(Function::Cos)) && e.has(x) && trig.is_none() {
            let Some((k, power)) = monomial(&e.args()[0], x)? else {
                return Ok(None);
            };
            if power != Basic::one() {
                return Ok(None);
            }
            trig = Some((k, sine));
        } else {
            rest.push(e);
        }
    }
    let Some((k, sine)) = trig else {
        return Ok(None);
    };
    let Some(fraction) = rational_function(&mul(&rest)?, x)? else {
        return Ok(None);
    };
    if fraction.excess() < 1 {
        return Ok(None);
    }
    if half {
        // the integrand must be even: R odd for sin, even for cos
        let r: Basic = fraction.as_expr()?;
        let mirrored: Basic = replace1(&r, x, &minus(x)?)?;
        let parity: i64 = if sine { -1 } else { 1 };
        if simplify(&add(&[mirrored, mul(&[int(-parity), r])?])?)? != Basic::zero() {
            return Ok(None);
        }
    }
    let Some(real_poles) = poles(&fraction.denominator)? else {
        return Ok(None);
    };
    let mut conditions: Vec<Basic> = Vec::new();
    if is_number(&k) {
        if !is_true(&gt(&k, &Basic::zero())?) {
            return Ok(None);
        }
    } else if crate::assumptions::ask::ask(&gt(&k, &Basic::zero())?, assumptions, None)? != Some(true) {
        conditions.push(gt(&k, &Basic::zero())?);
    }
    let mut locator: Locator = Locator::new(assumptions);
    let z: Basic = fresh("z", false);
    let wave: Basic = exp(&mul(&[i_unit(), k.clone(), z.clone()])?)?;
    let mut total: Basic = Basic::zero();
    for p in &real_poles {
        let Some(s) = locator.imaginary_sign(p)? else {
            return Ok(None);
        };
        if s < 0 {
            continue;
        }
        let Some(value) = residue(&fraction, p, &wave, &z, None)? else {
            return Ok(None);
        };
        if s > 0 {
            total = add(&[total, mul(&[int(2), Basic::pi(), i_unit(), value])?])?;
        } else {
            if p.multiplicity != 1 {
                return Ok(None);
            }
            // the integrand must be regular there: the trigonometric
            // factor vanishes
            let at: Basic = mul(&[k.clone(), p.point.clone()])?;
            let vanishing: Basic = if sine {
                crate::functions::elementary::trigonometric::sin(&at)?
            } else {
                crate::functions::elementary::trigonometric::cos(&at)?
            };
            if simplify(&vanishing)? != Basic::zero() {
                return Ok(None);
            }
            total = add(&[total, mul(&[Basic::pi(), i_unit(), value])?])?;
        }
    }
    let Some(found) = part(&total, !sine, assumptions)? else {
        return Ok(None);
    };
    let value: Basic = if half { mul(&[found, Basic::half()])? } else { found };
    Ok(Some(ConditionalValue::new(
        tidy(&value, assumptions)?,
        and_(&conditions),
    )))
}

/// The first of the three contours which applies to `Integral(f, (x, a,
/// b))` (`contour_integral`).
pub fn contour_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    let trigonometric: bool = has_function(f, &[Function::Sin, Function::Cos]);
    if *a == Basic::atom(Head::NegativeInfinity) && *b == oo() {
        let mut exponential: Vec<Function> = HYPERBOLIC.to_vec();
        exponential.push(Function::Exp);
        if has_function(f, &exponential) && !trigonometric {
            if let Some(found) = rectangular_integral(f, x, assumptions)? {
                return Ok(Some(found));
            }
        }
    }
    if *a == Basic::zero() && *b == oo() && !has_function(f, &[Function::Sin, Function::Cos, Function::Exp]) {
        if let Some(found) = sector_integral(f, x)? {
            return Ok(Some(found));
        }
    }
    if trigonometric {
        return indented_integral(f, x, a, b, assumptions);
    }
    Ok(None)
}
