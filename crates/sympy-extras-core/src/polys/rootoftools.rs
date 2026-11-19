//! Port of `sympy_extras/polys/rootoftools.py` (SymPy's `RootSum`): the
//! node `RootSum(f, Lambda(var, body))` of the core, `Head::RootSum` with
//! the arguments `(f, Tuple(var), body, t)` (the `Lambda` flattened, as in
//! `ImageSet`; `t` the bound variable of `f`), which never evaluates, and
//! [`root_sum`], which evaluates as SymPy's constructor does: the
//! polynomial preprocessed (`preprocess_roots`), the terms of the function
//! free of the variable taken out, the polynomial factored, the sums over
//! linear factors (and quadratic ones with `quadratic`) written with the
//! roots, a rational function of the variable summed by the trace in
//! `K[t]/(q)` (Newton's sums), the rest as `RootSum` nodes. The node's
//! `doit` uses `roots`, its derivative is a `RootSum` of the derivative,
//! and without free symbols its value is the sum at the numerical roots
//! (mpmath's `polyroots`, ported). Coefficients with radicals of numbers
//! are taken as elements of `EX` (the polynomial is not factored).

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::basic::{Basic, Head};
use crate::core::evalf::{Complex, Real};
use crate::core::symbol::Dummy;
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::*;
use crate::polys::domains::{Domain, EX};
use crate::polys::polytools::Poly;

type R = PolyResult<Basic>;

fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn tuple(v: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, v)
}

/// The node `RootSum(f, Lambda(var, body))` with the variable `t` of `f`.
pub fn node(f: &Basic, var: &Basic, body: &Basic, t: &Basic) -> Basic {
    Basic::new(
        Head::RootSum,
        vec![f.clone(), tuple(vec![var.clone()]), body.clone(), t.clone()],
    )
}

/// The parts `(f, var, body, t)` of a node.
pub fn parts(e: &Basic) -> (&Basic, &Basic, &Basic, &Basic) {
    let a = e.args();
    (&a[0], &a[1].args()[0], &a[2], &a[3])
}

/// The coefficients of `f` in `t`, highest degree first, over any domain
/// (`polynomial_coefficients`); a `ValueError` when it is no polynomial.
pub fn polynomial_coefficients(f: &Basic, t: &Basic) -> PolyResult<Vec<Basic>> {
    let decomposed = match crate::polys::polytools::decompose_with_gens(f, std::slice::from_ref(t)) {
        Ok(d) => d,
        Err(PolyError::Polys(_, message)) => return Err(value_error(message)),
        Err(e) => return Err(e),
    };
    let mut terms: Vec<(u32, Basic)> = Vec::new();
    for (monomial, parts) in &decomposed {
        let value: Basic = add(parts)?;
        if value != Basic::zero() {
            terms.push((monomial[0], value));
        }
    }
    let Some(n) = terms.iter().map(|(k, _)| *k).max() else {
        return Ok(vec![]);
    };
    Ok((0..=n)
        .rev()
        .map(|k| {
            terms
                .iter()
                .find(|(d, _)| *d == k)
                .map(|(_, c)| c.clone())
                .unwrap_or_else(Basic::zero)
        })
        .collect())
}

/// `sum c_k t**k` of coefficients highest degree first
/// (`expr_of_coefficients`).
pub fn expr_of_coefficients(coefficients: &[Basic], t: &Basic) -> R {
    let n: usize = coefficients.len().saturating_sub(1);
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in coefficients.iter().enumerate() {
        if *c == Basic::zero() {
            continue;
        }
        let k: usize = n - i;
        terms.push(if k > 0 {
            mul(&[c.clone(), pow(t, &Basic::int(k as i64))?])?
        } else {
            c.clone()
        });
    }
    if terms.is_empty() {
        return Ok(Basic::zero());
    }
    add(&terms)
}

fn xreplace(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    crate::core::function::xreplace_evaluated(e, rule)
}

fn call(var: &Basic, body: &Basic, value: &Basic) -> R {
    xreplace(body, &HashMap::from([(var.clone(), value.clone())]))
}

fn is_radical_number(e: &Basic) -> bool {
    if matches!(e.head(), Head::ImaginaryUnit) {
        return true;
    }
    e.is_pow()
        && e.args()[1].is_rational_class()
        && e.args()[1].as_rational().is_some_and(|v| !v.is_int())
        && e.args()[0].free_symbols().is_empty()
}

/// Whether `e` has a radical of a number or `I` (`has_radical_number`).
pub fn has_radical_number(e: &Basic) -> bool {
    e.preorder().iter().any(is_radical_number)
}

/// `e` with its constants which are not rational numbers replaced by
/// dummies (with `radicals` also the radicals of numbers and `I`), and the
/// map back (`generic_constants`).
pub fn generic_constants(e: &Basic, radicals: bool) -> PolyResult<(Basic, HashMap<Basic, Basic>)> {
    fn walk(node: &Basic, radicals: bool, forward: &mut Vec<(Basic, Basic)>) {
        if node.is_rational_class() || (!node.free_symbols().is_empty() && node.args().is_empty()) {
            return;
        }
        let integer_power: bool = node.is_pow()
            && node.args()[1].is_rational_class()
            && node.args()[1].as_rational().is_some_and(|v| v.is_int());
        if node.free_symbols().is_empty()
            && !node.is_add()
            && !node.is_mul()
            && (radicals || !has_radical_number(node))
            && !integer_power
        {
            if !forward.iter().any(|(k, _)| k == node) {
                let d: Basic = Dummy::new(&format!("c{}", forward.len()), &[]).0;
                forward.push((node.clone(), d));
            }
            return;
        }
        for a in node.args() {
            if !matches!(a.head(), Head::Tuple) && !crate::core::logic::is_boolean(a) {
                walk(a, radicals, forward);
            }
        }
    }
    let mut forward: Vec<(Basic, Basic)> = Vec::new();
    walk(e, radicals, &mut forward);
    let back: HashMap<Basic, Basic> = forward.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
    if forward.is_empty() {
        return Ok((e.clone(), back));
    }
    Ok((xreplace(e, &forward.into_iter().collect())?, back))
}

/// The polynomial of a `RootSum` after `preprocess_roots`: a `Poly` over a
/// polynomial domain (`back` restoring its constants), or the
/// coefficients over `EX` (`poly` is `None`).
pub(crate) struct Polynomial {
    poly: Option<Poly>,
    back: HashMap<Basic, Basic>,
    coefficients: Vec<Basic>,
    t: Basic,
}

impl Polynomial {
    pub(crate) fn new(
        poly: Option<Poly>,
        coefficients: Vec<Basic>,
        t: &Basic,
        back: HashMap<Basic, Basic>,
    ) -> PolyResult<Polynomial> {
        let restored: Vec<Basic> = coefficients
            .iter()
            .map(|c| {
                if back.is_empty() {
                    Ok(c.clone())
                } else {
                    xreplace(c, &back)
                }
            })
            .collect::<PolyResult<_>>()?;
        Ok(Polynomial {
            poly,
            back,
            coefficients: restored,
            t: t.clone(),
        })
    }
    fn restored(&self, e: &Basic) -> R {
        if self.back.is_empty() {
            Ok(e.clone())
        } else {
            xreplace(e, &self.back)
        }
    }
    fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }
    fn as_expr(&self) -> R {
        match &self.poly {
            Some(p) => self.restored(&p.as_expr()?),
            None => expr_of_coefficients(&self.coefficients, &self.t),
        }
    }
}

pub(crate) fn is_failure(e: &PolyError) -> bool {
    !matches!(
        e,
        PolyError::Index(_) | PolyError::Recursion(_) | PolyError::Arithmetic(_)
    )
}

fn transformed(f: &Basic, t: &Basic) -> PolyResult<(Basic, Polynomial)> {
    if polynomial_coefficients(f, t)?.is_empty() {
        return Err(value_error("the zero polynomial has no roots to sum over".to_string()));
    }
    let canceled: Basic = crate::polys::polytools::cancel(f, std::slice::from_ref(t))?;
    let numerator: Basic = crate::core::expr::as_numer_denom(&canceled)?.0;
    let ex = |numerator: &Basic| -> PolyResult<(Basic, Polynomial)> {
        Ok((
            Basic::one(),
            Polynomial::new(None, polynomial_coefficients(numerator, t)?, t, HashMap::new())?,
        ))
    };
    if has_radical_number(&numerator) {
        return ex(&numerator);
    }
    let (generic, back) = generic_constants(&numerator, false)?;
    let native: Poly = match Poly::new(&generic, std::slice::from_ref(t)) {
        Ok(p) => p,
        Err(e) if is_failure(&e) => return ex(&numerator),
        Err(e) => return Err(e),
    };
    if native.gens.as_slice() != std::slice::from_ref(t) {
        return ex(&numerator);
    }
    let (coeff, poly) = crate::polys::polyroots::preprocess_roots(&native)?;
    let coefficients: Vec<Basic> = poly.all_coeffs()?;
    let p: Polynomial = Polynomial::new(Some(poly), coefficients, t, back)?;
    Ok((p.restored(&coeff)?, p))
}

fn pure_factors(p: &Polynomial) -> PolyResult<Vec<(Polynomial, usize)>> {
    let Some(poly) = &p.poly else {
        return Ok(vec![(
            Polynomial::new(None, p.coefficients.clone(), &p.t, HashMap::new())?,
            1,
        )]);
    };
    let (_, factors) = poly.factor_list()?;
    factors
        .into_iter()
        .map(|(g, k)| {
            Ok((
                Polynomial::new(Some(g.clone()), g.all_coeffs()?, &p.t, p.back.clone())?,
                k,
            ))
        })
        .collect()
}

fn simplified(e: &Basic) -> R {
    let found = crate::timeout::attempt_within(crate::settings::settings().timeout.map(|t| t / 10.0), || {
        crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
    })?;
    Ok(found.unwrap_or_else(|| e.clone()))
}

pub(crate) fn linear_root(p: &Polynomial) -> R {
    if let Some(poly) = &p.poly {
        return p.restored(&crate::polys::polyroots::roots_linear(poly)?[0]);
    }
    let (b, c) = (&p.coefficients[0], &p.coefficients[1]);
    simplified(&mul(&[
        Basic::negative_one(),
        c.clone(),
        pow(b, &Basic::negative_one())?,
    ])?)
}

pub(crate) fn quadratic_roots(p: &Polynomial) -> PolyResult<Vec<Basic>> {
    if let Some(poly) = &p.poly {
        return crate::polys::polyroots::roots_quadratic(poly)?
            .iter()
            .map(|r| p.restored(r))
            .collect();
    }
    let (a, b, c) = (&p.coefficients[0], &p.coefficients[1], &p.coefficients[2]);
    let zero: Basic = Basic::zero();
    let minus_one: Basic = Basic::negative_one();
    if *c == zero {
        return Ok(vec![
            zero,
            simplified(&mul(&[minus_one, b.clone(), pow(a, &Basic::negative_one())?])?)?,
        ]);
    }
    let sqrt = crate::polys::polyroots::sqrt_taken;
    if *b == zero {
        let r: Basic = sqrt(&simplified(&mul(&[
            minus_one,
            c.clone(),
            pow(a, &Basic::negative_one())?,
        ])?)?)?;
        return Ok(vec![neg(&r)?, r]);
    }
    let d: Basic = simplified(&add(&[
        pow(b, &Basic::int(2))?,
        mul(&[Basic::int(-4), a.clone(), c.clone()])?,
    ])?)?;
    let big_a: Basic = mul(&[Basic::int(2), a.clone()])?;
    let big_b: Basic = simplified(&mul(&[minus_one, b.clone(), pow(&big_a, &Basic::negative_one())?])?)?;
    let big_d: Basic = crate::core::exprtools::factor_terms(
        &mul(&[sqrt(&d)?, pow(&big_a, &Basic::negative_one())?])?,
        false,
        false,
        false,
        true,
    )?;
    let (mut r0, mut r1) = (crate::core::add::sub(&big_b, &big_d)?, add(&[big_b, big_d])?);
    if crate::core::expr::is_true(a, crate::core::assumptions::Predicate::negative) {
        std::mem::swap(&mut r0, &mut r1);
    } else {
        r0 = crate::polys::polyroots::expand_2arg(&r0)?;
        r1 = crate::polys::polyroots::expand_2arg(&r1)?;
    }
    Ok(vec![r0, r1])
}

fn is_rational_function(e: &Basic, var: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(var),
        crate::core::expr::PolynomialKind::Rational,
    ) == Some(true)
}

/// The sum of a rational function over the roots of an irreducible `p`:
/// its trace in `K[t]/(p)` by Newton's sums (`_rational_case`).
fn rational_case(p: &Polynomial, var: &Basic, body: &Basic) -> PolyResult<Option<Basic>> {
    let n: usize = p.degree();
    let canceled: Basic = crate::polys::polytools::cancel(body, std::slice::from_ref(var))?;
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&canceled)?;
    let (Ok(big_n), Ok(big_d)) = (
        polynomial_coefficients(&numerator, var),
        polynomial_coefficients(&denominator, var),
    ) else {
        return Ok(None);
    };
    let k = EX;
    let lc: Basic = p.coefficients[0].clone();
    let q: Vec<Basic> = crate::polys::densebasic::dup_strip(
        &p.coefficients
            .iter()
            .map(|c| k.exquo(c, &lc))
            .collect::<PolyResult<Vec<Basic>>>()?,
        &k,
    );
    let reduced: Vec<Basic> = crate::polys::densearith::dup_rem(&big_d, &q, &k)?;
    let inverse: Vec<Basic> = match crate::polys::euclidtools::dup_invert(&reduced, &q, &k) {
        Ok(v) => v,
        Err(PolyError::Value(_)) | Err(PolyError::ZeroDivision(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let s: Vec<Basic> =
        crate::polys::densearith::dup_rem(&crate::polys::densearith::dup_mul(&big_n, &inverse, &k), &q, &k)?;
    let c: &Vec<Basic> = &q;
    let mut sums: Vec<Basic> = vec![Basic::int(n as i64)];
    for kk in 1..n {
        let mut total: Basic = mul(&[Basic::int(kk as i64), c[kk].clone()])?;
        for i in 1..kk {
            total = k.add(&total, &k.mul(&c[i], &sums[kk - i]));
        }
        sums.push(if total != Basic::zero() {
            k.neg(&total)
        } else {
            Basic::zero()
        });
    }
    let mut trace: Basic = Basic::zero();
    let m: isize = crate::polys::densebasic::dup_degree(&s);
    for (i, si) in s.iter().enumerate() {
        trace = k.add(&trace, &k.mul(si, &sums[(m - i as isize) as usize]));
    }
    Ok(Some(crate::polys::polytools::factor(
        &crate::polys::domains::expressiondomain::ex_normal(&trace)?,
        &[],
        false,
        true,
    )?))
}

/// SymPy's `RootSum.new`: a function free of the variable times the
/// degree, the rational case, or the node (`_root_sum`).
fn root_sum_new(poly: &Polynomial, var: &Basic, body: &Basic, auto: bool) -> R {
    if !body.has(var) {
        return mul(&[Basic::int(poly.degree() as i64), body.clone()]);
    }
    if auto && is_rational_function(body, var) {
        if let Some(value) = rational_case(poly, var, body)? {
            return Ok(value);
        }
    }
    Ok(node(&poly.as_expr()?, var, body, &poly.t))
}

/// The sum of `Lambda(var, body)` (the identity without one) over the
/// roots of the polynomial `f` in `t` (the one symbol of `f` without it),
/// evaluated as SymPy's `RootSum` (`root_sum`).
pub fn root_sum(f: &Basic, fun: Option<(&Basic, &Basic)>, t: Option<&Basic>, auto: bool, quadratic: bool) -> R {
    let t: Basic = match t {
        Some(t) => t.clone(),
        None => {
            let mut free: Vec<Basic> = f.free_symbols().into_iter().collect();
            free.sort_by_cached_key(crate::core::sorting::sort_key);
            if free.len() != 1 {
                return Err(value_error(format!("the variable of {} is needed", f)));
            }
            free.remove(0)
        }
    };
    let (coeff, poly) = transformed(f, &t)?;
    let (var, mut expr): (Basic, Basic) = match fun {
        Some((v, b)) => (v.clone(), b.clone()),
        None => (t.clone(), t.clone()),
    };
    if coeff != Basic::one() {
        expr = crate::core::subs::subs(&expr, &var, &mul(&[coeff, var.clone()])?)?;
    }
    let deg: Basic = Basic::int(poly.degree() as i64);
    if !expr.has(&var) {
        return mul(&[deg, expr]);
    }
    let (mut add_const, mut mul_const) = (Basic::zero(), Basic::one());
    if expr.is_add() {
        (add_const, expr) = crate::core::expr::as_independent(&expr, &var, true)?;
    }
    if expr.is_mul() {
        (mul_const, expr) = crate::core::expr::as_independent(&expr, &var, false)?;
    }
    let rational: bool = is_rational_function(&expr, &var);
    let mut terms: Vec<Basic> = Vec::new();
    for (factor_poly, k) in pure_factors(&poly)? {
        let term: Basic = if factor_poly.degree() == 1 {
            call(&var, &expr, &linear_root(&factor_poly)?)?
        } else if quadratic && factor_poly.degree() == 2 {
            let mut values: Vec<Basic> = vec![Basic::zero()];
            for r in quadratic_roots(&factor_poly)? {
                values.push(call(&var, &expr, &r)?);
            }
            add(&values)?
        } else {
            let value: Option<Basic> = if rational && auto {
                rational_case(&factor_poly, &var, &expr)?
            } else {
                None
            };
            match value {
                Some(v) => v,
                None => node(&factor_poly.as_expr()?, &var, &expr, &t),
            }
        };
        terms.push(mul(&[Basic::int(k as i64), term])?);
    }
    add(&[mul(&[mul_const, add(&terms)?])?, mul(&[deg, add_const])?])
}

/// The free symbols of a node: those of the polynomial but its variable,
/// and those of the function but its variable.
pub fn free_symbols(e: &Basic) -> std::collections::BTreeSet<Basic> {
    let (f, var, body, t) = parts(e);
    let mut result: std::collections::BTreeSet<Basic> = f.free_symbols().into_iter().filter(|s| s != t).collect();
    result.extend(body.free_symbols().into_iter().filter(|s| s != var));
    result
}

fn replaced(e: &Basic, f: Basic, body: Basic, evaluated: bool) -> R {
    let (f0, var, body0, t) = parts(e);
    if f == *f0 && body == *body0 {
        return Ok(e.clone());
    }
    if evaluated {
        root_sum(&f, Some((var, &body)), Some(t), true, false)
    } else {
        Ok(node(&f, var, &body, t))
    }
}

/// `subs`: the bound variables left alone, the result evaluated.
pub fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    if e == old {
        return Ok(new.clone());
    }
    let (f, var, body, t) = parts(e);
    let f2: Basic = if old == t {
        f.clone()
    } else {
        crate::core::subs::subs(f, old, new)?
    };
    let body2: Basic = if old == var {
        body.clone()
    } else {
        crate::core::subs::subs(body, old, new)?
    };
    replaced(e, f2, body2, true)
}

/// `_xreplace_evaluated`.
pub fn xreplace_evaluated(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    if let Some(v) = rule.get(e) {
        return Ok(v.clone());
    }
    let (f, _, body, _) = parts(e);
    replaced(e, xreplace(f, rule)?, xreplace(body, rule)?, true)
}

/// `_xreplace` (unevaluated).
pub fn xreplace_plain(e: &Basic, rule: &HashMap<Basic, Basic>) -> Basic {
    if let Some(v) = rule.get(e) {
        return v.clone();
    }
    let (f, var, body, t) = parts(e);
    let (f2, body2) = (f.xreplace(rule), body.xreplace(rule));
    if f2 == *f && body2 == *body {
        return e.clone();
    }
    node(&f2, var, &body2, t)
}

/// The rebuild evaluated (`_evaluated`).
pub fn evaluated(args: &[Basic]) -> R {
    root_sum(
        &args[0],
        Some((&args[1].args()[0], &args[2])),
        Some(&args[3]),
        true,
        false,
    )
}

/// The sum written with the roots when they are all found (`doit`).
pub fn doit(e: &Basic) -> R {
    let (f, var, body, t) = parts(e);
    let degree: usize = polynomial_coefficients(f, t)?.len().saturating_sub(1);
    let found: Vec<Basic> = match crate::polys::polyroots::roots_list(f, std::slice::from_ref(t), true, true, None) {
        Ok(found) => found,
        Err(err) if is_failure(&err) => return Ok(e.clone()),
        Err(err) => return Err(err),
    };
    if found.len() < degree {
        return Ok(e.clone());
    }
    let values: Vec<Basic> = found.iter().map(|r| call(var, body, r)).collect::<PolyResult<_>>()?;
    add(&values)
}

/// The derivative: a `RootSum` of the derivative (`_eval_derivative`).
pub fn derivative(e: &Basic, s: &Basic) -> R {
    let (f, var, body, t) = parts(e);
    let d: Basic = crate::core::function::derivative(body, s)?;
    let poly: Polynomial = Polynomial::new(None, polynomial_coefficients(f, t)?, t, HashMap::new())?;
    root_sum_new(&poly, var, &d, true)
}

fn rat_of_real(r: &Real) -> Rat {
    let repr = r.repr();
    let significand: Int = repr.significand().clone();
    let exponent: isize = repr.exponent();
    if exponent >= 0 {
        Rat::from(significand << (exponent as usize))
    } else {
        rat_from_parts(significand, Int::ONE << ((-exponent) as usize))
    }
}

/// The value of a node without free symbols: the sum of the function at
/// the numerical roots (`_numeric_node`).
pub fn numeric_value(e: &Basic, prec: usize) -> Option<Complex> {
    if !free_symbols(e).is_empty() {
        return None;
    }
    let (f, var, body, t) = parts(e);
    let working: usize = prec + 30;
    let coefficients: Vec<Basic> = polynomial_coefficients(f, t).ok()?;
    let mut values: Vec<Complex> = Vec::new();
    for c in &coefficients {
        values.push(crate::core::evalf::numeric_value(c, working)?);
    }
    if values.len() < 2 {
        return Some(crate::core::evalf::complex_real(
            crate::core::evalf::real_rat(&Rat::ZERO, prec),
            prec,
        ));
    }
    let found: Vec<Complex> =
        crate::polys::regularchains::solutions::polyroots_complex(&values, 500, 2 * working, working)?;
    let mut total: Option<Complex> = None;
    for z in found {
        let (re, im) = crate::core::evalf::parts(&z);
        let point: Basic = add(&[
            Basic::from_rational(rat_of_real(&re)),
            mul(&[Basic::from_rational(rat_of_real(&im)), Basic::imaginary_unit()]).ok()?,
        ])
        .ok()?;
        let value: Complex = crate::core::evalf::numeric_value(&call(var, body, &point).ok()?, working)?;
        total = Some(match total {
            None => value,
            Some(t) => t + value,
        });
    }
    total
}

/// `str` of a node.
pub fn print(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let (f, var, body, _) = parts(e);
    format!("RootSum({}, Lambda({}, {}))", doprint(f), doprint(var), doprint(body))
}
