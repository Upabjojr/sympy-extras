//! Port of `sympy_extras/solvers/dsolve.py`: the general solutions of
//! ordinary differential equations by the classical methods (linear
//! equations with constant coefficients, Euler's equations, the reduction
//! to the lowest derivative, the algorithms for rational coefficients,
//! variation of parameters; separable, Bernoulli, exact and homogeneous
//! first order equations; products of factors).

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, BTreeSet};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::solvers::kovacic::{coefficients as poly_coefficients, failure, replaced};

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
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    if n == 0 {
        return Ok(e.clone());
    }
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn sum(terms: &[Basic]) -> R {
    if terms.is_empty() {
        Ok(Basic::zero())
    } else {
        add(terms)
    }
}
fn eq(a: &Basic, b: &Basic) -> Basic {
    crate::core::relational::rel(RelOp::Eq, a, b)
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b) && !crate::sets::sets::is_set(b) && !matches!(b.head(), Head::Tuple)
}
fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn factors_of(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn is_eq(e: &Basic) -> bool {
    matches!(e.head(), Head::Relational(RelOp::Eq))
}

/// The normal form by the native `cancel`, `e` itself when it fails
/// (`_normal`).
fn normal(e: &Basic) -> R {
    match crate::polys::polytools::cancel(e, &[]) {
        Ok(c) => Ok(c),
        Err(err) if failure(&err) => Ok(e.clone()),
        Err(err) => Err(err),
    }
}

/// The expression which is zero (`_as_zero`).
fn as_zero(equation: &Basic) -> R {
    if is_eq(equation) {
        let (lhs, rhs) = (&equation.args()[0], &equation.args()[1]);
        if is_expr(lhs) && is_expr(rhs) {
            return sub(lhs, rhs);
        }
        return Err(PolyError::Type(format!(
            "{} is not an equation of expressions",
            equation
        )));
    }
    if !is_expr(equation) {
        return Err(PolyError::Type(format!("{} is not an equation", equation)));
    }
    Ok(equation.clone())
}

/// The variable of `f(x)` (`_variable`).
fn variable(f: &Basic) -> PolyResult<Basic> {
    if f.args().len() != 1 || !f.args()[0].is_symbol() {
        return Err(PolyError::Value(format!("{} is not a function of one symbol", f)));
    }
    Ok(f.args()[0].clone())
}

/// The order of the highest derivative of `f` in `e` (`_order`).
fn order(e: &Basic, f: &Basic) -> PolyResult<u64> {
    let x: Basic = variable(f)?;
    let mut n: u64 = 0;
    for d in e.atoms(Some(&|t: &Basic| matches!(t.head(), Head::Derivative))) {
        if d.args()[0] == *f {
            let mut count: u64 = 0;
            for pair in &d.args()[1..] {
                if pair.args()[0] == x {
                    count += pair.args()[1]
                        .as_rational()
                        .and_then(|r| u64::try_from(&crate::numbers::numer(r)).ok())
                        .unwrap_or(1);
                }
            }
            n = n.max(count);
        }
    }
    Ok(n)
}

/// `n` constants `C1, C2, ...` not in `taken` (`_constants`).
fn constants(taken: &BTreeSet<Basic>, n: usize) -> Vec<Basic> {
    let names: BTreeSet<String> = taken.iter().map(|s| s.to_string()).collect();
    let mut result: Vec<Basic> = Vec::new();
    let mut k: usize = 1;
    while result.len() < n {
        let name: String = format!("C{}", k);
        if !names.contains(&name) {
            result.push(crate::core::symbol::Symbol::new(&name, &[]).0);
        }
        k += 1;
    }
    result
}

fn symbols_of(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect()
}

fn closed_integral(e: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    crate::integrals::indefinite::closed_antiderivative(e, x, false, false, None)
}

// ---------------------------------------------------------------------------
// linear equations

/// `([a_0, ..., a_n], b)` with `a_n y^(n) + ... + a_0 y = b`, when the
/// equation is linear in `f` (`linear_coefficients`).
pub fn linear_coefficients(equation: &Basic, f: &Basic) -> PolyResult<Option<(Vec<Basic>, Basic)>> {
    let e: Basic = as_zero(equation)?;
    let x: Basic = variable(f)?;
    let n: u64 = order(&e, f)?;
    let mut coefficients: Vec<Basic> = Vec::new();
    let mut rest: Basic = expand(&e)?;
    for i in (0..=n).rev() {
        let term: Basic = diff(f, &x, i)?;
        let c: Basic = crate::core::expr::coeff(&rest, &term)?;
        if c != Basic::zero() {
            rest = expand(&sub(&rest, &mul(&[c.clone(), term])?)?)?;
        }
        coefficients.push(c);
    }
    if rest.has(f) || coefficients.iter().any(|c| c.has(f)) {
        return Ok(None);
    }
    coefficients.reverse();
    if *coefficients.last().expect("nonempty") == Basic::zero() {
        return Ok(None);
    }
    Ok(Some((coefficients, neg(&rest)?)))
}

/// The coefficients times the product of their distinct denominators
/// (`_cleared`).
fn cleared(p: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut denominators: Vec<Basic> = Vec::new();
    for c in p {
        let d: Basic = crate::core::expr::as_numer_denom(&normal(c)?)?.1;
        if d != Basic::one() {
            let mut known: bool = false;
            for e in &denominators {
                if expand(&sub(&d, e)?)? == Basic::zero() {
                    known = true;
                    break;
                }
            }
            if !known {
                denominators.push(d);
            }
        }
    }
    if denominators.is_empty() {
        return Ok(p.to_vec());
    }
    let common: Basic = mul(&denominators)?;
    p.iter().map(|c| normal(&mul(&[c.clone(), common.clone()])?)).collect()
}

/// `e`, a polynomial in `m`, with the denominators of its coefficients
/// cleared (`_cleared_polynomial`).
fn cleared_polynomial(e: &Basic, m: &Basic) -> R {
    let terms: BTreeMap<i64, Basic> = poly_coefficients(e, m)?;
    let degree: i64 = terms.keys().next_back().copied().unwrap_or(0);
    let coefficients: Vec<Basic> = (0..=degree)
        .map(|i| terms.get(&i).cloned().unwrap_or_else(Basic::zero))
        .collect();
    let c: Vec<Basic> = cleared(&coefficients)?;
    let mut parts: Vec<Basic> = Vec::new();
    for (i, ci) in c.iter().enumerate() {
        parts.push(mul(&[ci.clone(), pow(m, &int(i as i64))?])?);
    }
    expand(&sum(&parts)?)
}

type Roots = Vec<(Basic, usize)>;

fn sorted_roots(mut pairs: Roots) -> Roots {
    pairs.sort_by_cached_key(|(r, _)| crate::core::sorting::sort_key(r));
    pairs
}

/// The roots of `sum p_i m**i` with their multiplicities, `None` when they
/// are not all found (`_characteristic_roots`).
fn characteristic_roots(p: &[Basic]) -> PolyResult<Option<Roots>> {
    let m: Basic = dummy("m");
    let mut parts: Vec<Basic> = Vec::new();
    for (i, c) in p.iter().enumerate() {
        parts.push(mul(&[c.clone(), pow(&m, &int(i as i64))?])?);
    }
    let polynomial: Basic = sum(&parts)?;
    let found = match crate::polys::polyroots::roots(&polynomial, std::slice::from_ref(&m), true, true, None) {
        Ok(found) => found,
        Err(e) if failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let pairs: Roots = found
        .iter()
        .filter(|(r, _)| is_expr(r))
        .map(|(r, k)| (r.clone(), *k))
        .collect();
    if pairs.iter().map(|(_, k)| *k).sum::<usize>() != p.len() - 1 {
        return Ok(None);
    }
    Ok(Some(sorted_roots(pairs)))
}

/// `(a, b)` with `r = a + b I` when `r` has an explicit imaginary part
/// (`_real_imaginary`).
fn real_imaginary(r: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let i_unit: Basic = Basic::atom(Head::ImaginaryUnit);
    if !r.has(&i_unit) {
        return Ok(None);
    }
    let a: Basic = replaced(r, &[(i_unit.clone(), Basic::zero())])?;
    let b: Basic = crate::core::expr::coeff(&expand(&sub(r, &a)?)?, &i_unit)?;
    if b == Basic::zero()
        || b.has(&i_unit)
        || is(&b, Predicate::extended_real) != Some(true)
        || is(&a, Predicate::extended_real) != Some(true)
        || expand(&sub(&add(&[a.clone(), mul(&[b.clone(), i_unit])?])?, r)?)? != Basic::zero()
    {
        return Ok(None);
    }
    Ok(Some((expand(&a)?, b)))
}

/// The solutions for the roots, a conjugate pair as real functions
/// (`_root_basis`).
fn root_basis(pairs: &Roots, x: &Basic, t: &Basic) -> PolyResult<Vec<Basic>> {
    use crate::functions::elementary::trigonometric::{cos, sin};
    let euler: bool = t != x;
    let mut basis: Vec<Basic> = Vec::new();
    let mut done: Vec<Basic> = Vec::new();
    let power = |k: usize| -> R {
        if k != 0 {
            pow(t, &int(k as i64))
        } else {
            Ok(Basic::one())
        }
    };
    let growth = |a: &Basic| -> R {
        if euler {
            pow(x, a)
        } else {
            exp(&mul(&[a.clone(), x.clone()])?)
        }
    };
    for (r, k) in pairs {
        let mut seen: bool = false;
        for d in &done {
            if expand(&sub(r, d)?)? == Basic::zero() {
                seen = true;
                break;
            }
        }
        if seen {
            continue;
        }
        if let Some((a, mut b)) = real_imaginary(r)? {
            let i_unit: Basic = Basic::atom(Head::ImaginaryUnit);
            let conjugate: Basic = sub(&a, &mul(&[b.clone(), i_unit])?)?;
            let mut partner: bool = false;
            for (q, j) in pairs {
                if j == k && expand(&sub(q, &conjugate)?)? == Basic::zero() {
                    partner = true;
                    break;
                }
            }
            if partner {
                done.push(conjugate);
                if b.is_number_node() && is(&b, Predicate::negative) == Some(true) {
                    b = neg(&b)?;
                }
                let bt: Basic = mul(&[b.clone(), t.clone()])?;
                for j in 0..*k {
                    basis.push(mul(&[power(j)?, growth(&a)?, sin(&bt)?])?);
                    basis.push(mul(&[power(j)?, growth(&a)?, cos(&bt)?])?);
                }
                continue;
            }
        }
        done.push(r.clone());
        for j in 0..*k {
            basis.push(mul(&[power(j)?, growth(r)?])?);
        }
    }
    Ok(basis)
}

/// A basis of the solutions of `sum a_i y^(i) = 0`, `None` when not all of
/// them are found (`linear_basis`).
pub fn linear_basis(coefficients: &[Basic], x: &Basic, general: bool) -> PolyResult<Option<Vec<Basic>>> {
    let n: usize = coefficients.len() - 1;
    let leading: &Basic = &coefficients[n];
    let p: Vec<Basic> = coefficients
        .iter()
        .map(|c| normal(&div(c, leading)?))
        .collect::<PolyResult<_>>()?;
    if n == 0 {
        return Ok(Some(Vec::new()));
    }
    if p.iter().all(|c| !c.has(x)) {
        return match characteristic_roots(&cleared(&p)?)? {
            None => Ok(None),
            Some(pairs) => Ok(Some(root_basis(&pairs, x, x)?)),
        };
    }
    let scaled: Vec<Basic> = p
        .iter()
        .enumerate()
        .map(|(i, c)| normal(&mul(&[c.clone(), pow(x, &int((n - i) as i64))?])?))
        .collect::<PolyResult<_>>()?;
    if scaled.iter().all(|c| !c.has(x)) {
        // Euler: x**n y^(n) + ... with y = x**m
        let m: Basic = dummy("m");
        let mut indicial: Basic = Basic::zero();
        for (i, c) in scaled.iter().enumerate() {
            let mut falling: Basic = Basic::one();
            for j in 0..i {
                falling = mul(&[falling, sub(&m, &int(j as i64))?])?;
            }
            indicial = add(&[indicial, mul(&[c.clone(), falling])?])?;
        }
        let found: Roots = match cleared_polynomial(&indicial, &m)
            .and_then(|poly| crate::polys::polyroots::roots(&poly, std::slice::from_ref(&m), true, true, None))
        {
            Ok(found) => found
                .iter()
                .filter(|(r, _)| is_expr(r))
                .map(|(r, k)| (r.clone(), *k))
                .collect(),
            Err(e) if failure(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        if found.iter().map(|(_, k)| *k).sum::<usize>() == n {
            return Ok(Some(root_basis(&sorted_roots(found), x, &log(x)?)?));
        }
    }
    if n == 1 {
        let minus: Basic = neg(&p[0])?;
        return Ok(Some(vec![match closed_integral(&minus, x)? {
            None => exp(&crate::integrals::integral::integral(&minus, x, None)?)?,
            Some(exponent) => simplified_exp(&exponent)?,
        }]));
    }
    let k: usize = p.iter().take_while(|c| **c == Basic::zero()).count();
    if k > 0 {
        let Some(reduced) = linear_basis(&p[k..], x, general)? else {
            return Ok(None);
        };
        let mut basis: Vec<Basic> = (0..k)
            .map(|j| {
                if j != 0 {
                    pow(x, &int(j as i64))
                } else {
                    Ok(Basic::one())
                }
            })
            .collect::<PolyResult<_>>()?;
        for v in reduced {
            let mut w: Basic = v;
            for _ in 0..k {
                let Some(integral) = closed_integral(&w, x)? else {
                    return Ok(None);
                };
                w = integral;
            }
            basis.push(w);
        }
        return Ok(Some(basis));
    }
    if !general {
        return Ok(None);
    }
    let y: Basic =
        std::sync::Arc::new(crate::core::basic::UndefinedFunction::new("_y", None, &[])).apply(vec![x.clone()])?;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in p.iter().enumerate() {
        terms.push(mul(&[c.clone(), diff(&y, x, i as u64)?])?);
    }
    let equation: Basic = sum(&terms)?;
    let found: Option<Vec<Basic>> =
        match attempt(|| crate::solvers::linear_ode::dsolve_linear(&equation, &y, true, false, false)) {
            Ok(found) => found,
            Err(e) if failure(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
    Ok(found.filter(|b| b.len() == n))
}

/// `exp(exponent)` with the logarithms of the exponent taken out
/// (`_simplified_exp`).
fn simplified_exp(exponent: &Basic) -> R {
    let value: Basic = exp(exponent)?;
    match crate::simplify::powsimp::powsimp_default(&value) {
        Ok(v) => Ok(v),
        Err(e) if failure(&e) => Ok(value),
        Err(e) => Err(e),
    }
}

/// The general solution of a linear equation (`_linear`).
fn linear(coefficients: &[Basic], b: &Basic, f: &Basic, taken: &BTreeSet<Basic>) -> PolyResult<Option<Vec<Basic>>> {
    let x: Basic = variable(f)?;
    let n: usize = coefficients.len() - 1;
    let Some(found_basis) = linear_basis(coefficients, &x, true)?.filter(|v| v.len() == n) else {
        return Ok(None);
    };
    // a basis is defined up to constant factors: the numbers are left out
    let basis: Vec<Basic> = found_basis
        .iter()
        .map(|b| without_number(b, &x))
        .collect::<PolyResult<_>>()?;
    let mut particular: Basic = Basic::zero();
    if *b != Basic::zero() {
        let l = crate::solvers::linear_ode::LinearOperator::new(coefficients, &x);
        let found: Option<Basic> =
            match attempt(|| crate::solvers::factorization::variation_of_parameters(&l, &basis, b)) {
                Ok(found) => found,
                Err(e) if failure(&e) => return Ok(None),
                Err(e) => return Err(e),
            };
        let Some(found) = found else {
            return Ok(None);
        };
        particular = reduced_particular(&tidy(&found)?, &basis, &x)?;
    }
    let cs: Vec<Basic> = constants(taken, n);
    let mut parts: Vec<Basic> = vec![particular.clone()];
    for (c, s) in cs.iter().zip(&basis) {
        parts.push(mul(&[c.clone(), s.clone()])?);
    }
    let mut general: Basic = add(&parts)?;
    if n == 1 && particular != Basic::zero() {
        // SymPy's form of the first order (C1 + Integral(Q/y1)) y1 when shorter
        let quotient: Basic = tidy(&div(&particular, &basis[0])?)?;
        let factored: Basic = mul(&[add(&[cs[0].clone(), quotient])?, basis[0].clone()])?;
        if ops(&factored)? < ops(&general)? {
            general = factored;
        }
    }
    Ok(Some(vec![eq(f, &general)]))
}

fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

/// The particular solution without its terms which are constant multiples
/// of solutions of the basis (`_reduced_particular`).
fn reduced_particular(particular: &Basic, basis: &[Basic], x: &Basic) -> R {
    let terms: Vec<Basic> = terms_of(&expand(particular)?);
    let mut kept: Vec<Basic> = Vec::new();
    for term in &terms {
        let mut absorbed: bool = false;
        for b in basis {
            if *b != Basic::zero() && !normal(&div(term, b)?)?.has(x) {
                absorbed = true;
                break;
            }
        }
        if !absorbed {
            kept.push(term.clone());
        }
    }
    if kept.len() == terms.len() {
        return Ok(particular.clone());
    }
    sum(&kept)
}

/// `b` without a constant factor which is a nonzero number
/// (`_without_number`).
fn without_number(b: &Basic, _x: &Basic) -> R {
    if !b.is_mul() {
        return Ok(b.clone());
    }
    let numbers: usize = b
        .args()
        .iter()
        .filter(|f| is_expr(f) && f.free_symbols().is_empty())
        .count();
    if numbers == 0 {
        return Ok(b.clone());
    }
    let rest: Vec<Basic> = b
        .args()
        .iter()
        .filter(|f| is_expr(f) && !f.free_symbols().is_empty())
        .cloned()
        .collect();
    if rest.is_empty() {
        Ok(b.clone())
    } else {
        mul(&rest)
    }
}

/// A particular solution in a short form (`_tidy`).
fn tidy(e: &Basic) -> R {
    let options = crate::simplify::simplify::Options::default();
    Ok(attempt(|| crate::simplify::simplify::simplify(e, &options))?.unwrap_or_else(|| e.clone()))
}

// ---------------------------------------------------------------------------
// first order equations

/// The `y` with `g(y) = value` by elementary inversions (`_invert`).
fn invert(g: &Basic, y: &Basic, value: &Basic) -> PolyResult<Vec<Basic>> {
    if g == y {
        return Ok(vec![value.clone()]);
    }
    if !g.has(y) {
        return Ok(Vec::new());
    }
    if g.is_add() {
        let (independent, dependent) = crate::core::expr::as_independent(g, y, true)?;
        if independent != Basic::zero() {
            return invert(&dependent, y, &sub(value, &independent)?);
        }
    } else if g.is_mul() {
        let (independent, dependent) = crate::core::expr::as_independent(g, y, false)?;
        if independent != Basic::one() {
            return invert(&dependent, y, &div(value, &independent)?);
        }
    } else if g.is_pow() {
        let (base, exponent) = (&g.args()[0], &g.args()[1]);
        if !exponent.has(y) {
            if *exponent == int(2) {
                let mut found: Vec<Basic> = invert(base, y, &sqrt(value)?)?;
                found.extend(invert(base, y, &neg(&sqrt(value)?)?)?);
                return Ok(found);
            }
            return invert(base, y, &pow(value, &pow(exponent, &int(-1))?)?);
        }
        if !base.has(y) {
            return invert(exponent, y, &div(&log(value)?, &log(base)?)?);
        }
    } else if g.is_function(Function::Exp) {
        let argument: &Basic = &g.args()[0];
        if crate::core::printing::could_extract_minus_sign(argument) {
            // SymPy's form: exp(-y) = v gives y = log(1/v)
            return invert(&neg(argument)?, y, &log(&normal(&pow(value, &int(-1))?)?)?);
        }
        return invert(argument, y, &log(value)?);
    } else if g.is_function(Function::Log) {
        return invert(&g.args()[0], y, &exp(value)?);
    }
    // a polynomial of degree two in y
    let terms: BTreeMap<i64, Basic> = match poly_coefficients(&sub(g, value)?, y) {
        Ok(t) => t,
        Err(PolyError::Value(_)) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    if terms.keys().next_back() == Some(&2) && terms.keys().next().is_some_and(|k| *k >= 0) {
        let get = |k: i64| terms.get(&k).cloned().unwrap_or_else(Basic::zero);
        let (a, b, c) = (get(2), get(1), get(0));
        let d: Basic = sqrt(&sub(&pow(&b, &int(2))?, &mul(&[int(4), a.clone(), c])?)?)?;
        let two_a: Basic = mul(&[int(2), a])?;
        return Ok(vec![
            normal(&div(&add(&[neg(&b)?, d.clone()])?, &two_a)?)?,
            normal(&div(&sub(&neg(&b)?, &d)?, &two_a)?)?,
        ]);
    }
    Ok(Vec::new())
}

/// `lhs(u) = rhs` solved for `u = f` when inverted, else the implicit
/// relation in `f` (`_implicit_or_explicit`).
fn implicit_or_explicit(lhs: &Basic, rhs: &Basic, u: &Basic, f: &Basic) -> PolyResult<Vec<Basic>> {
    let values: Vec<Basic> = invert(lhs, u, rhs)?;
    let rule: Vec<(Basic, Basic)> = vec![(u.clone(), f.clone())];
    if !values.is_empty() {
        return values.iter().map(|v| Ok(eq(f, &replaced(v, &rule)?))).collect();
    }
    Ok(vec![eq(&replaced(lhs, &rule)?, &replaced(rhs, &rule)?)])
}

/// `(g(x), h(u))` with `R = g h`, `None` when `R` does not separate
/// (`_separated`).
fn separated(r: &Basic, x: &Basic, u: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    if !r.has(u) {
        return Ok(Some((r.clone(), Basic::one())));
    }
    let mut candidates: Vec<Basic> = vec![r.clone()];
    match crate::polys::polytools::factor(r, &[], false, true) {
        Ok(f) => candidates.push(f),
        Err(e) if failure(&e) => {}
        Err(e) => return Err(e),
    }
    for candidate in candidates {
        let mut g_parts: Vec<Basic> = Vec::new();
        let mut h_parts: Vec<Basic> = Vec::new();
        let mut separates: bool = true;
        for factor in factors_of(&candidate) {
            if !factor.has(u) {
                g_parts.push(factor);
            } else if !factor.has(x) {
                h_parts.push(factor);
            } else {
                separates = false;
                break;
            }
        }
        if separates {
            let g: Basic = if g_parts.is_empty() {
                Basic::one()
            } else {
                mul(&g_parts)?
            };
            let h: Basic = if h_parts.is_empty() {
                Basic::one()
            } else {
                mul(&h_parts)?
            };
            return Ok(Some((g, h)));
        }
    }
    Ok(None)
}

type Method = fn(&Basic, &Basic, &Basic, &Basic, &Basic) -> PolyResult<Option<Vec<Basic>>>;

fn separable(r: &Basic, x: &Basic, u: &Basic, f: &Basic, c: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let Some((g, h)) = separated(r, x, u)? else {
        return Ok(None);
    };
    let left: Option<Basic> = closed_integral(&normal(&pow(&h, &int(-1))?)?, u)?;
    let right: Option<Basic> = closed_integral(&g, x)?;
    let (Some(left), Some(right)) = (left, right) else {
        return Ok(None);
    };
    Ok(Some(implicit_or_explicit(&left, &add(&[right, c.clone()])?, u, f)?))
}

/// `y' = P y + Q y**n` (`_bernoulli`).
fn bernoulli(r: &Basic, x: &Basic, u: &Basic, f: &Basic, c: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let mut p: Basic = Basic::zero();
    let mut q: Basic = Basic::zero();
    let mut n: Option<Basic> = None;
    for term in terms_of(&expand(r)?) {
        let (coefficient, rest) = crate::core::expr::as_independent(&term, u, false)?;
        if rest == *u {
            p = add(&[p, coefficient])?;
            continue;
        }
        if rest == Basic::one() {
            return Ok(None);
        }
        if rest.is_pow() && rest.args()[0] == *u && !rest.args()[1].has(x) && !rest.args()[1].has(u) {
            let exponent: Basic = rest.args()[1].clone();
            let same: bool = match &n {
                None => true,
                Some(m) => expand(&sub(m, &exponent)?)? == Basic::zero(),
            };
            if same {
                n = Some(exponent);
                q = add(&[q, coefficient])?;
                continue;
            }
        }
        return Ok(None);
    }
    let Some(n) = n else {
        return Ok(None);
    };
    if q == Basic::zero() || n == Basic::one() || n == Basic::zero() {
        return Ok(None);
    }
    let k: Basic = sub(&int(1), &n)?;
    let Some(integrating) = closed_integral(&neg(&mul(&[k.clone(), p])?)?, x)? else {
        return Ok(None);
    };
    let mu: Basic = simplified_exp(&integrating)?;
    let Some(quadrature) = closed_integral(&normal(&mul(&[k.clone(), q, mu.clone()])?)?, x)? else {
        return Ok(None);
    };
    let v: Basic = div(&add(&[quadrature, c.clone()])?, &mu)?;
    Ok(Some(vec![eq(f, &pow(&v, &pow(&k, &int(-1))?)?)]))
}

/// `M + N y' = 0` with `M_y = N_x` (`_exact`).
fn exact(m: &Basic, n: &Basic, x: &Basic, u: &Basic, f: &Basic, c: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    if normal(&sub(&diff(m, u, 1)?, &diff(n, x, 1)?)?)? != Basic::zero() {
        return Ok(None);
    }
    let Some(first) = closed_integral(m, x)? else {
        return Ok(None);
    };
    let rest: Basic = normal(&sub(n, &diff(&first, u, 1)?)?)?;
    if rest.has(x) {
        return Ok(None);
    }
    let second: Option<Basic> = if rest != Basic::zero() {
        closed_integral(&rest, u)?
    } else {
        Some(Basic::zero())
    };
    let Some(second) = second else {
        return Ok(None);
    };
    Ok(Some(implicit_or_explicit(&add(&[first, second])?, c, u, f)?))
}

/// `R(x, x w) = R(1, w)` (`_homogeneous`).
fn homogeneous(r: &Basic, x: &Basic, u: &Basic, f: &Basic, c: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let w: Basic = dummy("w");
    let scaled: Basic = normal(&replaced(r, &[(u.clone(), mul(&[x.clone(), w.clone()])?)])?)?;
    if scaled.has(x) {
        return Ok(None);
    }
    let denominator: Basic = normal(&sub(&scaled, &w)?)?;
    if denominator == Basic::zero() {
        return Ok(None);
    }
    let Some(left) = closed_integral(&normal(&pow(&denominator, &int(-1))?)?, &w)? else {
        return Ok(None);
    };
    let relation: Basic = sub(&replaced(&left, &[(w, div(u, x)?)])?, &log(x)?)?;
    Ok(Some(implicit_or_explicit(&relation, c, u, f)?))
}

fn soft<T>(found: PolyResult<Option<T>>) -> PolyResult<Option<T>> {
    match found {
        Err(e) if failure(&e) => Ok(None),
        other => other,
    }
}

fn first_order(e: &Basic, f: &Basic, taken: &BTreeSet<Basic>) -> PolyResult<Option<Vec<Basic>>> {
    let x: Basic = variable(f)?;
    let (u, p) = (dummy("y"), dummy("p"));
    let derivative: Basic = crate::core::function::derivative_node(f, &[(x.clone(), 1)]);
    let replaced_e: Basic = replaced(&replaced(e, &[(derivative, p.clone())])?, &[(f.clone(), u.clone())])?;
    if replaced_e.has(f)
        || replaced_e
            .preorder()
            .iter()
            .any(|n| matches!(n.head(), Head::Derivative))
    {
        return Ok(None);
    }
    let terms: BTreeMap<i64, Basic> = match poly_coefficients(&replaced_e, &p) {
        Ok(t) => t,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(err) => return Err(err),
    };
    if terms.keys().any(|k| *k != 0 && *k != 1) || !terms.contains_key(&1) {
        return Ok(None);
    }
    let (n, m) = (terms[&1].clone(), terms.get(&0).cloned().unwrap_or_else(Basic::zero));
    let c: Basic = constants(taken, 1).remove(0);
    let r: Basic = normal(&neg(&div(&m, &n)?)?)?;
    let methods: [Method; 2] = [separable, bernoulli];
    for method in methods {
        if let Some(found) = soft(method(&r, &x, &u, f, &c))? {
            return Ok(Some(found));
        }
    }
    if let Some(found) = soft(exact(&m, &n, &x, &u, f, &c))? {
        return Ok(Some(found));
    }
    soft(homogeneous(&r, &x, &u, f, &c))
}

// ---------------------------------------------------------------------------
// the dispatcher

/// The factors of `e` with `f` in them, when `e` is a product of several
/// (`_factors`).
fn factors(e: &Basic, f: &Basic) -> PolyResult<Vec<Basic>> {
    let x: Basic = variable(f)?;
    let n: u64 = order(e, f)?;
    let jets: Vec<Basic> = (0..=n).map(|i| dummy(&format!("y{}", i))).collect();
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for i in (1..=n).rev() {
        rule.push((
            crate::core::function::derivative_node(f, &[(x.clone(), i)]),
            jets[i as usize].clone(),
        ));
    }
    rule.push((f.clone(), jets[0].clone()));
    let replaced_e: Basic = replaced(e, &rule)?;
    let factored: Basic = match crate::polys::polytools::factor(&replaced_e, &[], false, true) {
        Ok(fac) => fac,
        Err(err) if failure(&err) => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    if !factored.is_mul() {
        return Ok(Vec::new());
    }
    let back: Vec<(Basic, Basic)> = (0..=n)
        .map(|i| Ok((jets[i as usize].clone(), diff(f, &x, i)?)))
        .collect::<PolyResult<_>>()?;
    let mut found: Vec<Basic> = Vec::new();
    for g in factored.args() {
        if jets.iter().any(|j| g.has(j)) {
            found.push(replaced(g, &back)?);
        }
    }
    Ok(if found.len() > 1 { found } else { Vec::new() })
}

/// The right hand side of the explicit general solution of [`dsolve`] and
/// its constants, `None` when it does not give one explicit solution
/// (`explicit_solution`).
pub fn explicit_solution(equation: &Basic, f: &Basic) -> PolyResult<Option<(Basic, Vec<Basic>)>> {
    let found: Vec<Basic> = dsolve(equation, f)?;
    if found.len() != 1 || found[0].args()[0] != *f {
        return Ok(None);
    }
    let rhs: Basic = found[0].args()[1].clone();
    if !is_expr(&rhs) {
        return Ok(None);
    }
    let taken: BTreeSet<String> = symbols_of(&as_zero(equation)?).iter().map(|s| s.to_string()).collect();
    let mut cs: Vec<(usize, Basic)> = Vec::new();
    for s in symbols_of(&rhs) {
        let name: String = s.to_string();
        if taken.contains(&name) {
            continue;
        }
        if let Some(digits) = name.strip_prefix('C') {
            if !digits.is_empty() && digits.chars().all(|ch| ch.is_ascii_digit()) {
                cs.push((digits.parse().unwrap_or(0), s));
            }
        }
    }
    cs.sort_by_key(|(k, _)| *k);
    Ok(Some((rhs, cs.into_iter().map(|(_, s)| s).collect())))
}

/// Independent solutions of a homogeneous linear equation by the methods
/// which need no algorithm for rational coefficients (`solution_parts`).
pub fn solution_parts(equation: &Basic, f: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let Some((coefficients, b)) = linear_coefficients(equation, f)? else {
        return Ok(None);
    };
    if b != Basic::zero() {
        return Ok(None);
    }
    let Some(basis) = linear_basis(&coefficients, &variable(f)?, false)? else {
        return Ok(None);
    };
    let parts: Vec<Basic> = basis
        .into_iter()
        .filter(|b| !crate::solvers::linear_ode::has_integral(b))
        .collect();
    Ok(if parts.is_empty() { None } else { Some(parts) })
}

/// The general solution of an ordinary differential equation in `f` by the
/// classical methods, as a list of equations; `[]` when none applies
/// (`dsolve`).
pub fn dsolve(equation: &Basic, f: &Basic) -> PolyResult<Vec<Basic>> {
    let e: Basic = as_zero(equation)?;
    variable(f)?;
    let n: u64 = order(&e, f)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let taken: BTreeSet<Basic> = symbols_of(&e);
    if let Some((coefficients, b)) = linear_coefficients(&e, f)? {
        return Ok(linear(&coefficients, &b, f, &taken)?.unwrap_or_default());
    }
    let found_factors: Vec<Basic> = factors(&e, f)?;
    if !found_factors.is_empty() {
        let mut solutions: Vec<Basic> = Vec::new();
        for g in &found_factors {
            if order(g, f)? == 0 {
                continue;
            }
            for s in dsolve(g, f)? {
                if !solutions.contains(&s) {
                    solutions.push(s);
                }
            }
        }
        if !solutions.is_empty() {
            return Ok(solutions);
        }
    }
    if n == 1 {
        return Ok(first_order(&e, f, &taken)?.unwrap_or_default());
    }
    Ok(Vec::new())
}
