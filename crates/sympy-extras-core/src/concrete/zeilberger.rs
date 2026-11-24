//! Port of `sympy_extras/concrete/zeilberger.py`: Zeilberger's algorithm
//! (creative telescoping) for definite sums of hypergeometric terms, the
//! Wilf–Zeilberger certificates and proofs, and the closed forms of
//! definite sums through the recurrence of the algorithm solved by
//! [`crate::solvers::recurr::rsolve_linear`].

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::concrete::gosper::{gosper_normal, UPoly};
use crate::concrete::karr::SumPoint;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp, UndefinedFunction};
use crate::core::expr::{is, polynomial_like, PolynomialKind};
use crate::error::{PolyError, PolyResult, NOT_PORTED_PREFIX};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::rationalfield::QQ;
use crate::polys::domains::sparsefractionfield::SparseFractionField;
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;

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
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn factor(e: &Basic) -> R {
    crate::polys::polytools::factor(e, &[], false, true)
}
/// `combsimp(e, generic=True)`: the identity of meromorphic functions the
/// algorithms on hypergeometric terms need.
fn combsimp(e: &Basic) -> R {
    crate::simplify::gammasimp::combsimp_generic(e)
}
/// `gammasimp(e, generic=True)`.
fn gammasimp(e: &Basic) -> R {
    crate::simplify::gammasimp::gammasimp_generic(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn lcm_list(seq: &[Basic]) -> R {
    crate::polys::polytools::lcm_list(seq, &[])
}
fn gcd_list(seq: &[Basic]) -> R {
    crate::polys::polytools::gcd_list(seq, &[])
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn limit(e: &Basic, z: &Basic, z0: &Basic) -> R {
    crate::core::limits::limit(e, z, z0, "+")
}
fn rewrite(e: &Basic, target: &str) -> R {
    crate::core::rewriting::rewrite(e, &[], target, &crate::core::rewriting::RewriteHints::default())
}
fn is_rational_function(e: &Basic, k: &Basic) -> bool {
    polynomial_like(e, std::slice::from_ref(k), PolynomialKind::Rational) == Some(true)
}
fn is_polynomial(e: &Basic, k: &Basic) -> bool {
    polynomial_like(e, std::slice::from_ref(k), PolynomialKind::Polynomial) == Some(true)
}
fn integer_of(e: &Basic) -> Option<i64> {
    if !matches!(e.head(), Head::Rational { value, .. } if value.is_int()) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
}
fn tuple(v: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, v)
}
fn has_nan_or_zoo(e: &Basic) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity))
}
fn has_sum_or_product(e: &Basic) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Sum | Head::Product))
}
fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b) && !crate::sets::sets::is_set(b) && !matches!(b.head(), Head::Tuple)
}
/// `to_expr`: `TypeError` for an object which is no expression.
fn to_expr(e: &Basic) -> R {
    if !is_expr(e) {
        return Err(PolyError::Type(format!("an expression is expected, got {}", e)));
    }
    Ok(e.clone())
}

/// The function `S` of the recurrences.
fn s_function() -> Arc<UndefinedFunction> {
    Arc::new(UndefinedFunction::new("S", None, &[]))
}
fn dummy(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    crate::core::symbol::Dummy::new(name, assumptions).0
}

/// The errors of the native algebra on input it does not handle
/// (`_Failures`: `ValueError`, `TypeError`, `NotImplementedError`,
/// `AttributeError`, `BasePolynomialError`); a part of the core which is not
/// ported is no such failure.
fn failure(e: &PolyError) -> bool {
    match e {
        PolyError::Value(_) | PolyError::Type(_) | PolyError::Polys(..) => true,
        PolyError::NotImplemented(m) => !m.starts_with(NOT_PORTED_PREFIX),
        _ => false,
    }
}

/// The output of Zeilberger's algorithm for a term `F(n, k)`
/// (`Telescoper`): the polynomials `a_0(n), ..., a_J(n)` of the recurrence
/// and the rational function `R(n, k)` with `G = R*F`.
#[derive(Clone, Debug)]
pub struct Telescoper {
    pub term: Basic,
    pub n: Basic,
    pub k: Basic,
    pub coefficients: Vec<Basic>,
    pub certificate: Basic,
}

impl Telescoper {
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// The recurrence `sum_j a_j(n) S(n + j)` satisfied by the sum over
    /// `k`, as an expression in the function `name(n)` (`recurrence`).
    pub fn recurrence(&self, name: &str) -> R {
        let f: Arc<UndefinedFunction> = Arc::new(UndefinedFunction::new(name, None, &[]));
        let mut terms: Vec<Basic> = Vec::new();
        for (j, a) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[
                a.clone(),
                f.apply(vec![add(&[self.n.clone(), int(j as i64)])?])?,
            ])?);
        }
        add(&terms)
    }

    /// Verify `sum_j a_j F(n + j, k) = G(n, k + 1) - G(n, k)` by
    /// simplification (`check`).
    pub fn check(&self) -> PolyResult<bool> {
        let (n, k, f) = (&self.n, &self.k, &self.term);
        let mut terms: Vec<Basic> = Vec::new();
        for (j, a) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[a.clone(), subs(f, n, &add(&[n.clone(), int(j as i64)])?)?])?);
        }
        let lhs: Basic = add(&terms)?;
        let g: Basic = mul(&[self.certificate.clone(), f.clone()])?;
        let shifted: Basic = subs(&g, k, &add(&[k.clone(), int(1)])?)?;
        let difference: Basic = combsimp(&div(&add(&[lhs, neg(&shifted)?, g])?, f)?)?;
        Ok(simplify(&difference)? == Basic::zero())
    }
}

/// `f/g` as a rational function of `k`, or `None` (`_rational_ratio`).
fn rational_ratio(f: &Basic, g: &Basic, k: &Basic) -> R2 {
    let mut ratio: Basic = cancel(&combsimp(&div(f, g)?)?)?;
    if !is_rational_function(&ratio, k) {
        // a second attempt through the hypergeometric simplifier
        ratio = match rewrite(&ratio, "gamma")
            .and_then(|r| combsimp(&r))
            .and_then(|r| cancel(&r))
        {
            Ok(r) => r,
            Err(e) if failure(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        if !is_rational_function(&ratio, k) {
            return Ok(None);
        }
    }
    Ok(Some(ratio))
}

/// The bound of Gosper's algorithm on the degree of the polynomial
/// solution `f` of `A(k) f(k+1) - B(k-1) f(k) = c(k)`
/// (`_gosper_degree_bound`).
fn gosper_degree_bound(a: &UPoly, b_shifted: &UPoly, degree_c: i64) -> PolyResult<i64> {
    let (da, db) = (a.degree(), b_shifted.degree());
    if da != db || cancel(&sub(&a.lc(), &b_shifted.lc())?)? != Basic::zero() {
        return Ok(degree_c - da.max(db));
    }
    let m: i64 = da;
    let mut candidates: Vec<i64> = vec![degree_c - m + 1];
    if m >= 1 {
        let d0: Basic = cancel(&div(&sub(&b_shifted.nth(m - 1), &a.nth(m - 1))?, &a.lc())?)?;
        if let Some(v) = integer_of(&d0).filter(|v| *v >= 0) {
            candidates.push(v);
        }
    }
    Ok(*candidates.iter().max().expect("nonempty"))
}

/// Zeilberger's algorithm: a recurrence in `n` for the sum over `k` of a
/// term hypergeometric in `n` and `k`, with its certificate; the orders
/// `min_order` to `max_order` tried in turn (`zeilberger`).
pub fn zeilberger(
    term: &Basic,
    n: &Basic,
    k: &Basic,
    max_order: usize,
    min_order: usize,
) -> PolyResult<Option<Telescoper>> {
    let f: Basic = to_expr(term)?;
    let Some(r) = rational_ratio(&subs(&f, k, &add(&[k.clone(), int(1)])?)?, &f, k)? else {
        return Err(PolyError::Value(format!("{} is not hypergeometric in {}", f, k)));
    };
    let mut ratios: Vec<Basic> = Vec::new();
    for j in 0..=max_order {
        let Some(s) = rational_ratio(&subs(&f, n, &add(&[n.clone(), int(j as i64)])?)?, &f, k)? else {
            return Err(PolyError::Value(format!("{} is not hypergeometric in {}", f, n)));
        };
        ratios.push(s);
    }
    for order in min_order..=max_order {
        if let Some(result) = telescope(&f, n, k, &r, &ratios[..order + 1])? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

fn upoly(e: &Basic, k: &Basic) -> PolyResult<Option<UPoly>> {
    match UPoly::of(e, k) {
        Ok(p) => Ok(Some(p)),
        Err(PolyError::Value(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn telescope(f: &Basic, n: &Basic, k: &Basic, r: &Basic, ratios: &[Basic]) -> PolyResult<Option<Telescoper>> {
    let order: usize = ratios.len() - 1;
    let a: Vec<Basic> = (0..=order).map(|j| dummy(&format!("a{}", j), &[])).collect();
    // sum_j a_j F(n+j, k) = F(n, k) P(k)/D(k)
    let parts: Vec<(Basic, Basic)> = ratios.iter().map(numer_denom).collect::<PolyResult<_>>()?;
    let denominators: Vec<Basic> = parts.iter().map(|p| p.1.clone()).collect();
    let d: Basic = lcm_list(&denominators)?;
    let mut terms: Vec<Basic> = Vec::new();
    for j in 0..=order {
        terms.push(mul(&[
            a[j].clone(),
            cancel(&div(&mul(&[parts[j].0.clone(), d.clone()])?, &parts[j].1)?)?,
        ])?);
    }
    let p: Basic = add(&terms)?;
    let Some(p_poly) = upoly(&p, k)? else {
        return Ok(None);
    };
    // the ratio of the term without P: r(k) D(k)/D(k+1)
    let rho: Basic = cancel(&div(
        &mul(&[r.clone(), d.clone()])?,
        &subs(&d, k, &add(&[k.clone(), int(1)])?)?,
    )?)?;
    let (num, den) = numer_denom(&rho)?;
    let (ga, gb, gc) = match gosper_normal(&num, &den, k) {
        Ok(normal) => normal,
        Err(e) if failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let b_shift: UPoly = gb.shift(&int(-1))?;
    let degree_c: i64 = gc.degree() + p_poly.degree();
    let bound: i64 = gosper_degree_bound(&ga, &b_shift, degree_c)?;
    // d < 0 leaves f = 0: the shifts of the term in n may still be
    // linearly dependent over K(n)
    let f_coefficients: Vec<Basic> = (0..bound.max(-1) + 1).map(|i| dummy(&format!("f{}", i), &[])).collect();
    let polynomial = |cs: &[Basic]| -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in cs.iter().enumerate() {
            terms.push(mul(&[c.clone(), pow(k, &int(i as i64))?])?);
        }
        add(&terms)
    };
    let fk: Basic = polynomial(&f_coefficients)?;
    let equation: Basic = add(&[
        mul(&[ga.as_expr(k)?, subs(&fk, k, &add(&[k.clone(), int(1)])?)?])?,
        neg(&mul(&[b_shift.as_expr(k)?, fk.clone()])?)?,
        neg(&mul(&[gc.as_expr(k)?, p.clone()])?)?,
    ])?;
    let mut unknowns: Vec<Basic> = f_coefficients.clone();
    unknowns.extend(a.iter().cloned());
    let Some(rows) = upoly(&equation, k)? else {
        return Ok(None);
    };
    let matrix: Vec<Vec<Basic>> = rows
        .c
        .iter()
        .map(|row| {
            unknowns
                .iter()
                .map(|u| cancel(&crate::core::function::diff(row, &[(u.clone(), 1)])?))
                .collect::<PolyResult<Vec<Basic>>>()
        })
        .collect::<PolyResult<_>>()?;
    let nf: usize = f_coefficients.len();
    for vector in nullspace(&matrix, unknowns.len())? {
        let cancelled: Vec<Basic> = vector.iter().map(cancel).collect::<PolyResult<_>>()?;
        let values: Vec<Basic> = normalized(&cancelled, nf, n)?;
        let coefficients: Vec<Basic> = values[nf..].to_vec();
        if coefficients.iter().all(|c| *c == Basic::zero()) {
            continue;
        }
        let f_value: Basic = polynomial(&values[..nf])?;
        let certificate: Basic = factor(&cancel(&div(
            &mul(&[b_shift.as_expr(k)?, f_value])?,
            &mul(&[gc.as_expr(k)?, d.clone()])?,
        )?)?)?;
        return Ok(Some(Telescoper {
            term: f.clone(),
            n: n.clone(),
            k: k.clone(),
            coefficients: coefficients.iter().map(factor).collect::<PolyResult<_>>()?,
            certificate,
        }));
    }
    Ok(None)
}

/// The basis of the null space of a matrix of rational functions read from
/// its reduced row echelon form, the free coordinate one, over the
/// rational functions of its symbols; over the expressions when an entry
/// is not a rational function (`_nullspace`).
fn nullspace(rows: &[Vec<Basic>], columns: usize) -> PolyResult<Vec<Vec<Basic>>> {
    let mut symbols: Vec<Basic> = Vec::new();
    for row in rows {
        for e in row {
            for s in e.free_symbols() {
                if !symbols.contains(&s) {
                    symbols.push(s);
                }
            }
        }
    }
    symbols.sort_by_cached_key(crate::core::sorting::sort_key);
    let mut names: Vec<String> = (0..symbols.len()).map(|i| format!("_zn{}", i)).collect();
    if names.is_empty() {
        names.push("_zn".to_string());
    }
    let symbol = |name: &str| crate::core::symbol::Symbol::new(name, &[]).0;
    let renamed: HashMap<Basic, Basic> = symbols
        .iter()
        .zip(&names)
        .map(|(s, name)| (s.clone(), symbol(name)))
        .collect();
    let restored: HashMap<Basic, Basic> = symbols
        .iter()
        .zip(&names)
        .map(|(s, name)| (symbol(name), s.clone()))
        .collect();
    let field = SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names));
    let mut converted = Vec::new();
    for row in rows {
        let mut line = Vec::new();
        for e in row {
            match field.from_expr(&e.xreplace(&renamed)) {
                Ok(v) => line.push(v),
                Err(err) if failure(&err) => return expression_nullspace(rows, columns),
                Err(err) => return Err(err),
            }
        }
        converted.push(line);
    }
    field
        .nullspace(&converted, columns)?
        .iter()
        .map(|vector| {
            vector
                .iter()
                .map(|v| Ok(field.to_expr(v)?.xreplace(&restored)))
                .collect()
        })
        .collect()
}

/// [`nullspace`] over the expressions (`_expression_nullspace`).
fn expression_nullspace(rows: &[Vec<Basic>], columns: usize) -> PolyResult<Vec<Vec<Basic>>> {
    let (reduced, pivots) = crate::solvers::recurr::row_reduce(rows)?;
    let mut basis: Vec<Vec<Basic>> = Vec::new();
    for free in 0..columns {
        if pivots.contains(&free) {
            continue;
        }
        let mut vector: Vec<Basic> = vec![Basic::zero(); columns];
        vector[free] = Basic::one();
        for (row, pivot) in pivots.iter().enumerate() {
            vector[*pivot] = neg(&reduced[row][free])?;
        }
        basis.push(vector);
    }
    Ok(basis)
}

/// The solution scaled so that the recurrence coefficients are polynomials
/// in `n` without a common factor (`_normalized`).
fn normalized(values: &[Basic], nf: usize, n: &Basic) -> PolyResult<Vec<Basic>> {
    let mut denominators: Vec<Basic> = Vec::new();
    for c in &values[nf..] {
        if *c != Basic::zero() {
            denominators.push(numer_denom(c)?.1);
        }
    }
    if denominators.is_empty() {
        return Ok(values.to_vec());
    }
    let scale: Basic = lcm_list(&denominators)?;
    let mut scaled: Vec<Basic> = values
        .iter()
        .map(|v| cancel(&mul(&[v.clone(), scale.clone()])?))
        .collect::<PolyResult<_>>()?;
    let numerators: Vec<Basic> = scaled[nf..].iter().filter(|c| **c != Basic::zero()).cloned().collect();
    if !numerators.is_empty() && numerators.iter().all(|c| is_polynomial(c, n)) {
        let common: Basic = gcd_list(&numerators)?;
        if common != Basic::zero() && common != Basic::one() {
            scaled = scaled
                .iter()
                .map(|v| cancel(&div(v, &common)?))
                .collect::<PolyResult<_>>()?;
        }
    }
    Ok(scaled)
}

/// The Wilf–Zeilberger certificate `R(n, k)` proving
/// `sum_k term(n, k) == closed_form(n)`; `None` when there is no such
/// rational function (`wz_certificate`).
pub fn wz_certificate(term: &Basic, closed_form: &Basic, n: &Basic, k: &Basic) -> R2 {
    let f: Basic = div(&to_expr(term)?, &to_expr(closed_form)?)?;
    let Some(z) = zeilberger(&f, n, k, 1, 1)? else {
        return Ok(None);
    };
    let (a0, a1) = (&z.coefficients[0], &z.coefficients[1]);
    if cancel(&div(a0, a1)?)? != int(-1) {
        return Ok(None);
    }
    Ok(Some(factor(&cancel(&div(&z.certificate, a1)?)?)?))
}

/// Prove `sum_k term(n, k) == closed_form(n)` for `n >= n0` with a WZ
/// certificate: `Some(true)` when a certificate exists and the identity
/// holds at `n0`, `None` when no certificate is found (`wz_prove`).
pub fn wz_prove(term: &Basic, closed_form: &Basic, n: &Basic, k: &Basic, n0: i64) -> PolyResult<Option<bool>> {
    let Some(r) = wz_certificate(term, closed_form, n, k)? else {
        return Ok(None);
    };
    let f: Basic = div(term, closed_form)?;
    let g: Basic = mul(&[r, f.clone()])?;
    let identity: Basic = simplify(&combsimp(&div(
        &add(&[
            subs(&f, n, &add(&[n.clone(), int(1)])?)?,
            neg(&f)?,
            neg(&subs(&g, k, &add(&[k.clone(), int(1)])?)?)?,
            g.clone(),
        ])?,
        &f,
    )?)?)?;
    if identity != Basic::zero() {
        return Ok(None);
    }
    // the base case, summed over the support
    let Some(base) = finite_sum(term, k, n, n0)? else {
        return Ok(None);
    };
    Ok(Some(
        simplify(&sub(&base, &subs(closed_form, n, &int(n0))?)?)? == Basic::zero(),
    ))
}

/// `sum_k term(value, k)` over the support of the term, found by expanding
/// from `k = 0` in both directions while the terms are nonzero
/// (`_finite_sum`).
fn finite_sum(term: &Basic, k: &Basic, n: &Basic, value: i64) -> R2 {
    let f: Basic = subs(term, n, &int(value))?;
    let mut total: Basic = Basic::zero();
    for direction in [1i64, -1] {
        let mut i: i64 = if direction == 1 { 0 } else { -1 };
        let mut zeros: i64 = 0;
        while zeros < 3 && i.abs() < 400 {
            let t: Basic = subs(&f, k, &int(i))?;
            if t.has(k) || !crate::core::assumptions::is_number(&t) {
                return Ok(None);
            }
            if t == Basic::zero() {
                zeros += 1;
            } else {
                zeros = 0;
                total = add(&[total, t])?;
            }
            i += direction;
        }
    }
    Ok(Some(total))
}

/// A closed form of the definite sum `sum_{k=lo}^{hi} term` with `lo` an
/// integer and `hi` either `n` (plus an integer) or `oo`, through the
/// recurrence of Zeilberger's algorithm solved by `rsolve_linear`; the
/// recurrence itself (an equation in `S(n)`) when it finds no closed form;
/// `None` when there is no recurrence (`zeilberger_sum`).
pub fn zeilberger_sum(term: &Basic, limits: &[Basic], n: Option<&Basic>, max_order: usize, special_values: bool) -> R2 {
    if limits.len() != 3 {
        let shown: Vec<String> = limits.iter().map(|l| l.to_string()).collect();
        return Err(PolyError::Value(format!(
            "the limits of a sum are (k, lo, hi), not ({})",
            shown.join(", ")
        )));
    }
    let k: Basic = if limits[0].is_symbol() {
        limits[0].clone()
    } else {
        crate::core::symbol::Symbol::new(&limits[0].to_string(), &[]).0
    };
    let (lo, hi) = (to_expr(&limits[1])?, to_expr(&limits[2])?);
    let f: Basic = to_expr(term)?;
    let n: Basic = match n {
        Some(n) => n.clone(),
        None => {
            let mut candidates: Vec<Basic> = f.free_symbols().into_iter().filter(|s| *s != k).collect();
            candidates.sort_by_key(|s| s.to_string());
            let mut inside: Vec<Basic> = candidates.iter().filter(|s| hi.has(s)).cloned().collect();
            if inside.is_empty() {
                // the upper limit itself when it is a symbol
                inside = if hi.is_symbol() && hi != k {
                    vec![hi.clone()]
                } else {
                    candidates
                };
            }
            if inside.len() != 1 {
                return Err(PolyError::Value("the summation variable n must be given".to_string()));
            }
            inside[0].clone()
        }
    };
    let solution: Option<Basic> = zeilberger_sum_cases_free(&f, &k, &lo, &hi, &n, max_order)?;
    let Some(solution) = solution else {
        return Ok(None);
    };
    if !special_values || !is_expr(&solution) {
        return Ok(Some(solution));
    }
    let at_point = |point: &SumPoint| -> R2 {
        let rule: HashMap<Basic, Basic> = point.iter().cloned().collect();
        sum_at_point(&f.xreplace(&rule), &k, &lo, &hi, &n, max_order)
    };
    Ok(Some(crate::concrete::karr::with_special_values_sum(
        &solution,
        &f,
        &[k.clone(), n.clone()],
        &[lo.clone(), hi.clone()],
        &at_point,
        true,
        None,
    )?))
}

/// The sum at a point of the parameters: by [`zeilberger_sum`], else by
/// Karr's summation when Zeilberger's algorithm gives no closed form there
/// (`_sum_at_point`).
fn sum_at_point(f: &Basic, k: &Basic, lo: &Basic, hi: &Basic, n: &Basic, max_order: usize) -> R2 {
    let limits: [Basic; 3] = [k.clone(), lo.clone(), hi.clone()];
    let found: Option<Basic> = match zeilberger_sum(f, &limits, Some(n), max_order, true) {
        Ok(found) => found,
        Err(PolyError::Value(_)) => None,
        Err(PolyError::NotImplemented(m)) if !m.starts_with(NOT_PORTED_PREFIX) => None,
        Err(e) => return Err(e),
    };
    if let Some(found) = found.filter(is_expr) {
        return Ok(Some(found));
    }
    Ok(Some(crate::concrete::karr::summation(
        f,
        &[(k.clone(), lo.clone(), hi.clone())],
        &[],
        true,
        true,
    )?))
}

/// `Sum(F, (k, lo, hi))` evaluated (left as a sum where it is not)
/// (`_direct_sum`).
fn direct_sum(f: &Basic, k: &Basic, lo: &Basic, hi: &Basic) -> R {
    let s: Basic = crate::concrete::expr_with_limits::sum_(f, &[tuple(vec![k.clone(), lo.clone(), hi.clone()])])?;
    crate::core::function::doit(&s)
}

/// [`zeilberger_sum`] without the cases of the parameters
/// (`_zeilberger_sum`).
fn zeilberger_sum_cases_free(f: &Basic, k: &Basic, lo: &Basic, hi: &Basic, n: &Basic, max_order: usize) -> R2 {
    let Some(z) = zeilberger(f, n, k, max_order, 0)? else {
        return Ok(None);
    };
    let order: usize = z.order();
    // G in a form which is finite where the term vanishes
    let g: Basic = gammasimp(&combsimp(&mul(&[z.certificate.clone(), f.clone()])?)?)?;
    let s_fun: Arc<UndefinedFunction> = s_function();
    let infinite: bool = matches!(hi.head(), Head::Infinity);
    let shifted_n = |j: usize| add(&[n.clone(), int(j as i64)]);
    let mut rhs: Basic;
    if infinite {
        let top: Basic = limit(&g, k, hi)?;
        if top.has(k) {
            return Ok(None);
        }
        rhs = sub(&top, &value(&g, k, lo)?)?;
    } else {
        let big_k: Basic = subs(hi, n, &shifted_n(order)?)?;
        rhs = sub(&value(&g, k, &add(&[big_k.clone(), int(1)])?)?, &value(&g, k, lo)?)?;
        for (j, a) in z.coefficients.iter().enumerate() {
            let upper: Basic = subs(hi, n, &shifted_n(j)?)?;
            let count: Basic = simplify(&sub(&big_k, &upper)?)?;
            let Some(count) = integer_of(&count).filter(|c| *c >= 0) else {
                return Ok(None);
            };
            for i in 1..=count {
                let at: Basic = subs(&subs(f, n, &shifted_n(j)?)?, k, &add(&[upper.clone(), int(i)])?)?;
                rhs = sub(&rhs, &mul(&[a.clone(), at])?)?;
            }
        }
    }
    let rhs: Basic = simplify(&combsimp(&rhs)?)?;
    let mut terms: Vec<Basic> = Vec::new();
    for (j, a) in z.coefficients.iter().enumerate() {
        terms.push(mul(&[a.clone(), s_fun.apply(vec![shifted_n(j)?])?])?);
    }
    let recurrence: Basic = sub(&add(&terms)?, &rhs)?;
    // initial values, from the first n at which the leading coefficient of
    // the recurrence does not vanish any more
    let mut initial: Vec<(Basic, Basic)> = Vec::new();
    let n0: i64 = start(z.coefficients.last().expect("nonempty"), n)?;
    for i in 0..order as i64 {
        let at: Basic = int(n0 + i);
        let v: Basic = direct_sum(&subs(f, n, &at)?, k, lo, &subs(hi, n, &at)?)?;
        if matches!(v.head(), Head::Sum) || (infinite && !crate::core::assumptions::is_number(&v)) {
            return Ok(recurrence_equation(&recurrence));
        }
        initial.push((s_fun.apply(vec![at])?, v));
    }
    let mut solution: Option<Basic> = None;
    if order == 0 {
        // the term is Gosper-summable: a_0 S(n) = rhs
        solution = Some(cancel(&div(&rhs, &z.coefficients[0])?)?);
    } else if order == 1 && rhs == Basic::zero() && !initial.is_empty() {
        // a first order homogeneous recurrence: a product of its ratio
        let (a0, a1) = (&z.coefficients[0], &z.coefficients[1]);
        let idx: Basic = dummy("i", &[(Predicate::integer, true)]);
        let ratio: Basic = subs(&cancel(&neg(&div(a0, a1)?)?)?, n, &idx)?;
        let prod: Basic = crate::concrete::products::product(&ratio, &(idx, int(n0), sub(n, &int(1))?))?;
        if !matches!(prod.head(), Head::Product) {
            solution = Some(mul(&[initial[0].1.clone(), prod])?);
        }
    }
    if solution.is_none() {
        // the hypergeometric and d'Alembertian solutions; a solution which
        // is itself a sum is no closed form
        let given: Option<&[(Basic, Basic)]> = if initial.is_empty() { None } else { Some(&initial) };
        let solved: Option<Basic> =
            match crate::solvers::recurr::rsolve_linear(&recurrence, &s_fun.apply(vec![n.clone()])?, given) {
                Ok(solved) => solved,
                Err(e) if failure(&e) => None,
                Err(e) => return Err(e),
            };
        if let Some(solved) = solved.filter(|s| !has_sum_or_product(s)) {
            solution = Some(solved);
        }
    }
    let Some(solution) = solution else {
        return Ok(recurrence_equation(&recurrence));
    };
    let solution: Basic = nicer(&solution, n)?;
    // the closed form is checked at a few values
    for v in n0..n0 + order as i64 + 3 {
        let at: Basic = int(v);
        let direct: Basic = direct_sum(&subs(f, n, &at)?, k, lo, &subs(hi, n, &at)?)?;
        if matches!(direct.head(), Head::Sum) {
            break;
        }
        if simplify(&sub(&direct, &subs(&solution, n, &at)?)?)? != Basic::zero() {
            // an order-zero recurrence a0 S(n) = rhs would only re-assert
            // the answer the check has just rejected (sympy-extras#33)
            return Ok(if order == 0 {
                None
            } else {
                recurrence_equation(&recurrence)
            });
        }
    }
    Ok(Some(solution))
}

fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

fn is_true(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

/// The closed form simplified, the gamma functions of half-integer
/// arguments removed with the duplication formula and factorials or
/// binomials preferred when smaller (`_nicer`).
fn nicer(solution: &Basic, n: &Basic) -> R {
    use crate::functions::special::gamma_functions::gamma;
    let half: Basic = Basic::half();
    let third: Basic = Basic::rational(crate::numbers::Int::ONE, crate::numbers::Int::from(3))?;
    let pi: Basic = Basic::atom(Head::Pi);
    let rule = |g: &Basic| -> R {
        let z: &Basic = &g.args()[0];
        if !is_expr(z) {
            return Ok(g.clone());
        }
        let shifted: Basic = sub(z, &half)?;
        let integral: bool = is_true(&shifted, Predicate::integer)
            || (is_polynomial(&shifted, n)
                && is_true(&constant_term(&shifted, n)?, Predicate::integer)
                && is(&shifted, Predicate::integer) != Some(false));
        if integral {
            let doubled: Basic = mul(&[int(2), shifted.clone()])?;
            return div(
                &mul(&[
                    gamma(&doubled)?,
                    pow(&pi, &half)?,
                    pow(&int(2), &sub(&int(1), &doubled)?)?,
                ])?,
                &gamma(&shifted)?,
            );
        }
        Ok(g.clone())
    };
    let duplicate = |e: &Basic| -> R {
        crate::core::function::replace_where(e, &|g: &Basic| g.is_function(Function::Gamma), &rule)
    };
    let triplicate = |e: &Basic| -> R {
        // gamma(z + 1/3) gamma(z + 2/3) = 2 pi 3**(1/2 - 3z) gamma(3z)/gamma(z)
        let gammas: Vec<Basic> = e.atoms(Some(&|g: &Basic| g.is_function(Function::Gamma)));
        for g in &gammas {
            let a: &Basic = &g.args()[0];
            for h in &gammas {
                let b: &Basic = &h.args()[0];
                if !is_expr(a) || !is_expr(b) {
                    continue;
                }
                let z: Basic = sub(a, &third)?;
                if sub(b, a)? == third && is_polynomial(&z, n) {
                    let three_z: Basic = mul(&[int(3), z.clone()])?;
                    let formula: Basic = div(
                        &mul(&[
                            int(2),
                            pi.clone(),
                            pow(&int(3), &sub(&half, &three_z)?)?,
                            gamma(&three_z)?,
                        ])?,
                        &mul(&[gamma(&z)?, h.clone()])?,
                    )?;
                    let replacement: HashMap<Basic, Basic> = [(g.clone(), formula)].into_iter().collect();
                    return Ok(e.xreplace(&replacement));
                }
            }
        }
        Ok(e.clone())
    };
    let mut candidates: Vec<Basic> = vec![simplify(&combsimp(solution)?)?];
    candidates.push(combsimp(&duplicate(&candidates[0])?)?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(combsimp(&triplicate(&last)?)?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(combsimp(&rewrite(&last, "factorial")?)?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(rewrite(&last, "binomial")?);
    let last = candidates.last().expect("nonempty").clone();
    candidates.push(simplify(&last)?);
    let mut best: Option<((i64, usize), Basic)> = None;
    for c in candidates {
        let key: (i64, usize) = (ops(&c)?, c.to_string().chars().count());
        if best.as_ref().map_or(true, |(b, _)| key < *b) {
            best = Some((key, c));
        }
    }
    Ok(best.expect("nonempty").1)
}

/// The term of degree zero of the polynomial `e` in `n`
/// (`_constant_term`).
fn constant_term(e: &Basic, n: &Basic) -> R {
    Ok(match upoly(e, n)? {
        Some(p) => p.nth(0),
        None => e.clone(),
    })
}

/// The first nonnegative integer from which the leading coefficient has no
/// more integer zeros (`_start`).
fn start(leading: &Basic, n: &Basic) -> PolyResult<i64> {
    if !is_polynomial(leading, n) || !leading.has(n) {
        return Ok(0);
    }
    let mut first: i64 = 0;
    for (f, _) in crate::polys::polytools::factor_list(leading, std::slice::from_ref(n))?.1 {
        let Some(linear) = upoly(&f, n)? else {
            continue;
        };
        if linear.degree() != 1 {
            continue;
        }
        let root: Basic = cancel(&neg(&div(&linear.nth(0), &linear.lc())?)?)?;
        if let Some(r) = integer_of(&root).filter(|r| *r >= first) {
            first = r + 1;
        }
    }
    Ok(first)
}

/// `G` at `k = at`, by a limit when the substitution is not defined
/// (`_value`).
fn value(g: &Basic, k: &Basic, at: &Basic) -> R {
    let mut v: Basic = subs(g, k, at)?;
    if has_nan_or_zoo(&v) {
        v = limit(g, k, at)?;
    }
    simplify(&v)
}

/// The recurrence as an unevaluated equation, or `None` when it is not one
/// (`_Recurrence`).
fn recurrence_equation(recurrence: &Basic) -> Option<Basic> {
    if has_nan_or_zoo(recurrence) {
        return None;
    }
    Some(crate::core::relational::rel(RelOp::Eq, recurrence, &Basic::zero()))
}
