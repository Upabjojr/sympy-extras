//! Port of the helpers of `sympy_extras/polys/polytools.py` which need no
//! expressions: `_factor_all` (the factorisation over `QQ` in all the
//! variables of a ring) and `_resultant_first` (the resultant in the first
//! variable). The class `Poly` and the rest of the module wait for the core.

#![allow(clippy::mutable_key_type)]

use std::sync::Arc;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::densebasic::dup_from_dict;
use super::domains::{IntegerRing, RationalField, SparsePolynomialRing, QQ, ZZ};
use super::euclidtools::{dup_prs_resultant, dup_resultant};
use super::multifactor as mf;
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

/// The coefficient and the integer primitive factors (positive leading
/// coefficients), sorted, of `f` over `QQ` in all the variables of its ring
/// (SymPy's `dup_factor_list`/`dmp_factor_list` over `ZZ` or `QQ`; the
/// port of `polytools._factor_all`); `ValueError` for zero.
pub fn factor_all(f: &PolyElement<RationalField>) -> PolyResult<(Rat, mf::Factors)> {
    if f.is_zero() {
        return Err(value_error("min() arg is an empty sequence".to_string()));
    }
    let names: Vec<String> = f.ring.names.clone();
    let n: usize = names.len();
    let exponents: Vec<u32> = (0..n)
        .map(|i| f.terms.keys().map(|m| m[i]).min().expect("a nonzero polynomial"))
        .collect();
    let shifted: Terms<Rat> = f
        .terms
        .iter()
        .map(|(m, c)| (m.iter().zip(&exponents).map(|(e, s)| e - s).collect(), c.clone()))
        .collect();
    let mut content: Rat = Rat::ZERO;
    for c in shifted.values() {
        let (cn, cd) = (numer(&content), denom(&content));
        let (vn, vd) = (numer(c), denom(c));
        content = rat_from_parts(gcd(&cn, &vn), &cd * &vd / gcd(&cd, &vd));
    }
    let ints: Terms<Int> = shifted
        .iter()
        .map(|(m, c)| (m.clone(), numer(&(c / &content))))
        .collect();
    let ring: Arc<PolyRing<IntegerRing>> = PolyRing::new(ZZ, names.clone(), MonomialOrder::Lex)?;
    let (coefficient, mut factors): (Int, mf::Factors) = if n == 1 {
        // dup_zz_factor, sorted below as the Python sorts it
        mf::zz_factor(&ring.from_dict(ints))?
    } else {
        let used: Vec<usize> = (0..n).filter(|&i| ints.keys().any(|m| m[i] != 0)).collect();
        if used.is_empty() {
            let zero: Monomial = vec![0; n];
            (ints.get(&zero).cloned().unwrap_or(Int::ZERO), Vec::new())
        } else {
            let sub: Arc<PolyRing<IntegerRing>> =
                PolyRing::new(ZZ, used.iter().map(|&i| names[i].clone()).collect(), MonomialOrder::Lex)?;
            let g: PolyElement<IntegerRing> = sub.from_dict(
                ints.iter()
                    .map(|(m, c)| (used.iter().map(|&i| m[i]).collect(), c.clone()))
                    .collect(),
            );
            let (coefficient, sub_factors) = mf::zz_factor(&g)?;
            let mut factors: mf::Factors = Vec::new();
            for (h, k) in sub_factors {
                let mut terms: Terms<Int> = Terms::new();
                for (m, value) in &h.terms {
                    let mut full: Monomial = vec![0; n];
                    for (position, e) in used.iter().zip(m) {
                        full[*position] = *e;
                    }
                    terms.insert(full, value.clone());
                }
                factors.push((ring.from_dict(terms), k));
            }
            (coefficient, factors)
        }
    };
    for i in (0..n).rev() {
        if exponents[i] != 0 {
            let monomial: Monomial = (0..n).map(|k| u32::from(k == i)).collect();
            factors.insert(0, (ring.term(monomial, Int::ONE), exponents[i] as usize));
        }
    }
    // the key of the univariate branch (degree, multiplicity, dense
    // coefficients) is that of `_sorted` in one variable
    Ok((rat_from_int(&coefficient) * content, mf::sorted(factors)))
}

/// The resultant in the first variable of two polynomials over `QQ`, as a
/// polynomial in all the variables (constant in the first); `integral`
/// computes it over `ZZ` (the coefficients must be integers).
pub fn resultant_first(
    f: &PolyElement<RationalField>,
    g: &PolyElement<RationalField>,
    names: &[String],
    integral: bool,
) -> PolyElement<RationalField> {
    if names.len() == 1 {
        let rf: Vec<Rat> = dup_from_dict(&f.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect(), &QQ);
        let rg: Vec<Rat> = dup_from_dict(&g.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect(), &QQ);
        let value: Rat = dup_resultant(&rf, &rg, &QQ);
        let mut terms: Terms<Rat> = Terms::new();
        terms.insert(vec![0], value);
        return f.ring.from_dict(terms);
    }
    if integral {
        let rz: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::new(ZZ, names.to_vec());
        let inner: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::new(ZZ, names[1..].to_vec());
        let to_int = |p: &PolyElement<RationalField>| -> PolyElement<IntegerRing> {
            rz.ring
                .from_dict(p.terms.iter().map(|(m, c)| (m.clone(), numer(c))).collect())
        };
        let r: PolyElement<IntegerRing> = dup_prs_resultant(
            &rz.to_univariate(&to_int(f), &inner.ring, &inner),
            &rz.to_univariate(&to_int(g), &inner.ring, &inner),
            &inner,
        );
        return f.ring.from_dict(
            r.terms
                .iter()
                .map(|(m, c)| {
                    let mut key: Monomial = vec![0];
                    key.extend_from_slice(m);
                    (key, rat_from_int(c))
                })
                .collect(),
        );
    }
    let rq: SparsePolynomialRing<RationalField> = SparsePolynomialRing::new(QQ, names.to_vec());
    let inner: SparsePolynomialRing<RationalField> = SparsePolynomialRing::new(QQ, names[1..].to_vec());
    let r: PolyElement<RationalField> = dup_prs_resultant(
        &rq.to_univariate(&rq.ring.from_dict(f.terms.clone()), &inner.ring, &inner),
        &rq.to_univariate(&rq.ring.from_dict(g.terms.clone()), &inner.ring, &inner),
        &inner,
    );
    f.ring.from_dict(
        r.terms
            .iter()
            .map(|(m, c)| {
                let mut key: Monomial = vec![0];
                key.extend_from_slice(m);
                (key, c.clone())
            })
            .collect(),
    )
}

// =====================================================================
// SymPy's `Poly` on the core (polytools.py, the class and its helpers)

use crate::core::add::add as core_add;
use crate::core::basic::{Basic, Head};
use crate::core::exprtools::decompose_power;
use crate::core::function::expand_default;
use crate::core::mul::mul as core_mul;
use crate::core::power::pow as core_pow;
use crate::error::{polys_error, PolyError, PolysErrorKind};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::sparsepolynomialring::IteratedPolynomialRing;
use std::collections::BTreeMap;

fn polynomial_error(message: String) -> PolyError {
    polys_error(PolysErrorKind::PolynomialError, message)
}

/// SymPy's order of the generators (`_gens_order`).
fn gens_order(name: &str) -> i64 {
    match name {
        "a" => 301,
        "b" => 302,
        "c" => 303,
        "d" => 304,
        "e" => 305,
        "f" => 306,
        "g" => 307,
        "h" => 308,
        "i" => 309,
        "j" => 310,
        "k" => 311,
        "l" => 312,
        "m" => 313,
        "n" => 314,
        "o" => 315,
        "p" => 216,
        "q" => 217,
        "r" => 218,
        "s" => 219,
        "t" => 220,
        "u" => 221,
        "v" => 222,
        "w" => 223,
        "x" => 124,
        "y" => 125,
        "z" => 126,
        _ => 1000,
    }
}

/// The sort key of a generator (`_gen_key`): the order of its name, the
/// name, the trailing number, and the creation index of a dummy.
fn gen_key(gen: &Basic) -> (i64, String, Int, i64) {
    let text: String = gen.to_string();
    let digits: usize = text.chars().rev().take_while(|c| c.is_ascii_digit()).count();
    let (name, index) = text.split_at(text.len() - digits);
    let created: i64 = match gen.head() {
        Head::Dummy(_, i) => *i as i64,
        _ => -1,
    };
    let index: Int = if index.is_empty() {
        Int::ZERO
    } else {
        index.parse::<Int>().unwrap_or(Int::ZERO)
    };
    (gens_order(name), name.to_string(), index, created)
}

/// The generators in SymPy's order (`sort_gens`).
pub fn sort_gens(gens: &[Basic]) -> Vec<Basic> {
    let mut sorted: Vec<Basic> = gens.to_vec();
    sorted.sort_by_cached_key(gen_key);
    sorted
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

/// The monomials with the coefficients of their terms.
pub(crate) type Decomposed = BTreeMap<Monomial, Vec<Basic>>;

fn exponent_of(n: i64) -> PolyResult<u32> {
    u32::try_from(n).map_err(|_| polynomial_error("a huge exponent".to_string()))
}

/// The terms of `e` (expanded) in the generators `gens`
/// (`_parallel_dict_from_expr_if_gens`).
pub(crate) fn decompose_with_gens(e: &Basic, gens: &[Basic]) -> PolyResult<Decomposed> {
    let mut result: Decomposed = Decomposed::new();
    for term in terms_of(&expand_default(e)?) {
        let mut coefficient: Vec<Basic> = Vec::new();
        let mut monomial: Monomial = vec![0; gens.len()];
        for factor in factors_of(&term) {
            if factor.is_number_class() {
                coefficient.push(factor);
                continue;
            }
            let (mut base, mut exponent) = decompose_power(&factor)?;
            if exponent < 0 {
                exponent = -exponent;
                base = core_pow(&base, &Basic::negative_one())?;
            }
            if let Some(i) = gens.iter().position(|g| *g == base) {
                monomial[i] = exponent_of(exponent)?;
            } else if !gens.iter().any(|g| factor.has(g)) {
                coefficient.push(factor);
            } else {
                return Err(polynomial_error(format!(
                    "{} contains an element of the set of generators.",
                    factor
                )));
            }
        }
        result.entry(monomial).or_default().push(core_mul(&coefficient)?);
    }
    Ok(result)
}

/// The terms of the expressions with every factor which is not a number a
/// generator, the generators of all of them together
/// (`_decompose_without_gens_many`, SymPy's
/// `_parallel_dict_from_expr_no_gens`).
fn decompose_without_gens_many(exprs: &[Basic], expand: bool) -> PolyResult<(Vec<Decomposed>, Vec<Basic>)> {
    let mut found: Vec<Basic> = Vec::new();
    let mut raws: Vec<Vec<(Vec<Basic>, Vec<(Basic, i64)>)>> = Vec::new();
    for e in exprs {
        let mut raw: Vec<(Vec<Basic>, Vec<(Basic, i64)>)> = Vec::new();
        let expanded: Basic = if expand { expand_default(e)? } else { e.clone() };
        for term in terms_of(&expanded) {
            let mut coefficient: Vec<Basic> = Vec::new();
            let mut elements: Vec<(Basic, i64)> = Vec::new();
            for factor in factors_of(&term) {
                if factor.is_number_class() || matches!(factor.head(), Head::ImaginaryUnit) {
                    coefficient.push(factor);
                    continue;
                }
                let (mut base, mut exponent) = decompose_power(&factor)?;
                if exponent < 0 {
                    exponent = -exponent;
                    base = core_pow(&base, &Basic::negative_one())?;
                }
                match elements.iter_mut().find(|(b, _)| *b == base) {
                    Some((_, k)) => *k += exponent,
                    None => elements.push((base.clone(), exponent)),
                }
                if !found.contains(&base) {
                    found.push(base);
                }
            }
            raw.push((coefficient, elements));
        }
        raws.push(raw);
    }
    let gens: Vec<Basic> = sort_gens(&found);
    let mut results: Vec<Decomposed> = Vec::new();
    for raw in raws {
        let mut result: Decomposed = Decomposed::new();
        for (coefficient, elements) in raw {
            let mut monomial: Monomial = vec![0; gens.len()];
            for (base, exponent) in elements {
                let i: usize = gens.iter().position(|g| *g == base).expect("a generator");
                monomial[i] = exponent_of(exponent)?;
            }
            result.entry(monomial).or_default().push(core_mul(&coefficient)?);
        }
        results.push(result);
    }
    Ok((results, gens))
}

fn decompose_without_gens(e: &Basic) -> PolyResult<(Decomposed, Vec<Basic>)> {
    let (mut decomposed, gens) = decompose_without_gens_many(std::slice::from_ref(e), true)?;
    Ok((decomposed.remove(0), gens))
}

/// SymPy's domain of a `Poly`, restricted to `ZZ`, `QQ`, `ZZ[params]`,
/// `QQ[params]` (`_Domain`).
#[derive(Debug, Clone, PartialEq)]
pub struct PolyDomain {
    pub integral: bool,
    pub params: Vec<Basic>,
}

impl PolyDomain {
    pub fn name(&self) -> String {
        let ground: &str = if self.integral { "ZZ" } else { "QQ" };
        if self.params.is_empty() {
            return ground.to_string();
        }
        let names: Vec<String> = self.params.iter().map(|p| p.to_string()).collect();
        format!("{}[{}]", ground, names.join(","))
    }
}

fn unify_domains(a: &PolyDomain, b: &PolyDomain) -> PolyDomain {
    let mut all: Vec<Basic> = a.params.clone();
    for p in &b.params {
        if !all.contains(p) {
            all.push(p.clone());
        }
    }
    let params: Vec<Basic> = sort_gens(&all).into_iter().filter(|p| p.is_symbol()).collect();
    PolyDomain {
        integral: a.integral && b.integral,
        params,
    }
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Wild(w) => w.symbol.name.clone(),
        _ => s.to_string(),
    }
}

/// SymPy's `construct_domain` for rational coefficients and polynomial
/// rings (`_construct_domain`).
fn construct_domain(coefficients: &[Basic]) -> PolyResult<PolyDomain> {
    let mut symbols: Vec<Basic> = Vec::new();
    for c in coefficients {
        if c.has(&Basic::imaginary_unit()) {
            return Err(polys_error(
                PolysErrorKind::CoercionFailed,
                format!("the domain ZZ_I of {} is not ported yet", c),
            ));
        }
        for s in c.free_symbols() {
            if !symbols.contains(&s) {
                symbols.push(s);
            }
        }
    }
    let params: Vec<Basic> = sort_gens(&symbols).into_iter().filter(|p| p.is_symbol()).collect();
    let names: Vec<String> = params.iter().map(symbol_name).collect();
    let mut integral: bool = true;
    for c in coefficients {
        let values: Vec<Rat> = if !names.is_empty() {
            match crate::polys::polyutils::dict_from_expr(c, &names, &QQ) {
                Ok(terms) => terms.into_values().collect(),
                Err(_) => {
                    return Err(polys_error(
                        PolysErrorKind::CoercionFailed,
                        format!("the domain of {} (EX or a fraction field) is not ported yet", c),
                    ))
                }
            }
        } else {
            match c.as_rational() {
                Some(v) => vec![v.clone()],
                None => {
                    return Err(polys_error(
                        PolysErrorKind::CoercionFailed,
                        format!("the domain of {} (EX, RR or an extension) is not ported yet", c),
                    ))
                }
            }
        };
        if values.iter().any(|v| !v.is_int()) {
            integral = false;
        }
    }
    Ok(PolyDomain { integral, params })
}

fn internal_names(n_gens: usize, n_params: usize) -> Vec<String> {
    let mut names: Vec<String> = (0..n_gens).map(|i| format!("_g{}", i)).collect();
    names.extend((0..n_params).map(|i| format!("_p{}", i)));
    names
}

fn rational_ring(names: Vec<String>) -> Arc<PolyRing<RationalField>> {
    PolyRing::new(QQ, names, MonomialOrder::Lex).expect("at least one variable")
}

fn rep_from_coefficients(
    coefficients: &BTreeMap<Monomial, Basic>,
    gens: &[Basic],
    domain: &PolyDomain,
) -> PolyResult<PolyElement<RationalField>> {
    let names: Vec<String> = domain.params.iter().map(symbol_name).collect();
    let ring = rational_ring(internal_names(gens.len(), domain.params.len()));
    let mut terms: Terms<Rat> = Terms::new();
    for (m, c) in coefficients {
        if !names.is_empty() {
            for (pm, value) in crate::polys::polyutils::dict_from_expr(c, &names, &QQ)? {
                if !value.is_zero() {
                    let mut key: Monomial = m.clone();
                    key.extend(pm);
                    let total: Rat = terms.remove(&key).unwrap_or(Rat::ZERO) + value;
                    terms.insert(key, total);
                }
            }
        } else if let Some(v) = c.as_rational() {
            if !v.is_zero() {
                let total: Rat = terms.remove(m).unwrap_or(Rat::ZERO) + v;
                terms.insert(m.clone(), total);
            }
        } else {
            return Err(polys_error(
                PolysErrorKind::CoercionFailed,
                format!("cannot convert {} into {}", c, domain.name()),
            ));
        }
    }
    terms.retain(|_, c| !c.is_zero());
    Ok(ring.from_dict(terms))
}

fn terms_gcd(values: &[Rat]) -> Int {
    let mut result: Int = Int::ZERO;
    for v in values {
        result = gcd(&result, &numer(v));
    }
    result
}

fn denominators_lcm(f: &PolyElement<RationalField>) -> Int {
    let mut result: Int = Int::ONE;
    for c in f.terms.values() {
        let d: Int = denom(c);
        result = &result * &d / gcd(&result, &d);
    }
    result
}

/// A polynomial in generators with coefficients in `ZZ`, `QQ`,
/// `ZZ[params]` or `QQ[params]` (SymPy's `Poly`): a sparse polynomial over
/// `QQ` in the generators followed by the parameters.
#[derive(Debug, Clone)]
pub struct Poly {
    pub rep: PolyElement<RationalField>,
    pub gens: Vec<Basic>,
    pub domain: PolyDomain,
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        if self.gens != other.gens {
            return false;
        }
        if self.domain.params == other.domain.params {
            return self.rep.terms == other.rep.terms;
        }
        match (self.as_expr(), other.as_expr()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
}

/// `R.ring` of a plain or an iterated sparse polynomial ring.
trait WithRing<K: Domain> {
    fn base_ring(&self) -> &Arc<PolyRing<K>>;
}

impl<K: crate::polys::domains::sparsepolynomialring::SparseGround> WithRing<K> for SparsePolynomialRing<K> {
    fn base_ring(&self) -> &Arc<PolyRing<K>> {
        &self.ring
    }
}

impl<K: crate::polys::domains::sparsepolynomialring::SparseGround> WithRing<K> for IteratedPolynomialRing<K> {
    fn base_ring(&self) -> &Arc<PolyRing<K>> {
        self.ring()
    }
}

impl Poly {
    /// `Poly(expr, *gens)`: the generators found when none is given.
    pub fn new(expr: &Basic, gens: &[Basic]) -> PolyResult<Poly> {
        if crate::core::expr::is_false(expr, crate::core::assumptions::Predicate::commutative) {
            return Err(polynomial_error(
                "non-commutative expressions are not supported".to_string(),
            ));
        }
        let (decomposed, generators) = if !gens.is_empty() {
            let distinct: Vec<&Basic> = gens.iter().collect();
            if (0..distinct.len()).any(|i| distinct[i + 1..].contains(&distinct[i])) {
                let shown: Vec<String> = gens.iter().map(|g| g.to_string()).collect();
                return Err(polys_error(
                    PolysErrorKind::GeneratorsError,
                    format!("duplicated generators: ({})", shown.join(", ")),
                ));
            }
            (decompose_with_gens(expr, gens)?, gens.to_vec())
        } else {
            let (d, g) = decompose_without_gens(expr)?;
            if g.is_empty() {
                return Err(polys_error(
                    PolysErrorKind::GeneratorsNeeded,
                    format!("specify generators to give {} a polynomial representation", expr),
                ));
            }
            (d, g)
        };
        let mut coefficients: BTreeMap<Monomial, Basic> = BTreeMap::new();
        for (m, cs) in decomposed {
            coefficients.insert(m, core_add(&cs)?);
        }
        let values: Vec<Basic> = coefficients.values().cloned().collect();
        let domain: PolyDomain = construct_domain(&values)?;
        let rep = rep_from_coefficients(&coefficients, &generators, &domain)?;
        Ok(Poly {
            rep,
            gens: generators,
            domain,
        })
    }

    /// The polynomial of rational coefficients in `gens` over `ZZ` or `QQ`
    /// (`Poly.from_dict`).
    pub fn from_dict(terms: &[(Monomial, Rat)], gens: &[Basic]) -> PolyResult<Poly> {
        if gens.is_empty() {
            return Err(polys_error(
                PolysErrorKind::GeneratorsNeeded,
                "generators are needed to build a polynomial from a dictionary".to_string(),
            ));
        }
        let mut values: Terms<Rat> = Terms::new();
        for (m, c) in terms {
            if m.len() != gens.len() {
                return Err(polynomial_error(
                    "the monomial does not match the generators".to_string(),
                ));
            }
            if !c.is_zero() {
                let total: Rat = values.remove(m).unwrap_or(Rat::ZERO) + c;
                values.insert(m.clone(), total);
            }
        }
        values.retain(|_, c| !c.is_zero());
        let integral: bool = values.values().all(|c| c.is_int());
        let ring = rational_ring(internal_names(gens.len(), 0));
        Ok(Poly {
            rep: ring.from_dict(values),
            gens: gens.to_vec(),
            domain: PolyDomain {
                integral,
                params: Vec::new(),
            },
        })
    }

    fn same_ring(&self, rep: PolyElement<RationalField>, integral: Option<bool>) -> Poly {
        let domain: PolyDomain = match integral {
            None => self.domain.clone(),
            Some(i) => PolyDomain {
                integral: i,
                params: self.domain.params.clone(),
            },
        };
        Poly {
            rep,
            gens: self.gens.clone(),
            domain,
        }
    }

    // -- structure

    pub(crate) fn split(&self) -> BTreeMap<Monomial, BTreeMap<Monomial, Rat>> {
        let n: usize = self.gens.len();
        let mut grouped: BTreeMap<Monomial, BTreeMap<Monomial, Rat>> = BTreeMap::new();
        for (m, c) in &self.rep.terms {
            grouped
                .entry(m[..n].to_vec())
                .or_default()
                .insert(m[n..].to_vec(), c.clone());
        }
        grouped
    }

    pub(crate) fn coefficient_expr(&self, terms: &BTreeMap<Monomial, Rat>) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (m, c) in terms {
            let mut factors: Vec<Basic> = vec![Basic::from_rational(c.clone())];
            for (p, k) in self.domain.params.iter().zip(m) {
                if *k > 0 {
                    factors.push(core_pow(p, &Basic::int(i64::from(*k)))?);
                }
            }
            parts.push(core_mul(&factors)?);
        }
        core_add(&parts)
    }

    /// The terms `(monomial, coefficient)` in lex order.
    pub fn terms(&self) -> PolyResult<Vec<(Monomial, Basic)>> {
        let mut result: Vec<(Monomial, Basic)> = Vec::new();
        for (m, c) in self.split().iter().rev() {
            result.push((m.clone(), self.coefficient_expr(c)?));
        }
        Ok(result)
    }

    pub fn monoms(&self) -> PolyResult<Vec<Monomial>> {
        let m: Vec<Monomial> = self.terms()?.into_iter().map(|(m, _)| m).collect();
        Ok(if m.is_empty() {
            vec![vec![0; self.gens.len()]]
        } else {
            m
        })
    }

    pub fn coeffs(&self) -> PolyResult<Vec<Basic>> {
        let c: Vec<Basic> = self.terms()?.into_iter().map(|(_, c)| c).collect();
        Ok(if c.is_empty() { vec![Basic::zero()] } else { c })
    }

    pub fn as_expr(&self) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (m, c) in self.terms()? {
            let mut factors: Vec<Basic> = vec![c];
            for (g, k) in self.gens.iter().zip(&m) {
                if *k > 0 {
                    factors.push(core_pow(g, &Basic::int(i64::from(*k)))?);
                }
            }
            parts.push(core_mul(&factors)?);
        }
        core_add(&parts)
    }

    /// The coefficient of the monomial `exponents` (`nth`, `coeff_monomial`).
    pub fn nth(&self, exponents: &[u32]) -> PolyResult<Basic> {
        Ok(self
            .terms()?
            .into_iter()
            .find(|(m, _)| m.as_slice() == exponents)
            .map(|(_, c)| c)
            .unwrap_or_else(Basic::zero))
    }

    #[allow(non_snake_case)]
    pub fn LC(&self) -> PolyResult<Basic> {
        Ok(self.coeffs()?.remove(0))
    }

    /// The coefficients of a univariate polynomial, the leading one first.
    pub fn all_coeffs(&self) -> PolyResult<Vec<Basic>> {
        if self.gens.len() != 1 {
            return Err(polynomial_error("multivariate polynomials not supported".to_string()));
        }
        let n: i64 = self.degree(0)?;
        if n < 0 {
            return Ok(vec![Basic::zero()]);
        }
        let terms: Vec<(Monomial, Basic)> = self.terms()?;
        Ok((0..=n)
            .map(|i| {
                terms
                    .iter()
                    .find(|(m, _)| i64::from(m[0]) == n - i)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(Basic::zero)
            })
            .collect())
    }

    /// The degree in the generator of index `i` (`-1` for zero).
    pub fn degree(&self, i: usize) -> PolyResult<i64> {
        if i >= self.gens.len() {
            return Err(polynomial_error(format!(
                "-{} <= gen < {} expected, got {}",
                self.gens.len(),
                self.gens.len(),
                i
            )));
        }
        Ok(self.rep.terms.keys().map(|m| i64::from(m[i])).max().unwrap_or(-1))
    }

    pub fn total_degree(&self) -> i64 {
        let n: usize = self.gens.len();
        self.rep
            .terms
            .keys()
            .map(|m| m[..n].iter().map(|e| i64::from(*e)).sum())
            .max()
            .unwrap_or(-1)
    }

    pub fn is_zero(&self) -> bool {
        self.rep.terms.is_empty()
    }

    /// The number of terms in the generators (`length`).
    pub fn length(&self) -> usize {
        self.split().len()
    }

    /// Whether the polynomial is a constant in its generators.
    pub fn is_ground(&self) -> bool {
        self.split().keys().all(|m| m.iter().all(|e| *e == 0))
    }

    // -- printing

    /// `str(poly)` as SymPy prints `Poly` (`_print_Poly`).
    pub fn to_str(&self) -> PolyResult<String> {
        let atom = |e: &Basic| -> String {
            let text: String = e.to_string();
            if e.is_symbol() || e.is_number_class() {
                text
            } else {
                format!("({})", text)
            }
        };
        let gens: Vec<String> = self.gens.iter().map(atom).collect();
        let mut pieces: Vec<String> = Vec::new();
        for (monomial, coefficient) in self.terms()? {
            let monomial_text: String = monomial
                .iter()
                .enumerate()
                .filter(|(_, e)| **e > 0)
                .map(|(i, e)| {
                    if *e == 1 {
                        gens[i].clone()
                    } else {
                        format!("{}**{}", gens[i], e)
                    }
                })
                .collect::<Vec<_>>()
                .join("*");
            let text: String = if coefficient.is_add() {
                if monomial_text.is_empty() {
                    coefficient.to_string()
                } else {
                    format!("({})", coefficient)
                }
            } else {
                if !monomial_text.is_empty() && coefficient == Basic::one() {
                    pieces.push("+".to_string());
                    pieces.push(monomial_text);
                    continue;
                }
                if !monomial_text.is_empty() && coefficient == Basic::negative_one() {
                    pieces.push("-".to_string());
                    pieces.push(monomial_text);
                    continue;
                }
                match coefficient.as_rational() {
                    Some(v) if !v.is_int() => format!("{}/{}", numer(v), denom(v)),
                    _ => coefficient.to_string(),
                }
            };
            let term: String = if monomial_text.is_empty() {
                text
            } else {
                format!("{}*{}", text, monomial_text)
            };
            match term.strip_prefix('-') {
                Some(rest) => {
                    pieces.push("-".to_string());
                    pieces.push(rest.to_string());
                }
                None => {
                    pieces.push("+".to_string());
                    pieces.push(term);
                }
            }
        }
        if pieces.is_empty() {
            pieces = vec!["+".to_string(), "0".to_string()];
        }
        if pieces[0] == "-" || pieces[0] == "+" {
            let sign: String = pieces.remove(0);
            if sign == "-" {
                pieces[0] = format!("-{}", pieces[0]);
            }
        }
        let names: Vec<String> = gens
            .iter()
            .map(|g| {
                if g.len() > 2 && g.starts_with('(') && g.ends_with(')') {
                    g[1..g.len() - 1].to_string()
                } else {
                    g.clone()
                }
            })
            .collect();
        Ok(format!(
            "Poly({}, {}, domain='{}')",
            pieces.join(" "),
            names.join(", "),
            self.domain.name()
        ))
    }

    // -- arithmetic

    /// `self` and `other` over the same generators and domain
    /// (`_unified`).
    pub fn unified(&self, other: &Poly) -> PolyResult<(Poly, Poly)> {
        if other.gens != self.gens {
            let mut gens: Vec<Basic> = self.gens.clone();
            for g in &other.gens {
                if !gens.contains(g) {
                    gens.push(g.clone());
                }
            }
            return Ok((
                Poly::new(&self.as_expr()?, &gens)?,
                Poly::new(&other.as_expr()?, &gens)?,
            ));
        }
        if other.domain == self.domain {
            return Ok((self.clone(), other.clone()));
        }
        let domain: PolyDomain = unify_domains(&self.domain, &other.domain);
        Ok((self.converted(&domain), other.converted(&domain)))
    }

    fn converted(&self, domain: &PolyDomain) -> Poly {
        let n: usize = self.gens.len();
        let ring = rational_ring(internal_names(n, domain.params.len()));
        let mut terms: Terms<Rat> = Terms::new();
        for (m, c) in &self.rep.terms {
            let mut exponents: Vec<u32> = vec![0; domain.params.len()];
            for (p, k) in self.domain.params.iter().zip(&m[n..]) {
                let position: usize = domain.params.iter().position(|q| q == p).expect("a parameter");
                exponents[position] = *k;
            }
            let mut key: Monomial = m[..n].to_vec();
            key.extend(exponents);
            terms.insert(key, c.clone());
        }
        Poly {
            rep: ring.from_dict(terms),
            gens: self.gens.clone(),
            domain: domain.clone(),
        }
    }

    pub fn neg(&self) -> Poly {
        self.same_ring(self.rep.neg(), None)
    }

    pub fn add(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        Ok(a.same_ring(a.rep.add(&b.rep), None))
    }

    pub fn sub(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        Ok(a.same_ring(a.rep.sub(&b.rep), None))
    }

    pub fn mul(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        Ok(a.same_ring(a.rep.mul(&b.rep), None))
    }

    pub fn pow(&self, n: u64) -> Poly {
        self.same_ring(self.rep.pow(n), None)
    }

    pub fn diff(&self, i: usize) -> Poly {
        self.same_ring(self.rep.diff(i), None)
    }

    // -- algorithms on the representation

    fn integer_rep(&self, ring: &Arc<PolyRing<IntegerRing>>) -> PolyElement<IntegerRing> {
        ring.from_dict(self.rep.terms.iter().map(|(m, c)| (m.clone(), numer(c))).collect())
    }

    fn rational_rep(&self, ring: &Arc<PolyRing<RationalField>>) -> PolyElement<RationalField> {
        ring.from_dict(self.rep.terms.clone())
    }

    fn from_integers(&self, f: &PolyElement<IntegerRing>) -> Poly {
        self.same_ring(
            self.rep
                .ring
                .from_dict(f.terms.iter().map(|(m, c)| (m.clone(), rat_from_int(c))).collect()),
            None,
        )
    }

    fn from_rationals(&self, f: &PolyElement<RationalField>) -> Poly {
        self.same_ring(self.rep.ring.from_dict(f.terms.clone()), None)
    }

    fn names(&self) -> Vec<String> {
        internal_names(self.gens.len(), self.domain.params.len())
    }

    /// `op` in the integer ring of the polynomial (`_integer_ring`).
    fn with_integer_ring<T>(&self, op: impl Fn(&dyn IntegerRingOps) -> T) -> T {
        let names: Vec<String> = self.names();
        let n: usize = self.gens.len();
        if self.domain.params.is_empty() {
            op(&SparsePolynomialRing::sympy(ZZ, names))
        } else {
            op(&IteratedPolynomialRing::new(
                SparsePolynomialRing::sympy(ZZ, names[n..].to_vec()),
                names[..n].to_vec(),
            ))
        }
    }

    fn with_rational_ring<T>(&self, op: impl Fn(&dyn RationalRingOps) -> T) -> T {
        let names: Vec<String> = self.names();
        let n: usize = self.gens.len();
        if self.domain.params.is_empty() {
            op(&SparsePolynomialRing::sympy(QQ, names))
        } else {
            op(&IteratedPolynomialRing::new(
                SparsePolynomialRing::sympy(QQ, names[n..].to_vec()),
                names[..n].to_vec(),
            ))
        }
    }

    /// SymPy's gcd (`gcd`): a positive leading coefficient over `ZZ`, monic
    /// over `QQ`.
    pub fn gcd(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        if a.domain.integral {
            let g = a.with_integer_ring(|z| {
                let ring = z.ring_of();
                z.gcd_of(&a.integer_rep(&ring), &b.integer_rep(&ring))
            });
            return Ok(a.from_integers(&g));
        }
        let mut g = a.with_rational_ring(|q| {
            let ring = q.ring_of();
            q.gcd_of(&a.rational_rep(&ring), &b.rational_rep(&ring))
        });
        if a.domain.params.is_empty() && !g.terms.is_empty() {
            g = g.exquo_ground(&g.LC())?;
        }
        Ok(a.from_rationals(&g))
    }

    /// SymPy's lcm (`lcm`).
    pub fn lcm(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        if a.is_zero() || b.is_zero() {
            return Ok(a.same_ring(a.rep.ring.zero(), None));
        }
        if !a.domain.params.is_empty() {
            if a.domain.integral {
                let l = a.with_integer_ring(|z| {
                    let ring = z.ring_of();
                    z.lcm_of(&a.integer_rep(&ring), &b.integer_rep(&ring))
                });
                return Ok(a.from_integers(&l));
            }
            let l = a.with_rational_ring(|q| {
                let ring = q.ring_of();
                q.lcm_of(&a.rational_rep(&ring), &b.rational_rep(&ring))
            });
            return Ok(a.from_rationals(&l));
        }
        if !a.domain.integral {
            let h = a.rep.mul(&b.rep).exquo(&a.gcd(&b)?.rep)?;
            let lc: Rat = h.LC();
            return Ok(a.same_ring(h.exquo_ground(&lc)?, None));
        }
        let values_a: Vec<Rat> = a.rep.terms.values().cloned().collect();
        let values_b: Vec<Rat> = b.rep.terms.values().cloned().collect();
        let (fc, gc) = (terms_gcd(&values_a), terms_gcd(&values_b));
        let f = a.rep.exquo_ground(&rat_from_int(&fc))?;
        let g = b.rep.exquo_ground(&rat_from_int(&gc))?;
        let mut c: Rat = rat_from_int(&(&fc * &gc / gcd(&fc, &gc)));
        let h = f
            .mul(&g)
            .exquo(&a.same_ring(f.clone(), None).gcd(&a.same_ring(g.clone(), None))?.rep)?;
        if a.gens.len() == 1 && rat_is_negative(&h.LC()) {
            c = -c;
        }
        Ok(a.same_ring(h.mul_ground(&c), None))
    }

    /// `(gcd, self/gcd, other/gcd)`.
    pub fn cofactors(&self, other: &Poly) -> PolyResult<(Poly, Poly, Poly)> {
        let (a, b) = self.unified(other)?;
        let g: Poly = a.gcd(&b)?;
        if g.is_zero() {
            return Ok((g, a, b));
        }
        let (p, q) = (a.rep.exquo(&g.rep)?, b.rep.exquo(&g.rep)?);
        Ok((g, a.same_ring(p, None), b.same_ring(q, None)))
    }

    /// `(c, p, q)` with `self/other = c p/q` in lowest terms (SymPy's
    /// `Poly.cancel`).
    pub fn cancel(&self, other: &Poly) -> PolyResult<(Basic, Poly, Poly)> {
        let (a, b) = self.unified(other)?;
        if b.is_zero() {
            return Err(crate::error::zero_division("cancel by a zero polynomial"));
        }
        let (mut cp, mut cq): (Int, Int) = (Int::ONE, Int::ONE);
        let (mut f, mut g) = (a.clone(), b.clone());
        if !a.domain.integral && a.domain.params.is_empty() {
            cq = denominators_lcm(&a.rep);
            cp = denominators_lcm(&b.rep);
            f = a.same_ring(a.rep.mul_ground(&rat_from_int(&cq)), Some(true));
            g = b.same_ring(b.rep.mul_ground(&rat_from_int(&cp)), Some(true));
        }
        let h: Poly = f.gcd(&g)?;
        let mut p: Poly = f.same_ring(f.rep.exquo(&h.rep)?, Some(a.domain.integral));
        let mut q: Poly = g.same_ring(g.rep.exquo(&h.rep)?, Some(a.domain.integral));
        let common: Int = gcd(&cp, &cq);
        cp = &cp / &common;
        cq = &cq / &common;
        let (p_neg, q_neg) = (p.ground_negative(), q.ground_negative());
        if p_neg && q_neg {
            p = p.neg();
            q = q.neg();
        } else if p_neg {
            cp = -cp;
            p = p.neg();
        } else if q_neg {
            cp = -cp;
            q = q.neg();
        }
        Ok((Basic::from_rational(rat_from_parts(cp, cq)), p, q))
    }

    fn ground_negative(&self) -> bool {
        !self.rep.terms.is_empty() && rat_is_negative(&self.rep.LC())
    }

    /// Division with remainder of univariate polynomials over `QQ`
    /// (`div`; a polynomial domain only by a divisor with a numerical
    /// leading coefficient).
    pub fn div(&self, other: &Poly) -> PolyResult<(Poly, Poly)> {
        let (a, b) = self.unified(other)?;
        if b.is_zero() {
            return Err(crate::error::zero_division("polynomial division by zero"));
        }
        if !a.domain.params.is_empty() {
            return a.div_over_parameters(&b);
        }
        let (q, r) = super::densearith::dup_div(&a.univariate_rational()?, &b.univariate_rational()?, &QQ)?;
        let integral: bool = a.domain.integral && q.iter().chain(&r).all(|c| c.is_int());
        Ok((a.from_univariate(&q, integral), a.from_univariate(&r, integral)))
    }

    fn univariate_rational(&self) -> PolyResult<Vec<Rat>> {
        if self.gens.len() != 1 || !self.domain.params.is_empty() {
            return Err(polys_error(
                PolysErrorKind::OperationNotSupported,
                "only univariate polynomials over ZZ or QQ".to_string(),
            ));
        }
        let dict: BTreeMap<usize, Rat> = self.rep.terms.iter().map(|(m, c)| (m[0] as usize, c.clone())).collect();
        Ok(dup_from_dict(&dict, &QQ))
    }

    fn from_univariate(&self, f: &[Rat], integral: bool) -> Poly {
        let n: usize = f.len().saturating_sub(1);
        let terms: Terms<Rat> = f
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| (vec![(n - i) as u32], c.clone()))
            .collect();
        self.same_ring(self.rep.ring.from_dict(terms), Some(integral))
    }

    fn div_over_parameters(&self, b: &Poly) -> PolyResult<(Poly, Poly)> {
        if self.gens.len() != 1 {
            return Err(polys_error(
                PolysErrorKind::OperationNotSupported,
                "division of multivariate polynomials is not ported yet".to_string(),
            ));
        }
        let divisor = b.split();
        let top: u32 = divisor.keys().map(|m| m[0]).max().expect("a nonzero divisor");
        let lead = &divisor[&vec![top]];
        let zero: Monomial = vec![0; self.domain.params.len()];
        if lead.len() != 1 || !lead.contains_key(&zero) {
            return Err(polys_error(
                PolysErrorKind::CoercionFailed,
                "the division needs the fraction field (not ported yet)".to_string(),
            ));
        }
        let lc: Rat = lead[&zero].clone();
        let mut remainder: BTreeMap<u32, BTreeMap<Monomial, Rat>> =
            self.split().into_iter().map(|(m, c)| (m[0], c)).collect();
        let mut quotient: BTreeMap<u32, BTreeMap<Monomial, Rat>> = BTreeMap::new();
        while let Some(&d) = remainder.keys().max() {
            if d < top {
                break;
            }
            let factor: BTreeMap<Monomial, Rat> = remainder[&d].iter().map(|(pm, c)| (pm.clone(), c / &lc)).collect();
            quotient.insert(d - top, factor.clone());
            for (e, coefficients) in &divisor {
                let key_degree: u32 = e[0] + d - top;
                let target = remainder.entry(key_degree).or_default();
                for (pm, c) in coefficients {
                    for (qm, qc) in &factor {
                        let key: Monomial = pm.iter().zip(qm).map(|(i, j)| i + j).collect();
                        let value: Rat = target.remove(&key).unwrap_or(Rat::ZERO) - c * qc;
                        if !value.is_zero() {
                            target.insert(key, value);
                        }
                    }
                }
                if target.is_empty() {
                    remainder.remove(&key_degree);
                }
            }
        }
        let integral: bool = self.domain.integral
            && quotient
                .values()
                .chain(remainder.values())
                .all(|cs| cs.values().all(|c| c.is_int()));
        Ok((
            self.from_grouped(&quotient, integral),
            self.from_grouped(&remainder, integral),
        ))
    }

    fn from_grouped(&self, grouped: &BTreeMap<u32, BTreeMap<Monomial, Rat>>, integral: bool) -> Poly {
        let mut terms: Terms<Rat> = Terms::new();
        for (d, coefficients) in grouped {
            for (pm, c) in coefficients {
                if !c.is_zero() {
                    let mut key: Monomial = vec![*d];
                    key.extend(pm.iter().copied());
                    terms.insert(key, c.clone());
                }
            }
        }
        self.same_ring(self.rep.ring.from_dict(terms), Some(integral))
    }

    pub fn exquo(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        match a.rep.exquo(&b.rep) {
            Ok(q) => Ok(a.same_ring(q, None)),
            Err(_) => Err(polys_error(
                PolysErrorKind::ExactQuotientFailed,
                format!("{} does not divide {}", b.to_str()?, a.to_str()?),
            )),
        }
    }
}

/// The gcd and the lcm of a plain or iterated integer ring.
trait IntegerRingOps {
    fn ring_of(&self) -> Arc<PolyRing<IntegerRing>>;
    fn gcd_of(&self, a: &PolyElement<IntegerRing>, b: &PolyElement<IntegerRing>) -> PolyElement<IntegerRing>;
    fn lcm_of(&self, a: &PolyElement<IntegerRing>, b: &PolyElement<IntegerRing>) -> PolyElement<IntegerRing>;
}

trait RationalRingOps {
    fn ring_of(&self) -> Arc<PolyRing<RationalField>>;
    fn gcd_of(&self, a: &PolyElement<RationalField>, b: &PolyElement<RationalField>) -> PolyElement<RationalField>;
    fn lcm_of(&self, a: &PolyElement<RationalField>, b: &PolyElement<RationalField>) -> PolyElement<RationalField>;
}

macro_rules! ring_ops {
    ($trait:ident, $k:ty, $ring:ty) => {
        impl $trait for $ring {
            fn ring_of(&self) -> Arc<PolyRing<$k>> {
                self.base_ring().clone()
            }
            fn gcd_of(&self, a: &PolyElement<$k>, b: &PolyElement<$k>) -> PolyElement<$k> {
                Domain::gcd(self, a, b)
            }
            fn lcm_of(&self, a: &PolyElement<$k>, b: &PolyElement<$k>) -> PolyElement<$k> {
                Domain::lcm(self, a, b)
            }
        }
    };
}

ring_ops!(IntegerRingOps, IntegerRing, SparsePolynomialRing<IntegerRing>);
ring_ops!(IntegerRingOps, IntegerRing, IteratedPolynomialRing<IntegerRing>);
ring_ops!(RationalRingOps, RationalField, SparsePolynomialRing<RationalField>);
ring_ops!(RationalRingOps, RationalField, IteratedPolynomialRing<RationalField>);

// ---------------------------------------------------------------------
// Poly, stage 2: monic, content, eval, resultants, sqf_list, factor_list

/// A polynomial, or an expression when no generator is left (the results
/// of `eval`, `resultant`, `discriminant`).
#[derive(Debug, Clone, PartialEq)]
pub enum PolyOrExpr {
    Poly(Poly),
    Expr(Basic),
}

/// SymPy's sort key of a polynomial over a polynomial domain
/// (`_domain_key`): nested over the generators, the coefficients by their
/// number of terms and then their terms in decreasing lex order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum DomainKey {
    Coefficient(usize, Vec<(Monomial, Int)>),
    Level(Vec<DomainKey>),
}

fn domain_key_level(terms: &[(Monomial, Int)], i: usize, n: usize) -> DomainKey {
    if i == n {
        let mut ordered: Vec<(Monomial, Int)> = terms.iter().map(|(m, c)| (m[n..].to_vec(), c.clone())).collect();
        ordered.sort();
        ordered.reverse();
        return DomainKey::Coefficient(ordered.len(), ordered);
    }
    if terms.is_empty() {
        return DomainKey::Level(Vec::new());
    }
    let top: u32 = terms.iter().map(|(m, _)| m[i]).max().expect("terms");
    DomainKey::Level(
        (0..=top)
            .rev()
            .map(|d| {
                let part: Vec<(Monomial, Int)> = terms.iter().filter(|(m, _)| m[i] == d).cloned().collect();
                domain_key_level(&part, i + 1, n)
            })
            .collect(),
    )
}

fn domain_key(f: &PolyElement<IntegerRing>, n: usize) -> DomainKey {
    let terms: Vec<(Monomial, Int)> = f.terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect();
    domain_key_level(&terms, 0, n)
}

impl Poly {
    fn param_ring_rep(&self, terms: Terms<Rat>, n_gens: usize) -> Poly {
        let ring = rational_ring(internal_names(n_gens, self.domain.params.len()));
        Poly {
            rep: ring.from_dict(terms),
            gens: self.gens[self.gens.len() - n_gens..].to_vec(),
            domain: self.domain.clone(),
        }
    }

    /// The polynomial divided by its leading coefficient (`monic`; over a
    /// polynomial domain only when that coefficient is a number).
    pub fn monic(&self) -> PolyResult<Poly> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        let lc: Rat = self.rep.LC();
        if !self.domain.params.is_empty() {
            let split = self.split();
            let lead = split.values().next_back().expect("a nonzero polynomial");
            let zero: Monomial = vec![0; self.domain.params.len()];
            if lead.len() != 1 || !lead.contains_key(&zero) {
                return Err(polys_error(
                    PolysErrorKind::CoercionFailed,
                    format!("monic of {} needs the fraction field (not ported yet)", self.to_str()?),
                ));
            }
        }
        let integral: Option<bool> = if lc != Rat::ONE { Some(false) } else { None };
        Ok(self.same_ring(self.rep.exquo_ground(&lc)?, integral))
    }

    /// The content (`content`; over `ZZ` the gcd of the coefficients, over
    /// `QQ` one).
    pub fn content(&self) -> PolyResult<Basic> {
        if !self.domain.params.is_empty() {
            return Err(polys_error(
                PolysErrorKind::OperationNotSupported,
                "the content over a polynomial domain is not ported yet".to_string(),
            ));
        }
        let values: Vec<Rat> = self.rep.terms.values().cloned().collect();
        if values.is_empty() {
            return Ok(Basic::zero());
        }
        if self.domain.integral {
            return Ok(Basic::integer(terms_gcd(&values)));
        }
        Ok(Basic::one())
    }

    /// `(content, primitive part)`.
    pub fn primitive(&self) -> PolyResult<(Basic, Poly)> {
        let c: Basic = self.content()?;
        match c.as_rational() {
            Some(v) if !(v.is_zero() || *v == Rat::ONE) => {
                let q = self.rep.exquo_ground(v)?;
                Ok((c.clone(), self.same_ring(q, None)))
            }
            _ => Ok((c, self.clone())),
        }
    }

    /// The polynomial with the generator of index `i` set to `value`.
    pub fn eval(&self, i: usize, value: &Basic) -> PolyResult<PolyOrExpr> {
        if i >= self.gens.len() {
            return Err(polynomial_error(format!(
                "-{} <= gen < {} expected, got {}",
                self.gens.len(),
                self.gens.len(),
                i
            )));
        }
        let mut rule: std::collections::HashMap<Basic, Basic> = std::collections::HashMap::new();
        if self.gens[i] != *value {
            rule.insert(self.gens[i].clone(), value.clone());
        }
        let result: Basic = expand_default(&crate::core::function::xreplace_evaluated(&self.as_expr()?, &rule)?)?;
        let mut rest: Vec<Basic> = self.gens.clone();
        rest.remove(i);
        if rest.is_empty() {
            return Ok(PolyOrExpr::Expr(result));
        }
        Ok(PolyOrExpr::Poly(Poly::new(&result, &rest)?))
    }

    fn drop_first(&self, value: &PolyElement<RationalField>) -> PolyResult<PolyOrExpr> {
        let n: usize = self.gens.len();
        if n == 1 {
            let terms: BTreeMap<Monomial, Rat> =
                value.terms.iter().map(|(m, c)| (m[1..].to_vec(), c.clone())).collect();
            return Ok(PolyOrExpr::Expr(self.coefficient_expr(&terms)?));
        }
        let terms: Terms<Rat> = value.terms.iter().map(|(m, c)| (m[1..].to_vec(), c.clone())).collect();
        Ok(PolyOrExpr::Poly(self.param_ring_rep(terms, n - 1)))
    }

    /// The resultant in the first generator.
    pub fn resultant(&self, other: &Poly) -> PolyResult<PolyOrExpr> {
        let (a, b) = self.unified(other)?;
        let value = resultant_first(&a.rep, &b.rep, &a.names(), a.domain.integral);
        a.drop_first(&value)
    }

    /// `(-1)**(n (n - 1)/2) Res(f, f')/lc(f)` in the first generator.
    pub fn discriminant(&self) -> PolyResult<PolyOrExpr> {
        let n: i64 = self.degree(0)?;
        if n < 1 {
            return Ok(PolyOrExpr::Expr(Basic::zero()));
        }
        let mut value = resultant_first(&self.rep, &self.rep.diff(0), &self.names(), self.domain.integral);
        let lc_terms: Terms<Rat> = self
            .rep
            .terms
            .iter()
            .filter(|(m, _)| i64::from(m[0]) == n)
            .map(|(m, c)| {
                let mut key: Monomial = vec![0];
                key.extend_from_slice(&m[1..]);
                (key, c.clone())
            })
            .collect();
        value = value.exquo(&self.rep.ring.from_dict(lc_terms))?;
        if (n * (n - 1) / 2) % 2 == 1 {
            value = value.neg();
        }
        self.drop_first(&value)
    }

    fn with_int_coefficients(&self, f: &PolyElement<IntegerRing>) -> Poly {
        self.same_ring(
            self.rep
                .ring
                .from_dict(f.terms.iter().map(|(m, c)| (m.clone(), rat_from_int(c))).collect()),
            None,
        )
    }

    fn coefficient_of_full(&self, f: &Terms<Rat>) -> PolyResult<Basic> {
        let n: usize = self.gens.len();
        let terms: BTreeMap<Monomial, Rat> = f.iter().map(|(m, c)| (m[n..].to_vec(), c.clone())).collect();
        self.coefficient_expr(&terms)
    }

    /// The square-free decomposition `(coeff, [(g, k)])` (`sqf_list`).
    pub fn sqf_list(&self) -> PolyResult<(Basic, Vec<(Poly, usize)>)> {
        use super::multisqf::{sqf_list, IntegerSqfDomain, ParameterSqfDomain, RationalSqfDomain};
        use super::sqfreetools::dup_sqf_list;
        let n: usize = self.gens.len();
        let names: Vec<String> = self.rep.ring.names.clone();
        if self.domain.params.is_empty() && n == 1 {
            let f: Vec<Rat> = self.univariate_rational()?;
            if self.domain.integral {
                let ints: Vec<Int> = f.iter().map(numer).collect();
                let (coefficient, factors) = dup_sqf_list(&ints, &ZZ);
                return Ok((
                    Basic::integer(coefficient),
                    factors
                        .into_iter()
                        .map(|(g, k)| {
                            (
                                self.from_univariate(&g.iter().map(rat_from_int).collect::<Vec<_>>(), true),
                                k,
                            )
                        })
                        .collect(),
                ));
            }
            let (coefficient, factors) = dup_sqf_list(&f, &QQ);
            return Ok((
                Basic::from_rational(coefficient),
                factors
                    .into_iter()
                    .map(|(g, k)| (self.from_univariate(&g, false), k))
                    .collect(),
            ));
        }
        if self.domain.integral {
            let to_int = |ring: &Arc<PolyRing<IntegerRing>>| -> PolyElement<IntegerRing> {
                ring.from_dict(self.rep.terms.iter().map(|(m, c)| (m.clone(), numer(c))).collect())
            };
            let (cz, fz) = if self.domain.params.is_empty() {
                let d = IntegerSqfDomain::new(names);
                sqf_list(&to_int(&d.domain.ring), &d, false)
            } else {
                let d = ParameterSqfDomain::new(
                    SparsePolynomialRing::sympy(ZZ, names[n..].to_vec()),
                    names[..n].to_vec(),
                );
                sqf_list(&to_int(d.iterated.ring()), &d, false)
            };
            let c: Terms<Rat> = cz.terms.iter().map(|(m, c)| (m.clone(), rat_from_int(c))).collect();
            return Ok((
                self.coefficient_of_full(&c)?,
                fz.iter().map(|(g, k)| (self.with_int_coefficients(g), *k)).collect(),
            ));
        }
        let (cq, fq) = if self.domain.params.is_empty() {
            let d = RationalSqfDomain::new(names);
            let f = d.domain.ring.from_dict(self.rep.terms.clone());
            sqf_list(&f, &d, false)
        } else {
            let d = ParameterSqfDomain::new(
                SparsePolynomialRing::sympy(QQ, names[n..].to_vec()),
                names[..n].to_vec(),
            );
            let f = d.iterated.ring().from_dict(self.rep.terms.clone());
            sqf_list(&f, &d, false)
        };
        Ok((
            self.coefficient_of_full(&cq.terms)?,
            fq.iter().map(|(g, k)| (self.from_rationals(g), *k)).collect(),
        ))
    }

    /// The irreducible factors with their multiplicities and the leading
    /// constant (`factor_list`), sorted as SymPy sorts them.
    pub fn factor_list(&self) -> PolyResult<(Basic, Vec<(Poly, usize)>)> {
        if self.is_zero() {
            return Ok((Basic::zero(), Vec::new()));
        }
        if self.domain.params.is_empty() {
            let (coefficient, factors) = factor_all(&self.rep)?;
            return Ok((
                Basic::from_rational(coefficient),
                factors
                    .iter()
                    .map(|(f, k)| (self.with_int_coefficients(f), *k))
                    .collect(),
            ));
        }
        let n: usize = self.gens.len();
        let names: Vec<String> = self.names();
        let exponents: Vec<u32> = (0..n)
            .map(|i| self.rep.terms.keys().map(|m| m[i]).min().expect("terms"))
            .collect();
        let shifted: Terms<Rat> = self
            .rep
            .terms
            .iter()
            .map(|(m, c)| {
                let key: Monomial = m
                    .iter()
                    .enumerate()
                    .map(|(i, e)| if i < n { e - exponents[i] } else { *e })
                    .collect();
                (key, c.clone())
            })
            .collect();
        let (f, content): (PolyElement<RationalField>, Basic) = if self.domain.integral {
            let z = IteratedPolynomialRing::new(
                SparsePolynomialRing::sympy(ZZ, names[n..].to_vec()),
                names[..n].to_vec(),
            );
            let mut fz: PolyElement<IntegerRing> = z
                .ring()
                .from_dict(shifted.iter().map(|(m, c)| (m.clone(), numer(c))).collect());
            let content_z = z.ground_content(&fz);
            if !z.params.is_one(&content_z) {
                fz = fz.exquo(&z.embed_params(&content_z))?;
            }
            let c: BTreeMap<Monomial, Rat> = content_z
                .terms
                .iter()
                .map(|(m, c)| (m.clone(), rat_from_int(c)))
                .collect();
            (
                self.rep
                    .ring
                    .from_dict(fz.terms.iter().map(|(m, c)| (m.clone(), rat_from_int(c))).collect()),
                self.coefficient_expr(&c)?,
            )
        } else {
            let q = IteratedPolynomialRing::new(
                SparsePolynomialRing::sympy(QQ, names[n..].to_vec()),
                names[..n].to_vec(),
            );
            let mut fq: PolyElement<RationalField> = q.ring().from_dict(shifted);
            let content_q = q.ground_content(&fq);
            if !q.params.is_one(&content_q) {
                fq = fq.exquo(&q.embed_params(&content_q))?;
            }
            let c: BTreeMap<Monomial, Rat> = content_q.terms.clone().into_iter().collect();
            (self.rep.ring.from_dict(fq.terms.clone()), self.coefficient_expr(&c)?)
        };
        let (coefficient, factors) = factor_all(&f)?;
        let mut result: mf::Factors = factors;
        let ring_z: Arc<PolyRing<IntegerRing>> = PolyRing::new(ZZ, self.rep.ring.names.clone(), MonomialOrder::Lex)?;
        let ngens: usize = ring_z.names.len();
        for i in (0..n).rev() {
            if exponents[i] != 0 {
                let monomial: Monomial = (0..ngens).map(|k| u32::from(k == i)).collect();
                result.insert(0, (ring_z.term(monomial, Int::ONE), exponents[i] as usize));
            }
        }
        result.sort_by_cached_key(|(g, k)| {
            let first: i64 = g.terms.keys().map(|m| i64::from(m[0])).max().unwrap_or(-1);
            (first + 1, *k, domain_key(g, n))
        });
        Ok((
            core_mul(&[Basic::from_rational(coefficient), content])?,
            result
                .iter()
                .map(|(g, k)| (self.with_int_coefficients(g), *k))
                .collect(),
        ))
    }
}

// ---------------------------------------------------------------------
// the functions of the module (on expressions)

/// `Poly(expr, *gens)`.
pub fn poly(expr: &Basic, gens: &[Basic]) -> PolyResult<Poly> {
    Poly::new(expr, gens)
}

/// The degree of `f` in the generator `gen` (`degree(f, gen)`).
pub fn degree(f: &Basic, gen: &Basic) -> PolyResult<i64> {
    Poly::new(f, std::slice::from_ref(gen))?.degree(0)
}

/// The degree of `f` in its generator of index `i` (`degree(f, i)`).
pub fn degree_index(f: &Basic, i: usize) -> PolyResult<i64> {
    Poly::new(f, &[])?.degree(i)
}

#[allow(non_snake_case)]
pub fn LC(f: &Basic, gens: &[Basic]) -> PolyResult<Basic> {
    Poly::new(f, gens)?.LC()
}

/// The generators of `f` and `g` together (`_common_gens`).
pub(crate) fn common_gens(f: &Basic, g: &Basic) -> PolyResult<Vec<Basic>> {
    Ok(decompose_without_gens_many(&[f.clone(), g.clone()], true)?.1)
}

fn numbers_only(f: &Basic, g: &Basic, gens: &[Basic]) -> PolyResult<bool> {
    Ok(gens.is_empty() && common_gens(f, g)?.is_empty())
}

fn integer_lcm(a: &Int, b: &Int) -> Int {
    if a.is_zero() || b.is_zero() {
        return Int::ZERO;
    }
    abs(&(a * b)) / gcd(a, b)
}

fn first_poly(f: &Basic, g: &Basic, gens: &[Basic]) -> PolyResult<Poly> {
    if !gens.is_empty() {
        return Poly::new(f, gens);
    }
    Poly::new(f, &common_gens(f, g)?)
}

/// The gcd of two polynomial expressions (`gcd`): of two rationals the gcd
/// of the numerators over the lcm of the denominators, of other numbers 1.
pub fn gcd_expr(f: &Basic, g: &Basic, gens: &[Basic]) -> PolyResult<Basic> {
    if numbers_only(f, g, gens)? {
        if let (Some(a), Some(b)) = (f.as_rational(), g.as_rational()) {
            return Ok(Basic::from_rational(rat_from_parts(
                gcd(&numer(a), &numer(b)),
                integer_lcm(&denom(a), &denom(b)),
            )));
        }
        return Ok(Basic::one());
    }
    let pf: Poly = first_poly(f, g, gens)?;
    let pg: Poly = Poly::new(g, &pf.gens)?;
    pf.gcd(&pg)?.as_expr()
}

/// The lcm of two polynomial expressions (`lcm`): of two rationals the lcm
/// of the numerators over the gcd of the denominators, of other numbers
/// their product.
pub fn lcm_expr(f: &Basic, g: &Basic, gens: &[Basic]) -> PolyResult<Basic> {
    if numbers_only(f, g, gens)? {
        if let (Some(a), Some(b)) = (f.as_rational(), g.as_rational()) {
            return Ok(Basic::from_rational(rat_from_parts(
                integer_lcm(&numer(a), &numer(b)),
                gcd(&denom(a), &denom(b)),
            )));
        }
        return core_mul(&[f.clone(), g.clone()]);
    }
    let pf: Poly = first_poly(f, g, gens)?;
    let pg: Poly = Poly::new(g, &pf.gens)?;
    pf.lcm(&pg)?.as_expr()
}

/// The gcd of a list (`0` of an empty list), folding [`gcd_expr`].
pub fn gcd_list(seq: &[Basic], gens: &[Basic]) -> PolyResult<Basic> {
    let Some(first) = seq.first() else {
        return Ok(Basic::zero());
    };
    let mut result: Basic = first.clone();
    for g in &seq[1..] {
        result = gcd_expr(&result, g, gens)?;
    }
    Ok(result)
}

/// The lcm of a list (`1` of an empty list), folding [`lcm_expr`].
pub fn lcm_list(seq: &[Basic], gens: &[Basic]) -> PolyResult<Basic> {
    let Some(first) = seq.first() else {
        return Ok(Basic::one());
    };
    let mut result: Basic = first.clone();
    for g in &seq[1..] {
        result = lcm_expr(&result, g, gens)?;
    }
    Ok(result)
}

fn either_expr(r: PolyOrExpr) -> PolyResult<Basic> {
    match r {
        PolyOrExpr::Poly(p) => p.as_expr(),
        PolyOrExpr::Expr(e) => Ok(e),
    }
}

/// The resultant in the first generator, as an expression.
pub fn resultant_expr(f: &Basic, g: &Basic, gens: &[Basic]) -> PolyResult<Basic> {
    let pf: Poly = Poly::new(f, gens)?;
    let pg: Poly = Poly::new(g, if gens.is_empty() { &pf.gens } else { gens })?;
    either_expr(pf.resultant(&pg)?)
}

/// The discriminant in the first generator, as an expression.
pub fn discriminant_expr(f: &Basic, gens: &[Basic]) -> PolyResult<Basic> {
    either_expr(Poly::new(f, gens)?.discriminant()?)
}

/// SymPy's `sring` of several expressions without generators, the
/// coefficients rational (`_joint_rings`; `CoercionFailed` otherwise).
pub(crate) fn joint_rings(exprs: &[Basic], expand: bool) -> PolyResult<(Vec<Basic>, Vec<BTreeMap<Monomial, Rat>>)> {
    let (decomposed, gens) = decompose_without_gens_many(exprs, expand)?;
    if gens.is_empty() {
        return Ok((gens, exprs.iter().map(|_| BTreeMap::new()).collect()));
    }
    let mut result: Vec<BTreeMap<Monomial, Rat>> = Vec::new();
    for terms in decomposed {
        let mut rep: BTreeMap<Monomial, Rat> = BTreeMap::new();
        for (m, cs) in terms {
            let c: Basic = core_add(&cs)?;
            let Some(v) = c.as_rational() else {
                return Err(polys_error(
                    PolysErrorKind::CoercionFailed,
                    format!("the domain of {} (ZZ_I, RR, EX) is not ported yet", c),
                ));
            };
            let value: Rat = rep.remove(&m).unwrap_or(Rat::ZERO) + v;
            if !value.is_zero() {
                rep.insert(m, value);
            }
        }
        result.push(rep);
    }
    Ok((gens, result))
}

fn denominators_lcm_of(f: &BTreeMap<Monomial, Rat>) -> Int {
    let mut result: Int = Int::ONE;
    for c in f.values() {
        let d: Int = denom(c);
        result = &result * &d / gcd(&result, &d);
    }
    result
}

/// `f/g` in lowest terms over `ZZ` (`_cancel_reps`), the leading
/// coefficient of the denominator positive.
fn cancel_reps(
    f: &BTreeMap<Monomial, Rat>,
    g: &BTreeMap<Monomial, Rat>,
    n: usize,
) -> PolyResult<(PolyElement<IntegerRing>, PolyElement<IntegerRing>)> {
    let z: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::sympy(ZZ, internal_names(n, 0));
    if f.is_empty() {
        return Ok((z.ring.zero(), z.ring.one()));
    }
    let (cq, cp) = (denominators_lcm_of(f), denominators_lcm_of(g));
    let to_int = |h: &BTreeMap<Monomial, Rat>, c: &Int| -> PolyElement<IntegerRing> {
        z.ring.from_dict(
            h.iter()
                .map(|(m, v)| (m.clone(), numer(&(v * rat_from_int(c)))))
                .collect(),
        )
    };
    let (big_f, big_g) = (to_int(f, &cq), to_int(g, &cp));
    let h = Domain::gcd(&z, &big_f, &big_g);
    let (mut p, mut q) = (big_f.exquo(&h)?, big_g.exquo(&h)?);
    let common: Int = gcd(&cp, &cq);
    p = p.mul_ground(&(&cp / &common));
    q = q.mul_ground(&(&cq / &common));
    if is_negative(&q.LC()) {
        p = p.neg();
        q = q.neg();
    }
    Ok((p, q))
}

fn expr_of(f: &PolyElement<IntegerRing>, gens: &[Basic]) -> PolyResult<Basic> {
    let mut parts: Vec<Basic> = Vec::new();
    for (m, c) in f.terms.iter().rev() {
        let mut factors: Vec<Basic> = vec![Basic::integer(c.clone())];
        for (g, k) in gens.iter().zip(m) {
            if *k > 0 {
                factors.push(core_pow(g, &Basic::int(i64::from(*k)))?);
            }
        }
        parts.push(core_mul(&factors)?);
    }
    core_add(&parts)
}

fn not_invertible() -> PolyError {
    polys_error(PolysErrorKind::NotInvertible, "zero divisor".to_string())
}

fn modular_inverse(a: &Int, m: &Int) -> PolyResult<Basic> {
    // Python's pow(a, -1, m): the result in [0, |m|) for m > 0, (m, 0] for m < 0
    if m.is_zero() {
        return Err(not_invertible());
    }
    let modulus: Int = abs(m);
    let Some(mut r) = crate::numbers::mod_inverse(a, &modulus) else {
        return Err(not_invertible());
    };
    if is_negative(m) && !r.is_zero() {
        r -= &modulus;
    }
    Ok(Basic::integer(r))
}

/// The inverse of `f` modulo `g` (`invert`): of two integers the modular
/// inverse, of univariate polynomials over `QQ` the polynomial `s` with
/// `s*f = 1 (mod g)`; `NotInvertible` when they are not coprime.
pub fn invert(f: &Basic, g: &Basic) -> PolyResult<Basic> {
    if let (
        Head::Rational {
            value: a,
            integer_class: true,
        },
        Head::Rational {
            value: b,
            integer_class: true,
        },
    ) = (f.head(), g.head())
    {
        return modular_inverse(&numer(a), &numer(b));
    }
    let (gens, reps) = joint_rings(&[f.clone(), g.clone()], true)?;
    if gens.len() > 1 {
        return Err(value_error("univariate polynomial expected".to_string()));
    }
    if gens.is_empty() {
        let (Some(a), Some(b)) = (f.as_rational(), g.as_rational()) else {
            return Err(crate::error::not_ported(
                "invert of numbers in ZZ_I, QQ_I, RR, CC or EX",
            ));
        };
        if a.is_int() && b.is_int() {
            return modular_inverse(&numer(a), &numer(b));
        }
        return Err(not_invertible());
    }
    let dense = |terms: &BTreeMap<Monomial, Rat>| -> Vec<Rat> {
        let n: u32 = terms.keys().map(|m| m[0]).max().unwrap_or(0);
        (0..=n)
            .rev()
            .map(|k| terms.get(&vec![k]).cloned().unwrap_or(Rat::ZERO))
            .collect()
    };
    let h: Vec<Rat> = match super::euclidtools::dup_invert(&dense(&reps[0]), &dense(&reps[1]), &QQ) {
        Ok(h) => h,
        Err(PolyError::Value(_)) | Err(PolyError::ZeroDivision(_)) => return Err(not_invertible()),
        Err(e) => return Err(e),
    };
    let n: usize = h.len().saturating_sub(1);
    let mut parts: Vec<Basic> = Vec::new();
    for (i, c) in h.iter().enumerate() {
        if !c.is_zero() {
            parts.push(core_mul(&[
                Basic::from_rational(c.clone()),
                core_pow(&gens[0], &Basic::int((n - i) as i64))?,
            ])?);
        }
    }
    core_add(&parts)
}

/// `p/q` in lowest terms as a pair of expanded polynomials, the leading
/// coefficient of the denominator positive (`cancel_fraction`).
pub fn cancel_fraction(p: &Basic, q: &Basic) -> PolyResult<(Basic, Basic)> {
    let (gens, reps) = joint_rings(&[expand_default(p)?, expand_default(q)?], true)?;
    if gens.is_empty() {
        return Ok((
            expand_default(&core_mul(&[p.clone(), core_pow(q, &Basic::negative_one())?])?)?,
            Basic::one(),
        ));
    }
    if reps[1].is_empty() {
        return Err(crate::error::zero_division(&format!("cancel of {}/0", p)));
    }
    let (big_p, big_q) = cancel_reps(&reps[0], &reps[1], gens.len())?;
    Ok((expr_of(&big_p, &gens)?, expr_of(&big_q, &gens)?))
}

/// A rational function in lowest terms `P/Q` (`cancel`), the generators
/// every non-numeric factor, or `gens` with the other factors in the
/// coefficients; the domains `ZZ_I`, `RR` and `EX` are `NotImplemented`.
pub fn cancel(f: &Basic, gens: &[Basic]) -> PolyResult<Basic> {
    let e: Basic = crate::simplify::signsimp::signsimp(f, true)?;
    if e.is_number_class() || crate::core::expr::is_false(&e, crate::core::assumptions::Predicate::commutative) {
        return Ok(e);
    }
    let factored: Basic = crate::core::exprtools::factor_terms(&e, true, false, false, true)?;
    let (p, q) = crate::core::expr::as_numer_denom(&factored)?;
    let (p, q) = (expand_default(&p)?, expand_default(&q)?);
    if !gens.is_empty() {
        return cancel_with_gens(&p, &q, &factored, gens);
    }
    let (joint, reps) = match joint_rings(&[p.clone(), q.clone()], true) {
        Ok(r) => r,
        Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
            let (ggens, greps) = match joint_rings_gaussian(&[p, q], true) {
                Ok(r) => r,
                Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
                    return Err(crate::error::not_ported("cancel over RR or EX"))
                }
                Err(e) => return Err(e),
            };
            if ggens.is_empty() {
                return expand_default(&factored);
            }
            let (gp, gq) = cancel_gaussian(&greps[0], &greps[1], ggens.len())?;
            return core_mul(&[
                gaussian_expr_of(&gp, &ggens)?,
                core_pow(&gaussian_expr_of(&gq, &ggens)?, &Basic::negative_one())?,
            ]);
        }
        Err(e) => return Err(e),
    };
    if joint.is_empty() {
        return expand_default(&factored);
    }
    let (big_p, big_q) = cancel_reps(&reps[0], &reps[1], joint.len())?;
    core_mul(&[
        expr_of(&big_p, &joint)?,
        core_pow(&expr_of(&big_q, &joint)?, &Basic::negative_one())?,
    ])
}

fn cancel_with_gens(p: &Basic, q: &Basic, factored: &Basic, gens: &[Basic]) -> PolyResult<Basic> {
    let (joint, reps) = match joint_rings(&[p.clone(), q.clone()], true) {
        Ok(r) => r,
        Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
            return Err(crate::error::not_ported("cancel with generators over ZZ_I, RR or EX"))
        }
        Err(e) => return Err(e),
    };
    let mut joint_symbols: Vec<Basic> = Vec::new();
    for j in &joint {
        joint_symbols.extend(j.free_symbols());
    }
    if gens
        .iter()
        .any(|g| !joint.contains(g) && g.free_symbols().iter().any(|s| joint_symbols.contains(s)))
    {
        return Err(crate::error::not_ported(
            "cancel by a generator which is not one of the factors",
        ));
    }
    if joint.is_empty() {
        return expand_default(factored);
    }
    let (mut big_p, mut big_q) = cancel_reps(&reps[0], &reps[1], joint.len())?;
    let gen_index: Vec<Option<usize>> = gens.iter().map(|g| joint.iter().position(|j| j == g)).collect();
    let param_index: Vec<usize> = (0..joint.len()).filter(|i| !gens.contains(&joint[*i])).collect();
    let split = |m: &Monomial| -> (Vec<u32>, Vec<u32>) {
        (
            gen_index.iter().map(|i| i.map(|i| m[i]).unwrap_or(0)).collect(),
            param_index.iter().map(|&i| m[i]).collect(),
        )
    };
    let lead: &Monomial = big_q
        .terms
        .keys()
        .max_by_key(|m| split(m))
        .expect("a nonzero denominator");
    if is_negative(&big_q.terms[lead]) {
        big_p = big_p.neg();
        big_q = big_q.neg();
    }
    let expr_of_split = |f: &PolyElement<IntegerRing>| -> PolyResult<Basic> {
        let mut grouped: Vec<(Vec<u32>, Vec<Basic>)> = Vec::new();
        for (m, c) in &f.terms {
            let (gm, pm) = split(m);
            let mut factors: Vec<Basic> = vec![Basic::integer(c.clone())];
            for (i, k) in param_index.iter().zip(&pm) {
                if *k > 0 {
                    factors.push(core_pow(&joint[*i], &Basic::int(i64::from(*k)))?);
                }
            }
            let term: Basic = core_mul(&factors)?;
            match grouped.iter_mut().find(|(g, _)| *g == gm) {
                Some((_, cs)) => cs.push(term),
                None => grouped.push((gm, vec![term])),
            }
        }
        let mut parts: Vec<Basic> = Vec::new();
        for (gm, cs) in grouped {
            let mut factors: Vec<Basic> = vec![core_add(&cs)?];
            for (g, k) in gens.iter().zip(&gm) {
                if *k > 0 {
                    factors.push(core_pow(g, &Basic::int(i64::from(*k)))?);
                }
            }
            parts.push(core_mul(&factors)?);
        }
        core_add(&parts)
    };
    core_mul(&[
        expr_of_split(&big_p)?,
        core_pow(&expr_of_split(&big_q)?, &Basic::negative_one())?,
    ])
}

// ---------------------------------------------------------------------
// factorisation of expressions (`factor`, `factor_list`, `sqf`, ...)

/// A factor: a polynomial, or an expression which is not one (a number
/// base, a constant without generators).
#[derive(Debug, Clone, PartialEq)]
pub enum FactorBase {
    Poly(Poly),
    Expr(Basic),
}

impl FactorBase {
    fn expr(&self) -> PolyResult<Basic> {
        match self {
            FactorBase::Poly(p) => p.as_expr(),
            FactorBase::Expr(e) => Ok(e.clone()),
        }
    }
}

type SymbolicFactors = Vec<(FactorBase, Basic)>;

/// Whether the factorisation is into irreducible or square-free factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorMethod {
    Factor,
    Sqf,
}

/// Whether `v` is `a + b*I` with numbers `a`, `b` (`_pure_complex`).
fn pure_complex(v: &Basic) -> bool {
    let (_, t) = crate::core::expr::as_coeff_add(v);
    if t.is_number_class() && t.as_rational().is_some_and(|r| r.is_zero()) {
        return false;
    }
    let (_, i) = crate::core::expr::as_coeff_mul(&t);
    matches!(i.head(), Head::ImaginaryUnit)
}

fn factors_product(factors: &[(FactorBase, Basic)]) -> PolyResult<Basic> {
    let mut parts: Vec<Basic> = Vec::new();
    for (g, k) in factors {
        parts.push(core_pow(&g.expr()?, k)?);
    }
    core_mul(&parts)
}

fn product_of(bases: &[FactorBase]) -> PolyResult<FactorBase> {
    let mut result: FactorBase = bases[0].clone();
    for b in &bases[1..] {
        result = match (&result, b) {
            (FactorBase::Poly(p), FactorBase::Poly(q)) => FactorBase::Poly(p.mul(q)?),
            _ => FactorBase::Expr(core_mul(&[result.expr()?, b.expr()?])?),
        };
    }
    Ok(result)
}

/// `(0, value, "")` for a rational exponent, `(1, 0, str)` otherwise
/// (`_exponent_key`).
fn exponent_key(k: &Basic) -> (u8, Rat, String) {
    match k.as_rational() {
        Some(v) => (0, v.clone(), String::new()),
        None => (1, Rat::ZERO, k.to_string()),
    }
}

/// `poly.rep.to_list()` as SymPy compares it (`_rep_key`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum RepKey {
    Number(Rat),
    Coefficient(usize, Vec<(Monomial, Rat)>),
    Level(Vec<RepKey>),
}

fn rep_level(terms: &[(Monomial, Rat)], i: usize, n: usize, params: bool) -> RepKey {
    if i == n {
        if params {
            let mut ordered: Vec<(Monomial, Rat)> = terms.iter().map(|(m, c)| (m[n..].to_vec(), c.clone())).collect();
            ordered.sort();
            ordered.reverse();
            return RepKey::Coefficient(ordered.len(), ordered);
        }
        return RepKey::Number(terms.iter().fold(Rat::ZERO, |s, (_, c)| s + c));
    }
    if terms.is_empty() {
        return RepKey::Level(Vec::new());
    }
    let top: u32 = terms.iter().map(|(m, _)| m[i]).max().expect("terms");
    let part = |d: u32| -> Vec<(Monomial, Rat)> { terms.iter().filter(|(m, _)| m[i] == d).cloned().collect() };
    if i + 1 == n && !params {
        return RepKey::Level(
            (0..=top)
                .rev()
                .map(|d| RepKey::Number(part(d).iter().fold(Rat::ZERO, |s, (_, c)| s + c)))
                .collect(),
        );
    }
    RepKey::Level((0..=top).rev().map(|d| rep_level(&part(d), i + 1, n, params)).collect())
}

fn rep_key(p: &Poly) -> RepKey {
    let terms: Vec<(Monomial, Rat)> = p.rep.terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect();
    rep_level(&terms, 0, p.gens.len(), !p.domain.params.is_empty())
}

fn rep_length(p: &Poly) -> usize {
    if p.is_zero() {
        return 0;
    }
    p.degree(0).map(|d| d.max(0) as usize + 1).unwrap_or(0)
}

/// SymPy's order of the factors (`_sorted_factors`).
fn sorted_factors(mut factors: Vec<(Poly, Basic)>, method: FactorMethod) -> Vec<(Poly, Basic)> {
    type Key = (usize, usize, (u8, Rat, String), String, RepKey);
    let key = |(p, k): &(Poly, Basic)| -> Key {
        (
            rep_length(p),
            p.gens.len(),
            exponent_key(k),
            p.domain.name(),
            rep_key(p),
        )
    };
    match method {
        FactorMethod::Factor => factors.sort_by_cached_key(key),
        FactorMethod::Sqf => factors.sort_by_cached_key(|item| {
            let (a, b, c, d, e) = key(item);
            (c, a, b, d, e)
        }),
    }
    factors
}

/// The constant and the factors of a product (`_symbolic_factor_list`).
fn symbolic_factor_list(expr: &Basic, gens: &[Basic], method: FactorMethod) -> PolyResult<(Basic, SymbolicFactors)> {
    use crate::core::assumptions::Predicate;
    use crate::core::expr::is_true;
    let mut coefficient: Basic = Basic::one();
    let mut factors: SymbolicFactors = Vec::new();
    let args: Vec<Basic> = if expr.is_mul() {
        expr.args().to_vec()
    } else {
        vec![expr.clone()]
    };
    for arg in args {
        if arg.is_number_class() || pure_complex(&arg) {
            coefficient = core_mul(&[coefficient, arg])?;
            continue;
        }
        let (base, exponent) = if arg.is_pow() {
            let (b, e) = (arg.args()[0].clone(), arg.args()[1].clone());
            if b.is_number_class() && e.is_number_class() {
                coefficient = core_mul(&[coefficient, arg])?;
                continue;
            }
            if b.is_number_class() {
                factors.push((FactorBase::Expr(b), e));
                continue;
            }
            (b, e)
        } else {
            (arg.clone(), Basic::one())
        };
        let (c, mut found): (Basic, SymbolicFactors) = match Poly::new(&base, gens) {
            Err(PolyError::Polys(PolysErrorKind::GeneratorsNeeded, _)) => {
                factors.push((FactorBase::Expr(expand_default(&base)?), exponent));
                continue;
            }
            Err(PolyError::Polys(PolysErrorKind::CoercionFailed, message)) => {
                if !gens.is_empty() || method != FactorMethod::Factor {
                    return Err(polys_error(PolysErrorKind::CoercionFailed, message));
                }
                return Err(crate::error::not_ported("the factorisation over ZZ_I, QQ_I or RR"));
            }
            Err(e) => return Err(e),
            Ok(p) => {
                let (c, poly_factors) = match method {
                    FactorMethod::Factor => p.factor_list()?,
                    FactorMethod::Sqf => p.sqf_list()?,
                };
                (
                    c,
                    poly_factors
                        .into_iter()
                        .map(|(g, k)| (FactorBase::Poly(g), Basic::int(k as i64)))
                        .collect(),
                )
            }
        };
        let exponent_is_integer: bool = matches!(
            exponent.head(),
            Head::Rational {
                integer_class: true,
                ..
            }
        );
        if !(matches!(
            c.head(),
            Head::Rational {
                integer_class: true,
                ..
            }
        ) && c == Basic::one())
        {
            if exponent_is_integer {
                coefficient = core_mul(&[coefficient, core_pow(&c, &exponent)?])?;
            } else if is_true(&c, Predicate::positive) {
                factors.push((FactorBase::Expr(c), exponent.clone()));
            } else {
                found.push((FactorBase::Expr(c), Basic::one()));
            }
        }
        if exponent_is_integer && exponent == Basic::one() {
            factors.extend(found);
        } else if is_true(&exponent, Predicate::integer) {
            for (g, k) in found {
                factors.push((g, core_mul(&[k, exponent.clone()])?));
            }
        } else {
            let mut other: SymbolicFactors = Vec::new();
            for (g, k) in found {
                if is_true(&g.expr()?, Predicate::positive) {
                    factors.push((g, core_mul(&[k, exponent.clone()])?));
                } else {
                    other.push((g, k));
                }
            }
            factors.push((FactorBase::Expr(factors_product(&other)?), exponent));
        }
    }
    if method == FactorMethod::Sqf {
        let mut exponents: Vec<Basic> = Vec::new();
        for (_, k) in &factors {
            if !exponents.contains(k) {
                exponents.push(k.clone());
            }
        }
        exponents.sort_by_cached_key(exponent_key);
        let mut grouped: SymbolicFactors = Vec::new();
        for k in exponents {
            let bases: Vec<FactorBase> = factors
                .iter()
                .filter(|(_, m)| *m == k)
                .map(|(g, _)| g.clone())
                .collect();
            grouped.push((product_of(&bases)?, k));
        }
        factors = grouped;
    }
    let mut collected: SymbolicFactors = Vec::new();
    for (g, k) in factors {
        match collected.iter_mut().find(|(h, _)| *h == g) {
            Some((_, total)) => *total = core_add(&[total.clone(), k])?,
            None => collected.push((g, k)),
        }
    }
    Ok((coefficient, collected))
}

type PolyFactors = Vec<(Poly, Basic)>;

fn factor_lists(f: &Basic, gens: &[Basic], method: FactorMethod) -> PolyResult<(Basic, PolyFactors, PolyFactors)> {
    let together: Basic = super::rationaltools::together(f, false, true)?;
    let (numer, denom) = crate::core::expr::as_numer_denom(&together)?;
    let (cp, fp) = symbolic_factor_list(&numer, gens, method)?;
    let (cq, fq) = symbolic_factor_list(&denom, gens, method)?;
    let as_polys = |fs: SymbolicFactors| -> PolyResult<PolyFactors> {
        let mut out: PolyFactors = Vec::new();
        for (g, k) in fs {
            out.push((
                match g {
                    FactorBase::Poly(p) => p,
                    FactorBase::Expr(e) => Poly::new(&e, gens)?,
                },
                k,
            ));
        }
        Ok(out)
    };
    let (pp, pq) = (as_polys(fp)?, as_polys(fq)?);
    Ok((
        core_mul(&[cp, core_pow(&cq, &Basic::negative_one())?])?,
        sorted_factors(pp, method),
        sorted_factors(pq, method),
    ))
}

type ExprFactors = Vec<(Basic, Basic)>;

fn as_exprs(fs: PolyFactors) -> PolyResult<ExprFactors> {
    let mut out: ExprFactors = Vec::new();
    for (g, k) in fs {
        out.push((g.as_expr()?, k));
    }
    Ok(out)
}

fn polynomial_list(f: &Basic, gens: &[Basic], method: FactorMethod) -> PolyResult<(Basic, ExprFactors)> {
    let (c, fp, fq) = factor_lists(f, gens, method)?;
    if !fq.is_empty() {
        return Err(polynomial_error(format!("a polynomial expected, got {}", f)));
    }
    Ok((c, as_exprs(fp)?))
}

/// The constant and the irreducible factors with their multiplicities of
/// a polynomial expression, sorted as SymPy sorts them (`factor_list`).
pub fn factor_list(f: &Basic, gens: &[Basic]) -> PolyResult<(Basic, ExprFactors)> {
    polynomial_list(f, gens, FactorMethod::Factor)
}

/// The constant and the square-free factors (`sqf_list`).
pub fn sqf_list(f: &Basic, gens: &[Basic]) -> PolyResult<(Basic, ExprFactors)> {
    polynomial_list(f, gens, FactorMethod::Sqf)
}

/// The constant and the factors of the numerator and of the denominator
/// (`factor_list(f, frac=True)`, `sqf_list(f, frac=True)`).
pub fn factor_list_fraction(
    f: &Basic,
    gens: &[Basic],
    method: FactorMethod,
) -> PolyResult<(Basic, ExprFactors, ExprFactors)> {
    let (c, fp, fq) = factor_lists(f, gens, method)?;
    Ok((c, as_exprs(fp)?, as_exprs(fq)?))
}

fn symbolic_factor(e: &Basic, gens: &[Basic], fraction: bool, method: FactorMethod) -> PolyResult<Basic> {
    let together: Basic = super::rationaltools::together(e, false, fraction)?;
    let (coefficient, factors) = symbolic_factor_list(&together, gens, method)?;
    crate::core::mul::keep_coeff(&coefficient, &factors_product(&factors)?)
}

/// The expression as a product of irreducible factors over `ZZ`/`QQ`
/// (`factor`); with `deep` the subexpressions too. `EX`, `ZZ_I`, `RR` and
/// non-commutative expressions are `NotImplemented`.
pub fn factor(e: &Basic, gens: &[Basic], deep: bool, fraction: bool) -> PolyResult<Basic> {
    if deep {
        let attempt = |b: &Basic| -> PolyResult<Basic> {
            let factored: Basic = factor(b, gens, false, true)?;
            Ok(if factored.is_mul() || factored.is_pow() {
                factored
            } else {
                b.clone()
            })
        };
        let result: Basic = crate::core::function::replace_where(e, &|b: &Basic| !b.args().is_empty(), &attempt)?;
        let mut partials: std::collections::HashMap<Basic, Basic> = std::collections::HashMap::new();
        for node in result.atoms(Some(&|n: &Basic| n.is_mul() || n.is_add())) {
            let factored: Basic = factor(&node, gens, false, true)?;
            if (factored.is_mul() || factored.is_pow()) && factored != node {
                partials.insert(node, factored);
            }
        }
        if partials.is_empty() {
            return Ok(result);
        }
        return crate::core::function::xreplace_evaluated(&result, &partials);
    }
    if crate::core::expr::is_false(e, crate::core::assumptions::Predicate::commutative) {
        return Err(crate::error::not_ported("factor_nc"));
    }
    match symbolic_factor(e, gens, fraction, FactorMethod::Factor) {
        Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
            Err(crate::error::not_ported("factor over EX, ZZ_I, QQ_I or RR"))
        }
        r => r,
    }
}

/// The square-free factorisation of an expression (`sqf`).
pub fn sqf(e: &Basic, gens: &[Basic], fraction: bool) -> PolyResult<Basic> {
    symbolic_factor(e, gens, fraction, FactorMethod::Sqf)
}

// ---------------------------------------------------------------------
// the Gaussian domains ZZ_I and QQ_I in cancel

use super::domains::gaussiandomains::{GaussianInteger, GaussianRational, QQ_I, ZZ_I};

type GaussianTerms = BTreeMap<Monomial, GaussianRational>;

/// `_joint_rings` with Gaussian coefficients `a + b*I`
/// (`_joint_rings_gaussian`); `CoercionFailed` for other coefficients.
fn joint_rings_gaussian(exprs: &[Basic], expand: bool) -> PolyResult<(Vec<Basic>, Vec<GaussianTerms>)> {
    let (decomposed, gens) = decompose_without_gens_many(exprs, expand)?;
    if gens.is_empty() {
        return Ok((gens, exprs.iter().map(|_| GaussianTerms::new()).collect()));
    }
    let mut result: Vec<GaussianTerms> = Vec::new();
    for terms in decomposed {
        let mut rep: GaussianTerms = GaussianTerms::new();
        for (m, cs) in terms {
            let total: Basic = core_add(&cs)?;
            let c: GaussianRational = match QQ_I.from_expr(&total) {
                Ok(c) => c,
                Err(PolyError::Value(_)) => {
                    return Err(polys_error(
                        PolysErrorKind::CoercionFailed,
                        format!("the domain of {} (RR, EX) is not ported yet", total),
                    ))
                }
                Err(e) => return Err(e),
            };
            let value: GaussianRational = QQ_I.add(&rep.remove(&m).unwrap_or_else(|| QQ_I.zero()), &c);
            if !value.is_zero() {
                rep.insert(m, value);
            }
        }
        result.push(rep);
    }
    Ok((gens, result))
}

/// `f/g` in lowest terms over `ZZ_I` (the denominators of `QQ_I`
/// cleared), canonical with respect to the quadrant of the leading
/// coefficient of the denominator (`_cancel_gaussian`).
fn cancel_gaussian(f: &GaussianTerms, g: &GaussianTerms, n: usize) -> PolyResult<(GaussianTerms, GaussianTerms)> {
    use super::domains::domain::ConvertFrom;
    let ring: SparsePolynomialRing<super::domains::gaussiandomains::GaussianIntegerRing> =
        SparsePolynomialRing::new(ZZ_I, internal_names(n, 0));
    if f.is_empty() {
        let mut one: GaussianTerms = GaussianTerms::new();
        one.insert(vec![0; n], QQ_I.one());
        return Ok((GaussianTerms::new(), one));
    }
    let cleared = |h: &GaussianTerms| -> PolyResult<(
        GaussianInteger,
        PolyElement<super::domains::gaussiandomains::GaussianIntegerRing>,
    )> {
        let mut common: GaussianInteger = ZZ_I.one();
        for c in h.values() {
            common = ZZ_I.lcm(&common, &ZZ_I.denom_of(c));
        }
        let k: GaussianRational = QQ_I.convert(&common, &ZZ_I)?;
        let mut terms: Terms<GaussianInteger> = Terms::new();
        for (m, c) in h {
            terms.insert(m.clone(), ZZ_I.convert(&QQ_I.mul(c, &k), &QQ_I)?);
        }
        Ok((common, ring.ring.from_dict(terms)))
    };
    let (cq, big_f) = cleared(f)?;
    let (cp, big_g) = cleared(g)?;
    let h = Domain::gcd(&ring, &big_f, &big_g);
    let (big_p, big_q) = (big_f.exquo(&h)?, big_g.exquo(&h)?);
    let common: GaussianInteger = ZZ_I.gcd(&cp, &cq);
    let (cp, cq) = (ZZ_I.quo(&cp, &common)?, ZZ_I.quo(&cq, &common)?);
    let scaled = |poly: &PolyElement<super::domains::gaussiandomains::GaussianIntegerRing>,
                  c: &GaussianInteger|
     -> PolyResult<GaussianTerms> {
        let factor: GaussianRational = QQ_I.convert(c, &ZZ_I)?;
        let mut out: GaussianTerms = GaussianTerms::new();
        for (m, v) in &poly.terms {
            out.insert(m.clone(), QQ_I.mul(&QQ_I.convert(v, &ZZ_I)?, &factor));
        }
        Ok(out)
    };
    let mut p: GaussianTerms = scaled(&big_p, &cp)?;
    let mut q: GaussianTerms = scaled(&big_q, &cq)?;
    let lead: GaussianRational = q.values().next_back().expect("a nonzero denominator").clone();
    let u: GaussianRational = QQ_I.canonical_unit(&lead);
    if u != QQ_I.one() {
        for c in p.values_mut() {
            *c = QQ_I.mul(c, &u);
        }
        for c in q.values_mut() {
            *c = QQ_I.mul(c, &u);
        }
    }
    Ok((p, q))
}

fn gaussian_expr_of(f: &GaussianTerms, gens: &[Basic]) -> PolyResult<Basic> {
    let mut parts: Vec<Basic> = Vec::new();
    for (m, c) in f.iter().rev() {
        let mut factors: Vec<Basic> = vec![QQ_I.to_expr(c)?];
        for (g, k) in gens.iter().zip(m) {
            if *k > 0 {
                factors.push(core_pow(g, &Basic::int(i64::from(*k)))?);
            }
        }
        parts.push(core_mul(&factors)?);
    }
    core_add(&parts)
}
