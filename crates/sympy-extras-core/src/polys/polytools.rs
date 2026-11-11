//! Port of the helpers of `sympy_extras/polys/polytools.py` which need no
//! expressions: `_factor_all` (the factorisation over `QQ` in all the
//! variables of a ring) and `_resultant_first` (the resultant in the first
//! variable). The class `Poly` and the rest of the module wait for the core.

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
type Decomposed = BTreeMap<Monomial, Vec<Basic>>;

fn exponent_of(n: i64) -> PolyResult<u32> {
    u32::try_from(n).map_err(|_| polynomial_error("a huge exponent".to_string()))
}

/// The terms of `e` (expanded) in the generators `gens`
/// (`_parallel_dict_from_expr_if_gens`).
fn decompose_with_gens(e: &Basic, gens: &[Basic]) -> PolyResult<Decomposed> {
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

/// The terms of `e` with every factor which is not a number a generator
/// (`_parallel_dict_from_expr_no_gens`).
fn decompose_without_gens(e: &Basic) -> PolyResult<(Decomposed, Vec<Basic>)> {
    let mut found: Vec<Basic> = Vec::new();
    let mut raw: Vec<(Vec<Basic>, Vec<(Basic, i64)>)> = Vec::new();
    for term in terms_of(&expand_default(e)?) {
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
    let gens: Vec<Basic> = sort_gens(&found);
    let mut result: Decomposed = Decomposed::new();
    for (coefficient, elements) in raw {
        let mut monomial: Monomial = vec![0; gens.len()];
        for (base, exponent) in elements {
            let i: usize = gens.iter().position(|g| *g == base).expect("a generator");
            monomial[i] = exponent_of(exponent)?;
        }
        result.entry(monomial).or_default().push(core_mul(&coefficient)?);
    }
    Ok((result, gens))
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

    fn split(&self) -> BTreeMap<Monomial, BTreeMap<Monomial, Rat>> {
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

    fn coefficient_expr(&self, terms: &BTreeMap<Monomial, Rat>) -> PolyResult<Basic> {
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
