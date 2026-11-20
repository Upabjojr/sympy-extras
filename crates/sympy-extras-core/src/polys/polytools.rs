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
pub(crate) fn gen_key(gen: &Basic) -> (i64, String, Int, i64) {
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

/// Whether `factor` depends on the generator `g` (`_contains_generator`):
/// a symbol only when it is free (SymPy's `has` also sees the bound
/// variable of a `CRootOf`, sympy-extras#25), another generator as a
/// subtree.
fn contains_generator(factor: &Basic, g: &Basic) -> bool {
    if matches!(g.head(), Head::Symbol(_) | Head::Dummy(_, _)) {
        return factor.free_symbols().contains(g);
    }
    factor.has(g)
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
            } else if !gens.iter().any(|g| contains_generator(&factor, g)) {
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
/// `QQ[params]` and the fraction fields `ZZ(params)`, `QQ(params)`
/// (`_Domain`).
#[derive(Debug, Clone, PartialEq)]
pub struct PolyDomain {
    pub integral: bool,
    pub params: Vec<Basic>,
    /// a fraction field (only with parameters)
    pub fraction: bool,
}

impl PolyDomain {
    pub fn new(integral: bool, params: Vec<Basic>, fraction: bool) -> PolyDomain {
        let fraction: bool = fraction && !params.is_empty();
        PolyDomain {
            integral,
            params,
            fraction,
        }
    }

    pub fn name(&self) -> String {
        let ground: &str = if self.integral { "ZZ" } else { "QQ" };
        if self.params.is_empty() {
            return ground.to_string();
        }
        let names: Vec<String> = self.params.iter().map(|p| p.to_string()).collect();
        if self.fraction {
            format!("{}({})", ground, names.join(","))
        } else {
            format!("{}[{}]", ground, names.join(","))
        }
    }

    /// The fraction field of a polynomial domain (`ZZ[y]` to `ZZ(y)`; `ZZ`
    /// to `QQ`), the domain itself for a field (`to_field`).
    pub fn to_field(&self) -> PolyDomain {
        if self.params.is_empty() {
            return PolyDomain::new(false, Vec::new(), false);
        }
        PolyDomain::new(self.integral, self.params.clone(), true)
    }
}

// PORTED FROM SYMPY: sympy/polys/domains/domain.py:Domain.unify_composite (SymPy 1.14)
fn unify_domains(a: &PolyDomain, b: &PolyDomain) -> PolyDomain {
    let mut all: Vec<Basic> = a.params.clone();
    for p in &b.params {
        if !all.contains(p) {
            all.push(p.clone());
        }
    }
    let params: Vec<Basic> = sort_gens(&all).into_iter().filter(|p| p.is_symbol()).collect();
    let mut integral: bool = a.integral && b.integral;
    let other_params: &Vec<Basic> = if b.fraction { &a.params } else { &b.params };
    if a.fraction != b.fraction && !other_params.is_empty() && !(!a.integral && !b.integral) {
        // ZZ[x] and QQ(x), QQ[x] and ZZ(x): ZZ(x)
        integral = true;
    }
    PolyDomain::new(integral, params, a.fraction || b.fraction)
}

type ParamTerms = BTreeMap<Monomial, Rat>;

/// The numerator and the denominator of a coefficient as polynomials in
/// the parameters named `names` (`as_numer_denom`), `None` when they are
/// not polynomials (`_coefficient_dicts`).
fn coefficient_dicts(c: &Basic, names: &[String]) -> PolyResult<Option<(ParamTerms, ParamTerms)>> {
    let (n, d) = crate::core::expr::as_numer_denom(c)?;
    let numer = match crate::polys::polyutils::dict_from_expr(&expand_default(&n)?, names, &QQ) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let denom = match crate::polys::polyutils::dict_from_expr(&expand_default(&d)?, names, &QQ) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    Ok(Some((numer.into_iter().collect(), denom.into_iter().collect())))
}

fn lcm_of_denominators<'a>(values: impl Iterator<Item = &'a Rat>) -> Int {
    let mut result: Int = Int::ONE;
    for c in values {
        let d: Int = denom(c);
        result = &result * &d / gcd(&result, &d);
    }
    result
}

fn integer_ring(names: Vec<String>) -> SparsePolynomialRing<IntegerRing> {
    SparsePolynomialRing::sympy(ZZ, names)
}

fn to_integer_terms(terms: &Terms<Rat>, scale: &Int) -> Terms<Int> {
    terms
        .iter()
        .map(|(m, c)| (m.clone(), numer(&(c * &rat_from_int(scale)))))
        .collect()
}

fn to_rational_terms(terms: &Terms<Int>) -> Terms<Rat> {
    terms.iter().map(|(m, c)| (m.clone(), rat_from_int(c))).collect()
}

// PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.cancel (SymPy 1.14)
/// `rep/den` in lowest terms (`den` free of the generators): both with
/// integer coefficients, without a common factor in `ZZ[params]`, the
/// leading coefficient of `den` positive (`_reduce_fraction`).
fn reduce_fraction(
    rep: &PolyElement<RationalField>,
    den: &PolyElement<RationalField>,
) -> PolyResult<(PolyElement<RationalField>, PolyElement<RationalField>)> {
    let ring = rep.ring.clone();
    if den.is_zero() {
        return Err(crate::error::zero_division("a zero denominator"));
    }
    if rep.is_zero() {
        return Ok((ring.zero(), ring.one()));
    }
    let scale: Int = lcm_of_denominators(rep.terms.values().chain(den.terms.values()));
    let numer_terms: Terms<Int> = to_integer_terms(&rep.terms, &scale);
    let denom_terms: Terms<Int> = to_integer_terms(&den.terms, &scale);
    if denom_terms.len() == 1 && denom_terms.keys().next().expect("a term").iter().all(|e| *e == 0) {
        // a number as the denominator: the integer content only
        let mut common: Int = Int::ZERO;
        for v in numer_terms.values().chain(denom_terms.values()) {
            common = gcd(&common, v);
        }
        if *denom_terms.values().next().expect("a term") < Int::ZERO {
            common = -common;
        }
        let divide = |t: &Terms<Int>| -> Terms<Rat> {
            t.iter()
                .map(|(m, v)| (m.clone(), rat_from_int(&(v / &common))))
                .collect()
        };
        return Ok((
            ring.from_dict(divide(&numer_terms)),
            ring.from_dict(divide(&denom_terms)),
        ));
    }
    let z = integer_ring(ring.names.clone());
    let (p, q) = z.cancel(&z.ring.from_dict(numer_terms), &z.ring.from_dict(denom_terms));
    Ok((
        ring.from_dict(to_rational_terms(&p.terms)),
        ring.from_dict(to_rational_terms(&q.terms)),
    ))
}

// PORTED FROM SYMPY: sympy/polys/constructor.py:_construct_composite (SymPy 1.14)
/// `rep/den` of the coefficients (rational functions of the parameters):
/// `den` the lcm of their denominators, reduced (`_fraction_from_coefficients`).
fn fraction_from_coefficients(
    coefficients: &BTreeMap<Monomial, Basic>,
    gens: &[Basic],
    domain: &PolyDomain,
) -> PolyResult<(PolyElement<RationalField>, PolyElement<RationalField>)> {
    let names: Vec<String> = domain.params.iter().map(symbol_name).collect();
    let n: usize = gens.len();
    let ring = rational_ring(internal_names(n, names.len()));
    let z = integer_ring(internal_names(0, names.len()));
    let mut parts: Vec<(Monomial, PolyElement<IntegerRing>, PolyElement<IntegerRing>)> = Vec::new();
    let mut common: PolyElement<IntegerRing> = z.ring.one();
    for (m, c) in coefficients {
        let Some((numer_terms, denom_terms)) = coefficient_dicts(c, &names)? else {
            return Err(polys_error(
                PolysErrorKind::CoercionFailed,
                format!("cannot convert {} into {}", c, domain.name()),
            ));
        };
        if numer_terms.is_empty() {
            continue;
        }
        let scale: Int = lcm_of_denominators(numer_terms.values().chain(denom_terms.values()));
        let d = z.ring.from_dict(to_integer_terms(&denom_terms, &scale));
        let numerator = z.ring.from_dict(to_integer_terms(&numer_terms, &scale));
        common = Domain::lcm(&z, &common, &d);
        parts.push((m.clone(), numerator, d));
    }
    let mut terms: Terms<Rat> = Terms::new();
    for (m, numerator, d) in parts {
        for (pm, v) in numerator.mul(&common.exquo(&d)?).terms {
            let mut key: Monomial = m.clone();
            key.extend(pm);
            let total: Rat = terms.remove(&key).unwrap_or(Rat::ZERO) + rat_from_int(&v);
            terms.insert(key, total);
        }
    }
    terms.retain(|_, v| !v.is_zero());
    let den_terms: Terms<Rat> = common
        .terms
        .iter()
        .map(|(pm, v)| {
            let mut key: Monomial = vec![0; n];
            key.extend(pm.iter().copied());
            (key, rat_from_int(v))
        })
        .collect();
    reduce_fraction(&ring.from_dict(terms), &ring.from_dict(den_terms))
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
    let mut polynomial_values: Vec<Rat> = Vec::new();
    let mut fraction: bool = false;
    for c in coefficients {
        if !names.is_empty() {
            match crate::polys::polyutils::dict_from_expr(c, &names, &QQ) {
                Ok(terms) => polynomial_values.extend(terms.into_values()),
                Err(_) => {
                    if coefficient_dicts(c, &names)?.is_none() {
                        return Err(polys_error(
                            PolysErrorKind::CoercionFailed,
                            format!("the domain of {} (EX) is not ported yet", c),
                        ));
                    }
                    fraction = true;
                }
            }
        } else {
            match c.as_rational() {
                Some(v) => polynomial_values.push(v.clone()),
                None => {
                    return Err(polys_error(
                        PolysErrorKind::CoercionFailed,
                        format!("the domain of {} (EX, RR or an extension) is not ported yet", c),
                    ))
                }
            }
        }
    }
    if fraction {
        // SymPy's _construct_composite: the coefficients of every numerator
        // and denominator decide between ZZ(params) and QQ(params)
        for c in coefficients {
            let Some((n, d)) = coefficient_dicts(c, &names)? else {
                return Err(polys_error(
                    PolysErrorKind::CoercionFailed,
                    format!("the domain of {} (EX) is not ported yet", c),
                ));
            };
            if n.values().chain(d.values()).any(|v| !v.is_int()) {
                integral = false;
            }
        }
        return Ok(PolyDomain::new(integral, params, true));
    }
    if polynomial_values.iter().any(|v| !v.is_int()) {
        integral = false;
    }
    Ok(PolyDomain::new(integral, params, false))
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
///
/// Over a fraction field the polynomial is `rep/den`: `den` a polynomial
/// in the parameters, both with integer coefficients, without common
/// factor and the leading coefficient of `den` positive.
#[derive(Debug, Clone)]
pub struct Poly {
    pub rep: PolyElement<RationalField>,
    pub gens: Vec<Basic>,
    pub domain: PolyDomain,
    pub den: Option<PolyElement<RationalField>>,
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        if self.gens != other.gens {
            return false;
        }
        if self.domain.params == other.domain.params {
            // one ring: the representations are equal exactly when the
            // polynomials are (the fractions reduced)
            if self.den.is_none() && other.den.is_none() {
                return self.rep.terms == other.rep.terms;
            }
            let field = |p: &Poly| -> Option<Poly> {
                if p.den.is_some() {
                    Some(p.clone())
                } else {
                    p.converted(&p.domain.to_field()).ok()
                }
            };
            return match (field(self), field(other)) {
                (Some(a), Some(b)) => a.rep.terms == b.rep.terms && a.den().terms == b.den().terms,
                _ => false,
            };
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
        if domain.fraction {
            let (rep, den) = fraction_from_coefficients(&coefficients, &generators, &domain)?;
            return Ok(Poly::new_rep(rep, generators, domain, Some(den)));
        }
        let rep = rep_from_coefficients(&coefficients, &generators, &domain)?;
        Ok(Poly::new_rep(rep, generators, domain, None))
    }

    /// The polynomial of a representation (`_new`); over a fraction field
    /// `rep/den` (`den` one when it is not given), which must be reduced.
    pub fn new_rep(
        rep: PolyElement<RationalField>,
        gens: Vec<Basic>,
        domain: PolyDomain,
        den: Option<PolyElement<RationalField>>,
    ) -> Poly {
        let den = if domain.fraction {
            Some(den.unwrap_or_else(|| rep.ring.one()))
        } else {
            None
        };
        Poly { rep, gens, domain, den }
    }

    /// The denominator of the representation (`1` outside the fraction
    /// fields).
    pub fn den(&self) -> PolyElement<RationalField> {
        self.den.clone().unwrap_or_else(|| self.rep.ring.one())
    }

    /// Whether the domain is a fraction field.
    pub fn is_fraction_field(&self) -> bool {
        self.domain.fraction
    }

    /// `rep/den` reduced, over the fraction field of the domain (or over
    /// `domain`) (`_fraction`).
    fn fraction(
        &self,
        rep: &PolyElement<RationalField>,
        den: &PolyElement<RationalField>,
        domain: Option<&PolyDomain>,
    ) -> PolyResult<Poly> {
        let target: PolyDomain = domain.cloned().unwrap_or_else(|| self.domain.to_field());
        let (p, q) = reduce_fraction(rep, den)?;
        Ok(Poly::new_rep(p, self.gens.clone(), target, Some(q)))
    }

    /// `rep` over `ZZ[params]` (`_numerator`).
    fn numerator(&self) -> Poly {
        Poly::new_rep(
            self.rep.clone(),
            self.gens.clone(),
            PolyDomain::new(true, self.domain.params.clone(), false),
            None,
        )
    }

    /// The polynomial over `domain`, which must contain its domain
    /// (`_set_domain`; `CoercionFailed` otherwise).
    fn set_domain(&self, domain: &PolyDomain) -> PolyResult<Poly> {
        if *domain == self.domain {
            return Ok(self.clone());
        }
        let own: &PolyDomain = &self.domain;
        if !own.params.iter().all(|p| domain.params.contains(p))
            || (own.fraction && !domain.fraction)
            || (domain.integral && !domain.fraction && !own.integral)
        {
            return Err(polys_error(
                PolysErrorKind::CoercionFailed,
                format!("cannot convert {} into {}", self.to_str()?, domain.name()),
            ));
        }
        self.converted(domain)
    }

    /// The terms of a polynomial of the ring which is free of the
    /// generators, as terms in the parameters (`_param_terms`).
    fn param_terms(&self, f: &PolyElement<RationalField>) -> ParamTerms {
        let n: usize = self.gens.len();
        f.terms.iter().map(|(m, c)| (m[n..].to_vec(), c.clone())).collect()
    }

    // PORTED FROM SYMPY: sympy/polys/fields.py:FracElement.as_expr (SymPy 1.14)
    /// `numer/denom` (terms in the parameters) in lowest terms, as SymPy
    /// writes an element of `ZZ(params)` (`_fraction_expr`).
    fn fraction_expr(&self, numer_terms: &ParamTerms, denom_terms: &ParamTerms) -> PolyResult<Basic> {
        if numer_terms.is_empty() {
            return Ok(Basic::zero());
        }
        if denom_terms.len() == 1 {
            let (m, d) = denom_terms.iter().next().expect("a term");
            if m.iter().all(|e| *e == 0) && *d == Rat::ONE {
                return self.coefficient_expr(numer_terms);
            }
        }
        let z = integer_ring(internal_names(0, self.domain.params.len()));
        let scale: Int = lcm_of_denominators(numer_terms.values().chain(denom_terms.values()));
        let (p, q) = z.cancel(
            &z.ring.from_dict(to_integer_terms(numer_terms, &scale)),
            &z.ring.from_dict(to_integer_terms(denom_terms, &scale)),
        );
        let numer_expr: Basic = self.coefficient_expr(&to_rational_terms(&p.terms))?;
        if z.is_one(&q) {
            return Ok(numer_expr);
        }
        core_mul(&[
            numer_expr,
            core_pow(
                &self.coefficient_expr(&to_rational_terms(&q.terms))?,
                &Basic::negative_one(),
            )?,
        ])
    }

    /// A rational function of the parameters in lowest terms, as SymPy
    /// writes the element of the fraction field (`_domain_element_expr`).
    fn domain_element_expr(&self, e: &Basic) -> PolyResult<Basic> {
        let names: Vec<String> = self.domain.params.iter().map(symbol_name).collect();
        match coefficient_dicts(e, &names)? {
            Some((n, d)) => self.fraction_expr(&n, &d),
            None => Ok(e.clone()),
        }
    }

    /// SymPy's `FracElement.as_expr` of an element in its normal form
    /// (`_ratio_expr`).
    fn ratio_expr(&self, numer_terms: &ParamTerms, denom_terms: &ParamTerms) -> PolyResult<Basic> {
        let numer_expr: Basic = self.coefficient_expr(numer_terms)?;
        let one: ParamTerms = BTreeMap::from([(vec![0; self.domain.params.len()], Rat::ONE)]);
        if *denom_terms == one {
            return Ok(numer_expr);
        }
        core_mul(&[
            numer_expr,
            core_pow(&self.coefficient_expr(denom_terms)?, &Basic::negative_one())?,
        ])
    }

    /// A polynomial in the parameters as one of the ring of `rep`
    /// (`_embedded`).
    fn embedded(&self, f: &ParamTerms) -> PolyElement<RationalField> {
        let n: usize = self.gens.len();
        self.rep.ring.from_dict(
            f.iter()
                .map(|(m, c)| {
                    let mut key: Monomial = vec![0; n];
                    key.extend(m.iter().copied());
                    (key, c.clone())
                })
                .collect(),
        )
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
        Ok(Poly::new_rep(
            ring.from_dict(values),
            gens.to_vec(),
            PolyDomain::new(integral, Vec::new(), false),
            None,
        ))
    }

    /// A polynomial over the domain of `self` (or its `ZZ`/`QQ` variant);
    /// over a fraction field the polynomial `rep` itself (`_same_ring`).
    fn same_ring(&self, rep: PolyElement<RationalField>, integral: Option<bool>) -> Poly {
        let domain: PolyDomain = match integral {
            None => self.domain.clone(),
            Some(i) => PolyDomain::new(i, self.domain.params.clone(), self.domain.fraction),
        };
        if domain.fraction {
            let (p, q) = reduce_fraction(&rep, &rep.ring.one()).expect("a nonzero denominator");
            return Poly::new_rep(p, self.gens.clone(), domain, Some(q));
        }
        Poly::new_rep(rep, self.gens.clone(), domain, None)
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
        let denom_terms: Option<ParamTerms> = self.den.as_ref().map(|d| self.param_terms(d));
        for (m, c) in self.split().iter().rev() {
            let coefficient: Basic = match &denom_terms {
                Some(d) => self.fraction_expr(c, d)?,
                None => self.coefficient_expr(c)?,
            };
            result.push((m.clone(), coefficient));
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
        Ok((self.converted(&domain)?, other.converted(&domain)?))
    }

    /// The polynomial over a larger domain (more parameters, `QQ` for
    /// `ZZ`, the fraction field of a polynomial ring) (`_converted`).
    fn converted(&self, domain: &PolyDomain) -> PolyResult<Poly> {
        let n: usize = self.gens.len();
        let (rep, den) = if domain.params == self.domain.params {
            (self.rep.clone(), self.den.clone())
        } else {
            let ring = rational_ring(internal_names(n, domain.params.len()));
            let moved = |f: &PolyElement<RationalField>| -> PolyElement<RationalField> {
                let mut terms: Terms<Rat> = Terms::new();
                for (m, c) in &f.terms {
                    let mut exponents: Vec<u32> = vec![0; domain.params.len()];
                    for (p, k) in self.domain.params.iter().zip(&m[n..]) {
                        let position: usize = domain.params.iter().position(|q| q == p).expect("a parameter");
                        exponents[position] = *k;
                    }
                    let mut key: Monomial = m[..n].to_vec();
                    key.extend(exponents);
                    terms.insert(key, c.clone());
                }
                ring.from_dict(terms)
            };
            (moved(&self.rep), self.den.as_ref().map(moved))
        };
        if !domain.fraction {
            return Ok(Poly::new_rep(rep, self.gens.clone(), domain.clone(), None));
        }
        if den.is_some() {
            return Ok(Poly::new_rep(rep, self.gens.clone(), domain.clone(), den));
        }
        let (p, q) = reduce_fraction(&rep, &rep.ring.one())?;
        Ok(Poly::new_rep(p, self.gens.clone(), domain.clone(), Some(q)))
    }

    pub fn neg(&self) -> Poly {
        if let Some(d) = &self.den {
            return Poly::new_rep(self.rep.neg(), self.gens.clone(), self.domain.clone(), Some(d.clone()));
        }
        self.same_ring(self.rep.neg(), None)
    }

    pub fn add(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        if let (Some(da), Some(db)) = (&a.den, &b.den) {
            if da == db {
                return a.fraction(&a.rep.add(&b.rep), da, None);
            }
            return a.fraction(&a.rep.mul(db).add(&b.rep.mul(da)), &da.mul(db), None);
        }
        Ok(a.same_ring(a.rep.add(&b.rep), None))
    }

    pub fn sub(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        if a.den.is_some() {
            return a.add(&b.neg());
        }
        Ok(a.same_ring(a.rep.sub(&b.rep), None))
    }

    pub fn mul(&self, other: &Poly) -> PolyResult<Poly> {
        let (a, b) = self.unified(other)?;
        if let (Some(da), Some(db)) = (&a.den, &b.den) {
            return a.fraction(&a.rep.mul(&b.rep), &da.mul(db), None);
        }
        Ok(a.same_ring(a.rep.mul(&b.rep), None))
    }

    pub fn pow(&self, n: u64) -> Poly {
        if let Some(d) = &self.den {
            // the powers of a reduced fraction are reduced
            return Poly::new_rep(self.rep.pow(n), self.gens.clone(), self.domain.clone(), Some(d.pow(n)));
        }
        self.same_ring(self.rep.pow(n), None)
    }

    pub fn diff(&self, i: usize) -> Poly {
        if let Some(d) = &self.den {
            return self
                .fraction(&self.rep.diff(i), d, None)
                .expect("a nonzero denominator");
        }
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
        if a.den.is_some() {
            return a.ground_monic(&a.numerator().gcd(&b.numerator())?.rep);
        }
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
        if a.den.is_some() {
            return a.ground_monic(&a.numerator().lcm(&b.numerator())?.rep);
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
        if a.den.is_some() {
            let (p, q) = (a.exquo(&g)?, b.exquo(&g)?);
            return Ok((g, p, q));
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
        if a.den.is_some() {
            return a.cancel_fraction(&b);
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
        if a.den.is_some() {
            return a.div_fraction(&b);
        }
        if !a.domain.params.is_empty() {
            if a.gens.len() == 1 {
                let split = b.split();
                let top: Monomial = vec![b.degree(0)? as u32];
                let zero: Monomial = vec![0; a.domain.params.len()];
                let lead = &split[&top];
                if !(lead.len() == 1 && lead.contains_key(&zero)) {
                    let field: PolyDomain = a.domain.to_field();
                    let (fq, fr) = a.converted(&field)?.div_fraction(&b.converted(&field)?)?;
                    if let (Some(q), Some(r)) = (fq.retracted(&a.domain)?, fr.retracted(&a.domain)?) {
                        return Ok((q, r));
                    }
                    return Ok((fq, fr));
                }
            }
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
        if a.den.is_some() {
            // over a fraction field: by the primitive part of the divisor
            // (Gauss's lemma), the content going into the denominator
            if b.is_zero() {
                return Err(crate::error::zero_division("polynomial division by zero"));
            }
            let content = b.ground_content_rep()?;
            let quotient = match b.rep.exquo(&content).and_then(|primitive| a.rep.exquo(&primitive)) {
                Ok(q) => q,
                Err(_) => {
                    return Err(polys_error(
                        PolysErrorKind::ExactQuotientFailed,
                        format!("{} does not divide {}", b.to_str()?, a.to_str()?),
                    ))
                }
            };
            return a.fraction(&quotient.mul(&b.den()), &a.den().mul(&content), Some(&a.domain));
        }
        match a.rep.exquo(&b.rep) {
            Ok(q) => Ok(a.same_ring(q, None)),
            Err(_) => Err(polys_error(
                PolysErrorKind::ExactQuotientFailed,
                format!("{} does not divide {}", b.to_str()?, a.to_str()?),
            )),
        }
    }
}

impl Poly {
    /// The leading coefficient of `f` in the generators (lex, SymPy's
    /// `dmp_ground_LC`) as a polynomial of the ring in the parameters.
    fn ground_lc_rep(&self, f: &PolyElement<RationalField>) -> PolyElement<RationalField> {
        let n: usize = self.gens.len();
        let top: Monomial = f
            .terms
            .keys()
            .map(|m| m[..n].to_vec())
            .max()
            .expect("a nonzero polynomial");
        f.ring.from_dict(
            f.terms
                .iter()
                .filter(|(m, _)| m[..n] == top[..])
                .map(|(m, c)| {
                    let mut key: Monomial = vec![0; n];
                    key.extend_from_slice(&m[n..]);
                    (key, c.clone())
                })
                .collect(),
        )
    }

    // PORTED FROM SYMPY: sympy/polys/densetools.py:dmp_ground_monic (SymPy 1.14)
    /// `f` divided by its leading coefficient in the generators, over the
    /// fraction field of the domain (`_ground_monic`).
    fn ground_monic(&self, f: &PolyElement<RationalField>) -> PolyResult<Poly> {
        if f.is_zero() {
            return self.fraction(f, &f.ring.one(), None);
        }
        self.fraction(f, &self.ground_lc_rep(f), None)
    }

    /// The content of `rep` (integer coefficients) in `ZZ[params]`, with a
    /// positive leading coefficient, as a polynomial of the ring
    /// (`_ground_content_rep`).
    fn ground_content_rep(&self) -> PolyResult<PolyElement<RationalField>> {
        let n: usize = self.gens.len();
        let names: Vec<String> = self.rep.ring.names.clone();
        let z = IteratedPolynomialRing::new(
            SparsePolynomialRing::sympy(ZZ, names[n..].to_vec()),
            names[..n].to_vec(),
        );
        let fz = z
            .ring()
            .from_dict(self.rep.terms.iter().map(|(m, c)| (m.clone(), numer(c))).collect());
        let mut content = z.ground_content(&fz);
        if !content.is_zero() && content.LC() < Int::ZERO {
            content = content.neg();
        }
        Ok(self.embedded(
            &content
                .terms
                .iter()
                .map(|(m, c)| (m.clone(), rat_from_int(c)))
                .collect(),
        ))
    }

    /// The polynomial over the polynomial domain `ring` (SymPy's
    /// `to_ring`), `None` when its coefficients are not in it (`_retracted`).
    fn retracted(&self, ring: &PolyDomain) -> PolyResult<Option<Poly>> {
        let den = self.den();
        if den.terms.keys().any(|m| m.iter().any(|e| *e != 0)) {
            return Ok(None);
        }
        let d: Rat = den.LC();
        if ring.integral && d != Rat::ONE {
            return Ok(None);
        }
        Ok(Some(Poly::new_rep(
            self.rep.exquo_ground(&d)?,
            self.gens.clone(),
            ring.clone(),
            None,
        )))
    }

    // PORTED FROM SYMPY: sympy/polys/densearith.py:dup_pdiv (SymPy 1.14)
    /// `self = q b + r` over the fraction field: `lc^k P = Q B + R` on the
    /// numerators, `q = Q den(b)/(lc^k den)`, `r = R/(lc^k den)`
    /// (`_div_fraction`).
    fn div_fraction(&self, b: &Poly) -> PolyResult<(Poly, Poly)> {
        if self.gens.len() != 1 {
            return Err(polys_error(
                PolysErrorKind::OperationNotSupported,
                "division of multivariate polynomials is not ported yet".to_string(),
            ));
        }
        let ring = self.rep.ring.clone();
        let (p, bb) = (&self.rep, &b.rep);
        let m: i64 = b.degree(0)?;
        let n: i64 = self.degree(0)?;
        if n < m {
            return Ok((
                self.fraction(&ring.zero(), &ring.one(), Some(&self.domain))?,
                self.clone(),
            ));
        }
        let lc = self.ground_lc_rep(bb);
        let (mut q, mut r) = (ring.zero(), p.clone());
        let mut k: i64 = n - m + 1;
        let nvars: usize = ring.names.len();
        while !r.is_zero() && r.degree_in(0) >= m {
            let d: i64 = r.degree_in(0);
            let lr = ring.from_dict(
                r.terms
                    .iter()
                    .filter(|(mm, _)| i64::from(mm[0]) == d)
                    .map(|(mm, c)| {
                        let mut key: Monomial = vec![0];
                        key.extend_from_slice(&mm[1..]);
                        (key, c.clone())
                    })
                    .collect(),
            );
            let shift: Monomial = (0..nvars).map(|i| if i == 0 { (d - m) as u32 } else { 0 }).collect();
            q = q.mul(&lc).add(&lr.mul_monom(&shift));
            r = r.mul(&lc).sub(&lr.mul(bb).mul_monom(&shift));
            k -= 1;
        }
        let scale = lc.pow(k as u64);
        q = q.mul(&scale);
        r = r.mul(&scale);
        let factor = lc.pow((n - m + 1) as u64).mul(&self.den());
        Ok((
            self.fraction(&q.mul(&b.den()), &factor, Some(&self.domain))?,
            self.fraction(&r, &factor, Some(&self.domain))?,
        ))
    }

    // PORTED FROM SYMPY: sympy/polys/euclidtools.py:dmp_cancel (SymPy 1.14)
    /// `self = P/D`, `b = B/E`: `P/B` cancelled over `ZZ[params]`, the
    /// constant `E/D` in lowest terms (`_cancel_fraction`).
    fn cancel_fraction(&self, b: &Poly) -> PolyResult<(Basic, Poly, Poly)> {
        if !self.domain.integral {
            return self.cancel_rational_field(b);
        }
        let (c, p, q) = self.numerator().cancel(&b.numerator())?;
        let z = integer_ring(self.rep.ring.names.clone());
        let to_z = |f: &PolyElement<RationalField>| {
            z.ring
                .from_dict(f.terms.iter().map(|(m, v)| (m.clone(), numer(v))).collect())
        };
        let (mut cp, cq) = z.cancel(&to_z(&b.den()), &to_z(&self.den()));
        if c == Basic::negative_one() {
            cp = cp.neg();
        }
        let n: usize = self.gens.len();
        let params = |f: &PolyElement<IntegerRing>| -> ParamTerms {
            f.terms
                .iter()
                .map(|(m, v)| (m[n..].to_vec(), rat_from_int(v)))
                .collect()
        };
        let numer_expr: Basic = self.coefficient_expr(&params(&cp))?;
        let coefficient: Basic = if z.is_one(&cq) {
            numer_expr
        } else {
            core_mul(&[
                numer_expr,
                core_pow(&self.coefficient_expr(&params(&cq))?, &Basic::negative_one())?,
            ])?
        };
        let one = self.rep.ring.one();
        Ok((
            coefficient,
            Poly::new_rep(p.rep, self.gens.clone(), self.domain.clone(), Some(one.clone())),
            Poly::new_rep(q.rep, self.gens.clone(), self.domain.clone(), Some(one)),
        ))
    }

    // PORTED FROM SYMPY: sympy/polys/euclidtools.py:dmp_cancel, sympy/polys/densetools.py:dmp_clear_denoms (SymPy 1.14)
    /// `QQ(params)`: the denominators cleared into `QQ[params]`
    /// (`_cancel_rational_field`).
    fn cancel_rational_field(&self, b: &Poly) -> PolyResult<(Basic, Poly, Poly)> {
        use crate::polys::domains::sparsefractionfield::SparseFractionField;
        let r = SparsePolynomialRing::sympy(QQ, internal_names(0, self.domain.params.len()));
        let ring_domain = PolyDomain::new(false, self.domain.params.clone(), false);
        let cleared = |f: &Poly| -> PolyResult<(PolyElement<RationalField>, Poly)> {
            // the lcm of the denominators in QQ[params], in the dense order
            let field = SparseFractionField::new(r.clone());
            let den = r.ring.from_dict(f.param_terms(&f.den()));
            let mut common = r.ring.one();
            for terms in f.split().values().rev() {
                let element = field.new_element(&r.ring.from_dict(terms.clone()), &den)?;
                common = Domain::lcm(&r, &common, &element.denom);
            }
            // f*common: the numerator times common/den
            let scaled = f.rep.mul(&f.embedded(&common.terms.clone()));
            Ok((
                common,
                Poly::new_rep(scaled.exquo(&f.den())?, f.gens.clone(), ring_domain.clone(), None),
            ))
        };
        let (cq, f) = cleared(self)?;
        let (cp, g) = cleared(b)?;
        let (c, p, q) = f.cancel(&g)?;
        let h = Domain::gcd(&r, &cp, &cq);
        let mut cp = cp.exquo(&h)?;
        let cq = cq.exquo(&h)?;
        if c == Basic::negative_one() {
            cp = cp.neg();
        }
        let numer_expr: Basic = self.coefficient_expr(&cp.terms.clone())?;
        let coefficient: Basic = if r.is_one(&cq) {
            numer_expr
        } else {
            core_mul(&[
                numer_expr,
                core_pow(&self.coefficient_expr(&cq.terms.clone())?, &Basic::negative_one())?,
            ])?
        };
        Ok((coefficient, p.converted(&self.domain)?, q.converted(&self.domain)?))
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
        Poly::new_rep(
            ring.from_dict(terms),
            self.gens[self.gens.len() - n_gens..].to_vec(),
            self.domain.clone(),
            None,
        )
    }

    // PORTED FROM SYMPY: sympy/polys/densetools.py:dmp_ground_content, sympy/polys/domains/field.py:Field.gcd (SymPy 1.14)
    /// The content over the fraction field, as SymPy computes it: its
    /// numerator and denominator as terms in the parameters
    /// (`_field_content`, `_folded_field_content`).
    fn field_content(&self) -> PolyResult<(ParamTerms, ParamTerms)> {
        let names: Vec<String> = internal_names(0, self.domain.params.len());
        let den: ParamTerms = self.param_terms(&self.den());
        if self.domain.integral {
            let r = integer_ring(names);
            let to_z = |t: &ParamTerms| r.ring.from_dict(t.iter().map(|(m, v)| (m.clone(), numer(v))).collect());
            let dz = to_z(&den);
            let mut coefficients: BTreeMap<Monomial, (PolyElement<IntegerRing>, PolyElement<IntegerRing>)> =
                BTreeMap::new();
            for (m, terms) in self.split() {
                let (p, q) = r.cancel(&to_z(&terms), &dz);
                coefficients.insert(m, (p, q));
            }
            let (n, d) = folded_field_content(&r, &coefficients);
            return Ok((
                n.terms.iter().map(|(m, v)| (m.clone(), rat_from_int(v))).collect(),
                d.terms.iter().map(|(m, v)| (m.clone(), rat_from_int(v))).collect(),
            ));
        }
        let r = SparsePolynomialRing::sympy(QQ, names);
        let dq = r.ring.from_dict(den.clone());
        let mut coefficients: BTreeMap<Monomial, (PolyElement<RationalField>, PolyElement<RationalField>)> =
            BTreeMap::new();
        for (m, terms) in self.split() {
            coefficients.insert(m, r.cancel(&r.ring.from_dict(terms), &dq));
        }
        let (n, d) = folded_field_content(&r, &coefficients);
        Ok((n.terms.clone(), d.terms.clone()))
    }

    /// The polynomial divided by its leading coefficient (`monic`; over a
    /// polynomial domain only when that coefficient is a number).
    pub fn monic(&self) -> PolyResult<Poly> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        if self.den.is_some() {
            return self.ground_monic(&self.rep);
        }
        let lc: Rat = self.rep.LC();
        if !self.domain.params.is_empty() {
            let split = self.split();
            let lead = split.values().next_back().expect("a nonzero polynomial");
            let zero: Monomial = vec![0; self.domain.params.len()];
            if lead.len() != 1 || !lead.contains_key(&zero) {
                return self.ground_monic(&self.rep);
            }
        }
        let integral: Option<bool> = if lc != Rat::ONE { Some(false) } else { None };
        Ok(self.same_ring(self.rep.exquo_ground(&lc)?, integral))
    }

    /// The content (`content`; over `ZZ` the gcd of the coefficients, over
    /// `QQ` one).
    pub fn content(&self) -> PolyResult<Basic> {
        if self.den.is_some() {
            if self.is_zero() {
                return Ok(Basic::zero());
            }
            let (n, d) = self.field_content()?;
            return self.ratio_expr(&n, &d);
        }
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
        if self.den.is_some() {
            if self.is_zero() {
                return Ok((Basic::zero(), self.clone()));
            }
            let (n, d) = self.field_content()?;
            if n == d {
                return Ok((Basic::one(), self.clone()));
            }
            let primitive = self.fraction(
                &self.rep.mul(&self.embedded(&d)),
                &self.den().mul(&self.embedded(&n)),
                Some(&self.domain),
            )?;
            return Ok((self.ratio_expr(&n, &d)?, primitive));
        }
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
        if self.den.is_some() {
            // SymPy evaluates in the fraction field: an element of it, a
            // polynomial over it
            if rest.is_empty() {
                return Ok(PolyOrExpr::Expr(self.domain_element_expr(&result)?));
            }
            let evaluated: Poly = Poly::new(&result, &rest)?;
            return Ok(PolyOrExpr::Poly(match evaluated.set_domain(&self.domain) {
                Ok(p) => p,
                Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => evaluated,
                Err(e) => return Err(e),
            }));
        }
        if rest.is_empty() {
            return Ok(PolyOrExpr::Expr(result));
        }
        Ok(PolyOrExpr::Poly(Poly::new(&result, &rest)?))
    }

    /// A polynomial of the ring which is free of the first generator
    /// (`value/den` over a fraction field): an expression for a univariate
    /// polynomial, a polynomial in the other generators otherwise
    /// (`_after_first_gen`).
    fn drop_first(
        &self,
        value: &PolyElement<RationalField>,
        den: Option<&PolyElement<RationalField>>,
    ) -> PolyResult<PolyOrExpr> {
        let n: usize = self.gens.len();
        let tail = |f: &PolyElement<RationalField>| -> Terms<Rat> {
            f.terms.iter().map(|(m, c)| (m[1..].to_vec(), c.clone())).collect()
        };
        if n == 1 {
            return Ok(PolyOrExpr::Expr(match den {
                None => self.coefficient_expr(&tail(value))?,
                Some(d) => self.fraction_expr(&tail(value), &tail(d))?,
            }));
        }
        let Some(d) = den else {
            return Ok(PolyOrExpr::Poly(self.param_ring_rep(tail(value), n - 1)));
        };
        let ring = rational_ring(internal_names(n - 1, self.domain.params.len()));
        let (p, q) = reduce_fraction(&ring.from_dict(tail(value)), &ring.from_dict(tail(d)))?;
        Ok(PolyOrExpr::Poly(Poly::new_rep(
            p,
            self.gens[1..].to_vec(),
            self.domain.clone(),
            Some(q),
        )))
    }

    /// The resultant in the first generator; over a fraction field
    /// `Res(P/D, B/E) = Res(P, B)/(D**deg(b) E**deg(a))`.
    pub fn resultant(&self, other: &Poly) -> PolyResult<PolyOrExpr> {
        let (a, b) = self.unified(other)?;
        let value = resultant_first(&a.rep, &b.rep, &a.names(), a.domain.integral || a.den.is_some());
        let den: Option<PolyElement<RationalField>> = if a.den.is_some() {
            if a.is_zero() || b.is_zero() {
                Some(a.rep.ring.one())
            } else {
                Some(
                    a.den()
                        .pow(b.degree(0)?.max(0) as u64)
                        .mul(&b.den().pow(a.degree(0)?.max(0) as u64)),
                )
            }
        } else {
            None
        };
        a.drop_first(&value, den.as_ref())
    }

    /// `(-1)**(n (n - 1)/2) Res(f, f')/lc(f)` in the first generator.
    pub fn discriminant(&self) -> PolyResult<PolyOrExpr> {
        let n: i64 = self.degree(0)?;
        if n < 1 {
            return Ok(PolyOrExpr::Expr(Basic::zero()));
        }
        let mut value = resultant_first(
            &self.rep,
            &self.rep.diff(0),
            &self.names(),
            self.domain.integral || self.den.is_some(),
        );
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
        let den: Option<PolyElement<RationalField>> = self.den.as_ref().map(|d| d.pow((2 * n - 2) as u64));
        self.drop_first(&value, den.as_ref())
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
        if self.den.is_some() {
            if self.is_zero() {
                return Ok((Basic::zero(), Vec::new()));
            }
            let (_, numerator_factors) = self.numerator().sqf_list()?;
            let mut factors: Vec<(Poly, usize)> = Vec::new();
            for (g, k) in numerator_factors {
                if g.total_degree() > 0 {
                    factors.push((self.ground_monic(&g.rep)?, k));
                }
            }
            return Ok((self.LC()?, factors));
        }
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
        if let Some(den) = &self.den {
            let (c, numerator_factors) = self.numerator().factor_list()?;
            // the constant c (a polynomial in the parameters) over the
            // denominator: SymPy's coeff/denom of dmp_factor_list
            let names: Vec<String> = self.domain.params.iter().map(symbol_name).collect();
            let parts = coefficient_dicts(&c, &names)?;
            let Some((numer_terms, denom_terms)) =
                parts.filter(|(_, d)| d.len() == 1 && d.keys().next().expect("a term").iter().all(|e| *e == 0))
            else {
                return Err(polys_error(
                    PolysErrorKind::CoercionFailed,
                    format!("the content {} is not in {}", c, self.domain.name()),
                ));
            };
            let d: Rat = denom_terms.values().next().expect("a term").clone();
            let scaled: ParamTerms = self.param_terms(den).into_iter().map(|(m, v)| (m, v * &d)).collect();
            let constant: Basic = self.fraction_expr(&numer_terms, &scaled)?;
            return Ok((
                constant,
                numerator_factors
                    .into_iter()
                    .map(|(g, k)| (self.same_ring(g.rep, None), k))
                    .collect(),
            ));
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

/// The content over a fraction field as SymPy folds it over the dense
/// coefficients (zeros included, stopping at `1`): `Field.gcd` is the gcd
/// of the numerators over the lcm of the denominators in the ring
/// (`_folded_field_content`); the elements are `(numer, denom)` in lowest
/// terms.
fn folded_field_content<K: crate::polys::domains::sparsepolynomialring::SparseGround>(
    r: &SparsePolynomialRing<K>,
    coefficients: &BTreeMap<Monomial, (PolyElement<K>, PolyElement<K>)>,
) -> (PolyElement<K>, PolyElement<K>) {
    type Element<K> = (PolyElement<K>, PolyElement<K>);
    let zero = |r: &SparsePolynomialRing<K>| -> Element<K> { (r.ring.zero(), r.ring.one()) };
    let gcd_of = |u: &Element<K>, v: &Element<K>| -> Element<K> {
        let numerator = Domain::gcd(r, &u.0, &v.0);
        let denominator = Domain::lcm(r, &u.1, &v.1);
        if numerator.is_zero() {
            return (numerator, r.ring.one());
        }
        r.cancel(&numerator, &denominator)
    };
    fn content<K: crate::polys::domains::sparsepolynomialring::SparseGround>(
        r: &SparsePolynomialRing<K>,
        terms: &BTreeMap<Monomial, Element<K>>,
        zero: &dyn Fn(&SparsePolynomialRing<K>) -> Element<K>,
        gcd_of: &dyn Fn(&Element<K>, &Element<K>) -> Element<K>,
    ) -> Element<K> {
        if terms.is_empty() {
            return zero(r);
        }
        let mut grouped: BTreeMap<u32, BTreeMap<Monomial, Element<K>>> = BTreeMap::new();
        for (m, c) in terms {
            grouped.entry(m[0]).or_default().insert(m[1..].to_vec(), c.clone());
        }
        let mut result: Element<K> = zero(r);
        let top: u32 = *grouped.keys().next_back().expect("terms");
        for d in (0..=top).rev() {
            let value: Element<K> = match grouped.get(&d) {
                None => zero(r),
                Some(part) if part.is_empty() => zero(r),
                Some(part) => match part.get(&Vec::new()) {
                    Some(v) => v.clone(),
                    None => content(r, part, zero, gcd_of),
                },
            };
            result = gcd_of(&result, &value);
            if r.is_one(&result.0) && r.is_one(&result.1) {
                break;
            }
        }
        result
    }
    content(r, coefficients, &zero, &gcd_of)
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

/// The cancellation over `EX` (`_cancel_complex_or_ex`; the core has no
/// floats, so `RR` and `CC` do not arise).
fn cancel_ex(p: &Basic, q: &Basic, factored: &Basic) -> PolyResult<Basic> {
    use crate::polys::domains::domain::Domain;
    use crate::polys::domains::expressiondomain::EX;
    let (decomposed, gens) = decompose_without_gens_many(&[p.clone(), q.clone()], true)?;
    let mut reps: Vec<BTreeMap<Monomial, Basic>> = Vec::new();
    for terms in decomposed {
        let mut rep: BTreeMap<Monomial, Basic> = BTreeMap::new();
        for (m, cs) in terms {
            let c: Basic = core_add(&cs)?;
            if !EX.is_zero(&c) {
                rep.insert(m, c);
            }
        }
        reps.push(rep);
    }
    if gens.is_empty() {
        return expand_default(factored);
    }
    let n: usize = gens.len();
    let (f, g) = (&reps[0], &reps[1]);
    let (mut big_p, mut big_q): (BTreeMap<Monomial, Basic>, BTreeMap<Monomial, Basic>) = if f.is_empty() {
        let mut one: BTreeMap<Monomial, Basic> = BTreeMap::new();
        one.insert(vec![0; n], Basic::one());
        (BTreeMap::new(), one)
    } else if f.len() == 1 || g.len() == 1 {
        let (term, other) = if f.len() == 1 { (f, g) } else { (g, f) };
        let (mt, ct) = term.iter().next().expect("one term");
        let mut common: Monomial = mt.clone();
        for m in other.keys() {
            for (c, b) in common.iter_mut().zip(m) {
                *c = (*c).min(*b);
            }
        }
        let shift = |m: &Monomial| -> Monomial { m.iter().zip(&common).map(|(a, b)| a - b).collect() };
        let reduced_term: BTreeMap<Monomial, Basic> = [(shift(mt), ct.clone())].into_iter().collect();
        let reduced_other: BTreeMap<Monomial, Basic> = other.iter().map(|(m, c)| (shift(m), c.clone())).collect();
        if f.len() == 1 {
            (reduced_term, reduced_other)
        } else {
            (reduced_other, reduced_term)
        }
    } else if n == 1
        && f.values()
            .chain(g.values())
            .all(|c| crate::core::expr::is(c, crate::core::assumptions::Predicate::finite) == Some(true))
    {
        // (with zoo, oo or nan the subresultants need not end: nothing is
        // cancelled, as in several generators)
        let dense = |h: &BTreeMap<Monomial, Basic>| -> Vec<Basic> {
            let degree: usize = h.keys().map(|m| m[0] as usize).max().unwrap_or(0);
            let mut out: Vec<Basic> = vec![Basic::zero(); degree + 1];
            for (m, c) in h {
                out[degree - m[0] as usize] = c.clone();
            }
            out
        };
        let (df, dg) = (dense(f), dense(g));
        let last: Vec<Basic> = crate::polys::euclidtools::dup_subresultants(&df, &dg, &EX)
            .pop()
            .unwrap_or_default();
        let quotients = crate::polys::densetools::dup_monic(&last, &EX).and_then(|h| {
            Ok((
                crate::polys::densearith::dup_quo(&df, &h, &EX)?,
                crate::polys::densearith::dup_quo(&dg, &h, &EX)?,
            ))
        });
        let (qf, qg) = match quotients {
            Ok(v) => v,
            // a zero EX does not detect (oo - oo): nothing is cancelled
            Err(PolyError::Polys(..)) => return core_mul(&[p.clone(), core_pow(q, &Basic::negative_one())?]),
            Err(e) => return Err(e),
        };
        let sparse = |d: &[Basic]| -> BTreeMap<Monomial, Basic> {
            let k: usize = d.len().saturating_sub(1);
            d.iter()
                .enumerate()
                .filter(|(_, c)| !EX.is_zero(c))
                .map(|(i, c)| (vec![(k - i) as u32], c.clone()))
                .collect()
        };
        (sparse(&qf), sparse(&qg))
    } else if n > 1
        && f.values()
            .chain(g.values())
            .all(|c| crate::core::expr::is(c, crate::core::assumptions::Predicate::finite) == Some(true))
    {
        return Err(crate::error::not_ported("cancel over EX in several generators"));
    } else {
        (f.clone(), g.clone())
    };
    if let Some((_, lead)) = big_q.iter().next_back() {
        if EX.is_negative(lead) {
            big_p = big_p.iter().map(|(m, c)| (m.clone(), EX.neg(c))).collect();
            big_q = big_q.iter().map(|(m, c)| (m.clone(), EX.neg(c))).collect();
        }
    }
    let expr_of = |h: &BTreeMap<Monomial, Basic>| -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (m, c) in h.iter().rev() {
            let mut factors: Vec<Basic> = vec![c.clone()];
            for (g, k) in gens.iter().zip(m) {
                if *k != 0 {
                    factors.push(core_pow(g, &Basic::int(*k as i64))?);
                }
            }
            parts.push(core_mul(&factors)?);
        }
        core_add(&parts)
    };
    core_mul(&[expr_of(&big_p)?, core_pow(&expr_of(&big_q)?, &Basic::negative_one())?])
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
            let (ggens, greps) = match joint_rings_gaussian(&[p.clone(), q.clone()], true) {
                Ok(r) => r,
                Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => return cancel_ex(&p, &q, &factored),
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
                if method != FactorMethod::Factor {
                    return Err(polys_error(PolysErrorKind::CoercionFailed, message));
                }
                // Gaussian coefficients: SymPy's factorisation over ZZ_I and
                // QQ_I through QQ<I> (the real and infinite domains wait)
                match gaussian_factor_list(&base) {
                    Ok((c, gaussian)) => (
                        c,
                        gaussian
                            .into_iter()
                            .map(|(g, k)| (FactorBase::Expr(g), Basic::int(k as i64)))
                            .collect(),
                    ),
                    Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
                        return Err(crate::error::not_ported("the factorisation over RR or EX"));
                    }
                    Err(e) => return Err(e),
                }
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

// -- the factorisation over ZZ_I and QQ_I

fn zz_i_of(c: &GaussianRational) -> PolyResult<GaussianInteger> {
    use crate::polys::domains::domain::ConvertFrom;
    ZZ_I.convert(c, &QQ_I)
}

fn qq_i_of(c: &GaussianInteger) -> PolyResult<GaussianRational> {
    use crate::polys::domains::domain::ConvertFrom;
    QQ_I.convert(c, &ZZ_I)
}

/// The factorisation of a polynomial with Gaussian coefficients over
/// `ZZ_I` (or `QQ_I`): over `QQ<I>` by Trager's algorithm, the contents of
/// the factors made Gaussian integers as SymPy does; several generators
/// through [`super::gaussianfactor`] (`_gaussian_factor_list`).
#[allow(clippy::type_complexity)]
pub fn gaussian_factor_list(base: &Basic) -> PolyResult<(Basic, Vec<(Basic, usize)>)> {
    let (gens, reps) = joint_rings_gaussian(std::slice::from_ref(base), true)?;
    if gens.is_empty() {
        return Err(polys_error(PolysErrorKind::CoercionFailed, "no generator".to_string()));
    }
    let rep: &GaussianTerms = &reps[0];
    let integral: bool = rep.values().all(|c| denom(&c.x) == Int::ONE && denom(&c.y) == Int::ONE);
    if gens.len() > 1 {
        let ring = PolyRing::new(QQ_I, internal_names(gens.len(), 0), MonomialOrder::Lex)?;
        let (coefficient, multivariate) =
            super::gaussianfactor::gaussian_factor_list(&ring.from_dict(rep.clone()), integral)?;
        let mut factors: Vec<(Basic, usize)> = Vec::new();
        for (g, k) in multivariate {
            factors.push((gaussian_expr_of(&g.terms, &gens)?, k));
        }
        return Ok((QQ_I.to_expr(&coefficient)?, factors));
    }
    let degree: u32 = rep.keys().map(|m| m[0]).max().unwrap_or(0);
    let mut dense: Vec<GaussianRational> = (0..=degree)
        .map(|i| rep.get(&vec![degree - i]).cloned().unwrap_or_else(|| QQ_I.zero()))
        .collect();
    // dup_terms_gcd
    let mut j: usize = 0;
    while dense.last().is_some_and(|c| c.is_zero()) {
        dense.pop();
        j += 1;
    }
    // dup_primitive (the content over ZZ_I, 1 over the field QQ_I)
    let mut cont: GaussianRational = QQ_I.one();
    if integral {
        let mut content: GaussianInteger = ZZ_I.zero();
        for c in &dense {
            content = ZZ_I.gcd(&content, &zz_i_of(c)?);
            if content == ZZ_I.one() {
                break;
            }
        }
        cont = qq_i_of(&content)?;
        dense = dense.iter().map(|c| QQ_I.exquo(c, &cont)).collect::<PolyResult<_>>()?;
    }
    // dup_qq_i_factor: the factorisation over QQ<I>
    let k1 = super::domains::algebraicfield::AlgebraicField::new(
        vec![Rat::ONE, Rat::ZERO, Rat::ONE],
        Basic::imaginary_unit(),
    )?;
    let ext: Vec<_> = dense
        .iter()
        .map(|c| k1.from_expr(&QQ_I.to_expr(c)?))
        .collect::<PolyResult<_>>()?;
    let (lc, ext_factors) = super::factortools::dup_ext_factor(&ext, &k1)?;
    let mut coeff: GaussianRational = QQ_I.from_expr(&k1.to_expr(&lc)?)?;
    let mut found: Vec<(Vec<GaussianRational>, usize)> = Vec::new();
    for (fac, k) in ext_factors {
        let cs: Vec<GaussianRational> = fac
            .iter()
            .map(|c| QQ_I.from_expr(&k1.to_expr(c)?))
            .collect::<PolyResult<_>>()?;
        found.push((cs, k));
    }
    if integral {
        // dup_zz_i_factor: the denominators and the contents of the factors
        // over ZZ_I moved into the coefficient
        let mut new_found: Vec<(Vec<GaussianRational>, usize)> = Vec::new();
        for (fac, k) in found {
            let mut fac_denom: GaussianInteger = ZZ_I.one();
            for c in &fac {
                fac_denom = ZZ_I.lcm(&fac_denom, &ZZ_I.denom_of(c));
            }
            let d: GaussianRational = qq_i_of(&fac_denom)?;
            let numerators: Vec<GaussianInteger> = fac
                .iter()
                .map(|c| zz_i_of(&QQ_I.mul(c, &d)))
                .collect::<PolyResult<_>>()?;
            let mut content: GaussianInteger = ZZ_I.zero();
            for nc in &numerators {
                content = ZZ_I.gcd(&content, nc);
                if content == ZZ_I.one() {
                    break;
                }
            }
            let primitive: Vec<GaussianRational> = numerators
                .iter()
                .map(|nc| qq_i_of(&ZZ_I.exquo(nc, &content)?))
                .collect::<PolyResult<_>>()?;
            let content_q: GaussianRational = qq_i_of(&content)?;
            coeff = QQ_I.exquo(
                &QQ_I.mul(&coeff, &QQ_I.pow(&content_q, k as u64)),
                &QQ_I.pow(&d, k as u64),
            )?;
            new_found.push((primitive, k));
        }
        found = new_found;
    }
    if j > 0 {
        found.insert(0, (vec![QQ_I.one(), QQ_I.zero()], j));
    }
    // _sort_factors (multiple): by length, multiplicity, coefficients (the
    // imaginary parts first)
    found.sort_by(|a, b| {
        let key = |item: &(Vec<GaussianRational>, usize)| -> (usize, usize, Vec<(Rat, Rat)>) {
            (
                item.0.len(),
                item.1,
                item.0.iter().map(|c| (c.y.clone(), c.x.clone())).collect(),
            )
        };
        key(a).cmp(&key(b))
    });
    let x: &Basic = &gens[0];
    let mut factors: Vec<(Basic, usize)> = Vec::new();
    for (fac, k) in found {
        let n: usize = fac.len() - 1;
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in fac.iter().enumerate() {
            if !c.is_zero() {
                terms.push(core_mul(&[
                    QQ_I.to_expr(c)?,
                    core_pow(x, &Basic::int((n - i) as i64))?,
                ])?);
            }
        }
        factors.push((core_add(&terms)?, k));
    }
    Ok((QQ_I.to_expr(&QQ_I.mul(&coeff, &cont))?, factors))
}
