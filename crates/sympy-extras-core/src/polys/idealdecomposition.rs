//! Decompositions of ideals over the rationals, ported from
//! `sympy_extras.polys.idealdecomposition`: the radical (the intersection
//! of the saturated ideals of squarefree regular chains in Kalkbrener's
//! sense), the equidimensional parts, the minimal primes (a separating
//! linear form of each chain and the factors of its minimal polynomial),
//! a maximal independent set, and the primary decomposition of Gianni,
//! Trager and Zacharias with the separators of Shimoyama and Yokoyama;
//! associated primes and the primality test.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::basic::Basic;
use crate::core::symbol::Dummy;
use crate::error::{arithmetic_error, value_error, PolyResult};

use super::domains::{Domain, RationalField, QQ};
use super::ideals::{groebner, separating_forms, Ideal, IdealOrPoly, PolyLike};
use super::polytools::Poly;
use super::regularchains::regularchain::{triangularize, RegularChain};
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing};

/// An ideal over the rationals.
pub type RationalIdeal = Ideal<RationalField>;

/// A primary ideal and its associated prime.
pub type PrimaryComponent = (RationalIdeal, RationalIdeal);

fn like(ideal: &RationalIdeal, exprs: &[Basic]) -> PolyResult<RationalIdeal> {
    Ideal::new(exprs, &ideal.symbols, QQ, ideal.order.clone())
}

fn whole_ring(ideal: &RationalIdeal) -> PolyResult<RationalIdeal> {
    like(ideal, &[Basic::one()])
}

/// The saturated ideal of a chain (`RegularChain.saturated_ideal`).
pub fn saturated_ideal(chain: &RegularChain) -> PolyResult<RationalIdeal> {
    let polys: Vec<Basic> = chain.polys()?;
    let given: Vec<Basic> = if polys.is_empty() {
        vec![Basic::zero()]
    } else {
        polys.clone()
    };
    let ideal: RationalIdeal = Ideal::new(&given, chain.symbols(), QQ, MonomialOrder::Grevlex)?;
    if polys.is_empty() {
        return Ok(ideal);
    }
    let product: Basic = crate::core::mul::mul(&chain.initials()?)?;
    ideal.saturate(&IdealOrPoly::Poly(PolyLike::Expr(&product)))
}

/// Squarefree regular chains whose saturated ideals intersect in the
/// radical, without those which contain another one (`_components`).
fn components(ideal: &RationalIdeal) -> PolyResult<Vec<(RegularChain, RationalIdeal)>> {
    let mut found: Vec<(RegularChain, RationalIdeal)> = Vec::new();
    for chain in triangularize(&ideal.exprs()?, &ideal.symbols, &[], true)? {
        let saturated: RationalIdeal = saturated_ideal(&chain)?;
        found.push((chain, like(ideal, &saturated.exprs()?)?.reduced()?));
    }
    let mut kept: Vec<(RegularChain, RationalIdeal)> = Vec::new();
    for i in 0..found.len() {
        let mut redundant: bool = false;
        for j in 0..found.len() {
            if i != j && found[j].1.subset(&found[i].1)? && (j < i || !found[i].1.subset(&found[j].1)?) {
                redundant = true;
                break;
            }
        }
        if !redundant {
            kept.push(found[i].clone());
        }
    }
    Ok(kept)
}

fn intersection(ideals: &[RationalIdeal], within: &RationalIdeal) -> PolyResult<RationalIdeal> {
    let mut result: Option<RationalIdeal> = None;
    for ideal in ideals {
        result = Some(match result {
            None => ideal.clone(),
            Some(r) => r.intersect(ideal)?,
        });
    }
    match result {
        Some(r) => r.reduced(),
        None => whole_ring(within)?.reduced(),
    }
}

/// `ideal : other**oo` for a radical ideal (`_without`).
fn without(ideal: &RationalIdeal, other: &RationalIdeal) -> PolyResult<RationalIdeal> {
    let mut parts: Vec<RationalIdeal> = Vec::new();
    for g in &other.gens {
        if !ideal.contains(&PolyLike::Element(g))? {
            parts.push(ideal.saturate(&IdealOrPoly::Poly(PolyLike::Element(g)))?);
        }
    }
    if parts.is_empty() {
        return whole_ring(ideal);
    }
    let mut proper: Vec<RationalIdeal> = Vec::new();
    for part in parts {
        if !part.is_whole_ring()? {
            proper.push(part);
        }
    }
    intersection(&proper, ideal)
}

/// The radical, as the intersection of the saturated ideals of squarefree
/// regular chains (`radical`).
pub fn radical(ideal: &RationalIdeal) -> PolyResult<RationalIdeal> {
    if ideal.gens.is_empty() {
        return Ok(ideal.clone());
    }
    let saturated: Vec<RationalIdeal> = components(ideal)?.into_iter().map(|(_, s)| s).collect();
    intersection(&saturated, ideal)
}

/// The equidimensional parts of the radical, by decreasing dimension.
pub fn equidimensional_parts(ideal: &RationalIdeal) -> PolyResult<Vec<RationalIdeal>> {
    if ideal.gens.is_empty() {
        return Ok(vec![ideal.clone()]);
    }
    let mut by_dimension: Vec<(usize, Vec<RationalIdeal>)> = Vec::new();
    for (chain, saturated) in components(ideal)? {
        let d: usize = chain.dimension();
        match by_dimension.iter_mut().find(|(k, _)| *k == d) {
            Some(entry) => entry.1.push(saturated),
            None => by_dimension.push((d, vec![saturated])),
        }
    }
    by_dimension.sort_by_key(|b| std::cmp::Reverse(b.0));
    let mut parts: Vec<RationalIdeal> = Vec::new();
    for (_, ideals) in &by_dimension {
        let mut part: RationalIdeal = intersection(ideals, ideal)?;
        for greater in &parts {
            part = without(&part, greater)?;
        }
        if !part.is_whole_ring()? {
            parts.push(part);
        }
    }
    Ok(parts)
}

fn degree_in_first<K: Domain>(f: &PolyElement<K>) -> i64 {
    f.terms.keys().map(|m| m[0] as i64).max().unwrap_or(-1)
}

/// A separating linear form in the main variables of the chain and its
/// minimal polynomial over the free ones, in a new variable
/// (`_separated`).
fn separated(chain: &RegularChain, saturated: &RationalIdeal) -> PolyResult<(Basic, Basic, Basic)> {
    let (main, free, degree) = (chain.main_variables(), chain.free_variables(), chain.degree());
    let y: Basic = Dummy::new("y", &[]).0;
    for form in separating_forms(&main, degree as usize)? {
        let mut exprs: Vec<Basic> = saturated.exprs()?;
        exprs.push(crate::core::add::add(&[
            y.clone(),
            crate::core::mul::mul(&[Basic::int(-1), form.clone()])?,
        ])?);
        let mut symbols: Vec<Basic> = main.clone();
        symbols.push(y.clone());
        symbols.extend(free.iter().cloned());
        let extended: RationalIdeal = Ideal::new(&exprs, &symbols, QQ, saturated.order.clone())?;
        let eliminated: RationalIdeal = extended.eliminate(&main)?;
        let eliminant: Vec<PolyElement<RationalField>> = eliminated.groebner_basis(None)?;
        if eliminant.len() != 1 {
            return Err(arithmetic_error(
                "the eliminant of an unmixed ideal is not principal".to_string(),
            ));
        }
        if degree_in_first(&eliminant[0]) == degree {
            return Ok((form, eliminated.to_expr(&eliminant[0])?, y));
        }
    }
    Err(arithmetic_error("no separating linear form was found".to_string()))
}

/// The prime components of the saturated ideal of a squarefree chain
/// (`_primes`).
fn primes(chain: &RegularChain, saturated: &RationalIdeal) -> PolyResult<Vec<RationalIdeal>> {
    if chain.degree() == 1 {
        return Ok(vec![saturated.clone()]);
    }
    let (form, minimal, y) = separated(chain, saturated)?;
    let mut variables: Vec<Basic> = vec![y.clone()];
    variables.extend(chain.free_variables());
    let (_, found) = Poly::new(&minimal, &variables)?.factor_list()?;
    let rule: HashMap<Basic, Basic> = HashMap::from([(y, form)]);
    let mut factors: Vec<Basic> = Vec::new();
    for (f, _) in &found {
        if f.degree(0)? > 0 {
            factors.push(crate::core::function::xreplace_evaluated(&f.as_expr()?, &rule)?);
        }
    }
    if factors.len() == 1 {
        return Ok(vec![saturated.clone()]);
    }
    let mut result: Vec<RationalIdeal> = Vec::new();
    for j in 0..factors.len() {
        let others: Vec<Basic> = factors
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != j)
            .map(|(_, f)| f.clone())
            .collect();
        let product: Basic = crate::core::mul::mul(&others)?;
        result.push(
            saturated
                .saturate(&IdealOrPoly::Poly(PolyLike::Expr(&product)))?
                .reduced()?,
        );
    }
    Ok(result)
}

fn sort_by_dimension(
    ideals: Vec<RationalIdeal>,
    prime_of: impl Fn(&RationalIdeal) -> &RationalIdeal,
) -> PolyResult<Vec<RationalIdeal>> {
    let mut keyed: Vec<(i64, String, RationalIdeal)> = Vec::new();
    for i in ideals {
        let p: &RationalIdeal = prime_of(&i);
        keyed.push((-p.dimension()?, p.to_string(), i.clone()));
    }
    keyed.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    Ok(keyed.into_iter().map(|(_, _, i)| i).collect())
}

/// The minimal primes, by decreasing dimension (`minimal_primes`).
pub fn minimal_primes(ideal: &RationalIdeal) -> PolyResult<Vec<RationalIdeal>> {
    if ideal.gens.is_empty() {
        return Ok(vec![ideal.clone()]);
    }
    let mut candidates: Vec<RationalIdeal> = Vec::new();
    for (chain, saturated) in components(ideal)? {
        for prime in primes(&chain, &saturated)? {
            let mut known: bool = false;
            for other in &candidates {
                if prime.equals(other)? {
                    known = true;
                    break;
                }
            }
            if !known {
                candidates.push(prime);
            }
        }
    }
    let mut minimal: Vec<RationalIdeal> = Vec::new();
    for (i, p) in candidates.iter().enumerate() {
        let mut contains_another: bool = false;
        for (j, q) in candidates.iter().enumerate() {
            if i != j && q.subset(p)? {
                contains_another = true;
                break;
            }
        }
        if !contains_another {
            minimal.push(p.clone());
        }
    }
    sort_by_dimension(minimal, |p| p)
}

/// A maximal independent set of variables modulo a proper ideal
/// (`independent_set`).
pub fn independent_set(ideal: &RationalIdeal) -> PolyResult<Vec<Basic>> {
    if ideal.is_whole_ring()? {
        return Err(value_error("the whole ring has no independent set".to_string()));
    }
    let leading: Vec<Monomial> = ideal.leading_monomials(Some(&MonomialOrder::Grevlex))?;
    let n: usize = ideal.symbols.len();
    let size: usize = ideal.dimension()? as usize;
    for subset in combinations(n, size) {
        if leading
            .iter()
            .all(|m| m.iter().enumerate().any(|(i, e)| *e > 0 && !subset.contains(&i)))
        {
            return Ok(subset.iter().map(|i| ideal.symbols[*i].clone()).collect());
        }
    }
    Err(arithmetic_error(
        "no independent set of the size of the dimension".to_string(),
    ))
}

/// The `k`-subsets of `0..n` in the order of `itertools.combinations`.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    if k > n {
        return result;
    }
    let mut current: Vec<usize> = (0..k).collect();
    loop {
        result.push(current.clone());
        let Some(i) = (0..k).rev().find(|i| current[*i] != *i + n - k) else {
            return result;
        };
        current[i] += 1;
        for j in i + 1..k {
            current[j] = current[j - 1] + 1;
        }
    }
}

/// The contraction of the extension to `K(u)[x]` and the polynomial `h` of
/// `K[u]` with contraction `I : h**oo` (`_contraction`).
fn contraction(ideal: &RationalIdeal, free: &[Basic]) -> PolyResult<(RationalIdeal, Basic)> {
    let main: Vec<usize> = (0..ideal.symbols.len())
        .filter(|i| !free.contains(&ideal.symbols[*i]))
        .collect();
    if free.is_empty() || main.is_empty() {
        return Ok((ideal.clone(), Basic::one()));
    }
    let positions: Vec<usize> = main
        .iter()
        .copied()
        .chain(
            free.iter()
                .map(|s| ideal.symbols.iter().position(|t| t == s).expect("a variable")),
        )
        .collect();
    let k: usize = main.len();
    let names: Vec<String> = (0..positions.len()).map(|i| format!("v{}", i)).collect();
    let r = PolyRing::new(
        QQ,
        names,
        MonomialOrder::elimination(k, positions.len(), MonomialOrder::Grevlex),
    )?;
    let gens: Vec<PolyElement<RationalField>> = ideal
        .gens
        .iter()
        .map(|g| {
            r.from_dict(
                g.terms
                    .iter()
                    .map(|(m, c)| (positions.iter().map(|i| m[*i]).collect(), c.clone()))
                    .collect(),
            )
        })
        .collect();
    let basis: Vec<PolyElement<RationalField>> = groebner(&gens, &r)?;
    let mut h: Poly = Poly::new(&Basic::one(), free)?;
    for g in &basis {
        let lm: Monomial = g.LM();
        let top: &[u32] = &lm[..k];
        let mut terms: Vec<Basic> = Vec::new();
        for (m, c) in &g.terms {
            if &m[..k] == top {
                let mut factors: Vec<Basic> = vec![QQ.to_expr(c)?];
                for (v, e) in free.iter().zip(&m[k..]) {
                    if *e > 0 {
                        factors.push(crate::core::power::pow(v, &Basic::int(*e as i64))?);
                    }
                }
                terms.push(crate::core::mul::mul(&factors)?);
            }
        }
        h = h.lcm(&Poly::new(&crate::core::add::add(&terms)?, free)?)?;
    }
    let (_, parts) = h.sqf_list()?;
    if parts.is_empty() {
        return Ok((ideal.clone(), Basic::one()));
    }
    let factor: Basic = crate::core::mul::mul(
        &parts
            .iter()
            .map(|(p, _)| p.as_expr())
            .collect::<PolyResult<Vec<Basic>>>()?,
    )?;
    Ok((
        ideal.saturate(&IdealOrPoly::Poly(PolyLike::Expr(&factor)))?.reduced()?,
        factor,
    ))
}

/// A polynomial in every one of the `others` and not in `prime`
/// (`_separator`).
fn separator(prime: &RationalIdeal, others: &[RationalIdeal]) -> PolyResult<Basic> {
    let mut factors: Vec<Basic> = Vec::new();
    for other in others {
        let mut outside: Vec<PolyElement<RationalField>> = Vec::new();
        for g in other.groebner_basis(None)? {
            if !prime.contains(&PolyLike::Element(&g))? {
                outside.push(g);
            }
        }
        let key = |g: &PolyElement<RationalField>| {
            (
                g.terms.keys().map(|m| m.iter().sum::<u32>()).max().unwrap_or(0),
                g.terms.len(),
            )
        };
        let chosen = outside
            .iter()
            .min_by(|a, b| key(a).cmp(&key(b)))
            .ok_or_else(|| arithmetic_error("a prime contains another one of the same dimension".to_string()))?;
        factors.push(other.to_expr(chosen)?);
    }
    crate::core::mul::mul(&factors)
}

/// The primary components of an ideal of dimension zero
/// (`_zero_dimensional_components`).
fn zero_dimensional_components(ideal: &RationalIdeal) -> PolyResult<Vec<PrimaryComponent>> {
    use super::densearith::dup_quo;
    use super::densetools::dup_diff;
    use super::euclidtools::dup_gcd;
    let root: RationalIdeal = ideal.radical()?;
    let points: usize = root.vector_space_dimension()?;
    let y: Basic = Dummy::new("y", &[]).0;
    for form in separating_forms(&ideal.symbols, points)? {
        let minimal = ideal.minimal_polynomial(&PolyLike::Expr(&form))?;
        let sqf = dup_quo(&minimal, &dup_gcd(&minimal, &dup_diff(&minimal, 1, &QQ), &QQ), &QQ)?;
        if sqf.len() - 1 < points {
            continue;
        }
        let rule: HashMap<Basic, Basic> = HashMap::from([(y.clone(), form.clone())]);
        let mut result: Vec<PrimaryComponent> = Vec::new();
        for (factor, k) in QQ.factor_dense(&minimal)?.1 {
            let value: Basic =
                crate::core::function::xreplace_evaluated(&super::ideals::dense_to_expr(&factor, &QQ, &y)?, &rule)?;
            let power: RationalIdeal = like(ideal, &[crate::core::power::pow(&value, &Basic::int(k as i64))?])?;
            let prime: RationalIdeal = like(ideal, &[value])?;
            result.push((ideal.add(&power)?.reduced()?, root.add(&prime)?.reduced()?));
        }
        return Ok(result);
    }
    Err(arithmetic_error("no separating linear form was found".to_string()))
}

/// The top-dimensional primary components, their intersection and `h`
/// (`_top_components`).
fn top_components(ideal: &RationalIdeal) -> PolyResult<(Vec<PrimaryComponent>, RationalIdeal, Basic)> {
    let free: Vec<Basic> = independent_set(ideal)?;
    if free.is_empty() {
        return Ok((zero_dimensional_components(ideal)?, ideal.clone(), Basic::one()));
    }
    let (contracted, h) = contraction(ideal, &free)?;
    let found: Vec<RationalIdeal> = minimal_primes(&contracted)?;
    let mut unmixed: bool = !found.is_empty();
    for p in &found {
        if p.dimension()? != free.len() as i64 {
            unmixed = false;
        }
    }
    if !unmixed {
        return Err(arithmetic_error(
            "the contraction of the extension is not unmixed".to_string(),
        ));
    }
    if found.len() == 1 {
        return Ok((vec![(contracted.clone(), found[0].clone())], contracted, h));
    }
    let mut result: Vec<PrimaryComponent> = Vec::new();
    for (i, prime) in found.iter().enumerate() {
        let others: Vec<RationalIdeal> = found
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| p.clone())
            .collect();
        let s: Basic = separator(prime, &others)?;
        result.push((
            contracted.saturate(&IdealOrPoly::Poly(PolyLike::Expr(&s)))?.reduced()?,
            prime.clone(),
        ));
    }
    Ok((result, contracted, h))
}

/// The least `m` with `I : h**m = I : h**oo` (`_saturation_exponent`).
fn saturation_exponent(ideal: &RationalIdeal, saturated: &RationalIdeal, h: &Basic) -> PolyResult<usize> {
    let mut m: usize = 0;
    let factor: PolyElement<RationalField> = ideal.normal_form(&PolyLike::Expr(h), None)?;
    for g in &saturated.gens {
        let mut product: PolyElement<RationalField> = ideal.normal_form(&PolyLike::Element(g), None)?;
        for _ in 0..m {
            product = ideal.normal_form(&PolyLike::Element(&product.mul(&factor)), None)?;
        }
        while !product.is_zero() {
            m += 1;
            product = ideal.normal_form(&PolyLike::Element(&product.mul(&factor)), None)?;
        }
    }
    Ok(m)
}

/// An irredundant primary decomposition over the rationals, by decreasing
/// dimension of the primes (`primary_decomposition`).
pub fn primary_decomposition(ideal: &RationalIdeal) -> PolyResult<Vec<PrimaryComponent>> {
    if ideal.is_whole_ring()? {
        return Ok(Vec::new());
    }
    if ideal.gens.is_empty() {
        return Ok(vec![(ideal.clone(), ideal.clone())]);
    }
    let mut found: Vec<PrimaryComponent> = Vec::new();
    let mut current: RationalIdeal = ideal.reduced()?;
    let mut found_so_far: Option<RationalIdeal> = None;
    while !current.is_whole_ring()? {
        let (parts, contracted, h) = top_components(&current)?;
        found.extend(parts);
        if contracted.equals(&current)? {
            break;
        }
        let so_far: RationalIdeal = match found_so_far {
            None => contracted.clone(),
            Some(previous) => previous.intersect(&contracted)?.reduced()?,
        };
        let mut done: bool = true;
        for g in &so_far.gens {
            if !ideal.contains(&PolyLike::Element(g))? {
                done = false;
                break;
            }
        }
        found_so_far = Some(so_far);
        if done {
            break;
        }
        let m: usize = saturation_exponent(&current, &contracted, &h)?;
        let power: RationalIdeal = like(ideal, &[crate::core::power::pow(&h, &Basic::int(m as i64))?])?;
        current = current.add(&power)?.reduced()?;
    }
    let mut merged: Vec<PrimaryComponent> = Vec::new();
    for (q, p) in found {
        let mut placed: bool = false;
        for entry in merged.iter_mut() {
            if p.equals(&entry.1)? {
                entry.0 = entry.0.intersect(&q)?.reduced()?;
                placed = true;
                break;
            }
        }
        if !placed {
            merged.push((q, p));
        }
    }
    let mut kept: Vec<usize> = (0..merged.len()).collect();
    for index in (0..merged.len()).rev() {
        let others: Vec<usize> = kept.iter().copied().filter(|i| *i != index).collect();
        if !others.is_empty() {
            let qs: Vec<RationalIdeal> = others.iter().map(|i| merged[*i].0.clone()).collect();
            if intersection(&qs, &merged[index].0)?.subset(&merged[index].0)? {
                kept = others;
            }
        }
    }
    let components: Vec<PrimaryComponent> = kept.into_iter().map(|i| merged[i].clone()).collect();
    let mut keyed: Vec<(i64, String, PrimaryComponent)> = Vec::new();
    for c in components {
        keyed.push((-c.1.dimension()?, c.1.to_string(), c));
    }
    keyed.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    Ok(keyed.into_iter().map(|(_, _, c)| c).collect())
}

/// The associated primes, minimal and embedded, by decreasing dimension.
pub fn associated_primes(ideal: &RationalIdeal) -> PolyResult<Vec<RationalIdeal>> {
    Ok(primary_decomposition(ideal)?.into_iter().map(|(_, p)| p).collect())
}

/// Whether the ideal is primary (`is_primary`).
pub fn is_primary(ideal: &RationalIdeal) -> PolyResult<bool> {
    if ideal.is_whole_ring()? {
        return Ok(false);
    }
    if ideal.gens.is_empty() {
        return Ok(true);
    }
    let (contracted, _) = contraction(ideal, &independent_set(ideal)?)?;
    Ok(contracted.equals(ideal)? && minimal_primes(ideal)?.len() == 1)
}
