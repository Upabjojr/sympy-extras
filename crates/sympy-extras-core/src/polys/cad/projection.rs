//! Port of `sympy_extras/polys/cad/projection.py`: the projection operators
//! of cylindrical algebraic decomposition. [`mccallum_projection`]
//! (coefficients, discriminants and pairwise resultants of a squarefree
//! basis, reduced by an equational constraint) and [`hong_projection`]
//! (the reducta and their principal subresultant coefficients), the
//! squarefree basis by levels, and the projection factor sets, plain and
//! augmented.

use std::collections::BTreeMap;

use crate::core::basic::Basic;
use crate::error::{polys_error, value_error, PolyError, PolyResult, PolysErrorKind};
use crate::numbers::*;

use super::super::domains::{SparsePolynomialRing, ZZ};
use super::super::euclidtools::dup_inner_subresultants;
use super::super::polytools::{Poly, PolyOrExpr};
use super::super::rings::{Monomial, PolyElement};

/// The coefficients of a polynomial: monomials to rational numbers.
type Terms = BTreeMap<Monomial, Rat>;

/// The projection operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    McCallum,
    Hong,
}

impl Method {
    pub fn from_name(name: &str) -> PolyResult<Method> {
        match name {
            "mccallum" => Ok(Method::McCallum),
            "hong" => Ok(Method::Hong),
            _ => Err(value_error(format!("unknown projection method '{}'", name))),
        }
    }
}

fn polynomial_error(message: String) -> PolyError {
    polys_error(PolysErrorKind::PolynomialError, message)
}

fn from_terms(terms: &Terms, gens: &[Basic]) -> PolyResult<Poly> {
    let items: Vec<(Monomial, Rat)> = terms.iter().map(|(m, c)| (m.clone(), c.clone())).collect();
    Poly::from_dict(&items, gens)
}

/// `f` written in the generators `gens`, which contain the generators it
/// depends on (`_in_gens`).
fn in_gens(f: &Poly, gens: &[Basic]) -> PolyResult<Poly> {
    if f.gens == gens {
        return Ok(f.clone());
    }
    let mut positions: Vec<Option<usize>> = Vec::new();
    for (i, g) in f.gens.iter().enumerate() {
        if let Some(p) = gens.iter().position(|t| t == g) {
            positions.push(Some(p));
        } else if f.degree(i)? > 0 {
            let shown: Vec<String> = gens.iter().map(|g| g.to_string()).collect();
            return Err(polynomial_error(format!(
                "{} depends on {}, which is not in ({})",
                f.to_str()?,
                g,
                shown.join(", ")
            )));
        } else {
            positions.push(None);
        }
    }
    let mut terms: Terms = Terms::new();
    for (m, c) in &f.rep.terms {
        let mut exponents: Monomial = vec![0; gens.len()];
        for (i, e) in m[..positions.len()].iter().enumerate() {
            if *e != 0 {
                exponents[positions[i].expect("a generator it depends on")] = *e;
            }
        }
        terms.insert(exponents, c.clone());
    }
    from_terms(&terms, gens)
}

/// The expressions as polynomials over `ZZ` in `gens`, the rational
/// coefficients cleared of their denominators (`to_polys`).
pub fn to_polys(polys: &[Basic], gens: &[Basic]) -> PolyResult<Vec<Poly>> {
    let mut result: Vec<Poly> = Vec::new();
    for p in polys {
        let q: Poly = match Poly::new(p, gens) {
            Ok(q) => q,
            Err(PolyError::Polys(_, message)) | Err(PolyError::HeuristicGCDFailed(message)) => {
                return Err(polynomial_error(format!(
                    "polynomial with rational coefficients expected: {}",
                    message
                )))
            }
            Err(e) => return Err(e),
        };
        if !q.domain.params.is_empty() {
            return Err(polynomial_error(format!(
                "polynomial with rational coefficients expected, got {}",
                q.to_str()?
            )));
        }
        if q.domain.integral {
            result.push(q);
            continue;
        }
        let mut d: Int = Int::ONE;
        for c in q.rep.terms.values() {
            let den: Int = denom(c);
            d = &d * &den / gcd(&d, &den);
        }
        let scale: Rat = rat_from_int(&d);
        let terms: Terms = q.rep.terms.iter().map(|(m, c)| (m.clone(), c * &scale)).collect();
        result.push(from_terms(&terms, gens)?);
    }
    Ok(result)
}

/// The index of the last generator of `gens` that `f` depends on.
pub fn level_of(f: &Poly, gens: &[Basic]) -> PolyResult<usize> {
    for k in (1..=gens.len()).rev() {
        if let Some(i) = f.gens.iter().position(|g| *g == gens[k - 1]) {
            if f.degree(i)? > 0 {
                return Ok(k);
            }
        }
    }
    Ok(0)
}

fn leading_negative(f: &Poly) -> bool {
    f.rep.terms.values().next_back().is_some_and(rat_is_negative)
}

/// The irreducible non-constant factors with positive leading coefficient.
fn factors(f: &Poly) -> PolyResult<Vec<Poly>> {
    let mut out: Vec<Poly> = Vec::new();
    for (g, _) in f.factor_list()?.1 {
        if g.is_ground() {
            continue;
        }
        out.push(if leading_negative(&g) { g.neg() } else { g });
    }
    Ok(out)
}

/// The finest squarefree basis of `polys`, split by level
/// (`squarefree_basis`).
pub fn squarefree_basis(polys: &[Basic], gens: &[Basic]) -> PolyResult<Vec<Vec<Poly>>> {
    let mut levels: Vec<Vec<Poly>> = vec![Vec::new(); gens.len()];
    for f in to_polys(polys, gens)? {
        for g in factors(&f)? {
            let k: usize = level_of(&g, gens)?;
            if k > 0 && !levels[k - 1].contains(&g) {
                levels[k - 1].push(g);
            }
        }
    }
    Ok(levels)
}

fn main_first(f: &Poly, x: &Basic) -> PolyResult<Poly> {
    let mut gens: Vec<Basic> = vec![x.clone()];
    gens.extend(f.gens.iter().filter(|g| *g != x).cloned());
    in_gens(f, &gens)
}

/// The coefficients in the first generator from the leading one down
/// (zeros included), as terms in the other generators.
fn coefficients(f: &Poly) -> PolyResult<Vec<Terms>> {
    let n: usize = f.degree(0)?.max(0) as usize;
    let mut out: Vec<Terms> = vec![Terms::new(); n + 1];
    for (m, c) in &f.rep.terms {
        out[n - m[0] as usize].insert(m[1..].to_vec(), c.clone());
    }
    Ok(out)
}

fn is_constant(terms: &Terms) -> bool {
    terms.keys().all(|m| m.iter().all(|e| *e == 0))
}

fn poly_of(terms: &Terms, f: &Poly) -> PolyResult<Poly> {
    from_terms(terms, &f.gens[1..])
}

/// The principal subresultant coefficients of `f` and `g` in their first
/// generator, the constant ones left out (`_psc`).
fn psc(f: &Poly, g: &Poly) -> PolyResult<Vec<Poly>> {
    let rest: &[Basic] = &f.gens[1..];
    let names: Vec<String> = (0..rest.len()).map(|i| format!("_r{}", i)).collect();
    let ring: SparsePolynomialRing<super::super::domains::IntegerRing> = SparsePolynomialRing::sympy(ZZ, names);
    let dense = |h: &Poly| -> PolyResult<Vec<PolyElement<super::super::domains::IntegerRing>>> {
        let n: i64 = h.degree(0)?;
        if n < 0 {
            return Ok(Vec::new());
        }
        let mut by_degree: Vec<BTreeMap<Monomial, Int>> = vec![BTreeMap::new(); n as usize + 1];
        for (m, c) in &h.rep.terms {
            by_degree[n as usize - m[0] as usize].insert(m[1..].to_vec(), numer(c));
        }
        Ok(by_degree.into_iter().map(|d| ring.ring.from_dict(d)).collect())
    };
    let (a, b) = (dense(f)?, dense(g)?);
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }
    let (s, t) = dup_inner_subresultants(&a, &b, &ring);
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let m: usize = a.len().min(b.len()) - 1;
    let mut coefficients: Vec<PolyElement<super::super::domains::IntegerRing>> = vec![ring.ring.zero(); m + 1];
    for (r, c) in s.iter().skip(1).zip(t.iter().skip(1)) {
        coefficients[r.len() - 1] = c.clone();
    }
    let mut out: Vec<Poly> = Vec::new();
    for c in coefficients {
        if c.terms.keys().all(|m| m.iter().all(|e| *e == 0)) {
            continue;
        }
        let terms: Terms = c.terms.iter().map(|(m, v)| (m.clone(), rat_from_int(v))).collect();
        out.push(from_terms(&terms, rest)?);
    }
    Ok(out)
}

/// The nonzero reducta in the first generator (`_reducta`).
fn reducta(f: &Poly) -> PolyResult<Vec<Poly>> {
    let mut out: Vec<Poly> = Vec::new();
    let mut terms: Terms = f.rep.terms.clone().into_iter().collect();
    while !terms.is_empty() {
        out.push(from_terms(&terms, &f.gens)?);
        let top: u32 = terms.keys().map(|m| m[0]).max().expect("terms");
        terms.retain(|m, _| m[0] != top);
    }
    Ok(out)
}

/// The irreducible factors of the non-constant polynomials (`_collect`).
fn collect(items: Vec<Poly>) -> PolyResult<Vec<Poly>> {
    let mut result: Vec<Poly> = Vec::new();
    for c in items {
        if c.is_ground() {
            continue;
        }
        for g in factors(&c)? {
            if !result.contains(&g) {
                result.push(g);
            }
        }
    }
    Ok(result)
}

/// McCallum's projection of the squarefree basis `polys` in `x`, reduced
/// by the equational constraints `equational` (a subset of `polys`).
pub fn mccallum_projection(polys: &[Poly], x: &Basic, equational: &[Poly]) -> PolyResult<Vec<Poly>> {
    let main: Vec<Poly> = polys.iter().map(|f| main_first(f, x)).collect::<PolyResult<_>>()?;
    let mut constraints: Vec<Poly> = equational.iter().map(|f| main_first(f, x)).collect::<PolyResult<_>>()?;
    if !constraints.is_empty() {
        if constraints.iter().any(|e| !main.contains(e)) {
            return Err(value_error(
                "the equational constraints must be among the polynomials".to_string(),
            ));
        }
    } else {
        constraints = main.clone();
    }
    let others: Vec<Poly> = main.iter().filter(|f| !constraints.contains(f)).cloned().collect();
    let mut items: Vec<Poly> = Vec::new();
    let push = |value: PolyOrExpr, items: &mut Vec<Poly>| {
        if let PolyOrExpr::Poly(p) = value {
            items.push(p);
        }
    };
    for f in &constraints {
        if f.gens.len() > 1 {
            for c in coefficients(f)? {
                if !is_constant(&c) {
                    items.push(poly_of(&c, f)?);
                } else if !c.is_empty() {
                    break;
                }
            }
        }
        if f.degree(0)? >= 2 {
            push(f.discriminant()?, &mut items);
        }
    }
    for (i, f) in constraints.iter().enumerate() {
        for g in &constraints[i + 1..] {
            push(f.resultant(g)?, &mut items);
        }
    }
    for f in &constraints {
        for g in &others {
            push(f.resultant(g)?, &mut items);
        }
    }
    collect(items)
}

/// Hong's projection of the squarefree basis `polys` in `x`.
pub fn hong_projection(polys: &[Poly], x: &Basic) -> PolyResult<Vec<Poly>> {
    let main: Vec<Poly> = polys.iter().map(|f| main_first(f, x)).collect::<PolyResult<_>>()?;
    if main.iter().all(|f| f.gens.len() == 1) {
        return Ok(Vec::new());
    }
    let mut items: Vec<Poly> = Vec::new();
    for f in &main {
        for r in reducta(f)? {
            items.push(poly_of(&coefficients(&r)?[0], &r)?);
            if r.degree(0)? >= 2 {
                items.extend(psc(&r, &r.diff(0))?);
            }
        }
    }
    for (i, f) in main.iter().enumerate() {
        for g in &main[i + 1..] {
            for r in reducta(f)? {
                if r.degree(0)? == 0 {
                    items.push(poly_of(&coefficients(&r)?[0], &r)?);
                } else {
                    items.extend(psc(&r, g)?);
                }
            }
        }
    }
    collect(items)
}

fn project(method: Method, polys: &[Poly], x: &Basic) -> PolyResult<Vec<Poly>> {
    match method {
        Method::McCallum => mccallum_projection(polys, x, &[]),
        Method::Hong => hong_projection(polys, x),
    }
}

/// The projection factor sets `[P_1, ..., P_n]` of `polys` for `gens`, the
/// last generator projected first (`projection_sets`).
pub fn projection_sets(
    polys: &[Basic],
    gens: &[Basic],
    method: Method,
    equational: Option<&Basic>,
) -> PolyResult<Vec<Vec<Poly>>> {
    if gens.is_empty() {
        return Err(value_error("at least one generator is needed".to_string()));
    }
    let mut levels: Vec<Vec<Poly>> = squarefree_basis(polys, gens)?;
    let n: usize = gens.len();
    let mut constraints: Vec<Poly> = Vec::new();
    if let (Some(e), Method::McCallum) = (equational, method) {
        let constraint: Poly = to_polys(std::slice::from_ref(e), gens)?.remove(0);
        let fs: Vec<Poly> = factors(&constraint)?;
        let mut all_top: bool = true;
        for g in &fs {
            if level_of(g, gens)? != n {
                all_top = false;
            }
        }
        if !fs.is_empty() && all_top {
            constraints = fs;
        }
    }
    let mut result: Vec<Vec<Poly>> = vec![Vec::new(); n];
    for k in (1..=n).rev() {
        let current: Vec<Poly> = levels[k - 1]
            .iter()
            .map(|f| in_gens(f, &gens[..k]))
            .collect::<PolyResult<_>>()?;
        result[k - 1] = current.clone();
        if k == 1 {
            break;
        }
        let projected: Vec<Poly> = if k == n && !constraints.is_empty() {
            let reduced: Vec<Poly> = constraints
                .iter()
                .map(|f| in_gens(f, &gens[..k]))
                .collect::<PolyResult<_>>()?;
            mccallum_projection(&current, &gens[k - 1], &reduced)?
        } else {
            project(method, &current, &gens[k - 1])?
        };
        for g in projected {
            let g: Poly = in_gens(&g, gens)?;
            let j: usize = level_of(&g, gens)?;
            if j > 0 && !levels[j - 1].contains(&g) {
                levels[j - 1].push(g);
            }
        }
    }
    Ok(result)
}

/// The projection factor sets with the factors of `extra` added at their
/// levels and the levels below re-projected (`augmented_projection_sets`).
pub fn augmented_projection_sets(
    projection: &[Vec<Poly>],
    gens: &[Basic],
    extra: &[Basic],
    method: Method,
) -> PolyResult<Vec<Vec<Poly>>> {
    let n: usize = gens.len();
    if projection.len() != n {
        return Err(value_error(format!("expected {} projection factor sets", n)));
    }
    let mut levels: Vec<Vec<Poly>> = projection.to_vec();
    let mut new: Vec<Vec<Poly>> = vec![Vec::new(); n];
    let add = |g: &Poly, levels: &mut Vec<Vec<Poly>>, new: &mut Vec<Vec<Poly>>| -> PolyResult<()> {
        let g: Poly = in_gens(g, gens)?;
        let j: usize = level_of(&g, gens)?;
        if j > 0 {
            let g: Poly = in_gens(&g, &gens[..j])?;
            if !levels[j - 1].contains(&g) {
                levels[j - 1].push(g.clone());
                new[j - 1].push(g);
            }
        }
        Ok(())
    };
    for f in to_polys(extra, gens)? {
        for g in factors(&f)? {
            add(&g, &mut levels, &mut new)?;
        }
    }
    for k in (2..=n).rev() {
        if new[k - 1].is_empty() {
            continue;
        }
        let projected: Vec<Poly> = match method {
            Method::McCallum => mccallum_projection(&levels[k - 1], &gens[k - 1], &new[k - 1])?,
            Method::Hong => project(method, &levels[k - 1], &gens[k - 1])?,
        };
        for g in projected {
            add(&g, &mut levels, &mut new)?;
        }
    }
    Ok(levels)
}
