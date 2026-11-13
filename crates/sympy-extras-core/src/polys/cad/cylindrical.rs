//! Port of `sympy_extras/polys/cad/cylindrical.py`: cylindrical
//! descriptions of semialgebraic sets (`cylindrical_formula`,
//! `cylindrical_set`, `cylindrical_cases`, and the last resort of
//! `quantifier_elimination`): the true cells of a decomposition described
//! level by level, each variable between root functions of the ones
//! before (rational functions, square roots, `IndexedRoot`), sections
//! joined to their neighbours when the descriptions meet at the boundary.
//! Python gives the decision that two bounds meet two seconds
//! (`attempt`); here it is decided without a time limit.

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::add::{add, sub};
use crate::core::assumptions::{is_number, Predicate};
use crate::core::basic::{Basic, Head};
use crate::core::expr::{is, is_false};
use crate::core::exprtools::factor_terms;
use crate::core::logic::{and_, false_, or_, true_};
use crate::core::mul::{div, mul, neg};
use crate::core::power::pow;
use crate::core::relational::{eq, ge, gt, le, lt};
use crate::core::subs::subs;
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::Int;
use crate::polys::polytools::Poly;
use crate::sets::conditionset::condition_set;
use crate::sets::sets::{empty_set, finite_set, interval, point_set, product_set, reals, union};

use super::indexedroot::{indexed_root, real_number};
use super::lifting::{Cad, CadCell};
use super::projection::Method;
use super::qe::{truth_values, CellTruth, Prune, Quantifier};
use super::samplepoints::{order_of, Real, Sign};

/// A real root over a cell: the `index`-th of `poly` (`_Root`).
struct Root {
    poly: Poly,
    index: usize,
    value: Real,
    degree: usize,
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

/// The coefficients in the last generator, the leading one first
/// (`_coefficients`).
fn coefficients(poly: &Poly, gens: &[Basic]) -> PolyResult<Vec<Basic>> {
    Poly::new(
        &poly.as_expr()?,
        std::slice::from_ref(gens.last().expect("a generator")),
    )?
    .all_coeffs()
}

/// The degree over the parent cell, the coefficients from it down and
/// their signs there (`_degree_over`).
fn degree_over(poly: &Poly, parent: &CadCell, gens: &[Basic]) -> PolyResult<(usize, Vec<Basic>, Vec<Sign>)> {
    let mut cs: Vec<Basic> = coefficients(poly, gens)?;
    let lower: &[Basic] = &gens[..gens.len() - 1];
    let mut signs: Vec<Sign> = Vec::new();
    for c in &cs {
        if lower.is_empty() {
            let Some(v) = c.as_rational() else {
                return Err(value_error(format!("a rational coefficient expected, got {}", c)));
            };
            signs.push(if v.is_zero() {
                0
            } else if crate::numbers::rat_is_negative(v) {
                -1
            } else {
                1
            });
        } else {
            signs.push(parent.sample()?.sign(c, lower)?);
        }
    }
    while !signs.is_empty() && signs[0] == 0 {
        signs.remove(0);
        cs.remove(0);
    }
    Ok((cs.len().saturating_sub(1), cs, signs))
}

/// The distinct real roots of the polynomials over the parent cell, each
/// as a root of the polynomial of least degree which has it
/// (`_root_sequence`).
fn root_sequence(parent: &CadCell, polys: &[Poly], gens: &[Basic]) -> PolyResult<Vec<Root>> {
    let mut found: Vec<Root> = Vec::new();
    for poly in polys {
        let roots: Vec<Real> = parent.sample()?.real_roots(&poly.as_expr()?, gens)?.unwrap_or_default();
        if roots.is_empty() {
            continue;
        }
        let degree: usize = degree_over(poly, parent, gens)?.0;
        for (index, value) in roots.into_iter().enumerate() {
            found.push(Root {
                poly: poly.clone(),
                index,
                value,
                degree,
            });
        }
    }
    let values: Vec<Real> = found.iter().map(|r| r.value.clone()).collect();
    let mut out: Vec<Root> = Vec::new();
    for group in order_of(&values)? {
        // min returns the first of the least
        let mut best: usize = group[0];
        for i in &group {
            if found[*i].degree < found[best].degree {
                best = *i;
            }
        }
        let r: &Root = &found[best];
        out.push(Root {
            poly: r.poly.clone(),
            index: r.index,
            value: r.value.clone(),
            degree: r.degree,
        });
    }
    Ok(out)
}

/// The root as an expression in the variables of the parent cell
/// (`_root_function`).
fn root_function(root: &Root, parent: &CadCell, gens: &[Basic]) -> PolyResult<Basic> {
    let x: &Basic = gens.last().expect("a generator");
    if gens.len() == 1 {
        return real_number(&root.value, &x.to_string());
    }
    let (degree, cs, signs) = degree_over(&root.poly, parent, gens)?;
    if degree == 1 {
        return div(&neg(&cs[1])?, &cs[0]);
    }
    if degree == 2 {
        let (a, b, c) = (&cs[0], &cs[1], &cs[2]);
        let roots: Option<Vec<Real>> = parent.sample()?.real_roots(&root.poly.as_expr()?, gens)?;
        let two_a: Basic = mul(&[int(2), a.clone()])?;
        if roots.as_ref().is_some_and(|r| r.len() == 1) {
            return div(&neg(b)?, &two_a);
        }
        let branch: i64 = (if signs[0] > 0 { -1 } else { 1 }) * (if root.index == 0 { 1 } else { -1 });
        let discriminant: Basic = sub(&pow(b, &int(2))?, &mul(&[int(4), a.clone(), c.clone()])?)?;
        let radical: Basic = pow(&factor_terms(&discriminant, false, false, false, true)?, &Basic::half())?;
        let numerator: Basic = add(&[neg(b)?, mul(&[int(branch), radical])?])?;
        return div(&numerator, &two_a);
    }
    indexed_root(&root.poly.as_expr()?, x, root.index)
}

/// Consecutive cells of a stack with one description of the higher
/// coordinates (`_Run`).
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub lower: Option<Basic>,
    pub lower_closed: bool,
    pub upper: Option<Basic>,
    pub upper_closed: bool,
    pub inner: Description,
}

impl Run {
    fn point(&self) -> bool {
        self.lower.is_some() && self.lower == self.upper
    }
}

/// The true cells over a cell: all of the stack, none, or runs
/// (`_Description`).
#[derive(Debug, Clone, PartialEq)]
pub enum Description {
    All(bool),
    Runs(Vec<Run>),
}

fn has_head(e: &Basic, test: &dyn Fn(&Head) -> bool) -> bool {
    test(e.head()) || e.args().iter().any(|a| has_head(a, test))
}

fn has_indexed_root(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::IndexedRoot))
}

/// The description at `x = value` (`_specialized`); `None` when not
/// decided.
fn specialized(description: &Description, x: &Basic, value: &Basic, own: bool) -> PolyResult<Option<Description>> {
    let runs_in: &[Run] = match description {
        Description::All(b) => return Ok(Some(Description::All(*b))),
        Description::Runs(r) => r,
    };
    let mut runs: Vec<Run> = Vec::new();
    for run in runs_in {
        let mut ends: Vec<Option<Basic>> = Vec::new();
        for end in [&run.lower, &run.upper] {
            let Some(end) = end else {
                ends.push(None);
                continue;
            };
            if !own && has_indexed_root(end) && end.has(x) {
                return Ok(None);
            }
            let moved: Basic = subs(end, x, value)?;
            let bad = |h: &Head| {
                matches!(
                    h,
                    Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
                )
            };
            if has_head(&moved, &bad) || is_false(&moved, Predicate::extended_real) {
                return Ok(None);
            }
            ends.push(Some(moved));
        }
        let (lower, mut upper) = (ends[0].clone(), ends[1].clone());
        let Some(inner) = specialized(&run.inner, x, value, own)? else {
            return Ok(None);
        };
        if let (Some(l), Some(u)) = (&lower, &upper) {
            if l != u {
                let gap: Basic = sub(u, l)?;
                if gap.free_symbols().is_empty() {
                    // attempt(gap.is_zero): None when undecided or failing
                    match is(&gap, Predicate::zero) {
                        None => return Ok(None),
                        Some(true) => upper = lower.clone(),
                        Some(false) => {}
                    }
                }
            }
        }
        if lower.is_some() && lower == upper && !(run.lower_closed && run.upper_closed) {
            continue;
        }
        if inner == Description::All(false) {
            continue;
        }
        let moved_run = Run {
            lower,
            lower_closed: run.lower_closed,
            upper,
            upper_closed: run.upper_closed,
            inner,
        };
        if moved_run.point() && runs.contains(&moved_run) {
            continue;
        }
        runs.push(moved_run);
    }
    Ok(Some(if runs.is_empty() {
        Description::All(false)
    } else {
        Description::Runs(runs)
    }))
}

type Children = HashMap<Vec<usize>, Vec<Arc<CadCell>>>;

/// The description of the true cells over `parent` (`_stack`).
fn stack(
    cad: &Cad,
    children: &Children,
    truth: &HashMap<Vec<usize>, bool>,
    parent: &Arc<CadCell>,
    level: usize,
) -> PolyResult<Description> {
    let gens: &[Basic] = &cad.gens[..level];
    let x: &Basic = &gens[level - 1];
    let cells: Vec<Arc<CadCell>> = children.get(&parent.index).cloned().unwrap_or_default();
    let mut descriptions: Vec<Description> = Vec::new();
    for cell in &cells {
        match truth.get(&cell.index) {
            Some(v) => descriptions.push(Description::All(*v)),
            None => descriptions.push(stack(cad, children, truth, cell, level + 1)?),
        }
    }
    if descriptions.iter().all(|d| *d == Description::All(false)) {
        return Ok(Description::All(false));
    }
    let roots: Vec<Root> = root_sequence(parent, &cad.projection[level - 1], gens)?;
    let mut functions: HashMap<usize, Basic> = HashMap::new();
    let mut bound = |position: usize| -> PolyResult<Basic> {
        if let Some(f) = functions.get(&position) {
            return Ok(f.clone());
        }
        let f: Basic = root_function(&roots[(position - 1) / 2], parent, gens)?;
        functions.insert(position, f.clone());
        Ok(f)
    };
    for i in 0..cells.len() {
        if !cells[i].is_section() || descriptions[i] == Description::All(false) {
            continue;
        }
        let b: Basic = bound(i)?;
        let Some(here) = specialized(&descriptions[i], x, &b, true)? else {
            continue;
        };
        for j in [i.wrapping_sub(1), i + 1] {
            if j < cells.len() && descriptions[j] != Description::All(false) && descriptions[j] != descriptions[i] {
                let there: Option<Description> = specialized(&descriptions[j], x, &b, false)?;
                if there.as_ref() == Some(&here) {
                    descriptions[i] = descriptions[j].clone();
                    break;
                }
            }
        }
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut i: usize = 0;
    while i < cells.len() {
        if descriptions[i] == Description::All(false) {
            i += 1;
            continue;
        }
        let mut j: usize = i;
        while j + 1 < cells.len() && descriptions[j + 1] == descriptions[i] {
            j += 1;
        }
        let mut inner: Description = descriptions[i].clone();
        if i == j && cells[i].is_section() {
            let value: Basic = bound(i)?;
            if is_number(&value) && !has_indexed_root(&value) {
                if let Some(at_value) = specialized(&inner, x, &value, true)? {
                    inner = at_value;
                }
            }
            runs.push(Run {
                lower: Some(value.clone()),
                lower_closed: true,
                upper: Some(value),
                upper_closed: true,
                inner,
            });
        } else {
            let lower: Option<Basic> = if cells[i].is_section() {
                Some(bound(i)?)
            } else if i > 0 {
                Some(bound(i - 1)?)
            } else {
                None
            };
            let upper: Option<Basic> = if cells[j].is_section() {
                Some(bound(j)?)
            } else if j + 1 < cells.len() {
                Some(bound(j + 1)?)
            } else {
                None
            };
            runs.push(Run {
                lower,
                lower_closed: cells[i].is_section(),
                upper,
                upper_closed: cells[j].is_section(),
                inner,
            });
        }
        i = j + 1;
    }
    if runs.len() == 1 && runs[0].lower.is_none() && runs[0].upper.is_none() && runs[0].inner == Description::All(true)
    {
        return Ok(Description::All(true));
    }
    Ok(Description::Runs(runs))
}

/// `_formula`.
fn formula_of(description: &Description, gens: &[Basic], level: usize) -> PolyResult<Basic> {
    let runs: &[Run] = match description {
        Description::All(b) => return Ok(if *b { true_() } else { false_() }),
        Description::Runs(r) => r,
    };
    let x: &Basic = &gens[level - 1];
    let mut parts: Vec<Basic> = Vec::new();
    for run in runs {
        let mut conditions: Vec<Basic> = Vec::new();
        if run.point() {
            conditions.push(eq(x, run.lower.as_ref().expect("a bound"))?);
        } else {
            if let Some(l) = &run.lower {
                conditions.push(if run.lower_closed { ge(x, l)? } else { gt(x, l)? });
            }
            if let Some(u) = &run.upper {
                conditions.push(if run.upper_closed { le(x, u)? } else { lt(x, u)? });
            }
        }
        conditions.push(formula_of(&run.inner, gens, level + 1)?);
        parts.push(and_(&conditions));
    }
    Ok(or_(&parts))
}

/// The description of the true cells of the space of the first `last`
/// variables (`_described`).
fn described(cad: &Cad, last: usize, cells: &[CellTruth]) -> PolyResult<Description> {
    let truth: HashMap<Vec<usize>, bool> = cells.iter().map(|(c, v)| (c.index.clone(), *v)).collect();
    let mut children: Children = HashMap::new();
    for level in 1..=last {
        for cell in cad.cells_at(level)? {
            let parent: &Arc<CadCell> = cell.parent.as_ref().expect("a parent");
            children.entry(parent.index.clone()).or_default().push(cell.clone());
        }
    }
    let root: Arc<CadCell> = cad.cells_at(1)?[0].parent.clone().expect("the root");
    stack(cad, &children, &truth, &root, 1)
}

/// The cylindrical formula of the true cells (for
/// `quantifier_elimination`, `described_by_root_functions`).
pub fn described_by_root_functions(cad: &Cad, free: &[Basic], cells: &[CellTruth]) -> PolyResult<Basic> {
    formula_of(&described(cad, free.len(), cells)?, free, 1)
}

fn description_of(
    formula: &Basic,
    gens: &[Basic],
    prefix: &[(Quantifier, Basic)],
    method: Option<Method>,
    partial: bool,
) -> PolyResult<(Vec<Basic>, Description)> {
    for g in gens {
        if !g.is_symbol() {
            return Err(PolyError::Type(format!("expected a symbol, got {}", g)));
        }
    }
    let (cad, free, _, cells) = truth_values(formula, Some(gens), prefix, method, partial, Prune::Both)?;
    if free.is_empty() {
        let shown: Vec<String> = gens.iter().map(|g| g.to_string()).collect();
        return Err(value_error(format!(
            "the formula has no free variable among [{}]",
            shown.join(", ")
        )));
    }
    let d: Description = described(&cad, free.len(), &cells)?;
    Ok((free, d))
}

/// A cylindrical description of the set at which the formula holds
/// (`cylindrical_formula`).
pub fn cylindrical_formula(
    formula: &Basic,
    gens: &[Basic],
    prefix: &[(Quantifier, Basic)],
    method: Option<Method>,
    partial: bool,
) -> PolyResult<Basic> {
    let (free, d) = description_of(formula, gens, prefix, method, partial)?;
    formula_of(&d, &free, 1)
}

/// `_all_points`.
fn all_points(description: &Description) -> bool {
    match description {
        Description::All(b) => *b,
        Description::Runs(runs) => runs
            .iter()
            .all(|r| r.point() && r.lower.as_ref().is_some_and(is_number) && all_points(&r.inner)),
    }
}

/// The points of a description whose runs are all points (`_points`).
fn points(description: &Description, gens: &[Basic]) -> PolyResult<Vec<Vec<Basic>>> {
    let runs: &[Run] = match description {
        Description::All(_) => return Ok(vec![Vec::new()]),
        Description::Runs(r) => r,
    };
    let mut found: Vec<Vec<Basic>> = Vec::new();
    for run in runs {
        let value: Basic = run.lower.clone().expect("a point");
        for tail in points(&run.inner, &gens[1..])? {
            let mut point: Vec<Basic> = vec![value.clone()];
            for coordinate in tail {
                point.push(subs(&coordinate, &gens[0], &value)?);
            }
            found.push(point);
        }
    }
    Ok(found)
}

/// The isolated points with numbers for coordinates, and the rest
/// (`_isolated`).
fn isolated(description: &Description, gens: &[Basic]) -> PolyResult<(Vec<Vec<Basic>>, Description)> {
    let runs: &[Run] = match description {
        Description::All(b) => {
            return Ok((if *b { vec![Vec::new()] } else { Vec::new() }, Description::All(false)));
        }
        Description::Runs(r) => r,
    };
    let mut found: Vec<Vec<Basic>> = Vec::new();
    let mut rest: Vec<Run> = Vec::new();
    for run in runs {
        if run.point() && run.lower.as_ref().is_some_and(is_number) && run.inner != Description::All(true) {
            let value: Basic = run.lower.clone().expect("a point");
            if let Some(inner) = specialized(&run.inner, &gens[0], &value, true)? {
                if all_points(&inner) {
                    for tail in points(&inner, &gens[1..])? {
                        let mut point: Vec<Basic> = vec![value.clone()];
                        point.extend(tail);
                        found.push(point);
                    }
                    continue;
                }
            }
        }
        rest.push(run.clone());
    }
    Ok((
        found,
        if rest.is_empty() {
            Description::All(false)
        } else {
            Description::Runs(rest)
        },
    ))
}

fn space(n: usize) -> PolyResult<Basic> {
    product_set(&vec![reals(); n])
}

fn tuple(items: &[Basic]) -> Basic {
    Basic::new(Head::Tuple, items.to_vec())
}

/// The set of the real points at which the formula holds
/// (`cylindrical_set`).
pub fn cylindrical_set(formula: &Basic, gens: &[Basic], method: Option<Method>, partial: bool) -> PolyResult<Basic> {
    let (free, d) = description_of(formula, gens, &[], method, partial)?;
    let whole: Basic = space(free.len())?;
    if d == Description::All(true) {
        return Ok(whole);
    }
    let (found, rest) = isolated(&d, &free)?;
    let finite: Basic = if found.is_empty() {
        empty_set()
    } else {
        point_set(&found.iter().map(|p| tuple(p)).collect::<Vec<_>>())?
    };
    if rest == Description::All(false) {
        return Ok(finite);
    }
    let region: Basic = condition_set(&tuple(&free), &formula_of(&rest, &free, 1)?, &whole)?;
    if found.is_empty() {
        Ok(region)
    } else {
        union(&[finite, region])
    }
}

fn every_run_is_a_point(description: &Description) -> bool {
    match description {
        Description::All(b) => *b,
        Description::Runs(runs) => runs.iter().all(|r| r.point() && every_run_is_a_point(&r.inner)),
    }
}

/// The set of the points of a description (`_as_set`).
fn as_set(description: &Description, gens: &[Basic]) -> PolyResult<Basic> {
    let whole: Basic = if gens.len() == 1 { reals() } else { space(gens.len())? };
    let runs: &[Run] = match description {
        Description::All(b) => return Ok(if *b { whole } else { empty_set() }),
        Description::Runs(r) => r,
    };
    if every_run_is_a_point(description) {
        let found: Vec<Vec<Basic>> = points(description, gens)?;
        if gens.len() == 1 {
            return Ok(finite_set(&found.iter().map(|p| p[0].clone()).collect::<Vec<_>>()));
        }
        return point_set(&found.iter().map(|p| tuple(p)).collect::<Vec<_>>());
    }
    if gens.len() == 1 {
        let mut pieces: Vec<Basic> = Vec::new();
        for run in runs {
            if run.point() {
                pieces.push(finite_set(&[run.lower.clone().expect("a point")]));
            } else {
                pieces.push(interval(
                    &run.lower.clone().unwrap_or_else(Basic::negative_oo),
                    &run.upper.clone().unwrap_or_else(Basic::oo),
                    !run.lower_closed,
                    !run.upper_closed,
                )?);
            }
        }
        return union(&pieces);
    }
    condition_set(&tuple(gens), &formula_of(description, gens, 1)?, &whole)
}

/// The real solutions in the unknowns for every real value of the
/// parameters (`cylindrical_cases`).
pub fn cylindrical_cases(
    formula: &Basic,
    parameters: &[Basic],
    unknowns: &[Basic],
    method: Option<Method>,
    partial: bool,
) -> PolyResult<Vec<(Basic, Basic)>> {
    let mut all: Vec<Basic> = parameters.to_vec();
    all.extend(unknowns.iter().cloned());
    let (free, d) = description_of(formula, &all, &[], method, partial)?;
    if free != all {
        return Err(value_error(
            "the parameters and the unknowns are the symbols of the formula".to_string(),
        ));
    }
    #[allow(clippy::too_many_arguments)]
    fn walk(
        current: &Description,
        level: usize,
        conditions: Vec<Basic>,
        parameters: &[Basic],
        unknowns: &[Basic],
        cases: &mut Vec<(Basic, Basic)>,
    ) -> PolyResult<()> {
        if level == parameters.len() {
            let found: Basic = as_set(current, unknowns)?;
            if !matches!(found.head(), Head::EmptySet) {
                cases.push((and_(&conditions), found));
            }
            return Ok(());
        }
        let runs: &[Run] = match current {
            Description::All(true) => {
                return walk(
                    &Description::All(true),
                    parameters.len(),
                    conditions,
                    parameters,
                    unknowns,
                    cases,
                )
            }
            Description::All(false) => return Ok(()),
            Description::Runs(r) => r,
        };
        for run in runs {
            let only = Description::Runs(vec![Run {
                lower: run.lower.clone(),
                lower_closed: run.lower_closed,
                upper: run.upper.clone(),
                upper_closed: run.upper_closed,
                inner: Description::All(true),
            }]);
            let mut next: Vec<Basic> = conditions.clone();
            next.push(formula_of(&only, parameters, level + 1)?);
            walk(&run.inner, level + 1, next, parameters, unknowns, cases)?;
        }
        Ok(())
    }
    let mut cases: Vec<(Basic, Basic)> = Vec::new();
    walk(&d, 0, Vec::new(), parameters, unknowns, &mut cases)?;
    let mut merged: Vec<(Basic, Basic)> = Vec::new();
    for (condition, found) in cases {
        match merged.iter().position(|(_, same)| *same == found) {
            Some(k) => merged[k].0 = or_(&[merged[k].0.clone(), condition]),
            None => merged.push((condition, found)),
        }
    }
    Ok(merged)
}
