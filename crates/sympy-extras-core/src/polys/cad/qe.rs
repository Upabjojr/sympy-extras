//! Port of `sympy_extras/polys/cad/qe.py`: quantifier elimination over the
//! reals by cylindrical algebraic decomposition. The formula is compiled
//! into a tree evaluated on sign vectors ([`Compiled`]), its truth values
//! propagated down the cells (on a partial decomposition by default:
//! Collins and Hong's trial evaluation), and the answer written as a union
//! of intervals (one free variable) or a minimum cover of the true sign
//! vectors by prime implicants (several), the projection augmented after
//! Hong when the signs do not tell the true cells from the false ones.
//! The last resort of `quantifier_elimination`, the root functions of
//! `cad.cylindrical`, is not ported yet.

// the nodes are immutable: their cached facts do not change the order
#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::core::basic::{Basic, Head, RelOp};
use crate::core::logic::{and_, false_, or_, true_};
use crate::core::relational::{eq, ge, gt, le, lt, ne};
use crate::error::{not_ported, value_error, PolyError, PolyResult, PolysErrorKind};
use crate::polys::polytools::Poly;
use crate::sets::sets::{finite_set, interval, interval_parts, union};

use super::indexedroot::real_root;
use super::lifting::{
    cylindrical_algebraic_decomposition, factor_signs, lifting_for, Cad, CadCell, FactorSigns, Lifting,
};
use super::projection::{augmented_projection_sets, project, projection_sets, to_polys, Method};
use super::samplepoints::Sign;

/// A quantifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantifier {
    Exists,
    Forall,
}

impl Quantifier {
    pub fn from_name(name: &str) -> PolyResult<Quantifier> {
        match name {
            "exists" => Ok(Quantifier::Exists),
            "forall" => Ok(Quantifier::Forall),
            _ => Err(value_error(format!("unknown quantifier '{}'", name))),
        }
    }
}

/// The quantifier prefix: `(kind, variable)` pairs, outermost first.
pub type QuantifierPrefix = Vec<(Quantifier, Basic)>;

/// The truth value of a formula on a cell.
pub type CellTruth = (Arc<CadCell>, bool);

/// The pruning of the space of the free variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prune {
    None,
    False,
    Both,
}

impl Prune {
    pub fn from_name(name: &str) -> PolyResult<Prune> {
        match name {
            "none" => Ok(Prune::None),
            "false" => Ok(Prune::False),
            "both" => Ok(Prune::Both),
            _ => Err(value_error(format!("unknown pruning '{}'", name))),
        }
    }
}

/// The n-ary connectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connective {
    And,
    Or,
    Xor,
    Implies,
    Equivalent,
}

/// A formula compiled into a tree evaluated on sign vectors (`_Compiled`).
#[derive(Debug, Clone)]
pub enum Compiled {
    Const(bool),
    /// The sign test of the relation on the polynomial of the index.
    Atom(usize, RelOp),
    Not(Box<Compiled>),
    Connective(Connective, Vec<Compiled>),
    /// Several formulas: true when the known signs determine all of them
    /// (`_Tables`).
    Tables(Vec<Compiled>),
}

fn test(op: RelOp, s: Sign) -> bool {
    match op {
        RelOp::Lt => s < 0,
        RelOp::Gt => s > 0,
        RelOp::Le => s <= 0,
        RelOp::Ge => s >= 0,
        RelOp::Eq => s == 0,
        RelOp::Ne => s != 0,
    }
}

impl Compiled {
    /// The truth value on a sign vector (`__call__`).
    pub fn call(&self, signs: &[Sign]) -> bool {
        match self {
            Compiled::Const(v) => *v,
            Compiled::Atom(i, op) => test(*op, signs[*i]),
            Compiled::Not(a) => !a.call(signs),
            Compiled::Tables(_) => true,
            Compiled::Connective(kind, args) => {
                let values: Vec<bool> = args.iter().map(|a| a.call(signs)).collect();
                match kind {
                    Connective::And => values.iter().all(|v| *v),
                    Connective::Or => values.iter().any(|v| *v),
                    Connective::Xor => values.iter().filter(|v| **v).count() % 2 == 1,
                    Connective::Implies => !values[0] || values[1],
                    Connective::Equivalent => values.iter().all(|v| *v == values[0]),
                }
            }
        }
    }

    /// The truth value when the known signs determine it (`trial`).
    pub fn trial(&self, signs: &[Option<Sign>]) -> Option<bool> {
        match self {
            Compiled::Const(v) => Some(*v),
            Compiled::Atom(i, op) => signs[*i].map(|s| test(*op, s)),
            Compiled::Not(a) => a.trial(signs).map(|v| !v),
            Compiled::Tables(args) => {
                if args.iter().any(|a| a.trial(signs).is_none()) {
                    None
                } else {
                    Some(true)
                }
            }
            Compiled::Connective(kind, args) => {
                let values: Vec<Option<bool>> = args.iter().map(|a| a.trial(signs)).collect();
                let any_none: bool = values.iter().any(|v| v.is_none());
                match kind {
                    Connective::And => {
                        if values.contains(&Some(false)) {
                            Some(false)
                        } else if any_none {
                            None
                        } else {
                            Some(true)
                        }
                    }
                    Connective::Or => {
                        if values.contains(&Some(true)) {
                            Some(true)
                        } else if any_none {
                            None
                        } else {
                            Some(false)
                        }
                    }
                    Connective::Implies => {
                        if values[0] == Some(false) || values[1] == Some(true) {
                            Some(true)
                        } else if values[0].is_none() || values[1].is_none() {
                            None
                        } else {
                            Some(false)
                        }
                    }
                    Connective::Equivalent | Connective::Xor => {
                        if *kind == Connective::Equivalent
                            && values.contains(&Some(true))
                            && values.contains(&Some(false))
                        {
                            return Some(false);
                        }
                        if any_none {
                            return None;
                        }
                        if *kind == Connective::Xor {
                            Some(values.iter().filter(|v| **v == Some(true)).count() % 2 == 1)
                        } else {
                            Some(true)
                        }
                    }
                }
            }
        }
    }
}

/// Compile `formula`, collecting the polynomials of the atoms (`_compile`).
fn compile(formula: &Basic, gens: &[Basic], polys: &mut Vec<Poly>) -> PolyResult<Compiled> {
    Ok(match formula.head() {
        Head::BooleanTrue => Compiled::Const(true),
        Head::BooleanFalse => Compiled::Const(false),
        Head::Relational(op) => {
            let difference: Basic = crate::core::add::sub(&formula.args()[0], &formula.args()[1])?;
            let p: Poly = to_polys(&[difference], gens)?.remove(0);
            let i: usize = match polys.iter().position(|q| *q == p) {
                Some(i) => i,
                None => {
                    polys.push(p);
                    polys.len() - 1
                }
            };
            Compiled::Atom(i, *op)
        }
        Head::Not => Compiled::Not(Box::new(compile(&formula.args()[0], gens, polys)?)),
        h @ (Head::And | Head::Or | Head::Xor | Head::Implies | Head::Equivalent) => {
            let kind: Connective = match h {
                Head::And => Connective::And,
                Head::Or => Connective::Or,
                Head::Xor => Connective::Xor,
                Head::Implies => Connective::Implies,
                _ => Connective::Equivalent,
            };
            let mut args: Vec<Compiled> = Vec::new();
            for a in formula.args() {
                args.push(compile(a, gens, polys)?);
            }
            Compiled::Connective(kind, args)
        }
        _ => return Err(value_error(format!("unsupported formula {}", formula))),
    })
}

fn sorted_by_name(symbols: impl IntoIterator<Item = Basic>) -> Vec<Basic> {
    let mut out: Vec<Basic> = symbols.into_iter().collect();
    out.sort_by_key(|s| s.to_string());
    out
}

fn not_well_oriented(e: &PolyError) -> bool {
    matches!(e, PolyError::Polys(PolysErrorKind::NotWellOriented, _))
}

/// What `_truth_values` gives: the decomposition, the free variables, the
/// truth value without free variables, the truth values of the cells of
/// the space of the free variables.
pub type TruthValues = (Cad, Vec<Basic>, Option<bool>, Vec<CellTruth>);

/// `_truth_values`.
pub fn truth_values(
    formula: &Basic,
    free: Option<&[Basic]>,
    prefix: &[(Quantifier, Basic)],
    method: Option<Method>,
    partial: bool,
    prune: Prune,
) -> PolyResult<TruthValues> {
    let bound: Vec<Basic> = prefix.iter().map(|(_, v)| v.clone()).collect();
    for v in &bound {
        if !v.is_symbol() {
            return Err(PolyError::Type(format!("expected a symbol, got {}", v)));
        }
    }
    let symbols: BTreeSet<Basic> = formula.free_symbols();
    let free_list: Vec<Basic> = match free {
        None => sorted_by_name(symbols.iter().filter(|s| !bound.contains(s)).cloned()),
        Some(f) => {
            for v in f {
                if !v.is_symbol() {
                    return Err(PolyError::Type(format!("expected a symbol, got {}", v)));
                }
            }
            f.to_vec()
        }
    };
    let mut gens: Vec<Basic> = free_list.clone();
    gens.extend(bound.iter().cloned());
    let distinct: BTreeSet<Basic> = gens.iter().cloned().collect();
    if distinct.len() != gens.len() {
        return Err(value_error("a variable is both free and quantified".to_string()));
    }
    let mut extra: Vec<String> = symbols
        .iter()
        .filter(|s| !gens.contains(s))
        .map(|s| s.to_string())
        .collect();
    if !extra.is_empty() {
        extra.sort();
        return Err(value_error(format!("variables not declared: {}", extra.join(", "))));
    }
    let mut polys: Vec<Poly> = Vec::new();
    let compiled: Compiled = compile(formula, &gens, &mut polys)?;
    if partial {
        let equational: Option<Poly> = if bound.is_empty() {
            None
        } else {
            equational_constraint(formula, &gens)?
        };
        return partial_truth_values(
            &compiled,
            &polys,
            &gens,
            &free_list,
            prefix,
            method,
            prune,
            equational.as_ref(),
        );
    }
    let exprs: Vec<Basic> = polys.iter().map(|p| p.as_expr()).collect::<PolyResult<_>>()?;
    let cad: Cad = cylindrical_algebraic_decomposition(&exprs, &gens, method)?;
    let (n, k) = (gens.len(), free_list.len());
    let mut truth: HashMap<Vec<usize>, bool> = HashMap::new();
    for cell in cad.cells() {
        let signs: Vec<Sign> = cell.signs.lock().expect("unpoisoned signs").clone();
        truth.insert(cell.index.clone(), compiled.call(&signs));
    }
    for level in (k + 1..=n).rev() {
        let kind: Quantifier = prefix[level - k - 1].0;
        let mut lower: HashMap<Vec<usize>, Vec<bool>> = HashMap::new();
        for cell in cad.cells_at(level)? {
            let parent: Vec<usize> = cell.index[..level - 1].to_vec();
            lower.entry(parent).or_default().push(truth[&cell.index]);
        }
        truth = lower
            .into_iter()
            .map(|(parent, values)| {
                let v: bool = match kind {
                    Quantifier::Exists => values.iter().any(|v| *v),
                    Quantifier::Forall => values.iter().all(|v| *v),
                };
                (parent, v)
            })
            .collect();
    }
    if k == 0 {
        let value: bool = *truth.values().next().expect("one value");
        return Ok((cad, free_list, Some(value), Vec::new()));
    }
    let cells: Vec<CellTruth> = cad.cells_at(k)?.iter().map(|c| (c.clone(), truth[&c.index])).collect();
    Ok((cad, free_list, None, cells))
}

/// The signs of the input polynomials which the projection factors of the
/// levels of the cell determine (`_known_signs`).
pub fn known_signs(cell: &CadCell, signs_of_factors: &FactorSigns) -> PolyResult<Vec<Option<Sign>>> {
    let mut signs: Vec<Option<Sign>> = Vec::new();
    for (constant, items) in signs_of_factors {
        let mut sign: Option<Sign> = Some(*constant);
        for (level, i, exponent) in items {
            if *level > cell.level() {
                if sign != Some(0) {
                    sign = None;
                }
                continue;
            }
            let factor: Sign = cell.factor_sign(*level, *i)?;
            if factor == 0 {
                sign = Some(0);
            } else if let Some(s) = sign {
                sign = Some(s * factor.pow(*exponent as u32));
            }
        }
        signs.push(sign);
    }
    Ok(signs)
}

/// `_PartialEvaluation`.
struct PartialEvaluation<'a> {
    compiled: &'a Compiled,
    lifting: &'a Lifting,
    factor_signs: FactorSigns,
    free: usize,
    prefix: &'a [(Quantifier, Basic)],
    prune: Prune,
}

impl PartialEvaluation<'_> {
    fn truth(&self, cell: &Arc<CadCell>) -> PolyResult<bool> {
        if let Some(value) = self.compiled.trial(&known_signs(cell, &self.factor_signs)?) {
            return Ok(value);
        }
        let kind: Quantifier = self.prefix[cell.level() - self.free].0;
        let stack: Vec<Arc<CadCell>> = self.lifting.stack(cell)?;
        let settling: bool = kind == Quantifier::Exists;
        let order = stack.iter().step_by(2).chain(stack.iter().skip(1).step_by(2));
        for child in order {
            if self.truth(child)? == settling {
                return Ok(settling);
            }
        }
        Ok(!settling)
    }

    fn free_cells(&self, cell: &Arc<CadCell>, found: &mut Vec<CellTruth>) -> PolyResult<()> {
        if cell.level() == self.free {
            found.push((cell.clone(), self.truth(cell)?));
            return Ok(());
        }
        if self.prune != Prune::None && cell.level() > 0 {
            let value: Option<bool> = self.compiled.trial(&known_signs(cell, &self.factor_signs)?);
            if value == Some(false) || (value == Some(true) && self.prune == Prune::Both) {
                found.push((cell.clone(), value.expect("a value")));
                return Ok(());
            }
        }
        for child in self.lifting.stack(cell)? {
            self.free_cells(&child, found)?;
        }
        Ok(())
    }
}

/// A polynomial which the formula implies to vanish, of positive degree in
/// the last variable (`_equational_constraint`).
fn equational_constraint(formula: &Basic, gens: &[Basic]) -> PolyResult<Option<Poly>> {
    let atoms: Vec<Basic> = match formula.head() {
        Head::Relational(RelOp::Eq) => vec![formula.clone()],
        Head::And => formula.args().to_vec(),
        _ => Vec::new(),
    };
    for atom in atoms {
        if matches!(atom.head(), Head::Relational(RelOp::Eq)) {
            let difference: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
            let p: Poly = to_polys(&[difference], gens)?.remove(0);
            if p.degree(gens.len() - 1)? > 0 {
                return Ok(Some(p));
            }
        }
    }
    Ok(None)
}

/// `_partial_truth_values`.
#[allow(clippy::too_many_arguments)]
fn partial_truth_values(
    compiled: &Compiled,
    polys: &[Poly],
    gens: &[Basic],
    free: &[Basic],
    prefix: &[(Quantifier, Basic)],
    method: Option<Method>,
    prune: Prune,
    equational: Option<&Poly>,
) -> PolyResult<TruthValues> {
    let methods: Vec<Method> = match method {
        None => vec![Method::McCallum, Method::Hong],
        Some(m) => vec![m],
    };
    if gens.is_empty() {
        return Err(value_error("at least one generator is needed".to_string()));
    }
    let exprs: Vec<Basic> = polys.iter().map(|p| p.as_expr()).collect::<PolyResult<_>>()?;
    let polys: Vec<Poly> = to_polys(&exprs, gens)?;
    let equational_expr: Option<Basic> = equational.map(|p| p.as_expr()).transpose()?;
    for (position, m) in methods.iter().enumerate() {
        let projection: Vec<Vec<Poly>> = projection_sets(&exprs, gens, *m, equational_expr.as_ref())?;
        let lifting = lifting_for(&projection, gens, *m);
        let evaluation = PartialEvaluation {
            compiled,
            lifting: &lifting,
            factor_signs: factor_signs(&polys, &projection, gens)?,
            free: free.len(),
            prefix,
            prune,
        };
        let mut cells: Vec<CellTruth> = Vec::new();
        let mut value: Option<bool> = None;
        let outcome: PolyResult<()> = if free.is_empty() {
            evaluation.truth(&lifting.root).map(|v| value = Some(v))
        } else {
            evaluation.free_cells(&lifting.root, &mut cells)
        };
        match outcome {
            Ok(()) => {}
            Err(e) if not_well_oriented(&e) && position + 1 < methods.len() => continue,
            Err(e) => return Err(e),
        }
        let cad: Cad = Cad::from_parts(gens.to_vec(), polys, projection, *m, lifting.levels());
        return Ok((cad, free.to_vec(), value, cells));
    }
    unreachable!("the last method returns")
}

/// The union of the true cells of the real line as a set
/// (`_interval_union`).
pub fn interval_union(cells: &[CellTruth], x: &Basic) -> PolyResult<Basic> {
    let name: String = x.to_string();
    let end = |cell: &CadCell| real_root(&cell.point[0], &name);
    let mut intervals: Vec<Basic> = Vec::new();
    let mut i: usize = 0;
    while i < cells.len() {
        if !cells[i].1 {
            i += 1;
            continue;
        }
        let mut j: usize = i;
        while j + 1 < cells.len() && cells[j + 1].1 {
            j += 1;
        }
        let (first, last) = (&cells[i].0, &cells[j].0);
        let (left, left_open): (Basic, bool) = if first.is_section() {
            (end(first)?, false)
        } else if i > 0 {
            (end(&cells[i - 1].0)?, true)
        } else {
            (Basic::negative_oo(), true)
        };
        let (right, right_open): (Basic, bool) = if last.is_section() {
            (end(last)?, false)
        } else if j + 1 < cells.len() {
            (end(&cells[j + 1].0)?, true)
        } else {
            (Basic::oo(), true)
        };
        if Arc::ptr_eq(first, last) && first.is_section() {
            intervals.push(finite_set(&[left]));
        } else {
            intervals.push(interval(&left, &right, left_open, right_open)?);
        }
        i = j + 1;
    }
    union(&intervals)
}

fn is_real_line(s: &Basic) -> bool {
    match s.head() {
        Head::Reals => true,
        Head::Interval => {
            let (a, b, _, _) = interval_parts(s);
            matches!(a.head(), Head::NegativeInfinity) && matches!(b.head(), Head::Infinity)
        }
        _ => false,
    }
}

/// The relational form of a union of intervals and points, without
/// conditions on the infinities (`_as_relational`).
fn as_relational(sets: &Basic, x: &Basic) -> PolyResult<Basic> {
    let parts: Vec<Basic> = if matches!(sets.head(), Head::Union) {
        sets.args().to_vec()
    } else {
        vec![sets.clone()]
    };
    let mut terms: Vec<Basic> = Vec::new();
    for part in parts {
        if matches!(part.head(), Head::FiniteSet) {
            for v in part.args() {
                terms.push(eq(x, v)?);
            }
            continue;
        }
        if !matches!(part.head(), Head::Interval | Head::Reals) {
            return Err(PolyError::Type(format!("unexpected set {}", part)));
        }
        let (start, end, lo, ro) = interval_parts(&part);
        let mut conditions: Vec<Basic> = Vec::new();
        if !matches!(start.head(), Head::NegativeInfinity) {
            conditions.push(if lo { gt(x, &start)? } else { ge(x, &start)? });
        }
        if !matches!(end.head(), Head::Infinity) {
            conditions.push(if ro { lt(x, &end)? } else { le(x, &end)? });
        }
        terms.push(and_(&conditions));
    }
    Ok(or_(&terms))
}

/// The signs of the projection factors of all levels at the cell
/// (`_sign_vector`).
fn sign_vector(cell: &CadCell) -> Vec<Sign> {
    let mut levels: Vec<&[Sign]> = Vec::new();
    let mut current: Option<&CadCell> = Some(cell);
    while let Some(c) = current {
        if c.level() == 0 {
            break;
        }
        levels.push(c.own_signs());
        current = c.parent.as_deref();
    }
    levels.iter().rev().flat_map(|l| l.iter().copied()).collect()
}

/// The true and the false cells of each sign vector, in the order of the
/// first appearance (`_sign_classes`).
type SignClasses = Vec<(Vec<Sign>, Vec<Arc<CadCell>>, Vec<Arc<CadCell>>)>;

fn sign_classes(cells: &[CellTruth]) -> SignClasses {
    let mut classes: SignClasses = Vec::new();
    let mut position: HashMap<Vec<Sign>, usize> = HashMap::new();
    for (cell, value) in cells {
        let vector: Vec<Sign> = sign_vector(cell);
        let k: usize = *position.entry(vector.clone()).or_insert_with(|| {
            classes.push((vector, Vec::new(), Vec::new()));
            classes.len() - 1
        });
        if *value {
            classes[k].1.push(cell.clone());
        } else {
            classes[k].2.push(cell.clone());
        }
    }
    classes
}

/// The pairs of a true and a false cell with one sign vector
/// (`_conflicts`).
fn conflicts(classes: &SignClasses) -> Vec<(Arc<CadCell>, Arc<CadCell>)> {
    let mut out: Vec<(Arc<CadCell>, Arc<CadCell>)> = Vec::new();
    for (_, trues, falses) in classes {
        for t in trues {
            for f in falses {
                out.push((t.clone(), f.clone()));
            }
        }
    }
    out
}

/// The cell and the cells below it, from level 1 up (`_ancestors`).
fn ancestors(cell: &Arc<CadCell>) -> Vec<Arc<CadCell>> {
    let mut chain: Vec<Arc<CadCell>> = Vec::new();
    let mut current: Option<Arc<CadCell>> = Some(cell.clone());
    while let Some(c) = current {
        if c.level() == 0 {
            break;
        }
        current = c.parent.clone();
        chain.push(c);
    }
    chain.reverse();
    chain
}

/// The first level at which two cells of one sign vector part, and the
/// projection factors of that level vanishing on the sections between
/// them (`_separating_factors`).
fn separating_factors(cad: &Cad, a: &Arc<CadCell>, b: &Arc<CadCell>) -> PolyResult<(usize, Vec<Poly>)> {
    let (chain_a, chain_b) = (ancestors(a), ancestors(b));
    let mut found: Option<(usize, Arc<CadCell>, Arc<CadCell>)> = None;
    for (k, (cell_a, cell_b)) in chain_a.iter().zip(chain_b.iter()).enumerate() {
        if cell_a.index != cell_b.index {
            found = Some((k + 1, cell_a.clone(), cell_b.clone()));
            break;
        }
    }
    let Some((level, cell_a, cell_b)) = found else {
        return Err(value_error("the cells are one cell".to_string()));
    };
    let stack: Vec<Arc<CadCell>> = match &cell_a.parent {
        None => cad.cells_at(1)?.to_vec(),
        Some(p) => cad.children(p),
    };
    let (x, y) = (
        *cell_a.index.last().expect("an index"),
        *cell_b.index.last().expect("an index"),
    );
    let (low, high) = (x.min(y), x.max(y));
    let mut factors: Vec<Poly> = Vec::new();
    for cell in &stack {
        let last: usize = *cell.index.last().expect("an index");
        if cell.is_section() && low <= last && last <= high {
            for (i, sign) in cell.own_signs().iter().enumerate() {
                let g: &Poly = &cad.projection[level - 1][i];
                if *sign == 0 && !factors.contains(g) {
                    factors.push(g.clone());
                }
            }
        }
    }
    Ok((level, factors))
}

/// The truth values of the cells of level `k` of a refining lifting
/// (`_inherited_truth`).
fn inherited_truth(
    lifting: &Lifting,
    counts: &[usize],
    truth: &HashMap<Vec<usize>, bool>,
    k: usize,
) -> PolyResult<Vec<CellTruth>> {
    fn walk(
        lifting: &Lifting,
        counts: &[usize],
        truth: &HashMap<Vec<usize>, bool>,
        k: usize,
        parent: &Arc<CadCell>,
        index: Vec<usize>,
        found: &mut Vec<CellTruth>,
    ) -> PolyResult<()> {
        let stack: Vec<Arc<CadCell>> = lifting.stack(parent)?;
        let count: usize = counts[parent.level()];
        let kept: Vec<usize> = (0..count)
            .filter(|i| stack.iter().any(|c| c.own_signs()[*i] != 0))
            .collect();
        let mut below: usize = 0;
        for child in &stack {
            let mut child_index: Vec<usize> = index.clone();
            if kept.iter().any(|i| child.own_signs()[*i] == 0) {
                below += 1;
                child_index.push(2 * below);
            } else {
                child_index.push(2 * below + 1);
            }
            if child.level() == k {
                let value: bool = *truth
                    .get(&child_index)
                    .ok_or_else(|| value_error(format!("KeyError: {:?}", child_index)))?;
                found.push((child.clone(), value));
            } else {
                walk(lifting, counts, truth, k, child, child_index, found)?;
            }
        }
        Ok(())
    }
    let mut found: Vec<CellTruth> = Vec::new();
    walk(lifting, counts, truth, k, &lifting.root, Vec::new(), &mut found)?;
    Ok(found)
}

/// The rounds of augmentation of the projection tried (`_ROUNDS`).
const ROUNDS: usize = 16;

fn gen_position(g: &Poly, x: &Basic) -> Option<usize> {
    g.gens.iter().position(|t| t == x)
}

/// The last of the `free` variables which `g` depends on, from 1
/// (`_free_level`).
fn free_level(g: &Poly, free: &[Basic]) -> PolyResult<usize> {
    for j in (1..=free.len()).rev() {
        if let Some(i) = gen_position(g, &free[j - 1]) {
            if g.degree(i)? > 0 {
                return Ok(j);
            }
        }
    }
    Ok(0)
}

/// A refinement of the decomposition of the space of the free variables
/// on which the signs tell the true cells from the false ones
/// (`_separating_refinement`).
fn separating_refinement(cad: &Cad, cells: &[CellTruth], k: usize) -> PolyResult<Option<(Cad, Vec<CellTruth>)>> {
    let gens: Vec<Basic> = cad.gens.clone();
    let free: Vec<Basic> = gens[..k].to_vec();
    let projection: Vec<Vec<Poly>> = cad.projection[..k].to_vec();
    let truth: HashMap<Vec<usize>, bool> = cells.iter().map(|(c, v)| (c.index.clone(), *v)).collect();
    let counts: Vec<usize> = projection.iter().map(|l| l.len()).collect();
    let mut method: Method = cad.method;
    let mut current: Option<Cad> = None;
    let mut current_cells: Vec<CellTruth> = cells.to_vec();
    let mut differentiated: Vec<Poly> = Vec::new();
    let mut extra: Vec<Poly> = Vec::new();
    let mut above_added: bool = false;
    for _ in 0..ROUNDS {
        let found: Vec<(Arc<CadCell>, Arc<CadCell>)> = conflicts(&sign_classes(&current_cells));
        if found.is_empty() {
            let result: Cad = match current {
                Some(c) => c,
                None => Cad::from_parts(
                    cad.gens.clone(),
                    cad.polys.clone(),
                    cad.projection.clone(),
                    cad.method,
                    cad.levels().to_vec(),
                ),
            };
            return Ok(Some((result, current_cells)));
        }
        let mut added: Vec<Poly> = Vec::new();
        if !above_added && k < gens.len() {
            above_added = true;
            for g in project(cad.method, &cad.projection[k], &gens[k])? {
                let g: Poly = Poly::new(&g.as_expr()?, &free)?;
                if g.is_ground() {
                    continue;
                }
                let level: usize = free_level(&g, &free)?;
                let level_set: &Vec<Poly> = if level == 0 {
                    projection.last().expect("a level")
                } else {
                    &projection[level - 1]
                };
                if !level_set.contains(&g) && !added.contains(&g) {
                    added.push(g);
                }
            }
        }
        if added.is_empty() {
            let reference: &Cad = current.as_ref().unwrap_or(cad);
            for (t, f) in &found {
                let (level, factors) = separating_factors(reference, t, f)?;
                for g in factors {
                    if !differentiated.contains(&g) {
                        differentiated.push(g.clone());
                        let i: usize = gen_position(&g, &free[level - 1])
                            .ok_or_else(|| value_error(format!("{} is not a generator", free[level - 1])))?;
                        let derivative: Poly = g.diff(i);
                        if !derivative.is_ground() {
                            added.push(derivative);
                        }
                    }
                }
            }
        }
        if added.is_empty() {
            return Ok(None);
        }
        extra.extend(added);
        let extra_exprs: Vec<Basic> = extra.iter().map(|p| p.as_expr()).collect::<PolyResult<_>>()?;
        let (refined, lifting, levels) = loop {
            let refined: Vec<Vec<Poly>> = augmented_projection_sets(&projection, &free, &extra_exprs, method)?;
            let lifting = lifting_for(&refined, &free, method);
            match lifting.lift_all() {
                Ok(levels) => break (refined, lifting, levels),
                Err(e) if not_well_oriented(&e) && method != Method::Hong => method = Method::Hong,
                Err(e) => return Err(e),
            }
        };
        current_cells = inherited_truth(&lifting, &counts, &truth, k)?;
        let mut full_projection: Vec<Vec<Poly>> = refined;
        full_projection.extend(cad.projection[k..].iter().cloned());
        let mut all_levels: Vec<Vec<Arc<CadCell>>> = levels;
        all_levels.extend(gens[k..].iter().map(|_| Vec::new()));
        current = Some(Cad::from_parts(
            gens.clone(),
            cad.polys.clone(),
            full_projection,
            method,
            all_levels,
        ));
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// the solution formula

/// The allowed signs of a factor as a bit mask: bit 0 for -1, bit 1 for 0,
/// bit 2 for 1.
fn bit(s: Sign) -> u8 {
    match s {
        -1 => 1,
        0 => 2,
        _ => 4,
    }
}

const ALL: u8 = 7;

/// An implicant: the allowed signs of every factor.
type Implicant = Vec<u8>;

const PRIME_LIMIT: usize = 4000;
const EXACT_COVER: usize = 18;

fn covers(implicant: &[u8], vector: &[Sign]) -> bool {
    implicant.iter().zip(vector).all(|(mask, s)| mask & bit(*s) != 0)
}

fn atoms(implicant: &[u8]) -> usize {
    implicant.iter().filter(|m| **m != ALL).count()
}

/// The minimal hitting sets of the hyperedges, `None` past `limit`
/// (`_minimal_transversals`).
fn minimal_transversals(edges: &[BTreeSet<usize>], limit: usize) -> Option<Vec<BTreeSet<usize>>> {
    let proper_subset = |a: &BTreeSet<usize>, b: &BTreeSet<usize>| a.len() < b.len() && a.is_subset(b);
    let mut transversals: Vec<BTreeSet<usize>> = vec![BTreeSet::new()];
    for edge in edges {
        let hitting: Vec<BTreeSet<usize>> = transversals.iter().filter(|t| !t.is_disjoint(edge)).cloned().collect();
        let mut extended: Vec<BTreeSet<usize>> = Vec::new();
        for t in &transversals {
            if !t.is_disjoint(edge) {
                continue;
            }
            for e in edge {
                let mut candidate: BTreeSet<usize> = t.clone();
                candidate.insert(*e);
                if hitting.iter().any(|h| proper_subset(h, &candidate)) || extended.contains(&candidate) {
                    continue;
                }
                extended.push(candidate);
            }
        }
        let snapshot: Vec<BTreeSet<usize>> = extended.clone();
        extended.retain(|c| !snapshot.iter().any(|d| proper_subset(d, c)));
        transversals = hitting;
        transversals.extend(extended);
        if transversals.len() > limit {
            return None;
        }
    }
    Some(transversals)
}

/// The prime implicants, `None` when too many (`_prime_implicants`).
fn prime_implicants(trues: &[Vec<Sign>], falses: &[Vec<Sign>]) -> Option<Vec<Implicant>> {
    if trues.is_empty() {
        return Some(Vec::new());
    }
    let m: usize = trues[0].len();
    let mut found: Vec<Implicant> = Vec::new();
    for v in trues {
        let mut edges: Vec<BTreeSet<usize>> = Vec::new();
        for f in falses {
            let edge: BTreeSet<usize> = (0..m)
                .filter(|i| f[*i] != v[*i])
                .map(|i| (3 * i as i64 + f[i] as i64 + 1) as usize)
                .collect();
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
        edges.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        let snapshot: Vec<BTreeSet<usize>> = edges.clone();
        edges.retain(|e| !snapshot.iter().any(|d| d.len() < e.len() && d.is_subset(e)));
        let transversals: Vec<BTreeSet<usize>> = minimal_transversals(&edges, PRIME_LIMIT)?;
        for t in transversals {
            let mut masks: Implicant = vec![ALL; m];
            for code in t {
                masks[code / 3] &= !(1u8 << (code % 3));
            }
            if !found.contains(&masks) {
                found.push(masks);
            }
        }
        if found.len() > PRIME_LIMIT {
            return None;
        }
    }
    Some(found)
}

fn popcount(x: &[bool]) -> usize {
    x.iter().filter(|b| **b).count()
}

/// A cover of the true vectors by prime implicants with the fewest
/// conditions (`_minimum_cover`).
fn minimum_cover(primes: &[Implicant], trues: &[Vec<Sign>]) -> Vec<Implicant> {
    let coverage: Vec<Vec<bool>> = primes
        .iter()
        .map(|p| trues.iter().map(|v| covers(p, v)).collect())
        .collect();
    let n: usize = trues.len();
    let union_of = |indices: &[usize]| -> Vec<bool> {
        let mut total: Vec<bool> = vec![false; n];
        for i in indices {
            for (t, c) in total.iter_mut().zip(&coverage[*i]) {
                *t |= *c;
            }
        }
        total
    };
    let full = |u: &[bool]| u.iter().all(|b| *b);
    let cost =
        |indices: &[usize]| -> (usize, usize) { (indices.iter().map(|i| atoms(&primes[*i])).sum(), indices.len()) };
    let mut order: Vec<usize> = (0..primes.len()).collect();
    order.sort_by(|a, b| {
        popcount(&coverage[*b])
            .cmp(&popcount(&coverage[*a]))
            .then(atoms(&primes[*a]).cmp(&atoms(&primes[*b])))
            .then(primes[*a].cmp(&primes[*b]))
    });
    let mut chosen: Vec<usize> = Vec::new();
    let mut covered: Vec<bool> = vec![false; n];
    while !full(&covered) {
        // max returns the first maximum
        let mut best: Option<(usize, (usize, i64))> = None;
        for i in &order {
            let new: usize = coverage[*i].iter().zip(&covered).filter(|(c, d)| **c && !**d).count();
            let key: (usize, i64) = (new, -(atoms(&primes[*i]) as i64));
            if best.as_ref().map_or(true, |(_, k)| key > *k) {
                best = Some((*i, key));
            }
        }
        let best: usize = best.expect("a prime").0;
        chosen.push(best);
        for (t, c) in covered.iter_mut().zip(&coverage[best]) {
            *t |= *c;
        }
    }
    for i in chosen.clone() {
        let rest: Vec<usize> = chosen.iter().copied().filter(|j| *j != i).collect();
        if full(&union_of(&rest)) {
            chosen = rest;
        }
    }
    let mut best_cost: (usize, usize) = cost(&chosen);
    if primes.len() <= EXACT_COVER {
        for size in 1..=chosen.len() {
            for combination in combinations(&order, size) {
                if full(&union_of(&combination)) && cost(&combination) < best_cost {
                    best_cost = cost(&combination);
                    chosen = combination;
                }
            }
        }
    }
    chosen.sort_by(|a, b| primes[*a].cmp(&primes[*b]));
    chosen.into_iter().map(|i| primes[i].clone()).collect()
}

/// The combinations of `size` items, in the order of `itertools`.
fn combinations(items: &[usize], size: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let n: usize = items.len();
    if size > n {
        return out;
    }
    let mut indices: Vec<usize> = (0..size).collect();
    loop {
        out.push(indices.iter().map(|i| items[*i]).collect());
        let Some(i) = (0..size).rev().find(|j| indices[*j] != j + n - size) else {
            return out;
        };
        indices[i] += 1;
        for j in i + 1..size {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

type Conditions = BTreeMap<usize, BTreeSet<Sign>>;

/// A cover by implicants when the primes are too many
/// (`_greedy_implicants`).
fn greedy_implicants(trues: &[Vec<Sign>], falses: &[Vec<Sign>]) -> Vec<Implicant> {
    let covers_ = |c: &Conditions, v: &[Sign]| c.iter().all(|(i, allowed)| allowed.contains(&v[*i]));
    let consistent = |c: &Conditions| !falses.iter().any(|v| covers_(c, v));
    let mut implicants: Vec<Conditions> = trues
        .iter()
        .map(|v| v.iter().enumerate().map(|(i, s)| (i, BTreeSet::from([*s]))).collect())
        .collect();
    let reduced: Vec<Conditions> = loop {
        let mut merged: bool = true;
        while merged {
            merged = false;
            'search: for a in 0..implicants.len() {
                for b in a + 1..implicants.len() {
                    let (p, q) = (&implicants[a], &implicants[b]);
                    if !p.keys().eq(q.keys()) {
                        continue;
                    }
                    let diff: Vec<usize> = p.keys().copied().filter(|i| p[i] != q[i]).collect();
                    if diff.len() == 1 {
                        let i: usize = diff[0];
                        let mut new: Conditions = p.clone();
                        let joined: BTreeSet<Sign> = p[&i].union(&q[&i]).copied().collect();
                        if joined.len() == 3 {
                            new.remove(&i);
                        } else {
                            new.insert(i, joined);
                        }
                        if consistent(&new) {
                            implicants[a] = new;
                            implicants.remove(b);
                            merged = true;
                            break 'search;
                        }
                    }
                }
            }
        }
        let mut reduced: Vec<Conditions> = Vec::new();
        for conditions in &implicants {
            let mut conditions: Conditions = conditions.clone();
            let keys: Vec<usize> = conditions.keys().rev().copied().collect();
            for i in keys {
                let mut trial: Conditions = conditions.clone();
                trial.remove(&i);
                if consistent(&trial) {
                    conditions = trial;
                }
            }
            if !reduced.contains(&conditions) {
                reduced.push(conditions);
            }
        }
        if reduced == implicants {
            break reduced;
        }
        implicants = reduced;
    };
    let covered = |c: &Conditions| -> BTreeSet<Vec<Sign>> { trues.iter().filter(|v| covers_(c, v)).cloned().collect() };
    let mut finals: Vec<Conditions> = reduced;
    let mut dropped: bool = true;
    while dropped {
        dropped = false;
        for k in 0..finals.len() {
            let mut cover: BTreeSet<Vec<Sign>> = BTreeSet::new();
            for (j, c) in finals.iter().enumerate() {
                if j != k {
                    cover.extend(covered(c));
                }
            }
            if covered(&finals[k]).is_subset(&cover) {
                finals.remove(k);
                dropped = true;
                break;
            }
        }
    }
    let m: usize = trues.first().map(|v| v.len()).unwrap_or(0);
    finals
        .iter()
        .map(|c| {
            (0..m)
                .map(|i| c.get(&i).map(|s| s.iter().map(|x| bit(*x)).sum()).unwrap_or(ALL))
                .collect()
        })
        .collect()
}

/// The relation to 0 which allows the signs of a mask (`_SIGN_RELATIONS`).
fn sign_relation(mask: u8, f: &Basic) -> PolyResult<Basic> {
    let z: Basic = Basic::zero();
    match mask {
        1 => lt(f, &z),
        2 => eq(f, &z),
        4 => gt(f, &z),
        3 => le(f, &z),
        6 => ge(f, &z),
        5 => ne(f, &z),
        _ => Err(value_error(format!("no relation for the mask {}", mask))),
    }
}

/// A formula in the projection factors of the first `k` levels holding
/// exactly on the true cells, `None` when their signs do not decide
/// (`_sign_formula`).
fn sign_formula(cad: &Cad, cells: &[CellTruth], k: usize) -> PolyResult<Option<Basic>> {
    let factors: Vec<&Poly> = cad.projection[..k].iter().flatten().collect();
    let classes: SignClasses = sign_classes(cells);
    if !conflicts(&classes).is_empty() {
        return Ok(None);
    }
    let trues: BTreeSet<Vec<Sign>> = classes
        .iter()
        .filter(|c| !c.1.is_empty())
        .map(|c| c.0.clone())
        .collect();
    let falses: BTreeSet<Vec<Sign>> = classes
        .iter()
        .filter(|c| !c.2.is_empty())
        .map(|c| c.0.clone())
        .collect();
    if trues.is_empty() {
        return Ok(Some(false_()));
    }
    if falses.is_empty() {
        return Ok(Some(true_()));
    }
    let trues: Vec<Vec<Sign>> = trues.into_iter().collect();
    let falses: Vec<Vec<Sign>> = falses.into_iter().collect();
    let cover: Vec<Implicant> = match prime_implicants(&trues, &falses) {
        None => greedy_implicants(&trues, &falses),
        Some(primes) => minimum_cover(&primes, &trues),
    };
    let mut terms: Vec<Basic> = Vec::new();
    for implicant in cover {
        let mut conditions: Vec<Basic> = Vec::new();
        for (i, mask) in implicant.iter().enumerate() {
            if *mask != ALL {
                conditions.push(sign_relation(*mask, &factors[i].as_expr()?)?);
            }
        }
        terms.push(and_(&conditions));
    }
    Ok(Some(or_(&terms)))
}

// ---------------------------------------------------------------------
// the functions

/// Truth values of several quantifier-free formulas on the cells of one
/// decomposition (`truth_tables`).
pub fn truth_tables(
    formulas: &[Basic],
    gens: &[Basic],
    method: Option<Method>,
    partial: bool,
) -> PolyResult<(Vec<Arc<CadCell>>, Vec<Vec<bool>>)> {
    for g in gens {
        if !g.is_symbol() {
            return Err(PolyError::Type(format!("expected a symbol, got {}", g)));
        }
    }
    let mut polys: Vec<Poly> = Vec::new();
    let mut compiled: Vec<Compiled> = Vec::new();
    for f in formulas {
        compiled.push(compile(f, gens, &mut polys)?);
    }
    let mut tables: Vec<Vec<bool>> = Vec::new();
    if !partial {
        let exprs: Vec<Basic> = polys.iter().map(|p| p.as_expr()).collect::<PolyResult<_>>()?;
        let cad: Cad = cylindrical_algebraic_decomposition(&exprs, gens, method)?;
        for c in &compiled {
            tables.push(
                cad.cells()
                    .iter()
                    .map(|cell| c.call(&cell.signs.lock().expect("unpoisoned signs")))
                    .collect(),
            );
        }
        return Ok((cad.cells().to_vec(), tables));
    }
    let (cad, _, _, cells) = partial_truth_values(
        &Compiled::Tables(compiled.clone()),
        &polys,
        gens,
        gens,
        &[],
        method,
        Prune::Both,
        None,
    )?;
    let signs_of_factors: FactorSigns = factor_signs(&cad.polys, &cad.projection, &cad.gens)?;
    for c in &compiled {
        let mut table: Vec<bool> = Vec::new();
        for (cell, _) in &cells {
            match c.trial(&known_signs(cell, &signs_of_factors)?) {
                Some(v) => table.push(v),
                None => return Err(value_error("a cell does not determine the formula".to_string())),
            }
        }
        tables.push(table);
    }
    Ok((cells.into_iter().map(|(c, _)| c).collect(), tables))
}

/// Eliminate the quantifiers of a formula over the reals
/// (`quantifier_elimination`).
pub fn quantifier_elimination(
    formula: &Basic,
    prefix: &[(Quantifier, Basic)],
    free: Option<&[Basic]>,
    method: Option<Method>,
    partial: bool,
) -> PolyResult<Basic> {
    let (cad, free_vars, value, cells) = truth_values(formula, free, prefix, method, partial, Prune::None)?;
    if free_vars.is_empty() {
        return Ok(crate::core::logic::boolean(value.expect("a value")));
    }
    if free_vars.len() == 1 {
        let result: Basic = interval_union(&cells, &free_vars[0])?;
        if is_real_line(&result) {
            return Ok(true_());
        }
        if matches!(result.head(), Head::EmptySet) {
            return Ok(false_());
        }
        return as_relational(&result, &free_vars[0]);
    }
    let k: usize = free_vars.len();
    if let Some(f) = sign_formula(&cad, &cells, k)? {
        return Ok(f);
    }
    if let Some((refined, refined_cells)) = separating_refinement(&cad, &cells, k)? {
        if let Some(f) = sign_formula(&refined, &refined_cells, k)? {
            return Ok(f);
        }
    }
    Err(not_ported("described_by_root_functions (polys.cad.cylindrical)"))
}

/// The truth value of a formula with all its variables quantified
/// (`decide`).
pub fn decide(
    formula: &Basic,
    prefix: &[(Quantifier, Basic)],
    method: Option<Method>,
    partial: bool,
) -> PolyResult<bool> {
    let (_, _, value, _) = truth_values(formula, Some(&[]), prefix, method, partial, Prune::None)?;
    Ok(value.unwrap_or(false))
}

/// Sample points of the cells on which a quantifier-free formula holds
/// (`sample_points`).
pub fn sample_points(
    formula: &Basic,
    gens: &[Basic],
    method: Option<Method>,
    partial: bool,
) -> PolyResult<Vec<Vec<(Basic, Basic)>>> {
    let (_, _, _, cells) = truth_values(formula, Some(gens), &[], method, partial, Prune::False)?;
    let mut out: Vec<Vec<(Basic, Basic)>> = Vec::new();
    for (cell, value) in cells {
        if value {
            let mut point: Vec<(Basic, Basic)> = Vec::new();
            for (v, c) in gens.iter().zip(&cell.point) {
                point.push((v.clone(), real_root(c, &v.to_string())?));
            }
            out.push(point);
        }
    }
    Ok(out)
}

/// The set of values of `x` for which the quantified formula holds
/// (`solution_set`).
pub fn solution_set(
    formula: &Basic,
    x: &Basic,
    prefix: &[(Quantifier, Basic)],
    method: Option<Method>,
    partial: bool,
) -> PolyResult<Basic> {
    let (_, free_vars, _, cells) = truth_values(
        formula,
        Some(std::slice::from_ref(x)),
        prefix,
        method,
        partial,
        Prune::None,
    )?;
    interval_union(&cells, &free_vars[0])
}
