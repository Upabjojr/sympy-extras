//! Port of `sympy_extras/polys/cad/lifting.py`: the lifting phase of
//! cylindrical algebraic decomposition. Over every cell of a level the
//! projection factors of the next level are specialized at its sample
//! point; their real roots split the cylinder into sections and sectors,
//! each with a sample point, and the signs of the factors on the cells are
//! read off the roots ([`Lifting`], one stack at a time and on demand, and
//! [`cylindrical_algebraic_decomposition`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::core::basic::Basic;
use crate::error::{polys_error, value_error, PolyError, PolyResult, PolysErrorKind};
use crate::numbers::*;

use super::super::polytools::Poly;
use super::projection::{projection_sets, to_polys, Method};
use super::samplepoints::{
    rational, rational_above, rational_below, rational_between, Real, SamplePoint, Sign, Specialization,
};

/// For every input polynomial, the sign of its constant factor and the
/// positions `(level, index, exponent)` of its irreducible factors.
pub type FactorSigns = Vec<(Sign, Vec<(usize, usize, usize)>)>;

/// A cell of a cylindrical algebraic decomposition (`CADCell`).
#[derive(Debug)]
pub struct CadCell {
    /// The position: one integer per level, odd for a sector, even for a
    /// section, from 1 upwards.
    pub index: Vec<usize>,
    /// The coordinates of the sample point.
    pub point: Vec<Real>,
    sample: Mutex<Option<SamplePoint>>,
    pub parent: Option<Arc<CadCell>>,
    /// The signs of the projection factors of this level at the point.
    factor_signs: Vec<Sign>,
    /// The signs of the input polynomials (filled for the top level).
    pub signs: Mutex<Vec<Sign>>,
}

impl CadCell {
    fn new(
        index: Vec<usize>,
        point: Vec<Real>,
        sample: Option<SamplePoint>,
        parent: Option<Arc<CadCell>>,
        factor_signs: Vec<Sign>,
    ) -> CadCell {
        CadCell {
            index,
            point,
            sample: Mutex::new(sample),
            parent,
            factor_signs,
            signs: Mutex::new(Vec::new()),
        }
    }

    /// The sample point with its coordinates in a common field, computed
    /// when first asked for.
    pub fn sample(&self) -> PolyResult<SamplePoint> {
        if let Some(s) = self.sample.lock().expect("an unpoisoned sample").as_ref() {
            return Ok(s.clone());
        }
        let computed: SamplePoint = match &self.parent {
            None => SamplePoint::origin(),
            Some(parent) => parent.sample()?.extend(self.point.last().expect("a coordinate"))?,
        };
        *self.sample.lock().expect("an unpoisoned sample") = Some(computed.clone());
        Ok(computed)
    }

    pub fn level(&self) -> usize {
        self.index.len()
    }

    /// The number of sectors in the index.
    pub fn dimension(&self) -> usize {
        self.index.iter().filter(|i| *i % 2 == 1).count()
    }

    pub fn is_section(&self) -> bool {
        self.index.last().is_some_and(|i| i % 2 == 0)
    }

    /// `repr(cell)`: `CADCell((1, 2), (Algebraic(...), ...))`.
    pub fn repr(&self) -> PolyResult<String> {
        let index: Vec<String> = self.index.iter().map(|i| i.to_string()).collect();
        let point: Vec<String> = self.point.iter().map(|r| r.repr()).collect::<PolyResult<_>>()?;
        Ok(format!("CADCell({}, {})", python_tuple(&index), python_tuple(&point)))
    }

    /// The signs of the projection factors of this level at the point
    /// (`_signs`).
    pub fn own_signs(&self) -> &[Sign] {
        &self.factor_signs
    }

    /// The sign of the `i`-th projection factor of `level` at the point.
    pub fn factor_sign(&self, level: usize, i: usize) -> PolyResult<Sign> {
        let mut cell: &CadCell = self;
        while cell.level() > level {
            match &cell.parent {
                None => {
                    return Err(value_error(format!(
                        "no cell of level {} below {}",
                        level,
                        self.repr()?
                    )))
                }
                Some(p) => cell = p,
            }
        }
        Ok(cell.factor_signs[i])
    }
}

/// Python's printing of a tuple: `()`, `(a,)`, `(a, b)`.
fn python_tuple(items: &[String]) -> String {
    match items.len() {
        0 => "()".to_string(),
        1 => format!("({},)", items[0]),
        _ => format!("({})", items.join(", ")),
    }
}

/// A cylindrical algebraic decomposition (`CAD`).
#[derive(Debug)]
pub struct Cad {
    pub gens: Vec<Basic>,
    pub polys: Vec<Poly>,
    pub projection: Vec<Vec<Poly>>,
    pub method: Method,
    levels: Vec<Vec<Arc<CadCell>>>,
}

impl Cad {
    /// The decomposition of its parts (`CAD(gens, polys, projection,
    /// method, levels)`).
    pub fn from_parts(
        gens: Vec<Basic>,
        polys: Vec<Poly>,
        projection: Vec<Vec<Poly>>,
        method: Method,
        levels: Vec<Vec<Arc<CadCell>>>,
    ) -> Cad {
        Cad {
            gens,
            polys,
            projection,
            method,
            levels,
        }
    }

    /// The cells of `R^n`, in lexicographic order of their indices.
    pub fn cells(&self) -> &[Arc<CadCell>] {
        self.levels.last().map(|l| l.as_slice()).unwrap_or(&[])
    }

    /// The cells of `R^level`.
    pub fn cells_at(&self, level: usize) -> PolyResult<&[Arc<CadCell>]> {
        if level < 1 || level > self.gens.len() {
            return Err(value_error(format!("level must be between 1 and {}", self.gens.len())));
        }
        Ok(&self.levels[level - 1])
    }

    pub fn levels(&self) -> &[Vec<Arc<CadCell>>] {
        &self.levels
    }

    /// The cells of the next level over `cell`.
    pub fn children(&self, cell: &CadCell) -> Vec<Arc<CadCell>> {
        let k: usize = cell.level();
        if k >= self.levels.len() {
            return Vec::new();
        }
        self.levels[k]
            .iter()
            .filter(|c| c.index[..k] == cell.index[..])
            .cloned()
            .collect()
    }

    /// `repr(cad)`.
    pub fn repr(&self) -> String {
        let gens: Vec<String> = self.gens.iter().map(|g| g.to_string()).collect();
        format!("CAD({} cells, {})", self.cells().len(), gens.join(", "))
    }
}

fn not_well_oriented(message: String) -> PolyError {
    polys_error(PolysErrorKind::NotWellOriented, message)
}

/// The sections of a stack from the roots of its polynomials with their
/// indices: a root of several polynomials is one section (`_sections`).
fn sections(roots: &[(Real, usize)]) -> PolyResult<Vec<(Real, Vec<usize>)>> {
    let values: Vec<Real> = roots.iter().map(|(r, _)| r.clone()).collect();
    let mut out: Vec<(Real, Vec<usize>)> = Vec::new();
    for group in super::samplepoints::order_of(&values)? {
        let mut vanishing: Vec<usize> = group.iter().map(|i| roots[*i].1).collect();
        vanishing.sort_unstable();
        vanishing.dedup();
        out.push((values[group[0]].clone(), vanishing));
    }
    Ok(out)
}

/// The lifting phase, one stack at a time and on demand (`Lifting`).
#[derive(Debug)]
pub struct Lifting {
    pub projection: Vec<Vec<Poly>>,
    pub gens: Vec<Basic>,
    pub method: Method,
    pub root: Arc<CadCell>,
    stacks: RefCell<HashMap<Vec<usize>, Vec<Arc<CadCell>>>>,
    /// The number of cells built so far.
    pub lifted: RefCell<usize>,
}

impl Lifting {
    pub fn new(projection: &[Vec<Poly>], gens: &[Basic], method: Method) -> Lifting {
        Lifting {
            projection: projection.to_vec(),
            gens: gens.to_vec(),
            method,
            root: Arc::new(CadCell::new(
                Vec::new(),
                Vec::new(),
                Some(SamplePoint::origin()),
                None,
                Vec::new(),
            )),
            stacks: RefCell::new(HashMap::new()),
            lifted: RefCell::new(0),
        }
    }

    /// Whether the stack over `parent` was built.
    pub fn is_lifted(&self, parent: &CadCell) -> bool {
        self.stacks.borrow().contains_key(&parent.index)
    }

    /// The cells of the next level over `parent`, from the lowest up;
    /// `NotWellOriented` when a factor of McCallum's projection vanishes
    /// identically over a cell of positive dimension.
    pub fn stack(&self, parent: &Arc<CadCell>) -> PolyResult<Vec<Arc<CadCell>>> {
        if let Some(found) = self.stacks.borrow().get(&parent.index) {
            return Ok(found.clone());
        }
        let k: usize = parent.level() + 1;
        if k > self.gens.len() {
            return Err(value_error("the cell is of the last level".to_string()));
        }
        let polys: &[Poly] = &self.projection[k - 1];
        let level_gens: &[Basic] = &self.gens[..k];
        let sample: SamplePoint = parent.sample()?;
        let mut specializations: Vec<Specialization> = Vec::new();
        let mut roots_found: Vec<(Real, usize)> = Vec::new();
        for (i, f) in polys.iter().enumerate() {
            let specialized: Specialization = sample.specialization(&f.as_expr()?, level_gens)?;
            if specialized.degree < 0 && self.method == Method::McCallum && parent.dimension() > 0 {
                return Err(not_well_oriented(format!(
                    "{} vanishes identically on a cell of dimension {}",
                    f.as_expr()?,
                    parent.dimension()
                )));
            }
            roots_found.extend(specialized.roots.iter().map(|(r, _)| (r.clone(), i)));
            specializations.push(specialized);
        }
        let sections: Vec<(Real, Vec<usize>)> = sections(&roots_found)?;
        let roots: Vec<Real> = sections.iter().map(|(r, _)| r.clone()).collect();
        let mut values: Vec<Real> = Vec::new();
        if roots.is_empty() {
            values.push(rational(&Rat::ZERO));
        } else {
            values.push(rational(&rational_below(&roots[0])?));
            for (j, root) in roots.iter().enumerate() {
                values.push(root.clone());
                if j + 1 < roots.len() {
                    values.push(rational(&rational_between(root, &roots[j + 1])?));
                }
            }
            values.push(rational(&rational_above(roots.last().expect("a root"))?));
        }
        let mut passed: Vec<usize> = vec![0; polys.len()];
        let mut cells: Vec<Arc<CadCell>> = Vec::new();
        for (j, value) in values.into_iter().enumerate().map(|(j, v)| (j + 1, v)) {
            let vanishing: &[usize] = if j % 2 == 0 { &sections[j / 2 - 1].1 } else { &[] };
            let mut signs: Vec<Sign> = Vec::new();
            for (i, specialized) in specializations.iter().enumerate() {
                if vanishing.contains(&i) {
                    signs.push(0);
                    passed[i] += 1;
                } else {
                    signs.push(specialized.sign_before(passed[i])?);
                }
            }
            let mut index: Vec<usize> = parent.index.clone();
            index.push(j);
            let mut point: Vec<Real> = parent.point.clone();
            point.push(value);
            cells.push(Arc::new(CadCell::new(index, point, None, Some(parent.clone()), signs)));
        }
        *self.lifted.borrow_mut() += cells.len();
        self.stacks.borrow_mut().insert(parent.index.clone(), cells.clone());
        Ok(cells)
    }

    /// The cells built so far, by level, in lexicographic order.
    pub fn levels(&self) -> Vec<Vec<Arc<CadCell>>> {
        let stacks = self.stacks.borrow();
        let mut result: Vec<Vec<Arc<CadCell>>> = Vec::new();
        let mut current: Vec<Arc<CadCell>> = vec![self.root.clone()];
        for _ in &self.gens {
            let cells: Vec<Arc<CadCell>> = current
                .iter()
                .filter_map(|p| stacks.get(&p.index))
                .flat_map(|s| s.iter().cloned())
                .collect();
            result.push(cells.clone());
            current = cells;
        }
        result
    }

    /// Every stack: the full decomposition.
    pub fn lift_all(&self) -> PolyResult<Vec<Vec<Arc<CadCell>>>> {
        let mut current: Vec<Arc<CadCell>> = vec![self.root.clone()];
        for _ in &self.gens {
            let mut next: Vec<Arc<CadCell>> = Vec::new();
            for parent in &current {
                next.extend(self.stack(parent)?);
            }
            current = next;
        }
        Ok(self.levels())
    }
}

/// How many liftings [`lifting_for`] keeps.
pub const KEPT: usize = 64;

type LiftingKey = (Vec<Vec<Poly>>, Vec<Basic>, Method);

thread_local! {
    static LIFTINGS: RefCell<Vec<(LiftingKey, Rc<Lifting>)>> = const { RefCell::new(Vec::new()) };
}

/// The lifting of `projection`: the one of an earlier question with the
/// same sets, whose stacks are kept, or a new one (the last [`KEPT`]).
pub fn lifting_for(projection: &[Vec<Poly>], gens: &[Basic], method: Method) -> Rc<Lifting> {
    LIFTINGS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(position) = cache
            .iter()
            .position(|(k, _)| k.0 == projection && k.1 == gens && k.2 == method)
        {
            let entry = cache.remove(position);
            let found: Rc<Lifting> = entry.1.clone();
            cache.push(entry);
            return found;
        }
        let found: Rc<Lifting> = Rc::new(Lifting::new(projection, gens, method));
        cache.push(((projection.to_vec(), gens.to_vec(), method), found.clone()));
        while cache.len() > KEPT {
            cache.remove(0);
        }
        found
    })
}

/// For every input polynomial, the sign of its constant factor and the
/// positions of its irreducible factors in the projection sets
/// (`factor_signs`).
pub fn factor_signs(polys: &[Poly], projection: &[Vec<Poly>], gens: &[Basic]) -> PolyResult<FactorSigns> {
    let mut positions: Vec<(Poly, (usize, usize))> = Vec::new();
    for (k, level) in projection.iter().enumerate() {
        for (i, g) in level.iter().enumerate() {
            let key: Poly = Poly::new(&g.as_expr()?, gens)?;
            match positions.iter_mut().find(|(p, _)| *p == key) {
                Some(entry) => entry.1 = (k + 1, i),
                None => positions.push((key, (k + 1, i))),
            }
        }
    }
    let mut result: FactorSigns = Vec::new();
    for f in polys {
        let (coeff, factors) = f.factor_list()?;
        let mut c: Sign = if f.is_zero() {
            0
        } else if crate::core::expr::is_true(&coeff, crate::core::assumptions::Predicate::positive) {
            1
        } else {
            -1
        };
        let mut items: Vec<(usize, usize, usize)> = Vec::new();
        for (g, e) in factors {
            let mut g: Poly = g;
            if g.rep.terms.values().next_back().is_some_and(rat_is_negative) {
                g = g.neg();
                if e % 2 == 1 {
                    c = -c;
                }
            }
            let Some((_, (level, i))) = positions.iter().find(|(p, _)| *p == g) else {
                return Err(value_error(format!("KeyError: {}", g.to_str()?)));
            };
            items.push((*level, *i, e));
        }
        result.push((c, items));
    }
    Ok(result)
}

/// The cylindrical algebraic decomposition sign-invariant for `polys`
/// (`cylindrical_algebraic_decomposition`): McCallum's projection first and
/// Hong's when not well-oriented, unless `method` is given.
pub fn cylindrical_algebraic_decomposition(polys: &[Basic], gens: &[Basic], method: Option<Method>) -> PolyResult<Cad> {
    if gens.is_empty() {
        return Err(value_error("at least one generator is needed".to_string()));
    }
    let inputs: Vec<Poly> = to_polys(polys, gens)?;
    let input_exprs: Vec<Basic> = inputs.iter().map(|p| p.as_expr()).collect::<PolyResult<_>>()?;
    let methods: Vec<Method> = match method {
        None => vec![Method::McCallum, Method::Hong],
        Some(m) => vec![m],
    };
    let mut levels: Vec<Vec<Arc<CadCell>>> = Vec::new();
    let mut projection: Vec<Vec<Poly>> = Vec::new();
    let mut used: Method = methods[0];
    for (position, m) in methods.iter().enumerate() {
        used = *m;
        projection = projection_sets(&input_exprs, gens, *m, None)?;
        match Lifting::new(&projection, gens, *m).lift_all() {
            Ok(l) => {
                levels = l;
                break;
            }
            Err(PolyError::Polys(PolysErrorKind::NotWellOriented, message)) => {
                if position == methods.len() - 1 {
                    return Err(not_well_oriented(message));
                }
            }
            Err(e) => return Err(e),
        }
    }
    let signs_of_factors: FactorSigns = factor_signs(&inputs, &projection, gens)?;
    for cell in levels.last().expect("a level") {
        let mut signs: Vec<Sign> = Vec::new();
        for (c, items) in &signs_of_factors {
            let mut s: Sign = *c;
            for (level, i, e) in items {
                s *= cell.factor_sign(*level, *i)?.pow(*e as u32);
            }
            signs.push(s);
        }
        *cell.signs.lock().expect("unpoisoned signs") = signs;
    }
    Ok(Cad {
        gens: gens.to_vec(),
        polys: inputs,
        projection,
        method: used,
        levels,
    })
}
