//! Real root counting on zero-dimensional ideals over the rationals by
//! Hermite's quadratic form, ported from `sympy_extras.polys.hermite`: the
//! quotient algebra with its products of standard monomials (FGLM's scheme
//! on the border) and traces, the Tarski queries as signatures, the counts
//! of real and complex solutions, Ben-Or–Kozen–Reif sign determination and
//! the decision of conjunctions with finitely many complex solutions.
//! `hermite_matrix` returns rows of rationals (the Python's `ExprMatrix`
//! is not ported).

#![allow(clippy::needless_range_loop)]

use std::collections::{BTreeMap, HashMap};

use crate::core::basic::{Basic, Head, RelOp};
use crate::error::{PolyError, PolyResult};
use crate::numbers::Rat;

use super::domains::{RationalField, QQ};
use super::ideals::{normal_form, Ideal, PolyLike};
use super::rings::{Monomial, MonomialOrder, PolyElement};

/// A vector of rationals: coordinates in the basis of standard monomials,
/// or a linear form on the quotient algebra.
pub type Vector = Vec<Rat>;
/// A square matrix of rationals, as rows.
pub type Rows = Vec<Vec<Rat>>;
/// A sign condition on a list of polynomials.
pub type SignCondition = Vec<i8>;

/// The numbers of distinct real zeros at which a polynomial is zero,
/// positive and negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignCounts {
    pub zero: i64,
    pub positive: i64,
    pub negative: i64,
}

/// A system: an ideal over the rationals, or polynomials in variables.
pub enum System<'a> {
    Ideal(&'a Ideal<RationalField>),
    Polys(&'a [Basic], &'a [Basic]),
}

fn zero_vector(n: usize) -> Vector {
    vec![Rat::ZERO; n]
}

/// The quotient algebra of a zero-dimensional ideal over the rationals
/// (`_Algebra`).
pub struct Algebra {
    pub ideal: Ideal<RationalField>,
    pub empty: bool,
    pub monomials: Vec<Monomial>,
    index: HashMap<Monomial, usize>,
    products: Vec<Vec<Vector>>,
    pub trace: Vector,
}

impl Algebra {
    pub fn new(ideal: Ideal<RationalField>) -> PolyResult<Algebra> {
        let empty: bool = ideal.is_whole_ring()?;
        let mut algebra: Algebra = Algebra {
            ideal,
            empty,
            monomials: Vec::new(),
            index: HashMap::new(),
            products: Vec::new(),
            trace: Vec::new(),
        };
        if empty {
            return Ok(algebra);
        }
        if !algebra.ideal.is_zero_dimensional()? {
            return Err(PolyError::NotImplemented(
                "the ideal is not zero-dimensional".to_string(),
            ));
        }
        algebra.monomials = algebra.ideal.standard_exponents(None)?;
        algebra.index = algebra
            .monomials
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), i))
            .collect();
        let d: usize = algebra.monomials.len();
        let border: HashMap<(usize, usize), Vector> = algebra.border()?;
        let mut rows: Vec<Vec<Vector>> = vec![vec![Vec::new(); d]; d];
        let mut by_degree: Vec<usize> = (0..d).collect();
        by_degree.sort_by_key(|i| algebra.monomials[*i].iter().sum::<u32>());
        for i in by_degree {
            let m: Monomial = algebra.monomials[i].clone();
            if m.iter().all(|e| *e == 0) {
                for j in 0..d {
                    rows[i][j] = algebra.unit(j);
                }
                continue;
            }
            let k: usize = m.iter().position(|e| *e > 0).expect("a variable");
            let mut lower: Monomial = m.clone();
            lower[k] -= 1;
            let smaller: usize = algebra.index[&lower];
            for j in 0..d {
                let mut v: Vector = zero_vector(d);
                for (c, a) in rows[smaller][j].iter().enumerate() {
                    if !a.is_zero() {
                        for (l, b) in border[&(k, c)].iter().enumerate() {
                            if !b.is_zero() {
                                v[l] += a * b;
                            }
                        }
                    }
                }
                rows[i][j] = v;
            }
        }
        algebra.trace = (0..d)
            .map(|k| (0..d).fold(Rat::ZERO, |s, l| s + &rows[k][l][l]))
            .collect();
        algebra.products = rows;
        Ok(algebra)
    }

    /// The dimension of the quotient algebra.
    pub fn dimension(&self) -> usize {
        self.monomials.len()
    }

    fn unit(&self, j: usize) -> Vector {
        let mut v: Vector = zero_vector(self.monomials.len());
        v[j] = Rat::ONE;
        v
    }

    /// The normal forms of `x_k b_c` for every variable and standard
    /// monomial (`_border`).
    fn border(&self) -> PolyResult<HashMap<(usize, usize), Vector>> {
        let n: usize = self.ideal.symbols.len();
        let d: usize = self.monomials.len();
        let basis: Vec<PolyElement<RationalField>> = self.ideal.groebner_basis(None)?;
        let leading: HashMap<Monomial, &PolyElement<RationalField>> = basis.iter().map(|g| (g.LM(), g)).collect();
        let mut forms: HashMap<Monomial, Vector> = HashMap::new();
        let mut pending: Vec<Monomial> = Vec::new();
        let shifted = |b: &Monomial, k: usize| -> Monomial {
            let mut m: Monomial = b.clone();
            m[k] += 1;
            m
        };
        for k in 0..n {
            for b in &self.monomials {
                let m: Monomial = shifted(b, k);
                if !self.index.contains_key(&m) && !pending.contains(&m) {
                    pending.push(m);
                }
            }
        }
        let order: MonomialOrder = self.ideal.ring(None)?.order.clone();
        pending.sort_by(|a, b| order.cmp(a, b));
        for m in pending {
            if let Some(g) = leading.get(&m) {
                let lc: Rat = g.LC();
                let mut v: Vector = zero_vector(d);
                for (mm, coefficient) in &g.terms {
                    if *mm != m {
                        v[self.index[mm]] = -(coefficient / &lc);
                    }
                }
                forms.insert(m, v);
                continue;
            }
            let mut position: usize = 0;
            let mut smaller: Monomial = m.clone();
            for (l, e) in m.iter().enumerate() {
                if *e == 0 {
                    continue;
                }
                position = l;
                smaller = m.clone();
                smaller[l] -= 1;
                if !self.index.contains_key(&smaller) {
                    break;
                }
            }
            let mut v: Vector = zero_vector(d);
            for (c, a) in forms[&smaller].clone().iter().enumerate() {
                if !a.is_zero() {
                    let product: Monomial = shifted(&self.monomials[c], position);
                    let column: Vector = match self.index.get(&product) {
                        Some(i) => self.unit(*i),
                        None => forms[&product].clone(),
                    };
                    for (i, w) in column.iter().enumerate() {
                        if !w.is_zero() {
                            v[i] += a * w;
                        }
                    }
                }
            }
            forms.insert(m, v);
        }
        let mut border: HashMap<(usize, usize), Vector> = HashMap::new();
        for k in 0..n {
            for (c, b) in self.monomials.iter().enumerate() {
                let m: Monomial = shifted(b, k);
                border.insert(
                    (k, c),
                    match self.index.get(&m) {
                        Some(i) => self.unit(*i),
                        None => forms[&m].clone(),
                    },
                );
            }
        }
        Ok(border)
    }

    /// The coordinates of the normal form of `f`.
    pub fn coordinates(&self, f: &PolyElement<RationalField>) -> PolyResult<Vector> {
        let r: PolyElement<RationalField> = normal_form(f, &self.ideal.groebner_basis(None)?)?;
        let mut v: Vector = zero_vector(self.monomials.len());
        for (m, c) in &r.terms {
            v[self.index[m]] = c.clone();
        }
        Ok(v)
    }

    /// `g` as an element of the ring of the ideal.
    pub fn element(&self, g: &PolyLike<RationalField>) -> PolyResult<PolyElement<RationalField>> {
        self.ideal.poly(g, None)
    }

    /// The transpose of the matrix of multiplication by `g`.
    pub fn transposed_multiplication(&self, g: &PolyElement<RationalField>) -> PolyResult<Rows> {
        let d: usize = self.monomials.len();
        let coordinates: Vector = self.coordinates(g)?;
        let mut rows: Rows = Vec::new();
        for j in 0..d {
            let mut v: Vector = zero_vector(d);
            for (c, a) in coordinates.iter().enumerate() {
                if !a.is_zero() {
                    for (l, b) in self.products[c][j].iter().enumerate() {
                        if !b.is_zero() {
                            v[l] += a * b;
                        }
                    }
                }
            }
            rows.push(v);
        }
        Ok(rows)
    }

    /// The linear form `h -> form(g h)` for the transposed multiplication
    /// matrix of `g`.
    pub fn compose(&self, form: &[Rat], transposed: &Rows) -> Vector {
        transposed
            .iter()
            .map(|row| row.iter().zip(form).fold(Rat::ZERO, |s, (c, w)| s + c * w))
            .collect()
    }

    /// The matrix of `(p, q) -> form(p q)` in the standard monomials.
    pub fn hermite(&self, form: &[Rat]) -> Rows {
        let d: usize = self.monomials.len();
        let mut h: Rows = vec![zero_vector(d); d];
        for i in 0..d {
            for j in i..d {
                let v: Rat = self.products[i][j]
                    .iter()
                    .zip(form)
                    .fold(Rat::ZERO, |s, (c, w)| s + c * w);
                h[i][j] = v.clone();
                h[j][i] = v;
            }
        }
        h
    }

    /// The linear form `h -> Tr(g h)` (`g = 1` by default).
    pub fn form(&self, g: Option<&PolyLike<RationalField>>) -> PolyResult<Vector> {
        match g {
            None => Ok(self.trace.clone()),
            Some(g) => Ok(self.compose(&self.trace, &self.transposed_multiplication(&self.element(g)?)?)),
        }
    }
}

/// The numbers of positive and negative squares of a symmetric rational
/// matrix (Lagrange's reduction, `_inertia`).
pub fn inertia(matrix: &Rows) -> (i64, i64) {
    let mut a: Rows = matrix.clone();
    let mut active: Vec<usize> = (0..a.len()).collect();
    let (mut positive, mut negative) = (0i64, 0i64);
    while !active.is_empty() {
        let mut pivot: Option<usize> = active.iter().copied().find(|i| !a[*i][*i].is_zero());
        if pivot.is_none() {
            let mut pair: Option<(usize, usize)> = None;
            'search: for &i in &active {
                for &j in &active {
                    if i < j && !a[i][j].is_zero() {
                        pair = Some((i, j));
                        break 'search;
                    }
                }
            }
            let Some((i, j)) = pair else { break };
            for &k in &active {
                let v: Rat = &a[i][k] + &a[j][k];
                a[i][k] = v;
            }
            for &k in &active {
                let v: Rat = &a[k][i] + &a[k][j];
                a[k][i] = v;
            }
            pivot = Some(i);
        }
        let p: usize = pivot.expect("a pivot");
        let d: Rat = a[p][p].clone();
        if d > Rat::ZERO {
            positive += 1;
        } else {
            negative += 1;
        }
        active.retain(|i| *i != p);
        for &i in &active {
            let f: Rat = &a[i][p] / &d;
            if !f.is_zero() {
                for &j in &active {
                    let v: Rat = &a[i][j] - &f * &a[p][j];
                    a[i][j] = v;
                }
            }
        }
    }
    (positive, negative)
}

/// The quotient algebra of a system (`_algebra`).
pub fn algebra(system: &System) -> PolyResult<Algebra> {
    let ideal: Ideal<RationalField> = match system {
        System::Ideal(i) => (*i).clone(),
        System::Polys(polys, gens) => Ideal::new(polys, gens, QQ, MonomialOrder::Grevlex)?,
    };
    Algebra::new(ideal)
}

/// The matrix of Hermite's quadratic form `(p, q) -> Tr(g p q)` in the
/// standard monomials, as rows (`hermite_matrix`).
pub fn hermite_matrix(system: &System, g: Option<&PolyLike<RationalField>>) -> PolyResult<Rows> {
    let a: Algebra = algebra(system)?;
    Ok(a.hermite(&a.form(g)?))
}

/// The Tarski query of `g` on the real zeros (`tarski_query`).
pub fn tarski_query(system: &System, g: Option<&PolyLike<RationalField>>) -> PolyResult<i64> {
    let a: Algebra = algebra(system)?;
    if a.empty {
        return Ok(0);
    }
    let (positive, negative) = inertia(&a.hermite(&a.form(g)?));
    Ok(positive - negative)
}

/// The number of distinct real solutions (`count_real_solutions`).
pub fn count_real_solutions(system: &System) -> PolyResult<i64> {
    tarski_query(system, None)
}

/// The number of distinct complex solutions (`count_complex_solutions`).
pub fn count_complex_solutions(system: &System) -> PolyResult<i64> {
    let a: Algebra = algebra(system)?;
    if a.empty {
        return Ok(0);
    }
    let (positive, negative) = inertia(&a.hermite(&a.form(None)?));
    Ok(positive + negative)
}

/// The numbers of distinct real solutions at which `g` is zero, positive
/// and negative (`real_sign_counts`).
pub fn real_sign_counts(system: &System, g: &PolyLike<RationalField>) -> PolyResult<SignCounts> {
    let counts: BTreeMap<SignCondition, i64> = sign_determination(system, std::slice::from_ref(g))?;
    let get = |s: i8| counts.get(&vec![s]).copied().unwrap_or(0);
    Ok(SignCounts {
        zero: get(0),
        positive: get(1),
        negative: get(-1),
    })
}

fn solve(matrix: &Rows, rhs: &[Rat]) -> Vector {
    let n: usize = matrix.len();
    let mut a: Rows = matrix
        .iter()
        .zip(rhs)
        .map(|(row, b)| row.iter().cloned().chain(std::iter::once(b.clone())).collect())
        .collect();
    for col in 0..n {
        let pivot: usize = (col..n).find(|r| !a[*r][col].is_zero()).expect("an invertible system");
        a.swap(col, pivot);
        let p: Rat = a[col][col].clone();
        for r in 0..n {
            if r != col && !a[r][col].is_zero() {
                let f: Rat = &a[r][col] / &p;
                let row: Vector = a[r].iter().zip(&a[col]).map(|(x, y)| x - &f * y).collect();
                a[r] = row;
            }
        }
    }
    (0..n).map(|i| &a[i][n] / &a[i][i]).collect()
}

fn independent_rows(matrix: &Rows) -> Vec<usize> {
    let mut reduced: Vec<(usize, Vector)> = Vec::new();
    let mut chosen: Vec<usize> = Vec::new();
    for (k, row) in matrix.iter().enumerate() {
        let mut v: Vector = row.clone();
        for (col, r) in &reduced {
            if !v[*col].is_zero() {
                let f: Rat = &v[*col] / &r[*col];
                v = v.iter().zip(r).map(|(x, y)| x - &f * y).collect();
            }
        }
        if let Some(lead) = v.iter().position(|x| !x.is_zero()) {
            reduced.push((lead, v));
            chosen.push(k);
        }
    }
    chosen
}

fn matrix_entry(a: &[u8], s: &[i8]) -> i64 {
    let mut value: i64 = 1;
    for (e, sign) in a.iter().zip(s) {
        for _ in 0..*e {
            value *= *sign as i64;
        }
    }
    value
}

/// The realizable sign conditions of the polynomials on the real solutions
/// with their numbers of solutions (`sign_determination`).
pub fn sign_determination(
    system: &System,
    polys: &[PolyLike<RationalField>],
) -> PolyResult<BTreeMap<SignCondition, i64>> {
    let a: Algebra = algebra(system)?;
    if a.empty {
        return Ok(BTreeMap::new());
    }
    let (positive, negative) = inertia(&a.hermite(&a.form(None)?));
    let total: i64 = positive - negative;
    if total == 0 {
        return Ok(BTreeMap::new());
    }
    let mut signs: Vec<SignCondition> = vec![Vec::new()];
    let mut exponents: Vec<Vec<u8>> = vec![Vec::new()];
    let mut counts: Vector = vec![Rat::from(crate::numbers::Int::from(total))];
    let mut forms: HashMap<Vec<u8>, Vector> = HashMap::from([(Vec::new(), a.form(None)?)]);
    for p in polys {
        let transposed: Rows = a.transposed_multiplication(&a.element(p)?)?;
        let candidates: Vec<SignCondition> = signs
            .iter()
            .flat_map(|s| {
                [0i8, 1, -1]
                    .into_iter()
                    .map(move |e| s.iter().copied().chain([e]).collect())
            })
            .collect();
        let rows: Vec<Vec<u8>> = exponents
            .iter()
            .flat_map(|x| {
                [0u8, 1, 2]
                    .into_iter()
                    .map(move |k| x.iter().copied().chain([k]).collect())
            })
            .collect();
        let mut queries: Vector = Vec::new();
        for row in &rows {
            let known: Vector = match forms.get(row) {
                Some(f) => f.clone(),
                None => {
                    let last: u8 = row[row.len() - 1];
                    let mut previous: Vec<u8> = row[..row.len() - 1].to_vec();
                    let f: Vector = if last > 0 {
                        previous.push(last - 1);
                        a.compose(&forms[&previous], &transposed)
                    } else {
                        forms[&previous].clone()
                    };
                    forms.insert(row.clone(), f.clone());
                    f
                }
            };
            let (positive, negative) = inertia(&a.hermite(&known));
            queries.push(Rat::from(crate::numbers::Int::from(positive - negative)));
        }
        let matrix: Rows = rows
            .iter()
            .map(|x| {
                candidates
                    .iter()
                    .map(|s| Rat::from(crate::numbers::Int::from(matrix_entry(x, s))))
                    .collect()
            })
            .collect();
        let solution: Vector = solve(&matrix, &queries);
        let kept: Vec<usize> = (0..solution.len()).filter(|k| !solution[*k].is_zero()).collect();
        signs = kept.iter().map(|k| candidates[*k].clone()).collect();
        counts = kept.iter().map(|k| solution[*k].clone()).collect();
        let restricted: Rows = matrix
            .iter()
            .map(|row| kept.iter().map(|k| row[*k].clone()).collect())
            .collect();
        exponents = independent_rows(&restricted)
            .into_iter()
            .map(|k| rows[k].clone())
            .collect();
        forms.retain(|key, _| exponents.contains(key));
    }
    Ok(signs
        .into_iter()
        .zip(counts)
        .map(|(s, c)| (s, i64::try_from(c.numerator()).expect("a count")))
        .collect())
}

/// Whether a conjunction of polynomial relations with finitely many
/// complex solutions of its equations has a real solution; `None` when the
/// formula is not of that form (`decide_zero_dimensional`).
pub fn decide_zero_dimensional(formula: &Basic, gens: &[Basic], max_dimension: usize) -> PolyResult<Option<bool>> {
    match formula.head() {
        Head::BooleanTrue => return Ok(Some(true)),
        Head::BooleanFalse => return Ok(Some(false)),
        _ => {}
    }
    let atoms: Vec<Basic> = if matches!(formula.head(), Head::And) {
        formula.args().to_vec()
    } else {
        vec![formula.clone()]
    };
    let mut equations: Vec<Basic> = Vec::new();
    let mut sides: Vec<(Basic, Vec<i8>)> = Vec::new();
    for atom in &atoms {
        let Head::Relational(op) = atom.head() else {
            return Ok(None);
        };
        let allowed: Vec<i8> = match op {
            RelOp::Eq => vec![0],
            RelOp::Ne => vec![1, -1],
            RelOp::Gt => vec![1],
            RelOp::Ge => vec![0, 1],
            RelOp::Lt => vec![-1],
            RelOp::Le => vec![0, -1],
        };
        let (lhs, rhs) = (&atom.args()[0], &atom.args()[1]);
        let difference: Basic =
            crate::core::add::add(&[lhs.clone(), crate::core::mul::mul(&[Basic::int(-1), rhs.clone()])?])?;
        if allowed == [0] {
            equations.push(difference);
        } else {
            sides.push((difference, allowed));
        }
    }
    if gens.is_empty() {
        return Ok(None);
    }
    let ideal: Ideal<RationalField> = match Ideal::new(&equations, gens, QQ, MonomialOrder::Grevlex) {
        Ok(i) => i,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(other) => return Err(other),
    };
    let mut polys: Vec<PolyElement<RationalField>> = Vec::new();
    for (p, _) in &sides {
        match ideal.poly(&PolyLike::Expr(p), None) {
            Ok(q) => polys.push(q),
            Err(PolyError::Value(_)) => return Ok(None),
            Err(other) => return Err(other),
        }
    }
    if equations.len() < gens.len() {
        return Ok(None);
    }
    if ideal.is_whole_ring()? {
        return Ok(Some(false));
    }
    if !ideal.is_zero_dimensional()? || ideal.vector_space_dimension()? > max_dimension {
        return Ok(None);
    }
    let likes: Vec<PolyLike<RationalField>> = polys.iter().map(PolyLike::Element).collect();
    let counts: BTreeMap<SignCondition, i64> = sign_determination(&System::Ideal(&ideal), &likes)?;
    Ok(Some(counts.keys().any(|condition| {
        condition
            .iter()
            .zip(&sides)
            .all(|(s, (_, allowed))| allowed.contains(s))
    })))
}
