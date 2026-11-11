//! Port of `sympy_extras/solvers/simplex.py`: exact linear programming over
//! the rationals, with certificates; integer programming by branch and
//! bound inside a proved box; Fourier–Motzkin elimination with the
//! redundant inequalities removed by linear programs.
//!
//! [`linear_program`] maximises `c . x` subject to `A x <= b` for
//! unrestricted rational `x` by the two-phase simplex method in exact
//! arithmetic (Bland's rule), and every answer comes with a certificate:
//! dual multipliers at an optimum, a Farkas certificate when infeasible, a
//! feasible point and a ray when unbounded. [`linear_feasible`] decides
//! systems with strict inequalities by maximising a common slack.
//! [`integer_program`] confines the search to the boxes of the proximity
//! theorem of Cook, Gerards, Schrijver and Tardos (Hadamard's bound on the
//! subdeterminants). [`eliminate_variables`] projects a system along some
//! unknowns.
//!
//! References
//! ==========
//!
//! 1. A. Schrijver, *Theory of Linear and Integer Programming*, Wiley, 1986.
//! 2. R. G. Bland, *New finite pivoting rules for the simplex method*,
//!    Mathematics of Operations Research 2 (1977), 103-107.
//! 3. W. Cook, A. M. H. Gerards, A. Schrijver, É. Tardos, *Sensitivity
//!    theorems in integer linear programming*, Mathematical Programming 34
//!    (1986), 251-264.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{arithmetic_error, value_error, PolyResult};
use crate::numbers::*;

/// The status of a linear or integer program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name: &str = match self {
            Status::Optimal => "optimal",
            Status::Infeasible => "infeasible",
            Status::Unbounded => "unbounded",
        };
        write!(f, "{}", name)
    }
}

/// The outcome of [`linear_program`] (`LinearProgramResult`).
///
/// * optimal: `value`, the maximiser `point` and the multipliers `dual`
///   (`dual >= 0`, `dual^T A = c`, `dual . b = value`);
/// * infeasible: `dual` is a Farkas certificate (`dual >= 0`,
///   `dual^T A = 0`, `dual . b < 0`);
/// * unbounded: a feasible `point` and a `ray` (`A ray <= 0`,
///   `c . ray > 0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearProgramResult {
    pub status: Status,
    pub value: Option<Rat>,
    pub point: Option<Vec<Rat>>,
    pub dual: Option<Vec<Rat>>,
    pub ray: Option<Vec<Rat>>,
}

/// A simplex tableau `rows . z = rhs` with a basis: the column `basis[k]` is
/// the `k`-th unit vector.
struct Tableau {
    rows: Vec<Vec<Rat>>,
    rhs: Vec<Rat>,
    basis: Vec<usize>,
}

impl Tableau {
    fn pivot(&mut self, k: usize, j: usize) {
        let inverse: Rat = recip(&self.rows[k][j]);
        let row: Vec<Rat> = self.rows[k].iter().map(|v| v * &inverse).collect();
        self.rows[k] = row.clone();
        self.rhs[k] = &self.rhs[k] * &inverse;
        for i in 0..self.rows.len() {
            if i != k {
                let factor: Rat = self.rows[i][j].clone();
                if !factor.is_zero() {
                    self.rows[i] = self.rows[i].iter().zip(&row).map(|(a, c)| a - &factor * c).collect();
                    let t: Rat = &factor * &self.rhs[k];
                    self.rhs[i] -= t;
                }
            }
        }
        self.basis[k] = j;
    }

    fn reduced_cost(&self, cost: &[Rat], j: usize) -> Rat {
        let mut total: Rat = Rat::ZERO;
        for k in 0..self.rows.len() {
            total += &cost[self.basis[k]] * &self.rows[k][j];
        }
        &cost[j] - total
    }

    /// Simplex iterations with Bland's rule on the columns below `allowed`:
    /// `None` at an optimum, the entering column of an unbounded direction
    /// otherwise.
    fn maximise(&mut self, cost: &[Rat], allowed: usize) -> Option<usize> {
        loop {
            let entering: usize = (0..allowed).find(|&j| self.reduced_cost(cost, j) > Rat::ZERO)?;
            let mut leaving: Option<usize> = None;
            let mut best: Rat = Rat::ZERO;
            for k in 0..self.rows.len() {
                let coefficient: &Rat = &self.rows[k][entering];
                if *coefficient > Rat::ZERO {
                    let ratio: Rat = &self.rhs[k] / coefficient;
                    let better: bool = match leaving {
                        None => true,
                        Some(l) => ratio < best || (ratio == best && self.basis[k] < self.basis[l]),
                    };
                    if better {
                        leaving = Some(k);
                        best = ratio;
                    }
                }
            }
            match leaving {
                None => return Some(entering),
                Some(l) => self.pivot(l, entering),
            }
        }
    }

    /// `c_B B**-1`: the columns `identity` were the unit vectors of the
    /// initial tableau, and hold `B**-1` now.
    fn multipliers(&self, cost: &[Rat], identity: &[usize]) -> Vec<Rat> {
        identity
            .iter()
            .map(|&column| {
                let mut total: Rat = Rat::ZERO;
                for k in 0..self.rows.len() {
                    total += &cost[self.basis[k]] * &self.rows[k][column];
                }
                total
            })
            .collect()
    }
}

fn dot(u: &[Rat], v: &[Rat]) -> Rat {
    let mut total: Rat = Rat::ZERO;
    for (a, c) in u.iter().zip(v) {
        total += a * c;
    }
    total
}

/// Maximise `c . x` subject to `A x <= b`, the unknowns rational and of
/// either sign. `ValueError` for rows of the wrong length or a wrong number
/// of right-hand sides.
pub fn linear_program(c: &[Rat], a: &[Vec<Rat>], b: &[Rat]) -> PolyResult<LinearProgramResult> {
    let n: usize = c.len();
    let m: usize = a.len();
    if b.len() != m {
        return Err(value_error("one right-hand side for each row".to_string()));
    }
    if a.iter().any(|row| row.len() != n) {
        return Err(value_error(format!("the rows must have {} coefficients", n)));
    }
    // columns: x+ (n), x- (n), slacks (m), artificials (one for each row
    // whose right-hand side is negative)
    let signs: Vec<i64> = b.iter().map(|v| if *v >= Rat::ZERO { 1 } else { -1 }).collect();
    let artificial_rows: Vec<usize> = (0..m).filter(|&i| signs[i] < 0).collect();
    let width: usize = 2 * n + m + artificial_rows.len();
    let mut rows: Vec<Vec<Rat>> = Vec::new();
    let mut basis: Vec<usize> = Vec::new();
    let mut identity: Vec<usize> = Vec::new();
    for i in 0..m {
        let sign: Rat = rat(signs[i], 1);
        let mut line: Vec<Rat> = vec![Rat::ZERO; width];
        for j in 0..n {
            line[j] = &sign * &a[i][j];
            line[n + j] = -(&sign * &a[i][j]);
        }
        line[2 * n + i] = sign.clone();
        let column: usize = if signs[i] < 0 {
            let column: usize = 2 * n + m + artificial_rows.iter().position(|&r| r == i).expect("an artificial row");
            line[column] = Rat::ONE;
            column
        } else {
            2 * n + i
        };
        rows.push(line);
        basis.push(column);
        identity.push(column);
    }
    let rhs: Vec<Rat> = (0..m).map(|i| rat(signs[i], 1) * &b[i]).collect();
    let mut tableau: Tableau = Tableau { rows, rhs, basis };
    let real_columns: usize = 2 * n + m;
    let point = |tableau: &Tableau| -> Vec<Rat> {
        let mut z: Vec<Rat> = vec![Rat::ZERO; width];
        for (k, column) in tableau.basis.iter().enumerate() {
            z[*column] = tableau.rhs[k].clone();
        }
        (0..n).map(|j| &z[j] - &z[n + j]).collect()
    };
    let original = |multipliers: Vec<Rat>| -> Vec<Rat> {
        multipliers
            .into_iter()
            .enumerate()
            .map(|(i, y)| y * rat(signs[i], 1))
            .collect()
    };
    if !artificial_rows.is_empty() {
        let mut first: Vec<Rat> = vec![Rat::ZERO; real_columns];
        first.extend(vec![-Rat::ONE; artificial_rows.len()]);
        tableau.maximise(&first, width);
        let mut infeasibility: Rat = Rat::ZERO;
        for k in 0..m {
            if tableau.basis[k] >= real_columns {
                infeasibility += &tableau.rhs[k];
            }
        }
        if infeasibility > Rat::ZERO {
            return Ok(LinearProgramResult {
                status: Status::Infeasible,
                value: None,
                point: None,
                dual: Some(original(tableau.multipliers(&first, &identity))),
                ray: None,
            });
        }
        // artificial variables left in the basis at level zero go out
        for k in 0..m {
            if tableau.basis[k] >= real_columns {
                if let Some(j) = (0..real_columns).find(|&j| !tableau.rows[k][j].is_zero()) {
                    tableau.pivot(k, j);
                }
            }
        }
    }
    let mut cost: Vec<Rat> = c.to_vec();
    cost.extend(c.iter().map(|v| -v));
    cost.extend(vec![Rat::ZERO; m + artificial_rows.len()]);
    if let Some(entering) = tableau.maximise(&cost, real_columns) {
        let mut direction: Vec<Rat> = vec![Rat::ZERO; width];
        direction[entering] = Rat::ONE;
        for (k, column) in tableau.basis.iter().enumerate() {
            direction[*column] = -&tableau.rows[k][entering];
        }
        return Ok(LinearProgramResult {
            status: Status::Unbounded,
            value: None,
            point: Some(point(&tableau)),
            dual: None,
            ray: Some((0..n).map(|j| &direction[j] - &direction[n + j]).collect()),
        });
    }
    let x: Vec<Rat> = point(&tableau);
    let value: Rat = dot(c, &x);
    Ok(LinearProgramResult {
        status: Status::Optimal,
        value: Some(value),
        point: Some(x),
        dual: Some(original(tableau.multipliers(&cost, &identity))),
        ray: None,
    })
}

/// A rational point with `A_i x <= b_i` on every row, strictly (`<`) on the
/// rows marked in `strict` (none when `None`); `None` when there is none.
pub fn linear_feasible(a: &[Vec<Rat>], b: &[Rat], strict: Option<&[bool]>) -> PolyResult<Option<Vec<Rat>>> {
    let m: usize = a.len();
    let flags: Vec<bool> = match strict {
        None => vec![false; m],
        Some(s) => s.to_vec(),
    };
    if flags.len() != m || b.len() != m {
        return Err(value_error("one right-hand side and one flag for each row".to_string()));
    }
    let n: usize = if m > 0 { a[0].len() } else { 0 };
    if !flags.iter().any(|f| *f) {
        let result: LinearProgramResult = linear_program(&vec![Rat::ZERO; n], a, b)?;
        return Ok(if result.status == Status::Optimal {
            result.point
        } else {
            None
        });
    }
    // maximise t with A_i x + t <= b_i on the strict rows and t <= 1
    let mut rows: Vec<Vec<Rat>> = a
        .iter()
        .zip(&flags)
        .map(|(row, flag)| {
            let mut r: Vec<Rat> = row.clone();
            r.push(if *flag { Rat::ONE } else { Rat::ZERO });
            r
        })
        .collect();
    let mut last: Vec<Rat> = vec![Rat::ZERO; n];
    last.push(Rat::ONE);
    rows.push(last);
    let mut bounds: Vec<Rat> = b.to_vec();
    bounds.push(Rat::ONE);
    let mut objective: Vec<Rat> = vec![Rat::ZERO; n];
    objective.push(Rat::ONE);
    let result: LinearProgramResult = linear_program(&objective, &rows, &bounds)?;
    match (result.status, result.point, result.value) {
        (Status::Optimal, Some(point), Some(value)) if value > Rat::ZERO => Ok(Some(point[..n].to_vec())),
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// integer programming

/// The outcome of [`integer_program`] (`IntegerProgramResult`): optimal
/// (with `value` and the integer `point`), infeasible (no integer point) or
/// unbounded (with an integer `point` of the polyhedron).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerProgramResult {
    pub status: Status,
    pub value: Option<Rat>,
    pub point: Option<Vec<Int>>,
}

fn floor_rat(v: &Rat) -> Int {
    div_floor(&numer(v), &denom(v))
}

fn ceil_rat(v: &Rat) -> Int {
    -floor_rat(&-v)
}

/// Each row multiplied by the common denominator of its entries.
fn integer_rows(rows: &[Vec<Rat>]) -> Vec<Vec<Int>> {
    rows.iter()
        .map(|row| {
            let mut common: Int = Int::ONE;
            for v in row {
                common = lcm(&common, &denom(v));
            }
            let scale: Rat = rat_from_int(&common);
            row.iter().map(|v| numer(&(v * &scale))).collect()
        })
        .collect()
}

/// A bound on the absolute values of the subdeterminants of an integer
/// matrix: the product of the Euclidean lengths of its nonzero rows (each
/// at least one), rounded up.
fn hadamard(rows: &[Vec<Int>]) -> Int {
    let mut bound: Int = Int::ONE;
    for row in rows {
        let mut norm: Int = Int::ZERO;
        for v in row {
            norm += v * v;
        }
        if !norm.is_zero() {
            bound *= isqrt(&(norm - Int::ONE)) + Int::ONE;
        }
    }
    bound
}

/// The best integer point of `A x <= b` in the box `low <= x <= high` for
/// the objective `c`, with its value; `None` when the box has no integer
/// point of the polyhedron.
fn branch_and_bound(
    c: &[Rat],
    a: &[Vec<Rat>],
    b: &[Rat],
    low: Vec<Int>,
    high: Vec<Int>,
) -> PolyResult<Option<(Rat, Vec<Int>)>> {
    let n: usize = c.len();
    let mut best: Option<(Rat, Vec<Int>)> = None;
    let mut pending: Vec<(Vec<Int>, Vec<Int>)> = vec![(low, high)];
    while let Some((lower, upper)) = pending.pop() {
        if lower.iter().zip(&upper).any(|(l, u)| l > u) {
            continue;
        }
        let mut rows: Vec<Vec<Rat>> = a.to_vec();
        let mut bounds: Vec<Rat> = b.to_vec();
        for j in 0..n {
            rows.push((0..n).map(|k| if k == j { Rat::ONE } else { Rat::ZERO }).collect());
            bounds.push(rat_from_int(&upper[j]));
            rows.push((0..n).map(|k| if k == j { -Rat::ONE } else { Rat::ZERO }).collect());
            bounds.push(rat_from_int(&-&lower[j]));
        }
        let relaxed: LinearProgramResult = linear_program(c, &rows, &bounds)?;
        let (Status::Optimal, Some(point), Some(value)) = (relaxed.status, relaxed.point, relaxed.value) else {
            continue;
        };
        if let Some((best_value, _)) = &best {
            if value <= *best_value {
                continue;
            }
        }
        let Some(fractional) = (0..n).find(|&j| !rat_is_integer(&point[j])) else {
            best = Some((value, point.iter().map(numer).collect()));
            continue;
        };
        let v: &Rat = &point[fractional];
        let mut down: Vec<Int> = upper.clone();
        down[fractional] = floor_rat(v);
        let mut up: Vec<Int> = lower.clone();
        up[fractional] = ceil_rat(v);
        pending.push((lower, down));
        pending.push((up, upper));
    }
    Ok(best)
}

/// Maximise `c . x` subject to `A x <= b` over the integer vectors `x` (of
/// either sign), by branch and bound on exact linear relaxations inside a
/// box which is proved to contain an optimum.
pub fn integer_program(c: &[Rat], a: &[Vec<Rat>], b: &[Rat]) -> PolyResult<IntegerProgramResult> {
    let n: usize = c.len();
    let relaxed: LinearProgramResult = linear_program(c, a, b)?;
    let infeasible = IntegerProgramResult {
        status: Status::Infeasible,
        value: None,
        point: None,
    };
    match relaxed.status {
        Status::Infeasible => Ok(infeasible),
        Status::Unbounded => {
            // an integer point, if there is one, within (n + 1) Delta([A b])
            let augmented: Vec<Vec<Rat>> = a
                .iter()
                .zip(b)
                .map(|(row, value)| {
                    let mut r: Vec<Rat> = row.clone();
                    r.push(value.clone());
                    r
                })
                .collect();
            let radius: Int = Int::from(n + 1) * hadamard(&integer_rows(&augmented));
            let found = branch_and_bound(&vec![Rat::ZERO; n], a, b, vec![-&radius; n], vec![radius; n])?;
            Ok(match found {
                None => infeasible,
                Some((_, point)) => IntegerProgramResult {
                    status: Status::Unbounded,
                    value: None,
                    point: Some(point),
                },
            })
        }
        Status::Optimal => {
            let Some(point) = relaxed.point else {
                return Err(arithmetic_error("an optimum without a point".to_string()));
            };
            // an integer optimum, if there is one, within n Delta(A) of the
            // optimum of the relaxation
            let radius: Int = Int::from(n) * hadamard(&integer_rows(a));
            let low: Vec<Int> = point.iter().map(|v| floor_rat(v) - &radius).collect();
            let high: Vec<Int> = point.iter().map(|v| ceil_rat(v) + &radius).collect();
            Ok(match branch_and_bound(c, a, b, low, high)? {
                None => infeasible,
                Some((value, point)) => IntegerProgramResult {
                    status: Status::Optimal,
                    value: Some(value),
                    point: Some(point),
                },
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Fourier-Motzkin elimination

/// The inequalities `A_i x <= b_i`, strict (`<`) on the rows marked in
/// `strict` (`LinearSystem`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearSystem {
    pub a: Vec<Vec<Rat>>,
    pub b: Vec<Rat>,
    pub strict: Vec<bool>,
}

impl LinearSystem {
    /// Whether the point satisfies every inequality.
    pub fn contains(&self, point: &[Rat]) -> bool {
        for ((row, bound), flag) in self.a.iter().zip(&self.b).zip(&self.strict) {
            let value: Rat = dot(row, point);
            if value > *bound || (*flag && value == *bound) {
                return false;
            }
        }
        true
    }

    /// Whether the system has a solution.
    pub fn is_feasible(&self) -> PolyResult<bool> {
        if self.a.is_empty() {
            return Ok(true);
        }
        Ok(linear_feasible(&self.a, &self.b, Some(&self.strict))?.is_some())
    }
}

/// An inequality: the coefficients, the bound, whether it is strict.
type Row = (Vec<Rat>, Rat, bool);

fn abs_rat(v: &Rat) -> Rat {
    if *v < Rat::ZERO {
        -v
    } else {
        v.clone()
    }
}

/// The row divided by the absolute value of its first nonzero coefficient
/// (of the bound for a row without unknowns).
fn normal_row(coefficients: &[Rat], bound: Rat, strict: bool) -> Row {
    let mut scale: Rat = coefficients
        .iter()
        .find(|v| !v.is_zero())
        .map(abs_rat)
        .unwrap_or(Rat::ZERO);
    if scale.is_zero() {
        scale = if bound.is_zero() { Rat::ONE } else { abs_rat(&bound) };
    }
    (
        coefficients.iter().map(|v| v / &scale).collect(),
        bound / &scale,
        strict,
    )
}

/// The rows without duplicates, without those implied by a row with the
/// same left-hand side, without the trivial ones, and without those which a
/// linear program proves implied by the others (in the order of the first
/// occurrences, as the Python dict keeps them).
fn without_redundant(rows: Vec<Row>) -> PolyResult<Vec<Row>> {
    let mut order: Vec<Vec<Rat>> = Vec::new();
    let mut strongest: BTreeMap<Vec<Rat>, (Rat, bool)> = BTreeMap::new();
    for (coefficients, bound, strict) in rows {
        if coefficients.iter().all(|v| v.is_zero()) {
            if bound > Rat::ZERO || (bound.is_zero() && !strict) {
                continue;
            }
            // 0 <= negative, or 0 < 0: the system is contradictory
            return Ok(vec![(coefficients, -Rat::ONE, false)]);
        }
        let replace: bool = match strongest.get(&coefficients) {
            None => {
                order.push(coefficients.clone());
                true
            }
            Some((current, current_strict)) => bound < *current || (bound == *current && strict && !*current_strict),
        };
        if replace {
            strongest.insert(coefficients, (bound, strict));
        }
    }
    let mut kept: Vec<Row> = order
        .into_iter()
        .map(|coefficients| {
            let (bound, strict) = strongest[&coefficients].clone();
            (coefficients, bound, strict)
        })
        .collect();
    let mut index: usize = 0;
    while index < kept.len() && kept.len() > 1 {
        let (coefficients, bound, strict) = kept[index].clone();
        let others: Vec<Row> = kept
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, row)| row.clone())
            .collect();
        let matrix: Vec<Vec<Rat>> = others.iter().map(|row| row.0.clone()).collect();
        let bounds: Vec<Rat> = others.iter().map(|row| row.1.clone()).collect();
        let result: LinearProgramResult = linear_program(&coefficients, &matrix, &bounds)?;
        if result.status == Status::Infeasible {
            return Ok(vec![(vec![Rat::ZERO; coefficients.len()], -Rat::ONE, false)]);
        }
        // the supremum over the closure of the others: below the bound, the
        // row says nothing new (at the bound, a strict row still does)
        let redundant: bool = match (&result.status, &result.value) {
            (Status::Optimal, Some(value)) => *value < bound || (*value == bound && !strict),
            _ => false,
        };
        if redundant {
            kept = others;
        } else {
            index += 1;
        }
    }
    Ok(kept)
}

/// The projection of `{x: A x <= b}` (strict on the rows marked in
/// `strict`) along the unknowns `variables`: a system in the other
/// unknowns, in their order, satisfied by exactly the points which extend
/// to a solution. `ValueError` for an unknown out of range or a wrong number
/// of bounds or flags.
pub fn eliminate_variables(
    a: &[Vec<Rat>],
    b: &[Rat],
    variables: &[usize],
    strict: Option<&[bool]>,
) -> PolyResult<LinearSystem> {
    let m: usize = a.len();
    let n: usize = if m > 0 { a[0].len() } else { 0 };
    let flags: Vec<bool> = match strict {
        None => vec![false; m],
        Some(s) => s.to_vec(),
    };
    if b.len() != m || flags.len() != m {
        return Err(value_error("one right-hand side and one flag for each row".to_string()));
    }
    if let Some(j) = variables.iter().find(|&&j| j >= n) {
        return Err(value_error(format!("no unknown number {}", j)));
    }
    let mut rows: Vec<Row> = a
        .iter()
        .zip(b)
        .zip(&flags)
        .map(|((row, bound), flag)| normal_row(row, bound.clone(), *flag))
        .collect();
    let mut eliminated: Vec<usize> = variables.to_vec();
    eliminated.sort();
    eliminated.dedup();
    for &j in &eliminated {
        let upper: Vec<&Row> = rows.iter().filter(|row| row.0[j] > Rat::ZERO).collect();
        let lower: Vec<&Row> = rows.iter().filter(|row| row.0[j] < Rat::ZERO).collect();
        let mut combined: Vec<Row> = rows.iter().filter(|row| row.0[j].is_zero()).cloned().collect();
        for p in &upper {
            for q in &lower {
                let wp: Rat = recip(&p.0[j]);
                let wq: Rat = -recip(&q.0[j]);
                let mut coefficients: Vec<Rat> = p.0.iter().zip(&q.0).map(|(u, v)| &wp * u + &wq * v).collect();
                coefficients[j] = Rat::ZERO;
                combined.push(normal_row(&coefficients, &wp * &p.1 + &wq * &q.1, p.2 || q.2));
            }
        }
        rows = without_redundant(combined)?;
    }
    let remaining: Vec<usize> = (0..n).filter(|j| !eliminated.contains(j)).collect();
    Ok(LinearSystem {
        a: rows
            .iter()
            .map(|row| remaining.iter().map(|&j| row.0[j].clone()).collect())
            .collect(),
        b: rows.iter().map(|row| row.1.clone()).collect(),
        strict: rows.iter().map(|row| row.2).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(v: &[i64]) -> Vec<Rat> {
        v.iter().map(|x| rat(*x, 1)).collect()
    }

    #[test]
    fn docstring_examples() {
        let a: Vec<Vec<Rat>> = vec![r(&[1, 1]), r(&[1, 0]), r(&[0, 1]), r(&[-1, 0]), r(&[0, -1])];
        let result = linear_program(&r(&[3, 2]), &a, &r(&[4, 3, 3, 0, 0])).unwrap();
        assert_eq!(result.status, Status::Optimal);
        assert_eq!(result.value, Some(rat(11, 1)));
        assert_eq!(result.point, Some(r(&[3, 1])));
        assert_eq!(result.dual, Some(r(&[2, 1, 0, 0, 0])));
        assert_eq!(
            linear_program(&r(&[1]), &[r(&[1]), r(&[-1])], &r(&[0, -1]))
                .unwrap()
                .status,
            Status::Infeasible
        );
        assert_eq!(
            linear_program(&r(&[1, 0]), &[r(&[0, 1])], &r(&[1])).unwrap().status,
            Status::Unbounded
        );
        let farkas = linear_program(&r(&[0, 0]), &[r(&[1, 1]), r(&[-1, 0]), r(&[0, -1])], &r(&[-1, 0, 0])).unwrap();
        assert_eq!(farkas.dual, Some(r(&[1, 1, 1])));
        assert_eq!(
            linear_feasible(
                &[r(&[1, 1]), r(&[-1, 0]), r(&[0, -1])],
                &r(&[1, 0, 0]),
                Some(&[true, true, true])
            )
            .unwrap(),
            Some(vec![rat(1, 3), rat(1, 3)])
        );
        assert_eq!(
            linear_feasible(&[r(&[1]), r(&[-1])], &r(&[1, -1]), Some(&[true, false])).unwrap(),
            None
        );
        let ip = integer_program(
            &r(&[5, 4]),
            &[r(&[6, 4]), r(&[1, 2]), r(&[-1, 0]), r(&[0, -1])],
            &r(&[24, 6, 0, 0]),
        )
        .unwrap();
        assert_eq!(
            (ip.status, ip.value, ip.point),
            (Status::Optimal, Some(rat(20, 1)), Some(vec![int(4), int(0)]))
        );
        assert_eq!(
            integer_program(&r(&[0, 0]), &[r(&[3, -3]), r(&[-3, 3])], &r(&[2, -1]))
                .unwrap()
                .status,
            Status::Infeasible
        );
        let triangle: Vec<Vec<Rat>> = vec![r(&[-1, 0]), r(&[0, -1]), r(&[1, 1])];
        let shadow = eliminate_variables(&triangle, &r(&[0, 0, 1]), &[1], None).unwrap();
        let mut pairs: Vec<(Vec<Rat>, Rat)> = shadow.a.iter().cloned().zip(shadow.b.iter().cloned()).collect();
        pairs.sort();
        assert_eq!(pairs, vec![(r(&[-1]), rat(0, 1)), (r(&[1]), rat(1, 1))]);
        let open = eliminate_variables(&triangle, &r(&[0, 0, 1]), &[1], Some(&[false, true, false])).unwrap();
        assert!(!open.contains(&r(&[1])) && open.contains(&[rat(1, 2)]));
        assert!(!eliminate_variables(&[r(&[1]), r(&[-1])], &r(&[0, -1]), &[0], None)
            .unwrap()
            .is_feasible()
            .unwrap());
    }
}
