//! Port of `sympy_extras/matrices/riccati.py`: exact solutions of the
//! continuous and discrete algebraic Riccati equations with rational data
//! as points of a zero-dimensional system over `QQbar`; the stabilising
//! solution decided by the Routh–Hurwitz test of the closed loop, first in
//! interval arithmetic, then exactly; LQR gains.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::domainmatrix::DomainMatrix;
use crate::matrices::linalg::square;
use crate::numbers::{Int, Rat};
use crate::polys::domains::rationalfield::{RationalField, QQ};
use crate::polys::numberfields::qqbar::Algebraic;
use crate::polys::rings::{MonomialOrder, PolyElement, PolyRing};

type Rows = Vec<Vec<Rat>>;
type P = PolyElement<RationalField>;
type AMatrix = Vec<Vec<Algebraic>>;
type Interval = (Rat, Rat);

fn q(n: i64) -> Rat {
    Rat::from(Int::from(n))
}
fn zero() -> Rat {
    q(0)
}
fn shape(m: &ExprMatrix) -> String {
    format!("({}, {})", m.rows, m.cols)
}
fn alg(r: &Rat) -> Algebraic {
    Algebraic::from_rational(r)
}

fn rational_rows(m: &ExprMatrix, name: &str) -> PolyResult<Rows> {
    let mut rows: Rows = Vec::new();
    for i in 0..m.rows {
        let mut line: Vec<Rat> = Vec::new();
        for e in m.row(i) {
            match e.as_rational() {
                Some(r) => line.push(r.clone()),
                None => {
                    return Err(value_error(format!(
                        "the entries of {} must be rational numbers: {}",
                        name, e
                    )))
                }
            }
        }
        rows.push(line);
    }
    Ok(rows)
}

fn product(a: &Rows, b: &Rows) -> Rows {
    let columns: usize = b.first().map(|r| r.len()).unwrap_or(0);
    a.iter()
        .map(|row| {
            (0..columns)
                .map(|j| {
                    row.iter()
                        .zip(b.iter())
                        .fold(zero(), |s, (p, brow)| s + p.clone() * brow[j].clone())
                })
                .collect()
        })
        .collect()
}

fn transpose(a: &Rows) -> Rows {
    let columns: usize = a.first().map(|r| r.len()).unwrap_or(0);
    (0..columns)
        .map(|j| a.iter().map(|row| row[j].clone()).collect())
        .collect()
}

fn names(pairs: &[(usize, usize)], prefix: &str) -> Vec<String> {
    pairs.iter().map(|(i, j)| format!("{}_{}_{}", prefix, i, j)).collect()
}

struct Problem {
    n: usize,
    a: Rows,
    q: Rows,
    g: Rows,
    s: Rows,
}

impl Problem {
    fn new(a: &ExprMatrix, b: &ExprMatrix, qm: &ExprMatrix, r: &ExprMatrix) -> PolyResult<Problem> {
        let n: usize = square(a)?;
        let m: usize = square(r)?;
        if b.shape() != (n, m) || qm.shape() != (n, n) {
            return Err(value_error(format!(
                "shapes: A {}, B {}, Q {}, R {}",
                shape(a),
                shape(b),
                shape(qm),
                shape(r)
            )));
        }
        let (ar, qr) = (rational_rows(a, "A")?, rational_rows(qm, "Q")?);
        let (br, rr) = (rational_rows(b, "B")?, rational_rows(r, "R")?);
        if qr != transpose(&qr) || rr != transpose(&rr) {
            return Err(value_error("Q and R must be symmetric".to_string()));
        }
        let inverse: Rows = match DomainMatrix::new(rr, QQ, Some(m))?.inv() {
            Ok(i) => i.rows,
            Err(PolyError::Value(_)) | Err(PolyError::ZeroDivision(_)) => {
                return Err(value_error("R must be invertible".to_string()))
            }
            Err(e) => return Err(e),
        };
        let g: Rows = product(&inverse, &transpose(&br));
        let s: Rows = product(&br, &g);
        Ok(Problem { n, a: ar, q: qr, g, s })
    }

    fn solutions(&self) -> PolyResult<Vec<AMatrix>> {
        let n: usize = self.n;
        if n == 0 {
            return Ok(vec![Vec::new()]);
        }
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
        let ring = PolyRing::new(QQ, names(&pairs, "x"), MonomialOrder::Lex)?;
        let gens: Vec<P> = ring.gens();
        let unknown = |i: usize, j: usize| -> P {
            let key = (i.min(j), i.max(j));
            gens[pairs.iter().position(|p| *p == key).expect("a pair")].clone()
        };
        let x: Vec<Vec<P>> = (0..n).map(|i| (0..n).map(|j| unknown(i, j)).collect()).collect();
        let left = |c: &Rows, mm: &Vec<Vec<P>>| -> Vec<Vec<P>> {
            c.iter()
                .map(|row| {
                    (0..n)
                        .map(|j| {
                            let mut total: P = ring.zero();
                            for (k, coefficient) in row.iter().enumerate() {
                                if *coefficient != zero() {
                                    total = total.add(&mm[k][j].mul_ground(coefficient));
                                }
                            }
                            total
                        })
                        .collect()
                })
                .collect()
        };
        let atx = left(&transpose(&self.a), &x);
        let sx = left(&self.s, &x);
        let mut equations: Vec<P> = Vec::new();
        for (i, j) in &pairs {
            let mut total: P = atx[*i][*j].add(&atx[*j][*i]).add(&ring.ground(self.q[*i][*j].clone()));
            for k in 0..n {
                total = total.sub(&x[*i][k].mul(&sx[k][*j]));
            }
            equations.push(total);
        }
        let mut found: Vec<AMatrix> = Vec::new();
        for point in crate::polys::zerodim::solve_zero_dimensional(&equations, &ring)? {
            if point.iter().all(|v| v.is_real()) {
                let value = |i: usize, j: usize| {
                    point[pairs.iter().position(|p| *p == (i.min(j), i.max(j))).expect("a pair")].clone()
                };
                found.push((0..n).map(|i| (0..n).map(|j| value(i, j)).collect()).collect());
            }
        }
        Ok(found)
    }

    fn closed_loop(&self, x: &AMatrix) -> PolyResult<AMatrix> {
        let n: usize = self.n;
        let mut result: AMatrix = Vec::new();
        for i in 0..n {
            let mut line: Vec<Algebraic> = Vec::new();
            for j in 0..n {
                let mut total: Algebraic = alg(&self.a[i][j]);
                for k in 0..n {
                    if self.s[i][k] != zero() {
                        total = total.sub(&x[k][j].mul(&alg(&self.s[i][k]))?)?;
                    }
                }
                line.push(total);
            }
            result.push(line);
        }
        Ok(result)
    }
}

fn characteristic(m: &AMatrix) -> PolyResult<Vec<Algebraic>> {
    let n: usize = m.len();
    let (one, zero_) = (alg(&q(1)), alg(&q(0)));
    let mut coefficients: Vec<Algebraic> = vec![one.clone()];
    let mut current: AMatrix = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { one.clone() } else { zero_.clone() })
                .collect()
        })
        .collect();
    for k in 1..=n {
        let mut prod: AMatrix = Vec::new();
        for i in 0..n {
            let mut row: Vec<Algebraic> = Vec::new();
            for j in 0..n {
                let mut total: Algebraic = zero_.clone();
                for l in 0..n {
                    total = total.add(&m[i][l].mul(&current[l][j])?)?;
                }
                row.push(total);
            }
            prod.push(row);
        }
        let mut trace: Algebraic = zero_.clone();
        for i in 0..n {
            trace = trace.add(&prod[i][i])?;
        }
        let c: Algebraic = trace.neg().div(&alg(&q(k as i64)))?;
        coefficients.push(c.clone());
        current = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            prod[i][j].add(&c)
                        } else {
                            Ok(prod[i][j].clone())
                        }
                    })
                    .collect::<PolyResult<Vec<_>>>()
            })
            .collect::<PolyResult<_>>()?;
    }
    Ok(coefficients)
}

fn hurwitz(coefficients: &[Algebraic]) -> PolyResult<bool> {
    let zero_: Algebraic = alg(&q(0));
    let degree: usize = coefficients.len() - 1;
    let mut first: Vec<Algebraic> = coefficients.iter().step_by(2).cloned().collect();
    let mut second: Vec<Algebraic> = coefficients.iter().skip(1).step_by(2).cloned().collect();
    for _ in 0..degree {
        if second.is_empty() || second[0].sign()? <= 0 {
            return Ok(false);
        }
        let ratio: Algebraic = first[0].div(&second[0])?;
        let mut following: Vec<Algebraic> = Vec::new();
        for k in 1..first.len() {
            let below: Algebraic = if k < second.len() {
                second[k].clone()
            } else {
                zero_.clone()
            };
            following.push(first[k].sub(&ratio.mul(&below)?)?);
        }
        first = std::mem::replace(&mut second, following);
    }
    Ok(true)
}

fn i_add(a: &Interval, b: &Interval) -> Interval {
    (a.0.clone() + b.0.clone(), a.1.clone() + b.1.clone())
}
fn i_sub(a: &Interval, b: &Interval) -> Interval {
    (a.0.clone() - b.1.clone(), a.1.clone() - b.0.clone())
}
fn i_mul(a: &Interval, b: &Interval) -> Interval {
    let products: [Rat; 4] = [
        a.0.clone() * b.0.clone(),
        a.0.clone() * b.1.clone(),
        a.1.clone() * b.0.clone(),
        a.1.clone() * b.1.clone(),
    ];
    let lo: Rat = products.iter().min().expect("four").clone();
    let hi: Rat = products.iter().max().expect("four").clone();
    (lo, hi)
}
fn i_scale(a: &Interval, c: &Rat) -> Interval {
    if *c >= zero() {
        (a.0.clone() * c.clone(), a.1.clone() * c.clone())
    } else {
        (a.1.clone() * c.clone(), a.0.clone() * c.clone())
    }
}

fn interval_characteristic(closed: &[Vec<Interval>]) -> Vec<Interval> {
    let n: usize = closed.len();
    let zi: Interval = (zero(), zero());
    let one: Interval = (q(1), q(1));
    let mut coefficients: Vec<Interval> = vec![one.clone()];
    let mut current: Vec<Vec<Interval>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { one.clone() } else { zi.clone() }).collect())
        .collect();
    for k in 1..=n {
        let mut prod: Vec<Vec<Interval>> = Vec::new();
        for i in 0..n {
            let mut row: Vec<Interval> = Vec::new();
            for j in 0..n {
                let mut total: Interval = zi.clone();
                for l in 0..n {
                    total = i_add(&total, &i_mul(&closed[i][l], &current[l][j]));
                }
                row.push(total);
            }
            prod.push(row);
        }
        let mut trace: Interval = zi.clone();
        for i in 0..n {
            trace = i_add(&trace, &prod[i][i]);
        }
        let c: Interval = i_scale(&trace, &(q(-1) / q(k as i64)));
        coefficients.push(c.clone());
        current = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            i_add(&prod[i][j], &c)
                        } else {
                            prod[i][j].clone()
                        }
                    })
                    .collect()
            })
            .collect();
    }
    coefficients
}

fn interval_hurwitz(given: &[Interval]) -> Option<bool> {
    let zi: Interval = (zero(), zero());
    let leading: &Interval = &given[0];
    if *leading == zi {
        return Some(false);
    }
    if leading.0 <= zero() && zero() <= leading.1 {
        return None;
    }
    let coefficients: Vec<Interval> = if leading.1 < zero() {
        given.iter().map(|c| (-c.1.clone(), -c.0.clone())).collect()
    } else {
        given.to_vec()
    };
    let mut first: Vec<Interval> = coefficients.iter().step_by(2).cloned().collect();
    let mut second: Vec<Interval> = coefficients.iter().skip(1).step_by(2).cloned().collect();
    for _ in 0..coefficients.len() - 1 {
        if second.is_empty() {
            return Some(false);
        }
        let pivot: Interval = second[0].clone();
        if pivot.1 < zero() || pivot == zi {
            return Some(false);
        }
        if pivot.0 <= zero() {
            return None;
        }
        let inverse: Interval = (q(1) / pivot.1.clone(), q(1) / pivot.0.clone());
        let ratio: Interval = i_mul(&first[0], &inverse);
        let mut following: Vec<Interval> = Vec::new();
        for k in 1..first.len() {
            let below: Interval = if k < second.len() {
                second[k].clone()
            } else {
                zi.clone()
            };
            following.push(i_sub(&first[k], &i_mul(&ratio, &below)));
        }
        first = std::mem::replace(&mut second, following);
    }
    Some(true)
}

fn moebius_weights(n: usize) -> Vec<Vec<i64>> {
    let mut rows: Vec<Vec<i64>> = Vec::new();
    for k in 0..=n {
        let mut polynomial: Vec<i64> = vec![1];
        let factors: Vec<i64> = std::iter::repeat(1)
            .take(n - k)
            .chain(std::iter::repeat(-1).take(k))
            .collect();
        for factor in factors {
            let mut a: Vec<i64> = polynomial.clone();
            a.push(0);
            let mut b: Vec<i64> = vec![0];
            b.extend(polynomial.iter());
            polynomial = a.iter().zip(b.iter()).map(|(x, y)| x + factor * y).collect();
        }
        rows.push(polynomial);
    }
    rows
}

fn interval_moebius(coefficients: &[Interval]) -> Vec<Interval> {
    let n: usize = coefficients.len() - 1;
    let weights: Vec<Vec<i64>> = moebius_weights(n);
    (0..=n)
        .map(|j| {
            let mut total: Interval = (zero(), zero());
            for k in 0..=n {
                total = i_add(&total, &i_scale(&coefficients[k], &q(weights[k][j])));
            }
            total
        })
        .collect()
}

fn box_of(a: &Algebraic, eps: &Rat) -> PolyResult<Interval> {
    let b = a.enclosure(eps)?;
    Ok((b.0, b.1))
}

fn stable_by_intervals(problem: &Problem, x: &AMatrix, eps: &Rat) -> PolyResult<Option<bool>> {
    let n: usize = problem.n;
    let boxes: Vec<Vec<Interval>> = x
        .iter()
        .map(|row| row.iter().map(|v| box_of(v, eps)).collect::<PolyResult<_>>())
        .collect::<PolyResult<_>>()?;
    let mut closed: Vec<Vec<Interval>> = Vec::new();
    for i in 0..n {
        let mut line: Vec<Interval> = Vec::new();
        for j in 0..n {
            let mut total: Interval = (problem.a[i][j].clone(), problem.a[i][j].clone());
            for k in 0..n {
                if problem.s[i][k] != zero() {
                    total = i_sub(&total, &i_scale(&boxes[k][j], &problem.s[i][k]));
                }
            }
            line.push(total);
        }
        closed.push(line);
    }
    Ok(interval_hurwitz(&interval_characteristic(&closed)))
}

fn eps_of(bits: u32) -> Rat {
    q(1) / Rat::from(Int::from(2).pow(bits as usize))
}

fn stabilising(problem: &Problem, x: &AMatrix) -> PolyResult<bool> {
    for bits in [32u32, 96, 256] {
        if let Some(decided) = stable_by_intervals(problem, x, &eps_of(bits))? {
            return Ok(decided);
        }
    }
    hurwitz(&characteristic(&problem.closed_loop(x)?)?)
}

fn expression(a: &Algebraic) -> PolyResult<Basic> {
    if a.is_rational() {
        return Ok(Basic::from_rational(a.rational_value()?));
    }
    let minpoly: Vec<Int> = a.minpoly();
    if minpoly.len() == 3 {
        let (p, qq, r) = (minpoly[0].clone(), minpoly[1].clone(), minpoly[2].clone());
        let root: Basic = crate::matrices::eigen::sqrt(&Basic::from_rational(Rat::from(
            qq.clone() * qq.clone() - Int::from(4) * p.clone() * r,
        )))?;
        let center: Rat = Rat::from(-qq.clone()) / Rat::from(Int::from(2) * p.clone());
        let larger: bool = a.compare(&alg(&center))? > 0;
        let two_p: Basic = Basic::from_rational(Rat::from(Int::from(2) * p.clone()));
        let minus_q: Basic = Basic::from_rational(Rat::from(-qq));
        let numerator: Basic = if larger == (p > Int::from(0)) {
            crate::core::add::add(&[minus_q, root])?
        } else {
            crate::core::add::sub(&minus_q, &root)?
        };
        return crate::core::mul::div(&numerator, &two_p);
    }
    crate::polys::cad::indexedroot::real_root(&Arc::new(a.clone()), "x")
}

fn matrix(x: &AMatrix) -> PolyResult<ExprMatrix> {
    let n: usize = x.len();
    ExprMatrix::new(n, n, x.iter().flatten().map(expression).collect::<PolyResult<_>>()?)
}

/// The real symmetric solutions of `A^T X + X A - X B R^-1 B^T X + Q = 0`
/// (`riccati_solutions`).
pub fn riccati_solutions(
    a: &ExprMatrix,
    b: &ExprMatrix,
    qm: &ExprMatrix,
    r: &ExprMatrix,
) -> PolyResult<Vec<ExprMatrix>> {
    Problem::new(a, b, qm, r)?.solutions()?.iter().map(matrix).collect()
}

/// The stabilising solution, or `None` (`care_solve`).
pub fn care_solve(a: &ExprMatrix, b: &ExprMatrix, qm: &ExprMatrix, r: &ExprMatrix) -> PolyResult<Option<ExprMatrix>> {
    let problem: Problem = Problem::new(a, b, qm, r)?;
    for x in problem.solutions()? {
        if stabilising(&problem, &x)? {
            return Ok(Some(matrix(&x)?));
        }
    }
    Ok(None)
}

/// `(K, X)`: the optimal gain `R^-1 B^T X` and the stabilising solution
/// (`lqr`).
pub fn lqr(a: &ExprMatrix, b: &ExprMatrix, qm: &ExprMatrix, r: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let problem: Problem = Problem::new(a, b, qm, r)?;
    for x in problem.solutions()? {
        if stabilising(&problem, &x)? {
            let n: usize = problem.n;
            let mut gain: Vec<Basic> = Vec::new();
            for row in &problem.g {
                for j in 0..n {
                    let mut total: Algebraic = alg(&zero());
                    for k in 0..n {
                        if row[k] != zero() {
                            total = total.add(&x[k][j].mul(&alg(&row[k]))?)?;
                        }
                    }
                    gain.push(expression(&total)?);
                }
            }
            return Ok((ExprMatrix::new(problem.g.len(), n, gain)?, matrix(&x)?));
        }
    }
    Err(value_error(
        "the Riccati equation has no stabilising solution".to_string(),
    ))
}

/// Whether `A - B K` is Hurwitz (`closed_loop_is_stable`).
pub fn closed_loop_is_stable(a: &ExprMatrix, b: &ExprMatrix, k: &ExprMatrix) -> PolyResult<bool> {
    let (ar, br, kr) = (rational_rows(a, "A")?, rational_rows(b, "B")?, rational_rows(k, "K")?);
    let bk: Rows = product(&br, &kr);
    let closed: AMatrix = ar
        .iter()
        .zip(bk.iter())
        .map(|(row, other)| {
            row.iter()
                .zip(other.iter())
                .map(|(p, s)| alg(&(p.clone() - s.clone())))
                .collect()
        })
        .collect();
    hurwitz(&characteristic(&closed)?)
}

struct DiscreteProblem {
    n: usize,
    m: usize,
    a: Rows,
    b: Rows,
    q: Rows,
    r: Rows,
}

impl DiscreteProblem {
    fn new(a: &ExprMatrix, b: &ExprMatrix, qm: &ExprMatrix, r: &ExprMatrix) -> PolyResult<DiscreteProblem> {
        let n: usize = square(a)?;
        let m: usize = square(r)?;
        if b.shape() != (n, m) || qm.shape() != (n, n) {
            return Err(value_error(format!(
                "shapes: A {}, B {}, Q {}, R {}",
                shape(a),
                shape(b),
                shape(qm),
                shape(r)
            )));
        }
        let problem = DiscreteProblem {
            n,
            m,
            a: rational_rows(a, "A")?,
            b: rational_rows(b, "B")?,
            q: rational_rows(qm, "Q")?,
            r: rational_rows(r, "R")?,
        };
        if problem.q != transpose(&problem.q) || problem.r != transpose(&problem.r) {
            return Err(value_error("Q and R must be symmetric".to_string()));
        }
        Ok(problem)
    }

    #[allow(clippy::type_complexity)]
    fn solutions(&self) -> PolyResult<Vec<(AMatrix, AMatrix)>> {
        let (n, m) = (self.n, self.m);
        if n == 0 {
            return Ok(vec![(Vec::new(), (0..m).map(|_| Vec::new()).collect())]);
        }
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
        let gains: Vec<(usize, usize)> = (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).collect();
        let mut all_names: Vec<String> = names(&pairs, "x");
        all_names.extend(names(&gains, "k"));
        let ring = PolyRing::new(QQ, all_names, MonomialOrder::Lex)?;
        let gens: Vec<P> = ring.gens();
        let unknown =
            |i: usize, j: usize| gens[pairs.iter().position(|p| *p == (i.min(j), i.max(j))).expect("a pair")].clone();
        let x: Vec<Vec<P>> = (0..n).map(|i| (0..n).map(|j| unknown(i, j)).collect()).collect();
        let k: Vec<Vec<P>> = (0..m)
            .map(|i| (0..n).map(|j| gens[pairs.len() + i * n + j].clone()).collect())
            .collect();
        let constant = |c: &Rows| -> Vec<Vec<P>> {
            c.iter()
                .map(|row| row.iter().map(|v| ring.ground(v.clone())).collect())
                .collect()
        };
        let times = |a: &Vec<Vec<P>>, b: &Vec<Vec<P>>| -> Vec<Vec<P>> {
            let columns: usize = b.first().map(|r| r.len()).unwrap_or(0);
            a.iter()
                .map(|row| {
                    (0..columns)
                        .map(|j| {
                            let mut total: P = ring.zero();
                            for (l, entry) in row.iter().enumerate() {
                                if !entry.terms.is_empty() && !b[l][j].terms.is_empty() {
                                    total = total.add(&entry.mul(&b[l][j]));
                                }
                            }
                            total
                        })
                        .collect()
                })
                .collect()
        };
        let (am, bm) = (constant(&self.a), constant(&self.b));
        let (at, bt) = (constant(&transpose(&self.a)), constant(&transpose(&self.b)));
        let btx = times(&bt, &x);
        let atx = times(&at, &x);
        let gain_left = times(&btx, &bm);
        let gain_right = times(&btx, &am);
        let gain_product = times(&gain_left, &k);
        let rk = times(&constant(&self.r), &k);
        let mut equations: Vec<P> = Vec::new();
        for i in 0..m {
            for j in 0..n {
                equations.push(rk[i][j].add(&gain_product[i][j]).sub(&gain_right[i][j]));
            }
        }
        let bk = times(&bm, &k);
        let closed: Vec<Vec<P>> = (0..n)
            .map(|i| (0..n).map(|j| am[i][j].sub(&bk[i][j])).collect())
            .collect();
        let update = times(&atx, &closed);
        for i in 0..n {
            for j in 0..n {
                equations.push(update[i][j].add(&ring.ground(self.q[i][j].clone())).sub(&x[i][j]));
            }
        }
        let mut found: Vec<(AMatrix, AMatrix)> = Vec::new();
        for point in crate::polys::zerodim::solve_zero_dimensional(&equations, &ring)? {
            if point.iter().all(|v| v.is_real()) {
                let value = |i: usize, j: usize| {
                    point[pairs.iter().position(|p| *p == (i.min(j), i.max(j))).expect("a pair")].clone()
                };
                let solution: AMatrix = (0..n).map(|i| (0..n).map(|j| value(i, j)).collect()).collect();
                let gain: AMatrix = (0..m)
                    .map(|i| (0..n).map(|j| point[pairs.len() + i * n + j].clone()).collect())
                    .collect();
                found.push((solution, gain));
            }
        }
        Ok(found)
    }

    fn stabilising(&self, k: &AMatrix) -> PolyResult<bool> {
        let (n, m) = (self.n, self.m);
        for bits in [32u32, 96, 256] {
            let eps: Rat = eps_of(bits);
            let boxes: Vec<Vec<Interval>> = k
                .iter()
                .map(|row| row.iter().map(|v| box_of(v, &eps)).collect::<PolyResult<_>>())
                .collect::<PolyResult<_>>()?;
            let mut closed: Vec<Vec<Interval>> = Vec::new();
            for i in 0..n {
                let mut line: Vec<Interval> = Vec::new();
                for j in 0..n {
                    let mut total: Interval = (self.a[i][j].clone(), self.a[i][j].clone());
                    for l in 0..m {
                        if self.b[i][l] != zero() {
                            total = i_sub(&total, &i_scale(&boxes[l][j], &self.b[i][l]));
                        }
                    }
                    line.push(total);
                }
                closed.push(line);
            }
            if let Some(decided) = interval_hurwitz(&interval_moebius(&interval_characteristic(&closed))) {
                return Ok(decided);
            }
        }
        let mut exact: AMatrix = (0..n).map(|i| (0..n).map(|j| alg(&self.a[i][j])).collect()).collect();
        for i in 0..n {
            for j in 0..n {
                for l in 0..m {
                    if self.b[i][l] != zero() {
                        exact[i][j] = exact[i][j].sub(&k[l][j].mul(&alg(&self.b[i][l]))?)?;
                    }
                }
            }
        }
        let coefficients: Vec<Algebraic> = characteristic(&exact)?;
        let weights: Vec<Vec<i64>> = moebius_weights(n);
        let mut transformed: Vec<Algebraic> = Vec::new();
        for j in 0..=n {
            let mut total: Algebraic = alg(&zero());
            for kk in 0..=n {
                if weights[kk][j] != 0 {
                    total = total.add(&coefficients[kk].mul(&alg(&q(weights[kk][j])))?)?;
                }
            }
            transformed.push(total);
        }
        if transformed[0].sign()? == 0 {
            return Ok(false);
        }
        let first: Algebraic = transformed[0].clone();
        let normalised: Vec<Algebraic> = transformed.iter().map(|c| c.div(&first)).collect::<PolyResult<_>>()?;
        hurwitz(&normalised)
    }
}

/// The real symmetric solutions of the discrete Riccati equation
/// (`discrete_riccati_solutions`).
pub fn discrete_riccati_solutions(
    a: &ExprMatrix,
    b: &ExprMatrix,
    qm: &ExprMatrix,
    r: &ExprMatrix,
) -> PolyResult<Vec<ExprMatrix>> {
    DiscreteProblem::new(a, b, qm, r)?
        .solutions()?
        .iter()
        .map(|(x, _)| matrix(x))
        .collect()
}

/// The stabilising solution of the discrete equation, or `None`
/// (`dare_solve`).
pub fn dare_solve(a: &ExprMatrix, b: &ExprMatrix, qm: &ExprMatrix, r: &ExprMatrix) -> PolyResult<Option<ExprMatrix>> {
    let problem: DiscreteProblem = DiscreteProblem::new(a, b, qm, r)?;
    for (x, k) in problem.solutions()? {
        if problem.stabilising(&k)? {
            return Ok(Some(matrix(&x)?));
        }
    }
    Ok(None)
}

/// `(K, X)` of the discrete LQR (`dlqr`).
pub fn dlqr(a: &ExprMatrix, b: &ExprMatrix, qm: &ExprMatrix, r: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let problem: DiscreteProblem = DiscreteProblem::new(a, b, qm, r)?;
    for (x, k) in problem.solutions()? {
        if problem.stabilising(&k)? {
            let gain: ExprMatrix = ExprMatrix::new(
                problem.m,
                problem.n,
                k.iter().flatten().map(expression).collect::<PolyResult<_>>()?,
            )?;
            return Ok((gain, matrix(&x)?));
        }
    }
    Err(value_error(
        "the discrete Riccati equation has no stabilising solution".to_string(),
    ))
}
