//! Port of `sympy_extras/concrete/guess.py`: guessing linear recurrences
//! with polynomial coefficients, linear differential equations and
//! algebraic equations from the first terms of a sequence or a power
//! series. The orders and degrees are screened by the rank of the
//! system modulo the prime `2**61 - 1`, the kernel computed over `QQ`,
//! with a margin of equations more than unknowns.

use std::sync::Arc;

use crate::core::basic::{Basic, UndefinedFunction};
use crate::error::{value_error, PolyResult};
use crate::numbers::{denom, gcd, numer, Int, Rat};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::rationalfield::QQ;

const PRIME: u128 = (1u128 << 61) - 1;

/// The default margin of equations over unknowns.
pub const MARGIN: usize = 3;

fn modulo(a: &Int) -> u128 {
    let p: Int = Int::from(PRIME);
    let r: Int = crate::numbers::mod_floor(a, &p);
    u128::try_from(&r).unwrap_or(0)
}

fn power_mod(mut base: u128, mut exponent: u128) -> u128 {
    let mut result: u128 = 1;
    base %= PRIME;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base % PRIME;
        }
        base = base * base % PRIME;
        exponent >>= 1;
    }
    result
}

fn inverse(a: u128) -> u128 {
    power_mod(a, PRIME - 2)
}

fn modular(q: &Rat) -> u128 {
    let d: u128 = modulo(&denom(q));
    if d == 0 {
        return 0;
    }
    modulo(&numer(q)) * inverse(d) % PRIME
}

fn has_kernel(rows: &[Vec<u128>], columns: &[usize]) -> bool {
    let mut work: Vec<Vec<u128>> = rows
        .iter()
        .map(|row| columns.iter().map(|c| row[*c]).collect())
        .collect();
    let width: usize = columns.len();
    for (rank, column) in (0..width).enumerate() {
        let Some(pivot) = (rank..work.len()).find(|i| work[*i][column] != 0) else {
            return true;
        };
        work.swap(rank, pivot);
        let top: Vec<u128> = work[rank].clone();
        let inv: u128 = inverse(top[column]);
        for row in work.iter_mut().skip(rank + 1) {
            let factor: u128 = row[column] * inv % PRIME;
            if factor != 0 {
                for j in column + 1..width {
                    row[j] = (row[j] + PRIME - factor * top[j] % PRIME) % PRIME;
                }
                row[column] = 0;
            }
        }
    }
    false
}

struct Candidate {
    order: usize,
    degree: usize,
}

impl Candidate {
    fn size(&self) -> usize {
        (self.order + 1) * (self.degree + 1)
    }
}

type Entry<'a> = dyn FnMut(usize, usize, usize) -> Rat + 'a;

fn candidates(
    entry: &mut Entry,
    equations: &dyn Fn(usize) -> i64,
    orders: &[usize],
    max_degree: Option<usize>,
    margin: usize,
) -> Vec<Candidate> {
    let mut found: Vec<Candidate> = Vec::new();
    for &r in orders {
        let count: i64 = equations(r);
        let mut top: i64 = (count - margin as i64).div_euclid(r as i64 + 1) - 1;
        if let Some(d) = max_degree {
            top = top.min(d as i64);
        }
        if top < 0 || count < 0 {
            continue;
        }
        let top: usize = top as usize;
        let mut rows: Vec<Vec<u128>> = Vec::new();
        for m in 0..count as usize {
            let mut row: Vec<u128> = Vec::new();
            for i in 0..=r {
                for j in 0..=top {
                    row.push(modular(&entry(i, j, m)));
                }
            }
            rows.push(row);
        }
        let columns =
            |d: usize| -> Vec<usize> { (0..=r).flat_map(|i| (0..=d).map(move |j| i * (top + 1) + j)).collect() };
        if !has_kernel(&rows, &columns(top)) {
            continue;
        }
        let (mut low, mut high) = (0usize, top);
        while low < high {
            let middle: usize = (low + high) / 2;
            if has_kernel(&rows, &columns(middle)) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        found.push(Candidate { order: r, degree: low });
    }
    found.sort_by_key(|c| (c.size(), c.order));
    found
}

fn integer_polynomials(vector: &[Rat], order: usize, degree: usize) -> Vec<Vec<Int>> {
    let mut common: Int = Int::from(1);
    for c in vector {
        let d: Int = denom(c);
        common = &common * &d / gcd(&common, &d);
    }
    let integers: Vec<Int> = vector
        .iter()
        .map(|c| numer(&(c.clone() * Rat::from(common.clone()))))
        .collect();
    let mut content: Int = Int::from(0);
    for value in &integers {
        content = gcd(&content, value);
    }
    let mut polynomials: Vec<Vec<Int>> = Vec::new();
    for i in 0..=order {
        let mut ascending: Vec<Int> = integers[i * (degree + 1)..(i + 1) * (degree + 1)]
            .iter()
            .map(|v| v / &content)
            .collect();
        while ascending.last().is_some_and(|c| *c == Int::from(0)) {
            ascending.pop();
        }
        ascending.reverse();
        polynomials.push(ascending);
    }
    while polynomials.last().is_some_and(|p| p.is_empty()) {
        polynomials.pop();
    }
    if polynomials.last().is_some_and(|p| crate::numbers::is_negative(&p[0])) {
        polynomials = polynomials
            .into_iter()
            .map(|p| p.into_iter().map(|c| -c).collect())
            .collect();
    }
    polynomials
}

fn search(
    entry: &mut Entry,
    equations: &dyn Fn(usize) -> i64,
    orders: &[usize],
    max_degree: Option<usize>,
    margin: usize,
) -> PolyResult<Option<Vec<Vec<Int>>>> {
    if margin < 1 {
        return Err(value_error("the margin must be positive".to_string()));
    }
    for candidate in candidates(entry, equations, orders, max_degree, margin) {
        let (r, d) = (candidate.order, candidate.degree);
        let mut rows: Vec<Vec<Rat>> = Vec::new();
        for m in 0..equations(r).max(0) as usize {
            let mut row: Vec<Rat> = Vec::new();
            for i in 0..=r {
                for j in 0..=d {
                    row.push(entry(i, j, m));
                }
            }
            rows.push(row);
        }
        let kernel: Vec<Vec<Rat>> = QQ.nullspace(&rows, candidate.size())?;
        if kernel.is_empty() {
            continue;
        }
        let nonzeros = |v: &Vec<Rat>| v.iter().filter(|c| **c != Rat::ZERO).count();
        let mut best: &Vec<Rat> = &kernel[0];
        for v in &kernel[1..] {
            if nonzeros(v) < nonzeros(best) {
                best = v;
            }
        }
        return Ok(Some(integer_polynomials(best, r, d)));
    }
    Ok(None)
}

fn polynomial_value(p: &[Int], at: i64) -> Int {
    let mut value: Int = Int::from(0);
    for c in p {
        value = value * Int::from(at) + c;
    }
    value
}

fn polynomial_expr(p: &[Int], x: &Basic) -> PolyResult<Basic> {
    let mut terms: Vec<Basic> = Vec::new();
    let degree: usize = p.len().saturating_sub(1);
    for (k, c) in p.iter().enumerate() {
        if *c != Int::from(0) {
            let power: Basic = crate::core::power::pow(x, &Basic::int((degree - k) as i64))?;
            terms.push(crate::core::mul::mul(&[
                Basic::from_rational(Rat::from(c.clone())),
                power,
            ])?);
        }
    }
    let mut result: Basic = Basic::int(0);
    for t in terms {
        result = crate::core::add::add(&[result, t])?;
    }
    Ok(result)
}

fn degree_of(polynomials: &[Vec<Int>]) -> i64 {
    polynomials.iter().map(|p| p.len() as i64 - 1).max().unwrap_or(-1)
}

/// `sum_i p_i(n) a(n + i) = 0`, the polynomials highest degree first
/// (`GuessedRecurrence`).
#[derive(Debug, Clone, PartialEq)]
pub struct GuessedRecurrence {
    pub coefficients: Vec<Vec<Int>>,
}

impl GuessedRecurrence {
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub fn degree(&self) -> i64 {
        degree_of(&self.coefficients)
    }

    /// Whether the terms satisfy the recurrence (`holds`).
    pub fn holds(&self, terms: &[Rat]) -> bool {
        let r: usize = self.order();
        for n in 0..terms.len().saturating_sub(r) {
            let mut total: Rat = Rat::ZERO;
            for (i, p) in self.coefficients.iter().enumerate() {
                total += Rat::from(polynomial_value(p, n as i64)) * terms[n + i].clone();
            }
            if total != Rat::ZERO {
                return false;
            }
        }
        true
    }

    /// The terms continued to `count` of them (`extend`).
    pub fn extend(&self, terms: &[Rat], count: usize) -> PolyResult<Vec<Rat>> {
        let mut a: Vec<Rat> = terms.to_vec();
        let r: usize = self.order();
        if a.len() < r {
            return Err(value_error(format!("{} initial terms are needed", r)));
        }
        while a.len() < count {
            let n: usize = a.len() - r;
            let leading: Int = polynomial_value(&self.coefficients[r], n as i64);
            if leading == Int::from(0) {
                return Err(value_error(format!("the leading coefficient vanishes at n = {}", n)));
            }
            let mut rest: Rat = Rat::ZERO;
            for (i, p) in self.coefficients[..r].iter().enumerate() {
                rest += Rat::from(polynomial_value(p, n as i64)) * a[n + i].clone();
            }
            a.push(-rest / Rat::from(leading));
        }
        Ok(a)
    }

    /// `sum_i p_i(n) a(n + i)` (`as_expr`).
    pub fn as_expr(&self, a: &Arc<UndefinedFunction>, n: &Basic) -> PolyResult<Basic> {
        let mut result: Basic = Basic::int(0);
        for (i, p) in self.coefficients.iter().enumerate() {
            if !p.is_empty() {
                let shifted: Basic = crate::core::add::add(&[n.clone(), Basic::int(i as i64)])?;
                let term: Basic = crate::core::mul::mul(&[polynomial_expr(p, n)?, a.apply(vec![shifted])?])?;
                result = crate::core::add::add(&[result, term])?;
            }
        }
        Ok(result)
    }
}

/// A recurrence of the terms, the smallest `(order + 1) (degree + 1)`
/// first (`guess_recurrence`).
pub fn guess_recurrence(
    terms: &[Rat],
    max_order: Option<usize>,
    max_degree: Option<usize>,
    margin: usize,
) -> PolyResult<Option<GuessedRecurrence>> {
    let a: Vec<Rat> = terms.to_vec();
    let count: usize = a.len();
    let mut entry = |i: usize, j: usize, m: usize| -> Rat { Rat::from(Int::from(m).pow(j)) * a[m + i].clone() };
    let top: usize = match max_order {
        None => count,
        Some(o) => count.min(o + 1),
    };
    let orders: Vec<usize> = (0..top).collect();
    let equations = |r: usize| count as i64 - r as i64;
    match search(&mut entry, &equations, &orders, max_degree, margin)? {
        Some(p) if !p.is_empty() => Ok(Some(GuessedRecurrence { coefficients: p })),
        _ => Ok(None),
    }
}

/// `sum_k q_k(x) f^(k)(x) + p(x) = 0` (`GuessedDifferentialEquation`).
#[derive(Debug, Clone, PartialEq)]
pub struct GuessedDifferentialEquation {
    pub coefficients: Vec<Vec<Int>>,
    pub inhomogeneous: Vec<Int>,
}

impl GuessedDifferentialEquation {
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub fn degree(&self) -> i64 {
        degree_of(&self.coefficients)
    }
}

/// A linear differential equation of the power series with these
/// coefficients (`guess_differential_equation`).
pub fn guess_differential_equation(
    terms: &[Rat],
    max_order: Option<usize>,
    max_degree: Option<usize>,
    margin: usize,
) -> PolyResult<Option<GuessedDifferentialEquation>> {
    let a: Vec<Rat> = terms.to_vec();
    let count: usize = a.len();
    let derivative = |k: usize| -> Vec<Rat> {
        let mut result: Vec<Rat> = Vec::new();
        for m in 0..count.saturating_sub(k) {
            let mut c: Rat = a[m + k].clone();
            for t in 1..=k {
                c *= Rat::from(Int::from(m + t));
            }
            result.push(c);
        }
        result
    };
    let mut derivatives: Vec<Vec<Rat>> = Vec::new();
    let mut entry = |i: usize, j: usize, m: usize| -> Rat {
        if i == 0 {
            return if m == j { Rat::from(Int::from(1)) } else { Rat::ZERO };
        }
        while derivatives.len() < i {
            let k: usize = derivatives.len();
            derivatives.push(derivative(k));
        }
        if m >= j {
            derivatives[i - 1].get(m - j).cloned().unwrap_or(Rat::ZERO)
        } else {
            Rat::ZERO
        }
    };
    if count == 0 {
        return Ok(None);
    }
    let top: usize = match max_order {
        None => count - 1,
        Some(o) => (count - 1).min(o),
    };
    let orders: Vec<usize> = (1..top + 2).collect();
    let equations = |r: usize| count as i64 - (r as i64 - 1);
    match search(&mut entry, &equations, &orders, max_degree, margin)? {
        Some(p) if p.len() >= 2 => Ok(Some(GuessedDifferentialEquation {
            coefficients: p[1..].to_vec(),
            inhomogeneous: p[0].clone(),
        })),
        _ => Ok(None),
    }
}

/// `sum_k q_k(x) f**k = 0` (`GuessedAlgebraicEquation`).
#[derive(Debug, Clone, PartialEq)]
pub struct GuessedAlgebraicEquation {
    pub coefficients: Vec<Vec<Int>>,
}

fn series_product(a: &[Rat], b: &[Rat], count: usize) -> Vec<Rat> {
    let mut result: Vec<Rat> = vec![Rat::ZERO; count];
    for (i, c) in a.iter().take(count).enumerate() {
        if *c != Rat::ZERO {
            for (j, e) in b.iter().take(count - i).enumerate() {
                result[i + j] += c.clone() * e.clone();
            }
        }
    }
    result
}

/// An algebraic equation of the power series with these coefficients
/// (`guess_algebraic_equation`).
pub fn guess_algebraic_equation(
    terms: &[Rat],
    max_order: Option<usize>,
    max_degree: Option<usize>,
    margin: usize,
) -> PolyResult<Option<GuessedAlgebraicEquation>> {
    let a: Vec<Rat> = terms.to_vec();
    let count: usize = a.len();
    let mut first: Vec<Rat> = vec![Rat::ZERO; count];
    if count > 0 {
        first[0] = Rat::from(Int::from(1));
    }
    let mut powers: Vec<Vec<Rat>> = vec![first];
    let mut entry = |i: usize, j: usize, m: usize| -> Rat {
        while powers.len() <= i {
            let next: Vec<Rat> = series_product(&powers[powers.len() - 1], &a, count);
            powers.push(next);
        }
        if m >= j {
            powers[i][m - j].clone()
        } else {
            Rat::ZERO
        }
    };
    if count == 0 {
        return Ok(None);
    }
    let top: usize = match max_order {
        None => count - 1,
        Some(o) => (count - 1).min(o),
    };
    let orders: Vec<usize> = (1..top + 1).collect();
    let equations = |_: usize| count as i64;
    match search(&mut entry, &equations, &orders, max_degree, margin)? {
        Some(p) if p.len() >= 2 => Ok(Some(GuessedAlgebraicEquation { coefficients: p })),
        _ => Ok(None),
    }
}

/// The seconds given to the solution of the guessed recurrence
/// (`_SOLVE_SECONDS`).
const SOLVE_SECONDS: f64 = 10.0;

/// An expression in `n` whose values at `0, 1, ...` are the terms: the
/// solution of the guessed recurrence with the first terms as initial
/// values, when the recurrence has a full set of d'Alembertian solutions;
/// `None` otherwise. The expression is checked on every term given
/// (`guess_closed_form`).
pub fn guess_closed_form(
    terms: &[Rat],
    n: &Basic,
    max_order: Option<usize>,
    max_degree: Option<usize>,
    margin: usize,
) -> PolyResult<Option<Basic>> {
    let Some(recurrence) = guess_recurrence(terms, max_order, max_degree, margin)? else {
        return Ok(None);
    };
    let r: usize = recurrence.order();
    if r == 0 {
        // p_0(n) a(n) = 0: zero outside the roots of p_0
        return Ok(if terms.iter().all(|t| *t == Rat::ZERO) {
            Some(Basic::int(0))
        } else {
            None
        });
    }
    // the initial values are taken after the last zero of the leading
    // coefficient, from where on the recurrence determines the sequence
    let mut start: usize = 0;
    for k in 0..terms.len().saturating_sub(r) {
        if polynomial_value(&recurrence.coefficients[r], k as i64) == Int::ZERO {
            start = k + 1;
        }
    }
    if start + r > terms.len() {
        return Ok(None);
    }
    let y: Arc<UndefinedFunction> = Arc::new(UndefinedFunction::new("y", None, &[]));
    let mut initial: Vec<(Basic, Basic)> = Vec::new();
    for (k, term) in terms.iter().enumerate().skip(start).take(r) {
        initial.push((y.apply(vec![Basic::int(k as i64)])?, Basic::from_rational(term.clone())));
    }
    let equation: Basic = recurrence.as_expr(&y, n)?;
    let unknown: Basic = y.apply(vec![n.clone()])?;
    let solution: Option<Basic> = crate::timeout::attempt_within(Some(SOLVE_SECONDS), || {
        crate::solvers::recurr::rsolve_linear(&equation, &unknown, Some(&initial))
    })?
    .flatten();
    let Some(solution) = solution else {
        return Ok(None);
    };
    for (k, value) in terms.iter().enumerate() {
        let at: Basic = crate::core::subs::subs(&solution, n, &Basic::int(k as i64))?;
        let evaluated: Basic = crate::core::function::expand_default(&crate::core::function::doit(&at)?)?;
        if evaluated != Basic::from_rational(value.clone()) {
            return Ok(None);
        }
    }
    Ok(Some(solution))
}
