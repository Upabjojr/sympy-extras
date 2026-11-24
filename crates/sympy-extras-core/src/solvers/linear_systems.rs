//! Port of `sympy_extras/solvers/linear_systems.py`: first order linear
//! systems `Y' = A Y + b` with `A` rational in `x`: cyclic vectors and the
//! scalar equation, rational solutions by Barkatou's method,
//! hyperexponential solutions, the matrix exponential for constant
//! coefficients and for `A = g'(x) M`, and variation of constants.

#![allow(clippy::mutable_key_type, clippy::needless_range_loop)]

use std::sync::Arc;

use crate::core::basic::{Basic, Function, Head, UndefinedFunction};
use crate::error::{PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::polys::domains::domain::{Domain, SparseRow};
use crate::polys::domains::rationalfield::RationalField;
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::solvers::linear_ode::{
    factor_list, field_failure, has_integral, hyperexponential_search, indicial_at, indicial_at_infinity, inverse, lcm,
    log_derivative, normal, nullspace, resultant, solve_operator, sum, LinearOperator, RationalRows, UPoly,
};

type R = PolyResult<Basic>;
/// The rows of a matrix of rational functions (`Rows`).
type Rows = Vec<Vec<Basic>>;
type Field = SparseFractionField<RationalField>;
type Element = SparseFracElement<RationalField>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn is_rational_function(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    crate::core::function::xreplace_evaluated(e, &rule.iter().cloned().collect())
}
fn is_genuine_not_implemented(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if !m.starts_with(crate::error::NOT_PORTED_PREFIX))
}

/// The seed of the generator of candidate cyclic vectors (`_CYCLIC_SEED`).
const CYCLIC_SEED: i64 = 2002;
/// The number of random candidates after the unit vectors
/// (`_CYCLIC_TRIALS`).
const CYCLIC_TRIALS: usize = 24;

/// The entries of the square matrix, rational in `x` and normalised
/// (`_entries`).
fn entries(a: &ExprMatrix, x: &Basic) -> PolyResult<Rows> {
    if a.rows != a.cols || a.rows == 0 {
        return Err(PolyError::Value(
            "the matrix of the system must be square and nonempty".to_string(),
        ));
    }
    let mut rows: Rows = Vec::new();
    for i in 0..a.rows {
        let mut row: Vec<Basic> = Vec::new();
        for j in 0..a.cols {
            let e: &Basic = a.at(i, j);
            if !is_rational_function(e, x) {
                return Err(PolyError::Value(format!(
                    "the entry {} is not a rational function of {}",
                    e, x
                )));
            }
            row.push(normal(e)?);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn column(b: &ExprMatrix, n: usize) -> PolyResult<Vec<Basic>> {
    if b.shape() != (n, 1) {
        return Err(PolyError::Value(format!(
            "the inhomogeneous term must be a column of length {}",
            n
        )));
    }
    Ok(b.entries.clone())
}

fn matrix(rows: &Rows) -> PolyResult<ExprMatrix> {
    ExprMatrix::from_rows(rows)
}

/// The columns side by side (`_columns_matrix`).
fn columns_matrix(columns: &[ExprMatrix], n: usize) -> PolyResult<ExprMatrix> {
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..n {
        for c in columns {
            entries.push(c.at(i, 0).clone());
        }
    }
    ExprMatrix::new(n, columns.len(), entries)
}

/// `c A + c'` (`_next_row`).
fn next_row(a: &Rows, c: &[Basic], x: &Basic) -> PolyResult<Vec<Basic>> {
    let n: usize = c.len();
    (0..n)
        .map(|j| {
            let mut terms: Vec<Basic> = Vec::new();
            for i in 0..n {
                if c[i] != Basic::zero() && a[i][j] != Basic::zero() {
                    terms.push(mul(&[c[i].clone(), a[i][j].clone()])?);
                }
            }
            normal(&add(&[diff(&c[j], x, 1)?, sum(&terms)?])?)
        })
        .collect()
}

/// The unit vectors, then random vectors (`_candidates`).
fn candidates(n: usize, x: &Basic) -> PolyResult<Rows> {
    let mut result: Rows = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { Basic::one() } else { Basic::zero() })
                .collect()
        })
        .collect();
    let mut generator = crate::random::PyRandom::from_int(&crate::numbers::Int::from(CYCLIC_SEED));
    for trial in 0..CYCLIC_TRIALS {
        let degree: usize = if trial < CYCLIC_TRIALS / 3 {
            0
        } else {
            (n - 1).min(1 + trial % n)
        };
        let mut vector: Vec<Basic> = Vec::new();
        for _ in 0..n {
            let mut terms: Vec<Basic> = Vec::new();
            for k in 0..=degree {
                let c: Basic = int(generator.randint(-3, 3));
                terms.push(if k > 0 { mul(&[c, pow(x, &int(k as i64))?])? } else { c });
            }
            vector.push(sum(&terms)?);
        }
        if vector.iter().any(|v| *v != Basic::zero()) {
            result.push(vector);
        }
    }
    Ok(result)
}

/// The scalar form of `Y' = A Y` given by a cyclic row vector
/// (`CyclicReduction`).
#[derive(Clone, Debug)]
pub struct CyclicReduction {
    pub vector: ExprMatrix,
    pub matrix: ExprMatrix,
    pub operator: LinearOperator,
    pub x: Basic,
    pub inverse: Rows,
}

impl CyclicReduction {
    fn new(
        vector: Vec<Basic>,
        rows: &Rows,
        inverse: Rows,
        operator: LinearOperator,
        x: &Basic,
    ) -> PolyResult<CyclicReduction> {
        let n: usize = vector.len();
        Ok(CyclicReduction {
            vector: ExprMatrix::new(1, n, vector)?,
            matrix: matrix(rows)?,
            operator,
            x: x.clone(),
            inverse,
        })
    }

    /// The solution of the system given by a solution `u` of the operator
    /// (`to_system`).
    pub fn to_system(&self, u: &Basic) -> PolyResult<ExprMatrix> {
        let x: &Basic = &self.x;
        let inverse: &Rows = &self.inverse;
        let n: usize = inverse.len();
        let rho: Basic = if u.has(x) { log_derivative(u, x)? } else { Basic::zero() };
        if is_rational_function(&rho, x) {
            // u^(k) = u R_k with R_0 = 1 and R_{k+1} = R_k' + rho R_k
            let mut ratios: Vec<Basic> = vec![Basic::one()];
            for _ in 0..n.saturating_sub(1) {
                let last: Basic = ratios.last().expect("a ratio").clone();
                ratios.push(normal(&add(&[diff(&last, x, 1)?, mul(&[rho.clone(), last])?])?)?);
            }
            let parts: Vec<Basic> = (0..n)
                .map(|i| {
                    let terms: Vec<Basic> = (0..n)
                        .map(|k| mul(&[inverse[i][k].clone(), ratios[k].clone()]))
                        .collect::<PolyResult<_>>()?;
                    normal(&sum(&terms)?)
                })
                .collect::<PolyResult<_>>()?;
            if is_rational_function(u, x) {
                return Ok(ExprMatrix::column(
                    parts
                        .iter()
                        .map(|p| normal(&mul(&[u.clone(), p.clone()])?))
                        .collect::<PolyResult<_>>()?,
                ));
            }
            return Ok(ExprMatrix::column(
                parts
                    .iter()
                    .map(|p| mul(&[u.clone(), p.clone()]))
                    .collect::<PolyResult<_>>()?,
            ));
        }
        let mut derivatives: Vec<Basic> = vec![u.clone()];
        for k in 1..n {
            derivatives.push(diff(u, x, k as u64)?);
        }
        let mut entries: Vec<Basic> = Vec::new();
        for row in inverse {
            let mut terms: Vec<Basic> = Vec::new();
            for k in 0..n {
                if row[k] != Basic::zero() {
                    terms.push(mul(&[row[k].clone(), derivatives[k].clone()])?);
                }
            }
            entries.push(simplified(&sum(&terms)?)?);
        }
        Ok(ExprMatrix::column(entries))
    }

    /// (`__repr__`)
    pub fn repr(&self) -> String {
        let vector: Vec<String> = self.vector.entries.iter().map(|e| e.to_string()).collect();
        format!("CyclicReduction([[{}]], {:?})", vector.join(", "), self.operator)
    }
}

/// An antiderivative of a rational function with the roots of the
/// logarithmic part of degree at most two written out (`rational_integral`).
pub fn rational_integral(f: &Basic, x: &Basic) -> R {
    use crate::integrals::risch::rationaltools::{polynomial_part, ratint_logpart, ratint_ratpart};
    let failure = |e: &PolyError| {
        matches!(e, PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_))
            || matches!(e, PolyError::NotImplemented(_))
    };
    let (numerator, denominator) = numer_denom(&crate::polys::polytools::cancel(f, &[])?)?;
    let (mut result, remainder) = polynomial_part(&numerator, &denominator, x)?;
    if remainder == Basic::zero() {
        return Ok(result);
    }
    let (rational, logarithmic) = ratint_ratpart(&remainder, &denominator, x)?;
    result = add(&[result, rational])?;
    let (top, bottom) = numer_denom(&crate::polys::polytools::cancel(&logarithmic, &[])?)?;
    if top == Basic::zero() {
        return Ok(result);
    }
    let t: Basic = crate::core::symbol::Dummy::new("t", &[]).0;
    for (argument, resultant_poly) in ratint_logpart(&top, &bottom, x, Some(&t))? {
        let factors: Vec<Basic> = match crate::polys::polytools::factor_list(&resultant_poly, std::slice::from_ref(&t))
        {
            Ok((_, found)) => found.into_iter().map(|(g, _)| g).collect(),
            Err(e) if failure(&e) => vec![resultant_poly.clone()],
            Err(e) => return Err(e),
        };
        for factor in factors {
            let degree: i64 = crate::polys::rootoftools::polynomial_coefficients(&factor, &t)?.len() as i64 - 1;
            if degree <= 0 {
                continue;
            }
            let mut zeros = None;
            if degree <= 2 {
                zeros = match crate::polys::polyroots::roots(&factor, std::slice::from_ref(&t), true, true, None) {
                    Ok(found) => Some(found),
                    Err(e) if failure(&e) => None,
                    Err(e) => return Err(e),
                };
            }
            match zeros {
                Some(zeros) if zeros.values().sum::<usize>() as i64 == degree => {
                    for (a, _) in zeros.iter() {
                        result = add(&[
                            result,
                            mul(&[
                                a.clone(),
                                crate::functions::elementary::exponential::log(&crate::core::subs::subs(
                                    &argument, &t, a,
                                )?)?,
                            ])?,
                        ])?;
                    }
                }
                _ => {
                    let body: Basic = mul(&[t.clone(), crate::functions::elementary::exponential::log(&argument)?])?;
                    result = add(&[
                        result,
                        crate::polys::rootoftools::root_sum(&factor, Some((&t, &body)), Some(&t), true, false)?,
                    ])?;
                }
            }
        }
    }
    Ok(result)
}

fn ops(e: &Basic) -> i64 {
    crate::core::function::count_ops(e).unwrap_or(0)
}

/// `simplify(e)` within the time limit, except for the expressions with
/// integrals and `RootSum`s (`_simplified`).
fn simplified(e: &Basic) -> R {
    let has_root_sum = e.preorder().iter().any(|n| matches!(n.head(), Head::RootSum));
    if ops(e) >= 400 || has_root_sum || has_integral(e) {
        return Ok(e.clone());
    }
    let Some(s) = attempt(|| simplify(e))? else {
        return Ok(e.clone());
    };
    let bad = s
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(Function::ExpPolar) | Head::Piecewise));
    Ok(if bad { e.clone() } else { s })
}

/// The inverse of a square matrix over the field, `None` when singular
/// (`_field_inverse`).
fn field_inverse(field: &Field, rows: &[Vec<Element>]) -> Option<Vec<Vec<Element>>> {
    let n: usize = rows.len();
    let mut sparse: Vec<SparseRow<Element>> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let mut entries: SparseRow<Element> = row
            .iter()
            .enumerate()
            .filter(|(_, c)| !field.is_zero(c))
            .map(|(j, c)| (j, c.clone()))
            .collect();
        entries.insert(n + i, field.one());
        sparse.push(entries);
    }
    let echelon: Vec<SparseRow<Element>> = field.sparse_echelon(&sparse);
    let mut pivot_rows: std::collections::BTreeMap<usize, SparseRow<Element>> = std::collections::BTreeMap::new();
    for r in echelon {
        let first: usize = *r.keys().next().expect("a nonzero row");
        pivot_rows.insert(first, r);
    }
    let keys: Vec<usize> = pivot_rows.keys().copied().take(n).collect();
    if keys != (0..n).collect::<Vec<usize>>() {
        return None;
    }
    Some(
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| pivot_rows[&i].get(&(n + j)).cloned().unwrap_or_else(|| field.zero()))
                    .collect()
            })
            .collect(),
    )
}

/// The cyclic reduction computed in the native field of rational
/// functions (`_field_reduction`).
fn field_reduction(a: &Rows, x: &Basic) -> PolyResult<Option<CyclicReduction>> {
    let n: usize = a.len();
    let mut all: Rows = a.clone();
    all.push(vec![x.clone()]);
    let matrix_rows: RationalRows = match RationalRows::new(&all) {
        Ok(m) => m,
        Err(e) if field_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let field: &Field = &matrix_rows.field;
    let index: usize = matrix_rows
        .symbols
        .iter()
        .position(|s| s == x)
        .expect("x among the symbols");
    let renamed: Vec<(Basic, Basic)> = matrix_rows
        .restored
        .iter()
        .map(|(name, s)| (s.clone(), name.clone()))
        .collect();
    let entries: &[Vec<Element>] = &matrix_rows.rows[..n];
    let following = |c: &[Element]| -> Vec<Element> {
        (0..n)
            .map(|j| {
                let mut value: Element = field.diff(&c[j], index);
                for i in 0..n {
                    if !field.is_zero(&c[i]) && !field.is_zero(&entries[i][j]) {
                        value = field.add(&value, &field.mul(&c[i], &entries[i][j]));
                    }
                }
                value
            })
            .collect()
    };
    for c0 in candidates(n, x)? {
        let first: Vec<Element> = c0
            .iter()
            .map(|e| field.from_expr(&replaced(e, &renamed)?))
            .collect::<PolyResult<_>>()?;
        let mut rows: Vec<Vec<Element>> = vec![first];
        for _ in 0..n - 1 {
            let next: Vec<Element> = following(rows.last().expect("a row"));
            rows.push(next);
        }
        let Some(inv) = field_inverse(field, &rows) else {
            continue;
        };
        let last: Vec<Element> = following(rows.last().expect("a row"));
        // c_n = sum_k a_k c_k with (a_k) = c_n M**-1
        let mut coefficients: Vec<Element> = Vec::new();
        for k in 0..n {
            let mut value: Element = field.zero();
            for j in 0..n {
                if !field.is_zero(&last[j]) {
                    value = field.add(&value, &field.mul(&last[j], &inv[j][k]));
                }
            }
            coefficients.push(value);
        }
        let mut operator_coefficients: Vec<Basic> = coefficients
            .iter()
            .map(|e| normal(&neg(&matrix_rows.expr(e)?)?))
            .collect::<PolyResult<_>>()?;
        operator_coefficients.push(Basic::one());
        let operator: LinearOperator = LinearOperator::new(&operator_coefficients, x).primitive()?;
        let to_rows = |m: &[Vec<Element>]| -> PolyResult<Rows> {
            m.iter()
                .map(|row| row.iter().map(|e| normal(&matrix_rows.expr(e)?)).collect())
                .collect()
        };
        return Ok(Some(CyclicReduction::new(
            c0,
            &to_rows(&rows)?,
            to_rows(&inv)?,
            operator,
            x,
        )?));
    }
    Err(PolyError::Value(
        "no cyclic vector was found among the candidates".to_string(),
    ))
}

fn reduction(a: &Rows, x: &Basic) -> PolyResult<CyclicReduction> {
    if let Some(found) = field_reduction(a, x)? {
        return Ok(found);
    }
    let n: usize = a.len();
    for c0 in candidates(n, x)? {
        let mut rows: Rows = vec![c0.clone()];
        for _ in 0..n - 1 {
            let next: Vec<Basic> = next_row(a, rows.last().expect("a row"), x)?;
            rows.push(next);
        }
        let Some(inv) = inverse(&rows)? else {
            continue;
        };
        let following: Vec<Basic> = next_row(a, rows.last().expect("a row"), x)?;
        let mut coefficients: Vec<Basic> = Vec::new();
        for k in 0..n {
            let mut terms: Vec<Basic> = Vec::new();
            for j in 0..n {
                if following[j] != Basic::zero() {
                    terms.push(mul(&[following[j].clone(), inv[j][k].clone()])?);
                }
            }
            coefficients.push(normal(&sum(&terms)?)?);
        }
        let mut operator_coefficients: Vec<Basic> = coefficients
            .iter()
            .map(|e| normal(&neg(e)?))
            .collect::<PolyResult<_>>()?;
        operator_coefficients.push(Basic::one());
        let operator: LinearOperator = LinearOperator::new(&operator_coefficients, x).primitive()?;
        return CyclicReduction::new(c0, &rows, inv, operator, x);
    }
    Err(PolyError::Value(
        "no cyclic vector was found among the candidates".to_string(),
    ))
}

/// The scalar equation of a cyclic vector of `Y' = A Y`
/// (`cyclic_reduction`).
pub fn cyclic_reduction(a: &ExprMatrix, x: &Basic) -> PolyResult<CyclicReduction> {
    reduction(&entries(a, x)?, x)
}

/// A cyclic row vector and its scalar operator (`cyclic_vector`).
pub fn cyclic_vector(a: &ExprMatrix, x: &Basic) -> PolyResult<(ExprMatrix, LinearOperator)> {
    let r: CyclicReduction = cyclic_reduction(a, x)?;
    Ok((r.vector, r.operator))
}

/// The scalar operator of a cyclic vector and the matrix `M`
/// (`system_to_scalar`).
pub fn system_to_scalar(a: &ExprMatrix, x: &Basic) -> PolyResult<(LinearOperator, ExprMatrix)> {
    let r: CyclicReduction = cyclic_reduction(a, x)?;
    Ok((r.operator, r.matrix))
}

// ---------------------------------------------------------------------------
// rational solutions

/// The integer roots of a polynomial in `variable` (`_integer_roots`).
fn integer_roots(p: &Basic, variable: &Basic) -> PolyResult<Vec<i64>> {
    let mut found: Vec<i64> = Vec::new();
    for (f, _) in factor_list(p, variable)? {
        let q: UPoly = match UPoly::of(&f, variable) {
            Ok(q) => q,
            Err(PolyError::Value(_)) => continue,
            Err(e) => return Err(e),
        };
        if q.degree() != 1 {
            continue;
        }
        let c: Vec<Basic> = q.all_coeffs();
        let root: Basic = normal(&div(&neg(&c[1])?, &c[0])?)?;
        if let Some(r) = root.as_rational() {
            if crate::numbers::rat_is_integer(r) {
                if let Ok(value) = i64::try_from(&crate::numbers::numer(r)) {
                    found.push(value);
                }
            }
        }
    }
    found.sort_unstable();
    found.dedup();
    Ok(found)
}

/// The largest `k` with `p**k` dividing `q` (`_multiplicity`).
fn multiplicity(q: &UPoly, p: &UPoly) -> PolyResult<i64> {
    let mut q: UPoly = q.clone();
    let mut k: i64 = 0;
    while !q.is_zero() && q.degree() >= p.degree() {
        let (quotient, remainder) = q.div(p)?;
        if !remainder.is_zero() {
            break;
        }
        q = quotient;
        k += 1;
    }
    Ok(k)
}

/// `deg numerator - deg denominator`, `None` for zero (`_degree`).
fn degree(e: &Basic, x: &Basic) -> PolyResult<Option<i64>> {
    if *e == Basic::zero() {
        return Ok(None);
    }
    let (numerator, denominator) = numer_denom(&normal(e)?)?;
    Ok(Some(
        UPoly::of(&numerator, x)?.degree() - UPoly::of(&denominator, x)?.degree(),
    ))
}

/// The limit at infinity of a rational function of degree at most zero
/// (`_at_infinity`).
fn at_infinity(e: &Basic, x: &Basic) -> R {
    let (numerator, denominator) = numer_denom(&normal(e)?)?;
    let (top, bottom) = (UPoly::of(&numerator, x)?, UPoly::of(&denominator, x)?);
    if top.is_zero() || top.degree() < bottom.degree() {
        return Ok(Basic::zero());
    }
    if top.degree() > bottom.degree() {
        return Err(PolyError::Value(format!("{} has no finite limit at infinity", e)));
    }
    normal(&div(&top.lc(), &bottom.lc())?)
}

/// The bounds for the rational solutions from the scalar equation of a
/// cyclic vector, built when first needed (`_ScalarBounds`).
struct ScalarBounds<'a> {
    a: &'a Rows,
    x: &'a Basic,
    reduction: Option<CyclicReduction>,
}

impl ScalarBounds<'_> {
    fn reduction(&mut self) -> PolyResult<&CyclicReduction> {
        if self.reduction.is_none() {
            self.reduction = Some(reduction(self.a, self.x)?);
        }
        Ok(self.reduction.as_ref().expect("the reduction"))
    }

    /// A bound for the order of the pole at the roots of `p` (`pole`).
    fn pole(&mut self, p: &UPoly) -> PolyResult<i64> {
        let x: Basic = self.x.clone();
        let n: usize = self.a.len();
        let r: &CyclicReduction = self.reduction()?;
        let l: &LinearOperator = &r.operator;
        let mut m: i64 = 0;
        if multiplicity(&UPoly::of(l.leading_coefficient(), &x)?, p)? > 0 {
            let indicial: UPoly = indicial_at(l, p)?;
            let negative: Vec<i64> = integer_roots(&indicial.as_expr()?, &indicial.x)?
                .into_iter()
                .filter(|r| *r < 0)
                .collect();
            if let Some(first) = negative.first() {
                m = -first;
            }
        }
        let mut bound: i64 = 0;
        for i in 0..n {
            for k in 0..n {
                let e: &Basic = &r.inverse[i][k];
                if *e == Basic::zero() {
                    continue;
                }
                let denominator: UPoly = UPoly::of(&numer_denom(&normal(e)?)?.1, &x)?;
                let extra: i64 = if m > 0 { m + k as i64 } else { 0 };
                bound = bound.max(multiplicity(&denominator, p)? + extra);
            }
        }
        Ok(bound)
    }

    /// A bound for the degree of the components (`degree`).
    fn degree(&mut self) -> PolyResult<Option<i64>> {
        let x: Basic = self.x.clone();
        let n: usize = self.a.len();
        let r: &CyclicReduction = self.reduction()?;
        let indicial: UPoly = indicial_at_infinity(&r.operator)?;
        let exponents: Vec<i64> = integer_roots(&indicial.as_expr()?, &indicial.x)?;
        let Some(&delta) = exponents.last() else {
            return Ok(None);
        };
        let mut bound: Option<i64> = None;
        for i in 0..n {
            for k in 0..n {
                if let Some(d) = degree(&r.inverse[i][k], &x)? {
                    let value: i64 = d + delta - k as i64;
                    bound = Some(bound.map_or(value, |b| b.max(value)));
                }
            }
        }
        Ok(bound)
    }
}

fn determinant(rows: &Rows) -> R {
    expand(&crate::solvers::ode_cases::determinant(rows)?)
}

/// `det(lam scale I - rows)`, expanded (`_characteristic`).
fn characteristic(rows: &Rows, lam: &Basic, scale: &Basic) -> R {
    let n: usize = rows.len();
    let matrix_rows: Rows = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        sub(&mul(&[lam.clone(), scale.clone()])?, &rows[i][j])
                    } else {
                        neg(&rows[i][j])
                    }
                })
                .collect()
        })
        .collect::<PolyResult<_>>()?;
    determinant(&matrix_rows)
}

/// The pole order bound at a simple pole (`_residue_bound`).
fn residue_bound(nn: &Rows, d: &UPoly, p: &UPoly, x: &Basic) -> PolyResult<i64> {
    let lam: Basic = crate::core::symbol::Dummy::new("lambda", &[]).0;
    let q: UPoly = d.div(p)?.0;
    let scale: Basic = expand(&mul(&[p.diff()?.as_expr()?, q.as_expr()?])?)?;
    let chi: Basic = characteristic(nn, &lam, &scale)?;
    let eigen: Basic = if p.degree() == 1 {
        let c: Vec<Basic> = p.all_coeffs();
        expand(&replaced(&chi, &[(x.clone(), normal(&div(&neg(&c[1])?, &c[0])?)?)])?)?
    } else {
        resultant(&chi, &p.as_expr()?, x)?
    };
    let negative: Vec<i64> = integer_roots(&eigen, &lam)?.into_iter().filter(|r| *r < 0).collect();
    Ok(negative.first().map(|r| -r).unwrap_or(0))
}

/// A basis of the rational solutions of `Y' = A Y` (`_rational_basis`).
fn rational_basis(a: &Rows, x: &Basic) -> PolyResult<Rows> {
    let n: usize = a.len();
    let parts: Vec<Vec<(Basic, Basic)>> = a
        .iter()
        .map(|row| row.iter().map(|e| numer_denom(&normal(e)?)).collect())
        .collect::<PolyResult<_>>()?;
    let denominators: Vec<Basic> = parts.iter().flatten().map(|(_, den)| den.clone()).collect();
    let d_expr: Basic = lcm(&denominators)?;
    let d: UPoly = UPoly::of(&d_expr, x)?;
    let nn: Rows = parts
        .iter()
        .map(|row| {
            row.iter()
                .map(|(num, den)| normal(&div(&mul(&[num.clone(), d_expr.clone()])?, den)?))
                .collect()
        })
        .collect::<PolyResult<_>>()?;
    let mut scalar = ScalarBounds { a, x, reduction: None };
    // the universal denominator
    let mut big_d: Basic = Basic::one();
    let mut determinant_: Option<UPoly> = None;
    for (f, e) in factor_list(&d_expr, x)? {
        let p: UPoly = UPoly::of(&f, x)?;
        if p.degree() <= 0 {
            continue;
        }
        let m: i64 = if e == 1 {
            residue_bound(&nn, &d, &p, x)?
        } else {
            if determinant_.is_none() {
                determinant_ = Some(UPoly::of(&determinant(&nn)?, x)?);
            }
            let det: &UPoly = determinant_.as_ref().expect("the determinant");
            if !det.is_zero() && multiplicity(det, &p)? == 0 {
                // the leading matrix at the pole is invertible
                return Ok(Vec::new());
            }
            scalar.pole(&p)?
        };
        if m != 0 {
            big_d = mul(&[big_d, pow(&p.as_expr()?, &int(m))?])?;
        }
    }
    let big_d: Basic = expand(&big_d)?;
    // the degree at infinity: deg Y = deg P - deg D
    let mut orders: Vec<i64> = Vec::new();
    for row in a {
        for e in row {
            if let Some(k) = degree(e, x)? {
                orders.push(-k);
            }
        }
    }
    let shift: Option<i64> = orders.iter().copied().min();
    let delta: Option<i64> = match shift {
        // A = 0: the constant solutions
        None => Some(0),
        Some(s) if s >= 1 => {
            // A = O(1/x): the integer eigenvalues of lim x A
            let lam: Basic = crate::core::symbol::Dummy::new("lambda", &[]).0;
            let residue: Rows = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            if s == 1 {
                                at_infinity(&mul(&[x.clone(), a[i][j].clone()])?, x)
                            } else {
                                Ok(Basic::zero())
                            }
                        })
                        .collect()
                })
                .collect::<PolyResult<_>>()?;
            integer_roots(&characteristic(&residue, &lam, &Basic::one())?, &lam)?
                .last()
                .copied()
        }
        Some(s) => {
            // A = O(x**q), q = -shift >= 0: the leading matrix
            let q: i64 = -s;
            let leading: Rows = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| at_infinity(&div(&a[i][j], &pow(x, &int(q))?)?, x))
                        .collect()
                })
                .collect::<PolyResult<_>>()?;
            if normal(&determinant(&leading)?)? != Basic::zero() {
                return Ok(Vec::new());
            }
            scalar.degree()?
        }
    };
    let Some(delta) = delta else {
        return Ok(Vec::new());
    };
    let top: i64 = UPoly::of(&big_d, x)?.degree() + delta;
    if top < 0 {
        return Ok(Vec::new());
    }
    numerators(a, &nn, &d_expr, &big_d, top as usize, x)
}

/// The solutions `P/D` with `deg P <= top` (`_numerators`).
fn numerators(a: &Rows, nn: &Rows, d: &Basic, big_d: &Basic, top: usize, x: &Basic) -> PolyResult<Rows> {
    let n: usize = a.len();
    let d_d: Basic = expand(&mul(&[d.clone(), big_d.clone()])?)?;
    let d_dprime: Basic = expand(&mul(&[d.clone(), diff(big_d, x, 1)?])?)?;
    let mut columns: Vec<Vec<UPoly>> = Vec::new();
    for i in 0..n {
        for j in 0..=top {
            let monomial: Basic = if j > 0 { pow(x, &int(j as i64))? } else { Basic::one() };
            let mut image: Vec<UPoly> = Vec::new();
            for k in 0..n {
                let mut value: Basic = neg(&mul(&[big_d.clone(), nn[k][i].clone(), monomial.clone()])?)?;
                if k == i {
                    let first: Basic = if j > 0 {
                        mul(&[d_d.clone(), int(j as i64), pow(x, &int(j as i64 - 1))?])?
                    } else {
                        Basic::zero()
                    };
                    value = sub(&add(&[value, first])?, &mul(&[d_dprime.clone(), monomial.clone()])?)?;
                }
                image.push(UPoly::of(&expand(&value)?, x)?);
            }
            columns.push(image);
        }
    }
    let size: i64 = columns
        .iter()
        .flatten()
        .filter(|p| !p.is_zero())
        .map(|p| p.degree())
        .max()
        .unwrap_or(0)
        + 1;
    let mut rows: Rows = Vec::new();
    for k in 0..n {
        for power in 0..size {
            let row: Vec<Basic> = columns.iter().map(|column| column[k].coeff(power)).collect();
            if row.iter().any(|e| *e != Basic::zero()) {
                rows.push(row);
            }
        }
    }
    let mut basis: Rows = Vec::new();
    for vector in nullspace(&rows, columns.len())? {
        let mut entries: Vec<Basic> = Vec::new();
        for i in 0..n {
            let mut terms: Vec<Basic> = Vec::new();
            for j in 0..=top {
                let c: Basic = vector[i * (top + 1) + j].clone();
                terms.push(if j > 0 { mul(&[c, pow(x, &int(j as i64))?])? } else { c });
            }
            entries.push(normal(&div(&sum(&terms)?, big_d)?)?);
        }
        basis.push(entries);
    }
    Ok(basis)
}

/// A basis of the rational solutions of `Y' = A Y`
/// (`rational_system_solutions`).
pub fn rational_system_solutions(a: &ExprMatrix, x: &Basic) -> PolyResult<Vec<ExprMatrix>> {
    Ok(rational_basis(&entries(a, x)?, x)?
        .into_iter()
        .map(ExprMatrix::column)
        .collect())
}

fn augmented(a: &Rows, b: &[Basic]) -> PolyResult<Rows> {
    let n: usize = a.len();
    let mut rows: Rows = Vec::new();
    for i in 0..n {
        let mut row: Vec<Basic> = a[i].clone();
        row.push(normal(&b[i])?);
        rows.push(row);
    }
    rows.push(vec![Basic::zero(); n + 1]);
    Ok(rows)
}

/// A rational solution of `Y' = A Y + b` (`_rational_particular`).
fn rational_particular(a: &Rows, b: &[Basic], x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let n: usize = a.len();
    for vector in rational_basis(&augmented(a, b)?, x)? {
        let z: &Basic = &vector[n];
        if *z != Basic::zero() && !z.has(x) {
            return Ok(Some(
                (0..n)
                    .map(|i| normal(&div(&vector[i], z)?))
                    .collect::<PolyResult<_>>()?,
            ));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// hyperexponential and general solutions

/// Independent solutions `h R` of `Y' = A Y` with `h'/h` and `R` rational
/// (`hyperexponential_system_solutions`).
pub fn hyperexponential_system_solutions(a: &ExprMatrix, x: &Basic) -> PolyResult<Vec<ExprMatrix>> {
    let r: CyclicReduction = cyclic_reduction(a, x)?;
    let (found, _) = hyperexponential_search(&r.operator, 200)?;
    found.iter().map(|u| r.to_system(u)).collect()
}

/// Whether `Y' = A Y` (`_verified`).
fn verified(a: &Rows, y: &[Basic], x: &Basic) -> PolyResult<bool> {
    let n: usize = a.len();
    for i in 0..n {
        let terms: Vec<Basic> = (0..n)
            .map(|j| mul(&[a[i][j].clone(), y[j].clone()]))
            .collect::<PolyResult<_>>()?;
        let residual: Basic = sub(&diff(&y[i], x, 1)?, &sum(&terms)?)?;
        if attempt(|| simplify(&residual))? != Some(Basic::zero()) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn is_constant(a: &Rows, x: &Basic) -> bool {
    a.iter().flatten().all(|e| !e.has(x))
}

/// The columns of `exp(A x)` for a constant matrix
/// (`_exponential_columns`).
fn exponential_columns(a: &Rows, x: &Basic) -> PolyResult<Rows> {
    let n: usize = a.len();
    let phi: ExprMatrix = match crate::solvers::dae::matrix_exponential(&matrix(a)?, x) {
        Ok(phi) => phi,
        Err(e) if is_genuine_not_implemented(&e) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    Ok((0..n).map(|j| (0..n).map(|i| phi.at(i, j).clone()).collect()).collect())
}

/// The fundamental solutions `exp(M g(x))` when `A = g'(x) M`
/// (`_commuting_columns`).
fn commuting_columns(a: &Rows, x: &Basic) -> PolyResult<Rows> {
    let n: usize = a.len();
    let pivots: Vec<&Basic> = a.iter().flatten().filter(|e| **e != Basic::zero()).collect();
    if pivots.is_empty() || pivots.iter().all(|e| !e.has(x)) {
        return Ok(Vec::new());
    }
    let pivot: Basic = pivots[0].clone();
    let m: Rows = a
        .iter()
        .map(|row| row.iter().map(|e| normal(&div(e, &pivot)?)).collect())
        .collect::<PolyResult<_>>()?;
    if m.iter().flatten().any(|e| e.has(x)) {
        return Ok(Vec::new());
    }
    let Some(g) = crate::integrals::indefinite::closed_antiderivative(&pivot, x, false, true, None)? else {
        return Ok(Vec::new());
    };
    let s: Basic = crate::core::symbol::Dummy::new("s", &[]).0;
    let phi: ExprMatrix = match crate::solvers::dae::matrix_exponential(&matrix(&m)?, &s) {
        Ok(phi) => phi,
        Err(e) if is_genuine_not_implemented(&e) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut columns: Rows = Vec::new();
    for j in 0..n {
        let column: Vec<Basic> = (0..n)
            .map(|i| simplified(&replaced(phi.at(i, j), &[(s.clone(), g.clone())])?))
            .collect::<PolyResult<_>>()?;
        if verified(a, &column, x)? {
            columns.push(column);
        }
    }
    Ok(if columns.len() == n { columns } else { Vec::new() })
}

/// `det Phi` as `C exp(Integral(trace A))` when the ratio is constant
/// (`_liouville`).
fn liouville(a: &Rows, determinant_: &Basic, x: &Basic) -> R {
    let trace: Basic = normal(&sum(&(0..a.len()).map(|i| a[i][i].clone()).collect::<Vec<_>>())?)?;
    let integral: Option<Basic> =
        attempt(|| crate::integrals::indefinite::closed_antiderivative(&trace, x, false, true, None))?.flatten();
    if let Some(integral) = integral {
        let abel: Basic = crate::functions::elementary::exponential::exp(&integral)?;
        let ratio: Option<Basic> = attempt(|| simplify(&div(determinant_, &abel)?))?;
        if let Some(ratio) = ratio {
            if ratio != Basic::zero() && !ratio.has(x) {
                return mul(&[ratio, abel]);
            }
        }
    }
    Ok(attempt(|| simplify(determinant_))?.unwrap_or_else(|| determinant_.clone()))
}

/// `Phi Integral(Phi**-1 b)` by Cramer's rule (`_variation_of_constants`).
fn variation_of_constants(a: &Rows, fundamental: &ExprMatrix, b: &[Basic], x: &Basic) -> PolyResult<Vec<Basic>> {
    let n: usize = a.len();
    let rows: Rows = (0..n).map(|i| fundamental.row(i).to_vec()).collect();
    let w: Basic = liouville(a, &crate::solvers::ode_cases::determinant(&rows)?, x)?;
    if w == Basic::zero() {
        return Err(PolyError::Value("the fundamental matrix is singular".to_string()));
    }
    let mut total: Vec<Basic> = vec![Basic::zero(); n];
    for j in 0..n {
        let replaced_rows: Rows = (0..n)
            .map(|i| {
                (0..n)
                    .map(|k| if k == j { b[i].clone() } else { rows[i][k].clone() })
                    .collect()
            })
            .collect();
        let integrand: Basic = div(&crate::solvers::ode_cases::determinant(&replaced_rows)?, &w)?;
        let c: Basic = if is_rational_function(&integrand, x) {
            rational_integral(&integrand, x)?
        } else if has_integral(&integrand) {
            crate::integrals::integral::integral(&integrand, x, None)?
        } else {
            crate::solvers::factorization::quadrature(&integrand, x)?
        };
        for i in 0..n {
            total[i] = add(&[total[i].clone(), mul(&[rows[i][j].clone(), c.clone()])?])?;
        }
    }
    total.iter().map(simplified).collect()
}

/// The solutions of `Y' = A Y + b` found by [`dsolve_system`]
/// (`LinearSystemSolution`).
#[derive(Clone, Debug)]
pub struct LinearSystemSolution {
    pub fundamental: ExprMatrix,
    pub particular: Option<ExprMatrix>,
    pub complete: Option<bool>,
    pub reduction: CyclicReduction,
}

impl LinearSystemSolution {
    /// The number of independent homogeneous solutions found (`rank`).
    pub fn rank(&self) -> usize {
        self.fundamental.cols
    }

    /// The independent homogeneous solutions (`columns`).
    pub fn columns(&self) -> Vec<ExprMatrix> {
        let f: &ExprMatrix = &self.fundamental;
        (0..self.rank())
            .map(|j| ExprMatrix::column((0..f.rows).map(|i| f.at(i, j).clone()).collect()))
            .collect()
    }

    /// `fundamental * C + particular` (`general_solution`).
    pub fn general_solution(&self, constants: &[Basic]) -> PolyResult<ExprMatrix> {
        if constants.len() != self.rank() {
            return Err(PolyError::Value(format!("{} constants are needed", self.rank())));
        }
        let f: &ExprMatrix = &self.fundamental;
        let mut entries: Vec<Basic> = (0..f.rows)
            .map(|i| {
                let terms: Vec<Basic> = (0..self.rank())
                    .map(|j| mul(&[constants[j].clone(), f.at(i, j).clone()]))
                    .collect::<PolyResult<_>>()?;
                sum(&terms)
            })
            .collect::<PolyResult<_>>()?;
        if let Some(p) = &self.particular {
            entries = entries
                .iter()
                .zip(&p.entries)
                .map(|(e, q)| add(&[e.clone(), q.clone()]))
                .collect::<PolyResult<_>>()?;
        }
        Ok(ExprMatrix::column(entries))
    }
}

/// The solutions of `Y' = A Y + b` with `A` rational in `x`
/// (`dsolve_system`).
pub fn dsolve_system(a: &ExprMatrix, x: &Basic, b: Option<&ExprMatrix>) -> PolyResult<LinearSystemSolution> {
    let rows: Rows = entries(a, x)?;
    let n: usize = rows.len();
    let mut vector: Option<Vec<Basic>> = None;
    if let Some(b) = b {
        let v: Vec<Basic> = column(b, n)?;
        if !v.iter().all(|e| *e == Basic::zero()) {
            vector = Some(v);
        }
    }
    let r: CyclicReduction = reduction(&rows, x)?;
    let y: Basic = Arc::new(UndefinedFunction::new("y", None, &[])).apply(vec![x.clone()])?;
    let scalar: Vec<Basic> = solve_operator(&r.operator, &y, true, true, true)?;
    let mut columns: Vec<ExprMatrix> = scalar.iter().map(|u| r.to_system(u)).collect::<PolyResult<_>>()?;
    if columns.len() < n {
        let mut found: Rows = if is_constant(&rows, x) {
            exponential_columns(&rows, x)?
        } else {
            Vec::new()
        };
        if found.len() < n {
            found = commuting_columns(&rows, x)?;
        }
        if found.len() > columns.len() {
            columns = found.into_iter().map(ExprMatrix::column).collect();
        }
    }
    let fundamental: ExprMatrix = columns_matrix(&columns, n)?;
    let mut particular: Option<ExprMatrix> = Some(ExprMatrix::zeros(n, 1));
    if let Some(v) = &vector {
        let rational: Option<Vec<Basic>> = if v.iter().all(|e| is_rational_function(e, x)) {
            rational_particular(&rows, v, x)?
        } else {
            None
        };
        particular = if let Some(rational) = rational {
            Some(ExprMatrix::column(rational))
        } else if columns.len() == n {
            Some(ExprMatrix::column(variation_of_constants(&rows, &fundamental, v, x)?))
        } else {
            None
        };
    }
    let complete: Option<bool> = if columns.len() == n && particular.is_some() {
        Some(true)
    } else {
        None
    };
    Ok(LinearSystemSolution {
        fundamental,
        particular,
        complete,
        reduction: r,
    })
}

/// Independent solution vectors of `Y' = A Y` (`dsolve_linear_system`).
pub fn dsolve_linear_system(a: &ExprMatrix, x: &Basic) -> PolyResult<Vec<ExprMatrix>> {
    Ok(dsolve_system(a, x, None)?.columns())
}
