//! Port of `sympy_extras/solvers/dae.py`: linear differential-algebraic
//! systems `A y' + B y = f(x)` with constant coefficients through the
//! core-nilpotent decomposition of the regular pencil, and the matrix
//! exponential by Putzer's algorithm.

#![allow(clippy::mutable_key_type)]

use crate::core::basic::{Basic, Head, RelOp};
use crate::error::{PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::solvers::charpit::{power_antiderivative, replaced};

type R = PolyResult<Basic>;
type M = PolyResult<ExprMatrix>;

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
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn value_error(message: &str) -> PolyError {
    PolyError::Value(message.to_string())
}

// ---------------------------------------------------------------------------
// exact linear algebra on matrices of expressions

/// `e` in lowest terms (`_normal`).
fn normal(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}

/// Whether `e` is zero (`_is_zero`).
fn is_zero(e: &Basic) -> PolyResult<bool> {
    let c: Basic = normal(e)?;
    if c == Basic::zero() {
        return Ok(true);
    }
    let symbols: Vec<Basic> = c.free_symbols().into_iter().collect();
    if crate::core::expr::polynomial_like(&c, &symbols, crate::core::expr::PolynomialKind::Rational) == Some(true) {
        return Ok(false);
    }
    Ok(simplify(&c)? == Basic::zero())
}

/// The submatrix of the rows and columns of the ranges (`_block`).
fn block(m: &ExprMatrix, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> M {
    let (r, c) = (rows.len(), cols.len());
    let mut entries: Vec<Basic> = Vec::new();
    for i in rows {
        for j in cols.clone() {
            entries.push(m.at(i, j).clone());
        }
    }
    ExprMatrix::new(r, c, entries)
}

/// The columns side by side (`_hstack`).
fn hstack(columns: &[ExprMatrix], rows: usize) -> M {
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..rows {
        for c in columns {
            entries.push(c.at(i, 0).clone());
        }
    }
    ExprMatrix::new(rows, columns.len(), entries)
}

fn vstack(top: &ExprMatrix, bottom: &ExprMatrix) -> M {
    let cols: usize = if top.rows > 0 { top.cols } else { bottom.cols };
    let mut entries: Vec<Basic> = top.entries.clone();
    entries.extend(bottom.entries.iter().cloned());
    ExprMatrix::new(top.rows + bottom.rows, cols, entries)
}

/// `A*B` with the entries in lowest terms (`_product`).
fn product(a: &ExprMatrix, b: &ExprMatrix) -> M {
    a.matmul(b)?.applyfunc(&normal)
}

/// The reduced row echelon form and its pivot columns (`_rref`).
fn rref(m: &ExprMatrix) -> PolyResult<(ExprMatrix, Vec<usize>)> {
    let mut rows: Vec<Vec<Basic>> = (0..m.rows).map(|i| m.row(i).to_vec()).collect();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r: usize = 0;
    for j in 0..m.cols {
        if r == rows.len() {
            break;
        }
        let mut pivot: Option<usize> = None;
        for (i, row) in rows.iter().enumerate().skip(r) {
            if !is_zero(&row[j])? {
                pivot = Some(i);
                break;
            }
        }
        let Some(pivot) = pivot else {
            continue;
        };
        rows.swap(r, pivot);
        let inverse: Basic = pow(&rows[r][j], &int(-1))?;
        rows[r] = rows[r]
            .iter()
            .map(|e| normal(&mul(&[e.clone(), inverse.clone()])?))
            .collect::<PolyResult<_>>()?;
        for i in 0..rows.len() {
            if i != r && !is_zero(&rows[i][j])? {
                let factor: Basic = rows[i][j].clone();
                let reduced: Vec<Basic> = rows[i]
                    .iter()
                    .zip(&rows[r])
                    .map(|(a, b)| normal(&sub(a, &mul(&[factor.clone(), b.clone()])?)?))
                    .collect::<PolyResult<_>>()?;
                rows[i] = reduced;
            }
        }
        pivots.push(j);
        r += 1;
    }
    let mut entries: Vec<Basic> = Vec::new();
    for row in &rows {
        for e in row {
            entries.push(cleaned(e)?);
        }
    }
    Ok((ExprMatrix::new(m.rows, m.cols, entries)?, pivots))
}

fn cleaned(e: &Basic) -> R {
    Ok(if is_zero(e)? { Basic::zero() } else { e.clone() })
}

fn rank(m: &ExprMatrix) -> PolyResult<usize> {
    Ok(rref(m)?.1.len())
}

/// The columns at the pivots of the echelon form (`_columnspace`).
fn columnspace(m: &ExprMatrix) -> PolyResult<Vec<ExprMatrix>> {
    let (_, pivots) = rref(m)?;
    pivots.iter().map(|&j| block(m, 0..m.rows, j..j + 1)).collect()
}

/// A basis of the null space (`_nullspace`).
fn nullspace(m: &ExprMatrix) -> PolyResult<Vec<ExprMatrix>> {
    let (r, pivots) = rref(m)?;
    let mut basis: Vec<ExprMatrix> = Vec::new();
    for free in (0..m.cols).filter(|j| !pivots.contains(j)) {
        let mut entries: Vec<Basic> = vec![Basic::zero(); m.cols];
        entries[free] = Basic::one();
        for (i, &j) in pivots.iter().enumerate() {
            entries[j] = neg(r.at(i, free))?;
        }
        basis.push(ExprMatrix::column(entries));
    }
    Ok(basis)
}

/// The inverse by Gauss-Jordan elimination (`_inverse`).
fn inverse(m: &ExprMatrix) -> M {
    let n: usize = m.rows;
    let mut entries: Vec<Basic> = Vec::new();
    for i in 0..n {
        entries.extend(m.row(i).iter().cloned());
        entries.extend((0..n).map(|j| if i == j { Basic::one() } else { Basic::zero() }));
    }
    let (r, pivots) = rref(&ExprMatrix::new(n, 2 * n, entries)?)?;
    if pivots.len() < n || pivots[..n] != (0..n).collect::<Vec<usize>>()[..] {
        return Err(value_error("the matrix is not invertible"));
    }
    block(&r, 0..n, n..2 * n)
}

/// The determinant by Gaussian elimination (`_determinant`).
fn determinant(m: &ExprMatrix) -> R {
    let mut rows: Vec<Vec<Basic>> = (0..m.rows).map(|i| m.row(i).to_vec()).collect();
    let n: usize = rows.len();
    let mut result: Basic = Basic::one();
    for j in 0..n {
        let mut pivot: Option<usize> = None;
        for (i, row) in rows.iter().enumerate().skip(j) {
            if !is_zero(&row[j])? {
                pivot = Some(i);
                break;
            }
        }
        let Some(pivot) = pivot else {
            return Ok(Basic::zero());
        };
        if pivot != j {
            rows.swap(j, pivot);
            result = neg(&result)?;
        }
        result = normal(&mul(&[result, rows[j][j].clone()])?)?;
        let inv: Basic = pow(&rows[j][j], &int(-1))?;
        for i in j + 1..n {
            let factor: Basic = normal(&mul(&[rows[i][j].clone(), inv.clone()])?)?;
            let reduced: Vec<Basic> = rows[i]
                .iter()
                .zip(&rows[j])
                .map(|(a, b)| normal(&sub(a, &mul(&[factor.clone(), b.clone()])?)?))
                .collect::<PolyResult<_>>()?;
            rows[i] = reduced;
        }
    }
    Ok(result)
}

fn power(m: &ExprMatrix, k: usize) -> M {
    let mut result: ExprMatrix = ExprMatrix::eye(m.rows);
    for _ in 0..k {
        result = product(&result, m)?;
    }
    Ok(result)
}

/// `det(t I - M)` by the Faddeev-LeVerrier recurrence (`_charpoly`).
fn charpoly(m: &ExprMatrix, t: &Basic) -> R {
    let n: usize = m.rows;
    let mut coefficients: Vec<Basic> = vec![Basic::one()];
    let mut current: ExprMatrix = ExprMatrix::zeros(n, n);
    for k in 1..=n {
        let last: Basic = coefficients.last().expect("a coefficient").clone();
        current = product(m, &current.add(&ExprMatrix::eye(n).scale(&last)?)?)?;
        let trace: Basic = add(&(0..n).map(|i| current.at(i, i).clone()).collect::<Vec<_>>())?;
        coefficients.push(normal(&crate::core::mul::div(&neg(&trace)?, &int(k as i64))?)?);
    }
    let terms: Vec<Basic> = coefficients
        .iter()
        .enumerate()
        .map(|(i, c)| mul(&[c.clone(), pow(t, &int((n - i) as i64))?]))
        .collect::<PolyResult<_>>()?;
    add(&terms)
}

/// The eigenvalues repeated by their multiplicities (`_eigenvalues`).
fn eigenvalues(m: &ExprMatrix) -> PolyResult<Vec<Basic>> {
    let t: Basic = crate::core::symbol::Dummy::new("t", &[]).0;
    let (numerator, _) =
        crate::simplify::radsimp::fraction(&crate::polys::rationaltools::together(&charpoly(m, &t)?, false, true)?)?;
    let found = crate::polys::polyroots::roots(&numerator, std::slice::from_ref(&t), true, true, None)?;
    if found.values().sum::<usize>() != m.rows {
        return Err(PolyError::NotImplemented(format!(
            "the eigenvalues of {} are not found",
            m.repr()
        )));
    }
    let mut values: Vec<Basic> = Vec::new();
    for value in crate::core::sorting::ordered(found.keys().cloned().collect()) {
        let count: usize = *found.get(&value).expect("a root");
        values.extend(std::iter::repeat(value).take(count));
    }
    Ok(values)
}

/// A sum of polynomials times exponentials (`ExponentialPolynomial`): the
/// exponents in their order of insertion with the coefficients of the
/// powers `x**j`, lowest first.
type ExponentialPolynomial = Vec<(Basic, Vec<Basic>)>;

fn add_term(target: &mut ExponentialPolynomial, mu: &Basic, j: usize, c: &Basic) -> PolyResult<()> {
    let index: usize = match target.iter().position(|(m, _)| m == mu) {
        Some(i) => i,
        None => {
            target.push((mu.clone(), Vec::new()));
            target.len() - 1
        }
    };
    let coefficients: &mut Vec<Basic> = &mut target[index].1;
    while coefficients.len() <= j {
        coefficients.push(Basic::zero());
    }
    coefficients[j] = normal(&add(&[coefficients[j].clone(), c.clone()])?)?;
    Ok(())
}

fn factorial(n: usize) -> crate::numbers::Int {
    (1..=n).fold(crate::numbers::Int::from(1), |acc, k| {
        acc * crate::numbers::Int::from(k)
    })
}

/// `int_0^x exp(lam (x - s)) r(s) ds` (`_convolution`).
fn convolution(r: &ExponentialPolynomial, lam: &Basic) -> PolyResult<ExponentialPolynomial> {
    let mut result: ExponentialPolynomial = Vec::new();
    for (mu, coefficients) in r {
        let d: Basic = normal(&sub(mu, lam)?)?;
        for (j, c) in coefficients.iter().enumerate() {
            if is_zero(c)? {
                continue;
            }
            if is_zero(&d)? {
                add_term(&mut result, lam, j + 1, &crate::core::mul::div(c, &int(j as i64 + 1))?)?;
                continue;
            }
            for i in 0..=j {
                let sign: crate::numbers::Int = if i % 2 == 0 { 1.into() } else { (-1).into() };
                let coefficient: Basic = Basic::integer(sign * (factorial(j) / factorial(j - i)));
                add_term(
                    &mut result,
                    mu,
                    j - i,
                    &mul(&[c.clone(), coefficient, pow(&d, &int(-(i as i64 + 1)))?])?,
                )?;
            }
            let sign: crate::numbers::Int = if j % 2 == 0 { 1.into() } else { (-1).into() };
            let last: Basic = mul(&[
                neg(c)?,
                Basic::integer(sign * factorial(j)),
                pow(&d, &int(-(j as i64 + 1)))?,
            ])?;
            add_term(&mut result, lam, 0, &last)?;
        }
    }
    Ok(result)
}

/// `exp(M*x)` for a constant square matrix by Putzer's algorithm
/// (`matrix_exponential`).
pub fn matrix_exponential(m: &ExprMatrix, x: &Basic) -> M {
    let n: usize = m.rows;
    if m.cols != n {
        return Err(PolyError::Value(format!(
            "a square matrix is expected, got shape ({}, {})",
            m.rows, m.cols
        )));
    }
    if n == 0 {
        return Ok(ExprMatrix::zeros(0, 0));
    }
    let values: Vec<Basic> = eigenvalues(m)?;
    let mut r: ExponentialPolynomial = vec![(values[0].clone(), vec![Basic::one()])];
    let mut p: ExprMatrix = ExprMatrix::eye(n);
    let mut entries: Vec<ExponentialPolynomial> = vec![Vec::new(); n * n];
    for k in 0..n {
        if k > 0 {
            p = product(&p, &m.sub(&ExprMatrix::eye(n).scale(&values[k - 1])?)?)?;
            r = convolution(&r, &values[k])?;
        }
        for (index, entry) in p.entries.iter().enumerate() {
            if is_zero(entry)? {
                continue;
            }
            for (mu, coefficients) in &r {
                for (j, c) in coefficients.iter().enumerate() {
                    add_term(&mut entries[index], mu, j, &mul(&[c.clone(), entry.clone()])?)?;
                }
            }
        }
    }
    let mut result: Vec<Basic> = Vec::new();
    for entry in &entries {
        let mut terms: Vec<Basic> = Vec::new();
        for (mu, coefficients) in entry {
            let polynomial: Basic = add(&coefficients
                .iter()
                .enumerate()
                .map(|(j, c)| mul(&[c.clone(), pow(x, &int(j as i64))?]))
                .collect::<PolyResult<Vec<_>>>()?)?;
            if !is_zero(&polynomial)? {
                terms.push(mul(&[
                    polynomial,
                    crate::functions::elementary::exponential::exp(&mul(&[mu.clone(), x.clone()])?)?,
                ])?);
            }
        }
        result.push(add(&terms)?);
    }
    ExprMatrix::new(n, n, result)
}

// ---------------------------------------------------------------------------
// the decomposition and the solver

/// The smallest `k` with `rank(M**k) == rank(M**(k + 1))`
/// (`_index_of_nilpotency`).
fn index_of_nilpotency(m: &ExprMatrix) -> PolyResult<usize> {
    let n: usize = m.rows;
    let mut p: ExprMatrix = ExprMatrix::eye(n);
    let mut previous: usize = n;
    for k in 0..=n {
        let current: usize = rank(&p)?;
        if current == previous && k > 0 {
            return Ok(k - 1);
        }
        previous = current;
        p = product(&p, m)?;
    }
    Ok(n)
}

/// The core-nilpotent (Drazin) decomposition `(T, C, N, k)`
/// (`core_nilpotent_decomposition`).
pub fn core_nilpotent_decomposition(m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix, ExprMatrix, usize)> {
    if m.rows != m.cols {
        return Err(PolyError::Value(format!(
            "a square matrix is expected, got shape ({}, {})",
            m.rows, m.cols
        )));
    }
    let n: usize = m.rows;
    let k: usize = index_of_nilpotency(m)?;
    let p: ExprMatrix = power(m, k)?;
    let core_columns: Vec<ExprMatrix> = columnspace(&p)?;
    let null_columns: Vec<ExprMatrix> = nullspace(&p)?;
    if core_columns.len() + null_columns.len() != n {
        return Err(value_error(
            "the range and the null space of M**k do not span the space",
        ));
    }
    let r: usize = core_columns.len();
    let columns: Vec<ExprMatrix> = core_columns.into_iter().chain(null_columns).collect();
    let t: ExprMatrix = hstack(&columns, n)?;
    let b: ExprMatrix = product(&product(&inverse(&t)?, m)?, &t)?;
    let c: ExprMatrix = block(&b, 0..r, 0..r)?;
    let nn: ExprMatrix = block(&b, r..n, r..n)?;
    Ok((t, c, nn, k))
}

/// An integer `lambda` with `det(lambda*A + B) != 0` (`_regular_lambda`).
fn regular_lambda(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<Option<Basic>> {
    for value in 0..=a.rows {
        let candidate: Basic = int(value as i64);
        if !is_zero(&determinant(&a.scale(&candidate)?.add(b)?)?)? {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// The index of the regular pencil `lambda*A + B` (`dae_index`).
pub fn dae_index(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<usize> {
    let Some(value) = regular_lambda(a, b)? else {
        return Err(value_error("the pencil lambda*A + B is singular"));
    };
    index_of_nilpotency(&product(&inverse(&a.scale(&value)?.add(b)?)?, a)?)
}

/// The general solution of `A y' + B y = f` (`DAESolution`).
#[derive(Clone, Debug)]
pub struct DaeSolution {
    pub solution: ExprMatrix,
    pub constants: Vec<Basic>,
    pub index: usize,
}

impl DaeSolution {
    /// (`__repr__`)
    pub fn repr(&self) -> String {
        let constants: Vec<String> = self.constants.iter().map(|c| c.to_string()).collect();
        format!(
            "DAESolution({}, constants=[{}], index={})",
            self.solution.repr(),
            constants.join(", "),
            self.index
        )
    }
}

/// An antiderivative: term by term for a sum of powers, else the native
/// one, else the unevaluated integral (`_integrate`).
fn integrate(e: &Basic, x: &Basic) -> R {
    if let Some(native) = power_antiderivative(e, x)? {
        return Ok(native);
    }
    match crate::integrals::indefinite::closed_antiderivative(e, x, true, false, None)? {
        Some(found) => Ok(found),
        None => crate::integrals::integral::integral(e, x, None),
    }
}

fn rewrite(e: &Basic, target: &str) -> R {
    crate::core::rewriting::rewrite(e, &[], target, &crate::core::rewriting::RewriteHints::default())
}

/// `exp(M*x)` with sines and cosines for a real matrix
/// (`_real_exponential`).
fn real_exponential(m: &ExprMatrix, x: &Basic) -> M {
    let result: ExprMatrix = matrix_exponential(m, x)?;
    let mut real: bool = true;
    for e in &m.entries {
        let product: Basic = mul(&[e.clone(), x.clone()])?;
        if crate::core::expr::is(&product, crate::core::assumptions::Predicate::extended_real) == Some(false) {
            real = false;
            break;
        }
    }
    if real {
        return result.applyfunc(&|e| simplify(&rewrite(e, "cos")?));
    }
    Ok(result)
}

fn constants(count: usize, taken: &[Basic]) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    let mut i: usize = 1;
    while result.len() < count {
        let c: Basic = crate::core::symbol::Symbol::new(&format!("C{}", i), &[]).0;
        if !taken.contains(&c) {
            result.push(c);
        }
        i += 1;
    }
    result
}

/// The general solution of `A y' + B y = f(x)` with constant square
/// matrices (`dsolve_dae`).
pub fn dsolve_dae(a: &ExprMatrix, b: &ExprMatrix, f: Option<&ExprMatrix>, x: &Basic) -> PolyResult<DaeSolution> {
    let n: usize = a.rows;
    if a.shape() != (n, n) || b.shape() != (n, n) {
        return Err(value_error("square matrices of the same size are expected"));
    }
    if let Some(f) = f {
        if f.shape() != (n, 1) {
            return Err(PolyError::Value(format!(
                "the right-hand side must be a column of length {}",
                n
            )));
        }
    }
    if a.entries.iter().chain(&b.entries).any(|e| e.free_symbols().contains(x)) {
        return Err(value_error("the coefficients must be constant"));
    }
    let Some(value) = regular_lambda(a, b)? else {
        return Err(value_error("the pencil lambda*A + B is singular"));
    };
    let p: ExprMatrix = inverse(&a.scale(&value)?.add(b)?)?;
    let a_hat: ExprMatrix = product(&p, a)?;
    let f_hat: ExprMatrix = match f {
        Some(f) => p.matmul(f)?,
        None => ExprMatrix::zeros(n, 1),
    };
    let (t, c, nn, k) = core_nilpotent_decomposition(&a_hat)?;
    let r: usize = c.rows;
    let g: ExprMatrix = inverse(&t)?.matmul(&f_hat)?;
    let (g1, g2) = (block(&g, 0..r, 0..1)?, block(&g, r..n, 0..1)?);
    let mut taken: Vec<Basic> = Vec::new();
    if f.is_some() {
        for e in &f_hat.entries {
            for s in e.free_symbols() {
                if !taken.contains(&s) {
                    taken.push(s);
                }
            }
        }
    }
    let cs: Vec<Basic> = constants(r, &taken);
    // regular part: z1' = M z1 + h with M = -C**-1 (I - lambda C)
    let mut z1: ExprMatrix = ExprMatrix::zeros(0, 1);
    if r > 0 {
        let c_inv: ExprMatrix = inverse(&c)?;
        let m: ExprMatrix = product(&c_inv, &ExprMatrix::eye(r).sub(&c.scale(&value)?)?)?.neg()?;
        let h: ExprMatrix = c_inv.matmul(&g1)?;
        let phi: ExprMatrix = real_exponential(&m, x)?;
        let phi_inv: ExprMatrix = real_exponential(&m.neg()?, x)?;
        let particular: ExprMatrix = phi_inv.matmul(&h)?;
        let integrals: ExprMatrix = ExprMatrix::column(
            particular
                .entries
                .iter()
                .map(|e| integrate(e, x))
                .collect::<PolyResult<_>>()?,
        );
        z1 = phi
            .matmul(&ExprMatrix::column(cs.clone()).add(&integrals)?)?
            .applyfunc(&simplify)?;
    }
    // nilpotent part: z2 = sum_i (-Q N)**i Q g2**(i)
    let m: usize = n - r;
    let mut z2: ExprMatrix = ExprMatrix::zeros(m, 1);
    if m > 0 {
        let q: ExprMatrix = inverse(&ExprMatrix::eye(m).sub(&nn.scale(&value)?)?)?;
        let mut step: ExprMatrix = ExprMatrix::eye(m);
        let mut derivative: ExprMatrix = g2;
        for _ in 0..k {
            z2 = z2.add(&step.matmul(&q)?.matmul(&derivative)?)?;
            step = product(&step, &product(&q, &nn)?.neg()?)?;
            derivative = derivative.diff(x)?;
        }
        z2 = z2.applyfunc(&simplify)?;
    }
    let z: ExprMatrix = vstack(&z1, &z2)?;
    let y: ExprMatrix = t.matmul(&z)?.applyfunc(&simplify)?;
    Ok(DaeSolution {
        solution: y,
        constants: cs,
        index: k,
    })
}

fn row(equation: &Basic) -> R {
    if matches!(equation.head(), Head::Relational(RelOp::Eq)) {
        return sub(&equation.args()[0], &equation.args()[1]);
    }
    if crate::core::logic::is_boolean(equation) || matches!(equation.head(), Head::Tuple) {
        return Err(PolyError::Type(format!(
            "an equation or an expression is expected, got {}",
            equation
        )));
    }
    Ok(equation.clone())
}

/// The matrices `A`, `B`, the column `f` and `x` of a linear system with
/// constant coefficients (`dae_matrices`).
pub fn dae_matrices(
    equations: &[Basic],
    functions: &[Basic],
) -> PolyResult<(ExprMatrix, ExprMatrix, ExprMatrix, Basic)> {
    if functions.is_empty() {
        return Err(value_error("no unknown function"));
    }
    let x: Basic = match functions[0].args().first() {
        Some(x) if x.is_symbol() => x.clone(),
        _ => return Err(value_error("the functions must depend on a symbol")),
    };
    let derivatives: Vec<Basic> = (0..functions.len())
        .map(|i| crate::core::symbol::Dummy::new(&format!("d{}", i), &[]).0)
        .collect();
    let values: Vec<Basic> = (0..functions.len())
        .map(|i| crate::core::symbol::Dummy::new(&format!("u{}", i), &[]).0)
        .collect();
    let mut first: Vec<(Basic, Basic)> = Vec::new();
    for (f, d) in functions.iter().zip(&derivatives) {
        first.push((crate::core::function::diff(f, &[(x.clone(), 1)])?, d.clone()));
    }
    let functions_rule: Vec<(Basic, Basic)> = functions.iter().cloned().zip(values.iter().cloned()).collect();
    let unknowns: Vec<Basic> = derivatives.iter().chain(&values).cloned().collect();
    let zero: Vec<(Basic, Basic)> = unknowns.iter().map(|s| (s.clone(), Basic::zero())).collect();
    let n: usize = functions.len();
    let (mut a_entries, mut b_entries, mut rhs): (Vec<Basic>, Vec<Basic>, Vec<Basic>) =
        (Vec::new(), Vec::new(), Vec::new());
    let nonlinear = |equation: &Basic| {
        PolyError::Value(format!(
            "{} is not linear in the functions and their first derivatives",
            equation
        ))
    };
    for equation in equations {
        let r1: Basic = replaced(&row(equation)?, &first)?;
        if r1.preorder().iter().any(|n| matches!(n.head(), Head::Derivative)) {
            return Err(nonlinear(equation));
        }
        let r2: Basic = replaced(&r1, &functions_rule)?;
        if functions.iter().any(|f| r2.has(f)) {
            return Err(nonlinear(equation));
        }
        let coefficients: Vec<Basic> = unknowns
            .iter()
            .map(|s| crate::core::function::diff(&r2, &[(s.clone(), 1)]))
            .collect::<PolyResult<_>>()?;
        let constant: Basic = replaced(&r2, &zero)?;
        let has_unknown = |e: &Basic| unknowns.iter().any(|s| e.has(s));
        if coefficients.iter().any(has_unknown) || has_unknown(&constant) {
            return Err(nonlinear(equation));
        }
        a_entries.extend(coefficients[..n].iter().cloned());
        b_entries.extend(coefficients[n..].iter().cloned());
        rhs.push(neg(&constant)?);
    }
    let rows: usize = rhs.len();
    let (a, b) = (
        ExprMatrix::new(rows, n, a_entries)?,
        ExprMatrix::new(rows, n, b_entries)?,
    );
    if a.entries
        .iter()
        .chain(&b.entries)
        .any(|e| e.free_symbols().contains(&x))
    {
        return Err(value_error("the coefficients must be constant"));
    }
    Ok((a, b, ExprMatrix::column(rhs), x))
}

/// Whether `y` solves `A y' + B y = f` (`check_dae`).
pub fn check_dae(
    a: &ExprMatrix,
    b: &ExprMatrix,
    f: Option<&ExprMatrix>,
    y: &ExprMatrix,
    x: &Basic,
) -> PolyResult<bool> {
    let zero: ExprMatrix = ExprMatrix::zeros(a.rows, 1);
    let residual: ExprMatrix = a.matmul(&y.diff(x)?)?.add(&b.matmul(y)?)?.sub(f.unwrap_or(&zero))?;
    for e in &residual.entries {
        if !vanishes(&crate::core::function::doit(e)?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `simplify` makes `e` zero, as given or written with
/// exponentials (`_vanishes`).
fn vanishes(e: &Basic) -> PolyResult<bool> {
    if simplify(e)? == Basic::zero() {
        return Ok(true);
    }
    Ok(simplify(&rewrite(e, "exp")?)? == Basic::zero())
}
