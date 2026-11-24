//! Port of `sympy_extras/solvers/factorization.py`: the factorisation of
//! linear differential operators (right factors of order one from the
//! hyperexponential solutions and Kovacic's algorithm, of higher order by
//! Beke's method on the associated equations, left factors through the
//! adjoint) and the solutions of a factored operator by variation of
//! parameters.

#![allow(clippy::mutable_key_type)]

use crate::core::basic::{Basic, Function, Head};
use crate::error::{PolyError, PolyResult};
use crate::solvers::kovacic::replaced;
use crate::solvers::linear_ode::{
    exterior_system, hyperexponential_search, inverse, log_derivative, normal, resultant, root_multiplicities,
    scalar_operator, LinearOperator, UPoly,
};

type R = PolyResult<Basic>;

/// A truth value of three states (`Truth`).
pub type Truth = Option<bool>;

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
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn sum(terms: &[Basic]) -> R {
    if terms.is_empty() {
        Ok(Basic::zero())
    } else {
        add(terms)
    }
}
fn product(factors: &[Basic]) -> R {
    if factors.is_empty() {
        Ok(Basic::one())
    } else {
        mul(factors)
    }
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    if n == 0 {
        return Ok(e.clone());
    }
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
fn integer_of(e: &Basic) -> Option<i64> {
    if !matches!(e.head(), Head::Rational { value, .. } if value.is_int()) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn sign(k: usize) -> i64 {
    if k % 2 == 0 {
        1
    } else {
        -1
    }
}

/// `L = unit * factors[0] * ... * factors[-1]` with monic factors
/// (`OperatorFactorization`).
#[derive(Clone, Debug)]
pub struct OperatorFactorization {
    pub unit: Basic,
    pub factors: Vec<LinearOperator>,
    pub irreducible: Vec<Truth>,
    pub x: Basic,
}

impl OperatorFactorization {
    /// The product `unit * factors[0] * ... * factors[-1]` (`expand`).
    pub fn expand(&self) -> PolyResult<LinearOperator> {
        let mut result: LinearOperator = LinearOperator::new(std::slice::from_ref(&self.unit), &self.x);
        for factor in &self.factors {
            result = result.mul(factor)?;
        }
        Ok(result)
    }

    /// Whether every factor is proven irreducible (`is_complete`).
    pub fn is_complete(&self) -> bool {
        self.irreducible.iter().all(|f| *f == Some(true))
    }
}

type Found = (Option<LinearOperator>, bool);

// ---------------------------------------------------------------------------
// right factors of order one

/// A monic right factor `D - rho` of the monic `M`, and whether the search
/// was exhaustive (`_first_order_factor`).
fn first_order_factor(m: &LinearOperator) -> PolyResult<Found> {
    let x: &Basic = &m.x;
    let (solutions, mut complete) = hyperexponential_search(m, 200)?;
    for h in solutions {
        let rho: Basic = log_derivative(&h, x)?;
        if !is_rational_function(&rho, x) {
            complete = false;
            continue;
        }
        let r: LinearOperator = LinearOperator::new(&[neg(&rho)?, Basic::one()], x);
        if m.right_remainder(&r)?.is_zero() {
            return Ok((Some(r), complete));
        }
    }
    if m.order() == 2 {
        return kovacic_factor(m, complete);
    }
    Ok((None, complete))
}

/// Kovacic's algorithm on the monic `M = D**2 + p D + q`
/// (`_kovacic_factor`).
fn kovacic_factor(m: &LinearOperator, complete: bool) -> PolyResult<Found> {
    use crate::solvers::kovacic::{liouvillian_solution, normal_form};
    let x: &Basic = &m.x;
    let (q, p) = (&m.coefficients[0], &m.coefficients[1]);
    let r: Basic = normal_form(p, q, x)?;
    let Some(solution) = attempt(|| liouvillian_solution(&r, x))? else {
        return Ok((None, complete));
    };
    if let Some(s) = &solution {
        if s.case == 1 {
            if let Some(omega) = &s.omega {
                let rho: Basic = normal(&sub(omega, &div(p, &int(2))?)?)?;
                if is_rational_function(&rho, x) {
                    let right: LinearOperator = LinearOperator::new(&[neg(&rho)?, Basic::one()], x);
                    if m.right_remainder(&right)?.is_zero() {
                        return Ok((Some(right), complete));
                    }
                }
                return Ok((None, complete));
            }
        }
    }
    let denominator: UPoly = UPoly::of(&crate::core::expr::as_numer_denom(&normal(&r)?)?.1, x)?;
    let found = root_multiplicities(&denominator)?;
    let poles_found: bool = found.iter().map(|(_, m)| *m as i64).sum::<i64>() == denominator.degree();
    Ok((None, complete || poles_found))
}

// ---------------------------------------------------------------------------
// right factors of higher order: Beke's method

fn solve_with_inverse(inverse: &[Vec<Basic>], vector: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    for row in inverse {
        let mut terms: Vec<Basic> = Vec::new();
        for (j, v) in vector.iter().enumerate() {
            if row[j] != Basic::zero() && *v != Basic::zero() {
                terms.push(mul(&[row[j].clone(), v.clone()])?);
            }
        }
        result.push(normal(&sum(&terms)?)?);
    }
    Ok(result)
}

/// A rational function `f` with `f'/f = d`, and whether that was decided
/// (`_rational_ratio`).
fn rational_ratio(d: &Basic, x: &Basic) -> PolyResult<(Option<Basic>, bool)> {
    let d: Basic = normal(d)?;
    if d == Basic::zero() {
        return Ok((Some(Basic::one()), true));
    }
    let (p, q) = crate::core::expr::as_numer_denom(&d)?;
    let (numerator, denominator) = (UPoly::of(&p, x)?, UPoly::of(&q, x)?);
    if numerator.degree() >= denominator.degree() {
        return Ok((None, true));
    }
    let derivative: UPoly = denominator.diff()?;
    if denominator.gcd(&derivative)?.degree() > 0 {
        return Ok((None, true));
    }
    let t: Basic = dummy("t");
    let shifted: Basic = sub(&numerator.as_expr()?, &mul(&[t.clone(), derivative.as_expr()?])?)?;
    let rt: UPoly = UPoly::of(&resultant(&shifted, &denominator.as_expr()?, x)?, &t)?;
    let found = root_multiplicities(&rt)?;
    if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != rt.degree() {
        return Ok((None, false));
    }
    let mut f: Basic = Basic::one();
    for (k, _) in &found {
        let Some(kv) = integer_of(k) else {
            return Ok((None, true));
        };
        let g: UPoly = numerator.sub_poly(&derivative.scaled(k)?)?.gcd(&denominator)?;
        f = mul(&[f, pow(&g.as_expr()?, &int(kv))?])?;
    }
    if normal(&sub(&log_derivative(&f, x)?, &d)?)? != Basic::zero() {
        return Ok((None, false));
    }
    Ok((Some(f), true))
}

type Class = (Basic, Vec<Basic>);

/// The solutions grouped by their class modulo rational factors, and
/// whether every comparison was decided (`_classes`).
fn classes(solutions: &[Basic], x: &Basic) -> PolyResult<(Vec<Class>, bool)> {
    let mut found: Vec<(Basic, Vec<Basic>, Basic)> = Vec::new();
    let mut decided: bool = true;
    for s in solutions {
        let rho: Basic = log_derivative(s, x)?;
        let mut placed: bool = false;
        for (_, ratios, rho_h) in found.iter_mut() {
            let (ratio, known) = rational_ratio(&sub(&rho, rho_h)?, x)?;
            decided = decided && known;
            if let Some(ratio) = ratio {
                let mut candidate: Vec<Basic> = ratios.clone();
                candidate.push(ratio.clone());
                if independent_rational(&candidate, x)? {
                    ratios.push(ratio);
                }
                placed = true;
                break;
            }
        }
        if !placed {
            found.push((s.clone(), vec![Basic::one()], rho));
        }
    }
    Ok((found.into_iter().map(|(h, ratios, _)| (h, ratios)).collect(), decided))
}

/// Whether the rational functions are linearly independent over the
/// constants (`_independent_rational`).
fn independent_rational(functions: &[Basic], x: &Basic) -> PolyResult<bool> {
    let parts: Vec<(Basic, Basic)> = functions
        .iter()
        .map(crate::core::expr::as_numer_denom)
        .collect::<PolyResult<_>>()?;
    let common: Basic = product(&parts.iter().map(|(_, q)| q.clone()).collect::<Vec<_>>())?;
    let numerators: Vec<UPoly> = parts
        .iter()
        .map(|(p, q)| UPoly::of(&normal(&div(&mul(&[p.clone(), common.clone()])?, q)?)?, x))
        .collect::<PolyResult<_>>()?;
    let degree: i64 = numerators.iter().map(|p| p.degree()).max().unwrap_or(-1);
    let rows: Vec<Vec<Basic>> = (0..=degree)
        .map(|i| numerators.iter().map(|p| p.coeff(i)).collect())
        .collect();
    Ok(crate::solvers::recurr::row_reduce(&rows)?.1.len() == functions.len())
}

type Position = Vec<Vec<usize>>;

fn position_of(position: &Position, rows: &[usize]) -> usize {
    position.iter().position(|s| s == rows).expect("a subset")
}

/// The coordinate `w_I` for rows in any order (`_minor`).
fn minor(vector: &[Basic], position: &Position, rows: &[usize]) -> R {
    let mut listed: Vec<usize> = rows.to_vec();
    let mut unique: Vec<usize> = rows.to_vec();
    unique.sort();
    unique.dedup();
    if unique.len() < rows.len() {
        return Ok(Basic::zero());
    }
    let mut s: i64 = 1;
    for i in 0..listed.len() {
        for j in 0..listed.len() - 1 - i {
            if listed[j] > listed[j + 1] {
                listed.swap(j, j + 1);
                s = -s;
            }
        }
    }
    mul(&[int(s), vector[position_of(position, &listed)].clone()])
}

/// `R = sum (-1)**(k - i) w_{I_i}/w_{I_k} D**i` (`_factor_from_minors`).
fn factor_from_minors(
    vector: &[Basic],
    position: &Position,
    k: usize,
    x: &Basic,
) -> PolyResult<Option<LinearOperator>> {
    let first: Vec<usize> = (0..k).collect();
    let wronskian: Basic = vector[position_of(position, &first)].clone();
    if wronskian == Basic::zero() {
        return Ok(None);
    }
    let mut coefficients: Vec<Basic> = Vec::new();
    for i in 0..=k {
        let rows: Vec<usize> = (0..=k).filter(|j| *j != i).collect();
        let entry: Basic = vector[position_of(position, &rows)].clone();
        coefficients.push(normal(&div(&mul(&[int(sign(k - i)), entry])?, &wronskian)?)?);
    }
    Ok(Some(LinearOperator::new(&coefficients, x)))
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, n: usize, k: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            rec(i + 1, n, k, current, out);
            current.pop();
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    rec(0, n, k, &mut Vec::new(), &mut out);
    out
}

/// Nonzero combinations `sum c_j w_j` satisfying the Grassmann–Plücker
/// relations, one point per component found, and whether the polynomial
/// system was solved (`_decomposable`).
fn decomposable(
    vectors: &[Vec<Basic>],
    position: &Position,
    n: usize,
    k: usize,
    x: &Basic,
) -> PolyResult<(Vec<Vec<Basic>>, bool)> {
    let m: usize = vectors.len();
    let unknowns: Vec<Basic> = (0..m).map(|j| dummy(&format!("c{}", j))).collect();
    let mut combined: Vec<Basic> = Vec::new();
    for p in 0..vectors[0].len() {
        let mut terms: Vec<Basic> = Vec::new();
        for (c, v) in unknowns.iter().zip(vectors) {
            terms.push(mul(&[c.clone(), v[p].clone()])?);
        }
        combined.push(sum(&terms)?);
    }
    let mut equations: Vec<Basic> = Vec::new();
    for i_set in combinations(n, k - 1) {
        for j_set in combinations(n, k + 1) {
            let mut terms: Vec<Basic> = Vec::new();
            for l in 0..=k {
                let mut left: Vec<usize> = i_set.clone();
                left.push(j_set[l]);
                let right: Vec<usize> = j_set
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != l)
                    .map(|(_, v)| *v)
                    .collect();
                terms.push(mul(&[
                    int(sign(l)),
                    minor(&combined, position, &left)?,
                    minor(&combined, position, &right)?,
                ])?);
            }
            let relation: Basic = sum(&terms)?;
            let numerator: Basic =
                crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&relation, false, true)?)?.0;
            if numerator == Basic::zero() {
                continue;
            }
            for coefficient in UPoly::of(&numerator, x)?.all_coeffs() {
                let e: Basic = expand(&coefficient)?;
                if e != Basic::zero() && !equations.contains(&e) {
                    equations.push(e);
                }
            }
        }
    }
    let mut candidates: Vec<Vec<Basic>> = Vec::new();
    let mut complete: bool = true;
    for chart in 0..m {
        let mut fixed: Vec<(Basic, Basic)> = (0..chart).map(|j| (unknowns[j].clone(), Basic::zero())).collect();
        fixed.push((unknowns[chart].clone(), Basic::one()));
        let mut reduced: Vec<Basic> = Vec::new();
        for e in &equations {
            let r: Basic = expand(&replaced(e, &fixed)?)?;
            if r != Basic::zero() {
                reduced.push(r);
            }
        }
        if reduced.iter().any(crate::core::assumptions::is_number) {
            continue;
        }
        let free: Vec<Basic> = unknowns[chart + 1..].to_vec();
        let found: Option<Vec<crate::solvers::solvers::Solution>> = if reduced.is_empty() {
            Some(vec![Default::default()])
        } else {
            attempt(|| crate::solvers::solvers::solve_system(&reduced, &free))?.flatten()
        };
        let Some(found) = found else {
            complete = false;
            continue;
        };
        for solution in found {
            for default in [Basic::zero(), Basic::one()] {
                let defaults: Vec<(Basic, Basic)> = unknowns.iter().map(|u| (u.clone(), default.clone())).collect();
                let mut values: Vec<(Basic, Basic)> = fixed.clone();
                for (c, v) in solution_pairs(&solution) {
                    values.push((c, replaced(&v, &defaults)?));
                }
                for u in &unknowns {
                    if !values.iter().any(|(k, _)| k == u) {
                        values.push((u.clone(), default.clone()));
                    }
                }
                let vector: Vec<Basic> = combined
                    .iter()
                    .map(|e| normal(&replaced(e, &values)?))
                    .collect::<PolyResult<_>>()?;
                if vector.iter().any(|e| *e != Basic::zero()) {
                    candidates.push(vector);
                    break;
                }
            }
        }
    }
    Ok((candidates, complete))
}

fn solution_pairs(solution: &crate::solvers::solvers::Solution) -> Vec<(Basic, Basic)> {
    solution.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// `e_0`, then a few random cyclic vector candidates (seeded)
/// (`_starting_vectors`).
fn starting_vectors(size: usize) -> Vec<Vec<Basic>> {
    let mut generator = crate::random::PyRandom::from_int(&crate::numbers::Int::from(size as i64));
    let mut starts: Vec<Vec<Basic>> = vec![(0..size)
        .map(|i| if i == 0 { Basic::one() } else { Basic::zero() })
        .collect()];
    for _ in 0..3 {
        // randint(-3, 3)
        starts.push((0..size).map(|_| int(generator.randbelow(7) as i64 - 3)).collect());
    }
    starts
}

/// A monic right factor of order `k >= 2` of the monic `M` by Beke's
/// method, and whether the search was exhaustive (`_beke_factor`).
fn beke_factor(m: &LinearOperator, k: usize) -> PolyResult<Found> {
    let x: &Basic = &m.x;
    let n: usize = m.order();
    let (subsets, system) = exterior_system(m, k)?;
    let size: usize = subsets.len();
    let mut associated: Option<LinearOperator> = None;
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for start in starting_vectors(size) {
        let (operator, found_rows) = scalar_operator(&system, &start, x)?;
        let full: bool = operator.order() == size;
        associated = Some(operator);
        rows = found_rows;
        if full {
            break;
        }
    }
    let Some(associated) = associated.filter(|a| a.order() >= size) else {
        return Ok((None, false));
    };
    let Some(inv) = inverse(&rows)? else {
        return Ok((None, false));
    };
    let (solutions, mut complete) = hyperexponential_search(&associated, 200)?;
    let (found, decided) = classes(&solutions, x)?;
    complete = complete && decided;
    for (h, ratios) in found {
        let rho: Basic = log_derivative(&h, x)?;
        let mut vectors: Vec<Vec<Basic>> = Vec::new();
        for ratio in &ratios {
            // (h q)^(i)/h by T_{i+1} = T_i' + rho T_i
            let mut derivatives: Vec<Basic> = vec![ratio.clone()];
            for _ in 0..size - 1 {
                let last: Basic = derivatives.last().expect("nonempty").clone();
                derivatives.push(normal(&add(&[diff(&last, x, 1)?, mul(&[rho.clone(), last])?])?)?);
            }
            vectors.push(solve_with_inverse(&inv, &derivatives)?);
        }
        let candidates: Vec<Vec<Basic>> = if vectors.len() == 1 {
            vectors
        } else {
            let (c, solved) = decomposable(&vectors, &subsets, n, k, x)?;
            complete = complete && solved;
            c
        };
        for vector in candidates {
            if let Some(r) = factor_from_minors(&vector, &subsets, k, x)? {
                if m.right_remainder(&r)?.is_zero() {
                    return Ok((Some(r), complete));
                }
            }
        }
    }
    Ok((None, complete))
}

fn right_factor_of(m: &LinearOperator, k: usize) -> PolyResult<Found> {
    if k == 1 {
        return first_order_factor(m);
    }
    beke_factor(m, k)
}

/// A monic right factor of `L` of the given order, or `None`
/// (`right_factor`).
pub fn right_factor(l: &LinearOperator, order: usize) -> PolyResult<Option<LinearOperator>> {
    if order < 1 || order >= l.order() {
        return Err(PolyError::Value(format!(
            "the order of a proper right factor is between 1 and {}",
            l.order() as i64 - 1
        )));
    }
    Ok(right_factor_of(&l.monic()?, order)?.0)
}

/// A monic left factor of `L` of the given order, or `None`
/// (`left_factor`).
pub fn left_factor(l: &LinearOperator, order: usize) -> PolyResult<Option<LinearOperator>> {
    if order < 1 || order >= l.order() {
        return Err(PolyError::Value(format!(
            "the order of a proper left factor is between 1 and {}",
            l.order() as i64 - 1
        )));
    }
    let adjoint: LinearOperator = l.monic()?.adjoint()?.monic()?;
    match right_factor_of(&adjoint, order)?.0 {
        None => Ok(None),
        Some(r) => Ok(Some(r.adjoint()?.monic()?)),
    }
}

/// A factorisation of `L` into monic factors (`factor_operator`).
pub fn factor_operator(l: &LinearOperator) -> PolyResult<OperatorFactorization> {
    if l.is_zero() {
        return Err(PolyError::Value("the zero operator has no factorisation".to_string()));
    }
    let (factors, flags) = factor_monic(&l.monic()?)?;
    Ok(OperatorFactorization {
        unit: l.leading_coefficient().clone(),
        factors,
        irreducible: flags,
        x: l.x.clone(),
    })
}

type Factored = (Vec<LinearOperator>, Vec<Truth>);

fn factor_monic(m: &LinearOperator) -> PolyResult<Factored> {
    let n: usize = m.order();
    if n <= 1 {
        return Ok((vec![m.clone()], vec![Some(true)]));
    }
    let mut proven: bool = true;
    let mut adjoint: Option<LinearOperator> = None;
    for k in 1..=n / 2 {
        let (r, complete) = right_factor_of(m, k)?;
        proven = proven && complete;
        if let Some(r) = r {
            return Ok(joined(factor_monic(&m.right_quotient(&r)?)?, factor_monic(&r)?));
        }
        if 2 * k == n {
            // a left factor of order k is a right factor of order n - k = k
            continue;
        }
        if adjoint.is_none() {
            adjoint = Some(m.adjoint()?.monic()?);
        }
        let adj: &LinearOperator = adjoint.as_ref().expect("set above");
        let (r, complete) = right_factor_of(adj, k)?;
        proven = proven && complete;
        if let Some(r) = r {
            // M* = S R, so M = R* S* and S* is a right factor of order n - k
            let right: LinearOperator = adj.right_quotient(&r)?.adjoint()?.monic()?;
            let (quotient, remainder) = m.right_divmod(&right)?;
            if remainder.is_zero() {
                return Ok(joined(factor_monic(&quotient)?, factor_monic(&right)?));
            }
            proven = false;
        }
    }
    Ok((vec![m.clone()], vec![if proven { Some(true) } else { None }]))
}

fn joined(left: Factored, right: Factored) -> Factored {
    let (mut factors, mut flags) = left;
    factors.extend(right.0);
    flags.extend(right.1);
    (factors, flags)
}

// ---------------------------------------------------------------------------
// solving through the factorisation

/// The elementary functions (`_ELEMENTARY`).
const ELEMENTARY: [Function; 18] = [
    Function::Exp,
    Function::Log,
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Asin,
    Function::Acos,
    Function::Atan,
    Function::Acot,
    Function::Asinh,
    Function::Acosh,
    Function::Atanh,
    Function::Acoth,
];

/// Whether the functions in `e` are elementary (`_elementary`).
fn elementary(e: &Basic) -> bool {
    for node in e.preorder() {
        match node.head() {
            Head::Integral => continue,
            Head::Function(f) if !ELEMENTARY.contains(f) => return false,
            Head::Undefined(_) => return false,
            _ => {}
        }
    }
    true
}

fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

/// `Integral(integrand, x)`, evaluated when possible within the time limit
/// (`_quadrature`).
pub(crate) fn quadrature(integrand: &Basic, x: &Basic) -> R {
    let mut limit: Option<f64> = timeout();
    if limit.is_some() && !elementary(integrand) {
        limit = limit.map(|l| l / 6.0);
    }
    let simplified: Option<Basic> = if ops(integrand)? < 200 {
        crate::timeout::attempt_within(limit, || simplify(integrand))?
    } else {
        None
    };
    let f: Basic = simplified.unwrap_or_else(|| integrand.clone());
    let value: Option<Basic> = crate::timeout::attempt_within(limit, || {
        crate::integrals::indefinite::closed_antiderivative(&f, x, false, true, None)
    })?
    .flatten();
    match value {
        Some(v) => Ok(v),
        None => crate::integrals::integral::integral(&f, x, None),
    }
}

/// A solution of `L(y) = rhs` from a basis of the solutions of `L(y) = 0`
/// (`variation_of_parameters`).
pub fn variation_of_parameters(l: &LinearOperator, basis: &[Basic], rhs: &Basic) -> R {
    variation(l, basis, rhs, true)
}

/// `exp(-Integral(a_{n-1}/a_n))` (`_abel_wronskian`).
fn abel_wronskian(l: &LinearOperator) -> PolyResult<Option<Basic>> {
    let x: &Basic = &l.x;
    let p: Basic = normal(&div(
        &l.coefficients[l.coefficients.len() - 2],
        l.leading_coefficient(),
    )?)?;
    let Some(integral) = crate::integrals::indefinite::closed_antiderivative(&p, x, false, true, None)? else {
        return Ok(None);
    };
    let w: Basic = crate::functions::elementary::exponential::exp(&neg(&integral)?)?;
    Ok(Some(attempt(|| simplify(&w))?.unwrap_or(w)))
}

/// Variation of parameters (`_variation`).
fn variation(l: &LinearOperator, basis: &[Basic], rhs: &Basic, exact: bool) -> R {
    let x: &Basic = &l.x;
    let k: usize = l.order();
    if basis.len() != k {
        return Err(PolyError::Value(format!("a basis of {} solutions is needed", k)));
    }
    let g: Basic = div(rhs, l.leading_coefficient())?;
    if k == 1 {
        return mul(&[basis[0].clone(), quadrature(&div(&g, &basis[0])?, x)?]);
    }
    let wronskian_matrix: Vec<Vec<Basic>> = (0..k)
        .map(|i| basis.iter().map(|b| diff(b, x, i as u64)).collect::<PolyResult<_>>())
        .collect::<PolyResult<_>>()?;
    let determinant: Basic = crate::solvers::ode_cases::determinant(&wronskian_matrix)?;
    let abel: Option<Basic> = abel_wronskian(l)?;
    let mut w: Option<Basic> = None;
    if let Some(abel) = &abel {
        if !exact {
            w = Some(abel.clone());
        } else {
            let ratio: Option<Basic> = attempt(|| simplify(&div(&determinant, abel)?))?;
            if let Some(ratio) = ratio.filter(|r| *r != Basic::zero() && !r.has(x)) {
                w = Some(mul(&[ratio, abel.clone()])?);
            }
        }
    }
    let w: Basic = match w {
        Some(w) => w,
        None => attempt(|| simplify(&determinant))?.unwrap_or(determinant),
    };
    if w == Basic::zero() {
        return Err(PolyError::Value("the solutions are not independent".to_string()));
    }
    let mut terms: Vec<Basic> = Vec::new();
    for (j, b) in basis.iter().enumerate() {
        let minor_matrix: Vec<Vec<Basic>> = wronskian_matrix[..k - 1]
            .iter()
            .map(|row| (0..k).filter(|c| *c != j).map(|c| row[c].clone()).collect())
            .collect();
        let cofactor: Basic = mul(&[
            int(sign(k - 1 + j)),
            crate::solvers::ode_cases::determinant(&minor_matrix)?,
        ])?;
        terms.push(mul(&[
            b.clone(),
            quadrature(&div(&mul(&[g.clone(), cofactor])?, &w)?, x)?,
        ])?);
    }
    sum(&terms)
}

/// A solver of homogeneous equations (`Solver`).
pub type Solver<'a> = &'a dyn Fn(&LinearOperator) -> PolyResult<Vec<Basic>>;

/// Independent solutions of `factors[0] * ... * factors[-1]`
/// (`solve_factored`).
pub fn solve_factored(factors: &[LinearOperator], solver: Solver) -> PolyResult<Vec<Basic>> {
    let right: &LinearOperator = factors.last().expect("a factor");
    let basis: Vec<Basic> = solver(right)?;
    if factors.len() == 1 || basis.len() < right.order() {
        return Ok(basis);
    }
    let mut solutions: Vec<Basic> = basis.clone();
    for z in solve_factored(&factors[..factors.len() - 1], solver)? {
        let limit: Option<f64> = timeout().map(|t| 4.0 * t);
        let y: Option<Basic> = crate::timeout::attempt_within(limit, || variation(right, &basis, &z, false))?;
        if let Some(y) = y.filter(|y| *y != Basic::zero()) {
            solutions.push(y);
        }
    }
    Ok(solutions)
}
