//! Port of `sympy_extras/solvers/_special_values.py`: the cases of the
//! isolated values of the parameters of a differential equation at which
//! its general solution (or the basis of a linear homogeneous equation) is
//! undefined or not general, with the helpers ported from SymPy it needs
//! (`ode_order`, the Berkowitz determinant, `dotprodsimp`).

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::error::{PolyError, PolyResult};
use crate::special_values::Point;

type R = PolyResult<Basic>;

/// The seconds given to the symbolic check of a solution at a point
/// (`_CHECK_SECONDS`).
const CHECK_SECONDS: f64 = 2.0;
/// The most equations in a condition which is simplified
/// (`_MAX_LOGIC_ATOMS`).
const MAX_LOGIC_ATOMS: usize = 6;
/// The digits of the numerical checks (`_DIGITS`).
const DIGITS: usize = 30;

/// The general solution of an equation (with a point substituted), or
/// `None` (`SolveAt`).
pub type SolveAt<'a> = &'a dyn Fn(&Basic) -> PolyResult<Option<Basic>>;
/// The independent solutions of a linear equation (with a point
/// substituted), or `None` (`BasisAt`).
pub type BasisAt<'a> = &'a dyn Fn(&Basic) -> PolyResult<Option<Vec<Basic>>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn rat(p: i64, q: i64) -> Basic {
    Basic::rational(crate::numbers::Int::from(p), crate::numbers::Int::from(q)).expect("a nonzero denominator")
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    if n == 0 {
        return Ok(e.clone());
    }
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    let value: Basic = crate::core::function::xreplace_evaluated(e, &map)?;
    if !is_expr(&value) {
        return Err(PolyError::Type(format!("{} is not an expression", value)));
    }
    Ok(value)
}
fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b) && !crate::sets::sets::is_set(b) && !matches!(b.head(), Head::Tuple)
}
fn symbols_of(e: &Basic) -> BTreeSet<Basic> {
    e.free_symbols()
}
fn sorted(symbols: BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().collect();
    v.sort_by_key(|s| s.to_string());
    v
}

/// The variable of `f(x)` (`_variable`).
fn variable(f: &Basic) -> PolyResult<Basic> {
    match f.args().first() {
        Some(x) if x.is_symbol() => Ok(x.clone()),
        _ => Err(PolyError::Value("the function must depend on a symbol".to_string())),
    }
}

/// The number of variables of a derivative, with their counts
/// (`len(Derivative.variables)`).
fn derivative_count(d: &Basic) -> usize {
    d.args()[1..]
        .iter()
        .map(|pair| {
            if matches!(pair.head(), Head::Tuple) && pair.args().len() == 2 {
                pair.args()[1]
                    .as_rational()
                    .and_then(|r| usize::try_from(&crate::numbers::numer(r)).ok())
                    .unwrap_or(1)
            } else {
                1
            }
        })
        .sum()
}

/// The order of the highest derivative of `func` in `expr` (SymPy's
/// `ode_order`).
pub fn ode_order(expr: &Basic, func: &Basic) -> usize {
    if !expr.has(func) {
        return 0;
    }
    if matches!(expr.head(), Head::Derivative) {
        if expr.args()[0] == *func {
            return derivative_count(expr);
        }
        let inner: &[Basic] = expr.args()[0].args();
        let mut order: usize = derivative_count(expr);
        if !inner.is_empty() {
            order += inner.iter().map(|a| ode_order(a, func)).max().unwrap_or(0);
        }
        return order;
    }
    expr.args().iter().map(|a| ode_order(a, func)).max().unwrap_or(0)
}

/// The determinant of a square matrix of expressions as SymPy's
/// `det(method='berkowitz')` computes it (`determinant`).
pub fn determinant(rows: &[Vec<Basic>]) -> R {
    let n: usize = rows.len();
    if rows.iter().any(|row| row.len() != n) {
        return Err(PolyError::Value("the matrix is not square".to_string()));
    }
    let m = |i: usize, j: usize| rows[i][j].clone();
    match n {
        0 => Ok(Basic::one()),
        1 => Ok(m(0, 0)),
        2 => dotprodsimp(&sub(&mul(&[m(0, 0), m(1, 1)])?, &mul(&[m(0, 1), m(1, 0)])?)?),
        3 => dotprodsimp(&add(&[
            mul(&[m(0, 0), m(1, 1), m(2, 2)])?,
            mul(&[m(0, 1), m(1, 2), m(2, 0)])?,
            mul(&[m(0, 2), m(1, 0), m(2, 1)])?,
            neg(&mul(&[m(0, 2), m(1, 1), m(2, 0)])?)?,
            neg(&mul(&[m(0, 0), m(1, 2), m(2, 1)])?)?,
            neg(&mul(&[m(0, 1), m(1, 0), m(2, 2)])?)?,
        ])?),
        _ => {
            let vector: Vec<Basic> = berkowitz_vector(rows)?;
            let last: Basic = vector.last().expect("nonempty").clone();
            if n % 2 == 0 {
                Ok(last)
            } else {
                neg(&last)
            }
        }
    }
}

/// SymPy's `dotprodsimp` (`dotprodsimp`).
pub fn dotprodsimp(expr: &Basic) -> R {
    if !expr.is_add() && !expr.is_mul() && !expr.is_pow() {
        return Ok(expr.clone());
    }
    let hints = crate::core::function::Hints {
        mul: true,
        multinomial: true,
        power_base: false,
        power_exp: false,
        log: false,
        ..crate::core::function::Hints::DEFAULT
    };
    let mut expr: Basic = crate::core::function::expand_with(expr, true, hints)?;
    let (exprops, ratfunc) = count_ops_alg(&expr)?;
    if exprops >= 6 {
        // SymPy's empirical cutoff for the expensive simplification
        if ratfunc {
            let mut dummies: Vec<(Basic, Basic)> = Vec::new();
            let expr2: Basic = nonalgebraic_dummies(&expr, &mut dummies)?;
            if expr2.ptr_eq(&expr) || count_ops_alg(&expr2)?.0 >= 6 {
                let expr3: Basic = cancel(&expr2)?;
                if expr3 != expr2 {
                    let back: Vec<(Basic, Basic)> = dummies.iter().map(|(e, d)| (d.clone(), e.clone())).collect();
                    let restored: Basic = crate::core::subs::subs_pairs(&expr3, &back)?;
                    if !is_expr(&restored) {
                        return Err(PolyError::Type(format!("{} is not an expression", restored)));
                    }
                    expr = restored;
                }
            }
        }
    } else if exprops == 5 && expr.is_add() && special_quotient(&expr) {
        // the very special case x/(x - 1) - 1/(x - 1) = 1
        let expr2: Basic = crate::polys::rationaltools::together(&expr, false, true)?;
        if count_ops_alg(&expr2)?.0 < exprops {
            expr = expr2;
        }
    }
    Ok(expr)
}

fn special_quotient(expr: &Basic) -> bool {
    let args: &[Basic] = expr.args();
    if args.len() < 2 || !args[0].is_mul() || !args[1].is_mul() {
        return false;
    }
    let reciprocal = |m: &Basic| {
        let last: &Basic = m.args().last().expect("a product has factors");
        last.is_pow() && last.args()[1] == int(-1)
    };
    reciprocal(&args[0]) && reciprocal(&args[1])
}

/// `(m.args[0], Mul(*m.args[1:]))` (`as_two_terms`).
fn as_two_terms(m: &Basic) -> PolyResult<(Basic, Basic)> {
    let args: &[Basic] = m.args();
    Ok((args[0].clone(), mul(&args[1..])?))
}

/// The algebraic operations of `expr` without recursion into its
/// non-algebraic parts, and whether it may be a rational function
/// (`_count_ops_alg`).
fn count_ops_alg(expr: &Basic) -> PolyResult<(usize, bool)> {
    let mut ops: usize = 0;
    let mut pending: Vec<Basic> = vec![expr.clone()];
    let mut ratfunc: bool = false;
    while let Some(a) = pending.pop() {
        if let Head::Rational { value, .. } = a.head() {
            if a != Basic::one() {
                // -1/3 = NEG + DIV
                ops += usize::from(*value < crate::numbers::Rat::ZERO) + usize::from(!value.is_int());
            }
        } else if a.is_mul() {
            let mut m: Basic = a.clone();
            if crate::core::printing::could_extract_minus_sign(&m) {
                ops += 1;
                if m.args()[0] == int(-1) {
                    m = as_two_terms(&m)?.1;
                } else {
                    m = neg(&m)?;
                }
            }
            let (n, d) = crate::simplify::radsimp::fraction(&m)?;
            let integer_n: Option<i64> =
                n.as_rational()
                    .filter(|r| r.is_int())
                    .map(|r| if *r < crate::numbers::Rat::ZERO { -1 } else { 1 });
            if let Some(sign) = integer_n {
                ops += 1 + usize::from(sign < 0);
                pending.push(d); // won't be -Mul but could be Add
            } else if d != Basic::one() {
                if !d.as_rational().is_some_and(|r| r.is_int()) {
                    pending.push(d);
                    ratfunc = true;
                }
                ops += 1;
                pending.push(n); // could be -Mul
            } else {
                ops += m.args().len().saturating_sub(1);
                pending.extend(m.args().iter().filter(|e| is_expr(e)).cloned());
            }
        } else if a.is_add() {
            let count: usize = a.args().len();
            let mut negatives: usize = 0;
            for term in a.args() {
                if !is_expr(term) {
                    continue;
                }
                if crate::core::printing::could_extract_minus_sign(term) {
                    negatives += 1;
                    pending.push(neg(term)?);
                } else {
                    pending.push(term.clone());
                }
            }
            ops += count - usize::from(negatives != count); // -x - y = NEG + SUB
        } else if a.is_pow() {
            ops += 1;
            pending.push(a.args()[0].clone());
            if !ratfunc {
                ratfunc =
                    crate::core::expr::is(&a.args()[1], crate::core::assumptions::Predicate::negative) != Some(false);
            }
        }
    }
    Ok((ops, ratfunc))
}

/// `expr` with dummies for its non-algebraic parts
/// (`_nonalgebraic_dummies`).
fn nonalgebraic_dummies(expr: &Basic, dummies: &mut Vec<(Basic, Basic)>) -> R {
    if expr.args().is_empty() {
        return Ok(expr.clone());
    }
    if expr.is_add() || expr.is_mul() || expr.is_pow() {
        let mut args: Option<Vec<Basic>> = None;
        for (i, a) in expr.args().iter().enumerate() {
            if !is_expr(a) {
                continue;
            }
            let c: Basic = nonalgebraic_dummies(a, dummies)?;
            if c.ptr_eq(a) {
                continue;
            }
            let list: &mut Vec<Basic> = args.get_or_insert_with(|| expr.args().to_vec());
            list[i] = c;
        }
        let Some(args) = args else {
            return Ok(expr.clone());
        };
        return crate::core::function::rebuild_evaluated(expr, args);
    }
    if let Some((_, d)) = dummies.iter().find(|(e, _)| e == expr) {
        return Ok(d.clone());
    }
    let d: Basic = crate::core::symbol::Dummy::new("Dummy", &[]).0;
    dummies.push((expr.clone(), d.clone()));
    Ok(d)
}

/// The coefficients of the characteristic polynomial by Berkowitz's
/// recursion (`_berkowitz_vector`).
fn berkowitz_vector(m: &[Vec<Basic>]) -> PolyResult<Vec<Basic>> {
    let n: usize = m.len();
    if n == 0 {
        return Ok(vec![Basic::one()]);
    }
    if n == 1 {
        return Ok(vec![Basic::one(), neg(&m[0][0])?]);
    }
    let a: &Basic = &m[0][0];
    let r: &[Basic] = &m[0][1..];
    let c: Vec<Basic> = m[1..].iter().map(|row| row[0].clone()).collect();
    let sub_matrix: Vec<Vec<Basic>> = m[1..].iter().map(|row| row[1..].to_vec()).collect();
    // the first column of the Toeplitz matrix: 1, -a, -R C, -R A C, ...
    let mut diags: Vec<Vec<Basic>> = vec![c];
    for _ in 0..n - 2 {
        let previous: Vec<Basic> = diags.last().expect("nonempty").clone();
        let next: Vec<Basic> = sub_matrix
            .iter()
            .map(|row| dot(row, &previous))
            .collect::<PolyResult<_>>()?;
        diags.push(next);
    }
    let mut column: Vec<Basic> = vec![Basic::one(), neg(a)?];
    for d in &diags {
        column.push(neg(&dot(r, d)?)?);
    }
    let below: Vec<Basic> = berkowitz_vector(&sub_matrix)?;
    // the Toeplitz matrix (n + 1) x n times the vector of the submatrix
    let mut result: Vec<Basic> = Vec::new();
    for i in 0..=n {
        let mut total: Basic = Basic::zero();
        for j in 0..(i + 1).min(below.len()) {
            total = add(&[total, mul(&[column[i - j].clone(), below[j].clone()])?])?;
        }
        result.push(total);
    }
    Ok(result)
}

fn dot(u: &[Basic], v: &[Basic]) -> R {
    let mut total: Basic = Basic::zero();
    for (a, b) in u.iter().zip(v) {
        total = add(&[total, mul(&[a.clone(), b.clone()])?])?;
    }
    Ok(total)
}

/// The Jacobian determinant of `(y, y', ..., y^(m-1))` with respect to the
/// `m` constants for the family `y = solution` (`solution_jacobian`).
pub fn solution_jacobian(solution: &Basic, constants: &[Basic], x: &Basic) -> R {
    let columns: Vec<Basic> = constants
        .iter()
        .map(|c| diff(solution, c, 1))
        .collect::<PolyResult<_>>()?;
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for i in 0..constants.len() {
        rows.push(
            columns
                .iter()
                .map(|column| diff(column, x, i as u64))
                .collect::<PolyResult<_>>()?,
        );
    }
    determinant(&rows)
}

/// The generic branch of a value with cases (`_generic`).
fn generic(value: &Basic) -> Basic {
    if matches!(value.head(), Head::Piecewise) {
        return value.args().last().expect("a branch").args()[0].clone();
    }
    value.clone()
}

/// The sample values of the variable and of the constants in the
/// numerical checks (`_SAMPLES`).
fn samples() -> [[Basic; 5]; 2] {
    [
        [rat(7, 5), rat(3, 7), rat(5, 11), rat(-2, 3), rat(9, 13)],
        [rat(23, 9), rat(-4, 5), rat(7, 3), rat(3, 8), rat(-5, 7)],
    ]
}

/// The sample values of `x` and of the other `symbols` (`_values`).
fn values(x: &Basic, symbols: &[Basic], sample: &[Basic; 5]) -> PolyResult<Point> {
    let mut point: Point = vec![(x.clone(), sample[0].clone())];
    for (i, c) in symbols.iter().enumerate() {
        let k: usize = sample.len() - 1;
        point.push((c.clone(), add(&[sample[1 + i % k].clone(), int((i / k) as i64)])?));
    }
    Ok(point)
}

/// Whether `e` is zero at the sample points of its symbols
/// (`_zero_at_samples`).
fn zero_at_samples(e: &Basic, x: &Basic) -> PolyResult<Option<bool>> {
    let mut tested: usize = 0;
    let mut others: BTreeSet<Basic> = symbols_of(e);
    others.remove(x);
    let others: Vec<Basic> = sorted(others);
    let all = samples();
    for sample in &all {
        let Some(number) = crate::numeric::reliable_value(e, DIGITS, &values(x, &others, sample)?) else {
            continue;
        };
        let (re_part, im_part) = crate::core::evalf::parts(&number);
        let magnitude: f64 = re_part.to_f64().value().hypot(im_part.to_f64().value());
        if !magnitude.is_finite() {
            continue;
        }
        if magnitude > 1e-20 {
            return Ok(Some(false));
        }
        tested += 1;
    }
    Ok(if tested == all.len() { Some(true) } else { None })
}

/// Whether `f = solution` satisfies `equation = 0` (`_solves`).
fn solves(equation: &Basic, f: &Basic, solution: &Basic) -> PolyResult<bool> {
    let x: Basic = variable(f)?;
    let substituted: Basic = crate::core::subs::subs(equation, f, solution)?;
    if !is_expr(&substituted) {
        return Ok(false);
    }
    let residual: Basic = crate::core::function::doit(&substituted)?;
    if residual.preorder().iter().any(|n| matches!(n.head(), Head::Derivative)) {
        return Ok(false);
    }
    let simplified: Option<Basic> = crate::timeout::attempt_within(Some(CHECK_SECONDS), || simplify(&residual))?;
    if simplified == Some(Basic::zero()) {
        return Ok(true);
    }
    Ok(zero_at_samples(&residual, &x)? == Some(true))
}

/// Whether the family has all its constants (`_is_general`).
fn is_general(solution: &Basic, constants: &[Basic], x: &Basic) -> PolyResult<bool> {
    if constants.is_empty() {
        return Ok(true);
    }
    Ok(zero_at_samples(&solution_jacobian(solution, constants, x)?, x)? == Some(false))
}

/// Whether an equation with a point of its parameters substituted is still
/// defined as a function of `x` (`defined_equation`).
pub fn defined_equation(equation: &Basic, x: &Basic) -> bool {
    for node in equation.preorder() {
        if matches!(node.head(), Head::NaN | Head::ComplexInfinity) {
            return false;
        }
        if node.is_pow()
            && crate::core::expr::is(&node.args()[0], crate::core::assumptions::Predicate::zero) == Some(true)
        {
            return false;
        }
        if node.is_function(Function::DiracDelta) && !node.args()[0].free_symbols().contains(x) {
            return false;
        }
    }
    true
}

/// `general`, the general solution of `equation = 0` for `f`, with a case
/// for each isolated value of the parameters at which it is undefined or
/// not general (`general_solution_cases`).
pub fn general_solution_cases(general: &Basic, equation: &Basic, f: &Basic, solve_at: SolveAt) -> R {
    let x: Basic = variable(f)?;
    let mut parameter_set: BTreeSet<Basic> = symbols_of(equation);
    parameter_set.remove(&x);
    let parameters: Vec<Basic> = sorted(parameter_set);
    if parameters.is_empty() {
        return Ok(general.clone());
    }
    let equation_symbols: BTreeSet<Basic> = symbols_of(equation);
    let constants: Vec<Basic> = sorted(symbols_of(general).difference(&equation_symbols).cloned().collect());
    let order: usize = ode_order(equation, f);
    let defined = |point: &Point| -> bool {
        match replaced(equation, point) {
            Ok(at) => defined_equation(&at, &x) && ode_order(&at, f) == order,
            Err(_) => false,
        }
    };
    let compute = |point: &Point, _assumptions: &[Basic]| -> PolyResult<Option<Basic>> {
        let at: Basic = replaced(equation, point)?;
        let Some(value) = solve_at(&at)? else {
            return Ok(None);
        };
        let generic_value: Basic = generic(&value);
        let at_symbols: BTreeSet<Basic> = symbols_of(&at);
        let found: BTreeSet<Basic> = symbols_of(&generic_value).difference(&at_symbols).cloned().collect();
        let expected: BTreeSet<Basic> = constants.iter().cloned().collect();
        if found != expected {
            // other constants, or fewer: no specialisation of the family
            return Ok(None);
        }
        if !solves(&at, f, &generic_value)? || !is_general(&generic_value, &constants, &x)? {
            return Ok(None);
        }
        Ok(Some(value))
    };
    let degenerate = |branch: &Basic| -> PolyResult<Vec<Basic>> {
        let present: usize = constants.iter().filter(|c| branch.has(c)).count();
        if present != constants.len() || constants.is_empty() {
            return Ok(Vec::new());
        }
        let (numerator, _) = crate::simplify::radsimp::fraction(&cancel(&solution_jacobian(branch, &constants, &x)?)?)?;
        Ok(vec![numerator])
    };
    crate::special_values::with_special_values(
        general,
        &parameters,
        &[],
        &compute,
        Some(&defined),
        false,
        None,
        true,
        Some(&degenerate),
    )
}

/// The independent solutions `basis` of a homogeneous linear `equation =
/// 0`, each with the cases of [`general_solution_cases`] (`basis_cases`).
pub fn basis_cases(basis: &[Basic], equation: &Basic, f: &Basic, solve_at: BasisAt) -> PolyResult<Vec<Basic>> {
    if basis.is_empty() {
        return Ok(Vec::new());
    }
    let constants: Vec<Basic> = (0..basis.len())
        .map(|i| crate::core::symbol::Dummy::new(&format!("C{}", i + 1), &[]).0)
        .collect();
    let combination = |functions: &[Basic]| -> R {
        let mut total: Basic = Basic::zero();
        for (c, g) in constants.iter().zip(functions) {
            total = add(&[total, mul(&[c.clone(), g.clone()])?])?;
        }
        crate::functions::elementary::piecewise::piecewise_fold(&total, None)
    };
    let combined_at = |at: &Basic| -> PolyResult<Option<Basic>> {
        let Some(functions) = solve_at(at)? else {
            return Ok(None);
        };
        if functions.len() != basis.len() {
            return Ok(None);
        }
        Ok(Some(combination(&functions)?))
    };
    let value: Basic = general_solution_cases(&combination(basis)?, equation, f, &combined_at)?;
    if !matches!(value.head(), Head::Piecewise) {
        return Ok(basis.to_vec());
    }
    let mut result: Vec<Basic> = Vec::new();
    for c in &constants {
        let mut cases: Vec<(Basic, Basic)> = Vec::new();
        for pair in value.args() {
            cases.push((diff(&pair.args()[0], c, 1)?, pair.args()[1].clone()));
        }
        result.push(function_cases(cases)?);
    }
    Ok(result)
}

/// The `Piecewise` of the cases of one function, the conditions of
/// consecutive cases with the same value joined and simplified
/// (`_function_cases`).
fn function_cases(cases: Vec<(Basic, Basic)>) -> R {
    let mut joined: Vec<(Basic, Basic)> = Vec::new();
    for (e, condition) in cases {
        match joined.last_mut() {
            Some(last) if last.0 == e => {
                let mut combined: Basic = crate::core::logic::or_(&[last.1.clone(), condition]);
                let equations: usize = combined
                    .atoms(Some(&|n: &Basic| matches!(n.head(), Head::Relational(RelOp::Eq))))
                    .len();
                if combined != crate::core::logic::true_() && equations <= MAX_LOGIC_ATOMS {
                    combined = crate::logic::boolalg::simplify_logic(&combined, None, false, None)?;
                }
                last.1 = combined;
            }
            _ => joined.push((e, condition)),
        }
    }
    crate::functions::elementary::piecewise::piecewise(&joined)
}

/// The expression which is zero: `lhs - rhs` for an `Eq` (`as_zero`).
pub fn as_zero(equation: &Basic) -> R {
    if matches!(equation.head(), Head::Relational(RelOp::Eq)) {
        return sub(&equation.args()[0], &equation.args()[1]);
    }
    if !is_expr(equation) {
        return Err(PolyError::Type(format!("{} is not an equation", equation)));
    }
    Ok(equation.clone())
}

/// The right-hand side of the only solution, when there is one and it is
/// explicit (`explicit_solution`).
pub fn explicit_solution(solutions: &[Basic], f: &Basic) -> Option<Basic> {
    if solutions.len() != 1 {
        return None;
    }
    let solution: &Basic = &solutions[0];
    if matches!(solution.head(), Head::Relational(RelOp::Eq)) && solution.args()[0] == *f && !solution.args()[1].has(f)
    {
        return Some(solution.args()[1].clone());
    }
    None
}

/// The only solution `Eq(f, rhs)` of `equation` with the cases of
/// [`general_solution_cases`] (`solution_cases`).
pub fn solution_cases(
    solutions: &[Basic],
    equation: &Basic,
    f: &Basic,
    solve: &dyn Fn(&Basic) -> PolyResult<Vec<Basic>>,
) -> PolyResult<Option<Basic>> {
    let Some(general) = explicit_solution(solutions, f) else {
        return Ok(None);
    };
    let solve_at = |at: &Basic| -> PolyResult<Option<Basic>> { Ok(explicit_solution(&solve(at)?, f)) };
    let value: Basic = general_solution_cases(&general, &as_zero(equation)?, f, &solve_at)?;
    if value == general {
        return Ok(None);
    }
    Ok(Some(crate::core::relational::rel(RelOp::Eq, f, &value)))
}
