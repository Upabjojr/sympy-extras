//! Port of `sympy_extras/concrete/delta.py`: sums and products of
//! expressions with Kronecker deltas (`deltasummation`, `deltaproduct`),
//! the deltas solved for the index.

use crate::core::basic::{Basic, Function, Head};
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;
pub(crate) type Limit = (Basic, Basic, Basic);

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn is_kd(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(Function::KroneckerDelta))
}
fn has_kd(e: &Basic) -> bool {
    e.preorder().iter().any(is_kd)
}
fn delta(i: &Basic, j: &Basic) -> R {
    crate::functions::special::tensor_functions::kronecker_delta(i, j)
}
fn decided_negative(e: &Basic) -> PolyResult<bool> {
    Ok(crate::core::relational::lt(e, &int(0))? == crate::core::logic::true_())
}
fn not_solvable(what: String) -> PolyError {
    PolyError::NotImplemented(what)
}

/// SymPy's `solve(e, v)` natively (`_solve`).
pub(crate) fn solve(e: &Basic, v: &Basic) -> PolyResult<Vec<Basic>> {
    if let Some(known) = crate::solvers::linsolve::solve_one(e, v)? {
        return Ok(known);
    }
    if let Some(listed) = finite_solutions(e, v)? {
        return Ok(listed);
    }
    match crate::solvers::solvers::solve_equation(e, v)? {
        Some(found) => Ok(found),
        None => Err(not_solvable(format!("no native method solves {} = 0 for {}", e, v))),
    }
}

fn finite_solutions(e: &Basic, v: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let equation: Basic = crate::core::relational::eq(e, &int(0))?;
    let complexes: Basic = Basic::atom(Head::Complexes);
    let found = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::assumptions::solve::solve(
            std::slice::from_ref(&equation),
            Some(std::slice::from_ref(v)),
            &[],
            Some(&complexes),
            false,
        )
    })?;
    let Some(found) = found else { return Ok(None) };
    if matches!(found.head(), Head::EmptySet) {
        return Ok(Some(Vec::new()));
    }
    if !matches!(found.head(), Head::FiniteSet) {
        return Ok(None);
    }
    let solutions: Vec<Basic> = found.args().to_vec();
    if solutions.iter().any(|s| crate::core::logic::is_boolean(s) || s.has(v)) {
        return Ok(None);
    }
    Ok(Some(solutions))
}

fn solve_dicts(eqs: &[Basic]) -> PolyResult<Vec<Vec<(Basic, Basic)>>> {
    if let Some(solution) = crate::solvers::linsolve::numeric_linear_solution(eqs)? {
        return Ok(vec![solution]);
    }
    let mut symbols: Vec<Basic> = Vec::new();
    for e in eqs {
        for s in e.free_symbols() {
            if s.is_symbol() && !symbols.contains(&s) {
                symbols.push(s);
            }
        }
    }
    let symbols: Vec<Basic> = crate::core::sorting::ordered(symbols);
    match crate::solvers::solvers::solve_system(eqs, &symbols)? {
        Some(found) => Ok(found),
        None => Err(not_solvable(format!(
            "no native method solves {:?}",
            eqs.iter().map(|e| e.to_string()).collect::<Vec<_>>()
        ))),
    }
}

fn expand_delta(expr: &Basic, index: &Basic) -> R {
    if !expr.is_mul() {
        return Ok(expr.clone());
    }
    let mut found: bool = false;
    let mut terms: Vec<Basic> = vec![int(1)];
    for h in expr.args() {
        if !found && h.is_add() && has_simple_delta(h, index)? {
            found = true;
            terms = h.args().iter().map(|t| m2(&terms[0], t)).collect::<PolyResult<_>>()?;
        } else {
            terms = terms.iter().map(|t| m2(t, h)).collect::<PolyResult<_>>()?;
        }
    }
    add_all(&terms)
}

fn extract_delta(expr: &Basic, index: &Basic) -> PolyResult<(Option<Basic>, Basic)> {
    if !has_simple_delta(expr, index)? {
        return Ok((None, expr.clone()));
    }
    if is_kd(expr) {
        return Ok((Some(expr.clone()), int(1)));
    }
    if !expr.is_mul() {
        return Err(PolyError::Value("Incorrect expr".to_string()));
    }
    let mut found: Option<Basic> = None;
    let mut terms: Vec<Basic> = Vec::new();
    for a in expr.args() {
        if found.is_none() && is_simple_delta(a, index)? {
            found = Some(a.clone());
        } else {
            terms.push(a.clone());
        }
    }
    Ok((found, mul_all(&terms)?))
}

/// Whether the expression has a delta linear in the index (`_has_simple_delta`).
pub(crate) fn has_simple_delta(expr: &Basic, index: &Basic) -> PolyResult<bool> {
    if has_kd(expr) {
        if is_simple_delta(expr, index)? {
            return Ok(true);
        }
        if expr.is_add() || expr.is_mul() {
            for a in expr.args() {
                if has_simple_delta(a, index)? {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

fn is_simple_delta(d: &Basic, index: &Basic) -> PolyResult<bool> {
    if is_kd(d) && d.has(index) {
        return match crate::concrete::gosper::UPoly::of(&s2(&d.args()[0], &d.args()[1])?, index) {
            Ok(p) => Ok(p.degree() == 1),
            Err(PolyError::Value(_)) => Ok(false),
            Err(e) => Err(e),
        };
    }
    Ok(false)
}

fn remove_multiple_delta(expr: &Basic) -> R {
    if expr.is_add() {
        let parts: Vec<Basic> = expr
            .args()
            .iter()
            .map(remove_multiple_delta)
            .collect::<PolyResult<_>>()?;
        return add_all(&parts);
    }
    if !expr.is_mul() {
        return Ok(expr.clone());
    }
    let mut eqs: Vec<Basic> = Vec::new();
    let mut newargs: Vec<Basic> = Vec::new();
    for a in expr.args() {
        if is_kd(a) {
            eqs.push(s2(&a.args()[0], &a.args()[1])?);
        } else {
            newargs.push(a.clone());
        }
    }
    if eqs.is_empty() {
        return Ok(expr.clone());
    }
    let solns = solve_dicts(&eqs)?;
    if solns.is_empty() {
        return Ok(int(0));
    }
    if solns.len() == 1 {
        for (k, v) in &solns[0] {
            newargs.push(delta(k, v)?);
        }
        let expr2: Basic = mul_all(&newargs)?;
        if *expr != expr2 {
            return remove_multiple_delta(&expr2);
        }
    }
    Ok(expr.clone())
}

fn simplify_delta(expr: &Basic) -> R {
    if is_kd(expr) {
        let slns = match solve_dicts(&[s2(&expr.args()[0], &expr.args()[1])?]) {
            Ok(s) => s,
            Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {
                return Ok(expr.clone())
            }
            Err(e) => return Err(e),
        };
        if slns.len() == 1 {
            let parts: Vec<Basic> = slns[0].iter().map(|(k, v)| delta(k, v)).collect::<PolyResult<_>>()?;
            return mul_all(&parts);
        }
    }
    Ok(expr.clone())
}

/// The product of `f` with Kronecker deltas over the limit (`deltaproduct`).
pub fn deltaproduct(f: &Basic, limit: &Limit) -> R {
    let (i, a, b) = limit;
    if decided_negative(&s2(b, a)?)? {
        return Ok(int(1));
    }
    if !has_kd(f) {
        return crate::concrete::products::product(f, limit);
    }
    if f.is_add() {
        let mut found: Option<Basic> = None;
        let mut terms: Vec<Basic> = Vec::new();
        let mut sorted: Vec<Basic> = f.args().to_vec();
        sorted.sort_by_cached_key(crate::core::sorting::sort_key);
        for arg in sorted {
            if found.is_none() && has_simple_delta(&arg, i)? {
                found = Some(arg);
            } else {
                terms.push(arg);
            }
        }
        let found: Basic = found.ok_or_else(|| PolyError::Value("a simple delta".to_string()))?;
        let newexpr: Basic = add_all(&terms)?;
        let k: Basic =
            crate::core::symbol::Dummy::new("kprime", &[(crate::core::assumptions::Predicate::integer, true)]).0;
        let left: Basic = deltaproduct(&newexpr, &(i.clone(), a.clone(), s2(&k, &int(1))?))?;
        let right: Basic = deltaproduct(&newexpr, &(i.clone(), a2(&k, &int(1))?, b.clone()))?;
        let inner: Basic = m2(&m2(&left, &subs(&found, i, &k)?)?, &right)?;
        let summed: Basic = deltasummation(&inner, &(k, a.clone(), b.clone()), has_simple_delta(&newexpr, i)?)?;
        let result: Basic = a2(&deltaproduct(&newexpr, limit)?, &summed)?;
        return remove_multiple_delta(&result);
    }
    let (found, _) = extract_delta(f, i)?;
    if found.is_none() {
        let g: Basic = expand_delta(f, i)?;
        if *f != g {
            return crate::polys::polytools::factor(&deltaproduct(&g, limit)?, &[], false, true);
        }
        return crate::concrete::products::product(f, limit);
    }
    let first: Basic = remove_multiple_delta(&m2(&subs(f, i, a)?, &delta(b, a)?)?)?;
    a2(&first, &m2(&int(1), &simplify_delta(&delta(b, &s2(a, &int(1))?)?)?)?)
}

/// The sum of `f` with Kronecker deltas over the limit: the term at the
/// solution of the delta when it lies in the range (unconditional with
/// `no_piecewise`) (`deltasummation`).
pub fn deltasummation(f: &Basic, limit: &Limit, no_piecewise: bool) -> R {
    let (x, a, b) = limit;
    if decided_negative(&s2(b, a)?)? {
        return Ok(int(0));
    }
    if !has_kd(f) {
        return crate::concrete::summations::summation(f, limit);
    }
    let g: Basic = expand_delta(f, x)?;
    if g.is_add() {
        let parts: Vec<Basic> = g
            .args()
            .iter()
            .map(|h| deltasummation(h, limit, no_piecewise))
            .collect::<PolyResult<_>>()?;
        return crate::functions::elementary::piecewise::piecewise_fold(&add_all(&parts)?, None);
    }
    let (found, expr) = extract_delta(&g, x)?;
    let Some(found) = found else {
        return crate::concrete::summations::summation(f, limit);
    };
    let solns: Vec<Basic> = solve(&s2(&found.args()[0], &found.args()[1])?, x)?;
    if solns.is_empty() {
        return Ok(int(0));
    }
    if solns.len() != 1 {
        let tuple: Basic = Basic::new(Head::Tuple, vec![x.clone(), a.clone(), b.clone()]);
        return crate::concrete::expr_with_limits::sum_(f, &[tuple]);
    }
    let value: &Basic = &solns[0];
    if no_piecewise {
        return subs(&expr, x, value);
    }
    // the index runs over a + 0, a + 1, ...: a solution off this lattice
    // is no term of the sum (sympy/sympy#31275, V47)
    let infinite = |e: &Basic| matches!(e.head(), Head::Infinity | Head::NegativeInfinity);
    let anchor: Basic = if !infinite(a) {
        a.clone()
    } else if !infinite(b) {
        b.clone()
    } else {
        int(0)
    };
    let offset: Basic = crate::core::add::sub(value, &anchor)?;
    let on_lattice: Option<bool> = crate::core::expr::is(&offset, crate::core::assumptions::Predicate::integer);
    if on_lattice == Some(false) {
        return Ok(int(0));
    }
    let mut condition: Basic =
        crate::sets::sets::as_relational(&crate::sets::sets::interval(a, b, false, false)?, value)?;
    if on_lattice.is_none() {
        condition = crate::core::logic::and_(&[
            condition,
            crate::sets::sets::contains(&Basic::atom(Head::Integers), &offset)?,
        ]);
    }
    crate::functions::elementary::piecewise::piecewise(&[
        (subs(&expr, x, value)?, condition),
        (int(0), crate::core::logic::true_()),
    ])
}
