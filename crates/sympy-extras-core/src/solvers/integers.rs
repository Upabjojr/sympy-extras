//! Linear problems over the integers, ported from
//! `sympy_extras.solvers.integers`: the general integer solution of a
//! linear system by a Hermite normal form with its unimodular transform,
//! the minimal nonnegative solutions and Hilbert bases (Contejean and
//! Devie), and Cooper's quantifier elimination for Presburger arithmetic.
//! The relations are visited in the order of their first occurrence (the
//! Python iterates a set).

#![allow(clippy::needless_range_loop, clippy::mutable_key_type)]

use crate::assumptions::quantifiers::{Prefix, QuantifierKind};
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::logic::{and_, boolean, not_, or_};
use crate::core::relational::{eq, ge, gt, le, lt, ne};
use crate::core::symbol::Symbol;
use crate::error::{arithmetic_error, value_error, PolyError, PolyResult};
use crate::numbers::{div_floor, Int};
use crate::polys::virtual_substitution::{to_nnf, with_sides};

/// An integer matrix as rows.
pub type IntMatrix = Vec<Vec<Int>>;
/// An integer vector.
pub type IntVector = Vec<Int>;

fn extended_gcd(a: &Int, b: &Int) -> (Int, Int, Int) {
    let (mut old_r, mut r) = (a.clone(), b.clone());
    let (mut old_u, mut u) = (Int::ONE, Int::ZERO);
    let (mut old_v, mut v) = (Int::ZERO, Int::ONE);
    while !r.is_zero() {
        let q: Int = div_floor(&old_r, &r);
        let next_r: Int = &old_r - &q * &r;
        old_r = std::mem::replace(&mut r, next_r);
        let next_u: Int = &old_u - &q * &u;
        old_u = std::mem::replace(&mut u, next_u);
        let next_v: Int = &old_v - &q * &v;
        old_v = std::mem::replace(&mut v, next_v);
    }
    if old_r < Int::ZERO {
        return (-old_r, -old_u, -old_v);
    }
    (old_r, old_u, old_v)
}

/// The column Hermite normal form `H` and the unimodular `U` with
/// `A U = H` (`hermite_normal_form_with_transform`).
pub fn hermite_normal_form_with_transform(a: &IntMatrix) -> (IntMatrix, IntMatrix) {
    let m: usize = a.len();
    let n: usize = a.first().map(|r| r.len()).unwrap_or(0);
    let mut h: IntMatrix = a.clone();
    let mut u: IntMatrix = (0..n)
        .map(|i| (0..n).map(|j| if i == j { Int::ONE } else { Int::ZERO }).collect())
        .collect();
    fn combine(rows: &mut IntMatrix, j: usize, k: usize, p: &Int, q: &Int, r: &Int, s: &Int) {
        for row in rows.iter_mut() {
            let (cj, ck) = (row[j].clone(), row[k].clone());
            row[j] = p * &cj + q * &ck;
            row[k] = r * &cj + s * &ck;
        }
    }
    let mut pivot: usize = 0;
    for i in 0..m {
        if pivot >= n {
            break;
        }
        for k in pivot + 1..n {
            if h[i][k].is_zero() {
                continue;
            }
            let (x, y) = (h[i][pivot].clone(), h[i][k].clone());
            let (g, s1, t1) = extended_gcd(&x, &y);
            let (r1, s2) = (div_floor(&-y.clone(), &g), div_floor(&x, &g));
            combine(&mut h, pivot, k, &s1, &t1, &r1, &s2);
            combine(&mut u, pivot, k, &s1, &t1, &r1, &s2);
        }
        if h[i][pivot].is_zero() {
            continue;
        }
        if h[i][pivot] < Int::ZERO {
            for rows in [&mut h, &mut u] {
                for row in rows.iter_mut() {
                    row[pivot] = -row[pivot].clone();
                }
            }
        }
        let p: Int = h[i][pivot].clone();
        for k in 0..pivot {
            let q: Int = div_floor(&h[i][k], &p);
            if !q.is_zero() {
                for rows in [&mut h, &mut u] {
                    for row in rows.iter_mut() {
                        let value: Int = &row[k] - &q * &row[pivot];
                        row[k] = value;
                    }
                }
            }
        }
        pivot += 1;
    }
    (h, u)
}

fn integer_of(b: &Basic) -> Option<Int> {
    b.as_rational().filter(|v| v.is_int()).map(|v| v.numerator().clone())
}

/// `A`, `b` with the equations `A x = b`, `None` when the system is not of
/// this kind (`_linear_system`).
fn linear_system(equations: &[Basic], symbols: &[Basic]) -> PolyResult<Option<(IntMatrix, IntVector)>> {
    if symbols.is_empty() {
        return Err(value_error("no unknown to solve for".to_string()));
    }
    let n: usize = symbols.len();
    let (mut a, mut b): (IntMatrix, IntVector) = (Vec::new(), Vec::new());
    for e in equations {
        if !e.free_symbols().iter().all(|s| symbols.contains(s)) {
            return Ok(None);
        }
        let poly = match crate::polys::polytools::Poly::new(e, symbols) {
            Ok(p) => p,
            Err(PolyError::Polys(..)) => return Ok(None),
            Err(other) => return Err(other),
        };
        if poly.total_degree() > 1 {
            return Ok(None);
        }
        let terms = poly.terms()?;
        let coefficient = |monomial: &[u32]| -> Basic {
            terms
                .iter()
                .find(|(m, _)| m.as_slice() == monomial)
                .map(|(_, c)| c.clone())
                .unwrap_or_else(Basic::zero)
        };
        let mut row: IntVector = Vec::new();
        for i in 0..n {
            let unit: Vec<u32> = (0..n).map(|j| if j == i { 1 } else { 0 }).collect();
            let Some(c) = integer_of(&coefficient(&unit)) else {
                return Ok(None);
            };
            row.push(c);
        }
        let Some(constant) = integer_of(&coefficient(&vec![0; n])) else {
            return Ok(None);
        };
        a.push(row);
        b.push(-constant);
    }
    Ok(Some((a, b)))
}

/// The general integer solution of a linear system with integer
/// coefficients: the values of the symbols in free integer parameters,
/// and the parameters; `None` without integer solution
/// (`linear_diophantine_system`).
pub fn linear_diophantine_system(
    equations: &[Basic],
    symbols: &[Basic],
) -> PolyResult<Option<(Vec<Basic>, Vec<Basic>)>> {
    let exprs: Vec<Basic> = equations
        .iter()
        .map(|e| match e.head() {
            Head::Relational(RelOp::Eq) => crate::core::add::sub(&e.args()[0], &e.args()[1]),
            _ => Ok(e.clone()),
        })
        .collect::<PolyResult<_>>()?;
    let Some((a, b)) = linear_system(&exprs, symbols)? else {
        return Err(value_error(
            "a linear system with integer coefficients is expected".to_string(),
        ));
    };
    let n: usize = symbols.len();
    let (h, u) = hermite_normal_form_with_transform(&a);
    let mut y: Vec<Option<Int>> = vec![None; n];
    let mut rank: usize = 0;
    for i in 0..a.len() {
        let pivot: Option<usize> = (0..n).find(|j| !h[i][*j].is_zero() && (0..i).all(|r| h[r][*j].is_zero()));
        let mut residual: Int = b[i].clone();
        for j in 0..n {
            if let Some(value) = &y[j] {
                residual -= &h[i][j] * value;
            }
        }
        let Some(p) = pivot else {
            if !residual.is_zero() {
                return Ok(None);
            }
            continue;
        };
        if !(&residual % &h[i][p]).is_zero() {
            return Ok(None);
        }
        y[p] = Some(div_floor(&residual, &h[i][p]));
        rank += 1;
    }
    let parameters: Vec<Basic> = (0..n - rank)
        .map(|k| {
            Symbol::new(
                &format!("t{}", k),
                &[(crate::core::assumptions::Predicate::integer, true)],
            )
            .0
        })
        .collect();
    let mut free: usize = 0;
    let mut y_values: Vec<Basic> = Vec::new();
    for value in &y {
        match value {
            None => {
                y_values.push(parameters[free].clone());
                free += 1;
            }
            Some(v) => y_values.push(Basic::integer(v.clone())),
        }
    }
    let mut values: Vec<Basic> = Vec::new();
    for i in 0..n {
        let terms: Vec<Basic> = (0..n)
            .map(|j| crate::core::mul::mul(&[Basic::integer(u[i][j].clone()), y_values[j].clone()]))
            .collect::<PolyResult<_>>()?;
        values.push(crate::core::add::add(&terms)?);
    }
    Ok(Some((values, parameters)))
}

fn defect(a: &IntMatrix, v: &IntVector) -> IntVector {
    a.iter()
        .map(|row| row.iter().zip(v).fold(Int::ZERO, |s, (x, y)| s + x * y))
        .collect()
}

fn dominates(v: &IntVector, w: &IntVector) -> bool {
    v.iter().zip(w).all(|(a, b)| a >= b)
}

/// The minimal nonzero nonnegative solutions of `A x = 0` (Contejean and
/// Devie, `hilbert_basis`); beyond `limit` candidates an `Arithmetic`
/// error (Python's `RuntimeError`).
pub fn hilbert_basis(a: &IntMatrix, limit: usize) -> PolyResult<Vec<IntVector>> {
    let n: usize = a.first().map(|r| r.len()).unwrap_or(0);
    let mut solutions: Vec<IntVector> = Vec::new();
    let mut frontier: Vec<IntVector> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { Int::ONE } else { Int::ZERO }).collect())
        .collect();
    let mut seen: std::collections::HashSet<IntVector> = frontier.iter().cloned().collect();
    let mut examined: usize = 0;
    while !frontier.is_empty() {
        let mut next: Vec<IntVector> = Vec::new();
        for v in &frontier {
            examined += 1;
            if examined > limit {
                return Err(arithmetic_error(
                    "the completion did not finish within the limit".to_string(),
                ));
            }
            let d: IntVector = defect(a, v);
            if d.iter().all(|c| c.is_zero()) {
                if !solutions.iter().any(|s| dominates(v, s)) {
                    solutions.push(v.clone());
                }
                continue;
            }
            if solutions.iter().any(|s| dominates(v, s)) {
                continue;
            }
            for i in 0..n {
                let direction: Int = (0..a.len()).fold(Int::ZERO, |s, r| s + &d[r] * &a[r][i]);
                if direction < Int::ZERO {
                    let mut w: IntVector = v.clone();
                    w[i] += Int::ONE;
                    if seen.insert(w.clone()) {
                        next.push(w);
                    }
                }
            }
        }
        frontier = next;
    }
    solutions.sort();
    Ok(solutions)
}

/// The minimal nonnegative solutions of `A x = b` and the Hilbert basis of
/// `A x = 0` (`minimal_nonnegative_solutions`).
pub fn minimal_nonnegative_solutions(
    a: &IntMatrix,
    b: &IntVector,
    limit: usize,
) -> PolyResult<(Vec<IntVector>, Vec<IntVector>)> {
    let n: usize = a.first().map(|r| r.len()).unwrap_or(0);
    let homogenized: IntMatrix = a
        .iter()
        .zip(b)
        .map(|(row, c)| row.iter().cloned().chain(std::iter::once(-c.clone())).collect())
        .collect();
    let basis: Vec<IntVector> = hilbert_basis(&homogenized, limit)?;
    let mut particular: Vec<IntVector> = basis
        .iter()
        .filter(|v| v[n] == Int::ONE)
        .map(|v| v[..n].to_vec())
        .collect();
    let mut homogeneous: Vec<IntVector> = basis
        .iter()
        .filter(|v| v[n].is_zero())
        .map(|v| v[..n].to_vec())
        .collect();
    particular.sort();
    homogeneous.sort();
    Ok((particular, homogeneous))
}

// -- Cooper's algorithm

fn relation(op: RelOp, lhs: &Basic, rhs: &Basic) -> PolyResult<Basic> {
    match op {
        RelOp::Eq => eq(lhs, rhs),
        RelOp::Ne => ne(lhs, rhs),
        RelOp::Lt => lt(lhs, rhs),
        RelOp::Le => le(lhs, rhs),
        RelOp::Gt => gt(lhs, rhs),
        RelOp::Ge => ge(lhs, rhs),
    }
}

fn rel_op(e: &Basic) -> Option<RelOp> {
    match e.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}

/// `coefficient * x + rest`.
struct Linear {
    coefficient: Int,
    rest: Basic,
}

fn split_linear(e: &Basic, x: &Basic) -> PolyResult<Option<Linear>> {
    if !e.has(x) {
        return Ok(Some(Linear {
            coefficient: Int::ZERO,
            rest: e.clone(),
        }));
    }
    let poly = match crate::polys::polytools::Poly::new(e, std::slice::from_ref(x)) {
        Ok(p) => p,
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(other) => return Err(other),
    };
    if poly.degree(0)? != 1 {
        return Ok(None);
    }
    let c: Vec<Basic> = poly.all_coeffs()?;
    let Some(a) = integer_of(&c[0]) else { return Ok(None) };
    Ok(Some(Linear {
        coefficient: a,
        rest: c[1].clone(),
    }))
}

fn relations(formula: &Basic) -> Vec<Basic> {
    formula.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Relational(_))))
}

fn is_mod(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(Function::Mod))
}

/// `(k, e - r)` for `Eq(Mod(e, k), r)` or `Ne(Mod(e, k), r)`
/// (`_divisibility`).
fn divisibility(atom: &Basic) -> PolyResult<Option<(Int, Basic)>> {
    if !matches!(rel_op(atom), Some(RelOp::Eq | RelOp::Ne)) {
        return Ok(None);
    }
    let Some(residue) = integer_of(&atom.args()[1]) else {
        return Ok(None);
    };
    let mut lhs: Basic = atom.args()[0].clone();
    if residue.is_zero() && lhs.is_mul() && lhs.args().len() == 2 {
        if let Some(c) = integer_of(&lhs.args()[0]) {
            if !c.is_zero() {
                lhs = lhs.args()[1].clone();
            }
        }
    }
    if is_mod(&lhs) {
        let (e, k) = (&lhs.args()[0], &lhs.args()[1]);
        if let Some(k) = integer_of(k) {
            if k > Int::ZERO && residue >= Int::ZERO && residue < k {
                return Ok(Some((k, crate::core::add::sub(e, &Basic::integer(residue))?)));
            }
        }
    }
    Ok(None)
}

fn impossible_residue(atom: &Basic) -> Option<Basic> {
    let op: RelOp = rel_op(atom)?;
    if !matches!(op, RelOp::Eq | RelOp::Ne) || !is_mod(&atom.args()[0]) {
        return None;
    }
    let r: Int = integer_of(&atom.args()[1])?;
    let k: Int = integer_of(&atom.args()[0].args()[1])?;
    if k > Int::ZERO && !(r >= Int::ZERO && r < k) {
        return Some(boolean(op != RelOp::Eq));
    }
    None
}

fn divides(op: RelOp, k: &Int, e: &Basic) -> PolyResult<Basic> {
    if k.is_one() {
        return Ok(boolean(op == RelOp::Eq));
    }
    relation(
        op,
        &crate::core::modulo::modulo(e, &Basic::integer(k.clone()))?,
        &Basic::zero(),
    )
}

fn substitute(formula: &Basic, replacements: &[(Basic, Basic)]) -> PolyResult<Basic> {
    if let Some((_, v)) = replacements.iter().find(|(k, _)| k == formula) {
        return Ok(v.clone());
    }
    if matches!(
        formula.head(),
        Head::And | Head::Or | Head::Not | Head::Implies | Head::Equivalent | Head::Xor | Head::Ite
    ) {
        let args: Vec<Basic> = formula
            .args()
            .iter()
            .map(|a| substitute(a, replacements))
            .collect::<PolyResult<_>>()?;
        return crate::core::function::rebuild_evaluated(formula, args);
    }
    Ok(formula.clone())
}

fn canonical_divisibilities(formula: &Basic) -> PolyResult<Basic> {
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    for atom in relations(formula) {
        if let Some((k, e)) = divisibility(&atom)? {
            replacements.push((atom.clone(), divides(rel_op(&atom).expect("a relation"), &k, &e)?));
            continue;
        }
        if let Some(value) = impossible_residue(&atom) {
            replacements.push((atom, value));
        }
    }
    substitute(formula, &replacements)
}

fn integer_linear(e: &Basic, integers: &[Basic]) -> PolyResult<bool> {
    let symbols = e.free_symbols();
    if !symbols.iter().all(|s| integers.contains(s)) {
        return Ok(false);
    }
    if symbols.is_empty() {
        return Ok(integer_of(e).is_some()
            && matches!(
                e.head(),
                Head::Rational {
                    integer_class: true,
                    ..
                }
            ));
    }
    let mut gens: Vec<Basic> = symbols.into_iter().collect();
    gens.sort_by_key(|s| s.to_string());
    let poly = match crate::polys::polytools::Poly::new(e, &gens) {
        Ok(p) => p,
        Err(PolyError::Polys(..)) => return Ok(false),
        Err(other) => return Err(other),
    };
    Ok(poly.total_degree() <= 1
        && poly.coeffs()?.iter().all(|c| {
            matches!(
                c.head(),
                Head::Rational {
                    integer_class: true,
                    ..
                }
            )
        }))
}

/// Whether the formula is Presburger arithmetic in the integer variables
/// (`is_presburger`).
pub fn is_presburger(formula: &Basic, integers: &[Basic]) -> PolyResult<bool> {
    for atom in relations(formula) {
        if let Some((_, e)) = divisibility(&atom)? {
            if !integer_linear(&e, integers)? {
                return Ok(false);
            }
            continue;
        }
        if impossible_residue(&atom).is_some() {
            continue;
        }
        if !integer_linear(&crate::core::add::sub(&atom.args()[0], &atom.args()[1])?, integers)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn lcm(values: &[Int]) -> Int {
    let mut result: Int = Int::ONE;
    for v in values {
        let g: Int = crate::core::intfunc::igcd(&[result.clone(), v.clone()]);
        result = div_floor(&(&result * v), &g);
    }
    result
}

fn abs_int(v: &Int) -> Int {
    if *v < Int::ZERO {
        -v.clone()
    } else {
        v.clone()
    }
}

fn linear_of(atom: &Basic, x: &Basic) -> PolyResult<Linear> {
    let e: Basic = match divisibility(atom)? {
        Some((_, e)) => e,
        None => crate::core::add::sub(&atom.args()[0], &atom.args()[1])?,
    };
    split_linear(&e, x)?.ok_or_else(|| value_error(format!("{} is not linear in {}", atom, x)))
}

fn normalize_coefficients(formula: &Basic, x: &Basic) -> PolyResult<(Basic, Int)> {
    let mut parts: Vec<(Basic, Linear)> = Vec::new();
    let mut coefficients: Vec<Int> = Vec::new();
    for atom in relations(formula) {
        let lin: Linear = linear_of(&atom, x)?;
        if !lin.coefficient.is_zero() {
            coefficients.push(abs_int(&lin.coefficient));
        }
        parts.push((atom, lin));
    }
    let m: Int = if coefficients.is_empty() {
        Int::ONE
    } else {
        lcm(&coefficients)
    };
    let mut replacements: Vec<(Basic, Basic)> = Vec::new();
    for (atom, lin) in &parts {
        if lin.coefficient.is_zero() {
            continue;
        }
        let factor: Int = div_floor(&m, &abs_int(&lin.coefficient));
        let sign: i64 = if lin.coefficient > Int::ZERO { 1 } else { -1 };
        let new_rest: Basic = crate::core::mul::mul(&[Basic::integer(factor.clone()), lin.rest.clone()])?;
        let signed_x: Basic =
            crate::core::add::add(&[crate::core::mul::mul(&[Basic::int(sign), x.clone()])?, new_rest])?;
        let op: RelOp = rel_op(atom).expect("a relation");
        let value: Basic = match divisibility(atom)? {
            Some((k, _)) => divides(op, &(k * &factor), &signed_x)?,
            None => relation(op, &signed_x, &Basic::zero())?,
        };
        replacements.push((atom.clone(), value));
    }
    Ok((substitute(formula, &replacements)?, m))
}

fn subs(e: &Basic, x: &Basic, v: &Basic) -> PolyResult<Basic> {
    crate::core::subs::subs(e, x, v)
}

/// `exists x in Z` of a Presburger formula, quantifier free (Cooper,
/// `cooper`).
pub fn cooper(formula: &Basic, x: &Basic) -> PolyResult<Basic> {
    if !crate::core::logic::is_boolean(formula) {
        return Err(PolyError::Type(format!(
            "a Boolean formula is expected, got {}",
            formula
        )));
    }
    let formula: Basic = canonical_divisibilities(&to_nnf(formula))?;
    if !formula.has(x) {
        return Ok(formula);
    }
    let (mut normalized, m) = normalize_coefficients(&formula, x)?;
    if !m.is_one() {
        normalized = and_(&[
            normalized,
            eq(
                &crate::core::modulo::modulo(x, &Basic::integer(m.clone()))?,
                &Basic::zero(),
            )?,
        ]);
    }
    let mut parts: Vec<(Basic, Linear, bool)> = Vec::new();
    let mut moduli: Vec<Int> = Vec::new();
    let mut thresholds: Vec<Basic> = Vec::new();
    let one: Basic = Basic::one();
    for atom in relations(&normalized) {
        let divisible: Option<(Int, Basic)> = divisibility(&atom)?;
        let lin: Linear = linear_of(&atom, x)?;
        if let Some((k, _)) = &divisible {
            if !lin.coefficient.is_zero() {
                moduli.push(k.clone());
            }
            parts.push((atom, lin, true));
            continue;
        }
        if lin.coefficient.is_zero() {
            parts.push((atom, lin, false));
            continue;
        }
        let op: RelOp = rel_op(&atom).expect("a relation");
        let c: Basic = if lin.coefficient.is_one() {
            let c: Basic = crate::core::mul::neg(&lin.rest)?;
            match op {
                RelOp::Gt => thresholds.push(c.clone()),
                RelOp::Ge => thresholds.push(crate::core::add::sub(&c, &one)?),
                _ => {}
            }
            c
        } else {
            let c: Basic = lin.rest.clone();
            match op {
                RelOp::Lt => thresholds.push(c.clone()),
                RelOp::Le => thresholds.push(crate::core::add::sub(&c, &one)?),
                _ => {}
            }
            c
        };
        match op {
            RelOp::Eq => thresholds.push(crate::core::add::sub(&c, &one)?),
            RelOp::Ne => thresholds.push(c),
            _ => {}
        }
        parts.push((atom, lin, false));
    }
    let delta: Int = if moduli.is_empty() { Int::ONE } else { lcm(&moduli) };
    let delta_count: i64 = i64::try_from(&delta).map_err(|_| value_error("a huge modulus".to_string()))?;
    let mut disjuncts: Vec<Basic> = Vec::new();
    for j in 1..=delta_count {
        let mut replacements: Vec<(Basic, Basic)> = Vec::new();
        for (atom, lin, divisible) in &parts {
            if lin.coefficient.is_zero() {
                continue;
            }
            let value: Basic = if *divisible {
                subs(atom, x, &Basic::int(j))?
            } else {
                let positive: bool = lin.coefficient.is_one();
                match rel_op(atom).expect("a relation") {
                    RelOp::Eq => boolean(false),
                    RelOp::Ne => boolean(true),
                    RelOp::Lt | RelOp::Le => boolean(positive),
                    RelOp::Gt | RelOp::Ge => boolean(!positive),
                }
            };
            replacements.push((atom.clone(), value));
        }
        disjuncts.push(substitute(&normalized, &replacements)?);
    }
    for bound in &thresholds {
        for j in 1..=delta_count {
            disjuncts.push(subs(
                &normalized,
                x,
                &crate::core::add::add(&[bound.clone(), Basic::int(j)])?,
            )?);
        }
    }
    clean(&or_(&disjuncts))
}

/// `(x, e)` when the disjunct pins `x` to `e` (`_point`).
fn point(disjunct: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let conjuncts: Vec<Basic> = if matches!(disjunct.head(), Head::And) {
        disjunct.args().to_vec()
    } else {
        vec![disjunct.clone()]
    };
    for atom in &conjuncts {
        if rel_op(atom) != Some(RelOp::Eq) || is_mod(&atom.args()[0]) {
            continue;
        }
        let e: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
        let mut symbols: Vec<Basic> = e.free_symbols().into_iter().collect();
        symbols.sort_by_key(|s| s.to_string());
        for x in symbols {
            if let Some(lin) = split_linear(&e, &x)? {
                if lin.coefficient.is_one() || lin.coefficient == Int::NEG_ONE {
                    let value: Basic = crate::core::mul::mul(&[
                        crate::core::mul::neg(&lin.rest)?,
                        Basic::integer(lin.coefficient.clone()),
                    ])?;
                    return Ok(Some((x, value)));
                }
            }
        }
    }
    Ok(None)
}

fn drop_covered(formula: &Basic) -> PolyResult<Basic> {
    if !matches!(formula.head(), Head::Or) {
        return Ok(formula.clone());
    }
    let disjuncts: Vec<Basic> = formula.args().to_vec();
    let mut kept: Vec<Basic> = Vec::new();
    for (i, d) in disjuncts.iter().enumerate() {
        if let Some((x, v)) = point(d)? {
            let mut others: Vec<Basic> = kept.clone();
            others.extend(disjuncts[i + 1..].iter().cloned());
            if matches!(subs(&or_(&others), &x, &v)?.head(), Head::BooleanTrue) {
                continue;
            }
        }
        kept.push(d.clone());
    }
    Ok(or_(&kept))
}

fn clean(formula: &Basic) -> PolyResult<Basic> {
    let formula: Basic = canonical_divisibilities(formula)?;
    if matches!(formula.head(), Head::BooleanTrue | Head::BooleanFalse) {
        return Ok(formula);
    }
    let result: Basic = match crate::logic::boolalg::simplify_logic(&formula, None, false, None) {
        Ok(r) => r,
        Err(PolyError::Value(_)) | Err(PolyError::Type(_)) => formula,
        Err(e) => return Err(e),
    };
    with_sides(&drop_covered(&result)?)
}

/// Eliminate all the quantifiers of a Presburger formula, innermost first
/// (`presburger_quantifier_elimination`).
pub fn presburger_quantifier_elimination(formula: &Basic, prefix: &Prefix) -> PolyResult<Basic> {
    let mut current: Basic = formula.clone();
    for (kind, v) in prefix.iter().rev() {
        current = match kind {
            QuantifierKind::Exists => cooper(&current, v)?,
            QuantifierKind::ForAll => clean(&not_(&cooper(&not_(&current), v)?))?,
        };
    }
    Ok(current)
}
