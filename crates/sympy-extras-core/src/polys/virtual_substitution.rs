//! Quantifier elimination by virtual substitution (Loos and Weispfenning),
//! ported from `sympy_extras.polys.virtual_substitution`: for a formula
//! linear in the quantified variable, `exists x` is the disjunction of the
//! formula at the test points `-oo`, the roots of the equations and weak
//! inequalities and the roots plus an infinitesimal for the strict ones,
//! each atom replaced by the polynomial condition of its truth there; the
//! quadratic case adds the roots `(-b ± sqrt(b**2 - 4ac))/(2a)` with their
//! guards. The results are normalized and simplified by `simplify_logic`.
//!
//! The relations are visited in the order of their first occurrence (the
//! Python iterates a set); the results are equivalent, and equal once
//! simplified in the cases tested.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::assumptions::quantifiers::{Prefix, QuantifierKind};
use crate::core::basic::{Basic, Head, RelOp};
use crate::core::logic::{and_, not_, or_};
use crate::core::relational::{eq, ge, gt, le, lt, ne};
use crate::error::{value_error, PolyError, PolyResult, PolysErrorKind};

fn zero() -> Basic {
    Basic::zero()
}

fn kind(atom: &Basic) -> RelOp {
    match atom.head() {
        Head::Relational(op) => *op,
        _ => unreachable!("a relation"),
    }
}

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

fn rel0(op: RelOp, lhs: &Basic) -> PolyResult<Basic> {
    relation(op, lhs, &zero())
}

fn atoms(formula: &Basic) -> Vec<Basic> {
    formula.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Relational(_))))
}

fn m(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}

fn s(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    crate::core::add::sub(a, b)
}

fn neg(a: &Basic) -> PolyResult<Basic> {
    crate::core::mul::neg(a)
}

fn expand(e: &Basic) -> PolyResult<Basic> {
    crate::core::function::expand_default(e)
}

/// The coefficients of `lhs - rhs` in `x`, highest degree first (`[0]`
/// for zero), `None` when it is not a polynomial (`_coefficients`).
fn coefficients(atom: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let p: Basic = s(&atom.args()[0], &atom.args()[1])?;
    if !p.has(x) {
        return Ok(Some(vec![p]));
    }
    match crate::polys::polytools::Poly::new(&p, std::slice::from_ref(x)) {
        Ok(poly) => return Ok(Some(poly.all_coeffs()?)),
        Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) | Err(PolyError::NotImplemented(_)) => {}
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(e) => return Err(e),
    }
    let decomposed = match crate::polys::polytools::decompose_with_gens(&p, std::slice::from_ref(x)) {
        Ok(d) => d,
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut terms: Vec<(u32, Basic)> = Vec::new();
    for (monomial, cs) in &decomposed {
        let c: Basic = expand(&crate::core::add::add(cs)?)?;
        if c != zero() {
            terms.push((monomial[0], c));
        }
    }
    if terms.is_empty() {
        return Ok(Some(vec![zero()]));
    }
    let top: u32 = terms.iter().map(|(k, _)| *k).max().expect("a term");
    Ok(Some(
        (0..=top)
            .rev()
            .map(|k| {
                terms
                    .iter()
                    .find(|(d, _)| *d == k)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(zero)
            })
            .collect(),
    ))
}

fn degree(coefficients: &[Basic]) -> i64 {
    if coefficients.len() == 1 && coefficients[0] == zero() {
        return -1;
    }
    coefficients.len() as i64 - 1
}

fn linear_coefficients(atom: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let p: Basic = s(&atom.args()[0], &atom.args()[1])?;
    if !p.has(x) {
        return Ok(Some((zero(), p)));
    }
    let Some(c) = coefficients(atom, x)? else {
        return Ok(None);
    };
    if degree(&c) != 1 {
        return Ok(None);
    }
    Ok(Some((c[0].clone(), c[1].clone())))
}

/// Whether every relation of the formula is linear in `x`.
pub fn is_linear_in(formula: &Basic, x: &Basic) -> PolyResult<bool> {
    for a in atoms(formula) {
        if linear_coefficients(&a, x)?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn at_root(atom: &Basic, a: &Basic, b: &Basic, c: &Basic, d: &Basic) -> PolyResult<Basic> {
    let numerator: Basic = s(&m(a, d)?, &m(b, c)?)?;
    let op: RelOp = kind(atom);
    if matches!(op, RelOp::Eq | RelOp::Ne) {
        return rel0(op, &numerator);
    }
    rel0(op, &m(a, &numerator)?)
}

fn at_root_plus_epsilon(atom: &Basic, a: &Basic, b: &Basic, c: &Basic, d: &Basic) -> PolyResult<Basic> {
    let numerator: Basic = s(&m(a, d)?, &m(b, c)?)?;
    let value_sign: Basic = m(a, &numerator)?;
    let z: Basic = zero();
    Ok(match kind(atom) {
        RelOp::Eq => and_(&[eq(&numerator, &z)?, eq(c, &z)?]),
        RelOp::Ne => or_(&[ne(&numerator, &z)?, ne(c, &z)?]),
        RelOp::Lt => or_(&[lt(&value_sign, &z)?, and_(&[eq(&numerator, &z)?, lt(c, &z)?])]),
        RelOp::Le => or_(&[lt(&value_sign, &z)?, and_(&[eq(&numerator, &z)?, le(c, &z)?])]),
        RelOp::Gt => or_(&[gt(&value_sign, &z)?, and_(&[eq(&numerator, &z)?, gt(c, &z)?])]),
        RelOp::Ge => or_(&[gt(&value_sign, &z)?, and_(&[eq(&numerator, &z)?, ge(c, &z)?])]),
    })
}

fn at_minus_infinity(atom: &Basic, c: &Basic, d: &Basic) -> PolyResult<Basic> {
    let z: Basic = zero();
    Ok(match kind(atom) {
        RelOp::Eq => and_(&[eq(c, &z)?, eq(d, &z)?]),
        RelOp::Ne => or_(&[ne(c, &z)?, ne(d, &z)?]),
        RelOp::Lt => or_(&[gt(c, &z)?, and_(&[eq(c, &z)?, lt(d, &z)?])]),
        RelOp::Le => or_(&[gt(c, &z)?, and_(&[eq(c, &z)?, le(d, &z)?])]),
        RelOp::Gt => or_(&[lt(c, &z)?, and_(&[eq(c, &z)?, gt(d, &z)?])]),
        RelOp::Ge => or_(&[lt(c, &z)?, and_(&[eq(c, &z)?, ge(d, &z)?])]),
    })
}

fn is_connective(f: &Basic) -> bool {
    matches!(
        f.head(),
        Head::And | Head::Or | Head::Not | Head::Implies | Head::Equivalent | Head::Xor | Head::Ite
    )
}

/// The relations replaced, the connectives evaluated (`_substitute`).
fn substitute(formula: &Basic, replacements: &[(Basic, Basic)]) -> PolyResult<Basic> {
    if matches!(formula.head(), Head::Relational(_)) {
        return Ok(replacements
            .iter()
            .find(|(k, _)| k == formula)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| formula.clone()));
    }
    if is_connective(formula) {
        let args: Vec<Basic> = formula
            .args()
            .iter()
            .map(|a| substitute(a, replacements))
            .collect::<PolyResult<_>>()?;
        return crate::core::function::rebuild_evaluated(formula, args);
    }
    Ok(formula.clone())
}

fn combinations(items: &[Basic], k: usize) -> Vec<Vec<Basic>> {
    let n: usize = items.len();
    let mut result: Vec<Vec<Basic>> = Vec::new();
    if k > n {
        return result;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        result.push(idx.iter().map(|i| items[*i].clone()).collect());
        let Some(i) = (0..k).rev().find(|i| idx[*i] != *i + n - k) else {
            return result;
        };
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

fn parity_clauses(items: &[Basic], start: usize) -> Vec<Basic> {
    let mut clauses: Vec<Basic> = Vec::new();
    let mut i: usize = start;
    while i <= items.len() {
        for negated in combinations(items, i) {
            clauses.push(or_(&items
                .iter()
                .map(|x| if negated.contains(x) { not_(x) } else { x.clone() })
                .collect::<Vec<Basic>>()));
        }
        i += 2;
    }
    clauses
}

/// The negation normal form (SymPy's `to_nnf(simplify=False)`).
pub fn to_nnf(formula: &Basic) -> Basic {
    let args: &[Basic] = formula.args();
    match formula.head() {
        Head::And => and_(&args.iter().map(to_nnf).collect::<Vec<Basic>>()),
        Head::Or => or_(&args.iter().map(to_nnf).collect::<Vec<Basic>>()),
        Head::Implies => or_(&[to_nnf(&not_(&args[0])), to_nnf(&args[1])]),
        Head::Equivalent => {
            let mut clauses: Vec<Basic> = args.windows(2).map(|w| or_(&[not_(&w[0]), w[1].clone()])).collect();
            clauses.push(or_(&[not_(&args[args.len() - 1]), args[0].clone()]));
            and_(&clauses.iter().map(to_nnf).collect::<Vec<Basic>>())
        }
        Head::Xor => and_(&parity_clauses(args, 0).iter().map(to_nnf).collect::<Vec<Basic>>()),
        Head::Ite => and_(&[
            to_nnf(&or_(&[not_(&args[0]), args[1].clone()])),
            to_nnf(&or_(&[args[0].clone(), args[2].clone()])),
        ]),
        Head::Not => negation_to_nnf(&args[0]),
        _ => formula.clone(),
    }
}

fn negation_to_nnf(inner: &Basic) -> Basic {
    let args: &[Basic] = inner.args();
    match inner.head() {
        Head::And => or_(&args.iter().map(|a| to_nnf(&not_(a))).collect::<Vec<Basic>>()),
        Head::Or => and_(&args.iter().map(|a| to_nnf(&not_(a))).collect::<Vec<Basic>>()),
        Head::Implies => and_(&[to_nnf(&args[0]), to_nnf(&not_(&args[1]))]),
        Head::Equivalent => and_(&[
            to_nnf(&or_(args)),
            to_nnf(&or_(&args.iter().map(not_).collect::<Vec<Basic>>())),
        ]),
        Head::Xor => and_(&parity_clauses(args, 1).iter().map(to_nnf).collect::<Vec<Basic>>()),
        Head::Ite => and_(&[
            to_nnf(&or_(&[args[0].clone(), not_(&args[2])])),
            to_nnf(&or_(&[not_(&args[0]), not_(&args[1])])),
        ]),
        Head::Not => to_nnf(&args[0]),
        _ => not_(inner),
    }
}

fn pinned_value(formula: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let conjuncts: Vec<Basic> = if matches!(formula.head(), Head::And) {
        formula.args().to_vec()
    } else {
        vec![formula.clone()]
    };
    for atom in &conjuncts {
        if !matches!(atom.head(), Head::Relational(RelOp::Eq)) {
            continue;
        }
        let Some((a, b)) = linear_coefficients(atom, x)? else {
            continue;
        };
        if crate::core::assumptions::is_number(&a) && a != zero() {
            return Ok(Some(neg(&crate::core::mul::div(&b, &a)?)?));
        }
    }
    Ok(None)
}

fn sides(atom: &Basic) -> PolyResult<Basic> {
    if atom.args()[1] != zero() {
        return Ok(atom.clone());
    }
    let lhs: &Basic = &atom.args()[0];
    let terms: Vec<Basic> = if lhs.is_add() {
        lhs.args().to_vec()
    } else {
        vec![lhs.clone()]
    };
    let mut positive: Vec<Basic> = Vec::new();
    let mut negative: Vec<Basic> = Vec::new();
    for term in terms {
        if crate::core::printing::could_extract_minus_sign(&term) {
            negative.push(neg(&term)?);
        } else {
            positive.push(term);
        }
    }
    if negative.is_empty() && positive.len() > 1 {
        if let Some(i) = positive.iter().position(|t| t.is_number_class()) {
            let constant: Basic = positive.remove(i);
            negative.push(neg(&constant)?);
        }
    }
    if negative.is_empty() || positive.is_empty() {
        return Ok(atom.clone());
    }
    relation(
        kind(atom),
        &crate::core::add::add(&positive)?,
        &crate::core::add::add(&negative)?,
    )
}

fn is_boolean_function(f: &Basic) -> bool {
    is_connective(f)
}

fn clean(formula: &Basic) -> PolyResult<Basic> {
    let result: Basic = crate::assumptions::facts::normalize(formula)?;
    if matches!(result.head(), Head::BooleanTrue | Head::BooleanFalse) {
        return Ok(result);
    }
    if !is_boolean_function(&result) {
        return with_sides(&result);
    }
    let simplified: Basic = match crate::logic::boolalg::simplify_logic(&result, None, false, None) {
        Ok(s) => s,
        Err(PolyError::Value(_)) | Err(PolyError::Type(_)) => result,
        Err(e) => return Err(e),
    };
    with_sides(&simplified)
}

/// Every relation `p - q rel 0` written as `p rel q` (`with_sides`).
pub fn with_sides(formula: &Basic) -> PolyResult<Basic> {
    let replacements: Vec<(Basic, Basic)> = atoms(formula)
        .iter()
        .map(|a| Ok((a.clone(), sides(a)?)))
        .collect::<PolyResult<_>>()?;
    substitute(formula, &replacements)
}

fn xreplace_one(formula: &Basic, x: &Basic, v: &Basic) -> PolyResult<Basic> {
    let rule: HashMap<Basic, Basic> = HashMap::from([(x.clone(), v.clone())]);
    crate::core::function::xreplace_evaluated(formula, &rule)
}

/// `exists x` of a formula linear in `x`, quantifier free
/// (`eliminate_linear`).
pub fn eliminate_linear(formula: &Basic, x: &Basic) -> PolyResult<Basic> {
    let formula: Basic = to_nnf(formula);
    if !formula.has(x) {
        return Ok(formula);
    }
    if let Some(pinned) = pinned_value(&formula, x)? {
        return clean(&xreplace_one(&formula, x, &pinned)?);
    }
    let mut coefficients: Vec<(Basic, (Basic, Basic))> = Vec::new();
    for atom in atoms(&formula) {
        let Some(pair) = linear_coefficients(&atom, x)? else {
            return Err(value_error(format!("{} is not linear in {}", atom, x)));
        };
        coefficients.push((atom, pair));
    }
    let mut disjuncts: Vec<Basic> = Vec::new();
    let at_infinity: Vec<(Basic, Basic)> = coefficients
        .iter()
        .map(|(atom, (c, d))| Ok((atom.clone(), at_minus_infinity(atom, c, d)?)))
        .collect::<PolyResult<_>>()?;
    disjuncts.push(substitute(&formula, &at_infinity)?);
    for (atom, (a, b)) in &coefficients {
        if *a == zero() {
            continue;
        }
        let shifted: bool = matches!(kind(atom), RelOp::Lt | RelOp::Gt | RelOp::Ne);
        let mut replacements: Vec<(Basic, Basic)> = Vec::new();
        for (other, (c, d)) in &coefficients {
            let value: Basic = if shifted {
                at_root_plus_epsilon(other, a, b, c, d)?
            } else {
                at_root(other, a, b, c, d)?
            };
            replacements.push((other.clone(), value));
        }
        disjuncts.push(and_(&[ne(a, &zero())?, substitute(&formula, &replacements)?]));
    }
    clean(&or_(&disjuncts))
}

fn eliminate_for(
    kind: QuantifierKind,
    current: &Basic,
    v: &Basic,
    eliminate: fn(&Basic, &Basic) -> PolyResult<Basic>,
) -> PolyResult<Basic> {
    match kind {
        QuantifierKind::Exists => eliminate(current, v),
        QuantifierKind::ForAll => clean(&not_(&eliminate(&not_(current), v)?)),
    }
}

/// Eliminate, innermost first, the quantified variables in which the
/// formula is linear (`linear_quantifier_elimination`).
pub fn linear_quantifier_elimination(formula: &Basic, prefix: &Prefix) -> PolyResult<(Basic, Prefix)> {
    let mut remaining: Prefix = prefix.clone();
    let mut current: Basic = formula.clone();
    while let Some((k, v)) = remaining.last().cloned() {
        if !is_linear_in(&current, &v)? {
            break;
        }
        remaining.pop();
        current = eliminate_for(k, &current, &v, eliminate_linear)?;
    }
    Ok((current, remaining))
}

// -- the quadratic case

fn quadratic_coefficients(atom: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    let p: Basic = s(&atom.args()[0], &atom.args()[1])?;
    if !p.has(x) {
        return Ok(Some((zero(), zero(), p)));
    }
    let Some(mut c) = coefficients(atom, x)? else {
        return Ok(None);
    };
    if degree(&c) > 2 {
        return Ok(None);
    }
    while c.len() < 3 {
        c.insert(0, zero());
    }
    Ok(Some((c[0].clone(), c[1].clone(), c[2].clone())))
}

/// Whether every relation has degree at most two in `x`.
pub fn is_quadratic_in(formula: &Basic, x: &Basic) -> PolyResult<bool> {
    for a in atoms(formula) {
        if quadratic_coefficients(&a, x)?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A test point `(p + q*sqrt(r))/s` with its guard (`_Root`).
struct Root {
    p: Basic,
    q: Basic,
    r: Basic,
    s: Basic,
    guard: Basic,
}

fn sq(a: &Basic) -> PolyResult<Basic> {
    crate::core::power::pow(a, &Basic::int(2))
}

fn parts(a2: &Basic, a1: &Basic, a0: &Basic, root: &Root) -> PolyResult<(Basic, Basic)> {
    let (p, q, r, s_) = (&root.p, &root.q, &root.r, &root.s);
    let a_part: Basic = crate::core::add::add(&[
        m(a2, &crate::core::add::add(&[sq(p)?, m(&sq(q)?, r)?])?)?,
        m(&m(a1, s_)?, p)?,
        m(a0, &sq(s_)?)?,
    ])?;
    let b_part: Basic = crate::core::add::add(&[m(&m(&m(&Basic::int(2), a2)?, p)?, q)?, m(&m(a1, s_)?, q)?])?;
    Ok((expand(&a_part)?, expand(&b_part)?))
}

fn sign_with_sqrt(op: RelOp, a: &Basic, b: &Basic, r: &Basic) -> PolyResult<Basic> {
    if *b == zero() {
        return rel0(op, a);
    }
    let d: Basic = expand(&s(&sq(a)?, &m(&sq(b)?, r)?)?)?;
    let z: Basic = zero();
    Ok(match op {
        RelOp::Eq => or_(&[
            and_(&[eq(a, &z)?, eq(b, &z)?]),
            and_(&[le(&m(a, b)?, &z)?, eq(&d, &z)?]),
        ]),
        RelOp::Ne => not_(&sign_with_sqrt(RelOp::Eq, a, b, r)?),
        RelOp::Lt => or_(&[
            and_(&[le(b, &z)?, or_(&[lt(a, &z)?, lt(&d, &z)?])]),
            and_(&[gt(b, &z)?, lt(a, &z)?, gt(&d, &z)?]),
        ]),
        RelOp::Le => or_(&[
            and_(&[le(b, &z)?, or_(&[le(a, &z)?, le(&d, &z)?])]),
            and_(&[gt(b, &z)?, le(a, &z)?, ge(&d, &z)?]),
        ]),
        RelOp::Gt => sign_with_sqrt(RelOp::Lt, &neg(a)?, &neg(b)?, r)?,
        RelOp::Ge => sign_with_sqrt(RelOp::Le, &neg(a)?, &neg(b)?, r)?,
    })
}

fn at_algebraic(atom: &Basic, c: &(Basic, Basic, Basic), root: &Root) -> PolyResult<Basic> {
    let (a, b) = parts(&c.0, &c.1, &c.2, root)?;
    sign_with_sqrt(kind(atom), &a, &b, &root.r)
}

fn at_algebraic_plus_epsilon(atom: &Basic, c: &(Basic, Basic, Basic), root: &Root) -> PolyResult<Basic> {
    let (a2, a1, a0) = c;
    let op: RelOp = kind(atom);
    let two: Basic = Basic::int(2);
    let derivatives: [(Basic, Basic, Basic); 3] = [
        (a2.clone(), a1.clone(), a0.clone()),
        (zero(), m(&two, a2)?, a1.clone()),
        (zero(), zero(), m(&two, a2)?),
    ];
    let values: Vec<(Basic, Basic)> = derivatives
        .iter()
        .map(|(d2, d1, d0)| parts(d2, d1, d0, root))
        .collect::<PolyResult<_>>()?;
    let zeros: Vec<Basic> = values
        .iter()
        .map(|(a, b)| sign_with_sqrt(RelOp::Eq, a, b, &root.r))
        .collect::<PolyResult<_>>()?;
    if op == RelOp::Eq {
        return Ok(and_(&zeros));
    }
    if op == RelOp::Ne {
        return Ok(not_(&and_(&zeros)));
    }
    let strict: RelOp = if matches!(op, RelOp::Lt | RelOp::Le) {
        RelOp::Lt
    } else {
        RelOp::Gt
    };
    let mut cases: Vec<Basic> = Vec::new();
    for (i, (a, b)) in values.iter().enumerate() {
        let mut parts_: Vec<Basic> = zeros[..i].to_vec();
        parts_.push(sign_with_sqrt(strict, a, b, &root.r)?);
        cases.push(and_(&parts_));
    }
    if matches!(op, RelOp::Le | RelOp::Ge) {
        cases.push(and_(&zeros));
    }
    Ok(or_(&cases))
}

fn at_minus_infinity_quadratic(atom: &Basic, c: &(Basic, Basic, Basic)) -> PolyResult<Basic> {
    let (a, b, cc) = c;
    let z: Basic = zero();
    let op: RelOp = kind(atom);
    if op == RelOp::Eq {
        return Ok(and_(&[eq(a, &z)?, eq(b, &z)?, eq(cc, &z)?]));
    }
    if op == RelOp::Ne {
        return Ok(or_(&[ne(a, &z)?, ne(b, &z)?, ne(cc, &z)?]));
    }
    let strict: RelOp = if matches!(op, RelOp::Lt | RelOp::Le) {
        RelOp::Lt
    } else {
        RelOp::Gt
    };
    let reverse: RelOp = if strict == RelOp::Lt { RelOp::Gt } else { RelOp::Lt };
    Ok(or_(&[
        rel0(strict, a)?,
        and_(&[eq(a, &z)?, rel0(reverse, b)?]),
        and_(&[eq(a, &z)?, eq(b, &z)?, rel0(op, cc)?]),
    ]))
}

/// `exists x` of a formula of degree at most two in `x`
/// (`eliminate_quadratic`).
pub fn eliminate_quadratic(formula: &Basic, x: &Basic) -> PolyResult<Basic> {
    let formula: Basic = to_nnf(formula);
    if !formula.has(x) {
        return Ok(formula);
    }
    if let Some(pinned) = pinned_value(&formula, x)? {
        return clean(&xreplace_one(&formula, x, &pinned)?);
    }
    let mut coefficients: Vec<(Basic, (Basic, Basic, Basic))> = Vec::new();
    for atom in atoms(&formula) {
        let Some(triple) = quadratic_coefficients(&atom, x)? else {
            return Err(value_error(format!("{} has degree more than 2 in {}", atom, x)));
        };
        coefficients.push((atom, triple));
    }
    let at_infinity: Vec<(Basic, Basic)> = coefficients
        .iter()
        .map(|(atom, t)| Ok((atom.clone(), at_minus_infinity_quadratic(atom, t)?)))
        .collect::<PolyResult<_>>()?;
    let mut disjuncts: Vec<Basic> = vec![substitute(&formula, &at_infinity)?];
    let z: Basic = zero();
    for (atom, (a, b, c)) in &coefficients {
        if *a == z && *b == z {
            continue;
        }
        let shifted: bool = matches!(kind(atom), RelOp::Lt | RelOp::Gt | RelOp::Ne);
        let mut roots: Vec<Root> = Vec::new();
        if *b != z || *a != z {
            roots.push(Root {
                p: neg(c)?,
                q: z.clone(),
                r: z.clone(),
                s: b.clone(),
                guard: and_(&[eq(a, &z)?, ne(b, &z)?]),
            });
        }
        if *a != z {
            let r: Basic = expand(&s(&sq(b)?, &m(&m(&Basic::int(4), a)?, c)?)?)?;
            for sign in [Basic::one(), Basic::negative_one()] {
                roots.push(Root {
                    p: neg(b)?,
                    q: sign,
                    r: r.clone(),
                    s: m(&Basic::int(2), a)?,
                    guard: and_(&[ne(a, &z)?, ge(&r, &z)?]),
                });
            }
        }
        for root in &roots {
            let mut replacements: Vec<(Basic, Basic)> = Vec::new();
            for (other, triple) in &coefficients {
                let value: Basic = if shifted {
                    at_algebraic_plus_epsilon(other, triple, root)?
                } else {
                    at_algebraic(other, triple, root)?
                };
                replacements.push((other.clone(), value));
            }
            disjuncts.push(and_(&[root.guard.clone(), substitute(&formula, &replacements)?]));
        }
    }
    clean(&or_(&disjuncts))
}

/// Eliminate, innermost first, the quantified variables in which the
/// formula is linear or quadratic (`virtual_substitution_elimination`).
pub fn virtual_substitution_elimination(
    formula: &Basic,
    prefix: &Prefix,
    max_atoms: usize,
    max_free: usize,
) -> PolyResult<(Basic, Prefix)> {
    let mut remaining: Prefix = prefix.clone();
    let mut current: Basic = formula.clone();
    let bound: Vec<Basic> = prefix.iter().map(|(_, v)| v.clone()).collect();
    let free: usize = current.free_symbols().iter().filter(|s| !bound.contains(s)).count();
    while let Some((k, v)) = remaining.last().cloned() {
        let eliminate: fn(&Basic, &Basic) -> PolyResult<Basic> = if is_linear_in(&current, &v)? {
            eliminate_linear
        } else if is_quadratic_in(&current, &v)? && atoms(&current).len() <= max_atoms && free <= max_free {
            eliminate_quadratic
        } else {
            break;
        };
        remaining.pop();
        current = eliminate_for(k, &current, &v, eliminate)?;
    }
    Ok((current, remaining))
}
