//! The algebraic normal form (Zhegalkin polynomial) of Boolean formulas,
//! ported from `sympy_extras.logic.anf`: SymPy's `Xor` constructor
//! ([`xor_evaluated`]), its `And` with relations compared by canonical
//! forms ([`and_evaluated`]), [`is_anf`], [`to_anf`],
//! [`distribute_xor_over_and`], [`anf_coeffs`] (the binary Moebius
//! transform), [`bool_monomial`] and [`anf_form`]. Relations and the
//! other non-connectives are variables of the normal form.

// PORTED FROM SYMPY: sympy/logic/boolalg.py (Xor.__new__, to_anf, is_anf,
// BooleanFunction._to_anf, And.to_anf, Or.to_anf, Not.to_anf, Implies.to_anf,
// Equivalent.to_anf, distribute_xor_over_and, _distribute, ANFform,
// anf_coeffs, bool_monomial, _convert_to_varsANF, And._new_args_filter)
// (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::basic::{Basic, Head};
use crate::core::logic::{and_, false_, ite, not_, true_};
use crate::core::relational::{canonical, negated};
use crate::core::sorting::ordered;
use crate::error::{value_error, PolyResult};

use super::boolalg::is_literal;

fn is_connective(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::And | Head::Or | Head::Not | Head::Xor | Head::Implies | Head::Equivalent | Head::Ite
    )
}

fn is_relational(e: &Basic) -> bool {
    matches!(e.head(), Head::Relational(_))
}

/// `b` added to the arguments, or removed when it is there.
fn toggle(found: &mut Vec<Basic>, b: Basic) {
    if let Some(i) = found.iter().position(|a| *a == b) {
        found.remove(i);
    } else {
        found.push(b);
    }
}

fn remove_first(found: &mut Vec<Basic>, b: &Basic) {
    if let Some(i) = found.iter().position(|a| a == b) {
        found.remove(i);
    }
}

/// The canonical forms of a relation and of its negation.
fn canonical_forms(r: &Basic) -> PolyResult<(Basic, Basic)> {
    Ok((canonical(r)?, canonical(&negated(r))?))
}

// PORTED FROM SYMPY: sympy/logic/boolalg.py:Xor.__new__
/// SymPy's `Xor(*args, remove_true=remove_true)`.
pub fn xor_evaluated(args: &[Basic], remove_true: bool) -> PolyResult<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for a in args {
        let mut b: Basic = a.clone();
        match b.head() {
            Head::BooleanFalse => continue,
            Head::BooleanTrue => b = true_(),
            _ => {}
        }
        if matches!(b.head(), Head::Xor) {
            for c in b.args() {
                toggle(&mut found, c.clone());
            }
        } else {
            toggle(&mut found, b);
        }
    }
    // the relations: (relation, canonical form, canonical form of the negation)
    let mut relations: Vec<(Basic, Basic, Basic)> = Vec::new();
    for r in ordered(found.clone()) {
        if is_relational(&r) {
            let (c, nc) = canonical_forms(&r)?;
            relations.push((r, c, nc));
        }
    }
    let mut odd: bool = false;
    let mut used: Vec<usize> = Vec::new();
    let mut remove: Vec<(Basic, Basic)> = Vec::new();
    for i in 0..relations.len() {
        if used.contains(&i) {
            continue;
        }
        let (r, c, nc) = relations[i].clone();
        for (j, (rj, cj, _)) in relations.iter().enumerate().skip(i + 1) {
            if used.contains(&j) {
                continue;
            }
            if *cj == nc {
                odd = !odd;
            } else if *cj != c {
                continue;
            }
            used.push(i);
            used.push(j);
            remove.push((r.clone(), rj.clone()));
            break;
        }
    }
    if odd {
        toggle(&mut found, true_());
    }
    for (r1, r2) in &remove {
        remove_first(&mut found, r1);
        remove_first(&mut found, r2);
    }
    if found.is_empty() {
        return Ok(false_());
    }
    if found.len() == 1 {
        return Ok(found.remove(0));
    }
    if remove_true && found.contains(&true_()) {
        remove_first(&mut found, &true_());
        return Ok(not_(&xor_evaluated(&found, true)?));
    }
    Ok(Basic::new(Head::Xor, ordered(found)))
}

// PORTED FROM SYMPY: sympy/logic/boolalg.py:And._new_args_filter
/// SymPy's `And(*args)`: the core's `and_` with the relations compared by
/// their canonical forms.
pub fn and_evaluated(args: &[Basic]) -> PolyResult<Basic> {
    let r: Basic = and_(args);
    if !matches!(r.head(), Head::And) || r.args().iter().filter(|a| is_relational(a)).count() < 2 {
        return Ok(r);
    }
    let mut kept: Vec<Basic> = Vec::new();
    let mut canonicals: Vec<Basic> = Vec::new();
    for a in r.args() {
        if is_relational(a) {
            let (c, nc) = canonical_forms(a)?;
            if canonicals.contains(&c) {
                continue;
            }
            if canonicals.contains(&nc) {
                return Ok(false_());
            }
            canonicals.push(c);
        }
        kept.push(a.clone());
    }
    Ok(if kept.len() < r.args().len() { and_(&kept) } else { r })
}

/// Whether `expr` is in algebraic normal form (`is_anf`).
pub fn is_anf(expr: &Basic) -> bool {
    if is_literal(expr) && !matches!(expr.head(), Head::Not) {
        return true;
    }
    match expr.head() {
        Head::And => expr.args().iter().all(|a| a.is_symbol()),
        Head::Xor => {
            for a in expr.args() {
                if matches!(a.head(), Head::And) {
                    if !a.args().iter().all(|s| s.is_symbol()) {
                        return false;
                    }
                } else if is_literal(a) {
                    if matches!(a.head(), Head::Not) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// The arguments of `BooleanFunction._to_anf`.
fn anf_arguments(args: &[Basic], deep: bool) -> PolyResult<Vec<Basic>> {
    let mut found: Vec<Basic> = Vec::new();
    for a in args {
        if deep && is_connective(a) && (!is_literal(a) || matches!(a.head(), Head::Not)) {
            found.push(method_anf(a, deep)?);
        } else {
            found.push(a.clone());
        }
    }
    Ok(found)
}

fn xor_anf(args: &[Basic], deep: bool) -> PolyResult<Basic> {
    xor_evaluated(&anf_arguments(args, deep)?, false)
}

/// The `k`-element subsets in `itertools.combinations` order.
fn combinations(args: &[Basic], k: usize) -> Vec<Vec<Basic>> {
    fn go(args: &[Basic], k: usize, start: usize, current: &mut Vec<Basic>, out: &mut Vec<Vec<Basic>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..args.len() {
            current.push(args[i].clone());
            go(args, k, i + 1, current, out);
            current.pop();
        }
    }
    let mut out: Vec<Vec<Basic>> = Vec::new();
    go(args, k, 0, &mut Vec::new(), &mut out);
    out
}

/// The `to_anf` method of the class of `expr`.
fn method_anf(expr: &Basic, deep: bool) -> PolyResult<Basic> {
    let args: &[Basic] = expr.args();
    match expr.head() {
        Head::And => {
            if !deep {
                return Ok(expr.clone());
            }
            distribute_xor_over_and(&and_evaluated(&anf_arguments(args, deep)?)?)
        }
        Head::Or => {
            let mut parts: Vec<Basic> = Vec::new();
            for k in 1..=args.len() {
                for subset in combinations(args, k) {
                    let s: Basic = and_evaluated(&subset)?;
                    parts.push(if deep { to_anf(&s, deep)? } else { s });
                }
            }
            xor_evaluated(&parts, false)
        }
        Head::Not => xor_anf(&[true_(), args[0].clone()], deep),
        Head::Implies => {
            let (a, b) = (&args[0], &args[1]);
            xor_anf(&[true_(), a.clone(), and_evaluated(&[a.clone(), b.clone()])?], deep)
        }
        Head::Equivalent => {
            let conjunction: Basic = and_evaluated(args)?;
            let mut negated_args: Vec<Basic> = Vec::new();
            for a in args {
                negated_args.push(to_anf(&not_(a), false)?);
            }
            let negations: Basic = distribute_xor_over_and(&and_evaluated(&negated_args)?)?;
            xor_anf(&[conjunction, negations], deep)
        }
        Head::Xor => xor_anf(args, deep),
        Head::Ite => {
            let parts: Vec<Basic> = anf_arguments(args, deep)?;
            Ok(ite(&parts[0], &parts[1], &parts[2]))
        }
        _ => Ok(expr.clone()),
    }
}

/// The algebraic normal form of `expr`; with `deep` false, the arguments
/// of the formula are taken as variables (`to_anf`).
pub fn to_anf(expr: &Basic, deep: bool) -> PolyResult<Basic> {
    if is_anf(expr) || !is_connective(expr) {
        return Ok(expr.clone());
    }
    method_anf(expr, deep)
}

/// The conjunctions distributed over the exclusive disjunctions
/// (`distribute_xor_over_and`).
pub fn distribute_xor_over_and(expr: &Basic) -> PolyResult<Basic> {
    match expr.head() {
        Head::And => {
            for arg in expr.args() {
                if matches!(arg.head(), Head::Xor) {
                    let rest_args: Vec<Basic> = expr.args().iter().filter(|a| *a != arg).cloned().collect();
                    let rest: Basic = and_evaluated(&rest_args)?;
                    let mut parts: Vec<Basic> = Vec::new();
                    for c in arg.args() {
                        parts.push(distribute_xor_over_and(&and_evaluated(&[c.clone(), rest.clone()])?)?);
                    }
                    return xor_evaluated(&parts, false);
                }
            }
            Ok(expr.clone())
        }
        Head::Xor => {
            let mut parts: Vec<Basic> = Vec::new();
            for a in expr.args() {
                parts.push(distribute_xor_over_and(a)?);
            }
            xor_evaluated(&parts, false)
        }
        _ => Ok(expr.clone()),
    }
}

/// The coefficients (0 or 1) of the Zhegalkin polynomial with the truth
/// table `truthvalues` (`anf_coeffs`).
pub fn anf_coeffs(truthvalues: &[u8]) -> PolyResult<Vec<u8>> {
    let len: usize = truthvalues.len();
    let n: u32 = usize::BITS - len.leading_zeros() - u32::from(len > 0);
    let n: u32 = if len == 0 { 0 } else { n };
    if len != 1usize << n {
        return Err(value_error(format!(
            "The number of truth values must be a power of two, got {}",
            len
        )));
    }
    let mut coeffs: Vec<Vec<u8>> = truthvalues.iter().map(|v| vec![*v]).collect();
    for i in 0..n {
        let mut tmp: Vec<Vec<u8>> = Vec::new();
        for j in 0..(1usize << (n - i - 1)) {
            let mut row: Vec<u8> = coeffs[2 * j].clone();
            row.extend(coeffs[2 * j].iter().zip(&coeffs[2 * j + 1]).map(|(p, q)| p ^ q));
            tmp.push(row);
        }
        coeffs = tmp;
    }
    Ok(coeffs.remove(0))
}

/// The conjunction of the variables whose digit is 1 (`true` for none).
fn convert_to_vars_anf(term: &[u8], variables: &[Basic]) -> PolyResult<Basic> {
    let chosen: Vec<Basic> = term
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == 1)
        .map(|(n, _)| variables[n].clone())
        .collect();
    if chosen.is_empty() {
        return Ok(true_());
    }
    and_evaluated(&chosen)
}

/// The `k`-th Zhegalkin monomial (`bool_monomial` of an integer).
pub fn bool_monomial(k: i64, variables: &[Basic]) -> PolyResult<Basic> {
    let n: usize = variables.len();
    if k < 0 || (n < 63 && k >= 1i64 << n) {
        return Err(value_error(format!("Integer {} out of range for {} variables", k, n)));
    }
    let term: Vec<u8> = (0..n).map(|i| ((k >> (n - 1 - i)) & 1) as u8).collect();
    convert_to_vars_anf(&term, variables)
}

/// The Zhegalkin monomial of the digits `term` (`bool_monomial` of a list).
pub fn bool_monomial_digits(term: &[u8], variables: &[Basic]) -> PolyResult<Basic> {
    convert_to_vars_anf(term, variables)
}

/// The algebraic normal form of the function of `variables` with the truth
/// table `truthvalues` (`ANFform`).
pub fn anf_form(variables: &[Basic], truthvalues: &[u8]) -> PolyResult<Basic> {
    let n: usize = variables.len();
    if truthvalues.len() != 1usize << n {
        return Err(value_error(format!(
            "The number of truth values must be equal to 2^{}, got {}",
            n,
            truthvalues.len()
        )));
    }
    let coeffs: Vec<u8> = anf_coeffs(truthvalues)?;
    let mut parts: Vec<Basic> = Vec::new();
    for (i, c) in coeffs.iter().enumerate() {
        if *c == 1 {
            let term: Vec<u8> = (0..n).map(|b| ((i >> (n - 1 - b)) & 1) as u8).collect();
            parts.push(convert_to_vars_anf(&term, variables)?);
        }
    }
    xor_evaluated(&parts, false)
}
