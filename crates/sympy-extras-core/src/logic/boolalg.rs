//! Minimal sums of products and products of sums of the Booleans of the
//! core, and the normal forms, ported from `sympy_extras.logic.boolalg`:
//! [`simplify_logic`], [`sop_form`], [`pos_form`] (Quine–McCluskey with
//! SymPy's cover by row and column dominance), [`is_literal`],
//! [`is_cnf`], [`is_dnf`], [`is_nnf`], [`to_nnf`],
//! [`eliminate_implications`], [`distribute_and_over_or`],
//! [`distribute_or_over_and`], [`to_cnf`] and [`to_dnf`].
//!
//! The relations of a formula are replaced by dummies (a relation and its
//! negation by one dummy and its negation) in SymPy's `ordered` order, so
//! that the result does not depend on hashes; SymPy's `deep=True` (the
//! relations simplified) is not ported.

// PORTED FROM SYMPY: sympy/logic/boolalg.py (is_literal, _is_form, is_cnf, is_dnf,
// _find_predicates, _check_pair, _convert_to_varsSOP, _convert_to_varsPOS,
// _simplified_pairs, _rem_redundancy, SOPform, POSform, _get_truthtable,
// simplify_logic, to_nnf, is_nnf, eliminate_implications, _distribute,
// distribute_and_over_or, distribute_or_over_and, to_cnf, to_dnf) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::{HashMap, HashSet};

use crate::core::basic::{Basic, Head};
use crate::core::logic::{and_, false_, not_, or_, true_};
use crate::error::{PolyError, PolyResult};

/// A term of the truth table: 0, 1, or 3 for a variable which does not
/// matter.
pub type Term = Vec<u8>;

/// SymPy's limit on the number of variables without `force`.
const MAX_VARIABLES: usize = 8;

/// The normal form asked of [`simplify_logic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Cnf,
    Dnf,
}

fn value_error(message: String) -> PolyError {
    crate::error::value_error(message)
}

/// Whether `e` is a `BooleanFunction` (a connective).
fn is_connective(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::And | Head::Or | Head::Not | Head::Xor | Head::Implies | Head::Equivalent | Head::Ite
    )
}

/// Whether `e` is an `And` (`conjunction`) or an `Or`.
fn is_lattice(e: &Basic, conjunction: bool) -> bool {
    if conjunction {
        matches!(e.head(), Head::And)
    } else {
        matches!(e.head(), Head::Or)
    }
}

fn is_atom_value(e: &Basic) -> bool {
    matches!(e.head(), Head::BooleanTrue | Head::BooleanFalse)
}

/// Whether `expr` is a literal: an atom (anything which is not a
/// connective, with atomic arguments) or the negation of one.
pub fn is_literal(expr: &Basic) -> bool {
    if matches!(expr.head(), Head::Not) {
        return is_literal(&expr.args()[0]);
    }
    if expr.args().is_empty() {
        return true;
    }
    !is_connective(expr) && expr.args().iter().all(|a| a.args().is_empty())
}

/// SymPy's `_is_form`: `expr` is a literal, an `inner` of literals, or an
/// `outer` of those.
fn is_form(expr: &Basic, outer_and: bool) -> bool {
    let args: Vec<Basic> = if is_lattice(expr, outer_and) {
        expr.args().to_vec()
    } else {
        vec![expr.clone()]
    };
    for arg in &args {
        if is_lattice(arg, !outer_and) {
            if !arg.args().iter().all(is_literal) {
                return false;
            }
        } else if !is_literal(arg) {
            return false;
        }
    }
    true
}

/// Whether `expr` is in conjunctive normal form.
pub fn is_cnf(expr: &Basic) -> bool {
    is_form(expr, true)
}

/// Whether `expr` is in disjunctive normal form.
pub fn is_dnf(expr: &Basic) -> bool {
    is_form(expr, false)
}

/// The atoms of the formula (everything which is not a connective),
/// without duplicates, in the order first met.
fn find_predicates(expr: &Basic) -> Vec<Basic> {
    fn walk(e: &Basic, found: &mut Vec<Basic>, seen: &mut HashSet<Basic>) {
        if is_connective(e) {
            for a in e.args() {
                walk(a, found, seen);
            }
        } else if seen.insert(e.clone()) {
            found.push(e.clone());
        }
    }
    let mut found: Vec<Basic> = Vec::new();
    walk(expr, &mut found, &mut HashSet::new());
    found
}

/// The index of the only bit where the terms differ, `None` if they differ
/// in no bit or in several.
fn check_pair(minterm1: &Term, minterm2: &Term) -> Option<usize> {
    let mut index: Option<usize> = None;
    for (x, i) in minterm1.iter().enumerate() {
        if *i != minterm2[x] {
            if index.is_none() {
                index = Some(x);
            } else {
                return None;
            }
        }
    }
    index
}

fn convert_to_vars_sop(minterm: &Term, variables: &[Basic]) -> Basic {
    let parts: Vec<Basic> = minterm
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 3)
        .map(|(n, v)| {
            if *v == 1 {
                variables[n].clone()
            } else {
                not_(&variables[n])
            }
        })
        .collect();
    and_(&parts)
}

fn convert_to_vars_pos(maxterm: &Term, variables: &[Basic]) -> Basic {
    let parts: Vec<Basic> = maxterm
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 3)
        .map(|(n, v)| {
            if *v == 0 {
                variables[n].clone()
            } else {
                not_(&variables[n])
            }
        })
        .collect();
    or_(&parts)
}

/// The prime implicants of the terms (Quine–McCluskey): pairs of terms
/// which differ in one bit merged, until no pair merges.
fn simplified_pairs(terms: &[Term]) -> Vec<Term> {
    if terms.is_empty() {
        return Vec::new();
    }
    let mut simplified_terms: Vec<Term> = Vec::new();
    let mut todo: Vec<Option<usize>> = (0..terms.len()).map(Some).collect();
    // the number of ones: two terms only merge if they differ by one
    let mut termdict: HashMap<usize, Vec<usize>> = HashMap::new();
    for (n, term) in terms.iter().enumerate() {
        termdict
            .entry(term.iter().filter(|t| **t == 1).count())
            .or_default()
            .push(n);
    }
    let variables: usize = terms[0].len();
    let empty: Vec<usize> = Vec::new();
    for k in 0..variables {
        let lower: &Vec<usize> = termdict.get(&k).unwrap_or(&empty);
        let upper: &Vec<usize> = termdict.get(&(k + 1)).unwrap_or(&empty);
        for &i in lower {
            for &j in upper {
                if let Some(index) = check_pair(&terms[i], &terms[j]) {
                    todo[i] = None;
                    todo[j] = None;
                    let mut newterm: Term = terms[i].clone();
                    newterm[index] = 3;
                    if !simplified_terms.contains(&newterm) {
                        simplified_terms.push(newterm);
                    }
                }
            }
        }
    }
    if !simplified_terms.is_empty() {
        simplified_terms = simplified_pairs(&simplified_terms);
    }
    simplified_terms.extend(todo.iter().flatten().map(|i| terms[*i].clone()));
    simplified_terms
}

/// A cover of the terms by the prime implicants `l1`: dominated rows and
/// dominated columns of the prime implicant table removed in turn, then
/// the implicant covering most terms chosen.
fn rem_redundancy(l1: &[Term], terms: &[Term]) -> Vec<Term> {
    if terms.is_empty() {
        return Vec::new();
    }
    let (nterms, nl1) = (terms.len(), l1.len());
    let mut dommatrix: Vec<Vec<u8>> = vec![vec![0; nl1]; nterms];
    let mut colcount: Vec<i64> = vec![0; nl1];
    let mut rowcount: Vec<i64> = vec![0; nterms];
    for (primei, prime) in l1.iter().enumerate() {
        for (termi, term) in terms.iter().enumerate() {
            if prime.iter().zip(term).all(|(t, mt)| *t == 3 || t == mt) {
                dommatrix[termi][primei] = 1;
                colcount[primei] += 1;
                rowcount[termi] += 1;
            }
        }
    }
    let mut anythingchanged: bool = true;
    while anythingchanged {
        anythingchanged = false;
        for rowi in 0..nterms {
            if rowcount[rowi] != 0 {
                let row: Vec<u8> = dommatrix[rowi].clone();
                for row2i in 0..nterms {
                    if rowi != row2i && rowcount[rowi] != 0 && rowcount[rowi] <= rowcount[row2i] {
                        let row2: Vec<u8> = dommatrix[row2i].clone();
                        if (0..nl1).all(|n| row2[n] >= row[n]) {
                            // row2 dominates row: row2 removed
                            rowcount[row2i] = 0;
                            anythingchanged = true;
                            for (primei, prime_entry) in row2.iter().enumerate() {
                                if *prime_entry != 0 {
                                    dommatrix[row2i][primei] = 0;
                                    colcount[primei] -= 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // SymPy's cache: a column is read once per pass, before the
        // removals of that pass
        let mut colcache: HashMap<usize, Vec<u8>> = HashMap::new();
        let column = |dommatrix: &Vec<Vec<u8>>, c: usize| -> Vec<u8> { (0..nterms).map(|i| dommatrix[i][c]).collect() };
        for coli in 0..nl1 {
            if colcount[coli] != 0 {
                let col: Vec<u8> = colcache.entry(coli).or_insert_with(|| column(&dommatrix, coli)).clone();
                for col2i in 0..nl1 {
                    if coli != col2i && colcount[col2i] != 0 && colcount[coli] >= colcount[col2i] {
                        let col2: Vec<u8> = colcache
                            .entry(col2i)
                            .or_insert_with(|| column(&dommatrix, col2i))
                            .clone();
                        if (0..nterms).all(|n| col[n] >= col2[n]) {
                            // col dominates col2: col2 removed
                            colcount[col2i] = 0;
                            anythingchanged = true;
                            for (termi, term_entry) in col2.iter().enumerate() {
                                if *term_entry != 0 && dommatrix[termi][col2i] != 0 {
                                    dommatrix[termi][col2i] = 0;
                                    rowcount[termi] -= 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        if !anythingchanged {
            // the prime implicant covering most terms, if it covers two
            let mut maxterms: i64 = 0;
            let mut bestcolidx: Option<usize> = None;
            for (coli, s) in colcount.iter().enumerate() {
                if *s > maxterms {
                    bestcolidx = Some(coli);
                    maxterms = *s;
                }
            }
            if let Some(best) = bestcolidx.filter(|_| maxterms > 1) {
                let bestcol: Vec<u8> = colcache.get(&best).cloned().expect("a cached column");
                for primei in 0..nl1 {
                    if primei != best {
                        for (termi, term_entry) in bestcol.iter().enumerate() {
                            if *term_entry != 0 && dommatrix[termi][primei] != 0 {
                                dommatrix[termi][primei] = 0;
                                anythingchanged = true;
                                rowcount[termi] -= 1;
                                colcount[primei] -= 1;
                            }
                        }
                    }
                }
            }
        }
    }
    (0..nl1).filter(|i| colcount[*i] != 0).map(|i| l1[i].clone()).collect()
}

fn checked_terms(terms: &[Term], bits: usize) -> PolyResult<Vec<Term>> {
    for t in terms {
        if t.len() != bits {
            return Err(value_error(format!(
                "Each term must contain {bits} bits as there are\n{bits} variables (or be an integer).",
                bits = bits
            )));
        }
    }
    Ok(terms.to_vec())
}

fn sop(variables: &[Basic], minterms: &[Term], dontcares: &[Term]) -> Basic {
    let all: Vec<Term> = minterms.iter().chain(dontcares).cloned().collect();
    let new: Vec<Term> = simplified_pairs(&all);
    let essential: Vec<Term> = rem_redundancy(&new, minterms);
    or_(&essential
        .iter()
        .map(|x| convert_to_vars_sop(x, variables))
        .collect::<Vec<Basic>>())
}

/// All the terms of `n` bits in lexicographic order (`product((0, 1),
/// repeat=n)`).
fn all_terms(n: usize) -> Vec<Term> {
    let mut result: Vec<Term> = vec![Vec::new()];
    for _ in 0..n {
        let mut next: Vec<Term> = Vec::new();
        for t in &result {
            for bit in [0u8, 1] {
                let mut u: Term = t.clone();
                u.push(bit);
                next.push(u);
            }
        }
        result = next;
    }
    result
}

fn pos(variables: &[Basic], minterms: &[Term], dontcares: &[Term]) -> Basic {
    let maxterms: Vec<Term> = all_terms(variables.len())
        .into_iter()
        .filter(|t| !minterms.contains(t) && !dontcares.contains(t))
        .collect();
    let all: Vec<Term> = maxterms.iter().chain(dontcares).cloned().collect();
    let new: Vec<Term> = simplified_pairs(&all);
    let essential: Vec<Term> = rem_redundancy(&new, &maxterms);
    and_(
        &essential
            .iter()
            .map(|x| convert_to_vars_pos(x, variables))
            .collect::<Vec<Basic>>(),
    )
}

fn checked(variables: &[Basic], minterms: &[Term], dontcares: &[Term]) -> PolyResult<(Vec<Term>, Vec<Term>)> {
    let m: Vec<Term> = checked_terms(minterms, variables.len())?;
    let d: Vec<Term> = checked_terms(dontcares, variables.len())?;
    for t in &d {
        if m.contains(t) {
            return Err(value_error(format!("{:?} in minterms is also in dontcares", t)));
        }
    }
    Ok((m, d))
}

/// The smallest sum of products found by the Quine–McCluskey method which
/// is true on the `minterms` and false elsewhere, the `dontcares` free
/// (`SOPform`).
pub fn sop_form(variables: &[Basic], minterms: &[Term], dontcares: &[Term]) -> PolyResult<Basic> {
    if minterms.is_empty() {
        return Ok(false_());
    }
    let (m, d) = checked(variables, minterms, dontcares)?;
    Ok(sop(variables, &m, &d))
}

/// The smallest product of sums (`POSform`).
pub fn pos_form(variables: &[Basic], minterms: &[Term], dontcares: &[Term]) -> PolyResult<Basic> {
    if minterms.is_empty() {
        return Ok(false_());
    }
    let (m, d) = checked(variables, minterms, dontcares)?;
    Ok(pos(variables, &m, &d))
}

fn replaced(expr: &Basic, rule: &HashMap<Basic, Basic>) -> PolyResult<Basic> {
    let e: Basic = crate::core::function::xreplace_evaluated(expr, rule)?;
    if !crate::core::logic::is_boolean(&e) && !e.is_symbol() && !matches!(e.head(), Head::Dummy(..)) {
        return Err(PolyError::Type(format!("{} is not a Boolean", e)));
    }
    Ok(e)
}

/// The rows of the variables (with the constants first) where the formula
/// is true.
fn get_truthtable(variables: &[Basic], expr: &Basic, constants: &[u8]) -> PolyResult<Vec<Term>> {
    let mut rows: Vec<(Basic, Term)> = vec![(expr.clone(), Vec::new())];
    for v in variables.iter().rev() {
        let mut table: Vec<(Basic, Term)> = Vec::new();
        for value in [false_(), true_()] {
            let bit: u8 = u8::from(value == true_());
            let rule: HashMap<Basic, Basic> = HashMap::from([(v.clone(), value)]);
            for (e, bits) in &rows {
                if matches!(e.head(), Head::BooleanFalse) {
                    continue;
                }
                let mut b: Term = vec![bit];
                b.extend(bits.iter().copied());
                table.push((replaced(e, &rule)?, b));
            }
        }
        rows = table;
    }
    let mut res: Vec<Term> = Vec::new();
    for (e, bits) in rows {
        if !is_atom_value(&e) {
            return Err(PolyError::Type(format!(
                "the formula {} is not decided by its atoms",
                e
            )));
        }
        if matches!(e.head(), Head::BooleanTrue) {
            let mut t: Term = constants.to_vec();
            t.extend(bits);
            res.push(t);
        }
    }
    if res.len() == 1 && res[0].is_empty() {
        return Ok(Vec::new());
    }
    Ok(res)
}

/// The atoms of `e` which are relations, without duplicates.
fn relations_of(e: &Basic, found: &mut Vec<Basic>) {
    for n in e.preorder() {
        if matches!(n.head(), Head::Relational(_)) && !found.contains(&n) {
            found.push(n);
        }
    }
}

/// The formula as a minimal sum of products (`Form::Dnf`) or product of
/// sums (`Form::Cnf`), by default the one SymPy's `simplify_logic(expr,
/// deep=False)` chooses; a formula with more than eight variables is
/// returned as it is unless `force`. The rows where `dontcare` is true are
/// free.
pub fn simplify_logic(expr: &Basic, form: Option<Form>, force: bool, dontcare: Option<&Basic>) -> PolyResult<Basic> {
    use crate::core::sorting::ordered;
    if let Some(f) = form {
        let form_ok: bool = if f == Form::Cnf { is_cnf(expr) } else { is_dnf(expr) };
        if form_ok && (!is_connective(expr) || expr.args().iter().all(is_literal)) {
            return Ok(expr.clone());
        }
    }
    if !is_connective(expr) {
        return Ok(expr.clone());
    }
    // the relations replaced by variables, a relation and its negation by
    // one variable and its negation
    let mut repl: HashMap<Basic, Basic> = HashMap::new();
    let mut undo: HashMap<Basic, Basic> = HashMap::new();
    let mut relations: Vec<Basic> = Vec::new();
    relations_of(expr, &mut relations);
    if let Some(dc) = dontcare {
        relations_of(dc, &mut relations);
    }
    for var in ordered(relations.clone()) {
        if repl.contains_key(&var) {
            continue;
        }
        let d: Basic = crate::core::Dummy::new("Dummy", &[]).0;
        undo.insert(d.clone(), var.clone());
        repl.insert(var.clone(), d.clone());
        let nvar: Basic = crate::core::relational::negated(&var);
        if relations.contains(&nvar) {
            repl.insert(nvar, not_(&d));
        }
    }
    let replaced_expr: Basic = replaced(expr, &repl)?;
    let mut dc: Option<Basic> = match dontcare {
        Some(d) => Some(replaced(d, &repl)?),
        None => None,
    };
    let mut variables: Vec<Basic> = find_predicates(&replaced_expr);
    if !force && variables.len() > MAX_VARIABLES {
        return replaced(&replaced_expr, &undo);
    }
    if let Some(d) = &dc {
        for v in find_predicates(d) {
            if !variables.contains(&v) {
                variables.push(v);
            }
        }
        if !force && variables.len() > MAX_VARIABLES {
            variables = find_predicates(&replaced_expr);
            dc = None;
        }
    }
    let ordered_variables: Vec<Basic> = ordered(variables);
    let constants: Vec<Basic> = ordered_variables.iter().filter(|v| is_atom_value(v)).cloned().collect();
    let free: Vec<Basic> = ordered_variables
        .iter()
        .filter(|v| !is_atom_value(v))
        .cloned()
        .collect();
    let mut all_variables: Vec<Basic> = constants.clone();
    all_variables.extend(free.iter().cloned());
    let constant_bits: Vec<u8> = constants
        .iter()
        .map(|c| u8::from(matches!(c.head(), Head::BooleanTrue)))
        .collect();
    let mut truthtable: Vec<Term> = get_truthtable(&free, &replaced_expr, &constant_bits)?;
    let dctruthtable: Vec<Term> = match &dc {
        Some(d) => {
            let table: Vec<Term> = get_truthtable(&free, d, &constant_bits)?;
            truthtable.retain(|t| !table.contains(t));
            table
        }
        None => Vec::new(),
    };
    // len(truthtable) >= 2**(len(all_variables) - 1)
    let big: bool = 2 * (truthtable.len() as u128) >= 1u128 << all_variables.len();
    if form == Some(Form::Dnf) || (form.is_none() && big) {
        return replaced(&sop(&all_variables, &truthtable, &dctruthtable), &undo);
    }
    if truthtable.is_empty() {
        return Ok(false_());
    }
    replaced(&pos(&all_variables, &truthtable, &dctruthtable), &undo)
}

// ---------------------------------------------------------------------
// the normal forms
// ---------------------------------------------------------------------

/// SymPy's `And._to_nnf` and `Or._to_nnf`.
fn lattice_nnf(conjunction: bool, args: &[Basic], simplify: bool) -> PolyResult<Basic> {
    let mut kept: Vec<Basic> = Vec::new();
    let mut seen: HashSet<Basic> = HashSet::new();
    for arg in args {
        let arg: Basic = if is_literal(arg) {
            arg.clone()
        } else {
            nnf(arg, simplify)?
        };
        if simplify {
            let parts: Vec<Basic> = if is_lattice(&arg, conjunction) {
                arg.args().to_vec()
            } else {
                vec![arg.clone()]
            };
            for a in parts {
                if seen.contains(&not_(&a)) {
                    return Ok(if conjunction { false_() } else { true_() });
                }
                if seen.insert(a.clone()) {
                    kept.push(a);
                }
            }
        } else if seen.insert(arg.clone()) {
            kept.push(arg);
        }
    }
    Ok(if conjunction { and_(&kept) } else { or_(&kept) })
}

/// The `k`-element subsets of `args`, in `itertools.combinations` order.
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

/// The clauses of `Xor` (`first = 0`) or of its negation (`first = 1`).
fn xor_clauses(args: &[Basic], first: usize) -> Vec<Basic> {
    let mut clauses: Vec<Basic> = Vec::new();
    let mut i: usize = first;
    while i <= args.len() {
        for negated in combinations(args, i) {
            let parts: Vec<Basic> = args
                .iter()
                .map(|a| if negated.contains(a) { not_(a) } else { a.clone() })
                .collect();
            clauses.push(or_(&parts));
        }
        i += 2;
    }
    clauses
}

/// The `to_nnf` method of the class of `expr`.
fn nnf(expr: &Basic, simplify: bool) -> PolyResult<Basic> {
    let args: &[Basic] = expr.args();
    match expr.head() {
        Head::And => lattice_nnf(true, args, simplify),
        Head::Or => lattice_nnf(false, args, simplify),
        Head::Not => {
            if is_literal(expr) {
                return Ok(expr.clone());
            }
            let inner: &Basic = &args[0];
            let a: &[Basic] = inner.args();
            let negated = |v: &[Basic]| -> Vec<Basic> { v.iter().map(not_).collect() };
            match inner.head() {
                Head::And => lattice_nnf(false, &negated(a), simplify),
                Head::Or => lattice_nnf(true, &negated(a), simplify),
                Head::Implies => lattice_nnf(true, &[a[0].clone(), not_(&a[1])], simplify),
                Head::Equivalent => lattice_nnf(true, &[or_(a), or_(&negated(a))], simplify),
                Head::Xor => lattice_nnf(true, &xor_clauses(a, 1), simplify),
                Head::Ite => {
                    let (p, q, r) = (&a[0], &a[1], &a[2]);
                    lattice_nnf(true, &[or_(&[p.clone(), not_(r)]), or_(&[not_(p), not_(q)])], simplify)
                }
                _ => Err(value_error(format!(
                    "Illegal operator {} in expression",
                    inner.class_name()
                ))),
            }
        }
        Head::Xor => lattice_nnf(true, &xor_clauses(args, 0), simplify),
        Head::Implies => lattice_nnf(false, &[not_(&args[0]), args[1].clone()], simplify),
        Head::Equivalent => {
            let mut clauses: Vec<Basic> = args.windows(2).map(|w| or_(&[not_(&w[0]), w[1].clone()])).collect();
            clauses.push(or_(&[not_(&args[args.len() - 1]), args[0].clone()]));
            lattice_nnf(true, &clauses, simplify)
        }
        Head::Ite => {
            let (p, q, r) = (&args[0], &args[1], &args[2]);
            lattice_nnf(
                true,
                &[or_(&[not_(p), q.clone()]), or_(&[p.clone(), r.clone()])],
                simplify,
            )
        }
        _ => Ok(expr.clone()),
    }
}

/// Whether `expr` is in negation normal form; with `simplified`, no `And`
/// or `Or` with an argument and its negation.
pub fn is_nnf(expr: &Basic, simplified: bool) -> bool {
    if is_literal(expr) {
        return true;
    }
    let mut stack: Vec<Basic> = vec![expr.clone()];
    while let Some(e) = stack.pop() {
        if matches!(e.head(), Head::And | Head::Or) {
            let args: &[Basic] = e.args();
            if simplified && args.iter().any(|a| args.contains(&not_(a))) {
                return false;
            }
            stack.extend(args.iter().cloned());
        } else if !is_literal(&e) {
            return false;
        }
    }
    true
}

/// The negation normal form; with `simplify`, no redundant clauses.
pub fn to_nnf(expr: &Basic, simplify: bool) -> PolyResult<Basic> {
    if is_nnf(expr, simplify) {
        return Ok(expr.clone());
    }
    nnf(expr, simplify)
}

/// `Implies`, `Equivalent`, `Xor` and `ITE` written with `And`, `Or` and
/// `Not`.
pub fn eliminate_implications(expr: &Basic) -> PolyResult<Basic> {
    to_nnf(expr, false)
}

fn build(conjunction: bool, args: &[Basic]) -> Basic {
    if conjunction {
        and_(args)
    } else {
        or_(args)
    }
}

/// `distribute_and_over_or` (`conjunction`) and `distribute_or_over_and`.
fn distribute(e: &Basic, conjunction: bool) -> Basic {
    if is_lattice(e, !conjunction) {
        for arg in e.args() {
            if is_lattice(arg, conjunction) {
                let rest_args: Vec<Basic> = e.args().iter().filter(|a| *a != arg).cloned().collect();
                let rest: Basic = build(!conjunction, &rest_args);
                let parts: Vec<Basic> = arg
                    .args()
                    .iter()
                    .map(|c| distribute(&build(!conjunction, &[c.clone(), rest.clone()]), conjunction))
                    .collect();
                return build(conjunction, &parts);
            }
        }
        return e.clone();
    }
    if is_lattice(e, conjunction) {
        let parts: Vec<Basic> = e.args().iter().map(|a| distribute(a, conjunction)).collect();
        return build(conjunction, &parts);
    }
    e.clone()
}

/// An `Or` of `And` distributed into an `And` of `Or`.
pub fn distribute_and_over_or(expr: &Basic) -> Basic {
    distribute(expr, true)
}

/// An `And` of `Or` distributed into an `Or` of `And`.
pub fn distribute_or_over_and(expr: &Basic) -> Basic {
    distribute(expr, false)
}

fn check_size(expr: &Basic, force: bool) -> PolyResult<()> {
    if !force && find_predicates(expr).len() > MAX_VARIABLES {
        return Err(value_error(
            "To simplify a logical expression with more than 8 variables may take a long time and requires the use of `force=True`."
                .to_string(),
        ));
    }
    Ok(())
}

/// The conjunctive normal form; with `simplify`, the minimal one.
pub fn to_cnf(expr: &Basic, simplify: bool, force: bool) -> PolyResult<Basic> {
    if !is_connective(expr) {
        return Ok(expr.clone());
    }
    if simplify {
        check_size(expr, force)?;
        return simplify_logic(expr, Some(Form::Cnf), force, None);
    }
    if is_cnf(expr) {
        return Ok(expr.clone());
    }
    Ok(distribute_and_over_or(&eliminate_implications(expr)?))
}

/// The disjunctive normal form; with `simplify`, the minimal one.
pub fn to_dnf(expr: &Basic, simplify: bool, force: bool) -> PolyResult<Basic> {
    if !is_connective(expr) {
        return Ok(expr.clone());
    }
    if simplify {
        check_size(expr, force)?;
        return simplify_logic(expr, Some(Form::Dnf), force, None);
    }
    if is_dnf(expr) {
        return Ok(expr.clone());
    }
    Ok(distribute_or_over_and(&eliminate_implications(expr)?))
}
