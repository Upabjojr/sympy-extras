//! Reduction over the complex numbers, ported from
//! `sympy_extras.polys.comprehensive`: comprehensive Gröbner systems
//! (Kapur, Sun and Wang), the existential closure of a system of equations
//! and inequations (Rabinowitsch's trick for the inequations), quantifier
//! elimination for Boolean combinations of polynomial equations and
//! inequations, and their simplification by Gröbner bases.

use crate::core::basic::{Basic, Head};
use crate::core::logic::{and_, boolean, not_, or_};
use crate::core::relational::{eq, ne};
use crate::core::symbol::Dummy;
use crate::error::{value_error, PolyResult};

use super::domains::{Domain, RationalField, QQ};
use super::ideals::{Ideal, PolyLike};
use super::rings::{monomial_divides, Monomial, MonomialOrder, PolyElement};

/// A branch of a comprehensive Gröbner system (`Branch`): on the
/// parameters where the `equations` vanish and the `nonzero` polynomials do
/// not, the `basis` specialises to a Gröbner basis of the system.
#[derive(Debug, Clone, PartialEq)]
pub struct Branch {
    pub equations: Vec<Basic>,
    pub nonzero: Vec<Basic>,
    pub basis: Vec<Basic>,
}

impl Branch {
    /// The constructible set of the branch, as a formula.
    pub fn condition(&self) -> PolyResult<Basic> {
        let zero: Basic = Basic::zero();
        let mut parts: Vec<Basic> = Vec::new();
        for e in &self.equations {
            parts.push(eq(e, &zero)?);
        }
        for n in &self.nonzero {
            parts.push(ne(n, &zero)?);
        }
        Ok(and_(&parts))
    }

    /// Whether the system has solutions on the branch.
    pub fn solvable(&self) -> bool {
        !self
            .basis
            .iter()
            .any(|g| g.free_symbols().is_empty() && *g != Basic::zero())
    }
}

/// A quantifier of a prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantifier {
    Exists,
    Forall,
}

fn ideal(polynomials: &[Basic], symbols: &[Basic]) -> PolyResult<Ideal<RationalField>> {
    Ideal::new(polynomials, symbols, QQ, MonomialOrder::Grevlex)
}

/// Whether some parameter value satisfies the conditions (`_consistent`).
fn consistent(equations: &[Basic], nonzero: &[Basic], parameters: &[Basic]) -> PolyResult<bool> {
    if equations.is_empty() {
        return Ok(true);
    }
    let product: Basic = if nonzero.is_empty() {
        Basic::one()
    } else {
        crate::core::mul::mul(nonzero)?
    };
    Ok(!ideal(equations, parameters)?.radical_contains(&PolyLike::Expr(&product))?)
}

/// The nonconstant irreducible factors, without repetition.
fn irreducible_factors(e: &Basic, symbols: &[Basic]) -> PolyResult<Vec<Basic>> {
    let (_, factors) = super::polytools::factor_list(e, symbols)?;
    let mut result: Vec<Basic> = Vec::new();
    for (f, _) in factors {
        if !f.free_symbols().is_empty() && !result.contains(&f) {
            result.push(f);
        }
    }
    Ok(result)
}

fn is_parametric(g: &PolyElement<RationalField>, k: usize) -> bool {
    g.terms.keys().all(|m| m[..k].iter().all(|e| *e == 0))
}

/// The coefficient, a polynomial in the parameters, of the leading monomial
/// of `g` in the first `k` variables (`_leading_coefficient`).
fn leading_coefficient(g: &PolyElement<RationalField>, k: usize, parameters: &[Basic]) -> PolyResult<Basic> {
    let lm: Monomial = g.LM();
    let head: &[u32] = &lm[..k];
    let mut terms: Vec<Basic> = Vec::new();
    for (m, c) in &g.terms {
        if &m[..k] == head {
            let mut factors: Vec<Basic> = vec![QQ.to_expr(c)?];
            for (p, e) in parameters.iter().zip(&m[k..]) {
                if *e > 0 {
                    factors.push(crate::core::power::pow(p, &Basic::int(*e as i64))?);
                }
            }
            terms.push(crate::core::mul::mul(&factors)?);
        }
    }
    crate::core::add::add(&terms)
}

fn cgs(
    equations: Vec<Basic>,
    nonzero: Vec<Basic>,
    polynomials: &[Basic],
    variables: &[Basic],
    parameters: &[Basic],
    branches: &mut Vec<Branch>,
) -> PolyResult<()> {
    if !consistent(&equations, &nonzero, parameters)? {
        return Ok(());
    }
    let k: usize = variables.len();
    let ordered: Vec<Basic> = variables.iter().chain(parameters).cloned().collect();
    let gens: Vec<Basic> = polynomials
        .iter()
        .chain(&equations)
        .filter(|f| **f != Basic::zero())
        .cloned()
        .collect();
    let order: MonomialOrder = MonomialOrder::elimination(k, ordered.len(), MonomialOrder::Grevlex);
    let id: Ideal<RationalField> = Ideal::new(&gens, &ordered, QQ, order)?;
    let g_basis: Vec<PolyElement<RationalField>> = if gens.is_empty() {
        Vec::new()
    } else {
        id.groebner_basis(None)?
    };
    if g_basis.iter().any(|g| g.is_ground()) {
        branches.push(Branch {
            equations,
            nonzero,
            basis: vec![Basic::one()],
        });
        return Ok(());
    }
    let parametric: Vec<&PolyElement<RationalField>> = g_basis.iter().filter(|g| is_parametric(g, k)).collect();
    let mixed: Vec<&PolyElement<RationalField>> = g_basis.iter().filter(|g| !is_parametric(g, k)).collect();
    for g in &parametric {
        let h: Basic = id.to_expr(g)?;
        let mut more: Vec<Basic> = nonzero.clone();
        more.push(h);
        if consistent(&equations, &more, parameters)? {
            branches.push(Branch {
                equations: equations.clone(),
                nonzero: more,
                basis: vec![Basic::one()],
            });
        }
    }
    let equations: Vec<Basic> = parametric.iter().map(|g| id.to_expr(g)).collect::<PolyResult<_>>()?;
    if !consistent(&equations, &nonzero, parameters)? {
        return Ok(());
    }
    let basis: Vec<Basic> = mixed.iter().map(|g| id.to_expr(g)).collect::<PolyResult<_>>()?;
    if mixed.is_empty() {
        branches.push(Branch {
            equations,
            nonzero,
            basis: Vec::new(),
        });
        return Ok(());
    }
    let leading: Vec<(Monomial, &PolyElement<RationalField>)> =
        mixed.iter().map(|g| (g.LM()[..k].to_vec(), *g)).collect();
    let minimal: Vec<&PolyElement<RationalField>> = leading
        .iter()
        .filter(|(m, _)| !leading.iter().any(|(m2, _)| m2 != m && monomial_divides(m2, m)))
        .map(|(_, g)| *g)
        .collect();
    let mut coefficients: Vec<Basic> = Vec::new();
    for g in minimal {
        for f in irreducible_factors(&leading_coefficient(g, k, parameters)?, parameters)? {
            if !coefficients.contains(&f) {
                coefficients.push(f);
            }
        }
    }
    let mut all_nonzero: Vec<Basic> = nonzero.clone();
    all_nonzero.extend(coefficients.iter().cloned());
    branches.push(Branch {
        equations: equations.clone(),
        nonzero: all_nonzero,
        basis: basis.clone(),
    });
    for (i, h) in coefficients.iter().enumerate() {
        let mut more_equations: Vec<Basic> = equations.clone();
        more_equations.push(h.clone());
        let mut more_nonzero: Vec<Basic> = nonzero.clone();
        more_nonzero.extend(coefficients[..i].iter().cloned());
        cgs(more_equations, more_nonzero, &basis, variables, parameters, branches)?;
    }
    Ok(())
}

/// The comprehensive Gröbner system of the polynomials in the variables
/// with coefficients depending on the parameters
/// (`comprehensive_groebner_system`).
pub fn comprehensive_groebner_system(
    polynomials: &[Basic],
    variables: &[Basic],
    parameters: &[Basic],
) -> PolyResult<Vec<Branch>> {
    let mut branches: Vec<Branch> = Vec::new();
    cgs(
        Vec::new(),
        Vec::new(),
        polynomials,
        variables,
        parameters,
        &mut branches,
    )?;
    Ok(branches)
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

/// The condition on the parameters (the other symbols) for the system of
/// equations and inequations to have a complex solution in the variables
/// (`exists_complex`).
pub fn exists_complex(equations: &[Basic], inequations: &[Basic], variables: &[Basic]) -> PolyResult<Basic> {
    let mut variables: Vec<Basic> = variables.to_vec();
    let mut polynomials: Vec<Basic> = equations.to_vec();
    for g in inequations {
        let z: Basic = Dummy::new("z", &[]).0;
        variables.push(z.clone());
        polynomials.push(crate::core::add::sub(
            &crate::core::mul::mul(&[z, g.clone()])?,
            &Basic::one(),
        )?);
    }
    let mut parameters: Vec<Basic> = Vec::new();
    for p in &polynomials {
        for s in p.free_symbols() {
            if !variables.contains(&s) && !parameters.contains(&s) {
                parameters.push(s);
            }
        }
    }
    parameters.sort_by_key(symbol_name);
    if parameters.is_empty() {
        if polynomials.is_empty() {
            return Ok(boolean(true));
        }
        return Ok(boolean(!ideal(&polynomials, &variables)?.is_whole_ring()?));
    }
    let mut conditions: Vec<Basic> = Vec::new();
    for branch in comprehensive_groebner_system(&polynomials, &variables, &parameters)? {
        if branch.solvable() {
            conditions.push(branch.condition()?);
        }
    }
    Ok(or_(&conditions))
}

// ---------------------------------------------------------------------------
// formulas

/// `lhs - rhs` of an equation or inequation with rational coefficients
/// (`_polynomial_atom`).
fn polynomial_atom(atom: &Basic) -> PolyResult<Basic> {
    if !matches!(
        atom.head(),
        Head::Relational(crate::core::basic::RelOp::Eq | crate::core::basic::RelOp::Ne)
    ) {
        return Err(value_error(format!(
            "only equations and inequations are meaningful over the complex numbers, got {}",
            atom
        )));
    }
    let e: Basic = crate::core::add::sub(&atom.args()[0], &atom.args()[1])?;
    let mut names: Vec<String> = e.free_symbols().iter().map(symbol_name).collect();
    names.sort();
    names.dedup();
    if super::polyutils::dict_from_expr(&e, &names, &QQ).is_err() {
        return Err(value_error(format!(
            "{} is not a polynomial relation with rational coefficients",
            atom
        )));
    }
    Ok(e)
}

/// The literals of the conjunctions of a disjunctive normal form
/// (`_disjunctive_form`).
fn disjunctive_form(formula: &Basic, positive: bool) -> PolyResult<Vec<Vec<Basic>>> {
    match formula.head() {
        Head::BooleanTrue => return Ok(if positive { vec![Vec::new()] } else { Vec::new() }),
        Head::BooleanFalse => return Ok(if positive { Vec::new() } else { vec![Vec::new()] }),
        Head::Not => return disjunctive_form(&formula.args()[0], !positive),
        Head::And | Head::Or | Head::Implies | Head::Equivalent | Head::Xor | Head::Ite => {}
        _ => return Ok(vec![vec![if positive { formula.clone() } else { not_(formula) }]]),
    }
    let args: &[Basic] = formula.args();
    match formula.head() {
        Head::And | Head::Or => {
            if matches!(formula.head(), Head::And) == positive {
                let mut result: Vec<Vec<Basic>> = vec![Vec::new()];
                for a in args {
                    let parts: Vec<Vec<Basic>> = disjunctive_form(a, positive)?;
                    result = result
                        .iter()
                        .flat_map(|left| parts.iter().map(move |right| [left.clone(), right.clone()].concat()))
                        .collect();
                }
                Ok(result)
            } else {
                let mut result: Vec<Vec<Basic>> = Vec::new();
                for a in args {
                    result.extend(disjunctive_form(a, positive)?);
                }
                Ok(result)
            }
        }
        Head::Implies => disjunctive_form(&or_(&[not_(&args[0]), args[1].clone()]), positive),
        Head::Equivalent => {
            let negated: Vec<Basic> = args.iter().map(not_).collect();
            disjunctive_form(&or_(&[and_(args), and_(&negated)]), positive)
        }
        Head::Xor => {
            let mut parity: Basic = args[0].clone();
            for a in &args[1..] {
                parity = or_(&[and_(&[parity.clone(), not_(a)]), and_(&[not_(&parity), a.clone()])]);
            }
            disjunctive_form(&parity, positive)
        }
        _ => disjunctive_form(
            &or_(&[
                and_(&[args[0].clone(), args[1].clone()]),
                and_(&[not_(&args[0]), args[2].clone()]),
            ]),
            positive,
        ),
    }
}

/// The equations and inequations of each conjunction of the disjunctive
/// normal form (`_conjunctions`).
fn conjunctions(formula: &Basic) -> PolyResult<Vec<(Vec<Basic>, Vec<Basic>)>> {
    let mut result: Vec<(Vec<Basic>, Vec<Basic>)> = Vec::new();
    let mut seen: Vec<Basic> = Vec::new();
    for literals in disjunctive_form(formula, true)? {
        let conjunction: Basic = and_(&literals);
        if matches!(conjunction.head(), Head::BooleanFalse) || seen.contains(&conjunction) {
            continue;
        }
        seen.push(conjunction.clone());
        let (mut equations, mut inequations) = (Vec::new(), Vec::new());
        let items: Vec<Basic> = match conjunction.head() {
            Head::And => conjunction.args().to_vec(),
            Head::BooleanTrue => Vec::new(),
            _ => vec![conjunction.clone()],
        };
        for literal in &items {
            if !matches!(literal.head(), Head::Relational(_)) {
                return Err(value_error(format!("{} is not a polynomial relation", literal)));
            }
            let e: Basic = polynomial_atom(literal)?;
            if matches!(literal.head(), Head::Relational(crate::core::basic::RelOp::Eq)) {
                equations.push(e);
            } else {
                inequations.push(e);
            }
        }
        result.push((equations, inequations));
    }
    Ok(result)
}

fn eliminate_block(formula: &Basic, variables: &[Basic]) -> PolyResult<Basic> {
    let mut disjuncts: Vec<Basic> = Vec::new();
    for (equations, inequations) in conjunctions(formula)? {
        disjuncts.push(exists_complex(&equations, &inequations, variables)?);
    }
    Ok(or_(&disjuncts))
}

/// Eliminate the quantifiers (outermost first in `prefix`) of a Boolean
/// combination of polynomial equations and inequations over the complex
/// numbers (`complex_quantifier_elimination`).
pub fn complex_quantifier_elimination(formula: &Basic, prefix: &[(Quantifier, Basic)]) -> PolyResult<Basic> {
    let mut current: Basic = formula.clone();
    let mut remaining: Vec<(Quantifier, Basic)> = prefix.to_vec();
    while let Some(&(kind, _)) = remaining.last() {
        let mut block: Vec<Basic> = Vec::new();
        while remaining.last().is_some_and(|(k, _)| *k == kind) {
            block.insert(0, remaining.pop().expect("an element").1);
        }
        current = match kind {
            Quantifier::Exists => eliminate_block(&current, &block)?,
            Quantifier::Forall => not_(&eliminate_block(&not_(&current), &block)?),
        };
    }
    reduce_complex(&current)
}

fn reduce_conjunction(equations: &[Basic], inequations: &[Basic]) -> PolyResult<Basic> {
    let mut symbols: Vec<Basic> = Vec::new();
    for e in equations.iter().chain(inequations) {
        for s in e.free_symbols() {
            if !symbols.contains(&s) {
                symbols.push(s);
            }
        }
    }
    symbols.sort_by_key(symbol_name);
    let zero: Basic = Basic::zero();
    if symbols.is_empty() {
        let mut parts: Vec<Basic> = Vec::new();
        for e in equations {
            parts.push(eq(e, &zero)?);
        }
        for g in inequations {
            parts.push(ne(g, &zero)?);
        }
        return Ok(and_(&parts));
    }
    let equations: Vec<Basic> = equations.iter().filter(|e| **e != zero).cloned().collect();
    let mut id: Option<Ideal<RationalField>> = None;
    let mut basis: Vec<Basic> = Vec::new();
    if !equations.is_empty() {
        let i: Ideal<RationalField> = ideal(&equations, &symbols)?;
        if i.is_whole_ring()? {
            return Ok(boolean(false));
        }
        basis = i
            .groebner_basis(None)?
            .iter()
            .map(|g| i.to_expr(g))
            .collect::<PolyResult<_>>()?;
        id = Some(i);
    }
    let mut factors: Vec<Basic> = Vec::new();
    for g in inequations {
        let mut reduced: Basic = g.clone();
        if let Some(i) = &id {
            let remainder: PolyElement<RationalField> = i.normal_form(&PolyLike::Expr(g), None)?;
            if remainder.is_zero() {
                return Ok(boolean(false));
            }
            reduced = i.to_expr(&remainder)?;
        } else if *g == zero {
            return Ok(boolean(false));
        }
        if reduced.free_symbols().is_empty() {
            continue;
        }
        for f in irreducible_factors(&reduced, &symbols)? {
            if let Some(i) = &id {
                if i.radical_contains(&PolyLike::Expr(&f))? {
                    return Ok(boolean(false));
                }
                let mut with: Vec<Basic> = basis.clone();
                with.push(f.clone());
                if ideal(&with, &symbols)?.is_whole_ring()? {
                    continue;
                }
            }
            if !factors.contains(&f) {
                factors.push(f);
            }
        }
    }
    let mut parts: Vec<Basic> = Vec::new();
    for g in &basis {
        parts.push(eq(g, &zero)?);
    }
    for f in &factors {
        parts.push(ne(f, &zero)?);
    }
    Ok(and_(&parts))
}

/// Simplify a quantifier-free Boolean combination of polynomial equations
/// and inequations over the complex numbers (`reduce_complex`).
pub fn reduce_complex(formula: &Basic) -> PolyResult<Basic> {
    if matches!(formula.head(), Head::BooleanTrue | Head::BooleanFalse) {
        return Ok(formula.clone());
    }
    let mut disjuncts: Vec<Basic> = Vec::new();
    for (equations, inequations) in conjunctions(formula)? {
        disjuncts.push(reduce_conjunction(&equations, &inequations)?);
    }
    Ok(or_(&disjuncts))
}
