//! Port of `sympy_extras/solvers/lie.py`: Lie point symmetries of
//! ordinary and partial differential equations. The prolongation of a
//! vector field `X = sum xi^i d/dx_i + sum eta^a d/du^a` (Olver's
//! recursive formula) applied to the equations, the leading derivatives
//! eliminated, gives the determining equations; they are solved with a
//! polynomial ansatz of bounded degree (a linear system over `QQ` or the
//! field of rational functions of the parameters), or kept as a linear PDE
//! system whose Janet basis gives the dimension of the symmetry algebra.

#![allow(clippy::mutable_key_type, clippy::type_complexity, clippy::needless_range_loop)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use crate::core::basic::{Basic, Head, UndefinedFunction};
use crate::core::expr::{factors_of, terms_of};
use crate::core::symbol::{Dummy, Symbol};
use crate::error::{not_ported, value_error, PolyError, PolyResult};
use crate::numbers::*;
use crate::polys::domains::Domain;

type R = PolyResult<Basic>;

/// A jet coordinate `u^a_J`: the dependent variable and the multi-index.
pub type Jet = (usize, Vec<u32>);

fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn diff1(e: &Basic, v: &Basic) -> R {
    crate::core::function::derivative(e, v)
}
fn replaced(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    crate::core::function::xreplace_evaluated(e, rule)
}
fn expand_power_exp(e: &Basic) -> R {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: true,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}
fn numer(e: &Basic) -> R {
    crate::simplify::radsimp::numer_of(e)
}
fn sym(name: &str) -> Basic {
    Symbol::new(name, &[]).0
}
fn name_of(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Undefined(u) => u.name.clone(),
        _ => s.to_string(),
    }
}
fn is_dummy(s: &Basic) -> bool {
    matches!(s.head(), Head::Dummy(..))
}
fn unit(p: usize, i: usize) -> Vec<u32> {
    (0..p).map(|j| u32::from(j == i)).collect()
}

/// The jet coordinates of the dependent variables (`JetSpace`).
#[derive(Debug)]
pub struct JetSpace {
    pub x: Vec<Basic>,
    pub functions: Vec<Basic>,
    pub u: Vec<Basic>,
    jets: RefCell<Vec<(Jet, Basic)>>,
}

impl JetSpace {
    pub fn new(functions: &[Basic]) -> PolyResult<Rc<JetSpace>> {
        let Some(first) = functions.first() else {
            return Err(value_error("at least one dependent variable is needed".to_string()));
        };
        let args: Vec<Basic> = first.args().to_vec();
        for f in functions {
            if f.args() != args.as_slice() {
                return Err(value_error(
                    "the dependent variables must have the same arguments".to_string(),
                ));
            }
        }
        if args.iter().any(|a| !a.is_symbol()) {
            return Err(PolyError::Type(
                "the arguments of the dependent variables must be symbols".to_string(),
            ));
        }
        let u: Vec<Basic> = functions.iter().map(|f| sym(&name_of(f))).collect();
        let zero: Vec<u32> = vec![0; args.len()];
        let jets: Vec<(Jet, Basic)> = u
            .iter()
            .enumerate()
            .map(|(i, s)| ((i, zero.clone()), s.clone()))
            .collect();
        Ok(Rc::new(JetSpace {
            x: args,
            functions: functions.to_vec(),
            u,
            jets: RefCell::new(jets),
        }))
    }

    pub fn p(&self) -> usize {
        self.x.len()
    }

    pub fn q(&self) -> usize {
        self.u.len()
    }

    /// The jet coordinate `u^a_J`.
    pub fn jet(&self, a: usize, j: &[u32]) -> Basic {
        let key: Jet = (a, j.to_vec());
        if let Some((_, s)) = self.jets.borrow().iter().find(|(k, _)| *k == key) {
            return s.clone();
        }
        let suffix: String = self
            .x
            .iter()
            .zip(j)
            .map(|(x, k)| name_of(x).repeat(*k as usize))
            .collect();
        let s: Basic = sym(&format!("{}_{}", name_of(&self.u[a]), suffix));
        self.jets.borrow_mut().push((key, s.clone()));
        s
    }

    /// The `(a, J)` of a jet coordinate, by its name (`index`).
    pub fn index(&self, s: &Basic) -> Option<Jet> {
        if let Some((k, _)) = self.jets.borrow().iter().find(|(_, v)| v == s) {
            return Some(k.clone());
        }
        if is_dummy(s) || !s.is_symbol() {
            return None;
        }
        let parsed: Jet = self.parse(&name_of(s))?;
        self.jets.borrow_mut().push((parsed.clone(), s.clone()));
        Some(parsed)
    }

    fn parse(&self, name: &str) -> Option<Jet> {
        let mut names: Vec<(usize, String)> = self.x.iter().enumerate().map(|(i, v)| (i, name_of(v))).collect();
        names.sort_by_key(|(_, n)| std::cmp::Reverse(n.len()));
        for (a, u) in self.u.iter().enumerate() {
            let prefix: String = format!("{}_", name_of(u));
            let Some(mut suffix) = name.strip_prefix(&prefix) else {
                continue;
            };
            let mut counts: Vec<u32> = vec![0; self.p()];
            while !suffix.is_empty() {
                let (i, n) = names.iter().find(|(_, n)| suffix.starts_with(n.as_str()))?;
                counts[*i] += 1;
                suffix = &suffix[n.len()..];
            }
            if counts.iter().sum::<u32>() > 0 {
                return Some((a, counts));
            }
        }
        None
    }

    /// The jet coordinates occurring in `expr`, sorted (`jets_in`).
    pub fn jets_in(&self, expr: &Basic) -> Vec<Jet> {
        let mut found: Vec<Jet> = expr.free_symbols().iter().filter_map(|s| self.index(s)).collect();
        found.sort();
        found
    }

    /// The highest order of a jet coordinate in `expr` (`order`).
    pub fn order(&self, expr: &Basic) -> u32 {
        self.jets_in(expr)
            .iter()
            .map(|(_, j)| j.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// The derivatives of the dependent variables as jet coordinates
    /// (`to_jet`).
    pub fn to_jet(&self, expr: &Basic) -> R {
        // an unevaluated derivative of an expression which is not one of
        // the dependent variables, Derivative(u**2*u_x, x), is evaluated
        // first: replacing u by a symbol inside it would lose its
        // dependence on x
        let expr: &Basic = &self.evaluated_derivatives(expr)?;
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for d in expr.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Derivative))) {
            let f: &Basic = &d.args()[0];
            let Some(a) = self.functions.iter().position(|g| g == f) else {
                continue;
            };
            let mut counts: Vec<u32> = vec![0; self.p()];
            for vc in &d.args()[1..] {
                let (v, n) = (&vc.args()[0], &vc.args()[1]);
                let Some(i) = self.x.iter().position(|x| x == v) else {
                    return Err(value_error(format!("derivative with respect to {}", v)));
                };
                counts[i] += n
                    .as_rational()
                    .and_then(|r| u32::try_from(&numer_of_rat(r)).ok())
                    .unwrap_or(1);
            }
            rule.insert(d.clone(), self.jet(a, &counts));
        }
        for (a, f) in self.functions.iter().enumerate() {
            rule.insert(f.clone(), self.u[a].clone());
        }
        replaced(expr, &rule)
    }

    /// `expr` with the derivatives of expressions other than the dependent
    /// variables evaluated (`_evaluated_derivatives`).
    fn evaluated_derivatives(&self, expr: &Basic) -> R {
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for d in expr.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Derivative))) {
            if !self.functions.contains(&d.args()[0]) {
                let value: Basic = crate::core::function::doit(&d)?;
                rule.insert(d, value);
            }
        }
        if rule.is_empty() {
            return Ok(expr.clone());
        }
        replaced(expr, &rule)
    }

    /// The jet coordinates as derivatives of the functions (`from_jet`).
    pub fn from_jet(&self, expr: &Basic) -> R {
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for ((a, j), s) in self.jets.borrow().iter() {
            if j.iter().sum::<u32>() == 0 {
                rule.insert(s.clone(), self.functions[*a].clone());
            } else {
                let counts: Vec<(Basic, u64)> = self
                    .x
                    .iter()
                    .zip(j)
                    .filter(|(_, k)| **k > 0)
                    .map(|(x, k)| (x.clone(), u64::from(*k)))
                    .collect();
                rule.insert(s.clone(), crate::core::function::diff(&self.functions[*a], &counts)?);
            }
        }
        canonical_derivatives(&replaced(expr, &rule)?)
    }

    /// The total derivative `D_i` (`total_derivative`).
    pub fn total_derivative(&self, expr: &Basic, i: usize) -> R {
        let mut result: Basic = diff1(expr, &self.x[i])?;
        for (a, j) in self.jets_in(expr) {
            let partial: Basic = diff1(expr, &self.jet(a, &j))?;
            if partial != Basic::zero() {
                let mut next: Vec<u32> = j.clone();
                next[i] += 1;
                result = add(&[result, mul(&[self.jet(a, &next), partial])?])?;
            }
        }
        Ok(result)
    }

    /// The monomials in the independent and dependent variables of total
    /// degree at most `degree` (`monomials`).
    pub fn monomials(&self, degree: u32) -> PolyResult<Vec<Basic>> {
        let variables: Vec<Basic> = self.x.iter().chain(&self.u).cloned().collect();
        let mut exponents: Vec<Vec<u32>> = vec![vec![]];
        for _ in &variables {
            exponents = exponents
                .iter()
                .flat_map(|e| {
                    let used: u32 = e.iter().sum();
                    (0..=degree - used).map(move |k| e.iter().cloned().chain(std::iter::once(k)).collect::<Vec<u32>>())
                })
                .collect();
        }
        let mut found: Vec<(u32, String, Basic)> = Vec::new();
        for e in exponents {
            let factors: Vec<Basic> = variables
                .iter()
                .zip(&e)
                .filter(|(_, k)| **k > 0)
                .map(|(v, k)| pow(v, &Basic::int(i64::from(*k))))
                .collect::<PolyResult<_>>()?;
            let m: Basic = mul(&factors)?;
            found.push((e.iter().sum(), m.to_string(), m));
        }
        found.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        Ok(found.into_iter().map(|(_, _, m)| m).collect())
    }
}

fn numer_of_rat(r: &Rat) -> Int {
    crate::numbers::numer(r)
}

/// A vector field on the independent and dependent variables
/// (`Symmetry`).
#[derive(Clone, Debug)]
pub struct Symmetry {
    pub jets: Rc<JetSpace>,
    pub xi: Vec<Basic>,
    pub eta: Vec<Basic>,
}

impl PartialEq for Symmetry {
    fn eq(&self, other: &Symmetry) -> bool {
        self.xi == other.xi && self.eta == other.eta
    }
}

impl Symmetry {
    pub fn new(jets: &Rc<JetSpace>, xi: Vec<Basic>, eta: Vec<Basic>) -> PolyResult<Symmetry> {
        if xi.len() != jets.p() || eta.len() != jets.q() {
            return Err(value_error(format!("expected {} xi and {} eta", jets.p(), jets.q())));
        }
        Ok(Symmetry {
            jets: jets.clone(),
            xi,
            eta,
        })
    }

    /// The vector field written with `d/dx` and `d/du` (`generator`).
    pub fn generator(&self) -> String {
        let mut terms: Vec<String> = Vec::new();
        for (c, v) in self
            .xi
            .iter()
            .zip(&self.jets.x)
            .chain(self.eta.iter().zip(&self.jets.u))
        {
            if *c == Basic::zero() {
                continue;
            }
            let name: String = name_of(v);
            terms.push(if *c == Basic::one() {
                format!("d/d{}", name)
            } else if c.is_add() {
                format!("({})*d/d{}", c, name)
            } else {
                format!("{}*d/d{}", c, name)
            });
        }
        if terms.is_empty() {
            return "0".to_string();
        }
        terms.join(" + ").replace("+ -", "- ")
    }

    pub fn is_zero(&self) -> bool {
        self.xi.iter().chain(&self.eta).all(|c| *c == Basic::zero())
    }

    /// `X f` for a function of the independent and dependent variables.
    pub fn apply_point(&self, f: &Basic) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (c, v) in self
            .xi
            .iter()
            .zip(&self.jets.x)
            .chain(self.eta.iter().zip(&self.jets.u))
        {
            terms.push(mul(&[c.clone(), diff1(f, v)?])?);
        }
        add(&terms)
    }

    /// The Lie bracket `[X, Y]` (`commutator`).
    pub fn commutator(&self, other: &Symmetry) -> PolyResult<Symmetry> {
        let bracket = |a: &[Basic], b: &[Basic]| -> PolyResult<Vec<Basic>> {
            a.iter()
                .zip(b)
                .map(|(p, q)| expand(&sub(&self.apply_point(q)?, &other.apply_point(p)?)?))
                .collect()
        };
        Symmetry::new(
            &self.jets,
            bracket(&self.xi, &other.xi)?,
            bracket(&self.eta, &other.eta)?,
        )
    }

    /// The characteristic `Q^a = eta^a - sum xi^i u^a_i`.
    pub fn characteristic(&self) -> PolyResult<Vec<Basic>> {
        let jets = &self.jets;
        (0..jets.q())
            .map(|a| {
                let mut q: Basic = self.eta[a].clone();
                for i in 0..jets.p() {
                    q = sub(&q, &mul(&[self.xi[i].clone(), jets.jet(a, &unit(jets.p(), i))])?)?;
                }
                Ok(q)
            })
            .collect()
    }

    /// The coefficients `eta^a_J` of the prolongation for `|J| <= order`
    /// (`prolongation`).
    pub fn prolongation(&self, order: u32) -> PolyResult<Vec<(Jet, Basic)>> {
        let jets = &self.jets;
        let p: usize = jets.p();
        let zero: Vec<u32> = vec![0; p];
        let mut coefficients: Vec<(Jet, Basic)> = Vec::new();
        let mut d_xi: Vec<Vec<Basic>> = Vec::new();
        for i in 0..p {
            d_xi.push(
                (0..p)
                    .map(|k| jets.total_derivative(&self.xi[k], i))
                    .collect::<PolyResult<_>>()?,
            );
        }
        for a in 0..jets.q() {
            coefficients.push(((a, zero.clone()), self.eta[a].clone()));
        }
        for n in 0..order {
            for ((a, j), phi) in coefficients.clone() {
                if j.iter().sum::<u32>() != n {
                    continue;
                }
                for i in 0..p {
                    if (i + 1..p).any(|jj| j[jj] > 0) {
                        continue;
                    }
                    let mut j_i: Vec<u32> = j.clone();
                    j_i[i] += 1;
                    if coefficients.iter().any(|(k, _)| *k == (a, j_i.clone())) {
                        continue;
                    }
                    let mut value: Basic = jets.total_derivative(&phi, i)?;
                    for k in 0..p {
                        let mut j_k: Vec<u32> = j.clone();
                        j_k[k] += 1;
                        value = sub(&value, &mul(&[jets.jet(a, &j_k), d_xi[i][k].clone()])?)?;
                    }
                    coefficients.push(((a, j_i), expand(&value)?));
                }
            }
        }
        Ok(coefficients)
    }

    /// `pr X` applied to an expression in the jet coordinates (`apply`).
    pub fn apply(&self, expr: &Basic) -> R {
        let jets = &self.jets;
        let coefficients = self.prolongation(jets.order(expr))?;
        let mut terms: Vec<Basic> = Vec::new();
        for (c, v) in self.xi.iter().zip(&jets.x) {
            terms.push(mul(&[c.clone(), diff1(expr, v)?])?);
        }
        for ((a, j), phi) in coefficients {
            let partial: Basic = diff1(expr, &jets.jet(a, &j))?;
            if partial != Basic::zero() {
                terms.push(mul(&[phi, partial])?);
            }
        }
        expand(&add(&terms)?)
    }
}

// -- the invariance condition

fn as_zero(e: &Basic) -> R {
    if matches!(e.head(), Head::Relational(crate::core::basic::RelOp::Eq)) {
        return sub(&e.args()[0], &e.args()[1]);
    }
    Ok(e.clone())
}

fn prepare(equations: &[Basic], functions: &[Basic]) -> PolyResult<(Rc<JetSpace>, Vec<Basic>)> {
    let jets: Rc<JetSpace> = JetSpace::new(functions)?;
    let eqs: Vec<Basic> = equations
        .iter()
        .map(|e| jets.to_jet(&as_zero(e)?))
        .collect::<PolyResult<_>>()?;
    Ok((jets, eqs))
}

fn solve_for(eq: &Basic, s: &Basic) -> PolyResult<Vec<Basic>> {
    if let Ok(poly) = crate::polys::polytools::Poly::new(eq, std::slice::from_ref(s)) {
        if poly.degree(0)? == 2 {
            return crate::polys::polyroots::roots_quadratic(&poly);
        }
    }
    Ok(crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        super::solvers::solve_equation(eq, s)
    })?
    .flatten()
    .unwrap_or_default())
}

/// The degree of `eq` as a polynomial in `s`, `-1` when it is not one
/// (`_degree_in`).
fn degree_in(eq: &Basic, s: &Basic) -> PolyResult<i64> {
    let Some(terms) = polynomial_terms(eq, std::slice::from_ref(s))? else {
        return Ok(-1);
    };
    Ok(terms.iter().map(|(m, _)| m[0] as i64).max().unwrap_or(0))
}

/// Every equation solved for its highest derivative, preferring one it is
/// linear in (`_leading_derivatives`); with `polynomial`, an equation which
/// is a polynomial of degree at least two in a highest derivative is not
/// solved: the derivative is recorded under the index of the equation.
fn leading_derivatives_of(
    jets: &JetSpace,
    eqs: &[Basic],
    mut polynomial: Option<&mut Vec<(usize, Basic)>>,
) -> PolyResult<Vec<(Basic, Basic)>> {
    let mut solved: Vec<(Basic, Basic)> = Vec::new();
    for (index, eq) in eqs.iter().enumerate() {
        let mut candidates: Vec<Jet> = jets.jets_in(eq);
        candidates.sort_by(|a, b| (b.1.iter().sum::<u32>(), &b.1, b.0).cmp(&(a.1.iter().sum::<u32>(), &a.1, a.0)));
        let top: u32 = candidates.first().map(|(_, j)| j.iter().sum()).unwrap_or(0);
        let mut found: bool = false;
        for (a, j) in &candidates {
            if j.iter().sum::<u32>() < top {
                break;
            }
            let s: Basic = jets.jet(*a, j);
            if solved.iter().any(|(k, _)| *k == s) {
                continue;
            }
            let coefficient: Basic = diff1(eq, &s)?;
            if diff1(&coefficient, &s)? == Basic::zero() {
                let rest: Basic = sub(eq, &mul(&[coefficient.clone(), s.clone()])?)?;
                let value: Basic = crate::core::mul::div(&crate::core::mul::neg(&rest)?, &coefficient)?;
                solved.push((s, expand_power_exp(&value)?));
                found = true;
                break;
            }
        }
        if let (false, Some(poly)) = (found, polynomial.as_deref_mut()) {
            for (a, j) in &candidates {
                if j.iter().sum::<u32>() < top {
                    break;
                }
                let s: Basic = jets.jet(*a, j);
                if solved.iter().any(|(k, _)| *k == s) || poly.iter().any(|(_, v)| *v == s) {
                    continue;
                }
                if degree_in(eq, &s)? >= 2 {
                    poly.push((index, s));
                    found = true;
                    break;
                }
            }
        }
        if !found {
            for (a, j) in &candidates {
                let s: Basic = jets.jet(*a, j);
                if solved.iter().any(|(k, _)| *k == s) {
                    continue;
                }
                let roots: Vec<Basic> = solve_for(eq, &s)?;
                if let Some(r) = roots.into_iter().next() {
                    solved.push((s, r));
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return Err(not_ported(&format!(
                "cannot solve {} for a leading derivative",
                jets.from_jet(eq)?
            )));
        }
    }
    Ok(solved)
}

/// The highest derivative of every equation solved for
/// (`leading_derivatives`).
pub fn leading_derivatives(jets: &JetSpace, eqs: &[Basic]) -> PolyResult<Vec<(Basic, Basic)>> {
    leading_derivatives_of(jets, eqs, None)
}

/// The numerator of `e` with the solved leading derivatives substituted
/// (`_eliminated`).
fn eliminated(jets: &JetSpace, e: &Basic, leading: &[(Basic, Basic)]) -> R {
    let rule: HashMap<Basic, Basic> = leading.iter().cloned().collect();
    let mut c: Basic = e.clone();
    for _ in 0..leading.len() {
        let new: Basic = replaced(&c, &rule)?;
        if new == c {
            break;
        }
        c = new;
    }
    c = expand_power_exp(&c)?;
    let base: Vec<Basic> = jets.x.iter().chain(&jets.u).cloned().collect();
    let rational: bool =
        crate::core::expr::polynomial_like(&c, &base, crate::core::expr::PolynomialKind::Rational) == Some(true);
    if rational {
        numer(&crate::simplify::ratsimp::ratsimp(&c)?)
    } else {
        numer(&crate::polys::rationaltools::together(&c, false, true)?)
    }
}

/// The pseudo-remainder of `g` by the polynomial in `s` whose coefficients
/// are `f` (degree to coefficient), `None` when `g` is not a polynomial in
/// `s` (`_pseudo_remainder`).
fn pseudo_remainder(g: &Basic, f: &BTreeMap<u32, Basic>, s: &Basic) -> PolyResult<Option<Basic>> {
    let Some(terms) = polynomial_terms(g, std::slice::from_ref(s))? else {
        return Ok(None);
    };
    let mut remainder: BTreeMap<u32, Basic> = terms.into_iter().map(|(m, c)| (m[0], c)).collect();
    let d: u32 = *f.keys().max().expect("a nonzero polynomial");
    let lc: Basic = f[&d].clone();
    while let Some(n) = remainder.keys().max().copied().filter(|n| *n >= d) {
        crate::timeout::check()?;
        let c: Basic = remainder.remove(&n).expect("the leading term");
        let mut new: BTreeMap<u32, Basic> = BTreeMap::new();
        for (k, v) in &remainder {
            new.insert(*k, expand(&mul(&[lc.clone(), v.clone()])?)?);
        }
        for (k, fk) in f {
            if *k < d {
                let key: u32 = k + n - d;
                let old: Basic = new.get(&key).cloned().unwrap_or_else(Basic::zero);
                new.insert(key, expand(&sub(&old, &mul(&[c.clone(), fk.clone()])?)?)?);
            }
        }
        remainder = new.into_iter().filter(|(_, v)| *v != Basic::zero()).collect();
    }
    let mut parts: Vec<Basic> = Vec::new();
    for (k, c) in &remainder {
        parts.push(mul(&[c.clone(), crate::core::power::pow(s, &Basic::int(*k as i64))?])?);
    }
    Ok(Some(crate::core::add::add(&parts)?))
}

/// The invariance conditions (`_condition`): `pr X` applied to every
/// equation, on its solutions; an equation which is a polynomial of degree
/// at least two in its leading derivative is not solved for it, the
/// conditions are reduced modulo it, so that all its branches count.
fn condition(jets: &JetSpace, eqs: &[Basic], symmetry: &Symmetry) -> PolyResult<Vec<Basic>> {
    let mut polynomial: Vec<(usize, Basic)> = Vec::new();
    let leading: Vec<(Basic, Basic)> = leading_derivatives_of(jets, eqs, Some(&mut polynomial))?;
    let mut moduli: Vec<(Basic, BTreeMap<u32, Basic>)> = Vec::new();
    for (index, s) in &polynomial {
        let Some(terms) = polynomial_terms(&eliminated(jets, &eqs[*index], &leading)?, std::slice::from_ref(s))? else {
            continue;
        };
        let coefficients: BTreeMap<u32, Basic> = terms.into_iter().map(|(m, c)| (m[0], c)).collect();
        if coefficients.keys().max().is_some_and(|d| *d >= 1) {
            moduli.push((s.clone(), coefficients));
        }
    }
    let mut result: Vec<Basic> = Vec::new();
    for eq in eqs {
        let mut c: Basic = eliminated(jets, &symmetry.apply(eq)?, &leading)?;
        for (s, f) in &moduli {
            if let Some(remainder) = pseudo_remainder(&c, f, s)? {
                c = remainder;
            }
        }
        result.push(c);
    }
    Ok(result)
}

/// `pr X(Delta)` on the solutions of the equations, as numerators
/// (`infinitesimal_condition`).
pub fn infinitesimal_condition(equations: &[Basic], functions: &[Basic], symmetry: &[Basic]) -> PolyResult<Vec<Basic>> {
    let (jets, eqs) = prepare(equations, functions)?;
    let p: usize = jets.p();
    let s: Symmetry = Symmetry::new(&jets, symmetry[..p].to_vec(), symmetry[p..].to_vec())?;
    condition(&jets, &eqs, &s)
}

/// Whether the vector field (the `xi` then the `eta`) is a point
/// symmetry of the equations (`check_symmetry`).
pub fn check_symmetry(equations: &[Basic], functions: &[Basic], symmetry: &[Basic]) -> PolyResult<bool> {
    for c in infinitesimal_condition(equations, functions, symmetry)? {
        if c != Basic::zero()
            && crate::simplify::simplify::simplify(&c, &crate::simplify::simplify::Options::default())? != Basic::zero()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

// -- the ansatz

fn coefficient_of(poly: &crate::polys::polytools::Poly, monomial: &[u32]) -> PolyResult<Basic> {
    Ok(poly
        .terms()?
        .into_iter()
        .find(|(m, _)| m.as_slice() == monomial)
        .map(|(_, c)| c)
        .unwrap_or_else(Basic::zero))
}

fn symbolic_powers(expr: &Basic, unknowns: &BTreeSet<Basic>, powers: &mut Vec<((Basic, Basic), Basic)>) -> R {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for power in expr.atoms(Some(&|n: &Basic| n.is_pow())) {
        let (base, exponent) = (&power.args()[0], &power.args()[1]);
        let mut parameters: Vec<Basic> = exponent
            .free_symbols()
            .into_iter()
            .filter(|s| !unknowns.contains(s))
            .collect();
        parameters.sort_by_key(name_of);
        if parameters.is_empty() {
            continue;
        }
        let Ok(poly) = crate::polys::polytools::Poly::new(exponent, &parameters) else {
            continue;
        };
        if poly.total_degree() > 1 {
            continue;
        }
        let n: usize = parameters.len();
        let mut value: Basic = pow(base, &coefficient_of(&poly, &vec![0; n])?)?;
        for (i, parameter) in parameters.iter().enumerate() {
            let c: Basic = coefficient_of(&poly, &unit(n, i))?;
            if c == Basic::zero() {
                continue;
            }
            let key = (base.clone(), parameter.clone());
            let symbol: Basic = match powers.iter().find(|(k, _)| *k == key) {
                Some((_, s)) => s.clone(),
                None => {
                    let s: Basic = Dummy::new(&format!("P{}", powers.len()), &[]).0;
                    powers.push((key, s.clone()));
                    s
                }
            };
            value = mul(&[value, pow(&symbol, &c)?])?;
        }
        rule.insert(power.clone(), value);
    }
    if rule.is_empty() {
        Ok(expr.clone())
    } else {
        replaced(expr, &rule)
    }
}

fn split(expr: &Basic, unknowns: &[Basic], variables: &BTreeSet<Basic>) -> PolyResult<Vec<Basic>> {
    let mut expr: Basic = expand(expr)?;
    if expr == Basic::zero() {
        return Ok(vec![]);
    }
    let unknown_set: BTreeSet<Basic> = unknowns.iter().cloned().collect();
    let mut powers: Vec<((Basic, Basic), Basic)> = Vec::new();
    let with_powers: Basic = symbolic_powers(&expr, &unknown_set, &mut powers)?;
    let mut variables: BTreeSet<Basic> = variables.clone();
    if with_powers != expr {
        expr = expand(&numer(&crate::polys::rationaltools::together(
            &with_powers,
            false,
            true,
        )?)?)?;
        variables.extend(powers.iter().map(|(_, s)| s.clone()));
    }
    let mut grouped: Vec<(Basic, Vec<Basic>)> = Vec::new();
    for term in terms_of(&expr) {
        let (mut monomial, mut coefficient): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
        for factor in factors_of(&term) {
            let symbols = factor.free_symbols();
            if symbols.iter().any(|s| variables.contains(s)) && !symbols.iter().any(|s| unknown_set.contains(s)) {
                monomial.push(factor);
            } else {
                coefficient.push(factor);
            }
        }
        let key: Basic = mul(&monomial)?;
        let value: Basic = mul(&coefficient)?;
        match grouped.iter().position(|(k, _)| *k == key) {
            Some(i) => grouped[i].1.push(value),
            None => grouped.push((key, vec![value])),
        }
    }
    grouped.into_iter().map(|(_, cs)| add(&cs)).collect()
}

fn polynomial_terms(expr: &Basic, gens: &[Basic]) -> PolyResult<Option<Vec<(Vec<u32>, Basic)>>> {
    let mut collected: BTreeMap<Vec<u32>, Vec<Basic>> = BTreeMap::new();
    for term in terms_of(&expand(expr)?) {
        let mut exponents: Vec<u32> = vec![0; gens.len()];
        let mut coefficient: Vec<Basic> = Vec::new();
        for factor in factors_of(&term) {
            if !factor.free_symbols().iter().any(|s| gens.contains(s)) {
                coefficient.push(factor);
                continue;
            }
            let (base, exponent) = if factor.is_pow() {
                (factor.args()[0].clone(), factor.args()[1].clone())
            } else {
                (factor.clone(), Basic::one())
            };
            let Some(position) = gens.iter().position(|g| *g == base) else {
                return Ok(None);
            };
            let Some(k) = exponent
                .as_rational()
                .filter(|v| crate::core::expr::is_integer_class(&exponent) && !rat_is_negative(v))
                .map(numer_of_rat)
            else {
                return Ok(None);
            };
            exponents[position] += u32::try_from(&k).unwrap_or(0);
        }
        collected.entry(exponents).or_default().push(mul(&coefficient)?);
    }
    let mut result: Vec<(Vec<u32>, Basic)> = Vec::new();
    for (m, cs) in collected.into_iter().rev() {
        let c: Basic = add(&cs)?;
        if c != Basic::zero() {
            result.push((m, c));
        }
    }
    Ok(Some(result))
}

struct Ansatz {
    xi: Vec<Basic>,
    eta: Vec<Basic>,
    unknowns: Vec<Basic>,
    multiplied: Vec<Basic>,
}

fn ansatz(jets: &JetSpace, degree: u32, basis: &[Basic]) -> PolyResult<Ansatz> {
    let monomials: Vec<Basic> = jets.monomials(degree)?;
    let mut factors: Vec<Basic> = vec![Basic::one()];
    factors.extend(basis.iter().cloned());
    let mut terms: Vec<Basic> = Vec::new();
    for b in &factors {
        for m in &monomials {
            terms.push(mul(&[m.clone(), b.clone()])?);
        }
    }
    let mut unknowns: Vec<Basic> = Vec::new();
    let mut multiplied: Vec<Basic> = Vec::new();
    let make = |unknowns: &mut Vec<Basic>, multiplied: &mut Vec<Basic>| -> R {
        let mut parts: Vec<Basic> = Vec::new();
        for term in &terms {
            let c: Basic = Dummy::new(&format!("c{}", unknowns.len()), &[]).0;
            unknowns.push(c.clone());
            multiplied.push(term.clone());
            parts.push(mul(&[c, term.clone()])?);
        }
        add(&parts)
    };
    let xi: Vec<Basic> = (0..jets.p())
        .map(|_| make(&mut unknowns, &mut multiplied))
        .collect::<PolyResult<_>>()?;
    let eta: Vec<Basic> = (0..jets.q())
        .map(|_| make(&mut unknowns, &mut multiplied))
        .collect::<PolyResult<_>>()?;
    Ok(Ansatz {
        xi,
        eta,
        unknowns,
        multiplied,
    })
}

fn linear_equations(jets: &Rc<JetSpace>, eqs: &[Basic], a: &Ansatz) -> PolyResult<Vec<Basic>> {
    let conditions: Vec<Basic> = condition(jets, eqs, &Symmetry::new(jets, a.xi.clone(), a.eta.clone())?)?;
    let mut linear: Vec<Basic> = Vec::new();
    for c in conditions {
        let mut variables: BTreeSet<Basic> = jets.x.iter().chain(&jets.u).cloned().collect();
        for (aa, j) in jets.jets_in(&c) {
            variables.insert(jets.jet(aa, &j));
        }
        linear.extend(split(&c, &a.unknowns, &variables)?);
    }
    Ok(linear)
}

/// The determining equations of the ansatz: the jet space, the unknown
/// coefficients, the ansatz of `xi` and `eta`, the linear equations
/// (`determining_equations`).
pub fn determining_equations(
    equations: &[Basic],
    functions: &[Basic],
    degree: u32,
    basis: &[Basic],
) -> PolyResult<(Rc<JetSpace>, Vec<Basic>, Vec<Basic>, Vec<Basic>, Vec<Basic>)> {
    let (jets, eqs) = prepare(equations, functions)?;
    let a: Ansatz = ansatz(&jets, degree, basis)?;
    let linear: Vec<Basic> = linear_equations(&jets, &eqs, &a)?;
    Ok((jets, a.unknowns, a.xi, a.eta, linear))
}

fn canonical_derivatives(e: &Basic) -> R {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for d in e.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Derivative))) {
        let mut counts: Vec<(Basic, u64)> = d.args()[1..]
            .iter()
            .map(|vc| {
                (
                    vc.args()[0].clone(),
                    vc.args()[1]
                        .as_rational()
                        .and_then(|r| u64::try_from(&numer_of_rat(r)).ok())
                        .unwrap_or(1),
                )
            })
            .collect();
        let original: Vec<(Basic, u64)> = counts.clone();
        counts.sort_by_cached_key(|(v, _)| crate::core::sorting::sort_key(v));
        if counts != original {
            rule.insert(d.clone(), crate::core::function::derivative_node(&d.args()[0], &counts));
        }
    }
    if rule.is_empty() {
        Ok(e.clone())
    } else {
        replaced(e, &rule)
    }
}

fn undefined(name: &str, args: &[Basic]) -> Basic {
    Basic::new(
        Head::Undefined(Arc::new(UndefinedFunction::new(name, None, &[]))),
        args.to_vec(),
    )
}

/// The determining equations as a linear PDE system: the equations, the
/// unknown functions `xi`, `eta`, their arguments (`determining_system`).
pub fn determining_system(
    equations: &[Basic],
    functions: &[Basic],
) -> PolyResult<(Vec<Basic>, Vec<Basic>, Vec<Basic>)> {
    let (jets, eqs) = prepare(equations, functions)?;
    let base: Vec<Basic> = jets.x.iter().chain(&jets.u).cloned().collect();
    let mut unknowns: Vec<Basic> = (0..jets.p()).map(|i| undefined(&format!("xi{}", i), &base)).collect();
    unknowns.extend((0..jets.q()).map(|a| undefined(&format!("eta{}", a), &base)));
    let xi: Vec<Basic> = unknowns[..jets.p()].to_vec();
    let eta: Vec<Basic> = unknowns[jets.p()..].to_vec();
    let conditions: Vec<Basic> = condition(&jets, &eqs, &Symmetry::new(&jets, xi, eta)?)?;
    let mut system: Vec<Basic> = Vec::new();
    for c in conditions {
        let c: Basic = expand(&canonical_derivatives(&c)?)?;
        let gens: Vec<Basic> = jets
            .jets_in(&c)
            .into_iter()
            .filter(|(_, j)| j.iter().sum::<u32>() > 0)
            .map(|(a, j)| jets.jet(a, &j))
            .collect();
        if gens.is_empty() {
            if c != Basic::zero() {
                system.push(c);
            }
            continue;
        }
        let Some(cs) = polynomial_terms(&c, &gens)? else {
            return Err(not_ported(
                "the invariance condition is not polynomial in the derivatives",
            ));
        };
        system.extend(cs.into_iter().map(|(_, v)| v));
    }
    Ok((system, unknowns, base))
}

/// The Janet basis of the determining system: its dimension is that of
/// the symmetry algebra (`symmetry_janet_basis`).
pub fn symmetry_janet_basis(
    equations: &[Basic],
    functions: &[Basic],
) -> PolyResult<crate::polys::differential::janet::JanetBasis> {
    let (system, unknowns, base) = determining_system(equations, functions)?;
    crate::polys::differential::janet::janet_basis(&system, &unknowns, Some(&base), None)
}

// -- the linear algebra of the ansatz

fn constants(c: &Basic, names: &mut Vec<(Basic, Basic)>) -> R {
    if c.is_rational_class() || c.is_symbol() {
        return Ok(c.clone());
    }
    if c.is_add() {
        let parts: Vec<Basic> = c
            .args()
            .iter()
            .map(|a| constants(a, names))
            .collect::<PolyResult<_>>()?;
        return add(&parts);
    }
    let fs: Vec<Basic> = factors_of(c);
    if fs.len() > 1 {
        let parts: Vec<Basic> = fs.iter().map(|a| constants(a, names)).collect::<PolyResult<_>>()?;
        return mul(&parts);
    }
    if c.is_pow() && crate::core::expr::is_integer_class(&c.args()[1]) {
        return pow(&constants(&c.args()[0], names)?, &c.args()[1]);
    }
    if let Some((_, s)) = names.iter().find(|(k, _)| k == c) {
        return Ok(s.clone());
    }
    let s: Basic = sym(&format!("_lc{}", names.len()));
    names.push((c.clone(), s.clone()));
    Ok(s)
}

struct Rows {
    rows: Vec<BTreeMap<usize, Basic>>,
    symbols: Vec<Basic>,
    constants: HashMap<Basic, Basic>,
}

fn rows_of(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Rows> {
    let mut names: Vec<(Basic, Basic)> = Vec::new();
    let mut rows: Vec<BTreeMap<usize, Basic>> = Vec::new();
    for e in equations {
        let mut collected: BTreeMap<usize, Vec<Basic>> = BTreeMap::new();
        for term in terms_of(&expand(e)?) {
            let mut position: Option<usize> = None;
            let mut rest: Vec<Basic> = Vec::new();
            for factor in factors_of(&term) {
                let column: Option<usize> = if factor.is_symbol() {
                    unknowns.iter().position(|u| *u == factor)
                } else {
                    None
                };
                if column.is_some() && position.is_none() {
                    position = column;
                } else if factor.free_symbols().iter().any(|s| unknowns.contains(s)) {
                    return Err(value_error(format!("the equation {} is not linear in the unknowns", e)));
                } else {
                    rest.push(constants(&factor, &mut names)?);
                }
            }
            let Some(p) = position else {
                return Err(value_error(format!(
                    "the equation {} is not homogeneous in the unknowns",
                    e
                )));
            };
            collected.entry(p).or_default().push(mul(&rest)?);
        }
        let mut row: BTreeMap<usize, Basic> = BTreeMap::new();
        for (j, cs) in collected {
            let c: Basic = add(&cs)?;
            if c != Basic::zero() {
                row.insert(j, c);
            }
        }
        rows.push(row);
    }
    let mut symbols: BTreeSet<Basic> = BTreeSet::new();
    for row in &rows {
        for c in row.values() {
            symbols.extend(c.free_symbols());
        }
    }
    let mut symbols: Vec<Basic> = symbols.into_iter().collect();
    symbols.sort_by_cached_key(crate::core::sorting::sort_key);
    Ok(Rows {
        rows,
        symbols,
        constants: names.into_iter().map(|(c, s)| (s, c)).collect(),
    })
}

fn reduced_basis_over<K: Domain>(
    domain: &K,
    from_expr: &dyn Fn(&Basic) -> PolyResult<K::Elem>,
    to_expr: &dyn Fn(&K::Elem) -> R,
    rows: &[BTreeMap<usize, Basic>],
    ncols: usize,
    order: &[usize],
) -> PolyResult<Vec<BTreeMap<usize, Basic>>> {
    let mut sparse: Vec<BTreeMap<usize, K::Elem>> = Vec::new();
    for row in rows {
        let mut converted: BTreeMap<usize, K::Elem> = BTreeMap::new();
        for (j, c) in row {
            let v: K::Elem = from_expr(c)?;
            if !domain.is_zero(&v) {
                converted.insert(*j, v);
            }
        }
        if !converted.is_empty() {
            sparse.push(converted);
        }
    }
    let mut pivot_rows: BTreeMap<usize, BTreeMap<usize, K::Elem>> = BTreeMap::new();
    for r in domain.try_sparse_echelon(&sparse)? {
        let pivot: usize = *r.keys().next().expect("a nonzero row");
        pivot_rows.insert(pivot, r);
    }
    let position: HashMap<usize, usize> = order.iter().enumerate().map(|(k, j)| (*j, k)).collect();
    let mut basis: Vec<BTreeMap<usize, K::Elem>> = Vec::new();
    for free in 0..ncols {
        if pivot_rows.contains_key(&free) {
            continue;
        }
        let mut vector: BTreeMap<usize, K::Elem> = BTreeMap::new();
        vector.insert(position[&free], domain.one());
        for (pivot, r) in &pivot_rows {
            if let Some(v) = r.get(&free) {
                vector.insert(position[pivot], domain.neg(v));
            }
        }
        basis.push(vector);
    }
    domain
        .try_sparse_echelon(&basis)?
        .iter()
        .map(|r| r.iter().map(|(j, c)| Ok((*j, to_expr(c)?))).collect())
        .collect()
}

fn reduced_basis(rows: &Rows, ncols: usize, order: &[usize]) -> PolyResult<Vec<BTreeMap<usize, Basic>>> {
    use crate::polys::domains::{SparseFractionField, SparsePolynomialRing, QQ};
    if rows.symbols.is_empty() {
        return reduced_basis_over(&QQ, &|e| QQ.from_expr(e), &|c| QQ.to_expr(c), &rows.rows, ncols, order);
    }
    let names: Vec<String> = (0..rows.symbols.len()).map(|i| format!("_ls{}", i)).collect();
    let renamed: HashMap<Basic, Basic> = rows
        .symbols
        .iter()
        .zip(&names)
        .map(|(s, n)| (s.clone(), sym(n)))
        .collect();
    let restored: HashMap<Basic, Basic> = rows
        .symbols
        .iter()
        .zip(&names)
        .map(|(s, n)| (sym(n), s.clone()))
        .collect();
    let field = SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names));
    let reduced = reduced_basis_over(
        &field,
        &|e| field.from_expr(&replaced(e, &renamed)?),
        &|c| replaced(&field.to_expr(c)?, &restored),
        &rows.rows,
        ncols,
        order,
    )?;
    if rows.constants.is_empty() {
        return Ok(reduced);
    }
    reduced
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(j, c)| Ok((j, replaced(&c, &rows.constants)?)))
                .collect()
        })
        .collect()
}

fn is_linear_homogeneous(jets: &JetSpace, eqs: &[Basic]) -> PolyResult<bool> {
    for eq in eqs {
        let variables: Vec<Basic> = jets.jets_in(eq).into_iter().map(|(a, j)| jets.jet(a, &j)).collect();
        if variables.is_empty() {
            return Ok(false);
        }
        match polynomial_terms(eq, &variables)? {
            Some(terms) if !terms.is_empty() && terms.iter().all(|(m, _)| m.iter().sum::<u32>() == 1) => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// The point symmetries with polynomial infinitesimals of total degree at
/// most `degree` (times the functions of `basis`), as the reduced row
/// echelon form of their coefficients (`symmetries`).
pub fn symmetries(
    equations: &[Basic],
    functions: &[Basic],
    degree: u32,
    basis: &[Basic],
    superposition: bool,
) -> PolyResult<Vec<Symmetry>> {
    let (jets, eqs) = prepare(equations, functions)?;
    let a: Ansatz = ansatz(&jets, degree, basis)?;
    if a.unknowns.is_empty() {
        return Ok(vec![]);
    }
    let linear: Vec<Basic> = linear_equations(&jets, &eqs, &a)?;
    let superposition: bool = superposition || !is_linear_homogeneous(&jets, &eqs)?;
    let n: usize = a.unknowns.len();
    let n_xi: usize = n * jets.p() / (jets.p() + jets.q());
    let superposition_columns: Vec<usize> = (n_xi..n)
        .filter(|j| !a.multiplied[*j].free_symbols().iter().any(|s| jets.u.contains(s)))
        .collect();
    let mut order: Vec<usize> = (0..n).filter(|j| !superposition_columns.contains(j)).collect();
    order.extend(superposition_columns.iter().cloned());
    let reduced = reduced_basis(&rows_of(&linear, &a.unknowns)?, n, &order)?;
    let n_regular: usize = n - superposition_columns.len();
    let mut result: Vec<Symmetry> = Vec::new();
    for row in reduced {
        if !superposition && row.keys().next().is_some_and(|m| *m >= n_regular) {
            continue;
        }
        let mut values: HashMap<Basic, Basic> = a.unknowns.iter().map(|u| (u.clone(), Basic::zero())).collect();
        for (j, c) in &row {
            values.insert(a.unknowns[order[*j]].clone(), c.clone());
        }
        let xi: Vec<Basic> =
            a.xi.iter()
                .map(|e| expand(&replaced(e, &values)?))
                .collect::<PolyResult<_>>()?;
        let eta: Vec<Basic> = a
            .eta
            .iter()
            .map(|e| expand(&replaced(e, &values)?))
            .collect::<PolyResult<_>>()?;
        let s: Symmetry = Symmetry::new(&jets, xi, eta)?;
        if !s.is_zero() {
            result.push(s);
        }
    }
    Ok(result)
}
