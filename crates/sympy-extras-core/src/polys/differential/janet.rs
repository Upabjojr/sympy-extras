//! Port of `sympy_extras/polys/differential/janet.py`: Janet bases of
//! linear systems of partial differential equations (Janet's algorithm:
//! autoreduction, completion of the leaders for Janet's division, the
//! integrability conditions as Janet normal forms), with the parametric
//! derivatives, the Hilbert function, series and polynomial, the
//! dimension of the solution space and the power series solutions.
//!
//! `series_solution` expands its result with the default `expand` of the
//! core, the guarded [`expand_default`] (the hints `power_base`,
//! `power_exp` and `log` are not ported).

// a node caches its keys in `OnceLock`s that the hash never reads
#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::core::add::{add, sub};
use crate::core::basic::Basic;
use crate::core::function::expand_default;
use crate::core::mul::{div, mul};
use crate::core::power::pow;
use crate::core::Symbol;
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;
use crate::orderedmap::OrderedMap;
use crate::polys::domains::domain::Domain;
use crate::polys::rings::Monomial;

use super::ring::{
    integer_product, integer_quotient, unit_jet, Coefficient, DifferentialRing, IntegerCoefficient, Jet,
};

/// A linear differential polynomial: the coefficients of the derivatives
/// (the derivative of index `-1` stands for the inhomogeneous part).
pub type LinearForm = OrderedMap<Jet, IntegerCoefficient>;
/// A cone of multi-indices: the vertex and the flags of the
/// multiplicative variables.
pub type Cone = (Monomial, Vec<bool>);

/// The multiplicative variables of Janet's division for every element of
/// a finite set of multi-indices (`janet_multiplicative_variables`).
pub fn janet_multiplicative_variables(monomials: &[Monomial]) -> Vec<Vec<bool>> {
    let mut result: Vec<Vec<bool>> = Vec::new();
    for u in monomials {
        let mut flags: Vec<bool> = Vec::new();
        for i in 0..u.len() {
            let top: u32 = monomials
                .iter()
                .filter(|v| v[..i] == u[..i])
                .map(|v| v[i])
                .max()
                .expect("u itself");
            flags.push(u[i] == top);
        }
        result.push(flags);
    }
    result
}

/// Janet's decomposition of the complement of the multiples of a complete
/// set of multi-indices into disjoint cones (`_complement`).
fn complement(monomials: &[Monomial], n: usize) -> Vec<Cone> {
    if monomials.is_empty() {
        return vec![(vec![0; n], vec![true; n])];
    }
    let mut cones: Vec<Cone> = Vec::new();
    fn visit(group: Vec<&Monomial>, i: usize, prefix: Monomial, flags: Vec<bool>, n: usize, cones: &mut Vec<Cone>) {
        if i == n {
            return;
        }
        let mut values: Vec<u32> = group.iter().map(|u| u[i]).collect();
        values.sort();
        values.dedup();
        let top: u32 = *values.last().expect("a value");
        for d in 0..top {
            if !values.contains(&d) {
                let mut vertex: Monomial = prefix.clone();
                vertex.push(d);
                vertex.extend(vec![0; n - i - 1]);
                let mut f: Vec<bool> = flags.clone();
                f.push(false);
                f.extend(vec![true; n - i - 1]);
                cones.push((vertex, f));
            }
        }
        for d in values {
            let sub_group: Vec<&Monomial> = group.iter().filter(|u| u[i] == d).copied().collect();
            let mut p: Monomial = prefix.clone();
            p.push(d);
            let mut f: Vec<bool> = flags.clone();
            f.push(d == top);
            visit(sub_group, i + 1, p, f, n, cones);
        }
    }
    visit(monomials.iter().collect(), 0, Vec::new(), Vec::new(), n, &mut cones);
    cones
}

fn add_term(form: &mut LinearForm, jet: Jet, c: IntegerCoefficient) {
    let value: IntegerCoefficient = match form.get(&jet) {
        Some(existing) => existing.add(&c),
        None => c,
    };
    if value.is_zero() {
        form.remove(&jet);
    } else {
        form.insert(jet, value);
    }
}

/// The arithmetic of linear forms over a ring (`_Forms`).
struct Forms {
    ring: Rc<DifferentialRing>,
    unit: Jet,
}

impl Forms {
    fn new(ring: Rc<DifferentialRing>) -> Forms {
        let unit: Jet = unit_jet(ring.n());
        Forms { ring, unit }
    }

    /// The form of an equation with coefficients in the ring of the
    /// integers, and the element of the field it was multiplied by.
    fn parse(&self, equation: &Basic) -> PolyResult<(LinearForm, Coefficient)> {
        let terms = self.ring.terms(equation)?;
        for monomial in terms.keys() {
            if !monomial.is_empty() && !(monomial.len() == 1 && monomial[0].1 == 1) {
                return Err(PolyError::Value(format!(
                    "{} is not linear in the functions and their derivatives: use rosenfeld_groebner",
                    equation
                )));
            }
        }
        let mut form: LinearForm = LinearForm::new();
        let domain = &self.ring.domain;
        let mut scale: Coefficient = domain.one();
        for (monomial, c) in self.ring.integral(&terms).iter() {
            let jet: Jet = if monomial.is_empty() {
                self.unit.clone()
            } else {
                monomial[0].0.clone()
            };
            form.insert(jet, c.clone());
            scale = domain.exquo(&self.ring.field_element(c), terms.get(monomial).expect("a term"))?;
        }
        Ok((form, scale))
    }

    /// The expression of the form divided by `divisor` (by the coefficient
    /// of its leader by default).
    fn to_expr(&self, form: &LinearForm, divisor: Option<&Coefficient>) -> PolyResult<Basic> {
        if form.is_empty() {
            return Ok(Basic::zero());
        }
        let ring = &self.ring;
        let below: Coefficient = match divisor {
            Some(d) => d.clone(),
            None => ring.field_element(form.get(&self.leader(form)).expect("the leader")),
        };
        let mut ordered: Vec<&Jet> = form.keys().collect();
        ordered.sort_by_cached_key(|j| std::cmp::Reverse(ring.rank_key(j)));
        let mut parts: Vec<Basic> = Vec::new();
        for jet in ordered {
            let c: Coefficient = ring
                .domain
                .exquo(&ring.field_element(form.get(jet).expect("a term")), &below)?;
            parts.push(mul(&[ring.coefficient_expr(&c)?, ring.to_expr(jet)])?);
        }
        add(&parts)
    }

    fn leader(&self, form: &LinearForm) -> Jet {
        let jets: Vec<Jet> = form.keys().cloned().collect();
        self.ring.highest(&jets)
    }

    /// The form divided by the gcd of its coefficients, with a positive
    /// coefficient of the leader.
    fn primitive(&self, form: &LinearForm) -> LinearForm {
        let z = &self.ring.integers;
        let one: IntegerCoefficient = z.one();
        let values: Vec<&IntegerCoefficient> = form.values().collect();
        let mut divisor: IntegerCoefficient = values[0].clone();
        for c in &values[1..] {
            if divisor == one {
                break;
            }
            divisor = z.gcd(&divisor, c);
        }
        if is_negative(&form.get(&self.leader(form)).expect("the leader").LC()) != is_negative(&divisor.LC()) {
            divisor = divisor.neg();
        }
        if divisor == one {
            return form.clone();
        }
        form.iter()
            .map(|(jet, c)| {
                (
                    jet.clone(),
                    integer_quotient(c, &divisor).expect("the gcd divides the coefficients"),
                )
            })
            .collect()
    }

    fn derivative(&self, form: &LinearForm, i: usize) -> LinearForm {
        let mut result: LinearForm = LinearForm::new();
        for (jet, c) in form.iter() {
            let d: IntegerCoefficient = self.ring.integer_derivative(c, i);
            if !d.is_zero() {
                add_term(&mut result, jet.clone(), d);
            }
            if jet.0 >= 0 {
                add_term(&mut result, self.ring.differentiate(jet, i), c.clone());
            }
        }
        result
    }
}

/// The divisors among a list of forms (`_Division`): every derivative of
/// a leader, or the derivatives in the involutive cones only.
struct Division<'a> {
    forms: &'a Forms,
    elements: Vec<LinearForm>,
    leaders: Vec<Jet>,
    multiplicative: Option<Vec<Vec<bool>>>,
    prolonged_cache: RefCell<HashMap<(usize, Monomial), LinearForm>>,
}

impl<'a> Division<'a> {
    fn new(forms: &'a Forms, elements: Vec<LinearForm>, multiplicative: Option<Vec<Vec<bool>>>) -> Division<'a> {
        let leaders: Vec<Jet> = elements.iter().map(|p| forms.leader(p)).collect();
        Division {
            forms,
            elements,
            leaders,
            multiplicative,
            prolonged_cache: RefCell::new(HashMap::new()),
        }
    }

    /// The index of an element whose leader divides the derivative, with
    /// the quotient.
    fn divisor(&self, jet: &Jet, skip: Option<usize>) -> Option<(usize, Monomial)> {
        let (a, j) = jet;
        for (k, (b, l)) in self.leaders.iter().enumerate() {
            if b != a || Some(k) == skip || j.iter().zip(l).any(|(x, y)| x < y) {
                continue;
            }
            if let Some(m) = &self.multiplicative {
                let flags: &Vec<bool> = &m[k];
                if j.iter().zip(l).zip(flags).any(|((x, y), flag)| x > y && !flag) {
                    continue;
                }
            }
            return Some((k, j.iter().zip(l).map(|(x, y)| x - y).collect()));
        }
        None
    }

    fn prolonged(&self, k: usize, theta: &Monomial) -> LinearForm {
        let key: (usize, Monomial) = (k, theta.clone());
        if let Some(form) = self.prolonged_cache.borrow().get(&key) {
            return form.clone();
        }
        let form: LinearForm = match theta.iter().position(|t| *t > 0) {
            None => self.elements[k].clone(),
            Some(i) => {
                let mut lower: Monomial = theta.clone();
                lower[i] -= 1;
                self.forms.derivative(&self.prolonged(k, &lower), i)
            }
        };
        self.prolonged_cache.borrow_mut().insert(key, form.clone());
        form
    }

    /// A multiple of the form with every divisible term reduced, and the
    /// multiplier (`normal_form`, free of fractions).
    fn normal_form(&self, form: &LinearForm, skip: Option<usize>) -> PolyResult<(LinearForm, IntegerCoefficient)> {
        let ring = &self.forms.ring;
        let z = &ring.integers;
        let one: IntegerCoefficient = z.one();
        let mut multiplier: IntegerCoefficient = one.clone();
        let mut result: LinearForm = LinearForm::new();
        let mut rest: LinearForm = form.clone();
        while !rest.is_empty() {
            let jets: Vec<Jet> = rest.keys().cloned().collect();
            let jet: Jet = ring.highest(&jets);
            let c: IntegerCoefficient = rest.remove(&jet).expect("a term");
            let Some((k, theta)) = self.divisor(&jet, skip) else {
                result.insert(jet, c);
                continue;
            };
            let divisor: LinearForm = self.prolonged(k, &theta);
            let lead: &IntegerCoefficient = divisor.get(&jet).expect("the leader of the divisor");
            let common: IntegerCoefficient = z.gcd(lead, &c);
            let (a, b) = (integer_quotient(lead, &common)?, integer_quotient(&c, &common)?);
            if a != one {
                rest = rest.iter().map(|(o, e)| (o.clone(), integer_product(e, &a))).collect();
                result = result
                    .iter()
                    .map(|(o, e)| (o.clone(), integer_product(e, &a)))
                    .collect();
                multiplier = integer_product(&multiplier, &a);
            }
            for (other, e) in divisor.iter() {
                if *other != jet {
                    add_term(&mut rest, other.clone(), integer_product(&b, e).neg());
                }
            }
        }
        Ok((result, multiplier))
    }
}

/// Primitive forms none of whose terms is a derivative of the leader of
/// another one (`_autoreduce`).
fn autoreduce(forms: &Forms, elements: &[LinearForm]) -> PolyResult<Vec<LinearForm>> {
    let mut current: Vec<LinearForm> = elements
        .iter()
        .filter(|p| !p.is_empty())
        .map(|p| forms.primitive(p))
        .collect();
    loop {
        current.sort_by_cached_key(|p| forms.ring.rank_key(&forms.leader(p)));
        let division: Division = Division::new(forms, current.clone(), None);
        let mut changed: bool = false;
        for k in 0..current.len() {
            if current[k].keys().all(|jet| division.divisor(jet, Some(k)).is_none()) {
                continue;
            }
            let (r, _) = division.normal_form(&current[k], Some(k))?;
            if r.is_empty() {
                current.remove(k);
            } else {
                current[k] = forms.primitive(&r);
            }
            changed = true;
            break;
        }
        if !changed {
            return Ok(current);
        }
    }
}

fn multiplicative(forms: &Forms, elements: &[LinearForm]) -> Vec<Vec<bool>> {
    let leaders: Vec<Jet> = elements.iter().map(|p| forms.leader(p)).collect();
    let mut result: Vec<Vec<bool>> = vec![Vec::new(); leaders.len()];
    let functions: HashSet<i64> = leaders.iter().map(|l| l.0).collect();
    for a in functions {
        let indices: Vec<usize> = (0..leaders.len()).filter(|&k| leaders[k].0 == a).collect();
        let monomials: Vec<Monomial> = indices.iter().map(|&k| leaders[k].1.clone()).collect();
        for (k, f) in indices.into_iter().zip(janet_multiplicative_variables(&monomials)) {
            result[k] = f;
        }
    }
    result
}

/// The elements with the derivatives which make the set of the leaders
/// complete for Janet's division, and the multiplicative variables
/// (`_complete`).
fn complete(forms: &Forms, elements: &[LinearForm]) -> (Vec<LinearForm>, Vec<Vec<bool>>) {
    let ring = &forms.ring;
    let mut current: Vec<LinearForm> = elements.to_vec();
    loop {
        let flags: Vec<Vec<bool>> = multiplicative(forms, &current);
        let division: Division = Division::new(forms, current.clone(), Some(flags.clone()));
        let mut missing: Vec<(Vec<i64>, usize, usize)> = Vec::new();
        for (k, leader) in division.leaders.iter().enumerate() {
            for (i, flag) in flags[k].iter().enumerate() {
                if !flag {
                    let jet: Jet = ring.differentiate(leader, i);
                    if division.divisor(&jet, None).is_none() {
                        missing.push((ring.rank_key(&jet), k, i));
                    }
                }
            }
        }
        let Some((_, k, i)) = missing.into_iter().min() else {
            return (current, flags);
        };
        let derived: LinearForm = forms.derivative(&current[k], i);
        current.push(derived);
    }
}

/// The multi-indices of total order at most `order` (`_multi_indices`).
fn multi_indices(n: usize, order: u32) -> Vec<Monomial> {
    let mut result: Vec<Monomial> = vec![Vec::new()];
    for _ in 0..n {
        let mut next: Vec<Monomial> = Vec::new();
        for j in &result {
            let used: u32 = j.iter().sum();
            for k in 0..=(order - used) {
                let mut m: Monomial = j.clone();
                m.push(k);
                next.push(m);
            }
        }
        result = next;
    }
    result
}

/// The Janet basis of a linear system of differential equations
/// (`JanetBasis`; built by [`janet_basis`]).
pub struct JanetBasis {
    pub ring: Rc<DifferentialRing>,
    /// The equations with coprime polynomial coefficients, by increasing
    /// leader.
    pub forms: Vec<LinearForm>,
    /// The flags of the multiplicative variables of every equation.
    pub multiplicative: Vec<Vec<bool>>,
    arithmetic: Forms,
}

impl JanetBasis {
    fn new(ring: Rc<DifferentialRing>, forms: Vec<LinearForm>, multiplicative: Vec<Vec<bool>>) -> JanetBasis {
        let arithmetic: Forms = Forms::new(ring.clone());
        JanetBasis {
            ring,
            forms,
            multiplicative,
            arithmetic,
        }
    }

    fn division(&self) -> Division<'_> {
        Division::new(&self.arithmetic, self.forms.clone(), Some(self.multiplicative.clone()))
    }

    fn leaders_jets(&self) -> Vec<Jet> {
        self.forms.iter().map(|p| self.arithmetic.leader(p)).collect()
    }

    /// Whether the system has solutions: the basis is not `{1}`.
    pub fn is_consistent(&self) -> bool {
        !self.leaders_jets().iter().any(|l| l.0 < 0)
    }

    /// The equations other than the constancy of a function in a variable
    /// it does not depend on (`_shown`).
    fn shown(&self) -> Vec<usize> {
        let hidden: HashSet<Jet> = self.ring.constancy().into_iter().collect();
        let n: usize = self.ring.n();
        let mut result: Vec<usize> = Vec::new();
        for (k, (a, j)) in self.leaders_jets().into_iter().enumerate() {
            if a >= 0
                && j.iter().enumerate().any(|(i, e)| {
                    let mut unit: Monomial = vec![0; n];
                    unit[i] = 1;
                    *e > 0 && hidden.contains(&(a, unit))
                })
            {
                continue;
            }
            result.push(k);
        }
        result
    }

    /// The expressions which vanish, with the coefficient one at their
    /// leaders, by increasing leader.
    pub fn equations(&self) -> PolyResult<Vec<Basic>> {
        self.shown()
            .into_iter()
            .map(|k| self.arithmetic.to_expr(&self.forms[k], None))
            .collect()
    }

    /// The leaders of the equations.
    pub fn leaders(&self) -> Vec<Basic> {
        let jets: Vec<Jet> = self.leaders_jets();
        self.shown().into_iter().map(|k| self.ring.to_expr(&jets[k])).collect()
    }

    /// The multiplicative variables of the equations.
    pub fn multiplicative_variables(&self) -> Vec<Vec<Basic>> {
        self.shown()
            .into_iter()
            .map(|k| {
                self.ring
                    .variables
                    .iter()
                    .zip(&self.multiplicative[k])
                    .filter(|(_, flag)| **flag)
                    .map(|(s, _)| s.clone())
                    .collect()
            })
            .collect()
    }

    /// The Janet normal form of a linear differential expression
    /// (`reduce`).
    pub fn reduce(&self, expr: &Basic) -> PolyResult<Basic> {
        let (form, scale) = self.arithmetic.parse(expr)?;
        let (reduced, multiplier) = self.division().normal_form(&form, None)?;
        let divisor: Coefficient = self.ring.domain.mul(&scale, &self.ring.field_element(&multiplier));
        self.arithmetic.to_expr(&reduced, Some(&divisor))
    }

    /// Whether the expression vanishes on the solutions of the system.
    pub fn contains(&self, expr: &Basic) -> PolyResult<bool> {
        let (form, _) = self.arithmetic.parse(expr)?;
        Ok(self.division().normal_form(&form, None)?.0.is_empty())
    }

    fn require_consistent(&self) -> PolyResult<()> {
        if !self.is_consistent() {
            return Err(PolyError::Value(
                "the system is inconsistent: it has no solutions".to_string(),
            ));
        }
        Ok(())
    }

    /// The cones of the parametric derivatives, per function.
    fn cones(&self) -> PolyResult<Vec<(usize, Cone)>> {
        self.require_consistent()?;
        let jets: Vec<Jet> = self.leaders_jets();
        let mut result: Vec<(usize, Cone)> = Vec::new();
        for a in 0..self.ring.functions.len() {
            let leaders: Vec<Monomial> = jets.iter().filter(|l| l.0 == a as i64).map(|l| l.1.clone()).collect();
            for cone in complement(&leaders, self.ring.n()) {
                result.push((a, cone));
            }
        }
        Ok(result)
    }

    /// Whether the derivative is not a derivative of a leader.
    pub fn is_parametric(&self, derivative: &Basic) -> PolyResult<bool> {
        let Some(jet) = self.ring.jet(derivative) else {
            return Err(PolyError::Value(format!(
                "{} is not a derivative of a function of the ring",
                derivative
            )));
        };
        Ok(self.division().divisor(&jet, None).is_none())
    }

    /// The parametric derivatives up to the given order, by increasing rank.
    pub fn parametric_derivatives(&self, order: u32) -> PolyResult<Vec<Basic>> {
        Ok(self.parametric(order)?.iter().map(|j| self.ring.to_expr(j)).collect())
    }

    fn parametric(&self, order: u32) -> PolyResult<Vec<Jet>> {
        self.require_consistent()?;
        let division: Division = self.division();
        let mut jets: Vec<Jet> = Vec::new();
        for a in 0..self.ring.functions.len() {
            for j in multi_indices(self.ring.n(), order) {
                let jet: Jet = (a as i64, j);
                if division.divisor(&jet, None).is_none() {
                    jets.push(jet);
                }
            }
        }
        jets.sort_by_cached_key(|j| self.ring.rank_key(j));
        Ok(jets)
    }

    /// The number of the parametric derivatives of exactly the given order.
    pub fn hilbert_function(&self, order: u32) -> PolyResult<Int> {
        let mut total: Int = Int::ZERO;
        for (_, (vertex, flags)) in self.cones()? {
            let k: u64 = flags.iter().filter(|f| **f).count() as u64;
            let degree: u32 = vertex.iter().sum();
            if degree > order {
                continue;
            }
            let d: u64 = u64::from(order - degree);
            if k == 0 {
                if d == 0 {
                    total += Int::ONE;
                }
            } else {
                total += binomial(d + k - 1, k - 1);
            }
        }
        Ok(total)
    }

    /// The generating function of [`Self::hilbert_function`].
    pub fn hilbert_series(&self, t: &Basic) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (_, (vertex, flags)) in self.cones()? {
            let numerator: Basic = pow(t, &Basic::int(i64::from(vertex.iter().sum::<u32>())))?;
            let base: Basic = sub(&Basic::one(), t)?;
            let denominator: Basic = pow(&base, &Basic::int(flags.iter().filter(|f| **f).count() as i64))?;
            parts.push(div(&numerator, &denominator)?);
        }
        add(&parts)
    }

    /// The polynomial which the Hilbert function equals for the large
    /// orders `s` (`hilbert_polynomial`).
    pub fn hilbert_polynomial(&self, s: &Basic) -> PolyResult<Basic> {
        let mut total: Vec<Rat> = Vec::new();
        for (_, (vertex, flags)) in self.cones()? {
            let k: i64 = flags.iter().filter(|f| **f).count() as i64;
            if k == 0 {
                continue;
            }
            let shift: i64 = k - 1 - i64::from(vertex.iter().sum::<u32>());
            let mut coefficients: Vec<Rat> = vec![Rat::ONE];
            for i in 0..(k - 1) {
                let c: Rat = rat(shift - i, i + 1);
                let previous: Vec<Rat> = coefficients.clone();
                coefficients = (0..=previous.len())
                    .map(|j| {
                        let kept: Rat = if j < previous.len() {
                            &previous[j] * &c
                        } else {
                            Rat::ZERO
                        };
                        let raised: Rat = if j > 0 {
                            &previous[j - 1] / rat(i + 1, 1)
                        } else {
                            Rat::ZERO
                        };
                        kept + raised
                    })
                    .collect();
            }
            for (j, c) in coefficients.into_iter().enumerate() {
                while total.len() <= j {
                    total.push(Rat::ZERO);
                }
                total[j] += c;
            }
        }
        let mut parts: Vec<Basic> = Vec::new();
        for (j, c) in total.into_iter().enumerate() {
            if !c.is_zero() {
                parts.push(mul(&[Basic::from_rational(c), pow(s, &Basic::int(j as i64))?])?);
            }
        }
        add(&parts)
    }

    /// The dimension of the solution space, `oo` when it is infinite.
    pub fn dimension(&self) -> PolyResult<Basic> {
        let cones = self.cones()?;
        if cones.iter().any(|(_, (_, flags))| flags.iter().any(|f| *f)) {
            return Ok(Basic::oo());
        }
        Ok(Basic::int(cones.len() as i64))
    }

    /// The Taylor polynomials of degree `order` of the general solution at
    /// a point (the origin by default), the parametric derivatives at the
    /// point being the constants `C0, C1, ...` (`series_solution`).
    pub fn series_solution(
        &self,
        order: u32,
        point: Option<&[Basic]>,
        constant: &str,
    ) -> PolyResult<Vec<(Basic, Basic)>> {
        self.require_consistent()?;
        let ring = &self.ring;
        let n: usize = ring.n();
        let values: Vec<Basic> = match point {
            Some(p) => p.to_vec(),
            None => vec![Basic::zero(); n],
        };
        if values.len() != n {
            return Err(PolyError::Value(format!("the point must have {} coordinates", n)));
        }
        let at: HashMap<Basic, Basic> = ring.variables.iter().cloned().zip(values.iter().cloned()).collect();
        for form in &self.forms {
            let lead: Basic =
                ring.integer_expr(form.get(&self.arithmetic.leader(form)).expect("the leader"), Some(&at))?;
            if lead.as_rational().is_some_and(|v| v.is_zero()) {
                let shown: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                let point: String = if shown.len() == 1 {
                    format!("({},)", shown[0])
                } else {
                    format!("({})", shown.join(", "))
                };
                return Err(PolyError::Value(format!(
                    "the point {} is singular: the coefficient of a leader vanishes there",
                    point
                )));
            }
        }
        let division: Division = self.division();
        let mut expansions: Vec<(usize, Monomial, LinearForm, IntegerCoefficient)> = Vec::new();
        let mut needed: HashSet<Jet> = HashSet::new();
        for a in 0..ring.functions.len() {
            for j in multi_indices(n, order) {
                let mut seed: LinearForm = LinearForm::new();
                seed.insert((a as i64, j.clone()), ring.integers.one());
                let (form, multiplier) = division.normal_form(&seed, None)?;
                needed.extend(form.keys().filter(|jet| jet.0 >= 0).cloned());
                expansions.push((a, j, form, multiplier));
            }
        }
        let mut needed: Vec<Jet> = needed.into_iter().collect();
        needed.sort_by_cached_key(|j| ring.rank_key(j));
        let constants: HashMap<Jet, Basic> = needed
            .into_iter()
            .enumerate()
            .map(|(k, jet)| (jet, Symbol::new(&format!("{}{}", constant, k), &[]).0))
            .collect();
        let mut result: Vec<(Basic, Basic)> = ring.functions.iter().map(|f| (f.clone(), Basic::zero())).collect();
        for (a, j, form, multiplier) in expansions {
            let mut value: Basic = Basic::zero();
            let below: Basic = ring.integer_expr(&multiplier, Some(&at))?;
            for (jet, c) in form.iter() {
                let factor: Basic = div(&ring.integer_expr(c, Some(&at))?, &below)?;
                let unknown: Basic = if jet.0 >= 0 {
                    constants[jet].clone()
                } else {
                    Basic::one()
                };
                value = add(&[value, mul(&[factor, unknown])?])?;
            }
            let mut term: Basic = value;
            for ((s, x0), e) in ring.variables.iter().zip(&values).zip(&j) {
                let shifted: Basic = pow(&sub(s, x0)?, &Basic::int(i64::from(*e)))?;
                term = div(&mul(&[term, shifted])?, &Basic::integer(factorial(*e)))?;
            }
            result[a].1 = add(&[result[a].1.clone(), term])?;
        }
        result.into_iter().map(|(f, e)| Ok((f, expand_default(&e)?))).collect()
    }
}

fn binomial(n: u64, k: u64) -> Int {
    let mut c: Int = Int::ONE;
    for j in 0..k {
        c = c * Int::from(n - j) / Int::from(j + 1);
    }
    c
}

fn factorial(n: u32) -> Int {
    (1..=n).fold(Int::ONE, |acc, k| acc * Int::from(k))
}

/// The Janet basis of a linear system of differential equations with
/// coefficients rational in the variables (`janet_basis`); the symbols
/// which are not variables are constant parameters.
pub fn janet_basis(
    equations: &[Basic],
    functions: &[Basic],
    variables: Option<&[Basic]>,
    ranking: Option<&[Vec<Basic>]>,
) -> PolyResult<JanetBasis> {
    let probe: DifferentialRing = DifferentialRing::new(functions, variables, ranking, &[])?;
    let mut symbols: Vec<Basic> = Vec::new();
    for e in equations {
        for s in e.free_symbols() {
            if !probe.variables.contains(&s) && !symbols.contains(&s) {
                symbols.push(s);
            }
        }
    }
    // by the names (`s.name`)
    symbols.sort_by_key(|s| match s.head() {
        crate::core::basic::Head::Symbol(d) | crate::core::basic::Head::Dummy(d, _) => d.name.clone(),
        crate::core::basic::Head::Wild(w) => w.symbol.name.clone(),
        _ => s.to_string(),
    });
    let ring: Rc<DifferentialRing> = Rc::new(DifferentialRing::new(functions, variables, ranking, &symbols)?);
    let forms: Forms = Forms::new(ring.clone());
    let one: IntegerCoefficient = ring.integers.one();
    let mut elements: Vec<LinearForm> = Vec::new();
    for e in equations {
        elements.push(forms.parse(e)?.0);
    }
    for jet in ring.constancy() {
        elements.push(std::iter::once((jet, one.clone())).collect());
    }
    let mut current: Vec<LinearForm> = autoreduce(&forms, &elements)?;
    let (completed, flags) = loop {
        if current.iter().any(|p| forms.leader(p).0 < 0) {
            let unit: LinearForm = std::iter::once((forms.unit.clone(), one.clone())).collect();
            return Ok(JanetBasis::new(ring.clone(), vec![unit], vec![vec![true; ring.n()]]));
        }
        let (completed, flags) = complete(&forms, &current);
        let division: Division = Division::new(&forms, completed.clone(), Some(flags.clone()));
        let mut conditions: Vec<LinearForm> = Vec::new();
        for k in 0..completed.len() {
            for (i, flag) in flags[k].iter().enumerate() {
                if !flag {
                    let (r, _) = division.normal_form(&forms.derivative(&completed[k], i), None)?;
                    if !r.is_empty() {
                        conditions.push(forms.primitive(&r));
                    }
                }
            }
        }
        if conditions.is_empty() {
            break (completed, flags);
        }
        let mut all: Vec<LinearForm> = completed;
        all.extend(conditions);
        current = autoreduce(&forms, &all)?;
    };
    let division: Division = Division::new(&forms, completed.clone(), Some(flags.clone()));
    let mut reduced: Vec<LinearForm> = Vec::new();
    for (k, p) in completed.iter().enumerate() {
        let leader: &Jet = &division.leaders[k];
        let rest: LinearForm = p
            .iter()
            .filter(|(jet, _)| *jet != leader)
            .map(|(j, c)| (j.clone(), c.clone()))
            .collect();
        let (mut tail, multiplier) = division.normal_form(&rest, None)?;
        tail.insert(
            leader.clone(),
            integer_product(p.get(leader).expect("the leader"), &multiplier),
        );
        reduced.push(forms.primitive(&tail));
    }
    let mut order: Vec<usize> = (0..reduced.len()).collect();
    order.sort_by_cached_key(|&k| ring.rank_key(&division.leaders[k]));
    let forms_sorted: Vec<LinearForm> = order.iter().map(|&k| reduced[k].clone()).collect();
    let flags_sorted: Vec<Vec<bool>> = order.iter().map(|&k| flags[k].clone()).collect();
    Ok(JanetBasis::new(ring, forms_sorted, flags_sorted))
}
