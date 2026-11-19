//! Port of `sympy_extras/integrals/elliptic.py`: definite integrals of
//! `R(x) * P(x)**(+-1/2)` with `P` of degree three or four, reduced to
//! Legendre's normal form `R(s) / sqrt(s (1 - s) (1 - m s))` on `(0, 1)`
//! by the substitution of the cell between the real roots, the cosine
//! map of a complex pair, or the tangent map of two pairs, then
//! integrated in `K`, `E`, `F`, `Pi` and elementary terms.

#![allow(clippy::too_many_arguments, clippy::type_complexity, clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::evalf::{numeric_value, parts, real_rat, Complex, Real};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;
use crate::numbers::{denom, numer, Int, Rat};
use crate::polys::domains::algebraicfield::{AlgebraicElement, AlgebraicField};
use crate::polys::domains::domain::Domain;
use crate::polys::numberfields::qqbar::Algebraic;

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn half() -> Basic {
    Basic::from_rational(crate::numbers::rat(1, 2))
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
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn pwi(a: &Basic, k: i64) -> R {
    pw(a, &int(k))
}
fn sqrt(a: &Basic) -> R {
    pw(a, &half())
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
/// A product built from the left as Python's `a * b * c` (`Mul` of two
/// factors distributes a number over a sum).
fn chain(v: &[Basic]) -> R {
    let mut result: Basic = v[0].clone();
    for factor in &v[1..] {
        result = m2(&result, factor)?;
    }
    Ok(result)
}
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn apply(f: Function, args: &[Basic]) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), args.to_vec()), args.to_vec())
}
fn asin(a: &Basic) -> R {
    apply(Function::Asin, std::slice::from_ref(a))
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::derivative(e, x)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn replaced_plain(e: &Basic, rule: &[(Basic, Basic)]) -> Basic {
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    e.xreplace(&map)
}
fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn is_oo(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity)
}
fn is_neg_oo(e: &Basic) -> bool {
    matches!(e.head(), Head::NegativeInfinity)
}
fn pi() -> Basic {
    Basic::pi()
}
fn is_dummy(e: &Basic) -> bool {
    matches!(e.head(), Head::Dummy(_, _))
}
fn free(e: &Basic) -> Vec<Basic> {
    let mut v: Vec<Basic> = e.free_symbols().into_iter().filter(|s| s.is_symbol()).collect();
    v.sort_by_cached_key(crate::core::sorting::sort_key);
    v
}
fn is_polynomial(e: &Basic, s: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(s),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}
fn is_rational_function(e: &Basic, s: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(s), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}
fn rel(op: RelOp, a: &Basic, b: &Basic) -> R {
    use crate::core::relational as r;
    match op {
        RelOp::Gt => r::gt(a, b),
        RelOp::Ge => r::ge(a, b),
        RelOp::Lt => r::lt(a, b),
        RelOp::Le => r::le(a, b),
        RelOp::Eq => r::eq(a, b),
        RelOp::Ne => r::ne(a, b),
    }
}
fn relop(e: &Basic) -> Option<RelOp> {
    match e.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}
fn gives_up(e: &PolyError) -> bool {
    matches!(e, PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_))
        || matches!(e, PolyError::NotImplemented(m) if !m.starts_with("not ported"))
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A dummy of a new name (`_fresh`).
fn fresh(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    let k: usize = COUNTER.fetch_add(1, Ordering::Relaxed);
    crate::core::symbol::Dummy::new(&format!("{}{}", name, k), assumptions).0
}
fn positive_dummy(name: &str) -> Basic {
    fresh(name, &[(Predicate::positive, true)])
}
fn real_dummy(name: &str) -> Basic {
    fresh(name, &[(Predicate::real, true)])
}

fn budget() -> Option<f64> {
    crate::settings::settings().timeout.map(|t| t / 4.0)
}

/// `ask` with the numeric values of the dummies decided numerically and the
/// symbols with assumptions answered on plain symbols (`_ask`).
fn ask_(query: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    let (values, rest) = numeric_values(assumptions);
    let symbols: Vec<Basic> = free(query);
    if !values.is_empty() && !symbols.is_empty() && symbols.iter().all(|s| values.iter().any(|(k, _)| k == s)) {
        return decide_numerically(query, &values);
    }
    let plain: Vec<(Basic, Basic)> = crate::integrals::conditions::plain_symbols(&symbols);
    if plain.is_empty() {
        let found = crate::timeout::attempt_within(budget(), || crate::assumptions::ask::ask(query, &rest, None))?;
        return Ok(found.flatten());
    }
    let on_plain: Basic = replaced(query, &plain)?;
    let facts: Vec<Basic> = crate::integrals::conditions::with_symbol_facts(&rest, &symbols, &plain)?;
    let found = crate::timeout::attempt_within(budget(), || crate::assumptions::ask::ask(&on_plain, &facts, None))?;
    Ok(found.flatten())
}

fn ask_rel(op: RelOp, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    ask_(&rel(op, a, b)?, assumptions)
}

fn numeric_values(assumptions: &[Basic]) -> (Vec<(Basic, Basic)>, Vec<Basic>) {
    let mut values: Vec<(Basic, Basic)> = Vec::new();
    let mut rest: Vec<Basic> = Vec::new();
    for item in assumptions {
        if relop(item) == Some(RelOp::Eq) && is_dummy(&item.args()[0]) {
            values.push((item.args()[0].clone(), item.args()[1].clone()));
        } else {
            rest.push(item.clone());
        }
    }
    (values, rest)
}

fn numeric(e: &Basic, values: &[(Basic, Basic)], evaluated: bool) -> R {
    let mut result: Basic = e.clone();
    for _ in 0..8 {
        if !values.iter().any(|(k, _)| result.has(k)) {
            break;
        }
        result = if evaluated {
            replaced(&result, values)?
        } else {
            replaced_plain(&result, values)
        };
    }
    Ok(result)
}

fn decide_numerically(query: &Basic, values: &[(Basic, Basic)]) -> PolyResult<Option<bool>> {
    let Some(op) = relop(query) else { return Ok(None) };
    let difference: Basic = numeric(&s2(&query.args()[0], &query.args()[1])?, values, true)?;
    if op == RelOp::Eq {
        return Ok(if difference == int(0) { Some(true) } else { None });
    }
    if !matches!(op, RelOp::Gt | RelOp::Ge | RelOp::Lt | RelOp::Le) || !free(&difference).is_empty() {
        return Ok(None);
    }
    let Some(approximation) = crate::numeric::reliable_value(&difference, 60, &[]) else {
        return Ok(None);
    };
    let (re, im) = parts(&approximation);
    if im != Real::ZERO {
        return Ok(None);
    }
    if re.to_f64().value().abs() < 1e-40 {
        return Ok(None);
    }
    let positive: bool = re > Real::ZERO;
    Ok(Some(if matches!(op, RelOp::Gt | RelOp::Ge) {
        positive
    } else {
        !positive
    }))
}

fn with_facts(assumptions: &[Basic], facts: &[Basic]) -> Vec<Basic> {
    if facts.is_empty() {
        return assumptions.to_vec();
    }
    let mut combined: Vec<Basic> = facts.to_vec();
    combined.extend(assumptions.iter().cloned());
    combined
}

const MAX_DEPTH: usize = 4;

/// The constants of an expression as dummies (`_Placeholders`).
struct Placeholders {
    variable: Option<Basic>,
    forward: Vec<(Basic, Basic)>,
    back: Vec<(Basic, Basic)>,
}

impl Placeholders {
    fn new(variable: Option<&Basic>) -> Placeholders {
        Placeholders {
            variable: variable.cloned(),
            forward: Vec::new(),
            back: Vec::new(),
        }
    }

    fn hide(&mut self, e: &Basic) -> R {
        if e.is_rational_class() || e.is_symbol() {
            return Ok(e.clone());
        }
        if e.is_add() {
            let v: Vec<Basic> = e.args().iter().map(|a| self.hide(a)).collect::<PolyResult<_>>()?;
            return add_all(&v);
        }
        if e.is_mul() {
            let v: Vec<Basic> = e.args().iter().map(|a| self.hide(a)).collect::<PolyResult<_>>()?;
            return mul_all(&v);
        }
        if e.is_pow() && e.args()[1].is_integer_value() {
            let base: Basic = self.hide(&e.args()[0])?;
            return pw(&base, &e.args()[1]);
        }
        let depends: bool = match &self.variable {
            Some(v) => e.has(v),
            None => !free(e).is_empty(),
        };
        if depends {
            return Ok(e.clone());
        }
        if e.is_pow() {
            if let Some(r) = e.args()[1].as_rational() {
                let p: Int = numer(r);
                if p != Int::from(1) {
                    let root: Basic = pw(
                        &e.args()[0],
                        &Basic::from_rational(Rat::from(Int::from(1)) / Rat::from(denom(r))),
                    )?;
                    let placeholder: Basic = self.placeholder(&root);
                    return pw(&placeholder, &Basic::from_rational(Rat::from(p)));
                }
            }
        }
        Ok(self.placeholder(e))
    }

    fn placeholder(&mut self, e: &Basic) -> Basic {
        if let Some((_, d)) = self.forward.iter().find(|(k, _)| k == e) {
            return d.clone();
        }
        let d: Basic = fresh("c", &[]);
        self.forward.push((e.clone(), d.clone()));
        self.back.push((d.clone(), e.clone()));
        d
    }

    fn restore(&self, e: &Basic) -> R {
        replaced(e, &self.back)
    }
}

type Coefficients = BTreeMap<i64, Basic>;

/// The coefficients of a polynomial in `s` (`_coefficients`).
fn coefficients(e: &Basic, s: &Basic) -> PolyResult<Option<Coefficients>> {
    let mut collected: BTreeMap<i64, Vec<Basic>> = BTreeMap::new();
    for term in crate::core::expr::terms_of(&expand(e)?) {
        if term == int(0) {
            continue;
        }
        let (coefficient, power) = crate::core::expr::as_independent(&term, s, false)?;
        let degree: i64 = if power == int(1) {
            0
        } else if power == *s {
            1
        } else if power.is_pow()
            && power.args()[0] == *s
            && power.args()[1].is_integer_value()
            && power.args()[1].as_rational().is_some_and(|r| *r > Rat::ZERO)
        {
            i64::try_from(&numer(power.args()[1].as_rational().expect("an integer"))).unwrap_or(i64::MAX)
        } else {
            return Ok(None);
        };
        collected.entry(degree).or_default().push(coefficient);
    }
    let mut result: Coefficients = Coefficients::new();
    for (degree, parts_) in collected {
        let total: Basic = add_all(&parts_)?;
        if total != int(0) {
            result.insert(degree, total);
        }
    }
    Ok(Some(result))
}

fn get(c: &Coefficients, k: i64) -> Basic {
    c.get(&k).cloned().unwrap_or_else(|| int(0))
}

fn top(c: &Coefficients) -> i64 {
    c.keys().next_back().copied().unwrap_or(-1)
}

fn degree(e: &Basic, s: &Basic) -> PolyResult<Option<i64>> {
    Ok(coefficients(e, s)?.map(|c| top(&c)))
}

fn factor_list_(e: &Basic, s: &Basic) -> PolyResult<Option<(Basic, Vec<(Basic, i64)>)>> {
    let mut hidden: Placeholders = Placeholders::new(Some(s));
    let hid: Basic = hidden.hide(e)?;
    let (coefficient, factors) = match crate::polys::polytools::factor_list(&hid, std::slice::from_ref(s)) {
        Ok(found) => found,
        Err(err) if gives_up(&err) => return Ok(None),
        Err(err) => return Err(err),
    };
    let mut result: Vec<(Basic, i64)> = Vec::new();
    for (factor, multiplicity) in factors {
        if !multiplicity.is_integer_value() {
            return Ok(None);
        }
        let k: i64 = i64::try_from(&numer(multiplicity.as_rational().expect("an integer"))).unwrap_or(1);
        result.push((hidden.restore(&factor)?, k));
    }
    Ok(Some((hidden.restore(&coefficient)?, result)))
}

fn rational_coefficients(e: &Basic, s: &Basic) -> PolyResult<Option<Vec<Rat>>> {
    let Some(c) = coefficients(e, s)? else { return Ok(None) };
    if c.is_empty() {
        return Ok(None);
    }
    let mut dense: Vec<Rat> = Vec::new();
    for d in (0..=top(&c)).rev() {
        match get(&c, d).as_rational() {
            Some(r) => dense.push(r.clone()),
            None => return Ok(None),
        }
    }
    Ok(Some(dense))
}

fn low_degree_roots(e: &Basic, s: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let mut hidden: Placeholders = Placeholders::new(Some(s));
    let hid: Basic = hidden.hide(&expand(e)?)?;
    let found: Vec<Basic> = match crate::polys::polyroots::roots_list(&hid, std::slice::from_ref(s), true, true, None) {
        Ok(found) => found,
        Err(err) if gives_up(&err) => return Ok(None),
        Err(err) => return Err(err),
    };
    let mut result: Vec<Basic> = Vec::new();
    for r in found {
        let restored: Basic = hidden.restore(&r)?;
        if !result.contains(&restored) {
            result.push(restored);
        }
    }
    Ok(Some(result))
}

fn real_roots_rational(e: &Basic, s: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let Some(dense) = rational_coefficients(e, s)? else {
        return Ok(None);
    };
    let count: i64 = crate::polys::rootisolation::dup_count_real_roots(&dense, None, None);
    let mut found: Vec<Basic> = Vec::new();
    for i in 0..count.max(0) as usize {
        found.push(crate::polys::cad::indexedroot::indexed_root(e, s, i)?);
    }
    Ok(Some(found))
}

/// The integral `int_lower^upper Q(s) / sqrt(s (1 - s) (1 - m s)) ds` of
/// Legendre's normal form (`Reduction`).
#[derive(Debug, Clone)]
pub struct Reduction {
    pub q: Basic,
    pub m: Basic,
    pub s: Basic,
    pub lower: Basic,
    pub upper: Basic,
}

const MAX_FIELD_DEGREE: usize = 32;

fn algebraic(atom: &Basic) -> PolyResult<Option<Algebraic>> {
    let value: Option<Algebraic> = if matches!(atom.head(), Head::ImaginaryUnit) {
        Some(Algebraic::imaginary_unit())
    } else if atom.is_pow() {
        match (atom.args()[0].as_rational(), atom.args()[1].as_rational()) {
            (Some(b), Some(e)) if denom(e) > Int::from(1) => {
                let q: usize = usize::try_from(&denom(e)).unwrap_or(1);
                let p: i64 = i64::try_from(&numer(e)).unwrap_or(1);
                Some(Algebraic::from_rational(b).nth_root(q)?.pow(p)?)
            }
            _ => None,
        }
    } else {
        None
    };
    let Some(value) = value else { return Ok(None) };
    let Some(number) = numeric_value(atom, 100) else {
        return Ok(None);
    };
    let (re, im) = parts(&number);
    let (nr, ni) = (re.to_f64().value(), im.to_f64().value());
    let (vr, vi) = value.to_complex()?;
    let size: f64 = (nr * nr + ni * ni).sqrt();
    if ((vr - nr).powi(2) + (vi - ni).powi(2)).sqrt() > 1e-9 * (1.0 + size) {
        return Ok(None);
    }
    Ok(Some(value))
}

fn horner(coefficients_: &[Rat], point: &AlgebraicElement, k: &AlgebraicField) -> AlgebraicElement {
    let mut result: AlgebraicElement = k.zero();
    for c in coefficients_ {
        result = k.add(&k.mul(&result, point), &k.from_rational(c));
    }
    result
}

fn monic(minpoly: &[Int]) -> Vec<Rat> {
    let lead: Rat = Rat::from(minpoly[0].clone());
    minpoly.iter().map(|c| Rat::from(c.clone()) / lead.clone()).collect()
}

/// The field `QQ(theta)` of algebraic constants (`_NumberField`).
struct NumberField {
    theta: Basic,
    field: AlgebraicField,
    theta_expr: Basic,
    forms: Vec<(Basic, Basic)>,
    powers: RefCell<Vec<Vec<(Basic, Rat)>>>,
}

impl NumberField {
    fn new(atoms: &[Basic], values: &[Algebraic]) -> PolyResult<NumberField> {
        let theta: Basic = fresh("theta", &[]);
        let mut gamma: Algebraic = values[0].clone();
        let mut gamma_expr: Basic = atoms[0].clone();
        let mut coordinates: Vec<AlgebraicElement> = vec![vec![Rat::from(Int::from(1)), Rat::ZERO]];
        for (atom, value) in atoms[1..].iter().zip(values[1..].iter()) {
            let (new, p, q) = crate::polys::numberfields::qqbar::primitive_element(&gamma, value)?;
            if new.degree() > MAX_FIELD_DEGREE {
                return Err(PolyError::Arithmetic(
                    "the field of the constants is too large".to_string(),
                ));
            }
            let k: AlgebraicField = AlgebraicField::new(monic(&new.minpoly()), theta.clone())?;
            let gamma_in_new: AlgebraicElement = horner(&p, &k.alpha(), &k);
            let mut next: Vec<AlgebraicElement> = coordinates.iter().map(|c| horner(c, &gamma_in_new, &k)).collect();
            next.push(horner(&q, &k.alpha(), &k));
            coordinates = next;
            let limit: i64 = (4 * gamma.degree() * value.degree() + 2) as i64;
            let mut found: Option<i64> = None;
            for c in 1..limit {
                let shifted: Algebraic = gamma.add(&Algebraic::from_rational(&Rat::from(Int::from(c))).mul(value)?)?;
                if shifted.equals(&new)? {
                    found = Some(c);
                    break;
                }
            }
            let c: i64 = found.ok_or_else(|| crate::error::value_error("no primitive combination".to_string()))?;
            gamma = new;
            gamma_expr = a2(&gamma_expr, &m2(&int(c), atom)?)?;
        }
        let field: AlgebraicField = AlgebraicField::new(monic(&gamma.minpoly()), theta.clone())?;
        let forms: Vec<(Basic, Basic)> = atoms
            .iter()
            .zip(coordinates.iter())
            .map(|(a, e)| Ok((a.clone(), field.to_expr(e)?)))
            .collect::<PolyResult<_>>()?;
        Ok(NumberField {
            theta,
            field,
            theta_expr: gamma_expr,
            forms,
            powers: RefCell::new(Vec::new()),
        })
    }

    fn power(&self, k: usize) -> PolyResult<Vec<(Basic, Rat)>> {
        loop {
            let n: usize = self.powers.borrow().len();
            if n > k {
                break;
            }
            let mut terms: Vec<(Basic, Rat)> = Vec::new();
            for term in crate::core::expr::terms_of(&expand(&pwi(&self.theta_expr, n as i64)?)?) {
                let (c, rest) = crate::core::expr::as_coeff_mul(&term);
                let (c, rest) = match c.as_rational() {
                    Some(r) => (r.clone(), rest),
                    None => (Rat::from(Int::from(1)), term.clone()),
                };
                match terms.iter_mut().find(|(t, _)| *t == rest) {
                    Some(entry) => entry.1 = entry.1.clone() + c,
                    None => terms.push((rest, c)),
                }
            }
            self.powers.borrow_mut().push(terms);
        }
        Ok(self.powers.borrow()[k].clone())
    }

    fn expression(&self, element: &AlgebraicElement) -> R {
        let n: usize = element.len().saturating_sub(1);
        let mut terms: Vec<(Basic, Rat)> = Vec::new();
        for (i, c) in element.iter().enumerate() {
            if *c == Rat::ZERO {
                continue;
            }
            for (term, d) in self.power(n - i)? {
                let value: Rat = c.clone() * d;
                match terms.iter_mut().find(|(t, _)| *t == term) {
                    Some(entry) => entry.1 = entry.1.clone() + value,
                    None => terms.push((term, value)),
                }
            }
        }
        let parts_: Vec<Basic> = terms
            .into_iter()
            .filter(|(_, c)| *c != Rat::ZERO)
            .map(|(t, c)| m2(&Basic::from_rational(c), &t))
            .collect::<PolyResult<_>>()?;
        add_all(&parts_)
    }
}

/// The number field of the constants of some expressions (`_ConstantField`).
struct ConstantField {
    rule: Vec<(Basic, Basic)>,
    field: Option<NumberField>,
}

impl ConstantField {
    fn new(exprs: &[Basic], s: &Basic) -> PolyResult<ConstantField> {
        let mut hidden: Placeholders = Placeholders::new(None);
        for e in exprs {
            hidden.hide(&expand(e)?)?;
        }
        let mut free_: Vec<Basic> = Vec::new();
        for e in exprs {
            for f in free(e) {
                if !free_.contains(&f) {
                    free_.push(f);
                }
            }
        }
        let mut atoms: Vec<Basic> = Vec::new();
        let mut values: Vec<Algebraic> = Vec::new();
        for (constant, _) in &hidden.forward {
            match algebraic(constant)? {
                Some(value) => {
                    atoms.push(constant.clone());
                    values.push(value);
                }
                None => {
                    atoms.clear();
                    break;
                }
            }
        }
        let mut field: Option<NumberField> = None;
        if !atoms.is_empty() && free_.iter().all(|f| f == s) {
            field = match NumberField::new(&atoms, &values) {
                Ok(f) => Some(f),
                Err(PolyError::Arithmetic(_)) => None,
                Err(e) => return Err(e),
            };
        }
        let rule: Vec<(Basic, Basic)> = field.as_ref().map(|f| f.forms.clone()).unwrap_or_default();
        Ok(ConstantField { rule, field })
    }

    fn element(&self, c: &Basic) -> PolyResult<Option<AlgebraicElement>> {
        let Some(field) = &self.field else { return Ok(None) };
        let written: Basic = replaced(c, &self.rule)?;
        if free(&written).iter().any(|f| *f != field.theta) {
            return Ok(None);
        }
        match field.field.from_expr(&expand(&written)?) {
            Ok(e) => Ok(Some(e)),
            Err(PolyError::Value(_) | PolyError::ZeroDivision(_) | PolyError::Polys(..)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn canonical(&self, c: &Basic) -> R {
        match (self.element(c)?, &self.field) {
            (Some(element), Some(field)) => field.expression(&element),
            _ => Ok(c.clone()),
        }
    }

    fn factor_list(&self, e: &Basic, s: &Basic) -> PolyResult<Option<(Basic, Vec<(Basic, i64)>)>> {
        let Some(field) = &self.field else { return Ok(None) };
        let Some(c) = coefficients(e, s)? else { return Ok(None) };
        if c.is_empty() {
            return Ok(None);
        }
        let mut dense: Vec<AlgebraicElement> = Vec::new();
        for d in (0..=top(&c)).rev() {
            match self.element(&get(&c, d))? {
                Some(element) => dense.push(element),
                None => return Ok(None),
            }
        }
        let (lead, factors) = match crate::polys::factortools::dup_ext_factor(&dense, &field.field) {
            Ok(found) => found,
            Err(PolyError::Value(_) | PolyError::ZeroDivision(_) | PolyError::Polys(..)) => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut result: Vec<(Basic, i64)> = Vec::new();
        for (factor, multiplicity) in factors {
            let n: usize = factor.len() - 1;
            let mut terms: Vec<Basic> = Vec::new();
            for (i, coefficient) in factor.iter().enumerate() {
                terms.push(m2(&field.expression(coefficient)?, &pwi(s, (n - i) as i64)?)?);
            }
            result.push((add_all(&terms)?, multiplicity as i64));
        }
        Ok(Some((field.expression(&lead)?, result)))
    }
}

/// `(R, P, e)` with `f = R(x) * P(x)**e`, `e = +-1/2`, `P` of degree three
/// or four (`radicand`).
pub fn radicand(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    let mut halves: Vec<(Basic, Basic)> = Vec::new();
    let mut rest: Basic = int(1);
    let factors: Vec<Basic> = if f.is_mul() { f.args().to_vec() } else { vec![f.clone()] };
    for factor in factors {
        let half_power: bool = factor.is_pow()
            && factor.has(x)
            && factor.args()[1].as_rational().is_some_and(|r| denom(r) == Int::from(2));
        if half_power {
            let base: Basic = factor.args()[0].clone();
            if !is_polynomial(&base, x) {
                return Ok(None);
            }
            halves.push((base, factor.args()[1].clone()));
        } else {
            rest = m2(&rest, &factor)?;
        }
    }
    if halves.is_empty() {
        return Ok(None);
    }
    let mut exponent: Basic = neg(&half())?;
    let mut best: i64 = -1;
    for (base, q) in &halves {
        let Some(d) = degree(base, x)? else { return Ok(None) };
        let absolute: bool = q
            .as_rational()
            .is_some_and(|r| *r == crate::numbers::rat(1, 2) || *r == crate::numbers::rat(-1, 2));
        if absolute && d > best {
            best = d;
            exponent = q.clone();
        }
    }
    let mut p: Basic = int(1);
    for (base, q) in &halves {
        p = m2(&p, base)?;
        rest = m2(&rest, &pw(base, &s2(q, &exponent)?)?)?;
    }
    let Some(d) = degree(&p, x)? else { return Ok(None) };
    let rest: Basic = cancel(&rest)?;
    if !(d == 3 || d == 4) || !is_rational_function(&rest, x) {
        return Ok(None);
    }
    Ok(Some((rest, p, exponent)))
}

/// The real roots of `P` in increasing order when all of them are real and
/// simple, or `None` (`real_roots`).
pub fn real_roots(p: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let Some(d) = degree(p, x)? else { return Ok(None) };
    if d < 1 {
        return Ok(None);
    }
    let mut found: Vec<Basic> = Vec::new();
    if free(p).iter().all(|s| s == x) {
        if rational_coefficients(p, x)?.is_none() {
            return Ok(None);
        }
        let Some(reals) = real_roots_rational(p, x)? else {
            return Ok(None);
        };
        if reals.len() as i64 != d {
            return Ok(None);
        }
        found = reals;
    } else {
        let Some((_, factors)) = factor_list_(p, x)? else {
            return Ok(None);
        };
        for (factor, multiplicity) in factors {
            if multiplicity != 1 || !factor.has(x) {
                if factor.has(x) {
                    return Ok(None);
                }
                continue;
            }
            match degree(&factor, x)? {
                Some(fd) if fd <= 2 => {}
                _ => return Ok(None),
            }
            let Some(solutions) = low_degree_roots(&factor, x)? else {
                return Ok(None);
            };
            for r in solutions {
                if is(&r, Predicate::extended_real) == Some(false) {
                    return Ok(None);
                }
                let imaginary: Basic = crate::functions::elementary::complexes::im(&r)?;
                let symbolic: bool = imaginary
                    .preorder()
                    .iter()
                    .any(|n| matches!(func(n), Some(Function::Im | Function::Re)));
                if imaginary != int(0)
                    && !symbolic
                    && ask_rel(RelOp::Eq, &imaginary, &int(0), assumptions)? != Some(true)
                {
                    return Ok(None);
                }
                found.push(r);
            }
        }
    }
    if found.len() as i64 != d {
        return Ok(None);
    }
    let mut ordered: Vec<Basic> = Vec::new();
    for r in found {
        let mut position: usize = 0;
        for q in &ordered {
            match less(q, &r, assumptions)? {
                None => return Ok(None),
                Some(true) => position += 1,
                Some(false) => {}
            }
        }
        ordered.insert(position, r);
    }
    Ok(Some(ordered))
}

fn less(a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    if a == b {
        return Ok(None);
    }
    if matches!(a.head(), Head::IndexedRoot) || matches!(b.head(), Head::IndexedRoot) {
        let Some(difference) = numeric_value(&s2(b, a)?, 100) else {
            return Ok(None);
        };
        let (re, im) = parts(&difference);
        if im != Real::ZERO || re == Real::ZERO {
            return Ok(None);
        }
        return Ok(Some(re > Real::ZERO));
    }
    let (numerator, denominator) = numer_denom(&cancel(&s2(b, a)?)?)?;
    let query: Basic = match rel(RelOp::Gt, &m2(&numerator, &denominator)?, &int(0)) {
        Ok(q) => q,
        Err(PolyError::Type(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    ask_(&query, assumptions)
}

fn position(point: &Basic, roots: &[Basic], assumptions: &[Basic]) -> PolyResult<Option<i64>> {
    let n: i64 = roots.len() as i64;
    if is_neg_oo(point) {
        return Ok(Some(-1));
    }
    if is_oo(point) {
        return Ok(Some(2 * n - 1));
    }
    for (i, r) in roots.iter().enumerate() {
        let i: i64 = i as i64;
        if point == r {
            return Ok(Some(2 * i));
        }
        let difference: Basic = s2(point, r)?;
        if ask_rel(RelOp::Ge, &difference, &int(0), assumptions)? == Some(true)
            && ask_rel(RelOp::Le, &difference, &int(0), assumptions)? == Some(true)
        {
            return Ok(Some(2 * i));
        }
        match less(point, r, assumptions)? {
            None => return Ok(None),
            Some(true) => return Ok(Some(2 * i - 1)),
            Some(false) => {}
        }
    }
    Ok(Some(2 * n - 1))
}

struct Substitution {
    x: Basic,
    m: Basic,
}

fn cell_substitution(roots: &[Basic], cell: i64, s: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Substitution>> {
    let n: i64 = roots.len() as i64;
    if cell < -1 || cell > 2 * n - 1 || cell.rem_euclid(2) == 0 {
        return Ok(None);
    }
    let one_minus_s: Basic = s2(&int(1), s)?;
    if cell == -1 || cell == 2 * n - 1 {
        if n != 3 {
            return Ok(None);
        }
        let (r1, r2, r3) = (&roots[0], &roots[1], &roots[2]);
        if cell == 2 * n - 1 {
            let x: Basic = a2(r3, &d2(&m2(&s2(r3, r2)?, s)?, &one_minus_s)?)?;
            return Ok(Some(Substitution {
                x,
                m: d2(&s2(r2, r1)?, &s2(r3, r1)?)?,
            }));
        }
        let x: Basic = s2(r1, &d2(&m2(&s2(r2, r1)?, s)?, &one_minus_s)?)?;
        return Ok(Some(Substitution {
            x,
            m: d2(&s2(r3, r2)?, &s2(r3, r1)?)?,
        }));
    }
    let i: usize = ((cell - 1) / 2) as usize;
    let (lo, hi) = (&roots[i], &roots[i + 1]);
    let others: Vec<Basic> = roots
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i && *j != i + 1)
        .map(|(_, r)| r.clone())
        .collect();
    if n == 3 {
        let r: &Basic = &others[0];
        let width: Basic = s2(hi, lo)?;
        if less(hi, r, assumptions)? == Some(true) {
            return Ok(Some(Substitution {
                x: a2(lo, &m2(&width, s)?)?,
                m: d2(&width, &s2(r, lo)?)?,
            }));
        }
        return Ok(Some(Substitution {
            x: s2(hi, &m2(&width, s)?)?,
            m: d2(&width, &s2(hi, r)?)?,
        }));
    }
    for (pole, other) in [(&others[0], &others[1]), (&others[1], &others[0])] {
        for (start, end) in [(lo, hi), (hi, lo)] {
            let gamma: Basic = d2(&s2(pole, end)?, &s2(end, start)?)?;
            let x: Basic = d2(&a2(&m2(pole, s)?, &m2(start, &gamma)?)?, &a2(s, &gamma)?)?;
            let Some(image) = preimage(&x, s, other)? else { continue };
            if image == int(0) {
                continue;
            }
            let m: Basic = cancel(&d2(&int(1), &image)?)?;
            if ask_rel(RelOp::Gt, &m, &int(0), assumptions)? == Some(true)
                && ask_rel(RelOp::Lt, &m, &int(1), assumptions)? == Some(true)
            {
                return Ok(Some(Substitution { x, m }));
            }
        }
    }
    Ok(None)
}

fn preimage(x: &Basic, s: &Basic, point: &Basic) -> OptR {
    if is_oo(point) || is_neg_oo(point) {
        return Ok(Some(int(1)));
    }
    let (numerator, denominator) = numer_denom(&cancel(&s2(x, point)?)?)?;
    let Some(c) = coefficients(&numerator, s)? else {
        return Ok(None);
    };
    if c.keys().any(|k| *k != 0 && *k != 1) || !c.contains_key(&1) {
        return Ok(None);
    }
    let solution: Basic = cancel(&d2(&neg(&get(&c, 0))?, &c[&1])?)?;
    if cancel(&subs(&denominator, s, &solution)?)? == int(0) {
        return Ok(None);
    }
    Ok(Some(solution))
}

fn reduce_on_cell(
    r: &Basic,
    p: &Basic,
    e: &Basic,
    x: &Basic,
    roots: &[Basic],
    lo: &Basic,
    hi: &Basic,
    cell: i64,
    assumptions: &[Basic],
) -> PolyResult<Option<Reduction>> {
    let s: Basic = positive_dummy("s");
    let Some(substitution) = cell_substitution(roots, cell, &s, assumptions)? else {
        return Ok(None);
    };
    let (big_x, m) = (substitution.x, substitution.m);
    let (Some(mut s_lo), Some(mut s_hi)) = (preimage(&big_x, &s, lo)?, preimage(&big_x, &s, hi)?) else {
        return Ok(None);
    };
    let mut sign: Basic = int(1);
    if ask_rel(RelOp::Lt, &s_lo, &s_hi, assumptions)? == Some(false) {
        std::mem::swap(&mut s_lo, &mut s_hi);
        sign = int(-1);
    }
    for bound in [&s_lo, &s_hi] {
        if ask_rel(RelOp::Ge, bound, &int(0), assumptions)? != Some(true)
            || ask_rel(RelOp::Le, bound, &int(1), assumptions)? != Some(true)
        {
            return Ok(None);
        }
    }
    let Some((c, part)) = split_radicand(&subs(p, x, &big_x)?, &s, &m, assumptions)? else {
        return Ok(None);
    };
    let xp: Basic = diff(&big_x, &s)?;
    let r_x: Basic = subs(r, x, &big_x)?;
    let q: Basic = if *e == neg(&half())? {
        d2(&m2(&r_x, &xp)?, &m2(&sqrt(&c)?, &part)?)?
    } else {
        chain(&[
            r_x,
            xp,
            sqrt(&c)?,
            part,
            s.clone(),
            s2(&int(1), &s)?,
            s2(&int(1), &m2(&m, &s)?)?,
        ])?
    };
    let q: Basic = cancel(&m2(&sign, &q)?)?;
    if !is_rational_function(&q, &s) {
        return Ok(None);
    }
    Ok(Some(Reduction {
        q,
        m,
        s,
        lower: s_lo,
        upper: s_hi,
    }))
}

fn split_radicand(p_s: &Basic, s: &Basic, m: &Basic, assumptions: &[Basic]) -> PolyResult<Option<(Basic, Basic)>> {
    let (numerator, denominator) = numer_denom(&cancel(p_s)?)?;
    let mut c: Basic = int(1);
    let forms: [Basic; 3] = [s.clone(), s2(&int(1), s)?, s2(&int(1), &m2(m, s)?)?];
    let mut exponents: [i64; 3] = [0, 0, 0];
    let mut others: Vec<(Basic, i64)> = Vec::new();
    for (polynomial, orientation) in [(&numerator, 1i64), (&denominator, -1)] {
        let Some((coefficient, factors)) = factor_list_(polynomial, s)? else {
            return Ok(None);
        };
        c = m2(&c, &pwi(&coefficient, orientation)?)?;
        for (factor, multiplicity) in factors {
            let exponent: i64 = orientation * multiplicity;
            if !factor.has(s) {
                c = m2(&c, &pwi(&factor, exponent)?)?;
                continue;
            }
            let mut matched: bool = false;
            for (k, form) in forms.iter().enumerate() {
                let ratio: Basic = cancel(&d2(&factor, form)?)?;
                if !ratio.has(s) {
                    c = m2(&c, &pwi(&ratio, exponent)?)?;
                    exponents[k] += exponent;
                    matched = true;
                    break;
                }
            }
            if !matched {
                others.push((factor, exponent));
            }
        }
    }
    if exponents.iter().any(|v| v.rem_euclid(2) == 0) {
        return Ok(None);
    }
    if others.iter().any(|(_, k)| k.rem_euclid(2) != 0) {
        return Ok(None);
    }
    if ask_rel(RelOp::Gt, &c, &int(0), assumptions)? != Some(true) {
        return Ok(None);
    }
    let mut part: Basic = int(1);
    for (k, form) in forms.iter().enumerate() {
        part = m2(&part, &pwi(form, (exponents[k] - 1).div_euclid(2))?)?;
    }
    for (factor, exponent) in others {
        let half_: i64 = exponent.div_euclid(2);
        let mut factor: Basic = factor;
        if half_.rem_euclid(2) != 0 {
            let at_half: Basic = subs(&factor, s, &half())?;
            if ask_rel(RelOp::Lt, &at_half, &int(0), assumptions)? == Some(true) {
                factor = neg(&factor)?;
            } else if ask_rel(RelOp::Gt, &at_half, &int(0), assumptions)? != Some(true) {
                return Ok(None);
            }
        }
        part = m2(&part, &pwi(&factor, half_)?)?;
    }
    Ok(Some((c, part)))
}

thread_local! {
    static FUNCTION_DUMMIES: RefCell<HashMap<(Function, Vec<Basic>), Basic>> = RefCell::new(HashMap::new());
    static FUNCTION_VALUES: RefCell<Vec<(Basic, Basic)>> = const { RefCell::new(Vec::new()) };
}

const ELLIPTIC: [Function; 4] = [
    Function::EllipticK,
    Function::EllipticE,
    Function::EllipticF,
    Function::EllipticPi,
];

fn function_name(f: Function, n: usize) -> &'static str {
    match (f, n) {
        (Function::EllipticK, _) => "elliptic_k",
        (Function::EllipticF, _) => "elliptic_f",
        (Function::EllipticE, _) => "elliptic_e",
        _ => "elliptic_pi",
    }
}

/// An elliptic function of constant arguments as a dummy, the same one
/// for the same arguments (`_cached`).
fn cached(f: Function, args: &[Basic]) -> R {
    let key: (Function, Vec<Basic>) = (f, args.to_vec());
    if let Some(found) = FUNCTION_DUMMIES.with(|d| d.borrow().get(&key).cloned()) {
        return Ok(found);
    }
    let value: Basic = apply(f, args)?;
    if !matches!(func(&value), Some(g) if ELLIPTIC.contains(&g)) || !free(&value).is_empty() {
        return Ok(value);
    }
    let symbol: Basic = fresh(function_name(f, args.len()), &[]);
    FUNCTION_DUMMIES.with(|d| d.borrow_mut().insert(key, symbol.clone()));
    FUNCTION_VALUES.with(|v| v.borrow_mut().push((symbol.clone(), value)));
    Ok(symbol)
}

fn reveal(e: &Basic) -> R {
    let rule: Vec<(Basic, Basic)> =
        FUNCTION_VALUES.with(|v| v.borrow().iter().filter(|(d, _)| e.has(d)).cloned().collect());
    if rule.is_empty() {
        return Ok(e.clone());
    }
    replaced(e, &rule)
}

fn forget_functions() {
    if FUNCTION_VALUES.with(|v| v.borrow().len()) > 4096 {
        FUNCTION_DUMMIES.with(|d| d.borrow_mut().clear());
        FUNCTION_VALUES.with(|v| v.borrow_mut().clear());
    }
}

/// The radical `sqrt(W(s))` of a normal form (`_Legendre`, `_Elementary`).
#[derive(Clone)]
enum Radical {
    Legendre(Basic),
    Elementary(Basic),
}

impl Radical {
    fn m(&self) -> &Basic {
        match self {
            Radical::Legendre(m) | Radical::Elementary(m) => m,
        }
    }

    fn w(&self, s: &Basic) -> R {
        match self {
            Radical::Legendre(m) => chain(&[s.clone(), s2(&int(1), s)?, s2(&int(1), &m2(m, s)?)?]),
            Radical::Elementary(m) => m2(s, &s2(&int(1), &m2(m, s)?)?),
        }
    }

    /// `int_0^upper s**j / sqrt(W(s)) ds`.
    fn power(&self, j: i64, upper: &Basic) -> R {
        let m: &Basic = self.m();
        let boundary: Basic = sqrt(&self.w(upper)?)?;
        let mut values: HashMap<i64, Basic> = HashMap::new();
        match self {
            Radical::Legendre(_) => {
                let (f, e) = if *upper == int(1) {
                    (
                        cached(Function::EllipticK, std::slice::from_ref(m))?,
                        cached(Function::EllipticE, std::slice::from_ref(m))?,
                    )
                } else {
                    let phi: Basic = asin(&sqrt(upper)?)?;
                    (
                        cached(Function::EllipticF, &[phi.clone(), m.clone()])?,
                        cached(Function::EllipticE, &[phi, m.clone()])?,
                    )
                };
                values.insert(0, f.clone());
                values.insert(1, d2(&s2(&f, &e)?, m)?);
                let one_plus_m: Basic = a2(&int(1), m)?;
                for k in 1..j {
                    let value: Basic = d2(
                        &add_all(&[
                            chain(&[int(2 * k), one_plus_m.clone(), values[&k].clone()])?,
                            neg(&m2(&int(2 * k - 1), &values[&(k - 1)])?)?,
                            m2(&pwi(upper, k - 1)?, &boundary)?,
                        ])?,
                        &m2(&int(2 * k + 1), m)?,
                    )?;
                    values.insert(k + 1, value);
                }
                let mut k: i64 = 0;
                while k > j {
                    let value: Basic = d2(
                        &add_all(&[
                            chain(&[int(2 * k), one_plus_m.clone(), values[&k].clone()])?,
                            neg(&chain(&[int(2 * k + 1), m.clone(), values[&(k + 1)].clone()])?)?,
                            m2(&pwi(upper, k - 1)?, &boundary)?,
                        ])?,
                        &int(2 * k - 1),
                    )?;
                    values.insert(k - 1, value);
                    k -= 1;
                }
                m2(&int(2), &values[&j])
            }
            Radical::Elementary(_) => {
                values.insert(0, d2(&m2(&int(2), &asin(&sqrt(&m2(m, upper)?)?)?)?, &sqrt(m)?)?);
                let k_half = |k: i64| Basic::from_rational(crate::numbers::rat(2 * k + 1, 2));
                for k in 0..j {
                    let value: Basic = d2(
                        &s2(&m2(&k_half(k), &values[&k])?, &m2(&pwi(upper, k)?, &boundary)?)?,
                        &m2(&int(k + 1), m)?,
                    )?;
                    values.insert(k + 1, value);
                }
                let mut k: i64 = -1;
                while k > j - 1 {
                    let value: Basic = d2(
                        &a2(
                            &chain(&[int(k + 1), m.clone(), values[&(k + 1)].clone()])?,
                            &m2(&pwi(upper, k)?, &boundary)?,
                        )?,
                        &k_half(k),
                    )?;
                    values.insert(k, value);
                    k -= 1;
                }
                Ok(values[&j].clone())
            }
        }
    }

    /// `int_0^upper c / ((alpha s + beta) sqrt(W(s))) ds`.
    fn simple_pole(&self, c: &Basic, alpha: &Basic, beta: &Basic, upper: &Basic, assumptions: &[Basic]) -> OptR {
        match self {
            Radical::Legendre(m) => {
                let n: Basic = cancel(&d2(&neg(alpha)?, beta)?)?;
                let factor: Basic = d2(&m2(&int(2), c)?, beta)?;
                if *upper == int(1) {
                    return Ok(Some(m2(&factor, &cached(Function::EllipticPi, &[n, m.clone()])?)?));
                }
                Ok(Some(m2(
                    &factor,
                    &cached(Function::EllipticPi, &[n, asin(&sqrt(upper)?)?, m.clone()])?,
                )?))
            }
            Radical::Elementary(m) => {
                let p: Basic = cancel(&d2(&neg(beta)?, alpha)?)?;
                let big_a: Basic = cancel(&self.w(&p)?)?;
                let big_b: Basic = s2(&int(1), &chain(&[int(2), m.clone(), p.clone()])?)?;
                let Some(far) = ask_rel(RelOp::Gt, upper, &p, assumptions)? else {
                    return Ok(None);
                };
                let sigma: Basic = int(if far { 1 } else { -1 });
                let w_upper: Basic = cancel(&self.w(upper)?)?;
                let radical: Basic = if w_upper == int(0) { int(0) } else { sqrt(&w_upper)? };
                let distance: Basic = s2(upper, &p)?;
                let linear: Basic = a2(&m2(&int(2), &big_a)?, &m2(&big_b, &distance)?)?;
                let (at_upper, at_zero) = if ask_rel(RelOp::Gt, &big_a, &int(0), assumptions)? == Some(true) {
                    let inside: Basic = d2(
                        &a2(
                            &linear,
                            &chain(&[int(2), sigma.clone(), sqrt(&big_a)?, radical.clone()])?,
                        )?,
                        &m2(&sigma, &distance)?,
                    )?;
                    (d2(&log(&inside)?, &sqrt(&big_a)?)?, int(0))
                } else if ask_rel(RelOp::Lt, &big_a, &int(0), assumptions)? == Some(true) {
                    let root: Basic = sqrt(&neg(&big_a)?)?;
                    (
                        d2(&neg(&asin(&d2(&linear, &distance)?)?)?, &root)?,
                        d2(&pi(), &m2(&int(2), &root)?)?,
                    )
                } else {
                    return Ok(None);
                };
                Ok(Some(m2(
                    &d2(&m2(&neg(&sigma)?, c)?, alpha)?,
                    &s2(&at_upper, &at_zero)?,
                )?))
            }
        }
    }
}

/// The value of a reduction to Legendre's normal form (`legendre_reduction`).
pub fn legendre_reduction(reduction: &Reduction, assumptions: &[Basic]) -> OptR {
    let (q, m, s) = (&reduction.q, &reduction.m, &reduction.s);
    let (lower, upper) = (&reduction.lower, &reduction.upper);
    if *lower == int(0) && pole_at_zero(q, s)? {
        return Ok(None);
    }
    let mut total: Basic = int(0);
    let radical: Radical = Radical::Legendre(m.clone());
    for (bound, sign) in [(upper, 1i64), (lower, -1)] {
        let Some(value) = from_zero(q, s, bound, &radical, assumptions, lower, upper)? else {
            return Ok(None);
        };
        total = a2(&total, &m2(&int(sign), &value)?)?;
    }
    Ok(Some(reveal(&total)?))
}

fn pole_at_zero(q: &Basic, s: &Basic) -> PolyResult<bool> {
    let (_, denominator) = numer_denom(&cancel(q)?)?;
    Ok(subs(&denominator, s, &int(0))? == int(0))
}

struct PartialFractions {
    polynomial: Coefficients,
    poles: Vec<(Basic, Basic, Coefficients)>,
}

fn decomposed(q: &Basic, s: &Basic) -> PolyResult<Option<PartialFractions>> {
    let (numerator, denominator) = numer_denom(&cancel(q)?)?;
    let constants: ConstantField = ConstantField::new(&[numerator.clone(), denominator.clone()], s)?;
    let factored = match constants.factor_list(&denominator, s)? {
        Some(found) => Some(found),
        None => factor_list_(&denominator, s)?,
    };
    let Some((mut lead, factors)) = factored else {
        return Ok(None);
    };
    let mut linear: Vec<(Basic, Basic, i64)> = Vec::new();
    for (factor, multiplicity) in factors {
        if !factor.has(s) {
            lead = m2(&lead, &pwi(&factor, multiplicity)?)?;
            continue;
        }
        let Some(c) = coefficients(&factor, s)? else {
            return Ok(None);
        };
        if top(&c) != 1 {
            return Ok(None);
        }
        linear.push((c[&1].clone(), get(&c, 0), multiplicity));
    }
    let (Some(nc), Some(dc)) = (coefficients(&numerator, s)?, coefficients(&denominator, s)?) else {
        return Ok(None);
    };
    let mut polynomial: Coefficients = Coefficients::new();
    for (j, c) in quotient(&nc, &dc)? {
        polynomial.insert(j, constants.canonical(&c)?);
    }
    let mut poles: Vec<(Basic, Basic, Coefficients)> = Vec::new();
    let u: Basic = fresh("u", &[]);
    for (i, (alpha, beta, k)) in linear.iter().enumerate() {
        let mut rest: Basic = lead.clone();
        for (j, (alpha_j, beta_j, k_j)) in linear.iter().enumerate() {
            if j != i {
                rest = m2(&rest, &pwi(&a2(&m2(alpha_j, s)?, beta_j)?, *k_j)?)?;
            }
        }
        let at_u: Basic = d2(&s2(&u, beta)?, alpha)?;
        let big_a = coefficients(&cancel(&subs(&numerator, s, &at_u)?)?, &u)?;
        let big_b = coefficients(&cancel(&subs(&rest, s, &at_u)?)?, &u)?;
        let (Some(big_a), Some(big_b)) = (big_a, big_b) else {
            return Ok(None);
        };
        if !big_b.contains_key(&0) {
            return Ok(None);
        }
        let mut taylor: Vec<Basic> = Vec::new();
        for n in 0..*k {
            let mut term: Basic = get(&big_a, n);
            for j in 1..=n {
                term = s2(&term, &m2(&get(&big_b, j), &taylor[(n - j) as usize])?)?;
            }
            taylor.push(constants.canonical(&cancel(&d2(&term, &big_b[&0])?)?)?);
        }
        let mut powers: Coefficients = Coefficients::new();
        for (n, c) in taylor.into_iter().enumerate() {
            if c != int(0) {
                powers.insert(k - n as i64, c);
            }
        }
        poles.push((alpha.clone(), beta.clone(), powers));
    }
    Ok(Some(PartialFractions { polynomial, poles }))
}

fn quotient(numerator: &Coefficients, denominator: &Coefficients) -> PolyResult<Coefficients> {
    let mut remainder: Coefficients = numerator.clone();
    let d: i64 = top(denominator);
    let lead: Basic = denominator[&d].clone();
    let mut result: Coefficients = Coefficients::new();
    while !remainder.is_empty() && top(&remainder) >= d {
        let t: i64 = top(&remainder);
        let q: Basic = cancel(&d2(&remainder[&t], &lead)?)?;
        result.insert(t - d, q.clone());
        for (j, c) in denominator {
            let degree_: i64 = j + t - d;
            let value: Basic = cancel(&s2(&get(&remainder, degree_), &m2(&q, c)?)?)?;
            if value == int(0) || degree_ == t {
                remainder.remove(&degree_);
            } else {
                remainder.insert(degree_, value);
            }
        }
    }
    Ok(result.into_iter().filter(|(_, c)| *c != int(0)).collect())
}

fn from_zero(
    q: &Basic,
    s: &Basic,
    point: &Basic,
    radical: &Radical,
    assumptions: &[Basic],
    lower: &Basic,
    upper: &Basic,
) -> OptR {
    if *point == int(0) {
        return Ok(Some(int(0)));
    }
    let Some(parts_) = decomposed(q, s)? else {
        return Ok(None);
    };
    let mut total: Basic = int(0);
    for (j, c) in &parts_.polynomial {
        total = a2(&total, &m2(c, &radical.power(*j, point)?)?)?;
    }
    for (alpha, beta, powers) in &parts_.poles {
        for (k, c) in powers {
            let Some(value) = pole_from_zero(c, alpha, beta, *k, s, point, radical, assumptions, lower, upper)? else {
                return Ok(None);
            };
            total = a2(&total, &value)?;
        }
    }
    Ok(Some(total))
}

fn pole_from_zero(
    c: &Basic,
    alpha: &Basic,
    beta: &Basic,
    k: i64,
    s: &Basic,
    upper: &Basic,
    radical: &Radical,
    assumptions: &[Basic],
    range_lower: &Basic,
    range_upper: &Basic,
) -> OptR {
    if *beta == int(0) {
        return Ok(Some(m2(&d2(c, &pwi(alpha, k)?)?, &radical.power(-k, upper)?)?));
    }
    let at = |t: &Basic| a2(&m2(alpha, t)?, beta);
    if ask_rel(
        RelOp::Gt,
        &m2(&at(range_lower)?, &at(range_upper)?)?,
        &int(0),
        assumptions,
    )? != Some(true)
    {
        return Ok(None);
    }
    let pole: Basic = d2(&neg(beta)?, alpha)?;
    if k == 1 {
        return radical.simple_pole(c, alpha, beta, upper, assumptions);
    }
    let w: Basic = radical.w(s)?;
    let big_d: Basic = at(s)?;
    let big_n: Basic = s2(
        &m2(&diff(&w, s)?, &big_d)?,
        &chain(&[int(2 * (k - 1)), alpha.clone(), w.clone()])?,
    )?;
    let leading: Basic = cancel(&subs(&big_n, s, &pole)?)?;
    if leading == int(0) {
        return Ok(None);
    }
    let rest: Basic = cancel(&d2(&s2(&big_n, &leading)?, &pwi(&big_d, k)?)?)?;
    let Some(lower_orders) = from_zero(&rest, s, upper, radical, assumptions, range_lower, range_upper)? else {
        return Ok(None);
    };
    let boundary: Basic = d2(
        &m2(&int(2), &sqrt(&radical.w(upper)?)?)?,
        &pwi(&subs(&big_d, s, upper)?, k - 1)?,
    )?;
    Ok(Some(m2(&d2(c, &leading)?, &s2(&boundary, &lower_orders)?)?))
}

const TRIGONOMETRIC: [Function; 6] = [
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
];

/// The definite integral of an elliptic integrand, or `None`
/// (`elliptic_integral`).
pub fn elliptic_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    forget_functions();
    if f.preorder()
        .iter()
        .any(|n| matches!(func(n), Some(g) if TRIGONOMETRIC.contains(&g)))
    {
        return trigonometric(f, x, a, b, assumptions);
    }
    reduce(f, x, a, b, assumptions, 0)
}

fn plain_value(value: Basic) -> ConditionalValue {
    ConditionalValue::new(value, crate::core::logic::true_())
}

fn reduce(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    depth: usize,
) -> PolyResult<Option<ConditionalValue>> {
    if depth > MAX_DEPTH {
        return Ok(None);
    }
    let (constant, rest) = crate::core::expr::as_independent(f, x, false)?;
    let Some((r, p, e)) = radicand(&rest, x)? else {
        return Ok(None);
    };
    if let Some(even) = even_reduction(&r, &p, &e, x, a, b, assumptions, depth)? {
        return Ok(Some(even.scaled(&constant)?));
    }
    let Some(roots) = real_roots(&p, x, assumptions)? else {
        let mut found = complex_reduction(&r, &p, &e, x, a, b, assumptions, depth, None)?;
        if found.is_none() {
            found = two_pairs_reduction(&r, &p, &e, x, a, b, assumptions)?;
        }
        return match found {
            Some(v) => Ok(Some(v.scaled(&constant)?)),
            None => Ok(None),
        };
    };
    let (Some(lo_cell), Some(hi_cell)) = (position(a, &roots, assumptions)?, position(b, &roots, assumptions)?) else {
        return Ok(None);
    };
    let cell: i64 = if lo_cell.rem_euclid(2) != 0 {
        lo_cell
    } else {
        lo_cell + 1
    };
    if hi_cell != cell && hi_cell != cell + 1 {
        return Ok(None);
    }
    let n: i64 = roots.len() as i64;
    if n == 4 && (cell == -1 || cell == 2 * n - 1) {
        return match inverted_reduction(&r, &p, &e, x, &roots, a, b, cell, assumptions, depth, None)? {
            Some(v) => Ok(Some(v.scaled(&constant)?)),
            None => Ok(None),
        };
    }
    let Some(reduction) = reduce_on_cell(&r, &p, &e, x, &roots, a, b, cell, assumptions)? else {
        return Ok(None);
    };
    let Some(value) = legendre_reduction(&reduction, assumptions)? else {
        return Ok(None);
    };
    Ok(Some(plain_value(m2(&constant, &value)?)))
}

fn even_reduction(
    r: &Basic,
    p: &Basic,
    e: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    depth: usize,
) -> PolyResult<Option<ConditionalValue>> {
    let minus_x: Basic = neg(x)?;
    if subs(p, x, &minus_x)? != *p || subs(r, x, &minus_x)? != *r {
        return Ok(None);
    }
    if is_neg_oo(a) || ask_rel(RelOp::Ge, a, &int(0), assumptions)? != Some(true) {
        return Ok(None);
    }
    let y: Basic = positive_dummy("y");
    let root: Basic = sqrt(&y)?;
    let r_y: Basic = cancel(&subs(r, x, &root)?)?;
    let p_y: Basic = cancel(&subs(p, x, &root)?)?;
    if !is_rational_function(&r_y, &y) || !is_polynomial(&p_y, &y) {
        return Ok(None);
    }
    let radical: Basic = pw(&m2(&y, &p_y)?, e)?;
    let g: Basic = if *e == neg(&half())? {
        d2(&m2(&r_y, &radical)?, &int(2))?
    } else {
        d2(&m2(&r_y, &radical)?, &m2(&int(2), &y)?)?
    };
    let upper: Basic = if is_oo(b) { oo() } else { pwi(b, 2)? };
    reduce(&g, &y, &pwi(a, 2)?, &upper, assumptions, depth + 1)
}

/// The factors of a radicand with one complex pair: `lam`, the real
/// roots, and `(x - p)**2 + q2` (`_ComplexFactors`).
#[derive(Clone)]
struct ComplexFactors {
    lam: Basic,
    roots: Vec<Basic>,
    p: Basic,
    q2: Basic,
    facts: Vec<Basic>,
}

fn inverted_reduction(
    r: &Basic,
    p: &Basic,
    e: &Basic,
    x: &Basic,
    roots: &[Basic],
    a: &Basic,
    b: &Basic,
    cell: i64,
    assumptions: &[Basic],
    depth: usize,
    factors: Option<&ComplexFactors>,
) -> PolyResult<Option<ConditionalValue>> {
    let u: Basic = positive_dummy("u");
    let upper_cell: bool = cell == 2 * roots.len() as i64 - 1;
    let root: Basic = if upper_cell {
        roots[roots.len() - 1].clone()
    } else {
        roots[0].clone()
    };
    let inverse: Basic = d2(&int(1), &u)?;
    let big_x: Basic = if upper_cell {
        a2(&root, &inverse)?
    } else {
        s2(&root, &inverse)?
    };
    let mut factors_u: Option<ComplexFactors> = None;
    let p_u: Basic = match factors {
        None => cancel(&m2(&subs(p, x, &big_x)?, &pwi(&u, 4)?)?)?,
        Some(f) => {
            let other: Basic = if upper_cell {
                f.roots[0].clone()
            } else {
                f.roots[1].clone()
            };
            let shift: Basic = s2(&root, &f.p)?;
            let c: Basic = a2(&pwi(&shift, 2)?, &f.q2)?;
            let p_u_: Basic = if upper_cell {
                d2(&neg(&shift)?, &c)?
            } else {
                d2(&shift, &c)?
            };
            let q2_u: Basic = d2(&f.q2, &pwi(&c, 2)?)?;
            let root_u: Basic = d2(&int(-1), &s2(&root, &other)?)?;
            let lam_u: Basic = chain(&[f.lam.clone(), s2(&root, &other)?, c.clone()])?;
            let value: Basic = chain(&[lam_u.clone(), s2(&u, &root_u)?, a2(&pwi(&s2(&u, &p_u_)?, 2)?, &q2_u)?])?;
            factors_u = Some(ComplexFactors {
                lam: lam_u,
                roots: vec![root_u],
                p: p_u_,
                q2: q2_u,
                facts: f.facts.clone(),
            });
            value
        }
    };
    let r_u: Basic = cancel(&subs(r, x, &big_x)?)?;
    let weight: Basic = pw(&u, &s2(&m2(&int(-4), e)?, &int(2))?)?;
    let g: Basic = chain(&[r_u.clone(), pw(&p_u, e)?, weight.clone()])?;
    let (lower, upper) = if upper_cell {
        (
            if is_oo(b) { int(0) } else { d2(&int(1), &s2(b, &root)?)? },
            if *a == root { oo() } else { d2(&int(1), &s2(a, &root)?)? },
        )
    } else {
        (
            if is_neg_oo(a) {
                int(0)
            } else {
                d2(&int(1), &s2(&root, a)?)?
            },
            if *b == root { oo() } else { d2(&int(1), &s2(&root, b)?)? },
        )
    };
    if !is_polynomial(&p_u, &u) {
        return Ok(None);
    }
    if let Some(fu) = factors_u {
        return complex_reduction(
            &m2(&r_u, &weight)?,
            &p_u,
            e,
            &u,
            &lower,
            &upper,
            assumptions,
            depth + 1,
            Some(fu),
        );
    }
    reduce(&g, &u, &lower, &upper, assumptions, depth + 1)
}

const ROOTOF_SIZE: usize = 2500;

fn printed_size(e: &Basic) -> usize {
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for n in e.preorder() {
        if let Head::Dummy(data, _) = n.head() {
            if !rule.iter().any(|(k, _)| *k == n) {
                let stripped: &str = data.name.trim_end_matches(|c: char| c.is_ascii_digit());
                rule.push((n.clone(), crate::core::symbol::Dummy::new(stripped, &[]).0));
            }
        }
    }
    replaced_plain(e, &rule).to_string().len()
}

fn restore(value: &Basic, radicals: &[(Basic, Basic)], facts: &[Basic]) -> OptR {
    let (values, _) = numeric_values(facts);
    if values
        .iter()
        .any(|(_, v)| v.preorder().iter().any(|n| matches!(n.head(), Head::IndexedRoot)))
    {
        let mut value: Basic = value.clone();
        let timeout: Option<f64> = crate::settings::settings().timeout;
        let compact = crate::timeout::attempt_within(timeout.map(|t| t / 8.0), || {
            crate::simplify::radsimp::radsimp(&value, true, 4)
        })?;
        if let Some(compact) = compact {
            if printed_size(&compact) < printed_size(&value) {
                value = compact;
            }
        }
        if printed_size(&value) > ROOTOF_SIZE {
            return Ok(None);
        }
        return Ok(Some(numeric(&replaced_plain(&value, radicals), &values, false)?));
    }
    Ok(Some(numeric(&replaced(value, radicals)?, &values, true)?))
}

fn complex_factors(p: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<ComplexFactors>> {
    let Some((coefficient, factors)) = factor_list_(p, x)? else {
        return Ok(None);
    };
    let mut lam: Basic = coefficient;
    let mut roots: Vec<Basic> = Vec::new();
    let mut quadratic: Option<(Basic, Basic)> = None;
    let mut pair: Option<(Basic, Basic)> = None;
    let mut facts: Vec<Basic> = Vec::new();
    for (factor, multiplicity) in factors {
        if !factor.has(x) {
            lam = m2(&lam, &pwi(&factor, multiplicity)?)?;
            continue;
        }
        if multiplicity != 1 {
            return Ok(None);
        }
        let Some(c) = coefficients(&factor, x)? else {
            return Ok(None);
        };
        let d: i64 = top(&c);
        let lead: Basic = c[&d].clone();
        if d == 1 {
            roots.push(cancel(&d2(&neg(&get(&c, 0))?, &lead)?)?);
        } else if d == 2 && quadratic.is_none() {
            quadratic = Some((cancel(&d2(&get(&c, 1), &lead)?)?, cancel(&d2(&get(&c, 0), &lead)?)?));
        } else if (d == 3 || d == 4)
            && quadratic.is_none()
            && pair.is_none()
            && roots.is_empty()
            && free(&factor).iter().all(|s| s == x)
        {
            let Some(numeric_) = numeric_factors(&cancel(&d2(&factor, &lead)?)?, x)? else {
                return Ok(None);
            };
            if numeric_.pairs.len() != 1 {
                return Ok(None);
            }
            roots.extend(numeric_.reals);
            pair = Some(numeric_.pairs[0].clone());
            facts = numeric_.facts;
        } else {
            return Ok(None);
        }
        lam = m2(&lam, &lead)?;
    }
    if let Some((c1, c0)) = quadratic {
        let p_: Basic = d2(&neg(&c1)?, &int(2))?;
        let q2: Basic = cancel(&s2(&c0, &pwi(&p_, 2)?)?)?;
        pair = Some((p_, q2));
    }
    let Some((p_, q2)) = pair else { return Ok(None) };
    if roots.len() != 1 && roots.len() != 2 {
        return Ok(None);
    }
    let combined: Vec<Basic> = with_facts(assumptions, &facts);
    if ask_rel(RelOp::Gt, &q2, &int(0), &combined)? != Some(true) {
        return Ok(None);
    }
    if roots.len() == 2 {
        match less(&roots[0], &roots[1], &combined)? {
            None => return Ok(None),
            Some(false) => roots.reverse(),
            Some(true) => {}
        }
    }
    Ok(Some(ComplexFactors {
        lam: cancel(&lam)?,
        roots,
        p: p_,
        q2,
        facts,
    }))
}

/// The map `t -> x` of a cell with a complex pair (`_CosineMap`).
struct CosineMap {
    x: Basic,
    m: Basic,
    g2: Basic,
    sigma: Basic,
    preimage: Basic,
    infinite: Option<Basic>,
    radicals: Vec<(Basic, Basic)>,
    facts: Vec<Basic>,
}

impl CosineMap {
    fn preimage(&self, y: &Basic, x: &Basic) -> R {
        if Some(y) == self.infinite.as_ref() {
            return Ok(int(-1));
        }
        cancel(&subs(&self.preimage, x, y)?)
    }
}

fn radical(square: &Basic, radicals: &mut Vec<(Basic, Basic)>, facts: &mut Vec<Basic>) -> R {
    if free(square).is_empty() {
        return sqrt(square);
    }
    let symbol: Basic = positive_dummy("A");
    let root: Basic = sqrt(square)?;
    radicals.push((symbol.clone(), root.clone()));
    facts.push(rel(RelOp::Eq, &symbol, &root)?);
    Ok(symbol)
}

fn cosine_map(factors: &ComplexFactors, cell: i64, t: &Basic, x: &Basic) -> PolyResult<Option<CosineMap>> {
    let (p, q2) = (&factors.p, &factors.q2);
    let mut radicals: Vec<(Basic, Basic)> = Vec::new();
    let mut facts: Vec<Basic> = Vec::new();
    let square = |r: &Basic| a2(&pwi(&s2(r, p)?, 2)?, q2);
    if factors.roots.len() == 2 {
        if cell != 1 {
            return Ok(None);
        }
        let (b, a) = (&factors.roots[0], &factors.roots[1]);
        let big_a: Basic = radical(&square(a)?, &mut radicals, &mut facts)?;
        let big_b: Basic = radical(&square(b)?, &mut radicals, &mut facts)?;
        let gap: Basic = pwi(&s2(a, b)?, 2)?;
        if !radicals.is_empty() {
            facts.push(rel(RelOp::Gt, &gap, &pwi(&s2(&big_a, &big_b)?, 2)?)?);
            facts.push(rel(RelOp::Lt, &gap, &pwi(&a2(&big_a, &big_b)?, 2)?)?);
        }
        let ab: Basic = m2(a, &big_b)?;
        let ba: Basic = m2(b, &big_a)?;
        let big_x: Basic = d2(
            &s2(&a2(&ab, &ba)?, &m2(t, &s2(&ab, &ba)?)?)?,
            &a2(&a2(&big_a, &big_b)?, &m2(t, &s2(&big_a, &big_b)?)?)?,
        )?;
        let m: Basic = cancel(&d2(
            &s2(&gap, &pwi(&s2(&big_a, &big_b)?, 2)?)?,
            &chain(&[int(4), big_a.clone(), big_b.clone()])?,
        )?)?;
        let preimage: Basic = d2(
            &s2(&a2(&ab, &ba)?, &m2(x, &a2(&big_a, &big_b)?)?)?,
            &a2(&s2(&ab, &ba)?, &m2(x, &s2(&big_a, &big_b)?)?)?,
        )?;
        let g2: Basic = d2(&int(1), &m2(&big_a, &big_b)?)?;
        return Ok(Some(CosineMap {
            x: big_x,
            m,
            g2,
            sigma: int(-1),
            preimage,
            infinite: None,
            radicals,
            facts,
        }));
    }
    let a: &Basic = &factors.roots[0];
    let big_a: Basic = radical(&square(a)?, &mut radicals, &mut facts)?;
    let shift: Basic = s2(a, p)?;
    if !radicals.is_empty() {
        facts.push(rel(RelOp::Gt, &pwi(&big_a, 2)?, &pwi(&shift, 2)?)?);
    }
    if cell == 1 {
        let r: Basic = d2(&s2(x, a)?, &big_a)?;
        let big_x: Basic = a2(a, &d2(&m2(&big_a, &s2(&int(1), t)?)?, &a2(&int(1), t)?)?)?;
        let m: Basic = cancel(&d2(&s2(&big_a, &shift)?, &m2(&int(2), &big_a)?)?)?;
        let preimage: Basic = d2(&s2(&int(1), &r)?, &a2(&int(1), &r)?)?;
        return Ok(Some(CosineMap {
            x: big_x,
            m,
            g2: d2(&int(1), &big_a)?,
            sigma: int(-1),
            preimage,
            infinite: Some(oo()),
            radicals,
            facts,
        }));
    }
    if cell == -1 {
        let r: Basic = d2(&s2(a, x)?, &big_a)?;
        let big_x: Basic = s2(a, &d2(&m2(&big_a, &s2(&int(1), t)?)?, &a2(&int(1), t)?)?)?;
        let m: Basic = cancel(&d2(&a2(&big_a, &shift)?, &m2(&int(2), &big_a)?)?)?;
        let preimage: Basic = d2(&s2(&int(1), &r)?, &a2(&int(1), &r)?)?;
        return Ok(Some(CosineMap {
            x: big_x,
            m,
            g2: d2(&int(1), &big_a)?,
            sigma: int(1),
            preimage,
            infinite: Some(neg(&oo())?),
            radicals,
            facts,
        }));
    }
    Ok(None)
}

fn parity_parts(g: &Basic, t: &Basic, w: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let (numerator, denominator) = numer_denom(&cancel(g)?)?;
    if !is_polynomial(&numerator, t) || !is_polynomial(&denominator, t) {
        return Ok(None);
    }
    let minus_t: Basic = neg(t)?;
    let mirrored: Basic = subs(&denominator, t, &minus_t)?;
    let numerator: Basic = expand(&m2(&numerator, &mirrored)?)?;
    let denominator: Basic = expand(&m2(&denominator, &mirrored)?)?;
    let reflected: Basic = subs(&numerator, t, &minus_t)?;
    let even: Basic = expand(&d2(&a2(&numerator, &reflected)?, &int(2))?)?;
    let odd: Basic = cancel(&d2(&s2(&numerator, &reflected)?, &m2(&int(2), t)?)?)?;
    let mut converted_parts: Vec<Basic> = Vec::new();
    for polynomial in [&even, &odd, &denominator] {
        let Some(c) = coefficients(polynomial, t)? else {
            return Ok(None);
        };
        let mut converted: Basic = int(0);
        for (k, value) in &c {
            if k.rem_euclid(2) != 0 {
                return Ok(None);
            }
            converted = a2(&converted, &m2(value, &pwi(w, k / 2)?)?)?;
        }
        converted_parts.push(converted);
    }
    Ok(Some((
        cancel(&d2(&converted_parts[0], &converted_parts[2])?)?,
        cancel(&d2(&converted_parts[1], &converted_parts[2])?)?,
    )))
}

fn complex_reduction(
    r: &Basic,
    p: &Basic,
    e: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    depth: usize,
    factors: Option<ComplexFactors>,
) -> PolyResult<Option<ConditionalValue>> {
    let factors: ComplexFactors = match factors {
        Some(f) => f,
        None => match complex_factors(p, x, assumptions)? {
            Some(f) => f,
            None => return Ok(None),
        },
    };
    let roots: &[Basic] = &factors.roots;
    let assumptions: Vec<Basic> = with_facts(assumptions, &factors.facts);
    let (Some(lo_cell), Some(hi_cell)) = (position(a, roots, &assumptions)?, position(b, roots, &assumptions)?) else {
        return Ok(None);
    };
    let cell: i64 = if lo_cell.rem_euclid(2) != 0 {
        lo_cell
    } else {
        lo_cell + 1
    };
    if hi_cell != cell && hi_cell != cell + 1 {
        return Ok(None);
    }
    if roots.len() == 2 && (cell == -1 || cell == 3) {
        let numeric_: Option<&ComplexFactors> = if factors.facts.is_empty() { None } else { Some(&factors) };
        return inverted_reduction(r, p, e, x, roots, a, b, cell, &assumptions, depth, numeric_);
    }
    let lam: Basic = if cell == 2 * roots.len() as i64 - 1 {
        factors.lam.clone()
    } else {
        neg(&factors.lam)?
    };
    if ask_rel(RelOp::Gt, &lam, &int(0), &assumptions)? != Some(true) {
        return Ok(None);
    }
    let t: Basic = fresh("t", &[]);
    let Some(mapping) = cosine_map(&factors, cell, &t, x)? else {
        return Ok(None);
    };
    let (big_x, m) = (&mapping.x, &mapping.m);
    let assumptions: Vec<Basic> = with_facts(&assumptions, &mapping.facts);
    let (t0, t1) = (mapping.preimage(a, x)?, mapping.preimage(b, x)?);
    let xp: Basic = diff(big_x, &t)?;
    let root: Basic = sqrt(&d2(&lam, &mapping.g2)?)?;
    let r_x: Basic = subs(r, x, big_x)?;
    let g: Basic = if *e == neg(&half())? {
        d2(
            &m2(&r_x, &xp)?,
            &chain(&[root.clone(), mapping.sigma.clone(), xp.clone()])?,
        )?
    } else {
        chain(&[
            r_x.clone(),
            xp.clone(),
            root.clone(),
            mapping.sigma.clone(),
            xp.clone(),
            s2(&int(1), &pwi(&t, 2)?)?,
            a2(&s2(&int(1), m)?, &m2(m, &pwi(&t, 2)?)?)?,
        ])?
    };
    let g: Basic = cancel(&g)?;
    if !is_rational_function(&g, &t) {
        return Ok(None);
    }
    let parts_: CosineParts = CosineParts::new(&g, &t, m, &assumptions)?;
    let Some(value) = parts_.between(&t0, &t1)? else {
        return Ok(None);
    };
    let mut all_facts: Vec<Basic> = factors.facts.clone();
    all_facts.extend(mapping.facts.iter().cloned());
    Ok(restore(&reveal(&value)?, &mapping.radicals, &all_facts)?.map(plain_value))
}

/// The even and odd parts of the integrand after the cosine map (`_CosineParts`).
struct CosineParts {
    m: Basic,
    assumptions: Vec<Basic>,
    s: Basic,
    valid: bool,
    q_e: Basic,
    q_o: Basic,
    poles: Vec<Basic>,
}

impl CosineParts {
    fn new(g: &Basic, t: &Basic, m: &Basic, assumptions: &[Basic]) -> PolyResult<CosineParts> {
        let s: Basic = positive_dummy("s");
        let w: Basic = positive_dummy("w");
        let found = parity_parts(g, t, &w)?;
        let (mut q_e, mut q_o) = (int(0), int(0));
        let one_minus_s: Basic = s2(&int(1), &s)?;
        if let Some((even, odd)) = &found {
            q_e = cancel(&d2(&subs(even, &w, &one_minus_s)?, &int(2))?)?;
            q_o = cancel(&d2(&subs(odd, &w, &one_minus_s)?, &int(2))?)?;
        }
        let (_, denominator) = numer_denom(&cancel(g)?)?;
        let mut poles: Vec<Basic> = Vec::new();
        for end in [int(1), int(-1)] {
            if subs(&denominator, t, &end)? == int(0) {
                poles.push(end);
            }
        }
        Ok(CosineParts {
            m: m.clone(),
            assumptions: assumptions.to_vec(),
            s,
            valid: found.is_some(),
            q_e,
            q_o,
            poles,
        })
    }

    fn between(&self, t0: &Basic, t1: &Basic) -> OptR {
        if !self.valid || self.poles.contains(t0) || self.poles.contains(t1) {
            return Ok(None);
        }
        let ends: bool = (*t0 == int(1) && *t1 == int(-1)) || (*t0 == int(-1) && *t1 == int(1));
        if ends {
            let Some(value) = self.piece(&int(0), &int(1), &int(1), true)? else {
                return Ok(None);
            };
            return Ok(Some(chain(&[int(2), t1.clone(), value])?));
        }
        let (Some(first_sign), Some(second_sign)) = (self.sign(t0)?, self.sign(t1)?) else {
            return Ok(None);
        };
        if first_sign == 0 || second_sign == 0 || first_sign == second_sign {
            let sigma: i64 = if first_sign == 1 || second_sign == 1 { 1 } else { -1 };
            return self.piece(&m2(&int(sigma), t0)?, &m2(&int(sigma), t1)?, &int(sigma), false);
        }
        let first = self.piece(&m2(&int(first_sign), t0)?, &int(0), &int(first_sign), false)?;
        let second = self.piece(&int(0), &m2(&int(second_sign), t1)?, &int(second_sign), false)?;
        match (first, second) {
            (Some(a), Some(b)) => Ok(Some(a2(&a, &b)?)),
            _ => Ok(None),
        }
    }

    fn sign(&self, tau: &Basic) -> PolyResult<Option<i64>> {
        if *tau == int(0) {
            return Ok(Some(0));
        }
        if ask_rel(RelOp::Gt, tau, &int(0), &self.assumptions)? == Some(true) {
            return Ok(Some(1));
        }
        if ask_rel(RelOp::Lt, tau, &int(0), &self.assumptions)? == Some(true) {
            return Ok(Some(-1));
        }
        Ok(None)
    }

    fn piece(&self, u0: &Basic, u1: &Basic, sigma: &Basic, even_only: bool) -> OptR {
        let s0: Basic = cancel(&s2(&int(1), &pwi(u0, 2)?)?)?;
        let s1: Basic = cancel(&s2(&int(1), &pwi(u1, 2)?)?)?;
        if s0 == s1 {
            return Ok(Some(int(0)));
        }
        let Some(increasing) = ask_rel(RelOp::Lt, &s0, &s1, &self.assumptions)? else {
            return Ok(None);
        };
        let (lower, upper) = if increasing { (&s0, &s1) } else { (&s1, &s0) };
        let mut total: Basic = int(0);
        let pieces: [(&Basic, Radical, Basic, bool); 2] = [
            (&self.q_e, Radical::Legendre(self.m.clone()), sigma.clone(), true),
            (&self.q_o, Radical::Elementary(self.m.clone()), int(1), false),
        ];
        for (q, radical_, weight, legendre) in pieces {
            if *q == int(0) || (even_only && !legendre) {
                continue;
            }
            let at_s0 = from_zero(q, &self.s, &s0, &radical_, &self.assumptions, lower, upper)?;
            let at_s1 = from_zero(q, &self.s, &s1, &radical_, &self.assumptions, lower, upper)?;
            let (Some(at_s0), Some(at_s1)) = (at_s0, at_s1) else {
                return Ok(None);
            };
            total = a2(&total, &m2(&weight, &s2(&at_s0, &at_s1)?)?)?;
        }
        Ok(Some(total))
    }
}

/// The numerical factorisation of a radicand with rational coefficients:
/// its real roots and its complex pairs as dummies with numeric values
/// (`_Numeric`).
struct Numeric {
    reals: Vec<Basic>,
    pairs: Vec<(Basic, Basic)>,
    facts: Vec<Basic>,
}

const PRECISION: usize = 200;

fn numeric_factors(p: &Basic, x: &Basic) -> PolyResult<Option<Numeric>> {
    let Some(dense) = rational_coefficients(p, x)? else {
        return Ok(None);
    };
    let Some(exact_reals) = real_roots_rational(p, x)? else {
        return Ok(None);
    };
    let degree_: usize = dense.len() - 1;
    let mut common: Int = Int::from(1);
    for c in &dense {
        let d: Int = denom(c);
        common = crate::numbers::lcm(&common, &d);
    }
    let integers: Vec<Int> = dense
        .iter()
        .map(|c| numer(&(c.clone() * Rat::from(common.clone()))))
        .collect();
    let Some(numerical) = crate::identification::polyroots(&integers, PRECISION, 200) else {
        return Ok(None);
    };
    // mpmath's cleanup: the imaginary parts of the real roots are rounding noise
    let noise: Real = real_rat(
        &Rat::from_parts(Int::from(1), dashu_int::UBig::ONE << (PRECISION - 20)),
        PRECISION,
    );
    let mut upper: Vec<Complex> = numerical.into_iter().filter(|r| parts(r).1 > noise).collect();
    upper.sort_by(|a, b| {
        parts(a)
            .0
            .to_f64()
            .value()
            .partial_cmp(&parts(b).0.to_f64().value())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if exact_reals.len() + 2 * upper.len() != degree_ {
        return Ok(None);
    }
    let mut reals: Vec<Basic> = Vec::new();
    let mut pairs: Vec<(Basic, Basic)> = Vec::new();
    let mut facts: Vec<Basic> = Vec::new();
    for r in exact_reals {
        if r.is_rational_class() {
            reals.push(r);
            continue;
        }
        let c: Basic = real_dummy("r");
        facts.push(rel(RelOp::Eq, &c, &r)?);
        reals.push(c);
    }
    for z in &upper {
        let Some((real_part, modulus2)) = algebraic_parts(p, x, z)? else {
            return Ok(None);
        };
        let (pp, q2) = (real_dummy("p"), positive_dummy("q2"));
        facts.push(rel(RelOp::Eq, &pp, &real_part)?);
        facts.push(rel(RelOp::Eq, &q2, &s2(&modulus2, &pwi(&real_part, 2)?)?)?);
        pairs.push((pp, q2));
    }
    Ok(Some(Numeric { reals, pairs, facts }))
}

fn algebraic_parts(p: &Basic, x: &Basic, r: &Complex) -> PolyResult<Option<(Basic, Basic)>> {
    let (y, z) = (fresh("y", &[]), fresh("z", &[]));
    let Some(n) = degree(p, x)? else { return Ok(None) };
    let p_y: Basic = subs(p, x, &y)?;
    let images: [Basic; 2] = [
        subs(p, x, &s2(&m2(&int(2), &z)?, &y)?)?,
        expand(&m2(&pwi(&y, n)?, &subs(p, x, &d2(&z, &y)?)?)?)?,
    ];
    let (re, im) = parts(r);
    let targets: [Real; 2] = [re.clone(), &(&re * &re) + &(&im * &im)];
    let tiny = |k: i64| {
        let mut ten: Int = Int::from(1);
        for _ in 0..2 * k {
            ten *= Int::from(10);
        }
        real_rat(&(Rat::from(Int::from(1)) / Rat::from(ten)), PRECISION)
    };
    let mut found: Vec<Basic> = Vec::new();
    for (image, target) in images.iter().zip(targets.iter()) {
        let squarefree = (|| -> PolyResult<Vec<(Basic, Basic)>> {
            let combined: Basic = crate::polys::polytools::resultant_expr(&p_y, image, std::slice::from_ref(&y))?;
            Ok(crate::polys::polytools::sqf_list(&combined, std::slice::from_ref(&z))?.1)
        })();
        let squarefree = match squarefree {
            Ok(v) => v,
            Err(err) if gives_up(&err) => return Ok(None),
            Err(err) => return Err(err),
        };
        let mut part: Basic = int(1);
        for (base, _) in &squarefree {
            part = m2(&part, base)?;
        }
        let candidates: Vec<Basic> = match real_roots_rational(&expand(&part)?, &z)? {
            Some(c) if !c.is_empty() => c,
            _ => return Ok(None),
        };
        // squared distances, compared with the squares of the tolerances
        let mut distances: Vec<(Real, usize)> = Vec::new();
        for (i, c) in candidates.iter().enumerate() {
            let Some(value) = numeric_value(c, PRECISION) else {
                return Ok(None);
            };
            let (vr, vi) = parts(&value);
            let dr: Real = &vr - target;
            distances.push((&(&dr * &dr) + &(&vi * &vi), i));
        }
        distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        if distances.len() > 1 && distances[1].0 < tiny(20) {
            return Ok(None);
        }
        if distances[0].0 > tiny(30) {
            return Ok(None);
        }
        found.push(candidates[distances[0].1].clone());
    }
    Ok(Some((found[0].clone(), found[1].clone())))
}

struct TwoPairs {
    lam: Basic,
    pairs: Vec<(Basic, Basic)>,
    facts: Vec<Basic>,
}

fn two_pairs(p: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<TwoPairs>> {
    let Some((coefficient, factors)) = factor_list_(p, x)? else {
        return Ok(None);
    };
    let mut lam: Basic = coefficient;
    let mut pairs: Vec<(Basic, Basic)> = Vec::new();
    let mut facts: Vec<Basic> = Vec::new();
    for (factor, multiplicity) in factors {
        if !factor.has(x) {
            lam = m2(&lam, &pwi(&factor, multiplicity)?)?;
            continue;
        }
        if multiplicity != 1 {
            return Ok(None);
        }
        let Some(c) = coefficients(&factor, x)? else {
            return Ok(None);
        };
        let d: i64 = top(&c);
        let lead: Basic = c[&d].clone();
        lam = m2(&lam, &lead)?;
        if d == 2 {
            let c1: Basic = cancel(&d2(&get(&c, 1), &lead)?)?;
            let c0: Basic = cancel(&d2(&get(&c, 0), &lead)?)?;
            let p_: Basic = d2(&neg(&c1)?, &int(2))?;
            pairs.push((p_.clone(), cancel(&s2(&c0, &pwi(&p_, 2)?)?)?));
        } else if d == 4 && free(&factor).iter().all(|s| s == x) {
            let Some(numeric_) = numeric_factors(&cancel(&d2(&factor, &lead)?)?, x)? else {
                return Ok(None);
            };
            if !numeric_.reals.is_empty() {
                return Ok(None);
            }
            pairs.extend(numeric_.pairs);
            facts.extend(numeric_.facts);
        } else {
            return Ok(None);
        }
    }
    if pairs.len() != 2 {
        return Ok(None);
    }
    let combined: Vec<Basic> = with_facts(assumptions, &facts);
    for (_, q2) in &pairs {
        if ask_rel(RelOp::Gt, q2, &int(0), &combined)? != Some(true) {
            return Ok(None);
        }
    }
    Ok(Some(TwoPairs {
        lam: cancel(&lam)?,
        pairs,
        facts,
    }))
}

/// The map `t -> x` of two complex pairs (`_TangentMap`).
struct TangentMap {
    x: Basic,
    xp: Basic,
    preimage: Basic,
    pole: Option<Basic>,
    infinite: Basic,
    lam_: Basic,
    m: Basic,
    b1b2: Basic,
    bilinear: bool,
    radicals: Vec<(Basic, Basic)>,
    facts: Vec<Basic>,
}

impl TangentMap {
    fn preimage(&self, y: &Basic, x: &Basic) -> R {
        if is_oo(y) || is_neg_oo(y) {
            return Ok(if self.bilinear {
                self.infinite.clone()
            } else {
                y.clone()
            });
        }
        cancel(&subs(&self.preimage, x, y)?)
    }
}

fn named(value: &Basic, facts: &mut Vec<Basic>, numeric_: bool) -> R {
    if !numeric_ || free(value).is_empty() {
        return Ok(value.clone());
    }
    let symbol: Basic = positive_dummy("c");
    facts.push(rel(RelOp::Eq, &symbol, value)?);
    Ok(symbol)
}

fn tangent_map(
    pairs: &[(Basic, Basic)],
    t: &Basic,
    x: &Basic,
    assumptions: &[Basic],
    numeric_: bool,
) -> PolyResult<Option<TangentMap>> {
    let ((p1, q1), (p2, q2)) = (&pairs[0], &pairs[1]);
    let mut radicals: Vec<(Basic, Basic)> = Vec::new();
    let mut facts: Vec<Basic> = Vec::new();
    let (big_x, mut big_a, mut big_b, preimage_, xp, pole, infinite, bilinear);
    if cancel(&s2(p1, p2)?)? == int(0) {
        big_x = a2(p1, t)?;
        big_a = vec![int(1), int(1)];
        big_b = vec![q1.clone(), q2.clone()];
        preimage_ = s2(x, p1)?;
        xp = int(1);
        pole = None;
        infinite = oo();
        bilinear = false;
    } else {
        let c1: Basic = a2(&pwi(p1, 2)?, q1)?;
        let c2: Basic = a2(&pwi(p2, 2)?, q2)?;
        let u: Basic = cancel(&d2(&s2(&c1, &c2)?, &s2(p1, p2)?)?)?;
        let v: Basic = cancel(&s2(&m2(p1, &u)?, &c1)?)?;
        let d: Basic = radical(
            &cancel(&s2(&pwi(&u, 2)?, &m2(&int(4), &v)?)?)?,
            &mut radicals,
            &mut facts,
        )?;
        if !radicals.is_empty() {
            facts.push(rel(RelOp::Gt, &pwi(&d, 2)?, &pwi(&s2(&u, &m2(&int(2), p1)?)?, 2)?)?);
        }
        let mut alpha: Basic = d2(&a2(&u, &d)?, &int(2))?;
        let mut beta: Basic = d2(&s2(&u, &d)?, &int(2))?;
        if numeric_ {
            let (alpha_, beta_) = (real_dummy("alpha"), real_dummy("beta"));
            facts.push(rel(RelOp::Eq, &alpha_, &alpha)?);
            facts.push(rel(RelOp::Eq, &beta_, &beta)?);
            alpha = alpha_;
            beta = beta_;
        }
        big_x = d2(&a2(&beta, &m2(&alpha, t)?)?, &a2(&int(1), t)?)?;
        let at = |point: &Basic, pp: &Basic, qq: &Basic| a2(&pwi(&s2(point, pp)?, 2)?, qq);
        let a1: Basic = at(&alpha, p1, q1)?;
        let a2_: Basic = at(&alpha, p2, q2)?;
        let b1: Basic = at(&beta, p1, q1)?;
        let b2: Basic = at(&beta, p2, q2)?;
        big_a = vec![named(&a1, &mut facts, numeric_)?, named(&a2_, &mut facts, numeric_)?];
        big_b = vec![named(&b1, &mut facts, numeric_)?, named(&b2, &mut facts, numeric_)?];
        preimage_ = d2(&s2(x, &beta)?, &s2(&alpha, x)?)?;
        xp = d2(&s2(&alpha, &beta)?, &pwi(&a2(&int(1), t)?, 2)?)?;
        pole = Some(alpha);
        infinite = int(-1);
        bilinear = true;
    }
    let assumptions: Vec<Basic> = with_facts(assumptions, &facts);
    let order = |aa: &[Basic], bb: &[Basic]| -> PolyResult<Option<bool>> {
        ask_rel(RelOp::Lt, &m2(&aa[1], &bb[0])?, &m2(&aa[0], &bb[1])?, &assumptions)
    };
    let Some(ordered) = order(&big_a, &big_b)? else {
        return Ok(None);
    };
    if !ordered {
        big_a.reverse();
        big_b.reverse();
        if order(&big_a, &big_b)? != Some(true) {
            return Ok(None);
        }
    }
    let lam_: Basic = radical(&cancel(&d2(&big_b[0], &big_a[0])?)?, &mut radicals, &mut facts)?;
    let m: Basic = cancel(&s2(
        &int(1),
        &d2(&m2(&big_a[1], &big_b[0])?, &m2(&big_a[0], &big_b[1])?)?,
    )?)?;
    Ok(Some(TangentMap {
        x: big_x,
        xp,
        preimage: preimage_,
        pole,
        infinite,
        lam_,
        m,
        b1b2: m2(&big_b[0], &big_b[1])?,
        bilinear,
        radicals,
        facts,
    }))
}

fn two_pairs_reduction(
    r: &Basic,
    p: &Basic,
    e: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    let Some(factors) = two_pairs(p, x, assumptions)? else {
        return Ok(None);
    };
    let assumptions: Vec<Basic> = with_facts(assumptions, &factors.facts);
    if ask_rel(RelOp::Gt, &factors.lam, &int(0), &assumptions)? != Some(true) {
        return Ok(None);
    }
    if ask_rel(RelOp::Lt, a, b, &assumptions)? != Some(true) {
        return Ok(None);
    }
    let t: Basic = fresh("t", &[]);
    let Some(mut mapping) = tangent_map(&factors.pairs, &t, x, &assumptions, !factors.facts.is_empty())? else {
        return Ok(None);
    };
    let assumptions: Vec<Basic> = with_facts(&assumptions, &mapping.facts);
    let mut pieces: Vec<(Basic, Basic)> = vec![(a.clone(), b.clone())];
    let pole: Option<Basic> = mapping.pole.clone();
    if let Some(pole) = &pole {
        let Some(inside) = position_of(pole, a, b, &assumptions)? else {
            return Ok(None);
        };
        if inside {
            pieces = vec![(a.clone(), pole.clone()), (pole.clone(), b.clone())];
        }
    }
    let (lam_, m) = (mapping.lam_.clone(), mapping.m.clone());
    let product: Basic = cancel(&m2(&factors.lam, &mapping.b1b2)?)?;
    let root: Basic = radical(&product, &mut mapping.radicals, &mut mapping.facts)?;
    let assumptions: Vec<Basic> = with_facts(&assumptions, &mapping.facts);
    let mut g: Basic = m2(&subs(r, x, &mapping.x)?, &mapping.xp)?;
    if mapping.bilinear {
        g = m2(&g, &pw(&pwi(&a2(&int(1), &t)?, 2)?, &m2(&int(-2), e)?)?)?;
    }
    let g: Basic = cancel(&g)?;
    if !is_rational_function(&g, &t) {
        return Ok(None);
    }
    let w: Basic = positive_dummy("w");
    let Some((even, odd)) = parity_parts(&g, &t, &w)? else {
        return Ok(None);
    };
    let s: Basic = positive_dummy("s");
    let one_minus_s: Basic = s2(&int(1), &s)?;
    let value_w: Basic = d2(&m2(&pwi(&lam_, 2)?, &s)?, &one_minus_s)?;
    let g_e: Basic = subs(&even, &w, &value_w)?;
    let g_o: Basic = subs(&odd, &w, &value_w)?;
    let (q, h) = if *e == neg(&half())? {
        let c0: Basic = d2(&lam_, &root)?;
        (
            cancel(&d2(&m2(&c0, &g_e)?, &int(2))?)?,
            cancel(&d2(
                &chain(&[c0.clone(), lam_.clone(), g_o.clone()])?,
                &m2(&int(2), &one_minus_s)?,
            )?)?,
        )
    } else {
        let c0: Basic = m2(&lam_, &root)?;
        let one_minus_ms: Basic = s2(&int(1), &m2(&m, &s)?)?;
        (
            cancel(&d2(
                &chain(&[c0.clone(), g_e.clone(), one_minus_ms.clone()])?,
                &m2(&int(2), &pwi(&one_minus_s, 2)?)?,
            )?)?,
            cancel(&d2(
                &chain(&[c0.clone(), lam_.clone(), g_o.clone(), one_minus_ms])?,
                &m2(&int(2), &pwi(&one_minus_s, 3)?)?,
            )?)?,
        )
    };
    let parts_ = TangentParts {
        q,
        h,
        s,
        m,
        lam_,
        assumptions: assumptions.clone(),
    };
    let mut total: Basic = int(0);
    for (lo, hi) in pieces {
        let t0: Basic = if Some(&lo) == pole.as_ref() {
            neg(&oo())?
        } else {
            mapping.preimage(&lo, x)?
        };
        let t1: Basic = if Some(&hi) == pole.as_ref() {
            oo()
        } else {
            mapping.preimage(&hi, x)?
        };
        let Some(value) = parts_.between(&t0, &t1)? else {
            return Ok(None);
        };
        total = a2(&total, &value)?;
    }
    let mut all_facts: Vec<Basic> = factors.facts.clone();
    all_facts.extend(mapping.facts.iter().cloned());
    Ok(restore(&reveal(&total)?, &mapping.radicals, &all_facts)?.map(plain_value))
}

fn position_of(point: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    let below: Option<bool> = if is_neg_oo(a) {
        Some(true)
    } else {
        let mut below = ask_rel(RelOp::Lt, a, point, assumptions)?;
        if below.is_none() && ask_rel(RelOp::Ge, a, point, assumptions)? == Some(true) {
            below = Some(false);
        }
        below
    };
    let above: Option<bool> = if is_oo(b) {
        Some(true)
    } else {
        let mut above = ask_rel(RelOp::Lt, point, b, assumptions)?;
        if above.is_none() && ask_rel(RelOp::Ge, point, b, assumptions)? == Some(true) {
            above = Some(false);
        }
        above
    };
    match (below, above) {
        (Some(x), Some(y)) => Ok(Some(x && y)),
        _ => Ok(None),
    }
}

/// The parts of the integrand after the tangent map (`_TangentParts`).
struct TangentParts {
    q: Basic,
    h: Basic,
    s: Basic,
    m: Basic,
    lam_: Basic,
    assumptions: Vec<Basic>,
}

impl TangentParts {
    fn between(&self, t0: &Basic, t1: &Basic) -> OptR {
        let (Some(first_sign), Some(second_sign)) = (self.sign(t0)?, self.sign(t1)?) else {
            return Ok(None);
        };
        if first_sign == 0 || second_sign == 0 || first_sign == second_sign {
            let sigma: i64 = if first_sign == 1 || second_sign == 1 { 1 } else { -1 };
            return self.piece(&m2(&int(sigma), t0)?, &m2(&int(sigma), t1)?, &int(sigma));
        }
        let first = self.piece(&m2(&int(first_sign), t0)?, &int(0), &int(first_sign))?;
        let second = self.piece(&int(0), &m2(&int(second_sign), t1)?, &int(second_sign))?;
        match (first, second) {
            (Some(a), Some(b)) => Ok(Some(a2(&a, &b)?)),
            _ => Ok(None),
        }
    }

    fn sign(&self, tau: &Basic) -> PolyResult<Option<i64>> {
        if *tau == int(0) {
            return Ok(Some(0));
        }
        if is_oo(tau) {
            return Ok(Some(1));
        }
        if is_neg_oo(tau) {
            return Ok(Some(-1));
        }
        if ask_rel(RelOp::Gt, tau, &int(0), &self.assumptions)? == Some(true) {
            return Ok(Some(1));
        }
        if ask_rel(RelOp::Lt, tau, &int(0), &self.assumptions)? == Some(true) {
            return Ok(Some(-1));
        }
        Ok(None)
    }

    fn s_of(&self, u: &Basic) -> R {
        if is_oo(u) {
            return Ok(int(1));
        }
        let u2: Basic = pwi(u, 2)?;
        cancel(&d2(&u2, &a2(&pwi(&self.lam_, 2)?, &u2)?)?)
    }

    fn piece(&self, u0: &Basic, u1: &Basic, sigma: &Basic) -> OptR {
        let (s0, s1) = (self.s_of(u0)?, self.s_of(u1)?);
        if s0 == s1 {
            return Ok(Some(int(0)));
        }
        let Some(increasing) = ask_rel(RelOp::Lt, &s0, &s1, &self.assumptions)? else {
            return Ok(None);
        };
        let (lower, upper) = if increasing { (&s0, &s1) } else { (&s1, &s0) };
        let mut total: Basic = int(0);
        if self.q != int(0) {
            let legendre: Radical = Radical::Legendre(self.m.clone());
            let at_s0 = from_zero(&self.q, &self.s, &s0, &legendre, &self.assumptions, lower, upper)?;
            let at_s1 = from_zero(&self.q, &self.s, &s1, &legendre, &self.assumptions, lower, upper)?;
            let (Some(at_s0), Some(at_s1)) = (at_s0, at_s1) else {
                return Ok(None);
            };
            total = a2(&total, &m2(sigma, &s2(&at_s1, &at_s0)?)?)?;
        }
        if self.h != int(0) {
            if !poles_outside(&self.h, &self.s, lower, upper, &self.assumptions)? {
                return Ok(None);
            }
            let Some(big_f) = odd_antiderivative(&self.h, &self.s, &self.m)? else {
                return Ok(None);
            };
            total = s2(&a2(&total, &subs(&big_f, &self.s, &s1)?)?, &subs(&big_f, &self.s, &s0)?)?;
        }
        Ok(Some(total))
    }
}

fn poles_outside(h: &Basic, s: &Basic, lower: &Basic, upper: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    let (_, denominator) = numer_denom(&cancel(h)?)?;
    if !denominator.has(s) {
        return Ok(true);
    }
    let Some((_, factors)) = factor_list_(&denominator, s)? else {
        return Ok(false);
    };
    for (factor, _) in factors {
        if !factor.has(s) {
            continue;
        }
        let Some(d) = degree(&factor, s)? else { return Ok(false) };
        let candidates = if d <= 2 {
            low_degree_roots(&factor, s)?
        } else {
            real_roots_rational(&factor, s)?
        };
        let Some(candidates) = candidates else { return Ok(false) };
        for p in candidates {
            let real: Option<bool> = is(&p, Predicate::extended_real);
            if real == Some(false) {
                continue;
            }
            if real.is_none()
                && ask_rel(
                    RelOp::Eq,
                    &crate::functions::elementary::complexes::im(&p)?,
                    &int(0),
                    assumptions,
                )? == Some(false)
            {
                continue;
            }
            if ask_rel(RelOp::Gt, &m2(&s2(&p, lower)?, &s2(&p, upper)?)?, &int(0), assumptions)? != Some(true) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn odd_antiderivative(h: &Basic, s: &Basic, m: &Basic) -> OptR {
    let v: Basic = positive_dummy("v");
    let at_v: Basic = d2(&s2(&int(1), &pwi(&v, 2)?)?, m)?;
    let rational: Basic = cancel(&d2(&m2(&int(-2), &subs(h, s, &at_v)?)?, m)?)?;
    if !is_rational_function(&rational, &v) {
        return Ok(None);
    }
    let symbol: Basic = crate::core::symbol::Symbol::new("t", &[]).0;
    let found = crate::timeout::attempt_within(budget(), || {
        let integrand = crate::integrals::risch::rationaltools::Integrand::of(&rational)?;
        crate::integrals::risch::rationaltools::ratint(&integrand, &v, &symbol, None)
    })?;
    let Some(big_f) = found else { return Ok(None) };
    Ok(Some(subs(&big_f, &v, &sqrt(&s2(&int(1), &m2(m, s)?)?)?)?))
}

fn at_arcsine(f: &Basic, x: &Basic, s: &Basic) -> R {
    let (root, co_root) = (sqrt(s)?, sqrt(&s2(&int(1), s)?)?);
    let node = |g: Function| apply(g, std::slice::from_ref(x));
    let rule: Vec<(Basic, Basic)> = vec![
        (node(Function::Sin)?, root.clone()),
        (node(Function::Cos)?, co_root.clone()),
        (node(Function::Tan)?, d2(&root, &co_root)?),
        (node(Function::Cot)?, d2(&co_root, &root)?),
        (node(Function::Sec)?, d2(&int(1), &co_root)?),
        (node(Function::Csc)?, d2(&int(1), &root)?),
    ];
    subs(&replaced(f, &rule)?, x, &asin(&root)?)
}

fn trigonometric(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    let half_pi: Basic = d2(&pi(), &int(2))?;
    if *a != int(0)
        || ask_rel(RelOp::Gt, b, &int(0), assumptions)? != Some(true)
        || ask_rel(RelOp::Le, b, &half_pi, assumptions)? != Some(true)
    {
        return Ok(None);
    }
    let s: Basic = positive_dummy("s");
    let h: Basic = d2(&at_arcsine(f, x, &s)?, &int(2))?;
    let mut radical_: Option<(Basic, Basic)> = None;
    let mut seen: Vec<Basic> = Vec::new();
    for node in h.preorder() {
        if !node.is_pow() || seen.contains(&node) {
            continue;
        }
        seen.push(node.clone());
        let exponent: &Basic = &node.args()[1];
        let is_half: bool = exponent
            .as_rational()
            .is_some_and(|r| *r == crate::numbers::rat(1, 2) || *r == crate::numbers::rat(-1, 2));
        if is_half && node.has(&s) {
            let base: &Basic = &node.args()[0];
            if radical_.is_some() || !is_polynomial(base, &s) || degree(base, &s)? != Some(1) {
                return Ok(None);
            }
            radical_ = Some((base.clone(), exponent.clone()));
        }
    }
    let Some((radical_, exponent)) = radical_ else {
        return Ok(None);
    };
    let alpha: Basic = subs(&radical_, &s, &int(0))?;
    if ask_rel(RelOp::Gt, &alpha, &int(0), assumptions)? != Some(true) {
        return Ok(None);
    }
    let m: Basic = cancel(&d2(&neg(&s2(&radical_, &alpha)?)?, &m2(&alpha, &s)?)?)?;
    if m.has(&s) {
        return Ok(None);
    }
    let mut q: Basic = cancel(&d2(&h, &pw(&radical_, &exponent)?)?)?;
    if !is_rational_function(&q, &s) {
        return Ok(None);
    }
    q = if exponent == neg(&half())? {
        d2(&q, &sqrt(&alpha)?)?
    } else {
        chain(&[q.clone(), sqrt(&alpha)?, s2(&int(1), &m2(&m, &s)?)?])?
    };
    let upper: Basic = if *b == half_pi {
        int(1)
    } else {
        pwi(&apply(Function::Sin, std::slice::from_ref(b))?, 2)?
    };
    let found = legendre_reduction(
        &Reduction {
            q,
            m,
            s: s.clone(),
            lower: int(0),
            upper,
        },
        assumptions,
    )?;
    Ok(found.map(plain_value))
}
