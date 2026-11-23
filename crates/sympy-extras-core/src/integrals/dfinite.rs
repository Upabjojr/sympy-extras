//! Port of `sympy_extras/integrals/dfinite.py`: Chyzak's algorithm with
//! Koutschan's ansatz for integrands which are products of D-finite
//! factors (Bessel, Airy, error functions, exponentials, trigonometric
//! functions, logarithms and the functions whose derivative is one), the
//! annihilators of the integrand, and the differential equation of the
//! integral solved as in `telescoping`.

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::basic::{Basic, Function, Head};
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;
use crate::integrals::telescoping::{equation, polynomial_failure, solve, verified, Context, Element, KPoly};
use crate::orderedmap::OrderedMap;
use crate::polys::domains::domain::Domain;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn diff(e: &Basic, v: &Basic, n: u64) -> R {
    if n == 0 {
        return Ok(e.clone());
    }
    crate::core::function::diff(e, &[(v.clone(), n)])
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn binomial(n: usize, k: usize) -> i64 {
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }
    result
}
fn is_rational_function(e: &Basic, symbols: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, symbols, crate::core::expr::PolynomialKind::Rational) == Some(true)
}
fn is_polynomial(e: &Basic, symbols: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, symbols, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
}
fn is_application(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(_) | Head::Undefined(_))
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}

static NAMES: AtomicUsize = AtomicUsize::new(0);

/// A dummy with a name of its own (`_fresh`).
fn fresh(prefix: &str) -> Basic {
    let n: usize = NAMES.fetch_add(1, Ordering::SeqCst);
    crate::core::symbol::Dummy::new(&format!("_{}{}", prefix, n), &[]).0
}

/// A mixed derivative `d^i/dx^i d^j/dt^j F` by its orders (`Derivative_`).
pub type Orders = (usize, usize);
/// A combination of the basis derivatives (`NormalForm`).
pub type NormalForm = OrderedMap<Orders, Basic>;
/// A basis element: the orders of the derivatives of the factors (`Basis`).
type Basis = Vec<usize>;
/// An element of the closure (`Vector`).
type Vector = OrderedMap<Basis, Basic>;

/// A linear differential operator `sum_i c_i d^i/dv^i` in `v`
/// (`Annihilator`).
#[derive(Clone, Debug)]
pub struct Annihilator {
    pub variable: Basic,
    pub coefficients: Vec<Basic>,
}

impl Annihilator {
    pub fn new(variable: &Basic, coefficients: &[Basic]) -> PolyResult<Annihilator> {
        let mut stripped: Vec<Basic> = coefficients.to_vec();
        while stripped.last().is_some_and(|c| *c == Basic::zero()) {
            stripped.pop();
        }
        if stripped.len() < 2 {
            return Err(PolyError::Value("an annihilator has order at least one".to_string()));
        }
        Ok(Annihilator {
            variable: variable.clone(),
            coefficients: stripped,
        })
    }

    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub fn leading(&self) -> &Basic {
        self.coefficients.last().expect("nonempty")
    }

    /// The `e_i` with `d^r F = sum_{i < r} e_i d^i F` (`reduction`).
    pub fn reduction(&self) -> PolyResult<Vec<Basic>> {
        let n: usize = self.coefficients.len() - 1;
        self.coefficients[..n]
            .iter()
            .map(|c| cancel(&neg(&div(c, self.leading())?)?))
            .collect()
    }

    /// `sum_i c_i d^i f/dv^i` (`apply`).
    pub fn apply(&self, f: &Basic) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[c.clone(), diff(f, &self.variable, i as u64)?])?);
        }
        add(&terms)
    }
}

/// The output of Chyzak's algorithm (`DFiniteTelescoper`).
#[derive(Clone, Debug)]
pub struct DFiniteTelescoper {
    pub term: Basic,
    pub x: Basic,
    pub t: Basic,
    pub coefficients: Vec<Basic>,
    pub certificate: Vec<(Basic, Basic)>,
}

impl DFiniteTelescoper {
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// `sum_k a_k(t) I^(k)(t)` (`operator`).
    pub fn operator(&self, name: &str) -> R {
        let f: Basic = std::sync::Arc::new(crate::core::basic::UndefinedFunction::new(name, None, &[]))
            .apply(vec![self.t.clone()])?;
        let mut terms: Vec<Basic> = Vec::new();
        for (k, a) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[a.clone(), diff(&f, &self.t, k as u64)?])?);
        }
        add(&terms)
    }

    /// The certificate `G` as an expression (`function`).
    pub fn function(&self) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (b, g) in &self.certificate {
            terms.push(mul(&[g.clone(), b.clone()])?);
        }
        add(&terms)
    }

    /// Verify `P F = dG/dx` (`check`).
    pub fn check(&self) -> PolyResult<bool> {
        let mut terms: Vec<Basic> = Vec::new();
        for (k, a) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[a.clone(), diff(&self.term, &self.t, k as u64)?])?);
        }
        let difference: Basic = sub(&add(&terms)?, &diff(&self.function()?, &self.x, 1)?)?;
        let options = crate::simplify::simplify::Options::default();
        Ok(crate::simplify::simplify::simplify(&crate::core::function::doit(&difference)?, &options)? == Basic::zero())
    }
}

// ---------------------------------------------------------------------------
// Annihilators

fn vector_add(total: &mut Vector, key: &Basis, value: &Basic) -> PolyResult<()> {
    let current: Basic = total.get(key).cloned().unwrap_or_else(Basic::zero);
    total.insert(key.clone(), add(&[current, value.clone()])?);
    Ok(())
}

/// The annihilator of least order of `F` in `v` within the closure of its
/// terms (`_minimal_annihilator`).
fn minimal_annihilator(f: &Basic, v: &Basic, parameter: &Basic, depth: usize) -> PolyResult<Option<Annihilator>> {
    if *f == Basic::zero() {
        return Ok(None);
    }
    let terms: Vec<Basic> = if f.is_add() { f.args().to_vec() } else { vec![f.clone()] };
    let mut spaces: Vec<Vec<Factor>> = Vec::new();
    let mut current: Vec<Vector> = Vec::new();
    let mut keys: Vec<Vec<String>> = Vec::new();
    let mut dimension: usize = 0;
    for term in &terms {
        let Some((coefficient, mut factors)) = closure_of(term, v, parameter, true, depth)? else {
            return Ok(None);
        };
        factors.sort_by_key(|f| f.repr());
        let zero: Basis = vec![0; factors.len()];
        // the terms with the same factors share their space
        let key: Vec<String> = factors.iter().map(|f| f.repr()).collect();
        if let Some(position) = keys.iter().position(|k| *k == key) {
            vector_add(&mut current[position], &zero, &coefficient)?;
            continue;
        }
        keys.push(key);
        let size: usize = factors.iter().map(|f| f.order()).product();
        spaces.push(factors);
        let mut vector: Vector = OrderedMap::new();
        vector.insert(zero, coefficient);
        current.push(vector);
        dimension += size;
    }
    let mut columns: Vec<Vec<Vector>> = vec![current];
    for _ in 0..dimension {
        let last: &Vec<Vector> = columns.last().expect("nonempty");
        let next: Vec<Vector> = spaces
            .iter()
            .zip(last)
            .map(|(factors, vector)| derive(vector, v, factors))
            .collect::<PolyResult<_>>()?;
        columns.push(next);
        if let Some(relation) = relation(&columns, v)? {
            return match Annihilator::new(v, &relation) {
                Ok(a) => Ok(Some(a)),
                Err(PolyError::Value(_)) => Ok(None),
                Err(e) => Err(e),
            };
        }
    }
    Ok(None)
}

/// The coefficients of a linear dependency of the derivatives
/// (`_relation`).
fn relation(columns: &[Vec<Vector>], v: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let mut keys: Vec<(usize, Basis)> = Vec::new();
    for column in columns {
        for (s, vector) in column.iter().enumerate() {
            for basis in vector.keys() {
                let key = (s, basis.clone());
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }
    }
    let entries: Vec<Basic> = columns
        .iter()
        .flatten()
        .flat_map(|vector| vector.values().cloned())
        .collect();
    let width: usize = columns.len();
    let found = (|| -> PolyResult<(Context, Vec<Vec<Element>>)> {
        let k: Context = Context::new(&entries, &fresh("z"), &[], false)?;
        let mut null = Vec::new();
        let rows: Vec<Vec<Element>> = keys
            .iter()
            .map(|(s, basis)| {
                (0..width)
                    .map(|c| k.element(&columns[c][*s].get(basis).cloned().unwrap_or_else(Basic::zero)))
                    .collect::<PolyResult<Vec<Element>>>()
            })
            .collect::<PolyResult<_>>()?;
        null.extend(k.field.nullspace(&rows, width)?);
        Ok((k, null))
    })();
    let (k, null) = match found {
        Ok(v) => v,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some(first) = null.first() else {
        return Ok(None);
    };
    let values: Vec<Basic> = first.iter().map(|c| k.element_expr(c)).collect::<PolyResult<_>>()?;
    let (coefficients, _) = normalized(&values, v)?;
    if *coefficients.last().expect("nonempty") == Basic::zero() {
        return Ok(None);
    }
    Ok(Some(coefficients))
}

/// The annihilators `L_x` and `L_t` of `F` of least order in the closure
/// of its terms (`annihilators`).
pub fn annihilators(f: &Basic, x: &Basic, t: &Basic) -> PolyResult<Option<(Annihilator, Annihilator)>> {
    let Some(l_x) = minimal_annihilator(f, x, t, 0)? else {
        return Ok(None);
    };
    let Some(l_t) = minimal_annihilator(f, t, x, 0)? else {
        return Ok(None);
    };
    Ok(Some((l_x, l_t)))
}

// ---------------------------------------------------------------------------
// Normal forms modulo the two annihilators

fn form_add(total: &mut NormalForm, key: Orders, value: &Basic) -> PolyResult<()> {
    let current: Basic = total.get(&key).cloned().unwrap_or_else(Basic::zero);
    total.insert(key, add(&[current, value.clone()])?);
    Ok(())
}

/// The reduction `(i, j) -> d^i/dx^i d^j/dt^j F` as a combination of the
/// basis derivatives, memoised (`normal_form`).
pub struct NormalForms {
    x: Basic,
    t: Basic,
    r: usize,
    s: usize,
    e_x: Vec<Basic>,
    e_t: Vec<Basic>,
    cache: RefCell<HashMap<Orders, NormalForm>>,
}

/// The normal forms modulo the two annihilators (`normal_form`).
pub fn normal_form(l_x: &Annihilator, l_t: &Annihilator, x: &Basic, t: &Basic) -> PolyResult<NormalForms> {
    Ok(NormalForms {
        x: x.clone(),
        t: t.clone(),
        r: l_x.order(),
        s: l_t.order(),
        e_x: l_x.reduction()?,
        e_t: l_t.reduction()?,
        cache: RefCell::new(HashMap::new()),
    })
}

impl NormalForms {
    /// `d^i_x d^m_t (coefficient * d^base F)` in normal form (`_leibniz`).
    fn leibniz(&self, coefficient: &Basic, i: usize, m: usize, base: Orders) -> PolyResult<NormalForm> {
        let mut result: NormalForm = OrderedMap::new();
        for a in 0..=i {
            for b in 0..=m {
                let binomials: Basic = int(binomial(i, a) * binomial(m, b));
                let factor: Basic = if i - a != 0 || m - b != 0 {
                    let d: Basic = diff(&diff(coefficient, &self.x, (i - a) as u64)?, &self.t, (m - b) as u64)?;
                    mul(&[binomials, d])?
                } else {
                    mul(&[binomials, coefficient.clone()])?
                };
                if factor == Basic::zero() {
                    continue;
                }
                let inner: NormalForm = self.reduce(base.0 + a, base.1 + b)?;
                for (key, value) in inner.iter() {
                    form_add(&mut result, *key, &mul(&[factor.clone(), value.clone()])?)?;
                }
            }
        }
        Ok(result)
    }

    /// `d^i/dx^i d^j/dt^j F` reduced (`reduce`).
    pub fn reduce(&self, i: usize, j: usize) -> PolyResult<NormalForm> {
        if let Some(found) = self.cache.borrow().get(&(i, j)) {
            return Ok(found.clone());
        }
        let mut result: NormalForm = OrderedMap::new();
        if i < self.r && j < self.s {
            result.insert((i, j), Basic::one());
            self.cache.borrow_mut().insert((i, j), result.clone());
            return Ok(result);
        }
        if j >= self.s {
            // d^i_x d^j_t F = d^i_x d^(j-s)_t sum_k e_k d^k_t F
            for (k, e) in self.e_t.iter().enumerate() {
                if *e == Basic::zero() {
                    continue;
                }
                for (key, value) in self.leibniz(e, i, j - self.s, (0, k))?.iter() {
                    form_add(&mut result, *key, value)?;
                }
            }
        } else {
            // i >= r: d^i_x d^j_t F = d^(i-r)_x d^j_t sum_k e_k d^k_x F
            for (k, e) in self.e_x.iter().enumerate() {
                if *e == Basic::zero() {
                    continue;
                }
                for (key, value) in self.leibniz(e, i - self.r, j, (k, 0))?.iter() {
                    form_add(&mut result, *key, value)?;
                }
            }
        }
        let mut cancelled: NormalForm = OrderedMap::new();
        for (key, value) in result.iter() {
            let c: Basic = cancel(value)?;
            if c != Basic::zero() {
                cancelled.insert(*key, c);
            }
        }
        self.cache.borrow_mut().insert((i, j), cancelled.clone());
        Ok(cancelled)
    }
}

// ---------------------------------------------------------------------------
// The closure of the derivatives of a product of D-finite factors

/// A D-finite factor `f(u)` of the integrand at `u = argument` (`Factor`).
#[derive(Clone, Debug)]
pub struct Factor {
    pub function: Basic,
    pub u: Basic,
    pub argument: Basic,
    pub reduction: Vec<Basic>,
}

impl Factor {
    pub fn order(&self) -> usize {
        self.reduction.len()
    }

    /// `f^(m)(argument)` (`derivative`).
    pub fn derivative(&self, m: usize) -> R {
        subs(&diff(&self.function, &self.u, m as u64)?, &self.u, &self.argument)
    }

    /// `Factor(f(argument), order)` (`__repr__`).
    pub fn repr(&self) -> String {
        let at: String = match subs(&self.function, &self.u, &self.argument) {
            Ok(e) => e.to_string(),
            Err(_) => self.function.to_string(),
        };
        format!("Factor({}, {})", at, self.order())
    }
}

/// The coefficients of the annihilator of `function` in `u`, written down
/// for the common factors (`_table`).
fn table(function: &Basic, u: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    if !is_application(function) || function.args().last() != Some(u) {
        return Ok(None);
    }
    let Head::Function(f) = function.head() else {
        return Ok(None);
    };
    let u2: Basic = pow(u, &int(2))?;
    Ok(Some(match f {
        Function::Exp => vec![int(-1), int(1)],
        Function::Sin | Function::Cos => vec![int(1), int(0), int(1)],
        Function::Sinh | Function::Cosh => vec![int(-1), int(0), int(1)],
        Function::Erf | Function::Erfc => vec![int(0), mul(&[int(2), u.clone()])?, int(1)],
        Function::Erfi => vec![int(0), mul(&[int(-2), u.clone()])?, int(1)],
        Function::Log if function.args().len() == 1 => vec![int(0), int(1), u.clone()],
        Function::AiryAi | Function::AiryBi => vec![neg(u)?, int(0), int(1)],
        // w = Ai': w' = u Ai, w'' = Ai + u w = w'/u + u w
        Function::AiryAiPrime | Function::AiryBiPrime => vec![neg(&u2)?, int(-1), u.clone()],
        Function::BesselJ | Function::BesselY | Function::Hankel1 | Function::Hankel2 => {
            let nu2: Basic = pow(&function.args()[0], &int(2))?;
            vec![sub(&u2, &nu2)?, u.clone(), u2]
        }
        Function::BesselI | Function::BesselK => {
            let nu2: Basic = pow(&function.args()[0], &int(2))?;
            vec![neg(&add(&[u2.clone(), nu2])?)?, u.clone(), u2]
        }
        _ => return Ok(None),
    }))
}

/// The depth of the derivatives through which the annihilator of a factor
/// outside the table is sought (`_MAX_DEPTH`).
const MAX_DEPTH: usize = 3;

/// The annihilator of `function` as `L o d/du` (`_derivative_annihilator`).
fn derivative_annihilator(function: &Basic, u: &Basic, depth: usize) -> PolyResult<Option<Vec<Basic>>> {
    if depth >= MAX_DEPTH {
        return Ok(None);
    }
    let derivative: Basic = diff(function, u, 1)?;
    if derivative
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Derivative))
        || derivative.atoms(Some(&is_application)).contains(function)
    {
        return Ok(None);
    }
    let Some(inner) = minimal_annihilator(&derivative, u, &fresh("w"), depth + 1)? else {
        return Ok(None);
    };
    let mut coefficients: Vec<Basic> = vec![Basic::zero()];
    coefficients.extend(inner.coefficients);
    Ok(Some(coefficients))
}

/// `f` as a D-finite function of a polynomial argument (rational with
/// `rational`) (`_dfinite_factor`).
fn dfinite_factor(f: &Basic, x: &Basic, t: &Basic, rational: bool, depth: usize) -> PolyResult<Option<Factor>> {
    let xt: [Basic; 2] = [x.clone(), t.clone()];
    if rational && f.is_pow() && !f.args()[1].has(x) && !f.args()[1].has(t) && is_rational_function(&f.args()[0], &xt) {
        let w: Basic = fresh("u");
        let exponent: Basic = f.args()[1].clone();
        return Ok(Some(Factor {
            function: pow(&w, &exponent)?,
            u: w.clone(),
            argument: f.args()[0].clone(),
            reduction: vec![div(&exponent, &w)?],
        }));
    }
    if !is_application(f) || f.args().is_empty() {
        return Ok(None);
    }
    let arguments: Vec<Basic> = f.args().to_vec();
    if arguments.iter().any(crate::core::logic::is_boolean) {
        return Ok(None);
    }
    let argument: Basic = arguments.last().expect("nonempty").clone();
    if arguments[..arguments.len() - 1].iter().any(|a| a.has(x) || a.has(t)) {
        return Ok(None);
    }
    if rational {
        if !is_rational_function(&argument, &xt) {
            return Ok(None);
        }
    } else if !is_polynomial(&argument, &xt) {
        return Ok(None);
    }
    let u: Basic = fresh("u");
    let mut new_args: Vec<Basic> = arguments[..arguments.len() - 1].to_vec();
    new_args.push(u.clone());
    let function: Basic = crate::core::function::rebuild_evaluated(f, new_args)?;
    let mut coefficients: Option<Vec<Basic>> = table(&function, &u)?;
    if coefficients.is_none() {
        coefficients = derivative_annihilator(&function, &u, depth)?;
    }
    let Some(coefficients) = coefficients.filter(|c| c.len() >= 2 && *c.last().expect("nonempty") != Basic::zero())
    else {
        return Ok(None);
    };
    let leading: &Basic = coefficients.last().expect("nonempty");
    let reduction = coefficients[..coefficients.len() - 1]
        .iter()
        .map(|c| cancel(&neg(&div(c, leading)?)?))
        .collect::<PolyResult<Vec<Basic>>>();
    let reduction: Vec<Basic> = match reduction {
        Ok(r) => r,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    Ok(Some(Factor {
        function,
        u,
        argument,
        reduction,
    }))
}

/// The integrand as a rational coefficient times D-finite factors
/// (`closure`).
pub fn closure(f: &Basic, x: &Basic, t: &Basic) -> PolyResult<Option<(Basic, Vec<Factor>)>> {
    closure_of(f, x, t, false, 0)
}

fn closure_of(
    f: &Basic,
    x: &Basic,
    t: &Basic,
    rational: bool,
    depth: usize,
) -> PolyResult<Option<(Basic, Vec<Factor>)>> {
    let xt: [Basic; 2] = [x.clone(), t.clone()];
    let mut coefficient: Basic = Basic::one();
    let mut factors: Vec<Factor> = Vec::new();
    let parts: Vec<Basic> = if f.is_mul() { f.args().to_vec() } else { vec![f.clone()] };
    for part in parts {
        let mut e: Basic = part.clone();
        if is_rational_function(&e, &xt) {
            coefficient = mul(&[coefficient, e])?;
            continue;
        }
        let mut multiplicity: usize = 1;
        if e.is_pow() {
            if let Some(k) = e.args()[1]
                .as_rational()
                .filter(|r| r.is_int() && *r > &crate::numbers::Rat::ZERO)
            {
                multiplicity = usize::try_from(&crate::numbers::numer(k)).unwrap_or(1);
                e = e.args()[0].clone();
            }
        }
        let Some(factor) = dfinite_factor(&e, x, t, rational, depth)? else {
            let expanded: Basic = crate::functions::expand_func::expand_func(&e, true)?;
            if expanded != e {
                let Some((inner_coefficient, inner_factors)) = closure_of(&expanded, x, t, rational, depth)? else {
                    return Ok(None);
                };
                coefficient = mul(&[coefficient, pow(&inner_coefficient, &int(multiplicity as i64))?])?;
                for _ in 0..multiplicity {
                    factors.extend(inner_factors.iter().cloned());
                }
                continue;
            }
            return Ok(None);
        };
        for _ in 0..multiplicity {
            factors.push(factor.clone());
        }
    }
    Ok(Some((coefficient, factors)))
}

fn cancelled(vector: &Vector) -> PolyResult<Vector> {
    let mut result: Vector = OrderedMap::new();
    for (basis, value) in vector.iter() {
        let c: Basic = cancel(value)?;
        if c != Basic::zero() {
            result.insert(basis.clone(), c);
        }
    }
    Ok(result)
}

/// The derivative of an element of the closure with respect to `var`
/// (`_derive`).
fn derive(vector: &Vector, var: &Basic, factors: &[Factor]) -> PolyResult<Vector> {
    let mut result: Vector = OrderedMap::new();
    for (basis, coefficient) in vector.iter() {
        let derived: Basic = diff(coefficient, var, 1)?;
        if derived != Basic::zero() {
            vector_add(&mut result, basis, &derived)?;
        }
        for (k, factor) in factors.iter().enumerate() {
            let chain: Basic = diff(&factor.argument, var, 1)?;
            if chain == Basic::zero() {
                continue;
            }
            let m: usize = basis[k] + 1;
            if m < factor.order() {
                let mut key: Basis = basis.clone();
                key[k] = m;
                vector_add(&mut result, &key, &mul(&[coefficient.clone(), chain])?)?;
            } else {
                // f^(n)(argument) reduced by the differential equation of f
                for (m_, e) in factor.reduction.iter().enumerate() {
                    if *e == Basic::zero() {
                        continue;
                    }
                    let mut key: Basis = basis.clone();
                    key[k] = m_;
                    let at: Basic = subs(e, &factor.u, &factor.argument)?;
                    vector_add(&mut result, &key, &mul(&[coefficient.clone(), chain.clone(), at])?)?;
                }
            }
        }
    }
    cancelled(&result)
}

/// The product of the irreducible factors in `x` of the denominators of
/// the coefficients (`_denominators`).
fn denominators(vectors: &[&Vector], x: &Basic) -> R {
    let mut factors: Vec<Basic> = Vec::new();
    for vector in vectors {
        for value in vector.values() {
            let denominator: Basic =
                crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(value, false, true)?)?.1;
            if !denominator.has(x) {
                continue;
            }
            let parts = match crate::polys::polytools::factor_list(&denominator, std::slice::from_ref(x)) {
                Ok((_, parts)) => parts,
                Err(e) if polynomial_failure(&e) => continue,
                Err(e) => return Err(e),
            };
            for (part, _) in parts {
                if part.has(x) && !factors.contains(&part) {
                    factors.push(part);
                }
            }
        }
    }
    mul(&factors)
}

/// The integrand in its closure (`_Closure`).
struct ClosureData {
    f: Basic,
    x: Basic,
    t: Basic,
    factors: Vec<Factor>,
    basis: Vec<Basis>,
    vector: Vector,
    t_derivatives: RefCell<Vec<Vector>>,
    x_derivatives: Vec<(Basis, Vector)>,
}

impl ClosureData {
    fn new(f: &Basic, x: &Basic, t: &Basic, coefficient: &Basic, factors: &[Factor]) -> PolyResult<ClosureData> {
        let mut basis: Vec<Basis> = vec![Vec::new()];
        for factor in factors {
            let mut next: Vec<Basis> = Vec::new();
            for b in &basis {
                for m in 0..factor.order() {
                    let mut c: Basis = b.clone();
                    c.push(m);
                    next.push(c);
                }
            }
            basis = next;
        }
        let zero: Basis = vec![0; factors.len()];
        let mut vector: Vector = OrderedMap::new();
        vector.insert(zero, coefficient.clone());
        let mut x_derivatives: Vec<(Basis, Vector)> = Vec::new();
        for b in &basis {
            let mut single: Vector = OrderedMap::new();
            single.insert(b.clone(), Basic::one());
            x_derivatives.push((b.clone(), derive(&single, x, factors)?));
        }
        Ok(ClosureData {
            f: f.clone(),
            x: x.clone(),
            t: t.clone(),
            factors: factors.to_vec(),
            basis,
            vector: vector.clone(),
            t_derivatives: RefCell::new(vec![vector]),
            x_derivatives,
        })
    }

    fn t_derivative(&self, k: usize) -> PolyResult<Vector> {
        while self.t_derivatives.borrow().len() <= k {
            let last: Vector = self.t_derivatives.borrow().last().expect("nonempty").clone();
            let next: Vector = derive(&last, &self.t, &self.factors)?;
            self.t_derivatives.borrow_mut().push(next);
        }
        Ok(self.t_derivatives.borrow()[k].clone())
    }

    /// The basis element as an expression (`element`).
    fn element(&self, basis: &Basis) -> R {
        let parts: Vec<Basic> = self
            .factors
            .iter()
            .zip(basis)
            .map(|(factor, m)| factor.derivative(*m))
            .collect::<PolyResult<_>>()?;
        mul(&parts)
    }
}

/// The telescoper of the given order with the certificate of the given
/// degree over `D**power` (`_telescope`).
fn telescope(
    closure_data: &ClosureData,
    order: usize,
    degree: usize,
    power: i64,
) -> PolyResult<Option<DFiniteTelescoper>> {
    let (x, t) = (&closure_data.x, &closure_data.t);
    let lhs: Vec<Vector> = (0..=order)
        .map(|k| closure_data.t_derivative(k))
        .collect::<PolyResult<_>>()?;
    let mut all: Vec<&Vector> = lhs.iter().collect();
    all.extend(closure_data.x_derivatives.iter().map(|(_, v)| v));
    all.push(&closure_data.vector);
    let d: Basic = pow(&denominators(&all, x)?, &int(power))?;
    let width: usize = order + 1 + closure_data.basis.len() * (degree + 1);
    let unknown = |b: usize, m: usize| order + 1 + b * (degree + 1) + m;
    // the coefficient of each unknown at each basis element
    let mut total: OrderedMap<Basis, OrderedMap<usize, Basic>> = OrderedMap::new();
    let mut contribute = |key: &Basis, column: usize, value: &Basic| -> PolyResult<()> {
        let mut entries: OrderedMap<usize, Basic> = total.get(key).cloned().unwrap_or_else(OrderedMap::new);
        let current: Basic = entries.get(&column).cloned().unwrap_or_else(Basic::zero);
        entries.insert(column, add(&[current, value.clone()])?);
        total.insert(key.clone(), entries);
        Ok(())
    };
    for (k, vector) in lhs.iter().enumerate() {
        for (key, value) in vector.iter() {
            contribute(key, k, value)?;
        }
    }
    for (i, b) in closure_data.basis.iter().enumerate() {
        for m in 0..=degree {
            let g: Basic = div(&pow(x, &int(m as i64))?, &d)?;
            let column: usize = unknown(i, m);
            contribute(b, column, &neg(&diff(&g, x, 1)?)?)?;
            let derivative: &Vector = &closure_data.x_derivatives[i].1;
            for (key, value) in derivative.iter() {
                contribute(key, column, &neg(&mul(&[g.clone(), value.clone()])?)?)?;
            }
        }
    }
    let entries: Vec<Basic> = total.values().flat_map(|row| row.values().cloned()).collect();
    let built = (|| -> PolyResult<(Context, Vec<Vec<Element>>)> {
        let k: Context = Context::new(&entries, x, &[], false)?;
        let mut rows: Vec<Vec<Element>> = Vec::new();
        for row in total.values() {
            let mut fractions: Vec<(usize, KPoly, KPoly)> = Vec::new();
            for (column, value) in row.iter() {
                let (p, q) = k.fraction(value)?;
                fractions.push((*column, p, q));
            }
            let mut common: KPoly = k.constant(&k.field.one());
            for (_, _, q) in &fractions {
                common = common.lcm(q)?;
            }
            let mut numerators: Vec<(usize, KPoly)> = Vec::new();
            for (column, p, q) in &fractions {
                numerators.push((*column, p.mul(&common.quo(q)?)?));
            }
            let height: isize = numerators
                .iter()
                .filter(|(_, p)| !p.is_zero())
                .map(|(_, p)| p.degree())
                .max()
                .unwrap_or(-1)
                + 1;
            for power_ in 0..height.max(0) as usize {
                rows.push(
                    (0..width)
                        .map(|c| match numerators.iter().find(|(col, _)| *col == c) {
                            Some((_, p)) => p.coeff(power_),
                            None => k.field.zero(),
                        })
                        .collect(),
                );
            }
        }
        Ok((k, rows))
    })();
    let (k, rows) = match built {
        Ok(v) => v,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    for null_vector in k.field.nullspace(&rows, width)? {
        let solution: Vec<Basic> = null_vector
            .iter()
            .map(|v| k.element_expr(v))
            .collect::<PolyResult<_>>()?;
        if solution[..=order].iter().all(|v| *v == Basic::zero()) {
            continue;
        }
        let (coefficients, scale) = normalized(&solution[..=order], t)?;
        let mut certificate: Vec<(Basic, Basic)> = Vec::new();
        for (i, b) in closure_data.basis.iter().enumerate() {
            let mut terms: Vec<Basic> = Vec::new();
            for m in 0..=degree {
                terms.push(mul(&[solution[unknown(i, m)].clone(), pow(x, &int(m as i64))?])?);
            }
            let g: Basic = div(&add(&terms)?, &d)?;
            let g_: Basic = cancel(&mul(&[g, scale.clone()])?)?;
            if g_ != Basic::zero() {
                certificate.push((closure_data.element(b)?, g_));
            }
        }
        return Ok(Some(DFiniteTelescoper {
            term: closure_data.f.clone(),
            x: x.clone(),
            t: t.clone(),
            coefficients,
            certificate,
        }));
    }
    Ok(None)
}

/// The coefficients with denominators cleared and common factors removed,
/// and the factor they were multiplied by (`_normalized`).
fn normalized(values: &[Basic], t: &Basic) -> PolyResult<(Vec<Basic>, Basic)> {
    let mut parts: Vec<Basic> = Vec::new();
    for v in values {
        parts.push(mul(&[v.clone(), fresh("n")])?);
    }
    let combined: Basic = crate::polys::rationaltools::together(&add(&parts)?, false, true)?;
    let (_, denominator) = crate::core::expr::as_numer_denom(&combined)?;
    let mut scaled: Vec<Basic> = values
        .iter()
        .map(|v| cancel(&mul(&[v.clone(), denominator.clone()])?))
        .collect::<PolyResult<_>>()?;
    let mut scale: Basic = denominator.clone();
    let attempt_common = (|| -> PolyResult<Option<(Vec<Basic>, Basic)>> {
        let polys: Vec<crate::polys::polytools::Poly> = scaled
            .iter()
            .map(|v| crate::polys::polytools::Poly::new(v, std::slice::from_ref(t)))
            .collect::<PolyResult<_>>()?;
        let mut common = polys[0].clone();
        for p in &polys[1..] {
            common = common.gcd(p)?;
        }
        if common.degree(0)? > 0 || common.LC()? != Basic::one() {
            let quotients: Vec<Basic> = polys
                .iter()
                .map(|p| p.div(&common)?.0.as_expr())
                .collect::<PolyResult<_>>()?;
            return Ok(Some((quotients, div(&denominator, &common.as_expr()?)?)));
        }
        Ok(None)
    })();
    match attempt_common {
        Ok(Some((q, s))) => {
            scaled = q;
            scale = s;
        }
        Ok(None) => {}
        Err(e) if polynomial_failure(&e) => {}
        Err(e) => return Err(e),
    }
    if scaled
        .last()
        .is_some_and(crate::core::printing::could_extract_minus_sign)
    {
        scaled = scaled.iter().map(neg).collect::<PolyResult<_>>()?;
        scale = neg(&scale)?;
    }
    Ok((scaled, scale))
}

/// A telescoper and certificate for `F(x, t)` by Chyzak's algorithm with
/// Koutschan's ansatz (`chyzak`).
pub fn chyzak(
    f: &Basic,
    x: &Basic,
    t: &Basic,
    max_order: usize,
    max_degree: usize,
) -> PolyResult<Option<DFiniteTelescoper>> {
    let Some((coefficient, factors)) = closure(f, x, t)? else {
        return Ok(None);
    };
    let closure_data: ClosureData = ClosureData::new(f, x, t, &coefficient, &factors)?;
    for order in 0..=max_order {
        for power in [1i64, 2] {
            for degree in 0..=max_degree {
                let found =
                    crate::timeout::attempt_within(timeout(), || telescope(&closure_data, order, degree, power))?;
                if let Some(Some(telescoper)) = found {
                    return Ok(Some(telescoper));
                }
            }
        }
    }
    Ok(None)
}

/// The linear differential equation satisfied by `Integral(F, (x, a, b))`
/// (`dfinite_ode`).
pub fn dfinite_ode(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    assumptions: &[Basic],
    name: &str,
) -> PolyResult<Option<Basic>> {
    let Some(telescoper) = chyzak(f, x, t, 3, 6)? else {
        return Ok(None);
    };
    let g: Basic = crate::core::function::doit(&telescoper.function()?)?;
    equation(&telescoper.coefficients, &g, x, a, b, t, assumptions, name)
}

/// `Integral(F, (x, a, b))` as a function of the parameter `t` by Chyzak's
/// algorithm (`dfinite_integral`).
pub fn dfinite_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    let Some(telescoper) = chyzak(f, x, t, 3, 6)? else {
        return Ok(None);
    };
    let g: Basic = crate::core::function::doit(&telescoper.function()?)?;
    let Some(eq) = equation(&telescoper.coefficients, &g, x, a, b, t, assumptions, "I")? else {
        return Ok(None);
    };
    let Some(solution) = solve(&telescoper.coefficients, &eq, f, x, a, b, t, assumptions)? else {
        return Ok(None);
    };
    let allowed = f.free_symbols();
    if solution.free_symbols().iter().any(|s| !allowed.contains(s) && s != x) {
        return Ok(None);
    }
    let value: Basic = crate::integrals::marichev::tidy(&solution, assumptions, &crate::core::logic::true_())?;
    let bad: bool = value.preorder().iter().any(|n| {
        matches!(
            n.head(),
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity | Head::Piecewise
        )
    });
    if bad || !verified(&value, f, x, a, b, assumptions)? {
        return Ok(None);
    }
    Ok(Some(ConditionalValue::new(value, crate::core::logic::true_())))
}
