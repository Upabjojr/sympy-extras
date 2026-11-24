//! Port of `sympy_extras/solvers/linear_ode.py`: linear differential
//! operators with rational coefficients (the Ore ring `K[D]`), their
//! polynomial, rational and hyperexponential solutions, the reduction of
//! order, the greatest common right divisor and least common left multiple,
//! the exterior and symmetric powers, and the independent solutions of a
//! homogeneous linear equation.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::core::basic::{Basic, Function, Head, RelOp, UndefinedFunction};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};
use crate::solvers::kovacic::{coefficients as poly_coefficients, failure, replaced, roots};

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
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn diff(e: &Basic, x: &Basic, n: u64) -> R {
    if n == 0 {
        return Ok(e.clone());
    }
    crate::core::function::diff(e, &[(x.clone(), n)])
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
pub(crate) fn sum(terms: &[Basic]) -> R {
    if terms.is_empty() {
        Ok(Basic::zero())
    } else {
        add(terms)
    }
}
fn product(factors: &[Basic]) -> R {
    if factors.is_empty() {
        Ok(Basic::one())
    } else {
        mul(factors)
    }
}
fn integer_of(e: &Basic) -> Option<i64> {
    if !matches!(e.head(), Head::Rational { value, .. } if value.is_int()) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
}
fn rational_of(e: &Basic) -> Option<Rat> {
    match e.head() {
        Head::Rational { value, .. } => Some(value.clone()),
        _ => None,
    }
}
fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}
fn is_rational_function(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(e, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn binomial(n: usize, k: usize) -> i64 {
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }
    result
}
fn function_of(name: &str, x: &Basic) -> R {
    Arc::new(UndefinedFunction::new(name, None, &[])).apply(vec![x.clone()])
}

/// The normal form `p/q` of a rational function, `e` itself when the
/// native `cancel` cannot take its coefficients (`_normal`).
pub(crate) fn normal(e: &Basic) -> R {
    match crate::polys::polytools::cancel(e, &[]) {
        Ok(c) => Ok(c),
        Err(err) if failure(&err) => Ok(e.clone()),
        Err(err) => Err(err),
    }
}

/// Whether `e` has an unevaluated integral (`has_integral`).
pub fn has_integral(e: &Basic) -> bool {
    crate::integrals::definite::is_unevaluated(e)
}

// ---------------------------------------------------------------------------
// polynomials in one variable with any constant coefficients

/// The coefficients the typed `Poly` refuses replaced by dummies, and back
/// (`_Generic`).
struct Generic {
    x: Basic,
    back: Vec<(Basic, Basic)>,
}

impl Generic {
    fn new(x: &Basic) -> Generic {
        Generic {
            x: x.clone(),
            back: Vec::new(),
        }
    }

    fn to(&mut self, e: &Basic) -> R {
        let (walked, back) = crate::solvers::kovacic::generic_coefficients(e, &self.x)?;
        // the dummies numbered on from the ones already given
        let mut renamed: Vec<(Basic, Basic)> = Vec::new();
        let mut result: Basic = walked;
        for (d, node) in back {
            if let Some((existing, _)) = self.back.iter().find(|(_, n)| *n == node) {
                renamed.push((d.clone(), existing.clone()));
            } else {
                let fresh: Basic = dummy(&format!("g{}", self.back.len()));
                renamed.push((d.clone(), fresh.clone()));
                self.back.push((fresh, node));
            }
        }
        if !renamed.is_empty() {
            result = replaced(&result, &renamed)?;
        }
        Ok(result)
    }

    fn restore(&self, e: &Basic) -> R {
        if self.back.is_empty() {
            Ok(e.clone())
        } else {
            replaced(e, &self.back)
        }
    }
}

/// A polynomial in `x` whose coefficients are any expressions free of `x`
/// (`UPoly`).
#[derive(Clone, Debug)]
pub struct UPoly {
    pub terms: BTreeMap<i64, Basic>,
    pub x: Basic,
}

impl UPoly {
    pub(crate) fn new(coefficients: BTreeMap<i64, Basic>, x: &Basic) -> UPoly {
        UPoly {
            terms: coefficients.into_iter().filter(|(_, c)| *c != Basic::zero()).collect(),
            x: x.clone(),
        }
    }

    /// The polynomial of an expression; `ValueError` when it is no
    /// polynomial in `x` (`of`).
    pub fn of(e: &Basic, x: &Basic) -> PolyResult<UPoly> {
        Ok(UPoly::new(poly_coefficients(e, x)?, x))
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn degree(&self) -> i64 {
        self.terms.keys().next_back().copied().unwrap_or(-1)
    }

    pub fn lc(&self) -> Basic {
        self.terms.values().next_back().cloned().unwrap_or_else(Basic::zero)
    }

    pub fn coeff(&self, k: i64) -> Basic {
        self.terms.get(&k).cloned().unwrap_or_else(Basic::zero)
    }

    pub fn all_coeffs(&self) -> Vec<Basic> {
        (0..=self.degree()).rev().map(|k| self.coeff(k)).collect()
    }

    pub fn as_expr(&self) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (k, c) in self.terms.iter().rev() {
            terms.push(if *k != 0 {
                mul(&[c.clone(), pow(&self.x, &int(*k))?])?
            } else {
                c.clone()
            });
        }
        sum(&terms)
    }

    pub(crate) fn diff(&self) -> PolyResult<UPoly> {
        let mut terms: BTreeMap<i64, Basic> = BTreeMap::new();
        for (k, c) in &self.terms {
            if *k != 0 {
                terms.insert(k - 1, mul(&[int(*k), c.clone()])?);
            }
        }
        Ok(UPoly::new(terms, &self.x))
    }

    pub(crate) fn scaled(&self, c: &Basic) -> PolyResult<UPoly> {
        let mut terms: BTreeMap<i64, Basic> = BTreeMap::new();
        for (k, a) in &self.terms {
            terms.insert(*k, normal(&mul(&[c.clone(), a.clone()])?)?);
        }
        Ok(UPoly::new(terms, &self.x))
    }

    /// `self - other` (`__sub__`).
    pub(crate) fn sub_poly(&self, other: &UPoly) -> PolyResult<UPoly> {
        let mut keys: Vec<i64> = self.terms.keys().chain(other.terms.keys()).copied().collect();
        keys.sort();
        keys.dedup();
        let mut terms: BTreeMap<i64, Basic> = BTreeMap::new();
        for k in keys {
            terms.insert(k, normal(&sub(&self.coeff(k), &other.coeff(k))?)?);
        }
        Ok(UPoly::new(terms, &self.x))
    }

    /// The monic greatest common divisor (the native `gcd`, Euclid's
    /// algorithm on coefficients it cannot take) (`gcd`).
    pub(crate) fn gcd(&self, other: &UPoly) -> PolyResult<UPoly> {
        let x: Basic = self.x.clone();
        let native = (|| -> PolyResult<UPoly> {
            UPoly::of(&polynomial_gcd(&self.as_expr()?, &other.as_expr()?, &x)?, &x)?.monic()
        })();
        match native {
            Ok(g) => return Ok(g),
            Err(e) if failure(&e) => {}
            Err(e) => return Err(e),
        }
        let (mut a, mut b) = (self.clone(), other.clone());
        while !b.is_zero() {
            let r: UPoly = a.rem(&b)?;
            a = b;
            b = r;
        }
        a.monic()
    }

    /// `(q, r)` with `self = q other + r` (`div`).
    pub(crate) fn div(&self, other: &UPoly) -> PolyResult<(UPoly, UPoly)> {
        if other.is_zero() {
            return Err(PolyError::ZeroDivision("polynomial division by zero".to_string()));
        }
        let m: i64 = other.degree();
        let lc: Basic = other.lc();
        let mut rest: BTreeMap<i64, Basic> = self.terms.clone();
        let mut quotient: BTreeMap<i64, Basic> = BTreeMap::new();
        while let Some((&top, _)) = rest.iter().next_back() {
            if top < m {
                break;
            }
            let value: Basic = rest.remove(&top).expect("present");
            let c: Basic = normal(&div(&value, &lc)?)?;
            if c == Basic::zero() {
                continue;
            }
            quotient.insert(top - m, c.clone());
            for (k, b) in &other.terms {
                if *k == m {
                    continue;
                }
                let key: i64 = k + top - m;
                let current: Basic = rest.get(&key).cloned().unwrap_or_else(Basic::zero);
                let v: Basic = normal(&sub(&current, &mul(&[c.clone(), b.clone()])?)?)?;
                if v == Basic::zero() {
                    rest.remove(&key);
                } else {
                    rest.insert(key, v);
                }
            }
        }
        Ok((UPoly::new(quotient, &self.x), UPoly::new(rest, &self.x)))
    }

    pub(crate) fn rem(&self, other: &UPoly) -> PolyResult<UPoly> {
        Ok(self.div(other)?.1)
    }

    pub(crate) fn monic(&self) -> PolyResult<UPoly> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        self.scaled(&pow(&self.lc(), &int(-1))?)
    }
}

/// The native `gcd` of two polynomials in `x`, with the coefficients it
/// cannot take as generators (`_polynomial_gcd`).
fn polynomial_gcd(f: &Basic, g: &Basic, x: &Basic) -> R {
    match crate::polys::polytools::gcd_list(&[f.clone(), g.clone()], std::slice::from_ref(x)) {
        Ok(h) => return Ok(h),
        Err(e) if failure(&e) => {}
        Err(e) => return Err(e),
    }
    let mut generic = Generic::new(x);
    let (gf, gg) = (generic.to(f)?, generic.to(g)?);
    generic.restore(&crate::polys::polytools::gcd_list(&[gf, gg], std::slice::from_ref(x))?)
}

/// The native `factor_list` of a polynomial in `x` without the constant
/// (`_factor_list`).
pub(crate) fn factor_list(e: &Basic, x: &Basic) -> PolyResult<Vec<(Basic, i64)>> {
    let found: Vec<(Basic, Basic)> = match crate::polys::polytools::factor_list(e, std::slice::from_ref(x)) {
        Ok((_, found)) => found,
        Err(err) if failure(&err) => {
            let mut generic = Generic::new(x);
            let attempt = (|| -> PolyResult<Vec<(Basic, Basic)>> {
                let (_, found) = crate::polys::polytools::factor_list(&generic.to(e)?, std::slice::from_ref(x))?;
                found.into_iter().map(|(f, m)| Ok((generic.restore(&f)?, m))).collect()
            })();
            match attempt {
                Ok(found) => found,
                Err(err) if failure(&err) => return Ok(vec![(e.clone(), 1)]),
                Err(err) => return Err(err),
            }
        }
        Err(err) => return Err(err),
    };
    Ok(found
        .into_iter()
        .filter_map(|(f, m)| integer_of(&m).map(|m| (f, m)))
        .collect())
}

/// The resultant of two polynomials in `c` (`_resultant`).
pub(crate) fn resultant(f: &Basic, g: &Basic, c: &Basic) -> R {
    match crate::polys::polytools::resultant_expr(f, g, std::slice::from_ref(c)) {
        Ok(r) => return Ok(r),
        Err(err) if failure(&err) => {}
        Err(err) => return Err(err),
    }
    let (p, q) = (UPoly::of(f, c)?.all_coeffs(), UPoly::of(g, c)?.all_coeffs());
    let (m, n) = (p.len() - 1, q.len() - 1);
    let size: usize = m + n;
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for i in 0..n {
        let mut row: Vec<Basic> = vec![Basic::zero(); i];
        row.extend(p.iter().cloned());
        row.extend(vec![Basic::zero(); size - m - 1 - i]);
        rows.push(row);
    }
    for i in 0..m {
        let mut row: Vec<Basic> = vec![Basic::zero(); i];
        row.extend(q.iter().cloned());
        row.extend(vec![Basic::zero(); size - n - 1 - i]);
        rows.push(row);
    }
    expand(&crate::solvers::ode_cases::determinant(&rows)?)
}

/// The least common multiple of polynomials (`_lcm`).
pub(crate) fn lcm(exprs: &[Basic]) -> R {
    match crate::polys::polytools::lcm_list(exprs, &[]) {
        Ok(l) => return Ok(l),
        Err(err) if failure(&err) => {}
        Err(err) => return Err(err),
    }
    let mut seen: Vec<Basic> = Vec::new();
    let mut result: Basic = Basic::one();
    for e in exprs {
        if !seen.contains(e) {
            seen.push(e.clone());
            result = mul(&[result, e.clone()])?;
        }
    }
    expand(&result)
}

fn gcd_list(exprs: &[Basic]) -> R {
    match crate::polys::polytools::gcd_list(exprs, &[]) {
        Ok(g) => Ok(g),
        Err(err) if failure(&err) => Ok(Basic::one()),
        Err(err) => Err(err),
    }
}

/// The roots the native `roots` finds, with their multiplicities
/// (`_root_multiplicities`).
pub(crate) fn root_multiplicities(p: &UPoly) -> PolyResult<Vec<(Basic, usize)>> {
    if p.degree() < 1 {
        return Ok(Vec::new());
    }
    match roots(&p.as_expr()?, &p.x) {
        Ok(found) => Ok(found),
        Err(err) if failure(&err) => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

// ---------------------------------------------------------------------------

/// `a_n D**n + ... + a_1 D + a_0` with coefficients `[a_0, ..., a_n]`
/// (rational functions) in `x` (`LinearOperator`).
#[derive(Clone, Debug)]
pub struct LinearOperator {
    pub coefficients: Vec<Basic>,
    pub x: Basic,
}

impl LinearOperator {
    pub fn new(coefficients: &[Basic], x: &Basic) -> LinearOperator {
        let mut c: Vec<Basic> = coefficients.to_vec();
        while c.len() > 1 && *c.last().expect("nonempty") == Basic::zero() {
            c.pop();
        }
        if c.is_empty() {
            c.push(Basic::zero());
        }
        LinearOperator {
            coefficients: c,
            x: x.clone(),
        }
    }

    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// The operator of a homogeneous linear equation in `f`, the
    /// coefficients made polynomial by clearing denominators
    /// (`from_equation`).
    pub fn from_equation(equation: &Basic, f: &Basic) -> PolyResult<LinearOperator> {
        let x: Basic = match f.args().first() {
            Some(x) if x.is_symbol() => x.clone(),
            _ => return Err(PolyError::Value("the function must depend on a symbol".to_string())),
        };
        let lhs: Basic = if matches!(equation.head(), Head::Relational(RelOp::Eq)) {
            sub(&equation.args()[0], &equation.args()[1])?
        } else if crate::core::logic::is_boolean(equation) {
            return Err(PolyError::Type(format!("{} is not an equation", equation)));
        } else {
            equation.clone()
        };
        let expanded: Basic = expand(&lhs)?;
        let mut n: u64 = 0;
        for d in expanded.atoms(Some(&|e: &Basic| matches!(e.head(), Head::Derivative))) {
            if d.args()[0] == *f {
                let mut count: u64 = 0;
                for pair in &d.args()[1..] {
                    if pair.args()[0] == x {
                        count += integer_of(&pair.args()[1]).unwrap_or(1) as u64;
                    }
                }
                n = n.max(count);
            }
        }
        let mut coefficients: Vec<Basic> = Vec::new();
        let mut rest: Basic = expanded;
        for i in (0..=n).rev() {
            let term: Basic = if i > 0 { diff(f, &x, i)? } else { f.clone() };
            let c: Basic = crate::core::expr::coeff(&rest, &term)?;
            rest = expand(&sub(&rest, &mul(&[c.clone(), term])?)?)?;
            coefficients.push(c);
        }
        if rest != Basic::zero() || coefficients.iter().any(|c| c.has(f)) {
            return Err(PolyError::Value(format!(
                "{} is not a homogeneous linear equation in {}",
                equation, f
            )));
        }
        coefficients.reverse();
        let denominators: Vec<Basic> = coefficients
            .iter()
            .map(|c| Ok(numer_denom(&normal(c)?)?.1))
            .collect::<PolyResult<_>>()?;
        let common: Basic = lcm(&denominators)?;
        let cleared: Vec<Basic> = coefficients
            .iter()
            .map(|c| normal(&mul(&[c.clone(), common.clone()])?))
            .collect::<PolyResult<_>>()?;
        for c in &cleared {
            if !is_polynomial(c, &x) {
                return Err(PolyError::Value(format!(
                    "the coefficients are not rational functions of {}",
                    x
                )));
            }
        }
        Ok(LinearOperator::new(&cleared, &x))
    }

    /// `L(y)` (`__call__`).
    pub fn apply(&self, y: &Basic) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[c.clone(), diff(y, &self.x, i as u64)?])?);
        }
        expand(&sum(&terms)?)
    }

    /// The operator `M` with `L(factor * u) = factor * M(u)`, denominators
    /// cleared (`transformed`).
    pub fn transformed(&self, factor: &Basic) -> PolyResult<LinearOperator> {
        let x: &Basic = &self.x;
        let rho: Basic = log_derivative(factor, x)?;
        if is_rational_function(&rho, x) {
            let mut ratios: Vec<Basic> = vec![Basic::one()];
            for _ in 0..self.order() {
                let last: Basic = ratios.last().expect("nonempty").clone();
                ratios.push(normal(&add(&[diff(&last, x, 1)?, mul(&[rho.clone(), last])?])?)?);
            }
            let mut shifted: Vec<Basic> = vec![Basic::zero(); self.order() + 1];
            for (i, a) in self.coefficients.iter().enumerate() {
                if *a == Basic::zero() {
                    continue;
                }
                for l in 0..=i {
                    shifted[l] = add(&[
                        shifted[l].clone(),
                        mul(&[int(binomial(i, l)), a.clone(), ratios[i - l].clone()])?,
                    ])?;
                }
            }
            let normalized: Vec<Basic> = shifted.iter().map(normal).collect::<PolyResult<_>>()?;
            return LinearOperator::new(&normalized, x).cleared();
        }
        let u: Basic = function_of("u", x)?;
        let image: Basic = expand(&normal(&div(
            &self.apply(&mul(&[factor.clone(), u.clone()])?)?,
            factor,
        )?)?)?;
        let mut coefficients: Vec<Basic> = Vec::new();
        let mut rest: Basic = image;
        for i in (0..=self.order()).rev() {
            let term: Basic = if i > 0 { diff(&u, x, i as u64)? } else { u.clone() };
            let c: Basic = crate::core::expr::coeff(&rest, &term)?;
            rest = expand(&sub(&rest, &mul(&[c.clone(), term])?)?)?;
            coefficients.push(normal(&c)?);
        }
        coefficients.reverse();
        let denominators: Vec<Basic> = coefficients
            .iter()
            .map(|c| Ok(numer_denom(&normal(c)?)?.1))
            .collect::<PolyResult<_>>()?;
        let common: Basic = lcm(&denominators)?;
        let cleared: Vec<Basic> = coefficients
            .iter()
            .map(|c| normal(&mul(&[c.clone(), common.clone()])?))
            .collect::<PolyResult<_>>()?;
        Ok(LinearOperator::new(&cleared, x))
    }

    pub fn leading_coefficient(&self) -> &Basic {
        self.coefficients.last().expect("nonempty")
    }

    pub fn is_zero(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0] == Basic::zero()
    }

    /// Equality of operators with the coefficients compared by their normal
    /// forms (`__eq__`).
    pub fn equals(&self, other: &LinearOperator) -> PolyResult<bool> {
        if self.x != other.x || self.order() != other.order() {
            return Ok(false);
        }
        for (a, b) in self.coefficients.iter().zip(&other.coefficients) {
            if normal(&sub(a, b)?)? != Basic::zero() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn add(&self, other: &LinearOperator) -> PolyResult<LinearOperator> {
        let n: usize = self.order().max(other.order()) + 1;
        let get = |c: &[Basic], i: usize| c.get(i).cloned().unwrap_or_else(Basic::zero);
        let sums: Vec<Basic> = (0..n)
            .map(|i| normal(&add(&[get(&self.coefficients, i), get(&other.coefficients, i)])?))
            .collect::<PolyResult<_>>()?;
        Ok(LinearOperator::new(&sums, &self.x))
    }

    pub fn neg(&self) -> PolyResult<LinearOperator> {
        let c: Vec<Basic> = self.coefficients.iter().map(neg).collect::<PolyResult<_>>()?;
        Ok(LinearOperator::new(&c, &self.x))
    }

    pub fn sub(&self, other: &LinearOperator) -> PolyResult<LinearOperator> {
        self.add(&other.neg()?)
    }

    /// The composition `self(other(y))` (`__mul__`).
    pub fn mul(&self, other: &LinearOperator) -> PolyResult<LinearOperator> {
        let x: &Basic = &self.x;
        let top: usize = self
            .coefficients
            .iter()
            .enumerate()
            .filter(|(_, a)| **a != Basic::zero())
            .map(|(i, _)| i)
            .max()
            .unwrap_or(0);
        let mut derivatives: Vec<Vec<Basic>> = Vec::new();
        for b in &other.coefficients {
            let mut row: Vec<Basic> = vec![b.clone()];
            for _ in 0..top {
                let last: Basic = row.last().expect("nonempty").clone();
                row.push(normal(&diff(&last, x, 1)?)?);
            }
            derivatives.push(row);
        }
        let mut result: Vec<Basic> = vec![Basic::zero(); self.order() + other.order() + 1];
        for (i, a) in self.coefficients.iter().enumerate() {
            if *a == Basic::zero() {
                continue;
            }
            for l in 0..=i {
                let weight: Basic = mul(&[int(binomial(i, l)), a.clone()])?;
                for (j, row) in derivatives.iter().enumerate() {
                    let b: &Basic = &row[l];
                    if *b != Basic::zero() {
                        result[i - l + j] = add(&[result[i - l + j].clone(), mul(&[weight.clone(), b.clone()])?])?;
                    }
                }
            }
        }
        let normalized: Vec<Basic> = result.iter().map(normal).collect::<PolyResult<_>>()?;
        Ok(LinearOperator::new(&normalized, x))
    }

    /// The operator `c L` (`scaled`).
    pub fn scaled(&self, c: &Basic) -> PolyResult<LinearOperator> {
        let scaled: Vec<Basic> = self
            .coefficients
            .iter()
            .map(|a| normal(&mul(&[c.clone(), a.clone()])?))
            .collect::<PolyResult<_>>()?;
        Ok(LinearOperator::new(&scaled, &self.x))
    }

    /// The operator divided by its leading coefficient (`monic`).
    pub fn monic(&self) -> PolyResult<LinearOperator> {
        if self.is_zero() {
            return Err(PolyError::Value(
                "the zero operator has no leading coefficient".to_string(),
            ));
        }
        self.scaled(&pow(self.leading_coefficient(), &int(-1))?)
    }

    /// The operator times a rational function so that the coefficients are
    /// polynomials in `x` without common factor (`primitive`).
    pub fn primitive(&self) -> PolyResult<LinearOperator> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        let x: &Basic = &self.x;
        let parts: Vec<(Basic, Basic)> = self
            .coefficients
            .iter()
            .map(|c| numer_denom(&normal(c)?))
            .collect::<PolyResult<_>>()?;
        let common: Basic = lcm(&parts.iter().map(|(_, q)| q.clone()).collect::<Vec<_>>())?;
        let mut numerators: Vec<Basic> = parts
            .iter()
            .map(|(p, q)| normal(&div(&mul(&[p.clone(), common.clone()])?, q)?))
            .collect::<PolyResult<_>>()?;
        let nonzero: Vec<Basic> = numerators.iter().filter(|c| **c != Basic::zero()).cloned().collect();
        let content: Basic = gcd_list(&nonzero)?;
        if content.has(x) || crate::core::assumptions::is_number(&content) {
            numerators = numerators
                .iter()
                .map(|c| normal(&div(c, &content)?))
                .collect::<PolyResult<_>>()?;
        }
        Ok(LinearOperator::new(&numerators, x))
    }

    /// The operator times the least common multiple of the denominators of
    /// its coefficients (`cleared`).
    pub fn cleared(&self) -> PolyResult<LinearOperator> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        let parts: Vec<(Basic, Basic)> = self
            .coefficients
            .iter()
            .map(|c| numer_denom(&normal(c)?))
            .collect::<PolyResult<_>>()?;
        let common: Basic = lcm(&parts.iter().map(|(_, q)| q.clone()).collect::<Vec<_>>())?;
        let cleared: Vec<Basic> = parts
            .iter()
            .map(|(p, q)| normal(&div(&mul(&[p.clone(), common.clone()])?, q)?))
            .collect::<PolyResult<_>>()?;
        Ok(LinearOperator::new(&cleared, &self.x))
    }

    /// `(Q, R)` with `self = Q*other + R` (`right_divmod`).
    pub fn right_divmod(&self, other: &LinearOperator) -> PolyResult<(LinearOperator, LinearOperator)> {
        if other.is_zero() {
            return Err(PolyError::ZeroDivision("division by the zero operator".to_string()));
        }
        let x: &Basic = &self.x;
        let mut quotient: LinearOperator = LinearOperator::new(&[Basic::zero()], x);
        let mut rest: LinearOperator = self.clone();
        let m: usize = other.order();
        let lc: Basic = other.leading_coefficient().clone();
        while !rest.is_zero() && rest.order() >= m {
            let d: usize = rest.order() - m;
            let mut c: Vec<Basic> = vec![Basic::zero(); d];
            c.push(normal(&div(rest.leading_coefficient(), &lc)?)?);
            let term: LinearOperator = LinearOperator::new(&c, x);
            quotient = quotient.add(&term)?;
            let difference: LinearOperator = rest.sub(&term.mul(other)?)?;
            // the leading coefficient cancels exactly: drop it
            let coefficients: Vec<Basic> = if difference.order() >= rest.order() {
                difference.coefficients[..rest.order()].to_vec()
            } else {
                difference.coefficients.clone()
            };
            rest = if coefficients.is_empty() {
                LinearOperator::new(&[Basic::zero()], x)
            } else {
                LinearOperator::new(&coefficients, x)
            };
        }
        Ok((quotient, rest))
    }

    pub fn right_quotient(&self, other: &LinearOperator) -> PolyResult<LinearOperator> {
        Ok(self.right_divmod(other)?.0)
    }

    pub fn right_remainder(&self, other: &LinearOperator) -> PolyResult<LinearOperator> {
        Ok(self.right_divmod(other)?.1)
    }

    /// The adjoint `L* = sum (-1)**i D**i a_i` (`adjoint`).
    pub fn adjoint(&self) -> PolyResult<LinearOperator> {
        let x: &Basic = &self.x;
        let n: usize = self.order();
        let mut derivatives: Vec<Vec<Basic>> = Vec::new();
        for (i, a) in self.coefficients.iter().enumerate() {
            let mut row: Vec<Basic> = vec![a.clone()];
            for _ in 0..i {
                let last: Basic = row.last().expect("nonempty").clone();
                row.push(normal(&diff(&last, x, 1)?)?);
            }
            derivatives.push(row);
        }
        let mut result: Vec<Basic> = Vec::new();
        for l in 0..=n {
            let mut total: Basic = Basic::zero();
            for (i, row) in derivatives.iter().enumerate().skip(l) {
                let sign: i64 = if i % 2 == 0 { 1 } else { -1 };
                total = add(&[total, mul(&[int(sign * binomial(i, l)), row[i - l].clone()])?])?;
            }
            result.push(normal(&total)?);
        }
        Ok(LinearOperator::new(&result, x))
    }

    /// The `k`-th associated operator (`exterior_power`).
    pub fn exterior_power(&self, k: usize) -> PolyResult<LinearOperator> {
        if k < 1 || k > self.order() {
            return Err(PolyError::Value(
                "the power must be between 1 and the order".to_string(),
            ));
        }
        let (subsets, system) = exterior_system(&self.monic()?, k)?;
        let start: Vec<Basic> = (0..subsets.len())
            .map(|i| if i == 0 { Basic::one() } else { Basic::zero() })
            .collect();
        Ok(scalar_operator(&system, &start, &self.x)?.0)
    }

    /// The `m`-th symmetric power (`symmetric_power`).
    pub fn symmetric_power(&self, m: usize) -> PolyResult<LinearOperator> {
        if m < 1 {
            return Err(PolyError::Value("the power must be positive".to_string()));
        }
        let (monomials, system) = symmetric_system(&self.monic()?, m)?;
        let start: Vec<Basic> = (0..monomials.len())
            .map(|i| if i == 0 { Basic::one() } else { Basic::zero() })
            .collect();
        Ok(scalar_operator(&system, &start, &self.x)?.0)
    }
}

// ---------------------------------------------------------------------------
// helpers of the operator arithmetic

/// `h'/h`, computed factor by factor (`log_derivative`).
pub fn log_derivative(h: &Basic, x: &Basic) -> R {
    if !h.has(x) {
        return Ok(Basic::zero());
    }
    if h.is_mul() {
        let parts: Vec<Basic> = h
            .args()
            .iter()
            .filter(|f| !crate::core::logic::is_boolean(f))
            .map(|f| log_derivative(f, x))
            .collect::<PolyResult<_>>()?;
        return normal(&sum(&parts)?);
    }
    if h.is_pow() && !h.args()[1].has(x) {
        let (base, exponent) = (&h.args()[0], &h.args()[1]);
        return normal(&div(&mul(&[exponent.clone(), diff(base, x, 1)?])?, base)?);
    }
    if h.is_function(Function::Exp) {
        return normal(&diff(&h.args()[0], x, 1)?);
    }
    normal(&div(&diff(h, x, 1)?, h)?)
}

/// A matrix of rational functions over the native field of rational
/// functions of its symbols (`_RationalRows`).
pub(crate) struct RationalRows {
    pub(crate) restored: HashMap<Basic, Basic>,
    /// the symbols in the order of their names (`restored.values()`)
    pub(crate) symbols: Vec<Basic>,
    pub(crate) field: crate::polys::domains::sparsefractionfield::SparseFractionField<
        crate::polys::domains::rationalfield::RationalField,
    >,
    pub(crate) rows: Vec<
        Vec<
            crate::polys::domains::sparsefractionfield::SparseFracElement<
                crate::polys::domains::rationalfield::RationalField,
            >,
        >,
    >,
}

impl RationalRows {
    pub(crate) fn new(rows: &[Vec<Basic>]) -> PolyResult<RationalRows> {
        use crate::polys::domains::domain::Domain;
        let mut symbols: Vec<Basic> = Vec::new();
        for row in rows {
            for e in row {
                for s in e.free_symbols() {
                    if !symbols.contains(&s) {
                        symbols.push(s);
                    }
                }
            }
        }
        symbols.sort_by_cached_key(crate::core::sorting::sort_key);
        let mut names: Vec<String> = (0..symbols.len()).map(|i| format!("_ln{}", i)).collect();
        if names.is_empty() {
            names.push("_ln".to_string());
        }
        let plain = |name: &str| crate::core::symbol::Symbol::new(name, &[]).0;
        let renamed: Vec<(Basic, Basic)> = symbols.iter().zip(&names).map(|(s, n)| (s.clone(), plain(n))).collect();
        let restored: HashMap<Basic, Basic> = symbols.iter().zip(&names).map(|(s, n)| (plain(n), s.clone())).collect();
        let field = crate::polys::domains::sparsefractionfield::SparseFractionField::new(
            crate::polys::domains::sparsepolynomialring::SparsePolynomialRing::sympy(
                crate::polys::domains::rationalfield::QQ,
                names,
            ),
        );
        let mut converted = Vec::new();
        for row in rows {
            let mut line = Vec::new();
            for e in row {
                line.push(field.from_expr(&replaced(e, &renamed)?)?);
            }
            converted.push(line);
        }
        Ok(RationalRows {
            restored,
            symbols,
            field,
            rows: converted,
        })
    }

    pub(crate) fn expr(
        &self,
        v: &crate::polys::domains::sparsefractionfield::SparseFracElement<
            crate::polys::domains::rationalfield::RationalField,
        >,
    ) -> R {
        use crate::polys::domains::domain::Domain;
        crate::core::function::xreplace_evaluated(&self.field.to_expr(v)?, &self.restored)
    }
}

pub(crate) fn field_failure(e: &PolyError) -> bool {
    failure(e)
}

/// A basis of the vectors annihilated by the rows (`_nullspace`).
pub(crate) fn nullspace(rows: &[Vec<Basic>], columns: usize) -> PolyResult<Vec<Vec<Basic>>> {
    use crate::polys::domains::domain::Domain;
    if rows.is_empty() {
        return Ok((0..columns)
            .map(|n| {
                (0..columns)
                    .map(|k| if k == n { Basic::one() } else { Basic::zero() })
                    .collect()
            })
            .collect());
    }
    let matrix: RationalRows = match RationalRows::new(rows) {
        Ok(m) => m,
        Err(e) if field_failure(&e) => return expression_nullspace(rows, columns),
        Err(e) => return Err(e),
    };
    matrix
        .field
        .nullspace(&matrix.rows, columns)?
        .iter()
        .map(|vector| vector.iter().map(|v| matrix.expr(v)).collect())
        .collect()
}

/// The inverse of a square matrix of rational functions, `None` when it
/// is singular (`_inverse`).
pub(crate) fn inverse(rows: &[Vec<Basic>]) -> PolyResult<Option<Vec<Vec<Basic>>>> {
    use crate::polys::domains::domain::Domain;
    let n: usize = rows.len();
    let augmented: Vec<Vec<Basic>> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r: Vec<Basic> = row.clone();
            r.extend((0..n).map(|j| if i == j { Basic::one() } else { Basic::zero() }));
            r
        })
        .collect();
    let matrix: RationalRows = match RationalRows::new(&augmented) {
        Ok(m) => m,
        Err(e) if field_failure(&e) => {
            let (reduced, pivots) = crate::solvers::recurr::row_reduce(&augmented)?;
            if pivots.len() < n || pivots[..n] != (0..n).collect::<Vec<usize>>()[..] {
                return Ok(None);
            }
            return Ok(Some(
                (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| normal(&reduced[i][n + j]))
                            .collect::<PolyResult<Vec<Basic>>>()
                    })
                    .collect::<PolyResult<_>>()?,
            ));
        }
        Err(e) => return Err(e),
    };
    let field = &matrix.field;
    let sparse: Vec<crate::polys::domains::domain::SparseRow<_>> = matrix
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, c)| !field.is_zero(c))
                .map(|(j, c)| (j, c.clone()))
                .collect()
        })
        .filter(|r: &crate::polys::domains::domain::SparseRow<_>| !r.is_empty())
        .collect();
    let echelon = field.sparse_echelon(&sparse);
    let mut pivot_rows: BTreeMap<usize, &crate::polys::domains::domain::SparseRow<_>> = BTreeMap::new();
    for r in &echelon {
        if let Some((&first, _)) = r.iter().next() {
            pivot_rows.insert(first, r);
        }
    }
    let leads: Vec<usize> = pivot_rows.keys().copied().collect();
    if leads.len() < n || leads[..n] != (0..n).collect::<Vec<usize>>()[..] {
        return Ok(None);
    }
    let mut result: Vec<Vec<Basic>> = Vec::new();
    for i in 0..n {
        let row = pivot_rows[&i];
        let mut line: Vec<Basic> = Vec::new();
        for j in 0..n {
            line.push(match row.get(&(n + j)) {
                Some(v) => matrix.expr(v)?,
                None => Basic::zero(),
            });
        }
        result.push(line);
    }
    Ok(Some(result))
}

fn expression_nullspace(rows: &[Vec<Basic>], columns: usize) -> PolyResult<Vec<Vec<Basic>>> {
    let (reduced, pivots) = crate::solvers::recurr::row_reduce(rows)?;
    let mut basis: Vec<Vec<Basic>> = Vec::new();
    for free in 0..columns {
        if pivots.contains(&free) {
            continue;
        }
        let mut vector: Vec<Basic> = vec![Basic::zero(); columns];
        vector[free] = Basic::one();
        for (row, pivot) in pivots.iter().enumerate() {
            vector[*pivot] = neg(&reduced[row][free])?;
        }
        basis.push(vector);
    }
    Ok(basis)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    fn rec(start: usize, n: usize, k: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            rec(i + 1, n, k, current, result);
            current.pop();
        }
    }
    rec(0, n, k, &mut current, &mut result);
    result
}

pub(crate) type System = Vec<Vec<Basic>>;

/// The system `w' = A w` of the `k x k` minors of the Wronskian matrix
/// (`_exterior_system`).
pub(crate) fn exterior_system(l: &LinearOperator, k: usize) -> PolyResult<(Vec<Vec<usize>>, System)> {
    let n: usize = l.order();
    let a: &[Basic] = &l.coefficients;
    let subsets: Vec<Vec<usize>> = combinations(n, k);
    let position = |subset: &[usize]| subsets.iter().position(|s| s == subset).expect("a subset");
    let mut system: System = vec![vec![Basic::zero(); subsets.len()]; subsets.len()];
    for (p, subset) in subsets.iter().enumerate() {
        for (t, &i) in subset.iter().enumerate() {
            if i + 1 < n {
                if subset.contains(&(i + 1)) {
                    continue;
                }
                let mut j: Vec<usize> = subset.clone();
                j[t] = i + 1;
                let q: usize = position(&j);
                system[p][q] = add(&[system[p][q].clone(), int(1)])?;
            } else {
                let base: &[usize] = &subset[..subset.len() - 1];
                for (m, am) in a.iter().enumerate().take(n) {
                    if base.contains(&m) || *am == Basic::zero() {
                        continue;
                    }
                    let mut j: Vec<usize> = base.to_vec();
                    j.push(m);
                    j.sort();
                    let sign: i64 = if base.iter().filter(|b| **b > m).count() % 2 == 0 {
                        1
                    } else {
                        -1
                    };
                    let q: usize = position(&j);
                    system[p][q] = sub(&system[p][q], &mul(&[int(sign), am.clone()])?)?;
                }
            }
        }
    }
    Ok((subsets, system))
}

/// The system of the monomials of degree `m` in `y, ..., y^(n-1)`
/// (`_symmetric_system`).
fn symmetric_system(l: &LinearOperator, m: usize) -> PolyResult<(Vec<Vec<usize>>, System)> {
    let n: usize = l.order();
    let a: &[Basic] = &l.coefficients;
    let mut monomials: Vec<Vec<usize>> = vec![Vec::new()];
    for _ in 0..n {
        let mut next: Vec<Vec<usize>> = Vec::new();
        for e in &monomials {
            for c in 0..=m {
                let mut f: Vec<usize> = e.clone();
                f.push(c);
                next.push(f);
            }
        }
        monomials = next;
    }
    monomials.retain(|e| e.iter().sum::<usize>() == m);
    monomials.sort_by(|p, q| q.cmp(p));
    let position = |e: &[usize]| monomials.iter().position(|f| f == e).expect("a monomial");
    let mut system: System = vec![vec![Basic::zero(); monomials.len()]; monomials.len()];
    for (p, e) in monomials.iter().enumerate() {
        for i in 0..n {
            if e[i] == 0 {
                continue;
            }
            let mut lowered: Vec<usize> = e.clone();
            lowered[i] -= 1;
            if i + 1 < n {
                let mut raised: Vec<usize> = lowered.clone();
                raised[i + 1] += 1;
                let q: usize = position(&raised);
                system[p][q] = add(&[system[p][q].clone(), int(e[i] as i64)])?;
            } else {
                for (j, aj) in a.iter().enumerate().take(n) {
                    if *aj == Basic::zero() {
                        continue;
                    }
                    let mut raised: Vec<usize> = lowered.clone();
                    raised[j] += 1;
                    let q: usize = position(&raised);
                    system[p][q] = sub(&system[p][q], &mul(&[int(e[i] as i64), aj.clone()])?)?;
                }
            }
        }
    }
    Ok((monomials, system))
}

/// The monic operator of least order annihilating `u = start . w` for the
/// solutions of `w' = A w` (`_scalar_operator`).
pub(crate) fn scalar_operator(
    system: &System,
    start: &[Basic],
    x: &Basic,
) -> PolyResult<(LinearOperator, Vec<Vec<Basic>>)> {
    let size: usize = start.len();
    let mut rows: Vec<Vec<Basic>> = vec![start.iter().map(normal).collect::<PolyResult<_>>()?];
    loop {
        let v: Vec<Basic> = rows.last().expect("nonempty").clone();
        let mut following: Vec<Basic> = Vec::new();
        for j in 0..size {
            let mut terms: Vec<Basic> = vec![diff(&v[j], x, 1)?];
            for (i, vi) in v.iter().enumerate() {
                if *vi != Basic::zero() && system[i][j] != Basic::zero() {
                    terms.push(mul(&[vi.clone(), system[i][j].clone()])?);
                }
            }
            following.push(normal(&sum(&terms)?)?);
        }
        let mut columns: Vec<Vec<Basic>> = rows.clone();
        columns.push(following.clone());
        let matrix: Vec<Vec<Basic>> = (0..size)
            .map(|i| columns.iter().map(|c| c[i].clone()).collect())
            .collect();
        for relation in nullspace(&matrix, columns.len())? {
            let last: Basic = relation.last().expect("nonempty").clone();
            if last != Basic::zero() {
                let coefficients: Vec<Basic> = relation
                    .iter()
                    .map(|c| normal(&div(c, &last)?))
                    .collect::<PolyResult<_>>()?;
                return Ok((LinearOperator::new(&coefficients, x), rows));
            }
        }
        rows.push(following);
    }
}

/// The monic greatest common right divisor of `a` and `b` (`gcrd`).
pub fn gcrd(a: &LinearOperator, b: &LinearOperator) -> PolyResult<LinearOperator> {
    let (mut a, mut b) = (a.clone(), b.clone());
    while !b.is_zero() {
        let r: LinearOperator = a.right_remainder(&b)?;
        a = b;
        b = r;
    }
    if a.is_zero() {
        Ok(a)
    } else {
        a.monic()
    }
}

/// The monic least common left multiple of `a` and `b` (`lclm`).
pub fn lclm(a: &LinearOperator, b: &LinearOperator) -> PolyResult<LinearOperator> {
    let x: Basic = a.x.clone();
    let (a, b) = (a.monic()?, b.monic()?);
    let mut remainders: Vec<(LinearOperator, LinearOperator)> = Vec::new();
    let mut power: LinearOperator = LinearOperator::new(&[Basic::one()], &x);
    let d: LinearOperator = LinearOperator::new(&[Basic::zero(), Basic::one()], &x);
    let (na, nb) = (a.order(), b.order());
    loop {
        remainders.push((power.right_remainder(&a)?, power.right_remainder(&b)?));
        let vectors: Vec<Vec<Basic>> = remainders
            .iter()
            .map(|(r, s)| {
                let mut v: Vec<Basic> = r.coefficients.clone();
                v.resize(na, Basic::zero());
                let mut w: Vec<Basic> = s.coefficients.clone();
                w.resize(nb, Basic::zero());
                v.extend(w);
                v
            })
            .collect();
        let size: usize = na + nb;
        let matrix: Vec<Vec<Basic>> = (0..size)
            .map(|i| vectors.iter().map(|v| v[i].clone()).collect())
            .collect();
        for relation in nullspace(&matrix, vectors.len())? {
            let last: Basic = relation.last().expect("nonempty").clone();
            if last != Basic::zero() {
                let coefficients: Vec<Basic> = relation
                    .iter()
                    .map(|c| normal(&div(c, &last)?))
                    .collect::<PolyResult<_>>()?;
                return Ok(LinearOperator::new(&coefficients, &x));
            }
        }
        power = d.mul(&power)?;
    }
}

/// The falling factorial `d (d - 1) ... (d - i + 1)` (`_falling`).
fn falling(d: &Basic, i: usize) -> R {
    let factors: Vec<Basic> = (0..i).map(|j| sub(d, &int(j as i64))).collect::<PolyResult<_>>()?;
    product(&factors)
}

/// The polynomial in `d` whose nonnegative integer roots bound the degree
/// of the polynomial solutions (`_indicial_at_infinity`).
pub(crate) fn indicial_at_infinity(l: &LinearOperator) -> PolyResult<UPoly> {
    let x: &Basic = &l.x;
    let d: Basic = dummy("d");
    let mut data: Vec<(usize, i64, Basic)> = Vec::new();
    for (i, c) in l.coefficients.iter().enumerate() {
        if *c == Basic::zero() {
            continue;
        }
        let p: UPoly = UPoly::of(c, x)?;
        data.push((i, p.degree() - i as i64, p.lc()));
    }
    let top: i64 = data.iter().map(|(_, s, _)| *s).max().unwrap_or(0);
    let mut terms: Vec<Basic> = Vec::new();
    for (i, shift, lc) in &data {
        if *shift == top {
            terms.push(mul(&[lc.clone(), falling(&d, *i)?])?);
        }
    }
    UPoly::of(&sum(&terms)?, &d)
}

fn integer_roots_where(p: &UPoly, keep: &dyn Fn(i64) -> bool) -> PolyResult<Vec<i64>> {
    let mut found: Vec<i64> = root_multiplicities(p)?
        .iter()
        .filter_map(|(r, _)| integer_of(r))
        .filter(|v| keep(*v))
        .collect();
    found.sort();
    Ok(found)
}

/// A basis of the polynomial solutions of `L(y) = 0`
/// (`polynomial_solutions`).
pub fn polynomial_solutions(l: &LinearOperator) -> PolyResult<Vec<Basic>> {
    let l: LinearOperator = polynomial_operator(l)?;
    let x: &Basic = &l.x;
    let indicial: UPoly = indicial_at_infinity(&l)?;
    if indicial.is_zero() {
        return Ok(Vec::new());
    }
    let degrees: Vec<i64> = integer_roots_where(&indicial, &|v| v >= 0)?;
    let Some(bound) = degrees.last().copied() else {
        return Ok(Vec::new());
    };
    let unknowns: Vec<Basic> = (0..=bound).map(|i| dummy(&format!("c{}", i))).collect();
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in unknowns.iter().enumerate() {
        terms.push(mul(&[c.clone(), pow(x, &int(i as i64))?])?);
    }
    let ansatz: Basic = sum(&terms)?;
    let image: Basic = expand(&l.apply(&ansatz)?)?;
    let rows: Vec<Basic> = match UPoly::of(&image, x) {
        Ok(p) => p.all_coeffs(),
        Err(PolyError::Value(_)) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let matrix: Vec<Vec<Basic>> = rows
        .iter()
        .map(|row| {
            unknowns
                .iter()
                .map(|u| crate::core::expr::coeff(row, u))
                .collect::<PolyResult<_>>()
        })
        .collect::<PolyResult<_>>()?;
    let mut solutions: Vec<Basic> = Vec::new();
    for vector in nullspace(&matrix, unknowns.len())? {
        let mut parts: Vec<Basic> = Vec::new();
        for (i, v) in vector.iter().enumerate() {
            parts.push(mul(&[v.clone(), pow(x, &int(i as i64))?])?);
        }
        solutions.push(sum(&parts)?);
    }
    Ok(solutions)
}

/// The order of `p` at the roots of `q` and the cofactor (`_order_at`).
pub(crate) fn order_at(p: &UPoly, q: &UPoly) -> PolyResult<(i64, UPoly)> {
    let mut order: i64 = 0;
    let mut p: UPoly = p.clone();
    while !p.is_zero() {
        let (quotient, remainder) = p.div(q)?;
        if !remainder.is_zero() {
            break;
        }
        p = quotient;
        order += 1;
    }
    Ok((order, p))
}

/// The indicial polynomial at the roots of the irreducible `q`
/// (`_indicial_at`).
pub(crate) fn indicial_at(l: &LinearOperator, q: &UPoly) -> PolyResult<UPoly> {
    let x: &Basic = &l.x;
    let m: Basic = dummy("m");
    let c: Basic = dummy("c");
    let q_c: UPoly = UPoly::of(&replaced(&q.as_expr()?, &[(x.clone(), c.clone())])?, &c)?;
    let derivative: Basic = q.diff()?.as_expr()?;
    let mut orders: Vec<(usize, i64, Basic)> = Vec::new();
    for (i, coefficient) in l.coefficients.iter().enumerate() {
        if *coefficient == Basic::zero() {
            continue;
        }
        let (order, p) = order_at(&UPoly::of(coefficient, x)?, q)?;
        let cofactor: Basic = mul(&[p.as_expr()?, pow(&derivative, &int(order))?])?;
        let value: Basic = if q.degree() > 1 {
            UPoly::of(&replaced(&cofactor, &[(x.clone(), c.clone())])?, &c)?
                .rem(&q_c)?
                .as_expr()?
        } else {
            let coefficients: Vec<Basic> = q.all_coeffs();
            let root: Basic = neg(&div(&coefficients[1], &coefficients[0])?)?;
            normal(&replaced(&cofactor, &[(x.clone(), root)])?)?
        };
        orders.push((i, order - i as i64, value));
    }
    let lowest: i64 = orders.iter().map(|(_, s, _)| *s).min().unwrap_or(0);
    let mut terms: Vec<Basic> = Vec::new();
    for (i, shift, value) in &orders {
        if *shift == lowest {
            terms.push(mul(&[value.clone(), falling(&m, *i)?])?);
        }
    }
    let mut expression: Basic = sum(&terms)?;
    if q.degree() > 1 {
        expression = resultant(&expression, &q_c.as_expr()?, &c)?;
    }
    UPoly::of(&expression, &m)
}

/// The irreducible factors of positive degree of the leading coefficient
/// (`_places`).
pub(crate) fn places(leading: &Basic, x: &Basic) -> PolyResult<Vec<(UPoly, i64)>> {
    let mut result: Vec<(UPoly, i64)> = Vec::new();
    for (f, multiplicity) in factor_list(leading, x)? {
        let q: UPoly = UPoly::of(&f, x)?;
        if q.degree() > 0 {
            result.push((q, multiplicity));
        }
    }
    Ok(result)
}

fn denominator_bound(l: &LinearOperator) -> R {
    let mut result: Basic = Basic::one();
    for (q, _) in places(l.leading_coefficient(), &l.x)? {
        let indicial: UPoly = indicial_at(l, &q)?;
        if indicial.is_zero() {
            continue;
        }
        let negative: Vec<i64> = integer_roots_where(&indicial, &|v| v < 0)?;
        if let Some(first) = negative.first() {
            result = mul(&[result, pow(&q.as_expr()?, &int(-first))?])?;
        }
    }
    Ok(result)
}

/// A basis of the rational solutions of `L(y) = 0` (`rational_solutions`).
pub fn rational_solutions(l: &LinearOperator) -> PolyResult<Vec<Basic>> {
    let l: LinearOperator = polynomial_operator(l)?;
    let q: Basic = denominator_bound(&l)?;
    if q == Basic::one() {
        return polynomial_solutions(&l);
    }
    let m: LinearOperator = l.transformed(&pow(&q, &int(-1))?)?;
    polynomial_solutions(&m)?.iter().map(|p| normal(&div(p, &q)?)).collect()
}

// ---------------------------------------------------------------------------
// hyperexponential solutions

/// Whether every finite singular point is regular singular
/// (`_is_fuchsian`).
pub fn is_fuchsian(l: &LinearOperator) -> PolyResult<bool> {
    let n: usize = l.order();
    for (q, multiplicity) in places(l.leading_coefficient(), &l.x)? {
        for (i, c) in l.coefficients[..n].iter().enumerate() {
            if *c == Basic::zero() {
                continue;
            }
            let (order, _) = order_at(&UPoly::of(c, &l.x)?, &q)?;
            if order < multiplicity - (n - i) as i64 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn rat_key(r: &Rat) -> (Int, Int) {
    (crate::numbers::numer(r), crate::numbers::denom(r))
}

/// The rational local exponents at the roots of `q`, and whether they are
/// all the exponents there (`_exponents_status`).
fn exponents_status(l: &LinearOperator, q: &UPoly) -> PolyResult<(Vec<Rat>, bool)> {
    let indicial: UPoly = indicial_at(l, q)?;
    if indicial.is_zero() {
        return Ok((Vec::new(), false));
    }
    let found: Vec<(Basic, usize)> = root_multiplicities(&indicial)?;
    let mut rational: Vec<Rat> = Vec::new();
    for (r, _) in &found {
        if let Some(v) = rational_of(r) {
            if !rational.contains(&v) {
                rational.push(v);
            }
        }
    }
    rational.sort_by_key(rat_key);
    let total: i64 = found.iter().map(|(_, m)| *m as i64).sum();
    let mut complete: bool = total == indicial.degree() && rational.len() == found.len();
    if q.degree() > 1 && rational.iter().any(|e| !integer_difference(e, &rational[0])) {
        complete = false;
    }
    Ok((rational, complete))
}

fn integer_difference(a: &Rat, b: &Rat) -> bool {
    (a.clone() - b.clone()).is_int()
}

/// Candidate leading terms `c x**s` of `rho = y'/y` at infinity from the
/// Newton polygon, and whether every root was found
/// (`_newton_polygon_parts`).
fn newton_polygon_parts(l: &LinearOperator) -> PolyResult<(Vec<Basic>, bool)> {
    let x: &Basic = &l.x;
    let mut points: Vec<(i64, i64, Basic)> = Vec::new();
    for (i, c) in l.coefficients.iter().enumerate() {
        if *c != Basic::zero() {
            let p: UPoly = UPoly::of(c, x)?;
            points.push((i as i64, p.degree(), p.lc()));
        }
    }
    if points.len() < 2 {
        return Ok((Vec::new(), true));
    }
    // upper convex hull from the left
    let mut hull: Vec<(i64, i64, Basic)> = Vec::new();
    for point in &points {
        while hull.len() >= 2 {
            let (i1, d1) = (hull[hull.len() - 2].0, hull[hull.len() - 2].1);
            let (i2, d2) = (hull[hull.len() - 1].0, hull[hull.len() - 1].1);
            let (i3, d3) = (point.0, point.1);
            if (d2 - d1) * (i3 - i1) <= (d3 - d1) * (i2 - i1) {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(point.clone());
    }
    let mut candidates: Vec<Basic> = Vec::new();
    let mut complete: bool = true;
    let t: Basic = dummy("t");
    for pair in hull.windows(2) {
        let (i1, d1, i2, d2) = (pair[0].0, pair[0].1, pair[1].0, pair[1].1);
        let slope: Rat = Rat::from(d2 - d1) / Rat::from(i2 - i1);
        let s: Rat = -slope.clone();
        if s < Rat::ZERO || !s.is_int() {
            continue;
        }
        let mut edge_terms: Vec<Basic> = Vec::new();
        for (i, d, lc) in &points {
            if Rat::from(d - d1) == slope.clone() * Rat::from(i - i1) {
                edge_terms.push(mul(&[lc.clone(), pow(&t, &int(*i))?])?);
            }
        }
        let characteristic: UPoly = UPoly::of(&sum(&edge_terms)?, &t)?;
        let found: Vec<(Basic, usize)> = root_multiplicities(&characteristic)?;
        if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != characteristic.degree() {
            complete = false;
        }
        let s_int: i64 = i64::try_from(&crate::numbers::numer(&s)).unwrap_or(0);
        for (root, _) in found {
            if root != Basic::zero() && !root.has(x) {
                candidates.push(mul(&[root, pow(x, &int(s_int))?])?);
            } else if root != Basic::zero() {
                complete = false;
            }
        }
    }
    Ok((candidates, complete))
}

/// The antiderivative of a polynomial in `x` without constant
/// (`_polynomial_integral`).
fn polynomial_integral(p: &Basic, x: &Basic) -> R {
    let terms: BTreeMap<i64, Basic> = UPoly::of(p, x)?.terms;
    let mut parts: Vec<Basic> = Vec::new();
    for (k, c) in terms.iter().rev() {
        parts.push(div(&mul(&[c.clone(), pow(x, &int(k + 1))?])?, &int(k + 1))?);
    }
    sum(&parts)
}

/// The polynomial parts `P` of `rho` at infinity (`_exponential_parts`).
fn exponential_parts(l: &LinearOperator, depth: usize, below: Option<i64>) -> PolyResult<(Vec<Basic>, bool)> {
    let x: &Basic = &l.x;
    let mut parts: Vec<Basic> = vec![Basic::zero()];
    let (mut leading_terms, mut complete) = newton_polygon_parts(l)?;
    if let Some(below) = below {
        let mut kept: Vec<Basic> = Vec::new();
        for t in leading_terms {
            if UPoly::of(&t, x)?.degree() < below {
                kept.push(t);
            }
        }
        leading_terms = kept;
    }
    if depth > 4 {
        return Ok((parts, complete && leading_terms.is_empty()));
    }
    for leading in leading_terms {
        let exponent: Basic = polynomial_integral(&leading, x)?;
        let m: LinearOperator = l.transformed(&exp(&exponent)?)?;
        let (lower_parts, lower_complete) = exponential_parts(&m, depth + 1, Some(UPoly::of(&leading, x)?.degree()))?;
        complete = complete && lower_complete;
        for lower in lower_parts {
            let candidate: Basic = add(&[leading.clone(), lower])?;
            if !parts.contains(&candidate) {
                parts.push(candidate);
            }
        }
    }
    Ok((parts, complete))
}

/// `L` with polynomial coefficients (`_polynomial_operator`).
fn polynomial_operator(l: &LinearOperator) -> PolyResult<LinearOperator> {
    if l.coefficients.iter().all(|c| is_polynomial(c, &l.x)) {
        return Ok(l.clone());
    }
    l.primitive()
}

/// Hyperexponential solutions (`hyperexponential_solutions`).
pub fn hyperexponential_solutions(l: &LinearOperator, max_combinations: usize) -> PolyResult<Vec<Basic>> {
    Ok(hyperexponential_search(l, max_combinations)?.0)
}

/// The hyperexponential solutions and whether the search was exhaustive
/// (`hyperexponential_search`).
pub fn hyperexponential_search(l: &LinearOperator, max_combinations: usize) -> PolyResult<(Vec<Basic>, bool)> {
    let x: Basic = l.x.clone();
    let l: LinearOperator = polynomial_operator(l)?;
    let mut solutions: Vec<Basic> = Vec::new();
    let (parts, mut complete) = exponential_parts(&l, 0, None)?;
    // the exponential parts at the irregular finite singular points do not
    // depend on the one at infinity
    let (finite_parts, finite_complete) = finite_exponential_parts(&l, max_combinations)?;
    complete = complete && finite_complete;
    for part in &parts {
        for finite in &finite_parts {
            let (m, factor_part) = if *part == Basic::zero() && *finite == Basic::zero() {
                (l.clone(), Basic::one())
            } else {
                let factor_part: Basic = exp(&add(&[polynomial_integral(part, &x)?, finite.clone()])?)?;
                (polynomial_operator(&l.transformed(&factor_part)?)?, factor_part)
            };
            let (found, found_complete) = hyperexponential_regular(&m, max_combinations)?;
            complete = complete && found_complete;
            for s in found {
                let candidate: Basic = mul(&[factor_part.clone(), s])?;
                if !solutions.contains(&candidate) {
                    solutions.push(candidate);
                }
            }
        }
    }
    Ok((independent(&solutions, &x)?, complete))
}

/// The exponential parts at the irregular finite singular points
/// (`irregular.finite_exponential_parts`).
fn finite_exponential_parts(l: &LinearOperator, max_combinations: usize) -> PolyResult<(Vec<Basic>, bool)> {
    crate::solvers::irregular::finite_exponential_parts(l, max_combinations)
}

/// The ansatz with the local exponents at the finite singular points
/// (`_hyperexponential_regular`).
fn hyperexponential_regular(l: &LinearOperator, max_combinations: usize) -> PolyResult<(Vec<Basic>, bool)> {
    let x: &Basic = &l.x;
    let mut found_places: Vec<(Basic, i64, Vec<Rat>)> = Vec::new();
    let mut complete: bool = true;
    for (q, _) in places(l.leading_coefficient(), x)? {
        let (exponents, exponents_complete) = exponents_status(l, &q)?;
        complete = complete && exponents_complete;
        if exponents.is_empty() {
            return Ok((Vec::new(), complete));
        }
        let mut representatives: Vec<Rat> = Vec::new();
        for e in exponents {
            if !representatives.iter().any(|r| integer_difference(&e, r)) {
                representatives.push(e);
            }
        }
        found_places.push((q.as_expr()?, q.degree(), representatives));
    }
    let at_infinity: UPoly = indicial_at_infinity(l)?;
    let d: Basic = at_infinity.x.clone();
    let t: Basic = dummy("t");
    let mut combinations: Vec<Vec<Rat>> = vec![Vec::new()];
    for (_, _, exponents) in &found_places {
        let mut next: Vec<Vec<Rat>> = Vec::new();
        for c in &combinations {
            for e in exponents {
                let mut n: Vec<Rat> = c.clone();
                n.push(e.clone());
                next.push(n);
            }
        }
        combinations = next;
    }
    if combinations.len() > max_combinations {
        combinations.truncate(max_combinations);
        complete = false;
    }
    let mut solutions: Vec<Basic> = Vec::new();
    for combination in combinations {
        let mut degree: Rat = Rat::ZERO;
        for ((_, k, _), e) in found_places.iter().zip(&combination) {
            degree += e.clone() * Rat::from(*k);
        }
        let degree_expr: Basic = Basic::from_rational(degree);
        // y ~ x**(degree + deg Q) at infinity
        if !at_infinity.is_zero() {
            let shifted: Basic = replaced(&at_infinity.as_expr()?, &[(d.clone(), add(&[degree_expr, t.clone()])?)])?;
            if integer_roots_where(&UPoly::of(&shifted, &t)?, &|v| v >= 0)?.is_empty() {
                continue;
            }
        }
        let mut factors: Vec<Basic> = Vec::new();
        for ((f, _, _), e) in found_places.iter().zip(&combination) {
            factors.push(pow(f, &Basic::from_rational(e.clone()))?);
        }
        let factor: Basic = product(&factors)?;
        let m: LinearOperator = l.transformed(&factor)?;
        for p in polynomial_solutions(&m)? {
            solutions.push(mul(&[factor.clone(), p])?);
        }
    }
    Ok((independent(&solutions, x)?, complete))
}

// ---------------------------------------------------------------------------
// reduction of order

/// The operator `M` of order `n - 1` with `L(y1 * w) = y1 * M(w')`
/// (`reduce_order_linear`).
pub fn reduce_order_linear(l: &LinearOperator, y1: &Basic) -> PolyResult<LinearOperator> {
    let t: LinearOperator = l.transformed(y1)?;
    if normal(&t.coefficients[0])? != Basic::zero() {
        return Err(PolyError::Value(format!("{} is not a solution", y1)));
    }
    Ok(LinearOperator::new(&t.coefficients[1..], &l.x))
}

/// Drop the solutions which are constant multiples of earlier ones
/// (`_independent`).
fn independent(solutions: &[Basic], x: &Basic) -> PolyResult<Vec<Basic>> {
    let mut kept: Vec<Basic> = Vec::new();
    for s in solutions {
        if *s == Basic::zero() {
            continue;
        }
        let mut proportional_to_kept: bool = false;
        for k in &kept {
            if proportional(s, k, x)? {
                proportional_to_kept = true;
                break;
            }
        }
        if !proportional_to_kept {
            kept.push(s.clone());
        }
    }
    Ok(kept)
}

/// Whether `s/k` is constant (`_proportional`).
fn proportional(s: &Basic, k: &Basic, x: &Basic) -> PolyResult<bool> {
    let ratio: Basic = normal(&div(s, k)?)?;
    if !ratio.has(x) {
        return Ok(true);
    }
    if has_integral(&ratio) {
        return Ok(false);
    }
    Ok(attempt(|| crate::core::expr::is_constant(&ratio, std::slice::from_ref(x)))?.flatten() == Some(true))
}

/// Independent solutions of a homogeneous linear equation with rational
/// coefficients (`dsolve_linear`).
pub fn dsolve_linear(
    equation: &Basic,
    f: &Basic,
    use_kovacic: bool,
    use_dsolve: bool,
    special_values: bool,
) -> PolyResult<Vec<Basic>> {
    let l: LinearOperator = LinearOperator::from_equation(equation, f)?;
    let found: Vec<Basic> = solve_operator(&l, f, use_kovacic, use_dsolve, true)?;
    if !special_values {
        return Ok(found);
    }
    let solve_at = |at: &Basic| -> PolyResult<Option<Vec<Basic>>> {
        Ok(Some(dsolve_linear(at, f, use_kovacic, use_dsolve, true)?))
    };
    crate::solvers::ode_cases::basis_cases(&found, &crate::solvers::ode_cases::as_zero(equation)?, f, &solve_at)
}

/// The equation `L(y) = 0` in the function `y` of `x` (`_equation`).
fn equation_of(l: &LinearOperator) -> R {
    let y: Basic = function_of("y", &l.x)?;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in l.coefficients.iter().enumerate() {
        terms.push(mul(&[c.clone(), diff(&y, &l.x, i as u64)?])?);
    }
    sum(&terms)
}

pub(crate) fn solve_operator(
    l: &LinearOperator,
    f: &Basic,
    use_kovacic: bool,
    use_dsolve: bool,
    factorize: bool,
) -> PolyResult<Vec<Basic>> {
    let x: &Basic = &l.x;
    let n: usize = l.order();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        // y' = -a_0/a_1 y
        let rho: Basic = normal(&neg(&div(&l.coefficients[0], &l.coefficients[1])?)?)?;
        return Ok(vec![
            match crate::integrals::indefinite::closed_antiderivative(&rho, x, true, false, None)? {
                Some(integral) => exp(&integral)?,
                None => exp(&crate::integrals::integral::integral(&rho, x, None)?)?,
            },
        ]);
    }
    let mut found: Vec<Basic> = Vec::new();
    found.extend(rational_solutions(l)?);
    if found.len() < n {
        found.extend(hyperexponential_solutions(l, 200)?);
    }
    found = independent(&found, x)?;
    if found.len() < n && n >= 3 && factorize {
        // the right factor's solutions, then variation of parameters for
        // the solutions of the left factor
        let by_factors: Vec<Basic> = solve_by_factorization(l, f, use_kovacic, use_dsolve)?;
        if by_factors.len() > found.len() {
            found = by_factors;
        }
    }
    if found.len() == 1 && n == 2 {
        let reduced: Option<Vec<Basic>> = by_reduction(l, f, &found, use_kovacic, use_dsolve, factorize)?;
        if let Some(reduced) = reduced {
            if reduced.len() == 2 && !has_integral(&reduced[1]) {
                return Ok(reduced);
            }
        }
    }
    if found.len() < n && n == 2 && use_kovacic {
        let equation: Basic = equation_of(l)?;
        let y: Basic = function_of("y", x)?;
        let liouvillian: Option<Vec<Basic>> =
            attempt(|| crate::solvers::kovacic::dsolve_kovacic(&equation, &y, false))?.flatten();
        if let Some(liouvillian) = liouvillian.filter(|v| !v.is_empty()) {
            let mut all: Vec<Basic> = found.clone();
            all.extend(liouvillian);
            found = independent(&all, x)?;
        }
    }
    if found.len() < n && n == 2 {
        let equation: Basic = equation_of(l)?;
        let y: Basic = function_of("y", x)?;
        let special: Option<Vec<Basic>> = attempt(|| special_solutions(&equation, &y))?.flatten();
        if let Some(special) = special.filter(|v| !v.is_empty()) {
            let mut all: Vec<Basic> = found.clone();
            all.extend(special);
            found = independent(&all, x)?;
        }
    }
    let mut reduced_by_constant: bool = false;
    if !found.is_empty() && found.len() < n {
        reduced_by_constant = !found[0].has(x);
        match by_reduction(l, f, &found, use_kovacic, use_dsolve, factorize)? {
            None => return Ok(found),
            Some(reduced) => found = reduced,
        }
    }
    // an equation without y reduced by the constant solution is the
    // equation in y' which dsolve has been handed already
    if found.len() < n && use_dsolve && !(reduced_by_constant && l.coefficients[0] == Basic::zero()) {
        let equation: Basic = equation_of(l)?;
        let y: Basic = function_of("y", x)?;
        let parts: Option<Vec<Basic>> = attempt(|| solution_parts(&equation, &y))?.flatten();
        for part in parts.unwrap_or_default() {
            let mut all: Vec<Basic> = found.clone();
            all.push(part);
            found = independent(&all, x)?;
        }
    }
    found.truncate(n);
    Ok(found)
}

/// The special-function solutions (`special.special_solutions`).
fn special_solutions(equation: &Basic, y: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    crate::solvers::special::special_solutions(equation, y)
}

/// The parts of the solutions of `dsolve` (`dsolve.solution_parts`).
fn solution_parts(equation: &Basic, y: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    crate::solvers::dsolve::solution_parts(equation, y)
}

/// `found` extended by the solutions of the operator reduced by the first
/// solution (`_by_reduction`).
fn by_reduction(
    l: &LinearOperator,
    f: &Basic,
    found: &[Basic],
    use_kovacic: bool,
    use_dsolve: bool,
    factorize: bool,
) -> PolyResult<Option<Vec<Basic>>> {
    let x: &Basic = &l.x;
    let y1: &Basic = &found[0];
    let m: LinearOperator = match reduce_order_linear(l, y1) {
        Ok(m) => m,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let reduced: Vec<Basic> = solve_operator(&m, f, use_kovacic, use_dsolve, factorize)?;
    let mut found: Vec<Basic> = found.to_vec();
    for v in reduced {
        let w: Basic = match crate::integrals::indefinite::closed_antiderivative(&v, x, false, true, None)? {
            Some(w) => w,
            // the quadrature is left unevaluated
            None => crate::integrals::integral::integral(&v, x, None)?,
        };
        let product: Basic = mul(&[y1.clone(), w])?;
        let entry: Basic = if is_rational_function(&product, x) {
            normal(&product)?
        } else {
            product
        };
        found.push(entry);
        found = independent(&found, x)?;
    }
    Ok(Some(found))
}

/// Independent solutions of `L` through a factorisation into at least two
/// factors (`_solve_by_factorization`).
fn solve_by_factorization(
    l: &LinearOperator,
    f: &Basic,
    use_kovacic: bool,
    use_dsolve: bool,
) -> PolyResult<Vec<Basic>> {
    use crate::solvers::factorization::{factor_operator, solve_factored};
    let limit: Option<f64> = timeout().map(|t| 4.0 * t);
    let factorization = crate::timeout::attempt_within(limit, || factor_operator(l))?;
    let Some(factorization) = factorization.filter(|fac| fac.factors.len() >= 2) else {
        return Ok(Vec::new());
    };
    let solver = |m: &LinearOperator| -> PolyResult<Vec<Basic>> {
        solve_operator(&m.primitive()?, f, use_kovacic, use_dsolve, false)
    };
    independent(&solve_factored(&factorization.factors, &solver)?, &l.x)
}
