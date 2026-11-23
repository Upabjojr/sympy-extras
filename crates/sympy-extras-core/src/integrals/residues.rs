//! Port of `sympy_extras/integrals/residues.py`: definite integrals by the
//! residue theorem. Four families, each with the conditions under which
//! the contour integral equals the real one:
//!
//! 1. rational functions over the real line (`deg Q >= deg P + 2`, no real
//!    pole): `2*pi*I` times the residues in the upper half plane;
//! 2. Fourier integrals `R(x)*exp(I*k*x)` (and with `cos`, `sin`) over the
//!    real line or, for an even integrand, the half line, by Jordan's
//!    lemma;
//! 3. `x**alpha*log(x)**n*R(x)` over `(0, oo)` by the keyhole contour
//!    around the branch cut of `z**alpha` (`0 < arg z < 2*pi`);
//! 4. rational functions of `sin(x)` and `cos(x)` over `k` periods,
//!    through `z = exp(I*x)` and the residues inside the unit circle, split
//!    into the cases of the sign of a parameter (or into the intervals
//!    between the values where a pole crosses the circle) when the
//!    assumptions leave the place of a pole open.
//!
//! The poles are the roots of the denominators (`roots`, the irrational
//! constants of the coefficients held by placeholders), their place
//! decided exactly for numbers, by `ask` under the assumptions for
//! parameters, or (with the numerical checks on) at a sample of the
//! parameters, the value then verified numerically.

#![allow(clippy::mutable_key_type)]
#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeSet, HashMap};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;
use crate::random::PyRandom;

type R = PolyResult<Basic>;
type OptCV = PolyResult<Option<ConditionalValue>>;

/// The trigonometric functions (SymPy's `TrigonometricFunction`).
const TRIGONOMETRIC: [Function; 6] = [
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
];

/// The most values of a parameter at which a pole may cross the unit
/// circle for which the cases are written (`_MAX_CROSSINGS`).
const MAX_CROSSINGS: usize = 6;

/// The samples of the parameters at which a value placed by sampling is
/// checked (`_SAMPLES`).
const SAMPLES: usize = 2;

// -- small helpers

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn neg_oo() -> Basic {
    Basic::atom(Head::NegativeInfinity)
}
fn i_unit() -> Basic {
    Basic::imaginary_unit()
}
fn pi() -> Basic {
    Basic::pi()
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
fn inverse(a: &Basic) -> R {
    pow(a, &int(-1))
}
fn neg(a: &Basic) -> R {
    mul(&[int(-1), a.clone()])
}
fn sub(a: &Basic, b: &Basic) -> R {
    add(&[a.clone(), neg(b)?])
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
fn ge(a: &Basic, b: &Basic) -> R {
    crate::core::relational::ge(a, b)
}
fn le(a: &Basic, b: &Basic) -> R {
    crate::core::relational::le(a, b)
}
fn eq(a: &Basic, b: &Basic) -> R {
    crate::core::relational::eq(a, b)
}
fn ne(a: &Basic, b: &Basic) -> R {
    crate::core::relational::ne(a, b)
}
fn and_(args: &[Basic]) -> Basic {
    crate::core::logic::and_(args)
}
fn or_(args: &[Basic]) -> Basic {
    crate::core::logic::or_(args)
}
fn true_() -> Basic {
    crate::core::logic::true_()
}
fn is_true(b: &Basic) -> bool {
    *b == true_()
}
fn is_false(b: &Basic) -> bool {
    *b == crate::core::logic::false_()
}
fn reals() -> Basic {
    Basic::atom(Head::Reals)
}
fn element(x: &Basic, domain: &Basic) -> R {
    crate::assumptions::facts::element(x, domain)
}
fn ask(query: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    crate::assumptions::ask::ask(query, assumptions, None)
}
fn has_head(e: &Basic, f: &dyn Fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| f(n.head()))
}
fn has_function(e: &Basic, fs: &[Function]) -> bool {
    has_head(e, &|h| matches!(h, Head::Function(f) if fs.contains(f)))
}
fn has_infinity(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::Infinity | Head::NegativeInfinity))
}
/// Whether `e` contains `nan` or `zoo` (`_is_bad`).
pub(crate) fn is_bad(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::NaN | Head::ComplexInfinity))
}
fn is_application(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(_) | Head::Undefined(_))
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn is_number(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}
fn args_of(e: &Basic, sum: bool) -> Vec<Basic> {
    if (sum && e.is_add()) || (!sum && e.is_mul()) {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn budget() -> Option<f64> {
    crate::settings::settings().timeout
}
fn eighth() -> Option<f64> {
    budget().map(|t| t / 8.0)
}
fn within<T>(seconds: Option<f64>, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(seconds, f)
}
/// The polynomial errors, `NotImplementedError`, `ValueError` and
/// `TypeError` which the Python code catches; a part of the port which is
/// missing goes through.
fn polynomial_failure(e: &PolyError) -> bool {
    match e {
        PolyError::NotImplemented(m) => !m.starts_with(crate::error::NOT_PORTED_PREFIX),
        PolyError::Polys(..) | PolyError::HeuristicGCDFailed(_) | PolyError::Value(_) | PolyError::Type(_) => true,
        _ => false,
    }
}
fn caught<T>(r: PolyResult<T>) -> PolyResult<Option<T>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if polynomial_failure(&e) => Ok(None),
        Err(e) => Err(e),
    }
}
/// A dummy with a name of its own (`_fresh`).
pub(crate) fn fresh(name: &str, real: bool) -> Basic {
    let label: String = format!("{}_", name);
    if real {
        crate::integrals::definite::fresh(&label, &[(Predicate::real, true)])
    } else {
        crate::integrals::definite::fresh(&label, &[])
    }
}
fn dummy(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    crate::core::symbol::Dummy::new(name, assumptions).0
}
fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(data) | Head::Dummy(data, _) => data.name.clone(),
        _ => s.to_string(),
    }
}
/// `e` with the replacements made and the rebuilt nodes evaluated
/// (`_replace`, `_replace_boolean`).
pub(crate) fn replace(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    crate::core::function::xreplace_evaluated(e, rule)
}
pub(crate) fn replace1(e: &Basic, old: &Basic, new: &Basic) -> R {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(old.clone(), new.clone());
    replace(e, &rule)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn expand_complex(e: &Basic) -> R {
    crate::core::function::expand_with(
        e,
        true,
        crate::core::function::Hints {
            complex: true,
            ..crate::core::function::Hints::DEFAULT
        },
    )
}
fn expand_trig(e: &Basic) -> R {
    crate::core::function::expand_with(
        e,
        true,
        crate::core::function::Hints {
            mul: false,
            multinomial: false,
            power_base: false,
            power_exp: false,
            log: false,
            force: false,
            complex: false,
            trig: true,
            func: false,
        },
    )
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    crate::core::expr::as_numer_denom(e)
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::derivative(e, x)
}
fn rewrite_to(e: &Basic, target: &str) -> R {
    crate::core::rewriting::rewrite(e, &[], target, &crate::core::rewriting::RewriteHints::default())
}
fn size(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}
fn atoms_pow(e: &Basic) -> Vec<Basic> {
    e.atoms(Some(&|n: &Basic| n.is_pow()))
}
fn without(mut s: BTreeSet<Basic>, x: &Basic) -> BTreeSet<Basic> {
    s.remove(x);
    s
}

/// The symbols among `symbols`, sorted by name and by the order of
/// creation of the dummies of one name (`_sorted_symbols`).
fn sorted_symbols(symbols: &BTreeSet<Basic>) -> Vec<Basic> {
    let mut found: Vec<(String, i64, Basic)> = Vec::new();
    for s in symbols {
        match s.head() {
            Head::Symbol(data) => found.push((data.name.clone(), -1, s.clone())),
            Head::Dummy(data, index) => found.push((data.name.clone(), *index as i64, s.clone())),
            _ => {}
        }
    }
    found.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
    found.into_iter().map(|(_, _, s)| s).collect()
}

// ---------------------------------------------------------------------------
// Polynomials with irrational constants in their coefficients

/// The irrational constants of expressions replaced by dummies, so that
/// the typed polynomials take them as parameters (`_Placeholders`).
struct Placeholders {
    variable: Basic,
    gaussian: bool,
    forward: HashMap<Basic, Basic>,
    back: HashMap<Basic, Basic>,
}

impl Placeholders {
    fn new(variable: &Basic, gaussian: bool) -> Placeholders {
        Placeholders {
            variable: variable.clone(),
            gaussian,
            forward: HashMap::new(),
            back: HashMap::new(),
        }
    }

    fn hide(&mut self, e: &Basic) -> R {
        if e.as_rational().is_some() || e.is_symbol() {
            return Ok(e.clone());
        }
        if self.gaussian && *e == i_unit() {
            return Ok(e.clone());
        }
        if e.is_add() {
            let mut parts: Vec<Basic> = Vec::new();
            for a in e.args() {
                parts.push(self.hide(a)?);
            }
            return add(&parts);
        }
        if e.is_mul() {
            let mut parts: Vec<Basic> = Vec::new();
            for a in e.args() {
                parts.push(self.hide(a)?);
            }
            return mul(&parts);
        }
        if e.is_pow() && integer_of(&e.args()[1]).is_some() {
            let base: Basic = self.hide(&e.args()[0])?;
            return pow(&base, &e.args()[1]);
        }
        if e.has(&self.variable) {
            return Ok(e.clone());
        }
        if e.is_pow() {
            if let Some(r) = e.args()[1].as_rational() {
                let p = crate::numbers::numer(r);
                if p != crate::numbers::Int::from(1) {
                    // a**(p/q) is (a**(1/q))**p, as SymPy's generators are
                    let q = crate::numbers::denom(r);
                    let root: Basic = pow(&e.args()[0], &Basic::rational(crate::numbers::Int::from(1), q)?)?;
                    let held: Basic = self.placeholder(&root);
                    return pow(&held, &Basic::rational(p, crate::numbers::Int::from(1))?);
                }
            }
        }
        if e.is_function(Function::Exp) {
            // exp(k*t) is exp(t)**k for an integer k
            let (coefficient, rest) = crate::core::expr::as_coeff_mul(&e.args()[0]);
            if let Some(k) = integer_of(&coefficient) {
                if k != 0 && k != 1 {
                    let base: Basic = crate::functions::elementary::exponential::exp(&rest)?;
                    let held: Basic = self.placeholder(&base);
                    return pow(&held, &coefficient);
                }
            }
        }
        Ok(self.placeholder(e))
    }

    fn placeholder(&mut self, e: &Basic) -> Basic {
        if let Some(found) = self.forward.get(e) {
            return found.clone();
        }
        let found: Basic = fresh("c", false);
        self.forward.insert(e.clone(), found.clone());
        self.back.insert(found.clone(), e.clone());
        found
    }

    fn restore(&self, e: &Basic) -> R {
        replace(e, &self.back)
    }
}

/// The coefficients of the polynomial `e` in `s` by degree, the zero ones
/// left out; `None` when `e` is not a polynomial in `s` (`_coefficients`).
fn coefficients(e: &Basic, s: &Basic) -> PolyResult<Option<Vec<(usize, Basic)>>> {
    let mut collected: Vec<(usize, Vec<Basic>)> = Vec::new();
    for term in args_of(&expand(e)?, true) {
        if term == Basic::zero() {
            continue;
        }
        let (coefficient, power) = crate::core::expr::as_independent(&term, s, false)?;
        let degree: usize = if power == Basic::one() {
            0
        } else if power == *s {
            1
        } else if power.is_pow() && power.args()[0] == *s && integer_of(&power.args()[1]).is_some_and(|k| k > 0) {
            integer_of(&power.args()[1]).unwrap_or(0) as usize
        } else {
            return Ok(None);
        };
        match collected.iter_mut().find(|(d, _)| *d == degree) {
            Some((_, parts)) => parts.push(coefficient),
            None => collected.push((degree, vec![coefficient])),
        }
    }
    let mut result: Vec<(usize, Basic)> = Vec::new();
    for (degree, parts) in collected {
        let total: Basic = add(&parts)?;
        if total != Basic::zero() {
            result.push((degree, total));
        }
    }
    Ok(Some(result))
}

/// A polynomial in `x` with coefficients free of `x`, dense, the highest
/// degree first and no leading zero (`_Polynomial`).
#[derive(Clone, Debug)]
pub struct Polynomial {
    pub coefficients: Vec<Basic>,
    pub x: Basic,
}

impl Polynomial {
    pub fn new(coefficients: Vec<Basic>, x: &Basic) -> Polynomial {
        let start: usize = coefficients
            .iter()
            .position(|c| *c != Basic::zero())
            .unwrap_or(coefficients.len());
        Polynomial {
            coefficients: coefficients[start..].to_vec(),
            x: x.clone(),
        }
    }

    /// The polynomial of an expression polynomial in `x` (`from_expr`).
    pub fn from_expr(e: &Basic, x: &Basic) -> PolyResult<Option<Polynomial>> {
        let Some(found) = coefficients(e, x)? else {
            return Ok(None);
        };
        if found.iter().any(|(_, c)| c.has(x)) {
            return Ok(None);
        }
        let Some(top) = found.iter().map(|(d, _)| *d).max() else {
            return Ok(Some(Polynomial::new(Vec::new(), x)));
        };
        let mut dense: Vec<Basic> = Vec::new();
        for d in (0..=top).rev() {
            dense.push(
                found
                    .iter()
                    .find(|(k, _)| *k == d)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(Basic::zero),
            );
        }
        Ok(Some(Polynomial::new(dense, x)))
    }

    pub fn degree(&self) -> i64 {
        self.coefficients.len() as i64 - 1
    }

    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The nonzero coefficients.
    pub fn coeffs(&self) -> Vec<Basic> {
        self.coefficients
            .iter()
            .filter(|c| **c != Basic::zero())
            .cloned()
            .collect()
    }

    pub fn as_expr(&self) -> R {
        let n: i64 = self.degree();
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in self.coefficients.iter().enumerate() {
            terms.push(mul(&[c.clone(), pow(&self.x, &int(n - i as i64))?])?);
        }
        add(&terms)
    }

    pub fn diff(&self) -> PolyResult<Polynomial> {
        let n: i64 = self.degree();
        let mut out: Vec<Basic> = Vec::new();
        for (i, c) in self
            .coefficients
            .iter()
            .take(self.coefficients.len().saturating_sub(1))
            .enumerate()
        {
            out.push(mul(&[int(n - i as i64), c.clone()])?);
        }
        Ok(Polynomial::new(out, &self.x))
    }

    /// The value at `value` (Horner's scheme).
    pub fn eval(&self, value: &Basic) -> R {
        let mut total: Basic = Basic::zero();
        for c in &self.coefficients {
            total = add(&[mul(&[total, value.clone()])?, c.clone()])?;
        }
        Ok(total)
    }

    /// The polynomial multiplied by `x**k`.
    pub fn times_power(&self, k: usize) -> Polynomial {
        let mut c: Vec<Basic> = self.coefficients.clone();
        c.extend(std::iter::repeat(Basic::zero()).take(k));
        Polynomial::new(c, &self.x)
    }

    /// The quotient by `(x - point)**m` by synthetic division for a root
    /// of multiplicity at least `m` (`quotient_by_root`).
    pub fn quotient_by_root(&self, point: &Basic, m: usize) -> PolyResult<Polynomial> {
        let mut current: Vec<Basic> = self.coefficients.clone();
        for _ in 0..m {
            let mut quotient: Vec<Basic> = Vec::new();
            let mut carry: Basic = Basic::zero();
            for c in current.iter().take(current.len().saturating_sub(1)) {
                carry = expand(&add(&[c.clone(), mul(&[point.clone(), carry])?])?)?;
                quotient.push(carry.clone());
            }
            current = quotient;
        }
        Ok(Polynomial::new(current, &self.x))
    }

    /// The typed `Poly` of the polynomial with its placeholders
    /// (`to_poly`); `None` when it does not take it.
    fn to_poly(&self) -> PolyResult<Option<(crate::polys::polytools::Poly, Placeholders)>> {
        let mut hidden: Placeholders = Placeholders::new(&self.x, false);
        let mut e: Basic = hidden.hide(&self.as_expr()?)?;
        if atoms_pow(&e)
            .iter()
            .any(|p| is(&p.args()[1], Predicate::negative) == Some(true))
        {
            // the negative powers of the placeholders cleared: a constant
            // factor, which leaves the roots alone
            e = numer_denom(&cancel(&e)?)?.0;
        }
        Ok(caught(crate::polys::polytools::Poly::new(&e, std::slice::from_ref(&self.x)))?.map(|p| (p, hidden)))
    }
}

/// A rational function `numerator/denominator` of `x` (`_Fraction`).
#[derive(Clone, Debug)]
pub struct Fraction {
    pub numerator: Polynomial,
    pub denominator: Polynomial,
    pub x: Basic,
}

impl Fraction {
    pub fn as_expr(&self) -> R {
        mul(&[self.numerator.as_expr()?, inverse(&self.denominator.as_expr()?)?])
    }

    /// `deg Q - deg P`.
    pub fn excess(&self) -> i64 {
        self.denominator.degree() - self.numerator.degree()
    }
}

/// `f` as a quotient of polynomials in `x` with coefficients free of `x`
/// and the common factors cancelled (`rational_function`).
pub fn rational_function(f: &Basic, x: &Basic) -> PolyResult<Option<Fraction>> {
    if !f.has(x) {
        return Ok(None);
    }
    if crate::core::expr::polynomial_like(f, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        != Some(true)
    {
        return Ok(None);
    }
    let mut hidden: Placeholders = Placeholders::new(x, true);
    let h: Basic = hidden.hide(f)?;
    let Some(reduced) = within(budget(), || cancel(&h))? else {
        return Ok(None);
    };
    let (numerator, denominator) = numer_denom(&reduced)?;
    let (Some(p), Some(q)) = (
        Polynomial::from_expr(&numerator, x)?,
        Polynomial::from_expr(&denominator, x)?,
    ) else {
        return Ok(None);
    };
    let mut pc: Vec<Basic> = Vec::new();
    for c in &p.coefficients {
        pc.push(hidden.restore(c)?);
    }
    let mut qc: Vec<Basic> = Vec::new();
    for c in &q.coefficients {
        qc.push(hidden.restore(c)?);
    }
    if pc.iter().chain(qc.iter()).any(|c| c.has(x)) {
        return Ok(None);
    }
    let p: Polynomial = Polynomial::new(pc, x);
    let q: Polynomial = Polynomial::new(qc, x);
    if q.is_zero() {
        return Ok(None);
    }
    Ok(Some(Fraction {
        numerator: p,
        denominator: q,
        x: x.clone(),
    }))
}

/// A pole with its multiplicity; `numeric` for a root of a numerical
/// polynomial found as a `CRootOf` (`_Pole`).
#[derive(Clone, Debug)]
pub struct Pole {
    pub point: Basic,
    pub multiplicity: usize,
    pub numeric: bool,
}

fn pole(point: Basic, multiplicity: usize, numeric: bool) -> Pole {
    Pole {
        point,
        multiplicity,
        numeric,
    }
}

/// `q` split into its factors over `ZZ_I` when `I` is among its
/// coefficients (`_gaussian_factors`).
fn gaussian_factors(q: &Polynomial) -> PolyResult<Vec<(Polynomial, usize)>> {
    let whole = || vec![(q.clone(), 1usize)];
    if !q.coefficients.iter().any(|c| c.has(&i_unit())) {
        return Ok(whole());
    }
    let e: Basic = q.as_expr()?;
    let Some(factored) = within(budget(), || crate::polys::polytools::factor(&e, &[], false, true))? else {
        return Ok(whole());
    };
    let mut result: Vec<(Polynomial, usize)> = Vec::new();
    for part in args_of(&factored, false) {
        let (mut base, mut multiplicity) = (part.clone(), 1usize);
        if part.is_pow() {
            if let Some(k) = integer_of(&part.args()[1]).filter(|k| *k > 0) {
                base = part.args()[0].clone();
                multiplicity = k as usize;
            }
        }
        if !base.has(&q.x) {
            continue;
        }
        let Some(p) = Polynomial::from_expr(&base, &q.x)? else {
            return Ok(whole());
        };
        result.push((p, multiplicity));
    }
    if result.iter().map(|(p, m)| p.degree() * *m as i64).sum::<i64>() != q.degree() {
        return Ok(whole());
    }
    Ok(result)
}

/// The roots of `q` with multiplicities by `roots` (`_roots_in_radicals`);
/// `None` when the polynomials do not take `q`.
fn roots_in_radicals(q: &Polynomial) -> PolyResult<Option<Vec<(Basic, usize)>>> {
    let mut result: Vec<(Basic, usize)> = Vec::new();
    for (part, multiplicity) in gaussian_factors(q)? {
        let Some((poly, hidden)) = part.to_poly()? else {
            return Ok(None);
        };
        let e: Basic = poly.as_expr()?;
        let Some(found) = caught(crate::polys::polyroots::roots(
            &e,
            std::slice::from_ref(&q.x),
            true,
            true,
            None,
        ))?
        else {
            return Ok(None);
        };
        for (r, m) in found.iter() {
            let point: Basic = hidden.restore(r)?;
            match result.iter_mut().find(|(p, _)| *p == point) {
                Some((_, k)) => *k += m * multiplicity,
                None => result.push((point, m * multiplicity)),
            }
        }
    }
    Ok(Some(result))
}

/// Whether a root in radicals is too large to work with (`_unwieldy`).
fn unwieldy(root: &Basic) -> PolyResult<bool> {
    Ok(size(root)? > 60)
}

/// The roots of `q` with multiplicities, or `None` when they cannot all
/// be found (`poles`).
pub fn poles(q: &Polynomial) -> PolyResult<Option<Vec<Pole>>> {
    if q.degree() <= 0 {
        return Ok(Some(Vec::new()));
    }
    let parametric: bool = q.coeffs().iter().any(|c| !c.free_symbols().is_empty());
    let found: Option<Vec<(Basic, usize)>> = within(budget(), || roots_in_radicals(q))?.flatten();
    if let Some(found) = &found {
        if found.iter().map(|(_, m)| *m as i64).sum::<i64>() == q.degree() {
            let mut wieldy: bool = true;
            if !parametric {
                for (r, _) in found {
                    if unwieldy(r)? {
                        wieldy = false;
                        break;
                    }
                }
            }
            if wieldy {
                return Ok(Some(found.iter().map(|(r, m)| pole(r.clone(), *m, false)).collect()));
            }
        }
    }
    if parametric || !q.coefficients.iter().all(|c| c.as_rational().is_some()) {
        return Ok(None);
    }
    let Some(poly) = caught(crate::polys::polytools::Poly::new(
        &q.as_expr()?,
        std::slice::from_ref(&q.x),
    ))?
    else {
        return Ok(None);
    };
    let Some((_, factors)) = caught(poly.factor_list())? else {
        return Ok(None);
    };
    let mut result: Vec<Pole> = Vec::new();
    for (g, e) in factors {
        let degree: i64 = g.degree(0)?;
        if degree <= 0 {
            continue;
        }
        if degree <= 2 {
            let Some(low) = roots_in_radicals(&Polynomial::new(g.all_coeffs()?, &q.x))? else {
                return Ok(None);
            };
            result.extend(low.into_iter().map(|(r, m)| pole(r, m * e, false)));
            continue;
        }
        let ge: Basic = g.as_expr()?;
        let Some(found_roots) = within(budget(), || {
            crate::polys::complexrootof::all_roots(&ge, Some(&q.x), true)
        })?
        else {
            return Ok(None);
        };
        result.extend(found_roots.into_iter().map(|r| pole(r, e, true)));
    }
    if result.iter().map(|p| p.multiplicity as i64).sum::<i64>() != q.degree() {
        return Ok(None);
    }
    Ok(Some(result))
}

// ---------------------------------------------------------------------------
// Deciding where a pole lies

/// Answers questions about a pole: exactly, under the assumptions, or
/// (with the numerical checks on) at a sample of the parameters
/// (`_Locator`).
pub(crate) struct Locator {
    assumptions: Vec<Basic>,
    sampled: bool,
    sample: Option<Vec<(Basic, Basic)>>,
    no_sample: bool,
    slow: bool,
}

impl Locator {
    pub(crate) fn new(assumptions: &[Basic]) -> Locator {
        Locator {
            assumptions: assumptions.to_vec(),
            sampled: false,
            sample: None,
            no_sample: false,
            slow: false,
        }
    }

    pub(crate) fn decide(&mut self, query: &Basic) -> PolyResult<Option<bool>> {
        if is_true(query) {
            return Ok(Some(true));
        }
        if is_false(query) {
            return Ok(Some(false));
        }
        if query.free_symbols().is_empty() {
            if let Some(verdict) = numerically(query)? {
                return Ok(Some(verdict));
            }
        }
        let checks: bool = crate::settings::settings().numerical_checks;
        if checks && has_radicals(query) {
            if let Some(verdict) = self.at_sample(query)? {
                return Ok(Some(verdict));
            }
        }
        if self.slow {
            return Ok(None);
        }
        let assumptions: Vec<Basic> = self.assumptions.clone();
        let Some(verdict) = within(budget(), || ask(query, &assumptions))? else {
            self.slow = true;
            return Ok(None);
        };
        if verdict.is_some() {
            return Ok(verdict);
        }
        if !checks {
            return Ok(None);
        }
        self.at_sample(query)
    }

    fn at_sample(&mut self, q: &Basic) -> PolyResult<Option<bool>> {
        let Some(sample) = self.sample_of(&sorted_symbols(&q.free_symbols()))? else {
            return Ok(None);
        };
        let rule: HashMap<Basic, Basic> = sample.into_iter().collect();
        let at: Basic = replace(q, &rule)?;
        if is_true(&at) {
            self.sampled = true;
            return Ok(Some(true));
        }
        if is_false(&at) {
            self.sampled = true;
            return Ok(Some(false));
        }
        let verdict: Option<bool> = ask(&at, &[])?;
        if verdict.is_some() {
            self.sampled = true;
        }
        Ok(verdict)
    }

    /// A point of the parameter space satisfying the assumptions
    /// (`sample`).
    fn sample_of(&mut self, symbols: &[Basic]) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
        if self.no_sample {
            return Ok(None);
        }
        if let Some(found) = &self.sample {
            if symbols.iter().all(|s| found.iter().any(|(k, _)| k == s)) {
                return Ok(Some(found.clone()));
            }
        }
        let facts = crate::assumptions::facts::Facts::new(&self.assumptions, None, symbols)?;
        let mut all: BTreeSet<Basic> = symbols.iter().cloned().collect();
        all.extend(facts.formula.free_symbols());
        let wanted: Vec<Basic> = sorted_symbols(&all);
        let names: Vec<String> = wanted.iter().map(symbol_name).collect();
        let distinct: bool = names.iter().collect::<BTreeSet<_>>().len() == names.len();
        let mut plain: HashMap<Basic, Basic> = HashMap::new();
        for (s, name) in wanted.iter().zip(&names) {
            plain.insert(s.clone(), if distinct { dummy(name, &[]) } else { fresh(name, false) });
        }
        let mut items: Vec<Basic> = vec![replace(&facts.formula, &plain)?];
        for s in &wanted {
            let p: &Basic = &plain[s];
            if is(s, Predicate::positive) == Some(true) {
                items.push(gt(p, &Basic::zero())?);
            } else if is(s, Predicate::negative) == Some(true) {
                items.push(lt(p, &Basic::zero())?);
            } else if is(s, Predicate::nonnegative) == Some(true) {
                items.push(ge(p, &Basic::zero())?);
            } else if is(s, Predicate::nonzero) == Some(true) {
                items.push(ne(p, &Basic::zero())?);
            }
        }
        let formula: Basic = and_(&items);
        if is_true(&formula) {
            // nothing constrains the parameters: a sample says nothing
            self.no_sample = true;
            return Ok(None);
        }
        let variables: Vec<Basic> = wanted.iter().map(|s| plain[s].clone()).collect();
        let witnesses = crate::assumptions::sat::find_instance(&formula, &variables, Some(&reals()), &[], 1)?;
        let Some(first) = witnesses.and_then(|w| w.into_iter().next()) else {
            self.no_sample = true;
            return Ok(None);
        };
        let mut sample: Vec<(Basic, Basic)> = Vec::new();
        for s in &wanted {
            let value: Basic = first
                .iter()
                .find(|(k, _)| *k == plain[s])
                .map(|(_, v)| v.clone())
                .unwrap_or(Basic::rational(3.into(), 2.into())?);
            sample.push((s.clone(), value));
        }
        self.sample = Some(sample.clone());
        Ok(Some(sample))
    }

    /// The sign of the imaginary part of a pole: `1`, `-1`, `0` (real) or
    /// `None` (`imaginary_sign`).
    pub(crate) fn imaginary_sign(&mut self, p: &Pole) -> PolyResult<Option<i32>> {
        let point: &Basic = &p.point;
        if p.numeric {
            if is(point, Predicate::extended_real) == Some(true) {
                return Ok(Some(0));
            }
            let Some(value) = crate::core::evalf::numeric_value(point, 100) else {
                return Ok(None);
            };
            let (_, imaginary) = crate::core::evalf::parts(&value);
            return Ok(Some(if imaginary > crate::core::evalf::Real::ZERO {
                1
            } else {
                -1
            }));
        }
        let imaginary: Basic = crate::functions::elementary::complexes::im(point)?;
        if self.decide(&gt(&imaginary, &Basic::zero())?)? == Some(true) {
            return Ok(Some(1));
        }
        if self.decide(&lt(&imaginary, &Basic::zero())?)? == Some(true) {
            return Ok(Some(-1));
        }
        if self.decide(&element(point, &reals())?)? == Some(true) {
            return Ok(Some(0));
        }
        Ok(None)
    }

    /// Whether `|point| < 1`; `None` on the circle or undecided
    /// (`inside_unit_circle`).
    fn inside_unit_circle(&mut self, p: &Pole) -> PolyResult<Option<bool>> {
        let point: &Basic = &p.point;
        if p.numeric || point.free_symbols().is_empty() {
            let Some(value) = crate::numeric::reliable_value(point, 30, &[]) else {
                return Ok(None);
            };
            let (a, b) = crate::core::evalf::parts(&value);
            let a = a.with_precision(120).value();
            let b = b.with_precision(120).value();
            let modulus = (&a * &a + &b * &b).sqrt();
            let distance: f64 = (modulus - crate::core::evalf::Real::ONE).to_f64().value();
            if distance.abs() < 1e-20 {
                return Ok(None);
            }
            return Ok(Some(distance < 0.0));
        }
        let modulus: Basic = crate::functions::elementary::complexes::abs_(point)?;
        match self.decide(&lt(&modulus, &Basic::one())?)? {
            Some(true) => return Ok(Some(true)),
            Some(false) => {
                return Ok(if self.decide(&gt(&modulus, &Basic::one())?)? == Some(true) {
                    Some(false)
                } else {
                    None
                })
            }
            None => {}
        }
        if self.decide(&element(point, &reals())?)? == Some(true) {
            let square: Basic = pow(point, &int(2))?;
            if self.decide(&lt(&square, &Basic::one())?)? == Some(true) {
                return Ok(Some(true));
            }
            if self.decide(&gt(&square, &Basic::one())?)? == Some(true) {
                return Ok(Some(false));
            }
        }
        Ok(None)
    }
}

/// A relation between numbers decided by evaluation; an equality is never
/// confirmed this way, only refuted (`_numerically`).
fn numerically(q: &Basic) -> PolyResult<Option<bool>> {
    let Head::Relational(op) = q.head() else {
        return Ok(None);
    };
    let (lhs, rhs) = (&q.args()[0], &q.args()[1]);
    let difference: Basic = sub(lhs, rhs)?;
    let Some(value) = crate::numeric::reliable_value(&difference, 50, &[]) else {
        return Ok(None);
    };
    let (real, imaginary) = crate::core::evalf::parts(&value);
    if !imaginary.repr().significand().is_zero() {
        return Ok(None);
    }
    let r: f64 = real.to_f64().value();
    if r.abs() < 1e-40 {
        return Ok(None);
    }
    let positive: bool = r > 0.0;
    Ok(match op {
        RelOp::Gt | RelOp::Ge => Some(positive),
        RelOp::Lt | RelOp::Le => Some(!positive),
        RelOp::Eq => Some(false),
        RelOp::Ne => Some(true),
    })
}

/// Whether the question involves non-polynomial functions of the
/// parameters (`_has_radicals`).
fn has_radicals(q: &Basic) -> bool {
    if atoms_pow(q).iter().any(|n| integer_of(&n.args()[1]).is_none()) {
        return true;
    }
    !q.atoms(Some(&is_application)).is_empty()
}

// ---------------------------------------------------------------------------
// Residues

/// `dh/dx`, the symbol `branch` standing for `log(x)` (`_derivative`).
fn derivative(h: &Basic, x: &Basic, branch: Option<&Basic>) -> R {
    let mut found: Basic = diff(h, x)?;
    if let Some(b) = branch {
        if h.has(b) {
            found = add(&[found, mul(&[diff(h, b)?, inverse(x)?])?])?;
        }
    }
    Ok(found)
}

/// The residue of `fraction(z)*factor(z)` at the pole, `factor`
/// holomorphic there (`_residue`).
pub(crate) fn residue(
    fraction: &Fraction,
    p: &Pole,
    factor: &Basic,
    z: &Basic,
    branch: Option<&Basic>,
) -> PolyResult<Option<Basic>> {
    let x: &Basic = &fraction.x;
    let point: &Basic = &p.point;
    let g: Basic = replace1(factor, z, x)?;
    if p.multiplicity == 1 {
        let slope: Basic = fraction.denominator.diff()?.eval(point)?;
        let at: Basic = replace1(&g, x, point)?;
        return Ok(Some(mul(&[fraction.numerator.eval(point)?, at, inverse(&slope)?])?));
    }
    let m: usize = p.multiplicity;
    let quotient: Polynomial = fraction.denominator.quotient_by_root(point, m)?;
    let mut h: Basic = mul(&[fraction.numerator.as_expr()?, inverse(&quotient.as_expr()?)?, g])?;
    for _ in 0..m - 1 {
        h = derivative(&h, x, branch)?;
    }
    let scale: Basic = inverse(&crate::functions::combinatorial::factorials::factorial(&int(
        m as i64 - 1
    ))?)?;
    let mut value: Basic = mul(&[replace1(&h, x, point)?, scale.clone()])?;
    if is_bad(&value) {
        let found: Option<Basic> = within(budget(), || crate::core::limits::limit(&h, x, point, "+"))?;
        let Some(found) = found else {
            return Ok(None);
        };
        if matches!(found.head(), Head::Limit) || is_bad(&found) || has_infinity(&found) {
            return Ok(None);
        }
        value = mul(&[found, scale])?;
    }
    Ok(Some(value))
}

/// `log|point| + I*theta` with `0 < theta < 2*pi` (`_branch_log`).
fn branch_log(point: &Basic, imaginary_sign: i32) -> R {
    use crate::functions::elementary::exponential::log;
    if imaginary_sign == 0 {
        return add(&[log(&neg(point)?)?, mul(&[i_unit(), pi()])?]);
    }
    let mut theta: Basic = crate::functions::elementary::argument::arg(point)?;
    if imaginary_sign < 0 {
        theta = add(&[theta, mul(&[int(2), pi()])?])?;
    }
    add(&[
        log(&crate::functions::elementary::complexes::abs_(point)?)?,
        mul(&[i_unit(), theta])?,
    ])
}

fn known_real(s: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    Ok(is(s, Predicate::extended_real) == Some(true) || ask(&element(s, &reals())?, assumptions)? == Some(true))
}

/// The value simplified, with the imaginary parts which cancel for real
/// parameters removed, each simplification under an eighth of the time
/// limit (`_tidy`).
pub(crate) fn tidy(value: &Basic, assumptions: &[Basic]) -> R {
    let mut result: Basic = value.clone();
    let parameters: Vec<Basic> = sorted_symbols(&result.free_symbols());
    let mut reals_rule: HashMap<Basic, Basic> = HashMap::new();
    let mut back: HashMap<Basic, Basic> = HashMap::new();
    for s in &parameters {
        if known_real(s, assumptions)? {
            let d: Basic = fresh(&symbol_name(s), true);
            reals_rule.insert(s.clone(), d.clone());
            back.insert(d, s.clone());
        }
    }
    let unit: Basic = i_unit();
    if !reals_rule.is_empty() && result.has(&unit) {
        let replaced: Basic = replace(&result, &reals_rule)?;
        if let Some(expanded) = within(eighth(), || expand_complex(&replaced))? {
            if !expanded.has(&unit) {
                result = replace(&expanded, &back)?;
            }
        }
    }
    let current: Basic = result.clone();
    if let Some(simpler) = within(eighth(), || simplify(&cancel(&current)?))? {
        if size(&simpler)? <= size(&result)? {
            result = simpler;
        }
    }
    if result.has(&unit) {
        // the half-integer powers of a base negative under the assumptions
        // written through the positive one
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for node in atoms_pow(&result) {
            let (base, exponent) = (&node.args()[0], &node.args()[1]);
            let Some(r) = exponent.as_rational() else {
                continue;
            };
            if crate::numbers::denom(r) == crate::numbers::Int::from(2)
                && !base.free_symbols().is_empty()
                && ask(&lt(base, &Basic::zero())?, assumptions)? == Some(true)
            {
                let p: Basic = Basic::rational(crate::numbers::numer(r), crate::numbers::Int::from(1))?;
                rule.insert(node.clone(), mul(&[pow(&unit, &p)?, pow(&neg(base)?, exponent)?])?);
            }
        }
        if !rule.is_empty() {
            let rewritten: Basic = replace(&result, &rule)?;
            let simpler: Option<Basic> = within(eighth(), || simplify(&rewritten))?;
            result = match simpler {
                Some(s) if size(&s)? <= size(&rewritten)? => s,
                _ => rewritten,
            };
        }
    }
    let radical_of_parameters: bool = atoms_pow(&result).iter().any(|node| {
        node.args()[1].as_rational().is_some_and(|r| !r.is_int()) && !node.args()[0].free_symbols().is_empty()
    });
    if radical_of_parameters {
        // radicals of the parameters in a denominator rationalised first
        let current: Basic = result.clone();
        if let Some(rationalised) = within(eighth(), || {
            simplify(&crate::simplify::radsimp::radsimp(&current, true, 4)?)
        })? {
            if size(&rationalised)? < size(&result)? {
                result = rationalised;
            }
        }
    }
    if result.has(&unit) && reals_rule.is_empty() && parameters.is_empty() {
        let current: Basic = result.clone();
        if let Some(simpler) = within(eighth(), || simplify(&expand_complex(&current)?))? {
            if !simpler.has(&unit) {
                result = simpler;
            }
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// 1. and 2.: the real line

/// The sum of the residues of `fraction*factor` in the upper (or lower)
/// half plane; `None` with a real pole or an undecided one
/// (`_sum_residues`).
fn sum_residues(fraction: &Fraction, factor: &Basic, z: &Basic, locator: &mut Locator, upper: bool) -> R2 {
    let Some(found) = poles(&fraction.denominator)? else {
        return Ok(None);
    };
    let mut total: Basic = Basic::zero();
    for p in &found {
        let s: i32 = match locator.imaginary_sign(p)? {
            None | Some(0) => return Ok(None),
            Some(s) => s,
        };
        if (s > 0) == upper {
            let Some(value) = residue(fraction, p, factor, z, None)? else {
                return Ok(None);
            };
            total = add(&[total, value])?;
        }
    }
    Ok(Some(total))
}

type R2 = PolyResult<Option<Basic>>;

fn two_pi_i(e: &Basic) -> R {
    mul(&[int(2), pi(), i_unit(), e.clone()])
}

/// `Integral(R(x), (x, -oo, oo))` for a rational function with `deg Q >=
/// deg P + 2` and no real poles (`real_line_integral`).
pub fn real_line_integral(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptCV {
    let Some(fraction) = rational_function(f, x)? else {
        return Ok(None);
    };
    if fraction.excess() < 2 {
        return Ok(None);
    }
    let mut locator: Locator = Locator::new(assumptions);
    let z: Basic = fresh("z", false);
    let Some(total) = sum_residues(&fraction, &Basic::one(), &z, &mut locator, true)? else {
        return Ok(None);
    };
    let value: Basic = tidy(&two_pi_i(&total)?, assumptions)?;
    checked(
        ConditionalValue::new(value, true_()),
        f,
        x,
        &neg_oo(),
        &oo(),
        assumptions,
        &locator,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Exp,
    Cos,
    Sin,
}

/// `(R, k, kind)` when `f == R(x)*g(k*x)` with `g` one of `exp(I*.)`,
/// `cos`, `sin` (`_fourier_factor`).
fn fourier_factor(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Kind)>> {
    let mut rest: Vec<Basic> = Vec::new();
    let mut found: Option<(Basic, Kind)> = None;
    for e in args_of(f, false) {
        let kind: Option<Kind> = if e.is_function(Function::Exp) {
            Some(Kind::Exp)
        } else if e.is_function(Function::Cos) {
            Some(Kind::Cos)
        } else if e.is_function(Function::Sin) {
            Some(Kind::Sin)
        } else {
            None
        };
        let Some(kind) = kind.filter(|_| found.is_none() && e.has(x)) else {
            rest.push(e);
            continue;
        };
        let (coefficient, variable) = crate::core::expr::as_independent(&e.args()[0], x, false)?;
        if variable != *x {
            return Ok(None);
        }
        let mut k: Basic = coefficient;
        if kind == Kind::Exp {
            k = mul(&[k, inverse(&i_unit())?])?;
            let real_number: bool = is_number(&k)
                && crate::functions::elementary::complexes::im(&k)? == Basic::zero()
                && k != Basic::zero();
            if is(&k, Predicate::extended_real) != Some(true) && !real_number {
                return Ok(None);
            }
        }
        found = Some((k, kind));
    }
    let Some((k, kind)) = found else {
        return Ok(None);
    };
    Ok(Some((mul(&rest)?, k, kind)))
}

/// Real dummies for the parameters of `e`, or `None` when one is not
/// known to be real (`_real_parameters`).
fn real_parameters(e: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    let mut found: Vec<(Basic, Basic)> = Vec::new();
    for s in sorted_symbols(&e.free_symbols()) {
        if known_real(&s, assumptions)? {
            let d: Basic = fresh(&symbol_name(&s), true);
            found.push((s, d));
        } else {
            return Ok(None);
        }
    }
    Ok(Some(found))
}

/// The real (or imaginary) part of `e` for real parameters, when it comes
/// out free of `re`, `im` and `I` (`_part`).
pub(crate) fn part(e: &Basic, real: bool, assumptions: &[Basic]) -> R2 {
    let Some(pairs) = real_parameters(e, assumptions)? else {
        return Ok(None);
    };
    let forward: HashMap<Basic, Basic> = pairs.iter().cloned().collect();
    let back: HashMap<Basic, Basic> = pairs.iter().map(|(s, d)| (d.clone(), s.clone())).collect();
    let replaced: Basic = replace(e, &forward)?;
    let found: Option<Basic> = within(budget(), || {
        let projected: Basic = if real {
            crate::functions::elementary::complexes::re(&replaced)?
        } else {
            crate::functions::elementary::complexes::im(&replaced)?
        };
        expand_complex(&projected)
    })?;
    let Some(found) = found else {
        return Ok(None);
    };
    if has_function(&found, &[Function::Re, Function::Im]) || found.has(&i_unit()) {
        return Ok(None);
    }
    Ok(Some(replace(&found, &back)?))
}

/// `Integral(R(x)*exp(I*k*x), (x, -oo, oo))`, and with `cos(k*x)` or
/// `sin(k*x)`, by Jordan's lemma; with `half` the integral over `(0, oo)`
/// of an even integrand (`fourier_integral`).
pub fn fourier_integral(f: &Basic, x: &Basic, assumptions: &[Basic], half: bool) -> OptCV {
    let Some((r, k, kind)) = fourier_factor(f, x)? else {
        return Ok(None);
    };
    let Some(fraction) = rational_function(&r, x)? else {
        return Ok(None);
    };
    if fraction.excess() < 1 {
        return Ok(None);
    }
    if half {
        let mirrored: Basic = replace1(f, x, &neg(x)?)?;
        let even: Option<Basic> = within(budget(), || cancel(&sub(f, &mirrored)?))?;
        if even != Some(Basic::zero()) {
            return Ok(None);
        }
    }
    let mut locator: Locator = Locator::new(assumptions);
    let z: Basic = fresh("z", false);
    let mut condition: Basic = true_();
    let sign_k: i32 = if is(&k, Predicate::positive) == Some(true) {
        1
    } else if is(&k, Predicate::negative) == Some(true) {
        -1
    } else {
        match ask(&gt(&k, &Basic::zero())?, assumptions)? {
            Some(true) => 1,
            Some(false) if ask(&lt(&k, &Basic::zero())?, assumptions)? == Some(true) => -1,
            None => {
                condition = gt(&k, &Basic::zero())?;
                1
            }
            Some(false) => return Ok(None),
        }
    };
    let kappa: Basic = if sign_k > 0 { k.clone() } else { neg(&k)? };
    let exp = crate::functions::elementary::exponential::exp;
    let Some(upper) = sum_residues(
        &fraction,
        &exp(&mul(&[i_unit(), kappa.clone(), z.clone()])?)?,
        &z,
        &mut locator,
        true,
    )?
    else {
        return Ok(None);
    };
    let plus: Basic = two_pi_i(&upper)?;
    let lower_factor: Basic = exp(&mul(&[int(-1), i_unit(), kappa.clone(), z.clone()])?)?;
    let mut value: Basic;
    if kind == Kind::Exp {
        if sign_k > 0 {
            value = plus;
        } else {
            let Some(lower) = sum_residues(&fraction, &lower_factor, &z, &mut locator, false)? else {
                return Ok(None);
            };
            value = mul(&[int(-2), pi(), i_unit(), lower])?;
        }
    } else {
        // the real and imaginary parts of E(kappa) for real coefficients
        match part(&plus, kind == Kind::Cos, assumptions)? {
            Some(v) => value = v,
            None => {
                let Some(lower) = sum_residues(&fraction, &lower_factor, &z, &mut locator, false)? else {
                    return Ok(None);
                };
                let minus: Basic = mul(&[int(-2), pi(), i_unit(), lower])?;
                value = if kind == Kind::Cos {
                    mul(&[add(&[plus.clone(), minus])?, Basic::half()])?
                } else {
                    mul(&[sub(&plus, &minus)?, inverse(&mul(&[int(2), i_unit()])?)?])?
                };
            }
        }
        if kind == Kind::Sin && sign_k < 0 {
            value = neg(&value)?;
        }
    }
    if half {
        value = mul(&[value, Basic::half()])?;
    }
    let lower_bound: Basic = if half { Basic::zero() } else { neg_oo() };
    checked(
        ConditionalValue::new(tidy(&value, assumptions)?, condition),
        f,
        x,
        &lower_bound,
        &oo(),
        assumptions,
        &locator,
    )
}

// ---------------------------------------------------------------------------
// 3.: the keyhole contour

/// `(alpha, n, R)` when `f == x**alpha*log(x)**n*R(x)` (`_power_and_logs`).
fn power_and_logs(f: &Basic, x: &Basic) -> PolyResult<(Basic, usize, Basic)> {
    let mut alpha: Basic = Basic::zero();
    let mut n: usize = 0;
    let mut rest: Vec<Basic> = Vec::new();
    for e in args_of(f, false) {
        if e == *x {
            alpha = add(&[alpha, Basic::one()])?;
        } else if e.is_pow() && e.args()[0] == *x && !e.args()[1].has(x) {
            alpha = add(&[alpha, e.args()[1].clone()])?;
        } else if e.is_function(Function::Log) && e.args()[0] == *x {
            n += 1;
        } else if e.is_pow()
            && e.args()[0].is_function(Function::Log)
            && e.args()[0].args()[0] == *x
            && integer_of(&e.args()[1]).is_some_and(|k| k > 0)
        {
            n += integer_of(&e.args()[1]).unwrap_or(0) as usize;
        } else {
            rest.push(e);
        }
    }
    Ok((alpha, n, mul(&rest)?))
}

/// The sign of the imaginary part of a pole which is neither 0 nor on the
/// positive real axis (`_place_off_axis`).
fn place_off_axis(p: &Pole, locator: &mut Locator) -> PolyResult<Option<i32>> {
    if p.point == Basic::zero() {
        return Ok(None);
    }
    let Some(s) = locator.imaginary_sign(p)? else {
        return Ok(None);
    };
    if s == 0 {
        let real: Basic = crate::functions::elementary::complexes::re(&p.point)?;
        if locator.decide(&gt(&real, &Basic::zero())?)? != Some(false) {
            return Ok(None);
        }
    }
    Ok(Some(s))
}

/// `sum Res z**alpha*R(z)` over the poles off the positive real axis
/// (`_keyhole_sum`, `factor` `None`), or `-sum Res R(z)*L(z)`
/// (`_log_sum`, `alpha` `None`).
fn branch_sum(fraction: &Fraction, alpha: Option<&Basic>, locator: &mut Locator, z: &Basic) -> R2 {
    let Some(found) = poles(&fraction.denominator)? else {
        return Ok(None);
    };
    let branch: Basic = fresh("L", false);
    let factor: Basic = match alpha {
        Some(a) => crate::functions::elementary::exponential::exp(&mul(&[a.clone(), branch.clone()])?)?,
        None => branch.clone(),
    };
    let mut total: Basic = Basic::zero();
    for p in &found {
        let Some(s) = place_off_axis(p, locator)? else {
            return Ok(None);
        };
        let Some(value) = residue(fraction, p, &factor, z, Some(&branch))? else {
            return Ok(None);
        };
        total = add(&[total, replace1(&value, &branch, &branch_log(&p.point, s)?)?])?;
    }
    if alpha.is_none() {
        return Ok(Some(neg(&total)?));
    }
    Ok(Some(total))
}

/// The factor `x**m` of the numerator or the denominator moved into the
/// power (`_absorb_zero`).
fn absorb_zero(fraction: &Fraction, alpha: &Basic) -> PolyResult<(Fraction, Basic)> {
    let x: &Basic = &fraction.x;
    let mut alpha: Basic = alpha.clone();
    let mut p: Polynomial = fraction.numerator.clone();
    let mut q: Polynomial = fraction.denominator.clone();
    while p.degree() > 0 && p.coefficients[p.coefficients.len() - 1] == Basic::zero() {
        p = Polynomial::new(p.coefficients[..p.coefficients.len() - 1].to_vec(), x);
        alpha = add(&[alpha, Basic::one()])?;
    }
    while q.degree() > 0 && q.coefficients[q.coefficients.len() - 1] == Basic::zero() {
        q = Polynomial::new(q.coefficients[..q.coefficients.len() - 1].to_vec(), x);
        alpha = add(&[alpha, int(-1)])?;
    }
    Ok((
        Fraction {
            numerator: p,
            denominator: q,
            x: x.clone(),
        },
        alpha,
    ))
}

fn keyhole_factor(total: &Basic, alpha: &Basic) -> R {
    let rotation: Basic =
        crate::functions::elementary::exponential::exp(&mul(&[int(2), pi(), i_unit(), alpha.clone()])?)?;
    mul(&[
        int(2),
        pi(),
        i_unit(),
        total.clone(),
        inverse(&sub(&Basic::one(), &rotation)?)?,
    ])
}

/// `Integral(x**alpha*log(x)**n*R(x), (x, 0, oo))` for a rational `R`
/// without poles on the positive real axis, by the keyhole contour
/// (`keyhole_integral`).
pub fn keyhole_integral(f: &Basic, x: &Basic, assumptions: &[Basic]) -> OptCV {
    let (alpha, n, r) = power_and_logs(f, x)?;
    let Some(fraction) = rational_function(&r, x)? else {
        return Ok(None);
    };
    let (fraction, alpha) = absorb_zero(&fraction, &alpha)?;
    let mut locator: Locator = Locator::new(assumptions);
    let z: Basic = fresh("z", false);
    let lower_bound: Basic = gt(&alpha, &int(-1))?;
    let upper_bound: Basic = lt(&alpha, &int(fraction.excess() - 1))?;
    let condition: Basic = and_(&[lower_bound, upper_bound]);
    if is_false(&condition) {
        return Ok(None);
    }
    if n > 0 {
        return keyhole_with_logs(&fraction, &alpha, n, &mut locator, &z, &condition, f, x, assumptions);
    }
    if is(&alpha, Predicate::integer) == Some(true) {
        let Some(shift) = integer_of(&alpha) else {
            return Ok(None);
        };
        let (mut p, mut q) = (fraction.numerator.clone(), fraction.denominator.clone());
        if shift >= 0 {
            p = p.times_power(shift as usize);
        } else {
            q = q.times_power((-shift) as usize);
        }
        let folded: Fraction = Fraction {
            numerator: p,
            denominator: q,
            x: x.clone(),
        };
        if folded.excess() < 2 || shift < 0 {
            return Ok(None);
        }
        let Some(total) = branch_sum(&folded, None, &mut locator, &z)? else {
            return Ok(None);
        };
        return checked(
            ConditionalValue::new(tidy(&total, assumptions)?, true_()),
            f,
            x,
            &Basic::zero(),
            &oo(),
            assumptions,
            &locator,
        );
    }
    let Some(total) = branch_sum(&fraction, Some(&alpha), &mut locator, &z)? else {
        return Ok(None);
    };
    let value: Basic = keyhole_factor(&total, &alpha)?;
    checked(
        ConditionalValue::new(tidy(&value, assumptions)?, condition),
        f,
        x,
        &Basic::zero(),
        &oo(),
        assumptions,
        &locator,
    )
}

/// The power formula with a symbolic exponent differentiated `n` times,
/// the limit at an integer exponent (`_keyhole_with_logs`).
fn keyhole_with_logs(
    fraction: &Fraction,
    alpha: &Basic,
    n: usize,
    locator: &mut Locator,
    z: &Basic,
    condition: &Basic,
    f: &Basic,
    x: &Basic,
    assumptions: &[Basic],
) -> OptCV {
    let a: Basic = fresh("alpha", true);
    let Some(total) = branch_sum(fraction, Some(&a), locator, z)? else {
        return Ok(None);
    };
    let mut value: Basic = keyhole_factor(&total, &a)?;
    for _ in 0..n {
        value = diff(&value, &a)?;
    }
    if is(alpha, Predicate::integer) == Some(true) {
        let current: Basic = value.clone();
        let at: Option<Basic> = within(budget(), || crate::core::limits::limit(&current, &a, alpha, "+"))?;
        let Some(at) = at else {
            return Ok(None);
        };
        if matches!(at.head(), Head::Limit) || has_infinity(&at) {
            return Ok(None);
        }
        value = at;
    } else {
        value = replace1(&value, &a, alpha)?;
    }
    checked(
        ConditionalValue::new(tidy(&value, assumptions)?, condition.clone()),
        f,
        x,
        &Basic::zero(),
        &oo(),
        assumptions,
        locator,
    )
}

// ---------------------------------------------------------------------------
// 4.: a period

/// `Integral(f, (x, a, b))` over `k` periods, `b - a = 2*pi*k`, for a
/// rational function of `sin(x)` and `cos(x)` (`period_integral`).
pub fn period_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    let Some(k) = crate::integrals::definite::periods(a, b)? else {
        return Ok(None);
    };
    if !has_function(f, &TRIGONOMETRIC) {
        return Ok(None);
    }
    period_cases(f, x, a, b, k, assumptions, 2)
}

/// `period_integral` under the assumptions, with at most `splits`
/// parameters split by their signs (`_period_cases`).
fn period_cases(
    e: &Basic,
    x: &Basic,
    lower: &Basic,
    upper: &Basic,
    k: i64,
    assumptions: &[Basic],
    splits: usize,
) -> OptCV {
    let (found, undecided) = period_value(e, x, lower, upper, k, assumptions)?;
    if found.is_some() || splits == 0 {
        return Ok(found);
    }
    for s in &undecided {
        if ask(&gt(s, &Basic::zero())?, assumptions)?.is_some() || ask(&lt(s, &Basic::zero())?, assumptions)?.is_some()
        {
            continue;
        }
        if let Some(split) = sign_cases(e, x, lower, upper, k, assumptions, splits, s)? {
            return Ok(Some(split));
        }
    }
    let parameters: Vec<Basic> = sorted_symbols(&without(e.free_symbols(), x));
    if parameters.len() == 1 && undecided.contains(&parameters[0]) {
        return interval_cases(e, x, lower, upper, k, assumptions, &parameters[0]);
    }
    Ok(None)
}

/// A piece of the real line of the parameter: `(lower end, upper end,
/// value)`, a point being a piece whose ends are equal.
type Piece = (Basic, Basic, Option<Basic>);

/// The integral in the cases of the only parameter `s`, the real line cut
/// where a pole may cross the unit circle, neighbouring cases with equal
/// values joined (`_interval_cases`).
fn interval_cases(
    e: &Basic,
    x: &Basic,
    lower: &Basic,
    upper: &Basic,
    k: i64,
    assumptions: &[Basic],
    s: &Basic,
) -> OptCV {
    let Some(points) = crossings(e, x, s)? else {
        return Ok(None);
    };
    let mut pieces: Vec<Piece> = Vec::new();
    let mut ends: Vec<Basic> = vec![neg_oo()];
    ends.extend(points);
    ends.push(oo());
    for i in 0..ends.len() - 1 {
        let (lo, hi) = (&ends[i], &ends[i + 1]);
        if i > 0 {
            pieces.push((
                lo.clone(),
                lo.clone(),
                value_at(e, x, lower, upper, k, assumptions, s, lo)?,
            ));
        }
        let mut interval: Vec<Basic> = Vec::new();
        if *lo != neg_oo() {
            interval.push(gt(s, lo)?);
        }
        if *hi != oo() {
            interval.push(lt(s, hi)?);
        }
        let mut value: Option<Basic> = None;
        if ask(&and_(&interval), assumptions)? != Some(false) {
            let (found, _) = period_value(e, x, lower, upper, k, &with_facts(assumptions, &interval))?;
            if let Some(found) = found {
                if is_true(&found.condition) {
                    value = Some(found.value);
                }
            }
        }
        pieces.push((lo.clone(), hi.clone(), value));
    }
    // (start, start closed, end, end closed, value)
    let mut joined: Vec<(Basic, bool, Basic, bool, Basic)> = Vec::new();
    let mut adjacent: bool = false;
    for (lo, hi, value) in pieces {
        let Some(value) = value else {
            adjacent = false;
            continue;
        };
        if adjacent {
            let (start, start_closed, end, end_closed, current) = joined[joined.len() - 1].clone();
            let last: usize = joined.len() - 1;
            if lo == hi {
                let at_cut: Basic = replace1(&current, s, &lo)?;
                if !is_bad(&at_cut) && same(&at_cut, &value, &with_facts(assumptions, &[eq(s, &lo)?]))? {
                    joined[last] = (start, start_closed, lo.clone(), true, current);
                    continue;
                }
            } else if start == end {
                let at_cut: Basic = replace1(&value, s, &lo)?;
                if !is_bad(&at_cut) && same(&at_cut, &current, &with_facts(assumptions, &[eq(s, &lo)?]))? {
                    joined[last] = (start, true, hi.clone(), false, value);
                    continue;
                }
            } else if end_closed {
                let interval: Vec<Basic> = if hi == oo() {
                    vec![gt(s, &lo)?]
                } else {
                    vec![gt(s, &lo)?, lt(s, &hi)?]
                };
                if same(&value, &current, &with_facts(assumptions, &interval))? {
                    joined[last] = (start, start_closed, hi.clone(), false, current);
                    continue;
                }
            }
        }
        let point: bool = lo == hi;
        joined.push((lo, point, hi, point, value));
        adjacent = true;
    }
    if joined.is_empty() {
        return Ok(None);
    }
    let mut cases: Vec<(Basic, Basic)> = Vec::new();
    for (start, start_closed, end, end_closed, value) in joined {
        if start == end {
            cases.push((value, eq(s, &start)?));
            continue;
        }
        let mut bounds: Vec<Basic> = Vec::new();
        if start != neg_oo() {
            bounds.push(if start_closed { ge(s, &start)? } else { gt(s, &start)? });
        }
        if end != oo() {
            bounds.push(if end_closed { le(s, &end)? } else { lt(s, &end)? });
        }
        cases.push((value, and_(&bounds)));
    }
    if cases.len() == 1 {
        let (value, condition) = cases.remove(0);
        return Ok(Some(ConditionalValue::new(value, condition)));
    }
    let conditions: Vec<Basic> = cases.iter().map(|(_, c)| c.clone()).collect();
    Ok(Some(ConditionalValue::new(
        crate::functions::elementary::piecewise::piecewise(&cases)?,
        or_(&conditions),
    )))
}

/// The integral at `s = point`; `None` when it has no value found or the
/// problem is not defined there (`_value_at`).
fn value_at(
    e: &Basic,
    x: &Basic,
    lower: &Basic,
    upper: &Basic,
    k: i64,
    assumptions: &[Basic],
    s: &Basic,
    point: &Basic,
) -> R2 {
    if ask(&eq(s, point)?, assumptions)? == Some(false) {
        return Ok(None);
    }
    let at_point: Basic = replace1(e, s, point)?;
    if is_bad(&at_point) {
        return Ok(None);
    }
    if !at_point.has(x) {
        return Ok(Some(mul(&[at_point, sub(upper, lower)?])?));
    }
    let (found, _) = period_value(&at_point, x, lower, upper, k, assumptions)?;
    Ok(found.filter(|f| is_true(&f.condition)).map(|f| f.value))
}

/// The assumptions with more facts (`_with_facts`).
fn with_facts(assumptions: &[Basic], facts: &[Basic]) -> Vec<Basic> {
    let mut items: Vec<Basic> = assumptions.to_vec();
    items.extend(facts.iter().cloned());
    items
}

/// The integral in the cases `s > 0`, `s < 0` and `s = 0` of the real
/// parameter `s`, joined when they agree (`_sign_cases`).
fn sign_cases(
    e: &Basic,
    x: &Basic,
    lower: &Basic,
    upper: &Basic,
    k: i64,
    assumptions: &[Basic],
    splits: usize,
    s: &Basic,
) -> OptCV {
    let mut cases: Vec<(Basic, Basic)> = Vec::new();
    for (condition, sign) in [(gt(s, &Basic::zero())?, 1), (lt(s, &Basic::zero())?, -1)] {
        if ask(&condition, assumptions)? == Some(false) {
            continue;
        }
        let Some(value) = period_cases(
            e,
            x,
            lower,
            upper,
            k,
            &with_fact(assumptions, s, &condition, sign)?,
            splits - 1,
        )?
        else {
            return Ok(None);
        };
        if !is_true(&value.condition) {
            return Ok(None);
        }
        cases.push((value.value, condition));
    }
    let mut complete: bool = true;
    let at_zero: Basic = replace1(e, s, &Basic::zero())?;
    let zero_case: Basic = eq(s, &Basic::zero())?;
    if ask(&zero_case, assumptions)? == Some(false) {
    } else if !at_zero.has(x) {
        if is_bad(&at_zero) {
            // the problem is not defined at s = 0: no case there
            complete = false;
        } else {
            cases.push((mul(&[at_zero, sub(upper, lower)?])?, zero_case.clone()));
        }
    } else {
        let Some(value) = period_cases(&at_zero, x, lower, upper, k, assumptions, splits - 1)? else {
            return Ok(None);
        };
        if !is_true(&value.condition) {
            return Ok(None);
        }
        cases.push((value.value, zero_case.clone()));
    }
    if cases.is_empty() {
        return Ok(None);
    }
    let mut covered: Basic = if complete { true_() } else { ne(s, &Basic::zero())? };
    if !known_real(s, assumptions)? {
        covered = and_(&[covered, element(s, &reals())?]);
    }
    let first: Basic = cases[0].0.clone();
    let mut all_same: bool = true;
    for (value, c) in &cases[1..] {
        let other: Basic = if matches!(c.head(), Head::Relational(RelOp::Eq)) {
            replace1(&first, s, &Basic::zero())?
        } else {
            first.clone()
        };
        if !same(value, &other, &with_fact(assumptions, s, c, 1)?)? {
            all_same = false;
            break;
        }
    }
    if all_same {
        return Ok(Some(ConditionalValue::new(first, covered)));
    }
    Ok(Some(ConditionalValue::new(
        crate::functions::elementary::piecewise::piecewise(&cases)?,
        covered,
    )))
}

/// The assumptions with the sign `fact` of `s` added, and `Abs(s)` in
/// them written `sign*s` (`_with_fact`).
fn with_fact(assumptions: &[Basic], s: &Basic, fact: &Basic, sign: i64) -> PolyResult<Vec<Basic>> {
    let absolute: Basic = crate::functions::elementary::complexes::abs_(s)?;
    let signed: Basic = mul(&[int(sign), s.clone()])?;
    let mut items: Vec<Basic> = Vec::new();
    for item in assumptions {
        items.push(replace1(item, &absolute, &signed)?);
    }
    items.push(fact.clone());
    Ok(items)
}

/// Whether the two values are equal by `simplify`, the parameters the
/// assumptions make positive, negative or real taken as such (`_same`).
fn same(value: &Basic, other: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    let mut symbols: BTreeSet<Basic> = value.free_symbols();
    symbols.extend(other.free_symbols());
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for p in sorted_symbols(&symbols) {
        if is(&p, Predicate::positive) == Some(true) || is(&p, Predicate::negative) == Some(true) {
            continue;
        }
        let name: String = symbol_name(&p);
        if ask(&gt(&p, &Basic::zero())?, assumptions)? == Some(true) {
            rule.insert(p, dummy(&name, &[(Predicate::positive, true)]));
        } else if ask(&lt(&p, &Basic::zero())?, assumptions)? == Some(true) {
            rule.insert(p, dummy(&name, &[(Predicate::negative, true)]));
        } else if known_real(&p, assumptions)? {
            rule.insert(p, dummy(&name, &[(Predicate::real, true)]));
        }
    }
    let difference: Basic = replace(&sub(value, other)?, &rule)?;
    let simplified: Option<Basic> = within(eighth(), || simplify(&difference))?;
    Ok(simplified == Some(Basic::zero()))
}

/// `e` with its trigonometric functions rewritten through `sin` and
/// `cos` and expanded; `None` when one of another argument than `x` is
/// left.
fn trigonometric_form(e: &Basic, x: &Basic) -> R2 {
    let g: Basic = expand_trig(&rewrite_to(&rewrite_to(e, "cos")?, "sin")?)?;
    for node in g.atoms(Some(
        &|n: &Basic| matches!(n.head(), Head::Function(f) if TRIGONOMETRIC.contains(f)),
    )) {
        let plain: bool = (node.is_function(Function::Sin) || node.is_function(Function::Cos)) && node.args()[0] == *x;
        if node.has(x) && !plain {
            return Ok(None);
        }
    }
    Ok(Some(g))
}

fn sin_cos_rule(x: &Basic, sine: Basic, cosine: Basic) -> PolyResult<HashMap<Basic, Basic>> {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(crate::functions::elementary::trigonometric::sin(x)?, sine);
    rule.insert(crate::functions::elementary::trigonometric::cos(x)?, cosine);
    Ok(rule)
}

/// The integrand over a period as a rational function of `z = exp(I*x)`,
/// `dx = dz/(I*z)` included (`_period_fraction`).
fn period_fraction(e: &Basic, x: &Basic, z: &Basic) -> PolyResult<Option<Fraction>> {
    let Some(g) = trigonometric_form(e, x)? else {
        return Ok(None);
    };
    let reciprocal: Basic = inverse(z)?;
    let rule = sin_cos_rule(
        x,
        mul(&[sub(z, &reciprocal)?, inverse(&mul(&[int(2), i_unit()])?)?])?,
        mul(&[add(&[z.clone(), reciprocal])?, Basic::half()])?,
    )?;
    let substituted: Basic = mul(&[replace(&g, &rule)?, inverse(&mul(&[i_unit(), z.clone()])?)?])?;
    if substituted.has(x) {
        return Ok(None);
    }
    rational_function(&substituted, z)
}

/// The value over `k` periods; with `None`, the parameters of the poles
/// whose place against the unit circle was undecided (`_period_value`).
fn period_value(
    e: &Basic,
    x: &Basic,
    lower: &Basic,
    upper: &Basic,
    k: i64,
    assumptions: &[Basic],
) -> PolyResult<(Option<ConditionalValue>, Vec<Basic>)> {
    let z: Basic = fresh("z", false);
    let Some(fraction) = period_fraction(e, x, &z)? else {
        return Ok((None, Vec::new()));
    };
    let Some(found) = poles(&fraction.denominator)? else {
        return Ok((None, Vec::new()));
    };
    let mut locator: Locator = Locator::new(assumptions);
    let mut total: Basic = Basic::zero();
    for p in &found {
        let Some(inside) = locator.inside_unit_circle(p)? else {
            return Ok((None, sorted_symbols(&p.point.free_symbols())));
        };
        if inside {
            let Some(value) = residue(&fraction, p, &Basic::one(), &z, None)? else {
                return Ok((None, Vec::new()));
            };
            total = add(&[total, value])?;
        }
    }
    let parameters: Vec<Basic> = sorted_symbols(&without(e.free_symbols(), x));
    if locator.sampled && parameters.len() == 1 {
        // a place decided at a sample holds where the poles do not cross
        // the unit circle: the assumptions must keep the parameter on one
        // side of every value where one may
        let s: &Basic = &parameters[0];
        let Some(points) = crossings(e, x, s)? else {
            return Ok((None, Vec::new()));
        };
        for c in &points {
            if ask(&gt(s, c)?, assumptions)?.is_none() && ask(&lt(s, c)?, assumptions)?.is_none() {
                return Ok((None, vec![s.clone()]));
            }
        }
    }
    let value: Basic = mul(&[int(2), pi(), i_unit(), int(k), total])?;
    let result = checked(
        ConditionalValue::new(tidy(&value, assumptions)?, true_()),
        e,
        x,
        lower,
        upper,
        assumptions,
        &locator,
    )?;
    Ok((result, Vec::new()))
}

/// The real values of the only parameter `s` at which a pole in `z =
/// exp(I*x)` may meet the unit circle, sorted; `None` when they are not
/// found in radicals (`_crossings`).
fn crossings(e: &Basic, x: &Basic, s: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let Some(g) = trigonometric_form(e, x)? else {
        return Ok(None);
    };
    let t: Basic = fresh("t", true);
    let square: Basic = add(&[int(1), pow(&t, &int(2))?])?;
    let rule = sin_cos_rule(
        x,
        mul(&[int(2), t.clone(), inverse(&square)?])?,
        mul(&[sub(&int(1), &pow(&t, &int(2))?)?, inverse(&square)?])?,
    )?;
    let substituted: Basic = replace(&g, &rule)?;
    let mut free: BTreeSet<Basic> = substituted.free_symbols();
    free.remove(&t);
    free.remove(s);
    if substituted.has(x) || !free.is_empty() {
        return Ok(None);
    }
    let Some(reduced) = within(budget(), || cancel(&substituted))? else {
        return Ok(None);
    };
    let denominator: Basic = numer_denom(&reduced)?.1;
    let mut factors: Vec<Basic> = Vec::new();
    let gathered: PolyResult<()> = (|| {
        let (content, parts) = crate::polys::polytools::sqf_list(&denominator, std::slice::from_ref(&t))?;
        factors.push(content);
        for (part, _) in parts {
            let polynomial = crate::polys::polytools::Poly::new(&part, std::slice::from_ref(&t))?;
            factors.push(polynomial.LC()?);
            if polynomial.degree(0)? >= 2 {
                factors.push(crate::polys::polytools::discriminant_expr(
                    &part,
                    std::slice::from_ref(&t),
                )?);
            }
        }
        Ok(())
    })();
    if caught(gathered)?.is_none() {
        return Ok(None);
    }
    // and where a pole in z goes to 0 or to infinity, from the
    // coefficients in z
    let Some(fraction) = period_fraction(e, x, &fresh("z", false))? else {
        return Ok(None);
    };
    let coefficients: &[Basic] = &fraction.denominator.coefficients;
    for coefficient in [&coefficients[0], &coefficients[coefficients.len() - 1]] {
        let Some(components) = real_and_imaginary(coefficient, s)? else {
            return Ok(None);
        };
        factors.extend(components);
    }
    let mut points: Vec<(f64, Basic)> = Vec::new();
    for factor in &factors {
        let product: Basic = expand(factor)?;
        if product == Basic::zero() || !product.has(s) {
            continue;
        }
        let found = caught((|| {
            let polynomial = crate::polys::polytools::Poly::new(&product, std::slice::from_ref(s))?;
            let degree: i64 = polynomial.degree(0)?;
            let found = crate::polys::polyroots::roots(&product, std::slice::from_ref(s), true, true, None)?;
            Ok((degree, found))
        })())?;
        let Some((degree, found)) = found else {
            return Ok(None);
        };
        if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != degree {
            return Ok(None);
        }
        for (root, _) in found.iter() {
            let real: Option<bool> = is(root, Predicate::extended_real);
            if !root.free_symbols().is_empty() || real == Some(false) {
                continue;
            }
            let Some(value) = crate::core::evalf::numeric_value(root, 120) else {
                return Ok(None);
            };
            let (re_part, im_part) = crate::core::evalf::parts(&value);
            if real != Some(true) {
                if im_part.to_f64().value().abs() > 1e-30 {
                    continue;
                }
                if !im_part.repr().significand().is_zero() {
                    // a real root which evaluates with a rounding error in
                    // the imaginary part, or a complex one close to the line
                    return Ok(None);
                }
            }
            if points.iter().all(|(_, other)| other != root) {
                points.push((re_part.to_f64().value(), root.clone()));
            }
        }
    }
    if points.len() > MAX_CROSSINGS {
        return Ok(None);
    }
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Some(points.into_iter().map(|(_, r)| r).collect()))
}

/// Polynomials in `s` whose common real roots are the real roots of the
/// polynomial `c` with Gaussian rational coefficients
/// (`_real_and_imaginary`).
fn real_and_imaginary(c: &Basic, s: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let w: Basic = fresh("w", false);
    let written: Basic = expand(&replace1(c, &i_unit(), &w)?)?;
    let mut free: BTreeSet<Basic> = written.free_symbols();
    free.remove(s);
    free.remove(&w);
    if !free.is_empty() {
        return Ok(None);
    }
    if !written.has(&w) {
        return Ok(Some(vec![written]));
    }
    let Some(coefficients) =
        caught(crate::polys::polytools::Poly::new(&written, std::slice::from_ref(&w)).and_then(|p| p.all_coeffs()))?
    else {
        return Ok(None);
    };
    let mut real: Vec<Basic> = Vec::new();
    let mut imaginary: Vec<Basic> = Vec::new();
    for (k, coefficient) in coefficients.iter().rev().enumerate() {
        let sign: Basic = int(if k % 4 == 2 || k % 4 == 3 { -1 } else { 1 });
        let term: Basic = mul(&[sign, coefficient.clone()])?;
        if k % 2 == 0 {
            real.push(term);
        } else {
            imaginary.push(term);
        }
    }
    let real: Basic = expand(&add(&real)?)?;
    let imaginary: Basic = expand(&add(&imaginary)?)?;
    if real == Basic::zero() || imaginary == Basic::zero() {
        return Ok(Some(vec![add(&[real, imaginary])?]));
    }
    if !real.has(s) || !imaginary.has(s) {
        // a nonzero constant part: no real root
        return Ok(Some(Vec::new()));
    }
    Ok(caught(crate::polys::polytools::gcd_expr(
        &real,
        &imaginary,
        std::slice::from_ref(s),
    ))?
    .map(|g| vec![g]))
}

// ---------------------------------------------------------------------------
// Verification and the entry point

/// `Integral(f, (x, a, b))` of an integrand free of parameters with an
/// error bound: mpmath's two rules when they agree to ten digits, else
/// the validated quadrature (`_quadrature`).
fn quadrature(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<((f64, f64), f64)>> {
    let rough = within(budget(), || crate::integrals::recognize::quadrature(f, x, a, b, 15))?.flatten();
    if let Some(rough) = rough {
        let r: f64 = rough.to_f64().value();
        return Ok(Some(((r, 0.0), 1e-10 * (1.0 + r.abs()))));
    }
    let quarter: Option<f64> = budget().map(|t| t / 4.0);
    let Some(Some((value, error))) = within(quarter, || {
        crate::integrals::validated::validated_integral(f, x, a, b, 8)
    })?
    else {
        return Ok(None);
    };
    let center: f64 = value.to_f64().value();
    let bound: f64 = error.to_f64().value().abs();
    Ok(Some(((center, 0.0), bound)))
}

/// Whether `value` agrees with a quadrature at samples of the parameters
/// satisfying the assumptions; `None` when nothing could be checked
/// (`_verified`).
fn verified(
    value: &Basic,
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<bool>> {
    let mut symbols: BTreeSet<Basic> = f.free_symbols();
    symbols.extend(a.free_symbols());
    symbols.extend(b.free_symbols());
    let parameters: Vec<Basic> = sorted_symbols(&without(symbols, x));
    let mut rng: PyRandom = PyRandom::from_string(&format!("({}, {}, {})", f, a, b));
    let mut verdict: Option<bool> = None;
    for _ in 0..(if parameters.is_empty() { 1 } else { SAMPLES }) {
        let Some(values) = crate::integrals::conditions::sample_values(&parameters, assumptions, &mut rng)? else {
            return Ok(None);
        };
        let rule: HashMap<Basic, Basic> = values.iter().cloned().collect();
        let integrand: Basic = replace(f, &rule)?;
        let lower: Basic = replace(a, &rule)?;
        let upper: Basic = replace(b, &rule)?;
        if !without(integrand.free_symbols(), x).is_empty()
            || !lower.free_symbols().is_empty()
            || !upper.free_symbols().is_empty()
        {
            return Ok(None);
        }
        let Some((center, error)) = quadrature(&integrand, x, &lower, &upper)? else {
            return Ok(None);
        };
        let Some(ours) = crate::numeric::reliable_complex(value, 20, &values) else {
            return Ok(None);
        };
        let distance: f64 = (ours.0 - center.0).hypot(ours.1 - center.1);
        if distance > error + 1e-6 * (1.0 + center.0.hypot(center.1)) {
            return Ok(Some(false));
        }
        verdict = Some(true);
    }
    Ok(verdict)
}

/// The result, verified numerically when a pole was placed by sampling
/// (`_checked`).
fn checked(
    result: ConditionalValue,
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    locator: &Locator,
) -> OptCV {
    if !locator.sampled {
        return Ok(Some(result));
    }
    let mut verdict: Option<bool> = verified(&result.value, f, x, a, b, assumptions)?;
    if verdict.is_none() {
        verdict = crate::integrals::definite::verify_numerically(&result.value, f, x, a, b, assumptions, 2, &[])?;
    }
    if verdict == Some(false) {
        return Ok(None);
    }
    Ok(Some(result))
}

/// `Integral(f, (x, a, b))` by the residue theorem when `f` is one of the
/// families of the module documentation (`residue_integral`).
pub fn residue_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> OptCV {
    if !f.has(x) {
        return Ok(None);
    }
    let (constant, rest) = crate::core::expr::as_independent(f, x, false)?;
    if constant != Basic::one() {
        let found: Option<ConditionalValue> = residue_integral(&rest, x, a, b, assumptions)?;
        return match found {
            None => Ok(None),
            Some(found) => Ok(Some(found.scaled(&constant)?)),
        };
    }
    let fourier: bool = has_function(&rest, &[Function::Exp, Function::Sin, Function::Cos]);
    if *a == neg_oo() && *b == oo() {
        if fourier {
            return fourier_integral(&rest, x, assumptions, false);
        }
        return real_line_integral(&rest, x, assumptions);
    }
    if *a == Basic::zero() && *b == oo() {
        if fourier {
            return fourier_integral(&rest, x, assumptions, true);
        }
        if has_function(&rest, &TRIGONOMETRIC) {
            return Ok(None);
        }
        return keyhole_integral(&rest, x, assumptions);
    }
    if is_number(a) && is_number(b) && crate::integrals::definite::periods(a, b)?.is_some() {
        return period_integral(&rest, x, a, b, assumptions);
    }
    Ok(None)
}
