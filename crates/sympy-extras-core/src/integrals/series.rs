//! Port of `sympy_extras/integrals/series.py`: definite integrals by the
//! series expansion of a factor of the integrand and termwise integration
//! (power series about 0, geometric series of exponentials, the classical
//! and the parametric Fourier series, Fourier series with computed
//! coefficients), the sums of the resulting series in closed form, and the
//! justification of the interchange of sum and integral.

#![allow(clippy::mutable_key_type)]

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{is, polynomial_like, PolynomialKind};
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;
type OptCV = PolyResult<Option<ConditionalValue>>;

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
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn expand_mul(e: &Basic) -> R {
    crate::core::function::expand_mul(e, true)
}
fn sin(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::sin(e)
}
fn cos(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::cos(e)
}
fn tan(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::tan(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn abs_(e: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(e)
}
fn sign(e: &Basic) -> R {
    crate::functions::elementary::complexes::sign(e)
}
fn sqrt(e: &Basic) -> R {
    pow(e, &Basic::half())
}
fn evaluated(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    crate::core::function::xreplace_evaluated(e, rule)
}
fn rule_of(pairs: &[(Basic, Basic)]) -> HashMap<Basic, Basic> {
    pairs.iter().cloned().collect()
}
fn ask(q: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    crate::assumptions::ask::ask(q, assumptions, None)
}
fn le(a: &Basic, b: &Basic) -> R {
    crate::core::relational::le(a, b)
}
fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
fn ge(a: &Basic, b: &Basic) -> R {
    crate::core::relational::ge(a, b)
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn and_(args: &[Basic]) -> R {
    Ok(crate::core::logic::and_(args))
}
fn true_() -> Basic {
    crate::core::logic::true_()
}
fn pi() -> Basic {
    Basic::atom(Head::Pi)
}
fn oo() -> Basic {
    Basic::oo()
}
fn is_infinite_bound(e: &Basic) -> bool {
    *e == Basic::oo() || *e == Basic::negative_oo()
}
fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    polynomial_like(e, std::slice::from_ref(x), PolynomialKind::Polynomial) == Some(true)
}
fn is_rational_function(e: &Basic, x: &Basic) -> bool {
    polynomial_like(e, std::slice::from_ref(x), PolynomialKind::Rational) == Some(true)
}
fn integer_of(e: &Basic) -> Option<i64> {
    if !matches!(e.head(), Head::Rational { value, .. } if value.is_int()) {
        return None;
    }
    i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
}
fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn factors_of(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn has_head(e: &Basic, pred: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(pred)
}
fn is_trig(t: &Basic) -> bool {
    t.is_function(Function::Sin) || t.is_function(Function::Cos) || t.is_function(Function::Tan)
}
fn has_infinity(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::Infinity | Head::NegativeInfinity))
}
fn has_undefined(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity))
}
fn is_sum(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::Sum))
}
fn has_piecewise(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::Piecewise))
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
/// `attempt(f, settings.timeout, native=True)`.
fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn attempt_opt<T>(f: impl FnOnce() -> PolyResult<Option<T>>) -> PolyResult<Option<T>> {
    Ok(attempt(f)?.flatten())
}
fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}
fn monomial(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    crate::integrals::mellin::monomial(e, x)
}
fn positive(e: &Basic) -> bool {
    is(e, Predicate::positive) == Some(true)
}
fn element(x: &Basic, domain: Head) -> R {
    crate::assumptions::facts::element(x, &Basic::atom(domain))
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A dummy with a name of its own (`_fresh`): the typed polynomials map
/// their symbols by name.
fn fresh(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    let number: usize = COUNTER.fetch_add(1, Ordering::SeqCst);
    crate::core::symbol::Dummy::new(&format!("{}{}", name, number), assumptions).0
}

thread_local! {
    /// The depth of the Fourier coefficient and moment integrals in
    /// progress (`_nesting`).
    static NESTING: Cell<usize> = const { Cell::new(0) };
    /// The variable and the index of the table of Fourier series (`_u`, `_k`).
    static TABLE_SYMBOLS: (Basic, Basic) = (
        fresh("u", &[]),
        fresh("k", &[(Predicate::integer, true), (Predicate::positive, true)]),
    );
    static FOURIER_TABLE: Vec<FourierEntry> = fourier_table().expect("the table of Fourier series");
}

fn nesting() -> usize {
    NESTING.with(|n| n.get())
}

/// The nesting counter raised while `f` runs.
fn nested<T>(f: impl FnOnce() -> T) -> T {
    NESTING.with(|n| n.set(n.get() + 1));
    let result: T = f();
    NESTING.with(|n| n.set(n.get() - 1));
    result
}

fn table_u() -> Basic {
    TABLE_SYMBOLS.with(|s| s.0.clone())
}
fn table_k() -> Basic {
    TABLE_SYMBOLS.with(|s| s.1.clone())
}

/// A factor of the integrand expanded as a series (`Expansion`):
/// `constant + Sum(coefficient(j) * basis(j), (j, start, oo))` under
/// `condition`.
#[derive(Clone, Debug)]
pub struct Expansion {
    pub factor: Basic,
    pub coefficient: Basic,
    pub index: Basic,
    pub basis: Basic,
    pub start: i64,
    pub constant: Basic,
    pub condition: Basic,
}

impl Expansion {
    fn new(factor: Basic, coefficient: Basic, index: Basic, basis: Basic, start: i64) -> Expansion {
        Expansion {
            factor,
            coefficient,
            index,
            basis,
            start,
            constant: Basic::zero(),
            condition: true_(),
        }
    }
}

/// The first index from which the coefficient formula is finite
/// (`_first_finite`).
fn first_finite(coefficient: &Basic, j: &Basic) -> PolyResult<Option<i64>> {
    for start in 0..6 {
        let mut finite: bool = true;
        for i in 0..6 {
            let v: Basic = subs(coefficient, j, &int(start + i))?;
            if has_undefined(&v) || has_infinity(&v) {
                finite = false;
                break;
            }
        }
        if finite {
            return Ok(Some(start));
        }
    }
    Ok(None)
}

/// The power series of `g` about 0 as an [`Expansion`]
/// (`_power_expansion`).
fn power_expansion(g: &Basic, x: &Basic) -> PolyResult<Option<Expansion>> {
    let Some(series) = crate::integrals::brackets::taylor_coefficient(g, x)? else {
        return Ok(None);
    };
    let j: Basic = series.index.clone();
    let Some(start) = first_finite(&series.coefficient, &j)? else {
        return Ok(None);
    };
    let basis: Basic = pow(
        x,
        &add(&[mul(&[int(series.period), j.clone()])?, series.offset.clone()])?,
    )?;
    Ok(Some(Expansion::new(g.clone(), series.coefficient, j, basis, start)))
}

/// `1/(exp(c x) - 1)` and `1/(exp(c x) + 1)` (`c > 0`) as geometric series
/// of exponentials, valid for `x > 0` (`_exponential_expansion`).
fn exponential_expansion(g: &Basic, x: &Basic) -> PolyResult<Option<Expansion>> {
    if !g.is_pow() || g.args()[1] != int(-1) || !g.args()[0].is_add() || g.args()[0].args().len() != 2 {
        return Ok(None);
    }
    let terms: &[Basic] = g.args()[0].args();
    for (constant, other) in [(&terms[0], &terms[1]), (&terms[1], &terms[0])] {
        if constant.has(x) || !other.is_function(Function::Exp) {
            continue;
        }
        let Some((c, power)) = monomial(&other.args()[0], x)? else {
            continue;
        };
        if power != Basic::one() || !positive(&c) {
            continue;
        }
        let j: Basic = fresh("j", &[(Predicate::integer, true), (Predicate::positive, true)]);
        let basis: Basic = exp(&neg(&mul(&[c.clone(), j.clone(), x.clone()])?)?)?;
        if *constant == int(-1) {
            return Ok(Some(Expansion::new(g.clone(), Basic::one(), j, basis, 1)));
        }
        if *constant == int(1) {
            let coefficient: Basic = pow(&int(-1), &sub(&j, &int(1))?)?;
            return Ok(Some(Expansion::new(g.clone(), coefficient, j, basis, 1)));
        }
    }
    Ok(None)
}

/// A Fourier series `function(u) = constant + Sum(coefficient(k) *
/// basis(k, u), (k, 1, oo))` valid for `lower < u < upper`
/// (`_FourierEntry`).
struct FourierEntry {
    function: Basic,
    constant: Basic,
    coefficient: Basic,
    basis: Basic,
    lower: Basic,
    upper: Basic,
}

/// The classical Fourier series ([GR] 1.441-1.444), in the variable `u`
/// and the index `k >= 1` (`_FOURIER_TABLE`).
fn fourier_table() -> PolyResult<Vec<FourierEntry>> {
    let (u, k) = TABLE_SYMBOLS.with(|s| s.clone());
    let pi: Basic = pi();
    let minus_one = int(-1);
    let alternating = |shift: i64| -> R { pow(&minus_one, &add(&[k.clone(), int(shift)])?) };
    let entry =
        |function: Basic, constant: Basic, coefficient: Basic, basis: Basic, lower: Basic, upper: Basic| FourierEntry {
            function,
            constant,
            coefficient,
            basis,
            lower,
            upper,
        };
    let ku = |m: i64| -> R { mul(&[int(m), k.clone(), u.clone()]) };
    let two_k_minus = |m: i64, s: i64| -> R { mul(&[add(&[mul(&[int(m), k.clone()])?, int(-s)])?, u.clone()]) };
    let log2: Basic = log(&int(2))?;
    let inv_k: Basic = pow(&k, &int(-1))?;
    let inv_k2: Basic = pow(&k, &int(-2))?;
    let half_pi: Basic = div(&pi, &int(2))?;
    let two_pi: Basic = mul(&[int(2), pi.clone()])?;
    let four_k2_minus_1: Basic = add(&[mul(&[int(4), pow(&k, &int(2))?])?, int(-1)])?;
    let two_k_minus_1: Basic = add(&[mul(&[int(2), k.clone()])?, int(-1)])?;
    Ok(vec![
        // log(sin u) = -log 2 - sum cos(2ku)/k, log(cos u) = -log 2 - sum (-1)^k cos(2ku)/k
        entry(
            log(&sin(&u)?)?,
            neg(&log2)?,
            neg(&inv_k)?,
            cos(&ku(2)?)?,
            Basic::zero(),
            pi.clone(),
        ),
        entry(
            log(&cos(&u)?)?,
            neg(&log2)?,
            div(&alternating(1)?, &k)?,
            cos(&ku(2)?)?,
            neg(&half_pi)?,
            half_pi.clone(),
        ),
        // log(tan u) = -2 sum cos((4k - 2)u)/(2k - 1)
        entry(
            log(&tan(&u)?)?,
            Basic::zero(),
            div(&int(-2), &two_k_minus_1)?,
            cos(&two_k_minus(4, 2)?)?,
            Basic::zero(),
            half_pi.clone(),
        ),
        // log(1 + cos u), log(1 - cos u)
        entry(
            log(&add(&[int(1), cos(&u)?])?)?,
            neg(&log2)?,
            mul(&[int(2), alternating(1)?, inv_k.clone()])?,
            cos(&ku(1)?)?,
            neg(&pi)?,
            pi.clone(),
        ),
        entry(
            log(&sub(&int(1), &cos(&u)?)?)?,
            neg(&log2)?,
            mul(&[int(-2), inv_k.clone()])?,
            cos(&ku(1)?)?,
            Basic::zero(),
            two_pi.clone(),
        ),
        // the sawtooth waves
        entry(
            u.clone(),
            pi.clone(),
            mul(&[int(-2), inv_k.clone()])?,
            sin(&ku(1)?)?,
            Basic::zero(),
            two_pi.clone(),
        ),
        entry(
            u.clone(),
            Basic::zero(),
            mul(&[int(2), alternating(1)?, inv_k.clone()])?,
            sin(&ku(1)?)?,
            neg(&pi)?,
            pi.clone(),
        ),
        entry(
            pow(&u, &int(2))?,
            div(&mul(&[int(4), pow(&pi, &int(2))?])?, &int(3))?,
            int(1),
            sub(
                &mul(&[int(4), cos(&ku(1)?)?, inv_k2.clone()])?,
                &mul(&[int(4), pi.clone(), sin(&ku(1)?)?, inv_k.clone()])?,
            )?,
            Basic::zero(),
            two_pi.clone(),
        ),
        entry(
            pow(&u, &int(2))?,
            div(&pow(&pi, &int(2))?, &int(3))?,
            mul(&[int(4), alternating(0)?, inv_k2.clone()])?,
            cos(&ku(1)?)?,
            neg(&pi)?,
            pi.clone(),
        ),
        // the rectified and the square waves, valid everywhere
        entry(
            abs_(&sin(&u)?)?,
            div(&int(2), &pi)?,
            div(&int(-4), &mul(&[pi.clone(), four_k2_minus_1.clone()])?)?,
            cos(&ku(2)?)?,
            Basic::negative_oo(),
            oo(),
        ),
        entry(
            abs_(&cos(&u)?)?,
            div(&int(2), &pi)?,
            div(
                &mul(&[int(-4), alternating(0)?])?,
                &mul(&[pi.clone(), four_k2_minus_1])?,
            )?,
            cos(&ku(2)?)?,
            Basic::negative_oo(),
            oo(),
        ),
        entry(
            sign(&sin(&u)?)?,
            Basic::zero(),
            div(&int(4), &mul(&[pi.clone(), two_k_minus_1.clone()])?)?,
            sin(&two_k_minus(2, 1)?)?,
            Basic::negative_oo(),
            oo(),
        ),
        entry(
            sign(&cos(&u)?)?,
            Basic::zero(),
            div(&mul(&[int(4), alternating(1)?])?, &mul(&[pi.clone(), two_k_minus_1])?)?,
            cos(&two_k_minus(2, 1)?)?,
            Basic::negative_oo(),
            oo(),
        ),
    ])
}

/// `(m, h)` with `g(x) = h(m x)`, `m > 0` (`_scaled`).
fn scaled(g: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let u: Basic = table_u();
    let mut arguments: Vec<Basic> = Vec::new();
    for t in g.atoms(Some(&is_trig)) {
        if !arguments.contains(&t.args()[0]) {
            arguments.push(t.args()[0].clone());
        }
    }
    if arguments.is_empty() {
        return Ok(Some((Basic::one(), evaluated(g, &rule_of(&[(x.clone(), u)]))?)));
    }
    if arguments.len() > 1 {
        return Ok(None);
    }
    let argument: Basic = arguments.pop().expect("one argument");
    let Some((m, power)) = monomial(&argument, x)? else {
        return Ok(None);
    };
    if power != Basic::one() || !positive(&m) {
        return Ok(None);
    }
    let h: Basic = evaluated(g, &rule_of(&[(argument, u)]))?;
    if h.has(x) {
        return Ok(None);
    }
    Ok(Some((m, h)))
}

/// Whether `[a, b]` lies in `[lower, upper]` (`_within`).
fn within(a: &Basic, b: &Basic, lower: &Basic, upper: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if *lower != Basic::negative_oo() && ask(&le(lower, a)?, assumptions)? != Some(true) {
        return Ok(false);
    }
    if *upper != oo() && ask(&le(b, upper)?, assumptions)? != Some(true) {
        return Ok(false);
    }
    Ok(true)
}

type Kernel = (Basic, Basic, Basic, Basic);

/// `(P, Q, a, condition)` for `h = P + Q*cos(u)` with `P > |Q|`
/// (`_cosine_kernel`).
fn cosine_kernel(h: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Kernel>> {
    if !h.is_add() {
        return Ok(None);
    }
    let u: Basic = table_u();
    let (p, rest) = crate::core::expr::as_independent(h, &u, true)?;
    let (q, harmonic) = crate::core::expr::as_independent(&rest, &u, false)?;
    if harmonic != cos(&u)? || p == Basic::zero() {
        return Ok(None);
    }
    let condition: Basic = gt(&p, &abs_(&q)?)?;
    let verdict: Option<bool> = ask(&condition, assumptions)?;
    if verdict == Some(false) {
        return Ok(None);
    }
    let root: Basic =
        crate::assumptions::refine::refine(&sqrt(&sub(&pow(&p, &int(2))?, &pow(&q, &int(2))?)?)?, assumptions, None)?;
    let a: Basic = cancel(&div(&sub(&root, &p)?, &q)?)?;
    Ok(Some((p, q, a, if verdict == Some(true) { true_() } else { condition })))
}

/// `(constant, coefficient, basis, condition)` of the Fourier series of
/// `log(P + Q*cos(u))` and `1/(P + Q*cos(u))` for `P > |Q|`
/// (`_parametric_entry`).
fn parametric_entry(h: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Kernel>> {
    let (u, k) = (table_u(), table_k());
    let ku: Basic = mul(&[k.clone(), u.clone()])?;
    if h.is_function(Function::Log) {
        let Some((p, _q, a, condition)) = cosine_kernel(&h.args()[0], assumptions)? else {
            return Ok(None);
        };
        let constant: Basic = log(&div(&p, &add(&[int(1), pow(&a, &int(2))?])?)?)?;
        let coefficient: Basic = div(&mul(&[int(-2), pow(&a, &k)?])?, &k)?;
        return Ok(Some((constant, coefficient, cos(&ku)?, condition)));
    }
    if h.is_pow() && h.args()[1] == int(-1) {
        let Some((p, _q, a, condition)) = cosine_kernel(&h.args()[0], assumptions)? else {
            return Ok(None);
        };
        let a2: Basic = pow(&a, &int(2))?;
        let scale: Basic = cancel(&div(&add(&[int(1), a2.clone()])?, &mul(&[p, sub(&int(1), &a2)?])?)?)?;
        let coefficient: Basic = mul(&[int(2), scale.clone(), pow(&a, &k)?])?;
        return Ok(Some((scale, coefficient, cos(&ku)?, condition)));
    }
    Ok(None)
}

/// Whether `e` has an unevaluated integral, a `Piecewise` or an undefined
/// value (`_bad_value`).
fn bad_value(e: &Basic) -> bool {
    crate::integrals::laplace::has_integral(e) || has_piecewise(e) || has_undefined(e)
}

/// The Fourier series of `g` on the bounded range `(a, b)` with the
/// coefficients computed as integrals with the symbolic index `k`
/// (`_computed_expansion`).
fn computed_expansion(
    g: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Expansion>> {
    if nesting() > 0 || is_infinite_bound(a) || is_infinite_bound(b) {
        return Ok(None);
    }
    if g.atoms(Some(&is_trig)).iter().any(|t| t.has(x)) {
        // a harmonic in g resonates with the basis at one index
        return Ok(None);
    }
    let length: Basic = sub(b, a)?;
    let k: Basic = fresh("k", &[(Predicate::integer, true), (Predicate::positive, true)]);
    let mut facts: Vec<Basic> = vec![element(&k, Head::Naturals)?];
    facts.extend(assumptions.iter().cloned());
    let t: Basic = fresh("t", &[(Predicate::positive, true)]);
    let shifted: Basic = subs(g, x, &add(&[a.clone(), t.clone()])?)?;
    let coefficient = |weight: &Basic| -> OptCV {
        let integrand: Basic = mul(&[shifted.clone(), weight.clone()])?;
        let mut found: Option<ConditionalValue> = crate::integrals::antiderivative::antiderivative_integral(
            &integrand,
            &t,
            &Basic::zero(),
            &length,
            &facts,
            false,
            None,
            None,
        )?;
        if found.is_none() {
            found = crate::integrals::definite::conditional_integral(
                &integrand,
                &t,
                &Basic::zero(),
                &length,
                &facts,
                false,
                false,
                false,
            )?;
        }
        let Some(found) = found.filter(|f| !bad_value(&f.value) && !has_infinity(&f.value)) else {
            return Ok(None);
        };
        let value: Basic = found.value.clone();
        let budget: Option<f64> = timeout().map(|s| s / 8.0);
        let simpler: Option<Basic> = crate::timeout::attempt_within(budget, || simplify(&value))?;
        if let Some(simpler) = simpler {
            if ops(&simpler)? < ops(&value)? {
                return Ok(Some(ConditionalValue::new(simpler, found.condition)));
            }
        }
        Ok(Some(found))
    };
    let basis_of = |weight: &Basic| -> R {
        // the weight in x: cos(w (x - a)) = cos(w x) cos(w a) + sin(w x) sin(w a)
        let omega: Basic = div(&weight.args()[0], &t)?;
        let (wx, wa) = (mul(&[omega.clone(), x.clone()])?, mul(&[omega.clone(), a.clone()])?);
        if weight.is_function(Function::Cos) {
            return add(&[mul(&[cos(&wx)?, cos(&wa)?])?, mul(&[sin(&wx)?, sin(&wa)?])?]);
        }
        sub(&mul(&[sin(&wx)?, cos(&wa)?])?, &mul(&[cos(&wx)?, sin(&wa)?])?)
    };
    let systems = || -> PolyResult<Option<Expansion>> {
        let Some(mean) = coefficient(&Basic::one())? else {
            return Ok(None);
        };
        let constant: Basic = div(&mean.value, &length)?;
        let mut conditions: Vec<Basic> = vec![mean.condition.clone()];
        let half: Basic = div(&mul(&[pi(), k.clone()])?, &length)?;
        let full: Basic = div(&mul(&[int(2), pi(), k.clone()])?, &length)?;
        let (ht, ft) = (mul(&[half, t.clone()])?, mul(&[full, t.clone()])?);
        let mut list: Vec<Vec<Basic>> = vec![vec![cos(&ht)?], vec![sin(&ht)?], vec![cos(&ft)?, sin(&ft)?]];
        if *a != Basic::zero() {
            list.rotate_left(2);
        }
        for weights in list {
            let mut parts: Vec<(Basic, Basic)> = Vec::new();
            for weight in &weights {
                let Some(found) = coefficient(weight)? else {
                    break;
                };
                conditions.push(found.condition.clone());
                parts.push((div(&mul(&[int(2), found.value.clone()])?, &length)?, basis_of(weight)?));
            }
            if parts.len() < weights.len() {
                continue;
            }
            let nonzero: Vec<(Basic, Basic)> = parts.iter().filter(|(amp, _)| *amp != Basic::zero()).cloned().collect();
            if nonzero.is_empty() {
                if sub(g, &constant)? == Basic::zero() {
                    let mut e = Expansion::new(g.clone(), Basic::zero(), k.clone(), parts[0].1.clone(), 1);
                    e.constant = constant;
                    e.condition = and_(&conditions)?;
                    return Ok(Some(e));
                }
                continue;
            }
            if nonzero.len() == 1 {
                let (mut amplitude, mut basis) = nonzero[0].clone();
                // (-1)**k sin(k x) from the shift: the sign into the coefficient
                let (factor, harmonic) = crate::core::expr::as_independent(&basis, x, false)?;
                if !basis.is_add() {
                    amplitude = mul(&[amplitude, factor])?;
                    basis = harmonic;
                }
                let mut e = Expansion::new(g.clone(), amplitude, k.clone(), basis, 1);
                e.constant = constant;
                e.condition = and_(&conditions)?;
                return Ok(Some(e));
            }
            let mut terms: Vec<Basic> = Vec::new();
            for (amplitude, b_) in &nonzero {
                terms.push(mul(&[amplitude.clone(), b_.clone()])?);
            }
            let mut e = Expansion::new(g.clone(), Basic::one(), k.clone(), add(&terms)?, 1);
            e.constant = constant;
            e.condition = and_(&conditions)?;
            return Ok(Some(e));
        }
        Ok(None)
    };
    nested(|| attempt_opt(systems))
}

/// The Fourier series of `g` valid on `(a, b)` from the table of the
/// classical series and the parametric entries; with `compute`, the
/// coefficients of the other factors computed as integrals
/// (`fourier_expansion`).
pub fn fourier_expansion(
    g: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    compute: bool,
) -> PolyResult<Option<Expansion>> {
    let polynomial: bool = is_polynomial(g, x);
    let found: Option<Expansion> = if compute && polynomial {
        None
    } else {
        tabulated_expansion(g, x, a, b, assumptions)?
    };
    if found.is_some() || !compute {
        return Ok(found);
    }
    if !g.has(x) || harmonic_parts(g, x)?.is_some() {
        return Ok(None);
    }
    let computed: Option<Expansion> = computed_expansion(g, x, a, b, assumptions)?;
    if computed.is_none() && polynomial {
        return tabulated_expansion(g, x, a, b, assumptions);
    }
    Ok(computed)
}

/// The table and the parametric entries of [`fourier_expansion`]
/// (`_tabulated_expansion`).
fn tabulated_expansion(
    g: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Expansion>> {
    let Some((scale, mut h)) = scaled(g, x)? else {
        return Ok(None);
    };
    let mut shift: Basic = Basic::zero();
    if h.is_function(Function::Log) {
        let (factor, argument) = crate::core::expr::as_coeff_mul(&h.args()[0]);
        if factor != Basic::one() {
            if !positive(&factor) {
                return Ok(None);
            }
            h = log(&argument)?;
            shift = log(&factor)?;
        }
    }
    let (u, table_index) = (table_u(), table_k());
    let entries: Vec<(Basic, Basic, Basic, Basic, Basic, Basic)> = FOURIER_TABLE.with(|table| {
        table
            .iter()
            .map(|e| {
                (
                    e.function.clone(),
                    e.constant.clone(),
                    e.coefficient.clone(),
                    e.basis.clone(),
                    e.lower.clone(),
                    e.upper.clone(),
                )
            })
            .collect()
    });
    for (function, constant, coefficient, basis, lower, upper) in entries {
        if function != h {
            continue;
        }
        if !within(a, b, &div(&lower, &scale)?, &div(&upper, &scale)?, assumptions)? {
            continue;
        }
        let k: Basic = fresh("k", &[(Predicate::integer, true), (Predicate::positive, true)]);
        let replacement = rule_of(&[
            (u.clone(), mul(&[scale.clone(), x.clone()])?),
            (table_index.clone(), k.clone()),
        ]);
        let mut e = Expansion::new(
            g.clone(),
            evaluated(&coefficient, &replacement)?,
            k,
            evaluated(&basis, &replacement)?,
            1,
        );
        e.constant = add(&[constant, shift.clone()])?;
        return Ok(Some(e));
    }
    if let Some((constant, coefficient, basis, condition)) = parametric_entry(&h, assumptions)? {
        let k: Basic = fresh("k", &[(Predicate::integer, true), (Predicate::positive, true)]);
        let rule = rule_of(&[(u, mul(&[scale, x.clone()])?), (table_index, k.clone())]);
        let mut e = Expansion::new(
            g.clone(),
            evaluated(&coefficient, &rule)?,
            k,
            evaluated(&basis, &rule)?,
            1,
        );
        e.constant = add(&[constant, shift])?;
        e.condition = condition;
        return Ok(Some(e));
    }
    Ok(None)
}

/// The ways of writing `f` as an expanded factor times a rest
/// (`expansions`): every factor of `f` (and `f` itself, last) with a power
/// series about 0, and the geometric series of exponentials on `(0, oo)`.
/// Python produces them lazily; here the caller gives the consumer, which
/// returns `Some` to stop.
pub fn expansions<T>(
    f: &Basic,
    x: &Basic,
    _a: &Basic,
    b: &Basic,
    consume: &mut dyn FnMut(Expansion, Basic) -> PolyResult<Option<T>>,
) -> PolyResult<Option<T>> {
    let factors: Vec<Basic> = factors_of(f);
    let mut candidates: Vec<Basic> = factors.clone();
    if factors.len() > 1 {
        candidates.push(f.clone());
    }
    let mut seen: Vec<Basic> = Vec::new();
    for g in candidates {
        if !g.has(x) || seen.contains(&g) || is_polynomial(&g, x) || monomial(&g, x)?.is_some() {
            continue;
        }
        seen.push(g.clone());
        let rest: Basic = div(f, &g)?;
        if *b == oo() {
            if let Some(exponential) = exponential_expansion(&g, x)? {
                if let Some(done) = consume(exponential, rest.clone())? {
                    return Ok(Some(done));
                }
            }
        }
        if !analytic_at_zero(&g, x)? {
            continue;
        }
        if let Some(power) = power_expansion(&g, x)? {
            if let Some(done) = consume(power, rest)? {
                return Ok(Some(done));
            }
        }
    }
    Ok(None)
}

/// Whether `g` has a finite limit at 0 (`_analytic_at_zero`).
fn analytic_at_zero(g: &Basic, x: &Basic) -> PolyResult<bool> {
    let value: Option<Basic> = attempt(|| crate::core::limits::limit(g, x, &Basic::zero(), "+"))?;
    Ok(match value {
        Some(v) => !has_infinity(&v) && !has_undefined(&v) && !has_head(&v, &|n| matches!(n.head(), Head::Limit)),
        None => false,
    })
}

/// `Integral(basis(j) * rest, (x, a, b))` as a closed form in the index
/// (`moments`).
pub fn moments(expansion: &Expansion, rest: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> R2 {
    let j: &Basic = &expansion.index;
    let domain: Head = if expansion.start == 0 {
        Head::Naturals0
    } else {
        Head::Naturals
    };
    let mut facts: Vec<Basic> = vec![element(j, domain)?];
    facts.extend(assumptions.iter().cloned());
    let integrand: Basic = mul(&[expansion.basis.clone(), rest.clone()])?;
    let options = crate::integrals::definite::Options {
        special_values: false,
        ..Default::default()
    };
    let value: Option<Basic> =
        nested(|| attempt(|| crate::integrals::definite::definite_integral(&integrand, x, a, b, &facts, &options)))?;
    Ok(value.filter(|v| !bad_value(v)))
}

type RootMap = Vec<(Basic, usize)>;

/// The roots of `p` in closed form with their multiplicities (`_roots_of`).
fn roots_of(p: &Basic, j: &Basic) -> PolyResult<RootMap> {
    let found = crate::polys::polyroots::roots(p, std::slice::from_ref(j), true, true, None)?;
    Ok(found.iter().map(|(r, m)| (r.clone(), *m)).collect())
}

fn poly_failure(e: &PolyError) -> bool {
    matches!(e, PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_))
        || matches!(e, PolyError::NotImplemented(m) if !m.starts_with(crate::error::NOT_PORTED_PREFIX))
}

/// The sum from `start` of a hypergeometric term as a `hyper` expanded by
/// `hyperexpand` (`_hypergeometric_sum`).
fn hypergeometric_sum(term: &Basic, j: &Basic, start: i64) -> R2 {
    let Some(ratio) = attempt_opt(|| crate::concrete::gosper::hypersimp(term, j))? else {
        return Ok(None);
    };
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&ratio)?;
    let polys = (|| -> PolyResult<(crate::polys::polytools::Poly, crate::polys::polytools::Poly)> {
        Ok((
            crate::polys::polytools::Poly::new(&numerator, std::slice::from_ref(j))?,
            crate::polys::polytools::Poly::new(&denominator, std::slice::from_ref(j))?,
        ))
    })();
    let (p_num, p_den) = match polys {
        Ok(p) => p,
        Err(e) if poly_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let num_roots: RootMap = roots_of(&p_num.as_expr()?, j)?;
    let den_roots: RootMap = roots_of(&p_den.as_expr()?, j)?;
    let total = |roots: &RootMap| roots.iter().map(|(_, m)| *m as i64).sum::<i64>();
    if total(&num_roots) != p_num.degree(0)? || total(&den_roots) != p_den.degree(0)? {
        return Ok(None);
    }
    let first: Basic = subs(term, j, &int(start))?;
    let z: Basic = div(&p_num.LC()?, &p_den.LC()?)?;
    let parameters = |roots: &RootMap| -> PolyResult<Vec<Basic>> {
        let mut out: Vec<Basic> = Vec::new();
        for (r, m) in roots {
            for _ in 0..*m {
                out.push(add(&[neg(r)?, int(start)])?);
            }
        }
        Ok(out)
    };
    let mut upper: Vec<Basic> = parameters(&num_roots)?;
    let mut lower: Vec<Basic> = parameters(&den_roots)?;
    // the series is first * pFq(u; l; z) when the lower parameters include
    // a 1 for the k!
    if let Some(position) = lower.iter().position(|l| *l == Basic::one()) {
        lower.remove(position);
    } else {
        upper.push(Basic::one());
    }
    if upper.len() > lower.len() + 1 {
        return Ok(None);
    }
    if upper.len() == lower.len() + 1 && ask(&lt(&abs_(&z)?, &int(1))?, &[])? != Some(true) {
        return Ok(None);
    }
    let series: Option<Basic> = attempt(|| {
        let h: Basic = mul(&[
            first.clone(),
            crate::functions::special::hyper::hyper(&upper, &lower, &z)?,
        ])?;
        crate::simplify::hyperexpand::hyperexpand(&h, false, crate::simplify::hyperexpand::Rewrite::Default, None)
    })?;
    Ok(series.filter(|s| !has_head(s, &|n| matches!(n.head(), Head::Hyper))))
}

/// The full partial fraction decomposition of a proper rational function
/// of `j` as `(c, r, m)` for the terms `c/(j - r)**m`
/// (`_linear_partial_fractions`).
fn linear_partial_fractions(term: &Basic, j: &Basic) -> PolyResult<Option<Vec<(Basic, Basic, i64)>>> {
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&cancel(term)?)?;
    let polys = (|| -> PolyResult<(crate::polys::polytools::Poly, crate::polys::polytools::Poly)> {
        Ok((
            crate::polys::polytools::Poly::new(&numerator, std::slice::from_ref(j))?,
            crate::polys::polytools::Poly::new(&denominator, std::slice::from_ref(j))?,
        ))
    })();
    let (p_num, p_den) = match polys {
        Ok(p) => p,
        Err(e) if poly_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    if p_num.degree(0)? >= p_den.degree(0)? {
        return Ok(None);
    }
    let found: RootMap = roots_of(&p_den.as_expr()?, j)?;
    if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != p_den.degree(0)? {
        return Ok(None);
    }
    let leading: Basic = p_den.LC()?;
    let mut terms: Vec<(Basic, Basic, i64)> = Vec::new();
    for (r, m) in &found {
        let mut factors: Vec<Basic> = vec![leading.clone()];
        for (s, n) in &found {
            if s != r {
                factors.push(pow(&sub(j, s)?, &int(*n as i64))?);
            }
        }
        let others: Basic = mul(&factors)?;
        if *m == 1 {
            // the residue numerator(r)/denominator'(r)
            let c: Basic = cancel(&div(&subs(&numerator, j, r)?, &subs(&others, j, r)?)?)?;
            terms.push((c, r.clone(), 1));
            continue;
        }
        let h: Basic = div(&numerator, &others)?;
        for i in (1..=*m as i64).rev() {
            let order: i64 = *m as i64 - i;
            let derivative: Basic = if order != 0 {
                crate::core::function::diff(&h, &[(j.clone(), order as u64)])?
            } else {
                h.clone()
            };
            let c: Basic = cancel(&div(
                &subs(&derivative, j, r)?,
                &crate::functions::combinatorial::factorials::factorial(&int(order))?,
            )?)?;
            if c != Basic::zero() {
                terms.push((c, r.clone(), i));
            }
        }
    }
    Ok(Some(terms))
}

/// The sum from `start` of a rational function of `j` by partial
/// fractions (`_rational_sum`).
fn rational_sum(term: &Basic, j: &Basic, start: i64) -> R2 {
    if !is_rational_function(term, j) {
        return Ok(None);
    }
    let Some(parts) = attempt_opt(|| linear_partial_fractions(term, j))? else {
        return Ok(None);
    };
    let mut total: Basic = Basic::zero();
    let mut simple: Basic = Basic::zero();
    for (c, root, m) in parts {
        if c.has(j) {
            return Ok(None);
        }
        // 1/(j - root)^m = 1/(i + r)^m with i = j - start
        let r: Basic = add(&[neg(&root)?, int(start)])?;
        if is(&r, Predicate::integer) == Some(true) && le(&r, &Basic::zero())? == true_() {
            return Ok(None);
        }
        if m >= 2 {
            total = add(&[
                total,
                mul(&[c, crate::functions::special::zeta_functions::zeta(&int(m), Some(&r))?])?,
            ])?;
        } else {
            simple = add(&[simple, c.clone()])?;
            total = sub(
                &total,
                &mul(&[
                    c,
                    crate::functions::special::gamma_functions::polygamma(&Basic::zero(), &r)?,
                ])?,
            )?;
        }
    }
    if cancel(&simple)? != Basic::zero() {
        return Ok(None);
    }
    Ok(Some(polygamma_form(&total)?))
}

/// The sum from `start` of `c * w**j / j**s` through `polylog(s, w)`
/// (`_polylog_sum`).
fn polylog_sum(term: &Basic, j: &Basic, start: i64, assumptions: &[Basic]) -> R2 {
    let mut pieces: Vec<(Basic, Basic)> = Vec::new();
    for summand in terms_of(term) {
        // the part free of the index stays as it is
        let (outer, inner) = crate::core::expr::as_independent(&summand, j, false)?;
        let rewritten: Basic = crate::core::rewriting::rewrite(&inner, &[], "exp", &Default::default())?;
        for piece in terms_of(&expand_mul(&rewritten)?) {
            pieces.push((outer.clone(), piece));
        }
    }
    let mut total: Basic = Basic::zero();
    for (outer, summand) in pieces {
        let mut coefficient: Basic = outer;
        let mut base: Basic = Basic::one();
        let mut order: Option<i64> = None;
        for factor in factors_of(&summand) {
            if !factor.has(j) {
                coefficient = mul(&[coefficient, factor])?;
                continue;
            }
            if factor.is_pow() && factor.args()[0] == *j {
                let exponent: Option<i64> = integer_of(&factor.args()[1]);
                if order.is_some() || !matches!(exponent, Some(e) if e < 0) {
                    return Ok(None);
                }
                order = Some(-exponent.expect("checked above"));
                continue;
            }
            let (b_, e_) = if factor.is_pow() {
                (factor.args()[0].clone(), factor.args()[1].clone())
            } else if factor.is_function(Function::Exp) {
                (Basic::atom(Head::Exp1), factor.args()[0].clone())
            } else {
                return Ok(None);
            };
            let p: Basic = crate::core::expr::coeff(&e_, j)?;
            let q: Basic = expand_mul(&sub(&e_, &mul(&[p.clone(), j.clone()])?)?)?;
            if b_.has(j) || p.has(j) || q.has(j) {
                return Ok(None);
            }
            base = mul(&[base, pow(&b_, &p)?])?;
            coefficient = mul(&[coefficient, pow(&b_, &q)?])?;
        }
        let Some(order) = order else {
            return Ok(None);
        };
        if ask(&le(&abs_(&base)?, &int(1))?, assumptions)? != Some(true) {
            return Ok(None);
        }
        if order == 1 && ask(&crate::core::relational::ne(&base, &int(1))?, assumptions)? != Some(true) {
            return Ok(None);
        }
        let mut value: Basic = crate::functions::special::zeta_functions::polylog(&int(order), &base)?;
        for i in 1..start {
            value = sub(&value, &div(&pow(&base, &int(i))?, &pow(&int(i), &int(order))?)?)?;
        }
        total = add(&[total, mul(&[coefficient, value])?])?;
    }
    Ok(Some(total))
}

/// `Sum(term, (j, start, oo))` in closed form (`sum_series`).
pub fn sum_series(term: &Basic, j: &Basic, start: i64, assumptions: &[Basic]) -> R2 {
    let mut total: Option<Basic> =
        attempt(|| crate::concrete::summations::summation(term, &(j.clone(), int(start), oo())))?;
    if let Some(t) = total.clone().filter(has_piecewise) {
        // the branch of a parametric sum which the assumptions settle
        let folded: Basic = crate::functions::elementary::piecewise::piecewise_fold(&t, None)?;
        total = if matches!(folded.head(), Head::Piecewise) {
            settled_branch(&folded, assumptions)?
        } else {
            None
        };
    }
    if let Some(t) = total
        .clone()
        .filter(|t| has_head(t, &|n| n.is_function(Function::ExpPolar)))
    {
        // polylog(2, a*exp_polar(I*pi)) is polylog(2, -a) off the cut
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for node in t.atoms(Some(&|n: &Basic| n.is_function(Function::ExpPolar))) {
            rule.insert(node.clone(), exp(&node.args()[0])?);
        }
        total = Some(evaluated(&t, &rule)?);
    }
    if let Some(t) = &total {
        if !is_sum(t)
            && !has_undefined(t)
            && !has_piecewise(t)
            && !has_head(t, &|n| matches!(n.head(), Head::Infinity))
            && !crate::integrals::laplace::has_integral(t)
        {
            return Ok(Some(polygamma_form(&settled_abs(t)?)?));
        }
    }
    let trigonometric: bool = has_head(term, &|n| {
        n.is_function(Function::Sin) || n.is_function(Function::Cos) || n.is_function(Function::Exp) || n.is_pow()
    });
    if trigonometric {
        if let Some(clausen) = attempt_opt(|| polylog_sum(term, j, start, assumptions))? {
            return Ok(Some(clausen));
        }
    }
    if let Some(rational) = rational_sum(term, j, start)? {
        return Ok(Some(rational));
    }
    let n: Basic = fresh("n", &[(Predicate::integer, true), (Predicate::positive, true)]);
    let shifted: Basic = subs(term, j, &n)?;
    let value: Option<Basic> =
        attempt_opt(|| crate::concrete::eulersums::polygamma_series(&shifted, &n, &int(start.max(1))))?;
    if value.is_some() && start >= 1 {
        return Ok(value);
    }
    let closed: Option<Basic> = attempt_opt(|| {
        crate::concrete::zeilberger::zeilberger_sum(&shifted, &[n.clone(), int(start), oo()], None, 4, false)
    })?;
    if let Some(closed) = closed {
        let expression: bool = !crate::core::logic::is_boolean(&closed);
        if expression
            && !is_sum(&closed)
            && !has_undefined(&closed)
            && !has_head(&closed, &|n| matches!(n.head(), Head::Infinity))
        {
            return Ok(Some(closed));
        }
    }
    hypergeometric_sum(term, j, start)
}

/// `value` with the absolute values of numbers whose argument expands to a
/// real number evaluated (`_settled_abs`).
fn settled_abs(value: &Basic) -> R {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    let i_unit: Basic = Basic::atom(Head::ImaginaryUnit);
    for node in value.atoms(Some(&|n: &Basic| n.is_function(Function::Abs))) {
        let argument: &Basic = &node.args()[0];
        if crate::core::assumptions::is_number(argument) && argument.has(&i_unit) {
            let expanded: Basic = expand_mul(argument)?;
            if !expanded.has(&i_unit) {
                rule.insert(node.clone(), abs_(&expanded)?);
            }
        }
    }
    if rule.is_empty() {
        Ok(value.clone())
    } else {
        evaluated(value, &rule)
    }
}

/// The value of the first branch whose condition the assumptions settle as
/// true (`_settled_branch`).
fn settled_branch(total: &Basic, assumptions: &[Basic]) -> R2 {
    for pair in total.args() {
        match ask(&pair.args()[1], assumptions)? {
            Some(true) => return Ok(Some(pair.args()[0].clone())),
            None => return Ok(None),
            Some(false) => {}
        }
    }
    Ok(None)
}

/// Whether the series of the absolute values of the integrated terms
/// converges (`_justified`).
fn justified(term: &Basic, j: &Basic, start: i64, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    let verdict: Option<Basic> = attempt_opt(|| {
        crate::concrete::convergence::sum_convergence(&abs_(term)?, j, assumptions, &int(start.max(1)))
    })?;
    let Some(verdict) = verdict else {
        return Ok(None);
    };
    if verdict == true_() {
        return Ok(Some(true));
    }
    if verdict == crate::core::logic::false_() {
        return Ok(Some(false));
    }
    ask(&verdict, assumptions)
}

/// A trigonometric series `constant + Sum(coefficient(k) * (cosine(k) *
/// cos(frequency(k) * x) + sine(k) * sin(frequency(k) * x)), (k, start,
/// oo))`, or a single harmonic (`index` `None`) (`_Harmonics`).
#[derive(Clone, Debug)]
struct Harmonics {
    constant: Basic,
    coefficient: Basic,
    index: Option<Basic>,
    start: i64,
    cosine: Basic,
    sine: Basic,
    frequency: Basic,
}

impl Harmonics {
    fn at(&self, value: &Basic, index: &Basic) -> R {
        match &self.index {
            None => Ok(value.clone()),
            Some(i) => subs(value, i, index),
        }
    }
}

/// `(cosine, sine, frequency)` of `basis = cosine * cos(frequency * x) +
/// sine * sin(frequency * x)` (`_harmonic_parts`).
fn harmonic_parts(basis: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    let mut cosine: Basic = Basic::zero();
    let mut sine: Basic = Basic::zero();
    let mut frequency: Option<Basic> = None;
    for term in terms_of(basis) {
        let (amplitude, harmonic) = crate::core::expr::as_independent(&term, x, false)?;
        let is_cos: bool = harmonic.is_function(Function::Cos);
        if !is_cos && !harmonic.is_function(Function::Sin) {
            return Ok(None);
        }
        let Some((w, power)) = monomial(&harmonic.args()[0], x)? else {
            return Ok(None);
        };
        if power != Basic::one() {
            return Ok(None);
        }
        match &frequency {
            None => frequency = Some(w),
            Some(f) if *f != w => return Ok(None),
            _ => {}
        }
        if is_cos {
            cosine = add(&[cosine, amplitude])?;
        } else {
            sine = add(&[sine, amplitude])?;
        }
    }
    Ok(frequency.map(|f| (cosine, sine, f)))
}

/// The Fourier [`Expansion`] as a [`Harmonics`] (`_harmonics`).
fn harmonics(expansion: &Expansion, x: &Basic) -> PolyResult<Option<Harmonics>> {
    let Some((cosine, sine, frequency)) = harmonic_parts(&expansion.basis, x)? else {
        return Ok(None);
    };
    Ok(Some(Harmonics {
        constant: expansion.constant.clone(),
        coefficient: expansion.coefficient.clone(),
        index: Some(expansion.index.clone()),
        start: expansion.start,
        cosine,
        sine,
        frequency,
    }))
}

/// A single harmonic `c * cos(n * x)` or `c * sin(n * x)` with `n` a
/// positive integer (`_harmonic`).
fn harmonic(rest: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Harmonics>> {
    let Some((cosine, sine, frequency)) = harmonic_parts(rest, x)? else {
        return Ok(None);
    };
    if ask(&element(&frequency, Head::Naturals)?, assumptions)? != Some(true) {
        return Ok(None);
    }
    Ok(Some(Harmonics {
        constant: Basic::zero(),
        coefficient: Basic::one(),
        index: None,
        start: 0,
        cosine,
        sine,
        frequency,
    }))
}

/// `(p, q)` with `frequency = p*k + q`, integers with `p > 0` (`_linear`).
fn linear(frequency: &Basic, k: &Basic) -> PolyResult<Option<(i64, i64)>> {
    let expanded: Basic = expand_mul(frequency)?;
    let p: Basic = crate::core::expr::coeff(&expanded, k)?;
    let q: Basic = expand_mul(&sub(frequency, &mul(&[p.clone(), k.clone()])?)?)?;
    match (integer_of(&p), integer_of(&q)) {
        (Some(p), Some(q)) if p > 0 => Ok(Some((p, q))),
        _ => Ok(None),
    }
}

/// The integral of a term of the series over `(a, b)` as a function of the
/// index (`_primitive`).
fn primitive(s: &Harmonics, a: &Basic, b: &Basic) -> R {
    let w: &Basic = &s.frequency;
    let (wa, wb) = (mul(&[w.clone(), a.clone()])?, mul(&[w.clone(), b.clone()])?);
    let integral: Basic = add(&[
        div(&mul(&[s.cosine.clone(), sub(&sin(&wb)?, &sin(&wa)?)?])?, w)?,
        div(&mul(&[s.sine.clone(), sub(&cos(&wa)?, &cos(&wb)?)?])?, w)?,
    ])?;
    expand_mul(&mul(&[s.coefficient.clone(), integral])?)
}

/// `Integral(f - constant, (x, a, b))` of the series `f` (`_integrated`).
fn integrated(s: &Harmonics, a: &Basic, b: &Basic, assumptions: &[Basic]) -> R2 {
    let term: Basic = primitive(s, a, b)?;
    let Some(index) = &s.index else {
        return Ok(Some(term));
    };
    if term == Basic::zero() {
        return Ok(Some(term));
    }
    Ok(sum_series(&term, index, s.start, assumptions)?.filter(|v| !v.has(index)))
}

/// Whether the harmonics of the two series with different frequencies are
/// orthogonal on `(a, b)` (`_orthogonal`).
fn orthogonal(s1: &Harmonics, s2: &Harmonics, a: &Basic, b: &Basic) -> PolyResult<bool> {
    let zero = Basic::zero();
    let cosines: bool = (s1.cosine != zero && s2.cosine != zero) || (s1.sine != zero && s2.sine != zero);
    let mixed: bool = (s1.cosine != zero && s2.sine != zero) || (s1.sine != zero && s2.cosine != zero);
    for m in [
        expand_mul(&add(&[s1.frequency.clone(), s2.frequency.clone()])?)?,
        expand_mul(&sub(&s1.frequency, &s2.frequency)?)?,
    ] {
        let (ma, mb) = (mul(&[m.clone(), a.clone()])?, mul(&[m.clone(), b.clone()])?);
        if cosines && expand_mul(&sub(&sin(&mb)?, &sin(&ma)?)?)? != zero {
            return Ok(false);
        }
        if mixed && expand_mul(&sub(&cos(&ma)?, &cos(&mb)?)?)? != zero {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A term of the diagonal series: `(term, index, start)` (`_Diagonal`).
type Diagonal = (Basic, Option<Basic>, i64);

/// The terms of the double series of `Integral(f1 * f2)` with equal
/// frequencies as a single series (`_diagonal`).
fn diagonal(s1: &Harmonics, s2: &Harmonics, length: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Diagonal>> {
    let (first, second) = if s1.index.is_none() && s2.index.is_some() {
        (s2, s1)
    } else {
        (s1, s2)
    };
    let product = |j: &Basic, k: &Basic| -> R {
        let inner: Basic = add(&[
            mul(&[first.at(&first.cosine, j)?, second.at(&second.cosine, k)?])?,
            mul(&[first.at(&first.sine, j)?, second.at(&second.sine, k)?])?,
        ])?;
        div(
            &mul(&[
                first.at(&first.coefficient, j)?,
                second.at(&second.coefficient, k)?,
                inner,
                length.clone(),
            ])?,
            &int(2),
        )
    };
    let Some(first_index) = &first.index else {
        return Ok(None);
    };
    if second.index.is_none() {
        // the harmonic n of the second factor picks the term with
        // frequency1(j) = p j + q = n
        let Some((p, q)) = linear(&first.frequency, first_index)? else {
            return Ok(None);
        };
        let j0: Basic = div(&sub(&second.frequency, &int(q))?, &int(p))?;
        match is(&j0, Predicate::integer) {
            Some(false) => return Ok(Some((Basic::zero(), None, 0))),
            None => return Ok(None),
            Some(true) => {}
        }
        match ask(&ge(&j0, &int(first.start))?, assumptions)? {
            Some(false) => return Ok(Some((Basic::zero(), None, 0))),
            None => return Ok(None),
            Some(true) => {}
        }
        return Ok(Some((product(&j0, &Basic::zero())?, None, 0)));
    }
    paired(first, second, &product)
}

/// The single series of the pairs `(j, k)` with `frequency1(j) =
/// frequency2(k)` (`_paired`).
fn paired(s1: &Harmonics, s2: &Harmonics, product: &dyn Fn(&Basic, &Basic) -> R) -> PolyResult<Option<Diagonal>> {
    let (Some(j), Some(k)) = (&s1.index, &s2.index) else {
        return Ok(None);
    };
    let (Some((p1, q1)), Some((p2, q2))) = (linear(&s1.frequency, j)?, linear(&s2.frequency, k)?) else {
        return Ok(None);
    };
    if p1 % p2 == 0 && (q1 - q2).rem_euclid(p2) == 0 {
        // k = (p1 j + q1 - q2)/p2 >= start2 from j >= (p2 start2 - q1 + q2)/p1
        let ratio = |a: i64, b: i64| Basic::rational(crate::numbers::Int::from(a), crate::numbers::Int::from(b));
        let paired_index: Basic = add(&[mul(&[ratio(p1, p2)?, j.clone()])?, ratio(q1 - q2, p2)?])?;
        let start: i64 = s1.start.max(-(q1 - q2 - p2 * s2.start).div_euclid(p1));
        return Ok(Some((product(j, &paired_index)?, Some(j.clone()), start)));
    }
    if p2 % p1 == 0 && (q2 - q1).rem_euclid(p1) == 0 {
        return paired(s2, s1, &|kk: &Basic, jj: &Basic| product(jj, kk));
    }
    if (q2 - q1).rem_euclid(gcd(p1, p2)) != 0 {
        // no common frequency at all
        return Ok(Some((Basic::zero(), None, 0)));
    }
    Ok(None)
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `Integral(f1 * f2, (x, a, b))` of two trigonometric series by
/// orthogonality (`_fourier_product`).
fn fourier_product(s1: &Harmonics, s2: &Harmonics, a: &Basic, b: &Basic, assumptions: &[Basic]) -> R2 {
    let length: Basic = sub(b, a)?;
    let mut total: Basic = mul(&[s1.constant.clone(), s2.constant.clone(), length.clone()])?;
    for (s, other) in [(s1, s2), (s2, s1)] {
        if other.constant == Basic::zero() {
            continue;
        }
        let Some(value) = integrated(s, a, b, assumptions)? else {
            return Ok(None);
        };
        total = add(&[total, mul(&[other.constant.clone(), value])?])?;
    }
    if !orthogonal(s1, s2, a, b)? {
        return Ok(None);
    }
    let Some((term, index, start)) = diagonal(s1, s2, &length, assumptions)? else {
        return Ok(None);
    };
    let Some(index) = index.filter(|_| term != Basic::zero()) else {
        return Ok(Some(add(&[total, term])?));
    };
    let Some(value) = sum_series(&term, &index, start, assumptions)?.filter(|v| !v.has(&index)) else {
        return Ok(None);
    };
    Ok(Some(add(&[total, value])?))
}

/// The constant of `f` and its factors depending on `x` with multiplicity
/// (`_fourier_factors`).
fn fourier_factors(f: &Basic, x: &Basic) -> PolyResult<(Basic, Vec<Basic>)> {
    let mut constant: Basic = Basic::one();
    let mut factors: Vec<Basic> = Vec::new();
    for t in factors_of(f) {
        if !t.has(x) {
            constant = mul(&[constant, t])?;
        } else if t.is_pow() && t.args()[1] == int(2) && monomial(&t.args()[0], x)?.is_none() {
            factors.push(t.args()[0].clone());
            factors.push(t.args()[0].clone());
        } else {
            factors.push(t);
        }
    }
    Ok((constant, factors))
}

/// Whether a harmonic of the rest may share a frequency with the series
/// (`_resonant`).
fn resonant(expansion: &Expansion, rest: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if !rest.atoms(Some(&is_trig)).iter().any(|t| t.has(x)) {
        return Ok(false);
    }
    let (Some(parts), Some(series)) = (harmonic_parts(rest, x)?, harmonic_parts(&expansion.basis, x)?) else {
        return Ok(true);
    };
    let Some((p, q)) = linear(&series.2, &expansion.index)? else {
        return Ok(true);
    };
    let j0: Basic = div(&sub(&parts.2, &int(q))?, &int(p))?;
    let integer: Option<bool> = is(&j0, Predicate::integer);
    if integer == Some(false) {
        return Ok(false);
    }
    Ok(!(integer == Some(true) && ask(&lt(&j0, &int(expansion.start))?, assumptions)? == Some(true)))
}

/// Whether the numerical check confirms `value` (`_verified`).
fn verified(value: &Basic, f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if !crate::settings::settings().numerical_checks {
        return Ok(false);
    }
    Ok(crate::integrals::definite::verify_numerically(value, f, x, a, b, assumptions, 2, &[])? == Some(true))
}

/// `Integral(factor * rest, (x, a, b))` from the Fourier series of the
/// factor integrated termwise against the rest (`_termwise`).
fn termwise(expansion: &Expansion, rest: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> R2 {
    if resonant(expansion, rest, x, assumptions)? {
        return Ok(None);
    }
    let mut mean: Basic = Basic::zero();
    if expansion.constant != Basic::zero() {
        let options = crate::integrals::definite::Options {
            special_values: false,
            ..Default::default()
        };
        let found: Option<Basic> =
            attempt(|| crate::integrals::definite::definite_integral(rest, x, a, b, assumptions, &options))?;
        let Some(found) = found.filter(|f| !bad_value(f)) else {
            return Ok(None);
        };
        mean = found;
    }
    let Some(moment) = moments(expansion, rest, x, a, b, assumptions)? else {
        return Ok(None);
    };
    let j: &Basic = &expansion.index;
    let term: Basic = mul(&[expansion.coefficient.clone(), moment])?;
    let Some(value) = sum_series(&term, j, expansion.start, assumptions)?.filter(|v| !v.has(j)) else {
        return Ok(None);
    };
    let total: Basic = add(&[mul(&[expansion.constant.clone(), mean])?, value])?;
    let verdict: Option<bool> = justified(&term, j, expansion.start, assumptions)?;
    if verdict == Some(false) {
        return Ok(None);
    }
    if verdict.is_none()
        && !verified(
            &total,
            &mul(&[expansion.factor.clone(), rest.clone()])?,
            x,
            a,
            b,
            assumptions,
        )?
    {
        return Ok(None);
    }
    Ok(Some(total))
}

fn tidy(value: &Basic, assumptions: &[Basic]) -> R {
    crate::integrals::marichev::tidy(value, assumptions, &true_())
}

/// `Integral(f, (x, a, b))` through the Fourier series of the factors of
/// `f` (`fourier_integral`).
pub fn fourier_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    expand: Option<&Basic>,
) -> OptCV {
    if ask(&lt(a, b)?, assumptions)? != Some(true) {
        return Ok(None);
    }
    let (constant, factors) = fourier_factors(f, x)?;
    let expanded: Vec<Option<Expansion>> = factors
        .iter()
        .map(|g| fourier_expansion(g, x, a, b, assumptions, false))
        .collect::<PolyResult<_>>()?;
    if let Some(found) = fourier_routes(f, &constant, &factors, &expanded, x, a, b, assumptions, expand)? {
        return Ok(Some(found));
    }
    if is_infinite_bound(a) || is_infinite_bound(b) || nesting() > 0 {
        return Ok(None);
    }
    if expanded.iter().all(|e| e.is_none()) {
        let mut any_harmonic: bool = false;
        for g in &factors {
            if harmonic(g, x, assumptions)?.is_some() {
                any_harmonic = true;
                break;
            }
        }
        if !any_harmonic {
            // nothing trigonometric with integer frequencies to pair with
            return Ok(None);
        }
    }
    // the factors outside the table with computed coefficients; the
    // polynomial factors too; `fresh` marks the expansions computed here
    let mut computed: Vec<Option<Expansion>> = Vec::new();
    let mut fresh_expansion: Vec<bool> = Vec::new();
    for (g, e) in factors.iter().zip(&expanded) {
        if e.is_none() || is_polynomial(g, x) {
            let c: Option<Expansion> = fourier_expansion(g, x, a, b, assumptions, true)?;
            fresh_expansion.push(c.is_some());
            computed.push(c);
        } else {
            fresh_expansion.push(false);
            computed.push(e.clone());
        }
    }
    if !fresh_expansion.iter().any(|f| *f) {
        return Ok(None);
    }
    fourier_routes(f, &constant, &factors, &computed, x, a, b, assumptions, expand)
}

/// The routes of [`fourier_integral`] for given expansions of the factors
/// (`_fourier_routes`).
#[allow(clippy::too_many_arguments)]
fn fourier_routes(
    f: &Basic,
    constant: &Basic,
    factors: &[Basic],
    expanded: &[Option<Expansion>],
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    expand: Option<&Basic>,
) -> OptCV {
    if expanded.iter().all(|e| e.is_none()) {
        return Ok(None);
    }
    if let Some(target) = expand {
        if expanded
            .iter()
            .all(|e| e.as_ref().map_or(true, |e| e.factor != *target))
        {
            return Ok(None);
        }
    }
    let conditions: Vec<Basic> = expanded.iter().flatten().map(|e| e.condition.clone()).collect();
    let condition: Basic = and_(&conditions)?;
    let mut value: Option<Basic> = None;
    if factors.len() == 1 {
        if let Some(first) = &expanded[0] {
            if let Some(s) = harmonics(first, x)? {
                if let Some(value_) = integrated(&s, a, b, assumptions)? {
                    value = Some(add(&[mul(&[s.constant.clone(), sub(b, a)?])?, value_])?);
                }
            }
        }
    } else if factors.len() == 2 {
        let mut series: Vec<Option<Harmonics>> = Vec::new();
        for (e, g) in expanded.iter().zip(factors) {
            series.push(match e {
                Some(e) => harmonics(e, x)?,
                None => harmonic(g, x, assumptions)?,
            });
        }
        if let (Some(s1), Some(s2)) = (&series[0], &series[1]) {
            value = fourier_product(s1, s2, a, b, assumptions)?;
        }
    }
    if let Some(value) = value {
        let tidied: Basic = tidy(&mul(&[constant.clone(), value])?, assumptions)?;
        return Ok(Some(ConditionalValue::new(polygamma_form(&tidied)?, condition)));
    }
    // an expanded factor against the rest of the integrand, the polynomial
    // factors last
    let mut ordered: Vec<(&Basic, &Option<Expansion>)> = factors.iter().zip(expanded).collect();
    ordered.sort_by_key(|(g, _)| is_polynomial(g, x));
    let mut seen: Vec<Basic> = Vec::new();
    for (g, e) in ordered {
        let Some(e) = e else {
            continue;
        };
        if seen.contains(g) || expand.is_some_and(|target| g != target) {
            continue;
        }
        seen.push(g.clone());
        if let Some(found) = termwise(e, &div(f, g)?, x, a, b, assumptions)? {
            let tidied: Basic = tidy(&found, assumptions)?;
            return Ok(Some(ConditionalValue::new(
                polygamma_form(&tidied)?,
                e.condition.clone(),
            )));
        }
    }
    Ok(None)
}

/// `Integral(f, (x, a, b))` by series expansion of a factor and termwise
/// integration (`series_integral`).
pub fn series_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    expand: Option<&Basic>,
) -> OptCV {
    if let Some(fourier) = fourier_integral(f, x, a, b, assumptions, expand)? {
        return Ok(Some(fourier));
    }
    if *a != Basic::zero() || (*b != oo() && ask(&gt(b, &Basic::zero())?, assumptions)? != Some(true)) {
        return Ok(None);
    }
    if *b != oo() && *b != Basic::one() {
        // (0, c): x = c t
        let t: Basic = fresh("t", &[(Predicate::positive, true)]);
        let mapped: Basic = mul(&[subs(f, x, &mul(&[b.clone(), t.clone()])?)?, b.clone()])?;
        return series_integral(&mapped, &t, &Basic::zero(), &Basic::one(), assumptions, None);
    }
    let mut consume = |expansion: Expansion, rest: Basic| -> OptCV {
        if expand.is_some_and(|target| expansion.factor != *target) {
            return Ok(None);
        }
        let Some(moment) = moments(&expansion, &rest, x, a, b, assumptions)? else {
            return Ok(None);
        };
        let j: &Basic = &expansion.index;
        let term: Basic = mul(&[expansion.coefficient.clone(), moment])?;
        let Some(value) = sum_series(&term, j, expansion.start, assumptions)? else {
            return Ok(None);
        };
        if value.free_symbols().contains(j) {
            return Ok(None);
        }
        let verdict: Option<bool> = justified(&term, j, expansion.start, assumptions)?;
        if verdict == Some(false) {
            return Ok(None);
        }
        if verdict.is_none() && !verified(&value, f, x, a, b, assumptions)? {
            return Ok(None);
        }
        Ok(Some(ConditionalValue::new(
            polygamma_form(&tidy(&value, assumptions)?)?,
            true_(),
        )))
    };
    expansions(f, x, a, b, &mut consume)
}

/// Hurwitz zeta values `zeta(n, a)` at integers `n >= 2` (also written
/// `lerchphi(1, n, a)`) as polygamma functions, and Dirichlet eta values
/// as zeta values (`_polygamma_form`).
fn polygamma_form(value: &Basic) -> R {
    let mut value: Basic = value.clone();
    if has_head(&value, &|n| n.is_function(Function::DirichletEta)) {
        value = crate::core::rewriting::rewrite(&value, &[], "zeta", &Default::default())?;
    }
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    for node in value.atoms(Some(&|n: &Basic| n.is_function(Function::LerchPhi))) {
        if node.args()[0] == Basic::one() {
            let z: Basic = crate::functions::special::zeta_functions::zeta(&node.args()[1], Some(&node.args()[2]))?;
            replacement.insert(node.clone(), z);
        }
    }
    if !replacement.is_empty() {
        value = evaluated(&value, &replacement)?;
        replacement = HashMap::new();
    }
    for node in value.atoms(Some(&|n: &Basic| n.is_function(Function::Zeta))) {
        if node.args().len() == 2 {
            let (n, a) = (&node.args()[0], &node.args()[1]);
            if let Some(order) = integer_of(n).filter(|o| *o >= 2) {
                // zeta(n, a) = (-1)^n psi^(n-1)(a) / (n - 1)!
                let term: Basic = div(
                    &mul(&[
                        pow(&int(-1), n)?,
                        crate::functions::special::gamma_functions::polygamma(&int(order - 1), a)?,
                    ])?,
                    &crate::functions::combinatorial::factorials::factorial(&int(order - 1))?,
                )?;
                replacement.insert(node.clone(), term);
            }
        }
    }
    if replacement.is_empty() {
        Ok(value)
    } else {
        cancel(&evaluated(&value, &replacement)?)
    }
}
