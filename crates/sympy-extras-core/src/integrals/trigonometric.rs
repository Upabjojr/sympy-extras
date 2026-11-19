//! Port of `sympy_extras/integrals/trigonometric.py`: antiderivatives of
//! trigonometric and hyperbolic integrands by their reduction formulas and
//! substitutions (`sin**n*cos**m`, `tan**n`, `sec**n`, ..., their
//! hyperbolic counterparts, `x**k*exp(b*x)*sin(a*x)`, products to sums,
//! rational functions of `sin` and `cos` by `t = tan(a*x/2)`, the inverse
//! functions times polynomials), every result checked by differentiation.
//! The expressions are built in the order of the Python's operators.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_independent, factors_of, is, terms_of};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn ratio(p: i64, q: i64) -> Basic {
    Basic::from_rational(Rat::from(Int::from(p)) / Rat::from(Int::from(q)))
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
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn pi_(a: &Basic, k: i64) -> R {
    pw(a, &int(k))
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn sqrt(a: &Basic) -> R {
    pw(a, &ratio(1, 2))
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
fn f1(f: Function, a: &Basic) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), vec![a.clone()]), vec![a.clone()])
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn budget() -> Option<f64> {
    crate::settings::settings().timeout.map(|t| t / 4.0)
}
fn small(e: &Basic) -> Option<i64> {
    if e.is_integer_value() {
        i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
    } else {
        None
    }
}

use Function as F;
const TRIGONOMETRIC: &[Function] = &[F::Sin, F::Cos, F::Tan, F::Cot, F::Sec, F::Csc];
const HYPERBOLIC: &[Function] = &[F::Sinh, F::Cosh, F::Tanh, F::Coth, F::Sech, F::Csch];
const INVERSE: &[Function] = &[
    F::Asin,
    F::Acos,
    F::Atan,
    F::Acot,
    F::Asinh,
    F::Acosh,
    F::Atanh,
    F::Acoth,
];

fn kind_function(kind: &str) -> Function {
    match kind {
        "sin" => F::Sin,
        "cos" => F::Cos,
        "sinh" => F::Sinh,
        _ => F::Cosh,
    }
}

fn ratint(f: &Basic, v: &Basic) -> OptR {
    let symbol: Basic = crate::core::symbol::Symbol::new("t", &[]).0;
    crate::timeout::attempt_within(budget(), || {
        let integrand = crate::integrals::risch::rationaltools::Integrand::of(f)?;
        crate::integrals::risch::rationaltools::ratint(&integrand, v, &symbol, None)
    })
}

/// The parsed product (`_Term`).
struct Term {
    x: Basic,
    coefficient: Basic,
    polynomial: Basic,
    rate: Basic,
    powers: Vec<((&'static str, Basic), i64)>,
    inverse: Option<(Function, Basic, i64)>,
}

impl Term {
    fn kinds(&self) -> Vec<&'static str> {
        let mut k: Vec<&'static str> = self.powers.iter().map(|((kind, _), _)| *kind).collect();
        k.sort_unstable();
        k.dedup();
        k
    }
    fn trigonometric(&self) -> bool {
        self.powers.iter().any(|((k, _), _)| matches!(*k, "sin" | "cos"))
    }
    fn hyperbolic(&self) -> bool {
        self.powers.iter().any(|((k, _), _)| matches!(*k, "sinh" | "cosh"))
    }
    fn power(&self, kind: &str, a: &Basic) -> i64 {
        self.powers
            .iter()
            .find(|((k, b), _)| *k == kind && b == a)
            .map(|(_, p)| *p)
            .unwrap_or(0)
    }
}

fn frequency(argument: &Basic, x: &Basic) -> OptR {
    if !argument.has(x) {
        return Ok(None);
    }
    let a: Basic = cancel(&d2(argument, x)?)?;
    Ok(if a.has(x) { None } else { Some(a) })
}

fn sincos_powers(f: Function, k: i64) -> Vec<(&'static str, i64)> {
    let table: &[(&'static str, i64)] = match f {
        F::Sin => &[("sin", 1)],
        F::Cos => &[("cos", 1)],
        F::Tan => &[("sin", 1), ("cos", -1)],
        F::Cot => &[("cos", 1), ("sin", -1)],
        F::Sec => &[("cos", -1)],
        F::Csc => &[("sin", -1)],
        F::Sinh => &[("sinh", 1)],
        F::Cosh => &[("cosh", 1)],
        F::Tanh => &[("sinh", 1), ("cosh", -1)],
        F::Coth => &[("cosh", 1), ("sinh", -1)],
        F::Sech => &[("cosh", -1)],
        _ => &[("sinh", -1)],
    };
    table.iter().map(|(kind, p)| (*kind, p * k)).collect()
}

fn parse(term: &Basic, x: &Basic) -> PolyResult<Option<Term>> {
    let mut result = Term {
        x: x.clone(),
        coefficient: int(1),
        polynomial: int(1),
        rate: int(0),
        powers: Vec::new(),
        inverse: None,
    };
    for f in factors_of(term) {
        if !f.has(x) {
            result.coefficient = m2(&result.coefficient, &f)?;
            continue;
        }
        let (base, exponent) = if f.is_pow() {
            (f.args()[0].clone(), f.args()[1].clone())
        } else {
            (f.clone(), int(1))
        };
        if func(&base) == Some(F::Exp) && is(&exponent, Predicate::integer) == Some(true) {
            let Some(rate) = frequency(&base.args()[0], x)? else {
                return Ok(None);
            };
            result.rate = a2(&result.rate, &m2(&rate, &exponent)?)?;
            continue;
        }
        let fb: Option<Function> = func(&base);
        if fb.is_some_and(|g| TRIGONOMETRIC.contains(&g) || HYPERBOLIC.contains(&g)) && exponent.is_integer_value() {
            let Some(a) = frequency(&base.args()[0], x)? else {
                return Ok(None);
            };
            for (kind, power) in sincos_powers(fb.expect("a function"), small(&exponent).expect("an integer")) {
                match result.powers.iter_mut().find(|((k, b), _)| *k == kind && *b == a) {
                    Some(entry) => entry.1 += power,
                    None => result.powers.push(((kind, a.clone()), power)),
                }
            }
            continue;
        }
        if fb.is_some_and(|g| INVERSE.contains(&g)) && small(&exponent).is_some_and(|k| k > 0) {
            let c: Option<Basic> = frequency(&base.args()[0], x)?;
            let Some(c) = c else { return Ok(None) };
            if result.inverse.is_some() {
                return Ok(None);
            }
            result.inverse = Some((fb.expect("a function"), c, small(&exponent).expect("an integer")));
            continue;
        }
        if crate::core::expr::polynomial_like(
            &f,
            std::slice::from_ref(x),
            crate::core::expr::PolynomialKind::Polynomial,
        ) == Some(true)
        {
            result.polynomial = m2(&result.polynomial, &f)?;
            continue;
        }
        return Ok(None);
    }
    result.powers.retain(|(_, k)| *k != 0);
    Ok(Some(result))
}

fn laurent_antiderivative(e: &Basic, v: &Basic) -> OptR {
    let expanded: Basic = expand(e)?;
    let mut total: Basic = int(0);
    for term in terms_of(&expanded) {
        let (coefficient, rest) = as_independent(&term, v, false)?;
        if rest == int(1) {
            total = a2(&total, &m2(&coefficient, v)?)?;
        } else if rest == *v {
            total = a2(&total, &d2(&m2(&coefficient, &pi_(v, 2)?)?, &int(2))?)?;
        } else if rest.is_pow() && rest.args()[0] == *v && rest.args()[1].is_integer_value() {
            let k: i64 = small(&rest.args()[1]).expect("an integer");
            let piece: Basic = if k == -1 {
                log(v)?
            } else {
                d2(&pi_(v, k + 1)?, &int(k + 1))?
            };
            total = a2(&total, &m2(&coefficient, &piece)?)?;
        } else {
            return ratint(&expanded, v);
        }
    }
    Ok(Some(total))
}

fn odd_substitution(n: i64, m: i64) -> bool {
    n.rem_euclid(2) == 1 && !(m.rem_euclid(2) == 1 && (m > 0 && 0 > n || (n < 0 && m < 0 && n < m)))
}

fn sin_cos(n: i64, m: i64, u: &Basic) -> OptR {
    let (sin, cos) = (f1(F::Sin, u)?, f1(F::Cos, u)?);
    if n == 0 && m == 0 {
        return Ok(Some(u.clone()));
    }
    if n % 2 != 0 || m % 2 != 0 {
        let v: Basic = dummy("v");
        if odd_substitution(n, m) {
            // -(1 - v**2)**((n - 1) // 2) * v**m
            let integrand: Basic = m2(
                &neg(&pi_(&s2(&int(1), &pi_(&v, 2)?)?, (n - 1).div_euclid(2))?)?,
                &pi_(&v, m)?,
            )?;
            return Ok(match laurent_antiderivative(&integrand, &v)? {
                None => None,
                Some(found) => Some(subs(&found, &v, &cos)?),
            });
        }
        let integrand: Basic = m2(&pi_(&v, n)?, &pi_(&s2(&int(1), &pi_(&v, 2)?)?, (m - 1).div_euclid(2))?)?;
        return Ok(match laurent_antiderivative(&integrand, &v)? {
            None => None,
            Some(found) => Some(subs(&found, &v, &sin)?),
        });
    }
    if n < 0 {
        let Some(rest) = sin_cos(n + 2, m, u)? else {
            return Ok(None);
        };
        let first: Basic = d2(&m2(&pi_(&sin, n + 1)?, &pi_(&cos, m + 1)?)?, &int(n + 1))?;
        return Ok(Some(a2(&first, &m2(&ratio(n + m + 2, n + 1), &rest)?)?));
    }
    if m < 0 {
        let Some(rest) = sin_cos(n, m + 2, u)? else {
            return Ok(None);
        };
        let first: Basic = d2(&m2(&neg(&pi_(&sin, n + 1)?)?, &pi_(&cos, m + 1)?)?, &int(m + 1))?;
        return Ok(Some(a2(&first, &m2(&ratio(n + m + 2, m + 1), &rest)?)?));
    }
    if n > 0 {
        let Some(rest) = sin_cos(n - 2, m, u)? else {
            return Ok(None);
        };
        let first: Basic = d2(&m2(&neg(&pi_(&sin, n - 1)?)?, &pi_(&cos, m + 1)?)?, &int(n + m))?;
        return Ok(Some(a2(&first, &m2(&ratio(n - 1, n + m), &rest)?)?));
    }
    let Some(rest) = sin_cos(n, m - 2, u)? else {
        return Ok(None);
    };
    let first: Basic = d2(&m2(&pi_(&sin, n + 1)?, &pi_(&cos, m - 1)?)?, &int(n + m))?;
    Ok(Some(a2(&first, &m2(&ratio(m - 1, n + m), &rest)?)?))
}

fn tangent_family(n: i64, m: i64, u: &Basic) -> OptR {
    if n > 0 && m == -n {
        return Ok(Some(tan_power(n, u)?));
    }
    if m > 0 && n == -m {
        return Ok(Some(cot_power(m, u)?));
    }
    if n == 0 && m < 0 {
        return Ok(Some(sec_power(-m, u)?));
    }
    if m == 0 && n < 0 {
        return Ok(Some(csc_power(-n, u)?));
    }
    Ok(None)
}

fn tan_power(k: i64, u: &Basic) -> R {
    if k == 0 {
        return Ok(u.clone());
    }
    if k == 1 {
        return neg(&log(&f1(F::Cos, u)?)?);
    }
    s2(&d2(&pi_(&f1(F::Tan, u)?, k - 1)?, &int(k - 1))?, &tan_power(k - 2, u)?)
}

fn cot_power(k: i64, u: &Basic) -> R {
    if k == 0 {
        return Ok(u.clone());
    }
    if k == 1 {
        return log(&f1(F::Sin, u)?);
    }
    s2(
        &d2(&neg(&pi_(&f1(F::Cot, u)?, k - 1)?)?, &int(k - 1))?,
        &cot_power(k - 2, u)?,
    )
}

fn sec_power(k: i64, u: &Basic) -> R {
    if k == 1 {
        return log(&a2(&f1(F::Sec, u)?, &f1(F::Tan, u)?)?);
    }
    if k == 2 {
        return f1(F::Tan, u);
    }
    let first: Basic = d2(&m2(&pi_(&f1(F::Sec, u)?, k - 2)?, &f1(F::Tan, u)?)?, &int(k - 1))?;
    a2(&first, &m2(&ratio(k - 2, k - 1), &sec_power(k - 2, u)?)?)
}

fn csc_power(k: i64, u: &Basic) -> R {
    if k == 1 {
        return log(&f1(F::Tan, &d2(u, &int(2))?)?);
    }
    if k == 2 {
        return neg(&f1(F::Cot, u)?);
    }
    let first: Basic = d2(&m2(&neg(&pi_(&f1(F::Csc, u)?, k - 2)?)?, &f1(F::Cot, u)?)?, &int(k - 1))?;
    a2(&first, &m2(&ratio(k - 2, k - 1), &csc_power(k - 2, u)?)?)
}

fn sinh_cosh(n: i64, m: i64, u: &Basic) -> OptR {
    let (sinh, cosh) = (f1(F::Sinh, u)?, f1(F::Cosh, u)?);
    if n == 0 && m == 0 {
        return Ok(Some(u.clone()));
    }
    if n % 2 != 0 || m % 2 != 0 {
        let v: Basic = dummy("v");
        if odd_substitution(n, m) {
            let integrand: Basic = m2(&pi_(&s2(&pi_(&v, 2)?, &int(1))?, (n - 1).div_euclid(2))?, &pi_(&v, m)?)?;
            return Ok(match laurent_antiderivative(&integrand, &v)? {
                None => None,
                Some(found) => Some(subs(&found, &v, &cosh)?),
            });
        }
        let integrand: Basic = m2(&pi_(&v, n)?, &pi_(&a2(&pi_(&v, 2)?, &int(1))?, (m - 1).div_euclid(2))?)?;
        return Ok(match laurent_antiderivative(&integrand, &v)? {
            None => None,
            Some(found) => Some(subs(&found, &v, &sinh)?),
        });
    }
    if n < 0 {
        let Some(rest) = sinh_cosh(n + 2, m, u)? else {
            return Ok(None);
        };
        let first: Basic = d2(&m2(&pi_(&sinh, n + 1)?, &pi_(&cosh, m + 1)?)?, &int(n + 1))?;
        return Ok(Some(s2(&first, &m2(&ratio(n + m + 2, n + 1), &rest)?)?));
    }
    if m < 0 {
        let Some(rest) = sinh_cosh(n, m + 2, u)? else {
            return Ok(None);
        };
        let first: Basic = d2(&m2(&neg(&pi_(&sinh, n + 1)?)?, &pi_(&cosh, m + 1)?)?, &int(m + 1))?;
        return Ok(Some(a2(&first, &m2(&ratio(n + m + 2, m + 1), &rest)?)?));
    }
    if n > 0 {
        let Some(rest) = sinh_cosh(n - 2, m, u)? else {
            return Ok(None);
        };
        let first: Basic = d2(&m2(&pi_(&sinh, n - 1)?, &pi_(&cosh, m + 1)?)?, &int(n + m))?;
        return Ok(Some(s2(&first, &m2(&ratio(n - 1, n + m), &rest)?)?));
    }
    let Some(rest) = sinh_cosh(n, m - 2, u)? else {
        return Ok(None);
    };
    let first: Basic = d2(&m2(&pi_(&sinh, n + 1)?, &pi_(&cosh, m - 1)?)?, &int(n + m))?;
    Ok(Some(a2(&first, &m2(&ratio(m - 1, n + m), &rest)?)?))
}

fn hyperbolic_family(n: i64, m: i64, u: &Basic) -> OptR {
    if n > 0 && m == -n {
        return Ok(Some(tanh_power(n, u)?));
    }
    if m > 0 && n == -m {
        return Ok(Some(coth_power(m, u)?));
    }
    if n == 0 && m < 0 {
        return Ok(Some(sech_power(-m, u)?));
    }
    if m == 0 && n < 0 {
        return Ok(Some(csch_power(-n, u)?));
    }
    Ok(None)
}

fn tanh_power(k: i64, u: &Basic) -> R {
    if k == 0 {
        return Ok(u.clone());
    }
    if k == 1 {
        return log(&f1(F::Cosh, u)?);
    }
    a2(
        &d2(&neg(&pi_(&f1(F::Tanh, u)?, k - 1)?)?, &int(k - 1))?,
        &tanh_power(k - 2, u)?,
    )
}

fn coth_power(k: i64, u: &Basic) -> R {
    if k == 0 {
        return Ok(u.clone());
    }
    if k == 1 {
        return log(&f1(F::Sinh, u)?);
    }
    a2(
        &d2(&neg(&pi_(&f1(F::Coth, u)?, k - 1)?)?, &int(k - 1))?,
        &coth_power(k - 2, u)?,
    )
}

fn sech_power(k: i64, u: &Basic) -> R {
    if k == 1 {
        return f1(F::Atan, &f1(F::Sinh, u)?);
    }
    if k == 2 {
        return f1(F::Tanh, u);
    }
    let first: Basic = d2(&m2(&pi_(&f1(F::Sech, u)?, k - 2)?, &f1(F::Tanh, u)?)?, &int(k - 1))?;
    a2(&first, &m2(&ratio(k - 2, k - 1), &sech_power(k - 2, u)?)?)
}

fn csch_power(k: i64, u: &Basic) -> R {
    if k == 1 {
        return log(&f1(F::Tanh, &d2(u, &int(2))?)?);
    }
    if k == 2 {
        return neg(&f1(F::Coth, u)?);
    }
    let first: Basic = d2(
        &m2(&neg(&pi_(&f1(F::Csch, u)?, k - 2)?)?, &f1(F::Coth, u)?)?,
        &int(k - 1),
    )?;
    s2(&first, &m2(&ratio(k - 2, k - 1), &csch_power(k - 2, u)?)?)
}

fn powers_of_one_argument(term: &Term) -> OptR {
    if term.polynomial != int(1) || term.rate != int(0) || term.inverse.is_some() || term.powers.is_empty() {
        return Ok(None);
    }
    let kinds: Vec<&'static str> = term.kinds();
    let mut frequencies: Vec<Basic> = Vec::new();
    for ((k, a), _) in &term.powers {
        if kinds.contains(k) && !frequencies.contains(a) {
            frequencies.push(a.clone());
        }
    }
    if frequencies.len() != 1 {
        return Ok(None);
    }
    let a: Basic = frequencies[0].clone();
    let hyperbolic: bool = term.hyperbolic();
    if hyperbolic && term.trigonometric() {
        return Ok(None);
    }
    let n: i64 = term.power(if hyperbolic { "sinh" } else { "sin" }, &a);
    let m: i64 = term.power(if hyperbolic { "cosh" } else { "cos" }, &a);
    let u: Basic = dummy("u");
    let found: Option<Basic> = if hyperbolic {
        match hyperbolic_family(n, m, &u)? {
            Some(v) => Some(v),
            None => sinh_cosh(n, m, &u)?,
        }
    } else {
        match tangent_family(n, m, &u)? {
            Some(v) => Some(v),
            None => sin_cos(n, m, &u)?,
        }
    };
    let Some(found) = found else { return Ok(None) };
    Ok(Some(d2(
        &m2(&term.coefficient, &subs(&found, &u, &m2(&a, &term.x)?)?)?,
        &a,
    )?))
}

fn factorial_i(n: i64) -> Int {
    (1..=n).fold(Int::ONE, |acc, k| acc * Int::from(k))
}

fn x_power_exp(k: i64, c: &Basic, x: &Basic) -> R {
    if *c == int(0) {
        return d2(&pi_(x, k + 1)?, &int(k + 1));
    }
    let mut total: Basic = int(0);
    for j in 0..=k {
        let sign: Int = if j % 2 == 0 { Int::ONE } else { -Int::ONE };
        let coefficient: Basic = Basic::from_rational(Rat::from(sign * (factorial_i(k) / factorial_i(k - j))));
        total = a2(&total, &d2(&m2(&coefficient, &pi_(x, k - j)?)?, &pi_(c, j + 1)?)?)?;
    }
    m2(&exp(&m2(c, x)?)?, &total)
}

fn x_power_exp_trig(k: i64, b: &Basic, a: &Basic, kind: &str, x: &Basic) -> R {
    let den: Basic = a2(&pi_(a, 2)?, &pi_(b, 2)?)?;
    let ax: Basic = m2(a, x)?;
    let (sin, cos) = (f1(F::Sin, &ax)?, f1(F::Cos, &ax)?);
    let ebx: Basic = exp(&m2(b, x)?)?;
    let i0: Basic = d2(&m2(&ebx, &s2(&m2(b, &sin)?, &m2(a, &cos)?)?)?, &den)?;
    let j0: Basic = d2(&m2(&ebx, &a2(&m2(b, &cos)?, &m2(a, &sin)?)?)?, &den)?;
    let (mut i_prev, mut j_prev) = (i0.clone(), j0.clone());
    for j in 1..=k {
        let i_next: Basic = s2(
            &m2(&pi_(x, j)?, &i0)?,
            &d2(&m2(&int(j), &s2(&m2(b, &i_prev)?, &m2(a, &j_prev)?)?)?, &den)?,
        )?;
        let j_next: Basic = s2(
            &m2(&pi_(x, j)?, &j0)?,
            &d2(&m2(&int(j), &a2(&m2(b, &j_prev)?, &m2(a, &i_prev)?)?)?, &den)?,
        )?;
        i_prev = i_next;
        j_prev = j_next;
    }
    Ok(if kind == "sin" { i_prev } else { j_prev })
}

fn single_harmonic(term: &Term) -> OptR {
    if term.inverse.is_some() || term.hyperbolic() {
        return Ok(None);
    }
    let x: &Basic = &term.x;
    let mut harmonic: Option<(&'static str, Basic)> = None;
    if !term.powers.is_empty() {
        if term.powers.len() != 1 {
            return Ok(None);
        }
        let ((kind, a), power) = &term.powers[0];
        if *power != 1 {
            return Ok(None);
        }
        harmonic = Some((kind, a.clone()));
    }
    let Some(coefficients) = crate::integrals::radicals::coefficients(&term.polynomial, x)? else {
        return Ok(None);
    };
    let mut ks: Vec<i64> = coefficients.iter().map(|(k, _)| *k).collect();
    ks.sort_unstable_by(|p, q| q.cmp(p));
    let mut total: Basic = int(0);
    for k in ks {
        let c: Basic = coefficients
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.clone())
            .expect("a coefficient");
        let piece: Basic = match &harmonic {
            None => m2(&c, &x_power_exp(k, &term.rate, x)?)?,
            Some((kind, a)) => {
                if *a == int(0) {
                    if *kind == "cos" {
                        m2(&c, &x_power_exp(k, &term.rate, x)?)?
                    } else {
                        int(0)
                    }
                } else {
                    m2(&c, &x_power_exp_trig(k, &term.rate, a, kind, x)?)?
                }
            }
        };
        total = a2(&total, &piece)?;
    }
    Ok(Some(m2(&term.coefficient, &total)?))
}

fn atoms_with(e: &Basic, fs: &[Function]) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if func(&n).is_some_and(|f| fs.contains(&f)) && !found.contains(&n) {
            found.push(n);
        }
    }
    found
}

fn in_exponentials(e: &Basic) -> R {
    let i: Basic = Basic::atom(Head::ImaginaryUnit);
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    for node in atoms_with(e, &[F::Sin, F::Cos, F::Sinh, F::Cosh]) {
        let u: Basic = node.args()[0].clone();
        let value: Basic = match func(&node).expect("a function") {
            F::Sin => {
                let iu: Basic = m2(&i, &u)?;
                d2(&m2(&neg(&i)?, &s2(&exp(&iu)?, &exp(&neg(&iu)?)?)?)?, &int(2))?
            }
            F::Cos => {
                let iu: Basic = m2(&i, &u)?;
                a2(&d2(&exp(&iu)?, &int(2))?, &d2(&exp(&neg(&iu)?)?, &int(2))?)?
            }
            F::Sinh => s2(&d2(&exp(&u)?, &int(2))?, &d2(&exp(&neg(&u)?)?, &int(2))?)?,
            _ => a2(&d2(&exp(&u)?, &int(2))?, &d2(&exp(&neg(&u)?)?, &int(2))?)?,
        };
        replacement.insert(node, value);
    }
    Ok(if replacement.is_empty() {
        e.clone()
    } else {
        e.xreplace(&replacement)
    })
}

fn solutions(c: &Basic, s: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let poly = match crate::polys::polytools::Poly::new(c, std::slice::from_ref(s)) {
        Ok(p) => p,
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let degree: i64 = poly.degree(0)?;
    if degree < 1 {
        return Ok(None);
    }
    let found = crate::timeout::attempt_within(budget(), || {
        crate::polys::polyroots::roots(c, std::slice::from_ref(s), true, true, None)
    })?;
    let Some(found) = found else { return Ok(None) };
    if found.iter().map(|(_, m)| *m as i64).sum::<i64>() != degree {
        return Ok(None);
    }
    let assumptions: Vec<(Predicate, Option<bool>)> = match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.assumptions.iter().map(|(p, v)| (*p, Some(*v))).collect(),
        _ => vec![],
    };
    Ok(Some(
        found
            .iter()
            .map(|(r, _)| r.clone())
            .filter(|r| crate::core::assumptions::check_assumptions(r, &assumptions) != Some(false))
            .collect(),
    ))
}

type Integrate<'a> = &'a mut dyn FnMut(&Basic) -> OptR;

fn product_to_sum(term: &Term, integrate: Integrate) -> OptR {
    if term.inverse.is_some() || term.powers.is_empty() {
        return Ok(None);
    }
    let x: &Basic = &term.x;
    let mut product: Basic = int(1);
    for ((kind, a), k) in &term.powers {
        product = m2(&product, &pi_(&f1(kind_function(kind), &m2(a, x)?)?, *k)?)?;
    }
    let expanded: Basic = if term.hyperbolic() {
        expand(&in_exponentials(&product)?)?
    } else {
        expand(&crate::simplify::fu::tr8(&product, true)?)?
    };
    if expanded == product {
        return Ok(None);
    }
    let integrand: Basic = expand(&m2(
        &m2(&m2(&term.coefficient, &term.polynomial)?, &exp(&m2(&term.rate, x)?)?)?,
        &expanded,
    )?)?;
    let mut doubtful: Vec<Basic> = Vec::new();
    for node in atoms_with(&integrand, &[F::Sin, F::Cos]) {
        if let Some(c) = frequency(&node.args()[0], x)? {
            if !c.free_symbols().is_empty() && is(&c, Predicate::zero).is_none() && !doubtful.contains(&c) {
                doubtful.push(c);
            }
        }
    }
    let Some(generic) = integrate(&integrand)? else {
        return Ok(None);
    };
    if doubtful.len() != 1 {
        return Ok(Some(generic));
    }
    let c: Basic = doubtful[0].clone();
    let mut symbols: Vec<Basic> = c.free_symbols().into_iter().filter(|s| s.is_symbol()).collect();
    symbols.sort_by_key(|s| s.to_string());
    let last: Basic = symbols.last().cloned().expect("a symbol");
    let Some(found) = solutions(&c, &last)? else {
        return Ok(Some(generic));
    };
    if found.len() != 1 {
        return Ok(Some(generic));
    }
    let special_integrand: Basic = subs(&integrand, &last, &found[0])?;
    let Some(special) = integrate(&special_integrand)? else {
        return Ok(Some(generic));
    };
    if !verified(&special, &special_integrand, x)? {
        return Ok(Some(generic));
    }
    Ok(Some(crate::functions::elementary::piecewise::piecewise(&[
        (generic, crate::core::relational::ne(&c, &int(0))?),
        (special, crate::core::logic::boolean(true)),
    ])?))
}

fn in_sines_and_cosines(f: &Basic) -> R {
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    let rules: [(Function, fn(&Basic) -> R); 8] = [
        (F::Tan, |u| d2(&f1(F::Sin, u)?, &f1(F::Cos, u)?)),
        (F::Cot, |u| d2(&f1(F::Cos, u)?, &f1(F::Sin, u)?)),
        (F::Sec, |u| d2(&int(1), &f1(F::Cos, u)?)),
        (F::Csc, |u| d2(&int(1), &f1(F::Sin, u)?)),
        (F::Tanh, |u| d2(&f1(F::Sinh, u)?, &f1(F::Cosh, u)?)),
        (F::Coth, |u| d2(&f1(F::Cosh, u)?, &f1(F::Sinh, u)?)),
        (F::Sech, |u| d2(&int(1), &f1(F::Cosh, u)?)),
        (F::Csch, |u| d2(&int(1), &f1(F::Sinh, u)?)),
    ];
    for (g, rule) in rules {
        for node in atoms_with(f, &[g]) {
            let v: Basic = rule(&node.args()[0])?;
            replacement.insert(node, v);
        }
    }
    Ok(if replacement.is_empty() {
        f.clone()
    } else {
        f.xreplace(&replacement)
    })
}

fn rational_in_sincos(f: &Basic, x: &Basic) -> OptR {
    let g: Basic = in_sines_and_cosines(f)?;
    let mut frequencies: Vec<Basic> = Vec::new();
    let mut kinds: Vec<&str> = Vec::new();
    for node in atoms_with(&g, &[F::Sin, F::Cos, F::Sinh, F::Cosh]) {
        let Some(a) = frequency(&node.args()[0], x)? else {
            return Ok(None);
        };
        if !frequencies.contains(&a) {
            frequencies.push(a);
        }
        let k: &str = if matches!(func(&node), Some(F::Sinh | F::Cosh)) {
            "hyperbolic"
        } else {
            "trigonometric"
        };
        if !kinds.contains(&k) {
            kinds.push(k);
        }
    }
    if frequencies.len() != 1 || kinds.len() != 1 {
        return Ok(None);
    }
    let a: Basic = frequencies[0].clone();
    let (s, c, t) = (dummy("s"), dummy("c"), dummy("t"));
    let ax: Basic = m2(&a, x)?;
    let t2: Basic = pi_(&t, 2)?;
    let rational_in = |h: &Basic| {
        crate::core::expr::polynomial_like(h, &[s.clone(), c.clone()], crate::core::expr::PolynomialKind::Rational)
            == Some(true)
    };
    let (rational, back): (Basic, Basic) = if kinds[0] == "trigonometric" {
        let to_symbols: HashMap<Basic, Basic> = [(f1(F::Sin, &ax)?, s.clone()), (f1(F::Cos, &ax)?, c.clone())]
            .into_iter()
            .collect();
        let one_t2: Basic = a2(&int(1), &t2)?;
        let to_t: HashMap<Basic, Basic> = [
            (s.clone(), d2(&m2(&int(2), &t)?, &one_t2)?),
            (c.clone(), d2(&s2(&int(1), &t2)?, &one_t2)?),
        ]
        .into_iter()
        .collect();
        let h: Basic = g.xreplace(&to_symbols);
        if h.has(x) || !rational_in(&h) {
            return Ok(None);
        }
        // cancel(h.xreplace(to_t) * 2 / (a * (1 + t**2)))
        let r: Basic = cancel(&d2(&m2(&h.xreplace(&to_t), &int(2))?, &m2(&a, &one_t2)?)?)?;
        (r, f1(F::Tan, &d2(&ax, &int(2))?)?)
    } else {
        let to_symbols: HashMap<Basic, Basic> = [(f1(F::Sinh, &ax)?, s.clone()), (f1(F::Cosh, &ax)?, c.clone())]
            .into_iter()
            .collect();
        let inv_t: Basic = d2(&int(1), &t)?;
        let to_t: HashMap<Basic, Basic> = [
            (s.clone(), d2(&s2(&t, &inv_t)?, &int(2))?),
            (c.clone(), d2(&a2(&t, &inv_t)?, &int(2))?),
        ]
        .into_iter()
        .collect();
        let h: Basic = g.xreplace(&to_symbols);
        if h.has(x) || !rational_in(&h) {
            return Ok(None);
        }
        let r: Basic = cancel(&d2(&h.xreplace(&to_t), &m2(&a, &t)?)?)?;
        (r, exp(&ax)?)
    };
    let Some(found) = ratint(&rational, &t)? else {
        return Ok(None);
    };
    Ok(Some(subs(&found, &t, &back)?))
}

fn inverse_times_power(term: &Term) -> OptR {
    let Some((function, c, power)) = term.inverse.clone() else {
        return Ok(None);
    };
    if !term.powers.is_empty() || term.rate != int(0) {
        return Ok(None);
    }
    let x: &Basic = &term.x;
    let argument: Basic = m2(&c, x)?;
    let arg2: Basic = pi_(&argument, 2)?;
    let derivative: Basic = match function {
        F::Asin => d2(&int(1), &sqrt(&s2(&int(1), &arg2)?)?)?,
        F::Acos => d2(&int(-1), &sqrt(&s2(&int(1), &arg2)?)?)?,
        F::Atan => d2(&int(1), &a2(&int(1), &arg2)?)?,
        F::Acot => d2(&int(-1), &a2(&int(1), &arg2)?)?,
        F::Asinh => d2(&int(1), &sqrt(&a2(&int(1), &arg2)?)?)?,
        F::Acosh => d2(&int(1), &sqrt(&s2(&arg2, &int(1))?)?)?,
        _ => d2(&int(1), &s2(&int(1), &arg2)?)?,
    };
    let g: Basic = f1(function, &argument)?;
    if power == 1 {
        let Some(coefficients) = crate::integrals::radicals::coefficients(&term.polynomial, x)? else {
            return Ok(None);
        };
        let mut ks: Vec<i64> = coefficients.iter().map(|(k, _)| *k).collect();
        ks.sort_unstable_by(|p, q| q.cmp(p));
        let mut primitive: Basic = int(0);
        for n in ks {
            let cn: Basic = coefficients
                .iter()
                .find(|(k, _)| *k == n)
                .map(|(_, v)| v.clone())
                .expect("a coefficient");
            primitive = a2(&primitive, &d2(&m2(&cn, &pi_(x, n + 1)?)?, &int(n + 1))?)?;
        }
        let remainder: Basic = cancel(&m2(&m2(&primitive, &derivative)?, &c)?)?;
        let Some(rest) = algebraic_or_rational(&remainder, x)? else {
            return Ok(None);
        };
        return Ok(Some(m2(&term.coefficient, &s2(&m2(&primitive, &g)?, &rest)?)?));
    }
    if power == 2 && term.polynomial == int(1) {
        let g2: Basic = pi_(&g, 2)?;
        let xg2: Basic = m2(x, &g2)?;
        let two_x: Basic = m2(&int(2), x)?;
        let radical = |inside: Basic| -> R { d2(&m2(&m2(&int(2), &sqrt(&inside)?)?, &g)?, &c) };
        let square: Basic = match function {
            F::Asin => s2(&a2(&xg2, &radical(s2(&int(1), &arg2)?)?)?, &two_x)?,
            F::Acos => s2(&s2(&xg2, &radical(s2(&int(1), &arg2)?)?)?, &two_x)?,
            F::Asinh => a2(&s2(&xg2, &radical(a2(&int(1), &arg2)?)?)?, &two_x)?,
            F::Acosh => a2(&s2(&xg2, &radical(s2(&arg2, &int(1))?)?)?, &two_x)?,
            _ => return Ok(None),
        };
        return Ok(Some(m2(&term.coefficient, &square)?));
    }
    Ok(None)
}

fn algebraic_or_rational(f: &Basic, x: &Basic) -> OptR {
    if crate::core::expr::polynomial_like(f, std::slice::from_ref(x), crate::core::expr::PolynomialKind::Rational)
        == Some(true)
    {
        return ratint(f, x);
    }
    crate::integrals::radicals::quadratic_radical_antiderivative(f, x, &[], true)
}

fn facts_of(f: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    use crate::core::relational::{gt, lt};
    let mut facts: Vec<Basic> = Vec::new();
    let mut seen: Vec<Basic> = Vec::new();
    for node in f.preorder() {
        if node.is_pow()
            && node.args()[1].is_rational_class()
            && crate::numbers::denom(node.args()[1].as_rational().expect("rational")) == Int::from(2u8)
            && node.args()[0].has(x)
            && !seen.contains(&node)
        {
            seen.push(node.clone());
            facts.push(gt(&node.args()[0], &int(0))?);
        }
    }
    for node in atoms_with(f, &[F::Asin, F::Acos, F::Atanh]) {
        facts.push(gt(&node.args()[0], &int(-1))?);
        facts.push(lt(&node.args()[0], &int(1))?);
    }
    for node in atoms_with(f, &[F::Acosh]) {
        facts.push(gt(&node.args()[0], &int(1))?);
    }
    for node in atoms_with(f, &[F::Acoth]) {
        facts.push(gt(&node.args()[0], &int(1))?);
    }
    Ok(facts)
}

fn verified(big_f: &Basic, f: &Basic, x: &Basic) -> PolyResult<bool> {
    if matches!(big_f.head(), Head::Piecewise) {
        let pair: &Basic = &big_f.args()[0];
        return verified(&pair.args()[0], f, x);
    }
    let derivative: Basic = crate::core::function::derivative(big_f, x)?;
    let difference: Basic = s2(&derivative, f)?;
    if crate::timeout::attempt_within(budget(), || cancel(&difference))?.is_some_and(|z| z == int(0)) {
        return Ok(true);
    }
    let simplified = crate::timeout::attempt_within(budget(), || {
        crate::simplify::simplify::simplify(&difference, &crate::simplify::simplify::Options::default())
    })?;
    if simplified.is_some_and(|z| z == int(0)) {
        return Ok(true);
    }
    crate::integrals::conditions::numerically_equal(&derivative, f, &facts_of(f, x)?, 3)
}

fn simpler(big_f: &Basic) -> R {
    let candidate = crate::timeout::attempt_within(budget(), || {
        crate::core::exprtools::factor_terms(big_f, false, false, false, true)
    })?;
    if let Some(c) = candidate {
        if crate::core::function::count_ops(&c)? < crate::core::function::count_ops(big_f)? {
            return Ok(c);
        }
    }
    Ok(big_f.clone())
}

fn antiderivative(f: &Basic, x: &Basic, depth: usize) -> OptR {
    if depth > 3 {
        return Ok(None);
    }
    if !f.has(x) {
        return Ok(Some(m2(f, x)?));
    }
    if f.is_add() {
        let mut total: Basic = int(0);
        for term in f.args() {
            let Some(found) = antiderivative(term, x, depth)? else {
                return Ok(None);
            };
            total = a2(&total, &found)?;
        }
        return Ok(Some(total));
    }
    let (coefficient, rest) = as_independent(f, x, false)?;
    if coefficient != int(1) {
        return Ok(match antiderivative(&rest, x, depth)? {
            None => None,
            Some(found) => Some(m2(&coefficient, &found)?),
        });
    }
    if let Some(structure) = parse(f, x)? {
        if let Some(found) = powers_of_one_argument(&structure)? {
            return Ok(Some(found));
        }
        if let Some(found) = single_harmonic(&structure)? {
            return Ok(Some(found));
        }
        if let Some(found) = inverse_times_power(&structure)? {
            return Ok(Some(found));
        }
        let mut deeper = |g: &Basic| antiderivative(g, x, depth + 1);
        if let Some(found) = product_to_sum(&structure, &mut deeper)? {
            return Ok(Some(found));
        }
    }
    rational_in_sincos(f, x)
}

/// An antiderivative of a trigonometric or hyperbolic integrand, checked
/// by differentiation (`trigonometric_antiderivative`); `None` otherwise.
pub fn trigonometric_antiderivative(f: &Basic, x: &Basic) -> OptR {
    let all: Vec<Function> = TRIGONOMETRIC.iter().chain(HYPERBOLIC).chain(INVERSE).copied().collect();
    if atoms_with(f, &all).is_empty() {
        return Ok(None);
    }
    let Some(found) = antiderivative(f, x, 0)? else {
        return Ok(None);
    };
    let found: Basic = simpler(&found)?;
    Ok(if verified(&found, f, x)? { Some(found) } else { None })
}
