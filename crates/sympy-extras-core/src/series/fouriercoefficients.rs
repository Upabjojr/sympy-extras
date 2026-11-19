//! Port of `sympy_extras/series/fouriercoefficients.py`: the Fourier
//! coefficients of piecewise polynomial functions on an interval, in
//! closed form in the index `n` (repeated integration by parts), and the
//! partial sums of the series.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::{value_error, PolyError, PolyResult};
use crate::simplify::normalform::normal_form;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
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
fn trig(f: Function, a: &Basic) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), vec![a.clone()]), vec![a.clone()])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn positive(e: &Basic) -> bool {
    is(e, Predicate::positive) == Some(true)
}
fn is_polynomial(e: &Basic, x: &Basic) -> bool {
    crate::core::expr::polynomial_like(
        e,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true)
}

/// The coefficients `a0`, `an`, `bn` in the index `n` (`FourierCoefficients`).
#[derive(Debug, Clone)]
pub struct FourierCoefficients {
    pub a0: Basic,
    pub an: Basic,
    pub bn: Basic,
    pub n: Basic,
    pub x: Basic,
    pub omega: Basic,
}

impl FourierCoefficients {
    /// `(a_k, b_k)` (`coefficients`).
    pub fn coefficients(&self, k: i64) -> PolyResult<(Basic, Basic)> {
        if k < 1 {
            return Err(value_error("the index of a harmonic is at least one".to_string()));
        }
        Ok((subs(&self.an, &self.n, &int(k))?, subs(&self.bn, &self.n, &int(k))?))
    }

    /// The `k`-th term of the series (`term`).
    pub fn term(&self, k: i64) -> R {
        if k == 0 {
            return d2(&self.a0, &int(2));
        }
        let (a, b) = self.coefficients(k)?;
        let angle: Basic = m2(&m2(&int(k), &self.omega)?, &self.x)?;
        a2(
            &m2(&a, &trig(Function::Cos, &angle)?)?,
            &m2(&b, &trig(Function::Sin, &angle)?)?,
        )
    }

    /// The partial sum up to the harmonic `order` (`partial_sum`).
    pub fn partial_sum(&self, order: i64) -> R {
        let terms: Vec<Basic> = (0..=order).map(|k| self.term(k)).collect::<PolyResult<_>>()?;
        crate::core::add::add(&terms)
    }

    pub fn repr(&self) -> String {
        format!("FourierCoefficients(a0={}, an={}, bn={})", self.a0, self.an, self.bn)
    }
}

fn pieces(f: &Basic, x: &Basic, low: &Basic, high: &Basic) -> PolyResult<Vec<(Basic, Basic, Basic)>> {
    if !matches!(f.head(), Head::Piecewise) {
        return Ok(vec![(f.clone(), low.clone(), high.clone())]);
    }
    let mut result: Vec<(Basic, Basic, Basic)> = Vec::new();
    let mut start: Basic = low.clone();
    for pair in f.args() {
        let (expression, condition) = (&pair.args()[0], &pair.args()[1]);
        let mut end: Basic = if matches!(condition.head(), Head::BooleanTrue) {
            high.clone()
        } else if matches!(condition.head(), Head::Relational(RelOp::Lt | RelOp::Le))
            && condition.args()[0] == *x
            && !condition.args()[1].has(x)
        {
            condition.args()[1].clone()
        } else {
            return Err(PolyError::NotImplemented(format!(
                "the condition {}: the pieces must be given by x < c or x <= c in increasing order",
                condition
            )));
        };
        if positive(&s2(&end, high)?) {
            end = high.clone();
        }
        let gap: Basic = s2(&end, &start)?;
        if positive(&gap) {
            result.push((expression.clone(), start.clone(), end.clone()));
            start = end;
        } else if is(&gap, Predicate::zero) != Some(true) && is(&gap, Predicate::negative) != Some(true) {
            return Err(PolyError::NotImplemented(format!(
                "the order of the points {} and {} is not known",
                start, end
            )));
        }
        if start == *high {
            break;
        }
    }
    if start != *high {
        return Err(value_error(format!("the function is not defined up to {}", high)));
    }
    Ok(result)
}

fn definite(p: &Basic, x: &Basic, k: &Basic, sine: bool, low: &Basic, high: &Basic) -> R {
    let mut derivatives: Vec<Basic> = Vec::new();
    let mut current: Basic = expand(p)?;
    let mut ended: bool = false;
    for _ in 0..64 {
        if current == int(0) {
            ended = true;
            break;
        }
        derivatives.push(current.clone());
        current = expand(&crate::core::function::derivative(&current, x)?)?;
    }
    if !ended || derivatives.iter().any(|d| !is_polynomial(d, x)) {
        return Err(PolyError::NotImplemented(format!("{} is not a polynomial in {}", p, x)));
    }
    let antiderivative = |point: &Basic| -> R {
        let kp: Basic = m2(k, point)?;
        let (s, c) = (trig(Function::Sin, &kp)?, trig(Function::Cos, &kp)?);
        let cycle: [Basic; 4] = if !sine {
            [s.clone(), neg(&c)?, neg(&s)?, c.clone()]
        } else {
            [neg(&c)?, neg(&s)?, c.clone(), s.clone()]
        };
        let mut total: Vec<Basic> = Vec::new();
        for (j, derivative) in derivatives.iter().enumerate() {
            let sign: i64 = if j % 2 == 1 { -1 } else { 1 };
            let value: Basic = m2(&m2(&int(sign), &subs(derivative, x, point)?)?, &cycle[j % 4])?;
            total.push(d2(&value, &pw(k, &int(j as i64 + 1))?)?);
        }
        crate::core::add::add(&total)
    };
    s2(&antiderivative(high)?, &antiderivative(low)?)
}

fn polynomial_integral(p: &Basic, x: &Basic, low: &Basic, high: &Basic) -> R {
    let polynomial: Basic = expand(p)?;
    if !is_polynomial(&polynomial, x) {
        return Err(PolyError::NotImplemented(format!("{} is not a polynomial in {}", p, x)));
    }
    let terms: Vec<Basic> = if polynomial.is_add() {
        polynomial.args().to_vec()
    } else {
        vec![polynomial.clone()]
    };
    let mut total: Vec<Basic> = Vec::new();
    for term in &terms {
        let (coefficient, mut power) = crate::core::expr::as_independent(term, x, false)?;
        let mut degree: i64 = 0;
        while power != int(1) && degree < 4096 {
            degree += 1;
            power = d2(&power, x)?;
        }
        let primitive: Basic = d2(&m2(&coefficient, &pw(x, &int(degree + 1))?)?, &int(degree + 1))?;
        total.push(s2(&subs(&primitive, x, high)?, &subs(&primitive, x, low)?)?);
    }
    crate::core::add::add(&total)
}

/// The Fourier coefficients of `f` on `(low, high)` (`fourier_coefficients`).
pub fn fourier_coefficients(
    f: &Basic,
    x: &Basic,
    interval: (&Basic, &Basic),
    n: Option<&Basic>,
) -> PolyResult<FourierCoefficients> {
    let (low, high) = interval;
    let period: Basic = s2(high, low)?;
    if !positive(&period) {
        return Err(value_error(format!(
            "the interval ({}, {}) is not increasing",
            low, high
        )));
    }
    let index: Basic = match n {
        Some(n) => n.clone(),
        None => crate::core::symbol::Symbol::new("n", &[(Predicate::integer, true), (Predicate::positive, true)]).0,
    };
    if is(&index, Predicate::integer) != Some(true) || is(&index, Predicate::positive) != Some(true) {
        return Err(value_error(
            "the index must be a symbol declared a positive integer".to_string(),
        ));
    }
    let omega: Basic = d2(&m2(&int(2), &Basic::pi())?, &period)?;
    let found = pieces(f, x, low, high)?;
    let k: Basic = m2(&index, &omega)?;
    let mean: Basic = crate::core::add::add(
        &found
            .iter()
            .map(|(p, s, e)| polynomial_integral(p, x, s, e))
            .collect::<PolyResult<Vec<_>>>()?,
    )?;
    let cosine: Basic = crate::core::add::add(
        &found
            .iter()
            .map(|(p, s, e)| definite(p, x, &k, false, s, e))
            .collect::<PolyResult<Vec<_>>>()?,
    )?;
    let sine: Basic = crate::core::add::add(
        &found
            .iter()
            .map(|(p, s, e)| definite(p, x, &k, true, s, e))
            .collect::<PolyResult<Vec<_>>>()?,
    )?;
    let scale: Basic = d2(&int(2), &period)?;
    Ok(FourierCoefficients {
        a0: normal_form(&m2(&scale, &mean)?)?,
        an: normal_form(&m2(&scale, &cosine)?)?,
        bn: normal_form(&m2(&scale, &sine)?)?,
        n: index,
        x: x.clone(),
        omega,
    })
}

/// The partial sum of the Fourier series (`fourier_partial_sum`).
pub fn fourier_partial_sum(f: &Basic, x: &Basic, interval: (&Basic, &Basic), order: i64) -> R {
    fourier_coefficients(f, x, interval, None)?.partial_sum(order)
}
