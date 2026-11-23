//! Port of `sympy_extras/integrals/periodic.py`: integrals of analytic
//! periodic integrands over whole periods by the mean value theorem,
//! `2*pi*k` times the constant coefficient of the Laurent series of
//! `F(z)`, `z = exp(I*x)`. The constant coefficient of a product
//! `G(z)*H(1/z)` of two entire functions is the sum `sum_j g_j h_j`,
//! summed in closed form; a power `z**m` shifts the index. The Taylor
//! coefficients of `exp`, `sin`, `cos`, `sinh` and `cosh` are written
//! down, those of the other functions come from the formal power series
//! (`brackets::taylor_coefficient`).

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::error::PolyResult;
use crate::integrals::brackets::SeriesCoefficient;
use crate::integrals::conditions::ConditionalValue;

type R = PolyResult<Basic>;
type R2 = PolyResult<Option<Basic>>;

/// The trigonometric and hyperbolic functions, which an outer function
/// of a Laurent polynomial is rewritten from
/// (`_TRIGONOMETRIC_AND_HYPERBOLIC`).
const TRIGONOMETRIC_AND_HYPERBOLIC: [Function; 12] = [
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];

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
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn rational(p: i64, q: i64) -> R {
    Basic::rational(p.into(), q.into())
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn is_application(e: &Basic) -> bool {
    matches!(e.head(), Head::Function(_) | Head::Undefined(_))
}
fn args_of(e: &Basic, sum: bool) -> Vec<Basic> {
    if (sum && e.is_add()) || (!sum && e.is_mul()) {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn dummy(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    crate::core::symbol::Dummy::new(name, assumptions).0
}
fn replace(e: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    crate::core::function::xreplace_evaluated(e, rule)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}

/// `k` when `argument == k*x` with an integer `k` (`_multiple`).
fn multiple(argument: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let (coefficient, rest) = crate::core::expr::as_independent(argument, x, false)?;
    if rest != *x || integer_of(&coefficient).is_none() {
        return Ok(None);
    }
    Ok(Some(coefficient))
}

/// `f` with the trigonometric functions of integer multiples of `x` (and
/// `exp(I*k*x)`) written in `z = exp(I*x)`; `None` when `x` occurs
/// otherwise (`to_laurent`).
pub fn to_laurent(f: &Basic, x: &Basic, z: &Basic) -> R2 {
    let unit: Basic = Basic::imaginary_unit();
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    for node in f.atoms(Some(&is_application)) {
        if !node.has(x) || node.args().len() != 1 {
            continue;
        }
        let argument: &Basic = &node.args()[0];
        if node.is_function(Function::Exp) {
            if let Some(k) = multiple(&div(argument, &unit)?, x)? {
                replacement.insert(node.clone(), pow(z, &k)?);
            }
            // an outer function of trigonometric functions is left as it
            // is: its argument is replaced
            continue;
        }
        let Some(k) = multiple(argument, x)? else {
            continue;
        };
        let up: Basic = pow(z, &k)?;
        let down: Basic = pow(z, &crate::core::mul::neg(&k)?)?;
        let cosine: Basic = div(&add(&[up.clone(), down.clone()])?, &int(2))?;
        let sine: Basic = div(&sub(&up, &down)?, &mul(&[int(2), unit.clone()])?)?;
        let value: Basic = match node.head() {
            Head::Function(Function::Cos) => cosine,
            Head::Function(Function::Sin) => sine,
            Head::Function(Function::Tan) => div(&sine, &cosine)?,
            Head::Function(Function::Cot) => div(&cosine, &sine)?,
            Head::Function(Function::Sec) => div(&int(1), &cosine)?,
            Head::Function(Function::Csc) => div(&int(1), &sine)?,
            _ => return Ok(None),
        };
        replacement.insert(node.clone(), value);
    }
    let mut result: Basic = replace(f, &replacement)?;
    if result.has(x) {
        return Ok(None);
    }
    // the outer trigonometric and hyperbolic functions of a Laurent
    // polynomial as exponentials
    if result
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(g) if TRIGONOMETRIC_AND_HYPERBOLIC.contains(g)))
    {
        result =
            crate::core::rewriting::rewrite(&result, &[], "exp", &crate::core::rewriting::RewriteHints::default())?;
    }
    Ok(Some(result))
}

/// A factor `g(c*z**p)` of a term: the series of `g` with its exponents
/// in `z` scaled by `p`, `sign` that of `p`, `scale` `c**(1/|p|)`
/// (`_Factor`).
struct Factor {
    series: SeriesCoefficient,
    sign: i64,
    scale: Basic,
}

impl Factor {
    /// The coefficient of `z**exponent(j)`, the scale of the argument
    /// included.
    fn coefficient(&self, j: &Basic) -> R {
        let exponent: Basic = add(&[self.series.offset.clone(), mul(&[int(self.series.period), j.clone()])?])?;
        mul(&[
            crate::core::subs::subs(&self.series.coefficient, &self.series.index, j)?,
            pow(&self.scale, &exponent)?,
        ])
    }
}

/// The Taylor series of `exp(y)`, `sin(y)`, `cos(y)`, `sinh(y)` and
/// `cosh(y)` written down (`_known_series`).
fn known_series(head: &Basic, y: &Basic) -> PolyResult<Option<SeriesCoefficient>> {
    let Head::Function(f) = head.head() else {
        return Ok(None);
    };
    if !matches!(
        f,
        Function::Exp | Function::Sin | Function::Cos | Function::Sinh | Function::Cosh
    ) || head.args()[0] != *y
    {
        return Ok(None);
    }
    use crate::functions::combinatorial::factorials::{factorial, rf};
    let j: Basic = dummy("j", &[(Predicate::integer, true), (Predicate::nonnegative, true)]);
    if *f == Function::Exp {
        return Ok(Some(SeriesCoefficient {
            coefficient: div(&int(1), &factorial(&j)?)?,
            index: j,
            period: 1,
            offset: Basic::zero(),
        }));
    }
    let odd: bool = matches!(f, Function::Sin | Function::Sinh);
    let sign: Basic = if matches!(f, Function::Sin | Function::Cos) {
        rational(-1, 4)?
    } else {
        rational(1, 4)?
    };
    let start: Basic = if odd { rational(3, 2)? } else { rational(1, 2)? };
    let coefficient: Basic = div(&pow(&sign, &j)?, &mul(&[rf(&start, &j)?, factorial(&j)?])?)?;
    Ok(Some(SeriesCoefficient {
        coefficient,
        index: j,
        period: 2,
        offset: if odd { Basic::one() } else { Basic::zero() },
    }))
}

/// A function of `c*z**p` as a series in `z` (`p > 0`) or in `1/z` (`p <
/// 0`) (`_factor`).
fn factor_of(e: &Basic, z: &Basic) -> PolyResult<Option<Factor>> {
    if !is_application(e) || e.args().len() != 1 {
        return Ok(None);
    }
    let Some((scale, power)) = crate::integrals::mellin::monomial(&e.args()[0], z)? else {
        return Ok(None);
    };
    let Some(power) = integer_of(&power).filter(|p| *p != 0) else {
        return Ok(None);
    };
    let y: Basic = dummy("y", &[]);
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(e.args()[0].clone(), y.clone());
    let head: Basic = replace(e, &rule)?;
    let series: SeriesCoefficient = match known_series(&head, &y)? {
        Some(s) => s,
        None => match crate::integrals::brackets::taylor_coefficient(&head, &y)? {
            Some(s) => s,
            None => return Ok(None),
        },
    };
    let p: i64 = power.abs();
    let scaled: SeriesCoefficient = SeriesCoefficient {
        coefficient: series.coefficient.clone(),
        index: series.index.clone(),
        period: series.period * p,
        offset: mul(&[series.offset.clone(), int(p)])?,
    };
    Ok(Some(Factor {
        series: scaled,
        sign: if power > 0 { 1 } else { -1 },
        scale: pow(&scale, &rational(1, p)?)?,
    }))
}

/// The coefficient of `z**m` in one term `constant*z**n*g(c*z**p)*h(d*z**q)`
/// (`_term_coefficient`).
fn term_coefficient(term: &Basic, z: &Basic, m: &Basic) -> R2 {
    let mut constant: Basic = Basic::one();
    let mut shift: Basic = Basic::zero();
    let mut factors: Vec<Factor> = Vec::new();
    for e in args_of(&merged_exponentials(term, z)?, false) {
        if !e.has(z) {
            constant = mul(&[constant, e])?;
        } else if e == *z {
            shift = add(&[shift, Basic::one()])?;
        } else if e.is_pow() && e.args()[0] == *z && integer_of(&e.args()[1]).is_some() {
            shift = add(&[shift, e.args()[1].clone()])?;
        } else if e.is_pow() && integer_of(&e.args()[1]).is_some_and(|k| k > 0) && is_application(&e.args()[0]) {
            for _ in 0..integer_of(&e.args()[1]).unwrap_or(0) {
                let Some(found) = factor_of(&e.args()[0], z)? else {
                    return Ok(None);
                };
                factors.push(found);
            }
        } else {
            let Some(found) = factor_of(&e, z)? else {
                return Ok(None);
            };
            factors.push(found);
        }
    }
    let target: Basic = sub(m, &shift)?;
    match factors.len() {
        0 => Ok(Some(if target == Basic::zero() {
            constant
        } else {
            Basic::zero()
        })),
        1 => Ok(Some(mul(&[constant, single(&factors[0], &target)?])?)),
        2 => match double(&factors[0], &factors[1], &target)? {
            None => Ok(None),
            Some(value) => Ok(Some(mul(&[constant, value])?)),
        },
        _ => Ok(None),
    }
}

/// The exponential factors of a term combined into one exponential per
/// power of `z` (`_merged_exponentials`).
fn merged_exponentials(term: &Basic, z: &Basic) -> R {
    let mut others: Vec<Basic> = Vec::new();
    let mut by_power: Vec<(Basic, Basic)> = Vec::new();
    for e in args_of(term, false) {
        if e.is_function(Function::Exp) && e.has(z) {
            let argument: Basic = crate::core::function::expand_method(&e.args()[0])?;
            for piece in args_of(&argument, true) {
                match crate::integrals::mellin::monomial(&piece, z)? {
                    None => others.push(exp(&piece)?),
                    Some((scale, power)) => match by_power.iter_mut().find(|(p, _)| *p == power) {
                        Some((_, s)) => *s = add(&[s.clone(), scale])?,
                        None => by_power.push((power, scale)),
                    },
                }
            }
        } else {
            others.push(e);
        }
    }
    let mut merged: Basic = Basic::one();
    for (power, scale) in by_power {
        merged = mul(&[merged, exp(&mul(&[scale, pow(z, &power)?])?)?])?;
    }
    mul(&[mul(&others)?, merged])
}

/// The coefficient of `z**target` in one series (`_single`).
fn single(factor: &Factor, target: &Basic) -> R {
    let j: Basic = div(
        &sub(&mul(&[target.clone(), int(factor.sign)])?, &factor.series.offset)?,
        &int(factor.series.period),
    )?;
    match integer_of(&j) {
        Some(k) if k >= 0 => factor.coefficient(&j),
        _ => Ok(Basic::zero()),
    }
}

/// The coefficient of `z**target` in the product of two series, summed
/// in closed form (`_double`).
fn double(first: &Factor, second: &Factor, target: &Basic) -> R2 {
    let t: Basic = dummy("t", &[(Predicate::integer, true), (Predicate::nonnegative, true)]);
    let (m1, r1, s1) = (first.series.period, &first.series.offset, first.sign);
    let (m2, r2, s2) = (second.series.period, &second.series.offset, second.sign);
    let mut total: Basic = Basic::zero();
    let slope: Basic = rational(-s2 * m2 * m1, s1 * m1)?;
    for residue in 0..m1 {
        let remainder: Basic = sub(
            &sub(target, &mul(&[int(s2), add(&[r2.clone(), int(m2 * residue)])?])?)?,
            &mul(&[int(s1), r1.clone()])?,
        )?;
        match integer_of(&remainder) {
            Some(k) if k.rem_euclid(m1) == 0 => {}
            _ => continue,
        }
        let j2: Basic = add(&[mul(&[int(m1), t.clone()])?, int(residue)])?;
        let j1: Basic = div(
            &sub(
                &sub(
                    target,
                    &mul(&[int(s2), add(&[r2.clone(), mul(&[int(m2), j2.clone()])?])?])?,
                )?,
                &mul(&[int(s1), r1.clone()])?,
            )?,
            &int(s1 * m1),
        )?;
        let mut lower: Basic = Basic::zero();
        let mut upper: Basic = Basic::atom(Head::Infinity);
        let bound: Basic = crate::core::subs::subs(&j1, &t, &Basic::zero())?;
        let Some(b) = bound.as_rational() else {
            return Ok(None);
        };
        let negative: bool = *b < crate::numbers::Rat::from(0);
        if slope.as_rational().is_some_and(|s| *s > crate::numbers::Rat::from(0)) {
            if negative {
                lower =
                    crate::functions::elementary::integers::ceiling(&div(&crate::core::mul::neg(&bound)?, &slope)?)?;
            }
        } else {
            if negative {
                continue;
            }
            upper = crate::functions::elementary::integers::floor(&div(&bound, &crate::core::mul::neg(&slope)?)?)?;
        }
        let summand: Basic = mul(&[first.coefficient(&j1)?, second.coefficient(&j2)?])?;
        let limits = (t.clone(), lower, upper);
        let value: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            crate::concrete::summations::summation(&summand, &limits)
        })?;
        let Some(value) = value else {
            return Ok(None);
        };
        if value.preorder().iter().any(|n| matches!(n.head(), Head::Sum)) {
            return Ok(None);
        }
        total = add(&[total, value])?;
    }
    Ok(Some(simplify(&total)?))
}

/// The coefficient of `z**m` in the Laurent expansion of `f` about 0, for
/// a sum of terms with at most two functions of `z`
/// (`laurent_coefficient`).
pub fn laurent_coefficient(f: &Basic, z: &Basic, m: &Basic) -> R2 {
    let expression: Basic = crate::core::function::expand_method(f)?;
    let (_, denominator) = crate::core::expr::as_numer_denom(&expression)?;
    for e in args_of(&denominator, false) {
        if !e.has(z) || e == *z || (e.is_pow() && e.args()[0] == *z) || e.is_function(Function::Exp) {
            continue;
        }
        // a denominator vanishing away from the origin: the residues, not
        // the series
        return Ok(None);
    }
    let mut total: Basic = Basic::zero();
    for term in args_of(&expression, true) {
        let Some(value) = term_coefficient(&term, z, m)? else {
            return Ok(None);
        };
        total = add(&[total, value])?;
    }
    Ok(Some(simplify(&total)?))
}

/// `Integral(f, (x, a, b))` over `k` full periods of an integrand
/// analytic in `z = exp(I*x)` away from the origin: `2*pi*k` times the
/// constant Laurent coefficient (`mean_value_integral`).
pub fn mean_value_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<ConditionalValue>> {
    let periods: Basic = div(&sub(b, a)?, &mul(&[int(2), Basic::pi()])?)?;
    if !integer_of(&periods).is_some_and(|k| k > 0) {
        return Ok(None);
    }
    let z: Basic = dummy("z", &[(Predicate::positive, true)]);
    let Some(laurent) = to_laurent(f, x, &z)? else {
        return Ok(None);
    };
    let Some(coefficient) = laurent_coefficient(&laurent, &z, &Basic::zero())? else {
        return Ok(None);
    };
    Ok(Some(ConditionalValue::new(
        mul(&[int(2), Basic::pi(), periods, coefficient])?,
        crate::core::logic::true_(),
    )))
}
