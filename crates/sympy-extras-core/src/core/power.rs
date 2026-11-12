//! Port of `sympy_extras/core/power.py`: [`integer_root_parts`] (the
//! decomposition of `p**e` for a positive integer `p` and `0 < e < 1`) and
//! the evaluation of powers, [`pow`] and [`number_pow`].
//!
//! The branches of the Python `pow` which build functions not in the Rust
//! core yet (`exp`, `Abs`, `re`, `floor`, `sign`, the relationals) are the
//! error `NotImplemented`, as are `Float`, `Order` and the accumulation
//! bounds.

use crate::error::{not_ported, zero_division, PolyResult};
use crate::ntheory::factorint;
use crate::numbers::*;

use super::add::add;
use super::assumptions::Predicate;
use super::basic::{Basic, Head};
use super::expr::{
    as_base_exp, as_coeff_mul, coeff, factors_of, is, is_false, is_rational_value, is_true, is_zero_number,
};
use super::mul::mul;
use super::numbers::number_mul;
use super::parameters::{divide_error, evaluate};
use crate::functions::elementary::complexes::{abs_, im, re, real_imag_parts};

/// The integers up to this size are factored completely to find their
/// perfect powers; larger ones only by trial division up to [`TRIAL`].
pub const FULL_FACTORING_BITS: u64 = 64;

/// The trial division bound for the large integers.
pub const TRIAL: u64 = 1 << 16;

/// `p**e` for a positive integer `p` and `0 < e = m/n < 1` as `(r, t, g)`
/// with `p**e == r * t**(g/n)`, `g/n < 1` and `t**g` free of `n`-th powers
/// in the factorisation used (complete below 64 bits, by trial division
/// above).
pub fn integer_root_parts(p: &Int, e: &Rat) -> (Int, Int, Int) {
    let (m, n) = (numer(e), denom(e));
    let factors = if bits(p) <= FULL_FACTORING_BITS {
        factorint(p, None)
    } else {
        factorint(p, Some(&Int::from(TRIAL)))
    };
    let mut r: Int = Int::ONE;
    let mut remainders: Vec<(Int, Int)> = Vec::new();
    for (prime, multiplicity) in factors {
        let (q, rem) = div_mod_floor(&(Int::from(multiplicity) * &m), &n);
        r *= prime.pow(usize::try_from(&q).expect("a small exponent"));
        if !rem.is_zero() {
            remainders.push((prime, rem));
        }
    }
    if remainders.is_empty() {
        return (r, Int::ONE, Int::ZERO);
    }
    let mut g: Int = Int::ZERO;
    for (_, rem) in &remainders {
        g = gcd(&g, rem);
    }
    let mut t: Int = Int::ONE;
    for (prime, rem) in &remainders {
        t *= prime.pow(usize::try_from(&div_floor(rem, &g)).expect("a small exponent"));
    }
    (r, t, g)
}

/// Whether `pow(base, exp)` keeps the node `Pow(base, exp)` for a positive
/// rational `base` and a rational `exp` with denominator `> 1` (the base is
/// then not a perfect power of the exponent's denominator, and the power is
/// irrational): an integer base above 1, `0 < exp < 1`, and
/// [`integer_root_parts`] giving `(1, base, numerator)`. This is what the
/// rational branch of `pow` decides; it is used by the assumptions until
/// `pow` is ported.
pub fn rational_power_is_kept(base: &Rat, exp: &Rat) -> bool {
    if !base.is_int() || *base <= Rat::ONE || *exp <= Rat::ZERO || *exp >= Rat::ONE {
        return false;
    }
    let (r, t, g) = integer_root_parts(base.numerator(), exp);
    r.is_one() && t == *base.numerator() && g == numer(exp)
}

fn pow_node(base: Basic, exp: Basic) -> Basic {
    Basic::new(Head::Pow, vec![base, exp])
}

fn floor_rat(v: &Rat) -> Int {
    div_floor(&numer(v), &denom(v))
}

/// `(-1)**e` with the exponent reduced to `(0, 1]`, as SymPy.
pub fn minus_one_pow(e: &Rat) -> Basic {
    let two: Rat = rat(2, 1);
    // in (-1, 1]
    let mut reduced: Rat = e - &two * rat_from_int(&floor_rat(&((e + Rat::ONE) / &two)));
    let mut sign: i64 = 1;
    if rat_is_negative(&reduced) {
        reduced += Rat::ONE;
        sign = -1;
    }
    if reduced.is_zero() {
        return Basic::int(sign);
    }
    if reduced == Rat::ONE {
        return Basic::int(-sign);
    }
    if reduced == rat(1, 2) {
        return if sign > 0 {
            Basic::imaginary_unit()
        } else {
            Basic::new(Head::Mul, vec![Basic::negative_one(), Basic::imaginary_unit()])
        };
    }
    let base: Basic = pow_node(Basic::negative_one(), Basic::from_rational(reduced));
    if sign > 0 {
        base
    } else {
        Basic::new(Head::Mul, vec![Basic::negative_one(), base])
    }
}

/// `b**e` for a non-integer rational exponent (`_rational_pow`).
fn rational_pow(b: &Rat, e: &Rat) -> PolyResult<Basic> {
    if *b == Rat::ONE {
        return Ok(Basic::one());
    }
    if rat_is_negative(e) {
        // b**(-m/n) = b**(k - m/n) * b**(-k) with k = ceil(m/n)
        let k: Int = -floor_rat(e);
        let shifted: Rat = rat_from_int(&k) + e;
        let power: Rat = rat_from_parts(
            numer(b).pow(usize::try_from(&k).expect("a small exponent")),
            denom(b).pow(usize::try_from(&k).expect("a small exponent")),
        );
        return mul(&[rational_pow(b, &shifted)?, Basic::from_rational(recip(&power))]);
    }
    if denom(b) > Int::ONE {
        let k: Int = -floor_rat(&-e);
        let k_rat: Rat = rat_from_int(&k);
        let p: Int = numer(b);
        let q: Int = denom(b);
        let first: Basic = if p != Int::ONE {
            rational_pow(&rat_from_int(&p), e)?
        } else {
            Basic::one()
        };
        let second: Basic = if k_rat != *e {
            rational_pow(&rat_from_int(&q), &(&k_rat - e))?
        } else {
            Basic::one()
        };
        let third: Basic = Basic::from_rational(rat_from_parts(
            Int::ONE,
            q.pow(usize::try_from(&k).expect("a small exponent")),
        ));
        return mul(&[first, second, third]);
    }
    let integer_part: Int = floor_rat(e);
    let fraction: Rat = e - rat_from_int(&integer_part);
    let p: Int = numer(b);
    let (r, t, g) = integer_root_parts(&abs(&p), &fraction);
    let coefficient: Int = p.pow(usize::try_from(&integer_part).expect("a small exponent")) * &r;
    let coefficient_node: Basic = Basic::integer(coefficient.clone());
    let n: Int = denom(&fraction);
    if !rat_is_negative(b) {
        if t == Int::ONE {
            return Ok(coefficient_node);
        }
        let radical: Basic = pow_node(Basic::integer(t), Basic::from_rational(rat_from_parts(g, n)));
        if coefficient == Int::ONE {
            return Ok(radical);
        }
        return Ok(Basic::new(Head::Mul, vec![coefficient_node, radical]));
    }
    // a negative base: (-1)**f is I for a half-integer exponent, otherwise
    // the sign stays with the n-th power free part when it is not a perfect
    // power
    let unit: Basic = minus_one_pow(&fraction);
    let mut parts: Vec<Basic> = vec![coefficient_node];
    if unit.is_pow() && g == numer(&fraction) && t > Int::ONE {
        parts.push(pow_node(Basic::integer(-t), Basic::from_rational(fraction)));
    } else if t == Int::ONE {
        parts.push(unit);
    } else {
        parts.push(unit);
        parts.push(pow_node(Basic::integer(t), Basic::from_rational(rat_from_parts(g, n))));
    }
    mul(&parts)
}

/// Powers involving `oo`, `-oo` or `zoo` (`_infinite_pow`).
fn infinite_pow(b: &Basic, e: &Basic) -> PolyResult<Basic> {
    if let Some(ev) = e.as_rational() {
        let sign: i32 = e.number_sign();
        return Ok(match b.head() {
            Head::Infinity => {
                if sign > 0 {
                    Basic::oo()
                } else {
                    Basic::zero()
                }
            }
            Head::NegativeInfinity => {
                if sign < 0 {
                    Basic::zero()
                } else if ev.is_int() {
                    if bit(&numer(ev), 0) {
                        Basic::negative_oo()
                    } else {
                        Basic::oo()
                    }
                } else {
                    mul(&[Basic::oo(), number_pow(&Basic::negative_one(), e)?])?
                }
            }
            Head::ComplexInfinity => {
                if sign > 0 {
                    Basic::zoo()
                } else {
                    Basic::zero()
                }
            }
            _ => Basic::nan(),
        });
    }
    if matches!(e.head(), Head::ComplexInfinity) {
        return Ok(Basic::nan());
    }
    if !b.is_finite_number() {
        return Ok(match b.head() {
            Head::Infinity => {
                if matches!(e.head(), Head::Infinity) {
                    Basic::oo()
                } else {
                    Basic::zero()
                }
            }
            Head::NegativeInfinity => {
                if matches!(e.head(), Head::Infinity) {
                    Basic::nan()
                } else {
                    Basic::zero()
                }
            }
            _ => Basic::nan(),
        });
    }
    let Some(vb) = b.as_rational() else {
        return Ok(Basic::nan());
    };
    let one: Rat = Rat::ONE;
    let minus_one: Rat = -Rat::ONE;
    Ok(match e.head() {
        Head::Infinity => {
            if *vb > one {
                Basic::oo()
            } else if *vb == one || *vb == minus_one {
                Basic::nan()
            } else if *vb > minus_one && *vb < one {
                Basic::zero()
            } else {
                Basic::zoo()
            }
        }
        Head::NegativeInfinity => {
            if *vb > one || *vb < minus_one {
                Basic::zero()
            } else if *vb == one || *vb == minus_one {
                Basic::nan()
            } else if vb.is_zero() {
                Basic::zoo()
            } else if !rat_is_negative(vb) {
                Basic::oo()
            } else {
                Basic::zoo()
            }
        }
        _ => Basic::nan(),
    })
}

/// `b**e` for two numbers (the result need not be a number).
pub fn number_pow(b: &Basic, e: &Basic) -> PolyResult<Basic> {
    if matches!(e.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if e.as_rational().is_some_and(|v| v.is_zero()) {
        return Ok(Basic::one());
    }
    if matches!(b.head(), Head::NaN) {
        return Ok(Basic::nan());
    }
    if is_rational_value(e, 1) {
        return Ok(b.clone());
    }
    if !b.is_finite_number() || !e.is_finite_number() {
        return infinite_pow(b, e);
    }
    let (vb, ve) = (
        b.as_rational().expect("a rational"),
        e.as_rational().expect("a rational"),
    );
    if *vb == Rat::ONE {
        return Ok(Basic::one());
    }
    if vb.is_zero() {
        if !rat_is_negative(ve) {
            return Ok(Basic::zero());
        }
        if divide_error() {
            return Err(zero_division("division by zero"));
        }
        return Ok(Basic::zoo());
    }
    if ve.is_int() {
        let n: Int = numer(ve);
        let k: usize = usize::try_from(abs(&n)).map_err(|_| not_ported("a rational to a huge integer power"))?;
        let power: Rat = rat_from_parts(numer(vb).pow(k), denom(vb).pow(k));
        return Ok(Basic::from_rational(if is_negative(&n) {
            recip(&power)
        } else {
            power
        }));
    }
    rational_pow(vb, ve)
}

fn imaginary_unit_power(n: &Int) -> PolyResult<Basic> {
    let i: Basic = Basic::imaginary_unit();
    Ok(match u64::try_from(mod_floor(n, &int(4))).expect("a residue") {
        0 => Basic::one(),
        1 => i,
        2 => Basic::negative_one(),
        _ => mul(&[Basic::negative_one(), i])?,
    })
}

/// `1/(a + b*I)` as `(a - b*I)/(a**2 + b**2)` for rational `a`, `b`.
fn gaussian_inverse(base: &Basic) -> Option<Basic> {
    if base.args().len() != 2 {
        return None;
    }
    let mut real: Option<Rat> = None;
    let mut imaginary: Option<Rat> = None;
    for term in base.args() {
        if let Some(v) = term.as_rational() {
            real = Some(v.clone());
        } else if matches!(term.head(), Head::ImaginaryUnit) {
            imaginary = Some(Rat::ONE);
        } else if term.is_mul()
            && term.args().len() == 2
            && term.args()[0].as_rational().is_some()
            && matches!(term.args()[1].head(), Head::ImaginaryUnit)
        {
            imaginary = term.args()[0].as_rational().cloned();
        }
    }
    let (a, b) = (real?, imaginary?);
    let norm: Rat = recip(&(&a * &a + &b * &b));
    let real_node: Basic = Basic::from_rational(a);
    let conjugate: Basic = if b != -Rat::ONE {
        Basic::new(
            Head::Add,
            vec![
                real_node,
                Basic::new(Head::Mul, vec![Basic::from_rational(-b), Basic::imaginary_unit()]),
            ],
        )
    } else {
        Basic::new(Head::Add, vec![real_node, Basic::imaginary_unit()])
    };
    Some(Basic::new(Head::Mul, vec![Basic::from_rational(norm), conjugate]))
}

/// Whether `e` is half an integer (`_half`).
fn half(e: &Basic) -> PolyResult<bool> {
    if let Some(v) = e.as_rational() {
        if denom(v) == int(2) {
            return Ok(true);
        }
    }
    let (n, d) = super::expr::as_numer_denom(e)?;
    Ok(is_true(&n, Predicate::integer) && is_rational_value(&d, 2))
}

/// `Lt(Abs(e), bound).doit() is true` (`_lt(abs_(e), bound)`).
fn abs_below(e: &Basic, bound: i64) -> PolyResult<bool> {
    super::relational::lt_true(&abs_(e)?, &Basic::int(bound))
}

/// `(b**e)**expt` as `s*b**(e*expt)` when the factor `s` (the branch
/// correction) is known (`Pow._eval_power`); `None` otherwise.
fn power_of_power(b: &Basic, e: &Basic, expt: &Basic) -> PolyResult<Option<Basic>> {
    let mut b: Basic = b.clone();
    let mut s: Option<Basic> = None;
    if is_true(expt, Predicate::integer) || is_true(&b, Predicate::polar) {
        s = Some(Basic::one());
    } else if let Some(real) = is(e, Predicate::extended_real) {
        if !real {
            // exp(2*I*pi*expt*floor(1/2 - im(e*log(b))/(2*pi))), kept when it
            // is a real sign
            use crate::functions::elementary::{
                exponential::{exp, log},
                integers::floor,
            };
            let two_pi: Basic = mul(&[Basic::int(2), Basic::pi()])?;
            let angle: Basic = im(&mul(&[e.clone(), log(&b)?])?)?;
            let inner: Basic = add(&[
                Basic::half(),
                mul(&[Basic::negative_one(), angle, pow(&two_pi, &Basic::negative_one())?])?,
            ])?;
            let factor: Basic = exp(&mul(&[
                Basic::int(2),
                Basic::imaginary_unit(),
                Basic::pi(),
                expt.clone(),
                floor(&inner)?,
            ])?)?;
            return match real_sign(&factor)? {
                Some(s) => Ok(Some(mul(&[s, pow(&b, &mul(&[e.clone(), expt.clone()])?)?])?)),
                None => Ok(None),
            };
        }
        if is_rational_value(e, -1) {
            if half(expt)? {
                if is_true(&b, Predicate::negative) {
                    let a: Basic = pow(&Basic::negative_one(), expt)?;
                    let c: Basic = pow(
                        &mul(&[Basic::negative_one(), b.clone()])?,
                        &mul(&[e.clone(), expt.clone()])?,
                    )?;
                    return Ok(Some(mul(&[a, c])?));
                }
                if is_false(&b, Predicate::negative) {
                    return Ok(Some(pow(&b, &mul(&[Basic::negative_one(), expt.clone()])?)?));
                }
            }
        } else if is_true(e, Predicate::even) {
            if is_true(&b, Predicate::extended_real) {
                b = abs_(&b)?;
            }
            if is_true(&b, Predicate::imaginary) {
                b = mul(&[abs_(&im(&b)?)?, Basic::imaginary_unit()])?;
            }
        }
        if abs_below(e, 1)?
            || is_rational_value(e, 1)
            || is_true(&b, Predicate::extended_nonnegative)
            || (is_true(&re(&b)?, Predicate::extended_nonnegative) && abs_below(e, 2)?)
        {
            s = Some(Basic::one());
        } else if half(expt)? {
            // exp(2*pi*I*expt*floor(1/2 - e*arg(b)/(2*pi))), kept when it
            // is a real sign
            use crate::functions::elementary::{argument::arg, exponential::exp, integers::floor};
            let two_pi: Basic = mul(&[Basic::int(2), Basic::pi()])?;
            let inner: Basic = add(&[
                Basic::half(),
                mul(&[
                    Basic::negative_one(),
                    e.clone(),
                    arg(&b)?,
                    pow(&two_pi, &Basic::negative_one())?,
                ])?,
            ])?;
            let factor: Basic = exp(&mul(&[
                Basic::int(2),
                Basic::pi(),
                Basic::imaginary_unit(),
                expt.clone(),
                floor(&inner)?,
            ])?)?;
            s = real_sign(&factor)?;
        }
    }
    match s {
        Some(s) => Ok(Some(mul(&[s, pow(&b, &mul(&[e.clone(), expt.clone()])?)?])?)),
        None => Ok(None),
    }
}

/// The factor when it is a real sign (`_real_sign`): `sign(s)` when
/// `sign(s) - s` is zero (evaluated to 2 digits in Python: exactly zero).
fn real_sign(s: &Basic) -> PolyResult<Option<Basic>> {
    if !is_true(s, Predicate::extended_real) {
        return Ok(None);
    }
    let sg: Basic = crate::functions::elementary::complexes::sign(s)?;
    let difference: Basic = add(&[sg.clone(), mul(&[Basic::negative_one(), s.clone()])?])?;
    let zero: bool = difference == Basic::zero()
        || super::evalf::numeric_value(&difference, 7).is_some_and(|v| {
            let (re, im) = super::evalf::parts(&v);
            re == super::evalf::Real::ZERO && im == super::evalf::Real::ZERO
        });
    Ok(if zero { Some(sg) } else { None })
}

/// `(a*b)**e` distributed when the rules allow it (`_power_of_product`).
fn power_of_product(base: &Basic, exp: &Basic) -> PolyResult<Option<Basic>> {
    let commutative: Vec<Basic> = base
        .args()
        .iter()
        .filter(|f| !is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    let noncommutative: Vec<Basic> = base
        .args()
        .iter()
        .filter(|f| is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    if let Some(x) = exp.as_rational() {
        if x.is_int() {
            let mut parts: Vec<Basic> = Vec::new();
            for f in &commutative {
                parts.push(pow(f, exp)?);
            }
            if !noncommutative.is_empty() {
                parts.push(pow_node(mul(&noncommutative)?, exp.clone()));
            }
            return Ok(Some(mul(&parts)?));
        }
    }
    let Some(x) = exp.as_rational() else {
        return Ok(None);
    };
    if !noncommutative.is_empty() {
        return Ok(None);
    }
    if denom(x) == int(2) && is_true(base, Predicate::imaginary) {
        let imaginary: Basic = real_imag_parts(base)?.1;
        if let Some(v) = imaginary.as_rational() {
            let half_value: Rat = (if rat_is_negative(v) { -v.clone() } else { v.clone() }) / rat(2, 1);
            let (n, d) = (numer(&half_value), denom(&half_value));
            if integer_root_parts(&n, &rat(1, 2)).1 == Int::ONE && integer_root_parts(&d, &rat(1, 2)).1 == Int::ONE {
                return Err(not_ported("unevaluated_mul of sqrt(2*I)"));
            }
        }
    }
    let (coefficient, rest) = as_coeff_mul(base);
    // SymPy's _eval_expand_power_base: the positive and the polar factors
    // come out of the power
    let positive_or_polar = |f: &Basic| is_true(f, Predicate::positive) || is_true(f, Predicate::polar);
    let positive: Vec<Basic> = factors_of(&rest).into_iter().filter(|f| positive_or_polar(f)).collect();
    let others: Vec<Basic> = factors_of(&rest)
        .into_iter()
        .filter(|f| !positive_or_polar(f))
        .collect();
    if coefficient.is_finite_number() && coefficient.number_sign() != 0 && !is_rational_value(&coefficient, 1) {
        if coefficient.number_sign() > 0 || others.is_empty() {
            let mut parts: Vec<Basic> = vec![pow(&coefficient, exp)?];
            for f in &positive {
                parts.push(pow(f, exp)?);
            }
            parts.push(if others.is_empty() {
                Basic::one()
            } else {
                pow(&mul(&others)?, exp)?
            });
            return Ok(Some(mul(&parts)?));
        }
        let negated: Basic = number_mul(&coefficient, &Basic::negative_one());
        if is_rational_value(&negated, 1) && positive.is_empty() {
            return Ok(None); // (-x)**e stays
        }
        let mut parts: Vec<Basic> = vec![pow(&negated, exp)?];
        for f in &positive {
            parts.push(pow(f, exp)?);
        }
        let mut negative_others: Vec<Basic> = vec![Basic::negative_one()];
        negative_others.extend(others);
        parts.push(pow(&mul(&negative_others)?, exp)?);
        return Ok(Some(mul(&parts)?));
    }
    if !positive.is_empty() && !others.is_empty() {
        let mut parts: Vec<Basic> = Vec::new();
        for f in &positive {
            parts.push(pow(f, exp)?);
        }
        parts.push(pow(&mul(&others)?, exp)?);
        return Ok(Some(mul(&parts)?));
    }
    if !positive.is_empty() {
        let mut parts: Vec<Basic> = Vec::new();
        for f in &positive {
            parts.push(pow(f, exp)?);
        }
        return Ok(Some(mul(&parts)?));
    }
    Ok(None)
}

/// The evaluated power `b**e`.
pub fn pow(base: &Basic, exp: &Basic) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(pow_node(base.clone(), exp.clone()));
    }
    if base.is_number_node() && exp.is_number_node() {
        return number_pow(base, exp);
    }
    if exp.is_number_node() {
        if matches!(exp.head(), Head::NaN) {
            return Ok(Basic::nan());
        }
        if let Some(x) = exp.as_rational() {
            if x.is_zero() {
                return Ok(Basic::one());
            }
            if *x == Rat::ONE {
                return Ok(base.clone());
            }
        }
    }
    if base.is_number_node() {
        if matches!(base.head(), Head::NaN) {
            return Ok(Basic::nan());
        }
        if is_rational_value(base, 1) {
            return Ok(Basic::one());
        }
        if matches!(
            base.head(),
            Head::Rational {
                integer_class: true,
                ..
            }
        ) && is_rational_value(base, -1)
        {
            // the parity of a symbolic exponent (NegativeOne._eval_power)
            if is_true(exp, Predicate::odd) {
                return Ok(Basic::negative_one());
            }
            if is_true(exp, Predicate::even) {
                return Ok(Basic::one());
            }
        }
        if base.as_rational().is_some_and(|v| v.is_zero()) {
            // Zero._eval_power
            if is_true(exp, Predicate::extended_positive) {
                return Ok(Basic::zero());
            }
            if is_true(exp, Predicate::extended_negative) {
                return Ok(Basic::zoo());
            }
            if is_false(exp, Predicate::extended_real) {
                return Ok(Basic::nan());
            }
            if is_true(exp, Predicate::zero) {
                return Ok(Basic::one());
            }
            // the leading number of the exponent is dropped
            let (c, rest) = as_coeff_mul(exp);
            if is_true(&c, Predicate::negative) {
                return pow(&Basic::zoo(), &rest);
            }
            if !is_rational_value(&c, 1) {
                return pow(base, &rest);
            }
            return Ok(pow_node(base.clone(), exp.clone()));
        }
    }
    if base.is_add()
        && base.args().len() == 2
        && base.args().iter().any(|t| is_true(t, Predicate::infinite))
        && is_false(exp, Predicate::zero)
        && *exp != Basic::one()
    {
        // Add._eval_power: a + I*b with real a, b and an infinite term
        let i: Basic = Basic::imaginary_unit();
        let (mut a, mut b) = (base.args()[0].clone(), base.args()[1].clone());
        if !is_zero_number(&coeff(&a, &i)?) {
            std::mem::swap(&mut a, &mut b);
        }
        let imaginary: Basic = coeff(&b, &i)?;
        if !is_zero_number(&imaginary)
            && is_true(&imaginary, Predicate::extended_real)
            && is_true(&a, Predicate::extended_real)
        {
            if is_true(exp, Predicate::extended_negative) {
                return Ok(Basic::zero());
            }
            if is_true(exp, Predicate::extended_positive) {
                return Ok(Basic::zoo());
            }
        }
    }
    if base.is_mul() {
        if let Some(x) = exp.as_rational() {
            if denom(x) != Int::ONE {
                let (coefficient, direction) = as_coeff_mul(base);
                if matches!(coefficient.head(), Head::Infinity | Head::NegativeInfinity)
                    && is_true(&direction, Predicate::finite)
                    && is_false(&direction, Predicate::zero)
                {
                    if rat_is_negative(x) {
                        return Ok(Basic::zero());
                    }
                    let sign: Basic = if matches!(coefficient.head(), Head::Infinity) {
                        Basic::one()
                    } else {
                        Basic::negative_one()
                    };
                    return mul(&[Basic::oo(), pow(&mul(&[sign, direction])?, exp)?]);
                }
            }
        }
    }
    if super::accumbounds::is_bounds(exp) && matches!(base.head(), Head::Exp1) {
        // Pow.__new__: E**AccumBounds
        use crate::functions::elementary::exponential::exp as exponential;
        return super::accumbounds::accum_bounds(&exponential(&exp.args()[0])?, &exponential(&exp.args()[1])?);
    }
    if super::accumbounds::is_bounds(base) {
        // AccumulationBounds._eval_power
        return super::accumbounds::ab_pow(base, exp);
    }
    if base.is_function(super::basic::Function::Abs) {
        if let Some(own) = crate::functions::elementary::complexes::abs_power(base, exp)? {
            return Ok(own);
        }
    }
    if matches!(base.head(), Head::Infinity) && !exp.is_number_node() {
        // Infinity._eval_power
        if is_true(exp, Predicate::extended_positive) {
            return Ok(Basic::oo());
        }
        if is_true(exp, Predicate::extended_negative) {
            return Ok(Basic::zero());
        }
        if is_false(exp, Predicate::extended_real) && super::assumptions::is_number(exp) {
            let real: Basic = re(exp)?;
            if is_true(&real, Predicate::positive) {
                return Ok(Basic::zoo());
            }
            if is_true(&real, Predicate::negative) {
                return Ok(Basic::zero());
            }
            if is_true(&real, Predicate::zero) {
                return Ok(Basic::nan());
            }
        }
    }
    if matches!(base.head(), Head::Exp1) {
        // E**a is exp(a)
        return crate::functions::elementary::exponential::exp(exp);
    }
    if matches!(base.head(), Head::ImaginaryUnit) {
        if let Some(x) = exp.as_rational() {
            if x.is_int() {
                return imaginary_unit_power(&numer(x));
            }
            let two: Rat = rat(2, 1);
            let reduced: Rat = x - &two * rat_from_int(&floor_rat(&(x / &two)));
            let k: Int = floor_rat(&(x / &two));
            let power: Basic = pow_node(Basic::imaginary_unit(), Basic::from_rational(reduced));
            let even: bool = mod_floor(&k, &int(2)).is_zero();
            return Ok(if even {
                power
            } else {
                Basic::new(Head::Mul, vec![Basic::negative_one(), power])
            });
        }
    }
    if base.is_add() && is_rational_value(exp, -1) && exp.as_rational().is_some_and(|x| x.is_int()) {
        if let Some(inverse) = gaussian_inverse(base) {
            return Ok(inverse);
        }
    }
    if base.is_pow() || base.is_function(super::basic::Function::Exp) {
        let (inner_base, inner_exp) = as_base_exp(base)?;
        if let Some(denested) = power_of_power(&inner_base, &inner_exp, exp)? {
            return Ok(denested);
        }
    }
    if base.is_mul() {
        if let Some(distributed) = power_of_product(base, exp)? {
            return Ok(distributed);
        }
    }
    Ok(pow_node(base.clone(), exp.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctests() {
        assert_eq!(integer_root_parts(&int(12), &rat(1, 2)), (int(2), int(3), int(1)));
        assert_eq!(integer_root_parts(&int(48), &rat(1, 3)), (int(2), int(6), int(1)));
        assert_eq!(integer_root_parts(&int(32), &rat(1, 6)), (int(1), int(2), int(5)));
        assert_eq!(integer_root_parts(&int(72), &rat(1, 3)), (int(2), int(3), int(2)));
        assert!(rational_power_is_kept(&rat(2, 1), &rat(1, 2)));
        assert!(rational_power_is_kept(&rat(6, 1), &rat(2, 3)));
        assert!(!rational_power_is_kept(&rat(4, 1), &rat(1, 2)));
        assert!(!rational_power_is_kept(&rat(12, 1), &rat(1, 2)));
        assert!(!rational_power_is_kept(&rat(1, 2), &rat(1, 2)));
        assert!(!rational_power_is_kept(&rat(2, 1), &rat(3, 2)));
    }
}
