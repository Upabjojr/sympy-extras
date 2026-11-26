//! The values of `Float` (`sympy_extras/core/numbers.py`): an exact dyadic
//! rational with a precision in bits, rounded half to even
//! ([`round_to_prec`]), and its decimal digits as mpmath prints them
//! ([`to_str`], a port of `mpmath.libmp.to_str` on the exact value).
//!
//! The decimal exponents beyond about ±1050 (`abs(exp + bc) > 3500` in
//! mpmath, where the power of ten is found with logarithms at a higher
//! precision) are not ported.

// PORTED FROM SYMPY: mpmath/libmp/libmpf.py:to_digits_exp, to_str, dps_to_prec, prec_to_dps,
// repr_dps (mpmath 1.3, BSD license; through sympy_extras/core/numbers.py)

use crate::error::{not_ported, PolyResult};
use crate::numbers::*;

/// `math.log(10, 2)` as Python computes it.
const LOG2_10: f64 = 3.3219280948873626;

/// The number of bits for `n` decimal digits (`dps_to_prec`).
pub fn dps_to_prec(n: i64) -> u32 {
    (((n + 1) as f64 * LOG2_10).round() as i64).max(1) as u32
}

/// The number of decimal digits of `n` bits (`prec_to_dps`).
pub fn prec_to_dps(n: u32) -> i64 {
    (((n as f64) / LOG2_10).round() as i64 - 1).max(1)
}

/// The digits which identify a float of `n` bits (`repr_dps`).
pub fn repr_dps(n: u32) -> i64 {
    let dps: i64 = prec_to_dps(n);
    if dps == 15 {
        17
    } else {
        dps + 3
    }
}

/// `value` rounded to `prec` significant bits, half to even
/// (`_round_to_prec`).
pub fn round_to_prec(value: &Rat, prec: u32) -> Rat {
    if value.is_zero() {
        return value.clone();
    }
    let negative: bool = rat_is_negative(value);
    let p: Int = abs(&numer(value));
    let q: Int = denom(value);
    // 2**e <= p/q < 2**(e + 1)
    let mut e: i64 = bits(&p) as i64 - bits(&q) as i64;
    let lhs: Int = p.clone() << (-e).max(0) as usize;
    let rhs: Int = q.clone() << e.max(0) as usize;
    if lhs < rhs {
        e -= 1;
    }
    let shift: i64 = prec as i64 - 1 - e;
    let (num, den): (Int, Int) = if shift >= 0 {
        (p << shift as usize, q)
    } else {
        (p, q << (-shift) as usize)
    };
    let (mut mantissa, remainder): (Int, Int) = div_rem_trunc(&num, &den);
    let twice: Int = remainder * int(2);
    if twice > den || (twice == den && bit(&mantissa, 0)) {
        mantissa += int(1);
    }
    let magnitude: Rat = if shift >= 0 {
        rat_from_parts(mantissa, int(1) << shift as usize)
    } else {
        rat_from_int(&(mantissa << (-shift) as usize))
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// The odd mantissa and the exponent of a nonzero dyadic magnitude:
/// `|value| = man * 2**exp` (mpmath's normalised `mpf`).
fn dyadic_parts(value: &Rat) -> (Int, i64) {
    let mut man: Int = abs(&numer(value));
    let den: Int = denom(value);
    debug_assert!(
        den.clone() & (den.clone() - int(1)) == Int::ZERO,
        "{} is not dyadic",
        value
    );
    let mut exp: i64 = -(bits(&den) as i64 - 1);
    let zeros: usize = man.trailing_zeros().unwrap_or(0);
    if zeros > 0 {
        man >>= zeros;
        exp += zeros as i64;
    }
    (man, exp)
}

/// The sign, the digits rounded toward zero and the decimal exponent of
/// a nonzero dyadic `value` with `dps` digits (`to_digits_exp`).
fn to_digits_exp(value: &Rat, dps: i64) -> PolyResult<(bool, String, i64)> {
    let negative: bool = rat_is_negative(value);
    let (man, exp): (Int, i64) = dyadic_parts(value);
    let bc: i64 = bits(&man) as i64;
    let bitprec: i64 = (dps as f64 * LOG2_10) as i64 + 10;
    if (exp + bc).abs() > 3500 {
        return Err(not_ported("the digits of a float with a decimal exponent beyond 1000"));
    }
    let fixprec: i64 = (bitprec - exp - bc).max(0);
    let fixdps: i64 = (fixprec as f64 / LOG2_10 + 0.5) as i64;
    let offset: i64 = exp + fixprec;
    let sf: Int = if offset >= 0 {
        man << offset as usize
    } else {
        man >> (-offset) as usize
    };
    let sd: Int = (sf * Int::from(10u8).pow(fixdps as usize)) >> fixprec as usize;
    let digits: String = sd.to_string();
    let exponent: i64 = digits.len() as i64 - fixdps - 1;
    Ok((negative, digits, exponent))
}

/// The decimal literal of a dyadic `value` with at most `dps` digits, as
/// `mpmath.libmp.to_str(value, dps, strip_zeros, min_fixed, max_fixed)`.
pub fn to_str(
    value: &Rat,
    dps: i64,
    strip_zeros: bool,
    min_fixed: Option<i64>,
    max_fixed: Option<i64>,
) -> PolyResult<String> {
    if value.is_zero() {
        return Ok(if dps != 0 { "0.0" } else { ".0" }.to_string());
    }
    let min_fixed: i64 = min_fixed.unwrap_or_else(|| (-(dps.div_euclid(3))).min(-5));
    let max_fixed: i64 = max_fixed.unwrap_or(dps);
    // to_digits_exp rounds to floor
    let (negative, mut digits, mut exponent) = to_digits_exp(value, dps + 3)?;
    let sign: &str = if negative { "-" } else { "" };
    if dps == 0 {
        if matches!(digits.as_bytes()[0], b'5'..=b'9') {
            exponent += 1;
        }
        digits = ".0".to_string();
    } else {
        let d: usize = dps as usize;
        let bytes: &[u8] = digits.as_bytes();
        if bytes.len() > d && matches!(bytes[d], b'5'..=b'9') {
            // rounding up kills some instances of "...99999"
            let mut kept: Vec<u8> = bytes[..d].to_vec();
            let mut i: i64 = d as i64 - 1;
            while i >= 0 && kept[i as usize] == b'9' {
                i -= 1;
            }
            if i >= 0 {
                let i: usize = i as usize;
                kept[i] += 1;
                for k in kept.iter_mut().skip(i + 1) {
                    *k = b'0';
                }
                digits = String::from_utf8(kept).expect("ASCII digits");
            } else {
                digits = format!("1{}", "0".repeat(d - 1));
                exponent += 1;
            }
        } else {
            digits.truncate(d);
        }
        // prettify the numbers close to unit magnitude
        let split: usize;
        if min_fixed < exponent && exponent < max_fixed {
            if exponent < 0 {
                digits = format!("{}{}", "0".repeat((-exponent) as usize), digits);
                split = 1;
            } else {
                split = exponent as usize + 1;
                if split > d {
                    digits.push_str(&"0".repeat(split - d));
                }
            }
            exponent = 0;
        } else {
            split = 1;
        }
        digits = format!("{}.{}", &digits[..split], &digits[split..]);
        if strip_zeros {
            // clean up the trailing zeros
            let trimmed: &str = digits.trim_end_matches('0');
            digits = if trimmed.ends_with('.') {
                format!("{}0", trimmed)
            } else {
                trimmed.to_string()
            };
        }
    }
    if exponent == 0 && dps != 0 {
        return Ok(format!("{}{}", sign, digits));
    }
    if exponent >= 0 {
        return Ok(format!("{}{}e+{}", sign, digits, exponent));
    }
    Ok(format!("{}{}e{}", sign, digits, exponent))
}

/// A function of an expression with a float: Python evaluates it with
/// mpmath (`_should_evalf`, the float terms of a sum split out), which is
/// not ported; an error rather than a different answer.
pub fn refuse_floats(x: &crate::core::basic::Basic, function: &str) -> PolyResult<()> {
    if x.preorder().iter().any(|e| e.is_float()) {
        return Err(not_ported(&format!(
            "{} of an expression with a float (mpmath)",
            function
        )));
    }
    Ok(())
}

/// The string form of a float: its decimal digits at its precision
/// (`str(Float)`); beyond the ported exponents, the exact value.
pub fn float_str(value: &Rat, prec: u32) -> String {
    to_str(value, prec_to_dps(prec), false, None, None).unwrap_or_else(|_| format!("Float({})", value))
}

/// The digits inside an expression, without the trailing zeros
/// (`short_str`).
pub fn float_short_str(value: &Rat, prec: u32) -> String {
    to_str(value, prec_to_dps(prec), true, None, None).unwrap_or_else(|_| format!("Float({})", value))
}

/// The digits of `srepr` (`repr_str`).
pub fn float_repr_str(value: &Rat, prec: u32) -> String {
    to_str(value, repr_dps(prec), true, None, None).unwrap_or_else(|_| format!("Float({})", value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(p: i64, q: i64) -> Rat {
        rat(p, q)
    }

    #[test]
    fn precisions() {
        assert_eq!(dps_to_prec(15), 53);
        assert_eq!(prec_to_dps(53), 15);
        assert_eq!(repr_dps(53), 17);
        assert_eq!(dps_to_prec(30), 103);
    }

    #[test]
    fn rounding() {
        // 1/3 at 53 bits
        let third: Rat = round_to_prec(&r(1, 3), 53);
        assert_eq!(third, rat_from_parts(int(6004799503160661), int(1) << 54usize));
        // half to even
        assert_eq!(round_to_prec(&r(5, 2), 2), r(2, 1));
        assert_eq!(round_to_prec(&r(7, 2), 2), r(4, 1));
        assert_eq!(round_to_prec(&r(-7, 2), 2), r(-4, 1));
        assert_eq!(round_to_prec(&r(3, 1), 53), r(3, 1));
    }

    #[test]
    fn digits() {
        let tenth: Rat = round_to_prec(&r(1, 10), 53);
        assert_eq!(to_str(&tenth, 15, false, None, None).unwrap(), "0.100000000000000");
        assert_eq!(to_str(&tenth, 15, true, None, None).unwrap(), "0.1");
        assert_eq!(to_str(&tenth, 17, true, None, None).unwrap(), "0.10000000000000001");
        assert_eq!(to_str(&r(5, 2), 15, false, None, None).unwrap(), "2.50000000000000");
        assert_eq!(to_str(&r(-5, 2), 15, true, None, None).unwrap(), "-2.5");
        let big: Rat = round_to_prec(&(r(25, 1) * rat_from_int(&Int::from(10u8).pow(19))), 53);
        assert_eq!(to_str(&big, 15, false, None, None).unwrap(), "2.50000000000000e+20");
        let small: Rat = round_to_prec(&r(1, 100000000), 53);
        assert_eq!(to_str(&small, 15, true, None, None).unwrap(), "1.0e-8");
        assert_eq!(to_str(&Rat::ZERO, 15, false, None, None).unwrap(), "0.0");
    }
}
