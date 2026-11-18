//! Integer functions, ported from `sympy_extras.core.intfunc`: gcd, lcm,
//! the extended Euclidean algorithm, the modular inverse, integer roots
//! and logarithms, digit counts and trailing zero bits.

use dashu_base::{Abs, Gcd};

use crate::error::{value_error, PolyResult};
use crate::numbers::{bits, div_floor, is_negative, isqrt, mod_floor, Int};

/// The nonnegative greatest common divisor of integers (`igcd`).
pub fn igcd(args: &[Int]) -> Int {
    let mut result: Int = Int::ZERO;
    for a in args {
        // gcd(0, 0) is 0 in Python (dashu panics on it)
        result = if result.is_zero() {
            a.clone().abs()
        } else {
            Int::from(Gcd::gcd(&result, a))
        };
    }
    result
}

/// The nonnegative least common multiple of integers, `0` when one is
/// zero (`ilcm`).
pub fn ilcm(args: &[Int]) -> Int {
    let mut result: Int = Int::ONE;
    for a in args {
        if result.is_zero() || a.is_zero() {
            result = Int::ZERO;
            continue;
        }
        let g: Int = Int::from(Gcd::gcd(&result, a));
        result = (&result * a / g).abs();
    }
    result
}

/// `(x, y, g)` with `a*x + b*y == g == gcd(a, b)` (`igcdex`).
pub fn igcdex(a: &Int, b: &Int) -> (Int, Int, Int) {
    if a.is_zero() && b.is_zero() {
        return (Int::ZERO, Int::ONE, Int::ZERO);
    }
    let sign_a: Int = if is_negative(a) { Int::NEG_ONE } else { Int::ONE };
    let sign_b: Int = if is_negative(b) { Int::NEG_ONE } else { Int::ONE };
    let (mut a, mut b): (Int, Int) = (a.clone().abs(), b.clone().abs());
    let (mut x0, mut x1, mut y0, mut y1): (Int, Int, Int, Int) = (Int::ONE, Int::ZERO, Int::ZERO, Int::ONE);
    while !b.is_zero() {
        let q: Int = div_floor(&a, &b);
        let r: Int = &a - &q * &b;
        a = std::mem::replace(&mut b, r);
        let x2: Int = &x0 - &q * &x1;
        x0 = std::mem::replace(&mut x1, x2);
        let y2: Int = &y0 - &q * &y1;
        y0 = std::mem::replace(&mut y1, y2);
    }
    (sign_a * x0, sign_b * y0, a)
}

/// The inverse of `a` modulo `m`, in `[1, m)` for a positive `m`; a
/// `ValueError` when `a` and `m` are not coprime (`mod_inverse`).
pub fn mod_inverse(a: &Int, m: &Int) -> PolyResult<Int> {
    let (x, _, g) = igcdex(a, m);
    if !g.is_one() {
        return Err(value_error(format!("inverse of {} (mod {}) does not exist", a, m)));
    }
    if *m > Int::ONE {
        return Ok(mod_floor(&x, m));
    }
    if *m < Int::NEG_ONE {
        let negated: Int = -m.clone();
        return Ok(-mod_floor(&x, &negated));
    }
    Ok(x)
}

/// The integer part of the `n`-th root of `y >= 0` and whether the root is
/// exact (`integer_nthroot`).
pub fn integer_nthroot(y: &Int, n: &Int) -> PolyResult<(Int, bool)> {
    if is_negative(y) {
        return Err(value_error("y must be nonnegative".to_string()));
    }
    if *n < Int::ONE {
        return Err(value_error("n must be positive".to_string()));
    }
    if y.is_zero() || y.is_one() || n.is_one() {
        return Ok((y.clone(), true));
    }
    let Ok(n) = usize::try_from(n) else {
        // a root of order beyond the bit length of y is 1
        return Ok((Int::ONE, false));
    };
    if n == 2 {
        let r: Int = isqrt(y);
        let exact: bool = &r * &r == *y;
        return Ok((r, exact));
    }
    // Newton's iteration on the integers from an upper bound
    let nn: Int = Int::from(n);
    let shift: usize = (bits(y) as usize).div_ceil(n);
    let mut r: Int = Int::ONE << shift;
    loop {
        let s: Int = div_floor(&((&nn - Int::ONE) * &r + div_floor(y, &r.pow(n - 1))), &nn);
        if s >= r {
            break;
        }
        r = s;
    }
    while r.pow(n) > *y {
        r -= Int::ONE;
    }
    while (&r + Int::ONE).pow(n) <= *y {
        r += Int::ONE;
    }
    let exact: bool = r.pow(n) == *y;
    Ok((r, exact))
}

/// The largest `e` with `b**e <= y` and whether `b**e == y`, for `y >= 1`
/// and `b >= 2` (`integer_log`).
pub fn integer_log(y: &Int, b: &Int) -> PolyResult<(Int, bool)> {
    if *y < Int::ONE {
        return Err(value_error("y must be positive".to_string()));
    }
    if *b < Int::from(2) {
        return Err(value_error("b must be at least 2".to_string()));
    }
    let mut e: Int = Int::ZERO;
    let mut power: Int = Int::ONE;
    while &power * b <= *y {
        power *= b;
        e += Int::ONE;
    }
    let exact: bool = power == *y;
    Ok((e, exact))
}

/// The number of digits of `|n|` in the given base, `1` for zero
/// (`num_digits`).
pub fn num_digits(n: &Int, base: &Int) -> Int {
    let mut n: Int = n.clone().abs();
    if *base == Int::from(2) {
        return Int::from(bits(&n).max(1));
    }
    let mut digits: u64 = 0;
    while !n.is_zero() {
        n = div_floor(&n, base);
        digits += 1;
    }
    Int::from(digits.max(1))
}

/// The number of trailing zero bits of `n`, `0` for zero (`trailing`).
pub fn trailing(n: &Int) -> Int {
    if n.is_zero() {
        return Int::ZERO;
    }
    let mut m: Int = n.clone().abs();
    let mut count: u64 = 0;
    while (&m % Int::from(2)).is_zero() {
        m = &m >> 1;
        count += 1;
    }
    Int::from(count)
}
