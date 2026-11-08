//! Port of `sympy_extras/ntheory/primetest.py`: primality testing, the
//! Baillie–PSW test.
//!
//! `isprime(n)` runs trial division by the small primes, a strong Fermat
//! (Miller–Rabin) test to base 2 and a strong Lucas test with Selfridge's
//! parameters (the first `D` in `5, -7, 9, -11, ...` with Jacobi symbol
//! `(D/n) = -1`, `P = 1`, `Q = (1 - D)/4`). No composite number passing both
//! is known; every composite below `2**64` fails.
//!
//! References
//! ==========
//!
//! 1. R. Baillie, S. S. Wagstaff, *Lucas pseudoprimes*, Mathematics of
//!    Computation 35 (1980), 1391-1417.
//! 2. C. Pomerance, J. L. Selfridge, S. S. Wagstaff, *The pseudoprimes to 25
//!    * 10^9*, Mathematics of Computation 35 (1980), 1003-1026.
//! 3. R. Crandall, C. Pomerance, *Prime Numbers: A Computational
//!    Perspective*, 2nd ed., Springer, 2005, Sections 3.5 and 3.6.

use std::sync::OnceLock;

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

/// The primes below 1000, for trial division.
pub fn small_primes() -> &'static [u64] {
    static PRIMES: OnceLock<Vec<u64>> = OnceLock::new();
    PRIMES.get_or_init(|| {
        (2..1000u64)
            .filter(|p| (2..*p).take_while(|q| q * q <= *p).all(|q| p % q != 0))
            .collect()
    })
}

/// `base**exponent mod modulus` (Python's three-argument `pow`, `exponent >=
/// 0`, `modulus > 0`).
pub fn powmod(base: &Int, exponent: &Int, modulus: &Int) -> Int {
    let mut result: Int = mod_floor(&Int::ONE, modulus);
    let b: Int = mod_floor(base, modulus);
    let n: u64 = bits(exponent);
    for i in (0..n as usize).rev() {
        result = mod_floor(&(&result * &result), modulus);
        if bit(exponent, i) {
            result = mod_floor(&(&result * &b), modulus);
        }
    }
    result
}

fn is_even(n: &Int) -> bool {
    !bit(&abs(n), 0)
}

/// The Jacobi symbol `(a/n)` for odd `n > 0` (`Value` otherwise).
pub fn jacobi_symbol(a: &Int, n: &Int) -> PolyResult<i32> {
    if *n <= Int::ZERO || is_even(n) {
        return Err(value_error(format!(
            "the Jacobi symbol needs an odd positive modulus, got {}",
            n
        )));
    }
    let eight: Int = Int::from(8u8);
    let four: Int = Int::from(4u8);
    let mut n: Int = n.clone();
    let mut a: Int = mod_floor(a, &n);
    let mut result: i32 = 1;
    while !a.is_zero() {
        while is_even(&a) {
            a = div_floor(&a, &Int::from(2u8));
            let r: Int = mod_floor(&n, &eight);
            if r == Int::from(3u8) || r == Int::from(5u8) {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if mod_floor(&a, &four) == Int::from(3u8) && mod_floor(&n, &four) == Int::from(3u8) {
            result = -result;
        }
        a = mod_floor(&a, &n);
    }
    Ok(if n.is_one() { result } else { 0 })
}

/// `(d, s)` with `m = 2**s d`, `d` odd (`m > 0`).
fn split_twos(m: &Int) -> (Int, u64) {
    let mut d: Int = m.clone();
    let mut s: u64 = 0;
    while is_even(&d) {
        d = div_floor(&d, &Int::from(2u8));
        s += 1;
    }
    (d, s)
}

/// Whether the odd `n > 2` is a strong probable prime to `base`.
pub fn is_strong_probable_prime(n: &Int, base: &Int) -> bool {
    let n_minus_one: Int = n - Int::ONE;
    let (d, s) = split_twos(&n_minus_one);
    let mut x: Int = powmod(base, &d, n);
    if x.is_one() || x == n_minus_one {
        return true;
    }
    for _ in 1..s {
        x = mod_floor(&(&x * &x), n);
        if x == n_minus_one {
            return true;
        }
    }
    false
}

/// The strong Lucas test with Selfridge's parameters, for odd `n` which is
/// not a perfect square.
pub fn is_strong_lucas_probable_prime(n: &Int) -> bool {
    let mut d_param: Int = Int::from(5u8);
    loop {
        let j: i32 = jacobi_symbol(&d_param, n).expect("an odd positive modulus");
        if j == -1 {
            break;
        }
        if j == 0 && abs(&d_param) != *n {
            return false;
        }
        d_param = if is_negative(&d_param) {
            -&d_param + Int::from(2u8)
        } else {
            -&d_param - Int::from(2u8)
        };
    }
    let p: Int = Int::ONE;
    let q: Int = div_floor(&(Int::ONE - &d_param), &Int::from(4u8));
    let (d, s) = split_twos(&(n + Int::ONE));
    // U_k, V_k, Q**k by the binary expansion of d (doubling and adding one)
    let (mut u, mut v, mut qk): (Int, Int, Int) = (Int::ZERO, Int::from(2u8), Int::ONE);
    let inverse_two: Int = div_floor(&(n + Int::ONE), &Int::from(2u8));
    for i in (0..bits(&d) as usize).rev() {
        // doubling: U_2k = U_k V_k, V_2k = V_k**2 - 2 Q**k
        let u2: Int = mod_floor(&(&u * &v), n);
        let v2: Int = mod_floor(&(&v * &v - Int::from(2u8) * &qk), n);
        qk = mod_floor(&(&qk * &qk), n);
        u = u2;
        v = v2;
        if bit(&d, i) {
            // k -> k + 1: U_(k+1) = (P U + V)/2, V_(k+1) = (D U + P V)/2
            let u1: Int = mod_floor(&((&p * &u + &v) * &inverse_two), n);
            let v1: Int = mod_floor(&((&d_param * &u + &p * &v) * &inverse_two), n);
            u = u1;
            v = v1;
            qk = mod_floor(&(&qk * &q), n);
        }
    }
    if u.is_zero() || v.is_zero() {
        return true;
    }
    for _ in 1..s {
        v = mod_floor(&(&v * &v - Int::from(2u8) * &qk), n);
        qk = mod_floor(&(&qk * &qk), n);
        if v.is_zero() {
            return true;
        }
    }
    false
}

/// Whether `n` is prime (Baillie–PSW; deterministic below `2**64`).
pub fn isprime(n: &Int) -> bool {
    if *n < Int::from(2u8) {
        return false;
    }
    for p in small_primes() {
        let pi: Int = Int::from(*p);
        if *n == pi {
            return true;
        }
        if mod_floor(n, &pi).is_zero() {
            return false;
        }
    }
    if *n < Int::from(1_000_000u32) {
        return true;
    }
    if !is_strong_probable_prime(n, &Int::from(2u8)) {
        return false;
    }
    let root: Int = isqrt(n);
    if &root * &root == *n {
        return false;
    }
    is_strong_lucas_probable_prime(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctests() {
        let small: Vec<i64> = (0..30).filter(|n| isprime(&int(*n))).collect();
        assert_eq!(small, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
        let m61: Int = Int::from(2u8).pow(61) - Int::ONE;
        assert!(isprime(&m61));
        assert!(!isprime(&(&m61 + Int::from(2u8))));
        assert!(!isprime(&int(3215031751)));
        assert!(is_strong_probable_prime(&int(3215031751), &int(2)));
        assert_eq!(jacobi_symbol(&int(5), &int(21)).unwrap(), 1);
        assert!(jacobi_symbol(&int(5), &int(20)).is_err());
        assert_eq!(powmod(&int(3), &int(100), &int(7)), int(4));
    }
}
