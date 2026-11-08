//! Port of `sympy_extras/ntheory/generate.py`: the next and the previous
//! prime.

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::primetest::isprime;

fn is_even(n: &Int) -> bool {
    !bit(&abs(n), 0)
}

/// The least prime greater than `n`.
pub fn nextprime(n: &Int) -> Int {
    if *n < Int::from(2u8) {
        return Int::from(2u8);
    }
    let mut candidate: Int = if is_even(n) { n + Int::ONE } else { n + Int::from(2u8) };
    while !isprime(&candidate) {
        candidate += Int::from(2u8);
    }
    candidate
}

/// The largest prime less than `n` (`n > 2`; `Value` otherwise).
pub fn prevprime(n: &Int) -> PolyResult<Int> {
    if *n <= Int::from(2u8) {
        return Err(value_error(format!("no prime below {}", n)));
    }
    if *n == Int::from(3u8) {
        return Ok(Int::from(2u8));
    }
    let mut candidate: Int = if is_even(n) { n - Int::ONE } else { n - Int::from(2u8) };
    while !isprime(&candidate) {
        candidate -= Int::from(2u8);
    }
    Ok(candidate)
}

/// The primes `p` with `a <= p < b`, in increasing order (Python's
/// generator `primerange`).
pub fn primerange(a: &Int, b: &Int) -> PrimeRange {
    PrimeRange {
        next: nextprime(&(a - Int::ONE)),
        end: b.clone(),
    }
}

/// The iterator of [`primerange`].
#[derive(Debug, Clone)]
pub struct PrimeRange {
    next: Int,
    end: Int,
}

impl Iterator for PrimeRange {
    type Item = Int;

    fn next(&mut self) -> Option<Int> {
        if self.next >= self.end {
            return None;
        }
        let p: Int = self.next.clone();
        self.next = nextprime(&p);
        Some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctests() {
        assert_eq!(nextprime(&int(100)), int(101));
        assert_eq!(nextprime(&int(2)), int(3));
        assert_eq!(prevprime(&int(100)).unwrap(), int(97));
        assert_eq!(prevprime(&int(3)).unwrap(), int(2));
        assert!(prevprime(&int(2)).is_err());
        assert_eq!(
            primerange(&int(10), &int(30)).collect::<Vec<Int>>(),
            [11, 13, 17, 19, 23, 29].map(int).to_vec()
        );
        let big: Int = Int::from(2u8).pow(1024);
        assert_eq!(&big - prevprime(&big).unwrap(), int(105));
    }
}
