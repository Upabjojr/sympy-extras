//! Port of `sympy_extras/core/power.py`, so far its integer part:
//! [`integer_root_parts`], the decomposition of `p**e` for a positive
//! integer `p` and `0 < e < 1`. `Pow`, `pow` and `number_pow` follow.

use crate::ntheory::factorint;
use crate::numbers::*;

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
