//! Port of `sympy_extras/polys/densetools.py`: derivatives, evaluation,
//! composition, Taylor shifts, contents and primitive parts, monic forms,
//! truncation modulo a ground element, clearing denominators.
//!
//! References
//! ==========
//!
//! 1. D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
//!    Addison-Wesley, 1998, Section 4.6.4 (evaluation, Horner's rule).
//! 2. K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//!    Algebra*, Kluwer, 1992, Section 2.6 (content and primitive part).

use crate::error::PolyResult;
use crate::numbers::*;

use super::densearith::{dup_add_term, dup_exquo_ground, dup_mul, dup_mul_ground};
use super::densebasic::dup_strip_vec;
use super::domains::{Domain, RationalField};

/// The `m`-th derivative.
pub fn dup_diff<K: Domain>(f: &[K::Elem], m: usize, k: &K) -> Vec<K::Elem> {
    let mut result: Vec<K::Elem> = f.to_vec();
    for _ in 0..m {
        if result.len() <= 1 {
            return Vec::new();
        }
        let n: usize = result.len() - 1;
        let derivative: Vec<K::Elem> = result[..n]
            .iter()
            .enumerate()
            .map(|(i, c)| k.mul(c, &k.from_int((n - i) as i64)))
            .collect();
        result = dup_strip_vec(derivative, k);
    }
    result
}

/// `f(a)` by Horner's rule.
pub fn dup_eval<K: Domain>(f: &[K::Elem], a: &K::Elem, k: &K) -> K::Elem {
    let mut result: K::Elem = k.zero();
    for c in f {
        result = k.add(&k.mul(&result, a), c);
    }
    result
}

/// `f(x + a)` (the Taylor shift, by Horner's rule in `x + a`).
pub fn dup_shift<K: Domain>(f: &[K::Elem], a: &K::Elem, k: &K) -> Vec<K::Elem> {
    if f.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<K::Elem> = vec![f[0].clone()];
    for c in &f[1..] {
        // result = result (x + a) + c
        let mut shifted: Vec<K::Elem> = result.clone();
        shifted.push(k.zero());
        for i in 0..result.len() {
            shifted[i + 1] = k.add(&shifted[i + 1], &k.mul(&result[i], a));
        }
        let last: usize = shifted.len() - 1;
        shifted[last] = k.add(&shifted[last], c);
        result = shifted;
    }
    dup_strip_vec(result, k)
}

/// `f(x + a)`: the same as [`dup_shift`].
pub fn dup_taylor_shift<K: Domain>(f: &[K::Elem], a: &K::Elem, k: &K) -> Vec<K::Elem> {
    dup_shift(f, a, k)
}

/// `f(g(x))` by Horner's rule.
pub fn dup_compose<K: Domain>(f: &[K::Elem], g: &[K::Elem], k: &K) -> Vec<K::Elem> {
    if f.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<K::Elem> = vec![f[0].clone()];
    for c in &f[1..] {
        result = dup_add_term(&dup_mul(&result, g, k), c, 0, k);
    }
    result
}

/// `f(-x)`.
pub fn dup_mirror<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    let n: usize = f.len();
    f.iter()
        .enumerate()
        .map(|(i, c)| if (n - 1 - i) % 2 == 0 { c.clone() } else { k.neg(c) })
        .collect()
}

/// `f(a x)`.
pub fn dup_scale<K: Domain>(f: &[K::Elem], a: &K::Elem, k: &K) -> Vec<K::Elem> {
    let mut power: K::Elem = k.one();
    let mut scaled: Vec<K::Elem> = vec![k.zero(); f.len()];
    for i in (0..f.len()).rev() {
        scaled[i] = k.mul(&f[i], &power);
        power = k.mul(&power, a);
    }
    dup_strip_vec(scaled, k)
}

/// The gcd of the coefficients (`0` for the zero polynomial, `1` for
/// nonzero polynomials over a field).
pub fn dup_content<K: Domain>(f: &[K::Elem], k: &K) -> K::Elem {
    if f.is_empty() {
        return k.zero();
    }
    if k.is_field() {
        return k.one();
    }
    let mut content: K::Elem = k.zero();
    for c in f {
        content = k.gcd(&content, c);
        if k.is_one(&content) {
            break;
        }
    }
    content
}

/// The content and the primitive part (`f = content * part`).
pub fn dup_primitive<K: Domain>(f: &[K::Elem], k: &K) -> (K::Elem, Vec<K::Elem>) {
    if f.is_empty() {
        return (k.zero(), Vec::new());
    }
    let content: K::Elem = dup_content(f, k);
    if k.is_one(&content) {
        return (content, f.to_vec());
    }
    let part: Vec<K::Elem> = dup_exquo_ground(f, &content, k).expect("the content divides every coefficient");
    (content, part)
}

/// The content with the sign of the leading coefficient and the primitive
/// part with positive leading coefficient (ordered domains); the same as
/// [`dup_primitive`] otherwise.
pub fn dup_decompose_content<K: Domain>(f: &[K::Elem], k: &K) -> (K::Elem, Vec<K::Elem>) {
    let (content, part) = dup_primitive(f, k);
    if !part.is_empty() && k.is_ordered() && k.is_negative(&part[0]) {
        return (k.neg(&content), part.iter().map(|c| k.neg(c)).collect());
    }
    (content, part)
}

/// `f` divided by its leading coefficient (a field, or a ring where the
/// leading coefficient divides every coefficient; `Value` otherwise).
pub fn dup_monic<K: Domain>(f: &[K::Elem], k: &K) -> PolyResult<Vec<K::Elem>> {
    if f.is_empty() {
        return Ok(Vec::new());
    }
    let lc: &K::Elem = &f[0];
    if k.is_one(lc) {
        return Ok(f.to_vec());
    }
    dup_exquo_ground(f, lc, k)
}

/// [`dup_monic`] over a field, where it cannot fail.
pub(crate) fn dup_monic_field<K: Domain>(f: &[K::Elem], k: &K) -> Vec<K::Elem> {
    dup_monic(f, k).expect("a nonzero leading coefficient is invertible in a field")
}

/// The least common multiple of the denominators and `f` times it (a
/// polynomial over `QQ` with integer coefficients).
pub fn dup_clear_denoms(f: &[Rat], k: &RationalField) -> (Int, Vec<Rat>) {
    let mut common: Int = Int::ONE;
    for c in f {
        common = lcm(&common, &k.denom(c));
    }
    if common.is_one() {
        return (common, f.to_vec());
    }
    let scaled: Vec<Rat> = dup_mul_ground(f, &k.from_integer(&common), k);
    (common, scaled)
}

/// The coefficients reduced modulo `p` (in an ordered domain into the
/// symmetric range `(-p/2, p/2]`, as the modular algorithms need; with the
/// domain's `rem` otherwise).
pub fn dup_trunc<K: Domain>(f: &[K::Elem], p: &K::Elem, k: &K) -> PolyResult<Vec<K::Elem>> {
    let mut result: Vec<K::Elem> = Vec::with_capacity(f.len());
    for c in f {
        let mut r: K::Elem = k.rem(c, p)?;
        if k.is_ordered() {
            if k.is_negative(&r) {
                r = k.add(&r, p);
            }
            // now 0 <= r < p: into (-p/2, p/2]
            if k.is_negative(&k.sub(p, &k.add(&r, &r))) {
                r = k.sub(&r, p);
            }
        }
        result.push(r);
    }
    Ok(dup_strip_vec(result, k))
}

/// The `m`-th antiderivative with zero constants (a field, or a ring where
/// the divisions are exact; `Value` otherwise).
pub fn dup_integrate<K: Domain>(f: &[K::Elem], m: usize, k: &K) -> PolyResult<Vec<K::Elem>> {
    let mut result: Vec<K::Elem> = f.to_vec();
    for _ in 0..m {
        if result.is_empty() {
            continue;
        }
        let n: usize = result.len();
        let mut integrated: Vec<K::Elem> = Vec::with_capacity(n + 1);
        for (i, c) in result.iter().enumerate() {
            integrated.push(k.exquo(c, &k.from_int((n - i) as i64))?);
        }
        integrated.push(k.zero());
        result = integrated;
    }
    Ok(dup_strip_vec(result, k))
}

/// The number of sign changes in the sequence of coefficients (ordered
/// domains; zeros skipped).
pub fn dup_sign_variations<K: Domain>(f: &[K::Elem], k: &K) -> usize {
    let mut count: usize = 0;
    let mut previous: i32 = 0;
    for c in f {
        let s: i32 = k.sign(c);
        if s != 0 && previous != 0 && s != previous {
            count += 1;
        }
        if s != 0 {
            previous = s;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::super::densebasic::testing::{q, qq, zz};
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;
    use crate::polys::domains::{QQ, ZZ};

    #[test]
    fn doctests() {
        assert_eq!(dup_diff(&zz(&[1, 0, -2, 5]), 1, &ZZ), zz(&[3, 0, -2]));
        assert_eq!(dup_eval(&zz(&[1, 0, -2]), &Int::from(3), &ZZ), Int::from(7));
        assert_eq!(dup_shift(&zz(&[1, 0, 0]), &Int::from(1), &ZZ), zz(&[1, 2, 1]));
        assert_eq!(dup_compose(&zz(&[1, 0, 1]), &zz(&[1, 1]), &ZZ), zz(&[1, 2, 2]));
        assert_eq!(dup_content(&zz(&[4, -6, 8]), &ZZ), Int::from(2));
        assert_eq!(dup_primitive(&zz(&[4, -6, 8]), &ZZ), (Int::from(2), zz(&[2, -3, 4])));
        assert_eq!(dup_monic(&qq(&[2, 4]), &QQ).unwrap(), qq(&[1, 2]));
        assert_eq!(dup_clear_denoms(&[q(1, 2), q(1, 3)], &QQ), (Int::from(6), qq(&[3, 2])));
        assert_eq!(
            dup_trunc(&zz(&[7, 8, 9]), &Int::from(5), &ZZ).unwrap(),
            zz(&[2, -2, -1])
        );
        assert_eq!(dup_integrate(&qq(&[3, 0, -2]), 1, &QQ).unwrap(), qq(&[1, 0, -2, 0]));
        assert_eq!(dup_mirror(&zz(&[1, 2, 3]), &ZZ), zz(&[1, -2, 3]));
        assert_eq!(dup_scale(&zz(&[1, 1, 1]), &Int::from(2), &ZZ), zz(&[4, 2, 1]));
        assert_eq!(dup_sign_variations(&zz(&[1, 0, -1, 2, 2, -3]), &ZZ), 3);
        assert_eq!(dup_decompose_content(&zz(&[-2, 4]), &ZZ), (Int::from(-2), zz(&[1, -2])));
    }
}
