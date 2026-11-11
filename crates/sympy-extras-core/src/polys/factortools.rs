//! Port of `sympy_extras/polys/factortools.py` (the part over `ZZ` and
//! `QQ`): factorisation of dense univariate polynomials, Zassenhaus'
//! algorithm.
//!
//! For a primitive square-free `f` in `ZZ[x]` with leading coefficient `l`:
//! a prime `p` not dividing `l` such that `f` stays square-free modulo `p`
//! is chosen (among the first few such primes, the one giving the fewest
//! modular factors), `f` is factored over `GF(p)`, the factorisation `f = l
//! f_1 ... f_r (mod p)` is lifted to a modulus `m = p**(2**j)` above twice
//! the Mignotte bound by quadratic Hensel lifting (pairs of factors, the
//! list split in halves recursively), and the true factors are found by
//! trial division of the primitive parts of `l prod_{i in S} f_i mod m`
//! (symmetric residues) for the subsets `S` of increasing size up to half
//! the remaining factors. From ten modular factors on, van Hoeij's knapsack
//! recombination ([`dup_zz_van_hoeij`]) replaces the subset search. Over
//! `QQ` the denominators are cleared first. The square-free decomposition
//! comes first.
//!
//! Over a number field `QQ(alpha)` ([`AlgebraicField`]) Trager's algorithm:
//! a shift `x -> x - s alpha` making the norm square-free, the norm factored
//! over `QQ`, and each of its factors giving a factor by a gcd over the
//! field.
//!
//! References
//! ==========
//!
//! 1. J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
//!    Cambridge University Press, 2013, Chapter 15 (Algorithms 15.10,
//!    15.17, 15.19; the Mignotte bound, Corollary 6.33).
//! 2. H. Zassenhaus, *On Hensel factorization I*, Journal of Number Theory
//!    1 (1969), 291-311.
//! 3. K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//!    Algebra*, Kluwer, 1992, Chapter 8.

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::densearith::{dup_add, dup_add_term, dup_div, dup_max_norm, dup_mul, dup_mul_ground, dup_sub};
use super::densebasic::{dup_LC, dup_convert, dup_degree, dup_strip};
use super::densetools::{
    dup_clear_denoms, dup_decompose_content, dup_diff, dup_monic, dup_primitive, dup_shift, dup_trunc,
};
use super::domains::algebraicfield::{AlgebraicElement, AlgebraicField};
use super::domains::{Domain, FiniteField, IntegerRing, PolynomialRing, RationalField, QQ, ZZ};
use super::euclidtools::{dup_gcd, dup_gcdex, dup_resultant};
use super::galoistools::gf_factor_sqf;
use super::lattice::{default_delta, hermite_normal_form, lll_reduction};
use super::sqfreetools::{dup_sqf_list, dup_sqf_p, Factor};

/// How many good primes are tried, keeping the one with the fewest factors.
const PRIMES_TRIED: usize = 5;
/// From this number of modular factors on, the factors are recombined by
/// lattice reduction (van Hoeij) rather than by the subset search.
const KNAPSACK_THRESHOLD: usize = 10;
/// How many times the lifting is squared before van Hoeij's recombination
/// gives up for the subset search.
const KNAPSACK_LIFTS: usize = 4;

fn trunc(f: &[Int], m: &Int) -> Vec<Int> {
    dup_trunc(f, m, &ZZ).expect("a nonzero modulus")
}

/// The division of the Hensel steps, by a monic polynomial, which is exact.
fn div_monic(f: &[Int], g: &[Int]) -> (Vec<Int>, Vec<Int>) {
    dup_div(f, g, &ZZ).expect("a division by a monic polynomial")
}

fn to_field(f: &[Int], field: &FiniteField) -> Vec<u64> {
    dup_convert(f, &ZZ, field).expect("integers have images modulo p")
}

fn from_field(f: &[u64], field: &FiniteField) -> Vec<Int> {
    dup_convert(f, field, &ZZ).expect("residues are integers")
}

/// The sort key of the Python code, `(len(g), g)`.
fn by_degree_then_coefficients(a: &Vec<Int>, b: &Vec<Int>) -> std::cmp::Ordering {
    (a.len(), a).cmp(&(b.len(), b))
}

/// `(isqrt(n + 1) + 1) 2**n ||f||_inf |lc(f)|`, a bound on the absolute
/// values of the coefficients of `lc(f) g` for every factor `g` of `f`
/// (Mignotte).
pub fn dup_zz_mignotte_bound(f: &[Int], k: &IntegerRing) -> Int {
    let n: usize = f.len() - 1;
    let root: Int = isqrt(&Int::from(n + 1)) + Int::ONE;
    root * Int::from(2u8).pow(n) * dup_max_norm(f, k) * abs(&f[0])
}

/// The first 60 primes, the ones tried in turn.
fn primes() -> Vec<u64> {
    let mut found: Vec<u64> = Vec::new();
    let mut candidate: u64 = 2;
    while found.len() < 60 {
        if found.iter().all(|q| candidate % q != 0) {
            found.push(candidate);
        }
        candidate += 1;
    }
    found
}

/// One quadratic Hensel step: from `f = g h`, `s g + t h = 1` modulo `m`
/// (`h` monic, `deg s < deg h`, `deg t < deg g`) to the same congruences
/// modulo `m**2`.
pub fn dup_zz_hensel_step(
    m: &Int,
    f: &[Int],
    g: &[Int],
    h: &[Int],
    s: &[Int],
    t: &[Int],
) -> (Vec<Int>, Vec<Int>, Vec<Int>, Vec<Int>) {
    let k: &IntegerRing = &ZZ;
    let big_m: Int = m * m;
    let e: Vec<Int> = trunc(&dup_sub(f, &dup_mul(g, h, k), k), &big_m);
    let (q, r) = div_monic(&dup_mul(s, &e, k), h);
    let (q, r) = (trunc(&q, &big_m), trunc(&r, &big_m));
    let g1: Vec<Int> = trunc(
        &dup_add(&dup_add(g, &dup_mul(t, &e, k), k), &dup_mul(&q, g, k), k),
        &big_m,
    );
    let h1: Vec<Int> = trunc(&dup_add(h, &r, k), &big_m);
    let b: Vec<Int> = trunc(
        &dup_sub(&dup_add(&dup_mul(s, &g1, k), &dup_mul(t, &h1, k), k), &[Int::ONE], k),
        &big_m,
    );
    let (c, d) = div_monic(&dup_mul(s, &b, k), &h1);
    let (c, d) = (trunc(&c, &big_m), trunc(&d, &big_m));
    let s1: Vec<Int> = trunc(&dup_sub(s, &d, k), &big_m);
    let t1: Vec<Int> = trunc(
        &dup_sub(&dup_sub(t, &dup_mul(t, &b, k), k), &dup_mul(&c, &g1, k), k),
        &big_m,
    );
    (g1, h1, s1, t1)
}

/// `f = g h` modulo `p` lifted to modulus `m = p**(2**j) >= target`; `h`
/// monic. Returns `(g, h, m)`.
fn lift_pair(f: &[Int], g: &[Int], h: &[Int], p: u64, target: &Int) -> PolyResult<(Vec<Int>, Vec<Int>, Int)> {
    let field: FiniteField = FiniteField::new(p)?;
    let gp: Vec<u64> = to_field(g, &field);
    let hp: Vec<u64> = to_field(h, &field);
    let (sp, tp, one) = dup_gcdex(&gp, &hp, &field)?;
    if dup_degree(&one) != 0 {
        return Err(value_error("the modular factors are not coprime".to_string()));
    }
    let pi: Int = Int::from(p);
    let mut s: Vec<Int> = trunc(&from_field(&sp, &field), &pi);
    let mut t: Vec<Int> = trunc(&from_field(&tp, &field), &pi);
    let (mut g, mut h): (Vec<Int>, Vec<Int>) = (trunc(g, &pi), trunc(h, &pi));
    let mut m: Int = pi;
    while &m < target {
        (g, h, s, t) = dup_zz_hensel_step(&m, f, &g, &h, &s, &t);
        m = &m * &m;
    }
    Ok((g, h, m))
}

/// The factorisation `f = lc(f) f_1 ... f_r` modulo `p` (monic `f_i`)
/// lifted to a modulus `m = p**(2**j) >= target`: the lifted monic factors
/// and `m` (multifactor lifting by halves).
pub fn dup_zz_hensel_lift(p: u64, f: &[Int], factors: &[Vec<Int>], target: &Int) -> PolyResult<(Vec<Vec<Int>>, Int)> {
    let r: usize = factors.len();
    let lc: &Int = &f[0];
    let pi: Int = Int::from(p);
    if r == 1 {
        // f = lc f_1: the factor is f made monic modulo m
        let mut m: Int = pi;
        while &m < target {
            m = &m * &m;
        }
        let inverse: Int = mod_inverse(lc, &m)
            .ok_or_else(|| value_error("base is not invertible for the given modulus".to_string()))?;
        return Ok((vec![trunc(&dup_mul_ground(f, &inverse, &ZZ), &m)], m));
    }
    let half: usize = r / 2;
    let field: FiniteField = FiniteField::new(p)?;
    let lc_p: u64 = field.from_integer(lc);
    let mut g: Vec<u64> = if lc_p != 0 { vec![lc_p] } else { Vec::new() };
    for factor in &factors[..half] {
        g = dup_mul(&g, &to_field(factor, &field), &field);
    }
    let mut h: Vec<u64> = vec![1];
    for factor in &factors[half..] {
        h = dup_mul(&h, &to_field(factor, &field), &field);
    }
    let (g_lifted, h_lifted, m) = lift_pair(f, &from_field(&g, &field), &from_field(&h, &field), p, target)?;
    let (mut left, _) = dup_zz_hensel_lift(p, &g_lifted, &factors[..half], target)?;
    let (right, _) = dup_zz_hensel_lift(p, &h_lifted, &factors[half..], target)?;
    left.extend(right);
    Ok((left, m))
}

/// A prime not dividing the leading coefficient modulo which `f` stays
/// square-free, with the modular factors; the one with the fewest factors
/// among the first few.
fn good_prime(f: &[Int]) -> PolyResult<(u64, Vec<Vec<Int>>)> {
    let lc: &Int = &f[0];
    let mut best: (u64, Vec<Vec<Int>>) = (0, Vec::new());
    let mut tried: usize = 0;
    for p in primes() {
        if mod_floor(lc, &Int::from(p)).is_zero() {
            continue;
        }
        let field: FiniteField = FiniteField::new(p)?;
        let fp: Vec<u64> = to_field(f, &field);
        if dup_degree(&fp) != dup_degree(f) {
            continue;
        }
        if dup_degree(&dup_gcd(&fp, &dup_diff(&fp, 1, &field), &field)) > 0 {
            continue;
        }
        let (_, factors) = gf_factor_sqf(&fp, &field);
        if best.0 == 0 || factors.len() < best.1.len() {
            best = (p, factors.iter().map(|g| from_field(g, &field)).collect());
        }
        tried += 1;
        if tried >= PRIMES_TRIED || best.1.len() == 1 {
            break;
        }
    }
    if best.0 == 0 {
        return Err(value_error(format!("no suitable prime found for {:?}", f)));
    }
    Ok(best)
}

/// The `n`-th cyclotomic polynomial: `x**n - 1` divided by the cyclotomic
/// polynomials of the proper divisors of `n`.
pub fn dup_zz_cyclotomic_poly(n: usize) -> Vec<Int> {
    let mut result: Vec<Int> = vec![Int::ZERO; n + 1];
    result[0] = Int::ONE;
    result[n] = -Int::ONE;
    for d in 1..n {
        if n % d == 0 {
            result = div_monic(&result, &dup_zz_cyclotomic_poly(d)).0;
        }
    }
    result
}

/// The irreducible factors of `x**n - 1` (the cyclotomic polynomials of the
/// divisors of `n`) and of `x**n + 1` (those of the divisors of `2 n` which
/// do not divide `n`); `None` when `f` has another shape.
pub fn dup_zz_cyclotomic_factor(f: &[Int]) -> Option<Vec<Vec<Int>>> {
    let n: isize = dup_degree(f);
    if n < 1 {
        return None;
    }
    let n: usize = n as usize;
    let last: &Int = &f[n];
    if !f[0].is_one() || f[1..n].iter().any(|c| !c.is_zero()) || !abs(last).is_one() {
        return None;
    }
    let divisors: Vec<usize> = if is_negative(last) {
        (1..n + 1).filter(|d| n % d == 0).collect()
    } else {
        (1..2 * n + 1).filter(|d| (2 * n) % d == 0 && n % d != 0).collect()
    };
    let mut factors: Vec<Vec<Int>> = divisors.into_iter().map(dup_zz_cyclotomic_poly).collect();
    factors.sort_by(by_degree_then_coefficients);
    Some(factors)
}

/// The subsets of `0..n` of size `size` in lexicographic order (Python's
/// `itertools.combinations(range(n), size)`).
fn combinations(n: usize, size: usize) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    if size > n {
        return result;
    }
    let mut indices: Vec<usize> = (0..size).collect();
    loop {
        result.push(indices.clone());
        // the rightmost index which can still move
        let movable: Option<usize> = (0..size).rev().find(|&j| indices[j] != j + n - size);
        let i: usize = match movable {
            Some(i) => i,
            None => return result,
        };
        indices[i] += 1;
        for j in i + 1..size {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

/// The irreducible factors (primitive, positive leading coefficient) of a
/// primitive square-free `f` with positive leading coefficient, sorted.
pub fn dup_zz_zassenhaus(f: &[Int]) -> PolyResult<Vec<Vec<Int>>> {
    if dup_degree(f) <= 1 {
        return Ok(vec![f.to_vec()]);
    }
    if let Some(cyclotomic) = dup_zz_cyclotomic_factor(f) {
        return Ok(cyclotomic);
    }
    let (p, modular) = good_prime(f)?;
    if modular.len() == 1 {
        return Ok(vec![f.to_vec()]);
    }
    if modular.len() >= KNAPSACK_THRESHOLD {
        return knapsack_recombination(f, p, &modular);
    }
    let bound: Int = dup_zz_mignotte_bound(f, &ZZ);
    let target: Int = Int::from(2u8) * bound + Int::ONE;
    let (lifted, m) = dup_zz_hensel_lift(p, f, &modular, &target)?;
    Ok(subset_recombination(f, lifted, &m))
}

/// The irreducible factors of `f` from its monic factors modulo `m` (above
/// twice the Mignotte bound): trial division by the products over the
/// subsets of increasing size.
fn subset_recombination(f: &[Int], lifted: Vec<Vec<Int>>, m: &Int) -> Vec<Vec<Int>> {
    let mut remaining: Vec<Vec<Int>> = lifted;
    let mut factors: Vec<Vec<Int>> = Vec::new();
    let mut current: Vec<Int> = f.to_vec();
    let mut size: usize = 1;
    while 2 * size <= remaining.len() {
        let mut found: bool = false;
        for subset in combinations(remaining.len(), size) {
            let chosen: Vec<Vec<Int>> = subset.iter().map(|&i| remaining[i].clone()).collect();
            let candidate: Vec<Int> = candidate_factor(&current, &chosen, m);
            let quotient: Option<Vec<Int>> = if divides_leading(&current, &candidate) {
                zz_trial_division(&current, &candidate)
            } else {
                None
            };
            if let Some(quotient) = quotient {
                factors.push(candidate);
                current = quotient;
                remaining = remaining
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !subset.contains(i))
                    .map(|(_, g)| g)
                    .collect();
                found = true;
                break;
            }
        }
        if !found {
            size += 1;
        }
    }
    if dup_degree(&current) > 0 {
        factors.push(current);
    }
    factors.sort_by(by_degree_then_coefficients);
    factors
}

/// The primitive part, with positive leading coefficient, of `lc(f)` times
/// the product of the monic factors modulo `m` (symmetric residues): the
/// factor of `f` they lift, if they lift one.
fn candidate_factor(f: &[Int], modular: &[Vec<Int>], m: &Int) -> Vec<Int> {
    let mut candidate: Vec<Int> = vec![mod_floor(&dup_LC(f, &ZZ), m)];
    for g in modular {
        candidate = trunc(&dup_mul(&candidate, g, &ZZ), m);
    }
    let (_, mut candidate) = dup_primitive(&candidate, &ZZ);
    if is_negative(&dup_LC(&candidate, &ZZ)) {
        candidate = candidate.iter().map(|c| -c).collect();
    }
    candidate
}

// ---------------------------------------------------------------------------
// van Hoeij's recombination

/// The smallest integer `r >= 0` with `r**k >= a`.
fn root_ceiling(a: &Int, k: usize) -> Int {
    if *a <= Int::ZERO {
        return Int::ZERO;
    }
    let (mut low, mut high): (Int, Int) = (Int::ZERO, Int::ONE);
    while high.pow(k) < *a {
        high *= Int::from(2u8);
    }
    while &high - &low > Int::ONE {
        let middle: Int = div_floor(&(&low + &high), &Int::from(2u8));
        if middle.pow(k) >= *a {
            high = middle;
        } else {
            low = middle;
        }
    }
    high
}

/// An integer `R >= |lc(f) alpha|` for every complex root `alpha` of `f`:
/// Fujiwara's bound `2 max |c_i|**(1/i)` for the monic polynomial
/// `y**n + sum c_i y**(n - i)` with the roots `lc(f) alpha`,
/// `c_i = f_i lc(f)**(i - 1)`.
fn scaled_root_bound(f: &[Int]) -> Int {
    let lc: Int = abs(&f[0]);
    let mut bound: Int = Int::ONE;
    for (i, c) in f.iter().enumerate().skip(1) {
        let candidate: Int = root_ceiling(&(abs(c) * lc.pow(i - 1)), i);
        if candidate > bound {
            bound = candidate;
        }
    }
    Int::from(2u8) * bound
}

/// The power sums `p_1, ..., p_count` of the roots of the monic `g` modulo
/// `m` (Newton's identities: `p_k = -k g_k - sum_{0 < i < k} g_i p_{k - i}`,
/// without division).
fn power_sums(g: &[Int], count: usize, m: &Int) -> Vec<Int> {
    let d: usize = g.len() - 1;
    let mut sums: Vec<Int> = Vec::new();
    for k in 1..=count {
        let mut total: Int = if k <= d { -Int::from(k) * &g[k] } else { Int::ZERO };
        for i in 1..=std::cmp::min(k - 1, d) {
            total -= &g[i] * &sums[k - i - 1];
        }
        sums.push(mod_floor(&total, m));
    }
    sums
}

/// The reduced row echelon form over `QQ` of independent rows.
fn rational_echelon(rows: &[Vec<Int>]) -> Vec<Vec<Rat>> {
    let mut work: Vec<Vec<Rat>> = rows.iter().map(|row| row.iter().map(rat_from_int).collect()).collect();
    let width: usize = work.first().map_or(0, |row| row.len());
    let mut rank: usize = 0;
    for column in 0..width {
        let Some(pivot) = (rank..work.len()).find(|&i| !work[i][column].is_zero()) else {
            continue;
        };
        work.swap(rank, pivot);
        let inverse: Rat = recip(&work[rank][column]);
        work[rank] = work[rank].iter().map(|v| v * &inverse).collect();
        for i in 0..work.len() {
            if i != rank && !work[i][column].is_zero() {
                let c: Rat = work[i][column].clone();
                let top: Vec<Rat> = work[rank].clone();
                work[i] = work[i].iter().zip(&top).map(|(a, b)| a - &c * b).collect();
            }
        }
        rank += 1;
    }
    work.truncate(rank);
    work
}

/// The partition of the modular factors a lattice basis describes: when its
/// reduced echelon form consists of 0-1 vectors with exactly one 1 in every
/// column, the index sets of the rows; `None` otherwise.
fn partition_of(basis: &[Vec<Int>]) -> Option<Vec<Vec<usize>>> {
    let echelon: Vec<Vec<Rat>> = rational_echelon(basis);
    if echelon.is_empty() {
        return None;
    }
    let one: Rat = Rat::ONE;
    let width: usize = echelon[0].len();
    for column in 0..width {
        let mut ones: usize = 0;
        for row in &echelon {
            if row[column] == one {
                ones += 1;
            } else if !row[column].is_zero() {
                return None;
            }
        }
        if ones != 1 {
            return None;
        }
    }
    Some(
        echelon
            .iter()
            .map(|row| (0..width).filter(|&i| row[i] == one).collect())
            .collect(),
    )
}

/// `a**e mod m` in `0..m` (Python's `pow(a, e, m)`).
fn pow_mod(a: &Int, e: usize, m: &Int) -> Int {
    let mut result: Int = mod_floor(&Int::ONE, m);
    let mut base: Int = mod_floor(a, m);
    let mut e: usize = e;
    while e > 0 {
        if e & 1 == 1 {
            result = mod_floor(&(&result * &base), m);
        }
        base = mod_floor(&(&base * &base), m);
        e >>= 1;
    }
    result
}

/// The number of bits of a machine integer (Python's `int.bit_length`).
fn bit_length(n: usize) -> usize {
    (usize::BITS - n.leading_zeros()) as usize
}

/// The irreducible factors of the primitive square-free `f` from its monic
/// irreducible factors modulo `p`, by van Hoeij's algorithm (see
/// [`dup_zz_van_hoeij`]).
fn knapsack_recombination(f: &[Int], p: u64, modular: &[Vec<Int>]) -> PolyResult<Vec<Vec<Int>>> {
    let n: usize = dup_degree(f) as usize;
    let r: usize = modular.len();
    let lc: Int = dup_LC(f, &ZZ);
    let root_bound: Int = scaled_root_bound(f);
    let traces: usize = 2;
    let p_int: Int = Int::from(p);
    let n_int: Int = Int::from(n);
    let target: Int = Int::from(2u8) * dup_zz_mignotte_bound(f, &ZZ) + Int::ONE;
    let (mut lifted, mut m) = dup_zz_hensel_lift(p, f, modular, &target)?;
    let mut basis: Vec<Vec<Int>> = (0..r)
        .map(|i| (0..r).map(|j| if i == j { Int::ONE } else { Int::ZERO }).collect())
        .collect();
    // enough digits for the Gram-Schmidt lengths of the vectors which do not
    // come from factors to pass the bound: W**traces > bound**(r + traces)
    let window_bits: u64 = (((r + traces) * bit_length(r + 1)) / traces + 8) as u64;
    let mut window: Int = p_int.clone();
    while bits(&window) <= window_bits {
        window *= &p_int;
    }
    let mut k: usize = 1;
    let mut lifts: usize = 0;
    let mut sums: Vec<Vec<Int>> = Vec::new();
    loop {
        // the next traces whose digits above their bound are known
        let mut chosen: Vec<(usize, Int)> = Vec::new();
        while k < n && chosen.len() < traces {
            let mut low: Int = p_int.clone();
            let needed: Int = Int::from(2u8) * &n_int * root_bound.pow(k);
            while low < needed {
                low *= &p_int;
            }
            if &low * &window <= m {
                chosen.push((k, low));
            }
            k += 1;
        }
        if chosen.is_empty() {
            if k >= n && Int::from(2u8) * &n_int * &root_bound * &p_int * &window <= m {
                // every trace was used: a second pass with more digits
                window = &window * &window;
                k = 1;
                continue;
            }
            lifts += 1;
            if lifts > KNAPSACK_LIFTS {
                return Ok(subset_recombination(f, lifted, &m));
            }
            let squared: Int = &m * &m;
            let (next, next_m) = dup_zz_hensel_lift(p, f, modular, &squared)?;
            lifted = next;
            m = next_m;
            sums = Vec::new();
            k = 1;
            continue;
        }
        if sums.is_empty() {
            sums = lifted.iter().map(|g| power_sums(g, n - 1, &m)).collect();
        }
        let mut columns: Vec<Vec<Int>> = Vec::new();
        for (index, low) in &chosen {
            let modulus: Int = low * &window;
            let scale: Int = pow_mod(&lc, *index, &modulus);
            let mut column: Vec<Int> = Vec::new();
            for sum in sums.iter().take(r) {
                let mut s: Int = mod_floor(&(&scale * &sum[index - 1]), &modulus);
                if Int::from(2u8) * &s > modulus {
                    s -= &modulus;
                }
                let mut remainder: Int = mod_floor(&s, low);
                if Int::from(2u8) * &remainder > *low {
                    remainder -= low;
                }
                column.push(div_floor(&(&s - &remainder), low));
            }
            columns.push(column);
        }
        let mut rows: Vec<Vec<Int>> = Vec::new();
        for w in &basis {
            let mut row: Vec<Int> = w.clone();
            for column in &columns {
                let mut total: Int = Int::ZERO;
                for i in 0..r {
                    total += &w[i] * &column[i];
                }
                row.push(total);
            }
            rows.push(row);
        }
        for j in 0..columns.len() {
            let mut row: Vec<Int> = vec![Int::ZERO; r];
            row.extend((0..columns.len()).map(|i| if i == j { window.clone() } else { Int::ZERO }));
            rows.push(row);
        }
        // an irreducible factor gives a 0-1 vector v with, for each trace,
        // |sum v_i T_i - c W| <= (r + 1)/2
        let bound: Rat = rat_from_parts(Int::from(4 * r + columns.len() * (r + 1) * (r + 1)), Int::from(4u8));
        let short: Vec<Vec<Int>> = lll_reduction(&rows, &default_delta())?.short_part(&bound);
        let projected: Vec<Vec<Int>> = short.iter().map(|row| row[..r].to_vec()).collect();
        basis = hermite_normal_form(&projected)?;
        if basis.len() == 1 {
            return Ok(vec![f.to_vec()]);
        }
        let Some(partition) = partition_of(&basis) else {
            continue;
        };
        let mut factors: Vec<Vec<Int>> = Vec::new();
        let mut proved: bool = true;
        for subset in &partition {
            let chosen_factors: Vec<Vec<Int>> = subset.iter().map(|&i| lifted[i].clone()).collect();
            let candidate: Vec<Int> = candidate_factor(f, &chosen_factors, &m);
            if !divides_leading(f, &candidate) || zz_trial_division(f, &candidate).is_none() {
                proved = false;
                break;
            }
            factors.push(candidate);
        }
        if proved {
            factors.sort_by(by_degree_then_coefficients);
            return Ok(factors);
        }
    }
}

/// The irreducible factors (primitive, positive leading coefficient,
/// sorted) of a primitive square-free `f` with positive leading
/// coefficient, by van Hoeij's knapsack algorithm: the traces
/// `S_k(g) = l**k sum alpha**k` of the modular factors, cut to the digits
/// between `p**b > 2 n R**k` and `p**b W`, make the 0-1 vectors of the true
/// factors short vectors of a lattice; LLL reduction shrinks the lattice of
/// candidate vectors until its echelon form is a partition of the modular
/// factors, proved by trial division (M. van Hoeij, *Factoring polynomials
/// and the knapsack problem*, J. Number Theory 95 (2002), 167-189).
pub fn dup_zz_van_hoeij(f: &[Int]) -> PolyResult<Vec<Vec<Int>>> {
    if dup_degree(f) <= 1 {
        return Ok(vec![f.to_vec()]);
    }
    let (p, modular) = good_prime(f)?;
    if modular.len() == 1 {
        return Ok(vec![f.to_vec()]);
    }
    knapsack_recombination(f, p, &modular)
}

/// The quotient `f/g` over `ZZ` when `g` divides `f` exactly, `None`
/// otherwise (the division stops at the first coefficient which is not
/// divisible).
fn zz_trial_division(f: &[Int], g: &[Int]) -> Option<Vec<Int>> {
    if f.len() < g.len() {
        return None;
    }
    let (df, dg) = (f.len() - 1, g.len() - 1);
    let lc: &Int = &g[0];
    let mut remainder: Vec<Int> = f.to_vec();
    let mut quotient: Vec<Int> = vec![Int::ZERO; df - dg + 1];
    for i in 0..df - dg + 1 {
        let (q, r) = div_rem_trunc(&remainder[i], lc);
        if !r.is_zero() {
            return None;
        }
        if !q.is_zero() {
            for j in 1..dg + 1 {
                let product: Int = &q * &g[j];
                remainder[i + j] -= product;
            }
        }
        quotient[i] = q;
    }
    if remainder[df - dg + 1..].iter().any(|c| !c.is_zero()) {
        return None;
    }
    Some(quotient)
}

/// Whether a trial division of `f` by `g` can succeed: the degrees and the
/// leading and trailing coefficients allow it.
fn divides_leading(f: &[Int], g: &[Int]) -> bool {
    if g.is_empty() || g.len() > f.len() {
        return false;
    }
    if !div_rem_trunc(&f[0], &g[0]).1.is_zero() {
        return false;
    }
    let (tf, tg) = (&f[f.len() - 1], &g[g.len() - 1]);
    if !tg.is_zero() && !div_rem_trunc(tf, tg).1.is_zero() {
        return false;
    }
    true
}

/// The content (with the sign) and the irreducible factors of a
/// square-free `f`.
pub fn dup_zz_factor_sqf(f: &[Int], k: &IntegerRing) -> PolyResult<(Int, Vec<Vec<Int>>)> {
    let (content, part) = dup_decompose_content(f, k);
    if dup_degree(&part) <= 0 {
        return Ok((content, Vec::new()));
    }
    Ok((content, dup_zz_zassenhaus(&part)?))
}

/// The content (with the sign of the leading coefficient) and the
/// irreducible factors with their multiplicities, primitive with positive
/// leading coefficients, sorted by degree then lexicographically.
pub fn dup_zz_factor(f: &[Int], k: &IntegerRing) -> PolyResult<(Int, Vec<Factor<Int>>)> {
    if f.is_empty() {
        return Ok((Int::ZERO, Vec::new()));
    }
    let (mut content, sqf) = dup_sqf_list(f, k);
    let mut factors: Vec<Factor<Int>> = Vec::new();
    for (g, multiplicity) in &sqf {
        // a power of x comes out of the trailing zeros
        let mut g: Vec<Int> = g.clone();
        let mut zeros: usize = 0;
        while g.last().is_some_and(|c| c.is_zero()) {
            g.pop();
            zeros += 1;
        }
        if zeros > 0 {
            factors.push((vec![Int::ONE, Int::ZERO], multiplicity * zeros));
        }
        let (unit, irreducibles) = dup_zz_factor_sqf(&g, k)?;
        if is_negative(&unit) {
            content = -content;
        }
        for h in irreducibles {
            factors.push((h, *multiplicity));
        }
    }
    factors.sort_by(|a, b| (a.0.len(), &a.0, a.1).cmp(&(b.0.len(), &b.0, b.1)));
    Ok((content, factors))
}

/// The constant and the irreducible factors (primitive with integer
/// coefficients, positive leading coefficients) with their multiplicities.
pub fn dup_qq_factor(f: &[Rat], k: &RationalField) -> PolyResult<(Rat, Vec<Factor<Rat>>)> {
    if f.is_empty() {
        return Ok((Rat::ZERO, Vec::new()));
    }
    let (common, cleared) = dup_clear_denoms(f, k);
    let integers: Vec<Int> = cleared.iter().map(numer).collect();
    let (content, factors) = dup_zz_factor(&integers, &ZZ)?;
    let rational: Vec<Factor<Rat>> = factors
        .into_iter()
        .map(|(g, m)| (g.iter().map(rat_from_int).collect(), m))
        .collect();
    Ok((rat_from_parts(content, common), rational))
}

/// Whether `f` is irreducible over `ZZ` (a primitive polynomial of positive
/// degree with one factor of multiplicity one).
pub fn dup_zz_irreducible_p(f: &[Int], k: &IntegerRing) -> PolyResult<bool> {
    if dup_degree(f) <= 0 {
        return Ok(false);
    }
    let (content, factors) = dup_zz_factor(f, k)?;
    Ok(abs(&content).is_one() && factors.len() == 1 && factors[0].1 == 1)
}

// -- number fields: Trager's algorithm

/// The norm `N(f)` of a polynomial over `QQ(alpha)`: the resultant in
/// `alpha` of the minimal polynomial and `f`, a polynomial in `x` with
/// rational coefficients (the product of the conjugates of `f`).
pub fn dup_ext_norm(f: &[AlgebraicElement], k: &AlgebraicField) -> Vec<Rat> {
    let r: PolynomialRing<RationalField> = PolynomialRing::new(QQ, "x");
    // f as a polynomial in alpha with coefficients in QQ[x]
    let n: usize = dup_degree(f).max(0) as usize;
    let mut by_alpha: Vec<Vec<Rat>> = vec![Vec::new(); k.degree];
    for (i, c) in f.iter().enumerate() {
        for (j, a) in c.iter().enumerate() {
            // the coefficient a of alpha**(len(c) - 1 - j) x**(n - i)
            let power: usize = c.len() - 1 - j;
            let slot: usize = k.degree - 1 - power;
            by_alpha[slot] = dup_add_term(&by_alpha[slot], a, n - i, &QQ);
        }
    }
    let m_x: Vec<Vec<Rat>> = k
        .minpoly
        .iter()
        .map(|c| if c.is_zero() { Vec::new() } else { vec![c.clone()] })
        .collect();
    dup_resultant(&m_x, &dup_strip(&by_alpha, &r), &r)
}

/// `(s, g, N)` with `g = f(x - s alpha)` and `N = N(g)` square-free (the
/// shifts `s = 0, 1, -1, 2, ...` are tried in turn; finitely many fail for
/// a square-free `f`).
pub fn dup_ext_sqf_norm(f: &[AlgebraicElement], k: &AlgebraicField) -> (i64, Vec<AlgebraicElement>, Vec<Rat>) {
    let mut s: i64 = 0;
    loop {
        let shift: AlgebraicElement = k.mul(&k.from_int(-s), &k.alpha());
        let g: Vec<AlgebraicElement> = if s != 0 { dup_shift(f, &shift, k) } else { f.to_vec() };
        let norm: Vec<Rat> = dup_ext_norm(&g, k);
        if dup_sqf_p(&norm, &QQ) {
            return (s, g, norm);
        }
        s = if s > 0 { -s } else { 1 - s };
    }
}

/// The leading coefficient and the monic irreducible factors over
/// `QQ(alpha)` with their multiplicities, sorted by degree and
/// multiplicity.
#[allow(clippy::type_complexity)]
pub fn dup_ext_factor(
    f: &[AlgebraicElement],
    k: &AlgebraicField,
) -> PolyResult<(AlgebraicElement, Vec<Factor<AlgebraicElement>>)> {
    if f.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let lc: AlgebraicElement = dup_LC(f, k);
    let monic: Vec<AlgebraicElement> = dup_monic(f, k)?;
    let (_, sqf) = dup_sqf_list(&monic, k);
    let mut factors: Vec<Factor<AlgebraicElement>> = Vec::new();
    for (g, multiplicity) in sqf {
        if dup_degree(&g) == 1 {
            factors.push((g, multiplicity));
            continue;
        }
        let (s, shifted, norm) = dup_ext_sqf_norm(&g, k);
        let (_, norm_factors) = dup_qq_factor(&norm, &QQ)?;
        let back: AlgebraicElement = k.mul(&k.from_int(s), &k.alpha());
        for (h, _) in norm_factors {
            let over_field: Vec<AlgebraicElement> = h.iter().map(|c| k.from_rational(c)).collect();
            let factor: Vec<AlgebraicElement> = dup_gcd(&shifted, &over_field, k);
            if dup_degree(&factor) > 0 {
                let unshifted: Vec<AlgebraicElement> = if s != 0 { dup_shift(&factor, &back, k) } else { factor };
                factors.push((unshifted, multiplicity));
            }
        }
    }
    factors.sort_by_key(|(g, m)| (g.len(), *m));
    Ok((lc, factors))
}

#[cfg(test)]
mod tests {
    use super::super::densebasic::testing::{q, zz};
    use super::*;
    use crate::polys::domains::QQ;

    #[test]
    fn doctests() {
        assert_eq!(
            dup_zz_factor(&zz(&[2, 0, -8]), &ZZ).unwrap(),
            (int(2), vec![(zz(&[1, -2]), 1), (zz(&[1, 2]), 1)])
        );
        assert_eq!(
            dup_zz_factor(&zz(&[1, 0, 0, 0, 1]), &ZZ).unwrap(),
            (int(1), vec![(zz(&[1, 0, 0, 0, 1]), 1)])
        );
        assert_eq!(
            dup_zz_factor(&zz(&[-1, 0, 2, 0, -1]), &ZZ).unwrap(),
            (int(-1), vec![(zz(&[1, -1]), 2), (zz(&[1, 1]), 2)])
        );
        let (c, factors) = dup_qq_factor(&[q(1, 2), q(0, 1), q(-2, 1)], &QQ).unwrap();
        assert_eq!(c, q(1, 2));
        assert_eq!(factors, vec![(vec![q(1, 1), q(-2, 1)], 1), (vec![q(1, 1), q(2, 1)], 1)]);
    }

    #[test]
    fn combinations_follow_itertools() {
        assert_eq!(
            combinations(4, 2),
            vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3], vec![2, 3]]
        );
        assert_eq!(combinations(3, 3), vec![vec![0, 1, 2]]);
        assert_eq!(combinations(2, 3), Vec::<Vec<usize>>::new());
        assert_eq!(combinations(3, 1), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn swinnerton_dyer_like_product() {
        // (x**4 - 10 x**2 + 1) (x**2 - 3) (2 x + 5): many modular factors
        let f: Vec<Int> = dup_mul(
            &dup_mul(&zz(&[1, 0, -10, 0, 1]), &zz(&[1, 0, -3]), &ZZ),
            &zz(&[2, 5]),
            &ZZ,
        );
        let (c, factors) = dup_zz_factor(&f, &ZZ).unwrap();
        assert_eq!(c, int(1));
        assert_eq!(
            factors,
            vec![(zz(&[2, 5]), 1), (zz(&[1, 0, -3]), 1), (zz(&[1, 0, -10, 0, 1]), 1)]
        );
    }
}
