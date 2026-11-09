//! Port of `sympy_extras/polys/modulargroebner.py`: reduced Gröbner bases
//! over `QQ` (and `ZZ`) by the modular algorithm of Arnold, proven.
//!
//! The input is made integral and primitive and, unless it is homogeneous,
//! homogenised; the order is then the homogeneous order (total degree
//! first, then the order of the ring on the original variables). For
//! every prime dividing no leading coefficient the reduced Gröbner basis
//! modulo `p` is computed (by [`f4`] over [`BigFiniteField`]), the images
//! of unlucky primes are discarded by comparing their leading monomials
//! degree by degree ([`Luckiness`]), the images of the kept primes are
//! combined by the Chinese remainder theorem and the rational coefficients
//! recovered by Wang's [`rational_reconstruction`]. A candidate is
//! verified over `ZZ`, fraction-free: every input polynomial reduces to
//! zero and the S-polynomials of the critical pairs left by the criteria
//! of Gebauer and Möller reduce to zero. For homogeneous input this proves
//! the result (Arnold, theorem 7.1); setting the homogenising variable to
//! one and interreducing gives the reduced basis of the input. See the
//! Python module for the full discussion.
//!
//! The Python module also offers `direct_groebner` and `groebner`, which
//! choose between SymPy's Buchberger algorithm, F4 and the modular
//! algorithm by time limits; they are not ported ([`f4`] is the direct
//! computation here).
//!
//! References
//! ==========
//!
//! 1. E. A. Arnold, *Modular algorithms for computing Gröbner bases*, J.
//!    Symbolic Computation 35 (2003) 403-419.
//! 2. F. Pauer, *On lucky ideals for Gröbner basis computations*, J.
//!    Symbolic Computation 14 (1992) 471-482.
//! 3. P. S. Wang, M. J. T. Guy, J. H. Davenport, *P-adic reconstruction of
//!    rational numbers*, ACM SIGSAM Bulletin 16 (1982) 2-3.
//! 4. M. Monagan, *Maximal quotient rational reconstruction*, Proc. ISSAC
//!    2004, 243-249.
//! 5. R. Gebauer, H. M. Möller, *On an installation of Buchberger's
//!    algorithm*, J. Symbolic Computation 6 (1988) 275-286.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::error::{value_error, PolyResult};
use crate::ntheory::generate::prevprime;
use crate::numbers::*;

use super::domains::{BigFiniteField, IntegerRing, RationalField};
use super::f4::f4;
use super::rings::{monomial_divides, Monomial, MonomialOrder, PolyElement, PolyRing, Terms};

/// A polynomial with integer coefficients, by exponent vectors.
pub type IntPoly = BTreeMap<Monomial, Int>;
/// A polynomial with rational coefficients `(numerator, denominator)`.
pub type RatPoly = BTreeMap<Monomial, (Int, Int)>;
/// A polynomial with its leading monomial.
type Led = (Monomial, IntPoly);

/// A candidate whose coefficients are all within `sqrt(M/2**(1 +
/// GUARD_BITS))` is verified without a confirming prime.
pub const GUARD_BITS: usize = 32;

// ----------------------------------------------------------------------
// integers

/// The fraction `n/d` congruent to `a` modulo `m` with `|n| <= bound`,
/// `0 < d <= bound` and `gcd(n, d) = 1`, or `None` (Wang's algorithm). The
/// default bound is the largest for which the answer is unique,
/// `isqrt((m - 1)//2)`; `Value` for `m <= 1` or a bound beyond it.
pub fn rational_reconstruction(a: &Int, m: &Int, bound: Option<&Int>) -> PolyResult<Option<(Int, Int)>> {
    if *m <= Int::ONE {
        return Err(value_error("the modulus must be greater than 1".to_string()));
    }
    let limit: Int = isqrt(&((m - Int::ONE) / Int::from(2u8)));
    let bound: Int = match bound {
        None => limit,
        Some(b) => {
            if is_negative(b) || *b > limit {
                return Err(value_error("the bound must satisfy 2*bound**2 < m".to_string()));
            }
            b.clone()
        }
    };
    Ok(reconstruct_within(a, m, &bound))
}

fn reconstruct_within(a: &Int, m: &Int, bound: &Int) -> Option<(Int, Int)> {
    let (mut r0, mut t0): (Int, Int) = (m.clone(), Int::ZERO);
    let (mut r1, mut t1): (Int, Int) = (mod_floor(a, m), Int::ONE);
    while r1 > *bound {
        let q: Int = div_floor(&r0, &r1);
        let r2: Int = &r0 - &q * &r1;
        r0 = std::mem::replace(&mut r1, r2);
        let t2: Int = &t0 - &q * &t1;
        t0 = std::mem::replace(&mut t1, t2);
    }
    if is_negative(&t1) {
        r1 = -r1;
        t1 = -t1;
    }
    if t1.is_zero() || t1 > *bound || !gcd(&r1, &t1).is_one() {
        return None;
    }
    Some((r1, t1))
}

/// The integer in `0..m*p` congruent to `a` modulo `m` and to `b` modulo
/// `p`, for coprime `m` and `p` (`Value` otherwise).
pub fn chinese_remainder(a: &Int, m: &Int, b: &Int, p: &Int) -> PolyResult<Int> {
    let inverse: Int = mod_inverse(m, p).ok_or_else(|| value_error(format!("{} and {} are not coprime", m, p)))?;
    Ok(lift(a, m, b, p, &inverse))
}

fn lift(a: &Int, m: &Int, b: &Int, p: &Int, m_inverse: &Int) -> Int {
    a + m * mod_floor(&(mod_floor(&(b - a), p) * m_inverse), p)
}

/// The primes below `2**1024` found so far, decreasing.
static LARGE_PRIMES: Mutex<Vec<Int>> = Mutex::new(Vec::new());

/// The primes used by [`modular_groebner`] by default: the primes below
/// `2**1024` in decreasing order, starting at `2**1024 - 105`.
pub struct DefaultPrimes {
    index: usize,
}

pub fn default_primes() -> DefaultPrimes {
    DefaultPrimes { index: 0 }
}

impl Iterator for DefaultPrimes {
    type Item = Int;

    fn next(&mut self) -> Option<Int> {
        let mut primes = LARGE_PRIMES.lock().expect("the prime cache");
        if primes.is_empty() {
            primes.push((Int::ONE << 1024) - Int::from(105u8));
        }
        while primes.len() <= self.index {
            let next: Int = prevprime(primes.last().expect("a prime")).expect("primes below 2**1024");
            primes.push(next);
        }
        self.index += 1;
        Some(primes[self.index - 1].clone())
    }
}

// ----------------------------------------------------------------------
// polynomials with integer coefficients

fn degree(m: &[u32]) -> u64 {
    m.iter().map(|e| *e as u64).sum()
}

fn leading_monomial(f: &IntPoly, order: &MonomialOrder) -> Monomial {
    f.keys()
        .max_by(|a, b| order.cmp(a, b))
        .expect("a nonzero polynomial")
        .clone()
}

/// `f` divided by its content, with a positive coefficient at `leading` if
/// given.
fn primitive(f: IntPoly, leading: Option<&Monomial>) -> IntPoly {
    let mut content: Int = Int::ZERO;
    for c in f.values() {
        content = gcd(&content, c);
        if content.is_one() {
            break;
        }
    }
    if let Some(lm) = leading {
        if is_negative(&f[lm]) {
            content = if content.is_zero() { -Int::ONE } else { -content };
        }
    }
    if content.is_zero() || content.is_one() {
        return f;
    }
    f.into_iter().map(|(m, c)| (m, c / &content)).collect()
}

fn scale(f: &mut IntPoly, u: &Int) {
    for c in f.values_mut() {
        *c *= u;
    }
}

fn divide(f: &mut IntPoly, content: &Int) {
    for c in f.values_mut() {
        *c = &*c / content;
    }
}

/// A non-zero integer multiple of the normal form of `f` modulo `basis`,
/// computed without fractions. With `full = false` the reduction stops at
/// the first leading monomial which cannot be reduced: the result is zero
/// if and only if the normal form is.
fn normal_form(f: &IntPoly, basis: &[Led], order: &MonomialOrder, full: bool) -> IntPoly {
    let mut f: IntPoly = f.clone();
    let mut rest: IntPoly = IntPoly::new();
    while !f.is_empty() {
        let m: Monomial = leading_monomial(&f, order);
        let c: Int = f[&m].clone();
        let Some((lm, g)) = basis.iter().find(|(lm, _)| monomial_divides(lm, &m)) else {
            if !full {
                return f;
            }
            f.remove(&m);
            rest.insert(m, c);
            continue;
        };
        let a: &Int = &g[lm];
        let d: Int = gcd(a, &c);
        let (u, v): (Int, Int) = (a / &d, &c / &d);
        if !u.is_one() {
            scale(&mut f, &u);
            scale(&mut rest, &u);
        }
        let shift: Vec<u32> = m.iter().zip(lm).map(|(i, j)| i - j).collect();
        for (mg, cg) in g {
            let mm: Monomial = mg.iter().zip(&shift).map(|(i, j)| i + j).collect();
            let value: Int = f.get(&mm).cloned().unwrap_or(Int::ZERO) - &v * cg;
            if value.is_zero() {
                f.remove(&mm);
            } else {
                f.insert(mm, value);
            }
        }
        if !u.is_one() {
            let mut content: Int = Int::ZERO;
            for value in f.values().chain(rest.values()) {
                content = gcd(&content, value);
                if content.is_one() {
                    break;
                }
            }
            if content > Int::ONE {
                divide(&mut f, &content);
                divide(&mut rest, &content);
            }
        }
    }
    rest
}

fn s_polynomial(lf: &Monomial, f: &IntPoly, lg: &Monomial, g: &IntPoly) -> IntPoly {
    let d: Int = gcd(&f[lf], &g[lg]);
    let (a, b): (Int, Int) = (&f[lf] / &d, &g[lg] / &d);
    let sf: Vec<u32> = lf.iter().zip(lg).map(|(i, j)| i.max(j) - i).collect();
    let sg: Vec<u32> = lf.iter().zip(lg).map(|(i, j)| i.max(j) - j).collect();
    let shifted = |m: &Monomial, s: &[u32]| -> Monomial { m.iter().zip(s).map(|(i, j)| i + j).collect() };
    let mut s: IntPoly = f.iter().map(|(m, c)| (shifted(m, &sf), &b * c)).collect();
    for (m, c) in g {
        let mm: Monomial = shifted(m, &sg);
        let value: Int = s.get(&mm).cloned().unwrap_or(Int::ZERO) - &a * c;
        if value.is_zero() {
            s.remove(&mm);
        } else {
            s.insert(mm, value);
        }
    }
    s
}

fn lcm_monomial(a: &[u32], b: &[u32]) -> Monomial {
    a.iter().zip(b).map(|(i, j)| *i.max(j)).collect()
}

/// The pairs whose S-polynomials must reduce to zero for polynomials with
/// these leading monomials to be a Gröbner basis: those left by the
/// criteria of Gebauer and Möller (B, M, F and the coprime criterion) when
/// the polynomials are inserted one after the other.
fn critical_pairs(leading: &[Monomial]) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize, Monomial)> = Vec::new();
    for (k, h) in leading.iter().enumerate() {
        let lcms: Vec<Monomial> = (0..k).map(|i| lcm_monomial(&leading[i], h)).collect();
        pairs.retain(|(i, j, l)| !(monomial_divides(h, l) && lcms[*i] != *l && lcms[*j] != *l));
        // the lcms in the order of their first pair, as a Python dict
        let mut by_lcm: Vec<(Monomial, Vec<usize>)> = Vec::new();
        for (i, l) in lcms.iter().enumerate() {
            match by_lcm.iter_mut().find(|(other, _)| other == l) {
                Some((_, indices)) => indices.push(i),
                None => by_lcm.push((l.clone(), vec![i])),
            }
        }
        for (l, indices) in &by_lcm {
            if by_lcm.iter().any(|(other, _)| other != l && monomial_divides(other, l)) {
                continue;
            }
            let coprime = |i: &usize| leading[*i].iter().zip(h).all(|(a, b)| *a == 0 || *b == 0);
            if indices.iter().any(coprime) {
                continue;
            }
            pairs.push((indices[0], k, l.clone()));
        }
    }
    pairs.into_iter().map(|(i, j, _)| (i, j)).collect()
}

/// Whether the polynomials (with their leading monomials) are a Gröbner
/// basis of the ideal they generate.
fn is_groebner(basis: &[Led], order: &MonomialOrder) -> bool {
    let leading: Vec<Monomial> = basis.iter().map(|(lm, _)| lm.clone()).collect();
    for (i, j) in critical_pairs(&leading) {
        let s: IntPoly = s_polynomial(&basis[i].0, &basis[i].1, &basis[j].0, &basis[j].1);
        if !normal_form(&s, basis, order, false).is_empty() {
            return false;
        }
    }
    true
}

/// The reduced Gröbner basis (up to integer factors, sorted by decreasing
/// leading monomials) of the ideal of which `basis` is a Gröbner basis.
fn reduced(basis: &[IntPoly], order: &MonomialOrder) -> Vec<Led> {
    let mut with_leading: Vec<Led> = basis.iter().map(|g| (leading_monomial(g, order), g.clone())).collect();
    with_leading.sort_by(|a, b| order.cmp(&a.0, &b.0));
    let mut minimal: Vec<Led> = Vec::new();
    for (lm, g) in with_leading {
        if !minimal.iter().any(|(other, _)| monomial_divides(other, &lm)) {
            minimal.push((lm, g));
        }
    }
    let mut result: Vec<Led> = Vec::new();
    for i in 0..minimal.len() {
        let others: Vec<Led> = minimal
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, pair)| pair.clone())
            .collect();
        let (lm, g) = &minimal[i];
        result.push((lm.clone(), primitive(normal_form(g, &others, order, true), Some(lm))));
    }
    result.reverse();
    result
}

// ----------------------------------------------------------------------
// the images and their lifting

/// The sort key of the leading monomials of an image: for every degree
/// from zero up, the leading monomials of that degree, decreasing (Python:
/// their number and their order keys). Images of a homogeneous ideal with
/// different leading monomials have different keys, and the smaller key
/// comes from an unlucky prime. Compared by [`compare_luckiness`].
pub type Luckiness = Vec<Vec<Monomial>>;

fn luckiness(leading: &[Monomial], order: &MonomialOrder) -> Luckiness {
    if leading.is_empty() {
        return Vec::new();
    }
    let top: usize = leading.iter().map(|m| degree(m)).max().unwrap_or(0) as usize;
    let mut by_degree: Luckiness = vec![Vec::new(); top + 1];
    let mut sorted: Vec<Monomial> = leading.to_vec();
    sorted.sort_by(|a, b| order.cmp(b, a));
    for m in sorted {
        by_degree[degree(&m) as usize].push(m);
    }
    by_degree
}

/// The comparison of the Python keys `((len(keys), keys), ...)`.
pub fn compare_luckiness(a: &Luckiness, b: &Luckiness, order: &MonomialOrder) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let by_count: Ordering = x.len().cmp(&y.len());
        if by_count != Ordering::Equal {
            return by_count;
        }
        for (m, n) in x.iter().zip(y) {
            let by_key: Ordering = order.cmp(m, n);
            if by_key != Ordering::Equal {
                return by_key;
            }
        }
    }
    a.len().cmp(&b.len())
}

/// What [`modular_groebner`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModularTrace {
    /// Whether the input was homogenised.
    pub homogenized: bool,
    /// The primes whose images were computed, in order.
    pub used: Vec<Int>,
    /// The primes which divide a leading coefficient of the input.
    pub skipped: Vec<Int>,
    /// The primes whose images were discarded as unlucky.
    pub unlucky: Vec<Int>,
    /// The primes whose images gave the result.
    pub lucky: Vec<Int>,
    /// The number of stable candidates which were not the Gröbner basis.
    pub failed_verifications: usize,
}

/// The reduced Gröbner basis of the polynomials modulo `p`, by increasing
/// leading monomials, with coefficients in `0..p`.
fn image(polys: &[IntPoly], names: &[String], order: &MonomialOrder, p: &Int) -> PolyResult<Vec<Led>> {
    let ring: Arc<PolyRing<BigFiniteField>> =
        PolyRing::new(BigFiniteField::new(p.clone())?, names.to_vec(), order.clone())?;
    let reduced: Vec<PolyElement<BigFiniteField>> = polys
        .iter()
        .map(|f| ring.from_dict(f.iter().map(|(m, c)| (m.clone(), mod_floor(c, p))).collect()))
        .filter(|f| !f.is_zero())
        .collect();
    let mut result: Vec<Led> = f4(&reduced, &ring)?.into_iter().map(|g| (g.LM(), g.terms)).collect();
    result.sort_by(|a, b| order.cmp(&a.0, &b.0));
    Ok(result)
}

/// The images of the kept primes, combined modulo their product.
struct Lifting {
    modulus: Int,
    primes: Vec<Int>,
    luckiness: Luckiness,
    leading: Vec<Monomial>,
    residues: Vec<IntPoly>,
    /// The coefficient on which the last reconstruction failed, tried first
    /// by the next.
    hardest: Option<(usize, Monomial)>,
}

impl Lifting {
    fn new() -> Lifting {
        Lifting {
            modulus: Int::ONE,
            primes: Vec::new(),
            luckiness: Vec::new(),
            leading: Vec::new(),
            residues: Vec::new(),
            hardest: None,
        }
    }

    fn reset(&mut self, luckiness: Luckiness) {
        *self = Lifting::new();
        self.luckiness = luckiness;
    }

    fn add(&mut self, image: &[Led], p: &Int) {
        if self.primes.is_empty() {
            self.leading = image.iter().map(|(lm, _)| lm.clone()).collect();
            self.residues = image.iter().map(|(_, g)| g.clone()).collect();
        } else {
            let m: &Int = &self.modulus;
            let inverse: Int = mod_inverse(m, p).expect("distinct primes");
            for (residues, (_, g)) in self.residues.iter_mut().zip(image) {
                let mut monomials: Vec<Monomial> = residues.keys().cloned().collect();
                monomials.extend(g.keys().filter(|k| !residues.contains_key(*k)).cloned());
                for monomial in monomials {
                    let a: Int = residues.get(&monomial).cloned().unwrap_or(Int::ZERO);
                    let b: Int = g.get(&monomial).cloned().unwrap_or(Int::ZERO);
                    residues.insert(monomial, lift(&a, m, &b, p, &inverse));
                }
            }
        }
        self.modulus *= p;
        self.primes.push(p.clone());
    }

    /// The polynomials with rational coefficients having these residues,
    /// and whether all the coefficients are within the guarded bound;
    /// `None` if a coefficient is not a fraction within Wang's bound.
    fn reconstruct(&mut self) -> Option<(Vec<RatPoly>, bool)> {
        let m: Int = self.modulus.clone();
        let bound: Int = isqrt(&((&m - Int::ONE) / Int::from(2u8)));
        if let Some((index, monomial)) = &self.hardest {
            reconstruct_within(&self.residues[*index][monomial], &m, &bound)?;
        }
        let guard: Int = isqrt(&((&m - Int::ONE) >> (1 + GUARD_BITS)));
        let mut guarded: bool = true;
        let mut candidate: Vec<RatPoly> = Vec::new();
        for (index, residues) in self.residues.iter().enumerate() {
            let mut g: RatPoly = RatPoly::new();
            // the coefficients share their denominators: a residue whose
            // product by the denominators found so far is within the bound
            // is that product over them; Euclid's algorithm runs on the
            // others only
            let mut common: Int = Int::ONE;
            for (monomial, residue) in residues {
                let mut fraction: Option<(Int, Int)> = None;
                if common <= bound {
                    let mut n: Int = mod_floor(&(residue * &common), &m);
                    if n > bound {
                        n -= &m;
                    }
                    if -&bound <= n {
                        let d: Int = gcd(&n, &common);
                        fraction = Some((&n / &d, &common / &d));
                    }
                }
                let fraction: (Int, Int) = match fraction {
                    Some(found) => found,
                    None => match reconstruct_within(residue, &m, &bound) {
                        Some(found) => {
                            common = lcm(&common, &found.1);
                            found
                        }
                        None => {
                            self.hardest = Some((index, monomial.clone()));
                            return None;
                        }
                    },
                };
                if !fraction.0.is_zero() {
                    if abs(&fraction.0) > guard || fraction.1 > guard {
                        guarded = false;
                    }
                    g.insert(monomial.clone(), fraction);
                }
            }
            candidate.push(g);
        }
        Some((candidate, guarded))
    }
}

/// Whether the candidate reduces to the image modulo `p`.
fn agrees(candidate: &[RatPoly], image: &[Led], p: &Int) -> bool {
    for (g, (_, gp)) in candidate.iter().zip(image) {
        let mut count: usize = 0;
        for (monomial, (n, d)) in g {
            let Some(inverse) = mod_inverse(d, p) else {
                return false;
            };
            let value: Int = mod_floor(&(n * inverse), p);
            if value != gp.get(monomial).cloned().unwrap_or(Int::ZERO) {
                return false;
            }
            if !value.is_zero() {
                count += 1;
            }
        }
        if count != gp.len() {
            return false;
        }
    }
    true
}

fn clear_denominators(g: &RatPoly) -> IntPoly {
    let mut common: Int = Int::ONE;
    for (_, d) in g.values() {
        common = lcm(&common, d);
    }
    g.iter().map(|(m, (n, d))| (m.clone(), n * (&common / d))).collect()
}

/// The candidate with integer coefficients if it is a Gröbner basis of an
/// ideal which contains `polys`.
fn verified(
    candidate: &[RatPoly],
    leading: &[Monomial],
    polys: &[IntPoly],
    order: &MonomialOrder,
) -> Option<Vec<IntPoly>> {
    let cleared: Vec<IntPoly> = candidate.iter().map(clear_denominators).collect();
    // the reductions prefer the divisors with few terms
    let mut by_size: Vec<Led> = leading.iter().cloned().zip(cleared.iter().cloned()).collect();
    by_size.sort_by_key(|(_, g)| g.len());
    if polys.iter().any(|f| !normal_form(f, &by_size, order, false).is_empty()) {
        return None;
    }
    if !is_groebner(&by_size, order) {
        return None;
    }
    Some(cleared)
}

// ----------------------------------------------------------------------
// the interface

/// The primitive integer multiple of a polynomial over `QQ` (Python's
/// `_integer_polys` on one polynomial).
fn integer_poly(f: &PolyElement<RationalField>) -> IntPoly {
    let mut common: Int = Int::ONE;
    for c in f.terms.values() {
        common = lcm(&common, &denom(c));
    }
    let g: IntPoly = f
        .terms
        .iter()
        .map(|(m, c)| (m.clone(), numer(c) * (&common / denom(c))))
        .collect();
    primitive(g, None)
}

/// The modular computation on primitive integer polynomials in `nvars`
/// variables (`names`) for `base`: the reduced basis up to integer factors
/// (primitive, positive leading coefficients), by decreasing leading
/// monomials.
fn modular_basis(
    integer: Vec<IntPoly>,
    names: &[String],
    base: &MonomialOrder,
    primes: &mut dyn Iterator<Item = Int>,
    trace: &mut ModularTrace,
) -> PolyResult<Vec<Led>> {
    let nvars: usize = names.len();
    let homogenized: bool = integer.iter().any(|f| {
        let first: u64 = degree(f.keys().next().expect("nonzero"));
        f.keys().any(|m| degree(m) != first)
    });
    trace.homogenized = homogenized;
    let mut symbols: Vec<String> = names.to_vec();
    let mut integer: Vec<IntPoly> = integer;
    if homogenized {
        let mut t: String = "t".to_string();
        while symbols.contains(&t) {
            t.insert(0, '_');
        }
        symbols.push(t);
        integer = integer
            .into_iter()
            .map(|f| {
                let top: u64 = f.keys().map(|m| degree(m)).max().expect("nonzero");
                f.into_iter()
                    .map(|(mut m, c)| {
                        let missing: u32 = (top - degree(&m)) as u32;
                        m.push(missing);
                        (m, c)
                    })
                    .collect()
            })
            .collect();
    }
    let order: MonomialOrder = MonomialOrder::Homogeneous {
        base: Box::new(base.clone()),
        nvars,
    };
    let leading_coefficients: Vec<Int> = integer
        .iter()
        .map(|f| f[&leading_monomial(f, &order)].clone())
        .collect();

    let mut lifting: Lifting = Lifting::new();
    let mut candidate: Option<Vec<RatPoly>> = None;
    let mut rejected: Option<Vec<RatPoly>> = None;
    let mut proven: Option<Vec<IntPoly>> = None;
    for p in primes {
        if leading_coefficients.iter().any(|c| mod_floor(c, &p).is_zero()) {
            trace.skipped.push(p);
            continue;
        }
        let img: Vec<Led> = image(&integer, &symbols, &order, &p)?;
        trace.used.push(p.clone());
        let leading: Vec<Monomial> = img.iter().map(|(lm, _)| lm.clone()).collect();
        let key: Luckiness = luckiness(&leading, &order);
        let comparison: Ordering = compare_luckiness(&key, &lifting.luckiness, &order);
        if !lifting.primes.is_empty() && comparison == Ordering::Less {
            trace.unlucky.push(p);
            continue;
        }
        if !lifting.primes.is_empty() && comparison == Ordering::Greater {
            trace.unlucky.extend(lifting.primes.iter().cloned());
            candidate = None;
        }
        if lifting.primes.is_empty() || comparison == Ordering::Greater {
            lifting.reset(key);
        }
        if let Some(c) = &candidate {
            if Some(c) != rejected.as_ref() && agrees(c, &img, &p) {
                proven = verified(c, &lifting.leading, &integer, &order);
                if proven.is_some() {
                    lifting.primes.push(p);
                    break;
                }
                rejected = candidate.clone();
                trace.failed_verifications += 1;
            }
        }
        lifting.add(&img, &p);
        candidate = None;
        if let Some((found, guarded)) = lifting.reconstruct() {
            if guarded && Some(&found) != rejected.as_ref() {
                proven = verified(&found, &lifting.leading, &integer, &order);
                if proven.is_some() {
                    break;
                }
                rejected = Some(found.clone());
                trace.failed_verifications += 1;
            }
            candidate = Some(found);
        }
    }
    let Some(mut proven) = proven else {
        return Err(value_error(
            "the primes ran out before the Gröbner basis was found".to_string(),
        ));
    };
    trace.lucky = lifting.primes.clone();
    if homogenized {
        proven = proven
            .into_iter()
            .map(|g| g.into_iter().map(|(m, c)| (m[..nvars].to_vec(), c)).collect())
            .collect();
    }
    Ok(reduced(&proven, base))
}

/// The reduced Gröbner basis of the ideal generated by `polys` in `ring`
/// (over `QQ`), computed modulo `primes` ([`default_primes`] if `None`)
/// and proven: monic, sorted by decreasing leading monomials, the basis of
/// [`f4`]. `Value` when the primes run out.
pub fn modular_groebner(
    polys: &[PolyElement<RationalField>],
    ring: &Arc<PolyRing<RationalField>>,
    primes: Option<&mut dyn Iterator<Item = Int>>,
    trace: Option<&mut ModularTrace>,
) -> PolyResult<Vec<PolyElement<RationalField>>> {
    let integer: Vec<IntPoly> = polys.iter().filter(|f| !f.is_zero()).map(integer_poly).collect();
    if integer.is_empty() {
        return Ok(Vec::new());
    }
    let mut own_trace: ModularTrace = ModularTrace::default();
    let trace: &mut ModularTrace = trace.unwrap_or(&mut own_trace);
    let mut defaults: DefaultPrimes = default_primes();
    let primes: &mut dyn Iterator<Item = Int> = match primes {
        Some(given) => given,
        None => &mut defaults,
    };
    let basis: Vec<Led> = modular_basis(integer, &ring.names, &ring.order, primes, trace)?;
    Ok(basis
        .into_iter()
        .map(|(lm, g)| {
            let lc: Int = g[&lm].clone();
            let terms: Terms<Rat> = g.into_iter().map(|(m, c)| (m, rat_from_parts(c, lc.clone()))).collect();
            ring.from_dict(terms)
        })
        .collect())
}

/// [`modular_groebner`] over `ZZ`: the basis is primitive with positive
/// leading coefficients, as SymPy's.
pub fn modular_groebner_zz(
    polys: &[PolyElement<IntegerRing>],
    ring: &Arc<PolyRing<IntegerRing>>,
    primes: Option<&mut dyn Iterator<Item = Int>>,
    trace: Option<&mut ModularTrace>,
) -> PolyResult<Vec<PolyElement<IntegerRing>>> {
    let integer: Vec<IntPoly> = polys
        .iter()
        .filter(|f| !f.is_zero())
        .map(|f| primitive(f.terms.clone(), None))
        .collect();
    if integer.is_empty() {
        return Ok(Vec::new());
    }
    let mut own_trace: ModularTrace = ModularTrace::default();
    let trace: &mut ModularTrace = trace.unwrap_or(&mut own_trace);
    let mut defaults: DefaultPrimes = default_primes();
    let primes: &mut dyn Iterator<Item = Int> = match primes {
        Some(given) => given,
        None => &mut defaults,
    };
    let basis: Vec<Led> = modular_basis(integer, &ring.names, &ring.order, primes, trace)?;
    Ok(basis.into_iter().map(|(_, g)| ring.from_dict(g)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntheory::generate::primerange;
    use crate::polys::domains::QQ;

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn integers() {
        let p: Int = Int::from(101u8);
        let a: Int = mod_floor(&(mod_inverse(&Int::from(3u8), &p).unwrap() * Int::from(-2)), &p);
        assert_eq!(
            rational_reconstruction(&a, &p, None).unwrap(),
            Some((Int::from(-2), Int::from(3u8)))
        );
        assert_eq!(rational_reconstruction(&Int::from(10u8), &p, None).unwrap(), None);
        assert!(rational_reconstruction(&Int::from(10u8), &Int::ONE, None).is_err());
        assert_eq!(
            chinese_remainder(&Int::from(2u8), &Int::from(3u8), &Int::from(3u8), &Int::from(5u8)).unwrap(),
            Int::from(8u8)
        );
        let mut primes = default_primes();
        let top: Int = Int::ONE << 1024;
        assert_eq!(&top - primes.next().unwrap(), Int::from(105u8));
        assert_eq!(&top - primes.next().unwrap(), Int::from(179u8));
    }

    #[test]
    fn doctests() {
        let r = PolyRing::new(QQ, names(&["x", "y", "z"]), MonomialOrder::Lex).unwrap();
        let v = r.gens();
        let (x, y, z) = (&v[0], &v[1], &v[2]);
        let third: Rat = rat(1, 3);
        let f = vec![
            x.pow(2).add(&y.pow(2)).add(&z.pow(2)).sub(&r.one()),
            x.mul(y).sub(&z.mul_ground(&third)),
            x.add(y).sub(&z.mul_ground(&rat(2, 1))),
        ];
        let g = modular_groebner(&f, &r, None, None).unwrap();
        let shown: Vec<String> = g.iter().map(|p| p.to_string()).collect();
        assert_eq!(
            shown,
            vec!["x + y - 2*z", "y**2 - 2*y*z + 1/3*z", "z**2 - 2/15*z - 1/5"]
        );
        assert_eq!(g, f4(&f, &r).unwrap());
        // unlucky primes: x**2 + y, x**2 + 7*y modulo 2 and 3
        let l = PolyRing::new(QQ, names(&["x", "y"]), MonomialOrder::Lex).unwrap();
        let w = l.gens();
        let f = vec![w[0].pow(2).add(&w[1]), w[0].pow(2).add(&w[1].mul_ground(&rat(7, 1)))];
        let mut trace = ModularTrace::default();
        let mut small = primerange(&Int::from(2u8), &Int::from(100u8));
        let g = modular_groebner(&f, &l, Some(&mut small), Some(&mut trace)).unwrap();
        let shown: Vec<String> = g.iter().map(|p| p.to_string()).collect();
        assert_eq!(shown, vec!["x**2", "y"]);
        let ints = |v: &[u8]| -> Vec<Int> { v.iter().map(|p| Int::from(*p)).collect() };
        assert_eq!(
            (trace.unlucky.clone(), trace.lucky.clone()),
            (ints(&[2, 3]), ints(&[5, 7]))
        );
        // the counterexample of the module documentation, p = 5
        let mut primes = primerange(&Int::from(2u8), &Int::from(1000u16));
        let five = rat(5, 1);
        let r3 = PolyRing::new(QQ, names(&["x", "y", "z"]), MonomialOrder::Grevlex).unwrap();
        let u = r3.gens();
        let f = vec![
            u[0].mul(&u[2].add(&u[1].mul_ground(&five)).add(&r3.one())),
            u[0].mul(&u[2].add(&u[1].mul_ground(&rat(10, 1))).add(&r3.from_int(2))),
        ];
        let g = modular_groebner(&f, &r3, Some(&mut primes), None).unwrap();
        assert_eq!(g, f4(&f, &r3).unwrap());
    }
}
