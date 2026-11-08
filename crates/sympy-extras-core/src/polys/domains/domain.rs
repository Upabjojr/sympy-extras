//! Port of `sympy_extras/polys/domains/domain.py`: the trait of the
//! coefficient domains.
//!
//! A domain is a ring (or field) whose elements have one Rust type
//! ([`Domain::Elem`]): [`Int`] for the integers,
//! [`Rat`] for the rationals, `u64` for
//! the prime fields. The domain value carries the arithmetic as methods, so
//! that the polynomial algorithms are written once, generic over a
//! `K: Domain`. Nothing is coerced silently: conversions between domains
//! are explicit ([`ConvertFrom`]).
//!
//! References
//! ==========
//!
//! 1. K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
//!    Algebra*, Kluwer, 1992, Chapter 2.
//!
//! [`Int`]: crate::numbers::Int
//! [`Rat`]: crate::numbers::Rat

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::Bound::{Excluded, Unbounded};

use crate::error::PolyResult;
use crate::numbers::*;

/// A sparse row of a matrix over a domain: column -> nonzero entry
/// (Python's `dict[int, T]`).
pub type SparseRow<T> = BTreeMap<usize, T>;

/// The smallest column of a row strictly after `c`.
pub(crate) fn next_column<T>(row: &SparseRow<T>, c: usize) -> Option<usize> {
    row.range((Excluded(c), Unbounded)).next().map(|(k, _)| *k)
}

/// A ring whose elements have the type `Elem`.
pub trait Domain: Clone + PartialEq + Debug {
    /// The type of the elements.
    type Elem: Clone + PartialEq + Debug;

    /// A short name for printing (`ZZ`, `QQ`, `GF(7)`).
    fn name(&self) -> String;
    /// Whether every nonzero element is invertible.
    fn is_field(&self) -> bool;
    /// Whether the domain is ordered (`is_negative`, `abs` are defined).
    fn is_ordered(&self) -> bool;
    /// The characteristic (`0` or a prime).
    fn characteristic(&self) -> u64;

    // -- constants and predicates

    fn zero(&self) -> Self::Elem;
    fn one(&self) -> Self::Elem;

    fn is_zero(&self, a: &Self::Elem) -> bool {
        *a == self.zero()
    }

    fn is_one(&self, a: &Self::Elem) -> bool {
        *a == self.one()
    }

    /// Whether `a < 0` (ordered domains; panics otherwise).
    fn is_negative(&self, _a: &Self::Elem) -> bool {
        panic!("{} is not ordered", self.name())
    }

    /// Whether `a` has an inverse in the domain.
    fn is_unit(&self, a: &Self::Elem) -> bool {
        if self.is_field() {
            return !self.is_zero(a);
        }
        panic!("is_unit is not implemented for {}", self.name())
    }

    // -- arithmetic

    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    /// `a**n` by repeated squaring.
    fn pow(&self, a: &Self::Elem, n: u64) -> Self::Elem {
        let mut n = n;
        let mut result = self.one();
        let mut base = a.clone();
        while n != 0 {
            if n & 1 == 1 {
                result = self.mul(&result, &base);
            }
            n >>= 1;
            if n != 0 {
                base = self.mul(&base, &base);
            }
        }
        result
    }

    /// `a**n` for any integer `n` (`n < 0` through the inverse; Python's
    /// `Domain.pow` with a negative exponent).
    fn pow_signed(&self, a: &Self::Elem, n: i64) -> PolyResult<Self::Elem> {
        if n < 0 {
            let inverse = self.invert(a)?;
            return Ok(self.pow(&inverse, n.unsigned_abs()));
        }
        Ok(self.pow(a, n as u64))
    }

    /// The exact quotient `a/b`: `ZeroDivision` for `b = 0`, `Value` when
    /// the division is not exact.
    fn exquo(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<Self::Elem>;

    /// The quotient of the division with remainder (`a/b` in a field).
    fn quo(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<Self::Elem>;

    /// The remainder of the division (`0` in a field).
    fn rem(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<Self::Elem>;

    fn divmod(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<(Self::Elem, Self::Elem)> {
        Ok((self.quo(a, b)?, self.rem(a, b)?))
    }

    /// The inverse of a unit.
    fn invert(&self, a: &Self::Elem) -> PolyResult<Self::Elem>;

    /// A greatest common divisor, normalised (nonnegative in an ordered
    /// domain, `1` for nonzero elements of a field).
    fn gcd(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    fn lcm(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        if self.is_zero(a) || self.is_zero(b) {
            return self.zero();
        }
        self.exquo(&self.mul(a, b), &self.gcd(a, b))
            .expect("the gcd divides the product")
    }

    /// `(s, t, g)` with `s a + t b = g = gcd(a, b)`.
    fn gcdex(&self, a: &Self::Elem, b: &Self::Elem) -> PolyResult<(Self::Elem, Self::Elem, Self::Elem)>;

    /// The absolute value (ordered domains).
    fn abs(&self, a: &Self::Elem) -> Self::Elem {
        if self.is_negative(a) {
            self.neg(a)
        } else {
            a.clone()
        }
    }

    /// `-1`, `0` or `1` (ordered domains).
    fn sign(&self, a: &Self::Elem) -> i32 {
        if self.is_zero(a) {
            0
        } else if self.is_negative(a) {
            -1
        } else {
            1
        }
    }

    /// The unit `u` such that `u a` is in canonical form (positive in an
    /// ordered domain, `1/a` in a field, `1` otherwise); `1` for `a = 0`.
    fn canonical_unit(&self, _a: &Self::Elem) -> Self::Elem {
        self.one()
    }

    /// A measure of the size of an element (the number of bits of the
    /// numbers it is made of; `1` for domains without a better one).
    fn size(&self, _a: &Self::Elem) -> u64 {
        1
    }

    // -- sparse linear algebra over a field

    /// The rows with every column which has a pivot eliminated (the pivots
    /// have leading coefficient one at their key column); the rows which
    /// become zero are dropped. The columns are visited in increasing
    /// order, as with the heap of the Python code.
    fn sparse_reduce(
        &self,
        rows: &[SparseRow<Self::Elem>],
        pivots: &BTreeMap<usize, SparseRow<Self::Elem>>,
    ) -> Vec<SparseRow<Self::Elem>> {
        let mut result: Vec<SparseRow<Self::Elem>> = Vec::new();
        for original in rows {
            let mut row: SparseRow<Self::Elem> = original.clone();
            let mut current: Option<usize> = row.keys().next().copied();
            while let Some(c) = current {
                if let Some(pivot) = pivots.get(&c) {
                    let factor: Self::Elem = row[&c].clone();
                    eliminate_with_pivot(self, &mut row, &factor, pivot);
                }
                current = next_column(&row, c);
            }
            if !row.is_empty() {
                result.push(row);
            }
        }
        result
    }

    /// The reduced row echelon form: rows with distinct leading (smallest)
    /// columns, leading coefficient one, and no entry in the leading
    /// column of another row, sorted by leading column.
    fn sparse_echelon(&self, rows: &[SparseRow<Self::Elem>]) -> Vec<SparseRow<Self::Elem>> {
        let mut pivots: BTreeMap<usize, SparseRow<Self::Elem>> = BTreeMap::new();
        for original in rows {
            let reduced: Vec<SparseRow<Self::Elem>> = self.sparse_reduce(std::slice::from_ref(original), &pivots);
            if reduced.is_empty() {
                continue;
            }
            let row: &SparseRow<Self::Elem> = &reduced[0];
            let (lead, lead_value) = row.iter().next().expect("a nonzero row");
            let inverse: Self::Elem = self.invert(lead_value).expect("a nonzero element of a field");
            let monic: SparseRow<Self::Elem> = row
                .iter()
                .map(|(column, value)| (*column, self.mul(value, &inverse)))
                .collect();
            pivots.insert(*lead, monic);
        }
        let leads: Vec<usize> = pivots.keys().rev().copied().collect();
        for &c in &leads {
            let pivot: SparseRow<Self::Elem> = pivots[&c].clone();
            for &other in &leads {
                if other >= c || !pivots[&other].contains_key(&c) {
                    continue;
                }
                let row: &mut SparseRow<Self::Elem> = pivots.get_mut(&other).expect("a pivot row");
                let factor: Self::Elem = row[&c].clone();
                eliminate_with_pivot(self, row, &factor, &pivot);
            }
        }
        pivots.into_values().collect()
    }

    // -- dense polynomial kernels

    /// Whether [`Domain::dense_mul`] and [`Domain::dense_div`] are provided.
    fn has_dense_kernels(&self) -> bool {
        false
    }

    /// The product of two dense polynomials (highest degree first, no
    /// leading zeros), for domains with `has_dense_kernels`.
    fn dense_mul(&self, _f: &[Self::Elem], _g: &[Self::Elem]) -> Vec<Self::Elem> {
        panic!("dense_mul is not implemented for {}", self.name())
    }

    /// The quotient and remainder of dense polynomials, for domains with
    /// `has_dense_kernels` (a field, or the division is exact).
    fn dense_div(&self, _f: &[Self::Elem], _g: &[Self::Elem]) -> PolyResult<(Vec<Self::Elem>, Vec<Self::Elem>)> {
        panic!("dense_div is not implemented for {}", self.name())
    }

    // -- algorithms which exist for particular domains only: the domain
    // dispatches to the implementation

    /// The irreducible factorisation of a dense polynomial: the constant and
    /// the factors with their multiplicities (`ZZ`, `QQ`, `GF(p)`; panics
    /// for a domain without one, as Python's `NotImplementedError`).
    fn factor_dense(&self, _f: &[Self::Elem]) -> PolyResult<(Self::Elem, Vec<(Vec<Self::Elem>, usize)>)> {
        panic!("no factorisation over {}", self.name())
    }

    // -- conversions

    /// The image of a machine integer.
    fn from_int(&self, n: i64) -> Self::Elem;

    /// The image of an integer.
    fn from_integer(&self, n: &Int) -> Self::Elem;

    /// The element as a string (Python's `str`).
    fn to_str(&self, a: &Self::Elem) -> String {
        format!("{:?}", a)
    }

    // -- enumeration (finite domains)

    fn elements(&self) -> Box<dyn Iterator<Item = Self::Elem> + '_> {
        panic!("{} is not finite", self.name())
    }
}

/// `row -= factor * pivot` entrywise, the zero entries removed.
fn eliminate_with_pivot<K: Domain>(
    k: &K,
    row: &mut SparseRow<K::Elem>,
    factor: &K::Elem,
    pivot: &SparseRow<K::Elem>,
) {
    for (column, value) in pivot {
        let product: K::Elem = k.mul(factor, value);
        match row.get(column) {
            Some(entry) => {
                let updated: K::Elem = k.sub(entry, &product);
                if k.is_zero(&updated) {
                    row.remove(column);
                } else {
                    row.insert(*column, updated);
                }
            }
            None => {
                row.insert(*column, k.neg(&product));
            }
        }
    }
}

/// An exact conversion from the elements of the domain `K0` (Python's
/// `Domain.convert(a, K0)`): `Value` when the element has no image.
pub trait ConvertFrom<K0: Domain>: Domain {
    fn convert(&self, a: &K0::Elem, k0: &K0) -> PolyResult<Self::Elem>;
}
