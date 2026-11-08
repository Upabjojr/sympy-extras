//! Port of `sympy_extras/polys/domains/rationalfield.py`: the field of
//! rational numbers, with [`Rat`] as the element type.

use std::collections::BTreeMap;

use crate::error::{zero_division, PolyResult};
use crate::numbers::*;

use super::domain::{next_column, ConvertFrom, Domain, SparseRow};
use super::finitefield::FiniteField;
use super::integerring::IntegerRing;

/// The rational numbers `QQ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RationalField;

/// The field of rational numbers.
pub const QQ: RationalField = RationalField;

impl RationalField {
    pub fn numer(&self, a: &Rat) -> Int {
        a.numerator().clone()
    }

    pub fn denom(&self, a: &Rat) -> Int {
        denom(a).clone()
    }
}

impl Domain for RationalField {
    type Elem = Rat;

    fn name(&self) -> String {
        "QQ".to_string()
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        true
    }

    fn characteristic(&self) -> u64 {
        0
    }

    fn zero(&self) -> Rat {
        Rat::ZERO
    }

    fn one(&self) -> Rat {
        Rat::ONE
    }

    fn is_zero(&self, a: &Rat) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Rat) -> bool {
        a.is_one()
    }

    fn is_negative(&self, a: &Rat) -> bool {
        rat_is_negative(a)
    }

    fn is_unit(&self, a: &Rat) -> bool {
        !a.is_zero()
    }

    fn add(&self, a: &Rat, b: &Rat) -> Rat {
        a + b
    }

    fn sub(&self, a: &Rat, b: &Rat) -> Rat {
        a - b
    }

    fn neg(&self, a: &Rat) -> Rat {
        -a
    }

    fn mul(&self, a: &Rat, b: &Rat) -> Rat {
        a * b
    }

    fn exquo(&self, a: &Rat, b: &Rat) -> PolyResult<Rat> {
        if b.is_zero() {
            return Err(zero_division("division by zero in QQ"));
        }
        Ok(a / b)
    }

    fn quo(&self, a: &Rat, b: &Rat) -> PolyResult<Rat> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &Rat, b: &Rat) -> PolyResult<Rat> {
        if b.is_zero() {
            return Err(zero_division("division by zero in QQ"));
        }
        Ok(Rat::ZERO)
    }

    fn invert(&self, a: &Rat) -> PolyResult<Rat> {
        if a.is_zero() {
            return Err(zero_division("0 has no inverse"));
        }
        Ok(recip(a))
    }

    fn gcd(&self, a: &Rat, b: &Rat) -> Rat {
        if a.is_zero() && b.is_zero() {
            Rat::ZERO
        } else {
            Rat::ONE
        }
    }

    fn lcm(&self, a: &Rat, b: &Rat) -> Rat {
        if a.is_zero() || b.is_zero() {
            Rat::ZERO
        } else {
            Rat::ONE
        }
    }

    fn gcdex(&self, a: &Rat, b: &Rat) -> PolyResult<(Rat, Rat, Rat)> {
        if !a.is_zero() {
            return Ok((recip(a), Rat::ZERO, Rat::ONE));
        }
        if !b.is_zero() {
            return Ok((Rat::ZERO, recip(b), Rat::ONE));
        }
        Ok((Rat::ZERO, Rat::ZERO, Rat::ZERO))
    }

    fn abs(&self, a: &Rat) -> Rat {
        if rat_is_negative(a) {
            -a
        } else {
            a.clone()
        }
    }

    fn canonical_unit(&self, a: &Rat) -> Rat {
        if a.is_zero() {
            Rat::ONE
        } else {
            recip(a)
        }
    }

    fn size(&self, a: &Rat) -> u64 {
        bits(a.numerator()) + bits(&denom(a))
    }

    /// Fraction-free: rows and pivots as primitive integer rows, each step
    /// `r = (L/g) r - (r_c/g) p` with `g = gcd(L, r_c)`, the content
    /// removed once per row.
    fn sparse_reduce(&self, rows: &[SparseRow<Rat>], pivots: &BTreeMap<usize, SparseRow<Rat>>) -> Vec<SparseRow<Rat>> {
        let integer_pivots: BTreeMap<usize, SparseRow<Int>> =
            pivots.iter().map(|(c, p)| (*c, primitive_integer_row(p))).collect();
        let mut result: Vec<SparseRow<Rat>> = Vec::new();
        for row in rows {
            let reduced: SparseRow<Int> = integer_reduce(primitive_integer_row(row), &integer_pivots);
            if !reduced.is_empty() {
                result.push(monic_rational_row(&reduced));
            }
        }
        result
    }

    /// The reduced row echelon form, fraction-free over the integers and
    /// made monic at the end.
    fn sparse_echelon(&self, rows: &[SparseRow<Rat>]) -> Vec<SparseRow<Rat>> {
        let mut pivots: BTreeMap<usize, SparseRow<Int>> = BTreeMap::new();
        for row in rows {
            let reduced: SparseRow<Int> = integer_reduce(primitive_integer_row(row), &pivots);
            if let Some(lead) = reduced.keys().next().copied() {
                pivots.insert(lead, reduced);
            }
        }
        let leads: Vec<usize> = pivots.keys().rev().copied().collect();
        for &c in &leads {
            let pivot: SparseRow<Int> = pivots[&c].clone();
            let lead_value: Int = pivot[&c].clone();
            for &other in &leads {
                if other >= c || !pivots[&other].contains_key(&c) {
                    continue;
                }
                let eliminated: SparseRow<Int> = eliminate(&pivots[&other], c, &pivot, &lead_value);
                pivots.insert(other, eliminated);
            }
        }
        pivots.values().map(monic_rational_row).collect()
    }

    fn factor_dense(&self, f: &[Rat]) -> PolyResult<(Rat, Vec<(Vec<Rat>, usize)>)> {
        crate::polys::factortools::dup_qq_factor(f, self)
    }

    fn isolate_real_roots_dense(&self, f: &[Rat], eps: Option<&Rat>) -> Vec<((Rat, Rat), usize)> {
        let rational: Vec<Rat> = f.to_vec();
        crate::polys::rootisolation::dup_isolate_real_roots(&rational, eps)
    }

    fn count_real_roots_dense(&self, f: &[Rat], inf: Option<&Rat>, sup: Option<&Rat>) -> i64 {
        let rational: Vec<Rat> = f.to_vec();
        crate::polys::rootisolation::dup_count_real_roots(&rational, inf, sup)
    }

    fn from_int(&self, n: i64) -> Rat {
        Rat::from(Int::from(n))
    }

    fn from_integer(&self, n: &Int) -> Rat {
        Rat::from(n.clone())
    }

    fn to_str(&self, a: &Rat) -> String {
        a.to_string()
    }
}

/// The row times the lcm of its denominators, divided by the gcd of the
/// resulting integers, with a positive leading entry.
fn primitive_integer_row(row: &SparseRow<Rat>) -> SparseRow<Int> {
    let mut common: Int = Int::ONE;
    for value in row.values() {
        common = lcm(&common, &denom(value));
    }
    let integers: SparseRow<Int> = row
        .iter()
        .map(|(c, v)| (*c, v.numerator() * (&common / denom(v))))
        .collect();
    primitive(integers)
}

fn primitive(row: SparseRow<Int>) -> SparseRow<Int> {
    let mut content: Int = Int::ZERO;
    for value in row.values() {
        content = gcd(&content, value);
        if content.is_one() {
            break;
        }
    }
    if let Some(lead) = row.values().next() {
        if is_negative(lead) {
            content = -content;
        }
    }
    if content.is_zero() || content.is_one() {
        return row;
    }
    row.into_iter().map(|(c, v)| (c, div_floor(&v, &content))).collect()
}

/// `row` with column `c` cleared by `pivot` (whose entry there is
/// `lead_value`), fraction-free, then made primitive.
fn eliminate(row: &SparseRow<Int>, c: usize, pivot: &SparseRow<Int>, lead_value: &Int) -> SparseRow<Int> {
    let factor: &Int = &row[&c];
    let g: Int = gcd(lead_value, factor);
    let scale: Int = div_floor(lead_value, &g);
    let multiple: Int = div_floor(factor, &g);
    let mut result: SparseRow<Int> = if scale.is_one() {
        row.clone()
    } else {
        row.iter().map(|(column, value)| (*column, value * &scale)).collect()
    };
    for (column, value) in pivot {
        let current: Int = result.get(column).cloned().unwrap_or(Int::ZERO);
        let updated: Int = current - &multiple * value;
        if updated.is_zero() {
            result.remove(column);
        } else {
            result.insert(*column, updated);
        }
    }
    primitive(result)
}

/// The integer row with every pivot column eliminated, in increasing column
/// order.
fn integer_reduce(row: SparseRow<Int>, pivots: &BTreeMap<usize, SparseRow<Int>>) -> SparseRow<Int> {
    let mut row: SparseRow<Int> = row;
    let mut current: Option<usize> = row.keys().next().copied();
    while let Some(c) = current {
        if let Some(pivot) = pivots.get(&c) {
            row = eliminate(&row, c, pivot, &pivot[&c]);
        }
        current = next_column(&row, c);
    }
    row
}

fn monic_rational_row(row: &SparseRow<Int>) -> SparseRow<Rat> {
    let lead: &Int = row.values().next().expect("a nonzero row");
    row.iter()
        .map(|(c, v)| (*c, rat_from_parts(v.clone(), lead.clone())))
        .collect()
}

impl ConvertFrom<RationalField> for RationalField {
    fn convert(&self, a: &Rat, _k0: &RationalField) -> PolyResult<Rat> {
        Ok(a.clone())
    }
}

impl ConvertFrom<IntegerRing> for RationalField {
    fn convert(&self, a: &Int, _k0: &IntegerRing) -> PolyResult<Rat> {
        Ok(Rat::from(a.clone()))
    }
}

impl ConvertFrom<FiniteField> for RationalField {
    fn convert(&self, a: &u64, _k0: &FiniteField) -> PolyResult<Rat> {
        Ok(Rat::from(Int::from(*a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::numbers::*;

    fn q(p: i64, d: i64) -> Rat {
        rat_from_parts(Int::from(p), Int::from(d))
    }

    #[test]
    fn doctests() {
        assert_eq!(QQ.add(&q(1, 2), &q(1, 3)), q(5, 6));
        assert_eq!(QQ.invert(&q(-2, 3)).unwrap(), q(-3, 2));
        assert_eq!(QQ.numer(&q(-6, 4)), Int::from(-3));
        assert_eq!(QQ.denom(&q(-6, 4)), Int::from(2));
    }

    #[test]
    fn fraction_free_echelon_agrees_with_the_generic_one() {
        let rows: Vec<SparseRow<Rat>> = vec![
            [(0, q(2, 1)), (1, q(1, 3)), (3, q(-1, 1))].into_iter().collect(),
            [(0, q(1, 1)), (2, q(5, 2))].into_iter().collect(),
            [(1, q(1, 1)), (2, q(1, 1)), (3, q(1, 7))].into_iter().collect(),
            [(0, q(3, 1)), (1, q(1, 3)), (2, q(5, 2)), (3, q(-1, 1))]
                .into_iter()
                .collect(),
        ];
        #[derive(Debug, Clone, PartialEq)]
        struct Generic;
        impl Domain for Generic {
            type Elem = Rat;
            fn name(&self) -> String {
                "QQ".to_string()
            }
            fn is_field(&self) -> bool {
                true
            }
            fn is_ordered(&self) -> bool {
                true
            }
            fn characteristic(&self) -> u64 {
                0
            }
            fn zero(&self) -> Rat {
                QQ.zero()
            }
            fn one(&self) -> Rat {
                QQ.one()
            }
            fn add(&self, a: &Rat, b: &Rat) -> Rat {
                QQ.add(a, b)
            }
            fn sub(&self, a: &Rat, b: &Rat) -> Rat {
                QQ.sub(a, b)
            }
            fn neg(&self, a: &Rat) -> Rat {
                QQ.neg(a)
            }
            fn mul(&self, a: &Rat, b: &Rat) -> Rat {
                QQ.mul(a, b)
            }
            fn exquo(&self, a: &Rat, b: &Rat) -> PolyResult<Rat> {
                QQ.exquo(a, b)
            }
            fn quo(&self, a: &Rat, b: &Rat) -> PolyResult<Rat> {
                QQ.quo(a, b)
            }
            fn rem(&self, a: &Rat, b: &Rat) -> PolyResult<Rat> {
                QQ.rem(a, b)
            }
            fn invert(&self, a: &Rat) -> PolyResult<Rat> {
                QQ.invert(a)
            }
            fn gcd(&self, a: &Rat, b: &Rat) -> Rat {
                QQ.gcd(a, b)
            }
            fn gcdex(&self, a: &Rat, b: &Rat) -> PolyResult<(Rat, Rat, Rat)> {
                QQ.gcdex(a, b)
            }
            fn from_int(&self, n: i64) -> Rat {
                QQ.from_int(n)
            }
            fn from_integer(&self, n: &Int) -> Rat {
                QQ.from_integer(n)
            }
        }
        assert_eq!(QQ.sparse_echelon(&rows), Generic.sparse_echelon(&rows));
        let echelon = QQ.sparse_echelon(&rows);
        assert_eq!(echelon.len(), 3);
    }
}
