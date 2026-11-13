//! Port of `sympy_extras/polys/numberfields/recognize.py`: recognition of
//! algebraic numbers from approximations by integer relations (LLL).
//!
//! For degrees `d = 1, 2, ...` the lattice of the rows
//! `(e_i, round(10**digits Re z**i), round(10**digits Im z**i))` is reduced;
//! a first vector much shorter than a random one gives a polynomial, whose
//! root closest to the approximation (within ten units of the last digit)
//! is returned as an exact [`Algebraic`].

use crate::error::{value_error, PolyResult};
use crate::numbers::*;

use super::super::lattice::{default_delta, lll};
use super::qqbar::{algebraic_roots, Algebraic};

fn rounded(q: &Rat) -> Int {
    let (p, d) = (numer(q), denom(q));
    div_floor(&(Int::from(2u8) * p + &d), &(Int::from(2u8) * d))
}

fn rat_abs(v: &Rat) -> Rat {
    if *v < Rat::ZERO {
        -v
    } else {
        v.clone()
    }
}

/// An algebraic number of degree at most `max_degree` with small minimal
/// polynomial which agrees with `re + im*i` to `digits` decimal digits, or
/// `None` when no convincing one exists (`ValueError` unless `digits` and
/// `max_degree` are positive).
pub fn recognize_algebraic(re: &Rat, im: &Rat, digits: usize, max_degree: usize) -> PolyResult<Option<Algebraic>> {
    if digits < 1 || max_degree < 1 {
        return Err(value_error("digits and max_degree must be positive".to_string()));
    }
    let scale: Int = Int::from(10u8).pow(digits);
    let scale_rat: Rat = rat_from_int(&scale);
    let tolerance: Rat = rat_from_parts(Int::from(10u8), scale.clone());
    let constraints: usize = if im.is_zero() { 1 } else { 2 };
    for degree in 1..=max_degree {
        let mut rows: Vec<Vec<Int>> = Vec::new();
        let (mut power_re, mut power_im): (Rat, Rat) = (Rat::ONE, Rat::ZERO);
        for i in 0..=degree {
            let mut row: Vec<Int> = (0..=degree)
                .map(|j| if j == i { Int::ONE } else { Int::ZERO })
                .collect();
            row.push(rounded(&(&scale_rat * &power_re)));
            if constraints == 2 {
                row.push(rounded(&(&scale_rat * &power_im)));
            }
            rows.push(row);
            let next_re: Rat = &power_re * re - &power_im * im;
            let next_im: Rat = &power_re * im + &power_im * re;
            power_re = next_re;
            power_im = next_im;
        }
        let vector: Vec<Int> = lll(&rows, &default_delta())?[0].clone();
        let coefficients: Vec<Int> = vector[..=degree].to_vec();
        // a number without relation gives vectors of length about
        // scale**(constraints/(degree + 1 + constraints)); a relation is
        // believed when it is ten times shorter
        let mut norm_squared: Int = Int::ZERO;
        for c in &vector {
            norm_squared += c * c;
        }
        let k: usize = degree + 1 + constraints;
        if norm_squared.pow(k) * Int::from(10u8).pow(2 * k) > scale.pow(2 * constraints) {
            continue;
        }
        let mut polynomial: Vec<Int> = coefficients.into_iter().rev().collect();
        let start: usize = polynomial.iter().position(|c| !c.is_zero()).unwrap_or(polynomial.len());
        polynomial.drain(..start);
        if polynomial.len() < 2 {
            continue;
        }
        let two: Rat = rat(2, 1);
        for root in algebraic_roots(&polynomial)? {
            let b = root.enclosure(&(&tolerance / rat(4, 1)))?;
            if rat_abs(&((&b.0 + &b.1) / &two - re)) < tolerance
                && rat_abs(&((&b.2 + &b.3) / &two - im)) < tolerance
                && root.is_real() == im.is_zero()
            {
                return Ok(Some(root));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(r: Option<Algebraic>) -> Option<String> {
        r.map(|a| a.repr().unwrap())
    }

    #[test]
    fn docstring_examples() {
        let e15: Int = Int::from(10u8).pow(15);
        let q = |n: i64| rat_from_parts(Int::from(n), e15.clone());
        assert_eq!(
            shown(recognize_algebraic(&q(1618033988749895), &Rat::ZERO, 15, 6).unwrap()).as_deref(),
            Some("Algebraic([1, -1, -1], 1.618033989)")
        );
        assert_eq!(
            shown(recognize_algebraic(&rat(3, 7), &Rat::ZERO, 15, 6).unwrap()).as_deref(),
            Some("Algebraic([7, -3], 3/7)")
        );
        assert_eq!(
            shown(recognize_algebraic(&q(3146264369941972), &Rat::ZERO, 15, 4).unwrap()).as_deref(),
            Some("Algebraic([1, 0, -10, 0, 1], 3.14626437)")
        );
        assert_eq!(
            shown(recognize_algebraic(&rat(-1, 2), &q(866025403784439), 15, 6).unwrap()).as_deref(),
            Some("Algebraic([1, 1, 1], -0.5 + 0.8660254038*I)")
        );
        assert!(recognize_algebraic(&q(3141592653589793), &Rat::ZERO, 15, 4)
            .unwrap()
            .is_none());
    }
}
