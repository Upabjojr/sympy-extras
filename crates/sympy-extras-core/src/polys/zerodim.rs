//! Port of `sympy_extras/polys/zerodim.py`: the exact complex solutions of
//! zero-dimensional polynomial systems over `QQ`.
//!
//! A `grevlex` basis (F4) decides the dimension; the radical is taken by
//! adding the square-free parts of the univariate polynomials of the ideal;
//! a linear change of the last coordinate (the first of up to sixty small
//! directions) puts the ideal in shape position, whose lexicographic basis
//! (FGLM) is `x_i - g_i(x_n), h(x_n)`; the solutions are the values of the
//! `g_i` at the roots of `h`, as exact [`Algebraic`] numbers.

use std::sync::Arc;

use crate::error::{arithmetic_error, value_error, PolyResult};
use crate::numbers::*;

use super::domains::{RationalField, QQ};
use super::f4::f4;
use super::fglm::{fglm, is_zero_dimensional};
use super::numberfields::qqbar::{algebraic_roots, Algebraic};
use super::rings::{Monomial, MonomialOrder, PolyElement, PolyRing, Terms};
use super::sqfreetools::dup_sqf_part;

type Poly = PolyElement<RationalField>;
type Ring = Arc<PolyRing<RationalField>>;

/// How many changes of coordinates are tried for the shape position.
const DIRECTIONS: usize = 60;

/// The monic polynomial of least degree in the variable number `index`
/// alone which lies in the zero-dimensional ideal of the Gröbner basis
/// `basis` (dense, highest degree first): the first linear dependence
/// between the normal forms of the powers of the variable. The pivot of a
/// new echelon row is its largest monomial (Python: the first dict key); the
/// dependence does not depend on it.
pub fn univariate_in_ideal(basis: &[Poly], ring: &Ring, index: usize) -> PolyResult<Vec<Rat>> {
    let n: usize = ring.ngens;
    let unit: Monomial = (0..n).map(|k| u32::from(k == index)).collect();
    let generators: Vec<Poly> = basis.iter().filter(|g| !g.is_zero()).cloned().collect();
    // echelon rows: pivot monomial, vector, combination of the powers
    let mut rows: Vec<(Monomial, Terms<Rat>, Vec<Rat>)> = Vec::new();
    let mut power: Poly = ring.one().rem(&generators)?;
    let mut degree: usize = 0;
    loop {
        let mut vector: Terms<Rat> = power.terms.clone();
        let mut combination: Vec<Rat> = vec![Rat::ZERO; degree + 1];
        combination[degree] = Rat::ONE;
        for (pivot, row, row_combination) in &rows {
            let Some(c) = vector.get(pivot).cloned() else {
                continue;
            };
            for (m, v) in row {
                let value: Rat = vector.get(m).cloned().unwrap_or(Rat::ZERO) - &c * v;
                if value.is_zero() {
                    vector.remove(m);
                } else {
                    vector.insert(m.clone(), value);
                }
            }
            for (k, v) in row_combination.iter().enumerate() {
                combination[k] -= &c * v;
            }
        }
        if vector.is_empty() {
            // sum combination[k] x**k is in the ideal
            let lead: Rat = combination[degree].clone();
            return Ok(combination.iter().rev().map(|c| c / &lead).collect());
        }
        let pivot: Monomial = vector
            .keys()
            .max_by(|a, b| ring.order.cmp(a, b))
            .expect("a term")
            .clone();
        let inverse: Rat = recip(&vector[&pivot]);
        rows.push((
            pivot,
            vector.iter().map(|(m, v)| (m.clone(), v * &inverse)).collect(),
            combination.iter().map(|c| c * &inverse).collect(),
        ));
        power = power.mul_monom(&unit).rem(&generators)?;
        degree += 1;
    }
}

fn univariate_element(coefficients: &[Rat], ring: &Ring, index: usize) -> Poly {
    let n: usize = ring.ngens;
    let degree: usize = coefficients.len() - 1;
    let mut terms: Terms<Rat> = Terms::new();
    for (k, c) in coefficients.iter().enumerate() {
        if !c.is_zero() {
            terms.insert(
                (0..n)
                    .map(|j| if j == index { (degree - k) as u32 } else { 0 })
                    .collect(),
                c.clone(),
            );
        }
    }
    ring.from_dict(terms)
}

/// `f` with the last variable replaced by `x_n - sum c_i x_i`.
fn shear(f: &Poly, direction: &[i64]) -> Poly {
    let ring: &Ring = &f.ring;
    let gens: Vec<Poly> = ring.gens();
    let n: usize = gens.len();
    let mut replacement: Poly = gens[n - 1].clone();
    for (i, c) in direction.iter().enumerate() {
        if *c != 0 {
            replacement = replacement.sub(&gens[i].mul_ground(&rat(*c, 1)));
        }
    }
    let mut powers: Vec<Poly> = vec![ring.one()];
    let mut result: Poly = ring.zero();
    for (monomial, coefficient) in &f.terms {
        let e: usize = monomial[n - 1] as usize;
        while powers.len() <= e {
            let next: Poly = powers[powers.len() - 1].mul(&replacement);
            powers.push(next);
        }
        let mut head: Monomial = monomial[..n - 1].to_vec();
        head.push(0);
        result = result.add(&powers[e].mul_term(&head, coefficient));
    }
    result
}

/// The images `g_1, ..., g_(n-1)` and `h` of a lexicographic basis
/// `x_i - g_i(x_n), h(x_n)`; `None` for another shape.
type Shape = (Vec<Vec<Rat>>, Vec<Rat>);

fn shape(basis: &[Poly], n: usize) -> Option<Shape> {
    if basis.len() != n {
        return None;
    }
    let dense = |f: &Poly, skip: Option<&Monomial>| -> Option<Vec<Rat>> {
        let degree: usize = f.terms.keys().map(|m| m[n - 1] as usize).max().unwrap_or(0);
        let mut coefficients: Vec<Rat> = vec![Rat::ZERO; degree + 1];
        for (m, c) in &f.terms {
            if Some(m) == skip {
                continue;
            }
            if m[..n - 1].iter().any(|e| *e != 0) {
                return None;
            }
            coefficients[degree - m[n - 1] as usize] = c.clone();
        }
        Some(coefficients)
    };
    let mut images: Vec<Vec<Rat>> = Vec::new();
    for (i, f) in basis.iter().enumerate().take(n - 1) {
        let leading: Monomial = (0..n).map(|k| u32::from(k == i)).collect();
        if f.LM() != leading || f.LC() != Rat::ONE {
            return None;
        }
        let rest: Vec<Rat> = dense(f, Some(&leading))?;
        images.push(rest.iter().map(|c| -c).collect());
    }
    let last: Vec<Rat> = dense(&basis[n - 1], None)?;
    Some((images, last))
}

fn strip(r: &mut Vec<Rat>) {
    let start: usize = r.iter().position(|c| !c.is_zero()).unwrap_or(r.len());
    r.drain(..start);
}

/// `f` modulo the monic `h` (dense, highest degree first).
fn remainder(f: &[Rat], h: &[Rat]) -> Vec<Rat> {
    let mut r: Vec<Rat> = f.to_vec();
    strip(&mut r);
    let d: usize = h.len() - 1;
    while !r.is_empty() && r.len() > d {
        let c: Rat = r[0].clone();
        for k in 1..=d {
            let t: Rat = &c * &h[k];
            r[k] -= t;
        }
        r.remove(0);
        strip(&mut r);
    }
    r
}

/// The directions of `itertools.product(range(-b, b + 1), repeat=k)` for
/// `b = 1, 2, 3` after the zero one, without repetitions.
fn directions(k: usize) -> Vec<Vec<i64>> {
    let mut result: Vec<Vec<i64>> = vec![vec![0; k]];
    for bound in 1..=3i64 {
        let values: Vec<i64> = (-bound..=bound).collect();
        let mut current: Vec<Vec<i64>> = vec![Vec::new()];
        for _ in 0..k {
            let mut next: Vec<Vec<i64>> = Vec::new();
            for prefix in &current {
                for v in &values {
                    let mut d: Vec<i64> = prefix.clone();
                    d.push(*v);
                    next.push(d);
                }
            }
            current = next;
        }
        for d in current {
            if !result.contains(&d) {
                result.push(d);
            }
        }
    }
    result
}

fn common_denominator(f: &[Rat]) -> Int {
    let mut common: Int = Int::ONE;
    for c in f {
        common = lcm(&common, &denom(c));
    }
    common
}

/// All the complex solutions of the system `polys = 0` over the rationals,
/// when there are finitely many: points as lists of algebraic numbers in the
/// order of the variables of `ring`; multiplicities are forgotten.
/// `ValueError` when the system has infinitely many solutions.
pub fn solve_zero_dimensional(polys: &[Poly], ring: &Ring) -> PolyResult<Vec<Vec<Algebraic>>> {
    let n: usize = ring.ngens;
    if n == 0 {
        return Err(value_error("a ring without variables".to_string()));
    }
    let graded: Ring = ring.with_order(MonomialOrder::Grevlex);
    let mut system: Vec<Poly> = polys
        .iter()
        .filter(|f| !f.is_zero())
        .map(|f| graded.from_dict(f.terms.clone()))
        .collect();
    let basis: Vec<Poly> = f4(&system, &graded)?;
    if basis.iter().any(|g| !g.is_zero() && g.LM().iter().all(|e| *e == 0)) {
        return Ok(Vec::new());
    }
    if !is_zero_dimensional(&basis, &graded) {
        return Err(value_error("the system has infinitely many solutions".to_string()));
    }
    // the radical: the square-free parts of the univariate polynomials
    for i in 0..n {
        let univariate: Vec<Rat> = univariate_in_ideal(&basis, &graded, i)?;
        system.push(univariate_element(&dup_sqf_part(&univariate, &QQ), &graded, i));
    }
    for direction in directions(n - 1).into_iter().take(DIRECTIONS) {
        let sheared: Vec<Poly> = system.iter().map(|f| shear(f, &direction)).collect();
        let Some((images, last)) = shape(&fglm(&f4(&sheared, &graded)?, &graded, &MonomialOrder::Lex)?, n) else {
            continue;
        };
        // the last coordinate is u = x_n + sum c_i x_i, a root t of h:
        // x_n = t - sum c_i g_i(t), modulo h
        let mut finale: Vec<Rat> = vec![Rat::ONE, Rat::ZERO];
        for (c, g) in direction.iter().zip(&images) {
            if *c != 0 {
                let size: usize = std::cmp::max(finale.len(), g.len());
                let mut a: Vec<Rat> = vec![Rat::ZERO; size - finale.len()];
                a.extend(finale.iter().cloned());
                let mut b: Vec<Rat> = vec![Rat::ZERO; size - g.len()];
                b.extend(g.iter().cloned());
                let factor: Rat = rat(*c, 1);
                finale = a.iter().zip(&b).map(|(x, y)| x - &factor * y).collect();
            }
        }
        let mut coordinates: Vec<Vec<Rat>> = images;
        coordinates.push(remainder(&finale, &last));
        let scale: Rat = rat_from_int(&common_denominator(&last));
        let integers: Vec<Int> = last.iter().map(|c| numer(&(c * &scale))).collect();
        let mut solutions: Vec<Vec<Algebraic>> = Vec::new();
        for root in algebraic_roots(&integers)? {
            let mut point: Vec<Algebraic> = Vec::new();
            for g in &coordinates {
                point.push(root.polynomial_value(g)?);
            }
            solutions.push(point);
        }
        return Ok(solutions);
    }
    Err(arithmetic_error(format!(
        "no change of coordinates among {} gives the shape position",
        DIRECTIONS
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(solutions: &[Vec<Algebraic>]) -> String {
        let points: Vec<String> = solutions
            .iter()
            .map(|p| {
                format!(
                    "[{}]",
                    p.iter().map(|a| a.repr().unwrap()).collect::<Vec<String>>().join(", ")
                )
            })
            .collect();
        format!("[{}]", points.join(", "))
    }

    #[test]
    fn docstring_examples() {
        let r: Ring = PolyRing::new(QQ, vec!["x".into(), "y".into()], MonomialOrder::Lex).unwrap();
        let g = r.gens();
        let (x, y) = (&g[0], &g[1]);
        let solutions = solve_zero_dimensional(
            &[
                x.mul(x).add(&y.mul(y)).sub(&r.from_int(5)),
                x.mul(y).sub(&r.from_int(2)),
            ],
            &r,
        )
        .unwrap();
        assert_eq!(
            shown(&solutions),
            "[[Algebraic([1, 1], -1), Algebraic([1, 2], -2)], [Algebraic([1, 2], -2), Algebraic([1, 1], -1)], \
             [Algebraic([1, -2], 2), Algebraic([1, -1], 1)], [Algebraic([1, -1], 1), Algebraic([1, -2], 2)]]"
        );
        let basis = f4(
            &[
                x.mul(x).add(&y.mul(y)).sub(&r.from_int(5)),
                x.mul(y).sub(&r.from_int(2)),
            ],
            &r,
        )
        .unwrap();
        assert_eq!(
            univariate_in_ideal(&basis, &r, 0).unwrap(),
            vec![rat(1, 1), rat(0, 1), rat(-5, 1), rat(0, 1), rat(4, 1)]
        );
        assert_eq!(
            shown(&solve_zero_dimensional(&[x.mul(x), y.mul(y)], &r).unwrap()),
            "[[Algebraic([1, 0], 0), Algebraic([1, 0], 0)]]"
        );
        assert_eq!(
            shown(&solve_zero_dimensional(&[x.mul(x).sub(&r.one()), y.mul(y).sub(&r.one())], &r).unwrap()),
            "[[Algebraic([1, -1], 1), Algebraic([1, 1], -1)], [Algebraic([1, -1], 1), Algebraic([1, -1], 1)], \
             [Algebraic([1, 1], -1), Algebraic([1, 1], -1)], [Algebraic([1, 1], -1), Algebraic([1, -1], 1)]]"
        );
        assert!(solve_zero_dimensional(&[x.add(y), r.one()], &r).unwrap().is_empty());
        assert!(solve_zero_dimensional(&[x.mul(y)], &r).is_err());
    }
}
