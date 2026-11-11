//! Port of `sympy_extras/polys/numberfields/basis.py`: rings of integers of
//! number fields, integral bases and field discriminants by the Round 2
//! algorithm (Cohen, Algorithm 6.1.8).
//!
//! The order `ZZ[lc theta]` is enlarged at each prime whose square divides
//! the discriminant: the `p`-radical `I_p` is the kernel of `x -> x**q` on
//! `O/pO`, and `O' = {x : x I_p in I_p}` is `(1/p) U + O` for the kernel `U`
//! of the multiplications on `I_p/p I_p`; the step is repeated until nothing
//! changes (the criterion of Pohst and Zassenhaus). Linear algebra over
//! `GF(p)` and Hermite normal forms ([`super::super::lattice`]).
//!
//! References
//! ==========
//!
//! 1. H. Cohen, *A Course in Computational Algebraic Number Theory*,
//!    Springer, 1993, Section 6.1.
//! 2. M. Pohst, H. Zassenhaus, *Algorithmic Algebraic Number Theory*,
//!    Cambridge University Press, 1989, Section 4.6.

// the matrix products read columns by index, as the Python does
#![allow(clippy::needless_range_loop)]

use crate::error::{arithmetic_error, value_error, PolyResult};
use crate::ntheory::factorint;
use crate::numbers::*;

use super::super::domains::ZZ;
use super::super::euclidtools::dup_resultant;
use super::super::lattice::hermite_normal_form;

/// The discriminant of an integer polynomial (highest degree first):
/// `(-1)**(n (n - 1)/2) Res(f, f')/lc(f)`; `ValueError` for a constant.
pub fn polynomial_discriminant(f: &[Int]) -> PolyResult<Int> {
    if f.len() < 2 {
        return Err(value_error("a polynomial of positive degree is needed".to_string()));
    }
    let n: usize = f.len() - 1;
    let derivative: Vec<Int> = f[..n].iter().enumerate().map(|(k, c)| c * Int::from(n - k)).collect();
    let resultant: Int = dup_resultant(f, &derivative, &ZZ);
    let value: Int = div_floor(&resultant, &f[0]);
    Ok(if (n * (n - 1) / 2) % 2 == 1 { -value } else { value })
}

/// A basis of `{x in F_p**m : x . rows = 0}` for the `m` rows of `width`
/// entries: the left kernel modulo `p`.
fn kernel_mod(rows: &[Vec<Int>], width: usize, p: &Int) -> Vec<Vec<Int>> {
    let m: usize = rows.len();
    // eliminate on [rows | identity]
    let mut work: Vec<Vec<Int>> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut line: Vec<Int> = row.iter().map(|v| mod_floor(v, p)).collect();
            line.extend((0..m).map(|k| if k == i { Int::ONE } else { Int::ZERO }));
            line
        })
        .collect();
    let mut rank: usize = 0;
    for column in 0..width {
        let Some(pivot) = (rank..m).find(|&i| !work[i][column].is_zero()) else {
            continue;
        };
        work.swap(rank, pivot);
        let inverse: Int = mod_inverse(&work[rank][column], p).expect("a unit modulo a prime");
        work[rank] = work[rank].iter().map(|v| mod_floor(&(v * &inverse), p)).collect();
        for i in 0..m {
            if i != rank && !work[i][column].is_zero() {
                let factor: Int = work[i][column].clone();
                let top: Vec<Int> = work[rank].clone();
                work[i] = work[i]
                    .iter()
                    .zip(&top)
                    .map(|(a, b)| mod_floor(&(a - &factor * b), p))
                    .collect();
            }
        }
        rank += 1;
    }
    work[rank..].iter().map(|row| row[width..].to_vec()).collect()
}

fn inverse_matrix(rows: &[Vec<Int>]) -> PolyResult<Vec<Vec<Rat>>> {
    let n: usize = rows.len();
    let mut work: Vec<Vec<Rat>> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut line: Vec<Rat> = row.iter().map(rat_from_int).collect();
            line.extend((0..n).map(|k| if k == i { Rat::ONE } else { Rat::ZERO }));
            line
        })
        .collect();
    for column in 0..n {
        let Some(pivot) = (column..n).find(|&i| !work[i][column].is_zero()) else {
            return Err(arithmetic_error("a singular basis".to_string()));
        };
        work.swap(column, pivot);
        let inverse: Rat = recip(&work[column][column]);
        work[column] = work[column].iter().map(|v| v * &inverse).collect();
        for i in 0..n {
            if i != column && !work[i][column].is_zero() {
                let factor: Rat = work[i][column].clone();
                let top: Vec<Rat> = work[column].clone();
                work[i] = work[i].iter().zip(&top).map(|(a, b)| a - &factor * b).collect();
            }
        }
    }
    Ok(work.into_iter().map(|row| row[n..].to_vec()).collect())
}

fn integer_of(c: &Rat) -> PolyResult<Int> {
    if rat_is_integer(c) {
        Ok(numer(c))
    } else {
        Err(arithmetic_error("a coordinate is not an integer".to_string()))
    }
}

/// An order of `QQ[x]/(T)`, `T` monic: the rows of `basis` divided by
/// `denominator` are a `ZZ`-basis in the powers `1, x, ..., x**(n-1)`
/// (lowest first).
#[derive(Debug, Clone)]
struct Order {
    /// Monic, highest degree first.
    modulus: Vec<Int>,
    n: usize,
    basis: Vec<Vec<Int>>,
    denominator: Int,
    /// The inverse of the basis matrix, to read coordinates.
    inverse: Vec<Vec<Rat>>,
    /// `table[i][j]`: the integer coordinates of the product of the basis
    /// elements `i` and `j`.
    table: Vec<Vec<Vec<Int>>>,
}

impl Order {
    fn new(modulus: Vec<Int>, basis: Vec<Vec<Int>>, denominator: Int) -> PolyResult<Order> {
        let n: usize = modulus.len() - 1;
        let mut common: Int = denominator.clone();
        for row in &basis {
            for v in row {
                common = gcd(&common, v);
            }
        }
        let basis: Vec<Vec<Int>> = basis
            .iter()
            .map(|row| row.iter().map(|v| div_floor(v, &common)).collect())
            .collect();
        let denominator: Int = div_floor(&denominator, &common);
        let inverse: Vec<Vec<Rat>> = inverse_matrix(&basis)?;
        let mut order: Order = Order {
            modulus,
            n,
            basis,
            denominator,
            inverse,
            table: Vec::new(),
        };
        order.table = order.compute_table()?;
        Ok(order)
    }

    /// The product of two elements given in the powers of `x`.
    fn multiply(&self, a: &[Rat], b: &[Rat]) -> Vec<Rat> {
        let n: usize = self.n;
        let mut product: Vec<Rat> = vec![Rat::ZERO; 2 * n - 1];
        for (i, u) in a.iter().enumerate() {
            if !u.is_zero() {
                for (j, v) in b.iter().enumerate() {
                    product[i + j] += u * v;
                }
            }
        }
        // x**n = -(c_(n-1) x**(n-1) + ... + c_0)
        for k in (n..=2 * n - 2).rev() {
            let c: Rat = product[k].clone();
            if !c.is_zero() {
                for j in 0..n {
                    product[k - n + j] -= &c * rat_from_int(&self.modulus[n - j]);
                }
            }
        }
        product.truncate(n);
        product
    }

    /// The coordinates in the basis of the order.
    fn coordinates(&self, element: &[Rat]) -> Vec<Rat> {
        let d: Rat = rat_from_int(&self.denominator);
        (0..self.n)
            .map(|i| {
                let mut total: Rat = Rat::ZERO;
                for k in 0..self.n {
                    total += &element[k] * &self.inverse[k][i];
                }
                total * &d
            })
            .collect()
    }

    fn compute_table(&self) -> PolyResult<Vec<Vec<Vec<Int>>>> {
        let d: Rat = rat_from_int(&self.denominator);
        let elements: Vec<Vec<Rat>> = self
            .basis
            .iter()
            .map(|row| row.iter().map(|v| rat_from_int(v) / &d).collect())
            .collect();
        let mut table: Vec<Vec<Vec<Int>>> = Vec::new();
        for i in 0..self.n {
            let mut line: Vec<Vec<Int>> = Vec::new();
            for j in 0..self.n {
                let coordinates: Vec<Rat> = self.coordinates(&self.multiply(&elements[i], &elements[j]));
                if coordinates.iter().any(|c| !rat_is_integer(c)) {
                    return Err(arithmetic_error(
                        "the module is not closed under multiplication".to_string(),
                    ));
                }
                line.push(coordinates.iter().map(numer).collect());
            }
            table.push(line);
        }
        Ok(table)
    }

    /// The product of two elements given by their coordinates, the
    /// coordinates of the result modulo `p` (not reduced for `p = 0`).
    fn product(&self, a: &[Int], b: &[Int], p: &Int) -> Vec<Int> {
        let mut result: Vec<Int> = vec![Int::ZERO; self.n];
        for (i, u) in a.iter().enumerate() {
            if u.is_zero() {
                continue;
            }
            for (j, v) in b.iter().enumerate() {
                if v.is_zero() {
                    continue;
                }
                let c: Int = u * v;
                for (k, entry) in self.table[i][j].iter().enumerate() {
                    result[k] += &c * entry;
                }
            }
        }
        if p.is_zero() {
            result
        } else {
            result.iter().map(|v| mod_floor(v, p)).collect()
        }
    }

    /// The order `{x : x I_p in I_p}`, which is this one exactly when it is
    /// `p`-maximal.
    fn enlarged(&self, p: &Int) -> PolyResult<Order> {
        let n: usize = self.n;
        let mut q: Int = p.clone();
        while q < Int::from(n) {
            q *= p;
        }
        // the images of the basis under x -> x**q in O/pO
        let mut one: Vec<Rat> = vec![Rat::ZERO; n];
        one[0] = Rat::ONE;
        let one_coordinates: Vec<Int> = self.coordinates(&one).iter().map(|c| mod_floor(&numer(c), p)).collect();
        let mut images: Vec<Vec<Int>> = Vec::new();
        for i in 0..n {
            let mut power: Vec<Int> = one_coordinates.clone();
            let mut base: Vec<Int> = (0..n).map(|k| if k == i { Int::ONE } else { Int::ZERO }).collect();
            let length: usize = bits(&q) as usize;
            for b in 0..length {
                if bit(&q, b) {
                    power = self.product(&power, &base, p);
                }
                if b + 1 < length {
                    base = self.product(&base, &base, p);
                }
            }
            images.push(power);
        }
        let mut generators: Vec<Vec<Int>> = kernel_mod(&images, n, p);
        // I_p = radical + p O, in the coordinates of the order
        generators.extend((0..n).map(|i| (0..n).map(|k| if k == i { p.clone() } else { Int::ZERO }).collect()));
        let ideal: Vec<Vec<Int>> = hermite_normal_form(&generators)?;
        let ideal_inverse: Vec<Vec<Rat>> = inverse_matrix(&ideal)?;
        // the multiplication by each basis element on I_p/p I_p
        let mut maps: Vec<Vec<Int>> = Vec::new();
        for j in 0..n {
            let unit: Vec<Int> = (0..n).map(|k| if k == j { Int::ONE } else { Int::ZERO }).collect();
            let mut flat: Vec<Int> = Vec::new();
            for gamma in &ideal {
                let image: Vec<Int> = self.product(&unit, gamma, &Int::ZERO);
                for i in 0..n {
                    let mut total: Rat = Rat::ZERO;
                    for k in 0..n {
                        total += rat_from_int(&image[k]) * &ideal_inverse[k][i];
                    }
                    if !rat_is_integer(&total) {
                        return Err(arithmetic_error("the radical is not an ideal".to_string()));
                    }
                    flat.push(mod_floor(&integer_of(&total)?, p));
                }
            }
            maps.push(flat);
        }
        let kernel: Vec<Vec<Int>> = kernel_mod(&maps, n * n, p);
        // O' = O + (1/p) kernel: generators over the denominator p d
        let mut generators: Vec<Vec<Int>> = self
            .basis
            .iter()
            .map(|row| row.iter().map(|v| p * v).collect())
            .collect();
        for u in &kernel {
            generators.push(
                (0..n)
                    .map(|k| {
                        let mut total: Int = Int::ZERO;
                        for i in 0..n {
                            total += &u[i] * &self.basis[i][k];
                        }
                        total
                    })
                    .collect(),
            );
        }
        Order::new(
            self.modulus.clone(),
            hermite_normal_form(&generators)?,
            p * &self.denominator,
        )
    }
}

/// The maximal order of `QQ[x]/(T)` for the monic polynomial `T` with the
/// roots `lc theta`; with `lc` and the discriminant of `T`.
fn maximal_order(minpoly: &[Int]) -> PolyResult<(Order, Int, Int)> {
    if minpoly.len() < 2 {
        return Err(value_error("a polynomial of positive degree is needed".to_string()));
    }
    let n: usize = minpoly.len() - 1;
    let minpoly: Vec<Int> = if is_negative(&minpoly[0]) {
        minpoly.iter().map(|c| -c).collect()
    } else {
        minpoly.to_vec()
    };
    let lc: Int = minpoly[0].clone();
    // the monic polynomial of lc theta
    let monic: Vec<Int> = minpoly
        .iter()
        .enumerate()
        .map(|(k, c)| if k == 0 { Int::ONE } else { c * lc.pow(k - 1) })
        .collect();
    let discriminant: Int = polynomial_discriminant(&monic)?;
    if discriminant.is_zero() {
        return Err(value_error("the polynomial is not square-free".to_string()));
    }
    let identity: Vec<Vec<Int>> = (0..n)
        .map(|i| (0..n).map(|k| if k == i { Int::ONE } else { Int::ZERO }).collect())
        .collect();
    let mut order: Order = Order::new(monic, identity, Int::ONE)?;
    for (p, exponent) in factorint(&abs(&discriminant), None) {
        if exponent < 2 {
            continue;
        }
        loop {
            let larger: Order = order.enlarged(&p)?;
            if larger.basis == order.basis && larger.denominator == order.denominator {
                break;
            }
            order = larger;
        }
    }
    Ok((order, lc, discriminant))
}

/// An integral basis of the number field `QQ(theta)`, `theta` a root of the
/// square-free integer polynomial `minpoly` (highest degree first): `n`
/// polynomials in `theta` with rational coefficients (highest degree first),
/// the `k`-th of degree `k`, in Hermite normal form.
pub fn integral_basis(minpoly: &[Int]) -> PolyResult<Vec<Vec<Rat>>> {
    let (order, lc, _) = maximal_order(minpoly)?;
    // echelon from the highest power down: one element of each degree
    let reversed: Vec<Vec<Int>> = order
        .basis
        .iter()
        .map(|row| row.iter().rev().cloned().collect())
        .collect();
    let basis: Vec<Vec<Int>> = hermite_normal_form(&reversed)?
        .into_iter()
        .map(|row| row.into_iter().rev().collect())
        .collect();
    let mut result: Vec<Vec<Rat>> = Vec::new();
    for row in basis {
        // (lc theta)**k = lc**k theta**k
        let mut coefficients: Vec<Rat> = row
            .iter()
            .enumerate()
            .map(|(k, v)| rat_from_parts(v * lc.pow(k), order.denominator.clone()))
            .collect();
        while coefficients.len() > 1 && coefficients[coefficients.len() - 1].is_zero() {
            coefficients.pop();
        }
        coefficients.reverse();
        result.push(coefficients);
    }
    result.sort_by_key(|c| c.len());
    Ok(result)
}

/// The discriminant of the number field `QQ(theta)`: the discriminant of
/// the polynomial divided by the square of the index of `ZZ[theta]` in the
/// ring of integers.
pub fn field_discriminant(minpoly: &[Int]) -> PolyResult<Int> {
    let (order, _, discriminant) = maximal_order(minpoly)?;
    let n: usize = order.basis.len();
    let mut determinant: Int = Int::ONE;
    for i in 0..n {
        determinant *= &order.basis[i][i];
    }
    // the index of ZZ[lc theta] in the maximal order is d**n/det
    let index: Rat = rat_from_parts(order.denominator.pow(n), abs(&determinant));
    let value: Rat = rat_from_int(&discriminant) / (&index * &index);
    if !rat_is_integer(&value) {
        return Err(arithmetic_error(
            "the index does not divide the discriminant".to_string(),
        ));
    }
    Ok(numer(&value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: &[i64]) -> Vec<Int> {
        v.iter().map(|c| int(*c)).collect()
    }

    #[test]
    fn docstring_examples() {
        assert_eq!(
            integral_basis(&z(&[1, 0, -5])).unwrap(),
            vec![vec![rat(1, 1)], vec![rat(1, 2), rat(1, 2)]]
        );
        assert_eq!(field_discriminant(&z(&[1, 0, -5])).unwrap(), int(5));
        assert_eq!(field_discriminant(&z(&[1, 0, 1])).unwrap(), int(-4));
        assert_eq!(field_discriminant(&z(&[1, 0, 0, -2])).unwrap(), int(-108));
        assert_eq!(
            integral_basis(&z(&[1, -1, -2, -8])).unwrap(),
            vec![
                vec![rat(1, 1)],
                vec![rat(1, 1), rat(0, 1)],
                vec![rat(1, 2), rat(1, 2), rat(0, 1)]
            ]
        );
        assert_eq!(field_discriminant(&z(&[1, -1, -2, -8])).unwrap(), int(-503));
        assert_eq!(
            integral_basis(&z(&[4, 0, -5])).unwrap(),
            vec![vec![rat(1, 1)], vec![rat(1, 1), rat(1, 2)]]
        );
        assert_eq!(field_discriminant(&z(&[1, 1, 1, 1, 1])).unwrap(), int(125));
        assert_eq!(field_discriminant(&z(&[1, 0, -1, -1])).unwrap(), int(-23));
        assert_eq!(polynomial_discriminant(&z(&[2, 0, 1])).unwrap(), int(-8));
    }
}
