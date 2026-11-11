//! Port of `sympy_extras/polys/functionfields.py`: the algebraic function
//! fields `K(x)[y]/(F)` (arithmetic of their [`Element`]s), integral bases
//! by Trager's algorithm (the trace radical and its idealizer, as
//! polynomial linear algebra over `K[x]`), the local integral bases at
//! infinity and the bases normal at infinity.
//!
//! [`FunctionField`] is generic in its field of constants `K` (`QQ`, a
//! number field, the rational functions of parameters). An [`Element`]
//! holds its coordinates only; the arithmetic is on the field. The caches
//! of the Python class (the derivative of `Y`, the power traces, the
//! bases) are kept; the cache of the inverses of the basis matrices, by
//! the identity of the lists, is not (it only saves time).
//!
//! [`Element::as_expr`] expands with the guarded default `expand` of the
//! core ([`expand_default`]).

// the matrix algorithms index rows and columns as the Python module does
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::core::add::add;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is_integer_class;
use crate::core::function::expand_default;
use crate::core::mul::{div, mul};
use crate::core::power::pow;
use crate::error::{value_error, zero_division, PolyResult};
use crate::numbers::*;

use super::densebasic::{dup_degree, dup_from_dict, dup_strip};
use super::domains::domain::Domain;
use super::domains::fractionfield::{FracElement, FractionField};
use super::domains::polynomialring::PolynomialRing;
use super::domains::rationalfield::QQ;
use super::domains::sparsefractionfield::SparseFractionField;
use super::domains::sparsepolynomialring::SparsePolynomialRing;
use super::polyutils::dict_from_expr;
use super::sqfreetools::dup_sqf_list;

/// A matrix, by rows.
pub type Matrix<T> = Vec<Vec<T>>;

type Frac<K> = FracElement<<K as Domain>::Elem>;
type Poly<K> = Vec<<K as Domain>::Elem>;

/// An element of a [`FunctionField`]: its coordinates in the power basis
/// `1, Y, ..., Y^(n-1)` of the integral generator over `K(x)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Element<K: Domain> {
    pub coefficients: Vec<Frac<K>>,
}

/// The algebraic function field `K(x)[y]/(F)` of the curve `F(x, y) = 0`
/// (`FunctionField`); when the leading coefficient `a_n(x)` is not
/// constant the generator is `Y = a_n y`, and `unit` records `a_n`.
pub struct FunctionField<K: Domain> {
    pub x: Basic,
    pub y: Basic,
    pub k: K,
    pub r: PolynomialRing<K>,
    pub kx: FractionField<K>,
    pub n: usize,
    /// The leading coefficient `a_n` of the curve.
    pub unit: Poly<K>,
    /// The monic polynomial of `Y` over `K[x]`, highest degree first.
    pub f: Vec<Frac<K>>,
    /// The coefficients of the curve in `y`, highest degree first.
    pub curve: Vec<Poly<K>>,
    dy: RefCell<Option<Element<K>>>,
    power_traces_cache: RefCell<Option<Vec<Frac<K>>>>,
    integral_basis_cache: RefCell<Option<Vec<Element<K>>>>,
    local_at_infinity_cache: RefCell<Option<Vec<Element<K>>>>,
    normal_cache: RefCell<Option<Vec<Element<K>>>>,
}

fn name_of(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Wild(w) => w.symbol.name.clone(),
        _ => s.to_string(),
    }
}

// ---------------------------------------------------------------------
// polynomials with coefficients in K(x), low degree first

fn poly_mul<K: Domain>(a: &[Frac<K>], b: &[Frac<K>], kx: &FractionField<K>) -> Vec<Frac<K>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<Frac<K>> = vec![kx.zero(); a.len() + b.len() - 1];
    for (i, c) in a.iter().enumerate() {
        if c.numer.is_empty() {
            continue;
        }
        for (j, d) in b.iter().enumerate() {
            if !d.numer.is_empty() {
                result[i + j] = kx.add(&result[i + j], &kx.mul(c, d));
            }
        }
    }
    result
}

fn strip<K: Domain>(a: &[Frac<K>]) -> Vec<Frac<K>> {
    let mut end: usize = a.len();
    while end > 0 && a[end - 1].numer.is_empty() {
        end -= 1;
    }
    a[..end].to_vec()
}

fn poly_sub<K: Domain>(a: &[Frac<K>], b: &[Frac<K>], kx: &FractionField<K>) -> Vec<Frac<K>> {
    let n: usize = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = if i < a.len() { a[i].clone() } else { kx.zero() };
            let y = if i < b.len() { b[i].clone() } else { kx.zero() };
            kx.sub(&x, &y)
        })
        .collect()
}

fn poly_divmod<K: Domain>(
    a: &[Frac<K>],
    b: &[Frac<K>],
    kx: &FractionField<K>,
) -> PolyResult<(Vec<Frac<K>>, Vec<Frac<K>>)> {
    let b: Vec<Frac<K>> = strip::<K>(b);
    let mut r: Vec<Frac<K>> = strip::<K>(a);
    if b.is_empty() {
        return Err(zero_division("division by zero"));
    }
    let lead: Frac<K> = b.last().expect("a coefficient").clone();
    let mut q: Vec<Frac<K>> = vec![kx.zero(); (r.len() + 1).saturating_sub(b.len()).max(1)];
    while r.len() >= b.len() {
        let k: usize = r.len() - b.len();
        let c: Frac<K> = kx.quo(r.last().expect("a coefficient"), &lead)?;
        q[k] = c.clone();
        for (i, coefficient) in b.iter().enumerate() {
            r[k + i] = kx.sub(&r[k + i], &kx.mul(&c, coefficient));
        }
        r = if r.last().expect("a coefficient").numer.is_empty() {
            strip::<K>(&r[..r.len() - 1])
        } else {
            strip::<K>(&r)
        };
    }
    Ok((q, r))
}

/// The derivative of an element of `K(x)`.
fn derivative_of_fraction<K: Domain>(c: &Frac<K>, kx: &FractionField<K>) -> PolyResult<Frac<K>> {
    if c.numer.is_empty() {
        return Ok(c.clone());
    }
    let r = &kx.ring;
    let (n, d) = (&c.numer, &c.denom);
    kx.new_element(r.sub(&r.mul(&r.diff(n), d), &r.mul(n, &r.diff(d))), r.mul(d, d))
}

impl<K: Domain> FunctionField<K> {
    /// The function field of the curve of the coefficients in `y` (highest
    /// degree first, polynomials in `x` over `K`, highest degree first).
    pub fn new(coefficients: &[Poly<K>], x: &Basic, y: &Basic, k: K) -> PolyResult<FunctionField<K>> {
        let n: usize = coefficients.len().saturating_sub(1);
        if n < 1 || coefficients[0].is_empty() {
            return Err(value_error("the curve has no root y".to_string()));
        }
        let r: PolynomialRing<K> = PolynomialRing::new(k.clone(), &name_of(x));
        let kx: FractionField<K> = FractionField::new(r.clone());
        let leading: Poly<K> = dup_strip(&coefficients[0], &k);
        let mut monic: Vec<Poly<K>> = vec![r.one()];
        for (i, c) in coefficients.iter().enumerate().skip(1) {
            monic.push(r.mul(&dup_strip(c, &k), &r.pow(&leading, (i - 1) as u64)));
        }
        let f: Vec<Frac<K>> = monic.iter().map(|c| kx.from_ring(c)).collect();
        let curve: Vec<Poly<K>> = coefficients.iter().map(|c| dup_strip(c, &k)).collect();
        Ok(FunctionField {
            x: x.clone(),
            y: y.clone(),
            k,
            r,
            kx,
            n,
            unit: leading,
            f,
            curve,
            dy: RefCell::new(None),
            power_traces_cache: RefCell::new(None),
            integral_basis_cache: RefCell::new(None),
            local_at_infinity_cache: RefCell::new(None),
            normal_cache: RefCell::new(None),
        })
    }

    /// The polynomial of the curve as an expression (`original`).
    pub fn original(&self) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (k, c) in self.curve.iter().enumerate() {
            if !c.is_empty() {
                parts.push(mul(&[
                    self.r.to_expr(c)?,
                    pow(&self.y, &Basic::int((self.n - k) as i64))?,
                ])?);
            }
        }
        add(&parts)
    }

    // -- elements

    /// The element of the coordinates, padded or cut to `n`.
    pub fn element_of(&self, coefficients: Vec<Frac<K>>) -> Element<K> {
        let mut padded: Vec<Frac<K>> = coefficients;
        padded.resize(self.n, self.kx.zero());
        padded.truncate(self.n);
        Element { coefficients: padded }
    }

    pub fn zero(&self) -> Element<K> {
        self.element_of(Vec::new())
    }

    pub fn one(&self) -> Element<K> {
        self.element_of(vec![self.kx.one()])
    }

    /// The integral generator `Y = a_n y`.
    pub fn generator(&self) -> Element<K> {
        self.element_of(vec![self.kx.zero(), self.kx.one()])
    }

    /// The element of `K(x)` given by an expression.
    pub fn constant(&self, c: &Basic) -> PolyResult<Element<K>> {
        Ok(self.element_of(vec![self.kx.from_expr(c)?]))
    }

    /// The element given by a rational expression in `x` and `y`.
    pub fn element(&self, expr: &Basic) -> PolyResult<Element<K>> {
        if !expr.free_symbols().contains(&self.y) {
            return self.constant(expr);
        }
        if *expr == self.y {
            // y = Y/a_n
            let inverse: Frac<K> = self.kx.invert(&self.kx.from_ring(&self.unit))?;
            return Ok(self.element_of(vec![self.kx.zero(), inverse]));
        }
        match expr.head() {
            Head::Add => {
                let mut total: Element<K> = self.zero();
                for a in expr.args() {
                    total = self.add(&total, &self.element(a)?);
                }
                Ok(total)
            }
            Head::Mul => {
                let mut product: Element<K> = self.one();
                for a in expr.args() {
                    product = self.mul(&product, &self.element(a)?);
                }
                Ok(product)
            }
            Head::Pow if is_integer_class(&expr.args()[1]) => {
                let base: Element<K> = self.element(&expr.args()[0])?;
                let k: &Int = expr.args()[1].as_rational().expect("an integer").numerator();
                let e: u64 = u64::try_from(&abs(k)).map_err(|_| value_error("a huge exponent".to_string()))?;
                if is_negative(k) {
                    Ok(self.power(&self.invert(&base)?, e))
                } else {
                    Ok(self.power(&base, e))
                }
            }
            _ => Err(value_error(format!(
                "{} is not a rational function of {} and {}",
                expr, self.x, self.y
            ))),
        }
    }

    // -- arithmetic

    pub fn add(&self, a: &Element<K>, b: &Element<K>) -> Element<K> {
        self.element_of(
            a.coefficients
                .iter()
                .zip(&b.coefficients)
                .map(|(x, y)| self.kx.add(x, y))
                .collect(),
        )
    }

    pub fn sub(&self, a: &Element<K>, b: &Element<K>) -> Element<K> {
        self.element_of(
            a.coefficients
                .iter()
                .zip(&b.coefficients)
                .map(|(x, y)| self.kx.sub(x, y))
                .collect(),
        )
    }

    pub fn neg(&self, a: &Element<K>) -> Element<K> {
        self.element_of(a.coefficients.iter().map(|x| self.kx.neg(x)).collect())
    }

    /// The element times `c` of `K(x)`.
    pub fn scaled(&self, a: &Element<K>, c: &Frac<K>) -> Element<K> {
        self.element_of(a.coefficients.iter().map(|x| self.kx.mul(c, x)).collect())
    }

    pub fn mul(&self, a: &Element<K>, b: &Element<K>) -> Element<K> {
        self.element_of(self.reduce(&poly_mul::<K>(&a.coefficients, &b.coefficients, &self.kx)))
    }

    pub fn power(&self, a: &Element<K>, mut k: u64) -> Element<K> {
        let mut result: Element<K> = self.one();
        let mut base: Element<K> = a.clone();
        while k > 0 {
            if k % 2 == 1 {
                result = self.mul(&result, &base);
            }
            k /= 2;
            if k > 0 {
                base = self.mul(&base, &base);
            }
        }
        result
    }

    pub fn is_zero(&self, a: &Element<K>) -> bool {
        a.coefficients.iter().all(|c| c.numer.is_empty())
    }

    pub fn divide(&self, a: &Element<K>, b: &Element<K>) -> PolyResult<Element<K>> {
        Ok(self.mul(a, &self.invert(b)?))
    }

    /// The coefficients (low degree first) reduced modulo `F`.
    pub fn reduce(&self, coefficients: &[Frac<K>]) -> Vec<Frac<K>> {
        let kx = &self.kx;
        let n: usize = self.n;
        let mut work: Vec<Frac<K>> = coefficients.to_vec();
        let lower: Vec<Frac<K>> = self.f[1..].iter().rev().cloned().collect(); // a_0, ..., a_{n-1}
        if work.len() > n {
            for k in (n..work.len()).rev() {
                let c: Frac<K> = work[k].clone();
                if c.numer.is_empty() {
                    continue;
                }
                work[k] = kx.zero();
                for (i, a) in lower.iter().enumerate() {
                    if !a.numer.is_empty() {
                        work[k - n + i] = kx.sub(&work[k - n + i], &kx.mul(&c, a));
                    }
                }
            }
        }
        let mut result: Vec<Frac<K>> = work.into_iter().take(n).collect();
        result.resize(n, kx.zero());
        result
    }

    /// The inverse: the extended Euclidean algorithm modulo `F`.
    pub fn invert(&self, e: &Element<K>) -> PolyResult<Element<K>> {
        let kx = &self.kx;
        let f: Vec<Frac<K>> = strip::<K>(&e.coefficients);
        if f.is_empty() {
            return Err(zero_division("inverse of zero"));
        }
        let g: Vec<Frac<K>> = self.f.iter().rev().cloned().collect();
        let (mut r0, mut r1) = (g, f);
        let (mut s0, mut s1) = (vec![kx.zero()], vec![kx.one()]);
        while r1.len() > 1 {
            let (q, r) = poly_divmod::<K>(&r0, &r1, kx)?;
            r0 = std::mem::replace(&mut r1, strip::<K>(&r));
            let next: Vec<Frac<K>> = strip::<K>(&poly_sub::<K>(&s0, &poly_mul::<K>(&q, &s1, kx), kx));
            s0 = std::mem::replace(&mut s1, next);
            if r1.is_empty() {
                return Err(zero_division("not invertible: the curve is reducible"));
            }
        }
        let c: Frac<K> = r1[0].clone();
        let mut inverse: Vec<Frac<K>> = Vec::new();
        for a in &s1 {
            inverse.push(kx.quo(a, &c)?);
        }
        Ok(self.element_of(self.reduce(&inverse)))
    }

    /// The derivative with respect to `x` (`y' = -F_x/F_y`).
    pub fn derivative(&self, e: &Element<K>) -> PolyResult<Element<K>> {
        let kx = &self.kx;
        let mut plain: Vec<Frac<K>> = Vec::new();
        for c in &e.coefficients {
            plain.push(derivative_of_fraction(c, kx)?);
        }
        let chain: Vec<Frac<K>> = (1..self.n)
            .map(|i| kx.mul(&e.coefficients[i], &kx.from_int(i as i64)))
            .collect();
        let plain: Element<K> = self.element_of(plain);
        let chain: Element<K> = self.element_of(chain);
        Ok(self.add(&plain, &self.mul(&chain, &self.derivative_of_y()?)))
    }

    /// `Y' = -F_x(Y)/F_Y(Y)` for the integral generator (cached).
    pub fn derivative_of_y(&self) -> PolyResult<Element<K>> {
        if let Some(dy) = self.dy.borrow().as_ref() {
            return Ok(dy.clone());
        }
        let kx = &self.kx;
        let n: usize = self.n;
        let low: Vec<Frac<K>> = self.f.iter().rev().cloned().collect(); // a_0 ... a_n
        let mut fx: Vec<Frac<K>> = Vec::new();
        for a in &low[..n] {
            fx.push(derivative_of_fraction(a, kx)?);
        }
        let fy: Vec<Frac<K>> = (1..=n).map(|k| kx.mul(&low[k], &kx.from_int(k as i64))).collect();
        let fx: Element<K> = self.element_of(fx);
        let fy: Element<K> = self.element_of(fy);
        let dy: Element<K> = self.neg(&self.mul(&fx, &self.invert(&fy)?));
        *self.dy.borrow_mut() = Some(dy.clone());
        Ok(dy)
    }

    /// `Tr(Y^k)` for `k < 2n - 1` by Newton's identities (cached).
    pub fn power_traces(&self) -> Vec<Frac<K>> {
        if let Some(p) = self.power_traces_cache.borrow().as_ref() {
            return p.clone();
        }
        let kx = &self.kx;
        let n: usize = self.n;
        let a: Vec<Frac<K>> = self.f.iter().rev().cloned().collect();
        let mut p: Vec<Frac<K>> = vec![kx.from_int(n as i64)];
        for k in 1..(2 * n - 1) {
            let mut total: Frac<K> = kx.zero();
            for i in 1..=(k.saturating_sub(1)).min(n) {
                total = kx.sub(&total, &kx.mul(&a[n - i], &p[k - i]));
            }
            if k <= n {
                total = kx.sub(&total, &kx.mul(&kx.from_int(k as i64), &a[n - k]));
            }
            p.push(total);
        }
        *self.power_traces_cache.borrow_mut() = Some(p.clone());
        p
    }

    pub fn trace(&self, e: &Element<K>) -> Frac<K> {
        let p: Vec<Frac<K>> = self.power_traces();
        let kx = &self.kx;
        let mut total: Frac<K> = kx.zero();
        for (k, c) in e.coefficients.iter().enumerate() {
            if !c.numer.is_empty() {
                total = kx.add(&total, &kx.mul(c, &p[k]));
            }
        }
        total
    }

    // -- bases

    /// The coordinates of `e` in `basis` (elements of `K(x)`).
    pub fn coordinates(&self, e: &Element<K>, basis: &[Element<K>]) -> PolyResult<Vec<Frac<K>>> {
        let rows: Matrix<Frac<K>> = basis.iter().map(|w| w.coefficients.clone()).collect();
        let inverse: Matrix<Frac<K>> = inverse_matrix(&rows, &self.kx)?;
        let kx = &self.kx;
        let mut result: Vec<Frac<K>> = Vec::new();
        for j in 0..self.n {
            let mut total: Frac<K> = kx.zero();
            for (i, c) in e.coefficients.iter().enumerate() {
                if !c.numer.is_empty() && !inverse[i][j].numer.is_empty() {
                    total = kx.add(&total, &kx.mul(c, &inverse[i][j]));
                }
            }
            result.push(total);
        }
        Ok(result)
    }

    /// `det(Tr(w_i w_j))`.
    pub fn discriminant(&self, basis: &[Element<K>]) -> PolyResult<Frac<K>> {
        let traces: Matrix<Frac<K>> = basis
            .iter()
            .map(|u| basis.iter().map(|v| self.trace(&self.mul(u, v))).collect())
            .collect();
        determinant_over_field(&traces, &self.kx)
    }

    /// Whether `e` is integral over `K[x]`.
    pub fn is_integral(&self, e: &Element<K>) -> bool {
        let kx = &self.kx;
        let rows: Matrix<Frac<K>> = (0..self.n)
            .map(|k| {
                let mut unit: Vec<Frac<K>> = vec![kx.zero(); k];
                unit.push(kx.one());
                self.mul(e, &self.element_of(unit)).coefficients
            })
            .collect();
        charpoly(&rows, kx).iter().all(|c| is_polynomial::<K>(c))
    }

    /// An integral basis in Hermite normal form (Trager's algorithm,
    /// cached).
    pub fn integral_basis(&self) -> PolyResult<Vec<Element<K>>> {
        if let Some(b) = self.integral_basis_cache.borrow().as_ref() {
            return Ok(b.clone());
        }
        let kx = &self.kx;
        let mut basis: Vec<Element<K>> = (0..self.n)
            .map(|k| {
                let mut unit: Vec<Frac<K>> = vec![kx.zero(); k];
                unit.push(kx.one());
                self.element_of(unit)
            })
            .collect();
        while let Some(enlarged) = self.round2_step(&basis)? {
            basis = enlarged;
        }
        let result: Vec<Element<K>> = self.triangular(&basis)?;
        *self.integral_basis_cache.borrow_mut() = Some(result.clone());
        Ok(result)
    }

    fn round2_step(&self, basis: &[Element<K>]) -> PolyResult<Option<Vec<Element<K>>>> {
        let (r, kx, k, n) = (&self.r, &self.kx, &self.k, self.n);
        let mut traces: Matrix<Poly<K>> = Vec::new();
        for u in basis {
            let mut row: Vec<Poly<K>> = Vec::new();
            for v in basis {
                row.push(as_polynomial(&self.trace(&self.mul(u, v)), r)?);
            }
            traces.push(row);
        }
        let disc: Poly<K> = determinant(&traces, r)?;
        if disc.is_empty() {
            return Err(value_error("the curve is not squarefree".to_string()));
        }
        let mut q: Poly<K> = r.one();
        for (factor, multiplicity) in dup_sqf_list(&disc, k).1 {
            if multiplicity >= 2 {
                q = r.mul(&q, &factor);
            }
        }
        if dup_degree(&q) < 1 {
            return Ok(None);
        }
        // the trace radical: the kernel over K[x] of [T | -Q I]
        let augmented: Matrix<Poly<K>> = traces
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut full: Vec<Poly<K>> = row.clone();
                full.extend((0..n).map(|j| if i == j { r.neg(&q) } else { r.zero() }));
                full
            })
            .collect();
        let (_, u, pivots) = column_echelon(&augmented, r, k)?;
        let kernel: Matrix<Poly<K>> = (0..2 * n)
            .filter(|c| !pivots.contains(c))
            .map(|c| (0..2 * n).map(|row| u[row][c].clone()).collect())
            .collect();
        if kernel.len() != n {
            return Err(value_error("the trace radical has the wrong rank".to_string()));
        }
        let a: Matrix<Poly<K>> = kernel.iter().map(|v| v[..n].to_vec()).collect();
        let j: Vec<Element<K>> = kernel.iter().map(|v| self.combination(&v[..n], basis)).collect();
        let det_a: Poly<K> = determinant(&a, r)?;
        let adj_a: Matrix<Poly<K>> = adjugate(&a, r, kx)?;
        let mut stacked: Matrix<Poly<K>> = vec![Vec::new(); n];
        for j_k in &j {
            let mut m_k: Matrix<Poly<K>> = Vec::new();
            for w in basis {
                let mut row: Vec<Poly<K>> = Vec::new();
                for c in self.coordinates(&self.mul(w, j_k), basis)? {
                    row.push(as_polynomial(&c, r)?);
                }
                m_k.push(row);
            }
            let product: Matrix<Poly<K>> = matrix_product(&m_k, &adj_a, r);
            for (i, row) in product.into_iter().enumerate() {
                stacked[i].extend(row);
            }
        }
        let (h, _, pivots) = column_echelon(&stacked, r, k)?;
        let h_square: Matrix<Poly<K>> = (0..n)
            .map(|i| pivots.iter().map(|&c| h[i][c].clone()).collect())
            .collect();
        let det_h: Poly<K> = determinant(&h_square, r)?;
        let adj_h: Matrix<Poly<K>> = adjugate(&h_square, r, kx)?;
        let rows: Matrix<Poly<K>> = (0..n)
            .map(|i| (0..n).map(|jj| r.mul(&det_a, &adj_h[i][jj])).collect())
            .collect();
        let rows: Matrix<Poly<K>> = row_hermite(&rows, r, k)?;
        let mut new_basis: Vec<Element<K>> = Vec::new();
        let mut fractions: Matrix<Frac<K>> = Vec::new();
        for row in &rows {
            let mut vector: Vec<Frac<K>> = Vec::new();
            for c in row {
                vector.push(kx.new_element(c.clone(), det_h.clone())?);
            }
            new_basis.push(self.combination_fraction(&vector, basis));
            fractions.push(vector);
        }
        let change: Frac<K> = determinant_over_field(&fractions, kx)?;
        if degree_of_fraction::<K>(&change) == Some(0) {
            return Ok(None);
        }
        Ok(Some(new_basis))
    }

    /// The basis of the same module in triangular form (`_triangular`).
    fn triangular(&self, basis: &[Element<K>]) -> PolyResult<Vec<Element<K>>> {
        let (r, kx) = (&self.r, &self.kx);
        let mut denominator: Poly<K> = r.one();
        for w in basis {
            for c in &w.coefficients {
                denominator = r.lcm(&denominator, &c.denom);
            }
        }
        let scale: Frac<K> = kx.from_ring(&denominator);
        let mut rows: Matrix<Poly<K>> = Vec::new();
        for w in basis {
            let mut row: Vec<Poly<K>> = Vec::new();
            for c in &w.coefficients {
                row.push(as_polynomial(&kx.mul(c, &scale), r)?);
            }
            rows.push(row);
        }
        let rows: Matrix<Poly<K>> = row_hermite(&rows, r, &self.k)?;
        let mut result: Vec<Element<K>> = Vec::new();
        for row in rows {
            let mut coefficients: Vec<Frac<K>> = Vec::new();
            for c in row {
                coefficients.push(if c.is_empty() {
                    kx.zero()
                } else {
                    kx.new_element(c, denominator.clone())?
                });
            }
            result.push(self.element_of(coefficients));
        }
        // stable, by the highest power of Y present
        result.sort_by_key(|w| w.coefficients.iter().rposition(|c| !c.numer.is_empty()).unwrap_or(0));
        Ok(result)
    }

    /// A basis of the elements integral at infinity (cached).
    pub fn local_basis_at_infinity(&self) -> PolyResult<Vec<Element<K>>> {
        if let Some(b) = self.local_at_infinity_cache.borrow().as_ref() {
            return Ok(b.clone());
        }
        let (n, k, r, kx) = (self.n, &self.k, &self.r, &self.kx);
        let low: Vec<Poly<K>> = self.f.iter().rev().map(|c| c.numer.clone()).collect(); // a_0 ... a_n
        let mut m: i64 = 0;
        for (i, a) in low.iter().enumerate().take(n) {
            if !a.is_empty() {
                let d: i64 = dup_degree(a) as i64;
                let parts: i64 = (n - i) as i64;
                m = m.max(-((-d).div_euclid(parts)));
            }
        }
        let mut g: Vec<Poly<K>> = Vec::new();
        for i in (0..=n).rev() {
            let mut reversed: Poly<K> = low[i].iter().rev().cloned().collect();
            let length: usize = (m as usize) * (n - i) + 1;
            while reversed.len() < length {
                reversed.push(k.zero());
            }
            g.push(dup_strip(&reversed, k));
        }
        let other: FunctionField<K> = FunctionField::new(&g, &self.x, &self.y, k.clone())?;
        let mut result: Vec<Element<K>> = Vec::new();
        for w in other.integral_basis()? {
            let mut coefficients: Vec<Frac<K>> = Vec::new();
            for (i, c) in w.coefficients.iter().enumerate() {
                if c.numer.is_empty() {
                    coefficients.push(kx.zero());
                    continue;
                }
                let exponent: i64 = dup_degree(&c.denom) as i64 - dup_degree(&c.numer) as i64 - m * i as i64;
                let mut numerator: Poly<K> = dup_strip(&c.numer.iter().rev().cloned().collect::<Vec<_>>(), k);
                let mut denominator: Poly<K> = dup_strip(&c.denom.iter().rev().cloned().collect::<Vec<_>>(), k);
                if exponent >= 0 {
                    numerator = r.mul(&numerator, &monomial(exponent as usize, k));
                } else {
                    denominator = r.mul(&denominator, &monomial((-exponent) as usize, k));
                }
                coefficients.push(kx.new_element(numerator, denominator)?);
            }
            result.push(self.element_of(coefficients));
        }
        *self.local_at_infinity_cache.borrow_mut() = Some(result.clone());
        Ok(result)
    }

    /// The least `k` with `x^(-k) e` integral at infinity.
    pub fn pole_order_at_infinity(&self, e: &Element<K>) -> PolyResult<i64> {
        let local: Vec<Element<K>> = self.local_basis_at_infinity()?;
        let orders: Vec<i64> = self
            .coordinates(e, &local)?
            .iter()
            .filter_map(|c| degree_of_fraction::<K>(c))
            .collect();
        orders
            .into_iter()
            .max()
            .ok_or_else(|| value_error("the pole order of zero".to_string()))
    }

    /// An integral basis normal at infinity (Trager, cached).
    pub fn normal_basis(&self) -> PolyResult<Vec<Element<K>>> {
        if let Some(b) = self.normal_cache.borrow().as_ref() {
            return Ok(b.clone());
        }
        let mut basis: Vec<Element<K>> = self.integral_basis()?;
        let local: Vec<Element<K>> = self.local_basis_at_infinity()?;
        let (k, kx, n) = (&self.k, &self.kx, self.n);
        loop {
            let mut orders: Vec<i64> = Vec::new();
            for w in &basis {
                orders.push(self.pole_order_at_infinity(w)?);
            }
            let mut leading: Matrix<K::Elem> = Vec::new();
            for (w, order) in basis.iter().zip(&orders) {
                let mut row: Vec<K::Elem> = Vec::new();
                for c in self.coordinates(w, &local)? {
                    row.push(value_at_infinity::<K>(&c, *order, k)?);
                }
                leading.push(row);
            }
            let transposed: Matrix<K::Elem> = (0..n)
                .map(|j| (0..n).map(|i| leading[i][j].clone()).collect())
                .collect();
            let kernel: Vec<Vec<K::Elem>> = k.nullspace(&transposed, n)?;
            let Some(vector) = kernel.first() else { break };
            let top: i64 = (0..n)
                .filter(|&i| !k.is_zero(&vector[i]))
                .map(|i| orders[i])
                .max()
                .expect("a nonzero entry");
            let replaced: usize = (0..n)
                .filter(|&i| !k.is_zero(&vector[i]) && orders[i] == top)
                .max()
                .expect("a nonzero entry");
            let mut combination: Element<K> = self.zero();
            for i in 0..n {
                if !k.is_zero(&vector[i]) {
                    let factor: Frac<K> = kx.from_ring(&monomial_times(&vector[i], (top - orders[i]) as usize, k));
                    combination = self.add(&combination, &self.scaled(&basis[i], &factor));
                }
            }
            basis[replaced] = combination;
        }
        *self.normal_cache.borrow_mut() = Some(basis.clone());
        Ok(basis)
    }

    pub fn pole_orders_at_infinity(&self, basis: &[Element<K>]) -> PolyResult<Vec<i64>> {
        basis.iter().map(|w| self.pole_order_at_infinity(w)).collect()
    }

    fn combination(&self, vector: &[Poly<K>], basis: &[Element<K>]) -> Element<K> {
        let mut total: Element<K> = self.zero();
        for (c, w) in vector.iter().zip(basis) {
            if !c.is_empty() {
                total = self.add(&total, &self.scaled(w, &self.kx.from_ring(c)));
            }
        }
        total
    }

    fn combination_fraction(&self, vector: &[Frac<K>], basis: &[Element<K>]) -> Element<K> {
        let mut total: Element<K> = self.zero();
        for (c, w) in vector.iter().zip(basis) {
            if !c.numer.is_empty() {
                total = self.add(&total, &self.scaled(w, c));
            }
        }
        total
    }

    /// The element as an expression in `x` and the generator `y` of the
    /// curve as given (`as_expr`).
    pub fn as_expr(&self, e: &Element<K>) -> PolyResult<Basic> {
        let r = &self.r;
        let mut denominator: Poly<K> = r.one();
        for c in &e.coefficients {
            denominator = r.lcm(&denominator, &c.denom);
        }
        let y_integral: Basic = mul(&[r.to_expr(&self.unit)?, self.y.clone()])?;
        let mut terms: Vec<Basic> = Vec::new();
        for (i, c) in e.coefficients.iter().enumerate() {
            if !c.numer.is_empty() {
                let numerator: Poly<K> = r.exquo(&r.mul(&c.numer, &denominator), &c.denom)?;
                terms.push(mul(&[
                    r.to_expr(&numerator)?,
                    pow(&y_integral, &Basic::int(i as i64))?,
                ])?);
            }
        }
        if terms.is_empty() {
            return Ok(Basic::zero());
        }
        div(&expand_default(&add(&terms)?)?, &r.to_expr(&denominator)?)
    }
}

fn monomial<K: Domain>(k: usize, d: &K) -> Poly<K> {
    let mut m: Poly<K> = vec![d.one()];
    m.extend(std::iter::repeat_with(|| d.zero()).take(k));
    m
}

fn monomial_times<K: Domain>(c: &K::Elem, k: usize, d: &K) -> Poly<K> {
    let mut m: Poly<K> = vec![c.clone()];
    m.extend(std::iter::repeat_with(|| d.zero()).take(k));
    m
}

/// The symbols of `f` other than `x` and `y`, sorted by name
/// (`parameters_of`).
pub fn parameters_of(f: &Basic, x: &Basic, y: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = f.free_symbols().into_iter().filter(|s| s != x && s != y).collect();
    found.sort_by_key(name_of);
    found
}

/// The function field of the curve `F(x, y) = 0` with coefficients in `K`
/// (`function_field`).
pub fn function_field<K: Domain>(f: &Basic, x: &Basic, y: &Basic, k: K) -> PolyResult<FunctionField<K>> {
    let terms = dict_from_expr(f, &[name_of(y), name_of(x)], &k)?;
    let n: usize = terms.keys().map(|m| m[0] as usize).max().unwrap_or(0);
    if n < 1 {
        return Err(value_error(format!("{} has no root {}", f, y)));
    }
    let mut dense: Vec<BTreeMap<usize, K::Elem>> = vec![BTreeMap::new(); n + 1];
    for (m, c) in terms {
        dense[n - m[0] as usize].insert(m[1] as usize, c);
    }
    let coefficients: Vec<Poly<K>> = dense.iter().map(|d| dup_from_dict(d, &k)).collect();
    FunctionField::new(&coefficients, x, y, k)
}

// ---------------------------------------------------------------------
// polynomial matrices over K[x]

fn is_polynomial<K: Domain>(c: &Frac<K>) -> bool {
    dup_degree(&c.denom) <= 0
}

/// An element of `K(x)` which is a polynomial, as an element of `K[x]`.
fn as_polynomial<K: Domain>(c: &Frac<K>, r: &PolynomialRing<K>) -> PolyResult<Poly<K>> {
    if dup_degree(&c.denom) > 0 {
        return Err(value_error(format!(
            "{}/({}) is not a polynomial",
            r.to_expr(&c.numer)?,
            r.to_expr(&c.denom)?
        )));
    }
    if r.is_one(&c.denom) {
        return Ok(c.numer.clone());
    }
    Ok(r.mul(&c.numer, &vec![r.ground.invert(&c.denom[0])?]))
}

fn degree_of_fraction<K: Domain>(c: &Frac<K>) -> Option<i64> {
    if c.numer.is_empty() {
        return None;
    }
    Some(dup_degree(&c.numer) as i64 - dup_degree(&c.denom) as i64)
}

fn value_at_infinity<K: Domain>(c: &Frac<K>, k: i64, d: &K) -> PolyResult<K::Elem> {
    if c.numer.is_empty() || (dup_degree(&c.numer) as i64 - dup_degree(&c.denom) as i64) < k {
        return Ok(d.zero());
    }
    d.quo(&c.numer[0], &c.denom[0])
}

type Echelon<K> = (Matrix<Poly<K>>, Matrix<Poly<K>>, Vec<usize>);

/// `(H, U, pivots)` with `M U = H` in column echelon form, `U` unimodular
/// over `K[x]` (`_column_echelon`).
fn column_echelon<K: Domain>(m: &Matrix<Poly<K>>, r: &PolynomialRing<K>, k: &K) -> PolyResult<Echelon<K>> {
    let rows: usize = m.len();
    let columns: usize = m.first().map_or(0, |row| row.len());
    let mut h: Matrix<Poly<K>> = m.clone();
    let mut u: Matrix<Poly<K>> = (0..columns)
        .map(|i| (0..columns).map(|j| if i == j { r.one() } else { r.zero() }).collect())
        .collect();
    let mut pivots: Vec<usize> = Vec::new();
    let mut current: usize = 0;
    for i in 0..rows {
        if current >= columns {
            break;
        }
        for j in (current + 1)..columns {
            while !h[i][j].is_empty() {
                if h[i][current].is_empty() {
                    swap_columns(&mut h, &mut u, current, j);
                    continue;
                }
                let q: Poly<K> = r.quo(&h[i][j], &h[i][current])?;
                add_column(&mut h, &mut u, j, current, &r.neg(&q), r);
                if !h[i][j].is_empty() {
                    swap_columns(&mut h, &mut u, current, j);
                }
            }
        }
        if !h[i][current].is_empty() {
            let lc: K::Elem = h[i][current][0].clone();
            if !k.is_one(&lc) {
                let inverse: Poly<K> = vec![k.invert(&lc)?];
                for row in h.iter_mut() {
                    row[current] = r.mul(&row[current], &inverse);
                }
                for row in u.iter_mut() {
                    row[current] = r.mul(&row[current], &inverse);
                }
            }
            pivots.push(current);
            current += 1;
        }
    }
    Ok((h, u, pivots))
}

fn swap_columns<T>(h: &mut Matrix<T>, u: &mut Matrix<T>, a: usize, b: usize) {
    for row in h.iter_mut() {
        row.swap(a, b);
    }
    for row in u.iter_mut() {
        row.swap(a, b);
    }
}

fn add_column<K: Domain>(
    h: &mut Matrix<Poly<K>>,
    u: &mut Matrix<Poly<K>>,
    target: usize,
    source: usize,
    factor: &Poly<K>,
    r: &PolynomialRing<K>,
) {
    for row in h.iter_mut().chain(u.iter_mut()) {
        if !row[source].is_empty() {
            row[target] = r.add(&row[target], &r.mul(factor, &row[source]));
        }
    }
}

/// The rows of a square matrix over `K[x]` spanning the same module, in
/// echelon form with monic pivots and the entries above them reduced
/// (`_row_hermite`).
fn row_hermite<K: Domain>(rows: &Matrix<Poly<K>>, r: &PolynomialRing<K>, k: &K) -> PolyResult<Matrix<Poly<K>>> {
    let transposed: Matrix<Poly<K>> = (0..rows[0].len())
        .map(|c| (0..rows.len()).map(|row| rows[row][c].clone()).collect())
        .collect();
    let (h, _, pivots) = column_echelon(&transposed, r, k)?;
    let mut result: Matrix<Poly<K>> = pivots
        .iter()
        .map(|&c| (0..h.len()).map(|row| h[row][c].clone()).collect())
        .collect();
    for i in 0..result.len() {
        let pivot_column: usize = result[i].iter().position(|v| !v.is_empty()).expect("a nonzero row");
        for j in 0..i {
            if !result[j][pivot_column].is_empty() {
                let q: Poly<K> = r.quo(&result[j][pivot_column], &result[i][pivot_column])?;
                let reduced: Vec<Poly<K>> = result[j]
                    .iter()
                    .zip(&result[i])
                    .map(|(a, b)| r.sub(a, &r.mul(&q, b)))
                    .collect();
                result[j] = reduced;
            }
        }
    }
    Ok(result)
}

/// The determinant over `K[x]`, fraction-free (Bareiss).
fn determinant<K: Domain>(m: &Matrix<Poly<K>>, r: &PolynomialRing<K>) -> PolyResult<Poly<K>> {
    let n: usize = m.len();
    if n == 0 {
        return Ok(r.one());
    }
    let mut a: Matrix<Poly<K>> = m.clone();
    let mut positive: bool = true;
    let mut previous: Poly<K> = r.one();
    for k in 0..n - 1 {
        if a[k][k].is_empty() {
            let Some(swap) = (k + 1..n).find(|&i| !a[i][k].is_empty()) else {
                return Ok(r.zero());
            };
            a.swap(k, swap);
            positive = !positive;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let value: Poly<K> = r.sub(&r.mul(&a[i][j], &a[k][k]), &r.mul(&a[i][k], &a[k][j]));
                a[i][j] = r.exquo(&value, &previous)?;
            }
            a[i][k] = r.zero();
        }
        previous = a[k][k].clone();
    }
    Ok(if positive {
        a[n - 1][n - 1].clone()
    } else {
        r.neg(&a[n - 1][n - 1])
    })
}

/// The determinant over a field (Gaussian elimination).
fn determinant_over_field<K: Domain>(m: &Matrix<Frac<K>>, f: &FractionField<K>) -> PolyResult<Frac<K>> {
    let n: usize = m.len();
    let mut a: Matrix<Frac<K>> = m.clone();
    let mut result: Frac<K> = f.one();
    for k in 0..n {
        let Some(pivot) = (k..n).find(|&i| !a[i][k].numer.is_empty()) else {
            return Ok(f.zero());
        };
        if pivot != k {
            a.swap(k, pivot);
            result = f.neg(&result);
        }
        result = f.mul(&result, &a[k][k]);
        let inverse: Frac<K> = f.invert(&a[k][k])?;
        for i in k + 1..n {
            if !a[i][k].numer.is_empty() {
                let factor: Frac<K> = f.mul(&a[i][k], &inverse);
                for j in k + 1..n {
                    if !a[k][j].numer.is_empty() {
                        a[i][j] = f.sub(&a[i][j], &f.mul(&factor, &a[k][j]));
                    }
                }
            }
        }
    }
    Ok(result)
}

/// The inverse of a square matrix over a field (Gauss-Jordan).
fn inverse_matrix<K: Domain>(m: &Matrix<Frac<K>>, f: &FractionField<K>) -> PolyResult<Matrix<Frac<K>>> {
    let n: usize = m.len();
    let mut a: Matrix<Frac<K>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut full: Vec<Frac<K>> = row.clone();
            full.extend((0..n).map(|j| if i == j { f.one() } else { f.zero() }));
            full
        })
        .collect();
    for k in 0..n {
        let Some(pivot) = (k..n).find(|&i| !a[i][k].numer.is_empty()) else {
            return Err(zero_division("the matrix is singular"));
        };
        a.swap(k, pivot);
        let inverse: Frac<K> = f.invert(&a[k][k])?;
        a[k] = a[k].iter().map(|v| f.mul(v, &inverse)).collect();
        for i in 0..n {
            if i != k && !a[i][k].numer.is_empty() {
                let factor: Frac<K> = a[i][k].clone();
                let pivot_row: Vec<Frac<K>> = a[k].clone();
                a[i] = a[i]
                    .iter()
                    .zip(&pivot_row)
                    .map(|(x, b)| {
                        if b.numer.is_empty() {
                            x.clone()
                        } else {
                            f.sub(x, &f.mul(&factor, b))
                        }
                    })
                    .collect();
            }
        }
    }
    Ok(a.into_iter().map(|row| row[n..].to_vec()).collect())
}

fn dot<K: Domain>(a: &[Frac<K>], b: &[Frac<K>], f: &FractionField<K>) -> Frac<K> {
    let mut total: Frac<K> = f.zero();
    for (u, v) in a.iter().zip(b) {
        if !u.numer.is_empty() && !v.numer.is_empty() {
            total = f.add(&total, &f.mul(u, v));
        }
    }
    total
}

/// The characteristic polynomial `det(t I - M)`, highest degree first
/// (Berkowitz).
fn charpoly<K: Domain>(m: &Matrix<Frac<K>>, f: &FractionField<K>) -> Vec<Frac<K>> {
    let n: usize = m.len();
    let mut poly: Vec<Frac<K>> = vec![f.one()];
    for k in 0..n {
        let a: &Frac<K> = &m[k][k];
        let row: &[Frac<K>] = &m[k][..k];
        let mut column: Vec<Frac<K>> = vec![f.one(), f.neg(a)];
        let mut vector: Vec<Frac<K>> = (0..k).map(|i| m[i][k].clone()).collect();
        for _ in 0..k {
            column.push(f.neg(&dot(row, &vector, f)));
            vector = (0..k).map(|i| dot(&m[i][..k], &vector, f)).collect();
        }
        let mut new: Vec<Frac<K>> = Vec::new();
        for i in 0..k + 2 {
            let mut total: Frac<K> = f.zero();
            for (j, p) in poly.iter().enumerate().take(i.min(k) + 1) {
                if i - j < column.len() {
                    total = f.add(&total, &f.mul(&column[i - j], p));
                }
            }
            new.push(total);
        }
        poly = new;
    }
    poly
}

/// The adjugate matrix `det(M) M^(-1)` (the cofactors when `M` is
/// singular).
fn adjugate<K: Domain>(
    m: &Matrix<Poly<K>>,
    r: &PolynomialRing<K>,
    kx: &FractionField<K>,
) -> PolyResult<Matrix<Poly<K>>> {
    let n: usize = m.len();
    let det: Poly<K> = determinant(m, r)?;
    if !det.is_empty() {
        let lifted: Matrix<Frac<K>> = m
            .iter()
            .map(|row| row.iter().map(|c| kx.from_ring(c)).collect())
            .collect();
        let inverse: Matrix<Frac<K>> = inverse_matrix(&lifted, kx)?;
        let scale: Frac<K> = kx.from_ring(&det);
        let mut result: Matrix<Poly<K>> = Vec::new();
        for row in inverse {
            let mut out: Vec<Poly<K>> = Vec::new();
            for c in row {
                out.push(as_polynomial(&kx.mul(&scale, &c), r)?);
            }
            result.push(out);
        }
        return Ok(result);
    }
    let mut result: Matrix<Poly<K>> = vec![vec![r.zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            let minor: Matrix<Poly<K>> = (0..n)
                .filter(|&row| row != j)
                .map(|row| (0..n).filter(|&c| c != i).map(|c| m[row][c].clone()).collect())
                .collect();
            let cofactor: Poly<K> = determinant(&minor, r)?;
            result[i][j] = if (i + j) % 2 == 0 { cofactor } else { r.neg(&cofactor) };
        }
    }
    Ok(result)
}

fn matrix_product<K: Domain>(a: &Matrix<Poly<K>>, b: &Matrix<Poly<K>>, r: &PolynomialRing<K>) -> Matrix<Poly<K>> {
    let (n, m, p) = (a.len(), b.len(), b[0].len());
    let mut result: Matrix<Poly<K>> = vec![vec![r.zero(); p]; n];
    for i in 0..n {
        for k in 0..m {
            if a[i][k].is_empty() {
                continue;
            }
            for j in 0..p {
                if !b[k][j].is_empty() {
                    result[i][j] = r.add(&result[i][j], &r.mul(&a[i][k], &b[k][j]));
                }
            }
        }
    }
    result
}

/// An integral basis of the function field of the curve `F(x, y) = 0`
/// over `QQ`, or the rational functions of the other symbols of `F`
/// (`integral_basis`).
pub fn integral_basis(f: &Basic, x: &Basic, y: &Basic) -> PolyResult<Vec<Basic>> {
    let parameters: Vec<Basic> = parameters_of(f, x, y);
    if !parameters.is_empty() {
        let k = SparseFractionField::new(SparsePolynomialRing::sympy(
            QQ,
            parameters.iter().map(name_of).collect(),
        ));
        let field = function_field(f, x, y, k)?;
        return field.integral_basis()?.iter().map(|w| field.as_expr(w)).collect();
    }
    let field = function_field(f, x, y, QQ)?;
    field.integral_basis()?.iter().map(|w| field.as_expr(w)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    fn shown(basis: &[Basic]) -> Vec<String> {
        basis.iter().map(sstr).collect()
    }

    #[test]
    fn docstring_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let y: Basic = Symbol::new("y", &[]).0;
        let p = |e: Basic, n: i64| pow(&e, &Basic::int(n)).unwrap();
        // y**2 - x**3: [1, y/x]
        let cusp: Basic = add(&[p(y.clone(), 2), mul(&[Basic::negative_one(), p(x.clone(), 3)]).unwrap()]).unwrap();
        assert_eq!(shown(&integral_basis(&cusp, &x, &y).unwrap()), vec!["1", "y/x"]);
        // x*y**2 - 1: [1, x*y]
        let hyperbola: Basic = add(&[mul(&[x.clone(), p(y.clone(), 2)]).unwrap(), Basic::negative_one()]).unwrap();
        assert_eq!(shown(&integral_basis(&hyperbola, &x, &y).unwrap()), vec!["1", "x*y"]);
    }
}
