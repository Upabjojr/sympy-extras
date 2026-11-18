//! Trager's algorithm for the integration of algebraic functions on an
//! arbitrary plane curve, ported from `sympy_extras.integrals.curves`.
//! This part: the radicals of the integrand and their algebra, the
//! primitive element with its minimal polynomial `F` (the curve), the
//! integrand as an element of the function field, the expressions of the
//! typed objects, and the change of variable `x = N + 1/z`.
//!
//! The rationals are the number field of degree one of
//! [`crate::polys::puiseux`], so that every coefficient is a list of
//! rationals.

// References: B. M. Trager, Integration of algebraic functions, PhD thesis, MIT, 1984;
// M. Bronstein, Symbolic integration tutorial, ISSAC 1998, section 2.

#![allow(clippy::mutable_key_type)]
// the matrix loops index as the Python ones do
#![allow(clippy::needless_range_loop)]

use std::collections::{BTreeMap, HashMap};

use crate::core::add::add;
use crate::core::basic::{Basic, Head};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::{value_error, PolyResult};
use crate::numbers::{Int, Rat};
use crate::polys::densearith::{dup_add, dup_mul, dup_mul_ground, dup_pow};
use crate::polys::densebasic::{dup_degree, dup_strip};
use crate::polys::domains::{AlgebraicField, Domain, FracElement, FractionField, PolynomialRing, QQ};
use crate::polys::functionfields::{inverse_matrix, Element, FunctionField};
use crate::polys::puiseux::{rationals, Curve, Number};

/// A polynomial over a number field, highest degree first.
pub type KPoly = Vec<Number>;
/// An element of `K(x)`.
pub type Fraction = FracElement<Number>;
/// The exponents of a monomial in the radicals.
pub type Monomial = Vec<u32>;
/// An element of the algebra of the radicals by its monomials.
pub type AlgebraElement = BTreeMap<Monomial, Fraction>;
type Matrix = Vec<Vec<Fraction>>;

/// The multiples of a residue divisor tried before giving up (Mazur's
/// bound for a curve of genus one over the rationals).
pub const TORSION_BOUND: usize = 12;
/// The degree beyond which the field of the residues is not built.
pub const RESIDUE_FIELD_DEGREE: usize = 24;

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn name_of(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Wild(w) => w.symbol.name.clone(),
        _ => s.to_string(),
    }
}

/// A radical `g = base^(1/q)` of the integrand, the base a rational
/// function of `x` and of the earlier generators (their symbols)
/// (`_Generator`).
#[derive(Debug, Clone)]
pub struct Generator {
    pub symbol: Basic,
    pub base: Basic,
    pub q: u32,
    pub original: Basic,
}

/// The algebra `QQ(x)[g_1, ..., g_m]/(g_i^q_i - base_i)` with the monomial
/// basis `prod g_i^e_i`, `e_i < q_i` (`_Algebra`).
pub struct Algebra {
    pub generators: Vec<Generator>,
    pub x: Basic,
    pub r: PolynomialRing<AlgebraicField>,
    pub kx: FractionField<AlgebraicField>,
    pub names: Vec<String>,
    pub dimension: usize,
    pub bases: Vec<AlgebraElement>,
    pub monomials: Vec<Monomial>,
    pub index: HashMap<Monomial, usize>,
}

impl Algebra {
    pub fn new(generators: Vec<Generator>, x: &Basic) -> PolyResult<Algebra> {
        let r = PolynomialRing::new(rationals(), &name_of(x));
        let kx = FractionField::new(r.clone());
        let names: Vec<String> = generators.iter().map(|g| name_of(&g.symbol)).collect();
        let dimension: usize = generators.iter().map(|g| g.q as usize).product();
        let mut algebra = Algebra {
            generators,
            x: x.clone(),
            r,
            kx,
            names,
            dimension,
            bases: Vec::new(),
            monomials: Vec::new(),
            index: HashMap::new(),
        };
        for k in 0..algebra.generators.len() {
            // the base in the earlier generators only: their bases are
            // already there for the reduction, as Python appends them
            let base: Basic = algebra.generators[k].base.clone();
            let element: AlgebraElement = algebra.from_expr(&base)?;
            algebra.bases.push(element);
        }
        let mut monomials: Vec<Monomial> = Vec::new();
        algebra.enumerate(Vec::new(), 0, &mut monomials);
        algebra.index = monomials.iter().enumerate().map(|(i, m)| (m.clone(), i)).collect();
        algebra.monomials = monomials;
        Ok(algebra)
    }

    fn enumerate(&self, prefix: Monomial, k: usize, monomials: &mut Vec<Monomial>) {
        if k == self.generators.len() {
            monomials.push(prefix);
            return;
        }
        for e in 0..self.generators[k].q {
            let mut next: Monomial = prefix.clone();
            next.push(e);
            self.enumerate(next, k + 1, monomials);
        }
    }

    /// A polynomial in the generator symbols with coefficients rational in
    /// `x` as an element (`from_expr`).
    pub fn from_expr(&self, expr: &Basic) -> PolyResult<AlgebraElement> {
        if self.names.is_empty() {
            let c: Fraction = self.kx.from_expr(expr)?;
            return Ok(self.add(&AlgebraElement::new(), &AlgebraElement::from([(Vec::new(), c)])));
        }
        let mut result: AlgebraElement = AlgebraElement::new();
        for (monomial, coefficient) in crate::polys::polyutils::dict_from_expr(expr, &self.names, &self.kx)? {
            result = self.add(&result, &self.reduce(&AlgebraElement::from([(monomial, coefficient)])));
        }
        Ok(result)
    }

    pub fn add(&self, a: &AlgebraElement, b: &AlgebraElement) -> AlgebraElement {
        let mut result: AlgebraElement = a.clone();
        for (monomial, c) in b {
            let total: Fraction = self
                .kx
                .add(&result.get(monomial).cloned().unwrap_or_else(|| self.kx.zero()), c);
            if total.numer.is_empty() {
                result.remove(monomial);
            } else {
                result.insert(monomial.clone(), total);
            }
        }
        result
    }

    /// The element with every exponent below the index of its generator
    /// (`g_i^q_i` replaced by its base) (`reduce`).
    pub fn reduce(&self, a: &AlgebraElement) -> AlgebraElement {
        let mut result: AlgebraElement = AlgebraElement::new();
        let mut pending: AlgebraElement = a.clone();
        while let Some((monomial, c)) = pending.pop_last() {
            let high: Option<usize> = (0..monomial.len())
                .rev()
                .find(|i| monomial[*i] >= self.generators[*i].q);
            let Some(high) = high else {
                result = self.add(&result, &AlgebraElement::from([(monomial, c)]));
                continue;
            };
            let mut lowered: Monomial = monomial.clone();
            lowered[high] -= self.generators[high].q;
            for (base_monomial, d) in &self.bases[high] {
                let product: Monomial = lowered.iter().zip(base_monomial).map(|(u, v)| u + v).collect();
                pending = self.add(&pending, &AlgebraElement::from([(product, self.kx.mul(&c, d))]));
            }
        }
        result
    }

    pub fn multiply(&self, a: &AlgebraElement, b: &AlgebraElement) -> AlgebraElement {
        let mut product: AlgebraElement = AlgebraElement::new();
        for (m1, c) in a {
            for (m2, d) in b {
                let m: Monomial = m1.iter().zip(m2).map(|(u, v)| u + v).collect();
                product = self.add(&product, &AlgebraElement::from([(m, self.kx.mul(c, d))]));
            }
        }
        self.reduce(&product)
    }

    pub fn vector(&self, a: &AlgebraElement) -> Vec<Fraction> {
        let mut row: Vec<Fraction> = vec![self.kx.zero(); self.dimension];
        for (monomial, c) in a {
            row[self.index[monomial]] = c.clone();
        }
        row
    }
}

/// Whether the expression is a rational function in `syms`, SymPy's
/// three-valued answer (`_is_polynomial_like` with the kind `rational`).
fn is_rational_function(e: &Basic, syms: &[Basic]) -> Option<bool> {
    let free = e.free_symbols();
    if !syms.iter().any(|s| free.contains(s)) {
        return Some(true);
    }
    if matches!(e.head(), Head::Symbol(_) | Head::Dummy(..) | Head::Wild(_)) {
        return Some(true);
    }
    if e.is_add() || e.is_mul() {
        return Some(e.args().iter().all(|a| is_rational_function(a, syms) == Some(true)));
    }
    if e.is_pow() {
        let exponent: &Basic = &e.args()[1];
        let exponent_free = exponent.free_symbols();
        if syms.iter().any(|s| exponent_free.contains(s)) {
            return Some(false);
        }
        let base: Option<bool> = is_rational_function(&e.args()[0], syms);
        if base != Some(true) {
            return base;
        }
        return Some(exponent.as_rational().is_some_and(|r| r.is_int()));
    }
    None
}

fn atoms_pow(e: &Basic) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for node in e.preorder() {
        if node.is_pow() && !out.contains(&node) {
            out.push(node);
        }
    }
    out
}

/// `expr` with every power `base**(p/r)` replaced by `symbol**(p*q/r)`,
/// the rebuilt nodes evaluated (`_rewrite_radical`).
fn rewrite_radical(expr: &Basic, base: &Basic, symbol: &Basic, q: u32) -> PolyResult<Basic> {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for node in atoms_pow(expr) {
        if node.args()[0] == *base {
            if let Some(r) = node.args()[1].as_rational() {
                let p: Int = r.numerator().clone();
                let d: Int = Int::from(r.denominator().clone());
                let exponent: Int = crate::numbers::div_floor(&(&p * Int::from(q)), &d);
                rule.insert(node.clone(), pow(symbol, &Basic::integer(exponent))?);
            }
        }
    }
    if rule.is_empty() {
        return Ok(expr.clone());
    }
    crate::core::function::xreplace_evaluated(expr, &rule)
}

/// The radicals of `f` depending on `x`, innermost first, each a
/// generator `base^(1/q)`, and `f` rewritten in their symbols
/// (`_generators`).
fn generators_of(f: &Basic, x: &Basic) -> PolyResult<(Vec<Generator>, Basic)> {
    let mut radicals: Vec<(Basic, u32)> = Vec::new();
    for node in atoms_pow(f) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if let Some(r) = exponent.as_rational() {
            if !r.is_int() && base.has(x) {
                let q: u32 = u32::try_from(r.denominator()).map_err(|_| value_error("a large index".to_string()))?;
                match radicals.iter_mut().find(|(b, _)| b == base) {
                    Some(entry) => {
                        entry.1 = crate::core::intfunc::ilcm(&[Int::from(entry.1), Int::from(q)])
                            .try_into()
                            .map_err(|_| value_error("a large index".to_string()))?
                    }
                    None => radicals.push((base.clone(), q)),
                }
            }
        }
    }
    let mut ordered: Vec<Basic> = Vec::new();
    let mut remaining: Vec<Basic> = radicals.iter().map(|(b, _)| b.clone()).collect();
    while !remaining.is_empty() {
        // innermost first: a base containing no other base
        let mut candidates: Vec<Basic> = remaining.clone();
        candidates.sort_by_key(|b| {
            let s: String = b.to_string();
            (s.chars().count(), s)
        });
        let found: Option<Basic> = candidates
            .into_iter()
            .find(|base| !remaining.iter().any(|other| other != base && base.has(other)));
        let Some(base) = found else {
            return Err(value_error("circular radicals".to_string()));
        };
        remaining.retain(|b| *b != base);
        ordered.push(base);
    }
    let mut generators: Vec<Generator> = Vec::new();
    let mut rewritten: Basic = f.clone();
    for base in ordered {
        let q: u32 = radicals
            .iter()
            .find(|(b, _)| *b == base)
            .map(|(_, q)| *q)
            .expect("a radical");
        let symbol: Basic = crate::core::Dummy::new(&format!("curve_g{}", generators.len()), &[]).0;
        // the base with the earlier generators substituted
        let mut rewritten_base: Basic = base.clone();
        for g in &generators {
            rewritten_base = rewrite_radical(&rewritten_base, &g.base, &g.symbol, g.q)?;
        }
        let original: Basic = pow(&base, &Basic::rational(Int::ONE, Int::from(q))?)?;
        rewritten = rewrite_radical(&rewritten, &rewritten_base, &symbol, q)?;
        generators.push(Generator {
            symbol,
            base: rewritten_base,
            q,
            original,
        });
    }
    Ok((generators, rewritten))
}

fn unit_monomial(m: usize, i: usize) -> Monomial {
    (0..m).map(|j| u32::from(j == i)).collect()
}

/// The monic polynomial with coefficients in `QQ(x)` times the lcm of
/// their denominators, primitive with integer coefficients, a negative
/// constant leading coefficient changed in sign (`_primitive_curve`).
fn primitive_curve(monic: &[Fraction], r: &PolynomialRing<AlgebraicField>) -> PolyResult<Vec<KPoly>> {
    let mut denominator: KPoly = r.one();
    for c in monic {
        if !c.numer.is_empty() {
            denominator = r.lcm(&denominator, &c.denom);
        }
    }
    let mut polys: Vec<KPoly> = Vec::new();
    for c in monic {
        polys.push(if c.numer.is_empty() {
            r.zero()
        } else {
            r.exquo(&r.mul(&c.numer, &denominator), &c.denom)?
        });
    }
    let values: Vec<Rat> = polys
        .iter()
        .flat_map(|p| p.iter().flat_map(|c| c.iter().cloned()))
        .collect();
    let mut common: Int = Int::ONE;
    for q in &values {
        common = crate::core::intfunc::ilcm(&[common, Int::from(q.denominator().clone())]);
    }
    let mut content: Int = Int::ZERO;
    for q in &values {
        let scaled: Int = (q * Rat::from(common.clone())).numerator().clone();
        content = crate::core::intfunc::igcd(&[content, scaled]);
    }
    let mut scale: Rat = Rat::from_parts_signed(common, content);
    if dup_degree(&polys[0]) == 0 && polys[0][0][0] < Rat::ZERO {
        scale = -scale;
    }
    Ok(polys
        .iter()
        .map(|p| p.iter().map(|c| c.iter().map(|q| q * &scale).collect()).collect())
        .collect())
}

/// The polynomial in `y` over `QQ[x]` as an expression.
fn curve_expr(curve: &[KPoly], x: &Basic, y: &Basic) -> PolyResult<Basic> {
    let n: usize = curve.len() - 1;
    let k: AlgebraicField = rationals();
    let mut parts: Vec<Basic> = Vec::new();
    for (i, c) in curve.iter().enumerate() {
        if !c.is_empty() {
            parts.push(mul(&[
                poly_expr(c, x, &k, &Basic::zero())?,
                pow(y, &int((n - i) as i64))?,
            ])?);
        }
    }
    add(&parts)
}

/// Whether the primitive polynomial in `y` over `QQ[x]` is irreducible over
/// `QQ(x)`: at a specialisation of `x` keeping the degree, else by the
/// bivariate factorisation (`_is_irreducible`).
fn is_irreducible(curve: &[KPoly], x: &Basic, y: &Basic) -> PolyResult<bool> {
    let n: usize = curve.len() - 1;
    let k: AlgebraicField = rationals();
    for x0 in [2, 3, 5, 7] {
        let point: Number = k.from_int(x0);
        let values: Vec<Number> = curve
            .iter()
            .map(|c| crate::polys::densetools::dup_eval(c, &point, &k))
            .collect();
        if values[0].is_empty() {
            continue;
        }
        let rational: Vec<Rat> = values.iter().map(|v| v.first().cloned().unwrap_or(Rat::ZERO)).collect();
        let (_, factors) = QQ.factor_dense(&rational)?;
        if factors.len() == 1 && factors[0].1 == 1 && dup_degree(&factors[0].0) == n as isize {
            return Ok(true);
        }
    }
    let expression: Basic = curve_expr(curve, x, y)?;
    let (_, found) = crate::polys::polytools::factor_list(&expression, &[y.clone(), x.clone()])?;
    let in_y: Vec<(Basic, Basic)> = found
        .into_iter()
        .filter(|(f, _)| f.free_symbols().contains(y))
        .collect();
    Ok(in_y.len() == 1 && in_y[0].1 == Basic::one() && crate::polys::polytools::degree(&in_y[0].0, y)? == n as i64)
}

type Found = (Matrix, Matrix, Vec<KPoly>);

/// The powers of `theta` in the monomial basis, the inverse of their
/// matrix and the minimal polynomial of `theta` over `QQ(x)` when it has
/// the degree of the algebra (`_minimal_polynomial`).
fn minimal_polynomial(algebra: &Algebra, theta: &AlgebraElement, y: &Basic) -> PolyResult<Option<Found>> {
    let kx = &algebra.kx;
    let n: usize = algebra.dimension;
    let m: usize = algebra.generators.len();
    let mut power: AlgebraElement = AlgebraElement::from([(vec![0; m], kx.one())]);
    let mut rows: Matrix = Vec::new();
    for _ in 0..n {
        rows.push(algebra.vector(&power));
        power = algebra.multiply(&power, theta);
    }
    let Ok(inverse) = inverse_matrix(&rows, kx) else {
        return Ok(None);
    };
    // theta^n = sum_i a_i theta^i: a = v P^-1
    let last: Vec<Fraction> = algebra.vector(&power);
    let mut a: Vec<Fraction> = vec![kx.zero(); n];
    for i in 0..n {
        for j in 0..n {
            if !last[i].numer.is_empty() && !inverse[i][j].numer.is_empty() {
                a[j] = kx.add(&a[j], &kx.mul(&last[i], &inverse[i][j]));
            }
        }
    }
    let mut monic: Vec<Fraction> = vec![kx.one()];
    monic.extend((0..n).map(|k| kx.neg(&a[n - 1 - k])));
    let curve: Vec<KPoly> = primitive_curve(&monic, &algebra.r)?;
    if !is_irreducible(&curve, &algebra.x, y)? {
        return Ok(None);
    }
    Ok(Some((rows, inverse, curve)))
}

// ---------------------------------------------------------------------
// Expressions of the typed objects

/// The element `c` of `K` as an expression, `theta` the expression of the
/// generator (`_number_expr`).
pub fn number_expr(c: &Number, theta: &Basic) -> PolyResult<Basic> {
    if c.len() <= 1 {
        return Ok(c.first().map_or_else(Basic::zero, |q| Basic::from_rational(q.clone())));
    }
    let n: usize = c.len() - 1;
    let mut parts: Vec<Basic> = Vec::new();
    for (i, a) in c.iter().enumerate() {
        if !a.is_zero() {
            let k: usize = n - i;
            let q: Basic = Basic::from_rational(a.clone());
            parts.push(if k == 0 {
                q
            } else {
                let power: Basic = if k == 1 {
                    theta.clone()
                } else {
                    pow(theta, &int(k as i64))?
                };
                mul(&[q, power])?
            });
        }
    }
    add(&parts)
}

/// The polynomial `p` over `K` as an expression in `x` (`_poly_expr`).
pub fn poly_expr(p: &KPoly, x: &Basic, _k: &AlgebraicField, theta: &Basic) -> PolyResult<Basic> {
    let n: usize = p.len().saturating_sub(1);
    let mut parts: Vec<Basic> = Vec::new();
    for (i, c) in p.iter().enumerate() {
        if !c.is_empty() {
            let k: usize = n - i;
            let value: Basic = number_expr(c, theta)?;
            parts.push(if k == 0 {
                value
            } else {
                let power: Basic = if k == 1 { x.clone() } else { pow(x, &int(k as i64))? };
                mul(&[value, power])?
            });
        }
    }
    if parts.is_empty() {
        return Ok(Basic::zero());
    }
    add(&parts)
}

/// `_fraction_expr`.
pub fn fraction_expr(c: &Fraction, x: &Basic, k: &AlgebraicField, theta: &Basic) -> PolyResult<Basic> {
    let numerator: Basic = poly_expr(&c.numer, x, k, theta)?;
    if dup_degree(&c.denom) == 0 && c.denom[0] == k.one() {
        return Ok(numerator);
    }
    mul(&[numerator, pow(&poly_expr(&c.denom, x, k, theta)?, &int(-1))?])
}

/// An element of a field over the rationals as an expression in `x` and
/// the original `y` (`_element_expr`).
pub fn element_expr(
    field: &FunctionField<AlgebraicField>,
    e: &Element<AlgebraicField>,
    x: &Basic,
    y: &Basic,
) -> PolyResult<Basic> {
    let k: AlgebraicField = rationals();
    let big_y: Basic = mul(&[poly_expr(&field.unit, x, &k, &Basic::zero())?, y.clone()])?;
    let mut terms: Vec<Basic> = Vec::new();
    for (i, c) in e.coefficients.iter().enumerate() {
        if !c.numer.is_empty() {
            terms.push(mul(&[
                fraction_expr(c, x, &k, &Basic::zero())?,
                pow(&big_y, &int(i as i64))?,
            ])?);
        }
    }
    if terms.is_empty() {
        return Ok(Basic::zero());
    }
    crate::polys::polytools::cancel(&add(&terms)?, &[])
}

/// `B^d p(A/B)` for `d >= deg p` (`_substituted`).
pub fn substituted(p: &KPoly, a: &KPoly, b: &KPoly, d: usize, k: &AlgebraicField) -> KPoly {
    let mut total: KPoly = Vec::new();
    let n: usize = p.len().saturating_sub(1);
    for (i, c) in p.iter().enumerate() {
        if !c.is_empty() {
            let e: usize = n - i;
            let term: KPoly = dup_mul(&dup_pow(a, e, k), &dup_pow(b, d - e, k), k);
            total = dup_add(&total, &dup_mul_ground(&term, c, k), k);
        }
    }
    total
}

/// `c(A/B)` for `c` in `K(t)`, an element of `target`
/// (`_substituted_fraction`).
pub fn substituted_fraction(
    c: &Fraction,
    a: &KPoly,
    b: &KPoly,
    target: &FractionField<AlgebraicField>,
) -> PolyResult<Fraction> {
    if c.numer.is_empty() {
        return Ok(target.zero());
    }
    let k: &AlgebraicField = &target.ring.ground;
    let da: usize = dup_degree(&c.numer) as usize;
    let db: usize = dup_degree(&c.denom) as usize;
    let mut numerator: KPoly = substituted(&c.numer, a, b, da, k);
    let mut denominator: KPoly = substituted(&c.denom, a, b, db, k);
    if db >= da {
        numerator = dup_mul(&numerator, &dup_pow(b, db - da, k), k);
    } else {
        denominator = dup_mul(&denominator, &dup_pow(b, da - db, k), k);
    }
    target.new_element(numerator, denominator)
}

// ---------------------------------------------------------------------
// The integrand

/// An integrand rational in `x` and in nested radicals, as an element of
/// the function field of a primitive element of the radicals
/// (`AlgebraicIntegrand`).
pub struct AlgebraicIntegrand {
    pub x: Basic,
    pub original: Basic,
    pub generators: Vec<Generator>,
    pub algebra: Algebra,
    pub y: Basic,
    pub weights: Vec<i64>,
    pub theta: Basic,
    /// the minimal polynomial of the primitive element: its coefficients
    /// in `y` (highest first), polynomials in `x`
    pub curve_coefficients: Vec<KPoly>,
    pub curve: Basic,
    pub field: FunctionField<AlgebraicField>,
    pub power_matrix: Matrix,
    pub power_matrix_inverse: Matrix,
    pub element: Element<AlgebraicField>,
}

impl AlgebraicIntegrand {
    pub fn new(f: &Basic, x: &Basic) -> PolyResult<AlgebraicIntegrand> {
        let free = f.free_symbols();
        if free.iter().any(|s| s != x) {
            return Err(value_error(
                "parameters are not supported on a general curve".to_string(),
            ));
        }
        let (generators, rewritten) = generators_of(f, x)?;
        if generators.is_empty() {
            return Err(value_error(format!("{} is rational in {}", f, x)));
        }
        let mut syms: Vec<Basic> = vec![x.clone()];
        syms.extend(generators.iter().map(|g| g.symbol.clone()));
        if is_rational_function(&rewritten, &syms) != Some(true) {
            return Err(value_error(format!("{} is not an algebraic function of {}", f, x)));
        }
        let algebra: Algebra = Algebra::new(generators.clone(), x)?;
        let y: Basic = crate::core::Dummy::new("y", &[]).0;
        let m: usize = generators.len();
        let n: usize = algebra.dimension;
        // a primitive element: a combination with small integer weights
        let mut rng = crate::random::PyRandom::from_string(&f.to_string());
        let mut weights: Vec<i64> = vec![1; m];
        let mut found: Option<Found> = None;
        for _ in 0..6 {
            let terms: AlgebraElement = weights
                .iter()
                .enumerate()
                .map(|(i, k)| (unit_monomial(m, i), algebra.kx.from_int(*k)))
                .collect();
            let theta_element: AlgebraElement = algebra.reduce(&terms);
            found = minimal_polynomial(&algebra, &theta_element, &y)?;
            if found.is_some() {
                break;
            }
            weights = (0..m).map(|_| rng.randint(1, 7)).collect();
        }
        let Some((powers, inverse, curve)) = found else {
            return Err(value_error(format!(
                "the radicals of {} are dependent (no primitive element of degree {}, the algebra dimension)",
                f, n
            )));
        };
        let mut theta_terms: Vec<Basic> = Vec::new();
        for (k, g) in weights.iter().zip(&generators) {
            theta_terms.push(mul(&[int(*k), g.original.clone()])?);
        }
        let theta: Basic = add(&theta_terms)?;
        let curve_expression: Basic = curve_expr(&curve, x, &y)?;
        let field: FunctionField<AlgebraicField> = FunctionField::new(&curve, x, &y, rationals())?;
        let mut integrand = AlgebraicIntegrand {
            x: x.clone(),
            original: f.clone(),
            generators,
            algebra,
            y,
            weights,
            theta,
            curve_coefficients: curve,
            curve: curve_expression,
            field,
            power_matrix: powers,
            power_matrix_inverse: inverse,
            element: Element {
                coefficients: Vec::new(),
            },
        };
        integrand.element = integrand.to_field(&rewritten).map_err(|error| match error {
            crate::error::PolyError::ZeroDivision(_) => {
                value_error(format!("{} has a denominator which vanishes on the curve", f))
            }
            other => other,
        })?;
        Ok(integrand)
    }

    /// The element with these coordinates in the powers of `y`
    /// (`_from_y_coordinates`).
    fn from_y_coordinates(&self, coordinates: &[Fraction]) -> PolyResult<Element<AlgebraicField>> {
        let kx = &self.field.kx;
        let unit: Fraction = kx.from_ring(&self.field.unit);
        let mut result: Vec<Fraction> = Vec::new();
        let mut scale: Fraction = kx.one();
        for c in coordinates {
            result.push(kx.quo(c, &scale)?);
            scale = kx.mul(&scale, &unit);
        }
        Ok(Element { coefficients: result })
    }

    /// A rational function of `x` and the generator symbols as an element
    /// of the field (`to_field`).
    pub fn to_field(&self, expr: &Basic) -> PolyResult<Element<AlgebraicField>> {
        let (numerator, denominator) =
            crate::simplify::radsimp::fraction(&crate::polys::polytools::cancel(expr, &[])?)?;
        let top: Element<AlgebraicField> = self.polynomial_to_field(&numerator)?;
        let bottom: Element<AlgebraicField> = self.polynomial_to_field(&denominator)?;
        Ok(self.field.mul(&top, &self.field.invert(&bottom)?))
    }

    fn polynomial_to_field(&self, p: &Basic) -> PolyResult<Element<AlgebraicField>> {
        let kx = &self.algebra.kx;
        let vector: Vec<Fraction> = self.algebra.vector(&self.algebra.from_expr(p)?);
        let n: usize = self.algebra.dimension;
        let mut coordinates: Vec<Fraction> = vec![kx.zero(); n];
        for (i, c) in vector.iter().enumerate() {
            if c.numer.is_empty() {
                continue;
            }
            let row: &Vec<Fraction> = &self.power_matrix_inverse[i];
            for j in 0..n {
                if !row[j].numer.is_empty() {
                    coordinates[j] = kx.add(&coordinates[j], &kx.mul(c, &row[j]));
                }
            }
        }
        self.from_y_coordinates(&coordinates)
    }

    /// The integrand in `x` and the primitive element `y` (`element_expr`).
    pub fn element_expr(&self) -> PolyResult<Basic> {
        element_expr(&self.field, &self.element, &self.x, &self.y)
    }

    /// A rational expression in `x` and `y` written in the original
    /// radicals (`polynomial_to_expr`).
    pub fn polynomial_to_expr(&self, p: &Basic) -> PolyResult<Basic> {
        let e: Element<AlgebraicField> = self.field.element(p)?;
        let kx = &self.field.kx;
        let unit: Fraction = kx.from_ring(&self.field.unit);
        let mut coordinates: Vec<Fraction> = Vec::new();
        let mut scale: Fraction = kx.one();
        for c in &e.coefficients {
            coordinates.push(kx.mul(c, &scale));
            scale = kx.mul(&scale, &unit);
        }
        self.radicals_expr(&coordinates, &rationals(), &Basic::zero())
    }

    /// The element with these coordinates in the powers of `y` (elements
    /// of `K(x)`) in the original radicals, cancelled (`radicals_expr`).
    pub fn radicals_expr(&self, coordinates: &[Fraction], k: &AlgebraicField, theta: &Basic) -> PolyResult<Basic> {
        let r: PolynomialRing<AlgebraicField> = PolynomialRing::new(k.clone(), &name_of(&self.x));
        let kx: FractionField<AlgebraicField> = FractionField::new(r.clone());
        let mut totals: Vec<Fraction> = Vec::new();
        for j in 0..self.algebra.monomials.len() {
            let mut total: Fraction = kx.zero();
            for (i, c) in coordinates.iter().enumerate() {
                let v: &Fraction = &self.power_matrix[i][j];
                if !c.numer.is_empty() && !v.numer.is_empty() {
                    // the power matrix is over the rationals: its entries are those of K
                    let v_k: Fraction = kx.new_element(v.numer.clone(), v.denom.clone())?;
                    total = kx.add(&total, &kx.mul(c, &v_k));
                }
            }
            totals.push(total);
        }
        let mut denominator: KPoly = r.one();
        for c in &totals {
            if !c.numer.is_empty() {
                denominator = r.lcm(&denominator, &c.denom);
            }
        }
        let radicals: Vec<Basic> = self.generators.iter().map(|g| g.original.clone()).collect();
        let mut terms: Vec<Basic> = Vec::new();
        for (monomial, c) in self.algebra.monomials.iter().zip(&totals) {
            if c.numer.is_empty() {
                continue;
            }
            let numerator: KPoly = r.exquo(&r.mul(&c.numer, &denominator), &c.denom)?;
            let mut factors: Vec<Basic> = Vec::new();
            for (s, e) in radicals.iter().zip(monomial) {
                factors.push(pow(s, &int(*e as i64))?);
            }
            let term: Basic = mul(&factors)?;
            terms.push(mul(&[poly_expr(&numerator, &self.x, k, theta)?, term])?);
        }
        if terms.is_empty() {
            return Ok(Basic::zero());
        }
        let total: Basic = mul(&[
            add(&terms)?,
            pow(&poly_expr(&denominator, &self.x, k, theta)?, &int(-1))?,
        ])?;
        crate::polys::polytools::cancel(&total, &[])
    }
}

// ---------------------------------------------------------------------
// The curve in z = 1/(x - N)

/// The integrand and the curve after `x = N + 1/z` (`_Transformed`).
pub struct Transformed<'a> {
    pub integrand: &'a AlgebraicIntegrand,
    pub n: i64,
    pub z: Basic,
    pub field: FunctionField<AlgebraicField>,
    /// the curve in `z` and `y` for the Puiseux expansions
    pub curve: Curve,
    pub unit_ratio: Fraction,
    pub integrand_element: Element<AlgebraicField>,
}

impl<'a> Transformed<'a> {
    pub fn new(integrand: &'a AlgebraicIntegrand, n: i64) -> PolyResult<Transformed<'a>> {
        let z: Basic = crate::core::Dummy::new("z", &[]).0;
        let k: AlgebraicField = rationals();
        let curve_x: &Vec<KPoly> = &integrand.curve_coefficients;
        let d: usize = curve_x.iter().map(|c| dup_degree(c).max(0) as usize).max().unwrap_or(0);
        // x = (N z + 1)/z
        let forward_a: KPoly = dup_strip(&[k.from_int(n), k.one()], &k);
        let forward_b: KPoly = vec![k.one(), k.zero()];
        let curve: Vec<KPoly> = curve_x
            .iter()
            .map(|c| substituted(c, &forward_a, &forward_b, d, &k))
            .collect();
        let field: FunctionField<AlgebraicField> = FunctionField::new(&curve, &z, &integrand.y, k.clone())?;
        let degree_y: usize = curve.len() - 1;
        let mut points: Curve = Curve::new();
        for (i, c) in curve.iter().enumerate() {
            if !c.is_empty() {
                let top: usize = c.len() - 1;
                let terms: BTreeMap<i64, Number> = c
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(j, v)| ((top - j) as i64, v.clone()))
                    .collect();
                points.insert((degree_y - i) as i64, terms);
            }
        }
        let kz = field.kx.clone();
        // the generators Y_x = a_n(x) y and Y_z = a_z(z) y: y is the same function
        let unit_x: Fraction = substituted_fraction(
            &integrand.field.kx.from_ring(&integrand.field.unit),
            &forward_a,
            &forward_b,
            &kz,
        )?;
        let unit_ratio: Fraction = kz.quo(&unit_x, &kz.from_ring(&field.unit))?;
        let mut transformed = Transformed {
            integrand,
            n,
            z,
            field,
            curve: points,
            unit_ratio,
            integrand_element: Element {
                coefficients: Vec::new(),
            },
        };
        let minus_z2: Fraction = kz.new_element(vec![k.from_int(-1)], vec![k.one(), k.zero(), k.zero()])?;
        let element: Element<AlgebraicField> = transformed.to_z(&integrand.element)?;
        transformed.integrand_element = transformed.field.scaled(&element, &minus_z2);
        Ok(transformed)
    }

    /// An element of the field in `x` as an element of the field in `z`
    /// (`to_z`).
    pub fn to_z(&self, e: &Element<AlgebraicField>) -> PolyResult<Element<AlgebraicField>> {
        let kz = &self.field.kx;
        let k: AlgebraicField = rationals();
        let a: KPoly = dup_strip(&[k.from_int(self.n), k.one()], &k);
        let b: KPoly = vec![k.one(), k.zero()];
        let mut coefficients: Vec<Fraction> = Vec::new();
        let mut ratio: Fraction = kz.one();
        for c in &e.coefficients {
            // c(x) Y_x^i = c(N + 1/z) (a_x/a_z)^i Y_z^i
            coefficients.push(kz.mul(&substituted_fraction(c, &a, &b, kz)?, &ratio));
            ratio = kz.mul(&ratio, &self.unit_ratio);
        }
        Ok(Element { coefficients })
    }

    /// An element of `K(z)` as an element of `K(x)`, `z = 1/(x - N)`
    /// (`to_x`).
    pub fn to_x(&self, c: &Fraction, k: &AlgebraicField) -> PolyResult<Fraction> {
        let target: FractionField<AlgebraicField> =
            FractionField::new(PolynomialRing::new(k.clone(), &name_of(&self.integrand.x)));
        let a: KPoly = vec![k.one()];
        let b: KPoly = dup_strip(&[k.one(), k.from_int(-self.n)], k);
        substituted_fraction(c, &a, &b, &target)
    }

    /// An element of a field in `z` (over the number field `k`) as an
    /// expression in `x` and the original radicals (`to_x_expr`).
    pub fn to_x_expr(
        &self,
        field: &FunctionField<AlgebraicField>,
        e: &Element<AlgebraicField>,
        theta: &Basic,
    ) -> PolyResult<Basic> {
        let k: &AlgebraicField = &field.k;
        let kz = &field.kx;
        let unit: Fraction = kz.from_ring(&dup_strip(
            &self
                .field
                .unit
                .iter()
                .map(|c| c.first().map_or_else(|| k.zero(), |q| k.from_rational(q)))
                .collect::<Vec<_>>(),
            k,
        ));
        let mut coordinates: Vec<Fraction> = Vec::new();
        let mut scale: Fraction = kz.one();
        for c in &e.coefficients {
            // the coefficient of y^i: c_i a_z^i
            coordinates.push(self.to_x(&kz.mul(c, &scale), k)?);
            scale = kz.mul(&scale, &unit);
        }
        self.integrand.radicals_expr(&coordinates, k, theta)
    }
}

// ---------------------------------------------------------------------
// The Hermite reduction

type Field = FunctionField<AlgebraicField>;
type FieldElement = Element<AlgebraicField>;

/// The reduced integrand `h = sum_i A_i w_i / D`, `D` squarefree, with the
/// elementary part `g` subtracted, `f = g' + h` (`_Reduced`).
pub struct Reduced {
    pub g: FieldElement,
    /// elements of `K[z]`
    pub numerators: Vec<KPoly>,
    pub d: KPoly,
}

fn degree_of(p: &KPoly) -> i64 {
    if p.is_empty() {
        -1
    } else {
        dup_degree(p) as i64
    }
}

/// `(A, D)` with `e = sum(A_i w_i)/D`, `D` monic and coprime to the `A_i`
/// together (`_coordinates_over_common_denominator`).
fn coordinates_over_common_denominator(
    field: &Field,
    e: &FieldElement,
    basis: &[FieldElement],
) -> PolyResult<(Vec<KPoly>, KPoly)> {
    let r = &field.r;
    let k = &field.k;
    let coordinates: Vec<Fraction> = field.coordinates(e, basis)?;
    let mut d: KPoly = r.one();
    for c in &coordinates {
        d = r.lcm(&d, &c.denom);
    }
    let mut a: Vec<KPoly> = Vec::new();
    for c in &coordinates {
        a.push(r.mul(&c.numer, &r.exquo(&d, &c.denom)?));
    }
    let mut content: KPoly = d.clone();
    for p in &a {
        content = r.gcd(&content, p);
    }
    if degree_of(&content) > 0 {
        a = a.iter().map(|p| r.exquo(p, &content)).collect::<PolyResult<_>>()?;
        d = r.exquo(&d, &content)?;
    }
    let lc: Number = d[0].clone();
    if lc != k.one() {
        let inverse: KPoly = vec![k.quo(&k.one(), &lc)?];
        a = a.iter().map(|p| r.mul(p, &inverse)).collect();
        d = r.mul(&d, &inverse);
    }
    Ok((a, d))
}

/// `(e, M)` with `e W' = M W`, `e` a polynomial and `M` a polynomial
/// matrix (`_derivative_matrix`).
fn derivative_matrix(field: &Field, basis: &[FieldElement]) -> PolyResult<(KPoly, Vec<Vec<KPoly>>)> {
    let r = &field.r;
    let mut rows: Vec<Vec<Fraction>> = Vec::new();
    for w in basis {
        rows.push(field.coordinates(&field.derivative(w)?, basis)?);
    }
    let mut e: KPoly = r.one();
    for row in &rows {
        for c in row {
            e = r.lcm(&e, &c.denom);
        }
    }
    let mut m: Vec<Vec<KPoly>> = Vec::new();
    for row in &rows {
        let mut line: Vec<KPoly> = Vec::new();
        for c in row {
            line.push(r.mul(&c.numer, &r.exquo(&e, &c.denom)?));
        }
        m.push(line);
    }
    Ok((e, m))
}

fn combination(field: &Field, a: &[KPoly], d: &KPoly, basis: &[FieldElement]) -> PolyResult<FieldElement> {
    let kx = &field.kx;
    let mut total: FieldElement = field.zero();
    for (p, w) in a.iter().zip(basis) {
        if !p.is_empty() {
            total = field.add(&total, &field.scaled(w, &kx.new_element(p.clone(), d.clone())?));
        }
    }
    Ok(total)
}

/// `b` with `b Phi = A` modulo `V` (`_solve_modulo`).
fn solve_modulo(
    phi: &[Vec<KPoly>],
    a: &[KPoly],
    v: &KPoly,
    r: &PolynomialRing<AlgebraicField>,
    kx: &FractionField<AlgebraicField>,
) -> PolyResult<Vec<KPoly>> {
    use crate::polys::functionfields::{adjugate, determinant};
    let n: usize = a.len();
    let phi: Vec<Vec<KPoly>> = phi.to_vec();
    let det: KPoly = determinant(&phi, r)?;
    let adj: Vec<Vec<KPoly>> = adjugate(&phi, r, kx)?;
    let inverse: KPoly = crate::polys::euclidtools::dup_invert(&det, v, &r.ground)
        .map_err(|_| value_error("the Hermite reduction step has no solution: not an integral basis".to_string()))?;
    let mut b: Vec<KPoly> = Vec::new();
    for j in 0..n {
        let mut total: KPoly = r.zero();
        for i in 0..n {
            total = r.add(&total, &r.mul(&a[i], &adj[i][j]));
        }
        b.push(r.rem(&r.mul(&total, &inverse), v)?);
    }
    Ok(b)
}

/// Trager's Hermite reduction of `f` in the integral `basis`
/// (`_hermite_reduce`).
pub fn hermite_reduce(field: &Field, basis: &[FieldElement], f: &FieldElement) -> PolyResult<Reduced> {
    let r = &field.r;
    let kx = &field.kx;
    let k = &field.k;
    let n: usize = field.n;
    let (e, m) = derivative_matrix(field, basis)?;
    let mut g: FieldElement = field.zero();
    let (mut a, mut d) = coordinates_over_common_denominator(field, f, basis)?;
    loop {
        let factors = crate::polys::sqfreetools::dup_sqf_list(&d, k).1;
        let top: usize = factors.iter().map(|(_, mult)| *mult).max().unwrap_or(1);
        if top < 2 {
            break;
        }
        let mut v: KPoly = r.one();
        for (factor, multiplicity) in &factors {
            if *multiplicity == top {
                v = r.mul(&v, factor);
            }
        }
        let u: KPoly = r.exquo(&d, &r.pow(&v, top as u64))?;
        // e must divide U V: otherwise numerator and denominator are
        // multiplied by the missing part
        let missing: KPoly = r.exquo(&e, &r.gcd(&e, &r.mul(&u, &v)))?;
        if degree_of(&missing) > 0 {
            a = a.iter().map(|p| r.mul(p, &missing)).collect();
            d = r.mul(&d, &missing);
            continue;
        }
        // b (U V M / e - (m - 1) U V' I) = A (mod V)
        let uv_over_e: KPoly = r.exquo(&r.mul(&u, &v), &e)?;
        let shift: KPoly = r.mul(&r.mul(&r.from_int((top - 1) as i64), &u), &r.diff(&v));
        let mut phi: Vec<Vec<KPoly>> = Vec::new();
        for i in 0..n {
            let mut row: Vec<KPoly> = Vec::new();
            for j in 0..n {
                let entry: KPoly = r.mul(&uv_over_e, &m[i][j]);
                row.push(if i == j { r.sub(&entry, &shift) } else { entry });
            }
            phi.push(row);
        }
        let b: Vec<KPoly> = solve_modulo(&phi, &a, &v, r, kx)?;
        let mut big_b: FieldElement = field.zero();
        for (b_i, w) in b.iter().zip(basis) {
            if !b_i.is_empty() {
                big_b = field.add(&big_b, &field.scaled(w, &kx.from_ring(b_i)));
            }
        }
        let term: FieldElement = field.scaled(&big_b, &kx.invert(&kx.from_ring(&r.pow(&v, (top - 1) as u64)))?);
        g = field.add(&g, &term);
        let remainder: FieldElement = field.sub(&combination(field, &a, &d, basis)?, &field.derivative(&term)?);
        (a, d) = coordinates_over_common_denominator(field, &remainder, basis)?;
    }
    Ok(Reduced { g, numerators: a, d })
}

/// The integral basis of `field` (over the rationals) normal at infinity,
/// with its pole orders there, computed over the plain rationals and
/// converted (`_normal_basis`).
pub fn normal_basis(field: &Field) -> PolyResult<(Vec<FieldElement>, Vec<i64>)> {
    let coefficients: Vec<Vec<Rat>> = field
        .curve
        .iter()
        .map(|p| p.iter().map(|c| c.first().cloned().unwrap_or(Rat::ZERO)).collect())
        .collect();
    let plain: FunctionField<crate::polys::domains::RationalField> =
        FunctionField::new(&coefficients, &field.x, &field.y, QQ)?;
    let basis = plain.normal_basis()?;
    let orders: Vec<i64> = plain.pole_orders_at_infinity(&basis)?;
    let lifted = |p: &[Rat]| -> KPoly {
        p.iter()
            .map(|c| if c.is_zero() { Vec::new() } else { vec![c.clone()] })
            .collect()
    };
    let converted: Vec<FieldElement> = basis
        .iter()
        .map(|w| Element {
            coefficients: w
                .coefficients
                .iter()
                .map(|c| FracElement {
                    numer: lifted(&c.numer),
                    denom: lifted(&c.denom),
                })
                .collect(),
        })
        .collect();
    Ok((converted, orders))
}

/// An integer which is neither a pole of the integrand nor a root of the
/// discriminant of the curve (`_regular_integer`).
pub fn regular_integer(integrand: &AlgebraicIntegrand) -> PolyResult<i64> {
    let k: AlgebraicField = rationals();
    let r = &integrand.algebra.r;
    let mut polynomials: Vec<KPoly> = vec![crate::polys::euclidtools::dup_discriminant(
        &integrand.curve_coefficients,
        r,
    )];
    for c in &integrand.element.coefficients {
        polynomials.push(c.denom.clone());
    }
    let mut expressions: Vec<Basic> = Vec::new();
    for g in &integrand.generators {
        expressions.push(crate::simplify::radsimp::fraction(&crate::polys::polytools::cancel(&g.base, &[])?)?.1);
    }
    for n in [0i64, 1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 11, 13, 17, 19, 23] {
        let point: Number = k.from_int(n);
        if polynomials
            .iter()
            .any(|p| crate::polys::densetools::dup_eval(p, &point, &k).is_empty())
        {
            continue;
        }
        let rule: HashMap<Basic, Basic> = HashMap::from([(integrand.x.clone(), int(n))]);
        let mut regular: bool = true;
        for d in &expressions {
            if crate::core::function::xreplace_evaluated(d, &rule)? == Basic::zero() {
                regular = false;
                break;
            }
        }
        if regular {
            return Ok(n);
        }
    }
    Err(value_error("no regular integer found".to_string()))
}

/// The pieces of the Hermite part of the integral: the reduction, the
/// verdict of Trager's criterion at infinity (`Some(true)`: not
/// elementary; `Some(false)`: nothing left; `None`: the logarithmic part
/// decides) and the remainder in `z` (`integrate_on_curve` with
/// `logarithms=False`).
pub struct HermitePart<'a> {
    pub transformed: Transformed<'a>,
    pub basis: Vec<FieldElement>,
    pub orders: Vec<i64>,
    pub reduced: Reduced,
    pub nonelementary: Option<bool>,
    pub remainder: Option<FieldElement>,
}

/// `integrate_on_curve(f, x, logarithms=False)`: the Hermite reduction and
/// Trager's criterion at infinity.
pub fn hermite_part(integrand: &AlgebraicIntegrand) -> PolyResult<HermitePart<'_>> {
    let n: i64 = regular_integer(integrand)?;
    let transformed: Transformed = Transformed::new(integrand, n)?;
    let (basis, orders) = normal_basis(&transformed.field)?;
    let reduced: Reduced = hermite_reduce(&transformed.field, &basis, &transformed.integrand_element)?;
    let nonzero: Vec<(&KPoly, i64)> = reduced
        .numerators
        .iter()
        .zip(orders.iter().copied())
        .filter(|(a, _)| !a.is_empty())
        .collect();
    if nonzero.is_empty() {
        return Ok(HermitePart {
            transformed,
            basis,
            orders,
            reduced,
            nonelementary: Some(false),
            remainder: None,
        });
    }
    let remainder: FieldElement = combination(&transformed.field, &reduced.numerators, &reduced.d, &basis)?;
    // Trager's criterion: h z^2 integral at infinity
    let pole: i64 = nonzero
        .iter()
        .map(|(a, k)| degree_of(a) - degree_of(&reduced.d) + k)
        .max()
        .expect("a numerator");
    let verdict: Option<bool> = if pole > -2 { Some(true) } else { None };
    Ok(HermitePart {
        transformed,
        basis,
        orders,
        reduced,
        nonelementary: verdict,
        remainder: Some(remainder),
    })
}

impl HermitePart<'_> {
    /// The elementary part of the reduction in `x` and the radicals.
    pub fn elementary_part(&self) -> PolyResult<Basic> {
        self.transformed
            .to_x_expr(&self.transformed.field, &self.reduced.g, &Basic::zero())
    }

    /// The remaining integrand in `x` (`remainder_expr`).
    pub fn remainder_expr(&self) -> PolyResult<Basic> {
        let Some(remainder) = &self.remainder else {
            return Ok(Basic::zero());
        };
        let t = &self.transformed;
        let x: &Basic = &t.integrand.x;
        // the remainder is a function of z integrated in dz = -dx/(x - N)^2
        let back: Basic = t.to_x_expr(&t.field, remainder, &Basic::zero())?;
        let shift: Basic = pow(&add(&[x.clone(), int(-t.n)])?, &int(-2))?;
        crate::polys::polytools::cancel(&mul(&[int(-1), back, shift])?, &[])
    }
}

// ---------------------------------------------------------------------
// The places above the roots of the denominator and the residues

use crate::polys::puiseux::{
    adjoin_root, degree as field_degree, expansions_in_field, factor_over, generator, in_subfield,
    minimal_polynomial_of, Embedding, LaurentSeries, PuiseuxExpansion,
};

/// A factor of the denominator by its coefficients (`FactorKey`).
type FactorKey = Vec<Number>;

/// A place above a root of the factor `g` (over the coefficient field)
/// of the denominator, a representative of its conjugates, with the
/// branch, the embedding of the field into that of the branch, and the
/// residue of `h dz` there (`_Place`).
#[derive(Debug, Clone)]
pub struct Place {
    pub g: KPoly,
    pub key: FactorKey,
    pub branch: PuiseuxExpansion,
    pub embedding: Embedding,
    pub residue: Number,
}

impl Place {
    pub fn ramification(&self) -> i64 {
        self.branch.ramification
    }

    /// The residue as an element of the coefficient field (`residue_in`).
    pub fn residue_in(&self) -> PolyResult<Number> {
        in_subfield(&self.residue, &self.branch.field, &self.embedding)
            .ok_or_else(|| value_error("a residue outside the field of the residues".to_string()))
    }
}

fn floor_div(a: i64, b: i64) -> i64 {
    let q: i64 = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// A polynomial as `{exponent: coefficient}` (`dup_to_dict`).
fn poly_terms(p: &KPoly) -> BTreeMap<i64, Number> {
    let top: usize = p.len().saturating_sub(1);
    p.iter()
        .enumerate()
        .filter(|(_, c)| !c.is_empty())
        .map(|(j, c)| ((top - j) as i64, c.clone()))
        .collect()
}

/// A polynomial in `z` as a curve without `y` (`_poly_curve`).
fn poly_curve(p: &KPoly) -> Curve {
    if p.is_empty() {
        Curve::new()
    } else {
        Curve::from([(0, poly_terms(p))])
    }
}

/// The irreducible factors in SymPy's order of `factor_list`
/// (`_sorted_factors`).
fn sorted_factors(factors: Vec<(KPoly, usize)>, k: &AlgebraicField) -> Vec<(KPoly, usize)> {
    let key = |factor: &(KPoly, usize)| -> (usize, usize, Vec<Number>) {
        let (f, m) = factor;
        if k.degree > 1 {
            return (f.len(), *m, f.clone());
        }
        let values: Vec<Rat> = f.iter().map(|c| c.first().cloned().unwrap_or(Rat::ZERO)).collect();
        let mut common: Int = Int::ONE;
        for q in &values {
            common = crate::core::intfunc::ilcm(&[common, Int::from(q.denominator().clone())]);
        }
        let mut content: Int = Int::ZERO;
        for q in &values {
            content = crate::core::intfunc::igcd(&[content, (q * Rat::from(common.clone())).numerator().clone()]);
        }
        let scale: Rat = Rat::from_parts_signed(common, content);
        (
            f.len(),
            *m,
            values
                .iter()
                .map(|q| if q.is_zero() { Vec::new() } else { vec![q * &scale] })
                .collect(),
        )
    };
    let mut sorted: Vec<(KPoly, usize)> = factors;
    sorted.sort_by_key(key);
    sorted
}

fn factors_over(p: &KPoly, k: &AlgebraicField) -> PolyResult<Vec<(KPoly, usize)>> {
    Ok(sorted_factors(factor_over(k, p)?, k))
}

/// The series of the basis elements along `branch`, known below
/// `t^order` (`_basis_series`).
fn basis_series(
    field: &Field,
    basis: &[FieldElement],
    branch: &PuiseuxExpansion,
    order: i64,
) -> PolyResult<Vec<LaurentSeries>> {
    let r = &field.r;
    let mut result: Vec<LaurentSeries> = Vec::new();
    for w in basis {
        let mut d: KPoly = r.one();
        for c in &w.coefficients {
            d = r.lcm(&d, &c.denom);
        }
        // the coordinates are in the powers of the integral generator a_n y,
        // the branches are those of y
        let mut numerator: Curve = Curve::new();
        let mut unit_power: KPoly = r.one();
        for (i, c) in w.coefficients.iter().enumerate() {
            if !c.numer.is_empty() {
                let p: KPoly = r.mul(&r.exquo(&r.mul(&c.numer, &d), &c.denom)?, &unit_power);
                numerator.insert(i as i64, poly_terms(&p));
            }
            unit_power = r.mul(&unit_power, &field.unit);
        }
        let series_denominator: LaurentSeries = branch.evaluate(&poly_curve(&d), None)?;
        let Some(v_d) = series_denominator.valuation() else {
            return Err(value_error("a denominator vanishing along the branch".to_string()));
        };
        let series_numerator: LaurentSeries = branch.evaluate(&numerator, None)?.truncated(order + v_d);
        let relative: i64 = match series_numerator.valuation() {
            Some(v_n) => (order - (v_n - v_d) + 1).max(1),
            None => 1,
        };
        result.push(
            series_numerator
                .mul(&series_denominator.inverse(relative)?)
                .truncated(order),
        );
    }
    Ok(result)
}

/// The multiplicity of `g` in the denominator of `w`
/// (`_multiplicity_of_root`).
fn multiplicity_of_root(field: &Field, w: &FieldElement, g: &KPoly, k: &AlgebraicField) -> PolyResult<usize> {
    let r = &field.r;
    let mut d: KPoly = r.one();
    for c in &w.coefficients {
        d = r.lcm(&d, &c.denom);
    }
    let mut p: KPoly = d;
    let mut multiplicity: usize = 0;
    while degree_of(&p) >= degree_of(g) {
        let (quotient, remainder) = crate::polys::densearith::dup_div(&p, g, k)?;
        if !remainder.is_empty() {
            break;
        }
        p = quotient;
        multiplicity += 1;
    }
    Ok(multiplicity)
}

fn max_multiplicity(field: &Field, basis: &[FieldElement], g: &KPoly, k: &AlgebraicField) -> PolyResult<usize> {
    let mut best: usize = 0;
    for w in basis {
        best = best.max(multiplicity_of_root(field, w, g, k)?);
    }
    Ok(best)
}

/// The places above one root of each irreducible factor over `k` of the
/// denominator of the reduced integrand, with their residues (`_places`).
fn places_of(
    t: &Transformed,
    basis: &[FieldElement],
    reduced: &Reduced,
    k: &AlgebraicField,
    orders: &HashMap<FactorKey, i64>,
) -> PolyResult<Vec<Place>> {
    let field = &t.field;
    let (z, big_y) = (&field.x, &field.y);
    let mut places: Vec<Place> = Vec::new();
    for (g, _) in factors_over(&reduced.d, k)? {
        if degree_of(&g) < 1 {
            continue;
        }
        let (k_point, into, beta) = adjoin_root(k, &g)?;
        let multiplicity: usize = max_multiplicity(field, basis, &g, k)?;
        let order: i64 = (3 + multiplicity as i64).max(orders.get(&g).copied().unwrap_or(0));
        let rational_into: Embedding = Embedding::new(rationals(), k_point.clone(), None)?;
        for branch in expansions_in_field(&t.curve, z, big_y, &k_point, &rational_into, Some(&beta), order, None)? {
            let l: &AlgebraicField = &branch.field;
            let e: i64 = branch.ramification;
            // h = S/D with D of valuation e: the coefficient of t^(-e) needs S below t^1
            let series: Vec<LaurentSeries> = basis_series(field, basis, &branch, 1)?;
            let mut s: LaurentSeries = LaurentSeries::constant(l, l.zero());
            for (a, w) in reduced.numerators.iter().zip(&series) {
                s = s.add(&branch.evaluate(&poly_curve(a), None)?.mul(w).truncated(1));
            }
            let d_series: LaurentSeries = branch.evaluate(&poly_curve(&reduced.d), None)?;
            let residue: Number = match s.valuation() {
                // no pole on this branch
                None => l.zero(),
                Some(v_s) => {
                    let h: LaurentSeries = s.mul(&d_series.inverse((2 - v_s).max(1))?);
                    if h.order.is_some_and(|o| o <= -e) {
                        return Err(value_error("insufficient precision for the residue".to_string()));
                    }
                    // dz = scale e t^(e-1) dt: the residue is scale e [t^(-e)] h
                    l.mul(&l.mul(&h.coefficient(-e), &branch.scale), &l.from_int(e))
                }
            };
            let embedding: Embedding = into.then(&branch.embedding)?;
            places.push(Place {
                g: g.clone(),
                key: g.clone(),
                branch,
                embedding,
                residue,
            });
        }
    }
    Ok(places)
}

// ---------------------------------------------------------------------
// The logarithmic part

/// `beta/(n m) log(N/den)` (`_LogTerm`).
#[derive(Debug, Clone)]
pub struct LogTerm {
    pub beta: Number,
    pub n: i64,
    pub m: i64,
    pub numerator: FieldElement,
    /// an element of `K[z]`
    pub den: KPoly,
}

/// The splitting field over the rationals of the minimal polynomials of
/// the residues (`_residue_field`).
fn residue_field(places: &[Place]) -> PolyResult<AlgebraicField> {
    let mut k: AlgebraicField = rationals();
    for place in places {
        let m: Vec<Rat> = minimal_polynomial_of(&place.residue, &place.branch.field);
        if m.len() - 1 == 1 {
            continue;
        }
        loop {
            let lifted: KPoly = m.iter().map(|c| k.from_rational(c)).collect();
            let factors: Vec<KPoly> = factors_over(&lifted, &k)?
                .into_iter()
                .map(|(f, _)| f)
                .filter(|f| degree_of(f) > 1)
                .collect();
            let Some(factor) = factors.first() else {
                break;
            };
            if field_degree(&k) * degree_of(factor) as usize > RESIDUE_FIELD_DEGREE {
                return Err(value_error("the field of the residues is too large".to_string()));
            }
            k = adjoin_root(&k, factor)?.0;
        }
    }
    Ok(k)
}

/// The coordinates of an element of `k` in the powers of its generator,
/// low degree first (`_rational_coordinates`).
fn rational_coordinates(value: &Number, k: &AlgebraicField) -> Vec<Rat> {
    let mut coefficients: Vec<Rat> = value.iter().rev().cloned().collect();
    while coefficients.len() < field_degree(k) {
        coefficients.push(Rat::ZERO);
    }
    coefficients
}

fn sparse(v: &[Rat]) -> BTreeMap<usize, Rat> {
    v.iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(j, c)| (j, c.clone()))
        .collect()
}

fn rank(vectors: &[Vec<Rat>]) -> usize {
    let rows: Vec<BTreeMap<usize, Rat>> = vectors.iter().map(|v| sparse(v)).collect();
    QQ.sparse_echelon(&rows).len()
}

/// A basis over the rationals of the span of `values` and the
/// coordinates of every value in it (`_rational_basis`).
fn rational_basis(values: &[Number], k: &AlgebraicField) -> PolyResult<(Vec<Number>, Vec<Vec<Rat>>)> {
    let vectors: Vec<Vec<Rat>> = values.iter().map(|v| rational_coordinates(v, k)).collect();
    let dimension: usize = vectors[0].len();
    let mut chosen: Vec<usize> = Vec::new();
    let mut current: usize = 0;
    for (i, vector) in vectors.iter().enumerate() {
        let mut trial: Vec<Vec<Rat>> = chosen.iter().map(|j| vectors[*j].clone()).collect();
        trial.push(vector.clone());
        if rank(&trial) > current {
            chosen.push(i);
            current += 1;
        }
    }
    let basis: Vec<Number> = chosen.iter().map(|i| values[*i].clone()).collect();
    let r: usize = chosen.len();
    let mut coordinates: Vec<Vec<Rat>> = Vec::new();
    for vector in &vectors {
        // c B = v: one equation per coordinate, the unknowns c_j
        let mut rows: Vec<BTreeMap<usize, Rat>> = Vec::new();
        for kk in 0..dimension {
            let mut row: BTreeMap<usize, Rat> = (0..r)
                .filter(|j| !vectors[chosen[*j]][kk].is_zero())
                .map(|j| (j, vectors[chosen[j]][kk].clone()))
                .collect();
            if !vector[kk].is_zero() {
                row.insert(r, vector[kk].clone());
            }
            if !row.is_empty() {
                rows.push(row);
            }
        }
        let mut solution: Vec<Rat> = vec![Rat::ZERO; r];
        for row in QQ.sparse_echelon(&rows) {
            let pivot: usize = *row.keys().next().expect("a nonzero row");
            if pivot == r {
                return Err(value_error("a residue outside the span of the basis".to_string()));
            }
            solution[pivot] = row.get(&r).cloned().unwrap_or(Rat::ZERO);
        }
        coordinates.push(solution);
    }
    Ok((basis, coordinates))
}

/// `(N, den)` with `u = N/den` of divisor `n sum_P c_P P`, or `None`
/// (`_function_with_divisor`).
#[allow(clippy::too_many_arguments)]
fn function_with_divisor(
    t: &Transformed,
    basis: &[FieldElement],
    orders: &[i64],
    places: &[Place],
    coefficients: &[i64],
    n: i64,
    k: &AlgebraicField,
    e_k: &Field,
) -> PolyResult<Option<(FieldElement, KPoly)>> {
    let field = &t.field;
    let r_k = &e_k.r;
    let mut k_of: HashMap<FactorKey, i64> = HashMap::new();
    let mut factor_of: HashMap<FactorKey, KPoly> = HashMap::new();
    for (place, c) in places.iter().zip(coefficients) {
        let needed: i64 = if *c < 0 {
            -floor_div(n * c, place.ramification())
        } else {
            0
        };
        let entry = k_of.entry(place.key.clone()).or_insert(0);
        *entry = (*entry).max(needed);
        factor_of.insert(place.key.clone(), place.g.clone());
    }
    let mut den: KPoly = r_k.one();
    for (key, kk) in &k_of {
        den = r_k.mul(&den, &r_k.pow(&factor_of[key], *kk as u64));
    }
    let degree_den: i64 = degree_of(&den);
    let mut unknowns: Vec<(usize, i64)> = Vec::new();
    for (i, kk) in orders.iter().enumerate() {
        for s in 0..(degree_den - kk + 1).max(0) {
            unknowns.push((i, s));
        }
    }
    if unknowns.is_empty() {
        return Ok(None);
    }
    let d_k: usize = field_degree(k);
    let columns_count: usize = unknowns.len() * d_k;
    let mut rows: Vec<Vec<Rat>> = Vec::new();
    for (place, c) in places.iter().zip(coefficients) {
        let branch = &place.branch;
        let e: i64 = branch.ramification;
        let required: i64 = n * c + k_of[&place.key] * e; // ord_P(N) >= required
        let multiplicity: i64 = max_multiplicity(field, basis, &place.g, k)? as i64;
        let lowest: i64 = -e * (multiplicity + 1);
        if required <= lowest {
            continue;
        }
        let lk: &AlgebraicField = &branch.field;
        let d_lk: usize = field_degree(lk);
        let mut theta_powers: Vec<Number> = vec![lk.one()];
        let generator_k: Number = place.embedding.apply(&generator(k));
        for _ in 0..d_k.saturating_sub(1) {
            let next: Number = lk.mul(&theta_powers[theta_powers.len() - 1], &generator_k);
            theta_powers.push(next);
        }
        let series: Vec<LaurentSeries> = basis_series(field, basis, branch, required)?;
        let z_series: LaurentSeries = branch.x_series()?;
        let mut columns: Vec<LaurentSeries> = Vec::new();
        for (i, s) in &unknowns {
            columns.push(z_series.power(*s as u64).mul(&series[*i]).truncated(required));
        }
        if columns.iter().any(|s| s.order.is_some_and(|o| o < required)) {
            return Err(value_error(
                "insufficient precision in the divisor conditions".to_string(),
            ));
        }
        for exponent in lowest..required {
            for coordinate in 0..d_lk {
                let mut row: Vec<Rat> = vec![Rat::ZERO; columns_count];
                for (u, s) in columns.iter().enumerate() {
                    let value: Number = s.coefficient(exponent);
                    if value.is_empty() {
                        continue;
                    }
                    for rr in 0..d_k {
                        row[u * d_k + rr] =
                            rational_coordinates(&lk.mul(&value, &theta_powers[rr]), lk)[coordinate].clone();
                    }
                }
                rows.push(row);
            }
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let kernel: Vec<Vec<Rat>> = QQ.nullspace(&rows, columns_count)?;
    let Some(solution) = kernel.first() else {
        return Ok(None);
    };
    let kz = &e_k.kx;
    let mut big_n: FieldElement = e_k.zero();
    for (u, (i, s)) in unknowns.iter().enumerate() {
        let mut coefficient: Number = k.zero();
        let mut power: Number = k.one();
        for rr in 0..d_k {
            coefficient = k.add(&coefficient, &k.mul(&k.from_rational(&solution[u * d_k + rr]), &power));
            power = k.mul(&power, &generator(k));
        }
        if coefficient.is_empty() {
            continue;
        }
        let mut monomial: KPoly = vec![coefficient];
        monomial.extend((0..*s).map(|_| k.zero()));
        big_n = e_k.add(&big_n, &e_k.scaled(&basis[*i], &kz.from_ring(&monomial)));
    }
    if e_k.is_zero(&big_n) {
        return Ok(None);
    }
    Ok(Some((big_n, den)))
}

type Divisors = (Vec<(Number, i64, Vec<i64>)>, Vec<Place>);

/// The places and, for each element of a basis of the span of the
/// residues over the rationals, the common denominator and the integer
/// coefficients of its divisor (`_divisors`).
fn divisors_of(
    t: &Transformed,
    basis: &[FieldElement],
    reduced: &Reduced,
    k: &AlgebraicField,
    orders: &HashMap<FactorKey, i64>,
) -> PolyResult<Divisors> {
    let places: Vec<Place> = places_of(t, basis, reduced, k, orders)?;
    let residues: Vec<Number> = places.iter().map(|p| p.residue_in()).collect::<PolyResult<_>>()?;
    if residues.iter().all(|r| r.is_empty()) {
        return Ok((Vec::new(), places));
    }
    let (span, coordinates) = rational_basis(&residues, k)?;
    let mut divisors: Vec<(Number, i64, Vec<i64>)> = Vec::new();
    for (j, beta) in span.into_iter().enumerate() {
        let fractions: Vec<Rat> = (0..places.len()).map(|kk| coordinates[kk][j].clone()).collect();
        let mut m: Int = Int::ONE;
        for q in &fractions {
            m = crate::core::intfunc::ilcm(&[m, Int::from(q.denominator().clone())]);
        }
        let m_rat: Rat = Rat::from(m.clone());
        let ints: Vec<i64> = fractions
            .iter()
            .map(|q| i64::try_from((q * &m_rat).numerator()).map_err(|_| value_error("a large divisor".to_string())))
            .collect::<PolyResult<_>>()?;
        let m_i64: i64 = i64::try_from(m).map_err(|_| value_error("a large divisor".to_string()))?;
        divisors.push((beta, m_i64, ints));
    }
    Ok((divisors, places))
}

/// The order (in units of `z`) to which the branches above the roots of
/// each factor must be known (`_orders_needed`).
fn orders_needed(
    field: &Field,
    basis: &[FieldElement],
    places: &[Place],
    coefficients: &[i64],
    n: i64,
    k: &AlgebraicField,
) -> PolyResult<HashMap<FactorKey, i64>> {
    let mut k_of: HashMap<FactorKey, i64> = HashMap::new();
    for (place, c) in places.iter().zip(coefficients) {
        let needed: i64 = if *c < 0 {
            -floor_div(n * c, place.ramification())
        } else {
            0
        };
        let entry = k_of.entry(place.key.clone()).or_insert(0);
        *entry = (*entry).max(needed);
    }
    let mut orders: HashMap<FactorKey, i64> = HashMap::new();
    for (place, c) in places.iter().zip(coefficients) {
        let e: i64 = place.ramification();
        let multiplicity: i64 = max_multiplicity(field, basis, &place.g, k)? as i64;
        let required: i64 = n * c + k_of[&place.key] * e;
        // the series of y to order `required` plus the denominators' orders
        let needed: i64 = -floor_div(-(required + e * (multiplicity + 1) + 2), e) + 1;
        let entry = orders.entry(place.key.clone()).or_insert(0);
        *entry = (*entry).max(needed);
    }
    Ok(orders)
}

// ---------------------------------------------------------------------
// The algorithm

/// The pieces of the integral (`CurveIntegral`): the elementary part of
/// the Hermite reduction and the logarithms in `z`, the field of the
/// residues, the verdict (`Some(true)` proved not elementary,
/// `Some(false)` found, `None` undecided) and the integrand left.
pub struct CurveIntegral<'a> {
    pub hermite: HermitePart<'a>,
    pub logs: Vec<LogTerm>,
    pub residue_field: AlgebraicField,
    pub nonelementary: Option<bool>,
    pub remainder: Option<FieldElement>,
    residue_field_function_field: Option<Field>,
}

impl CurveIntegral<'_> {
    fn theta(&self, theta: Option<&Basic>) -> PolyResult<Basic> {
        let k = &self.residue_field;
        if let Some(t) = theta {
            return Ok(t.clone());
        }
        if k.degree == 1 {
            return Ok(Basic::zero());
        }
        if matches!(k.alpha_expr.head(), Head::Symbol(_)) {
            return Err(value_error(format!(
                "the generator of {} has no expression (a root of its minimal polynomial which is not real and not in radicals)",
                k.name()
            )));
        }
        Ok(k.alpha_expr.clone())
    }

    /// The elementary part found, in `x` and the original radicals
    /// (`elementary_part`).
    pub fn elementary_part(&self, theta: Option<&Basic>) -> PolyResult<Basic> {
        let t = &self.hermite.transformed;
        let generator_expr: Basic = self.theta(theta)?;
        let mut total: Basic = t.to_x_expr(&t.field, &self.hermite.reduced.g, &generator_expr)?;
        let k = &self.residue_field;
        let x: &Basic = &t.integrand.x;
        for term in &self.logs {
            let e_k: &Field = self.residue_field_function_field.as_ref().unwrap_or(&t.field);
            let den: Fraction = t.to_x(
                &FracElement {
                    numer: term.den.clone(),
                    denom: vec![k.one()],
                },
                k,
            )?;
            let quotient: Basic = mul(&[
                t.to_x_expr(e_k, &term.numerator, &generator_expr)?,
                pow(&fraction_expr(&den, x, k, &generator_expr)?, &int(-1))?,
            ])?;
            let u: Basic = crate::polys::polytools::cancel(&quotient, &[])?;
            let coefficient: Basic = mul(&[
                number_expr(&term.beta, &generator_expr)?,
                Basic::rational(Int::ONE, Int::from(term.n * term.m))?,
            ])?;
            let logarithm: Basic = crate::functions::elementary::exponential::log(&u)?;
            total = add(&[total, mul(&[coefficient, logarithm])?])?;
        }
        Ok(total)
    }

    /// The remaining integrand in `x` (`remainder_expr`).
    pub fn remainder_expr(&self, theta: Option<&Basic>) -> PolyResult<Basic> {
        let Some(remainder) = &self.remainder else {
            return Ok(Basic::zero());
        };
        let t = &self.hermite.transformed;
        let x: &Basic = &t.integrand.x;
        let field: &Field = self.residue_field_function_field.as_ref().unwrap_or(&t.field);
        // the remainder is a function of z integrated in dz = -dx/(x - N)^2
        let back: Basic = t.to_x_expr(field, remainder, &self.theta(theta)?)?;
        let shift: Basic = pow(&add(&[x.clone(), int(-t.n)])?, &int(-2))?;
        crate::polys::polytools::cancel(&mul(&[int(-1), back, shift])?, &[])
    }
}

fn plain(hermite: HermitePart<'_>, k: AlgebraicField, verdict: Option<bool>, logs: Vec<LogTerm>) -> CurveIntegral<'_> {
    let remainder: Option<FieldElement> = hermite.remainder.clone();
    CurveIntegral {
        hermite,
        logs,
        residue_field: k,
        nonelementary: verdict,
        remainder,
        residue_field_function_field: None,
    }
}

/// Trager's algorithm on the curve of the radicals of the integrand
/// (`integrate_on_curve`).
pub fn integrate_on_curve(integrand: &AlgebraicIntegrand, logarithms: bool) -> PolyResult<CurveIntegral<'_>> {
    let hermite: HermitePart = hermite_part(integrand)?;
    match hermite.nonelementary {
        Some(false) => {
            let mut result = plain(hermite, rationals(), Some(false), Vec::new());
            result.remainder = None;
            return Ok(result);
        }
        Some(true) => return Ok(plain(hermite, rationals(), Some(true), Vec::new())),
        None => {}
    }
    if !logarithms {
        return Ok(plain(hermite, rationals(), None, Vec::new()));
    }
    let t = &hermite.transformed;
    let field = &t.field;
    let basis = &hermite.basis;
    let reduced = &hermite.reduced;
    let remainder: FieldElement = hermite.remainder.clone().expect("a remainder");
    let places: Vec<Place> = places_of(t, basis, reduced, &rationals(), &HashMap::new())?;
    let k: AlgebraicField = residue_field(&places)?;
    let e_k: Field = FunctionField::new(&field.curve, &field.x, &field.y, k.clone())?;
    let (mut divisors, mut support) = divisors_of(t, basis, reduced, &k, &HashMap::new())?;
    if divisors.is_empty() {
        // no residue and a nonzero remainder: a differential of the first kind
        let mut result = plain(hermite, k, Some(true), Vec::new());
        result.residue_field_function_field = Some(e_k);
        return Ok(result);
    }
    let mut logs: Vec<LogTerm> = Vec::new();
    let mut undecided: bool = false;
    let mut precision: HashMap<FactorKey, i64> = HashMap::new();
    for j in 0..divisors.len() {
        let mut found: Option<(FieldElement, KPoly)> = None;
        for n in 1..=TORSION_BOUND as i64 {
            let needed = orders_needed(field, basis, &support, &divisors[j].2, n, &k)?;
            if needed
                .iter()
                .any(|(g, order)| *order > precision.get(g).copied().unwrap_or(0))
            {
                for (g, order) in &needed {
                    let entry = precision.entry(g.clone()).or_insert(0);
                    *entry = (*entry).max(*order);
                }
                (divisors, support) = divisors_of(t, basis, reduced, &k, &precision)?;
            }
            let (beta, m, c) = divisors[j].clone();
            found = function_with_divisor(t, basis, &hermite.orders, &support, &c, n, &k, &e_k)?;
            if let Some((numerator, den)) = &found {
                logs.push(LogTerm {
                    beta,
                    n,
                    m,
                    numerator: numerator.clone(),
                    den: den.clone(),
                });
                break;
            }
        }
        if found.is_none() {
            undecided = true;
        }
    }
    // the exact check: h - sum beta/(n m) u'/u
    let kz = &e_k.kx;
    let mut h_k: FieldElement = Element {
        coefficients: remainder.coefficients.clone(),
    };
    for term in &logs {
        let u: FieldElement = e_k.scaled(&term.numerator, &kz.invert(&kz.from_ring(&term.den))?);
        let scale: Number = k.mul(
            &term.beta,
            &k.from_rational(&Rat::from_parts_signed(Int::ONE, Int::from(term.n * term.m))),
        );
        let ratio: FieldElement = e_k.mul(&e_k.derivative(&u)?, &e_k.invert(&u)?);
        h_k = e_k.sub(&h_k, &e_k.scaled(&ratio, &kz.from_ring(&[scale])));
    }
    let verdict: Option<bool> = if e_k.is_zero(&h_k) {
        Some(false)
    } else if undecided {
        None
    } else {
        Some(true)
    };
    let left: Option<FieldElement> = if verdict == Some(false) { None } else { Some(h_k) };
    let mut result = plain(hermite, k, verdict, logs);
    result.remainder = left;
    result.residue_field_function_field = Some(e_k);
    Ok(result)
}

// ---------------------------------------------------------------------
// The public functions

/// `(F, h)` with `f == F' + h` (`curve_reduce`); `None` when `f` is not an
/// algebraic function the parser understands.
pub fn curve_reduce(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let integrand = match AlgebraicIntegrand::new(f, x) {
        Ok(i) => i,
        Err(crate::error::PolyError::Value(_)) => return Ok(None),
        Err(other) => return Err(other),
    };
    match integrate_on_curve(&integrand, true) {
        Ok(result) => Ok(Some((result.elementary_part(None)?, result.remainder_expr(None)?))),
        Err(crate::error::PolyError::Value(_)) => Ok(None),
        Err(other) => Err(other),
    }
}

/// An antiderivative of `f` by Trager's algorithm, `None` when not
/// elementary, not found or `f` is not such a function
/// (`curve_antiderivative`).
pub fn curve_antiderivative(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let integrand = match AlgebraicIntegrand::new(f, x) {
        Ok(i) => i,
        Err(crate::error::PolyError::Value(_)) => return Ok(None),
        Err(other) => return Err(other),
    };
    let result = match integrate_on_curve(&integrand, true) {
        Ok(r) => r,
        Err(crate::error::PolyError::Value(_)) => return Ok(None),
        Err(other) => return Err(other),
    };
    if result.nonelementary != Some(false) || result.remainder.is_some() {
        return Ok(None);
    }
    match result.elementary_part(None) {
        Ok(e) => Ok(Some(e)),
        Err(crate::error::PolyError::Value(_)) => Ok(None),
        Err(other) => Err(other),
    }
}

/// Whether the integral of `f` is not elementary: `Some(true)` proved,
/// `Some(false)` found, `None` undecided or not such a function
/// (`is_nonelementary_on_curve`).
pub fn is_nonelementary_on_curve(f: &Basic, x: &Basic) -> PolyResult<Option<bool>> {
    let integrand = match AlgebraicIntegrand::new(f, x) {
        Ok(i) => i,
        Err(crate::error::PolyError::Value(_)) => return Ok(None),
        Err(other) => return Err(other),
    };
    match integrate_on_curve(&integrand, true) {
        Ok(result) => Ok(result.nonelementary),
        Err(crate::error::PolyError::Value(_)) => Ok(None),
        Err(other) => Err(other),
    }
}
