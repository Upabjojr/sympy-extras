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
