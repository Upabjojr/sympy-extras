//! The rational univariate representation of a zero-dimensional ideal
//! (Rouillier), ported from `sympy_extras.polys.rur`: the separating form
//! found by a modular test (the first small forms of
//! `separating_forms`, then random forms from Python's `random.Random`),
//! the characteristic polynomial from the traces by Newton's identities,
//! its squarefree part, and the coordinates as `g_v/g_1` from the Horner
//! polynomials. The exact solutions are the roots in SymPy's order
//! substituted in the coordinate polynomials. `coordinates` (it needs
//! `radsimp`) and `numerical_solutions` (mpmath's root finding) are not
//! ported.

use crate::core::basic::Basic;
use crate::core::symbol::Symbol;
use crate::error::{arithmetic_error, PolyError, PolyResult};
use crate::numbers::{Int, Rat};

use super::domains::{Domain, FiniteField, RationalField, QQ};
use super::hermite::{algebra, inertia, Algebra, Rows, System, Vector};
use super::ideals::{dense_to_expr, separating_forms, PolyLike};
use super::polyclasses::DensePoly;
use super::rings::{Monomial, PolyElement};

/// A dense polynomial over the rationals.
pub type QPoly = DensePoly<RationalField>;

/// A zero of the ideal: the values of the variables, in their order.
pub type Point = Vec<(Basic, Basic)>;

fn constant(c: Rat, t: &Basic) -> QPoly {
    DensePoly::new(&[c], QQ, Symbol(t.clone()))
}

/// A dense polynomial over the rationals as an expression in its variable.
pub fn to_expr(p: &QPoly) -> PolyResult<Basic> {
    dense_to_expr(&p.rep, &QQ, &p.gen.0)
}

/// The zeros of a representation: the roots of `polynomial` in SymPy's
/// order substituted in the coordinate polynomials, expanded
/// (`exact_solutions`).
pub fn exact_solutions(polynomial: &QPoly, polys: &[QPoly], symbols: &[Basic], real: bool) -> PolyResult<Vec<Point>> {
    let t: &Basic = &polynomial.gen.0;
    if !t.is_symbol() {
        return Err(PolyError::Type(format!("the polynomial is not in a symbol: {}", t)));
    }
    let roots: Vec<Basic> = super::complexrootof::polynomial_roots(&polynomial.rep, t, true, real)?
        .iter()
        .map(super::complexrootof::real_roots_indexed)
        .collect::<PolyResult<_>>()?;
    let exprs: Vec<Basic> = polys.iter().map(to_expr).collect::<PolyResult<_>>()?;
    let mut points: Vec<Point> = Vec::new();
    for root in &roots {
        let mut point: Point = Vec::new();
        for (s, e) in symbols.iter().zip(&exprs) {
            point.push((
                s.clone(),
                crate::core::function::expand_default(&crate::core::subs::subs(e, t, root)?)?,
            ));
        }
        points.push(point);
    }
    Ok(points)
}

/// The rational univariate representation (`RationalUnivariateRepresentation`).
#[derive(Debug, Clone)]
pub struct RationalUnivariateRepresentation {
    pub symbols: Vec<Basic>,
    pub t: Basic,
    pub form: Basic,
    pub polynomial: QPoly,
    pub characteristic: QPoly,
    pub denominator: QPoly,
    pub numerators: Vec<QPoly>,
}

impl RationalUnivariateRepresentation {
    /// The coordinates as polynomials in `t` of degree less than that of
    /// `polynomial` (`polynomials`).
    pub fn polynomials(&self) -> PolyResult<Vec<QPoly>> {
        if self.polynomial.degree() <= 0 {
            return Ok(self.numerators.iter().map(|_| constant(Rat::ZERO, &self.t)).collect());
        }
        let inverse: QPoly = self.denominator.invert(&self.polynomial)?;
        self.numerators
            .iter()
            .map(|g| g.mul(&inverse)?.rem(&self.polynomial))
            .collect()
    }

    /// The zeros, each as the values of the variables (`solutions`).
    pub fn solutions(&self, real: bool) -> PolyResult<Vec<Point>> {
        if self.polynomial.degree() <= 0 {
            return Ok(Vec::new());
        }
        exact_solutions(&self.polynomial, &self.polynomials()?, &self.symbols, real)
    }

    /// The multiplicity of the zero at a root of `polynomial` (`multiplicity`).
    pub fn multiplicity(&self, root: &Basic) -> PolyResult<usize> {
        let coefficients: Vec<Rat> = super::numberfields::minpoly::minimal_polynomial_coeffs(root)?
            .into_iter()
            .map(Rat::from)
            .collect();
        let minimal: QPoly = DensePoly::new(&coefficients, QQ, Symbol(self.t.clone()));
        let mut count: usize = 0;
        let mut p: QPoly = self.characteristic.clone();
        while p.degree() > 0 {
            let (q, r) = p.div(&minimal)?;
            if !r.is_zero() {
                break;
            }
            p = q;
            count += 1;
        }
        Ok(count)
    }

    /// `coordinates` needs `radsimp` (not ported).
    pub fn coordinates(&self, _root: &Basic) -> PolyResult<Vec<Basic>> {
        Err(PolyError::NotImplemented(
            "RationalUnivariateRepresentation.coordinates needs radsimp".to_string(),
        ))
    }

    /// `numerical_solutions` needs mpmath's root finding (not ported).
    pub fn numerical_solutions(&self, _digits: usize) -> PolyResult<Vec<Point>> {
        Err(PolyError::NotImplemented(
            "RationalUnivariateRepresentation.numerical_solutions".to_string(),
        ))
    }
}

/// The characteristic polynomial from the power sums (Newton's
/// identities, `_newton_characteristic`).
fn newton_characteristic(power_sums: &[Rat], t: &Basic) -> QPoly {
    let d: usize = power_sums.len();
    let mut e: Vec<Rat> = vec![Rat::ONE];
    for k in 1..=d {
        let mut total: Rat = Rat::ZERO;
        for i in 1..=k {
            let term: Rat = &e[k - i] * &power_sums[i - 1];
            if i % 2 == 1 {
                total += term;
            } else {
                total -= term;
            }
        }
        e.push(total / Rat::from(Int::from(k)));
    }
    let rep: Vec<Rat> = (0..=d)
        .map(|k| if k % 2 == 0 { e[k].clone() } else { -e[k].clone() })
        .collect();
    DensePoly::new(&rep, QQ, Symbol(t.clone()))
}

/// The prime of the modular separating test.
const PRIME: u64 = (1u64 << 61) - 1;
/// How many small forms are tried before random ones.
const SMALL_FORMS: usize = 6;

/// The candidate separating forms (`_candidates`), lazily.
struct Candidates {
    small: std::vec::IntoIter<Basic>,
    symbols: Vec<Basic>,
    bound: i64,
    rng: crate::random::PyRandom,
}

impl Candidates {
    fn new(symbols: &[Basic], count: usize) -> PolyResult<Candidates> {
        let small: Vec<Basic> = separating_forms(symbols, count)?
            .into_iter()
            .take(SMALL_FORMS)
            .collect();
        let pairs: i64 = (count * count.saturating_sub(1) / 2) as i64;
        let bound: i64 = 2i64.max(if symbols.len() == 2 { pairs } else { count as i64 });
        Ok(Candidates {
            small: small.into_iter(),
            symbols: symbols.to_vec(),
            bound,
            rng: crate::random::PyRandom::from_int(&Int::from(count)),
        })
    }

    fn next(&mut self) -> PolyResult<Basic> {
        if let Some(form) = self.small.next() {
            return Ok(form);
        }
        let mut terms: Vec<Basic> = vec![self.symbols[0].clone()];
        for s in &self.symbols[1..] {
            terms.push(crate::core::mul::mul(&[
                Basic::int(self.rng.randint(-self.bound, self.bound)),
                s.clone(),
            ])?);
        }
        crate::core::add::add(&terms)
    }
}

fn inverse_mod(a: &Int, p: u64) -> Option<u64> {
    let r: Int = a % Int::from(p);
    let r: u64 = u64::try_from(&((r + Int::from(p)) % Int::from(p))).expect("a residue");
    if r == 0 {
        return None;
    }
    // Fermat
    let field: FiniteField = FiniteField::new(p).expect("a prime");
    Some(field.pow(&r, p - 2))
}

/// Whether the form of the transposed multiplication matrix is
/// separating, decided modulo a prime (`_is_separating`).
fn is_separating(trace: &[Rat], transposed: &Rows, one_index: usize, count: usize) -> bool {
    let p: u64 = PRIME;
    let field: FiniteField = FiniteField::new(p).expect("a prime");
    let modulo = |q: &Rat| -> Option<u64> {
        let inverse: u64 = inverse_mod(&Int::from(q.denominator().clone()), p)?;
        let n: Int = q.numerator() % Int::from(p);
        let n: u64 = u64::try_from(&((n + Int::from(p)) % Int::from(p))).expect("a residue");
        Some(field.mul(&n, &inverse))
    };
    let current: Option<Vec<u64>> = trace.iter().map(modulo).collect();
    let rows: Option<Vec<Vec<u64>>> = transposed.iter().map(|row| row.iter().map(modulo).collect()).collect();
    let (Some(mut current), Some(rows)) = (current, rows) else {
        return false;
    };
    let d: usize = trace.len();
    let mut power_sums: Vec<u64> = Vec::new();
    for _ in 0..d {
        current = rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&current)
                    .fold(0u64, |s, (a, b)| field.add(&s, &field.mul(a, b)))
            })
            .collect();
        power_sums.push(current[one_index]);
    }
    let mut e: Vec<u64> = vec![1];
    for k in 1..=d {
        let mut total: u64 = 0;
        for i in 1..=k {
            let term: u64 = field.mul(&e[k - i], &power_sums[i - 1]);
            total = if i % 2 == 1 {
                field.add(&total, &term)
            } else {
                field.sub(&total, &term)
            };
        }
        let inverse: u64 = inverse_mod(&Int::from(k), p).expect("k below the prime");
        e.push(field.mul(&total, &inverse));
    }
    let coefficients: Vec<u64> = (0..=d)
        .map(|k| if k % 2 == 0 { e[k] } else { field.neg(&e[k]) })
        .collect();
    let f: Vec<u64> = super::densebasic::dup_strip(&coefficients, &field);
    let derivative: Vec<u64> = super::densetools::dup_diff(&f, 1, &field);
    let g: Vec<u64> = super::euclidtools::dup_gcd(&f, &derivative, &field);
    let degree: isize = super::densebasic::dup_degree(&g);
    d as isize - degree == count as isize
}

fn dot(a: &[Rat], b: &[Rat]) -> Rat {
    a.iter().zip(b).fold(Rat::ZERO, |s, (x, y)| s + x * y)
}

/// The rational univariate representation of a zero-dimensional system
/// (Rouillier's algorithm, `rur`); `t` the variable (`t` by default).
pub fn rur(system: &System, t: Option<&Basic>) -> PolyResult<RationalUnivariateRepresentation> {
    let tt: Basic = t.cloned().unwrap_or_else(|| Symbol::new("t", &[]).0);
    let a: Algebra = match algebra(system) {
        Ok(a) => a,
        Err(PolyError::Value(message)) => {
            return Err(PolyError::NotImplemented(format!(
                "the coefficients of the system are not rational: {}",
                message
            )));
        }
        Err(other) => return Err(other),
    };
    let symbols: Vec<Basic> = a.ideal.symbols.clone();
    if a.empty {
        let one: QPoly = constant(Rat::ONE, &tt);
        return Ok(RationalUnivariateRepresentation {
            form: symbols.first().cloned().unwrap_or_else(Basic::zero),
            numerators: symbols.iter().map(|_| constant(Rat::ZERO, &tt)).collect(),
            symbols,
            t: tt.clone(),
            polynomial: one.clone(),
            characteristic: one,
            denominator: constant(Rat::ZERO, &tt),
        });
    }
    let d: usize = a.dimension();
    let (positive, negative) = inertia(&a.hermite(&a.trace));
    let count: usize = (positive + negative) as usize;
    let coordinates: Vec<Vector> = symbols
        .iter()
        .map(|s| -> PolyResult<Vector> { a.coordinates(&a.element(&PolyLike::Expr(s))?) })
        .collect::<PolyResult<_>>()?;
    let origin: Monomial = vec![0; symbols.len()];
    let one_index: usize = a.index_of(&origin).expect("the monomial 1");
    let mut candidates: Candidates = Candidates::new(&symbols, count)?;
    loop {
        let form: Basic = candidates.next()?;
        let element: PolyElement<RationalField> = a.element(&PolyLike::Expr(&form))?;
        let transposed: Rows = a.transposed_multiplication(&element)?;
        if !is_separating(&a.trace, &transposed, one_index, count) {
            continue;
        }
        let mut forms: Vec<Vector> = vec![a.trace.clone()];
        for _ in 0..d {
            let next: Vector = a.compose(forms.last().expect("a form"), &transposed);
            forms.push(next);
        }
        let power_sums: Vec<Rat> = forms[1..].iter().map(|f| f[one_index].clone()).collect();
        let characteristic: QPoly = newton_characteristic(&power_sums, &tt);
        let squarefree: QPoly = characteristic.quo(&characteristic.gcd(&characteristic.diff(1))?)?;
        let delta: usize = squarefree.degree() as usize;
        if delta != count {
            return Err(arithmetic_error("the modular separating test is a proof".to_string()));
        }
        let horner: Vec<QPoly> = (0..delta)
            .map(|j| DensePoly::new(&squarefree.rep[..j + 1], QQ, Symbol(tt.clone())))
            .collect();
        let g_of = |values: &[Rat]| -> PolyResult<QPoly> {
            let mut total: QPoly = constant(Rat::ZERO, &tt);
            for j in 0..delta {
                total = total.add(&horner[delta - 1 - j].mul_ground(&values[j]))?;
            }
            Ok(total)
        };
        let denominator: QPoly = g_of(&forms.iter().map(|f| f[one_index].clone()).collect::<Vec<Rat>>())?;
        let numerators: Vec<QPoly> = coordinates
            .iter()
            .map(|v| g_of(&forms.iter().map(|f| dot(f, v)).collect::<Vec<Rat>>()))
            .collect::<PolyResult<_>>()?;
        return Ok(RationalUnivariateRepresentation {
            symbols,
            t: tt,
            form,
            polynomial: squarefree,
            characteristic,
            denominator,
            numerators,
        });
    }
}
