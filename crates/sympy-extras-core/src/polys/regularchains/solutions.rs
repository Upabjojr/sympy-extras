//! Port of `sympy_extras/polys/regularchains/solutions.py`: the exact
//! solutions of a squarefree regular chain without free variables, solved
//! from the bottom with algebraic numbers. A coordinate whose polynomial
//! has rational coefficients at the point found so far is one of its root
//! objects; a linear one is a division; otherwise the coordinate is a root
//! of an eliminant (iterated resultants), told from the others by
//! matching numerical roots of the polynomial at the point (mpmath's
//! `polyroots`, ported below with its Durand-Kerner iteration, its
//! starting values and its cleanup) with the exact values of the root
//! objects. The match is refused (`NotImplemented`) when it is not clear.

#![allow(clippy::needless_range_loop)]

use crate::core::basic::{Basic, Head};
use crate::core::evalf::Real;
use crate::error::{not_ported, PolyError, PolyResult};
use crate::numbers::*;
use crate::polys::complexrootof::{algebraic, algebraic_number, complex_value, polynomial_roots, real_roots_indexed};
use crate::polys::domains::QQ;
use crate::polys::polytools::Poly;
use crate::polys::sqfreetools::dup_sqf_part;

/// The digits of the numerical matching, over those of the coefficients.
const DIGITS: usize = 40;

/// A point: the coordinates found so far.
pub type Point = Vec<(Basic, Basic)>;

/// A complex number of two binary floats, the arithmetic at an explicit
/// precision.
#[derive(Clone, Debug)]
struct C {
    re: Real,
    im: Real,
}

fn real(v: f64, prec: usize) -> Real {
    Real::try_from(v).expect("a finite float").with_precision(prec).value()
}

fn rat_real(r: &Rat, prec: usize) -> Real {
    crate::core::evalf::real_rat(r, prec).with_precision(prec).value()
}

impl C {
    fn zero(prec: usize) -> C {
        C {
            re: real(0.0, prec),
            im: real(0.0, prec),
        }
    }
    fn at(&self, prec: usize) -> C {
        C {
            re: self.re.clone().with_precision(prec).value(),
            im: self.im.clone().with_precision(prec).value(),
        }
    }
    fn add(&self, o: &C) -> C {
        C {
            re: &self.re + &o.re,
            im: &self.im + &o.im,
        }
    }
    fn sub(&self, o: &C) -> C {
        C {
            re: &self.re - &o.re,
            im: &self.im - &o.im,
        }
    }
    fn mul(&self, o: &C) -> C {
        C {
            re: &self.re * &o.re - &self.im * &o.im,
            im: &self.re * &o.im + &self.im * &o.re,
        }
    }
    fn is_zero(&self) -> bool {
        self.re.repr().significand().is_zero() && self.im.repr().significand().is_zero()
    }
    fn div(&self, o: &C) -> C {
        let d: Real = &o.re * &o.re + &o.im * &o.im;
        C {
            re: (&self.re * &o.re + &self.im * &o.im) / &d,
            im: (&self.im * &o.re - &self.re * &o.im) / &d,
        }
    }
    fn neg(&self) -> C {
        C {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }
    fn abs(&self) -> Real {
        (&self.re * &self.re + &self.im * &self.im).sqrt()
    }
    fn from_complex(z: &crate::core::evalf::Complex, prec: usize) -> C {
        let (re, im) = crate::core::evalf::parts(z);
        C { re, im }.at(prec)
    }
}

fn abs_real(r: &Real) -> Real {
    if *r < Real::ZERO {
        -r.clone()
    } else {
        r.clone()
    }
}

fn is_zero_real(r: &Real) -> bool {
    r.repr().significand().is_zero()
}

/// mpmath's `polyroots` without `error`: `None` for `NoConvergence`,
/// an error for a zero leading coefficient.
fn polyroots(coeffs: &[C], maxsteps: usize, extraprec: usize, prec: usize) -> PolyResult<Option<Vec<C>>> {
    if coeffs.len() <= 1 {
        if coeffs.is_empty() || coeffs[0].is_zero() {
            return Err(PolyError::Value(
                "Input to polyroots must not be the zero polynomial".to_string(),
            ));
        }
        return Ok(Some(vec![]));
    }
    let tol: Real = real(1.0, prec) / real(2.0, prec).powi((prec - 1).into());
    let wp: usize = prec + extraprec;
    let deg: usize = coeffs.len() - 1;
    let lead: C = coeffs[0].at(wp);
    if lead.is_zero() {
        return Err(PolyError::ZeroDivision("a zero leading coefficient".to_string()));
    }
    let one: C = C {
        re: real(1.0, wp),
        im: real(0.0, wp),
    };
    let monic: Vec<C> = if lead.re == one.re && is_zero_real(&lead.im) {
        coeffs.iter().map(|c| c.at(wp)).collect()
    } else {
        coeffs.iter().map(|c| c.at(wp).div(&lead)).collect()
    };
    let f = |x: &C| -> C {
        let mut p: C = monic[0].clone();
        for c in &monic[1..] {
            p = c.add(&x.mul(&p));
        }
        p
    };
    // Python's (0.4+0.9j)**n in doubles (CPython's c_powu)
    let start = |n: usize| -> (f64, f64) {
        let (mut r, mut p): ((f64, f64), (f64, f64)) = ((1.0, 0.0), (0.4, 0.9));
        let mut mask: usize = 1;
        while mask > 0 && n >= mask {
            if n & mask != 0 {
                r = (r.0 * p.0 - r.1 * p.1, r.0 * p.1 + r.1 * p.0);
            }
            mask <<= 1;
            p = (p.0 * p.0 - p.1 * p.1, p.0 * p.1 + p.1 * p.0);
        }
        r
    };
    let mut roots: Vec<C> = (0..deg)
        .map(|n| {
            let (a, b) = start(n);
            C {
                re: real(a, wp),
                im: real(b, wp),
            }
        })
        .collect();
    let mut err: Vec<Real> = vec![real(1.0, wp); deg];
    let max_err = |err: &[Real]| -> Real {
        err.iter()
            .cloned()
            .fold(real(0.0, wp), |m, e| if e > m { e } else { m })
    };
    for _ in 0..maxsteps {
        if abs_real(&max_err(&err)) < tol {
            break;
        }
        for i in 0..deg {
            let p: C = roots[i].clone();
            let mut x: C = f(&p);
            for j in 0..deg {
                if i != j {
                    let d: C = p.sub(&roots[j]);
                    if d.is_zero() {
                        continue;
                    }
                    x = x.div(&d);
                }
            }
            roots[i] = p.sub(&x);
            err[i] = x.abs();
        }
    }
    if abs_real(&max_err(&err)) >= tol {
        return Ok(None);
    }
    for r in roots.iter_mut() {
        if r.abs() < tol {
            *r = C::zero(wp);
        } else if abs_real(&r.im) < tol {
            *r = C {
                re: r.re.clone(),
                im: real(0.0, wp),
            };
        } else if abs_real(&r.re) < tol {
            *r = C {
                re: real(0.0, wp),
                im: r.im.clone(),
            };
        }
    }
    roots.sort_by(|a, b| {
        (abs_real(&a.im), a.re.clone())
            .partial_cmp(&(abs_real(&b.im), b.re.clone()))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(Some(roots.iter().map(|r| r.at(prec)).collect()))
}

fn sign_real(r: &Real) -> i32 {
    if *r > Real::ZERO {
        1
    } else if *r < Real::ZERO {
        -1
    } else {
        0
    }
}

/// The roots as SymPy's `Poly.nroots` finds and sorts them
/// (`_numerical_roots`).
fn numerical_roots(numeric: &[C], prec: usize) -> PolyResult<Option<Vec<C>>> {
    let degree: usize = numeric.len().saturating_sub(1);
    for extra in [10, 15] {
        let found: Vec<C> = match polyroots(numeric, 200, degree * extra, prec) {
            Ok(Some(found)) => found,
            Ok(None) => continue,
            Err(PolyError::ZeroDivision(_)) | Err(PolyError::Value(_)) => return Ok(None),
            Err(other) => return Err(other),
        };
        let mut roots: Vec<C> = found;
        roots.sort_by(|a, b| {
            let ka = (
                u8::from(!is_zero_real(&a.im)),
                a.re.clone(),
                abs_real(&a.im),
                sign_real(&a.im),
            );
            let kb = (
                u8::from(!is_zero_real(&b.im)),
                b.re.clone(),
                abs_real(&b.im),
                sign_real(&b.im),
            );
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
        return Ok(Some(roots));
    }
    Ok(None)
}

/// A root of a polynomial with rational coefficients (`_Root`).
#[derive(Clone)]
struct Root {
    exact: Basic,
    numeric: C,
    real: bool,
}

/// The coordinates found so far (`_Point`).
#[derive(Clone)]
struct PointData {
    exact: Point,
    numeric: Vec<(Basic, C)>,
    real: bool,
}

impl PointData {
    fn extended(&self, v: &Basic, exact: Basic, numeric: C, real: bool) -> PointData {
        let mut first: Point = self.exact.clone();
        first.push((v.clone(), exact));
        let mut second: Vec<(Basic, C)> = self.numeric.clone();
        second.push((v.clone(), numeric));
        PointData {
            exact: first,
            numeric: second,
            real: self.real && real,
        }
    }
}

type Terms = Vec<(Vec<u32>, Rat)>;

/// A polynomial of the chain in its main variable, the coefficients as
/// polynomials with rational coefficients in the variables below
/// (`_Polynomial`).
struct Polynomial {
    coefficients: Vec<Terms>,
    below: Vec<Basic>,
    exprs: Vec<Basic>,
}

fn terms_expr(terms: &Terms, gens: &[Basic]) -> PolyResult<Basic> {
    if gens.is_empty() {
        return Ok(Basic::from_rational(
            terms
                .iter()
                .find(|(m, _)| m.is_empty())
                .map(|(_, c)| c.clone())
                .unwrap_or(Rat::ZERO),
        ));
    }
    Poly::from_dict(terms, gens)?.as_expr()
}

impl Polynomial {
    fn new(p: &Basic, v: &Basic, below: &[Basic]) -> PolyResult<Polynomial> {
        let mut gens: Vec<Basic> = vec![v.clone()];
        gens.extend(below.iter().cloned());
        let poly: Poly = Poly::new(p, &gens)?;
        let n: usize = usize::try_from(poly.degree(0)?)
            .map_err(|_| crate::error::value_error("the zero polynomial".to_string()))?;
        let mut terms: Vec<Terms> = vec![Vec::new(); n + 1];
        for (monomial, c) in poly.terms()? {
            let Some(value) = c.as_rational().filter(|_| c.is_rational_class()) else {
                return Err(not_ported(&format!(
                    "the chain has a coefficient which is not rational: {}",
                    c
                )));
            };
            terms[n - monomial[0] as usize].push((monomial[1..].to_vec(), value.clone()));
        }
        let exprs: Vec<Basic> = terms.iter().map(|t| terms_expr(t, below)).collect::<PolyResult<_>>()?;
        Ok(Polynomial {
            coefficients: terms,
            below: below.to_vec(),
            exprs,
        })
    }

    fn numeric(&self, point: &[(Basic, C)], prec: usize) -> Vec<C> {
        let values: Vec<C> = self
            .below
            .iter()
            .map(|s| point.iter().find(|(k, _)| k == s).expect("a coordinate").1.clone())
            .collect();
        let mut found: Vec<C> = Vec::new();
        for terms in &self.coefficients {
            let mut total: C = C::zero(prec);
            for (monomial, c) in terms {
                let mut term: C = C {
                    re: rat_real(c, prec),
                    im: real(0.0, prec),
                };
                for (value, e) in values.iter().zip(monomial) {
                    for _ in 0..*e {
                        term = term.mul(value);
                    }
                }
                total = total.add(&term);
            }
            found.push(total);
        }
        found
    }
}

fn has_root_objects(e: &Basic) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::ComplexRootOf(_) | Head::IndexedRoot))
}

fn roots_of(coefficients: &[Rat], v: &Basic, prec: usize) -> PolyResult<Vec<Root>> {
    let mut found: Vec<Root> = Vec::new();
    for root in polynomial_roots(coefficients, v, false, false)? {
        let nodes: Vec<Basic> = root
            .preorder()
            .into_iter()
            .filter(|n| matches!(n.head(), Head::ComplexRootOf(_)))
            .collect();
        if nodes.is_empty() {
            let Some(a) = algebraic_number(&root)? else {
                return Err(not_ported(&format!("no value for the root {}", root)));
            };
            let numeric: C = C::from_complex(&complex_value(&a, prec)?, prec);
            found.push(Root {
                exact: root,
                numeric,
                real: a.is_real(),
            });
            continue;
        }
        let node: Basic = nodes[0].clone();
        let a = algebraic(&node)?;
        let scale: Basic = if root == node {
            Basic::one()
        } else {
            crate::core::mul::mul(&[root.clone(), crate::core::power::pow(&node, &Basic::negative_one())?])?
        };
        let Some(s) = scale
            .as_rational()
            .filter(|v| scale.is_rational_class() && v.is_int())
            .cloned()
        else {
            return Err(not_ported(&format!("unexpected form of the root {}", root)));
        };
        let base: C = C::from_complex(&complex_value(&a, prec)?, prec);
        let numeric: C = base.mul(&C {
            re: rat_real(&s, prec),
            im: real(0.0, prec),
        });
        let form: Option<Basic> =
            crate::timeout::attempt_within(Some(5.0), || crate::polys::roots::radical_form(&node))?.flatten();
        let exact: Basic = match form {
            Some(f) => f,
            None => real_roots_indexed(&node)?,
        };
        let exact: Basic = if s.is_one() {
            exact
        } else {
            crate::core::mul::mul(&[scale, exact])?
        };
        found.push(Root {
            exact,
            numeric,
            real: a.is_real(),
        });
    }
    Ok(found)
}

fn to_f64(r: &Real) -> f64 {
    r.to_f64().value()
}

/// The candidate whose value is `value`, when it is clearly the only one
/// (`_nearest`).
fn nearest(candidates: &[Root], value: &C, digits: usize) -> Option<Root> {
    let near: f64 = 10f64.powi(-((digits / 2) as i32));
    let far: f64 = 10f64.powi(-((digits / 4) as i32));
    let mut distances: Vec<(f64, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (to_f64(&value.sub(&c.numeric).abs()), i))
        .collect();
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if distances.is_empty() || distances[0].0 >= near || (distances.len() > 1 && distances[1].0 <= far) {
        return None;
    }
    Some(candidates[distances[0].1].clone())
}

fn identified(candidates: &[Root], numeric: &[C], digits: usize, prec: usize) -> PolyResult<Option<Vec<Root>>> {
    let Some(values) = numerical_roots(numeric, prec)? else {
        return Ok(None);
    };
    let mut chosen: Vec<Root> = Vec::new();
    for value in &values {
        let Some(candidate) = nearest(candidates, value, digits) else {
            return Ok(None);
        };
        if chosen.iter().any(|c| c.exact == candidate.exact) {
            return Ok(None);
        }
        chosen.push(candidate);
    }
    Ok(Some(chosen))
}

fn rational_coefficients(e: &Basic, v: &Basic) -> PolyResult<Vec<Rat>> {
    let mut coefficients: Vec<Rat> = Vec::new();
    for c in Poly::new(e, std::slice::from_ref(v))?.all_coeffs()? {
        let Some(value) = c.as_rational().filter(|_| c.is_rational_class()) else {
            return Err(not_ported(&format!(
                "a polynomial in {} alone is expected, got {}",
                v, e
            )));
        };
        coefficients.push(value.clone());
    }
    Ok(dup_sqf_part(&coefficients, &QQ))
}

fn eliminant(polys: &[Basic], variables: &[Basic], k: usize) -> PolyResult<Option<Vec<Rat>>> {
    let mut r: Basic = polys[k].clone();
    for j in (0..k).rev() {
        r = crate::polys::polytools::resultant_expr(&r, &polys[j], std::slice::from_ref(&variables[j]))?;
    }
    if r == Basic::zero() {
        return Ok(None);
    }
    Ok(Some(rational_coefficients(&r, &variables[k])?))
}

fn digits_of(polys: &[Basic], variables: &[Basic]) -> PolyResult<usize> {
    let mut total: usize = DIGITS;
    for p in polys {
        let mut coefficients: Vec<Rat> = Vec::new();
        for (_, term) in Poly::new(p, variables)?.terms()? {
            if let Some(v) = term.as_rational().filter(|_| term.is_rational_class()) {
                coefficients.push(v.clone());
            }
        }
        let mut denominator: Int = Int::ONE;
        for q in &coefficients {
            let d: Int = denom(q);
            let g: Int = gcd(&denominator, &d);
            denominator = &denominator * &d / g;
        }
        let largest: Int = coefficients
            .iter()
            .map(|q| abs(&(numer(q) * (&denominator / denom(q)))))
            .max()
            .unwrap_or(Int::ZERO);
        total += largest.to_string().len();
    }
    Ok(total)
}

/// A polynomial in `v` alone which vanishes on the chain, asked for when
/// a resultant vanishes.
pub type Fallback<'a> = &'a dyn Fn(&Basic) -> PolyResult<Basic>;

/// The points of the squarefree regular chain `polys` with the main
/// `variables` (by increasing main variable, no free variable), the real
/// ones only with `real` (`exact_solutions`).
pub fn exact_solutions(
    polys: &[Basic],
    variables: &[Basic],
    real_only: bool,
    fallback: Option<Fallback>,
) -> PolyResult<Vec<Point>> {
    let digits: usize = digits_of(polys, variables)?;
    let prec: usize = crate::numeric::dps_to_prec(digits);
    let mut points: Vec<PointData> = vec![PointData {
        exact: vec![],
        numeric: vec![],
        real: true,
    }];
    for (k, (p, v)) in polys.iter().zip(variables).enumerate() {
        let polynomial: Polynomial = Polynomial::new(p, v, &variables[..k])?;
        let mut candidates: Option<Vec<Root>> = None;
        let mut searched: bool = false;
        let mut following: Vec<PointData> = Vec::new();
        for point in &points {
            if real_only && !point.real {
                continue;
            }
            let exact: Vec<Basic> = if point.exact.is_empty() {
                polynomial.exprs.clone()
            } else {
                polynomial
                    .exprs
                    .iter()
                    .map(|c| crate::core::subs::subs_mapping(c, &point.exact))
                    .collect::<PolyResult<_>>()?
            };
            let numeric: Vec<C> = polynomial.numeric(&point.numeric, prec);
            if exact.len() == 2 {
                let mut value: Basic = crate::core::mul::mul(&[
                    Basic::negative_one(),
                    exact[1].clone(),
                    crate::core::power::pow(&exact[0], &Basic::negative_one())?,
                ])?;
                let approximate: C = numeric[1].div(&numeric[0]).neg();
                if !has_root_objects(&value) {
                    if let Some(simpler) = crate::timeout::attempt_within(Some(5.0), || {
                        crate::simplify::radsimp::radsimp(&value, true, 4)
                    })? {
                        value = simpler;
                    }
                } else {
                    if candidates.is_none() && !searched {
                        searched = true;
                        if let Some(e) = eliminant(polys, variables, k)? {
                            candidates = Some(roots_of(&e, v, prec)?);
                        }
                    }
                    if let Some(root) = candidates.as_ref().and_then(|c| nearest(c, &approximate, digits)) {
                        value = root.exact;
                    }
                }
                following.push(point.extended(v, value, approximate, true));
                continue;
            }
            let found: Option<Vec<Root>> = if exact.iter().all(|c| c.is_rational_class()) {
                let coefficients: Vec<Rat> = exact
                    .iter()
                    .map(|c| c.as_rational().expect("a rational").clone())
                    .collect();
                Some(roots_of(&coefficients, v, prec)?)
            } else {
                if candidates.is_none() {
                    searched = true;
                    let mut e: Option<Vec<Rat>> = eliminant(polys, variables, k)?;
                    if e.is_none() {
                        if let Some(fb) = fallback {
                            e = Some(rational_coefficients(&fb(v)?, v)?);
                        }
                    }
                    let Some(e) = e else {
                        return Err(not_ported(&format!("no polynomial in {} alone was found", v)));
                    };
                    candidates = Some(roots_of(&e, v, prec)?);
                }
                identified(candidates.as_ref().expect("candidates"), &numeric, digits, prec)?
            };
            let Some(found) = found else {
                let shown: Vec<String> = point.exact.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
                return Err(not_ported(&format!(
                    "the roots of {} at {{{}}} were not told apart with {} digits",
                    p,
                    shown.join(", "),
                    digits
                )));
            };
            for root in found {
                following.push(point.extended(v, root.exact, root.numeric, root.real));
            }
        }
        let _ = searched;
        points = following;
    }
    Ok(points
        .into_iter()
        .filter(|p| p.real || !real_only)
        .map(|p| p.exact)
        .collect())
}

/// mpmath's `polyroots` (with its cleanup) on complex coefficients at
/// `prec` bits; `None` without convergence or with a zero leading
/// coefficient.
pub(crate) fn polyroots_complex(
    coeffs: &[crate::core::evalf::Complex],
    maxsteps: usize,
    extraprec: usize,
    prec: usize,
) -> Option<Vec<crate::core::evalf::Complex>> {
    let cs: Vec<C> = coeffs.iter().map(|z| C::from_complex(z, prec)).collect();
    let found: Vec<C> = polyroots(&cs, maxsteps, extraprec, prec).ok()??;
    Some(
        found
            .into_iter()
            .map(|c| crate::core::evalf::complex(c.re, c.im))
            .collect(),
    )
}
