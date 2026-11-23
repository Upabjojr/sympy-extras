//! Port of `sympy_extras/integrals/definite.py`, the definite integrals
//! under assumptions on the parameters.
//!
//! This first part holds the numerical verification of the values found:
//! the quadrature of the integrand by mpmath's rules (two tanh-sinh
//! subdivisions and Gauss–Legendre, trusted when they agree, with
//! `quadosc` for an oscillatory half-line), the crossings of the branch
//! cuts of a complex segment, the check of a claimed divergence, and
//! `verify_numerically` at samples of the parameters, every branch of a
//! `Piecewise` value in its own region and the isolated values of the
//! parameters. The integrator itself is not ported yet.

#![allow(clippy::mutable_key_type)]
#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeSet, HashMap};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, numeric_value, parts, Complex, Real};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::integrals::conditions::Values;
use crate::integrals::quadrature::{quad_rule, Point, Rule, Value};
use crate::numbers::{Int, Rat};
use crate::random::PyRandom;

type R = PolyResult<Basic>;

/// The precision of the checks: mpmath's `workdps(20)`.
const PREC: usize = 70;

/// The branches of a `Piecewise` value checked at a sample of their own
/// region (`_BRANCHES_CHECKED`).
const BRANCHES_CHECKED: usize = 4;
/// The samples tried in the region of a branch (`_BRANCH_SAMPLES`).
const BRANCH_SAMPLES: usize = 3;

fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn neg_oo() -> Basic {
    Basic::atom(Head::NegativeInfinity)
}
fn infinite(e: &Basic) -> bool {
    *e == oo() || *e == neg_oo()
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn has_head(e: &Basic, f: &dyn Fn(&Head) -> bool) -> bool {
    e.preorder().iter().any(|n| f(n.head()))
}
fn has_function(e: &Basic, fs: &[Function]) -> bool {
    has_head(e, &|h| matches!(h, Head::Function(f) if fs.contains(f)))
}
fn has_nan_or_zoo(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::NaN | Head::ComplexInfinity))
}

/// `e` with the symbols replaced by values, the rebuilt nodes evaluated
/// (`_xr`).
fn xr(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

/// Symbols sorted by name (`_sorted_symbols`).
fn sorted_symbols(symbols: BTreeSet<Basic>) -> Vec<Basic> {
    let mut v: Vec<Basic> = symbols.into_iter().collect();
    v.sort_by_key(|s| s.to_string());
    v
}

fn parameters_of(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> Vec<Basic> {
    let mut s: BTreeSet<Basic> = f.free_symbols();
    s.extend(a.free_symbols());
    s.extend(b.free_symbols());
    s.remove(x);
    sorted_symbols(s)
}

/// Whether both bounds are real or infinite (`_real_bounds`).
pub fn real_bounds(a: &Basic, b: &Basic) -> bool {
    for bound in [a, b] {
        if infinite(bound) {
            continue;
        }
        match is(bound, Predicate::extended_real) {
            Some(false) => return false,
            None if bound.free_symbols().is_empty() => return false,
            _ => {}
        }
    }
    true
}

// -- numbers

fn rreal(n: i64) -> Real {
    crate::core::evalf::real_int(&Int::from(n), PREC)
}

fn at_prec(x: &Real) -> Real {
    x.clone().with_precision(PREC).value()
}

fn to_f64(x: &Real) -> f64 {
    x.to_f64().value()
}

fn c64(v: &Value) -> (f64, f64) {
    (to_f64(&v.0), to_f64(&v.1))
}

fn cabs64(z: (f64, f64)) -> f64 {
    z.0.hypot(z.1)
}

fn csub64(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}

fn finite64(z: (f64, f64)) -> bool {
    z.0.is_finite() && z.1.is_finite()
}

/// A real number at 20 digits, the infinities as points (`_mpf`).
fn mpf(e: &Basic) -> Option<Point> {
    if *e == oo() {
        return Some(Point::PosInf);
    }
    if *e == neg_oo() {
        return Some(Point::NegInf);
    }
    let value: Complex = numeric_value(e, PREC)?;
    Some(Point::Finite(at_prec(&parts(&value).0)))
}

fn point_value(p: &Point) -> Option<&Real> {
    match p {
        Point::Finite(r) => Some(r),
        _ => None,
    }
}

/// `f` at the real `u` as a pair of reals; `None` where it has no
/// finite value.
fn value_at(f: &Basic, x: &Basic, u: &Real, prec: usize) -> Option<Value> {
    let z: Complex = complex(u.clone(), Real::ZERO);
    let found: Complex = crate::integrals::evalf::value_at(f, x, z, prec)?;
    let (re, im) = parts(&found);
    if re.repr().is_infinite() || im.repr().is_infinite() {
        return None;
    }
    Some((re, im))
}

// -- the quadrature

/// `Integral(f, (x, a, b))` by mpmath's quadrature, trusted only when the
/// rules agree; a complex bound means the straight segment from `a` to
/// `b`; the numeric points `inside` subdivide a finite real range; `None`
/// for an infinite or untrusted value (`_quadrature`).
pub fn quadrature(f: &Basic, x: &Basic, a: &Basic, b: &Basic, inside: &[Basic]) -> PolyResult<Option<(f64, f64)>> {
    if !real_bounds(a, b) {
        if infinite(a) || infinite(b) || !a.free_symbols().is_empty() || !b.free_symbols().is_empty() {
            return Ok(None);
        }
        let t: Basic = crate::core::symbol::Dummy::new("t", &[(Predicate::real, true)]).0;
        let width: Basic = s2(b, a)?;
        let g: Basic = m2(&subs(f, x, &a2(a, &m2(&width, &t)?)?)?, &width)?;
        let crossings: Vec<Basic> = axis_crossings(&g, &t)?;
        return quadrature(&g, &t, &Basic::zero(), &Basic::one(), &crossings);
    }
    let (Some(lo), Some(hi)) = (mpf(a), mpf(b)) else {
        return Ok(None);
    };
    let points: Vec<Point> = match (&lo, &hi) {
        (Point::NegInf, Point::PosInf) => {
            let mut v: Vec<Point> = vec![lo.clone()];
            v.extend([-10, -1, 0, 1, 10].iter().map(|k| Point::Finite(rreal(*k))));
            v.push(hi.clone());
            v
        }
        (Point::Finite(l), Point::PosInf) => {
            let mut v: Vec<Point> = vec![lo.clone()];
            v.extend([1, 10, 100].iter().map(|k| Point::Finite(at_prec(&(l + rreal(*k))))));
            v.push(hi.clone());
            v
        }
        (Point::NegInf, Point::Finite(h)) => {
            let mut v: Vec<Point> = vec![lo.clone()];
            v.extend([100, 10, 1].iter().map(|k| Point::Finite(at_prec(&(h - rreal(*k))))));
            v.push(hi.clone());
            v
        }
        (Point::Finite(l), Point::Finite(h)) => {
            let (low, high) = if l < h { (l, h) } else { (h, l) };
            let mut nodes: Vec<Real> = Vec::new();
            for q in inside {
                if is(q, Predicate::real) == Some(true) && q.free_symbols().is_empty() {
                    if let Some(Point::Finite(node)) = mpf(q) {
                        if *low < node && node < *high && !nodes.contains(&node) {
                            nodes.push(node);
                        }
                    }
                }
            }
            nodes.sort_by(|u, v| u.partial_cmp(v).expect("finite"));
            if h < l {
                nodes.reverse();
            }
            let mut v: Vec<Point> = vec![lo.clone()];
            v.extend(nodes.into_iter().map(Point::Finite));
            v.push(hi.clone());
            v
        }
        _ => vec![lo.clone(), hi.clone()],
    };
    let mut finer: Vec<Point> = Vec::new();
    for w in points.windows(2) {
        finer.push(w[0].clone());
        if let (Point::Finite(p), Point::Finite(q)) = (&w[0], &w[1]) {
            finer.push(Point::Finite(at_prec(&((p + q) / rreal(2)))));
        }
    }
    finer.push(points.last().expect("two points").clone());
    let mut singular: Vec<Real> = Vec::new();
    let mut finite = |u: &Real| -> Option<Value> {
        crate::timeout::check().ok()?;
        match value_at(f, x, u, PREC) {
            Some(v) => Some(v),
            None => {
                singular.push(u.clone());
                Some((Real::ZERO, Real::ZERO))
            }
        }
    };
    let Some((first, e1)) = quad_rule(&mut finite, &points, PREC, Rule::TanhSinh) else {
        crate::timeout::check()?;
        return Ok(None);
    };
    let Some((second, _)) = quad_rule(&mut finite, &finer, PREC, Rule::TanhSinh) else {
        return Ok(None);
    };
    let Some((third, _)) = quad_rule(&mut finite, &points, PREC, Rule::GaussLegendre) else {
        return Ok(None);
    };
    crate::timeout::check()?;
    let ends: Vec<f64> = finer.iter().filter_map(point_value).map(to_f64).collect();
    let bounded: bool = lo.is_finite() && hi.is_finite();
    for u in &singular {
        let u: f64 = to_f64(u);
        if !ends.iter().any(|p| (u - p).abs() < 1e-10 * (1.0 + p.abs())) && !(u.abs() > 1e10 && !bounded) {
            return Ok(None);
        }
    }
    let (v1, e1) = (c64(&first), to_f64(&e1).abs());
    let (v2, v3) = (c64(&second), c64(&third));
    if !(finite64(v1) && finite64(v2)) {
        return Ok(None);
    }
    let scale: f64 = 1.0 + cabs64(v1).max(cabs64(v2));
    if cabs64(csub64(v1, v2)) > 1e-8 * scale || e1 > 1e-6 * scale {
        return oscillatory_quadrature(f, x, &lo, &hi);
    }
    if cabs64(csub64(v1, v3)) > 1e-6 * scale && cabs64(csub64(v2, v3)) > 1e-6 * scale && e1 > 1e-12 * scale {
        return oscillatory_quadrature(f, x, &lo, &hi);
    }
    Ok(Some(v1))
}

/// The real roots in `(0, 1)` of a polynomial over `QQ` of degree one to
/// four (`_unit_roots`).
fn unit_roots(p: &crate::polys::polytools::Poly) -> PolyResult<Vec<Basic>> {
    let degree: i64 = p.degree(0)?;
    let candidates: Vec<Basic> = if degree == 1 {
        crate::polys::polyroots::roots_linear(p)?
    } else if degree == 2 {
        crate::polys::polyroots::roots_quadratic(p)?
    } else {
        let mut coefficients: Vec<Rat> = Vec::new();
        for c in p.all_coeffs()? {
            let Some(r) = c.as_rational() else {
                return Ok(Vec::new());
            };
            coefficients.push(r.clone());
        }
        crate::polys::cad::samplepoints::real_roots_of(&coefficients)?
            .iter()
            .map(|r| crate::polys::cad::indexedroot::real_root(r, "t"))
            .collect::<PolyResult<_>>()?
    };
    let mut found: Vec<Basic> = Vec::new();
    for root in candidates {
        if is(&root, Predicate::extended_real) != Some(true) {
            continue;
        }
        if let Some(Point::Finite(v)) = mpf(&root) {
            if v > Real::ZERO && v < rreal(1) && !found.contains(&root) {
                found.push(root);
            }
        }
    }
    Ok(found)
}

/// The points of `(0, 1)` where the argument of a function in `g`, or the
/// base of a fractional power, polynomial in the real `t`, crosses the real
/// or the imaginary axis (`_axis_crossings`).
pub fn axis_crossings(g: &Basic, t: &Basic) -> PolyResult<Vec<Basic>> {
    let mut found: Vec<Basic> = Vec::new();
    let nodes: Vec<Basic> = g.atoms(Some(&|n: &Basic| {
        n.is_pow() || matches!(n.head(), Head::Function(_) | Head::Undefined(_))
    }));
    for node in nodes {
        let arguments: Vec<Basic> = if node.is_pow() {
            if is(&node.args()[1], Predicate::integer) == Some(true) {
                continue;
            }
            vec![node.args()[0].clone()]
        } else {
            node.args().to_vec()
        };
        for u in arguments {
            if !u.free_symbols().contains(t)
                || crate::core::expr::polynomial_like(
                    &u,
                    std::slice::from_ref(t),
                    crate::core::expr::PolynomialKind::Polynomial,
                ) != Some(true)
            {
                continue;
            }
            let expanded: Basic = crate::core::function::expand_default(&u)?;
            let (re, im) = match crate::functions::elementary::complexes::as_real_imag(&expanded, true, None)? {
                Some(p) => p,
                None => (
                    crate::functions::elementary::complexes::re(&expanded)?,
                    crate::functions::elementary::complexes::im(&expanded)?,
                ),
            };
            for part in [re, im] {
                let Ok(p) = crate::polys::polytools::Poly::new(&part, std::slice::from_ref(t)) else {
                    continue;
                };
                if p.is_zero() {
                    continue;
                }
                let degree: i64 = p.degree(0)?;
                if !(1..=4).contains(&degree) || !matches!(p.domain.name().as_str(), "QQ" | "ZZ") {
                    continue;
                }
                for root in unit_roots(&p)? {
                    if !found.contains(&root) {
                        found.push(root);
                    }
                }
            }
        }
    }
    Ok(found)
}

/// `Integral(f, (x, lo, hi))` over a half-line by `quadosc` when `f`
/// oscillates, trusted when two choices of the period agree
/// (`_oscillatory_quadrature`).
fn oscillatory_quadrature(f: &Basic, x: &Basic, lo: &Point, hi: &Point) -> PolyResult<Option<(f64, f64)>> {
    let minus_x = || crate::core::mul::neg(x);
    match (lo, hi) {
        (Point::NegInf, Point::PosInf) => {
            let zero: Point = Point::Finite(rreal(0));
            let left = oscillatory_quadrature(&subs(f, x, &minus_x()?)?, x, &zero, hi)?;
            let right = oscillatory_quadrature(f, x, &zero, hi)?;
            return Ok(match (left, right) {
                (Some(l), Some(r)) => Some((l.0 + r.0, l.1 + r.1)),
                _ => None,
            });
        }
        (Point::NegInf, Point::Finite(h)) => {
            return oscillatory_quadrature(&subs(f, x, &minus_x()?)?, x, &Point::Finite(-h.clone()), &Point::PosInf);
        }
        (_, Point::PosInf) => {}
        _ => return Ok(None),
    }
    let Point::Finite(lo) = lo else {
        return Ok(None);
    };
    let mut frequencies: Vec<Rat> = Vec::new();
    for node in f.atoms(Some(&|n: &Basic| {
        n.is_function(Function::Sin) || n.is_function(Function::Cos)
    })) {
        let Some((c, e)) = crate::integrals::mellin::monomial(&node.args()[0], x)? else {
            return Ok(None);
        };
        let Some(k) = c.as_rational().filter(|_| e == Basic::one()) else {
            return Ok(None);
        };
        frequencies.push(k.clone());
    }
    if frequencies.is_empty() {
        return Ok(None);
    }
    let (mut numerator, mut denominator) = (Int::ZERO, Int::ONE);
    for k in &frequencies {
        let p: Int = crate::numbers::numer(k);
        let p: Int = if p < Int::ZERO { -p } else { p };
        numerator = gcd(&numerator, &p);
        let q: Int = crate::numbers::denom(k);
        denominator = &denominator / gcd(&denominator, &q) * &q;
    }
    let pi: Real = Real::pi(PREC + 20);
    let period: Real = at_prec(
        &(rreal(2) * pi * crate::core::evalf::real_int(&denominator, PREC)
            / crate::core::evalf::real_int(&numerator, PREC)),
    );
    let mut g = |u: &Real, p: usize| -> Option<Value> {
        crate::timeout::check().ok()?;
        value_at(f, x, u, p)
    };
    let Some(first) = crate::integrals::extrapolation::quadosc(&mut g, lo, &period, PREC) else {
        crate::timeout::check()?;
        return Ok(None);
    };
    let double: Real = at_prec(&(rreal(2) * &period));
    let Some(second) = crate::integrals::extrapolation::quadosc(&mut g, lo, &double, PREC) else {
        crate::timeout::check()?;
        return Ok(None);
    };
    let (first, second) = (c64(&first), c64(&second));
    if cabs64(csub64(first, second)).partial_cmp(&(1e-8 * (1.0 + cabs64(first)))) != Some(std::cmp::Ordering::Less)
        && cabs64(csub64(first, second)) != 1e-8 * (1.0 + cabs64(first))
    {
        return Ok(None);
    }
    Ok(Some(first))
}

fn gcd(a: &Int, b: &Int) -> Int {
    let (mut a, mut b) = (a.clone(), b.clone());
    while b != Int::ZERO {
        let r: Int = &a % &b;
        a = b;
        b = r;
    }
    if a < Int::ZERO {
        -a
    } else {
        a
    }
}

/// Whether the quadrature gives a number of a plausible size at some
/// sample of the parameters (`_quadrature_converges`).
pub fn quadrature_converges(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if has_function(f, &[Function::DiracDelta]) {
        return Ok(false);
    }
    let parameters: Vec<Basic> = parameters_of(f, x, a, b);
    let mut rng: PyRandom = PyRandom::from_string(&format!("({}, {}, {})", f, a, b));
    for _ in 0..(if parameters.is_empty() { 1 } else { 2 }) {
        let Some(values) = sample(&parameters, assumptions, &mut rng)? else {
            return Ok(false);
        };
        let (g, lower, upper) = (xr(f, &values)?, xr(a, &values)?, xr(b, &values)?);
        if grows_like_a_pole(&g, x, &lower, true)? || grows_like_a_pole(&g, x, &upper, false)? {
            continue;
        }
        if let Some(found) = quadrature(&g, x, &lower, &upper, &[])? {
            if cabs64(found) < 1e8 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether `|g(point + h)| h` does not shrink with `h` on the given side
/// of a finite `point` (`_grows_like_a_pole`).
fn grows_like_a_pole(g: &Basic, x: &Basic, point: &Basic, right: bool) -> PolyResult<bool> {
    if infinite(point) || !crate::core::assumptions::is_number(point) {
        return Ok(false);
    }
    let mut sizes: Vec<f64> = Vec::new();
    for k in [3u32, 5, 7] {
        let h: Basic = Basic::from_rational(Rat::from_parts(
            Int::from(if right { 1 } else { -1 }),
            dashu_int::UBig::from(10u8).pow(k as usize),
        ));
        let Some(value) = numeric_value(&subs(g, x, &a2(point, &h)?)?, PREC) else {
            return Ok(false);
        };
        let (re, im) = parts(&value);
        if im != Real::ZERO || !to_f64(&re).is_finite() {
            return Ok(false);
        }
        sizes.push(to_f64(&re).abs() * 10f64.powi(-(k as i32)));
    }
    Ok(sizes[0] > 0.0 && sizes[1] >= 0.9 * sizes[0] && sizes[2] >= 0.9 * sizes[1])
}

// -- verification

/// The sample values of the parameters with the roots of linear and
/// quadratic polynomials in radicals (`_sample_values`).
fn sample(symbols: &[Basic], assumptions: &[Basic], rng: &mut PyRandom) -> PolyResult<Option<Values>> {
    let Some(values) = crate::integrals::conditions::sample_values(symbols, assumptions, rng)? else {
        return Ok(None);
    };
    let mut out: Values = Vec::new();
    for (s, v) in values {
        out.push((s, radical_sample(&v)?));
    }
    Ok(Some(out))
}

fn radical_sample(v: &Basic) -> R {
    if !matches!(v.head(), Head::IndexedRoot) {
        return Ok(v.clone());
    }
    let (expr, variable, _) = crate::polys::cad::indexedroot::parts(v);
    let Some(coefficients) = coefficients(expr, variable)? else {
        return Ok(v.clone());
    };
    if coefficients.len() > 3 {
        return Ok(v.clone());
    }
    match crate::polys::roots::radical_form(v)? {
        Some(found) if !found.has(&Basic::atom(Head::ImaginaryUnit)) => Ok(found),
        _ => Ok(v.clone()),
    }
}

/// The coefficients of `q` as a polynomial in `x` by the power, the
/// leading one nonzero (`_coefficients`).
pub fn coefficients(q: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    if crate::core::expr::polynomial_like(
        q,
        std::slice::from_ref(x),
        crate::core::expr::PolynomialKind::Polynomial,
    ) != Some(true)
    {
        return Ok(None);
    }
    let expanded: Basic = crate::core::function::expand_default(q)?;
    let terms_list: Vec<Basic> = if expanded.is_add() {
        expanded.args().to_vec()
    } else {
        vec![expanded.clone()]
    };
    let mut terms: std::collections::BTreeMap<usize, Basic> = std::collections::BTreeMap::new();
    for term in terms_list {
        let (c, rest) = crate::core::expr::as_independent(&term, x, false)?;
        let k: usize = if rest == Basic::one() {
            0
        } else if rest == *x {
            1
        } else if rest.is_pow()
            && rest.args()[0] == *x
            && rest.args()[1]
                .as_rational()
                .is_some_and(|r| r.is_int() && *r > Rat::ZERO)
        {
            usize::try_from(&crate::numbers::numer(rest.args()[1].as_rational().expect("checked"))).unwrap_or(0)
        } else {
            return Ok(None);
        };
        let old: Basic = terms.get(&k).cloned().unwrap_or_else(Basic::zero);
        terms.insert(k, a2(&old, &c)?);
    }
    let top: usize = terms
        .iter()
        .filter(|(_, c)| **c != Basic::zero())
        .map(|(k, _)| *k)
        .max()
        .unwrap_or(0);
    Ok(Some(
        (0..=top)
            .map(|k| terms.get(&k).cloned().unwrap_or_else(Basic::zero))
            .collect(),
    ))
}

/// Whether `value` at the sample agrees with the quadrature; `None` when
/// its number cannot be trusted or computed (`_agrees`).
fn agrees(value: &Basic, expected: (f64, f64), values: &Values) -> Option<bool> {
    let number: Complex = crate::numeric::reliable_value(value, 20, values)?;
    let (re, im) = parts(&number);
    let ours: (f64, f64) = (to_f64(&re), to_f64(&im));
    if !finite64(ours) {
        return None;
    }
    Some(cabs64(csub64(ours, expected)) <= 1e-6 * (1.0 + cabs64(expected)))
}

/// The `(value, condition)` pairs of a `Piecewise` (`_pairs`).
pub fn pairs(node: &Basic) -> Vec<(Basic, Basic)> {
    node.args()
        .iter()
        .filter(|p| matches!(p.head(), Head::ExprCondPair))
        .map(|p| (p.args()[0].clone(), p.args()[1].clone()))
        .collect()
}

/// The assumptions together with `extra` (`_with`).
pub fn with(assumptions: &[Basic], extra: &[Basic]) -> Vec<Basic> {
    let mut items: Vec<Basic> = extra.to_vec();
    items.extend(assumptions.iter().cloned());
    items
}

/// The regions of the parameters in which each branch of a `Piecewise`
/// value is taken (`_branch_regions`).
pub fn branch_regions(value: &Basic, parameters: &[Basic]) -> PolyResult<Vec<Basic>> {
    if !has_head(value, &|h| matches!(h, Head::Piecewise)) {
        return Ok(Vec::new());
    }
    let folded: Basic = crate::functions::elementary::piecewise::piecewise_fold(value, None)?;
    if !matches!(folded.head(), Head::Piecewise) {
        return Ok(Vec::new());
    }
    let mut regions: Vec<Basic> = Vec::new();
    let mut earlier: Vec<Basic> = Vec::new();
    for (_, condition) in pairs(&folded).into_iter().take(BRANCHES_CHECKED) {
        let mut parts: Vec<Basic> = vec![condition.clone()];
        parts.extend(earlier.iter().map(crate::core::logic::not_));
        let region: Basic = crate::core::logic::and_(&parts);
        if region.free_symbols().iter().any(|s| parameters.contains(s)) && !isolated(&condition)? {
            regions.push(region);
        }
        earlier.push(condition);
    }
    Ok(regions)
}

/// Whether a condition holds at isolated points only (`_isolated`).
fn isolated(condition: &Basic) -> PolyResult<bool> {
    let disjuncts: Vec<Basic> = if matches!(condition.head(), Head::Or) {
        condition.args().to_vec()
    } else {
        vec![condition.clone()]
    };
    for d in disjuncts {
        match crate::special_values::condition_point(&d)? {
            Some(p) if !p.is_empty() => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// Whether `value` agrees with a numerical quadrature of `Integral(f, (x,
/// a, b))` at `samples` random values of the parameters satisfying the
/// assumptions: `Some(false)` when they disagree where the quadrature is
/// trusted, `Some(true)` when they agree, `None` when nothing could be
/// checked (`verify_numerically`).
pub fn verify_numerically(
    value: &Basic,
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    samples: usize,
    inside: &[Basic],
) -> PolyResult<Option<bool>> {
    if has_function(f, &[Function::DiracDelta]) {
        return Ok(None);
    }
    let parameters: Vec<Basic> = parameters_of(f, x, a, b);
    let mut rng: PyRandom = PyRandom::from_string(&format!("({}, {}, {})", f, a, b));
    let at = |values: &Values| -> PolyResult<Option<(f64, f64)>> {
        let points: Vec<Basic> = inside.iter().map(|p| xr(p, values)).collect::<PolyResult<_>>()?;
        quadrature(&xr(f, values)?, x, &xr(a, values)?, &xr(b, values)?, &points)
    };
    let mut verdict: Option<bool> = None;
    for _ in 0..(if parameters.is_empty() { 1 } else { samples }) {
        let Some(values) = sample(&parameters, assumptions, &mut rng)? else {
            return Ok(None);
        };
        let Some(expected) = at(&values)? else {
            continue;
        };
        match agrees(value, expected, &values) {
            None => return Ok(None),
            Some(false) => return Ok(Some(false)),
            Some(true) => verdict = Some(true),
        }
    }
    for region in branch_regions(value, &parameters)? {
        for _ in 0..BRANCH_SAMPLES {
            let Some(values) = sample(&parameters, &with(assumptions, std::slice::from_ref(&region)), &mut rng)? else {
                break;
            };
            let found: Option<bool> = match at(&values)? {
                None => None,
                Some(expected) => agrees(value, expected, &values),
            };
            if found == Some(false) {
                return Ok(Some(false));
            }
            if found == Some(true) {
                break;
            }
        }
    }
    let points = crate::special_values::equality_points(value, &parameters)?;
    for point in points.into_iter().take(BRANCHES_CHECKED) {
        let Some(at_point) = crate::special_values::point_assumptions(&point, assumptions)? else {
            continue;
        };
        let g: Basic = xr(f, &point)?;
        if has_nan_or_zoo(&g) {
            continue;
        }
        let Ok(special) = xr(value, &point) else {
            continue;
        };
        let points: Vec<Basic> = inside.iter().map(|p| xr(p, &point)).collect::<PolyResult<_>>()?;
        let there: Option<bool> =
            verify_numerically(&special, &g, x, &xr(a, &point)?, &xr(b, &point)?, &at_point, 1, &points)?;
        if there == Some(false) {
            return Ok(Some(false));
        }
    }
    Ok(verdict)
}

// -- the helpers of the integrator

thread_local! {
    static COUNTER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A dummy with a name of its own (`_fresh`).
pub fn fresh(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    let n: usize = COUNTER.with(|c| {
        let n = c.get();
        c.set(n + 1);
        n
    });
    crate::core::symbol::Dummy::new(&format!("{}{}", name, n), assumptions).0
}

fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn p2(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn ng(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rat(p: i64, q: i64) -> Basic {
    Basic::from_rational(crate::numbers::rat(p, q))
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn apply(f: Function, args: &[Basic]) -> R {
    crate::functions::apply_function(f, args)
}
fn unevaluated(f: Function, arg: &Basic) -> Basic {
    Basic::new(Head::Function(f), vec![arg.clone()])
}
fn add_args(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn is_true_b(e: &Basic) -> bool {
    matches!(e.head(), Head::BooleanTrue)
}
fn is_false_b(e: &Basic) -> bool {
    matches!(e.head(), Head::BooleanFalse)
}
fn true_b() -> Basic {
    crate::core::logic::true_()
}
fn rel_op(e: &Basic) -> Option<crate::core::basic::RelOp> {
    match e.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}
fn is_number(e: &Basic) -> bool {
    crate::core::assumptions::is_number(e)
}
fn integer_of(e: &Basic) -> Option<i64> {
    let r: &Rat = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    i64::try_from(&crate::numbers::numer(r)).ok()
}
fn rational_of(e: &Basic) -> Option<(i64, i64)> {
    let r: &Rat = e.as_rational()?;
    Some((
        i64::try_from(&crate::numbers::numer(r)).ok()?,
        i64::try_from(&crate::numbers::denom(r)).ok()?,
    ))
}
fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn sorted_by_str(mut v: Vec<Basic>) -> Vec<Basic> {
    v.sort_by_cached_key(|e| e.to_string());
    v
}
fn replace_where(e: &Basic, query: &dyn Fn(&Basic) -> bool, value: &dyn Fn(&Basic) -> R) -> R {
    crate::core::function::replace_where(e, query, value)
}
fn to_float(e: &Basic) -> Option<f64> {
    let v: Complex = numeric_value(e, 53)?;
    let (re, im) = parts(&v);
    if im != Real::ZERO {
        return None;
    }
    Some(to_f64(&re))
}
fn has_any_function(e: &Basic, fs: &[Function]) -> bool {
    has_function(e, fs)
}
const TRIGONOMETRIC: [Function; 6] = [
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Cot,
    Function::Sec,
    Function::Csc,
];
const HYPERBOLIC: [Function; 6] = [
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];
const INVERSE_HYPERBOLIC: [Function; 6] = [
    Function::Asinh,
    Function::Acosh,
    Function::Atanh,
    Function::Acoth,
    Function::Asech,
    Function::Acsch,
];

/// Whether `f` has a trigonometric function (`_has_trigonometric`).
pub fn has_trigonometric(f: &Basic) -> bool {
    has_any_function(f, &TRIGONOMETRIC)
}

/// `F` at an endpoint: its value when finite there, the limit from
/// inside the range otherwise (`_endpoint`).
pub fn endpoint(big_f: &Basic, x: &Basic, point: &Basic, direction: &str) -> PolyResult<Option<Basic>> {
    let bad = |v: &Basic| {
        has_head(v, &|h| {
            matches!(
                h,
                Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
            )
        })
    };
    if !infinite(point) {
        let value: Basic = subs(big_f, x, point)?;
        if !bad(&value) {
            return Ok(Some(value));
        }
    }
    let dir: &str = if infinite(point) || is(point, Predicate::extended_real) != Some(true) {
        "+"
    } else {
        direction
    };
    let value: Basic = crate::core::limits::limit(big_f, x, point, dir)?;
    if matches!(value.head(), Head::Limit) || has_nan_or_zoo(&value) {
        return Ok(None);
    }
    Ok(Some(value))
}

const NOWHERE_POINTS: usize = 8;
const OUTSIDE_SAMPLES: usize = 2;

fn random_values(symbols: &[Basic], rng: &mut PyRandom) -> Values {
    symbols
        .iter()
        .map(|s| {
            let sign: i64 = if rng.randbelow(2) == 0 { -1 } else { 1 };
            let p: i64 = sign * rng.randint(1, 19);
            let q: i64 = rng.randint(1, 4);
            (s.clone(), rat(p, q))
        })
        .collect()
}

fn facts_hold(facts: &[Basic], values: &Values, plain: &[(Basic, Basic)]) -> PolyResult<bool> {
    let at: Values = values
        .iter()
        .map(|(s, v)| {
            let key: Basic = plain
                .iter()
                .find(|(k, _)| k == s)
                .map(|(_, p)| p.clone())
                .unwrap_or_else(|| s.clone());
            (key, v.clone())
        })
        .collect();
    for fact in facts {
        if !is_true_b(&xr(fact, &at)?) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `condition` is false at random real points satisfying the
/// assumptions, at `_NOWHERE_POINTS` of them at least (`_holds_nowhere`).
pub fn holds_nowhere(condition: &Basic, symbols: &[Basic], assumptions: &[Basic]) -> PolyResult<bool> {
    if is_true_b(condition) || symbols.is_empty() {
        return Ok(false);
    }
    let plain = crate::integrals::conditions::plain_symbols(symbols);
    let facts: Vec<Basic> = crate::integrals::conditions::with_symbol_facts(assumptions, symbols, &plain)?;
    let mut rng: PyRandom = PyRandom::from_string(&condition.to_string());
    let mut tested: usize = 0;
    for _ in 0..4 * NOWHERE_POINTS {
        let values: Values = random_values(symbols, &mut rng);
        if !facts_hold(&facts, &values, &plain)? {
            continue;
        }
        if !is_false_b(&xr(condition, &values)?) {
            return Ok(false);
        }
        tested += 1;
    }
    Ok(tested >= NOWHERE_POINTS)
}

/// Whether `value` agrees with the quadrature at random samples where
/// `condition` is false (`_holds_outside`).
pub fn holds_outside(
    value: &Basic,
    condition: &Basic,
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<bool> {
    let parameters: Vec<Basic> = parameters_of(f, x, a, b);
    if parameters.is_empty() {
        return Ok(true);
    }
    let plain = crate::integrals::conditions::plain_symbols(&parameters);
    let facts: Vec<Basic> = crate::integrals::conditions::with_symbol_facts(assumptions, &parameters, &plain)?;
    let mut rng: PyRandom = PyRandom::from_string(&format!("({}, {})", f, condition));
    let mut checked: usize = 0;
    for _ in 0..4 * NOWHERE_POINTS {
        let values: Values = random_values(&parameters, &mut rng);
        let outside: bool = match (facts_hold(&facts, &values, &plain), xr(condition, &values)) {
            (Ok(true), Ok(c)) => is_false_b(&c),
            (Err(e), _) | (_, Err(e)) if matches!(e, crate::error::PolyError::Type(_)) => false,
            (Err(e), _) | (_, Err(e)) => return Err(e),
            _ => false,
        };
        if !outside {
            continue;
        }
        let expected = quadrature(&xr(f, &values)?, x, &xr(a, &values)?, &xr(b, &values)?, &[])?;
        match expected {
            None => return Ok(false),
            Some(e) if agrees(value, e, &values) == Some(false) => return Ok(false),
            _ => {}
        }
        checked += 1;
        if checked >= OUTSIDE_SAMPLES {
            break;
        }
    }
    Ok(true)
}

/// The points without repetitions, in their order (`_distinct`).
pub fn distinct(points: &[Basic]) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for p in points {
        if !found.contains(p) {
            found.push(p.clone());
        }
    }
    found
}

/// The candidate singularities of `f` on `(a, b)` (`_singular_points`).
pub fn singular_points(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let mut bounds: Vec<Basic> = Vec::new();
    if *a != neg_oo() {
        bounds.push(crate::core::relational::gt(x, a)?);
    }
    if *b != oo() {
        bounds.push(crate::core::relational::lt(x, b)?);
    }
    let zeros = |u: &Basic| trigonometric_zeros(u, x, a, b);
    let mut found = crate::integrals::antiderivative::singularities(f, x, &bounds, Some(&zeros))?;
    if found.is_none() {
        let Some(all) = crate::integrals::antiderivative::singularities(f, x, &[], None)? else {
            return Ok(None);
        };
        let mut kept: Vec<Basic> = Vec::new();
        for p in all {
            if !complex_for_real_parameters(&p)? {
                kept.push(p);
            }
        }
        found = Some(kept);
    }
    let points: Vec<Basic> = found
        .expect("found")
        .iter()
        .map(real_odd_roots)
        .collect::<PolyResult<_>>()?;
    Ok(Some(distinct(&points)))
}

fn real_symbols_rule(p: &Basic) -> Values {
    sorted_symbols(p.free_symbols())
        .into_iter()
        .map(|s| {
            let name: String = s.to_string();
            (s, fresh(&name, &[(Predicate::real, true)]))
        })
        .collect()
}

/// Whether `p` has the imaginary unit and is not known real for real
/// symbols (`_complex_for_real_parameters`).
pub fn complex_for_real_parameters(p: &Basic) -> PolyResult<bool> {
    if !p.has(&Basic::atom(Head::ImaginaryUnit)) {
        return Ok(false);
    }
    Ok(is(&xr(p, &real_symbols_rule(p))?, Predicate::extended_real) != Some(true))
}

/// Whether `p` is real once its symbols are taken real
/// (`_real_for_real_parameters`).
pub fn real_for_real_parameters(p: &Basic) -> PolyResult<bool> {
    Ok(is(&xr(p, &real_symbols_rule(p))?, Predicate::extended_real) == Some(true))
}

/// The points of a set of zeros on the range (`_candidate_points`).
pub fn candidate_points(found: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let mut found: Basic = found.clone();
    if matches!(found.head(), Head::Intersection) && found.args().len() == 2 {
        let finite: Vec<&Basic> = found
            .args()
            .iter()
            .filter(|p| matches!(p.head(), Head::FiniteSet))
            .collect();
        let intervals: usize = found
            .args()
            .iter()
            .filter(|p| matches!(p.head(), Head::Interval))
            .count();
        if finite.len() == 1 && intervals == 1 {
            found = finite[0].clone();
        }
    }
    if matches!(found.head(), Head::ConditionSet) && matches!(found.args()[2].head(), Head::FiniteSet) {
        let mut free: BTreeSet<Basic> = found.args()[1].free_symbols();
        free.remove(x);
        for s in a.free_symbols().union(&b.free_symbols()) {
            free.remove(s);
        }
        if free.is_empty() {
            found = found.args()[2].clone();
        }
    }
    if matches!(found.head(), Head::Union) {
        let mut points: Vec<Basic> = Vec::new();
        for part in found.args() {
            let Some(listed) = candidate_points(part, x, a, b)? else {
                return Ok(None);
            };
            for p in listed {
                if !points.contains(&p) {
                    points.push(p);
                }
            }
        }
        return Ok(Some(points));
    }
    if matches!(found.head(), Head::FiniteSet) {
        return Ok(Some(
            found.args().iter().map(real_odd_roots).collect::<PolyResult<_>>()?,
        ));
    }
    if matches!(found.head(), Head::EmptySet) {
        return Ok(Some(Vec::new()));
    }
    Ok(None)
}

/// The point with every real odd root `Abs(w)**(1/n)*sign(w)` of a
/// perfect power `w = v**n` written `v` (`_real_odd_roots`).
pub fn real_odd_roots(p: &Basic) -> R {
    fn root(node: &Basic) -> PolyResult<Option<Basic>> {
        if !node.is_mul() {
            return Ok(None);
        }
        let factors: Vec<Basic> = node.args().to_vec();
        let signs: Vec<&Basic> = factors.iter().filter(|f| f.is_function(Function::Sign)).collect();
        for power in &factors {
            if !(power.is_pow() && power.args()[0].is_function(Function::Abs)) {
                continue;
            }
            let Some((pn, q)) = rational_of(&power.args()[1]) else {
                continue;
            };
            if !(pn == 1 && q % 2 == 1) {
                continue;
            }
            let w: Basic = power.args()[0].args()[0].clone();
            let sign_w: Basic = apply(Function::Sign, std::slice::from_ref(&w))?;
            if !signs.contains(&&sign_w) {
                continue;
            }
            for s in [Basic::one(), int(-1)] {
                let v: Basic = crate::simplify::powsimp::powdenest(&p2(&m2(&s, &w)?, &rat(1, q))?, true, false)?;
                let pows: Vec<Basic> = v.atoms(Some(&|n: &Basic| n.is_pow()));
                if pows.is_empty() || pows.iter().all(|e| is(&e.args()[1], Predicate::integer) == Some(true)) {
                    let mut rest: Vec<Basic> = vec![m2(&s, &v)?];
                    rest.extend(factors.iter().filter(|f| *f != power && **f != sign_w).cloned());
                    return Ok(Some(mul_all(&rest)?));
                }
            }
        }
        Ok(None)
    }
    replace_where(p, &|node: &Basic| matches!(root(node), Ok(Some(_))), &|node: &Basic| {
        Ok(root(node)?.unwrap_or_else(|| node.clone()))
    })
}

/// The positions of the point `s` against the endpoints of `(a, b)`
/// (`_positions`).
pub fn positions(s: &Basic, a: &Basic, b: &Basic, singular: bool) -> PolyResult<Vec<(Basic, i32)>> {
    use crate::core::relational as rl;
    let below_a: Basic = if singular { rl::lt(s, a)? } else { rl::le(s, a)? };
    let below_b: Basic = if singular { rl::lt(s, b)? } else { rl::le(s, b)? };
    let above_a: Basic = if singular { rl::gt(s, a)? } else { rl::ge(s, a)? };
    let above_b: Basic = if singular { rl::gt(s, b)? } else { rl::ge(s, b)? };
    let and = |v: Vec<Basic>| crate::core::logic::and_(&v);
    let mut found: Vec<(Basic, i32)> = Vec::new();
    if *a != neg_oo() {
        found.push((and(vec![below_a, if *b == oo() { true_b() } else { below_b }]), 0));
    }
    if *b != oo() {
        found.push((and(vec![if *a == neg_oo() { true_b() } else { above_a }, above_b]), 0));
    }
    found.push((
        and(vec![
            if *a == neg_oo() { true_b() } else { rl::gt(s, a)? },
            if *b == oo() { true_b() } else { rl::lt(s, b)? },
        ]),
        1,
    ));
    if *a != neg_oo() && *b != oo() {
        found.push((and(vec![rl::lt(s, a)?, rl::gt(s, b)?]), -1));
    }
    found
        .into_iter()
        .map(|(c, p)| Ok((canonical_relations(&c)?, p)))
        .collect()
}

/// The condition with a closed linear bound made strict where `value` is
/// not defined on it (`_strict_where_undefined`).
pub fn strict_where_undefined(condition: &Basic, value: &Basic) -> R {
    use crate::core::basic::RelOp;
    let mut replacements: HashMap<Basic, Basic> = HashMap::new();
    for atom in condition.atoms(Some(&|n: &Basic| matches!(rel_op(n), Some(RelOp::Le | RelOp::Ge)))) {
        let (lhs, rhs) = (&atom.args()[0], &atom.args()[1]);
        let difference: Basic = s2(lhs, rhs)?;
        let symbols: Vec<Basic> = sorted_symbols(difference.free_symbols());
        if symbols.len() != 1 {
            continue;
        }
        let Some(cs) = coefficients(&difference, &symbols[0])? else {
            continue;
        };
        if cs.len() != 2 {
            continue;
        }
        let point: Basic = ng(&d2(&cs[0], &cs[1])?)?;
        let at: Basic = subs(value, &symbols[0], &point)?;
        if has_head(&at, &|h| {
            matches!(
                h,
                Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
            )
        }) {
            let strict: Basic = if rel_op(&atom) == Some(RelOp::Le) {
                crate::core::relational::lt(lhs, rhs)?
            } else {
                crate::core::relational::gt(lhs, rhs)?
            };
            replacements.insert(atom.clone(), strict);
        }
    }
    if replacements.is_empty() {
        return Ok(condition.clone());
    }
    crate::core::function::xreplace_evaluated(condition, &replacements)
}

/// The disjunction of two relations between the same sides as one
/// relation (`_union_of_relations`).
fn union_of_relations(r: &Basic, s: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::basic::RelOp::*;
    if r.args() != s.args() {
        return Ok(None);
    }
    let (Some(p), Some(q)) = (rel_op(r), rel_op(s)) else {
        return Ok(None);
    };
    let kinds = |x: crate::core::basic::RelOp, y: crate::core::basic::RelOp| (p == x && q == y) || (p == y && q == x);
    let (lhs, rhs) = (&r.args()[0], &r.args()[1]);
    if kinds(Lt, Gt) {
        return Ok(Some(crate::core::relational::ne(lhs, rhs)?));
    }
    if kinds(Le, Gt) || kinds(Lt, Ge) || kinds(Le, Ge) || kinds(Eq, Ne) {
        return Ok(Some(true_b()));
    }
    if kinds(Lt, Eq) {
        return Ok(Some(crate::core::relational::le(lhs, rhs)?));
    }
    if kinds(Gt, Eq) {
        return Ok(Some(crate::core::relational::ge(lhs, rhs)?));
    }
    Ok(None)
}

/// A disjunction of conjunctions of relations with the disjuncts which
/// differ in one relation merged (`_merged_disjunction`).
pub fn merged_disjunction(condition: &Basic) -> R {
    if !matches!(condition.head(), Head::Or) {
        return Ok(condition.clone());
    }
    let mut terms: Vec<BTreeSet<Basic>> = condition
        .args()
        .iter()
        .map(|d| {
            if matches!(d.head(), Head::And) {
                d.args().iter().cloned().collect()
            } else {
                std::iter::once(d.clone()).collect()
            }
        })
        .collect();
    let mut changed: bool = true;
    while changed && terms.len() > 1 {
        changed = false;
        'outer: for i in 0..terms.len() {
            for j in 0..terms.len() {
                if i == j {
                    continue;
                }
                if terms[i].is_subset(&terms[j]) {
                    terms.remove(j);
                    changed = true;
                    break 'outer;
                }
                let only_i: Vec<&Basic> = terms[i].difference(&terms[j]).collect();
                let only_j: Vec<&Basic> = terms[j].difference(&terms[i]).collect();
                if only_i.len() != 1 || only_j.len() != 1 {
                    continue;
                }
                let (r, s) = (only_i[0], only_j[0]);
                if rel_op(r).is_none() || rel_op(s).is_none() {
                    continue;
                }
                let Some(union) = union_of_relations(r, s)? else {
                    continue;
                };
                let mut merged: BTreeSet<Basic> = terms[i].intersection(&terms[j]).cloned().collect();
                if !is_true_b(&union) {
                    merged.insert(union);
                }
                let mut next: Vec<BTreeSet<Basic>> = terms
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .map(|(_, t)| t.clone())
                    .collect();
                next.push(merged);
                terms = next;
                changed = true;
                break 'outer;
            }
        }
    }
    if terms.iter().any(|t| t.is_empty()) {
        return Ok(true_b());
    }
    let disjuncts: Vec<Basic> = terms
        .iter()
        .map(|t| crate::core::logic::and_(&sorted_by_str(t.iter().cloned().collect())))
        .collect();
    Ok(crate::core::logic::or_(&disjuncts))
}

/// The relations of a condition in their canonical form
/// (`_canonical_relations`).
pub fn canonical_relations(condition: &Basic) -> R {
    replace_where(condition, &|n: &Basic| rel_op(n).is_some(), &|n: &Basic| {
        crate::core::relational::canonical(n)
    })
}

/// Whether `e` has an unevaluated integral (`is_unevaluated`).
pub fn is_unevaluated(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::Integral | Head::NonElementaryIntegral))
}

/// A result of `integrate` as a conditional value (`_from_integrate`).
pub fn from_integrate(value: &Basic, x: &Basic) -> Option<crate::integrals::conditions::ConditionalValue> {
    use crate::integrals::conditions::ConditionalValue;
    if is_unevaluated(value) {
        if matches!(value.head(), Head::Piecewise) && value.args().len() == 2 {
            let (first, condition) = pairs(value).into_iter().next()?;
            if !is_unevaluated(&first) && !condition.has(x) {
                return Some(ConditionalValue::new(first, condition));
            }
        }
        return None;
    }
    if value.has(x) {
        return None;
    }
    Some(ConditionalValue::new(value.clone(), true_b()))
}

/// The integrand in the forms the Mellin table reads (`_forms`).
pub fn forms(g: &Basic, t: &Basic, assumptions: &[Basic]) -> PolyResult<Vec<Basic>> {
    let _ = t;
    let mut found: Vec<Basic> = vec![g.clone()];
    let mut add_form = |e: Basic| {
        if !found.contains(&e) {
            found.push(e);
        }
    };
    let mut hints = crate::core::function::Hints::DEFAULT;
    hints.power_exp = false;
    add_form(combined_exponentials(
        &crate::core::function::expand_with_hints(g, Some(&hints), None)?,
        None,
    )?);
    add_form(denest(g, assumptions)?);
    if has_any_function(g, &[Function::Sin, Function::Cos]) {
        add_form(expand(&crate::simplify::fu::tr8(g, true)?)?);
    }
    let hints_r = crate::core::rewriting::RewriteHints::default();
    if has_any_function(g, &HYPERBOLIC) {
        add_form(expand(&crate::core::rewriting::rewrite(g, &[], "exp", &hints_r)?)?);
    }
    if has_any_function(g, &[Function::Exp]) {
        add_form(crate::polys::polytools::cancel(g, &[])?);
    }
    if has_any_function(g, &INVERSE_HYPERBOLIC) {
        let rewritten: Basic = expand(&crate::core::rewriting::rewrite(g, &[], "log", &hints_r)?)?;
        add_form(rewritten.clone());
        add_form(crate::core::function::expand_log(&rewritten, true, true, false)?);
    }
    let orthogonal: bool = has_head(g, &|h| {
        matches!(
            h,
            Head::Function(
                Function::Jacobi
                    | Function::Gegenbauer
                    | Function::ChebyshevT
                    | Function::ChebyshevU
                    | Function::Legendre
                    | Function::AssocLegendre
                    | Function::Hermite
                    | Function::HermiteProb
                    | Function::Laguerre
                    | Function::AssocLaguerre
            )
        )
    });
    if orthogonal {
        add_form(expand(&crate::functions::expand_func::expand_func(g, true)?)?);
        let laguerre: Basic = replace_where(
            g,
            &|n: &Basic| n.is_function(Function::Laguerre) && integer_of(&n.args()[0]).is_none(),
            &|n: &Basic| {
                let (k, u) = (&n.args()[0], &n.args()[1]);
                m2(
                    &apply(Function::Exp, std::slice::from_ref(u))?,
                    &crate::functions::special::hyper::hyper(&[a2(k, &Basic::one())?], &[Basic::one()], &ng(u)?)?,
                )
            },
        )?;
        add_form(combined_exponentials(&laguerre, None)?);
        if has_any_function(g, &[Function::Hermite]) {
            let hermite: Basic = replace_where(
                g,
                &|n: &Basic| n.is_function(Function::Hermite) && integer_of(&n.args()[0]).is_none(),
                &|n: &Basic| hermite_as_kummer(n, assumptions),
            )?;
            add_form(combined_exponentials(&hermite, None)?);
        }
    }
    Ok(found)
}

/// A number with a root of higher order written through its square roots
/// (`_simplest_root`); `nsimplify` is not ported, so a number which has
/// both raises not ported.
pub fn simplest_root(e: &Basic) -> R {
    if !is_number(e) || !e.preorder().iter().any(|n| n.is_pow()) {
        return Ok(e.clone());
    }
    let pows: Vec<Basic> = e.atoms(Some(&|n: &Basic| n.is_pow()));
    let square: bool = pows
        .iter()
        .any(|n| n.args()[1] == Basic::half() && integer_of(&n.args()[0]).is_some());
    let higher: bool = pows
        .iter()
        .any(|n| rational_of(&n.args()[1]).is_some_and(|(_, q)| q > 2));
    if !higher || !square {
        return Ok(e.clone());
    }
    Err(crate::error::not_ported("nsimplify (the simplest root of a number)"))
}

/// Whether `log(x)` sits inside another function or a denominator
/// (`_logarithm_inside`).
pub fn logarithm_inside(f: &Basic, x: &Basic) -> PolyResult<bool> {
    let logarithms: [Basic; 2] = [log(x)?, log(&d2(&Basic::one(), x)?)?];
    for node in f.atoms(Some(&|n: &Basic| {
        matches!(n.head(), Head::Function(_) | Head::Undefined(_))
    })) {
        if logarithms.iter().any(|l| node.has(l)) && !logarithms.contains(&node) {
            return Ok(true);
        }
    }
    for node in f.atoms(Some(&|n: &Basic| n.is_pow())) {
        let exponent: &Basic = &node.args()[1];
        if logarithms.contains(&node.args()[0])
            && (yes(exponent, Predicate::negative) || is(exponent, Predicate::integer) == Some(false))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

/// `zeta(s)` replaced by its Laurent expansion about 1 to the second
/// order (`_zeta_about_one`).
pub fn zeta_about_one(e: &Basic, s: &Basic) -> R {
    let sm1: Basic = s2(s, &Basic::one())?;
    let mut expansion: Basic = d2(&Basic::one(), &sm1)?;
    for n in 0..3i64 {
        let term: Basic = d2(
            &m2(
                &m2(
                    &p2(&int(-1), &int(n))?,
                    &crate::functions::special::zeta_functions::stieltjes(&int(n), None)?,
                )?,
                &p2(&sm1, &int(n))?,
            )?,
            &apply(Function::Factorial, &[int(n)])?,
        )?;
        expansion = a2(&expansion, &term)?;
    }
    replace_where(
        e,
        &|n: &Basic| n.is_function(Function::Zeta) && n.args() == std::slice::from_ref(s),
        &|_| Ok(expansion.clone()),
    )
}

/// Whether `f` takes a finite value at `point` (`_finite_at`).
pub fn finite_at(f: &Basic, x: &Basic, point: &Basic) -> PolyResult<bool> {
    let value: Basic = match subs(f, x, point) {
        Ok(v) => v,
        Err(crate::error::PolyError::Type(_) | crate::error::PolyError::Value(_)) => return Ok(false),
        Err(e) => return Err(e),
    };
    if has_head(&value, &|h| {
        matches!(
            h,
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
        )
    }) {
        return Ok(false);
    }
    if is_number(&value) {
        return Ok(match numeric_value(&value, 53) {
            None => false,
            Some(n) => {
                let (re, im) = parts(&n);
                !re.repr().is_infinite() && !im.repr().is_infinite()
            }
        });
    }
    Ok(yes(&value, Predicate::finite))
}

/// `(u**p)**q` written `u**(p*q)` for a base positive on the range
/// (`_flattened_powers`).
pub fn flattened_powers(h: &Basic, t: &Basic, lower: &Basic, upper: &Basic) -> R {
    let mut facts: Vec<Basic> = Vec::new();
    if !infinite(lower) {
        facts.push(crate::core::relational::gt(t, lower)?);
    }
    if !infinite(upper) {
        facts.push(crate::core::relational::lt(t, upper)?);
    }
    let positive = |u: &Basic| -> PolyResult<bool> {
        if yes(u, Predicate::positive) {
            return Ok(true);
        }
        Ok(crate::assumptions::ask::ask(&crate::core::relational::gt(u, &Basic::zero())?, &facts, None)? == Some(true))
    };
    let flatten = |node: &Basic| -> R {
        let (mut base, exponent) = (node.args()[0].clone(), node.args()[1].clone());
        if base.is_mul() && base.args().len() == 2 && base.args()[0] == int(-1) {
            let inner: Basic = base.args()[1].clone();
            if inner.is_pow()
                && integer_of(&inner.args()[1]).is_some_and(|k| k.rem_euclid(2) == 1)
                && inner.args()[0].is_add()
            {
                base = p2(&expand(&ng(&inner.args()[0])?)?, &inner.args()[1])?;
            }
        }
        if base.is_pow()
            && exponent.as_rational().is_some()
            && base.args()[1].as_rational().is_some()
            && positive(&base.args()[0])?
        {
            return p2(&base.args()[0], &m2(&base.args()[1], &exponent)?);
        }
        Ok(node.clone())
    };
    replace_where(
        h,
        &|n: &Basic| n.is_pow() && (n.args()[0].is_pow() || n.args()[0].is_mul()),
        &flatten,
    )
}

fn rational_function(e: &Basic, gens: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, gens, crate::core::expr::PolynomialKind::Rational) == Some(true)
}

/// Whether `h` is a rational function of `t` and of powers of
/// polynomials in `t` with rational exponents (`_algebraic`).
pub fn algebraic(h: &Basic, t: &Basic) -> PolyResult<bool> {
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    let mut placeholders: Vec<Basic> = vec![t.clone()];
    for node in h.atoms(Some(&|n: &Basic| n.is_pow())) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if base.has(t) && rational_of(exponent).is_some_and(|(_, q)| q != 1) {
            if !rational_function(base, std::slice::from_ref(t)) {
                return Ok(false);
            }
            let r: Basic = fresh("r", &[]);
            placeholders.push(r.clone());
            replacement.insert(node.clone(), r);
        }
    }
    let replaced: Basic = crate::core::function::xreplace_evaluated(h, &replacement)?;
    Ok(rational_function(&replaced, &placeholders))
}

/// `tan(v)`, the infinities at `+-pi/2` (`_tangent_of`).
pub fn tangent_of(v: &Basic) -> R {
    let half_pi: Basic = d2(&Basic::pi(), &int(2))?;
    if *v == half_pi {
        return Ok(oo());
    }
    if *v == ng(&half_pi)? {
        return Ok(neg_oo());
    }
    apply(Function::Tan, std::slice::from_ref(v))
}

/// The distinct bases of the non-integer rational powers of `f` depending
/// on `x` (`_radical_bases`).
pub fn radical_bases(f: &Basic, x: &Basic) -> Vec<Basic> {
    let mut bases: Vec<Basic> = Vec::new();
    for node in f.atoms(Some(&|n: &Basic| n.is_pow())) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if base.has(x) && rational_of(exponent).is_some_and(|(_, q)| q != 1) && !bases.contains(base) {
            bases.push(base.clone());
        }
    }
    bases
}

/// `(q, n)` for the substitution `u = q**(1/n)` (`_radical_base`).
pub fn radical_base(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, i64)>> {
    let polynomial: Vec<Basic> = radical_bases(f, x)
        .into_iter()
        .filter(|b| {
            crate::core::expr::polynomial_like(
                b,
                std::slice::from_ref(x),
                crate::core::expr::PolynomialKind::Polynomial,
            ) == Some(true)
        })
        .collect();
    if polynomial.len() != 1 {
        return Ok(None);
    }
    let q: Basic = polynomial[0].clone();
    let Some(cs) = coefficients(&q, x)? else {
        return Ok(None);
    };
    if cs.len() > 3 {
        return Ok(None);
    }
    let mut n: i64 = 1;
    for node in f.atoms(Some(&|n: &Basic| n.is_pow())) {
        if node.args()[0] == q {
            if let Some((_, d)) = rational_of(&node.args()[1]) {
                n = n / gcd_i64(n, d) * d;
            }
        }
    }
    Ok(if n > 1 { Some((q, n)) } else { None })
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i64(b, a % b)
    }
}

/// `powdenest(g, force=True)` with the signs of the parameters respected
/// (`_denest`).
pub fn denest(g: &Basic, assumptions: &[Basic]) -> R {
    let mut replacement: Values = Vec::new();
    let mut back: Values = Vec::new();
    for p in sorted_symbols(g.free_symbols()) {
        if yes(&p, Predicate::positive) {
            continue;
        }
        let name: String = p.to_string();
        if crate::assumptions::ask::ask(&crate::core::relational::gt(&p, &Basic::zero())?, assumptions, None)?
            == Some(true)
        {
            let d: Basic = fresh(&name, &[(Predicate::positive, true)]);
            replacement.push((p.clone(), d.clone()));
            back.push((d, p));
        } else if crate::assumptions::ask::ask(&crate::core::relational::lt(&p, &Basic::zero())?, assumptions, None)?
            == Some(true)
        {
            let d: Basic = fresh(&name, &[(Predicate::positive, true)]);
            replacement.push((p.clone(), ng(&d)?));
            back.push((d, ng(&p)?));
        } else {
            return crate::simplify::powsimp::powdenest(g, false, false);
        }
    }
    let denested: Basic = crate::simplify::powsimp::powdenest(&xr(g, &replacement)?, true, false)?;
    xr(&denested, &back)
}

/// `f` at `x = asin(sqrt(u))` with its trigonometric functions written in
/// `u` (`_arcsine_substitution`).
pub fn arcsine_substitution(f: &Basic, x: &Basic, u: &Basic) -> R {
    let s: Basic = p2(u, &Basic::half())?;
    let c: Basic = p2(&s2(&Basic::one(), u)?, &Basic::half())?;
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(unevaluated(Function::Sin, x), s.clone());
    rule.insert(unevaluated(Function::Cos, x), c.clone());
    rule.insert(unevaluated(Function::Tan, x), d2(&s, &c)?);
    rule.insert(unevaluated(Function::Cot, x), d2(&c, &s)?);
    rule.insert(unevaluated(Function::Sec, x), d2(&Basic::one(), &c)?);
    rule.insert(unevaluated(Function::Csc, x), d2(&Basic::one(), &s)?);
    let replaced: Basic = crate::core::function::xreplace_evaluated(f, &rule)?;
    subs(&replaced, x, &apply(Function::Asin, &[s])?)
}

/// `f` with `x = t + a` (`_shift`).
pub fn shift(f: &Basic, x: &Basic, t: &Basic, a: &Basic) -> R {
    subs(f, x, &a2(t, a)?)
}

/// `k` when `e == k*pi/2` (`_quarter`).
pub fn quarter(e: &Basic) -> PolyResult<Option<i64>> {
    let k: Basic = d2(&m2(&int(2), e)?, &Basic::pi())?;
    Ok(integer_of(&k))
}

/// The common generator `c` of the exponentials `exp(c_i x)` of `f`
/// (`_exponential_generator`).
pub fn exponential_generator(f: &Basic, x: &Basic, negative: bool) -> PolyResult<Option<Basic>> {
    let mut coefficients: Vec<Rat> = Vec::new();
    let mut symbolic: usize = 0;
    for node in f.atoms(Some(&|n: &Basic| n.is_function(Function::Exp))) {
        let argument: &Basic = &node.args()[0];
        if argument
            .atoms(Some(&|n: &Basic| n.is_function(Function::Exp)))
            .iter()
            .any(|inner| inner.has(x))
        {
            continue;
        }
        let Some((c, e)) = crate::integrals::mellin::monomial(argument, x)? else {
            return Ok(None);
        };
        if e != Basic::one() {
            return Ok(None);
        }
        match c.as_rational() {
            Some(r) => coefficients.push(r.clone()),
            None => symbolic += 1,
        }
    }
    if coefficients.is_empty() && symbolic == 0 {
        return Ok(None);
    }
    let (mut numerator, mut denominator) = (Int::ZERO, Int::ONE);
    for c in &coefficients {
        numerator = gcd(&numerator, &crate::numbers::numer(c));
        let q: Int = crate::numbers::denom(c);
        denominator = &denominator / gcd(&denominator, &q) * &q;
    }
    let generator: Basic = if coefficients.is_empty() {
        Basic::one()
    } else {
        Basic::from_rational(Rat::from(numerator) / Rat::from(denominator))
    };
    if !coefficients.is_empty() && coefficients.iter().all(|k| *k < Rat::ZERO) {
        return Ok(Some(ng(&generator)?));
    }
    if !coefficients.is_empty() && coefficients.iter().all(|k| *k > Rat::ZERO) {
        return Ok(Some(generator));
    }
    Ok(Some(if negative { ng(&generator)? } else { generator }))
}

/// `exp(c*log(u) + rest)` written `u**c*exp(rest)` (`_powers_of_logarithms`).
pub fn powers_of_logarithms(g: &Basic, u: &Basic) -> R {
    let logarithm: Basic = log(u)?;
    let rewrite = |node: &Basic| -> R {
        let mut power: Basic = Basic::one();
        let mut rest: Vec<Basic> = Vec::new();
        for term in add_args(&node.args()[0]) {
            let (coefficient, factor) = crate::core::expr::as_independent(&term, &logarithm, false)?;
            if factor == logarithm && !coefficient.has(u) {
                power = m2(&power, &p2(u, &coefficient)?)?;
            } else {
                rest.push(term);
            }
        }
        m2(&power, &exp(&add_all(&rest)?)?)
    };
    replace_where(
        g,
        &|n: &Basic| n.is_function(Function::Exp) && n.has(&logarithm),
        &rewrite,
    )
}

/// Whether `log(u)` occurs only as `log(u)**n` (`_log_powers_only`).
pub fn log_powers_only(g: &Basic, u: &Basic) -> PolyResult<bool> {
    let logarithm: Basic = log(u)?;
    Ok(g.atoms(Some(&|n: &Basic| n.is_function(Function::Log)))
        .iter()
        .all(|n| !n.has(u) || *n == logarithm))
}

/// The zeros of `sin(w*x + c)` or `cos(w*x + c)` inside a bounded numeric
/// range, enumerated (`_trigonometric_zeros`).
pub fn trigonometric_zeros(u: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let sine: bool = u.is_function(Function::Sin);
    if !(sine || u.is_function(Function::Cos)) || !(is_number(a) && is_number(b)) || infinite(a) || infinite(b) {
        return Ok(None);
    }
    let Some(cs) = coefficients(&u.args()[0], x)? else {
        return Ok(None);
    };
    if cs.len() != 2 {
        return Ok(None);
    }
    let (w, c) = (cs[1].clone(), cs[0].clone());
    if !(is_number(&w) && is_number(&c) && yes(&w, Predicate::real) && yes(&c, Predicate::real)) || w == Basic::zero() {
        return Ok(None);
    }
    let shift_: Basic = if sine {
        Basic::zero()
    } else {
        d2(&Basic::pi(), &int(2))?
    };
    let end = |p: &Basic| -> PolyResult<Option<f64>> {
        Ok(to_float(&d2(&s2(&a2(&m2(&w, p)?, &c)?, &shift_)?, &Basic::pi())?))
    };
    let (Some(e0), Some(e1)) = (end(a)?, end(b)?) else {
        return Ok(None);
    };
    let (lo, hi) = if e0 <= e1 { (e0, e1) } else { (e1, e0) };
    if hi - lo > 1000.0 {
        return Ok(None);
    }
    let mut zeros: Vec<Basic> = Vec::new();
    for k in (lo.trunc() as i64 - 1)..(hi.trunc() as i64 + 2) {
        let point: Basic = d2(&s2(&a2(&m2(&int(k), &Basic::pi())?, &shift_)?, &c)?, &w)?;
        let above: Option<bool> = is(&s2(&point, a)?, Predicate::positive);
        let below: Option<bool> = is(&s2(b, &point)?, Predicate::positive);
        let (Some(above), Some(below)) = (above, below) else {
            return Ok(None);
        };
        if above && below {
            zeros.push(point);
        }
    }
    let mut keyed: Vec<(f64, Basic)> = zeros
        .into_iter()
        .map(|z| (to_float(&z).unwrap_or(f64::NAN), z))
        .collect();
    keyed.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Some(keyed.into_iter().map(|(_, z)| z).collect()))
}

/// `hermite(2n, u)` or `hermite(2n + 1, u)` through Kummer's function of a
/// negative argument (`_hermite_as_kummer`).
pub fn hermite_as_kummer(node: &Basic, assumptions: &[Basic]) -> R {
    let (degree, u) = (&node.args()[0], &node.args()[1]);
    let gamma = |e: &Basic| apply(Function::Gamma, std::slice::from_ref(e));
    for shift_ in [0i64, 1] {
        let n: Basic = d2(&s2(degree, &int(shift_))?, &int(2))?;
        let integer: bool = is(&n, Predicate::integer) == Some(true)
            || crate::assumptions::ask::ask(
                &crate::assumptions::facts::element(&n, &Basic::atom(Head::Integers))?,
                assumptions,
                None,
            )? == Some(true);
        if !integer {
            continue;
        }
        let u2: Basic = p2(u, &int(2))?;
        let sign: Basic = p2(&int(-1), &n)?;
        if shift_ == 0 {
            let factor: Basic = d2(
                &m2(&sign, &gamma(&a2(&m2(&int(2), &n)?, &int(1))?)?)?,
                &gamma(&a2(&n, &int(1))?)?,
            )?;
            let kummer: Basic =
                crate::functions::special::hyper::hyper(&[a2(&n, &Basic::half())?], &[Basic::half()], &ng(&u2)?)?;
            return m2(&m2(&factor, &exp(&u2)?)?, &kummer);
        }
        let factor: Basic = d2(
            &m2(&sign, &gamma(&a2(&m2(&int(2), &n)?, &int(2))?)?)?,
            &gamma(&a2(&n, &int(1))?)?,
        )?;
        let kummer: Basic = crate::functions::special::hyper::hyper(&[a2(&n, &rat(3, 2))?], &[rat(3, 2)], &ng(&u2)?)?;
        return m2(&m2(&m2(&m2(&factor, &int(2))?, u)?, &exp(&u2)?)?, &kummer);
    }
    Ok(node.clone())
}

/// The exponential factors of each product combined into one
/// (`_combined_exponentials`).
pub fn combined_exponentials(f: &Basic, x: Option<&Basic>) -> R {
    if !has_any_function(f, &[Function::Exp]) {
        return Ok(f.clone());
    }
    let combine = |node: &Basic| -> R {
        let exponentials: Vec<Basic> = node
            .args()
            .iter()
            .filter(|p| p.is_function(Function::Exp) && x.map_or(true, |x| p.has(x)))
            .cloned()
            .collect();
        if exponentials.len() < 2 {
            return Ok(node.clone());
        }
        let rest: Vec<Basic> = node
            .args()
            .iter()
            .filter(|p| !exponentials.contains(p))
            .cloned()
            .collect();
        let arguments: Vec<Basic> = exponentials.iter().map(|e| e.args()[0].clone()).collect();
        m2(&mul_all(&rest)?, &exp(&add_all(&arguments)?)?)
    };
    replace_where(f, &|n: &Basic| n.is_mul(), &combine)
}

/// A quotient of sums of sines and cosines with the sums written as
/// products and cancelled (`_sums_to_products`).
pub fn sums_to_products(f: &Basic, x: &Basic) -> R {
    let (numerator, denominator) = crate::core::expr::as_numer_denom(f)?;
    let trigonometric_sum = |e: &Basic| -> PolyResult<bool> {
        if !e.is_add() {
            return Ok(false);
        }
        for term in e.args() {
            let (_, rest) = crate::core::expr::as_independent(term, x, false)?;
            if !(rest.is_function(Function::Sin) || rest.is_function(Function::Cos))
                || crate::core::expr::polynomial_like(
                    &rest.args()[0],
                    std::slice::from_ref(x),
                    crate::core::expr::PolynomialKind::Polynomial,
                ) != Some(true)
            {
                return Ok(false);
            }
        }
        Ok(true)
    };
    if !(trigonometric_sum(&numerator)? || trigonometric_sum(&denominator)?) {
        return Ok(f.clone());
    }
    let rewritten: Basic = crate::polys::polytools::cancel(
        &d2(
            &crate::simplify::fu::tr9(&numerator)?,
            &crate::simplify::fu::tr9(&denominator)?,
        )?,
        &[],
    )?;
    Ok(if ops(&rewritten)? < ops(f)? {
        rewritten
    } else {
        f.clone()
    })
}

/// The trigonometric sums which are squares written as such
/// (`_trigonometric_squares`).
pub fn trigonometric_squares(base: &Basic) -> R {
    fn half_angle(node: &Basic) -> R {
        if !node.is_add() || node.args().len() != 2 {
            return Ok(node.clone());
        }
        let terms: Vec<Basic> = node.args().to_vec();
        for (constant, other) in [(&terms[0], &terms[1]), (&terms[1], &terms[0])] {
            if has_trigonometric(constant) {
                continue;
            }
            let cosines: Vec<Basic> = mul_args(other)
                .into_iter()
                .filter(|p| p.is_function(Function::Cos))
                .collect();
            if cosines.len() != 1 {
                continue;
            }
            let coefficient: Basic = d2(other, &cosines[0])?;
            let half: Basic = d2(&cosines[0].args()[0], &int(2))?;
            if coefficient == *constant {
                return m2(&m2(&int(2), constant)?, &p2(&apply(Function::Cos, &[half])?, &int(2))?);
            }
            if coefficient == ng(constant)? {
                return m2(&m2(&int(2), constant)?, &p2(&apply(Function::Sin, &[half])?, &int(2))?);
            }
        }
        Ok(node.clone())
    }
    fn squares(node: &Basic) -> R {
        if node.is_add() && has_trigonometric(node) {
            let simplified: Basic =
                crate::simplify::trigsimp::trigsimp(node, crate::simplify::trigsimp::Method::Matching)?;
            let found: Basic = replace_where(&simplified, &|n: &Basic| n.is_add(), &half_angle)?;
            if !found.is_add() {
                return Ok(found);
            }
        }
        if node.args().is_empty() {
            return Ok(node.clone());
        }
        let args: Vec<Basic> = node.args().iter().map(squares).collect::<PolyResult<_>>()?;
        crate::core::function::rebuild_evaluated(node, args)
    }
    replace_where(&squares(base)?, &|n: &Basic| n.is_add(), &half_angle)
}

/// The largest base of a power factored for its signs (`_FACTOR_LIMIT`).
const FACTOR_LIMIT: i64 = 200;

/// The factored argument of a power or a logarithm: the constant, the
/// factors with their multiplicities, whether a square appeared
/// (`_Factored`).
pub type Factored = (Basic, Vec<(Basic, Basic)>, bool);

/// `base` as a constant times powers of continuous factors depending on
/// `x` (`_factored`).
pub fn factored(base: &Basic, x: &Basic) -> PolyResult<Option<Factored>> {
    if ops(base)? > FACTOR_LIMIT {
        return Ok(None);
    }
    let g: Basic = if has_trigonometric(base) {
        trigonometric_squares(base)?
    } else {
        base.clone()
    };
    let mut squares: bool = g != *base;
    let attempted = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::polys::polytools::factor(
            &crate::polys::rationaltools::together(&g, false, true)?,
            &[],
            false,
            true,
        )
    });
    let factored: Basic = match attempted {
        Ok(Some(f)) => f,
        Ok(None) => return Ok(None),
        Err(
            crate::error::PolyError::Polys(..) | crate::error::PolyError::Value(_) | crate::error::PolyError::Type(_),
        ) => return Ok(None),
        Err(e) => return Err(e),
    };
    let (constant, rest) = crate::core::expr::as_independent(&factored, x, false)?;
    let mut factors: Vec<(Basic, Basic)> = Vec::new();
    let excluded: [Function; 8] = [
        Function::Tan,
        Function::Cot,
        Function::Sec,
        Function::Csc,
        Function::Floor,
        Function::Ceiling,
        Function::Frac,
        Function::Sign,
    ];
    for part in mul_args(&rest) {
        let (mut b_, mut e_) = crate::core::expr::as_base_exp(&part)?;
        if e_.as_rational().is_none() {
            b_ = part.clone();
            e_ = Basic::one();
        }
        if has_any_function(&b_, &excluded)
            || has_any_function(&b_, &[Function::Heaviside])
            || has_head(&b_, &|h| matches!(h, Head::Piecewise))
            || b_
                .atoms(Some(&|n: &Basic| n.is_pow()))
                .iter()
                .any(|n| n.args()[0].has(x) && yes(&n.args()[1], Predicate::negative))
        {
            return Ok(None);
        }
        squares = squares || integer_of(&e_).is_some_and(|k| k.abs() >= 2);
        factors.push((b_, e_));
    }
    Ok(Some((constant, factors, squares)))
}

/// Whether a multiplicity is an even integer (`_even`).
pub fn even(e: &Basic) -> bool {
    integer_of(e).is_some_and(|k| k % 2 == 0)
}

/// The powers with a non-integer exponent and the logarithms whose
/// argument depends on `x` (`_signed_nodes`).
pub fn signed_nodes(f: &Basic, x: &Basic) -> Vec<Basic> {
    let nodes: Vec<Basic> = sorted_by_str(f.atoms(Some(&|n: &Basic| n.is_pow() || n.is_function(Function::Log))));
    nodes
        .into_iter()
        .filter(|node| {
            if node.is_pow() {
                integer_of(&node.args()[1]).is_none() && node.args()[0].has(x)
            } else {
                node.args()[0].has(x)
            }
        })
        .collect()
}

// -- the integrator

use crate::integrals::conditions::ConditionalValue;
use std::cell::RefCell;

type OptCV = PolyResult<Option<ConditionalValue>>;

/// The points strictly inside a range, sorted, and those whose position
/// against an endpoint the assumptions do not settle (`_Placement`).
pub type Placement = (Vec<Basic>, Vec<Basic>);

/// The depth of the recursive splitting (`_MAX_DEPTH`).
const MAX_DEPTH: usize = 6;
/// The points whose positions make the cases (`_MAX_CASE_POINTS`).
const MAX_CASE_POINTS: usize = 2;
/// The integer levels a function may cross (`_MAX_LEVELS`).
const MAX_LEVELS: i64 = 64;

const KINKS: [Function; 5] = [
    Function::Abs,
    Function::Sign,
    Function::Heaviside,
    Function::Max,
    Function::Min,
];

fn has_kinks(f: &Basic) -> bool {
    has_any_function(f, &KINKS) || has_head(f, &|h| matches!(h, Head::Piecewise))
}

fn zero_cv() -> ConditionalValue {
    ConditionalValue::new(Basic::zero(), true_b())
}

fn has_bad(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::NaN | Head::ComplexInfinity))
}

fn has_infinity(e: &Basic) -> bool {
    has_head(e, &|h| {
        matches!(
            h,
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity
        )
    })
}

/// The strategies, sharing the assumptions (`_Integrator`).
pub struct Integrator {
    pub assumptions: Vec<Basic>,
    pub parametric: bool,
    pub principal_value: bool,
    pub finite_part: bool,
    pub regularize: bool,
    /// whether an unsettled singularity or kink splits the parameters into
    /// cases
    pub cases: bool,
    factored_cache: RefCell<HashMap<Basic, Option<Factored>>>,
    zero_cache: RefCell<HashMap<(Basic, Basic, Basic), Option<Placement>>>,
    singular_cache: RefCell<HashMap<(Basic, Basic, Basic), Option<Placement>>>,
}

impl Integrator {
    pub fn new(
        assumptions: &[Basic],
        parametric: bool,
        principal_value: bool,
        finite_part: bool,
        regularize: bool,
        cases: bool,
    ) -> Integrator {
        Integrator {
            assumptions: assumptions.to_vec(),
            parametric,
            principal_value,
            finite_part,
            regularize,
            cases,
            factored_cache: RefCell::new(HashMap::new()),
            zero_cache: RefCell::new(HashMap::new()),
            singular_cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn ask(&self, query: &Basic) -> PolyResult<Option<bool>> {
        crate::assumptions::ask::ask(query, &self.assumptions, None)
    }

    fn holds(&self, query: PolyResult<Basic>) -> PolyResult<Option<bool>> {
        self.ask(&query?)
    }

    fn factors_of(&self, base: &Basic, x: &Basic) -> PolyResult<Option<Factored>> {
        if let Some(found) = self.factored_cache.borrow().get(base) {
            return Ok(found.clone());
        }
        let found: Option<Factored> = factored(base, x)?;
        self.factored_cache.borrow_mut().insert(base.clone(), found.clone());
        Ok(found)
    }

    /// The powers and logarithms of `f` to split, with the constant and
    /// the factors of their arguments (`_nodes`).
    #[allow(clippy::type_complexity)]
    fn nodes(&self, f: &Basic, x: &Basic, full: bool) -> PolyResult<Vec<(Basic, Basic, Vec<(Basic, Basic)>)>> {
        let mut found = Vec::new();
        for node in signed_nodes(f, x) {
            let base: Basic = node.args()[0].clone();
            if let Some((constant, factors, squares)) = self.factors_of(&base, x)? {
                if full || squares {
                    found.push((node, constant, factors));
                }
            }
        }
        Ok(found)
    }

    /// The points of `(a, b)` where `u` takes an integer value
    /// (`_integer_crossings`).
    fn integer_crossings(&self, u: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
        let Some(singular) = self.singular_placement(u, x, a, b)? else {
            return Ok(None);
        };
        if !singular.0.is_empty() || !singular.1.is_empty() {
            return Ok(None);
        }
        let derivative: Basic = crate::core::function::derivative(u, x)?;
        let mut critical: Vec<Basic> = Vec::new();
        if derivative.has(x) {
            let Some(zeros) = self.zeros(&derivative, x, a, b)? else {
                return Ok(None);
            };
            critical = zeros;
        }
        let mut values: Vec<f64> = Vec::new();
        let mut points: Vec<Basic> = vec![a.clone(), b.clone()];
        points.extend(critical);
        for point in &points {
            let Some(number) = crate::numeric::reliable_value(&subs(u, x, point)?, 30, &[]) else {
                return Ok(None);
            };
            let (re, im) = parts(&number);
            let (re, im) = (to_f64(&re), to_f64(&im));
            if im.abs() > 1e-20 * (1.0 + re.abs()) {
                return Ok(None);
            }
            values.push(re);
        }
        let min: f64 = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max: f64 = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (lowest, highest) = ((min - 1e-9).ceil() as i64, (max + 1e-9).floor() as i64);
        if highest - lowest > MAX_LEVELS {
            return Ok(None);
        }
        let mut found: Vec<Basic> = Vec::new();
        for level in lowest..=highest {
            let Some(zeros) = self.zeros(&s2(u, &int(level))?, x, a, b)? else {
                return Ok(None);
            };
            found.extend(zeros);
        }
        Ok(Some(found))
    }

    /// The points inside `(a, b)`, sorted (`_points_in`).
    fn points_in(&self, points: &[Basic], a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
        match self.place(points, a, b)? {
            Some((inside, undecided)) if undecided.is_empty() => Ok(Some(inside)),
            _ => Ok(None),
        }
    }

    /// The points strictly inside `(a, b)`, sorted, and the undecided
    /// ones (`_place`).
    pub fn place(&self, points: &[Basic], a: &Basic, b: &Basic) -> PolyResult<Option<Placement>> {
        use crate::core::relational as rl;
        let mut inside: Vec<Basic> = Vec::new();
        let mut undecided: Vec<Basic> = Vec::new();
        for p in points {
            if is(p, Predicate::real) == Some(false) || inside.contains(p) || undecided.contains(p) {
                continue;
            }
            if is(p, Predicate::extended_real) != Some(true)
                && !real_for_real_parameters(p)?
                && self.ask(&crate::assumptions::facts::element(p, &Basic::atom(Head::Reals))?)? != Some(true)
            {
                return Ok(None);
            }
            let below: Option<bool> = if *a != neg_oo() {
                self.holds(rl::gt(p, a))?
            } else {
                Some(true)
            };
            let above: Option<bool> = if *b != oo() {
                self.holds(rl::lt(p, b))?
            } else {
                Some(true)
            };
            if below == Some(false) || above == Some(false) {
                continue;
            }
            if below.is_none() || above.is_none() {
                undecided.push(p.clone());
            } else {
                inside.push(p.clone());
            }
        }
        for i in 0..inside.len() {
            for j in i + 1..inside.len() {
                if self.holds(rl::lt(&inside[i], &inside[j]))?.is_none() {
                    return Ok(None);
                }
            }
        }
        let mut keyed: Vec<(Vec<u8>, Basic)> = Vec::new();
        for p in &inside {
            let mut key: Vec<u8> = Vec::new();
            for q in &inside {
                key.push(if self.holds(rl::lt(p, q))? == Some(true) { 0 } else { 1 });
            }
            keyed.push((key, p.clone()));
        }
        keyed.sort_by(|u, v| u.0.cmp(&v.0));
        Ok(Some((keyed.into_iter().map(|(_, p)| p).collect(), undecided)))
    }

    /// The zeros of `u` in `(a, b)` (`_zeros`).
    fn zeros(&self, u: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
        match self.zero_placement(u, x, a, b)? {
            Some((inside, undecided)) if undecided.is_empty() => Ok(Some(inside)),
            _ => Ok(None),
        }
    }

    fn zero_placement(&self, u: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Placement>> {
        let key = (u.clone(), a.clone(), b.clone());
        if let Some(found) = self.zero_cache.borrow().get(&key) {
            return Ok(found.clone());
        }
        let found: Option<Placement> = self.solve_zeros(u, x, a, b)?;
        self.zero_cache.borrow_mut().insert(key, found.clone());
        Ok(found)
    }

    fn solve_zeros(&self, u: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Placement>> {
        if let Some(zeros) = trigonometric_zeros(u, x, a, b)? {
            return Ok(Some((zeros, Vec::new())));
        }
        let mut bounds: Vec<Basic> = Vec::new();
        if *a != neg_oo() {
            bounds.push(crate::core::relational::gt(x, a)?);
        }
        if *b != oo() {
            bounds.push(crate::core::relational::lt(x, b)?);
        }
        let free: BTreeSet<Basic> = u.free_symbols();
        let relevant: Vec<Basic> = self
            .assumptions
            .iter()
            .filter(|item| item.free_symbols().iter().any(|s| free.contains(s)))
            .cloned()
            .collect();
        let assumptions: Vec<Basic> = with(&relevant, &bounds);
        let reals: Basic = Basic::atom(Head::Reals);
        let found = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
            crate::assumptions::solve::solve(
                std::slice::from_ref(u),
                Some(std::slice::from_ref(x)),
                &assumptions,
                Some(&reals),
                false,
            )
        });
        let found: Basic = match found {
            Ok(Some(v)) => v,
            Ok(None) => return Ok(None),
            Err(crate::error::PolyError::Value(_) | crate::error::PolyError::Type(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        let Some(candidates) = candidate_points(&found, x, a, b)? else {
            return Ok(None);
        };
        self.place(&candidates, a, b)
    }

    /// The singularities of `f` placed against `(a, b)`
    /// (`_singular_placement`).
    fn singular_placement(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Placement>> {
        let key = (f.clone(), a.clone(), b.clone());
        if let Some(found) = self.singular_cache.borrow().get(&key) {
            return Ok(found.clone());
        }
        let found: Option<Vec<Basic>> =
            crate::timeout::attempt_within(crate::settings::settings().timeout, || singular_points(f, x, a, b))?
                .flatten();
        let placed: Option<Placement> = match found {
            None => None,
            Some(points) => self.place(&points, a, b)?,
        };
        self.singular_cache.borrow_mut().insert(key, placed.clone());
        Ok(placed)
    }

    /// The points where a branch of a kink of `f` changes, placed
    /// (`_kink_placement`).
    fn kink_placement(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Placement>> {
        let mut arguments: Vec<Basic> = Vec::new();
        for node in f.atoms(Some(&|n: &Basic| {
            n.is_function(Function::Abs) || n.is_function(Function::Sign) || n.is_function(Function::Heaviside)
        })) {
            arguments.push(node.args()[0].clone());
        }
        for node in f.atoms(Some(&|n: &Basic| {
            n.is_function(Function::Max) || n.is_function(Function::Min)
        })) {
            if node.args().len() == 2 {
                arguments.push(s2(&node.args()[0], &node.args()[1])?);
            }
        }
        for node in f.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Piecewise))) {
            for (_, condition) in pairs(&node) {
                for atom in condition.atoms(Some(&|n: &Basic| rel_op(n).is_some())) {
                    arguments.push(s2(&atom.args()[0], &atom.args()[1])?);
                }
            }
        }
        let mut inside: Vec<Basic> = Vec::new();
        let mut undecided: Vec<Basic> = Vec::new();
        for u in arguments {
            if !u.has(x) {
                continue;
            }
            let Some((i, un)) = self.zero_placement(&u, x, a, b)? else {
                return Ok(None);
            };
            inside.extend(i);
            for p in un {
                if !undecided.contains(&p) {
                    undecided.push(p);
                }
            }
        }
        Ok(Some((inside, undecided)))
    }

    /// Whether `f` has a kink or a singularity whose position the
    /// assumptions do not settle (`_parametric_points`).
    pub fn parametric_points(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<bool> {
        if !self.cases || parameters_of(f, x, a, b).is_empty() {
            return Ok(false);
        }
        if let Some((_, undecided)) = self.kink_placement(f, x, a, b)? {
            if !undecided.is_empty() {
                return Ok(true);
            }
        }
        if has_kinks(f) {
            return Ok(false);
        }
        Ok(self.singular_placement(f, x, a, b)?.is_some_and(|(_, u)| !u.is_empty()))
    }

    fn sample(&self, a: &Basic, b: &Basic) -> R {
        if *a == neg_oo() && *b == oo() {
            return Ok(Basic::zero());
        }
        if *a == neg_oo() {
            return s2(b, &Basic::one());
        }
        if *b == oo() {
            return a2(a, &Basic::one());
        }
        d2(&a2(a, b)?, &int(2))
    }

    fn sign_at(&self, u: &Basic, x: &Basic, sample: &Basic) -> PolyResult<Option<i64>> {
        use crate::core::relational as rl;
        let value: Basic = subs(u, x, sample)?;
        let zero: Basic = Basic::zero();
        if self.holds(rl::gt(&value, &zero))? == Some(true) {
            return Ok(Some(1));
        }
        if self.holds(rl::lt(&value, &zero))? == Some(true) {
            return Ok(Some(-1));
        }
        if self.holds(rl::ge(&value, &zero))? == Some(true) {
            return Ok(Some(1));
        }
        if self.holds(rl::le(&value, &zero))? == Some(true) {
            return Ok(Some(-1));
        }
        Ok(None)
    }

    /// `f` with its kinks replaced by their branch on `(a, b)` and its
    /// powers and logarithms split over the factors of known sign
    /// (`_branch`).
    pub fn branch(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        unplaced: &[Basic],
        full: bool,
    ) -> PolyResult<Option<Basic>> {
        let sample: Basic = self.sample(a, b)?;
        let mut result: Basic = f.clone();
        let sign_of = |u: &Basic| self.sign_at(u, x, &sample);
        for (node, constant, factors) in self.nodes(&result, x, full)? {
            if let Some(rewritten) = self.signed(&node, &constant, &factors, &sign_of, unplaced, full)? {
                result = replaced_node(&result, &node, &rewritten)?;
            }
        }
        let kinks: Vec<Basic> = sorted_by_str(result.atoms(Some(&|n: &Basic| {
            n.is_function(Function::Abs) || n.is_function(Function::Sign) || n.is_function(Function::Heaviside)
        })));
        for kink in kinks {
            let u: Basic = kink.args()[0].clone();
            if !u.has(x) {
                continue;
            }
            let Some(s) = sign_of(&u)? else {
                return Ok(None);
            };
            let replacement: Basic = if kink.is_function(Function::Abs) {
                m2(&int(s), &u)?
            } else if kink.is_function(Function::Sign) {
                int(s)
            } else if s > 0 {
                Basic::one()
            } else {
                Basic::zero()
            };
            result = replaced_node(&result, &kink, &replacement)?;
        }
        let extrema: Vec<Basic> = sorted_by_str(result.atoms(Some(&|n: &Basic| {
            n.is_function(Function::Max) || n.is_function(Function::Min)
        })));
        for extremum in extrema {
            if !extremum.has(x) || extremum.args().len() != 2 {
                continue;
            }
            let (u, v) = (extremum.args()[0].clone(), extremum.args()[1].clone());
            let Some(s) = sign_of(&s2(&u, &v)?)? else {
                return Ok(None);
            };
            let (larger, smaller) = if s > 0 { (u, v) } else { (v, u) };
            let chosen: Basic = if extremum.is_function(Function::Max) {
                larger
            } else {
                smaller
            };
            result = replaced_node(&result, &extremum, &chosen)?;
        }
        for branches in result.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Piecewise))) {
            if !branches.has(x) {
                continue;
            }
            let mut chosen: Option<Basic> = None;
            for (value, cond) in pairs(&branches) {
                if is_true_b(&cond) {
                    chosen = Some(value);
                    break;
                }
                match self.ask(&subs(&cond, x, &sample)?)? {
                    None => return Ok(None),
                    Some(true) => {
                        chosen = Some(value);
                        break;
                    }
                    Some(false) => {}
                }
            }
            let Some(chosen) = chosen else {
                return Ok(None);
            };
            result = replaced_node(&result, &branches, &chosen)?;
        }
        Ok(Some(result))
    }

    /// A power or a logarithm of a product split over the factors of known
    /// sign (`_signed`).
    fn signed(
        &self,
        node: &Basic,
        constant: &Basic,
        factors: &[(Basic, Basic)],
        sign_of: &dyn Fn(&Basic) -> PolyResult<Option<i64>>,
        unplaced: &[Basic],
        full: bool,
    ) -> PolyResult<Option<Basic>> {
        let exponent: Basic = if node.is_pow() {
            node.args()[1].clone()
        } else {
            Basic::one()
        };
        let mut positive: Vec<(Basic, Basic)> = Vec::new();
        let mut remainder: Vec<Basic> = Vec::new();
        let mut total_sign: Basic = Basic::one();
        for (g, e) in factors {
            if unplaced.contains(g) || (!full && !even(e)) {
                remainder.push(p2(g, e)?);
                continue;
            }
            let Some(s) = sign_of(g)? else {
                remainder.push(p2(g, e)?);
                continue;
            };
            positive.push((m2(&int(s), g)?, e.clone()));
            total_sign = m2(&total_sign, &p2(&int(s), e)?)?;
        }
        if *constant != Basic::one() {
            use crate::core::relational as rl;
            if self.holds(rl::gt(constant, &Basic::zero()))? == Some(true) {
                positive.push((constant.clone(), Basic::one()));
            } else if self.holds(rl::lt(constant, &Basic::zero()))? == Some(true) {
                positive.push((ng(constant)?, Basic::one()));
                total_sign = ng(&total_sign)?;
            } else {
                remainder.push(constant.clone());
            }
        }
        if positive.is_empty() {
            return Ok(None);
        }
        let rest: Basic = m2(&total_sign, &mul_all(&remainder)?)?;
        let rewritten: Basic = if node.is_pow() {
            let powers: Vec<Basic> = positive
                .iter()
                .map(|(g, e)| p2(g, &m2(e, &exponent)?))
                .collect::<PolyResult<_>>()?;
            let tail: Basic = if rest != Basic::one() {
                p2(&rest, &exponent)?
            } else {
                Basic::one()
            };
            m2(&mul_all(&powers)?, &tail)?
        } else {
            let logs: Vec<Basic> = positive
                .iter()
                .map(|(g, e)| m2(e, &log(g)?))
                .collect::<PolyResult<_>>()?;
            let tail: Basic = if rest != Basic::one() {
                log(&rest)?
            } else {
                Basic::zero()
            };
            a2(&add_all(&logs)?, &tail)?
        };
        Ok(if rewritten == *node { None } else { Some(rewritten) })
    }

    /// The points where a branch of `f` changes, the unplaced factors and
    /// the undecided kinks (`_breakpoints`).
    #[allow(clippy::type_complexity)]
    fn breakpoints(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        full: bool,
    ) -> PolyResult<Option<(Vec<Basic>, Vec<Basic>, Vec<Basic>)>> {
        let Some((kinks_inside, kinks_undecided)) = self.kink_placement(f, x, a, b)? else {
            return Ok(None);
        };
        let mut points: Vec<Basic> = kinks_inside;
        let mut unplaced: Vec<Basic> = Vec::new();
        for (_, _, factors) in self.nodes(f, x, full)? {
            for (g, e) in factors {
                if !full && !even(&e) {
                    continue;
                }
                match self.zeros(&g, x, a, b)? {
                    None => {
                        if !unplaced.contains(&g) {
                            unplaced.push(g)
                        }
                    }
                    Some(z) => points.extend(z),
                }
            }
        }
        let placed: Vec<Basic> = if points.is_empty() {
            Vec::new()
        } else {
            match self.points_in(&distinct(&points), a, b)? {
                None => return Ok(None),
                Some(p) => p,
            }
        };
        Ok(Some((placed, unplaced, kinks_undecided)))
    }

    fn split_at(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        points: &[Basic],
        depth: usize,
        branch: bool,
        unplaced: &[Basic],
        full: bool,
    ) -> OptCV {
        let mut bounds: Vec<Basic> = vec![a.clone()];
        bounds.extend(points.iter().cloned());
        bounds.push(b.clone());
        let mut total: ConditionalValue = zero_cv();
        for w in bounds.windows(2) {
            let (lo, hi) = (&w[0], &w[1]);
            let piece: Basic = if branch {
                match self.branch(f, x, lo, hi, unplaced, full)? {
                    None => return Ok(None),
                    Some(p) => p,
                }
            } else {
                f.clone()
            };
            let Some(found) = self.integrate(&piece, x, lo, hi, depth + 1, !full, full)? else {
                return Ok(None);
            };
            if has_bad(&found.value) {
                return Ok(None);
            }
            total = total.add(&found)?;
        }
        if has_bad(&total.value) {
            return Ok(None);
        }
        Ok(Some(total))
    }

    fn split_signs(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        depth: usize,
        full: bool,
    ) -> PolyResult<(bool, Option<ConditionalValue>)> {
        if !has_kinks(f) && self.nodes(f, x, full)?.is_empty() {
            return Ok((false, None));
        }
        let Some((points, unplaced, undecided)) = self.breakpoints(f, x, a, b, full)? else {
            return Ok((false, None));
        };
        if !undecided.is_empty() {
            if !self.cases || full {
                return Ok((false, None));
            }
            let marked: Vec<(Basic, bool)> = undecided.into_iter().map(|p| (p, false)).collect();
            return Ok((true, self.cases_of(f, x, a, b, &marked, depth)?));
        }
        if points.is_empty() {
            let Some(piece) = self.branch(f, x, a, b, &unplaced, full)? else {
                return Ok((false, None));
            };
            if piece == *f {
                return Ok((false, None));
            }
            return Ok((false, self.integrate(&piece, x, a, b, depth + 1, !full, full)?));
        }
        Ok((false, self.split_at(f, x, a, b, &points, depth, true, &unplaced, full)?))
    }

    /// `f` split by the signs of its factors when none changes sign inside
    /// (`_signed_form`).
    pub fn signed_form(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> R {
        if self.nodes(f, x, true)?.is_empty() {
            return Ok(f.clone());
        }
        let Some((points, unplaced, undecided)) = self.breakpoints(f, x, a, b, true)? else {
            return Ok(f.clone());
        };
        if !points.is_empty() || !undecided.is_empty() {
            return Ok(f.clone());
        }
        Ok(self.branch(f, x, a, b, &unplaced, true)?.unwrap_or_else(|| f.clone()))
    }

    fn split_branches(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        depth: usize,
    ) -> PolyResult<(bool, Option<ConditionalValue>)> {
        self.split_signs(f, x, a, b, depth, false)
    }

    fn split_powers(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        Ok(self.split_signs(f, x, a, b, depth, true)?.1)
    }

    fn split_singularities(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        depth: usize,
    ) -> PolyResult<(bool, Option<ConditionalValue>)> {
        if has_kinks(f) {
            return Ok((false, None));
        }
        let Some((points, undecided)) = self.singular_placement(f, x, a, b)? else {
            return Ok((false, None));
        };
        if !undecided.is_empty() {
            if !self.cases {
                return Ok((false, None));
            }
            let marked: Vec<(Basic, bool)> = undecided.into_iter().map(|p| (p, true)).collect();
            return Ok((true, self.cases_of(f, x, a, b, &marked, depth)?));
        }
        if points.is_empty() {
            return Ok((false, None));
        }
        Ok((true, self.split_at(f, x, a, b, &points, depth, false, &[], false)?))
    }

    // -- the cases of the parameters

    /// The integral for every position of the points against the
    /// endpoints which the assumptions do not settle (`_cases`).
    fn cases_of(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, points: &[(Basic, bool)], depth: usize) -> OptCV {
        if points.len() > MAX_CASE_POINTS {
            return Ok(None);
        }
        let alternatives: Vec<Vec<(Basic, i32)>> = points
            .iter()
            .map(|(p, singular)| positions(p, a, b, *singular))
            .collect::<PolyResult<_>>()?;
        // the product of the alternatives, the first varying slowest
        let mut combinations: Vec<Vec<(Basic, i32)>> = vec![Vec::new()];
        for options in &alternatives {
            let mut next: Vec<Vec<(Basic, i32)>> = Vec::new();
            for c in &combinations {
                for o in options {
                    let mut d = c.clone();
                    d.push(o.clone());
                    next.push(d);
                }
            }
            combinations = next;
        }
        let mut cases: Vec<(Basic, Vec<Basic>, bool)> = Vec::new();
        for combination in &combinations {
            let mut conjuncts: Vec<Basic> = Vec::new();
            for (c, _) in combination {
                if matches!(c.head(), Head::And) {
                    conjuncts.extend(c.args().iter().cloned());
                } else {
                    conjuncts.push(c.clone());
                }
            }
            let squared: Vec<Basic> = conjuncts.iter().map(|c| self.squared(c)).collect::<PolyResult<_>>()?;
            let Some(condition) = self.reduced(&squared)? else {
                continue;
            };
            let inside: Vec<Basic> = points
                .iter()
                .zip(combination)
                .filter(|(_, (_, position))| *position != 0)
                .map(|((p, _), _)| p.clone())
                .collect();
            cases.push((condition, inside, combination.iter().any(|(_, p)| *p < 0)));
        }
        if cases.is_empty() {
            return Ok(None);
        }
        let known: Option<Basic> = crate::integrals::antiderivative::antiderivative(f, x, &self.assumptions, false)?;
        let budget: Option<f64> = crate::settings::settings().timeout.map(|t| t / 4.0);
        let mut results: Vec<(Basic, Basic)> = Vec::new();
        for (condition, inside, reversed) in &cases {
            let mut ordered: Vec<(i64, Basic)> = Vec::new();
            for (_, value) in &results {
                ordered.push((ops(value)?, value.clone()));
            }
            ordered.sort_by_key(|item| item.0);
            let candidates: Vec<Basic> = ordered.into_iter().map(|(_, v)| v).collect();
            let assumptions: Vec<Basic> = with(&self.assumptions, std::slice::from_ref(condition));
            let found: Option<ConditionalValue> = crate::timeout::attempt_within(budget, || {
                self.case_value(
                    f,
                    x,
                    a,
                    b,
                    inside,
                    *reversed,
                    &assumptions,
                    depth,
                    known.as_ref(),
                    &candidates,
                )
            })?
            .flatten();
            let Some(found) = found else {
                continue;
            };
            if has_infinity(&found.value) {
                continue;
            }
            let mut case: Basic =
                canonical_relations(&crate::core::logic::and_(&[condition.clone(), found.condition.clone()]))?;
            if !is_true_b(&found.condition) && matches!(case.head(), Head::And) {
                let mut parts: Vec<(i64, Basic)> = Vec::new();
                for part in case.args() {
                    parts.push((ops(part)?, part.clone()));
                }
                // sorted by the number of operations, the largest first (stable)
                parts.sort_by_key(|p| std::cmp::Reverse(p.0));
                let parts: Vec<Basic> = parts.into_iter().map(|(_, p)| p).collect();
                let Some(reduced) = self.reduced(&parts)? else {
                    continue;
                };
                case = canonical_relations(&reduced)?;
            }
            case = strict_where_undefined(&case, &found.value)?;
            if crate::settings::settings().numerical_checks
                && verify_numerically(
                    &found.value,
                    f,
                    x,
                    a,
                    b,
                    &with(&self.assumptions, std::slice::from_ref(&case)),
                    2,
                    inside,
                )? == Some(false)
            {
                continue;
            }
            results.push((case, found.value));
        }
        let mut joined: Vec<(Basic, Basic)> = Vec::new();
        for (condition, value) in self.joined(&results, f, x, a, b)? {
            joined.push((merged_disjunction(&condition)?, value));
        }
        if joined.is_empty() {
            return Ok(None);
        }
        if joined.len() == 1 {
            let (c, v) = joined.remove(0);
            return Ok(Some(ConditionalValue::new(v, c)));
        }
        let pieces: Vec<Basic> = joined
            .iter()
            .map(|(c, v)| Basic::new(Head::ExprCondPair, vec![v.clone(), c.clone()]))
            .collect();
        let conditions: Vec<Basic> = joined.iter().map(|(c, _)| c.clone()).collect();
        Ok(Some(ConditionalValue::new(
            Basic::new(Head::Piecewise, pieces),
            crate::core::logic::or_(&conditions),
        )))
    }

    /// A relation `k*sqrt(u) > e` written without the root (`_squared`).
    fn squared(&self, condition: &Basic) -> R {
        use crate::core::basic::RelOp;
        use crate::core::relational as rl;
        let Some(op) = rel_op(condition).filter(|op| matches!(op, RelOp::Lt | RelOp::Le | RelOp::Gt | RelOp::Ge))
        else {
            return Ok(condition.clone());
        };
        let (lhs, rhs) = (&condition.args()[0], &condition.args()[1]);
        for (side, other, flipped) in [(lhs, rhs, false), (rhs, lhs, true)] {
            let (k, root) = crate::core::expr::as_coeff_mul(side);
            if !(root.is_pow() && root.args()[1] == Basic::half() && k.as_rational().is_some_and(|r| *r != Rat::ZERO))
                || other.has(&root)
            {
                continue;
            }
            let bound: Basic = d2(other, &k)?;
            let mut greater: bool = matches!(op, RelOp::Gt | RelOp::Ge) != flipped;
            if k.as_rational().is_some_and(|r| *r < Rat::ZERO) {
                greater = !greater;
            }
            let strict: bool = matches!(op, RelOp::Gt | RelOp::Lt);
            let u: Basic = root.args()[0].clone();
            if self.holds(rl::lt(&bound, &Basic::zero()))? == Some(true) {
                return Ok(crate::core::logic::boolean(greater));
            }
            if self.holds(rl::ge(&bound, &Basic::zero()))? != Some(true) {
                return Ok(condition.clone());
            }
            let square: Basic = p2(&bound, &int(2))?;
            return match (greater, strict) {
                (true, true) => rl::gt(&u, &square),
                (true, false) => rl::ge(&u, &square),
                (false, true) => rl::lt(&u, &square),
                (false, false) => rl::le(&u, &square),
            };
        }
        Ok(condition.clone())
    }

    /// The conjunction without the conditions which follow from the
    /// others, `None` when unsatisfiable (`_reduced`).
    fn reduced(&self, conjuncts: &[Basic]) -> PolyResult<Option<Basic>> {
        use crate::assumptions::sat::{satisfiable, Sat};
        let mut kept: Vec<Basic> = conjuncts.to_vec();
        for part in &kept {
            if self.ask(part)? == Some(false) {
                return Ok(None);
            }
        }
        if matches!(
            satisfiable(&crate::core::logic::and_(&kept), &self.assumptions, None)?,
            Sat::Unsatisfiable
        ) {
            return Ok(None);
        }
        for part in kept.clone() {
            let others: Vec<Basic> = kept.iter().filter(|q| **q != part).cloned().collect();
            let mut related: BTreeSet<Basic> = BTreeSet::new();
            for q in others.iter().chain(self.assumptions.iter()) {
                related.extend(q.free_symbols());
            }
            if !part.free_symbols().iter().any(|s| related.contains(s)) {
                continue;
            }
            if matches!(
                satisfiable(
                    &crate::core::logic::not_(&part),
                    &with(&self.assumptions, &others),
                    None
                )?,
                Sat::Unsatisfiable
            ) {
                kept.retain(|q| *q != part);
            }
        }
        Ok(Some(crate::core::logic::and_(&kept)))
    }

    /// The integral in one case, under its assumptions (`_case_value`).
    #[allow(clippy::too_many_arguments)]
    fn case_value(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        inside: &[Basic],
        reversed: bool,
        assumptions: &[Basic],
        depth: usize,
        known: Option<&Basic>,
        candidates: &[Basic],
    ) -> OptCV {
        use crate::integrals::antiderivative::{one_sided_limit, real_logarithms};
        let mut f: Basic = f.clone();
        if !inside.is_empty() && rational_function(&f, std::slice::from_ref(x)) {
            let cancelled: Basic = crate::polys::polytools::cancel(&f, &[])?;
            let denominator: Basic = crate::core::expr::as_numer_denom(&cancelled)?.1;
            for s in inside {
                if crate::polys::polytools::cancel(&subs(&denominator, x, s)?, &[])? == Basic::zero() {
                    return Ok(None);
                }
            }
        }
        let integrator: Integrator = Integrator::new(
            assumptions,
            self.parametric,
            self.principal_value,
            self.finite_part,
            self.regularize,
            false,
        );
        let (lower, upper) = if reversed { (b, a) } else { (a, b) };
        let ordered: Option<Placement> = integrator.place(inside, lower, upper)?;
        if let (Some((points, _)), Some(known)) = (&ordered, known) {
            if !points.is_empty() {
                let real: Basic = real_logarithms(known, x)?;
                for s in points {
                    for side in ["-", "+"] {
                        if let Some(v) = one_sided_limit(&real, x, s, side, assumptions)? {
                            if v == oo() || v == neg_oo() {
                                return Ok(None);
                            }
                        }
                    }
                }
            }
        }
        for candidate in candidates {
            if crate::settings::settings().numerical_checks
                && verify_numerically(candidate, &f, x, a, b, assumptions, 3, inside)? == Some(true)
            {
                return Ok(Some(ConditionalValue::new(candidate.clone(), true_b())));
            }
        }
        if inside.is_empty() && has_kinks(&f) {
            if let Some((i, u)) = integrator.kink_placement(&f, x, lower, upper)? {
                if i.is_empty() && u.is_empty() {
                    if let Some(branch) = integrator.branch(&f, x, lower, upper, &[], false)? {
                        f = branch;
                    }
                }
            }
        }
        if let Some(found) = integrator.last(&f, x, a, b, depth)? {
            if !has_head(&found.value, &|h| matches!(h, Head::Piecewise)) {
                return Ok(Some(found));
            }
        }
        if let (Some((points, undecided)), Some(known)) = (&ordered, known) {
            if undecided.is_empty() {
                let mut bounds: Vec<Basic> = vec![lower.clone()];
                bounds.extend(points.iter().cloned());
                bounds.push(upper.clone());
                let mut total: Option<ConditionalValue> = Some(zero_cv());
                for w in bounds.windows(2) {
                    let piece: Option<ConditionalValue> =
                        integrator.antiderivative_method(&f, x, &w[0], &w[1], depth, false, Some(known))?;
                    if let Some(p) = &piece {
                        if has_infinity(&p.value) {
                            return Ok(None);
                        }
                    }
                    total = match (total, piece) {
                        (Some(t), Some(p)) => Some(t.add(&p)?),
                        _ => None,
                    };
                }
                if let Some(total) = total {
                    return Ok(Some(if reversed { total.scaled(&int(-1))? } else { total }));
                }
            }
        }
        integrator.integrate(&f, x, a, b, depth, true, false)
    }

    /// The cases with equal values joined (`_joined`).
    fn joined(
        &self,
        results: &[(Basic, Basic)],
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
    ) -> PolyResult<Vec<(Basic, Basic)>> {
        let agrees = |value: &Basic, condition: &Basic| -> PolyResult<bool> {
            Ok(crate::settings::settings().numerical_checks
                && verify_numerically(
                    value,
                    f,
                    x,
                    a,
                    b,
                    &with(&self.assumptions, std::slice::from_ref(condition)),
                    3,
                    &[],
                )? == Some(true))
        };
        let mut sorted: Vec<(i64, String, Basic, Basic)> = Vec::new();
        for (c, v) in results {
            sorted.push((ops(v)?, v.to_string(), c.clone(), v.clone()));
        }
        sorted.sort_by(|p, q| (p.0, &p.1).cmp(&(q.0, &q.1)));
        let mut joined: Vec<(Vec<Basic>, Basic)> = Vec::new();
        for (_, _, condition, value) in sorted {
            let mut placed: bool = false;
            for entry in joined.iter_mut() {
                if entry.1 == value || agrees(&entry.1, &condition)? {
                    entry.0.push(condition.clone());
                    placed = true;
                    break;
                }
            }
            if !placed {
                joined.push((vec![condition], value));
            }
        }
        let mut merged: Vec<(Vec<Basic>, Basic)> = Vec::new();
        for (conditions, value) in joined.into_iter().rev() {
            let mut placed: bool = false;
            for entry in merged.iter_mut() {
                let mut all: bool = true;
                for c in &conditions {
                    if !agrees(&entry.1, c)? {
                        all = false;
                        break;
                    }
                }
                if all {
                    entry.0.extend(conditions.iter().cloned());
                    placed = true;
                    break;
                }
            }
            if !placed {
                merged.push((conditions, value));
            }
        }
        Ok(merged
            .into_iter()
            .rev()
            .map(|(cs, v)| (crate::core::logic::or_(&cs), v))
            .collect())
    }

    // -- the methods on a plain piece

    fn finish(&self, found: Option<ConditionalValue>) -> OptCV {
        let Some(found) = found else {
            return Ok(None);
        };
        if has_bad(&found.value) {
            return Ok(None);
        }
        let Some(condition) = crate::integrals::conditions::decide(&found.condition, &self.assumptions)? else {
            return Ok(None);
        };
        Ok(Some(ConditionalValue::new(found.value, condition)))
    }

    fn mellin_on(&self, g: &Basic, t: &Basic, cutoff: Option<&str>) -> OptCV {
        for candidate in forms(g, t, &self.assumptions)? {
            if let Some(found) =
                crate::integrals::marichev::mellin_integrate(&candidate, t, &self.assumptions, cutoff, self.regularize)?
            {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    // -- the strategies

    fn canonical(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        let t: Basic = fresh("t", &[(Predicate::positive, true)]);
        let (zero, one) = (Basic::zero(), Basic::one());
        let g = || subs(f, x, &t);
        if *a == zero && *b == oo() {
            let mut found: Option<ConditionalValue> = self.mellin_on(&g()?, &t, None)?;
            if found.is_none() && has_any_function(f, &[Function::Exp]) {
                found = self.exponential_substitution(f, x, a, b)?;
            }
            if found.is_none() {
                found = self.series_mellin(&g()?, &t)?;
            }
            return self.finish(found);
        }
        if *a == zero && *b == one {
            return self.finish(self.mellin_on(&g()?, &t, Some("lower"))?);
        }
        if *a == one && *b == oo() {
            return self.finish(self.mellin_on(&g()?, &t, Some("upper"))?);
        }
        if *a == neg_oo() && *b == oo() && has_any_function(f, &[Function::Exp]) {
            return self.finish(self.exponential_substitution(f, x, a, b)?);
        }
        Ok(None)
    }

    fn mapped(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        let t: Basic = fresh("t", &[(Predicate::positive, true)]);
        let zero: Basic = Basic::zero();
        if *a == neg_oo() && *b == oo() {
            let Some(right) = self.integrate(f, x, &zero, &oo(), depth + 1, false, true)? else {
                return Ok(None);
            };
            let Some(left) = self.integrate(f, x, &neg_oo(), &zero, depth + 1, false, true)? else {
                return Ok(None);
            };
            let total: ConditionalValue = left.add(&right)?;
            let mut symbols: BTreeSet<Basic> = f.free_symbols();
            symbols.extend(total.condition.free_symbols());
            symbols.remove(x);
            if holds_nowhere(&total.condition, &sorted_symbols(symbols), &self.assumptions)? {
                return Ok(None);
            }
            return Ok(Some(total));
        }
        if *a == neg_oo() {
            return self.integrate(&subs(f, x, &s2(b, &t)?)?, &t, &zero, &oo(), depth + 1, false, true);
        }
        if *b == oo() {
            if *a == zero {
                return Ok(None);
            }
            return self.integrate(&shift(f, x, &t, a)?, &t, &zero, &oo(), depth + 1, false, true);
        }
        if *a == zero {
            if *b == Basic::one() {
                return Ok(None);
            }
            if self.holds(crate::core::relational::gt(b, &zero))? != Some(true) {
                return Ok(None);
            }
            let g: Basic = m2(&subs(f, x, &m2(b, &t)?)?, b)?;
            return self.integrate(&g, &t, &zero, &Basic::one(), depth + 1, false, true);
        }
        let length: Basic = s2(b, a)?;
        let g: Basic = m2(&subs(f, x, &a2(a, &m2(&length, &t)?)?)?, &length)?;
        if let Some(found) = self.integrate(&g, &t, &zero, &Basic::one(), depth + 1, false, true)? {
            return Ok(Some(found));
        }
        let g: Basic = m2(&subs(f, x, &s2(b, &m2(&length, &t)?)?)?, &length)?;
        self.integrate(&g, &t, &zero, &Basic::one(), depth + 1, false, true)
    }

    fn series_mellin(&self, g: &Basic, t: &Basic) -> OptCV {
        let s: Basic = fresh("s", &[]);
        let Some(found) = crate::integrals::brackets::mellin_transform_series(g, t, &s, &[])? else {
            return Ok(None);
        };
        let transform: Basic = found.transform.clone();
        let mut value: Basic = subs(&transform, &s, &Basic::one())?;
        if has_infinity(&value) {
            let limited: Option<Basic> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
                crate::core::limits::limit(&transform, &s, &Basic::one(), "+")
            })?;
            match limited {
                Some(l) if !has_infinity(&l) => value = l,
                _ => return Ok(None),
            }
        }
        let (lower, upper) = &found.strip;
        let mut parts: Vec<Basic> = Vec::new();
        if *lower != neg_oo() {
            parts.push(crate::core::relational::lt(lower, &Basic::one())?);
        }
        if *upper != oo() {
            parts.push(crate::core::relational::gt(upper, &Basic::one())?);
        }
        parts.push(found.condition.clone());
        Ok(Some(ConditionalValue::new(value, crate::core::logic::and_(&parts))))
    }

    fn exponential_substitution(&self, f: &Basic, x: &Basic, a: &Basic, _b: &Basic) -> OptCV {
        let Some(c) = exponential_generator(f, x, *a == Basic::zero())? else {
            return Ok(None);
        };
        let u: Basic = fresh("u", &[(Predicate::positive, true)]);
        let log_u: Basic = log(&u)?;
        let at: Basic = d2(&subs(f, x, &d2(&log_u, &c)?)?, &m2(&c, &u)?)?;
        let g: Basic = denest(&powers_of_logarithms(&at, &u)?, &self.assumptions)?;
        if g.has(&log_u) && !log_powers_only(&g, &u)? {
            return Ok(None);
        }
        let negative: bool = yes(&c, Predicate::negative);
        let sign: Basic = if negative { int(-1) } else { Basic::one() };
        let cutoff: Option<&str> = if *a == Basic::zero() {
            Some(if negative { "lower" } else { "upper" })
        } else {
            None
        };
        let found: Option<ConditionalValue> = self.mellin_on(&g, &u, cutoff)?;
        match found {
            None => Ok(None),
            Some(v) => Ok(Some(v.scaled(&sign)?)),
        }
    }

    fn inversion(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        let (zero, one) = (Basic::zero(), Basic::one());
        if !((*a == one && *b == oo()) || (*a == zero && *b == one)) {
            return Ok(None);
        }
        let u: Basic = fresh("u", &[(Predicate::positive, true)]);
        let g: Basic = denest(&d2(&subs(f, x, &d2(&one, &u)?)?, &p2(&u, &int(2))?)?, &self.assumptions)?;
        let cutoff: &str = if *b == oo() { "lower" } else { "upper" };
        self.finish(self.mellin_on(&g, &u, Some(cutoff))?)
    }

    fn trigonometric(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if !has_trigonometric(f) || infinite(a) || infinite(b) {
            return Ok(None);
        }
        let (Some(ka), Some(kb)) = (quarter(a)?, quarter(b)?) else {
            return Ok(None);
        };
        let half_pi: Basic = d2(&Basic::pi(), &int(2))?;
        let zero: Basic = Basic::zero();
        if kb - ka > 1 {
            let t: Basic = fresh("t", &[]);
            let mut total: ConditionalValue = zero_cv();
            for k in ka..kb {
                let shifted: Basic = subs(f, x, &a2(&t, &m2(&int(k), &half_pi)?)?)?;
                let Some(piece) = self.integrate(&shifted, &t, &zero, &half_pi, depth + 1, false, true)? else {
                    return Ok(None);
                };
                total = total.add(&piece)?;
            }
            return Ok(Some(total));
        }
        if ka != 0 {
            let t: Basic = fresh("t", &[]);
            let shifted: Basic = subs(f, x, &a2(&t, &m2(&int(ka), &half_pi)?)?)?;
            return self.integrate(&shifted, &t, &zero, &half_pi, depth + 1, false, true);
        }
        let u: Basic = fresh("u", &[(Predicate::positive, true)]);
        let f: Basic = self.signed_form(f, x, &zero, &half_pi)?;
        let jacobian: Basic = m2(
            &m2(&int(2), &p2(&u, &Basic::half())?)?,
            &p2(&s2(&Basic::one(), &u)?, &Basic::half())?,
        )?;
        let g: Basic = d2(&arcsine_substitution(&f, x, &u)?, &jacobian)?;
        let g: Basic = denest(&g, &self.assumptions)?;
        let g: Basic = self.signed_form(&g, &u, &zero, &Basic::one())?;
        if has_any_function(&g, &[Function::Asin]) {
            return Ok(None);
        }
        if let Some(found) = self.finish(self.mellin_on(&g, &u, Some("lower"))?)? {
            return Ok(Some(found));
        }
        self.integrate(&g, &u, &zero, &Basic::one(), depth + 1, false, true)
    }

    fn mean_value(&self, f: &Basic, _x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        if !has_trigonometric(f) || infinite(a) || infinite(b) {
            return Ok(None);
        }
        Err(crate::error::not_ported("integrals/periodic (mean_value_integral)"))
    }

    fn residues(&self, _f: &Basic, _x: &Basic, _a: &Basic, _b: &Basic, _depth: usize) -> OptCV {
        Err(crate::error::not_ported("integrals/residues (residue_integral)"))
    }

    fn real_parts(&self, condition: &Basic) -> R {
        let mut replacement: HashMap<Basic, Basic> = HashMap::new();
        for node in condition.atoms(Some(&|n: &Basic| {
            n.is_function(Function::Re) || n.is_function(Function::Im)
        })) {
            let p: &Basic = &node.args()[0];
            if matches!(p.head(), Head::Symbol(_) | Head::Dummy(..))
                && (yes(p, Predicate::extended_real)
                    || self.ask(&crate::assumptions::facts::element(p, &Basic::atom(Head::Reals))?)? == Some(true))
            {
                replacement.insert(
                    node.clone(),
                    if node.is_function(Function::Re) {
                        p.clone()
                    } else {
                        Basic::zero()
                    },
                );
            }
        }
        if replacement.is_empty() {
            return Ok(condition.clone());
        }
        crate::core::function::xreplace_evaluated(condition, &replacement)
    }

    fn radicals(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if a.has(x)
            || b.has(x)
            || crate::integrals::radicals::quadratic_radical_antiderivative(f, x, &[], true)?.is_none()
        {
            return Ok(None);
        }
        self.under_budget(|| self.antiderivative_method(f, x, a, b, depth, false, None), 4.0)
    }

    fn arctangent_pairs(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> R {
        use crate::core::relational as rl;
        if !has_any_function(f, &[Function::Atan]) {
            return Ok(f.clone());
        }
        let orientation: i64 = if self.holds(rl::ge(a, &Basic::zero()))? == Some(true) {
            1
        } else if self.holds(rl::le(b, &Basic::zero()))? == Some(true) {
            -1
        } else {
            return Ok(f.clone());
        };
        let signed: Basic = fresh(&x.to_string(), &[(Predicate::positive, true)]);
        let pairs_of = |node: &Basic| -> R {
            let terms: Vec<Basic> = node.args().to_vec();
            for (i, first) in terms.iter().enumerate() {
                if !first.is_function(Function::Atan) || !first.has(x) {
                    continue;
                }
                let u: &Basic = &first.args()[0];
                for (j, second) in terms.iter().enumerate() {
                    if j <= i || !second.is_function(Function::Atan) {
                        continue;
                    }
                    let v: &Basic = &second.args()[0];
                    if crate::polys::polytools::cancel(&s2(&m2(u, v)?, &Basic::one())?, &[])? != Basic::zero() {
                        continue;
                    }
                    let Some(positive) = is(&subs(u, x, &m2(&int(orientation), &signed)?)?, Predicate::positive) else {
                        continue;
                    };
                    let mut rest: Vec<Basic> = terms
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i && *k != j)
                        .map(|(_, t)| t.clone())
                        .collect();
                    let half_pi: Basic = d2(&Basic::pi(), &int(2))?;
                    rest.push(if positive { half_pi } else { ng(&half_pi)? });
                    return add_all(&rest);
                }
            }
            Ok(node.clone())
        };
        replace_where(
            f,
            &|n: &Basic| n.is_add() && has_any_function(n, &[Function::Atan]),
            &pairs_of,
        )
    }

    fn quick_antiderivative(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        use crate::integrals::antiderivative::{by_parts_shape, quick_shape};
        if a.has(x) || b.has(x) || !quick_shape(f, x)? {
            return Ok(None);
        }
        if (infinite(a) || infinite(b)) && !by_parts_shape(f, x)? {
            return Ok(None);
        }
        self.under_budget(|| self.generic_antiderivative(f, x, a, b, depth), 4.0)
    }

    fn generic_antiderivative(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if a.has(x) || b.has(x) {
            return Ok(None);
        }
        let Some(big_f) = crate::integrals::antiderivative::antiderivative(f, x, &self.assumptions, false)? else {
            return Ok(None);
        };
        let generic: Basic = crate::integrals::antiderivative::generic_antiderivative(&big_f, x)?;
        if generic == big_f {
            return self.antiderivative_method(f, x, a, b, depth, false, Some(&big_f));
        }
        let Some(found) = self.evaluated_antiderivative(&generic, f, x, a, b)? else {
            return Ok(None);
        };
        if found.value.has(&oo()) || found.value.has(&neg_oo()) {
            return Ok(None);
        }
        if crate::settings::settings().numerical_checks
            && verify_numerically(&found.value, f, x, a, b, &self.assumptions, 2, &[])? != Some(true)
        {
            return Ok(None);
        }
        self.finish(Some(found))
    }

    #[allow(clippy::too_many_arguments)]
    fn antiderivative_method(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        _depth: usize,
        late: bool,
        known: Option<&Basic>,
    ) -> OptCV {
        if a.has(x) || b.has(x) {
            return Ok(None);
        }
        let Some(found) = crate::integrals::antiderivative::antiderivative_integral(
            f,
            x,
            a,
            b,
            &self.assumptions,
            late,
            known,
            None,
        )?
        else {
            return Ok(None);
        };
        if found.value == oo() || found.value == neg_oo() {
            if crate::settings::settings().numerical_checks && quadrature_converges(f, x, a, b, &self.assumptions)? {
                return Ok(None);
            }
            return self.finish(Some(found));
        }
        if crate::settings::settings().numerical_checks
            && verify_numerically(&found.value, f, x, a, b, &self.assumptions, 2, &[])? != Some(true)
        {
            return Ok(None);
        }
        self.finish(Some(found))
    }

    fn radical_substitution(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if f.is_add() {
            return Ok(None);
        }
        let Some((q, n)) = radical_base(f, x)? else {
            return Ok(None);
        };
        let u: Basic = fresh("u", &[(Predicate::positive, true)]);
        let Some((x_u, lower, upper)) = self.radical_inverse(&q, n, x, &u, a, b)? else {
            return Ok(None);
        };
        let jacobian: Basic = crate::core::function::derivative(&x_u, &u)?;
        let g: Basic = denest(
            &crate::core::function::expand_log(&m2(&subs(f, x, &x_u)?, &jacobian)?, true, false, false)?,
            &self.assumptions,
        )?;
        if g.has(x) || radical_bases(&g, &u).len() >= radical_bases(f, x).len() {
            return Ok(None);
        }
        self.under_budget(|| self.integrate(&g, &u, &lower, &upper, depth + 1, false, true), 2.0)
    }

    fn under_budget(&self, compute: impl FnOnce() -> OptCV, share: f64) -> OptCV {
        let budget: Option<f64> = crate::settings::settings().timeout.map(|t| t / share);
        Ok(crate::timeout::attempt_within(budget, compute)?.flatten())
    }

    #[allow(clippy::type_complexity)]
    fn radical_inverse(
        &self,
        q: &Basic,
        n: i64,
        x: &Basic,
        u: &Basic,
        a: &Basic,
        b: &Basic,
    ) -> PolyResult<Option<(Basic, Basic, Basic)>> {
        use crate::core::relational as rl;
        let Some(cs) = coefficients(q, x)? else {
            return Ok(None);
        };
        let zero: Basic = Basic::zero();
        if cs.len() == 2 {
            let (p, r) = (cs[1].clone(), cs[0].clone());
            let mut ends: Vec<Basic> = Vec::new();
            for end in [a, b] {
                if infinite(end) {
                    if self.holds(rl::gt(&m2(&p, end)?, &zero))? != Some(true) {
                        return Ok(None);
                    }
                    ends.push(oo());
                    continue;
                }
                let value: Basic = subs(q, x, end)?;
                if self.holds(rl::ge(&value, &zero))? != Some(true) {
                    return Ok(None);
                }
                ends.push(simplest_root(&p2(&value, &rat(1, n))?)?);
            }
            let x_u: Basic = d2(&s2(&p2(u, &int(n))?, &r)?, &p)?;
            return Ok(Some((x_u, ends[0].clone(), ends[1].clone())));
        }
        if cs.len() != 3 || n != 2 || !(is_number(a) && is_number(b)) {
            return Ok(None);
        }
        let (big_a, big_b, big_c) = (cs[2].clone(), cs[1].clone(), cs[0].clone());
        let vertex: Basic = ng(&d2(&big_b, &m2(&int(2), &big_a)?)?)?;
        if !(self.holds(rl::ge(a, &vertex))? == Some(true) || self.holds(rl::le(b, &vertex))? == Some(true)) {
            return Ok(None);
        }
        let mut ends: Vec<Basic> = Vec::new();
        for end in [a, b] {
            if infinite(end) {
                if self.holds(rl::gt(&big_a, &zero))? != Some(true) {
                    return Ok(None);
                }
                ends.push(oo());
                continue;
            }
            let value: Basic = subs(q, x, end)?;
            if self.holds(rl::ge(&value, &zero))? != Some(true) {
                return Ok(None);
            }
            ends.push(p2(&value, &Basic::half())?);
        }
        let sample: Basic = self.sample(a, b)?;
        for sign in [Basic::one(), int(-1)] {
            let discriminant: Basic = s2(
                &p2(&big_b, &int(2))?,
                &m2(&m2(&int(4), &big_a)?, &s2(&big_c, &p2(u, &int(2))?)?)?,
            )?;
            let x_u: Basic = d2(
                &a2(&ng(&big_b)?, &m2(&sign, &p2(&discriminant, &Basic::half())?)?)?,
                &m2(&int(2), &big_a)?,
            )?;
            let back: Basic = subs(&x_u, u, &p2(&subs(q, x, &sample)?, &Basic::half())?)?;
            if self.holds(rl::eq(&back, &sample))? == Some(true) {
                return Ok(Some((x_u, ends[0].clone(), ends[1].clone())));
            }
        }
        Ok(None)
    }

    fn completed_square(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if f.is_add() || (!infinite(a) && !infinite(b)) {
            return Ok(None);
        }
        let Some((q, _)) = radical_base(f, x)? else {
            return Ok(None);
        };
        let Some(cs) = coefficients(&q, x)? else {
            return Ok(None);
        };
        if cs.len() != 3 {
            return Ok(None);
        }
        let (big_a, big_b) = (cs[2].clone(), cs[1].clone());
        if big_b == Basic::zero() {
            return Ok(None);
        }
        let c: Basic = d2(&big_b, &m2(&int(2), &big_a)?)?;
        let u: Basic = fresh("u", &[(Predicate::real, true)]);
        let g: Basic = expand(&subs(f, x, &s2(&u, &c)?)?)?;
        let g: Basic = replace_where(&g, &|n: &Basic| n.is_pow() && n.has(&u), &|n: &Basic| {
            p2(&expand(&n.args()[0])?, &n.args()[1])
        })?;
        let lower: Basic = if !infinite(a) { a2(a, &c)? } else { a.clone() };
        let upper: Basic = if !infinite(b) { a2(b, &c)? } else { b.clone() };
        self.under_budget(|| self.integrate(&g, &u, &lower, &upper, depth + 1, false, true), 2.0)
    }

    fn tangent_substitution(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        use crate::core::relational as rl;
        if f.is_add() || !has_trigonometric(f) || infinite(a) || infinite(b) {
            return Ok(None);
        }
        let trig = |n: &Basic| matches!(n.head(), Head::Function(g) if TRIGONOMETRIC.contains(g));
        let mut placeholders: HashMap<Basic, Basic> = HashMap::new();
        let mut names: Vec<Basic> = vec![x.clone()];
        for node in f.atoms(Some(&trig)) {
            if node.has(x) {
                let w: Basic = fresh("w", &[]);
                names.push(w.clone());
                placeholders.insert(node.clone(), w);
            }
        }
        let replaced: Basic = crate::core::function::xreplace_evaluated(f, &placeholders)?;
        if rational_function(&replaced, &names) {
            return Ok(None);
        }
        let mut g: Basic = f.clone();
        let mut needs_expansion: bool = false;
        for node in f.atoms(Some(&trig)) {
            if node.has(x) && crate::integrals::mellin::monomial(&node.args()[0], x)?.is_none() {
                needs_expansion = true;
            }
        }
        if needs_expansion {
            let mut hints = crate::core::function::Hints::DEFAULT;
            hints.mul = false;
            hints.multinomial = false;
            hints.power_base = false;
            hints.power_exp = false;
            hints.log = false;
            hints.trig = true;
            g = crate::core::function::expand_with_hints(f, Some(&hints), None)?;
        }
        let mut k: Option<Basic> = None;
        for node in g.atoms(Some(&trig)) {
            if !node.has(x) {
                continue;
            }
            let Some((c, e)) = crate::integrals::mellin::monomial(&node.args()[0], x)? else {
                return Ok(None);
            };
            if e != Basic::one() || k.as_ref().is_some_and(|k| *k != c) {
                return Ok(None);
            }
            k = Some(c);
        }
        let Some(frequency) = k else {
            return Ok(None);
        };
        if self.holds(rl::gt(&frequency, &Basic::zero()))? != Some(true) {
            return Ok(None);
        }
        for half in [false, true] {
            let width: Basic = if half { Basic::pi() } else { d2(&Basic::pi(), &int(2))? };
            if self.holds(rl::ge(&m2(&frequency, a)?, &ng(&width)?))? != Some(true)
                || self.holds(rl::le(&m2(&frequency, b)?, &width))? != Some(true)
            {
                continue;
            }
            let t: Basic = if self.holds(rl::ge(a, &Basic::zero()))? == Some(true) {
                fresh("t", &[(Predicate::positive, true)])
            } else {
                fresh("t", &[(Predicate::real, true)])
            };
            let one: Basic = Basic::one();
            let t2: Basic = p2(&t, &int(2))?;
            let (sine, cosine, jacobian) = if half {
                (
                    d2(&m2(&int(2), &t)?, &a2(&one, &t2)?)?,
                    d2(&s2(&one, &t2)?, &a2(&one, &t2)?)?,
                    d2(&int(2), &m2(&frequency, &a2(&one, &t2)?)?)?,
                )
            } else {
                let root: Basic = p2(&a2(&one, &t2)?, &Basic::half())?;
                (
                    d2(&t, &root)?,
                    d2(&one, &root)?,
                    d2(&one, &m2(&frequency, &a2(&one, &t2)?)?)?,
                )
            };
            let mut replacement: HashMap<Basic, Basic> = HashMap::new();
            for node in g.atoms(Some(&trig)) {
                if !node.has(x) {
                    continue;
                }
                let Head::Function(fun) = node.head() else {
                    unreachable!()
                };
                let value: Basic = match fun {
                    Function::Sin => sine.clone(),
                    Function::Cos => cosine.clone(),
                    Function::Tan => d2(&sine, &cosine)?,
                    Function::Cot => d2(&cosine, &sine)?,
                    Function::Sec => d2(&one, &cosine)?,
                    Function::Csc => d2(&one, &sine)?,
                    _ => return Ok(None),
                };
                replacement.insert(node.clone(), value);
            }
            let at = |e: &Basic| -> R {
                if half {
                    tangent_of(&d2(&m2(&frequency, e)?, &int(2))?)
                } else {
                    tangent_of(&m2(&frequency, e)?)
                }
            };
            let (lower, upper) = (at(a)?, at(b)?);
            let h: Basic = m2(&crate::core::function::xreplace_evaluated(&g, &replacement)?, &jacobian)?;
            let h: Basic = replace_where(
                &h,
                &|n: &Basic| n.is_pow() && rational_of(&n.args()[1]).is_some_and(|(_, q)| q != 1) && n.has(&t),
                &|n: &Basic| {
                    p2(
                        &crate::polys::polytools::factor(
                            &crate::polys::rationaltools::together(&n.args()[0], false, true)?,
                            &[],
                            false,
                            true,
                        )?,
                        &n.args()[1],
                    )
                },
            )?;
            let h: Basic = denest(&h, &self.assumptions)?;
            let h: Basic = flattened_powers(&h, &t, &lower, &upper)?;
            if h.has(x) || !algebraic(&h, &t)? {
                continue;
            }
            if let Some(value) =
                self.under_budget(|| self.integrate(&h, &t, &lower, &upper, depth + 1, false, true), 2.0)?
            {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn asymptotic_divergence(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> OptCV {
        if has_head(f, &|h| matches!(h, Head::Piecewise))
            || has_any_function(f, &[Function::Heaviside, Function::Abs, Function::Sign])
            || self.regularize
            || self.finite_part
        {
            return Ok(None);
        }
        let mut verdicts: Vec<Basic> = Vec::new();
        for (end, side) in [(a, "+"), (b, "-")] {
            if infinite(end) || !is_number(end) || finite_at(f, x, end)? {
                continue;
            }
            if let Some(verdict) = self.divergence_at(f, x, end, side)? {
                verdicts.push(verdict);
            }
        }
        if verdicts.is_empty() || verdicts.iter().any(|v| *v != verdicts[0]) {
            return Ok(None);
        }
        let Some(singular) =
            crate::timeout::attempt_within(crate::settings::settings().timeout.map(|t| t / 8.0), || {
                singular_points(f, x, a, b)
            })?
            .flatten()
        else {
            return Ok(None);
        };
        match self.place(&singular, a, b)? {
            Some((inside, undecided)) if inside.is_empty() && undecided.is_empty() => {}
            _ => return Ok(None),
        }
        if crate::settings::settings().numerical_checks && quadrature_converges(f, x, a, b, &self.assumptions)? {
            return Ok(None);
        }
        Ok(Some(ConditionalValue::new(verdicts.remove(0), true_b())))
    }

    fn divergence_at(&self, f: &Basic, x: &Basic, end: &Basic, side: &str) -> PolyResult<Option<Basic>> {
        use crate::core::relational as rl;
        let h: Basic = fresh("h", &[(Predicate::positive, true)]);
        let point: Basic = if side == "+" { a2(end, &h)? } else { s2(end, &h)? };
        let g: Basic = subs(f, x, &point)?;
        let budget: Option<f64> = crate::settings::settings().timeout.map(|t| t / 8.0);
        let found =
            crate::timeout::attempt_within(budget, || crate::core::series::leadterm(&g, &h, None, &Basic::zero()));
        let (coefficient, exponent) = match found {
            Ok(Some(found)) => found,
            Ok(None) => return Ok(None),
            Err(
                crate::error::PolyError::Value(_)
                | crate::error::PolyError::NotImplemented(_)
                | crate::error::PolyError::Pole(_)
                | crate::error::PolyError::Type(_),
            ) => return Ok(None),
            Err(e) => return Err(e),
        };
        if coefficient.has(&h)
            || !is_number(&exponent)
            || is(&exponent, Predicate::extended_real) != Some(true)
            || !is_true_b(&rl::le(&exponent, &int(-1))?)
        {
            return Ok(None);
        }
        if exponent == int(-1) && has_any_function(&coefficient, &[Function::Log]) {
            return Ok(None);
        }
        if self.holds(rl::gt(&coefficient, &Basic::zero()))? == Some(true) {
            return Ok(Some(oo()));
        }
        if self.holds(rl::lt(&coefficient, &Basic::zero()))? == Some(true) {
            return Ok(Some(neg_oo()));
        }
        Ok(None)
    }

    fn binomial_differential(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        use crate::integrals::algebraic::{binomial_exponents, chebyshev_elementary};
        if f.free_symbols() != std::iter::once(x.clone()).collect::<BTreeSet<Basic>>()
            || a.has(x)
            || b.has(x)
            || !yes(a, Predicate::finite)
            || !yes(b, Predicate::finite)
        {
            return Ok(None);
        }
        let radical: bool = f
            .atoms(Some(&|n: &Basic| n.is_pow()))
            .iter()
            .any(|n| n.args()[1].as_rational().is_some_and(|r| !r.is_int()) && n.args()[0].has(x));
        if !radical {
            return Ok(None);
        }
        let Some(binomial) = binomial_exponents(f, x)? else {
            return Ok(None);
        };
        if chebyshev_elementary(&binomial) {
            return Ok(None);
        }
        let ratio: Basic = ng(&d2(&binomial.b, &binomial.a)?)?;
        if is(&ratio, Predicate::extended_real) != Some(true) {
            return Ok(None);
        }
        let n_basic: Basic = Basic::from_rational(binomial.n.clone());
        let boundary: Basic = p2(
            &crate::functions::elementary::complexes::abs_(&ratio)?,
            &d2(&int(-1), &n_basic)?,
        )?;
        let n_integer: bool = binomial.n.is_int();
        let mut candidates: Vec<Basic> = Vec::new();
        for side in [1i64, -1] {
            if (side == 1 || n_integer) && yes(&m2(&ratio, &p2(&int(side), &n_basic)?)?, Predicate::extended_positive) {
                candidates.push(m2(&int(side), &boundary)?);
            }
        }
        let Some(cuts) = self.points_in(&distinct(&candidates), a, b)? else {
            return Ok(None);
        };
        for cheap in 0..3 {
            let found: Option<ConditionalValue> = match cheap {
                0 => self.table(f, x, a, b, depth)?,
                1 => self.canonical(f, x, a, b, depth)?,
                _ => self.elliptic(f, x, a, b, depth)?,
            };
            if found.is_some() {
                return Ok(found);
            }
        }
        let compute = || -> OptCV {
            let Some(known) =
                crate::integrals::indefinite::verified_antiderivative(f, x, &self.assumptions, Some(&["binomial"]))?
            else {
                return Ok(None);
            };
            let mut bounds: Vec<Basic> = vec![a.clone()];
            bounds.extend(cuts.iter().cloned());
            bounds.push(b.clone());
            let mut total: ConditionalValue = zero_cv();
            for w in bounds.windows(2) {
                let Some(piece) = self.antiderivative_method(f, x, &w[0], &w[1], depth, false, Some(&known.0))? else {
                    return Ok(None);
                };
                total = total.add(&piece)?;
            }
            self.finish(Some(total))
        };
        self.under_budget(compute, 2.0)
    }

    fn late_antiderivative(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        self.antiderivative_method(f, x, a, b, depth, true, None)
    }

    fn termwise(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        let mut terms: Vec<Basic> = add_args(f);
        if terms.len() < 2 {
            terms = add_args(&expand(f)?);
            if terms.len() < 2 || terms.len() > TERMWISE_LIMIT {
                return Ok(None);
            }
        } else if terms.len() > TERMWISE_LIMIT {
            return Ok(None);
        }
        let mut total: Option<ConditionalValue> = None;
        for term in &terms {
            let Some(found) = self.integrate(term, x, a, b, depth + 1, false, false)? else {
                return Ok(None);
            };
            if has_infinity(&found.value) {
                return Ok(None);
            }
            total = Some(match total {
                None => found,
                Some(t) => t.add(&found)?,
            });
        }
        let Some(total) = total else {
            return Ok(None);
        };
        if crate::settings::settings().numerical_checks
            && verify_numerically(&total.value, f, x, a, b, &self.assumptions, 2, &[])? == Some(false)
        {
            return Ok(None);
        }
        self.finish(Some(total))
    }

    fn regularized_termwise(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        if *a != Basic::zero() || *b != oo() {
            return Ok(None);
        }
        let mut hints = crate::core::function::Hints::DEFAULT;
        hints.power_exp = false;
        let terms: Vec<Basic> = add_args(&crate::core::function::expand_with_hints(f, Some(&hints), None)?);
        if terms.len() < 2 || terms.len() > TERMWISE_LIMIT {
            return Ok(None);
        }
        let t: Basic = fresh("t", &[(Predicate::positive, true)]);
        let mut total: Option<ConditionalValue> = None;
        for term in &terms {
            let mut found: Option<ConditionalValue> = None;
            for candidate in forms(&subs(term, x, &t)?, &t, &self.assumptions)? {
                found = crate::integrals::marichev::mellin_integrate(&candidate, &t, &self.assumptions, None, true)?;
                if found.is_some() {
                    break;
                }
            }
            let Some(found) = found else {
                total = None;
                break;
            };
            total = Some(match total {
                None => found,
                Some(acc) => acc.add(&found)?,
            });
        }
        let mut at_one: bool = false;
        if total.as_ref().is_none_or_infinite() {
            total = self.transforms_at_one(&terms, x, &t)?;
            if total.is_none() {
                return Ok(None);
            }
            at_one = true;
        }
        let total: ConditionalValue = total.expect("found");
        if !crate::settings::settings().numerical_checks {
            return Ok(None);
        }
        let verdict: Option<bool> = verify_numerically(&total.value, f, x, a, b, &self.assumptions, 2, &[])?;
        if verdict != Some(true) && !(at_one && verdict.is_none()) {
            return Ok(None);
        }
        self.finish(Some(total))
    }

    fn transforms_at_one(&self, terms: &[Basic], x: &Basic, t: &Basic) -> OptCV {
        let s: Basic = fresh("s", &[]);
        let mut summed: Basic = Basic::zero();
        let mut condition: Basic = true_b();
        for term in terms {
            let mut transform = None;
            for candidate in forms(&subs(term, x, t)?, t, &self.assumptions)? {
                transform = crate::integrals::mellin::mellin_transform(&candidate, t, &s)?;
                if transform.is_some() {
                    break;
                }
            }
            let Some(transform) = transform else {
                return Ok(None);
            };
            summed = a2(&summed, &transform.transform)?;
            condition = crate::core::logic::and_(&[condition, transform.condition.clone()]);
        }
        let expanded: Basic = zeta_about_one(&summed, &s)?;
        let budget: Option<f64> = crate::settings::settings().timeout.map(|t| t / 4.0);
        let value: Option<Basic> =
            crate::timeout::attempt_within(budget, || crate::core::limits::limit(&expanded, &s, &Basic::one(), "+"))?;
        match value {
            Some(v)
                if !has_infinity(&v)
                    && !matches!(v.head(), Head::Limit)
                    && !has_head(&v, &|h| matches!(h, Head::Limit))
                    && !v.has(&s) =>
            {
                Ok(Some(ConditionalValue::new(v, condition)))
            }
            _ => Ok(None),
        }
    }

    fn logarithmic_substitution(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if *a != Basic::zero() || *b != Basic::one() || !logarithm_inside(f, x)? {
            return Ok(None);
        }
        let u: Basic = fresh("u", &[(Predicate::positive, true)]);
        let e_u: Basic = exp(&ng(&u)?)?;
        let g: Basic = crate::core::function::expand_log(&m2(&subs(f, x, &e_u)?, &e_u)?, true, true, false)?;
        let Some(found) = self.under_budget(
            || self.integrate(&g, &u, &Basic::zero(), &oo(), depth + 1, false, true),
            2.0,
        )?
        else {
            return Ok(None);
        };
        if crate::settings::settings().numerical_checks
            && verify_numerically(&found.value, f, x, a, b, &self.assumptions, 2, &[])? == Some(false)
        {
            return Ok(None);
        }
        Ok(Some(found))
    }

    fn parameter_exponentials(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        let hints = crate::core::rewriting::RewriteHints::default();
        let g: Basic = replace_where(
            f,
            &|n: &Basic| matches!(n.head(), Head::Function(h) if HYPERBOLIC.contains(h)) && !n.has(x),
            &|n: &Basic| crate::core::rewriting::rewrite(n, &[], "exp", &hints),
        )?;
        let exponentials: Vec<Basic> = g
            .atoms(Some(&|n: &Basic| n.is_function(Function::Exp)))
            .into_iter()
            .filter(|n| !n.has(x))
            .collect();
        if exponentials.is_empty() || !rational_function(&g, std::slice::from_ref(x)) {
            return Ok(None);
        }
        let arguments: Vec<Basic> = exponentials.iter().map(|n| n.args()[0].clone()).collect();
        let mut symbols: BTreeSet<Basic> = BTreeSet::new();
        for argument in &arguments {
            symbols.extend(argument.free_symbols());
        }
        let parameters: Vec<Basic> = sorted_symbols(symbols);
        if parameters.len() != 1 {
            return Ok(None);
        }
        let p: Basic = parameters[0].clone();
        let mut coefficients: Vec<Rat> = Vec::new();
        for argument in &arguments {
            let c: Basic = d2(argument, &p)?;
            if c.has(&p) {
                return Ok(None);
            }
            let Some(r) = c.as_rational() else {
                return Ok(None);
            };
            coefficients.push(r.clone());
        }
        let (mut numerator, mut denominator) = (Int::ZERO, Int::ONE);
        for c in &coefficients {
            numerator = gcd(&numerator, &crate::numbers::numer(c));
            let q: Int = crate::numbers::denom(c);
            denominator = &denominator / gcd(&denominator, &q) * &q;
        }
        let generator: Rat = Rat::from(numerator) / Rat::from(denominator);
        let y: Basic = fresh("y", &[(Predicate::positive, true)]);
        let mut rule: HashMap<Basic, Basic> = HashMap::new();
        for (node, c) in exponentials.iter().zip(&coefficients) {
            rule.insert(node.clone(), p2(&y, &Basic::from_rational(c / &generator))?);
        }
        let h: Basic = crate::core::function::xreplace_evaluated(&g, &rule)?;
        if h.has(&p) {
            return Ok(None);
        }
        let Some(found) = self.under_budget(|| self.integrate(&h, x, a, b, depth + 1, false, true), 2.0)? else {
            return Ok(None);
        };
        let real: Option<bool> = self.ask(&crate::assumptions::facts::element(&p, &Basic::atom(Head::Reals))?)?;
        let extra: Basic = if real == Some(true) {
            true_b()
        } else {
            crate::assumptions::facts::element(&p, &Basic::atom(Head::Reals))?
        };
        let condition: Basic = crate::core::logic::and_(&[found.condition.clone(), extra]);
        let value: Basic = subs(&found.value, &y, &exp(&m2(&Basic::from_rational(generator), &p)?)?)?;
        Ok(Some(ConditionalValue::new(value, condition)))
    }

    fn dirichlet(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        let zero: Basic = Basic::zero();
        if !(*a == zero || *a == neg_oo())
            || !(*b == zero || *b == oo())
            || !has_any_function(f, &[Function::Sin, Function::Cos, Function::Exp])
        {
            return Ok(None);
        }
        self.finish(crate::integrals::dirichlet::dirichlet_integral(
            f,
            x,
            a,
            b,
            &self.assumptions,
        )?)
    }

    fn table(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        self.finish(crate::integrals::tables::table_integral(f, x, a, b, &self.assumptions)?)
    }

    fn table_termwise(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        let mut hints = crate::core::function::Hints::DEFAULT;
        hints.power_exp = false;
        let terms: Vec<Basic> = add_args(&crate::core::function::expand_with_hints(f, Some(&hints), None)?);
        if terms.len() < 2 || terms.len() > TERMWISE_LIMIT {
            return Ok(None);
        }
        let mut total: ConditionalValue = zero_cv();
        for term in &terms {
            let (constant, rest) = crate::core::expr::as_independent(&combined_exponentials(term, Some(x))?, x, false)?;
            let Some(found) = crate::integrals::tables::table_integral(&rest, x, a, b, &self.assumptions)? else {
                return Ok(None);
            };
            total = total.add(&found.scaled(&constant)?)?;
        }
        self.finish(Some(total))
    }

    fn elliptic(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        let square_root: bool = f
            .atoms(Some(&|n: &Basic| n.is_pow()))
            .iter()
            .any(|n| rational_of(&n.args()[1]).is_some_and(|(_, q)| q == 2));
        if !square_root {
            return Ok(None);
        }
        self.finish(crate::integrals::elliptic::elliptic_integral(
            f,
            x,
            a,
            b,
            &self.assumptions,
        )?)
    }

    fn algebraic_method(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        let radical: bool = f
            .atoms(Some(&|n: &Basic| n.is_pow()))
            .iter()
            .any(|n| n.args()[1].as_rational().is_some_and(|r| !r.is_int()) && n.args()[0].has(x));
        if depth > 1 || !radical {
            return Ok(None);
        }
        self.finish(algebraic_integral(f, x, a, b, &self.assumptions)?)
    }

    fn series(&self, f: &Basic, _x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if depth > 0 {
            return Ok(None);
        }
        if *a != Basic::zero() {
            if infinite(a) || infinite(b) || !(has_any_function(f, &[Function::Log]) || has_trigonometric(f)) {
                return Ok(None);
            }
        } else if *b != Basic::one() && *b != oo() && !yes(b, Predicate::positive) {
            return Ok(None);
        }
        Err(crate::error::not_ported("integrals/series (series_integral)"))
    }

    fn laplace(&self, f: &Basic, _x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if depth > 1 || *a != Basic::zero() || *b != oo() || !has_any_function(f, &[Function::Exp]) {
            return Ok(None);
        }
        Err(crate::error::not_ported("integrals/laplace (laplace_integral)"))
    }

    fn contours(&self, _f: &Basic, _x: &Basic, _a: &Basic, _b: &Basic, _depth: usize) -> OptCV {
        Err(crate::error::not_ported("integrals/contours (contour_integral)"))
    }

    fn transformations(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if depth > 1 {
            return Ok(None);
        }
        self.finish(transformation_integral(f, x, a, b, &self.assumptions)?)
    }

    fn one_parameter(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> Vec<Basic> {
        let mut s: BTreeSet<Basic> = f.free_symbols();
        s.remove(x);
        for t in a.free_symbols().union(&b.free_symbols()) {
            s.remove(t);
        }
        sorted_symbols(s)
    }

    fn holonomic(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if !self.parametric || depth > 1 {
            return Ok(None);
        }
        if self.one_parameter(f, x, a, b).len() != 1 {
            return Ok(None);
        }
        Err(crate::error::not_ported(
            "integrals/telescoping and integrals/reduction (holonomic_integral)",
        ))
    }

    fn dfinite(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if !self.parametric || depth > 1 {
            return Ok(None);
        }
        if self.one_parameter(f, x, a, b).len() != 1 {
            return Ok(None);
        }
        Err(crate::error::not_ported(
            "integrals/dfinite and integrals/telescoping (dfinite_integral)",
        ))
    }

    fn parametric_method(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        if !self.parametric || depth > 1 {
            return Ok(None);
        }
        let integrator: DefiniteIntegrator = DefiniteIntegrator::new(&self.assumptions);
        for p in crate::integrals::parametric::candidate_parameters(f, x) {
            if let Some(found) = crate::integrals::parametric::parametric_integral(
                f,
                x,
                a,
                b,
                &p,
                &self.assumptions,
                depth,
                Some(&integrator),
            )? {
                return self.finish(Some(found));
            }
        }
        Ok(None)
    }

    // -- the last of the methods

    /// The Meijer G-function theorems and an antiderivative at the
    /// endpoints, each under a quarter of the time limit and checked
    /// numerically (`_last`).
    pub fn last(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, _depth: usize) -> OptCV {
        self.last_routes(f, x, a, b, true)
    }

    fn last_routes(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, endpoints: bool) -> OptCV {
        let budget: Option<f64> = crate::settings::settings().timeout.map(|t| t / 4.0);
        let infinite_range: bool = infinite(a) || infinite(b);
        let mut routes: Vec<u8> = if endpoints { vec![0, 1] } else { vec![0] };
        if !infinite_range {
            routes.reverse();
        }
        let delta: bool = has_any_function(f, &[Function::DiracDelta]);
        for route in routes {
            let value = crate::timeout::attempt_within(budget, || {
                if route == 0 {
                    self.meijer_value(f, x, a, b)
                } else {
                    self.endpoint_value(f, x, a, b)
                }
            });
            let value: Option<ConditionalValue> = match value {
                Ok(v) => v.flatten(),
                Err(crate::error::PolyError::Index(_)) => None,
                Err(e) => return Err(e),
            };
            let Some(value) = value else {
                continue;
            };
            if has_bad(&value.value) || value.condition.has(x) {
                continue;
            }
            let Some(found) = self.finish(Some(ConditionalValue::new(
                value.value.clone(),
                self.real_parts(&value.condition)?,
            )))?
            else {
                continue;
            };
            if crate::settings::settings().numerical_checks && !delta {
                if found.value.has(&oo()) || found.value.has(&neg_oo()) {
                    continue;
                }
                if verify_numerically(&found.value, f, x, a, b, &self.assumptions, 2, &[])? != Some(true) {
                    continue;
                }
            }
            let mut tidied: Basic = crate::integrals::marichev::tidy(&found.value, &self.assumptions, &true_b())?;
            if tidied != found.value
                && crate::settings::settings().numerical_checks
                && !delta
                && verify_numerically(&tidied, f, x, a, b, &self.assumptions, 2, &[])? == Some(false)
            {
                tidied = found.value.clone();
            }
            return Ok(Some(ConditionalValue::new(tidied, found.condition)));
        }
        Ok(None)
    }

    fn meijer_value(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> OptCV {
        let Some((value, condition)) = crate::integrals::meijerint::meijerint_definite(f, x, a, b)? else {
            return Ok(None);
        };
        if is_false_b(&condition) {
            return Ok(None);
        }
        Ok(Some(ConditionalValue::new(value, condition)))
    }

    fn endpoint_value(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> OptCV {
        if has_any_function(f, &[Function::DiracDelta]) {
            let Some(big_f) = crate::integrals::deltafunctions::deltaintegrate(f, x)? else {
                return Ok(None);
            };
            return self.evaluated_antiderivative(&big_f, f, x, a, b);
        }
        let exact: Vec<&str> = crate::integrals::antiderivative::METHODS
            .iter()
            .copied()
            .filter(|n| *n != "meijer")
            .collect();
        let heuristics: Vec<&str> = crate::integrals::indefinite::METHODS
            .iter()
            .copied()
            .filter(|n| *n != "meijer" && !exact.contains(n))
            .collect();
        let mut free: BTreeSet<Basic> = f.free_symbols();
        free.remove(x);
        let passes: Vec<Vec<&str>> = if real_bounds(a, b) && !free.is_empty() {
            vec![exact]
        } else {
            vec![exact, heuristics]
        };
        for names in passes {
            let Some(found) =
                crate::integrals::indefinite::verified_antiderivative(f, x, &self.assumptions, Some(&names))?
            else {
                continue;
            };
            if let Some(value) = self.evaluated_antiderivative(&found.0, f, x, a, b)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn evaluated_antiderivative(&self, big_f: &Basic, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> OptCV {
        use crate::core::relational as rl;
        if is_unevaluated(big_f) {
            return Ok(None);
        }
        let piecewise = |e: &Basic| has_head(e, &|h| matches!(h, Head::Piecewise));
        if piecewise(big_f) && !piecewise(f) {
            return Ok(None);
        }
        if real_bounds(a, b) && !has_any_function(f, &[Function::DiracDelta]) {
            if let Some(found) = crate::integrals::antiderivative::antiderivative_integral(
                f,
                x,
                a,
                b,
                &self.assumptions,
                false,
                Some(big_f),
                None,
            )? {
                if !found.value.has(&oo()) && !found.value.has(&neg_oo()) {
                    return Ok(Some(found));
                }
            }
            let Some(inside) = crate::integrals::antiderivative::singularities(f, x, &self.assumptions, None)? else {
                return Ok(None);
            };
            let (lower, upper) = (
                crate::polys::polytools::cancel(a, &[])?,
                crate::polys::polytools::cancel(b, &[])?,
            );
            for point in &inside {
                if self.holds(rl::lt(&lower, point))? != Some(false)
                    && self.holds(rl::lt(point, &upper))? != Some(false)
                {
                    return Ok(None);
                }
            }
        }
        let mut values: Vec<Basic> = Vec::new();
        for (point, direction) in [(b, "-"), (a, "+")] {
            let Some(value) = endpoint(big_f, x, point, direction)? else {
                return Ok(None);
            };
            values.push(value);
        }
        let total: Basic = s2(&values[0], &values[1])?;
        if total.has(&oo()) || total.has(&neg_oo()) {
            return Ok(None);
        }
        Ok(Some(ConditionalValue::new(total, true_b())))
    }

    // -- the entry point

    /// `Integral(f, (x, a, b))` (`integrate`): without `fallback` the last
    /// method is not tried, inside a `mapped` range the slow methods are
    /// skipped.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate(
        &self,
        f: &Basic,
        x: &Basic,
        a: &Basic,
        b: &Basic,
        depth: usize,
        fallback: bool,
        mapped: bool,
    ) -> OptCV {
        use crate::core::relational as rl;
        crate::timeout::check()?;
        if depth > MAX_DEPTH {
            return Ok(None);
        }
        if a == b {
            return Ok(Some(zero_cv()));
        }
        if self.holds(rl::gt(a, b))? == Some(true) {
            return Ok(match self.integrate(f, x, b, a, depth, fallback, mapped)? {
                None => None,
                Some(r) => Some(r.scaled(&int(-1))?),
            });
        }
        if *f == Basic::zero() {
            return Ok(Some(zero_cv()));
        }
        if !f.has(x) {
            if infinite(a) || infinite(b) {
                if self.holds(rl::gt(f, &Basic::zero()))? == Some(true) {
                    return Ok(Some(ConditionalValue::new(oo(), true_b())));
                }
                if self.holds(rl::lt(f, &Basic::zero()))? == Some(true) {
                    return Ok(Some(ConditionalValue::new(neg_oo(), true_b())));
                }
                return Ok(None);
            }
            return Ok(Some(ConditionalValue::new(m2(f, &s2(b, a)?)?, true_b())));
        }
        let mut f: Basic = if has_head(f, &|h| matches!(h, Head::Piecewise)) {
            crate::functions::elementary::piecewise::piecewise_fold(f, None)?
        } else {
            f.clone()
        };
        if has_any_function(&f, &[Function::DiracDelta]) {
            return self.last(&f, x, a, b, depth);
        }
        f = combined_exponentials(&f, Some(x))?;
        f = sums_to_products(&f, x)?;
        f = self.arctangent_pairs(&f, x, a, b)?;
        let (constant, rest) = crate::core::expr::as_independent(
            &crate::core::exprtools::factor_terms(&f, false, false, true, false)?,
            x,
            false,
        )?;
        if constant != Basic::one() {
            return Ok(match self.integrate(&rest, x, a, b, depth, fallback, mapped)? {
                None => None,
                Some(r) => Some(r.scaled(&constant)?),
            });
        }
        if let Some(found) = self.split_integer_parts(&f, x, a, b, depth)? {
            return Ok(Some(found));
        }
        if !mapped {
            if let Some(found) = self.binomial_differential(&f, x, a, b, depth)? {
                return Ok(Some(found));
            }
        }
        let (parametric, found) = self.split_branches(&f, x, a, b, depth)?;
        if found.is_some() {
            return Ok(found);
        }
        if parametric {
            return Ok(None);
        }
        if !mapped {
            if let Some(found) = self.transformations(&f, x, a, b, depth)? {
                return Ok(Some(found));
            }
        }
        let (singular, found) = self.split_singularities(&f, x, a, b, depth)?;
        if singular
            && depth == 0
            && (self.finite_part || self.principal_value)
            && found.as_ref().is_none_or_infinite_or_zoo()
        {
            let regular: Option<ConditionalValue> = if self.finite_part {
                self.finish(crate::integrals::antiderivative::finite_part_integral(
                    &f,
                    x,
                    a,
                    b,
                    &self.assumptions,
                    None,
                )?)?
            } else {
                self.finish(crate::integrals::antiderivative::principal_value_integral(
                    &f,
                    x,
                    a,
                    b,
                    &self.assumptions,
                    None,
                )?)?
            };
            if regular.is_some() {
                return Ok(regular);
            }
        }
        if found.is_some() {
            return Ok(found);
        }
        if singular {
            return Ok(None);
        }
        let mut allowed: BTreeSet<Basic> = f.free_symbols();
        allowed.extend(a.free_symbols());
        allowed.extend(b.free_symbols());
        allowed.remove(x);
        if let Some(found) = self.asymptotic_divergence(&f, x, a, b)? {
            return Ok(Some(found));
        }
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum S {
            Table,
            TableTermwise,
            Dirichlet,
            Radicals,
            Quick,
            Canonical,
            Regularized,
            MeanValue,
            Elliptic,
            Trigonometric,
            Residues,
            Logarithmic,
            ParameterExponentials,
            RadicalSubstitution,
            CompletedSquare,
            Tangent,
            Mapped,
            Inversion,
            Contours,
            Algebraic,
            SplitPowers,
            Antiderivative,
            Holonomic,
            Laplace,
            Parametric,
            Series,
            Dfinite,
            Termwise,
            Late,
        }
        let mut strategies: Vec<S> = vec![
            S::Table,
            S::TableTermwise,
            S::Dirichlet,
            S::Radicals,
            S::Quick,
            S::Canonical,
            S::Regularized,
            S::MeanValue,
            S::Elliptic,
            S::Trigonometric,
        ];
        if has_trigonometric(&f) && whole_periods(a, b)? {
            let at: usize = strategies.iter().position(|s| *s == S::MeanValue).expect("mean value");
            strategies.insert(at, S::Residues);
        }
        strategies.extend([S::Logarithmic, S::ParameterExponentials]);
        if !mapped && depth == 0 {
            strategies.extend([S::RadicalSubstitution, S::CompletedSquare, S::Tangent]);
        }
        let rational: bool = rational_function(&f, std::slice::from_ref(x));
        if rational {
            strategies.push(S::Residues);
        }
        strategies.extend([S::Mapped, S::Inversion]);
        if !rational && !strategies.contains(&S::Residues) {
            strategies.push(S::Residues);
        }
        strategies.extend([S::Contours, S::Algebraic]);
        if *a == neg_oo() && *b == oo() && has_any_function(&f, &HYPERBOLIC) {
            strategies.retain(|s| *s != S::Contours);
            strategies.insert(0, S::Contours);
        }
        strategies.push(S::SplitPowers);
        strategies.push(S::Antiderivative);
        if !mapped {
            strategies.extend([S::Holonomic, S::Laplace, S::Parametric, S::Series, S::Dfinite]);
        }
        strategies.push(S::Termwise);
        if depth == 0 {
            strategies.push(S::Late);
        }
        for strategy in strategies {
            let found: PolyResult<Option<ConditionalValue>> = match strategy {
                S::Table => self.table(&f, x, a, b, depth),
                S::TableTermwise => self.table_termwise(&f, x, a, b, depth),
                S::Dirichlet => self.dirichlet(&f, x, a, b, depth),
                S::Radicals => self.radicals(&f, x, a, b, depth),
                S::Quick => self.quick_antiderivative(&f, x, a, b, depth),
                S::Canonical => self.canonical(&f, x, a, b, depth),
                S::Regularized => self.regularized_termwise(&f, x, a, b, depth),
                S::MeanValue => self.mean_value(&f, x, a, b, depth),
                S::Elliptic => self.elliptic(&f, x, a, b, depth),
                S::Trigonometric => self.trigonometric(&f, x, a, b, depth),
                S::Residues => self.residues(&f, x, a, b, depth),
                S::Logarithmic => self.logarithmic_substitution(&f, x, a, b, depth),
                S::ParameterExponentials => self.parameter_exponentials(&f, x, a, b, depth),
                S::RadicalSubstitution => self.radical_substitution(&f, x, a, b, depth),
                S::CompletedSquare => self.completed_square(&f, x, a, b, depth),
                S::Tangent => self.tangent_substitution(&f, x, a, b, depth),
                S::Mapped => self.mapped(&f, x, a, b, depth),
                S::Inversion => self.inversion(&f, x, a, b, depth),
                S::Contours => self.contours(&f, x, a, b, depth),
                S::Algebraic => self.algebraic_method(&f, x, a, b, depth),
                S::SplitPowers => self.split_powers(&f, x, a, b, depth),
                S::Antiderivative => self.antiderivative_method(&f, x, a, b, depth, false, None),
                S::Holonomic => self.holonomic(&f, x, a, b, depth),
                S::Laplace => self.laplace(&f, x, a, b, depth),
                S::Parametric => self.parametric_method(&f, x, a, b, depth),
                S::Series => self.series(&f, x, a, b, depth),
                S::Dfinite => self.dfinite(&f, x, a, b, depth),
                S::Termwise => self.termwise(&f, x, a, b, depth),
                S::Late => self.late_antiderivative(&f, x, a, b, depth),
            };
            // Python skips a method on ZeroDivisionError, AttributeError,
            // AssertionError, OverflowError
            let found: Option<ConditionalValue> = match found {
                Ok(v) => v,
                Err(crate::error::PolyError::ZeroDivision(_)) => None,
                Err(e) => return Err(e),
            };
            if let Some(found) = found {
                let leaked = |e: &Basic| e.free_symbols().iter().any(|s| !allowed.contains(s));
                if leaked(&found.value) || leaked(&found.condition) {
                    continue;
                }
                return Ok(Some(found));
            }
        }
        if fallback {
            return self.last(&f, x, a, b, depth);
        }
        Ok(None)
    }

    /// `floor`, `ceiling` and `frac` of real functions of `x` over a
    /// finite numeric range, cut at the integer crossings
    /// (`_split_integer_parts`).
    fn split_integer_parts(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        let integer_part = |n: &Basic| {
            (n.is_function(Function::Floor) || n.is_function(Function::Ceiling) || n.is_function(Function::Frac))
                && n.has(x)
        };
        let nodes: Vec<Basic> = sorted_by_str(f.atoms(Some(&integer_part)));
        if nodes.is_empty() || infinite(a) || infinite(b) || !is_number(a) || !is_number(b) {
            return Ok(None);
        }
        let mut points: Vec<Basic> = Vec::new();
        for node in &nodes {
            let u: &Basic = &node.args()[0];
            if u.free_symbols() != std::iter::once(x.clone()).collect::<BTreeSet<Basic>>() {
                return Ok(None);
            }
            let Some(found) = self.integer_crossings(u, x, a, b)? else {
                return Ok(None);
            };
            points.extend(found);
        }
        let placed: Vec<Basic> = if points.is_empty() {
            Vec::new()
        } else {
            match self.points_in(&distinct(&points), a, b)? {
                None => return Ok(None),
                Some(p) => p,
            }
        };
        let mut bounds: Vec<Basic> = vec![a.clone()];
        bounds.extend(placed);
        bounds.push(b.clone());
        let mut total: ConditionalValue = zero_cv();
        for w in bounds.windows(2) {
            let middle: Basic = d2(&a2(&w[0], &w[1])?, &int(2))?;
            let mut piece: Basic = f.clone();
            for node in &nodes {
                let u: &Basic = &node.args()[0];
                let level: Basic = apply(Function::Floor, &[subs(u, x, &middle)?])?;
                if integer_of(&level).is_none() {
                    return Ok(None);
                }
                let replacement: Basic = if node.is_function(Function::Floor) {
                    level
                } else if node.is_function(Function::Ceiling) {
                    a2(&level, &Basic::one())?
                } else {
                    s2(u, &level)?
                };
                piece = replaced_node(&piece, node, &replacement)?;
            }
            if !piece.atoms(Some(&integer_part)).is_empty() {
                return Ok(None);
            }
            let Some(value) = self.integrate(&piece, x, &w[0], &w[1], depth + 1, true, false)? else {
                return Ok(None);
            };
            if has_bad(&value.value) {
                return Ok(None);
            }
            total = total.add(&value)?;
        }
        Ok(Some(total))
    }
}

trait OptionalValue {
    fn is_none_or_infinite(&self) -> bool;
    fn is_none_or_infinite_or_zoo(&self) -> bool;
}

impl OptionalValue for Option<&ConditionalValue> {
    fn is_none_or_infinite(&self) -> bool {
        match self {
            None => true,
            Some(v) => has_infinity(&v.value),
        }
    }
    fn is_none_or_infinite_or_zoo(&self) -> bool {
        match self {
            None => true,
            Some(v) => v.value.has(&oo()) || v.value.has(&neg_oo()) || v.value.has(&Basic::atom(Head::ComplexInfinity)),
        }
    }
}

/// The terms a sum may have to be integrated term by term
/// (`_TERMWISE_LIMIT`).
const TERMWISE_LIMIT: usize = 8;

/// `k` when `b - a == 2*pi*k` for a positive integer `k` (`_periods` of
/// `integrals/residues`).
pub fn periods(a: &Basic, b: &Basic) -> PolyResult<Option<i64>> {
    let k: Basic = m2(&a2(b, &m2(&int(-1), a)?)?, &p2(&m2(&int(2), &Basic::pi())?, &int(-1))?)?;
    Ok(integer_of(&k).filter(|k| *k > 0))
}

/// Whether `b - a` is a positive integer multiple of `2*pi`
/// (`_whole_periods`).
pub fn whole_periods(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(is_number(a) && is_number(b) && periods(a, b)?.is_some())
}

/// The integrator without differentiation under the integral sign and its
/// numerical check, under the assumptions (`DefiniteIntegrator`).
pub struct DefiniteIntegrator {
    assumptions: Vec<Basic>,
    integrator: Integrator,
}

impl DefiniteIntegrator {
    pub fn new(assumptions: &[Basic]) -> DefiniteIntegrator {
        DefiniteIntegrator {
            assumptions: assumptions.to_vec(),
            integrator: Integrator::new(assumptions, false, false, false, false, true),
        }
    }
}

impl crate::integrals::parametric::ParametricIntegrator for DefiniteIntegrator {
    fn integrate(&self, f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> OptCV {
        self.integrator.integrate(f, x, a, b, depth, true, false)
    }

    fn verify(&self, value: &Basic, f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
        verify_numerically(value, f, x, a, b, &self.assumptions, 2, &[])
    }
}

use crate::integrals::transformations::transformation_integral;

use crate::integrals::algebraic::algebraic_integral;

/// `e` with the subtree `old` replaced by `new`, the rebuilt nodes
/// evaluated (`_replaced`).
fn replaced_node(e: &Basic, old: &Basic, new: &Basic) -> R {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    rule.insert(old.clone(), new.clone());
    crate::core::function::xreplace_evaluated(e, &rule)
}

// -- the entry points

/// The options of [`definite_integral`].
#[derive(Clone, Debug)]
pub struct Options {
    /// `'piecewise'` (true) or `'none'` (false)
    pub piecewise: bool,
    pub recognize: bool,
    pub principal_value: bool,
    pub finite_part: bool,
    pub numeric: bool,
    pub digits: usize,
    pub regularize: bool,
    pub summability: Option<String>,
    pub special_values: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            piecewise: true,
            recognize: false,
            principal_value: false,
            finite_part: false,
            numeric: false,
            digits: 15,
            regularize: false,
            summability: None,
            special_values: true,
        }
    }
}

/// The value with `function` applied, case by case for a `Piecewise` of
/// cases (`ConditionalValue.mapped`).
fn mapped(found: &ConditionalValue, function: &dyn Fn(&Basic, &Basic) -> R) -> PolyResult<ConditionalValue> {
    match found.cases() {
        None => Ok(ConditionalValue::new(
            function(&found.value, &found.condition)?,
            found.condition.clone(),
        )),
        Some(branches) => {
            // the class, unevaluated (Python's Piecewise(*...))
            let pieces: Vec<Basic> = branches
                .iter()
                .map(|(v, c)| Ok(Basic::new(Head::ExprCondPair, vec![function(v, c)?, c.clone()])))
                .collect::<PolyResult<_>>()?;
            Ok(ConditionalValue::new(
                Basic::new(Head::Piecewise, pieces),
                found.condition.clone(),
            ))
        }
    }
}

fn has_nan(e: &Basic) -> bool {
    has_head(e, &|h| matches!(h, Head::NaN))
}

/// `Integral(f, (x, a, b))` under assumptions on the parameters
/// (`definite_integral`).
pub fn definite_integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, assumptions: &[Basic], options: &Options) -> R {
    let unevaluated = || crate::integrals::integral::integral(f, x, Some((a, b)));
    let mut found: Option<ConditionalValue> = match &options.summability {
        Some(_) => return Err(crate::error::not_ported("integrals/summability (summable_integral)")),
        None => conditional_integral(
            f,
            x,
            a,
            b,
            assumptions,
            options.principal_value,
            options.finite_part,
            options.regularize,
        )?,
    };
    let missing = |found: &Option<ConditionalValue>| found.as_ref().is_none_or_nan();
    if missing(&found) && options.recognize {
        let constants: Vec<Basic> = crate::integrals::recognize::default_constants()?;
        if let Some(guessed) = crate::integrals::recognize::recognize_integral(f, x, a, b, &constants, 30)? {
            return Ok(guessed);
        }
    }
    if missing(&found) && options.numeric {
        // validated_integral returns a Float, which the core does not have
        if crate::integrals::validated::validated_integral(f, x, a, b, options.digits)?.is_some() {
            return Err(crate::error::not_ported(
                "a Float value of definite_integral(numeric=True)",
            ));
        }
    }
    if missing(&found) {
        return unevaluated();
    }
    let found_value: ConditionalValue = found.take().expect("found");
    let found_value: ConditionalValue = mapped(&found_value, &|value, condition| {
        crate::integrals::marichev::checked_tidy(value, assumptions, condition)
    })?;
    let result: Basic = if options.piecewise {
        found_value.as_piecewise(&unevaluated()?)?
    } else {
        found_value.value.clone()
    };
    if !options.special_values {
        return Ok(result);
    }
    let parameters: Vec<Basic> = parameters_of(f, x, a, b);
    let failure: RefCell<Option<crate::error::PolyError>> = RefCell::new(None);
    let at_point = |point: &crate::special_values::Point, at: &[Basic]| -> PolyResult<Option<Basic>> {
        let (g, lower, upper) = (xr(f, point)?, xr(a, point)?, xr(b, point)?);
        let mut inner: Options = options.clone();
        inner.special_values = true;
        let value: Basic = definite_integral(&g, x, &lower, &upper, at, &inner)?;
        if !is_unevaluated(&value) {
            return Ok(Some(value));
        }
        Ok(Some(
            checked_limit(&result, point, &g, x, &lower, &upper, at)?.unwrap_or(value),
        ))
    };
    let defined = |point: &crate::special_values::Point| -> bool {
        if failure.borrow().is_some() {
            return false;
        }
        let check = || -> PolyResult<bool> {
            if !crate::special_values::defined_problem(&xr(f, point)?, std::slice::from_ref(x)) {
                return Ok(false);
            }
            for e in [a, b] {
                if has_nan_or_zoo(&xr(e, point)?) {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match check() {
            Ok(v) => v,
            Err(e) => {
                *failure.borrow_mut() = Some(e);
                false
            }
        }
    };
    let value: R = crate::special_values::with_special_values(
        &result,
        &parameters,
        assumptions,
        &at_point,
        Some(&defined),
        false,
        None,
        true,
        None,
    );
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }
    value
}

trait NanCheck {
    fn is_none_or_nan(&self) -> bool;
}
impl NanCheck for Option<&ConditionalValue> {
    fn is_none_or_nan(&self) -> bool {
        match self {
            None => true,
            Some(v) => has_nan(&v.value),
        }
    }
}

/// The limit of the branch of `value` taken at the point (one
/// parameter), when finite and confirmed by the quadrature
/// (`_checked_limit`).
fn checked_limit(
    value: &Basic,
    point: &crate::special_values::Point,
    g: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<Basic>> {
    if point.len() != 1 || !crate::settings::settings().numerical_checks {
        return Ok(None);
    }
    let (p, v) = &point[0];
    let branches: Vec<(Basic, Basic)> = if matches!(value.head(), Head::Piecewise) {
        pairs(value)
    } else {
        vec![(value.clone(), true_b())]
    };
    for (e, condition) in branches {
        if is_unevaluated(&e) || is_false_b(&xr(&condition, point)?) {
            continue;
        }
        let budget: Option<f64> = crate::settings::settings().timeout.map(|t| t / 4.0);
        let found: Option<Basic> =
            crate::timeout::attempt_within(budget, || crate::core::limits::limit(&e, p, v, "+-"))?;
        let Some(found) = found else {
            return Ok(None);
        };
        if is(&found, Predicate::finite) != Some(true) || has_head(&found, &|h| matches!(h, Head::Limit)) {
            return Ok(None);
        }
        return Ok(
            if verify_numerically(&found, g, x, a, b, assumptions, 2, &[])? == Some(true) {
                Some(found)
            } else {
                None
            },
        );
    }
    Ok(None)
}

/// The integrand and bounds with the equalities among the assumptions
/// substituted (`_with_equalities`).
fn with_equalities(
    f: &Basic,
    a: &Basic,
    b: &Basic,
    x: &Basic,
    assumptions: &[Basic],
) -> PolyResult<(Basic, Basic, Basic)> {
    use crate::core::basic::RelOp;
    let mut replacement: Values = Vec::new();
    let is_symbol = |e: &Basic| matches!(e.head(), Head::Symbol(_) | Head::Dummy(..));
    for item in assumptions {
        let atoms: Vec<Basic> = if rel_op(item) == Some(RelOp::Eq) {
            vec![item.clone()]
        } else if matches!(item.head(), Head::And) {
            item.atoms(Some(&|n: &Basic| rel_op(n) == Some(RelOp::Eq)))
        } else {
            Vec::new()
        };
        for atom in atoms {
            let (lhs, rhs) = (&atom.args()[0], &atom.args()[1]);
            if is_symbol(lhs) && lhs != x && !rhs.has(lhs) && !rhs.has(x) {
                replacement.retain(|(k, _)| k != lhs);
                replacement.push((lhs.clone(), rhs.clone()));
            } else if is_symbol(rhs) && rhs != x && !lhs.has(rhs) && !lhs.has(x) {
                replacement.retain(|(k, _)| k != rhs);
                replacement.push((rhs.clone(), lhs.clone()));
            }
        }
    }
    if replacement.is_empty() {
        return Ok((f.clone(), a.clone(), b.clone()));
    }
    Ok((xr(f, &replacement)?, xr(a, &replacement)?, xr(b, &replacement)?))
}

/// Whether `f` has a power of a power with non-real exponents
/// (`_nested_complex_powers`).
fn nested_complex_powers(f: &Basic) -> bool {
    f.atoms(Some(&|n: &Basic| n.is_pow())).iter().any(|node| {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        is(exponent, Predicate::extended_real) == Some(false)
            && base.is_pow()
            && is(&base.args()[1], Predicate::extended_real) == Some(false)
    })
}

/// The value of `Integral(f, (x, a, b))` with the condition under which it
/// holds (`conditional_integral`).
#[allow(clippy::too_many_arguments)]
pub fn conditional_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
    principal_value: bool,
    finite_part: bool,
    regularize: bool,
) -> OptCV {
    let (f, a, b) = with_equalities(f, a, b, x, assumptions)?;
    let integrator: Integrator = Integrator::new(assumptions, true, principal_value, finite_part, regularize, true);
    if !real_bounds(&a, &b) || nested_complex_powers(&f) {
        return integrator.last(&f, x, &a, &b, 0);
    }
    let parametric: bool = integrator.parametric_points(&f, x, &a, &b)?;
    let budget: Option<f64> = crate::settings::settings()
        .timeout
        .map(|t| t / if parametric { 1.0 } else { 2.0 });
    let found: Option<ConditionalValue> =
        crate::timeout::attempt_within(budget, || integrator.integrate(&f, x, &a, &b, 0, false, false))?.flatten();
    if let Some(found) = &found {
        if is_true_b(&found.condition) {
            return Ok(Some(found.clone()));
        }
    }
    if parametric {
        return Ok(found);
    }
    let fallback: Option<ConditionalValue> = integrator.last(&f, x, &a, &b, 0)?;
    if let (Some(fb), Some(fd)) = (&fallback, &found) {
        if is_true_b(&fb.condition)
            && crate::settings::settings().numerical_checks
            && !holds_outside(&fb.value, &fd.condition, &f, x, &a, &b, assumptions)?
        {
            return Ok(found);
        }
    }
    if let Some(fb) = &fallback {
        if found.is_none() || is_true_b(&fb.condition) {
            return Ok(fallback);
        }
    }
    Ok(found)
}
