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
