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
