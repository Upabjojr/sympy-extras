//! Port of `sympy_extras/solvers/isolation.py`: the real roots of
//! transcendental functions of one variable isolated exactly, each
//! enclosed in an interval with rational ends containing no other root
//! (`TranscendentalRoot`, `Head::TranscendentalRoot` with the arguments
//! `(f, x, a, b)`); the roots of the derivative first (Rolle's theorem),
//! the signs certified by interval arithmetic or exact evaluation.

use crate::assumptions::analysis::{
    between, certified_compare, certified_sign, continuous_on, endpoint_value, finite_zeros, has_discrete_zeros,
    pieces, sign_on, sorted_constants,
};
use crate::assumptions::intervals::{evaluate, interval_sign, PrecisionGuard};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::*;
use crate::sets::sets::{interval, interval_parts};

type R = PolyResult<Basic>;

fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn number(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}
fn is_root(e: &Basic) -> bool {
    matches!(e.head(), Head::TranscendentalRoot)
}
fn rat_of(e: &Basic) -> Rat {
    e.as_rational().cloned().expect("a rational end")
}
fn pow2(k: usize) -> Rat {
    Rat::from(Int::ONE << k)
}

// -- the root objects

/// The parts `(f, x, lower, upper)` of a root object.
pub fn parts(e: &Basic) -> (Basic, Basic, Rat, Rat) {
    (
        e.args()[0].clone(),
        e.args()[1].clone(),
        rat_of(&e.args()[2]),
        rat_of(&e.args()[3]),
    )
}

/// A real root of `f` isolated in `(a, b)`: the signs at the ends are
/// checked opposite (`TranscendentalRoot`).
pub fn root(f: &Basic, x: &Basic, a: &Rat, b: &Rat) -> R {
    if !x.is_symbol() {
        return Err(PolyError::Type(format!("the variable must be a symbol, got {}", x)));
    }
    if f.free_symbols().iter().any(|s| s != x) {
        return Err(value_error(format!("{} has parameters besides {}", f, x)));
    }
    if a >= b {
        return Err(value_error(format!(
            "an interval with a < b is expected, got ({}, {})",
            a, b
        )));
    }
    let (left, right) = (sign_at(f, x, a)?, sign_at(f, x, b)?);
    match (left, right) {
        (Some(l), Some(r)) if l * r < 0 => {}
        _ => {
            return Err(value_error(format!(
                "{} does not change sign between {} and {}",
                f, a, b
            )))
        }
    }
    Ok(Basic::new(
        Head::TranscendentalRoot,
        vec![f.clone(), x.clone(), number(a), number(b)],
    ))
}

/// The node rebuilt from new arguments (`_evaluated`).
pub fn evaluated(args: &[Basic]) -> R {
    if !(args[1].is_symbol() && args[2].is_rational_class() && args[3].is_rational_class()) {
        return Err(PolyError::Type(format!(
            "invalid arguments of a TranscendentalRoot: {:?}",
            args
        )));
    }
    root(&args[0], &args[1], &rat_of(&args[2]), &rat_of(&args[3]))
}

/// The facts of a root object (`_eval_facts`).
pub fn root_facts(e: &Basic) -> crate::core::assumptions::FactList {
    use Predicate as P;
    let (_, _, lo, hi) = parts(e);
    let mut facts = crate::core::assumptions::FactList::new();
    facts.set(P::commutative, true);
    facts.set(P::real, true);
    facts.set(P::finite, true);
    facts.set(P::zero, false);
    if !rat_is_negative(&lo) {
        facts.set(P::positive, true);
    } else if hi <= Rat::ZERO {
        facts.set(P::positive, false);
    }
    if hi <= Rat::ZERO {
        facts.set(P::negative, true);
    } else if !rat_is_negative(&lo) {
        facts.set(P::negative, false);
    }
    facts
}

/// `TranscendentalRoot(f, x, a, b)` as Python prints it.
pub fn print(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let a: Vec<String> = e.args().iter().map(doprint).collect();
    format!("TranscendentalRoot({}, {}, {}, {})", a[0], a[1], a[2], a[3])
}

/// The sign of `f` at a rational point (`_sign_at`).
fn sign_at(f: &Basic, x: &Basic, point: &Rat) -> PolyResult<Option<i32>> {
    let value: Basic = number(point);
    if let Some(enclosure) = evaluate(f, &[(x.clone(), (value.clone(), value.clone()))]) {
        if let Some(s) = interval_sign(&enclosure) {
            return Ok(Some(s));
        }
    }
    certified_sign(&crate::core::subs::subs(f, x, &value)?)
}

/// Halve a bracketing interval; `(m, m)` when the midpoint is a root
/// (`_bisect_once`).
fn bisect_once(f: &Basic, x: &Basic, lo: &Rat, hi: &Rat) -> PolyResult<(Rat, Rat)> {
    let Some(left) = sign_at(f, x, lo)? else {
        return Err(value_error(format!("the sign of {} at {} cannot be certified", f, lo)));
    };
    for (p, q) in [(1, 2), (7, 16), (9, 16), (3, 8), (5, 8)] {
        let shift: Rat = Rat::from_parts(Int::from(p), dashu_int::UBig::from(q as u32));
        let m: Rat = lo + &shift * (hi - lo);
        let Some(s) = sign_at(f, x, &m)? else { continue };
        if s == 0 {
            return Ok((m.clone(), m));
        }
        if s == left {
            return Ok((m, hi.clone()));
        }
        return Ok((lo.clone(), m));
    }
    Err(value_error(format!(
        "the sign of {} near the middle of ({}, {}) cannot be certified",
        f, lo, hi
    )))
}

/// The root with its interval bisected `steps` times (`refined`).
pub fn refined(e: &Basic, steps: usize) -> R {
    let (f, x, mut lo, mut hi) = parts(e);
    for _ in 0..steps {
        (lo, hi) = bisect_once(&f, &x, &lo, &hi)?;
        if lo == hi {
            break;
        }
    }
    if lo == hi {
        let tiny: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::ONE << 64usize);
        return root(&f, &x, &(&lo - &tiny), &(&hi + &tiny));
    }
    root(&f, &x, &lo, &hi)
}

/// A rational within `2**-(prec + 4)` of the root (`approximation`).
pub fn approximation(e: &Basic, prec: usize) -> PolyResult<Rat> {
    let (f, x, mut lo, mut hi) = parts(e);
    let width: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::ONE << (prec + 4));
    let _guard = PrecisionGuard::raise_to(prec + 30);
    while &hi - &lo > width {
        (lo, hi) = bisect_once(&f, &x, &lo, &hi)?;
        if lo == hi {
            break;
        }
    }
    Ok((lo + hi) / Rat::from(2))
}

/// The numerical value at `prec` bits (`_numeric_node`).
pub fn numeric_value(e: &Basic, prec: usize) -> Option<crate::core::evalf::Complex> {
    let middle: Rat = approximation(e, prec).ok()?;
    Some(crate::core::evalf::complex_real(
        crate::core::evalf::real_rat(&middle, prec),
        prec,
    ))
}

// -- the comparison of points

fn bounds(p: &Basic) -> (Basic, Basic) {
    if is_root(p) {
        return (p.args()[2].clone(), p.args()[3].clone());
    }
    (p.clone(), p.clone())
}

/// The sign of `u - v` for exact constants, infinities and root objects
/// (`_compare`).
fn compare(u: &Basic, v: &Basic) -> PolyResult<Option<i32>> {
    if is_root(u) || is_root(v) {
        let (mut u, mut v) = (u.clone(), v.clone());
        for _ in 0..80 {
            let (lo_u, hi_u) = bounds(&u);
            let (lo_v, hi_v) = bounds(&v);
            if certified_compare(&lo_u, &hi_v)? == Some(1) {
                return Ok(Some(1));
            }
            if certified_compare(&lo_v, &hi_u)? == Some(1) {
                return Ok(Some(-1));
            }
            if is_root(&u) {
                u = refined(&u, 1)?;
            }
            if is_root(&v) {
                v = refined(&v, 1)?;
            }
        }
        return Ok(None);
    }
    certified_compare(u, v)
}

/// The sign of `u - v` for exact real constants, infinities and root
/// objects (`compare_points`).
pub fn compare_points(u: &Basic, v: &Basic) -> PolyResult<Option<i32>> {
    compare(u, v)
}

/// A rational number strictly between `p < q` (`point_between`).
pub fn point_between(p: &Basic, q: &Basic) -> PolyResult<Option<Basic>> {
    let (mut p, mut q) = (p.clone(), q.clone());
    let mut separated: Option<(Basic, Basic)> = None;
    for _ in 0..80 {
        let (lo, hi) = (bounds(&p).1, bounds(&q).0);
        if certified_compare(&lo, &hi)? == Some(-1) {
            separated = Some((lo, hi));
            break;
        }
        if is_root(&p) {
            p = refined(&p, 1)?;
        }
        if is_root(&q) {
            q = refined(&q, 1)?;
        }
    }
    let Some((lo, hi)) = separated else { return Ok(None) };
    if is_root(&p) && is_root(&q) {
        return Ok(Some(number(&((rat_of(&lo) + rat_of(&hi)) / Rat::from(2)))));
    }
    Ok(between(&lo, &hi)?.filter(|m| m.is_rational_class()))
}

fn strictly_inside(r: &Basic, a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(compare(r, a)? == Some(1) && compare(b, r)? == Some(1))
}

// -- the isolation

fn floor_of(v: &Rat) -> Int {
    let r: Rat = crate::assumptions::analysis::round_nearest(v, 53);
    let (n, d) = (numer(&r), Int::from(denom(&r)));
    let q: Int = &n / &d;
    if rat_is_negative(&r) && Rat::from(q.clone()) != r {
        q - Int::ONE
    } else {
        q
    }
}

fn ceiling_of(v: &Rat) -> Int {
    let r: Rat = crate::assumptions::analysis::round_nearest(v, 53);
    let f: Int = floor_of(&r);
    if Rat::from(f.clone()) == r {
        f
    } else {
        f + Int::ONE
    }
}

fn infinite(p: &Basic) -> bool {
    matches!(p.head(), Head::Infinity | Head::NegativeInfinity)
}

fn sign_near(f: &Basic, x: &Basic, point: &Basic, direction: &str) -> PolyResult<(Option<i32>, Basic)> {
    if infinite(point) {
        let value: Option<Basic> = endpoint_value(f, x, point, direction)?;
        return Ok((
            match value {
                Some(v) => certified_sign(&v)?,
                None => None,
            },
            point.clone(),
        ));
    }
    if is_root(point) {
        let mut current: Basic = point.clone();
        for step in 0..80 {
            let (_, _, lo, hi) = parts(&current);
            let Some(enclosure) = evaluate(f, &[(x.clone(), (number(&lo), number(&hi)))]) else {
                return Ok((None, current));
            };
            if let Some(s) = interval_sign(&enclosure) {
                return Ok((Some(s), current));
            }
            if matches!(step, 6 | 16 | 40) {
                if let Some(exact) = exact_zero_inside(f, x, &lo, &hi)? {
                    return Ok((Some(0), number(&exact)));
                }
            }
            current = refined(&current, 1)?;
        }
        return Ok((None, current));
    }
    let value: Basic = crate::core::subs::subs(f, x, point)?;
    if is(&value, Predicate::finite) != Some(false) {
        if let Some(s) = certified_sign(&value)? {
            return Ok((Some(s), point.clone()));
        }
    }
    let limit_value: Option<Basic> = endpoint_value(f, x, point, direction)?;
    Ok((
        match limit_value {
            Some(v) => certified_sign(&v)?,
            None => None,
        },
        point.clone(),
    ))
}

fn exact_zero_inside(f: &Basic, x: &Basic, lo: &Rat, hi: &Rat) -> PolyResult<Option<Rat>> {
    for k in 0..10usize {
        let scale: Rat = pow2(k);
        let (first, last) = (ceiling_of(&(lo * &scale)), floor_of(&(hi * &scale)));
        if &last - &first > Int::from(8u8) {
            return Ok(None);
        }
        let mut j: Int = first;
        while j <= last {
            let point: Rat = Rat::from(j.clone()) / &scale;
            if sign_at(f, x, &point)? == Some(0) {
                return Ok(Some(point));
            }
            j += Int::ONE;
        }
    }
    Ok(None)
}

fn approach(
    f: &Basic,
    x: &Basic,
    point: &Basic,
    other: &Basic,
    wanted: i32,
    from_right: bool,
) -> PolyResult<Option<Rat>> {
    for k in 0..60usize {
        let Some(candidate) = candidate(point, other, k, from_right)? else {
            return Ok(None);
        };
        if sign_at(f, x, &candidate)? == Some(wanted) {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn rational_near(value: &Basic, above: bool) -> PolyResult<Rat> {
    let digits: usize = crate::settings::settings().precision;
    let exact: Rat = match value.as_rational() {
        Some(r) => r.clone(),
        None => crate::assumptions::analysis::evalf_rational(value, digits)
            .ok_or_else(|| PolyError::Type(format!("{} is not a real number", value)))?,
    };
    let tolerance: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(10u8).pow(digits - 2));
    Ok(if above { exact + tolerance } else { exact - tolerance })
}

fn candidate(point: &Basic, other: &Basic, k: usize, from_right: bool) -> PolyResult<Option<Rat>> {
    if infinite(point) {
        let start: Rat = if infinite(other) {
            Rat::ZERO
        } else if is_root(other) {
            let (_, _, lo, hi) = parts(other);
            if from_right {
                lo
            } else {
                hi
            }
        } else {
            rational_near(other, !from_right)?
        };
        let step: Rat = pow2(k);
        return Ok(Some(if matches!(point.head(), Head::NegativeInfinity) {
            start - step
        } else {
            start + step
        }));
    }
    if is_root(point) {
        let r: Basic = refined(point, k)?;
        let (_, _, lo, hi) = parts(&r);
        return Ok(Some(if from_right { hi } else { lo }));
    }
    let base: Rat = match point.as_rational() {
        Some(r) => r.clone(),
        None => rational_near(point, from_right)?,
    };
    let gap: Rat = if infinite(other) {
        Rat::ONE
    } else {
        let distance: Rat = if is_root(other) {
            let (_, _, lo, hi) = parts(other);
            let v: Rat = if from_right { lo } else { hi };
            v - &base
        } else {
            rational_near(&crate::core::add::sub(other, point)?, false)?
        };
        let d: Rat = if rat_is_negative(&distance) {
            -distance
        } else {
            distance
        };
        if d < Rat::ONE {
            d
        } else {
            Rat::ONE
        }
    };
    let offset: Rat = gap / pow2(k + 1);
    let c: Rat = if from_right { &base + &offset } else { &base - &offset };
    let inside: bool = certified_compare(&number(&c), point)? == Some(if from_right { 1 } else { -1 });
    Ok(if inside { Some(c) } else { None })
}

fn isolate_between(f: &Basic, x: &Basic, p: &Basic, q: &Basic, sp: i32, sq: i32) -> PolyResult<Option<Basic>> {
    let (Some(lo), Some(hi)) = (approach(f, x, p, q, sp, true)?, approach(f, x, q, p, sq, false)?) else {
        return Ok(None);
    };
    if lo >= hi {
        return Ok(None);
    }
    let (mut lo, mut hi) = dyadic_bracket(f, x, &lo, &hi, p, q, sp, sq)?;
    let eighth: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(8u8));
    for _ in 0..64 {
        if &hi - &lo <= eighth {
            break;
        }
        (lo, hi) = bisect_once(f, x, &lo, &hi)?;
        if lo == hi {
            return Ok(Some(number(&lo)));
        }
    }
    Ok(Some(root(f, x, &lo, &hi)?))
}

#[allow(clippy::too_many_arguments)]
fn dyadic_bracket(
    f: &Basic,
    x: &Basic,
    lo: &Rat,
    hi: &Rat,
    p: &Basic,
    q: &Basic,
    sp: i32,
    sq: i32,
) -> PolyResult<(Rat, Rat)> {
    for k in 0..400usize {
        let scale: Rat = pow2(k);
        let l: Rat = Rat::from(floor_of(&(lo * &scale))) / &scale;
        let h: Rat = Rat::from(ceiling_of(&(hi * &scale))) / &scale;
        if l >= h {
            continue;
        }
        if !(strictly_inside(&number(&l), p, q)? && strictly_inside(&number(&h), p, q)?) {
            continue;
        }
        if sign_at(f, x, &l)? == Some(sp) && sign_at(f, x, &h)? == Some(sq) {
            return Ok((l, h));
        }
    }
    Ok((lo.clone(), hi.clone()))
}

/// The real roots of a polynomial over the rationals with their
/// multiplicities, in increasing order (`_real_roots`).
fn real_roots(poly: &crate::polys::polytools::Poly, name: &str) -> PolyResult<Vec<Basic>> {
    let (coefficient, reduced) = crate::polys::polyroots::preprocess_roots(poly)?;
    let (_, factors) = reduced.factor_list()?;
    let mut found: Vec<(crate::polys::cad::samplepoints::Real, Basic, usize)> = Vec::new();
    for (factor, multiplicity) in factors {
        if factor.degree(0)? < 1 {
            continue;
        }
        let trivial: Option<Vec<Basic>> = crate::polys::roots::trivial_roots(&factor)?;
        for (index, (a, _)) in crate::polys::cad::samplepoints::rational_real_roots(&factor)?
            .into_iter()
            .enumerate()
        {
            let r: Basic = match &trivial {
                None => crate::polys::cad::indexedroot::real_root(&a, name)?,
                Some(t) => t[index].clone(),
            };
            found.push((a, r, multiplicity));
        }
    }
    // the order of the real algebraic numbers (a stable insertion sort)
    let mut ordered: Vec<(crate::polys::cad::samplepoints::Real, Basic, usize)> = Vec::new();
    for entry in found {
        let mut position: usize = ordered.len();
        for (i, other) in ordered.iter().enumerate() {
            if crate::polys::cad::samplepoints::compare_real(&entry.0, &other.0)? < 0 {
                position = i;
                break;
            }
        }
        ordered.insert(position, entry);
    }
    let mut result: Vec<Basic> = Vec::new();
    for (_, r, m) in ordered {
        for _ in 0..m {
            result.push(crate::core::mul::mul(&[coefficient.clone(), r.clone()])?);
        }
    }
    Ok(result)
}

fn polynomial_roots(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let poly = match crate::polys::polytools::Poly::new(f, std::slice::from_ref(x)) {
        Ok(p) => p,
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if !poly.coeffs()?.iter().all(|c| c.is_rational_class()) {
        return Ok(None);
    }
    if poly.degree(0)? <= 0 {
        return Ok(if poly.is_zero() { None } else { Some(vec![]) });
    }
    let name: String = match x.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => x.to_string(),
    };
    let mut inside: Vec<Basic> = Vec::new();
    for r in real_roots(&poly, &name)? {
        if strictly_inside(&r, a, b)? {
            inside.push(r);
        }
    }
    Ok(Some(inside))
}

fn exact_zeros(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let outer: Basic = interval(&bounds(a).0, &bounds(b).1, true, true)?;
    let Some(zeros) = finite_zeros(f, x, &outer)? else {
        return Ok(None);
    };
    if !zeros.iter().all(|z| yes(z, Predicate::real)) || zeros.iter().any(|z| !z.free_symbols().is_empty()) {
        return Ok(None);
    }
    let mut inside: Vec<Basic> = Vec::new();
    for z in zeros {
        if strictly_inside(&z, a, b)? {
            inside.push(z);
        }
    }
    sorted_constants(&inside)
}

fn monotone_on(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
    if is_root(a) || is_root(b) {
        return Ok(None);
    }
    let piece: Basic = interval(a, b, true, true)?;
    if !matches!(piece.head(), Head::Interval | Head::Reals) {
        return Ok(None);
    }
    let derivative: Basic = crate::core::function::derivative(f, x)?;
    let Some(signs) = sign_on(&derivative, x, &piece, 0)? else {
        return Ok(None);
    };
    if (signs.contains(&1) && signs.contains(&-1)) || signs == [0].into_iter().collect() {
        return Ok(None);
    }
    if signs.contains(&0) && !has_discrete_zeros(&derivative, x, &piece)? {
        return Ok(None);
    }
    Ok(Some(true))
}

fn roots_between(f: &Basic, x: &Basic, a: &Basic, b: &Basic, depth: usize) -> PolyResult<Option<Vec<Basic>>> {
    if !f.has(x) {
        let s: Option<i32> = certified_sign(f)?;
        return Ok(if s.is_some_and(|v| v != 0) { Some(vec![]) } else { None });
    }
    if let Some(p) = polynomial_roots(f, x, a, b)? {
        return Ok(Some(p));
    }
    if let Some(e) = exact_zeros(f, x, a, b)? {
        return Ok(Some(e));
    }
    if depth == 0 {
        return Ok(None);
    }
    let derivative: Basic = crate::core::function::derivative(f, x)?;
    let critical: Vec<Basic> = match roots_between(&derivative, x, a, b, depth - 1)? {
        Some(c) => c,
        None => {
            if monotone_on(f, x, a, b)? != Some(true) {
                return Ok(None);
            }
            vec![]
        }
    };
    let mut points: Vec<Basic> = vec![a.clone()];
    points.extend(critical);
    points.push(b.clone());
    let mut result: Vec<Basic> = Vec::new();
    for i in 0..points.len() - 1 {
        let (sp, refined_p) = sign_near(f, x, &points[i], "+")?;
        points[i] = refined_p;
        let (sq, refined_q) = sign_near(f, x, &points[i + 1], "-")?;
        points[i + 1] = refined_q;
        let (Some(sp), Some(sq)) = (sp, sq) else {
            return Ok(None);
        };
        if i > 0 && sp == 0 {
            result.push(points[i].clone());
        }
        if sp * sq < 0 {
            let Some(r) = isolate_between(f, x, &points[i], &points[i + 1], sp, sq)? else {
                return Ok(None);
            };
            result.push(r);
        }
    }
    Ok(Some(result))
}

/// The real roots of `f` on `domain`, each exact or a root object, in
/// increasing order; `None` when they cannot be isolated
/// (`isolate_real_roots`).
pub fn isolate_real_roots(f: &Basic, x: &Basic, domain: &Basic, depth: usize) -> PolyResult<Option<Vec<Basic>>> {
    if f.free_symbols().iter().any(|s| s != x) {
        return Err(value_error(format!("{} has parameters besides {}", f, x)));
    }
    let Some(ps) = pieces(domain) else { return Ok(None) };
    let mut result: Vec<Basic> = Vec::new();
    for piece in ps {
        if !continuous_on(f, x, &piece)? {
            return Ok(None);
        }
        let (a, b, left_open, right_open) = interval_parts(&piece);
        if !left_open && yes(&a, Predicate::finite) && certified_sign(&crate::core::subs::subs(f, x, &a)?)? == Some(0) {
            result.push(a.clone());
        }
        let mut inside: Option<Vec<Basic>> = roots_between(f, x, &a, &b, depth)?;
        if inside.is_none() && !(yes(&a, Predicate::finite) && yes(&b, Predicate::finite)) {
            inside = roots_with_bounded_tails(f, x, &a, &b, depth)?;
        }
        let Some(inside) = inside else { return Ok(None) };
        result.extend(inside);
        if !right_open && yes(&b, Predicate::finite) && certified_sign(&crate::core::subs::subs(f, x, &b)?)? == Some(0)
        {
            result.push(b.clone());
        }
    }
    Ok(Some(result))
}

fn tail_sign(f: &Basic, x: &Basic, tail: &Basic) -> PolyResult<Option<i32>> {
    let Some(signs) = sign_on(f, x, tail, 0)? else {
        return Ok(None);
    };
    if signs.len() != 1 || signs.contains(&0) {
        return Ok(None);
    }
    Ok(signs.into_iter().next())
}

fn roots_with_bounded_tails(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    depth: usize,
) -> PolyResult<Option<Vec<Basic>>> {
    let (mut lo, mut hi) = (a.clone(), b.clone());
    for exponent in 0..8u32 {
        let r: Basic = Basic::int(1i64 << exponent);
        let nr: Basic = crate::core::mul::neg(&r)?;
        if matches!(lo.head(), Head::NegativeInfinity)
            && tail_sign(f, x, &interval(&Basic::negative_oo(), &nr, true, true)?)?.is_some()
        {
            lo = nr.clone();
        }
        if matches!(hi.head(), Head::Infinity) && tail_sign(f, x, &interval(&r, &Basic::oo(), true, true)?)?.is_some() {
            hi = r.clone();
        }
        if !matches!(lo.head(), Head::NegativeInfinity) && !matches!(hi.head(), Head::Infinity) {
            break;
        }
    }
    if infinite(&lo) || infinite(&hi) {
        return Ok(None);
    }
    if compare(&lo, &hi)? != Some(-1) {
        return Ok(None);
    }
    roots_between(f, x, &lo, &hi, depth)
}

/// The roots of [`isolate_real_roots`] as a set (`real_roots_of`).
pub fn real_roots_of(f: &Basic, x: &Basic, domain: &Basic, depth: usize) -> PolyResult<Option<Basic>> {
    Ok(isolate_real_roots(f, x, domain, depth)?.map(|r| crate::sets::sets::finite_set(&r)))
}
