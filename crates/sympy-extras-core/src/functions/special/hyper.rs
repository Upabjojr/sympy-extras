//! The generalised hypergeometric function and the Meijer G-function,
//! ported from `sympy_extras.functions.special.hyper`: the nodes
//! [`Head::Hyper`] (`hyper(Tuple(ap), Tuple(bq), z)`) and
//! [`Head::MeijerG`] (`meijerg(Tuple(Tuple(an), Tuple(aother)),
//! Tuple(Tuple(bm), Tuple(bother)), z)`), built as SymPy's constructors
//! build them (`hyper` cancels the parameters common to `ap` and `bq` and
//! orders them; `meijerg` orders its four lists and refuses infinite
//! parameters and an `a_i - b_j` that is a positive integer); the
//! derivatives (in the argument, and `meijerg`'s in the parameters when
//! they pair up by integer differences, else `Derivative`), `hyper`'s
//! `radius_of_convergence`, `eta` and `convergence_statement`, `meijerg`'s
//! `nu`, `delta` and `get_period`; Appell's `appellf1` (a function head).
//!
//! Values: `hyper` by its defining series where it converges (`p <= q`
//! for a moderate argument, `p = q + 1` for `|z| < 3/4`, a terminating
//! series); mpmath's continuations, `meijerg` and `appellf1` values are not
//! ported (no value). Not ported either: the series (`nseries`,
//! `as_leading_term`), `hyperexpand`, and the `simplify` of the
//! parameter differences (`sub` only: the differences of polynomial
//! parameters are already canonical).

// PORTED FROM SYMPY: sympy/functions/special/hyper.py (hyper, meijerg, _prep_tuple,
// TupleParametersBase._eval_derivative, meijerg.nu, delta, get_period, appellf1)
// (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::{add, sub};
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, parts, real_int, Complex, Real};
use crate::core::expr::is_true;
use crate::core::mul::{div, mul};
use crate::core::parameters::evaluate;
use crate::core::sorting::sort_key;
use crate::error::{value_error, PolyError, PolyResult};
use crate::numbers::{denom, lcm, numer, Int};

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn tuple(items: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, items)
}

fn ordered(mut p: Vec<Basic>) -> Vec<Basic> {
    p.sort_by_cached_key(sort_key);
    p
}

/// Whether `e` is a `Rational` node (`isinstance(e, Rational)`).
fn rational_of(e: &Basic) -> Option<crate::numbers::Rat> {
    if e.is_number_node() {
        e.as_rational().cloned()
    } else {
        None
    }
}

/// The integer of an `Integer` node.
fn integer_of(e: &Basic) -> Option<Int> {
    rational_of(e).filter(|v| denom(v) == Int::ONE).map(|v| numer(&v))
}

/// `isinstance(e, Integer)`.
fn is_integer_node(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}

fn depends(parameters: &[Basic], s: &Basic) -> bool {
    parameters.iter().any(|a| a.free_symbols().contains(s))
}

// ---------------------------------------------------------------------
// hyper
// ---------------------------------------------------------------------

/// The unevaluated node `Hyper(ap, bq, z)`.
pub fn hyper_node(ap: Vec<Basic>, bq: Vec<Basic>, z: Basic) -> Basic {
    Basic::new(Head::Hyper, vec![tuple(ap), tuple(bq), z])
}

/// `ap` of a `Hyper` node.
pub fn hyper_ap(e: &Basic) -> &[Basic] {
    e.args()[0].args()
}

/// `bq` of a `Hyper` node.
pub fn hyper_bq(e: &Basic) -> &[Basic] {
    e.args()[1].args()
}

/// `hyper(ap, bq, z)` as SymPy builds it: the parameters common to `ap`
/// and `bq` cancelled, both lists ordered.
pub fn hyper(ap: &[Basic], bq: &[Basic], z: &Basic) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(hyper_node(ordered(ap.to_vec()), ordered(bq.to_vec()), z.clone()));
    }
    let mut top: Vec<Basic> = ap.to_vec();
    let mut bottom: Vec<Basic> = Vec::new();
    for b in bq {
        if let Some(k) = top.iter().position(|a| a == b) {
            top.remove(k);
        } else {
            bottom.push(b.clone());
        }
    }
    let (top, bottom) = (ordered(top), ordered(bottom));
    let z: Basic = unpolarified_argument(&top, &bottom, z)?;
    Ok(hyper_node(top, bottom, z))
}

/// The argument without its polar numbers where the series is entire
/// (`p <= q`) or converges on `|z| <= 1` (`p = q + 1`).
fn unpolarified_argument(ap: &[Basic], bq: &[Basic], z: &Basic) -> PolyResult<Basic> {
    use crate::core::relational::le;
    use crate::functions::elementary::complexes::abs_;
    if ap.len() <= bq.len()
        || (ap.len() == bq.len() + 1 && matches!(le(&abs_(z)?, &Basic::one())?.head(), Head::BooleanTrue))
    {
        return crate::core::function::unpolarify(z);
    }
    Ok(z.clone())
}

/// The derivative in the argument: `prod(ap)/prod(bq)*hyper(ap + 1, bq + 1, z)`.
pub fn hyper_derivative_in_argument(e: &Basic) -> PolyResult<Basic> {
    let (ap, bq, z) = (hyper_ap(e), hyper_bq(e), &e.args()[2]);
    let factor: Basic = div(&mul(ap)?, &mul(bq)?)?;
    let raise = |p: &[Basic]| -> PolyResult<Vec<Basic>> { p.iter().map(|a| add(&[a.clone(), Basic::one()])).collect() };
    mul(&[factor, hyper(&raise(ap)?, &raise(bq)?, z)?])
}

/// `sum(ap) - sum(bq)`.
pub fn hyper_eta(e: &Basic) -> PolyResult<Basic> {
    sub(&add(hyper_ap(e))?, &add(hyper_bq(e))?)
}

/// The radius of convergence of the defining series: 0, 1 or oo.
pub fn radius_of_convergence(e: &Basic) -> PolyResult<Basic> {
    let (ap, bq) = (hyper_ap(e), hyper_bq(e));
    let nonpositive = |a: &Basic| -> Option<Int> { integer_of(a).filter(|v| *v <= Int::ZERO) };
    if ap
        .iter()
        .chain(bq)
        .any(|a| is_true(a, Predicate::integer) && nonpositive(a).is_some())
    {
        let mut aints: Vec<Int> = ap.iter().filter_map(nonpositive).collect();
        let bints: Vec<Int> = bq.iter().filter_map(nonpositive).collect();
        if aints.len() < bints.len() {
            return Ok(Basic::zero());
        }
        let mut popped: bool = false;
        for b in &bints {
            let mut cancelled: bool = false;
            while let Some(a) = aints.pop() {
                if a >= *b {
                    cancelled = true;
                    break;
                }
                popped = true;
            }
            if !cancelled {
                return Ok(Basic::zero());
            }
        }
        if !aints.is_empty() || popped {
            return Ok(Basic::oo());
        }
    }
    if ap.len() == bq.len() + 1 {
        return Ok(Basic::one());
    }
    if ap.len() <= bq.len() {
        return Ok(Basic::oo());
    }
    Ok(Basic::zero())
}

/// The condition under which the series converges (`false` for a zero
/// radius, `true` for an infinite one).
pub fn convergence_statement(e: &Basic) -> PolyResult<Basic> {
    use crate::core::logic::{and_, false_, or_, true_};
    use crate::core::relational::{ge, le, lt, ne};
    use crate::functions::elementary::complexes::{abs_, re};
    let r: Basic = radius_of_convergence(e)?;
    if r == Basic::zero() {
        return Ok(false_());
    }
    if r == Basic::oo() {
        return Ok(true_());
    }
    let one: Basic = Basic::one();
    let re_eta: Basic = re(&hyper_eta(e)?)?;
    let size: Basic = abs_(&e.args()[2])?;
    let c1: Basic = and_(&[lt(&re_eta, &Basic::zero())?, le(&size, &one)?]);
    let c2: Basic = and_(&[
        ge(&re_eta, &Basic::zero())?,
        lt(&re_eta, &one)?,
        le(&size, &one)?,
        ne(&e.args()[2], &one)?,
    ]);
    let c3: Basic = and_(&[ge(&re_eta, &one)?, lt(&size, &one)?]);
    Ok(or_(&[c1, c2, c3]))
}

// ---------------------------------------------------------------------
// meijerg
// ---------------------------------------------------------------------

/// The four parameter lists of a `MeijerG` node: `an`, `aother`, `bm`,
/// `bother`.
pub fn meijerg_lists(e: &Basic) -> [&[Basic]; 4] {
    let (first, second) = (&e.args()[0], &e.args()[1]);
    [
        first.args()[0].args(),
        first.args()[1].args(),
        second.args()[0].args(),
        second.args()[1].args(),
    ]
}

/// The unevaluated node `MeijerG(an, aother, bm, bother, z)`.
pub fn meijerg_node(lists: [Vec<Basic>; 4], z: Basic) -> Basic {
    let [an, aother, bm, bother] = lists;
    Basic::new(
        Head::MeijerG,
        vec![
            tuple(vec![tuple(an), tuple(aother)]),
            tuple(vec![tuple(bm), tuple(bother)]),
            z,
        ],
    )
}

/// `meijerg(an, aother, bm, bother, z)` as SymPy builds it: each list
/// ordered; infinite parameters and an `a_i - b_j` (over `an` and `bm`)
/// that is a positive integer refused.
pub fn meijerg(an: &[Basic], aother: &[Basic], bm: &[Basic], bother: &[Basic], z: &Basic) -> PolyResult<Basic> {
    let lists: [Vec<Basic>; 4] = [
        ordered(an.to_vec()),
        ordered(aother.to_vec()),
        ordered(bm.to_vec()),
        ordered(bother.to_vec()),
    ];
    if lists.iter().flatten().any(|a| {
        matches!(
            a.head(),
            Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity
        )
    }) {
        return Err(value_error("G-function parameters must be finite".to_string()));
    }
    for a in &lists[0] {
        for b in &lists[2] {
            if integer_of(&sub(a, b)?).is_some_and(|d| d > Int::ZERO) {
                return Err(value_error(
                    "no parameter a1, ..., an may differ from any b1, ..., bm by a positive integer".to_string(),
                ));
            }
        }
    }
    Ok(meijerg_node(lists, z.clone()))
}

/// `sum(bq) - sum(ap)`.
pub fn meijerg_nu(e: &Basic) -> PolyResult<Basic> {
    let [an, aother, bm, bother] = meijerg_lists(e);
    let mut total: Basic = Basic::zero();
    for b in bm.iter().chain(bother) {
        total = add(&[total, b.clone()])?;
    }
    for a in an.iter().chain(aother) {
        total = sub(&total, a)?;
    }
    Ok(total)
}

/// `m + n - (p + q)/2`.
pub fn meijerg_delta(e: &Basic) -> PolyResult<Basic> {
    let [an, aother, bm, bother] = meijerg_lists(e);
    let (p, q) = (an.len() + aother.len(), bm.len() + bother.len());
    let twice: i64 = 2 * (bm.len() + an.len()) as i64 - p as i64 - q as i64;
    Basic::rational(Int::from(twice), Int::from(2))
}

/// The period of the function in `log(z)`.
pub fn meijerg_get_period(e: &Basic) -> PolyResult<Basic> {
    use crate::core::modulo::modulo;
    // None for SymPy's oo: a parameter not rational, or two differing by an
    // integer (only rational differences reach the test: simplify is not needed)
    let compute = |parameters: &[Basic]| -> PolyResult<Option<Int>> {
        for (i, b) in parameters.iter().enumerate() {
            if rational_of(b).is_none() {
                return Ok(None);
            }
            for other in &parameters[i + 1..] {
                let r: Basic = modulo(&sub(b, other)?, &Basic::one())?;
                if integer_of(&r).is_some_and(|v| v == Int::ZERO) && is_integer_node(&r) {
                    return Ok(None);
                }
            }
        }
        let mut period: Int = Int::ONE;
        for b in parameters {
            if let Some(v) = rational_of(b) {
                period = lcm(&period, &denom(&v));
            }
        }
        Ok(Some(period))
    };
    let [an, aother, bm, bother] = meijerg_lists(e);
    let beta: Option<Int> = compute(bm)?;
    let alpha: Option<Int> = compute(an)?;
    let (p, q) = (an.len() + aother.len(), bm.len() + bother.len());
    let period = |k: Int| -> PolyResult<Basic> { mul(&[int(2), mul(&[Basic::pi(), Basic::integer(k)])?]) };
    if p == q {
        return match (alpha, beta) {
            (Some(a), Some(b)) => period(lcm(&a, &b)),
            _ => Ok(Basic::oo()),
        };
    }
    let chosen: Option<Int> = if p < q { beta } else { alpha };
    match chosen {
        Some(k) => period(k),
        None => Ok(Basic::oo()),
    }
}

/// The derivative of a `MeijerG` node in its argument.
pub fn meijerg_derivative_in_argument(e: &Basic) -> PolyResult<Basic> {
    let [an, aother, bm, bother] = meijerg_lists(e);
    let z: &Basic = &e.args()[2];
    let one: Basic = Basic::one();
    let inverse: Basic = div(&one, z)?;
    if !an.is_empty() {
        let first: Basic = sub(&an[0], &one)?;
        let mut lowered: Vec<Basic> = vec![first.clone()];
        lowered.extend(an[1..].iter().cloned());
        let g: Basic = meijerg(&lowered, aother, bm, bother, z)?;
        return mul(&[inverse, add(&[mul(&[first, e.clone()])?, g])?]);
    }
    if !bm.is_empty() {
        let mut raised: Vec<Basic> = vec![add(&[bm[0].clone(), one])?];
        raised.extend(bm[1..].iter().cloned());
        let g: Basic = meijerg(an, aother, &raised, bother, z)?;
        return mul(&[inverse, sub(&mul(&[bm[0].clone(), e.clone()])?, &g)?]);
    }
    Ok(Basic::zero())
}

/// The derivative in the parameter `idx` (over `an`, `aother`, `bm`,
/// `bother`), when the other parameters pair up by integer differences;
/// `NotImplemented` otherwise.
fn diff_wrt_parameter(e: &Basic, idx: usize) -> PolyResult<Basic> {
    use crate::core::modulo::modulo;
    use crate::functions::elementary::exponential::log;
    let lists: [&[Basic]; 4] = meijerg_lists(e);
    let z: &Basic = &e.args()[2];
    let mut work: [Vec<Basic>; 4] = [
        lists[0].to_vec(),
        lists[1].to_vec(),
        lists[2].to_vec(),
        lists[3].to_vec(),
    ];
    let mut k: usize = idx;
    for list in work.iter_mut() {
        if k < list.len() {
            list.remove(k);
            break;
        }
        k -= list.len();
    }
    let [mut an, mut ap, mut bm, mut bq] = work;
    let mut pairs1: Vec<(Basic, Basic)> = Vec::new();
    let mut pairs2: Vec<(Basic, Basic)> = Vec::new();
    for (l1, l2, pairs) in [(&mut an, &mut bq, &mut pairs1), (&mut ap, &mut bm, &mut pairs2)] {
        while let Some(x) = l1.pop() {
            let mut found: Option<usize> = None;
            for (i, y) in l2.iter().enumerate() {
                let r: Basic = modulo(&sub(&x, y)?, &Basic::one())?;
                if r.is_number_class() && r == Basic::zero() {
                    found = Some(i);
                    break;
                }
            }
            let Some(i) = found else {
                return Err(PolyError::NotImplemented(
                    "Derivative not expressible as G-function?".to_string(),
                ));
            };
            pairs.push((x, l2.remove(i)));
        }
    }
    let count = |d: &Basic| -> PolyResult<i64> {
        integer_of(d)
            .filter(|_| is_integer_node(d))
            .and_then(|v| i64::try_from(&v).ok())
            .ok_or_else(|| PolyError::Type(format!("{} cannot be interpreted as an integer", d)))
    };
    let mut res: Basic = mul(&[log(z)?, e.clone()])?;
    for (a, b) in &pairs1 {
        let (mut sign, mut n, mut base) = (1i64, count(&sub(a, b)?)?, b.clone());
        if n < 0 {
            (sign, n, base) = (-1, -n, a.clone());
        }
        for k in 0..n {
            let mut new_an: Vec<Basic> = lists[0].to_vec();
            new_an.push(add(&[add(&[base.clone(), int(k)])?, Basic::one()])?);
            let mut new_bother: Vec<Basic> = lists[3].to_vec();
            new_bother.push(add(&[base.clone(), int(k)])?);
            let g: Basic = meijerg(&new_an, lists[1], lists[2], &new_bother, z)?;
            res = sub(&res, &mul(&[int(sign), g])?)?;
        }
    }
    for (a, b) in &pairs2 {
        let (mut sign, mut n, mut base) = (1i64, count(&sub(b, a)?)?, a.clone());
        if n < 0 {
            (sign, n, base) = (-1, -n, b.clone());
        }
        for k in 0..n {
            let mut new_aother: Vec<Basic> = lists[1].to_vec();
            new_aother.push(add(&[add(&[base.clone(), int(k)])?, Basic::one()])?);
            let mut new_bm: Vec<Basic> = lists[2].to_vec();
            new_bm.push(add(&[base.clone(), int(k)])?);
            let g: Basic = meijerg(lists[0], &new_aother, &new_bm, lists[3], z)?;
            res = sub(&res, &mul(&[int(sign), g])?)?;
        }
    }
    Ok(res)
}

/// The derivative of a `Hyper` or `MeijerG` node in `s`.
pub fn hyper_derivative(e: &Basic, s: &Basic) -> PolyResult<Basic> {
    use crate::core::function::{derivative, derivative_node};
    let z: &Basic = &e.args()[2];
    if matches!(e.head(), Head::Hyper) {
        let parameters: Vec<Basic> = hyper_ap(e).iter().chain(hyper_bq(e)).cloned().collect();
        if depends(&parameters, s) {
            return Ok(derivative_node(e, &[(s.clone(), 1)]));
        }
        return mul(&[hyper_derivative_in_argument(e)?, derivative(z, s)?]);
    }
    let parameters: Vec<Basic> = meijerg_lists(e).iter().flat_map(|l| l.iter().cloned()).collect();
    if !depends(&parameters, s) {
        return mul(&[meijerg_derivative_in_argument(e)?, derivative(z, s)?]);
    }
    let attempt = || -> PolyResult<Basic> {
        let mut res: Basic = Basic::zero();
        for (i, p) in parameters.iter().enumerate() {
            let m: Basic = derivative(p, s)?;
            if !(m.is_number_class() && m == Basic::zero()) {
                res = add(&[res, mul(&[diff_wrt_parameter(e, i)?, m])?])?;
            }
        }
        add(&[res, mul(&[meijerg_derivative_in_argument(e)?, derivative(z, s)?])?])
    };
    match attempt() {
        Err(PolyError::NotImplemented(_)) => Ok(derivative_node(e, &[(s.clone(), 1)])),
        other => other,
    }
}

/// The node rebuilt from new arguments (`_evaluated`).
pub fn rebuild_hyper(head: &Head, args: &[Basic]) -> PolyResult<Basic> {
    let all_tuples = |a: &Basic| matches!(a.head(), Head::Tuple);
    if matches!(head, Head::Hyper) {
        if !(all_tuples(&args[0]) && all_tuples(&args[1])) {
            return Err(PolyError::Type(
                "hyper takes two parameter tuples and an expression".to_string(),
            ));
        }
        return hyper(args[0].args(), args[1].args(), &args[2]);
    }
    let (first, second) = (&args[0], &args[1]);
    if !(all_tuples(first) && all_tuples(second)) {
        return Err(PolyError::Type(
            "meijerg takes two pairs of parameter tuples and an expression".to_string(),
        ));
    }
    let lists: Vec<&Basic> = vec![&first.args()[0], &first.args()[1], &second.args()[0], &second.args()[1]];
    if !lists.iter().all(|t| all_tuples(t)) {
        return Err(PolyError::Type(
            "meijerg takes two pairs of parameter tuples".to_string(),
        ));
    }
    meijerg(
        lists[0].args(),
        lists[1].args(),
        lists[2].args(),
        lists[3].args(),
        &args[2],
    )
}

/// The facts: commutative (not with a noncommutative argument, as
/// `facts_of` of a function).
pub fn hyper_facts(e: &Basic) -> FactList {
    let mut facts: FactList = FactList::new();
    let noncommutative: bool = crate::core::assumptions::facts_of(&e.args()[2])
        .get(&Predicate::commutative)
        .is_some_and(|v| !*v);
    facts.set(Predicate::commutative, !noncommutative);
    facts
}

/// The printed node: `hyper((a, b), (c,), z)`, `meijerg(((an), (aother)),
/// ((bm), (bother)), z)`.
pub fn print_hyper(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let name: &str = if matches!(e.head(), Head::Hyper) {
        "hyper"
    } else {
        "meijerg"
    };
    format!(
        "{}({}, {}, {})",
        name,
        doprint(&e.args()[0]),
        doprint(&e.args()[1]),
        doprint(&e.args()[2])
    )
}

// ---------------------------------------------------------------------
// values
// ---------------------------------------------------------------------

fn modulus(w: &Complex) -> f64 {
    let (wr, wi) = parts(w);
    let (a, b) = (wr.to_f64().value(), wi.to_f64().value());
    (a * a + b * b).sqrt()
}

fn raise_to(z: &Complex, wp: usize) -> Complex {
    let (re, im) = parts(z);
    complex(re.with_precision(wp).value(), im.with_precision(wp).value())
}

fn c_int(n: i64, wp: usize) -> Complex {
    crate::core::evalf::complex_real(real_int(&Int::from(n), wp), wp)
}

/// The nonpositive integer of a value, if it is one.
fn nonpositive_integer(v: &Complex) -> Option<i64> {
    let (re, im) = parts(v);
    if im == Real::ZERO && re <= Real::ZERO && re.clone() == re.floor() {
        return re.to_f64().value().round().to_string().parse::<i64>().ok();
    }
    None
}

/// The sum of the defining series, at the working precision `wp`;
/// `None` when it does not settle. Returns the sum and the log2 of the
/// largest term (for the cancellation).
fn series(ap: &[Complex], bq: &[Complex], z: &Complex, terms: Option<i64>, wp: usize) -> Option<(Complex, f64)> {
    let mut term: Complex = c_int(1, wp);
    let mut total: Complex = term.clone();
    let mut largest: f64 = 0.0;
    let size: f64 = modulus(z);
    let scale: f64 = ap.iter().chain(bq).map(modulus).fold(0.0, f64::max);
    let mut k: i64 = 0;
    loop {
        if terms.is_some_and(|n| k >= n) {
            break;
        }
        let kk: Complex = c_int(k, wp);
        let mut num: Complex = z.clone();
        for a in ap {
            num = &num * &(a + &kk);
        }
        let mut den: Complex = c_int(k + 1, wp);
        for b in bq {
            den = &den * &(b + &kk);
        }
        term = &(&term * &num) / &den;
        total = &total + &term;
        k += 1;
        let t: f64 = modulus(&term);
        if !t.is_finite() {
            return None;
        }
        if t > 0.0 {
            largest = largest.max(t.log2());
        }
        if terms.is_none() {
            let small: bool = t == 0.0 || t.log2() < modulus(&total).log2() - (wp + 10) as f64;
            if small && (k as f64) > 2.0 * scale + 2.0 && (k as f64) > 2.0 * size {
                break;
            }
            if k > 200_000 {
                return None;
            }
        }
    }
    Some((total, largest))
}

/// The value of `hyper(ap, bq, z)` at the values of its parameters and
/// argument, by the defining series; `None` where it is not ported
/// (mpmath's continuations) or undefined.
pub fn hyper_value(ap: &[Complex], bq: &[Complex], z: &Complex, prec: usize) -> Option<Complex> {
    if bq.iter().any(|b| nonpositive_integer(b).is_some()) {
        return None; // poles (or mpmath's cancellations with a terminating ap)
    }
    let terms: Option<i64> = ap.iter().filter_map(nonpositive_integer).max().map(|m| -m + 1);
    let size: f64 = modulus(z);
    let one_f_one: bool = ap.len() == 1 && bq.len() == 1;
    if terms.is_none() && one_f_one && size >= 64.0 {
        // mpmath's _hyp1f1: the asymptotic series for mag(z) >= 7, the
        // convergent one when they do not converge (NoConvergence)
        if let Some(v) = hyp1f1_asymptotic(&ap[0], &bq[0], z, prec) {
            return Some(v);
        }
    }
    if terms.is_none() {
        let fits: bool =
            (ap.len() <= bq.len() && (one_f_one || size <= 40.0)) || (ap.len() == bq.len() + 1 && size < 0.75);
        if !fits {
            return None;
        }
    }
    let real: bool = ap
        .iter()
        .chain(bq)
        .chain(std::iter::once(z))
        .all(|v| parts(v).1 == Real::ZERO);
    let mut wp: usize = prec + 40;
    for _ in 0..4 {
        let lift = |v: &Complex| raise_to(v, wp);
        let (a, b): (Vec<Complex>, Vec<Complex>) = (ap.iter().map(lift).collect(), bq.iter().map(lift).collect());
        let (total, largest) = series(&a, &b, &lift(z), terms, wp)?;
        let size: f64 = modulus(&total);
        let lost: f64 = if size > 0.0 {
            largest - size.log2()
        } else {
            f64::INFINITY
        };
        if lost.is_finite() && lost + ((prec + 20) as f64) < wp as f64 {
            let (re, im) = parts(&total);
            let im: Real = if real {
                real_int(&Int::ZERO, prec)
            } else {
                im.with_precision(prec).value()
            };
            return Some(complex(re.with_precision(prec).value(), im));
        }
        if !lost.is_finite() {
            if terms.is_some() && size == 0.0 {
                return Some(c_int(0, prec)); // an exact zero of a polynomial
            }
            wp *= 2;
        } else {
            wp = prec + 40 + lost.ceil() as usize;
        }
    }
    None
}

/// `2F0(a1, a2; ; x)` as the asymptotic series: summed while the terms
/// decrease, `None` when they grow again before they are negligible.
fn hyp2f0_asymptotic(a1: &Complex, a2: &Complex, x: &Complex, wp: usize, tolerance: usize) -> Option<Complex> {
    let mut term: Complex = c_int(1, wp);
    let mut total: Complex = term.clone();
    let mut previous: f64 = f64::INFINITY;
    let terminating: bool = nonpositive_integer(a1).is_some() || nonpositive_integer(a2).is_some();
    for k in 0..100_000i64 {
        let kk: Complex = c_int(k, wp);
        term = &(&(&(&term * &(a1 + &kk)) * &(a2 + &kk)) * x) / &c_int(k + 1, wp);
        let t: f64 = modulus(&term);
        if t == 0.0 {
            return Some(total);
        }
        let s: f64 = modulus(&total);
        if !terminating && t > previous && (k as f64) > modulus(a1) + modulus(a2) + 2.0 {
            return None;
        }
        total = &total + &term;
        if s > 0.0 && t.log2() < s.log2() - tolerance as f64 {
            return Some(total);
        }
        previous = t;
    }
    None
}

/// `1F1(a; b; z)` for `|z| >= 64` by the two asymptotic series of
/// mpmath's `_hyp1f1` (DLMF 13.7.2):
/// `e^{i pi a} z^{-a} G(b)/G(b-a) 2F0(a, 1+a-b; -1/z) + e^z z^{a-b} G(b)/G(a)
/// 2F0(b-a, 1-a; 1/z)`, `e^{-i pi a}` below the real axis.
fn hyp1f1_asymptotic(a: &Complex, b: &Complex, z: &Complex, prec: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::gamma;
    let magz: usize = modulus(z).log2().ceil() as usize;
    let wp: usize = prec + 60 + magz;
    let tolerance: usize = prec + magz + 10;
    let (a, b, z) = (raise_to(a, wp), raise_to(b, wp), raise_to(z, wp));
    let one: Complex = c_int(1, wp);
    let pi: Real = Real::pi(wp);
    let below: bool = parts(&z).1 < Real::ZERO;
    let i_pi: Complex = complex(real_int(&Int::ZERO, wp), if below { -pi.clone() } else { pi.clone() });
    let power = |w: &Complex, c: &Complex| -> Complex { (c * &w.ln()).exp() };
    let rz: Complex = &one / &z;
    let gamma_b: Complex = gamma(&b, wp)?;
    let b_minus_a: Complex = &b - &a;
    let first: Complex = if nonpositive_integer(&b_minus_a).is_some() {
        c_int(0, wp)
    } else {
        let e: Complex = (&i_pi * &a).exp();
        let neg_a: Complex = &c_int(0, wp) - &a;
        let series: Complex = hyp2f0_asymptotic(&a, &(&(&one + &a) - &b), &(&c_int(0, wp) - &rz), wp, tolerance)?;
        &(&(&e * &power(&z, &neg_a)) * &(&gamma_b / &gamma(&b_minus_a, wp)?)) * &series
    };
    let series: Complex = hyp2f0_asymptotic(&b_minus_a, &(&one - &a), &rz, wp, tolerance)?;
    let second: Complex = &(&(&z.exp() * &power(&z, &(&a - &b))) * &(&gamma_b / &gamma(&a, wp)?)) * &series;
    let total: Complex = &first + &second;
    let real: bool = [&a, &b, &z].iter().all(|v| parts(v).1 == Real::ZERO);
    let (re, im) = parts(&total);
    let im: Real = if real {
        real_int(&Int::ZERO, prec)
    } else {
        im.with_precision(prec).value()
    };
    Some(complex(re.with_precision(prec).value(), im))
}

/// The value of a `Hyper` node.
pub fn hyper_node_value(e: &Basic, prec: usize) -> Option<Complex> {
    use crate::core::evalf::numeric_value;
    let wp: usize = prec + 40;
    let ap: Vec<Complex> = hyper_ap(e)
        .iter()
        .map(|a| numeric_value(a, wp))
        .collect::<Option<_>>()?;
    let bq: Vec<Complex> = hyper_bq(e)
        .iter()
        .map(|b| numeric_value(b, wp))
        .collect::<Option<_>>()?;
    let z: Complex = numeric_value(&e.args()[2], wp)?;
    hyper_value(&ap, &bq, &z, prec)
}

// ---------------------------------------------------------------------
// appellf1
// ---------------------------------------------------------------------

/// `appellf1(a, b1, b2, c, x, y)`, evaluated: the symmetry `(b1, x) <->
/// (b2, y)` put in canonical order, `1` at `x = y = 0`.
pub fn appellf1(args: &[Basic]) -> PolyResult<Basic> {
    let node = |args: Vec<Basic>| Basic::new(Head::Function(Function::AppellF1), args);
    if !evaluate() {
        return Ok(node(args.to_vec()));
    }
    let (a, b1, b2, c, x, y) = (&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]);
    if sort_key(b1) > sort_key(b2) {
        return appellf1(&[a.clone(), b2.clone(), b1.clone(), c.clone(), y.clone(), x.clone()]);
    }
    if b1 == b2 && sort_key(x) > sort_key(y) {
        return appellf1(&[a.clone(), b1.clone(), b2.clone(), c.clone(), y.clone(), x.clone()]);
    }
    if *x == Basic::zero() && *y == Basic::zero() {
        return Ok(Basic::one());
    }
    Ok(node(args.to_vec()))
}

/// The derivative of `appellf1` in the argument `argindex` (from 1).
pub fn appellf1_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    let (a, b1, b2, c, x, y) = (&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]);
    let one: Basic = Basic::one();
    let up = |v: &Basic| add(&[v.clone(), one.clone()]);
    if argindex == 5 {
        let factor: Basic = div(&mul(&[a.clone(), b1.clone()])?, c)?;
        let g: Basic = appellf1(&[up(a)?, up(b1)?, b2.clone(), up(c)?, x.clone(), y.clone()])?;
        return mul(&[factor, g]);
    }
    if argindex == 6 {
        let factor: Basic = div(&mul(&[a.clone(), b2.clone()])?, c)?;
        let g: Basic = appellf1(&[up(a)?, b1.clone(), up(b2)?, up(c)?, x.clone(), y.clone()])?;
        return mul(&[factor, g]);
    }
    let parameter: &Basic = &args[argindex - 1];
    if parameter.is_symbol() {
        return Ok(crate::core::function::derivative_node(e, &[(parameter.clone(), 1)]));
    }
    default_fdiff(e, argindex)
}

/// `DefinedFunction.fdiff`: `Derivative(f(x), x)` when the argument is a
/// symbol found in no other argument, else `Subs(Derivative(f(xi), xi),
/// xi, a)` (`f(xi)` the unevaluated node).
pub fn default_fdiff(e: &Basic, argindex: usize) -> PolyResult<Basic> {
    use crate::core::function::{derivative_node, subs_node};
    let index: usize = argindex - 1;
    let a: &Basic = &e.args()[index];
    let elsewhere: bool = e
        .args()
        .iter()
        .enumerate()
        .any(|(i, v)| i != index && v.free_symbols().contains(a));
    if a.is_symbol() && !elsewhere {
        return Ok(derivative_node(e, &[(a.clone(), 1)]));
    }
    let xi: Basic = crate::core::Dummy::new(&format!("xi_{}", argindex), &[]).0;
    let mut args: Vec<Basic> = e.args().to_vec();
    args[index] = xi.clone();
    let rebuilt: Basic = e.rebuild(args);
    Ok(subs_node(
        &derivative_node(&rebuilt, &[(xi.clone(), 1)]),
        vec![xi],
        vec![a.clone()],
    ))
}

/// The facts of `appellf1`.
pub fn appellf1_facts() -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    facts
}
