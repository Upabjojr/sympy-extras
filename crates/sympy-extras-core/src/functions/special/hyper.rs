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
//! series), `meijerg` by Slater's theorem (mpmath's `meijerg` and
//! `hypercomb`: the parameters perturbed at poles); mpmath's continuations
//! and `appellf1` values are not ported (no value). Not ported either: the series (`nseries`,
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
use crate::core::evalf::{complex, complex_real, parts, real_int, real_rat, Complex, Real};
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
/// Whether the parameters of `e` depend on `x`.
pub(crate) fn parameters_depend(e: &Basic, x: &Basic) -> bool {
    depends(hyper_ap(e), x) || depends(hyper_bq(e), x)
}

/// Whether the parameters are finite at `x = 0` and no lower one is a
/// nonpositive integer there, or may be one (`_continuous_parameters`).
pub(crate) fn continuous_parameters(e: &Basic, x: &Basic) -> PolyResult<bool> {
    use crate::core::expr::is;
    let zero: Basic = Basic::zero();
    let ap0: Vec<Basic> = hyper_ap(e)
        .iter()
        .map(|a| crate::core::subs::subs(a, x, &zero))
        .collect::<PolyResult<_>>()?;
    let bq0: Vec<Basic> = hyper_bq(e)
        .iter()
        .map(|b| crate::core::subs::subs(b, x, &zero))
        .collect::<PolyResult<_>>()?;
    Ok(ap0.iter().chain(&bq0).all(|c| is(c, Predicate::finite) == Some(true))
        && bq0
            .iter()
            .all(|b| is(b, Predicate::integer) == Some(false) || is(b, Predicate::positive) == Some(true)))
}

/// The leading term at `x = 0` when one lower parameter `b(x)` tends to a
/// nonpositive integer `-m` and the function is continuous in the other
/// parameters (`_pole_in_a_lower_parameter`): the terms of the series from
/// `z**(m + 1)` on have the factor `1/(b + m)`, and
///
/// `(b + m)*hyper(ap, [b] + cq, z) -> (-1)**m*prod((a)_(m+1))*z**(m+1)/(m!*(m + 1)!*prod((c)_(m+1)))*hyper(ap + m + 1, cq + m + 1 + [m + 2], z)`.
///
/// With an upper parameter a
/// nonpositive integer `-j`, `j <= m`, not depending on `x`, the function is
/// a polynomial of degree `j` whose terms have no pole: its value. `None`
/// when the limit is not of these kinds.
pub(crate) fn pole_in_a_lower_parameter(
    e: &Basic,
    x: &Basic,
    z0: &Basic,
    logx: Option<&Basic>,
    cdir: &Basic,
) -> PolyResult<Option<Basic>> {
    use crate::core::expr::is;
    use crate::functions::combinatorial::factorials::{factorial, rf};
    use crate::functions::elementary::complexes::abs_;
    let zero: Basic = Basic::zero();
    let (ap, bq) = (hyper_ap(e), hyper_bq(e));
    let ap0: Vec<Basic> = ap
        .iter()
        .map(|a| crate::core::subs::subs(a, x, &zero))
        .collect::<PolyResult<_>>()?;
    let bq0: Vec<Basic> = bq
        .iter()
        .map(|b| crate::core::subs::subs(b, x, &zero))
        .collect::<PolyResult<_>>()?;
    if z0.free_symbols().contains(x)
        || !ap0
            .iter()
            .chain(&bq0)
            .chain(std::iter::once(z0))
            .all(|c| is(c, Predicate::finite) == Some(true))
    {
        return Ok(None);
    }
    let unsafe_: Vec<usize> = bq0
        .iter()
        .enumerate()
        .filter(|(_, b)| !(is(b, Predicate::integer) == Some(false) || is(b, Predicate::positive) == Some(true)))
        .map(|(i, _)| i)
        .collect();
    if unsafe_.len() != 1 {
        return Ok(None);
    }
    let index: usize = unsafe_[0];
    let Some(pole) = integer_of(&bq0[index]) else {
        return Ok(None);
    };
    if !bq[index].free_symbols().contains(x) {
        return Ok(None);
    }
    let m: Int = -pole;
    let truncating: Vec<(Int, &Basic)> = ap0
        .iter()
        .zip(ap)
        .filter_map(|(a, given)| {
            integer_of(a)
                .filter(|p| *p <= Int::ZERO && -p.clone() <= m)
                .map(|p| (-p, given))
        })
        .collect();
    if !truncating.is_empty() {
        if truncating.iter().any(|(_, given)| given.free_symbols().contains(x)) {
            return Ok(None); // 0/0 in the terms
        }
        let degree: Int = truncating.iter().map(|(j, _)| j.clone()).min().expect("nonempty");
        let degree: i64 = degree.to_string().parse().unwrap_or(0);
        let mut terms: Vec<Basic> = Vec::new();
        for k in 0..=degree {
            let kk: Basic = int(k);
            let mut num: Vec<Basic> = ap0.iter().map(|a| rf(a, &kk)).collect::<PolyResult<_>>()?;
            num.push(crate::core::power::pow(z0, &kk)?);
            let mut den: Vec<Basic> = bq0.iter().map(|b| rf(b, &kk)).collect::<PolyResult<_>>()?;
            den.push(factorial(&kk)?);
            terms.push(div(&mul(&num)?, &mul(&den)?)?);
        }
        let value: Basic = add(&terms)?;
        return Ok(if is(&value, Predicate::zero) == Some(true) {
            None
        } else {
            Some(value)
        });
    }
    if ap0.len() > bq0.len() + 1
        || (ap0.len() == bq0.len() + 1 && is(&sub(&Basic::one(), &abs_(z0)?)?, Predicate::positive) != Some(true))
    {
        return Ok(None);
    }
    let m_i: i64 = m.to_string().parse().unwrap_or(0);
    let shift: Basic = int(m_i + 1);
    let others: Vec<Basic> = bq0
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != index)
        .map(|(_, b)| b.clone())
        .collect();
    let mut num: Vec<Basic> = vec![crate::core::power::pow(&int(-1), &int(m_i))?];
    for a in &ap0 {
        num.push(rf(a, &shift)?);
    }
    num.push(crate::core::power::pow(z0, &shift)?);
    let mut den: Vec<Basic> = vec![factorial(&int(m_i))?, factorial(&shift)?];
    for c in &others {
        den.push(rf(c, &shift)?);
    }
    let upper: Vec<Basic> = ap0
        .iter()
        .map(|a| add(&[a.clone(), shift.clone()]))
        .collect::<PolyResult<_>>()?;
    let mut lower: Vec<Basic> = others
        .iter()
        .map(|c| add(&[c.clone(), shift.clone()]))
        .collect::<PolyResult<_>>()?;
    lower.push(int(m_i + 2));
    // in closed form when it has one: its sign is the sign of an infinite limit
    let expanded: Basic = crate::simplify::hyperexpand::hyperexpand(
        &hyper(&upper, &lower, z0)?,
        false,
        crate::simplify::hyperexpand::Rewrite::Default,
        None,
    )?;
    let residue: Basic = mul(&[div(&mul(&num)?, &mul(&den)?)?, expanded])?;
    if is(&residue, Predicate::zero) == Some(true) {
        return Ok(None);
    }
    let lead: Basic = crate::core::series::as_leading_term(
        &add(&[bq[index].clone(), int(m_i)])?,
        std::slice::from_ref(x),
        logx,
        cdir,
    )?;
    Ok(Some(div(&residue, &lead)?))
}

/// `hyper` at `x = 0` when it is continuous there in its parameters and
/// its argument (of the value `z0`; `_value_at_parameters`): finite
/// parameters, no lower one a nonpositive integer, and `p <= q`, or
/// `p = q + 1` with `|z0| < 1` or `z0 = 1` and `re(sum(bq) - sum(ap)) > 0`;
/// `None` otherwise or for the value zero (which is no leading term).
pub(crate) fn value_at_parameters(e: &Basic, x: &Basic, z0: &Basic) -> PolyResult<Option<Basic>> {
    use crate::core::expr::is;
    use crate::functions::elementary::complexes::{abs_, re};
    let zero: Basic = Basic::zero();
    let ap0: Vec<Basic> = hyper_ap(e)
        .iter()
        .map(|a| crate::core::subs::subs(a, x, &zero))
        .collect::<PolyResult<_>>()?;
    let bq0: Vec<Basic> = hyper_bq(e)
        .iter()
        .map(|b| crate::core::subs::subs(b, x, &zero))
        .collect::<PolyResult<_>>()?;
    if z0.free_symbols().contains(x)
        || !ap0
            .iter()
            .chain(&bq0)
            .chain(std::iter::once(z0))
            .all(|c| is(c, Predicate::finite) == Some(true))
    {
        return Ok(None);
    }
    if bq0
        .iter()
        .any(|b| !(is(b, Predicate::integer) == Some(false) || is(b, Predicate::positive) == Some(true)))
    {
        return Ok(None);
    }
    if ap0.len() > bq0.len() + 1 {
        return Ok(None);
    }
    if ap0.len() == bq0.len() + 1 {
        let inside: bool = is(&sub(&Basic::one(), &abs_(z0)?)?, Predicate::positive) == Some(true);
        let gauss: bool =
            *z0 == Basic::one() && is(&re(&sub(&add(&bq0)?, &add(&ap0)?)?)?, Predicate::positive) == Some(true);
        if !(inside || gauss) {
            return Ok(None);
        }
    }
    if ap0.iter().any(|a| is(a, Predicate::zero) == Some(true)) {
        return Ok(Some(Basic::one()));
    }
    let value: Basic = hyper(&ap0, &bq0, z0)?;
    Ok(if is(&value, Predicate::zero) == Some(true) {
        None
    } else {
        Some(value)
    })
}

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

/// The radius of convergence of the defining series: 0, 1 or oo, with
/// cases for the integer parameters of unknown sign.
pub fn radius_of_convergence(e: &Basic) -> PolyResult<Basic> {
    radius_of(hyper_ap(e), hyper_bq(e))
}

/// The radius of the series of the parameters `ap`, `bq`
/// (`_radius_of_convergence`): equal parameters cancel; a nonpositive
/// integer numerator ends the series, a nonpositive integer denominator
/// divides by zero unless some a > b ended it before.
fn radius_of(ap: &[Basic], bq: &[Basic]) -> PolyResult<Basic> {
    use crate::core::logic::{and_, false_, not_, or_, true_};
    use crate::core::relational::{eq, gt, le};
    let may_be_nonpositive_integer =
        |a: &Basic| is_true(a, Predicate::integer) && crate::core::expr::is(a, Predicate::positive) != Some(true);
    let without = |parameters: &[Basic], parameter: &Basic| -> Vec<Basic> {
        let mut rest: Vec<Basic> = parameters.to_vec();
        if let Some(i) = rest.iter().position(|p| p == parameter) {
            rest.remove(i);
        }
        rest
    };
    let tops: Vec<Basic> = ap.iter().filter(|a| may_be_nonpositive_integer(a)).cloned().collect();
    let bottoms: Vec<Basic> = bq.iter().filter(|b| may_be_nonpositive_integer(b)).cloned().collect();
    // the pairs which are, or may be, equal: cancelled, in a case of their own
    let mut cancelled: Vec<(Basic, Basic)> = Vec::new();
    for a in &tops {
        for b in &bottoms {
            let equal: Basic = eq(a, b)?;
            if equal == true_() {
                return radius_of(&without(ap, a), &without(bq, b));
            }
            if equal != false_() && cancelled.len() < 2 {
                cancelled.push((radius_of(&without(ap, a), &without(bq, b))?, equal));
            }
        }
    }
    let zero: Basic = Basic::zero();
    let mut defined_parts: Vec<Basic> = Vec::new();
    for b in &bottoms {
        let mut options: Vec<Basic> = vec![gt(b, &zero)?];
        for a in &tops {
            options.push(and_(&[le(a, &zero)?, gt(a, b)?]));
        }
        defined_parts.push(or_(&options));
    }
    let defined: Basic = and_(&defined_parts);
    let ends: Basic = or_(&tops.iter().map(|a| le(a, &zero)).collect::<PolyResult<Vec<_>>>()?);
    let general: Basic = if ap.len() == bq.len() + 1 {
        Basic::one()
    } else if ap.len() <= bq.len() {
        Basic::oo()
    } else {
        zero.clone()
    };
    let mut cases: Vec<(Basic, Basic)> = cancelled;
    if defined == false_() {
        cases.push((zero.clone(), true_()));
    } else {
        if defined != true_() {
            cases.push((zero.clone(), not_(&defined)));
        }
        if ends == true_() {
            cases.push((Basic::oo(), true_()));
        } else if ends == false_() {
            cases.push((general, true_()));
        } else {
            cases.push((Basic::oo(), ends));
            cases.push((general, true_()));
        }
    }
    if cases.len() == 1 {
        return Ok(cases.remove(0).0);
    }
    crate::functions::elementary::piecewise::piecewise(&cases)
}

/// The condition under which the series converges (`false` for a zero
/// radius, `true` for an infinite one), case by case for a radius which
/// depends on the parameters.
pub fn convergence_statement(e: &Basic) -> PolyResult<Basic> {
    use crate::core::logic::{false_, true_};
    let r: Basic = radius_of_convergence(e)?;
    if r == Basic::zero() {
        return Ok(false_());
    }
    if r == Basic::oo() {
        return Ok(true_());
    }
    if matches!(r.head(), Head::Piecewise) {
        let simplified: Basic = crate::logic::boolsimp::simplify_boolean_default(&statement(e, &r)?, 1.7)?;
        return crate::logic::boolsimp::simplify_univariate(&simplified, false);
    }
    statement(e, &r)
}

fn statement(e: &Basic, r: &Basic) -> PolyResult<Basic> {
    use crate::core::logic::{and_, false_, not_, or_, true_};
    use crate::core::relational::{ge, le, lt, ne};
    use crate::functions::elementary::complexes::{abs_, re};
    if matches!(r.head(), Head::Piecewise) {
        // a radius which depends on the parameters: each case with its own statement
        let mut cases: Vec<Basic> = Vec::new();
        let mut before: Basic = true_();
        for pair in r.args() {
            let (value, condition) = (&pair.args()[0], &pair.args()[1]);
            cases.push(and_(&[before.clone(), condition.clone(), statement(e, value)?]));
            before = and_(&[before, not_(condition)]);
        }
        return Ok(or_(&cases));
    }
    if *r == Basic::zero() {
        return Ok(false_());
    }
    if *r == Basic::oo() {
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
    let two_f_one: bool = ap.len() == 2 && bq.len() == 1;
    if terms.is_none() && two_f_one && size > 0.8 {
        // mpmath's _hyp2f1: the transformations outside the disc |z| <= 0.8
        return hyp2f1_continued(&ap[0], &ap[1], &bq[0], z, prec);
    }
    if terms.is_none() {
        let fits: bool =
            // (p <= q: the series converges everywhere, and the loss of
            // precision to cancellation is measured below; mpmath's
            // asymptotic expansions for a large argument, not ported, fall
            // back to it when they do not converge)
            (ap.len() <= bq.len() && (one_f_one || size <= 1.0e4))
                || (ap.len() == bq.len() + 1 && (size < 0.75 || (two_f_one && size <= 0.8)));
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

/// The integer of a value, if it is one.
fn integer_value(v: &Complex) -> Option<i64> {
    let (re, im) = parts(v);
    if im == Real::ZERO && re.clone() == re.floor() {
        return re.to_f64().value().round().to_string().parse::<i64>().ok();
    }
    None
}

/// `2F1(a, b; c; z)` outside the disc `|z| <= 0.8` (mpmath's `_hyp2f1`):
/// Gauss's theorem at `z = 1`, the `1/z` transformation for `|z| >= 1.3`,
/// the `1 - z` transformation for `|1 - z| <= 0.75`, Pfaff's `z/(z - 1)`
/// for `|z/(z - 1)| <= 0.75`, and Gosper's recurrence on the rest of the
/// unit circle; the terms of the transformations through [`hypercomb`].
fn hyp2f1_continued(a: &Complex, b: &Complex, c: &Complex, z: &Complex, prec: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::gamma;
    let wp: usize = prec + 10;
    let one: Complex = c_int(1, wp);
    let (a, b, c, z) = (raise_to(a, wp), raise_to(b, wp), raise_to(c, wp), raise_to(z, wp));
    let is_nonpositive_in = |v: &Complex, low: i64| integer_value(v).is_some_and(|k| low <= k && k <= 0);
    if z == one {
        let convergent: bool = parts(&(&(&c - &a) - &b)).0 > Real::ZERO;
        let finite: bool = is_nonpositive_in(&a, i64::MIN) || is_nonpositive_in(&b, i64::MIN);
        let c_int_value: Option<i64> = integer_value(&c);
        let zerodiv: bool = c_int_value.is_some_and(|k| k <= 0)
            && !(is_nonpositive_in(&a, c_int_value.unwrap_or(0)) || is_nonpositive_in(&b, c_int_value.unwrap_or(0)));
        if !(convergent || finite) || zerodiv {
            return None; // a pole: mpmath's infinity
        }
        // gammaprod([c, c - a - b], [c - a, c - b])
        let mut v: Complex = gamma(&c, wp)?;
        v = &v * &gamma(&(&(&c - &a) - &b), wp)?;
        for d in [&c - &a, &c - &b] {
            match gamma(&d, wp) {
                Some(g) => v = &v / &g,
                None => return Some(c_int(0, prec)),
            }
        }
        return Some(raise_to(&v, prec));
    }
    let absz: f64 = modulus(&z);
    let value: Complex = if absz >= 1.3 {
        let (c, z) = (c.clone(), z.clone());
        let function = move |args: &[Complex]| -> Vec<HyperTerm> {
            let (a, b) = (&args[0], &args[1]);
            let wp: usize = parts(a).0.precision().max(prec);
            let one: Complex = c_int(1, wp);
            let t: Complex = &one - &c;
            let ab: Complex = a - b;
            let minus_z: Complex = &c_int(0, wp) - &z;
            let rz: Complex = &one / &z;
            let neg = |v: &Complex| &c_int(0, wp) - v;
            vec![
                HyperTerm {
                    bases: vec![minus_z.clone()],
                    exponents: vec![neg(a)],
                    gamma_numerator: vec![c.clone(), neg(&ab)],
                    gamma_denominator: vec![b.clone(), &c - a],
                    upper: vec![a.clone(), &t + a],
                    lower: vec![&one + &ab],
                    argument: rz.clone(),
                },
                HyperTerm {
                    bases: vec![minus_z],
                    exponents: vec![neg(b)],
                    gamma_numerator: vec![c.clone(), ab.clone()],
                    gamma_denominator: vec![a.clone(), &c - b],
                    upper: vec![b.clone(), &t + b],
                    lower: vec![&one - &ab],
                    argument: rz,
                },
            ]
        };
        hypercomb(&function, &[a.clone(), b.clone()], wp)?
    } else if modulus(&(&one - &z)) <= 0.75 {
        let (c, z) = (c.clone(), z.clone());
        let function = move |args: &[Complex]| -> Vec<HyperTerm> {
            let (a, b) = (&args[0], &args[1]);
            let wp: usize = parts(a).0.precision().max(prec);
            let one: Complex = c_int(1, wp);
            let t: Complex = &(&c - a) - b;
            let (ca, cb) = (&c - a, &c - b);
            let rz: Complex = &one - &z;
            vec![
                HyperTerm {
                    bases: vec![],
                    exponents: vec![],
                    gamma_numerator: vec![c.clone(), t.clone()],
                    gamma_denominator: vec![ca.clone(), cb.clone()],
                    upper: vec![a.clone(), b.clone()],
                    lower: vec![&one - &t],
                    argument: rz.clone(),
                },
                HyperTerm {
                    bases: vec![rz.clone()],
                    exponents: vec![t.clone()],
                    gamma_numerator: vec![c.clone(), &(a + b) - &c],
                    gamma_denominator: vec![a.clone(), b.clone()],
                    upper: vec![ca, cb],
                    lower: vec![&one + &t],
                    argument: rz,
                },
            ]
        };
        hypercomb(&function, &[a.clone(), b.clone()], wp)?
    } else if modulus(&(&z / &(&z - &one))) <= 0.75 {
        let w: Complex = &z / &(&z - &one);
        let inner: Complex = hyper_value(&[a.clone(), &c - &b], std::slice::from_ref(&c), &w, wp)?;
        &inner / &complex_power(&(&one - &z), &a, wp)?
    } else {
        hyp2f1_gosper(&a, &b, &c, &z, wp)?
    };
    Some(raise_to(&value, prec))
}

/// `2F1(a, b; c; z)` by Gosper's recurrence (mpmath's `_hyp2f1_gosper`),
/// for the part of the unit circle no transformation reaches; the working
/// precision raised by the cancellation measured.
fn hyp2f1_gosper(a: &Complex, b: &Complex, c: &Complex, z: &Complex, prec: usize) -> Option<Complex> {
    let maxprec: usize = 100 * prec;
    let mut extra: usize = 10;
    loop {
        let wp: usize = prec + extra;
        let (a, b, c, z) = (raise_to(a, wp), raise_to(b, wp), raise_to(c, wp), raise_to(z, wp));
        let n = |k: i64| c_int(k, wp);
        let half: Complex = &n(1) / &n(2);
        let (mut d, mut e, mut f) = (n(0), n(1), n(0));
        let abz: Complex = &(&a * &b) * &z;
        let ch: Complex = &c * &half;
        let c1h: Complex = &(&c + &n(1)) * &half;
        let nz: Complex = &n(1) - &z;
        let g: Complex = &z / &nz;
        let abg: Complex = &(&a * &b) * &g;
        let cba: Complex = &(&c - &b) - &a;
        let z2: Complex = &z - &n(2);
        let tol: f64 = -(wp as f64) - 10.0;
        let mut maxmag: f64 = f64::NEG_INFINITY;
        let mut k: i64 = 0;
        let f1: Complex = loop {
            let kk: Complex = n(k);
            let kch: Complex = &kk + &ch;
            let kakbz: Complex =
                &(&(&(&kk + &a) * &(&kk + &b)) * &z) / &(&(&(&n(4) * &(&kk + &n(1))) * &kch) * &(&kk + &c1h));
            let d1: Complex = &kakbz * &(&e - &(&(&(&kk + &cba) * &d) * &g));
            let e1: Complex = &kakbz * &(&(&d * &abg) + &(&(&kk + &c) * &e));
            let inner: Complex = &(&kk * &(&(&(&cba * &z) + &(&kk * &z2)) - &c)) - &abz;
            let ft: Complex = &(&d * &inner) / &(&(&n(2) * &kch) * &nz);
            let f1: Complex = &(&f + &e) - &ft;
            maxmag = maxmag.max(complex_magnitude(&f1));
            if complex_magnitude(&(&f1 - &f)) < tol {
                break f1;
            }
            (d, e, f) = (d1, e1, f1);
            k += 1;
            if k > 1_000_000 {
                return None;
            }
        };
        let cancellation: f64 = maxmag - complex_magnitude(&f1);
        if cancellation < extra as f64 {
            return Some(raise_to(&f1, prec));
        }
        extra += cancellation.ceil() as usize;
        if extra > maxprec {
            return None;
        }
    }
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
// meijerg values
// ---------------------------------------------------------------------

/// One term of a combination of hypergeometric functions,
/// `prod w_i**c_i * prod gamma(alpha_j) / prod gamma(beta_k) * pFq(upper; lower; argument)`
/// (the terms of mpmath's `hypercomb`).
struct HyperTerm {
    bases: Vec<Complex>,
    exponents: Vec<Complex>,
    gamma_numerator: Vec<Complex>,
    gamma_denominator: Vec<Complex>,
    upper: Vec<Complex>,
    lower: Vec<Complex>,
    argument: Complex,
}

/// `floor(log2|x|) + 1`, `-inf` at zero (mpmath's `mag`).
fn magnitude(x: &Real) -> f64 {
    if *x == Real::ZERO {
        return f64::NEG_INFINITY;
    }
    x.to_f64().value().abs().log2().floor() + 1.0
}

/// The magnitude of a complex number, the larger of its parts'.
fn complex_magnitude(z: &Complex) -> f64 {
    let (re, im) = parts(z);
    magnitude(&re).max(magnitude(&im))
}

fn is_zero_value(z: &Complex) -> bool {
    let (re, im) = parts(z);
    re == Real::ZERO && im == Real::ZERO
}

/// The nearest integer to `x` (halves away from zero) and the magnitude of
/// the distance to it, `-inf` for an integer (mpmath's `nint_distance`).
fn nint_distance(x: &Complex) -> (i64, f64) {
    let (re, im) = parts(x);
    let imaginary: f64 = if im == Real::ZERO {
        f64::NEG_INFINITY
    } else {
        magnitude(&im)
    };
    let half: Real = real_rat(&crate::numbers::rat(1, 2), re.precision().max(64));
    let negative: bool = re < Real::ZERO;
    let size: Real = if negative { -re.clone() } else { re.clone() };
    let nearest: Real = (size + half).floor();
    let nearest: Real = if negative { -nearest } else { nearest };
    let distance: Real = re - nearest.clone();
    let n: i64 = nearest.to_f64().value() as i64;
    (n, magnitude(&distance).max(imaginary))
}

/// `(perturb, recompute, extra precision, discarded terms)`: the poles of
/// the gamma factors and of the lower parameters (a term with more poles in
/// its reciprocal gammas than elsewhere is zero), a zero base to a power of
/// nonpositive real part (mpmath's `_check_need_perturb`).
fn check_need_perturb(terms: &[HyperTerm]) -> (bool, bool, usize, Vec<usize>) {
    let (mut perturb, mut recompute) = (false, false);
    let mut extra: usize = 0;
    let mut discard: Vec<usize> = Vec::new();
    for (index, term) in terms.iter().enumerate() {
        let mut singular_weight: bool = false;
        for (w, c) in term.bases.iter().zip(&term.exponents) {
            if is_zero_value(w) && parts(c).0 <= Real::ZERO && !is_zero_value(c) {
                perturb = true;
                recompute = true;
                singular_weight = true;
            }
        }
        let mut poles: [usize; 3] = [0, 0, 0];
        for (kind, data) in [&term.gamma_numerator, &term.gamma_denominator, &term.lower]
            .iter()
            .enumerate()
        {
            for x in data.iter() {
                let (n, d) = nint_distance(x);
                if n > 0 {
                    continue;
                }
                if d == f64::NEG_INFINITY {
                    // a pole of a lower parameter is harmless when an upper
                    // one terminates the series before it
                    let polynomial: bool = kind == 2
                        && term
                            .upper
                            .iter()
                            .any(|u| nonpositive_integer(u).is_some_and(|v| v >= n));
                    if !polynomial {
                        poles[kind] += 1;
                    }
                } else if d < -4.0 {
                    extra += (-d) as usize;
                    recompute = true;
                }
            }
        }
        if poles[1] > poles[0] + poles[2] && !singular_weight {
            discard.push(index);
        } else if poles.iter().sum::<usize>() > 0 {
            perturb = true;
            recompute = true;
        }
    }
    (perturb, recompute, extra, discard)
}

/// `w**c` on the principal branch, `0**c = 0` for a positive real part;
/// `None` for zero to a power of nonpositive real part (mpmath raises).
fn complex_power(w: &Complex, c: &Complex, wp: usize) -> Option<Complex> {
    if is_zero_value(c) {
        return Some(c_int(1, wp));
    }
    if is_zero_value(w) {
        return if parts(c).0 > Real::ZERO {
            Some(c_int(0, wp))
        } else {
            None
        };
    }
    Some((c * &w.ln()).exp())
}

/// The value of one term at the working precision; `None` where a
/// factor has no value.
fn term_value(term: &HyperTerm, wp: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::gamma;
    let mut v: Complex = hyper_value(&term.upper, &term.lower, &term.argument, wp)?;
    for a in &term.gamma_numerator {
        v = &v * &gamma(a, wp)?;
    }
    for b in &term.gamma_denominator {
        match gamma(b, wp) {
            Some(g) => v = &v / &g,
            None => return Some(c_int(0, wp)), // 1/gamma at a pole
        }
    }
    for (w, c) in term.bases.iter().zip(&term.exponents) {
        v = &v * &complex_power(w, c, wp)?;
    }
    Some(v)
}

/// A sum of hypergeometric terms given by `function` of the parameters,
/// to `prec` bits: the working precision raised until the cancellation
/// between the terms is paid for, and the parameters perturbed by tiny
/// independent amounts where gamma factors or lower parameters hit poles
/// (the limit then taken by comparing two perturbations). The method of
/// mpmath's `hypercomb` (F. Johansson, BSD license), whose control flow
/// this follows so that the values agree; `None` beyond its maximum
/// precision, or where a hypergeometric term has no value here.
fn hypercomb(function: &dyn Fn(&[Complex]) -> Vec<HyperTerm>, params: &[Complex], prec: usize) -> Option<Complex> {
    let orig: usize = prec;
    let maxprec: usize = (1000.0 * (orig as f64).powf(0.25) + 4.0 * orig as f64) as usize;
    let mut wp: usize = orig;
    let mut hextra: usize = 0;
    let mut reference: Option<Complex> = None;
    loop {
        wp += 10;
        if wp > maxprec {
            return None;
        }
        let orig2: usize = wp;
        let mut values: Vec<Complex> = params.iter().map(|p| raise_to(p, wp)).collect();
        let mut terms: Vec<HyperTerm> = function(&values);
        let (perturb, recompute, extra, discard) = check_need_perturb(&terms);
        wp += extra;
        if perturb {
            let hmag: usize = orig + 10 + hextra;
            wp = orig2 + 10 + hmag + 10;
            let mut h: Complex = &c_int(1, wp) / &complex_real(real_int(&(Int::ONE << hmag), wp), wp);
            for (k, v) in values.iter_mut().enumerate() {
                *v = &raise_to(v, wp) + &h;
                // independent perturbations, so that two do not cancel
                h = &h + &(&h / &c_int(k as i64 + 1, wp));
            }
        }
        if recompute {
            terms = function(&values);
        }
        let terms: Vec<HyperTerm> = terms
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !discard.contains(i))
            .map(|(_, t)| t)
            .collect();
        if terms.is_empty() {
            return Some(c_int(0, prec));
        }
        let mut evaluated: Vec<Complex> = Vec::new();
        for term in &terms {
            evaluated.push(term_value(term, wp)?);
        }
        if evaluated.len() == 1 && !perturb {
            return Some(raise_to(&evaluated[0], prec));
        }
        let mut total: Complex = c_int(0, wp);
        for v in &evaluated {
            total = &total + v;
        }
        let largest: f64 = evaluated
            .iter()
            .map(complex_magnitude)
            .fold(f64::NEG_INFINITY, f64::max);
        let cancellation: f64 = largest - complex_magnitude(&total);
        let precision_ok: bool = cancellation < (wp - orig) as f64;
        if (precision_ok && !perturb) || cancellation.is_nan() {
            return Some(raise_to(&total, prec));
        } else if precision_ok {
            match &reference {
                None => {
                    hextra += 20;
                    reference = Some(total);
                    continue;
                }
                Some(r) => {
                    if complex_magnitude(&(&total - r)) <= complex_magnitude(&total) - orig as f64 {
                        return Some(raise_to(&total, prec));
                    }
                    hextra *= 2;
                    reference = Some(total);
                }
            }
        } else {
            let increment: f64 = cancellation.max((orig / 2) as f64).min(extra.max(orig) as f64);
            wp += increment as usize;
        }
    }
}

/// `G^{m,n}_{p,q}(an, aother; bm, bother | z)` to `prec` bits by Slater's
/// theorem: the sum of `m` hypergeometric `pF(q-1)` of `z` when `p < q`
/// (or `p = q`, `|z| <= 1` or `m + n < p`), of `n` `qF(p-1)` of `1/z`
/// otherwise, through [`hypercomb`] (mpmath's `meijerg` with `r = 1`).
pub fn meijerg_value(
    an: &[Complex],
    aother: &[Complex],
    bm: &[Complex],
    bother: &[Complex],
    z: &Complex,
    prec: usize,
) -> Option<Complex> {
    let (n, m) = (an.len(), bm.len());
    let p: usize = n + aother.len();
    let q: usize = m + bother.len();
    let first_series: bool = match p.cmp(&q) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => !(m + n == p && modulus(z) > 1.0),
    };
    let z: Complex = z.clone();
    let function = |args: &[Complex]| -> Vec<HyperTerm> {
        let wp: usize = args
            .iter()
            .map(|v| parts(v).0.precision())
            .max()
            .unwrap_or(prec)
            .max(prec);
        let (a, b) = (&args[..p], &args[p..]);
        let one: Complex = c_int(1, wp);
        let zz: Complex = raise_to(&z, wp);
        let mut terms: Vec<HyperTerm> = Vec::new();
        if first_series {
            let sign: Complex = c_int(
                if (p as i64 - m as i64 - n as i64).rem_euclid(2) == 0 {
                    1
                } else {
                    -1
                },
                wp,
            );
            for k in 0..m {
                let mut gn: Vec<Complex> = (0..m).filter(|&j| j != k).map(|j| &b[j] - &b[k]).collect();
                gn.extend((0..n).map(|j| &(&one - &a[j]) + &b[k]));
                let mut gd: Vec<Complex> = (n..p).map(|j| &a[j] - &b[k]).collect();
                gd.extend((m..q).map(|j| &(&one - &b[j]) + &b[k]));
                terms.push(HyperTerm {
                    bases: vec![zz.clone()],
                    exponents: vec![b[k].clone()],
                    gamma_numerator: gn,
                    gamma_denominator: gd,
                    upper: (0..p).map(|j| &(&one - &a[j]) + &b[k]).collect(),
                    lower: (0..q).filter(|&j| j != k).map(|j| &(&one - &b[j]) + &b[k]).collect(),
                    argument: &sign * &zz,
                });
            }
        } else {
            let sign: Complex = c_int(
                if (q as i64 - m as i64 - n as i64).rem_euclid(2) == 0 {
                    1
                } else {
                    -1
                },
                wp,
            );
            for k in 0..n {
                let mut gn: Vec<Complex> = (0..n).filter(|&j| j != k).map(|j| &a[k] - &a[j]).collect();
                gn.extend((0..m).map(|j| &(&one - &a[k]) + &b[j]));
                let mut gd: Vec<Complex> = (m..q).map(|j| &a[k] - &b[j]).collect();
                gd.extend((n..p).map(|j| &(&one - &a[k]) + &a[j]));
                terms.push(HyperTerm {
                    bases: vec![zz.clone()],
                    exponents: vec![&a[k] - &one],
                    gamma_numerator: gn,
                    gamma_denominator: gd,
                    upper: (0..q).map(|j| &(&one - &a[k]) + &b[j]).collect(),
                    lower: (0..p).filter(|&j| j != k).map(|j| &(&one + &a[j]) - &a[k]).collect(),
                    argument: &sign / &zz,
                });
            }
        }
        terms
    };
    let params: Vec<Complex> = an.iter().chain(aother).chain(bm).chain(bother).cloned().collect();
    let value: Complex = hypercomb(&function, &params, prec)?;
    // real parameters and a positive argument: mpmath works with reals
    let real: bool = params
        .iter()
        .chain(std::iter::once(&z))
        .all(|v| parts(v).1 == Real::ZERO)
        && parts(&z).0 > Real::ZERO;
    if real {
        return Some(complex(parts(&value).0, real_int(&Int::ZERO, prec)));
    }
    Some(value)
}

/// The value of a `MeijerG` node, as the Python's `_numeric_node`: its
/// parameters and argument at the precision, then [`meijerg_value`].
pub fn meijerg_node_value(e: &Basic, prec: usize) -> Option<Complex> {
    use crate::core::evalf::numeric_value;
    let lists: Vec<Vec<Complex>> = meijerg_lists(e)
        .iter()
        .map(|l| {
            l.iter()
                .map(|v| numeric_value(v, prec))
                .collect::<Option<Vec<Complex>>>()
        })
        .collect::<Option<_>>()?;
    let z: Complex = numeric_value(&e.args()[2], prec)?;
    meijerg_value(&lists[0], &lists[1], &lists[2], &lists[3], &z, prec)
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
