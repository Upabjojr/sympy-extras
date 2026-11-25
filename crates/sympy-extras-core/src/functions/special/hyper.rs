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

/// `|w|` as a float: `hypot`, which does not overflow where the squares
/// would (the perturbed terms of `hypercomb` pass `2**512`, and a modulus
/// of `inf` stopped the series as if it diverged).
/// Whether either part of `w` is an infinity, on which dashu panics.
fn has_infinity(w: &Complex) -> bool {
    let (wr, wi) = parts(w);
    wr.repr().is_infinite() || wi.repr().is_infinite()
}

fn modulus(w: &Complex) -> f64 {
    let (wr, wi) = parts(w);
    wr.to_f64().value().hypot(wi.to_f64().value())
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
fn series(
    ap: &[Complex],
    bq: &[Complex],
    z: &Complex,
    terms: Option<i64>,
    wp: usize,
    maxterms: i64,
) -> Option<(Complex, f64)> {
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
            if k > maxterms {
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
    if terms.is_none() && matches!((ap.len(), bq.len()), (0, 1) | (1, 2) | (2, 2) | (2, 3)) {
        // mpmath's _hyp0f1, _hyp1f2, _hyp2f2, _hyp2f3: the asymptotic
        // expansions for a large argument, the series when they do not
        // apply or converge
        if let Some(v) = small_pq_asymptotic(ap, bq, z, prec) {
            return Some(v);
        }
    }
    let two_f_one: bool = ap.len() == 2 && bq.len() == 1;
    if terms.is_none() && two_f_one && size > 0.8 {
        // mpmath's _hyp2f1: the transformations outside the disc |z| <= 0.8
        return hyp2f1_continued(&ap[0], &ap[1], &bq[0], z, prec);
    }
    if terms.is_none() && ap.len() == bq.len() + 1 && ap.len() >= 3 && size >= 0.75 {
        // mpmath's _hypq1fq: the direct summation inside the unit circle
        // (hypsum's maxterms, wp*100), on NoConvergence or beyond it the
        // alternate summation of 3F2 near 1, the 'r+s' acceleration of
        // the series near the circle (its Euler-Maclaurin summation after
        // them is not ported: no value), the 1/z transformation outside
        if size < 1.0 {
            if let Some(v) = defining_series_value(ap, bq, z, None, prec, ((prec + 50) * 100) as i64) {
                return Some(v);
            }
        }
        if *z == c_int(1, prec) {
            // the series diverges at 1 unless re(sum b - sum a) > 0 (mpmath:
            // infinity); else the summations below, as mpmath
            let mut excess: Complex = c_int(0, prec + 10);
            for b in bq {
                excess = &excess + b;
            }
            for a in ap {
                excess = &excess - a;
            }
            if parts(&excess).0 <= Real::ZERO {
                return None;
            }
        }
        let near_one: bool = modulus(&(z - &c_int(1, prec + 10))) < 0.05;
        if ap.len() == 3 && near_one {
            if let Some(v) = hyp3f2_near_one(ap, bq, z, prec) {
                return Some(v);
            }
        }
        let re_beyond: bool = parts(z).0 > real_int(&Int::ONE, prec);
        if size < 1.1 && !re_beyond {
            return hypq1fq_accelerated(ap, bq, z, prec);
        }
        return hypq1fq_reciprocal(ap, bq, z, prec);
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
    defining_series_value(ap, bq, z, terms, prec, 200_000)
}

/// The defining series at `prec` bits, the working precision raised by the
/// loss to cancellation; `terms` for a terminating series, `None` after
/// `maxterms` terms.
fn defining_series_value(
    ap: &[Complex],
    bq: &[Complex],
    z: &Complex,
    terms: Option<i64>,
    prec: usize,
    maxterms: i64,
) -> Option<Complex> {
    let real: bool = ap
        .iter()
        .chain(bq)
        .chain(std::iter::once(z))
        .all(|v| parts(v).1 == Real::ZERO);
    let mut wp: usize = prec + 40;
    for _ in 0..4 {
        let lift = |v: &Complex| raise_to(v, wp);
        let (a, b): (Vec<Complex>, Vec<Complex>) = (ap.iter().map(lift).collect(), bq.iter().map(lift).collect());
        let (total, largest) = series(&a, &b, &lift(z), terms, wp, maxterms)?;
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

/// `(q+1)Fq` on and near the unit circle (`|z| < 1.1`, `re(z) <= 1`) by
/// mpmath's `nsum` of the defining series with the `'r+s'` acceleration
/// (`_hypq1fq`'s `sum_method` without `'e'`); `None` where it does not
/// converge (mpmath's Euler-Maclaurin summation after it is not ported).
fn hypq1fq_accelerated(ap: &[Complex], bq: &[Complex], z: &Complex, prec: usize) -> Option<Complex> {
    // PORTED FROM MPMATH: mpmath/functions/hypergeometric.py:_hypq1fq (the nsum branch)
    let mut cache: Vec<Complex> = Vec::new();
    let mut term = |k: i64, wp: usize| -> Option<(Real, Real)> {
        let k: usize = usize::try_from(k).ok()?;
        while cache.len() <= k {
            let n: usize = cache.len();
            let t: Complex = if n == 0 {
                c_int(1, wp)
            } else {
                let m: Complex = c_int(n as i64 - 1, wp);
                let mut t: Complex = cache[n - 1].clone();
                for a in ap {
                    t = &t * &(&raise_to(a, wp) + &m);
                }
                for b in bq {
                    t = &t / &(&raise_to(b, wp) + &m);
                }
                &(&t * &raise_to(z, wp)) / &c_int(n as i64, wp)
            };
            if has_infinity(&t) {
                return None;
            }
            cache.push(t);
        }
        Some(parts(&cache[k]))
    };
    let (re, im) = crate::integrals::extrapolation::nsum_strict(&mut term, prec)?;
    Some(complex(re, im))
}

/// `3F2(a1, a2, a3; b1, b2; z)` for `|z - 1| < 0.05` by mpmath's alternate
/// summation (Bühring's expansion at unit argument): `nsum` over
/// `t_k 2F1(a1, a2; b1 + b2 - a3 + k; z)` times
/// `gammaprod([b1, b2], [a1, a2, a3])`; `None` where it does not converge.
fn hyp3f2_near_one(ap: &[Complex], bq: &[Complex], z: &Complex, prec: usize) -> Option<Complex> {
    // PORTED FROM MPMATH: mpmath/functions/hypergeometric.py:_hypq1fq (the alternate summation of 3F2)
    use crate::functions::combinatorial::factorials::gamma_product;
    let mut cache: Vec<Complex> = Vec::new();
    let mut term = |k: i64, wp: usize| -> Option<(Real, Real)> {
        let k: usize = usize::try_from(k).ok()?;
        let lift = |v: &Complex| raise_to(v, wp);
        let (a1, a2, a3, b1, b2) = (lift(&ap[0]), lift(&ap[1]), lift(&ap[2]), lift(&bq[0]), lift(&bq[1]));
        let one: Complex = c_int(1, wp);
        let u0: Complex = &(&b1 + &b2) - &a3;
        if cache.is_empty() {
            let initial: Complex = gamma_product(
                &[&b2 - &a3, &b1 - &a3, a1.clone(), a2.clone()],
                &[&b2 - &a3, &b1 - &a3, one.clone(), u0.clone()],
                wp,
            )?;
            cache.push(initial);
        }
        while cache.len() <= k {
            let n: usize = cache.len();
            let kk: Complex = c_int(n as i64, wp);
            let u: Complex = &u0 + &kk;
            let mut t: Complex = cache[n - 1].clone();
            t = &(&t * &(&(&(&b1 + &kk) - &a3) - &one)) * &(&(&(&b2 + &kk) - &a3) - &one);
            let den: Complex = &kk * &(&u - &one);
            if is_zero_value(&den) {
                return None;
            }
            t = &t / &den;
            if has_infinity(&t) {
                return None;
            }
            cache.push(t);
        }
        let u: Complex = &u0 + &c_int(k as i64, wp);
        let h: Complex = hyper_value(&[a1, a2], &[u], &lift(z), wp)?;
        Some(parts(&(&cache[k] * &h)))
    };
    let (re, im) = crate::integrals::extrapolation::nsum_strict(&mut term, prec)?;
    let lift = |v: &Complex| raise_to(v, prec + 20);
    let factor: Complex = gamma_product(
        &[lift(&bq[0]), lift(&bq[1])],
        &[lift(&ap[0]), lift(&ap[1]), lift(&ap[2])],
        prec + 20,
    )?;
    let v: Complex = &complex(re, im) * &factor;
    let (re, im) = parts(&v);
    Some(complex(
        re.with_precision(prec).value(),
        im.with_precision(prec).value(),
    ))
}

/// `(q+1)Fq(ap; bq; z)` for `|z| > 1` by the `1/z` transformation (mpmath's
/// `_hypq1fq`, functions.wolfram.com 07.31.06.0004): `q + 1` terms
/// `(-z)**(-a_k)*prod gamma(b)*gamma(a_k)... * (q+1)Fq(1/z)`, through
/// [`hypercomb`], which perturbs the parameters where two upper ones differ
/// by an integer.
fn hypq1fq_reciprocal(ap: &[Complex], bq: &[Complex], z: &Complex, prec: usize) -> Option<Complex> {
    let (p, q) = (ap.len(), bq.len());
    let z: Complex = z.clone();
    let function = move |args: &[Complex]| -> Vec<HyperTerm> {
        let wp: usize = args
            .iter()
            .map(|v| parts(v).0.precision())
            .max()
            .unwrap_or(prec)
            .max(prec);
        let (a, b) = (&args[..p], &args[p..]);
        let one: Complex = c_int(1, wp);
        let zz: Complex = raise_to(&z, wp);
        let recz: Complex = &one / &zz;
        let negz: Complex = &c_int(0, wp) - &zz;
        let mut terms: Vec<HyperTerm> = Vec::new();
        for k in 0..=q {
            let ak: &Complex = &a[k];
            let mut gn: Vec<Complex> = b.to_vec();
            gn.push(ak.clone());
            gn.extend((0..=q).filter(|&j| j != k).map(|j| &a[j] - ak));
            let mut gd: Vec<Complex> = a.to_vec();
            gd.extend((0..q).map(|j| &b[j] - ak));
            let mut fn_: Vec<Complex> = vec![ak.clone()];
            fn_.extend((0..q).map(|j| &(ak - &b[j]) + &one));
            let fd: Vec<Complex> = (0..=q).filter(|&j| j != k).map(|j| &(&one - &a[j]) + ak).collect();
            terms.push(HyperTerm {
                bases: vec![negz.clone()],
                exponents: vec![&c_int(0, wp) - ak],
                gamma_numerator: gn,
                gamma_denominator: gd,
                upper: fn_,
                lower: fd,
                argument: recz.clone(),
            });
        }
        terms
    };
    let params: Vec<Complex> = ap.iter().chain(bq).cloned().collect();
    hypercomb(&function, &params, prec + 10).map(|v| raise_to(&v, prec))
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
    if has_infinity(w) || has_infinity(c) {
        // an overflow saturated to an infinity, on which dashu panics
        return None;
    }
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

thread_local! {
    /// The `maxterms` of mpmath's `force_series=True` while an asymptotic
    /// branch runs its `hypercomb`: the series of the terms summed as they
    /// are (divergent ones until their terms are below the precision),
    /// without the transformations of `hyper_value`.
    static FORCED: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// `f` with the series of `hypercomb`'s terms forced (`force_series=True,
/// maxterms=maxterms`).
fn with_forced_series<T>(maxterms: usize, f: impl FnOnce() -> T) -> T {
    /// Restores the previous setting on drop, a panic unwinding through
    /// `f` included (else the later values of the thread stay forced).
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED.with(|c| c.set(self.0));
        }
    }
    let _restore: Restore = Restore(FORCED.with(|c| c.replace(Some(maxterms))));
    f()
}

/// The defining series summed as it is, the convergent and the asymptotic
/// ones alike, until a term is below the working precision; `None` after
/// `maxterms` terms (mpmath's `NoConvergence`).
fn forced_series(ap: &[Complex], bq: &[Complex], z: &Complex, wp: usize, maxterms: usize) -> Option<Complex> {
    let mut term: Complex = c_int(1, wp);
    let mut total: Complex = term.clone();
    for k in 0..maxterms as i64 {
        let kk: Complex = c_int(k, wp);
        let mut num: Complex = z.clone();
        for a in ap {
            num = &num * &(a + &kk);
        }
        let mut den: Complex = c_int(k + 1, wp);
        for b in bq {
            den = &den * &(b + &kk);
        }
        if is_zero_value(&den) {
            return None;
        }
        term = &(&term * &num) / &den;
        total = &total + &term;
        let t: f64 = modulus(&term);
        if !t.is_finite() {
            return None;
        }
        if t == 0.0 || t.log2() < modulus(&total).log2() - wp as f64 {
            return Some(total);
        }
    }
    None
}

/// The value of one term at the working precision; `None` where a
/// factor has no value.
fn term_value(term: &HyperTerm, wp: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::gamma;
    let forced: Option<usize> = FORCED.with(|c| c.get());
    let mut v: Complex = match forced {
        Some(maxterms) => forced_series(&term.upper, &term.lower, &term.argument, wp, maxterms)?,
        None => hyper_value(&term.upper, &term.lower, &term.argument, wp)?,
    };
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

/// The numerical value of `appellf1(a, b1, b2, c, x, y)`, mpmath's
/// `appellf1`: the smaller of `x`, `y` in the outer loop of the double
/// series, which `hyper2d` sums as `sum_m (a)_m (b1)_m/((c)_m m!) x**m
/// 2F1(b2, a + m; c + m; y)`; for `|x| >= 0.99` the transformation to
/// `(x - y)/(x - 1)` (`None` when it does not bring the argument inside,
/// mpmath's "Analytic continuation not implemented").
pub fn appellf1_value(v: &[Complex], prec: usize) -> Option<Complex> {
    // PORTED FROM MPMATH: mpmath/functions/hypergeometric.py:appellf1, hyper2d
    let (a, mut b1, mut b2, c, mut x, mut y) = (
        v[0].clone(),
        v[1].clone(),
        v[2].clone(),
        v[3].clone(),
        v[4].clone(),
        v[5].clone(),
    );
    // the smaller in the outer loop
    if modulus(&x) > modulus(&y) {
        std::mem::swap(&mut x, &mut y);
        std::mem::swap(&mut b1, &mut b2);
    }
    let ok = |z: &Complex| modulus(z) < 0.99;
    // the finite cases
    if nonpositive_integer(&a).is_some() || nonpositive_integer(&b1).is_some() {
    } else if nonpositive_integer(&b2).is_some() {
        std::mem::swap(&mut x, &mut y);
        std::mem::swap(&mut b1, &mut b2);
    } else if !ok(&x) {
        // 2F1 has the analytic continuation, |y| > 1 is fine
        let one: Complex = c_int(1, prec);
        let u1: Complex = &(&x - &y) / &(&x - &one);
        if !ok(&u1) {
            return None;
        }
        let (one_x, one_y) = (&one - &x, &one - &y);
        if modulus(&one_x) == 0.0 || modulus(&one_y) == 0.0 {
            return None;
        }
        let power = |z: &Complex, w: &Complex| -> Complex { (w * &z.ln()).exp() };
        let front: Complex = &power(&one_x, &(&c_int(0, prec) - &b1)) * &power(&one_y, &(&(&c - &a) - &b2));
        let inner: Complex = appellf1_value(
            &[&c - &a, b1.clone(), &(&c - &b1) - &b2, c.clone(), u1, y.clone()],
            prec,
        )?;
        return Some(&front * &inner);
    }
    let wp: usize = prec + 10;
    let tol: f64 = 2f64.powi(1 - wp as i32);
    let maxterms: usize = 20 * prec;
    let (a, b1, b2, c, x, y) = (
        raise_to(&a, wp),
        raise_to(&b1, wp),
        raise_to(&b2, wp),
        raise_to(&c, wp),
        raise_to(&x, wp),
        raise_to(&y, wp),
    );
    let mut s: Complex = c_int(0, wp);
    let mut outer: Complex = c_int(1, wp);
    let mut ok_count: usize = 0;
    let mut m: usize = 0;
    loop {
        let shift: Complex = c_int(m as i64, wp);
        let (am, b1m, cm) = (&a + &shift, &b1 + &shift, &c + &shift);
        let inner: Complex = hyper_value(&[b2.clone(), am.clone()], std::slice::from_ref(&cm), &y, wp)?;
        let term: Complex = &outer * &inner;
        if modulus(&term) < tol {
            ok_count += 1;
        } else {
            ok_count = 0;
        }
        if ok_count >= 3 || modulus(&outer) == 0.0 {
            break;
        }
        s = &s + &term;
        if modulus(&cm) == 0.0 {
            return None;
        }
        outer = &(&(&outer * &b1m) * &am) / &cm;
        m += 1;
        outer = &(&outer * &x) / &c_int(m as i64, wp);
        if m > maxterms {
            return None;
        }
    }
    Some(s)
}

// -- mpmath's asymptotic expansions of 0F1, 1F2, 2F2 and 2F3 for a large
//    argument (functions.wolfram.com), through hypercomb with the series
//    of its terms forced

fn c_rat(p: i64, q: i64, wp: usize) -> Complex {
    complex_real(real_rat(&crate::numbers::rat(p, q), wp), wp)
}

/// mpmath's `mag` of a complex number: `floor(log2|z|) + 1`.
fn mag(z: &Complex) -> f64 {
    let m: f64 = modulus(z);
    if m == 0.0 {
        return f64::NEG_INFINITY;
    }
    m.log2().floor() + 1.0
}

fn all_real(values: &[&Complex]) -> bool {
    values.iter().all(|v| parts(v).1 == Real::ZERO)
}

fn real_part(v: &Complex, wp: usize) -> Complex {
    complex_real(parts(v).0, wp)
}

/// `exp(i x)`.
fn expj(x: &Complex, wp: usize) -> Complex {
    (&complex(Real::ZERO.with_precision(wp).value(), real_int(&Int::ONE, wp)) * x).exp()
}

fn pi_c(wp: usize) -> Complex {
    complex_real(Real::pi(wp), wp)
}

fn term(
    bases: Vec<Complex>,
    exponents: Vec<Complex>,
    gamma_numerator: Vec<Complex>,
    gamma_denominator: Vec<Complex>,
    upper: Vec<Complex>,
    lower: Vec<Complex>,
    argument: Complex,
) -> HyperTerm {
    HyperTerm {
        bases,
        exponents,
        gamma_numerator,
        gamma_denominator,
        upper,
        lower,
        argument,
    }
}

/// `0F1(; b; z)` for `mag(z) >= 8` (mpmath's `_hyp0f1`, Hypergeometric0F1/06/02/03/0004).
fn hyp0f1_asymptotic(b: &Complex, z: &Complex, prec: usize) -> Option<Complex> {
    use crate::functions::special::gamma_numeric::gamma;
    let magz: f64 = mag(z);
    if magz.is_nan() || magz < 8.0 {
        return None;
    }
    let wp: usize = prec + 12 + (magz as usize) / 2;
    let (b, z) = (raise_to(b, wp), raise_to(z, wp));
    let function = |_: &[Complex]| -> Vec<HyperTerm> {
        let w: Complex = crate::core::evalf::complex_sqrt(&(&c_int(0, wp) - &z), wp);
        let jw: Complex = &complex(Real::ZERO.with_precision(wp).value(), real_int(&Int::ONE, wp)) * &w;
        let u: Complex = &c_int(1, wp) / &(&c_int(4, wp) * &jw);
        let c: Complex = &c_rat(1, 2, wp) - &b;
        let e: Complex = (&c_int(2, wp) * &jw).exp();
        let upper: Vec<Complex> = vec![&b - &c_rat(1, 2, wp), &c_rat(3, 2, wp) - &b];
        vec![
            term(
                vec![&c_int(0, wp) - &jw, e.clone()],
                vec![c.clone(), c_int(-1, wp)],
                vec![],
                vec![],
                upper.clone(),
                vec![],
                &c_int(0, wp) - &u,
            ),
            term(
                vec![jw.clone(), e],
                vec![c, c_int(1, wp)],
                vec![],
                vec![],
                upper,
                vec![],
                u.clone(),
            ),
        ]
    };
    let v: Complex = with_forced_series(4 * wp, || hypercomb(&function, &[], wp))?;
    let front: Complex = &gamma(&b, wp)? / &(&c_int(2, wp) * &crate::core::evalf::complex_sqrt(&pi_c(wp), wp));
    let mut v: Complex = &front * &v;
    if all_real(&[&b, &z]) {
        v = real_part(&v, wp);
    }
    Some(raise_to(&v, prec))
}

/// The two exponential sums `s1`, `s2` of the leading series of 1F2 and
/// 2F3, from the coefficients `c(k)`; `None` when the series does not
/// converge quickly enough (mpmath's `NoConvergence`).
fn exponential_sums(c: &dyn Fn(usize) -> Option<Complex>, z: &Complex, wp: usize) -> Option<(Complex, Complex)> {
    let minus_z: Complex = &c_int(0, wp) - z;
    let eps: f64 = 2f64.powi(1 - wp as i32);
    let (mut s1, mut s2) = (c_int(0, wp), c_int(0, wp));
    let i: Complex = complex(Real::ZERO.with_precision(wp).value(), real_int(&Int::ONE, wp));
    let minus_i: Complex = &c_int(0, wp) - &i;
    let (mut p1, mut p2) = (c_int(1, wp), c_int(1, wp));
    let mut tprev: f64 = 0.0;
    let mut negligible: bool = false;
    let half: Complex = c_rat(1, 2, wp);
    let mut k: usize = 0;
    loop {
        let exponent: Complex = &c_rat(-(k as i64), 2, wp) * &c_int(1, wp);
        let ck: Complex = c(k)?;
        let power: Complex = complex_power(&minus_z, &exponent, wp)?;
        if has_infinity(&ck) || has_infinity(&power) {
            return None;
        }
        let w: Complex = &ck * &power;
        if has_infinity(&w) {
            return None;
        }
        let t1: Complex = &p1 * &w;
        let t2: Complex = &p2 * &w;
        let size: f64 = modulus(&t1);
        if !size.is_finite() {
            return None;
        }
        // two terms in a row below the precision: mpmath stops at the first,
        // which ends the sum too early where a coefficient vanishes exactly
        // (c1 of 1F2(1/3; 1/2, 2; z))
        if size < 0.1 * eps {
            if negligible {
                break;
            }
            negligible = true;
        } else {
            negligible = false;
            if k > 5 && tprev / size < 1.5 {
                return None;
            }
        }
        s1 = &s1 + &t1;
        s2 = &s2 + &t2;
        if !negligible {
            tprev = size;
        }
        k += 1;
        p1 = &(&p1 * &minus_i) * &half;
        p2 = &(&p2 * &i) * &half;
        if k > 4 * wp {
            return None;
        }
    }
    Some((s1, s2))
}

/// `1F2(a1; b1, b2; z)` for a large argument (mpmath's `_hyp1f2`,
/// Hypergeometric1F2/06/02/03).
fn hyp1f2_asymptotic(a1: &Complex, b1: &Complex, b2: &Complex, z: &Complex, prec: usize) -> Option<Complex> {
    let absz: f64 = modulus(z);
    if !(absz.log2().floor() + 1.0 > 19.0 && absz.sqrt() > 1.5 * prec as f64) {
        return None;
    }
    let wp: usize = prec + (mag(z) as usize) / 2;
    let (a1, b1, b2, z) = (raise_to(a1, wp), raise_to(b1, wp), raise_to(b2, wp), raise_to(z, wp));
    let function = |p: &[Complex]| -> Vec<HyperTerm> {
        let (a1, b1, b2) = (&p[0], &p[1], &p[2]);
        let r = |n: i64, d: i64| c_rat(n, d, wp);
        let x: Complex = &r(1, 2) * &(&(&(a1 - b1) - b2) + &r(1, 2));
        let d: Complex = &(a1 - b1) - b2;
        let three_a: Complex = &(&(&(&c_int(3, wp) * a1) + b1) + b2) - &c_int(2, wp);
        let bb: Complex = b1 * b2;
        let c1: Complex = &c_int(2, wp) * &(&(&(&r(1, 4) * &(&three_a * &d)) + &bb) - &r(3, 16));
        let inner: Complex = &(&bb + &(&r(1, 4) * &(&d * &three_a))) - &r(3, 16);
        let cubic: Complex = &(&(&(&c_int(-8, wp) * &(a1 * a1)) + &(&c_int(11, wp) * a1)) + &(b1 + b2)) - &c_int(2, wp);
        // c2 = 2*inner**2 + (-16*(2*a1 - 3)*b1*b2 + 4*(a1 - b1 - b2)*cubic - 3)/16
        let first: Complex = &(&c_int(-16, wp) * &(&(&c_int(2, wp) * a1) - &c_int(3, wp))) * &bb;
        let second: Complex = &(&c_int(4, wp) * &d) * &cubic;
        let bracket: Complex = &(&first + &second) - &c_int(3, wp);
        let c2: Complex = &(&c_int(2, wp) * &(&inner * &inner)) + &(&r(1, 16) * &bracket);
        let coefficients = std::cell::RefCell::new(vec![c_int(1, wp), c1, c2]);
        let c = |k: usize| -> Option<Complex> {
            let mut cs = coefficients.borrow_mut();
            while cs.len() <= k {
                let k_: Complex = c_int(cs.len() as i64, wp);
                let n: usize = cs.len();
                // uu1 = 3k**2 + (-6a1 + 2b1 + 2b2 - 4)k + 3a1**2 - (b1 - b2)**2
                //       - 2a1(b1 + b2 - 2) + 1/4
                let linear: Complex =
                    &(&(&(&c_int(-6, wp) * a1) + &(&c_int(2, wp) * b1)) + &(&c_int(2, wp) * b2)) - &c_int(4, wp);
                let diff: Complex = b1 - b2;
                let mut uu1: Complex = &(&c_int(3, wp) * &(&k_ * &k_)) + &(&linear * &k_);
                uu1 = &uu1 + &(&c_int(3, wp) * &(a1 * a1));
                uu1 = &uu1 - &(&diff * &diff);
                uu1 = &uu1 - &(&(&c_int(2, wp) * a1) * &(&(b1 + b2) - &c_int(2, wp)));
                uu1 = &uu1 + &r(1, 4);
                let uu2: Complex = &(&(&(&(&(&k_ - a1) + b1) - b2) - &r(1, 2))
                    * &(&(&(&(&k_ - a1) - b1) + b2) - &r(1, 2)))
                    * &(&(&(&(&k_ - a1) + b1) + b2) - &r(5, 2));
                let next: Complex = &(&(&uu1 * &cs[n - 1]) - &(&uu2 * &cs[n - 2])) / &(&c_int(2, wp) * &k_);
                if has_infinity(&next) {
                    return None;
                }
                cs.push(next);
            }
            Some(cs[k].clone())
        };
        let Some((s1, s2)) = exponential_sums(&c, &z, wp) else {
            return Vec::new();
        };
        let minus_z: Complex = &c_int(0, wp) - &z;
        let phase: Complex = &(&pi_c(wp) * &x) + &(&c_int(2, wp) * &crate::core::evalf::complex_sqrt(&minus_z, wp));
        let s: Complex = &(&expj(&phase, wp) * &s1) + &(&expj(&(&c_int(0, wp) - &phase), wp) * &s2);
        vec![
            term(
                vec![&r(1, 2) * &s, pi_c(wp), minus_z.clone()],
                vec![c_int(1, wp), r(-1, 2), x.clone()],
                vec![b1.clone(), b2.clone()],
                vec![a1.clone()],
                vec![],
                vec![],
                c_int(0, wp),
            ),
            term(
                vec![minus_z.clone()],
                vec![&c_int(0, wp) - a1],
                vec![b1.clone(), b2.clone()],
                vec![b1 - a1, b2 - a1],
                vec![a1.clone(), &(a1 - b1) + &c_int(1, wp), &(a1 - b2) + &c_int(1, wp)],
                vec![],
                &c_int(1, wp) / &z,
            ),
        ]
    };
    asymptotic_combination(
        &function,
        &[a1.clone(), b1.clone(), b2.clone()],
        &[&a1, &b1, &b2, &z],
        prec,
        wp,
    )
}

/// `hypercomb` of an asymptotic branch with its series forced, the real
/// part for real data, `None` when the leading series does not converge
/// (an empty list of terms) or hypercomb gives up.
fn asymptotic_combination(
    function: &dyn Fn(&[Complex]) -> Vec<HyperTerm>,
    params: &[Complex],
    data: &[&Complex],
    prec: usize,
    wp: usize,
) -> Option<Complex> {
    // the leading series of the branch did not converge (mpmath's
    // NoConvergence): an empty list of terms, which hypercomb would sum to 0
    let failed: std::cell::Cell<bool> = std::cell::Cell::new(false);
    let checked = |p: &[Complex]| -> Vec<HyperTerm> {
        let terms: Vec<HyperTerm> = function(p);
        if terms.is_empty() {
            failed.set(true);
        }
        terms
    };
    let found: Option<Complex> = with_forced_series(4 * wp, || hypercomb(&checked, params, wp));
    if failed.get() {
        return None;
    }
    let mut v: Complex = found?;
    if all_real(data) {
        v = real_part(&v, wp);
    }
    Some(raise_to(&v, prec))
}

/// `2F2(a1, a2; b1, b2; z)` for `mag(|z|) > 3` (mpmath's `_hyp2f2`,
/// Hypergeometric2F2/06/02/02/0002).
fn hyp2f2_asymptotic(
    a1: &Complex,
    a2: &Complex,
    b1: &Complex,
    b2: &Complex,
    z: &Complex,
    prec: usize,
) -> Option<Complex> {
    if mag(z).is_nan() || mag(z) <= 3.0 {
        return None;
    }
    let wp: usize = prec + mag(z) as usize;
    let (a1, a2, b1, b2, z) = (
        raise_to(a1, wp),
        raise_to(a2, wp),
        raise_to(b1, wp),
        raise_to(b2, wp),
        raise_to(z, wp),
    );
    let function = |p: &[Complex]| -> Vec<HyperTerm> {
        let (a1, a2, b1, b2) = (&p[0], &p[1], &p[2], &p[3]);
        let x: Complex = &(&(a1 + a2) - b1) - b2;
        let a_2: Complex = a1 + a2;
        let b_2: Complex = b1 + b2;
        let c1: Complex = &(&(&(&a_2 - &c_int(1, wp)) * &x) + &(b1 * b2)) - &(a1 * a2);
        let coefficients = std::cell::RefCell::new(vec![c_int(1, wp), c1]);
        let eps: f64 = 2f64.powi(1 - wp as i32);
        let mut s1: Complex = c_int(0, wp);
        let mut tprev: f64 = 0.0;
        let mut negligible: bool = false;
        let mut zk: Complex = c_int(1, wp);
        let inverse: Complex = &c_int(1, wp) / &z;
        let mut k: usize = 0;
        loop {
            {
                let mut cs = coefficients.borrow_mut();
                while cs.len() <= k {
                    let n: usize = cs.len();
                    let k_: Complex = c_int(n as i64, wp);
                    let uu1: Complex = &(&(&(&(&(&(&(&(&(&c_int(1, wp) - &b_2) + &(&c_int(2, wp) * a1))
                        + &(a1 * a1))
                        + &(&c_int(2, wp) * a2))
                        + &(a2 * a2))
                        - &(&a_2 * &b_2))
                        + &(a1 * a2))
                        + &(b1 * b2))
                        + &(&(&(&c_int(2, wp) * &b_2) - &(&c_int(3, wp) * &(&a_2 + &c_int(1, wp)))) * &k_))
                        + &(&c_int(2, wp) * &(&k_ * &k_));
                    let uu2: Complex = &(&(&(&(&k_ - &a_2) + b1) - &c_int(1, wp))
                        * &(&(&(&k_ - &a_2) + b2) - &c_int(1, wp)))
                        * &(&(&k_ - &x) - &c_int(2, wp));
                    let next: Complex = &(&(&uu1 * &cs[n - 1]) - &(&uu2 * &cs[n - 2])) / &k_;
                    if has_infinity(&next) {
                        return Vec::new();
                    }
                    cs.push(next);
                }
            }
            let t1: Complex = &coefficients.borrow()[k] * &zk;
            let size: f64 = modulus(&t1);
            if !size.is_finite() {
                return Vec::new();
            }
            // two terms in a row below the precision: mpmath stops at the first,
            // which ends the sum too early where a coefficient vanishes exactly
            // (c1 of 1F2(1/3; 1/2, 2; z))
            if size < 0.1 * eps {
                if negligible {
                    break;
                }
                negligible = true;
            } else {
                negligible = false;
                if k > 5 && tprev / size < 1.5 {
                    return Vec::new();
                }
            }
            s1 = &s1 + &t1;
            if !negligible {
                tprev = size;
            }
            k += 1;
            zk = &zk * &inverse;
            if k > 4 * wp {
                return Vec::new();
            }
        }
        let s: Complex = &z.exp() * &s1;
        let minus_z: Complex = &c_int(0, wp) - &z;
        let minus_inverse: Complex = &c_int(0, wp) - &inverse;
        let one: Complex = c_int(1, wp);
        vec![
            term(
                vec![z.clone(), s],
                vec![x.clone(), one.clone()],
                vec![b1.clone(), b2.clone()],
                vec![a1.clone(), a2.clone()],
                vec![],
                vec![],
                c_int(0, wp),
            ),
            term(
                vec![minus_z.clone()],
                vec![&c_int(0, wp) - a1],
                vec![b1.clone(), b2.clone(), a2 - a1],
                vec![a2.clone(), b1 - a1, b2 - a1],
                vec![a1.clone(), &(a1 - b1) + &one, &(a1 - b2) + &one],
                vec![&(a1 - a2) + &one],
                minus_inverse.clone(),
            ),
            term(
                vec![minus_z],
                vec![&c_int(0, wp) - a2],
                vec![b1.clone(), b2.clone(), a1 - a2],
                vec![a1.clone(), b1 - a2, b2 - a2],
                vec![a2.clone(), &(a2 - b1) + &one, &(a2 - b2) + &one],
                vec![&(a2 - a1) + &one],
                minus_inverse,
            ),
        ]
    };
    asymptotic_combination(
        &function,
        &[a1.clone(), a2.clone(), b1.clone(), b2.clone()],
        &[&a1, &a2, &b1, &b2, &z],
        prec,
        wp,
    )
}

/// `2F3(a1, a2; b1, b2, b3; z)` for a large argument (mpmath's `_hyp2f3`,
/// Hypergeometric2F3/06/02/03/01/0002).
fn hyp2f3_asymptotic(a: &[Complex], b: &[Complex], z: &Complex, prec: usize) -> Option<Complex> {
    let absz: f64 = modulus(z);
    if !(absz.log2().floor() + 1.0 > 19.0 && absz.sqrt() > 1.5 * prec as f64) {
        return None;
    }
    let wp: usize = prec + (mag(z) as usize) / 2;
    let params: Vec<Complex> = a.iter().chain(b).map(|v| raise_to(v, wp)).collect();
    let z: Complex = raise_to(z, wp);
    let function = |p: &[Complex]| -> Vec<HyperTerm> {
        let (a1, a2, b1, b2, b3) = (&p[0], &p[1], &p[2], &p[3], &p[4]);
        let r = |n: i64, d: i64| c_rat(n, d, wp);
        let ci = |n: i64| c_int(n, wp);
        let x: Complex = &r(1, 2) * &(&(&(&(&(a1 + a2) - b1) - b2) - b3) + &r(1, 2));
        let a_2: Complex = a1 + a2;
        let b_3: Complex = &(b1 + b2) + b3;
        let aa: Complex = a1 * a2;
        let bb: Complex = &(&(b1 * b2) + &(b3 * b2)) + &(b1 * b3);
        let rr: Complex = &(b1 * b2) * b3;
        let c1: Complex = &ci(2)
            * &(&(&(&bb - &aa) + &(&(&r(1, 4) * &(&(&(&ci(3) * &a_2) + &b_3) - &ci(2))) * &(&a_2 - &b_3))) - &r(3, 16));
        let quad: Complex = &(&(&(&(&ci(-8) * &(&a_2 * &a_2)) + &(&ci(11) * &a_2)) + &(&ci(8) * &aa)) + &b_3) - &ci(2);
        let c2: Complex = &(&r(1, 2) * &(&c1 * &c1))
            + &(&r(1, 16)
                * &(&(&(&(&ci(-16) * &(&(&ci(2) * &a_2) - &ci(3))) * &(&bb - &aa)) + &(&ci(32) * &rr))
                    + &(&(&(&ci(4) * &quad) * &(&a_2 - &b_3)) - &ci(3))));
        let coefficients = std::cell::RefCell::new(vec![ci(1), c1.clone(), c2]);
        let c = |k: usize| -> Option<Complex> {
            let mut cs = coefficients.borrow_mut();
            while cs.len() <= k {
                let n: usize = cs.len();
                let k_: Complex = ci(n as i64);
                let km: Complex = &k_ - &ci(1);
                let two_x: Complex = &ci(2) * &x;
                let uu1: Complex = &(&(&(&(&k_ - &two_x) - &ci(3)) * &(&(&(&k_ - &two_x) - &(&ci(2) * b1)) - &ci(1)))
                    * &(&(&(&k_ - &two_x) - &(&ci(2) * b2)) - &ci(1)))
                    * &(&(&(&k_ - &two_x) - &(&ci(2) * b3)) - &ci(1));
                let km2: Complex = &km * &km;
                let km3: Complex = &km2 * &km;
                let x2: Complex = &x * &x;
                let x3: Complex = &x2 * &x;
                let uu2: Complex = &(&(&(&(&(&(&(&(&ci(4) * &km3)
                    - &(&(&ci(6) * &(&(&ci(4) * &x) + &b_3)) * &km2))
                    + &(&(&ci(2)
                        * &(&(&(&(&(&ci(24) * &x2) + &(&(&ci(12) * &b_3) * &x)) + &(&ci(4) * &bb)) + &b_3)
                            - &ci(1)))
                        * &km))
                    - &(&ci(32) * &x3))
                    - &(&(&ci(24) * &b_3) * &x2))
                    - &(&ci(4) * &bb))
                    - &(&ci(8) * &rr))
                    - &(&(&ci(4) * &(&(&(&ci(4) * &bb) + &b_3) - &ci(1))) * &x))
                    + &(&(&ci(2) * &b_3) - &ci(1));
                let uu3: Complex = &(&(&ci(5) * &km2)
                    + &(&(&ci(2) * &(&(&(&(&ci(-10) * &x) + &a_2) - &(&ci(3) * &b_3)) + &ci(3))) * &km))
                    + &(&ci(2) * &c1);
                let next: Complex =
                    &(&(&(&uu1 * &cs[n - 3]) - &(&uu2 * &cs[n - 2])) + &(&uu3 * &cs[n - 1])) / &(&ci(2) * &k_);
                if has_infinity(&next) {
                    return None;
                }
                cs.push(next);
            }
            Some(cs[k].clone())
        };
        let Some((s1, s2)) = exponential_sums(&c, &z, wp) else {
            return Vec::new();
        };
        let minus_z: Complex = &ci(0) - &z;
        let phase: Complex = &(&pi_c(wp) * &x) + &(&ci(2) * &crate::core::evalf::complex_sqrt(&minus_z, wp));
        let s: Complex = &(&expj(&phase, wp) * &s1) + &(&expj(&(&ci(0) - &phase), wp) * &s2);
        let one: Complex = ci(1);
        let inverse: Complex = &one / &z;
        vec![
            term(
                vec![&r(1, 2) * &s, pi_c(wp), minus_z.clone()],
                vec![one.clone(), r(-1, 2), x.clone()],
                vec![b1.clone(), b2.clone(), b3.clone()],
                vec![a1.clone(), a2.clone()],
                vec![],
                vec![],
                ci(0),
            ),
            term(
                vec![minus_z.clone()],
                vec![&ci(0) - a1],
                vec![b1.clone(), b2.clone(), b3.clone(), a2 - a1],
                vec![a2.clone(), b1 - a1, b2 - a1, b3 - a1],
                vec![a1.clone(), &(a1 - b1) + &one, &(a1 - b2) + &one, &(a1 - b3) + &one],
                vec![&(a1 - a2) + &one],
                inverse.clone(),
            ),
            term(
                vec![minus_z],
                vec![&ci(0) - a2],
                vec![b1.clone(), b2.clone(), b3.clone(), a1 - a2],
                vec![a1.clone(), b1 - a2, b2 - a2, b3 - a2],
                vec![a2.clone(), &(a2 - b1) + &one, &(a2 - b2) + &one, &(a2 - b3) + &one],
                vec![&(a2 - a1) + &one],
                inverse,
            ),
        ]
    };
    let data: Vec<&Complex> = params.iter().chain(std::iter::once(&z)).collect();
    asymptotic_combination(&function, &params, &data, prec, wp)
}

/// The asymptotic branch of mpmath's `hyper` for `0F1`, `1F2`, `2F2` and
/// `2F3` at a large argument, `None` elsewhere (the defining series then).
fn small_pq_asymptotic(ap: &[Complex], bq: &[Complex], z: &Complex, prec: usize) -> Option<Complex> {
    if has_infinity(z) || ap.iter().chain(bq).any(has_infinity) {
        return None;
    }
    match (ap.len(), bq.len()) {
        (0, 1) => hyp0f1_asymptotic(&bq[0], z, prec),
        (1, 2) => hyp1f2_asymptotic(&ap[0], &bq[0], &bq[1], z, prec),
        (2, 2) => hyp2f2_asymptotic(&ap[0], &ap[1], &bq[0], &bq[1], z, prec),
        (2, 3) => hyp2f3_asymptotic(ap, bq, z, prec),
        _ => None,
    }
}
