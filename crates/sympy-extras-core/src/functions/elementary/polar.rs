//! Port of the polar numbers of `sympy_extras/functions/elementary`
//! (`exponential.py`: `ExpPolar`; `complexes.py`: `PolarLift`,
//! `PeriodicArgument`, `PrincipalBranch`, `polarify`, `unpolarify`): the
//! numbers on the Riemann surface of the logarithm, where
//! `exp_polar(2*I*pi)` is not `1`.

use crate::core::assumptions::{facts_of, is_number, FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::evalf::{complex, numeric_value, parts, Complex, Real};
use crate::core::expr::is;
use crate::error::PolyResult;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn factors_of(z: &Basic) -> Vec<Basic> {
    if z.is_mul() {
        z.args().to_vec()
    } else {
        vec![z.clone()]
    }
}
fn imaginary() -> Basic {
    Basic::atom(Head::ImaginaryUnit)
}
fn is_oo(e: &Basic) -> bool {
    matches!(e.head(), Head::Infinity)
}
fn arg(z: &Basic) -> R {
    super::argument::arg(z)
}
fn abs_(z: &Basic) -> R {
    super::complexes::abs_(z)
}
fn is_true_boolean(b: &Basic) -> bool {
    *b == crate::core::logic::true_()
}

/// `exp_polar(z)`: never evaluates.
pub fn exp_polar(z: &Basic) -> R {
    Ok(node(Function::ExpPolar, vec![z.clone()]))
}

/// `exp_polar(a)**b` is `exp_polar(a*b)` (`ExpPolar._eval_power_node`).
pub fn exp_polar_power(base: &Basic, exponent: &Basic) -> R {
    exp_polar(&m2(&base.args()[0], exponent)?)
}

/// `(exp_polar(1), z)`, `(self, 1)` for `exp_polar(0)` (`ExpPolar.as_base_exp`).
pub fn exp_polar_base_exp(e: &Basic) -> PolyResult<(Basic, Basic)> {
    if yes(&e.args()[0], Predicate::zero) {
        return Ok((e.clone(), int(1)));
    }
    Ok((node(Function::ExpPolar, vec![int(1)]), e.args()[0].clone()))
}

/// `polar_lift(z)`, evaluated (`PolarLift.eval`).
pub fn polar_lift(z: &Basic) -> R {
    if is_number(z) {
        let angle: Basic = arg(z)?;
        let half_pi: Basic = d2(&Basic::pi(), &int(2))?;
        let special: [Basic; 4] = [int(0), half_pi.clone(), crate::core::mul::neg(&half_pi)?, Basic::pi()];
        if special.contains(&angle) {
            return m2(&exp_polar(&m2(&imaginary(), &angle)?)?, &abs_(z)?);
        }
    }
    let factors: Vec<Basic> = factors_of(z);
    let (mut included, mut excluded, mut positive) = (Vec::new(), Vec::new(), Vec::new());
    for f in &factors {
        if yes(f, Predicate::polar) {
            included.push(f.clone());
        } else if yes(f, Predicate::positive) {
            positive.push(f.clone());
        } else {
            excluded.push(f.clone());
        }
    }
    if excluded.len() < factors.len() {
        let mut kept: Vec<Basic> = included.clone();
        kept.extend(positive.iter().cloned());
        if !excluded.is_empty() {
            return m2(&mul_all(&kept)?, &polar_lift(&mul_all(&excluded)?)?);
        }
        if !included.is_empty() {
            return mul_all(&kept);
        }
        return m2(&mul_all(&positive)?, &exp_polar(&int(0))?);
    }
    Ok(node(Function::PolarLift, vec![z.clone()]))
}

/// The argument of `z` on the Riemann surface of the logarithm
/// (`_unbranched`).
fn unbranched(z: &Basic) -> PolyResult<Option<Basic>> {
    let mut total: Basic = int(0);
    for a in factors_of(z) {
        if !yes(&a, Predicate::polar) {
            total = a2(&total, &arg(&a)?)?;
        } else if func(&a) == Some(Function::ExpPolar) {
            let Some((_, im)) = super::complexes::as_real_imag(&a.args()[0], true, None)? else {
                return Ok(None);
            };
            total = a2(&total, &im)?;
        } else if a.is_pow() {
            let Some((re, im)) = super::complexes::as_real_imag(&a.args()[1], true, None)? else {
                return Ok(None);
            };
            let base: &Basic = &a.args()[0];
            let modulus: Basic = super::exponential::log(&abs_(base)?)?;
            total = crate::core::add::add(&[total, m2(&re, &unbranched_argument(base)?)?, m2(&im, &modulus)?])?;
        } else if func(&a) == Some(Function::PolarLift) {
            total = a2(&total, &arg(&a.args()[0])?)?;
        } else {
            return Ok(None);
        }
    }
    Ok(Some(total))
}

/// `periodic_argument(z, period)`, evaluated (`PeriodicArgument.eval`).
pub fn periodic_argument(z: &Basic, period: &Basic) -> R {
    let unevaluated = || Ok(node(Function::PeriodicArgument, vec![z.clone(), period.clone()]));
    if !yes(period, Predicate::extended_positive) {
        return unevaluated();
    }
    if is_oo(period) && func(z) == Some(Function::PrincipalBranch) {
        return periodic_argument(&z.args()[0], &z.args()[1]);
    }
    if func(z) == Some(Function::PolarLift)
        && is_true_boolean(&crate::core::relational::ge(period, &m2(&int(2), &Basic::pi())?)?)
    {
        return periodic_argument(&z.args()[0], period);
    }
    if z.is_mul() {
        let kept: Vec<Basic> = z
            .args()
            .iter()
            .filter(|x| !yes(x, Predicate::positive))
            .cloned()
            .collect();
        if kept.len() != z.args().len() {
            return periodic_argument(&mul_all(&kept)?, period);
        }
    }
    let Some(whole) = unbranched(z)? else {
        return unevaluated();
    };
    let blocked: bool = whole.preorder().iter().any(|n| {
        matches!(
            func(n),
            Some(Function::PeriodicArgument | Function::Atan2 | Function::Atan)
        )
    });
    if blocked {
        return unevaluated();
    }
    if is_oo(period) {
        return Ok(whole);
    }
    let shift: Basic = s2(&d2(&whole, period)?, &Basic::from_rational(crate::numbers::rat(1, 2)))?;
    let n: Basic = m2(&super::integers::rounding(Function::Ceiling, &shift)?, period)?;
    if !n.preorder().iter().any(|x| func(x) == Some(Function::Ceiling)) {
        return s2(&whole, &n);
    }
    unevaluated()
}

/// `periodic_argument(z, oo)`.
pub fn unbranched_argument(z: &Basic) -> R {
    periodic_argument(z, &Basic::oo())
}

fn replace_polar_lifts(e: &Basic) -> R {
    if func(e) == Some(Function::PolarLift) {
        let inner: Basic = replace_polar_lifts(&e.args()[0])?;
        return if inner.is_symbol() {
            Ok(inner)
        } else {
            polar_lift(&inner)
        };
    }
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let args: Vec<Basic> = e.args().iter().map(replace_polar_lifts).collect::<PolyResult<_>>()?;
    crate::core::function::rebuild_evaluated(e, args)
}

/// `principal_branch(x, period)`, evaluated (`PrincipalBranch.eval`).
pub fn principal_branch(x: &Basic, period: &Basic) -> R {
    let unevaluated = || Ok(node(Function::PrincipalBranch, vec![x.clone(), period.clone()]));
    if func(x) == Some(Function::PolarLift) {
        return principal_branch(&x.args()[0], period);
    }
    if is_oo(period) {
        return Ok(x.clone());
    }
    let has_periodic = |e: &Basic| e.preorder().iter().any(|n| func(n) == Some(Function::PeriodicArgument));
    let mut whole: Basic = periodic_argument(x, &Basic::oo())?;
    let branch: Basic = periodic_argument(x, period)?;
    if whole != branch && !has_periodic(&whole) && !has_periodic(&branch) {
        let lifted: Basic = polar_lift(x)?;
        let replaced: Basic = replace_polar_lifts(&lifted)?;
        whole = periodic_argument(&replaced, &Basic::oo())?;
        if !replaced.preorder().iter().any(|n| func(n) == Some(Function::PolarLift)) {
            let mut result: Basic = if whole != branch {
                m2(&exp_polar(&m2(&imaginary(), &s2(&branch, &whole)?)?)?, &replaced)?
            } else {
                replaced
            };
            if !yes(&result, Predicate::polar) && !result.preorder().iter().any(|n| func(n) == Some(Function::ExpPolar))
            {
                result = m2(&result, &exp_polar(&int(0))?)?;
            }
            return Ok(result);
        }
    }
    let (mut c, rest): (Basic, Vec<Basic>) = if x.free_symbols().is_empty() {
        (x.clone(), Vec::new())
    } else {
        let all: Vec<Basic> = factors_of(x);
        let constant: Vec<Basic> = all.iter().filter(|f| f.free_symbols().is_empty()).cloned().collect();
        (
            mul_all(&constant)?,
            all.into_iter().filter(|f| !f.free_symbols().is_empty()).collect(),
        )
    };
    let mut others: Vec<Basic> = Vec::new();
    for y in rest {
        if yes(&y, Predicate::positive) {
            c = m2(&c, &y)?;
        } else {
            others.push(y);
        }
    }
    let angle: Basic = periodic_argument(&c, period)?;
    if has_periodic(&angle) {
        return unevaluated();
    }
    let numeric: bool = is_number(&angle);
    if numeric && (unbranched_argument(&c)? != angle || (angle == int(0) && !others.is_empty() && c != int(1))) {
        if angle == int(0) {
            return m2(&abs_(&c)?, &principal_branch(&mul_all(&others)?, period)?);
        }
        let mut inside: Vec<Basic> = vec![exp_polar(&m2(&imaginary(), &angle)?)?];
        inside.extend(others.iter().cloned());
        return m2(&principal_branch(&mul_all(&inside)?, period)?, &abs_(&c)?);
    }
    let half: Basic = d2(period, &int(2))?;
    if numeric
        && others.is_empty()
        && (is_true_boolean(&crate::core::relational::lt(&abs_(&angle)?, &half)?) || angle == half)
    {
        return m2(&exp_polar(&m2(&angle, &imaginary())?)?, &abs_(&c)?);
    }
    unevaluated()
}

/// The facts of the polar nodes (`_eval_facts`).
pub fn polar_facts(f: Function, args: &[Basic]) -> PolyResult<FactList> {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    match f {
        Function::ExpPolar => {
            facts.set(Predicate::polar, true);
            facts.set(Predicate::zero, false);
            let z: &Basic = &args[0];
            if facts_of(z).get(&Predicate::extended_real) == Some(&true) {
                facts.set(Predicate::extended_real, true);
            }
            if is_number(z) && z.free_symbols().is_empty() {
                if let Some((re, im)) = super::complexes::as_real_imag(z, true, None)? {
                    if facts_of(&re).get(&Predicate::real) == Some(&true) {
                        let ratio: Basic = d2(&im, &Basic::pi())?;
                        if yes(&ratio, Predicate::zero) {
                            facts.set(Predicate::extended_real, true);
                            facts.set(Predicate::real, true);
                            facts.set(Predicate::positive, true);
                            facts.set(Predicate::negative, false);
                        } else if ratio.is_number_node() && ratio == int(1) {
                            facts.set(Predicate::extended_real, true);
                            facts.set(Predicate::real, true);
                            facts.set(Predicate::positive, false);
                            facts.set(Predicate::negative, true);
                        }
                    }
                }
            }
        }
        Function::PolarLift | Function::PrincipalBranch => facts.set(Predicate::polar, true),
        _ => {}
    }
    Ok(facts)
}

fn pi_at(prec: usize) -> Real {
    Real::pi(prec)
}

/// The numerical values of the polar nodes (`_numeric_node`, `numeric`).
pub fn polar_value(e: &Basic, prec: usize) -> Option<Complex> {
    let f: Function = func(e)?;
    match f {
        Function::ExpPolar => {
            let z: &Basic = &e.args()[0];
            if let Some(part) = numeric_value(&super::complexes::im(z).ok()?, prec) {
                let (re, im) = parts(&part);
                let pi: Real = pi_at(prec);
                if im == Real::ZERO && -pi.clone() < re && re <= pi {
                    let value: Basic = super::exponential::exp(z).ok()?;
                    if func(&value) != Some(Function::Exp) {
                        return numeric_value(&value, prec);
                    }
                }
            }
            let value: Complex = numeric_value(z, prec)?;
            let (_, zi) = parts(&value);
            let pi: Real = pi_at(prec + 20);
            if !(-pi.clone() < zi && zi <= pi) {
                return None;
            }
            let result: Complex = value.exp();
            let (rr, ri) = parts(&result);
            if zi > Real::ZERO && ri < Real::ZERO {
                return Some(complex(rr, Real::ZERO));
            }
            Some(result)
        }
        Function::PolarLift => numeric_value(&e.args()[0], prec),
        Function::PeriodicArgument => periodic_value(&e.args()[0], &e.args()[1], prec),
        Function::PrincipalBranch => {
            let (z, period) = (&e.args()[0], &e.args()[1]);
            let p: Complex = periodic_value(z, period, prec)?;
            let (angle, im) = parts(&p);
            if im != Real::ZERO {
                return None;
            }
            let modulus: Complex = numeric_value(&abs_(z).ok()?, prec)?;
            let pi: Real = pi_at(prec + 20);
            let size: Real = if angle < Real::ZERO {
                -angle.clone()
            } else {
                angle.clone()
            };
            if size > pi || angle == -pi {
                return None;
            }
            let rotation: Complex = complex(Real::ZERO, angle).exp();
            Some(&modulus * &rotation)
        }
        _ => None,
    }
}

fn periodic_value(z: &Basic, period: &Basic, prec: usize) -> Option<Complex> {
    if is_oo(period) {
        return numeric_value(&unbranched(z).ok()??, prec);
    }
    let ub: Complex = periodic_value(z, &Basic::oo(), prec)?;
    let p: Complex = numeric_value(period, prec)?;
    let ((u, ui), (q, qi)) = (parts(&ub), parts(&p));
    if ui != Real::ZERO || qi != Real::ZERO || q == Real::ZERO {
        return None;
    }
    let half: Real = crate::core::evalf::real_rat(&crate::numbers::rat(1, 2), prec + 20);
    let n: Real = (&u / &q - &half).ceil();
    Some(complex(&u - &(&n * &q), Real::ZERO))
}

fn polarify_node(e: &Basic, lift: bool, pause: bool) -> R {
    if yes(e, Predicate::polar) {
        return Ok(e.clone());
    }
    if is_number(e) && !pause {
        return polar_lift(e);
    }
    if e.is_symbol() && !pause && lift {
        return polar_lift(e);
    }
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    if e.is_add() {
        let args: Vec<Basic> = e
            .args()
            .iter()
            .map(|a| polarify_node(a, lift, true))
            .collect::<PolyResult<_>>()?;
        let rebuilt: Basic = crate::core::add::add(&args)?;
        return if lift { polar_lift(&rebuilt) } else { Ok(rebuilt) };
    }
    if e.is_pow() && matches!(e.args()[0].head(), Head::Exp1) {
        return crate::core::power::pow(&e.args()[0], &polarify_node(&e.args()[1], lift, false)?);
    }
    if matches!(e.head(), Head::Function(_)) {
        let args: Vec<Basic> = e
            .args()
            .iter()
            .map(|a| polarify_node(a, lift, false))
            .collect::<PolyResult<_>>()?;
        return crate::core::function::rebuild_evaluated(e, args);
    }
    let args: Vec<Basic> = e
        .args()
        .iter()
        .map(|a| polarify_node(a, lift, pause))
        .collect::<PolyResult<_>>()?;
    crate::core::function::rebuild_evaluated(e, args)
}

/// `e` with its numbers (and, with `lift`, its symbols) polar; with
/// `subs` its symbols replaced by polar dummies and the map back
/// (`polarify`).
pub fn polarify(e: &Basic, subs: bool, lift: bool) -> PolyResult<(Basic, Vec<(Basic, Basic)>)> {
    let subs: bool = subs && !lift;
    let result: Basic = polarify_node(e, lift, false)?;
    if !subs {
        return Ok((result, Vec::new()));
    }
    let mut symbols: Vec<Basic> = result
        .free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_)))
        .collect();
    symbols.sort_by_cached_key(crate::core::sorting::sort_key);
    let mut forward: Vec<(Basic, Basic)> = Vec::new();
    let mut back: Vec<(Basic, Basic)> = Vec::new();
    for s in symbols {
        let name: String = match s.head() {
            Head::Symbol(data) => data.name.clone(),
            _ => continue,
        };
        let d: Basic = crate::core::symbol::Dummy::new(&name, &[(Predicate::polar, true)]).0;
        forward.push((s.clone(), d.clone()));
        back.push((d, s));
    }
    Ok((crate::core::subs::subs_pairs(&result, &forward)?, back))
}

/// The classes with `unbranched = True` (their arguments are unpolarified
/// with the exponents only).
const UNBRANCHED: [&str; 35] = [
    "Abs",
    "AiryAi",
    "AiryAiPrime",
    "AiryBi",
    "AiryBiPrime",
    "Beta",
    "BetaInc",
    "BetaIncRegularized",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "Erf",
    "Erfc",
    "Erfi",
    "Exp",
    "ExpPolar",
    "FresnelC",
    "FresnelS",
    "Gamma",
    "Im",
    "MathieuC",
    "MathieuCPrime",
    "MathieuS",
    "MathieuSPrime",
    "MultiGamma",
    "Re",
    "Sec",
    "Sech",
    "Sin",
    "Sinh",
    "Tan",
    "Tanh",
];

fn is_unbranched(e: &Basic) -> bool {
    match e.head() {
        Head::Function(f) => UNBRANCHED.contains(&f.class_name()),
        _ => false,
    }
}

fn unpolarify_node(e: &Basic, exponents_only: bool, pause: bool) -> R {
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    if !pause {
        match func(e) {
            Some(Function::ExpPolar) => {
                return super::exponential::exp(&unpolarify_node(&e.args()[0], exponents_only, false)?);
            }
            Some(Function::PrincipalBranch) if e.args()[1] == m2(&int(2), &Basic::pi())? => {
                return unpolarify_node(&e.args()[0], exponents_only, false);
            }
            Some(Function::PolarLift) => return unpolarify_node(&e.args()[0], exponents_only, false),
            _ => {}
        }
        let relation_ok: bool = match e.head() {
            Head::Relational(op) => {
                !matches!(op, crate::core::basic::RelOp::Eq | crate::core::basic::RelOp::Ne)
                    || e.args().contains(&int(0))
            }
            _ => false,
        };
        let boolean: bool = matches!(
            e.head(),
            Head::And | Head::Or | Head::Not | Head::Xor | Head::Implies | Head::Equivalent
        );
        if e.is_add() || e.is_mul() || boolean || relation_ok {
            let args: Vec<Basic> = e
                .args()
                .iter()
                .map(|a| unpolarify_node(a, exponents_only, false))
                .collect::<PolyResult<_>>()?;
            return crate::core::function::rebuild_evaluated(e, args);
        }
    }
    if e.is_pow() {
        let exponent: Basic = unpolarify_node(&e.args()[1], exponents_only, false)?;
        let integer: bool = yes(&exponent, Predicate::integer);
        let base: Basic = unpolarify_node(&e.args()[0], exponents_only, !(integer && !pause))?;
        return crate::core::power::pow(&base, &exponent);
    }
    if is_unbranched(e) {
        let args: Vec<Basic> = e
            .args()
            .iter()
            .map(|a| unpolarify_node(a, exponents_only, exponents_only))
            .collect::<PolyResult<_>>()?;
        return crate::core::function::rebuild_evaluated(e, args);
    }
    let args: Vec<Basic> = e
        .args()
        .iter()
        .map(|a| unpolarify_node(a, exponents_only, true))
        .collect::<PolyResult<_>>()?;
    crate::core::function::rebuild_evaluated(e, args)
}

/// `e` with its polar numbers made ordinary where that keeps its value
/// (`unpolarify`).
pub fn unpolarify(e: &Basic, exponents_only: bool) -> R {
    let mut current: Basic = e.clone();
    let pause: bool = exponents_only;
    loop {
        let result: Basic = unpolarify_node(&current, exponents_only, pause)?;
        if result == current {
            break;
        }
        current = result;
    }
    let rules: Vec<(Basic, Basic)> = vec![(exp_polar(&int(0))?, int(1)), (polar_lift(&int(0))?, int(0))];
    crate::core::subs::subs_pairs(&current, &rules)
}
