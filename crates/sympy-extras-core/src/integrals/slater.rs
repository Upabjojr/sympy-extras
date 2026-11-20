//! Port of `sympy_extras/integrals/slater.py`: a quotient of gamma
//! functions as the Mellin-Barnes integral of a Meijer G-function
//! (Gauss' multiplication formula for the scaled arguments), and the
//! G-function expanded by Slater's theorem: the residues at the poles of
//! the `gamma(b_j - s)` as a sum of hypergeometric series, by the
//! perturbation of the coincident parameters and a limit in the
//! logarithmic case; `line_conditions` gives the validity of the integral
//! along a vertical line.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;
use crate::integrals::mellin::{GammaFactor, GammaQuotient};
use crate::numbers::{denom, numer, Int, Rat};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn half() -> Basic {
    Basic::from_rational(crate::numbers::rat(1, 2))
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
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
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn gamma(z: &Basic) -> R {
    crate::functions::special::gamma_functions::gamma(z)
}
fn abs_(z: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(z)
}
fn has_head(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn has_hyper(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::Hyper))
}
fn has_meijerg(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::MeijerG))
}
fn has_infinity(e: &Basic) -> bool {
    has_head(e, &|n| {
        matches!(n.head(), Head::Infinity | Head::ComplexInfinity | Head::NaN)
    })
}
fn is_true_boolean(b: &Basic) -> bool {
    *b == crate::core::logic::true_()
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn ask(query: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    crate::assumptions::ask::ask(query, assumptions, None)
}

/// `prefactor * meijerg(an, ap, bm, bq, z)` (`MeijerG`).
#[derive(Debug, Clone, PartialEq)]
pub struct MeijerG {
    pub prefactor: Basic,
    pub an: Vec<Basic>,
    pub ap: Vec<Basic>,
    pub bm: Vec<Basic>,
    pub bq: Vec<Basic>,
    pub z: Basic,
}

impl MeijerG {
    pub fn m(&self) -> usize {
        self.bm.len()
    }
    pub fn n(&self) -> usize {
        self.an.len()
    }
    pub fn p(&self) -> usize {
        self.an.len() + self.ap.len()
    }
    pub fn q(&self) -> usize {
        self.bm.len() + self.bq.len()
    }

    /// `delta = m + n - (p + q)/2`.
    pub fn delta(&self) -> Rat {
        crate::numbers::rat(2 * (self.m() + self.n()) as i64 - self.p() as i64 - self.q() as i64, 2)
    }

    pub fn as_expr(&self) -> R {
        crate::functions::special::hyper::meijerg(&self.an, &self.ap, &self.bm, &self.bq, &self.z)
    }

    /// The G-function of `1/z` with the parameters `1 - b`, `1 - a` (`reflected`).
    pub fn reflected(&self) -> PolyResult<MeijerG> {
        let one_minus = |v: &[Basic]| v.iter().map(|x| s2(&int(1), x)).collect::<PolyResult<Vec<Basic>>>();
        Ok(MeijerG {
            prefactor: self.prefactor.clone(),
            an: one_minus(&self.bm)?,
            ap: one_minus(&self.bq)?,
            bm: one_minus(&self.an)?,
            bq: one_minus(&self.ap)?,
            z: d2(&int(1), &self.z)?,
        })
    }
}

fn rational_of(b: &Basic) -> Rat {
    b.as_rational().cloned().unwrap_or(Rat::ZERO)
}

/// Gauss' multiplication formula for `gamma(k*s + shift)` (`_multiplication`).
fn multiplication(shift: &Basic, k: i64) -> PolyResult<(Basic, Basic, Vec<GammaFactor>)> {
    let size: i64 = k.abs();
    let constant: Basic = m2(
        &pw(
            &m2(&int(2), &Basic::pi())?,
            &Basic::from_rational(crate::numbers::rat(1 - size, 2)),
        )?,
        &pw(&int(size), &s2(shift, &half())?)?,
    )?;
    let sign: Basic = int(if k > 0 { 1 } else { -1 });
    let factors: Vec<GammaFactor> = (0..size)
        .map(|j| Ok((d2(&a2(shift, &int(j))?, &int(size))?, sign.clone())))
        .collect::<PolyResult<_>>()?;
    Ok((constant, int(size), factors))
}

/// The Meijer G-function whose Mellin-Barnes integrand is the quotient of
/// gamma functions, or `None` (`mellin_barnes`).
pub fn mellin_barnes(quotient: &GammaQuotient) -> PolyResult<Option<MeijerG>> {
    if !quotient.is_gamma_quotient() {
        return Ok(None);
    }
    let mut lam: Int = Int::from(1);
    for (_, b) in quotient.numerator.iter().chain(quotient.denominator.iter()) {
        let r: Rat = rational_of(b);
        if r != Rat::ZERO {
            lam = crate::numbers::lcm(&lam, &denom(&r));
        }
    }
    let lam_basic: Basic = Basic::from_rational(Rat::from(lam.clone()));
    let mut constant: Basic = m2(&quotient.constant, &lam_basic)?;
    let mut z: Basic = int(1);
    for (base, coefficient) in &quotient.bases {
        z = m2(&z, &pw(base, &m2(coefficient, &lam_basic)?)?)?;
    }
    let (mut an, mut ap, mut bm, mut bq) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (factors, numerator) in [(&quotient.numerator, true), (&quotient.denominator, false)] {
        for (shift, scale) in factors {
            let r: Rat = rational_of(scale);
            let k: i64 = i64::try_from(&(numer(&r) * (&lam / denom(&r))))
                .map_err(|_| crate::error::value_error("a large scale".to_string()))?;
            if k == 0 {
                constant = if numerator {
                    m2(&constant, &gamma(shift)?)?
                } else {
                    d2(&constant, &gamma(shift)?)?
                };
                continue;
            }
            let mut unit: Vec<GammaFactor> = vec![(shift.clone(), int(k))];
            if k.abs() > 1 {
                let (c, base, u) = multiplication(shift, k)?;
                unit = u;
                if numerator {
                    constant = m2(&constant, &c)?;
                    z = m2(&z, &pw(&base, &int(k))?)?;
                } else {
                    constant = d2(&constant, &c)?;
                    z = m2(&z, &pw(&base, &int(-k))?)?;
                }
            }
            for (a, b) in unit {
                let positive: bool = rational_of(&b) > Rat::ZERO;
                if numerator && positive {
                    an.push(s2(&int(1), &a)?);
                } else if numerator {
                    bm.push(a);
                } else if positive {
                    bq.push(s2(&int(1), &a)?);
                } else {
                    ap.push(a);
                }
            }
        }
    }
    Ok(Some(MeijerG {
        prefactor: constant,
        an,
        ap,
        bm,
        bq,
        z,
    }))
}

fn integer_difference(a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
    Ok(is(&s2(a, b)?, Predicate::integer))
}

fn slater_series(g: &MeijerG) -> PolyResult<Option<Basic>> {
    let bs: Vec<Basic> = g.bm.iter().chain(g.bq.iter()).cloned().collect();
    let as_: Vec<Basic> = g.an.iter().chain(g.ap.iter()).cloned().collect();
    let mut coincident: Vec<usize> = Vec::new();
    for i in 0..g.m() {
        for j in i + 1..g.m() {
            match integer_difference(&g.bm[i], &g.bm[j])? {
                None => return Ok(None),
                Some(true) if !coincident.contains(&j) => coincident.push(j),
                _ => {}
            }
        }
    }
    if !coincident.is_empty() {
        return logarithmic_case(g, &coincident);
    }
    let mut total: Basic = int(0);
    let sign: Basic = pw(&int(-1), &int(g.p() as i64 - g.m() as i64 - g.n() as i64))?;
    for h in 0..g.m() {
        let bh: &Basic = &g.bm[h];
        let mut coefficient: Basic = int(1);
        for j in 0..g.m() {
            if j != h {
                coefficient = m2(&coefficient, &gamma(&s2(&g.bm[j], bh)?)?)?;
            }
        }
        for a in &g.an {
            coefficient = m2(&coefficient, &gamma(&s2(&a2(&int(1), bh)?, a)?)?)?;
        }
        for b in &g.bq {
            coefficient = d2(&coefficient, &gamma(&s2(&a2(&int(1), bh)?, b)?)?)?;
        }
        for a in &g.ap {
            coefficient = d2(&coefficient, &gamma(&s2(a, bh)?)?)?;
        }
        let upper: Vec<Basic> = as_
            .iter()
            .map(|a| s2(&a2(&int(1), bh)?, a))
            .collect::<PolyResult<_>>()?;
        let lower: Vec<Basic> = bs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != h)
            .map(|(_, b)| s2(&a2(&int(1), bh)?, b))
            .collect::<PolyResult<_>>()?;
        let series: Basic = crate::functions::special::hyper::hyper(&upper, &lower, &m2(&sign, &g.z)?)?;
        total = a2(&total, &m2(&m2(&coefficient, &pw(&g.z, bh)?)?, &series)?)?;
    }
    Ok(Some(total))
}

fn logarithmic_case(g: &MeijerG, coincident: &[usize]) -> PolyResult<Option<Basic>> {
    let epsilon: Basic =
        crate::core::symbol::Dummy::new("epsilon", &[(Predicate::positive, true), (Predicate::rational, false)]).0;
    let mut bm: Vec<Basic> = Vec::new();
    for (i, b) in g.bm.iter().enumerate() {
        match coincident.iter().position(|j| *j == i) {
            Some(index) => bm.push(a2(b, &m2(&int(index as i64 + 1), &epsilon)?)?),
            None => bm.push(b.clone()),
        }
    }
    let perturbed = MeijerG {
        prefactor: g.prefactor.clone(),
        an: g.an.clone(),
        ap: g.ap.clone(),
        bm,
        bq: g.bq.clone(),
        z: g.z.clone(),
    };
    let mut value: Option<Basic> = None;
    if let Some(series) = slater_series(&perturbed)? {
        let expanded: Basic = hyperexpand_safely(&series)?;
        if !has_hyper(&expanded) {
            let budget: Option<f64> = timeout().map(|t| t / 8.0);
            value = crate::timeout::attempt_within(budget, || {
                crate::core::limits::limit(&expanded, &epsilon, &int(0), "+")
            })?;
        }
    }
    if let Some(v) = value {
        if !(has_infinity(&v) || has_head(&v, &|n| matches!(n.head(), Head::Limit))) {
            return Ok(Some(v));
        }
    }
    let direct: Basic = match hyperexpand_safely(&g.as_expr()?) {
        Ok(d) => d,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    if has_meijerg(&direct) || has_hyper(&direct) || has_infinity(&direct) {
        return Ok(None);
    }
    Ok(Some(direct))
}

fn hyperexpand_safely(e: &Basic) -> R {
    use crate::simplify::hyperexpand::{hyperexpand, Rewrite};
    let expanded = crate::timeout::attempt_within(timeout(), || hyperexpand(e, false, Rewrite::Default, None))?;
    let mut result: Basic = expanded.unwrap_or_else(|| e.clone());
    if has_hyper(&result) {
        result = incomplete_gamma_forms(&result)?;
    }
    Ok(result)
}

fn incomplete_gamma_forms(e: &Basic) -> R {
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    let mut seen: Vec<Basic> = Vec::new();
    for h in e.preorder() {
        if !matches!(h.head(), Head::Hyper) || seen.contains(&h) {
            continue;
        }
        seen.push(h.clone());
        let (ap, bq, z) = (h.args()[0].args(), h.args()[1].args(), &h.args()[2]);
        if ap.len() == 1 && ap[0] == int(1) && bq.len() == 1 {
            let b: &Basic = &bq[0];
            let b1: Basic = s2(b, &int(1))?;
            if (is(&b1, Predicate::integer) == Some(true) && is(&b1, Predicate::nonpositive) == Some(true))
                || is(z, Predicate::negative) == Some(true)
            {
                continue;
            }
            let value: Basic = m2(
                &m2(
                    &m2(&b1, &pw(z, &s2(&int(1), b)?)?)?,
                    &crate::functions::elementary::exponential::exp(z)?,
                )?,
                &crate::functions::special::gamma_functions::lowergamma(&b1, z)?,
            )?;
            replacement.push((h.clone(), value));
        }
    }
    if replacement.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = replacement.into_iter().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

/// `g` expanded by Slater's theorem when it is a G-function
/// (`slater_expansion`).
pub fn slater_expansion(g: &Basic, assumptions: &[Basic]) -> R {
    if !matches!(g.head(), Head::MeijerG) {
        return Ok(g.clone());
    }
    let (a, b) = (&g.args()[0], &g.args()[1]);
    let m = MeijerG {
        prefactor: int(1),
        an: a.args()[0].args().to_vec(),
        ap: a.args()[1].args().to_vec(),
        bm: b.args()[0].args().to_vec(),
        bq: b.args()[1].args().to_vec(),
        z: g.args()[2].clone(),
    };
    Ok(match expand_meijerg(&m, assumptions)? {
        Some(v) => v.value,
        None => g.clone(),
    })
}

fn plain(value: Basic) -> ConditionalValue {
    ConditionalValue::new(value, crate::core::logic::true_())
}

/// The value of the G-function by Slater's theorem: the series in `z` or in
/// `1/z`, a `Piecewise` of both on the unit circle (`expand_meijerg`).
pub fn expand_meijerg(g: &MeijerG, assumptions: &[Basic]) -> PolyResult<Option<ConditionalValue>> {
    let fallback = |g: &MeijerG| -> PolyResult<Option<ConditionalValue>> {
        let expanded: Basic = match hyperexpand_safely(&g.as_expr()?) {
            Ok(x) => x,
            Err(PolyError::Value(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        if has_meijerg(&expanded) {
            return Ok(None);
        }
        Ok(Some(plain(m2(&g.prefactor, &expanded)?)))
    };
    let series: Option<Basic> = if g.p() < g.q() {
        slater_series(g)?
    } else if g.p() > g.q() {
        slater_series(&g.reflected()?)?
    } else {
        let modulus: Basic = abs_(&g.z)?;
        let inside = ask(&crate::core::relational::lt(&modulus, &int(1))?, assumptions)?;
        if inside == Some(true) {
            slater_series(g)?
        } else if inside == Some(false)
            && ask(&crate::core::relational::gt(&modulus, &int(1))?, assumptions)? == Some(true)
        {
            slater_series(&g.reflected()?)?
        } else if inside == Some(false) {
            let mut series: Option<Basic> = slater_series(g)?;
            if let Some(s) = &series {
                let mut all: bool = true;
                let mut seen: Vec<Basic> = Vec::new();
                for h in s.preorder() {
                    if matches!(h.head(), Head::Hyper) && !seen.contains(&h) {
                        seen.push(h.clone());
                        if !converges_on_the_circle(&h, assumptions)? {
                            all = false;
                            break;
                        }
                    }
                }
                if !all {
                    return Ok(None);
                }
                let expanded: Basic = hyperexpand_safely(s)?;
                if has_hyper(&expanded)
                    || has_head(&expanded, &|n| matches!(n.head(), Head::ComplexInfinity | Head::NaN))
                {
                    return Ok(None);
                }
                series = Some(expanded);
            }
            series
        } else {
            let small = slater_series(g)?;
            let large = slater_series(&g.reflected()?)?;
            let (Some(small), Some(large)) = (small, large) else {
                let expanded: Basic = match hyperexpand_safely(&g.as_expr()?) {
                    Ok(x) => x,
                    Err(PolyError::Value(_)) => return Ok(None),
                    Err(e) => return Err(e),
                };
                if has_meijerg(&expanded) || has_hyper(&expanded) {
                    return Ok(None);
                }
                return Ok(Some(plain(m2(&g.prefactor, &expanded)?)));
            };
            let (small, large) = (hyperexpand_safely(&small)?, hyperexpand_safely(&large)?);
            if same(&small, &large, assumptions)? {
                if let Some(choice) = real_choice(&small, &large, assumptions)? {
                    return Ok(Some(plain(m2(&g.prefactor, &choice)?)));
                }
            }
            let below: Basic = crate::core::relational::lt(&modulus, &int(1))?;
            let above: Basic = crate::core::relational::gt(&modulus, &int(1))?;
            let value: Basic = crate::functions::elementary::piecewise::piecewise(&[
                (small, below.clone()),
                (large, crate::core::logic::true_()),
            ])?;
            return Ok(Some(ConditionalValue::new(
                m2(&g.prefactor, &value)?,
                crate::core::logic::or_(&[above, below]),
            )));
        }
    };
    match series {
        None => fallback(g),
        Some(s) => Ok(Some(plain(m2(&g.prefactor, &hyperexpand_safely(&s)?)?))),
    }
}

fn converges_on_the_circle(h: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    let (ap, bq, z) = (h.args()[0].args(), h.args()[1].args(), &h.args()[2]);
    if ap
        .iter()
        .any(|a| is(a, Predicate::integer) == Some(true) && is(a, Predicate::nonpositive) == Some(true))
    {
        return Ok(true);
    }
    if ap.len() < bq.len() + 1 {
        return Ok(true);
    }
    if ap.len() > bq.len() + 1 {
        return Ok(false);
    }
    if *z != int(1) {
        return Ok(true);
    }
    let excess: Basic =
        crate::functions::elementary::complexes::re(&s2(&crate::core::add::add(bq)?, &crate::core::add::add(ap)?)?)?;
    Ok(is(&excess, Predicate::positive) == Some(true)
        || ask(&crate::core::relational::gt(&excess, &int(0))?, assumptions)? == Some(true))
}

fn is_function_node(n: &Basic) -> bool {
    matches!(
        n.head(),
        Head::Function(_) | Head::Undefined(_) | Head::Hyper | Head::MeijerG
    )
}

/// The small-argument expansion when it is real at sample values of the
/// parameters, else `None` (`_real_choice`).
fn real_choice(small: &Basic, large: &Basic, assumptions: &[Basic]) -> PolyResult<Option<Basic>> {
    let mut parameters: Vec<Basic> = small.free_symbols().into_iter().filter(|s| s.is_symbol()).collect();
    for s in large.free_symbols() {
        if s.is_symbol() && !parameters.contains(&s) {
            parameters.push(s);
        }
    }
    parameters.sort_by_key(symbol_name);
    let values: Vec<(Basic, Basic)> = if parameters.is_empty() {
        Vec::new()
    } else {
        let mut rng = crate::random::PyRandom::from_string(&format!("({}, {})", small, large));
        match crate::integrals::conditions::sample_values(&parameters, assumptions, &mut rng)? {
            Some(v) => v,
            None => return Ok(Some(small.clone())),
        }
    };
    let Some(Some(number)) =
        crate::timeout::attempt_within(timeout(), || Ok(crate::numeric::reliable_value(small, 20, &values)))?
    else {
        return Ok(Some(small.clone()));
    };
    let (re, im) = crate::core::evalf::parts(&number);
    let (re, im) = (re.to_f64().value(), im.to_f64().value());
    let size: f64 = (re * re + im * im).sqrt();
    if im.abs() >= 1e-12 * (1.0 + size) {
        return Ok(Some(small.clone()));
    }
    let rule: HashMap<Basic, Basic> = values.iter().cloned().collect();
    let mut seen: Vec<Basic> = Vec::new();
    for atom in small.preorder() {
        if !is_function_node(&atom) || seen.contains(&atom) {
            continue;
        }
        seen.push(atom.clone());
        let substituted: Basic = crate::core::function::xreplace_evaluated(&atom, &rule)?;
        let as_exp: Basic = polar_as_exp(&substituted)?;
        if let Some(value) = crate::core::evalf::numeric_value(&as_exp, 70) {
            let (_, vi) = crate::core::evalf::parts(&value);
            if vi != crate::core::evalf::Real::ZERO {
                return Ok(None);
            }
        }
    }
    Ok(Some(small.clone()))
}

fn symbol_name(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}

fn is_principal_polar(node: &Basic) -> bool {
    if !matches!(node.head(), Head::Function(Function::ExpPolar)) {
        return false;
    }
    let Ok(part) = crate::functions::elementary::complexes::im(&node.args()[0]) else {
        return false;
    };
    let Ok(minus_pi) = crate::core::mul::neg(&Basic::pi()) else {
        return false;
    };
    let lower = crate::core::relational::lt(&minus_pi, &part)
        .map(|b| is_true_boolean(&b))
        .unwrap_or(false);
    let upper = crate::core::relational::le(&part, &Basic::pi())
        .map(|b| is_true_boolean(&b))
        .unwrap_or(false);
    lower && upper
}

fn polar_as_exp(e: &Basic) -> R {
    crate::core::function::replace_where(e, &is_principal_polar, &|n| {
        crate::functions::elementary::exponential::exp(&n.args()[0])
    })
}

fn same(a: &Basic, b: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if has_hyper(a) || has_hyper(b) {
        return Ok(false);
    }
    let difference = crate::timeout::attempt_within(timeout(), || {
        crate::simplify::simplify::simplify(&s2(a, b)?, &crate::simplify::simplify::Options::default())
    })?;
    if difference == Some(int(0)) {
        return Ok(true);
    }
    crate::integrals::conditions::numerically_equal(a, b, assumptions, 3)
}

/// The conditions of the Mellin-Barnes integral of `g` along a vertical
/// line from `lower` to `upper` (`line_conditions`).
pub fn line_conditions(g: &MeijerG, lower: &Basic, upper: &Basic) -> PolyResult<Option<Basic>> {
    let delta: Rat = g.delta();
    let delta_basic: Basic = Basic::from_rational(delta.clone());
    let positive_delta: bool = delta > Rat::ZERO;
    let argument: Basic = crate::functions::elementary::argument::arg(&g.z)?;
    let bound: Basic = m2(&delta_basic, &Basic::pi())?;
    if positive_delta {
        let strict: Basic = crate::core::relational::lt(&abs_(&argument)?, &bound)?;
        if is_true_boolean(&strict) || argument == int(0) || positive_argument(&g.z) {
            return Ok(Some(crate::core::logic::true_()));
        }
    }
    if delta < Rat::ZERO {
        return Ok(None);
    }
    let Some(sigma) = sigma_of(g, lower, upper) else {
        return Ok(None);
    };
    let mut exponent: Basic = int(0);
    for a in &g.bm {
        exponent = a2(&exponent, &s2(&s2(a, &sigma)?, &half())?)?;
    }
    for a in &g.an {
        exponent = a2(&exponent, &s2(&a2(&s2(&int(1), a)?, &sigma)?, &half())?)?;
    }
    for b in &g.bq {
        exponent = s2(&exponent, &s2(&a2(&s2(&int(1), b)?, &sigma)?, &half())?)?;
    }
    for a in &g.ap {
        exponent = s2(&exponent, &s2(&s2(a, &sigma)?, &half())?)?;
    }
    let boundary: Basic = crate::core::logic::and_(&[
        crate::core::relational::le(&abs_(&argument)?, &bound)?,
        crate::core::relational::lt(&exponent, &int(-1))?,
    ]);
    if positive_delta {
        return Ok(Some(crate::core::logic::or_(&[
            crate::core::relational::lt(&abs_(&argument)?, &bound)?,
            boundary,
        ])));
    }
    Ok(Some(boundary))
}

fn positive_argument(z: &Basic) -> bool {
    let factors: Vec<Basic> = if z.is_mul() { z.args().to_vec() } else { vec![z.clone()] };
    factors.iter().all(|f| {
        let base: &Basic = if f.is_pow() { &f.args()[0] } else { f };
        is(base, Predicate::positive) == Some(true) || matches!(base.head(), Head::Function(Function::Abs))
    })
}

fn sigma_of(g: &MeijerG, lower: &Basic, upper: &Basic) -> Option<Basic> {
    let neg_inf = |e: &Basic| matches!(e.head(), Head::NegativeInfinity);
    let inf = |e: &Basic| matches!(e.head(), Head::Infinity);
    if g.p() == g.q() {
        if !neg_inf(lower) {
            return Some(lower.clone());
        }
        if !inf(upper) {
            return Some(upper.clone());
        }
        return Some(int(0));
    }
    if g.p() < g.q() {
        return if inf(upper) { None } else { Some(upper.clone()) };
    }
    if neg_inf(lower) {
        None
    } else {
        Some(lower.clone())
    }
}
