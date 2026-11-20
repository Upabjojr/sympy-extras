//! Port of `sympy_extras/integrals/marichev.py`: definite integrals on
//! `(0, oo)` by Marichev's method. The integrand is decomposed into a power
//! times one or two kernels with known Mellin transforms; one kernel is
//! evaluated at a point of its transform, two by the Mellin convolution
//! theorem as a Meijer G-function expanded by Slater's theorem; the value
//! is tidied (polar numbers, Bessel forms, real logarithms).

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;
use crate::integrals::mellin::{GammaQuotient, Match, Product};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn has_fn(e: &Basic, fs: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(func(n), Some(f) if fs.contains(&f)))
}
fn has_head(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn has_infinity(e: &Basic) -> bool {
    has_head(e, &|n| {
        matches!(n.head(), Head::NaN | Head::ComplexInfinity | Head::Infinity)
    })
}
fn has_i(e: &Basic) -> bool {
    has_head(e, &|n| matches!(n.head(), Head::ImaginaryUnit))
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn budget() -> Option<f64> {
    timeout().map(|t| t / 4.0)
}
fn attempt<T>(seconds: Option<f64>, f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(seconds, f)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn size(e: &Basic) -> i64 {
    crate::core::function::count_ops(e).unwrap_or(0)
}
fn ask(q: &Basic, assumptions: &[Basic]) -> PolyResult<Option<bool>> {
    crate::assumptions::ask::ask(q, assumptions, None)
}
fn atoms_of(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if p(&n) && !found.contains(&n) {
            found.push(n);
        }
    }
    found
}
fn factors(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn terms(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn exp_polar_of(e: &Basic) -> bool {
    func(e) == Some(Function::ExpPolar)
}

/// The value of the quotient at `point`, by a limit when it is singular
/// there (`evaluate_quotient`).
pub fn evaluate_quotient(quotient: &GammaQuotient, point: &Basic) -> PolyResult<Option<Basic>> {
    let value: Basic = quotient.as_expr(point)?;
    if !has_infinity(&value) {
        return Ok(Some(value));
    }
    let s: Basic = crate::core::symbol::Dummy::new("s", &[]).0;
    let expr: Basic = quotient.as_expr(&s)?;
    let found = attempt(timeout(), || crate::core::limits::limit(&expr, &s, point, "+"))?;
    match found {
        Some(f) if !has_infinity(&f) && !has_head(&f, &|n| matches!(n.head(), Head::Limit)) => Ok(Some(f)),
        _ => Ok(None),
    }
}

fn facts_of(assumptions: &[Basic], condition: &Basic) -> Vec<Basic> {
    let mut facts: Vec<Basic> = if *condition == crate::core::logic::true_() {
        Vec::new()
    } else {
        vec![condition.clone()]
    };
    facts.extend(assumptions.iter().cloned());
    facts
}

/// The value made readable: the polar numbers resolved, the Bessel and
/// gamma forms simplified, the logarithms of negative numbers made real
/// (`tidy`).
pub fn tidy(value: &Basic, assumptions: &[Basic], condition: &Basic) -> R {
    use Function as F;
    let mut result: Basic = value.clone();
    if has_fn(&result, &[F::LowerGamma]) && has_fn(&result, &[F::ExpPolar]) {
        result = polar_lowergamma(&result)?;
    }
    if has_fn(&result, &[F::ExpPolar]) && has_fn(&result, &[F::EllipticK, F::EllipticE, F::EllipticF, F::EllipticPi]) {
        result = polar_elliptic(&result)?;
    }
    if has_fn(&result, &[F::PolarLift, F::ExpPolar]) {
        result = projected_sums(&result)?;
    }
    if has_fn(&result, &[F::ExpPolar, F::PolarLift, F::PrincipalBranch]) {
        let current: Basic = result.clone();
        if let Some(u) = attempt(budget(), || {
            crate::functions::elementary::polar::unpolarify(&current, false)
        })? {
            result = u;
        }
    }
    if has_head(&result, &|n| matches!(n.head(), Head::MeijerG)) {
        return Ok(result);
    }
    if has_fn(&result, &[F::BesselI]) {
        result = bessel_k_forms(&result)?;
    }
    if has_fn(&result, &[F::PolyGamma]) {
        let current: Basic = result.clone();
        if let Some(expanded) = attempt(budget(), || crate::functions::expand_func::expand_func(&current, true))? {
            if size(&expanded) < size(&result) {
                result = expanded;
            }
        }
    }
    if has_fn(&result, &[F::LerchPhi, F::DirichletEta]) {
        let current: Basic = result.clone();
        if let Some(expanded) = attempt(budget(), || {
            simplify(&crate::functions::expand_func::expand_func(&current, true)?)
        })? {
            if !has_fn(&expanded, &[F::LerchPhi, F::DirichletEta]) {
                result = expanded;
            }
        }
    }
    let current: Basic = result.clone();
    if let Some(simpler) = attempt(budget(), || simplify(&crate::simplify::gammasimp::gammasimp(&current)?))? {
        if size(&simpler) <= size(&result) {
            result = simpler;
        }
    }
    let current: Basic = result.clone();
    let factored = attempt(budget(), || {
        crate::core::exprtools::factor_terms(
            &crate::polys::polytools::factor(&current, &[], false, true)?,
            false,
            false,
            false,
            false,
        )
    })?;
    if let Some(factored) = factored {
        if size(&factored) < size(&result) {
            result = factored;
        }
    }
    let facts: Vec<Basic> = facts_of(assumptions, condition);
    if has_fn(&result, &[F::Log]) {
        result = real_logarithms(&result, &facts)?;
    }
    if has_i(&result) {
        result = right_half_plane_powers(&result, &facts)?;
    }
    if result.free_symbols().is_empty() && has_complex_unit(&result) {
        let current: Basic = result.clone();
        if let Some(real) = attempt(budget(), || real_constant(&current))? {
            result = real;
        }
    }
    if !facts.is_empty() {
        let current: Basic = result.clone();
        if let Some(refined) = attempt(budget(), || crate::assumptions::refine::refine(&current, &facts, None))? {
            if size(&refined) <= size(&result) {
                result = refined;
            }
        }
    }
    if has_fn(&result, &[F::LowerGamma]) && has_fn(&result, &[F::ExpPolar]) {
        result = polar_lowergamma(&result)?;
    }
    Ok(result)
}

/// `tidy`, kept only when it agrees numerically with the value (`checked_tidy`).
pub fn checked_tidy(value: &Basic, assumptions: &[Basic], condition: &Basic) -> R {
    let tidied: Basic = tidy(value, assumptions, condition)?;
    if tidied == *value || !crate::settings::settings().numerical_checks {
        return Ok(tidied);
    }
    let verdict = attempt(budget(), || {
        crate::integrals::conditions::numerical_verdict(value, &tidied, assumptions, condition, 3)
    })?;
    if verdict == Some(Some(false)) {
        return Ok(value.clone());
    }
    Ok(tidied)
}

fn projected(e: &Basic) -> R {
    match func(e) {
        Some(Function::PolarLift) => return projected(&e.args()[0]),
        Some(Function::ExpPolar) => return crate::functions::elementary::exponential::exp(&projected(&e.args()[0])?),
        _ => {}
    }
    if e.is_mul() {
        let parts: Vec<Basic> = e.args().iter().map(projected).collect::<PolyResult<_>>()?;
        return mul_all(&parts);
    }
    if e.is_pow() && yes(&e.args()[1], Predicate::integer) {
        return pw(&projected(&e.args()[0])?, &e.args()[1]);
    }
    Ok(e.clone())
}

/// The sums with polar terms projected onto the plane (`projected_sums`).
pub fn projected_sums(value: &Basic) -> R {
    crate::core::function::replace_where(
        value,
        &|n| n.is_add() && has_fn(n, &[Function::PolarLift, Function::ExpPolar]),
        &|n| add_all(&terms(n).iter().map(projected).collect::<PolyResult<Vec<_>>>()?),
    )
}

fn split_polar(m: &Basic) -> PolyResult<(Basic, Basic)> {
    let fs: Vec<Basic> = factors(m);
    let free: Vec<Basic> = fs
        .iter()
        .filter(|f| !has_fn(f, &[Function::ExpPolar]))
        .cloned()
        .collect();
    let dependent: Vec<Basic> = fs
        .iter()
        .filter(|f| has_fn(f, &[Function::ExpPolar]))
        .cloned()
        .collect();
    Ok((mul_all(&free)?, mul_all(&dependent)?))
}

fn is_polar_minus_one(e: &Basic) -> PolyResult<bool> {
    Ok(exp_polar_of(e) && e.args()[0] == m2(&Basic::imaginary_unit(), &Basic::pi())?)
}

/// The parameters `m*exp_polar(I*pi)` of the elliptic integrals as `-m`
/// (`polar_elliptic`).
pub fn polar_elliptic(value: &Basic) -> R {
    use Function as F;
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for node in atoms_of(value, &|n| {
        matches!(
            func(n),
            Some(F::EllipticK | F::EllipticE | F::EllipticF | F::EllipticPi)
        )
    }) {
        let m: Basic = node.args()[node.args().len() - 1].clone();
        let (polar, rest) = split_polar(&m)?;
        if is_polar_minus_one(&rest)?
            && yes(&polar, Predicate::extended_real)
            && !replacement.iter().any(|(k, _)| *k == m)
        {
            replacement.push((m, neg(&polar)?));
        }
    }
    replaced(value, &replacement)
}

/// `lowergamma(a, w*exp_polar(I*pi))` as a hypergeometric function
/// (`polar_lowergamma`).
pub fn polar_lowergamma(value: &Basic) -> R {
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for node in atoms_of(value, &|n| func(n) == Some(Function::LowerGamma)) {
        let (a, argument) = (node.args()[0].clone(), node.args()[1].clone());
        let (polar, w) = split_polar(&argument)?;
        let form = |w: &Basic| -> R {
            let hyp: Basic = crate::functions::special::hyper::hyper(std::slice::from_ref(&a), &[a2(&a, &int(1))?], w)?;
            d2(&m2(&m2(&pw(&int(-1), &a)?, &pw(w, &a)?)?, &hyp)?, &a)
        };
        if is_polar_minus_one(&polar)? && yes(&w, Predicate::positive) {
            replacement.push((node.clone(), form(&w)?));
        } else if is_polar_minus_one(&w)? && yes(&polar, Predicate::positive) {
            replacement.push((node.clone(), form(&polar)?));
        }
    }
    replaced(value, &replacement)
}

/// `(w**2)**e` as `w**(2*e)` when `re(w) > 0` (`right_half_plane_powers`).
pub fn right_half_plane_powers(value: &Basic, assumptions: &[Basic]) -> R {
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for node in atoms_of(value, &|n| n.is_pow()) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if !(base.is_pow() && base.args()[1] == int(2)) || yes(exponent, Predicate::integer) {
            continue;
        }
        let w: &Basic = &base.args()[0];
        if yes(w, Predicate::extended_real) {
            continue;
        }
        if ask(
            &crate::assumptions::facts::element(w, &crate::sets::sets::reals())?,
            assumptions,
        )? == Some(true)
        {
            continue;
        }
        let real_part: Basic = crate::functions::elementary::complexes::re(w)?;
        if has_fn(&real_part, &[Function::Re, Function::Im]) {
            continue;
        }
        if ask(&crate::core::relational::gt(&real_part, &int(0))?, assumptions)? == Some(true) {
            replacement.push((node.clone(), pw(w, &m2(&int(2), exponent)?)?));
        }
    }
    replaced(value, &replacement)
}

fn is_order_derivative_at_zero(node: &Basic) -> bool {
    if !matches!(node.head(), Head::Subs) {
        return false;
    }
    let point: &[Basic] = node.args()[2].args();
    if point.len() != 1 || point[0] != int(0) {
        return false;
    }
    let derivative: &Basic = &node.args()[0];
    if !matches!(derivative.head(), Head::Derivative) || func(&derivative.args()[0]) != Some(Function::BesselI) {
        return false;
    }
    let counts = crate::core::function::derivative_counts(derivative);
    let variables: Vec<Basic> = counts
        .iter()
        .flat_map(|(v, k)| std::iter::repeat(v.clone()).take(*k as usize))
        .collect();
    variables.len() == 1
        && variables[0] == derivative.args()[0].args()[0]
        && node.args()[1].args() == [variables[0].clone()]
}

/// The order derivatives of `besseli` at `0` as `-besselk(0, z)` and the
/// pairs `besseli(-nu, z)`, `besseli(nu, z)` in `besselk` (`bessel_k_forms`).
pub fn bessel_k_forms(value: &Basic) -> R {
    use crate::functions::special::bessel::bessel;
    let value: Basic = crate::core::function::replace_where(value, &is_order_derivative_at_zero, &|n| {
        neg(&bessel(
            Function::BesselK,
            &[int(0), n.args()[0].args()[0].args()[1].clone()],
        )?)
    })?;
    let nodes: Vec<Basic> = atoms_of(&value, &|n| func(n) == Some(Function::BesselI));
    let orders: Vec<(Basic, Basic)> = nodes
        .iter()
        .map(|n| (n.args()[0].clone(), n.args()[1].clone()))
        .collect();
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    for node in &nodes {
        let (order, argument) = (&node.args()[0], &node.args()[1]);
        let minus: Basic = neg(order)?;
        if orders.iter().any(|(o, z)| *o == minus && z == argument)
            && crate::core::printing::could_extract_minus_sign(order)
        {
            let k: Basic = bessel(Function::BesselK, &[minus.clone(), argument.clone()])?;
            let sine: Basic = crate::functions::elementary::trigonometric::sin(&m2(&neg(&Basic::pi())?, order)?)?;
            let value_: Basic = a2(
                &bessel(Function::BesselI, &[minus.clone(), argument.clone()])?,
                &d2(&m2(&m2(&int(2), &sine)?, &k)?, &Basic::pi())?,
            )?;
            replacement.push((node.clone(), value_));
        }
    }
    if replacement.is_empty() {
        return Ok(value);
    }
    let rewritten: Basic = crate::polys::polytools::cancel(&replaced(&value, &replacement)?, &[])?;
    let simpler = attempt(timeout(), || {
        simplify(&crate::simplify::gammasimp::gammasimp(&rewritten)?)
    })?;
    Ok(simpler.unwrap_or(rewritten))
}

/// `log(x)` of a negative `x` as `log(-x) + I*pi` (`real_logarithms`).
pub fn real_logarithms(value: &Basic, assumptions: &[Basic]) -> R {
    let mut replacement: Vec<(Basic, Basic)> = Vec::new();
    let flipped = |argument: &Basic| -> R {
        a2(
            &crate::functions::elementary::exponential::log(&neg(argument)?)?,
            &m2(&Basic::imaginary_unit(), &Basic::pi())?,
        )
    };
    for node in atoms_of(value, &|n| func(n) == Some(Function::Log)) {
        let argument: Basic = node.args()[0].clone();
        if is(&argument, Predicate::extended_real) == Some(false) {
            continue;
        }
        if argument.free_symbols().is_empty() {
            // evalf(argument, 50) a negative Float
            let numeric = attempt(timeout(), || Ok(crate::core::evalf::numeric_value(&argument, 170)))?.flatten();
            if let Some(v) = numeric {
                let (re, im) = crate::core::evalf::parts(&v);
                if im == crate::core::evalf::Real::ZERO && re.to_f64().value() < -1e-40 {
                    replacement.push((node.clone(), flipped(&argument)?));
                }
            }
            continue;
        }
        let query = match crate::core::relational::lt(&argument, &int(0)) {
            Ok(q) => q,
            Err(PolyError::Type(_)) => continue,
            Err(e) => return Err(e),
        };
        if ask(&query, assumptions)? == Some(true) {
            replacement.push((node.clone(), flipped(&argument)?));
        }
    }
    if replacement.is_empty() {
        return Ok(value.clone());
    }
    let rewritten: Basic = replaced(value, &replacement)?;
    let simpler = attempt(timeout(), || simplify(&rewritten))?;
    Ok(simpler.unwrap_or(rewritten))
}

fn is_root_of_minus_one(n: &Basic) -> bool {
    n.is_pow() && n.args()[0] == int(-1) && is(&n.args()[1], Predicate::integer) != Some(true)
}

fn has_complex_unit(e: &Basic) -> bool {
    has_i(e) || has_head(e, &is_root_of_minus_one)
}

/// A constant written without the imaginary unit when it is real
/// (`real_constant`).
pub fn real_constant(value: &Basic) -> R {
    if !value.free_symbols().is_empty() || !has_complex_unit(value) {
        return Ok(value.clone());
    }
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: true,
        trig: false,
        func: false,
    };
    let candidate: Basic = simplify(&crate::core::function::expand_with(value, true, hints)?)?;
    if has_complex_unit(&candidate) || size(&candidate) > size(value) {
        return Ok(value.clone());
    }
    Ok(candidate)
}

fn one_kernel(product: &Product, assumptions: &[Basic]) -> PolyResult<Option<ConditionalValue>> {
    let quotient: GammaQuotient = product.matches[0].quotient(assumptions)?;
    let point: Basic = a2(&product.alpha, &int(1))?;
    let Some(value) = evaluate_quotient(&quotient, &point)? else {
        return Ok(None);
    };
    let condition: Basic = crate::core::logic::and_(&[quotient.strip_condition(&point)?, quotient.condition.clone()]);
    Ok(Some(ConditionalValue::new(m2(&product.constant, &value)?, condition)))
}

fn two_kernels(product: &Product, assumptions: &[Basic]) -> PolyResult<Option<ConditionalValue>> {
    let first: GammaQuotient = product.matches[0].quotient(assumptions)?;
    let second: GammaQuotient = product.matches[1].quotient(assumptions)?;
    if !first.is_gamma_quotient() || !second.is_gamma_quotient() {
        return Ok(None);
    }
    let point: Basic = a2(&product.alpha, &int(1))?;
    let integrand: GammaQuotient = first.times(&second.compose(&point, &int(-1))?)?.cancelled()?;
    let mut nonempty: Basic = crate::core::logic::true_();
    if !matches!(integrand.lower.head(), Head::NegativeInfinity) && !matches!(integrand.upper.head(), Head::Infinity) {
        nonempty = crate::core::relational::lt(&integrand.lower, &integrand.upper)?;
        if nonempty == crate::core::logic::false_() {
            return Ok(None);
        }
    }
    let Some(g) = crate::integrals::slater::mellin_barnes(&integrand)? else {
        return Ok(None);
    };
    let Some(line) = crate::integrals::slater::line_conditions(&g, &integrand.lower, &integrand.upper)? else {
        return Ok(None);
    };
    let Some(expanded) = crate::integrals::slater::expand_meijerg(&g, assumptions)? else {
        return Ok(None);
    };
    let condition: Basic =
        crate::core::logic::and_(&[nonempty, integrand.condition.clone(), line, expanded.condition.clone()]);
    Ok(Some(ConditionalValue::new(
        m2(&product.constant, &expanded.value)?,
        condition,
    )))
}

/// The integral on `(0, oo)` of a decomposed integrand (`integrate_product`).
pub fn integrate_product(
    product: &Product,
    assumptions: &[Basic],
    regularize: bool,
) -> PolyResult<Option<ConditionalValue>> {
    if product.matches.is_empty() {
        return Ok(None);
    }
    if product.one_minus_power > 0 {
        let beta: Basic = crate::core::symbol::Dummy::new("beta", &[(Predicate::positive, true)]).0;
        let mut matches: Vec<Match> = Vec::new();
        let mut parameter: Option<Basic> = None;
        for m in &product.matches {
            if m.kernel.name == crate::integrals::mellin::BETA_LOWER && parameter.is_none() {
                parameter = Some(m.kernel.parameters[0].clone());
                matches.push(Match {
                    kernel: crate::integrals::mellin::beta_kernel(&beta)?,
                    beta: m.beta.clone(),
                    gamma: m.gamma.clone(),
                    constant: m.constant.clone(),
                    cutoff: m.cutoff,
                });
            } else {
                matches.push(m.clone());
            }
        }
        let Some(parameter) = parameter else { return Ok(None) };
        let shifted = Product {
            constant: product.constant.clone(),
            alpha: product.alpha.clone(),
            log_power: product.log_power,
            matches,
            one_minus_power: 0,
        };
        let Some(found) = integrate_product(&shifted, assumptions, regularize)? else {
            return Ok(None);
        };
        let mut value: Basic = found.value.clone();
        for _ in 0..product.one_minus_power {
            value = crate::core::function::derivative(&value, &beta)?;
        }
        return Ok(Some(ConditionalValue::new(
            crate::core::subs::subs(&value, &beta, &parameter)?,
            crate::core::subs::subs(&found.condition, &beta, &parameter)?,
        )));
    }
    if product.log_power > 0 {
        let alpha: Basic = crate::core::symbol::Dummy::new("alpha", &[(Predicate::real, true)]).0;
        let shifted = Product {
            constant: product.constant.clone(),
            alpha: alpha.clone(),
            log_power: 0,
            matches: product.matches.clone(),
            one_minus_power: product.one_minus_power,
        };
        let Some(found) = integrate_product(&shifted, assumptions, regularize)? else {
            return Ok(None);
        };
        let mut value: Basic = found.value.clone();
        for _ in 0..product.log_power {
            value = crate::core::function::derivative(&value, &alpha)?;
        }
        return Ok(Some(ConditionalValue::new(
            crate::core::subs::subs(&value, &alpha, &product.alpha)?,
            crate::core::subs::subs(&found.condition, &alpha, &product.alpha)?,
        )));
    }
    let found = if product.matches.len() == 1 {
        one_kernel(product, assumptions)?
    } else {
        two_kernels(product, assumptions)?
    };
    let Some(found) = found else { return Ok(None) };
    if !regularize {
        return Ok(Some(found));
    }
    let kept: Vec<Basic> = product
        .matches
        .iter()
        .map(|m| Ok(m.quotient(assumptions)?.condition))
        .collect::<PolyResult<_>>()?;
    Ok(Some(ConditionalValue::new(
        found.value,
        crate::core::logic::and_(&kept),
    )))
}

fn combine_linear_exponentials(term: &Basic, x: &Basic) -> R {
    let mut linear: Vec<Basic> = Vec::new();
    let mut others: Vec<Basic> = Vec::new();
    for factor in factors(term) {
        let is_linear: bool = func(&factor) == Some(Function::Exp)
            && factor.has(x)
            && !crate::core::function::derivative(&factor.args()[0], x)?.has(x);
        if is_linear {
            linear.push(factor.args()[0].clone());
        } else {
            others.push(factor);
        }
    }
    if linear.len() < 2 {
        return Ok(term.clone());
    }
    let argument: Basic = crate::core::exprtools::factor_terms(&add_all(&linear)?, false, false, false, false)?;
    m2(
        &mul_all(&others)?,
        &crate::functions::elementary::exponential::exp(&argument)?,
    )
}

/// The trigonometric and hyperbolic factors as exponentials, the linear
/// exponentials of each summand combined (`exponential_form`).
pub fn exponential_form(term: &Basic, x: &Basic) -> R {
    use Function as F;
    let mut sums: Vec<Basic> = Vec::new();
    let mut others: Vec<Basic> = Vec::new();
    for piece in factors(term) {
        if piece.has(x) && matches!(func(&piece), Some(F::Sin | F::Cos | F::Sinh | F::Cosh)) {
            sums.push(crate::core::rewriting::rewrite(
                &piece,
                &[],
                "exp",
                &crate::core::rewriting::RewriteHints::default(),
            )?);
        } else {
            others.push(piece);
        }
    }
    if sums.is_empty() {
        return Ok(term.clone());
    }
    let rest: Basic = mul_all(&others)?;
    let summands: Vec<Basic> = terms(&crate::core::function::expand_default(&mul_all(&sums)?)?);
    let parts: Vec<Basic> = summands
        .iter()
        .map(|u| combine_linear_exponentials(&m2(u, &rest)?, x))
        .collect::<PolyResult<_>>()?;
    add_all(&parts)
}

/// The powers `(c + r(x))**e`, `e > 0` not an integer, split into
/// `(c + r)**(e - n)` times the expanded `(c + r)**n`
/// (`reduce_positive_powers`).
pub fn reduce_positive_powers(f: &Basic, x: &Basic) -> R {
    let rewrite = |node: &Basic| -> R {
        if !node.is_pow() || !node.args()[0].is_add() {
            return Ok(node.clone());
        }
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if !yes(exponent, Predicate::positive) || is(exponent, Predicate::integer) != Some(false) || exponent.has(x) {
            return Ok(node.clone());
        }
        let (constant, rest) = crate::core::expr::as_independent(base, x, true)?;
        if constant == int(0) || crate::integrals::mellin::monomial(&rest, x)?.is_none() {
            return Ok(node.clone());
        }
        let n: Basic = crate::functions::elementary::integers::rounding(Function::Ceiling, exponent)?;
        if !matches!(
            n.head(),
            Head::Rational {
                integer_class: true,
                ..
            }
        ) {
            return Ok(node.clone());
        }
        let expanded: Basic = crate::core::function::expand_default(&pw(base, &n)?)?;
        let tail: Basic = pw(base, &crate::core::add::sub(exponent, &n)?)?;
        add_all(
            &terms(&expanded)
                .iter()
                .map(|t| m2(t, &tail))
                .collect::<PolyResult<Vec<_>>>()?,
        )
    };
    let mut result: Basic = f.clone();
    for node in atoms_of(f, &|n| n.is_pow()) {
        let replacement: Basic = rewrite(&node)?;
        if replacement != node {
            result = replaced(&result, &[(node, replacement)])?;
        }
    }
    if result == *f {
        return Ok(f.clone());
    }
    let hints = crate::core::function::Hints {
        mul: true,
        multinomial: false,
        power_base: true,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(&result, true, hints)
}

/// The integral of `f` on `(0, oo)` by Marichev's method, or `None`
/// (`mellin_integrate`).
pub fn mellin_integrate(
    f: &Basic,
    x: &Basic,
    assumptions: &[Basic],
    cutoff: Option<&str>,
    regularize: bool,
) -> PolyResult<Option<ConditionalValue>> {
    use crate::integrals::mellin::decompose_integrand;
    let mut total: Option<ConditionalValue> = None;
    for term in terms(&reduce_positive_powers(f, x)?) {
        let pieces: Vec<Basic> = if decompose_integrand(&term, x, cutoff)?.is_none() {
            terms(&exponential_form(&term, x)?)
        } else {
            vec![term.clone()]
        };
        for piece in pieces {
            let Some(product) = decompose_integrand(&piece, x, cutoff)? else {
                return Ok(None);
            };
            let Some(found) = integrate_product(&product, assumptions, regularize)? else {
                return Ok(None);
            };
            total = Some(match total {
                None => found,
                Some(t) => t.add(&found)?,
            });
        }
    }
    let Some(total) = total else { return Ok(None) };
    let Some(condition) = crate::integrals::conditions::decide(&total.condition, assumptions)? else {
        return Ok(None);
    };
    let value: Basic = tidy(&total.value, assumptions, &condition)?;
    if has_fn(
        &value,
        &[Function::ExpPolar, Function::PolarLift, Function::PrincipalBranch],
    ) || has_head(&value, &|n| matches!(n.head(), Head::Limit))
    {
        return Ok(None);
    }
    Ok(Some(ConditionalValue::new(value, condition)))
}
