//! Port of `sympy_extras/integrals/rewriting.py`: the equivalent forms of
//! an integrand which the integrators find antiderivatives of more easily
//! (`rewritten_forms`: powers with the variable in the exponent as
//! exponentials, hyperbolic functions as exponentials, inverse hyperbolic
//! ones as logarithms, radicals recombined, absolute values resolved) and
//! the substitutions which rationalise it (`power_substitutions`: powers
//! of linear functions, exponentials, logarithms, even integrands,
//! binomial differentials, roots of a Möbius function, nested radicals).

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;

const LIMIT: usize = 5;
const SUBSTITUTIONS: usize = 3;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn fresh(name: &str, facts: &[(Predicate, bool)]) -> Basic {
    let k: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
    crate::core::symbol::Dummy::new(&format!("{}{}", name, k), facts).0
}

fn int(n: i64) -> Basic {
    Basic::int(n)
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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn rational_of(e: &Basic) -> Option<Rat> {
    match e.head() {
        Head::Rational { .. } => e.as_rational().cloned(),
        _ => None,
    }
}
fn rat(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}
fn den(r: &Rat) -> i64 {
    i64::try_from(&crate::numbers::denom(r)).unwrap_or(i64::MAX)
}
fn num(r: &Rat) -> i64 {
    i64::try_from(&crate::numbers::numer(r)).unwrap_or(i64::MAX)
}
fn is_integer_node(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}
fn has_head(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn pows(e: &Basic) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if n.is_pow() && !out.contains(&n) {
            out.push(n);
        }
    }
    out
}
fn rational_function(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Rational) == Some(true)
}
fn polynomial(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
}
fn timeout() -> Option<f64> {
    crate::settings::settings().timeout
}
fn attempt(f: impl FnOnce() -> R) -> PolyResult<Option<Basic>> {
    crate::timeout::attempt_within(timeout(), f)
}
fn powsimp_force(e: &Basic) -> R {
    crate::simplify::powsimp::powsimp(
        e,
        false,
        crate::simplify::powsimp::Combine::All,
        true,
        &crate::core::function::count_ops,
    )
}
fn operations(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}
fn name_of(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        _ => s.to_string(),
    }
}
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}
fn lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd(a, b) * b).abs()
}

fn coefficients(e: &Basic, x: &Basic) -> PolyResult<Option<std::collections::BTreeMap<i64, Basic>>> {
    if !polynomial(e, std::slice::from_ref(x)) {
        return Ok(None);
    }
    let mut found: std::collections::BTreeMap<i64, Basic> = std::collections::BTreeMap::new();
    for term in crate::core::expr::terms_of(&crate::core::function::expand_default(e)?) {
        let (coefficient, rest) = crate::core::expr::as_independent(&term, x, false)?;
        let n: i64 = if rest == Basic::one() {
            0
        } else if rest == *x {
            1
        } else if rest.is_pow() && rest.args()[0] == *x && is_integer_node(&rest.args()[1]) {
            match rational_of(&rest.args()[1]) {
                Some(r) if r > Rat::ZERO => num(&r),
                _ => return Ok(None),
            }
        } else {
            return Ok(None);
        };
        let value: Basic = match found.get(&n) {
            Some(c) => a2(c, &coefficient)?,
            None => coefficient,
        };
        found.insert(n, value);
    }
    Ok(Some(found.into_iter().filter(|(_, c)| *c != Basic::zero()).collect()))
}

fn linear(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some(found) = coefficients(e, x)? else {
        return Ok(None);
    };
    if found.keys().next_back() != Some(&1) {
        return Ok(None);
    }
    Ok(Some((
        found[&1].clone(),
        found.get(&0).cloned().unwrap_or_else(Basic::zero),
    )))
}

fn degree(e: &Basic, x: &Basic) -> PolyResult<Option<i64>> {
    Ok(coefficients(e, x)?.map(|f| f.keys().next_back().copied().unwrap_or(-1)))
}

/// An integrand in a new variable, the expression of the variable in the
/// old one (`back`), the old variable in the new (`forward`) and the facts
/// it holds under (`Substitution`).
#[derive(Clone, Debug)]
pub struct Substitution {
    pub integrand: Basic,
    pub variable: Basic,
    pub back: Basic,
    pub forward: Basic,
    pub facts: Vec<Basic>,
}

fn ask_true(query: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    Ok(crate::assumptions::ask::ask(query, assumptions, None)? == Some(true))
}

fn ask_not_false(query: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    Ok(crate::assumptions::ask::ask(query, assumptions, None)? != Some(false))
}

fn gt0(e: &Basic) -> R {
    crate::core::relational::gt(e, &Basic::zero())
}

fn positive(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if yes(e, Predicate::positive) {
        return Ok(true);
    }
    if e.free_symbols().is_empty() {
        return Ok(false);
    }
    ask_true(&gt0(e)?, assumptions)
}

fn may_be_real(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    let complexish: bool = has_head(e, &|n| {
        matches!(
            n.head(),
            Head::ImaginaryUnit | Head::Function(Function::ExpPolar | Function::PolarLift)
        )
    });
    if complexish || is(e, Predicate::extended_real) == Some(false) {
        return Ok(false);
    }
    ask_not_false(
        &crate::assumptions::facts::element(e, &crate::sets::sets::reals())?,
        assumptions,
    )
}

fn may_be_positive(e: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if positive(e, assumptions)? {
        return Ok(true);
    }
    if e.free_symbols().is_empty() || !may_be_real(e, assumptions)? {
        return Ok(false);
    }
    Ok(is(e, Predicate::positive) != Some(false) && ask_not_false(&gt0(e)?, assumptions)?)
}

/// The facts `b > 0` which make the powers `b**f(x)` exponentials, for the
/// bases which may be positive (`implied_assumptions`).
pub fn implied_assumptions(f: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut found: Vec<Basic> = Vec::new();
    for node in pows(f) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if !exponent.has(x) || base.has(x) || matches!(base.head(), Head::Exp1) || base.free_symbols().is_empty() {
            continue;
        }
        if !positive(base, assumptions)? && may_be_positive(base, assumptions)? {
            let fact: Basic = gt0(base)?;
            if !found.contains(&fact) {
                found.push(fact);
            }
        }
    }
    Ok(found)
}

fn bases_to_exponentials(f: &Basic, x: &Basic, assumptions: &[Basic]) -> R {
    let rewrite = |node: &Basic| -> R {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        if exponent.has(x) {
            if base.has(x) || matches!(base.head(), Head::Exp1) {
                return Ok(node.clone());
            }
            if may_be_positive(base, assumptions)? {
                return exp(&m2(exponent, &log(base)?)?);
            }
            return Ok(node.clone());
        }
        if !may_be_real(exponent, assumptions)? {
            return Ok(node.clone());
        }
        if base.is_pow()
            && base.args()[0] == *x
            && !base.args()[1].has(x)
            && may_be_real(&base.args()[1], assumptions)?
            && ask_true(&gt0(x)?, assumptions)?
        {
            return pw(x, &m2(&base.args()[1], exponent)?);
        }
        let mut arguments: Vec<Basic> = Vec::new();
        let mut rest: Basic = Basic::one();
        for part in crate::core::expr::factors_of(base) {
            if matches!(part.head(), Head::Function(Function::Exp)) && may_be_real(&part.args()[0], assumptions)? {
                arguments.push(part.args()[0].clone());
            } else if !part.has(x) && may_be_positive(&part, assumptions)? {
                rest = m2(&rest, &part)?;
            } else {
                return Ok(node.clone());
            }
        }
        if arguments.is_empty() {
            return Ok(node.clone());
        }
        m2(
            &exp(&m2(exponent, &crate::core::add::add(&arguments)?)?)?,
            &pw(&rest, exponent)?,
        )
    };
    crate::core::function::replace_where(f, &|n| n.is_pow(), &rewrite)
}

fn hyperbolic_to_exponentials(f: &Basic) -> R {
    let hyperbolic = |n: &Basic| {
        matches!(
            n.head(),
            Head::Function(
                Function::Sinh | Function::Cosh | Function::Tanh | Function::Coth | Function::Sech | Function::Csch
            )
        )
    };
    if !has_head(f, &hyperbolic) {
        return Ok(f.clone());
    }
    crate::core::rewriting::rewrite(f, &[], "exp", &crate::core::rewriting::RewriteHints::default())
}

fn inverse_hyperbolic_to_logarithms(f: &Basic) -> R {
    let inverse = |n: &Basic| {
        matches!(
            n.head(),
            Head::Function(
                Function::Asinh
                    | Function::Acosh
                    | Function::Atanh
                    | Function::Acoth
                    | Function::Asech
                    | Function::Acsch
            )
        )
    };
    if !has_head(f, &inverse) {
        return Ok(f.clone());
    }
    crate::core::rewriting::rewrite(f, &[], "log", &crate::core::rewriting::RewriteHints::default())
}

fn combined_exponentials(f: &Basic) -> R {
    let found = attempt(|| {
        crate::simplify::powsimp::powsimp(
            f,
            true,
            crate::simplify::powsimp::Combine::Exp,
            false,
            &crate::core::function::count_ops,
        )
    })?;
    Ok(found.unwrap_or_else(|| f.clone()))
}

fn products_of_exponentials(f: &Basic) -> bool {
    f.preorder().iter().any(|n| {
        n.is_mul()
            && n.args()
                .iter()
                .filter(|a| matches!(a.head(), Head::Function(Function::Exp)))
                .count()
                > 1
    })
}

fn with(assumptions: &[Basic], more: Vec<Basic>) -> Vec<Basic> {
    let mut items: Vec<Basic> = more;
    items.extend(assumptions.iter().cloned());
    items
}

fn nonnegative(e: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<bool> {
    if yes(e, Predicate::nonnegative) {
        return Ok(true);
    }
    let real: Basic = fresh(&name_of(x), &[(Predicate::real, true)]);
    let rule: HashMap<Basic, Basic> = [(x.clone(), real)].into_iter().collect();
    if yes(
        &crate::core::function::xreplace_evaluated(e, &rule)?,
        Predicate::nonnegative,
    ) {
        return Ok(true);
    }
    let query: Basic = crate::core::relational::ge(e, &Basic::zero())?;
    let facts: Vec<Basic> = with(
        assumptions,
        vec![crate::assumptions::facts::element(x, &crate::sets::sets::reals())?],
    );
    ask_true(&query, &facts)
}

fn recombined_radicals(f: &Basic, x: &Basic, assumptions: &[Basic]) -> R {
    let rewrite = |node: &Basic| -> R {
        let mut radicals: Vec<(Basic, Rat)> = Vec::new();
        let mut rest: Vec<Basic> = Vec::new();
        for part in node.args() {
            let r: Option<Rat> = if part.is_pow() {
                rational_of(&part.args()[1])
            } else {
                None
            };
            match r {
                Some(r) if den(&r) > 1 && part.has(x) && nonnegative(&part.args()[0], x, assumptions)? => {
                    radicals.push((part.args()[0].clone(), r));
                }
                _ => rest.push(part.clone()),
            }
        }
        if radicals.len() < 2 {
            return Ok(node.clone());
        }
        let q: i64 = den(&radicals[0].1);
        if radicals.iter().any(|(_, r)| den(r) != q) {
            return Ok(node.clone());
        }
        let mut radicand: Basic = Basic::one();
        for (base, r) in &radicals {
            radicand = m2(&radicand, &pw(base, &int(num(r)))?)?;
        }
        m2(&crate::core::mul::mul(&rest)?, &pw(&radicand, &d2(&int(1), &int(q))?)?)
    };
    crate::core::function::replace_where(f, &|n| n.is_mul(), &rewrite)
}

fn absolute_values(f: &Basic, assumptions: &[Basic]) -> R {
    let rewrite = |node: &Basic| -> R {
        let inner: &Basic = &node.args()[0];
        if ask_true(&crate::core::relational::ge(inner, &Basic::zero())?, assumptions)? {
            return Ok(inner.clone());
        }
        if ask_true(&crate::core::relational::le(inner, &Basic::zero())?, assumptions)? {
            return neg(inner);
        }
        Ok(node.clone())
    };
    crate::core::function::replace_where(f, &|n| matches!(n.head(), Head::Function(Function::Abs)), &rewrite)
}

/// Up to five equivalent forms of the integrand (`rewritten_forms`).
pub fn rewritten_forms(f: &Basic, x: &Basic, assumptions: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut forms: Vec<Basic> = Vec::new();
    let add_form = |candidate: Option<Basic>, forms: &mut Vec<Basic>| {
        if let Some(c) = candidate {
            if c != *f && !forms.contains(&c) && forms.len() < LIMIT {
                forms.push(c);
            }
        }
    };
    let signed: Basic = absolute_values(f, assumptions)?;
    let mut exponential: Basic = bases_to_exponentials(&signed, x, assumptions)?;
    exponential = hyperbolic_to_exponentials(&exponential)?;
    add_form(Some(exponential.clone()), &mut forms);
    let mut combined: Basic = combined_exponentials(&exponential)?;
    add_form(Some(combined.clone()), &mut forms);
    let mut hints = crate::core::function::Hints::DEFAULT;
    hints.power_base = false;
    hints.power_exp = false;
    let distributed: Basic = combined_exponentials(&crate::core::function::expand_with(&exponential, true, hints)?)?;
    if distributed != combined && !products_of_exponentials(&distributed) {
        add_form(Some(distributed.clone()), &mut forms);
        combined = distributed;
    }
    let logarithmic: Basic = inverse_hyperbolic_to_logarithms(&combined)?;
    add_form(Some(logarithmic.clone()), &mut forms);
    let radical: Basic = recombined_radicals(&logarithmic, x, assumptions)?;
    add_form(Some(radical.clone()), &mut forms);
    let expanded = attempt(|| cancel(&crate::core::function::expand_default(&radical)?))?;
    if let Some(e) = expanded {
        if operations(&e)? <= operations(&radical)? {
            add_form(Some(e), &mut forms);
        }
    }
    Ok(forms)
}

fn fractional_powers(f: &Basic, x: &Basic) -> PolyResult<Vec<(Basic, i64)>> {
    let mut found: Vec<(Basic, i64)> = Vec::new();
    for node in pows(f) {
        let (base, exponent) = (&node.args()[0], &node.args()[1]);
        let Some(r) = rational_of(exponent) else { continue };
        if den(&r) == 1 || !base.has(x) {
            continue;
        }
        if base != x && degree(base, x)? != Some(1) {
            continue;
        }
        match found.iter_mut().find(|(b, _)| b == base) {
            Some(entry) => entry.1 = lcm(entry.1, den(&r)),
            None => found.push((base.clone(), lcm(1, den(&r)))),
        }
    }
    Ok(found)
}

fn fractional_in(node: &Basic, u: &Basic) -> bool {
    node.is_pow()
        && node.has(u)
        && matches!(
            node.args()[1].head(),
            Head::Rational {
                integer_class: false,
                ..
            }
        )
}

fn power_substitution(f: &Basic, x: &Basic, base: &Basic, k: i64) -> PolyResult<Option<Substitution>> {
    let t: Basic = fresh("t", &[(Predicate::positive, true)]);
    let Some((a, b)) = linear(base, x)? else {
        return Ok(None);
    };
    let forward: Basic = d2(&s2(&pw(&t, &int(k))?, &b)?, &a)?;
    let jacobian: Basic = d2(&m2(&int(k), &pw(&t, &int(k - 1))?)?, &a)?;
    let Some(g) = attempt(|| cancel(&powsimp_force(&m2(&subs(f, x, &forward)?, &jacobian)?)?))? else {
        return Ok(None);
    };
    if pows(&g)
        .iter()
        .any(|n| rational_of(&n.args()[1]).is_some_and(|r| den(&r) > 1) && n.has(&t))
    {
        return Ok(None);
    }
    Ok(Some(Substitution {
        integrand: g,
        variable: t,
        back: pw(base, &d2(&int(1), &int(k))?)?,
        forward,
        facts: Vec::new(),
    }))
}

fn generator(coefficients: &[Rat]) -> Basic {
    let mut numerator: i64 = 0;
    let mut denominator: i64 = 1;
    for c in coefficients {
        numerator = gcd(numerator, num(c).abs());
        denominator = lcm(denominator, den(c));
    }
    rat(&(Rat::from(Int::from(numerator)) / Rat::from(Int::from(denominator))))
}

fn exponential_generator(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let mut coefficients: Vec<Rat> = Vec::new();
    let mut symbolic: Vec<Basic> = Vec::new();
    let mut seen: Vec<Basic> = Vec::new();
    for node in f.preorder() {
        if !matches!(node.head(), Head::Function(Function::Exp)) || seen.contains(&node) {
            continue;
        }
        seen.push(node.clone());
        let (_, argument) = crate::core::expr::as_independent(&node.args()[0], x, true)?;
        let (coefficient, rest) = crate::core::expr::as_independent(&argument, x, false)?;
        if rest != *x {
            return Ok(None);
        }
        match rational_of(&coefficient) {
            Some(r) => coefficients.push(r),
            None => symbolic.push(coefficient),
        }
    }
    if !symbolic.is_empty() && !coefficients.is_empty() {
        return Ok(None);
    }
    if !symbolic.is_empty() {
        let first: Basic = symbolic[0].clone();
        for other in &symbolic[1..] {
            let Some(ratio) = rational_of(&cancel(&d2(other, &first)?)?) else {
                return Ok(None);
            };
            coefficients.push(ratio);
        }
        coefficients.push(Rat::ONE);
        return Ok(Some(m2(&first, &generator(&coefficients))?));
    }
    if coefficients.is_empty() {
        return Ok(None);
    }
    Ok(Some(generator(&coefficients)))
}

fn algebraic_in(g: &Basic, u: &Basic) -> PolyResult<bool> {
    let us = std::slice::from_ref(u);
    if rational_function(g, us) {
        return Ok(true);
    }
    let powers: Vec<Basic> = pows(g);
    for node in &powers {
        if fractional_in(node, u) && !polynomial(&node.args()[0], us) {
            return Ok(false);
        }
    }
    let mut replaced: Basic = g.clone();
    for node in &powers {
        if fractional_in(node, u) {
            let rule: HashMap<Basic, Basic> = [(node.clone(), fresh("p", &[]))].into_iter().collect();
            replaced = crate::core::function::xreplace_evaluated(&replaced, &rule)?;
        }
    }
    Ok(rational_function(&replaced, us))
}

fn exponential_substitution(f: &Basic, x: &Basic) -> PolyResult<Option<Substitution>> {
    let Some(c) = exponential_generator(f, x)? else {
        return Ok(None);
    };
    let u: Basic = fresh("u", &[(Predicate::positive, true)]);
    let forward: Basic = d2(&log(&u)?, &c)?;
    let found = attempt(|| cancel(&powsimp_force(&d2(&subs(f, x, &forward)?, &m2(&c, &u)?)?)?))?;
    let Some(found) = found else { return Ok(None) };
    if !algebraic_in(&found, &u)? {
        return Ok(None);
    }
    let mut g: Basic = found.clone();
    if !rational_function(&g, std::slice::from_ref(&u)) {
        if let Some(factored) = attempt(|| crate::polys::polytools::factor(&found, &[], false, true))? {
            g = factored;
        }
    }
    Ok(Some(Substitution {
        integrand: g,
        variable: u,
        back: exp(&m2(&c, x)?)?,
        forward,
        facts: Vec::new(),
    }))
}

fn moebius_substitutions(f: &Basic, x: &Basic) -> PolyResult<Vec<Substitution>> {
    let mut found_all: Vec<Substitution> = Vec::new();
    for (g, facts) in extracted_powers(f, x)? {
        let Some(found) = crate::integrals::algebraic::moebius_root(&g, x)? else {
            continue;
        };
        let t: Basic = found.t.clone();
        let rule: HashMap<Basic, Basic> = found.radicals.iter().cloned().collect();
        let h = attempt(|| {
            let replaced: Basic = crate::core::function::xreplace_evaluated(&g, &rule)?;
            let jacobian: Basic = crate::core::function::derivative(&found.x_of_t, &t)?;
            cancel(&m2(&subs(&replaced, x, &found.x_of_t)?, &jacobian)?)
        })?;
        let Some(h) = h else { continue };
        if !rational_function(&h, std::slice::from_ref(&t)) || h.has(x) {
            continue;
        }
        found_all.push(Substitution {
            integrand: h,
            variable: t,
            back: found.t_of_x,
            forward: found.x_of_t,
            facts,
        });
    }
    Ok(found_all)
}

fn extracted_powers(f: &Basic, x: &Basic) -> PolyResult<Vec<(Basic, Vec<Basic>)>> {
    let atoms: Vec<Basic> = pows(f)
        .into_iter()
        .filter(|n| {
            rational_of(&n.args()[1]).is_some_and(|r| den(&r) > 1)
                && n.has(x)
                && polynomial(&n.args()[0], std::slice::from_ref(x))
        })
        .collect();
    if atoms.len() != 1 {
        return Ok(Vec::new());
    }
    let node: &Basic = &atoms[0];
    let (base, exponent) = (&node.args()[0], &node.args()[1]);
    let e: Rat = rational_of(exponent).expect("a rational");
    let n: i64 = den(&e);
    let (content, polys) =
        match crate::polys::polytools::Poly::new(base, std::slice::from_ref(x)).and_then(|p| p.factor_list()) {
            Ok(v) => v,
            Err(PolyError::Polys(..)) => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
    let mut factors: Vec<(Basic, i64)> = Vec::new();
    for (p, k) in polys {
        factors.push((p.as_expr()?, k as i64));
    }
    if factors.len() < 2 {
        return Ok(Vec::new());
    }
    let mut remainders: Option<Vec<i64>> = None;
    for signs in 0..(1u64 << factors.len()) {
        let mut found: Vec<i64> = Vec::new();
        let mut inside: Basic = content.clone();
        for (i, (piece, k)) in factors.iter().enumerate() {
            let mut r: i64 = k.rem_euclid(n);
            if r != 0 && (signs >> i) & 1 == 1 {
                r -= n;
            }
            found.push(r);
            inside = m2(&inside, &pw(piece, &int(r))?)?;
        }
        let (numerator, denominator) = crate::simplify::radsimp::fraction(&cancel(&inside)?)?;
        let degrees = [degree(&numerator, x)?, degree(&denominator, x)?];
        if factors.iter().zip(&found).any(|((_, k), r)| k != r) && degrees.iter().all(|d| d.is_some_and(|d| d <= 1)) {
            remainders = Some(found);
            break;
        }
    }
    let Some(remainders) = remainders else {
        return Ok(Vec::new());
    };
    let extracted: Vec<usize> = factors
        .iter()
        .zip(&remainders)
        .enumerate()
        .filter(|(_, ((_, k), r))| k != *r)
        .map(|(i, _)| i)
        .collect();
    let mut results: Vec<(Basic, Vec<Basic>)> = Vec::new();
    for signs in 0..(1u64 << extracted.len()) {
        let mut outside: Basic = Basic::one();
        let mut inside: Basic = content.clone();
        let mut facts: Vec<Basic> = Vec::new();
        for (i, ((piece, k), r)) in factors.iter().zip(&remainders).enumerate() {
            let mut factor: Basic = piece.clone();
            if let Some(pos) = extracted.iter().position(|j| *j == i) {
                if (signs >> pos) & 1 == 1 {
                    factor = neg(&factor)?;
                    inside = m2(&inside, &pw(&int(-1), &int(*k))?)?;
                }
            }
            if k != r {
                outside = m2(&outside, &pw(&factor, &int((k - r).div_euclid(n)))?)?;
                facts.push(gt0(&factor)?);
            }
            inside = m2(&inside, &pw(&factor, &int(*r))?)?;
        }
        let whole: Basic = m2(&pw(&outside, &m2(exponent, &int(n))?)?, &pw(&inside, exponent)?)?;
        let rule: HashMap<Basic, Basic> = [(node.clone(), whole)].into_iter().collect();
        results.push((crate::core::function::xreplace_evaluated(f, &rule)?, facts));
    }
    Ok(results)
}

/// A product built two factors at a time from the left, as Python's `*`.
fn chain(v: &[Basic]) -> R {
    let mut acc: Basic = v[0].clone();
    for f in &v[1..] {
        acc = m2(&acc, f)?;
    }
    Ok(acc)
}

fn binomial_substitution(f: &Basic, x: &Basic) -> PolyResult<Option<Substitution>> {
    let Some([c, m, a, b, n, p]) = crate::integrals::algebraic::binomial_differential(f, x)? else {
        return Ok(None);
    };
    let (Some(m_), Some(n_), Some(p_)) = (rational_of(&m), rational_of(&n), rational_of(&p)) else {
        return Ok(None);
    };
    if n_ == Rat::ZERO || is_integer_node(&p) {
        return Ok(None);
    }
    let q: Rat = (&m_ + Rat::ONE) / &n_;
    let t: Basic = fresh("t", &[(Predicate::positive, true)]);
    let s: i64 = den(&p_);
    let sb: Basic = int(s);
    let ts: Basic = pw(&t, &sb)?;
    let (g, back, forward) = if den(&q) == 1 {
        let g: Basic = chain(&[
            d2(&m2(&c, &sb)?, &m2(&n, &b)?)?,
            pw(&t, &int(s - 1))?,
            pw(&t, &rat(&(Rat::from(Int::from(s)) * &p_)))?,
            pw(&d2(&s2(&ts, &a)?, &b)?, &int(num(&q) - 1))?,
        ])?;
        let back: Basic = pw(&a2(&a, &m2(&b, &pw(x, &n)?)?)?, &d2(&int(1), &sb)?)?;
        let forward: Basic = pw(&d2(&s2(&ts, &a)?, &b)?, &d2(&int(1), &n)?)?;
        (g, back, forward)
    } else if den(&(&q + &p_)) == 1 {
        let exponent: i64 = -num(&(&q + &p_)) - 1;
        let power: Rat = Rat::from(Int::from(s - 1)) + Rat::from(Int::from(s)) * &p_;
        let g: Basic = chain(&[
            d2(&m2(&neg(&c)?, &sb)?, &m2(&n, &a)?)?,
            pw(&t, &rat(&power))?,
            pw(&d2(&s2(&ts, &b)?, &a)?, &int(exponent))?,
        ])?;
        let back: Basic = pw(&a2(&m2(&a, &pw(x, &neg(&n)?)?)?, &b)?, &d2(&int(1), &sb)?)?;
        let forward: Basic = pw(&d2(&s2(&ts, &b)?, &a)?, &d2(&int(-1), &n)?)?;
        (g, back, forward)
    } else {
        return Ok(None);
    };
    let g: Basic = cancel(&g)?;
    if !rational_function(&g, std::slice::from_ref(&t)) {
        return Ok(None);
    }
    Ok(Some(Substitution {
        integrand: g,
        variable: t,
        back,
        forward,
        facts: Vec::new(),
    }))
}

fn sqrt(e: &Basic) -> R {
    pw(e, &d2(&int(1), &int(2))?)
}

fn even_substitution(f: &Basic, x: &Basic) -> PolyResult<Option<Substitution>> {
    if !f.has(x)
        || !pows(f)
            .iter()
            .any(|p| p.has(x) && rational_of(&p.args()[1]).is_some_and(|r| den(&r) == 2))
    {
        return Ok(None);
    }
    let odd = attempt(|| cancel(&a2(&subs(f, x, &neg(x)?)?, f)?))?;
    if odd != Some(Basic::zero()) {
        return Ok(None);
    }
    let s: Basic = fresh("s", &[(Predicate::positive, true)]);
    let root: Basic = sqrt(&s)?;
    let g = attempt(|| cancel(&powsimp_force(&d2(&subs(f, x, &root)?, &m2(&int(2), &root)?)?)?))?;
    let Some(g) = g else { return Ok(None) };
    if g.has(&root) {
        return Ok(None);
    }
    Ok(Some(Substitution {
        integrand: g,
        variable: s,
        back: pw(x, &int(2))?,
        forward: root,
        facts: Vec::new(),
    }))
}

fn exponential_polynomial(g: &Basic, t: &Basic) -> PolyResult<bool> {
    let mut replacement: HashMap<Basic, Basic> = HashMap::new();
    let mut generators: Vec<Basic> = vec![t.clone()];
    for node in g.preorder() {
        if !matches!(node.head(), Head::Function(Function::Exp)) || !node.has(t) || replacement.contains_key(&node) {
            continue;
        }
        if !polynomial(&node.args()[0], std::slice::from_ref(t)) {
            return Ok(false);
        }
        let generator_: Basic = fresh("e", &[]);
        replacement.insert(node.clone(), generator_.clone());
        generators.push(generator_);
    }
    Ok(rational_function(
        &crate::core::function::xreplace_evaluated(g, &replacement)?,
        &generators,
    ))
}

fn logarithmic_substitution(f: &Basic, x: &Basic) -> PolyResult<Option<Substitution>> {
    let log_x: Basic = log(x)?;
    if !f.has(&log_x) {
        return Ok(None);
    }
    let t: Basic = fresh("t", &[(Predicate::real, true)]);
    let et: Basic = exp(&t)?;
    let g = attempt(|| cancel(&powsimp_force(&m2(&subs(f, x, &et)?, &et)?)?))?;
    let Some(g) = g else { return Ok(None) };
    if has_head(&g, &|n| matches!(n.head(), Head::Function(Function::Log))) {
        return Ok(None);
    }
    let l: Basic = fresh("L", &[]);
    if !rational_function(&subs(f, &log_x, &l)?, std::slice::from_ref(&l)) && !exponential_polynomial(&g, &t)? {
        return Ok(None);
    }
    Ok(Some(Substitution {
        integrand: g,
        variable: t,
        back: log_x,
        forward: et,
        facts: Vec::new(),
    }))
}

/// The rationalising substitutions of the integrand, at most three
/// (`power_substitutions`).
pub fn power_substitutions(f: &Basic, x: &Basic) -> PolyResult<Vec<Substitution>> {
    let mut found: Vec<Substitution> = Vec::new();
    for (base, k) in fractional_powers(f, x)? {
        if found.len() >= SUBSTITUTIONS {
            break;
        }
        if let Some(s) = power_substitution(f, x, &base, k)? {
            found.push(s);
        }
    }
    type Maker = fn(&Basic, &Basic) -> PolyResult<Option<Substitution>>;
    let makers: [Maker; 4] = [
        exponential_substitution,
        logarithmic_substitution,
        even_substitution,
        binomial_substitution,
    ];
    for maker in makers {
        if found.len() >= SUBSTITUTIONS {
            break;
        }
        if let Some(s) = maker(f, x)? {
            if found.iter().all(|other| s.back != other.back) {
                found.push(s);
            }
        }
    }
    if found.len() < SUBSTITUTIONS {
        let regions: Vec<Substitution> = moebius_substitutions(f, x)?;
        if regions
            .iter()
            .all(|region| found.iter().all(|other| region.back != other.back))
        {
            found.extend(regions);
        }
    }
    if found.len() < SUBSTITUTIONS {
        for s in nested_substitutions(f, x)? {
            if found.iter().all(|other| s.back != other.back) {
                found.push(s);
            }
        }
    }
    Ok(found)
}

fn nested_substitutions(f: &Basic, x: &Basic) -> PolyResult<Vec<Substitution>> {
    let radicals: Vec<Basic> = pows(f)
        .into_iter()
        .filter(|n| rational_of(&n.args()[1]).is_some_and(|r| den(&r) > 1) && n.has(x))
        .collect();
    let nested = |node: &Basic| radicals.iter().any(|other| other != node && node.args()[0].has(other));
    if !radicals.iter().any(nested) {
        return Ok(Vec::new());
    }
    let mut found: Vec<Substitution> = Vec::new();
    for node in &radicals {
        if nested(node) {
            continue;
        }
        let base: &Basic = &node.args()[0];
        let Some(r) = rational_of(&node.args()[1]) else {
            continue;
        };
        let Some((a, b)) = linear(base, x)? else { continue };
        let k: i64 = den(&r);
        let t: Basic = fresh("t", &[(Predicate::positive, true)]);
        let forward: Basic = d2(&s2(&pw(&t, &int(k))?, &b)?, &a)?;
        let g = attempt(|| {
            let jacobian: Basic = d2(&chain(&[subs(f, x, &forward)?, int(k), pw(&t, &int(k - 1))?])?, &a)?;
            cancel(&powsimp_force(&jacobian)?)
        })?;
        let Some(g) = g else { continue };
        let root: Basic = pw(base, &d2(&int(1), &int(k))?)?;
        for s in power_substitutions(&g, &t)? {
            found.push(Substitution {
                integrand: s.integrand,
                variable: s.variable,
                back: subs(&s.back, &t, &root)?,
                forward: subs(&forward, &t, &s.forward)?,
                facts: s.facts,
            });
        }
    }
    Ok(found)
}

/// The antiderivative in the new variable back in the old one
/// (`substitute_back`).
pub fn substitute_back(big_f: &Basic, t: &Basic, back: &Basic) -> R {
    subs(big_f, t, back)
}
