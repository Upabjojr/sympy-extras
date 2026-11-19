//! Port of `sympy_extras/integrals/heurisch.py`: the heuristic (parallel)
//! Risch integrator, Bronstein's "poor man's integrator": the integrand
//! written in its components as a rational function `F(t_1, ..., t_n)`,
//! an antiderivative sought as `P/Q + sum b_j log q_j + sum c_k atan r_k`
//! with unknown coefficients, solved as a linear system over the field of
//! the constants (`QQ`, `QQ(parameters)` or a number field).

#![allow(clippy::mutable_key_type, clippy::type_complexity, clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{value_error, PolyError, PolyResult};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rat(p: i64, q: i64) -> Basic {
    Basic::from_rational(
        crate::numbers::Rat::from(crate::numbers::Int::from(p))
            / crate::numbers::Rat::from(crate::numbers::Int::from(q)),
    )
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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sqrt(a: &Basic) -> R {
    pw(a, &rat(1, 2))
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
fn apply(f: Function, args: &[Basic]) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), args.to_vec()), args.to_vec())
}
fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::derivative(e, x)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn expand_mul(e: &Basic) -> R {
    crate::core::function::expand_mul(e, true)
}
fn expand_power_exp(e: &Basic) -> R {
    use crate::core::function::Hints;
    let hints = Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: true,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}
fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn sorted(items: Vec<Basic>) -> Vec<Basic> {
    crate::core::sorting::ordered(items)
}
fn ops(e: &Basic) -> i64 {
    crate::core::function::count_ops(e).unwrap_or(0)
}
fn has_any(e: &Basic, fs: &[Function]) -> bool {
    e.preorder()
        .iter()
        .any(|n| matches!(func(n), Some(f) if fs.contains(&f)))
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn imaginary() -> Basic {
    Basic::atom(Head::ImaginaryUnit)
}
fn insert(set: &mut Vec<Basic>, e: Basic) {
    if !set.contains(&e) {
        set.push(e);
    }
}
fn extend(set: &mut Vec<Basic>, more: Vec<Basic>) {
    for e in more {
        insert(set, e);
    }
}
fn is_polynomial_in(e: &Basic, syms: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, syms, crate::core::expr::PolynomialKind::Polynomial) == Some(true)
}

/// The functions whose integrands are refused (`_REFUSED`).
const REFUSED: [Function; 8] = [
    Function::Abs,
    Function::Re,
    Function::Im,
    Function::Sign,
    Function::Heaviside,
    Function::DiracDelta,
    Function::Floor,
    Function::Ceiling,
];

/// The components of `f` in `x`: the variable, the function applications
/// and the non-integer powers it is built from (`components`).
pub fn components(f: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let mut result: Vec<Basic> = Vec::new();
    if !f.has(x) {
        return Ok(result);
    }
    if f.is_symbol() {
        insert(&mut result, f.clone());
    } else if matches!(f.head(), Head::Function(_) | Head::Undefined(_) | Head::Derivative) {
        for g in f.args() {
            extend(&mut result, components(g, x)?);
        }
        insert(&mut result, f.clone());
    } else if f.is_pow() {
        let (base, exponent) = (&f.args()[0], &f.args()[1]);
        extend(&mut result, components(base, x)?);
        if !exponent.is_integer_value() {
            if let Some(r) = exponent.as_rational() {
                insert(
                    &mut result,
                    pw(
                        base,
                        &Basic::from_rational(
                            crate::numbers::Rat::from(crate::numbers::Int::from(1))
                                / crate::numbers::Rat::from(crate::numbers::denom(r)),
                        ),
                    )?,
                );
            } else {
                extend(&mut result, components(exponent, x)?);
                insert(&mut result, f.clone());
            }
        }
    } else {
        for g in f.args() {
            extend(&mut result, components(g, x)?);
        }
    }
    Ok(result)
}

/// The dummies of a name, the same ones for every call (`_symbols`).
fn symbols(name: &str, n: usize) -> Vec<Basic> {
    static CACHE: OnceLock<std::sync::Mutex<HashMap<String, Vec<Basic>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("the cache of the symbols");
    let known: &mut Vec<Basic> = guard.entry(name.to_string()).or_default();
    while known.len() < n {
        let k: usize = known.len();
        known.push(dummy(&format!("{}{}", name, k)));
    }
    known[..n].to_vec()
}

const BESSEL: [Function; 8] = [
    Function::BesselJ,
    Function::BesselY,
    Function::Hankel1,
    Function::Hankel2,
    Function::BesselI,
    Function::BesselK,
    Function::SphericalJ,
    Function::SphericalY,
];

/// `(d/dz f(n, z), d/dz f(n - 1, z))` in terms of `f(n - 1, z)`, `f(n, z)`
/// (`_BesselTable`).
fn bessel_diffs(f: Function, n: &Basic, z: &Basic) -> PolyResult<(Basic, Basic)> {
    let at = |order: &Basic| apply(f, &[order.clone(), z.clone()]);
    let n1: Basic = s2(n, &int(1))?;
    let (lower, current) = (at(&n1)?, at(n)?);
    let n_over: Basic = d2(&m2(n, &current)?, z)?;
    let lower_over: Basic = d2(&m2(&n1, &lower)?, z)?;
    Ok(match f {
        Function::BesselI => (s2(&lower, &n_over)?, a2(&lower_over, &current)?),
        Function::BesselK => (s2(&neg(&lower)?, &n_over)?, s2(&lower_over, &current)?),
        Function::SphericalJ | Function::SphericalY => (
            s2(&lower, &d2(&m2(&a2(n, &int(1))?, &current)?, z)?)?,
            s2(&lower_over, &current)?,
        ),
        _ => (s2(&lower, &n_over)?, s2(&lower_over, &current)?),
    })
}

/// The derivatives of the components, cached (`_DiffCache`).
struct DiffCache {
    cache: Vec<(Basic, Basic)>,
    x: Basic,
}

impl DiffCache {
    fn lookup(&self, f: &Basic) -> Option<Basic> {
        self.cache.iter().find(|(k, _)| k == f).map(|(_, v)| v.clone())
    }

    fn set(&mut self, f: Basic, value: Basic) {
        match self.cache.iter_mut().find(|(k, _)| *k == f) {
            Some(entry) => entry.1 = value,
            None => self.cache.push((f, value)),
        }
    }

    fn get(&mut self, f: &Basic) -> R {
        if let Some(v) = self.lookup(f) {
            return Ok(v);
        }
        match func(f) {
            Some(g) if BESSEL.contains(&g) && f.args().len() == 2 => {
                let (n, z) = (f.args()[0].clone(), f.args()[1].clone());
                let (d0, d1) = bessel_diffs(g, &n, &z)?;
                let dz: Basic = self.get(&z)?;
                self.set(f.clone(), m2(&d0, &dz)?);
                self.set(apply(g, &[s2(&n, &int(1))?, z.clone()])?, m2(&d1, &dz)?);
            }
            _ => {
                let value: Basic = expand_power_exp(&cancel(&diff(f, &self.x)?)?)?;
                self.set(f.clone(), value);
            }
        }
        Ok(self.lookup(f).expect("cached"))
    }
}

fn wild(name: &str, x: &Basic) -> Basic {
    crate::core::symbol::Wild::new(name, vec![x.clone()], Vec::new(), &[]).0
}

fn matched(e: &Basic, pattern: &Basic, wilds: &[&Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let Some(found) = crate::core::matching::match_pattern(e, pattern, &[], false)? else {
        return Ok(None);
    };
    Ok(Some(
        wilds
            .iter()
            .map(|w| {
                found
                    .iter()
                    .find(|(k, _)| k == *w)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(|| int(0))
            })
            .collect(),
    ))
}

fn special_candidates(terms: &[Basic], x: &Basic, cache: &mut DiffCache, f: &Basic) -> PolyResult<Vec<Basic>> {
    let (a, b, c) = (wild("a", x), wild("b", x), wild("c", x));
    let mut found: Vec<Basic> = Vec::new();
    let denominator: Basic = crate::core::expr::as_numer_denom(f)?.1;
    let over_x: bool =
        is_polynomial_in(&denominator, std::slice::from_ref(x)) && subs(&denominator, x, &int(0))? == int(0);
    let with_erf: bool = has_any(f, &[Function::Erf, Function::Erfi]);
    let a_xb: Basic = m2(&a, &pw(x, &b)?)?;
    let x2: Basic = pw(x, &int(2))?;
    let positive = |e: &Basic| is(e, Predicate::positive) == Some(true);
    let negative = |e: &Basic| is(e, Predicate::negative) == Some(true);
    for g in terms {
        // li(...) has no node in the Rust core: the branch of `LiNode` cannot apply
        if func(g) == Some(Function::Exp) {
            let argument: &Basic = &g.args()[0];
            if let Some(values) = matched(argument, &m2(&a, &x2)?, &[&a])? {
                if with_erf && negative(&values[0]) {
                    let inner: Basic = m2(&m2(&sqrt(&int(2))?, &sqrt(&neg(&values[0])?)?)?, x)?;
                    insert(&mut found, apply(Function::Erf, &[inner])?);
                }
            }
            if let Some(values) = matched(argument, &a_xb, &[&a, &b])? {
                if over_x && values[0] != int(0) && is(&values[1], Predicate::nonzero) == Some(true) {
                    insert(
                        &mut found,
                        apply(Function::Ei, &[m2(&values[0], &pw(x, &values[1])?)?])?,
                    );
                }
            }
        } else if matches!(func(g), Some(Function::Sin | Function::Cos)) && over_x {
            if let Some(values) = matched(&g.args()[0], &a_xb, &[&a, &b])? {
                if values[0] != int(0) && is(&values[1], Predicate::nonzero) == Some(true) {
                    let argument: Basic = m2(&values[0], &pw(x, &values[1])?)?;
                    insert(&mut found, apply(Function::Si, std::slice::from_ref(&argument))?);
                    insert(&mut found, apply(Function::Ci, &[argument])?);
                }
            }
        } else if g.is_pow()
            && g.args()[1]
                .as_rational()
                .is_some_and(|r| crate::numbers::denom(r) == crate::numbers::Int::from(2))
        {
            let base: &Basic = &g.args()[0];
            if let Some(values) = matched(base, &a2(&m2(&a, &x2)?, &b)?, &[&a, &b])? {
                let (aa, bb) = (&values[0], &values[1]);
                if positive(bb) {
                    if positive(aa) {
                        insert(&mut found, apply(Function::Asinh, &[m2(&sqrt(&d2(aa, bb)?)?, x)?])?);
                    } else if negative(aa) {
                        insert(
                            &mut found,
                            apply(Function::Asin, &[m2(&sqrt(&d2(&neg(aa)?, bb)?)?, x)?])?,
                        );
                    }
                }
            }
            if let Some(values) = matched(base, &s2(&m2(&a, &x2)?, &b)?, &[&a, &b])? {
                let (aa, bb) = (&values[0], &values[1]);
                if positive(bb) {
                    let radicand: Basic = s2(&m2(aa, &x2)?, bb)?;
                    if positive(aa) {
                        let inside: Basic = a2(
                            &m2(&m2(&int(2), &sqrt(aa)?)?, &sqrt(&radicand)?)?,
                            &m2(&m2(&int(2), aa)?, x)?,
                        )?;
                        let primitive: Basic = d2(&log(&inside)?, &sqrt(aa)?)?;
                        cache.set(primitive.clone(), d2(&int(1), &sqrt(&radicand)?)?);
                        insert(&mut found, primitive);
                    } else if negative(aa) {
                        let ratio: Basic = d2(&m2(&sqrt(&neg(aa)?)?, x)?, &sqrt(&radicand)?)?;
                        let value: Basic = m2(
                            &m2(&d2(&neg(bb)?, &int(2))?, &sqrt(&neg(aa)?)?)?,
                            &apply(Function::Atan, &[ratio])?,
                        )?;
                        insert(&mut found, value);
                    }
                }
            }
            let quadratic: Basic = a2(&a2(&m2(&a, &x2)?, &m2(&b, x)?)?, &c)?;
            if let Some(values) = matched(base, &quadratic, &[&a, &b, &c])? {
                if values[0] != int(0) && values[1] != int(0) {
                    let (aa, bb, cc) = (&values[0], &values[1], &values[2]);
                    let discriminant: Basic = s2(&pw(bb, &int(2))?, &m2(&m2(&int(4), aa)?, cc)?)?;
                    let linear: Basic = a2(&m2(&m2(&int(2), aa)?, x)?, bb)?;
                    if negative(aa) && positive(&discriminant) {
                        insert(
                            &mut found,
                            apply(Function::Asin, &[d2(&linear, &sqrt(&discriminant)?)?])?,
                        );
                    } else if positive(aa) && negative(&discriminant) {
                        insert(
                            &mut found,
                            apply(Function::Asinh, &[d2(&linear, &sqrt(&neg(&discriminant)?)?)?])?,
                        );
                    } else if positive(aa) && positive(&discriminant) {
                        let inside: Basic = a2(
                            &a2(
                                &m2(&m2(&int(2), &sqrt(aa)?)?, &sqrt(base)?)?,
                                &m2(&m2(&int(2), aa)?, x)?,
                            )?,
                            bb,
                        )?;
                        let primitive: Basic = d2(&log(&inside)?, &sqrt(aa)?)?;
                        cache.set(primitive.clone(), d2(&int(1), &sqrt(base)?)?);
                        insert(&mut found, primitive);
                    }
                }
            }
        }
    }
    Ok(found)
}

fn denominator_of(q: &Basic) -> i64 {
    q.as_rational()
        .and_then(|r| i64::try_from(&crate::numbers::denom(r)).ok())
        .unwrap_or(1)
}

fn exponential_family(g: &Basic) -> PolyResult<(Basic, Basic)> {
    let (content, mut w) = crate::core::exprtools::as_content_primitive(&g.args()[0], false, true)?;
    let mut q: Basic = if content.is_rational_class() { content } else { int(1) };
    if crate::core::printing::could_extract_minus_sign(&w) {
        w = neg(&w)?;
        q = neg(&q)?;
    }
    Ok((w, q))
}

fn gcd_i(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i(b, a % b)
    }
}

fn exponential_bases(terms: &[Basic]) -> PolyResult<Vec<Basic>> {
    let mut families: Vec<(Basic, Vec<Basic>)> = Vec::new();
    for g in terms {
        if func(g) == Some(Function::Exp) {
            let (w, q) = exponential_family(g)?;
            match families.iter_mut().find(|(k, _)| *k == w) {
                Some(entry) => entry.1.push(q),
                None => families.push((w, vec![q])),
            }
        }
    }
    let mut result: Vec<Basic> = terms
        .iter()
        .filter(|g| func(g) != Some(Function::Exp))
        .cloned()
        .collect();
    for (w, members) in families {
        let mut d: i64 = 1;
        for q in &members {
            let k: i64 = denominator_of(q);
            d = d / gcd_i(d, k) * k;
        }
        insert(&mut result, exp(&d2(&w, &int(d))?)?);
    }
    Ok(result)
}

fn power_coefficients(e: &Basic, v: &Basic) -> PolyResult<Option<Vec<(i64, Basic)>>> {
    let mut coefficients: Vec<(i64, Vec<Basic>)> = Vec::new();
    for term in crate::core::expr::terms_of(&expand(e)?) {
        let mut degree: i64 = 0;
        let mut rest: Vec<Basic> = Vec::new();
        let factors: Vec<Basic> = if term.is_mul() {
            term.args().to_vec()
        } else {
            vec![term.clone()]
        };
        for factor in factors {
            if factor == *v {
                degree += 1;
            } else if factor.is_pow()
                && factor.args()[0] == *v
                && factor.args()[1].is_integer_value()
                && factor.args()[1]
                    .as_rational()
                    .is_some_and(|r| *r > crate::numbers::Rat::from(crate::numbers::Int::from(0)))
            {
                degree += i64::try_from(&crate::numbers::numer(
                    factor.args()[1].as_rational().expect("an integer"),
                ))
                .unwrap_or(0);
            } else if factor.has(v) {
                return Ok(None);
            } else {
                rest.push(factor);
            }
        }
        let product: Basic = mul_all(&rest)?;
        match coefficients.iter_mut().find(|(d, _)| *d == degree) {
            Some(entry) => entry.1.push(product),
            None => coefficients.push((degree, vec![product])),
        }
    }
    let mut result: Vec<(i64, Basic)> = Vec::new();
    for (degree, parts) in coefficients {
        let total: Basic = add_all(&parts)?;
        if total != int(0) {
            result.push((degree, total));
        }
    }
    Ok(Some(result))
}

fn degree(e: &Basic, v: &Basic) -> PolyResult<i64> {
    Ok(power_coefficients(e, v)?
        .and_then(|c| c.iter().map(|(d, _)| *d).max())
        .unwrap_or(0))
}

fn degree_in(e: &Basic, g: &Basic) -> PolyResult<i64> {
    let d: Basic = dummy("g");
    let combined: Basic = crate::simplify::powsimp::powsimp(
        e,
        true,
        crate::simplify::powsimp::Combine::Exp,
        false,
        &crate::simplify::powsimp::count_ops,
    )?;
    degree(&replaced(&combined, &[(g.clone(), d.clone())])?, &d)
}

/// The constants of an expression as placeholders, so that the
/// polynomial routines see the generators only (`_Placeholders`).
struct Placeholders {
    gens: Vec<Basic>,
    rational: bool,
    forward: Vec<(Basic, Basic)>,
    back: Vec<(Basic, Basic)>,
}

impl Placeholders {
    fn new(gens: &[Basic], rational: bool) -> Placeholders {
        Placeholders {
            gens: gens.to_vec(),
            rational,
            forward: Vec::new(),
            back: Vec::new(),
        }
    }

    fn placeholder(&mut self, e: &Basic) -> Basic {
        if let Some((_, s)) = self.forward.iter().find(|(k, _)| k == e) {
            return s.clone();
        }
        let symbol: Basic = dummy(&format!("c{}", self.forward.len()));
        self.forward.push((e.clone(), symbol.clone()));
        self.back.push((symbol.clone(), e.clone()));
        symbol
    }

    fn is_placeholder(&self, s: &Basic) -> bool {
        self.back.iter().any(|(k, _)| k == s)
    }

    fn abstract_(&mut self, e: &Basic) -> R {
        if e.is_rational_class() || (e.is_symbol() && !self.is_placeholder(e)) {
            return Ok(e.clone());
        }
        if *e == imaginary() {
            return Ok(self.placeholder(e));
        }
        if e.is_add() {
            let parts: Vec<Basic> = e.args().iter().map(|a| self.abstract_(a)).collect::<PolyResult<_>>()?;
            return add_all(&parts);
        }
        if e.is_mul() {
            let parts: Vec<Basic> = e.args().iter().map(|a| self.abstract_(a)).collect::<PolyResult<_>>()?;
            return mul_all(&parts);
        }
        if e.is_pow() && e.args()[1].is_integer_value() {
            let base: Basic = self.abstract_(&e.args()[0])?;
            let nonnegative: bool = e.args()[1]
                .as_rational()
                .is_some_and(|r| *r >= crate::numbers::Rat::from(crate::numbers::Int::from(0)));
            if nonnegative || self.rational || self.gens.iter().any(|g| base.has(g)) {
                return pw(&base, &e.args()[1]);
            }
            return Ok(self.placeholder(e));
        }
        if e.free_symbols().iter().any(|s| self.gens.contains(s)) {
            return Ok(e.clone());
        }
        Ok(self.placeholder(e))
    }

    fn restore(&self, e: &Basic) -> R {
        let mut e: Basic = e.clone();
        while !self.back.is_empty() && e.free_symbols().iter().any(|s| self.is_placeholder(s)) {
            e = replaced(&e, &self.back)?;
        }
        Ok(e)
    }
}

fn with_constants(gens: &[Basic], exprs: &[Basic]) -> PolyResult<(Placeholders, Vec<Basic>)> {
    let mut p: Placeholders = Placeholders::new(gens, false);
    let out: Vec<Basic> = exprs.iter().map(|e| p.abstract_(e)).collect::<PolyResult<_>>()?;
    Ok((p, out))
}

fn factor_list_of(e: &Basic, gens: &[Basic]) -> PolyResult<(Basic, Vec<(Basic, i64)>)> {
    let (placeholders, abstracted) = with_constants(gens, std::slice::from_ref(e))?;
    let (mut coefficient, factors) = crate::polys::polytools::factor_list(&abstracted[0], gens)?;
    let mut result: Vec<(Basic, i64)> = Vec::new();
    for (base, k) in factors {
        let restored: Basic = placeholders.restore(&base)?;
        if !k.is_integer_value() {
            return Err(value_error("a symbolic multiplicity".to_string()));
        }
        if !gens.iter().any(|g| restored.has(g)) {
            coefficient = m2(&coefficient, &pw(&base, &k)?)?;
            continue;
        }
        result.push((
            restored,
            i64::try_from(&crate::numbers::numer(k.as_rational().expect("an integer"))).unwrap_or(1),
        ));
    }
    Ok((placeholders.restore(&coefficient)?, result))
}

fn gcd_of(f: &Basic, g: &Basic, gens: &[Basic]) -> R {
    let (placeholders, v) = with_constants(gens, &[f.clone(), g.clone()])?;
    placeholders.restore(&crate::polys::polytools::gcd_list(&v, gens)?)
}

fn lcm_of(f: &Basic, g: &Basic, gens: &[Basic]) -> R {
    let (placeholders, v) = with_constants(gens, &[f.clone(), g.clone()])?;
    placeholders.restore(&crate::polys::polytools::lcm_list(&v, gens)?)
}

fn total_degree(e: &Basic, gens: &[Basic]) -> PolyResult<Option<i64>> {
    let (_, v) = with_constants(gens, std::slice::from_ref(e))?;
    match crate::polys::polytools::Poly::new(&v[0], gens) {
        Ok(p) => Ok(Some(p.total_degree())),
        Err(PolyError::Polys(..)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn primitive(p: &Basic, y: &Basic) -> PolyResult<(Basic, Basic)> {
    let Some(coefficients) = power_coefficients(p, y)? else {
        return Err(value_error(format!("{} is not a polynomial in {}", p, y)));
    };
    let mut placeholders: Placeholders = Placeholders::new(&[], false);
    let values: Vec<Basic> = coefficients
        .iter()
        .map(|(_, c)| placeholders.abstract_(c))
        .collect::<PolyResult<_>>()?;
    let mut gens: Vec<Basic> = Vec::new();
    for v in &values {
        for s in v.free_symbols() {
            if s.is_symbol() && !placeholders.is_placeholder(&s) && !gens.contains(&s) {
                gens.push(s);
            }
        }
    }
    gens.sort_by_cached_key(crate::core::sorting::sort_key);
    let content: Basic = placeholders.restore(&crate::polys::polytools::gcd_list(&values, &gens)?)?;
    if content == int(0) {
        return Ok((int(0), p.clone()));
    }
    let mut terms: Vec<Basic> = Vec::new();
    for (k, c) in &coefficients {
        terms.push(m2(&cancel(&d2(c, &content)?)?, &pw(y, &int(*k))?)?);
    }
    Ok((content, add_all(&terms)?))
}

fn root_factors(f: &Basic, z: &Basic, field: Option<&str>) -> PolyResult<Vec<Basic>> {
    let (placeholders, v) = with_constants(std::slice::from_ref(z), std::slice::from_ref(f))?;
    let poly = crate::polys::polytools::Poly::new(&v[0], std::slice::from_ref(z))?;
    let zeros = crate::polys::polyroots::roots(&v[0], std::slice::from_ref(z), true, true, field)?;
    if zeros.is_empty() {
        return Ok(vec![f.clone()]);
    }
    let mut factors: Vec<Basic> = Vec::new();
    let mut found: i64 = 0;
    let keys: Vec<Basic> = sorted(zeros.iter().map(|(r, _)| r.clone()).collect());
    for r in keys {
        let k: usize = zeros.iter().find(|(q, _)| **q == r).map(|(_, k)| *k).unwrap_or(0);
        let factor: Basic = s2(z, &placeholders.restore(&r)?)?;
        for _ in 0..k {
            factors.push(factor.clone());
        }
        found += k as i64;
    }
    if found < poly.degree(0)? {
        let product: Basic = mul_all(&factors)?;
        factors.push(cancel(&d2(&expand(f)?, &product)?)?);
    }
    Ok(factors)
}

fn gives_up(e: &PolyError) -> bool {
    matches!(
        e,
        PolyError::Polys(..) | PolyError::Value(_) | PolyError::Type(_) | PolyError::ZeroDivision(_)
    ) || matches!(e, PolyError::NotImplemented(m) if !m.starts_with("not ported"))
}

fn square_root_form(e: &Basic, x: &Basic, terms: &[Basic]) -> PolyResult<Option<(Basic, Basic)>> {
    let expanded: Basic = expand(e)?;
    let mut families: Vec<(Basic, i64)> = Vec::new();
    for g in components(&expanded, x)? {
        if func(&g) == Some(Function::Exp) {
            let (w, q) = exponential_family(&g)?;
            let k: i64 = denominator_of(&q);
            match families.iter_mut().find(|(f, _)| *f == w) {
                Some(entry) => entry.1 = entry.1 / gcd_i(entry.1, k) * k,
                None => families.push((w, k)),
            }
        }
    }
    let mut dummies: Vec<(Basic, Basic)> = Vec::new();
    for (w, d) in &families {
        dummies.push((exp(&d2(w, &int(2 * d))?)?, dummy("e")));
    }
    let mut reduced: Basic = expanded.clone();
    for (g, d) in &dummies {
        reduced = subs(&reduced, g, d)?;
    }
    let back: Vec<(Basic, Basic)> = dummies.iter().map(|(g, d)| (d.clone(), g.clone())).collect();
    let mut gens: Vec<Basic> = components(&reduced, x)?
        .into_iter()
        .filter(|g| func(g) != Some(Function::Exp))
        .collect();
    gens.sort_by_key(|g| g.to_string());
    gens.extend(dummies.iter().map(|(_, d)| d.clone()));
    if gens.is_empty() {
        return Ok(None);
    }
    let (constant, factors) = match factor_list_of(&reduced, &gens) {
        Ok(found) => found,
        Err(e) if gives_up(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    if !factors.is_empty()
        && factors.iter().all(|(_, k)| k % 2 == 0)
        && is(&constant, Predicate::positive) == Some(true)
    {
        let mut u: Basic = sqrt(&constant)?;
        for (base, k) in &factors {
            u = m2(&u, &pw(base, &int(k / 2))?)?;
        }
        return Ok(Some((replaced(&u, &back)?, int(0))));
    }
    for v in &gens {
        let Some(coefficients) = power_coefficients(&reduced, v)? else {
            continue;
        };
        if coefficients.is_empty() || coefficients.iter().map(|(d, _)| *d).max() != Some(2) {
            continue;
        }
        let get = |k: i64| {
            coefficients
                .iter()
                .find(|(d, _)| *d == k)
                .map(|(_, c)| c.clone())
                .unwrap_or_else(|| int(0))
        };
        let aa: Basic = replaced(&get(2), &back)?;
        let bb: Basic = replaced(&get(1), &back)?;
        let component: Basic = back
            .iter()
            .find(|(d, _)| d == v)
            .map(|(_, g)| g.clone())
            .unwrap_or_else(|| v.clone());
        if aa.has(x) || is(&aa, Predicate::positive) != Some(true) {
            continue;
        }
        let (number, rest) = crate::core::expr::as_coeff_add(&bb);
        let linears: Vec<Basic> = if number == int(0) || rest == int(0) {
            vec![bb.clone()]
        } else {
            vec![bb.clone(), rest.clone()]
        };
        for linear in linears {
            let u: Basic = m2(&sqrt(&aa)?, &a2(&component, &d2(&linear, &m2(&int(2), &aa)?)?)?)?;
            let c: Basic = cancel(&s2(&expand(e)?, &expand(&pw(&u, &int(2))?)?)?)?;
            if c.has(x) {
                let within = |g: &Basic| -> PolyResult<bool> {
                    Ok(components(&exp(g)?, x)?
                        .iter()
                        .filter(|t| *t != x)
                        .all(|t| terms.contains(t)))
                };
                if !(within(&c)? || within(&neg(&c)?)?) {
                    continue;
                }
            }
            return Ok(Some((u, c)));
        }
    }
    Ok(None)
}

fn factors_of(e: &Basic) -> PolyResult<Vec<Basic>> {
    match crate::polys::polytools::factor_list(e, &[]) {
        Ok((_, found)) => Ok(found.into_iter().map(|(q, _)| q).collect()),
        Err(err) if gives_up(&err) => Ok(vec![e.clone()]),
        Err(err) => Err(err),
    }
}

fn tower_candidates(terms: &[Basic], x: &Basic, cache: &mut DiffCache, f: &Basic) -> PolyResult<Vec<Basic>> {
    let mut found: Vec<Basic> = Vec::new();
    let (numerator, denominator) = crate::core::expr::as_numer_denom(f)?;
    let factors: Vec<Basic> = factors_of(&denominator)?;
    let logarithms: Vec<Basic> = f
        .atoms(Some(&|n: &Basic| func(n) == Some(Function::Log)))
        .iter()
        .map(|g| g.args()[0].clone())
        .collect();
    for g in terms {
        let is_exp: bool = func(g) == Some(Function::Exp);
        let (theta, exponential): (Basic, Basic) = if is_exp {
            (g.args()[0].clone(), g.clone())
        } else if func(g) == Some(Function::Log) {
            (g.clone(), g.args()[0].clone())
        } else {
            continue;
        };
        let dtheta: Basic = cancel(&diff(&theta, x)?)?;
        if dtheta == int(0) {
            continue;
        }
        let above: i64 = degree_in(&numerator, &exponential)? + if is_exp { 0 } else { 1 };
        let below: i64 = degree_in(&denominator, &exponential)?;
        let mut multiples: Vec<i64> = (1..=above.max(1)).collect();
        multiples.extend((1..=below).map(|k| -k));
        for q in &factors {
            if !q.has(x) {
                continue;
            }
            let alpha: Basic = cancel(&d2(&diff(q, x)?, &dtheta)?)?;
            if alpha == int(0) || alpha.has(x) {
                continue;
            }
            let beta: Basic = cancel(&s2(q, &m2(&alpha, &theta)?)?)?;
            if beta.has(x) {
                continue;
            }
            let c: Basic = d2(&beta, &alpha)?;
            for k in &multiples {
                let kk: Basic = int(*k);
                let shifted: Basic = a2(&theta, &c)?;
                let candidate: Basic = apply(Function::Ei, &[m2(&kk, &shifted)?])?;
                let value: Basic = d2(
                    &m2(&m2(&exp(&m2(&kk, &c)?)?, &pw(&exponential, &kk)?)?, &dtheta)?,
                    &shifted,
                )?;
                cache.set(candidate.clone(), value);
                insert(&mut found, candidate);
            }
        }
        if !is_exp {
            continue;
        }
        for (offset, argument) in [(1i64, neg(g)?), (-1, g.clone())] {
            let near = |h: &Basic| -> PolyResult<bool> {
                Ok(cancel(&s2(&s2(h, &int(offset))?, g)?)? == int(0)
                    || cancel(&a2(&a2(h, &int(offset))?, g)?)? == int(0))
            };
            let mut hit: bool = false;
            for q in factors.iter().chain(logarithms.iter()) {
                if near(q)? {
                    hit = true;
                    break;
                }
            }
            if hit {
                let candidate: Basic = apply(Function::PolyLog, &[int(2), argument.clone()])?;
                cache.set(candidate.clone(), m2(&neg(&log(&s2(&int(1), &argument)?)?)?, &dtheta)?);
                insert(&mut found, candidate);
            }
        }
    }
    let mut exponents: Vec<Basic> = Vec::new();
    for term in crate::core::expr::terms_of(f) {
        let mut theta_term: Basic = int(0);
        let pieces: Vec<Basic> = if term.is_mul() {
            term.args().to_vec()
        } else {
            vec![term.clone()]
        };
        for piece in pieces {
            if func(&piece) == Some(Function::Exp) {
                theta_term = a2(&theta_term, &piece.args()[0])?;
                exponents.push(piece.args()[0].clone());
            } else if piece.is_pow() && func(&piece.args()[0]) == Some(Function::Exp) && !piece.args()[1].has(x) {
                theta_term = a2(&theta_term, &m2(&piece.args()[1], &piece.args()[0].args()[0])?)?;
            }
        }
        if theta_term != int(0) {
            exponents.push(theta_term);
        }
    }
    for theta_term in &exponents {
        for (sign_, function) in [(-1i64, Function::Erf), (1, Function::Erfi)] {
            let Some((u, c)) = square_root_form(&m2(&int(sign_), theta_term)?, x, terms)? else {
                continue;
            };
            let candidate: Basic = apply(function, std::slice::from_ref(&u))?;
            let raw: Basic = m2(
                &m2(
                    &m2(&d2(&int(2), &sqrt(&Basic::pi())?)?, &diff(&u, x)?)?,
                    &exp(&m2(&int(-sign_), &c)?)?,
                )?,
                &exp(theta_term)?,
            )?;
            cache.set(candidate.clone(), expand_mul(&expand_power_exp(&raw)?)?);
            insert(&mut found, candidate);
        }
    }
    Ok(found)
}

fn exponent_bound(g: &Basic) -> i64 {
    if g.is_pow() {
        if let Some(r) = g.args()[1].as_rational() {
            let (p, q) = (
                i64::try_from(&crate::numbers::numer(r)).unwrap_or(0),
                i64::try_from(&crate::numbers::denom(r)).unwrap_or(1),
            );
            if q != 1 {
                return if p > 0 { p + q - 1 } else { (p + q).abs() };
            }
        }
        return 1;
    }
    if !g.args().is_empty() {
        return g.args().iter().map(exponent_bound).max().unwrap_or(1);
    }
    1
}

fn logarithmic_part(term: &Basic) -> PolyResult<(Basic, Basic)> {
    let has_log = |e: &Basic| has_any(e, &[Function::Log]);
    if term.is_mul() {
        let free: Vec<Basic> = term.args().iter().filter(|a| !has_log(a)).cloned().collect();
        let dependent: Vec<Basic> = term.args().iter().filter(|a| has_log(a)).cloned().collect();
        return Ok((mul_all(&free)?, mul_all(&dependent)?));
    }
    Ok(if has_log(term) {
        (int(1), term.clone())
    } else {
        (term.clone(), int(1))
    })
}

/// The logarithms of complex arguments as real logarithms and arctangents
/// (`_real_forms`).
fn real_forms(f: &Basic) -> R {
    let i: Basic = imaginary();
    if !f.has(&i) {
        return Ok(f.clone());
    }
    let mut logs: Vec<((Basic, Basic), Vec<(Basic, i64)>)> = Vec::new();
    let mut rest: Vec<Basic> = Vec::new();
    for term in crate::core::expr::terms_of(f) {
        let (coefficient, function) = logarithmic_part(&term)?;
        if func(&function) != Some(Function::Log) || !function.args()[0].has(&i) {
            rest.push(term.clone());
            continue;
        }
        let parts = crate::simplify::radsimp::collect_dict(&function.args()[0], std::slice::from_ref(&i), None, None)?;
        let get = |k: &Basic| {
            parts
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| int(0))
        };
        let (p, q) = (get(&int(1)), get(&i));
        if p.has(&i) || q.has(&i) || q == int(0) {
            rest.push(term.clone());
            continue;
        }
        let (key, s) = if crate::core::printing::could_extract_minus_sign(&q) {
            ((p, neg(&q)?), -1)
        } else {
            ((p, q), 1)
        };
        match logs.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1.push((coefficient, s)),
            None => logs.push((key, vec![(coefficient, s)])),
        }
    }
    let mut result: Basic = add_all(&rest)?;
    for ((p, q), entries) in logs {
        let a: Basic = add_all(
            &entries
                .iter()
                .filter(|(_, s)| *s == 1)
                .map(|(c, _)| c.clone())
                .collect::<Vec<_>>(),
        )?;
        let b: Basic = add_all(
            &entries
                .iter()
                .filter(|(_, s)| *s == -1)
                .map(|(c, _)| c.clone())
                .collect::<Vec<_>>(),
        )?;
        let modulus: Basic = log(&a2(&pw(&p, &int(2))?, &pw(&q, &int(2))?)?)?;
        let first: Basic = m2(&d2(&a2(&a, &b)?, &int(2))?, &modulus)?;
        let second: Basic = m2(&m2(&i, &s2(&a, &b)?)?, &apply(Function::Atan, &[d2(&p, &q)?])?)?;
        let combined: Basic = expand(&s2(&first, &second)?)?;
        if combined.has(&i) {
            result = a2(
                &a2(&result, &m2(&a, &log(&a2(&p, &m2(&i, &q)?)?)?)?)?,
                &m2(&b, &log(&s2(&p, &m2(&i, &q)?)?)?)?,
            )?;
        } else {
            result = a2(&result, &combined)?;
        }
    }
    Ok(result)
}

fn timed<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(crate::settings::settings().timeout, f)
}

fn verified(big_f: &Basic, f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let derivative: Basic = diff(big_f, x)?;
    let difference: Basic = s2(&derivative, f)?;
    if timed(|| cancel(&difference))? == Some(int(0)) {
        return Ok(Some(big_f.clone()));
    }
    let options = crate::simplify::simplify::Options::default();
    if timed(|| crate::simplify::simplify::simplify(&difference, &options))? == Some(int(0)) {
        return Ok(Some(big_f.clone()));
    }
    if crate::integrals::conditions::numerically_equal(&derivative, f, &[], 3)? {
        return Ok(Some(big_f.clone()));
    }
    Ok(None)
}

/// An antiderivative of `f` by the heuristic Risch algorithm, verified,
/// or `None` (`heurisch_antiderivative`).
pub fn heurisch_antiderivative(
    f: &Basic,
    x: &Basic,
    hints: &[Basic],
    degree_offset: i64,
    retries: i64,
) -> PolyResult<Option<Basic>> {
    let mut refused: Vec<Function> = REFUSED.to_vec();
    refused.push(Function::Arg);
    // the core has no floats (`f.has(Float)`)
    if has_any(f, &refused) {
        return Ok(None);
    }
    if !f.has(x) {
        return Ok(Some(m2(f, x)?));
    }
    let search = || -> PolyResult<Option<Basic>> {
        let found = heurisch(f, x, false, hints, None, retries, degree_offset, None)?;
        if found.is_none() && degree_offset == 0 {
            return heurisch(f, x, false, hints, None, retries, 1, None);
        }
        Ok(found)
    };
    let Some(Some(result)) = timed(search)? else {
        return Ok(None);
    };
    verified(&real_forms(&result)?, f, x)
}

fn itermonomials(v: &[Basic], bound: i64) -> PolyResult<Vec<Basic>> {
    let mut monomials: Vec<Basic> = Vec::new();
    fn combos(v: &[Basic], start: usize, left: i64, chosen: &mut Vec<Basic>, out: &mut Vec<Basic>) -> PolyResult<()> {
        if left == 0 {
            out.push(crate::core::mul::mul(chosen)?);
            return Ok(());
        }
        for i in start..v.len() {
            chosen.push(v[i].clone());
            combos(v, i, left - 1, chosen, out)?;
            chosen.pop();
        }
        Ok(())
    }
    for degree in 0..=bound {
        combos(v, 0, degree, &mut Vec::new(), &mut monomials)?;
    }
    Ok(sorted(monomials))
}

fn find_non_syms(e: &Basic, syms: &[Basic], non_syms: &mut Vec<Basic>) -> PolyResult<()> {
    if e.is_rational_class() || syms.contains(e) {
        return Ok(());
    }
    if !syms.iter().any(|s| e.has(s)) {
        insert(non_syms, e.clone());
    } else if e.is_add() || e.is_mul() || e.is_pow() {
        for a in e.args() {
            find_non_syms(a, syms, non_syms)?;
        }
    } else {
        return Err(value_error(format!("{} is not a polynomial in the unknowns", e)));
    }
    Ok(())
}

use crate::polys::domains::algebraicfield::{AlgebraicElement, AlgebraicField};
use crate::polys::domains::domain::{Domain, SparseRow};
use crate::polys::domains::sparsepolynomialring::SparseGround;
use crate::polys::numberfields::qqbar::Algebraic;

/// An algebraic field which is not ordered, its elements never negative
/// (`_UnorderedField`: the canonical denominators of the fractions over it
/// keep their sign).
#[derive(Debug, Clone, PartialEq)]
struct UnorderedField(AlgebraicField);

impl Domain for UnorderedField {
    type Elem = AlgebraicElement;
    fn name(&self) -> String {
        self.0.name()
    }
    fn is_field(&self) -> bool {
        true
    }
    fn is_ordered(&self) -> bool {
        false
    }
    fn characteristic(&self) -> crate::numbers::Int {
        self.0.characteristic()
    }
    fn zero(&self) -> AlgebraicElement {
        self.0.zero()
    }
    fn one(&self) -> AlgebraicElement {
        self.0.one()
    }
    fn to_expr(&self, a: &AlgebraicElement) -> R {
        self.0.to_expr(a)
    }
    fn from_expr(&self, e: &Basic) -> PolyResult<AlgebraicElement> {
        self.0.from_expr(e)
    }
    fn is_zero(&self, a: &AlgebraicElement) -> bool {
        self.0.is_zero(a)
    }
    fn is_one(&self, a: &AlgebraicElement) -> bool {
        self.0.is_one(a)
    }
    fn is_negative(&self, _a: &AlgebraicElement) -> bool {
        false
    }
    fn is_unit(&self, a: &AlgebraicElement) -> bool {
        self.0.is_unit(a)
    }
    fn add(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        self.0.add(a, b)
    }
    fn sub(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        self.0.sub(a, b)
    }
    fn neg(&self, a: &AlgebraicElement) -> AlgebraicElement {
        self.0.neg(a)
    }
    fn mul(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        self.0.mul(a, b)
    }
    fn exquo(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        self.0.exquo(a, b)
    }
    fn quo(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        self.0.quo(a, b)
    }
    fn rem(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        self.0.rem(a, b)
    }
    fn invert(&self, a: &AlgebraicElement) -> PolyResult<AlgebraicElement> {
        self.0.invert(a)
    }
    fn gcd(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        self.0.gcd(a, b)
    }
    fn lcm(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
        self.0.lcm(a, b)
    }
    fn gcdex(
        &self,
        a: &AlgebraicElement,
        b: &AlgebraicElement,
    ) -> PolyResult<(AlgebraicElement, AlgebraicElement, AlgebraicElement)> {
        self.0.gcdex(a, b)
    }
    fn canonical_unit(&self, a: &AlgebraicElement) -> AlgebraicElement {
        self.0.canonical_unit(a)
    }
    fn from_int(&self, n: i64) -> AlgebraicElement {
        self.0.from_int(n)
    }
    fn from_integer(&self, n: &crate::numbers::Int) -> AlgebraicElement {
        self.0.from_integer(n)
    }
}

impl SparseGround for UnorderedField {}

const MAX_FIELD_DEGREE: usize = 32;

fn algebraic(atom: &Basic) -> PolyResult<Option<Algebraic>> {
    let value: Option<Algebraic> = if *atom == imaginary() {
        Some(Algebraic::imaginary_unit())
    } else if atom.is_pow() {
        match (atom.args()[0].as_rational(), atom.args()[1].as_rational()) {
            (Some(b), Some(e)) if crate::numbers::denom(e) > crate::numbers::Int::from(1) => {
                let q: usize = usize::try_from(&crate::numbers::denom(e)).unwrap_or(1);
                let p: i64 = i64::try_from(&crate::numbers::numer(e)).unwrap_or(1);
                Some(Algebraic::from_rational(b).nth_root(q)?.pow(p)?)
            }
            _ => None,
        }
    } else {
        None
    };
    let Some(value) = value else { return Ok(None) };
    let Some(numeric) = crate::core::evalf::numeric_value(atom, 100) else {
        return Ok(None);
    };
    let (re, im) = crate::core::evalf::parts(&numeric);
    let (nr, ni) = (re.to_f64().value(), im.to_f64().value());
    let (vr, vi) = value.to_complex()?;
    let size: f64 = (nr * nr + ni * ni).sqrt();
    if ((vr - nr).powi(2) + (vi - ni).powi(2)).sqrt() > 1e-9 * (1.0 + size) {
        return Ok(None);
    }
    Ok(Some(value))
}

fn horner(coefficients: &[crate::numbers::Rat], point: &AlgebraicElement, k: &AlgebraicField) -> AlgebraicElement {
    let mut result: AlgebraicElement = k.zero();
    for c in coefficients {
        result = k.add(&k.mul(&result, point), &k.from_rational(c));
    }
    result
}

/// The field `QQ(theta)` of a list of algebraic constants, each a
/// polynomial in `theta` (`_NumberField`).
struct NumberField {
    theta: Basic,
    field: UnorderedField,
    theta_expr: Basic,
    forms: Vec<(Basic, Basic)>,
}

fn monic(minpoly: &[crate::numbers::Int]) -> Vec<crate::numbers::Rat> {
    let lead: crate::numbers::Rat = crate::numbers::Rat::from(minpoly[0].clone());
    minpoly
        .iter()
        .map(|c| crate::numbers::Rat::from(c.clone()) / lead.clone())
        .collect()
}

impl NumberField {
    fn new(atoms: &[Basic], values: &[Algebraic]) -> PolyResult<NumberField> {
        let theta: Basic = crate::core::symbol::Symbol::new("_theta", &[]).0;
        let mut gamma: Algebraic = values[0].clone();
        let mut gamma_expr: Basic = atoms[0].clone();
        let one = crate::numbers::Rat::from(crate::numbers::Int::from(1));
        let zero = crate::numbers::Rat::from(crate::numbers::Int::from(0));
        let mut coordinates: Vec<AlgebraicElement> = vec![vec![one, zero]];
        for (atom, value) in atoms[1..].iter().zip(values[1..].iter()) {
            let (new, p, q) = crate::polys::numberfields::qqbar::primitive_element(&gamma, value)?;
            if new.degree() > MAX_FIELD_DEGREE {
                return Err(PolyError::Arithmetic(
                    "the field of the constants is too large".to_string(),
                ));
            }
            let k: AlgebraicField = AlgebraicField::new(monic(&new.minpoly()), theta.clone())?;
            let gamma_in_new: AlgebraicElement = horner(&p, &k.alpha(), &k);
            let mut next: Vec<AlgebraicElement> = coordinates.iter().map(|c| horner(c, &gamma_in_new, &k)).collect();
            next.push(horner(&q, &k.alpha(), &k));
            coordinates = next;
            let limit: usize = 4 * gamma.degree() * value.degree() + 2;
            let mut found: Option<i64> = None;
            for c in 1..limit as i64 {
                let shifted: Algebraic = gamma.add(
                    &Algebraic::from_rational(&crate::numbers::Rat::from(crate::numbers::Int::from(c))).mul(value)?,
                )?;
                if shifted.equals(&new)? {
                    found = Some(c);
                    break;
                }
            }
            let c: i64 = found.ok_or_else(|| value_error("no primitive combination".to_string()))?;
            gamma = new;
            gamma_expr = a2(&gamma_expr, &m2(&int(c), atom)?)?;
        }
        let field: UnorderedField = UnorderedField(AlgebraicField::new(monic(&gamma.minpoly()), theta.clone())?);
        let forms: Vec<(Basic, Basic)> = atoms
            .iter()
            .zip(coordinates.iter())
            .map(|(a, e)| Ok((a.clone(), field.to_expr(e)?)))
            .collect::<PolyResult<_>>()?;
        Ok(NumberField {
            theta,
            field,
            theta_expr: gamma_expr,
            forms,
        })
    }
}

type Monomial = Vec<u32>;
type Terms<T> = std::collections::BTreeMap<Monomial, T>;

/// The linear system of the coefficients of the polynomial in the
/// unknowns `v`, linear in the `c` (`_LinearSystem`).
struct LinearSystem {
    polynomial: Basic,
    v_names: Vec<String>,
    c_names: Vec<String>,
    p_names: Vec<String>,
}

impl LinearSystem {
    fn solve<G: Domain, K: Domain>(
        &self,
        g: &G,
        k: &K,
        lift: &dyn Fn(Terms<G::Elem>) -> PolyResult<K::Elem>,
    ) -> PolyResult<Option<Vec<(usize, Basic)>>> {
        let (n, m) = (self.v_names.len(), self.c_names.len());
        let mut names: Vec<String> = self.v_names.clone();
        names.extend(self.c_names.iter().cloned());
        names.extend(self.p_names.iter().cloned());
        let terms = crate::polys::polyutils::dict_from_expr(&self.polynomial, &names, g)?;
        let mut grouped: Vec<(Monomial, Vec<(usize, Terms<G::Elem>)>)> = Vec::new();
        for (monomial, value) in terms {
            let unknown: &[u32] = &monomial[n..n + m];
            let degree: u32 = unknown.iter().sum();
            if degree > 1 {
                return Err(value_error("the system is not linear".to_string()));
            }
            let column: usize = if degree == 1 {
                unknown.iter().position(|e| *e == 1).expect("one")
            } else {
                m
            };
            let key: Monomial = monomial[..n].to_vec();
            let rest: Monomial = monomial[n + m..].to_vec();
            let entry = match grouped.iter_mut().position(|(k_, _)| *k_ == key) {
                Some(i) => &mut grouped[i].1,
                None => {
                    grouped.push((key, Vec::new()));
                    let last = grouped.len() - 1;
                    &mut grouped[last].1
                }
            };
            match entry.iter_mut().find(|(c, _)| *c == column) {
                Some((_, parts)) => {
                    parts.insert(rest, value);
                }
                None => {
                    let mut parts: Terms<G::Elem> = Terms::new();
                    parts.insert(rest, value);
                    entry.push((column, parts));
                }
            }
        }
        let mut rows: Vec<SparseRow<K::Elem>> = Vec::new();
        for (_, entries) in grouped {
            let mut row: SparseRow<K::Elem> = SparseRow::new();
            for (column, parts) in entries {
                let value = lift(parts)?;
                if !k.is_zero(&value) {
                    row.insert(column, value);
                }
            }
            if !row.is_empty() {
                rows.push(row);
            }
        }
        let mut solution: Vec<(usize, Basic)> = Vec::new();
        for row in k.sparse_echelon(&rows) {
            let lead: usize = *row.keys().next().expect("a nonzero row");
            if lead == m {
                return Ok(None);
            }
            let value = match row.get(&m) {
                Some(v) => k.neg(v),
                None => k.zero(),
            };
            solution.push((lead, k.to_expr(&value)?));
        }
        Ok(Some(solution))
    }
}

fn solve_coefficients(
    numerator: &Basic,
    v: &[Basic],
    coefficients: &[Basic],
    non_syms: &[Basic],
) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    use crate::polys::domains::rationalfield::QQ;
    use crate::polys::domains::sparsefractionfield::SparseFractionField;
    use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;
    let mut gens: Vec<Basic> = v.to_vec();
    gens.extend(coefficients.iter().cloned());
    let mut placeholders: Placeholders = Placeholders::new(&gens, true);
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for c in non_syms {
        rule.push((c.clone(), placeholders.abstract_(c)?));
    }
    let mut abstracted: Basic = replaced(numerator, &rule)?;
    let mut atoms: Vec<Basic> = Vec::new();
    let mut values: Vec<Algebraic> = Vec::new();
    for (placeholder, constant) in &placeholders.back {
        if let Some(value) = algebraic(constant)? {
            atoms.push(placeholder.clone());
            values.push(value);
        }
    }
    let constant_of = |a: &Basic| {
        placeholders
            .back
            .iter()
            .find(|(k, _)| k == a)
            .map(|(_, c)| c.clone())
            .expect("a placeholder")
    };
    let number_field: Option<NumberField> = if atoms.is_empty() {
        None
    } else {
        match NumberField::new(&atoms.iter().map(constant_of).collect::<Vec<_>>(), &values) {
            Ok(nf) => Some(nf),
            Err(PolyError::Arithmetic(_)) => None,
            Err(e) => return Err(e),
        }
    };
    if let Some(nf) = &number_field {
        let forms: Vec<(Basic, Basic)> = atoms
            .iter()
            .map(|a| {
                let c: Basic = constant_of(a);
                (
                    a.clone(),
                    nf.forms
                        .iter()
                        .find(|(k, _)| *k == c)
                        .map(|(_, f)| f.clone())
                        .expect("a form"),
                )
            })
            .collect();
        abstracted = replaced(&abstracted, &forms)?;
    }
    let mut unknown_set: Vec<Basic> = gens.clone();
    if let Some(nf) = &number_field {
        unknown_set.push(nf.theta.clone());
    }
    let mut params: Vec<Basic> = abstracted
        .free_symbols()
        .into_iter()
        .filter(|s| s.is_symbol() && !unknown_set.contains(s))
        .collect();
    params.sort_by_cached_key(crate::core::sorting::sort_key);
    let v_names: Vec<String> = (0..v.len()).map(|i| format!("_v{}", i)).collect();
    let c_names: Vec<String> = (0..coefficients.len()).map(|i| format!("_c{}", i)).collect();
    let p_names: Vec<String> = (0..params.len()).map(|i| format!("_p{}", i)).collect();
    let symbol = |name: &str| crate::core::symbol::Symbol::new(name, &[]).0;
    let mut renamed: Vec<(Basic, Basic)> = Vec::new();
    for (s, name) in v
        .iter()
        .chain(coefficients.iter())
        .chain(params.iter())
        .zip(v_names.iter().chain(c_names.iter()).chain(p_names.iter()))
    {
        renamed.push((s.clone(), symbol(name)));
    }
    abstracted = replaced(&abstracted, &renamed)?;
    if !params.is_empty()
        && abstracted.preorder().iter().any(|n| {
            n.is_pow()
                && n.args()[1]
                    .as_rational()
                    .is_some_and(|r| r.is_int() && *r < crate::numbers::Rat::from(crate::numbers::Int::from(0)))
        })
    {
        abstracted = crate::core::expr::as_numer_denom(&abstracted)?.0;
    }
    let mut restored: Vec<(Basic, Basic)> = params
        .iter()
        .zip(p_names.iter())
        .map(|(p, name)| (symbol(name), p.clone()))
        .collect();
    if let Some(nf) = &number_field {
        restored.push((nf.theta.clone(), nf.theta_expr.clone()));
    }
    let system = LinearSystem {
        polynomial: abstracted,
        v_names,
        c_names,
        p_names: p_names.clone(),
    };
    let empty: Monomial = Vec::new();
    let found: Option<Vec<(usize, Basic)>> = match &number_field {
        Some(nf) => {
            let g: &UnorderedField = &nf.field;
            if params.is_empty() {
                system.solve(g, g, &|terms| {
                    Ok(terms.get(&empty).cloned().unwrap_or_else(|| g.zero()))
                })?
            } else {
                let ring = SparsePolynomialRing::new(g.clone(), p_names.clone());
                let k = SparseFractionField::new(ring.clone());
                system.solve(g, &k, &|terms| Ok(k.from_ring(&ring.ring.from_dict(terms))))?
            }
        }
        None => {
            if params.is_empty() {
                system.solve(&QQ, &QQ, &|terms| {
                    Ok(terms.get(&empty).cloned().unwrap_or_else(|| QQ.zero()))
                })?
            } else {
                let ring = SparsePolynomialRing::sympy(QQ, p_names.clone());
                let k = SparseFractionField::new(ring.clone());
                system.solve(&QQ, &k, &|terms| Ok(k.from_ring(&ring.ring.from_dict(terms))))?
            }
        }
    };
    let Some(found) = found else { return Ok(None) };
    let mut out: Vec<(Basic, Basic)> = Vec::new();
    for (column, value) in found {
        out.push((
            coefficients[column].clone(),
            placeholders.restore(&replaced(&value, &restored)?)?,
        ));
    }
    Ok(Some(out))
}

fn tangent_rule(node: &Basic) -> R {
    let Some(f) = func(node) else { return Ok(node.clone()) };
    if !matches!(
        f,
        Function::Sin | Function::Cos | Function::Cot | Function::Sinh | Function::Cosh | Function::Coth
    ) {
        return Ok(node.clone());
    }
    let half_argument: Basic = d2(&node.args()[0], &int(2))?;
    Ok(match f {
        Function::Sin => {
            let half: Basic = apply(Function::Tan, std::slice::from_ref(&half_argument))?;
            d2(&m2(&int(2), &half)?, &a2(&int(1), &pw(&half, &int(2))?)?)?
        }
        Function::Cos => {
            let half: Basic = pw(&apply(Function::Tan, std::slice::from_ref(&half_argument))?, &int(2))?;
            d2(&s2(&int(1), &half)?, &a2(&int(1), &half)?)?
        }
        Function::Cot => d2(&int(1), &apply(Function::Tan, &[node.args()[0].clone()])?)?,
        Function::Sinh => {
            let half: Basic = apply(Function::Tanh, std::slice::from_ref(&half_argument))?;
            d2(&m2(&int(2), &half)?, &s2(&int(1), &pw(&half, &int(2))?)?)?
        }
        Function::Cosh => {
            let half: Basic = pw(&apply(Function::Tanh, std::slice::from_ref(&half_argument))?, &int(2))?;
            d2(&a2(&int(1), &half)?, &s2(&int(1), &half)?)?
        }
        _ => d2(&int(1), &apply(Function::Tanh, &[node.args()[0].clone()])?)?,
    })
}

fn half_angle_tangents(f: &Basic) -> R {
    if f.args().is_empty() {
        return Ok(f.clone());
    }
    let args: Vec<Basic> = f.args().iter().map(half_angle_tangents).collect::<PolyResult<_>>()?;
    let rebuilt: Basic = crate::core::function::rebuild_evaluated(f, args)?;
    tangent_rule(&rebuilt)
}

type Pair = (Basic, Basic);

fn size(e: &Basic) -> usize {
    e.preorder().len()
}

fn ordered_pairs(group: &[Pair]) -> Vec<Pair> {
    let mut v: Vec<Pair> = group.to_vec();
    v.sort_by_cached_key(|(t, s)| {
        (
            size(t),
            crate::core::sorting::sort_key(t),
            crate::core::sorting::sort_key(s),
        )
    });
    v
}

/// The permutations of the groups in the order of `itertools.permutations`
/// (lexicographic in the indices), consumed across the retries.
struct Mappings {
    groups: Vec<Vec<Pair>>,
    next: Option<Vec<usize>>,
}

impl Mappings {
    fn new(groups: Vec<Vec<Pair>>) -> Mappings {
        let n: usize = groups.len();
        Mappings {
            groups,
            next: Some((0..n).collect()),
        }
    }

    fn take(&mut self) -> Option<Vec<Pair>> {
        let current: Vec<usize> = self.next.take()?;
        let mut p: Vec<usize> = current.clone();
        // the next permutation in lexicographic order
        let n: usize = p.len();
        let mut advanced: bool = false;
        if n > 1 {
            let mut i: usize = n - 1;
            while i > 0 && p[i - 1] >= p[i] {
                i -= 1;
            }
            if i > 0 {
                let mut j: usize = n - 1;
                while p[j] <= p[i - 1] {
                    j -= 1;
                }
                p.swap(i - 1, j);
                p[i..].reverse();
                advanced = true;
            }
        }
        self.next = if advanced { Some(p) } else { None };
        Some(current.iter().flat_map(|k| ordered_pairs(&self.groups[*k])).collect())
    }
}

fn heurisch(
    f: &Basic,
    x: &Basic,
    mut rewrite: bool,
    hints: &[Basic],
    mappings: Option<&mut Mappings>,
    retries: i64,
    degree_offset: i64,
    unnecessary: Option<Vec<Pair>>,
) -> PolyResult<Option<Basic>> {
    if !f.has(x) {
        return Ok(Some(m2(f, x)?));
    }
    let mut f: Basic = expand_power_exp(f)?;
    let mut indep: Basic = int(1);
    if !f.is_add() {
        let (i, rest) = crate::core::expr::as_independent(&f, x, false)?;
        indep = i;
        f = rest;
    }
    let trig = [Function::Sin, Function::Cos, Function::Cot];
    let hyper = [Function::Sinh, Function::Cosh, Function::Coth];
    if rewrite {
        f = half_angle_tangents(&f)?;
    } else if !has_any(&f, &trig) && !has_any(&f, &hyper) {
        rewrite = true;
    }
    let mut terms: Vec<Basic> = components(&f, x)?;
    let mut cache: DiffCache = DiffCache {
        cache: Vec::new(),
        x: x.clone(),
    };
    let special = special_candidates(&terms, x, &mut cache, &f)?;
    extend(&mut terms, special);
    let tower = tower_candidates(&terms, x, &mut cache, &f)?;
    extend(&mut terms, tower);
    extend(&mut terms, hints.to_vec());
    for g in terms.clone() {
        let d: Basic = cache.get(&g)?;
        extend(&mut terms, components(&d, x)?);
    }
    let terms: Vec<Basic> = exponential_bases(&terms)?;
    let ordered_terms: Vec<Basic> = sorted(terms);
    let v: Vec<Basic> = symbols("x", ordered_terms.len());
    let mut keyed: Vec<Pair> = ordered_terms.iter().cloned().zip(v.iter().cloned()).collect();
    let key = |t: &Basic| -> (i64, String) {
        let rest: Basic = crate::core::expr::as_independent(t, x, false)
            .map(|(_, r)| r)
            .unwrap_or_else(|_| t.clone());
        (ops(&rest), t.to_string())
    };
    keyed.sort_by_cached_key(|p| std::cmp::Reverse(key(&p.0)));
    let mut mapping: Vec<Pair> = keyed;
    let rev_mapping: Vec<(Basic, Basic)> = mapping.iter().map(|(t, s)| (s.clone(), t.clone())).collect();
    let mut own: Option<Mappings> = None;
    let unnecessary: Vec<Pair> = match &mappings {
        None => {
            let position: usize = mapping
                .iter()
                .position(|(t, _)| t == x)
                .ok_or_else(|| value_error("the variable is not a term".to_string()))?;
            let removed: Vec<Pair> = vec![mapping.remove(position)];
            let mut types: Vec<(String, Vec<Pair>)> = Vec::new();
            for pair in mapping.iter() {
                let name: String = match pair.0.head() {
                    Head::Function(g) => format!("{:?}", g),
                    other => other.class_name().to_string(),
                };
                match types.iter_mut().find(|(k, _)| *k == name) {
                    Some(entry) => entry.1.push(pair.clone()),
                    None => types.push((name, vec![pair.clone()])),
                }
            }
            own = Some(Mappings::new(types.into_iter().map(|(_, g)| g).collect()));
            removed
        }
        Some(_) => unnecessary.unwrap_or_default(),
    };
    let iterator: &mut Mappings = match mappings {
        Some(m) => m,
        None => own.as_mut().expect("the mappings"),
    };
    let substitute = |current: &[Pair], e: &Basic| -> R {
        let mut largest_first: Vec<Pair> = current.to_vec();
        largest_first.sort_by_cached_key(|(t, _)| (-ops(t), t.to_string()));
        crate::core::subs::subs_pairs(e, &largest_first)
    };
    let mut diffs: Vec<Basic> = Vec::new();
    let mut denom: Option<Basic> = None;
    let mut current: Vec<Pair> = Vec::new();
    while let Some(candidate) = iterator.take() {
        current = candidate;
        current.extend(unnecessary.iter().cloned());
        diffs = Vec::new();
        for g in &ordered_terms {
            let d: Basic = cache.get(g)?;
            diffs.push(substitute(&current, &d)?);
        }
        let denoms: Vec<Basic> = diffs
            .iter()
            .map(|g| crate::core::expr::as_numer_denom(g).map(|(_, d)| d))
            .collect::<PolyResult<_>>()?;
        let substituted: Basic = substitute(&current, &f)?;
        if denoms.iter().all(|h| is_polynomial_in(h, &v))
            && crate::core::expr::polynomial_like(&substituted, &v, crate::core::expr::PolynomialKind::Rational)
                == Some(true)
        {
            let mut common: Basic = denoms[0].clone();
            for q in &denoms[1..] {
                common = match lcm_of(&common, q, &v) {
                    Ok(c) => c,
                    Err(e) if gives_up(&e) => return Ok(None),
                    Err(e) => return Err(e),
                };
            }
            denom = Some(common);
            break;
        }
    }
    let Some(denom) = denom else {
        if !rewrite {
            if let Some(result) = heurisch(&f, x, true, hints, None, retries, degree_offset, Some(unnecessary))? {
                return Ok(Some(m2(&indep, &result)?));
            }
        }
        return Ok(None);
    };
    let found = match solve(
        &f,
        x,
        &denom,
        &diffs,
        &v,
        &ordered_terms,
        &|e| substitute(&current, e),
        degree_offset,
    ) {
        Ok(found) => found,
        Err(e) if gives_up(&e) => None,
        Err(e) => return Err(e),
    };
    if let Some(found) = found {
        let mut antiderivative: Basic = expand(&cancel(&replaced(&found, &rev_mapping)?)?)?;
        if antiderivative.is_add() {
            antiderivative = crate::core::expr::as_independent(&antiderivative, x, true)?.1;
        }
        return Ok(Some(m2(&indep, &antiderivative)?));
    }
    if retries >= 0 {
        if let Some(result) = heurisch(
            &f,
            x,
            rewrite,
            hints,
            Some(iterator),
            retries - 1,
            degree_offset,
            Some(unnecessary),
        )? {
            return Ok(Some(m2(&indep, &result)?));
        }
    }
    Ok(None)
}

fn solve(
    f: &Basic,
    _x: &Basic,
    denom: &Basic,
    diffs: &[Basic],
    v: &[Basic],
    ordered_terms: &[Basic],
    substitute: &dyn Fn(&Basic) -> R,
    degree_offset: i64,
) -> PolyResult<Option<Basic>> {
    let numers: Vec<Basic> = diffs
        .iter()
        .map(|g| cancel(&m2(denom, g)?))
        .collect::<PolyResult<_>>()?;
    let derivation = |h: &Basic| -> R {
        let terms: Vec<Basic> = numers
            .iter()
            .zip(v.iter())
            .map(|(d, y)| m2(d, &diff(h, y)?))
            .collect::<PolyResult<_>>()?;
        add_all(&terms)
    };
    fn deflation(p: &Basic, v: &[Basic], derivation: &dyn Fn(&Basic) -> R) -> R {
        for y in v {
            if !p.has(y) {
                continue;
            }
            if derivation(p)? != int(0) {
                let (c, q) = primitive(p, y)?;
                return m2(
                    &deflation(&c, v, derivation)?,
                    &gcd_of(&q, &diff(&q, y)?, std::slice::from_ref(y))?,
                );
            }
        }
        Ok(p.clone())
    }
    fn splitter(p: &Basic, v: &[Basic], derivation: &dyn Fn(&Basic) -> R) -> PolyResult<(Basic, Basic)> {
        for y in v {
            if !p.has(y) {
                continue;
            }
            if derivation(y)? != int(0) {
                let (c, q) = primitive(p, y)?;
                let h: Basic = gcd_of(&q, &derivation(&q)?, std::slice::from_ref(y))?;
                let s: Basic = cancel(&d2(&h, &gcd_of(&q, &diff(&q, y)?, std::slice::from_ref(y))?)?)?;
                let c_split = splitter(&c, v, derivation)?;
                if degree(&s, y)? == 0 {
                    return Ok((c_split.0, m2(&q, &c_split.1)?));
                }
                let q_split = splitter(&cancel(&d2(&q, &s)?)?, v, derivation)?;
                return Ok((m2(&m2(&c_split.0, &q_split.0)?, &s)?, m2(&c_split.1, &q_split.1)?));
            }
        }
        Ok((int(1), p.clone()))
    }
    let mut special: Vec<(Basic, bool)> = Vec::new();
    let mut put = |k: Basic, value: bool| match special.iter_mut().find(|(e, _)| *e == k) {
        Some(entry) => entry.1 = value,
        None => special.push((k, value)),
    };
    for term in ordered_terms {
        match func(term) {
            Some(Function::Tan) => put(a2(&int(1), &pw(&substitute(term)?, &int(2))?)?, false),
            Some(Function::Tanh) => {
                put(a2(&int(1), &substitute(term)?)?, false);
                put(s2(&int(1), &substitute(term)?)?, false);
            }
            Some(Function::LambertW) => put(substitute(term)?, true),
            _ => {}
        }
    }
    let big_f: Basic = substitute(f)?;
    let (p, q) = crate::core::expr::as_numer_denom(&big_f)?;
    let u_split = splitter(denom, v, &derivation)?;
    let v_split = splitter(&q, v, &derivation)?;
    let mut polys: Vec<Basic> = vec![v_split.0.clone(), v_split.1.clone()];
    insert(&mut polys, u_split.0.clone());
    for (k, _) in &special {
        insert(&mut polys, k.clone());
    }
    let mut s_factors: Vec<Basic> = vec![u_split.0.clone()];
    s_factors.extend(special.iter().filter(|(_, value)| *value).map(|(k, _)| k.clone()));
    let s: Basic = mul_all(&s_factors)?;
    let (Some(a), Some(b), Some(c)) = (total_degree(&s, v)?, total_degree(&p, v)?, total_degree(&q, v)?) else {
        return Ok(None);
    };
    let poly_denom: Basic = m2(&m2(&s, &v_split.0)?, &deflation(&v_split.1, v, &derivation)?)?;
    let (big_a, big_b) = (exponent_bound(f), a + b.max(c));
    let bound: i64 = if big_a > 1 && big_b > 1 {
        big_a + big_b - 1 + degree_offset
    } else {
        big_a + big_b + degree_offset
    };
    let monoms: Vec<Basic> = itermonomials(v, bound)?;
    let poly_coeffs: Vec<Basic> = symbols("A", monoms.len());
    let poly_part: Basic = add_all(
        &poly_coeffs
            .iter()
            .zip(monoms.iter())
            .map(|(c, m)| m2(c, m))
            .collect::<PolyResult<Vec<_>>>()?,
    )?;
    let mut reducibles: Vec<Basic> = Vec::new();
    for poly in sorted(polys) {
        let (coefficient, factors) = factor_list_of(&poly, v)?;
        insert(&mut reducibles, coefficient);
        for (factor, _) in factors {
            insert(&mut reducibles, factor);
        }
    }
    let i: Basic = imaginary();
    let integrate = |field: Option<&str>| -> PolyResult<Option<Basic>> {
        let mut atans: Vec<Basic> = Vec::new();
        let mut pairs: Vec<(Basic, Basic)> = Vec::new();
        let mut irreducibles: Vec<Basic> = Vec::new();
        if field == Some("Q") {
            irreducibles = reducibles.clone();
        } else {
            for poly in sorted(reducibles.clone()) {
                let free: Vec<Basic> = crate::core::traversal::iterfreeargs(&poly);
                let z_v: Vec<Basic> = v.iter().filter(|y| free.contains(y)).cloned().collect();
                if let Some(z) = sorted(z_v).first() {
                    extend(&mut irreducibles, root_factors(&poly, z, field)?);
                }
            }
        }
        let mut log_part: Vec<Basic> = Vec::new();
        let mut atan_part: Vec<Basic> = Vec::new();
        for poly in sorted(irreducibles.clone()) {
            let parts = crate::simplify::radsimp::collect_dict(&poly, std::slice::from_ref(&i), None, None)?;
            let get = |k: &Basic| {
                parts
                    .iter()
                    .find(|(key, _)| key == k)
                    .map(|(_, value)| value.clone())
                    .unwrap_or_else(|| int(0))
            };
            let y: Basic = get(&i);
            if y != int(0) {
                let x_: Basic = get(&int(1));
                if x_.has(&i) || y.has(&i) {
                    continue;
                }
                if !pairs.contains(&(x_.clone(), y.clone())) {
                    pairs.push((x_, y));
                }
                irreducibles.retain(|e| *e != poly);
            }
        }
        while let Some((x_, mut y)) = pairs.pop() {
            let conjugate: (Basic, Basic) = (x_.clone(), neg(&y)?);
            if let Some(position) = pairs.iter().position(|pq| *pq == conjugate) {
                pairs.remove(position);
                if crate::core::printing::could_extract_minus_sign(&y) {
                    y = neg(&y)?;
                }
                insert(&mut irreducibles, a2(&m2(&x_, &x_)?, &m2(&y, &y)?)?);
                insert(&mut atans, apply(Function::Atan, &[d2(&x_, &y)?])?);
            } else {
                insert(&mut irreducibles, a2(&x_, &m2(&i, &y)?)?);
            }
        }
        let mut coefficients: Vec<Basic> = poly_coeffs.clone();
        let b_: Vec<Basic> = symbols("B", irreducibles.len());
        let c_: Vec<Basic> = symbols("C", atans.len());
        let sorted_irreducibles: Vec<Basic> = sorted(irreducibles.clone());
        for (poly, coefficient) in sorted_irreducibles.iter().zip(b_.iter()).rev() {
            if v.iter().any(|y| poly.has(y)) {
                coefficients.push(coefficient.clone());
                log_part.push(m2(coefficient, &log(poly)?)?);
            }
        }
        let sorted_atans: Vec<Basic> = sorted(atans.clone());
        for (poly, coefficient) in sorted_atans.iter().zip(c_.iter()).rev() {
            if v.iter().any(|y| poly.has(y)) {
                coefficients.push(coefficient.clone());
                atan_part.push(m2(coefficient, poly)?);
            }
        }
        let candidate: Basic = a2(
            &a2(&d2(&poly_part, &poly_denom)?, &add_all(&log_part)?)?,
            &add_all(&atan_part)?,
        )?;
        let h: Basic = s2(&big_f, &d2(&derivation(&candidate)?, denom)?)?;
        let raw_numer: Basic = crate::core::expr::as_numer_denom(&h)?.0;
        let mut syms: Vec<Basic> = coefficients.clone();
        extend(&mut syms, v.to_vec());
        let mut non_syms: Vec<Basic> = Vec::new();
        find_non_syms(&raw_numer, &syms, &mut non_syms)?;
        let Some(Some(solution)) = timed(|| solve_coefficients(&raw_numer, v, &coefficients, &non_syms))? else {
            return Ok(None);
        };
        let zeros: Vec<(Basic, Basic)> = coefficients.iter().map(|c| (c.clone(), int(0))).collect();
        Ok(Some(replaced(&replaced(&candidate, &solution)?, &zeros)?))
    };
    let more_free: bool = big_f.free_symbols().iter().any(|s| s.is_symbol() && !v.contains(s));
    if !more_free {
        if let Some(found) = integrate(Some("Q"))? {
            return Ok(Some(found));
        }
    }
    integrate(None)
}

fn denominators(e: &Basic) -> PolyResult<Vec<Basic>> {
    let mut found: Vec<Basic> = Vec::new();
    for node in e.preorder() {
        if matches!(node.head(), Head::Tuple) || crate::sets::sets::is_set(&node) {
            continue;
        }
        let Ok((_, d)) = crate::core::expr::as_numer_denom(&node) else {
            continue;
        };
        if d == int(1) {
            continue;
        }
        let factors: Vec<Basic> = if d.is_mul() { d.args().to_vec() } else { vec![d.clone()] };
        for factor in factors {
            insert(&mut found, factor);
        }
    }
    Ok(found)
}

fn consistent(symbol: &Basic, value: &Basic) -> bool {
    let Some(facts) = symbol.symbol_facts() else {
        return true;
    };
    facts
        .iter()
        .all(|(key, flag)| *key == Predicate::commutative || is(value, *key).is_none_or_eq(*flag))
}

trait Flag {
    fn is_none_or_eq(&self, flag: bool) -> bool;
}

impl Flag for Option<bool> {
    fn is_none_or_eq(&self, flag: bool) -> bool {
        self.map_or(true, |b| b == flag)
    }
}

fn polynomial_roots(d: &Basic, s: &Basic) -> PolyResult<Vec<Basic>> {
    let (placeholders, v) = with_constants(std::slice::from_ref(s), std::slice::from_ref(d))?;
    let found = match crate::polys::polyroots::roots(&v[0], std::slice::from_ref(s), true, true, None) {
        Ok(found) => found,
        Err(e) if gives_up(&e) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    sorted(found.iter().map(|(r, _)| r.clone()).collect())
        .iter()
        .map(|r| placeholders.restore(r))
        .collect()
}

fn solutions_of(d: &Basic, exclude: &Basic) -> PolyResult<Vec<Vec<(Basic, Basic)>>> {
    let mut symbols_: Vec<Basic> = d
        .free_symbols()
        .into_iter()
        .filter(|s| s.is_symbol() && s != exclude)
        .collect();
    symbols_.sort_by_cached_key(crate::core::sorting::sort_key);
    let mut result: Vec<Vec<(Basic, Basic)>> = Vec::new();
    let mut got: Vec<Basic> = Vec::new();
    let mut nonlinear: Vec<Basic> = Vec::new();
    for s in &symbols_ {
        let coefficients = power_coefficients(d, s)?;
        let top: i64 = coefficients
            .as_ref()
            .and_then(|c| c.iter().map(|(k, _)| *k).max())
            .unwrap_or(0);
        match &coefficients {
            Some(c) if top == 1 && c.iter().any(|(k, _)| *k == 1) => {
                let get = |k: i64| {
                    c.iter()
                        .find(|(d_, _)| *d_ == k)
                        .map(|(_, v)| v.clone())
                        .unwrap_or_else(|| int(0))
                };
                let value: Basic = cancel(&d2(&neg(&get(0))?, &get(1))?)?;
                if value.free_symbols().iter().any(|t| got.contains(t)) {
                    continue;
                }
                if consistent(s, &value) {
                    got.push(s.clone());
                    result.push(vec![(s.clone(), value)]);
                }
            }
            Some(_) if top > 1 => nonlinear.push(s.clone()),
            _ => {}
        }
    }
    for s in nonlinear {
        let found: Vec<Basic> = polynomial_roots(d, &s)?;
        for value in &found {
            if value.free_symbols().iter().any(|t| got.contains(t)) || !consistent(&s, value) {
                continue;
            }
            result.push(vec![(s.clone(), value.clone())]);
        }
        if !found.is_empty() {
            got.push(s.clone());
        }
    }
    Ok(result)
}

fn joint(cases: &[Vec<(Basic, Basic)>]) -> Vec<Vec<(Basic, Basic)>> {
    let mut together: Vec<(Basic, Basic)> = Vec::new();
    for case in cases {
        for (key, value) in case {
            if together.iter().any(|(k, _)| k == key)
                || value
                    .free_symbols()
                    .iter()
                    .any(|s| s.is_symbol() && together.iter().any(|(k, _)| k == s))
            {
                return Vec::new();
            }
            together.push((key.clone(), value.clone()));
        }
    }
    if together.iter().any(|(_, value)| {
        value
            .free_symbols()
            .iter()
            .any(|s| s.is_symbol() && together.iter().any(|(k, _)| k == s))
    }) {
        return Vec::new();
    }
    vec![together]
}

/// The antiderivative with the special values of the parameters at which
/// the generic one is undefined as `Piecewise` cases (`heurisch_cases`).
pub fn heurisch_cases(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    if !f.has(x) {
        return Ok(Some(m2(f, x)?));
    }
    let Some(generic) = heurisch_antiderivative(f, x, &[], 0, 2)? else {
        return Ok(None);
    };
    let mut cases: Vec<Vec<(Basic, Basic)>> = Vec::new();
    for d in sorted(denominators(&generic)?) {
        if let Some(found) = timed(|| solutions_of(&d, x))? {
            cases.extend(found);
        }
    }
    if cases.is_empty() {
        return Ok(Some(generic));
    }
    let mut unique: Vec<Vec<(Basic, Basic)>> = Vec::new();
    for case in cases {
        if !unique.contains(&case) {
            unique.push(case);
        }
    }
    let mut original: Vec<Vec<(Basic, Basic)>> = Vec::new();
    for d in denominators(f)? {
        if let Some(found) = timed(|| solutions_of(&d, x))? {
            original.extend(found);
        }
    }
    let mut cases: Vec<Vec<(Basic, Basic)>> = unique.into_iter().filter(|c| !original.contains(c)).collect();
    if cases.is_empty() {
        return Ok(Some(generic));
    }
    if cases.len() > 1 {
        let mut all: Vec<Vec<(Basic, Basic)>> = joint(&cases);
        all.extend(cases);
        cases = all;
    }
    let mut pairs: Vec<(Basic, Basic)> = Vec::new();
    for case in &cases {
        let Some(special) = heurisch_antiderivative(&replaced(f, case)?, x, &[], 0, 2)? else {
            return Ok(None);
        };
        let condition: Basic = crate::core::logic::and_(
            &case
                .iter()
                .map(|(k, v)| crate::core::relational::eq(k, v))
                .collect::<PolyResult<Vec<_>>>()?,
        );
        pairs.push((special, condition));
    }
    let piecewise = |p: &[(Basic, Basic)]| crate::functions::elementary::piecewise::piecewise(p);
    if pairs.len() == 1 {
        let generic_condition: Basic = crate::core::logic::or_(
            &cases[0]
                .iter()
                .map(|(k, v)| crate::core::relational::ne(k, v))
                .collect::<PolyResult<Vec<_>>>()?,
        );
        return Ok(Some(piecewise(&[
            (generic, generic_condition),
            (pairs[0].0.clone(), crate::core::logic::boolean(true)),
        ])?));
    }
    pairs.push((generic, crate::core::logic::boolean(true)));
    Ok(Some(piecewise(&pairs)?))
}
