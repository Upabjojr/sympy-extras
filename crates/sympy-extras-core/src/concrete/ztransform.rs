//! Port of `sympy_extras/concrete/ztransform.py`: the unilateral
//! Z-transform `sum_{n >= 0} f(n) z**-n` of sums of products of
//! polynomials, geometric and exponential factors, sines and cosines of
//! linear arguments, `1/n!`, `1/(n + m)`, unit steps and impulses; the
//! inverse transform of rational functions by partial fractions over
//! `K[t]/(factor)` (the coefficients polynomials in the reciprocal pole),
//! in real form for the quadratic factors with complex roots.

use crate::core::basic::{Basic, Function, Head};
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::{Field, FunctionEntries, RationalEntries};
use crate::matrices::linalg::{kernels as kernels_of, normal, PolynomialLift};
use crate::numbers::{numer, Int, Rat};
use crate::polys::densearith::{dup_add, dup_div, dup_mul, dup_mul_ground, dup_pow, dup_rem};
use crate::polys::densebasic::dup_strip;
use crate::polys::domains::domain::Domain;
use crate::polys::domains::quotientfield::QuotientField;
use crate::polys::euclidtools::dup_invert;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn number(r: Rat) -> Basic {
    Basic::from_rational(r)
}
fn q(n: i64) -> Rat {
    Rat::from(Int::from(n))
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
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn apply(f: Function, a: &Basic) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), vec![a.clone()]), vec![a.clone()])
}
fn cos(a: &Basic) -> R {
    apply(Function::Cos, a)
}
fn sin(a: &Basic) -> R {
    apply(Function::Sin, a)
}
fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn all_coeffs(f: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    crate::polys::polytools::Poly::new(f, std::slice::from_ref(x))?.all_coeffs()
}
fn degree(f: &Basic, x: &Basic) -> PolyResult<i64> {
    crate::polys::polytools::Poly::new(f, std::slice::from_ref(x))?.degree(0)
}

fn linear(e: &Basic, n: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let slope: Basic = crate::core::function::derivative(e, n)?;
    if slope.has(n) {
        return Ok(None);
    }
    let constant: Basic = subs(e, n, &int(0))?;
    if expand(&s2(&s2(e, &m2(&slope, n)?)?, &constant)?)? != int(0) {
        return Ok(None);
    }
    Ok(Some((slope, constant)))
}

struct Term {
    coefficient: Basic,
    power: i64,
    base: Basic,
    trig: Vec<Basic>,
    factorial: bool,
    shift: Option<i64>,
    impulse: Option<i64>,
    steps: Vec<Basic>,
}

fn nonnegative_integer(e: &Basic) -> Option<i64> {
    if e.is_integer_value() {
        let v: i64 = i64::try_from(&numer(e.as_rational()?)).ok()?;
        if v >= 0 {
            return Some(v);
        }
    }
    None
}

fn classify(term: &Basic, n: &Basic) -> PolyResult<Option<Term>> {
    let mut r = Term {
        coefficient: int(1),
        power: 0,
        base: int(1),
        trig: Vec::new(),
        factorial: false,
        shift: None,
        impulse: None,
        steps: Vec::new(),
    };
    let factors: Vec<Basic> = if term.is_mul() {
        term.args().to_vec()
    } else {
        vec![term.clone()]
    };
    for factor in &factors {
        let f = func(factor);
        if !factor.has(n) {
            r.coefficient = m2(&r.coefficient, factor)?;
        } else if factor == n {
            r.power += 1;
        } else if f == Some(Function::Exp) {
            let Some((slope, constant)) = linear(&factor.args()[0], n)? else {
                return Ok(None);
            };
            r.base = m2(&r.base, &exp(&slope)?)?;
            r.coefficient = m2(&r.coefficient, &exp(&constant)?)?;
        } else if matches!(f, Some(Function::Cos | Function::Sin)) {
            if linear(&factor.args()[0], n)?.is_none() {
                return Ok(None);
            }
            r.trig.push(factor.clone());
        } else if f == Some(Function::Heaviside) {
            let parts = linear(&factor.args()[0], n)?;
            match parts {
                Some((slope, constant)) if slope == int(1) && constant.is_integer_value() => {
                    r.steps.push(factor.clone())
                }
                _ => return Ok(None),
            }
        } else if f == Some(Function::KroneckerDelta) {
            let (first, second) = (&factor.args()[0], &factor.args()[1]);
            let position = if second == n {
                nonnegative_integer(first)
            } else if first == n {
                nonnegative_integer(second)
            } else {
                None
            };
            if position.is_none() || r.impulse.is_some() {
                return Ok(None);
            }
            r.impulse = position;
        } else if factor.is_pow() {
            let (base, exponent) = (&factor.args()[0], &factor.args()[1]);
            if base == n {
                let Some(k) = nonnegative_integer(exponent) else {
                    return Ok(None);
                };
                r.power += k;
            } else if !base.has(n) {
                let Some((slope, constant)) = linear(exponent, n)? else {
                    return Ok(None);
                };
                r.base = m2(&r.base, &pw(base, &slope)?)?;
                r.coefficient = m2(&r.coefficient, &pw(base, &constant)?)?;
            } else if *exponent == int(-1)
                && func(base) == Some(Function::Factorial)
                && base.args()[0] == *n
                && !r.factorial
            {
                r.factorial = true;
            } else if *exponent == int(-1) && r.shift.is_none() {
                let Some((slope, constant)) = linear(base, n)? else {
                    return Ok(None);
                };
                if slope != int(1) {
                    return Ok(None);
                }
                match nonnegative_integer(&constant) {
                    Some(m) if m != 0 => r.shift = Some(m),
                    _ => return Ok(None),
                }
            } else if matches!(func(base), Some(Function::Cos | Function::Sin)) {
                let Some(k) = nonnegative_integer(exponent) else {
                    return Ok(None);
                };
                if linear(&base.args()[0], n)?.is_none() {
                    return Ok(None);
                }
                for _ in 0..k {
                    r.trig.push(base.clone());
                }
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        }
    }
    Ok(Some(r))
}

fn linearise(first: &Basic, second: &Basic) -> R {
    let (u, v) = (&first.args()[0], &second.args()[0]);
    let half: Basic = d2(&int(1), &int(2))?;
    let (fc, sc) = (func(first) == Some(Function::Cos), func(second) == Some(Function::Cos));
    if fc && sc {
        return m2(&half, &a2(&cos(&s2(u, v)?)?, &cos(&a2(u, v)?)?)?);
    }
    if !fc && !sc {
        return m2(&half, &s2(&cos(&s2(u, v)?)?, &cos(&a2(u, v)?)?)?);
    }
    if !fc {
        return m2(&half, &a2(&sin(&a2(u, v)?)?, &sin(&s2(u, v)?)?)?);
    }
    m2(&half, &s2(&sin(&a2(u, v)?)?, &sin(&s2(u, v)?)?)?)
}

fn unsupported(term: &Basic) -> PolyError {
    PolyError::NotImplemented(format!("the Z-transform of {}", term))
}

fn transform_term(term: &Basic, n: &Basic, z: &Basic) -> R {
    let Some(data) = classify(term, n)? else {
        return Err(unsupported(term));
    };
    if !data.steps.is_empty() {
        let step: &Basic = &data.steps[0];
        let argument: &Basic = &step.args()[0];
        let at_zero: Basic = if step.args().len() > 1 {
            step.args()[1].clone()
        } else {
            d2(&int(1), &int(2))?
        };
        let start: Basic = neg(&subs(argument, n, &int(0))?)?;
        if !start.is_integer_value() {
            return Err(unsupported(term));
        }
        let rest: Basic = d2(term, step)?;
        if crate::numbers::rat_is_negative(start.as_rational().expect("an integer")) {
            return z_transform(&rest, n, z);
        }
        let first: Basic = subs(&rest, n, &start)?;
        let inner: Basic = z_transform(&subs(&rest, n, &a2(n, &start)?)?, n, z)?;
        return m2(
            &pw(z, &neg(&start)?)?,
            &s2(&inner, &m2(&s2(&int(1), &at_zero)?, &first)?)?,
        );
    }
    if data.trig.len() > 1 {
        let mut rest: Basic = term.clone();
        for factor in &data.trig[..2] {
            rest = d2(&rest, factor)?;
        }
        return z_transform(&expand(&m2(&rest, &linearise(&data.trig[0], &data.trig[1])?)?)?, n, z);
    }
    if let Some(i) = data.impulse {
        let delta: Basic = crate::functions::special::tensor_functions::kronecker_delta(n, &int(i))?;
        let value: Basic = subs(&d2(term, &delta)?, n, &int(i))?;
        return m2(&value, &pw(z, &int(-i))?);
    }
    let a: Basic = data.base.clone();
    let special: usize = usize::from(data.factorial) + usize::from(data.shift.is_some()) + data.trig.len();
    if special > 1 {
        return Err(unsupported(term));
    }
    let mut rational: bool = true;
    let mut image: Basic;
    if data.factorial {
        image = exp(&d2(&a, z)?)?;
        rational = false;
    } else if let Some(m) = data.shift {
        let ratio: Basic = d2(&a, z)?;
        let mut parts: Vec<Basic> = Vec::new();
        for j in 1..m {
            parts.push(d2(&pw(&ratio, &int(j))?, &int(j))?);
        }
        let log_part: Basic = neg(&crate::functions::elementary::exponential::log(&s2(&int(1), &ratio)?)?)?;
        image = m2(
            &pw(&d2(z, &a)?, &int(m))?,
            &s2(&log_part, &crate::core::add::add(&parts)?)?,
        )?;
        rational = false;
    } else if !data.trig.is_empty() {
        let function: &Basic = &data.trig[0];
        let Some((omega, phase)) = linear(&function.args()[0], n)? else {
            return Err(unsupported(term));
        };
        let co: Basic = cos(&omega)?;
        let denominator: Basic = a2(
            &s2(&pw(z, &int(2))?, &m2(&m2(&m2(&int(2), &a)?, z)?, &co)?)?,
            &pw(&a, &int(2))?,
        )?;
        let even: Basic = m2(z, &s2(z, &m2(&a, &co)?)?)?;
        let odd: Basic = m2(&m2(&a, z)?, &sin(&omega)?)?;
        let (cp, sp) = (cos(&phase)?, sin(&phase)?);
        image = if func(function) == Some(Function::Cos) {
            d2(&s2(&m2(&cp, &even)?, &m2(&sp, &odd)?)?, &denominator)?
        } else {
            d2(&a2(&m2(&sp, &even)?, &m2(&cp, &odd)?)?, &denominator)?
        };
    } else {
        image = d2(z, &s2(z, &a)?)?;
    }
    for _ in 0..data.power {
        image = m2(&neg(z)?, &crate::core::function::derivative(&image, z)?)?;
    }
    let result: Basic = m2(&data.coefficient, &image)?;
    if rational {
        normal(&result)
    } else {
        Ok(result)
    }
}

fn has_node(e: &Basic, f: Function, z: &Basic) -> bool {
    e.preorder().iter().any(|node| func(node) == Some(f) && node.has(z))
}

/// The unilateral Z-transform of `f(n)` in `z` (`z_transform`).
pub fn z_transform(f: &Basic, n: &Basic, z: &Basic) -> R {
    let expression: Basic = expand(f)?;
    let terms: Vec<Basic> = if expression.is_add() {
        expression.args().to_vec()
    } else {
        vec![expression.clone()]
    };
    let mut rational: Vec<Basic> = Vec::new();
    let mut others: Vec<Basic> = Vec::new();
    for term in &terms {
        let image: Basic = transform_term(term, n, z)?;
        if has_node(&image, Function::Exp, z) || has_node(&image, Function::Log, z) {
            others.push(image);
        } else {
            rational.push(image);
        }
    }
    let total: Basic = if rational.is_empty() {
        int(0)
    } else {
        normal(&crate::core::add::add(&rational)?)?
    };
    let mut all: Vec<Basic> = vec![total];
    all.extend(others);
    crate::core::add::add(&all)
}

/// A factor of the denominator with the coefficients of its terms
/// (`_Part`).
struct Part {
    factor: Basic,
    symbol: Basic,
    coefficients: Vec<Basic>,
}

type Elem<F> = <<F as Field>::D as Domain>::Elem;

fn field_polynomial<F: Field>(field: &F, cs: &[Basic]) -> PolyResult<Vec<Elem<F>>> {
    let v: Vec<Elem<F>> = cs.iter().map(|c| field.element(c)).collect::<PolyResult<_>>()?;
    Ok(dup_strip(&v, field.domain()))
}

fn polynomial_expr<F: Field>(field: &F, f: &[Elem<F>], x: &Basic) -> R {
    let degree: i64 = f.len() as i64 - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (k, c) in f.iter().enumerate() {
        terms.push(m2(&field.expr(c)?, &pw(x, &int(degree - k as i64))?)?);
    }
    crate::core::add::add(&terms)
}

fn decompose<F: Field>(
    field: &F,
    numer_: &[Basic],
    denom_: &[Basic],
    t: &Basic,
    parameters: &[Basic],
) -> PolyResult<(Vec<Basic>, Vec<Part>)> {
    let k: &F::D = field.domain();
    let width: usize = numer_.len().max(denom_.len());
    let pad = |v: &[Basic]| -> Vec<Basic> {
        let mut out: Vec<Basic> = v.to_vec();
        out.extend(std::iter::repeat(int(0)).take(width - v.len()));
        out
    };
    let a = field_polynomial(field, &pad(numer_))?;
    let b = field_polynomial(field, &pad(denom_))?;
    if denom_.len() < width || b.is_empty() || k.is_zero(&b[b.len() - 1]) {
        return Err(value_error(
            "not the Z-transform of a sequence: the function is not finite at infinity".to_string(),
        ));
    }
    let (quotient, mut remainder) = dup_div(&a, &b, k)?;
    let impulses: Vec<Basic> = quotient
        .iter()
        .rev()
        .map(|c| field.expr(c))
        .collect::<PolyResult<_>>()?;
    if remainder.is_empty() || b.len() == 1 {
        return Ok((impulses, Vec::new()));
    }
    let leading = k.invert(&b[0])?;
    remainder = dup_mul_ground(&remainder, &leading, k);
    let expression: Basic = polynomial_expr(field, &dup_mul_ground(&b, &leading, k), t)?;
    let numerator: Basic = if parameters.is_empty() {
        expression
    } else {
        crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&expression, false, true)?)?.0
    };
    let mut gens: Vec<Basic> = vec![t.clone()];
    gens.extend(parameters.iter().cloned());
    let (_, listed) = crate::polys::polytools::factor_list(&numerator, &gens)?;
    let mut factors: Vec<(Vec<Elem<F>>, Basic, usize)> = Vec::new();
    for (factor, exponent) in listed {
        if !factor.has(t) || !exponent.is_integer_value() {
            continue;
        }
        let g = field_polynomial(field, &all_coeffs(&factor, t)?)?;
        let lead_inverse = k.invert(&g[0])?;
        let m: usize = usize::try_from(&numer(exponent.as_rational().expect("an integer")))
            .map_err(|_| value_error("multiplicity".to_string()))?;
        factors.push((dup_mul_ground(&g, &lead_inverse, k), factor, m));
    }
    let powers: Vec<Vec<Elem<F>>> = factors.iter().map(|(g, _, m)| dup_pow(g, *m, k)).collect();
    let mut parts: Vec<Part> = Vec::new();
    for (index, (g, factor, m)) in factors.iter().enumerate() {
        let mut cofactor: Vec<Elem<F>> = vec![k.one()];
        for (other, power) in powers.iter().enumerate() {
            if other != index {
                cofactor = dup_mul(&cofactor, power, k);
            }
        }
        let inverse: Vec<Elem<F>> = if factors.len() > 1 {
            dup_invert(&dup_rem(&cofactor, &powers[index], k)?, &powers[index], k)?
        } else {
            vec![k.one()]
        };
        let local: Vec<Elem<F>> = dup_rem(&dup_mul(&remainder, &inverse, k), &powers[index], k)?;
        let integral: Vec<Basic> = all_coeffs(factor, t)?;
        let mut reciprocal_terms: Vec<Basic> = Vec::new();
        for (kk, c) in integral.iter().enumerate() {
            reciprocal_terms.push(m2(c, &pw(t, &int(kk as i64))?)?);
        }
        let reciprocal_expr: Basic = crate::core::add::add(&reciprocal_terms)?;
        let reversed: Vec<Basic> = integral.iter().rev().cloned().collect();
        let reciprocal = field_polynomial(field, &reversed)?;
        let rlead = k.invert(&reciprocal[0])?;
        let f: QuotientField<F::D> = QuotientField::new(k.clone(), dup_mul_ground(&reciprocal, &rlead, k))?;
        let pole = f.generator();
        let root = f.invert(&pole)?;
        let lifted: Vec<_> = g.iter().map(|c| f.from_ground(c)).collect();
        let cofactor_at_root = dup_div(&lifted, &[f.one(), f.neg(&root)], &f)?.0;
        let shift = vec![f.one(), root.clone()];
        let shifted = |p: &[Vec<Elem<F>>]| -> Vec<Vec<Elem<F>>> {
            let mut total: Vec<Vec<Elem<F>>> = Vec::new();
            for c in p {
                total = dup_add(
                    &dup_mul(&total, &shift, &f),
                    &dup_strip(std::slice::from_ref(c), &f),
                    &f,
                );
            }
            total
        };
        let lifted_local: Vec<_> = local.iter().map(|c| f.from_ground(c)).collect();
        let top: Vec<_> = shifted(&lifted_local).into_iter().rev().collect();
        let bottom: Vec<_> = dup_pow(&shifted(&cofactor_at_root), *m, &f).into_iter().rev().collect();
        let mut series: Vec<Vec<Elem<F>>> = Vec::new();
        let scale = f.invert(&bottom[0])?;
        for kk in 0..*m {
            let mut value = if kk < top.len() { top[kk].clone() } else { f.zero() };
            for i in 1..=kk {
                if i < bottom.len() {
                    value = f.sub(&value, &f.mul(&bottom[i], &series[kk - i]));
                }
            }
            series.push(f.mul(&value, &scale));
        }
        let minus_inverse = f.neg(&pole);
        let mut coefficients: Vec<Basic> = Vec::new();
        for j in 1..=*m {
            let element = f.mul(&series[m - j], &f.pow(&minus_inverse, j as u64));
            coefficients.push(polynomial_expr(field, &element, t)?);
        }
        parts.push(Part {
            factor: reciprocal_expr,
            symbol: t.clone(),
            coefficients,
        });
    }
    Ok((impulses, parts))
}

fn binomial(n: &Basic, j: usize) -> R {
    if j == 1 {
        return Ok(int(1));
    }
    let mut denominator: i64 = 1;
    for i in 1..j as i64 {
        denominator *= i;
    }
    let factors: Vec<Basic> = (1..j as i64).map(|i| a2(n, &int(i))).collect::<PolyResult<_>>()?;
    d2(&crate::core::mul::mul(&factors)?, &int(denominator))
}

fn real_form(part: &Part, n: &Basic) -> PolyResult<Option<Basic>> {
    if degree(&part.factor, &part.symbol)? != 2 {
        return Ok(None);
    }
    let values: Vec<Basic> = all_coeffs(&part.factor, &part.symbol)?;
    if !values.iter().all(|c| c.is_rational_class()) {
        return Ok(None);
    }
    let r = |c: &Basic| c.as_rational().expect("rational").clone();
    let lead: Rat = r(&values[0]);
    let p: Rat = r(&values[1]) / lead.clone();
    let qq: Rat = r(&values[2]) / lead;
    let discriminant: Rat = qq.clone() - p.clone() * p.clone() / q(4);
    if discriminant <= q(0) {
        return Ok(None);
    }
    let s: Basic = crate::matrices::eigen::sqrt(&number(discriminant))?;
    let modulus: Basic = crate::matrices::eigen::sqrt(&number(qq))?;
    let theta: Basic = crate::functions::elementary::inverse::acos(&d2(&number(-p.clone() / q(2)), &modulus)?)?;
    let mut total: Vec<Basic> = Vec::new();
    for (index, coefficient) in part.coefficients.iter().enumerate() {
        let j: usize = index + 1;
        let lin: Vec<Basic> = if coefficient.has(&part.symbol) {
            all_coeffs(coefficient, &part.symbol)?
        } else {
            vec![coefficient.clone()]
        };
        let (e1, e0) = if lin.len() == 2 {
            (lin[0].clone(), lin[1].clone())
        } else {
            (int(0), lin[0].clone())
        };
        let u: Basic = s2(&e0, &m2(&e1, &number(p.clone() / q(2)))?)?;
        let v: Basic = m2(&e1, &s)?;
        let nt: Basic = m2(n, &theta)?;
        let wave: Basic = s2(&m2(&u, &cos(&nt)?)?, &m2(&v, &sin(&nt)?)?)?;
        total.push(m2(&m2(&int(2), &binomial(n, j)?)?, &wave)?);
    }
    Ok(Some(m2(&pw(&modulus, n)?, &crate::core::add::add(&total)?)?))
}

fn assemble(impulses: &[Basic], parts: &[Part], n: &Basic, real: bool) -> R {
    let mut terms: Vec<Basic> = Vec::new();
    for (j, c) in impulses.iter().enumerate() {
        if *c != int(0) {
            terms.push(m2(
                c,
                &crate::functions::special::tensor_functions::kronecker_delta(n, &int(j as i64))?,
            )?);
        }
    }
    for part in parts {
        if real {
            if let Some(form) = real_form(part, n)? {
                terms.push(form);
                continue;
            }
        }
        let d: i64 = degree(&part.factor, &part.symbol)?;
        let found = crate::polys::polyroots::roots(&part.factor, std::slice::from_ref(&part.symbol), true, true, None)?;
        let total: usize = found.iter().map(|(_, k)| *k).sum();
        if total as i64 != d {
            return Err(PolyError::NotImplemented(format!(
                "the roots of {} are not expressed in radicals",
                part.factor
            )));
        }
        let mut values: Vec<Basic> = found.iter().map(|(r, _)| r.clone()).collect();
        values.sort_by_cached_key(crate::core::sorting::sort_key);
        for value in &values {
            for (index, coefficient) in part.coefficients.iter().enumerate() {
                let at: Basic = subs(coefficient, &part.symbol, value)?;
                terms.push(m2(&m2(&at, &binomial(n, index + 1)?)?, &pw(value, n)?)?);
            }
        }
    }
    crate::core::add::add(&terms)
}

/// The sequence whose Z-transform is the rational function `F(z)`
/// (`inverse_z_transform`).
pub fn inverse_z_transform(given: &Basic, z: &Basic, n: &Basic) -> R {
    let expression: Basic = crate::polys::rationaltools::together(given, false, true)?;
    let (numerator, denominator) = crate::core::expr::as_numer_denom(&expression)?;
    let kernels: Vec<Basic> = kernels_of(&[&ExprMatrix::new(1, 2, vec![numerator.clone(), denominator.clone()])?]);
    if !kernels.iter().all(|k| k.is_symbol()) {
        return Err(PolyError::NotImplemented(format!(
            "the inverse Z-transform of {}, which is not a rational function of symbols",
            expression
        )));
    }
    let used: Vec<String> = kernels.iter().map(|k| k.to_string()).collect();
    let (mut name, mut count) = ("_t".to_string(), 0);
    while used.contains(&name) {
        name = format!("_t{}", count);
        count += 1;
    }
    let t: Basic = crate::core::symbol::Symbol::new(&name, &[]).0;
    let numer_: Vec<Basic> = if numerator.has(z) {
        all_coeffs(&numerator, z)?.into_iter().rev().collect()
    } else {
        vec![numerator.clone()]
    };
    let denom_: Vec<Basic> = if denominator.has(z) {
        all_coeffs(&denominator, z)?.into_iter().rev().collect()
    } else {
        vec![denominator.clone()]
    };
    let parameters: Vec<Basic> = kernels.iter().filter(|k| *k != z).cloned().collect();
    let (impulses, parts) = if parameters.is_empty() {
        decompose(&RationalEntries::new(), &numer_, &denom_, &t, &parameters)?
    } else {
        decompose(
            &FunctionEntries::new(PolynomialLift::new(&parameters)),
            &numer_,
            &denom_,
            &t,
            &parameters,
        )?
    };
    let result: Basic = assemble(&impulses, &parts, n, parameters.is_empty())?;
    if parameters.is_empty() {
        expand(&result)
    } else {
        Ok(result)
    }
}
