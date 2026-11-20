//! Port of `sympy_extras/concrete/products.py`: the evaluation of
//! `Product(f, (k, a, n))` (`eval_product`, `product_doit`, `product`):
//! direct products, the roots of a polynomial term as rising factorials,
//! products of products, powers with a summed exponent, deltas.

#![allow(clippy::mutable_key_type)]

use crate::concrete::delta::Limit;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{PolyError, PolyResult};

type R = PolyResult<Basic>;
type OptR = PolyResult<Option<Basic>>;

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
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn rf(a: &Basic, k: &Basic) -> R {
    crate::functions::combinatorial::factorials::rf(a, k)
}
fn count_of(n: &Basic, a: &Basic) -> R {
    a2(&s2(n, a)?, &int(1))
}
fn tuple(l: &Limit) -> Basic {
    Basic::new(Head::Tuple, vec![l.0.clone(), l.1.clone(), l.2.clone()])
}
fn as_int(e: &Basic) -> Option<i64> {
    match e.head() {
        Head::Rational {
            integer_class: true, ..
        } => i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok(),
        _ => None,
    }
}

fn eval_product_direct(term: &Basic, limits: &Limit) -> R {
    let (k, a, n) = limits;
    let count: i64 = as_int(&count_of(n, a)?).ok_or_else(|| PolyError::Value("an integer count".to_string()))?;
    let factors: Vec<Basic> = (0..count)
        .map(|j| subs(term, k, &a2(a, &int(j))?))
        .collect::<PolyResult<_>>()?;
    mul_all(&factors)
}

fn doit_of(term: &Basic, limits: &Limit) -> R {
    let p: Basic = crate::concrete::expr_with_limits::product_(term, &[tuple(limits)])?;
    if matches!(p.head(), Head::Product) {
        return product_doit(&p, true);
    }
    Ok(p)
}

/// The product of `term` over the limits, `None` when it is not found
/// (`eval_product`).
pub fn eval_product(term: &Basic, limits: &Limit) -> OptR {
    let (k, a, n) = limits;
    if !term.free_symbols().contains(k) {
        if is(&s2(term, &int(1))?, Predicate::zero) == Some(true) {
            return Ok(Some(int(1)));
        }
        return Ok(Some(pw(term, &count_of(n, a)?)?));
    }
    if a == n {
        return Ok(Some(subs(term, k, a)?));
    }
    if term
        .preorder()
        .iter()
        .any(|e| matches!(e.head(), Head::Function(Function::KroneckerDelta)))
        && crate::concrete::delta::has_simple_delta(term, k)?
    {
        return Ok(Some(crate::concrete::delta::deltaproduct(term, limits)?));
    }
    let dif: Basic = s2(n, a)?;
    let definite: Option<i64> = as_int(&dif);
    if definite.is_some_and(|d| d < 100) {
        return Ok(Some(eval_product_direct(term, limits)?));
    }
    let polynomial: bool = crate::core::expr::polynomial_like(
        term,
        std::slice::from_ref(k),
        crate::core::expr::PolynomialKind::Polynomial,
    ) == Some(true);
    if polynomial {
        let poly = match crate::polys::polytools::Poly::new(term, std::slice::from_ref(k)) {
            Ok(p) => p,
            Err(PolyError::Polys(crate::error::PolysErrorKind::CoercionFailed, _)) => {
                return eval_polynomial_ex(term, limits)
            }
            Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => {
                return eval_polynomial_ex(term, limits)
            }
            Err(e) => return Err(e),
        };
        let (mut big_a, mut big_b, mut big_d) = (int(1), int(1), int(1));
        let found = crate::polys::polyroots::roots(term, std::slice::from_ref(k), true, true, None)?;
        let mut multiplicity: i64 = 0;
        for (r, m) in found.iter() {
            let m: i64 = *m as i64;
            multiplicity += m;
            big_a = m2(&big_a, &pw(&rf(&s2(a, r)?, &count_of(n, a)?)?, &int(m))?)?;
            big_d = m2(&big_d, &pw(&s2(k, r)?, &int(m))?)?;
        }
        // SymPy builds Q = prod((n - r)**m), a constant in k, and takes the
        // product of quo(poly, Q.as_poly(k)), the whole polynomial again:
        // wrong when some roots are found, a recursion without end when none
        // is. The remaining factor is the monic cofactor of the roots found,
        // and with no root found the other methods are tried (issue #25)
        if multiplicity > 0 {
            if multiplicity < poly.degree(0)? {
                let divisor = crate::polys::polytools::Poly::new(&big_d, std::slice::from_ref(k))?;
                let cofactor: Basic = poly.div(&divisor)?.0.monic()?.as_expr()?;
                big_b = doit_of(&cofactor, limits)?;
            }
            let lc: Basic = poly.LC()?;
            return Ok(Some(m2(&m2(&pw(&lc, &count_of(n, a)?)?, &big_a)?, &big_b)?));
        }
    }
    if term.is_add() {
        let factored: Basic = crate::core::exprtools::factor_terms(term, false, false, true, true)?;
        if factored.is_mul() {
            return eval_product(&factored, limits);
        }
    } else if term.is_mul() {
        let without_k: Basic = mul_all(&term.args().iter().filter(|f| !f.has(k)).cloned().collect::<Vec<_>>())?;
        let with_k: Vec<Basic> = term.args().iter().filter(|f| f.has(k)).cloned().collect();
        let count: Basic = count_of(n, a)?;
        if with_k.len() >= 2 {
            let mut exclude: Vec<Basic> = Vec::new();
            let mut include: Vec<Basic> = Vec::new();
            for t in &with_k {
                match eval_product(t, limits)? {
                    Some(p) => exclude.push(p),
                    None => include.push(t.clone()),
                }
            }
            if exclude.is_empty() {
                return Ok(None);
            }
            let rest: Basic = doit_of(&mul_all(&include)?, limits)?;
            return Ok(Some(m2(&m2(&pw(&without_k, &count)?, &mul_all(&exclude)?)?, &rest)?));
        }
        let p1: Basic = match eval_product(&with_k[0], limits)? {
            Some(p) => p,
            None => doit_of(&with_k[0], limits)?,
        };
        return Ok(Some(m2(&pw(&without_k, &count)?, &p1)?));
    } else if term.is_pow() {
        let (base, exponent) = (&term.args()[0], &term.args()[1]);
        if !base.has(k) {
            let s: Basic = crate::concrete::summations::summation(exponent, limits)?;
            return Ok(Some(pw(base, &s)?));
        }
        if !exponent.has(k) {
            if let Some(p2) = eval_product(base, limits)? {
                return Ok(Some(pw(&p2, exponent)?));
            }
        }
    } else if matches!(term.head(), Head::Product) {
        let evaluated: Basic = product_doit(term, true)?;
        return Ok(Some(match eval_product(&evaluated, limits)? {
            Some(f) => f,
            None => crate::concrete::expr_with_limits::product_(&evaluated, &[tuple(limits)])?,
        }));
    }
    if definite.is_some() {
        return Ok(Some(eval_product_direct(term, limits)?));
    }
    Ok(None)
}

fn eval_polynomial_ex(term: &Basic, limits: &Limit) -> OptR {
    let (k, a, n) = limits;
    let poly = match crate::concrete::gosper::UPoly::of(term, k) {
        Ok(p) => p,
        Err(PolyError::Value(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let found: Vec<Basic> = match crate::polys::polyroots::ex_roots(&poly.as_expr(k)?, k) {
        Ok(f) => f,
        Err(PolyError::NotImplemented(m)) if !m.starts_with(crate::error::NOT_PORTED_PREFIX) => return Ok(None),
        Err(e) => return Err(e),
    };
    if found.is_empty() {
        return Ok(None);
    }
    let mut counts: Vec<(Basic, i64)> = Vec::new();
    for r in found {
        match counts.iter_mut().find(|(s, _)| *s == r) {
            Some(entry) => entry.1 += 1,
            None => counts.push((r, 1)),
        }
    }
    let mut big_a: Basic = int(1);
    let mut multiplicity: i64 = 0;
    for (r, m) in &counts {
        multiplicity += m;
        big_a = m2(&big_a, &pw(&rf(&s2(a, r)?, &count_of(n, a)?)?, &int(*m))?)?;
    }
    if multiplicity < poly.degree() {
        // SymPy's quo(poly, Q.as_poly(k)) divides by a constant in k (issue
        // #25): the cofactor of the roots found is not known here
        return Ok(None);
    }
    Ok(Some(m2(&pw(&poly.lc(), &count_of(n, a)?)?, &big_a)?))
}

/// The product evaluated over its limits in turn (`product_doit`).
pub fn product_doit(p: &Basic, deep: bool) -> R {
    use crate::concrete::expr_with_limits::{limits, product_};
    let lims: Vec<Limit> = limits(p);
    let mut reps: Vec<(Basic, Basic)> = Vec::new();
    for xab in &lims {
        if let Some(d) = crate::concrete::summations::dummy_with_inherited_properties(xab) {
            reps.push((xab.0.clone(), d));
        }
    }
    if !reps.is_empty() {
        let forward: std::collections::HashMap<Basic, Basic> = reps.iter().cloned().collect();
        let undo: std::collections::HashMap<Basic, Basic> = reps.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
        let replaced: Basic = p.xreplace(&forward);
        return Ok(product_doit(&replaced, deep)?.xreplace(&undo));
    }
    let mut f: Basic = p.args()[0].clone();
    for (index, (i, a, b)) in lims.iter().enumerate() {
        let (mut a, mut b) = (a.clone(), b.clone());
        let dif: Basic = s2(&b, &a)?;
        if is(&dif, Predicate::integer) == Some(true) && is(&dif, Predicate::negative) == Some(true) {
            let (na, nb) = (a2(&b, &int(1))?, s2(&a, &int(1))?);
            a = na;
            b = nb;
            f = d2(&int(1), &f)?;
        }
        match eval_product(&f, &(i.clone(), a, b))? {
            Some(g) if !matches!(g.head(), Head::NaN) => f = g,
            _ => {
                let rest: Vec<Basic> = lims[index..].iter().map(tuple).collect();
                return product_(&crate::simplify::powsimp::powsimp_default(&f)?, &rest);
            }
        }
    }
    if deep {
        return crate::core::function::doit(&f);
    }
    crate::simplify::powsimp::powsimp_default(&f)
}

/// `Product(f, *limits)` evaluated (`product`).
pub fn product(f: &Basic, limit: &Limit) -> R {
    let p: Basic = crate::concrete::expr_with_limits::product_(f, &[tuple(limit)])?;
    if !matches!(p.head(), Head::Product) {
        return Ok(p);
    }
    product_doit(&p, false)
}
