//! Port of `sympy_extras/simplify/radsimp.py`: [`fraction`], [`numer`],
//! [`denom`], [`unevaluated_mul`], [`split_surds`], [`rad_rationalize`]
//! and [`radsimp`]; `collect` is not ported yet. Where the Python asks
//! `nsimplify` whether a numerical denominator is a rational, the Rust
//! port tries the rational of its continued fraction and proves it with
//! `equals` (the core has no `Float`).

#![allow(clippy::mutable_key_type, clippy::needless_range_loop)]

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{
    as_base_exp, as_numer_denom, factors_of, is_false, is_integer_class, is_rational_value, is_true,
};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::error::PolyResult;
use crate::numbers::*;

/// `(numerator, denominator)` of a product, without simplification
/// (`fraction` with `exact=False`).
pub fn fraction(e: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut numerator: Vec<Basic> = Vec::new();
    let mut denominator: Vec<Basic> = Vec::new();
    for term in factors_of(e) {
        if !is_false(&term, Predicate::commutative) && (term.is_pow() || term.is_function(Function::Exp)) {
            let (b, ex) = as_base_exp(&term)?;
            if is_true(&ex, Predicate::negative) {
                if is_integer_class(&ex) && is_rational_value(&ex, -1) {
                    denominator.push(b);
                } else {
                    denominator.push(pow(&b, &mul(&[Basic::negative_one(), ex])?)?);
                }
            } else if is_true(&ex, Predicate::positive) {
                numerator.push(term);
            } else if ex.is_mul() {
                let (n, d) = as_numer_denom(&term)?;
                if !(is_integer_class(&n) && is_rational_value(&n, 1)) {
                    numerator.push(n);
                }
                denominator.push(d);
            } else {
                numerator.push(term);
            }
        } else if let Some(v) = term.as_rational().filter(|v| denom(v) != Int::ONE) {
            if numer(v) != Int::ONE {
                numerator.push(Basic::integer(numer(v)));
            }
            denominator.push(Basic::integer(denom(v)));
        } else {
            numerator.push(term);
        }
    }
    Ok((mul(&numerator)?, mul(&denominator)?))
}

/// The numerator of [`fraction`] (`numer`).
pub fn numer_of(e: &Basic) -> PolyResult<Basic> {
    Ok(fraction(e)?.0)
}

/// The denominator of [`fraction`] (`denom`).
pub fn denom_of(e: &Basic) -> PolyResult<Basic> {
    Ok(fraction(e)?.1)
}

/// A well-formed unevaluated product: the numbers multiplied into one
/// coefficient, products flattened, the factors sorted (SymPy's
/// `_unevaluated_Mul`).
pub fn unevaluated_mul(args: &[Basic]) -> PolyResult<Basic> {
    let mut pending: Vec<Basic> = args.to_vec();
    let mut commutative: Vec<Basic> = Vec::new();
    let mut noncommutative: Vec<Basic> = Vec::new();
    let mut coefficient: Basic = Basic::one();
    let mut i: usize = 0;
    while i < pending.len() {
        let a: Basic = pending[i].clone();
        i += 1;
        if a.is_mul() {
            for f in a.args() {
                if is_false(f, Predicate::commutative) {
                    noncommutative.push(f.clone());
                } else {
                    pending.push(f.clone());
                }
            }
        } else if a.is_number_class() {
            coefficient = mul(&[coefficient, a])?;
        } else if !is_false(&a, Predicate::commutative) {
            commutative.push(a);
        } else {
            noncommutative.push(a);
        }
    }
    commutative.sort_by(|x, y| x.compare_key().cmp(y.compare_key()));
    if !(is_integer_class(&coefficient) && is_rational_value(&coefficient, 1)) {
        commutative.insert(0, coefficient);
    }
    commutative.extend(noncommutative);
    Ok(crate::core::mul::raw_mul(commutative))
}

fn inverse(x: &Basic) -> PolyResult<Basic> {
    pow(x, &Basic::negative_one())
}

fn add(v: &[Basic]) -> PolyResult<Basic> {
    crate::core::add::add(v)
}

fn sqrt(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::half())
}

fn is_one(e: &Basic) -> bool {
    is_integer_class(e) && is_rational_value(e, 1)
}

/// The bases and exponents of a product, `I` counted, a rational split
/// into numerator and denominator (`_factors_of`, after SymPy's
/// `Factors`).
fn factors_dict(e: &Basic) -> PolyResult<Vec<(Basic, Basic)>> {
    let all: Vec<Basic> = factors_of(e);
    let commutative: Vec<Basic> = all
        .iter()
        .filter(|f| !is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    let noncommutative: Vec<Basic> = all
        .iter()
        .filter(|f| is_false(f, Predicate::commutative))
        .cloned()
        .collect();
    let i_unit: Basic = Basic::imaginary_unit();
    let units: i64 = commutative.iter().filter(|f| **f == i_unit).count() as i64;
    let rest: Vec<Basic> = commutative.into_iter().filter(|f| *f != i_unit).collect();
    let mut powers: Vec<(Basic, Basic)> = Vec::new();
    let upsert = |powers: &mut Vec<(Basic, Basic)>, k: Basic, v: Basic| -> PolyResult<()> {
        match powers.iter().position(|(b, _)| *b == k) {
            Some(i) => powers[i].1 = add(&[powers[i].1.clone(), v])?,
            None => powers.push((k, v)),
        }
        Ok(())
    };
    if !rest.is_empty() {
        let product: Basic = crate::core::mul::raw_mul(rest);
        for f in factors_of(&product) {
            let (b, ex) = as_base_exp(&f)?;
            upsert(&mut powers, b, ex)?;
        }
    }
    let keys: Vec<Basic> = powers.iter().map(|(k, _)| k.clone()).collect();
    for f in keys {
        let Some(v) = f
            .as_rational()
            .filter(|v| denom(v) != Int::ONE && f.is_rational_class())
            .cloned()
        else {
            continue;
        };
        let exponent: Basic = powers
            .iter()
            .find(|(k, _)| *k == f)
            .map(|(_, e)| e.clone())
            .expect("present");
        upsert(&mut powers, Basic::integer(numer(&v)), exponent.clone())?;
        upsert(
            &mut powers,
            Basic::integer(denom(&v)),
            crate::core::mul::neg(&exponent)?,
        )?;
        powers.retain(|(k, _)| *k != f);
    }
    if units != 0 {
        upsert(&mut powers, i_unit, Basic::int(units))?;
    }
    if !noncommutative.is_empty() {
        powers.retain(|(k, _)| !k.is_mul() || k.args() != noncommutative.as_slice());
        powers.push((crate::core::mul::raw_mul(noncommutative), Basic::one()));
    }
    Ok(powers)
}

fn split_gcd(a: &[Basic]) -> PolyResult<(Basic, Vec<Basic>, Vec<Basic>)> {
    let mut g: Basic = a[0].clone();
    let mut b1: Vec<Basic> = vec![g.clone()];
    let mut b2: Vec<Basic> = Vec::new();
    for x in &a[1..] {
        let g1: Basic = crate::polys::polytools::gcd_expr(&g, x, &[])?;
        if is_one(&g1) {
            b2.push(x.clone());
        } else {
            g = g1;
            b1.push(x.clone());
        }
    }
    Ok((g, b1, b2))
}

/// `(g, a, b)` with `expr = a*sqrt(g) + b`, `a` collecting the surds whose
/// radicands share the gcd `g` (`split_surds`).
pub fn split_surds(expr: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    let mut args: Vec<Basic> = expr.args().to_vec();
    args.sort_by_cached_key(crate::core::sorting::sort_key);
    let coeff_muls: Vec<(Basic, Basic)> = args.iter().map(crate::core::expr::as_coeff_mul).collect();
    let mut surds: Vec<Basic> = coeff_muls
        .iter()
        .filter(|(_, x)| x.is_pow())
        .map(|(_, x)| pow(x, &Basic::int(2)))
        .collect::<PolyResult<_>>()?;
    surds.sort_by_cached_key(crate::core::sorting::sort_key);
    let (g, b1, mut b2) = split_gcd(&surds)?;
    let mut g2: Basic = g.clone();
    if b2.is_empty() && b1.len() >= 2 {
        let ginv: Basic = inverse(&g)?;
        let b1n: Vec<Basic> = b1
            .iter()
            .map(|x| mul(&[x.clone(), ginv.clone()]))
            .collect::<PolyResult<Vec<Basic>>>()?
            .into_iter()
            .filter(|x| !is_one(x))
            .collect();
        let (g1, _, rest) = split_gcd(&b1n)?;
        b2 = rest;
        g2 = mul(&[g.clone(), g1])?;
    }
    let _ = b2;
    let (mut a1v, mut a2v): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for (c, sv) in &coeff_muls {
        if sv.is_pow() && sv.args()[1] == Basic::half() {
            let s1: &Basic = &sv.args()[0];
            if b1.contains(s1) {
                a1v.push(mul(&[c.clone(), sqrt(&mul(&[s1.clone(), inverse(&g2)?])?)?])?);
            } else {
                a2v.push(mul(&[c.clone(), sv.clone()])?);
            }
        } else {
            a2v.push(mul(&[c.clone(), sv.clone()])?);
        }
    }
    Ok((g2, add(&a1v)?, add(&a2v)?))
}

/// `num/den` with the square roots of the denominator removed
/// (`rad_rationalize`).
pub fn rad_rationalize(num: &Basic, den: &Basic) -> PolyResult<(Basic, Basic)> {
    let (mut num, mut den) = (num.clone(), den.clone());
    while den.is_add() {
        let (g, a, b) = split_surds(&den)?;
        let a: Basic = mul(&[a, sqrt(&g)?])?;
        let minus_b: Basic = crate::core::mul::neg(&b)?;
        num = mexpand(&mul(&[add(&[a.clone(), minus_b])?, num])?)?;
        den = mexpand(&add(&[
            pow(&a, &Basic::int(2))?,
            crate::core::mul::neg(&pow(&b, &Basic::int(2))?)?,
        ])?)?;
    }
    Ok((num, den))
}

fn ispow2(d: &Basic, symbolic: bool, log2: bool) -> PolyResult<bool> {
    if !d.is_pow() {
        return Ok(false);
    }
    let e: &Basic = &d.args()[1];
    let two: Basic = Basic::int(2);
    let rational_q: Option<Int> = e.as_rational().filter(|_| e.is_rational_class()).map(denom);
    if rational_q.as_ref().is_some_and(|q| *q == Int::from(2)) || (symbolic && denom_of(e)? == two) {
        return Ok(true);
    }
    if log2 {
        let mut q: Int = Int::ONE;
        if let Some(v) = rational_q {
            q = v;
        } else if symbolic {
            let dd: Basic = denom_of(e)?;
            if is_integer_class(&dd) {
                q = numer(dd.as_rational().expect("an integer"));
            }
        }
        if !q.is_one()
            && is_integer_class(&crate::functions::elementary::exponential::log_base(
                &Basic::integer(q),
                &two,
            )?)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn radsimp_numerator(rterms: &[(Basic, Basic)]) -> PolyResult<Basic> {
    let s = |i: usize| sqrt(&rterms[i].0);
    let c = |i: usize| rterms[i].1.clone();
    let big = |i: usize| rterms[i].0.clone();
    let p = |x: &Basic, k: i64| pow(x, &Basic::int(k));
    let m = |v: Vec<Basic>| mul(&v);
    let n = |k: i64| Basic::int(k);
    match rterms.len() {
        1 => sqrt(&rterms[0].0),
        2 => add(&[m(vec![s(0)?, c(0)])?, m(vec![n(-1), s(1)?, c(1)])?]),
        3 => {
            let first: Basic = add(&[
                m(vec![s(0)?, c(0)])?,
                m(vec![s(1)?, c(1)])?,
                m(vec![n(-1), s(2)?, c(2)])?,
            ])?;
            let second: Basic = add(&[
                m(vec![n(2), s(0)?, s(1)?, c(0), c(1)])?,
                m(vec![n(-1), big(0), p(&c(0), 2)?])?,
                m(vec![n(-1), big(1), p(&c(1), 2)?])?,
                m(vec![big(2), p(&c(2), 2)?])?,
            ])?;
            m(vec![first, second])
        }
        4 => {
            let (a, b, cc, d) = (c(0), c(1), c(2), c(3));
            let (aa, bb, cb, db) = (big(0), big(1), big(2), big(3));
            let (sa, sb, sc, sd) = (s(0)?, s(1)?, s(2)?, s(3)?);
            let f1: Basic = add(&[
                m(vec![sa.clone(), a.clone()])?,
                m(vec![sb.clone(), b.clone()])?,
                m(vec![n(-1), sc.clone(), cc.clone()])?,
                m(vec![n(-1), sd.clone(), d.clone()])?,
            ])?;
            let f2: Basic = add(&[
                m(vec![n(2), sa.clone(), sb.clone(), a.clone(), b.clone()])?,
                m(vec![n(-1), aa.clone(), p(&a, 2)?])?,
                m(vec![n(-1), bb.clone(), p(&b, 2)?])?,
                m(vec![n(-2), sc.clone(), sd.clone(), cc.clone(), d.clone()])?,
                m(vec![cb.clone(), p(&cc, 2)?])?,
                m(vec![db.clone(), p(&d, 2)?])?,
            ])?;
            let f3: Basic = add(&[
                m(vec![n(-8), sa, sb, sc, sd, a.clone(), b.clone(), cc.clone(), d.clone()])?,
                m(vec![p(&aa, 2)?, p(&a, 4)?])?,
                m(vec![n(-2), aa.clone(), bb.clone(), p(&a, 2)?, p(&b, 2)?])?,
                m(vec![n(-2), aa.clone(), cb.clone(), p(&a, 2)?, p(&cc, 2)?])?,
                m(vec![n(-2), aa.clone(), db.clone(), p(&a, 2)?, p(&d, 2)?])?,
                m(vec![p(&bb, 2)?, p(&b, 4)?])?,
                m(vec![n(-2), bb.clone(), cb.clone(), p(&b, 2)?, p(&cc, 2)?])?,
                m(vec![n(-2), bb.clone(), db.clone(), p(&b, 2)?, p(&d, 2)?])?,
                m(vec![p(&cb, 2)?, p(&cc, 4)?])?,
                m(vec![n(-2), cb.clone(), db.clone(), p(&cc, 2)?, p(&d, 2)?])?,
                m(vec![p(&db, 2)?, p(&d, 4)?])?,
            ])?;
            m(vec![f1, f2, f3])
        }
        _ => Err(crate::error::PolyError::NotImplemented(
            "radsimp of more than four radical terms".to_string(),
        )),
    }
}

fn mexpand(e: &Basic) -> PolyResult<Basic> {
    super::sqrtdenest::mexpand(e)
}

/// The rational equal to a numerical expression, when there is one: the
/// convergent of the continued fraction of its value proved by `equals`
/// (what `nsimplify` decides in the Python).
fn rational_equal_to(d: &Basic) -> PolyResult<Option<Basic>> {
    let Some((re, im)) = crate::numeric::reliable_complex(d, 30, &[]) else {
        return Ok(None);
    };
    let scale: f64 = re.abs().max(1.0);
    if im.abs() > 1e-12 * scale || !re.is_finite() {
        return Ok(None);
    }
    let (mut h0, mut h1, mut k0, mut k1): (i128, i128, i128, i128) = (0, 1, 1, 0);
    let mut x: f64 = re;
    for _ in 0..40 {
        let a: f64 = x.floor();
        if a.abs() > 1e15 {
            break;
        }
        let ai: i128 = a as i128;
        let (h2, k2) = (ai * h1 + h0, ai * k1 + k0);
        (h0, h1, k0, k1) = (h1, h2, k1, k2);
        if k1 > 1_000_000_000 {
            break;
        }
        let approx: f64 = h1 as f64 / k1 as f64;
        if (approx - re).abs() <= 1e-12 * scale {
            let candidate: Basic = Basic::from_rational(rat_from_parts(Int::from(h1), Int::from(k1)));
            if crate::core::expr::equals(&candidate, d)? {
                return Ok(Some(candidate));
            }
            return Ok(None);
        }
        let frac: f64 = x - a;
        if frac.abs() < 1e-300 {
            break;
        }
        x = 1.0 / frac;
    }
    Ok(None)
}

fn handle(x: &Basic, symbolic: bool, max_terms: usize) -> PolyResult<Basic> {
    use super::powsimp::{powdenest, powsimp_default};
    if x.args().is_empty() {
        return Ok(x.clone());
    }
    let (mut n, mut d) = fraction(x)?;
    if d.args().is_empty() && n.args().is_empty() {
        return Ok(x.clone());
    }
    if !n.args().is_empty() {
        let args: Vec<Basic> = n
            .args()
            .iter()
            .map(|a| handle(a, symbolic, max_terms))
            .collect::<PolyResult<_>>()?;
        let rebuilt: Basic = crate::core::function::rebuild_evaluated(&n, args)?;
        return unevaluated_mul(&[rebuilt, handle(&inverse(&d)?, symbolic, max_terms)?]);
    }
    if !is_one(&n) {
        return unevaluated_mul(&[n, handle(&inverse(&d)?, symbolic, max_terms)?]);
    }
    if d.is_mul() {
        let parts: Vec<Basic> = d
            .args()
            .iter()
            .map(|f| handle(&inverse(f)?, symbolic, max_terms))
            .collect::<PolyResult<_>>()?;
        return unevaluated_mul(&parts);
    }
    if !symbolic && !d.free_symbols().is_empty() {
        return Ok(x.clone());
    }
    if ispow2(&d, symbolic, false)? && d.is_pow() {
        let d2: Basic = pow(
            &super::sqrtdenest::sqrtdenest(&sqrt(&d.args()[0])?, 3)?,
            &numer_of(&d.args()[1])?,
        )?;
        if d2 != d {
            return handle(&inverse(&d2)?, symbolic, max_terms);
        }
    } else if d.is_pow() && (is_true(&d.args()[1], Predicate::integer) || is_true(&d.args()[0], Predicate::positive)) {
        return pow(&handle(&inverse(&d.args()[0])?, symbolic, max_terms)?, &d.args()[1]);
    }
    if !(d.is_add() || ispow2(&d, symbolic, false)?) {
        let args: Vec<Basic> = d
            .args()
            .iter()
            .map(|a| handle(a, symbolic, max_terms))
            .collect::<PolyResult<_>>()?;
        return inverse(&crate::core::function::rebuild_evaluated(&d, args)?);
    }
    let mut keep: bool = true;
    d = mexpand(&d)?;
    if d.args().is_empty() {
        return inverse(&d);
    }
    if crate::core::assumptions::is_number(&d) {
        if let Some(nd) = rational_equal_to(&d)? {
            return inverse(&nd);
        }
    }
    let i_unit: Basic = Basic::imaginary_unit();
    loop {
        let mut collected: Vec<(Vec<Basic>, Vec<Basic>)> = Vec::new();
        let terms: Vec<Basic> = if d.is_add() { d.args().to_vec() } else { vec![d.clone()] };
        for m in terms {
            let (mut p2, mut other): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
            for i in factors_of(&m) {
                if ispow2(&i, symbolic, true)? && i.is_pow() {
                    let (b, original) = (&i.args()[0], &i.args()[1]);
                    let mut ex: Basic = original.clone();
                    if let Some(r) = original.as_rational() {
                        let (p, q) = (crate::numbers::numer(r), crate::numbers::denom(r));
                        let whole: crate::numbers::Int = crate::numbers::div_floor(&p, &q);
                        let two: crate::numbers::Int = crate::numbers::Int::from(2);
                        if whole != crate::numbers::Int::ZERO && two * (p - whole.clone() * q.clone()) <= q {
                            // b**(3/2) is b*sqrt(b) for every b, not sqrt(b**3)
                            // (sympy/sympy#31275, V233): the integer part of the
                            // exponent is no radical
                            other.push(pow(b, &Basic::integer(whole.clone()))?);
                            ex = crate::core::add::add(&[ex, Basic::integer(-whole)])?;
                        }
                    }
                    p2.push(if ex == Basic::half() {
                        b.clone()
                    } else {
                        pow(b, &mul(&[Basic::int(2), ex])?)?
                    });
                } else if i == i_unit {
                    p2.push(Basic::negative_one());
                } else {
                    other.push(i);
                }
            }
            let key: Vec<Basic> = crate::core::sorting::ordered(p2);
            let value: Basic = mul(&other)?;
            match collected.iter().position(|(k, _)| *k == key) {
                Some(j) => collected[j].1.push(value),
                None => collected.push((key, vec![value])),
            }
        }
        // SymPy's ordered of the tuples of radicands
        let tuples: Vec<Basic> = collected
            .iter()
            .map(|(k, _)| Basic::new(Head::Tuple, k.clone()))
            .collect();
        let order: Vec<Basic> = crate::core::sorting::ordered(tuples);
        let mut rterms: Vec<(Basic, Basic)> = Vec::new();
        for t in &order {
            let (k, v) = collected.iter().find(|(k, _)| k.as_slice() == t.args()).expect("a key");
            rterms.push((mul(k)?, add(v)?));
        }
        let nrad: usize = rterms.len() - usize::from(is_one(&rterms[0].0));
        if nrad < 1 {
            break;
        }
        if nrad > max_terms {
            keep = false;
            break;
        }
        if rterms.len() > 4 {
            let mut all_rational: bool = true;
            for (x, y) in &rterms {
                if !(is_integer_class(x) && pow(y, &Basic::int(2))?.is_rational_class()) {
                    all_rational = false;
                    break;
                }
            }
            if all_rational {
                let terms: Vec<Basic> = rterms
                    .iter()
                    .map(|(x, y)| mul(&[sqrt(x)?, y.clone()]))
                    .collect::<PolyResult<_>>()?;
                let (nd2, d_new) = rad_rationalize(&Basic::one(), &crate::core::add::raw_add(terms))?;
                n = mul(&[n, nd2])?;
                d = d_new;
            } else {
                keep = false;
            }
            break;
        }
        let num: Basic = powsimp_default(&radsimp_numerator(&rterms)?)?;
        n = mul(&[n, num.clone()])?;
        d = mul(&[d, num])?;
        d = powdenest(&mexpand(&d)?, symbolic, false)?;
        let degenerate: bool = d.preorder().iter().any(|t| {
            matches!(t.head(), Head::NaN | Head::ComplexInfinity) || (is_integer_class(t) && is_rational_value(t, 0))
        });
        if degenerate {
            return Ok(x.clone());
        }
        if d.args().is_empty() {
            break;
        }
    }
    if !keep {
        return Ok(x.clone());
    }
    unevaluated_mul(&[n, inverse(&d)?])
}

/// `Expr.normal`: one fraction, a numerical denominator kept outside.
pub(crate) fn normal(e: &Basic) -> PolyResult<Basic> {
    let (n, d) = as_numer_denom(e)?;
    if d.is_number_class() {
        if is_one(&d) {
            return Ok(n);
        }
        return unevaluated_mul(&[n, inverse(&d)?]);
    }
    crate::core::mul::div(&n, &d)
}

fn count(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

/// The denominator rationalized: the square roots (and roots of index a
/// power of two) removed from a denominator with at most `max_terms`
/// radical terms; `symbolic` allows symbolic radicands (`radsimp`).
pub fn radsimp(e: &Basic, symbolic: bool, max_terms: usize) -> PolyResult<Basic> {
    use super::signsimp::signsimp;
    if matches!(e.head(), Head::Tuple) || crate::core::logic::is_boolean(e) {
        return Ok(e.clone());
    }
    let (coeff, rest) = crate::core::expr::as_coeff_add(e);
    let rest: Basic = normal(&rest)?;
    let old: (Basic, Basic) = fraction(&rest)?;
    let (mut n, mut d) = fraction(&handle(&rest, symbolic, max_terms)?)?;
    if old != (n.clone(), d.clone()) {
        if !d.args().is_empty() {
            let was: (Basic, Basic) = (n.clone(), d.clone());
            n = signsimp(&n, false)?;
            d = signsimp(&d, false)?;
            let u: Vec<(Basic, Basic)> = factors_dict(&unevaluated_mul(&[n.clone(), inverse(&d)?])?)?;
            let powers: Vec<Basic> = u.iter().map(|(k, v)| pow(k, v)).collect::<PolyResult<_>>()?;
            let u_expr: Basic = unevaluated_mul(&powers)?;
            (n, d) = fraction(&u_expr)?;
            if old == (n.clone(), d.clone()) {
                (n, d) = was;
            }
        }
        n = crate::core::function::expand_mul(&n, true)?;
        if d.is_number_class() || d.is_add() {
            let (n2, d2) = fraction(&crate::core::exprtools::gcd_terms(
                &unevaluated_mul(&[n.clone(), inverse(&d)?])?,
                false,
                true,
                true,
            )?)?;
            if d2.is_number_class() || count(&d2)? <= count(&d)? {
                n = signsimp(&n2, true)?;
                d = signsimp(&d2, true)?;
                if n.is_mul() && n.args()[0].is_number_class() {
                    n = mul(n.args())?;
                }
            }
        }
    }
    add(&[coeff, unevaluated_mul(&[n, inverse(&d)?])?])
}

// -- collect

/// A function applied to the collected coefficients.
pub type Transform<'a> = &'a dyn Fn(&Basic) -> PolyResult<Basic>;

type Deriv = Option<(Basic, Basic)>;
/// `(expr, rational exponent, symbolic exponent, derivative)`.
type ParsedTerm = (Basic, Basic, Option<Basic>, Deriv);

fn plain_mul(args: Vec<Basic>) -> Basic {
    crate::core::mul::raw_mul(args)
}

fn is_derivative(e: &Basic) -> bool {
    matches!(e.head(), Head::Derivative)
}

/// A derivative whose orders are numbers (a symbolic order is no count).
fn is_counted_derivative(e: &Basic) -> bool {
    is_derivative(e) && !crate::core::function::has_symbolic_order(e)
}

/// The variables of a derivative, repeated by their counts.
fn derivative_variables(d: &Basic) -> Vec<Basic> {
    let mut result: Vec<Basic> = Vec::new();
    for count in &d.args()[1..] {
        let n: usize = count.args()[1]
            .as_rational()
            .and_then(|v| usize::try_from(&numer(v)).ok())
            .unwrap_or(1);
        result.extend(std::iter::repeat(count.args()[0].clone()).take(n));
    }
    result
}

fn parse_derivative(d: &Basic) -> PolyResult<(Basic, (Basic, Basic))> {
    let variables: Vec<Basic> = derivative_variables(d);
    let (mut e, sym) = (d.args()[0].clone(), variables[0].clone());
    let mut order: usize = 1;
    for s in &variables[1..] {
        if *s == sym {
            order += 1;
        } else {
            return Err(crate::error::not_ported("Improve MV Derivative support in collect"));
        }
    }
    while is_counted_derivative(&e) {
        let inner: Vec<Basic> = derivative_variables(&e);
        let s0: Basic = inner[0].clone();
        if inner.iter().any(|s| *s != s0) {
            return Err(crate::error::not_ported("Improve MV Derivative support in collect"));
        }
        if s0 == sym {
            order += inner.len();
            e = e.args()[0].clone();
        } else {
            break;
        }
    }
    Ok((e, (sym, Basic::int(order as i64))))
}

fn parse_term(e: &Basic) -> PolyResult<ParsedTerm> {
    use super::powsimp::as_coeff_mul;
    use crate::functions::elementary::exponential::exp;
    let mut rat_expo: Basic = Basic::one();
    let mut sym_expo: Option<Basic> = None;
    let mut sexpr: Basic = e.clone();
    let mut deriv: Deriv = None;
    if e.is_pow() {
        let (base, ex) = (&e.args()[0], &e.args()[1]);
        if is_counted_derivative(base) {
            let (s, d) = parse_derivative(base)?;
            sexpr = s;
            deriv = Some(d);
        } else {
            sexpr = base.clone();
        }
        if matches!(base.head(), Head::Exp1) {
            if ex.is_rational_class() {
                sexpr = Basic::e();
                rat_expo = ex.clone();
            } else if ex.is_mul() {
                let (coeff, tail) = as_coeff_mul(ex, true)?;
                sexpr = exp(&tail)?;
                rat_expo = coeff;
            }
        } else if ex.is_number_class() {
            rat_expo = ex.clone();
        } else {
            let (coeff, tail) = as_coeff_mul(ex, false)?;
            if coeff.is_number_class() {
                rat_expo = coeff;
                sym_expo = Some(tail);
            } else {
                sym_expo = Some(ex.clone());
            }
        }
    } else if e.is_function(Function::Exp) {
        let arg: &Basic = &e.args()[0];
        if arg.is_rational_class() {
            sexpr = Basic::e();
            rat_expo = arg.clone();
        } else if arg.is_mul() {
            let (coeff, tail) = as_coeff_mul(arg, true)?;
            sexpr = exp(&tail)?;
            rat_expo = coeff;
        }
    } else if is_counted_derivative(e) {
        let (s, d) = parse_derivative(e)?;
        sexpr = s;
        deriv = Some(d);
    }
    Ok((sexpr, rat_expo, sym_expo, deriv))
}

fn make_expression(terms: &[ParsedTerm]) -> PolyResult<Basic> {
    let mut product: Vec<Basic> = Vec::new();
    for (term, rat, sym, deriv) in terms {
        let mut term: Basic = term.clone();
        if let Some((var, order)) = deriv {
            let Some(n) = order.as_rational().filter(|_| is_integer_class(order)).map(numer) else {
                return Err(crate::error::PolyError::Type("an integer is expected".to_string()));
            };
            let n: usize = usize::try_from(&n).unwrap_or(0);
            for _ in 0..n {
                term = crate::core::function::derivative_node(&term, &[(var.clone(), 1)]);
            }
        }
        match sym {
            None => product.push(if is_one(rat) { term } else { pow(&term, rat)? }),
            Some(s) => product.push(pow(&term, &mul(&[rat.clone(), s.clone()])?)?),
        }
    }
    mul(&product)
}

fn matches_pattern(e: &Basic, pattern: &Basic) -> PolyResult<bool> {
    Ok(crate::core::matching::match_pattern(e, pattern, &[], false)?.is_some())
}

type Parsed = (Vec<ParsedTerm>, Vec<ParsedTerm>, Option<Basic>, bool);

fn parse_expression(terms: &[Option<ParsedTerm>], pattern: &Basic, exact: bool) -> PolyResult<Option<Parsed>> {
    let pattern_args: Vec<Basic> = factors_of(pattern);
    if terms.iter().filter(|t| t.is_some()).count() < pattern_args.len() {
        return Ok(None);
    }
    let parsed_pattern: Vec<ParsedTerm> = pattern_args.iter().map(parse_term).collect::<PolyResult<_>>()?;
    let mut pending: Vec<Option<ParsedTerm>> = terms.to_vec();
    let mut elems: Vec<ParsedTerm> = Vec::new();
    let mut common_expo: Option<Basic> = None;
    let mut has_deriv: bool = false;
    for (elem, e_rat, e_sym, e_ord) in &parsed_pattern {
        if elem.is_number_class() && is_one(e_rat) && e_sym.is_none() {
            continue;
        }
        let mut matched: bool = false;
        for j in 0..pending.len() {
            let Some(current) = pending[j].clone() else { continue };
            let (term, t_rat, t_sym, t_ord) = &current;
            if t_ord.is_some() {
                has_deriv = true;
            }
            let symbols_match: bool = match (t_sym, e_sym) {
                (None, None) => true,
                (Some(a), Some(b)) => a == b || matches_pattern(a, b)?,
                _ => false,
            };
            if matches_pattern(term, elem)? && symbols_match {
                if !exact {
                    let expo: Basic = mul(&[t_rat.clone(), inverse(e_rat)?])?;
                    common_expo = match common_expo {
                        None => Some(expo),
                        Some(c) if c != expo => Some(Basic::one()),
                        Some(c) => Some(c),
                    };
                } else if e_rat != t_rat || e_ord != t_ord {
                    continue;
                }
                elems.push(current);
                pending[j] = None;
                matched = true;
                break;
            }
        }
        if !matched {
            return Ok(None);
        }
    }
    Ok(Some((
        pending.into_iter().flatten().collect(),
        elems,
        common_expo,
        has_deriv,
    )))
}

fn expand_power_base(e: &Basic) -> PolyResult<Basic> {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: true,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, false, hints)
}

/// `has_free`: a pattern whose symbols are free in `e` and whose
/// substitution changes `e`.
fn has_free(e: &Basic, patterns: &[Basic]) -> PolyResult<bool> {
    let free = e.free_symbols();
    for pattern in patterns {
        let symbols = pattern.free_symbols();
        if !symbols.is_empty() && !symbols.iter().all(|s| free.contains(s)) {
            continue;
        }
        let dummy: Basic = crate::core::symbol::Dummy::new("free", &[]).0;
        if crate::core::subs::subs(e, pattern, &dummy)? != *e {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The result of `collect`: an expression, or the coefficients by power.
pub enum Collected {
    Expr(Basic),
    Dict(Vec<(Basic, Basic)>),
}

fn terms_of_sum(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

fn collected_expr(e: &Basic, syms: &[Basic], func: Option<Transform>, exact: Option<bool>) -> PolyResult<Basic> {
    match collect_inner(e, syms, func, true, exact)? {
        Collected::Expr(b) => Ok(b),
        Collected::Dict(_) => Err(crate::error::PolyError::Type("an expression is expected".to_string())),
    }
}

fn collect_inner(
    e: &Basic,
    syms: &[Basic],
    func: Option<Transform>,
    evaluate: bool,
    exact: Option<bool>,
) -> PolyResult<Collected> {
    let simple = |x: &Basic| -> PolyResult<bool> {
        Ok(x.is_symbol()
            || crate::core::mul::neg(x)?.is_symbol()
            || x.preorder().iter().any(|n| matches!(n.head(), Head::Wild(_))))
    };
    let mut nonsyms: Vec<Basic> = Vec::new();
    for s in syms {
        if !simple(s)? {
            nonsyms.push(s.clone());
        }
    }
    if !nonsyms.is_empty() {
        let reps: Vec<(Basic, Basic)> = nonsyms
            .iter()
            .map(|s| (s.clone(), crate::core::symbol::Dummy::new("Dummy", &[]).0))
            .collect();
        let replaced: Vec<Basic> = syms
            .iter()
            .map(|s| {
                reps.iter()
                    .find(|(k, _)| k == s)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(|| s.clone())
            })
            .collect();
        let rv: Collected = collect_inner(
            &crate::core::subs::subs_mapping(e, &reps)?,
            &replaced,
            func,
            evaluate,
            exact,
        )?;
        let urep: std::collections::HashMap<Basic, Basic> = reps.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
        let back = |b: &Basic| crate::core::function::xreplace_evaluated(b, &urep);
        return Ok(match rv {
            Collected::Expr(b) => Collected::Expr(back(&b)?),
            Collected::Dict(d) => Collected::Dict(
                d.iter()
                    .map(|(k, v)| Ok((back(urep.get(k).unwrap_or(k))?, back(v)?)))
                    .collect::<PolyResult<_>>()?,
            ),
        });
    }
    let mut syms: Vec<Basic> = syms.to_vec();
    let mut e: Basic = e.clone();
    if exact.is_none() {
        let mut found: Vec<Basic> = Vec::new();
        for i in terms_of_sum(&e) {
            if is_non_expr(&i) || !has_free(&i, &syms)? || syms.contains(&i) {
                continue;
            }
            if !i.is_mul() && !syms.contains(&i) {
                if !found.contains(&i) {
                    found.push(i);
                }
            } else {
                let mut dependent: Vec<Basic> = Vec::new();
                for f in factors_of(&i) {
                    if has_free(&f, &syms)? {
                        dependent.push(f);
                    }
                }
                let g: Basic = plain_mul(dependent);
                if !syms.contains(&g) && !found.contains(&g) {
                    found.push(g);
                }
            }
        }
        let is_simple: bool = found.iter().all(|i| i.is_pow() && syms.contains(&i.args()[0]));
        syms.extend(crate::core::sorting::ordered(found));
        if !is_simple {
            return collect_inner(&e, &syms, func, evaluate, Some(false));
        }
    }
    if evaluate {
        if e.is_add() {
            let o: Option<Basic> = crate::core::order::get_o(&e)?;
            let parts: Vec<Basic> = e
                .args()
                .iter()
                .filter(|a| o.as_ref() != Some(*a))
                .map(|a| collected_expr(a, &syms, func, exact))
                .collect::<PolyResult<_>>()?;
            e = add(&[add(&parts)?, o.unwrap_or_else(Basic::zero)])?;
        } else if e.is_mul() {
            let parts: Vec<Basic> = e
                .args()
                .iter()
                .map(|t| collected_expr(t, &syms, func, exact))
                .collect::<PolyResult<_>>()?;
            return Ok(Collected::Expr(mul(&parts)?));
        } else if e.is_pow() {
            return Ok(Collected::Expr(pow(
                &collected_expr(&e.args()[0], &syms, func, exact)?,
                &e.args()[1],
            )?));
        }
    }
    let syms: Vec<Basic> = syms.iter().map(expand_power_base).collect::<PolyResult<_>>()?;
    // distribute_order_term: the order term free of the symbols added to
    // every coefficient
    let mut order_term: Option<Basic> = crate::core::order::get_o(&e)?;
    if let Some(o) = &order_term {
        if syms.iter().any(|sy| o.has(sy)) {
            order_term = None;
        } else {
            e = crate::core::order::remove_o(&e)?;
        }
    }
    let summa: Vec<Basic> = terms_of_sum(&e)
        .iter()
        .filter(|i| !is_non_expr(i))
        .map(expand_power_base)
        .collect::<PolyResult<_>>()?;
    let mut collected: Vec<(Basic, Vec<Basic>)> = Vec::new();
    let mut disliked: Basic = Basic::zero();
    for product in summa {
        let parts: Vec<Basic> = factors_of(&product);
        let c: Vec<Basic> = parts
            .iter()
            .filter(|f| !is_false(f, Predicate::commutative))
            .cloned()
            .collect();
        let nc: Vec<Basic> = parts
            .iter()
            .filter(|f| is_false(f, Predicate::commutative))
            .cloned()
            .collect();
        let mut args: Vec<Basic> = crate::core::sorting::ordered(c);
        args.extend(nc);
        let mut terms: Vec<Option<ParsedTerm>> = args
            .iter()
            .map(|i| parse_term(i).map(Some))
            .collect::<PolyResult<_>>()?;
        let mut small_first: bool = true;
        let mut placed: bool = false;
        for symbol in &syms {
            if is_derivative(symbol) && small_first {
                terms.reverse();
                small_first = !small_first;
            }
            let Some((rest_terms, elems, _, has_deriv)) = parse_expression(&terms, symbol, exact.unwrap_or(false))?
            else {
                continue;
            };
            if is_false(symbol, Predicate::commutative) {
                return Err(crate::error::PolyError::Type(
                    "Can not collect noncommutative symbol".to_string(),
                ));
            }
            let index: Basic = if !has_deriv {
                let mut margs: Vec<Basic> = Vec::new();
                for elem in &elems {
                    let expo: Basic = match &elem.2 {
                        None => elem.1.clone(),
                        Some(s) => mul(&[elem.1.clone(), s.clone()])?,
                    };
                    margs.push(pow(&elem.0, &expo)?);
                }
                mul(&margs)?
            } else {
                make_expression(&elems)?
            };
            let remaining: Basic = expand_power_base(&make_expression(&rest_terms)?)?;
            let index: Basic = expand_power_base(&index)?;
            match collected.iter().position(|(k, _)| *k == index) {
                Some(p) => collected[p].1.push(remaining),
                None => collected.push((index, vec![remaining])),
            }
            placed = true;
            break;
        }
        if !placed {
            disliked = add(&[disliked, product])?;
        }
    }
    let mut result: Vec<(Basic, Basic)> = collected
        .into_iter()
        .map(|(k, v)| Ok((k, add(&v)?)))
        .collect::<PolyResult<_>>()?;
    if !(is_integer_class(&disliked) && is_rational_value(&disliked, 0)) {
        match result.iter().position(|(k, _)| *k == Basic::one()) {
            Some(p) => result[p].1 = disliked,
            None => result.push((Basic::one(), disliked)),
        }
    }
    if let Some(o) = &order_term {
        result = result
            .into_iter()
            .map(|(k, v)| Ok((k, add(&[v, o.clone()])?)))
            .collect::<PolyResult<_>>()?;
    }
    if let Some(f) = func {
        result = result
            .into_iter()
            .map(|(k, v)| Ok((k, f(&v)?)))
            .collect::<PolyResult<_>>()?;
    }
    if evaluate {
        let terms: Vec<Basic> = result
            .iter()
            .map(|(k, v)| mul(&[k.clone(), v.clone()]))
            .collect::<PolyResult<_>>()?;
        return Ok(Collected::Expr(add(&terms)?));
    }
    Ok(Collected::Dict(result))
}

fn is_non_expr(e: &Basic) -> bool {
    matches!(e.head(), Head::Tuple) || crate::core::logic::is_boolean(e)
}

/// The terms of a sum grouped by the powers of `syms` (`collect`, the
/// order term distributed).
pub fn collect(e: &Basic, syms: &[Basic], func: Option<Transform>, exact: Option<bool>) -> PolyResult<Basic> {
    collected_expr(e, syms, func, exact)
}

/// SymPy's `collect(..., evaluate=False)`: the coefficients of the powers
/// of `syms`, the key `1` for the rest (`collect_dict`).
pub fn collect_dict(
    e: &Basic,
    syms: &[Basic],
    func: Option<Transform>,
    exact: Option<bool>,
) -> PolyResult<Vec<(Basic, Basic)>> {
    match collect_inner(e, syms, func, false, exact)? {
        Collected::Dict(d) => Ok(d),
        Collected::Expr(_) => Err(crate::error::PolyError::Type("a dictionary is expected".to_string())),
    }
}

/// `collect` applied to every sum of the tree, bottom up (`rcollect`).
pub fn rcollect(e: &Basic, vars: &[Basic]) -> PolyResult<Basic> {
    if e.args().is_empty() || !vars.iter().any(|v| e.has(v)) {
        return Ok(e.clone());
    }
    let args: Vec<Basic> = e
        .args()
        .iter()
        .map(|a| {
            if is_non_expr(a) {
                Ok(a.clone())
            } else {
                rcollect(a, vars)
            }
        })
        .collect::<PolyResult<_>>()?;
    let rebuilt: Basic = crate::core::function::rebuild_evaluated(e, args)?;
    if rebuilt.is_add() {
        return collect(&rebuilt, vars, None, Some(false));
    }
    Ok(rebuilt)
}
