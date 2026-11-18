//! Port of `sympy_extras/simplify/powsimp.py` (itself ported from SymPy's
//! `sympy.simplify.powsimp`): `powsimp` combines the powers of equal bases
//! (`combine = Exp`), of equal exponents (`Base`) or both (`All`);
//! `powdenest` denests powers of powers where valid. The Rust core has no
//! polar numbers: `is_polar` never holds, and `powdenest` with `polar` is
//! not implemented.

#![allow(clippy::mutable_key_type, clippy::collapsible_if)]

use super::simplify::{logcombine, posify, OrderedMap};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_base_exp, as_numer_denom, factors_of, is, is_false, is_integer_class, is_true, terms_of};
use crate::core::sorting::{ordered, sort_key};
use crate::core::symbol::Dummy;
use crate::error::{not_ported, PolyResult};
use crate::functions::elementary::exponential::{exp, log};
use crate::numbers::*;

/// What `powsimp` combines.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Combine {
    All,
    Exp,
    Base,
}

/// The measure of the bases tried first (`count_ops` by default).
pub type Measure<'a> = &'a dyn Fn(&Basic) -> PolyResult<i64>;

fn count_ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

thread_local! {
    static Y: Basic = Dummy::new("y", &[]).0;
}

fn y() -> Basic {
    Y.with(|y| y.clone())
}

fn mul(v: &[Basic]) -> PolyResult<Basic> {
    crate::core::mul::mul(v)
}

fn mul2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), b.clone()])
}

fn add(v: &[Basic]) -> PolyResult<Basic> {
    crate::core::add::add(v)
}

fn add2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), b.clone()])
}

fn pow(b: &Basic, e: &Basic) -> PolyResult<Basic> {
    crate::core::power::pow(b, e)
}

fn neg(a: &Basic) -> PolyResult<Basic> {
    crate::core::mul::neg(a)
}

fn int(n: &Int) -> Basic {
    Basic::integer(n.clone())
}

fn is_exp(e: &Basic) -> bool {
    e.is_function(Function::Exp)
}

fn is_log(e: &Basic) -> bool {
    e.is_function(Function::Log)
}

fn is_one(e: &Basic) -> bool {
    e.as_rational().is_some_and(|v| v.is_one())
}

fn is_rational(e: &Basic) -> bool {
    e.is_rational_class()
}

fn is_polar(e: &Basic) -> bool {
    is_true(e, Predicate::polar)
}

fn integer_of(e: &Basic) -> Option<Int> {
    if is_integer_class(e) {
        e.as_rational().map(numer)
    } else {
        None
    }
}

/// `as_coeff_Mul(rational)` as the Python core has it.
pub(crate) fn as_coeff_mul(e: &Basic, rational: bool) -> PolyResult<(Basic, Basic)> {
    if !rational {
        return Ok(crate::core::expr::as_coeff_mul(e));
    }
    if e.is_mul() {
        let first: &Basic = &e.args()[0];
        if first.is_number_node() && !matches!(first.head(), Head::ComplexInfinity) {
            if is_rational(first) {
                return Ok((first.clone(), crate::core::mul::raw_mul(e.args()[1..].to_vec())));
            }
            if is_true(first, Predicate::extended_negative) {
                let mut rest: Vec<Basic> = vec![neg(first)?];
                rest.extend(e.args()[1..].iter().cloned());
                return Ok((Basic::negative_one(), crate::core::mul::raw_mul(rest)));
            }
        }
        return Ok((Basic::one(), e.clone()));
    }
    Ok(crate::core::expr::as_coeff_mul(e))
}

fn expand_log(e: &Basic) -> PolyResult<Basic> {
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: true,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(e, true, hints)
}

fn int_remove(n: &Int, p: &Int) -> i64 {
    let mut n: Int = n.clone();
    let mut m: i64 = 0;
    while (&n % p).is_zero() {
        n = div_floor(&n, p);
        m += 1;
    }
    m
}

/// The greatest `m` with `p**m` dividing `n`, for rationals
/// (`multiplicity`).
pub fn multiplicity(p: &Rat, n: &Rat) -> PolyResult<i64> {
    let (pp, pq, np, nq) = (numer(p), denom(p), numer(n), denom(n));
    let one: Int = Int::ONE;
    if pq == one && nq == one {
        if np.is_zero() {
            return Err(crate::error::value_error(
                "no such integer exists: multiplicity of 0 is not defined".to_string(),
            ));
        }
        return Ok(if abs(&pp) > one {
            int_remove(&abs(&np), &abs(&pp))
        } else {
            0
        });
    }
    if pq == one {
        if np == one {
            return Ok(-int_remove(&nq, &pp));
        }
        return Ok(int_remove(&abs(&np), &pp) - int_remove(&nq, &pp));
    }
    if pp == one {
        return Ok(int_remove(&nq, &pq));
    }
    let like: i64 = int_remove(&abs(&np), &pp).min(int_remove(&nq, &pq));
    let cross: i64 = int_remove(&abs(&np), &pq).min(int_remove(&nq, &pp));
    Ok(like - cross)
}

fn number_gcd(a: &Basic, b: &Basic) -> Basic {
    match (a.as_rational(), b.as_rational()) {
        (Some(x), Some(y)) if is_rational(a) && is_rational(b) => {
            let g: Int = crate::core::intfunc::igcd(&[numer(x), numer(y)]);
            let (q1, q2) = (denom(x), denom(y));
            let gq: Int = crate::core::intfunc::igcd(&[q1.clone(), q2.clone()]);
            let l: Int = div_floor(&(&q1 * &q2), &gq);
            Basic::from_rational(rat_from_parts(g, l))
        }
        _ => Basic::one(),
    }
}

/// `(b**e1)**e2` denested where valid (`_denest_pow`).
fn denest_pow(eq: &Basic) -> PolyResult<Basic> {
    let mut eq: Basic = eq.clone();
    let (mut b, mut e) = as_base_exp(&eq)?;
    if (b.is_pow() || is_exp(&b)) && !is_one(&e) {
        // an exponential has no power rule of its own (`_eval_power_node`)
        if b.is_pow() {
            let candidate: Basic = pow(&b, &e)?;
            if !(candidate.is_pow() && candidate.args()[0] == b) {
                eq = candidate;
                (b, e) = as_base_exp(&eq)?;
            }
        }
    }
    if matches!(b.head(), Head::Exp1) && e.is_mul() {
        let (mut logs, mut other): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
        for ei in e.args() {
            if terms_of(ei).iter().any(is_log) {
                logs.push(ei.clone());
            } else {
                other.push(ei.clone());
            }
        }
        return pow(&exp(&logcombine(&mul(&logs)?, false)?)?, &mul(&other)?);
    }
    let (_, be) = as_base_exp(&b)?;
    let non_integer_rational: bool = b.as_rational().is_some_and(|v| !v.is_int()) && is_rational(&b);
    if is_one(&be) && !(b.is_mul() || non_integer_rational || is_true(&b, Predicate::positive)) {
        return Ok(eq);
    }
    let factors: Vec<Basic> = factors_of(&b);
    let polars: Vec<(Basic, Basic)> = factors
        .iter()
        .filter(|f| is_polar(f))
        .map(as_base_exp)
        .collect::<PolyResult<_>>()?;
    let nonpolars: Vec<Basic> = factors.iter().filter(|f| !is_polar(f)).cloned().collect();
    if polars.len() == 1 && !polars[0].0.is_mul() {
        return mul(&[
            pow(&polars[0].0, &mul2(&polars[0].1, &e)?)?,
            pow(&mul(&nonpolars)?, &e)?,
        ]);
    }
    if !polars.is_empty() {
        let mut parts: Vec<Basic> = polars
            .iter()
            .map(|(bb, ee)| pow(bb, &mul2(ee, &e)?))
            .collect::<PolyResult<_>>()?;
        parts.push(pow(&mul(&nonpolars)?, &e)?);
        return mul(&parts);
    }
    if is_integer_class(&b) {
        let logb: Basic = expand_log(&log(&b)?)?;
        if logb.is_mul() && logb.args().len() == 2 {
            let (c, logarithm) = (&logb.args()[0], &logb.args()[1]);
            if !logarithm.args().is_empty() {
                return pow(&logarithm.args()[0], &mul2(&e, c)?);
            }
        }
    }
    if !b.is_mul() || factors.iter().any(|s| s.args().is_empty()) {
        return Ok(eq);
    }
    let nc_gcd = |aa: &Basic, bb: &Basic| -> PolyResult<Basic> {
        let (a0, a1) = as_coeff_mul(aa, false)?;
        let (b0, b1) = as_coeff_mul(bb, false)?;
        let c: Basic = as_numer_denom(&number_gcd(&a0, &b0))?.0;
        let right: Vec<Basic> = factors_of(&b1)
            .into_iter()
            .filter(|f| !is_false(f, Predicate::commutative))
            .collect();
        let mut common: Vec<Basic> = Vec::new();
        for f in factors_of(&a1) {
            if !is_false(&f, Predicate::commutative) && right.contains(&f) && !common.contains(&f) {
                common.push(f);
            }
        }
        crate::core::mul::keep_coeff(&c, &mul(&common)?)
    };
    let mut glogb: Basic = expand_log(&log(&b)?)?;
    if glogb.is_add() {
        let args: Vec<Basic> = glogb.args().to_vec();
        let mut g: Basic = args[0].clone();
        for a in &args[1..] {
            g = nc_gcd(&g, a)?;
        }
        if !is_one(&g) {
            let (cg, rg) = as_coeff_mul(&g, false)?;
            let inverse: Basic = pow(&g, &Basic::negative_one())?;
            let terms: Vec<Basic> = args.iter().map(|a| mul2(a, &inverse)).collect::<PolyResult<_>>()?;
            glogb = crate::core::mul::keep_coeff(&cg, &mul2(&rg, &add(&terms)?)?)?;
        }
    }
    if is_log(&glogb) || !glogb.is_mul() {
        let inner: Basic = if glogb.args().is_empty() {
            glogb.clone()
        } else {
            glogb.args()[0].clone()
        };
        if inner.is_pow() || is_exp(&inner) {
            let denested: Basic = denest_pow(&inner)?;
            let (db, de) = as_base_exp(&denested)?;
            let size: Basic = crate::functions::elementary::complexes::abs_(&de)?;
            if matches!(
                crate::core::relational::lt(&size, &Basic::one())?.head(),
                Head::BooleanTrue
            ) {
                return pow(&db, &mul2(&de, &e)?);
            }
        }
        return Ok(eq);
    }
    let adds: Vec<Basic> = glogb.args().iter().filter(|a| a.is_add()).cloned().collect();
    let mut others: Vec<Basic> = vec![e.clone()];
    others.extend(glogb.args().iter().filter(|a| !a.is_add()).cloned());
    pow(&exp(&logcombine(&mul(&adds)?, false)?)?, &mul(&others)?)
}

/// `rule` applied top-down to the nodes it accepts, the rebuilt nodes
/// evaluated (`_xreplace_evaluated` with a `Transform`).
fn xreplace_transform(
    e: &Basic,
    accept: &dyn Fn(&Basic) -> bool,
    rule: &dyn Fn(&Basic) -> PolyResult<Basic>,
) -> PolyResult<Basic> {
    if accept(e) {
        return rule(e);
    }
    if e.args().is_empty() {
        return Ok(e.clone());
    }
    let args: Vec<Basic> = e
        .args()
        .iter()
        .map(|a| xreplace_transform(a, accept, rule))
        .collect::<PolyResult<_>>()?;
    if args.as_slice() == e.args() {
        return Ok(e.clone());
    }
    crate::core::function::rebuild_evaluated(e, args)
}

/// Powers of powers denested where valid (`powdenest`).
pub fn powdenest(eq: &Basic, force: bool, polar: bool) -> PolyResult<Basic> {
    let mut e: Basic = eq.clone();
    if force {
        fn denest(b: &Basic, ex: &Basic) -> PolyResult<(Option<bool>, Basic)> {
            if !(b.is_pow() || is_exp(b)) {
                return Ok((
                    is(b, Predicate::positive),
                    Basic::new(Head::Pow, vec![b.clone(), ex.clone()]),
                ));
            }
            let (bb, be) = as_base_exp(b)?;
            denest(&bb, &mul2(&be, ex)?)
        }
        let mut reps: Vec<(Basic, Basic)> = Vec::new();
        for p in e.atoms(Some(&|n: &Basic| n.is_pow())) {
            let base: &Basic = &p.args()[0];
            if base.is_pow() || is_exp(base) {
                let (ok, dp) = denest(base, &p.args()[1])?;
                if ok != Some(false) {
                    reps.push((p.clone(), dp));
                }
            }
        }
        if !reps.is_empty() {
            e = crate::core::subs::subs_mapping(&e, &reps)?;
        }
        let (positive, back) = posify(&e)?;
        return crate::core::function::xreplace_evaluated(&powdenest(&positive, false, polar)?, &back);
    }
    if polar {
        return Err(not_ported("powdenest(polar=True): the Rust core has no polar numbers"));
    }
    let new: Basic = powsimp(&e, false, Combine::All, false, &count_ops)?;
    xreplace_transform(&new, &|m: &Basic| m.is_pow() || is_exp(m), &denest_pow)
}

fn is_non_expr(w: &Basic) -> bool {
    matches!(w.head(), Head::Tuple) || crate::core::logic::is_boolean(w)
}

/// Powers with equal bases or exponents combined (`powsimp`).
pub fn powsimp(expr: &Basic, deep: bool, combine: Combine, force: bool, measure: Measure) -> PolyResult<Basic> {
    let recurse = |arg: &Basic, deep_: Option<bool>, combine_: Option<Combine>| -> PolyResult<Basic> {
        powsimp(arg, deep_.unwrap_or(deep), combine_.unwrap_or(combine), force, measure)
    };
    if expr.args().is_empty() {
        return Ok(expr.clone());
    }
    fn recurse_arg(w: &Basic, recurse: &dyn Fn(&Basic) -> PolyResult<Basic>) -> PolyResult<Basic> {
        if !is_non_expr(w) {
            return recurse(w);
        }
        if w.args().is_empty() {
            return Ok(w.clone());
        }
        let args: Vec<Basic> = w
            .args()
            .iter()
            .map(|a| recurse_arg(a, recurse))
            .collect::<PolyResult<_>>()?;
        crate::core::function::rebuild_evaluated(w, args)
    }
    let mut e: Basic = expr.clone();
    let y: Basic = y();
    if deep || e.is_add() || (e.is_mul() && !e.args().contains(&y)) {
        let plain = |a: &Basic| recurse(a, None, None);
        let args: Vec<Basic> = e
            .args()
            .iter()
            .map(|w| recurse_arg(w, &plain))
            .collect::<PolyResult<_>>()?;
        e = crate::core::function::rebuild_evaluated(&e, args)?;
    }
    if e.is_pow() {
        return mul2(
            &recurse(&mul2(&e, &y)?, Some(false), None)?,
            &pow(&y, &Basic::negative_one())?,
        );
    }
    if !e.is_mul() {
        return Ok(e);
    }
    if matches!(combine, Combine::Exp | Combine::All) {
        return combine_exponents(&e, deep, combine, &recurse, measure);
    }
    combine_bases(&e, deep, force, &recurse)
}

type Recurse<'a> = &'a dyn Fn(&Basic, Option<bool>, Option<Combine>) -> PolyResult<Basic>;
type Key = (Basic, Int);

/// `((b**s, q), p)` for `e = p/q * s` (`_bkey`).
fn bkey(b: &Basic, e: Option<&Basic>) -> PolyResult<(Key, Basic)> {
    let Some(e) = e else {
        let (b, e) = as_base_exp(b)?;
        return bkey(&b, Some(&e));
    };
    if is_integer_class(e) {
        return Ok(((b.clone(), Int::ONE), e.clone()));
    }
    if let Some(v) = e.as_rational().filter(|_| is_rational(e)) {
        return Ok(((b.clone(), denom(v)), int(&numer(v))));
    }
    let (c, m) = as_coeff_mul(e, true)?;
    if !is_one(&c) && is_rational(&c) {
        let v: Rat = c.as_rational().expect("a rational").clone();
        if is_true(&m, Predicate::integer) {
            return Ok(((b.clone(), denom(&v)), mul2(&m, &int(&numer(&v)))?));
        }
        return Ok(((pow(b, &m)?, denom(&v)), int(&numer(&v))));
    }
    Ok(((pow(b, e)?, Int::ONE), Basic::one()))
}

fn combine_exponents(e: &Basic, deep: bool, combine: Combine, recurse: Recurse, measure: Measure) -> PolyResult<Basic> {
    let mut c_lists: OrderedMap<Basic, Vec<Basic>> = OrderedMap::new();
    let mut nc_part: Vec<Basic> = Vec::new();
    let mut coeff: Basic = Basic::one();
    for term0 in e.args() {
        let mut term: Basic = term0.clone();
        if is_rational(&term) {
            coeff = mul2(&coeff, &term)?;
            continue;
        }
        if term.is_pow() {
            term = denest_pow(&term)?;
        }
        if !is_false(&term, Predicate::commutative) {
            let (mut b, mut ex) = as_base_exp(&term)?;
            if deep {
                b = recurse(&b, None, None)?;
                ex = recurse(&ex, None, None)?;
            }
            if b.is_pow() || is_exp(&b) {
                b = pow(&b, &ex)?;
                ex = Basic::one();
            }
            c_lists.entry(&b, Vec::new).push(ex);
        } else {
            if let Some(last) = nc_part.last() {
                let (b1, e1) = as_base_exp(last)?;
                let (b2, e2) = as_base_exp(&term)?;
                if b1 == b2 && !is_false(&e1, Predicate::commutative) && !is_false(&e2, Predicate::commutative) {
                    let n: usize = nc_part.len();
                    nc_part[n - 1] = pow(&b1, &add2(&e1, &e2)?)?;
                    continue;
                }
            }
            nc_part.push(term);
        }
    }
    let mut c_powers: OrderedMap<Basic, Basic> = OrderedMap::new();
    let keys: Vec<Basic> = ordered(c_lists.0.iter().map(|(k, _)| k.clone()).collect());
    for b in keys {
        let mut exps: Vec<Basic> = c_lists.get(&b).expect("a key").clone();
        let minus_one: Basic = Basic::negative_one();
        if is_rational(&b)
            && !b.as_rational().expect("a rational").is_zero()
            && !exps.iter().all(|ei| ei.is_number_class())
            && !is_one(&coeff)
            && b != Basic::one()
            && b != minus_one
            && is_rational(&coeff)
        {
            let absolute = |r: &Basic| -> Rat {
                let v: Rat = r.as_rational().expect("a rational").clone();
                if rat_is_negative(&v) {
                    -v
                } else {
                    v
                }
            };
            let m: i64 = multiplicity(&absolute(&b), &absolute(&coeff))?;
            if m != 0 {
                exps.push(Basic::int(m));
                coeff = mul2(&coeff, &pow(&b, &Basic::int(-m))?)?;
            }
        }
        c_powers.set(b, add(&exps)?);
    }
    if !is_one(&coeff) {
        let value: Basic = match c_powers.get(&coeff) {
            Some(v) => add2(v, &Basic::one())?,
            None => Basic::one(),
        };
        c_powers.set(coeff.clone(), value);
    }
    let mut skip: Vec<Basic> = Vec::new();
    for (b, ex) in c_powers.0.clone() {
        if skip.contains(&b) || !c_powers.contains(&b) {
            continue;
        }
        if is_true(&b, Predicate::positive) || is_polar(&b) {
            let binv: Basic = pow(&b, &Basic::negative_one())?;
            if b != binv && c_powers.contains(&binv) {
                if is_one(&as_numer_denom(&b)?.0) {
                    c_powers.pop(&b);
                    let v: Basic = add2(c_powers.get(&binv).expect("present"), &neg(&ex)?)?;
                    c_powers.set(binv, v);
                } else {
                    skip.push(binv.clone());
                    let removed: Basic = c_powers.pop(&binv).expect("present");
                    let v: Basic = add2(c_powers.get(&b).expect("present"), &neg(&removed)?)?;
                    c_powers.set(b, v);
                }
            }
        }
    }
    let minus_one: Basic = Basic::negative_one();
    for (b, ex) in c_powers.0.clone() {
        let nb: Basic = neg(&b)?;
        if (b.is_symbol() || b.is_add()) && c_powers.contains(&nb) && c_powers.contains(&b) {
            if is(&b, Predicate::positive).is_some() || is_true(&ex, Predicate::integer) {
                let moved: Basic;
                if is_true(&ex, Predicate::integer) || is_true(&b, Predicate::negative) {
                    let popped: Basic = c_powers.pop(&b).expect("present");
                    let v: Basic = add2(c_powers.get(&nb).expect("present"), &popped)?;
                    c_powers.set(nb, v);
                    moved = ex.clone();
                } else {
                    moved = c_powers.pop(&nb).expect("present");
                    let v: Basic = add2(c_powers.get(&b).expect("present"), &moved)?;
                    c_powers.set(b.clone(), v);
                }
                let v: Basic = match c_powers.get(&minus_one) {
                    Some(old) => add2(old, &moved)?,
                    None => moved,
                };
                c_powers.set(minus_one.clone(), v);
            }
        }
    }
    let pairs: Vec<(Basic, Basic)> = c_powers
        .0
        .into_iter()
        .filter(|(_, ex)| !(ex.is_number_class() && ex.as_rational().is_some_and(|v| v.is_zero())))
        .collect();

    let mut common_b: OrderedMap<Key, Basic> = OrderedMap::new();
    let mut bases: Vec<Key> = Vec::new();
    let mut done: Vec<(Basic, Basic)> = Vec::new();
    for (b0, e0) in &pairs {
        let (key, ex) = bkey(b0, Some(e0))?;
        let value: Basic = match common_b.get(&key) {
            Some(old) => add2(old, &ex)?,
            None => ex,
        };
        common_b.set(key.clone(), value);
        if !key.1.is_one() && key.0.is_mul() {
            bases.push(key);
        }
    }
    fn update(k: &Key, common_b: &mut OrderedMap<Key, Basic>, bases: &mut Vec<Key>) -> PolyResult<()> {
        let Some(value) = common_b.get(k).and_then(integer_of) else {
            return Ok(());
        };
        let newe: Int = div_floor(&value, &k.1);
        let r: Int = &value - &newe * &k.1;
        if r.is_zero() {
            common_b.pop(k);
            if !newe.is_zero() {
                for m in factors_of(&pow(&k.0, &int(&newe))?) {
                    let (kk, ee) = bkey(&m, None)?;
                    let v: Basic = add2(common_b.get(&kk).unwrap_or(&Basic::zero()), &ee)?;
                    common_b.set(kk.clone(), v);
                    if !kk.1.is_one() {
                        bases.push(kk);
                    }
                }
            }
        }
        Ok(())
    }
    let ratq = |x: &Basic| -> PolyResult<Int> { Ok(bkey(x, None)?.0 .1) };
    bases.sort_by_cached_key(|k| sort_key(&k.0));
    let measures: Vec<i64> = bases.iter().map(|k| measure(&k.0)).collect::<PolyResult<_>>()?;
    let mut indexed: Vec<(i64, Key)> = measures.into_iter().zip(bases).collect();
    indexed.sort_by_key(|(m, _)| std::cmp::Reverse(*m));
    let mut bases: Vec<Key> = indexed.into_iter().map(|(_, k)| k).collect();
    let mut index: usize = 0;
    while index < bases.len() {
        let base: Key = bases[index].clone();
        index += 1;
        if !common_b.contains(&base) {
            continue;
        }
        let (mut b, exponent_q) = base.clone();
        let mut last: bool = false;
        let mut qlcm: Int = Int::ONE;
        loop {
            let (bstart, qstart) = (b.clone(), qlcm.clone());
            let mut bb: Vec<Key> = Vec::new();
            let mut ee: Vec<(Int, Int)> = Vec::new();
            for bi in factors_of(&b) {
                let (bib, bie) = bkey(&bi, None)?;
                let current: Option<Int> = common_b.get(&bib).and_then(integer_of);
                let bie_int: Option<Int> = integer_of(&bie);
                match (current, bie_int) {
                    (Some(c), Some(i)) if c >= i => {
                        ee.push((i, c));
                        bb.push(bib);
                    }
                    _ => {
                        ee.clear();
                        bb.clear();
                        break;
                    }
                }
            }
            if !ee.is_empty() {
                let mut min1: Int = div_floor(&ee[0].1, &ee[0].0);
                let mut ok: bool = true;
                for pair in &ee[1..] {
                    let rat_: Int = div_floor(&pair.1, &pair.0);
                    if rat_ < Int::ONE {
                        ok = false;
                        break;
                    }
                    if rat_ < min1 {
                        min1 = rat_;
                    }
                }
                if ok {
                    for i in 0..bb.len() {
                        let v: Basic = add2(common_b.get(&bb[i]).expect("present"), &int(&-(&min1 * &ee[i].0)))?;
                        common_b.set(bb[i].clone(), v);
                        update(&bb[i], &mut common_b, &mut bases)?;
                    }
                    let current: Basic = common_b
                        .get(&base)
                        .cloned()
                        .ok_or_else(|| crate::error::value_error("a base was removed".to_string()))?;
                    common_b.set(base.clone(), add2(&current, &int(&(&min1 * &qstart * &exponent_q)))?);
                }
            }
            if last || common_b.0.len() == 1 || common_b.0.iter().all(|(k, _)| k.1.is_one()) {
                break;
            }
            let mut l: Int = Int::ONE;
            for bi in factors_of(&bstart) {
                let q: Int = ratq(&bi)?;
                let g: Int = crate::core::intfunc::igcd(&[l.clone(), q.clone()]);
                l = div_floor(&(&l * &q), &g);
            }
            qlcm = l;
            if qlcm.is_one() {
                break;
            }
            b = pow(&bstart, &int(&qlcm))?;
            qlcm = &qlcm * &qstart;
            let mut all_one: bool = true;
            for bi in factors_of(&b) {
                if !ratq(&bi)?.is_one() {
                    all_one = false;
                    break;
                }
            }
            if all_one {
                last = true;
            }
        }
        let (b, q) = base.clone();
        let value: Basic = common_b
            .pop(&base)
            .ok_or_else(|| crate::error::value_error("a base was removed".to_string()))?;
        let factor: Basic = if q.is_one() {
            Basic::one()
        } else {
            Basic::from_rational(rat_from_parts(Int::ONE, q))
        };
        done.push((b, mul2(&value, &factor)?));
    }
    for ((b, q), ex) in common_b.0 {
        let b: Basic = if (b.is_pow() || is_exp(&b)) && !q.is_one() && !is_rational(&as_base_exp(&b)?.1) {
            let (bb_, be_) = as_base_exp(&b)?;
            pow(&bb_, &mul2(&be_, &Basic::from_rational(rat_from_parts(Int::ONE, q)))?)?
        } else {
            pow(&b, &Basic::from_rational(rat_from_parts(Int::ONE, q)))?
        };
        done.push((b, ex));
    }
    let mut last: OrderedMap<Basic, Basic> = OrderedMap::new();
    for (b, ex) in done {
        last.set(b, ex);
    }
    let powers: Vec<Basic> = last.0.iter().map(|(b, ex)| pow(b, ex)).collect::<PolyResult<_>>()?;
    let newexpr: Basic = mul(&powers)?;
    if combine == Combine::Exp {
        return mul2(&newexpr, &mul(&nc_part)?);
    }
    mul2(
        &recurse(&mul(&nc_part)?, None, Some(Combine::Base))?,
        &recurse(&newexpr, None, Some(Combine::Base))?,
    )
}

fn count_terms(e: &Basic) -> usize {
    if e.is_add() {
        return e.args().iter().map(count_terms).sum();
    }
    if e.is_mul() {
        return e.args().iter().map(count_terms).product();
    }
    1
}

fn coeff_isneg(e: &Basic) -> bool {
    if e.is_number_class() {
        return is_true(e, Predicate::negative);
    }
    e.is_mul() && !e.args().is_empty() && e.args()[0].is_number_class() && is_true(&e.args()[0], Predicate::negative)
}

fn combine_bases(e: &Basic, deep: bool, force: bool, recurse: Recurse) -> PolyResult<Basic> {
    let mut c_powers: Vec<(Basic, Basic)> = Vec::new();
    let mut nc_part: Vec<Basic> = Vec::new();
    for term in e.args() {
        if !is_false(term, Predicate::commutative) {
            c_powers.push(as_base_exp(term)?);
        } else {
            nc_part.push(term.clone());
        }
    }
    for pair in c_powers.iter_mut() {
        let (b, ex) = pair.clone();
        let (n, d) = as_numer_denom(&b)?;
        let nonnegative: bool = is_true(&n, Predicate::nonnegative) && is_true(&d, Predicate::nonnegative);
        if !(nonnegative || is_true(&ex, Predicate::integer) || force || is_polar(&b)) {
            continue;
        }
        let (exp_c, exp_t) = as_coeff_mul(&ex, true)?;
        if !is_one(&exp_c) && !is_one(&exp_t) {
            *pair = (pow(&b, &exp_c)?, exp_t);
        }
    }
    let mut c_exp: OrderedMap<Basic, Vec<Basic>> = OrderedMap::new();
    for (mut b, mut ex) in c_powers {
        if deep {
            ex = recurse(&ex, None, None)?;
        }
        if ex.is_add() && (is_true(&b, Predicate::positive) || is_true(&ex, Predicate::integer)) {
            ex = crate::core::exprtools::factor_terms(&ex, false, false, false, true)?;
            if coeff_isneg(&ex) {
                ex = neg(&ex)?;
                b = pow(&b, &Basic::negative_one())?;
            }
        }
        c_exp.entry(&ex, Vec::new).push(b);
    }
    let mut result: OrderedMap<Basic, Vec<Basic>> = OrderedMap::new();
    for (ex, found) in c_exp.0 {
        let new_base: Basic = if found.len() == 1 {
            found[0].clone()
        } else if is_true(&ex, Predicate::integer) || force {
            mul(&found)?
        } else {
            let (mut unk, mut nonneg, mut negs): (Vec<Basic>, Vec<Basic>, Vec<Basic>) =
                (Vec::new(), Vec::new(), Vec::new());
            for bi in found {
                if is_true(&bi, Predicate::negative) {
                    negs.push(bi);
                } else if is_true(&bi, Predicate::nonnegative) || is_polar(&bi) {
                    nonneg.push(bi);
                } else {
                    unk.push(bi);
                }
            }
            if (unk.len() == 1 && negs.is_empty()) || (negs.len() == 1 && unk.is_empty()) {
                nonneg.append(&mut unk);
                nonneg.append(&mut negs);
            } else if !negs.is_empty() {
                let mut israt: bool = is_rational(&ex);
                if !israt {
                    let (p, d) = as_numer_denom(&ex)?;
                    israt = is_true(&p, Predicate::integer) && is_true(&d, Predicate::integer);
                }
                if israt {
                    let count: usize = negs.len();
                    negs = negs.iter().map(neg).collect::<PolyResult<_>>()?;
                    unk.extend(std::iter::repeat(Basic::negative_one()).take(count));
                } else {
                    unk.append(&mut negs);
                }
            }
            for b in &unk {
                result.entry(b, Vec::new).push(ex.clone());
            }
            let mut factors: Vec<Basic> = nonneg;
            factors.extend(negs);
            let mut new_base: Basic = mul(&factors)?;
            let expanded: Basic = crate::core::function::expand_mul(&new_base, false)?;
            if terms_of(&expanded).len() < count_terms(&new_base) {
                new_base = crate::core::exprtools::factor_terms(&expanded, false, false, false, true)?;
            }
            new_base
        };
        result.entry(&new_base, Vec::new).push(ex);
    }
    let mut parts: Vec<Basic> = Vec::new();
    for (b, es) in &result.0 {
        for ei in es {
            parts.push(pow(b, ei)?);
        }
    }
    parts.extend(nc_part);
    mul(&parts)
}

/// `powsimp` with SymPy's defaults (`deep=False`, `combine='all'`,
/// `force=False`, `count_ops`).
pub fn powsimp_default(e: &Basic) -> PolyResult<Basic> {
    powsimp(e, false, Combine::All, false, &count_ops)
}
