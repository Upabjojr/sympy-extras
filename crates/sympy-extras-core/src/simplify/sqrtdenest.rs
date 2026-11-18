//! Port of `sympy_extras/simplify/sqrtdenest.py` (itself ported from
//! SymPy's `sympy.simplify.sqrtdenest`): square roots of sums with
//! surds denested where possible (Borodin, Fagin, Hopcroft and Tompa's
//! `_sqrtdenest_rec`, the numeric and symbolic cases, the biquadratic
//! case and the general `_denester`). Python's `SqrtdenestStopIteration`
//! is `None` here.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::{as_coeff_mul, as_numer_denom, is_integer_class, is_true};
use crate::core::sorting::sort_key;
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;

use super::radsimp::{rad_rationalize, radsimp, split_surds};

fn mul(v: &[Basic]) -> PolyResult<Basic> {
    crate::core::mul::mul(v)
}

fn add(v: &[Basic]) -> PolyResult<Basic> {
    crate::core::add::add(v)
}

fn pow(b: &Basic, e: &Basic) -> PolyResult<Basic> {
    crate::core::power::pow(b, e)
}

fn sq(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::int(2))
}

pub(crate) fn sqrt(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::half())
}

fn inv(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::negative_one())
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    crate::core::mul::neg(e)
}

fn ops(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

/// SymPy's `_mexpand`: `expand_mul(expand_multinomial(e))`.
pub fn mexpand(e: &Basic) -> PolyResult<Basic> {
    crate::core::function::expand_mul(&crate::core::function::expand_multinomial(e, true)?, true)
}

/// SymPy's `bool` of a relational: a `TypeError` when undecided.
pub(crate) fn truth(b: &Basic) -> PolyResult<bool> {
    match b.head() {
        Head::BooleanTrue => Ok(true),
        Head::BooleanFalse => Ok(false),
        _ => Err(PolyError::Type(format!("cannot determine truth value of {}", b))),
    }
}

pub(crate) fn lt0(a: &Basic, b: &Basic) -> PolyResult<bool> {
    truth(&crate::core::relational::lt(a, b)?)
}

fn le0(a: &Basic, b: &Basic) -> PolyResult<bool> {
    truth(&crate::core::relational::le(a, b)?)
}

fn is_rational(e: &Basic) -> bool {
    e.is_rational_class()
}

fn is_one(e: &Basic) -> bool {
    is_integer_class(e) && e.as_rational().is_some_and(|v| v.is_one())
}

fn is_zero(e: &Basic) -> bool {
    is_integer_class(e) && e.as_rational().is_some_and(|v| v.is_zero())
}

fn positive_integer(e: &Basic) -> bool {
    is_integer_class(e) && e.as_rational().is_some_and(|v| *v > Rat::ZERO)
}

/// Whether `e` is `x**(1/2)` or `x**(-1/2)` (`is_sqrt`).
pub fn is_sqrt(e: &Basic) -> bool {
    e.is_pow()
        && e.args()[1].is_rational_class()
        && e.args()[1]
            .as_rational()
            .is_some_and(|v| denom(v) == Int::from(2) && abs(&numer(v)).is_one())
}

/// The depth of nesting of square roots, `I` counting as one
/// (`sqrt_depth`).
pub fn sqrt_depth(p: &Basic) -> usize {
    if matches!(p.head(), Head::ImaginaryUnit) {
        return 1;
    }
    if p.args().is_empty() {
        return 0;
    }
    if p.is_add() || p.is_mul() {
        return p.args().iter().map(sqrt_depth).max().unwrap_or(0);
    }
    if is_sqrt(p) {
        return sqrt_depth(&p.args()[0]) + 1;
    }
    0
}

/// Whether `p` is built from rationals by sums, products, integer powers
/// and square roots (`is_algebraic`).
pub fn is_algebraic(p: &Basic) -> bool {
    if is_rational(p) {
        return true;
    }
    if p.args().is_empty() {
        return false;
    }
    if p.is_pow() && (is_sqrt(p) || is_integer_class(&p.args()[1])) {
        return is_algebraic(&p.args()[0]);
    }
    if p.is_add() || p.is_mul() {
        return p.args().iter().all(is_algebraic);
    }
    false
}

fn subsets(n: usize) -> Vec<Vec<u8>> {
    match n {
        1 => vec![vec![1]],
        2 => vec![vec![1, 0], vec![0, 1], vec![1, 1]],
        3 => vec![
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 1],
            vec![1, 0, 1],
            vec![0, 1, 1],
            vec![1, 1, 1],
        ],
        _ => {
            let b: Vec<Vec<u8>> = subsets(n - 1);
            let mut result: Vec<Vec<u8>> = b.iter().map(|x| x.iter().cloned().chain([0]).collect()).collect();
            result.push(std::iter::repeat(0).take(n - 1).chain([1]).collect());
            result.extend(b.iter().map(|x| x.iter().cloned().chain([1]).collect()));
            result
        }
    }
}

/// The square roots of `expr` denested where possible, in at most
/// `max_iter` passes (`sqrtdenest`).
pub fn sqrtdenest(expr: &Basic, max_iter: usize) -> PolyResult<Basic> {
    let mut e: Basic = crate::core::function::expand_mul(expr, true)?;
    for _ in 0..max_iter {
        let z: Basic = sqrtdenest0(&e)?;
        if e == z {
            return Ok(e);
        }
        e = z;
    }
    Ok(e)
}

fn sorted_args(e: &Basic) -> Vec<Basic> {
    let mut args: Vec<Basic> = e.args().to_vec();
    args.sort_by_cached_key(sort_key);
    args
}

/// `[a, b, r]` with `p = a + b*sqrt(r)`, `r` the deepest radicand; empty
/// without one (`_sqrt_match`).
fn sqrt_match(p0: &Basic) -> PolyResult<Vec<Basic>> {
    let p: Basic = mexpand(p0)?;
    if p.is_number_class() {
        return Ok(vec![p, Basic::zero(), Basic::zero()]);
    }
    if p.is_add() {
        let mut pargs: Vec<Basic> = sorted_args(&p);
        let mut all_surds: bool = true;
        for x in &pargs {
            let x2: Basic = sq(x)?;
            if !(is_rational(&x2) && is_true(&x2, Predicate::positive)) {
                all_surds = false;
                break;
            }
        }
        if all_surds {
            let (rs, bs, as_) = split_surds(&p)?;
            return Ok(vec![as_, bs, rs]);
        }
        let mut v: Vec<(usize, Basic, usize)> = pargs
            .iter()
            .enumerate()
            .map(|(i, x)| (sqrt_depth(x), x.clone(), i))
            .collect();
        let nmax: (usize, Basic, usize) = v
            .iter()
            .max_by(|a, b| (a.0, sort_key(&a.1), a.2).cmp(&(b.0, sort_key(&b.1), b.2)))
            .cloned()
            .expect("terms");
        if nmax.0 == 0 {
            return Ok(vec![]);
        }
        let (depth, _, i) = nmax;
        let mut r: Basic = pargs.remove(i);
        v.remove(i);
        let mut b: Basic = Basic::one();
        if r.is_mul() {
            let bv: Vec<Basic> = r.args().iter().filter(|x| sqrt_depth(x) < depth).cloned().collect();
            let rv: Vec<Basic> = r.args().iter().filter(|x| sqrt_depth(x) >= depth).cloned().collect();
            b = crate::core::mul::raw_mul(bv);
            r = crate::core::mul::raw_mul(rv);
        }
        let mut a1: Vec<Basic> = Vec::new();
        let mut b1: Vec<Basic> = vec![b];
        for (xd, x1, _) in v {
            if xd < depth {
                a1.push(x1);
            } else if x1 == r {
                b1.push(Basic::one());
            } else if x1.is_mul() {
                let mut x1args: Vec<Basic> = x1.args().to_vec();
                if let Some(k) = x1args.iter().position(|t| *t == r) {
                    x1args.remove(k);
                    b1.push(mul(&x1args)?);
                } else {
                    a1.push(x1);
                }
            } else {
                a1.push(x1);
            }
        }
        return Ok(vec![add(&a1)?, add(&b1)?, sq(&r)?]);
    }
    let (bc, rr) = as_coeff_mul(&p);
    if is_sqrt(&rr) {
        return Ok(vec![Basic::zero(), bc, sq(&rr)?]);
    }
    Ok(vec![])
}

fn sqrtdenest0(e: &Basic) -> PolyResult<Basic> {
    if is_sqrt(e) {
        let (n, d) = as_numer_denom(e)?;
        if is_one(&d) {
            let mut e: Basic = e.clone();
            if n.is_pow() && n.args()[0].is_add() {
                let args: Vec<Basic> = sorted_args(&n.args()[0]);
                let mut all_integer: bool = args.len() > 2;
                if all_integer {
                    for x in &args {
                        if !is_integer_class(&sq(x)?) {
                            all_integer = false;
                            break;
                        }
                    }
                }
                if all_integer {
                    if let Some(r) = sqrtdenest_rec(&n)? {
                        return Ok(r);
                    }
                }
                let parts: Vec<Basic> = args.iter().map(sqrtdenest0).collect::<PolyResult<_>>()?;
                e = sqrt(&mexpand(&add(&parts)?)?)?;
            }
            return sqrtdenest1(&e, true);
        }
        return mul(&[sqrtdenest0(&n)?, inv(&sqrtdenest0(&d)?)?]);
    }
    if e.is_add() {
        let (mut cs, mut args2): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
        for arg in e.args() {
            let (c, a) = as_coeff_mul(arg);
            cs.push(c);
            args2.push(a);
        }
        if cs.iter().all(is_rational) && args2.iter().all(is_sqrt) {
            return sqrt_ratcomb(cs, args2);
        }
    }
    if !e.args().is_empty() {
        let args: Vec<Basic> = e
            .args()
            .iter()
            .map(|a| {
                if matches!(a.head(), Head::Tuple) || crate::core::logic::is_boolean(a) {
                    Ok(a.clone())
                } else {
                    sqrtdenest0(a)
                }
            })
            .collect::<PolyResult<_>>()?;
        return crate::core::function::rebuild_evaluated(e, args);
    }
    Ok(e.clone())
}

fn sqrt2() -> PolyResult<Basic> {
    sqrt(&Basic::int(2))
}

/// Denesting by the method of Borodin et al. (`_sqrtdenest_rec`); `None`
/// for Python's `SqrtdenestStopIteration`.
fn sqrtdenest_rec(e: &Basic) -> PolyResult<Option<Basic>> {
    if !e.is_pow() {
        return Ok(Some(sqrtdenest(e, 3)?));
    }
    let base: &Basic = &e.args()[0];
    if lt0(base, &Basic::zero())? {
        let Some(inner) = sqrtdenest_rec(&sqrt(&neg(base)?)?)? else {
            return Ok(None);
        };
        return Ok(Some(mul(&[sqrt(&Basic::negative_one())?, inner])?));
    }
    let (g, a, b) = split_surds(base)?;
    let (mut a, mut b) = (mul(&[a, sqrt(&g)?])?, b);
    if lt0(&a, &b)? {
        std::mem::swap(&mut a, &mut b);
    }
    let c2: Basic = mexpand(&add(&[sq(&a)?, neg(&sq(&b)?)?])?)?;
    let c: Basic = if c2.args().len() > 2 {
        let (g, a1, b1) = split_surds(&c2)?;
        let (mut a1, mut b1) = (mul(&[a1, sqrt(&g)?])?, b1);
        if lt0(&a1, &b1)? {
            std::mem::swap(&mut a1, &mut b1);
        }
        let c2_1: Basic = mexpand(&add(&[sq(&a1)?, neg(&sq(&b1)?)?])?)?;
        let Some(c_1) = sqrtdenest_rec(&sqrt(&c2_1)?)? else {
            return Ok(None);
        };
        let Some(d_1) = sqrtdenest_rec(&sqrt(&add(&[a1, c_1])?)?)? else {
            return Ok(None);
        };
        let (num, den) = rad_rationalize(&b1, &d_1)?;
        mexpand(&add(&[
            mul(&[d_1, inv(&sqrt2()?)?])?,
            mul(&[num, inv(&mul(&[den, sqrt2()?])?)?])?,
        ])?)?
    } else {
        sqrtdenest1(&sqrt(&c2)?, true)?
    };
    if sqrt_depth(&c) > 1 {
        return Ok(None);
    }
    let ac: Basic = add(&[a, c])?;
    if ac.args().len() >= e.args().len() && ops(&ac)? >= ops(base)? {
        return Ok(None);
    }
    let d: Basic = sqrtdenest(&sqrt(&ac)?, 3)?;
    if sqrt_depth(&d) > 1 {
        return Ok(None);
    }
    let (num, den) = rad_rationalize(&b, &d)?;
    let r: Basic = add(&[mul(&[d, inv(&sqrt2()?)?])?, mul(&[num, inv(&mul(&[den, sqrt2()?])?)?])?])?;
    Ok(Some(mexpand(&radsimp(&r, true, 4)?)?))
}

fn sqrtdenest1(e: &Basic, denester: bool) -> PolyResult<Basic> {
    if !is_sqrt(e) {
        return Ok(e.clone());
    }
    let a0: &Basic = &e.args()[0];
    if a0.args().is_empty() {
        return Ok(e.clone());
    }
    let val: Vec<Basic> = sqrt_match(a0)?;
    if val.is_empty() {
        return Ok(e.clone());
    }
    let (a, b, r) = (val[0].clone(), val[1].clone(), val[2].clone());
    let d2: Basic = mexpand(&add(&[sq(&a)?, neg(&mul(&[sq(&b)?, r.clone()])?)?])?)?;
    if is_rational(&d2) {
        if is_true(&d2, Predicate::positive) {
            if let Some(z) = sqrt_numeric_denest(&a, &b, &r, &d2)? {
                return Ok(z);
            }
        } else {
            let dr2: Basic = mexpand(&neg(&mul(&[d2.clone(), r.clone()])?)?)?;
            let dr: Basic = sqrt(&dr2)?;
            if is_rational(&dr) {
                if let Some(z) = sqrt_numeric_denest(&mexpand(&mul(&[b.clone(), r.clone()])?)?, &a, &r, &dr2)? {
                    return mul(&[z, inv(&pow(&r, &Basic::from_rational(rat(1, 4)))?)?]);
                }
            }
        }
    } else if let Some(z) = sqrt_symbolic_denest(&a, &b, &r)? {
        return Ok(z);
    }
    if !denester || !is_algebraic(e) {
        return Ok(e.clone());
    }
    if let Some(res) = sqrt_biquadratic_denest(e, &a, &b, &r, &d2)? {
        return Ok(res);
    }
    let mut av0: [Option<Basic>; 4] = [Some(a), Some(b), Some(r), Some(d2)];
    let z2: Option<Basic> = denester_(&[radsimp(&sq(e)?, true, 4)?], &mut av0, 0, sqrt_depth(e))?.0;
    if av0[1].is_none() {
        return Ok(e.clone());
    }
    if let Some(z2) = z2 {
        if sqrt_depth(&z2) == sqrt_depth(e) && ops(&z2)? > ops(e)? {
            return Ok(e.clone());
        }
        return Ok(z2);
    }
    Ok(e.clone())
}

fn sqrt_symbolic_denest(a: &Basic, b: &Basic, r: &Basic) -> PolyResult<Option<Basic>> {
    let rval: Vec<Basic> = sqrt_match(r)?;
    if rval.is_empty() {
        return Ok(None);
    }
    let (ra, rb, rr) = (&rval[0], &rval[1], &rval[2]);
    if is_zero(rb) {
        return Ok(None);
    }
    let y: Basic = Dummy::new("y", &[(Predicate::positive, true)]).0;
    let value: Basic = mul(&[add(&[sq(&y)?, neg(ra)?])?, inv(rb)?])?;
    let substituted: Basic = crate::core::subs::subs(a, &sqrt(rr)?, &value)?;
    let terms = match crate::polys::polytools::decompose_with_gens(&mexpand(&substituted)?, std::slice::from_ref(&y)) {
        Ok(t) => t,
        Err(PolyError::Polys(..)) => return Ok(None),
        Err(other) => return Err(other),
    };
    let mut coefficients: Vec<(u32, Basic)> = Vec::new();
    for (m, cs) in &terms {
        let c: Basic = add(cs)?;
        if !is_zero(&c) {
            coefficients.push((m[0], c));
        }
    }
    if coefficients.is_empty() || coefficients.iter().map(|(k, _)| *k).max() != Some(2) {
        return Ok(None);
    }
    let get = |k: u32| {
        coefficients
            .iter()
            .find(|(d, _)| *d == k)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Basic::zero)
    };
    let (ca, cb, cc) = (get(2), add(&[get(1), b.clone()])?, get(0));
    let discriminant: Basic = mexpand(&add(&[sq(&cb)?, mul(&[Basic::int(-4), ca.clone(), cc])?])?)?;
    if crate::core::expr::equals(&discriminant, &Basic::zero())? {
        let shift: Basic = mul(&[cb, inv(&mul(&[Basic::int(2), ca.clone()])?)?])?;
        let mut z: Basic = sqrt(&mul(&[ca, sq(&add(&[sqrt(r)?, shift])?)?])?)?;
        if crate::core::assumptions::is_number(&z) {
            let (c, p) = crate::core::exprtools::as_content_primitive(&z, false, true)?;
            z = mexpand(&if is_one(&c) {
                p
            } else {
                crate::core::mul::raw_mul(vec![c, p])
            })?;
        }
        return Ok(Some(z));
    }
    Ok(None)
}

fn sqrt_numeric_denest(a: &Basic, b: &Basic, r: &Basic, d2: &Basic) -> PolyResult<Option<Basic>> {
    let d: Basic = sqrt(d2)?;
    let s: Basic = add(&[a.clone(), d.clone()])?;
    if sqrt_depth(&s) < sqrt_depth(r) + 1 || is_rational(&sq(&s)?) {
        let sign = crate::functions::elementary::complexes::sign;
        let (mut s1, mut s2) = (sign(&s)?, sign(b)?);
        let minus_one: Basic = Basic::negative_one();
        if s1 == minus_one && s2 == minus_one {
            s1 = Basic::one();
            s2 = Basic::one();
        }
        let res: Basic = mul(&[
            add(&[
                mul(&[s1, sqrt(&add(&[a.clone(), d.clone()])?)?])?,
                mul(&[s2, sqrt(&add(&[a.clone(), neg(&d)?])?)?])?,
            ])?,
            sqrt2()?,
            Basic::half(),
        ])?;
        return Ok(Some(crate::core::function::expand_default(&res)?));
    }
    Ok(None)
}

/// The denesting of `sqrt(a + b*sqrt(r))` in a biquadratic field
/// (`sqrt_biquadratic_denest`).
pub fn sqrt_biquadratic_denest(e: &Basic, a: &Basic, b: &Basic, r: &Basic, d2: &Basic) -> PolyResult<Option<Basic>> {
    if !e.is_pow() {
        return Ok(None);
    }
    let zero: Basic = Basic::zero();
    if le0(r, &zero)? || lt0(d2, &zero)? || is_zero(b) || sqrt_depth(&e.args()[0]) < 2 {
        return Ok(None);
    }
    for x in [a, b, r] {
        for yv in x.args() {
            if !positive_integer(&sq(yv)?) {
                return Ok(None);
            }
        }
    }
    let sqd: Basic = mexpand(&sqrtdenest(&sqrt(&radsimp(d2, true, 4)?)?, 3)?)?;
    if sqrt_depth(&sqd) > 1 {
        return Ok(None);
    }
    let half: Basic = Basic::half();
    let halves: [Basic; 2] = [
        add(&[mul(&[a.clone(), half.clone()])?, mul(&[sqd.clone(), half.clone()])?])?,
        add(&[
            mul(&[a.clone(), half.clone()])?,
            neg(&mul(&[sqd.clone(), half.clone()])?)?,
        ])?,
    ];
    for x in halves {
        let big_a: Basic = sqrtdenest(&sqrt(&x)?, 3)?;
        if sqrt_depth(&big_a) > 1 {
            continue;
        }
        let (bn, bd) = rad_rationalize(b, &mexpand(&mul(&[Basic::int(2), big_a.clone()])?)?)?;
        let big_b: Basic = mul(&[bn, inv(&bd)?])?;
        let mut z: Basic = add(&[big_a, mul(&[big_b, sqrt(r)?])?])?;
        if lt0(&z, &zero)? {
            z = neg(&z)?;
        }
        return Ok(Some(mexpand(&z)?));
    }
    Ok(None)
}

type Denested = (Option<Basic>, Option<Vec<u8>>);

fn denester_(nested: &[Basic], av0: &mut [Option<Basic>; 4], h: usize, max_depth_level: usize) -> PolyResult<Denested> {
    if h > max_depth_level {
        return Ok((None, None));
    }
    if av0[1].is_none() {
        return Ok((None, None));
    }
    let zeros = |n: usize| -> Vec<u8> { vec![0; n] };
    if av0[0].is_none() && nested.iter().all(|n| n.is_number_class()) {
        for subset in subsets(nested.len()) {
            let chosen: Vec<Basic> = (0..subset.len())
                .filter(|i| subset[*i] == 1)
                .map(|i| nested[i].clone())
                .collect();
            let mut p: Basic = mexpand(&mul(&chosen)?)?;
            if subset.iter().filter(|v| **v == 1).count() > 1 && *subset.last().expect("a subset") == 1 {
                p = neg(&p)?;
            }
            let sqp: Basic = sqrt(&p)?;
            if is_rational(&sqp) {
                return Ok((Some(sqp), Some(subset)));
            }
        }
        return Ok((Some(sqrt(nested.last().expect("nested"))?), Some(zeros(nested.len()))));
    }
    let mut big_r: Option<Basic> = None;
    let values: Vec<Vec<Basic>>;
    let nested2: Vec<Basic>;
    if av0[0].is_some() {
        let (Some(first), Some(second), Some(third), Some(fourth)) =
            (av0[0].clone(), av0[1].clone(), av0[2].clone(), av0[3].clone())
        else {
            return Ok((None, None));
        };
        values = vec![vec![first, second]];
        big_r = Some(third.clone());
        nested2 = vec![fourth, third];
        av0[0] = None;
    } else {
        let mut found: Vec<Vec<Basic>> = Vec::new();
        for x in nested {
            let v: Vec<Basic> = sqrt_match(x)?;
            if !v.is_empty() {
                found.push(v);
            }
        }
        for v in &found {
            if !is_zero(&v[2]) {
                match &big_r {
                    Some(r) if *r != v[2] => {
                        av0[1] = None;
                        return Ok((None, None));
                    }
                    Some(_) => {}
                    None => big_r = Some(v[2].clone()),
                }
            }
        }
        let Some(r) = big_r.clone() else {
            return Ok((Some(sqrt(nested.last().expect("nested"))?), Some(zeros(nested.len()))));
        };
        let mut n2: Vec<Basic> = Vec::new();
        for v in &found {
            n2.push(add(&[
                mexpand(&sq(&v[0])?)?,
                neg(&mexpand(&mul(&[r.clone(), sq(&v[1])?])?)?)?,
            ])?);
        }
        n2.push(r);
        values = found;
        nested2 = n2;
    }
    let big_r: Basic = big_r.expect("a radicand");
    let (d, f) = denester_(&nested2, av0, h + 1, max_depth_level)?;
    let (Some(d), Some(f)) = (d, f) else {
        return Ok((None, None));
    };
    if f.is_empty() {
        return Ok((None, None));
    }
    let n: usize = nested.len();
    if !(0..n).any(|i| f[i] == 1) {
        let v: &Vec<Basic> = values.last().expect("values");
        return Ok((
            Some(sqrt(&add(&[v[0].clone(), mexpand(&mul(&[v[1].clone(), d])?)?])?)?),
            Some(f),
        ));
    }
    let chosen: Vec<Basic> = (0..n).filter(|i| f[*i] == 1).map(|i| nested[i].clone()).collect();
    let p: Basic = mul(&chosen)?;
    let mut vm: Vec<Basic> = sqrt_match(&p)?;
    if let Some(first_one) = f.iter().position(|v| *v == 1) {
        if first_one < n - 1 && f[n - 1] == 1 {
            vm[0] = neg(&vm[0])?;
            vm[1] = neg(&vm[1])?;
        }
    }
    if f[n] == 0 {
        let vad: Basic = mexpand(&add(&[vm[0].clone(), d.clone()])?)?;
        if le0(&vad, &Basic::zero())? {
            return Ok((Some(sqrt(nested.last().expect("nested"))?), Some(zeros(n))));
        }
        if !(sqrt_depth(&vad) <= sqrt_depth(&big_r) + 1 || sq(&vad)?.is_number_class()) {
            av0[1] = None;
            return Ok((None, None));
        }
        let sqvad: Basic = sqrtdenest1(&sqrt(&vad)?, false)?;
        if sqrt_depth(&sqvad) > sqrt_depth(&big_r) + 1 {
            av0[1] = None;
            return Ok((None, None));
        }
        let sqvad1: Basic = radsimp(&inv(&sqvad)?, true, 4)?;
        let res: Basic = mexpand(&add(&[
            mul(&[sqvad, inv(&sqrt2()?)?])?,
            mul(&[vm[1].clone(), sqrt(&big_r)?, sqvad1, inv(&sqrt2()?)?])?,
        ])?)?;
        return Ok((Some(res), Some(f)));
    }
    let s2: Basic = add(&[mexpand(&mul(&[vm[1].clone(), big_r.clone()])?)?, d])?;
    if le0(&s2, &Basic::zero())? {
        return Ok((Some(sqrt(nested.last().expect("nested"))?), Some(zeros(n))));
    }
    let fr: Basic = pow(&mexpand(&big_r)?, &Basic::from_rational(rat(1, 4)))?;
    let s: Basic = sqrt(&s2)?;
    let res: Basic = mexpand(&add(&[
        mul(&[s.clone(), inv(&mul(&[sqrt2()?, fr.clone()])?)?])?,
        mul(&[vm[0].clone(), fr.clone(), inv(&mul(&[sqrt2()?, s])?)?])?,
    ])?)?;
    Ok((Some(res), Some(f)))
}

fn sqrt_ratcomb(mut cs: Vec<Basic>, mut args: Vec<Basic>) -> PolyResult<Basic> {
    loop {
        let mut found: Option<(Basic, usize, usize)> = None;
        'search: for i in 0..args.len().saturating_sub(1) {
            for j in i + 1..args.len() {
                let (ai, aj) = (&args[i], &args[j]);
                if !(ai.is_pow() && aj.is_pow()) {
                    continue;
                }
                let p: Basic = mexpand(&mul(&[ai.args()[0].clone(), aj.args()[0].clone()])?)?;
                let s: Basic = sqrtdenest(&sqrt(&p)?, 3)?;
                if s != sqrt(&p)? {
                    found = Some((s, i, j));
                    break 'search;
                }
            }
        }
        let Some((s, i1, i2)) = found else {
            let terms: Vec<Basic> = cs
                .iter()
                .zip(&args)
                .map(|(c, a)| mul(&[c.clone(), a.clone()]))
                .collect::<PolyResult<_>>()?;
            return add(&terms);
        };
        let c2: Basic = cs.remove(i2);
        args.remove(i2);
        let a1: Basic = args[i1].clone();
        if a1.is_pow() {
            let extra: Basic = radsimp(&mul(&[c2, s, inv(&a1.args()[0])?])?, true, 4)?;
            cs[i1] = add(&[cs[i1].clone(), extra])?;
        }
    }
}
