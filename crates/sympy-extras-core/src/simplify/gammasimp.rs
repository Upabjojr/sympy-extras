//! `gammasimp` and `combsimp`, ported from `sympy_extras.simplify.gammasimp`:
//! gamma functions with arguments differing by integers merged through
//! the rising factorial; the reflection formula, the duplication formula
//! and Gauss's multiplication theorem; `combsimp` keeps integer arguments
//! integer and rewrites the result as factorials and binomials.

// PORTED FROM SYMPY: sympy/simplify/gammasimp.py (gammasimp, _gammasimp, _rf) and
// sympy/simplify/combsimp.py (combsimp, _gamma_as_comb) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::core::add::add;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_base_exp, as_coeff_add, as_numer_denom, factors_of, is};
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::core::sorting::{ordered, sort_key};
use crate::error::PolyResult;
use crate::numbers::{Int, Rat};

type Leaf = fn(&Basic, &Basic) -> PolyResult<Basic>;

fn one() -> Basic {
    Basic::one()
}

fn int(n: i64) -> Basic {
    Basic::integer(n.into())
}

fn inv(e: &Basic) -> PolyResult<Basic> {
    pow(e, &Basic::negative_one())
}

/// The value of an `Integer` node.
fn to_int(e: &Basic) -> Option<i64> {
    match e.head() {
        Head::Rational {
            integer_class: true,
            value,
        } => i64::try_from(value.numerator()).ok(),
        _ => None,
    }
}

fn gamma(x: &Basic) -> PolyResult<Basic> {
    crate::functions::special::gamma_functions::gamma(x)
}

fn is_gamma(b: &Basic) -> bool {
    matches!(b.head(), Head::Function(Function::Gamma))
}

fn has_gamma(b: &Basic) -> bool {
    is_gamma(b) || b.args().iter().any(has_gamma)
}

/// Whether the node is an expression (not a Boolean, a set or a tuple).
fn is_expr(b: &Basic) -> bool {
    !crate::core::logic::is_boolean(b) && !crate::sets::sets::is_set(b) && !matches!(b.head(), Head::Tuple)
}

/// Whether the node is an `Application` (a function applied).
fn is_application(b: &Basic) -> bool {
    matches!(b.head(), Head::Function(_) | Head::Undefined(_))
}

// PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_rf
/// The rising factorial `rf(a, b)` with the integer parts of `a` and `b`
/// taken out as products; the rest is `leaf(a, b)`.
fn rf(a: &Basic, b: &Basic, leaf: Leaf, fractional: bool) -> PolyResult<Basic> {
    if let Some(n) = to_int(b) {
        if n == 0 {
            return Ok(one());
        }
        if n > 0 {
            let factors: Vec<Basic> = (0..n).map(|i| add(&[a.clone(), int(i)])).collect::<PolyResult<_>>()?;
            return mul(&factors);
        }
        let factors: Vec<Basic> = (1..=-n)
            .map(|i| add(&[a.clone(), int(-i)]))
            .collect::<PolyResult<_>>()?;
        return inv(&mul(&factors)?);
    }
    if b.is_add() {
        let (c, b0) = as_coeff_add(b);
        if let Some(ci) = to_int(&c) {
            if ci > 0 {
                return mul(&[
                    rf(a, &b0, leaf, fractional)?,
                    rf(&add(&[a.clone(), b0.clone()])?, &c, leaf, fractional)?,
                ]);
            }
            if ci < 0 {
                let shifted: Basic = add(&[a.clone(), b0.clone(), c.clone()])?;
                return mul(&[
                    rf(a, &b0, leaf, fractional)?,
                    inv(&rf(&shifted, &int(-ci), leaf, fractional)?)?,
                ]);
            }
        }
        if let (true, None, Some(q)) = (fractional, to_int(&c), c.as_rational()) {
            let whole_: Int = {
                // Python's floor division
                let (n, d) = (crate::numbers::numer(q), crate::numbers::denom(q));
                let f: Int = &n / &d;
                if (&n % &d) != Int::ZERO && n < Int::ZERO {
                    f - Int::ONE
                } else {
                    f
                }
            };
            if whole_ != Int::ZERO {
                // not SymPy's: the integer part of a fractional shift taken
                // out too, so that gamma(n + 3/2)/gamma(n + 1/2) is n + 1/2
                // (SymPy leaves it, issue #25); the rest keeps a shift in
                // [0, 1)
                let w: i64 =
                    i64::try_from(&whole_).map_err(|_| crate::error::value_error("a huge shift".to_string()))?;
                let rest: Basic = add(&[b0.clone(), Basic::from_rational(q - Rat::from(whole_.clone()))])?;
                if w > 0 {
                    return mul(&[
                        rf(a, &rest, leaf, fractional)?,
                        rf(&add(&[a.clone(), rest.clone()])?, &int(w), leaf, fractional)?,
                    ]);
                }
                return mul(&[
                    rf(a, &rest, leaf, fractional)?,
                    inv(&rf(
                        &add(&[a.clone(), rest.clone(), int(w)])?,
                        &int(-w),
                        leaf,
                        fractional,
                    )?)?,
                ]);
            }
        }
    }
    if a.is_add() {
        let (c, a0) = as_coeff_add(a);
        if let Some(ci) = to_int(&c) {
            if ci > 0 {
                return mul(&[
                    rf(&a0, b, leaf, fractional)?,
                    rf(&add(&[a0.clone(), b.clone()])?, &c, leaf, fractional)?,
                    inv(&rf(&a0, &c, leaf, fractional)?)?,
                ]);
            }
            if ci < 0 {
                return mul(&[
                    rf(&a0, b, leaf, fractional)?,
                    rf(&add(&[a0.clone(), c.clone()])?, &int(-ci), leaf, fractional)?,
                    inv(&rf(
                        &add(&[a0.clone(), b.clone(), c.clone()])?,
                        &int(-ci),
                        leaf,
                        fractional,
                    )?)?,
                ]);
            }
        }
    }
    leaf(a, b)
}

fn gamma_leaf(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[gamma(&add(&[a.clone(), b.clone()])?)?, inv(&gamma(a)?)?])
}

fn comb_leaf(_: &Basic, b: &Basic) -> PolyResult<Basic> {
    gamma(&add(&[b.clone(), one()])?)
}

/// Every `gamma(n)` as `rf(1, n - 1)` with the given leaf.
fn through_rf(e: &Basic, leaf: Leaf, fractional: bool) -> PolyResult<Basic> {
    crate::core::function::replace_where(e, &is_gamma, &|b: &Basic| {
        let shifted: Basic = crate::core::function::expand_method(&add(&[b.args()[0].clone(), int(-1)])?)?;
        rf(&one(), &shifted, leaf, fractional)
    })
}

/// `expand_func(gamma(r))`: the integer part of a rational argument taken
/// out (`gamma(4/3)` is `gamma(1/3)/3`).
fn gamma_rational(r: &Rat) -> PolyResult<Basic> {
    let p: crate::numbers::Int = r.numerator().clone();
    let q = crate::numbers::Int::from(r.denominator().clone());
    if dashu_base::Abs::abs(p) > q {
        let n = r.floor();
        let rest: Rat = r - Rat::from(n.clone());
        let rest: Basic = Basic::from_rational(rest);
        return mul(&[
            gamma(&rest)?,
            crate::functions::combinatorial::factorials::rf(&rest, &Basic::integer(n))?,
        ]);
    }
    gamma(&Basic::from_rational(r.clone()))
}

fn gamma_factor(x: &Basic) -> bool {
    if is_gamma(x) {
        return true;
    }
    if x.is_add() || x.is_mul() {
        return x.args().iter().any(gamma_factor);
    }
    if x.is_pow() {
        let (base, exponent) = (&x.args()[0], &x.args()[1]);
        if is(exponent, Predicate::integer) == Some(true) || is(base, Predicate::positive) == Some(true) {
            return gamma_factor(base);
        }
    }
    false
}

fn gamma_count(x: &Basic) -> usize {
    x.preorder().iter().filter(|n| is_gamma(n)).count()
}

fn gamma_rat(x: &Basic) -> PolyResult<Basic> {
    let xx: Basic = through_rf(x, gamma_leaf, false)?;
    Ok(if gamma_count(&xx) < gamma_count(x) {
        xx
    } else {
        x.clone()
    })
}

/// `x % 1` of Python's fractions (the floor remainder).
fn mod_one(x: &Rat) -> Rat {
    x - Rat::from(x.floor())
}

// PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp._run
/// A run `u, u + 1/n, ..., u + (n-1)/n` (modulo integers) among the
/// coefficients, removed from them.
fn run(coeffs: &mut Vec<Rat>) -> Option<(i64, Rat, Vec<Rat>)> {
    let mut u: Vec<Rat> = Vec::new();
    for c in coeffs.iter() {
        if !u.contains(c) {
            u.push(c.clone());
        }
    }
    for i in 0..u.len() {
        let dj: Vec<(Rat, usize)> = ((i + 1)..u.len()).map(|j| (mod_one(&(&u[j] - &u[i])), j)).collect();
        for (first, _) in &dj {
            if *first.numerator() == crate::numbers::Int::from(1) && *first.denominator() != 1u8.into() {
                let n: i64 = i64::try_from(first.denominator()).unwrap_or(i64::MAX);
                let mut got: Vec<usize> = vec![i];
                let mut get: Vec<i64> = (1..n).collect();
                let mut complete: bool = false;
                for (d, j) in &dj {
                    let m: Rat = Rat::from(n) * d;
                    if m.is_int() {
                        let mv: Option<i64> = i64::try_from(m.numerator()).ok();
                        if let Some(pos) = mv.and_then(|mv| get.iter().position(|g| *g == mv)) {
                            get.remove(pos);
                            got.push(*j);
                            if get.is_empty() {
                                complete = true;
                                break;
                            }
                        }
                    }
                }
                if !complete {
                    continue;
                }
                let values: Vec<Rat> = got.iter().map(|j| u[*j].clone()).collect();
                for c in &values {
                    if let Some(pos) = coeffs.iter().position(|x| x == c) {
                        coeffs.remove(pos);
                    }
                }
                return Some((n, values[0].clone(), values[1..].to_vec()));
            }
        }
    }
    None
}

// PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp._mult_thm
/// Gauss's multiplication theorem on the runs of arguments.
fn mult_thm(gammas: &mut Vec<Basic>, numer: &mut Vec<Basic>) -> PolyResult<()> {
    let mut rats: Vec<(Basic, Vec<Rat>)> = Vec::new();
    let mut push = |key: Basic, value: Rat| match rats.iter_mut().find(|(k, _)| *k == key) {
        Some((_, list)) => list.push(value),
        None => rats.push((key, vec![value])),
    };
    for g in gammas.iter() {
        let (c, resid) = as_coeff_add(g);
        match c.as_rational() {
            Some(value) if matches!(c.head(), Head::Rational { .. }) => push(resid, value.clone()),
            _ => push(g.clone(), Rat::ZERO),
        }
    }
    rats.sort_by_key(|(k, _)| sort_key(k));
    let mut result: Vec<Basic> = Vec::new();
    for (resid, coeffs) in rats {
        let mut coeffs: Vec<Rat> = coeffs;
        coeffs.sort();
        let mut new: Vec<Basic> = Vec::new();
        while let Some((n, ui, other)) = run(&mut coeffs) {
            for u in &other {
                let con: Basic = add(&[resid.clone(), Basic::from_rational(u.clone()), int(-1)])?;
                let steps: i64 = i64::try_from((u - &ui).trunc()).unwrap_or(0);
                for k in 0..steps {
                    numer.push(add(&[con.clone(), int(-k)])?);
                }
            }
            let con: Basic = mul(&[int(n), add(&[resid.clone(), Basic::from_rational(ui.clone())])?])?;
            let two_pi: Basic = mul(&[int(2), Basic::pi()])?;
            let half_n: Basic = Basic::rational((n - 1).into(), 2.into())?;
            let rest: Basic = add(&[Basic::half(), mul(&[int(-1), con.clone()])?])?;
            numer.push(mul(&[pow(&two_pi, &half_n)?, pow(&int(n), &rest)?])?);
            new.push(con);
        }
        for c in &coeffs {
            result.push(add(&[resid.clone(), Basic::from_rational(c.clone())])?);
        }
        result.extend(new);
    }
    *gammas = result;
    Ok(())
}

/// `(None, [])` for `1`, `(True, arguments)` for a power of a gamma,
/// `(False, bases)` otherwise (`_explicate`).
fn explicate(p: &Basic) -> PolyResult<(Option<bool>, Vec<Basic>)> {
    if matches!(
        p.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) && *p == one()
    {
        return Ok((None, Vec::new()));
    }
    let (b, e) = as_base_exp(p)?;
    if let Some(n) = to_int(&e) {
        let count: usize = usize::try_from(n.max(0)).unwrap_or(0);
        if is_gamma(&b) {
            return Ok((Some(true), vec![b.args()[0].clone(); count]));
        }
        return Ok((Some(false), vec![b; count]));
    }
    Ok((Some(false), vec![p.clone()]))
}

/// The reflection formula on pairs of arguments summing to an integer.
fn reflect(gammas: &mut Vec<Basic>, numer: &mut Vec<Basic>, denom: &mut Vec<Basic>) -> PolyResult<()> {
    let mut new: Vec<Basic> = Vec::new();
    while let Some(g1) = gammas.pop() {
        if is(&g1, Predicate::integer) == Some(true) {
            new.push(g1);
            continue;
        }
        let mut matched: bool = false;
        for i in 0..gammas.len() {
            let Some(n) = to_int(&add(&[g1.clone(), gammas[i].clone(), int(-1)])?) else {
                continue;
            };
            numer.push(Basic::pi());
            denom.push(crate::functions::elementary::trigonometric::sin(&mul(&[
                Basic::pi(),
                g1.clone(),
            ])?)?);
            gammas.remove(i);
            if n > 0 {
                for k in 0..n {
                    numer.push(add(&[one(), mul(&[int(-1), g1.clone()])?, int(k)])?);
                }
            } else if n < 0 {
                for k in 0..-n {
                    denom.push(add(&[mul(&[int(-1), g1.clone()])?, int(-k)])?);
                }
            }
            matched = true;
            break;
        }
        if !matched {
            new.push(g1);
        }
    }
    *gammas = new;
    Ok(())
}

fn remove_first(list: &mut Vec<Basic>, x: &Basic) {
    if let Some(pos) = list.iter().position(|y| y == x) {
        list.remove(pos);
    }
}

/// The duplication formula on `gamma(x)/gamma(y)` with `x - 2*y` an integer.
fn duplicate(ng: &mut Vec<Basic>, dg: &mut Vec<Basic>, no: &mut Vec<Basic>, d_o: &mut Vec<Basic>) -> PolyResult<()> {
    loop {
        let mut found: Option<(Basic, Basic, i64)> = None;
        'outer: for x in ng.iter() {
            for y in dg.iter() {
                if let Some(n) = to_int(&add(&[x.clone(), mul(&[int(-2), y.clone()])?])?) {
                    found = Some((x.clone(), y.clone(), n));
                    break 'outer;
                }
            }
        }
        let Some((x, y, n)) = found else {
            return Ok(());
        };
        remove_first(ng, &x);
        remove_first(dg, &y);
        let two_y: Basic = mul(&[int(2), y.clone()])?;
        if n > 0 {
            for k in 0..n {
                no.push(add(&[two_y.clone(), int(k)])?);
            }
        } else if n < 0 {
            for k in 0..-n {
                d_o.push(add(&[two_y.clone(), int(-1 - k)])?);
            }
        }
        ng.push(add(&[y.clone(), Basic::half()])?);
        no.push(pow(&int(2), &add(&[two_y, int(-1)])?)?);
        d_o.push(pow(&Basic::pi(), &Basic::half())?);
    }
}

type Signature = (BTreeSet<Basic>, BTreeSet<Basic>);

fn compute_st(inv_map: &HashMap<Basic, Signature>, e: &Basic) -> Signature {
    if let Some(found) = inv_map.get(e) {
        return found.clone();
    }
    let mut kinds: BTreeSet<Basic> = BTreeSet::new();
    for node in e.preorder() {
        if node.is_pow() {
            kinds.insert(node.args()[1].clone());
        }
        if is_application(&node) {
            kinds.insert(node.clone());
        }
    }
    (e.free_symbols(), kinds)
}

fn find_fuzzy(inv_map: &HashMap<Basic, Signature>, lst: &[Basic], x: &Basic) -> PolyResult<Option<Basic>> {
    if lst.is_empty() {
        return Ok(None);
    }
    let (s1, t1) = compute_st(inv_map, x);
    for y in lst {
        let (s2, t2) = &inv_map[y];
        if t1 != *t2 || (s1.intersection(s2).next().is_none() && (!s1.is_empty() || !s2.is_empty())) {
            continue;
        }
        let quotient: Basic = crate::polys::polytools::cancel(&mul(&[x.clone(), inv(y)?])?, &[])?;
        if quotient.free_symbols().is_empty() && (!x.free_symbols().is_empty() || !y.free_symbols().is_empty()) {
            return Ok(Some(y.clone()));
        }
    }
    Ok(None)
}

fn fuzzy_pass(
    inv_map: &mut HashMap<Basic, Signature>,
    gammas: &mut Vec<Basic>,
    numer: &mut Vec<Basic>,
    denom: &mut Vec<Basic>,
) -> PolyResult<()> {
    let mut new: Vec<Basic> = Vec::new();
    while let Some(mut g) = gammas.pop() {
        let mut cont: bool = true;
        while cont {
            cont = false;
            if let Some(y) = find_fuzzy(inv_map, numer, &g)? {
                remove_first(numer, &y);
                if y != g {
                    let q: Basic = mul(&[y.clone(), inv(&g)?])?;
                    numer.push(q.clone());
                    let st = compute_st(inv_map, &q);
                    inv_map.insert(q, st);
                }
                g = add(&[g, one()])?;
                cont = true;
            }
            let g1: Basic = add(&[g.clone(), int(-1)])?;
            if let Some(y) = find_fuzzy(inv_map, denom, &g1)? {
                remove_first(denom, &y);
                if y != g1 {
                    let q: Basic = mul(&[g1.clone(), inv(&y)?])?;
                    numer.push(q.clone());
                    let st = compute_st(inv_map, &q);
                    inv_map.insert(q, st);
                }
                g = g1;
                cont = true;
            }
        }
        new.push(g);
    }
    *gammas = new;
    Ok(())
}

/// Factors `x` (`x - 1` in the other part) absorbed into `gamma(x)` when
/// their quotient is a number.
fn fuzzy(ng: &mut Vec<Basic>, dg: &mut Vec<Basic>, no: &mut Vec<Basic>, d_o: &mut Vec<Basic>) -> PolyResult<()> {
    let mut inv_map: HashMap<Basic, Signature> = HashMap::new();
    for e in ng.iter().chain(dg.iter()).chain(no.iter()).chain(d_o.iter()) {
        let st = compute_st(&inv_map, e);
        inv_map.insert(e.clone(), st);
    }
    fuzzy_pass(&mut inv_map, ng, no, d_o)?;
    fuzzy_pass(&mut inv_map, dg, d_o, no)
}

fn product_of(list: &[Basic]) -> PolyResult<Basic> {
    mul(list)
}

// PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp.rule_gamma
fn rule_gamma(expr: &Basic, as_comb: bool, level: u8) -> PolyResult<Basic> {
    let mut expr: Basic = expr.clone();
    let mut level: u8 = level;
    if expr.args().is_empty() {
        return Ok(expr);
    }
    if level == 0 {
        let mut args: Vec<Basic> = Vec::new();
        for x in expr.args() {
            args.push(if is_expr(x) {
                rule_gamma(x, as_comb, 1)?
            } else {
                x.clone()
            });
        }
        expr = crate::core::function::rebuild_evaluated(&expr, args)?;
        level = 1;
    }
    if !expr.is_mul() {
        return Ok(expr);
    }
    if level == 1 {
        let commutative = |f: &Basic| is(f, Predicate::commutative) != Some(false);
        let args: Vec<Basic> = factors_of(&expr).into_iter().filter(|f| commutative(f)).collect();
        let nc: Vec<Basic> = factors_of(&expr).into_iter().filter(|f| !commutative(f)).collect();
        if args.is_empty() {
            return Ok(expr);
        }
        if !nc.is_empty() {
            return mul(&[rule_gamma(&mul(&args)?, as_comb, 2)?, mul(&nc)?]);
        }
        level = 2;
    }
    if level == 2 {
        let found: Vec<Basic> = factors_of(&expr);
        let independent: Vec<Basic> = found.iter().filter(|f| !gamma_factor(f)).cloned().collect();
        let dependent: Vec<Basic> = found.iter().filter(|f| gamma_factor(f)).cloned().collect();
        let gamma_ind: Basic = mul(&independent)?;
        let d: Basic = mul(&dependent)?;
        let (mut nd, mut dd) = as_numer_denom(&d)?;
        for ipass in 0..2 {
            let mut factors: Vec<Basic> = ordered(factors_of(&nd));
            for slot in factors.iter_mut() {
                let ni: Basic = slot.clone();
                if ni.is_add() {
                    let mut terms: Vec<Basic> = Vec::new();
                    for a in ni.args() {
                        let quotient: Basic = mul(&[a.clone(), inv(&dd)?])?;
                        terms.push(rule_gamma(&gamma_rat(&quotient)?, as_comb, 3)?);
                    }
                    let (n_new, d_new) = as_numer_denom(&add(&terms)?)?;
                    *slot = n_new;
                    dd = d_new;
                    if !has_gamma(&dd) {
                        break;
                    }
                }
            }
            nd = mul(&factors)?;
            if ipass == 0 && !gamma_factor(&nd) {
                break;
            }
            std::mem::swap(&mut nd, &mut dd);
        }
        expr = mul(&[gamma_ind, nd.clone(), inv(&dd)?])?;
        if !(expr.is_mul() && (gamma_factor(&dd) || gamma_factor(&nd))) {
            return Ok(expr);
        }
        level = 3;
    }
    if level == 3 {
        loop {
            let was: Basic = expr.clone();
            expr = rule_gamma(&expr, as_comb, 4)?;
            if expr == was {
                return Ok(expr);
            }
        }
    }

    let mut numer_gammas: Vec<Basic> = Vec::new();
    let mut denom_gammas: Vec<Basic> = Vec::new();
    let mut numer_others: Vec<Basic> = Vec::new();
    let mut denom_others: Vec<Basic> = Vec::new();
    let mut newargs: Vec<Basic> = ordered(expr.args().to_vec());
    while let Some(arg) = newargs.pop() {
        let (n, d) = as_numer_denom(&arg)?;
        match explicate(&n)? {
            (Some(true), lst) => numer_gammas.extend(lst),
            (Some(false), lst) => numer_others.extend(lst),
            _ => {}
        }
        match explicate(&d)? {
            (Some(true), lst) => denom_gammas.extend(lst),
            (Some(false), lst) => denom_others.extend(lst),
            _ => {}
        }
    }
    if !as_comb {
        reflect(&mut numer_gammas, &mut numer_others, &mut denom_others)?;
        reflect(&mut denom_gammas, &mut denom_others, &mut numer_others)?;
        duplicate(
            &mut numer_gammas,
            &mut denom_gammas,
            &mut numer_others,
            &mut denom_others,
        )?;
        duplicate(
            &mut denom_gammas,
            &mut numer_gammas,
            &mut denom_others,
            &mut numer_others,
        )?;
        mult_thm(&mut numer_gammas, &mut numer_others)?;
        mult_thm(&mut denom_gammas, &mut denom_others)?;
    }
    fuzzy(
        &mut numer_gammas,
        &mut denom_gammas,
        &mut numer_others,
        &mut denom_others,
    )?;
    let ng: Vec<Basic> = numer_gammas.iter().map(gamma).collect::<PolyResult<_>>()?;
    let dg: Vec<Basic> = denom_gammas.iter().map(gamma).collect::<PolyResult<_>>()?;
    mul(&[
        product_of(&ng)?,
        inv(&product_of(&dg)?)?,
        product_of(&numer_others)?,
        inv(&product_of(&denom_others)?)?,
    ])
}

fn factor(e: &Basic) -> PolyResult<Basic> {
    crate::polys::polytools::factor(e, &[], false, true)
}

// PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp
fn gammasimp_inner(expr: &Basic, as_comb: bool) -> PolyResult<Basic> {
    let leaf: Leaf = if as_comb { comb_leaf } else { gamma_leaf };
    let integral: Basic = through_rf(expr, leaf, false)?;
    // not SymPy's: gamma functions whose arguments differ by an integer but
    // have a fractional shift (gamma(n + 3/2)/gamma(n + 1/2), left by SymPy,
    // issue #25) merged too, when that leaves fewer of them
    let fractional: Basic = through_rf(expr, leaf, true)?;
    let expr: Basic = if gamma_count(&fractional) < gamma_count(&integral) {
        fractional
    } else {
        integral
    };
    let was: Basic = factor(&expr)?;
    let mut expr: Basic = rule_gamma(&was, as_comb, 0)?;
    if expr != was {
        expr = factor(&expr)?;
    }
    crate::core::function::replace_where(&expr, &is_gamma, &|b: &Basic| {
        let arg: &Basic = &b.args()[0];
        match (arg.head(), arg.as_rational()) {
            (Head::Rational { .. }, Some(r)) => gamma_rational(r),
            _ => Ok(b.clone()),
        }
    })
}

/// The applications of `e` which do not depend on a variable bound by an
/// enclosing node (those surviving SymPy's `as_dummy`).
fn free_applications(e: &Basic, bound: &BTreeSet<Basic>, out: &mut Vec<Basic>) {
    if is_application(e) && e.free_symbols().intersection(bound).next().is_none() && !out.contains(e) {
        out.push(e.clone());
    }
    let own: Vec<Basic> = crate::core::traversal::bound_symbols(e);
    if own.is_empty() {
        for a in e.args() {
            free_applications(a, bound, out);
        }
    } else {
        let mut inner: BTreeSet<Basic> = bound.clone();
        inner.extend(own);
        for a in e.args() {
            free_applications(a, &inner, out);
        }
    }
}

fn applications(e: &Basic) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for node in e.preorder() {
        if is_application(&node) && !out.contains(&node) {
            out.push(node);
        }
    }
    out
}

fn rewrite_gamma(e: &Basic, piecewise: bool) -> PolyResult<Basic> {
    use crate::core::rewriting::{rewrite, RewriteHints};
    rewrite(e, &[], "gamma", &RewriteHints { piecewise })
}

// PORTED FROM SYMPY: sympy/simplify/gammasimp.py:gammasimp
/// The expression with its gamma functions (and the functions rewritten as
/// gamma) simplified (`gammasimp`).
pub fn gammasimp(expr: &Basic) -> PolyResult<Basic> {
    let e: Basic = rewrite_gamma(expr, true)?;
    let functions: Vec<Basic> = applications(&e);
    if !functions.iter().any(is_gamma) {
        return Ok(e);
    }
    let mut free: Vec<Basic> = Vec::new();
    free_applications(&e, &BTreeSet::new(), &mut free);
    let others: Vec<Basic> = functions
        .into_iter()
        .filter(|f| !is_gamma(f) && free.contains(f))
        .collect();
    if others.is_empty() {
        return gammasimp_inner(&e, false);
    }
    let fun: Vec<Basic> = ordered(others);
    let dum: Vec<Basic> = fun.iter().map(|_| crate::core::Dummy::new("Dummy", &[]).0).collect();
    let mut simp: Vec<Basic> = Vec::new();
    for f in &fun {
        let mut args: Vec<Basic> = Vec::new();
        for a in f.args() {
            args.push(if is_expr(a) {
                gammasimp_inner(a, false)?
            } else {
                a.clone()
            });
        }
        simp.push(crate::core::function::rebuild_evaluated(f, args)?);
    }
    let forward: HashMap<Basic, Basic> = fun.iter().cloned().zip(dum.iter().cloned()).collect();
    let d: Basic = crate::core::function::xreplace_evaluated(&e, &forward)?;
    let back: HashMap<Basic, Basic> = dum.into_iter().zip(simp).collect();
    crate::core::function::xreplace_evaluated(&gammasimp_inner(&d, false)?, &back)
}

// PORTED FROM SYMPY: sympy/simplify/combsimp.py:_gamma_as_comb
/// Gamma functions as factorials, and `(a + b)!/(a!*b!)` as binomials.
fn gamma_as_comb(expr: &Basic) -> PolyResult<Basic> {
    use crate::functions::combinatorial::factorials::{binomial, factorial};
    let expr: Basic = crate::core::function::replace_where(expr, &is_gamma, &|b: &Basic| {
        factorial(&add(&[b.args()[0].clone(), int(-1)])?)
    })?;
    let ops = |x: &Basic| crate::core::function::count_ops(x).unwrap_or(0);
    let f = |b: &Basic| -> PolyResult<Basic> {
        if !b.is_mul() {
            return Ok(b.clone());
        }
        let mut rvd: Vec<(Basic, Basic)> = Vec::new();
        for factor in factors_of(b) {
            let (base, e) = as_base_exp(&factor)?;
            match rvd.iter_mut().find(|(k, _)| *k == base) {
                Some((_, v)) => *v = add(&[v.clone(), e])?,
                None => rvd.push((base, e)),
            }
        }
        let mut numer: Vec<Basic> = Vec::new();
        let mut denom: Vec<Basic> = Vec::new();
        for (k, m) in rvd.iter_mut() {
            if matches!(k.head(), Head::Function(Function::Factorial)) {
                if let Some(n) = to_int(m) {
                    let count: usize = usize::try_from(n.unsigned_abs()).unwrap_or(0);
                    if n > 0 {
                        numer.extend(vec![k.args()[0].clone(); count]);
                    } else {
                        denom.extend(vec![k.args()[0].clone(); count]);
                    }
                    *m = Basic::zero();
                }
            }
        }
        if numer.is_empty() || denom.is_empty() {
            return Ok(b.clone());
        }
        let mut nd: [Vec<Basic>; 2] = [numer, denom];
        let mut hit: bool = false;
        for side in 0..2 {
            let mut i: usize = 0;
            while i < nd[side].len() {
                let ai: Basic = nd[side][i].clone();
                let mut paired: bool = false;
                for j in (i + 1)..nd[side].len() {
                    let aj: Basic = nd[side][j].clone();
                    let total: Basic = add(&[ai.clone(), aj.clone()])?;
                    if let Some(pos) = nd[1 - side].iter().position(|t| *t == total) {
                        hit = true;
                        nd[1 - side].remove(pos);
                        nd[side].remove(j);
                        nd[side].remove(i);
                        let chosen: &Basic = if ops(&ai) < ops(&aj) { &ai } else { &aj };
                        let key: Basic = binomial(&total, chosen)?;
                        let delta: Basic = int(if side == 0 { -1 } else { 1 });
                        match rvd.iter_mut().find(|(k, _)| *k == key) {
                            Some((_, v)) => *v = add(&[v.clone(), delta])?,
                            None => rvd.push((key, delta)),
                        }
                        paired = true;
                        break;
                    }
                }
                if !paired {
                    i += 1;
                }
            }
        }
        if !hit {
            return Ok(b.clone());
        }
        let mut top: Vec<Basic> = Vec::new();
        for (k, v) in &rvd {
            top.push(pow(k, v)?);
        }
        for k in &nd[0] {
            top.push(factorial(k)?);
        }
        let bottom: Vec<Basic> = nd[1].iter().map(factorial).collect::<PolyResult<_>>()?;
        mul(&[mul(&top)?, inv(&mul(&bottom)?)?])
    };
    crate::core::traversal::bottom_up(&expr, &f, false)
}

// PORTED FROM SYMPY: sympy/simplify/combsimp.py:combsimp
/// The combinatorial expression (factorials, binomials, rising and falling
/// factorials) simplified, as factorials and binomials; through
/// [`gammasimp`] when a gamma argument is not an integer (`combsimp`).
pub fn combsimp(expr: &Basic) -> PolyResult<Basic> {
    // binomial(n, k) is 0 for a negative integer k, whatever n: for an upper
    // index which may be a negative integer the quotient of gamma functions,
    // symmetric in k and n - k, is not both binomial(n, k) and
    // binomial(n, n - k). Binomials mirrored in the input are kept as they
    // are, and one which comes back mirrored is turned again (not SymPy's,
    // sympy/sympy#31275 V225, sympy/sympy#14612)
    let risky: Vec<Basic> = negative_upper_binomials(expr);
    // (binomial(2*n, n) and binomial(2*n + 2, n + 1) are mirrored and
    // aligned: one family)
    let conflicts: Vec<Basic> = risky
        .iter()
        .filter(|b| {
            risky
                .iter()
                .any(|c| mirrored(b, c).unwrap_or(false) && !aligned(b, c).unwrap_or(false))
        })
        .cloned()
        .collect();
    let kept: HashMap<Basic, Basic> = conflicts
        .iter()
        .map(|b| (b.clone(), crate::core::symbol::Dummy::new("binomial", &[]).0))
        .collect();
    let back: HashMap<Basic, Basic> = kept.iter().map(|(b, d)| (d.clone(), b.clone())).collect();
    let e: Basic = rewrite_gamma(&expr.xreplace(&kept), false)?;
    let result: Basic = if e
        .preorder()
        .iter()
        .any(|node| is_gamma(node) && is(&node.args()[0], Predicate::integer) != Some(true))
    {
        gammasimp(&e)?
    } else {
        gamma_as_comb(&gammasimp_inner(&e, true)?)?
    };
    let mut turned: HashMap<Basic, Basic> = HashMap::new();
    for b in negative_upper_binomials(&result) {
        let mut aligned_any: bool = false;
        let mut mirrored_any: bool = false;
        for c in &risky {
            aligned_any |= aligned(&b, c)?;
            mirrored_any |= mirrored(&b, c)?;
        }
        if aligned_any || !mirrored_any {
            continue;
        }
        let (upper, lower) = (&b.args()[0], &b.args()[1]);
        let mut mirror: Option<Basic> = None;
        for c in &risky {
            if c.args()[0] == *upper && add(&[c.args()[1].clone(), lower.clone()])? == *upper {
                mirror = Some(c.clone());
                break;
            }
        }
        let Some(m) = mirror else {
            // mirrored up to a shift: no form of the result is known to hold
            return Ok(expr.clone());
        };
        turned.insert(b, m);
    }
    Ok(result.xreplace(&turned).xreplace(&back))
}

/// The binomials of `e` whose upper index is an integer which may be
/// negative (`_negative_upper_binomials`).
fn negative_upper_binomials(e: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for node in e.preorder() {
        if node.is_function(Function::Binomial) && !found.contains(&node) {
            let upper: &Basic = &node.args()[0];
            if is(upper, Predicate::integer) == Some(true) && is(upper, Predicate::nonnegative) != Some(true) {
                found.push(node.clone());
            }
        }
    }
    found
}

fn constant_difference(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(crate::core::add::sub(a, b)?.is_integer_value())
}

/// The upper and the lower indices of the binomials differ by integers
/// (`_aligned`).
fn aligned(b: &Basic, c: &Basic) -> PolyResult<bool> {
    Ok(constant_difference(&b.args()[0], &c.args()[0])? && constant_difference(&b.args()[1], &c.args()[1])?)
}

/// The lower index of one binomial is, up to integers, the upper index
/// minus the lower index of the other (`_mirrored`).
fn mirrored(b: &Basic, c: &Basic) -> PolyResult<bool> {
    Ok(constant_difference(&b.args()[0], &c.args()[0])?
        && constant_difference(&add(&[b.args()[1].clone(), c.args()[1].clone()])?, &b.args()[0])?)
}
