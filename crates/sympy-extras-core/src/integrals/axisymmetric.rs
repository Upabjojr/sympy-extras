//! Region integrals with a rotational symmetry, ported from
//! `sympy_extras.integrals.axisymmetric`: when a group of `k >= 2`
//! integration variables enters the integrand and every relation of the
//! condition only through the sum of their squares, the integral reduces
//! to one over the profile in the half-space `rho > 0`, weighted by the
//! area of the unit sphere `S^(k-1)` times `rho**(k-1)`; a ball cut by
//! tilted planes first goes to the frame along their normal.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::basic::{Basic, Head, RelOp};
use crate::error::PolyResult;
use crate::integrals::regions::Integrator;

type R = PolyResult<Basic>;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn sqrt(e: &Basic) -> R {
    pow(e, &Basic::from_rational(crate::numbers::rat(1, 2)))
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn rational(p: i64, q: i64) -> Basic {
    Basic::from_rational(crate::numbers::rat(p, q))
}

/// A dummy with a name of its own (`_fresh`).
fn fresh(name: &str) -> Basic {
    let n: usize = COUNTER.fetch_add(1, Ordering::Relaxed);
    crate::core::symbol::Dummy::new(&format!("{}{}", name, n), &[]).0
}

/// `e.xreplace(rule)` with the rebuilt nodes evaluated (`_evaluated`).
#[allow(clippy::mutable_key_type)]
fn evaluated(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: std::collections::HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

fn attempt<T>(f: impl FnOnce() -> PolyResult<T>) -> PolyResult<Option<T>> {
    crate::timeout::attempt_within(crate::settings::settings().timeout, f)
}

fn is_relation(e: &Basic) -> bool {
    matches!(e.head(), Head::Relational(_))
}

/// The relations of `formula`, by their sort key (`_relations`).
fn relations(formula: &Basic) -> Vec<Basic> {
    let mut found: Vec<Basic> = formula.atoms(Some(&is_relation));
    found.sort_by_key(crate::core::sorting::sort_key);
    found
}

fn difference(atom: &Basic) -> R {
    sub(&atom.args()[0], &atom.args()[1])
}

/// The pairs of `0..n` in Python's `combinations` order.
fn pairs(n: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            out.push((i, j));
        }
    }
    out
}

/// The `k`-subsets of `0..n` in Python's `combinations` order.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn go(start: usize, n: usize, k: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            go(i + 1, n, k, current, out);
            current.pop();
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    if k <= n {
        go(0, n, k, &mut Vec::new(), &mut out);
    }
    out
}

/// Python's `str` of a tuple of expressions.
fn py_tuple(items: &[String]) -> String {
    if items.len() == 1 {
        return format!("({},)", items[0]);
    }
    format!("({})", items.join(", "))
}

/// The value of `e` as a complex number with 30 digits, `None` when it is
/// not a number.
fn value_at(e: &Basic, point: &[(Basic, Basic)]) -> PolyResult<Option<(f64, f64)>> {
    let at: Basic = evaluated(e, point)?;
    if !crate::core::assumptions::is_number(&at) {
        return Ok(None);
    }
    Ok(crate::core::evalf::numeric_value(&at, 100).map(|z| {
        let (re, im) = crate::core::evalf::parts(&z);
        (re.to_f64().value(), im.to_f64().value())
    }))
}

/// Whether `e` takes the same value at random rational points and at their
/// images under exact rotations in the coordinate planes of `group` (the
/// angles 37, 90, 127 and 143 degrees) (`rotation_invariant`).
pub fn rotation_invariant(e: &Basic, group: &[Basic], samples: usize) -> PolyResult<bool> {
    let p = crate::core::printing::sstr;
    let seed: String = py_tuple(&[p(e), py_tuple(&group.iter().map(p).collect::<Vec<String>>())]);
    let mut rng = crate::random::PyRandom::from_string(&seed);
    let mut others: Vec<Basic> = e.free_symbols().into_iter().filter(|s| !group.contains(s)).collect();
    others.sort_by_key(crate::core::sorting::sort_key);
    let variables: Vec<Basic> = group.iter().chain(&others).cloned().collect();
    for _ in 0..samples {
        let point: Vec<(Basic, Basic)> = variables
            .iter()
            .map(|v| (v.clone(), rational(rng.randint(-300, 300), 100)))
            .collect();
        let Some(base) = value_at(e, &point)? else {
            return Ok(false);
        };
        for (i, j) in pairs(group.len()) {
            for t in [rational(1, 3), int(1), int(2), int(3)] {
                let t2: Basic = pow(&t, &int(2))?;
                let denominator: Basic = add(&[int(1), t2.clone()])?;
                let c: Basic = div(&sub(&int(1), &t2)?, &denominator)?;
                let s: Basic = div(&mul(&[int(2), t.clone()])?, &denominator)?;
                let (pu, pv) = (point[i].1.clone(), point[j].1.clone());
                let mut rotated: Vec<(Basic, Basic)> = point.clone();
                rotated[i].1 = sub(&mul(&[c.clone(), pu.clone()])?, &mul(&[s.clone(), pv.clone()])?)?;
                rotated[j].1 = add(&[mul(&[s.clone(), pu])?, mul(&[c.clone(), pv])?])?;
                let Some(value) = value_at(e, &rotated)? else {
                    return Ok(false);
                };
                let gap: f64 = (base.0 - value.0).hypot(base.1 - value.1);
                let scale: f64 = 1.0 + base.0.hypot(base.1) + value.0.hypot(value.1);
                if !gap.is_finite() || !scale.is_finite() || gap > scale / 1e12 {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// Whether `e` depends on the variables of `group` through the sum of
/// their squares only: annihilated by every infinitesimal rotation and
/// unchanged by finite rotations at random points (`_invariant`).
fn invariant(e: &Basic, group: &[Basic]) -> PolyResult<bool> {
    if !rotation_invariant(e, group, 3)? {
        return Ok(false);
    }
    for (i, j) in pairs(group.len()) {
        let (u, v) = (&group[i], &group[j]);
        let generator: Basic = sub(
            &mul(&[u.clone(), crate::core::function::derivative(e, v)?])?,
            &mul(&[v.clone(), crate::core::function::derivative(e, u)?])?,
        )?;
        if generator == Basic::zero() {
            continue;
        }
        let Some(reduced) = attempt(|| cancel(&generator))? else {
            return Ok(false);
        };
        if reduced != Basic::zero() && !crate::integrals::conditions::numerically_equal(&reduced, &int(0), &[], 3)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The largest group of at least two variables through whose sum of
/// squares alone the integrand and every relation of the condition depend
/// on them; `None` when there is none (`rotation_group`).
pub fn rotation_group(f: &Basic, formula: &Basic, names: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    let mut expressions: Vec<Basic> = vec![f.clone()];
    for atom in relations(formula) {
        expressions.push(difference(&atom)?);
    }
    for size in (2..=names.len()).rev() {
        for chosen in subsets(names.len(), size) {
            let group: Vec<Basic> = chosen.iter().map(|&i| names[i].clone()).collect();
            let mut all: bool = true;
            for e in &expressions {
                if !invariant(e, &group)? {
                    all = false;
                    break;
                }
            }
            if all {
                return Ok(Some(group));
            }
        }
    }
    Ok(None)
}

/// The area of the unit sphere `S^(k-1)`, `2 pi**(k/2)/gamma(k/2)`
/// (`_sphere_area`).
fn sphere_area(k: usize) -> R {
    let half: Basic = rational(k as i64, 2);
    div(
        &mul(&[int(2), pow(&Basic::pi(), &half)?])?,
        &crate::functions::special::gamma_functions::gamma(&half)?,
    )
}

/// The coefficients of a polynomial of degree one in `names`, `None`
/// otherwise (`_normal`).
fn normal(p: &Basic, names: &[Basic]) -> PolyResult<Option<Vec<Basic>>> {
    if !names.iter().any(|n| p.has(n)) {
        return Ok(None);
    }
    let mut cs: Vec<Basic> = vec![Basic::zero(); names.len()];
    for term in crate::core::expr::terms_of(&expand(p)?) {
        let (coefficient, dependent) = independent_of_all(&term, names)?;
        if dependent == Basic::one() {
            continue;
        }
        let Some(i) = names.iter().position(|v| *v == dependent) else {
            return Ok(None);
        };
        cs[i] = add(&[cs[i].clone(), coefficient])?;
    }
    if cs.iter().all(|c| *c == Basic::zero()) {
        return Ok(None);
    }
    Ok(Some(cs))
}

fn independent_of_all(e: &Basic, names: &[Basic]) -> PolyResult<(Basic, Basic)> {
    let depends = |t: &Basic| names.iter().any(|s| t.has(s));
    if !e.is_mul() {
        return Ok(if depends(e) {
            (Basic::one(), e.clone())
        } else {
            (e.clone(), Basic::one())
        });
    }
    let (mut free, mut dependent): (Vec<Basic>, Vec<Basic>) = (Vec::new(), Vec::new());
    for f in e.args() {
        if depends(f) {
            dependent.push(f.clone());
        } else {
            free.push(f.clone());
        }
    }
    Ok((mul(&free)?, mul(&dependent)?))
}

/// Whether the nonzero vectors `u` and `v` are proportional: every minor
/// of order two cancels (`_proportional`).
fn proportional(u: &[Basic], v: &[Basic]) -> PolyResult<bool> {
    for (i, j) in pairs(u.len()) {
        let minor: Basic = sub(
            &mul(&[u[i].clone(), v[j].clone()])?,
            &mul(&[u[j].clone(), v[i].clone()])?,
        )?;
        if cancel(&expand(&minor)?)? != Basic::zero() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn expand_mul(e: &Basic) -> R {
    crate::core::function::expand_mul(e, true)
}

fn zero_verdict(v: &[Basic]) -> Option<bool> {
    let verdicts: Vec<Option<bool>> = v
        .iter()
        .map(|a| crate::core::expr::is(a, crate::core::assumptions::Predicate::zero))
        .collect();
    if verdicts.iter().all(|t| *t == Some(true)) {
        return Some(true);
    }
    if verdicts.contains(&Some(false)) {
        return Some(false);
    }
    None
}

/// SymPy's `Matrix.orthogonalize(*vectors, normalize=True,
/// rankcheck=False)`: the Gram-Schmidt process with the projections
/// expanded, the vectors which become zero dropped, the others normalised
/// (`_orthogonalize`).
fn orthogonalize(vectors: &[Vec<Basic>]) -> PolyResult<Vec<Vec<Basic>>> {
    // PORTED FROM SYMPY: sympy/matrices/decompositions.py:_QRdecomposition_optional
    let dot = |u: &[Basic], v: &[Basic]| -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (a, b) in u.iter().zip(v) {
            terms.push(mul(&[
                crate::functions::elementary::complexes::conjugate(a)?,
                b.clone(),
            ])?);
        }
        add(&terms)
    };
    let mut q: Vec<Vec<Basic>> = vectors.to_vec();
    let mut ranked: Vec<usize> = Vec::new();
    for j in 0..q.len() {
        for i in 0..j {
            if zero_verdict(&q[i]) == Some(true) {
                continue;
            }
            let r: Basic = expand_mul(&div(&dot(&q[i], &q[j])?, &dot(&q[i], &q[i])?)?)?;
            let qi: Vec<Basic> = q[i].clone();
            q[j] = q[j]
                .iter()
                .zip(&qi)
                .map(|(a, b)| sub(a, &mul(&[b.clone(), r.clone()])?))
                .collect::<PolyResult<_>>()?;
        }
        q[j] = q[j].iter().map(expand_mul).collect::<PolyResult<_>>()?;
        if zero_verdict(&q[j]) != Some(true) {
            ranked.push(j);
        }
    }
    let mut basis: Vec<Vec<Basic>> = Vec::new();
    for j in ranked {
        let mut terms: Vec<Basic> = Vec::new();
        for a in &q[j] {
            terms.push(pow(&crate::functions::elementary::complexes::abs_(a)?, &int(2))?);
        }
        let norm: Basic = sqrt(&add(&terms)?)?;
        basis.push(q[j].iter().map(|a| div(a, &norm)).collect::<PolyResult<_>>()?);
    }
    Ok(basis)
}

/// The transformed problem of [`aligned_frame`]: the integrand, the
/// condition and the variables in the new frame; the similarity has ratio
/// `1/ratio` (`Frame`).
#[derive(Clone, Debug)]
pub struct Frame {
    pub integrand: Basic,
    pub condition: Basic,
    pub variables: Vec<Basic>,
    pub ratio: Basic,
}

/// The integrand, the condition and the variables in the frame whose first
/// axis is the common normal `a` of the relations of degree one, when every
/// other relation is invariant under all rotations, scaled by `|a|`;
/// `None` when there is no such frame (`aligned_frame`).
pub fn aligned_frame(f: &Basic, formula: &Basic, names: &[Basic]) -> PolyResult<Option<Frame>> {
    let mut common: Option<Vec<Basic>> = None;
    let atoms: Vec<Basic> = relations(formula);
    for atom in &atoms {
        let p: Basic = difference(atom)?;
        if let Some(found) = normal(&p, names)? {
            match &common {
                None => common = Some(found),
                Some(c) if !proportional(c, &found)? => return Ok(None),
                _ => {}
            }
            continue;
        }
        if !invariant(&p, names)? {
            return Ok(None);
        }
    }
    let Some(common) = common else { return Ok(None) };
    let n: usize = names.len();
    let mut squares: Vec<Basic> = Vec::new();
    for c in &common {
        squares.push(mul(&[c.clone(), c.clone()])?);
    }
    let length: Basic = sqrt(&add(&squares)?)?;
    let mut vectors: Vec<Vec<Basic>> = vec![common.clone()];
    for j in 0..n {
        vectors.push((0..n).map(|i| if i == j { int(1) } else { int(0) }).collect());
    }
    let basis: Vec<Vec<Basic>> = orthogonalize(&vectors)?;
    if basis.len() != n {
        return Ok(None);
    }
    let ys: Vec<Basic> = (1..=n).map(|j| fresh(&format!("Y{}_", j))).collect();
    let mut image: Vec<Basic> = vec![Basic::zero(); n];
    for (e, yj) in basis.iter().zip(&ys) {
        for i in 0..n {
            image[i] = add(&[image[i].clone(), div(&mul(&[e[i].clone(), yj.clone()])?, &length)?])?;
        }
    }
    let replacement: Vec<(Basic, Basic)> = names
        .iter()
        .zip(&image)
        .map(|(v, im)| Ok((v.clone(), cancel(im)?)))
        .collect::<PolyResult<_>>()?;
    let mut new_atoms: Vec<(Basic, Basic)> = Vec::new();
    for atom in &atoms {
        let rewritten: Basic = cancel(&expand(&evaluated(&difference(atom)?, &replacement)?)?)?;
        let Head::Relational(op) = atom.head() else { continue };
        let relation: Basic = crate::core::relational::evaluate_relation(*op, &rewritten, &int(0))?;
        new_atoms.push((atom.clone(), relation));
    }
    let condition: Basic = evaluated(formula, &new_atoms)?;
    let integrand: Basic = cancel(&expand(&evaluated(f, &replacement)?)?)?;
    Ok(Some(Frame {
        integrand,
        condition,
        variables: ys,
        ratio: length,
    }))
}

/// The integral of `f` over the region `formula` of the variables `names`
/// through the profile of its rotational symmetry; `None` when there is no
/// group of variables to rotate or the profile integral is not evaluated
/// (`axisymmetric_integral`).
pub fn axisymmetric_integral(
    f: &Basic,
    formula: &Basic,
    names: &[Basic],
    assumptions: &[Basic],
    measure: &str,
    integrator: Integrator,
) -> PolyResult<Option<Basic>> {
    let Some(group) = rotation_group(f, formula, names)? else {
        // a ball cut by tilted planes: the frame along their normal
        let Some(frame) = aligned_frame(f, formula, names)? else {
            return Ok(None);
        };
        if rotation_group(&frame.integrand, &frame.condition, &frame.variables)?.is_none() {
            return Ok(None);
        }
        let Some(value) = axisymmetric_integral(
            &frame.integrand,
            &frame.condition,
            &frame.variables,
            assumptions,
            measure,
            integrator,
        )?
        else {
            return Ok(None);
        };
        // a similarity of ratio 1/|a| scales a measure of dimension d by
        // |a|**(-d)
        let equations: usize = formula
            .atoms(Some(&|n: &Basic| matches!(n.head(), Head::Relational(RelOp::Eq))))
            .len();
        let dimension: usize = names.len() - if measure == "hausdorff" { equations } else { 0 };
        return Ok(Some(div(&value, &pow(&frame.ratio, &int(dimension as i64))?)?));
    };
    let k: usize = group.len();
    // a plain dummy: with positive=True the relation rho > 0 would vanish
    let rho: Basic = fresh("rho");
    let mut replacement: Vec<(Basic, Basic)> = vec![(group[0].clone(), rho.clone())];
    for v in &group[1..] {
        replacement.push((v.clone(), int(0)));
    }
    let profile: Basic = evaluated(formula, &replacement)?;
    let rest: Vec<Basic> = names.iter().filter(|v| !group.contains(v)).cloned().collect();
    let weight: Basic = mul(&[sphere_area(k)?, pow(&rho, &int(k as i64 - 1))?])?;
    let reduced: Basic = mul(&[evaluated(f, &replacement)?, weight])?;
    let condition: Basic = crate::core::logic::and_(&[profile, crate::core::relational::gt(&rho, &Basic::zero())?]);
    // the radius last first: the profile is solved for it; the other order
    // is the fallback
    let mut last: Vec<Basic> = rest.clone();
    last.push(rho.clone());
    let mut first: Vec<Basic> = vec![rho.clone()];
    first.extend(rest);
    for order in [last, first] {
        let value: Option<Basic> = attempt(|| {
            crate::integrals::regions::integrate_by_ranges(
                &reduced,
                &condition,
                Some(&order),
                assumptions,
                Some(integrator),
                measure,
                None,
            )
        })?;
        if let Some(v) = value {
            let unevaluated: bool = v.preorder().iter().any(|n| matches!(n.head(), Head::IntegralByRanges));
            if !unevaluated && !v.free_symbols().contains(&rho) {
                return Ok(Some(v));
            }
        }
    }
    Ok(None)
}
