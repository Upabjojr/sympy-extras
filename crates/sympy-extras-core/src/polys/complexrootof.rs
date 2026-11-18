//! Complex roots of polynomials with rational coefficients as nodes of the
//! core, ported from `sympy_extras.polys.complexrootof` (SymPy's
//! `CRootOf`): the node [`Head::ComplexRootOf`] (`CRootOf(f, k)`, never
//! evaluating; two roots of the same polynomial in different variables are
//! equal, the variable being bound), the evaluating [`rootof`] and
//! [`all_roots`], [`real_roots`], [`root_count`], [`polynomial_roots`],
//! [`complex_rectangles`], [`algebraic_number`] and
//! [`real_roots_indexed`].
//!
//! The order of the roots is SymPy's: the real roots first, increasing;
//! then the others factor by factor (SymPy's `ordered` of the factors), the
//! roots of a factor by the bisection rectangles of Collins and Krandick,
//! the root below the real axis just before its conjugate. The rectangles
//! are found from exact isolating regions (`qqbar::Algebraic`), a root on a
//! cut line recognised exactly as a real root of the gcd of the real and
//! imaginary parts of `f` along the line.

// PORTED FROM SYMPY: sympy/polys/rootoftools.py (ComplexRootOf), sympy/polys/rootisolation.py
// (dup_isolate_complex_roots_sqf: the bisection order) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

#![allow(clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Head};
use crate::core::evalf::{complex, real_rat, Complex};
use crate::error::{value_error, PolyError, PolyResult, PolysErrorKind};
use crate::numbers::*;

use super::densetools::dup_shift;
use super::domains::{IntegerRing, QQ, ZZ};
use super::euclidtools::dup_qq_gcd;
use super::factortools::dup_zz_factor;
use super::numberfields::qqbar::{Algebraic, Box4};
use super::polytools::Poly;
use super::rootisolation::{dup_isolate_real_roots_sqf, dup_refine_real_root, Interval};
use super::sqfreetools::dup_sqf_part;

/// A polynomial with integer coefficients, highest degree first.
pub type IntegerPoly = Vec<Int>;

/// A root by its irreducible factor and its index among its roots.
pub type Address = (IntegerPoly, usize);

/// The coefficients and the variable of a `CRootOf` node (its head).
#[derive(Debug, Clone)]
pub struct RootData {
    pub coefficients: Vec<Rat>,
    pub variable: Basic,
}

const KEPT: usize = 2000;

// ---------------------------------------------------------------------------
// The roots on the lines of the cuts

/// The common real roots of the real and imaginary parts of a polynomial
/// along a line, as isolating intervals refined on demand.
struct LineRoots {
    g: Vec<Rat>,
    intervals: Vec<Interval>,
}

impl LineRoots {
    fn refined(&mut self, eps: &Rat) -> Vec<Interval> {
        self.intervals = self
            .intervals
            .iter()
            .map(|(a, b)| dup_refine_real_root(&self.g, a.clone(), b.clone(), eps))
            .collect();
        self.intervals.clone()
    }
}

fn stripped(f: Vec<Rat>) -> Vec<Rat> {
    let k: usize = f.iter().take_while(|c| c.is_zero()).count();
    f[k..].to_vec()
}

fn line_roots(real: Vec<Rat>, imaginary: Vec<Rat>) -> LineRoots {
    let g: Vec<Rat> = dup_qq_gcd(&stripped(real), &stripped(imaginary));
    if g.len() < 2 {
        return LineRoots {
            g: Vec::new(),
            intervals: Vec::new(),
        };
    }
    let g: Vec<Rat> = dup_sqf_part(&g, &QQ);
    let intervals: Vec<Interval> = dup_isolate_real_roots_sqf(&g, None);
    LineRoots { g, intervals }
}

/// The roots `c + i y` of `f`, `y` real.
fn vertical(f: &[Int], c: &Rat) -> LineRoots {
    let rational: Vec<Rat> = f.iter().map(rat_from_int).collect();
    let h: Vec<Rat> = dup_shift(&rational, c, &QQ);
    let n: usize = h.len() - 1;
    let (mut real, mut imaginary): (Vec<Rat>, Vec<Rat>) = (Vec::new(), Vec::new());
    // h(i y) = sum h_k i**k y**k, h_k the coefficient of degree k
    for k in (0..=n).rev() {
        let coefficient: &Rat = &h[n - k];
        let signed: Rat = if k % 4 == 0 || k % 4 == 1 {
            coefficient.clone()
        } else {
            -coefficient.clone()
        };
        if k % 2 == 0 {
            real.push(signed);
            imaginary.push(Rat::ZERO);
        } else {
            real.push(Rat::ZERO);
            imaginary.push(signed);
        }
    }
    line_roots(real, imaginary)
}

/// The roots `t + i c` of `f`, `t` real.
fn horizontal(f: &[Int], c: &Rat) -> LineRoots {
    // Horner's rule in t + i c, the coefficients as (real, imaginary)
    let (mut re, mut im): (Vec<Rat>, Vec<Rat>) = (Vec::new(), Vec::new());
    for a in f {
        let mut new_re: Vec<Rat> = re.clone();
        new_re.push(Rat::ZERO);
        let mut new_im: Vec<Rat> = im.clone();
        new_im.push(Rat::ZERO);
        for j in 0..re.len() {
            new_re[j + 1] = &new_re[j + 1] - c * &im[j];
            new_im[j + 1] = &new_im[j + 1] + c * &re[j];
        }
        let last: usize = new_re.len() - 1;
        new_re[last] = &new_re[last] + rat_from_int(a);
        re = new_re;
        im = new_im;
    }
    line_roots(re, im)
}

/// The roots of a polynomial on the lines of the cuts, computed once per
/// line (`_Lines`).
#[derive(Default)]
struct Lines {
    vertical: BTreeMap<Rat, LineRoots>,
    horizontal: BTreeMap<Rat, LineRoots>,
}

thread_local! {
    static LINES: RefCell<HashMap<IntegerPoly, Lines>> = RefCell::new(HashMap::new());
    /// the roots of the irreducible polynomials in SymPy's order, with the
    /// number of real roots (the refinements of the regions are kept)
    static ROOTS: RefCell<HashMap<IntegerPoly, (usize, Vec<Arc<Algebraic>>)>> = RefCell::new(HashMap::new());
}

/// `op` on the line roots of `f` on the line `c` (vertical or not).
fn with_line<T>(f: &[Int], c: &Rat, vertical_line: bool, op: impl FnOnce(&mut LineRoots) -> T) -> T {
    LINES.with(|cell| {
        let mut lines = cell.borrow_mut();
        if lines.len() > KEPT && !lines.contains_key(f) {
            lines.clear();
        }
        let entry: &mut Lines = lines.entry(f.to_vec()).or_default();
        let map: &mut BTreeMap<Rat, LineRoots> = if vertical_line {
            &mut entry.vertical
        } else {
            &mut entry.horizontal
        };
        let roots: &mut LineRoots = map.entry(c.clone()).or_insert_with(|| {
            if vertical_line {
                vertical(f, c)
            } else {
                horizontal(f, c)
            }
        });
        op(roots)
    })
}

/// A width for the line roots, below `eps` and the width of the region
/// they are to fit in.
fn finer(eps: &Rat, width: &Rat) -> Rat {
    if *width > Rat::ZERO {
        let quarter: Rat = width / rat(4, 1);
        if quarter < *eps {
            quarter
        } else {
            eps.clone()
        }
    } else {
        eps.clone()
    }
}

/// The sign of `re(z) - c` for a number which is not real, decided exactly.
fn compare_real_part(z: &Algebraic, c: &Rat, eps: &Rat) -> PolyResult<i32> {
    let mut eps: Rat = eps.clone();
    let minpoly: IntegerPoly = z.minpoly();
    loop {
        let (re_min, re_max, im_min, im_max) = z.enclosure(&eps)?;
        if re_max < *c {
            return Ok(-1);
        }
        if re_min > *c {
            return Ok(1);
        }
        let width: Rat = &im_max - &im_min;
        let target: Rat = finer(&eps, &width);
        let on_line: bool = with_line(&minpoly, c, true, |lines| {
            // a root of the minimal polynomial on the line, in the region
            // which isolates z, is z
            !lines.intervals.is_empty() && lines.refined(&target).iter().any(|(a, b)| im_min <= *a && *b <= im_max)
        });
        if on_line {
            return Ok(0);
        }
        eps = &eps / rat(16, 1);
    }
}

/// The sign of `im(z) - c`, decided exactly.
fn compare_imaginary_part(z: &Algebraic, c: &Rat, eps: &Rat) -> PolyResult<i32> {
    let mut eps: Rat = eps.clone();
    let minpoly: IntegerPoly = z.minpoly();
    loop {
        let (re_min, re_max, im_min, im_max) = z.enclosure(&eps)?;
        if im_max < *c {
            return Ok(-1);
        }
        if im_min > *c {
            return Ok(1);
        }
        let width: Rat = &re_max - &re_min;
        let target: Rat = finer(&eps, &width);
        let on_line: bool = with_line(&minpoly, c, false, |lines| {
            !lines.intervals.is_empty() && lines.refined(&target).iter().any(|(a, b)| re_min <= *a && *b <= re_max)
        });
        if on_line {
            return Ok(0);
        }
        eps = &eps / rat(16, 1);
    }
}

// ---------------------------------------------------------------------------
// SymPy's order of the roots of a square-free polynomial

/// SymPy's bound of the isolation: `2 max |a_i/a_n|`.
fn bound(f: &[Int]) -> Rat {
    let lc: Int = abs(&f[0]);
    let mut largest: Rat = Rat::ZERO;
    for c in f {
        let v: Rat = rat_from_parts(abs(c), lc.clone());
        if v > largest {
            largest = v;
        }
    }
    largest * rat(2, 1)
}

// PORTED FROM SYMPY: sympy/polys/rootisolation.py:dup_isolate_complex_roots_sqf
/// SymPy's isolating rectangles `(ax, ay, bx, by)` of the roots above the
/// real axis of the square-free `f` (integer coefficients, highest degree
/// first), in SymPy's order, each with the index of its root in `upper`
/// (all the roots of `f` above the axis).
pub fn complex_rectangles(f: &[Int], upper: &[Arc<Algebraic>]) -> PolyResult<Vec<(Box4, usize)>> {
    if upper.is_empty() {
        return Ok(Vec::new());
    }
    let b: Rat = bound(f);
    let two: Rat = rat(2, 1);
    let four: Rat = rat(4, 1);
    let mut found: Vec<(Box4, usize)> = Vec::new();
    let mut cells: Vec<(Rat, Rat, Rat, Rat, Vec<usize>)> =
        vec![(-b.clone(), Rat::ZERO, b.clone(), b, (0..upper.len()).collect())];
    let mut first: bool = true;
    while let Some((u, v, s, t, roots)) = cells.pop() {
        if roots.len() == 1 && !first {
            found.push(((u, v, s, t), roots[0]));
            continue;
        }
        first = false;
        if &s - &u > &t - &v {
            let x: Rat = (&u + &s) / &two;
            let eps: Rat = (&s - &u) / &four;
            let mut left: Vec<usize> = Vec::new();
            let mut right: Vec<usize> = Vec::new();
            for r in roots {
                if compare_real_part(&upper[r], &x, &eps)? < 0 {
                    left.push(r);
                } else {
                    right.push(r);
                }
            }
            if !left.is_empty() {
                cells.push((u.clone(), v.clone(), x.clone(), t.clone(), left));
            }
            if !right.is_empty() {
                cells.push((x, v, s, t, right));
            }
        } else {
            let y: Rat = (&v + &t) / &two;
            let eps: Rat = (&t - &v) / &four;
            let mut lower: Vec<usize> = Vec::new();
            let mut higher: Vec<usize> = Vec::new();
            for r in roots {
                if compare_imaginary_part(&upper[r], &y, &eps)? <= 0 {
                    lower.push(r);
                } else {
                    higher.push(r);
                }
            }
            if !lower.is_empty() {
                cells.push((u.clone(), v.clone(), s.clone(), y.clone(), lower));
            }
            if !higher.is_empty() {
                cells.push((u, y, s, t, higher));
            }
        }
    }
    found.sort_by(|a, b| (&a.0 .0, &a.0 .1).cmp(&(&b.0 .0, &b.0 .1)));
    Ok(found)
}

/// Whether a number which is not real lies above the real axis.
fn above(z: &Algebraic) -> PolyResult<bool> {
    let mut eps: Rat = Rat::ONE;
    loop {
        let (_, _, im_min, im_max) = z.enclosure(&eps)?;
        if im_min > Rat::ZERO {
            return Ok(true);
        }
        if im_max < Rat::ZERO {
            return Ok(false);
        }
        eps = &eps / rat(16, 1);
    }
}

/// A stable sort of real numbers by their order (Python's `sort` with
/// `_RealKey`).
fn sort_reals<T: Clone>(items: Vec<T>, value: impl Fn(&T) -> Arc<Algebraic>) -> PolyResult<Vec<T>> {
    let mut sorted: Vec<T> = Vec::new();
    for item in items {
        let mut position: usize = sorted.len();
        while position > 0 && value(&item).compare(&value(&sorted[position - 1]))? < 0 {
            position -= 1;
        }
        sorted.insert(position, item);
    }
    Ok(sorted)
}

/// The number of real roots and the roots of the irreducible `f`
/// (primitive, positive leading coefficient) in SymPy's order.
fn irreducible_roots(f: &[Int]) -> PolyResult<(usize, Vec<Arc<Algebraic>>)> {
    if let Some(found) = ROOTS.with(|cell| cell.borrow().get(f).cloned()) {
        return Ok(found);
    }
    let result: (usize, Vec<Arc<Algebraic>>) = if f.len() == 2 {
        let value: Rat = rat_from_parts(-f[1].clone(), f[0].clone());
        (1, vec![Arc::new(Algebraic::from_rational(&value))])
    } else {
        let (reals, others) = super::numberfields::qqbar::roots(f)?;
        let reals: Vec<Arc<Algebraic>> = sort_reals(reals.into_iter().map(Arc::new).collect(), |a| a.clone())?;
        let mut upper: Vec<Arc<Algebraic>> = Vec::new();
        for r in others {
            if above(&r)? {
                upper.push(Arc::new(r));
            }
        }
        let mut ordered: Vec<Arc<Algebraic>> = reals.clone();
        for (_, i) in complex_rectangles(f, &upper)? {
            ordered.push(Arc::new(upper[i].conjugate()));
            ordered.push(upper[i].clone());
        }
        (reals.len(), ordered)
    };
    ROOTS.with(|cell| {
        let mut roots = cell.borrow_mut();
        if roots.len() > KEPT {
            roots.clear();
        }
        roots.insert(f.to_vec(), result.clone());
    });
    Ok(result)
}

fn is_known_irreducible(f: &[Int]) -> bool {
    ROOTS.with(|cell| cell.borrow().contains_key(f))
}

// ---------------------------------------------------------------------------
// Polynomials

/// The coefficients times their common denominator.
fn integer_coefficients(coefficients: &[Rat]) -> Vec<Int> {
    let mut denominator: Int = Int::ONE;
    for c in coefficients {
        denominator = lcm(&denominator, &denom(c));
    }
    coefficients
        .iter()
        .map(|c| numer(&(c * rat_from_int(&denominator))))
        .collect()
}

fn factor_expr(f: &[Int], x: &Basic) -> PolyResult<Basic> {
    let n: usize = f.len() - 1;
    let terms: Vec<(Vec<u32>, Rat)> = f
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| (vec![(n - i) as u32], rat_from_int(c)))
        .collect();
    Poly::from_dict(&terms, std::slice::from_ref(x))?.as_expr()
}

// PORTED FROM SYMPY: sympy/polys/rootoftools.py:_pure_factors
/// The irreducible factors over `ZZ` with their multiplicities.
fn pure_factors(coefficients: &[Rat]) -> PolyResult<Vec<(IntegerPoly, usize)>> {
    let k: IntegerRing = ZZ;
    let (_, factors) = dup_zz_factor(&integer_coefficients(coefficients), &k)?;
    Ok(factors.into_iter().filter(|(g, _)| g.len() > 1).collect())
}

/// SymPy's `ordered` of the factors: `_nodes` then `default_sort_key`.
fn factor_key(f: &[Int], x: &Basic) -> PolyResult<(usize, crate::core::sorting::SortKey)> {
    let e: Basic = factor_expr(f, x)?;
    Ok((e.count_nodes(), crate::core::sorting::sort_key(&e)))
}

// PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._all_roots
/// The roots of the polynomial in SymPy's order, with multiplicities, by
/// their factors and indices: the real roots increasing, then (without
/// `real`) the others factor by factor.
fn addresses(coefficients: &[Rat], x: &Basic, real: bool) -> PolyResult<Vec<Address>> {
    let factors: Vec<(IntegerPoly, usize)> = pure_factors(coefficients)?;
    let mut reals: Vec<(Arc<Algebraic>, Address, usize)> = Vec::new();
    for (g, k) in &factors {
        let (count, roots) = irreducible_roots(g)?;
        for (i, r) in roots.iter().enumerate().take(count) {
            reals.push((r.clone(), (g.clone(), i), *k));
        }
    }
    let reals = sort_reals(reals, |item| item.0.clone())?;
    let mut result: Vec<Address> = Vec::new();
    for (_, address, k) in reals {
        for _ in 0..k {
            result.push(address.clone());
        }
    }
    if real {
        return Ok(result);
    }
    let mut keyed: Vec<((usize, crate::core::sorting::SortKey), IntegerPoly, usize)> = Vec::new();
    for (g, k) in factors {
        keyed.push((factor_key(&g, x)?, g, k));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, g, k) in keyed {
        let (count, roots) = irreducible_roots(&g)?;
        for i in count..roots.len() {
            for _ in 0..k {
                result.push((g.clone(), i));
            }
        }
    }
    Ok(result)
}

// PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._roots_trivial
/// The root of a linear factor, and with `radicals` of a quadratic one or
/// of a binomial, in radicals.
fn trivial(f: &[Int], index: usize, x: &Basic, radicals: bool) -> PolyResult<Option<Basic>> {
    if f.len() == 2 {
        return Ok(Some(Basic::from_rational(rat_from_parts(-f[1].clone(), f[0].clone()))));
    }
    if !radicals {
        return Ok(None);
    }
    let nonzero: usize = f.iter().filter(|c| !c.is_zero()).count();
    if f.len() == 3 || (nonzero == 2 && !f[f.len() - 1].is_zero()) {
        let poly: Poly = Poly::new(&factor_expr(f, x)?, std::slice::from_ref(x))?;
        if let Some(found) = super::roots::trivial_roots(&poly)? {
            if index < found.len() {
                return Ok(Some(found[index].clone()));
            }
        }
    }
    Ok(None)
}

fn root(address: &Address, x: &Basic, radicals: bool) -> PolyResult<Basic> {
    let (f, index) = address;
    if let Some(t) = trivial(f, *index, x, radicals)? {
        return Ok(t);
    }
    complex_root_of(&factor_expr(f, x)?, &Basic::integer(Int::from(*index)))
}

/// The roots of the polynomial with the rational `coefficients` (highest
/// degree first) in `x`, as SymPy's `Poly.all_roots` (with `real`,
/// `Poly.real_roots`) gives them.
pub fn polynomial_roots(coefficients: &[Rat], x: &Basic, radicals: bool, real: bool) -> PolyResult<Vec<Basic>> {
    if coefficients.iter().all(|c| c.is_zero()) {
        return Err(value_error("the roots of the zero polynomial".to_string()));
    }
    let (coeff, scaled_coefficients) = preprocessed(coefficients, x)?;
    let mut result: Vec<Basic> = Vec::new();
    for address in addresses(&scaled_coefficients, x, real)? {
        result.push(scaled(&coeff, root(&address, x, radicals)?)?);
    }
    Ok(result)
}

// PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._get_roots
/// SymPy's `preprocess_roots`: the roots are `coeff` times those of the
/// polynomial returned.
fn preprocessed(coefficients: &[Rat], x: &Basic) -> PolyResult<(Basic, Vec<Rat>)> {
    let n: usize = coefficients.len() - 1;
    let terms: Vec<(Vec<u32>, Rat)> = coefficients
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| (vec![(n - i) as u32], c.clone()))
        .collect();
    let poly: Poly = Poly::from_dict(&terms, std::slice::from_ref(x))?;
    let (coeff, scaled_poly) = super::polyroots::preprocess_roots(&poly)?;
    let mut found: Vec<Rat> = Vec::new();
    for c in scaled_poly.all_coeffs()? {
        match c.as_rational() {
            Some(v) if c.is_number_node() => found.push(v.clone()),
            _ => {
                return Err(PolyError::Type(format!(
                    "rational coefficients are expected, got {}",
                    c
                )))
            }
        }
    }
    Ok((coeff, found))
}

fn scaled(coeff: &Basic, root: Basic) -> PolyResult<Basic> {
    if *coeff == Basic::one() {
        return Ok(root);
    }
    crate::core::mul::mul(&[coeff.clone(), root])
}

/// The rational coefficients of the univariate polynomial `f` and its
/// generator (`x`, or the one symbol of `f`).
fn as_poly(f: &Basic, x: Option<&Basic>) -> PolyResult<(Vec<Rat>, Basic)> {
    let x: Basic = match x {
        Some(x) => x.clone(),
        None => {
            let mut symbols: Vec<Basic> = f.free_symbols().into_iter().collect();
            symbols.sort_by_key(|s| s.to_string());
            if symbols.len() != 1 {
                return Err(value_error(format!("a univariate polynomial is expected, got {}", f)));
            }
            symbols.remove(0)
        }
    };
    let poly: Poly = match Poly::new(f, std::slice::from_ref(&x)) {
        Ok(p) => p,
        Err(PolyError::Polys(PolysErrorKind::CoercionFailed, _)) => {
            return Err(PolyError::NotImplemented(format!(
                "CRootOf is not supported over the coefficients of {}",
                f
            )))
        }
        Err(PolyError::Polys(_, _)) => return Err(value_error(format!("{} is not a polynomial in {}", f, x))),
        Err(e) => return Err(e),
    };
    let domain: String = poly.domain.name();
    if (domain != "ZZ" && domain != "QQ") || poly.gens.len() != 1 {
        return Err(PolyError::NotImplemented(format!(
            "CRootOf is not supported over {}",
            domain
        )));
    }
    let mut coefficients: Vec<Rat> = Vec::new();
    for c in poly.all_coeffs()? {
        match c.as_rational() {
            Some(v) if c.is_number_node() => coefficients.push(v.clone()),
            _ => {
                return Err(PolyError::NotImplemented(format!(
                    "CRootOf is not supported over {}",
                    domain
                )))
            }
        }
    }
    Ok((coefficients, x))
}

/// The roots of `f` with multiplicities, in SymPy's order (`all_roots`).
pub fn all_roots(f: &Basic, x: Option<&Basic>, radicals: bool) -> PolyResult<Vec<Basic>> {
    let (coefficients, gen) = as_poly(f, x)?;
    if coefficients.len() < 2 {
        return Ok(Vec::new());
    }
    polynomial_roots(&coefficients, &gen, radicals, false)
}

/// The real roots of `f` with multiplicities, increasing (`real_roots`).
pub fn real_roots(f: &Basic, x: Option<&Basic>, radicals: bool) -> PolyResult<Vec<Basic>> {
    let (coefficients, gen) = as_poly(f, x)?;
    if coefficients.len() < 2 {
        return Ok(Vec::new());
    }
    polynomial_roots(&coefficients, &gen, radicals, true)
}

/// The numbers of real roots and of all roots of `f`, with
/// multiplicities (`root_count`).
pub fn root_count(f: &Basic, x: Option<&Basic>) -> PolyResult<(usize, usize)> {
    let (coefficients, gen) = as_poly(f, x)?;
    let (_, scaled_coefficients) = preprocessed(&coefficients, &gen)?;
    Ok((
        addresses(&scaled_coefficients, &gen, true)?.len(),
        coefficients.len() - 1,
    ))
}

/// The `index`-th root of `f` in SymPy's order, evaluated as SymPy's
/// `rootof` (a negative index counts from the end).
pub fn rootof(f: &Basic, index: i64, x: Option<&Basic>, radicals: bool) -> PolyResult<Basic> {
    let (coefficients, gen) = as_poly(f, x)?;
    let degree: i64 = coefficients.len() as i64 - 1;
    if degree <= 0 {
        return Err(value_error(format!("cannot construct CRootOf object for {}", f)));
    }
    if index < -degree || index >= degree {
        return Err(PolyError::Index(format!(
            "root index out of [{}, {}] range, got {}",
            -degree,
            degree - 1,
            index
        )));
    }
    let k: i64 = if index < 0 { index + degree } else { index };
    let (coeff, scaled_coefficients) = preprocessed(&coefficients, &gen)?;
    let found: Vec<Address> = addresses(&scaled_coefficients, &gen, false)?;
    scaled(&coeff, root(&found[k as usize], &gen, radicals)?)
}

// ---------------------------------------------------------------------------
// The node

/// The integer of an `Integer` node.
fn index_of(k: &Basic) -> PolyResult<i64> {
    match k.head() {
        Head::Rational {
            integer_class: true,
            value,
        } => i64::try_from(value.numerator())
            .map_err(|_| value_error(format!("expected an integer root index, got {}", k))),
        _ => Err(value_error(format!("expected an integer root index, got {}", k))),
    }
}

/// The node `CRootOf(f, k)`: its arguments checked and stored, never
/// evaluated (`ComplexRootOf(f, k)`).
pub fn complex_root_of(f: &Basic, k: &Basic) -> PolyResult<Basic> {
    let index: i64 = index_of(k)?;
    let symbols: Vec<Basic> = f.free_symbols().into_iter().collect();
    if symbols.len() != 1 {
        return Err(value_error(format!(
            "CRootOf(f, k) takes a univariate polynomial, got {}",
            f
        )));
    }
    let (coefficients, variable) = as_poly(f, Some(&symbols[0]))?;
    if coefficients.len() < 2 {
        return Err(value_error(format!("cannot construct CRootOf object for {}", f)));
    }
    if index < 0 || index >= coefficients.len() as i64 - 1 {
        return Err(PolyError::Index(format!(
            "root index out of [0, {}] range, got {}",
            coefficients.len() - 2,
            index
        )));
    }
    let data: RootData = RootData { coefficients, variable };
    Ok(Basic::new(
        Head::ComplexRootOf(Arc::new(data)),
        vec![f.clone(), Basic::integer(Int::from(index))],
    ))
}

/// The data of a `CRootOf` node.
pub fn data_of(e: &Basic) -> &RootData {
    match e.head() {
        Head::ComplexRootOf(d) => d,
        _ => unreachable!("a CRootOf node"),
    }
}

/// The index of a `CRootOf` node.
pub fn index_of_node(e: &Basic) -> usize {
    index_of(&e.args()[1]).expect("a checked index") as usize
}

/// The node rebuilt from new arguments, evaluated (`_evaluated`: `rootof`
/// without radicals).
pub fn rebuild_root(args: &[Basic]) -> PolyResult<Basic> {
    if !matches!(
        args[1].head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    ) {
        return Err(PolyError::Type(format!(
            "invalid arguments of a CRootOf: ({}, {})",
            args[0], args[1]
        )));
    }
    rootof(&args[0], index_of(&args[1])?, None, false)
}

fn address_of(coefficients: &[Rat], x: &Basic, index: usize) -> PolyResult<Address> {
    if coefficients.len() > 2 {
        let integers: Vec<Int> = integer_coefficients(coefficients);
        let mut content: Int = Int::ZERO;
        for c in &integers {
            content = gcd(&content, c);
        }
        let mut normalised: Vec<Int> = integers.iter().map(|c| c / &content).collect();
        if normalised[0] < Int::ZERO {
            normalised = normalised.into_iter().map(|c| -c).collect();
        }
        if is_known_irreducible(&normalised) {
            // already known to be irreducible
            return Ok((normalised, index));
        }
    }
    Ok(addresses(coefficients, x, false)?[index].clone())
}

/// The irreducible factor of the root and its index among the roots of
/// that factor (`address`).
pub fn address(e: &Basic) -> PolyResult<Address> {
    let data: &RootData = data_of(e);
    address_of(&data.coefficients, &data.variable, index_of_node(e))
}

/// The algebraic number of the root (`algebraic`).
pub fn algebraic(e: &Basic) -> PolyResult<Arc<Algebraic>> {
    let (f, index) = address(e)?;
    Ok(irreducible_roots(&f)?.1[index].clone())
}

/// Whether the number which is not real is on the imaginary axis.
fn real_part_is_zero(z: &Algebraic) -> PolyResult<bool> {
    Ok(compare_real_part(z, &Rat::ZERO, &Rat::ONE)? == 0)
}

/// The facts (`_eval_facts`).
pub fn root_facts(e: &Basic) -> FactList {
    let mut facts: FactList = FactList::new();
    let Ok(a) = algebraic(e) else {
        return facts;
    };
    for (p, v) in [
        (Predicate::commutative, true),
        (Predicate::complex, true),
        (Predicate::finite, true),
        (Predicate::algebraic, true),
        (Predicate::transcendental, false),
    ] {
        facts.set(p, v);
    }
    if a.is_real() {
        let Ok(s) = a.sign() else {
            return facts;
        };
        for (p, v) in [
            (Predicate::real, true),
            (Predicate::extended_real, true),
            (Predicate::imaginary, false),
            (Predicate::zero, s == 0),
            (Predicate::positive, s > 0),
            (Predicate::negative, s < 0),
            (Predicate::rational, a.is_rational()),
        ] {
            facts.set(p, v);
        }
        if !a.is_rational() {
            facts.set(Predicate::irrational, true);
        }
        return facts;
    }
    let imaginary: bool = real_part_is_zero(&a).unwrap_or(false);
    for (p, v) in [
        (Predicate::real, false),
        (Predicate::extended_real, false),
        (Predicate::zero, false),
        (Predicate::rational, false),
        (Predicate::integer, false),
        (Predicate::imaginary, imaginary),
    ] {
        facts.set(p, v);
    }
    facts
}

/// The polynomial and the index of a `CRootOf` node as SymPy's `CRootOf`
/// has them: the irreducible factor in the variable of the node
/// (`ComplexRootOf.polynomial_root`).
pub fn polynomial_root(e: &Basic) -> PolyResult<Option<(Poly, usize)>> {
    let Head::ComplexRootOf(data) = e.head() else {
        return Ok(None);
    };
    let (f, index) = address(e)?;
    let x: Basic = data.variable.clone();
    Ok(Some((
        Poly::new(&factor_expr(&f, &x)?, std::slice::from_ref(&x))?,
        index,
    )))
}

/// The value of an algebraic number at `prec` bits: the middle of its
/// isolating region refined until each part has a relative width below
/// `2**-prec` (a real part which is exactly zero is zero).
pub fn complex_value(a: &Algebraic, prec: usize) -> PolyResult<Complex> {
    let power = |k: usize| -> Rat { rat_from_parts(Int::ONE, Int::from(2).pow(k)) };
    let mut eps: Rat = power(prec + 4);
    let floor: Rat = power(4 * prec + 64);
    let relative: Rat = power(prec + 2);
    let mut imaginary: Option<bool> = if a.is_real() { Some(false) } else { None };
    let two: Rat = rat(2, 1);
    let (re_mid, im_mid) = loop {
        let (mut re_min, mut re_max, im_min, im_max) = a.enclosure(&eps)?;
        let mut re_mid: Rat = (&re_min + &re_max) / &two;
        let im_mid: Rat = (&im_min + &im_max) / &two;
        if imaginary.is_none() && re_min <= Rat::ZERO && Rat::ZERO <= re_max {
            imaginary = Some(real_part_is_zero(a)?);
        }
        if imaginary == Some(true) {
            re_mid = Rat::ZERO;
            re_min = Rat::ZERO;
            re_max = Rat::ZERO;
        }
        let abs_rat = |r: &Rat| -> Rat {
            if *r < Rat::ZERO {
                -r.clone()
            } else {
                r.clone()
            }
        };
        let real_done: bool = re_mid.is_zero() || &re_max - &re_min <= abs_rat(&re_mid) * &relative;
        let imaginary_done: bool = im_mid.is_zero() || &im_max - &im_min <= abs_rat(&im_mid) * &relative;
        if (real_done && imaginary_done) || eps < floor {
            break (re_mid, im_mid);
        }
        eps = &eps / rat(4, 1);
        let mut smallest: Rat = Rat::ONE;
        let mut any: bool = false;
        for m in [&re_mid, &im_mid] {
            if !m.is_zero() {
                let v: Rat = abs_rat(m);
                if !any || v < smallest {
                    smallest = v;
                    any = true;
                }
            }
        }
        if smallest < Rat::ONE {
            let candidate: Rat = &smallest * power(prec + 4);
            if candidate < eps {
                eps = candidate;
            }
        }
    };
    Ok(complex(real_rat(&re_mid, prec), real_rat(&im_mid, prec)))
}

/// The numerical value of a `CRootOf` node.
pub fn root_value(e: &Basic, prec: usize) -> Option<Complex> {
    let a: Arc<Algebraic> = algebraic(e).ok()?;
    complex_value(&a, prec).ok()
}

/// The algebraic number of a rational, of `I`, of a `CRootOf` or of an
/// `IndexedRoot` with rational coefficients (`algebraic_number`).
pub fn algebraic_number(e: &Basic) -> PolyResult<Option<Arc<Algebraic>>> {
    if e.is_number_node() {
        if let Some(v) = e.as_rational() {
            return Ok(Some(Arc::new(Algebraic::from_rational(v))));
        }
    }
    if matches!(e.head(), Head::ImaginaryUnit) {
        return Ok(Some(Arc::new(Algebraic::imaginary_unit())));
    }
    if matches!(e.head(), Head::ComplexRootOf(_)) {
        return Ok(Some(algebraic(e)?));
    }
    crate::polys::cad::indexedroot::algebraic_of(e)
}

/// `e` with its real `CRootOf` nodes written as the canonical `IndexedRoot`
/// of the same number, in the variable of the node (`real_roots_indexed`).
pub fn real_roots_indexed(e: &Basic) -> PolyResult<Basic> {
    let mut rule: HashMap<Basic, Basic> = HashMap::new();
    for node in e.preorder() {
        if matches!(node.head(), Head::ComplexRootOf(_)) && !rule.contains_key(&node) {
            let a: Arc<Algebraic> = algebraic(&node)?;
            if a.is_real() {
                let name: String = data_of(&node).variable.to_string();
                rule.insert(node.clone(), crate::polys::cad::indexedroot::real_root(&a, &name)?);
            }
        }
    }
    if rule.is_empty() {
        return Ok(e.clone());
    }
    crate::core::function::xreplace_evaluated(e, &rule)
}

/// `CRootOf(f, k)`.
pub fn print_root(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    format!("CRootOf({}, {})", doprint(&e.args()[0]), index_of_node(e))
}
