//! Port of `sympy_extras/concrete/gosper.py` (SymPy's `gosper.py` and
//! `hypersimp`): Gosper's algorithm for the indefinite and definite sums
//! of hypergeometric terms. The polynomials in the summation variable are
//! dense lists of expressions (the constant first), each coefficient
//! cancelled; the Python operators are mirrored one at a time.

use crate::core::basic::{Basic, Head};
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;

type R = PolyResult<Basic>;

fn add(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn mul(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn div(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn pow(b: &Basic, e: &Basic) -> R {
    crate::core::power::pow(b, e)
}
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn c(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn factor(e: &Basic) -> R {
    crate::polys::polytools::factor(e, &[], false, true)
}

/// A polynomial in `n`: the coefficients, the constant first, each
/// cancelled (`_UPoly`).
#[derive(Clone, Debug)]
pub struct UPoly {
    pub c: Vec<Basic>,
}

impl UPoly {
    pub fn new(coeffs: &[Basic]) -> PolyResult<UPoly> {
        let mut cs: Vec<Basic> = coeffs.iter().map(c).collect::<PolyResult<_>>()?;
        while cs.last().is_some_and(|v| *v == Basic::zero()) {
            cs.pop();
        }
        Ok(UPoly { c: cs })
    }

    /// `e`, a polynomial in `n` (`ValueError` otherwise).
    pub fn of(e: &Basic, n: &Basic) -> PolyResult<UPoly> {
        let mut by: std::collections::BTreeMap<i64, Vec<Basic>> = std::collections::BTreeMap::new();
        let ex: Basic = crate::core::function::expand_default(e)?;
        for t in crate::core::expr::terms_of(&ex) {
            let (coeff, k) = crate::solvers::solvers::as_coeff_exponent(&t, n)?;
            let k_int: Option<i64> = if k.is_integer_value() {
                k.as_rational().and_then(|r| i64::try_from(&numer(r)).ok())
            } else {
                None
            };
            match k_int {
                Some(kv) if !coeff.has(n) && kv >= 0 => by.entry(kv).or_default().push(coeff),
                _ => return Err(PolyError::Value(format!("{} is not a polynomial in {}", e, n))),
            }
        }
        let top: i64 = by.keys().max().copied().unwrap_or(0);
        let coeffs: Vec<Basic> = (0..=top)
            .map(|k| crate::core::add::add(by.get(&k).map(|v| v.as_slice()).unwrap_or(&[])))
            .collect::<PolyResult<_>>()?;
        UPoly::new(&coeffs)
    }

    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }
    pub fn degree(&self) -> i64 {
        self.c.len() as i64 - 1
    }
    pub fn lc(&self) -> Basic {
        self.c.last().cloned().unwrap_or_else(Basic::zero)
    }
    pub fn nth(&self, k: i64) -> Basic {
        if k >= 0 && (k as usize) < self.c.len() {
            self.c[k as usize].clone()
        } else {
            Basic::zero()
        }
    }
    /// The polynomial as an expression in `n`.
    /// The derivative (`diff`).
    pub fn diff(&self) -> PolyResult<UPoly> {
        let terms: Vec<Basic> = self
            .c
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, c)| crate::core::mul::mul(&[c.clone(), Basic::int(k as i64)]))
            .collect::<PolyResult<_>>()?;
        UPoly::new(&terms)
    }

    pub fn as_expr(&self, n: &Basic) -> R {
        let terms: Vec<Basic> = self
            .c
            .iter()
            .enumerate()
            .map(|(k, cv)| mul(cv, &pow(n, &int(k as i64))?))
            .collect::<PolyResult<_>>()?;
        crate::core::add::add(&terms)
    }
    fn minus(&self, other: &UPoly) -> PolyResult<UPoly> {
        let size: usize = self.c.len().max(other.c.len());
        let v: Vec<Basic> = (0..size)
            .map(|k| sub(&self.nth(k as i64), &other.nth(k as i64)))
            .collect::<PolyResult<_>>()?;
        UPoly::new(&v)
    }
    fn times(&self, other: &UPoly) -> PolyResult<UPoly> {
        if self.is_zero() || other.is_zero() {
            return UPoly::new(&[]);
        }
        let mut out: Vec<Basic> = vec![Basic::zero(); self.c.len() + other.c.len() - 1];
        for (i, a) in self.c.iter().enumerate() {
            for (j, b) in other.c.iter().enumerate() {
                out[i + j] = add(&out[i + j], &mul(a, b)?)?;
            }
        }
        UPoly::new(&out)
    }
    fn scale(&self, k: &Basic) -> PolyResult<UPoly> {
        let v: Vec<Basic> = self.c.iter().map(|cv| mul(cv, k)).collect::<PolyResult<_>>()?;
        UPoly::new(&v)
    }
    fn monic(&self) -> PolyResult<UPoly> {
        if self.c.is_empty() {
            return Ok(self.clone());
        }
        self.scale(&div(&Basic::one(), &self.lc())?)
    }
    pub fn divmod(&self, other: &UPoly) -> PolyResult<(UPoly, UPoly)> {
        let mut rem: Vec<Basic> = self.c.clone();
        let mut q: Vec<Basic> = vec![Basic::zero(); (rem.len() as i64 - other.c.len() as i64 + 1).max(1) as usize];
        let lc: Basic = other.lc();
        while rem.len() >= other.c.len() && !rem.is_empty() {
            let k: usize = rem.len() - other.c.len();
            let f: Basic = c(&div(rem.last().expect("a term"), &lc)?)?;
            q[k] = f.clone();
            for (j, b) in other.c.iter().enumerate() {
                rem[j + k] = c(&sub(&rem[j + k], &mul(&f, b)?)?)?;
            }
            while rem.last().is_some_and(|v| *v == Basic::zero()) {
                rem.pop();
            }
        }
        Ok((UPoly::new(&q)?, UPoly::new(&rem)?))
    }
    pub fn quo(&self, other: &UPoly) -> PolyResult<UPoly> {
        Ok(self.divmod(other)?.0)
    }
    pub fn gcd(&self, other: &UPoly) -> PolyResult<UPoly> {
        let (mut a, mut b) = (self.clone(), other.clone());
        while !b.is_zero() {
            let r: UPoly = a.divmod(&b)?.1;
            a = b;
            b = r;
        }
        if a.is_zero() {
            UPoly::new(&[Basic::one()])
        } else {
            a.monic()
        }
    }
    /// `p(n + s)`.
    pub fn shift(&self, s: &Basic) -> PolyResult<UPoly> {
        let mut out: Vec<Basic> = vec![Basic::zero(); self.c.len()];
        for (k, cv) in self.c.iter().enumerate() {
            for (j, slot) in out.iter_mut().enumerate().take(k + 1) {
                let binom: Basic =
                    crate::functions::combinatorial::factorials::binomial(&int(k as i64), &int(j as i64))?;
                *slot = add(slot, &mul(&mul(cv, &binom)?, &pow(s, &int((k - j) as i64))?)?)?;
            }
        }
        UPoly::new(&out)
    }
}

/// The determinant by fraction-free (Bareiss) elimination.
fn determinant(rows: &[Vec<Basic>]) -> R {
    let mut m: Vec<Vec<Basic>> = rows.to_vec();
    let size: usize = m.len();
    if size == 0 {
        return Ok(Basic::one());
    }
    let mut sign: i64 = 1;
    let mut previous: Basic = Basic::one();
    for k in 0..size - 1 {
        if m[k][k] == Basic::zero() {
            let Some(pivot) = (k + 1..size).find(|r| m[*r][k] != Basic::zero()) else {
                return Ok(Basic::zero());
            };
            m.swap(k, pivot);
            sign = -sign;
        }
        for i in k + 1..size {
            for j in k + 1..size {
                let v: Basic = sub(&mul(&m[i][j], &m[k][k])?, &mul(&m[i][k], &m[k][j])?)?;
                m[i][j] = c(&div(&v, &previous)?)?;
            }
        }
        previous = m[k][k].clone();
    }
    c(&mul(&int(sign), &m[size - 1][size - 1])?)
}

fn cleared(p: &UPoly, x: &Basic) -> R {
    let dens: Vec<Basic> =
        p.c.iter()
            .map(|cv| crate::core::expr::as_numer_denom(cv).map(|(_, d)| d))
            .collect::<PolyResult<_>>()?;
    let scale: Basic = crate::polys::polytools::lcm_list(&dens, &[])?;
    let terms: Vec<Basic> =
        p.c.iter()
            .enumerate()
            .map(|(k, cv)| mul(&c(&mul(cv, &scale)?)?, &pow(x, &int(k as i64))?))
            .collect::<PolyResult<_>>()?;
    crate::core::add::add(&terms)
}

/// The resultant of `a` and `b` up to a factor free of the symbols of `b`
/// not in `a`: by the typed polynomials, by the Sylvester determinant
/// when a coefficient is outside their domains.
fn resultant(a: &UPoly, b: &UPoly) -> R {
    let (da, db) = (a.degree(), b.degree());
    if da < 0 || db < 0 {
        return Ok(Basic::zero());
    }
    let size: usize = (da + db) as usize;
    if size == 0 {
        return Ok(Basic::one());
    }
    let x: Basic = Dummy::new("x", &[]).0;
    let typed = (|| -> R {
        let ca: Basic = cleared(a, &x)?;
        let cb: Basic = cleared(b, &x)?;
        crate::polys::polytools::resultant_expr(&ca, &cb, std::slice::from_ref(&x))
    })();
    match typed {
        Ok(r) => return Ok(r),
        Err(PolyError::NotImplemented(m)) if m.starts_with("not ported") => return Err(PolyError::NotImplemented(m)),
        Err(PolyError::Polys(..))
        | Err(PolyError::Value(_))
        | Err(PolyError::Type(_))
        | Err(PolyError::NotImplemented(_)) => {}
        Err(err) => return Err(err),
    }
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for k in 0..db as usize {
        let mut row: Vec<Basic> = vec![Basic::zero(); size];
        for (j, cv) in a.c.iter().rev().enumerate() {
            row[k + j] = cv.clone();
        }
        rows.push(row);
    }
    for k in 0..da as usize {
        let mut row: Vec<Basic> = vec![Basic::zero(); size];
        for (j, cv) in b.c.iter().rev().enumerate() {
            row[k + j] = cv.clone();
        }
        rows.push(row);
    }
    determinant(&rows)
}

/// The nonnegative integer roots of `r` in `h`, from the linear factors
/// of its factorization.
fn nonnegative_integer_roots(r: &Basic, h: &Basic) -> PolyResult<Vec<i64>> {
    let (numer_, _) = crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(r, false, true)?)?;
    let mut roots: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    let f: Basic = factor(&numer_)?;
    for t in crate::core::expr::factors_of(&f) {
        let base: Basic = if t.is_pow() { t.args()[0].clone() } else { t.clone() };
        if !base.has(h) {
            continue;
        }
        let p: UPoly = match UPoly::of(&base, h) {
            Ok(p) => p,
            Err(PolyError::Value(_)) => continue,
            Err(err) => return Err(err),
        };
        if p.degree() == 1 {
            let root: Basic = c(&div(&crate::core::mul::neg(&p.nth(0))?, &p.nth(1))?)?;
            if root.is_integer_value() {
                if let Some(v) = root.as_rational().and_then(|q| i64::try_from(&numer(q)).ok()) {
                    if v >= 0 {
                        roots.insert(v);
                    }
                }
            }
        }
    }
    Ok(roots.into_iter().collect())
}

/// The Gosper normal form `f/g = Z*A*C(n + 1)/(B*C(n))`: `(Z*A, B, C)`
/// (`gosper_normal`).
pub fn gosper_normal(f: &Basic, g: &Basic, n: &Basic) -> PolyResult<(UPoly, UPoly, UPoly)> {
    let (p, q) = (UPoly::of(f, n)?, UPoly::of(g, n)?);
    let (a, mut big_a) = (p.lc(), p.monic()?);
    let (b, mut big_b) = (q.lc(), q.monic()?);
    let mut big_c: UPoly = UPoly::new(&[Basic::one()])?;
    let z: Basic = c(&div(&a, &b)?)?;
    let h: Basic = Dummy::new("h", &[]).0;
    let shifted: UPoly = big_b.shift(&h)?;
    let r: Basic = resultant(&big_a, &shifted)?;
    for i in nonnegative_integer_roots(&r, &h)? {
        let d: UPoly = big_a.gcd(&big_b.shift(&int(i))?)?;
        big_a = big_a.quo(&d)?;
        big_b = big_b.quo(&d.shift(&int(-i))?)?;
        for j in 1..=i {
            big_c = big_c.times(&d.shift(&int(-j))?)?;
        }
    }
    Ok((big_a.scale(&z)?, big_b, big_c))
}

/// The ratio `f(k + 1)/f(k)` with the functions in gamma, expanded and
/// the powers combined, when it is a rational function of `k`
/// (`consecutive_ratio`).
pub fn consecutive_ratio(f: &Basic, k: &Basic) -> PolyResult<Option<Basic>> {
    let shifted: Basic = crate::core::subs::subs(f, k, &add(k, &Basic::one())?)?;
    let mut g: Basic = div(&shifted, f)?;
    g = crate::core::rewriting::rewrite(&g, &[], "gamma", &crate::core::rewriting::RewriteHints::default())?;
    if g.preorder().iter().any(|n| matches!(n.head(), Head::Piecewise)) {
        g = crate::functions::elementary::piecewise::piecewise_fold(&g, None)?;
        let last: &Basic = g
            .args()
            .last()
            .ok_or_else(|| PolyError::Type("a Piecewise is expected".to_string()))?;
        g = last.args()[0].clone();
    }
    let hints = crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: true,
    };
    g = crate::core::function::expand_with(&g, true, hints)?;
    g = crate::simplify::powsimp::powsimp(
        &g,
        true,
        crate::simplify::powsimp::Combine::Exp,
        false,
        &crate::simplify::powsimp::count_ops,
    )?;
    let rational: bool =
        crate::core::expr::polynomial_like(&g, std::slice::from_ref(k), crate::core::expr::PolynomialKind::Rational)
            == Some(true);
    Ok(if rational { Some(g) } else { None })
}

/// `consecutive_ratio` simplified (`hypersimp`).
pub fn hypersimp(f: &Basic, k: &Basic) -> PolyResult<Option<Basic>> {
    let Some(g) = consecutive_ratio(f, k)? else {
        return Ok(None);
    };
    let options = crate::simplify::simplify::Options {
        ratio: f64::INFINITY,
        ..Default::default()
    };
    Ok(Some(crate::simplify::simplify::simplify(&g, &options)?))
}

fn solve_linear(equations: &[Basic], unknowns: &[Basic]) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    let mut rows: Vec<Vec<Basic>> = Vec::new();
    for eq in equations {
        let mut row: Vec<Basic> = unknowns
            .iter()
            .map(|u| c(&crate::core::function::derivative(eq, u)?))
            .collect::<PolyResult<_>>()?;
        let mut constant: Basic = eq.clone();
        for u in unknowns {
            constant = crate::core::subs::subs(&constant, u, &Basic::zero())?;
        }
        row.push(c(&crate::core::mul::neg(&constant)?)?);
        rows.push(row);
    }
    let mut pivots: Vec<usize> = Vec::new();
    let mut r: usize = 0;
    let width: usize = unknowns.len();
    for col in 0..width {
        let Some(pivot) = (r..rows.len()).find(|i| rows[*i][col] != Basic::zero()) else {
            continue;
        };
        rows.swap(r, pivot);
        let lead: Basic = rows[r][col].clone();
        rows[r] = rows[r].iter().map(|v| c(&div(v, &lead)?)).collect::<PolyResult<_>>()?;
        for i in 0..rows.len() {
            if i != r && rows[i][col] != Basic::zero() {
                let f: Basic = rows[i][col].clone();
                let new: Vec<Basic> = rows[i]
                    .iter()
                    .zip(&rows[r])
                    .map(|(v, w)| c(&sub(v, &mul(&f, w)?)?))
                    .collect::<PolyResult<_>>()?;
                rows[i] = new;
            }
        }
        pivots.push(col);
        r += 1;
    }
    for row in &rows[r..] {
        if *row.last().expect("a constant") != Basic::zero() {
            return Ok(None);
        }
    }
    let mut solution: Vec<(Basic, Basic)> = Vec::new();
    for (i, col) in pivots.iter().enumerate() {
        let mut rest: Basic = rows[i][width].clone();
        for j in 0..width {
            if j != *col && !pivots.contains(&j) && rows[i][j] != Basic::zero() {
                rest = sub(&rest, &mul(&rows[i][j], &unknowns[j])?)?;
            }
        }
        solution.push((unknowns[*col].clone(), c(&rest)?));
    }
    Ok(Some(solution))
}

/// The rational function `g` for which `g*f` is an antidifference of `f`
/// (`gosper_term`); `None` when `f` is not Gosper-summable.
pub fn gosper_term(f: &Basic, n: &Basic) -> PolyResult<Option<Basic>> {
    let Some(r) = hypersimp(f, n)? else { return Ok(None) };
    let (p, q) = crate::core::expr::as_numer_denom(&r)?;
    let (big_a, big_b, big_c) = gosper_normal(&p, &q, n)?;
    let big_b: UPoly = big_b.shift(&Basic::negative_one())?;
    let (nn, m, kk) = (big_a.degree(), big_b.degree(), big_c.degree());
    let mut d_set: Vec<Basic> = Vec::new();
    if nn != m || big_a.lc() != big_b.lc() {
        d_set.push(int(kk - nn.max(m)));
    } else if nn == 0 {
        d_set.push(int(kk - nn + 1));
        d_set.push(Basic::zero());
    } else {
        d_set.push(int(kk - nn + 1));
        d_set.push(c(&div(&sub(&big_b.nth(nn - 1), &big_a.nth(nn - 1))?, &big_a.lc())?)?);
    }
    let candidates: Vec<i64> = d_set
        .iter()
        .filter(|d| d.is_integer_value())
        .filter_map(|d| d.as_rational().and_then(|q| i64::try_from(&numer(q)).ok()))
        .filter(|v| *v >= 0)
        .collect();
    let Some(d) = candidates.iter().max().copied() else {
        return Ok(None);
    };
    let coeffs: Vec<Basic> = (0..=d).map(|k| Dummy::new(&format!("c{}", k), &[]).0).collect();
    let reversed: Vec<Basic> = coeffs.iter().rev().cloned().collect();
    let x: UPoly = UPoly::new(&reversed)?;
    let h: UPoly = big_a
        .times(&x.shift(&Basic::one())?)?
        .minus(&big_b.times(&x)?)?
        .minus(&big_c)?;
    let Some(solution) = solve_linear(&h.c, &coeffs)? else {
        return Ok(None);
    };
    let mut xe: Basic = x.as_expr(n)?;
    xe = crate::core::subs::subs_pairs(&xe, &solution)?;
    for coeff in &coeffs {
        if !solution.iter().any(|(k, _)| k == coeff) {
            xe = crate::core::subs::subs(&xe, coeff, &Basic::zero())?;
        }
    }
    if crate::core::expr::is(&xe, crate::core::assumptions::Predicate::zero) == Some(true) || xe == Basic::zero() {
        return Ok(None);
    }
    Ok(Some(div(&mul(&big_b.as_expr(n)?, &xe)?, &big_c.as_expr(n)?)?))
}

/// The definite sum of `e` over `(var, a, b)` from its certificate `g`
/// (`gosper_definite`).
pub fn gosper_definite(e: &Basic, g: &Basic, var: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<Basic>> {
    let upper_e: Basic = mul(e, &add(g, &Basic::one())?)?;
    let lower_e: Basic = mul(e, g)?;
    let upper: Basic = crate::core::subs::subs(&upper_e, var, b)?;
    let lower: Basic = crate::core::subs::subs(&lower_e, var, a)?;
    let mut result: Option<Basic> = Some(sub(&upper, &lower)?);
    if result.as_ref().is_some_and(|r| matches!(r.head(), Head::NaN)) {
        let lim = (|| -> R {
            sub(
                &crate::core::limits::limit(&upper_e, var, b, "+")?,
                &crate::core::limits::limit(&lower_e, var, a, "+")?,
            )
        })();
        result = match lim {
            Ok(v) => Some(v),
            Err(PolyError::NotImplemented(m)) if !m.starts_with("not ported") => None,
            Err(err) => return Err(err),
        };
    }
    match result {
        Some(r) => Ok(Some(factor(&r)?)),
        None => Ok(None),
    }
}

/// The sum of the hypergeometric term `f` by Gosper's algorithm:
/// indefinite in `k`, or definite over `(k, a, b)` (`gosper_sum`).
pub fn gosper_sum(f: &Basic, k: &Basic, bounds: Option<(&Basic, &Basic)>) -> PolyResult<Option<Basic>> {
    let Some(g) = gosper_term(f, k)? else { return Ok(None) };
    match bounds {
        None => Ok(Some(factor(&mul(f, &g)?)?)),
        Some((a, b)) => gosper_definite(f, &g, k, a, b),
    }
}
