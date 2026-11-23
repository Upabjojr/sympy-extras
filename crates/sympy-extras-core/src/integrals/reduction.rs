//! Port of `sympy_extras/integrals/reduction.py`: the Hermite reduction
//! with respect to `D = d/dx + q` (the reduction-based creative
//! telescoping of hyperexponential functions, Bostan–Chen–Chyzak–Li) and
//! the minimal telescopers, differential equations and integrals it
//! gives.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeMap;

use crate::core::basic::{Basic, Head};
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::ConditionalValue;
use crate::integrals::telescoping::{
    equation, is_hyperexponential, logarithmic_derivative, normalized, nullspace, polynomial_failure, solve, terms,
    verified, Context, DifferentialTelescoper, KPoly,
};
use crate::polys::domains::domain::Domain;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(args: &[Basic]) -> R {
    crate::core::add::add(args)
}
fn mul(args: &[Basic]) -> R {
    crate::core::mul::mul(args)
}
fn pow(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn div(a: &Basic, b: &Basic) -> R {
    mul(&[a.clone(), pow(b, &int(-1))?])
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::diff(e, &[(x.clone(), 1)])
}

/// A pole whose conjugates have different integer residues (`_Split`).
#[derive(Debug)]
enum Outcome<T> {
    Done(T),
    Split,
}

/// `value` as an integer when it is one (`_integer`).
fn integer(value: &Basic) -> PolyResult<Option<i64>> {
    let v: Basic = cancel(value)?;
    if !matches!(v.head(), Head::Rational { value, .. } if value.is_int()) {
        return Ok(None);
    }
    Ok(v.as_rational()
        .and_then(|r| i64::try_from(&crate::numbers::numer(r)).ok()))
}

/// The Hermite reduction with respect to `D(g) = g' + q*g` for a fixed
/// rational `q` (`HermiteReducer`).
#[derive(Clone, Debug)]
pub struct HermiteReducer {
    pub x: Basic,
    pub t: Basic,
    context: Context,
    pub shell: Basic,
    a: KPoly,
    b: KPoly,
    beta: KPoly,
    cofactor: KPoly,
    pub q: Basic,
    echelon: BTreeMap<isize, (KPoly, KPoly)>,
    basis_size: usize,
}

impl HermiteReducer {
    pub fn new(q: &Basic, x: &Basic, t: &Basic, parameters: &[Basic]) -> PolyResult<HermiteReducer> {
        let mut exprs: Vec<Basic> = vec![q.clone(), t.clone()];
        exprs.extend(parameters.iter().cloned());
        let mut context: Context = Context::new(&exprs, x, &[], false)?;
        let (a0, b0) = context.fraction(q)?;
        let (shell, a, b) = match normalize(&context, &a0, &b0)? {
            Outcome::Done(v) => v,
            Outcome::Split => {
                // conjugate poles with different integer residues: factor
                // over the Gaussian rationals
                context = context.extended(&[], true)?;
                let (a1, b1) = context.fraction(q)?;
                match normalize(&context, &a1, &b1)? {
                    Outcome::Done(v) => v,
                    Outcome::Split => unreachable!("no split over the Gaussian rationals"),
                }
            }
        };
        let beta: KPoly = squarefree_part(&context, &b)?;
        let cofactor: KPoly = b.quo(&beta)?;
        let q_value: Basic = context.fraction_expr(&a, &b)?;
        Ok(HermiteReducer {
            x: x.clone(),
            t: t.clone(),
            context,
            shell,
            a,
            b,
            beta,
            cofactor,
            q: q_value,
            echelon: BTreeMap::new(),
            basis_size: 0,
        })
    }

    pub fn a_expr(&self) -> R {
        self.context.poly_expr(&self.a)
    }

    pub fn b_expr(&self) -> R {
        self.context.poly_expr(&self.b)
    }

    /// Move every polynomial to the field of the generators of `exprs` too
    /// (`_upgrade`).
    fn upgrade(&mut self, exprs: &[Basic]) -> PolyResult<()> {
        let old: Context = self.context.clone();
        let new: Context = old.extended(exprs, false)?;
        self.a = new.convert(&self.a, &old)?;
        self.b = new.convert(&self.b, &old)?;
        self.beta = new.convert(&self.beta, &old)?;
        self.cofactor = new.convert(&self.cofactor, &old)?;
        let mut echelon: BTreeMap<isize, (KPoly, KPoly)> = BTreeMap::new();
        for (d, (v, h)) in &self.echelon {
            echelon.insert(*d, (new.convert(v, &old)?, new.convert(h, &old)?));
        }
        self.echelon = echelon;
        self.context = new;
        Ok(())
    }

    /// `D(g) = g' + q_hat*g` (`derivation`).
    pub fn derivation(&self, g: &Basic) -> R {
        cancel(&add(&[diff(g, &self.x)?, mul(&[self.q.clone(), g.clone()])?])?)
    }

    fn numerator_denominator(&mut self, p: &Basic) -> PolyResult<(KPoly, KPoly)> {
        if !self.context.covers(std::slice::from_ref(p))? {
            self.upgrade(std::slice::from_ref(p))?;
        }
        self.context.fraction(p)
    }

    /// `Q` with every factor of `B` removed (`_coprime_part`).
    fn coprime_part(&self, q: &KPoly) -> PolyResult<KPoly> {
        let mut result: KPoly = q.clone();
        loop {
            let common: KPoly = result.gcd(&self.beta)?;
            if common.degree() == 0 {
                return Ok(result);
            }
            result = result.quo(&common)?;
        }
    }

    /// Step 2: the poles of order at least two away from `B`
    /// (`_reduce_coprime_poles`).
    fn reduce_coprime_poles(&mut self, p: &Basic) -> PolyResult<(Basic, Basic)> {
        let mut g: Basic = Basic::zero();
        let mut current: Basic = p.clone();
        loop {
            let (pp, qq) = self.numerator_denominator(&current)?;
            let q_c: KPoly = self.coprime_part(&qq)?;
            let mut target: Option<(KPoly, usize)> = None;
            for (factor, multiplicity) in q_c.sqf_list().1 {
                if factor.degree() > 0 && multiplicity >= 2 {
                    target = Some((factor.monic()?, multiplicity));
                    break;
                }
            }
            let Some((s, n)) = target else {
                return Ok((g, current));
            };
            let k: &Context = &self.context;
            // leading coefficient of the pole: P / (Q / s^n) mod s
            let rest: KPoly = qq.quo(&s.pow(n))?;
            let lead: KPoly = pp.mul(&rest.invert(&s)?)?.rem(&s)?;
            let scaled: KPoly = s.diff(1).mul_ground(&k.field.from_int(1 - n as i64));
            let h: KPoly = lead.mul(&scaled.invert(&s)?)?.rem(&s)?;
            let g_step: Basic = div(&k.poly_expr(&h)?, &pow(&k.poly_expr(&s)?, &int(n as i64 - 1))?)?;
            g = add(&[g, g_step.clone()])?;
            current = cancel(&sub(&current, &self.derivation(&g_step)?)?)?;
        }
    }

    /// Step 3: the poles at the factors of `B` (`_reduce_b_poles`).
    fn reduce_b_poles(&mut self, p: &Basic) -> PolyResult<(Basic, Basic)> {
        let (pp, qq) = self.numerator_denominator(p)?;
        let q_c: KPoly = self.coprime_part(&qq)?;
        let q_b: KPoly = qq.quo(&q_c)?;
        let mut g: Basic = Basic::zero();
        let k: &Context = &self.context;
        let (beta, a, b, cofactor) = (&self.beta, &self.a, &self.b, &self.cofactor);
        if q_b.degree() == 0 && beta.degree() == 0 {
            return Ok((g, p.clone()));
        }
        // the smallest m with Q_B | B beta^m
        let mut m: i64 = 0;
        while !b.mul(&beta.pow(m as usize))?.rem(&q_b)?.is_zero() {
            m += 1;
        }
        let mut n: KPoly = pp.mul(&b.mul(&beta.pow(m as usize))?.mul(&q_c)?.quo(&qq)?)?;
        let beta_prime: KPoly = beta.diff(1);
        let inverse_c: KPoly = q_c.invert(beta)?;
        while m >= 0 {
            let m_: crate::integrals::telescoping::Element = k.field.from_int(m);
            let modulus: KPoly = a.sub(&beta_prime.mul_ground(&m_).mul(cofactor)?)?;
            let h: KPoly = n.mul(&inverse_c)?.mul(&modulus.invert(beta)?)?.rem(beta)?;
            let phi: KPoly = beta
                .mul(&h.diff(1))?
                .sub(&beta_prime.mul_ground(&m_).mul(&h)?)?
                .mul(cofactor)?
                .add(&a.mul(&h)?)?;
            g = add(&[g, div(&k.poly_expr(&h)?, &pow(&k.poly_expr(beta)?, &int(m))?)?])?;
            n = n.sub(&phi.mul(&q_c)?)?.quo(beta)?;
            m -= 1;
        }
        // now the fraction is N / (cofactor Q_c)
        if cofactor.degree() > 0 {
            let inverse_a: KPoly = a.mul(&q_c)?.invert(cofactor)?;
            let h: KPoly = n.mul(&inverse_a)?.rem(cofactor)?;
            let psi: KPoly = beta
                .mul(&h.diff(1))?
                .add(&beta_prime.mul(&h)?)?
                .mul(cofactor)?
                .add(&a.mul(&h)?)?;
            g = add(&[g, mul(&[k.poly_expr(&h)?, k.poly_expr(beta)?])?])?;
            n = n.sub(&psi.mul(&q_c)?)?.quo(cofactor)?;
        }
        Ok((g, k.fraction_expr(&n, &q_c)?))
    }

    /// `E(x**j) = B (x**j)' + (A + B') x**j` (`_image`).
    fn image(&self, j: usize) -> PolyResult<KPoly> {
        let monomial: KPoly = self.monomial(j);
        self.b
            .mul(&monomial.diff(1))?
            .add(&self.a.add(&self.b.diff(1))?.mul(&monomial)?)
    }

    fn monomial(&self, j: usize) -> KPoly {
        let f = &self.context.field;
        let mut c = vec![f.one()];
        c.extend(std::iter::repeat_with(|| f.zero()).take(j));
        self.context.from_elements(&c)
    }

    /// The row echelon form of `E(1), ..., E(x**size)` with the preimages
    /// (`_extend_basis`).
    fn extend_basis(&mut self, size: usize) -> PolyResult<()> {
        while self.basis_size < size + 1 {
            let j: usize = self.basis_size;
            let (mut vector, mut preimage) = (self.image(j)?, self.monomial(j));
            self.basis_size += 1;
            while !vector.is_zero() {
                let degree: isize = vector.degree();
                if let Some((pivot, pivot_preimage)) = self.echelon.get(&degree) {
                    let f = &self.context.field;
                    let factor = f.quo(&vector.LC(), &pivot.LC())?;
                    vector = vector.sub(&pivot.mul_ground(&factor))?;
                    preimage = preimage.sub(&pivot_preimage.mul_ground(&factor))?;
                } else {
                    self.echelon.insert(degree, (vector, preimage));
                    break;
                }
            }
        }
        Ok(())
    }

    /// Step 4: `(g, r)` with `P = D(g) + r` (`_reduce_polynomial`).
    fn reduce_polynomial(&mut self, p: &KPoly) -> PolyResult<(Basic, KPoly)> {
        if p.is_zero() {
            return Ok((Basic::zero(), p.clone()));
        }
        let n_a: isize = if self.a.is_zero() { -1 } else { self.a.degree() };
        let n_b: isize = self.b.degree();
        let delta: isize = n_a.max(n_b - 1);
        let mut size: isize = p.degree() - delta + 2;
        if n_a == n_b - 1 && n_b >= 1 {
            // E(x**j) has a lower degree for j = -(n_b + lc(A)/lc(B))
            let k: &Context = &self.context;
            let mu: Basic = add(&[
                k.element_expr(&k.field.quo(&self.a.LC(), &self.b.LC())?)?,
                int(n_b as i64),
            ])?;
            if let Some(exceptional) = integer(&crate::core::mul::neg(&mu)?)?.filter(|e| *e >= 0) {
                size = size.max(exceptional as isize + 2);
            }
        }
        self.extend_basis(size.max(1) as usize)?;
        let k: &Context = &self.context;
        let f = &k.field;
        let mut remainder: KPoly = p.clone();
        let mut preimage: KPoly = k.constant(&f.zero());
        for (&degree, (pivot, pivot_preimage)) in self.echelon.iter().rev() {
            if remainder.is_zero() || degree > remainder.degree() {
                continue;
            }
            let coefficient = remainder.coeff(degree as usize);
            if f.is_zero(&coefficient) {
                continue;
            }
            let factor = f.quo(&coefficient, &pivot.LC())?;
            remainder = remainder.sub(&pivot.mul_ground(&factor))?;
            preimage = preimage.add(&pivot_preimage.mul_ground(&factor))?;
        }
        let g: Basic = mul(&[k.poly_expr(&self.b)?, k.poly_expr(&preimage)?])?;
        Ok((g, remainder))
    }

    /// `(g, r)` with `p*S = D(g) + r` and `r` a reduced form (`reduce`).
    pub fn reduce(&mut self, p: &Basic) -> PolyResult<(Basic, Basic)> {
        let p_: Basic = cancel(&mul(&[p.clone(), self.shell.clone()])?)?;
        let (g1, current) = self.reduce_coprime_poles(&p_)?;
        let (g2, current) = self.reduce_b_poles(&current)?;
        let (pp, qq) = self.numerator_denominator(&current)?;
        let (polynomial, proper_numerator) = pp.div(&qq)?;
        let (g3, reduced) = self.reduce_polynomial(&polynomial)?;
        let k: &Context = &self.context;
        let proper: Basic = k.fraction_expr(&proper_numerator, &qq)?;
        Ok((
            cancel(&add(&[g1, g2, g3])?)?,
            cancel(&add(&[proper, k.poly_expr(&reduced)?])?)?,
        ))
    }
}

fn squarefree_part(k: &Context, p: &KPoly) -> PolyResult<KPoly> {
    let mut result: KPoly = k.constant(&k.field.one());
    for (factor, _) in p.sqf_list().1 {
        if factor.degree() > 0 {
            result = result.mul(&factor)?;
        }
    }
    if result.degree() > 0 {
        result.monic()
    } else {
        Ok(result)
    }
}

type Normalized = (Basic, KPoly, KPoly);

/// Move the factors `d**n` of `F` into the shell (`_normalize`).
fn normalize(k: &Context, a: &KPoly, b: &KPoly) -> PolyResult<Outcome<Normalized>> {
    let mut shell: Basic = Basic::one();
    let mut q: Basic = k.fraction_expr(a, b)?;
    for (factor, multiplicity) in b.factor_list()?.1 {
        if factor.degree() == 0 || multiplicity != 1 {
            continue;
        }
        let d: KPoly = factor.monic()?;
        let cofactor: KPoly = b.quo(&d)?;
        let inverse: KPoly = match cofactor.mul(&d.diff(1)).and_then(|c| c.invert(&d)) {
            Ok(i) => i,
            Err(e) if polynomial_failure(&e) => continue,
            Err(e) => return Err(e),
        };
        let residue: KPoly = a.mul(&inverse)?.rem(&d)?;
        if residue.degree() > 0 {
            if k.gaussian {
                continue;
            }
            return Ok(Outcome::Split);
        }
        let n: Option<i64> = if residue.is_zero() {
            Some(0)
        } else {
            integer(&k.element_expr(&residue.LC())?)?
        };
        let Some(n) = n.filter(|n| *n != 0) else {
            continue;
        };
        let d_expr: Basic = k.poly_expr(&d)?;
        shell = mul(&[shell, pow(&d_expr, &int(n))?])?;
        // q - n d'/d
        q = cancel(&sub(&q, &div(&mul(&[int(n), k.poly_expr(&d.diff(1))?])?, &d_expr)?)?)?;
    }
    let (a, b) = k.fraction(&q)?;
    Ok(Outcome::Done((shell, a, b)))
}

/// `(g, r)` with `p = g' + q*g + r` and `r` a reduced form
/// (`hermite_reduce`).
pub fn hermite_reduce(p: &Basic, q: &Basic, x: &Basic, t: Option<&Basic>) -> PolyResult<(Basic, Basic)> {
    let t_: Basic = t
        .cloned()
        .unwrap_or_else(|| crate::core::symbol::Symbol::new("t", &[]).0);
    let mut reducer: HermiteReducer = HermiteReducer::new(q, x, &t_, std::slice::from_ref(p))?;
    let (g, r) = reducer.reduce(p)?;
    let s: Basic = reducer.shell.clone();
    Ok((cancel(&div(&g, &s)?)?, cancel(&div(&r, &s)?)?))
}

/// A nontrivial `a` with `sum(a_j v_j) = 0` and `a[-1] != 0`
/// (`_dependency`).
fn dependency(vectors: &[Vec<Basic>]) -> PolyResult<Option<Vec<Basic>>> {
    let width: usize = vectors.iter().map(|v| v.len()).max().unwrap_or(0);
    let rows: Vec<Vec<Basic>> = (0..width)
        .map(|i| {
            vectors
                .iter()
                .map(|v| v.get(i).cloned().unwrap_or_else(Basic::zero))
                .collect()
        })
        .collect();
    let basis: Vec<Vec<Basic>> = nullspace(&rows, vectors.len())?.unwrap_or_default();
    Ok(basis
        .into_iter()
        .find(|values| *values.last().expect("nonempty") != Basic::zero()))
}

/// The reduced forms over a common (monic) denominator, as coefficient
/// lists (lowest degree first) (`_coordinates`).
fn coordinates(forms: &[Basic], x: &Basic) -> PolyResult<Option<Vec<Vec<Basic>>>> {
    let setup = (|| -> PolyResult<(Context, Vec<(KPoly, KPoly)>)> {
        let k: Context = Context::new(forms, x, &[], false)?;
        let fractions: Vec<(KPoly, KPoly)> = forms.iter().map(|r| k.fraction(r)).collect::<PolyResult<_>>()?;
        Ok((k, fractions))
    })();
    let (k, fractions) = match setup {
        Ok(v) => v,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut common: KPoly = fractions[0].1.monic()?;
    for (_, q) in &fractions[1..] {
        common = common.lcm(q)?;
    }
    let mut vectors: Vec<Vec<Basic>> = Vec::new();
    for (p, q) in &fractions {
        let numerator: KPoly = p.mul(&common.quo(q)?)?;
        vectors.push(
            numerator
                .rep
                .iter()
                .rev()
                .map(|c| k.element_expr(c))
                .collect::<PolyResult<_>>()?,
        );
    }
    Ok(Some(vectors))
}

/// A hyperexponential term with its reducer and the sequence `p_j`
/// (`_Term`).
struct Term {
    f: Basic,
    t: Basic,
    p: Basic,
    reducer: HermiteReducer,
    ps: Vec<Basic>,
    gs: Vec<Basic>,
    rs: Vec<Basic>,
}

impl Term {
    fn new(f: &Basic, x: &Basic, t: &Basic) -> PolyResult<Term> {
        let (Some(q), Some(p)) = (logarithmic_derivative(f, x)?, logarithmic_derivative(f, t)?) else {
            return Err(PolyError::Value(format!("not hyperexponential: {}", f)));
        };
        let reducer: HermiteReducer = HermiteReducer::new(&q, x, t, std::slice::from_ref(&p))?;
        Ok(Term {
            f: f.clone(),
            t: t.clone(),
            p,
            reducer,
            ps: Vec::new(),
            gs: Vec::new(),
            rs: Vec::new(),
        })
    }

    /// The next `p_j` and its reduction (`extend`).
    fn extend(&mut self) -> PolyResult<()> {
        let next: Basic = match self.ps.last() {
            None => Basic::one(),
            Some(last) => cancel(&add(&[diff(last, &self.t)?, mul(&[self.p.clone(), last.clone()])?])?)?,
        };
        self.ps.push(next.clone());
        let (g, r) = self.reducer.reduce(&next)?;
        self.gs.push(g);
        self.rs.push(r);
        Ok(())
    }

    /// `R` for the term alone (`certificate`).
    fn certificate(&self, relation: &[Basic]) -> R {
        let mut parts: Vec<Basic> = Vec::new();
        for (a, g) in relation.iter().zip(&self.gs) {
            parts.push(mul(&[a.clone(), g.clone()])?);
        }
        cancel(&div(&add(&parts)?, &self.reducer.shell)?)
    }
}

/// `u/v` when it is a rational function of `x` and `t` (`_similar`).
fn similar(u: &Basic, v: &Basic, x: &Basic, t: &Basic) -> PolyResult<Option<Basic>> {
    let ratio: Basic = cancel(&div(u, v)?)?;
    let xt: [Basic; 2] = [x.clone(), t.clone()];
    Ok(
        if crate::core::expr::polynomial_like(&ratio, &xt, crate::core::expr::PolynomialKind::Rational) == Some(true) {
            Some(ratio)
        } else {
            None
        },
    )
}

/// The terms with the similar ones added together (`_grouped`).
fn grouped(found: &[Basic], x: &Basic, t: &Basic) -> PolyResult<Vec<Basic>> {
    let mut groups: Vec<Basic> = Vec::new();
    for term in found {
        let mut placed: bool = false;
        for group in groups.iter_mut() {
            if let Some(ratio) = similar(term, group, x, t)? {
                *group = cancel(&mul(&[group.clone(), add(&[int(1), ratio])?])?)?;
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(term.clone());
        }
    }
    Ok(groups.into_iter().filter(|g| *g != Basic::zero()).collect())
}

type Telescoped = (Vec<Basic>, Basic);

/// The coefficients and the certificate of the minimal common telescoper
/// of non-similar hyperexponential terms (`_telescope`).
fn telescope(found: &[Basic], x: &Basic, t: &Basic, max_order: usize) -> PolyResult<Option<Telescoped>> {
    let mut objects: Vec<Term> = found
        .iter()
        .map(|term| Term::new(term, x, t))
        .collect::<PolyResult<_>>()?;
    let f: Basic = add(found)?;
    for j in 0..=max_order {
        let mut vectors: Vec<Vec<Basic>> = vec![Vec::new(); j + 1];
        for term in objects.iter_mut() {
            term.extend()?;
            let Some(coords) = coordinates(&term.rs, x)? else {
                return Ok(None);
            };
            for (i, v) in coords.into_iter().enumerate() {
                vectors[i].extend(v);
            }
        }
        let Some(relation) = dependency(&vectors)? else {
            continue;
        };
        let (normalized_values, factor) = normalized(&relation, t)?;
        let certificate: Basic = if objects.len() == 1 {
            objects[0].certificate(&relation)?
        } else {
            let mut parts: Vec<Basic> = Vec::new();
            for term in &objects {
                parts.push(mul(&[term.certificate(&relation)?, term.f.clone()])?);
            }
            cancel(&div(&add(&parts)?, &f)?)?
        };
        return Ok(Some((normalized_values, cancel(&mul(&[certificate, factor])?)?)));
    }
    Ok(None)
}

/// The minimal telescoper of a hyperexponential `F(x, t)`, or of a sum of
/// hyperexponential terms, by Hermite reduction (`reduction_telescoper`).
pub fn reduction_telescoper(
    f: &Basic,
    x: &Basic,
    t: &Basic,
    max_order: usize,
) -> PolyResult<Option<DifferentialTelescoper>> {
    let groups: Vec<Basic> = if is_hyperexponential(f, x, t)? {
        vec![f.clone()]
    } else {
        let found: Vec<Basic> = terms(f, x, t)?;
        if found.is_empty() {
            return Ok(None);
        }
        let groups: Vec<Basic> = grouped(&found, x, t)?;
        if groups.is_empty() {
            return Ok(None);
        }
        groups
    };
    let result: Option<Telescoped> = match telescope(&groups, x, t, max_order) {
        Ok(r) => r,
        Err(e) if polynomial_failure(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some((coefficients, mut certificate)) = result else {
        return Ok(None);
    };
    if groups.len() > 1 {
        certificate = cancel(&div(&mul(&[certificate, add(&groups)?])?, f)?)?;
    }
    Ok(Some(DifferentialTelescoper {
        term: f.clone(),
        x: x.clone(),
        t: t.clone(),
        coefficients,
        certificate,
    }))
}

/// The differential equation of `Integral(F, (x, a, b))` from the
/// reduction-based telescoper (`reduction_ode`).
pub fn reduction_ode(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    assumptions: &[Basic],
    name: &str,
) -> PolyResult<Option<Basic>> {
    let Some(telescoper) = reduction_telescoper(f, x, t, 6)? else {
        return Ok(None);
    };
    let g: Basic = mul(&[telescoper.certificate.clone(), f.clone()])?;
    equation(&telescoper.coefficients, &g, x, a, b, t, assumptions, name)
}

/// `Integral(F, (x, a, b))` as a function of `t` by the holonomic method
/// with the reduction-based telescoper (`reduction_integral`).
pub fn reduction_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    t: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    let Some(telescoper) = reduction_telescoper(f, x, t, 6)? else {
        return Ok(None);
    };
    let g: Basic = mul(&[telescoper.certificate.clone(), f.clone()])?;
    let Some(eq) = equation(&telescoper.coefficients, &g, x, a, b, t, assumptions, "I")? else {
        return Ok(None);
    };
    let Some(solution) = solve(&telescoper.coefficients, &eq, f, x, a, b, t, assumptions)? else {
        return Ok(None);
    };
    let value: Basic = crate::integrals::marichev::tidy(&solution, assumptions, &crate::core::logic::true_())?;
    let bad: bool = value.preorder().iter().any(|n| {
        matches!(
            n.head(),
            Head::NaN | Head::ComplexInfinity | Head::Infinity | Head::NegativeInfinity | Head::Piecewise
        )
    });
    if bad || !verified(&value, f, x, a, b, assumptions)? {
        return Ok(None);
    }
    Ok(Some(ConditionalValue::new(value, crate::core::logic::true_())))
}
