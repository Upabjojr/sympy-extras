//! Port of `sympy_extras/integrals/mellin.py`: Mellin transforms written as
//! quotients of gamma functions (the Marichev–Adamchik representation):
//! the `GammaQuotient` algebra (shift, scale, products, composition),
//! the table of kernels with their strips and conditions, the matching of
//! the factors of an integrand against the kernels, and the transform.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::assumptions::ask::ask;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_independent, is};
use crate::core::logic::{and_, boolean};
use crate::error::{value_error, PolyResult};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn rat(p: i64, q: i64) -> Basic {
    Basic::from_rational(Rat::from(Int::from(p)) / Rat::from(Int::from(q)))
}
fn half() -> Basic {
    rat(1, 2)
}
fn oo() -> Basic {
    Basic::oo()
}
fn moo() -> Basic {
    Basic::negative_oo()
}
fn pi() -> Basic {
    Basic::pi()
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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn sqrt(a: &Basic) -> R {
    pw(a, &half())
}
fn gamma(a: &Basic) -> R {
    crate::functions::special::gamma_functions::gamma(a)
}
fn abs_(a: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(a)
}
fn max2(a: &Basic, b: &Basic) -> R {
    crate::functions::elementary::miscellaneous::max_(&[a.clone(), b.clone()])
}
fn min2(a: &Basic, b: &Basic) -> R {
    crate::functions::elementary::miscellaneous::min_(&[a.clone(), b.clone()])
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn replaced(e: &Basic, old: &Basic, new: &Basic) -> R {
    let rule: HashMap<Basic, Basic> = [(old.clone(), new.clone())].into_iter().collect();
    crate::core::function::xreplace_evaluated(e, &rule)
}

/// The variable of the transforms (`_S`).
fn big_s() -> &'static Basic {
    static S: OnceLock<Basic> = OnceLock::new();
    S.get_or_init(|| crate::core::symbol::Dummy::new("s", &[]).0)
}

/// The scale of the argument of a kernel (`BETA`).
pub fn beta() -> &'static Basic {
    static B: OnceLock<Basic> = OnceLock::new();
    B.get_or_init(|| crate::core::symbol::Dummy::new("beta", &[]).0)
}

/// A gamma factor `Gamma(shift + scale*s)`, the scale rational.
pub type GammaFactor = (Basic, Basic);

fn factor(shift: &Basic, scale: &Basic) -> PolyResult<GammaFactor> {
    if !scale.is_rational_class() {
        return Err(value_error(format!(
            "the coefficient of s in a gamma factor must be rational, got {}",
            scale
        )));
    }
    Ok((shift.clone(), scale.clone()))
}

fn gamma_product(factors: &[GammaFactor], s: &Basic) -> R {
    let mut result: Basic = int(1);
    for (shift, scale) in factors {
        result = m2(&result, &gamma(&a2(shift, &m2(scale, s)?)?)?)?;
    }
    Ok(result)
}

/// `constant * extra(s) * prod base**(c*s) * prod Gamma(num) / prod
/// Gamma(den)` in the strip `lower < Re s < upper` (`GammaQuotient`).
#[derive(Debug, Clone)]
pub struct GammaQuotient {
    pub constant: Basic,
    pub bases: Vec<(Basic, Basic)>,
    pub numerator: Vec<GammaFactor>,
    pub denominator: Vec<GammaFactor>,
    pub lower: Basic,
    pub upper: Basic,
    pub condition: Basic,
    pub extra: Basic,
}

impl GammaQuotient {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        constant: Basic,
        bases: Vec<(Basic, Basic)>,
        numerator: Vec<(Basic, Basic)>,
        denominator: Vec<(Basic, Basic)>,
        lower: Basic,
        upper: Basic,
        condition: Basic,
        extra: Basic,
    ) -> PolyResult<GammaQuotient> {
        Ok(GammaQuotient {
            constant,
            bases: bases.into_iter().filter(|(_, c)| *c != int(0)).collect(),
            numerator: numerator.iter().map(|(a, b)| factor(a, b)).collect::<PolyResult<_>>()?,
            denominator: denominator
                .iter()
                .map(|(a, b)| factor(a, b))
                .collect::<PolyResult<_>>()?,
            lower,
            upper,
            condition,
            extra,
        })
    }

    fn simple(
        constant: Basic,
        bases: Vec<(Basic, Basic)>,
        numerator: Vec<(Basic, Basic)>,
        denominator: Vec<(Basic, Basic)>,
        lower: Basic,
        upper: Basic,
    ) -> PolyResult<GammaQuotient> {
        GammaQuotient::new(
            constant,
            bases,
            numerator,
            denominator,
            lower,
            upper,
            boolean(true),
            int(1),
        )
    }

    /// The strip of convergence (`strip`).
    pub fn strip(&self) -> (Basic, Basic) {
        (self.lower.clone(), self.upper.clone())
    }

    /// The quotient at `s` (`as_expr`).
    pub fn as_expr(&self, s: &Basic) -> R {
        let mut result: Basic = m2(&self.constant, &replaced(&self.extra, big_s(), s)?)?;
        for (base, coefficient) in &self.bases {
            result = m2(&result, &pw(base, &m2(coefficient, s)?)?)?;
        }
        d2(
            &m2(&result, &gamma_product(&self.numerator, s)?)?,
            &gamma_product(&self.denominator, s)?,
        )
    }

    /// The constant multiplied (`scaled`).
    pub fn scaled(&self, f: &Basic) -> PolyResult<GammaQuotient> {
        let mut q: GammaQuotient = self.clone();
        q.constant = m2(&self.constant, f)?;
        Ok(q)
    }

    /// The product of two quotients (`times`).
    pub fn times(&self, other: &GammaQuotient) -> PolyResult<GammaQuotient> {
        let mut bases: Vec<(Basic, Basic)> = self.bases.clone();
        bases.extend(other.bases.iter().cloned());
        let mut numerator: Vec<(Basic, Basic)> = self.numerator.clone();
        numerator.extend(other.numerator.iter().cloned());
        let mut denominator: Vec<(Basic, Basic)> = self.denominator.clone();
        denominator.extend(other.denominator.iter().cloned());
        GammaQuotient::new(
            m2(&self.constant, &other.constant)?,
            bases,
            numerator,
            denominator,
            max_of(&self.lower, &other.lower)?,
            min_of(&self.upper, &other.upper)?,
            and_(&[self.condition.clone(), other.condition.clone()]),
            m2(&self.extra, &other.extra)?,
        )
    }

    /// `s -> p + q*s` (`compose`), `q` a nonzero rational.
    pub fn compose(&self, p: &Basic, q: &Basic) -> PolyResult<GammaQuotient> {
        if !q.is_rational_class() || *q == int(0) {
            return Err(value_error(format!(
                "the scale must be a nonzero rational number, got {}",
                q
            )));
        }
        let mut constant: Basic = self.constant.clone();
        let mut bases: Vec<(Basic, Basic)> = Vec::new();
        for (base, coefficient) in &self.bases {
            constant = m2(&constant, &pw(base, &m2(coefficient, p)?)?)?;
            bases.push((base.clone(), m2(coefficient, q)?));
        }
        let map = |v: &[GammaFactor]| -> PolyResult<Vec<(Basic, Basic)>> {
            v.iter().map(|(a, b)| Ok((a2(a, &m2(b, p)?)?, m2(b, q)?))).collect()
        };
        let (numerator, denominator) = (map(&self.numerator)?, map(&self.denominator)?);
        let positive: bool = !crate::numbers::rat_is_negative(q.as_rational().expect("rational"));
        let (lower, upper) = if positive {
            (d2(&s2(&self.lower, p)?, q)?, d2(&s2(&self.upper, p)?, q)?)
        } else {
            (d2(&s2(&self.upper, p)?, q)?, d2(&s2(&self.lower, p)?, q)?)
        };
        let extra: Basic = replaced(&self.extra, big_s(), &a2(p, &m2(q, big_s())?)?)?;
        GammaQuotient::new(
            constant,
            bases,
            numerator,
            denominator,
            lower,
            upper,
            self.condition.clone(),
            extra,
        )
    }

    /// Equal factors of the numerator and the denominator removed
    /// (`cancelled`).
    pub fn cancelled(&self) -> PolyResult<GammaQuotient> {
        let mut numerator: Vec<GammaFactor> = self.numerator.clone();
        let mut denominator: Vec<GammaFactor> = Vec::new();
        for f in &self.denominator {
            if let Some(i) = numerator.iter().position(|g| g == f) {
                numerator.remove(i);
            } else {
                denominator.push(f.clone());
            }
        }
        let mut q: GammaQuotient = self.clone();
        q.numerator = numerator;
        q.denominator = denominator;
        Ok(q)
    }

    /// `lower < s < upper` (`strip_condition`).
    pub fn strip_condition(&self, s: &Basic) -> R {
        let mut parts: Vec<Basic> = Vec::new();
        if self.lower != moo() {
            parts.push(crate::core::relational::gt(s, &self.lower)?);
        }
        if self.upper != oo() {
            parts.push(crate::core::relational::lt(s, &self.upper)?);
        }
        Ok(and_(&parts))
    }

    /// Whether there is no extra factor (`is_gamma_quotient`).
    pub fn is_gamma_quotient(&self) -> bool {
        self.extra == int(1)
    }

    pub fn repr(&self) -> PolyResult<String> {
        let s: Basic = crate::core::symbol::Symbol::new("s", &[]).0;
        Ok(format!(
            "GammaQuotient({}, ({}, {}))",
            self.as_expr(&s)?,
            self.lower,
            self.upper
        ))
    }
}

/// Whether the poles of the numerator lie on the proper sides of the strip
/// (`poles_separated`).
pub fn poles_separated(q: &GammaQuotient) -> PolyResult<Option<bool>> {
    let mut verdict: Option<bool> = Some(true);
    for (shift, scale) in &q.numerator {
        if *scale == int(0) {
            continue;
        }
        let pole: Basic = d2(&neg(shift)?, scale)?;
        let positive: bool = !crate::numbers::rat_is_negative(scale.as_rational().expect("rational"));
        let ok: Option<bool> = if positive {
            compare(&pole, &q.lower)?
        } else {
            compare(&q.upper, &pole)?
        };
        if ok == Some(false) {
            return Ok(Some(false));
        }
        if ok.is_none() {
            verdict = None;
        }
    }
    Ok(verdict)
}

fn compare(a: &Basic, b: &Basic) -> PolyResult<Option<bool>> {
    if *b == oo() || *a == moo() {
        return Ok(Some(true));
    }
    if *b == moo() || *a == oo() {
        return Ok(Some(false));
    }
    let d: Basic = s2(b, a)?;
    if is(&d, Predicate::nonnegative) == Some(true) {
        return Ok(Some(true));
    }
    if is(&d, Predicate::negative) == Some(true) {
        return Ok(Some(false));
    }
    Ok(None)
}

fn max_of(a: &Basic, b: &Basic) -> R {
    if *a == moo() {
        return Ok(b.clone());
    }
    if *b == moo() {
        return Ok(a.clone());
    }
    max2(a, b)
}

fn min_of(a: &Basic, b: &Basic) -> R {
    if *a == oo() {
        return Ok(b.clone());
    }
    if *b == oo() {
        return Ok(a.clone());
    }
    min2(a, b)
}

/// A transform with its strip and condition (`MellinTransform`).
#[derive(Debug, Clone)]
pub struct MellinTransform {
    pub transform: Basic,
    pub strip: (Basic, Basic),
    pub condition: Basic,
}

impl MellinTransform {
    pub fn repr(&self) -> String {
        if matches!(self.condition.head(), Head::BooleanTrue) {
            format!(
                "MellinTransform({}, ({}, {}))",
                self.transform, self.strip.0, self.strip.1
            )
        } else {
            format!(
                "MellinTransform({}, ({}, {}), {})",
                self.transform, self.strip.0, self.strip.1, self.condition
            )
        }
    }
}

/// A kernel: a function of `beta*x` whose transform is a gamma quotient
/// (`Kernel`).
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub quotient: GammaQuotient,
    pub parameters: Vec<Basic>,
    pub scale_condition: Basic,
    pub parity: i32,
}

fn kernel(name: &str, quotient: GammaQuotient, parameters: Vec<Basic>, scale_condition: Basic, parity: i32) -> Kernel {
    Kernel {
        name: name.to_string(),
        quotient,
        parameters,
        scale_condition,
        parity,
    }
}

fn positive_real(b: &Basic) -> R {
    crate::core::relational::gt(b, &int(0))
}
fn right_half_plane(b: &Basic) -> R {
    crate::core::relational::lt(
        &abs_(&crate::functions::elementary::argument::arg(b)?)?,
        &d2(&pi(), &int(2))?,
    )
}
fn cut_plane(b: &Basic) -> R {
    crate::core::relational::lt(&abs_(&crate::functions::elementary::argument::arg(b)?)?, &pi())
}

fn p(a: Basic, b: Basic) -> (Basic, Basic) {
    (a, b)
}

const BETA_LOWER: &str = "(1 - x)**(b - 1) theta(1 - x)";

fn exp_kernel() -> PolyResult<Kernel> {
    Ok(kernel(
        "exp",
        GammaQuotient::simple(int(1), vec![], vec![p(int(0), int(1))], vec![], int(0), oo())?,
        vec![],
        right_half_plane(beta())?,
        0,
    ))
}
fn exp_minus_one_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        int(-1),
        vec![],
        vec![p(int(1), int(1)), p(int(0), int(-1))],
        vec![p(int(1), int(-1))],
        int(-1),
        int(0),
    )?;
    Ok(kernel("exp - 1", q, vec![], right_half_plane(beta())?, 0))
}
fn power_kernel(a: &Basic) -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        int(1),
        vec![],
        vec![p(int(0), int(1)), p(a.clone(), int(-1))],
        vec![p(a.clone(), int(0))],
        int(0),
        a.clone(),
    )?;
    Ok(kernel("(1 + x)**(-a)", q, vec![a.clone()], cut_plane(beta())?, 0))
}
fn beta_kernel(b: &Basic) -> PolyResult<Kernel> {
    let q = GammaQuotient::new(
        int(1),
        vec![],
        vec![p(int(0), int(1)), p(b.clone(), int(0))],
        vec![p(b.clone(), int(1))],
        int(0),
        oo(),
        crate::core::relational::gt(b, &int(0))?,
        int(1),
    )?;
    Ok(kernel(BETA_LOWER, q, vec![b.clone()], positive_real(beta())?, 0))
}
fn beta_upper_kernel(b: &Basic) -> PolyResult<Kernel> {
    let one_b: Basic = s2(&int(1), b)?;
    let q = GammaQuotient::new(
        int(1),
        vec![],
        vec![p(b.clone(), int(0)), p(one_b.clone(), int(-1))],
        vec![p(int(1), int(-1))],
        moo(),
        one_b,
        crate::core::relational::gt(b, &int(0))?,
        int(1),
    )?;
    Ok(kernel(
        "(x - 1)**(b - 1) theta(x - 1)",
        q,
        vec![b.clone()],
        positive_real(beta())?,
        0,
    ))
}
fn log1p_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        int(1),
        vec![],
        vec![p(int(1), int(1)), p(int(0), int(-1)), p(int(0), int(-1))],
        vec![p(int(1), int(-1))],
        int(-1),
        int(0),
    )?;
    Ok(kernel("log(1 + x)", q, vec![], cut_plane(beta())?, 0))
}
fn sin_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&sqrt(&pi())?, &int(2))?,
        vec![p(int(2), int(1))],
        vec![p(half(), half())],
        vec![p(int(1), neg(&half())?)],
        int(-1),
        int(1),
    )?;
    Ok(kernel("sin", q, vec![], positive_real(beta())?, -1))
}
fn sin_minus_x_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&neg(&sqrt(&pi())?)?, &int(2))?,
        vec![p(int(2), int(1))],
        vec![p(rat(3, 2), half()), p(neg(&half())?, neg(&half())?)],
        vec![p(int(1), neg(&half())?), p(half(), neg(&half())?)],
        int(-3),
        int(-1),
    )?;
    Ok(kernel("sin - x", q, vec![], positive_real(beta())?, -1))
}
fn cos_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&sqrt(&pi())?, &int(2))?,
        vec![p(int(2), int(1))],
        vec![p(int(0), half())],
        vec![p(half(), neg(&half())?)],
        int(0),
        int(1),
    )?;
    Ok(kernel("cos", q, vec![], positive_real(beta())?, 1))
}
fn cos_minus_one_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&neg(&sqrt(&pi())?)?, &int(2))?,
        vec![p(int(2), int(1))],
        vec![p(int(1), half()), p(int(0), neg(&half())?)],
        vec![p(int(1), neg(&half())?), p(half(), neg(&half())?)],
        int(-2),
        int(0),
    )?;
    Ok(kernel("cos - 1", q, vec![], positive_real(beta())?, 1))
}
fn atan_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        half(),
        vec![],
        vec![p(int(0), int(-1)), p(half(), half()), p(half(), neg(&half())?)],
        vec![p(int(1), int(-1))],
        int(-1),
        int(0),
    )?;
    Ok(kernel("atan", q, vec![], positive_real(beta())?, -1))
}
fn atan_minus_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        neg(&half())?,
        vec![],
        vec![p(int(0), int(1)), p(half(), half()), p(half(), neg(&half())?)],
        vec![p(int(1), int(1))],
        int(0),
        int(1),
    )?;
    Ok(kernel("atan - pi/2", q, vec![], positive_real(beta())?, 0))
}
fn erfc_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&int(1), &sqrt(&pi())?)?,
        vec![],
        vec![p(half(), half()), p(int(0), int(1))],
        vec![p(int(1), int(1))],
        int(0),
        oo(),
    )?;
    Ok(kernel("erfc", q, vec![], positive_real(beta())?, 0))
}
fn erf_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&int(1), &sqrt(&pi())?)?,
        vec![],
        vec![p(half(), half()), p(int(0), int(-1))],
        vec![p(int(1), int(-1))],
        int(-1),
        int(0),
    )?;
    Ok(kernel("erf", q, vec![], positive_real(beta())?, -1))
}
fn e1_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        int(1),
        vec![],
        vec![p(int(0), int(1)), p(int(0), int(1))],
        vec![p(int(1), int(1))],
        int(0),
        oo(),
    )?;
    Ok(kernel("E1", q, vec![], right_half_plane(beta())?, 0))
}
fn expint_kernel(n: &Basic) -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        int(1),
        vec![],
        vec![p(int(0), int(1)), p(s2(n, &int(1))?, int(1))],
        vec![p(n.clone(), int(1))],
        max2(&int(0), &s2(&int(1), n)?)?,
        oo(),
    )?;
    Ok(kernel("expint", q, vec![n.clone()], right_half_plane(beta())?, 0))
}
fn hyper1f1_kernel(a: &Basic, b: &Basic) -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&gamma(b)?, &gamma(a)?)?,
        vec![],
        vec![p(int(0), int(1)), p(a.clone(), int(-1))],
        vec![p(b.clone(), int(-1))],
        int(0),
        a.clone(),
    )?;
    Ok(kernel(
        "hyper1f1",
        q,
        vec![a.clone(), b.clone()],
        right_half_plane(beta())?,
        0,
    ))
}
fn besselj_kernel(nu: &Basic) -> PolyResult<Kernel> {
    let nu2: Basic = d2(nu, &int(2))?;
    let q = GammaQuotient::simple(
        half(),
        vec![p(int(2), int(1))],
        vec![p(nu2.clone(), half())],
        vec![p(a2(&int(1), &nu2)?, neg(&half())?)],
        neg(nu)?,
        rat(3, 2),
    )?;
    Ok(kernel("besselj", q, vec![nu.clone()], positive_real(beta())?, 0))
}
fn bessely_kernel(nu: &Basic) -> PolyResult<Kernel> {
    let nu2: Basic = d2(nu, &int(2))?;
    let q = GammaQuotient::simple(
        neg(&half())?,
        vec![p(int(2), int(1))],
        vec![p(nu2.clone(), half()), p(neg(&nu2)?, half())],
        vec![p(s2(&half(), &nu2)?, half()), p(a2(&half(), &nu2)?, neg(&half())?)],
        abs_(nu)?,
        rat(3, 2),
    )?;
    Ok(kernel("bessely", q, vec![nu.clone()], positive_real(beta())?, 0))
}
fn besselk_kernel(nu: &Basic) -> PolyResult<Kernel> {
    let nu2: Basic = d2(nu, &int(2))?;
    let q = GammaQuotient::simple(
        rat(1, 4),
        vec![p(int(2), int(1))],
        vec![p(neg(&nu2)?, half()), p(nu2.clone(), half())],
        vec![],
        abs_(nu)?,
        oo(),
    )?;
    Ok(kernel("besselk", q, vec![nu.clone()], right_half_plane(beta())?, 0))
}
fn exp_besseli_kernel(nu: &Basic) -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&int(1), &sqrt(&pi())?)?,
        vec![p(half(), int(1))],
        vec![p(nu.clone(), int(1)), p(half(), int(-1))],
        vec![p(a2(&int(1), nu)?, int(-1))],
        neg(nu)?,
        half(),
    )?;
    Ok(kernel(
        "exp(-x)*besseli",
        q,
        vec![nu.clone()],
        positive_real(beta())?,
        0,
    ))
}
fn airyai_kernel() -> PolyResult<Kernel> {
    let constant: Basic = d2(&pw(&int(3), &rat(-7, 6))?, &m2(&int(2), &pi())?)?;
    let q = GammaQuotient::simple(
        constant,
        vec![p(int(3), rat(2, 3))],
        vec![p(int(0), rat(1, 3)), p(rat(1, 3), rat(1, 3))],
        vec![],
        int(0),
        oo(),
    )?;
    Ok(kernel("airyai", q, vec![], positive_real(beta())?, 0))
}
fn polylog_kernel(n: &Basic) -> PolyResult<Kernel> {
    let count: i64 = i64::try_from(&crate::numbers::numer(n.as_rational().expect("an integer"))).expect("small");
    let mut numerator: Vec<(Basic, Basic)> = vec![p(int(1), int(1))];
    for _ in 0..count + 1 {
        numerator.push(p(int(0), int(-1)));
    }
    let denominator: Vec<(Basic, Basic)> = (0..count).map(|_| p(int(1), int(-1))).collect();
    let q = GammaQuotient::simple(int(-1), vec![], numerator, denominator, int(-1), int(0))?;
    Ok(kernel("polylog(n, -x)", q, vec![n.clone()], cut_plane(beta())?, 0))
}
fn fresnel_kernel(name: &str, first: Basic, second: Basic) -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        sqrt(&m2(&int(2), &pi())?)?,
        vec![p(d2(&pi(), &int(2))?, neg(&half())?)],
        vec![p(rat(3, 2), half()), p(int(0), int(-1)), p(int(1), int(1))],
        vec![
            p(first, rat(1, 4)),
            p(second, rat(-1, 4)),
            p(int(1), int(-1)),
            p(int(2), int(1)),
        ],
        int(-1),
        int(0),
    )?;
    Ok(kernel(name, q, vec![], positive_real(beta())?, -1))
}
fn bessel_product_kernel(kind: &str, mu: &Basic, nu: &Basic) -> PolyResult<Kernel> {
    let h: Basic = half();
    let mh: Basic = neg(&h)?;
    let sum2: Basic = d2(&a2(mu, nu)?, &int(2))?;
    let diff2: Basic = d2(&s2(mu, nu)?, &int(2))?;
    let rdiff2: Basic = d2(&s2(nu, mu)?, &int(2))?;
    let one = |e: &Basic| a2(&int(1), e);
    match kind {
        "JJ" => {
            let q = GammaQuotient::simple(
                h.clone(),
                vec![p(int(2), int(1))],
                vec![p(int(1), int(-1)), p(sum2.clone(), h.clone())],
                vec![
                    p(one(&diff2)?, mh.clone()),
                    p(one(&rdiff2)?, mh.clone()),
                    p(one(&sum2)?, mh.clone()),
                ],
                neg(&a2(mu, nu)?)?,
                int(1),
            )?;
            Ok(kernel(
                "besselj*besselj",
                q,
                vec![mu.clone(), nu.clone()],
                positive_real(beta())?,
                0,
            ))
        }
        "KK" => {
            let q = GammaQuotient::simple(
                rat(1, 8),
                vec![p(int(2), int(1))],
                vec![
                    p(neg(&sum2)?, h.clone()),
                    p(diff2.clone(), h.clone()),
                    p(rdiff2.clone(), h.clone()),
                    p(sum2.clone(), h.clone()),
                ],
                vec![p(int(0), int(1))],
                max_of(&abs_(&s2(mu, nu)?)?, &abs_(&a2(mu, nu)?)?)?,
                oo(),
            )?;
            Ok(kernel(
                "besselk*besselk",
                q,
                vec![mu.clone(), nu.clone()],
                right_half_plane(beta())?,
                0,
            ))
        }
        "IK" => {
            let q = GammaQuotient::simple(
                rat(1, 4),
                vec![p(int(2), int(1))],
                vec![
                    p(int(1), int(-1)),
                    p(diff2.clone(), h.clone()),
                    p(sum2.clone(), h.clone()),
                ],
                vec![p(one(&diff2)?, mh.clone()), p(one(&sum2)?, mh.clone())],
                max_of(&s2(nu, mu)?, &neg(&a2(mu, nu)?)?)?,
                int(1),
            )?;
            Ok(kernel(
                "besseli*besselk",
                q,
                vec![mu.clone(), nu.clone()],
                positive_real(beta())?,
                0,
            ))
        }
        "JY" => {
            let q = GammaQuotient::simple(
                d2(&int(-1), &m2(&int(2), &sqrt(&pi())?)?)?,
                vec![],
                vec![p(nu.clone(), h.clone()), p(int(0), h.clone())],
                vec![p(one(nu)?, mh.clone()), p(h.clone(), h.clone())],
                max_of(&int(0), &m2(&int(-2), nu)?)?,
                int(2),
            )?;
            Ok(kernel(
                "besselj*bessely",
                q,
                vec![nu.clone()],
                positive_real(beta())?,
                0,
            ))
        }
        _ => {
            let nu2: Basic = d2(nu, &int(2))?;
            let q = GammaQuotient::simple(
                rat(1, 8),
                vec![p(int(2), int(1))],
                vec![p(nu2.clone(), rat(1, 4)), p(int(0), h.clone())],
                vec![p(one(&nu2)?, rat(-1, 4))],
                max_of(&int(0), &m2(&int(-2), nu)?)?,
                oo(),
            )?;
            Ok(kernel(
                "besselj*besselk",
                q,
                vec![nu.clone()],
                positive_real(beta())?,
                0,
            ))
        }
    }
}
fn erfc_exp_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&int(1), &sqrt(&pi())?)?,
        vec![p(half(), int(1))],
        vec![p(int(0), int(1)), p(half(), neg(&half())?)],
        vec![],
        int(0),
        int(1),
    )?;
    Ok(kernel("erfc(x)*exp(x**2)", q, vec![], positive_real(beta())?, 0))
}
fn si_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        neg(&pi())?,
        vec![],
        vec![p(int(1), int(1)), p(int(0), int(-1)), p(int(0), int(-1))],
        vec![
            p(int(0), half()),
            p(int(1), neg(&half())?),
            p(int(1), int(-1)),
            p(int(1), int(-1)),
        ],
        int(-1),
        int(0),
    )?;
    Ok(kernel("Si", q, vec![], positive_real(beta())?, -1))
}
fn si_minus_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&neg(&sqrt(&pi())?)?, &int(2))?,
        vec![p(int(2), int(1))],
        vec![p(half(), half()), p(int(0), int(1))],
        vec![p(int(1), neg(&half())?), p(int(1), int(1))],
        int(0),
        int(1),
    )?;
    Ok(kernel("Si - pi/2", q, vec![], positive_real(beta())?, 0))
}
fn ci_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        d2(&neg(&sqrt(&pi())?)?, &int(2))?,
        vec![p(int(2), int(1))],
        vec![p(int(0), half()), p(int(0), int(1))],
        vec![p(half(), neg(&half())?), p(int(1), int(1))],
        int(0),
        int(1),
    )?;
    Ok(kernel("Ci", q, vec![], positive_real(beta())?, 0))
}
fn theta_lower_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        int(1),
        vec![],
        vec![p(int(0), int(1))],
        vec![p(int(1), int(1))],
        int(0),
        oo(),
    )?;
    Ok(kernel("theta(1 - x)", q, vec![], positive_real(beta())?, 0))
}
fn theta_upper_kernel() -> PolyResult<Kernel> {
    let q = GammaQuotient::simple(
        int(1),
        vec![],
        vec![p(int(0), int(-1))],
        vec![p(int(1), int(-1))],
        moo(),
        int(0),
    )?;
    Ok(kernel("theta(x - 1)", q, vec![], positive_real(beta())?, 0))
}
fn log_power_lower_kernel(k: &Basic) -> PolyResult<Kernel> {
    let extra: Basic = pw(big_s(), &s2(&neg(k)?, &int(1))?)?;
    let q = GammaQuotient::new(
        int(1),
        vec![],
        vec![p(a2(k, &int(1))?, int(0))],
        vec![],
        int(0),
        oo(),
        crate::core::relational::gt(k, &int(-1))?,
        extra,
    )?;
    Ok(kernel(
        "(-log(x))**k theta(1 - x)",
        q,
        vec![k.clone()],
        positive_real(beta())?,
        0,
    ))
}
fn log_one_minus_kernel() -> PolyResult<Kernel> {
    let pg: Basic = crate::functions::special::gamma_functions::polygamma(&int(0), &a2(big_s(), &int(1))?)?;
    let extra: Basic = d2(&a2(&pg, &Basic::atom(Head::EulerGamma))?, big_s())?;
    let q = GammaQuotient::new(int(-1), vec![], vec![], vec![], int(0), oo(), boolean(true), extra)?;
    Ok(kernel("log(1 - x) theta(1 - x)", q, vec![], positive_real(beta())?, 0))
}
fn bose_kernel() -> PolyResult<Kernel> {
    let extra: Basic = crate::functions::special::zeta_functions::zeta(big_s(), None)?;
    let q = GammaQuotient::new(
        int(1),
        vec![],
        vec![p(int(0), int(1))],
        vec![],
        int(1),
        oo(),
        boolean(true),
        extra,
    )?;
    Ok(kernel("1/(exp(x) - 1)", q, vec![], positive_real(beta())?, 0))
}
fn fermi_kernel() -> PolyResult<Kernel> {
    let extra: Basic = crate::functions::special::zeta_functions::dirichlet_eta(big_s(), None)?;
    let q = GammaQuotient::new(
        int(1),
        vec![],
        vec![p(int(0), int(1))],
        vec![],
        int(0),
        oo(),
        boolean(true),
        extra,
    )?;
    Ok(kernel("1/(exp(x) + 1)", q, vec![], positive_real(beta())?, 0))
}
fn csch_kernel() -> PolyResult<Kernel> {
    let factor_: Basic = s2(&int(1), &pw(&int(2), &neg(big_s())?)?)?;
    let extra: Basic = m2(
        &factor_,
        &crate::functions::special::zeta_functions::zeta(big_s(), None)?,
    )?;
    let q = GammaQuotient::new(
        int(2),
        vec![],
        vec![p(int(0), int(1))],
        vec![],
        int(1),
        oo(),
        boolean(true),
        extra,
    )?;
    Ok(kernel("1/sinh(x)", q, vec![], positive_real(beta())?, -1))
}
fn sech_kernel() -> PolyResult<Kernel> {
    let extra: Basic = m2(
        &pw(&int(2), &neg(big_s())?)?,
        &crate::functions::special::zeta_functions::lerchphi(&int(-1), big_s(), &half())?,
    )?;
    let q = GammaQuotient::new(
        int(2),
        vec![],
        vec![p(int(0), int(1))],
        vec![],
        int(0),
        oo(),
        boolean(true),
        extra,
    )?;
    Ok(kernel("1/cosh(x)", q, vec![], positive_real(beta())?, 1))
}

/// The kernel of a name in the table (`KERNELS[name]`).
pub fn named_kernel(name: &str) -> PolyResult<Kernel> {
    match name {
        "exp" => exp_kernel(),
        "exp - 1" => exp_minus_one_kernel(),
        "log(1 + x)" => log1p_kernel(),
        "sin" => sin_kernel(),
        "sin - x" => sin_minus_x_kernel(),
        "cos" => cos_kernel(),
        "cos - 1" => cos_minus_one_kernel(),
        "atan" => atan_kernel(),
        "atan - pi/2" => atan_minus_kernel(),
        "erfc" => erfc_kernel(),
        "erf" => erf_kernel(),
        "E1" => e1_kernel(),
        "Si" => si_kernel(),
        "Si - pi/2" => si_minus_kernel(),
        "Ci" => ci_kernel(),
        "theta(1 - x)" => theta_lower_kernel(),
        "theta(x - 1)" => theta_upper_kernel(),
        "1/(exp(x) - 1)" => bose_kernel(),
        "1/(exp(x) + 1)" => fermi_kernel(),
        "1/sinh(x)" => csch_kernel(),
        "1/cosh(x)" => sech_kernel(),
        "airyai" => airyai_kernel(),
        "fresnels" => fresnel_kernel("fresnels", rat(1, 4), rat(3, 4)),
        "fresnelc" => fresnel_kernel("fresnelc", rat(3, 4), rat(1, 4)),
        "erfc(x)*exp(x**2)" => erfc_exp_kernel(),
        other => Err(value_error(format!("no kernel {}", other))),
    }
}

/// A factor matched by a kernel `f(beta*x**gamma)` (`Match`).
#[derive(Debug, Clone)]
pub struct Match {
    pub kernel: Kernel,
    pub beta: Basic,
    pub gamma: Basic,
    pub constant: Basic,
    pub cutoff: bool,
}

fn matched(kernel: Kernel, beta: Basic, gamma_: Basic) -> Match {
    Match {
        kernel,
        beta,
        gamma: gamma_,
        constant: int(1),
        cutoff: false,
    }
}

impl Match {
    pub fn repr(&self) -> String {
        format!(
            "Match({}, beta={}, gamma={}, constant={})",
            self.kernel.name, self.beta, self.gamma, self.constant
        )
    }

    /// The transform of the matched factor (`quotient`).
    pub fn quotient(&self, assumptions: &[Basic]) -> PolyResult<GammaQuotient> {
        let g: &Basic = &self.gamma;
        if !g.is_rational_class() || *g == int(0) {
            return Err(value_error(format!(
                "the power of x must be a nonzero rational number, got {}",
                g
            )));
        }
        let mut beta_: Basic = self.beta.clone();
        let mut constant: Basic = self.constant.clone();
        let mut condition: Basic = replaced(&self.kernel.scale_condition, beta(), &beta_)?;
        if self.kernel.parity != 0
            && ask(&crate::core::relational::gt(&beta_, &int(0))?, assumptions, None)? != Some(true)
            && (is(&beta_, Predicate::extended_real) == Some(true)
                || ask(
                    &crate::assumptions::facts::element(&beta_, &crate::sets::sets::reals())?,
                    assumptions,
                    None,
                )? == Some(true))
        {
            if self.kernel.parity < 0 {
                constant = d2(&m2(&constant, &beta_)?, &abs_(&beta_)?)?;
            }
            beta_ = abs_(&beta_)?;
            condition = boolean(true);
        }
        let inverse: Basic = d2(&int(1), g)?;
        let q: GammaQuotient = self.kernel.quotient.compose(&int(0), &inverse)?;
        let mut bases: Vec<(Basic, Basic)> = q.bases.clone();
        bases.push((beta_, neg(&inverse)?));
        GammaQuotient::new(
            d2(&m2(&q.constant, &constant)?, &abs_(g)?)?,
            bases,
            q.numerator.clone(),
            q.denominator.clone(),
            q.lower.clone(),
            q.upper.clone(),
            and_(&[q.condition.clone(), condition]),
            q.extra.clone(),
        )
    }
}

/// `(coefficient, exponent)` of `c*x**e`, `e` a nonzero rational
/// (`monomial`).
pub fn monomial(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let mut e: Basic = e.clone();
    if e.is_add() {
        e = crate::core::exprtools::factor_terms(&e, false, false, false, true)?;
        if e.is_add() {
            return Ok(None);
        }
    }
    let (coefficient, rest) = as_independent(&e, x, false)?;
    if rest == *x {
        return Ok(Some((coefficient, int(1))));
    }
    if rest.is_pow() && rest.args()[0] == *x {
        let exponent: &Basic = &rest.args()[1];
        if exponent.is_rational_class() && *exponent != int(0) {
            return Ok(Some((coefficient, exponent.clone())));
        }
    }
    Ok(None)
}

fn positive_monomial(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some((b, g)) = monomial(e, x)? else { return Ok(None) };
    if is(&b, Predicate::negative) == Some(true) || b == int(0) {
        return Ok(None);
    }
    Ok(Some((b, g)))
}

fn negated_monomial(e: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some((b, g)) = monomial(&neg(e)?, x)? else {
        return Ok(None);
    };
    if is(&b, Predicate::negative) == Some(true) || b == int(0) {
        return Ok(None);
    }
    Ok(Some((b, g)))
}

const ONE_ARGUMENT: &[(Function, &str)] = &[
    (Function::Sin, "sin"),
    (Function::Cos, "cos"),
    (Function::Atan, "atan"),
    (Function::Erf, "erf"),
    (Function::Erfc, "erfc"),
    (Function::Si, "Si"),
    (Function::Ci, "Ci"),
    (Function::AiryAi, "airyai"),
    (Function::FresnelS, "fresnels"),
    (Function::FresnelC, "fresnelc"),
];

fn match_function(f: &Basic, x: &Basic, cutoff: Option<&str>) -> PolyResult<Option<Match>> {
    if matches!(f.head(), Head::Hyper) {
        let (ap, bq) = (
            crate::functions::special::hyper::hyper_ap(f),
            crate::functions::special::hyper::hyper_bq(f),
        );
        if ap.len() == 1 && bq.len() == 1 {
            let found = negated_monomial(&f.args()[2], x)?;
            let (a, b) = (&ap[0], &bq[0]);
            if let Some((beta_, g)) = found {
                if !a.has(x) && !b.has(x) {
                    return Ok(Some(matched(hyper1f1_kernel(a, b)?, beta_, g)));
                }
            }
        }
        return Ok(None);
    }
    let Some(fun) = func(f) else { return Ok(None) };
    let args: &[Basic] = f.args();
    if fun == Function::Exp && args.len() == 1 {
        return Ok(match negated_monomial(&args[0], x)? {
            Some((b, g)) => Some(matched(exp_kernel()?, b, g)),
            None => None,
        });
    }
    for (g, name) in ONE_ARGUMENT {
        if fun == *g && args.len() == 1 {
            return Ok(match positive_monomial(&args[0], x)? {
                Some((b, gm)) => Some(matched(named_kernel(name)?, b, gm)),
                None => None,
            });
        }
    }
    if fun == Function::Log && args.len() == 1 {
        let inner: &Basic = &args[0];
        if inner.is_add() {
            let (constant, rest) = as_independent(inner, x, true)?;
            if constant == int(1) {
                if let Some((b, g)) = positive_monomial(&rest, x)? {
                    return Ok(Some(matched(log1p_kernel()?, b, g)));
                }
                if let Some((b, g)) = negated_monomial(&rest, x)? {
                    if cutoff == Some("lower") && b == int(1) && is(&g, Predicate::positive) == Some(true) {
                        return Ok(Some(Match {
                            kernel: log_one_minus_kernel()?,
                            beta: int(1),
                            gamma: g,
                            constant: int(1),
                            cutoff: true,
                        }));
                    }
                }
            }
        }
        return Ok(None);
    }
    if fun == Function::Ei && args.len() == 1 {
        return Ok(match negated_monomial(&args[0], x)? {
            Some((b, g)) => Some(Match {
                kernel: e1_kernel()?,
                beta: b,
                gamma: g,
                constant: int(-1),
                cutoff: false,
            }),
            None => None,
        });
    }
    if fun == Function::Expint && args.len() == 2 {
        if let Some((b, g)) = positive_monomial(&args[1], x)? {
            if !args[0].has(x) {
                return Ok(Some(matched(expint_kernel(&args[0])?, b, g)));
            }
        }
        return Ok(None);
    }
    if fun == Function::PolyLog && args.len() == 2 {
        let n: &Basic = &args[0];
        if let Some((b, g)) = negated_monomial(&args[1], x)? {
            if n.is_integer_value() && is(n, Predicate::positive) == Some(true) {
                return Ok(Some(matched(polylog_kernel(n)?, b, g)));
            }
        }
        return Ok(None);
    }
    if matches!(fun, Function::BesselJ | Function::BesselY | Function::BesselK) && args.len() == 2 && !args[0].has(x) {
        let Some((b, g)) = positive_monomial(&args[1], x)? else {
            return Ok(None);
        };
        let nu: &Basic = &args[0];
        let k: Kernel = match fun {
            Function::BesselJ => besselj_kernel(nu)?,
            Function::BesselY => bessely_kernel(nu)?,
            _ => besselk_kernel(nu)?,
        };
        return Ok(Some(matched(k, b, g)));
    }
    if fun == Function::Heaviside && !args.is_empty() {
        let inner: &Basic = &args[0];
        if inner.is_add() {
            let (constant, rest) = as_independent(inner, x, true)?;
            if is(&constant, Predicate::positive) == Some(true) {
                if let Some((b, g)) = negated_monomial(&rest, x)? {
                    return Ok(Some(matched(theta_lower_kernel()?, d2(&b, &constant)?, g)));
                }
                if let Some((b, g)) = positive_monomial(&rest, x)? {
                    if is(&constant, Predicate::negative) == Some(true) {
                        return Ok(Some(matched(theta_upper_kernel()?, d2(&b, &neg(&constant)?)?, g)));
                    }
                }
            }
            if is(&constant, Predicate::negative) == Some(true) {
                if let Some((b, g)) = positive_monomial(&rest, x)? {
                    return Ok(Some(matched(theta_upper_kernel()?, d2(&b, &neg(&constant)?)?, g)));
                }
            }
        }
        return Ok(None);
    }
    Ok(None)
}

fn match_power(f: &Basic, x: &Basic, cutoff: Option<&str>) -> PolyResult<Option<Match>> {
    let (base, exponent) = (&f.args()[0], &f.args()[1]);
    if exponent.has(x) {
        return Ok(None);
    }
    if func(base) == Some(Function::Exp) && base.args().len() == 1 {
        return Ok(match negated_monomial(&m2(&base.args()[0], exponent)?, x)? {
            Some((b, g)) => Some(matched(exp_kernel()?, b, g)),
            None => None,
        });
    }
    if *exponent == int(-1) {
        if base.is_add() && base.args().len() == 2 {
            let terms: &[Basic] = base.args();
            for (constant_term, other) in [(&terms[0], &terms[1]), (&terms[1], &terms[0])] {
                if !constant_term.has(x) && func(other) == Some(Function::Exp) {
                    if let Some((b, g)) = positive_monomial(&other.args()[0], x)? {
                        if *constant_term == int(-1) {
                            return Ok(Some(matched(bose_kernel()?, b, g)));
                        }
                        if *constant_term == int(1) {
                            return Ok(Some(matched(fermi_kernel()?, b, g)));
                        }
                    }
                }
            }
        }
        if func(base) == Some(Function::Sinh) {
            if let Some((b, g)) = positive_monomial(&base.args()[0], x)? {
                return Ok(Some(matched(csch_kernel()?, b, g)));
            }
        }
        if func(base) == Some(Function::Cosh) {
            if let Some((b, g)) = positive_monomial(&base.args()[0], x)? {
                return Ok(Some(matched(sech_kernel()?, b, g)));
            }
        }
    }
    if cutoff == Some("lower") && *base == neg(&crate::functions::elementary::exponential::log(x)?)? {
        return Ok(Some(Match {
            kernel: log_power_lower_kernel(exponent)?,
            beta: int(1),
            gamma: int(1),
            constant: int(1),
            cutoff: true,
        }));
    }
    if base.is_add() {
        let (constant, rest) = as_independent(base, x, true)?;
        if constant == int(0) {
            return Ok(None);
        }
        let Some((m, g)) = monomial(&rest, x)? else {
            return Ok(None);
        };
        let beta_: Basic = d2(&m, &constant)?;
        if beta_ == int(0) {
            return Ok(None);
        }
        if is(&beta_, Predicate::negative) == Some(true) {
            if beta_ != int(-1) || is(&g, Predicate::positive) != Some(true) {
                return Ok(None);
            }
            if cutoff == Some("lower") && is(&constant, Predicate::positive) == Some(true) {
                return Ok(Some(Match {
                    kernel: beta_kernel(&a2(exponent, &int(1))?)?,
                    beta: int(1),
                    gamma: g,
                    constant: pw(&constant, exponent)?,
                    cutoff: true,
                }));
            }
            if cutoff == Some("upper") && is(&constant, Predicate::negative) == Some(true) {
                return Ok(Some(Match {
                    kernel: beta_upper_kernel(&a2(exponent, &int(1))?)?,
                    beta: int(1),
                    gamma: g,
                    constant: pw(&neg(&constant)?, exponent)?,
                    cutoff: true,
                }));
            }
            return Ok(None);
        }
        return Ok(Some(Match {
            kernel: power_kernel(&neg(exponent)?)?,
            beta: beta_,
            gamma: g,
            constant: pw(&constant, exponent)?,
            cutoff: false,
        }));
    }
    Ok(None)
}

fn shifted(name: &str) -> Option<(&'static str, i32)> {
    // the shift: -1, or -pi/2 (code 2)
    match name {
        "exp" => Some(("exp - 1", 1)),
        "cos" => Some(("cos - 1", 1)),
        "atan" => Some(("atan - pi/2", 2)),
        "Si" => Some(("Si - pi/2", 2)),
        _ => None,
    }
}

fn match_add(f: &Basic, x: &Basic) -> PolyResult<Option<Match>> {
    if f.args().len() != 2 {
        return Ok(None);
    }
    for sign_ in [int(1), int(-1)] {
        let terms: Vec<Basic> = f.args().iter().map(|t| m2(&sign_, t)).collect::<PolyResult<_>>()?;
        for (constant, function) in [(&terms[0], &terms[1]), (&terms[1], &terms[0])] {
            if function.has(x) && !constant.has(x) {
                let Some(found) = match_function(function, x, None)? else {
                    continue;
                };
                if let Some((target, code)) = shifted(&found.kernel.name) {
                    let value: Basic = if code == 1 { int(-1) } else { d2(&neg(&pi())?, &int(2))? };
                    if found.constant == int(1) && *constant == value {
                        return Ok(Some(Match {
                            kernel: named_kernel(target)?,
                            beta: found.beta,
                            gamma: found.gamma,
                            constant: sign_.clone(),
                            cutoff: false,
                        }));
                    }
                }
            } else if function.has(x) && constant.has(x) {
                if let Some(found) = match_function(function, x, None)? {
                    if found.kernel.name == "sin" && found.constant == int(1) {
                        let target: Basic = neg(&m2(&found.beta, &pw(x, &found.gamma)?)?)?;
                        if *constant == target {
                            return Ok(Some(Match {
                                kernel: sin_minus_x_kernel()?,
                                beta: found.beta,
                                gamma: found.gamma,
                                constant: sign_.clone(),
                                cutoff: false,
                            }));
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

/// The kernel matching a factor (`mellin_kernel`).
pub fn mellin_kernel(f: &Basic, x: &Basic, cutoff: Option<&str>) -> PolyResult<Option<Match>> {
    if f.is_pow() {
        return match_power(f, x, cutoff);
    }
    if f.is_add() {
        return match_add(f, x);
    }
    match_function(f, x, cutoff)
}

/// The Mellin transform of `f` in `s` (`mellin_transform`).
pub fn mellin_transform(f: &Basic, x: &Basic, s: &Basic) -> PolyResult<Option<MellinTransform>> {
    let Some(product) = decompose_integrand(f, x, None)? else {
        return Ok(None);
    };
    if product.one_minus_power != 0 {
        return Ok(None);
    }
    let Some(quotient) = product.quotient(&[])? else {
        return Ok(None);
    };
    let mut transform: Basic = quotient.as_expr(s)?;
    if product.log_power > 0 {
        for _ in 0..product.log_power {
            transform = crate::core::function::derivative(&transform, s)?;
        }
        if product.log_power > 1 {
            transform = crate::core::exprtools::factor_terms(
                &crate::simplify::signsimp::signsimp(&transform, true)?,
                false,
                false,
                false,
                true,
            )?;
        }
    }
    Ok(Some(MellinTransform {
        transform,
        strip: quotient.strip(),
        condition: quotient.condition.clone(),
    }))
}

/// An integrand as `constant * x**alpha * log(x)**k * kernels`
/// (`Product`).
#[derive(Debug, Clone)]
pub struct Product {
    pub constant: Basic,
    pub alpha: Basic,
    pub log_power: i64,
    pub matches: Vec<Match>,
    pub one_minus_power: i64,
}

impl Product {
    /// The quotient of a single kernel (`quotient`).
    pub fn quotient(&self, assumptions: &[Basic]) -> PolyResult<Option<GammaQuotient>> {
        if self.matches.len() != 1 {
            return Ok(None);
        }
        Ok(Some(
            self.matches[0]
                .quotient(assumptions)?
                .compose(&self.alpha, &int(1))?
                .scaled(&self.constant)?,
        ))
    }
}

fn is_bessel(e: &Basic) -> bool {
    matches!(
        func(e),
        Some(Function::BesselJ | Function::BesselY | Function::BesselI | Function::BesselK)
    )
}

fn exp_besseli(factors: Vec<Basic>, x: &Basic) -> PolyResult<(Vec<Basic>, Option<Match>)> {
    for (i, f) in factors.iter().enumerate() {
        if func(f) == Some(Function::BesselI) && f.args().len() == 2 && !f.args()[0].has(x) {
            let Some(inner) = positive_monomial(&f.args()[1], x)? else {
                continue;
            };
            for (j, other) in factors.iter().enumerate() {
                if func(other) == Some(Function::Exp) && other.args().len() == 1 {
                    let Some(outer) = negated_monomial(&other.args()[0], x)? else {
                        continue;
                    };
                    if outer.1 != inner.1 {
                        continue;
                    }
                    let difference: Basic = s2(&outer.0, &inner.0)?;
                    let mut rest: Vec<Basic> = factors
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i && *k != j)
                        .map(|(_, f)| f.clone())
                        .collect();
                    if difference != int(0) {
                        rest.push(crate::functions::elementary::exponential::exp(&m2(
                            &neg(&difference)?,
                            &pw(x, &inner.1)?,
                        )?)?);
                    }
                    return Ok((rest, Some(matched(exp_besseli_kernel(&f.args()[0])?, inner.0, inner.1))));
                }
            }
        }
    }
    Ok((factors, None))
}

fn bessel_pair(first: Function, second: Function) -> Option<&'static str> {
    use Function as F;
    match (first, second) {
        (F::BesselJ, F::BesselJ) => Some("JJ"),
        (F::BesselK, F::BesselK) => Some("KK"),
        (F::BesselI, F::BesselK) => Some("IK"),
        (F::BesselJ, F::BesselY) => Some("JY"),
        (F::BesselJ, F::BesselK) => Some("JK"),
        _ => None,
    }
}

fn bessel_products(factors: Vec<Basic>, x: &Basic) -> PolyResult<(Vec<Basic>, Option<Match>)> {
    for (i, first) in factors.iter().enumerate() {
        if !is_bessel(first) || first.args().len() != 2 {
            continue;
        }
        let inner = positive_monomial(&first.args()[1], x)?;
        let Some(inner) = inner else { continue };
        if first.args()[0].has(x) {
            continue;
        }
        for (j, second) in factors.iter().enumerate() {
            if j == i || !is_bessel(second) || second.args().len() != 2 {
                continue;
            }
            if second.args()[0].has(x) || second.args()[1] != first.args()[1] {
                continue;
            }
            let Some(kind) = bessel_pair(func(first).expect("bessel"), func(second).expect("bessel")) else {
                continue;
            };
            let (mu, nu) = (&first.args()[0], &second.args()[0]);
            if matches!(kind, "JY" | "JK") && mu != nu {
                continue;
            }
            let rest: Vec<Basic> = factors
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i && *k != j)
                .map(|(_, f)| f.clone())
                .collect();
            return Ok((
                rest,
                Some(matched(bessel_product_kernel(kind, mu, nu)?, inner.0, inner.1)),
            ));
        }
    }
    Ok((factors, None))
}

fn erfc_exp(factors: Vec<Basic>, x: &Basic) -> PolyResult<(Vec<Basic>, Option<Match>)> {
    for (i, f) in factors.iter().enumerate() {
        if func(f) == Some(Function::Erfc) && f.args().len() == 1 {
            let Some(inner) = positive_monomial(&f.args()[0], x)? else {
                continue;
            };
            for (j, other) in factors.iter().enumerate() {
                if func(other) == Some(Function::Exp) && other.args().len() == 1 {
                    let Some(outer) = positive_monomial(&other.args()[0], x)? else {
                        continue;
                    };
                    if outer.1 != m2(&int(2), &inner.1)? {
                        continue;
                    }
                    let difference: Basic = s2(&outer.0, &pw(&inner.0, &int(2))?)?;
                    let mut rest: Vec<Basic> = factors
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i && *k != j)
                        .map(|(_, f)| f.clone())
                        .collect();
                    if difference != int(0) {
                        rest.push(crate::functions::elementary::exponential::exp(&m2(
                            &difference,
                            &pw(x, &outer.1)?,
                        )?)?);
                    }
                    return Ok((rest, Some(matched(erfc_exp_kernel()?, inner.0, inner.1))));
                }
            }
        }
    }
    Ok((factors, None))
}

fn small(e: &Basic) -> Option<i64> {
    if e.is_integer_value() {
        i64::try_from(&crate::numbers::numer(e.as_rational()?)).ok()
    } else {
        None
    }
}

/// An integrand split into a constant, a power of `x`, powers of
/// `log(x)` and kernels (`decompose_integrand`).
pub fn decompose_integrand(f: &Basic, x: &Basic, cutoff: Option<&str>) -> PolyResult<Option<Product>> {
    let mut constant: Basic = int(1);
    let mut alpha: Basic = int(0);
    let mut log_power: i64 = 0;
    let mut one_minus_power: i64 = 0;
    let mut matches: Vec<Match> = Vec::new();
    let mut factors: Vec<Basic> = Vec::new();
    let given: Vec<Basic> = if f.is_mul() { f.args().to_vec() } else { vec![f.clone()] };
    for e in given {
        let is_function_base: bool = e.is_pow() && matches!(e.args()[0].head(), Head::Function(_) | Head::Undefined(_));
        if e.is_pow() && small(&e.args()[1]).is_some_and(|k| k > 1) && is_function_base && e.has(x) {
            for _ in 0..small(&e.args()[1]).expect("an integer") {
                factors.push(e.args()[0].clone());
            }
        } else {
            factors.push(e);
        }
    }
    let mut combined: Vec<Match> = Vec::new();
    if factors.iter().filter(|e| is_bessel(e)).count() >= 2 {
        let (rest, found) = bessel_products(factors, x)?;
        factors = rest;
        combined.extend(found);
    }
    if factors.iter().any(|e| func(e) == Some(Function::BesselI)) {
        let (rest, found) = exp_besseli(factors, x)?;
        factors = rest;
        combined.extend(found);
    }
    if factors.iter().any(|e| func(e) == Some(Function::Erfc)) {
        let (rest, found) = erfc_exp(factors, x)?;
        factors = rest;
        combined.extend(found);
    }
    let one_minus_x: Basic = s2(&int(1), x)?;
    for e in &factors {
        if !e.has(x) {
            constant = m2(&constant, e)?;
            continue;
        }
        if *e == *x {
            alpha = a2(&alpha, &int(1))?;
            continue;
        }
        if e.is_pow() && e.args()[0] == *x && !e.args()[1].has(x) {
            alpha = a2(&alpha, &e.args()[1])?;
            continue;
        }
        let is_log_of = |n: &Basic, v: &Basic| func(n) == Some(Function::Log) && n.args()[0] == *v;
        if is_log_of(e, x) {
            log_power += 1;
            continue;
        }
        if e.is_pow() && is_log_of(&e.args()[0], x) && small(&e.args()[1]).is_some_and(|k| k > 0) {
            log_power += small(&e.args()[1]).expect("an integer");
            continue;
        }
        if cutoff == Some("lower") && is_log_of(e, &one_minus_x) {
            one_minus_power += 1;
            continue;
        }
        if cutoff == Some("lower")
            && e.is_pow()
            && is_log_of(&e.args()[0], &one_minus_x)
            && small(&e.args()[1]).is_some_and(|k| k > 0)
        {
            one_minus_power += small(&e.args()[1]).expect("an integer");
            continue;
        }
        let Some(found) = mellin_kernel(e, x, cutoff)? else {
            return Ok(None);
        };
        matches.push(found);
    }
    matches.extend(combined);
    if one_minus_power != 0 && cutoff == Some("lower") && !matches.iter().any(|m| m.cutoff) {
        matches.push(Match {
            kernel: beta_kernel(&int(1))?,
            beta: int(1),
            gamma: int(1),
            constant: int(1),
            cutoff: true,
        });
    }
    if one_minus_power != 0 && !matches.iter().any(|m| m.kernel.name == BETA_LOWER && m.gamma == int(1)) {
        return Ok(None);
    }
    if let Some(c) = cutoff {
        if !matches.iter().any(|m| m.cutoff) {
            let k: Kernel = if c == "lower" {
                theta_lower_kernel()?
            } else {
                theta_upper_kernel()?
            };
            matches.push(Match {
                kernel: k,
                beta: int(1),
                gamma: int(1),
                constant: int(1),
                cutoff: true,
            });
        }
    }
    if matches.len() > 2 {
        return Ok(None);
    }
    Ok(Some(Product {
        constant,
        alpha,
        log_power,
        matches,
        one_minus_power,
    }))
}
