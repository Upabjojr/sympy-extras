//! Fourier series, ported from `sympy_extras.series.fourier` (SymPy's
//! `sympy.series.fourier`): [`fourier_series`] computes the trigonometric
//! series of a function on an interval `(a, b)` as a [`FourierSeries`]
//! (the coefficients as formulas in a positive integer index) or, for a
//! trigonometric polynomial of the period, as a [`FiniteFourierSeries`]
//! (its coefficients read off, [`finite_check`] and
//! [`finite_fourier_series`]). The series are no expressions here: adding
//! one to an expression which is not a series is not ported.

// PORTED FROM SYMPY: sympy/series/fourier.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: strict typing; the coefficients integrated for a positive
// integer index; FiniteFourierSeries split from FourierSeries and from its
// conversion (finite_fourier_series).

use std::sync::Arc;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::{not_ported, value_error, PolyError, PolyResult};
use crate::series::sequences::{Seq, SeqFormula, SeqLimits};
use crate::series::series_class::SeriesBase;

type R = PolyResult<Basic>;

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
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn sin(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::sin(e)
}
fn cos(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::cos(e)
}
fn abs_(e: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(e)
}

/// `isinstance(e, Number) and e == 0` (`_is_zero`).
fn is_zero(e: &Basic) -> bool {
    e.is_number_class() && *e == Basic::zero()
}

/// The terms of a sum after its number (`as_coeff_add()`).
fn coeff_add(e: &Basic) -> (Basic, Vec<Basic>) {
    let (number, rest) = crate::core::expr::as_coeff_add(e);
    let terms: Vec<Basic> = if rest == Basic::zero() {
        Vec::new()
    } else if rest.is_add() {
        rest.args().to_vec()
    } else {
        vec![rest]
    };
    (number, terms)
}

/// `Integral(f, (x, a, b))`, with the index `n` a positive integer when
/// given (`_integral`).
fn integral(f: &Basic, x: &Basic, a: &Basic, b: &Basic, n: Option<&Basic>) -> R {
    let options = crate::integrals::definite::Options::default();
    let facts: Vec<Basic> = match n {
        None => Vec::new(),
        Some(n) => vec![crate::assumptions::facts::element(n, &Basic::atom(Head::Naturals))?],
    };
    crate::integrals::definite::definite_integral(f, x, a, b, &facts, &options)
}

fn index() -> Basic {
    crate::core::symbol::Dummy::new("n", &[(Predicate::integer, true), (Predicate::positive, true)]).0
}

/// The limits `(x, start, stop)` of a Fourier series.
#[derive(Clone, Debug, PartialEq)]
pub struct FourierLimits {
    pub x: Basic,
    pub start: Basic,
    pub stop: Basic,
}

/// The limits as given to [`fourier_series`].
pub enum GivenLimits {
    Default,
    Range(Basic, Basic),
    Variable(Basic, Basic, Basic),
}

fn formula_seq(formula: &Basic, variable: &Basic) -> PolyResult<SeqFormula> {
    SeqFormula::new(
        formula,
        Some(&SeqLimits::Variable(variable.clone(), Basic::one(), Basic::oo())),
    )
}

fn zero_seq() -> PolyResult<SeqFormula> {
    SeqFormula::new(&Basic::zero(), Some(&SeqLimits::Range(Basic::one(), Basic::oo())))
}

/// The constant term and the sequence of the cosine terms of the Fourier
/// series of `func`, the index `n` a positive integer (`fourier_cos_seq`).
pub fn fourier_cos_seq(func: &Basic, limits: &FourierLimits, n: &Basic) -> PolyResult<(Basic, SeqFormula)> {
    let FourierLimits { x, start: a, stop: b } = limits;
    let l: Basic = sub(b, a)?;
    let cos_term: Basic = cos(&div(&mul(&[int(2), n.clone(), Basic::pi(), x.clone()])?, &l)?)?;
    let a0: Basic = div(&integral(func, x, a, b, None)?, &l)?;
    let coefficient: Basic = integral(&mul(&[func.clone(), cos_term.clone()])?, x, a, b, Some(n))?;
    let formula: Basic = div(&mul(&[int(2), cos_term, coefficient])?, &l)?;
    Ok((a0, formula_seq(&formula, n)?))
}

/// The sequence of the sine terms of the Fourier series of `func`
/// (`fourier_sin_seq`).
pub fn fourier_sin_seq(func: &Basic, limits: &FourierLimits, n: &Basic) -> PolyResult<SeqFormula> {
    let FourierLimits { x, start: a, stop: b } = limits;
    let l: Basic = sub(b, a)?;
    let sin_term: Basic = sin(&div(&mul(&[int(2), n.clone(), Basic::pi(), x.clone()])?, &l)?)?;
    let coefficient: Basic = integral(&mul(&[func.clone(), sin_term.clone()])?, x, a, b, Some(n))?;
    formula_seq(&div(&mul(&[int(2), sin_term, coefficient])?, &l)?, n)
}

/// The limits `(x, start, stop)`: `x` from `func` when not given, `(-pi,
/// pi)` by default; both ends bounded (`_process_limits`).
pub fn process_limits(func: &Basic, limits: &GivenLimits) -> PolyResult<FourierLimits> {
    let find_x = || -> R {
        let free: Vec<Basic> = func.free_symbols().into_iter().collect();
        if free.len() == 1 {
            return Ok(free[0].clone());
        }
        if free.is_empty() {
            return Ok(crate::core::symbol::Dummy::new("k", &[]).0);
        }
        Err(value_error(format!(
            " specify dummy variables for {}. If the function contains more than one free symbol, a dummy variable \
             should be supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))",
            func
        )))
    };
    let (x, start, stop) = match limits {
        GivenLimits::Default => (find_x()?, crate::core::mul::neg(&Basic::pi())?, Basic::pi()),
        GivenLimits::Range(a, b) => (find_x()?, a.clone(), b.clone()),
        GivenLimits::Variable(x, a, b) => (x.clone(), a.clone(), b.clone()),
    };
    if !x.is_symbol() {
        return Err(value_error(format!(
            "Invalid limits given: ({}, {}, {})",
            x, start, stop
        )));
    }
    let unbounded = |e: &Basic| matches!(e.head(), Head::Infinity | Head::NegativeInfinity);
    if unbounded(&start) || unbounded(&stop) {
        return Err(value_error(
            "Both the start and end value should be bounded".to_string(),
        ));
    }
    Ok(FourierLimits { x, start, stop })
}

/// The wilds `a` (a nonzero integer) and `b` (free of `x`) of the
/// harmonics (`_harmonic_wilds`).
fn harmonic_wilds(x: &Basic) -> (Basic, Basic) {
    use crate::core::symbol::{Wild, WildProperty};
    let integer: WildProperty = Arc::new(|k: &Basic| is_integer_class(k));
    let nonzero: WildProperty = Arc::new(|k: &Basic| *k != Basic::zero());
    let xc: Basic = x.clone();
    let free: WildProperty = Arc::new(move |k: &Basic| !k.free_symbols().contains(&xc));
    let a: Basic = Wild::new("a", Vec::new(), vec![integer, nonzero], &[]).0;
    let b: Basic = Wild::new("b", Vec::new(), vec![free], &[]).0;
    (a, b)
}

fn bound(bindings: &[(Basic, Basic)], wild: &Basic) -> Option<Basic> {
    bindings.iter().find(|(w, _)| w == wild).map(|(_, v)| v.clone())
}

/// Whether `f` is a trigonometric polynomial of the period `2 L`, with its
/// form as a sum of sines and cosines of multiples of `pi x/L` (`f` itself
/// when it is not) (`finite_check`).
pub fn finite_check(f: &Basic, x: &Basic, l: &Basic) -> PolyResult<(bool, Basic)> {
    use crate::core::basic::Function;
    let (a, b) = harmonic_wilds(x);
    let pattern: Basic = add(&[mul(&[a, div(&Basic::pi(), l)?, x.clone()])?, b])?;
    let expr: Basic = crate::simplify::fu::sincos_to_sum(&crate::simplify::fu::tr2(&crate::simplify::fu::tr1(f)?)?)?;
    for s in coeff_add(&expr).1 {
        for t in crate::core::expr::as_coeff_mul_tuple(&s).1 {
            if !t.free_symbols().contains(x) {
                continue;
            }
            let harmonic: bool = matches!(t.head(), Head::Function(Function::Sin | Function::Cos))
                && crate::core::matching::match_pattern(&t.args()[0], &pattern, &[], false)?.is_some();
            if !harmonic {
                return Ok((false, f.clone()));
            }
        }
    }
    Ok((true, expr))
}

/// The substitution `old -> new` in a formula sequence; the bound variable
/// stays (`SeqFormula._subs`).
fn seq_subs(s: &SeqFormula, old: &Basic, new: &Basic) -> PolyResult<SeqFormula> {
    if *old == s.variable {
        return Ok(s.clone());
    }
    Ok(SeqFormula {
        formula: subs(&s.formula, old, new)?,
        variable: s.variable.clone(),
        start: subs(&s.start, old, new)?,
        stop: subs(&s.stop, old, new)?,
    })
}

fn as_formula(s: Seq) -> PolyResult<SeqFormula> {
    match s {
        Seq::Formula(f) => Ok(f),
        other => Err(PolyError::Type(format!(
            "a sequence of a formula is expected, got {}",
            other
        ))),
    }
}

/// The Fourier sine and cosine series of a function: `a0` and the
/// sequences `an`, `bn` of the cosine and sine terms (`FourierSeries`).
#[derive(Clone, Debug)]
pub struct FourierSeries {
    pub function: Basic,
    pub limits: FourierLimits,
    pub a0: Basic,
    pub an: SeqFormula,
    pub bn: SeqFormula,
}

/// A trigonometric polynomial as a Fourier series: the coefficients of
/// `cos(k pi x/L)` and `sin(k pi x/L)` by `k`, in the order found
/// (`FiniteFourierSeries`).
#[derive(Clone, Debug)]
pub struct FiniteFourierSeries {
    pub function: Basic,
    pub limits: FourierLimits,
    pub a0: Basic,
    pub an: Vec<(Basic, Basic)>,
    pub bn: Vec<(Basic, Basic)>,
}

/// What [`fourier_series`] and the operations on the series give.
#[derive(Clone, Debug)]
pub enum Fourier {
    /// The function itself (it does not depend on `x`).
    Expr(Basic),
    Series(FourierSeries),
    Finite(FiniteFourierSeries),
}

/// What the infinite and the finite Fourier series share (`_FourierBase`).
pub trait FourierBase: SeriesBase {
    fn limits(&self) -> &FourierLimits;
    fn function(&self) -> &Basic;
    fn a0(&self) -> &Basic;

    fn x(&self) -> &Basic {
        &self.limits().x
    }

    /// `(start, stop)` of the interval.
    fn period(&self) -> (Basic, Basic) {
        (self.limits().start.clone(), self.limits().stop.clone())
    }

    /// Half the period.
    fn half_period(&self) -> R {
        let (a, b) = self.period();
        div(&abs_(&sub(&b, &a)?)?, &int(2))
    }

    /// `s` checked free of `x` (`_check_free`).
    fn check_free(&self, s: &Basic) -> R {
        if s.free_symbols().contains(self.x()) {
            return Err(value_error(format!("'{}' should be independent of {}", s, self.x())));
        }
        Ok(s.clone())
    }

    /// The sum of the first `n` nonzero terms; all the terms of a finite
    /// series for `n` `None` (`truncate(oo)`).
    fn truncate(&self, n: Option<usize>) -> R {
        let bound: Option<i64> = self.length_bound()?;
        let mut terms: Vec<Basic> = Vec::new();
        let mut i: i64 = 0;
        while bound.map_or(true, |b| i < b) {
            if n.is_some_and(|limit| terms.len() == limit) {
                break;
            }
            let t: Basic = self.item(i)?;
            if !is_zero(&t) {
                terms.push(t);
            }
            i += 1;
        }
        add(&terms)
    }

    /// The sigma-approximation of order `n` (`sigma_approximation`): needs
    /// `sinc`, which the Rust core has not.
    fn sigma_approximation(&self, _n: usize) -> R {
        Err(not_ported("sinc (FourierSeries.sigma_approximation)"))
    }

    /// The first nonzero term (`_eval_as_leading_term`).
    fn leading_term(&self) -> R {
        let bound: Option<i64> = self.length_bound()?;
        let mut i: i64 = 0;
        while bound.map_or(true, |b| i < b) {
            let t: Basic = self.item(i)?;
            if !is_zero(&t) {
                return Ok(t);
            }
            i += 1;
        }
        Ok(Basic::zero())
    }
}

impl SeriesBase for FourierSeries {
    fn start(&self) -> R {
        Ok(Basic::zero())
    }
    fn stop(&self) -> R {
        Ok(Basic::oo())
    }
    fn length(&self) -> R {
        Ok(Basic::oo())
    }
    fn eval_term(&self, pt: &Basic) -> R {
        if *pt == Basic::zero() {
            return Ok(self.a0.clone());
        }
        add(&[
            Seq::Formula(self.an.clone()).coeff(pt)?,
            Seq::Formula(self.bn.clone()).coeff(pt)?,
        ])
    }
}

impl FourierBase for FourierSeries {
    fn limits(&self) -> &FourierLimits {
        &self.limits
    }
    fn function(&self) -> &Basic {
        &self.function
    }
    fn a0(&self) -> &Basic {
        &self.a0
    }
}

impl FourierSeries {
    fn make(&self, function: Basic, a0: Basic, an: SeqFormula, bn: SeqFormula) -> FourierSeries {
        FourierSeries {
            function,
            limits: self.limits.clone(),
            a0,
            an,
            bn,
        }
    }

    /// The series of `f(x) + s`, `s` free of `x` (`shift`).
    pub fn shift(&self, s: &Basic) -> PolyResult<FourierSeries> {
        let e: Basic = self.check_free(s)?;
        Ok(self.make(
            add(&[self.function.clone(), e.clone()])?,
            add(&[self.a0.clone(), e])?,
            self.an.clone(),
            self.bn.clone(),
        ))
    }

    fn substituted(&self, new: &Basic) -> PolyResult<FourierSeries> {
        let x: &Basic = &self.limits.x;
        Ok(self.make(
            subs(&self.function, x, new)?,
            self.a0.clone(),
            seq_subs(&self.an, x, new)?,
            seq_subs(&self.bn, x, new)?,
        ))
    }

    /// The series of `f(x + s)`, `s` free of `x` (`shiftx`).
    pub fn shiftx(&self, s: &Basic) -> PolyResult<FourierSeries> {
        let e: Basic = self.check_free(s)?;
        self.substituted(&add(&[self.limits.x.clone(), e])?)
    }

    /// The series of `s f(x)`, `s` free of `x` (`scale`).
    pub fn scale(&self, s: &Basic) -> PolyResult<FourierSeries> {
        let e: Basic = self.check_free(s)?;
        let an: SeqFormula = as_formula(Seq::Formula(self.an.clone()).coeff_mul(&e)?)?;
        let bn: SeqFormula = as_formula(Seq::Formula(self.bn.clone()).coeff_mul(&e)?)?;
        Ok(self.make(
            mul(&[self.function.clone(), e.clone()])?,
            mul(&[self.a0.clone(), e])?,
            an,
            bn,
        ))
    }

    /// The series of `f(s x)`, `s` free of `x` (`scalex`).
    pub fn scalex(&self, s: &Basic) -> PolyResult<FourierSeries> {
        let e: Basic = self.check_free(s)?;
        self.substituted(&mul(&[self.limits.x.clone(), e])?)
    }

    /// `-self`.
    pub fn neg(&self) -> PolyResult<FourierSeries> {
        self.scale(&int(-1))
    }

    /// The sum with another series of the same period (`__add__`).
    pub fn plus(&self, other: &FourierSeries) -> PolyResult<Fourier> {
        if self.period() != other.period() {
            return Err(value_error("Both the series should have same periods".to_string()));
        }
        let x: &Basic = &self.limits.x;
        let function: Basic = add(&[self.function.clone(), subs(&other.function, &other.limits.x, x)?])?;
        if !function.free_symbols().contains(x) {
            return Ok(Fourier::Expr(function));
        }
        let an: SeqFormula = as_formula(Seq::Formula(self.an.clone()).plus(&Seq::Formula(other.an.clone()))?)?;
        let bn: SeqFormula = as_formula(Seq::Formula(self.bn.clone()).plus(&Seq::Formula(other.bn.clone()))?)?;
        Ok(Fourier::Series(self.make(
            function,
            add(&[self.a0.clone(), other.a0.clone()])?,
            an,
            bn,
        )))
    }

    /// `self - other`.
    pub fn minus(&self, other: &FourierSeries) -> PolyResult<Fourier> {
        self.plus(&other.neg()?)
    }
}

/// `isinstance(e, Integer)`.
fn is_integer_class(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}

fn get(pairs: &[(Basic, Basic)], k: &Basic) -> Basic {
    pairs
        .iter()
        .find(|(key, _)| key == k)
        .map_or_else(Basic::zero, |(_, v)| v.clone())
}

fn as_small_integer(e: &Basic) -> Option<i64> {
    if !is_integer_class(e) {
        return None;
    }
    e.to_string().parse::<i64>().ok()
}

impl SeriesBase for FiniteFourierSeries {
    fn start(&self) -> R {
        Ok(Basic::zero())
    }
    fn stop(&self) -> R {
        let keys: Vec<i64> = self
            .an
            .iter()
            .chain(&self.bn)
            .filter_map(|(k, _)| as_small_integer(k))
            .collect();
        let length: i64 = i64::from(!is_zero(&self.a0)) + keys.iter().copied().max().unwrap_or(0) + 1;
        Ok(int(length))
    }
    fn length(&self) -> R {
        sub(&SeriesBase::stop(self)?, &SeriesBase::start(self)?)
    }
    fn eval_term(&self, pt: &Basic) -> R {
        if *pt == Basic::zero() {
            return Ok(self.a0.clone());
        }
        let angle: Basic = mul(&[
            pt.clone(),
            div(&Basic::pi(), &self.half_period()?)?,
            self.limits.x.clone(),
        ])?;
        add(&[
            mul(&[get(&self.an, pt), cos(&angle)?])?,
            mul(&[get(&self.bn, pt), sin(&angle)?])?,
        ])
    }
}

impl FourierBase for FiniteFourierSeries {
    fn limits(&self) -> &FourierLimits {
        &self.limits
    }
    fn function(&self) -> &Basic {
        &self.function
    }
    fn a0(&self) -> &Basic {
        &self.a0
    }
}

impl FiniteFourierSeries {
    /// The series of `f(x) + s`, `s` free of `x` (`shift`).
    pub fn shift(&self, s: &Basic) -> PolyResult<FiniteFourierSeries> {
        let e: Basic = self.check_free(s)?;
        Ok(FiniteFourierSeries {
            function: add(&[self.function.clone(), e.clone()])?,
            limits: self.limits.clone(),
            a0: add(&[self.a0.clone(), e])?,
            an: self.an.clone(),
            bn: self.bn.clone(),
        })
    }

    /// The series of `f(x + s)`, `s` free of `x` (`shiftx`).
    pub fn shiftx(&self, s: &Basic) -> PolyResult<FiniteFourierSeries> {
        let e: Basic = self.check_free(s)?;
        let x: &Basic = &self.limits.x;
        let new: Basic = add(&[x.clone(), e])?;
        finite_fourier_series(
            &subs(&self.function, x, &new)?,
            &self.limits,
            &subs(&self.truncate(None)?, x, &new)?,
        )
    }

    /// The series of `s f(x)`, `s` free of `x` (`scale`).
    pub fn scale(&self, s: &Basic) -> PolyResult<FiniteFourierSeries> {
        let e: Basic = self.check_free(s)?;
        finite_fourier_series(
            &mul(&[self.function.clone(), e.clone()])?,
            &self.limits,
            &mul(&[self.truncate(None)?, e])?,
        )
    }

    /// The series of `f(s x)`, `s` free of `x` (`scalex`).
    pub fn scalex(&self, s: &Basic) -> PolyResult<FiniteFourierSeries> {
        let e: Basic = self.check_free(s)?;
        let x: &Basic = &self.limits.x;
        let new: Basic = mul(&[x.clone(), e])?;
        finite_fourier_series(
            &subs(&self.function, x, &new)?,
            &self.limits,
            &subs(&self.truncate(None)?, x, &new)?,
        )
    }

    /// `-self`.
    pub fn neg(&self) -> PolyResult<FiniteFourierSeries> {
        self.scale(&int(-1))
    }

    /// The sum with a series of the same period (`__add__`): with an
    /// infinite one the infinite series of the sum, with a finite one the
    /// series of the sum.
    pub fn plus(&self, other: &Fourier) -> PolyResult<Fourier> {
        match other {
            Fourier::Series(o) => match fourier_series(&self.function, &limits_given(&self.limits), false)? {
                Fourier::Series(mine) => o.plus(&mine),
                _ => Err(value_error("the series of a function of x".to_string())),
            },
            Fourier::Finite(o) => {
                if self.period() != o.period() {
                    return Err(value_error("Both the series should have same periods".to_string()));
                }
                let x: &Basic = &self.limits.x;
                let function: Basic = add(&[self.function.clone(), subs(&o.function, &o.limits.x, x)?])?;
                if !function.free_symbols().contains(x) {
                    return Ok(Fourier::Expr(function));
                }
                fourier_series(&function, &limits_given(&self.limits), true)
            }
            Fourier::Expr(_) => Err(not_ported("the sum of a Fourier series and an expression")),
        }
    }
}

fn limits_given(l: &FourierLimits) -> GivenLimits {
    GivenLimits::Variable(l.x.clone(), l.start.clone(), l.stop.clone())
}

/// The finite Fourier series of `f` whose trigonometric polynomial (in
/// sines and cosines of multiples of `pi x/L`) is `expr`: the coefficients
/// read off after the angles are expanded (`TR10`)
/// (`FiniteFourierSeries.__new__`).
pub fn finite_fourier_series(f: &Basic, limits: &FourierLimits, expr: &Basic) -> PolyResult<FiniteFourierSeries> {
    let x: &Basic = &limits.x;
    let (c, rest) = coeff_add(expr);
    let mut parts: Vec<Basic> = vec![c];
    for t in &rest {
        parts.push(crate::simplify::fu::tr10(t, true)?);
    }
    let rexpr: Basic = add(&parts)?;
    let hints = crate::core::function::Hints {
        trig: false,
        power_base: false,
        power_exp: false,
        log: false,
        ..crate::core::function::Hints::DEFAULT
    };
    let expanded: Basic = crate::core::function::expand_with(&rexpr, true, hints)?;
    let (number, terms) = coeff_add(&expanded);
    let mut a0: Basic = number;
    let l: Basic = div(&abs_(&sub(&limits.stop, &limits.start)?)?, &int(2))?;
    let (a, b) = harmonic_wilds(x);
    let angle: Basic = mul(&[a.clone(), div(&Basic::pi(), &l)?, x.clone()])?;
    let cos_pattern: Basic = mul(&[b.clone(), cos(&angle)?])?;
    let sin_pattern: Basic = mul(&[b.clone(), sin(&angle)?])?;
    let mut an: Vec<(Basic, Basic)> = Vec::new();
    let mut bn: Vec<(Basic, Basic)> = Vec::new();
    let accumulate = |pairs: &mut Vec<(Basic, Basic)>, k: Basic, v: Basic| -> PolyResult<()> {
        match pairs.iter_mut().find(|(key, _)| *key == k) {
            Some(entry) => entry.1 = add(&[v, entry.1.clone()])?,
            None => pairs.push((k, v)),
        }
        Ok(())
    };
    for p in &terms {
        let t = crate::core::matching::match_pattern(p, &cos_pattern, &[], false)?;
        let q = crate::core::matching::match_pattern(p, &sin_pattern, &[], false)?;
        if let Some(t) = t.filter(|t| !t.is_empty()) {
            let (Some(k), Some(v)) = (bound(&t, &a), bound(&t, &b)) else {
                return Err(value_error("a harmonic without its multiple".to_string()));
            };
            accumulate(&mut an, k, v)?;
        } else if let Some(q) = q.filter(|q| !q.is_empty()) {
            let (Some(k), Some(v)) = (bound(&q, &a), bound(&q, &b)) else {
                return Err(value_error("a harmonic without its multiple".to_string()));
            };
            accumulate(&mut bn, k, v)?;
        } else {
            a0 = add(&[a0, p.clone()])?;
        }
    }
    Ok(FiniteFourierSeries {
        function: f.clone(),
        limits: limits.clone(),
        a0,
        an,
        bn,
    })
}

/// The Fourier trigonometric series of `f` on `(start, stop)` (`(-pi,
/// pi)` by default): a finite series for a trigonometric polynomial of the
/// period (with `finite`), an infinite one otherwise; `f` itself when it
/// does not depend on `x` (`fourier_series`).
pub fn fourier_series(f: &Basic, limits: &GivenLimits, finite: bool) -> PolyResult<Fourier> {
    let lims: FourierLimits = process_limits(f, limits)?;
    let FourierLimits { x, start, stop } = &lims;
    if !f.free_symbols().contains(x) {
        return Ok(Fourier::Expr(f.clone()));
    }
    if finite {
        let l: Basic = div(&abs_(&sub(stop, start)?)?, &int(2))?;
        let (is_finite, res_f) = finite_check(f, x, &l)?;
        if is_finite {
            return Ok(Fourier::Finite(finite_fourier_series(f, &lims, &res_f)?));
        }
    }
    let n: Basic = index();
    let center: Basic = div(&add(&[start.clone(), stop.clone()])?, &int(2))?;
    if is(&center, Predicate::zero) == Some(true) {
        let neg_f: Basic = subs(f, x, &crate::core::mul::neg(x)?)?;
        if *f == neg_f {
            let (a0, an) = fourier_cos_seq(f, &lims, &n)?;
            return Ok(Fourier::Series(FourierSeries {
                function: f.clone(),
                limits: lims,
                a0,
                an,
                bn: zero_seq()?,
            }));
        }
        if *f == crate::core::mul::neg(&neg_f)? {
            let bn: SeqFormula = fourier_sin_seq(f, &lims, &n)?;
            return Ok(Fourier::Series(FourierSeries {
                function: f.clone(),
                limits: lims,
                a0: Basic::zero(),
                an: zero_seq()?,
                bn,
            }));
        }
    }
    let (a0, an) = fourier_cos_seq(f, &lims, &n)?;
    let bn: SeqFormula = fourier_sin_seq(f, &lims, &n)?;
    Ok(Fourier::Series(FourierSeries {
        function: f.clone(),
        limits: lims,
        a0,
        an,
        bn,
    }))
}
