//! Sequences, ported from `sympy_extras.series.sequences` (SymPy's
//! `sympy.series.sequences`): the sequences given by a formula
//! ([`Seq::Formula`]), by a period ([`Seq::Per`]) or by a recurrence
//! ([`Seq::Recursive`]), their termwise sums and products ([`Seq::Add`],
//! [`Seq::Mul`]), [`sequence`], [`seq_add`] and [`seq_mul`] which build them,
//! and `find_linear_recurrence`. The Python classes are the variants of one
//! enum; the nodes never evaluate, the functions combine two formulas into
//! one formula on the intersection of their intervals as SymPy's `SeqAdd`
//! and `SeqMul` do. The sequences are no expressions (as in SymPy).

// PORTED FROM SYMPY: sympy/series/sequences.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.
// Changes: strict typing; the classes as the variants of an enum;
// unevaluating variants with the evaluating functions sequence, seq_add,
// seq_mul; RecursiveSeq indexes its cache from start (SymPy's from 0).

use std::cell::RefCell;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is_true;
use crate::error::{value_error, PolyError, PolyResult};

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
fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn tuple(v: Vec<Basic>) -> Basic {
    Basic::new(Head::Tuple, v)
}
fn sstr(e: &Basic) -> String {
    crate::core::printing::sstr(e)
}

/// Whether `a < b` is known to hold (`_less`).
fn less(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(is_true(&sub(b, a)?, Predicate::extended_positive))
}

fn max2(a: &Basic, b: &Basic) -> R {
    crate::functions::elementary::miscellaneous::max_(&[a.clone(), b.clone()])
}

fn min2(a: &Basic, b: &Basic) -> R {
    crate::functions::elementary::miscellaneous::min_(&[a.clone(), b.clone()])
}

fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}

fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}

/// Whether `e` is zero once expanded or simplified (`_is_zero`).
fn is_zero(e: &Basic) -> PolyResult<bool> {
    Ok(*e == Basic::zero() || expand(e)? == Basic::zero() || simplify(e)? == Basic::zero())
}

fn empty_range(start: &Basic, stop: &Basic) -> PolyResult<bool> {
    less(stop, start)
}

/// A sequence on a range of its own variable given by a formula
/// (`SeqFormula`).
#[derive(Clone, Debug)]
pub struct SeqFormula {
    pub formula: Basic,
    pub variable: Basic,
    pub start: Basic,
    pub stop: Basic,
}

/// The periodic sequence repeating `periodical` (`SeqPer`).
#[derive(Clone, Debug)]
pub struct SeqPer {
    pub periodical: Vec<Basic>,
    pub variable: Basic,
    pub start: Basic,
    pub stop: Basic,
}

/// The sequence `a(n) = f(a(n - 1), ..., a(n - d))` of a recurrence of fixed
/// degree, from `start` to `oo` (`RecursiveSeq`); the terms computed are
/// cached.
#[derive(Clone, Debug)]
pub struct RecursiveSeq {
    pub recurrence: Basic,
    pub yn: Basic,
    pub n: Basic,
    pub initial: Vec<Basic>,
    pub start: Basic,
    pub degree: usize,
    terms: RefCell<Vec<Basic>>,
}

/// A sequence (`SeqBase` and its classes).
#[derive(Clone, Debug)]
pub enum Seq {
    /// The sequence without terms (`EmptySequence`).
    Empty,
    Formula(SeqFormula),
    Per(SeqPer),
    Recursive(RecursiveSeq),
    /// The termwise sum, unevaluated (`SeqAdd`; [`seq_add`] evaluates).
    Add(Vec<Seq>),
    /// The termwise product, unevaluated (`SeqMul`; [`seq_mul`] evaluates).
    Mul(Vec<Seq>),
}

/// The limits of a sequence as given: `(start, stop)` or `(variable, start,
/// stop)`.
pub enum SeqLimits {
    Range(Basic, Basic),
    Variable(Basic, Basic, Basic),
}

fn find_x(free: &[Basic], periodical: bool) -> PolyResult<Basic> {
    if free.len() == 1 {
        return Ok(free[0].clone());
    }
    if free.is_empty() || periodical {
        return Ok(crate::core::symbol::Dummy::new("k", &[]).0);
    }
    Err(value_error(
        "specify dummy variables for the formula. If the formula contains more than one free symbol, a dummy \
         variable should be supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))"
            .to_string(),
    ))
}

fn parse_limits(free: &[Basic], limits: Option<&SeqLimits>, periodical: bool) -> PolyResult<(Basic, Basic, Basic)> {
    match limits {
        None => Ok((find_x(free, periodical)?, Basic::zero(), Basic::oo())),
        Some(SeqLimits::Variable(v, a, b)) => {
            if !v.is_symbol() {
                return Err(value_error(format!("Invalid limits given: ({}, {}, {})", v, a, b)));
            }
            Ok((v.clone(), a.clone(), b.clone()))
        }
        Some(SeqLimits::Range(a, b)) => Ok((find_x(free, periodical)?, a.clone(), b.clone())),
    }
}

fn check_limits(variable: &Basic, start: &Basic, stop: &Basic) -> PolyResult<()> {
    if !variable.is_symbol() {
        return Err(value_error(format!(
            "Invalid limits given: ({}, {}, {})",
            variable, start, stop
        )));
    }
    if matches!(start.head(), Head::NegativeInfinity) && matches!(stop.head(), Head::Infinity) {
        return Err(value_error(
            "Both the start and end value cannot be unbounded".to_string(),
        ));
    }
    Ok(())
}

fn free_symbols_of(items: &[Basic]) -> Vec<Basic> {
    let mut free: Vec<Basic> = Vec::new();
    for e in items {
        for s in e.free_symbols() {
            if !free.contains(&s) {
                free.push(s);
            }
        }
    }
    free
}

impl SeqFormula {
    /// `SeqFormula(formula, limits)`.
    pub fn new(formula: &Basic, limits: Option<&SeqLimits>) -> PolyResult<SeqFormula> {
        let (variable, start, stop) = parse_limits(&free_symbols_of(std::slice::from_ref(formula)), limits, false)?;
        check_limits(&variable, &start, &stop)?;
        Ok(SeqFormula {
            formula: formula.clone(),
            variable,
            start,
            stop,
        })
    }

    fn limits(&self) -> SeqLimits {
        SeqLimits::Variable(self.variable.clone(), self.start.clone(), self.stop.clone())
    }

    /// The sequence of the expanded formula (`expand`).
    pub fn expand(&self) -> PolyResult<SeqFormula> {
        SeqFormula::new(&expand(&self.formula)?, Some(&self.limits()))
    }
}

impl SeqPer {
    /// `SeqPer(periodical, limits)`.
    pub fn new(periodical: &[Basic], limits: Option<&SeqLimits>) -> PolyResult<SeqPer> {
        if periodical.is_empty() {
            return Err(value_error(
                "invalid period () should be something like e.g (1, 2)".to_string(),
            ));
        }
        let (variable, start, stop) = parse_limits(&free_symbols_of(periodical), limits, true)?;
        check_limits(&variable, &start, &stop)?;
        Ok(SeqPer {
            periodical: periodical.to_vec(),
            variable,
            start,
            stop,
        })
    }

    pub fn period(&self) -> usize {
        self.periodical.len()
    }

    fn limits(&self) -> SeqLimits {
        SeqLimits::Variable(self.variable.clone(), self.start.clone(), self.stop.clone())
    }
}

/// The integer of a value, if it is one.
fn as_integer(e: &Basic) -> Option<i64> {
    let r = e.as_rational()?;
    if !r.is_int() {
        return None;
    }
    r.to_string().parse::<i64>().ok()
}

impl RecursiveSeq {
    /// `RecursiveSeq(recurrence, yn, n, initial, start)`: `recurrence` the
    /// expression `f(y(n - 1), ..., y(n - d))`, `yn` the applied undefined
    /// function `y(n)`; `initial` the `d` values at `start, ..., start + d -
    /// 1` (dummies `c_0, ...` when omitted).
    pub fn new(
        recurrence: &Basic,
        yn: &Basic,
        n: &Basic,
        initial: Option<&[Basic]>,
        start: &Basic,
    ) -> PolyResult<RecursiveSeq> {
        if !matches!(yn.head(), Head::Undefined(_)) {
            return Err(PolyError::Type(format!(
                "recurrence sequence must be an applied undefined function, found `{}`",
                yn
            )));
        }
        if !n.is_symbol() {
            return Err(PolyError::Type(format!(
                "recurrence variable must be a symbol, found `{}`",
                n
            )));
        }
        if yn.args() != std::slice::from_ref(n) {
            return Err(PolyError::Type("recurrence sequence does not match symbol".to_string()));
        }
        let mut degree: usize = 0;
        // every application of y has one argument n + k with k a negative integer
        for prev in recurrence.preorder() {
            if !prev.head().same_head(yn.head()) {
                continue;
            }
            if prev.args().len() != 1 {
                return Err(PolyError::Type("Recurrence should be in a single variable".to_string()));
            }
            let shift: Basic = sub(&prev.args()[0], n)?;
            match as_integer(&shift) {
                Some(k) if k < 0 => degree = degree.max((-k) as usize),
                _ => {
                    return Err(PolyError::Type(format!(
                        "Recurrence should have constant, negative, integer shifts (found {})",
                        prev
                    )))
                }
            }
        }
        let values: Vec<Basic> = match initial {
            Some(v) if !v.is_empty() => v.to_vec(),
            _ => (0..degree)
                .map(|k| crate::core::symbol::Dummy::new(&format!("c_{}", k), &[]).0)
                .collect(),
        };
        if values.len() != degree {
            return Err(value_error("Number of initial terms must equal degree".to_string()));
        }
        Ok(RecursiveSeq {
            recurrence: recurrence.clone(),
            yn: yn.clone(),
            n: n.clone(),
            initial: values.clone(),
            start: start.clone(),
            degree,
            terms: RefCell::new(values),
        })
    }

    /// The equation `y(n) = f(...)` of the recurrence (`recurrence`).
    pub fn equation(&self) -> R {
        crate::core::relational::eq(&self.yn, &self.recurrence)
    }

    fn apply(&self, argument: &Basic) -> Basic {
        Basic::new(self.yn.head().clone(), vec![argument.clone()])
    }

    fn eval_coeff(&self, pt: &Basic) -> R {
        let Some(i) = as_integer(&sub(pt, &self.start)?) else {
            return Err(value_error(format!(
                "a recursive sequence takes the points start + k, not {}",
                pt
            )));
        };
        if i < 0 {
            return Err(PolyError::Index(format!("Index {} out of bounds", pt)));
        }
        let i: usize = i as usize;
        loop {
            let current: usize = self.terms.borrow().len();
            if current > i {
                break;
            }
            let point: Basic = add(&[self.start.clone(), int(current as i64)])?;
            let mut e: Basic = subs(&self.recurrence, &self.n, &point)?;
            // subs, not SymPy's xreplace: the native xreplace does not evaluate
            for k in current.saturating_sub(self.degree)..current {
                let key: Basic = self.apply(&add(&[self.start.clone(), int(k as i64)])?);
                let value: Basic = self.terms.borrow()[k].clone();
                e = subs(&e, &key, &value)?;
            }
            self.terms.borrow_mut().push(e);
        }
        Ok(self.terms.borrow()[i].clone())
    }
}

impl Seq {
    /// The first point of the sequence (included).
    pub fn start(&self) -> R {
        match self {
            Seq::Empty => Ok(Basic::oo()),
            Seq::Formula(f) => Ok(f.start.clone()),
            Seq::Per(p) => Ok(p.start.clone()),
            Seq::Recursive(r) => Ok(r.start.clone()),
            Seq::Add(s) | Seq::Mul(s) => {
                let mut result: Basic = s[0].start()?;
                for t in &s[1..] {
                    result = max2(&result, &t.start()?)?;
                }
                Ok(result)
            }
        }
    }

    /// The last point of the sequence (included).
    pub fn stop(&self) -> R {
        match self {
            Seq::Empty => crate::core::mul::neg(&Basic::oo()),
            Seq::Formula(f) => Ok(f.stop.clone()),
            Seq::Per(p) => Ok(p.stop.clone()),
            Seq::Recursive(_) => Ok(Basic::oo()),
            Seq::Add(s) | Seq::Mul(s) => {
                let mut result: Basic = s[0].stop()?;
                for t in &s[1..] {
                    result = min2(&result, &t.stop()?)?;
                }
                Ok(result)
            }
        }
    }

    /// The number of points (`oo` for an infinite sequence).
    pub fn length(&self) -> R {
        if matches!(self, Seq::Empty) {
            return Ok(Basic::zero());
        }
        add(&[sub(&self.stop()?, &self.start()?)?, Basic::one()])
    }

    /// The bound variables.
    pub fn variables(&self) -> Vec<Basic> {
        match self {
            Seq::Empty | Seq::Recursive(_) => Vec::new(),
            Seq::Formula(f) => vec![f.variable.clone()],
            Seq::Per(p) => vec![p.variable.clone()],
            Seq::Add(s) | Seq::Mul(s) => s.iter().flat_map(|t| t.variables()).collect(),
        }
    }

    /// The term at the point `pt`; `IndexError` outside the range (`coeff`).
    pub fn coeff(&self, pt: &Basic) -> R {
        let (start, stop) = (self.start()?, self.stop()?);
        if less(pt, &start)? || less(&stop, pt)? {
            return Err(PolyError::Index(format!(
                "Index {} out of bounds ({}, {})",
                pt, start, stop
            )));
        }
        self.eval_coeff(pt)
    }

    fn eval_coeff(&self, pt: &Basic) -> R {
        match self {
            Seq::Empty => Err(PolyError::NotImplemented("(EmptySequence)._eval_coeff".to_string())),
            Seq::Formula(f) => subs(&f.formula, &f.variable, pt),
            Seq::Per(p) => {
                let idx: Basic = if matches!(p.start.head(), Head::NegativeInfinity) {
                    sub(&p.stop, pt)?
                } else {
                    sub(pt, &p.start)?
                };
                let Some(i) = as_integer(&idx) else {
                    return Err(value_error(format!(
                        "a periodic sequence takes integer points, not {}",
                        pt
                    )));
                };
                let k: usize = i.rem_euclid(p.period() as i64) as usize;
                subs(&p.periodical[k], &p.variable, pt)
            }
            Seq::Recursive(r) => r.eval_coeff(pt),
            Seq::Add(s) => add(&s.iter().map(|t| t.coeff(pt)).collect::<PolyResult<Vec<Basic>>>()?),
            Seq::Mul(s) => mul(&s.iter().map(|t| t.coeff(pt)).collect::<PolyResult<Vec<Basic>>>()?),
        }
    }

    /// The `i`-th point: from the start, or back from the stop for a
    /// sequence from `-oo` (`_ith_point`).
    pub fn ith_point(&self, i: i64) -> R {
        let start: Basic = self.start()?;
        if matches!(start.head(), Head::NegativeInfinity) {
            return sub(&self.stop()?, &int(i));
        }
        add(&[start, int(i)])
    }

    /// The terms `[start, stop)` by index (`seq[start:stop:step]`); `stop`
    /// `None` for the whole of a finite sequence.
    pub fn slice(&self, start: i64, stop: Option<i64>, step: i64) -> PolyResult<Vec<Basic>> {
        let stop: i64 = match stop {
            Some(s) => s,
            None => match as_integer(&self.length()?) {
                Some(n) => n,
                None => {
                    return Err(value_error(
                        "an infinite sequence needs the end of the slice".to_string(),
                    ))
                }
            },
        };
        let mut terms: Vec<Basic> = Vec::new();
        let mut i: i64 = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            terms.push(self.coeff(&self.ith_point(i)?)?);
            i += step;
        }
        Ok(terms)
    }

    /// The term of index `i` (`seq[i]`).
    pub fn get(&self, i: i64) -> R {
        self.coeff(&self.ith_point(i)?)
    }

    /// The termwise sum when `self` knows how to add `other` (`_add`).
    fn add_known(&self, other: &Seq) -> PolyResult<Option<Seq>> {
        match (self, other) {
            (Seq::Formula(a), Seq::Formula(b)) => {
                let formula: Basic = add(&[a.formula.clone(), subs(&b.formula, &b.variable, &a.variable)?])?;
                let (start, stop) = (max2(&a.start, &b.start)?, min2(&a.stop, &b.stop)?);
                Ok(Some(Seq::Formula(SeqFormula::new(
                    &formula,
                    Some(&SeqLimits::Variable(a.variable.clone(), start, stop)),
                )?)))
            }
            (Seq::Per(a), Seq::Per(b)) => Ok(Some(Seq::Per(combine(a, b, false)?))),
            _ => Ok(None),
        }
    }

    /// The termwise product when `self` knows how to multiply `other`
    /// (`_mul`).
    fn mul_known(&self, other: &Seq) -> PolyResult<Option<Seq>> {
        match (self, other) {
            (Seq::Formula(a), Seq::Formula(b)) => {
                let formula: Basic = mul(&[a.formula.clone(), subs(&b.formula, &b.variable, &a.variable)?])?;
                let (start, stop) = (max2(&a.start, &b.start)?, min2(&a.stop, &b.stop)?);
                Ok(Some(Seq::Formula(SeqFormula::new(
                    &formula,
                    Some(&SeqLimits::Variable(a.variable.clone(), start, stop)),
                )?)))
            }
            (Seq::Per(a), Seq::Per(b)) => Ok(Some(Seq::Per(combine(a, b, true)?))),
            _ => Ok(None),
        }
    }

    /// The sequence multiplied termwise by the expression `c` (`coeff_mul`).
    pub fn coeff_mul(&self, c: &Basic) -> PolyResult<Seq> {
        match self {
            Seq::Empty => Ok(Seq::Empty),
            Seq::Formula(f) => Ok(Seq::Formula(SeqFormula::new(
                &mul(&[f.formula.clone(), c.clone()])?,
                Some(&f.limits()),
            )?)),
            Seq::Per(p) => {
                let elements: Vec<Basic> = p
                    .periodical
                    .iter()
                    .map(|e| mul(&[e.clone(), c.clone()]))
                    .collect::<PolyResult<_>>()?;
                Ok(Seq::Per(SeqPer::new(&elements, Some(&p.limits()))?))
            }
            Seq::Recursive(_) => Err(PolyError::NotImplemented(format!("({}).coeff_mul", self))),
            Seq::Add(s) => seq_add(&s.iter().map(|t| t.coeff_mul(c)).collect::<PolyResult<Vec<Seq>>>()?),
            Seq::Mul(s) => {
                let mut items: Vec<Seq> = vec![s[0].coeff_mul(c)?];
                items.extend(s[1..].iter().cloned());
                seq_mul(&items)
            }
        }
    }

    /// `-self`.
    pub fn neg(&self) -> PolyResult<Seq> {
        self.coeff_mul(&int(-1))
    }

    /// `self + other`.
    pub fn plus(&self, other: &Seq) -> PolyResult<Seq> {
        seq_add(&[self.clone(), other.clone()])
    }

    /// `self - other`.
    pub fn minus(&self, other: &Seq) -> PolyResult<Seq> {
        seq_add(&[self.clone(), other.neg()?])
    }

    /// `self * other`.
    pub fn times(&self, other: &Seq) -> PolyResult<Seq> {
        seq_mul(&[self.clone(), other.clone()])
    }

    /// The shortest linear recurrence of order at most `n/2` (at most
    /// `min(d, n/2)` with `d`) which the first `n` terms satisfy: the
    /// coefficients `[b(1), b(2), ...]` of `x(n) = b(1)*x(n-1) + ...`, empty
    /// when there is none; with `gfvar` also the ordinary generating
    /// function (`find_linear_recurrence`).
    pub fn find_linear_recurrence(
        &self,
        n: i64,
        d: Option<usize>,
        gfvar: Option<&Basic>,
    ) -> PolyResult<(Vec<Basic>, Option<Basic>)> {
        use crate::matrices::dense::ExprMatrix;
        let x: Vec<Basic> = self
            .slice(0, Some(n), 1)?
            .iter()
            .map(|t| simplify(&expand(t)?))
            .collect::<PolyResult<_>>()?;
        let lx: usize = x.len();
        let r: usize = match d {
            None => lx / 2,
            Some(d) => d.min(lx / 2),
        };
        let mut coeffs: Vec<Basic> = Vec::new();
        for l in 1..=r {
            let l2: usize = 2 * l;
            let rows: Vec<Vec<Basic>> = (0..l).map(|k| x[k..k + l].to_vec()).collect();
            let m: ExprMatrix = ExprMatrix::from_rows(&rows)?;
            if crate::matrices::linalg::det(&m)? == Basic::zero() {
                continue;
            }
            let Some(solution) = crate::matrices::linalg::linear_solve(&m, &ExprMatrix::column(x[l..l2].to_vec()))?
            else {
                return Err(value_error("the system has no solution".to_string()));
            };
            let sol: Vec<Basic> = solution
                .particular
                .entries
                .iter()
                .map(simplify)
                .collect::<PolyResult<_>>()?;
            if lx == l2 {
                coeffs = sol.iter().rev().cloned().collect();
                break;
            }
            let rest_rows: Vec<Vec<Basic>> = (l..lx - l).map(|k| x[k..k + l].to_vec()).collect();
            let rest: ExprMatrix = ExprMatrix::from_rows(&rest_rows)?;
            let product: ExprMatrix = rest.matmul(&ExprMatrix::column(sol.clone()))?;
            // SymPy compares the matrices structurally; a zero difference is enough here
            let mut holds: bool = true;
            for (a, b) in product.entries.iter().zip(&x[l2..]) {
                if !is_zero(&sub(a, b)?)? {
                    holds = false;
                    break;
                }
            }
            if holds {
                coeffs = sol.iter().rev().cloned().collect();
                break;
            }
        }
        let Some(g) = gfvar else { return Ok((coeffs, None)) };
        let l: usize = coeffs.len();
        if l == 0 {
            return Ok((Vec::new(), None));
        }
        let pw = |e: &Basic, k: usize| -> R { crate::core::power::pow(e, &int(k as i64)) };
        let mut num: Basic = mul(&[x[l - 1].clone(), pw(g, l - 1)?])?;
        let mut den: Basic = sub(&Basic::one(), &mul(&[coeffs[l - 1].clone(), pw(g, l)?])?)?;
        for i in 0..l - 1 {
            num = add(&[num, mul(&[x[i].clone(), pw(g, i)?])?])?;
            for (j, xj) in x.iter().enumerate().take(l - i - 1) {
                num = sub(&num, &mul(&[coeffs[i].clone(), xj.clone(), pw(g, i + j + 1)?])?)?;
            }
            den = sub(&den, &mul(&[coeffs[i].clone(), pw(g, i + 1)?])?)?;
        }
        let factor = |e: &Basic| crate::polys::polytools::factor(e, &[], false, true);
        let gf: Basic = simplify(&crate::core::mul::div(&factor(&num)?, &factor(&den)?)?)?;
        Ok((coeffs, Some(gf)))
    }
}

fn combine(a: &SeqPer, b: &SeqPer, product: bool) -> PolyResult<SeqPer> {
    let (p, q) = (a.period(), b.period());
    let n: usize = p / gcd(p, q) * q;
    let mut new: Vec<Basic> = Vec::new();
    for i in 0..n {
        let (x, y) = (a.periodical[i % p].clone(), b.periodical[i % q].clone());
        new.push(if product { mul(&[x, y])? } else { add(&[x, y])? });
    }
    let (start, stop) = (max2(&a.start, &b.start)?, min2(&a.stop, &b.stop)?);
    SeqPer::new(&new, Some(&SeqLimits::Variable(a.variable.clone(), start, stop)))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl std::fmt::Display for Seq {
    /// SymPy's `str` of the sequences.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limits = |v: &Basic, a: &Basic, b: &Basic| sstr(&tuple(vec![v.clone(), a.clone(), b.clone()]));
        match self {
            Seq::Empty => write!(f, "EmptySequence"),
            Seq::Formula(s) => write!(
                f,
                "SeqFormula({}, {})",
                sstr(&s.formula),
                limits(&s.variable, &s.start, &s.stop)
            ),
            Seq::Per(s) => write!(
                f,
                "SeqPer({}, {})",
                sstr(&tuple(s.periodical.clone())),
                limits(&s.variable, &s.start, &s.stop)
            ),
            Seq::Recursive(r) => write!(
                f,
                "RecursiveSeq({}, {}, {}, {}, {})",
                sstr(&r.recurrence),
                sstr(&r.yn),
                sstr(&r.n),
                sstr(&tuple(r.initial.clone())),
                sstr(&r.start)
            ),
            Seq::Add(s) | Seq::Mul(s) => {
                let name: &str = if matches!(self, Seq::Add(_)) {
                    "SeqAdd"
                } else {
                    "SeqMul"
                };
                let parts: Vec<String> = s.iter().map(|t| t.to_string()).collect();
                write!(f, "{}({})", name, parts.join(", "))
            }
        }
    }
}

fn flatten(sequences: &[Seq], product: bool) -> Vec<Seq> {
    let mut result: Vec<Seq> = Vec::new();
    for s in sequences {
        match (s, product) {
            (Seq::Add(inner), false) | (Seq::Mul(inner), true) => result.extend(flatten(inner, product)),
            _ => result.push(s.clone()),
        }
    }
    result
}

/// Adapted from `Union.reduce`: combine the pairs which know how (`_reduce`).
fn reduce(mut args: Vec<Seq>, product: bool) -> PolyResult<Vec<Seq>> {
    let mut changed: bool = true;
    while changed {
        changed = false;
        'outer: for i in 0..args.len() {
            for j in 0..args.len() {
                if i == j {
                    continue;
                }
                let combined = if product {
                    args[i].mul_known(&args[j])?
                } else {
                    args[i].add_known(&args[j])?
                };
                if let Some(c) = combined {
                    let mut next: Vec<Seq> = args
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i && *k != j)
                        .map(|(_, a)| a.clone())
                        .collect();
                    next.push(c);
                    args = next;
                    changed = true;
                    break 'outer;
                }
            }
        }
    }
    Ok(args)
}

fn empty(args: &[Seq]) -> PolyResult<bool> {
    let mut start: Basic = args[0].start()?;
    let mut stop: Basic = args[0].stop()?;
    for a in &args[1..] {
        start = max2(&start, &a.start()?)?;
        stop = min2(&stop, &a.stop()?)?;
    }
    empty_range(&start, &stop)
}

fn sorted_by_start(mut args: Vec<Seq>) -> PolyResult<Vec<Seq>> {
    let mut keyed: Vec<(String, Seq)> = Vec::new();
    for a in args.drain(..) {
        keyed.push((sstr(&a.start()?), a));
    }
    keyed.sort_by(|x, y| x.0.cmp(&y.0));
    Ok(keyed.into_iter().map(|(_, a)| a).collect())
}

/// The termwise sum of the sequences on the intersection of their ranges:
/// two formulas give one formula (SymPy's `SeqAdd`).
pub fn seq_add(sequences: &[Seq]) -> PolyResult<Seq> {
    let args: Vec<Seq> = flatten(sequences, false)
        .into_iter()
        .filter(|a| !matches!(a, Seq::Empty))
        .collect();
    if args.is_empty() || empty(&args)? {
        return Ok(Seq::Empty);
    }
    let reduced: Vec<Seq> = reduce(args, false)?;
    if reduced.len() == 1 {
        return Ok(reduced.into_iter().next().expect("one"));
    }
    Ok(Seq::Add(sorted_by_start(reduced)?))
}

/// The termwise product of the sequences on the intersection of their
/// ranges: two formulas give one formula (SymPy's `SeqMul`).
pub fn seq_mul(sequences: &[Seq]) -> PolyResult<Seq> {
    let args: Vec<Seq> = flatten(sequences, true);
    if args.is_empty() || args.iter().any(|a| matches!(a, Seq::Empty)) || empty(&args)? {
        return Ok(Seq::Empty);
    }
    let reduced: Vec<Seq> = reduce(args, true)?;
    if reduced.len() == 1 {
        return Ok(reduced.into_iter().next().expect("one"));
    }
    Ok(Seq::Mul(sorted_by_start(reduced)?))
}

/// What `sequence` takes: a formula, or the elements of a period.
pub enum SeqSource {
    Formula(Basic),
    Periodical(Vec<Basic>),
}

/// The sequence of a formula (`SeqFormula`) or, for a period, the periodic
/// sequence (`SeqPer`); `EmptySequence` for an empty range (`sequence`).
pub fn sequence(seq: &SeqSource, limits: Option<&SeqLimits>) -> PolyResult<Seq> {
    let result: Seq = match seq {
        SeqSource::Formula(f) => Seq::Formula(SeqFormula::new(f, limits)?),
        SeqSource::Periodical(p) => Seq::Per(SeqPer::new(p, limits)?),
    };
    if empty_range(&result.start()?, &result.stop()?)? {
        return Ok(Seq::Empty);
    }
    Ok(result)
}
