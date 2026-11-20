//! Port of `sympy_extras/integrals/risch/risch.py` (SymPy's
//! `sympy.integrals.risch` with Aaron Meurer's exp-log and hypertangent
//! cases), the transcendental Risch algorithm of Bronstein's *Symbolic
//! Integration I* on the typed polynomials of [`super::tower`]: the
//! differential extension of an integrand (`DifferentialExtension`), the
//! derivations, and the reductions and integrations of the cases.
//! The Risch differential equations (`rde`, `prde`, `cde`) follow.

#![allow(
    clippy::mutable_key_type,
    clippy::type_complexity,
    clippy::too_many_arguments,
    non_snake_case
)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::tower::{RationalFunction, TowerPoly};
use crate::core::basic::{Basic, Function, Head};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;

/// The type of an extension of the tower (`exts[i]` describes `T[i + 1]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extension {
    Exp,
    Log,
    Tan,
    Atan,
}

impl Extension {
    pub fn name(self) -> &'static str {
        match self {
            Extension::Exp => "exp",
            Extension::Log => "log",
            Extension::Tan => "tan",
            Extension::Atan => "atan",
        }
    }
}

/// The case of a derivation (`get_case`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    Base,
    Primitive,
    Exp,
    Tan,
    OtherLinear,
    OtherNonlinear,
}

impl Case {
    pub fn name(self) -> &'static str {
        match self {
            Case::Base => "base",
            Case::Primitive => "primitive",
            Case::Exp => "exp",
            Case::Tan => "tan",
            Case::OtherLinear => "other_linear",
            Case::OtherNonlinear => "other_nonlinear",
        }
    }
}

/// A function of the generators for the back substitution: the body in
/// its variable (SymPy's `Lambda(i, func(arg.subs(x, i)))`).
#[derive(Clone, Debug)]
pub struct TFunc {
    pub var: Basic,
    pub body: Basic,
}

impl TFunc {
    pub fn apply(&self, value: &Basic) -> R {
        crate::core::subs::subs(&self.body, &self.var, value)
    }
}

/// An error of the algorithm proving that the integral is not elementary
/// (`NonElementaryIntegralException`).
pub fn non_elementary(message: &str) -> PolyError {
    PolyError::Value(format!("NonElementaryIntegral: {}", message))
}

/// Whether the error is [`non_elementary`].
pub fn is_non_elementary(e: &PolyError) -> bool {
    matches!(e, PolyError::Value(m) if m.starts_with("NonElementaryIntegral"))
}

static DUMMIES: AtomicU64 = AtomicU64::new(0);

/// A dummy with a name of its own (`unique_dummy`).
pub fn unique_dummy(name: &str) -> Basic {
    let k: u64 = DUMMIES.fetch_add(1, Ordering::Relaxed);
    crate::core::symbol::Dummy::new(&format!("{}_{}", name, k), &[]).0
}

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn mul(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn tan(e: &Basic) -> R {
    crate::functions::apply_function(Function::Tan, std::slice::from_ref(e))
}
fn atan(e: &Basic) -> R {
    crate::functions::apply_function(Function::Atan, std::slice::from_ref(e))
}
fn rat(r: &Rat) -> Basic {
    Basic::from_rational(r.clone())
}

pub(crate) fn element(e: &Basic) -> PolyResult<RationalFunction> {
    RationalFunction::from_expr(e)
}

pub(crate) fn xrep(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    let map: std::collections::HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}

pub(crate) fn subs_seq(e: &Basic, pairs: &[(Basic, Basic)]) -> R {
    let mut out: Basic = e.clone();
    for (a, b) in pairs {
        out = crate::core::subs::subs(&out, a, b)?;
    }
    Ok(out)
}

pub(crate) fn is_rational_function(e: &Basic, t: &[Basic]) -> bool {
    crate::core::expr::polynomial_like(e, t, crate::core::expr::PolynomialKind::Rational) == Some(true)
}

pub(crate) fn has_any(e: &Basic, t: &[Basic]) -> bool {
    t.iter().any(|v| e.has(v))
}

fn unsupported_constant(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if !m.starts_with(crate::error::NOT_PORTED_PREFIX))
}

/// `a/b` when it is a rational number (`_ratio`).
fn ratio(a: &Basic, b: &Basic) -> PolyResult<Option<Rat>> {
    let (fa, fb) = match (element(a), element(b)) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(e), _) | (_, Err(e)) if unsupported_constant(&e) => return Ok(None),
        (Err(e), _) | (_, Err(e)) => return Err(e),
    };
    if fb.is_zero() {
        return Ok(None);
    }
    Ok(fa.div(&fb)?.as_fraction())
}

fn fraction_of(e: &Basic) -> PolyResult<Rat> {
    match e.head() {
        Head::Rational { .. } => Ok(e.as_rational().expect("a rational").clone()),
        _ => Err(PolyError::Type(format!("a rational number is expected, got {}", e))),
    }
}

fn integer_lcm(a: &Int, b: &Int) -> Int {
    use dashu_base::Gcd;
    let g: Int = Int::from(a.gcd(b));
    (a / &g) * b
}

/// The expressions as integer multiples of common bases, the multiples of
/// content one (`integer_powers`).
pub fn integer_powers(exprs: &[Basic]) -> PolyResult<Vec<(Basic, Vec<(Basic, Basic)>)>> {
    let mut terms: Vec<(Basic, Vec<(Basic, Basic)>)> = Vec::new();
    'outer: for term in exprs {
        for (trm, list) in terms.iter_mut() {
            if let Some(a) = ratio(term, trm)? {
                list.push((term.clone(), rat(&a)));
                continue 'outer;
            }
        }
        terms.push((term.clone(), vec![(term.clone(), Basic::one())]));
    }
    let mut newterms: Vec<(Basic, Vec<(Basic, Basic)>)> = Vec::new();
    for (term, list) in terms {
        let mut common: Int = Int::ONE;
        for (_, i) in &list {
            if let Some(r) = i.as_rational() {
                common = integer_lcm(&common, &crate::numbers::denom(r));
            }
        }
        let newterm: Basic = mul(&[term, rat(&(Rat::ONE / Rat::from(common.clone())))])?;
        let mut multiples: Vec<(Basic, Basic)> = Vec::new();
        for (i, j) in list {
            multiples.push((i, rat(&(fraction_of(&j)? * Rat::from(common.clone())))));
        }
        newterms.push((newterm, multiples));
    }
    newterms.sort_by_cached_key(|(k, _)| crate::core::sorting::sort_key(k));
    Ok(newterms)
}

/// SymPy's `rewrite(tan)` of `sin`, `cos`, `sec`, `csc` (through the
/// tangent of the half angle) and `cot` (`_tan_form`).
fn tan_form(e: &Basic) -> R {
    let u: &Basic = &e.args()[0];
    let Head::Function(f) = e.head() else {
        return Ok(e.clone());
    };
    if *f == Function::Cot {
        return pw(&tan(u)?, &int(-1));
    }
    let h: Basic = tan(&mul(&[crate::core::mul::div(&int(1), &int(2))?, u.clone()])?)?;
    let h2: Basic = pw(&h, &int(2))?;
    let one: Basic = Basic::one();
    let h2p1: Basic = add(&[h2.clone(), one.clone()])?;
    let one_m_h2: Basic = add(&[one.clone(), mul(&[int(-1), h2.clone()])?])?;
    match f {
        Function::Sin => mul(&[int(2), h.clone(), pw(&h2p1, &int(-1))?]),
        Function::Cos => mul(&[one_m_h2, pw(&h2p1, &int(-1))?]),
        Function::Sec => mul(&[h2p1, pw(&one_m_h2, &int(-1))?]),
        _ => mul(&[h2p1, pw(&mul(&[int(2), h])?, &int(-1))?]),
    }
}

/// The differential extension of an integrand (`DifferentialExtension`):
/// the tower `T` of generators over `x` with their derivations `D`, the
/// integrand as a fraction `fa/fd` in the top generator, and the
/// functions and back substitutions to recover the expression.
#[derive(Clone, Debug)]
pub struct DifferentialExtension {
    pub f: Basic,
    pub origf: Basic,
    pub x: Basic,
    pub T: Vec<Basic>,
    pub D: Vec<TowerPoly>,
    pub fa: TowerPoly,
    pub fd: TowerPoly,
    pub Tfuncs: Vec<TFunc>,
    pub backsubs: Vec<(Basic, Basic)>,
    pub sincos_args: Vec<Basic>,
    pub exts: Vec<Option<Extension>>,
    pub extargs: Vec<Option<Basic>>,
    pub cases: Vec<Case>,
    pub case: Case,
    pub t: Basic,
    pub d: TowerPoly,
    pub newf: Basic,
    pub level: i64,
    ts: u64,
    pub dummy: bool,
}

fn atoms(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> Vec<Basic> {
    let mut out: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if p(&n) && !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

fn is_fn(e: &Basic, fs: &[Function]) -> bool {
    matches!(e.head(), Head::Function(f) if fs.contains(f))
}

/// `update_sets`: the elements of `seq` still among `atoms`, and the new
/// atoms which satisfy `func` (in the order of the atoms).
fn update_sets(seq: &[Basic], atoms_: &[Basic], func: &dyn Fn(&Basic) -> bool) -> Vec<Basic> {
    let mut out: Vec<Basic> = atoms_.iter().filter(|a| seq.contains(a)).cloned().collect();
    for a in atoms_ {
        if !seq.contains(a) && func(a) && !out.contains(a) {
            out.push(a.clone());
        }
    }
    out
}

fn cancel_expr(e: &Basic) -> R {
    match element(e) {
        Ok(r) => r.as_expr(),
        Err(err) if unsupported_constant(&err) => crate::polys::polytools::cancel(e, &[]),
        Err(err) => Err(err),
    }
}

impl DifferentialExtension {
    /// The differential extension of `f` in `x` (`handle_first` `log` or
    /// `exp`).
    pub fn new(
        f: &Basic,
        x: &Basic,
        handle_first: &str,
        dummy: bool,
        rewrite_complex: Option<bool>,
    ) -> PolyResult<Self> {
        if handle_first != "log" && handle_first != "exp" {
            return Err(PolyError::Value(format!(
                "handle_first must be 'log' or 'exp', not {}.",
                handle_first
            )));
        }
        let mut de = DifferentialExtension {
            f: f.clone(),
            origf: f.clone(),
            x: x.clone(),
            T: Vec::new(),
            D: Vec::new(),
            fa: TowerPoly::zero(x),
            fd: TowerPoly::one(x),
            Tfuncs: Vec::new(),
            backsubs: Vec::new(),
            sincos_args: Vec::new(),
            exts: Vec::new(),
            extargs: Vec::new(),
            cases: Vec::new(),
            case: Case::Base,
            t: x.clone(),
            d: TowerPoly::one(x),
            newf: f.clone(),
            level: -1,
            ts: 0,
            dummy,
        };
        de.reset();
        let mut exp_new_extension: Option<bool> = Some(true);
        let mut log_new_extension: bool = true;
        let complex: bool = rewrite_complex.unwrap_or_else(|| de.f.has(&Basic::imaginary_unit()));
        if complex {
            let hints = crate::core::rewriting::RewriteHints::default();
            let trig: [Function; 8] = [
                Function::Sin,
                Function::Cos,
                Function::Cot,
                Function::Tan,
                Function::Sinh,
                Function::Cosh,
                Function::Coth,
                Function::Tanh,
            ];
            de.newf = crate::core::rewriting::rewrite(&de.newf, &trig, "exp", &hints)?;
            let inverse: [Function; 4] = [Function::Asin, Function::Acos, Function::Acot, Function::Atan];
            de.newf = crate::core::rewriting::rewrite(&de.newf, &inverse, "log", &hints)?;
            de.newf = cancel_expr(&de.newf)?;
        } else {
            let bad = atoms(&de.f, &|n| {
                is_fn(
                    n,
                    &[
                        Function::Sinh,
                        Function::Cosh,
                        Function::Coth,
                        Function::Tanh,
                        Function::Asin,
                        Function::Acos,
                    ],
                )
            });
            if bad.iter().any(|i| i.has(x)) {
                return Err(PolyError::NotImplemented(
                    "Hyperbolic, arc-sine and arc-cosine extensions are not supported (yet!).  Try rewriting in terms of exp and log, or using rewrite_complex=True."
                        .to_string(),
                ));
            }
            de.rewrite_trig()?;
        }
        let mut exps: Vec<Basic> = Vec::new();
        let mut pows: Vec<Basic> = Vec::new();
        let mut numpows: Vec<Basic> = Vec::new();
        let mut sympows: Vec<Basic> = Vec::new();
        let mut logs: Vec<Basic> = Vec::new();
        let mut symlogs: Vec<Basic> = Vec::new();
        let mut tans: Vec<Basic> = Vec::new();
        let mut atans: Vec<Basic> = Vec::new();
        let (mut tan_new_extension, mut atan_new_extension) = (true, true);
        loop {
            crate::timeout::check()?;
            if is_rational_function(&de.newf, &de.T) {
                break;
            }
            if !(exp_new_extension.unwrap_or(false) || log_new_extension || tan_new_extension || atan_new_extension) {
                return Err(PolyError::NotImplemented(format!(
                    "Couldn't find an elementary transcendental extension for {}.  Try using a manual extension with the extension flag.",
                    f
                )));
            }
            let r = de.rewrite_exps_pows(exps, pows, numpows, sympows, log_new_extension)?;
            exps = r.0;
            pows = r.1;
            numpows = r.2;
            sympows = r.3;
            log_new_extension = r.4;
            let (l, s) = de.rewrite_logs(logs, symlogs)?;
            logs = l;
            symlogs = s;
            if handle_first == "exp" || !log_new_extension {
                exp_new_extension = de.exp_part(&exps)?;
                if exp_new_extension.is_none() {
                    de.newf = subs_seq(&de.newf, &de.backsubs)?;
                    de.f = de.newf.clone();
                    de.reset();
                    de.rewrite_trig()?;
                    exp_new_extension = Some(true);
                    continue;
                }
            }
            if handle_first == "log" || exp_new_extension == Some(false) {
                log_new_extension = de.log_part(&logs)?;
            }
            let t_list: Vec<Basic> = de.T.clone();
            let generic = |i: &Basic| is_rational_function(&i.args()[0], &t_list) && has_any(&i.args()[0], &t_list);
            tans = update_sets(&tans, &atoms(&de.newf, &|n| is_fn(n, &[Function::Tan])), &generic);
            tan_new_extension = de.tan_part(&tans)?;
            let t_list: Vec<Basic> = de.T.clone();
            let generic = |i: &Basic| is_rational_function(&i.args()[0], &t_list) && has_any(&i.args()[0], &t_list);
            atans = update_sets(&atans, &atoms(&de.newf, &|n| is_fn(n, &[Function::Atan])), &generic);
            atan_new_extension = de.atan_part(&atans)?;
        }
        let (fa, fd) = frac_in_expr(&de.newf, &de.t, false)?;
        de.fa = fa;
        de.fd = fd;
        de.auto_attrs();
        Ok(de)
    }

    /// A differential extension given by its derivations (`manual`).
    pub fn manual(
        D: Vec<TowerPoly>,
        exts: Vec<Option<Extension>>,
        extargs: Vec<Option<Basic>>,
        x: Option<Basic>,
        fa: Option<TowerPoly>,
        fd: Option<TowerPoly>,
    ) -> Self {
        let T: Vec<Basic> = D.iter().map(|d| d.gen.clone()).collect();
        let x: Basic = x.unwrap_or_else(|| T[0].clone());
        let top: Basic = T[T.len() - 1].clone();
        let mut de = DifferentialExtension {
            f: Basic::zero(),
            origf: Basic::zero(),
            x: x.clone(),
            T,
            D,
            fa: fa.unwrap_or_else(|| TowerPoly::zero(&top)),
            fd: fd.unwrap_or_else(|| TowerPoly::one(&top)),
            Tfuncs: Vec::new(),
            backsubs: Vec::new(),
            sincos_args: Vec::new(),
            exts,
            extargs,
            cases: Vec::new(),
            case: Case::Base,
            t: x.clone(),
            d: TowerPoly::one(&x),
            newf: Basic::zero(),
            level: -1,
            ts: 0,
            dummy: false,
        };
        de.auto_attrs();
        de
    }

    fn next_t(&mut self) -> Basic {
        let k: u64 = self.ts;
        self.ts += 1;
        let name: String = format!("t{}", k);
        if self.dummy {
            crate::core::symbol::Dummy::new(&name, &[]).0
        } else {
            crate::core::symbol::Symbol::new(&name, &[]).0
        }
    }

    fn rewrite_trig(&mut self) -> PolyResult<()> {
        let mut acots: Vec<(Basic, Basic)> = Vec::new();
        loop {
            let mut reps: Vec<(Basic, Basic)> = Vec::new();
            let trig: [Function; 5] = [
                Function::Sin,
                Function::Cos,
                Function::Sec,
                Function::Csc,
                Function::Cot,
            ];
            for i in atoms(&self.newf, &|n| is_fn(n, &trig)) {
                if i.has(&self.x) {
                    reps.push((i.clone(), tan_form(&i)?));
                    if !is_fn(&i, &[Function::Cot]) && !self.sincos_args.contains(&i.args()[0]) {
                        self.sincos_args.push(i.args()[0].clone());
                    }
                }
            }
            for i in atoms(&self.newf, &|n| is_fn(n, &[Function::Acot])) {
                if i.has(&self.x) {
                    let value: Basic = match acots.iter().find(|(k, _)| *k == i) {
                        Some((_, v)) => v.clone(),
                        None => {
                            let branch_const: Basic = unique_dummy("acot_branch");
                            let new: Basic = atan(&pw(&i.args()[0], &int(-1))?)?;
                            self.backsubs
                                .push((branch_const.clone(), add(&[i.clone(), mul(&[int(-1), new.clone()])?])?));
                            let v: Basic = add(&[new, branch_const])?;
                            acots.push((i.clone(), v.clone()));
                            v
                        }
                    };
                    reps.push((i, value));
                }
            }
            if reps.is_empty() {
                return Ok(());
            }
            self.newf = xrep(&self.newf, &reps)?;
        }
    }

    fn rewrite_exps_pows(
        &mut self,
        exps: Vec<Basic>,
        pows: Vec<Basic>,
        numpows: Vec<Basic>,
        sympows: Vec<Basic>,
        log_new_extension: bool,
    ) -> PolyResult<(Vec<Basic>, Vec<Basic>, Vec<Basic>, Vec<Basic>, bool)> {
        let mut log_new_extension = log_new_extension;
        let mut ratpows_repl: Vec<(Basic, Basic)> = Vec::new();
        for i in atoms(&self.newf, &|n| n.is_pow()) {
            let (base, e) = (&i.args()[0], &i.args()[1]);
            if is_fn(base, &[Function::Exp]) && matches!(e.head(), Head::Rational { .. }) {
                ratpows_repl.push((i.clone(), exp(&mul(&[e.clone(), base.args()[0].clone()])?)?));
            }
        }
        let mut subs_map: Vec<(Basic, Basic)> = Vec::new();
        for (i, j) in ratpows_repl {
            if matches!(
                i.args()[1].head(),
                Head::Rational {
                    integer_class: true,
                    ..
                }
            ) {
                self.backsubs.push((j.clone(), i.clone()));
                subs_map.push((i, j));
            } else if self.origf.has(&i) {
                let ratio_: Basic = unique_dummy("exp_branch");
                self.backsubs
                    .push((ratio_.clone(), mul(&[i.clone(), pw(&j, &int(-1))?])?));
                subs_map.push((i, mul(&[ratio_, j])?));
            } else {
                subs_map.push((i, j));
            }
        }
        self.newf = xrep(&self.newf, &subs_map)?;
        let t_list: Vec<Basic> = self.T.clone();
        let mut exps = update_sets(&exps, &atoms(&self.newf, &|n| is_fn(n, &[Function::Exp])), &|i| {
            is_rational_function(&i.args()[0], &t_list) && has_any(&i.args()[0], &t_list)
        });
        let pows = update_sets(&pows, &atoms(&self.newf, &|n| n.is_pow()), &|i| {
            i.is_pow() && is_rational_function(&i.args()[1], &t_list) && has_any(&i.args()[1], &t_list)
        });
        let numpows = update_sets(&numpows, &pows, &|i| i.is_pow() && !has_any(&i.args()[0], &t_list));
        let rest: Vec<Basic> = pows.iter().filter(|p| !numpows.contains(p)).cloned().collect();
        let sympows = update_sets(&sympows, &rest, &|i| {
            i.is_pow()
                && is_rational_function(&i.args()[0], &t_list)
                && !matches!(
                    i.args()[1].head(),
                    Head::Rational {
                        integer_class: true,
                        ..
                    }
                )
        });
        for i in crate::core::sorting::ordered(pows.clone()) {
            if !i.is_pow() {
                continue;
            }
            let old: Basic = i.clone();
            let new: Basic = exp(&mul(&[i.args()[1].clone(), log(&i.args()[0])?])?)?;
            let newterm: Basic;
            if sympows.contains(&i) {
                if matches!(i.args()[1].head(), Head::Rational { .. }) {
                    return Err(PolyError::NotImplemented(format!(
                        "Algebraic extensions are not supported ({}).",
                        i
                    )));
                }
                let (basea, based) = frac_in_expr(&i.args()[0], &self.t, false)?;
                let found = crate::integrals::risch::prde::is_deriv_k(&basea, &based, self)?;
                let Some((_ans, u, constant)) = found else {
                    self.newf = xrep(&self.newf, &[(old.clone(), new.clone())])?;
                    self.backsubs.push((new.clone(), old.clone()));
                    log_new_extension = self.log_part(&[log(&i.args()[0])?])?;
                    let t_now: Vec<Basic> = self.T.clone();
                    exps = update_sets(&exps, &atoms(&self.newf, &|n| is_fn(n, &[Function::Exp])), &|e| {
                        is_rational_function(&e.args()[0], &t_now) && has_any(&e.args()[0], &t_now)
                    });
                    continue;
                };
                newterm = exp(&mul(&[i.args()[1].clone(), add(&[log(&constant)?, u])?])?)?;
                self.newf = xrep(&self.newf, &[(i.clone(), newterm.clone())])?;
            } else if !numpows.contains(&i) {
                continue;
            } else {
                newterm = new.clone();
            }
            self.backsubs.push((new, old.clone()));
            self.newf = xrep(&self.newf, &[(old, newterm.clone())])?;
            exps.push(newterm);
        }
        Ok((exps, pows, numpows, sympows, log_new_extension))
    }

    fn rewrite_logs(&mut self, logs: Vec<Basic>, symlogs: Vec<Basic>) -> PolyResult<(Vec<Basic>, Vec<Basic>)> {
        let found: Vec<Basic> = atoms(&self.newf, &|n| is_fn(n, &[Function::Log]));
        let t_list: Vec<Basic> = self.T.clone();
        let mut logs = update_sets(&logs, &found, &|i| {
            is_rational_function(&i.args()[0], &t_list) && has_any(&i.args()[0], &t_list)
        });
        let symlogs = update_sets(&symlogs, &found, &|i| {
            let a: &Basic = &i.args()[0];
            has_any(i, &t_list)
                && a.is_pow()
                && is_rational_function(&a.args()[0], &t_list)
                && !matches!(
                    a.args()[1].head(),
                    Head::Rational {
                        integer_class: true,
                        ..
                    }
                )
        });
        for i in crate::core::sorting::ordered(symlogs.clone()) {
            let a: Basic = i.args()[0].clone();
            if !a.is_pow() {
                continue;
            }
            let lbase: Basic = log(&a.args()[0])?;
            logs.push(lbase.clone());
            let new: Basic = mul(&[a.args()[1].clone(), lbase])?;
            self.newf = xrep(&self.newf, &[(i.clone(), new.clone())])?;
            self.backsubs.push((new, i));
        }
        let mut unique: Vec<Basic> = Vec::new();
        for l in logs {
            if !unique.contains(&l) {
                unique.push(l);
            }
        }
        unique.sort_by_cached_key(crate::core::sorting::sort_key);
        Ok((unique, symlogs))
    }

    fn auto_attrs(&mut self) {
        self.cases = self.D.iter().zip(&self.T).map(|(d, t)| get_case(d, t)).collect();
        self.level = -1;
        let n: usize = self.T.len();
        self.t = self.T[n - 1].clone();
        self.d = self.D[n - 1].clone();
        self.case = self.cases[n - 1];
    }

    fn darg(&self, arga: &TowerPoly, argd: &TowerPoly) -> PolyResult<RationalFunction> {
        let darga: TowerPoly = argd
            .mul(&derivation(arga, self, false)?)?
            .sub(&arga.mul(&derivation(argd, self, false)?)?)?;
        let dargd: TowerPoly = argd.pow(2);
        darga.as_element()?.div(&dargd.as_element()?)
    }

    fn exp_part(&mut self, exps: &[Basic]) -> PolyResult<Option<bool>> {
        let mut new_extension: bool = false;
        let expargs: Vec<Basic> = exps.iter().map(|i| i.args()[0].clone()).collect();
        for (arg, mut others) in integer_powers(&expargs)? {
            others.sort_by_key(|(_, k)| k.as_rational().cloned().unwrap_or(Rat::ZERO));
            let (arga, argd) = frac_in_expr(&arg, &self.t, false)?;
            if let Some((mut ans, mut u, mut n, mut constant)) =
                crate::integrals::risch::prde::is_log_deriv_k_t_radical(&arga, &argd, self)?
            {
                if n == -1 {
                    n = 1;
                    u = pw(&u, &int(-1))?;
                    constant = mul(&[int(-1), constant])?;
                    ans = ans
                        .into_iter()
                        .map(|(i, j)| Ok((i, mul(&[int(-1), j])?)))
                        .collect::<PolyResult<_>>()?;
                }
                let _ = u;
                if n == 1 {
                    let product = |c: &Basic| -> R {
                        let mut v: Vec<Basic> = vec![exp(c)?];
                        for (u_, power) in &ans {
                            v.push(pw(u_, power)?);
                        }
                        mul(&v)
                    };
                    self.newf = xrep(&self.newf, &[(exp(&arg)?, product(&constant)?)])?;
                    let mut rule: Vec<(Basic, Basic)> = Vec::new();
                    for (exparg, p) in &others {
                        rule.push((
                            exp(&mul(&[p.clone(), exparg.clone()])?)?,
                            product(&mul(&[constant.clone(), p.clone()])?)?,
                        ));
                    }
                    self.newf = xrep(&self.newf, &rule)?;
                    continue;
                }
                if constant != Basic::zero() || ans.len() > 1 {
                    let mut factors: Vec<Basic> = Vec::new();
                    for (term, power) in &ans {
                        factors.push(pw(
                            term,
                            &mul(&[power.clone(), rat(&(Rat::ONE / Rat::from(Int::from(n))))])?,
                        )?);
                    }
                    let rad: Basic = mul(&factors)?;
                    let mut rule: Vec<(Basic, Basic)> = Vec::new();
                    for (exparg, p) in &others {
                        rule.push((
                            exp(&mul(&[p.clone(), exparg.clone()])?)?,
                            mul(&[exp(&mul(&[constant.clone(), p.clone()])?)?, rad.clone()])?,
                        ));
                    }
                    self.newf = xrep(&self.newf, &rule)?;
                    let mut back: Vec<(Basic, Basic)> = Vec::new();
                    for (t, f) in self.T.iter().rev().zip(self.Tfuncs.iter().rev()) {
                        back.push((t.clone(), f.apply(&self.x)?));
                    }
                    self.newf = xrep(&self.newf, &back)?;
                    return Ok(None);
                }
                return Err(PolyError::NotImplemented(
                    "Cannot integrate over algebraic extensions.".to_string(),
                ));
            }
            let darg: RationalFunction = self.darg(&arga, &argd)?;
            let t_new: Basic = self.next_t();
            let D: TowerPoly = TowerPoly::from_element_strict(&darg, &t_new)?.mul(&TowerPoly::monomial(
                &t_new,
                1,
                &RationalFunction::one(),
            ))?;
            self.push(&arg, Extension::Exp, D, t_new, Function::Exp)?;
            let mut rule: Vec<(Basic, Basic)> = Vec::new();
            for (exparg, p) in &others {
                rule.push((exp(exparg)?, pw(&self.t, p)?));
            }
            self.newf = xrep(&self.newf, &rule)?;
            new_extension = true;
        }
        Ok(Some(new_extension))
    }

    fn push(&mut self, arg: &Basic, ext: Extension, D: TowerPoly, t_new: Basic, func: Function) -> PolyResult<()> {
        self.t = t_new.clone();
        self.T.push(t_new);
        self.extargs.push(Some(arg.clone()));
        self.exts.push(Some(ext));
        self.D.push(D);
        let i: Basic = if self.dummy {
            unique_dummy("i")
        } else {
            crate::core::symbol::Symbol::new("i", &[]).0
        };
        let inner: Basic = crate::core::subs::subs(arg, &self.x, &i)?;
        let body: Basic = crate::functions::apply_function(func, &[inner])?;
        self.Tfuncs.push(TFunc { var: i, body });
        Ok(())
    }

    fn concrete(&self) -> PolyResult<Vec<(Basic, Basic)>> {
        let mut out: Vec<(Basic, Basic)> = Vec::new();
        for (t, f) in self.T.iter().rev().zip(self.Tfuncs.iter().rev()) {
            out.push((t.clone(), f.apply(&self.x)?));
        }
        Ok(out)
    }

    fn log_part(&mut self, logs: &[Basic]) -> PolyResult<bool> {
        let mut new_extension: bool = false;
        let logargs: Vec<Basic> = logs.iter().map(|i| i.args()[0].clone()).collect();
        for arg in crate::core::sorting::ordered(logargs) {
            let (arga, argd) = frac_in_expr(&arg, &self.t, false)?;
            if let Some((_ans, u, constant)) = crate::integrals::risch::prde::is_deriv_k(&arga, &argd, self)? {
                let is_log_generator: bool = u.is_symbol()
                    && u != self.x
                    && self.T.contains(&u)
                    && crate::core::expr::is(&constant, crate::core::assumptions::Predicate::positive) == Some(true)
                    && self.exts[self.T.iter().position(|t| *t == u).expect("a generator") - 1] == Some(Extension::Log);
                if is_log_generator {
                    self.newf = xrep(&self.newf, &[(log(&arg)?, add(&[log(&constant)?, u])?)])?;
                    continue;
                }
                let branch_const: Basic = unique_dummy("log_branch");
                let concrete: Vec<(Basic, Basic)> = self.concrete()?;
                let difference: Basic = add(&[
                    log(&subs_seq(&arg, &concrete)?)?,
                    mul(&[int(-1), subs_seq(&u, &concrete)?])?,
                ])?;
                self.backsubs
                    .push((branch_const.clone(), subs_seq(&difference, &self.backsubs)?));
                self.newf = xrep(&self.newf, &[(log(&arg)?, add(&[branch_const, u])?)])?;
                continue;
            }
            let darg: RationalFunction = self.darg(&arga, &argd)?;
            let t_new: Basic = self.next_t();
            let D: TowerPoly = TowerPoly::from_element_strict(&darg.div(&element(&arg)?)?, &t_new)?;
            self.push(&arg, Extension::Log, D, t_new, Function::Log)?;
            self.newf = xrep(&self.newf, &[(log(&arg)?, self.t.clone())])?;
            new_extension = true;
        }
        Ok(new_extension)
    }

    fn tan_part(&mut self, tans: &[Basic]) -> PolyResult<bool> {
        let mut new_extension: bool = false;
        let mut split: Vec<(Basic, Vec<(Basic, Basic)>)> = Vec::new();
        for ti in tans {
            let (c, g) = independent_terms(&ti.args()[0], &self.T)?;
            match split.iter_mut().find(|(k, _)| *k == g) {
                Some(entry) => entry.1.push((ti.clone(), c)),
                None => split.push((g, vec![(ti.clone(), c)])),
            }
        }
        let keys: Vec<Basic> = split.iter().map(|(k, _)| k.clone()).collect();
        for (g, others) in integer_powers(&keys)? {
            let (ga, gd) = frac_in_expr(&g, &self.t, false)?;
            let tang: Basic = match crate::integrals::risch::prde::is_log_deriv_k_t_radical_tan(&ga, &gd, self)? {
                None => {
                    let darg: RationalFunction = self.darg(&ga, &gd)?;
                    let t_new: Basic = self.next_t();
                    let one: RationalFunction = RationalFunction::one();
                    let zero: RationalFunction = RationalFunction::zero();
                    let D: TowerPoly = TowerPoly::from_element_strict(&darg, &t_new)?
                        .mul(&TowerPoly::from_list(&[one.clone(), zero, one], &t_new))?;
                    self.push(&g, Extension::Tan, D, t_new, Function::Tan)?;
                    new_extension = true;
                    self.t.clone()
                }
                Some((ans, _u, n, constant)) => {
                    if n != 1 {
                        return Err(PolyError::NotImplemented(format!(
                            "Cannot integrate over algebraic extensions (tan({}) is algebraic of degree {} over the tower).",
                            g, n
                        )));
                    }
                    let mut tanterms: Vec<(Basic, Basic)> = Vec::new();
                    for (term, r) in ans {
                        if term.is_symbol() && self.T.contains(&term) {
                            let index: usize = self.T.iter().position(|t| *t == term).expect("a generator");
                            let Some(found) = self.extargs[index - 1].clone() else {
                                return Err(PolyError::NotImplemented(
                                    "a tangent of a generator without argument".to_string(),
                                ));
                            };
                            tanterms.push((found, r));
                        } else {
                            let index: usize = self
                                .extargs
                                .iter()
                                .position(|a| a.as_ref() == Some(&term))
                                .ok_or_else(|| PolyError::Value(format!("{} is not an argument of the tower", term)))?;
                            tanterms.push((self.T[index + 1].clone(), r));
                        }
                    }
                    tan_combination(&tanterms, &constant)?
                }
            };
            for (gi, p) in others {
                let entries: Vec<(Basic, Basic)> = split
                    .iter()
                    .find(|(k, _)| *k == gi)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default();
                for (tanatom, c) in entries {
                    let value: Basic = tan_combination(&[(tang.clone(), p.clone())], &c)?;
                    self.newf = xrep(&self.newf, &[(tanatom, value)])?;
                }
            }
        }
        Ok(new_extension)
    }

    fn atan_part(&mut self, atans: &[Basic]) -> PolyResult<bool> {
        let mut new_extension: bool = false;
        let mut args: Vec<Basic> = Vec::new();
        for i in atans {
            if !args.contains(&i.args()[0]) {
                args.push(i.args()[0].clone());
            }
        }
        for arg in crate::core::sorting::ordered(args) {
            let (arga, argd) = frac_in_expr(&arg, &self.t, false)?;
            if let Some((_ans, u)) = crate::integrals::risch::prde::is_deriv_k_atan(&arga, &argd, self)? {
                let branch_const: Basic = unique_dummy("atan_branch");
                let concrete: Vec<(Basic, Basic)> = self.concrete()?;
                let difference: Basic = add(&[
                    atan(&subs_seq(&arg, &concrete)?)?,
                    mul(&[int(-1), subs_seq(&u, &concrete)?])?,
                ])?;
                self.backsubs
                    .push((branch_const.clone(), subs_seq(&difference, &self.backsubs)?));
                self.newf = xrep(&self.newf, &[(atan(&arg)?, add(&[branch_const, u])?)])?;
                continue;
            }
            let darg: RationalFunction = self.darg(&arga, &argd)?;
            let t_new: Basic = self.next_t();
            let e: RationalFunction = element(&arg)?;
            let D: TowerPoly =
                TowerPoly::from_element_strict(&darg.div(&e.mul(&e)?.add(&RationalFunction::one())?)?, &t_new)?;
            self.push(&arg, Extension::Atan, D, t_new, Function::Atan)?;
            self.newf = xrep(&self.newf, &[(atan(&arg)?, self.t.clone())])?;
            new_extension = true;
        }
        Ok(new_extension)
    }

    /// `reset`: the tower back to `x` alone.
    pub fn reset(&mut self) {
        self.t = self.x.clone();
        self.T = vec![self.x.clone()];
        self.D = vec![TowerPoly::one(&self.x)];
        self.level = -1;
        self.exts = Vec::new();
        self.extargs = Vec::new();
        self.ts = 0;
        self.backsubs = Vec::new();
        self.Tfuncs = Vec::new();
        self.newf = self.f.clone();
    }

    /// The indices (from 1) of the extensions of a type (`indices`).
    pub fn indices(&self, extension: Extension) -> Vec<usize> {
        self.exts
            .iter()
            .enumerate()
            .filter(|(_, e)| **e == Some(extension))
            .map(|(i, _)| i + 1)
            .collect()
    }

    fn at(&self, level: i64) -> usize {
        (self.T.len() as i64 + level) as usize
    }

    /// `increment_level`.
    pub fn increment_level(&mut self) -> PolyResult<()> {
        if self.level >= -1 {
            return Err(PolyError::Value(
                "The level of the differential extension cannot be incremented any further.".to_string(),
            ));
        }
        self.level += 1;
        let i: usize = self.at(self.level);
        self.t = self.T[i].clone();
        self.d = self.D[i].clone();
        self.case = self.cases[i];
        Ok(())
    }

    /// `decrement_level`.
    pub fn decrement_level(&mut self) -> PolyResult<()> {
        if self.level <= -(self.T.len() as i64) {
            return Err(PolyError::Value(
                "The level of the differential extension cannot be decremented any further.".to_string(),
            ));
        }
        self.level -= 1;
        let i: usize = self.at(self.level);
        self.t = self.T[i].clone();
        self.d = self.D[i].clone();
        self.case = self.cases[i];
        Ok(())
    }

    /// The derivation of the `i`-th generator as an element of the field.
    pub fn derivation_of(&self, i: usize) -> PolyResult<RationalFunction> {
        self.D[i].as_element()
    }

    /// The back substitutions of the generators (`backsubstitutions`).
    pub fn backsubstitutions(&self) -> PolyResult<Vec<(Basic, Basic)>> {
        self.concrete()
    }
}

/// `e.as_independent(*T, as_Add=True)`: the terms free of the generators
/// and the others.
fn independent_terms(e: &Basic, t: &[Basic]) -> PolyResult<(Basic, Basic)> {
    let terms: Vec<Basic> = crate::core::expr::terms_of(e);
    let free: Vec<Basic> = terms.iter().filter(|a| !has_any(a, t)).cloned().collect();
    let dependent: Vec<Basic> = terms.iter().filter(|a| has_any(a, t)).cloned().collect();
    Ok((add(&free)?, add(&dependent)?))
}

/// `tan(sum r_i y_i + c)` expanded, with `tan(y_i)` the given terms
/// (`_tan_combination`).
fn tan_combination(tanterms: &[(Basic, Basic)], c: &Basic) -> R {
    let ys: Vec<Basic> = tanterms.iter().map(|_| unique_dummy("y")).collect();
    let cd: Basic = unique_dummy("c");
    let mut parts: Vec<Basic> = Vec::new();
    for ((_, r), y) in tanterms.iter().zip(&ys) {
        parts.push(mul(&[r.clone(), y.clone()])?);
    }
    let mut arg: Basic = add(&parts)?;
    if *c != Basic::zero() {
        arg = add(&[arg, cd.clone()])?;
    }
    let mut hints = crate::core::function::Hints::DEFAULT;
    hints.mul = false;
    hints.multinomial = false;
    hints.power_base = false;
    hints.power_exp = false;
    hints.log = false;
    hints.trig = true;
    let new: Basic = crate::core::function::expand_with(&tan(&arg)?, true, hints)?;
    let mut reps: Vec<(Basic, Basic)> = Vec::new();
    for ((i, _), y) in tanterms.iter().zip(&ys) {
        reps.push((tan(y)?, i.clone()));
    }
    reps.push((tan(&cd)?, tan(c)?));
    xrep(&new, &reps)
}

/// `s, t` with `s a + t b == c` and `deg s < deg b` (`gcdex_diophantine`).
pub fn gcdex_diophantine(a: &TowerPoly, b: &TowerPoly, c: &TowerPoly) -> PolyResult<(TowerPoly, TowerPoly)> {
    let (mut s, g) = a.half_gcdex(b)?;
    s = s.mul(&c.exquo(&g)?)?;
    if !s.is_zero() && s.degree(None) >= b.degree(None) {
        s = s.div(b)?.1;
    }
    let t: TowerPoly = c.sub(&s.mul(a)?)?.exquo(b)?;
    Ok((s, t))
}

/// `f` as a fraction of polynomials in `t` from an element (`frac_in`).
pub fn frac_in(g: &RationalFunction, t: &Basic, cancel: bool) -> PolyResult<(TowerPoly, TowerPoly)> {
    let fa = TowerPoly::from_element(&g.numerator()?, t)?;
    let fd = TowerPoly::from_element(&g.denominator()?, t)?;
    let (Some(fa), Some(fd)) = (fa, fd) else {
        return Err(PolyError::Value(format!(
            "Could not turn {} into a fraction in {}.",
            g, t
        )));
    };
    if cancel {
        return fa.cancel(&fd);
    }
    Ok((fa, fd))
}

/// [`frac_in`] of an expression.
pub fn frac_in_expr(f: &Basic, t: &Basic, cancel: bool) -> PolyResult<(TowerPoly, TowerPoly)> {
    let g: RationalFunction = element(f)?;
    if g.gens.iter().any(|v| v != t && v.has(t)) {
        return Err(PolyError::Value(format!(
            "Could not turn {} into a fraction in {}.",
            f, t
        )));
    }
    frac_in(&g, t, cancel)
}

/// [`frac_in`] of a fraction of polynomials.
pub fn frac_in_pair(fa: &TowerPoly, fd: &TowerPoly, t: &Basic, cancel: bool) -> PolyResult<(TowerPoly, TowerPoly)> {
    frac_in(&fa.as_element()?.div(&fd.as_element()?)?, t, cancel)
}

/// An element of `K[t, 1/t]` (`LaurentPolynomial`).
#[derive(Clone, Debug, PartialEq)]
pub struct LaurentPolynomial {
    pub coefficients: BTreeMap<i64, RationalFunction>,
    pub gen: Basic,
}

impl LaurentPolynomial {
    pub fn new(coefficients: BTreeMap<i64, RationalFunction>, gen: &Basic) -> Self {
        LaurentPolynomial {
            coefficients: coefficients.into_iter().filter(|(_, c)| !c.is_zero()).collect(),
            gen: gen.clone(),
        }
    }
    pub fn nth(&self, i: i64) -> RationalFunction {
        self.coefficients
            .get(&i)
            .cloned()
            .unwrap_or_else(RationalFunction::zero)
    }
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }
    pub fn degree(&self) -> i64 {
        self.coefficients.keys().next_back().copied().unwrap_or(0)
    }
    pub fn low_degree(&self) -> i64 {
        self.coefficients.keys().next().copied().unwrap_or(0)
    }
    pub fn as_element(&self) -> PolyResult<RationalFunction> {
        let t: RationalFunction = RationalFunction::generator(&self.gen);
        let mut total: RationalFunction = RationalFunction::zero();
        for (i, c) in &self.coefficients {
            total = total.add(&c.mul(&t.pow(*i)?)?)?;
        }
        Ok(total)
    }
    pub fn as_expr(&self) -> R {
        self.as_element()?.as_expr()
    }
}

/// `p` as an element of `K[t, 1/t]` (`as_poly_1t`).
pub fn as_poly_1t(p: &RationalFunction, t: &Basic) -> PolyResult<LaurentPolynomial> {
    let (pa, pd) = frac_in(p, t, true)?;
    if !pd.is_monomial() {
        return Err(crate::error::polys_error(
            crate::error::PolysErrorKind::PolynomialError,
            format!("{} is not an element of K[{}, 1/{}].", p, t, t),
        ));
    }
    let k: i64 = pd.degree(None);
    let lc: RationalFunction = pd.LC().invert()?;
    let mut coefficients: BTreeMap<i64, RationalFunction> = BTreeMap::new();
    for (i, c) in pa.terms() {
        coefficients.insert(i as i64 - k, c.mul(&lc)?);
    }
    Ok(LaurentPolynomial::new(coefficients, t))
}

fn top(de: &DifferentialExtension, coefficient_d: bool) -> usize {
    let mut n: i64 = de.T.len() as i64 + de.level + 1;
    if coefficient_d {
        n -= 1;
    }
    n.max(0) as usize
}

/// The derivation of an element of the field (`derivation_element`).
pub fn derivation_element(
    a: &RationalFunction,
    de: &DifferentialExtension,
    coefficient_d: bool,
) -> PolyResult<RationalFunction> {
    let mut result: RationalFunction = RationalFunction::zero();
    for i in 0..top(de, coefficient_d) {
        let v: &Basic = &de.T[i];
        if !a.has(std::slice::from_ref(v)) {
            continue;
        }
        result = result.add(&de.derivation_of(i)?.mul(&a.diff(v)?)?)?;
    }
    Ok(result)
}

/// The derivation of a polynomial (`derivation`).
pub fn derivation(p: &TowerPoly, de: &DifferentialExtension, coefficient_d: bool) -> PolyResult<TowerPoly> {
    if coefficient_d && de.level <= -(de.T.len() as i64) {
        return Ok(TowerPoly::zero(&p.gen));
    }
    let n: usize = top(de, coefficient_d);
    let gen: Basic = p.gen.clone();
    let rep: Vec<RationalFunction> = p
        .rep
        .iter()
        .map(|c| {
            if c.is_zero() {
                Ok(c.clone())
            } else {
                derivation_element(c, de, coefficient_d)
            }
        })
        .collect::<PolyResult<_>>()?;
    let mut result: TowerPoly = TowerPoly::new(rep, &gen);
    if de.T[..n].contains(&gen) && p.degree(None) > 0 {
        let i: usize = de.T.iter().position(|t| *t == gen).expect("a generator");
        let mut d: TowerPoly = de.D[i].clone();
        if d.gen != gen {
            d = d.as_poly_strict(&gen)?;
        }
        result = result.add(&p.diff(None)?.mul(&d)?)?;
    }
    Ok(result)
}

/// The case of a derivation (`get_case`).
pub fn get_case(d: &TowerPoly, t: &Basic) -> Case {
    if !d.has(std::slice::from_ref(t)) {
        return if d.is_one() { Case::Base } else { Case::Primitive };
    }
    let d: TowerPoly = if d.gen != *t {
        d.as_poly_strict(t).unwrap_or_else(|_| d.clone())
    } else {
        d.clone()
    };
    let one: RationalFunction = RationalFunction::one();
    let zero: RationalFunction = RationalFunction::zero();
    if d.rem(&TowerPoly::monomial(t, 1, &one)).is_ok_and(|r| r.is_zero()) {
        return Case::Exp;
    }
    if d.rem(&TowerPoly::from_list(&[one.clone(), zero, one], t))
        .is_ok_and(|r| r.is_zero())
    {
        return Case::Tan;
    }
    if d.degree(Some(t)) > 1 {
        return Case::OtherNonlinear;
    }
    Case::OtherLinear
}

/// `p` as a polynomial in the current generator (`_in_t`).
pub fn in_t(p: &TowerPoly, de: &DifferentialExtension) -> PolyResult<TowerPoly> {
    if p.gen == de.t {
        Ok(p.clone())
    } else {
        p.as_poly_strict(&de.t)
    }
}

/// `(p_n, p_s)`: the normal and the special parts of `p` (`splitfactor`).
pub fn splitfactor(
    p: &TowerPoly,
    de: &DifferentialExtension,
    coefficient_d: bool,
) -> PolyResult<(TowerPoly, TowerPoly)> {
    let (p, gen): (TowerPoly, Basic) = if !de.T.contains(&p.gen) && !p.is_ground() {
        (p.clone(), p.gen.clone())
    } else {
        (in_t(p, de)?, de.t.clone())
    };
    let one: TowerPoly = TowerPoly::one(&gen);
    if p.is_zero() || p.degree(None) <= 0 {
        return Ok((p, one));
    }
    let dp: TowerPoly = derivation(&p, de, coefficient_d)?;
    if dp.is_zero() {
        return Ok((p, one));
    }
    let h: TowerPoly = p.gcd(&dp)?;
    let g: TowerPoly = p.gcd(&p.diff(None)?)?;
    let s: TowerPoly = h.exquo(&g)?;
    if s.degree(None) == 0 {
        return Ok((p, one));
    }
    let (q0, q1) = splitfactor(&p.exquo(&s)?, de, coefficient_d)?;
    Ok((q0, q1.mul(&s)?))
}

/// The square-free normal and special parts with their multiplicities
/// (`splitfactor_sqf`).
pub fn splitfactor_sqf(
    p: &TowerPoly,
    de: &DifferentialExtension,
    coefficient_d: bool,
) -> PolyResult<(Vec<(TowerPoly, usize)>, Vec<(TowerPoly, usize)>)> {
    let mut special: Vec<(TowerPoly, usize)> = Vec::new();
    let mut normal: Vec<(TowerPoly, usize)> = Vec::new();
    if p.is_zero() {
        return Ok((vec![(p.clone(), 1)], Vec::new()));
    }
    for (pi, i) in p.sqf_list_include() {
        let si: TowerPoly = pi.gcd(&derivation(&pi, de, coefficient_d)?)?;
        let ni: TowerPoly = pi.exquo(&si)?;
        if !si.is_one() {
            special.push((si, i));
        }
        if !ni.is_one() {
            normal.push((ni, i));
        }
    }
    Ok((normal, special))
}

/// `a/d = q + b/d_s + c/d_n` (`canonical_representation`).
pub fn canonical_representation(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, (TowerPoly, TowerPoly), (TowerPoly, TowerPoly))> {
    let l: RationalFunction = d.LC().invert()?;
    let (a, d) = (a.mul_ground(&l), d.mul_ground(&l));
    let (q, r) = a.div(&d)?;
    let (dn, ds) = splitfactor(&d, de, false)?;
    let (b, c) = gcdex_diophantine(&in_t(&dn, de)?, &in_t(&ds, de)?, &in_t(&r, de)?)?;
    Ok((q, (b, ds), (c, dn)))
}

/// Hermite's reduction (`hermite_reduce`): `((ga, gd), (r, d), (rra, rrd))`
/// with `a/d = D(ga/gd) + r/d + rra/rrd`, `d` square-free normal and
/// `rra/rrd` reduced.
pub fn hermite_reduce(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<((TowerPoly, TowerPoly), (TowerPoly, TowerPoly), (TowerPoly, TowerPoly))> {
    let (a, d) = (in_t(a, de)?, in_t(d, de)?);
    let l: RationalFunction = d.LC().invert()?;
    let (a, d) = (a.mul_ground(&l), d.mul_ground(&l));
    let (fp, fs, fn_) = canonical_representation(&a, &d, de)?;
    let (a, d) = fn_;
    let l: RationalFunction = d.LC().invert()?;
    let (mut a, d) = (a.mul_ground(&l), d.mul_ground(&l));
    let mut ga: TowerPoly = TowerPoly::zero(&de.t);
    let mut gd: TowerPoly = TowerPoly::one(&de.t);
    let dd: TowerPoly = derivation(&d, de, false)?;
    let mut dm: TowerPoly = d.gcd(&dd)?;
    let (ds, _) = d.div(&dm)?;
    while dm.degree(Some(&de.t)) > 0 {
        crate::timeout::check()?;
        let ddm: TowerPoly = derivation(&dm, de, false)?;
        let dm2: TowerPoly = dm.gcd(&ddm)?;
        let (dms, _) = dm.div(&dm2)?;
        let ds_ddm: TowerPoly = ds.mul(&ddm)?;
        let (ds_ddm_dm, _) = ds_ddm.div(&dm)?;
        let (b, c) = gcdex_diophantine(&ds_ddm_dm.neg(), &dms, &a)?;
        let db: TowerPoly = derivation(&b, de, false)?;
        let (ds_dms, _) = ds.div(&dms)?;
        a = c.sub(&db.mul(&ds_dms)?)?;
        ga = ga.mul(&dm)?.add(&b.mul(&gd)?)?;
        gd = gd.mul(&dm)?;
        let (x, y) = ga.cancel(&gd)?;
        ga = x;
        gd = y;
        dm = dm2;
    }
    let (q, r) = a.div(&ds)?;
    let (ga, gd) = ga.cancel(&gd)?;
    let (r, d) = r.cancel(&ds)?;
    let rra: TowerPoly = q.mul(&fs.1)?.add(&fp.mul(&fs.1)?)?.add(&fs.0)?;
    let (rra, rrd) = rra.cancel(&fs.1)?;
    Ok(((ga, gd), (r, d), (rra, rrd)))
}

/// `(q, r)` with `p = Dq + r` and `deg r < deg Dt` (`polynomial_reduce`).
pub fn polynomial_reduce(p: &TowerPoly, de: &DifferentialExtension) -> PolyResult<(TowerPoly, TowerPoly)> {
    let mut p: TowerPoly = in_t(p, de)?;
    let mut q: TowerPoly = TowerPoly::zero(&de.t);
    let t: &Basic = &de.t;
    while p.degree(Some(t)) >= de.d.degree(Some(t)) {
        crate::timeout::check()?;
        let m: i64 = p.degree(Some(t)) - de.d.degree(Some(t)) + 1;
        let c: RationalFunction = p
            .LC()
            .div(&de.d.LC().mul(&RationalFunction::from_int(&Int::from(m)))?)?;
        let q0: TowerPoly = TowerPoly::monomial(t, m as usize, &c);
        q = q.add(&q0)?;
        p = p.sub(&derivation(&q0, de, false)?)?;
    }
    Ok((q, p))
}

fn unique_symbols(name: &str, n: usize) -> Vec<Basic> {
    (0..n).map(|i| unique_dummy(&format!("{}{}", name, i))).collect()
}

/// The principal parts of the Laurent series of `a/d` at the roots of a
/// normal factor `F` of multiplicity `n` (`laurent_series`).
pub fn laurent_series(
    a: &TowerPoly,
    d: &TowerPoly,
    big_f: &TowerPoly,
    n: usize,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly, Vec<TowerPoly>)> {
    let (a, d, big_f) = (in_t(a, de)?, in_t(d, de)?, in_t(big_f, de)?);
    if big_f.degree(None) == 0 {
        return Ok((TowerPoly::zero(&de.t), TowerPoly::one(&de.t), Vec::new()));
    }
    let mut z_list: Vec<Basic> = unique_symbols("z", n);
    let z: Basic = unique_dummy("z");
    z_list.insert(0, z.clone());
    let zf: RationalFunction = RationalFunction::generator(&z);
    let mut delta_a: TowerPoly = TowerPoly::zero(&de.t);
    let mut delta_d: TowerPoly = TowerPoly::one(&de.t);
    let e: TowerPoly = d.quo(&big_f.pow(n))?;
    let (mut ha, mut hd) = (a.clone(), e.mul_ground(&zf.pow(n as i64)?));
    let df: TowerPoly = derivation(&big_f, de, false)?;
    if big_f.gcd(&df)?.degree(Some(&de.t)) > 0 {
        return Err(PolyError::NotImplemented(
            "laurent_series() cannot compute the principal parts at special primes (gcd(F, DF) != 1).".to_string(),
        ));
    }
    let one: TowerPoly = TowerPoly::one(&de.t);
    let (b, _) = gcdex_diophantine(&e, &big_f, &one)?;
    let (c, _) = gcdex_diophantine(&df, &big_f, &one)?;
    let mut f_store: TowerPoly = big_f.clone();
    let mut v: Vec<RationalFunction> = Vec::new();
    let mut de_d_list: Vec<TowerPoly> = Vec::new();
    let mut h_list: Vec<TowerPoly> = Vec::new();
    for j in 0..n {
        f_store = derivation(&f_store, de, false)?;
        v.push(
            f_store
                .as_element()?
                .div(&RationalFunction::from_int(&Int::from(j as i64 + 1)))?,
        );
        de_d_list.push(TowerPoly::ground(
            &RationalFunction::generator(&z_list[j + 1]),
            &z_list[j],
        )?);
    }
    let de_new: DifferentialExtension =
        DifferentialExtension::manual(de_d_list, Vec::new(), Vec::new(), None, None, None);
    for j in 0..n {
        crate::timeout::check()?;
        let z_e_ha: TowerPoly = e.pow(j + 1).mul(&ha)?.mul_ground(&zf.pow((n + j) as i64)?);
        let (pa, pd) = z_e_ha.cancel(&hd)?;
        let mut q: TowerPoly = pa.quo(&pd)?;
        for i in 0..=j {
            q = q.subs(&z_list[i], &v[i])?;
        }
        let dha: TowerPoly = hd
            .mul(&derivation(&ha, de, false)?)?
            .sub(&ha.mul(&derivation(&hd, de, false)?)?)?
            .add(&hd.mul(&derivation(&ha, &de_new, false)?)?)?
            .sub(&ha.mul(&derivation(&hd, &de_new, false)?)?)?;
        let dhd: TowerPoly = hd
            .pow(2)
            .mul_ground(&RationalFunction::from_int(&Int::from(j as i64 + 1)));
        ha = dha;
        hd = dhd;
        let qbc: TowerPoly = q.mul(&b.pow(1 + j))?.mul(&c.pow(n + j))?;
        h_list.push(qbc.clone());
        let (ff, _) = big_f.div(&big_f.gcd(&q)?)?;
        let (f_stara, f_stard) = frac_in_pair(&ff, &TowerPoly::one(&ff.gen), &de.t, false)?;
        if f_stara.degree(Some(&de.t)) - f_stard.degree(Some(&de.t)) > 0 {
            let h: TowerPoly = qbc.mul(&f_stard)?.rem(&f_stara)?;
            for alpha in super::rde::rational_roots(&f_stara)? {
                let pa: TowerPoly = TowerPoly::from_list(&[RationalFunction::one(), alpha.neg()], &de.t).pow(n - j);
                delta_a = delta_a.mul(&pa)?.add(&delta_d.mul_ground(&h.eval(&alpha)))?;
                delta_d = delta_d.mul(&pa)?;
            }
        }
    }
    let (delta_a, delta_d) = delta_a.cancel(&delta_d)?;
    Ok((delta_a, delta_d, h_list))
}

/// Whether `a/d` is the derivative of an element of the field
/// (`recognize_derivative`).
pub fn recognize_derivative(a: &TowerPoly, d: &TowerPoly, de: &DifferentialExtension) -> PolyResult<bool> {
    let (a, d) = (in_t(a, de)?, in_t(d, de)?);
    let (a, d) = a.cancel(&d)?;
    let (_, r) = a.div(&d)?;
    let (np, sp) = splitfactor_sqf(&d, de, true)?;
    let normal: Vec<(TowerPoly, usize)> = np.into_iter().filter(|(s, _)| s.degree(Some(&de.t)) > 0).collect();
    if normal.iter().any(|(_, n)| *n == 1) {
        return Ok(false);
    }
    let mut undecidable_special: Vec<TowerPoly> = Vec::new();
    for (s, n) in sp {
        let mut s: TowerPoly = in_t(&s, de)?;
        let s_special: TowerPoly = s.gcd(&derivation(&s, de, false)?)?;
        if s_special.degree(Some(&de.t)) > 0 {
            undecidable_special.push(s_special.clone());
            s = s.quo(&s_special)?;
            if s.degree(Some(&de.t)) == 0 {
                continue;
            }
        }
        let (_, _, h) = laurent_series(&r, &d, &s, n, de)?;
        if !h[h.len() - 1].rem(&s)?.is_zero() {
            return Ok(false);
        }
    }
    if !normal.is_empty() {
        return Err(PolyError::NotImplemented(
            "recognize_derivative() cannot decide residue vanishing at nonconstant poles of order greater than 1."
                .to_string(),
        ));
    }
    if !undecidable_special.is_empty() {
        return Err(PolyError::NotImplemented(
            "recognize_derivative() cannot decide residue vanishing at special primes.".to_string(),
        ));
    }
    Ok(true)
}

/// Whether `a/d` is the logarithmic derivative of an element of the field
/// (`recognize_log_derivative`).
pub fn recognize_log_derivative(a: &TowerPoly, d: &TowerPoly, de: &DifferentialExtension) -> PolyResult<bool> {
    let z: Basic = unique_dummy("z");
    let (a, d) = (in_t(a, de)?, in_t(d, de)?);
    let (a, d) = a.cancel(&d)?;
    let (_, a) = a.div(&d)?;
    let pz: RationalFunction = RationalFunction::generator(&z);
    let dd: TowerPoly = derivation(&d, de, false)?;
    if d.gcd(&dd)?.degree(Some(&de.t)) > 0 {
        return Ok(false);
    }
    let q: TowerPoly = a.sub(&dd.mul_ground(&pz))?;
    let (res, _) = d.resultant_prs(&q)?;
    let r: TowerPoly = TowerPoly::from_element_strict(&res, &z)?;
    let (np, sp) = splitfactor_sqf(&r, de, true)?;
    if np.iter().any(|(s, _)| s.degree(None) > 0) {
        return Ok(false);
    }
    for (s, _) in sp {
        let found: Vec<RationalFunction> = super::rde::rational_roots(&s)?;
        if found.len() as i64 != s.degree(None) || !found.iter().all(|j| j.is_integer()) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The Lazard-Rioboo-Rothstein-Trager resultant reduction
/// (`residue_reduce`): the pairs `(s, h)` of the logarithmic part
/// `sum_{s(a) = 0} a log(h(a, t))` and whether the rest is in the field.
pub fn residue_reduce(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &DifferentialExtension,
    z: Option<&Basic>,
    invert: bool,
) -> PolyResult<(Vec<(TowerPoly, TowerPoly)>, bool)> {
    let z: Basic = z.cloned().unwrap_or_else(|| unique_dummy("z"));
    let (a, d) = (in_t(a, de)?, in_t(d, de)?);
    let (a, d) = a.cancel(&d)?;
    let lc: RationalFunction = d.LC().invert()?;
    let (a, d) = (a.mul_ground(&lc), d.mul_ground(&lc));
    if a.is_zero() {
        return Ok((Vec::new(), true));
    }
    let (_, a) = a.div(&d)?;
    let pz: RationalFunction = RationalFunction::generator(&z);
    let dd: TowerPoly = derivation(&d, de, false)?;
    let q: TowerPoly = a.sub(&dd.mul_ground(&pz))?;
    let (res, prs) = if dd.degree(Some(&de.t)) <= d.degree(Some(&de.t)) {
        d.resultant_prs(&q)?
    } else {
        q.resultant_prs(&d)?
    };
    let mut r_map: BTreeMap<i64, TowerPoly> = BTreeMap::new();
    for element in prs {
        r_map.insert(element.degree(None), element);
    }
    let mut h_list: Vec<(TowerPoly, TowerPoly)> = Vec::new();
    let r: TowerPoly = TowerPoly::from_element_strict(&res, &z)?;
    let (np, sp) = splitfactor_sqf(&r, de, true)?;
    for (s, i) in sp {
        if i as i64 == d.degree(Some(&de.t)) {
            h_list.push((s.monic()?, d.clone()));
            continue;
        }
        let Some(found) = r_map.get(&(i as i64)) else { continue };
        let mut h: TowerPoly = found.clone();
        let s: TowerPoly = s.monic()?;
        let h_lc: TowerPoly = TowerPoly::from_element_strict(&h.LC(), &z)?;
        for (a_, j) in h_lc.sqf_list_include() {
            let g: TowerPoly = a_.gcd(&s)?;
            h = h.exquo(&TowerPoly::ground(&g.as_element()?.pow(j as i64)?, &de.t)?)?;
        }
        if invert {
            let h_lc_z: TowerPoly = TowerPoly::from_element_strict(&h.LC(), &z)?;
            let inv: TowerPoly = h_lc_z.invert(&s)?;
            let mut coeffs: Vec<RationalFunction> = vec![RationalFunction::one()];
            for coeff in &h.rep[1..] {
                if coeff.is_zero() {
                    coeffs.push(coeff.clone());
                    continue;
                }
                coeffs.push(modulo(coeff, &s, &z, Some(&inv))?.as_element()?);
            }
            h = TowerPoly::new(coeffs, &de.t);
        }
        h_list.push((s, h));
    }
    let b: bool = !np.iter().any(|(i, _)| i.has(&[de.t.clone(), z.clone()]));
    Ok((h_list, b))
}

/// `e mod s` in `z`, times `factor` (`_mod`).
fn modulo(e: &RationalFunction, s: &TowerPoly, z: &Basic, factor: Option<&TowerPoly>) -> PolyResult<TowerPoly> {
    let (n, d) = frac_in(e, z, false)?;
    let mut value: TowerPoly = if d.is_ground() {
        n.mul_ground(&d.LC().invert()?)
    } else {
        n.mul(&d.invert(s)?)?
    };
    if let Some(f) = factor {
        value = value.mul(f)?;
    }
    value.rem(s)
}

/// The logarithmic part as an expression: real logarithms and arc-tangents
/// by Rioboo's conversion where the tower is real, sums over the roots
/// otherwise (`residue_reduce_to_basic`).
pub fn residue_reduce_to_basic(h_list: &[(TowerPoly, TowerPoly)], de: &DifferentialExtension, z: &Basic) -> R {
    let i: Basic = crate::core::symbol::Dummy::new("i", &[]).0;
    let s: Vec<(Basic, Basic)> = de.backsubstitutions()?;
    let imaginary: Basic = Basic::imaginary_unit();
    let mut real_tower: bool = true;
    for f in &de.Tfuncs {
        if f.apply(&de.x)?.has(&imaginary) {
            real_tower = false;
        }
    }
    let mut result: Basic = Basic::zero();
    for (sa, h) in h_list {
        let mut real: Option<Basic> = None;
        let i_list: [Basic; 1] = [imaginary.clone()];
        if real_tower && !(sa.has(&i_list) || h.has(&i_list)) {
            real = super::rationaltools::log_to_real(&h.as_expr()?, &sa.as_expr()?, &de.t, z, false)?;
        }
        match real {
            Some(value) => result = add(&[result, subs_seq(&value, &s)?])?,
            None => {
                let body: Basic = subs_seq(
                    &xrep(&mul(&[i.clone(), log(&h.as_expr()?)?])?, &[(z.clone(), i.clone())])?,
                    &s,
                )?;
                let f: Basic = crate::core::subs::subs(&sa.as_expr()?, z, &i)?;
                result = add(&[
                    result,
                    crate::polys::rootoftools::root_sum(&f, Some((&i, &body)), Some(&i), true, false)?,
                ])?;
            }
        }
    }
    Ok(result)
}

/// The derivative of the logarithmic part as a sum over the roots
/// (`residue_reduce_derivation`).
pub fn residue_reduce_derivation(h_list: &[(TowerPoly, TowerPoly)], de: &DifferentialExtension, z: &Basic) -> R {
    let i: Basic = crate::core::symbol::Dummy::new("i", &[]).0;
    let mut terms: Vec<Basic> = Vec::new();
    for (sa, h) in h_list {
        let ratio_: Basic = mul(&[derivation(h, de, false)?.as_expr()?, pw(&h.as_expr()?, &int(-1))?])?;
        let body: Basic = mul(&[i.clone(), xrep(&ratio_, &[(z.clone(), i.clone())])?])?;
        let f: Basic = crate::core::subs::subs(&sa.as_expr()?, z, &i)?;
        terms.push(crate::polys::rootoftools::root_sum(
            &f,
            Some((&i, &body)),
            Some(&i),
            true,
            false,
        )?);
    }
    add(&terms)
}

/// The derivative of the logarithmic part as an element of the field
/// (`_residue_derivation_element`).
pub fn residue_derivation_element(
    h_list: &[(TowerPoly, TowerPoly)],
    de: &DifferentialExtension,
    z: &Basic,
) -> PolyResult<RationalFunction> {
    let mut total: RationalFunction = RationalFunction::zero();
    for (sa, h) in h_list {
        total = total.add(&trace(sa, &derivation(h, de, false)?, h, z)?)?;
    }
    Ok(total)
}

/// `sum_{s(a) = 0} a num(a)/den(a)` by Newton's sums (`_trace`).
fn trace(sa: &TowerPoly, num: &TowerPoly, den: &TowerPoly, z: &Basic) -> PolyResult<RationalFunction> {
    let s: TowerPoly = sa.monic()?;
    let n: i64 = s.degree(None);
    if n <= 0 {
        return Ok(RationalFunction::zero());
    }
    let zf: RationalFunction = RationalFunction::generator(z);
    let s_t: TowerPoly = TowerPoly::new(s.rep.clone(), z);
    let value: TowerPoly = match modulo(&num.as_element()?.mul(&zf)?.div(&den.as_element()?)?, &s_t, z, None) {
        Ok(v) => v,
        Err(PolyError::Value(_)) => {
            return Err(PolyError::NotImplemented(format!(
                "a sum over the roots of {} with a pole",
                s
            )));
        }
        Err(e) => return Err(e),
    };
    let c: Vec<RationalFunction> = (0..=n).map(|k| s_t.nth(n - k)).collect();
    let mut p: Vec<RationalFunction> = vec![RationalFunction::from_int(&Int::from(n))];
    for k in 1..n {
        let mut acc: RationalFunction = c[k as usize].mul(&RationalFunction::from_int(&Int::from(k)))?;
        for j in 1..k {
            acc = acc.add(&c[j as usize].mul(&p[(k - j) as usize])?)?;
        }
        p.push(acc.neg());
    }
    let mut total: RationalFunction = RationalFunction::zero();
    for k in 0..n.min(value.degree(None) + 1) {
        total = total.add(&value.nth(k).mul(&p[k as usize])?)?;
    }
    Ok(total)
}

/// The integral of a polynomial in a primitive monomial
/// (`integrate_primitive_polynomial`): `(q, r, b)` with `p = Dq + r`, `b`
/// whether `r` is in the base field.
pub fn integrate_primitive_polynomial(
    p: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly, bool)> {
    let mut p: TowerPoly = in_t(p, de)?;
    let zero: TowerPoly = TowerPoly::zero(&de.t);
    let mut q: TowerPoly = TowerPoly::zero(&de.t);
    if !p.has(std::slice::from_ref(&de.t)) {
        return Ok((zero, p, true));
    }
    loop {
        crate::timeout::check()?;
        if !p.has(std::slice::from_ref(&de.t)) {
            return Ok((q, p, true));
        }
        let below: Basic = de.T[de.at(de.level - 1)].clone();
        let (dta, dtb) = frac_in_pair(&de.d, &TowerPoly::one(&de.d.gen), &below, false)?;
        de.decrement_level()?;
        let a: RationalFunction = p.LC();
        let outcome = (|| -> PolyResult<Option<((TowerPoly, TowerPoly), Vec<RationalFunction>)>> {
            let (aa, ad) = frac_in(&a, &de.t, false)?;
            super::prde::limited_integrate(&aa, &ad, &[(dta.clone(), dtb.clone())], de)
        })();
        de.increment_level()?;
        let ((ba, bd), c) = match outcome {
            Ok(Some(v)) => v,
            Ok(None) => return Ok((q, p, false)),
            Err(e) if is_non_elementary(&e) => return Ok((q, p, false)),
            Err(e) => return Err(e),
        };
        let m: i64 = p.degree(Some(&de.t));
        let q0: TowerPoly = TowerPoly::monomial(
            &de.t,
            (m + 1) as usize,
            &c[0].div(&RationalFunction::from_int(&Int::from(m + 1)))?,
        )
        .add(&TowerPoly::monomial(
            &de.t,
            m as usize,
            &ba.as_element()?.div(&bd.as_element()?)?,
        ))?;
        p = p.sub(&derivation(&q0, de, false)?)?;
        q = q.add(&q0)?;
    }
}

fn back(e: &RationalFunction, s: &[(Basic, Basic)]) -> R {
    subs_seq(&e.as_expr()?, s)
}

fn fraction_expr(a: &TowerPoly, d: &TowerPoly) -> R {
    if d.is_one() {
        return a.as_expr();
    }
    mul(&[a.as_expr()?, pw(&d.as_expr()?, &int(-1))?])
}

fn quotient(a: &TowerPoly, d: &TowerPoly) -> PolyResult<RationalFunction> {
    a.as_element()?.div(&d.as_element()?)
}

fn derivative_of_fraction(ga: &TowerPoly, gd: &TowerPoly, de: &DifferentialExtension) -> PolyResult<RationalFunction> {
    let top: TowerPoly = gd
        .mul(&derivation(ga, de, false)?)?
        .sub(&ga.mul(&derivation(gd, de, false)?)?)?;
    top.as_element()?.div(&gd.pow(2).as_element()?)
}

fn non_elementary_integral(f: &Basic, x: &Basic) -> R {
    crate::integrals::integral::non_elementary(f, x)
}

/// The integral of an element of `k(t)` with `t` primitive
/// (`integrate_primitive`): `(ans, i, b)`, `i` the part left (a
/// `NonElementaryIntegral` when `b` is false).
pub fn integrate_primitive(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<(Basic, Basic, bool)> {
    let z: Basic = unique_dummy("z");
    let s: Vec<(Basic, Basic)> = de.backsubstitutions()?;
    let (g1, h, r) = hermite_reduce(a, d, de)?;
    let (g2, b) = residue_reduce(&h.0, &h.1, de, Some(&z), true)?;
    if !b {
        let ie: RationalFunction = quotient(a, d)?
            .sub(&derivative_of_fraction(&g1.0, &g1.1, de)?)?
            .sub(&residue_derivation_element(&g2, de, &z)?)?;
        let i: Basic = non_elementary_integral(&back(&ie, &s)?, &de.x)?;
        let ans: Basic = add(&[
            subs_seq(&fraction_expr(&g1.0, &g1.1)?, &s)?,
            residue_reduce_to_basic(&g2, de, &z)?,
        ])?;
        return Ok((ans, i, b));
    }
    let pe: RationalFunction = quotient(&h.0, &h.1)?
        .sub(&residue_derivation_element(&g2, de, &z)?)?
        .add(&quotient(&r.0, &r.1)?)?;
    let p: TowerPoly = TowerPoly::from_element_strict(&pe, &de.t)?;
    let (q, ip, b) = integrate_primitive_polynomial(&p, de)?;
    let ret: Basic = add(&[
        subs_seq(&add(&[fraction_expr(&g1.0, &g1.1)?, q.as_expr()?])?, &s)?,
        residue_reduce_to_basic(&g2, de, &z)?,
    ])?;
    let i: Basic = if !b {
        non_elementary_integral(&back(&ip.as_element()?, &s)?, &de.x)?
    } else {
        ip.as_element()?.as_expr()?
    };
    Ok((ret, i, b))
}

/// The terms `(i, v_i)` of the integral of a Laurent polynomial in a
/// hyperexponential monomial, by Risch differential equations
/// (`_hyperexponential_terms`).
fn hyperexponential_terms(
    p: &LaurentPolynomial,
    de: &mut DifferentialExtension,
) -> PolyResult<(Vec<(i64, RationalFunction)>, bool)> {
    let dtt: TowerPoly = de.d.exquo(&TowerPoly::monomial(&de.t, 1, &RationalFunction::one()))?;
    let mut terms: Vec<(i64, RationalFunction)> = Vec::new();
    let mut b: bool = true;
    if p.is_zero() {
        return Ok((terms, b));
    }
    de.decrement_level()?;
    let outcome = (|| -> PolyResult<()> {
        for i in p.low_degree()..=p.degree() {
            if i == 0 {
                continue;
            }
            crate::timeout::check()?;
            let a: RationalFunction = p.nth(i);
            let (aa, ad) = frac_in(&a, &de.t, true)?;
            let (ida, idd) = frac_in(
                &dtt.as_element()?.mul(&RationalFunction::from_int(&Int::from(i)))?,
                &de.t,
                true,
            )?;
            match super::rde::risch_de(&ida, &idd, &aa, &ad, de) {
                Ok((va, vd)) => {
                    let v: RationalFunction = quotient(&va, &vd)?;
                    if !v.is_zero() {
                        terms.push((i, v));
                    }
                }
                Err(e) if is_non_elementary(&e) => b = false,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    })();
    de.increment_level()?;
    outcome?;
    Ok((terms, b))
}

/// The integral of a Laurent polynomial in a hyperexponential monomial
/// (`integrate_hyperexponential_polynomial`).
pub fn integrate_hyperexponential_polynomial(
    p: &LaurentPolynomial,
    de: &mut DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly, bool)> {
    let (terms, b) = hyperexponential_terms(p, de)?;
    let t1f: RationalFunction = RationalFunction::generator(&de.t);
    let mut q: RationalFunction = RationalFunction::zero();
    for (i, v) in terms {
        q = q.add(&v.mul(&t1f.pow(i)?)?)?;
    }
    let (qa, qd) = frac_in(&q, &de.t, false)?;
    Ok((qa, qd, b))
}

/// `sum v_k t**k` with one common denominator (`_laurent_expr`).
fn laurent_expr(terms: &[(i64, RationalFunction)], t: &Basic) -> PolyResult<(Basic, Basic)> {
    if terms.is_empty() {
        return Ok((Basic::zero(), Basic::one()));
    }
    let values: Vec<RationalFunction> = terms.iter().map(|(_, v)| v.clone()).collect();
    let (numerators, denominator) = super::tower::integral_pair(&values, &[RationalFunction::one()])?;
    let d: &RationalFunction = &denominator[0];
    let mut parts: Vec<Basic> = Vec::new();
    for ((k, _), n) in terms.iter().zip(&numerators) {
        parts.push(mul(&[n.as_expr()?, pw(t, &int(*k))?])?);
    }
    let total: Basic = add(&parts)?;
    if d.is_one() {
        return Ok((total, Basic::one()));
    }
    Ok((mul(&[total, pw(&d.as_expr()?, &int(-1))?])?, d.as_expr()?))
}

/// The integral of an element of `k(t)` with `t` hyperexponential
/// (`integrate_hyperexponential`), the special case of a vanishing
/// denominator as a `Piecewise` with `conds == "piecewise"`.
pub fn integrate_hyperexponential(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &mut DifferentialExtension,
    conds: &str,
) -> PolyResult<(Basic, Basic, bool)> {
    let z: Basic = unique_dummy("z");
    let s: Vec<(Basic, Basic)> = de.backsubstitutions()?;
    let (g1, h, r) = hermite_reduce(a, d, de)?;
    let (g2, b) = residue_reduce(&h.0, &h.1, de, Some(&z), true)?;
    if !b {
        let ie: RationalFunction = quotient(a, d)?
            .sub(&derivative_of_fraction(&g1.0, &g1.1, de)?)?
            .sub(&residue_derivation_element(&g2, de, &z)?)?;
        let i: Basic = non_elementary_integral(&cancel_expr(&back(&ie, &s)?)?, &de.x)?;
        let ans: Basic = add(&[
            subs_seq(&fraction_expr(&g1.0, &g1.1)?, &s)?,
            residue_reduce_to_basic(&g2, de, &z)?,
        ])?;
        return Ok((ans, i, b));
    }
    let pe: RationalFunction = quotient(&h.0, &h.1)?
        .sub(&residue_derivation_element(&g2, de, &z)?)?
        .add(&quotient(&r.0, &r.1)?)?;
    let pp: LaurentPolynomial = as_poly_1t(&pe, &de.t)?;
    let (terms, b) = hyperexponential_terms(&pp, de)?;
    let t1f: RationalFunction = RationalFunction::generator(&de.t);
    let mut q: RationalFunction = RationalFunction::zero();
    for (k, v) in &terms {
        q = q.add(&v.mul(&t1f.pow(*k)?)?)?;
    }
    let (qa, qd) = frac_in(&q, &de.t, false)?;
    let i0: RationalFunction = pp.nth(0);
    let mut ret: Basic = add(&[
        subs_seq(&fraction_expr(&g1.0, &g1.1)?, &s)?,
        residue_reduce_to_basic(&g2, de, &z)?,
    ])?;
    let (qexpr, qd_expr) = laurent_expr(&terms, &de.t)?;
    let qexpr: Basic = subs_seq(&qexpr, &s)?;
    let qds: Basic = subs_seq(&qd_expr, &s)?;
    if conds == "piecewise"
        && !qds.free_symbols().contains(&de.x)
        && crate::core::expr::is(&qds, crate::core::assumptions::Predicate::zero) != Some(false)
    {
        let inner: Basic = subs_seq(&pe.sub(&i0)?.subs(&de.t, &RationalFunction::one())?.as_expr()?, &s)?;
        let x: Basic = de.x.clone();
        let piece: Basic = crate::functions::elementary::piecewise::piecewise(&[
            (qexpr, crate::core::relational::ne(&qds, &Basic::zero())?),
            (
                risch_integrate(&inner, &x, None, "log", None, "piecewise")?,
                crate::core::logic::true_(),
            ),
        ])?;
        ret = add(&[ret, piece])?;
    } else {
        ret = add(&[ret, qexpr])?;
    }
    let i: Basic = if !b {
        let ie: RationalFunction = pe.sub(&derivative_of_fraction(&qa, &qd, de)?)?;
        non_elementary_integral(&back(&ie, &s)?, &de.x)?
    } else {
        i0.as_expr()?
    };
    Ok((ret, i, b))
}

fn t2_plus_1(t: &Basic) -> TowerPoly {
    TowerPoly::from_list(
        &[
            RationalFunction::one(),
            RationalFunction::zero(),
            RationalFunction::one(),
        ],
        t,
    )
}

/// The integral of a polynomial in a hypertangent monomial
/// (`integrate_hypertangent_polynomial`): `(q, c)` with `p - Dq - c Dt/(t**2 + 1)`
/// in the base field.
pub fn integrate_hypertangent_polynomial(
    p: &TowerPoly,
    de: &DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly)> {
    let (q, r) = polynomial_reduce(p, de)?;
    let a: TowerPoly = de.d.exquo(&t2_plus_1(&de.t))?;
    let c: TowerPoly = TowerPoly::ground(
        &r.nth(1)
            .div(&a.as_element()?.mul(&RationalFunction::from_int(&Int::from(2)))?)?,
        &de.t,
    )?;
    Ok((q, c))
}

/// The integral of a reduced element with a power of `t**2 + 1` as the
/// denominator (`integrate_hypertangent_reduced`), by coupled systems.
pub fn integrate_hypertangent_reduced(
    pa: &TowerPoly,
    pd: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<(TowerPoly, TowerPoly, bool)> {
    let h2: TowerPoly = t2_plus_1(&de.t);
    let eta: RationalFunction = de.d.exquo(&h2)?.as_element()?;
    let (mut qa, mut qd) = (TowerPoly::zero(&de.t), TowerPoly::one(&de.t));
    let (pa, pd) = (in_t(pa, de)?, in_t(pd, de)?);
    let (mut pa, mut pd) = pa.cancel(&pd)?;
    let tf: RationalFunction = RationalFunction::generator(&de.t);
    loop {
        crate::timeout::check()?;
        let m: i64 = super::rde::order_at(&pd, &h2, &de.t)?;
        if pd.degree(Some(&de.t)) != 2 * m {
            return Err(PolyError::Value(format!(
                "{}/{} is not reduced (the denominator must be a power of t**2 + 1).",
                pa, pd
            )));
        }
        if m <= 0 {
            return Ok((qa, qd, true));
        }
        let h: TowerPoly = pa.mul(&h2.pow(m as usize))?.exquo(&pd)?;
        let r: TowerPoly = h.rem(&h2)?;
        let (a, b) = (r.nth(1), r.nth(0));
        de.decrement_level()?;
        let outcome = (|| -> PolyResult<((TowerPoly, TowerPoly), (TowerPoly, TowerPoly))> {
            let f2 = frac_in(&eta.mul(&RationalFunction::from_int(&Int::from(2 * m)))?, &de.t, false)?;
            let g1 = frac_in(&a, &de.t, false)?;
            let g2 = frac_in(&b, &de.t, false)?;
            let zero = (TowerPoly::zero(&de.t), TowerPoly::one(&de.t));
            super::cde::coupled_de_system(&zero, &f2, &g1, &g2, de)
        })();
        de.increment_level()?;
        let ((ca, cd), (da, dd)) = match outcome {
            Ok(v) => v,
            Err(e) if is_non_elementary(&e) => return Ok((qa, qd, false)),
            Err(e) => return Err(e),
        };
        let c: RationalFunction = quotient(&ca, &cd)?;
        let d: RationalFunction = quotient(&da, &dd)?;
        let (q0a, q0d) = frac_in(&c.mul(&tf)?.add(&d)?.div(&h2.as_element()?.pow(m)?)?, &de.t, true)?;
        let dq0a: TowerPoly = q0d
            .mul(&derivation(&q0a, de, false)?)?
            .sub(&q0a.mul(&derivation(&q0d, de, false)?)?)?;
        let dq0d: TowerPoly = q0d.pow(2);
        let (npa, npd) = pa.mul(&dq0d)?.sub(&dq0a.mul(&pd)?)?.cancel(&pd.mul(&dq0d)?)?;
        pa = npa;
        pd = npd;
        let (nqa, nqd) = qa.mul(&q0d)?.add(&q0a.mul(&qd)?)?.cancel(&qd.mul(&q0d)?)?;
        qa = nqa;
        qd = nqd;
    }
}

/// The integral of an element of `k(t)` with `t` hypertangent
/// (`integrate_hypertangent`).
pub fn integrate_hypertangent(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<(Basic, Basic, bool)> {
    let z: Basic = unique_dummy("z");
    let s: Vec<(Basic, Basic)> = de.backsubstitutions()?;
    let (g1, h, r) = hermite_reduce(a, d, de)?;
    let (g2, b) = residue_reduce(&h.0, &h.1, de, Some(&z), true)?;
    if !b {
        let ie: RationalFunction = quotient(a, d)?
            .sub(&derivative_of_fraction(&g1.0, &g1.1, de)?)?
            .sub(&residue_derivation_element(&g2, de, &z)?)?;
        let i: Basic = non_elementary_integral(&back(&ie, &s)?, &de.x)?;
        let ans: Basic = add(&[
            subs_seq(&fraction_expr(&g1.0, &g1.1)?, &s)?,
            residue_reduce_to_basic(&g2, de, &z)?,
        ])?;
        return Ok((ans, i, b));
    }
    let mut pe: RationalFunction = quotient(&h.0, &h.1)?
        .sub(&residue_derivation_element(&g2, de, &z)?)?
        .add(&quotient(&r.0, &r.1)?)?;
    let (pa, pd) = frac_in(&pe, &de.t, true)?;
    let (q1a, q1d, b) = integrate_hypertangent_reduced(&pa, &pd, de)?;
    let mut ret: Basic = add(&[
        subs_seq(&add(&[fraction_expr(&g1.0, &g1.1)?, fraction_expr(&q1a, &q1d)?])?, &s)?,
        residue_reduce_to_basic(&g2, de, &z)?,
    ])?;
    pe = pe.sub(&derivative_of_fraction(&q1a, &q1d, de)?)?;
    if !b {
        return Ok((ret, non_elementary_integral(&back(&pe, &s)?, &de.x)?, b));
    }
    let pp: TowerPoly = TowerPoly::from_element_strict(&pe, &de.t)?;
    let (q2, c) = integrate_hypertangent_polynomial(&pp, de)?;
    ret = add(&[ret, subs_seq(&q2.as_expr()?, &s)?])?;
    let p: TowerPoly = pp.sub(&derivation(&q2, de, false)?)?;
    if derivation(&c, de, false)?.is_zero() {
        let c_elem: RationalFunction = c.as_element()?;
        let eta: RationalFunction = de.d.exquo(&t2_plus_1(&de.t))?.as_element()?;
        let logarithm: Basic = subs_seq(&log(&add(&[pw(&de.t, &int(2))?, Basic::one()])?)?, &s)?;
        ret = add(&[ret, mul(&[c_elem.as_expr()?, logarithm])?])?;
        let two: RationalFunction = RationalFunction::from_int(&Int::from(2));
        let i: Basic = p
            .as_element()?
            .sub(&c_elem.mul(&eta)?.mul(&RationalFunction::generator(&de.t))?.mul(&two)?)?
            .as_expr()?;
        return Ok((ret, i, true));
    }
    Ok((
        ret,
        non_elementary_integral(&back(&p.as_element()?, &s)?, &de.x)?,
        false,
    ))
}

/// The integral of an element of `k(t)` with `t` nonlinear and no special
/// polynomials (`integrate_nonlinear_no_specials`).
pub fn integrate_nonlinear_no_specials(
    a: &TowerPoly,
    d: &TowerPoly,
    de: &mut DifferentialExtension,
) -> PolyResult<(Basic, bool)> {
    let z: Basic = unique_dummy("z");
    let s: Vec<(Basic, Basic)> = de.backsubstitutions()?;
    let (g1, h, r) = hermite_reduce(a, d, de)?;
    let (g2, b) = residue_reduce(&h.0, &h.1, de, Some(&z), true)?;
    if !b {
        return Ok((
            add(&[
                subs_seq(&fraction_expr(&g1.0, &g1.1)?, &s)?,
                residue_reduce_to_basic(&g2, de, &z)?,
            ])?,
            b,
        ));
    }
    let pe: RationalFunction = quotient(&h.0, &h.1)?
        .sub(&residue_derivation_element(&g2, de, &z)?)?
        .add(&quotient(&r.0, &r.1)?)?;
    let p: TowerPoly = TowerPoly::from_element_strict(&pe, &de.t)?;
    let (q1, q2) = polynomial_reduce(&p, de)?;
    let b: bool = !q2.has(std::slice::from_ref(&de.t));
    let ret: Basic = add(&[
        back(&quotient(&g1.0, &g1.1)?.add(&q1.as_element()?)?, &s)?,
        residue_reduce_to_basic(&g2, de, &z)?,
    ])?;
    Ok((ret, b))
}

impl DifferentialExtension {
    /// The rational functions of the half-angle tangents back to sines and
    /// cosines (`restore_sincos`).
    pub fn restore_sincos(&self, expr: &Basic, _drop_constants: bool) -> R {
        if self.sincos_args.is_empty() {
            return Ok(expr.clone());
        }
        Err(crate::error::not_ported(
            "restore_sincos (the half-angle tangents back to sines and cosines)",
        ))
    }
}

/// `risch_integrate`: the antiderivative, with a `NonElementaryIntegral`
/// for the part proved not elementary.
pub fn risch_integrate(
    f: &Basic,
    x: &Basic,
    extension: Option<DifferentialExtension>,
    handle_first: &str,
    rewrite_complex: Option<bool>,
    conds: &str,
) -> R {
    let (result, i) = risch_integrate_separate(f, x, extension, handle_first, rewrite_complex, conds)?;
    add(&[result, i])
}

/// `risch_integrate_separate`: `(result, i)`, the elementary part and the
/// part proved not elementary.
pub fn risch_integrate_separate(
    f: &Basic,
    x: &Basic,
    extension: Option<DifferentialExtension>,
    handle_first: &str,
    rewrite_complex: Option<bool>,
    conds: &str,
) -> PolyResult<(Basic, Basic)> {
    let mut de: DifferentialExtension = match extension {
        Some(e) => e,
        None => DifferentialExtension::new(f, x, handle_first, true, rewrite_complex)?,
    };
    let (mut fa, mut fd) = (de.fa.clone(), de.fd.clone());
    let mut result: Basic = Basic::zero();
    let cases: Vec<Case> = de.cases.iter().rev().copied().collect();
    for case in cases {
        crate::timeout::check()?;
        let t_list: [Basic; 1] = [de.t.clone()];
        if !fa.has(&t_list) && !fd.has(&t_list) && case != Case::Base {
            de.decrement_level()?;
            let (a, d) = frac_in_pair(&fa, &fd, &de.t, false)?;
            fa = a;
            fd = d;
            continue;
        }
        let (a, d) = fa.cancel(&fd)?;
        fa = a;
        fd = d;
        let (ans, i, b): (Basic, Basic, bool) = match case {
            Case::Exp => integrate_hyperexponential(&fa, &fd, &mut de, conds)?,
            Case::Primitive => integrate_primitive(&fa, &fd, &mut de)?,
            Case::Tan => integrate_hypertangent(&fa, &fd, &mut de)?,
            Case::Base => {
                let (pe, qe) = (fa.as_expr()?, fd.as_expr()?);
                let integrand = super::rationaltools::Integrand::pair(&pe, &qe);
                let t: Basic = crate::core::symbol::Symbol::new("t", &[]).0;
                (
                    super::rationaltools::ratint(&integrand, &de.x, &t, None)?,
                    Basic::zero(),
                    false,
                )
            }
            _ => {
                return Err(PolyError::NotImplemented(
                    "Only exponential, logarithmic, hypertangent and primitive extensions are currently supported."
                        .to_string(),
                ))
            }
        };
        result = add(&[result, ans])?;
        if b {
            de.decrement_level()?;
            let (a, d) = frac_in_expr(&i, &de.t, false)?;
            fa = a;
            fd = d;
            continue;
        }
        result = de.restore_sincos(&subs_seq(&result, &de.backsubs)?, true)?;
        if matches!(i.head(), Head::NonElementaryIntegral) {
            let rest: Basic = if result == Basic::zero() {
                non_elementary_integral(f, x)?
            } else {
                non_elementary_integral(
                    &de.restore_sincos(&subs_seq(&i.args()[0], &de.backsubs)?, false)?,
                    &i.args()[1],
                )?
            };
            return Ok((result, rest));
        }
        return Ok((result, Basic::zero()));
    }
    Err(PolyError::NotImplemented(
        "The differential extension has no base case.".to_string(),
    ))
}
