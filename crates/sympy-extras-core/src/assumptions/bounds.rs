//! Polynomial bounds of elementary functions for the cylindrical algebraic
//! decomposition (the method of MetiTarski), ported from
//! `sympy_extras.assumptions.bounds`: every application `g(u)` of `exp`,
//! `log`, `sin`, `cos`, `atan`, `sinh`, `cosh`, `tanh`, of a rational
//! power or of `Abs` to a polynomial argument is replaced by a fresh real
//! dummy `t` with the polynomial constraints its true value satisfies
//! ([`polynomial_abstraction`]). The bounds are built with Python's
//! operators in their order.

#![allow(clippy::mutable_key_type)]

use std::collections::{BTreeSet, HashMap};

use crate::core::add::{add, sub};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is_true;
use crate::core::logic::{and_, implies, or_};
use crate::core::mul::{div, mul};
use crate::core::power::pow;
use crate::core::relational::{eq, ge, gt, le, lt};
use crate::error::PolyResult;
use crate::numbers::Int;

/// A formula with its function applications replaced by variables
/// (`Abstraction`).
#[derive(Debug, Clone)]
pub struct Abstraction {
    /// the polynomial formula
    pub formula: Basic,
    /// the polynomial constraints satisfied by the true values of the new
    /// variables
    pub constraints: Vec<Basic>,
    /// the new variables
    pub variables: Vec<Basic>,
    /// the conditions under which the replaced applications are real
    pub requirements: Vec<Basic>,
}

fn int(n: i64) -> Basic {
    Basic::integer(Int::from(n))
}

fn rational(p: i64, q: i64) -> Basic {
    Basic::rational(Int::from(p), Int::from(q)).expect("a nonzero denominator")
}

fn a2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    add(&[a.clone(), b.clone()])
}

fn m2(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    mul(&[a.clone(), b.clone()])
}

/// `u**k/d` (Python's `u**k/d`).
fn term(u: &Basic, k: i64, d: i64) -> PolyResult<Basic> {
    div(&pow(u, &int(k))?, &int(d))
}

/// `1 + u + u**2/2 + ...` with the given denominators from the square on.
fn series(first: &Basic, u: &Basic, denominators: &[(i64, i64)]) -> PolyResult<Basic> {
    let mut s: Basic = a2(first, u)?;
    for (k, d) in denominators {
        s = a2(&s, &term(u, *k, *d)?)?;
    }
    Ok(s)
}

fn exp_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let (zero, one) = (Basic::zero(), Basic::one());
    let u_nonneg: Basic = ge(u, &zero)?;
    let u_nonpos: Basic = le(u, &zero)?;
    Ok(vec![
        gt(t, &zero)?,
        ge(t, &a2(&one, u)?)?,
        implies(&u_nonneg, &ge(t, &series(&one, u, &[(2, 2), (3, 6), (4, 24)])?)?),
        implies(&u_nonpos, &le(t, &series(&one, u, &[(2, 2)])?)?),
        implies(&lt(u, &one)?, &le(&m2(t, &sub(&one, u)?)?, &one)?),
        implies(
            &and_(&[u_nonneg.clone(), le(u, &one)?]),
            &le(t, &series(&one, u, &[(2, 2), (3, 6), (4, 8)])?)?,
        ),
    ])
}

fn log_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let (zero, one, two) = (Basic::zero(), Basic::one(), int(2));
    let u_minus_one: Basic = sub(u, &one)?;
    let t_u_plus_one: Basic = m2(t, &a2(u, &one)?)?;
    let twice: Basic = m2(&two, &u_minus_one)?;
    Ok(vec![
        gt(u, &zero)?,
        le(t, &u_minus_one)?,
        ge(&m2(u, t)?, &u_minus_one)?,
        implies(&ge(u, &one)?, &ge(&t_u_plus_one, &twice)?),
        implies(&le(u, &one)?, &le(&t_u_plus_one, &twice)?),
    ])
}

fn sin_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let zero: Basic = Basic::zero();
    let (pos, neg) = (ge(u, &zero)?, le(u, &zero)?);
    let cubic: Basic = sub(u, &term(u, 3, 6)?)?;
    let quintic: Basic = a2(&cubic, &term(u, 5, 120)?)?;
    Ok(vec![
        ge(t, &Basic::negative_one())?,
        le(t, &Basic::one())?,
        implies(&pos, &le(t, u)?),
        implies(&neg, &ge(t, u)?),
        implies(&pos, &ge(t, &cubic)?),
        implies(&neg, &le(t, &cubic)?),
        implies(&pos, &le(t, &quintic)?),
        implies(&neg, &ge(t, &quintic)?),
    ])
}

fn cos_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let one: Basic = Basic::one();
    let quadratic: Basic = sub(&one, &term(u, 2, 2)?)?;
    Ok(vec![
        ge(t, &Basic::negative_one())?,
        le(t, &one)?,
        ge(t, &quadratic)?,
        le(t, &a2(&quadratic, &term(u, 4, 24)?)?)?,
    ])
}

fn atan_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let (zero, one) = (Basic::zero(), Basic::one());
    let cubic: Basic = sub(u, &term(u, 3, 3)?)?;
    let scaled: Basic = m2(t, &a2(&one, &pow(u, &int(2))?)?)?;
    let bound: Basic = rational(8, 5);
    Ok(vec![
        gt(t, &mul(&[Basic::negative_one(), bound.clone()])?)?,
        lt(t, &bound)?,
        implies(
            &ge(u, &zero)?,
            &and_(&[ge(t, &zero)?, le(t, u)?, ge(t, &cubic)?, ge(&scaled, u)?]),
        ),
        implies(
            &le(u, &zero)?,
            &and_(&[le(t, &zero)?, ge(t, u)?, le(t, &cubic)?, le(&scaled, u)?]),
        ),
    ])
}

fn tanh_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let zero: Basic = Basic::zero();
    let cubic: Basic = sub(u, &term(u, 3, 3)?)?;
    Ok(vec![
        gt(t, &Basic::negative_one())?,
        lt(t, &Basic::one())?,
        implies(&ge(u, &zero)?, &and_(&[ge(t, &zero)?, le(t, u)?, ge(t, &cubic)?])),
        implies(&le(u, &zero)?, &and_(&[le(t, &zero)?, ge(t, u)?, le(t, &cubic)?])),
    ])
}

fn sinh_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let zero: Basic = Basic::zero();
    let cubic: Basic = a2(u, &term(u, 3, 6)?)?;
    Ok(vec![
        implies(&ge(u, &zero)?, &and_(&[ge(t, u)?, ge(t, &cubic)?])),
        implies(&le(u, &zero)?, &and_(&[le(t, u)?, le(t, &cubic)?])),
    ])
}

fn cosh_bounds(t: &Basic, u: &Basic) -> PolyResult<Vec<Basic>> {
    let one: Basic = Basic::one();
    Ok(vec![ge(t, &one)?, ge(t, &a2(&one, &term(u, 2, 2)?)?)?])
}

type BoundRule = fn(&Basic, &Basic) -> PolyResult<Vec<Basic>>;

fn rule_of(e: &Basic) -> Option<BoundRule> {
    let Head::Function(f) = e.head() else {
        return None;
    };
    Some(match f {
        Function::Exp => exp_bounds,
        Function::Log => log_bounds,
        Function::Sin => sin_bounds,
        Function::Cos => cos_bounds,
        Function::Atan => atan_bounds,
        Function::Tanh => tanh_bounds,
        Function::Sinh => sinh_bounds,
        Function::Cosh => cosh_bounds,
        _ => return None,
    })
}

/// Whether `e` is a polynomial in `syms` (`is_polynomial`, `True` only).
fn is_polynomial_in(e: &Basic, syms: &BTreeSet<Basic>) -> bool {
    if !e.free_symbols().iter().any(|s| syms.contains(s)) {
        return true;
    }
    if e.is_symbol() {
        return true;
    }
    if e.is_add() || e.is_mul() {
        return e.args().iter().all(|a| is_polynomial_in(a, syms));
    }
    if e.is_pow() {
        let (base, exponent) = (&e.args()[0], &e.args()[1]);
        if exponent.free_symbols().iter().any(|s| syms.contains(s)) {
            return false;
        }
        return is_polynomial_in(base, syms)
            && matches!(exponent.head(), Head::Rational { .. })
            && exponent
                .as_rational()
                .is_some_and(|v| v.is_int() && *v >= crate::numbers::Rat::ZERO);
    }
    false
}

/// Whether a value is an expression (not a `Tuple` or a Boolean).
fn is_expr(e: &Basic) -> bool {
    !matches!(e.head(), Head::Tuple) && !crate::core::logic::is_boolean(e) || e.is_symbol()
}

struct Abstractor {
    real: BTreeSet<Basic>,
    constraints: Vec<Basic>,
    requirements: Vec<Basic>,
    variables: Vec<Basic>,
    known: HashMap<Basic, Basic>,
}

impl Abstractor {
    fn fresh(&mut self, name: &str) -> Basic {
        let t: Basic = crate::core::Dummy::new(name, &[(Predicate::real, true)]).0;
        self.variables.push(t.clone());
        t
    }

    fn polynomial(&self, e: &Basic) -> bool {
        let symbols: BTreeSet<Basic> = e.free_symbols();
        if symbols.is_empty() {
            return is_true(e, Predicate::rational);
        }
        if !symbols
            .iter()
            .all(|s| self.real.contains(s) || self.variables.contains(s))
        {
            return false;
        }
        let mut ordered: Vec<Basic> = symbols.iter().cloned().collect();
        ordered.sort_by_cached_key(crate::core::sorting::sort_key);
        if !is_polynomial_in(e, &symbols) {
            return false;
        }
        let Ok(p) = crate::polys::polytools::Poly::new(e, &ordered) else {
            return false;
        };
        match p.coeffs() {
            Ok(cs) => cs.iter().all(|c| is_true(c, Predicate::rational)),
            Err(_) => false,
        }
    }

    /// `e` with its function applications replaced, bottom-up; `None` when
    /// it cannot be.
    fn expression(&mut self, e: &Basic) -> PolyResult<Option<Basic>> {
        if let Some(t) = self.known.get(e) {
            return Ok(Some(t.clone()));
        }
        let rule: Option<BoundRule> = rule_of(e);
        let is_abs: bool = matches!(e.head(), Head::Function(Function::Abs));
        if rule.is_some() || is_abs {
            let argument: &Basic = &e.args()[0];
            if !is_expr(argument) {
                return Ok(None);
            }
            let Some(inner) = self.expression(argument)? else {
                return Ok(None);
            };
            if !self.polynomial(&inner) {
                return Ok(None);
            }
            let t: Basic = self.fresh("t");
            match rule {
                None => {
                    let minus: Basic = mul(&[Basic::negative_one(), inner.clone()])?;
                    self.constraints.push(ge(&t, &Basic::zero())?);
                    self.constraints.push(or_(&[eq(&t, &inner)?, eq(&t, &minus)?]));
                }
                Some(r) => {
                    if matches!(e.head(), Head::Function(Function::Log)) {
                        self.requirements.push(gt(&inner, &Basic::zero())?);
                    }
                    self.constraints.extend(r(&t, &inner)?);
                }
            }
            self.known.insert(e.clone(), t.clone());
            return Ok(Some(t));
        }
        if e.is_pow() {
            let (base, exponent) = (&e.args()[0], &e.args()[1]);
            if let Some(v) = exponent
                .as_rational()
                .filter(|v| !v.is_int() && exponent.is_number_node())
            {
                let Some(inner) = self.expression(base)? else {
                    return Ok(None);
                };
                if !self.polynomial(&inner) {
                    return Ok(None);
                }
                let (p, q) = (crate::numbers::numer(v), crate::numbers::denom(v));
                let t: Basic = self.fresh("t");
                let zero: Basic = Basic::zero();
                // a real root of a nonnegative base (the reality of the power)
                if p > Int::ZERO {
                    self.requirements.push(ge(&inner, &zero)?);
                    self.constraints.push(ge(&inner, &zero)?);
                    self.constraints.push(ge(&t, &zero)?);
                    self.constraints
                        .push(eq(&pow(&t, &Basic::integer(q))?, &pow(&inner, &Basic::integer(p))?)?);
                } else {
                    self.requirements.push(gt(&inner, &zero)?);
                    self.constraints.push(gt(&inner, &zero)?);
                    self.constraints.push(gt(&t, &zero)?);
                    let product: Basic = m2(&pow(&t, &Basic::integer(q))?, &pow(&inner, &Basic::integer(-p))?)?;
                    self.constraints.push(eq(&product, &Basic::one())?);
                }
                self.known.insert(e.clone(), t.clone());
                return Ok(Some(t));
            }
        }
        if !e.args().is_empty() && !e.is_symbol() {
            let mut new_args: Vec<Basic> = Vec::new();
            let mut changed: bool = false;
            for a in e.args() {
                if is_expr(a) {
                    let Some(b) = self.expression(a)? else {
                        return Ok(None);
                    };
                    changed = changed || !b.ptr_eq(a);
                    new_args.push(b);
                } else {
                    new_args.push(a.clone());
                }
            }
            if !changed {
                return Ok(Some(e.clone()));
            }
            let rebuilt: Basic = crate::core::function::rebuild_evaluated(e, new_args)?;
            return Ok(if is_expr(&rebuilt) { Some(rebuilt) } else { None });
        }
        Ok(Some(e.clone()))
    }

    fn formula(&mut self, f: &Basic) -> PolyResult<Option<Basic>> {
        match f.head() {
            Head::Relational(op) => {
                let Some(lhs) = self.expression(&f.args()[0])? else {
                    return Ok(None);
                };
                let Some(rhs) = self.expression(&f.args()[1])? else {
                    return Ok(None);
                };
                Ok(Some(crate::core::relational::evaluate_relation(*op, &lhs, &rhs)?))
            }
            Head::And | Head::Or | Head::Not | Head::Implies | Head::Equivalent | Head::Xor | Head::Ite => {
                let mut parts: Vec<Basic> = Vec::new();
                for a in f.args() {
                    let Some(b) = self.formula(a)? else {
                        return Ok(None);
                    };
                    parts.push(b);
                }
                Ok(Some(crate::core::function::rebuild_evaluated(f, parts)?))
            }
            Head::BooleanTrue | Head::BooleanFalse => Ok(Some(f.clone())),
            // memberships and predicates are not polynomial statements
            _ => Ok(None),
        }
    }
}

/// Replace the elementary functions of a formula by variables with
/// polynomial constraints (`polynomial_abstraction`); `None` when the
/// formula has no such application, more than `max_variables` of them, or
/// a part which is not polynomial.
pub fn polynomial_abstraction(
    formula: &Basic,
    real: &[Basic],
    max_variables: usize,
) -> PolyResult<Option<Abstraction>> {
    let mut abstractor = Abstractor {
        real: real.iter().cloned().collect(),
        constraints: Vec::new(),
        requirements: Vec::new(),
        variables: Vec::new(),
        known: HashMap::new(),
    };
    let result: Option<Basic> = abstractor.formula(formula)?;
    let Some(result) = result else {
        return Ok(None);
    };
    if abstractor.variables.is_empty() || abstractor.variables.len() > max_variables {
        return Ok(None);
    }
    Ok(Some(Abstraction {
        formula: result,
        constraints: abstractor.constraints,
        variables: abstractor.variables,
        requirements: abstractor.requirements,
    }))
}
