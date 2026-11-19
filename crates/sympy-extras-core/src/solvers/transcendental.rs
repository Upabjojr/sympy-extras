//! Port of `sympy_extras/solvers/transcendental.py`: transcendental
//! equations and inequalities in one real unknown reduced to polynomial
//! problems in a kernel variable (`t = exp(u) > 0`, `w = tan(u/2)`,
//! `l = log(u)`, `y = u**(1/q) >= 0`), solved by the cylindrical algebraic
//! decomposition and pulled back through the inverse images of the
//! kernels.

#![allow(clippy::mutable_key_type, clippy::type_complexity)]

use std::collections::HashMap;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::{is, terms_of};
use crate::core::symbol::Dummy;
use crate::error::PolyResult;
use crate::numbers::*;
use crate::sets::sets::{empty_set, finite_elements, finite_set, intersection, interval, interval_parts, union};

type R = PolyResult<Basic>;

fn add(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn add2(a: &Basic, b: &Basic) -> R {
    add(&[a.clone(), b.clone()])
}
fn sub(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn mul2(a: &Basic, b: &Basic) -> R {
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
fn exp(e: &Basic) -> R {
    crate::functions::elementary::exponential::exp(e)
}
fn log(e: &Basic) -> R {
    crate::functions::elementary::exponential::log(e)
}
fn sin(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::sin(e)
}
fn cos(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::cos(e)
}
fn tan(e: &Basic) -> R {
    crate::functions::elementary::trigonometric::tan(e)
}
fn atan(e: &Basic) -> R {
    crate::functions::elementary::inverse::atan(e)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn is_true(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanTrue)
}
fn is_false(b: &Basic) -> bool {
    matches!(b.head(), Head::BooleanFalse)
}
fn true_() -> Basic {
    crate::core::logic::boolean(true)
}
fn false_() -> Basic {
    crate::core::logic::boolean(false)
}
fn xreplace(b: &Basic, rule: &HashMap<Basic, Basic>) -> R {
    crate::core::function::xreplace_evaluated(b, rule)
}
fn rule(pairs: &[(Basic, Basic)]) -> HashMap<Basic, Basic> {
    pairs.iter().cloned().collect()
}
fn dummy(name: &str, assumptions: &[(Predicate, bool)]) -> Basic {
    Dummy::new(name, assumptions).0
}
fn hints() -> crate::core::function::Hints {
    crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    }
}

/// The kernel family: `exp(u)`, `tan(u/2)`, `log(u)`, `u**(1/q)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Exp,
    Trig,
    Log,
    Root,
}

/// A formula in the kernel variable equivalent to a transcendental
/// formula in `x`, with the way back (`Reduction`).
#[derive(Clone, Debug)]
pub struct Reduction {
    pub formula: Basic,
    pub variable: Basic,
    pub argument: Basic,
    pub family: Family,
    pub order: i64,
    pub special: Basic,
}

impl Reduction {
    /// The transcendental expression the variable stands for.
    pub fn kernel(&self) -> R {
        let u: &Basic = &self.argument;
        match self.family {
            Family::Exp => exp(u),
            Family::Trig => tan(&div(u, &int(2))?),
            Family::Log => log(u),
            Family::Root => pow(u, &Basic::rational(Int::ONE, Int::from(self.order))?),
        }
    }
}

fn rel_op(e: &Basic) -> Option<RelOp> {
    match e.head() {
        Head::Relational(op) => Some(*op),
        _ => None,
    }
}

/// `atom.func(lhs, rhs)`: the relation, evaluated (`_relation`).
pub(crate) fn relation(op: RelOp, lhs: &Basic, rhs: &Basic) -> R {
    use crate::core::relational as rl;
    match op {
        RelOp::Lt => rl::lt(lhs, rhs),
        RelOp::Le => rl::le(lhs, rhs),
        RelOp::Gt => rl::gt(lhs, rhs),
        RelOp::Ge => rl::ge(lhs, rhs),
        RelOp::Eq => rl::eq(lhs, rhs),
        RelOp::Ne => rl::ne(lhs, rhs),
    }
}

/// The nonzero coefficients of `e` as a polynomial in `v`, by degree in
/// the order met (`_coefficients`); `None` when it is not one.
pub fn coefficients(e: &Basic, v: &Basic) -> PolyResult<Option<Vec<(i64, Basic)>>> {
    let ex: Basic = expand(e)?;
    let mut gathered: Vec<(i64, Vec<Basic>)> = Vec::new();
    for t in terms_of(&ex) {
        let (c, k) = crate::solvers::solvers::as_coeff_exponent(&t, v)?;
        let kv: Option<i64> = if k.is_integer_value() {
            k.as_rational().and_then(|r| i64::try_from(&numer(r)).ok())
        } else {
            None
        };
        let Some(kv) = kv.filter(|d| *d >= 0) else {
            return Ok(None);
        };
        if c.has(v) {
            return Ok(None);
        }
        match gathered.iter().position(|(d, _)| *d == kv) {
            Some(i) => gathered[i].1.push(c),
            None => gathered.push((kv, vec![c])),
        }
    }
    let mut result: Vec<(i64, Basic)> = Vec::new();
    for (degree, cs) in gathered {
        let coefficient: Basic = add(&cs)?;
        if coefficient != Basic::zero() {
            result.push((degree, coefficient));
        }
    }
    Ok(Some(result))
}

fn coefficient_at(cs: &[(i64, Basic)], k: i64) -> Option<&Basic> {
    cs.iter().find(|(d, _)| *d == k).map(|(_, c)| c)
}

/// `Poly.as_expr` of the coefficients (`_polynomial_expr`).
fn polynomial_expr(cs: &[(i64, Basic)], v: &Basic) -> R {
    let terms: Vec<Basic> = cs
        .iter()
        .map(|(k, c)| mul2(c, &pow(v, &int(*k))?))
        .collect::<PolyResult<_>>()?;
    add(&terms)
}

fn operations(e: &Basic) -> PolyResult<i64> {
    crate::core::function::count_ops(e)
}

// -- rewriting with the functional relations

fn argument(a: &Basic) -> Basic {
    a.args()[0].clone()
}

fn is_function(a: &Basic, fs: &[Function]) -> bool {
    matches!(a.head(), Head::Function(f) if fs.contains(f))
}

/// Hyperbolic functions and general powers as exponentials, the other
/// trigonometric functions as sines and cosines, the exponentials of sums
/// expanded (`_rewrite`).
fn rewrite(e: &Basic, x: &Basic) -> R {
    use crate::core::function::replace_where;
    let hyperbolic =
        |a: &Basic| is_function(a, &[Function::Sinh, Function::Cosh, Function::Tanh, Function::Coth]) && a.has(x);
    let as_exp =
        |a: &Basic| crate::core::rewriting::rewrite(a, &[], "exp", &crate::core::rewriting::RewriteHints::default());
    let mut e: Basic = replace_where(e, &hyperbolic, &as_exp)?;
    let general_power =
        |a: &Basic| a.is_pow() && a.args()[1].has(x) && !a.args()[0].has(x) && yes(&a.args()[0], Predicate::positive);
    let as_exponential = |a: &Basic| exp(&mul2(&a.args()[1], &log(&a.args()[0])?)?);
    e = replace_where(&e, &general_power, &as_exponential)?;
    e = replace_where(&e, &|a| is_function(a, &[Function::Tan]) && a.has(x), &|a| {
        div(&sin(&argument(a))?, &cos(&argument(a))?)
    })?;
    e = replace_where(&e, &|a| is_function(a, &[Function::Cot]) && a.has(x), &|a| {
        div(&cos(&argument(a))?, &sin(&argument(a))?)
    })?;
    e = replace_where(&e, &|a| is_function(a, &[Function::Sec]) && a.has(x), &|a| {
        div(&Basic::one(), &cos(&argument(a))?)
    })?;
    e = replace_where(&e, &|a| is_function(a, &[Function::Csc]) && a.has(x), &|a| {
        div(&Basic::one(), &sin(&argument(a))?)
    })?;
    crate::core::function::expand_with(
        &e,
        true,
        crate::core::function::Hints {
            power_exp: true,
            ..hints()
        },
    )
}

/// The relations of a formula (`_atoms`).
fn atoms(formula: &Basic) -> Vec<Basic> {
    formula.atoms(Some(&|n: &Basic| matches!(n.head(), Head::Relational(_))))
}

fn is_compound(b: &Basic) -> bool {
    matches!(
        b.head(),
        Head::And | Head::Or | Head::Not | Head::Implies | Head::Equivalent | Head::Xor | Head::Ite
    )
}

/// The formula with every relation mapped (`_map_atoms`).
fn map_atoms(formula: &Basic, f: &mut dyn FnMut(&Basic) -> R) -> R {
    if rel_op(formula).is_some() {
        return f(formula);
    }
    if is_compound(formula) {
        let mut args: Vec<Basic> = Vec::new();
        for a in formula.args() {
            args.push(map_atoms(a, f)?);
        }
        return crate::core::function::rebuild_evaluated(formula, args);
    }
    Ok(formula.clone())
}

/// The maximal transcendental subexpressions containing `x`
/// (`_kernels`).
fn kernels(e: &Basic, x: &Basic) -> Vec<Basic> {
    fn visit(a: &Basic, x: &Basic, found: &mut Vec<Basic>) {
        if !a.has(x) {
            return;
        }
        let rational_power: bool = a.is_pow() && a.args()[1].is_rational_class() && !a.args()[1].is_integer_value();
        if is_function(a, &[Function::Exp, Function::Log, Function::Sin, Function::Cos]) || rational_power {
            if !found.contains(a) {
                found.push(a.clone());
            }
            return;
        }
        for b in a.args() {
            visit(b, x, found);
        }
    }
    let mut found: Vec<Basic> = Vec::new();
    visit(e, x, &mut found);
    found
}

fn lcm(values: &[Int]) -> Int {
    crate::core::intfunc::ilcm(values)
}

fn ratio(a: &Basic, b: &Basic) -> PolyResult<Option<Rat>> {
    let r: Basic = cancel(&div(a, b)?)?;
    Ok(if r.is_rational_class() {
        r.as_rational().cloned()
    } else {
        None
    })
}

/// `u` and the integers `m_i` with `arguments[i] == m_i*u`
/// (`_common_argument`).
fn common_argument(arguments: &[Basic]) -> PolyResult<Option<(Basic, Vec<Int>)>> {
    let first: &Basic = &arguments[0];
    let mut ratios: Vec<Rat> = Vec::new();
    for a in arguments {
        let Some(r) = ratio(a, first)? else { return Ok(None) };
        ratios.push(r);
    }
    let mut dens: Vec<Int> = vec![Int::ONE];
    dens.extend(ratios.iter().map(|r| Int::from(denom(r))));
    let d: Int = lcm(&dens);
    let u: Basic = div(first, &Basic::integer(d.clone()))?;
    let multiples: Vec<Int> = ratios
        .iter()
        .map(|r| numer(r) * d.clone() / Int::from(denom(r)))
        .collect();
    Ok(Some((u, multiples)))
}

fn rel_parts(atom: &Basic) -> (RelOp, Basic, Basic) {
    (
        rel_op(atom).expect("a relation"),
        atom.args()[0].clone(),
        atom.args()[1].clone(),
    )
}

/// The formula as a polynomial formula in a kernel variable, when `x`
/// occurs only inside elementary functions of one family with
/// proportional arguments (`polynomialize`).
pub fn polynomialize(formula: &Basic, x: &Basic) -> PolyResult<Option<Reduction>> {
    let rewritten: Basic = map_atoms(formula, &mut |a| {
        let (op, l, r) = rel_parts(a);
        relation(op, &rewrite(&l, x)?, &rewrite(&r, x)?)
    })?;
    let mut ks: Vec<Basic> = Vec::new();
    for atom in atoms(&rewritten) {
        let (_, l, r) = rel_parts(&atom);
        for k in kernels(&sub(&l, &r)?, x) {
            if !ks.contains(&k) {
                ks.push(k);
            }
        }
    }
    if ks.is_empty() {
        return Ok(None);
    }
    let family_of = |k: &Basic| match k.head() {
        Head::Function(Function::Exp) => Family::Exp,
        Head::Function(Function::Log) => Family::Log,
        Head::Function(Function::Sin) | Head::Function(Function::Cos) => Family::Trig,
        _ => Family::Root,
    };
    let family: Family = family_of(&ks[0]);
    if ks.iter().any(|k| family_of(k) != family) {
        return Ok(None);
    }
    if family == Family::Root {
        return polynomialize_roots(&rewritten, &ks, x);
    }
    if family == Family::Log {
        let arguments: Vec<Basic> = ks.iter().map(argument).collect();
        if arguments.iter().any(|a| *a != arguments[0]) {
            return Ok(None);
        }
        let variable: Basic = dummy("l", &[(Predicate::real, true)]);
        let reduced: Basic = xreplace(&rewritten, &ks.iter().map(|k| (k.clone(), variable.clone())).collect())?;
        if reduced.has(x) {
            return Ok(None);
        }
        return Ok(Some(Reduction {
            formula: reduced,
            variable,
            argument: arguments[0].clone(),
            family: Family::Log,
            order: 1,
            special: false_(),
        }));
    }
    let arguments: Vec<Basic> = ks.iter().map(argument).collect();
    let Some((u, multiples)) = common_argument(&arguments)? else {
        return Ok(None);
    };
    if family == Family::Exp {
        let t: Basic = dummy("t", &[(Predicate::positive, true)]);
        let mut rep: HashMap<Basic, Basic> = HashMap::new();
        for (k, m) in ks.iter().zip(&multiples) {
            rep.insert(k.clone(), pow(&t, &Basic::integer(m.clone()))?);
        }
        let reduced: Basic = xreplace(&rewritten, &rep)?;
        if reduced.has(x) {
            return Ok(None);
        }
        return clear(&reduced, &t, Family::Exp, &u, 1, &false_());
    }
    // sines and cosines of integer multiples of u, then the half-angle
    // substitution
    let w: Basic = dummy("w", &[(Predicate::real, true)]);
    let c: Basic = dummy("c", &[]);
    let s: Basic = dummy("s", &[]);
    let v: Basic = dummy("v", &[]);
    let mut reduced: Basic = rewritten.clone();
    for (k, m) in ks.iter().zip(&multiples) {
        let multiple: Basic = mul2(&Basic::integer(m.clone()), &v)?;
        let base: Basic = if is_function(k, &[Function::Sin]) {
            sin(&multiple)?
        } else {
            cos(&multiple)?
        };
        let image: Basic =
            crate::core::function::expand_with(&base, true, crate::core::function::Hints { trig: true, ..hints() })?;
        let image: Basic = xreplace(&image, &rule(&[(cos(&v)?, c.clone()), (sin(&v)?, s.clone())]))?;
        reduced = xreplace(&reduced, &rule(&[(k.clone(), image)]))?;
    }
    if reduced.has(x) {
        return Ok(None);
    }
    let special: Basic = xreplace(
        &reduced,
        &rule(&[(c.clone(), Basic::negative_one()), (s.clone(), Basic::zero())]),
    )?;
    if !(is_true(&special) || is_false(&special)) {
        return Ok(None);
    }
    let w2: Basic = pow(&w, &int(2))?;
    let one_plus: Basic = add2(&Basic::one(), &w2)?;
    let cv: Basic = div(&sub(&Basic::one(), &w2)?, &one_plus)?;
    let sv: Basic = div(&mul2(&int(2), &w)?, &one_plus)?;
    reduced = xreplace(&reduced, &rule(&[(c, cv), (s, sv)]))?;
    clear(&reduced, &w, Family::Trig, &u, 1, &special)
}

fn polynomialize_roots(formula: &Basic, ks: &[Basic], x: &Basic) -> PolyResult<Option<Reduction>> {
    let mut bases: Vec<Basic> = Vec::new();
    let mut orders: Vec<Int> = Vec::new();
    for k in ks {
        if !k.is_pow() || !k.args()[1].is_rational_class() {
            return Ok(None);
        }
        bases.push(k.args()[0].clone());
        orders.push(Int::from(denom(k.args()[1].as_rational().expect("a rational"))));
    }
    if bases.iter().any(|b| *b != bases[0]) {
        return Ok(None);
    }
    let u: Basic = bases[0].clone();
    let mut all: Vec<Int> = vec![Int::ONE];
    all.extend(orders);
    let q: Int = lcm(&all);
    let y: Basic = dummy("y", &[(Predicate::nonnegative, true)]);
    let mut rep: HashMap<Basic, Basic> = HashMap::new();
    for k in ks {
        let e: &Rat = k.args()[1].as_rational().expect("a rational");
        let p_: Int = numer(e) * q.clone() / Int::from(denom(e));
        rep.insert(k.clone(), pow(&y, &Basic::integer(p_))?);
    }
    let mut reduced: Basic = xreplace(formula, &rep)?;
    if reduced.has(x) {
        let Some(linear) = coefficients(&u, x)? else {
            return Ok(None);
        };
        let degrees: Vec<i64> = linear.iter().map(|(d, _)| *d).collect();
        let shape_ok: bool = {
            let mut sorted = degrees.clone();
            sorted.sort();
            sorted == vec![0, 1] || sorted == vec![1]
        };
        if !shape_ok || !linear.iter().all(|(_, c)| c.is_rational_class()) {
            return Ok(None);
        }
        let a: Basic = coefficient_at(&linear, 1).expect("a linear term").clone();
        let b: Basic = coefficient_at(&linear, 0).cloned().unwrap_or_else(Basic::zero);
        reduced = xreplace(
            &reduced,
            &rule(&[(x.clone(), div(&sub(&pow(&y, &Basic::integer(q.clone()))?, &b)?, &a)?)]),
        )?;
        if reduced.has(x) {
            return Ok(None);
        }
    }
    let order: i64 = i64::try_from(&q).unwrap_or(1);
    clear(&reduced, &y, Family::Root, &u, order, &false_())
}

/// The denominators of the atoms cleared: divided out when positive,
/// otherwise by a case distinction on their sign (`_clear`).
fn clear(
    formula: &Basic,
    variable: &Basic,
    family: Family,
    u: &Basic,
    order: i64,
    special: &Basic,
) -> PolyResult<Option<Reduction>> {
    let polynomial = |e: &Basic| -> PolyResult<Option<Basic>> {
        match coefficients(e, variable)? {
            Some(cs) => Ok(Some(polynomial_expr(&cs, variable)?)),
            None => Ok(None),
        }
    };
    let mut not_polynomial: bool = false;
    let cleared: Basic = map_atoms(formula, &mut |atom| {
        let (op, l, r) = rel_parts(atom);
        let e: Basic = cancel(&sub(&l, &r)?)?;
        let (numerator, denominator) = crate::core::expr::as_numer_denom(&e)?;
        let Some(f) = polynomial(&numerator)? else {
            not_polynomial = true;
            return Ok(atom.clone());
        };
        if positive_denominator(&denominator, variable, family)? {
            return relation(op, &f, &Basic::zero());
        }
        let Some(g) = polynomial(&denominator)? else {
            not_polynomial = true;
            return Ok(atom.clone());
        };
        use crate::core::logic::{and_, or_};
        use crate::core::relational as rl;
        Ok(match op {
            RelOp::Eq => and_(&[rl::eq(&f, &Basic::zero())?, rl::ne(&g, &Basic::zero())?]),
            RelOp::Ne => and_(&[rl::ne(&f, &Basic::zero())?, rl::ne(&g, &Basic::zero())?]),
            _ => or_(&[
                and_(&[relation(op, &f, &Basic::zero())?, rl::gt(&g, &Basic::zero())?]),
                and_(&[relation(op, &neg(&f)?, &Basic::zero())?, rl::lt(&g, &Basic::zero())?]),
            ]),
        })
    })?;
    if not_polynomial {
        return Ok(None);
    }
    Ok(Some(Reduction {
        formula: cleared,
        variable: variable.clone(),
        argument: u.clone(),
        family,
        order,
        special: special.clone(),
    }))
}

/// Whether the denominator is a positive constant times a power of
/// `1 + w**2` (trig) or of the positive kernel variable (exp, root)
/// (`_positive_denominator`).
fn positive_denominator(d: &Basic, variable: &Basic, family: Family) -> PolyResult<bool> {
    if !d.has(variable) {
        return Ok(yes(d, Predicate::positive));
    }
    let Some(cs) = coefficients(d, variable)? else {
        return Ok(false);
    };
    match family {
        Family::Trig => {
            let mut degree: i64 = cs.iter().map(|(k, _)| *k).max().unwrap_or(0);
            let mut dense: Vec<Basic> = (0..=degree)
                .map(|k| coefficient_at(&cs, k).cloned().unwrap_or_else(Basic::zero))
                .collect();
            while degree > 0 {
                if degree == 1 {
                    return Ok(false);
                }
                let mut quotient: Vec<Basic> = vec![Basic::zero(); (degree - 1) as usize];
                for k in (2..=degree as usize).rev() {
                    quotient[k - 2] = dense[k].clone();
                    dense[k - 2] = expand(&sub(&dense[k - 2], &dense[k])?)?;
                }
                if dense[0] != Basic::zero() || dense[1] != Basic::zero() {
                    return Ok(false);
                }
                dense = quotient;
                degree = dense.len() as i64 - 1;
            }
            Ok(yes(&dense[0], Predicate::positive))
        }
        Family::Exp | Family::Root => {
            let values: Vec<&Basic> = cs.iter().map(|(_, c)| c).collect();
            Ok(values.iter().all(|c| yes(c, Predicate::nonnegative))
                && values.iter().any(|c| yes(c, Predicate::positive)))
        }
        Family::Log => Ok(false),
    }
}

// -- inverse images

/// Solves a formula polynomial in the given variable (`Solver`).
pub type Solver<'a> = &'a dyn Fn(&Basic, &Basic) -> PolyResult<Option<Basic>>;
/// Decides a condition on the parameters (`Decider`).
pub type Decider<'a> = &'a dyn Fn(&Basic) -> PolyResult<Option<bool>>;

/// The solver, the decider and the depth of the recursion (`_Context`).
#[derive(Clone, Copy)]
pub struct Context<'a> {
    pub solver: Option<Solver<'a>>,
    pub decide: Option<Decider<'a>>,
    pub depth: usize,
    pub lambert: bool,
}

impl<'a> Context<'a> {
    fn truth(&self, condition: &Basic) -> PolyResult<Option<bool>> {
        if is_true(condition) {
            return Ok(Some(true));
        }
        if is_false(condition) {
            return Ok(Some(false));
        }
        match self.decide {
            Some(d) => d(condition),
            None => Ok(None),
        }
    }
    fn deeper(&self) -> Context<'a> {
        Context {
            depth: self.depth + 1,
            ..*self
        }
    }
}

/// A set of values with the condition under which they are solutions.
type Conditioned = (Basic, Basic);

fn integer_dummy(name: &str) -> Basic {
    dummy(name, &[(Predicate::integer, true)])
}

fn image(expr: &Basic, k: &Basic) -> R {
    crate::sets::fancysets::image_set(
        &Basic::new(Head::Tuple, vec![k.clone()]),
        expr,
        &[Basic::atom(Head::Integers)],
    )
}

/// `log(8)/log(2)` as `3`, when expanding the logarithms makes the
/// expression smaller (`_simplify`).
fn simplify_logs(e: &Basic) -> R {
    let expanded: Basic = crate::core::function::expand_with(
        e,
        true,
        crate::core::function::Hints {
            log: true,
            force: true,
            ..hints()
        },
    )?;
    let candidate: Basic = cancel(&expanded)?;
    Ok(if operations(&candidate)? < operations(e)? {
        candidate
    } else {
        e.clone()
    })
}

fn set_parts(values: &Basic) -> Vec<Basic> {
    values.args().to_vec()
}

fn kernel_preimage(reduction: &Reduction, values: &Basic, ctx: &Context) -> PolyResult<Option<Vec<Conditioned>>> {
    use crate::core::relational as rl;
    let family: Family = reduction.family;
    if family == Family::Trig {
        let k: Basic = integer_dummy("k");
        let Some(regular) = trig_preimage(values)? else {
            return Ok(None);
        };
        let mut result: Vec<Conditioned> = vec![(regular, true_())];
        if is_true(&reduction.special) {
            let pi: Basic = Basic::pi();
            result.push((image(&add2(&pi, &mul2(&mul2(&int(2), &pi)?, &k)?)?, &k)?, true_()));
        }
        return Ok(Some(result));
    }
    if matches!(values.head(), Head::EmptySet) {
        return Ok(Some(vec![]));
    }
    if matches!(values.head(), Head::Union | Head::Intersection) {
        let mut parts: Vec<Conditioned> = Vec::new();
        for a in set_parts(values) {
            let Some(part) = kernel_preimage(reduction, &a, ctx)? else {
                return Ok(None);
            };
            parts.extend(part);
        }
        if matches!(values.head(), Head::Intersection) {
            if parts.iter().any(|(_, c)| !is_true(c)) {
                return Ok(None);
            }
            let sets: Vec<Basic> = parts.into_iter().map(|(s, _)| s).collect();
            return Ok(Some(vec![(intersection(&sets)?, true_())]));
        }
        return Ok(Some(parts));
    }
    let interval_like: bool = matches!(values.head(), Head::Interval | Head::Reals);
    if family == Family::Log {
        if let Some(elements) = finite_elements(values) {
            let images: Vec<Basic> = elements.iter().map(exp).collect::<PolyResult<_>>()?;
            return Ok(Some(vec![(finite_set(&images), true_())]));
        }
        if interval_like {
            let (a, b, lo, ro) = interval_parts(values);
            return Ok(Some(vec![(interval(&exp(&a)?, &exp(&b)?, lo, ro)?, true_())]));
        }
        return Ok(None);
    }
    let lower: Basic = Basic::zero();
    let strict: bool = family == Family::Exp;
    let q: i64 = reduction.order;
    let inverse = |v: &Basic| -> R {
        if strict {
            log(v)
        } else {
            pow(v, &int(q))
        }
    };
    if let Some(elements) = finite_elements(values) {
        let mut conditioned: Vec<Conditioned> = Vec::new();
        for v in elements {
            let condition: Basic = if strict {
                rl::gt(&v, &lower)?
            } else {
                rl::ge(&v, &lower)?
            };
            let truth: Option<bool> = ctx.truth(&condition)?;
            if truth == Some(false) {
                continue;
            }
            conditioned.push((
                finite_set(&[inverse(&v)?]),
                if truth == Some(true) { true_() } else { condition },
            ));
        }
        return Ok(Some(conditioned));
    }
    if interval_like {
        let (mut start, end, mut left_open, right_open) = interval_parts(values);
        let below: Option<bool> = ctx.truth(&rl::gt(&start, &lower)?)?;
        if below.is_none() && start != lower {
            return Ok(None);
        }
        if below != Some(true) {
            let open_at_bound: bool = start == lower && left_open;
            start = lower.clone();
            left_open = strict || open_at_bound;
        }
        if ctx.truth(&rl::gt(&end, &lower)?)? == Some(false)
            || (strict && ctx.truth(&rl::ge(&end, &lower)?)? == Some(false))
        {
            return Ok(Some(vec![]));
        }
        let oo: Basic = Basic::oo();
        if end != oo && ctx.truth(&rl::gt(&end, &lower)?)?.is_none() {
            return Ok(None);
        }
        if end == lower && (strict || right_open) {
            return Ok(Some(vec![]));
        }
        let image_end: Basic = if end == oo { oo.clone() } else { inverse(&end)? };
        if family == Family::Exp {
            let zero: bool = start == Basic::zero();
            let image_start: Basic = if zero { Basic::negative_oo() } else { inverse(&start)? };
            return Ok(Some(vec![(
                interval(&image_start, &image_end, zero || left_open, right_open)?,
                true_(),
            )]));
        }
        return Ok(Some(vec![(
            interval(&inverse(&start)?, &image_end, left_open, right_open)?,
            true_(),
        )]));
    }
    Ok(None)
}

/// `tan(u/2) = v  <=>  u = 2 atan(v) + 2 pi k` (`_trig_preimage`).
fn trig_preimage(values: &Basic) -> PolyResult<Option<Basic>> {
    if matches!(values.head(), Head::EmptySet) {
        return Ok(Some(empty_set()));
    }
    if matches!(values.head(), Head::Union | Head::Intersection) {
        let mut members: Vec<Basic> = Vec::new();
        for a in set_parts(values) {
            let Some(m) = trig_preimage(&a)? else { return Ok(None) };
            members.push(m);
        }
        return Ok(Some(if matches!(values.head(), Head::Union) {
            union(&members)?
        } else {
            intersection(&members)?
        }));
    }
    if let Some(elements) = finite_elements(values) {
        let k: Basic = integer_dummy("k");
        let pi: Basic = Basic::pi();
        let mut images: Vec<Basic> = Vec::new();
        for v in elements {
            let e: Basic = add2(&mul2(&int(2), &atan(&v)?)?, &mul2(&mul2(&int(2), &pi)?, &k)?)?;
            images.push(image(&e, &k)?);
        }
        return Ok(Some(if images.is_empty() {
            empty_set()
        } else {
            union(&images)?
        }));
    }
    Ok(None)
}

/// The set of `x` with `u(x)` in the set of values (`_argument_preimage`).
fn argument_preimage(u: &Basic, x: &Basic, values: &Basic, ctx: &Context) -> PolyResult<Option<Basic>> {
    use crate::core::relational as rl;
    if u == x {
        if let Some(elements) = finite_elements(values) {
            let simplified: Vec<Basic> = elements.iter().map(simplify_logs).collect::<PolyResult<_>>()?;
            return Ok(Some(finite_set(&simplified)));
        }
        return Ok(Some(values.clone()));
    }
    if matches!(values.head(), Head::EmptySet) {
        return Ok(Some(empty_set()));
    }
    if matches!(values.head(), Head::Union | Head::Intersection) {
        let mut members: Vec<Basic> = Vec::new();
        for a in set_parts(values) {
            let Some(m) = argument_preimage(u, x, &a, ctx)? else {
                return Ok(None);
            };
            members.push(m);
        }
        return Ok(Some(if matches!(values.head(), Head::Union) {
            union(&members)?
        } else {
            intersection(&members)?
        }));
    }
    if let Some(elements) = finite_elements(values) {
        let mut pieces: Vec<Basic> = Vec::new();
        for v in elements {
            let Some(mut piece) = solve_real(&rl::eq(u, &v)?, x, ctx)? else {
                return Ok(None);
            };
            if let Some(es) = finite_elements(&piece) {
                let simplified: Vec<Basic> = es.iter().map(simplify_logs).collect::<PolyResult<_>>()?;
                piece = finite_set(&simplified);
            }
            pieces.push(piece);
        }
        return Ok(Some(if pieces.is_empty() {
            empty_set()
        } else {
            union(&pieces)?
        }));
    }
    if matches!(values.head(), Head::Interval | Head::Reals) {
        let (start, end, lo, ro) = interval_parts(values);
        let mut conditions: Vec<Basic> = Vec::new();
        if !matches!(start.head(), Head::NegativeInfinity) {
            conditions.push(if lo { rl::gt(u, &start)? } else { rl::ge(u, &start)? });
        }
        if !matches!(end.head(), Head::Infinity) {
            conditions.push(if ro { rl::lt(u, &end)? } else { rl::le(u, &end)? });
        }
        return solve_real(&crate::core::logic::and_(&conditions), x, ctx);
    }
    if matches!(values.head(), Head::ImageSet) {
        let (variables, image_expr, base_sets) = (
            values.args()[0].args().to_vec(),
            values.args()[1].clone(),
            values.args()[2..].to_vec(),
        );
        if variables.len() != 1 || base_sets.len() != 1 || !matches!(base_sets[0].head(), Head::Integers) {
            return Ok(None);
        }
        let k: Basic = variables[0].clone();
        let Some(inner) = solve_real(&rl::eq(u, &image_expr)?, x, ctx)? else {
            return Ok(None);
        };
        return parametric_union(&inner, &k);
    }
    Ok(None)
}

/// A condition on `k` for `e` to be real (`_real_condition`).
fn real_condition(e: &Basic, k: &Basic) -> R {
    let _ = k;
    let (mut base, mut power) = crate::core::expr::as_base_exp(e)?;
    if crate::core::printing::could_extract_minus_sign(e) {
        (base, power) = crate::core::expr::as_base_exp(&neg(e)?)?;
    }
    if power.as_rational().is_some_and(|r| denom(r) == Int::from(2u8)) && !base.has(&Basic::imaginary_unit()) {
        return crate::core::relational::ge(&base, &Basic::zero());
    }
    crate::sets::sets::contains(&crate::sets::sets::reals(), e)
}

/// The union over the integers `k` of a set depending on `k`
/// (`_parametric_union`).
fn parametric_union(solutions: &Basic, k: &Basic) -> PolyResult<Option<Basic>> {
    if let Some(elements) = finite_elements(solutions) {
        let images: Vec<Basic> = elements.iter().map(|e| image(e, k)).collect::<PolyResult<_>>()?;
        return Ok(Some(union(&images)?));
    }
    if matches!(solutions.head(), Head::EmptySet) {
        return Ok(Some(solutions.clone()));
    }
    if matches!(solutions.head(), Head::Union) {
        let mut members: Vec<Basic> = Vec::new();
        for a in set_parts(solutions) {
            let Some(m) = parametric_union(&a, k)? else {
                return Ok(None);
            };
            members.push(m);
        }
        return Ok(Some(union(&members)?));
    }
    let reals: Basic = crate::sets::sets::reals();
    if matches!(solutions.head(), Head::Intersection)
        && solutions.args().len() == 2
        && solutions.args().contains(&reals)
    {
        let other: Basic = solutions
            .args()
            .iter()
            .find(|a| **a != reals)
            .expect("the other set")
            .clone();
        if let Some(elements) = finite_elements(&other) {
            let mut images: Vec<Basic> = Vec::new();
            for e in elements {
                let condition: Basic = real_condition(&e, k)?;
                let integers: Basic = Basic::atom(Head::Integers);
                let base: Basic = if is_true(&condition) {
                    integers
                } else {
                    crate::sets::conditionset::condition_set(k, &condition, &integers)?
                };
                images.push(crate::sets::fancysets::image_set(
                    &Basic::new(Head::Tuple, vec![k.clone()]),
                    &e,
                    &[base],
                )?);
            }
            return Ok(Some(union(&images)?));
        }
    }
    Ok(None)
}

/// Polynomial in `x` with rational coefficients (`_polynomial`).
fn polynomial(formula: &Basic, x: &Basic) -> PolyResult<bool> {
    for atom in atoms(formula) {
        let (_, l, r) = rel_parts(&atom);
        match coefficients(&sub(&l, &r)?, x)? {
            Some(cs) if cs.iter().all(|(_, c)| yes(c, Predicate::rational)) => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// The real solutions of a relation linear in `x` with a real leading
/// coefficient of known sign (`_linear_solution`).
pub fn linear_solution(rel: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let (op, l, r) = rel_parts(rel);
    let e: Basic = sub(&l, &r)?;
    let Some(cs) = coefficients(&e, x)? else {
        return Ok(None);
    };
    if cs.iter().any(|(d, _)| *d != 0 && *d != 1) || coefficient_at(&cs, 1).is_none() {
        return Ok(None);
    }
    let a: Basic = coefficient_at(&cs, 1).expect("linear").clone();
    let b: Basic = crate::core::subs::subs(&e, x, &Basic::zero())?;
    if !(crate::core::assumptions::is_number(&a) && yes(&a, Predicate::real) && yes(&b, Predicate::real)) {
        return Ok(None);
    }
    let op: RelOp = if yes(&a, Predicate::positive) {
        op
    } else if yes(&a, Predicate::negative) {
        match op {
            RelOp::Lt => RelOp::Gt,
            RelOp::Le => RelOp::Ge,
            RelOp::Gt => RelOp::Lt,
            RelOp::Ge => RelOp::Le,
            other => other,
        }
    } else {
        return Ok(None);
    };
    let root: Basic = div(&neg(&b)?, &a)?;
    let (oo, moo) = (Basic::oo(), Basic::negative_oo());
    Ok(Some(match op {
        RelOp::Eq => finite_set(&[root]),
        RelOp::Ne => union(&[interval(&moo, &root, true, true)?, interval(&root, &oo, true, true)?])?,
        RelOp::Gt | RelOp::Ge => interval(&root, &oo, op == RelOp::Gt, true)?,
        _ => interval(&moo, &root, true, op == RelOp::Lt)?,
    }))
}

fn cad_solution(formula: &Basic, x: &Basic) -> R {
    crate::polys::roots::in_radicals(&crate::polys::cad::qe::solution_set(formula, x, &[], None, true)?)
}

/// The real solutions of a formula in `x` (`_solve_real`).
fn solve_real(formula: &Basic, x: &Basic, ctx: &Context) -> PolyResult<Option<Basic>> {
    if is_true(formula) {
        return Ok(Some(crate::sets::sets::reals()));
    }
    if is_false(formula) {
        return Ok(Some(empty_set()));
    }
    if !formula.has(x) {
        return Ok(None);
    }
    if polynomial(formula, x)? {
        return Ok(Some(cad_solution(formula, x)?));
    }
    let all_polynomial: bool = atoms(formula).iter().all(|a| {
        let (_, l, r) = rel_parts(a);
        sub(&l, &r).ok().and_then(|d| {
            crate::core::expr::polynomial_like(
                &d,
                std::slice::from_ref(x),
                crate::core::expr::PolynomialKind::Polynomial,
            )
        }) == Some(true)
    });
    if all_polynomial {
        if let Some(solver) = ctx.solver {
            return solver(formula, x);
        }
        if rel_op(formula).is_some() {
            if let Some(linear) = linear_solution(formula, x)? {
                return Ok(Some(linear));
            }
            return Ok(Some(super::solveset::solveset(
                formula,
                x,
                &crate::sets::sets::reals(),
            )?));
        }
        if matches!(formula.head(), Head::And | Head::Or) {
            let mut parts: Vec<Basic> = Vec::new();
            for a in formula.args() {
                let Some(part) = solve_real(a, x, ctx)? else {
                    return Ok(None);
                };
                parts.push(part);
            }
            return Ok(Some(if matches!(formula.head(), Head::And) {
                intersection(&parts)?
            } else {
                union(&parts)?
            }));
        }
        return Ok(None);
    }
    if ctx.depth > 3 {
        return Ok(None);
    }
    solve_transcendental_inner(formula, x, &ctx.deeper())
}

fn lambert_branches(v: &Basic) -> PolyResult<Vec<Basic>> {
    use crate::core::relational as rl;
    let mut result: Vec<Basic> = vec![v.clone()];
    for w in v.atoms(Some(&|n: &Basic| {
        matches!(n.head(), Head::Function(Function::LambertW))
    })) {
        if w.args().len() != 1 {
            continue;
        }
        let a: Basic = w.args()[0].clone();
        let bound: Basic = neg(&exp(&Basic::negative_one())?)?;
        if crate::core::assumptions::is_number(&a)
            && yes(&a, Predicate::real)
            && is_true(&rl::lt(&a, &Basic::zero())?)
            && is_true(&rl::ge(&a, &bound)?)
        {
            let lw: Basic = crate::functions::elementary::exponential::lambertw(&a, Some(&Basic::negative_one()))?;
            result.push(xreplace(v, &rule(&[(w.clone(), lw)]))?);
        }
    }
    Ok(result)
}

/// Whether the candidate is a real root of `e` (`_real_root`).
fn real_root(e: &Basic, x: &Basic, v: &Basic) -> PolyResult<Option<bool>> {
    let settings = crate::settings::settings();
    let mut v: Basic = v.clone();
    if settings.numerical_checks {
        match crate::numeric::reliable_form(&v, settings.precision) {
            Some(f) => v = f,
            None => return Ok(None),
        }
    }
    let mut real: Option<bool> = is(&v, Predicate::real);
    if real.is_none() && settings.numerical_checks {
        real = crate::numeric::reliable_value(&v, settings.precision, &[]).map(|z| {
            let (_, im) = crate::core::evalf::parts(&z);
            crate::assumptions::ivarith::real_to_rat(&im).is_zero()
        });
    }
    if real == Some(false) {
        return Ok(Some(false));
    }
    if !settings.numerical_checks {
        return Ok(None);
    }
    let residual = crate::numeric::reliable_value(&crate::core::subs::subs(e, x, &v)?, settings.precision, &[]);
    if let Some(z) = residual {
        let (re, im) = crate::core::evalf::parts(&z);
        let (a, b) = (
            crate::assumptions::ivarith::real_to_rat(&re),
            crate::assumptions::ivarith::real_to_rat(&im),
        );
        let threshold: Rat = Rat::from_parts(Int::ONE, dashu_int::UBig::from(10u8).pow(settings.precision * 2 / 3));
        // |residual| > threshold, compared squared
        if &a * &a + &b * &b > &threshold * &threshold {
            return Ok(Some(false));
        }
        return Ok(Some(true));
    }
    Ok(None)
}

/// The Lambert W candidates of an equation with `x` inside and outside an
/// exponential or a logarithm (`_lambert`).
fn lambert(formula: &Basic, x: &Basic, ctx: &Context) -> PolyResult<Option<Basic>> {
    if !ctx.lambert || rel_op(formula) != Some(RelOp::Eq) {
        return Ok(None);
    }
    let (_, l, r) = rel_parts(formula);
    let e: Basic = sub(&l, &r)?;
    if e.free_symbols().iter().any(|s| s != x) {
        return Ok(None);
    }
    let has_exp_log = e
        .preorder()
        .iter()
        .any(|n| matches!(n.head(), Head::Function(Function::Exp) | Head::Function(Function::Log)));
    if !has_exp_log {
        return Ok(None);
    }
    let found: Option<Vec<Basic>> = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        super::solvers::solve_equation(&e, x)
    })?
    .flatten();
    let Some(found) = found.filter(|f| !f.is_empty()) else {
        return Ok(None);
    };
    let mut candidates: Vec<Basic> = Vec::new();
    for v in found {
        let has_w = v
            .preorder()
            .iter()
            .any(|n| matches!(n.head(), Head::Function(Function::LambertW)));
        if v.has(x) || !has_w {
            continue;
        }
        for candidate in lambert_branches(&v)? {
            if real_root(&e, x, &candidate)? != Some(false) && !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    Ok(Some(finite_set(&candidates)))
}

/// Absolute values of expressions in `x` removed by case distinctions
/// (`_without_abs`).
fn without_abs(formula: &Basic, x: &Basic) -> R {
    use crate::core::logic::{and_, or_};
    use crate::core::relational as rl;
    let mut formula: Basic = formula.clone();
    loop {
        let found: Vec<Basic> = formula
            .atoms(Some(&|n: &Basic| matches!(n.head(), Head::Function(Function::Abs))))
            .into_iter()
            .filter(|a| a.has(x))
            .collect();
        let Some(a) = found.first() else { return Ok(formula) };
        let inner: Basic = argument(a);
        formula = or_(&[
            and_(&[
                rl::ge(&inner, &Basic::zero())?,
                xreplace(&formula, &rule(&[(a.clone(), inner.clone())]))?,
            ]),
            and_(&[
                rl::lt(&inner, &Basic::zero())?,
                xreplace(&formula, &rule(&[(a.clone(), neg(&inner)?)]))?,
            ]),
        ]);
    }
}

/// The real solutions of a transcendental formula in `x`, `None` when it
/// is not reducible to a polynomial one (`solve_transcendental`).
pub fn solve_transcendental(
    formula: &Basic,
    x: &Basic,
    solver: Option<Solver>,
    decide: Option<Decider>,
    lambert: bool,
) -> PolyResult<Option<Basic>> {
    solve_transcendental_inner(
        formula,
        x,
        &Context {
            solver,
            decide,
            depth: 0,
            lambert,
        },
    )
}

/// A formula rational in `x` with rational coefficients as a polynomial
/// one (`_cleared`).
fn cleared(formula: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let rational = |e: &Basic| -> PolyResult<bool> {
        Ok(matches!(coefficients(e, x)?, Some(cs) if cs.iter().all(|(_, c)| yes(c, Predicate::rational))))
    };
    let mut rep: HashMap<Basic, Basic> = HashMap::new();
    for atom in atoms(formula) {
        let (op, l, r) = rel_parts(&atom);
        let (numerator, denominator) =
            crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&sub(&l, &r)?, false, true)?)?;
        if !rational(&numerator)? || !rational(&denominator)? {
            return Ok(None);
        }
        let rel: Basic = relation(op, &expand(&mul2(&numerator, &denominator)?)?, &Basic::zero())?;
        let value: Basic = if denominator.has(x) {
            crate::core::logic::and_(&[rel, crate::core::relational::ne(&denominator, &Basic::zero())?])
        } else {
            rel
        };
        rep.insert(atom, value);
    }
    Ok(Some(xreplace(formula, &rep)?))
}

fn solve_transcendental_inner(formula: &Basic, x: &Basic, ctx: &Context) -> PolyResult<Option<Basic>> {
    let mut formula: Basic = without_abs(formula, x)?;
    formula = map_atoms(&formula, &mut |a| {
        let (op, l, r) = rel_parts(a);
        relation(op, &l, &r)
    })?;
    if is_false(&formula) {
        return Ok(Some(empty_set()));
    }
    if is_true(&formula) {
        return Ok(Some(crate::sets::sets::reals()));
    }
    if polynomial(&formula, x)? {
        return Ok(Some(cad_solution(&formula, x)?));
    }
    if let Some(rational) = cleared(&formula, x)? {
        if polynomial(&rational, x)? {
            return Ok(Some(cad_solution(&rational, x)?));
        }
    }
    let Some(reduction) = polynomialize(&formula, x)? else {
        return lambert(&formula, x, ctx);
    };
    use crate::core::relational as rl;
    let side: Basic = match reduction.family {
        Family::Exp => rl::gt(&reduction.variable, &Basic::zero())?,
        Family::Root => rl::ge(&reduction.variable, &Basic::zero())?,
        _ => true_(),
    };
    let target: Basic = crate::core::logic::and_(&[reduction.formula.clone(), side]);
    let Some(values) = solve_real(&target, &reduction.variable, ctx)? else {
        return Ok(None);
    };
    let Some(conditioned) = kernel_preimage(&reduction, &values, ctx)? else {
        return Ok(None);
    };
    let mut results: Vec<Basic> = Vec::new();
    for (argument_values, condition) in conditioned {
        let Some(result) = argument_preimage(&reduction.argument, x, &argument_values, ctx)? else {
            return Ok(None);
        };
        results.push(if is_true(&condition) {
            result
        } else {
            crate::sets::conditionset::condition_set(x, &condition, &result)?
        });
    }
    Ok(Some(if results.is_empty() {
        empty_set()
    } else {
        union(&results)?
    }))
}
