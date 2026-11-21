//! Port of `sympy_extras/integrals/meijerint.py`: integration by rewriting
//! the integrand as Meijer G-functions. A function is written as a sum of
//! terms `C x^s G(a x^b)` by a table of representations of the elementary
//! and special functions, by the recursive Mellin transform (the theorem
//! for two G-functions, inverted) or by the Mellin transforms of the
//! kernels of [`crate::integrals::mellin`]; the integrals are theorems on
//! G-functions (the indefinite integral, the integral over `(0, oo)` of
//! one or of the product of two, the inverse Laplace transform).
//!
//! The G-functions are `MeijerG` nodes; the table entries are records
//! whose computed terms and hints are enum cases. Python's binary
//! operators evaluate at each step: the expressions are built with the
//! binary helpers in the same association. The cardinal sine has no head
//! in the Rust core: its entry is left out.

#![allow(clippy::mutable_key_type)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::core::symbol::{Dummy, Wild};
use crate::error::{PolyError, PolyResult};
use crate::functions::special::hyper::{meijerg_delta, meijerg_get_period, meijerg_lists, meijerg_nu};
use crate::numbers::{Int, Rat};

type R = PolyResult<Basic>;

// ---------------------------------------------------------------------
// expressions with Python's binary operators

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    Basic::zero()
}
fn one() -> Basic {
    Basic::one()
}
fn half() -> Basic {
    Basic::half()
}
fn rational(p: i64, q: i64) -> Basic {
    Basic::from_rational(crate::numbers::rat(p, q))
}
fn pi() -> Basic {
    Basic::pi()
}
fn imag() -> Basic {
    Basic::atom(Head::ImaginaryUnit)
}
fn oo() -> Basic {
    Basic::atom(Head::Infinity)
}
fn neg_oo() -> Basic {
    Basic::atom(Head::NegativeInfinity)
}
fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
fn p2(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
fn ng(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn add_n(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn mul_n(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn sqrt(a: &Basic) -> R {
    p2(a, &half())
}
fn apply(f: Function, args: &[Basic]) -> R {
    crate::functions::apply_function(f, args)
}
fn f1(f: Function, a: &Basic) -> R {
    apply(f, std::slice::from_ref(a))
}
fn gamma(a: &Basic) -> R {
    f1(Function::Gamma, a)
}
fn exp(a: &Basic) -> R {
    crate::functions::elementary::exponential::exp(a)
}
fn log(a: &Basic) -> R {
    crate::functions::elementary::exponential::log(a)
}
fn sin(a: &Basic) -> R {
    f1(Function::Sin, a)
}
fn cos(a: &Basic) -> R {
    f1(Function::Cos, a)
}
fn re(a: &Basic) -> R {
    crate::functions::elementary::complexes::re(a)
}
fn im(a: &Basic) -> R {
    crate::functions::elementary::complexes::im(a)
}
fn abs_(a: &Basic) -> R {
    crate::functions::elementary::complexes::abs_(a)
}
fn sign(a: &Basic) -> R {
    crate::functions::elementary::complexes::sign(a)
}
fn arg_(a: &Basic) -> R {
    crate::functions::elementary::argument::arg(a)
}
fn heaviside(a: &Basic) -> R {
    crate::functions::special::delta_functions::heaviside(a, None)
}
fn exp_polar(a: &Basic) -> R {
    crate::functions::elementary::polar::exp_polar(a)
}
fn polar_lift(a: &Basic) -> R {
    crate::functions::elementary::polar::polar_lift(a)
}
fn unbranched_argument(a: &Basic) -> R {
    crate::functions::elementary::polar::unbranched_argument(a)
}
fn unpolarify(e: &Basic, exponents_only: bool) -> R {
    crate::functions::elementary::polar::unpolarify(e, exponents_only)
}
/// `polarify(e, subs=False, lift)[0]`.
fn polarify(e: &Basic, lift: bool) -> R {
    Ok(crate::functions::elementary::polar::polarify(e, false, lift)?.0)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn subs_map(e: &Basic, pairs: &[(Basic, Basic)]) -> R {
    crate::core::subs::subs_mapping(e, pairs)
}
fn xreplace(e: &Basic, rule: &[(Basic, Basic)]) -> R {
    if rule.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn expand_mul(e: &Basic) -> R {
    crate::core::function::expand_mul(e, true)
}
fn hints(power_base: bool, trig: bool) -> crate::core::function::Hints {
    crate::core::function::Hints {
        mul: false,
        multinomial: false,
        power_base,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig,
        func: false,
    }
}
fn expand_power_base(e: &Basic) -> R {
    crate::core::function::expand_with(e, true, hints(true, false))
}
fn expand_trig(e: &Basic) -> R {
    crate::core::function::expand_with(e, true, hints(false, true))
}
fn powsimp(e: &Basic) -> R {
    crate::simplify::powsimp::powsimp_default(e)
}
fn powdenest_polar(e: &Basic) -> R {
    crate::simplify::powsimp::powdenest(e, false, true)
}
fn hyperexpand(e: &Basic) -> R {
    crate::simplify::hyperexpand::hyperexpand(e, false, crate::simplify::hyperexpand::Rewrite::Default, None)
}
fn gammasimp(e: &Basic) -> R {
    crate::simplify::gammasimp::gammasimp(e)
}
fn cancel(e: &Basic) -> R {
    crate::polys::polytools::cancel(e, &[])
}
fn factor(e: &Basic) -> R {
    crate::polys::polytools::factor(e, &[], false, true)
}
fn factor_terms(e: &Basic) -> R {
    crate::core::exprtools::factor_terms(e, false, false, false, true)
}
fn piecewise_fold(e: &Basic, evaluate: Option<bool>) -> R {
    crate::functions::elementary::piecewise::piecewise_fold(e, evaluate)
}
fn piecewise(pairs: &[(Basic, Basic)]) -> R {
    crate::functions::elementary::piecewise::piecewise(pairs)
}
fn sort_key(e: &Basic) -> crate::core::sorting::SortKey {
    crate::core::sorting::sort_key(e)
}
fn sorted(mut v: Vec<Basic>) -> Vec<Basic> {
    v.sort_by_cached_key(sort_key);
    v
}
fn true_() -> Basic {
    crate::core::logic::true_()
}
fn false_() -> Basic {
    crate::core::logic::false_()
}
fn boolean(b: bool) -> Basic {
    crate::core::logic::boolean(b)
}
fn and_(v: &[Basic]) -> Basic {
    crate::core::logic::and_(v)
}
fn or_(v: &[Basic]) -> Basic {
    crate::core::logic::or_(v)
}
fn eq(a: &Basic, b: &Basic) -> R {
    crate::core::relational::eq(a, b)
}
fn ne(a: &Basic, b: &Basic) -> R {
    crate::core::relational::ne(a, b)
}
fn lt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::lt(a, b)
}
fn le(a: &Basic, b: &Basic) -> R {
    crate::core::relational::le(a, b)
}
fn gt(a: &Basic, b: &Basic) -> R {
    crate::core::relational::gt(a, b)
}
fn ge(a: &Basic, b: &Basic) -> R {
    crate::core::relational::ge(a, b)
}
fn is_true(b: &Basic) -> bool {
    *b == true_()
}
fn is_false(b: &Basic) -> bool {
    *b == false_()
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn is_fn(e: &Basic, fs: &[Function]) -> bool {
    matches!(func(e), Some(f) if fs.contains(&f))
}
fn is_meijerg(e: &Basic) -> bool {
    matches!(e.head(), Head::MeijerG)
}
fn is_rational(e: &Basic) -> bool {
    e.as_rational().is_some()
}
fn is_integer_node(e: &Basic) -> bool {
    e.as_rational().is_some_and(|q| q.is_int())
}
fn int_value(e: &Basic) -> Option<i64> {
    e.as_rational()
        .filter(|q| q.is_int())
        .and_then(|q| i64::try_from(crate::numbers::numer(q)).ok())
}
fn rat_of(e: &Basic) -> Option<Rat> {
    e.as_rational().cloned()
}
fn mul_args(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn add_args(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}
fn has_any(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> bool {
    e.preorder().iter().any(p)
}
fn has_heads(e: &Basic, heads: &[Function]) -> bool {
    has_any(e, &|n| is_fn(n, heads))
}
fn has_x(e: &Basic, x: &Basic) -> bool {
    e.has(x)
}
fn free_of(e: &Basic, x: &Basic) -> bool {
    !e.free_symbols().contains(x)
}
fn atoms_where(e: &Basic, p: &dyn Fn(&Basic) -> bool) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for n in e.preorder() {
        if p(&n) && !found.contains(&n) {
            found.push(n);
        }
    }
    found
}
fn value_error(message: &str) -> PolyError {
    PolyError::Value(message.to_string())
}
fn type_error(message: &str) -> PolyError {
    PolyError::Type(message.to_string())
}
fn is_value_error(e: &PolyError) -> bool {
    matches!(e, PolyError::Value(_))
}
/// The classes of `polys.polyerrors` (`BasePolynomialError`).
fn is_polynomial_error(e: &PolyError) -> bool {
    matches!(e, PolyError::Polys(..) | PolyError::HeuristicGCDFailed(_))
}
const HYPERBOLIC: [Function; 6] = [
    Function::Sinh,
    Function::Cosh,
    Function::Tanh,
    Function::Coth,
    Function::Sech,
    Function::Csch,
];
fn has_infinity(e: &Basic) -> bool {
    has_any(e, &|n| {
        matches!(
            n.head(),
            Head::Infinity | Head::ComplexInfinity | Head::NegativeInfinity
        )
    })
}
fn has_nan_zoo(e: &Basic) -> bool {
    has_any(e, &|n| matches!(n.head(), Head::NaN | Head::ComplexInfinity))
}

// ---------------------------------------------------------------------
// G-functions

/// The parameter lists `an`, `aother`, `bm`, `bother` of a `MeijerG`.
pub fn lists(g: &Basic) -> [Vec<Basic>; 4] {
    let [an, aother, bm, bother] = meijerg_lists(g);
    [an.to_vec(), aother.to_vec(), bm.to_vec(), bother.to_vec()]
}
fn g_an(g: &Basic) -> Vec<Basic> {
    meijerg_lists(g)[0].to_vec()
}
fn g_aother(g: &Basic) -> Vec<Basic> {
    meijerg_lists(g)[1].to_vec()
}
fn g_bm(g: &Basic) -> Vec<Basic> {
    meijerg_lists(g)[2].to_vec()
}
fn g_bother(g: &Basic) -> Vec<Basic> {
    meijerg_lists(g)[3].to_vec()
}
fn g_ap(g: &Basic) -> Vec<Basic> {
    let l = meijerg_lists(g);
    l[0].iter().chain(l[1]).cloned().collect()
}
fn g_bq(g: &Basic) -> Vec<Basic> {
    let l = meijerg_lists(g);
    l[2].iter().chain(l[3]).cloned().collect()
}
fn g_argument(g: &Basic) -> Basic {
    g.args()[2].clone()
}

/// `meijerg` as a G-function node (`ValueError` on parameters which the
/// constructor refuses), the parameters unpolarified (`_g`).
pub fn g_(an: &[Basic], aother: &[Basic], bm: &[Basic], bother: &[Basic], argument: &Basic) -> R {
    let prep = |params: &[Basic]| -> PolyResult<Vec<Basic>> { params.iter().map(|p| unpolarify(p, false)).collect() };
    crate::functions::special::hyper::meijerg(&prep(an)?, &prep(aother)?, &prep(bm)?, &prep(bother)?, argument)
}

/// `g` with the bindings substituted (`_subs_g`).
fn subs_g(g: &Basic, rule: &[(Basic, Basic)]) -> R {
    let mut new: Vec<Vec<Basic>> = Vec::new();
    for params in lists(g) {
        new.push(params.iter().map(|p| subs_map(p, rule)).collect::<PolyResult<_>>()?);
    }
    g_(&new[0], &new[1], &new[2], &new[3], &subs_map(&g_argument(g), rule)?)
}

fn replace_symbol(g: &Basic, old: &Basic, new: &Basic) -> R {
    subs_g(g, &[(old.clone(), new.clone())])
}

/// `e.as_coeff_mul(x)`: the factors free of `x` multiplied, and those with
/// `x` (`_coeff_mul`).
fn coeff_mul(e: &Basic, x: &Basic) -> PolyResult<(Basic, Vec<Basic>)> {
    if e.is_mul() {
        let mut free: Vec<Basic> = Vec::new();
        let mut dependent: Vec<Basic> = Vec::new();
        for f in e.args() {
            if has_x(f, x) {
                dependent.push(f.clone());
            } else {
                free.push(f.clone());
            }
        }
        return Ok((mul_n(&free)?, dependent));
    }
    if !has_x(e, x) {
        return Ok((e.clone(), Vec::new()));
    }
    Ok((one(), vec![e.clone()]))
}

/// `e.as_coeff_add(x)` (`_coeff_add`).
fn coeff_add(e: &Basic, x: &Basic) -> PolyResult<(Basic, Vec<Basic>)> {
    let mut free: Basic = zero();
    let mut dependent: Vec<Basic> = Vec::new();
    for term in add_args(e) {
        if has_x(&term, x) {
            dependent.push(term);
        } else {
            free = a2(&free, &term)?;
        }
    }
    Ok((free, dependent))
}

fn node_count(e: &Basic) -> usize {
    1 + e.args().iter().map(node_count).sum::<usize>()
}

/// Whether `res` has one of the nodes; for a `Piecewise` whether every
/// piece has one (`_has`).
fn has_where(res: &Basic, p: &dyn Fn(&Basic) -> bool) -> PolyResult<bool> {
    let res: Basic = piecewise_fold(res, Some(true))?;
    if matches!(res.head(), Head::Piecewise) {
        return Ok(res.args().iter().all(|pair| has_any(pair, p)));
    }
    Ok(has_any(&res, p))
}
fn is_hyper_or_g(n: &Basic) -> bool {
    matches!(n.head(), Head::Hyper | Head::MeijerG)
}

// ---------------------------------------------------------------------
// the table of representations

/// The computed terms of a table entry.
#[derive(Debug, Clone)]
pub enum Terms {
    /// `(factor, G)` pairs with the wilds of the formula.
    Fixed(Vec<(Basic, Basic)>),
    /// `log(t)**n*Heaviside(1 - t)`.
    Log1,
    /// `log(t)**n*Heaviside(t - 1)`.
    Log2,
    /// `log(t)**n`.
    Log3,
}

/// Whether a table entry applies to the bindings of its wilds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hint {
    Always,
    /// SymPy's `Not(IsNonPositiveInteger(a))`.
    NotNonpositiveInteger,
}

/// A formula of the table (`TableEntry`): `formula` (a pattern in the
/// dummy `z` with wilds) is `sum(factor * G)` over the terms when `cond`
/// holds and the hint accepts the bindings.
#[derive(Debug, Clone)]
pub struct TableEntry {
    pub formula: Basic,
    pub terms: Terms,
    pub cond: Basic,
    pub hint: Hint,
}

/// The wilds and the variable of the table.
pub struct TableSymbols {
    pub z: Basic,
    pub p: Basic,
    pub q: Basic,
    pub a: Basic,
    pub b: Basic,
    pub n: Basic,
}

pub type TableKey = Vec<String>;

fn is_function_node(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Function(_) | Head::Undefined(_) | Head::Hyper | Head::MeijerG | Head::Piecewise
    )
}

/// The sorted class names of the functions of `f` with `x` (`_mytype`).
fn mytype(f: &Basic, x: &Basic) -> TableKey {
    if free_of(f, x) {
        return Vec::new();
    }
    if is_function_node(f) {
        return vec![f.class_name().to_string()];
    }
    let mut v: Vec<String> = f.args().iter().flat_map(|a| mytype(a, x)).collect();
    v.sort();
    v
}

/// The variable and the wilds of the table, made once.
pub fn table_symbols() -> &'static TableSymbols {
    static T: OnceLock<TableSymbols> = OnceLock::new();
    T.get_or_init(|| {
        let z: Basic = Dummy::new("z", &[]).0;
        let wild = |name: &str| Wild::new(name, vec![z.clone()], vec![], &[]).0;
        let positive_integer: crate::core::symbol::WildProperty =
            Arc::new(|k: &Basic| k.as_rational().is_some_and(|q| q.is_int() && *q > Rat::from(Int::ZERO)));
        TableSymbols {
            p: wild("p"),
            q: wild("q"),
            a: wild("a"),
            b: wild("b"),
            n: Wild::new("n", vec![], vec![positive_integer], &[]).0,
            z,
        }
    })
}

/// The table of representations by the types of the functions of the
/// formulas (`lookup_table`).
pub fn lookup_table() -> PolyResult<&'static HashMap<TableKey, Vec<TableEntry>>> {
    static T: OnceLock<HashMap<TableKey, Vec<TableEntry>>> = OnceLock::new();
    if let Some(t) = T.get() {
        return Ok(t);
    }
    let table = create_lookup_table()?;
    Ok(T.get_or_init(|| table))
}

fn constant(c: &Basic) -> PolyResult<Vec<(Basic, Basic)>> {
    let z: &Basic = &table_symbols().z;
    Ok(vec![
        (c.clone(), g_(&[one()], &[], &[], &[zero()], z)?),
        (c.clone(), g_(&[], &[one()], &[zero()], &[], z)?),
    ])
}

/// The formulae of the function -> G-function table
/// (`_create_lookup_table`).
fn create_lookup_table() -> PolyResult<HashMap<TableKey, Vec<TableEntry>>> {
    let w: &TableSymbols = table_symbols();
    let (z, p, q, a, b, n) = (&w.z, &w.p, &w.q, &w.a, &w.b, &w.n);
    let t: Basic = m2(p, &p2(z, q)?)?;
    let mut table: HashMap<TableKey, Vec<TableEntry>> = HashMap::new();
    let mut push = |table: &mut HashMap<TableKey, Vec<TableEntry>>, entry: TableEntry| {
        let key: TableKey = mytype(&entry.formula, z);
        table.entry(key).or_default().push(entry);
    };
    table.insert(
        Vec::new(),
        vec![TableEntry {
            formula: a.clone(),
            terms: Terms::Fixed(constant(a)?),
            cond: true_(),
            hint: Hint::Always,
        }],
    );
    // add(formula, an, ap, bm, bq, arg, fac, cond, hint)
    let add = |table: &mut HashMap<TableKey, Vec<TableEntry>>,
               push: &mut dyn FnMut(&mut HashMap<TableKey, Vec<TableEntry>>, TableEntry),
               formula: Basic,
               lists: [Vec<Basic>; 4],
               arg: &Basic,
               fac: Basic,
               cond: Basic,
               hint: Hint|
     -> PolyResult<()> {
        let [an, ap, bm, bq] = lists;
        let g: Basic = g_(&an, &ap, &bm, &bq, arg)?;
        push(
            table,
            TableEntry {
                formula,
                terms: Terms::Fixed(vec![(fac, g)]),
                cond,
                hint,
            },
        );
        Ok(())
    };
    let gb = |x: &Basic| -> R { gt(x, &zero()) };

    // [P], Section 8.4.2
    let t_b: Basic = d2(&t, b)?;
    let gab: Basic = m2(&gamma(a)?, &p2(b, &s2(a, &one())?)?)?;
    let am1: Basic = s2(a, &one())?;
    add(
        &mut table,
        &mut push,
        m2(&heaviside(&s2(&t, b)?)?, &p2(&s2(&t, b)?, &am1)?)?,
        [vec![a.clone()], vec![], vec![], vec![zero()]],
        &t_b,
        gab.clone(),
        gb(b)?,
        Hint::Always,
    )?;
    add(
        &mut table,
        &mut push,
        m2(&heaviside(&s2(b, &t)?)?, &p2(&s2(b, &t)?, &am1)?)?,
        [vec![], vec![a.clone()], vec![zero()], vec![]],
        &t_b,
        gab.clone(),
        gb(b)?,
        Hint::Always,
    )?;
    let root: Basic = p2(&d2(b, p)?, &d2(&one(), q)?)?;
    add(
        &mut table,
        &mut push,
        m2(&heaviside(&s2(z, &root)?)?, &p2(&s2(&t, b)?, &am1)?)?,
        [vec![a.clone()], vec![], vec![], vec![zero()]],
        &t_b,
        gab.clone(),
        gb(b)?,
        Hint::Always,
    )?;
    add(
        &mut table,
        &mut push,
        m2(&heaviside(&s2(&root, z)?)?, &p2(&s2(b, &t)?, &am1)?)?,
        [vec![], vec![a.clone()], vec![zero()], vec![]],
        &t_b,
        gab,
        gb(b)?,
        Hint::Always,
    )?;
    add(
        &mut table,
        &mut push,
        p2(&a2(b, &t)?, &ng(a)?)?,
        [vec![s2(&one(), a)?], vec![], vec![zero()], vec![]],
        &t_b,
        d2(&p2(b, &ng(a)?)?, &gamma(a)?)?,
        true_(),
        Hint::NotNonpositiveInteger,
    )?;
    let one_a2: Basic = d2(&s2(&one(), a)?, &int(2))?;
    add(
        &mut table,
        &mut push,
        p2(&abs_(&s2(b, &t)?)?, &ng(a)?)?,
        [vec![s2(&one(), a)?], vec![one_a2.clone()], vec![zero()], vec![one_a2]],
        &t_b,
        m2(
            &m2(
                &m2(&int(2), &sin(&d2(&m2(&pi(), a)?, &int(2))?)?)?,
                &gamma(&s2(&one(), a)?)?,
            )?,
            &p2(&abs_(b)?, &ng(a)?)?,
        )?,
        lt(&re(a)?, &one())?,
        Hint::Always,
    )?;
    add(
        &mut table,
        &mut push,
        d2(&s2(&p2(&t, a)?, &p2(b, a)?)?, &s2(&t, b)?)?,
        [vec![zero(), a.clone()], vec![], vec![zero(), a.clone()], vec![]],
        &t_b,
        d2(&m2(&p2(b, &am1)?, &sin(&m2(a, &pi())?)?)?, &pi())?,
        true_(),
        Hint::Always,
    )?;

    // 12
    let a1 = |r: &Basic, sgn: i64, nu: &Basic| -> R {
        m2(
            &p2(&pi(), &rational(-1, 2))?,
            &p2(&d2(&m2(&int(-sgn), nu)?, &int(2))?, &s2(&one(), &m2(&int(2), r)?)?)?,
        )
    };
    for (r, sgn) in [(zero(), 1), (zero(), -1), (half(), 1), (half(), -1)] {
        let a_sq_t: Basic = a2(&p2(a, &int(2))?, &t)?;
        let formula: Basic = d2(&p2(&a2(&sqrt(&a_sq_t)?, &m2(&int(sgn), a)?)?, b)?, &p2(&a_sq_t, &r)?)?;
        let sb: Basic = m2(&int(sgn), b)?;
        add(
            &mut table,
            &mut push,
            formula,
            [
                vec![
                    d2(&a2(&one(), b)?, &int(2))?,
                    a2(&s2(&one(), &m2(&int(2), &r)?)?, &d2(b, &int(2))?)?,
                ],
                vec![],
                vec![d2(&s2(b, &sb)?, &int(2))?],
                vec![d2(&a2(b, &sb)?, &int(2))?],
            ],
            &d2(&t, &p2(a, &int(2))?)?,
            m2(&p2(a, &s2(b, &m2(&int(2), &r)?)?)?, &a1(&r, sgn, b)?)?,
            true_(),
            Hint::Always,
        )?;
    }

    // 13
    for (r, sgn) in [(zero(), 1), (zero(), -1), (half(), 1), (half(), -1)] {
        let pzq: Basic = m2(p, &p2(z, q)?)?;
        let formula: Basic = d2(
            &p2(
                &a2(
                    &sqrt(&a2(a, &pzq)?)?,
                    &m2(&m2(&int(sgn), &sqrt(p)?)?, &p2(z, &d2(q, &int(2))?)?)?,
                )?,
                b,
            )?,
            &p2(&a2(a, &pzq)?, &r)?,
        )?;
        let sb2: Basic = d2(&m2(&int(sgn), b)?, &int(2))?;
        add(
            &mut table,
            &mut push,
            formula,
            [
                vec![a2(&s2(&one(), &r)?, &sb2)?],
                vec![s2(&s2(&one(), &r)?, &sb2)?],
                vec![zero(), half()],
                vec![],
            ],
            &d2(&pzq, a)?,
            m2(&p2(a, &s2(&d2(b, &int(2))?, &r)?)?, &a1(&r, sgn, b)?)?,
            true_(),
            Hint::Always,
        )?;
    }

    // Section 8.4.3: SymPy's exp(polar_lift(-1)*t) with the factor
    // polar_lift(-1) moved into the argument of the G-function
    add(
        &mut table,
        &mut push,
        exp(&t)?,
        [vec![], vec![], vec![zero()], vec![]],
        &m2(&exp_polar(&m2(&ng(&imag())?, &pi())?)?, &t)?,
        one(),
        true_(),
        Hint::Always,
    )?;

    let t2_4: Basic = d2(&p2(&t, &int(2))?, &int(4))?;
    let pi32: Basic = p2(&pi(), &rational(3, 2))?;
    let sqrt_pi: Basic = sqrt(&pi())?;
    let simple: Vec<(Basic, [Vec<Basic>; 4], Basic, Basic)> = vec![
        // 8.4.4 (hyperbolic functions)
        (
            f1(Function::Sinh, &t)?,
            [vec![], vec![one()], vec![half()], vec![one(), zero()]],
            t2_4.clone(),
            pi32.clone(),
        ),
        (
            f1(Function::Cosh, &t)?,
            [vec![], vec![half()], vec![zero()], vec![half(), half()]],
            t2_4.clone(),
            pi32.clone(),
        ),
        // Section 8.4.5
        (
            sin(&t)?,
            [vec![], vec![], vec![half()], vec![zero()]],
            t2_4.clone(),
            sqrt_pi.clone(),
        ),
        (
            cos(&t)?,
            [vec![], vec![], vec![zero()], vec![half()]],
            t2_4.clone(),
            sqrt_pi.clone(),
        ),
    ];
    for (formula, l, arg, fac) in simple {
        add(&mut table, &mut push, formula, l, &arg, fac, true_(), Hint::Always)?;
    }
    // Section 8.4.6 (sinc): no head in the Rust core

    // Section 8.5.5
    let log_t: Basic = log(&t)?;
    let log_t_n: Basic = p2(&log_t, n)?;
    let addi = |table: &mut HashMap<TableKey, Vec<TableEntry>>,
                push: &mut dyn FnMut(&mut HashMap<TableKey, Vec<TableEntry>>, TableEntry),
                formula: Basic,
                terms: Terms| {
        push(
            table,
            TableEntry {
                formula,
                terms,
                cond: true_(),
                hint: Hint::Always,
            },
        );
    };
    addi(
        &mut table,
        &mut push,
        m2(&log_t_n, &heaviside(&s2(&one(), &t)?)?)?,
        Terms::Log1,
    );
    addi(
        &mut table,
        &mut push,
        m2(&log_t_n, &heaviside(&s2(&t, &one())?)?)?,
        Terms::Log2,
    );
    addi(&mut table, &mut push, log_t_n.clone(), Terms::Log3);
    let mut terms: Vec<(Basic, Basic)> = constant(&log(a)?)?;
    terms.push((one(), g_(&[one(), one()], &[], &[one()], &[zero()], &d2(&t, a)?)?));
    addi(&mut table, &mut push, log(&a2(&t, a)?)?, Terms::Fixed(terms));
    let mut terms: Vec<(Basic, Basic)> = constant(&log(&abs_(a)?)?)?;
    terms.push((
        pi(),
        g_(&[one(), one()], &[half()], &[one()], &[zero(), half()], &d2(&t, a)?)?,
    ));
    addi(&mut table, &mut push, log(&abs_(&s2(&t, a)?)?)?, Terms::Fixed(terms));

    // Section 8.4.11
    let mut terms: Vec<(Basic, Basic)> = constant(&m2(&ng(&imag())?, &pi())?)?;
    terms.push((
        int(-1),
        g_(&[], &[one()], &[zero(), zero()], &[], &m2(&t, &polar_lift(&int(-1))?)?)?,
    ));
    addi(&mut table, &mut push, f1(Function::Ei, &t)?, Terms::Fixed(terms));

    let mh: Basic = rational(-1, 2);
    let t2: Basic = p2(&t, &int(2))?;
    let pi2t4_16: Basic = d2(&m2(&p2(&pi(), &int(2))?, &p2(&t, &int(4))?)?, &int(16))?;
    let inv_sqrt_pi: Basic = d2(&one(), &sqrt_pi)?;
    let rows: Vec<(Basic, [Vec<Basic>; 4], Basic, Basic)> = vec![
        // Section 8.4.12
        (
            f1(Function::Si, &t)?,
            [vec![one()], vec![], vec![half()], vec![zero(), zero()]],
            t2_4.clone(),
            d2(&sqrt_pi, &int(2))?,
        ),
        (
            f1(Function::Ci, &t)?,
            [vec![], vec![one()], vec![zero(), zero()], vec![half()]],
            t2_4.clone(),
            d2(&ng(&sqrt_pi)?, &int(2))?,
        ),
        // Section 8.4.13
        (
            f1(Function::Shi, &t)?,
            [vec![half()], vec![], vec![zero()], vec![mh.clone(), mh.clone()]],
            d2(&m2(&polar_lift(&int(-1))?, &t2)?, &int(4))?,
            d2(&m2(&t, &sqrt_pi)?, &int(4))?,
        ),
        (
            f1(Function::Chi, &t)?,
            [vec![], vec![half(), one()], vec![zero(), zero()], vec![half(), half()]],
            t2_4.clone(),
            d2(&ng(&pi32)?, &int(2))?,
        ),
        // generalized exponential integral
        (
            apply(Function::Expint, &[a.clone(), t.clone()])?,
            [vec![], vec![a.clone()], vec![am1.clone(), zero()], vec![]],
            t.clone(),
            one(),
        ),
        // Section 8.4.14
        (
            f1(Function::Erf, &t)?,
            [vec![one()], vec![], vec![half()], vec![zero()]],
            t2.clone(),
            inv_sqrt_pi.clone(),
        ),
        (
            f1(Function::Erfc, &t)?,
            [vec![], vec![one()], vec![zero(), half()], vec![]],
            t2.clone(),
            inv_sqrt_pi.clone(),
        ),
        (
            f1(Function::Erfi, &t)?,
            [vec![half()], vec![], vec![zero()], vec![mh.clone()]],
            ng(&t2)?,
            d2(&t, &sqrt_pi)?,
        ),
        // Fresnel integrals
        (
            f1(Function::FresnelS, &t)?,
            [vec![one()], vec![], vec![rational(3, 4)], vec![zero(), rational(1, 4)]],
            pi2t4_16.clone(),
            half(),
        ),
        (
            f1(Function::FresnelC, &t)?,
            [vec![one()], vec![], vec![rational(1, 4)], vec![zero(), rational(3, 4)]],
            pi2t4_16,
            half(),
        ),
    ];
    for (formula, l, arg, fac) in rows {
        add(&mut table, &mut push, formula, l, &arg, fac, true_(), Hint::Always)?;
    }
    let a_2: Basic = d2(a, &int(2))?;
    let ma_2: Basic = d2(&ng(a)?, &int(2))?;
    let ma1_2: Basic = d2(&ng(&a2(a, &one())?)?, &int(2))?;
    let a1_2: Basic = d2(&a2(&one(), a)?, &int(2))?;
    let rows: Vec<(Basic, [Vec<Basic>; 4], Basic, Basic)> = vec![
        // Section 8.4.19
        (
            apply(Function::BesselJ, &[a.clone(), t.clone()])?,
            [vec![], vec![], vec![a_2.clone()], vec![ma_2.clone()]],
            t2_4.clone(),
            one(),
        ),
        // Section 8.4.20
        (
            apply(Function::BesselY, &[a.clone(), t.clone()])?,
            [
                vec![],
                vec![ma1_2.clone()],
                vec![a_2.clone(), ma_2.clone()],
                vec![ma1_2],
            ],
            t2_4.clone(),
            one(),
        ),
        // Section 8.4.22
        (
            apply(Function::BesselI, &[a.clone(), t.clone()])?,
            [vec![], vec![a1_2.clone()], vec![a_2.clone()], vec![ma_2.clone(), a1_2]],
            t2_4.clone(),
            pi(),
        ),
        // Section 8.4.23
        (
            apply(Function::BesselK, &[a.clone(), t.clone()])?,
            [vec![], vec![], vec![a_2, ma_2], vec![]],
            t2_4,
            half(),
        ),
        // complete elliptic integrals K(z) and E(z)
        (
            f1(Function::EllipticK, &t)?,
            [vec![half(), half()], vec![], vec![zero()], vec![zero()]],
            ng(&t)?,
            half(),
        ),
        (
            f1(Function::EllipticE, &t)?,
            [vec![half(), m2(&int(3), &half())?], vec![], vec![zero()], vec![zero()]],
            ng(&t)?,
            d2(&rational(-1, 2), &int(2))?,
        ),
    ];
    for (formula, l, arg, fac) in rows {
        add(&mut table, &mut push, formula, l, &arg, fac, true_(), Hint::Always)?;
    }
    Ok(table)
}

/// The terms of an entry for the bindings.
fn entry_terms(entry: &TableEntry, subs: &[(Basic, Basic)]) -> PolyResult<Vec<(Basic, Basic)>> {
    let w: &TableSymbols = table_symbols();
    let t: Basic = m2(&w.p, &p2(&w.z, &w.q)?)?;
    let order = || -> PolyResult<i64> {
        let found: Option<&Basic> = subs.iter().find(|(k, _)| *k == w.n).map(|(_, v)| v);
        match found.and_then(int_value) {
            Some(n) => Ok(n),
            None => Err(type_error("the order of the logarithm must be an integer")),
        }
    };
    let log1 = |n: i64| -> PolyResult<(Basic, Basic)> {
        let ones: Vec<Basic> = vec![one(); (n + 1) as usize];
        let zeros: Vec<Basic> = vec![zero(); (n + 1) as usize];
        Ok((
            m2(&p2(&int(-1), &int(n))?, &f1(Function::Factorial, &int(n))?)?,
            g_(&[], &ones, &zeros, &[], &t)?,
        ))
    };
    let log2 = |n: i64| -> PolyResult<(Basic, Basic)> {
        let ones: Vec<Basic> = vec![one(); (n + 1) as usize];
        let zeros: Vec<Basic> = vec![zero(); (n + 1) as usize];
        Ok((f1(Function::Factorial, &int(n))?, g_(&ones, &[], &[], &zeros, &t)?))
    };
    Ok(match &entry.terms {
        Terms::Fixed(v) => v.clone(),
        Terms::Log1 => vec![log1(order()?)?],
        Terms::Log2 => vec![log2(order()?)?],
        Terms::Log3 => {
            let n: i64 = order()?;
            vec![log1(n)?, log2(order()?)?]
        }
    })
}

fn entry_hint(entry: &TableEntry, subs: &[(Basic, Basic)]) -> PolyResult<bool> {
    match entry.hint {
        Hint::Always => Ok(true),
        Hint::NotNonpositiveInteger => {
            // False only for an integer a <= 0
            let value: Basic = unpolarify(&subs_map(&table_symbols().a, subs)?, false)?;
            Ok(!(is_integer_node(&value) && value.as_rational().is_some_and(|q| *q <= Rat::from(Int::ZERO))))
        }
    }
}

// ---------------------------------------------------------------------
// helpers

/// The message of the `ValueError` of [`get_coeff_exp`]
/// (`_CoeffExpValueError`).
const COEFF_EXP_ERROR: &str = "expr not of form a*x**b";

fn is_coeff_exp_error(e: &PolyError) -> bool {
    matches!(e, PolyError::Value(m) if m.starts_with(COEFF_EXP_ERROR))
}

/// `(c, b)` for `expr = c*x**b`; a `ValueError` when it is not of that
/// form (`_get_coeff_exp`).
pub fn get_coeff_exp(expr: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let (c, m) = coeff_mul(&expand_power_base(&powsimp(expr)?)?, x)?;
    if m.is_empty() {
        return Ok((c, zero()));
    }
    if m.len() > 1 {
        return Err(value_error(&format!("{}: {}", COEFF_EXP_ERROR, expr)));
    }
    let f: &Basic = &m[0];
    if f.is_pow() {
        if f.args()[0] != *x {
            return Err(value_error(COEFF_EXP_ERROR));
        }
        return Ok((c, f.args()[1].clone()));
    }
    if f == x {
        return Ok((c, one()));
    }
    Err(value_error(&format!("{}: {}", COEFF_EXP_ERROR, expr)))
}

/// The exponents of `x` in `expr` (`_exponents`).
fn exponents(expr: &Basic, x: &Basic) -> Vec<Basic> {
    fn walk(e: &Basic, x: &Basic, res: &mut Vec<Basic>) {
        if e == x {
            if !res.contains(&Basic::one()) {
                res.push(Basic::one());
            }
            return;
        }
        if e.is_pow() && e.args()[0] == *x {
            let ex: Basic = e.args()[1].clone();
            if !res.contains(&ex) {
                res.push(ex);
            }
            return;
        }
        for a in e.args() {
            walk(a, x, res);
        }
    }
    let mut res: Vec<Basic> = Vec::new();
    walk(expr, x, &mut res);
    res
}

/// The types of the functions of `expr` with `x` (`_functions`).
fn functions(expr: &Basic, x: &Basic) -> HashSet<String> {
    expr.preorder()
        .iter()
        .filter(|e| is_function_node(e) && !free_of(e, x))
        .map(|e| e.class_name().to_string())
        .collect()
}

/// The numbers `a` such that the substitution `x -> x + a` would
/// (hopefully) simplify `expr` (`_find_splitting_points`).
pub fn find_splitting_points(expr: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let p: Basic = Wild::new("p", vec![x.clone()], vec![], &[]).0;
    let q: Basic = Wild::new("q", vec![x.clone()], vec![], &[]).0;
    let pattern: Basic = a2(&m2(&p, x)?, &q)?;
    fn innermost(e: &Basic, pattern: &Basic, p: &Basic, q: &Basic, res: &mut Vec<Basic>) -> PolyResult<()> {
        if crate::core::logic::is_boolean(e) || is_tuple_like(e) {
            return Ok(());
        }
        let m = crate::core::matching::match_pattern(e, pattern, &[], false)?;
        if let Some(m) = m.filter(|m| !m.is_empty()) {
            let get = |w: &Basic| m.iter().find(|(k, _)| k == w).map(|(_, v)| v.clone());
            let pv: Basic = get(p).unwrap_or_else(Basic::zero);
            if pv != Basic::zero() {
                let qv: Basic = get(q).unwrap_or_else(Basic::zero);
                let point: Basic = d2(&ng(&qv)?, &pv)?;
                if !res.contains(&point) {
                    res.push(point);
                }
                return Ok(());
            }
        }
        if e.is_atom() {
            return Ok(());
        }
        for a in e.args() {
            innermost(a, pattern, p, q, res)?;
        }
        Ok(())
    }
    let mut res: Vec<Basic> = Vec::new();
    innermost(expr, &pattern, &p, &q, &mut res)?;
    Ok(res)
}

/// Whether a node is not an `Expr` of the Python core (tuples and the
/// pairs of a `Piecewise`).
fn is_tuple_like(e: &Basic) -> bool {
    matches!(e.head(), Head::Tuple | Head::ExprCondPair)
}

/// `f` as `fac * po * g` with `fac` constant, `po = x**s` and `g` the rest
/// (`_split_mul`).
pub fn split_mul(f: &Basic, x: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
    let mut fac: Basic = one();
    let mut po: Basic = one();
    let mut g: Basic = one();
    let f: Basic = expand_power_base(f)?;
    for a in mul_args(&f) {
        if a == *x {
            po = m2(&po, x)?;
        } else if free_of(&a, x) {
            fac = m2(&fac, &a)?;
        } else {
            if a.is_pow() && free_of(&a.args()[1], x) {
                let (base, ex) = (a.args()[0].clone(), a.args()[1].clone());
                let (mut c, mut t) = coeff_mul(&base, x)?;
                if t != [x.clone()] {
                    (c, t) = coeff_mul(&expand_mul(&base)?, x)?;
                }
                if t == [x.clone()] {
                    po = m2(&po, &p2(x, &ex)?)?;
                    fac = m2(&fac, &unpolarify(&polarify(&p2(&c, &ex)?, false)?, false)?)?;
                    continue;
                }
            }
            g = m2(&g, &a)?;
        }
    }
    Ok((fac, po, g))
}

/// The factors of `f`, an integer power repeated (`_mul_args`).
fn mul_args_repeated(f: &Basic) -> PolyResult<Vec<Basic>> {
    let mut gs: Vec<Basic> = Vec::new();
    for g in mul_args(f) {
        if g.is_pow() && is_integer_node(&g.args()[1]) {
            let Some(mut n) = int_value(&g.args()[1]) else {
                gs.push(g);
                continue;
            };
            let mut base: Basic = g.args()[0].clone();
            if n < 0 {
                n = -n;
                base = d2(&one(), &base)?;
            }
            for _ in 0..n {
                gs.push(base.clone());
            }
        } else {
            gs.push(g);
        }
    }
    Ok(gs)
}

/// All the ways to write `f` as a product of two factors; `None` when it
/// has fewer than two (`_mul_as_two_parts`).
fn mul_as_two_parts(f: &Basic) -> PolyResult<Option<Vec<(Basic, Basic)>>> {
    let gs: Vec<Basic> = mul_args_repeated(f)?;
    if gs.len() < 2 {
        return Ok(None);
    }
    if gs.len() == 2 {
        return Ok(Some(vec![(gs[0].clone(), gs[1].clone())]));
    }
    let mut found: Vec<(Basic, Basic)> = Vec::new();
    let rest: Vec<usize> = (1..gs.len()).collect();
    for size in 0..rest.len() {
        for chosen in combinations(&rest, size) {
            let mut first: Vec<Basic> = vec![gs[0].clone()];
            first.extend(chosen.iter().map(|&i| gs[i].clone()));
            let second: Vec<Basic> = rest
                .iter()
                .filter(|i| !chosen.contains(i))
                .map(|&i| gs[i].clone())
                .collect();
            let pair: (Basic, Basic) = (mul_n(&first)?, mul_n(&second)?);
            if !found.contains(&pair) {
                found.push(pair);
            }
        }
    }
    Ok(Some(found))
}

/// `itertools.combinations(items, k)`, in its order.
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let n: usize = items.len();
    if k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        let mut i: isize = k as isize - 1;
        while i >= 0 && idx[i as usize] == i as usize + n - k {
            i -= 1;
        }
        if i < 0 {
            return out;
        }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// `(C, h)` with `h` a G-function of argument `z**n` and `g = C*h`
/// ([L], page 150, equation (5)) (`_inflate_g`).
fn inflate_g(g: &Basic, n: i64) -> PolyResult<(Basic, Basic)> {
    let nn: Basic = int(n);
    let inflate = |params: &[Basic]| -> PolyResult<Vec<Basic>> {
        let mut out: Vec<Basic> = Vec::new();
        for a in params {
            for i in 0..n {
                out.push(d2(&a2(a, &int(i))?, &nn)?);
            }
        }
        Ok(out)
    };
    let v: Basic = int(g_ap(g).len() as i64 - g_bq(g).len() as i64);
    let mut c: Basic = p2(&nn, &a2(&a2(&one(), &meijerg_nu(g)?)?, &d2(&v, &int(2))?)?)?;
    c = d2(&c, &p2(&m2(&int(2), &pi())?, &m2(&int(n - 1), &meijerg_delta(g)?)?)?)?;
    let h: Basic = g_(
        &inflate(&g_an(g))?,
        &inflate(&g_aother(g))?,
        &inflate(&g_bm(g))?,
        &inflate(&g_bother(g))?,
        &m2(&p2(&g_argument(g), &nn)?, &p2(&nn, &m2(&nn, &v)?)?)?,
    )?;
    Ok((c, h))
}

/// The G-function of the inverse argument ([L], section 5.2)
/// (`_flip_g`).
fn flip_g(g: &Basic) -> R {
    let tr = |params: &[Basic]| -> PolyResult<Vec<Basic>> { params.iter().map(|a| s2(&one(), a)).collect() };
    g_(
        &tr(&g_bm(g))?,
        &tr(&g_bother(g))?,
        &tr(&g_an(g))?,
        &tr(&g_aother(g))?,
        &d2(&one(), &g_argument(g))?,
    )
}

fn dummies() -> &'static Mutex<HashMap<(String, String), Basic>> {
    static D: OnceLock<Mutex<HashMap<(String, String), Basic>>> = OnceLock::new();
    D.get_or_init(|| Mutex::new(HashMap::new()))
}

fn new_dummy(name: &str, positive: bool) -> Basic {
    if positive {
        Dummy::new(name, &[(Predicate::positive, true)]).0
    } else {
        Dummy::new(name, &[]).0
    }
}

/// The dummy of `name` and `token` (the same one each time), a new one
/// when it is in `expr` (`_dummy`).
fn dummy(name: &str, token: &str, expr: &Basic, positive: bool) -> Basic {
    let d: Basic = dummy_(name, token, positive);
    if expr.free_symbols().contains(&d) {
        return new_dummy(name, positive);
    }
    d
}

fn dummy_(name: &str, token: &str, positive: bool) -> Basic {
    let key: (String, String) = (
        name.to_string(),
        format!("{}{}", token, if positive { ":positive" } else { "" }),
    );
    let mut map = dummies().lock().unwrap_or_else(|e| e.into_inner());
    map.entry(key).or_insert_with(|| new_dummy(name, positive)).clone()
}

/// Whether `f`, written by G-functions on the positive reals, agrees with
/// them almost everywhere (`_is_analytic`).
fn is_analytic(f: &Basic, x: &Basic) -> bool {
    !atoms_where(f, &|n| is_fn(n, &[Function::Heaviside, Function::Abs]))
        .iter()
        .any(|e| !free_of(e, x))
}

// ---------------------------------------------------------------------
// the simplification of the conditions

fn is_relational(e: &Basic) -> bool {
    matches!(e.head(), Head::Relational(_))
}
fn rel_op(e: &Basic) -> RelOp {
    match e.head() {
        Head::Relational(op) => *op,
        _ => unreachable!("a relation"),
    }
}
/// `And`, `Or`, `Not`, ... (`BooleanFunction`).
fn is_boolean_function(e: &Basic) -> bool {
    crate::core::logic::is_boolean(e)
        && !is_relational(e)
        && !matches!(e.head(), Head::BooleanTrue | Head::BooleanFalse)
}
fn rebuild(e: &Basic, args: Vec<Basic>) -> R {
    crate::core::function::rebuild_evaluated(e, args)
}
fn replace_relations(e: &Basic, f: &dyn Fn(&Basic) -> R) -> R {
    crate::core::function::replace_where(e, &is_relational, f)
}

/// `-2*x + 1 < 0` as `x > 1/2` (`_canonical_coeff`).
fn canonical_coeff(node: &Basic) -> R {
    if !is_relational(node) {
        return Ok(node.clone());
    }
    let rel: Basic = crate::core::relational::canonical(node)?;
    let (lhs, rhs) = (&rel.args()[0], &rel.args()[1]);
    let (b, rest) = {
        let (c, rest) = crate::core::expr::as_coeff_add(lhs);
        if c.as_rational().is_some() {
            (c, rest)
        } else {
            (zero(), lhs.clone())
        }
    };
    let (m, lhs) = crate::functions::elementary::piecewise::as_coeff_mul_rational(&rest);
    if m == zero() || !is_rational(&m) {
        return Ok(rel);
    }
    let rhs: Basic = d2(&s2(rhs, &b)?, &m)?;
    let mut op: RelOp = rel_op(&rel);
    if yes(&m, Predicate::negative) {
        op = op.reversed();
    }
    crate::core::relational::evaluate_relation(op, &lhs, &rhs)
}

/// The rewriting rules of [`condsimp`], `(from, to)`.
fn condsimp_rules() -> PolyResult<&'static Vec<(Basic, Basic)>> {
    static RULES: OnceLock<Vec<(Basic, Basic)>> = OnceLock::new();
    if let Some(r) = RULES.get() {
        return Ok(r);
    }
    let rules = make_condsimp_rules()?;
    Ok(RULES.get_or_init(|| rules))
}

fn cond_wild_p() -> &'static Basic {
    static P: OnceLock<Basic> = OnceLock::new();
    P.get_or_init(|| Wild::new("p", vec![], vec![], &[]).0)
}
fn cond_wild_q() -> &'static Basic {
    static Q: OnceLock<Basic> = OnceLock::new();
    Q.get_or_init(|| Wild::new("q", vec![], vec![], &[]).0)
}

fn make_condsimp_rules() -> PolyResult<Vec<(Basic, Basic)>> {
    use crate::core::relational::rel;
    let p: &Basic = cond_wild_p();
    let q: &Basic = cond_wild_q();
    let and = |v: Vec<Basic>| Basic::new(Head::And, v);
    let or = |v: Vec<Basic>| Basic::new(Head::Or, v);
    let lt_ = |a: &Basic, b: &Basic| rel(RelOp::Lt, a, b);
    let le_ = |a: &Basic, b: &Basic| rel(RelOp::Le, a, b);
    let gt_ = |a: &Basic, b: &Basic| rel(RelOp::Gt, a, b);
    let eq_ = |a: &Basic, b: &Basic| rel(RelOp::Eq, a, b);
    let ne_ = |a: &Basic, b: &Basic| rel(RelOp::Ne, a, b);
    let pi: Basic = pi();
    let pi_2: Basic = d2(&pi, &int(2))?;
    let ap: Basic = arg_(p)?;
    let abs_ap: Basic = abs_(&ap)?;
    let two_ap: Basic = m2(&int(2), &ap)?;
    let p2_1: Basic = a2(&d2(&p2(p, &int(2))?, &int(2))?, &one())?;
    let ua = |e: &Basic| unbranched_argument(e);
    let i: Basic = imag();
    Ok(vec![
        // 0
        (or(vec![lt_(p, q), eq_(p, q)]), le(p, q)?),
        // 1
        (
            and(vec![le_(&abs_ap, &pi), le_(&abs_(&s2(&ap, &m2(&int(2), &pi)?)?)?, &pi)]),
            eq(&s2(&ap, &pi)?, &zero())?,
        ),
        // 2
        (
            and(vec![
                le_(&abs_(&a2(&two_ap, &pi)?)?, &pi),
                le_(&abs_(&s2(&two_ap, &pi)?)?, &pi),
            ]),
            eq(&ap, &zero())?,
        ),
        // 3
        (
            and(vec![
                lt_(&abs_(&a2(&two_ap, &pi)?)?, &pi),
                le_(&abs_(&s2(&two_ap, &pi)?)?, &pi),
            ]),
            false_(),
        ),
        // 4
        (
            and(vec![
                le_(&abs_(&s2(&ap, &pi_2)?)?, &pi_2),
                le_(&abs_(&a2(&ap, &pi_2)?)?, &pi_2),
            ]),
            eq(&ap, &zero())?,
        ),
        // 5
        (
            and(vec![
                le_(&abs_(&s2(&ap, &pi_2)?)?, &pi_2),
                lt_(&abs_(&a2(&ap, &pi_2)?)?, &pi_2),
            ]),
            false_(),
        ),
        // 6
        (
            and(vec![lt_(&abs_(&arg_(&p2_1)?)?, &pi), ne_(&abs_(&arg_(&p2_1)?)?, &pi)]),
            true_(),
        ),
        // 7
        (
            or(vec![lt_(&abs_(&arg_(&p2_1)?)?, &pi), ne_(&d2(&one(), &p2_1)?, &zero())]),
            true_(),
        ),
        // 8
        (
            and(vec![
                le_(&abs_(&ua(p)?)?, &pi),
                le_(&abs_(&ua(&m2(&exp_polar(&m2(&m2(&int(-2), &pi)?, &i)?)?, p)?)?)?, &pi),
            ]),
            eq(&ua(&m2(&exp_polar(&m2(&ng(&i)?, &pi)?)?, p)?)?, &zero())?,
        ),
        // 9
        (
            and(vec![
                le_(&abs_(&ua(p)?)?, &pi_2),
                le_(&abs_(&ua(&m2(&exp_polar(&m2(&ng(&pi)?, &i)?)?, p)?)?)?, &pi_2),
            ]),
            eq(&ua(&m2(&exp_polar(&d2(&m2(&ng(&i)?, &pi)?, &int(2))?)?, p)?)?, &zero())?,
        ),
        // 11
        (
            and(vec![ne_(&p2(p, &int(2))?, &one()), gt_(&p2(p, &int(2))?, &one())]),
            gt(&p2(p, &int(2))?, &one())?,
        ),
        // 12
        (
            and(vec![
                ne_(&d2(&one(), p)?, &one()),
                gt_(&m2(&cos(&abs_ap)?, &abs_(p)?)?, &one()),
            ]),
            gt(&abs_(p)?, &one())?,
        ),
        // 13
        (
            and(vec![ne_(p, &int(2)), gt_(&m2(&cos(&abs_ap)?, &abs_(p)?)?, &int(2))]),
            gt(&abs_(p)?, &int(2))?,
        ),
        // 14
        (
            and(vec![
                lt_(&abs_ap, &pi_2),
                gt_(&m2(&cos(&abs_ap)?, &sqrt(&abs_(&p2(p, &int(2))?)?)?)?, &one()),
            ]),
            gt(&p2(p, &int(2))?, &one())?,
        ),
    ])
    // rule 10 of SymPy, Or(p <= q, And(p < q, r)) -> p <= q, has a boolean
    // wild, which the matcher of the core does not take: left out
}

/// `Eq(arg(p)**q, 0)` (or of the unbranched argument) as `p > 0`
/// (`_rel_touchup`).
fn rel_touchup(node: &Basic) -> R {
    if !matches!(node.head(), Head::Relational(RelOp::Eq)) || node.args()[1] != zero() {
        return Ok(node.clone());
    }
    let lhs: &Basic = &node.args()[0];
    let p: &Basic = cond_wild_p();
    let q: &Basic = cond_wild_q();
    let mut m = crate::core::matching::match_pattern(lhs, &p2(&arg_(p)?, q)?, &[], false)?.filter(|m| !m.is_empty());
    if m.is_none() {
        m = crate::core::matching::match_pattern(lhs, &unbranched_argument(&p2(&polar_lift(p)?, q)?)?, &[], false)?
            .filter(|m| !m.is_empty());
    }
    let Some(m) = m else {
        if is_fn(lhs, &[Function::PeriodicArgument])
            && is(&lhs.args()[0], Predicate::polar) != Some(true)
            && lhs.args()[1] == oo()
        {
            return gt(&lhs.args()[0], &zero());
        }
        return Ok(node.clone());
    };
    let value: Basic = m
        .iter()
        .find(|(k, _)| k == p)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(zero);
    gt(&value, &zero())
}

/// Naive simplifications of `cond` (`_condsimp`).
pub fn condsimp(cond: &Basic, first: bool) -> R {
    let mut cond: Basic = cond.clone();
    if first {
        cond = replace_relations(&cond, &canonical_coeff)?;
    }
    if !is_boolean_function(&cond) {
        return Ok(cond);
    }
    let args: Vec<Basic> = cond
        .args()
        .iter()
        .map(|a| condsimp(a, false))
        .collect::<PolyResult<_>>()?;
    cond = rebuild(&cond, args)?;
    let rules = condsimp_rules()?;
    let mut change: bool = true;
    while change {
        change = false;
        for (fro, to) in rules.iter() {
            if !fro.head().same_head(cond.head()) {
                continue;
            }
            let cargs: Vec<Basic> = cond.args().to_vec();
            for (n, arg1) in cargs.iter().enumerate() {
                let Some(m) =
                    crate::core::matching::match_pattern(arg1, &fro.args()[0], &[], false)?.filter(|m| !m.is_empty())
                else {
                    continue;
                };
                let otherargs: Vec<Basic> = fro.args()[1..]
                    .iter()
                    .map(|x| subs_map(x, &m))
                    .collect::<PolyResult<_>>()?;
                let mut otherlist: Vec<usize> = vec![n];
                for arg2 in &otherargs {
                    for (k, arg3) in cargs.iter().enumerate() {
                        if otherlist.contains(&k) {
                            continue;
                        }
                        if arg2 == arg3 {
                            otherlist.push(k);
                            break;
                        }
                    }
                }
                if otherlist.len() != otherargs.len() + 1 {
                    continue;
                }
                let mut newargs: Vec<Basic> = cargs
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| !otherlist.contains(k))
                    .map(|(_, a)| a.clone())
                    .collect();
                newargs.push(subs_map(to, &m)?);
                cond = rebuild(&cond, newargs)?;
                change = true;
                break;
            }
            if !is_boolean_function(&cond) {
                break;
            }
        }
        if !is_boolean_function(&cond) {
            break;
        }
    }
    replace_relations(&cond, &rel_touchup)
}

/// The conditions evaluated again (`_eval_cond`).
fn eval_cond(cond: &Basic) -> R {
    condsimp(&crate::core::function::doit(cond)?, true)
}

/// `f` without polar numbers, the relations of its `Piecewise` in the
/// canonical coefficient form (`_my_unpolarify`).
fn my_unpolarify(f: &Basic) -> R {
    let e: Basic = crate::core::function::doit(&unpolarify(f, false)?)?;
    replace_relations(&e, &canonical_coeff)
}

fn my_unpolarify_cond(cond: &Basic) -> R {
    eval_cond(&unpolarify(cond, false)?)
}

// ---------------------------------------------------------------------
// the theorems

/// `expr` brought nearer to its principal branch by removing superfluous
/// factors (to the principal branch with `full_pb`)
/// (`_my_principal_branch`).
fn my_principal_branch(expr: &Basic, period: &Basic, full_pb: bool) -> R {
    let res: Basic = crate::functions::elementary::polar::principal_branch(expr, period)?;
    if full_pb {
        return Ok(res);
    }
    crate::core::function::replace_where(&res, &|n| is_fn(n, &[Function::PrincipalBranch]), &|n| {
        Ok(n.args()[0].clone())
    })
}

fn shifted(params: &[Basic], by: &dyn Fn(&Basic) -> R) -> PolyResult<Vec<Basic>> {
    params.iter().map(by).collect()
}

/// The integral of `fac*po*g` over `(0, oo)` as that of `C*G`, `G` with
/// argument `a*x` (`_rewrite_saxena_1`).
fn rewrite_saxena_1(fac: &Basic, po: &Basic, g: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let (_, s) = get_coeff_exp(po, x)?;
    let (a, b) = get_coeff_exp(&g_argument(g), x)?;
    let a: Basic = my_principal_branch(&a, &meijerg_get_period(g)?, false)?;
    let c: Basic = d2(
        fac,
        &m2(&abs_(&b)?, &p2(&a, &s2(&d2(&a2(&s, &one())?, &b)?, &one())?)?)?,
    )?;
    let shift: Basic = d2(&a2(&one(), &s)?, &b)?;
    let tr = |c: &Basic| -> R { s2(&a2(c, &shift)?, &one()) };
    Ok((
        c,
        g_(
            &shifted(&g_an(g), &tr)?,
            &shifted(&g_aother(g), &tr)?,
            &shifted(&g_bm(g), &tr)?,
            &shifted(&g_bother(g), &tr)?,
            &m2(&a, x)?,
        )?,
    ))
}

fn abs_ua(e: &Basic) -> R {
    abs_(&unbranched_argument(e)?)
}

/// The condition under which the integral of `g` over `(0, oo)` exists
/// ([L], section 5.6.1); with `helper` whether the integral over
/// `(1, oo)` exists (`_check_antecedents_1`).
fn check_antecedents_1(g: &Basic, x: &Basic, helper: bool) -> R {
    let delta: Basic = meijerg_delta(g)?;
    let (eta, _) = get_coeff_exp(&g_argument(g), x)?;
    let (m, n, p, q) = (g_bm(g).len(), g_an(g).len(), g_ap(g).len(), g_bq(g).len());
    if p > q {
        let tr = |c: &Basic| -> R { s2(&one(), c) };
        let flipped: Basic = g_(
            &shifted(&g_bm(g), &tr)?,
            &shifted(&g_bother(g), &tr)?,
            &shifted(&g_an(g), &tr)?,
            &shifted(&g_aother(g), &tr)?,
            &d2(x, &eta)?,
        )?;
        return check_antecedents_1(&flipped, x, false);
    }
    let mut tmp: Vec<Basic> = Vec::new();
    for b in g_bm(g) {
        tmp.push(lt(&ng(&re(&b)?)?, &one())?);
    }
    for a in g_an(g) {
        tmp.push(lt(&one(), &s2(&one(), &re(&a)?)?)?);
    }
    let cond_3: Basic = and_(&tmp);
    let (pi_, qi) = (p as i64, q as i64);
    let cond_4: Basic = gt(
        &a2(&ng(&re(&meijerg_nu(g)?)?)?, &rational(qi + 1 - pi_, 2))?,
        &int(qi - pi_),
    )?;
    let mut conds: Vec<Basic> = Vec::new();
    // case 1
    let tmp1: Vec<Basic> = vec![boolean(1 <= n), boolean(p < q), boolean(1 <= m)];
    let tmp2: Vec<Basic> = vec![
        boolean(1 <= p),
        boolean(1 <= m),
        boolean(q == p + 1),
        boolean(!(n == 0 && m == p + 1)),
    ];
    let mut tmp3: Vec<Basic> = vec![boolean(1 <= p), boolean(q == p)];
    let dq: Rat = rat_of(&delta).ok_or_else(|| type_error("delta is rational"))?;
    // ceiling(delta/2)
    let ceil: i64 = {
        let to = |i: Int| i64::try_from(i).map_err(|_| type_error("delta too large"));
        let num: i64 = to(crate::numbers::numer(&dq))?;
        let den: i64 = 2 * to(crate::numbers::denom(&dq))?;
        -((-num).div_euclid(den))
    };
    let ua_eta: Basic = abs_ua(&eta)?;
    for k in 0..(ceil + 1).max(0) {
        tmp3.push(ne(&ua_eta, &m2(&s2(&delta, &int(2 * k))?, &pi())?)?);
    }
    let delta_pi: Basic = m2(&delta, &pi())?;
    let common: Vec<Basic> = vec![gt(&delta, &zero())?, lt(&ua_eta, &delta_pi)?];
    let extra: Vec<Basic> = if helper {
        Vec::new()
    } else {
        vec![ne(&eta, &zero())?, cond_3.clone()]
    };
    for t in [tmp1, tmp2, tmp3] {
        let mut all: Vec<Basic> = t;
        all.extend(common.iter().cloned());
        all.extend(extra.iter().cloned());
        conds.push(and_(&all));
    }
    // case 2
    let extra: Vec<Basic> = if helper { Vec::new() } else { vec![cond_3.clone()] };
    let mut c: Vec<Basic> = vec![
        boolean(n == 0),
        boolean(p < m),
        boolean(m <= q),
        lt(&ua_eta, &delta_pi)?,
    ];
    c.extend(extra);
    conds.push(and_(&c));
    // case 3
    let extra: Vec<Basic> = if helper {
        Vec::new()
    } else {
        vec![cond_3.clone(), cond_4]
    };
    let mut c: Vec<Basic> = vec![
        boolean(p < q),
        boolean(1 <= m),
        gt(&delta, &zero())?,
        eq(&ua_eta, &delta_pi)?,
    ];
    c.extend(extra.iter().cloned());
    conds.push(and_(&c));
    let mut c: Vec<Basic> = vec![boolean(pi_ <= qi - 2), eq(&delta, &zero())?, eq(&ua_eta, &zero())?];
    c.extend(extra);
    conds.push(and_(&c));
    // the extra case from the Wolfram functions site
    let mut case_extra: Vec<Basic> = vec![
        boolean(p == q),
        eq(&delta, &zero())?,
        eq(&unbranched_argument(&eta)?, &zero())?,
        ne(&eta, &zero())?,
    ];
    if !helper {
        case_extra.push(cond_3.clone());
    }
    let mut total: Basic = zero();
    for (a, b) in g_ap(g).iter().zip(g_bq(g).iter()) {
        total = a2(&total, &s2(b, a)?)?;
    }
    case_extra.push(lt(&re(&total)?, &zero())?);
    conds.push(and_(&case_extra));
    let mut case_extra_2: Vec<Basic> = vec![and_(&[gt(&delta, &zero())?, lt(&ua_eta, &delta_pi)?])];
    if !helper {
        case_extra_2.push(cond_3);
    }
    conds.push(and_(&case_extra_2));
    Ok(or_(&conds))
}

/// The integral of `g` over `(0, oo)`, the conditions assumed
/// (`_int0oo_1`).
fn int0oo_1(g: &Basic, x: &Basic) -> R {
    let (eta, _) = get_coeff_exp(&g_argument(g), x)?;
    let mut res: Basic = d2(&one(), &eta)?;
    for b in g_bm(g) {
        res = m2(&res, &gamma(&a2(&b, &one())?)?)?;
    }
    for a in g_an(g) {
        res = m2(&res, &gamma(&s2(&s2(&one(), &a)?, &one())?)?)?;
    }
    for b in g_bother(g) {
        res = d2(&res, &gamma(&s2(&s2(&one(), &b)?, &one())?)?)?;
    }
    for a in g_aother(g) {
        res = d2(&res, &gamma(&a2(&a, &one())?)?)?;
    }
    gammasimp(&unpolarify(&res, false)?)
}

fn gcd_i128(a: i128, b: i128) -> i128 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i128(b, a % b)
    }
}

/// The integral of `fac*po*g1*g2` over `(0, oo)` as that of `C*f1*f2`
/// with G-functions of arguments `c*x`; `None` when the exponents are not
/// rational (`_rewrite_saxena`).
fn rewrite_saxena(
    fac: &Basic,
    po: &Basic,
    g1: &Basic,
    g2: &Basic,
    x: &Basic,
    full_pb: bool,
) -> PolyResult<Option<(Basic, Basic, Basic)>> {
    let pb = |g: &Basic| -> R {
        let (a, b) = get_coeff_exp(&g_argument(g), x)?;
        let [an, aother, bm, bother] = lists(g);
        g_(
            &an,
            &aother,
            &bm,
            &bother,
            &m2(&my_principal_branch(&a, &meijerg_get_period(g)?, full_pb)?, &p2(x, &b)?)?,
        )
    };
    let (_, s) = get_coeff_exp(po, x)?;
    let (_, mut b1) = get_coeff_exp(&g_argument(g1), x)?;
    let (_, mut b2) = get_coeff_exp(&g_argument(g2), x)?;
    let mut g1: Basic = g1.clone();
    let mut g2: Basic = g2.clone();
    if is_true(&lt(&b1, &zero())?) {
        b1 = ng(&b1)?;
        g1 = flip_g(&g1)?;
    }
    if is_true(&lt(&b2, &zero())?) {
        b2 = ng(&b2)?;
        g2 = flip_g(&g2)?;
    }
    let (Some(r1q), Some(r2q)) = (rat_of(&b1), rat_of(&b2)) else {
        return Ok(None);
    };
    if b1 == zero() || b2 == zero() {
        return Ok(None);
    }
    let to = |i: Int| -> PolyResult<i128> { i128::try_from(i).map_err(|_| type_error("exponent too large")) };
    let (m1, n1) = (to(crate::numbers::numer(&r1q))?, to(crate::numbers::denom(&r1q))?);
    let (m2_, n2) = (to(crate::numbers::numer(&r2q))?, to(crate::numbers::denom(&r2q))?);
    let (u, v) = (m1 * n2, m2_ * n1);
    let tau: i128 = u / gcd_i128(u, v) * v;
    let r1: i64 = (tau / u) as i64;
    let r2: i64 = (tau / v) as i64;
    let (c1, g1) = inflate_g(&g1, r1)?;
    let (c2, g2) = inflate_g(&g2, r2)?;
    let g1: Basic = pb(&g1)?;
    let g2: Basic = pb(&g2)?;
    let mut fac: Basic = m2(&m2(fac, &c1)?, &c2)?;
    let (a1, b) = get_coeff_exp(&g_argument(&g1), x)?;
    let (a2_, _) = get_coeff_exp(&g_argument(&g2), x)?;
    // arbitrarily the x**s part goes to g1
    let exponent: Basic = s2(&d2(&a2(&s, &one())?, &b)?, &one())?;
    fac = d2(&fac, &m2(&abs_(&b)?, &p2(&a1, &exponent)?)?)?;
    let tr = |c: &Basic| -> R { a2(c, &exponent) };
    let f1: Basic = g_(
        &shifted(&g_an(&g1), &tr)?,
        &shifted(&g_aother(&g1), &tr)?,
        &shifted(&g_bm(&g1), &tr)?,
        &shifted(&g_bother(&g1), &tr)?,
        &m2(&a1, x)?,
    )?;
    let [an, aother, bm, bother] = lists(&g2);
    let f2: Basic = g_(&an, &aother, &bm, &bother, &m2(&a2_, x)?)?;
    Ok(Some((powdenest_polar(&fac)?, f1, f2)))
}

fn positive(e: &Basic) -> Basic {
    boolean(yes(e, Predicate::positive))
}
fn negative(e: &Basic) -> Basic {
    boolean(yes(e, Predicate::negative))
}

/// The condition under which the theorem for the integral of the product
/// of two G-functions applies: the cases of [P], section 2.24.1, and four
/// of [L] (`_check_antecedents`).
fn check_antecedents(g1: &Basic, g2: &Basic, x: &Basic) -> R {
    let (sigma, _) = get_coeff_exp(&g_argument(g1), x)?;
    let (omega, _) = get_coeff_exp(&g_argument(g2), x)?;
    let len = |v: Vec<Basic>| int(v.len() as i64);
    let (s, t, u, v) = (len(g_bm(g1)), len(g_an(g1)), len(g_ap(g1)), len(g_bq(g1)));
    let (m, n, p, q) = (len(g_bm(g2)), len(g_an(g2)), len(g_ap(g2)), len(g_bq(g2)));
    let bstar: Basic = s2(&a2(&s, &t)?, &d2(&a2(&u, &v)?, &int(2))?)?;
    let cstar: Basic = s2(&a2(&m, &n)?, &d2(&a2(&p, &q)?, &int(2))?)?;
    let rho: Basic = a2(&a2(&meijerg_nu(g1)?, &d2(&s2(&u, &v)?, &int(2))?)?, &one())?;
    let mu: Basic = a2(&a2(&meijerg_nu(g2)?, &d2(&s2(&p, &q)?, &int(2))?)?, &one())?;
    let q_p: Basic = s2(&q, &p)?;
    let v_u: Basic = s2(&v, &u)?;
    let phi: Basic = s2(&q_p, &v_u)?;
    let eta: Basic = s2(&s2(&s2(&one(), &v_u)?, &mu)?, &rho)?;
    let ua_omega: Basic = abs_ua(&omega)?;
    let ua_sigma: Basic = abs_ua(&sigma)?;
    let psi: Basic = d2(&a2(&m2(&pi(), &s2(&s2(&q, &m)?, &n)?)?, &ua_omega)?, &q_p)?;
    let theta: Basic = d2(&a2(&m2(&pi(), &s2(&s2(&v, &s)?, &t)?)?, &ua_sigma)?, &v_u)?;

    let c1: bool = {
        let mut ok: bool = true;
        'outer: for g in [g1, g2] {
            for i in g_an(g) {
                for j in g_bm(g) {
                    let diff: Basic = s2(&i, &j)?;
                    if yes(&diff, Predicate::integer) && yes(&diff, Predicate::positive) {
                        ok = false;
                        break 'outer;
                    }
                }
            }
        }
        ok
    };
    let c1: Basic = boolean(c1);
    let mut c2v: Vec<Basic> = Vec::new();
    for i in g_bm(g1) {
        for j in g_bm(g2) {
            c2v.push(gt(&re(&a2(&a2(&one(), &i)?, &j)?)?, &zero())?);
        }
    }
    let c2: Basic = and_(&c2v);
    let mut c3v: Vec<Basic> = Vec::new();
    for i in g_an(g1) {
        for j in g_an(g2) {
            c3v.push(lt(&re(&a2(&a2(&one(), &i)?, &j)?)?, &int(2))?);
        }
    }
    let c3: Basic = and_(&c3v);
    let m32: Basic = rational(-3, 2);
    let p_q: Basic = s2(&p, &q)?;
    let u_v: Basic = s2(&u, &v)?;
    let cn = |params: Vec<Basic>, k: &Basic, w: &Basic, minus_one: bool| -> R {
        let mut out: Vec<Basic> = Vec::new();
        for i in params {
            let mut inner: Basic = a2(&one(), &i)?;
            if minus_one {
                inner = s2(&inner, &one())?;
            }
            out.push(gt(&s2(&m2(k, &re(&inner)?)?, &re(w)?)?, &m32)?);
        }
        Ok(and_(&out))
    };
    let c4: Basic = cn(g_an(g1), &p_q, &mu, true)?;
    let c5: Basic = cn(g_bm(g1), &p_q, &mu, false)?;
    let c6: Basic = cn(g_an(g2), &u_v, &rho, true)?;
    let c7: Basic = cn(g_bm(g2), &u_v, &rho, false)?;
    let inner: Basic = a2(
        &a2(&m2(&s2(&rho, &one())?, &q_p)?, &m2(&v_u, &q_p)?)?,
        &m2(&s2(&mu, &one())?, &v_u)?,
    )?;
    let two_re: Basic = m2(&int(2), &re(&inner)?)?;
    let c8: Basic = gt(&a2(&abs_(&phi)?, &two_re)?, &zero())?;
    let c9: Basic = gt(&s2(&abs_(&phi)?, &two_re)?, &zero())?;
    let bstar_pi: Basic = m2(&bstar, &pi())?;
    let cstar_pi: Basic = m2(&cstar, &pi())?;
    let c10: Basic = lt(&ua_sigma, &bstar_pi)?;
    let c11: Basic = eq(&ua_sigma, &bstar_pi)?;
    let c12: Basic = lt(&ua_omega, &cstar_pi)?;
    let c13: Basic = eq(&ua_omega, &cstar_pi)?;

    // the condition c14 as other people have read [P]
    let z0: Basic = exp(&m2(&m2(&ng(&a2(&bstar, &cstar)?)?, &pi())?, &imag())?)?;
    let zos: Basic = unpolarify(&d2(&m2(&z0, &omega)?, &sigma)?, false)?;
    let zso: Basic = unpolarify(&d2(&m2(&z0, &sigma)?, &omega)?, false)?;
    let mrvu: Basic = s2(&a2(&a2(&mu, &rho)?, &v)?, &u)?;
    let mrqp: Basic = s2(&a2(&a2(&mu, &rho)?, &q)?, &p)?;
    let c14: Basic = if zos == d2(&one(), &zso)? {
        and_(&[
            eq(&phi, &zero())?,
            le(&a2(&bstar, &cstar)?, &one())?,
            or_(&[ne(&zos, &one())?, lt(&re(&mrvu)?, &one())?, lt(&re(&mrqp)?, &one())?]),
        ])
    } else {
        let cond = |w: &Basic| -> R {
            // abs(arg(1 - w)) < pi, avoiding arg(0)
            if *w == one() {
                return Ok(false_());
            }
            lt(&abs_(&arg_(&s2(&one(), w)?)?)?, &pi())
        };
        let c14: Basic = and_(&[
            eq(&phi, &zero())?,
            le(&a2(&s2(&bstar, &one())?, &cstar)?, &zero())?,
            or_(&[
                and_(&[ne(&zos, &one())?, cond(&zos)?]),
                and_(&[lt(&re(&mrvu)?, &one())?, eq(&zos, &one())?]),
            ]),
        ]);
        let c14_alt: Basic = and_(&[
            eq(&phi, &zero())?,
            le(&a2(&s2(&cstar, &one())?, &bstar)?, &zero())?,
            or_(&[
                and_(&[ne(&zso, &one())?, cond(&zso)?]),
                and_(&[lt(&re(&mrqp)?, &one())?, eq(&zso, &one())?]),
            ]),
        ]);
        or_(&[c14, c14_alt])
    };

    let c15_attempt = || -> R {
        let aw: Basic = p2(&abs_(&omega)?, &d2(&one(), &q_p)?)?;
        let asg: Basic = p2(&abs_(&sigma)?, &d2(&one(), &v_u)?)?;
        let lambda_c: Basic = a2(
            &m2(&m2(&q_p, &aw)?, &cos(&psi)?)?,
            &m2(&m2(&v_u, &asg)?, &cos(&theta)?)?,
        )?;
        if !is_false(&eval_cond(&gt(&lambda_c, &zero())?)?) {
            return gt(&lambda_c, &zero());
        }
        let lambda_s0 = |k1: &Basic, k2: &Basic| -> R {
            a2(
                &m2(&m2(&m2(k1, &q_p)?, &aw)?, &sin(&psi)?)?,
                &m2(&m2(&m2(k2, &v_u)?, &asg)?, &sin(&theta)?)?,
            )
        };
        let us: Basic = unbranched_argument(&sigma)?;
        let uo: Basic = unbranched_argument(&omega)?;
        let (o, mo) = (one(), int(-1));
        let lambda_s: Basic = piecewise(&[
            (
                m2(&lambda_s0(&o, &o)?, &lambda_s0(&mo, &mo)?)?,
                and_(&[eq(&us, &zero())?, eq(&uo, &zero())?]),
            ),
            (
                m2(&lambda_s0(&sign(&uo)?, &o)?, &lambda_s0(&sign(&uo)?, &mo)?)?,
                and_(&[eq(&us, &zero())?, ne(&uo, &zero())?]),
            ),
            (
                m2(&lambda_s0(&o, &sign(&us)?)?, &lambda_s0(&mo, &sign(&us)?)?)?,
                and_(&[ne(&us, &zero())?, eq(&uo, &zero())?]),
            ),
            (lambda_s0(&sign(&uo)?, &sign(&us)?)?, true_()),
        ])?;
        Ok(or_(&[
            gt(&lambda_c, &zero())?,
            and_(&[
                eq(&lambda_c, &zero())?,
                ne(&lambda_s, &zero())?,
                gt(&re(&eta)?, &int(-1))?,
            ]),
            and_(&[
                eq(&lambda_c, &zero())?,
                eq(&lambda_s, &zero())?,
                gt(&re(&eta)?, &zero())?,
            ]),
        ]))
    };
    let c15: Basic = match c15_attempt() {
        Ok(c) => c,
        Err(PolyError::Type(_)) => false_(),
        Err(e) => return Err(e),
    };

    let b = |v: bool| boolean(v);
    let s_pos = positive(&s);
    let t_pos = positive(&t);
    let m_pos = positive(&m);
    let n_pos = positive(&n);
    let bs_pos = positive(&bstar);
    let cs_pos = positive(&cstar);
    let sig_pos = positive(&sigma);
    let om_pos = positive(&omega);
    let mut conds: Vec<Basic> = Vec::new();
    let mn_st: bool = m2(&m2(&m2(&m, &n)?, &s)?, &t)? != zero();
    let (e0, z_) = (|a: &Basic, b: &Basic| eq(a, b), zero());
    let re_rho_1: Basic = lt(&re(&rho)?, &one())?;
    let re_mu_1: Basic = lt(&re(&mu)?, &one())?;
    // 1
    conds.push(and_(&[
        b(mn_st),
        bs_pos.clone(),
        cs_pos.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c10.clone(),
        c12.clone(),
    ]));
    // 2
    conds.push(and_(&[
        e0(&u, &v)?,
        e0(&bstar, &z_)?,
        cs_pos.clone(),
        sig_pos.clone(),
        re_rho_1.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c12.clone(),
    ]));
    // 3
    conds.push(and_(&[
        e0(&p, &q)?,
        e0(&cstar, &z_)?,
        bs_pos.clone(),
        om_pos.clone(),
        re_mu_1.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c10.clone(),
    ]));
    // 4
    conds.push(and_(&[
        e0(&p, &q)?,
        e0(&u, &v)?,
        e0(&bstar, &z_)?,
        e0(&cstar, &z_)?,
        sig_pos.clone(),
        om_pos.clone(),
        re_mu_1.clone(),
        re_rho_1.clone(),
        ne(&sigma, &omega)?,
        c1.clone(),
        c2.clone(),
        c3.clone(),
    ]));
    // 5
    conds.push(and_(&[
        e0(&p, &q)?,
        e0(&u, &v)?,
        e0(&bstar, &z_)?,
        e0(&cstar, &z_)?,
        sig_pos.clone(),
        om_pos.clone(),
        lt(&re(&a2(&mu, &rho)?)?, &one())?,
        ne(&omega, &sigma)?,
        c1.clone(),
        c2.clone(),
        c3.clone(),
    ]));
    let cs_ge0: Basic = ge(&cstar, &z_)?;
    let bs_ge0: Basic = ge(&bstar, &z_)?;
    // 6
    conds.push(and_(&[
        gt(&p, &q)?,
        s_pos.clone(),
        bs_pos.clone(),
        cs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c5.clone(),
        c10.clone(),
        c13.clone(),
    ]));
    // 7
    conds.push(and_(&[
        lt(&p, &q)?,
        t_pos.clone(),
        bs_pos.clone(),
        cs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c4.clone(),
        c10.clone(),
        c13.clone(),
    ]));
    // 8
    conds.push(and_(&[
        gt(&u, &v)?,
        m_pos.clone(),
        cs_pos.clone(),
        bs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c7.clone(),
        c11.clone(),
        c12.clone(),
    ]));
    // 9
    conds.push(and_(&[
        lt(&u, &v)?,
        n_pos.clone(),
        cs_pos.clone(),
        bs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c6.clone(),
        c11.clone(),
        c12.clone(),
    ]));
    // 10
    conds.push(and_(&[
        gt(&p, &q)?,
        e0(&u, &v)?,
        e0(&bstar, &z_)?,
        cs_ge0.clone(),
        sig_pos.clone(),
        re_rho_1.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c5.clone(),
        c13.clone(),
    ]));
    // 11
    conds.push(and_(&[
        lt(&p, &q)?,
        e0(&u, &v)?,
        e0(&bstar, &z_)?,
        cs_ge0.clone(),
        sig_pos.clone(),
        re_rho_1.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c4.clone(),
        c13.clone(),
    ]));
    // 12
    conds.push(and_(&[
        e0(&p, &q)?,
        gt(&u, &v)?,
        bs_ge0.clone(),
        e0(&cstar, &z_)?,
        om_pos.clone(),
        re_mu_1.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c7.clone(),
        c11.clone(),
    ]));
    // 13
    conds.push(and_(&[
        e0(&p, &q)?,
        lt(&u, &v)?,
        bs_ge0.clone(),
        e0(&cstar, &z_)?,
        om_pos.clone(),
        re_mu_1.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c6.clone(),
        c11.clone(),
    ]));
    // 14
    conds.push(and_(&[
        lt(&p, &q)?,
        gt(&u, &v)?,
        bs_ge0.clone(),
        cs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c4.clone(),
        c7.clone(),
        c11.clone(),
        c13.clone(),
    ]));
    // 15
    conds.push(and_(&[
        gt(&p, &q)?,
        lt(&u, &v)?,
        bs_ge0.clone(),
        cs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c5.clone(),
        c6.clone(),
        c11.clone(),
        c13.clone(),
    ]));
    // 16
    conds.push(and_(&[
        gt(&p, &q)?,
        gt(&u, &v)?,
        bs_ge0.clone(),
        cs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c5.clone(),
        c7.clone(),
        c8,
        c11.clone(),
        c13.clone(),
        c14.clone(),
    ]));
    // 17
    conds.push(and_(&[
        lt(&p, &q)?,
        lt(&u, &v)?,
        bs_ge0.clone(),
        cs_ge0.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c4,
        c6,
        c9,
        c11.clone(),
        c13.clone(),
        c14.clone(),
    ]));
    let phi_pos = positive(&phi);
    let phi_neg = negative(&phi);
    let bs_neg = negative(&bstar);
    let cs_neg = negative(&cstar);
    // 18
    conds.push(and_(&[
        e0(&t, &z_)?,
        s_pos.clone(),
        bs_pos.clone(),
        phi_pos.clone(),
        c1.clone(),
        c2.clone(),
        c10.clone(),
    ]));
    // 19
    conds.push(and_(&[
        e0(&s, &z_)?,
        t_pos.clone(),
        bs_pos.clone(),
        phi_neg.clone(),
        c1.clone(),
        c3.clone(),
        c10.clone(),
    ]));
    // 20
    conds.push(and_(&[
        e0(&n, &z_)?,
        m_pos.clone(),
        cs_pos.clone(),
        phi_neg,
        c1.clone(),
        c2.clone(),
        c12.clone(),
    ]));
    // 21
    conds.push(and_(&[
        e0(&m, &z_)?,
        n_pos.clone(),
        cs_pos.clone(),
        phi_pos,
        c1.clone(),
        c3.clone(),
        c12.clone(),
    ]));
    // 22
    conds.push(and_(&[
        e0(&m2(&s, &t)?, &z_)?,
        bs_pos.clone(),
        cs_pos.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c10.clone(),
        c12.clone(),
    ]));
    // 23
    conds.push(and_(&[
        e0(&m2(&m, &n)?, &z_)?,
        bs_pos.clone(),
        cs_pos.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
        c10.clone(),
        c12.clone(),
    ]));

    // the cases of [Luke1969] not covered by Prudnikov's
    let mt1_exists: Basic = check_antecedents_1(g1, x, true)?;
    let mt2_exists: Basic = check_antecedents_1(g2, x, true)?;
    conds.push(and_(&[
        mt2_exists.clone(),
        e0(&t, &z_)?,
        lt(&u, &s)?,
        bs_pos.clone(),
        c10.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
    ]));
    conds.push(and_(&[
        mt2_exists,
        e0(&s, &z_)?,
        lt(&v, &t)?,
        bs_pos.clone(),
        c10.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
    ]));
    conds.push(and_(&[
        mt1_exists.clone(),
        e0(&n, &z_)?,
        lt(&p, &m)?,
        cs_pos.clone(),
        c12.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
    ]));
    conds.push(and_(&[
        mt1_exists,
        e0(&m, &z_)?,
        lt(&q, &n)?,
        cs_pos.clone(),
        c12.clone(),
        c1.clone(),
        c2.clone(),
        c3.clone(),
    ]));

    // a short cut when one of these holds: the rest are corner cases
    let r: Basic = or_(&conds);
    if !is_false(&eval_cond(&r)?) {
        return Ok(r);
    }

    let phi0: Basic = e0(&phi, &z_)?;
    let lt_ua = |a: &Basic, k: &Basic| -> R { lt(a, &m2(k, &pi())?) };
    let mnp1: Basic = a2(&s2(&a2(&m, &n)?, &p)?, &one())?;
    let mnq1: Basic = a2(&s2(&a2(&m, &n)?, &q)?, &one())?;
    let stu1: Basic = a2(&s2(&a2(&s, &t)?, &u)?, &one())?;
    let stv1: Basic = a2(&s2(&a2(&s, &t)?, &v)?, &one())?;
    // 24
    conds.push(and_(&[
        gt(&a2(&m, &n)?, &p)?,
        e0(&t, &z_)?,
        phi0.clone(),
        s_pos.clone(),
        bs_pos.clone(),
        cs_neg.clone(),
        lt_ua(&ua_omega, &mnp1)?,
        c1.clone(),
        c2.clone(),
        c10.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 25
    conds.push(and_(&[
        gt(&a2(&m, &n)?, &q)?,
        e0(&s, &z_)?,
        phi0.clone(),
        t_pos.clone(),
        bs_pos.clone(),
        cs_neg.clone(),
        lt_ua(&ua_omega, &mnq1)?,
        c1.clone(),
        c3.clone(),
        c10.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 26
    conds.push(and_(&[
        e0(&p, &s2(&q, &one())?)?,
        e0(&t, &z_)?,
        phi0.clone(),
        s_pos.clone(),
        bs_pos.clone(),
        cs_ge0.clone(),
        lt(&cstar_pi, &ua_omega)?,
        c1.clone(),
        c2.clone(),
        c10.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 27
    conds.push(and_(&[
        e0(&p, &a2(&q, &one())?)?,
        e0(&s, &z_)?,
        phi0.clone(),
        t_pos.clone(),
        bs_pos.clone(),
        cs_ge0.clone(),
        lt(&cstar_pi, &ua_omega)?,
        c1.clone(),
        c3.clone(),
        c10.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 28
    conds.push(and_(&[
        lt(&p, &s2(&q, &one())?)?,
        e0(&t, &z_)?,
        phi0.clone(),
        s_pos.clone(),
        bs_pos.clone(),
        cs_ge0.clone(),
        lt(&cstar_pi, &ua_omega)?,
        lt_ua(&ua_omega, &mnp1)?,
        c1.clone(),
        c2.clone(),
        c10.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 29
    conds.push(and_(&[
        gt(&p, &a2(&q, &one())?)?,
        e0(&s, &z_)?,
        phi0.clone(),
        t_pos.clone(),
        bs_pos.clone(),
        cs_ge0.clone(),
        lt(&cstar_pi, &ua_omega)?,
        lt_ua(&ua_omega, &mnq1)?,
        c1.clone(),
        c3.clone(),
        c10.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 30
    conds.push(and_(&[
        e0(&n, &z_)?,
        phi0.clone(),
        gt(&a2(&s, &t)?, &z_)?,
        m_pos.clone(),
        cs_pos.clone(),
        bs_neg.clone(),
        lt_ua(&ua_sigma, &stu1)?,
        c1.clone(),
        c2.clone(),
        c12.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 31
    conds.push(and_(&[
        e0(&m, &z_)?,
        phi0.clone(),
        gt(&a2(&s, &t)?, &v)?,
        n_pos.clone(),
        cs_pos.clone(),
        bs_neg.clone(),
        lt_ua(&ua_sigma, &stv1)?,
        c1.clone(),
        c3.clone(),
        c12.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    let bs1: Basic = a2(&bstar, &one())?;
    // 32
    conds.push(and_(&[
        e0(&n, &z_)?,
        phi0.clone(),
        e0(&u, &s2(&v, &one())?)?,
        m_pos.clone(),
        cs_pos.clone(),
        bs_ge0.clone(),
        lt(&bstar_pi, &ua_sigma)?,
        lt_ua(&ua_sigma, &bs1)?,
        c1.clone(),
        c2.clone(),
        c12.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 33
    conds.push(and_(&[
        e0(&m, &z_)?,
        phi0.clone(),
        e0(&u, &a2(&v, &one())?)?,
        n_pos.clone(),
        cs_pos.clone(),
        bs_ge0.clone(),
        lt(&bstar_pi, &ua_sigma)?,
        lt_ua(&ua_sigma, &bs1)?,
        c1.clone(),
        c3.clone(),
        c12.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 34
    conds.push(and_(&[
        e0(&n, &z_)?,
        phi0.clone(),
        lt(&u, &s2(&v, &one())?)?,
        m_pos,
        cs_pos.clone(),
        bs_ge0.clone(),
        lt(&bstar_pi, &ua_sigma)?,
        lt_ua(&ua_sigma, &stu1)?,
        c1.clone(),
        c2,
        c12.clone(),
        c14.clone(),
        c15.clone(),
    ]));
    // 35
    conds.push(and_(&[
        e0(&m, &z_)?,
        phi0,
        gt(&u, &a2(&v, &one())?)?,
        n_pos,
        cs_pos,
        bs_ge0,
        lt(&bstar_pi, &ua_sigma)?,
        lt_ua(&ua_sigma, &stv1)?,
        c1,
        c3,
        c12,
        c14,
        c15,
    ]));
    Ok(or_(&conds))
}

/// The integral of `g1*g2` over `(0, oo)` as a G-function, the conditions
/// assumed ([L], section 5.6.2, equation (1)) (`_int0oo`).
fn int0oo(g1: &Basic, g2: &Basic, x: &Basic) -> R {
    let (eta, _) = get_coeff_exp(&g_argument(g1), x)?;
    let (omega, _) = get_coeff_exp(&g_argument(g2), x)?;
    let negs = |params: Vec<Basic>| -> PolyResult<Vec<Basic>> { params.iter().map(ng).collect() };
    let mut a1: Vec<Basic> = negs(g_bm(g1))?;
    a1.extend(g_an(g2));
    let mut a2_: Vec<Basic> = g_aother(g2);
    a2_.extend(negs(g_bother(g1))?);
    let mut b1: Vec<Basic> = negs(g_an(g1))?;
    b1.extend(g_bm(g2));
    let mut b2: Vec<Basic> = g_bother(g2);
    b2.extend(negs(g_aother(g1))?);
    d2(&g_(&a1, &a2_, &b1, &b2, &d2(&omega, &eta)?)?, &eta)
}

// ---------------------------------------------------------------------
// the rewriting as G-functions

/// A term `C * x**s * G` of a rewriting: `(C, s, G)`.
pub type GTerm = (Basic, Basic, Basic);
/// The terms of a rewriting and the condition under which they hold.
pub type Rewriting = (Vec<GTerm>, Basic);

/// Whether a caught error is one of Python's `except` classes and not a
/// part which is not ported (which passes through).
fn not_ported(e: &PolyError) -> bool {
    matches!(e, PolyError::NotImplemented(m) if m.starts_with(crate::error::NOT_PORTED_PREFIX))
}

/// A binding with its numbers lifted to polar ones and its exponents
/// unpolarified (`_polar_binding`).
fn polar_binding(value: &Basic) -> R {
    if crate::core::logic::is_boolean(value) || is_tuple_like(value) {
        return Ok(value.clone());
    }
    unpolarify(&polarify(value, true)?, true)
}

/// `f` as a sum of G-functions by the table (`_rewrite_table`).
fn rewrite_table(f: &Basic, x: &Basic) -> PolyResult<Option<Rewriting>> {
    let z: &Basic = &table_symbols().z;
    let fz: Basic = subs(f, x, z)?;
    let table = lookup_table()?;
    let Some(entries) = table.get(&mytype(&fz, z)) else {
        return Ok(None);
    };
    for entry in entries {
        let Some(matched) =
            crate::core::matching::match_pattern(&fz, &entry.formula, &[], false)?.filter(|m| !m.is_empty())
        else {
            continue;
        };
        let sb: Vec<(Basic, Basic)> = matched
            .iter()
            .map(|(k, v)| Ok((k.clone(), polar_binding(v)?)))
            .collect::<PolyResult<_>>()?;
        match entry_hint(entry, &sb) {
            Ok(true) => {}
            Ok(false) | Err(PolyError::Type(_)) => continue,
            Err(e) => return Err(e),
        }
        let cond: Basic = unpolarify(&subs_map(&entry.cond, &sb)?, false)?;
        if is_false(&eval_cond(&cond)?) {
            continue;
        }
        let mut res: Vec<GTerm> = Vec::new();
        for (fac, g) in entry_terms(entry, &sb)? {
            let attempt = || -> PolyResult<(Basic, Basic, Basic)> {
                let value: Basic = unpolarify(&subs(&subs_map(&fac, &sb)?, z, x)?, true)?;
                let (coeff, power) = get_coeff_exp(&value, x)?;
                let h: Basic = replace_symbol(&subs_g(&g, &sb)?, z, x)?;
                Ok((coeff, power, h))
            };
            let (coeff, power, h) = match attempt() {
                Ok(v) => v,
                Err(e) if is_value_error(&e) => continue,
                Err(e) => return Err(e),
            };
            if has_infinity(&coeff) || has_infinity(&power) || has_infinity(&h) {
                continue;
            }
            let [an, aother, bm, bother] = lists(&h);
            let h: Basic = g_(&an, &aother, &bm, &bother, &unpolarify(&g_argument(&h), true)?)?;
            res.push((coeff, power, h));
        }
        if !res.is_empty() {
            return Ok(Some((res, cond)));
        }
    }
    Ok(None)
}

/// `f` as one G-function by its Mellin transform in the table of
/// [`crate::integrals::mellin`] (one kernel), inverted as a Mellin–Barnes
/// integral, checked at a sample point when `f` has no parameters
/// (`_rewrite_mellin`).
fn rewrite_mellin(f: &Basic, x: &Basic) -> PolyResult<Option<Rewriting>> {
    use crate::integrals::mellin::{decompose_integrand, poles_separated, GammaQuotient};
    let Some(product) = decompose_integrand(f, x, None)? else {
        return Ok(None);
    };
    if product.one_minus_power != 0 || product.log_power != 0 || product.matches.len() != 1 {
        return Ok(None);
    }
    let quotient: Option<GammaQuotient> = match product.quotient(&[]) {
        Ok(q) => q,
        Err(e) if is_value_error(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some(quotient) = quotient.filter(|q| q.is_gamma_quotient()) else {
        return Ok(None);
    };
    if !is_true(&lt(&quotient.lower, &quotient.upper)?) || poles_separated(&quotient)? != Some(true) {
        return Ok(None);
    }
    let mut inverse: GammaQuotient = quotient.clone();
    inverse.bases.push((x.clone(), int(-1)));
    let Some(barnes) = crate::integrals::slater::mellin_barnes(&inverse)? else {
        return Ok(None);
    };
    let attempt = || -> PolyResult<(Basic, Basic, Basic)> {
        let g: Basic = g_(&barnes.an, &barnes.ap, &barnes.bm, &barnes.bq, &barnes.z)?;
        let (coeff, power) = get_coeff_exp(&barnes.prefactor, x)?;
        get_coeff_exp(&g_argument(&g), x)?;
        Ok((coeff, power, g))
    };
    let (coeff, power, g) = match attempt() {
        Ok(v) => v,
        Err(e) if is_value_error(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    if has_infinity(&coeff) || has_infinity(&g) {
        return Ok(None);
    }
    let free = f.free_symbols();
    if free.len() == 1 && free.contains(x) {
        let point: Basic = rational(7, 10);
        let value = crate::numeric::reliable_complex(&subs(f, x, &point)?, 15, &[]);
        let represented =
            crate::numeric::reliable_complex(&subs(&m2(&m2(&coeff, &p2(x, &power)?)?, &g)?, x, &point)?, 15, &[]);
        let (Some(v), Some(r)) = (value, represented) else {
            if value.is_some() {
                // the Rust core has no numeric G-function (mpmath's meijerg)
                return Err(crate::error::not_ported(
                    "the numeric value of a Meijer G-function (mpmath.meijerg)",
                ));
            }
            return Ok(None);
        };
        let diff: f64 = ((v.0 - r.0).powi(2) + (v.1 - r.1).powi(2)).sqrt();
        let size: f64 = (v.0 * v.0 + v.1 * v.1).sqrt();
        if diff > 1e-8 * size.max(1.0) {
            return Ok(None);
        }
    }
    Ok(Some((vec![(coeff, power, g)], quotient.condition.clone())))
}

// the recursive Mellin transform: the transform of a product of two
// G-functions by the theorem, written as gamma functions and inverted

fn disjuncts(b: &Basic) -> Vec<Basic> {
    if matches!(b.head(), Head::Or) {
        b.args().to_vec()
    } else {
        vec![b.clone()]
    }
}
fn conjuncts(b: &Basic) -> Vec<Basic> {
    if matches!(b.head(), Head::And) {
        b.args().to_vec()
    } else {
        vec![b.clone()]
    }
}

/// `e.as_real_imag()[0]`.
fn real_part(e: &Basic) -> R {
    match crate::functions::elementary::complexes::as_real_imag(e, true, None)? {
        Some((r, _)) => Ok(r),
        None => re(e),
    }
}

/// A strip `a < re(s) < b` and the auxiliary conditions.
type Strip = (Basic, Basic, Basic);

/// `cond` (a conjunction) as a strip `a < re(s) < b` and the auxiliary
/// conditions (`_strip`).
fn strip_of(cond: &Basic, s: &Basic) -> PolyResult<Strip> {
    let max_ = |a: &Basic, b: &Basic| apply(Function::Max, &[a.clone(), b.clone()]);
    let min_ = |a: &Basic, b: &Basic| apply(Function::Min, &[a.clone(), b.clone()]);
    let mut a: Basic = neg_oo();
    let mut b: Basic = oo();
    let mut aux: Basic = true_();
    let t: Basic = Dummy::new("t", &[(Predicate::real, true)]).0;
    let re_s: Basic = re(s)?;
    for c in conjuncts(cond) {
        let mut a_: Basic = oo();
        let mut b_: Basic = neg_oo();
        let mut aux_: Vec<Basic> = Vec::new();
        for d in disjuncts(&c) {
            let replaced: Basic =
                crate::core::function::replace_where(&d, &|n| is_fn(n, &[Function::Re]), &|n| real_part(&n.args()[0]))?;
            let d_: Basic = subs(&replaced, &re_s, &t)?;
            if !is_relational(&d) || matches!(rel_op(&d), RelOp::Eq | RelOp::Ne) || d_.has(s) || !d_.has(&t) {
                aux_.push(d);
                continue;
            }
            let soln: Basic = match crate::solvers::inequalities::solve_inequality(&d_, &t) {
                Ok(v) => v,
                Err(e)
                    if !not_ported(&e)
                        && matches!(
                            e,
                            PolyError::NotImplemented(_) | PolyError::Value(_) | PolyError::Type(_)
                        ) =>
                {
                    aux_.push(d);
                    continue;
                }
                Err(e) => return Err(e),
            };
            if !is_relational(&soln) || matches!(rel_op(&soln), RelOp::Eq | RelOp::Ne) {
                aux_.push(d);
                continue;
            }
            let (smaller, greater) = if matches!(rel_op(&soln), RelOp::Lt | RelOp::Le) {
                (soln.args()[0].clone(), soln.args()[1].clone())
            } else {
                (soln.args()[1].clone(), soln.args()[0].clone())
            };
            if smaller == t {
                b_ = max_(&greater, &b_)?;
            } else if greater == t {
                a_ = min_(&smaller, &a_)?;
            } else {
                aux_.push(d);
            }
        }
        if !matches!(a_.head(), Head::Infinity) && a_ != b {
            a = max_(&a_, &a)?;
        } else if !matches!(b_.head(), Head::NegativeInfinity) && b_ != a {
            b = min_(&b_, &b)?;
        } else {
            aux = and_(&[aux, or_(&aux_)]);
        }
    }
    Ok((a, b, aux))
}

/// The Mellin transform of `f` by the theorem for two G-functions, with
/// its strip (`_mellin_of_product`).
fn mellin_of_product(f: &Basic, x: &Basic, s: &Basic) -> PolyResult<Option<(Basic, Strip)>> {
    let Some((res, cond)) = meijerint_definite_4(&m2(&p2(x, &s2(s, &one())?)?, f)?, x, true)? else {
        return Ok(None);
    };
    if is_false(&cond) {
        return Ok(None);
    }
    let expanded: Basic = crate::simplify::hyperexpand::hyperexpand(
        &res,
        false,
        crate::simplify::hyperexpand::Rewrite::NonRepSmall,
        None,
    )?;
    let big_f: Basic = my_unpolarify(&expanded)?;
    let mut strips: Vec<Strip> = Vec::new();
    for c in disjuncts(&cond) {
        let st: Strip = strip_of(&c, s)?;
        if !is_false(&st.2) {
            strips.push(st);
        }
    }
    if strips.is_empty() {
        return Ok(None);
    }
    let mut keyed: Vec<((crate::core::sorting::SortKey, i64), Strip)> = Vec::new();
    for st in strips {
        let ops: i64 = crate::core::function::count_ops(&st.2)?;
        keyed.push(((sort_key(&s2(&st.0, &st.1)?), ops), st));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Some((big_f, keyed.swap_remove(0).1)))
}

/// `sin(m*s + n)` as `fac/(gamma1*gamma2)`, the gammas on the right side
/// of the strip: `(gamma1, gamma2, fac)` (`_rewrite_sin`).
fn rewrite_sin(m: &Basic, n: &Basic, s: &Basic, a: Option<&Basic>) -> PolyResult<(Basic, Basic, Basic)> {
    let m_: Basic = expand_mul(&d2(m, &pi())?)?;
    let n_: Basic = expand_mul(&d2(n, &pi())?)?;
    let a: Basic = a.cloned().unwrap_or_else(zero);
    let r: Basic = f1(Function::Ceiling, &s2(&m2(&ng(&m_)?, &a)?, &real_part(&n_)?)?)?;
    Ok((
        gamma(&a2(&a2(&m2(&m_, s)?, &n_)?, &r)?)?,
        gamma(&s2(&s2(&s2(&one(), &n_)?, &r)?, &m2(&m_, s)?)?)?,
        m2(&p2(&int(-1), &r)?, &pi())?,
    ))
}

/// The error of `_rewrite_gamma` (`_GammaRewriteError`, a `ValueError`).
fn gamma_rewrite_error(message: &str) -> PolyError {
    value_error(message)
}

/// `e.as_poly(s)`: `None` when it is not a polynomial in `s`.
fn as_poly(e: &Basic, s: &Basic) -> PolyResult<Option<crate::polys::polytools::Poly>> {
    use crate::error::PolysErrorKind;
    match crate::polys::polytools::Poly::new(e, std::slice::from_ref(s)) {
        Ok(p) => Ok(Some(p)),
        Err(PolyError::Polys(PolysErrorKind::PolynomialError | PolysErrorKind::GeneratorsNeeded, _)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn linear_arg(arg: &Basic, s: &Basic) -> PolyResult<(Basic, Basic)> {
    let Some(p) = as_poly(arg, s)? else {
        return Err(gamma_rewrite_error(&format!("not linear in s: {}", arg)));
    };
    if p.degree(0)? != 1 {
        return Err(gamma_rewrite_error(&format!("not linear in s: {}", arg)));
    }
    let c: Vec<Basic> = p.all_coeffs()?;
    Ok((c[0].clone(), c[1].clone()))
}

fn sin_node(a: &Basic) -> Basic {
    Basic::new(Head::Function(Function::Sin), vec![a.clone()])
}

fn gcd_i(a: Int, b: Int) -> Int {
    crate::numbers::gcd(&a, &b)
}

/// The parameters, the argument, the exponent and the factor of an
/// inverse Mellin transform.
type GammaRewriting = (Vec<Basic>, Vec<Basic>, Vec<Basic>, Vec<Basic>, Basic, Basic, Basic);

/// `f(s)` as a product of gamma functions: `(an, ap, bm, bq, arg,
/// exponent, fac)` with `G(an, ap, bm, bq, arg/x**exponent)*fac` the
/// inverse Mellin transform of `f` over the strip `(a, b)`; a
/// `ValueError` when a factor has no such form (`_rewrite_gamma`).
fn rewrite_gamma(f: &Basic, s: &Basic, a: Option<&Basic>, b: Option<&Basic>) -> PolyResult<GammaRewriting> {
    let mut a_: Option<Basic> = a.cloned();
    let mut b_: Option<Basic> = b.cloned();

    // 1) the constant c such that the inversion is in s' = c*s
    let mut multipliers: Vec<Basic> = Vec::new();
    let independent_part = |g: &Basic| -> R {
        let argument: Basic = g.args()[0].clone();
        if argument.is_add() {
            return Ok(crate::core::expr::as_independent(&argument, s, true)?.1);
        }
        Ok(argument)
    };
    for g in atoms_where(f, &|n| is_fn(n, &[Function::Gamma])) {
        if !g.has(s) {
            continue;
        }
        multipliers.push(coeff_mul(&independent_part(&g)?, s)?.0);
    }
    for g in atoms_where(f, &|n| {
        is_fn(n, &[Function::Sin, Function::Cos, Function::Tan, Function::Cot])
    }) {
        if !g.has(s) {
            continue;
        }
        multipliers.push(d2(&coeff_mul(&independent_part(&g)?, s)?.0, &pi())?);
    }
    let multipliers: Vec<Basic> = multipliers
        .into_iter()
        .map(|m| {
            if yes(&m, Predicate::extended_real) {
                abs_(&m)
            } else {
                Ok(m)
            }
        })
        .collect::<PolyResult<_>>()?;
    let mut common: Basic = one();
    for m in &multipliers {
        if !is_rational(m) {
            common = m.clone();
            break;
        }
    }
    let multipliers: Vec<Basic> = multipliers.iter().map(|m| d2(m, &common)).collect::<PolyResult<_>>()?;
    let rationals: Vec<Rat> = multipliers.iter().filter_map(rat_of).collect();
    if rationals.len() != multipliers.len() || !yes(&common, Predicate::extended_real) {
        return Err(gamma_rewrite_error("Nonrational multiplier"));
    }
    let mut lcm_q: Int = Int::from(1);
    for m in &rationals {
        let q: Int = crate::numbers::denom(m);
        lcm_q = lcm_q.clone() / gcd_i(lcm_q.clone(), q.clone()) * q;
    }
    let mut s_multiplier: Basic = d2(&common, &Basic::integer(lcm_q))?;
    if s_multiplier == common && !rationals.is_empty() {
        let mut g: Int = crate::numbers::numer(&rationals[0]);
        for m in &rationals[1..] {
            g = gcd_i(g, crate::numbers::numer(m));
        }
        s_multiplier = m2(&common, &Basic::integer(g))?;
    }
    let f: Basic = subs(f, s, &d2(s, &s_multiplier)?)?;
    let mut fac: Basic = d2(&one(), &s_multiplier)?;
    let exponent: Basic = d2(&one(), &s_multiplier)?;
    if let Some(v) = &a_ {
        a_ = Some(m2(v, &s_multiplier)?);
    }
    if let Some(v) = &b_ {
        b_ = Some(m2(v, &s_multiplier)?);
    }

    // whether the pole at c lies to the left of the strip
    let left = |c: &Basic, is_numer: bool| -> PolyResult<Option<bool>> {
        let c: Basic = expand(&re(c)?)?;
        if a_.is_none() && b_.as_ref().is_some_and(|b| matches!(b.head(), Head::Infinity)) {
            return Ok(Some(true));
        }
        let Some(av) = &a_ else {
            return Ok(Some(match &b_ {
                Some(bv) => is_true(&lt(&c, bv)?),
                None => false,
            }));
        };
        let Some(bv) = &b_ else {
            return Ok(Some(is_true(&le(&c, av)?)));
        };
        if is_true(&ge(&c, bv)?) {
            return Ok(Some(false));
        }
        if is_true(&le(&c, av)?) {
            return Ok(Some(true));
        }
        if is_numer {
            return Ok(None);
        }
        if !av.free_symbols().is_empty() || !bv.free_symbols().is_empty() || !c.free_symbols().is_empty() {
            return Ok(None);
        }
        Err(gamma_rewrite_error("Pole inside critical strip?"))
    };

    // 2) the factors as gamma functions of s and exponentials of s
    let (numer, denom) = crate::simplify::radsimp::fraction(&f)?;
    let mut pending: Vec<(Basic, bool)> = mul_args(&numer).into_iter().map(|e| (e, true)).collect();
    pending.extend(mul_args(&denom).into_iter().map(|e| (e, false)));
    let mut facs: Vec<Basic> = Vec::new();
    let mut dfacs: Vec<Basic> = Vec::new();
    let mut numer_gammas: Vec<(Basic, Basic)> = Vec::new();
    let mut denom_gammas: Vec<(Basic, Basic)> = Vec::new();
    let mut exponentials: Vec<Basic> = Vec::new();
    while let Some((fact, is_numer)) = pending.pop() {
        let has_s: bool = fact.has(s);
        let poly = if has_s { as_poly(&fact, s)? } else { None };
        macro_rules! ufacs {
            () => {
                if is_numer {
                    &mut facs
                } else {
                    &mut dfacs
                }
            };
        }
        macro_rules! gammas {
            ($upper:expr, $v:expr) => {
                if is_numer == $upper {
                    numer_gammas.push($v)
                } else {
                    denom_gammas.push($v)
                }
            };
        }
        if !has_s {
            ufacs!().push(fact);
        } else if fact.is_pow() || is_fn(&fact, &[Function::Exp]) {
            let (mut base, exp_) = if fact.is_pow() {
                (fact.args()[0].clone(), fact.args()[1].clone())
            } else {
                (exp_polar(&one())?, fact.args()[0].clone())
            };
            if is_integer_node(&exp_) {
                let k: i64 = int_value(&exp_).ok_or_else(|| gamma_rewrite_error("exponent too large"))?;
                let cond: bool = if k > 0 { is_numer } else { !is_numer };
                for _ in 0..k.abs() {
                    pending.push((base.clone(), cond));
                }
                continue;
            }
            if base.has(s) {
                return Err(gamma_rewrite_error(&format!("Unrecognised form {}", fact)));
            }
            let (ca, cb) = linear_arg(&exp_, s)?;
            if !is_numer {
                base = d2(&one(), &base)?;
            }
            exponentials.push(p2(&base, &ca)?);
            facs.push(p2(&base, &cb)?);
        } else if let Some(poly) = poly {
            if poly.degree(0)? != 1 {
                let found =
                    crate::polys::polyroots::roots(&poly.as_expr()?, std::slice::from_ref(s), true, true, None)?;
                let total: usize = found.values().sum();
                if total as i64 != poly.degree(0)? {
                    return Err(gamma_rewrite_error(&format!("the roots of {}", fact)));
                }
                ufacs!().push(poly.LC()?);
                for (root, multiplicity) in found.iter() {
                    for _ in 0..*multiplicity {
                        pending.push((s2(s, root)?, is_numer));
                    }
                }
                continue;
            }
            let (ca, cc) = linear_arg(&fact, s)?;
            ufacs!().push(ca.clone());
            let c: Basic = d2(&ng(&cc)?, &ca)?;
            if left(&c, is_numer)? == Some(true) {
                gammas!(true, (one(), a2(&ng(&c)?, &one())?));
                gammas!(false, (one(), ng(&c)?));
            } else {
                ufacs!().push(int(-1));
                gammas!(true, (int(-1), a2(&c, &one())?));
                gammas!(false, (int(-1), c));
            }
        } else if is_fn(&fact, &[Function::Gamma]) {
            let (ca, cb) = linear_arg(&fact.args()[0], s)?;
            if is_numer {
                let pole: Basic = d2(&ng(&cb)?, &ca)?;
                if (is_true(&gt(&ca, &zero())?) && left(&pole, is_numer)? == Some(false))
                    || (is_true(&lt(&ca, &zero())?) && left(&pole, is_numer)? == Some(true))
                {
                    return Err(gamma_rewrite_error("Gammas partially over the strip."));
                }
            }
            gammas!(true, (ca, cb));
        } else if is_fn(&fact, &[Function::Sin]) {
            let argument: Basic = fact.args()[0].clone();
            let (gamma1, gamma2, fac_) = if is_numer {
                (
                    gamma(&d2(&argument, &pi())?)?,
                    gamma(&s2(&one(), &d2(&argument, &pi())?)?)?,
                    pi(),
                )
            } else {
                let (m_, n_) = linear_arg(&argument, s)?;
                rewrite_sin(&m_, &n_, s, a_.as_ref())?
            };
            pending.push((gamma1, !is_numer));
            pending.push((gamma2, !is_numer));
            ufacs!().push(fac_);
        } else if is_fn(&fact, &[Function::Tan]) {
            let argument: Basic = fact.args()[0].clone();
            pending.push((sin_node(&argument), is_numer));
            pending.push((sin_node(&s2(&d2(&pi(), &int(2))?, &argument)?), !is_numer));
        } else if is_fn(&fact, &[Function::Cos]) {
            let argument: Basic = fact.args()[0].clone();
            pending.push((sin_node(&s2(&d2(&pi(), &int(2))?, &argument)?), is_numer));
        } else if is_fn(&fact, &[Function::Cot]) {
            let argument: Basic = fact.args()[0].clone();
            pending.push((sin_node(&s2(&d2(&pi(), &int(2))?, &argument)?), is_numer));
            pending.push((sin_node(&argument), !is_numer));
        } else {
            return Err(gamma_rewrite_error(&format!("Unrecognised form {}", fact)));
        }
    }
    fac = d2(&m2(&fac, &mul_n(&facs)?)?, &mul_n(&dfacs)?)?;

    // 3) the gamma functions of argument c + s or c - s
    let mut an: Vec<Basic> = Vec::new();
    let mut ap: Vec<Basic> = Vec::new();
    let mut bm: Vec<Basic> = Vec::new();
    let mut bq: Vec<Basic> = Vec::new();
    for (mut gammas, is_numer) in [(numer_gammas, true), (denom_gammas, false)] {
        while let Some((ca, c)) = gammas.pop() {
            if ca != int(-1) && ca != one() {
                // Gauss's multiplication theorem
                if !is_integer_node(&ca) {
                    return Err(gamma_rewrite_error("a is not an integer"));
                }
                let size: i64 = int_value(&ca)
                    .ok_or_else(|| gamma_rewrite_error("a is too large"))?
                    .abs();
                let newa: Basic = d2(&ca, &int(size))?;
                for k in 0..size {
                    gammas.push((newa.clone(), a2(&d2(&c, &int(size))?, &rational(k, size))?));
                }
                let factor_: Basic = m2(
                    &p2(&m2(&int(2), &pi())?, &rational(1 - size, 2))?,
                    &p2(&int(size), &s2(&c, &half())?)?,
                )?;
                if is_numer {
                    fac = m2(
                        &m2(&fac, &p2(&m2(&int(2), &pi())?, &rational(1 - size, 2))?)?,
                        &p2(&int(size), &s2(&c, &half())?)?,
                    )?;
                    exponentials.push(p2(&int(size), &ca)?);
                } else {
                    fac = d2(&fac, &factor_)?;
                    exponentials.push(p2(&int(size), &ng(&ca)?)?);
                }
                continue;
            }
            let (plus, minus) = if is_numer {
                (&mut an, &mut bm)
            } else {
                (&mut bq, &mut ap)
            };
            if ca == one() {
                plus.push(s2(&one(), &c)?);
            } else {
                minus.push(c);
            }
        }
    }
    Ok((
        sorted(an),
        sorted(ap),
        sorted(bm),
        sorted(bq),
        mul_n(&exponentials)?,
        exponent,
        fac,
    ))
}

/// A real part, irrational, strictly inside the strip when its ends are
/// finite real numbers; `None` otherwise (`_contour_line`).
fn contour_line(strip: &(Basic, Basic)) -> PolyResult<Option<Basic>> {
    let (a, b) = strip;
    let number = |e: &Basic| crate::core::assumptions::is_number(e) && yes(e, Predicate::finite);
    if !(number(a) && number(b) && is_true(&lt(a, b)?)) {
        return Ok(None);
    }
    Ok(Some(a2(a, &d2(&m2(&s2(b, a)?, &sqrt(&int(2))?)?, &int(2))?)?))
}

/// `F` with each `gamma(alpha*s + beta)` (`alpha` rational) whose poles lie
/// on both sides of the line `re(s) = c` written by its recurrence as a
/// gamma whose poles lie on one side times a rational function of `s`
/// (`_shift_gammas`).
fn shift_gammas(big_f: &Basic, s: &Basic, c: &Basic) -> R {
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for g in atoms_where(big_f, &|n| is_fn(n, &[Function::Gamma])) {
        let u: Basic = g.args()[0].clone();
        let alpha: Basic = crate::core::function::diff(&u, &[(s.clone(), 1)])?;
        let beta: Basic = expand(&s2(&u, &m2(&alpha, s)?)?)?;
        if !is_rational(&alpha)
            || alpha == zero()
            || beta.free_symbols().contains(s)
            || !crate::core::assumptions::is_number(&beta)
        {
            continue;
        }
        // the pole nearest to the side the poles do not extend to
        let p0: Basic = d2(&ng(&beta)?, &alpha)?;
        let distance: Basic = m2(&s2(&p0, c)?, &alpha)?;
        if !is_true(&gt(&distance, &zero())?) {
            continue;
        }
        let shift: Basic = f1(Function::Floor, &distance)?;
        let Some(k) = int_value(&shift) else {
            continue;
        };
        let n: i64 = k + 1;
        let mut denominator: Basic = one();
        for j in 0..n {
            denominator = m2(&denominator, &a2(&u, &int(j))?)?;
        }
        rule.push((g.clone(), d2(&gamma(&a2(&u, &int(n))?)?, &denominator)?));
    }
    if rule.is_empty() {
        return Ok(big_f.clone());
    }
    xreplace(big_f, &rule)
}

/// The inverse Mellin transform of the sum `F` term by term, all the terms
/// inverted on one line of the strip; `None` when the strip has no numeric
/// ends or a term cannot be inverted there (`_inverse_mellin_on_line`).
fn inverse_mellin_on_line(big_f: &Basic, s: &Basic, x: &Basic, strip: &(Basic, Basic)) -> PolyResult<Option<Basic>> {
    let Some(line) = contour_line(strip)? else {
        return Ok(None);
    };
    let on: (Basic, Basic) = (line.clone(), line.clone());
    let mut terms: Vec<Basic> = Vec::new();
    for term in big_f.args() {
        let mut found: Option<Basic> = inverse_mellin_meijerg(term, s, x, &on)?;
        if found.is_none() {
            let shifted_: Basic = shift_gammas(term, s, &line)?;
            if shifted_ != *term {
                found = inverse_mellin_meijerg(&factor(&shifted_)?, s, x, &on)?;
            }
        }
        let Some(found) = found else {
            return Ok(None);
        };
        terms.push(found);
    }
    Ok(Some(add_n(&terms)?))
}

/// The inverse Mellin transform of `F` over the strip as a sum of
/// G-functions (times constants) of `x`; `None` when a term has no gamma
/// form or the G-function is not valid (`_inverse_mellin_meijerg`).
fn inverse_mellin_meijerg(big_f: &Basic, s: &Basic, x: &Basic, strip: &(Basic, Basic)) -> PolyResult<Option<Basic>> {
    let big_f: Basic =
        crate::core::rewriting::rewrite(big_f, &[], "gamma", &crate::core::rewriting::RewriteHints::default())?;
    let forms: [fn(&Basic) -> R; 3] = [factor, expand_mul, expand];
    for form in forms {
        let g: Basic = match form(&big_f) {
            Ok(g) => g,
            Err(e) if is_polynomial_error(&e) => continue,
            Err(e) => return Err(e),
        };
        if g.is_add() {
            if let Some(on_line) = inverse_mellin_on_line(&g, s, x, strip)? {
                return Ok(Some(on_line));
            }
            let mut terms: Vec<Basic> = Vec::new();
            for term in g.args() {
                let Some(found) = inverse_mellin_meijerg(term, s, x, strip)? else {
                    return Ok(None);
                };
                terms.push(found);
            }
            return Ok(Some(add_n(&terms)?));
        }
        let attempt = || -> PolyResult<(Basic, Basic)> {
            let (an, ap, bm, bq, c, e, fac) = rewrite_gamma(&g, s, Some(&strip.0), Some(&strip.1))?;
            let big_g: Basic = g_(&an, &ap, &bm, &bq, &d2(&c, &p2(x, &e)?)?)?;
            Ok((big_g, fac))
        };
        match attempt() {
            Ok((big_g, fac)) => return Ok(Some(m2(&big_g, &fac)?)),
            Err(e) if matches!(e, PolyError::Value(_) | PolyError::Type(_)) || is_polynomial_error(&e) => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

/// `f` as G-functions by its Mellin transform, computed by the theorem for
/// two G-functions, then inverted as G-functions (`_rewrite_recursive`).
fn rewrite_recursive(f: &Basic, x: &Basic) -> PolyResult<Option<Rewriting>> {
    let s: Basic = dummy("s", "rewrite-single", f, false);
    let mut g: Option<Basic> = None;
    if let Some((big_f, (a, b, _))) = mellin_of_product(f, x, &s)? {
        g = inverse_mellin_meijerg(&big_f, &s, x, &(a, b))?;
    }
    if g.is_none() {
        // by analytic continuation in a scale c of the argument, set to 1
        // at the end
        let scale: Basic = dummy_("a", "rewrite-single", false);
        if free_of(f, &scale) && is_analytic(f, x) {
            if let Some((big_f, (a, b, _))) = mellin_of_product(&subs(f, x, &m2(&scale, x)?)?, x, &s)? {
                let inverse: Option<Basic> = inverse_mellin_meijerg(&big_f, &s, x, &(a, b))?;
                g = match inverse {
                    Some(v) => Some(subs(&v, &scale, &one())?),
                    None => None,
                };
            }
        }
    }
    let Some(g) = g else {
        return Ok(None);
    };
    if has_any(&g, &|n| {
        matches!(n.head(), Head::Infinity | Head::NaN | Head::ComplexInfinity)
    }) {
        return Ok(None);
    }
    let mut res: Vec<GTerm> = Vec::new();
    for term in add_args(&g) {
        let (c, m) = coeff_mul(&term, x)?;
        if m.len() != 1 || !is_meijerg(&m[0]) {
            return Ok(None);
        }
        let h: &Basic = &m[0];
        let (coeff, power) = match get_coeff_exp(&g_argument(h), x) {
            Ok(v) => v,
            Err(e) if is_value_error(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        let lifted: Basic = unpolarify(&polarify(&coeff, true)?, true)?;
        let [an, aother, bm, bother] = lists(h);
        res.push((
            c,
            zero(),
            g_(&an, &aother, &bm, &bother, &m2(&lifted, &p2(x, &power)?)?)?,
        ));
    }
    Ok(Some((res, true_())))
}

thread_local! {
    static REWRITE_CACHE: std::cell::RefCell<HashMap<(Basic, Basic, bool), Option<Rewriting>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// `f` as a sum of single G-functions `C*x**s*G(a*x**b)`, `b` rational
/// and `C` free of `x`: the terms `(C, s, G)` and the condition under
/// which they hold; `None` on failure. `recursive` allows the Mellin
/// transforms when the table has no representation (`rewrite_single`,
/// cached).
pub fn rewrite_single(f: &Basic, x: &Basic, recursive: bool) -> PolyResult<Option<Rewriting>> {
    let key: (Basic, Basic, bool) = (f.clone(), x.clone(), recursive);
    if let Some(found) = REWRITE_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(found);
    }
    let found: Option<Rewriting> = rewrite_single_uncached(f, x, recursive)?;
    REWRITE_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 2000 {
            c.clear();
        }
        c.insert(key, found.clone());
    });
    Ok(found)
}

fn rewrite_single_uncached(f: &Basic, x: &Basic, recursive: bool) -> PolyResult<Option<Rewriting>> {
    if is_meijerg(f) {
        let factored: Basic = crate::polys::polytools::factor(&g_argument(f), std::slice::from_ref(x), false, true)?;
        let (coeff, m) = coeff_mul(&factored, x)?;
        if m.len() != 1 {
            return Ok(None);
        }
        let power: &Basic = &m[0];
        if power.is_pow() {
            if power.args()[0] != *x || !is_rational(&power.args()[1]) {
                return Ok(None);
            }
        } else if power != x {
            return Ok(None);
        }
        let [an, aother, bm, bother] = lists(f);
        return Ok(Some((
            vec![(one(), zero(), g_(&an, &aother, &bm, &bother, &m2(&coeff, power)?)?)],
            true_(),
        )));
    }
    if let Some(found) = rewrite_table(f, x)? {
        return Ok(Some(found));
    }
    if f.is_mul() && f.args().iter().filter(|e| e.has(x)).all(|e| is_fn(e, &[Function::Exp])) {
        // a product of exponentials is one exponential
        let mut exponent: Basic = zero();
        for e in f.args() {
            if is_fn(e, &[Function::Exp]) {
                exponent = a2(&exponent, &e.args()[0])?;
            }
        }
        let others: Vec<Basic> = f
            .args()
            .iter()
            .filter(|e| !is_fn(e, &[Function::Exp]))
            .cloned()
            .collect();
        let combined: Basic = m2(&mul_n(&others)?, &exp(&factor_terms(&exponent)?)?)?;
        if combined != *f {
            if let Some(found) = rewrite_table(&combined, x)? {
                return Ok(Some(found));
            }
        }
    }
    if !recursive {
        return Ok(None);
    }
    if let Some(found) = rewrite_recursive(f, x)? {
        return Ok(Some(found));
    }
    rewrite_mellin(f, x)
}

/// `f` as `fac*po*g`, `g` a sum of single G-functions (`_rewrite1`).
fn rewrite1(f: &Basic, x: &Basic, recursive: bool) -> PolyResult<Option<(Basic, Basic, Vec<GTerm>, Basic)>> {
    let (fac, po, g) = split_mul(f, x)?;
    Ok(rewrite_single(&g, x, recursive)?.map(|(terms, cond)| (fac, po, terms, cond)))
}

type PairKey = (
    usize,
    usize,
    usize,
    usize,
    crate::core::sorting::SortKey,
    crate::core::sorting::SortKey,
);

fn pair_key(pair: &(Basic, Basic), x: &Basic) -> PolyResult<PairKey> {
    let (a, b) = pair;
    Ok((
        exponents(a, x).len().max(exponents(b, x).len()),
        functions(a, x).len().max(functions(b, x).len()),
        find_splitting_points(a, x)?
            .len()
            .max(find_splitting_points(b, x)?.len()),
        node_count(a) + node_count(b),
        sort_key(a),
        sort_key(b),
    ))
}

/// `f` as `fac*po*g1*g2`, `g1` and `g2` sums of single G-functions
/// (`_rewrite2`).
fn rewrite2(f: &Basic, x: &Basic) -> PolyResult<Option<(Basic, Basic, Vec<GTerm>, Vec<GTerm>, Basic)>> {
    let (fac, po, g) = split_mul(f, x)?;
    for e in mul_args_repeated(&g)? {
        if rewrite_single(&e, x, false)?.is_none() {
            return Ok(None);
        }
    }
    let Some(parts) = mul_as_two_parts(&g)? else {
        return Ok(None);
    };
    if parts.is_empty() {
        return Ok(None);
    }
    let mut keyed: Vec<(PairKey, (Basic, Basic))> = Vec::new();
    for pair in parts {
        keyed.push((pair_key(&pair, x)?, pair));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    for recursive in [false, true] {
        for (_, (fac1, fac2)) in &keyed {
            let g1 = rewrite_single(fac1, x, recursive)?;
            let g2 = rewrite_single(fac2, x, recursive)?;
            if let (Some(g1), Some(g2)) = (g1, g2) {
                let cond: Basic = and_(&[g1.1.clone(), g2.1.clone()]);
                if !is_false(&cond) {
                    return Ok(Some((fac.clone(), po.clone(), g1.0, g2.0, cond)));
                }
            }
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// indefinite integration

/// A value and the condition under which it holds.
pub type ConditionalExpr = (Basic, Basic);

/// The hyperbolic functions of `f` as exponentials
/// (`_rewrite_hyperbolics_as_exp`).
fn rewrite_hyperbolics_as_exp(f: &Basic) -> R {
    let mut rule: Vec<(Basic, Basic)> = Vec::new();
    for h in atoms_where(f, &|n| is_fn(n, &HYPERBOLIC)) {
        let r: Basic =
            crate::core::rewriting::rewrite(&h, &[], "exp", &crate::core::rewriting::RewriteHints::default())?;
        rule.push((h, r));
    }
    xreplace(f, &rule)
}

fn ordered_first(results: Vec<ConditionalExpr>) -> ConditionalExpr {
    let key = |r: &ConditionalExpr| (node_count(&r.0) + node_count(&r.1), sort_key(&r.0), sort_key(&r.1));
    let mut best: Option<(_, ConditionalExpr)> = None;
    for r in results {
        let k = key(&r);
        if best.as_ref().map_or(true, |(bk, _)| k < *bk) {
            best = Some((k, r));
        }
    }
    best.expect("results are not empty").1
}

/// The hyperbolic integrand integrated as exponentials, the value
/// collected in the exponentials.
fn collected(rv: ConditionalExpr) -> PolyResult<ConditionalExpr> {
    let value: Basic = factor_terms(&rv.0)?;
    let exponentials: Vec<Basic> = sorted(atoms_where(&value, &|n| is_fn(n, &[Function::Exp])));
    Ok((
        crate::simplify::radsimp::collect(&value, &exponentials, None, Some(false))?,
        rv.1,
    ))
}

/// An antiderivative of `f` by rewriting it as G-functions, with the
/// condition under which it holds; `None` when `f` cannot be rewritten
/// (`meijerint_indefinite`).
pub fn meijerint_indefinite(f: &Basic, x: &Basic) -> PolyResult<Option<ConditionalExpr>> {
    let e: &Basic = f;
    let mut results: Vec<ConditionalExpr> = Vec::new();
    let mut points: Vec<Basic> = find_splitting_points(e, x)?;
    if !points.contains(&zero()) {
        points.push(zero());
    }
    for a in sorted(points) {
        let Some(found) = meijerint_indefinite_1(&subs(e, x, &a2(x, &a)?)?, x)? else {
            continue;
        };
        let res: ConditionalExpr = (subs(&found.0, x, &s2(x, &a)?)?, found.1);
        if has_where(&res.0, &is_hyper_or_g)? && (is_true(&res.1) || has_any(e, &is_hyper_or_g)) {
            results.push(res);
        } else {
            return Ok(Some(res));
        }
    }
    if has_heads(e, &HYPERBOLIC) {
        if let Some(rv) = meijerint_indefinite(&rewrite_hyperbolics_as_exp(e)?, x)? {
            return Ok(Some(collected(rv)?));
        }
    }
    if !results.is_empty() {
        return Ok(Some(ordered_first(results)));
    }
    Ok(None)
}

/// [`meijerint_indefinite`] without the substitutions
/// (`_meijerint_indefinite_1`).
fn meijerint_indefinite_1(f: &Basic, x: &Basic) -> PolyResult<Option<ConditionalExpr>> {
    let Some((fac, po, gl, cond)) = rewrite1(f, x, true)? else {
        return Ok(None);
    };
    let mut res: Basic = zero();
    for (big_c, s, g) in gl {
        let (a, b) = get_coeff_exp(&g_argument(&g), x)?;
        let (_, c) = get_coeff_exp(&po, x)?;
        let c: Basic = a2(&c, &s)?;
        // the substitution t = a*x**b, t**rho*G(params, t) = G(params +
        // rho, t), and the integral of G(params, t) is G(1, params + 1, 0,
        // t) or the one with 1 and 0 exchanged
        let fac_: Basic = d2(&m2(&m2(&fac, &big_c)?, &p2(x, &a2(&one(), &c)?)?)?, &b)?;
        let rho: Basic = d2(&a2(&c, &one())?, &b)?;
        let t: Basic = dummy("t", "meijerint-indefinite", &one(), false);
        let mut nonpositive_integer: bool = false;
        for p in g_bm(&g) {
            let shifted_: Basic = a2(&p, &rho)?;
            if yes(&shifted_, Predicate::integer) && is_true(&le(&shifted_, &zero())?) {
                nonpositive_integer = true;
                break;
            }
        }
        let [an, aother, bm, bother] = lists(&g);
        let r: Basic = if nonpositive_integer {
            let mut ao: Vec<Basic> = aother.clone();
            ao.push(s2(&one(), &rho)?);
            let mut b2: Vec<Basic> = bm.clone();
            b2.push(ng(&rho)?);
            ng(&g_(&an, &ao, &b2, &bother, &t)?)?
        } else {
            let mut a1: Vec<Basic> = an.clone();
            a1.push(s2(&one(), &rho)?);
            let mut bo: Vec<Basic> = bother.clone();
            bo.push(ng(&rho)?);
            g_(&a1, &aother, &bm, &bo, &t)?
        };
        // the antiderivative is most often expected to be defined near 0
        let at_zero: Basic = subs(f, x, &zero())?;
        let place: Option<Basic> = if yes(&b, Predicate::extended_nonnegative) && !has_nan_zoo(&at_zero) {
            Some(zero())
        } else {
            None
        };
        let expanded: Basic = crate::simplify::hyperexpand::hyperexpand(
            &subs(&r, &t, &m2(&a, &p2(x, &b)?)?)?,
            false,
            crate::simplify::hyperexpand::Rewrite::Default,
            place.as_ref(),
        )?;
        res = a2(&res, &powdenest_polar(&m2(&fac_, &expanded)?)?)?;
    }
    let clean = |e: &Basic| -> R {
        // the superfluous powers of x multiplied out and the constants
        // chopped off
        let e: Basic = crate::core::function::expand_mul(&cancel(e)?, false)?;
        add_n(&coeff_add(&e, x)?.1)
    };
    let res: Basic = piecewise_fold(&res, None)?;
    let res: Basic = if matches!(res.head(), Head::Piecewise) {
        let mut pairs: Vec<Basic> = Vec::new();
        for pair in res.args() {
            pairs.push(crate::functions::elementary::piecewise::expr_cond_pair(
                &my_unpolarify(&clean(&pair.args()[0])?)?,
                &pair.args()[1],
            ));
        }
        crate::functions::elementary::piecewise::piecewise_node(pairs)
    } else {
        my_unpolarify(&clean(&res)?)?
    };
    Ok(Some((res, my_unpolarify_cond(&cond)?)))
}

fn is_unevaluated(e: &Basic) -> bool {
    has_any(e, &|n| matches!(n.head(), Head::Integral | Head::NonElementaryIntegral))
}

/// SymPy's `integrate(f, x, meijerg=True, risch=False)`: each term, its
/// constant factor out, by [`meijerint_indefinite`] when the condition is
/// `True`, else by `manualintegrate`; a lone term whose products of `sin`
/// and `cos` are sums is integrated as that sum. `None` when a term has no
/// antiderivative (`meijer_antiderivative`).
pub fn meijer_antiderivative(f: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    let e: &Basic = f;
    if matches!(e.head(), Head::Piecewise) {
        return Err(crate::error::not_ported(
            "the antiderivative of a Piecewise (indefinite.verified_antiderivative)",
        ));
    }
    if !e.has(x) {
        return Ok(Some(m2(e, x)?));
    }
    let terms: Vec<Basic> = add_args(e);
    let mut parts: Vec<Basic> = Vec::new();
    for term in &terms {
        let (coeff, g) = crate::core::expr::as_independent(term, x, false)?;
        let mut h: Option<Basic> = None;
        if let Some(found) = meijerint_indefinite(&g, x)? {
            if !is_true(&found.1) {
                // SymPy's Piecewise with the unevaluated integral
                return Ok(None);
            }
            h = Some(found.0);
        } else {
            let manual: Basic = crate::integrals::manualintegrate::manualintegrate(&g, x)?;
            if !is_unevaluated(&manual) {
                h = Some(manual);
            }
        }
        if h.is_none() && terms.len() == 1 {
            let expanded: Basic = crate::core::function::expand_with(
                &crate::simplify::fu::sincos_to_sum(e)?,
                false,
                crate::core::function::Hints::DEFAULT,
            )?;
            if expanded.is_add() {
                return meijer_antiderivative(&expanded, x);
            }
        }
        let Some(h) = h else {
            return Ok(None);
        };
        parts.push(m2(&coeff, &h)?);
    }
    Ok(Some(add_n(&parts)?))
}

// ---------------------------------------------------------------------
// definite integration

fn minus(value: Option<ConditionalExpr>) -> PolyResult<Option<ConditionalExpr>> {
    match value {
        None => Ok(None),
        Some((v, c)) => Ok(Some((ng(&v)?, c))),
    }
}

/// The integral of `f` over `[a, b]` by rewriting it as a product of two
/// G-functions or as one: the value and its convergence condition, `None`
/// when no theorem applies (`meijerint_definite`).
pub fn meijerint_definite(f: &Basic, x: &Basic, a: &Basic, b: &Basic) -> PolyResult<Option<ConditionalExpr>> {
    let (mut e, mut lower, mut upper) = (f.clone(), a.clone(), b.clone());
    if has_heads(&e, &[Function::DiracDelta, Function::SingularityFunction]) {
        return Ok(None);
    }
    let (e_, x_, a_, b_) = (e.clone(), x.clone(), lower.clone(), upper.clone());
    // a dummy in case a bound has x
    let d: Basic = Dummy::new("x", &[]).0;
    e = subs(&e, x, &d)?;
    let x: &Basic = &d;
    if lower == upper {
        return Ok(Some((zero(), true_())));
    }
    let mut results: Vec<ConditionalExpr> = Vec::new();
    let is_oo = |v: &Basic| matches!(v.head(), Head::Infinity);
    let is_neg_oo = |v: &Basic| matches!(v.head(), Head::NegativeInfinity);
    let keep = |res: ConditionalExpr, results: &mut Vec<ConditionalExpr>| -> PolyResult<Option<ConditionalExpr>> {
        if has_where(&res.0, &is_meijerg)? {
            results.push(res);
            Ok(None)
        } else {
            Ok(Some(res))
        }
    };
    if is_neg_oo(&lower) && !is_oo(&upper) {
        return meijerint_definite(&subs(&e, x, &ng(x)?)?, x, &ng(&upper)?, &ng(&lower)?);
    } else if is_neg_oo(&lower) {
        // over (-oo, oo): split at a sensible point
        let mut innermost: Vec<Basic> = sorted(find_splitting_points(&e, x)?);
        innermost.reverse();
        innermost.push(zero());
        for c in innermost {
            if !yes(&c, Predicate::extended_real) {
                continue;
            }
            let Some(res1) = meijerint_definite_2(&subs(&e, x, &a2(x, &c)?)?, x)? else {
                continue;
            };
            let Some(res2) = meijerint_definite_2(&subs(&e, x, &s2(&c, x)?)?, x)? else {
                continue;
            };
            let cond: Basic = condsimp(&and_(&[res1.1.clone(), res2.1.clone()]), true)?;
            if is_false(&cond) {
                continue;
            }
            return Ok(Some((a2(&res1.0, &res2.0)?, cond)));
        }
    } else if is_oo(&lower) {
        return minus(meijerint_definite(&e, x, &upper, &oo())?);
    } else if lower == zero() && is_oo(&upper) {
        if let Some(res) = meijerint_definite_2(&e, x)? {
            if let Some(done) = keep(res, &mut results)? {
                return Ok(Some(done));
            }
        }
    } else {
        if is_oo(&upper) {
            for split in find_splitting_points(&e, x)? {
                if is_true(&ge(&s2(&lower, &split)?, &zero())?) {
                    let shifted_: Basic = m2(
                        &subs(&e, x, &a2(x, &split)?)?,
                        &heaviside(&s2(&a2(x, &split)?, &lower)?)?,
                    )?;
                    if let Some(res) = meijerint_definite_2(&shifted_, x)? {
                        if let Some(done) = keep(res, &mut results)? {
                            return Ok(Some(done));
                        }
                    }
                }
            }
        }
        e = subs(&e, x, &a2(x, &lower)?)?;
        upper = s2(&upper, &lower)?;
        lower = zero();
        let _ = &lower;
        if !is_oo(&upper) {
            let phi: Basic = exp(&m2(&imag(), &arg_(&upper)?)?)?;
            upper = abs_(&upper)?;
            e = subs(&e, x, &m2(&phi, x)?)?;
            e = m2(&m2(&e, &heaviside(&s2(&upper, x)?)?)?, &phi)?;
        }
        if let Some(res) = meijerint_definite_2(&e, x)? {
            if let Some(done) = keep(res, &mut results)? {
                return Ok(Some(done));
            }
        }
    }
    if has_heads(&e_, &HYPERBOLIC) {
        if let Some(rv) = meijerint_definite(&rewrite_hyperbolics_as_exp(&e_)?, &x_, &a_, &b_)? {
            return Ok(Some(collected(rv)?));
        }
    }
    if !results.is_empty() {
        return Ok(Some(ordered_first(results)));
    }
    Ok(None)
}

/// Sensible rewritings of the integrand (`_guess_expansion`).
fn guess_expansion(f: &Basic) -> PolyResult<Vec<Basic>> {
    let mut res: Vec<Basic> = vec![f.clone()];
    let mut push = |e: Basic| {
        if !res.contains(&e) {
            res.push(e);
        }
    };
    push(expand_mul(f)?);
    push(expand(f)?);
    let trig: [Function; 12] = [
        Function::Sin,
        Function::Cos,
        Function::Tan,
        Function::Cot,
        Function::Sec,
        Function::Csc,
        Function::Sinh,
        Function::Cosh,
        Function::Tanh,
        Function::Coth,
        Function::Sech,
        Function::Csch,
    ];
    if has_heads(f, &trig) {
        push(expand_mul(&expand_trig(f)?)?);
    }
    if has_heads(f, &[Function::Sin, Function::Cos]) {
        push(crate::simplify::fu::sincos_to_sum(f)?);
    }
    Ok(res)
}

/// The integral of `f` over `(0, oo)`, the forms of [`guess_expansion`]
/// in turn (`_meijerint_definite_2`).
fn meijerint_definite_2(f: &Basic, x: &Basic) -> PolyResult<Option<ConditionalExpr>> {
    let d: Basic = dummy("x", "meijerint-definite2", f, true);
    let f: Basic = subs(f, x, &d)?;
    let x: &Basic = &d;
    if f == zero() {
        return Ok(Some((zero(), true_())));
    }
    for g in guess_expansion(&f)? {
        if let Some(res) = meijerint_definite_3(&g, x)? {
            return Ok(Some(res));
        }
    }
    Ok(None)
}

/// The integral of `f` over `(0, oo)`, by linearity when the whole fails
/// (`_meijerint_definite_3`).
fn meijerint_definite_3(f: &Basic, x: &Basic) -> PolyResult<Option<ConditionalExpr>> {
    if let Some(res) = meijerint_definite_4(f, x, false)? {
        if !is_false(&res.1) {
            return Ok(Some(res));
        }
    }
    if f.is_add() {
        let mut found: Vec<Option<ConditionalExpr>> = Vec::new();
        for g in f.args() {
            found.push(meijerint_definite_4(g, x, false)?);
        }
        let mut values: Vec<Basic> = Vec::new();
        let mut conds: Vec<Basic> = Vec::new();
        for r in found {
            let Some((v, c)) = r else {
                return Ok(None);
            };
            values.push(v);
            conds.push(c);
        }
        let c: Basic = and_(&conds);
        if !is_false(&c) {
            return Ok(Some((add_n(&values)?, c)));
        }
    }
    Ok(None)
}

/// The integral of `f` over `(0, oo)` by the theorems: `f` as one
/// G-function (not with `only_double`) or as a product of two
/// (`_meijerint_definite_4`).
fn meijerint_definite_4(f: &Basic, x: &Basic, only_double: bool) -> PolyResult<Option<ConditionalExpr>> {
    if !only_double {
        if let Some((fac, po, g, mut cond)) = rewrite1(f, x, false)? {
            let mut res: Basic = zero();
            for (c, s, h) in g {
                if c == zero() {
                    continue;
                }
                let (c, h) = rewrite_saxena_1(&m2(&fac, &c)?, &m2(&po, &p2(x, &s)?)?, &h, x)?;
                res = a2(&res, &m2(&c, &int0oo_1(&h, x)?)?)?;
                cond = and_(&[cond, check_antecedents_1(&h, x, false)?]);
                if is_false(&cond) {
                    break;
                }
            }
            let cond: Basic = my_unpolarify_cond(&cond)?;
            if !is_false(&cond) {
                return Ok(Some((my_unpolarify(&hyperexpand(&res)?)?, cond)));
            }
        }
    }
    if let Some(gs2) = rewrite2(f, x)? {
        for full_pb in [false, true] {
            let (fac, po, g1, g2, mut cond) = gs2.clone();
            let mut res: Basic = zero();
            let mut done: bool = false;
            for (c1, s1, f1_) in &g1 {
                for (c2, s2_, f2_) in &g2 {
                    let Some((c, f1n, f2n)) = rewrite_saxena(
                        &m2(&m2(&fac, c1)?, c2)?,
                        &m2(&po, &p2(x, &a2(s1, s2_)?)?)?,
                        f1_,
                        f2_,
                        x,
                        full_pb,
                    )?
                    else {
                        // non-rational exponents
                        return Ok(None);
                    };
                    cond = and_(&[cond, check_antecedents(&f1n, &f2n, x)?]);
                    if is_false(&cond) {
                        done = true;
                        break;
                    }
                    res = a2(&res, &m2(&c, &int0oo(&f1n, &f2n, x)?)?)?;
                }
                if done {
                    break;
                }
            }
            let cond: Basic = my_unpolarify_cond(&cond)?;
            if !is_false(&cond) {
                if only_double {
                    return Ok(Some((res, cond)));
                }
                return Ok(Some((my_unpolarify(&hyperexpand(&res)?)?, cond)));
            }
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------
// the inverse Laplace transform

/// `(C, G)` with `C*G` the function `H` defined as `g` with the integrand
/// of its Mellin–Barnes integral divided by `Gamma(a*s)`, for a rational
/// `a` (`_inflate_fox_h`).
fn inflate_fox_h(g: &Basic, a: &Basic) -> PolyResult<Option<(Basic, Basic)>> {
    let Some(aq) = rat_of(a).filter(|_| *a != zero()) else {
        return Ok(None);
    };
    let to = |i: Int| i64::try_from(i).map_err(|_| type_error("exponent too large"));
    let (p, q) = (to(crate::numbers::numer(&aq))?, to(crate::numbers::denom(&aq))?);
    if p < 0 {
        return inflate_fox_h(&flip_g(g)?, &ng(a)?);
    }
    // the substitution s -> q*s, then Gauss' theorem on the factor Gamma(p*s)
    let (d, g) = inflate_g(g, q)?;
    let mut argument: Basic = g_argument(&g);
    let d: Basic = d2(
        &d,
        &m2(
            &p2(&m2(&int(2), &pi())?, &rational(1 - p, 2))?,
            &p2(&int(p), &rational(-1, 2))?,
        )?,
    )?;
    argument = d2(&argument, &p2(&int(p), &int(p))?)?;
    let mut bother: Vec<Basic> = g_bother(&g);
    for n in 0..p {
        bother.push(rational(n + 1, p));
    }
    Ok(Some((d, g_(&g_an(&g), &g_aother(&g), &g_bm(&g), &bother, &argument)?)))
}

/// `fac*po*g` as `C*G`: the power `po = x**s` absorbed into `g`
/// (`_rewrite_inversion`).
fn rewrite_inversion(fac: &Basic, po: &Basic, g: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let (_, s) = get_coeff_exp(po, x)?;
    let (a, b) = get_coeff_exp(&g_argument(g), x)?;
    let sb: Basic = d2(&s, &b)?;
    let tr = |c: &Basic| -> R { a2(c, &sb) };
    Ok((
        powdenest_polar(&d2(fac, &p2(&a, &sb)?)?)?,
        g_(
            &shifted(&g_an(g), &tr)?,
            &shifted(&g_aother(g), &tr)?,
            &shifted(&g_bm(g), &tr)?,
            &shifted(&g_bother(g), &tr)?,
            &g_argument(g),
        )?,
    ))
}

/// Half of [`statement`], on the ray of `exp(+-i*pi*re(c)/2)`
/// (`statement_half`).
fn statement_half(a: &Basic, b: &Basic, c: &Basic, z_: &Basic, x: &Basic, plus: bool) -> R {
    let (coeff, exponent) = get_coeff_exp(z_, x)?;
    let a: Basic = m2(a, &exponent)?;
    let b: Basic = m2(b, &p2(&coeff, c)?)?;
    let c: Basic = m2(c, &exponent)?;
    let unit: Basic = if plus { imag() } else { ng(&imag())? };
    let w: Basic = m2(&b, &exp(&d2(&m2(&m2(&unit, &re(&c)?)?, &pi())?, &int(2))?)?)?;
    Ok(or_(&[
        and_(&[
            or_(&[eq(&b, &zero())?, le(&re(&c)?, &zero())?]),
            le(&re(&a)?, &int(-1))?,
        ]),
        and_(&[
            ne(&b, &zero())?,
            eq(&im(&c)?, &zero())?,
            gt(&re(&c)?, &zero())?,
            lt(&re(&w)?, &zero())?,
        ]),
        and_(&[
            ne(&b, &zero())?,
            eq(&im(&c)?, &zero())?,
            gt(&re(&c)?, &zero())?,
            le(&re(&w)?, &zero())?,
            le(&re(&a)?, &int(-1))?,
        ]),
    ]))
}

/// The convergence of the inversion integral of `z**a * exp(b*z**c)` on
/// the two rays (`statement`).
fn statement(a: &Basic, b: &Basic, c: &Basic, z_: &Basic, x: &Basic) -> R {
    Ok(and_(&[
        statement_half(a, b, c, z_, x, true)?,
        statement_half(a, b, c, z_, x, false)?,
    ]))
}

/// The condition under which the inverse Laplace transform of `g` exists
/// and [`int_inversion`] computes it ([L], sections 5.7–5.10)
/// (`_check_antecedents_inversion`).
fn check_antecedents_inversion(g: &Basic, x: &Basic) -> R {
    let zarg: Basic = g_argument(g);
    let (_, e) = get_coeff_exp(&zarg, x)?;
    if is_true(&lt(&e, &zero())?) {
        // the argument should grow as |x| -> oo
        return check_antecedents_inversion(&flip_g(g)?, x);
    }
    let (mi, ni, pi_, qi) = (g_bm(g).len(), g_an(g).len(), g_ap(g).len(), g_bq(g).len());
    let (m, n, p, q) = (int(mi as i64), int(ni as i64), int(pi_ as i64), int(qi as i64));
    let tau: Basic = s2(&a2(&m, &n)?, &p)?;
    let nu: Basic = s2(&s2(&q, &m)?, &n)?;
    let rho: Basic = d2(&s2(&tau, &nu)?, &int(2))?;
    let sigma: Basic = s2(&q, &p)?;
    let epsilon: Basic = if qi as i64 - pi_ as i64 == 1 {
        half()
    } else if qi > pi_ + 1 {
        one()
    } else {
        Basic::atom(Head::NaN)
    };
    let bq: Vec<Basic> = g_bq(g);
    let ap: Vec<Basic> = g_ap(g);
    let bq_sum: Basic = if bq.is_empty() { zero() } else { add_n(&bq)? };
    let ap_sum: Basic = if ap.is_empty() { zero() } else { add_n(&ap)? };
    let delta: Basic = meijerg_delta(g)?;

    // whether the computation is valid
    if !(is_true(&ge(&delta, &d2(&e, &int(2))?)?) || (pi_ >= 1 && pi_ >= qi)) {
        return Ok(false_());
    }
    // condition A: a valid G-function
    for a in g_an(g) {
        for b in g_bm(g) {
            if is(&s2(&a, &b)?, Predicate::integer) == Some(true) && is_true(&gt(&a, &b)?) {
                return Ok(false_());
            }
        }
    }
    let big_e = |zz: &Basic| -> R {
        let mut v: Vec<Basic> = Vec::new();
        for a in g_an(g) {
            v.push(statement(&s2(&a, &one())?, &zero(), &zero(), zz, x)?);
        }
        Ok(and_(&v))
    };
    // p >= q: the asymptotic Slater expansion
    if pi_ >= qi {
        return big_e(&zarg);
    }
    let theta: Basic = d2(
        &s2(&a2(&rational(1 - (qi as i64 - pi_ as i64), 2), &bq_sum)?, &ap_sum)?,
        &sigma,
    )?;
    let msigma: Basic = ng(&sigma)?;
    let isigma: Basic = d2(&one(), &sigma)?;
    let big_h = |zz: &Basic| statement(&theta, &msigma, &isigma, zz, x);
    let hp = |zz: &Basic| statement_half(&theta, &msigma, &isigma, zz, x, true);
    let hm = |zz: &Basic| statement_half(&theta, &msigma, &isigma, zz, x, false);
    let ipi: Basic = m2(&imag(), &pi())?;
    let mipi: Basic = m2(&ng(&imag())?, &pi())?;
    let pi_2: Basic = d2(&pi(), &int(2))?;
    let ge_pi_2 = |k: &Basic| -> R { ge(&s2(&m2(k, &pi())?, &delta)?, &pi_2) };
    let b = |v: bool| boolean(v);
    let mut conds: Vec<Basic> = Vec::new();
    // theorem 1
    conds.push(and_(&[
        b(1 <= ni),
        b(1 <= mi),
        ge(&s2(&m2(&rho, &pi())?, &delta)?, &pi_2)?,
        gt(&delta, &zero())?,
        big_e(&m2(&zarg, &exp(&m2(&ipi, &a2(&nu, &one())?)?)?)?)?,
    ]));
    // theorem 2, statements (2) and (3)
    let q_m: Basic = s2(&q, &m)?;
    conds.push(and_(&[
        b(pi_ < mi),
        b(mi < qi),
        gt(&delta, &zero())?,
        lt(&delta, &pi_2)?,
        b(ni == 0),
        ge_pi_2(&a2(&s2(&m, &p)?, &one())?)?,
        hp(&m2(&zarg, &exp(&m2(&ipi, &q_m)?)?)?)?,
        hm(&m2(&zarg, &exp(&m2(&mipi, &q_m)?)?)?)?,
    ]));
    // theorem 2, statement (5)
    conds.push(and_(&[
        b(mi == qi),
        b(ni == 0),
        gt(&delta, &zero())?,
        ge_pi_2(&a2(&sigma, &epsilon)?)?,
        big_h(&zarg)?,
    ]));
    // theorem 3, statements (6) and (7)
    conds.push(and_(&[
        or_(&[
            and_(&[
                b(pi_ as i64 <= qi as i64 - 2),
                le(&one(), &tau)?,
                le(&tau, &d2(&sigma, &int(2))?)?,
            ]),
            and_(&[b(pi_ < mi + ni), le(&a2(&m, &n)?, &d2(&a2(&p, &q)?, &int(2))?)?]),
        ]),
        gt(&delta, &zero())?,
        lt(&delta, &pi_2)?,
        ge_pi_2(&a2(&tau, &one())?)?,
        hp(&m2(&zarg, &exp(&m2(&ipi, &nu)?)?)?)?,
        hm(&m2(&zarg, &exp(&m2(&mipi, &nu)?)?)?)?,
    ]));
    // theorem 4, statements (10) and (11)
    conds.push(and_(&[
        b(1 <= mi),
        gt(&rho, &zero())?,
        gt(&delta, &zero())?,
        lt(&a2(&delta, &m2(&rho, &pi())?)?, &pi_2)?,
        ge_pi_2(&a2(&tau, &epsilon)?)?,
        hp(&m2(&zarg, &exp(&m2(&ipi, &nu)?)?)?)?,
        hm(&m2(&zarg, &exp(&m2(&mipi, &nu)?)?)?)?,
    ]));
    // the trivial case
    conds.push(b(mi == 0));
    Ok(or_(&conds))
}

/// The inverse Laplace transform of `g` at `t`, the conditions assumed;
/// `None` when the exponent of `x` is not rational (`_int_inversion`).
fn int_inversion(g: &Basic, x: &Basic, t: &Basic) -> PolyResult<Option<Basic>> {
    let (b, a) = get_coeff_exp(&g_argument(g), x)?;
    let [an, aother, bm, bother] = lists(g);
    let Some((c, h)) = inflate_fox_h(&g_(&an, &aother, &bm, &bother, &d2(&b, &p2(t, &a)?)?)?, &ng(&a)?)? else {
        return Ok(None);
    };
    Ok(Some(m2(&d2(&c, t)?, &h)?))
}

/// `(g, shift)` with `f = g*exp(shift*x)` (`_split_shift`).
fn split_shift(f: &Basic, x: &Basic) -> PolyResult<(Basic, Basic)> {
    let mut args: Vec<Basic> = if f.is_mul() {
        f.args().to_vec()
    } else if is_fn(f, &[Function::Exp]) {
        vec![f.clone()]
    } else {
        return Ok((f.clone(), zero()));
    };
    let mut newargs: Vec<Basic> = Vec::new();
    let mut exponentials: Vec<Basic> = Vec::new();
    let coeff_exp_or_zero = |e: &Basic| -> PolyResult<(Basic, Basic)> {
        match get_coeff_exp(e, x) {
            Ok(v) => Ok(v),
            Err(err) if is_coeff_exp_error(&err) => Ok((zero(), zero())),
            Err(err) => Err(err),
        }
    };
    while let Some(arg) = args.pop() {
        if is_fn(&arg, &[Function::Exp]) {
            let arg2: Basic = expand(&arg)?;
            if arg2.is_mul() {
                args.extend(arg2.args().iter().cloned());
                continue;
            }
            let (a, b) = coeff_exp_or_zero(&arg.args()[0])?;
            if b == one() {
                exponentials.push(a);
            } else {
                newargs.push(arg);
            }
        } else if arg.is_pow() {
            let arg2: Basic = expand(&arg)?;
            if arg2.is_mul() {
                args.extend(arg2.args().iter().cloned());
                continue;
            }
            if free_of(&arg.args()[0], x) {
                let (a, b) = coeff_exp_or_zero(&arg.args()[1])?;
                if b == one() {
                    // the factor is the shift
                    exponentials.push(m2(&a, &log(&arg.args()[0])?)?);
                    continue;
                }
            }
            newargs.push(arg);
        } else {
            newargs.push(arg);
        }
    }
    Ok((
        if newargs.is_empty() { one() } else { mul_n(&newargs)? },
        if exponentials.is_empty() {
            zero()
        } else {
            add_n(&exponentials)?
        },
    ))
}

/// The inverse Laplace transform of `f(x)` at `t` by rewriting `f` as
/// G-functions, with the condition under which it holds; `None` when `f`
/// cannot be rewritten or the condition is false (`meijerint_inversion`).
pub fn meijerint_inversion(f: &Basic, x: &Basic, t: &Basic) -> PolyResult<Option<ConditionalExpr>> {
    // a polar t: no sqrt(t**2) = abs(t)
    let tp: Basic = Dummy::new("t", &[(Predicate::polar, true)]).0;
    let e: Basic = subs(f, t, &tp)?;
    if !is_analytic(&e, x) {
        return Ok(None);
    }
    // the exponentials are shifts, filtered out and applied to the result
    let (e, shift) = split_shift(&e, x)?;
    if free_of(&e, x) {
        let cond: Basic = eq(&im(&shift)?, &zero())?;
        if is_false(&cond) {
            // a nonreal shift: not a Laplace transform
            return Ok(None);
        }
        let res: Basic = m2(
            &e,
            &crate::functions::special::delta_functions::dirac_delta(&a2(&tp, &shift)?, None)?,
        )?;
        return Ok(Some((subs(&res, &tp, t)?, subs(&cond, &tp, t)?)));
    }
    let Some((fac, po, g, mut cond)) = rewrite1(&e, x, true)? else {
        return Ok(None);
    };
    let mut value: Basic = zero();
    for (c, s, h) in g {
        let (c, h) = rewrite_inversion(&m2(&fac, &c)?, &m2(&po, &p2(x, &s)?)?, &h, x)?;
        let Some(term) = int_inversion(&h, x, &tp)? else {
            return Ok(None);
        };
        value = a2(&value, &m2(&c, &term)?)?;
        cond = and_(&[cond, check_antecedents_inversion(&h, x)?]);
        if is_false(&cond) {
            break;
        }
    }
    let cond: Basic = my_unpolarify_cond(&cond)?;
    if is_false(&cond) {
        return Ok(None);
    }
    let mut value: Basic = my_unpolarify(&hyperexpand(&value)?)?;
    if !has_heads(&value, &[Function::Heaviside]) {
        value = m2(&value, &heaviside(&tp)?)?;
    }
    let moved: Basic = a2(&tp, &shift)?;
    let value: Basic = subs(&value, &tp, &moved)?;
    let cond: Basic = subs(&cond, &tp, &moved)?;
    Ok(Some((subs(&value, &tp, t)?, subs(&cond, &tp, t)?)))
}
