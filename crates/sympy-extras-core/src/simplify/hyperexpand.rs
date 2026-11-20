//! Port of `sympy_extras/simplify/hyperexpand.py` (SymPy's
//! `hyperexpand`): a hypergeometric function or a Meijer G-function
//! reduced, by shifting its parameters by integers, to one of a table of
//! known formulas, each a basis `B` of functions with `z*d/dz B = M*B`
//! and the closed form `C*B`; the shifts are operators on the basis.
//! The G-functions not in the table are expanded by Slater's theorem.

#![allow(clippy::type_complexity, clippy::too_many_arguments, clippy::mutable_key_type)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;

type R = PolyResult<Basic>;
type Repl = Vec<(Basic, Basic)>;

pub(crate) fn int(n: i64) -> Basic {
    Basic::int(n)
}
pub(crate) fn half() -> Basic {
    Basic::from_rational(crate::numbers::rat(1, 2))
}
pub(crate) fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}
pub(crate) fn a2(a: &Basic, b: &Basic) -> R {
    crate::core::add::add(&[a.clone(), b.clone()])
}
pub(crate) fn s2(a: &Basic, b: &Basic) -> R {
    crate::core::add::sub(a, b)
}
pub(crate) fn d2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::div(a, b)
}
pub(crate) fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
pub(crate) fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
/// `mul(*v)`: the n-ary product (Python's `mul(*...)`).
pub(crate) fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
pub(crate) fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn func(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}
fn apply(f: Function, args: &[Basic]) -> R {
    crate::core::function::rebuild_evaluated(&Basic::new(Head::Function(f), args.to_vec()), args.to_vec())
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn diff(e: &Basic, x: &Basic) -> R {
    crate::core::function::derivative(e, x)
}
fn subs(e: &Basic, old: &Basic, new: &Basic) -> R {
    crate::core::subs::subs(e, old, new)
}
fn replaced(e: &Basic, repl: &[(Basic, Basic)]) -> R {
    if repl.is_empty() {
        return Ok(e.clone());
    }
    let map: HashMap<Basic, Basic> = repl.iter().cloned().collect();
    crate::core::function::xreplace_evaluated(e, &map)
}
fn sort_key(e: &Basic) -> crate::core::sorting::SortKey {
    crate::core::sorting::sort_key(e)
}
fn yes(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}
fn no(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(false)
}
fn dummy(name: &str) -> Basic {
    crate::core::symbol::Dummy::new(name, &[]).0
}
fn as_int(e: &Basic) -> PolyResult<i64> {
    match e.as_rational() {
        Some(r) if r.is_int() => {
            i64::try_from(&crate::numbers::numer(r)).map_err(|_| value_error("a large integer".to_string()))
        }
        _ => Err(value_error(format!("{} is not an integer", e))),
    }
}
fn is_integer_node(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}
fn is_zero_number(e: &Basic) -> bool {
    e.is_number_node() && *e == int(0)
}

/// The truth of a comparison, as SymPy's `bool(relational)`: a `Type`
/// error when it is not decided (`_truth`).
fn truth(b: &Basic) -> PolyResult<bool> {
    if *b == crate::core::logic::true_() {
        return Ok(true);
    }
    if *b == crate::core::logic::false_() {
        return Ok(false);
    }
    Err(PolyError::Type(format!("cannot determine truth value of {}", b)))
}

fn lt(a: &Basic, b: &Basic) -> PolyResult<bool> {
    truth(&crate::core::relational::lt(a, b)?)
}
fn gt0(e: &Basic) -> PolyResult<bool> {
    truth(&crate::core::relational::gt(e, &int(0))?)
}
fn le0(e: &Basic) -> PolyResult<bool> {
    truth(&crate::core::relational::le(e, &int(0))?)
}

/// `sorted(values, key=_Ordered)`: a binary insertion sort by `<`
/// (CPython's for short lists), stable.
fn sort_ordered(values: &[Basic], key: &dyn Fn(&Basic) -> R) -> PolyResult<Vec<Basic>> {
    let keys: Vec<Basic> = values.iter().map(key).collect::<PolyResult<_>>()?;
    let mut order: Vec<usize> = Vec::new();
    for i in 0..values.len() {
        let (mut lo, mut hi) = (0usize, order.len());
        while lo < hi {
            let mid: usize = (lo + hi) / 2;
            if lt(&keys[i], &keys[order[mid]])? {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        order.insert(lo, i);
    }
    Ok(order.into_iter().map(|i| values[i].clone()).collect())
}

fn ordered(values: &[Basic]) -> PolyResult<Vec<Basic>> {
    sort_ordered(values, &|x| Ok(x.clone()))
}

fn min_of(values: &[Basic]) -> PolyResult<Basic> {
    let mut m: Basic = values[0].clone();
    for v in &values[1..] {
        if lt(v, &m)? {
            m = v.clone();
        }
    }
    Ok(m)
}

fn max_of(values: &[Basic]) -> PolyResult<Basic> {
    let mut m: Basic = values[0].clone();
    for v in &values[1..] {
        if lt(&m, v)? {
            m = v.clone();
        }
    }
    Ok(m)
}

/// `x mod 1` for numbers, the constant term `mod 1` otherwise (`_mod1`).
fn mod1(x: &Basic) -> R {
    if x.is_number_node() {
        return crate::core::modulo::modulo(x, &int(1));
    }
    let (c, rest) = crate::core::expr::as_coeff_add(x);
    a2(&crate::core::modulo::modulo(&c, &int(1))?, &rest)
}

type Buckets = Vec<(Basic, Vec<Basic>)>;

fn sift(params: &[Basic], key: &dyn Fn(&Basic) -> R) -> PolyResult<Buckets> {
    let mut buckets: Buckets = Vec::new();
    for p in params {
        let k: Basic = key(p)?;
        match buckets.iter_mut().find(|(b, _)| *b == k) {
            Some(entry) => entry.1.push(p.clone()),
            None => buckets.push((k, vec![p.clone()])),
        }
    }
    Ok(buckets)
}

fn bucket<'a>(b: &'a Buckets, k: &Basic) -> Option<&'a Vec<Basic>> {
    b.iter().find(|(key, _)| key == k).map(|(_, v)| v)
}

fn keys_union(a: &Buckets, b: &Buckets) -> Vec<Basic> {
    let mut keys: Vec<Basic> = Vec::new();
    for (k, _) in a.iter().chain(b.iter()) {
        if !keys.contains(k) {
            keys.push(k.clone());
        }
    }
    keys.sort_by_cached_key(sort_key);
    keys
}

/// `simp` of `make_simp`: the cancellation in `z` of a fraction, the
/// coefficients which are not numbers or symbols as dummies when the
/// polynomials do not take them (`make_simp`).
pub fn make_simp(z: &Basic) -> impl Fn(&Basic) -> R + '_ {
    move |e: &Basic| -> R {
        let cancel_in_z = |numer: &Basic, denom: &Basic| -> R {
            let gens = std::slice::from_ref(z);
            let (c, n, d) = crate::polys::polytools::Poly::new(numer, gens)?
                .cancel(&crate::polys::polytools::Poly::new(denom, gens)?)?;
            d2(&m2(&c, &n.as_expr()?)?, &d.as_expr()?)
        };
        let (numer, denom) = crate::core::expr::as_numer_denom(e)?;
        let numer: Basic = expand(&numer)?;
        match cancel_in_z(&numer, &denom) {
            Ok(r) => return Ok(r),
            Err(PolyError::Polys(crate::error::PolysErrorKind::CoercionFailed, _)) => {}
            Err(err) => return Err(err),
        }
        let rep: Repl = coefficient_generators(&[numer.clone(), expand(&denom)?], z)?;
        let back: Repl = rep.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
        replaced(&cancel_in_z(&replaced(&numer, &rep)?, &replaced(&denom, &rep)?)?, &back)
    }
}

fn coefficient_generators(exprs: &[Basic], z: &Basic) -> PolyResult<Repl> {
    let mut found: Vec<Basic> = Vec::new();
    for e in exprs {
        let terms: Vec<Basic> = if e.is_add() { e.args().to_vec() } else { vec![e.clone()] };
        for t in terms {
            let factors: Vec<Basic> = if t.is_mul() { t.args().to_vec() } else { vec![t.clone()] };
            for f in factors {
                if f.has(z) || f.is_number_node() || f.is_symbol() {
                    continue;
                }
                let (b, ex) = crate::core::expr::as_base_exp(&f)?;
                let g: Basic = if is_integer_node(&ex) && !(b.is_number_node() || b.is_symbol()) {
                    b
                } else {
                    f.clone()
                };
                if g.is_number_node() || g.is_symbol() || g.has(z) {
                    continue;
                }
                if !found.contains(&g) {
                    found.push(g);
                }
            }
        }
    }
    found.sort_by_cached_key(sort_key);
    Ok(found.into_iter().map(|g| (g, dummy("c"))).collect())
}

/// `hyper(ap, bq, z)` as its parameters (`HyperFunction`).
#[derive(Debug, Clone, PartialEq)]
pub struct HyperFunction {
    pub ap: Vec<Basic>,
    pub bq: Vec<Basic>,
}

type Invariants = (i64, Vec<(Basic, usize)>, Vec<(Basic, usize)>);

impl HyperFunction {
    pub fn new(ap: &[Basic], bq: &[Basic]) -> PolyResult<HyperFunction> {
        Ok(HyperFunction {
            ap: ap.iter().map(expand).collect::<PolyResult<_>>()?,
            bq: bq.iter().map(expand).collect::<PolyResult<_>>()?,
        })
    }

    fn sizes(&self) -> (usize, usize) {
        (self.ap.len(), self.bq.len())
    }

    fn gamma(&self) -> i64 {
        self.ap
            .iter()
            .filter(|x| yes(x, Predicate::integer) && yes(x, Predicate::negative))
            .count() as i64
    }

    pub fn call(&self, arg: &Basic) -> R {
        crate::functions::special::hyper::hyper(&self.ap, &self.bq, arg)
    }

    fn has(&self, x: &Basic) -> bool {
        self.ap.iter().chain(self.bq.iter()).any(|p| p.has(x))
    }

    fn xreplace(&self, repl: &[(Basic, Basic)]) -> PolyResult<HyperFunction> {
        let ap: Vec<Basic> = self.ap.iter().map(|a| replaced(a, repl)).collect::<PolyResult<_>>()?;
        let bq: Vec<Basic> = self.bq.iter().map(|b| replaced(b, repl)).collect::<PolyResult<_>>()?;
        HyperFunction::new(&ap, &bq)
    }

    fn build_invariants(&self) -> PolyResult<Invariants> {
        let tr = |b: Buckets| -> Vec<(Basic, usize)> {
            let mut items: Buckets = b;
            if !items.iter().any(|(k, _)| func(k) == Some(Function::Mod)) {
                items.sort_by_cached_key(|(k, _)| sort_key(k));
            }
            items
                .into_iter()
                .filter(|(_, v)| !v.is_empty())
                .map(|(k, v)| (k, v.len()))
                .collect()
        };
        Ok((self.gamma(), tr(sift(&self.ap, &mod1)?), tr(sift(&self.bq, &mod1)?)))
    }

    /// The distance to `func` by integer shifts, `-1` when unreachable
    /// (`difficulty`).
    fn difficulty(&self, func_: &HyperFunction) -> R {
        if self.gamma() != func_.gamma() {
            return Ok(int(-1));
        }
        let (oa, ob, a, b) = (
            sift(&self.ap, &mod1)?,
            sift(&self.bq, &mod1)?,
            sift(&func_.ap, &mod1)?,
            sift(&func_.bq, &mod1)?,
        );
        let mut total: Basic = int(0);
        for (bk, obk) in [(&a, &oa), (&b, &ob)] {
            for m in keys_union(bk, obk) {
                let (Some(x), Some(y)) = (bucket(bk, &m), bucket(obk, &m)) else {
                    return Ok(int(-1));
                };
                if x.len() != y.len() {
                    return Ok(int(-1));
                }
                for (i, j) in ordered(x)?.iter().zip(ordered(y)?.iter()) {
                    total = a2(&total, &crate::functions::elementary::complexes::abs_(&s2(i, j)?)?)?;
                }
            }
        }
        Ok(total)
    }

    fn is_suitable_origin(&self) -> PolyResult<bool> {
        for a in &self.ap {
            for b in &self.bq {
                let d: Basic = s2(a, b)?;
                if yes(&d, Predicate::integer) && no(&d, Predicate::negative) {
                    return Ok(false);
                }
            }
        }
        if self.ap.iter().any(|a| *a == int(0)) {
            return Ok(false);
        }
        Ok(!self
            .bq
            .iter()
            .any(|b| yes(b, Predicate::integer) && yes(b, Predicate::nonpositive)))
    }
}

/// `meijerg(an, ap, bm, bq, z)` as its parameters (`GFunction`).
#[derive(Debug, Clone, PartialEq)]
pub struct GFunction {
    pub an: Vec<Basic>,
    pub ap: Vec<Basic>,
    pub bm: Vec<Basic>,
    pub bq: Vec<Basic>,
}

impl GFunction {
    pub fn new(an: &[Basic], ap: &[Basic], bm: &[Basic], bq: &[Basic]) -> PolyResult<GFunction> {
        let e = |v: &[Basic]| v.iter().map(expand).collect::<PolyResult<Vec<Basic>>>();
        Ok(GFunction {
            an: e(an)?,
            ap: e(ap)?,
            bm: e(bm)?,
            bq: e(bq)?,
        })
    }

    pub fn call(&self, z: &Basic) -> R {
        crate::functions::special::hyper::meijerg(&self.an, &self.ap, &self.bm, &self.bq, z)
    }

    fn signature(&self) -> (usize, usize, usize, usize) {
        (self.an.len(), self.ap.len(), self.bm.len(), self.bq.len())
    }

    fn compute_buckets(&self) -> PolyResult<[Buckets; 4]> {
        let lists: [&Vec<Basic>; 4] = [&self.an, &self.ap, &self.bm, &self.bq];
        let flips: [bool; 4] = [true, false, false, true];
        let mut out: Vec<Buckets> = Vec::new();
        for (lis, flip) in lists.iter().zip(flips) {
            let mut dic: Buckets = Vec::new();
            for x in lis.iter() {
                let k: Basic = mod1(x)?;
                match dic.iter_mut().find(|(b, _)| *b == k) {
                    Some(entry) => entry.1.push(x.clone()),
                    None => dic.push((k, vec![x.clone()])),
                }
            }
            for entry in dic.iter_mut() {
                let x0: Basic = entry.1[0].clone();
                let mut sorted: Vec<Basic> = sort_ordered(&entry.1, &|x| s2(x, &x0))?;
                if flip {
                    // sorted(..., reverse=True) keeps the order of the equal items
                    let keys: Vec<Basic> = entry.1.iter().map(|x| s2(x, &x0)).collect::<PolyResult<_>>()?;
                    let mut indexed: Vec<(Basic, Basic)> = entry.1.iter().cloned().zip(keys).collect();
                    indexed.reverse();
                    let reversed_items: Vec<Basic> = indexed.iter().map(|(x, _)| x.clone()).collect();
                    sorted = sort_ordered(&reversed_items, &|x| s2(x, &x0))?;
                    sorted.reverse();
                }
                entry.1 = sorted;
            }
            out.push(dic);
        }
        let mut it = out.into_iter();
        Ok([
            it.next().unwrap_or_default(),
            it.next().unwrap_or_default(),
            it.next().unwrap_or_default(),
            it.next().unwrap_or_default(),
        ])
    }
}

thread_local! {
    static X: Basic = dummy("x");
}

/// The module's dummy `_x`, the variable of the operators.
fn x_() -> Basic {
    X.with(|x| x.clone())
}

fn solve_linear(e: &Basic, n: &Basic) -> R {
    let e: Basic = expand(e)?;
    let c1: Basic = expand(&diff(&e, n)?)?;
    if c1 == int(0) || c1.has(n) {
        return Err(value_error(format!("not one solution of {} = 0 in {}", e, n)));
    }
    let c0: Basic = subs(&e, n, &int(0))?;
    d2(&neg(&c0)?, &c1)
}

/// A closed form of a hypergeometric function with its basis, its
/// coefficients and the matrix of `z*d/dz` (`Formula`).
#[derive(Debug, Clone)]
pub struct Formula {
    pub func: HyperFunction,
    pub z: Basic,
    pub symbols: Vec<Basic>,
    pub b: ExprMatrix,
    pub c: ExprMatrix,
    pub m: ExprMatrix,
}

fn closed_form(c: &ExprMatrix, b: &ExprMatrix) -> R {
    let mut total: Basic = int(0);
    for (ci, bi) in c.entries.iter().zip(b.entries.iter()) {
        total = a2(&total, &m2(ci, bi)?)?;
    }
    Ok(total)
}

fn has_infinities(m: &ExprMatrix) -> bool {
    m.entries.iter().any(|e| {
        e.preorder().iter().any(|n| {
            matches!(
                n.head(),
                Head::NaN | Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity
            )
        })
    })
}

impl Formula {
    fn with_basis(
        func_: HyperFunction,
        z: &Basic,
        symbols: &[Basic],
        b: ExprMatrix,
        c: ExprMatrix,
        m: ExprMatrix,
    ) -> Formula {
        let symbols: Vec<Basic> = symbols.iter().filter(|x| func_.has(x)).cloned().collect();
        Formula {
            func: func_,
            z: z.clone(),
            symbols,
            b,
            c,
            m,
        }
    }

    pub fn from_closed_form(func_: HyperFunction, z: &Basic, res: &Basic, symbols: &[Basic]) -> PolyResult<Formula> {
        let x: Basic = x_();
        let afactors: Vec<Basic> = func_.ap.iter().map(|a| a2(&x, a)).collect::<PolyResult<_>>()?;
        let bfactors: Vec<Basic> = func_
            .bq
            .iter()
            .map(|b| s2(&a2(&x, b)?, &int(1)))
            .collect::<PolyResult<_>>()?;
        let e: Basic = s2(&m2(&x, &mul_all(&bfactors)?)?, &m2(z, &mul_all(&afactors)?)?)?;
        let poly = crate::polys::polytools::Poly::new(&e, std::slice::from_ref(&x))?;
        let n: usize = (poly.degree(0)? - 1) as usize;
        let mut b: Vec<Basic> = vec![res.clone()];
        for _ in 0..n {
            let last: Basic = b[b.len() - 1].clone();
            b.push(m2(z, &diff(&last, z)?)?);
        }
        let bm: ExprMatrix = ExprMatrix::column(b);
        let mut first: Vec<Basic> = vec![int(1)];
        first.extend(std::iter::repeat(int(0)).take(n));
        let cm: ExprMatrix = ExprMatrix::from_rows(&[first])?;
        let m: ExprMatrix = ExprMatrix::eye(n).col_insert(0, &ExprMatrix::zeros(n, 1))?;
        let coeffs: Vec<Basic> = poly.all_coeffs()?;
        let last: Vec<Basic> = coeffs[1..].iter().rev().cloned().collect();
        let row: ExprMatrix = ExprMatrix::from_rows(&[last])?.neg()?.div(&coeffs[0])?;
        let mm: ExprMatrix = m.row_insert(n, &row)?;
        Ok(Formula::with_basis(func_, z, symbols, bm, cm, mm))
    }

    pub fn closed_form(&self) -> R {
        closed_form(&self.c, &self.b)
    }

    /// The values of the symbols of the formula which bring its
    /// parameters near those of `func` (`find_instantiations`).
    fn find_instantiations(&self, func_: &HyperFunction) -> PolyResult<Vec<Repl>> {
        let (ap, bq) = (&func_.ap, &func_.bq);
        if ap.len() != self.func.ap.len() || bq.len() != self.func.bq.len() {
            return Err(PolyError::Type(
                "Cannot instantiate other number of parameters".to_string(),
            ));
        }
        let mut symbol_values: Vec<&Vec<Basic>> = Vec::new();
        for a in &self.symbols {
            if self.func.ap.contains(a) {
                symbol_values.push(ap);
            } else if self.func.bq.contains(a) {
                symbol_values.push(bq);
            } else {
                return Err(value_error(format!(
                    "At least one of the parameters of the formula must be equal to {}",
                    a
                )));
            }
        }
        let base_repl: Vec<Repl> = product(&symbol_values)
            .into_iter()
            .map(|values| self.symbols.iter().cloned().zip(values).collect())
            .collect();
        let (abuckets, bbuckets) = (sift(ap, &mod1)?, sift(bq, &mod1)?);
        let mut critical_values: Vec<Vec<Basic>> = self.symbols.iter().map(|_| vec![int(0)]).collect();
        let mut result: Vec<Repl> = Vec::new();
        let n_: Basic = dummy("n");
        let lookup = |repl: &Repl, a: &Basic| {
            repl.iter()
                .find(|(k, _)| k == a)
                .map(|(_, v)| v.clone())
                .expect("a symbol")
        };
        for repl in &base_repl {
            let key = |x: &Basic| mod1(&replaced(x, repl)?);
            let (symb_a, symb_b) = (sift(&self.func.ap, &key)?, sift(&self.func.bq, &key)?);
            for (bk, obk) in [(&abuckets, &symb_a), (&bbuckets, &symb_b)] {
                for m in keys_union(bk, obk) {
                    let (Some(bm), Some(om)) = (bucket(bk, &m), bucket(obk, &m)) else {
                        break;
                    };
                    if bm.len() != om.len() {
                        break;
                    }
                    for (a, vals) in self.symbols.iter().zip(critical_values.iter_mut()) {
                        let value: Basic = lookup(repl, a);
                        if !value.free_symbols().is_empty() {
                            continue;
                        }
                        let exprs: Vec<&Basic> = om.iter().filter(|e| e.has(a)).collect();
                        let mut repl0: Repl = repl.clone();
                        for entry in repl0.iter_mut() {
                            if entry.0 == *a {
                                entry.1 = a2(&value, &n_)?;
                            }
                        }
                        for e in exprs {
                            for target in bm {
                                let n0: Basic = solve_linear(&s2(&replaced(e, &repl0)?, target)?, &n_)?;
                                if !n0.free_symbols().is_empty() {
                                    return Err(value_error("Value should not be true".to_string()));
                                }
                                vals.push(n0);
                            }
                        }
                    }
                }
            }
            let mut values: Vec<Vec<Basic>> = Vec::new();
            for (a, vals) in self.symbols.iter().zip(critical_values.iter()) {
                let a0: Basic = lookup(repl, a);
                let lo: Basic = crate::functions::elementary::integers::rounding(Function::Floor, &min_of(vals)?)?;
                let hi: Basic = crate::functions::elementary::integers::rounding(Function::Ceiling, &max_of(vals)?)?;
                let (lo, hi) = (as_int(&lo)?, as_int(&hi)?);
                values.push((lo..=hi).map(|k| a2(&a0, &int(k))).collect::<PolyResult<_>>()?);
            }
            let refs: Vec<&Vec<Basic>> = values.iter().collect();
            for choice in product(&refs) {
                result.push(self.symbols.iter().cloned().zip(choice).collect());
            }
        }
        Ok(result)
    }
}

/// `itertools.product` of lists, in its order.
fn product(lists: &[&Vec<Basic>]) -> Vec<Vec<Basic>> {
    let mut out: Vec<Vec<Basic>> = vec![Vec::new()];
    for list in lists {
        let mut next: Vec<Vec<Basic>> = Vec::new();
        for prefix in &out {
            for v in list.iter() {
                let mut p: Vec<Basic> = prefix.clone();
                p.push(v.clone());
                next.push(p);
            }
        }
        out = next;
    }
    out
}

/// The table of formulas by sizes (`FormulaCollection`).
pub struct FormulaCollection {
    symbolic: Vec<((usize, usize), Vec<Formula>)>,
    concrete: Vec<((usize, usize), Vec<(Invariants, Formula)>)>,
}

impl FormulaCollection {
    fn new() -> PolyResult<FormulaCollection> {
        let mut formulae: Vec<Formula> = Vec::new();
        super::hyperexpand_tables::add_formulae(&mut formulae)?;
        let mut symbolic: Vec<((usize, usize), Vec<Formula>)> = Vec::new();
        let mut concrete: Vec<((usize, usize), Vec<(Invariants, Formula)>)> = Vec::new();
        for f in formulae {
            let sizes = f.func.sizes();
            if !f.symbols.is_empty() {
                match symbolic.iter_mut().find(|(s, _)| *s == sizes) {
                    Some(entry) => entry.1.push(f),
                    None => symbolic.push((sizes, vec![f])),
                }
            } else {
                let inv: Invariants = f.func.build_invariants()?;
                let entry = match concrete.iter_mut().position(|(s, _)| *s == sizes) {
                    Some(i) => &mut concrete[i].1,
                    None => {
                        concrete.push((sizes, Vec::new()));
                        let last = concrete.len() - 1;
                        &mut concrete[last].1
                    }
                };
                match entry.iter_mut().find(|(k, _)| *k == inv) {
                    Some(e) => e.1 = f,
                    None => entry.push((inv, f)),
                }
            }
        }
        Ok(FormulaCollection { symbolic, concrete })
    }

    /// The formula nearest to `func` (`lookup_origin`).
    pub fn lookup_origin(&self, func_: &HyperFunction) -> PolyResult<Option<Formula>> {
        let inv: Invariants = func_.build_invariants()?;
        let sizes = func_.sizes();
        if let Some((_, table)) = self.concrete.iter().find(|(s, _)| *s == sizes) {
            if let Some((_, f)) = table.iter().find(|(k, _)| *k == inv) {
                return Ok(Some(f.clone()));
            }
        }
        let Some((_, candidates)) = self.symbolic.iter().find(|(s, _)| *s == sizes) else {
            return Ok(None);
        };
        let mut possible: Vec<(Basic, Repl, &Formula, HyperFunction)> = Vec::new();
        for f in candidates {
            for repl in f.find_instantiations(func_)? {
                let func2: HyperFunction = f.func.xreplace(&repl)?;
                if !func2.is_suitable_origin()? {
                    continue;
                }
                let difference: Basic = func2.difficulty(func_)?;
                if difference == int(-1) {
                    continue;
                }
                possible.push((difference, repl, f, func2));
            }
        }
        let keys: Vec<Basic> = possible.iter().map(|p| p.0.clone()).collect();
        let order: Vec<Basic> = sort_ordered(&(0..keys.len()).map(|i| int(i as i64)).collect::<Vec<_>>(), &|i| {
            Ok(keys[as_int(i)? as usize].clone())
        })?;
        for i in order {
            let (_, repl, f, func2) = &possible[as_int(&i)? as usize];
            let f2 = Formula::with_basis(
                func2.clone(),
                &f.z,
                &[],
                f.b.subs(repl)?,
                f.c.subs(repl)?,
                f.m.subs(repl)?,
            );
            if ![&f2.b, &f2.m, &f2.c].iter().any(|m| has_infinities(m)) {
                return Ok(Some(f2));
            }
        }
        Ok(None)
    }
}

/// How a Meijer formula takes a G-function (`detect_uppergamma`, `detect_3113`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Matcher {
    UpperGamma,
    ThreeOneOneThree,
}

/// A closed form of a Meijer G-function (`MeijerFormula`).
#[derive(Debug, Clone)]
pub struct MeijerFormula {
    pub func: GFunction,
    pub z: Basic,
    pub symbols: Vec<Basic>,
    matcher: Option<Matcher>,
    pub b: ExprMatrix,
    pub c: ExprMatrix,
    pub m: ExprMatrix,
}

impl MeijerFormula {
    pub(crate) fn new(
        func_: GFunction,
        z: &Basic,
        symbols: &[Basic],
        b: ExprMatrix,
        c: ExprMatrix,
        m: ExprMatrix,
        matcher: Option<Matcher>,
    ) -> MeijerFormula {
        MeijerFormula {
            func: func_,
            z: z.clone(),
            symbols: symbols.to_vec(),
            matcher,
            b,
            c,
            m,
        }
    }

    pub fn closed_form(&self) -> R {
        closed_form(&self.c, &self.b)
    }

    fn try_instantiate(&self, func_: &GFunction) -> PolyResult<Option<MeijerFormula>> {
        if func_.signature() != self.func.signature() {
            return Ok(None);
        }
        let Some(matcher) = self.matcher else { return Ok(None) };
        let found = match matcher {
            Matcher::UpperGamma => super::hyperexpand_tables::detect_uppergamma(func_, &self.symbols)?,
            Matcher::ThreeOneOneThree => super::hyperexpand_tables::detect_3113(func_, &self.symbols)?,
        };
        let Some((repl, newfunc)) = found else { return Ok(None) };
        Ok(Some(MeijerFormula::new(
            newfunc,
            &self.z,
            &[],
            self.b.subs(&repl)?,
            self.c.subs(&repl)?,
            self.m.subs(&repl)?,
            None,
        )))
    }
}

/// The Meijer formulas by signature (`MeijerFormulaCollection`).
pub struct MeijerFormulaCollection {
    formulae: Vec<((usize, usize, usize, usize), Vec<MeijerFormula>)>,
}

impl MeijerFormulaCollection {
    fn new() -> PolyResult<MeijerFormulaCollection> {
        let mut list: Vec<MeijerFormula> = Vec::new();
        super::hyperexpand_tables::add_meijerg_formulae(&mut list)?;
        let mut formulae: Vec<((usize, usize, usize, usize), Vec<MeijerFormula>)> = Vec::new();
        for f in list {
            let s = f.func.signature();
            match formulae.iter_mut().find(|(k, _)| *k == s) {
                Some(entry) => entry.1.push(f),
                None => formulae.push((s, vec![f])),
            }
        }
        Ok(MeijerFormulaCollection { formulae })
    }

    pub fn lookup_origin(&self, func_: &GFunction) -> PolyResult<Option<MeijerFormula>> {
        if let Some((_, list)) = self.formulae.iter().find(|(k, _)| *k == func_.signature()) {
            for f in list {
                if let Some(found) = f.try_instantiate(func_)? {
                    return Ok(Some(found));
                }
            }
        }
        Ok(None)
    }
}

thread_local! {
    static COLLECTION: RefCell<Option<Rc<FormulaCollection>>> = const { RefCell::new(None) };
    static MEIJER_COLLECTION: RefCell<Option<Rc<MeijerFormulaCollection>>> = const { RefCell::new(None) };
}

fn formula_collection() -> PolyResult<Rc<FormulaCollection>> {
    if let Some(c) = COLLECTION.with(|c| c.borrow().clone()) {
        return Ok(c);
    }
    let c: Rc<FormulaCollection> = Rc::new(FormulaCollection::new()?);
    COLLECTION.with(|slot| *slot.borrow_mut() = Some(c.clone()));
    Ok(c)
}

fn meijer_collection() -> PolyResult<Rc<MeijerFormulaCollection>> {
    if let Some(c) = MEIJER_COLLECTION.with(|c| c.borrow().clone()) {
        return Ok(c);
    }
    let c: Rc<MeijerFormulaCollection> = Rc::new(MeijerFormulaCollection::new()?);
    MEIJER_COLLECTION.with(|slot| *slot.borrow_mut() = Some(c.clone()));
    Ok(c)
}

pub(crate) fn mod1_pub(x: &Basic) -> R {
    mod1(x)
}

pub(crate) fn truth_pub(b: &Basic) -> PolyResult<bool> {
    truth(b)
}

/// The coefficients of a polynomial in `x`, highest degree first, each
/// cancelled (`_poly_coeffs`).
fn poly_coeffs(e: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let expanded: Basic = expand(e)?;
    let terms: Vec<Basic> = if expanded.is_add() {
        expanded.args().to_vec()
    } else {
        vec![expanded.clone()]
    };
    let mut by_degree: std::collections::BTreeMap<i64, Vec<Basic>> = std::collections::BTreeMap::new();
    for t in terms {
        let (c, k) = crate::solvers::solvers::as_coeff_exponent(&t, x)?;
        let degree: Option<i64> = if is_integer_node(&k) { Some(as_int(&k)?) } else { None };
        match degree {
            Some(d) if d >= 0 && !c.has(x) => by_degree.entry(d).or_default().push(c),
            _ => {
                return Err(crate::error::polys_error(
                    crate::error::PolysErrorKind::PolynomialError,
                    format!("{} is not a polynomial in {}", e, x),
                ))
            }
        }
    }
    let degree: i64 = by_degree.keys().next_back().copied().unwrap_or(0);
    (0..=degree)
        .rev()
        .map(|k| {
            crate::polys::polytools::cancel(&add_all(by_degree.get(&k).map(|v| v.as_slice()).unwrap_or(&[]))?, &[])
        })
        .collect()
}

fn from_coeffs(coeffs: &[Basic], x: &Basic) -> R {
    let n: usize = coeffs.len() - 1;
    let terms: Vec<Basic> = coeffs
        .iter()
        .enumerate()
        .map(|(k, c)| m2(c, &pw(x, &int((n - k) as i64))?))
        .collect::<PolyResult<_>>()?;
    add_all(&terms)
}

/// A differential operator: a polynomial in `z*d/dz`, its coefficients
/// highest degree first (`Operator` and its subclasses).
#[derive(Debug, Clone)]
pub struct Operator {
    pub coeffs: Vec<Basic>,
}

impl Operator {
    fn apply(&self, obj: &Basic, op: &dyn Fn(&Basic) -> R) -> R {
        let coeffs: Vec<Basic> = self.coeffs.iter().rev().cloned().collect();
        let mut diffs: Vec<Basic> = vec![obj.clone()];
        for _ in &coeffs[1..] {
            let last: Basic = diffs[diffs.len() - 1].clone();
            diffs.push(op(&last)?);
        }
        let mut r: Basic = m2(&coeffs[0], &diffs[0])?;
        for (c, d) in coeffs[1..].iter().zip(diffs[1..].iter()) {
            r = a2(&r, &m2(c, d)?)?;
        }
        Ok(r)
    }

    fn apply_matrix(
        &self,
        obj: &ExprMatrix,
        op: &dyn Fn(&ExprMatrix) -> PolyResult<ExprMatrix>,
    ) -> PolyResult<ExprMatrix> {
        let coeffs: Vec<Basic> = self.coeffs.iter().rev().cloned().collect();
        let mut diffs: Vec<ExprMatrix> = vec![obj.clone()];
        for _ in &coeffs[1..] {
            let last: ExprMatrix = diffs[diffs.len() - 1].clone();
            diffs.push(op(&last)?);
        }
        let mut r: ExprMatrix = diffs[0].scale(&coeffs[0])?;
        for (c, d) in coeffs[1..].iter().zip(diffs[1..].iter()) {
            r = r.add(&d.scale(c)?)?;
        }
        Ok(r)
    }
}

fn op_of(e: &Basic) -> PolyResult<Operator> {
    Ok(Operator {
        coeffs: poly_coeffs(e, &x_())?,
    })
}

fn shift_a(ai: &Basic) -> PolyResult<Operator> {
    if *ai == int(0) {
        return Err(value_error("Cannot increment zero upper index.".to_string()));
    }
    op_of(&a2(&d2(&x_(), ai)?, &int(1))?)
}

fn shift_b(bi: &Basic) -> PolyResult<Operator> {
    if *bi == int(1) {
        return Err(value_error("Cannot decrement unit lower index.".to_string()));
    }
    op_of(&a2(&d2(&x_(), &s2(bi, &int(1))?)?, &int(1))?)
}

fn unshift(m: &Basic, n: &Basic, a: &Basic, value: &Basic, b0_sign: i64, minus_n: bool) -> PolyResult<Operator> {
    let coeffs: Vec<Basic> = poly_coeffs(n, a)?;
    let b0: Basic = m2(&int(b0_sign), &coeffs[coeffs.len() - 1])?;
    if b0 == int(0) {
        return Err(value_error("cancels".to_string()));
    }
    let nx: Basic = if coeffs.len() > 1 {
        from_coeffs(&coeffs[..coeffs.len() - 1], value)?
    } else {
        int(0)
    };
    op_of(&d2(&if minus_n { s2(m, &nx)? } else { s2(&nx, m)? }, &b0)?)
}

fn products(factors: impl Iterator<Item = R>) -> R {
    mul_all(&factors.collect::<PolyResult<Vec<Basic>>>()?)
}

fn unshift_a(ap: &[Basic], bq: &[Basic], i: usize, z: &Basic) -> PolyResult<Operator> {
    let x: Basic = x_();
    let mut aps: Vec<Basic> = ap.to_vec();
    let ai: Basic = s2(&aps.remove(i), &int(1))?;
    if ai == int(0) {
        return Err(value_error("Cannot decrement unit upper index.".to_string()));
    }
    let m: Basic = m2(&m2(z, &ai)?, &products(aps.iter().map(|a| a2(&x, a)))?)?;
    let big_a: Basic = dummy("A");
    let d: Basic = s2(&m2(&ai, &big_a)?, &ai)?;
    let n: Basic = m2(&d, &products(bq.iter().map(|b| a2(&d, &s2(b, &int(1))?)))?)?;
    unshift(&m, &n, &big_a, &a2(&d2(&x, &ai)?, &int(1))?, -1, false)
        .map_err(|_| value_error("Cannot decrement upper index: cancels with lower".to_string()))
}

fn unshift_b(ap: &[Basic], bq: &[Basic], i: usize, z: &Basic) -> PolyResult<Operator> {
    let x: Basic = x_();
    let mut bqs: Vec<Basic> = bq.to_vec();
    let bi: Basic = a2(&bqs.remove(i), &int(1))?;
    if bi == int(0) {
        return Err(value_error("Cannot increment -1 lower index.".to_string()));
    }
    let bi1: Basic = s2(&bi, &int(1))?;
    let m: Basic = m2(&m2(&x, &bi1)?, &products(bqs.iter().map(|b| s2(&a2(&x, b)?, &int(1))))?)?;
    let big_b: Basic = dummy("B");
    let d: Basic = a2(&s2(&m2(&bi1, &big_b)?, &bi)?, &int(1))?;
    let n: Basic = m2(z, &products(ap.iter().map(|a| a2(&d, a)))?)?;
    unshift(&m, &n, &big_b, &a2(&d2(&x, &bi1)?, &int(1))?, 1, true)
        .map_err(|_| value_error("Cannot increment index: cancels with upper".to_string()))
}

fn meijer_shift(kind: char, bi: &Basic) -> PolyResult<Operator> {
    let x: Basic = x_();
    let e: Basic = match kind {
        'A' => s2(bi, &x)?,
        'B' => a2(&s2(&int(1), bi)?, &x)?,
        'C' => a2(&neg(bi)?, &x)?,
        _ => s2(&s2(bi, &int(1))?, &x)?,
    };
    op_of(&e)
}

fn meijer_unshift(
    kind: char,
    an: &[Basic],
    ap: &[Basic],
    bm: &[Basic],
    bq: &[Basic],
    i: usize,
    z: &Basic,
) -> PolyResult<Operator> {
    let x: Basic = x_();
    let one: Basic = int(1);
    let rest = |list: &[Basic]| -> Vec<Basic> {
        list.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| v.clone())
            .collect()
    };
    match kind {
        'A' | 'C' => {
            let (bi, bms, bqs): (Basic, Vec<Basic>, Vec<Basic>) = if kind == 'A' {
                (s2(&bm[i], &one)?, rest(bm), bq.to_vec())
            } else {
                (s2(&bq[i], &one)?, bm.to_vec(), rest(bq))
            };
            let m: Basic = m2(
                &products(bms.iter().map(|b| s2(b, &x)))?,
                &products(bqs.iter().map(|b| s2(&x, b)))?,
            )?;
            let symbol: Basic = dummy(if kind == 'A' { "A" } else { "C" });
            let d: Basic = if kind == 'A' {
                s2(&bi, &symbol)?
            } else {
                a2(&bi, &symbol)?
            };
            let n: Basic = m2(
                &m2(z, &products(an.iter().map(|a| s2(&a2(&d, &one)?, a)))?)?,
                &products(ap.iter().map(|a| s2(&a2(&neg(&d)?, a)?, &one)))?,
            )?;
            let value: Basic = if kind == 'A' { s2(&bi, &x)? } else { s2(&x, &bi)? };
            unshift(&m, &n, &symbol, &value, 1, true).map_err(|_| {
                value_error(
                    if kind == 'A' {
                        "Cannot decrement upper b index (cancels)"
                    } else {
                        "Cannot decrement lower b index (cancels)"
                    }
                    .to_string(),
                )
            })
        }
        _ => {
            let (ai, ans, aps): (Basic, Vec<Basic>, Vec<Basic>) = if kind == 'B' {
                (a2(&an[i], &one)?, rest(an), ap.to_vec())
            } else {
                (a2(&ap[i], &one)?, an.to_vec(), rest(ap))
            };
            let m: Basic = m2(
                &m2(z, &products(ans.iter().map(|a| a2(&s2(&one, a)?, &x)))?)?,
                &products(aps.iter().map(|a| s2(&s2(a, &one)?, &x)))?,
            )?;
            let symbol: Basic = dummy("B");
            let d: Basic = if kind == 'B' {
                s2(&a2(&symbol, &ai)?, &one)?
            } else {
                s2(&s2(&ai, &one)?, &symbol)?
            };
            let n: Basic = m2(
                &products(bm.iter().map(|b| a2(&neg(&d)?, b)))?,
                &products(bq.iter().map(|b| s2(&d, b)))?,
            )?;
            let value: Basic = if kind == 'B' {
                a2(&s2(&one, &ai)?, &x)?
            } else {
                s2(&s2(&ai, &one)?, &x)?
            };
            unshift(&m, &n, &symbol, &value, 1, true).map_err(|_| {
                value_error(
                    if kind == 'B' {
                        "Cannot increment upper a index (cancels)"
                    } else {
                        "Cannot increment lower a index (cancels)"
                    }
                    .to_string(),
                )
            })
        }
    }
}

fn reduce_make(ai: &Basic, bj: &Basic) -> PolyResult<Option<Operator>> {
    let n: Basic = s2(ai, bj)?;
    if !is_integer_node(&n) || as_int(&n)? < 0 {
        return Ok(None);
    }
    if yes(bj, Predicate::integer) && yes(bj, Predicate::nonpositive) {
        return Ok(None);
    }
    let x: Basic = x_();
    let mut p: Basic = int(1);
    for k in 0..as_int(&n)? {
        p = d2(&m2(&p, &a2(&a2(&x, bj)?, &int(k))?)?, &a2(bj, &int(k))?)?;
    }
    Ok(Some(op_of(&p)?))
}

fn reduce_meijer(b: &Basic, a: &Basic, sign: i64) -> PolyResult<Option<Operator>> {
    let n: Basic = s2(b, a)?;
    if yes(&n, Predicate::negative) || !is_integer_node(&n) {
        return Ok(None);
    }
    let x: Basic = x_();
    let mut p: Basic = int(1);
    for k in 0..as_int(&n)? {
        p = m2(&p, &a2(&a2(&m2(&int(sign), &x)?, a)?, &int(k))?)?;
    }
    Ok(Some(op_of(&p)?))
}

fn reduce_order_with(
    ap: &[Basic],
    bq: &[Basic],
    gen: &dyn Fn(&Basic, &Basic) -> PolyResult<Option<Operator>>,
    key: &dyn Fn(&Basic) -> PolyResult<crate::core::sorting::SortKey>,
) -> PolyResult<(Vec<Basic>, Vec<Basic>, Vec<Operator>)> {
    let sorted = |v: &[Basic]| -> PolyResult<Vec<Basic>> {
        let mut keyed: Vec<(crate::core::sorting::SortKey, Basic)> =
            v.iter().map(|e| Ok((key(e)?, e.clone()))).collect::<PolyResult<_>>()?;
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(keyed.into_iter().map(|(_, e)| e).collect())
    };
    let aps: Vec<Basic> = sorted(ap)?;
    let mut bqs: Vec<Basic> = sorted(bq)?;
    let mut nap: Vec<Basic> = Vec::new();
    let mut operators: Vec<Operator> = Vec::new();
    for a in aps {
        let mut op: Option<Operator> = None;
        for i in 0..bqs.len() {
            op = gen(&a, &bqs[i])?;
            if op.is_some() {
                bqs.remove(i);
                break;
            }
        }
        match op {
            None => nap.push(a),
            Some(o) => operators.push(o),
        }
    }
    Ok((nap, bqs, operators))
}

/// `func` with the upper and lower parameters differing by a nonnegative
/// integer cancelled (`reduce_order`).
pub fn reduce_order(func_: &HyperFunction) -> PolyResult<(HyperFunction, Vec<Operator>)> {
    let (nap, nbq, operators) = reduce_order_with(&func_.ap, &func_.bq, &reduce_make, &|e| Ok(sort_key(e)))?;
    Ok((HyperFunction::new(&nap, &nbq)?, operators))
}

/// The same for a G-function (`reduce_order_meijer`).
pub fn reduce_order_meijer(func_: &GFunction) -> PolyResult<(GFunction, Vec<Operator>)> {
    let plus = |a: &Basic, b: &Basic| reduce_meijer(&s2(&int(1), a)?, &s2(&int(1), b)?, 1);
    let minus = |b: &Basic, a: &Basic| reduce_meijer(b, a, -1);
    let (nan, nbq, mut ops1) = reduce_order_with(&func_.an, &func_.bq, &plus, &|e| Ok(sort_key(&neg(e)?)))?;
    let (nbm, nap, ops2) = reduce_order_with(&func_.bm, &func_.ap, &minus, &|e| Ok(sort_key(e)))?;
    ops1.extend(ops2);
    Ok((GFunction::new(&nan, &nap, &nbm, &nbq)?, ops1))
}

fn derivative_operator(m: &ExprMatrix, z: &Basic) -> impl Fn(&ExprMatrix) -> PolyResult<ExprMatrix> {
    let (m, z) = (m.clone(), z.clone());
    move |c: &ExprMatrix| -> PolyResult<ExprMatrix> {
        let simp = make_simp(&z);
        c.diff(&z)?.scale(&z)?.add(&c.matmul(&m)?)?.applyfunc(&|e| simp(e))
    }
}

fn apply_operators(obj: &Basic, ops: &[Operator], op: &dyn Fn(&Basic) -> R) -> R {
    let mut res: Basic = obj.clone();
    for o in ops.iter().rev() {
        res = o.apply(&res, op)?;
    }
    Ok(res)
}

fn apply_operators_matrix(
    obj: &ExprMatrix,
    ops: &[Operator],
    op: &dyn Fn(&ExprMatrix) -> PolyResult<ExprMatrix>,
) -> PolyResult<ExprMatrix> {
    let mut res: ExprMatrix = obj.clone();
    for o in ops.iter().rev() {
        res = o.apply_matrix(&res, op)?;
    }
    Ok(res)
}

fn others(dic: &Buckets, key: &Basic) -> Vec<Basic> {
    dic.iter()
        .filter(|(k, _)| k != key)
        .flat_map(|(_, v)| v.iter().cloned())
        .collect()
}

fn sorted_by_key(v: &[Basic]) -> Vec<Basic> {
    let mut v: Vec<Basic> = v.to_vec();
    v.sort_by_cached_key(sort_key);
    v
}

fn set_bucket(b: &mut Buckets, k: &Basic, v: Vec<Basic>) {
    match b.iter_mut().find(|(key, _)| key == k) {
        Some(entry) => entry.1 = v,
        None => b.push((k.clone(), v)),
    }
}

/// The shifts taking `origin` to `target` (`devise_plan`).
pub fn devise_plan(target: &HyperFunction, origin: &HyperFunction, z: &Basic) -> PolyResult<Vec<Operator>> {
    let (abuckets, bbuckets) = (sift(&target.ap, &mod1)?, sift(&target.bq, &mod1)?);
    let (mut nabuckets, mut nbbuckets) = (sift(&origin.ap, &mod1)?, sift(&origin.bq, &mod1)?);
    let unreachable = || value_error(format!("{:?} not reachable from {:?}", target, origin));
    if abuckets.len() != nabuckets.len() || bbuckets.len() != nbbuckets.len() {
        return Err(unreachable());
    }
    let mut ops: Vec<Operator> = Vec::new();
    let do_shifts = |fro: &mut Vec<Basic>,
                     to: &[Basic],
                     inc: &dyn Fn(&[Basic], usize) -> PolyResult<Operator>,
                     dec: &dyn Fn(&[Basic], usize) -> PolyResult<Operator>|
     -> PolyResult<Vec<Operator>> {
        let mut shifts: Vec<Operator> = Vec::new();
        for i in 0..fro.len() {
            let up: bool = gt0(&s2(&to[i], &fro[i])?)?;
            while to[i] != fro[i] {
                shifts.push(if up { inc(fro, i)? } else { dec(fro, i)? });
                fro[i] = a2(&fro[i], &int(if up { 1 } else { -1 }))?;
            }
        }
        Ok(shifts)
    };
    let mut keys: Vec<Basic> = abuckets
        .iter()
        .map(|(k, _)| k.clone())
        .chain(bbuckets.iter().map(|(k, _)| k.clone()))
        .collect();
    keys.sort_by_cached_key(sort_key);
    for r in keys {
        let al: Vec<Basic> = sorted_by_key(bucket(&abuckets, &r).map(|v| v.as_slice()).unwrap_or(&[]));
        let nal: Vec<Basic> = if bucket(&abuckets, &r).is_some() {
            sorted_by_key(bucket(&nabuckets, &r).map(|v| v.as_slice()).unwrap_or(&[]))
        } else {
            Vec::new()
        };
        let bk: Vec<Basic> = sorted_by_key(bucket(&bbuckets, &r).map(|v| v.as_slice()).unwrap_or(&[]));
        let nbk: Vec<Basic> = if bucket(&bbuckets, &r).is_some() {
            sorted_by_key(bucket(&nbbuckets, &r).map(|v| v.as_slice()).unwrap_or(&[]))
        } else {
            Vec::new()
        };
        if al.len() != nal.len() || bk.len() != nbk.len() {
            return Err(unreachable());
        }
        let aother: Vec<Basic> = others(&nabuckets, &r);
        let bother: Vec<Basic> = others(&nbbuckets, &r);
        let shifts_a = |nal: &mut Vec<Basic>, nbk: &[Basic], al: &[Basic]| -> PolyResult<Vec<Operator>> {
            let inc = |p: &[Basic], i: usize| shift_a(&p[i]);
            let dec = |p: &[Basic], i: usize| {
                let mut ap: Vec<Basic> = p.to_vec();
                ap.extend(aother.iter().cloned());
                let mut bq: Vec<Basic> = nbk.to_vec();
                bq.extend(bother.iter().cloned());
                unshift_a(&ap, &bq, i, z)
            };
            do_shifts(nal, al, &inc, &dec)
        };
        let shifts_b = |nal: &[Basic], nbk: &mut Vec<Basic>, bk: &[Basic]| -> PolyResult<Vec<Operator>> {
            let inc = |p: &[Basic], i: usize| {
                let mut ap: Vec<Basic> = nal.to_vec();
                ap.extend(aother.iter().cloned());
                let mut bq: Vec<Basic> = p.to_vec();
                bq.extend(bother.iter().cloned());
                unshift_b(&ap, &bq, i, z)
            };
            let dec = |p: &[Basic], i: usize| shift_b(&p[i]);
            do_shifts(nbk, bk, &inc, &dec)
        };
        let (mut nal, mut nbk) = (nal, nbk);
        if al.is_empty() {
            ops.extend(shifts_b(&[], &mut nbk, &bk)?);
        } else if bk.is_empty() {
            ops.extend(shifts_a(&mut nal, &[], &al)?);
        } else {
            let (namax, amax) = (nal[nal.len() - 1].clone(), al[al.len() - 1].clone());
            if le0(&s2(&nbk[0], &namax)?)? || le0(&s2(&bk[0], &amax)?)? {
                return Err(value_error("Non-suitable parameters.".to_string()));
            }
            if gt0(&s2(&namax, &amax)?)? {
                let nbk_now: Vec<Basic> = nbk.clone();
                ops.extend(shifts_a(&mut nal, &nbk_now, &al)?);
                ops.extend(shifts_b(&al, &mut nbk, &bk)?);
            } else {
                let nal_now: Vec<Basic> = nal.clone();
                ops.extend(shifts_b(&nal_now, &mut nbk, &bk)?);
                ops.extend(shifts_a(&mut nal, &bk, &al)?);
            }
        }
        set_bucket(&mut nabuckets, &r, al);
        set_bucket(&mut nbbuckets, &r, bk);
    }
    ops.reverse();
    Ok(ops)
}

fn rf(a: &Basic, k: &Basic) -> R {
    crate::functions::combinatorial::factorials::rf(a, k)
}
fn factorial(n: &Basic) -> R {
    crate::functions::combinatorial::factorials::factorial(n)
}
fn gamma(z: &Basic) -> R {
    crate::functions::special::gamma_functions::gamma(z)
}
fn simplify(e: &Basic) -> R {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}
fn unpolarify(e: &Basic) -> R {
    crate::functions::elementary::polar::unpolarify(e, false)
}
fn powdenest_polar(e: &Basic) -> R {
    crate::simplify::powsimp::powdenest(e, false, true)
}

/// A sum shifted to start at the integer lower parameter (`try_shifted_sum`).
fn try_shifted_sum(func_: &HyperFunction, z: &Basic) -> PolyResult<Option<(HyperFunction, Vec<Operator>, Basic)>> {
    let zero: Basic = int(0);
    let (abuckets, bbuckets) = (sift(&func_.ap, &mod1)?, sift(&func_.bq, &mod1)?);
    let Some(a0) = bucket(&abuckets, &zero) else {
        return Ok(None);
    };
    if a0.len() != 1 {
        return Ok(None);
    }
    let r: Basic = a0[0].clone();
    if le0(&r)? {
        return Ok(None);
    }
    let Some(b0) = bucket(&bbuckets, &zero) else {
        return Ok(None);
    };
    let mut k: Basic = ordered(b0)?[0].clone();
    if le0(&k)? {
        return Ok(None);
    }
    let mut nap: Vec<Basic> = func_.ap.clone();
    if let Some(i) = nap.iter().position(|x| *x == r) {
        nap.remove(i);
    }
    let mut nbq: Vec<Basic> = func_.bq.clone();
    if let Some(i) = nbq.iter().position(|x| *x == k) {
        nbq.remove(i);
    }
    k = s2(&k, &int(1))?;
    let nap: Vec<Basic> = nap.iter().map(|x| s2(x, &k)).collect::<PolyResult<_>>()?;
    let nbq: Vec<Basic> = nbq.iter().map(|x| s2(x, &k)).collect::<PolyResult<_>>()?;
    let mut ops: Vec<Operator> = (0..as_int(&r)? - 1)
        .map(|n| shift_a(&int(n + 1)))
        .collect::<PolyResult<_>>()?;
    ops.reverse();
    let mut fac: Basic = d2(&factorial(&k)?, &pw(z, &k)?)?;
    fac = m2(&fac, &products(nbq.iter().map(|b| rf(b, &k)))?)?;
    fac = d2(&fac, &products(nap.iter().map(|a| rf(a, &k)))?)?;
    ops.push(op_of(&fac)?);
    let mut p: Basic = int(0);
    for n in 0..as_int(&k)? {
        let nn: Basic = int(n);
        let mut m: Basic = d2(&pw(z, &nn)?, &factorial(&nn)?)?;
        m = m2(&m, &products(nap.iter().map(|a| rf(a, &nn)))?)?;
        m = d2(&m, &products(nbq.iter().map(|b| rf(b, &nn)))?)?;
        p = a2(&p, &m)?;
    }
    Ok(Some((HyperFunction::new(&nap, &nbq)?, ops, neg(&p)?)))
}

/// The polynomial of a terminating series, `oo` for a divergent one
/// (`try_polynomial`).
fn try_polynomial(func_: &HyperFunction, z: &Basic) -> PolyResult<Option<Basic>> {
    let zero: Basic = int(0);
    let (abuckets, bbuckets) = (sift(&func_.ap, &mod1)?, sift(&func_.bq, &mod1)?);
    let a0: Vec<Basic> = ordered(bucket(&abuckets, &zero).map(|v| v.as_slice()).unwrap_or(&[]))?;
    let b0: Vec<Basic> = ordered(bucket(&bbuckets, &zero).map(|v| v.as_slice()).unwrap_or(&[]))?;
    let mut al0: Vec<Basic> = Vec::new();
    for x in a0 {
        if le0(&x)? {
            al0.push(x);
        }
    }
    let mut bl0: Vec<Basic> = Vec::new();
    for x in b0 {
        if le0(&x)? {
            bl0.push(x);
        }
    }
    if let Some(last) = bl0.last() {
        let mut all: bool = true;
        for a in &al0 {
            if !lt(a, last)? {
                all = false;
                break;
            }
        }
        if all {
            return Ok(Some(Basic::oo()));
        }
    }
    let Some(a) = al0.last() else { return Ok(None) };
    let mut fac: Basic = int(1);
    let mut res: Basic = int(1);
    for n in 0..-as_int(a)? {
        let nn: Basic = int(n);
        fac = m2(&fac, z)?;
        fac = d2(&fac, &int(n + 1))?;
        fac = m2(&fac, &products(func_.ap.iter().map(|x| a2(x, &nn)))?)?;
        fac = d2(&fac, &products(func_.bq.iter().map(|b| a2(b, &nn)))?)?;
        res = a2(&res, &fac)?;
    }
    Ok(Some(res))
}

/// A formula in `lerchphi` when the parameters pair up in the same buckets
/// (`try_lerchphi`).
fn try_lerchphi(func_: &HyperFunction) -> PolyResult<Option<Formula>> {
    let zero: Basic = int(0);
    let one: Basic = int(1);
    let abuckets = sift(&func_.ap, &mod1)?;
    let mut bbuckets = sift(&func_.bq, &mod1)?;
    let mut paired: Vec<(Basic, (Vec<Basic>, Vec<Basic>))> = Vec::new();
    for (key, value) in &abuckets {
        if *key != zero && bucket(&bbuckets, key).is_none() {
            return Ok(None);
        }
        let bvalue: Vec<Basic> = bucket(&bbuckets, key).cloned().unwrap_or_default();
        paired.push((key.clone(), (value.clone(), bvalue)));
        bbuckets.retain(|(k, _)| k != key);
    }
    if !bbuckets.is_empty() {
        return Ok(None);
    }
    let Some(entry) = paired.iter_mut().find(|(k, _)| *k == zero) else {
        return Ok(None);
    };
    entry.1 .1.push(one.clone());
    let t: Basic = dummy("t");
    let (mut numer, mut denom) = (int(1), int(1));
    for (_, (avalue, bvalue)) in &paired {
        if avalue.len() != bvalue.len() {
            return Ok(None);
        }
        for (a, b) in avalue.iter().zip(bvalue.iter()) {
            let d: Basic = s2(a, b)?;
            if yes(&d, Predicate::positive) {
                numer = m2(&numer, &rf(&a2(b, &t)?, &d)?)?;
                denom = m2(&denom, &rf(b, &d)?)?;
            } else {
                let k: Basic = s2(b, a)?;
                numer = m2(&numer, &rf(a, &k)?)?;
                denom = m2(&denom, &rf(&a2(a, &t)?, &k)?)?;
            }
        }
    }
    let part: Basic = crate::polys::partfrac::apart(&d2(&numer, &denom)?, Some(&t), false)?;
    let mut monomials: Vec<(Basic, i64)> = Vec::new();
    let mut terms: Vec<(Basic, Vec<(Basic, i64)>)> = Vec::new();
    let parts: Vec<Basic> = if part.is_add() {
        part.args().to_vec()
    } else {
        vec![part.clone()]
    };
    for arg in parts {
        let (num, den) = crate::core::expr::as_numer_denom(&arg)?;
        if !den.has(&t) {
            let p = crate::polys::polytools::Poly::new(&num, std::slice::from_ref(&t))?;
            if p.length() > 1 {
                return Err(PolyError::Type("p should be monomial".to_string()));
            }
            let (monomial, lc) = p
                .terms()?
                .into_iter()
                .next()
                .ok_or_else(|| PolyError::Type("p should be monomial".to_string()))?;
            monomials.push((d2(&lc, &den)?, monomial[0] as i64));
            continue;
        }
        if num.has(&t) {
            return Err(PolyError::NotImplemented(
                "Need partial fraction decomposition with linear denominators".to_string(),
            ));
        }
        let (mut indep, mut dep) = crate::core::expr::as_independent(&den, &t, false)?;
        if dep.args().len() > 1 && dep.is_mul() {
            return Err(value_error(format!("more than one factor depends on {}", t)));
        }
        let mut n: i64 = 1;
        if dep.is_pow() {
            n = as_int(&dep.args()[1])?;
            dep = dep.args()[0].clone();
        }
        let shift: Basic = if dep == t {
            int(0)
        } else if dep.is_add() {
            let (shift0, tmp) = crate::core::expr::as_independent(&dep, &t, true)?;
            let mut scale: Basic = int(1);
            if tmp != t {
                scale = crate::core::expr::as_independent(&tmp, &t, false)?.0;
            }
            if dep != a2(&m2(&scale, &t)?, &shift0)? {
                return Err(PolyError::NotImplemented(format!("unrecognised form {}", dep)));
            }
            indep = m2(&indep, &pw(&scale, &int(n))?)?;
            d2(&shift0, &scale)?
        } else {
            return Err(PolyError::NotImplemented(
                "unrecognised form of partial fraction".to_string(),
            ));
        };
        let entry: (Basic, i64) = (d2(&num, &indep)?, n);
        match terms.iter_mut().find(|(s, _)| *s == shift) {
            Some(e) => e.1.push(entry),
            None => terms.push((shift, vec![entry])),
        }
    }
    let z: Basic = dummy("z");
    let lerch = |k: i64, shift: &Basic| crate::functions::special::zeta_functions::lerchphi(&z, &int(k), shift);
    monomials.sort_by_key(|m| m.1);
    let mut mon: Vec<Basic> = vec![d2(&one, &s2(&one, &z)?)?];
    if let Some(last) = monomials.last() {
        for k in 0..last.1 {
            let next: Basic = m2(&z, &diff(&mon[k as usize], &z)?)?;
            mon.push(next);
        }
    }
    let mut coeffs: Vec<(Basic, Vec<Basic>)> = Vec::new();
    let push =
        |coeffs: &mut Vec<(Basic, Vec<Basic>)>, k: Basic, v: Basic| match coeffs.iter_mut().find(|(b, _)| *b == k) {
            Some(e) => e.1.push(v),
            None => coeffs.push((k, vec![v])),
        };
    for (coeff, n) in &monomials {
        push(&mut coeffs, one.clone(), m2(coeff, &mon[*n as usize])?);
    }
    let mut deriv: Vec<(Basic, Vec<(Basic, Basic)>)> = Vec::new();
    let set = |deriv: &mut Vec<(Basic, Vec<(Basic, Basic)>)>, k: Basic, v: Vec<(Basic, Basic)>| match deriv
        .iter_mut()
        .find(|(b, _)| *b == k)
    {
        Some(e) => e.1 = v,
        None => deriv.push((k, v)),
    };
    for (shift, lst) in terms.iter_mut() {
        for (c, k3) in lst.iter() {
            push(&mut coeffs, lerch(*k3, shift)?, c.clone());
        }
        lst.sort_by_key(|ck| ck.1);
        let top: i64 = lst[lst.len() - 1].1;
        for k4 in 2..=top {
            set(
                &mut deriv,
                lerch(k4, shift)?,
                vec![(neg(shift)?, lerch(k4, shift)?), (one.clone(), lerch(k4 - 1, shift)?)],
            );
        }
        set(
            &mut deriv,
            lerch(1, shift)?,
            vec![
                (neg(shift)?, lerch(1, shift)?),
                (d2(&one, &s2(&one, &z)?)?, one.clone()),
            ],
        );
    }
    let mut trans: Vec<Basic> = vec![one.clone()];
    for (b, _) in &deriv {
        if !trans.contains(b) {
            trans.push(b.clone());
        }
    }
    let index = |b: &Basic| trans.iter().position(|x| x == b).expect("a basis element");
    let basis: Vec<Basic> = trans
        .iter()
        .map(|b| crate::functions::expand_func::expand_func(b, true))
        .collect::<PolyResult<_>>()?;
    let size: usize = basis.len();
    let mut c_entries: Vec<Basic> = vec![int(0); size];
    for (b, cs) in &coeffs {
        c_entries[index(b)] = add_all(cs)?;
    }
    let mut m_entries: Vec<Basic> = vec![int(0); size * size];
    for (b, lst2) in &deriv {
        for (c, b2) in lst2 {
            m_entries[index(b) * size + index(b2)] = c.clone();
        }
    }
    Ok(Some(Formula::with_basis(
        func_.clone(),
        &z,
        &[],
        ExprMatrix::column(basis),
        ExprMatrix::new(1, size, c_entries)?,
        ExprMatrix::new(size, size, m_entries)?,
    )))
}

/// The formula of the function in the basis of its contiguous functions
/// (`build_hypergeometric_formula`).
fn build_hypergeometric_formula(func_: &HyperFunction) -> PolyResult<Formula> {
    let z: Basic = dummy("z");
    let x: Basic = x_();
    let hyper = |ap: &[Basic], bq: &[Basic]| crate::functions::special::hyper::hyper(ap, bq, &z);
    if !func_.ap.is_empty() {
        let afactors: Vec<Basic> = func_.ap.iter().map(|a| a2(&x, a)).collect::<PolyResult<_>>()?;
        let bfactors: Vec<Basic> = func_
            .bq
            .iter()
            .map(|b| s2(&a2(&x, b)?, &int(1)))
            .collect::<PolyResult<_>>()?;
        let poly = crate::polys::polytools::Poly::new(
            &s2(&m2(&x, &mul_all(&bfactors)?)?, &m2(&z, &mul_all(&afactors)?)?)?,
            std::slice::from_ref(&x),
        )?;
        let n: usize = poly.degree(0)? as usize;
        let mut basis: Vec<Basic> = Vec::new();
        let mut m: ExprMatrix = ExprMatrix::zeros(n, n);
        for k in 0..n {
            let a: Basic = a2(&func_.ap[0], &int(k as i64))?;
            let mut ap: Vec<Basic> = vec![a.clone()];
            ap.extend(func_.ap[1..].iter().cloned());
            basis.push(hyper(&ap, &func_.bq)?);
            if k < n - 1 {
                m = m.with_entry(k, k, neg(&a)?).with_entry(k, k + 1, a);
            }
        }
        let b: ExprMatrix = ExprMatrix::column(basis);
        let mut first: Vec<Basic> = vec![int(1)];
        first.extend(std::iter::repeat(int(0)).take(n - 1));
        let c: ExprMatrix = ExprMatrix::from_rows(&[first])?;
        let mut derivs: Vec<ExprMatrix> = vec![ExprMatrix::eye(n)];
        for k in 0..n {
            let next: ExprMatrix = m.matmul(&derivs[k])?;
            derivs.push(next);
        }
        let coeffs: Vec<Basic> = poly.all_coeffs()?;
        let mut res: Vec<Basic> = vec![int(0); n];
        for (k, coefficient) in coeffs.iter().rev().enumerate() {
            let row: ExprMatrix = c.matmul(&derivs[k])?;
            for (r, d) in row.entries.iter().enumerate() {
                res[r] = a2(&res[r], &m2(coefficient, d)?)?;
            }
        }
        for (k, value) in res.iter().enumerate() {
            let entry: Basic = d2(&d2(&neg(value)?, derivs[n - 1].at(0, n - 1))?, &coeffs[0])?;
            m = m.with_entry(n - 1, k, entry);
        }
        return Ok(Formula::with_basis(func_.clone(), &z, &[], b, c, m));
    }
    let mut basis: Vec<Basic> = Vec::new();
    let mut bq: Vec<Basic> = func_.bq.clone();
    for i in 0..bq.len() {
        basis.push(hyper(&[], &bq)?);
        bq[i] = a2(&bq[i], &int(1))?;
    }
    basis.push(hyper(&[], &bq)?);
    let b: ExprMatrix = ExprMatrix::column(basis);
    let n: usize = b.entries.len();
    let mut first: Vec<Basic> = vec![int(1)];
    first.extend(std::iter::repeat(int(0)).take(n - 1));
    let c: ExprMatrix = ExprMatrix::from_rows(&[first])?;
    let mut m: ExprMatrix = ExprMatrix::zeros(n, n).with_entry(0, n - 1, d2(&z, &mul_all(&func_.bq)?)?);
    for k in 1..n {
        m = m
            .with_entry(k, k - 1, func_.bq[k - 1].clone())
            .with_entry(k, k, neg(&func_.bq[k - 1])?);
    }
    Ok(Formula::with_basis(func_.clone(), &z, &[], b, c, m))
}

/// The values of `2F1` at `1` and `-1` (`hyperexpand_special`).
pub fn hyperexpand_special(ap: &[Basic], bq: &[Basic], z: &Basic) -> R {
    let z_: Basic = z.clone();
    let z: Basic = unpolarify(z)?;
    if z == int(0) {
        return Ok(int(1));
    }
    if ap.len() == 2 && bq.len() == 1 {
        let (mut a, mut b, c) = (ap[0].clone(), ap[1].clone(), bq[0].clone());
        if z == int(1) {
            let top: Basic = m2(&gamma(&s2(&s2(&c, &a)?, &b)?)?, &gamma(&c)?)?;
            return d2(&d2(&top, &gamma(&s2(&c, &a)?)?)?, &gamma(&s2(&c, &b)?)?);
        }
        if z == int(-1) && simplify(&a2(&s2(&b, &a)?, &c)?)? == int(1) {
            std::mem::swap(&mut a, &mut b);
        }
        if z == int(-1) && simplify(&a2(&s2(&a, &b)?, &c)?)? == int(1) {
            let b2: Basic = d2(&b, &int(2))?;
            let shifted: Basic = a2(&s2(&b, &a)?, &int(1))?;
            let half_shift: Basic = a2(&s2(&b2, &a)?, &int(1))?;
            if yes(&b, Predicate::integer) && yes(&b, Predicate::negative) {
                let cosine: Basic = apply(Function::Cos, &[d2(&m2(&Basic::pi(), &b)?, &int(2))?])?;
                let top: Basic = m2(&m2(&m2(&int(2), &cosine)?, &gamma(&neg(&b)?)?)?, &gamma(&shifted)?)?;
                return d2(&d2(&top, &gamma(&neg(&b2)?)?)?, &gamma(&half_shift)?);
            }
            let top: Basic = m2(&gamma(&a2(&b2, &int(1))?)?, &gamma(&shifted)?)?;
            return d2(&d2(&top, &gamma(&a2(&b, &int(1))?)?)?, &gamma(&half_shift)?);
        }
    }
    crate::functions::special::hyper::hyper(ap, bq, &z_)
}

fn hyper_parts(node: &Basic) -> (Vec<Basic>, Vec<Basic>, Basic) {
    (
        node.args()[0].args().to_vec(),
        node.args()[1].args().to_vec(),
        node.args()[2].clone(),
    )
}

fn replace_hyper_special(e: &Basic) -> R {
    crate::core::function::replace_where(e, &|n| matches!(n.head(), Head::Hyper), &|n| {
        let (ap, bq, z) = hyper_parts(n);
        hyperexpand_special(&ap, &bq, &z)
    })
}

/// How the representatives of the formulas are rewritten (`rewrite`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rewrite {
    Default,
    NonRepSmall,
    NonRep,
    Keep,
}

fn rewrite_with(e: &Basic, rewrite: Rewrite) -> R {
    match rewrite {
        Rewrite::NonRepSmall => crate::functions::special::hyperrep::rewrite_nonrep(e, true),
        Rewrite::NonRep => crate::functions::special::hyperrep::rewrite_nonrep(e, false),
        _ => Ok(e.clone()),
    }
}

thread_local! {
    static Z0: Basic = dummy("z0");
}

fn hyperexpand_hyper(
    func_: &HyperFunction,
    z: &Basic,
    ops0: &[Operator],
    z0: Option<&Basic>,
    premult: &Basic,
    prem: &Basic,
    rewrite: Rewrite,
) -> R {
    let z0: Basic = z0.cloned().unwrap_or_else(|| Z0.with(|d| d.clone()));
    if yes(z, Predicate::zero) {
        return Ok(int(1));
    }
    let z: Basic = crate::functions::elementary::polar::polarify(z, false, false)?.0;
    let rewrite: Rewrite = if rewrite == Rewrite::Default {
        Rewrite::NonRepSmall
    } else {
        rewrite
    };
    let carryout_plan = |f: &Formula, ops: &[Operator]| -> R {
        let to_z0: Repl = vec![(f.z.clone(), z0.clone())];
        let m0: ExprMatrix = f.m.subs(&to_z0)?;
        let mut c: ExprMatrix = apply_operators_matrix(&f.c.subs(&to_z0)?, ops, &derivative_operator(&m0, &z0))?;
        let shifted: ExprMatrix = m0.add(&ExprMatrix::eye(f.m.rows).scale(prem)?)?;
        c = apply_operators_matrix(&c, ops0, &derivative_operator(&shifted, &z0))?;
        if *premult == int(1) {
            let simp = make_simp(&z0);
            c = c.applyfunc(&|e| simp(e))?;
        }
        let r: Basic = m2(&closed_form(&c, &f.b.subs(&to_z0)?)?, premult)?;
        rewrite_with(&subs(&r, &z0, &z)?, rewrite)
    };
    let d = |f: &Basic| m2(&z0, &diff(f, &z0)?);
    let collection = formula_collection()?;
    let (func_, reduce_ops) = reduce_order(func_)?;
    let mut ops: Vec<Operator> = reduce_ops;
    if let Some(res) = try_polynomial(&func_, &z0)? {
        let p: Basic = apply_operators(&res, &ops, &d)?;
        let p: Basic = apply_operators(&m2(&p, premult)?, ops0, &d)?;
        return unpolarify(&subs(&simplify(&p)?, &z0, &z)?);
    }
    let mut p: Basic = int(0);
    let mut func_: HyperFunction = func_;
    if let Some((f, nops, q)) = try_shifted_sum(&func_, &z0)? {
        func_ = f;
        ops.extend(nops);
        p = q;
    }
    p = apply_operators(&p, &ops, &d)?;
    p = apply_operators(&m2(&p, premult)?, ops0, &d)?;
    p = subs(&simplify(&p)?, &z0, &z)?;
    let unit: Basic = unpolarify(&z)?;
    if (unit == int(1) || unit == int(-1)) && func_.sizes() == (2, 1) {
        let f: Formula = build_hypergeometric_formula(&func_)?;
        let r: Basic = replace_hyper_special(&carryout_plan(&f, &ops)?)?;
        if !r.preorder().iter().any(|n| matches!(n.head(), Head::Hyper)) {
            return a2(&r, &p);
        }
    }
    let formula: Formula = match collection.lookup_origin(&func_)? {
        Some(f) => f,
        None => match try_lerchphi(&func_)? {
            Some(f) => f,
            None => build_hypergeometric_formula(&func_)?,
        },
    };
    ops.extend(devise_plan(&func_, &formula.func, &z0)?);
    let r: Basic = a2(&carryout_plan(&formula, &ops)?, &p)?;
    replace_hyper_special(&powdenest_polar(&r)?)
}

/// The shifts taking the G-function `fro` to `to` (`devise_plan_meijer`).
pub fn devise_plan_meijer(fro: &GFunction, to: &GFunction, z: &Basic) -> PolyResult<Vec<Operator>> {
    let (mut fan, mut fap, mut fbm, mut fbq) = (fro.an.clone(), fro.ap.clone(), fro.bm.clone(), fro.bq.clone());
    let mut ops: Vec<Operator> = Vec::new();
    loop {
        let mut found: Option<Operator> = None;
        // the attempts in their order: (list, target, kind, diff, counter)
        for attempt in 0..8 {
            let (target, diff_, unshift_kind): (&Vec<Basic>, i64, Option<char>) = match attempt {
                0 => (&to.an, 1, Some('B')),
                1 => (&to.ap, 1, Some('D')),
                2 => (&to.bm, -1, Some('A')),
                3 => (&to.bq, -1, Some('C')),
                4 => (&to.an, -1, None),
                5 => (&to.ap, -1, None),
                6 => (&to.bm, 1, None),
                _ => (&to.bq, 1, None),
            };
            let counter: Vec<Basic> = match attempt {
                0 | 1 => fbm.iter().chain(fbq.iter()).cloned().collect(),
                2 | 3 => fan.iter().chain(fap.iter()).cloned().collect(),
                _ => Vec::new(),
            };
            let list: Vec<Basic> = match attempt % 4 {
                0 => fan.clone(),
                1 => fap.clone(),
                2 => fbm.clone(),
                _ => fbq.clone(),
            };
            let mut chosen: Option<usize> = None;
            for (idx, (a, b)) in list.iter().zip(target.iter()).enumerate() {
                let d: Basic = s2(a, b)?;
                if yes(&d, Predicate::integer) && gt0(&d2(&s2(b, a)?, &int(diff_))?)? && counter.iter().all(|x| a != x)
                {
                    chosen = Some(idx);
                    break;
                }
            }
            let Some(idx) = chosen else { continue };
            let op: Operator = match unshift_kind {
                Some(kind) => meijer_unshift(kind, &fan, &fap, &fbm, &fbq, idx, z)?,
                None => {
                    let kind: char = ['B', 'D', 'A', 'C'][attempt % 4];
                    meijer_shift(kind, &list[idx])?
                }
            };
            let slot: &mut Vec<Basic> = match attempt % 4 {
                0 => &mut fan,
                1 => &mut fap,
                2 => &mut fbm,
                _ => &mut fbq,
            };
            slot[idx] = a2(&slot[idx], &int(diff_))?;
            found = Some(op);
            break;
        }
        match found {
            Some(op) => ops.push(op),
            None => break,
        }
    }
    if fan != to.an || fap != to.ap || fbm != to.bm || fbq != to.bq {
        return Err(PolyError::NotImplemented("Could not devise plan.".to_string()));
    }
    ops.reverse();
    Ok(ops)
}

/// A condition of Slater's theorem: decided or a Boolean.
#[derive(Debug, Clone, PartialEq)]
enum Condition {
    Decided(bool),
    Open(Basic),
}

impl Condition {
    fn of(b: Basic) -> Condition {
        if b == crate::core::logic::true_() {
            Condition::Decided(true)
        } else if b == crate::core::logic::false_() {
            Condition::Decided(false)
        } else {
            Condition::Open(b)
        }
    }

    fn subs(&self, old: &Basic, new: &Basic) -> PolyResult<Condition> {
        Ok(match self {
            Condition::Decided(b) => Condition::Decided(*b),
            Condition::Open(e) => Condition::of(subs(e, old, new)?),
        })
    }

    fn is_false(&self) -> bool {
        *self == Condition::Decided(false)
    }

    fn is_true(&self) -> bool {
        *self == Condition::Decided(true)
    }

    fn as_basic(&self) -> Basic {
        match self {
            Condition::Decided(b) => crate::core::logic::boolean(*b),
            Condition::Open(e) => e.clone(),
        }
    }
}

fn can_do(pbm: &Buckets, pap: &Buckets) -> bool {
    for (i, values) in pbm {
        if values.len() > 1 {
            let n: usize = bucket(pap, i).map(|v| v.len()).unwrap_or(0);
            if n + 1 < values.len() {
                return false;
            }
        }
    }
    true
}

fn remove_first(list: &mut Vec<Basic>, item: &Basic) {
    if let Some(i) = list.iter().position(|x| x == item) {
        list.remove(i);
    }
}

fn do_slater(
    an: &[Basic],
    bm: &[Basic],
    ap: &[Basic],
    bq: &[Basic],
    z: &Basic,
    zfinal: &Basic,
    ops: &[Operator],
    t: &Basic,
) -> PolyResult<(Basic, Condition)> {
    let g: GFunction = GFunction::new(an, bm, ap, bq)?;
    let [_, pbm, pap, _] = g.compute_buckets()?;
    if !can_do(&pbm, &pap) {
        return Ok((int(0), Condition::Decided(false)));
    }
    let mut cond: Condition = Condition::Decided(an.len() + ap.len() < bm.len() + bq.len());
    if an.len() + ap.len() == bm.len() + bq.len() {
        cond = Condition::of(crate::core::relational::lt(
            &crate::functions::elementary::complexes::abs_(z)?,
            &int(1),
        )?);
    }
    if cond.is_false() {
        return Ok((int(0), Condition::Decided(false)));
    }
    let polar = |n: i64| crate::functions::elementary::polar::polar_lift(&pw(&int(-1), &int(n))?);
    let mut res: Basic = int(0);
    for (key, values) in &pbm {
        if values.len() == 1 {
            let bh: Basic = values[0].clone();
            let mut fac: Basic = int(1);
            let mut bo: Vec<Basic> = bm.to_vec();
            remove_first(&mut bo, &bh);
            for bj in &bo {
                fac = m2(&fac, &gamma(&s2(bj, &bh)?)?)?;
            }
            for aj in an {
                fac = m2(&fac, &gamma(&s2(&a2(&int(1), &bh)?, aj)?)?)?;
            }
            for bj in bq {
                fac = d2(&fac, &gamma(&s2(&a2(&int(1), &bh)?, bj)?)?)?;
            }
            for aj in ap {
                fac = d2(&fac, &gamma(&s2(aj, &bh)?)?)?;
            }
            let nap: Vec<Basic> = an
                .iter()
                .chain(ap.iter())
                .map(|a| s2(&a2(&int(1), &bh)?, a))
                .collect::<PolyResult<_>>()?;
            let nbq: Vec<Basic> = bo
                .iter()
                .chain(bq.iter())
                .map(|b| s2(&a2(&int(1), &bh)?, b))
                .collect::<PolyResult<_>>()?;
            let k: Basic = polar(ap.len() as i64 - bm.len() as i64)?;
            let harg: Basic = m2(&k, zfinal)?;
            let premult: Basic = pw(&d2(t, &k)?, &bh)?;
            let hyp: Basic = hyperexpand_hyper(
                &HyperFunction::new(&nap, &nbq)?,
                &harg,
                ops,
                Some(t),
                &premult,
                &bh,
                Rewrite::Keep,
            )?;
            res = a2(&res, &m2(&fac, &hyp)?)?;
        } else {
            let b_: Basic = values[0].clone();
            let ki: Vec<Basic> = values[1..].iter().map(|bi| s2(bi, &b_)).collect::<PolyResult<_>>()?;
            let u: usize = ki.len();
            let pap_m: Vec<Basic> = bucket(&pap, key)
                .cloned()
                .ok_or_else(|| value_error(format!("KeyError {}", key)))?;
            let li: Vec<Basic> = pap_m
                .iter()
                .take(u + 1)
                .map(|ai| s2(ai, &b_))
                .collect::<PolyResult<_>>()?;
            let mut bo: Vec<Basic> = bm.to_vec();
            for b in values {
                remove_first(&mut bo, b);
            }
            let mut ao: Vec<Basic> = ap.to_vec();
            for a in pap_m.iter().take(u) {
                remove_first(&mut ao, a);
            }
            let lu: Basic = li[li.len() - 1].clone();
            let di: Vec<Basic> = li
                .iter()
                .zip(ki.iter())
                .map(|(l, k)| s2(l, k))
                .collect::<PolyResult<_>>()?;
            let s: Basic = dummy("s");
            let mut integrand: Basic = pw(z, &s)?;
            for b in bm {
                let mut b: Basic = b.clone();
                if b.is_number_node() && is_zero_number(&crate::core::modulo::modulo(&b, &int(1))?) {
                    b = round_to_integer(&b)?;
                }
                integrand = m2(&integrand, &gamma(&s2(&b, &s)?)?)?;
            }
            for a in an {
                integrand = m2(&integrand, &gamma(&a2(&s2(&int(1), a)?, &s)?)?)?;
            }
            for b in bq {
                integrand = d2(&integrand, &gamma(&a2(&s2(&int(1), b)?, &s)?)?)?;
            }
            for a in ap {
                integrand = d2(&integrand, &gamma(&s2(a, &s)?)?)?;
            }
            integrand = crate::functions::expand_func::expand_func(&integrand, true)?;
            let lu_int: i64 = as_int(&round_to_integer(&lu)?)?;
            for r in 0..lu_int {
                let mut resid: Basic = crate::series::residues::residue(&integrand, &s, &a2(&b_, &int(r))?)?;
                resid = apply_operators(&resid, ops, &|e| m2(z, &diff(e, z)?))?;
                res = s2(&res, &resid)?;
            }
            let au: Basic = a2(&b_, &lu)?;
            let k: Basic = polar(ao.len() as i64 + bo.len() as i64 + 1)?;
            let harg: Basic = m2(&k, zfinal)?;
            let premult: Basic = pw(&d2(t, &k)?, &au)?;
            let mut nap: Vec<Basic> = an
                .iter()
                .chain(ap.iter())
                .map(|a| s2(&a2(&int(1), &au)?, a))
                .collect::<PolyResult<_>>()?;
            nap.push(int(1));
            let nbq: Vec<Basic> = bm
                .iter()
                .chain(bq.iter())
                .map(|b| s2(&a2(&int(1), &au)?, b))
                .collect::<PolyResult<_>>()?;
            let hyp: Basic = hyperexpand_hyper(
                &HyperFunction::new(&nap, &nbq)?,
                &harg,
                ops,
                Some(t),
                &premult,
                &au,
                Rewrite::Keep,
            )?;
            let mut c: Basic = d2(&pw(&int(-1), &lu)?, &factorial(&lu)?)?;
            for i in 0..u {
                c = d2(
                    &m2(&c, &pw(&int(-1), &di[i])?)?,
                    &rf(&a2(&s2(&lu, &li[i])?, &int(1))?, &di[i])?,
                )?;
            }
            for a in an {
                c = m2(&c, &gamma(&a2(&s2(&int(1), a)?, &au)?)?)?;
            }
            for b in &bo {
                c = m2(&c, &gamma(&s2(b, &au)?)?)?;
            }
            for a in &ao {
                c = d2(&c, &gamma(&s2(a, &au)?)?)?;
            }
            for b in bq {
                c = d2(&c, &gamma(&a2(&s2(&int(1), b)?, &au)?)?)?;
            }
            res = a2(&res, &m2(&c, &hyp)?)?;
        }
    }
    Ok((res, cond))
}

fn round_to_integer(b: &Basic) -> R {
    let value = crate::core::evalf::numeric_value(b, 64).ok_or_else(|| value_error(format!("no value of {}", b)))?;
    let (re, _) = crate::core::evalf::parts(&value);
    Ok(int(re.to_f64().value().round() as i64))
}

fn count_nodes(e: &Basic, head: &dyn Fn(&Basic) -> bool) -> i64 {
    e.preorder().iter().filter(|n| head(n)).count() as i64
}

fn has_infinity(e: &Basic) -> bool {
    e.preorder().iter().any(|n| {
        matches!(
            n.head(),
            Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity | Head::NaN
        )
    })
}

fn meijergexpand(func0: &GFunction, z0: &Basic, allow_hyper: bool, rewrite: Rewrite, place: Option<&Basic>) -> R {
    let collection = meijer_collection()?;
    let rewrite: Option<Rewrite> = if rewrite == Rewrite::Default {
        None
    } else {
        Some(rewrite)
    };
    let z: Basic = dummy("z");
    let (func_, reduce_ops) = reduce_order_meijer(func0)?;
    let mut ops: Vec<Operator> = reduce_ops;
    if let Some(f) = collection.lookup_origin(&func_)? {
        ops.extend(devise_plan_meijer(&f.func, &func_, &z)?);
        let to_z: Repl = vec![(f.z.clone(), z.clone())];
        let c: ExprMatrix =
            apply_operators_matrix(&f.c.subs(&to_z)?, &ops, &derivative_operator(&f.m.subs(&to_z)?, &z))?;
        let simp = make_simp(&z);
        let c: ExprMatrix = c.applyfunc(&|e| simp(e))?;
        let r: Basic = c.matmul(&f.b.subs(&to_z)?)?.at(0, 0).clone();
        return powdenest_polar(&subs(&r, &z, z0)?);
    }
    let t: Basic = dummy("t");
    let (slater1, cond1) = do_slater(&func_.an, &func_.bm, &func_.ap, &func_.bq, &z, z0, &ops, &t)?;
    let tr = |params: &[Basic]| -> PolyResult<Vec<Basic>> { params.iter().map(|x| s2(&int(1), x)).collect() };
    let inverse_t: Basic = d2(&int(1), &t)?;
    let x: Basic = x_();
    let minus_x: Vec<(Basic, Basic)> = vec![(x.clone(), neg(&x)?)];
    let mut ops2: Vec<Operator> = Vec::new();
    for op in &ops {
        let e: Basic = replaced(&subs(&from_coeffs(&op.coeffs, &x)?, &z, &inverse_t)?, &minus_x)?;
        ops2.push(op_of(&e)?);
    }
    let inverse_z0: Basic = d2(&int(1), z0)?;
    let (slater2, cond2) = do_slater(
        &tr(&func_.bm)?,
        &tr(&func_.an)?,
        &tr(&func_.bq)?,
        &tr(&func_.ap)?,
        &t,
        &inverse_z0,
        &ops2,
        &t,
    )?;
    let slater1: Basic = powdenest_polar(&subs(&slater1, &z, z0)?)?;
    let slater2: Basic = powdenest_polar(&subs(&slater2, &t, &inverse_z0)?)?;
    let mut cond2: Condition = cond2.subs(&t, &d2(&int(1), &z)?)?;
    let mut cond1: Condition = cond1;
    let full_ap: usize = func_.an.len() + func_.ap.len();
    let full_bq: usize = func_.bm.len() + func_.bq.len();
    let delta: Basic = s2(
        &int((func_.bm.len() + func_.an.len()) as i64),
        &Basic::from_rational(crate::numbers::rat((full_ap + full_bq) as i64, 2)),
    )?;
    let polar = |e: &Basic| crate::functions::elementary::polar::polar_lift(e);
    if gt0(&delta)? || (delta == int(0) && full_ap == full_bq && polar(z0)? == polar(&int(1))?) {
        if !cond1.is_false() {
            cond1 = Condition::Decided(true);
        }
        if !cond2.is_false() {
            cond2 = Condition::Decided(true);
        }
    }
    let choose = |c: &Condition| {
        rewrite.unwrap_or(if c.is_true() {
            Rewrite::NonRep
        } else {
            Rewrite::NonRepSmall
        })
    };
    let slater1: Basic = rewrite_with(&slater1, choose(&cond1))?;
    let slater2: Basic = rewrite_with(&slater2, choose(&cond2))?;
    if !cond1.is_false() && !cond2.is_false() {
        if place.is_some_and(|p| *p == int(0)) {
            cond2 = Condition::Decided(false);
        }
        if place.is_some_and(|p| matches!(p.head(), Head::ComplexInfinity)) {
            cond1 = Condition::Decided(false);
        }
    }
    let cond1: Condition = cond1.subs(&z, z0)?;
    let cond2: Condition = cond2.subs(&z, z0)?;
    let weight = |e: &Basic, cond: &Condition| -> PolyResult<(i64, i64, i64)> {
        let mut c0: i64 = match cond {
            Condition::Decided(true) => 0,
            Condition::Decided(false) => 1,
            Condition::Open(_) => 2,
        };
        if has_infinity(e) {
            c0 = 3;
        }
        Ok((
            c0,
            count_nodes(e, &|n| matches!(n.head(), Head::Hyper)),
            crate::core::function::count_ops(e)?,
        ))
    };
    let (w1, w2) = (weight(&slater1, &cond1)?, weight(&slater2, &cond2)?);
    let best = w1.min(w2);
    if best.0 < 0 || (best.0 == 0 && best.1 <= 1) {
        return Ok(if w1 < w2 { slater1 } else { slater2 });
    }
    let whole = || -> R {
        crate::functions::elementary::piecewise::piecewise(&[
            (slater1.clone(), cond1.as_basic()),
            (slater2.clone(), cond2.as_basic()),
            (func0.call(z0)?, crate::core::logic::true_()),
        ])
    };
    if w1.0.max(w2.0) <= 1 && w1.1.max(w2.1) <= 1 {
        return whole();
    }
    let r: Basic = whole()?;
    if !r.preorder().iter().any(|n| matches!(n.head(), Head::Hyper)) || allow_hyper {
        return Ok(r);
    }
    func0.call(z0)
}

/// `hyper` and `meijerg` expanded in named special functions
/// (`hyperexpand`).
pub fn hyperexpand(f: &Basic, allow_hyper: bool, rewrite: Rewrite, place: Option<&Basic>) -> R {
    let e: Basic = crate::core::function::replace_where(f, &|n| matches!(n.head(), Head::Hyper), &|n| {
        let (ap, bq, z) = hyper_parts(n);
        hyperexpand_hyper(&HyperFunction::new(&ap, &bq)?, &z, &[], None, &int(1), &int(0), rewrite)
    })?;
    crate::core::function::replace_where(&e, &|n| matches!(n.head(), Head::MeijerG), &|n| {
        let (ap, bq, z) = (&n.args()[0], &n.args()[1], &n.args()[2]);
        let (an, aother) = (ap.args()[0].args().to_vec(), ap.args()[1].args().to_vec());
        let (bm, bother) = (bq.args()[0].args().to_vec(), bq.args()[1].args().to_vec());
        let r: Basic = meijergexpand(
            &GFunction::new(&an, &aother, &bm, &bother)?,
            z,
            allow_hyper,
            rewrite,
            place,
        )?;
        if !has_infinity(&r) {
            return Ok(r);
        }
        crate::functions::special::hyper::meijerg(&an, &aother, &bm, &bother, z)
    })
}
