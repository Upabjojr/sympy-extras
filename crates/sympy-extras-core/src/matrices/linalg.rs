//! Port of `sympy_extras/matrices/linalg.py`: exact linear algebra of
//! matrices of expressions. The entries are lifted to fractions over
//! `ZZ` or over `ZZ[k_1, ..., k_n]` in their kernels (the non-rational
//! atoms), or, with `I` and square roots of rationals among the kernels,
//! to a multiquadratic field over `QQ(parameters)`; the computation is
//! fraction-free in the domain (`DomainMatrix`) and the pivots are
//! certified nonzero by the structure theorem when the kernels are not
//! all symbols.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;

use crate::core::basic::{Basic, Head};
use crate::error::{value_error, zero_division, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::domainmatrix::DomainMatrix;
use crate::numbers::{denom, numer, Int, Rat};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::integerring::{IntegerRing, ZZ};
use crate::polys::domains::multiquadratic::{MultiquadraticElement, MultiquadraticField};
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;
use crate::polys::rings::PolyElement;

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn integer(n: &Int) -> Basic {
    Basic::from_rational(Rat::from(n.clone()))
}
fn add_all(v: &[Basic]) -> R {
    crate::core::add::add(v)
}
fn mul_all(v: &[Basic]) -> R {
    crate::core::mul::mul(v)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}

pub(crate) fn integer_exponent(e: &Basic) -> Option<Int> {
    if e.is_pow() && e.args()[1].is_integer_value() {
        Some(numer(e.args()[1].as_rational().expect("an integer")))
    } else {
        None
    }
}

fn collect_kernels(e: &Basic, found: &mut Vec<Basic>) {
    if e.is_rational_class() {
        return;
    }
    if e.is_add() || e.is_mul() {
        for a in e.args() {
            collect_kernels(a, found);
        }
        return;
    }
    if integer_exponent(e).is_some() {
        collect_kernels(&e.args()[0], found);
        return;
    }
    if !found.contains(e) {
        found.push(e.clone());
    }
}

/// The lifting of expressions to a domain (`_Lift`).
pub(crate) trait Lift {
    type K: Domain + Clone;
    fn domain(&self) -> &Self::K;
    fn exact(&self) -> bool;
    fn atom(&self, e: &Basic) -> PolyResult<<Self::K as Domain>::Elem>;
    fn expr(&self, a: &<Self::K as Domain>::Elem) -> R;
    fn cancel(
        &self,
        n: &<Self::K as Domain>::Elem,
        d: &<Self::K as Domain>::Elem,
    ) -> PolyResult<(<Self::K as Domain>::Elem, <Self::K as Domain>::Elem)>;

    /// `e` as a fraction `(n, d)` over the domain (`fraction`).
    #[allow(clippy::type_complexity)]
    fn fraction(&self, e: &Basic) -> PolyResult<(<Self::K as Domain>::Elem, <Self::K as Domain>::Elem)> {
        let k: &Self::K = self.domain();
        if let Some(r) = e.as_rational() {
            return Ok((k.from_integer(&numer(r)), k.from_integer(&denom(r))));
        }
        if e.is_add() {
            let (mut n, mut d) = (k.zero(), k.one());
            for a in e.args() {
                let (p, q) = self.fraction(a)?;
                if q == d {
                    n = k.add(&n, &p);
                } else if k.is_one(&q) {
                    n = k.add(&n, &k.mul(&p, &d));
                } else if k.is_one(&d) {
                    n = k.add(&k.mul(&n, &q), &p);
                    d = q;
                } else {
                    let common = k.gcd(&d, &q);
                    if k.is_one(&common) {
                        n = k.add(&k.mul(&n, &q), &k.mul(&p, &d));
                        d = k.mul(&d, &q);
                    } else {
                        let cofactor = k.exquo(&q, &common)?;
                        n = k.add(&k.mul(&n, &cofactor), &k.mul(&p, &k.exquo(&d, &common)?));
                        d = k.mul(&d, &cofactor);
                    }
                }
            }
            return Ok((n, d));
        }
        if e.is_mul() {
            let (mut n, mut d) = (k.one(), k.one());
            for a in e.args() {
                let (p, q) = self.fraction(a)?;
                n = k.mul(&n, &p);
                d = k.mul(&d, &q);
            }
            return Ok((n, d));
        }
        if let Some(exponent) = integer_exponent(e) {
            let (mut p, mut q) = self.fraction(&e.args()[0])?;
            let mut kk: Int = exponent;
            if kk < Int::from(0) {
                if k.is_zero(&p) {
                    return Err(zero_division(&format!("division by zero in {}", e)));
                }
                std::mem::swap(&mut p, &mut q);
                kk = -kk;
            }
            let kk: u64 = u64::try_from(&kk).map_err(|_| value_error(format!("the exponent of {} is too large", e)))?;
            return Ok((k.pow(&p, kk), k.pow(&q, kk)));
        }
        Ok((self.atom(e)?, k.one()))
    }

    /// `n/d` as an expression in lowest terms (`ratio`).
    fn ratio(&self, n: &<Self::K as Domain>::Elem, d: &<Self::K as Domain>::Elem) -> R {
        let (p, q) = self.cancel(n, d)?;
        if self.domain().is_one(&q) {
            return self.expr(&p);
        }
        crate::core::mul::div(&self.expr(&p)?, &self.expr(&q)?)
    }
}

pub(crate) struct IntegerLift;

impl Lift for IntegerLift {
    type K = IntegerRing;
    fn domain(&self) -> &IntegerRing {
        &ZZ
    }
    fn exact(&self) -> bool {
        true
    }
    fn atom(&self, e: &Basic) -> PolyResult<Int> {
        Err(value_error(format!("{} is not a rational number", e)))
    }
    fn expr(&self, a: &Int) -> R {
        Ok(integer(a))
    }
    fn cancel(&self, n: &Int, d: &Int) -> PolyResult<(Int, Int)> {
        let mut g: Int = ZZ.gcd(n, d);
        if *d < Int::from(0) {
            g = -g;
        }
        Ok((ZZ.exquo(n, &g)?, ZZ.exquo(d, &g)?))
    }
}

pub(crate) type ZPoly = PolyElement<IntegerRing>;

pub(crate) struct PolynomialLift {
    pub(crate) ring: SparsePolynomialRing<IntegerRing>,
    pub(crate) kernels: Vec<Basic>,
    index: HashMap<Basic, ZPoly>,
    pub(crate) exact: bool,
}

impl PolynomialLift {
    pub(crate) fn new(kernels: &[Basic]) -> PolynomialLift {
        let names: Vec<String> = (0..kernels.len()).map(|i| format!("_k{}", i)).collect();
        let ring: SparsePolynomialRing<IntegerRing> = SparsePolynomialRing::sympy(ZZ, names);
        let gens: Vec<ZPoly> = ring.ring.gens();
        let index: HashMap<Basic, ZPoly> = kernels.iter().cloned().zip(gens).collect();
        PolynomialLift {
            ring,
            kernels: kernels.to_vec(),
            index,
            exact: kernels.iter().all(|k| k.is_symbol()),
        }
    }
}

impl Lift for PolynomialLift {
    type K = SparsePolynomialRing<IntegerRing>;
    fn domain(&self) -> &SparsePolynomialRing<IntegerRing> {
        &self.ring
    }
    fn exact(&self) -> bool {
        self.exact
    }
    fn atom(&self, e: &Basic) -> PolyResult<ZPoly> {
        self.index
            .get(e)
            .cloned()
            .ok_or_else(|| value_error(format!("KeyError: {}", e)))
    }
    fn expr(&self, a: &ZPoly) -> R {
        let mut parts: Vec<Basic> = Vec::new();
        for (monomial, coefficient) in &a.terms {
            let mut factors: Vec<Basic> = vec![integer(coefficient)];
            for (kernel, exponent) in self.kernels.iter().zip(monomial) {
                if *exponent == 1 {
                    factors.push(kernel.clone());
                } else if *exponent != 0 {
                    factors.push(pw(kernel, &int(i64::from(*exponent)))?);
                }
            }
            parts.push(mul_all(&factors)?);
        }
        if parts.is_empty() {
            Ok(int(0))
        } else {
            add_all(&parts)
        }
    }
    fn cancel(&self, n: &ZPoly, d: &ZPoly) -> PolyResult<(ZPoly, ZPoly)> {
        Ok(self.ring.cancel(n, d))
    }
}

pub(crate) fn kernels(matrices: &[&ExprMatrix]) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for m in matrices {
        for e in &m.entries {
            collect_kernels(e, &mut found);
        }
    }
    found
}

type Scaled<L> = (DomainMatrix<<L as Lift>::K>, Vec<<<L as Lift>::K as Domain>::Elem>);

fn scaled<L: Lift>(lift: &L, m: &ExprMatrix, common: bool) -> PolyResult<Scaled<L>> {
    let k: &L::K = lift.domain();
    let mut fractions: Vec<Vec<_>> = Vec::new();
    for i in 0..m.rows {
        let mut line = Vec::new();
        for e in m.row(i) {
            let (n, d) = lift.fraction(e)?;
            line.push(lift.cancel(&n, &d)?);
        }
        fractions.push(line);
    }
    let mut dens = Vec::new();
    for line in &fractions {
        let mut d = k.one();
        for (_, q) in line {
            if !k.is_one(q) {
                d = k.lcm(&d, q);
            }
        }
        dens.push(d);
    }
    if common {
        let mut total = k.one();
        for d in &dens {
            if !k.is_one(d) {
                total = k.lcm(&total, d);
            }
        }
        dens = vec![total; m.rows];
    }
    let mut rows = Vec::new();
    for (line, d) in fractions.iter().zip(dens.iter()) {
        let mut row = Vec::new();
        for (p, q) in line {
            row.push(if q == d { p.clone() } else { k.mul(p, &k.exquo(d, q)?) });
        }
        rows.push(row);
    }
    Ok((DomainMatrix::new(rows, k.clone(), Some(m.cols))?, dens))
}

fn certify<L: Lift>(lift: &L, pivots: &[<L::K as Domain>::Elem]) -> PolyResult<()> {
    if lift.exact() {
        return Ok(());
    }
    for value in pivots {
        let e: Basic = lift.expr(value)?;
        if e.is_rational_class() && e != int(0) {
            continue;
        }
        if crate::simplify::structure::is_zero(&e, &[])? != Some(false) {
            return Err(PolyError::NotImplemented(format!(
                "the pivot {} is not proved nonzero: simplify the entries of the matrix first",
                e
            )));
        }
    }
    Ok(())
}

pub(crate) fn square(m: &ExprMatrix) -> PolyResult<usize> {
    if m.rows != m.cols {
        return Err(value_error(format!("a {} x {} matrix is not square", m.rows, m.cols)));
    }
    Ok(m.rows)
}

fn det_with<L: Lift>(lift: &L, m: &ExprMatrix) -> R {
    let k: &L::K = lift.domain();
    let (s, dens) = scaled(lift, m, false)?;
    let mut denominator = k.one();
    for d in &dens {
        denominator = k.mul(&denominator, d);
    }
    lift.ratio(&s.det()?, &denominator)
}

fn charpoly_with<L: Lift>(lift: &L, m: &ExprMatrix) -> PolyResult<Vec<Basic>> {
    let k: &L::K = lift.domain();
    let (s, dens) = scaled(lift, m, true)?;
    let d = dens.first().cloned().unwrap_or_else(|| k.one());
    let mut coefficients: Vec<Basic> = Vec::new();
    let mut power = k.one();
    for c in s.charpoly()? {
        coefficients.push(lift.ratio(&c, &power)?);
        power = k.mul(&power, &d);
    }
    Ok(coefficients)
}

fn adjugate_with<L: Lift>(lift: &L, m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let k: &L::K = lift.domain();
    let n: usize = m.rows;
    let (s, dens) = scaled(lift, m, true)?;
    let d = if n > 0 {
        k.pow(&dens[0], (n - 1) as u64)
    } else {
        k.one()
    };
    let mut entries: Vec<Basic> = Vec::new();
    for row in &s.adjugate()?.rows {
        for a in row {
            entries.push(lift.ratio(a, &d)?);
        }
    }
    ExprMatrix::new(n, n, entries)
}

fn inverse_with<L: Lift>(lift: &L, m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let k: &L::K = lift.domain();
    let n: usize = m.rows;
    let (s, dens) = scaled(lift, m, false)?;
    let (reduced, den, pivots, values) = s.hstack(&DomainMatrix::eye(n, k.clone()))?.elimination()?;
    if pivots != (0..n).collect::<Vec<usize>>() {
        return Err(value_error("the matrix is singular".to_string()));
    }
    certify(lift, &values)?;
    let mut entries: Vec<Basic> = Vec::new();
    for row in &reduced.rows {
        for (j, a) in row[n..].iter().enumerate() {
            entries.push(lift.ratio(&k.mul(a, &dens[j]), &den)?);
        }
    }
    ExprMatrix::new(n, n, entries)
}

fn rref_with<L: Lift>(lift: &L, m: &ExprMatrix) -> PolyResult<(ExprMatrix, Vec<usize>)> {
    let (s, _) = scaled(lift, m, false)?;
    let (reduced, den, pivots, values) = s.elimination()?;
    certify(lift, &values)?;
    let mut entries: Vec<Basic> = Vec::new();
    for row in &reduced.rows {
        for a in row {
            entries.push(lift.ratio(a, &den)?);
        }
    }
    Ok((ExprMatrix::new(m.rows, m.cols, entries)?, pivots))
}

fn rank_with<L: Lift>(lift: &L, m: &ExprMatrix) -> PolyResult<usize> {
    let (s, _) = scaled(lift, m, false)?;
    let (_, _, pivots, values) = s.elimination()?;
    certify(lift, &values)?;
    Ok(pivots.len())
}

fn nullspace_vectors<L: Lift>(
    lift: &L,
    reduced: &DomainMatrix<L::K>,
    den: &<L::K as Domain>::Elem,
    pivots: &[usize],
    ncols: usize,
) -> PolyResult<Vec<ExprMatrix>> {
    let mut basis: Vec<ExprMatrix> = Vec::new();
    for free in 0..ncols {
        if pivots.contains(&free) {
            continue;
        }
        let mut vector: Vec<Basic> = vec![int(0); ncols];
        vector[free] = int(1);
        for (i, column) in pivots.iter().enumerate() {
            vector[*column] = lift.ratio(&lift.domain().neg(&reduced.rows[i][free]), den)?;
        }
        basis.push(ExprMatrix::column(vector));
    }
    Ok(basis)
}

fn nullspace_with<L: Lift>(lift: &L, m: &ExprMatrix) -> PolyResult<Vec<ExprMatrix>> {
    let (s, _) = scaled(lift, m, false)?;
    let (reduced, den, pivots, values) = s.elimination()?;
    certify(lift, &values)?;
    nullspace_vectors(lift, &reduced, &den, &pivots, m.cols)
}

/// A particular solution and a basis of the solutions of the homogeneous
/// system (`LinearSolution`).
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSolution {
    pub particular: ExprMatrix,
    pub nullspace: Vec<ExprMatrix>,
}

impl LinearSolution {
    pub fn is_unique(&self) -> bool {
        self.nullspace.is_empty()
    }

    pub fn repr(&self) -> String {
        let vectors: Vec<String> = self.nullspace.iter().map(|v| v.repr()).collect();
        format!("LinearSolution({}, [{}])", self.particular.repr(), vectors.join(", "))
    }
}

fn augmented(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let mut result: ExprMatrix = a.clone();
    for j in 0..b.cols {
        let column: Vec<Basic> = (0..b.rows).map(|i| b.at(i, j).clone()).collect();
        result = result.col_insert(result.cols, &ExprMatrix::column(column))?;
    }
    Ok(result)
}

fn linear_solve_with<L: Lift>(lift: &L, a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<Option<LinearSolution>> {
    let (s, _) = scaled(lift, &augmented(a, b)?, false)?;
    let (reduced, den, pivots, values) = s.elimination()?;
    certify(lift, &values)?;
    if pivots.iter().any(|c| *c >= a.cols) {
        return Ok(None);
    }
    let mut solution: Vec<Vec<Basic>> = vec![vec![int(0); b.cols]; a.cols];
    for (i, column) in pivots.iter().enumerate() {
        solution[*column] = reduced.rows[i][a.cols..]
            .iter()
            .map(|x| lift.ratio(x, &den))
            .collect::<PolyResult<_>>()?;
    }
    let particular: ExprMatrix = ExprMatrix::new(a.cols, b.cols, solution.into_iter().flatten().collect())?;
    Ok(Some(LinearSolution {
        particular,
        nullspace: nullspace_vectors(lift, &reduced, &den, &pivots, a.cols)?,
    }))
}

fn radicand(e: &Basic) -> Option<Rat> {
    if e.is_pow() {
        let (base, exponent) = (&e.args()[0], &e.args()[1]);
        if let (Some(b), Some(x)) = (base.as_rational(), exponent.as_rational()) {
            if *x == Rat::from(Int::from(1)) / Rat::from(Int::from(2)) && numer(b) > Int::from(0) {
                return Some(b.clone());
            }
        }
    }
    None
}

pub(crate) fn is_radical(e: &Basic) -> bool {
    matches!(e.head(), Head::ImaginaryUnit) || radicand(e).is_some()
}

fn squarefree(r: &Rat) -> (Rat, Vec<Int>) {
    let mut outside: Rat = Rat::from(Int::from(1)) / Rat::from(denom(r));
    let mut primes: Vec<Int> = Vec::new();
    for (prime, exponent) in crate::ntheory::factor_::factorint(&(numer(r) * denom(r)), None) {
        let mut p: Int = Int::from(1);
        for _ in 0..exponent / 2 {
            p *= prime.clone();
        }
        outside *= Rat::from(p);
        if exponent % 2 == 1 {
            primes.push(prime);
        }
    }
    (outside, primes)
}

pub(crate) type Ground = SparseFractionField<IntegerRing>;
pub(crate) type Radical = MultiquadraticElement<SparseFracElement<IntegerRing>>;

/// The lifting to `QQ(parameters)(I, sqrt(p), ...)` (`_ComplexLift`).
pub(crate) struct ComplexLift {
    pub(crate) lift: PolynomialLift,
    pub(crate) ground: Ground,
    pub(crate) field: MultiquadraticField<Ground>,
    pub(crate) position: Vec<(i64, usize)>,
}

impl ComplexLift {
    fn new(parameters: &[Basic], radicals: &[Basic]) -> PolyResult<ComplexLift> {
        let spare: Vec<Basic> = vec![crate::core::symbol::Symbol::new("_spare", &[]).0];
        let lift: PolynomialLift = PolynomialLift::new(if parameters.is_empty() { &spare } else { parameters });
        let ground: Ground = SparseFractionField::new(lift.ring.clone());
        let mut radicands: Vec<i64> = Vec::new();
        for kernel in radicals {
            match radicand(kernel) {
                None => radicands.push(-1),
                Some(r) => {
                    for p in squarefree(&r).1 {
                        radicands.push(
                            i64::try_from(&p).map_err(|_| value_error(format!("the radicand {} is too large", p)))?,
                        );
                    }
                }
            }
        }
        radicands.sort();
        radicands.dedup();
        let field = MultiquadraticField::new(ground.clone(), radicands.clone())?;
        let position: Vec<(i64, usize)> = radicands.iter().enumerate().map(|(i, d)| (*d, i)).collect();
        Ok(ComplexLift {
            lift,
            ground,
            field,
            position,
        })
    }

    fn index(&self, d: i64) -> usize {
        self.position
            .iter()
            .find(|(e, _)| *e == d)
            .expect("a radicand of the field")
            .1
    }

    pub(crate) fn element(&self, e: &Basic) -> PolyResult<Radical> {
        let f = &self.field;
        if matches!(e.head(), Head::ImaginaryUnit) {
            return Ok(f.generator(self.index(-1)));
        }
        if let Some(r) = radicand(e) {
            let (outside, primes) = squarefree(&r);
            let ring = &self.lift.ring;
            let mut value: Radical = f.from_ground(self.ground.new_element(
                &ring.from_integer(&numer(&outside)),
                &ring.from_integer(&denom(&outside)),
            )?);
            for prime in primes {
                let p: i64 = i64::try_from(&prime).expect("a small prime");
                value = f.mul(&value, &f.generator(self.index(p)));
            }
            return Ok(value);
        }
        if e.is_rational_class() || e.is_symbol() {
            let (n, d) = self.lift.fraction(e)?;
            return Ok(f.from_ground(self.ground.new_element(&n, &d)?));
        }
        if e.is_add() {
            let mut total: Radical = f.zero();
            for a in e.args() {
                total = f.add(&total, &self.element(a)?);
            }
            return Ok(total);
        }
        if e.is_mul() {
            let mut product: Radical = f.one();
            for a in e.args() {
                product = f.mul(&product, &self.element(a)?);
            }
            return Ok(product);
        }
        if let Some(k) = integer_exponent(e) {
            let k: i64 = i64::try_from(&k).map_err(|_| value_error(format!("the exponent of {} is too large", e)))?;
            return f.pow_signed(&self.element(&e.args()[0])?, k);
        }
        Err(value_error(format!(
            "{} is not a rational function of symbols, I and square roots of rationals",
            e
        )))
    }

    pub(crate) fn expr(&self, a: &Radical) -> R {
        let mut terms: Vec<Basic> = Vec::new();
        for (mask, c) in a {
            let coefficient: Basic = self.lift.ratio(&c.numer, &c.denom)?;
            let mut factors: Vec<Basic> = vec![coefficient];
            let mut product: Int = Int::from(1);
            for (d, i) in &self.position {
                if mask >> i & 1 == 1 {
                    if *d == -1 {
                        factors.push(Basic::atom(Head::ImaginaryUnit));
                    } else {
                        product *= Int::from(*d);
                    }
                }
            }
            if product != Int::from(1) {
                factors.push(pw(
                    &integer(&product),
                    &Basic::from_rational(Rat::from(Int::from(1)) / Rat::from(Int::from(2))),
                )?);
            }
            terms.push(mul_all(&factors)?);
        }
        if terms.is_empty() {
            Ok(int(0))
        } else {
            add_all(&terms)
        }
    }

    fn matrix(&self, m: &ExprMatrix) -> PolyResult<DomainMatrix<MultiquadraticField<Ground>>> {
        let mut rows: Vec<Vec<Radical>> = Vec::new();
        for i in 0..m.rows {
            rows.push(m.row(i).iter().map(|e| self.element(e)).collect::<PolyResult<_>>()?);
        }
        DomainMatrix::new(rows, self.field.clone(), Some(m.cols))
    }

    fn expr_matrix(&self, d: &DomainMatrix<MultiquadraticField<Ground>>) -> PolyResult<ExprMatrix> {
        let mut entries: Vec<Basic> = Vec::new();
        for row in &d.rows {
            for a in row {
                entries.push(self.expr(a)?);
            }
        }
        ExprMatrix::new(d.nrows, d.ncols, entries)
    }

    fn nullspace(
        &self,
        d: &DomainMatrix<MultiquadraticField<Ground>>,
        pivots: &[usize],
        ncols: usize,
    ) -> PolyResult<Vec<ExprMatrix>> {
        let mut basis: Vec<ExprMatrix> = Vec::new();
        for free in 0..ncols {
            if pivots.contains(&free) {
                continue;
            }
            let mut vector: Vec<Basic> = vec![int(0); ncols];
            vector[free] = int(1);
            for (i, column) in pivots.iter().enumerate() {
                vector[*column] = self.expr(&self.field.neg(&d.rows[i][free]))?;
            }
            basis.push(ExprMatrix::column(vector));
        }
        Ok(basis)
    }
}

pub(crate) fn complex_lift(kernels: &[Basic]) -> PolyResult<Option<ComplexLift>> {
    let radicals: Vec<Basic> = kernels.iter().filter(|k| is_radical(k)).cloned().collect();
    if radicals.is_empty() {
        return Ok(None);
    }
    let parameters: Vec<Basic> = kernels.iter().filter(|k| !is_radical(k)).cloned().collect();
    if !parameters.iter().all(|k| k.is_symbol()) {
        return Ok(None);
    }
    Ok(Some(ComplexLift::new(&parameters, &radicals)?))
}

/// `e` as a quotient of expanded polynomials in its kernels (`normal`).
pub fn normal(e: &Basic) -> R {
    let mut found: Vec<Basic> = Vec::new();
    collect_kernels(e, &mut found);
    if found.is_empty() {
        let (n, d) = IntegerLift.fraction(e)?;
        return IntegerLift.ratio(&n, &d);
    }
    let lift: PolynomialLift = PolynomialLift::new(&found);
    let (n, d) = lift.fraction(e)?;
    lift.ratio(&n, &d)
}

/// The determinant (`det`).
pub fn det(m: &ExprMatrix) -> R {
    square(m)?;
    let ks: Vec<Basic> = kernels(&[m]);
    if let Some(g) = complex_lift(&ks)? {
        return g.expr(&g.matrix(m)?.det()?);
    }
    if ks.is_empty() {
        det_with(&IntegerLift, m)
    } else {
        det_with(&PolynomialLift::new(&ks), m)
    }
}

/// The coefficients of `det(x*I - M)`, the leading one first
/// (`charpoly_coefficients`).
pub fn charpoly_coefficients(m: &ExprMatrix) -> PolyResult<Vec<Basic>> {
    square(m)?;
    let ks: Vec<Basic> = kernels(&[m]);
    if let Some(g) = complex_lift(&ks)? {
        return g.matrix(m)?.charpoly()?.iter().map(|c| g.expr(c)).collect();
    }
    if ks.is_empty() {
        charpoly_with(&IntegerLift, m)
    } else {
        charpoly_with(&PolynomialLift::new(&ks), m)
    }
}

/// `det(x*I - M)` (`charpoly`).
pub fn charpoly(m: &ExprMatrix, x: &Basic) -> R {
    let coefficients: Vec<Basic> = charpoly_coefficients(m)?;
    let n: i64 = coefficients.len() as i64 - 1;
    let mut terms: Vec<Basic> = Vec::new();
    for (k, c) in coefficients.iter().enumerate() {
        terms.push(mul_all(&[c.clone(), pw(x, &int(n - k as i64))?])?);
    }
    add_all(&terms)
}

/// The adjugate (`adjugate`).
pub fn adjugate(m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    square(m)?;
    let ks: Vec<Basic> = kernels(&[m]);
    if let Some(g) = complex_lift(&ks)? {
        return g.expr_matrix(&g.matrix(m)?.adjugate()?);
    }
    if ks.is_empty() {
        adjugate_with(&IntegerLift, m)
    } else {
        adjugate_with(&PolynomialLift::new(&ks), m)
    }
}

/// The inverse (`inverse`); `ValueError` for a singular matrix.
pub fn inverse(m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    square(m)?;
    let ks: Vec<Basic> = kernels(&[m]);
    if let Some(g) = complex_lift(&ks)? {
        return g.expr_matrix(&g.matrix(m)?.inv()?);
    }
    if ks.is_empty() {
        inverse_with(&IntegerLift, m)
    } else {
        inverse_with(&PolynomialLift::new(&ks), m)
    }
}

/// The rank (`rank`).
pub fn rank(m: &ExprMatrix) -> PolyResult<usize> {
    let ks: Vec<Basic> = kernels(&[m]);
    if let Some(g) = complex_lift(&ks)? {
        return g.matrix(m)?.rank();
    }
    if ks.is_empty() {
        rank_with(&IntegerLift, m)
    } else {
        rank_with(&PolynomialLift::new(&ks), m)
    }
}

/// The reduced row echelon form and its pivot columns (`rref`).
pub fn rref(m: &ExprMatrix) -> PolyResult<(ExprMatrix, Vec<usize>)> {
    let ks: Vec<Basic> = kernels(&[m]);
    if let Some(g) = complex_lift(&ks)? {
        let (echelon, columns) = g.matrix(m)?.rref()?;
        return Ok((g.expr_matrix(&echelon)?, columns));
    }
    if ks.is_empty() {
        rref_with(&IntegerLift, m)
    } else {
        rref_with(&PolynomialLift::new(&ks), m)
    }
}

/// A basis of the kernel (`nullspace`).
pub fn nullspace(m: &ExprMatrix) -> PolyResult<Vec<ExprMatrix>> {
    let ks: Vec<Basic> = kernels(&[m]);
    if let Some(g) = complex_lift(&ks)? {
        let (echelon, columns) = g.matrix(m)?.rref()?;
        return g.nullspace(&echelon, &columns, m.cols);
    }
    if ks.is_empty() {
        nullspace_with(&IntegerLift, m)
    } else {
        nullspace_with(&PolynomialLift::new(&ks), m)
    }
}

/// The solutions of `A*X = B` (`linear_solve`); `None` when there is none.
pub fn linear_solve(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<Option<LinearSolution>> {
    if a.rows != b.rows {
        return Err(value_error(format!(
            "{} equations and {} right-hand sides",
            a.rows, b.rows
        )));
    }
    let ks: Vec<Basic> = kernels(&[a, b]);
    if let Some(g) = complex_lift(&ks)? {
        let (echelon, columns) = g.matrix(a)?.hstack(&g.matrix(b)?)?.rref()?;
        if columns.iter().any(|c| *c >= a.cols) {
            return Ok(None);
        }
        let mut solution: Vec<Vec<Basic>> = vec![vec![int(0); b.cols]; a.cols];
        for (i, column) in columns.iter().enumerate() {
            solution[*column] = echelon.rows[i][a.cols..]
                .iter()
                .map(|x| g.expr(x))
                .collect::<PolyResult<_>>()?;
        }
        let particular: ExprMatrix = ExprMatrix::new(a.cols, b.cols, solution.into_iter().flatten().collect())?;
        return Ok(Some(LinearSolution {
            particular,
            nullspace: g.nullspace(&echelon, &columns, a.cols)?,
        }));
    }
    if ks.is_empty() {
        linear_solve_with(&IntegerLift, a, b)
    } else {
        linear_solve_with(&PolynomialLift::new(&ks), a, b)
    }
}

/// The Moore–Penrose pseudoinverse by a full-rank factorisation
/// (`pseudoinverse`).
pub fn pseudoinverse(m: &ExprMatrix) -> PolyResult<ExprMatrix> {
    let (reduced, pivots) = rref(m)?;
    let r: usize = pivots.len();
    if r == 0 {
        return Ok(ExprMatrix::zeros(m.cols, m.rows));
    }
    let mut b_entries: Vec<Basic> = Vec::new();
    for i in 0..m.rows {
        for j in &pivots {
            b_entries.push(m.at(i, *j).clone());
        }
    }
    let b: ExprMatrix = ExprMatrix::new(m.rows, r, b_entries)?;
    let mut c_entries: Vec<Basic> = Vec::new();
    for i in 0..r {
        for j in 0..m.cols {
            c_entries.push(reduced.at(i, j).clone());
        }
    }
    let c: ExprMatrix = ExprMatrix::new(r, m.cols, c_entries)?;
    let (bt, ct) = (b.transpose(), c.transpose());
    let product: ExprMatrix = ct
        .matmul(&inverse(&c.matmul(&ct)?)?)?
        .matmul(&inverse(&bt.matmul(&b)?)?)?
        .matmul(&bt)?;
    product.applyfunc(&normal)
}
