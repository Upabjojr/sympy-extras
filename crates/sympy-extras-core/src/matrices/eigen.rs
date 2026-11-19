//! Port of `sympy_extras/matrices/eigen.py`: eigenvalues and eigenvectors
//! of matrices of expressions. The characteristic polynomial is factored
//! over the field of the entries (`QQ`, `QQ(parameters)` or `QQ<I>`); for
//! every irreducible factor the eigenspace is computed in the extension
//! field `K[t]/(factor)` (a `QuotientField`), so the eigenvectors are
//! polynomials in the eigenvalue; the Jordan chains likewise from the
//! kernels of the powers of `M - t`.

use crate::core::basic::{Basic, Head};
use crate::core::expr::{as_independent, terms_of};
use crate::error::{value_error, PolyError, PolyResult, PolysErrorKind};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::domainmatrix::DomainMatrix;
use crate::matrices::linalg::{
    charpoly, complex_lift, is_radical, kernels as kernels_of, square, ComplexLift, Ground, IntegerLift, Lift,
    PolynomialLift, Radical,
};
use crate::numbers::{numer, Rat};
use crate::polys::domains::domain::Domain;
use crate::polys::domains::integerring::IntegerRing;
use crate::polys::domains::multiquadratic::MultiquadraticField;
use crate::polys::domains::quotientfield::QuotientField;
use crate::polys::domains::rationalfield::{RationalField, QQ};
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};

type R = PolyResult<Basic>;

fn int(n: i64) -> Basic {
    Basic::int(n)
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
fn neg(a: &Basic) -> R {
    crate::core::mul::neg(a)
}
fn pw(a: &Basic, b: &Basic) -> R {
    crate::core::power::pow(a, b)
}
pub(crate) fn sqrt(a: &Basic) -> R {
    pw(
        a,
        &Basic::from_rational(Rat::from(crate::numbers::Int::from(1)) / Rat::from(crate::numbers::Int::from(2))),
    )
}
fn expand(e: &Basic) -> R {
    crate::core::function::expand_default(e)
}
fn has_i(e: &Basic) -> bool {
    e.preorder().iter().any(|n| matches!(n.head(), Head::ImaginaryUnit))
}
fn sorted_by_key(mut v: Vec<Basic>) -> Vec<Basic> {
    v.sort_by_cached_key(crate::core::sorting::sort_key);
    v
}

/// The coefficients of a polynomial in `x`, the leading one first
/// (`_coefficients`).
fn coefficients(e: &Basic, x: &Basic) -> PolyResult<Vec<Basic>> {
    let expanded: Basic = expand(e)?;
    let mut found: Vec<(usize, Vec<Basic>)> = Vec::new();
    for term in terms_of(&expanded) {
        let (coefficient, power) = as_independent(&term, x, false)?;
        let degree: usize = if power == int(1) {
            0
        } else if power == *x {
            1
        } else if power.is_pow()
            && power.args()[0] == *x
            && power.args()[1].is_integer_value()
            && numer(power.args()[1].as_rational().expect("an integer")) > crate::numbers::Int::from(0)
        {
            usize::try_from(&numer(power.args()[1].as_rational().expect("an integer")))
                .map_err(|_| value_error("degree too large".to_string()))?
        } else {
            return Err(value_error(format!("{} is not a polynomial in {}", e, x)));
        };
        match found.iter_mut().find(|(d, _)| *d == degree) {
            Some(entry) => entry.1.push(coefficient),
            None => found.push((degree, vec![coefficient])),
        }
    }
    let top: usize = found.iter().map(|(d, _)| *d).max().unwrap_or(0);
    let mut out: Vec<Basic> = Vec::new();
    for k in (0..=top).rev() {
        let parts: Vec<Basic> = found
            .iter()
            .find(|(d, _)| *d == k)
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        out.push(crate::core::add::add(&parts)?);
    }
    Ok(out)
}

fn is_polynomial_error(e: &PolyError) -> bool {
    matches!(
        e,
        PolyError::Polys(PolysErrorKind::PolynomialError | PolysErrorKind::NotWellOriented, _)
    )
}

/// The roots of an irreducible factor in radicals, or `None`
/// (`_roots_in_radicals`).
pub fn roots_in_radicals(factor: &Basic, x: &Basic) -> PolyResult<Option<Vec<Basic>>> {
    let cs: Vec<Basic> = coefficients(factor, x)?;
    if cs.len() == 2 {
        return Ok(Some(vec![d2(&neg(&cs[1])?, &cs[0])?]));
    }
    if cs.iter().any(has_i) {
        if cs.len() != 3 {
            return Ok(None);
        }
        let (a, b, c) = (&cs[0], &cs[1], &cs[2]);
        let root: Basic = sqrt(&expand(&s2(&m2(b, b)?, &m2(&m2(&int(4), a)?, c)?)?)?)?;
        let two_a: Basic = m2(&int(2), a)?;
        let first: Basic = d2(&s2(&neg(b)?, &root)?, &two_a)?;
        let second: Basic = d2(&a2(&neg(b)?, &root)?, &two_a)?;
        return Ok(Some(sorted_by_key(vec![first, second])));
    }
    let found = match crate::polys::polyroots::roots(factor, std::slice::from_ref(x), true, true, None) {
        Ok(f) => f,
        Err(e) if is_polynomial_error(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let total: usize = found.iter().map(|(_, k)| *k).sum();
    if total != cs.len() - 1 {
        return Ok(None);
    }
    Ok(Some(sorted_by_key(found.iter().map(|(r, _)| r.clone()).collect())))
}

/// The eigenvectors of an irreducible factor of the characteristic
/// polynomial, polynomials in its root `symbol` (`Eigenspace`).
#[derive(Debug, Clone)]
pub struct Eigenspace {
    pub factor: Basic,
    pub symbol: Basic,
    pub multiplicity: usize,
    pub basis: Vec<ExprMatrix>,
}

impl Eigenspace {
    pub fn geometric_multiplicity(&self) -> usize {
        self.basis.len()
    }

    /// The roots of the factor in radicals (`values`).
    pub fn values(&self) -> PolyResult<Option<Vec<Basic>>> {
        roots_in_radicals(&self.factor, &self.symbol)
    }

    /// The eigenvectors of the eigenvalue `value` (`vectors`).
    pub fn vectors(&self, value: &Basic) -> PolyResult<Vec<ExprMatrix>> {
        self.basis
            .iter()
            .map(|v| v.subs(&[(self.symbol.clone(), value.clone())]))
            .collect()
    }

    pub fn repr(&self) -> String {
        let basis: Vec<String> = self.basis.iter().map(|v| v.repr()).collect();
        format!(
            "Eigenspace({}, {}, [{}])",
            self.factor,
            self.multiplicity,
            basis.join(", ")
        )
    }
}

/// The Jordan chains of an irreducible factor (`JordanChains`).
#[derive(Debug, Clone)]
pub struct JordanChains {
    pub factor: Basic,
    pub symbol: Basic,
    pub multiplicity: usize,
    pub chains: Vec<Vec<ExprMatrix>>,
}

impl JordanChains {
    pub fn block_sizes(&self) -> Vec<usize> {
        self.chains.iter().map(|c| c.len()).collect()
    }
}

/// The field of the entries of a matrix (`_Field`).
pub(crate) trait Field {
    type D: Domain + Clone;
    fn domain(&self) -> &Self::D;
    fn element(&self, e: &Basic) -> PolyResult<<Self::D as Domain>::Elem>;
    fn expr(&self, a: &<Self::D as Domain>::Elem) -> R;
    fn conjugate(&self, a: &<Self::D as Domain>::Elem) -> <Self::D as Domain>::Elem {
        a.clone()
    }
}

pub(crate) struct RationalEntries {
    lift: IntegerLift,
}

impl RationalEntries {
    pub(crate) fn new() -> RationalEntries {
        RationalEntries { lift: IntegerLift }
    }
}

impl Field for RationalEntries {
    type D = RationalField;
    fn domain(&self) -> &RationalField {
        &QQ
    }
    fn element(&self, e: &Basic) -> PolyResult<Rat> {
        let (n, d) = self.lift.fraction(e)?;
        Ok(Rat::from(n) / Rat::from(d))
    }
    fn expr(&self, a: &Rat) -> R {
        Ok(Basic::from_rational(a.clone()))
    }
}

pub(crate) struct FunctionEntries {
    field: SparseFractionField<IntegerRing>,
    lift: PolynomialLift,
}

impl FunctionEntries {
    pub(crate) fn new(lift: PolynomialLift) -> FunctionEntries {
        FunctionEntries {
            field: SparseFractionField::new(lift.ring.clone()),
            lift,
        }
    }
}

impl Field for FunctionEntries {
    type D = SparseFractionField<IntegerRing>;
    fn domain(&self) -> &SparseFractionField<IntegerRing> {
        &self.field
    }
    fn element(&self, e: &Basic) -> PolyResult<SparseFracElement<IntegerRing>> {
        let (n, d) = self.lift.fraction(e)?;
        self.field.new_element(&n, &d)
    }
    fn expr(&self, a: &SparseFracElement<IntegerRing>) -> R {
        self.lift.ratio(&a.numer, &a.denom)
    }
}

pub(crate) struct GaussianEntries {
    lift: ComplexLift,
}

impl GaussianEntries {
    pub(crate) fn new(lift: ComplexLift) -> GaussianEntries {
        GaussianEntries { lift }
    }
}

impl Field for GaussianEntries {
    type D = MultiquadraticField<Ground>;
    fn domain(&self) -> &MultiquadraticField<Ground> {
        &self.lift.field
    }
    fn element(&self, e: &Basic) -> PolyResult<Radical> {
        self.lift.element(e)
    }
    fn expr(&self, a: &Radical) -> R {
        self.lift.expr(a)
    }
    fn conjugate(&self, a: &Radical) -> Radical {
        match self.lift.position.iter().find(|(d, _)| *d == -1) {
            None => a.clone(),
            Some((_, i)) => self.lift.field.conjugate(a, *i),
        }
    }
}

fn fresh_symbol(m: &ExprMatrix) -> Basic {
    let mut used: Vec<String> = Vec::new();
    for e in &m.entries {
        for s in e.free_symbols() {
            if matches!(s.head(), Head::Symbol(_)) {
                used.push(s.to_string());
            }
        }
    }
    let (mut name, mut k) = ("_lambda".to_string(), 0);
    while used.contains(&name) {
        name = format!("_lambda{}", k);
        k += 1;
    }
    crate::core::symbol::Symbol::new(&name, &[]).0
}

type Ext<F> = QuotientField<<F as Field>::D>;
type ExtElem<F> = Vec<<<F as Field>::D as Domain>::Elem>;

fn field_nullspace<K: Domain + Clone>(matrix: &DomainMatrix<K>) -> PolyResult<Vec<Vec<K::Elem>>> {
    let f: &K = &matrix.domain;
    let (reduced, pivots) = matrix.rref()?;
    let mut basis: Vec<Vec<K::Elem>> = Vec::new();
    for free in 0..matrix.ncols {
        if pivots.contains(&free) {
            continue;
        }
        let mut vector: Vec<K::Elem> = vec![f.zero(); matrix.ncols];
        vector[free] = f.one();
        for (i, column) in pivots.iter().enumerate() {
            vector[*column] = f.neg(&reduced.rows[i][free]);
        }
        basis.push(vector);
    }
    Ok(basis)
}

fn extension<F: Field>(field: &F, factor: &Basic, unknown: &Basic) -> PolyResult<Ext<F>> {
    let k: &F::D = field.domain();
    let cs: Vec<_> = coefficients(factor, unknown)?
        .iter()
        .map(|c| field.element(c))
        .collect::<PolyResult<_>>()?;
    let leading = k.invert(&cs[0])?;
    QuotientField::new(k.clone(), cs.iter().map(|c| k.mul(c, &leading)).collect())
}

fn shifted<F: Field>(field: &F, f: &Ext<F>, m: &ExprMatrix) -> PolyResult<DomainMatrix<Ext<F>>> {
    let mut rows = Vec::new();
    for i in 0..m.rows {
        let mut row = Vec::new();
        for (j, e) in m.row(i).iter().enumerate() {
            let g = f.from_ground(&field.element(e)?);
            row.push(if i == j { f.sub(&g, &f.generator()) } else { g });
        }
        rows.push(row);
    }
    DomainMatrix::new(rows, f.clone(), Some(m.cols))
}

fn column<F: Field>(field: &F, vector: &[ExtElem<F>], x: &Basic) -> PolyResult<ExprMatrix> {
    let mut entries: Vec<Basic> = Vec::new();
    for element in vector {
        let degree: i64 = element.len() as i64 - 1;
        let mut parts: Vec<Basic> = Vec::new();
        for (k, c) in element.iter().enumerate() {
            parts.push(m2(&field.expr(c)?, &pw(x, &int(degree - k as i64))?)?);
        }
        entries.push(crate::core::add::add(&parts)?);
    }
    Ok(ExprMatrix::column(entries))
}

fn eigenspaces_with<F: Field>(
    field: &F,
    m: &ExprMatrix,
    factors: &[(Basic, usize)],
    unknown: &Basic,
    x: &Basic,
) -> PolyResult<Vec<Eigenspace>> {
    let mut spaces: Vec<Eigenspace> = Vec::new();
    for (factor, multiplicity) in factors {
        let f = extension(field, factor, unknown)?;
        let mut basis: Vec<ExprMatrix> = Vec::new();
        for vector in field_nullspace(&shifted(field, &f, m)?)? {
            basis.push(column(field, &vector, x)?);
        }
        spaces.push(Eigenspace {
            factor: crate::core::subs::subs(factor, unknown, x)?,
            symbol: x.clone(),
            multiplicity: *multiplicity,
            basis,
        });
    }
    Ok(spaces)
}

fn complete<K: Domain + Clone>(
    f: &K,
    size: usize,
    known: &[Vec<K::Elem>],
    candidates: &[Vec<K::Elem>],
) -> PolyResult<Vec<Vec<K::Elem>>> {
    let vectors: Vec<Vec<K::Elem>> = known.iter().chain(candidates.iter()).cloned().collect();
    if vectors.is_empty() {
        return Ok(Vec::new());
    }
    let rows: Vec<Vec<K::Elem>> = (0..size)
        .map(|i| vectors.iter().map(|v| v[i].clone()).collect())
        .collect();
    let columns = DomainMatrix::new(rows, f.clone(), Some(vectors.len()))?;
    let (_, pivots) = columns.rref()?;
    Ok(pivots
        .iter()
        .filter(|j| **j >= known.len())
        .map(|j| vectors[*j].clone())
        .collect())
}

fn jordan_chains_with<F: Field>(
    field: &F,
    m: &ExprMatrix,
    factors: &[(Basic, usize)],
    unknown: &Basic,
    x: &Basic,
) -> PolyResult<Vec<JordanChains>> {
    let n: usize = m.rows;
    let mut result: Vec<JordanChains> = Vec::new();
    for (factor, multiplicity) in factors {
        let f = extension(field, factor, unknown)?;
        let nm = shifted(field, &f, m)?;
        let image = |v: &[ExtElem<F>]| -> Vec<ExtElem<F>> {
            let mut out = Vec::new();
            for row in &nm.rows {
                let mut total = f.zero();
                for (a, b) in row.iter().zip(v.iter()) {
                    if !a.is_empty() && !b.is_empty() {
                        total = f.add(&total, &f.mul(a, b));
                    }
                }
                out.push(total);
            }
            out
        };
        let mut kernels: Vec<Vec<Vec<ExtElem<F>>>> = vec![Vec::new()];
        let mut power = nm.clone();
        loop {
            kernels.push(field_nullspace(&power)?);
            let last: usize = kernels[kernels.len() - 1].len();
            if last == *multiplicity {
                break;
            }
            if last == kernels[kernels.len() - 2].len() {
                return Err(PolyError::Arithmetic(format!(
                    "the generalised eigenspace of {} has not the dimension {}",
                    factor, multiplicity
                )));
            }
            power = power.mul(&nm)?;
        }
        let mut level: Vec<Vec<ExtElem<F>>> = Vec::new();
        let mut heads: Vec<(usize, Vec<ExtElem<F>>)> = Vec::new();
        for k in (1..kernels.len()).rev() {
            let images: Vec<Vec<ExtElem<F>>> = level.iter().map(|v| image(v)).collect();
            let mut known: Vec<Vec<ExtElem<F>>> = kernels[k - 1].clone();
            known.extend(images.iter().cloned());
            let fresh = complete(&f, n, &known, &kernels[k])?;
            heads.extend(fresh.iter().map(|v| (k, v.clone())));
            level = images;
            level.extend(fresh);
        }
        let mut chains: Vec<Vec<ExprMatrix>> = Vec::new();
        for (length, head) in heads {
            let mut chain: Vec<Vec<ExtElem<F>>> = vec![head];
            for _ in 0..length - 1 {
                let next = image(&chain[chain.len() - 1]);
                chain.push(next);
            }
            chain.reverse();
            chains.push(chain.iter().map(|v| column(field, v, x)).collect::<PolyResult<_>>()?);
        }
        result.push(JordanChains {
            factor: crate::core::subs::subs(factor, unknown, x)?,
            symbol: x.clone(),
            multiplicity: *multiplicity,
            chains,
        });
    }
    Ok(result)
}

fn characteristic_factors(m: &ExprMatrix, unknown: &Basic) -> PolyResult<(Vec<Basic>, Vec<(Basic, usize)>)> {
    square(m)?;
    let kernels: Vec<Basic> = kernels_of(&[m]);
    if !kernels.iter().all(|k| k.is_symbol() || is_radical(k)) {
        return Err(PolyError::NotImplemented(
            "eigenvalues of a matrix whose entries are not rational functions of symbols, I and square roots of rationals".to_string(),
        ));
    }
    if m.rows == 0 {
        return Ok((kernels, Vec::new()));
    }
    let polynomial: Basic = charpoly(m, unknown)?;
    let symbols: Vec<Basic> = kernels.iter().filter(|k| k.is_symbol()).cloned().collect();
    let numerator: Basic = if kernels.is_empty() {
        polynomial.clone()
    } else {
        expand(
            &crate::core::expr::as_numer_denom(&crate::polys::rationaltools::together(&polynomial, false, true)?)?.0,
        )?
    };
    let remaining: Vec<Basic> = kernels_of(&[&ExprMatrix::column(vec![numerator.clone()])]);
    if remaining
        .iter()
        .any(|k| is_radical(k) && !matches!(k.head(), Head::ImaginaryUnit))
    {
        let shown: Basic =
            crate::core::subs::subs(&polynomial, unknown, &crate::core::symbol::Symbol::new("x", &[]).0)?;
        return Err(PolyError::NotImplemented(format!(
            "the characteristic polynomial {} has irrational coefficients: its factorisation over the field they generate is needed",
            shown
        )));
    }
    if remaining.iter().any(|k| matches!(k.head(), Head::ImaginaryUnit)) {
        use crate::polys::domains::gaussiandomains::QQ_I;
        let mut names: Vec<String> = vec![unknown.to_string()];
        names.extend(symbols.iter().map(|s| s.to_string()));
        let ring = crate::polys::rings::PolyRing::new(QQ_I, names.clone(), crate::polys::rings::MonomialOrder::Lex)?;
        let terms = crate::polys::polyutils::dict_from_expr(&numerator, &names, &QQ_I)?;
        let (_, gaussian) = crate::polys::gaussianfactor::gaussian_ext_factor(&ring.from_dict(terms))?;
        let mut out: Vec<(Basic, usize)> = Vec::new();
        for (g, k) in gaussian {
            let e: Basic = crate::polys::polyutils::expr_from_dict(&g.terms, &names, &QQ_I)?;
            if e.has(unknown) {
                out.push((e, k));
            }
        }
        return Ok((kernels, out));
    }
    let mut gens: Vec<Basic> = vec![unknown.clone()];
    gens.extend(symbols);
    let (_, listed) = crate::polys::polytools::factor_list(&numerator, &gens)?;
    let mut factors: Vec<(Basic, usize)> = Vec::new();
    for (factor, exponent) in listed {
        if !factor.has(unknown) {
            continue;
        }
        if !exponent.is_integer_value() {
            return Err(PolyError::Type(format!(
                "a multiplicity which is not an integer: {}",
                exponent
            )));
        }
        let k: usize = usize::try_from(&numer(exponent.as_rational().expect("an integer")))
            .map_err(|_| value_error("multiplicity".to_string()))?;
        factors.push((factor, k));
    }
    Ok((kernels, factors))
}

/// The eigenspaces, one per irreducible factor of the characteristic
/// polynomial, in the symbol `x` of its root (`eigenspaces`).
pub fn eigenspaces(m: &ExprMatrix, x: &Basic) -> PolyResult<Vec<Eigenspace>> {
    let unknown: Basic = fresh_symbol(m);
    let (kernels, factors) = characteristic_factors(m, &unknown)?;
    if let Some(g) = complex_lift(&kernels)? {
        return eigenspaces_with(&GaussianEntries::new(g), m, &factors, &unknown, x);
    }
    if !kernels.is_empty() {
        return eigenspaces_with(
            &FunctionEntries::new(PolynomialLift::new(&kernels)),
            m,
            &factors,
            &unknown,
            x,
        );
    }
    eigenspaces_with(&RationalEntries::new(), m, &factors, &unknown, x)
}

/// The Jordan chains, one group per irreducible factor (`jordan_chains`).
pub fn jordan_chains(m: &ExprMatrix, x: &Basic) -> PolyResult<Vec<JordanChains>> {
    let unknown: Basic = fresh_symbol(m);
    let (kernels, factors) = characteristic_factors(m, &unknown)?;
    if let Some(g) = complex_lift(&kernels)? {
        return jordan_chains_with(&GaussianEntries::new(g), m, &factors, &unknown, x);
    }
    if !kernels.is_empty() {
        return jordan_chains_with(
            &FunctionEntries::new(PolynomialLift::new(&kernels)),
            m,
            &factors,
            &unknown,
            x,
        );
    }
    jordan_chains_with(&RationalEntries::new(), m, &factors, &unknown, x)
}

/// `(P, J)` with `M = P*J*P**-1`, `J` the Jordan form (`jordan_form`).
pub fn jordan_form(m: &ExprMatrix) -> PolyResult<(ExprMatrix, ExprMatrix)> {
    let n: usize = square(m)?;
    let mut blocks: Vec<(Basic, Vec<ExprMatrix>)> = Vec::new();
    for group in jordan_chains(m, &fresh_symbol(m))? {
        let Some(found) = roots_in_radicals(&group.factor, &group.symbol)? else {
            return Err(PolyError::NotImplemented(format!(
                "the roots of {} are not expressed in radicals: see jordan_chains",
                group.factor
            )));
        };
        for value in found {
            for vectors in &group.chains {
                let substituted: Vec<ExprMatrix> = vectors
                    .iter()
                    .map(|v| v.subs(&[(group.symbol.clone(), value.clone())]))
                    .collect::<PolyResult<_>>()?;
                blocks.push((value.clone(), substituted));
            }
        }
    }
    blocks.sort_by_cached_key(|b| (crate::core::sorting::sort_key(&b.0), std::cmp::Reverse(b.1.len())));
    let mut columns: Vec<ExprMatrix> = Vec::new();
    let mut j: Vec<Vec<Basic>> = vec![vec![int(0); n]; n];
    for (value, block) in &blocks {
        for (position, vector) in block.iter().enumerate() {
            let index: usize = columns.len();
            j[index][index] = value.clone();
            if position > 0 {
                j[index - 1][index] = int(1);
            }
            columns.push(vector.clone());
        }
    }
    let mut p: Vec<Basic> = Vec::new();
    for i in 0..n {
        for c in columns.iter().take(n) {
            p.push(c.at(i, 0).clone());
        }
    }
    Ok((
        ExprMatrix::new(n, n, p)?,
        ExprMatrix::new(n, n, j.into_iter().flatten().collect())?,
    ))
}

fn explicit(m: &ExprMatrix) -> PolyResult<Vec<(Basic, Eigenspace)>> {
    let mut pairs: Vec<(Basic, Eigenspace)> = Vec::new();
    for space in eigenspaces(m, &fresh_symbol(m))? {
        let Some(values) = space.values()? else {
            return Err(PolyError::NotImplemented(format!(
                "the roots of {} are not expressed in radicals: see eigenspaces",
                space.factor
            )));
        };
        for value in values {
            pairs.push((value, space.clone()));
        }
    }
    pairs.sort_by_cached_key(|p| crate::core::sorting::sort_key(&p.0));
    Ok(pairs)
}

/// The eigenvalues with their algebraic multiplicities (`eigenvalues`).
pub fn eigenvalues(m: &ExprMatrix) -> PolyResult<Vec<(Basic, usize)>> {
    Ok(explicit(m)?.into_iter().map(|(v, s)| (v, s.multiplicity)).collect())
}

/// `(eigenvalue, multiplicity, eigenvectors)` (`eigenvectors`).
#[allow(clippy::type_complexity)]
pub fn eigenvectors(m: &ExprMatrix) -> PolyResult<Vec<(Basic, usize, Vec<ExprMatrix>)>> {
    explicit(m)?
        .into_iter()
        .map(|(v, s)| Ok((v.clone(), s.multiplicity, s.vectors(&v)?)))
        .collect()
}

pub fn is_diagonalizable(m: &ExprMatrix) -> PolyResult<bool> {
    Ok(eigenspaces(m, &fresh_symbol(m))?
        .iter()
        .all(|s| s.geometric_multiplicity() == s.multiplicity))
}

/// `(P, D)` with `M = P*D*P**-1`, or `None` (`diagonalize`).
pub fn diagonalize(m: &ExprMatrix) -> PolyResult<Option<(ExprMatrix, ExprMatrix)>> {
    let n: usize = square(m)?;
    let mut columns: Vec<ExprMatrix> = Vec::new();
    let mut diagonal: Vec<Basic> = Vec::new();
    for (value, space) in explicit(m)? {
        if space.geometric_multiplicity() != space.multiplicity {
            return Ok(None);
        }
        for vector in space.vectors(&value)? {
            columns.push(vector);
            diagonal.push(value.clone());
        }
    }
    let mut p: Vec<Basic> = Vec::new();
    for i in 0..n {
        for c in columns.iter().take(n) {
            p.push(c.at(i, 0).clone());
        }
    }
    let d: Vec<Basic> = (0..n * n)
        .map(|k| {
            if k / n == k % n {
                diagonal[k / n].clone()
            } else {
                int(0)
            }
        })
        .collect();
    Ok(Some((ExprMatrix::new(n, n, p)?, ExprMatrix::new(n, n, d)?)))
}
