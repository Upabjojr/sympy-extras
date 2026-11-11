//! Port of `sympy_extras/concrete/pisigma.py`: ΠΣ-fields
//! `C(k)(t_1, ..., t_m)` with the shift `σ` and Karr's solver of the
//! parameterized first order equation `σ(g) - a g = c_1 f_1 + ... + c_r f_r`
//! (polynomial in a Σ-generator, Laurent polynomial in a Π-generator,
//! Abramov's universal denominator at the rational level).
//!
//! The elements are rational functions over `QQ` in the variables
//! `params + [k, t_1, ..., t_m]` ([`Rep`], canonical in lowest terms); the
//! constants are given by a [`Constants`] ([`RationalConstants`],
//! [`ParametricConstants`]). Python's `PiSigmaTower` and `PiSigmaField`
//! are one generic struct here, [`PiSigmaField`], and a [`PiSigmaElement`]
//! holds its representation only (the arithmetic is on the field).
//!
//! The sequences of the generators (`harmonic(k)`, `factorial(k)`) are
//! expressions of the core; the function classes they need may not be in
//! the Rust core yet, in which case a generator stands for itself.

// a node caches its keys in `OnceLock`s that the hash never reads; the
// linear algebra indexes as the Python module does
#![allow(clippy::mutable_key_type, clippy::needless_range_loop)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::core::add::add;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is_integer_class;
use crate::core::exprtools::factor_terms;
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::core::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;
use crate::polys::domains::domain::Domain;
use crate::polys::domains::integerring::IntegerRing;
use crate::polys::domains::rationalfield::{RationalField, QQ};
use crate::polys::domains::sparsefractionfield::{SparseFracElement, SparseFractionField};
use crate::polys::domains::sparsepolynomialring::SparsePolynomialRing;
use crate::polys::multifactor::zz_factor;
use crate::polys::rings::{Monomial, PolyElement, PolyRing, Terms};

/// An element of the field of rational functions in all the variables.
pub type Rep = SparseFracElement<RationalField>;
/// A polynomial in all the variables.
pub type Poly = PolyElement<RationalField>;
/// The image of a variable under the shift: a numerator and a denominator.
pub type Image = (Poly, Poly);

/// The identifiers of the fields, for the names of the generators.
static FIELD_IDS: AtomicUsize = AtomicUsize::new(0);

/// An element of a ΠΣ-field: a rational function in lowest terms.
#[derive(Debug, Clone, PartialEq)]
pub struct PiSigmaElement {
    pub rep: Rep,
}

impl PiSigmaElement {
    pub fn numer(&self) -> &Poly {
        &self.rep.numer
    }

    pub fn denom(&self) -> &Poly {
        &self.rep.denom
    }

    pub fn is_nonzero(&self) -> bool {
        !self.rep.numer.terms.is_empty()
    }
}

/// `p` in a ring with more variables, the new ones last (`_pad`).
fn pad(p: &Poly, ring: &Arc<PolyRing<RationalField>>) -> Poly {
    let extra: usize = ring.ngens - p.ring.ngens;
    let terms: Terms<Rat> = p
        .terms
        .iter()
        .map(|(m, c)| {
            let mut padded: Monomial = m.clone();
            padded.extend(vec![0; extra]);
            (padded, c.clone())
        })
        .collect();
    PolyElement {
        terms,
        ring: ring.clone(),
    }
}

/// `p`, free of the last variables, in the ring of the first ones
/// (`_truncate`).
fn truncate(p: &Poly, ring: &Arc<PolyRing<RationalField>>) -> Poly {
    let n: usize = ring.ngens;
    let terms: Terms<Rat> = p.terms.iter().map(|(m, c)| (m[..n].to_vec(), c.clone())).collect();
    PolyElement {
        terms,
        ring: ring.clone(),
    }
}

// ----------------------------------------------------------------------
// the constants

/// The field of constants of a ΠΣ-field with its embedding in the field
/// of the tower, whose first variables are the parameters (`Constants`).
pub trait Constants: Clone {
    type D: Domain;
    fn domain(&self) -> &Self::D;
    fn params(&self) -> &[Basic];
    /// The constant as an element of `field`.
    fn embed(&self, c: &<Self::D as Domain>::Elem, field: &SparseFractionField<RationalField>) -> Rep;
    /// The constant equal to `f`, which involves the parameters only.
    fn restrict(&self, f: &Rep) -> <Self::D as Domain>::Elem;
    /// The constant equal to the polynomial `p` in the parameters.
    fn from_poly(&self, p: &Poly) -> PolyResult<<Self::D as Domain>::Elem>;
    /// The value of `c` if it is a rational number.
    fn rational(&self, c: &<Self::D as Domain>::Elem) -> Option<Rat>;
}

/// The constants `QQ` (`RationalConstants`).
#[derive(Debug, Clone)]
pub struct RationalConstants;

impl Constants for RationalConstants {
    type D = RationalField;

    fn domain(&self) -> &RationalField {
        &QQ
    }

    fn params(&self) -> &[Basic] {
        &[]
    }

    fn embed(&self, c: &Rat, field: &SparseFractionField<RationalField>) -> Rep {
        let r = &field.ring.ring;
        SparseFracElement {
            numer: r.ground(rat_from_int(&numer(c))),
            denom: r.ground(rat_from_int(&denom(c))),
        }
    }

    fn restrict(&self, f: &Rep) -> Rat {
        f.numer.LC() / f.denom.LC()
    }

    fn from_poly(&self, p: &Poly) -> PolyResult<Rat> {
        Ok(p.coeff(&p.ring.zero_monom))
    }

    fn rational(&self, c: &Rat) -> Option<Rat> {
        Some(c.clone())
    }
}

/// The constants `QQ(params)` (`ParametricConstants`).
#[derive(Debug, Clone)]
pub struct ParametricConstants {
    pub params: Vec<Basic>,
    pub field: SparseFractionField<RationalField>,
}

impl ParametricConstants {
    pub fn new(params: &[Basic]) -> PolyResult<ParametricConstants> {
        if params.is_empty() {
            return Err(PolyError::Value(
                "the parametric constants need parameters (RationalConstants otherwise)".to_string(),
            ));
        }
        let names: Vec<String> = params.iter().map(name_of).collect();
        Ok(ParametricConstants {
            params: params.to_vec(),
            field: SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names)),
        })
    }
}

impl Constants for ParametricConstants {
    type D = SparseFractionField<RationalField>;

    fn domain(&self) -> &SparseFractionField<RationalField> {
        &self.field
    }

    fn params(&self) -> &[Basic] {
        &self.params
    }

    fn embed(&self, c: &Rep, field: &SparseFractionField<RationalField>) -> Rep {
        // in lowest terms in the parameters is in lowest terms in all the
        // variables (the lexicographic order agrees)
        let r = &field.ring.ring;
        SparseFracElement {
            numer: pad(&c.numer, r),
            denom: pad(&c.denom, r),
        }
    }

    fn restrict(&self, f: &Rep) -> Rep {
        let r = &self.field.ring.ring;
        SparseFracElement {
            numer: truncate(&f.numer, r),
            denom: truncate(&f.denom, r),
        }
    }

    fn from_poly(&self, p: &Poly) -> PolyResult<Rep> {
        let r = &self.field.ring.ring;
        self.field.new_element(&truncate(p, r), &r.one())
    }

    fn rational(&self, c: &Rep) -> Option<Rat> {
        if c.numer.is_ground() && c.denom.is_ground() {
            return Some(c.numer.LC() / c.denom.LC());
        }
        None
    }
}

fn name_of(s: &Basic) -> String {
    match s.head() {
        Head::Symbol(d) | Head::Dummy(d, _) => d.name.clone(),
        Head::Wild(w) => w.symbol.name.clone(),
        _ => s.to_string(),
    }
}

// ----------------------------------------------------------------------
// the tower

/// The kind of a generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `σ(t) = t + β`.
    Sigma,
    /// `σ(t) = α t`.
    Pi,
}

/// A generator of a ΠΣ-field (`Extension`): its kind, its symbol, the
/// defining expression of `α` or `β`, and the sequence it stands for.
#[derive(Debug, Clone)]
pub struct Extension {
    pub kind: Kind,
    pub symbol: Basic,
    pub value: Basic,
    pub expr: Basic,
}

/// A basis of solutions `(c_1, ..., c_r, g)`, or `None` when the equation
/// is out of scope.
pub type Solutions<C> = Option<Vec<(Vec<C>, PiSigmaElement)>>;

/// The right hand sides of a step of the coefficientwise recursion
/// (`Rhs`).
type Rhs<'a, C> =
    Box<dyn Fn(&[Vec<C>], &HashMap<i64, Vec<PiSigmaElement>>, usize) -> PolyResult<Vec<PiSigmaElement>> + 'a>;

type Elem<Cs> = <<Cs as Constants>::D as Domain>::Elem;

/// A ΠΣ-field `C(k)(t_1, ..., t_m)` with its shift and Karr's solver
/// (`PiSigmaTower` and `PiSigmaField`).
pub struct PiSigmaField<Cs: Constants> {
    pub k: Basic,
    pub params: Vec<Basic>,
    pub extensions: Vec<Extension>,
    id: usize,
    p: usize,
    pub field: SparseFractionField<RationalField>,
    ring: Arc<PolyRing<RationalField>>,
    values: Vec<Rep>,
    /// The symbols of the variables.
    pub symbols: Vec<Basic>,
    index: HashMap<Basic, usize>,
    pub gens: Vec<PiSigmaElement>,
    pub zero: PiSigmaElement,
    pub one: PiSigmaElement,
    images: Vec<Image>,
    inverse_images: RefCell<Vec<Image>>,
    power_cache: RefCell<HashMap<(bool, usize, u32, bool), Poly>>,
    pub constants: Cs,
}

impl<Cs: Constants> PiSigmaField<Cs> {
    /// The field `C(k)` with the extensions and their values (`α` or `β`
    /// in the fields below).
    pub fn new(
        k: &Basic,
        constants: Cs,
        extensions: Vec<Extension>,
        values: Vec<Rep>,
        field_id: Option<usize>,
    ) -> PolyResult<PiSigmaField<Cs>> {
        if extensions.len() != values.len() {
            return Err(PolyError::Value("one value per extension".to_string()));
        }
        if !k.is_symbol() {
            return Err(PolyError::Value("the summation index must be a symbol".to_string()));
        }
        let id: usize = field_id.unwrap_or_else(|| FIELD_IDS.fetch_add(1, Ordering::Relaxed));
        let params: Vec<Basic> = constants.params().to_vec();
        let p: usize = params.len();
        let mut names: Vec<String> = params.iter().map(name_of).collect();
        names.push(name_of(k));
        names.extend((1..=extensions.len()).map(|i| format!("_ps{}_t{}", id, i)));
        let field = SparseFractionField::new(SparsePolynomialRing::sympy(QQ, names));
        let ring: Arc<PolyRing<RationalField>> = field.ring.ring.clone();
        let values: Vec<Rep> = values
            .iter()
            .map(|v| SparseFracElement {
                numer: pad(&v.numer, &ring),
                denom: pad(&v.denom, &ring),
            })
            .collect();
        let mut symbols: Vec<Basic> = params.clone();
        symbols.push(k.clone());
        symbols.extend(extensions.iter().map(|e| e.symbol.clone()));
        let index: HashMap<Basic, usize> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let all_gens: Vec<Poly> = ring.gens();
        let gens: Vec<PiSigmaElement> = all_gens[p..]
            .iter()
            .map(|g| PiSigmaElement {
                rep: field.from_ring(g),
            })
            .collect();
        let zero = PiSigmaElement { rep: field.zero() };
        let one = PiSigmaElement { rep: field.one() };
        let mut images: Vec<Image> = vec![(all_gens[p].add(&ring.one()), ring.one())];
        for (i, (e, v)) in extensions.iter().zip(&values).enumerate() {
            let t: &Poly = &all_gens[p + i + 1];
            images.push(match e.kind {
                Kind::Pi => (v.numer.mul(t), v.denom.clone()),
                Kind::Sigma => (t.mul(&v.denom).add(&v.numer), v.denom.clone()),
            });
        }
        Ok(PiSigmaField {
            k: k.clone(),
            params,
            extensions,
            id,
            p,
            field,
            ring,
            values,
            symbols,
            index,
            gens,
            zero,
            one,
            images,
            inverse_images: RefCell::new(Vec::new()),
            power_cache: RefCell::new(HashMap::new()),
            constants,
        })
    }

    fn element(&self, rep: Rep) -> PiSigmaElement {
        PiSigmaElement { rep }
    }

    /// The number of extensions.
    pub fn level(&self) -> usize {
        self.extensions.len()
    }

    // -- the arithmetic of the elements

    pub fn add(&self, a: &PiSigmaElement, b: &PiSigmaElement) -> PiSigmaElement {
        self.element(self.field.add(&a.rep, &b.rep))
    }

    pub fn sub(&self, a: &PiSigmaElement, b: &PiSigmaElement) -> PiSigmaElement {
        self.element(self.field.sub(&a.rep, &b.rep))
    }

    pub fn mul(&self, a: &PiSigmaElement, b: &PiSigmaElement) -> PiSigmaElement {
        self.element(self.field.mul(&a.rep, &b.rep))
    }

    pub fn div(&self, a: &PiSigmaElement, b: &PiSigmaElement) -> PolyResult<PiSigmaElement> {
        Ok(self.element(self.field.exquo(&a.rep, &b.rep)?))
    }

    pub fn neg(&self, a: &PiSigmaElement) -> PiSigmaElement {
        self.element(self.field.neg(&a.rep))
    }

    pub fn pow(&self, a: &PiSigmaElement, n: i64) -> PolyResult<PiSigmaElement> {
        Ok(self.element(self.field.pow_signed(&a.rep, n)?))
    }

    pub fn from_int(&self, n: i64) -> PiSigmaElement {
        self.element(self.field.from_int(n))
    }

    fn scale_int(&self, n: &Int, a: &PiSigmaElement) -> PiSigmaElement {
        self.mul(&self.from_fraction(&rat_from_int(n)), a)
    }

    // -- conversions

    /// A rational expression in the index, the parameters and the
    /// generator symbols as an element (`from_expr`).
    pub fn from_expr(&self, e: &Basic) -> PolyResult<PiSigmaElement> {
        Ok(self.element(self.rep_from_expr(e)?))
    }

    fn rep_from_expr(&self, e: &Basic) -> PolyResult<Rep> {
        let f = &self.field;
        if e.is_symbol() {
            let Some(&i) = self.index.get(e) else {
                return Err(PolyError::Value(format!("{} is not a variable of the field", e)));
            };
            return Ok(f.from_ring(&self.ring.gens()[i]));
        }
        if let Some(v) = e.as_rational() {
            return Ok(self.rational(v));
        }
        match e.head() {
            Head::Add => {
                let mut result: Rep = f.zero();
                for a in e.args() {
                    result = f.add(&result, &self.rep_from_expr(a)?);
                }
                Ok(result)
            }
            Head::Mul => {
                let mut result: Rep = f.one();
                for a in e.args() {
                    result = f.mul(&result, &self.rep_from_expr(a)?);
                }
                Ok(result)
            }
            Head::Pow if is_integer_class(&e.args()[1]) => {
                let base: Rep = self.rep_from_expr(&e.args()[0])?;
                let n: &Int = e.args()[1].as_rational().expect("an integer").numerator();
                if is_negative(n) && base.numer.terms.is_empty() {
                    return Err(PolyError::Value(format!("division by zero in {}", e)));
                }
                let n: i64 = i64::try_from(n).map_err(|_| PolyError::Value("a huge exponent".to_string()))?;
                f.pow_signed(&base, n)
            }
            _ => {
                let shown: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
                Err(PolyError::Value(format!(
                    "{} is not a rational function of {}",
                    e,
                    shown.join(", ")
                )))
            }
        }
    }

    /// The rational number `c` as an element.
    pub fn from_fraction(&self, c: &Rat) -> PiSigmaElement {
        self.element(self.rational(c))
    }

    fn rational(&self, c: &Rat) -> Rep {
        SparseFracElement {
            numer: self.ring.ground(rat_from_int(&numer(c))),
            denom: self.ring.ground(rat_from_int(&denom(c))),
        }
    }

    fn poly_expr(&self, p: &Poly, images: &[Basic]) -> PolyResult<Basic> {
        let mut parts: Vec<Basic> = Vec::new();
        for (m, c) in &p.terms {
            let mut factors: Vec<Basic> = vec![Basic::from_rational(c.clone())];
            for (s, e) in images.iter().zip(m) {
                if *e > 0 {
                    factors.push(if *e > 1 {
                        pow(s, &Basic::int(i64::from(*e)))?
                    } else {
                        s.clone()
                    });
                }
            }
            parts.push(mul(&factors)?);
        }
        if parts.is_empty() {
            return Ok(Basic::zero());
        }
        add(&parts)
    }

    /// An element as an expression; with `substitute` the generators are
    /// replaced by their sequences and the common factors taken out; with
    /// `positive_denominator` the signs follow SymPy's fraction field over
    /// the parameters (`to_expr`).
    pub fn to_expr(&self, f: &PiSigmaElement, substitute: bool, positive_denominator: bool) -> PolyResult<Basic> {
        let mut images: Vec<Basic> = self.params.clone();
        images.push(self.k.clone());
        images.extend(
            self.extensions
                .iter()
                .map(|e| if substitute { e.expr.clone() } else { e.symbol.clone() }),
        );
        let (mut numer_poly, mut denom_poly) = (f.numer().clone(), f.denom().clone());
        if positive_denominator && self.negative_lead(&denom_poly) {
            numer_poly = numer_poly.neg();
            denom_poly = denom_poly.neg();
        }
        let numerator: Basic = self.poly_expr(&numer_poly, &images)?;
        let mut expr: Basic = numerator.clone();
        if denom_poly != self.ring.one() {
            expr = mul(&[
                numerator,
                pow(&self.poly_expr(&denom_poly, &images)?, &Basic::negative_one())?,
            ])?;
        }
        if substitute {
            expr = factor_terms(&expr, false, false, false, true)?;
        }
        Ok(expr)
    }

    fn negative_lead(&self, p: &Poly) -> bool {
        let pp: usize = self.p;
        if pp == 0 || p.terms.is_empty() {
            return false;
        }
        let key = |m: &Monomial| -> Monomial {
            let mut k: Monomial = m[pp..].to_vec();
            k.extend_from_slice(&m[..pp]);
            k
        };
        let leading: &Monomial = p.terms.keys().max_by_key(|m| key(m)).expect("a term");
        rat_is_negative(&p.terms[leading])
    }

    fn symbol_images(&self) -> Vec<Basic> {
        let mut images: Vec<Basic> = self.params.clone();
        images.push(self.k.clone());
        images.extend(self.extensions.iter().map(|e| e.symbol.clone()));
        images
    }

    /// The numerator of `f` as an expression in the symbols.
    pub fn numer_expr(&self, f: &PiSigmaElement) -> PolyResult<Basic> {
        self.poly_expr(f.numer(), &self.symbol_images())
    }

    /// The denominator of `f` as an expression in the symbols.
    pub fn denom_expr(&self, f: &PiSigmaElement) -> PolyResult<Basic> {
        self.poly_expr(f.denom(), &self.symbol_images())
    }

    // -- extensions

    /// The field with a new generator of the given kind, standing for the
    /// sequence `expr`, and its symbol (`extend`); `symbol` is the
    /// generator (a new `Dummy` named `name` or `t<i>` when `None`).
    pub fn extend(
        &self,
        kind: Kind,
        value: &PiSigmaElement,
        expr: &Basic,
        name: Option<&str>,
        symbol: Option<&Basic>,
    ) -> PolyResult<(PiSigmaField<Cs>, Basic)> {
        if kind == Kind::Pi && !value.is_nonzero() {
            return Err(PolyError::Value("alpha must be nonzero".to_string()));
        }
        let symbol: Basic = match symbol {
            Some(s) => s.clone(),
            None => {
                let default: String = format!("t{}", self.extensions.len() + 1);
                Dummy::new(name.unwrap_or(&default), &[]).0
            }
        };
        let extension = Extension {
            kind,
            symbol: symbol.clone(),
            value: self.to_expr(value, false, false)?,
            expr: expr.clone(),
        };
        let mut extensions: Vec<Extension> = self.extensions.clone();
        extensions.push(extension);
        let mut values: Vec<Rep> = self.values.clone();
        values.push(value.rep.clone());
        Ok((
            PiSigmaField::new(&self.k, self.constants.clone(), extensions, values, Some(self.id))?,
            symbol,
        ))
    }

    /// The field with a Σ-extension `σ(t) = t + β` (`add_sigma`).
    pub fn add_sigma(&self, beta: &Basic, expr: &Basic, name: Option<&str>) -> PolyResult<(PiSigmaField<Cs>, Basic)> {
        self.extend(Kind::Sigma, &self.from_expr(beta)?, expr, name, None)
    }

    /// The field with a Π-extension `σ(t) = α t` (`add_pi`).
    pub fn add_pi(&self, alpha: &Basic, expr: &Basic, name: Option<&str>) -> PolyResult<(PiSigmaField<Cs>, Basic)> {
        self.extend(Kind::Pi, &self.from_expr(alpha)?, expr, name, None)
    }

    // -- the shift

    /// `σ^j(f)` for any integer `j` (`sigma`).
    pub fn sigma(&self, f: &PiSigmaElement, power: i64) -> PolyResult<PiSigmaElement> {
        let inverse: bool = power < 0;
        let images: Vec<Image> = if inverse {
            self.inverse_gens()?
        } else {
            self.images.clone()
        };
        let mut rep: Rep = f.rep.clone();
        for _ in 0..power.unsigned_abs() {
            rep = self.sigma_rep(&rep, &images, inverse)?;
        }
        Ok(self.element(rep))
    }

    fn inverse_gens(&self) -> PolyResult<Vec<Image>> {
        if !self.inverse_images.borrow().is_empty() {
            return Ok(self.inverse_images.borrow().clone());
        }
        let (r, pp) = (&self.ring, self.p);
        let gens: Vec<Poly> = r.gens();
        let mut images: Vec<Image> = vec![(gens[pp].sub(&r.one()), r.one())];
        for (i, (e, v)) in self.extensions.iter().zip(&self.values).enumerate() {
            let w: Rep = self.sigma_rep(v, &images, true)?;
            let t: &Poly = &gens[pp + i + 1];
            images.push(match e.kind {
                Kind::Pi => (t.mul(&w.denom), w.numer.clone()),
                Kind::Sigma => (t.mul(&w.denom).sub(&w.numer), w.denom.clone()),
            });
        }
        *self.inverse_images.borrow_mut() = images.clone();
        Ok(images)
    }

    fn power(&self, images: &[Image], inverse: bool, i: usize, e: u32, numerator: bool) -> Poly {
        let key = (inverse, i, e, numerator);
        if let Some(v) = self.power_cache.borrow().get(&key) {
            return v.clone();
        }
        let image: &Image = &images[i - self.p];
        let value: Poly = (if numerator { &image.0 } else { &image.1 }).pow(u64::from(e));
        self.power_cache.borrow_mut().insert(key, value.clone());
        value
    }

    /// The image of `f` under the substitution of the variables from `k`
    /// on by `images` (`_sigma_rep`).
    fn sigma_rep(&self, f: &Rep, images: &[Image], inverse: bool) -> PolyResult<Rep> {
        let (r, pp) = (&self.ring, self.p);
        let n: usize = r.ngens;
        let one: Poly = r.one();
        let mut degrees: Vec<u32> = vec![0; n];
        for (i, degree) in degrees.iter_mut().enumerate().skip(pp) {
            if i - pp < images.len() && images[i - pp].1 != one {
                *degree = f.numer.degree_in(i).max(f.denom.degree_in(i)).max(0) as u32;
            }
        }
        let mut products: HashMap<Monomial, Poly> = HashMap::new();
        let mut image = |p: &Poly| -> Poly {
            let mut terms: Terms<Rat> = Terms::new();
            for (m, c) in &p.terms {
                let tail: Monomial = m[pp..].to_vec();
                let product: Poly = match products.get(&tail) {
                    Some(product) => product.clone(),
                    None => {
                        let mut product: Poly = one.clone();
                        for i in pp..n {
                            let e: u32 = m[i];
                            if e > 0 {
                                product = product.mul(&self.power(images, inverse, i, e, true));
                            }
                            if degrees[i] > e {
                                product = product.mul(&self.power(images, inverse, i, degrees[i] - e, false));
                            }
                        }
                        products.insert(tail, product.clone());
                        product
                    }
                };
                for (mm, cc) in &product.terms {
                    let key: Monomial = mm
                        .iter()
                        .enumerate()
                        .map(|(j, a)| a + if j < pp { m[j] } else { 0 })
                        .collect();
                    let value: Rat = terms.remove(&key).unwrap_or(Rat::ZERO) + c * cc;
                    if !value.is_zero() {
                        terms.insert(key, value);
                    }
                }
            }
            PolyElement { terms, ring: r.clone() }
        };
        let (numerator, denominator) = (image(&f.numer), image(&f.denom));
        self.field.new_element(&numerator, &denominator)
    }

    /// `σ^j` of a polynomial in `k` and the parameters (`shift_poly`).
    pub fn shift_poly(&self, p: &Poly, j: i64) -> Poly {
        let pp: usize = self.p;
        let mut terms: Terms<Rat> = Terms::new();
        let jj: Rat = rat(j, 1);
        for (m, c) in &p.terms {
            let e: u32 = m[pp];
            for r in 0..=e {
                let mut key: Monomial = m.clone();
                key[pp] = r;
                let factor: Rat = rat_from_int(&binomial(e, r)) * pow_rat(&jj, e - r);
                let value: Rat = terms.remove(&key).unwrap_or(Rat::ZERO) + c * factor;
                if !value.is_zero() {
                    terms.insert(key, value);
                }
            }
        }
        PolyElement {
            terms,
            ring: self.ring.clone(),
        }
    }

    // -- levels and decompositions

    /// The index of the highest generator in `f`: `-1` for a constant, `0`
    /// for an element of `C(k)` (`level_of`).
    pub fn level_of(&self, f: &PiSigmaElement) -> i64 {
        let pp: i64 = self.p as i64;
        let mut level: i64 = pp - 1;
        for p in [f.numer(), f.denom()] {
            for monom in p.terms.keys() {
                for i in ((level + 1)..(monom.len() as i64)).rev() {
                    if monom[i as usize] > 0 {
                        level = i;
                        break;
                    }
                }
            }
        }
        level - pp
    }

    pub fn is_constant(&self, f: &PiSigmaElement) -> bool {
        self.level_of(f) == -1
    }

    /// `f` as a (Laurent) polynomial in the generator of the given level
    /// (`as_poly_in`).
    pub fn as_poly_in(
        &self,
        f: &PiSigmaElement,
        level: usize,
        laurent: bool,
    ) -> PolyResult<Option<Vec<(i64, PiSigmaElement)>>> {
        let r = &self.ring;
        let i: usize = self.p + level;
        let mut den: Poly = f.denom().clone();
        let mut exponents: Vec<u32> = den.terms.keys().map(|m| m[i]).collect();
        exponents.sort();
        exponents.dedup();
        if exponents.len() != 1 {
            return Ok(None);
        }
        let d: u32 = exponents[0];
        let mut shift: i64 = 0;
        if d > 0 {
            if !laurent {
                return Ok(None);
            }
            shift = i64::from(d);
            den = PolyElement {
                terms: den
                    .terms
                    .iter()
                    .map(|(m, c)| {
                        let mut key: Monomial = m.clone();
                        key[i] = 0;
                        (key, c.clone())
                    })
                    .collect(),
                ring: r.clone(),
            };
        }
        let mut parts: Vec<(i64, Terms<Rat>)> = Vec::new();
        for (monom, coeff) in &f.numer().terms {
            let e: i64 = i64::from(monom[i]) - shift;
            let mut key: Monomial = monom.clone();
            key[i] = 0;
            match parts.iter_mut().find(|(x, _)| *x == e) {
                Some((_, t)) => {
                    t.insert(key, coeff.clone());
                }
                None => {
                    let mut t: Terms<Rat> = Terms::new();
                    t.insert(key, coeff.clone());
                    parts.push((e, t));
                }
            }
        }
        let mut result: Vec<(i64, PiSigmaElement)> = Vec::new();
        for (e, terms) in parts {
            let numerator: Poly = PolyElement { terms, ring: r.clone() };
            result.push((e, self.element(self.field.new_element(&numerator, &den)?)));
        }
        Ok(Some(result))
    }

    /// The coefficients of a polynomial in `k` and the parameters, by
    /// degree in `k` (`_k_coefficients`).
    fn k_coefficients(&self, p: &Poly) -> Vec<(u32, Poly)> {
        let pp: usize = self.p;
        let mut parts: Vec<(u32, Terms<Rat>)> = Vec::new();
        for (m, c) in &p.terms {
            let mut key: Monomial = m.clone();
            key[pp] = 0;
            match parts.iter_mut().find(|(d, _)| *d == m[pp]) {
                Some((_, t)) => {
                    t.insert(key, c.clone());
                }
                None => {
                    let mut t: Terms<Rat> = Terms::new();
                    t.insert(key, c.clone());
                    parts.push((m[pp], t));
                }
            }
        }
        parts
            .into_iter()
            .map(|(d, terms)| {
                (
                    d,
                    PolyElement {
                        terms,
                        ring: self.ring.clone(),
                    },
                )
            })
            .collect()
    }

    fn k_coefficient(&self, p: &Poly, d: u32) -> Option<Poly> {
        self.k_coefficients(p)
            .into_iter()
            .find(|(e, _)| *e == d)
            .map(|(_, c)| c)
    }

    /// The irreducible factors of positive degree in `k` over the
    /// constants (`_factors_in_k`).
    fn factors_in_k(&self, p: &Poly) -> PolyResult<Vec<Poly>> {
        let pp: usize = self.p;
        if p.degree_in(pp) <= 0 {
            return Ok(Vec::new());
        }
        let z: Arc<PolyRing<IntegerRing>> = PolyRing::new(
            crate::polys::domains::integerring::ZZ,
            self.ring.names[..=pp].to_vec(),
            crate::polys::rings::MonomialOrder::Lex,
        )?;
        let mut d: Int = Int::ONE;
        for c in p.terms.values() {
            let q: Int = denom(c);
            d = &d / gcd(&d, &q) * &q;
        }
        let cleared: PolyElement<IntegerRing> = PolyElement {
            terms: p
                .terms
                .iter()
                .map(|(m, c)| (m[..=pp].to_vec(), numer(&(c * rat_from_int(&d)))))
                .collect(),
            ring: z,
        };
        let extra: usize = self.ring.ngens - pp - 1;
        let mut result: Vec<Poly> = Vec::new();
        for (factor, _) in zz_factor(&cleared)?.1 {
            if factor.degree_in(pp) > 0 {
                result.push(PolyElement {
                    terms: factor
                        .terms
                        .iter()
                        .map(|(m, c)| {
                            let mut key: Monomial = m.clone();
                            key.extend(vec![0; extra]);
                            (key, rat_from_int(c))
                        })
                        .collect(),
                    ring: self.ring.clone(),
                });
            }
        }
        Ok(result)
    }

    // -- constants

    /// The constant equal to the constant element `f` (`to_constant`).
    pub fn to_constant(&self, f: &PiSigmaElement) -> PolyResult<Elem<Cs>> {
        if !self.is_constant(f) {
            return Err(PolyError::Value(format!(
                "{} is not a constant",
                self.to_expr(f, false, false)?
            )));
        }
        Ok(self.constants.restrict(&f.rep))
    }

    pub fn from_constant(&self, c: &Elem<Cs>) -> PiSigmaElement {
        self.element(self.constants.embed(c, &self.field))
    }

    /// The constant as an expression in the parameters.
    pub fn constant_to_expr(&self, c: &Elem<Cs>) -> PolyResult<Basic> {
        self.to_expr(&self.from_constant(c), false, false)
    }

    fn constant_of(&self, p: &Poly) -> PolyResult<Elem<Cs>> {
        self.constants.from_poly(p)
    }

    fn lc_in_k(&self, p: &Poly) -> PolyResult<(Elem<Cs>, u32)> {
        let coefficients = self.k_coefficients(p);
        let (d, c) = coefficients.iter().max_by_key(|(d, _)| *d).expect("a term");
        Ok((self.constant_of(c)?, *d))
    }

    fn c(&self) -> &Cs::D {
        self.constants.domain()
    }

    fn nullspace(&self, rows: &[Vec<Elem<Cs>>], ncols: usize) -> PolyResult<Vec<Vec<Elem<Cs>>>> {
        let k = self.c();
        if rows.is_empty() {
            return Ok((0..ncols)
                .map(|j| (0..ncols).map(|i| if i == j { k.one() } else { k.zero() }).collect())
                .collect());
        }
        k.nullspace(rows, ncols)
    }

    // -- the solver

    /// Solve `σ(g) - a g = c_1 f_1 + ... + c_r f_r` for `g` in the field of
    /// the given level (the whole field by default): a basis of the
    /// solutions `(c, g)`, `None` when out of scope (`solve`).
    pub fn solve(
        &self,
        a: &PiSigmaElement,
        fs: &[PiSigmaElement],
        level: Option<i64>,
    ) -> PolyResult<Solutions<Elem<Cs>>> {
        let level: i64 = level.unwrap_or(self.level() as i64);
        if level == -1 {
            return self.solve_constants(a, fs);
        }
        if level == 0 {
            return self.solve_rational(a, fs);
        }
        if self.level_of(a) >= level {
            return Ok(None);
        }
        match self.extensions[(level - 1) as usize].kind {
            Kind::Sigma => self.solve_sigma(level as usize, a, fs),
            Kind::Pi => self.solve_pi(level as usize, a, fs),
        }
    }

    /// [`Self::solve`] with the constants as elements of the field.
    pub fn solve_embedded(
        &self,
        a: &PiSigmaElement,
        fs: &[PiSigmaElement],
        level: Option<i64>,
    ) -> PolyResult<Option<Vec<(Vec<PiSigmaElement>, PiSigmaElement)>>> {
        Ok(self.solve(a, fs, level)?.map(|sols| {
            sols.into_iter()
                .map(|(c, g)| (c.iter().map(|ci| self.from_constant(ci)).collect(), g))
                .collect()
        }))
    }

    /// A solution `g` of `σ(g) - g = f`, or `None` (`telescope`).
    pub fn telescope(&self, f: &PiSigmaElement) -> PolyResult<Option<PiSigmaElement>> {
        let Some(sols) = self.solve(&self.one, std::slice::from_ref(f), None)? else {
            return Ok(None);
        };
        for (c, g) in sols {
            if !self.c().is_zero(&c[0]) {
                return Ok(Some(self.div(&g, &self.from_constant(&c[0]))?));
            }
        }
        Ok(None)
    }

    fn solve_constants(&self, a_: &PiSigmaElement, fs_: &[PiSigmaElement]) -> PolyResult<Solutions<Elem<Cs>>> {
        let k = self.c();
        let a: Elem<Cs> = self.to_constant(a_)?;
        let mut fs: Vec<Elem<Cs>> = Vec::new();
        for f in fs_ {
            fs.push(self.to_constant(f)?);
        }
        let s: usize = fs.len();
        let (one, zero) = (k.one(), k.zero());
        if a != one {
            let mut sols = Vec::new();
            for l in 0..s {
                let c: Vec<Elem<Cs>> = (0..s)
                    .map(|i| if i == l { one.clone() } else { zero.clone() })
                    .collect();
                sols.push((c, self.from_constant(&k.exquo(&fs[l], &k.sub(&one, &a))?)));
            }
            return Ok(Some(sols));
        }
        let mut sols: Vec<(Vec<Elem<Cs>>, PiSigmaElement)> = Vec::new();
        if s > 0 {
            for v in self.nullspace(&[fs], s)? {
                sols.push((v, self.zero.clone()));
            }
        }
        sols.push((vec![zero; s], self.one.clone()));
        Ok(Some(sols))
    }

    /// The largest `h >= 0` with `gcd(A(k), B(k + h)) != 1`, or `-1`
    /// (`_dispersion`).
    fn dispersion(&self, a: &Poly, b: &Poly) -> PolyResult<i64> {
        let k = self.c();
        let mut factors_b: Vec<(Poly, Vec<(u32, Poly)>)> = Vec::new();
        for q in self.factors_in_k(b)? {
            let cq = self.k_coefficients(&q);
            factors_b.push((q, cq));
        }
        let mut best: i64 = -1;
        for p in self.factors_in_k(a)? {
            let cp = self.k_coefficients(&p);
            let d: u32 = cp.iter().map(|(e, _)| *e).max().expect("a term");
            let get = |c: &[(u32, Poly)], e: u32| c.iter().find(|(x, _)| *x == e).map(|(_, p)| p.clone());
            let lp: Elem<Cs> = self.constant_of(&get(&cp, d).expect("the leading coefficient"))?;
            let sp: Elem<Cs> = match get(&cp, d.wrapping_sub(1)) {
                Some(c) if d >= 1 => k.exquo(&self.constant_of(&c)?, &lp)?,
                _ => k.zero(),
            };
            for (q, cq) in &factors_b {
                if cq.iter().map(|(e, _)| *e).max() != Some(d) {
                    continue;
                }
                let lq0: Elem<Cs> = self.constant_of(&get(cq, d).expect("the leading coefficient"))?;
                let sq: Elem<Cs> = match get(cq, d.wrapping_sub(1)) {
                    Some(c) if d >= 1 => k.exquo(&self.constant_of(&c)?, &lq0)?,
                    _ => k.zero(),
                };
                let shift: Option<Rat> = self
                    .constants
                    .rational(&k.exquo(&k.sub(&sp, &sq), &k.from_int(i64::from(d)))?);
                let Some(shift) = shift else { continue };
                if !shift.is_int() || rat_is_negative(&shift) {
                    continue;
                }
                let h: i64 = i64::try_from(&numer(&shift)).map_err(|_| PolyError::Value("a huge shift".to_string()))?;
                if h <= best {
                    continue;
                }
                let shifted: Poly = self.shift_poly(q, h);
                let lq: Poly = self.k_coefficient(&shifted, d).expect("the leading coefficient");
                let cpd: Poly = get(&cp, d).expect("the leading coefficient");
                if shifted.mul(&cpd).sub(&p.mul(&lq)).terms.is_empty() {
                    best = h;
                }
            }
        }
        Ok(best)
    }

    /// Abramov's universal denominator (`_universal_denominator`).
    fn universal_denominator(&self, p1: &Poly, p0: &Poly) -> PolyResult<Poly> {
        let ring = &self.field.ring;
        let pp: usize = self.p;
        let mut a: Poly = self.shift_poly(p1, -1);
        let mut b: Poly = p0.clone();
        let mut u: Poly = ring.one();
        let n: i64 = self.dispersion(&a, &b)?;
        for i in (0..=n).rev() {
            let d: Poly = ring.gcd(&a, &self.shift_poly(&b, i));
            if d.degree_in(pp) <= 0 {
                continue;
            }
            a = a.exquo(&d)?;
            b = b.exquo(&self.shift_poly(&d, -i))?;
            for j in 0..=i {
                u = u.mul(&self.shift_poly(&d, -j));
            }
        }
        Ok(u)
    }

    /// The coefficients of a polynomial in `k` as constants by degree
    /// (`_coeff_list`).
    fn coeff_list(&self, p: &Poly) -> PolyResult<Vec<Elem<Cs>>> {
        let coefficients = self.k_coefficients(p);
        let d: i64 = coefficients.iter().map(|(e, _)| i64::from(*e)).max().unwrap_or(-1);
        let mut result: Vec<Elem<Cs>> = Vec::new();
        for i in 0..=d {
            result.push(match coefficients.iter().find(|(e, _)| i64::from(*e) == i) {
                Some((_, c)) => self.constant_of(c)?,
                None => self.c().zero(),
            });
        }
        Ok(result)
    }

    fn solve_rational(&self, a: &PiSigmaElement, fs: &[PiSigmaElement]) -> PolyResult<Solutions<Elem<Cs>>> {
        let k = self.c();
        let ring = &self.field.ring;
        let pp: usize = self.p;
        let (an, bd) = (a.numer().clone(), a.denom().clone());
        let be: PiSigmaElement = self.element(self.field.from_ring(&bd));
        let bfs: Vec<PiSigmaElement> = fs.iter().map(|f| self.mul(&be, f)).collect();
        let mut w: Poly = ring.one();
        for f in &bfs {
            w = ring.lcm(&w, f.denom());
        }
        let we: PiSigmaElement = self.element(self.field.from_ring(&w));
        let mut s: Vec<Poly> = Vec::new();
        let mut scales: Vec<Elem<Cs>> = Vec::new();
        for f in &bfs {
            let wf: PiSigmaElement = self.mul(&we, f);
            if wf.denom().degree_in(pp) > 0 {
                return Err(PolyError::Arithmetic(format!(
                    "denominator not cleared: {}",
                    self.to_expr(&wf, false, false)?
                )));
            }
            s.push(wf.numer().clone());
            scales.push(self.constant_of(wf.denom())?);
        }
        let p1: Poly = w.mul(&bd);
        let p0: Poly = w.mul(&an).neg();
        let u: Poly = self.universal_denominator(&p1, &p0)?;
        let su: Poly = self.shift_poly(&u, 1);
        let (big_p1, big_p0) = (p1.mul(&u), p0.mul(&su));
        let q: Vec<Poly> = s.iter().map(|x| u.mul(&su).mul(x)).collect();
        let (c1, c0) = (self.coeff_list(&big_p1)?, self.coeff_list(&big_p0)?);
        let (d1, d0) = (c1.len() as i64 - 1, c0.len() as i64 - 1);
        let m: i64 = d1.max(d0);
        let mut dq: i64 = -1;
        for x in &q {
            dq = dq.max(self.coeff_list(x)?.len() as i64 - 1);
        }
        let mut degree: i64 = (dq - m).max(0);
        if d1 == d0 && k.is_zero(&k.add(c1.last().expect("a coefficient"), c0.last().expect("a coefficient"))) {
            degree = (dq - m + 1).max(0);
            let lower: Elem<Cs> = if m >= 1 {
                k.add(&c1[(m - 1) as usize], &c0[(m - 1) as usize])
            } else {
                k.zero()
            };
            let root: Option<Rat> = self
                .constants
                .rational(&k.neg(&k.exquo(&lower, c1.last().expect("a coefficient"))?));
            if let Some(root) = root {
                if root.is_int() && !rat_is_negative(&root) {
                    degree = degree
                        .max(i64::try_from(&numer(&root)).map_err(|_| PolyError::Value("a huge degree".to_string()))?);
                }
            }
        }
        let x: Poly = self.ring.gens()[pp].clone();
        let mut columns: Vec<Vec<Elem<Cs>>> = Vec::new();
        for j in 0..=degree {
            let power: Poly = x.pow(j as u64);
            columns.push(self.coeff_list(&big_p1.mul(&self.shift_poly(&power, 1)).add(&big_p0.mul(&power)))?);
        }
        for qq in &q {
            columns.push(self.coeff_list(qq)?.iter().map(|c| k.neg(c)).collect());
        }
        let nrows: usize = columns.iter().map(|c| c.len()).max().unwrap_or(0);
        let ncols: usize = columns.len();
        let rows: Vec<Vec<Elem<Cs>>> = (0..nrows)
            .map(|r| {
                columns
                    .iter()
                    .map(|col| if r < col.len() { col[r].clone() } else { k.zero() })
                    .collect()
            })
            .collect();
        let ue: PiSigmaElement = self.element(self.field.from_ring(&u));
        let mut sols: Vec<(Vec<Elem<Cs>>, PiSigmaElement)> = Vec::new();
        for v in self.nullspace(&rows, ncols)? {
            let mut p: PiSigmaElement = self.zero.clone();
            for j in 0..=degree {
                if !k.is_zero(&v[j as usize]) {
                    let term: PiSigmaElement =
                        self.mul(&self.from_constant(&v[j as usize]), &self.pow(&self.gens[0], j)?);
                    p = self.add(&p, &term);
                }
            }
            let constants: Vec<Elem<Cs>> = (0..s.len())
                .map(|l| k.mul(&v[(degree + 1) as usize + l], &scales[l]))
                .collect();
            sols.push((constants, self.div(&p, &ue)?));
        }
        Ok(Some(sols))
    }

    fn is_sigma_ratio(&self, a: &PiSigmaElement, level: i64) -> PolyResult<bool> {
        if *a == self.one {
            return Ok(true);
        }
        match self.solve(a, &[], Some(level))? {
            Some(sols) if !sols.is_empty() => Ok(sols.iter().any(|(_, g)| *g != self.zero)),
            _ => Ok(false),
        }
    }

    fn combination(&self, coefficients: &[Elem<Cs>], elements: &[PiSigmaElement]) -> PiSigmaElement {
        let mut result: PiSigmaElement = self.zero.clone();
        for (c, e) in coefficients.iter().zip(elements) {
            if !self.c().is_zero(c) {
                result = self.add(&result, &self.mul(&self.from_constant(c), e));
            }
        }
        result
    }

    fn normalize_basis(
        &self,
        cmap: &mut [Vec<Elem<Cs>>],
        g: &mut HashMap<i64, Vec<PiSigmaElement>>,
        s: usize,
        r: usize,
    ) -> PolyResult<()> {
        let k = self.c();
        for m in 0..s {
            let Some(pivot) = (0..r).map(|i| cmap[i][m].clone()).find(|c| !k.is_zero(c)) else {
                continue;
            };
            if k.is_one(&pivot) {
                continue;
            }
            let inverse: Elem<Cs> = k.exquo(&k.one(), &pivot)?;
            for row in cmap.iter_mut().take(r) {
                row[m] = k.mul(&row[m], &inverse);
            }
            let scale: PiSigmaElement = self.from_constant(&inverse);
            for coeffs in g.values_mut() {
                coeffs[m] = self.mul(&coeffs[m], &scale);
            }
        }
        Ok(())
    }

    /// The common driver of the Σ and Π cases (`_solve_steps`).
    fn solve_steps(
        &self,
        level: usize,
        steps: Vec<(i64, PiSigmaElement, Rhs<'_, Elem<Cs>>)>,
        r: usize,
    ) -> PolyResult<Solutions<Elem<Cs>>> {
        let k = self.c();
        let (one, zero) = (k.one(), k.zero());
        let mut s: usize = r;
        let mut cmap: Vec<Vec<Elem<Cs>>> = (0..r)
            .map(|i| {
                (0..s)
                    .map(|l| if i == l { one.clone() } else { zero.clone() })
                    .collect()
            })
            .collect();
        let mut g: HashMap<i64, Vec<PiSigmaElement>> = HashMap::new();
        for (j, a_j, rhs) in steps {
            let right: Vec<PiSigmaElement> = rhs(&cmap, &g, s)?;
            let Some(sols) = self.solve(&a_j, &right, Some(level as i64 - 1))? else {
                return Ok(None);
            };
            let p: usize = sols.len();
            let v: Vec<Vec<Elem<Cs>>> = sols.iter().map(|(c, _)| c.clone()).collect();
            g.insert(j, sols.into_iter().map(|(_, e)| e).collect());
            let keys: Vec<i64> = g.keys().copied().filter(|jj| *jj != j).collect();
            for jj in keys {
                let old: Vec<PiSigmaElement> = g[&jj].clone();
                let new: Vec<PiSigmaElement> = (0..p)
                    .map(|m| self.combination(&(0..s).map(|l| v[m][l].clone()).collect::<Vec<_>>(), &old))
                    .collect();
                g.insert(jj, new);
            }
            let mut new_cmap: Vec<Vec<Elem<Cs>>> = Vec::new();
            for row in cmap.iter().take(r) {
                let mut out: Vec<Elem<Cs>> = Vec::new();
                for vm in v.iter().take(p) {
                    let mut total: Elem<Cs> = zero.clone();
                    for l in 0..s {
                        if !k.is_zero(&row[l]) && !k.is_zero(&vm[l]) {
                            total = k.add(&total, &k.mul(&row[l], &vm[l]));
                        }
                    }
                    out.push(total);
                }
                new_cmap.push(out);
            }
            cmap = new_cmap;
            s = p;
            self.normalize_basis(&mut cmap, &mut g, s, r)?;
        }
        let t: &PiSigmaElement = &self.gens[level];
        let mut result: Vec<(Vec<Elem<Cs>>, PiSigmaElement)> = Vec::new();
        for m in 0..s {
            let mut total: PiSigmaElement = self.zero.clone();
            for (j, coeffs) in &g {
                total = self.add(&total, &self.mul(&coeffs[m], &self.pow(t, *j)?));
            }
            result.push(((0..r).map(|i| cmap[i][m].clone()).collect(), total));
        }
        Ok(Some(result))
    }

    fn solve_sigma(&self, level: usize, a: &PiSigmaElement, fs: &[PiSigmaElement]) -> PolyResult<Solutions<Elem<Cs>>> {
        let beta: PiSigmaElement = self.element(self.values[level - 1].clone());
        let mut polys: Vec<Vec<(i64, PiSigmaElement)>> = Vec::new();
        for f in fs {
            let Some(p) = self.as_poly_in(f, level, false)? else {
                return Ok(None);
            };
            polys.push(p);
        }
        let maxdeg: i64 = polys
            .iter()
            .flat_map(|p| p.iter().map(|(e, _)| *e))
            .max()
            .unwrap_or(0)
            .max(0);
        let degree: i64 = maxdeg
            + if self.is_sigma_ratio(a, level as i64 - 1)? {
                1
            } else {
                0
            };
        let r: usize = fs.len();
        let betas: RefCell<HashMap<i64, PiSigmaElement>> = RefCell::new(HashMap::new());
        let polys = &polys;
        let beta = &beta;
        let betas = &betas;
        let mut steps: Vec<(i64, PiSigmaElement, Rhs<'_, Elem<Cs>>)> = Vec::new();
        for j in (0..=degree).rev() {
            let rhs: Rhs<'_, Elem<Cs>> = Box::new(move |cmap, g, s| {
                let k = self.c();
                let mut right: Vec<PiSigmaElement> = Vec::new();
                for l in 0..s {
                    let mut value: PiSigmaElement = self.zero.clone();
                    for i in 0..r {
                        if let Some((_, fij)) = polys[i].iter().find(|(e, _)| *e == j) {
                            if !k.is_zero(&cmap[i][l]) {
                                value = self.add(&value, &self.mul(&self.from_constant(&cmap[i][l]), fij));
                            }
                        }
                    }
                    for (jj, coeffs) in g {
                        if *jj > j {
                            let gap: i64 = jj - j;
                            if !betas.borrow().contains_key(&gap) {
                                let power: PiSigmaElement = self.pow(beta, gap)?;
                                betas.borrow_mut().insert(gap, power);
                            }
                            let factor: PiSigmaElement = betas.borrow()[&gap].clone();
                            let shifted: PiSigmaElement = self.sigma(&coeffs[l], 1)?;
                            let term: PiSigmaElement =
                                self.scale_int(&binomial(*jj as u32, j as u32), &self.mul(&factor, &shifted));
                            value = self.sub(&value, &term);
                        }
                    }
                    right.push(value);
                }
                Ok(right)
            });
            steps.push((j, a.clone(), rhs));
        }
        self.solve_steps(level, steps, r)
    }

    fn pi_exponents(&self, a: &PiSigmaElement, alpha: &PiSigmaElement) -> PolyResult<Vec<i64>> {
        let k = self.c();
        if *a == self.one {
            return Ok(Vec::new());
        }
        if self.level_of(a) > 0 || self.level_of(alpha) > 0 {
            return Ok((-3..4).collect());
        }
        let lc_and_degree = |f: &PiSigmaElement| -> PolyResult<(Elem<Cs>, i64)> {
            let (ln, dn) = self.lc_in_k(f.numer())?;
            let (ld, dd) = self.lc_in_k(f.denom())?;
            Ok((k.exquo(&ln, &ld)?, i64::from(dn) - i64::from(dd)))
        };
        let (la, da) = lc_and_degree(a)?;
        let (lal, dal) = lc_and_degree(alpha)?;
        if dal != 0 {
            if da % dal == 0 {
                return Ok(vec![da / dal]);
            }
            return Ok(Vec::new());
        }
        if lal != k.one() && lal != k.neg(&k.one()) {
            let mut power: Elem<Cs> = k.one();
            for j in 0..40 {
                if power == la {
                    return Ok(vec![j]);
                }
                power = k.mul(&power, &lal);
            }
            let mut power: Elem<Cs> = k.one();
            for j in 0..40 {
                if power == la {
                    return Ok(vec![-j]);
                }
                power = k.exquo(&power, &lal)?;
            }
            return Ok(Vec::new());
        }
        Ok((-3..4).collect())
    }

    fn solve_pi(&self, level: usize, a: &PiSigmaElement, fs: &[PiSigmaElement]) -> PolyResult<Solutions<Elem<Cs>>> {
        let alpha: PiSigmaElement = self.element(self.values[level - 1].clone());
        let mut polys: Vec<Vec<(i64, PiSigmaElement)>> = Vec::new();
        for f in fs {
            let Some(p) = self.as_poly_in(f, level, true)? else {
                return Ok(None);
            };
            polys.push(p);
        }
        let mut exponents: Vec<i64> = vec![0];
        for p in &polys {
            exponents.extend(p.iter().map(|(e, _)| *e));
        }
        exponents.extend(self.pi_exponents(a, &alpha)?);
        exponents.sort();
        exponents.dedup();
        exponents.reverse();
        let r: usize = fs.len();
        let polys = &polys;
        let mut steps: Vec<(i64, PiSigmaElement, Rhs<'_, Elem<Cs>>)> = Vec::new();
        for j in exponents {
            let alpha_j: PiSigmaElement = self.pow(&alpha, j)?;
            let a_j: PiSigmaElement = self.div(a, &alpha_j)?;
            let rhs: Rhs<'_, Elem<Cs>> = Box::new(move |cmap, _g, s| {
                let k = self.c();
                let mut right: Vec<PiSigmaElement> = Vec::new();
                for l in 0..s {
                    let mut value: PiSigmaElement = self.zero.clone();
                    for i in 0..r {
                        if let Some((_, fij)) = polys[i].iter().find(|(e, _)| *e == j) {
                            if !k.is_zero(&cmap[i][l]) {
                                value = self.add(&value, &self.mul(&self.from_constant(&cmap[i][l]), fij));
                            }
                        }
                    }
                    right.push(self.div(&value, &alpha_j)?);
                }
                Ok(right)
            });
            steps.push((j, a_j, rhs));
        }
        self.solve_steps(level, steps, r)
    }
}

fn binomial(n: u32, k: u32) -> Int {
    let mut c: Int = Int::ONE;
    for j in 0..k {
        c = c * Int::from(n - j) / Int::from(j + 1);
    }
    c
}

fn pow_rat(x: &Rat, e: u32) -> Rat {
    (0..e).fold(Rat::ONE, |acc, _| acc * x)
}

/// The field `QQ(k)` with the shift `k -> k + 1`.
pub fn rational_pisigma_field(k: &Basic) -> PolyResult<PiSigmaField<RationalConstants>> {
    PiSigmaField::new(k, RationalConstants, Vec::new(), Vec::new(), None)
}

/// The field `QQ(params)(k)` with the shift `k -> k + 1`.
pub fn parametric_pisigma_field(k: &Basic, params: &[Basic]) -> PolyResult<PiSigmaField<ParametricConstants>> {
    PiSigmaField::new(k, ParametricConstants::new(params)?, Vec::new(), Vec::new(), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::core::Symbol;

    #[test]
    fn telescoping() {
        let k: Basic = Symbol::new("k", &[]).0;
        let f = rational_pisigma_field(&k).unwrap();
        // sum of k is k*(k - 1)/2
        let g = f.telescope(&f.gens[0]).unwrap().unwrap();
        assert_eq!(sstr(&f.to_expr(&g, true, false).unwrap()), "k*(k - 1)/2");
        // a geometric term 2**k telescopes to 2**k
        let two_k: Basic = pow(&Basic::int(2), &k).unwrap();
        let (f2, t) = f.add_pi(&Basic::int(2), &two_k, None).unwrap();
        let g2 = f2.telescope(&f2.from_expr(&t).unwrap()).unwrap().unwrap();
        assert_eq!(sstr(&f2.to_expr(&g2, true, false).unwrap()), "2**k");
    }
}
