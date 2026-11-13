//! Port of the decompositions of `sympy_extras/core/expr.py` and of their
//! overrides in the classes of the core (`as_coeff_Mul`, `as_base_exp`,
//! `terms_of`, `factors_of`), and the properties of the facts (`is_real`,
//! `is_positive`, ...: the structural facts of [`super::assumptions`], as
//! Python's `Expr._ask`).

use crate::numbers::*;

use super::add::add;
use super::assumptions::{ask_structural, Predicate};
use super::basic::{Basic, Head};
use super::mul::{mul, raw_mul};
use super::power::pow;
use crate::error::PolyResult;

/// A fact of the node: `Some(true)`, `Some(false)` or unknown.
pub fn is(e: &Basic, p: Predicate) -> Option<bool> {
    ask_structural(e, p)
}

/// Whether the fact is known to hold (`e.is_xxx is True`).
pub fn is_true(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(true)
}

/// Whether the fact is known to fail (`e.is_xxx is False`).
pub fn is_false(e: &Basic, p: Predicate) -> bool {
    is(e, p) == Some(false)
}

/// Whether `e` is the rational number `value` (`Integer` or `Rational`).
pub fn is_rational_value(e: &Basic, value: i64) -> bool {
    e.as_rational().is_some_and(|r| *r == rat(value, 1))
}

/// Whether `e` is the integer zero (`isinstance(e, Integer) and e.p == 0`,
/// a `Rational` zero included: its value decides).
pub fn is_zero_number(e: &Basic) -> bool {
    e.as_rational().is_some_and(|r| r.is_zero())
}

/// Whether the node is of Python's class `Integer` (a `Rational` with
/// denominator one is not).
pub fn is_integer_class(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Rational {
            integer_class: true,
            ..
        }
    )
}

/// The terms of a sum, `[e]` for anything else.
pub fn terms_of(e: &Basic) -> Vec<Basic> {
    if e.is_add() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// The factors of a product, `[e]` for anything else.
pub fn factors_of(e: &Basic) -> Vec<Basic> {
    if e.is_mul() {
        e.args().to_vec()
    } else {
        vec![e.clone()]
    }
}

/// The numerical coefficient and the rest of a product (`as_coeff_Mul`):
/// a number is `(self, 1)` (`-oo` is `(-1, oo)`), a product whose first
/// argument is a number other than `zoo` splits it off, anything else is
/// `(1, self)`.
pub fn as_coeff_mul(e: &Basic) -> (Basic, Basic) {
    match e.head() {
        Head::NegativeInfinity => (Basic::negative_one(), Basic::oo()),
        _ if e.is_number_node() => (e.clone(), Basic::one()),
        Head::Mul => {
            let first: &Basic = &e.args()[0];
            if first.is_number_node() && !matches!(first.head(), Head::ComplexInfinity) {
                (first.clone(), raw_mul(e.args()[1..].to_vec()))
            } else {
                (Basic::one(), e.clone())
            }
        }
        _ => (Basic::one(), e.clone()),
    }
}

/// The numerical term and the rest of a sum (`as_coeff_Add`): the numbers
/// among the terms of a sum added and the other terms, a number is
/// `(self, 0)`, anything else `(0, self)`.
pub fn as_coeff_add(e: &Basic) -> (Basic, Basic) {
    if e.is_number_class() {
        return (e.clone(), Basic::zero());
    }
    if !e.is_add() {
        return (Basic::zero(), e.clone());
    }
    let mut number: Basic = Basic::zero();
    let mut rest: Vec<Basic> = Vec::new();
    for a in e.args() {
        if a.is_number_class() {
            number = super::numbers::number_add(&number, a);
        } else {
            rest.push(a.clone());
        }
    }
    (number, super::add::raw_add(rest))
}

/// `as_coeff_mul` (rational coefficients only): the coefficient and the
/// tuple of the other factors.
pub fn as_coeff_mul_tuple(e: &Basic) -> (Basic, Vec<Basic>) {
    if e.is_number_class() {
        if e.as_rational().is_some() {
            return (e.clone(), Vec::new());
        }
        if is_true(e, Predicate::negative) {
            return (
                Basic::negative_one(),
                vec![super::mul::neg(e).expect("the negation of a number")],
            );
        }
        return (Basic::one(), vec![e.clone()]);
    }
    if e.is_mul() {
        let first: &Basic = &e.args()[0];
        if first.is_number_class() {
            if first.as_rational().is_some() {
                return (first.clone(), e.args()[1..].to_vec());
            }
            if is_true(first, Predicate::extended_negative) {
                let mut rest: Vec<Basic> = vec![super::mul::neg(first).expect("the negation of a number")];
                rest.extend(e.args()[1..].iter().cloned());
                return (Basic::negative_one(), rest);
            }
        }
        return (Basic::one(), e.args().to_vec());
    }
    (Basic::one(), vec![e.clone()])
}

/// The base and the exponent (`as_base_exp`): `(1/q)**e` is `q**(-e)`, a
/// product of powers with one integer exponent is `(product, exponent)`,
/// `I` is `(-1, 1/2)`, anything else is `(self, 1)`.
pub fn as_base_exp(e: &Basic) -> Result<(Basic, Basic), crate::error::PolyError> {
    match e.head() {
        Head::Pow => {
            let (base, exp) = (&e.args()[0], &e.args()[1]);
            if let Some(b) = base.as_rational() {
                if b.numerator().is_one() && denom(b) > Int::ONE {
                    return Ok((Basic::integer(denom(b)), mul(&[Basic::negative_one(), exp.clone()])?));
                }
            }
            Ok((base.clone(), exp.clone()))
        }
        Head::Mul => {
            // x**2*y**2 is (x*y)**2 when every factor has the same integer exponent
            let mut pairs: Vec<(Basic, Basic)> = Vec::new();
            for f in e.args() {
                pairs.push(as_base_exp(f)?);
            }
            let exponent: &Basic = &pairs[0].1;
            if pairs.iter().all(|(_, x)| x == exponent) {
                if let Some(r) = exponent.as_rational() {
                    if r.is_int() && *r != Rat::ONE {
                        let bases: Vec<Basic> = pairs.iter().map(|(b, _)| b.clone()).collect();
                        return Ok((mul(&bases)?, exponent.clone()));
                    }
                }
            }
            Ok((e.clone(), Basic::one()))
        }
        Head::ImaginaryUnit => Ok((Basic::negative_one(), Basic::half())),
        Head::Function(super::basic::Function::Exp) => Ok((Basic::e(), e.args()[0].clone())),
        _ => Ok((e.clone(), Basic::one())),
    }
}

/// The coefficient of `x` among the terms of a sum (`coeff` with `n = 1`).
pub fn coeff(e: &Basic, x: &Basic) -> PolyResult<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for term in terms_of(e) {
        let mut factors: Vec<Basic> = factors_of(&term);
        if let Some(k) = factors.iter().position(|f| f == x) {
            factors.remove(k);
            found.push(mul(&factors)?);
        }
    }
    add(&found)
}

/// The value of a rational number (`exact_value`; `None` for the others).
fn exact_value(e: &Basic) -> Option<&Rat> {
    e.as_rational()
}

/// `gcd` of two fractions: the gcd of the numerators over the lcm of the
/// denominators (`_gcd_fractions`).
fn gcd_fractions(a: &Rat, b: &Rat) -> Rat {
    let (an, ad) = (numer(a), denom(a));
    let (bn, bd) = (numer(b), denom(b));
    let n: Int = gcd(&an, &bn);
    let g: Int = gcd(&ad, &bd);
    let l: Int = if g.is_zero() { Int::ZERO } else { &ad / &g * &bd };
    rat_from_parts(n, l)
}

/// The positive rational content of the terms of a sum and the rest
/// (`Add.primitive`; `(1, e)` for anything else).
pub fn primitive(e: &Basic) -> PolyResult<(Basic, Basic)> {
    if !e.is_add() {
        return Ok((Basic::one(), e.clone()));
    }
    let mut g: Rat = Rat::ZERO;
    for a in e.args() {
        let (c, _) = as_coeff_mul(a);
        let Some(v) = exact_value(&c) else {
            return Ok((Basic::one(), e.clone()));
        };
        let magnitude: Rat = if rat_is_negative(v) { -v.clone() } else { v.clone() };
        g = gcd_fractions(&g, &magnitude);
    }
    if g.is_zero() || g == Rat::ONE {
        return Ok((Basic::one(), e.clone()));
    }
    let inverse: Basic = Basic::from_rational(recip(&g));
    let mut terms: Vec<Basic> = Vec::new();
    for a in e.args() {
        terms.push(mul(&[inverse.clone(), a.clone()])?);
    }
    Ok((Basic::from_rational(g), add(&terms)?))
}

/// `e / c` when `c` can be taken out without a change of form, else `None`
/// (`extract_multiplicatively`, SymPy's rules).
pub fn extract_multiplicatively(e: &Basic, other: &Basic) -> PolyResult<Option<Basic>> {
    if other.is_number_class() && is_rational_value(other, 1) {
        return Ok(Some(e.clone()));
    }
    if e == other {
        return Ok(Some(Basic::one()));
    }
    if e.is_number_class() && is_zero_number(e) {
        return Ok(Some(e.clone()));
    }
    if other.is_mul() {
        let mut result: Basic = e.clone();
        for f in other.args() {
            match extract_multiplicatively(&result, f)? {
                Some(r) => result = r,
                None => return Ok(None),
            }
        }
        return Ok(Some(result));
    }
    if e.is_add() {
        let (content, primitive_part) = primitive(e)?;
        if !is_rational_value(&content, 1) {
            let product: Basic = Basic::new(Head::Mul, vec![content, primitive_part]);
            return extract_multiplicatively(&product, other);
        }
        if other.as_rational().is_some_and(|v| denom(v) != Int::ONE) {
            return Ok(None);
        }
        let mut parts: Vec<Basic> = Vec::new();
        for t in e.args() {
            match extract_multiplicatively(t, other)? {
                Some(r) => parts.push(r),
                None => return Ok(None),
            }
        }
        return Ok(Some(add(&parts)?));
    }
    let (coefficient, rest) = as_coeff_mul(e);
    if other.is_number_class() {
        if matches!(coefficient.head(), Head::Infinity | Head::NegativeInfinity) {
            if is_true(other, Predicate::positive) {
                return Ok(Some(e.clone()));
            }
            if !is_true(other, Predicate::negative) {
                return Ok(None);
            }
            return Ok(Some(mul(&[Basic::negative_one(), e.clone()])?));
        }
        let (Some(vc), Some(vo)) = (exact_value(&coefficient), exact_value(other)) else {
            return Ok(None);
        };
        if vo.is_zero() {
            return Ok(None);
        }
        let integers: bool = vc.is_int() && vo.is_int();
        if (rat_is_negative(vo) && *vc > Rat::ZERO) || (integers && !mod_floor(&numer(vc), &numer(vo)).is_zero()) {
            if e.is_mul() {
                for (i, f) in e.args().iter().enumerate() {
                    if f.is_add() {
                        if let Some(reduced) = extract_multiplicatively(f, other)? {
                            let mut factors: Vec<Basic> = e.args()[..i].to_vec();
                            factors.push(reduced);
                            factors.extend(e.args()[i + 1..].iter().cloned());
                            return Ok(Some(mul(&factors)?));
                        }
                    }
                }
            }
            return Ok(None);
        }
        let quotient: Basic = Basic::from_rational(vc / vo);
        if rest.is_number_class() && is_rational_value(&rest, 1) {
            return Ok(Some(quotient));
        }
        return Ok(Some(mul(&[quotient, rest])?));
    }
    if e.is_pow() && other.is_pow() && e.args()[0] == other.args()[0] {
        let (n0, n1) = as_coeff_mul(&e.args()[1]);
        let (d0, d1) = as_coeff_mul(&other.args()[1]);
        if n1 == d1 {
            if let (Some(a), Some(b)) = (n0.as_rational(), d0.as_rational()) {
                let left: Rat = a - b;
                if !left.is_zero() && (left > Rat::ZERO) == (*b > Rat::ZERO) {
                    return Ok(Some(pow(&e.args()[0], &mul(&[Basic::from_rational(left), n1])?)?));
                }
            }
        }
        return Ok(None);
    }
    if e.is_pow() && e.args()[0] == *other {
        let (c, r) = as_coeff_mul(&e.args()[1]);
        if let Some(v) = c.as_rational() {
            if *v > Rat::ONE && r.is_number_class() && is_rational_value(&r, 1) {
                return Ok(Some(pow(&e.args()[0], &Basic::from_rational(v - Rat::ONE))?));
            }
        }
        return Ok(None);
    }
    if e.is_mul() {
        let factors: &[Basic] = e.args();
        for (i, f) in factors.iter().enumerate() {
            let rebuilt = |middle: Option<Basic>| -> PolyResult<Basic> {
                let mut v: Vec<Basic> = factors[..i].to_vec();
                v.extend(middle);
                v.extend(factors[i + 1..].iter().cloned());
                mul(&v)
            };
            if f == other {
                return Ok(Some(rebuilt(None)?));
            }
            if f.is_pow() || f.is_add() || other.is_pow() {
                if let Some(reduced) = extract_multiplicatively(f, other)? {
                    return Ok(Some(rebuilt(Some(reduced))?));
                }
            }
        }
        return Ok(None);
    }
    Ok(None)
}

/// The coefficient of `x` when `e` is `x` times something free of `x`
/// (`as_coefficient`).
pub fn as_coefficient(e: &Basic, x: &Basic) -> PolyResult<Option<Basic>> {
    Ok(extract_multiplicatively(e, x)?.filter(|r| !r.has(x)))
}

/// The part free of `pattern` and the part containing it, of a sum
/// (`as_add`) or of a product (`as_independent` with a non-symbol pattern:
/// the dependence is `has`).
pub fn as_independent(e: &Basic, pattern: &Basic, as_add: bool) -> PolyResult<(Basic, Basic)> {
    if e.is_number_class() && is_zero_number(e) {
        return Ok((e.clone(), e.clone()));
    }
    if as_add {
        if !e.is_add() {
            return Ok(if !e.has(pattern) {
                (e.clone(), Basic::zero())
            } else {
                (Basic::zero(), e.clone())
            });
        }
        let free: Vec<Basic> = e.args().iter().filter(|a| !a.has(pattern)).cloned().collect();
        let dependent: Vec<Basic> = e.args().iter().filter(|a| a.has(pattern)).cloned().collect();
        return Ok((add(&free)?, add(&dependent)?));
    }
    if !e.is_mul() {
        return Ok(if !e.has(pattern) {
            (e.clone(), Basic::one())
        } else {
            (Basic::one(), e.clone())
        });
    }
    let free: Vec<Basic> = e.args().iter().filter(|a| !a.has(pattern)).cloned().collect();
    let dependent: Vec<Basic> = e.args().iter().filter(|a| a.has(pattern)).cloned().collect();
    Ok((mul(&free)?, mul(&dependent)?))
}

/// Whether `e` is a real number which can be compared (`is_comparable`):
/// `oo`, `-oo`, or a number with an exactly zero imaginary part at 64 bits.
pub fn is_comparable(e: &Basic) -> bool {
    if matches!(e.head(), Head::Infinity | Head::NegativeInfinity) {
        return true;
    }
    if !super::assumptions::is_number(e) {
        return false;
    }
    match super::evalf::numeric_value(e, 64) {
        Some(v) => super::evalf::parts(&v).1.repr().significand().is_zero(),
        None => false,
    }
}

/// A numerator and a denominator whose quotient is `e` (`as_numer_denom`
/// of the classes of the core).
pub fn as_numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    match e.head() {
        Head::Rational { value, .. } => Ok((Basic::integer(numer(value)), Basic::integer(denom(value)))),
        _ if e.is_number_class() => Ok((e.clone(), Basic::one())),
        Head::Add => add_numer_denom(e),
        Head::Mul => {
            let mut ns: Vec<Basic> = Vec::new();
            let mut ds: Vec<Basic> = Vec::new();
            for a in e.args() {
                let (n, d) = as_numer_denom(a)?;
                ns.push(n);
                ds.push(d);
            }
            Ok((mul(&ns)?, mul(&ds)?))
        }
        Head::Pow => pow_numer_denom(e),
        Head::Function(super::basic::Function::Exp) => {
            let arg: &Basic = &e.args()[0];
            if is_true(arg, Predicate::negative) || super::printing::could_extract_minus_sign(arg) {
                let inverse: Basic =
                    crate::functions::elementary::exponential::exp(&mul(&[Basic::negative_one(), arg.clone()])?)?;
                return Ok((Basic::one(), inverse));
            }
            Ok((e.clone(), Basic::one()))
        }
        _ => Ok((e.clone(), Basic::one())),
    }
}

/// `Add.as_numer_denom`: the terms with the same denominator summed first.
fn add_numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    let (content, primitive_part) = primitive(e)?;
    if !primitive_part.is_add() {
        return as_numer_denom(&mul(&[content, primitive_part])?);
    }
    // the denominators with their numerators, in order (a Python dict)
    let mut groups: Vec<(Basic, Vec<Basic>)> = Vec::new();
    for a in primitive_part.args() {
        let (n, d) = as_numer_denom(a)?;
        match groups.iter_mut().find(|(g, _)| *g == d) {
            Some((_, ns)) => ns.push(n),
            None => groups.push((d, vec![n])),
        }
    }
    let (cn, cd) = as_numer_denom(&content)?;
    if groups.len() == 1 {
        let (d, ns) = &groups[0];
        let _guard = super::parameters::distribute_guard(false);
        let mut terms: Vec<Basic> = Vec::new();
        for n in ns {
            terms.push(mul(&[cn.clone(), n.clone()])?);
        }
        return Ok((add(&terms)?, mul(&[cd, d.clone()])?));
    }
    let mut numerators: Vec<Basic> = Vec::new();
    for (i, (_, ns)) in groups.iter().enumerate() {
        let mut factors: Vec<Basic> = vec![add(ns)?];
        factors.extend(
            groups
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, (d, _))| d.clone()),
        );
        numerators.push(mul(&factors)?);
    }
    let _guard = super::parameters::distribute_guard(false);
    let mut denominators: Vec<Basic> = vec![cd];
    denominators.extend(groups.iter().map(|(d, _)| d.clone()));
    Ok((mul(&[cn, add(&numerators)?])?, mul(&denominators)?))
}

/// `Pow.as_numer_denom`.
fn pow_numer_denom(e: &Basic) -> PolyResult<(Basic, Basic)> {
    if is_false(e, Predicate::commutative) {
        return Ok((e.clone(), Basic::one()));
    }
    let (base, mut exp) = as_base_exp(e)?;
    let (mut n, mut d) = as_numer_denom(&base)?;
    let mut neg_exp: bool = is_true(&exp, Predicate::negative);
    if exp.is_mul() && !neg_exp && !is_true(&exp, Predicate::positive) {
        neg_exp = super::printing::could_extract_minus_sign(&exp);
    }
    let int_exp: bool = is_true(&exp, Predicate::integer);
    if !(is_true(&d, Predicate::extended_real) || int_exp) {
        (n, d) = (base.clone(), Basic::one());
    }
    match is(&d, Predicate::nonpositive) {
        Some(true) => {
            n = mul(&[Basic::negative_one(), n])?;
            d = mul(&[Basic::negative_one(), d])?;
        }
        None if !int_exp => (n, d) = (base.clone(), Basic::one()),
        _ => {}
    }
    if neg_exp {
        (n, d) = (d, n);
        exp = mul(&[Basic::negative_one(), exp])?;
    }
    if is_true(&exp, Predicate::infinite) {
        let one = |x: &Basic| is_integer_class(x) && is_rational_value(x, 1);
        if one(&n) && !one(&d) {
            return Ok((n, pow(&d, &exp)?));
        }
        if !one(&n) && one(&d) {
            return Ok((pow(&n, &exp)?, d));
        }
    }
    Ok((pow(&n, &exp)?, pow(&d, &exp)?))
}

/// What [`polynomial_like`] tests: `is_polynomial`, `is_rational_function`
/// or `is_algebraic_expr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialKind {
    Polynomial,
    Rational,
    Algebraic,
}

/// SymPy's three-valued answer of `is_polynomial(*syms)` and its kin
/// (`_is_polynomial_like`): `None` for a function of the symbols, `all` of
/// the arguments of a sum or a product (`None` counting as `False`), and
/// for a power the answer of the base combined with the exponent.
pub fn polynomial_like(e: &Basic, syms: &[Basic], kind: PolynomialKind) -> Option<bool> {
    let free = e.free_symbols();
    if !free.iter().any(|s| syms.contains(s)) {
        return Some(true);
    }
    if e.is_symbol() {
        return Some(true);
    }
    if e.is_add() || e.is_mul() {
        return Some(e.args().iter().all(|a| polynomial_like(a, syms, kind) == Some(true)));
    }
    if e.is_pow() {
        let exponent: &Basic = &e.args()[1];
        if exponent.free_symbols().iter().any(|s| syms.contains(s)) {
            return Some(false);
        }
        let base: Option<bool> = polynomial_like(&e.args()[0], syms, kind);
        let rational: Option<&crate::numbers::Rat> = exponent.as_rational();
        if kind == PolynomialKind::Polynomial {
            return Some(
                base == Some(true) && rational.is_some_and(|r| r.is_int() && !crate::numbers::rat_is_negative(r)),
            );
        }
        if base != Some(true) {
            return base;
        }
        if kind == PolynomialKind::Rational {
            return Some(rational.is_some_and(|r| r.is_int()));
        }
        return Some(rational.is_some());
    }
    None
}

/// The facts a point of [`admissible_value`] can respect (`_POINT_FACTS`).
const POINT_FACTS: [Predicate; 25] = [
    Predicate::commutative,
    Predicate::complex,
    Predicate::finite,
    Predicate::integer,
    Predicate::rational,
    Predicate::real,
    Predicate::extended_real,
    Predicate::positive,
    Predicate::negative,
    Predicate::nonnegative,
    Predicate::nonpositive,
    Predicate::nonzero,
    Predicate::zero,
    Predicate::extended_positive,
    Predicate::extended_negative,
    Predicate::extended_nonnegative,
    Predicate::extended_nonpositive,
    Predicate::extended_nonzero,
    Predicate::even,
    Predicate::odd,
    Predicate::algebraic,
    Predicate::hermitian,
    Predicate::imaginary,
    Predicate::antihermitian,
    Predicate::polar,
];

/// A value of the symbol respecting its assumptions; `variant` 0, 1, 2
/// picks other signs, sizes and quadrants; `None` when a fact is not
/// handled (`_admissible_value`).
fn admissible_value(s: &Basic, i: i64, variant: usize) -> PolyResult<Option<Basic>> {
    use crate::numbers::Int;
    let facts = super::assumptions::facts_of(s);
    if facts.iter().any(|(k, v)| *v && !POINT_FACTS.contains(k)) {
        return Ok(None);
    }
    let yes = |p: Predicate| facts.get(&p) == Some(&true);
    if yes(Predicate::zero) {
        return Ok(Some(Basic::zero()));
    }
    let negative: bool = yes(Predicate::negative)
        || yes(Predicate::nonpositive)
        || yes(Predicate::extended_negative)
        || yes(Predicate::extended_nonpositive);
    let positive: bool = yes(Predicate::positive)
        || yes(Predicate::nonnegative)
        || yes(Predicate::extended_positive)
        || yes(Predicate::extended_nonnegative)
        || yes(Predicate::polar);
    let sign: i64 = if negative {
        -1
    } else if positive {
        1
    } else {
        [1, -1, -1][variant]
    };
    let scale: i64 = [1, 9, 1][variant];
    let rational = |p: i64, q: i64| Basic::rational(Int::from(p), Int::from(q));
    if yes(Predicate::integer) || yes(Predicate::even) || yes(Predicate::odd) {
        if yes(Predicate::even) {
            return Ok(Some(Basic::integer(Int::from(sign * scale * (4 + 2 * i)))));
        }
        return Ok(Some(Basic::integer(Int::from(sign * ((scale * (3 + 2 * i)) | 1)))));
    }
    if yes(Predicate::imaginary) || yes(Predicate::antihermitian) {
        return Ok(Some(super::mul::mul(&[
            Basic::integer(Int::from(sign)),
            rational(scale * (7 + 2 * i), 11)?,
            Basic::imaginary_unit(),
        ])?));
    }
    if yes(Predicate::real)
        || yes(Predicate::extended_real)
        || yes(Predicate::rational)
        || negative
        || positive
        || yes(Predicate::hermitian)
    {
        return Ok(Some(super::mul::mul(&[
            Basic::integer(Int::from(sign)),
            rational(scale * (7 + 3 * i), 11)?,
        ])?));
    }
    let quadrant: (i64, i64) = [(1, 1), (-1, -1), (-1, 1)][variant];
    Ok(Some(super::add::add(&[
        super::mul::mul(&[
            Basic::integer(Int::from(quadrant.0)),
            rational(scale * (7 + 3 * i), 11)?,
        ])?,
        super::mul::mul(&[
            Basic::integer(Int::from(quadrant.1)),
            rational(scale * (5 + i), 13)?,
            Basic::imaginary_unit(),
        ])?,
    ])?))
}

/// Whether `e` is shown not to be identically zero by its value at an
/// admissible point: well above the noise and the same at two precisions
/// (`_nonzero_at_a_point`).
fn nonzero_at_a_point(e: &Basic) -> PolyResult<bool> {
    let mut symbols: Vec<Basic> = e
        .free_symbols()
        .into_iter()
        .filter(|s| matches!(s.head(), Head::Symbol(_)))
        .collect();
    symbols.sort_by_key(|s| s.to_string());
    let variants: usize = if symbols.is_empty() { 1 } else { 3 };
    for variant in 0..variants {
        let mut point: std::collections::HashMap<Basic, Basic> = std::collections::HashMap::new();
        for (i, s) in symbols.iter().enumerate() {
            match admissible_value(s, i as i64, variant)? {
                None => return Ok(false),
                Some(v) => {
                    point.insert(s.clone(), v);
                }
            }
        }
        let at: Basic = if point.is_empty() {
            e.clone()
        } else {
            match super::function::xreplace_evaluated(e, &point) {
                Ok(v) => v,
                Err(
                    crate::error::PolyError::ZeroDivision(_)
                    | crate::error::PolyError::Value(_)
                    | crate::error::PolyError::Type(_),
                ) => continue,
                Err(other) => return Err(other),
            }
        };
        let (Some(low), Some(high)) = (
            super::evalf::numeric_value(&at, 60),
            super::evalf::numeric_value(&at, 200),
        ) else {
            continue;
        };
        let size = |z: &super::evalf::Complex| {
            let (a, b) = super::evalf::parts(z);
            (a.to_f64().value().powi(2) + b.to_f64().value().powi(2)).sqrt()
        };
        let difference: super::evalf::Complex = &low - &high;
        let h: f64 = size(&high);
        if h > 1e-30 && size(&difference) < 1e-12 * h {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether `a - b` is zero (`Expr.equals`): decided by the facts or by
/// a value at a point; what Python asks SymPy (the bridge) is not ported.
pub fn equals(a: &Basic, b: &Basic) -> PolyResult<bool> {
    let difference: Basic = super::add::sub(a, b)?;
    if difference.is_number_node() {
        return Ok(difference.as_rational().is_some_and(|v| v.is_zero()));
    }
    if let Some(zero) = is(&difference, Predicate::zero) {
        return Ok(zero);
    }
    if nonzero_at_a_point(&difference)? {
        return Ok(false);
    }
    Err(crate::error::not_ported("equals by SymPy (the bridge)"))
}
