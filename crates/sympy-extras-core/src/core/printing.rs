//! Port of `sympy_extras/core/printing.py` for the classes of the Rust
//! core: [`sstr`], the string form in SymPy's notation (`str(e)`), and the
//! printing order of the terms of a sum, [`ordered_terms`], which is also
//! the order of the arguments in the sort key of a sum.
//!
//! `Float`, `Mod`, the order terms, the relationals and the logical
//! operators are not in the Rust core, so their branches are absent.

use std::cmp::Ordering;

use crate::numbers::*;

use super::assumptions::{is_number, Predicate};
use super::basic::{Basic, Head};
use super::evalf::{numeric_value, parts};
use super::expr::{as_base_exp, as_coeff_mul, factors_of, is_false, is_true};
use super::mul::{mul, neg};
use super::power::pow;
use super::sorting::{sort_key, SortKey};

pub const PREC_ADD: i32 = 40;
pub const PREC_MUL: i32 = 50;
pub const PREC_POW: i32 = 60;
pub const PREC_ATOM: i32 = 1000;

/// Python's `NumberSymbol`: the named constants.
fn is_number_symbol(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Pi | Head::Exp1 | Head::EulerGamma | Head::Catalan | Head::GoldenRatio | Head::TribonacciConstant
    )
}

fn is_commutative_maybe(e: &Basic) -> bool {
    !is_false(e, Predicate::commutative)
}

/// Whether the expression is printed with a leading minus
/// (`could_extract_minus_sign`).
pub fn could_extract_minus_sign(e: &Basic) -> bool {
    if e.is_add() {
        let negatives: usize = e.args().iter().filter(|a| could_extract_minus_sign(a)).count();
        let positives: usize = e.args().len() - negatives;
        if negatives != positives {
            return negatives > positives;
        }
        return match mul(&[Basic::negative_one(), e.clone()]) {
            Ok(negated) => sort_key(e) < sort_key(&negated),
            Err(_) => false,
        };
    }
    let (c, _) = as_coeff_mul(e);
    is_true(&c, Predicate::extended_negative)
}

/// The precedence of the printed form.
pub fn precedence(e: &Basic) -> i32 {
    match e.head() {
        Head::Add => PREC_ADD,
        Head::Mul => {
            if could_extract_minus_sign(e) {
                PREC_ADD
            } else {
                PREC_MUL
            }
        }
        Head::Pow => PREC_POW,
        Head::Rational { value, .. } => {
            if rat_is_negative(value) {
                PREC_ADD
            } else if denom(value) > Int::ONE {
                PREC_MUL
            } else {
                PREC_ATOM
            }
        }
        Head::NegativeInfinity => PREC_ADD,
        _ => PREC_ATOM,
    }
}

/// `(b, n)` with `f == b**n` and `n` an integer (`_decompose_power`).
fn decompose_power(f: &Basic) -> (Basic, Rat) {
    let Ok((b, e)) = as_base_exp(f) else {
        return (f.clone(), Rat::ONE);
    };
    if let Some(x) = e.as_rational() {
        if x.is_int() {
            return (b, x.clone());
        }
        let root: Rat = rat_from_parts(Int::ONE, denom(x));
        return match pow(&b, &Basic::from_rational(root)) {
            Ok(r) => (r, rat_from_int(&numer(x))),
            Err(_) => (f.clone(), Rat::ONE),
        };
    }
    let (c, rest) = as_coeff_mul(&e);
    if let Some(v) = c.as_rational() {
        if v.is_int() && *v != Rat::ONE {
            if let Ok(r) = pow(&b, &rest) {
                return (r, v.clone());
            }
        }
    }
    (f.clone(), Rat::ONE)
}

/// The factors of the non-numerical part of a term.
fn rest_factors(term: &Basic) -> Vec<Basic> {
    let (_, rest) = as_coeff_mul(term);
    factors_of(&rest)
}

fn monomial_key(term: &Basic, gens: &[Basic]) -> Vec<Rat> {
    let mut bases: Vec<Basic> = Vec::new();
    let mut exponents: Vec<Rat> = Vec::new();
    for f in rest_factors(term) {
        if matches!(f.head(), Head::ImaginaryUnit) || is_number_symbol(&f) || is_number(&f) {
            continue;
        }
        let (b, n) = decompose_power(&f);
        if is_number_symbol(&b) {
            continue;
        }
        match bases.iter().position(|x| *x == b) {
            Some(k) => exponents[k] += n,
            None => {
                bases.push(b);
                exponents.push(n);
            }
        }
    }
    gens.iter()
        .map(|g| match bases.iter().position(|x| x == g) {
            Some(k) => -exponents[k].clone(),
            None => Rat::ZERO,
        })
        .collect()
}

fn generators(term: &Basic) -> Vec<Basic> {
    let mut gens: Vec<Basic> = Vec::new();
    for f in rest_factors(term) {
        if f.is_number_class() || matches!(f.head(), Head::ImaginaryUnit) || is_number_symbol(&f) || is_number(&f) {
            continue;
        }
        if is_false(&f, Predicate::commutative) {
            continue;
        }
        let (b, _) = decompose_power(&f);
        if is_number_symbol(&b) {
            continue;
        }
        gens.push(b);
    }
    gens
}

/// Whether the exponent has a negative numerical coefficient.
fn negative_exponent(e: &Basic) -> bool {
    let (c, _) = as_coeff_mul(e);
    c.as_rational().is_some_and(rat_is_negative)
}

/// Whether the arguments are as an evaluating function stores them: a
/// number first, then the `compare` order (`_in_compare_order`).
pub fn in_compare_order(args: &[Basic]) -> bool {
    let rest: &[Basic] = if !args.is_empty() && args[0].is_number_class() {
        &args[1..]
    } else {
        args
    };
    rest.windows(2).all(|w| w[0].compare_key() <= w[1].compare_key())
}

/// The factors of a product in the printing order (`_printing_order`).
fn printing_order(factors: &[Basic]) -> Vec<Basic> {
    let commutative: Vec<Basic> = factors.iter().filter(|f| is_commutative_maybe(f)).cloned().collect();
    if !in_compare_order(&commutative) {
        return factors.to_vec();
    }
    let mut ordered: Vec<Basic> = commutative;
    ordered.sort_by_cached_key(sort_key);
    ordered.extend(factors.iter().filter(|f| !is_commutative_maybe(f)).cloned());
    ordered
}

/// Among the terms with the same monomial, by the increasing numerical
/// coefficient, the imaginary ones after the others (`_coefficient_rank`).
#[derive(Debug, Clone, PartialEq)]
struct CoefficientRank {
    imaginary: bool,
    imag: f64,
    real: f64,
    imag_again: f64,
}

impl CoefficientRank {
    fn cmp(&self, other: &CoefficientRank) -> Ordering {
        let floats = |a: f64, b: f64| a.partial_cmp(&b).unwrap_or(Ordering::Equal);
        self.imaginary
            .cmp(&other.imaginary)
            .then(floats(self.imag, other.imag))
            .then(floats(self.real, other.real))
            .then(floats(self.imag_again, other.imag_again))
    }
}

fn coefficient_rank(term: &Basic) -> CoefficientRank {
    let numbers: Vec<Basic> = factors_of(term).into_iter().filter(is_number).collect();
    let zero = CoefficientRank {
        imaginary: false,
        imag: 0.0,
        real: 0.0,
        imag_again: 0.0,
    };
    let Ok(coefficient) = mul(&numbers) else {
        return zero;
    };
    let Some(value) = numeric_value(&coefficient, 53 + 20) else {
        return zero;
    };
    let (re, im) = parts(&value);
    let (real, imag): (f64, f64) = (re.to_f64().value(), im.to_f64().value());
    CoefficientRank {
        imaginary: imag != 0.0,
        imag,
        real,
        imag_again: imag,
    }
}

/// SymPy's special case of a sum of two terms printed as `1 - x`
/// (`_leads_a_difference`).
fn leads_a_difference(number: &Basic, term: &Basic) -> bool {
    if !(number.is_number_class() || is_number_symbol(number))
        || !is_true(number, Predicate::positive)
        || !is_true(number, Predicate::finite)
    {
        return false;
    }
    let (c, rest) = as_coeff_mul(term);
    c.as_rational().is_some_and(rat_is_negative) && !rest.is_mul()
}

/// The terms of a sum in the printing order: decreasing degrees in the
/// generators (lexicographically), numbers last (`ordered_terms`).
pub fn ordered_terms(terms: &[Basic]) -> Vec<Basic> {
    if terms.len() == 2 && leads_a_difference(&terms[0], &terms[1]) {
        return terms.to_vec();
    }
    let mut gens: Vec<Basic> = Vec::new();
    for t in terms {
        for g in generators(t) {
            if !gens.contains(&g) {
                gens.push(g);
            }
        }
    }
    gens.sort_by_cached_key(sort_key);
    type Key = (Vec<Rat>, Vec<SortKey>, CoefficientRank, SortKey);
    let keyed: Vec<(Key, Basic)> = terms
        .iter()
        .map(|t| {
            let noncommutative: Vec<SortKey> = factors_of(t)
                .iter()
                .filter(|f| is_false(f, Predicate::commutative))
                .map(sort_key)
                .collect();
            (
                (monomial_key(t, &gens), noncommutative, coefficient_rank(t), sort_key(t)),
                t.clone(),
            )
        })
        .collect();
    let mut keyed = keyed;
    keyed.sort_by(|(a, _), (b, _)| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    keyed.into_iter().map(|(_, t)| t).collect()
}

/// The printer of [`sstr`].
struct StrPrinter;

impl StrPrinter {
    fn doprint(&self, e: &Basic) -> String {
        match e.head() {
            Head::Add => self.print_add(e.args()),
            Head::Mul => self.print_mul(e.args()),
            Head::Pow => self.print_pow(&e.args()[0], &e.args()[1]),
            _ => e.to_string(),
        }
    }

    fn parenthesize(&self, e: &Basic, level: i32, strict: bool) -> String {
        let p: i32 = precedence(e);
        let s: String = self.doprint(e);
        if p < level || (strict && p <= level) {
            format!("({s})")
        } else {
            s
        }
    }

    fn print_add(&self, terms: &[Basic]) -> String {
        let ordered: Vec<Basic> = if in_compare_order(terms) {
            ordered_terms(terms)
        } else {
            terms.to_vec()
        };
        let mut out: String = String::new();
        for (i, term) in ordered.iter().enumerate() {
            let s: String = self.parenthesize(term, PREC_ADD, false);
            if i == 0 {
                out.push_str(&s);
            } else if let Some(stripped) = s.strip_prefix('-') {
                out.push_str(" - ");
                out.push_str(stripped);
            } else {
                out.push_str(" + ");
                out.push_str(&s);
            }
        }
        out
    }

    fn print_mul(&self, factors: &[Basic]) -> String {
        let factors: Vec<Basic> = printing_order(factors);
        let mut sign: &str = "";
        let mut numerator: Vec<String> = Vec::new();
        let mut denominator: Vec<String> = Vec::new();
        for (i, f) in factors.iter().enumerate() {
            if let (Head::Rational { value, .. }, 0) = (f.head(), i) {
                let mut p: Int = numer(value);
                let q: Int = denom(value);
                if is_negative(&p) {
                    sign = "-";
                    p = -p;
                }
                if p != Int::ONE || (q == Int::ONE && factors.len() == 1) {
                    numerator.push(p.to_string());
                }
                if q != Int::ONE {
                    denominator.push(q.to_string());
                }
                continue;
            }
            if f.is_pow() && negative_exponent(&f.args()[1]) && is_commutative_maybe(f) {
                let (base, exp) = (&f.args()[0], &f.args()[1]);
                let inverse: Basic = match neg(exp) {
                    Ok(minus) if minus == Basic::one() && minus.is_number_class() => base.clone(),
                    Ok(minus) => Basic::new(Head::Pow, vec![base.clone(), minus]),
                    Err(_) => f.clone(),
                };
                denominator.push(self.parenthesize(&inverse, PREC_MUL, true));
                continue;
            }
            numerator.push(self.parenthesize(f, PREC_MUL, i > 0 && f.is_number_class()));
        }
        if numerator.is_empty() {
            numerator.push("1".to_string());
        }
        let num: String = numerator.join("*");
        if denominator.is_empty() {
            return format!("{sign}{num}");
        }
        let den: String = if denominator.len() == 1 {
            denominator.remove(0)
        } else {
            format!("({})", denominator.join("*"))
        };
        format!("{sign}{num}/{den}")
    }

    fn print_pow(&self, base: &Basic, exp: &Basic) -> String {
        if let Head::Rational { value, .. } = exp.head() {
            let (p, q) = (numer(value), denom(value));
            if p == Int::ONE && q == int(2) {
                return format!("sqrt({})", self.doprint(base));
            }
            if p == -Int::ONE && q == int(2) && is_commutative_maybe(base) {
                return format!("1/sqrt({})", self.doprint(base));
            }
            if p == -Int::ONE && q == Int::ONE && is_commutative_maybe(base) {
                return format!("1/{}", self.parenthesize(base, PREC_MUL, true));
            }
        }
        format!(
            "{}**{}",
            self.parenthesize(base, PREC_POW, true),
            self.parenthesize(exp, PREC_POW, true)
        )
    }
}

/// The string form of `e` in SymPy's notation (`str(e)`).
pub fn sstr(e: &Basic) -> String {
    StrPrinter.doprint(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::add::add;
    use crate::core::Symbol;

    #[test]
    fn printed_forms() {
        let x: Basic = Symbol::new("x", &[]).0;
        let y: Basic = Symbol::new("y", &[]).0;
        let two: Basic = Basic::int(2);
        let p = add(&[
            pow(&x, &two).unwrap(),
            mul(&[two.clone(), x.clone()]).unwrap(),
            Basic::one(),
        ])
        .unwrap();
        assert_eq!(sstr(&p), "x**2 + 2*x + 1");
        let q = mul(&[
            Basic::from_rational(rat(-1, 2)),
            x.clone(),
            pow(&y, &Basic::negative_one()).unwrap(),
        ])
        .unwrap();
        assert_eq!(sstr(&q), "-x/(2*y)");
        assert_eq!(sstr(&pow(&x, &Basic::int(-2)).unwrap()), "x**(-2)");
        assert_eq!(sstr(&add(&[Basic::one(), neg(&x).unwrap()]).unwrap()), "1 - x");
        assert_eq!(sstr(&pow(&Basic::int(12), &Basic::half()).unwrap()), "2*sqrt(3)");
    }
}
