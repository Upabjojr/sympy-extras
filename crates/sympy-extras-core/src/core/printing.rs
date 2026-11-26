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
use super::expr::{as_base_exp, as_coeff_mul, factors_of, is_false, is_rational_value, is_true};
use super::mul::{mul, neg};
use super::power::pow;
use super::sorting::{sort_key, SortKey};

pub const PREC_ADD: i32 = 40;
pub const PREC_MUL: i32 = 50;
pub const PREC_POW: i32 = 60;
pub const PREC_ATOM: i32 = 1000;
pub const PREC_XOR: i32 = 10;
pub const PREC_IMPLIES: i32 = 10;
pub const PREC_OR: i32 = 20;
pub const PREC_AND: i32 = 30;
pub const PREC_RELATIONAL: i32 = 35;
pub const PREC_FUNC: i32 = 70;
pub const PREC_NOT: i32 = 100;
/// The arguments of `|`, `^` and `&` are parenthesized below these.
pub const PREC_OR_ARGS: i32 = 36;
pub const PREC_XOR_ARGS: i32 = 37;
pub const PREC_AND_ARGS: i32 = 38;

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
        Head::Pow => {
            // 1/x and 1/sqrt(x) are printed as quotients: as a base, or in
            // a denominator, they need their parentheses (1/(1/x), (1/x)**2)
            let quotient: bool = e.args()[1].as_rational().is_some_and(|r| {
                crate::numbers::numer(r) == crate::numbers::Int::from(-1)
                    && [crate::numbers::Int::from(1), crate::numbers::Int::from(2)].contains(&crate::numbers::denom(r))
            });
            if quotient {
                PREC_MUL
            } else {
                PREC_POW
            }
        }
        Head::Relational(op) => {
            if matches!(op, super::basic::RelOp::Eq | super::basic::RelOp::Ne) {
                PREC_FUNC
            } else {
                PREC_RELATIONAL
            }
        }
        Head::And => PREC_AND,
        Head::Or => PREC_OR,
        Head::Xor => PREC_XOR,
        Head::Implies | Head::Equivalent => PREC_IMPLIES,
        Head::Not => PREC_NOT,
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

/// The value of a factor without free symbols as a complex float, `None`
/// when it has none (an infinity, `Sum(f(k), (k, 0, 1))`) or does not fit
/// a float (`_numeric_value`, Python's `to_complex`).
fn factor_value(f: &Basic) -> Option<(f64, f64)> {
    let value = numeric_value(f, 53)?;
    let (re, im) = parts(&value);
    let (re, im): (f64, f64) = (re.to_f64().value(), im.to_f64().value());
    if re.is_finite() && im.is_finite() {
        Some((re, im))
    } else {
        None
    }
}

/// Whether the factor counts in the numerical coefficient of its term
/// (`_is_numeric`): SymPy's `as_terms` takes the factors without free
/// symbols which have a numerical value; one which has none is a
/// generator (`oo*I`, `zoo**EulerGamma`).
fn is_numeric(f: &Basic) -> bool {
    is_number(f) && factor_value(f).is_some()
}

fn monomial_key(term: &Basic, gens: &[Basic]) -> Vec<Rat> {
    let mut bases: Vec<Basic> = Vec::new();
    let mut exponents: Vec<Rat> = Vec::new();
    for f in rest_factors(term) {
        if matches!(f.head(), Head::ImaginaryUnit) || is_number_symbol(&f) || is_numeric(&f) {
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
        if f.is_number_class() || matches!(f.head(), Head::ImaginaryUnit) || is_number_symbol(&f) || is_numeric(&f) {
            continue;
        }
        // SymPy's as_terms: the factors which are not known to commute (a
        // Limit, whose commutativity is unknown) are not generators
        if !is_true(&f, Predicate::commutative) {
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
    let rest: &[Basic] =
        if !args.is_empty() && (args[0].is_number_class() || matches!(args[0].head(), Head::AccumBounds)) {
            &args[1..]
        } else {
            args
        };
    rest.windows(2).all(|w| w[0].compare_key() <= w[1].compare_key())
}

/// The factors of a product in the printing order (`_printing_order`).
pub(crate) fn printing_order(factors: &[Basic]) -> Vec<Basic> {
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
    // the product of the values of the numeric factors, as SymPy's
    // as_terms: a factor without a value is a generator
    let (mut real, mut imag): (f64, f64) = (1.0, 0.0);
    for f in factors_of(term) {
        if is_number(&f) {
            if let Some((a, b)) = factor_value(&f) {
                (real, imag) = (real * a - imag * b, real * b + imag * a);
            }
        }
    }
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
    let (mut plain, orders): (Vec<Basic>, Vec<Basic>) = keyed
        .into_iter()
        .map(|(_, t)| t)
        .partition(|t| !matches!(t.head(), Head::Order));
    if !orders.is_empty() {
        // a series is printed in increasing order: x - x**3/6 + O(x**5)
        plain.reverse();
    }
    plain.extend(orders);
    plain
}

/// The printer of [`sstr`].
struct StrPrinter;

impl StrPrinter {
    fn doprint(&self, e: &Basic) -> String {
        match e.head() {
            Head::Add => self.print_add(e.args()),
            Head::Mul => self.print_mul(e.args()),
            Head::Pow => self.print_pow(&e.args()[0], &e.args()[1]),
            Head::Function(f) => {
                let shown: Vec<Basic> = crate::functions::special::delta_functions::printed_args(e);
                let args: Vec<String> = shown.iter().map(|a| self.doprint(a)).collect();
                format!("{}({})", f.name(), args.join(", "))
            }
            Head::Undefined(u) => {
                let args: Vec<String> = e.args().iter().map(|a| self.doprint(a)).collect();
                format!("{}({})", u.name, args.join(", "))
            }
            Head::Tuple => {
                let args: Vec<String> = e.args().iter().map(|a| self.doprint(a)).collect();
                if args.len() == 1 {
                    format!("({},)", args[0])
                } else {
                    format!("({})", args.join(", "))
                }
            }
            Head::Derivative => {
                let mut parts: Vec<String> = vec![self.doprint(&e.args()[0])];
                for pair in &e.args()[1..] {
                    let (s, n) = (&pair.args()[0], &pair.args()[1]);
                    if is_rational_value(n, 1) {
                        parts.push(self.doprint(s));
                    } else {
                        parts.push(format!("({}, {})", self.doprint(s), n));
                    }
                }
                format!("Derivative({})", parts.join(", "))
            }
            _ if crate::sets::sets::is_set(e) || matches!(e.head(), Head::Contains) => {
                crate::sets::sets::print_set(e, &|a: &Basic| self.doprint(a))
            }
            Head::Piecewise => format!(
                "Piecewise({})",
                e.args()
                    .iter()
                    .map(|p| self.doprint(p))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Head::ComplexRootOf(_) => crate::polys::complexrootof::print_root(e, &|a: &Basic| self.doprint(a)),
            Head::ForAll | Head::Exists => {
                crate::assumptions::quantifiers::print_quantifier(e, &|a: &Basic| self.doprint(a))
            }
            Head::Predicate(p) => format!("Q.{}", p.name),
            Head::AppliedPredicate => format!(
                "{}({})",
                self.doprint(&e.args()[0]),
                e.args()[1..]
                    .iter()
                    .map(|a| self.doprint(a))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Head::Hyper | Head::MeijerG => {
                crate::functions::special::hyper::print_hyper(e, &|a: &Basic| self.doprint(a))
            }
            Head::ExprCondPair => format!("({}, {})", self.doprint(&e.args()[0]), self.doprint(&e.args()[1])),
            Head::AccumBounds => format!(
                "AccumBounds({}, {})",
                self.doprint(&e.args()[0]),
                self.doprint(&e.args()[1])
            ),
            Head::BooleanTrue => "True".to_string(),
            Head::BooleanFalse => "False".to_string(),
            Head::And | Head::Or | Head::Xor => {
                let (separator, level) = match e.head() {
                    Head::And => (" & ", PREC_AND_ARGS),
                    Head::Or => (" | ", PREC_OR_ARGS),
                    _ => (" ^ ", PREC_XOR_ARGS),
                };
                let parts: Vec<String> = e.args().iter().map(|a| self.parenthesize(a, level, false)).collect();
                parts.join(separator)
            }
            Head::Not => format!("~{}", self.parenthesize(&e.args()[0], PREC_FUNC, false)),
            Head::Implies | Head::Equivalent | Head::Ite => {
                let parts: Vec<String> = e.args().iter().map(|a| self.doprint(a)).collect();
                format!("{}({})", e.class_name(), parts.join(", "))
            }
            Head::Relational(op) => {
                let (lhs, rhs) = (&e.args()[0], &e.args()[1]);
                if matches!(op, super::basic::RelOp::Eq | super::basic::RelOp::Ne) {
                    format!("{}({}, {})", op.class_name(), self.doprint(lhs), self.doprint(rhs))
                } else {
                    format!(
                        "{} {} {}",
                        self.parenthesize(lhs, PREC_RELATIONAL, false),
                        op.symbol(),
                        self.parenthesize(rhs, PREC_RELATIONAL, false)
                    )
                }
            }
            Head::IndexedRoot => {
                let (f, t, k) = crate::polys::cad::indexedroot::shown_parts(e);
                format!("IndexedRoot({}, {}, {})", self.doprint(&f), self.doprint(&t), k)
            }
            Head::RootSum => crate::polys::rootoftools::print(e, &|a: &Basic| self.doprint(a)),
            Head::Order => crate::core::order::print(e, &|a: &Basic| self.doprint(a)),
            Head::Limit => crate::core::limits::print(e, &|a: &Basic| self.doprint(a)),
            Head::Sum | Head::Product => crate::concrete::expr_with_limits::print(e, &|a: &Basic| self.doprint(a)),
            Head::Integral | Head::NonElementaryIntegral => {
                crate::integrals::integral::print(e, &|a: &Basic| self.doprint(a))
            }
            Head::TranscendentalRoot => crate::solvers::isolation::print(e, &|a: &Basic| self.doprint(a)),
            Head::Subs => {
                let (expr, variables, point) = (&e.args()[0], e.args()[1].args(), e.args()[2].args());
                if variables.len() == 1 {
                    format!(
                        "Subs({}, {}, {})",
                        self.doprint(expr),
                        self.doprint(&variables[0]),
                        self.doprint(&point[0])
                    )
                } else {
                    let vs: Vec<String> = variables.iter().map(|v| self.doprint(v)).collect();
                    let ps: Vec<String> = point.iter().map(|p| self.doprint(p)).collect();
                    format!("Subs({}, ({}), ({}))", self.doprint(expr), vs.join(", "), ps.join(", "))
                }
            }
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
        // a Mod factor is parenthesized, as SymPy prints 2*(Mod(x, 3)), except in -Mod(x, 3)
        let bare_negation: bool = factors.len() == 2 && is_rational_value(&factors[0], -1);
        for (i, f) in factors.iter().enumerate() {
            if f.is_function(super::basic::Function::Mod) && !bare_negation {
                numerator.push(format!("({})", self.doprint(f)));
                continue;
            }
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
        if base.is_function(super::basic::Function::Mod) {
            return format!("({})**{}", self.doprint(base), self.parenthesize(exp, PREC_POW, true));
        }
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
