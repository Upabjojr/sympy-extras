//! Port of `sympy_extras/printing/pretty/rules_core.py`: the
//! pretty-printing rules of the core (numbers, symbols, sums, products,
//! powers, relations, logic, containers), as a `match` on the head; the
//! functions are in [`super::rules_functions`]. The rules of
//! `rules_calculus.py`, `rules_sets.py` and `rules_other.py` are not ported
//! yet: their nodes raise "not ported".

// PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_Add, _print_Mul, _print_Pow,
// _print_Rational, _print_Relational, the boolean connectives, the containers),
// sympy/printing/pretty/stringpict.py (prettyForm.__add__, __mul__, __truediv__, __pow__) (SymPy 1.14,
// BSD license; through sympy_extras/printing/pretty/rules_core.py)

use super::engine::PrettyPrinter;
use super::layout::{Picture, PrettyContext, PrettyResult};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is;
use crate::error::{not_ported, PolyResult};
use crate::numbers::Int;
use crate::printing::framework::{Binding, ChildOptions, Position, PrintResult, TreePrinter};

type R = PolyResult<PrettyResult>;

/// The negation of what is printed as `magnitude` (`negative`).
pub fn negative(magnitude: Picture, stacked: bool) -> PrettyResult {
    let sign: &str = if magnitude.height() > 1 { "- " } else { "-" };
    let mut r: PrettyResult = PrintResult::new(magnitude.ltext(sign), Binding::Negation);
    r.unsigned = Some(magnitude);
    r.stacked = stacked;
    r
}

/// An integer; a negative one is not an atom (`integer_result`).
pub fn integer_result(n: &Int) -> PrettyResult {
    if *n < Int::ZERO {
        return negative(Picture::of(&(-n.clone()).to_string()), false);
    }
    PrintResult::new(Picture::of(&n.to_string()), Binding::Atom)
}

/// The operands joined by `operator` (`infix`).
pub fn infix(p: &PrettyPrinter, args: &[Basic], operator: &str, binding: Binding, operand: &PrettyContext) -> R {
    let mut parts: Vec<Picture> = Vec::new();
    for a in args {
        parts.push(p.operand(a, operand)?);
    }
    Ok(PrintResult::new(p.join(&parts, &format!(" {} ", operator)), binding))
}

fn rational_parts(e: &Basic) -> Option<(Int, Int)> {
    e.as_rational()
        .map(|v| (crate::numbers::numer(v), crate::numbers::denom(v)))
}

/// Python's `isinstance(e, Number)`: a rational, a float, the
/// infinities, `nan`.
fn is_number_node(e: &Basic) -> bool {
    e.is_number_class()
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

/// The rule of a node, `None` for a node which prints as its string form.
pub fn rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> PolyResult<Option<PrettyResult>> {
    let s = &p.symbology;
    Ok(Some(match e.head() {
        Head::Rational { .. } => rational(p, e, ctx),
        Head::Float { value, prec } => {
            // all the digits for a float alone, as SymPy prints it (full_prec='auto')
            let full: bool = p.settings.full_prec.unwrap_or(ctx.position == Position::Top);
            let text: String = if full {
                crate::core::float::float_str(value, *prec)
            } else {
                crate::core::float::float_short_str(value, *prec)
            };
            if let Some(magnitude) = text.strip_prefix('-') {
                // a negative float is not an atom: (-2.5)**n (sympy/sympy#30844, root cause 5)
                negative(Picture::of(magnitude), false)
            } else {
                // 1.0e-5 is a product as a base: (1.0e-5)**2
                let binding: Binding = if text.contains('e') {
                    Binding::Product
                } else {
                    Binding::Atom
                };
                PrintResult::new(Picture::of(&text), binding)
            }
        }
        Head::Infinity => p.atom(&s.atom("Infinity", "oo")),
        Head::NegativeInfinity => negative(Picture::of(&s.atom("Infinity", "oo")), false),
        Head::ComplexInfinity => p.atom("zoo"),
        Head::NaN => p.atom("nan"),
        Head::Pi => p.atom(&s.atom("Pi", "pi")),
        Head::Exp1 => p.atom(&s.atom("Exp1", "e")),
        Head::GoldenRatio => p.atom(&s.greek("phi", "GoldenRatio")),
        Head::EulerGamma => p.atom(&s.greek("gamma", "EulerGamma")),
        Head::Catalan => p.atom("G"),
        Head::TribonacciConstant => p.atom("TribonacciConstant"),
        Head::ImaginaryUnit => p.atom(&s.atom("ImaginaryUnit", "I")),
        Head::Symbol(d) | Head::Dummy(d, _) => p.atom(&s.symbol(&d.name)),
        Head::Wild(w) => p.atom(&format!("{}_", s.symbol(&w.symbol.name))),
        Head::BooleanTrue | Head::BooleanFalse => p.atom(&crate::core::printing::sstr(e)),
        Head::Add => add_rule(p, e, ctx)?,
        Head::Mul => mul_rule(p, e, ctx)?,
        Head::Pow => pow_rule(p, e, ctx)?,
        Head::Function(Function::Mod) => {
            let left: Picture = p.operand(
                &e.args()[0],
                &ctx.at(
                    Position::Left,
                    Binding::Product,
                    ChildOptions {
                        followed: true,
                        ..ChildOptions::default()
                    },
                ),
            )?;
            let right: Picture = p.operand(
                &e.args()[1],
                &ctx.at(
                    Position::Right,
                    Binding::Product,
                    ChildOptions {
                        strict: true,
                        ..ChildOptions::default()
                    },
                ),
            )?;
            PrintResult::new(
                Picture::beside(&[&left, &Picture::of(" mod "), &right]),
                Binding::Modulo,
            )
        }
        Head::Relational(op) => {
            // a side which is itself a relation or a formula is bracketed
            let left: Picture = p.operand(&e.args()[0], &ctx.side())?;
            let right: Picture = p.operand(&e.args()[1], &ctx.side())?;
            let operator: Picture = Picture::of(&format!(" {} ", s.xsym(op.symbol())));
            PrintResult::new(Picture::beside(&[&left, &operator, &right]), Binding::Relation)
        }
        Head::And => connective(p, &sorted(e.args()), ctx, "And", "And", Binding::And)?,
        Head::Or => connective(p, &sorted(e.args()), ctx, "Or", "Or", Binding::Or)?,
        Head::Xor => connective(p, &sorted(e.args()), ctx, "Xor", "Xor", Binding::Xor)?,
        Head::Implies => negatable(p, e.args(), ctx, "Implies", "Arrow", "NotArrow")?,
        Head::Equivalent => negatable(p, &sorted(e.args()), ctx, "Equivalent", "Equiv", "NotEquiv")?,
        Head::Not => not_rule(p, e, ctx)?,
        Head::Tuple => {
            if e.args().len() == 1 {
                let inside: Picture = p.operand(&e.args()[0], &ctx.delimited())?.rtext(",");
                PrintResult::new(p.parens(&inside, "(", ")", true), Binding::Atom)
            } else {
                p.sequence(e.args(), ctx, "(", ")", ", ")?
            }
        }
        Head::AccumBounds => p.sequence(e.args(), ctx, "<", ">", ", ")?,
        _ => {
            if let Some(r) = super::rules_functions::function_rule(p, e, ctx)? {
                r
            } else if let Some(r) = super::rules_calculus::calculus_rule(p, e, ctx)? {
                r
            } else if let Some(r) = super::rules_sets::set_rule(p, e, ctx)? {
                r
            } else if let Some(r) = super::rules_other::other_rule(p, e, ctx)? {
                r
            } else {
                return Err(not_ported(&format!("the pretty printing of {}", e.class_name())));
            }
        }
    }))
}

fn sorted(args: &[Basic]) -> Vec<Basic> {
    let mut v: Vec<Basic> = args.to_vec();
    v.sort_by_cached_key(crate::core::sorting::sort_key);
    v
}

fn rational(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> PrettyResult {
    let (num, den) = rational_parts(e).expect("a rational");
    if den == Int::ONE {
        return integer_result(&num);
    }
    let negative_value: bool = num < Int::ZERO;
    let magnitude_num: Int = if negative_value { -num.clone() } else { num.clone() };
    if ctx.display || (magnitude_num >= Int::from(10) && den >= Int::from(10)) {
        let magnitude: Picture = p.fraction(&Picture::of(&magnitude_num.to_string()), &Picture::of(&den.to_string()));
        if negative_value {
            // (SymPy's layout: the minus sign in the numerator)
            let picture: Picture = p.fraction(
                &Picture::of(&format!("-{} ", magnitude_num)),
                &Picture::of(&den.to_string()),
            );
            let mut r: PrettyResult = PrintResult::new(picture, Binding::Negation);
            r.unsigned = Some(magnitude);
            r.stacked = true;
            return r;
        }
        let mut r: PrettyResult = PrintResult::new(magnitude, Binding::Fraction);
        r.stacked = true;
        return r;
    }
    // a quotient on one row is not an atom
    let magnitude: Picture = Picture::of(&format!("{}/{}", magnitude_num, den));
    if negative_value {
        return negative(magnitude, false);
    }
    PrintResult::new(magnitude, Binding::Quotient)
}

fn ordered_terms(p: &PrettyPrinter, terms: &[Basic]) -> Vec<Basic> {
    if p.settings.order.as_deref() == Some("none") || !crate::core::printing::in_compare_order(terms) {
        return terms.to_vec();
    }
    crate::core::printing::ordered_terms(terms)
}

fn ordered_factors(p: &PrettyPrinter, factors: &[Basic]) -> Vec<Basic> {
    if p.settings.order.as_deref() == Some("none") {
        return factors.to_vec();
    }
    crate::core::printing::printing_order(factors)
}

fn add_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    sum_of(p, &ordered_terms(p, e.args()), ctx)
}

/// The terms in a row, joined by `+`; a negated term is subtracted; a
/// fraction among tall terms is stacked (`sum_of`).
pub fn sum_of(p: &PrettyPrinter, terms: &[Basic], ctx: &PrettyContext) -> R {
    let c: PrettyContext = ctx.term();
    let mut results: Vec<Option<PrettyResult>> = Vec::new();
    for term in terms {
        let fraction: bool = rational_parts(term).is_some_and(|(_, q)| q > Int::ONE);
        results.push(if fraction { None } else { Some(p.rule(term, &c)?) });
    }
    let large: bool = results.iter().flatten().any(|r| r.output.height() > 1);
    let mut parts: Vec<Picture> = Vec::new();
    let mut open_right: bool = false;
    for (i, term) in terms.iter().enumerate() {
        let result: PrettyResult = match results[i].take() {
            Some(r) => r,
            None => p.rule(term, &if large { c.displayed() } else { c.clone() })?,
        };
        if let Some(magnitude) = &result.unsigned {
            // a - b, not a + -b
            let sign: &str = if i > 0 {
                " - "
            } else if magnitude.height() > 1 {
                "- "
            } else {
                "-"
            };
            parts.push(magnitude.ltext(sign));
            open_right = result.open_ended;
            continue;
        }
        let closed: PrettyResult = p.enclose(result, &c);
        if i > 0 {
            parts.push(Picture::of(" + "));
        }
        open_right = closed.open_ended;
        parts.push(closed.output);
    }
    let refs: Vec<&Picture> = parts.iter().collect();
    let mut r: PrettyResult = PrintResult::new(Picture::beside(&refs), Binding::Sum);
    r.open_ended = open_right;
    Ok(r)
}

/// The factors joined by the multiplication sign, and whether the product
/// is open to the right (`_product`).
fn product(
    p: &PrettyPrinter,
    factors: &[Basic],
    ctx: &PrettyContext,
    alone: Option<&PrettyContext>,
) -> PolyResult<(Picture, bool)> {
    if factors.is_empty() {
        return Ok((Picture::of("1"), false));
    }
    if factors.len() == 1 {
        if let Some(alone) = alone {
            let result: PrettyResult = p.print(&factors[0], alone)?;
            return Ok((result.output, result.open_ended));
        }
    }
    let mut parts: Vec<Picture> = Vec::new();
    let last: usize = factors.len() - 1;
    let mut open_right: bool = false;
    for (i, f) in factors.iter().enumerate() {
        // a factor open to the right captures the factors after it
        let result: PrettyResult = p.print(f, &ctx.factor(i < last))?;
        open_right = result.open_ended;
        parts.push(result.output);
    }
    Ok((p.join(&parts, p.symbology.xsym("*")), open_right))
}

fn is_negative_number(e: &Basic) -> bool {
    rational_parts(e).is_some_and(|(n, _)| n < Int::ZERO)
        || (e.is_float() && e.exact_value().is_some_and(crate::numbers::rat_is_negative))
}

fn absolute(e: &Basic) -> PolyResult<Basic> {
    if is_negative_number(e) {
        return crate::core::mul::neg(e);
    }
    Ok(e.clone())
}

fn mul_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let args: Vec<Basic> = ordered_factors(p, e.args());
    let first: &Basic = &args[0];
    let signed: bool = is_negative_number(first);
    // an unevaluated product is printed factor by factor
    let plain: bool = *first == Basic::one() || args[1..].iter().any(is_number_node);
    let mut numerator: Vec<Basic> = Vec::new();
    let mut denominator: Vec<Basic> = Vec::new();
    if plain {
        numerator.push(if signed { absolute(first)? } else { first.clone() });
        numerator.extend(args[1..].iter().cloned());
    } else {
        for (i, item) in args.iter().enumerate() {
            if item.is_pow()
                && is(item, Predicate::commutative) != Some(false)
                && rational_parts(&item.args()[1]).is_some_and(|(n, _)| n < Int::ZERO)
            {
                let minus: Basic = crate::core::mul::neg(&item.args()[1])?;
                denominator.push(if minus == Basic::one() {
                    item.args()[0].clone()
                } else {
                    crate::core::power::pow(&item.args()[0], &minus)?
                });
            } else if let Some((n, q)) = rational_parts(item) {
                let magnitude: Int = if n < Int::ZERO { -n.clone() } else { n.clone() };
                if magnitude != Int::ONE {
                    numerator.push(Basic::integer(magnitude));
                }
                if q != Int::ONE {
                    denominator.push(Basic::integer(q));
                }
            } else if i == 0 && signed {
                // a negative float first
                numerator.push(absolute(item)?);
            } else {
                numerator.push(item.clone());
            }
        }
    }
    if denominator.is_empty() {
        let (picture, open_right) = product(p, &numerator, ctx, None)?;
        if signed {
            let mut r: PrettyResult = PrintResult::new(picture.ltext("-"), Binding::Negation);
            r.unsigned = Some(picture);
            r.open_ended = open_right;
            return Ok(r);
        }
        let mut r: PrettyResult = PrintResult::new(picture, Binding::Product);
        r.open_ended = open_right;
        return Ok(r);
    }
    let numerator_ctx: PrettyContext = ctx.numerator(false);
    let (top, _) = product(p, &numerator, ctx, if signed { None } else { Some(&numerator_ctx) })?;
    let (bottom, _) = product(p, &denominator, ctx, Some(&ctx.denominator(false)))?;
    let magnitude: Picture = p.fraction(&top, &bottom);
    if signed {
        // (SymPy's layout: the minus sign in the numerator, followed by a space)
        let picture: Picture = p.fraction(&top.ltext("-").rtext(" "), &bottom);
        let mut r: PrettyResult = PrintResult::new(picture, Binding::Negation);
        r.unsigned = Some(magnitude);
        r.stacked = true;
        return Ok(r);
    }
    let mut r: PrettyResult = PrintResult::new(magnitude, Binding::Fraction);
    r.stacked = true;
    Ok(r)
}

/// `1/denominator` with a fraction bar (`reciprocal`).
fn reciprocal(p: &PrettyPrinter, denominator: &Basic, ctx: &PrettyContext) -> R {
    let bottom: Picture = p.operand(denominator, &ctx.denominator(false))?;
    let mut r: PrettyResult = PrintResult::new(p.fraction(&Picture::of("1"), &bottom), Binding::Fraction);
    r.stacked = true;
    Ok(r)
}

/// `base` with the picture `exponent` raised; a one-row exponent is
/// offered to the base (`raised`).
pub fn raised(p: &PrettyPrinter, base: &Basic, exponent: &Picture, ctx: &PrettyContext) -> R {
    let one_row: bool = exponent.height() == 1;
    let c: PrettyContext = ctx.at(
        Position::Base,
        if one_row { Binding::Power } else { Binding::Application },
        ChildOptions {
            strict: true,
            exponent_offered: if one_row { Some(exponent.clone()) } else { None },
            ..ChildOptions::default()
        },
    );
    let result: PrettyResult = p.rule(base, &c)?;
    if result.absorbed {
        return Ok(PrintResult::new(result.output, Binding::Power));
    }
    let enclosed: PrettyResult = p.enclose(result, &c);
    Ok(PrintResult::new(p.power(&enclosed.output, exponent), Binding::Power))
}

/// The picture of an exponent: raised, it needs no brackets, but a power
/// in it gets them (`exponent_picture`).
pub fn exponent_picture(p: &PrettyPrinter, exponent: &Basic, ctx: &PrettyContext) -> PolyResult<Picture> {
    p.operand(
        exponent,
        &ctx.at(
            Position::Exponent,
            Binding::Lowest,
            ChildOptions {
                excluded: Some(Binding::Power),
                ..ChildOptions::default()
            },
        ),
    )
}

fn root(p: &PrettyPrinter, base: &Basic, index: &Basic, ctx: &PrettyContext) -> PolyResult<Option<PrettyResult>> {
    let square: bool = is_integer_node(index) && *index == Basic::int(2);
    let index_picture: Option<Picture> = if square {
        None
    } else {
        Some(p.operand(index, &ctx.subscript())?)
    };
    if index_picture.as_ref().is_some_and(|i| i.height() != 1) {
        return Ok(None);
    }
    let radicand: Picture = p.operand(base, &ctx.delimited())?;
    let small: bool =
        radicand.width() == 1 || (is_integer_node(base) && rational_parts(base).is_some_and(|(n, _)| n >= Int::ZERO));
    Ok(Some(p.root(&radicand, index_picture.as_ref(), small)))
}

fn pow_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let (base, exp) = (&e.args()[0], &e.args()[1]);
    if is(base, Predicate::commutative) != Some(false) {
        if *exp == Basic::negative_one() {
            return reciprocal(p, base, ctx);
        }
        if p.settings.root_notation {
            let mut index: Option<Basic> = None;
            if let Some((n, q)) = rational_parts(exp) {
                if n == Int::ONE && q > Int::ONE {
                    index = Some(Basic::integer(q));
                }
            } else if exp.is_pow() && exp.args()[0].is_symbol() && exp.args()[1] == Basic::negative_one() {
                index = Some(exp.args()[0].clone());
            }
            if let Some(index) = index {
                if let Some(rooted) = root(p, base, &index, ctx)? {
                    return Ok(rooted);
                }
            }
        }
        if rational_parts(exp).is_some_and(|(n, _)| n < Int::ZERO) {
            let positive: Basic = crate::core::mul::neg(exp)?;
            return reciprocal(p, &crate::core::power::pow(base, &positive)?, ctx);
        }
    }
    let exponent: Picture = exponent_picture(p, exp, ctx)?;
    raised(p, base, &exponent, ctx)
}

fn connective(p: &PrettyPrinter, args: &[Basic], ctx: &PrettyContext, name: &str, atom: &str, binding: Binding) -> R {
    if !p.unicode() {
        return p.function(name, args, ctx);
    }
    // what binds less than a relation is bracketed: the other connectives
    infix(p, args, &p.symbology.atom(atom, name), binding, &ctx.connective())
}

/// A connective with a negated sign: it takes the negation its parent
/// offers and then stands for the negation (`_negatable`).
fn negatable(p: &PrettyPrinter, args: &[Basic], ctx: &PrettyContext, name: &str, atom: &str, negated: &str) -> R {
    if p.unicode() && ctx.negation_offered {
        let mut r: PrettyResult = connective(p, args, ctx, name, negated, Binding::Implication)?;
        r.absorbed = true;
        return Ok(r);
    }
    connective(p, args, ctx, name, atom, Binding::Implication)
}

fn not_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    if !p.unicode() {
        return p.function("Not", e.args(), ctx);
    }
    // the operand is bracketed unless it is an atom, an application or a negation
    let c: PrettyContext = ctx.at(
        Position::Negated,
        Binding::Affix,
        ChildOptions {
            negation_offered: true,
            ..ChildOptions::default()
        },
    );
    let result: PrettyResult = p.rule(&e.args()[0], &c)?;
    if result.absorbed {
        let mut r: PrettyResult = result;
        r.absorbed = false;
        return Ok(r);
    }
    let enclosed: PrettyResult = p.enclose(result, &c);
    Ok(PrintResult::new(
        enclosed.output.ltext(&p.symbology.atom("Not", "~")),
        Binding::Affix,
    ))
}
