//! Port of `sympy_extras/printing/pretty/rules_sets.py`: the
//! pretty-printing rules of the sets.

// PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_Interval, _print_FiniteSet, the set
// operations, _print_ProductSet, _print_ImageSet, _print_ConditionSet, _print_Contains) (SymPy 1.14,
// BSD license; through sympy_extras/printing/pretty/rules_sets.py)

use super::engine::PrettyPrinter;
use super::layout::{Picture, PrettyContext, PrettyResult};
use super::rules::{infix, raised};
use crate::core::basic::{Basic, Head};
use crate::error::PolyResult;
use crate::printing::framework::{Binding, PrintResult};

type R = PolyResult<PrettyResult>;

fn sorted(args: &[Basic]) -> Vec<Basic> {
    let mut v: Vec<Basic> = args.to_vec();
    v.sort_by_cached_key(crate::core::sorting::sort_key);
    v
}

fn operation(p: &PrettyPrinter, args: &[Basic], ctx: &PrettyContext, operator: &str) -> R {
    infix(
        p,
        args,
        operator,
        Binding::SetOperation,
        &ctx.right(Binding::SetOperation),
    )
}

/// The rule of a set, `None` for the other nodes.
pub fn set_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> PolyResult<Option<PrettyResult>> {
    let s = &p.symbology;
    Ok(Some(match e.head() {
        Head::EmptySet => p.atom(&s.atom("EmptySet", "EmptySet")),
        Head::UniversalSet => p.atom(&s.atom("Universe", "UniversalSet")),
        Head::Reals => p.atom(&s.atom("Reals", "(-oo, oo)")),
        Head::Naturals => p.atom(&s.atom("Naturals", "Naturals")),
        Head::Naturals0 => p.atom(&s.atom("Naturals0", "Naturals0")),
        Head::Integers => p.atom(&s.atom("Integers", "Integers")),
        Head::Rationals => p.atom(&s.atom("Rationals", "Rationals")),
        Head::Complexes => p.atom(&s.atom("Complexes", "Complexes")),
        Head::Interval => {
            let (start, end) = (&e.args()[0], &e.args()[1]);
            if start == end {
                return Ok(Some(p.sequence(std::slice::from_ref(start), ctx, "{", "}", ", ")?));
            }
            // an infinite end is not attained: it is drawn open
            let infinite = |v: &Basic| matches!(v.head(), Head::Infinity | Head::NegativeInfinity);
            let left_open: bool = matches!(e.args()[2].head(), Head::BooleanTrue) || infinite(start);
            let right_open: bool = matches!(e.args()[3].head(), Head::BooleanTrue) || infinite(end);
            p.sequence(
                &[start.clone(), end.clone()],
                ctx,
                if left_open { "(" } else { "[" },
                if right_open { ")" } else { "]" },
                ", ",
            )?
        }
        Head::FiniteSet | Head::PointSet | Head::SetOfSets => p.sequence(&sorted(e.args()), ctx, "{", "}", ", ")?,
        Head::Union => operation(p, e.args(), ctx, &s.atom("Union", "U"))?,
        Head::Intersection => operation(p, e.args(), ctx, &s.atom("Intersection", "n"))?,
        Head::Complement => operation(p, e.args(), ctx, "\\")?,
        Head::ProductSet => {
            let sets: &[Basic] = e.args();
            if !sets.is_empty() && sets.iter().all(|x| *x == sets[0]) {
                // the power of a set: the base is bracketed like any base
                return Ok(Some(raised(p, &sets[0], &Picture::of(&sets.len().to_string()), ctx)?));
            }
            operation(p, sets, ctx, &s.atom("Multiplication", "x"))?
        }
        Head::PowerSet => p.function("PowerSet", e.args(), ctx)?,
        Head::Contains => {
            if !p.unicode() {
                return Ok(Some(p.function("Contains", e.args(), ctx)?));
            }
            // a relation, the element bracketed like a side of a relation
            let element: Picture = p.operand(&e.args()[0], &ctx.side())?;
            let domain: Picture = p.operand(&e.args()[1], &ctx.side())?;
            let sign: Picture = Picture::of(&format!(" {} ", s.atom("ElementOf", "in")));
            PrintResult::new(Picture::beside(&[&element, &sign, &domain]), Binding::Relation)
        }
        Head::ImageSet => {
            let image: Picture = p.operand(&e.args()[1], &ctx.delimited())?;
            let mut members: Vec<Picture> = Vec::new();
            for (v, set) in e.args()[0].args().iter().zip(&e.args()[2..]) {
                members.push(membership(p, v, set, ctx)?);
            }
            PrintResult::new(p.separated(&image, &p.join(&members, ", "), "{", "}"), Binding::Atom)
        }
        Head::ConditionSet => {
            let (sym, condition, base_set) = (&e.args()[0], &e.args()[1], &e.args()[2]);
            let variables: Picture = p.operand(sym, &ctx.delimited())?;
            if matches!(base_set.head(), Head::UniversalSet) {
                let condition: Picture = p.operand(condition, &ctx.delimited())?;
                return Ok(Some(PrintResult::new(
                    p.separated(&variables, &condition, "{", "}"),
                    Binding::Atom,
                )));
            }
            // the condition is an operand of "and"
            let condition: Picture = p.operand(condition, &ctx.connective())?;
            let described: Picture = Picture::beside(&[
                &membership(p, sym, base_set, ctx)?,
                &Picture::of(&format!(" {} ", s.atom("And", "and"))),
                &condition,
            ]);
            PrintResult::new(p.separated(&variables, &described, "{", "}"), Binding::Atom)
        }
        _ => return Ok(None),
    }))
}

fn membership(p: &PrettyPrinter, variable: &Basic, domain: &Basic, ctx: &PrettyContext) -> PolyResult<Picture> {
    let inn: String = p.symbology.atom("SmallElementOf", "in");
    Ok(Picture::beside(&[
        &p.operand(variable, &ctx.delimited())?,
        &Picture::of(&format!(" {} ", inn)),
        &p.operand(domain, &ctx.side())?,
    ]))
}
