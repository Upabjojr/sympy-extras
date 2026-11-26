//! Port of the parts of `sympy_extras/printing/pretty/rules_other.py` on
//! the nodes of the core: the quantifiers, the non-elementary integrals,
//! the roots of polynomials. Not ported yet: the sequences and the series.

// PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_ComplexRootOf, _print_RootSum,
// _print_Lambda (SymPy 1.14, BSD license; through sympy_extras/printing/pretty/rules_other.py; the
// quantifiers are original)

use super::engine::PrettyPrinter;
use super::layout::{Picture, PrettyContext, PrettyResult};
use crate::core::basic::{Basic, Head};
use crate::error::PolyResult;
use crate::printing::framework::{Binding, ChildOptions, Position, PrintResult, TreePrinter};

/// The rule of a quantifier or a non-elementary integral, `None` for the
/// other nodes.
pub fn other_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> PolyResult<Option<PrettyResult>> {
    Ok(Some(match e.head() {
        Head::ForAll | Head::Exists => quantifier(p, e, ctx)?,
        Head::NonElementaryIntegral => {
            // an integral proved not to be elementary is written as an integral
            let integral: Basic = crate::integrals::integral::integral(&e.args()[0], &e.args()[1], None)?;
            p.rule(&integral, ctx)?
        }
        Head::ComplexRootOf(_) => {
            let index: usize = crate::polys::complexrootof::index_of_node(e);
            let inside: Picture = p.join(
                &[
                    p.operand(&e.args()[0], &ctx.delimited())?,
                    Picture::of(&index.to_string()),
                ],
                ", ",
            );
            p.applied(&Picture::of("CRootOf"), &inside, ctx)
        }
        Head::RootSum => {
            let (f, var, body, _) = crate::polys::rootoftools::parts(e);
            // the function is a Lambda of one variable (`_lambda`)
            let arrow: String = format!(" {} ", p.symbology.atom("ArrowFromBar", "->"));
            let fun: Picture = Picture::beside(&[
                &p.operand(var, &ctx.delimited())?,
                &Picture::of(&arrow),
                &p.operand(body, &ctx.at(Position::Body, Binding::Lowest, ChildOptions::default()))?,
            ]);
            let inside: Picture = p.join(&[p.operand(f, &ctx.delimited())?, fun], ", ");
            p.applied(&Picture::of("RootSum"), &inside, ctx)
        }
        Head::IndexedRoot => {
            let (f, t, k) = crate::polys::cad::indexedroot::shown_parts(e);
            let inside: Picture = p.join(
                &[
                    p.operand(&f, &ctx.delimited())?,
                    p.operand(&t, &ctx.delimited())?,
                    Picture::of(&k.to_string()),
                ],
                ", ",
            );
            p.applied(&Picture::of("IndexedRoot"), &inside, ctx)
        }
        // no notation of its own: its string form (`generic`)
        Head::TranscendentalRoot => p.generic(e),
        _ => return Ok(None),
    }))
}

fn quantifier(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> PolyResult<PrettyResult> {
    let name: &str = if matches!(e.head(), Head::ForAll) {
        "ForAll"
    } else {
        "Exists"
    };
    let variables: &[Basic] = e.args()[0].args();
    let formula: &Basic = &e.args()[1];
    if !p.unicode() {
        let first: Picture = if variables.len() == 1 {
            p.operand(&variables[0], &ctx.delimited())?
        } else {
            p.sequence(variables, ctx, "(", ")", ", ")?.output
        };
        let inside: Picture = p.join(&[first, p.operand(formula, &ctx.delimited())?], ", ");
        return Ok(p.applied(&Picture::of(name), &inside, ctx));
    }
    let sign: String = p.symbology.atom(name, name);
    let head: Picture = p.arguments(variables, ctx, ", ")?.ltext(&sign);
    // a prefix operator: its operand is bracketed unless it is an atom, an
    // application, a negation or another quantified formula
    let body: Picture = p.operand(
        formula,
        &ctx.at(Position::Operand, Binding::Affix, ChildOptions::default()),
    )?;
    Ok(PrintResult::new(
        Picture::beside(&[&head, &Picture::of(" "), &body]),
        Binding::Affix,
    ))
}
