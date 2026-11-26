//! Port of the parts of `sympy_extras/printing/pretty/rules_other.py` on
//! the nodes of the core: the quantifiers and the non-elementary
//! integrals. Not ported yet: the roots of polynomials, the sequences and
//! the series.

// PORTED FROM SYMPY: through sympy_extras/printing/pretty/rules_other.py (the quantifiers are original)

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
