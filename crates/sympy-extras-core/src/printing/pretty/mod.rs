//! Port of `sympy_extras/printing/pretty`: the pretty printer
//! (expressions as two-dimensional text) on the typed framework of the
//! printers.

pub mod engine;
pub mod layout;
pub mod rules;
pub mod rules_calculus;
pub mod rules_functions;
pub mod rules_other;
pub mod rules_sets;
pub mod symbology;

use crate::core::basic::Basic;
use crate::error::PolyResult;
use crate::printing::framework::{Context, TreePrinter};

/// The picture of `expr`, not cut into screens (`pretty_picture`).
pub fn pretty_picture(expr: &Basic, settings: engine::PrettySettings) -> PolyResult<layout::Picture> {
    let printer: engine::PrettyPrinter = engine::PrettyPrinter::new(settings)?;
    Ok(printer.rule(expr, &Context::default())?.output)
}

/// `expr` as two-dimensional text (`pretty`); the rows have the same width.
pub fn pretty(expr: &Basic, settings: engine::PrettySettings) -> PolyResult<String> {
    let (wrap_line, num_columns, unicode) = (settings.wrap_line, settings.num_columns, settings.use_unicode);
    let picture: layout::Picture = pretty_picture(expr, settings)?;
    Ok(picture.render(wrap_line, num_columns, unicode))
}
