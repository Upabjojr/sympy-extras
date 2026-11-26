//! Port of `sympy_extras/printing/pretty/engine.py`: the pretty printer on
//! the typed framework of the printers, and the layouts shared by its
//! rules. The table of rules of the Python (a `Dispatcher`) is a `match`
//! on the head of the node ([`super::rules::rule`]); a node without a rule
//! prints as its string form.

// PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (the layouts: _print_nth_root, _print_seq,
// _hprint_vseparator, prettyForm.__pow__, prettyForm.__truediv__) (SymPy 1.14, BSD license; through
// sympy_extras/printing/pretty/engine.py)

use super::layout::{Picture, PrettyContext, PrettyResult};
use super::symbology::Symbology;
use crate::core::basic::Basic;
use crate::error::{not_ported, PolyResult};
use crate::printing::framework::{Binding, PrintResult, TreePrinter};

/// The settings of a pretty printer (`PrettySettings`).
#[derive(Clone, Debug)]
pub struct PrettySettings {
    pub use_unicode: bool,
    pub wrap_line: bool,
    pub num_columns: Option<usize>,
    /// `None` (the printing order) or `"none"`; the monomial orders are not ported
    pub order: Option<String>,
    pub root_notation: bool,
    pub use_unicode_sqrt_char: bool,
    pub imaginary_unit: String,
    /// the floats with all their digits; `None`: a float alone only
    pub full_prec: Option<bool>,
}

impl Default for PrettySettings {
    fn default() -> PrettySettings {
        PrettySettings {
            use_unicode: false,
            wrap_line: true,
            num_columns: None,
            order: None,
            root_notation: true,
            use_unicode_sqrt_char: true,
            imaginary_unit: "i".to_string(),
            full_prec: None,
        }
    }
}

impl PrettySettings {
    /// The checks of `__post_init__`.
    pub fn check(&self) -> PolyResult<()> {
        if self.imaginary_unit != "i" && self.imaginary_unit != "j" {
            return Err(crate::error::value_error(format!(
                "'imaginary_unit' must be either 'i' or 'j', not '{}'",
                self.imaginary_unit
            )));
        }
        if let Some(order) = &self.order {
            if order != "none" {
                return Err(not_ported(&format!("the pretty printing order {:?}", order)));
            }
        }
        Ok(())
    }
}

/// Prints nodes as pictures (`PrettyPrinter`).
pub struct PrettyPrinter {
    pub settings: PrettySettings,
    pub symbology: Symbology,
}

impl PrettyPrinter {
    pub fn new(settings: PrettySettings) -> PolyResult<PrettyPrinter> {
        settings.check()?;
        let symbology: Symbology = Symbology {
            unicode: settings.use_unicode,
            imaginary_unit: settings.imaginary_unit.clone(),
        };
        Ok(PrettyPrinter { settings, symbology })
    }

    pub fn unicode(&self) -> bool {
        self.settings.use_unicode
    }

    /// A node without a rule: its string form, on one row (`generic`).
    pub fn generic(&self, e: &Basic) -> PrettyResult {
        self.opaque(&crate::core::printing::sstr(e))
    }

    /// A string of unknown structure: an atom when it is one token, an
    /// application when it reads `name(...)`, open otherwise (`opaque`).
    pub fn opaque(&self, text: &str) -> PrettyResult {
        if is_token(text) {
            return PrintResult::new(Picture::of(text), Binding::Atom);
        }
        let binding: Binding = if is_application(text) {
            Binding::Application
        } else {
            Binding::Lowest
        };
        PrintResult::new(Picture::of(text), binding)
    }

    /// The picture of the child `e` in `context`, bracketed if needed (`operand`).
    pub fn operand(&self, e: &Basic, context: &PrettyContext) -> PolyResult<Picture> {
        Ok(self.print(e, context)?.output)
    }

    /// A single token (`atom`).
    pub fn atom(&self, s: &str) -> PrettyResult {
        PrintResult::new(Picture::of(s), Binding::Atom)
    }

    /// The pictures in a row, separated by `delimiter` (`join`).
    pub fn join(&self, pictures: &[Picture], delimiter: &str) -> Picture {
        let d: Picture = Picture::of(delimiter);
        let mut parts: Vec<&Picture> = Vec::new();
        for p in pictures {
            if !parts.is_empty() {
                parts.push(&d);
            }
            parts.push(p);
        }
        Picture::beside(&parts)
    }

    /// `picture` between the delimiters of a notation (`parens`).
    pub fn parens(&self, picture: &Picture, left: &str, right: &str, one_row_in_ascii: bool) -> Picture {
        picture.parens(&self.symbology, left, right, one_row_in_ascii)
    }

    /// The arguments in a row, each one delimited by its parent (`arguments`).
    pub fn arguments(&self, args: &[Basic], context: &PrettyContext, delimiter: &str) -> PolyResult<Picture> {
        let mut pictures: Vec<Picture> = Vec::new();
        for a in args {
            pictures.push(self.operand(a, &context.delimited())?);
        }
        Ok(self.join(&pictures, delimiter))
    }

    /// A delimited sequence: `(a, b)`, `[a, b]`, `{a, b}` (`sequence`).
    pub fn sequence(
        &self,
        args: &[Basic],
        context: &PrettyContext,
        left: &str,
        right: &str,
        delimiter: &str,
    ) -> PolyResult<PrettyResult> {
        let inner: Picture = self.arguments(args, context, delimiter)?;
        Ok(PrintResult::new(self.parens(&inner, left, right, true), Binding::Atom))
    }

    /// `head(arguments)`: the exponent which the parent offers goes between
    /// the function and its arguments (`applied`).
    pub fn applied(&self, head: &Picture, arguments: &Picture, context: &PrettyContext) -> PrettyResult {
        let inside: Picture = self.parens(arguments, "(", ")", false);
        if let Some(exponent) = &context.exponent_offered {
            let raised: Picture = exponent.with_baseline(head.baseline + exponent.height() as i64);
            let mut r: PrettyResult = PrintResult::new(Picture::beside(&[head, &raised, &inside]), Binding::Power);
            r.absorbed = true;
            return r;
        }
        PrintResult::new(Picture::beside(&[head, &inside]), Binding::Application)
    }

    /// `name(args)` (`function`).
    pub fn function(&self, name: &str, args: &[Basic], context: &PrettyContext) -> PolyResult<PrettyResult> {
        let arguments: Picture = self.arguments(args, context, ", ")?;
        Ok(self.applied(&Picture::of(name), &arguments, context))
    }

    /// A fraction with a horizontal bar (`fraction`).
    pub fn fraction(&self, numerator: &Picture, denominator: &Picture) -> Picture {
        let width: usize = numerator.width().max(denominator.width());
        let bar: Picture = Picture::of(&self.symbology.hobj("-", width));
        Picture::new(
            Picture::stacked(&[numerator, &bar, denominator]),
            numerator.height() as i64,
        )
    }

    /// `base` with `exponent` raised at its right (`power`).
    pub fn power(&self, base: &Picture, exponent: &Picture) -> Picture {
        let top: Picture = Picture::beside(&[&Picture::blank(base.width(), 1), exponent]);
        let bottom: Picture = Picture::beside(&[base, &Picture::blank(exponent.width(), 1)]);
        bottom.above(&[&top])
    }

    /// `base` with `subscript` lowered at its right (`subscripted`).
    pub fn subscripted(&self, base: &Picture, subscript: &Picture) -> Picture {
        let bottom: Picture = Picture::beside(&[&Picture::blank(base.width(), 1), subscript]);
        let top: Picture = Picture::beside(&[base, &Picture::blank(subscript.width(), 1)]);
        top.below(&[&bottom])
    }

    /// `left_part | right_part` between delimiters, the bar as tall as the
    /// parts (`separated`).
    pub fn separated(&self, left_part: &Picture, right_part: &Picture, left: &str, right: &str) -> Picture {
        if !left.is_empty() && !self.unicode() {
            let row: Picture = Picture::beside(&[
                left_part,
                &Picture::of(" "),
                &Picture::of("|"),
                &Picture::of(" "),
                right_part,
            ]);
            return self.parens(&row, left, right, true);
        }
        let spacer: &str = if left.is_empty() { "" } else { " " };
        let mut probe: Picture = Picture::beside(&[left_part, &Picture::of(spacer), right_part]);
        if !left.is_empty() {
            // the bar is as tall as the delimiters (a brace has an odd height)
            probe = self.parens(&probe, left, right, true);
        }
        let bar: Picture = Picture::text(&self.symbology.vobj("|", probe.height()), probe.baseline);
        let row: Picture = Picture::beside(&[left_part, &Picture::of(spacer), &bar, &Picture::of(spacer), right_part]);
        if left.is_empty() {
            row
        } else {
            self.parens(&row, left, right, true)
        }
    }

    /// The root sign over `radicand`; `index` is `None` for a square root;
    /// `small` allows the one-character sign (`root`, SymPy's `_print_nth_root`).
    pub fn root(&self, radicand: &Picture, index: Option<&Picture>, small: bool) -> PrettyResult {
        let symbology: &Symbology = &self.symbology;
        if index.is_none() && small && self.settings.use_unicode_sqrt_char && radicand.height() == 1 {
            if let Some(sign) = symbology.root(2) {
                // no bar closes the radicand: it binds like a prefix operator
                return PrintResult::new(radicand.ltext(sign), Binding::Root);
            }
        }
        let slash: &str = symbology.xobj("/", 1)[0];
        let mut rootsign: String = format!("{}{}", symbology.xobj("\\", 1)[0], slash);
        let exp: String = match index {
            None => String::new(),
            Some(i) => format!("{:<2}", i.to_string()),
        };
        let exp_len: usize = exp.chars().count();
        if exp_len > 2 {
            rootsign = format!("{}{}", " ".repeat(exp_len - 2), rootsign);
        }
        let mut drawn: Picture = if exp.is_empty() {
            Picture::new(vec![String::new(), rootsign], 0)
        } else {
            Picture::text(&format!("{}\n{}", exp, rootsign), 0)
        };
        let length: usize = radicand.height().saturating_sub(1);
        if length > 0 {
            let rows: Vec<String> = (0..length)
                .map(|i| format!("{}{}{}", " ".repeat(length - i - 1), slash, " ".repeat(i)))
                .collect();
            let diagonal: Picture = Picture::new(rows, length as i64 - 1);
            drawn = drawn.right(&[&diagonal]);
        }
        drawn = drawn.with_baseline(radicand.baseline.max(1));
        let bar: Picture = Picture::of(&symbology.hobj("_", 2 + radicand.width()));
        PrintResult::new(radicand.above(&[&bar]).left(&[&drawn]), Binding::Atom)
    }
}

impl TreePrinter<Basic, Picture> for PrettyPrinter {
    fn rule(&self, node: &Basic, context: &PrettyContext) -> PolyResult<PrettyResult> {
        match super::rules::rule(self, node, context)? {
            Some(r) => Ok(r),
            None => Ok(self.generic(node)),
        }
    }

    /// The picture of `result` between parentheses (`bracket`).
    fn bracket(&self, result: &PrettyResult, _context: &PrettyContext) -> Picture {
        result.output.round(&self.symbology)
    }
}

fn is_token(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Whether `text` is a name followed by one bracketed group (`_is_application`).
fn is_application(text: &str) -> bool {
    let Some(start) = text.find('(') else {
        return false;
    };
    if start == 0 || !is_token(&text[..start]) || !text.ends_with(')') {
        return false;
    }
    let mut depth: i64 = 0;
    for (i, c) in text.char_indices().skip_while(|(i, _)| *i < start) {
        if c == '(' {
            depth += 1;
        } else if c == ')' {
            depth -= 1;
            if depth == 0 {
                return i == text.len() - 1;
            }
        }
    }
    false
}
