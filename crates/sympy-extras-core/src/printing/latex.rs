//! Port of `sympy_extras/printing/latex.py` and `_latex_functions.py`: the
//! LaTeX of the expressions of the core, on the typed framework of
//! [`super::framework`] (a rule returns the LaTeX and the binding of what
//! it printed, a parent passes each child its context, the brackets are
//! decided in [`needs_brackets`]). The rules are a `match` on the head of
//! the node. Not ported yet (a "not ported" error): `Float`, the matrices
//! and polynomials, `Lambda`, `Limit`, `Order`, `AccumBounds` and the nodes
//! of `_latex_objects.py` (`Piecewise`, sums, integrals, sets, sequences,
//! series, roots).

// PORTED FROM SYMPY: sympy/printing/latex.py (SymPy 1.14, BSD license; through
// sympy_extras/printing/latex.py and _latex_functions.py: rules by class, typed results and contexts,
// the defects of sympy/sympy#30844 fixed)

use super::framework::{needs_brackets, Binding, Context, Position, PrintResult, TreePrinter};
use super::latex_names::{escape_name, escape_text, name_latex, operator_name, translate, ACCEPTED_FUNCTIONS};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::error::{not_ported, PolyResult};

type Ctx = Context<String>;
type Res = PrintResult<String>;
type R = PolyResult<Res>;

/// The settings of [`latex`] (SymPy's; `LatexSettings`).
#[derive(Clone, Debug)]
pub struct LatexSettings {
    pub mode: String,
    pub fold_frac_powers: bool,
    pub fold_func_brackets: bool,
    pub fold_short_frac: Option<bool>,
    pub inv_trig_style: String,
    pub itex: bool,
    pub ln_notation: bool,
    pub long_frac_ratio: Option<f64>,
    pub mul_symbol: Option<String>,
    /// `None` (the printing order) or `"none"` (the stored arguments); the
    /// monomial orders are not ported
    pub order: Option<String>,
    pub root_notation: bool,
    pub imaginary_unit: String,
    pub gothic_re_im: bool,
    pub decimal_separator: String,
    pub parenthesize_super: bool,
    pub diff_operator: String,
    /// the floats with all the digits of their precision
    pub full_prec: bool,
    /// the bounds of the fixed-point notation of the floats (mpmath's
    /// `min_fixed` and `max_fixed`)
    pub min: Option<i64>,
    pub max: Option<i64>,
}

impl Default for LatexSettings {
    fn default() -> LatexSettings {
        LatexSettings {
            mode: "plain".to_string(),
            fold_frac_powers: false,
            fold_func_brackets: false,
            fold_short_frac: None,
            inv_trig_style: "abbreviated".to_string(),
            itex: false,
            ln_notation: false,
            long_frac_ratio: None,
            mul_symbol: None,
            order: None,
            root_notation: true,
            imaginary_unit: "i".to_string(),
            gothic_re_im: false,
            decimal_separator: "period".to_string(),
            parenthesize_super: true,
            diff_operator: "d".to_string(),
            full_prec: false,
            min: None,
            max: None,
        }
    }
}

fn value_error(message: &str) -> crate::error::PolyError {
    crate::error::value_error(message.to_string())
}

/// The separator of SymPy's named multiplication symbols.
fn named_mul_symbol(symbol: Option<&str>) -> Option<&'static str> {
    match symbol {
        None => Some(" "),
        Some("ldot") => Some(r" \,.\, "),
        Some("dot") => Some(r" \cdot "),
        Some("times") => Some(r" \times "),
        _ => None,
    }
}

/// The printer of [`latex`]: the settings (never modified) and the helpers
/// the rules share (`LatexPrinter`).
pub struct LatexPrinter {
    pub settings: LatexSettings,
    fold_short_frac: bool,
    /// between two factors
    separator: String,
    /// between two numbers
    number_separator: String,
    imaginary_unit: String,
    diff_operator: String,
    /// between the elements of a container and the arguments of a function
    comma: String,
}

impl LatexPrinter {
    pub fn new(s: LatexSettings) -> PolyResult<LatexPrinter> {
        if !["plain", "inline", "equation", "equation*"].contains(&s.mode.as_str()) {
            return Err(value_error(
                "'mode' must be one of 'inline', 'plain', 'equation' or 'equation*'",
            ));
        }
        if !["period", "comma"].contains(&s.decimal_separator.as_str()) {
            return Err(value_error("'decimal_separator' must be 'period' or 'comma'"));
        }
        if !["abbreviated", "full", "power"].contains(&s.inv_trig_style.as_str()) {
            return Err(value_error("'inv_trig_style' must be 'abbreviated', 'full' or 'power'"));
        }
        if let Some(order) = &s.order {
            if order != "none" {
                return Err(not_ported(&format!("the LaTeX printing order {:?}", order)));
            }
        }
        let fold_short_frac: bool = s.fold_short_frac.unwrap_or(s.mode == "inline");
        let separator: String = match named_mul_symbol(s.mul_symbol.as_deref()) {
            Some(sep) => sep.to_string(),
            None => {
                let symbol: &str = s.mul_symbol.as_deref().unwrap_or("").trim();
                if symbol.is_empty() {
                    " ".to_string()
                } else {
                    // a symbol of the user stands between spaces (sympy/sympy#30844: 2\starx)
                    format!(" {} ", symbol)
                }
            }
        };
        let spaces: [&str; 7] = ["", " ", "\\", "\\,", "\\:", "\\;", "\\quad"];
        let number_separator: String = match &s.mul_symbol {
            None => r" \cdot ".to_string(),
            Some(m) if spaces.contains(&m.trim()) => r" \cdot ".to_string(),
            Some(_) => separator.clone(),
        };
        let imaginary_unit: String = match s.imaginary_unit.as_str() {
            "i" => "i".to_string(),
            "ri" => r"\mathrm{i}".to_string(),
            "ti" => r"\text{i}".to_string(),
            "j" => "j".to_string(),
            "rj" => r"\mathrm{j}".to_string(),
            "tj" => r"\text{j}".to_string(),
            other => other.to_string(),
        };
        let diff_operator: String = match s.diff_operator.as_str() {
            "d" => "d".to_string(),
            "rd" => r"\mathrm{d}".to_string(),
            "td" => r"\text{d}".to_string(),
            other => other.to_string(),
        };
        let comma: String = if s.decimal_separator == "comma" { ";" } else { "," }.to_string();
        Ok(LatexPrinter {
            settings: s,
            fold_short_frac,
            separator,
            number_separator,
            imaginary_unit,
            diff_operator,
            comma,
        })
    }

    /// The LaTeX of `expr` in the mode of the settings (`doprint`).
    pub fn doprint(&self, expr: &Basic) -> PolyResult<String> {
        let s = &self.settings;
        let top: Ctx = if s.mode == "inline" {
            Context::inline_top()
        } else {
            Context::default()
        };
        let tex: String = self.print(expr, &top)?.output;
        Ok(match s.mode.as_str() {
            "plain" => tex,
            "inline" => format!("${}$", tex),
            _ if s.itex => format!("$${}$$", tex),
            mode => format!(r"\begin{{{}}}{}\end{{{}}}", mode, tex, mode),
        })
    }

    /// A result which was not printed by `print` put in a position (`place`).
    fn place(&self, result: Res, context: &Ctx) -> Res {
        self.enclose(result, context)
    }

    /// `node` printed where its parent delimits it (`inside`).
    pub(crate) fn inside(&self, node: &Basic, context: &Ctx) -> PolyResult<String> {
        Ok(self.print(node, &context.delimited())?.output)
    }

    /// The nodes printed as the elements of a list (`join`).
    pub(crate) fn join(&self, nodes: &[Basic], context: &Ctx, separator: Option<&str>) -> PolyResult<String> {
        let comma_space: String = format!("{} ", self.comma);
        let sep: &str = separator.unwrap_or(&comma_space);
        let parts: Vec<String> = nodes
            .iter()
            .map(|n| self.inside(n, context))
            .collect::<PolyResult<_>>()?;
        Ok(parts.join(sep))
    }

    /// The factors side by side, a number separated from the number before
    /// it; a single factor is printed in the context `sole` (`product`).
    fn product(&self, factors: &[Basic], context: &Ctx, sole: &Ctx) -> R {
        if factors.is_empty() {
            return Ok(PrintResult::new("1".to_string(), Binding::Atom));
        }
        if factors.len() == 1 {
            return self.print(&factors[0], sole);
        }
        let mut tex: String = String::new();
        let mut last: String = String::new();
        let mut applies: bool = false;
        let mut open_ended: bool = false;
        for (i, f) in factors.iter().enumerate() {
            let r: Res = self.print(f, &context.factor(i < factors.len() - 1))?;
            if !tex.is_empty() {
                if ends_with_number(&last) && starts_with_number(&r.output) {
                    tex.push_str(&self.number_separator);
                } else if r.output.starts_with(|c: char| c.is_ascii_digit()) {
                    tex.push_str(&self.number_separator); // x \cdot 3, never x 3
                } else {
                    tex.push_str(&self.separator);
                }
            }
            tex.push_str(&r.output);
            last = r.output.clone();
            applies = applies || r.applies;
            open_ended = r.open_ended;
        }
        let mut result: Res = PrintResult::new(tex, Binding::Product);
        result.open_ended = open_ended;
        result.applies = applies;
        Ok(result)
    }

    /// A function printed as `head` and `tail`: the exponent its parent
    /// offers goes on the head unless the head has a superscript (`attach`).
    fn attach(&self, head: &str, tail: &str, context: &Ctx, scripted_head: bool) -> Res {
        if let Some(exponent) = &context.exponent_offered {
            if !scripted_head {
                let mut r: Res = PrintResult::new(format!("{}^{{{}}}{}", head, exponent, tail), Binding::Power);
                r.absorbed = true;
                r.applies = true;
                return r;
            }
        }
        let mut r: Res = PrintResult::new(format!("{}{}", head, tail), Binding::Application);
        r.applies = true;
        r
    }

    /// `head` applied to the arguments as SymPy prints a function (`apply`).
    fn apply(&self, head: &str, args: &[Basic], context: &Ctx, foldable: bool, scripted_head: bool) -> R {
        let mut head: String = head.to_string();
        let absorbed: bool = context.exponent_offered.is_some();
        if let Some(exponent) = &context.exponent_offered {
            // the head is the base of the exponent: bracketed, where all brackets are, when it has a script
            let mut base: Res = PrintResult::new(head.clone(), Binding::Atom);
            base.scripted = scripted_head || head.contains('^');
            let base: Res = self.place(base, &context.base(None));
            head = format!("{}^{{{}}}", base.output, exponent);
        }
        let mut body: String = format!(r"{{\left({} \right)}}", self.join(args, context, Some(&self.comma))?);
        if self.settings.fold_func_brackets && foldable && args.len() == 1 {
            let bare: Ctx = context.bare_argument();
            let raw: Res = self.rule(&args[0], &bare)?;
            if !needs_brackets(&raw, &bare) {
                body = format!(" {{{}}}", raw.output);
            }
        }
        let mut r: Res = PrintResult::new(
            format!("{}{}", head, body),
            if absorbed { Binding::Power } else { Binding::Application },
        );
        r.absorbed = absorbed;
        r.applies = true;
        Ok(r)
    }

    /// The head of an undefined function and whether it has a superscript
    /// (`undefined_head`).
    fn undefined_head(&self, name: &str) -> (String, bool) {
        if name.contains('{') {
            let escaped: String = escape_name(name);
            let sup: bool = escaped.contains('^');
            return (escaped, sup);
        }
        let (base, supers, subs) = super::conventions::split_super_sub(name);
        let translated: String = translate(&base);
        let mut head: String = if base.chars().count() == 1 || translated != base {
            escape_name(&translated)
        } else {
            format!(r"\operatorname{{{}}}", escape_text(&base))
        };
        if !supers.is_empty() {
            let parts: Vec<String> = supers.iter().map(|s| escape_name(&translate(s))).collect();
            head.push_str(&format!("^{{{}}}", parts.join(" ")));
        }
        if !subs.is_empty() {
            let parts: Vec<String> = subs.iter().map(|s| escape_name(&translate(s))).collect();
            head.push_str(&format!("_{{{}}}", parts.join(" ")));
        }
        (head, !supers.is_empty())
    }

    /// The terms of a sum in the order of the settings (`ordered`).
    fn ordered(&self, terms: &[Basic]) -> Vec<Basic> {
        if self.settings.order.as_deref() == Some("none") || !crate::core::printing::in_compare_order(terms) {
            return terms.to_vec();
        }
        crate::core::printing::ordered_terms(terms)
    }

    /// The factors of a product in the printing order (`printed_factors`).
    fn ordered_factors(&self, factors: &[Basic]) -> Vec<Basic> {
        if self.settings.order.as_deref() == Some("none") {
            return factors.to_vec();
        }
        crate::core::printing::printing_order(factors)
    }

    /// The separator of the elements (`comma`).
    pub(crate) fn comma(&self) -> &str {
        &self.comma
    }

    /// The differential operator (`diff_operator`).
    pub(crate) fn diff_operator(&self) -> &str {
        &self.diff_operator
    }

    /// The elements as a tuple (`_tuple`).
    pub(crate) fn tuple_of(&self, elements: &[Basic], ctx: &Ctx) -> PolyResult<String> {
        Ok(tuple(self, elements, ctx)?.output)
    }

    /// `\left(a, b\right)` (`_round`).
    fn round(&self, args: &[Basic], ctx: &Ctx) -> PolyResult<String> {
        Ok(format!(r"\left({}\right)", self.join(args, ctx, None)?))
    }
}

impl TreePrinter<Basic, String> for LatexPrinter {
    fn rule(&self, node: &Basic, context: &Ctx) -> R {
        rule(self, node, context)
    }

    fn bracket(&self, result: &Res, context: &Ctx) -> String {
        let scripts_only: bool = result.scripted && context.scripts_follow && result.binding >= context.level;
        if scripts_only && !self.settings.parenthesize_super {
            return format!("{{{}}}", result.output);
        }
        format!(r"\left({}\right)", result.output)
    }
}

/// `[0-9][} ]*$`.
fn ends_with_number(s: &str) -> bool {
    let trimmed: &str = s.trim_end_matches(['}', ' ']);
    trimmed.ends_with(|c: char| c.is_ascii_digit())
}

/// `(\d|\\frac{\d+}{\d+})` at the start.
fn starts_with_number(s: &str) -> bool {
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        return true;
    }
    let Some(rest) = s.strip_prefix(r"\frac{") else {
        return false;
    };
    let digits: usize = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 || !rest[digits..].starts_with("}{") {
        return false;
    }
    let rest: &str = &rest[digits + 2..];
    let digits: usize = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    digits > 0 && rest[digits..].starts_with('}')
}

fn atom(s: &str) -> Res {
    PrintResult::new(s.to_string(), Binding::Atom)
}

/// A notation which delimits itself (`_delimited`).
fn delimited(tex: String) -> Res {
    PrintResult::new(tex, Binding::Atom)
}

/// A notation which takes no exponent on its head (`_applied`).
fn applied(tex: String) -> Res {
    let mut r: Res = PrintResult::new(tex, Binding::Application);
    r.applies = true;
    r
}

fn signed(sign: bool, r: Res) -> Res {
    if !sign {
        return r;
    }
    let mut s: Res = PrintResult::new(format!("- {}", r.output), Binding::Negation);
    s.unsigned = Some(r.output);
    s.open_ended = r.open_ended;
    s.applies = r.applies;
    s
}

fn rational_parts(e: &Basic) -> Option<(crate::numbers::Int, crate::numbers::Int)> {
    e.as_rational()
        .map(|v| (crate::numbers::numer(v), crate::numbers::denom(v)))
}

fn is_one(e: &Basic) -> bool {
    *e == Basic::one()
}

/// Python's `isinstance(e, Number)`: a rational, a float, the
/// infinities, `nan`.
fn is_number_node(e: &Basic) -> bool {
    e.is_number_class()
}

/// A negative constant: what SymPy's `fraction(exact=True)` moves to the
/// denominator when it is an exponent (`_negative_number`).
fn negative_number(e: &Basic) -> bool {
    if let Some((p, _)) = rational_parts(e) {
        return p < crate::numbers::Int::ZERO;
    }
    crate::core::assumptions::is_number(e) && is(e, Predicate::negative) == Some(true) && e.free_symbols().is_empty()
}

fn commutative_maybe(e: &Basic) -> bool {
    is(e, Predicate::commutative) != Some(false)
}

/// The sign, the factors of the numerator and those of the denominator of
/// a product in the printing order (`_split_fraction`).
fn split_fraction(factors: &[Basic]) -> PolyResult<(bool, Vec<Basic>, Vec<Basic>)> {
    use crate::numbers::Int;
    let mut negative: bool = false;
    let mut numerator: Vec<Basic> = Vec::new();
    let mut denominator: Vec<Basic> = Vec::new();
    for (i, f) in factors.iter().enumerate() {
        if i == 0 {
            if let Some((p, q)) = rational_parts(f) {
                negative = p < Int::ZERO;
                let magnitude: Int = if negative { -p.clone() } else { p.clone() };
                if magnitude != Int::ONE || (q == Int::ONE && factors.len() == 1) {
                    numerator.push(Basic::integer(magnitude));
                }
                if q != Int::ONE {
                    denominator.push(Basic::integer(q));
                }
                continue;
            }
            if let Head::Float { value, prec } = f.head() {
                if crate::numbers::rat_is_negative(value) {
                    negative = true;
                    numerator.push(Basic::float(&-value.clone(), *prec));
                    continue;
                }
            }
        }
        if f.is_pow() && commutative_maybe(&f.args()[0]) && negative_number(&f.args()[1]) {
            let exponent: Basic = crate::core::mul::neg(&f.args()[1])?;
            denominator.push(if is_one(&exponent) {
                f.args()[0].clone()
            } else {
                crate::core::power::pow(&f.args()[0], &exponent)?
            });
        } else if f.is_function(Function::Exp) && negative_number(&f.args()[0]) {
            denominator.push(crate::functions::elementary::exponential::exp(&crate::core::mul::neg(
                &f.args()[0],
            )?)?);
        } else {
            numerator.push(f.clone());
        }
    }
    Ok((negative, numerator, denominator))
}

fn rel_sign(op: RelOp) -> &'static str {
    match op {
        RelOp::Eq => "=",
        RelOp::Gt => ">",
        RelOp::Lt => "<",
        RelOp::Ge => r"\geq",
        RelOp::Le => r"\leq",
        RelOp::Ne => r"\neq",
    }
}

fn sorted_args(e: &Basic) -> Vec<Basic> {
    let mut args: Vec<Basic> = e.args().to_vec();
    args.sort_by_cached_key(crate::core::sorting::sort_key);
    args
}

/// The rule of a node (`RULES`).
fn rule(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    match e.head() {
        Head::Rational { .. } => rational(p, e, ctx),
        Head::Float { value, prec } => float(p, value, *prec),
        Head::Infinity => Ok(atom(r"\infty")),
        Head::NegativeInfinity => {
            let mut r: Res = PrintResult::new(r"-\infty".to_string(), Binding::Negation);
            r.unsigned = Some(r"\infty".to_string());
            Ok(r)
        }
        Head::ComplexInfinity => Ok(atom(r"\tilde{\infty}")),
        Head::NaN => Ok(atom(r"\text{NaN}")),
        Head::Pi => Ok(atom(r"\pi")),
        Head::Exp1 => Ok(atom("e")),
        Head::GoldenRatio => Ok(atom(r"\phi")),
        Head::EulerGamma => Ok(atom(r"\gamma")),
        Head::Catalan => Ok(atom("G")),
        Head::TribonacciConstant => Ok(atom(r"\text{TribonacciConstant}")),
        Head::ImaginaryUnit => Ok(atom(&p.imaginary_unit)),
        Head::Symbol(d) | Head::Dummy(d, _) => Ok(symbol(&d.name)),
        Head::Wild(w) => Ok(symbol(&w.symbol.name)),
        Head::BooleanTrue => Ok(atom(r"\text{True}")),
        Head::BooleanFalse => Ok(atom(r"\text{False}")),
        Head::Add => add_rule(p, e, ctx),
        Head::Mul => mul_rule(p, e, ctx),
        Head::Pow => pow_rule(p, e, ctx),
        Head::Relational(op) => {
            let mut sign: &str = rel_sign(*op);
            if p.settings.itex && (sign == "<" || sign == ">") {
                sign = if sign == ">" { r"\gt" } else { r"\lt" };
            }
            // a relation or a connective on a side is bracketed (SymPy: x = y = z)
            let lhs: Res = p.print(&e.args()[0], &ctx.side())?;
            let rhs: Res = p.print(&e.args()[1], &ctx.side())?;
            let mut r: Res = PrintResult::new(format!("{} {} {}", lhs.output, sign, rhs.output), Binding::Relation);
            r.open_ended = rhs.open_ended;
            Ok(r)
        }
        Head::And => connective(p, &sorted_args(e), r"\wedge", Binding::And, ctx, false),
        Head::Or => connective(p, &sorted_args(e), r"\vee", Binding::Or, ctx, false),
        Head::Xor => connective(p, &sorted_args(e), r"\veebar", Binding::Implication, ctx, false),
        Head::Implies => {
            if ctx.negation_offered {
                return connective(p, e.args(), r"\not\Rightarrow", Binding::Implication, ctx, true);
            }
            connective(p, e.args(), r"\Rightarrow", Binding::Implication, ctx, false)
        }
        Head::Equivalent => {
            let operands: Vec<Basic> = sorted_args(e);
            if ctx.negation_offered {
                return connective(p, &operands, r"\not\Leftrightarrow", Binding::Implication, ctx, true);
            }
            connective(p, &operands, r"\Leftrightarrow", Binding::Implication, ctx, false)
        }
        Head::Not => {
            // the operand is bracketed by what it printed (SymPy: \neg x > y)
            let operand: Res = p.print(&e.args()[0], &ctx.negated())?;
            if operand.absorbed {
                return Ok(PrintResult::new(operand.output, operand.binding));
            }
            Ok(PrintResult::new(format!(r"\neg {}", operand.output), Binding::Not))
        }
        Head::Ite => Ok(applied(format!(
            r"\operatorname{{ITE}}{{\left({} \right)}}",
            p.join(e.args(), ctx, None)?
        ))),
        Head::Undefined(u) => {
            let (head, scripted) = p.undefined_head(&u.name);
            // never without brackets: f x reads as a product
            p.apply(&head, e.args(), ctx, false, scripted)
        }
        Head::Function(f) => function_rule(p, *f, e, ctx),
        Head::Derivative => derivative(p, e, ctx),
        Head::Subs => {
            let below: Ctx = ctx.subscript();
            let (variables, points) = (e.args()[1].args(), e.args()[2].args());
            let mut pairs: Vec<String> = Vec::new();
            for (v, n) in variables.iter().zip(points) {
                pairs.push(format!("{}={}", p.print(v, &below)?.output, p.print(n, &below)?.output));
            }
            let tex: String = format!(
                r"\left. {} \right|_{{\substack{{ {} }}}}",
                p.inside(&e.args()[0], ctx)?,
                pairs.join(r"\\ ")
            );
            // ends with a subscript: bracketed as a base
            let mut r: Res = PrintResult::new(tex, Binding::Atom);
            r.scripted = true;
            Ok(r)
        }
        Head::Tuple => tuple(p, e.args(), ctx),
        Head::Hyper => {
            use crate::functions::special::hyper::{hyper_ap, hyper_bq};
            let (ap, bq) = (hyper_ap(e), hyper_bq(e));
            Ok(delimited(format!(
                r"{{{{}}_{{{}}}F_{{{}}}\left(\begin{{matrix}} {} \\ {} \end{{matrix}}\middle| {{{}}} \right)}}",
                ap.len(),
                bq.len(),
                p.join(ap, ctx, None)?,
                p.join(bq, ctx, None)?,
                p.inside(&e.args()[2], ctx)?
            )))
        }
        _ => match super::latex_objects::object_rule(p, e, ctx) {
            Some(r) => r,
            None => Err(not_ported(&format!("the LaTeX of {}", e.class_name()))),
        },
    }
}

fn symbol(name: &str) -> Res {
    let (tex, scripted) = name_latex(name);
    let mut r: Res = PrintResult::new(tex, Binding::Atom);
    r.scripted = scripted;
    r
}

fn rational(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    use crate::numbers::Int;
    let (num, den) = rational_parts(e).expect("a rational");
    let negative: bool = num < Int::ZERO;
    let magnitude_num: Int = if negative { -num.clone() } else { num.clone() };
    if den == Int::ONE {
        if negative {
            let mut r: Res = PrintResult::new(num.to_string(), Binding::Negation);
            r.unsigned = Some(magnitude_num.to_string());
            return Ok(r);
        }
        return Ok(atom(&num.to_string()));
    }
    if ctx.position == Position::Exponent && p.settings.fold_frac_powers {
        return Ok(PrintResult::new(format!("{}/{}", num, den), Binding::Quotient));
    }
    let (magnitude, binding) = if p.fold_short_frac {
        (format!("{} / {}", magnitude_num, den), Binding::Quotient)
    } else {
        (format!(r"\frac{{{}}}{{{}}}", magnitude_num, den), Binding::Product)
    };
    if negative {
        let mut r: Res = PrintResult::new(format!("- {}", magnitude), Binding::Negation);
        r.unsigned = Some(magnitude);
        return Ok(r);
    }
    Ok(PrintResult::new(magnitude, binding))
}

// PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Float
fn float(p: &LatexPrinter, value: &crate::numbers::Rat, prec: u32) -> R {
    let s: &LatexSettings = &p.settings;
    let mut digits: String =
        crate::core::float::to_str(value, crate::core::float::prec_to_dps(prec), !s.full_prec, s.min, s.max)?;
    let negative: bool = digits.starts_with('-');
    if negative {
        digits.remove(0);
    }
    let mut binding: Binding = Binding::Atom;
    if let Some((mantissa, exponent)) = digits.clone().split_once('e') {
        let exponent: &str = exponent.strip_prefix('+').unwrap_or(exponent);
        let mantissa: String = if s.decimal_separator == "comma" {
            mantissa.replace('.', "{,}")
        } else {
            mantissa.to_string()
        };
        // a product as printed: bracketed as the base of a power (SymPy: {2.5 \cdot 10^{20}}^{2})
        digits = format!("{}{}10^{{{}}}", mantissa, p.number_separator, exponent);
        binding = Binding::Product;
    } else if s.decimal_separator == "comma" {
        digits = digits.replace('.', "{,}");
    }
    if negative {
        let mut r: Res = PrintResult::new(format!("-{}", digits), Binding::Negation);
        r.unsigned = Some(digits);
        return Ok(r);
    }
    Ok(PrintResult::new(digits, binding))
}

fn add_rule(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let terms: Vec<Basic> = p.ordered(e.args());
    if terms.len() == 1 {
        return p.rule(&terms[0], &ctx.same());
    }
    let mut tex: String = String::new();
    let mut open_ended: bool = false;
    let mut applies: bool = false;
    for (i, term) in terms.iter().enumerate() {
        let r: Res = p.print(term, &ctx.term())?;
        if i == 0 {
            tex = r.output.clone();
        } else if let Some(unsigned) = &r.unsigned {
            tex.push_str(" - ");
            tex.push_str(unsigned);
        } else {
            tex.push_str(" + ");
            tex.push_str(&r.output);
        }
        open_ended = r.open_ended;
        applies = applies || r.applies;
    }
    let mut r: Res = PrintResult::new(tex, Binding::Sum);
    r.open_ended = open_ended;
    r.applies = applies;
    Ok(r)
}

fn mul_rule(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let args: &[Basic] = e.args();
    if args.is_empty() {
        return Ok(atom("1"));
    }
    if is_one(&args[0]) || args[1..].iter().any(is_number_node) {
        // an unevaluated product: the factors as they are
        return p.product(args, ctx, &ctx.factor(false));
    }
    let factors: Vec<Basic> = p.ordered_factors(args);
    let (negative, numerator, denominator) = split_fraction(&factors)?;
    if denominator.is_empty() {
        let placed: Res = p.place(p.product(&numerator, ctx, &ctx.factor(false))?, &ctx.factor(false));
        return Ok(signed(negative, placed));
    }
    Ok(signed(negative, fraction(p, &numerator, &denominator, ctx)?))
}

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

/// `numerator/denominator`: built up, on one line when short with
/// `fold_short_frac`, a part of a long numerator after it (`_fraction`).
fn fraction(p: &LatexPrinter, numerator: &[Basic], denominator: &[Basic], ctx: &Ctx) -> R {
    let built_den: Res = p.product(denominator, ctx, &ctx.denominator(false))?;
    let sdenom: String = built_den.output;
    let ldenom: usize = words(&sdenom);
    let built_num: Res = p.product(numerator, ctx, &ctx.numerator(false))?;
    let snumer: String = built_num.output;
    if p.fold_short_frac && ldenom <= 2 && !sdenom.contains('^') {
        let num: Res = p.place(p.product(numerator, ctx, &ctx.numerator(true))?, &ctx.numerator(true));
        let den: Res = p.place(
            p.product(denominator, ctx, &ctx.denominator(true))?,
            &ctx.denominator(true),
        );
        let mut r: Res = PrintResult::new(format!("{} / {}", num.output, den.output), Binding::Quotient);
        r.applies = num.applies || den.applies;
        return Ok(r);
    }
    if let Some(ratio) = p.settings.long_frac_ratio {
        if (words(&snumer) as f64) > ratio * (ldenom as f64) {
            if numerator.len() <= 1 {
                let after: Res = p.product(numerator, ctx, &ctx.factor(false))?;
                let mut r: Res = PrintResult::new(
                    format!(r"\frac{{1}}{{{}}}{}{}", sdenom, p.separator, after.output),
                    Binding::Product,
                );
                r.open_ended = after.open_ended;
                r.applies = true;
                return Ok(r);
            }
            let mut kept: Vec<Basic> = Vec::new();
            let mut moved: Vec<Basic> = Vec::new();
            for f in numerator {
                let followed: Ctx = ctx.factor(true);
                let mut trial: Vec<Basic> = kept.clone();
                trial.push(f.clone());
                let long: bool =
                    (words(&p.product(&trial, ctx, &ctx.numerator(false))?.output) as f64) > ratio * (ldenom as f64);
                let noncommutative: bool = moved
                    .last()
                    .is_some_and(|m| is(m, Predicate::commutative) == Some(false))
                    && is(f, Predicate::commutative) == Some(false);
                if needs_brackets(&p.rule(f, &followed)?, &followed) || long || noncommutative {
                    moved.push(f.clone());
                } else {
                    kept.push(f.clone());
                }
            }
            if !moved.is_empty() {
                let after: Res = p.place(p.product(&moved, ctx, &ctx.factor(false))?, &ctx.factor(false));
                let head: String = p.product(&kept, ctx, &ctx.numerator(false))?.output;
                let mut r: Res = PrintResult::new(
                    format!(r"\frac{{{}}}{{{}}}{}{}", head, sdenom, p.separator, after.output),
                    Binding::Product,
                );
                r.open_ended = after.open_ended;
                r.applies = true;
                return Ok(r);
            }
        }
    }
    Ok(PrintResult::new(
        format!(r"\frac{{{}}}{{{}}}", snumer, sdenom),
        Binding::Product,
    ))
}

fn pow_rule(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    use crate::numbers::Int;
    let (base, exp) = (&e.args()[0], &e.args()[1]);
    let s = &p.settings;
    if let Some((num, den)) = rational_parts(exp) {
        let abs_one: bool = num == Int::ONE || num == -Int::ONE;
        if abs_one && den != Int::ONE && s.root_notation {
            let radicand: String = p.inside(base, ctx)?;
            let tex: String = if den == Int::from(2) {
                format!(r"\sqrt{{{}}}", radicand)
            } else if s.itex {
                format!(r"\root{{{}}}{{{}}}", den, radicand)
            } else {
                format!(r"\sqrt[{}]{{{}}}", den, radicand)
            };
            if num < Int::ZERO {
                return Ok(reciprocal(p, PrintResult::new(tex, Binding::Power), ctx));
            }
            return Ok(PrintResult::new(tex, Binding::Power));
        }
        if num < Int::ZERO && commutative_maybe(base) && !is_one(base) && !(s.fold_frac_powers && den != Int::ONE) {
            let positive: Basic = crate::core::mul::neg(exp)?;
            let inverse_node: Basic = if is_one(&positive) {
                base.clone()
            } else {
                crate::core::power::pow(base, &positive)?
            };
            let inverse: Res = p.rule(&inverse_node, &ctx.denominator(false))?;
            return Ok(reciprocal(p, inverse, ctx));
        }
    }
    let exponent: String = p.print(exp, &ctx.exponent())?.output;
    let printed: Res = p.print(base, &ctx.base(Some(exponent.clone())))?;
    if printed.absorbed {
        // the base rendered the exponent: what it printed is this power
        let mut r: Res = printed;
        r.absorbed = false;
        return Ok(r);
    }
    let mut r: Res = PrintResult::new(format!("{}^{{{}}}", printed.output, exponent), Binding::Power);
    r.applies = printed.applies;
    Ok(r)
}

/// `1/denominator` for a denominator printed without brackets (`_reciprocal`).
fn reciprocal(p: &LatexPrinter, denominator: Res, ctx: &Ctx) -> Res {
    if p.fold_short_frac && words(&denominator.output) <= 2 && !denominator.output.contains('^') {
        let placed: Res = p.place(denominator, &ctx.denominator(true));
        let mut r: Res = PrintResult::new(format!("1 / {}", placed.output), Binding::Quotient);
        r.applies = placed.applies;
        return r;
    }
    PrintResult::new(format!(r"\frac{{1}}{{{}}}", denominator.output), Binding::Product)
}

fn connective(p: &LatexPrinter, operands: &[Basic], sign: &str, binding: Binding, ctx: &Ctx, absorbed: bool) -> R {
    let mut parts: Vec<String> = Vec::new();
    for a in operands {
        parts.push(p.print(a, &ctx.connective())?.output);
    }
    let mut r: Res = PrintResult::new(parts.join(&format!(" {} ", sign)), binding);
    r.absorbed = absorbed;
    Ok(r)
}

fn tuple(p: &LatexPrinter, elements: &[Basic], ctx: &Ctx) -> R {
    if elements.len() == 1 {
        return Ok(atom(&format!(
            r"\left( {}{}\right)",
            p.inside(&elements[0], ctx)?,
            p.comma
        )));
    }
    let sep: String = format!(r"{} \  ", p.comma);
    Ok(atom(&format!(r"\left( {}\right)", p.join(elements, ctx, Some(&sep))?)))
}

/// SymPy's `requires_partial`.
fn requires_partial(e: &Basic) -> bool {
    if matches!(e.head(), Head::Derivative) {
        return requires_partial(&e.args()[0]);
    }
    e.free_symbols()
        .iter()
        .filter(|s| is(s, Predicate::integer) != Some(true))
        .count()
        > 1
}

fn derivative(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let expr: &Basic = &e.args()[0];
    let d: String = if requires_partial(expr) {
        r"\partial".to_string()
    } else {
        p.diff_operator.clone()
    };
    let mut tex: String = String::new();
    // the orders are integers or symbolic; their sum is the order of the operator
    let mut total: Basic = Basic::zero();
    for pair in e.args()[1..].iter().rev() {
        let (x, count) = (&pair.args()[0], &pair.args()[1]);
        total = crate::core::add::add(&[total, count.clone()])?;
        if is_one(count) {
            tex.push_str(&format!("{} {}", d, p.inside(x, ctx)?));
        } else {
            tex.push_str(&format!(
                "{} {}^{{{}}}",
                d,
                p.print(x, &ctx.base(None))?.output,
                p.inside(count, ctx)?
            ));
        }
    }
    let tex: String = if is_one(&total) {
        format!(r"\frac{{{}}}{{{}}}", d, tex)
    } else {
        format!(r"\frac{{{}^{{{}}}}}{{{}}}", d, p.inside(&total, ctx)?, tex)
    };
    // the body is a product or tighter, the derivative bracketed when a factor follows it
    let body: Res = p.print(expr, &ctx.body())?;
    let mut r: Res = PrintResult::new(format!("{} {}", tex, body.output), Binding::Product);
    r.open_ended = true;
    r.applies = body.applies;
    Ok(r)
}

// -- the functions (`_latex_functions.py`)

/// The heads of the functions printed as `head\left(arguments\right)`
/// (`_LETTERS`).
fn letter_head(f: Function) -> Option<&'static str> {
    use Function as F;
    Some(match f {
        F::Gamma => r"\Gamma",
        F::LowerGamma => r"\gamma",
        F::UpperGamma => r"\Gamma",
        F::Beta => r"\operatorname{B}",
        F::Zeta => r"\zeta",
        F::DirichletEta => r"\eta",
        F::LerchPhi => r"\Phi",
        F::Chi => r"\operatorname{Chi}",
        F::FresnelS => "S",
        F::FresnelC => "C",
        F::AiryAi => "Ai",
        F::AiryBi => "Bi",
        F::Mobius => r"\mu",
        F::MathieuC => "C",
        F::MathieuS => "S",
        _ => return None,
    })
}

/// The heads with a superscript of their own (`_MARKED`).
fn marked_head(f: Function) -> Option<&'static str> {
    use Function as F;
    Some(match f {
        F::AiryAiPrime => r"Ai^\prime",
        F::AiryBiPrime => r"Bi^\prime",
        F::MathieuCPrime => r"C^{\prime}",
        F::MathieuSPrime => r"S^{\prime}",
        _ => return None,
    })
}

/// The functions which SymPy brackets whole under an exponent (`_WHOLE`).
fn whole_head(f: Function) -> Option<&'static str> {
    use Function as F;
    Some(match f {
        F::Totient => r"\phi",
        F::ReducedTotient => r"\lambda",
        F::PrimeNu => r"\nu",
        F::PrimeOmega => r"\Omega",
        _ => return None,
    })
}

/// The Bessel heads and whether they carry a superscript (`_BESSEL`).
fn bessel_head(f: Function) -> Option<(&'static str, bool)> {
    use Function as F;
    Some(match f {
        F::BesselJ => ("J", false),
        F::BesselI => ("I", false),
        F::BesselK => ("K", false),
        F::BesselY => ("Y", false),
        F::SphericalJ => ("j", false),
        F::SphericalY => ("y", false),
        F::Hankel1 => ("H^{(1)}", true),
        F::Hankel2 => ("H^{(2)}", true),
        F::SphericalHankel1 => ("h^{(1)}", true),
        F::SphericalHankel2 => ("h^{(2)}", true),
        _ => return None,
    })
}

/// The letter of an orthogonal polynomial and the number of its
/// parameters in the superscript (`_POLYNOMIALS`).
fn polynomial_head(f: Function) -> Option<(&'static str, usize)> {
    use Function as F;
    Some(match f {
        F::Jacobi => ("P", 2),
        F::Gegenbauer => ("C", 1),
        F::ChebyshevT => ("T", 0),
        F::ChebyshevU => ("U", 0),
        F::Legendre => ("P", 0),
        F::AssocLegendre => ("P", 1),
        F::Hermite => ("H", 0),
        F::Laguerre => ("L", 0),
        F::AssocLaguerre => ("L", 1),
        _ => return None,
    })
}

/// The letters of the number sequences (`_NUMBERS`).
fn number_head(f: Function) -> Option<&'static str> {
    use Function as F;
    Some(match f {
        F::Bernoulli => "B",
        F::Genocchi => "G",
        F::Bell => "B",
        F::Fibonacci => "F",
        F::Lucas => "L",
        F::Tribonacci => "T",
        F::Euler => "E",
        F::Catalan => "C",
        _ => return None,
    })
}

fn is_inverse_trig(f: Function) -> bool {
    use Function as F;
    matches!(
        f,
        F::Asin
            | F::Acos
            | F::Atan
            | F::Acsc
            | F::Asec
            | F::Acot
            | F::Asinh
            | F::Acosh
            | F::Atanh
            | F::Acsch
            | F::Asech
            | F::Acoth
    )
}

fn subscript_of(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> PolyResult<String> {
    Ok(p.print(e, &ctx.subscript())?.output)
}

fn superscript_of(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> PolyResult<String> {
    Ok(p.print(e, &ctx.superscript())?.output)
}

/// `\left(a; b\middle| m\right)` (`_bar`).
fn bar(p: &LatexPrinter, before: &[Basic], after: &Basic, ctx: &Ctx) -> PolyResult<String> {
    let parts: Vec<String> = before.iter().map(|a| p.inside(a, ctx)).collect::<PolyResult<_>>()?;
    Ok(format!(
        r"\left({}\middle| {}\right)",
        parts.join("; "),
        p.inside(after, ctx)?
    ))
}

fn function_rule(p: &LatexPrinter, f: Function, e: &Basic, ctx: &Ctx) -> R {
    use Function as F;
    let args: &[Basic] = e.args();
    if let Some(head) = letter_head(f) {
        return Ok(p.attach(head, &p.round(args, ctx)?, ctx, false));
    }
    if let Some(head) = marked_head(f) {
        return Ok(p.attach(head, &p.round(args, ctx)?, ctx, true));
    }
    if let Some(head) = whole_head(f) {
        return Ok(applied(format!("{}{}", head, p.round(args, ctx)?)));
    }
    if let Some((head, marked)) = bessel_head(f) {
        let order: String = subscript_of(p, &args[0], ctx)?;
        return Ok(p.attach(
            head,
            &format!("_{{{}}}{}", order, p.round(&args[1..], ctx)?),
            ctx,
            marked,
        ));
    }
    if let Some((letter, parameters)) = polynomial_head(f) {
        let mut tex: String = format!("{}_{{{}}}", letter, subscript_of(p, &args[0], ctx)?);
        if parameters > 0 {
            let above: Vec<String> = args[1..1 + parameters]
                .iter()
                .map(|a| superscript_of(p, a, ctx))
                .collect::<PolyResult<_>>()?;
            tex.push_str(&format!(r"^{{\left({}\right)}}", above.join(",")));
        }
        return Ok(applied(format!("{}{}", tex, p.round(&args[1 + parameters..], ctx)?)));
    }
    if let Some(letter) = number_head(f) {
        let tail: String = format!("_{{{}}}", subscript_of(p, &args[0], ctx)?);
        if args.len() == 1 {
            // an indexed letter: an exponent stands after the index
            return Ok(atom(&format!("{}{}", letter, tail)));
        }
        return Ok(p.attach(letter, &format!("{}{}", tail, p.round(&args[1..], ctx)?), ctx, false));
    }
    if is_inverse_trig(f) {
        let name: &str = f.name();
        let style: &str = &p.settings.inv_trig_style;
        let accepted = |n: &str| ACCEPTED_FUNCTIONS.contains(&n);
        if style == "power" {
            let short: &str = &name[1..];
            let head: String = if accepted(short) {
                format!("\\{}^{{-1}}", short)
            } else {
                format!(r"\operatorname{{{}}}^{{-1}}", short)
            };
            // the head has its superscript: an exponent goes around the whole
            return p.apply(&head, args, &ctx.without_offers(), true, false);
        }
        let full: String = if style == "full" {
            format!("{}{}", if name.ends_with('h') { "ar" } else { "arc" }, &name[1..])
        } else {
            name.to_string()
        };
        let head: String = if accepted(&full) {
            format!("\\{}", full)
        } else {
            format!(r"\operatorname{{{}}}", full)
        };
        return p.apply(&head, args, ctx, true, false);
    }
    match f {
        F::Exp => Ok(PrintResult::new(
            format!("e^{{{}}}", p.print(&args[0], &ctx.exponent())?.output),
            Binding::Power,
        )),
        F::Log => {
            let head: &str = if p.settings.ln_notation { r"\ln" } else { r"\log" };
            Ok(delimited(format!(
                r"{}{{\left({} \right)}}",
                head,
                p.inside(&args[0], ctx)?
            )))
        }
        F::Abs => Ok(delimited(format!(r"\left|{{{}}}\right|", p.inside(&args[0], ctx)?))),
        F::Floor => Ok(delimited(format!(
            r"\left\lfloor{{{}}}\right\rfloor",
            p.inside(&args[0], ctx)?
        ))),
        F::Ceiling => Ok(delimited(format!(
            r"\left\lceil{{{}}}\right\rceil",
            p.inside(&args[0], ctx)?
        ))),
        F::Frac => Ok(delimited(format!(
            r"\operatorname{{frac}}{{\left({}\right)}}",
            p.inside(&args[0], ctx)?
        ))),
        F::Conjugate => Ok(delimited(format!(r"\overline{{{}}}", p.inside(&args[0], ctx)?))),
        F::Re | F::Im => {
            let head: &str = match (f, p.settings.gothic_re_im) {
                (F::Re, true) => r"\Re",
                (F::Re, false) => r"\operatorname{re}",
                (_, true) => r"\Im",
                _ => r"\operatorname{im}",
            };
            Ok(applied(format!(
                r"{}{{\left({}\right)}}",
                head,
                p.inside(&args[0], ctx)?
            )))
        }
        F::Max | F::Min => {
            let sorted: Vec<Basic> = sorted_args(e);
            Ok(delimited(format!(
                r"\{}\left({}\right)",
                f.name().to_lowercase(),
                p.join(&sorted, ctx, None)?
            )))
        }
        F::Factorial => Ok(applied(format!("{}!", p.print(&args[0], &ctx.operand())?.output))),
        F::Factorial2 => Ok(applied(format!("{}!!", p.print(&args[0], &ctx.operand())?.output))),
        F::Subfactorial => Ok(applied(format!("!{}", p.print(&args[0], &ctx.operand())?.output))),
        F::Binomial => Ok(delimited(format!(
            r"{{\binom{{{}}}{{{}}}}}",
            p.inside(&args[0], ctx)?,
            p.inside(&args[1], ctx)?
        ))),
        F::RisingFactorial => {
            let base: String = p.print(&args[0], &ctx.operand())?.output;
            Ok(PrintResult::new(
                format!(r"{{{}}}^{{\left({}\right)}}", base, p.inside(&args[1], ctx)?),
                Binding::Power,
            ))
        }
        F::FallingFactorial => {
            let sub: String = subscript_of(p, &args[1], ctx)?;
            let mut r: Res = PrintResult::new(
                format!(r"{{\left({}\right)}}_{{{}}}", p.inside(&args[0], ctx)?, sub),
                Binding::Atom,
            );
            r.scripted = true;
            Ok(r)
        }
        F::QPochhammer => {
            let tex: String = format!(
                r"\left({}; {}\right)_{{{}}}",
                p.inside(&args[0], ctx)?,
                p.inside(&args[1], ctx)?,
                subscript_of(p, &args[2], ctx)?
            );
            let mut r: Res = PrintResult::new(tex, Binding::Atom);
            r.scripted = true;
            Ok(r)
        }
        F::EllipticK => Ok(p.attach("K", &p.round(args, ctx)?, ctx, false)),
        F::EllipticF => Ok(p.attach("F", &bar(p, &args[..1], &args[1], ctx)?, ctx, false)),
        F::EllipticE => {
            if args.len() == 2 {
                return Ok(p.attach("E", &bar(p, &args[..1], &args[1], ctx)?, ctx, false));
            }
            Ok(p.attach("E", &p.round(args, ctx)?, ctx, false))
        }
        F::EllipticPi => Ok(p.attach(
            r"\Pi",
            &bar(p, &args[..args.len() - 1], &args[args.len() - 1], ctx)?,
            ctx,
            false,
        )),
        F::BetaInc | F::BetaIncRegularized => {
            let letter: &str = if f == F::BetaInc { "B" } else { "I" };
            let bounds: String = format!(
                "({}, {})",
                subscript_of(p, &args[2], ctx)?,
                subscript_of(p, &args[3], ctx)?
            );
            Ok(p.attach(
                &format!(r"\operatorname{{{}}}_{{{}}}", letter, bounds),
                &p.round(&args[..2], ctx)?,
                ctx,
                false,
            ))
        }
        F::Expint => {
            let nu: String = subscript_of(p, &args[0], ctx)?;
            Ok(p.attach(
                &format!(r"\operatorname{{E}}_{{{}}}", nu),
                &p.round(&args[1..], ctx)?,
                ctx,
                false,
            ))
        }
        F::PolyLog => {
            let s: String = subscript_of(p, &args[0], ctx)?;
            Ok(p.attach(
                &format!(r"\operatorname{{Li}}_{{{}}}", s),
                &p.round(&args[1..], ctx)?,
                ctx,
                false,
            ))
        }
        F::Stieltjes => {
            let mut tail: String = format!("_{{{}}}", subscript_of(p, &args[0], ctx)?);
            if args.len() == 2 {
                tail.push_str(&p.round(&args[1..], ctx)?);
            }
            Ok(p.attach(r"\gamma", &tail, ctx, false))
        }
        F::LambertW => {
            let mut tail: String = p.round(&args[..1], ctx)?;
            if args.len() == 2 {
                tail = format!("_{{{}}}{}", subscript_of(p, &args[1], ctx)?, tail);
            }
            Ok(p.attach("W", &tail, ctx, false))
        }
        F::DivisorSigma | F::UDivisorSigma => {
            let mut tail: String = p.round(&args[..1], ctx)?;
            if args.len() == 2 {
                tail = format!("_{{{}}}{}", subscript_of(p, &args[1], ctx)?, tail);
            }
            if f == F::DivisorSigma {
                return Ok(p.attach(r"\sigma", &tail, ctx, false));
            }
            // the head has a superscript: an exponent goes around the whole
            Ok(p.attach(r"\sigma^*", &tail, ctx, true))
        }
        F::Ynm | F::Znm => {
            let letter: &str = if f == F::Ynm { "Y" } else { "Z" };
            let tex: String = format!(
                "{}_{{{}}}^{{{}}}",
                letter,
                subscript_of(p, &args[0], ctx)?,
                superscript_of(p, &args[1], ctx)?
            );
            Ok(applied(format!(
                r"{}\left({},{}\right)",
                tex,
                p.inside(&args[2], ctx)?,
                p.inside(&args[3], ctx)?
            )))
        }
        F::DiracDelta => {
            if args.len() == 1 || is(&args[1], Predicate::zero) == Some(true) {
                return Ok(applied(format!(r"\delta\left({}\right)", p.inside(&args[0], ctx)?)));
            }
            let order: String = superscript_of(p, &args[1], ctx)?;
            Ok(applied(format!(
                r"\delta^{{\left( {} \right)}}\left( {} \right)",
                order,
                p.inside(&args[0], ctx)?
            )))
        }
        F::Heaviside => {
            let pargs: &[Basic] = if args.len() == 2 && args[1] == Basic::half() {
                &args[..1]
            } else {
                args
            };
            Ok(applied(format!(r"\theta\left({}\right)", p.join(pargs, ctx, None)?)))
        }
        F::KroneckerDelta | F::LeviCivita => {
            let below: Ctx = ctx.subscript();
            let operands: &[Basic] = if f == F::KroneckerDelta { &args[..2] } else { args };
            let separator: &str = if operands.iter().all(|a| a.args().is_empty()) {
                " "
            } else {
                ", "
            };
            let parts: Vec<String> = operands
                .iter()
                .map(|a| Ok(p.print(a, &below)?.output))
                .collect::<PolyResult<_>>()?;
            let head: &str = if f == F::KroneckerDelta {
                r"\delta"
            } else {
                r"\varepsilon"
            };
            Ok(atom(&format!("{}_{{{}}}", head, parts.join(separator))))
        }
        F::SingularityFunction => {
            let shift: Basic = crate::core::add::sub(&args[0], &args[1])?;
            let power: String = superscript_of(p, &args[2], ctx)?;
            Ok(PrintResult::new(
                format!(
                    r"{{\left\langle {} \right\rangle}}^{{{}}}",
                    p.inside(&shift, ctx)?,
                    power
                ),
                Binding::Power,
            ))
        }
        F::Mod => {
            // the operands are products or tighter (SymPy: - x \bmod y, x \bmod y \bmod z)
            let left: Res = p.print(&args[0], &ctx.left(Binding::Negation))?;
            let right: Res = p.print(&args[1], &ctx.right(Binding::Negation))?;
            Ok(PrintResult::new(
                format!(r"{} \bmod {}", left.output, right.output),
                Binding::Modulo,
            ))
        }
        // a function without a form of its own
        _ => p.apply(&operator_name(f.name()), args, ctx, true, false),
    }
}

/// The LaTeX form of `expr` with the settings (`latex`).
pub fn latex(expr: &Basic, settings: LatexSettings) -> PolyResult<String> {
    LatexPrinter::new(settings)?.doprint(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_at_the_edges() {
        assert!(ends_with_number("x_{2}"));
        assert!(!ends_with_number("x"));
        assert!(starts_with_number(r"\frac{1}{2} x"));
        assert!(starts_with_number("3"));
        assert!(!starts_with_number(r"\frac{x}{2}"));
    }
}
