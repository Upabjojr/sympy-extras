//! Port of `sympy_extras/printing/pretty/rules_calculus.py`: derivatives,
//! integrals, sums, products, limits, substitutions and order terms.
//! `Lambda` (not a node of the core) is not ported.

// PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_Derivative, _print_Integral, _print_Sum,
// _print_Product, _print_Limit, _print_Subs, _print_Order), sympy/printing/conventions.py
// (requires_partial) (SymPy 1.14, BSD license; through sympy_extras/printing/pretty/rules_calculus.py)

use super::engine::PrettyPrinter;
use super::layout::{Picture, PrettyContext, PrettyResult};
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is;
use crate::error::PolyResult;
use crate::printing::framework::{needs_brackets, Binding, ChildOptions, Position, PrintResult, TreePrinter};

type R = PolyResult<PrettyResult>;

/// The rule of a calculus node, `None` for the other nodes.
pub fn calculus_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> PolyResult<Option<PrettyResult>> {
    Ok(Some(match e.head() {
        Head::Derivative => derivative(p, e, ctx)?,
        Head::Integral => integral(p, e, ctx)?,
        Head::Sum => sum(p, e, ctx)?,
        Head::Product => product(p, e, ctx)?,
        Head::Limit => limit(p, e, ctx)?,
        Head::Subs => subs(p, e, ctx)?,
        Head::Order => order(p, e, ctx)?,
        _ => return Ok(None),
    }))
}

/// Whether a derivative of `expr` is written with the partial sign
/// (`requires_partial`).
pub fn requires_partial(expr: &Basic) -> bool {
    if matches!(expr.head(), Head::Derivative) {
        return requires_partial(&expr.args()[0]);
    }
    expr.free_symbols()
        .iter()
        .filter(|s| is(s, Predicate::integer) != Some(true))
        .count()
        > 1
}

fn body(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext, minimum: Binding) -> R {
    let c: PrettyContext = ctx.at(Position::Body, minimum, ChildOptions::default());
    let r: PrettyResult = p.rule(e, &c)?;
    Ok(p.enclose(r, &c))
}

fn derivative(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let expr: &Basic = &e.args()[0];
    let d: String = if requires_partial(expr) {
        p.symbology.atom("Partial", "d")
    } else {
        "d".to_string()
    };
    let mut below: Vec<Picture> = Vec::new();
    let mut total: Basic = Basic::zero();
    for pair in e.args()[1..].iter().rev() {
        let (symbol, count) = (&pair.args()[0], &pair.args()[1]);
        let mut ds: Picture = p.operand(symbol, &ctx.operand())?.ltext(&d);
        total = crate::core::add::add(&[total, count.clone()])?;
        if *count != Basic::one() {
            ds = p.power(&ds, &p.operand(count, &ctx.delimited())?);
        }
        below.push(ds);
    }
    let denominator: Picture = p.join(&below, " ");
    let mut top: Picture = Picture::of(&d);
    if total != Basic::one() {
        top = p.power(&top, &p.operand(&total, &ctx.delimited())?);
    }
    let width: usize = top.width().max(denominator.width());
    let operator: Picture = top.below(&[&Picture::of(&p.symbology.hobj("-", width)), &denominator]);
    let operator: Picture = operator.with_baseline(operator.baseline + 1);
    // the operand between the parentheses of the notation
    let operand: Picture = p.parens(&p.operand(expr, &ctx.delimited())?, "(", ")", false);
    Ok(PrintResult::new(
        Picture::beside(&[&operator, &operand]),
        Binding::Product,
    ))
}

fn integral(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    // an integral of an integral is a multiple integral: the signs in a
    // row, the innermost variable first after the integrand
    let mut chain: Vec<Basic> = vec![e.clone()];
    while matches!(chain[chain.len() - 1].args()[0].head(), Head::Integral) {
        let inner: Basic = chain[chain.len() - 1].args()[0].clone();
        chain.push(inner);
    }
    chain.reverse();
    // a sum is bracketed; the differential closes the integrand
    let mut arg: Picture = body(p, &chain[0].args()[0], ctx, Binding::Negation)?.output;
    for integral in &chain {
        let (_, variable, _) = crate::integrals::integral::parts(integral);
        let mut variable: Picture = p.operand(variable, &ctx.operand())?;
        if variable.width() > 1 {
            variable = p.parens(&variable, "(", ")", false);
        }
        arg = Picture::beside(&[&arg, &Picture::of(" d"), &variable]);
    }
    let ascii_mode: bool = !p.unicode();
    let mut signs: Option<Picture> = None;
    for integral in &chain {
        let h: usize = arg.height();
        let height: usize = h + 2 + if ascii_mode { 2 } else { 0 };
        let mut sign: Picture = Picture::text(
            &p.symbology.vobj("int", height),
            arg.baseline + ((height - h) / 2) as i64,
        );
        let (_, _, bounds) = crate::integrals::integral::parts(integral);
        if let Some((a, b)) = bounds {
            let mut lower: Picture = p.operand(a, &ctx.delimited())?;
            let mut upper: Picture = p.operand(b, &ctx.delimited())?;
            if ascii_mode {
                // room, so that a bound is seen to belong to its sign
                upper = upper.left(&[&Picture::blank(1usize.max(3usize.saturating_sub(upper.width())), 1)]);
                lower = lower.right(&[&Picture::blank(1usize.max(4usize.saturating_sub(lower.width())), 1)]);
            }
            sign = sign.above(&[&upper]).below(&[&lower]);
        }
        if !ascii_mode {
            sign = sign.rtext(" ");
        }
        signs = Some(match signs {
            None => sign,
            Some(s) => s.left(&[&sign]),
        });
    }
    match signs {
        None => Ok(PrintResult::new(arg, Binding::Product)),
        Some(s) => Ok(PrintResult::new(arg.left(&[&s]), Binding::Product)),
    }
}

/// `variable = lower` and the upper bound (`_limits_pictures`).
fn limits_pictures(p: &PrettyPrinter, limit: &Basic, ctx: &PrettyContext) -> PolyResult<(Picture, Picture)> {
    let c: PrettyContext = ctx.delimited();
    let lower: Picture = Picture::beside(&[
        &p.operand(&limit.args()[0], &c)?,
        &Picture::of(" = "),
        &p.operand(&limit.args()[1], &c)?,
    ]);
    Ok((lower, p.operand(&limit.args()[2], &c)?))
}

/// The limits, innermost first, and the operand of a sum of sums (a
/// product of products) (`_flattened`).
fn flattened(e: &Basic) -> (Vec<Basic>, Basic) {
    let mut limits: Vec<Basic> = e.args()[1..].to_vec();
    let mut function: Basic = e.args()[0].clone();
    while std::mem::discriminant(function.head()) == std::mem::discriminant(e.head()) {
        let mut inner: Vec<Basic> = function.args()[1..].to_vec();
        inner.extend(limits);
        limits = inner;
        function = function.args()[0].clone();
    }
    (limits, function)
}

/// The sign of a sum for an operand of the given height (`_sum_sign`).
fn sum_sign(required: usize, ascii_mode: bool) -> (usize, usize, Vec<String>, usize) {
    let h: usize = required.max(2);
    let mut d: usize = h / 2;
    let mut w: usize = d + 1;
    let more: usize = required % 2;
    let mut lines: Vec<String> = Vec::new();
    let sp = |n: usize| " ".repeat(n);
    if ascii_mode {
        lines.push(format!("{} ", "_".repeat(w)));
        lines.push(format!("\\{}`", sp(w - 1)));
        for i in 1..d {
            lines.push(format!("{}\\{}", sp(i), sp(w - i)));
        }
        if more == 1 {
            lines.push(format!("{}){}", sp(d), sp(w - d)));
        }
        for i in (1..d).rev() {
            lines.push(format!("{}/{}", sp(i), sp(w - i)));
        }
        lines.push(format!("/{},", "_".repeat(w - 1)));
        return (d, h + more, lines, more);
    }
    w += more;
    d += more;
    lines.push("_".repeat(w));
    for i in 0..d {
        lines.push(format!("{}\u{2572}{}", sp(i), sp(w - i - 1)));
    }
    for i in (0..d).rev() {
        lines.push(format!("{}\u{2571}{}", sp(i), sp(w - i - 1)));
    }
    lines.push("\u{203e}".repeat(w));
    (d, h + 2 * more, lines, more)
}

fn sum(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let ascii_mode: bool = !p.unicode();
    let (limits, function) = flattened(e);
    let mut picture: Picture = body(p, &function, ctx, Binding::Negation)?.output;
    let required: usize = picture.height() + 2;
    let mut first: bool = true;
    let mut max_upper: usize = 0;
    let mut sign_height: usize = 0;
    let mut adjustment: usize = 0;
    for limit in &limits {
        let (lower, upper) = limits_pictures(p, limit, ctx)?;
        max_upper = max_upper.max(upper.height());
        let (d, h, lines, more) = sum_sign(required, ascii_mode);
        adjustment = more;
        let mut sign: Picture = Picture::new(lines, 1);
        if first {
            sign_height = sign.height();
        }
        sign = sign.above(&[&upper]).below(&[&lower]);
        if first {
            // the operand is centred on the sign
            picture = picture.with_baseline((picture.height() / 2) as i64 - d as i64);
            first = false;
        }
        let pad: Picture = Picture::new(vec![" ".to_string(); h], 1);
        picture = Picture::beside(&[&sign, &pad, &picture]);
    }
    // an ASCII sign of odd height has its middle on a row
    let ascii_adjustment: usize = if ascii_mode && adjustment == 0 { 1 } else { 0 };
    picture = picture.with_baseline((max_upper + sign_height / 2 + ascii_adjustment) as i64);
    let mut r: PrettyResult = PrintResult::new(picture, Binding::Product);
    r.open_ended = true;
    Ok(r)
}

fn product(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let (limits, function) = flattened(e);
    let mut picture: Picture = body(p, &function, ctx, Binding::Negation)?.output;
    let s = &p.symbology;
    let horizontal: &str = if p.unicode() { s.xobj("-", 1)[0] } else { "_" };
    let corner: &str = if p.unicode() { s.xobj("UpTack", 1)[0] } else { "_" };
    let vertical: &str = s.xobj("|", 1)[0];
    let func_height: usize = picture.height();
    let mut first: bool = true;
    let mut max_upper: usize = 0;
    let mut sign_height: usize = 0;
    for limit in &limits {
        let (lower, upper) = limits_pictures(p, limit, ctx)?;
        let width: usize = (func_height + 2) * 5 / 3 - 2;
        let mut lines: Vec<String> = vec![format!(
            "{}{}{}{}{}",
            horizontal,
            corner,
            horizontal.repeat(width - 2),
            corner,
            horizontal
        )];
        for _ in 0..func_height + 1 {
            lines.push(format!(" {}{}{} ", vertical, " ".repeat(width - 2), vertical));
        }
        let mut sign: Picture = Picture::new(lines, 1);
        max_upper = max_upper.max(upper.height());
        if first {
            sign_height = sign.height();
        }
        sign = sign.above(&[&upper]).below(&[&lower]);
        if first {
            picture = picture.with_baseline(0);
            first = false;
        }
        let padding: Picture = Picture::new(vec![" ".to_string(); sign.height() - 1], 1);
        picture = Picture::beside(&[&sign, &padding, &picture]);
    }
    picture = picture.with_baseline((max_upper + sign_height / 2) as i64);
    let mut r: PrettyResult = PrintResult::new(picture, Binding::Product);
    r.open_ended = true;
    Ok(r)
}

fn limit(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let (expr, variable, point, dir) = (&e.args()[0], &e.args()[1], &e.args()[2], &e.args()[3]);
    // the operand is bracketed unless it is a power, a function or an atom
    let c_body: PrettyContext = ctx.at(Position::Body, Binding::Root, ChildOptions::default());
    let found: PrettyResult = p.rule(expr, &c_body)?;
    let untouched: bool = !needs_brackets(&found, &c_body);
    let closed: PrettyResult = p.enclose(found, &c_body);
    let body: Picture = closed.output;
    let s = &p.symbology;
    let arrow: String = if p.unicode() {
        format!("{}{}", s.xobj("-", 1)[0], s.atom("Arrow", ">"))
    } else {
        "->".to_string()
    };
    let direction: String = dir.to_string();
    let marked: bool = direction != "+-" && !matches!(point.head(), Head::Infinity | Head::NegativeInfinity);
    // the direction mark applies to the whole point
    let c: PrettyContext = ctx.at(
        Position::Point,
        if marked { Binding::Application } else { Binding::Lowest },
        ChildOptions::default(),
    );
    let mut below: Picture = Picture::beside(&[
        &p.operand(variable, &ctx.delimited())?,
        &Picture::of(&arrow),
        &p.operand(point, &c)?,
    ]);
    if marked {
        let mark: String = if p.unicode() {
            s.atom(
                if direction == "+" {
                    "SuperscriptPlus"
                } else {
                    "SuperscriptMinus"
                },
                &direction,
            )
        } else {
            direction.clone()
        };
        below = below.rtext(&mark);
    }
    let mut sign: Picture = Picture::of("lim").below(&[&below]);
    if untouched && body.height() as i64 - body.baseline > 1 {
        // the rows of the operand under its baseline would stand next to the variable
        sign = sign.rtext(" ");
    }
    let mut r: PrettyResult = PrintResult::new(Picture::beside(&[&sign, &body]), Binding::Product);
    r.open_ended = true;
    Ok(r)
}

fn subs(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let mut picture: Picture = p.parens(&p.operand(&e.args()[0], &ctx.delimited())?, "(", ")", false);
    let height: usize = if picture.height() > 1 { picture.height() } else { 2 };
    let bar: Picture = Picture::text(&p.symbology.vobj("|", height), picture.baseline);
    picture = picture.right(&[&bar]);
    let baseline: i64 = picture.baseline;
    let mut assignments: Vec<Picture> = Vec::new();
    for (variable, value) in e.args()[1].args().iter().zip(e.args()[2].args()) {
        assignments.push(Picture::beside(&[
            &p.operand(variable, &ctx.delimited())?,
            &Picture::of("="),
            &p.operand(value, &ctx.delimited())?,
        ]));
    }
    let last: i64 = picture.height() as i64 - 1;
    picture = picture.with_baseline(last).right(&[&p.join(&assignments, ", ")]);
    Ok(PrintResult::new(picture.with_baseline(baseline), Binding::Product))
}

fn order(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let (expr, pairs) = crate::core::order::parts(e);
    let mut picture: Picture = p.operand(&expr, &ctx.delimited())?;
    let variables: Vec<Basic> = pairs.iter().map(|(v, _)| v.clone()).collect();
    let point: Vec<Basic> = pairs.iter().map(|(_, z)| z.clone()).collect();
    let zero_number = |v: &Basic| matches!(v.head(), Head::Rational { .. }) && *v == Basic::zero();
    if point.iter().any(|v| !zero_number(v)) || variables.len() > 1 {
        picture = picture.rtext("; ");
        if variables.len() > 1 {
            picture = picture.right(&[&p.sequence(&variables, ctx, "(", ")", ", ")?.output]);
        } else if let Some(v) = variables.first() {
            picture = picture.right(&[&p.operand(v, &ctx.delimited())?]);
        }
        picture = picture.rtext(&format!(" {} ", p.symbology.atom("Arrow", "->")));
        if point.len() > 1 {
            picture = picture.right(&[&p.sequence(&point, ctx, "(", ")", ", ")?.output]);
        } else {
            picture = picture.right(&[&p.operand(&point[0], &ctx.delimited())?]);
        }
    }
    Ok(PrintResult::new(
        p.parens(&picture, "(", ")", false).ltext("O"),
        Binding::Application,
    ))
}
