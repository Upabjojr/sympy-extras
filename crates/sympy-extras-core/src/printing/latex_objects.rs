//! Port of `sympy_extras/printing/_latex_objects.py` and of the calculus
//! rules of `latex.py` which are not functions: piecewise expressions,
//! sums, products and integrals, sets, quantifiers, limits, order terms,
//! accumulation bounds. Not ported yet: the sequences, the series and the
//! roots of polynomials (not nodes of the core, or not yet mapped).

// PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Piecewise, _print_Sum, _print_Product,
// _print_Integral, _print_ProductSet, _print_Interval, _print_ImageSet, _print_ConditionSet,
// _print_Contains, _print_Limit, _print_Order (SymPy 1.14, BSD license; through
// sympy_extras/printing/_latex_objects.py and latex.py)

use super::framework::{Binding, PrintResult, TreePrinter};
use super::latex::LatexPrinter;
use crate::core::basic::{Basic, Head};
use crate::error::PolyResult;

type Ctx = super::framework::Context<String>;
type Res = PrintResult<String>;
type R = PolyResult<Res>;

fn atom(s: String) -> Res {
    PrintResult::new(s, Binding::Atom)
}

fn sorted(args: &[Basic]) -> Vec<Basic> {
    let mut v: Vec<Basic> = args.to_vec();
    v.sort_by_cached_key(crate::core::sorting::sort_key);
    v
}

/// The rule of a node which is not a function, `None` for the others.
pub(crate) fn object_rule(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> Option<R> {
    Some(match e.head() {
        Head::Piecewise => piecewise(p, e, ctx),
        Head::ExprCondPair => (|| {
            Ok(atom(format!(
                r"\left( {}{} \  {}\right)",
                p.inside(&e.args()[0], ctx)?,
                p.comma(),
                p.inside(&e.args()[1], ctx)?
            )))
        })(),
        Head::Sum => with_limits(p, e, r"\sum", ctx),
        Head::Product => with_limits(p, e, r"\prod", ctx),
        Head::Integral => integral(p, e, ctx),
        Head::NonElementaryIntegral => integral_form(p, r"\int", &e.args()[0], &e.args()[1..2], ctx),
        Head::IntegralByRanges => (|| {
            let tex: String = format!(r"\int\limits_{{{}}}", p.print(&e.args()[1], &ctx.subscript())?.output);
            integral_form(p, &tex, &e.args()[0], e.args()[2].args(), ctx)
        })(),
        Head::Union => operation(p, e.args(), r"\cup", ctx),
        Head::Intersection => operation(p, e.args(), r"\cap", ctx),
        Head::Complement => operation(p, e.args(), r"\setminus", ctx),
        Head::ProductSet => (|| {
            let sets: &[Basic] = e.args();
            if !sets.is_empty() && sets.iter().all(|s| *s == sets[0]) {
                return Ok(PrintResult::new(
                    format!("{}^{{{}}}", p.print(&sets[0], &ctx.base(None))?.output, sets.len()),
                    Binding::Power,
                ));
            }
            operation(p, sets, r"\times", ctx)
        })(),
        Head::EmptySet => Ok(atom(r"\emptyset".to_string())),
        Head::UniversalSet => Ok(atom(r"\mathbb{U}".to_string())),
        Head::Naturals => Ok(atom(r"\mathbb{N}".to_string())),
        Head::Naturals0 => {
            let mut r: Res = atom(r"\mathbb{N}_0".to_string());
            r.scripted = true;
            Ok(r)
        }
        Head::Integers => Ok(atom(r"\mathbb{Z}".to_string())),
        Head::Rationals => Ok(atom(r"\mathbb{Q}".to_string())),
        Head::Reals => Ok(atom(r"\mathbb{R}".to_string())),
        Head::Complexes => Ok(atom(r"\mathbb{C}".to_string())),
        Head::Interval => interval(p, e, ctx),
        Head::FiniteSet | Head::PointSet | Head::SetOfSets => (|| {
            Ok(atom(format!(
                r"\left\{{{}\right\}}",
                p.join(&sorted(e.args()), ctx, None)?
            )))
        })(),
        Head::ImageSet => image_set(p, e, ctx),
        Head::ConditionSet => condition_set(p, e, ctx),
        Head::PowerSet => (|| {
            let mut r: Res = PrintResult::new(
                format!(r"\mathcal{{P}}\left({}\right)", p.inside(&e.args()[0], ctx)?),
                Binding::Application,
            );
            r.applies = true;
            Ok(r)
        })(),
        Head::Contains => (|| {
            Ok(PrintResult::new(
                format!(
                    r"{} \in {}",
                    p.print(&e.args()[0], &ctx.side())?.output,
                    p.print(&e.args()[1], &ctx.side())?.output
                ),
                Binding::Relation,
            ))
        })(),
        Head::ForAll => quantifier(p, e, r"\forall", ctx),
        Head::Exists => quantifier(p, e, r"\exists", ctx),
        Head::Limit => limit(p, e, ctx),
        Head::Order => order(p, e, ctx),
        Head::AccumBounds => (|| {
            Ok(atom(format!(
                r"\left\langle {}{} {}\right\rangle",
                p.inside(&e.args()[0], ctx)?,
                p.comma(),
                p.inside(&e.args()[1], ctx)?
            )))
        })(),
        _ => return None,
    })
}

fn piecewise(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let pairs: &[Basic] = e.args();
    let mut rows: Vec<String> = Vec::new();
    for (i, pair) in pairs.iter().enumerate() {
        let (expr, cond) = (&pair.args()[0], &pair.args()[1]);
        if i == pairs.len() - 1 && matches!(cond.head(), Head::BooleanTrue) {
            rows.push(format!(r"{} & \text{{otherwise}}", p.inside(expr, ctx)?));
        } else {
            rows.push(format!(
                r"{} & \text{{for}}\: {}",
                p.inside(expr, ctx)?,
                p.inside(cond, ctx)?
            ));
        }
    }
    // open on the right: bracketed wherever it is not delimited
    Ok(PrintResult::new(
        format!(r"\begin{{cases}} {} \end{{cases}}", rows.join(r" \\")),
        Binding::Block,
    ))
}

fn with_limits(p: &LatexPrinter, e: &Basic, sign: &str, ctx: &Ctx) -> R {
    let below: Ctx = ctx.subscript();
    let limits: &[Basic] = &e.args()[1..];
    let tex: String = if limits.len() == 1 {
        let (v, a, b) = (&limits[0].args()[0], &limits[0].args()[1], &limits[0].args()[2]);
        format!(
            "{}_{{{}={}}}^{{{}}} ",
            sign,
            p.print(v, &below)?.output,
            p.print(a, &below)?.output,
            p.print(b, &ctx.superscript())?.output
        )
    } else {
        let mut rows: Vec<String> = Vec::new();
        for l in limits {
            rows.push(format!(
                r"{} \leq {} \leq {}",
                p.print(&l.args()[1], &below)?.output,
                p.print(&l.args()[0], &below)?.output,
                p.print(&l.args()[2], &below)?.output
            ));
        }
        format!(r"{}_{{\substack{{{}}}}} ", sign, rows.join(r"\\"))
    };
    // the body is a product or tighter, the whole bracketed when a factor follows it
    let body: Res = p.print(&e.args()[0], &ctx.body())?;
    let mut r: Res = PrintResult::new(format!("{}{}", tex, body.output), Binding::Product);
    r.open_ended = true;
    r.applies = body.applies;
    Ok(r)
}

fn integral(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let (function, variable, bounds) = crate::integrals::integral::parts(e);
    let mut tex: String = r"\int".to_string();
    if let Some((a, b)) = bounds {
        if !ctx.inline && !p.settings.itex {
            tex.push_str(r"\limits");
        }
        tex.push_str(&format!(
            "_{{{}}}^{{{}}}",
            p.print(a, &ctx.subscript())?.output,
            p.print(b, &ctx.superscript())?.output
        ));
    }
    integral_form(p, &tex, function, std::slice::from_ref(variable), ctx)
}

fn integral_form(p: &LatexPrinter, sign: &str, integrand: &Basic, variables: &[Basic], ctx: &Ctx) -> R {
    let body: Res = p.print(integrand, &ctx.body())?;
    let mut differentials: String = String::new();
    for v in variables {
        differentials.push_str(&format!(r"\, {}{}", p.diff_operator(), p.inside(v, ctx)?));
    }
    let mut r: Res = PrintResult::new(format!("{} {}{}", sign, body.output, differentials), Binding::Product);
    r.open_ended = true;
    r.applies = body.applies;
    Ok(r)
}

/// The operands of a set operation: a nested operation is bracketed.
fn operation(p: &LatexPrinter, operands: &[Basic], sign: &str, ctx: &Ctx) -> R {
    let inner: Ctx = ctx.right(Binding::SetOperation);
    let mut parts: Vec<String> = Vec::new();
    for s in operands {
        parts.push(p.print(s, &inner)?.output);
    }
    Ok(PrintResult::new(
        parts.join(&format!(" {} ", sign)),
        Binding::SetOperation,
    ))
}

fn interval(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let (start, end) = (&e.args()[0], &e.args()[1]);
    if start == end {
        return Ok(atom(format!(r"\left\{{{}\right\}}", p.inside(start, ctx)?)));
    }
    let left_open: bool = matches!(e.args()[2].head(), Head::BooleanTrue);
    let right_open: bool = matches!(e.args()[3].head(), Head::BooleanTrue);
    // an infinite end is open, however the interval was built
    let left: &str = if left_open || matches!(start.head(), Head::NegativeInfinity) {
        "("
    } else {
        "["
    };
    let right: &str = if right_open || matches!(end.head(), Head::Infinity) {
        ")"
    } else {
        "]"
    };
    Ok(atom(format!(
        r"\left{}{}{} {}\right{}",
        left,
        p.inside(start, ctx)?,
        p.comma(),
        p.inside(end, ctx)?,
        right
    )))
}

fn image_set(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let side: Ctx = ctx.side();
    let variables: &[Basic] = e.args()[0].args();
    let base_sets: &[Basic] = &e.args()[2..];
    let mut members: Vec<String> = Vec::new();
    for (v, s) in variables.iter().zip(base_sets) {
        members.push(format!(r"{} \in {}", p.inside(v, ctx)?, p.print(s, &side)?.output));
    }
    Ok(atom(format!(
        r"\left\{{{}\; \middle|\; {}\right\}}",
        p.inside(&e.args()[1], ctx)?,
        members.join(", ")
    )))
}

fn condition_set(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let (sym, condition, base_set) = (&e.args()[0], &e.args()[1], &e.args()[2]);
    let variables: String = if matches!(sym.head(), Head::Tuple) && !sym.args().is_empty() {
        p.join(sym.args(), ctx, Some(", "))?
    } else {
        p.inside(sym, ctx)?
    };
    if matches!(base_set.head(), Head::UniversalSet) {
        return Ok(atom(format!(
            r"\left\{{{}\; \middle|\; {} \right\}}",
            variables,
            p.inside(condition, ctx)?
        )));
    }
    // the condition is an operand of the conjunction
    let condition: String = p.print(condition, &ctx.connective())?.output;
    let base: String = p.print(base_set, &ctx.side())?.output;
    Ok(atom(format!(
        r"\left\{{{}\; \middle|\; {} \in {} \wedge {} \right\}}",
        variables, variables, base, condition
    )))
}

fn quantifier(p: &LatexPrinter, e: &Basic, sign: &str, ctx: &Ctx) -> R {
    let mut variables: Vec<String> = Vec::new();
    for v in e.args()[0].args() {
        variables.push(p.inside(v, ctx)?);
    }
    // the formula extends to the right: bracketed wherever it is not delimited
    Ok(PrintResult::new(
        format!(
            r"{} {}\colon {}",
            sign,
            variables.join(", "),
            p.inside(&e.args()[1], ctx)?
        ),
        Binding::Block,
    ))
}

fn limit(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let (expr, variable, z0, dir) = (&e.args()[0], &e.args()[1], &e.args()[2], &e.args()[3]);
    let direction: String = dir.to_string();
    let point: String = if direction == "+-" || matches!(z0.head(), Head::Infinity | Head::NegativeInfinity) {
        p.print(z0, &ctx.subscript())?.output
    } else {
        // the mark of the direction is a script of the point
        format!("{}^{}", p.print(z0, &ctx.point())?.output, direction)
    };
    let body: Res = p.print(expr, &ctx.body())?;
    let tex: String = format!(
        r"\lim_{{{} \to {}}} {}",
        p.print(variable, &ctx.subscript())?.output,
        point,
        body.output
    );
    let mut r: Res = PrintResult::new(tex, Binding::Product);
    r.open_ended = true;
    r.applies = body.applies;
    Ok(r)
}

fn order(p: &LatexPrinter, e: &Basic, ctx: &Ctx) -> R {
    let (expr, pairs) = crate::core::order::parts(e);
    let mut tex: String = p.inside(&expr, ctx)?;
    let variables: Vec<Basic> = pairs.iter().map(|(v, _)| v.clone()).collect();
    let point: Vec<Basic> = pairs.iter().map(|(_, z)| z.clone()).collect();
    if (!point.is_empty() && point.iter().any(|z| *z != Basic::zero())) || variables.len() > 1 {
        tex.push_str("; ");
        if variables.len() > 1 {
            tex.push_str(&p.tuple_of(&variables, ctx)?);
        } else if let Some(v) = variables.first() {
            tex.push_str(&p.inside(v, ctx)?);
        }
        tex.push_str(r"\rightarrow ");
        if point.len() > 1 {
            tex.push_str(&p.tuple_of(&point, ctx)?);
        } else {
            tex.push_str(&p.inside(&point[0], ctx)?);
        }
    }
    let mut r: Res = PrintResult::new(format!(r"O\left({}\right)", tex), Binding::Application);
    r.applies = true;
    Ok(r)
}
