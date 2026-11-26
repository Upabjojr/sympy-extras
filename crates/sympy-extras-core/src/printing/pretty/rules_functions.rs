//! Port of `sympy_extras/printing/pretty/rules_functions.py`: the
//! pretty-printing rules of the functions (applications, the elementary
//! functions with a notation of their own, the special functions,
//! piecewise functions).

// PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_Function and the rules of the named
// functions, _print_hyper, _print_meijerg, _print_Piecewise, _print_binomial, _print_number_function,
// _print_KroneckerDelta, the elliptic integrals) (SymPy 1.14, BSD license; through
// sympy_extras/printing/pretty/rules_functions.py)

use super::engine::PrettyPrinter;
use super::layout::{Picture, PrettyContext, PrettyResult};
use super::rules::{exponent_picture, raised};
use super::symbology::{greek_unicode, sub_char};
use crate::core::basic::{Basic, Function, Head};
use crate::error::PolyResult;
use crate::printing::framework::{Binding, ChildOptions, Position, PrintResult};

type R = PolyResult<PrettyResult>;

/// The rule of a function node (and of `Piecewise`, `ITE`, `hyper`,
/// `meijerg`), `None` for the other nodes.
pub fn function_rule(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> PolyResult<Option<PrettyResult>> {
    Ok(Some(match e.head() {
        Head::Undefined(u) => {
            // an applied undefined function: its name is a symbol name
            let arguments: Picture = p.arguments(e.args(), ctx, ", ")?;
            p.applied(&Picture::of(&p.symbology.symbol(&u.name)), &arguments, ctx)
        }
        Head::Piecewise => {
            let pairs: Vec<(Basic, Basic)> = e
                .args()
                .iter()
                .map(|pair| (pair.args()[0].clone(), pair.args()[1].clone()))
                .collect();
            cases(p, &pairs, ctx)?
        }
        Head::Ite => {
            // as SymPy: the cases of ITE(c, a, b).rewrite(Piecewise)
            let (condition, then, otherwise) = (&e.args()[0], &e.args()[1], &e.args()[2]);
            cases(
                p,
                &[
                    (then.clone(), condition.clone()),
                    (otherwise.clone(), crate::core::logic::true_()),
                ],
                ctx,
            )?
        }
        Head::ExprCondPair => p.sequence(e.args(), ctx, "(", ")", ", ")?,
        Head::Hyper => hyper(p, e, ctx)?,
        Head::MeijerG => meijerg(p, e, ctx)?,
        Head::Function(f) => defined(p, *f, e, ctx)?,
        _ => return Ok(None),
    }))
}

fn named(p: &PrettyPrinter, unicode_name: &str, ascii_name: &str, args: &[Basic], ctx: &PrettyContext) -> R {
    p.function(if p.unicode() { unicode_name } else { ascii_name }, args, ctx)
}

fn greek_named(p: &PrettyPrinter, letter: &str, ascii_name: &str, args: &[Basic], ctx: &PrettyContext) -> R {
    named(p, greek_unicode(letter).expect("a Greek letter"), ascii_name, args, ctx)
}

fn operand_ctx(ctx: &PrettyContext) -> PrettyContext {
    ctx.at(Position::Operand, Binding::Application, ChildOptions::default())
}

fn defined(p: &PrettyPrinter, f: Function, e: &Basic, ctx: &PrettyContext) -> R {
    use Function as F;
    let args: &[Basic] = e.args();
    let s = &p.symbology;
    match f {
        F::Exp => raised(p, &Basic::e(), &exponent_picture(p, &args[0], ctx)?, ctx),
        F::Factorial => Ok(PrintResult::new(
            p.operand(&args[0], &operand_ctx(ctx))?.rtext("!"),
            Binding::Affix,
        )),
        F::Factorial2 => Ok(PrintResult::new(
            p.operand(&args[0], &operand_ctx(ctx))?.rtext("!!"),
            Binding::Affix,
        )),
        F::Subfactorial => Ok(PrintResult::new(
            p.operand(&args[0], &operand_ctx(ctx))?.ltext("!"),
            Binding::Affix,
        )),
        F::Binomial => {
            let n: Picture = p.operand(&args[0], &ctx.delimited())?;
            let k: Picture = p.operand(&args[1], &ctx.delimited())?;
            let bar: Picture = Picture::blank(n.width().max(k.width()), 1);
            let picture: Picture = p.parens(&k.above(&[&n, &bar]), "(", ")", false);
            let baseline: i64 = (picture.baseline + 1).div_euclid(2);
            Ok(PrintResult::new(picture.with_baseline(baseline), Binding::Atom))
        }
        F::Conjugate => {
            // the bar covers the operand: it needs no brackets and the result is closed
            let operand: Picture = p.operand(&args[0], &ctx.delimited())?;
            let bar: Picture = Picture::of(&s.hobj("_", operand.width()));
            Ok(PrintResult::new(operand.above(&[&bar]), Binding::Atom))
        }
        F::Abs => Ok(PrintResult::new(
            p.parens(&p.operand(&args[0], &ctx.delimited())?, "|", "|", false),
            Binding::Atom,
        )),
        F::Floor | F::Ceiling => {
            let (name, left, right) = if f == F::Floor {
                ("floor", "lfloor", "rfloor")
            } else {
                ("ceiling", "lceil", "rceil")
            };
            if !p.unicode() {
                return p.function(name, args, ctx);
            }
            Ok(PrintResult::new(
                p.parens(&p.operand(&args[0], &ctx.delimited())?, left, right, false),
                Binding::Atom,
            ))
        }
        F::MathieuC => named(p, "C", "C", args, ctx),
        F::MathieuS => named(p, "S", "S", args, ctx),
        F::MathieuCPrime => named(p, "C'", "C'", args, ctx),
        F::MathieuSPrime => named(p, "S'", "S'", args, ctx),
        F::PolyLog => subscripted_name(p, args, ctx, "Li", "polylog"),
        F::Expint => subscripted_name(p, args, ctx, "E", "expint"),
        F::LerchPhi => greek_named(p, "Phi", "lerchphi", args, ctx),
        F::DirichletEta => greek_named(p, "eta", "dirichlet_eta", args, ctx),
        F::Heaviside => {
            let name: String = s.greek("theta", "Heaviside");
            if args.len() == 2 && args[1] == Basic::half() {
                return p.function(&name, &args[..1], ctx);
            }
            p.function(&name, args, ctx)
        }
        F::FresnelS => named(p, "S", "S", args, ctx),
        F::FresnelC => named(p, "C", "C", args, ctx),
        F::AiryAi => named(p, "Ai", "Ai", args, ctx),
        F::AiryBi => named(p, "Bi", "Bi", args, ctx),
        F::AiryAiPrime => named(p, "Ai'", "Ai'", args, ctx),
        F::AiryBiPrime => named(p, "Bi'", "Bi'", args, ctx),
        F::LambertW => named(p, "W", "W", args, ctx),
        F::SingularityFunction => {
            // <x - a> raised to n; the angle brackets close the shift
            let shift_node: Basic = crate::core::add::sub(&args[0], &args[1])?;
            let shift: Picture = p.parens(&p.operand(&shift_node, &ctx.delimited())?, "<", ">", false);
            Ok(PrintResult::new(
                p.power(&shift, &exponent_picture(p, &args[2], ctx)?),
                Binding::Power,
            ))
        }
        F::Beta => greek_named(p, "Beta", "B", args, ctx),
        F::BetaInc => named(p, "B'", "B'", args, ctx),
        F::BetaIncRegularized => named(p, "I", "I", args, ctx),
        F::Gamma | F::UpperGamma => greek_named(p, "Gamma", "Gamma", args, ctx),
        F::LowerGamma => greek_named(p, "gamma", "lowergamma", args, ctx),
        F::DiracDelta => {
            if !p.unicode() {
                return p.function("DiracDelta", args, ctx);
            }
            let delta: Picture = Picture::of(greek_unicode("delta").expect("delta"));
            if args.len() == 2 {
                // the order of the derivative, between parentheses, as an exponent
                let order: Picture = p.parens(&p.operand(&args[1], &ctx.delimited())?, "(", ")", false);
                return Ok(p.applied(
                    &p.power(&delta, &order).rtext(" "),
                    &p.operand(&args[0], &ctx.delimited())?,
                    ctx,
                ));
            }
            Ok(p.applied(&delta, &p.operand(&args[0], &ctx.delimited())?, ctx))
        }
        F::Chi => named(p, "Chi", "Chi", args, ctx),
        F::EllipticK => p.function("K", args, ctx),
        F::EllipticE => {
            if args.len() == 1 {
                return p.function("E", args, ctx);
            }
            Ok(p.applied(&Picture::of("E"), &bar_separated(p, &args[0], &args[1], ctx)?, ctx))
        }
        F::EllipticF => Ok(p.applied(&Picture::of("F"), &bar_separated(p, &args[0], &args[1], ctx)?, ctx)),
        F::EllipticPi => {
            let name: Picture = Picture::of(&s.greek("Pi", "Pi"));
            if args.len() == 2 {
                return Ok(p.applied(&name, &bar_separated(p, &args[0], &args[1], ctx)?, ctx));
            }
            let inside: Picture = bar_separated(p, &args[1], &args[2], ctx)?.ltext("; ");
            let first: Picture = p.operand(&args[0], &ctx.delimited())?;
            Ok(p.applied(&name, &inside.left(&[&first]), ctx))
        }
        F::Euler => number_function(p, "E", "E", args, ctx),
        F::Catalan => number_function(p, "C", "C", args, ctx),
        F::Bernoulli | F::Bell => number_function(p, "B", "B", args, ctx),
        F::Lucas => number_function(p, "L", "L", args, ctx),
        F::Fibonacci => number_function(p, "F", "F", args, ctx),
        F::Tribonacci => number_function(p, "T", "T", args, ctx),
        F::Stieltjes => number_function(p, greek_unicode("gamma").expect("gamma"), "stieltjes", args, ctx),
        F::KroneckerDelta => {
            let c: PrettyContext = ctx.subscript();
            let indices: Picture =
                Picture::beside(&[&p.operand(&args[0], &c)?, &Picture::of(","), &p.operand(&args[1], &c)?]);
            Ok(PrintResult::new(
                p.subscripted(&Picture::of(&s.greek("delta", "d")), &indices),
                Binding::Application,
            ))
        }
        F::PrimeNu => greek_named(p, "nu", "nu", args, ctx),
        F::PrimeOmega => greek_named(p, "Omega", "Omega", args, ctx),
        F::Mod => unreachable!("Mod is a rule of the core"),
        _ => {
            let mut name: String = f.name().to_string();
            if p.unicode() {
                if let Some(g) = greek_unicode(&name) {
                    name = g.to_string(); // zeta
                }
            }
            p.function(&name, args, ctx)
        }
    }
}

/// `Li₂(z)` when the first argument has a Unicode subscript form
/// (`_subscripted_name`).
fn subscripted_name(p: &PrettyPrinter, args: &[Basic], ctx: &PrettyContext, letter: &str, name: &str) -> R {
    let index: String = p.operand(&args[0], &ctx.delimited())?.to_string();
    if p.symbology.subscriptable(&index) {
        let lowered: String = index
            .chars()
            .map(|c| sub_char(&c.to_string()).expect("a subscript"))
            .collect();
        return p.function(&format!("{}{}", letter, lowered), &args[1..], ctx);
    }
    p.function(name, args, ctx)
}

fn bar_separated(p: &PrettyPrinter, left: &Basic, right: &Basic, ctx: &PrettyContext) -> PolyResult<Picture> {
    Ok(p.separated(
        &p.operand(left, &ctx.delimited())?,
        &p.operand(right, &ctx.delimited())?,
        "",
        "",
    ))
}

/// `B_n`, and `B_n(x)` with a second argument (`_number_function`).
fn number_function(p: &PrettyPrinter, unicode_name: &str, ascii_name: &str, args: &[Basic], ctx: &PrettyContext) -> R {
    let name: Picture = Picture::of(if p.unicode() { unicode_name } else { ascii_name });
    let index: Picture = p.operand(&args[0], &ctx.subscript())?;
    let head: Picture = name.right(&[&Picture::blank(index.width(), 1).below(&[&index])]);
    if args.len() == 1 {
        return Ok(PrintResult::new(head, Binding::Application));
    }
    let arguments: Picture = p.arguments(&args[1..], ctx, ", ")?;
    Ok(p.applied(&head, &arguments, ctx))
}

/// The values with their conditions behind a brace (`_cases`).
fn cases(p: &PrettyPrinter, pairs: &[(Basic, Basic)], ctx: &PrettyContext) -> R {
    let mut rows: Vec<(Picture, Picture)> = Vec::new();
    for (expr, cond) in pairs {
        let value: Picture = p.operand(expr, &ctx.delimited())?;
        let condition: Picture = if matches!(cond.head(), Head::BooleanTrue) {
            Picture::of("otherwise")
        } else {
            p.operand(cond, &ctx.delimited())?.ltext("for ")
        };
        rows.push((value, condition));
    }
    if rows.is_empty() {
        return p.function("Piecewise", &[], ctx);
    }
    let widths: [usize; 2] = [
        rows.iter().map(|r| r.0.width()).max().unwrap_or(0),
        rows.iter().map(|r| r.1.width()).max().unwrap_or(0),
    ];
    let mut table: Option<Picture> = None;
    for (a, b) in &rows {
        let mut cells: Vec<Picture> = Vec::new();
        for (j, cell) in [a, b].into_iter().enumerate() {
            let delta: usize = widths[j] - cell.width();
            let left: usize = delta / 2;
            cells.push(Picture::beside(&[
                &Picture::blank(left, 1),
                cell,
                &Picture::blank(delta - left, 1),
            ]));
        }
        let line: Picture = Picture::beside(&[&cells[0], &Picture::blank(2, 1), &cells[1]]);
        table = Some(match table {
            None => line,
            Some(t) => t.below(&[&Picture::of(" "), &line]),
        });
    }
    let picture: Picture = p.parens(&table.expect("rows"), "{", "", false);
    // the brace is open on the right: the result is bracketed everywhere
    let baseline: i64 = (picture.height() / 2) as i64;
    Ok(PrintResult::new(picture.with_baseline(baseline), Binding::Lowest))
}

fn vector(p: &PrettyPrinter, items: &[Basic], ctx: &PrettyContext) -> PolyResult<Picture> {
    if items.is_empty() {
        return Ok(Picture::of(" "));
    }
    p.arguments(items, ctx, ", ")
}

/// The stylised letter with its indices in front of the table (`_lettered`).
fn lettered(p: &PrettyPrinter, letter: &str, table: &Picture, left: &Picture, right: &Picture) -> PrettyResult {
    let above: usize = (table.height() / 2).saturating_sub(1);
    let below: usize = table.height() - above - 1;
    let drawing = p.symbology.annotated(letter);
    let text: String = format!(
        "{}{}{}",
        "\n".repeat(above.saturating_sub(drawing.top)),
        drawing.image,
        "\n".repeat(below.saturating_sub(drawing.bottom))
    );
    let sign: Picture = Picture::text(&text, (above + drawing.size) as i64);
    let indexed: Picture = indices(&sign, left, right, above, drawing.add, drawing.size, letter);
    PrintResult::new(indexed.right(&[&Picture::of(" "), table]), Binding::Application)
}

fn indices(
    sign: &Picture,
    left: &Picture,
    right: &Picture,
    above: usize,
    add: usize,
    size: usize,
    letter: &str,
) -> Picture {
    if letter == "F" {
        return sign
            .left(&[left])
            .right(&[right])
            .with_baseline((above + size.div_ceil(2)) as i64);
    }
    let height: i64 = sign.baseline - above as i64 - 2;
    let block: Picture = if height > 0 {
        let empty: Picture = Picture::of("");
        let mut parts: Vec<&Picture> = vec![&empty; height as usize];
        parts.push(right);
        left.below(&parts)
    } else {
        left.below(&[right])
    };
    sign.with_baseline(above as i64).right(&[&block]).with_baseline((above + add) as i64)
}

fn hyper(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    use crate::functions::special::hyper::{hyper_ap, hyper_bq};
    let argument: Picture = p.operand(&e.args()[2], &ctx.delimited())?;
    let argument: Picture = argument.with_baseline((argument.height() / 2) as i64).ltext(" ");
    let (ap, bq) = (hyper_ap(e), hyper_bq(e));
    let table: Picture = vector(p, ap, ctx)?.below(&[&Picture::of(" "), &vector(p, bq, ctx)?]);
    let table: Picture = table.with_baseline((table.height() / 2) as i64).rtext(" ");
    let table: Picture = p.parens(&p.separated(&table, &argument, "", ""), "(", ")", false);
    Ok(lettered(
        p,
        "F",
        &table,
        &Picture::of(&ap.len().to_string()),
        &Picture::of(&bq.len().to_string()),
    ))
}

fn same_width(a: Picture, b: Picture) -> (Picture, Picture) {
    let (wa, wb) = (a.width(), b.width());
    if wa > wb {
        let padded: Picture = b.left(&[&Picture::blank(wa - wb, 1)]);
        return (a, padded);
    }
    if wb > wa {
        let padded: Picture = a.left(&[&Picture::blank(wb - wa, 1)]);
        return (padded, b);
    }
    (a, b)
}

fn meijerg(p: &PrettyPrinter, e: &Basic, ctx: &PrettyContext) -> R {
    let [an, aother, bm, bother] = crate::functions::special::hyper::meijerg_lists(e);
    let argument: Picture = p.operand(&e.args()[2], &ctx.delimited())?;
    let argument: Picture = argument.with_baseline((argument.height() / 2) as i64).ltext(" ");
    let mut cells: [[Picture; 2]; 2] = [
        [vector(p, an, ctx)?, vector(p, aother, ctx)?],
        [vector(p, bm, ctx)?, vector(p, bother, ctx)?],
    ];
    for i in 0..2 {
        let width: usize = cells[0][i].width().max(cells[1][i].width());
        for row in cells.iter_mut() {
            let cell: &Picture = &row[i];
            let left: usize = (width - cell.width()) / 2;
            let padded: Picture = Picture::beside(&[
                &Picture::blank(left, 1),
                cell,
                &Picture::blank(width - left - cell.width(), 1),
            ]);
            row[i] = padded;
        }
    }
    let top: Picture = Picture::beside(&[&cells[0][0], &Picture::blank(2, 1), &cells[0][1]]);
    let bottom: Picture = Picture::beside(&[&cells[1][0], &Picture::blank(2, 1), &cells[1][1]]);
    let table: Picture = top.below(&[&Picture::of(" "), &bottom]);
    let table: Picture = table.with_baseline((table.height() / 2) as i64).rtext(" ");
    let table: Picture = p.parens(&p.separated(&table, &argument, "", ""), "(", ")", false);
    let count = |n: usize| Picture::of(&n.to_string());
    let (pp, pm) = same_width(count(an.len() + aother.len()), count(bm.len()));
    let (pq, pn) = same_width(count(bm.len() + bother.len()), count(an.len()));
    let upper: Picture = Picture::beside(&[&pm, &Picture::of(", "), &pn]);
    let lower: Picture = Picture::beside(&[&pp, &Picture::of(", "), &pq]);
    Ok(lettered(p, "G", &table, &upper, &lower))
}
