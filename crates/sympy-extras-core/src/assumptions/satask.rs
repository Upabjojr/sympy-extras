//! The consistency of predicates with the known facts about them, ported
//! from `sympy_extras.assumptions.satask` (a typed port of SymPy's
//! `satask`, `sathandlers`, `facts` and `ask_generated`): the formula in
//! conjunctive normal form with the composite predicates as disjunctions of
//! primitive ones and the relations as binary predicates, the facts of the
//! classes of the arguments added recursively, and the known number facts
//! for every argument; consistent when the clauses have a model (the DPLL
//! solver of `logic::dpll2`).

use crate::core::ask::q_applied;
use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::expr::is;
use crate::core::kind::{kind_of, Kind};
use crate::core::logic::{and_, equivalent, implies, not_, or_};
use crate::error::PolyResult;
use crate::logic::cnf::{conjunction, to_clauses, Clauses};

/// The composite predicates as disjunctions of primitive ones.
fn composite_parts(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "real" => &["negative", "zero", "positive"],
        "integer" => &["even", "odd"],
        "nonpositive" => &["negative", "zero"],
        "nonzero" => &["negative", "positive"],
        "nonnegative" => &["zero", "positive"],
        "extended_real" => &["negative_infinite", "negative", "zero", "positive", "positive_infinite"],
        "extended_positive" => &["positive", "positive_infinite"],
        "extended_negative" => &["negative", "negative_infinite"],
        "extended_nonzero" => &["negative_infinite", "negative", "positive", "positive_infinite"],
        "extended_nonpositive" => &["negative_infinite", "negative", "zero"],
        "extended_nonnegative" => &["zero", "positive", "positive_infinite"],
        "complex" => &["algebraic", "transcendental"],
        _ => return None,
    })
}

/// The known facts between the unary number predicates in conjunctive
/// normal form: `(predicate, negated)` (`get_all_known_number_facts`).
const NUMBER_FACTS: &[&[(&str, bool)]] = &[
    &[("algebraic", false), ("imaginary", true), ("transcendental", false)],
    &[("algebraic", false), ("negative", true), ("transcendental", false)],
    &[("algebraic", false), ("positive", true), ("transcendental", false)],
    &[("algebraic", false), ("rational", true)],
    &[("algebraic", false), ("transcendental", false), ("zero", true)],
    &[("algebraic", true), ("finite", false)],
    &[("algebraic", true), ("transcendental", true)],
    &[("antihermitian", false), ("hermitian", false), ("zero", true)],
    &[("antihermitian", false), ("imaginary", true)],
    &[("commutative", false), ("finite", true)],
    &[("commutative", false), ("infinite", true)],
    &[
        ("composite", false),
        ("even", true),
        ("positive", true),
        ("prime", false),
    ],
    &[("composite", true), ("even", false), ("odd", false)],
    &[("composite", true), ("positive", false)],
    &[("composite", true), ("prime", true)],
    &[("even", false), ("odd", false), ("prime", true)],
    &[("even", false), ("zero", true)],
    &[("even", true), ("odd", true)],
    &[("even", true), ("rational", false)],
    &[("finite", false), ("transcendental", true)],
    &[("finite", true), ("infinite", true)],
    &[("hermitian", false), ("negative", true)],
    &[("hermitian", false), ("positive", true)],
    &[("hermitian", false), ("zero", true)],
    &[("imaginary", true), ("negative", true)],
    &[("imaginary", true), ("positive", true)],
    &[("imaginary", true), ("zero", true)],
    &[("infinite", false), ("negative_infinite", true)],
    &[("infinite", false), ("positive_infinite", true)],
    &[("irrational", false), ("negative", true), ("rational", false)],
    &[("irrational", false), ("positive", true), ("rational", false)],
    &[("irrational", false), ("rational", false), ("zero", true)],
    &[
        ("irrational", true),
        ("negative", false),
        ("positive", false),
        ("zero", false),
    ],
    &[("irrational", true), ("rational", true)],
    &[
        ("negative", false),
        ("positive", false),
        ("rational", true),
        ("zero", false),
    ],
    &[("negative", true), ("negative_infinite", true)],
    &[("negative", true), ("positive", true)],
    &[("negative", true), ("positive_infinite", true)],
    &[("negative", true), ("zero", true)],
    &[("negative_infinite", true), ("positive", true)],
    &[("negative_infinite", true), ("positive_infinite", true)],
    &[("negative_infinite", true), ("zero", true)],
    &[("odd", true), ("rational", false)],
    &[("positive", false), ("prime", true)],
    &[("positive", true), ("positive_infinite", true)],
    &[("positive", true), ("zero", true)],
    &[("positive_infinite", true), ("zero", true)],
];

/// The predicates of the number facts in the order of first use.
fn number_predicates() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for clause in NUMBER_FACTS {
        for (name, _) in clause.iter() {
            if !out.contains(name) {
                out.push(name);
            }
        }
    }
    out
}

fn q1(name: &str, e: &Basic) -> Basic {
    q_applied(name, std::slice::from_ref(e)).expect("a unary predicate of Q")
}

/// The relations as binary predicates, the composite predicates as
/// disjunctions of primitive ones (`_expand`).
fn expand(atom: &Basic) -> Option<Basic> {
    if let Head::Relational(op) = atom.head() {
        let name: &str = match op {
            RelOp::Eq => "eq",
            RelOp::Ne => "ne",
            RelOp::Gt => "gt",
            RelOp::Lt => "lt",
            RelOp::Ge => "ge",
            RelOp::Le => "le",
        };
        return Some(crate::core::assume::applied(
            &crate::core::ask::q(name).ok()?,
            atom.args(),
        ));
    }
    if matches!(atom.head(), Head::AppliedPredicate) {
        if let Head::Predicate(d) = atom.args()[0].head() {
            if d.defined {
                if let Some(parts) = composite_parts(&d.name) {
                    let arguments: &[Basic] = &atom.args()[1..];
                    let alternatives: Vec<Basic> = parts
                        .iter()
                        .map(|p| crate::core::assume::applied(&crate::core::ask::q(p).expect("a predicate"), arguments))
                        .collect();
                    return Some(or_(&alternatives));
                }
            }
        }
    }
    None
}

fn clauses_of(formula: &Basic) -> Clauses {
    to_clauses(formula, Some(&expand))
}

fn all_args(fact: &dyn Fn(&Basic) -> Basic, args: &[Basic]) -> Basic {
    and_(&args.iter().map(fact).collect::<Vec<Basic>>())
}

fn any_arg(fact: &dyn Fn(&Basic) -> Basic, args: &[Basic]) -> Basic {
    or_(&args.iter().map(fact).collect::<Vec<Basic>>())
}

fn exactly_one_arg(fact: &dyn Fn(&Basic) -> Basic, args: &[Basic]) -> Basic {
    let facts: Vec<Basic> = args.iter().map(fact).collect();
    let alternatives: Vec<Basic> = (0..facts.len())
        .map(|i| {
            let mut parts: Vec<Basic> = vec![facts[i].clone()];
            for (j, f) in facts.iter().enumerate() {
                if j != i {
                    parts.push(not_(f));
                }
            }
            and_(&parts)
        })
        .collect();
    or_(&alternatives)
}

fn q(name: &'static str) -> impl Fn(&Basic) -> Basic {
    move |e: &Basic| q1(name, e)
}

fn abs_facts(expr: &Basic) -> Vec<Basic> {
    let arg: &Basic = &expr.args()[0];
    vec![
        q1("nonnegative", expr),
        equivalent(&[not_(&q1("zero", arg)), not_(&q1("zero", expr))]),
        implies(&q1("even", arg), &q1("even", expr)),
        implies(&q1("odd", arg), &q1("odd", expr)),
        implies(&q1("integer", arg), &q1("integer", expr)),
    ]
}

fn add_facts(expr: &Basic) -> Vec<Basic> {
    let args: &[Basic] = expr.args();
    let not_integer = |e: &Basic| not_(&q1("integer", e));
    vec![
        implies(&all_args(&q("positive"), args), &q1("positive", expr)),
        implies(&all_args(&q("negative"), args), &q1("negative", expr)),
        implies(&all_args(&q("real"), args), &q1("real", expr)),
        implies(&all_args(&q("rational"), args), &q1("rational", expr)),
        implies(&all_args(&q("integer"), args), &q1("integer", expr)),
        implies(&exactly_one_arg(&not_integer, args), &not_(&q1("integer", expr))),
        implies(
            &all_args(&q("real"), args),
            &implies(&exactly_one_arg(&q("irrational"), args), &q1("irrational", expr)),
        ),
    ]
}

fn mul_facts(expr: &Basic) -> Vec<Basic> {
    let args: &[Basic] = expr.args();
    let not_rational = |e: &Basic| not_(&q1("rational", e));
    let real_or_imaginary = |e: &Basic| or_(&[q1("imaginary", e), q1("real", e)]);
    // sympy-extras#74 (3): SymPy's facts that one irrational, imaginary or
    // non-rational factor makes the product irrational, imaginary or not an
    // integer forget a zero factor (I*0 is 0); here the other factors are
    // nonzero
    let nonzero: Basic = all_args(&|e: &Basic| not_(&q1("zero", e)), args);
    vec![
        equivalent(&[q1("zero", expr), any_arg(&q("zero"), args)]),
        implies(&all_args(&q("positive"), args), &q1("positive", expr)),
        implies(&all_args(&q("real"), args), &q1("real", expr)),
        implies(&all_args(&q("rational"), args), &q1("rational", expr)),
        implies(&all_args(&q("integer"), args), &q1("integer", expr)),
        implies(
            &and_(&[nonzero.clone(), exactly_one_arg(&not_rational, args)]),
            &not_(&q1("integer", expr)),
        ),
        implies(&all_args(&q("commutative"), args), &q1("commutative", expr)),
        implies(&all_args(&q("prime"), args), &not_(&q1("prime", expr))),
        implies(
            &and_(&[nonzero.clone(), all_args(&real_or_imaginary, args)]),
            &implies(&exactly_one_arg(&q("imaginary"), args), &q1("imaginary", expr)),
        ),
        implies(
            &and_(&[nonzero, all_args(&q("real"), args)]),
            &implies(&exactly_one_arg(&q("irrational"), args), &q1("irrational", expr)),
        ),
        implies(
            &all_args(&q("integer"), args),
            &equivalent(&[any_arg(&q("even"), args), q1("even", expr)]),
        ),
    ]
}

fn pow_facts(expr: &Basic) -> Vec<Basic> {
    let (base, exp) = (&expr.args()[0], &expr.args()[1]);
    vec![
        implies(
            &and_(&[q1("real", base), q1("even", exp), q1("nonnegative", exp)]),
            &q1("nonnegative", expr),
        ),
        implies(
            &and_(&[q1("nonnegative", base), q1("odd", exp), q1("nonnegative", exp)]),
            &q1("nonnegative", expr),
        ),
        implies(
            &and_(&[q1("nonpositive", base), q1("odd", exp), q1("nonnegative", exp)]),
            &q1("nonpositive", expr),
        ),
        equivalent(&[q1("zero", expr), and_(&[q1("zero", base), q1("positive", exp)])]),
    ]
}

fn number_facts(expr: &Basic) -> Vec<Basic> {
    let known: [(&str, Predicate); 10] = [
        ("positive", Predicate::positive),
        ("zero", Predicate::zero),
        ("negative", Predicate::negative),
        ("rational", Predicate::rational),
        ("irrational", Predicate::irrational),
        ("even", Predicate::even),
        ("odd", Predicate::odd),
        ("imaginary", Predicate::imaginary),
        ("prime", Predicate::prime),
        ("composite", Predicate::composite),
    ];
    known
        .iter()
        .filter_map(|(name, p)| is(expr, *p).map(|v| if v { q1(name, expr) } else { not_(&q1(name, expr)) }))
        .collect()
}

fn is_number_symbol(e: &Basic) -> bool {
    matches!(
        e.head(),
        Head::Pi | Head::Exp1 | Head::EulerGamma | Head::Catalan | Head::GoldenRatio | Head::TribonacciConstant
    )
}

/// The facts of the class of `expr` (`_class_facts`).
fn class_facts(expr: &Basic) -> Vec<Basic> {
    if matches!(expr.head(), Head::Function(Function::Abs)) {
        return abs_facts(expr);
    }
    if expr.is_add() {
        return add_facts(expr);
    }
    if expr.is_mul() {
        return mul_facts(expr);
    }
    if expr.is_pow() {
        return pow_facts(expr);
    }
    if expr.is_number_class() || is_number_symbol(expr) || matches!(expr.head(), Head::ImaginaryUnit) {
        return number_facts(expr);
    }
    Vec::new()
}

fn push_unique(v: &mut Vec<Basic>, e: &Basic) {
    if !v.contains(e) {
        v.push(e.clone());
    }
}

/// The arguments of the predicates among the atoms, and the atoms which
/// are not predicates (`_arguments`).
fn arguments(cnf: &Clauses) -> Vec<Basic> {
    let mut found: Vec<Basic> = Vec::new();
    for atom in &cnf.atoms {
        if matches!(atom.head(), Head::AppliedPredicate) {
            for a in &atom.args()[1..] {
                push_unique(&mut found, a);
            }
        } else {
            push_unique(&mut found, atom);
        }
    }
    found
}

/// The known number facts instantiated on `expr` (`_number_facts_of`).
fn number_facts_of(expr: &Basic) -> Clauses {
    let names: Vec<&'static str> = number_predicates();
    let atoms: Vec<Basic> = names.iter().map(|n| q1(n, expr)).collect();
    let index = |name: &str| -> i64 { names.iter().position(|n| *n == name).expect("a predicate") as i64 + 1 };
    let clauses: Vec<Vec<i64>> = NUMBER_FACTS
        .iter()
        .map(|c| {
            let mut literals: Vec<i64> = c
                .iter()
                .map(|(name, negated)| if *negated { -index(name) } else { index(name) })
                .collect();
            literals.sort_by_key(|l| (l.abs(), *l));
            literals
        })
        .collect();
    Clauses { atoms, clauses }
}

/// Whether a Boolean combination of predicates of `Q` is consistent with
/// the known facts (`predicates_consistent`).
pub fn predicates_consistent(predicates: &Basic) -> PolyResult<bool> {
    let cnf: Clauses = clauses_of(predicates);
    if cnf.is_false() {
        return Ok(false);
    }
    let mut parts: Vec<Clauses> = vec![cnf.clone()];
    let mut seen: Vec<Basic> = Vec::new();
    let mut exprs: Vec<Basic> = arguments(&cnf);
    while !exprs.is_empty() {
        let mut new: Vec<Basic> = Vec::new();
        for e in &exprs {
            push_unique(&mut seen, e);
        }
        for e in &exprs {
            for fact in class_facts(e) {
                let fact_cnf: Clauses = clauses_of(&fact);
                for atom in &fact_cnf.atoms {
                    if matches!(atom.head(), Head::AppliedPredicate) {
                        for a in &atom.args()[1..] {
                            if !seen.contains(a) {
                                push_unique(&mut new, a);
                            }
                        }
                    }
                }
                parts.push(fact_cnf);
            }
        }
        exprs = new;
    }
    if seen
        .iter()
        .any(|e| matches!(kind_of(e), Kind::Number | Kind::Undefined))
    {
        for e in &seen {
            parts.push(number_facts_of(e));
        }
    }
    Ok(crate::logic::dpll2::is_satisfiable(&conjunction(&parts)))
}
