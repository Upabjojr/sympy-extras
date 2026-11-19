//! Port of `sympy_extras/integrals/tables.py`: definite integrals by table
//! lookup, closed forms from Gradshteyn and Ryzhik with the conditions on
//! their parameters. An entry is a pattern in the variable [`Wilds::x`] and
//! the `Wild` parameters, a range, the value and a condition; [`lookup`]
//! matches the integrand (as given and in a few rewritten forms) and
//! [`table_integral`] returns the value with the conditions the
//! assumptions do not settle.
//!
//! The entries are generated from the Python table node by node (the trees
//! as the evaluating constructors left them); the test compares them.

#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::core::basic::{Basic, Function, Head, RelOp};
use crate::core::function::xreplace_evaluated;
use crate::error::{PolyError, PolyResult};
use crate::integrals::conditions::{decide, ConditionalValue};
use crate::numbers::{Int, Rat};

/// The variable and the parameters of the patterns (`X`, `A`, `B`, `N`,
/// `M`, `C`, `K`).
pub struct Wilds {
    pub x: Basic,
    pub a: Basic,
    pub b: Basic,
    pub n: Basic,
    pub m: Basic,
    pub c: Basic,
    pub k: Basic,
}

/// One entry of the table (`TableEntry`).
#[derive(Debug, Clone)]
pub struct TableEntry {
    pub pattern: Basic,
    pub lower: Basic,
    pub upper: Basic,
    pub value: Basic,
    pub condition: Basic,
    pub reference: &'static str,
}

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn frac(p: i64, q: i64) -> Basic {
    Basic::from_rational(Rat::from(Int::from(p)) / Rat::from(Int::from(q)))
}

/// The variable and the parameters, made once.
pub fn wilds() -> &'static Wilds {
    static W: OnceLock<Wilds> = OnceLock::new();
    W.get_or_init(|| {
        use crate::core::symbol::{Dummy, Wild};
        let x: Basic = Dummy::new("x", &[]).0;
        let i: Basic = Basic::atom(Head::ImaginaryUnit);
        let wild = |name: &str, exclude: Vec<Basic>| Wild::new(name, exclude, vec![], &[]).0;
        Wilds {
            a: wild("a", vec![x.clone()]),
            b: wild("b", vec![x.clone()]),
            n: wild("n", vec![x.clone()]),
            m: wild("m", vec![x.clone()]),
            c: wild("c", vec![x.clone(), i.clone()]),
            k: wild("k", vec![x.clone(), i]),
            x,
        }
    })
}

/// The table (`TABLE`).
pub fn table() -> &'static [TableEntry] {
    static T: OnceLock<Vec<TableEntry>> = OnceLock::new();
    T.get_or_init(|| {
        let w: &Wilds = wilds();
        vec![
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        int(1),
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(
                                            Head::Mul,
                                            vec![
                                                int(-2),
                                                w.a.clone(),
                                                Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                                            ],
                                        ),
                                    ],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.n.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Pow, vec![w.a.clone(), w.n.clone()]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        int(1),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.a.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                int(-1),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                        Basic::new(
                            Head::Relational(RelOp::Lt),
                            vec![Basic::new(Head::Pow, vec![w.a.clone(), int(2)]), int(1)],
                        ),
                    ],
                ),
                reference: "GR 3.613.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                                w.n.clone(),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.n.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Pow,
                            vec![
                                int(2),
                                Basic::new(
                                    Head::Add,
                                    vec![int(-1), Basic::new(Head::Mul, vec![int(-1), w.n.clone()])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                reference: "GR 3.631.9",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                                w.n.clone(),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.n.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Pow,
                            vec![int(2), Basic::new(Head::Mul, vec![int(-1), w.n.clone()])],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                reference: "GR 3.631.9",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Sin),
                                    vec![Basic::new(Head::Mul, vec![frac(1, 2), w.x.clone()])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(
                                Head::Mul,
                                vec![Basic::new(Head::Add, vec![frac(1, 2), w.n.clone()]), w.x.clone()],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::atom(Head::Pi),
                condition: Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                reference: "GR 3.612.7 (the Dirichlet kernel)",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Sin),
                                    vec![Basic::new(Head::Mul, vec![w.n.clone(), w.x.clone()])],
                                ),
                                int(2),
                            ],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]), int(-2)],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(Head::Mul, vec![Basic::atom(Head::Pi), w.n.clone()]),
                condition: Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                reference: "GR 3.624.6 (the Fejer kernel)",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Add,
                            vec![
                                w.a.clone(),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        w.b.clone(),
                                        Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                                    ],
                                ),
                            ],
                        ),
                        int(-1),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.b.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(
                            Head::Relational(RelOp::Gt),
                            vec![
                                Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                            ],
                        ),
                    ],
                ),
                reference: "GR 3.613.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(
                                            Head::Pow,
                                            vec![Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]), int(2)],
                                        ),
                                    ],
                                ),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                                        Basic::new(
                                            Head::Pow,
                                            vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]), int(2)],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        int(-1),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Pow, vec![w.a.clone(), int(-1)]),
                        Basic::new(Head::Pow, vec![w.b.clone(), int(-1)]),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 3.642.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(
                                            Head::Pow,
                                            vec![Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]), int(2)],
                                        ),
                                    ],
                                ),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                                        Basic::new(
                                            Head::Pow,
                                            vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]), int(2)],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        int(-2),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 4),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Pow, vec![w.a.clone(), int(-3)]),
                        Basic::new(Head::Pow, vec![w.b.clone(), int(-3)]),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 3.642.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        w.x.clone(),
                        Basic::new(Head::Function(Function::Cot), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Function(Function::Log), vec![int(2)]),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 3.747.7",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]), int(-1)],
                        ),
                        w.x.clone(),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(Head::Mul, vec![int(2), Basic::atom(Head::Catalan)]),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 3.747.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]), int(-2)],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Function(Function::Log), vec![int(2)]),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 3.747.9",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-2)]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Sin),
                                    vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                                ),
                                int(2),
                            ],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi), w.a.clone()]),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                reference: "GR 3.821.9",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-3)]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Sin),
                                    vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                                ),
                                int(3),
                            ],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(3, 8),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                    ],
                ),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                reference: "GR 3.821.12",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-4)]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Sin),
                                    vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                                ),
                                int(4),
                            ],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 3),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Pow, vec![w.a.clone(), int(3)]),
                    ],
                ),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                reference: "GR 3.821.13",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-2)]),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Function(Function::Min), vec![w.a.clone(), w.b.clone()]),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 3.741.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), w.b.clone()]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 3.741.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: int(0),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), w.a.clone()]),
                    ],
                ),
                reference: "GR 3.741.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(-1), Basic::new(Head::Function(Function::Exp), vec![w.x.clone()])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Add,
                    vec![
                        Basic::new(
                            Head::Mul,
                            vec![frac(-1, 2), Basic::new(Head::Pow, vec![w.a.clone(), int(-1)])],
                        ),
                        Basic::new(
                            Head::Mul,
                            vec![
                                frac(1, 2),
                                Basic::atom(Head::Pi),
                                Basic::new(
                                    Head::Function(Function::Coth),
                                    vec![Basic::new(Head::Mul, vec![Basic::atom(Head::Pi), w.a.clone()])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                reference: "GR 3.911.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(1), Basic::new(Head::Function(Function::Exp), vec![w.x.clone()])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Add,
                    vec![
                        Basic::new(
                            Head::Mul,
                            vec![frac(1, 2), Basic::new(Head::Pow, vec![w.a.clone(), int(-1)])],
                        ),
                        Basic::new(
                            Head::Mul,
                            vec![
                                frac(-1, 2),
                                Basic::atom(Head::Pi),
                                Basic::new(
                                    Head::Pow,
                                    vec![
                                        Basic::new(
                                            Head::Function(Function::Sinh),
                                            vec![Basic::new(Head::Mul, vec![Basic::atom(Head::Pi), w.a.clone()])],
                                        ),
                                        int(-1),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                reference: "GR 3.911.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Cosh),
                                    vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Pow, vec![w.b.clone(), int(-1)]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Cosh),
                                    vec![Basic::new(
                                        Head::Mul,
                                        vec![
                                            frac(1, 2),
                                            Basic::atom(Head::Pi),
                                            w.a.clone(),
                                            Basic::new(Head::Pow, vec![w.b.clone(), int(-1)]),
                                        ],
                                    )],
                                ),
                                int(-1),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 3.981.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Function(Function::Sinh),
                                    vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Pow, vec![w.b.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Tanh),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    frac(1, 2),
                                    Basic::atom(Head::Pi),
                                    w.a.clone(),
                                    Basic::new(Head::Pow, vec![w.b.clone(), int(-1)]),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 3.981.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(Head::Function(Function::Cos), vec![w.a.clone()]),
                                Basic::new(Head::Function(Function::Cosh), vec![w.x.clone()]),
                            ],
                        ),
                        int(-1),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        w.a.clone(),
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Function(Function::Sin), vec![w.a.clone()]), int(-1)],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.a.clone(), Basic::atom(Head::Pi)]),
                    ],
                ),
                reference: "GR 3.513.2",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(Head::Function(Function::Cos), vec![w.a.clone()]),
                                Basic::new(Head::Function(Function::Cosh), vec![w.x.clone()]),
                            ],
                        ),
                        int(-1),
                    ],
                ),
                lower: Basic::atom(Head::NegativeInfinity),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        int(2),
                        w.a.clone(),
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Function(Function::Sin), vec![w.a.clone()]), int(-1)],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.a.clone(), Basic::atom(Head::Pi)]),
                    ],
                ),
                reference: "GR 3.513.2",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()])],
                        ),
                        int(2),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(
                                    Head::Pow,
                                    vec![Basic::new(Head::Function(Function::Log), vec![int(2)]), int(2)],
                                ),
                                Basic::new(
                                    Head::Mul,
                                    vec![frac(1, 12), Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.224.7",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Function(Function::Cos), vec![w.x.clone()])],
                        ),
                        int(2),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(
                                    Head::Pow,
                                    vec![Basic::new(Head::Function(Function::Log), vec![int(2)]), int(2)],
                                ),
                                Basic::new(
                                    Head::Mul,
                                    vec![frac(1, 12), Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.224.7",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        w.x.clone(),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(-1, 2),
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)]),
                        Basic::new(Head::Function(Function::Log), vec![int(2)]),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.224.9",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Log),
                    vec![Basic::new(
                        Head::Add,
                        vec![int(1), Basic::new(Head::Function(Function::Tan), vec![w.x.clone()])],
                    )],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 4), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 8),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Function(Function::Log), vec![int(2)]),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.225.?",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Log),
                    vec![Basic::new(
                        Head::Add,
                        vec![
                            Basic::new(
                                Head::Mul,
                                vec![
                                    Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                    Basic::new(
                                        Head::Pow,
                                        vec![Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]), int(2)],
                                    ),
                                ],
                            ),
                            Basic::new(
                                Head::Mul,
                                vec![
                                    Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                                    Basic::new(
                                        Head::Pow,
                                        vec![Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]), int(2)],
                                    ),
                                ],
                            ),
                        ],
                    )],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(
                                Head::Add,
                                vec![
                                    Basic::new(Head::Mul, vec![frac(1, 2), w.a.clone()]),
                                    Basic::new(Head::Mul, vec![frac(1, 2), w.b.clone()]),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 4.225.5",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Log),
                    vec![Basic::new(
                        Head::Add,
                        vec![
                            int(1),
                            Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                            Basic::new(
                                Head::Mul,
                                vec![
                                    int(-2),
                                    w.a.clone(),
                                    Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                                ],
                            ),
                        ],
                    )],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: int(0),
                condition: Basic::new(
                    Head::Relational(RelOp::Lt),
                    vec![Basic::new(Head::Pow, vec![w.a.clone(), int(2)]), int(1)],
                ),
                reference: "GR 4.224.14",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Log),
                    vec![Basic::new(
                        Head::Add,
                        vec![
                            int(1),
                            Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                            Basic::new(
                                Head::Mul,
                                vec![
                                    int(-2),
                                    w.a.clone(),
                                    Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                                ],
                            ),
                        ],
                    )],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        int(2),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Function(Function::Log), vec![w.a.clone()]),
                    ],
                ),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(1)]),
                reference: "GR 4.224.14",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Log),
                    vec![Basic::new(
                        Head::Add,
                        vec![
                            w.a.clone(),
                            Basic::new(
                                Head::Mul,
                                vec![
                                    w.b.clone(),
                                    Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                                ],
                            ),
                        ],
                    )],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(
                                Head::Add,
                                vec![
                                    Basic::new(Head::Mul, vec![frac(1, 2), w.a.clone()]),
                                    Basic::new(
                                        Head::Mul,
                                        vec![
                                            frac(1, 2),
                                            Basic::new(
                                                Head::Pow,
                                                vec![
                                                    Basic::new(
                                                        Head::Add,
                                                        vec![
                                                            Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                            Basic::new(
                                                                Head::Mul,
                                                                vec![
                                                                    int(-1),
                                                                    Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    frac(1, 2),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(
                            Head::Relational(RelOp::Ge),
                            vec![
                                Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                            ],
                        ),
                    ],
                ),
                reference: "GR 4.224.9",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(1), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Add, vec![int(1), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 8),
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Function(Function::Log), vec![int(2)]),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.291.8",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(1), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(Head::Function(Function::Log), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(Head::Mul, vec![int(-1), Basic::atom(Head::Catalan)]),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.231.12",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        int(1),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(Head::Function(Function::Log), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![frac(-1, 8), Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)])],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.231.2",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(-1), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(Head::Function(Function::Log), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![frac(1, 4), Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)])],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.231.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Add, vec![int(1), w.x.clone()]), int(-1)],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Function(Function::Log), vec![w.x.clone()]), int(2)],
                        ),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![frac(3, 2), Basic::new(Head::Function(Function::Zeta), vec![int(3)])],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.261.4",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(
                                Head::Add,
                                vec![int(1), Basic::new(Head::Mul, vec![int(-1), w.x.clone()])],
                            )],
                        ),
                        Basic::new(Head::Function(Function::Log), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Add,
                    vec![
                        int(2),
                        Basic::new(
                            Head::Mul,
                            vec![frac(-1, 6), Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)])],
                        ),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.221.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Add, vec![int(1), w.x.clone()])],
                        ),
                        Basic::new(Head::Function(Function::Log), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Add,
                    vec![
                        int(2),
                        Basic::new(
                            Head::Mul,
                            vec![int(-2), Basic::new(Head::Function(Function::Log), vec![int(2)])],
                        ),
                        Basic::new(
                            Head::Mul,
                            vec![frac(-1, 12), Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)])],
                        ),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.221.2",
            },
            TableEntry {
                pattern: Basic::new(Head::Function(Function::LogGamma), vec![w.x.clone()]),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Mul, vec![int(2), Basic::atom(Head::Pi)])],
                        ),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 6.441.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Log),
                    vec![Basic::new(Head::Function(Function::Gamma), vec![w.x.clone()])],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Mul, vec![int(2), Basic::atom(Head::Pi)])],
                        ),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 6.441.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(
                                Head::Mul,
                                vec![int(2), Basic::atom(Head::Pi), w.n.clone(), w.x.clone()],
                            )],
                        ),
                        Basic::new(Head::Function(Function::LogGamma), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![frac(1, 4), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.443.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(
                                Head::Mul,
                                vec![int(2), Basic::atom(Head::Pi), w.n.clone(), w.x.clone()],
                            )],
                        ),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Function(Function::Gamma), vec![w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![frac(1, 4), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.443.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Function(Function::LogGamma), vec![w.x.clone()]),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(
                                Head::Mul,
                                vec![int(2), Basic::atom(Head::Pi), w.n.clone(), w.x.clone()],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(-1)]),
                        Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::atom(Head::EulerGamma),
                                Basic::new(
                                    Head::Function(Function::Log),
                                    vec![Basic::new(Head::Mul, vec![int(2), Basic::atom(Head::Pi), w.n.clone()])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.443.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Function(Function::Gamma), vec![w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(
                                Head::Mul,
                                vec![int(2), Basic::atom(Head::Pi), w.n.clone(), w.x.clone()],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(-1)]),
                        Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::atom(Head::EulerGamma),
                                Basic::new(
                                    Head::Function(Function::Log),
                                    vec![Basic::new(Head::Mul, vec![int(2), Basic::atom(Head::Pi), w.n.clone()])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.443.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Exp),
                    vec![Basic::new(
                        Head::Mul,
                        vec![
                            int(-1),
                            Basic::atom(Head::ImaginaryUnit),
                            w.c.clone(),
                            Basic::new(Head::Pow, vec![w.x.clone(), w.n.clone()]),
                        ],
                    )],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                w.c.clone(),
                                Basic::new(
                                    Head::Mul,
                                    vec![int(-1), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    frac(-1, 2),
                                    Basic::atom(Head::Pi),
                                    Basic::atom(Head::ImaginaryUnit),
                                    Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                                ],
                            )],
                        ),
                        Basic::new(
                            Head::Function(Function::Gamma),
                            vec![Basic::new(
                                Head::Add,
                                vec![int(1), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.c.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), int(1)]),
                    ],
                ),
                reference: "GR 3.381.4, rotated",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Function(Function::Exp),
                    vec![Basic::new(
                        Head::Mul,
                        vec![
                            Basic::atom(Head::ImaginaryUnit),
                            w.c.clone(),
                            Basic::new(Head::Pow, vec![w.x.clone(), w.n.clone()]),
                        ],
                    )],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                w.c.clone(),
                                Basic::new(
                                    Head::Mul,
                                    vec![int(-1), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    frac(1, 2),
                                    Basic::atom(Head::Pi),
                                    Basic::atom(Head::ImaginaryUnit),
                                    Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                                ],
                            )],
                        ),
                        Basic::new(
                            Head::Function(Function::Gamma),
                            vec![Basic::new(
                                Head::Add,
                                vec![int(1), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.c.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), int(1)]),
                    ],
                ),
                reference: "GR 3.381.4, rotated",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![w.x.clone(), Basic::new(Head::Add, vec![int(-1), w.a.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    Basic::atom(Head::ImaginaryUnit),
                                    w.c.clone(),
                                    Basic::new(Head::Pow, vec![w.x.clone(), w.n.clone()]),
                                ],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                w.c.clone(),
                                Basic::new(
                                    Head::Mul,
                                    vec![int(-1), w.a.clone(), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                                ),
                            ],
                        ),
                        Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    frac(-1, 2),
                                    Basic::atom(Head::Pi),
                                    Basic::atom(Head::ImaginaryUnit),
                                    w.a.clone(),
                                    Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                                ],
                            )],
                        ),
                        Basic::new(
                            Head::Function(Function::Gamma),
                            vec![Basic::new(
                                Head::Mul,
                                vec![w.a.clone(), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.c.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), w.a.clone()]),
                    ],
                ),
                reference: "GR 3.381.4, rotated",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![w.x.clone(), Basic::new(Head::Add, vec![int(-1), w.a.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    Basic::atom(Head::ImaginaryUnit),
                                    w.c.clone(),
                                    Basic::new(Head::Pow, vec![w.x.clone(), w.n.clone()]),
                                ],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                w.c.clone(),
                                Basic::new(
                                    Head::Mul,
                                    vec![int(-1), w.a.clone(), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                                ),
                            ],
                        ),
                        Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    frac(1, 2),
                                    Basic::atom(Head::Pi),
                                    Basic::atom(Head::ImaginaryUnit),
                                    w.a.clone(),
                                    Basic::new(Head::Pow, vec![w.n.clone(), int(-1)]),
                                ],
                            )],
                        ),
                        Basic::new(
                            Head::Function(Function::Gamma),
                            vec![Basic::new(
                                Head::Mul,
                                vec![w.a.clone(), Basic::new(Head::Pow, vec![w.n.clone(), int(-1)])],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.c.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), w.a.clone()]),
                    ],
                ),
                reference: "GR 3.381.4, rotated",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.k.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: w.k.clone(),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Function(Function::BesselK),
                    vec![int(0), Basic::new(Head::Mul, vec![int(-1), w.c.clone(), w.k.clone()])],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.c.clone(), int(0)]),
                    ],
                ),
                reference: "GR 3.364.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![
                                int(0),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        w.a.clone(),
                                        Basic::new(
                                            Head::Pow,
                                            vec![
                                                Basic::new(
                                                    Head::Add,
                                                    vec![
                                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                                        Basic::new(
                                                            Head::Mul,
                                                            vec![
                                                                int(-1),
                                                                Basic::new(Head::Pow, vec![w.k.clone(), int(2)]),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                frac(1, 2),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: w.k.clone(),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    w.k.clone(),
                                    Basic::new(
                                        Head::Pow,
                                        vec![
                                            Basic::new(
                                                Head::Add,
                                                vec![
                                                    Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                    Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                                ],
                                            ),
                                            frac(1, 2),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.c.clone(), int(0)]),
                    ],
                ),
                reference: "AS 29.3.91",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselI),
                            vec![
                                int(0),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        w.a.clone(),
                                        Basic::new(
                                            Head::Pow,
                                            vec![
                                                Basic::new(
                                                    Head::Add,
                                                    vec![
                                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                                        Basic::new(
                                                            Head::Mul,
                                                            vec![
                                                                int(-1),
                                                                Basic::new(Head::Pow, vec![w.k.clone(), int(2)]),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                frac(1, 2),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: w.k.clone(),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.a.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    w.k.clone(),
                                    Basic::new(
                                        Head::Pow,
                                        vec![
                                            Basic::new(
                                                Head::Add,
                                                vec![
                                                    Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                                    Basic::new(
                                                        Head::Mul,
                                                        vec![int(-1), Basic::new(Head::Pow, vec![w.a.clone(), int(2)])],
                                                    ),
                                                ],
                                            ),
                                            frac(1, 2),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(
                            Head::Relational(RelOp::Lt),
                            vec![Basic::new(Head::Add, vec![w.a.clone(), w.c.clone()]), int(0)],
                        ),
                    ],
                ),
                reference: "AS 29.3.92",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.k.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![
                                int(1),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        w.a.clone(),
                                        Basic::new(
                                            Head::Pow,
                                            vec![
                                                Basic::new(
                                                    Head::Add,
                                                    vec![
                                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                                        Basic::new(
                                                            Head::Mul,
                                                            vec![
                                                                int(-1),
                                                                Basic::new(Head::Pow, vec![w.k.clone(), int(2)]),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                frac(1, 2),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: w.k.clone(),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.a.clone(), int(-1)]),
                        Basic::new(Head::Pow, vec![w.k.clone(), int(-1)]),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        int(-1),
                                        Basic::new(
                                            Head::Function(Function::Exp),
                                            vec![Basic::new(
                                                Head::Mul,
                                                vec![
                                                    int(-1),
                                                    w.k.clone(),
                                                    Basic::new(
                                                        Head::Pow,
                                                        vec![
                                                            Basic::new(
                                                                Head::Add,
                                                                vec![
                                                                    Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                                    Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                                                ],
                                                            ),
                                                            frac(1, 2),
                                                        ],
                                                    ),
                                                ],
                                            )],
                                        ),
                                    ],
                                ),
                                Basic::new(
                                    Head::Function(Function::Exp),
                                    vec![Basic::new(Head::Mul, vec![w.c.clone(), w.k.clone()])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.c.clone(), int(0)]),
                    ],
                ),
                reference: "AS 29.3.93",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.k.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::BesselI),
                            vec![
                                int(1),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        w.a.clone(),
                                        Basic::new(
                                            Head::Pow,
                                            vec![
                                                Basic::new(
                                                    Head::Add,
                                                    vec![
                                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                                        Basic::new(
                                                            Head::Mul,
                                                            vec![
                                                                int(-1),
                                                                Basic::new(Head::Pow, vec![w.k.clone(), int(2)]),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                frac(1, 2),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: w.k.clone(),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.a.clone(), int(-1)]),
                        Basic::new(Head::Pow, vec![w.k.clone(), int(-1)]),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        int(-1),
                                        Basic::new(
                                            Head::Function(Function::Exp),
                                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.k.clone()])],
                                        ),
                                    ],
                                ),
                                Basic::new(
                                    Head::Function(Function::Exp),
                                    vec![Basic::new(
                                        Head::Mul,
                                        vec![
                                            int(-1),
                                            w.k.clone(),
                                            Basic::new(
                                                Head::Pow,
                                                vec![
                                                    Basic::new(
                                                        Head::Add,
                                                        vec![
                                                            Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                                            Basic::new(
                                                                Head::Mul,
                                                                vec![
                                                                    int(-1),
                                                                    Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    frac(1, 2),
                                                ],
                                            ),
                                        ],
                                    )],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(
                            Head::Relational(RelOp::Lt),
                            vec![Basic::new(Head::Add, vec![w.a.clone(), w.c.clone()]), int(0)],
                        ),
                    ],
                ),
                reference: "AS 29.3.94",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        Basic::new(
                                            Head::Pow,
                                            vec![Basic::new(Head::Add, vec![w.k.clone(), w.x.clone()]), int(-1)],
                                        ),
                                        Basic::new(
                                            Head::Add,
                                            vec![Basic::new(Head::Mul, vec![int(-1), w.k.clone()]), w.x.clone()],
                                        ),
                                    ],
                                ),
                                Basic::new(Head::Mul, vec![frac(1, 2), w.n.clone()]),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![
                                w.n.clone(),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        w.a.clone(),
                                        Basic::new(
                                            Head::Pow,
                                            vec![
                                                Basic::new(
                                                    Head::Add,
                                                    vec![
                                                        Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                                        Basic::new(
                                                            Head::Mul,
                                                            vec![
                                                                int(-1),
                                                                Basic::new(Head::Pow, vec![w.k.clone(), int(2)]),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                frac(1, 2),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: w.k.clone(),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.a.clone(), w.n.clone()]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(
                                            Head::Pow,
                                            vec![
                                                Basic::new(
                                                    Head::Add,
                                                    vec![
                                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                        Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                                    ],
                                                ),
                                                frac(1, 2),
                                            ],
                                        ),
                                        Basic::new(Head::Mul, vec![int(-1), w.c.clone()]),
                                    ],
                                ),
                                Basic::new(Head::Mul, vec![int(-1), w.n.clone()]),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    w.k.clone(),
                                    Basic::new(
                                        Head::Pow,
                                        vec![
                                            Basic::new(
                                                Head::Add,
                                                vec![
                                                    Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                    Basic::new(Head::Pow, vec![w.c.clone(), int(2)]),
                                                ],
                                            ),
                                            frac(1, 2),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.n.clone(), int(-1)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.c.clone(), int(0)]),
                    ],
                ),
                reference: "AS 29.3.96",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), frac(-1, 2)]),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Add,
                                vec![
                                    Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()]),
                                    Basic::new(
                                        Head::Mul,
                                        vec![
                                            frac(-1, 4),
                                            Basic::new(Head::Pow, vec![w.k.clone(), int(2)]),
                                            Basic::new(Head::Pow, vec![w.x.clone(), int(-1)]),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), frac(1, 2)]),
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Mul, vec![int(-1), w.c.clone()]), frac(-1, 2)],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    w.k.clone(),
                                    Basic::new(
                                        Head::Pow,
                                        vec![Basic::new(Head::Mul, vec![int(-1), w.c.clone()]), frac(1, 2)],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.c.clone(), int(0)]),
                    ],
                ),
                reference: "AS 29.3.82",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Erfc),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    frac(1, 2),
                                    w.k.clone(),
                                    Basic::new(Head::Pow, vec![w.x.clone(), frac(-1, 2)]),
                                ],
                            )],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        int(-1),
                        Basic::new(Head::Pow, vec![w.c.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    w.k.clone(),
                                    Basic::new(
                                        Head::Pow,
                                        vec![Basic::new(Head::Mul, vec![int(-1), w.c.clone()]), frac(1, 2)],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Lt), vec![w.c.clone(), int(0)]),
                    ],
                ),
                reference: "AS 29.3.83",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Erfc),
                            vec![Basic::new(
                                Head::Add,
                                vec![
                                    Basic::new(
                                        Head::Mul,
                                        vec![w.a.clone(), Basic::new(Head::Pow, vec![w.x.clone(), frac(1, 2)])],
                                    ),
                                    Basic::new(
                                        Head::Mul,
                                        vec![
                                            frac(1, 2),
                                            w.k.clone(),
                                            Basic::new(Head::Pow, vec![w.x.clone(), frac(-1, 2)]),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(Head::Mul, vec![w.c.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        w.a.clone(),
                                        Basic::new(
                                            Head::Pow,
                                            vec![
                                                Basic::new(
                                                    Head::Add,
                                                    vec![
                                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                        Basic::new(Head::Mul, vec![int(-1), w.c.clone()]),
                                                    ],
                                                ),
                                                frac(1, 2),
                                            ],
                                        ),
                                    ],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(Head::Mul, vec![int(-1), w.c.clone()]),
                                    ],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    w.k.clone(),
                                    Basic::new(
                                        Head::Add,
                                        vec![
                                            w.a.clone(),
                                            Basic::new(
                                                Head::Pow,
                                                vec![
                                                    Basic::new(
                                                        Head::Add,
                                                        vec![
                                                            Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                                            Basic::new(Head::Mul, vec![int(-1), w.c.clone()]),
                                                        ],
                                                    ),
                                                    frac(1, 2),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.k.clone(), int(0)]),
                        Basic::new(
                            Head::Relational(RelOp::Lt),
                            vec![w.c.clone(), Basic::new(Head::Pow, vec![w.a.clone(), int(2)])],
                        ),
                    ],
                ),
                reference: "AS 29.3.89",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-1)]),
                        Basic::new(Head::Function(Function::Atan), vec![w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::atom(Head::Catalan),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.531.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(1), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(2), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                ),
                                frac(-1, 2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Atan),
                            vec![Basic::new(
                                Head::Pow,
                                vec![
                                    Basic::new(
                                        Head::Add,
                                        vec![int(2), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                    ),
                                    frac(1, 2),
                                ],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: int(1),
                value: Basic::new(
                    Head::Mul,
                    vec![frac(5, 96), Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(2)])],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "Ahmed 2002",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-2)]),
                        Basic::new(
                            Head::Pow,
                            vec![Basic::new(Head::Function(Function::Atan), vec![w.x.clone()]), int(2)],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(Head::Function(Function::Log), vec![int(2)]),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 4.535.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-2)]),
                        Basic::new(
                            Head::Function(Function::Atan),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Atan),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        Basic::new(Head::Add, vec![w.a.clone(), w.b.clone()]),
                                        Basic::new(
                                            Head::Function(Function::Log),
                                            vec![Basic::new(Head::Add, vec![w.a.clone(), w.b.clone()])],
                                        ),
                                    ],
                                ),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        int(-1),
                                        w.a.clone(),
                                        Basic::new(Head::Function(Function::Log), vec![w.a.clone()]),
                                    ],
                                ),
                                Basic::new(
                                    Head::Mul,
                                    vec![
                                        int(-1),
                                        w.b.clone(),
                                        Basic::new(Head::Function(Function::Log), vec![w.b.clone()]),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 4.536.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![int(1), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Atan),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 2),
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Function(Function::Log),
                            vec![Basic::new(Head::Add, vec![int(1), w.a.clone()])],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                reference: "GR 4.535.2",
            },
            TableEntry {
                pattern: Basic::new(Head::Function(Function::BesselJ), vec![w.a.clone(), w.x.clone()]),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: int(1),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(-1)]),
                reference: "GR 6.511.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-1)]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(Head::Function(Function::BesselJ), vec![w.a.clone(), w.x.clone()]),
                                int(2),
                            ],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![frac(1, 2), Basic::new(Head::Pow, vec![w.a.clone(), int(-1)])],
                ),
                condition: Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                reference: "GR 6.574.2 (mu = nu)",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![w.x.clone(), int(-1)]),
                        Basic::new(Head::Function(Function::BesselJ), vec![w.a.clone(), w.x.clone()]),
                        Basic::new(Head::Function(Function::BesselJ), vec![w.b.clone(), w.x.clone()]),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        int(2),
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), int(-1)]),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(
                                    Head::Add,
                                    vec![
                                        Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                        Basic::new(
                                            Head::Mul,
                                            vec![int(-1), Basic::new(Head::Pow, vec![w.b.clone(), int(2)])],
                                        ),
                                    ],
                                ),
                                int(-1),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    Basic::atom(Head::Pi),
                                    Basic::new(
                                        Head::Add,
                                        vec![
                                            Basic::new(Head::Mul, vec![frac(1, 2), w.a.clone()]),
                                            Basic::new(Head::Mul, vec![frac(-1, 2), w.b.clone()]),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Ne), vec![w.a.clone(), w.b.clone()]),
                        Basic::new(
                            Head::Relational(RelOp::Gt),
                            vec![Basic::new(Head::Add, vec![w.a.clone(), w.b.clone()]), int(0)],
                        ),
                    ],
                ),
                reference: "GR 6.574.2",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(0), Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(1), Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(Head::Pow, vec![w.b.clone(), int(-1)]),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), w.a.clone()]),
                    ],
                ),
                reference: "GR 6.512.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(0), Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(1), Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: int(0),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), w.b.clone()]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.512.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(0), Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                                Basic::new(
                                    Head::Mul,
                                    vec![int(-1), Basic::new(Head::Pow, vec![w.a.clone(), int(2)])],
                                ),
                            ],
                        ),
                        frac(-1, 2),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), w.a.clone()]),
                    ],
                ),
                reference: "GR 6.671.7",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(0), Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Sin),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: int(0),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), w.b.clone()]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.671.7",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(0), Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(
                            Head::Add,
                            vec![
                                Basic::new(Head::Pow, vec![w.a.clone(), int(2)]),
                                Basic::new(
                                    Head::Mul,
                                    vec![int(-1), Basic::new(Head::Pow, vec![w.b.clone(), int(2)])],
                                ),
                            ],
                        ),
                        frac(-1, 2),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), w.b.clone()]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.671.8",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::BesselJ),
                            vec![int(0), Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Cos),
                            vec![Basic::new(Head::Mul, vec![w.b.clone(), w.x.clone()])],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: int(0),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), w.a.clone()]),
                    ],
                ),
                reference: "GR 6.671.8",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![Basic::new(Head::Function(Function::Erfc), vec![w.x.clone()]), int(2)],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), frac(-1, 2)]),
                        Basic::new(
                            Head::Add,
                            vec![
                                int(2),
                                Basic::new(
                                    Head::Mul,
                                    vec![int(-1), Basic::new(Head::Pow, vec![int(2), frac(1, 2)])],
                                ),
                            ],
                        ),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 6.283.?",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Function(Function::Erf), vec![w.x.clone()]),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![int(-1), Basic::new(Head::Pow, vec![w.x.clone(), int(2)])],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        frac(1, 4),
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), frac(1, 2)]),
                    ],
                ),
                condition: Basic::atom(Head::BooleanTrue),
                reference: "GR 6.285.?",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(
                            Head::Function(Function::Erf),
                            vec![Basic::new(Head::Mul, vec![w.a.clone(), w.x.clone()])],
                        ),
                        Basic::new(
                            Head::Function(Function::Exp),
                            vec![Basic::new(
                                Head::Mul,
                                vec![
                                    int(-1),
                                    Basic::new(Head::Pow, vec![w.b.clone(), int(2)]),
                                    Basic::new(Head::Pow, vec![w.x.clone(), int(2)]),
                                ],
                            )],
                        ),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Infinity),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::new(Head::Pow, vec![Basic::atom(Head::Pi), frac(-1, 2)]),
                        Basic::new(Head::Pow, vec![w.b.clone(), int(-1)]),
                        Basic::new(
                            Head::Function(Function::Atan),
                            vec![Basic::new(
                                Head::Mul,
                                vec![w.a.clone(), Basic::new(Head::Pow, vec![w.b.clone(), int(-1)])],
                            )],
                        ),
                    ],
                ),
                condition: Basic::new(
                    Head::And,
                    vec![
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.a.clone(), int(0)]),
                        Basic::new(Head::Relational(RelOp::Gt), vec![w.b.clone(), int(0)]),
                    ],
                ),
                reference: "GR 6.285.1",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]),
                        Basic::new(Head::Mul, vec![int(2), w.n.clone()]),
                    ],
                ),
                lower: int(0),
                upper: Basic::new(Head::Mul, vec![frac(1, 2), Basic::atom(Head::Pi)]),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Pow,
                            vec![
                                int(2),
                                Basic::new(
                                    Head::Add,
                                    vec![int(-1), Basic::new(Head::Mul, vec![int(-2), w.n.clone()])],
                                ),
                            ],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(Head::Function(Function::Factorial), vec![w.n.clone()]),
                                int(-2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Factorial),
                            vec![Basic::new(Head::Mul, vec![int(2), w.n.clone()])],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                reference: "GR 3.621.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(Head::Function(Function::Sin), vec![w.x.clone()]),
                        Basic::new(Head::Mul, vec![int(2), w.n.clone()]),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Pow,
                            vec![int(2), Basic::new(Head::Mul, vec![int(-2), w.n.clone()])],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(Head::Function(Function::Factorial), vec![w.n.clone()]),
                                int(-2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Factorial),
                            vec![Basic::new(Head::Mul, vec![int(2), w.n.clone()])],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                reference: "GR 3.621.3",
            },
            TableEntry {
                pattern: Basic::new(
                    Head::Pow,
                    vec![
                        Basic::new(Head::Function(Function::Cos), vec![w.x.clone()]),
                        Basic::new(Head::Mul, vec![int(2), w.n.clone()]),
                    ],
                ),
                lower: int(0),
                upper: Basic::atom(Head::Pi),
                value: Basic::new(
                    Head::Mul,
                    vec![
                        Basic::atom(Head::Pi),
                        Basic::new(
                            Head::Pow,
                            vec![int(2), Basic::new(Head::Mul, vec![int(-2), w.n.clone()])],
                        ),
                        Basic::new(
                            Head::Pow,
                            vec![
                                Basic::new(Head::Function(Function::Factorial), vec![w.n.clone()]),
                                int(-2),
                            ],
                        ),
                        Basic::new(
                            Head::Function(Function::Factorial),
                            vec![Basic::new(Head::Mul, vec![int(2), w.n.clone()])],
                        ),
                    ],
                ),
                condition: Basic::new(Head::Contains, vec![w.n.clone(), Basic::atom(Head::Naturals0)]),
                reference: "GR 3.621.3",
            },
        ]
    })
}

fn has_wild(e: &Basic) -> bool {
    e.preorder().iter().any(|n| matches!(n.head(), Head::Wild(_)))
}

fn replaced(e: &Basic, rule: &[(Basic, Basic)]) -> PolyResult<Basic> {
    let map: HashMap<Basic, Basic> = rule.iter().cloned().collect();
    xreplace_evaluated(e, &map)
}

/// The integrand as given and in the rewritten forms the patterns may
/// need (`_forms`).
fn forms(g: &Basic) -> PolyResult<Vec<Basic>> {
    use crate::simplify::powsimp::{powsimp, Combine};
    let count = &crate::core::function::count_ops;
    let x: &Basic = &wilds().x;
    let mut out: Vec<Basic> = vec![g.clone()];
    let forced: Basic = powsimp(g, false, Combine::All, true, count)?;
    let combined: Basic = crate::core::function::replace_where(
        &forced,
        &|node: &Basic| node.is_pow() && node.has(x),
        &|node: &Basic| {
            crate::core::power::pow(&crate::core::function::expand_method(&node.args()[0])?, &node.args()[1])
        },
    )?;
    let candidates: Vec<Basic> = vec![
        powsimp(g, false, Combine::All, false, count)?,
        crate::core::function::expand_method(g)?,
        crate::simplify::trigsimp::trigsimp(g, crate::simplify::trigsimp::Method::Matching)?,
        combined,
    ];
    for c in candidates {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    Ok(out)
}

/// The first entry matching `Integral(f, (x, a, b))` whose condition the
/// assumptions do not refute, with the values of its parameters and the
/// condition left undecided (`lookup`).
pub fn lookup(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<(&'static TableEntry, Vec<(Basic, Basic)>, Basic)>> {
    use crate::core::matching::match_pattern;
    let w: &Wilds = wilds();
    let g: Basic = replaced(f, &[(x.clone(), w.x.clone())])?;
    let all_forms: Vec<Basic> = forms(&g)?;
    for entry in table() {
        if entry.upper != *b {
            continue;
        }
        let mut range_match: Vec<(Basic, Basic)> = Vec::new();
        if entry.lower != *a {
            if !has_wild(&entry.lower) || a.has(x) {
                continue;
            }
            let Some(found) = match_pattern(a, &entry.lower, &[], false)? else {
                continue;
            };
            range_match = found
                .into_iter()
                .filter(|(k, _)| matches!(k.head(), Head::Wild(_)))
                .collect();
        }
        'forms: for candidate in &all_forms {
            let Some(matched) = match_pattern(candidate, &entry.pattern, &[], false)? else {
                continue;
            };
            if range_match
                .iter()
                .any(|(key, value)| matched.iter().any(|(k, v)| k == key && v != value))
            {
                continue;
            }
            let mut all: Vec<(Basic, Basic)> = matched.clone();
            for (k, v) in &range_match {
                match all.iter_mut().find(|(key, _)| key == k) {
                    Some(entry) => entry.1 = v.clone(),
                    None => all.push((k.clone(), v.clone())),
                }
            }
            let mut values: Vec<(Basic, Basic)> = Vec::new();
            for (key, value) in all {
                if !matches!(key.head(), Head::Wild(_)) {
                    continue;
                }
                if value.has(&w.x) {
                    continue 'forms;
                }
                values.push((key, value));
            }
            let condition: Basic = match replaced(&entry.condition, &values) {
                Ok(c) => c,
                Err(PolyError::Type(_)) => continue,
                Err(e) => return Err(e),
            };
            let decided: Option<Basic> = match decide(&condition, assumptions) {
                Ok(d) => d,
                Err(PolyError::Type(_)) => continue,
                Err(e) => return Err(e),
            };
            let Some(condition) = decided else { continue };
            return Ok(Some((entry, values, condition)));
        }
    }
    Ok(None)
}

/// `Integral(f, (x, a, b))` from the table with the condition the
/// assumptions do not settle (`table_integral`).
pub fn table_integral(
    f: &Basic,
    x: &Basic,
    a: &Basic,
    b: &Basic,
    assumptions: &[Basic],
) -> PolyResult<Option<ConditionalValue>> {
    let Some((entry, values, condition)) = lookup(f, x, a, b, assumptions)? else {
        return Ok(None);
    };
    Ok(Some(ConditionalValue::new(replaced(&entry.value, &values)?, condition)))
}
