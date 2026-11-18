//! Quantified formulas on the core, ported from
//! `sympy_extras.assumptions.quantifiers`: the Boolean heads
//! [`Head::ForAll`] and [`Head::Exists`] (`ForAll(Tuple(variables),
//! formula)`, unevaluated like every node), the evaluating [`forall`] and
//! [`exists`] (the formula itself when it is a truth value or contains none
//! of the variables, the variables which do not occur in it dropped) and
//! [`prenex`], the prenex normal form. `Quantifier.simplify` (SymPy's
//! `simplify` of the formula) is not ported.

#![allow(clippy::mutable_key_type)]

use std::collections::BTreeSet;

use crate::core::basic::{Basic, Head};
use crate::core::logic::{and_, implies, not_, or_, xor};
use crate::error::{value_error, PolyError, PolyResult};

/// `'forall'` or `'exists'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantifierKind {
    ForAll,
    Exists,
}

impl QuantifierKind {
    pub fn name(self) -> &'static str {
        match self {
            QuantifierKind::ForAll => "forall",
            QuantifierKind::Exists => "exists",
        }
    }

    fn dual(self) -> QuantifierKind {
        match self {
            QuantifierKind::ForAll => QuantifierKind::Exists,
            QuantifierKind::Exists => QuantifierKind::ForAll,
        }
    }

    fn head(self) -> Head {
        match self {
            QuantifierKind::ForAll => Head::ForAll,
            QuantifierKind::Exists => Head::Exists,
        }
    }
}

/// The quantifier prefix of a prenex formula, outermost first.
pub type Prefix = Vec<(QuantifierKind, Basic)>;

/// Whether `e` is a `ForAll` or an `Exists`.
pub fn is_quantifier(e: &Basic) -> bool {
    matches!(e.head(), Head::ForAll | Head::Exists)
}

fn kind_of(e: &Basic) -> QuantifierKind {
    if matches!(e.head(), Head::ForAll) {
        QuantifierKind::ForAll
    } else {
        QuantifierKind::Exists
    }
}

/// The quantified variables of a quantifier node.
pub fn variables(e: &Basic) -> &[Basic] {
    e.args()[0].args()
}

/// The quantified formula of a quantifier node.
pub fn formula(e: &Basic) -> &Basic {
    &e.args()[1]
}

/// The variables checked (`_variables`): symbols, no repetition, at least
/// one.
pub fn checked_variables(vs: &[Basic]) -> PolyResult<Vec<Basic>> {
    for v in vs {
        if !v.is_symbol() {
            return Err(PolyError::Type(format!(
                "quantified variables must be symbols, got {}",
                v
            )));
        }
    }
    let distinct: BTreeSet<&Basic> = vs.iter().collect();
    if distinct.len() != vs.len() {
        let shown: Vec<String> = vs.iter().map(|v| v.to_string()).collect();
        let text: String = if shown.len() == 1 {
            format!("({},)", shown[0])
        } else {
            format!("({})", shown.join(", "))
        };
        return Err(value_error(format!("repeated quantified variable in {}", text)));
    }
    if vs.is_empty() {
        return Err(value_error("a quantifier needs at least one variable".to_string()));
    }
    Ok(vs.to_vec())
}

/// The unevaluated node `ForAll(variables, formula)` or `Exists(...)`.
pub fn quantifier_node(kind: QuantifierKind, variables: Vec<Basic>, formula: Basic) -> Basic {
    Basic::new(kind.head(), vec![Basic::new(Head::Tuple, variables), formula])
}

/// The free symbols: those of the formula but the variables.
pub fn quantifier_free_symbols(e: &Basic) -> BTreeSet<Basic> {
    let bound: &[Basic] = variables(e);
    formula(e)
        .free_symbols()
        .into_iter()
        .filter(|s| !bound.contains(s))
        .collect()
}

fn quantify(kind: QuantifierKind, variables: &[Basic], formula: &Basic) -> Basic {
    if matches!(formula.head(), Head::BooleanTrue | Head::BooleanFalse) {
        return formula.clone();
    }
    let symbols: BTreeSet<Basic> = formula.free_symbols();
    let occurring: Vec<Basic> = variables.iter().filter(|v| symbols.contains(v)).cloned().collect();
    if occurring.is_empty() {
        return formula.clone();
    }
    quantifier_node(kind, occurring, formula.clone())
}

/// `ForAll(variables, formula)` evaluated (`forall`).
pub fn forall(vs: &[Basic], formula: &Basic) -> PolyResult<Basic> {
    Ok(quantify(QuantifierKind::ForAll, &checked_variables(vs)?, formula))
}

/// `Exists(variables, formula)` evaluated (`exists`).
pub fn exists(vs: &[Basic], formula: &Basic) -> PolyResult<Basic> {
    Ok(quantify(QuantifierKind::Exists, &checked_variables(vs)?, formula))
}

/// The node rebuilt from new arguments, evaluated (`_evaluated`).
pub fn rebuild_quantifier(head: &Head, args: &[Basic]) -> PolyResult<Basic> {
    if !matches!(args[0].head(), Head::Tuple) {
        return Err(PolyError::Type("the variables of a quantifier are a Tuple".to_string()));
    }
    let kind: QuantifierKind = if matches!(head, Head::ForAll) {
        QuantifierKind::ForAll
    } else {
        QuantifierKind::Exists
    };
    Ok(quantify(kind, &checked_variables(args[0].args())?, &args[1]))
}

/// The substitution (`_subs`): nothing for a bound variable, the formula
/// substituted otherwise (unevaluated).
pub fn quantifier_subs(
    e: &Basic,
    old: &Basic,
    new: &Basic,
    inner: &dyn Fn(&Basic) -> PolyResult<Basic>,
) -> PolyResult<Basic> {
    if e == old {
        return Ok(new.clone());
    }
    if variables(e).contains(old) {
        return Ok(e.clone());
    }
    Ok(quantifier_node(kind_of(e), variables(e).to_vec(), inner(formula(e))?))
}

/// `ForAll(x, f)`, `ForAll((x, y), f)`.
pub fn print_quantifier(e: &Basic, doprint: &dyn Fn(&Basic) -> String) -> String {
    let vs: &[Basic] = variables(e);
    let shown: String = if vs.len() == 1 {
        doprint(&vs[0])
    } else {
        format!("({})", vs.iter().map(doprint).collect::<Vec<String>>().join(", "))
    };
    let name: &str = if matches!(e.head(), Head::ForAll) {
        "ForAll"
    } else {
        "Exists"
    };
    format!("{}({}, {})", name, shown, doprint(formula(e)))
}

/// The prenex normal form `(prefix, matrix)` (`prenex`): quantifiers under
/// `Not` flipped, those under the connectives moved in front, the bound
/// variables renamed (to dummies) when they clash.
pub fn prenex(f: &Basic) -> PolyResult<(Prefix, Basic)> {
    prenex_rec(f, &BTreeSet::new())
}

fn prenex_rec(f: &Basic, taken: &BTreeSet<Basic>) -> PolyResult<(Prefix, Basic)> {
    let args: Vec<Basic> = f.args().to_vec();
    match f.head() {
        Head::ForAll | Head::Exists => {
            let mut inner_taken: BTreeSet<Basic> = taken.clone();
            inner_taken.extend(variables(f).iter().cloned());
            let (mut inner_prefix, mut matrix) = prenex_rec(formula(f), &inner_taken)?;
            let mut prefix: Prefix = Vec::new();
            for v in variables(f) {
                if !matrix.free_symbols().contains(v) {
                    continue;
                }
                let mut v: Basic = v.clone();
                if taken.contains(&v) {
                    let data = match v.head() {
                        Head::Symbol(d) | Head::Dummy(d, _) => d.clone(),
                        Head::Wild(w) => std::sync::Arc::new(w.symbol.clone()),
                        _ => unreachable!("a symbol"),
                    };
                    let fresh: Basic = crate::core::Dummy::new(&data.name, &data.given).0;
                    matrix = crate::core::subs::subs(&matrix, &v, &fresh)?;
                    inner_prefix = inner_prefix
                        .into_iter()
                        .map(|(k, u)| (k, if u == v { fresh.clone() } else { u }))
                        .collect();
                    v = fresh;
                }
                prefix.push((kind_of(f), v));
            }
            prefix.extend(inner_prefix);
            Ok((prefix, matrix))
        }
        Head::Not => {
            let (prefix, matrix) = prenex_rec(&args[0], taken)?;
            Ok((prefix.into_iter().map(|(k, v)| (k.dual(), v)).collect(), not_(&matrix)))
        }
        Head::And | Head::Or => {
            let mut prefix: Prefix = Vec::new();
            let mut matrices: Vec<Basic> = Vec::new();
            let mut used: BTreeSet<Basic> = taken.clone();
            for a in &args {
                used.extend(a.free_symbols());
            }
            for arg in &args {
                let (p, m) = prenex_rec(arg, &used)?;
                used.extend(p.iter().map(|(_, v)| v.clone()));
                prefix.extend(p);
                matrices.push(m);
            }
            let matrix: Basic = if matches!(f.head(), Head::And) {
                and_(&matrices)
            } else {
                or_(&matrices)
            };
            Ok((prefix, matrix))
        }
        Head::Implies => prenex_rec(&or_(&[not_(&args[0]), args[1].clone()]), taken),
        Head::Equivalent => {
            let n: usize = args.len();
            let parts: Vec<Basic> = (0..n).map(|i| implies(&args[i], &args[(i + 1) % n])).collect();
            prenex_rec(&and_(&parts), taken)
        }
        Head::Xor => {
            let first: &Basic = &args[0];
            let rest: &[Basic] = &args[1..];
            let second: Basic = if rest.len() > 1 { xor(rest) } else { rest[0].clone() };
            let either: Basic = or_(&[
                and_(&[first.clone(), not_(&second)]),
                and_(&[not_(first), second.clone()]),
            ]);
            prenex_rec(&either, taken)
        }
        Head::Ite => {
            let (c, a, b) = (&args[0], &args[1], &args[2]);
            prenex_rec(
                &or_(&[and_(&[c.clone(), a.clone()]), and_(&[not_(c), b.clone()])]),
                taken,
            )
        }
        _ => Ok((Vec::new(), f.clone())),
    }
}
