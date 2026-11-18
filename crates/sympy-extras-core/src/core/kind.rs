//! Kinds, ported from `sympy_extras.core.kind`: a coarse classification of
//! the nodes, finer than the class. `NumberKind` for scalar expressions,
//! `BooleanKind` for relations and truth values, `UndefinedKind` for what
//! is not known to be one of those, `TupleKind(...)` for a tuple. The
//! `kind` properties of the Python classes are [`kind_of`].

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};

/// A kind (`Kind`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Kind {
    Number,
    Boolean,
    Undefined,
    Tuple(Vec<Kind>),
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kind::Number => write!(f, "NumberKind"),
            Kind::Boolean => write!(f, "BooleanKind"),
            Kind::Undefined => write!(f, "UndefinedKind"),
            // without parameters, the name alone (Python's Kind.__str__)
            Kind::Tuple(kinds) if kinds.is_empty() => write!(f, "TupleKind"),
            Kind::Tuple(kinds) => {
                let inner: Vec<String> = kinds.iter().map(|k| k.to_string()).collect();
                // SymPy prints the tuple of element kinds: TupleKind(NumberKind,)
                let comma: &str = if kinds.len() == 1 { "," } else { "" };
                write!(f, "TupleKind({}{})", inner.join(", "), comma)
            }
        }
    }
}

fn is_boolean_head(h: &Head) -> bool {
    matches!(
        h,
        Head::BooleanTrue
            | Head::BooleanFalse
            | Head::And
            | Head::Or
            | Head::Not
            | Head::Xor
            | Head::Implies
            | Head::Equivalent
            | Head::Ite
            | Head::Relational(_)
            | Head::Contains
            | Head::ForAll
            | Head::Exists
            | Head::Predicate(_)
            | Head::AppliedPredicate
    )
}

/// The nodes of the Python classes which are no `Expr` (sets and the
/// pairs of `Piecewise`): `Basic.kind`, `UndefinedKind`.
fn is_basic_only(h: &Head) -> bool {
    matches!(
        h,
        Head::EmptySet
            | Head::UniversalSet
            | Head::Reals
            | Head::Interval
            | Head::FiniteSet
            | Head::Union
            | Head::Intersection
            | Head::Complement
            | Head::ProductSet
            | Head::PointSet
            | Head::ConditionSet
            | Head::ExprCondPair
            | Head::SetOfSets
            | Head::PowerSet
            | Head::Naturals
            | Head::Naturals0
            | Head::Integers
            | Head::Rationals
            | Head::Complexes
            | Head::ImageSet
    )
}

/// The kind of a node (the `kind` property).
pub fn kind_of(e: &Basic) -> Kind {
    match e.head() {
        Head::Tuple => Kind::Tuple(e.args().iter().map(kind_of).collect()),
        Head::Function(Function::Exp) => Kind::Number,
        Head::Function(_) | Head::Undefined(_) | Head::Hyper | Head::MeijerG => Kind::Undefined,
        Head::Derivative | Head::Subs => kind_of(&e.args()[0]),
        Head::AccumBounds => Kind::Number,
        h if is_boolean_head(h) => Kind::Boolean,
        h if is_basic_only(h) => Kind::Undefined,
        _ => {
            // Expr.kind
            if e.args().is_empty() {
                let commutative: bool = !matches!(e.head(), Head::Symbol(_) | Head::Dummy(..) | Head::Wild(_))
                    || crate::core::expr::is_true(e, Predicate::commutative);
                return if commutative { Kind::Number } else { Kind::Undefined };
            }
            if e.args().iter().all(|a| kind_of(a) == Kind::Number) {
                Kind::Number
            } else {
                Kind::Undefined
            }
        }
    }
}
