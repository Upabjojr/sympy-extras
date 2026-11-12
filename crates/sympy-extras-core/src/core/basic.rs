//! Port of the structure of `sympy_extras/core/basic.py`: the node of the
//! expression tree.
//!
//! A node is a [`Head`] (the class, with the data of an atom) and a vector
//! of arguments. Nodes are immutable and shared ([`Basic`] is an `Arc`); two
//! nodes are equal exactly when their heads and their arguments are equal,
//! and the hash is derived from the same data. Nothing is evaluated when a
//! node is built: `Add(x, x)` keeps its two arguments.
//!
//! The traversals and rebuilds of Python's `Basic` (`args`, `func`, `atoms`,
//! `free_symbols`, `has`, `xreplace`, `count`, the `compare` order) are
//! here; `subs`, `doit`, `match`, `replace` need the evaluating functions
//! and come with them.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use crate::numbers::*;

use super::assumptions::Facts;
use super::sorting::{compare_key, CompareItem, CompareKey};
use super::symbol::{SymbolData, WildData};

/// The class of a node, with the data of an atom.
#[derive(Debug, Clone)]
pub enum Head {
    /// `Integer` and `Rational` (Python's two classes; `integer_class` says
    /// which one built the node, the value decides everything else).
    Rational {
        value: Rat,
        integer_class: bool,
    },
    Infinity,
    NegativeInfinity,
    ComplexInfinity,
    NaN,
    Pi,
    Exp1,
    EulerGamma,
    Catalan,
    GoldenRatio,
    TribonacciConstant,
    ImaginaryUnit,
    Symbol(Arc<SymbolData>),
    /// A dummy with its index: equal only to itself.
    Dummy(Arc<SymbolData>, u64),
    Wild(Arc<WildData>),
    Add,
    Mul,
    Pow,
    /// An applied function of the core (`Exp(x)`, `Log(x)`), unevaluated.
    Function(Function),
    /// An undefined function applied to the arguments (`f(x)`,
    /// `AppliedUndef`).
    Undefined(Arc<UndefinedFunction>),
    /// `Tuple(a, b, ...)` (a container, not an expression).
    Tuple,
    /// `Derivative(expr, Tuple(x, n), ...)`, unevaluated.
    Derivative,
    /// `Subs(expr, Tuple(variables), Tuple(points))`, unevaluated.
    Subs,
    /// `IndexedRoot(f, t, k)`: the `k`-th distinct real root of `f` in the
    /// bound variable `t` (`polys.cad.indexedroot`).
    IndexedRoot,
    /// `true` and `false` (`BooleanTrue`, `BooleanFalse`).
    BooleanTrue,
    BooleanFalse,
    /// The connectives, unevaluated (`core.logic`).
    And,
    Or,
    Not,
    Xor,
    Implies,
    Equivalent,
    Ite,
    /// A relation between two expressions (`Eq(a, b)`, `Lt(a, b)`, ...).
    Relational(RelOp),
    /// The sets of `sets.sets` (`EmptySet`, `UniversalSet`, `Reals` are
    /// atoms), and the boolean `Contains(x, s)`.
    EmptySet,
    UniversalSet,
    Reals,
    Interval,
    FiniteSet,
    Union,
    Intersection,
    Complement,
    Contains,
    ProductSet,
    PointSet,
    ConditionSet,
    /// `AccumBounds(min, max)` (`core.accumbounds`).
    AccumBounds,
}

/// The relations of `core.relational`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl RelOp {
    /// The name of the Python class.
    pub fn class_name(self) -> &'static str {
        match self {
            RelOp::Eq => "Eq",
            RelOp::Ne => "Ne",
            RelOp::Lt => "Lt",
            RelOp::Le => "Le",
            RelOp::Gt => "Gt",
            RelOp::Ge => "Ge",
        }
    }

    /// SymPy's class name (`Equality`, `StrictLessThan`, ...).
    pub fn sympy_name(self) -> &'static str {
        match self {
            RelOp::Eq => "Equality",
            RelOp::Ne => "Unequality",
            RelOp::Lt => "StrictLessThan",
            RelOp::Le => "LessThan",
            RelOp::Gt => "StrictGreaterThan",
            RelOp::Ge => "GreaterThan",
        }
    }

    /// The operator printed (`rel_op`).
    pub fn symbol(self) -> &'static str {
        match self {
            RelOp::Eq => "==",
            RelOp::Ne => "!=",
            RelOp::Lt => "<",
            RelOp::Le => "<=",
            RelOp::Gt => ">",
            RelOp::Ge => ">=",
        }
    }

    /// The relation with the sides exchanged (`_REVERSED`).
    pub fn reversed(self) -> RelOp {
        match self {
            RelOp::Lt => RelOp::Gt,
            RelOp::Gt => RelOp::Lt,
            RelOp::Le => RelOp::Ge,
            RelOp::Ge => RelOp::Le,
            other => other,
        }
    }

    /// The relation of the opposite meaning (`_NEGATED`).
    pub fn negated(self) -> RelOp {
        match self {
            RelOp::Eq => RelOp::Ne,
            RelOp::Ne => RelOp::Eq,
            RelOp::Lt => RelOp::Ge,
            RelOp::Ge => RelOp::Lt,
            RelOp::Le => RelOp::Gt,
            RelOp::Gt => RelOp::Le,
        }
    }
}

/// An undefined function (`UndefinedFunction`): its name, numbers of
/// arguments (any when `None`) and assumptions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UndefinedFunction {
    pub name: String,
    pub nargs: Option<Vec<usize>>,
    pub assumptions: Vec<(super::assumptions::Predicate, bool)>,
}

impl UndefinedFunction {
    pub fn new(name: &str, nargs: Option<Vec<usize>>, assumptions: &[(super::assumptions::Predicate, bool)]) -> Self {
        let mut given: Vec<(super::assumptions::Predicate, bool)> = assumptions.to_vec();
        // Python's tuple(sorted(assumptions.items())): by the names
        given.sort_by(|a, b| a.0.name().cmp(b.0.name()));
        UndefinedFunction {
            name: name.to_string(),
            nargs: nargs.map(|mut n| {
                n.sort();
                n.dedup();
                n
            }),
            assumptions: given,
        }
    }

    /// The function applied to `args` (`f(*args)`); `Value` for a wrong
    /// number of arguments (Python's `TypeError`).
    pub fn apply(self: &Arc<Self>, args: Vec<Basic>) -> crate::error::PolyResult<Basic> {
        if let Some(n) = &self.nargs {
            if !n.contains(&args.len()) {
                return Err(crate::error::PolyError::Value(format!(
                    "{} takes {:?} arguments, {} given",
                    self.name,
                    n,
                    args.len()
                )));
            }
        }
        Ok(Basic::new(Head::Undefined(self.clone()), args))
    }
}

/// The functions of the core (Python's `DefinedFunction` subclasses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Function {
    Exp,
    Log,
    Re,
    Im,
    Conjugate,
    Sign,
    Abs,
    Sin,
    Cos,
    Tan,
    Cot,
    Sec,
    Csc,
    Sinh,
    Cosh,
    Tanh,
    Coth,
    Sech,
    Csch,
    Asin,
    Acos,
    Atan,
    Acot,
    Asec,
    Acsc,
    Asinh,
    Acosh,
    Atanh,
    Acoth,
    Asech,
    Acsch,
    Arg,
    Atan2,
    Max,
    Min,
    Floor,
    Ceiling,
    Frac,
    DiracDelta,
    Heaviside,
    Factorial,
    Binomial,
    RisingFactorial,
    FallingFactorial,
    Subfactorial,
    Factorial2,
    Gamma,
    LogGamma,
    PolyGamma,
    DiGamma,
    TriGamma,
    Zeta,
    DirichletEta,
    PolyLog,
    LerchPhi,
    RiemannXi,
    Stieltjes,
}

impl Function {
    /// The printed name, SymPy's (`function_name`).
    pub fn name(self) -> &'static str {
        match self {
            Function::Exp => "exp",
            Function::Log => "log",
            Function::Re => "re",
            Function::Im => "im",
            Function::Conjugate => "conjugate",
            Function::Sign => "sign",
            Function::Abs => "Abs",
            Function::Sin => "sin",
            Function::Cos => "cos",
            Function::Tan => "tan",
            Function::Cot => "cot",
            Function::Sec => "sec",
            Function::Csc => "csc",
            Function::Sinh => "sinh",
            Function::Cosh => "cosh",
            Function::Tanh => "tanh",
            Function::Coth => "coth",
            Function::Sech => "sech",
            Function::Csch => "csch",
            Function::Asin => "asin",
            Function::Acos => "acos",
            Function::Atan => "atan",
            Function::Acot => "acot",
            Function::Asec => "asec",
            Function::Acsc => "acsc",
            Function::Asinh => "asinh",
            Function::Acosh => "acosh",
            Function::Atanh => "atanh",
            Function::Acoth => "acoth",
            Function::Asech => "asech",
            Function::Acsch => "acsch",
            Function::Arg => "arg",
            Function::Atan2 => "atan2",
            Function::Max => "Max",
            Function::Min => "Min",
            Function::Floor => "floor",
            Function::Ceiling => "ceiling",
            Function::Frac => "frac",
            Function::DiracDelta => "DiracDelta",
            Function::Heaviside => "Heaviside",
            Function::Factorial => "factorial",
            Function::Binomial => "binomial",
            Function::RisingFactorial => "RisingFactorial",
            Function::FallingFactorial => "FallingFactorial",
            Function::Subfactorial => "subfactorial",
            Function::Factorial2 => "factorial2",
            Function::Gamma => "gamma",
            Function::LogGamma => "loggamma",
            Function::PolyGamma => "polygamma",
            Function::DiGamma => "digamma",
            Function::TriGamma => "trigamma",
            Function::Zeta => "zeta",
            Function::DirichletEta => "dirichlet_eta",
            Function::PolyLog => "polylog",
            Function::LerchPhi => "lerchphi",
            Function::RiemannXi => "riemann_xi",
            Function::Stieltjes => "stieltjes",
        }
    }

    /// The name of the Python class.
    pub fn class_name(self) -> &'static str {
        match self {
            Function::Exp => "Exp",
            Function::Log => "Log",
            Function::Re => "Re",
            Function::Im => "Im",
            Function::Conjugate => "Conjugate",
            Function::Sign => "Sign",
            Function::Abs => "Abs",
            Function::Sin => "Sin",
            Function::Cos => "Cos",
            Function::Tan => "Tan",
            Function::Cot => "Cot",
            Function::Sec => "Sec",
            Function::Csc => "Csc",
            Function::Sinh => "Sinh",
            Function::Cosh => "Cosh",
            Function::Tanh => "Tanh",
            Function::Coth => "Coth",
            Function::Sech => "Sech",
            Function::Csch => "Csch",
            Function::Asin => "Asin",
            Function::Acos => "Acos",
            Function::Atan => "Atan",
            Function::Acot => "Acot",
            Function::Asec => "Asec",
            Function::Acsc => "Acsc",
            Function::Asinh => "Asinh",
            Function::Acosh => "Acosh",
            Function::Atanh => "Atanh",
            Function::Acoth => "Acoth",
            Function::Asech => "Asech",
            Function::Acsch => "Acsch",
            Function::Arg => "Arg",
            Function::Atan2 => "Atan2",
            Function::Max => "Max",
            Function::Min => "Min",
            Function::Floor => "Floor",
            Function::Ceiling => "Ceiling",
            Function::Frac => "Frac",
            Function::DiracDelta => "DiracDelta",
            Function::Heaviside => "Heaviside",
            Function::Factorial => "Factorial",
            Function::Binomial => "Binomial",
            Function::RisingFactorial => "RisingFactorial",
            Function::FallingFactorial => "FallingFactorial",
            Function::Subfactorial => "Subfactorial",
            Function::Factorial2 => "Factorial2",
            Function::Gamma => "Gamma",
            Function::LogGamma => "LogGamma",
            Function::PolyGamma => "PolyGamma",
            Function::DiGamma => "DiGamma",
            Function::TriGamma => "TriGamma",
            Function::Zeta => "Zeta",
            Function::DirichletEta => "DirichletEta",
            Function::PolyLog => "PolyLog",
            Function::LerchPhi => "LerchPhi",
            Function::RiemannXi => "RiemannXi",
            Function::Stieltjes => "Stieltjes",
        }
    }

    /// The rank of the function in the sort key (`_FUNCTION_RANKS`).
    pub fn rank(self) -> i64 {
        match self {
            Function::Exp => 10,
            Function::Log => 11,
            Function::Conjugate => 40,
            Function::Re => 41,
            Function::Im => 42,
            Function::Arg => 43,
            Function::Atan2 | Function::Max | Function::Min => 10000,
            Function::Floor | Function::Ceiling | Function::Frac => 10000,
            Function::DiracDelta | Function::Heaviside => 10000,
            Function::Factorial
            | Function::Binomial
            | Function::RisingFactorial
            | Function::FallingFactorial
            | Function::Subfactorial
            | Function::Factorial2 => 10000,
            Function::Zeta
            | Function::DirichletEta
            | Function::PolyLog
            | Function::LerchPhi
            | Function::RiemannXi
            | Function::Stieltjes
            | Function::Gamma
            | Function::LogGamma
            | Function::PolyGamma
            | Function::DiGamma
            | Function::TriGamma => 10000,
            Function::Sin => 20,
            Function::Cos => 21,
            Function::Tan => 22,
            Function::Cot => 23,
            Function::Sinh => 30,
            Function::Cosh => 31,
            Function::Tanh => 32,
            Function::Coth => 33,
            Function::Sign | Function::Abs | Function::Sec | Function::Csc | Function::Sech | Function::Csch => 10000,
            Function::Asin
            | Function::Acos
            | Function::Atan
            | Function::Acot
            | Function::Asec
            | Function::Acsc
            | Function::Asinh
            | Function::Acosh
            | Function::Atanh
            | Function::Acoth
            | Function::Asech
            | Function::Acsch => 10000,
        }
    }
}

impl Head {
    /// The name of the Python class (`type(e).__name__`).
    pub fn class_name(&self) -> &'static str {
        match self {
            Head::Rational {
                integer_class: true, ..
            } => "Integer",
            Head::Rational {
                integer_class: false, ..
            } => "Rational",
            Head::Infinity => "Infinity",
            Head::NegativeInfinity => "NegativeInfinity",
            Head::ComplexInfinity => "ComplexInfinity",
            Head::NaN => "NaN",
            Head::Pi => "Pi",
            Head::Exp1 => "Exp1",
            Head::EulerGamma => "EulerGammaType",
            Head::Catalan => "CatalanType",
            Head::GoldenRatio => "GoldenRatioType",
            Head::TribonacciConstant => "TribonacciConstantType",
            Head::ImaginaryUnit => "ImaginaryUnit",
            Head::Symbol(_) => "Symbol",
            Head::Dummy(_, _) => "Dummy",
            Head::Wild(_) => "Wild",
            Head::Add => "Add",
            Head::Mul => "Mul",
            Head::Pow => "Pow",
            Head::Function(f) => f.class_name(),
            Head::Undefined(_) => "AppliedUndef",
            Head::Tuple => "Tuple",
            Head::Derivative => "Derivative",
            Head::Subs => "Subs",
            Head::IndexedRoot => "IndexedRoot",
            Head::BooleanTrue => "BooleanTrue",
            Head::BooleanFalse => "BooleanFalse",
            Head::And => "And",
            Head::Or => "Or",
            Head::Not => "Not",
            Head::Xor => "Xor",
            Head::Implies => "Implies",
            Head::Equivalent => "Equivalent",
            Head::Ite => "ITE",
            Head::Relational(op) => op.class_name(),
            Head::EmptySet => "EmptySetClass",
            Head::UniversalSet => "UniversalSetClass",
            Head::Reals => "RealsClass",
            Head::Interval => "Interval",
            Head::FiniteSet => "FiniteSet",
            Head::Union => "Union",
            Head::Intersection => "Intersection",
            Head::Complement => "Complement",
            Head::Contains => "Contains",
            Head::ProductSet => "ProductSet",
            Head::PointSet => "PointSet",
            Head::ConditionSet => "ConditionSet",
            Head::AccumBounds => "AccumBounds",
        }
    }

    /// The name of the class in SymPy's `compare` order (`_compare_name`).
    fn compare_name(&self) -> &'static str {
        match self {
            Head::Rational { value, .. } => {
                if value.is_int() {
                    if value.is_zero() {
                        "Zero"
                    } else if value.is_one() {
                        "One"
                    } else if *value == -Rat::ONE {
                        "NegativeOne"
                    } else {
                        "Integer"
                    }
                } else if *value == rat(1, 2) {
                    "Half"
                } else {
                    "Rational"
                }
            }
            Head::EulerGamma => "EulerGamma",
            Head::Catalan => "Catalan",
            Head::GoldenRatio => "GoldenRatio",
            Head::TribonacciConstant => "TribonacciConstant",
            Head::Function(f) => f.name(),
            Head::Relational(op) => op.sympy_name(),
            Head::Undefined(u) => {
                // the compare name is the function's name, a String: leaked
                // once per distinct name so that the key stays 'static
                intern(&u.name)
            }
            other => other.class_name(),
        }
    }

    /// Whether the class is an atom (no arguments).
    pub fn is_atom(&self) -> bool {
        !matches!(
            self,
            Head::Add
                | Head::Mul
                | Head::Pow
                | Head::Function(_)
                | Head::Undefined(_)
                | Head::Tuple
                | Head::Derivative
                | Head::Subs
                | Head::IndexedRoot
                | Head::And
                | Head::Or
                | Head::Not
                | Head::Xor
                | Head::Implies
                | Head::Equivalent
                | Head::Ite
                | Head::Relational(_)
                | Head::Interval
                | Head::FiniteSet
                | Head::Union
                | Head::Intersection
                | Head::Complement
                | Head::Contains
                | Head::ProductSet
                | Head::PointSet
                | Head::ConditionSet
                | Head::AccumBounds
        )
    }

    /// Whether two nodes are equal for the same set of arguments in any
    /// order (`LatticeOp`, and the hashable content of `Xor` and
    /// `Equivalent`).
    pub fn args_are_a_set(&self) -> bool {
        matches!(self, Head::And | Head::Or | Head::Xor | Head::Equivalent)
    }

    fn same(&self, other: &Head) -> bool {
        match (self, other) {
            // a Rational equals the Integer of the same value
            (Head::Rational { value: a, .. }, Head::Rational { value: b, .. }) => a == b,
            (Head::Symbol(a), Head::Symbol(b)) => a.name == b.name && a.assumptions == b.assumptions,
            (Head::Dummy(a, i), Head::Dummy(b, j)) => i == j && a.name == b.name,
            (Head::Wild(a), Head::Wild(b)) => a.same(b),
            (Head::Function(a), Head::Function(b)) => a == b,
            (Head::Undefined(a), Head::Undefined(b)) => a == b,
            (Head::Relational(a), Head::Relational(b)) => a == b,
            (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b),
        }
    }

    fn hash_into<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Head::Relational(op) => op.hash(state),
            Head::Rational { value, .. } => {
                value.numerator().hash(state);
                value.denominator().hash(state);
            }
            Head::Symbol(s) => {
                s.name.hash(state);
                s.assumptions.hash(state);
            }
            Head::Dummy(s, index) => {
                s.name.hash(state);
                index.hash(state);
            }
            Head::Wild(w) => {
                w.symbol.name.hash(state);
                w.symbol.assumptions.hash(state);
            }
            Head::Function(f) => f.hash(state),
            Head::Undefined(u) => u.hash(state),
            _ => {}
        }
    }
}

/// A `'static` copy of a name (the names of the undefined functions in
/// the `compare` keys), interned.
fn intern(name: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .expect("the name table");
    if let Some(found) = names.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.insert(leaked);
    leaked
}

/// A node: its head and its arguments, with the hash and the `compare` key
/// computed once.
pub struct Node {
    head: Head,
    args: Vec<Basic>,
    hash: u64,
    compare_key: OnceLock<CompareKey>,
    facts: OnceLock<Facts>,
}

/// A shared immutable node of the expression tree (Python's `Basic`).
#[derive(Clone)]
pub struct Basic(Arc<Node>);

impl Basic {
    /// The node of `head` with `args`, unevaluated.
    pub fn new(head: Head, args: Vec<Basic>) -> Basic {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        head.hash_into(&mut hasher);
        if head.args_are_a_set() {
            // And, Or, Xor, Equivalent: equal for the same set of arguments
            let mut hashes: Vec<u64> = args.iter().map(|a| a.0.hash).collect();
            hashes.sort_unstable();
            hashes.dedup();
            hashes.hash(&mut hasher);
        } else {
            for a in &args {
                a.0.hash.hash(&mut hasher);
            }
        }
        let mut hash: u64 = hasher.finish();
        if matches!(head, Head::Reals) {
            // Reals equals Interval(-oo, oo, True, True), with its hash
            hash = crate::sets::sets::whole_line().0.hash;
        }
        Basic(Arc::new(Node {
            head,
            args,
            hash,
            compare_key: OnceLock::new(),
            facts: OnceLock::new(),
        }))
    }

    /// An atom.
    pub fn atom(head: Head) -> Basic {
        Basic::new(head, Vec::new())
    }

    pub fn head(&self) -> &Head {
        &self.0.head
    }

    /// The arguments.
    pub fn args(&self) -> &[Basic] {
        &self.0.args
    }

    /// The node with `args` in place of its arguments, unevaluated
    /// (`self.func(*args)`).
    pub fn rebuild(&self, args: Vec<Basic>) -> Basic {
        if args.is_empty() && self.0.args.is_empty() {
            return self.clone();
        }
        Basic::new(self.0.head.clone(), args)
    }

    /// The name of the Python class.
    pub fn class_name(&self) -> &'static str {
        self.0.head.class_name()
    }

    pub fn is_atom(&self) -> bool {
        self.0.args.is_empty() && self.0.head.is_atom()
    }

    pub fn ptr_eq(&self, other: &Basic) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    // -- the class flags of SymPy

    pub fn is_number_class(&self) -> bool {
        matches!(
            self.0.head,
            Head::Rational { .. } | Head::Infinity | Head::NegativeInfinity | Head::ComplexInfinity | Head::NaN
        )
    }

    pub fn is_rational_class(&self) -> bool {
        matches!(self.0.head, Head::Rational { .. })
    }

    /// Whether the node is an integer (`Rational(2, 1)` is one).
    pub fn is_integer_value(&self) -> bool {
        matches!(&self.0.head, Head::Rational { value, .. } if value.is_int())
    }

    pub fn is_symbol(&self) -> bool {
        matches!(self.0.head, Head::Symbol(_) | Head::Dummy(_, _) | Head::Wild(_))
    }

    pub fn is_add(&self) -> bool {
        matches!(self.0.head, Head::Add)
    }

    pub fn is_mul(&self) -> bool {
        matches!(self.0.head, Head::Mul)
    }

    /// Whether the node is the applied function `f`.
    pub fn is_function(&self, f: Function) -> bool {
        matches!(self.0.head, Head::Function(g) if g == f)
    }

    pub fn is_pow(&self) -> bool {
        matches!(self.0.head, Head::Pow)
    }

    /// The value of a rational.
    pub fn as_rational(&self) -> Option<&Rat> {
        match &self.0.head {
            Head::Rational { value, .. } => Some(value),
            _ => None,
        }
    }

    /// The assumptions of a symbol (with their consequences).
    pub fn symbol_facts(&self) -> Option<&Facts> {
        match &self.0.head {
            Head::Symbol(s) | Head::Dummy(s, _) => Some(&s.assumptions),
            Head::Wild(w) => Some(&w.symbol.assumptions),
            _ => None,
        }
    }

    /// The closed facts of the node, computed once by `compute` (Python's
    /// `_facts_cache`).
    pub fn cached_facts(&self, compute: impl FnOnce() -> Facts) -> &Facts {
        self.0.facts.get_or_init(compute)
    }

    // -- ordering

    fn compare_content(&self) -> Vec<CompareItem> {
        match &self.0.head {
            Head::Rational { value, .. } => {
                vec![
                    CompareItem::Int(value.numerator().clone()),
                    CompareItem::Int(denom(value)),
                ]
            }
            Head::Symbol(s) => s.compare_content(),
            Head::Dummy(s, index) => {
                let mut content: Vec<CompareItem> = s.compare_content();
                content.push(CompareItem::Int(Int::from(*index)));
                content
            }
            Head::Wild(w) => {
                // SymPy adds the exclusions and the properties
                let mut content: Vec<CompareItem> = w.symbol.compare_content();
                content.push(CompareItem::Int(Int::ZERO));
                content.push(CompareItem::Int(Int::ZERO));
                content
            }
            _ => self
                .0
                .args
                .iter()
                .map(|a| CompareItem::Key(a.compare_key().clone()))
                .collect(),
        }
    }

    /// The key of SymPy's `compare` order, computed once per node.
    pub fn compare_key(&self) -> &CompareKey {
        self.0
            .compare_key
            .get_or_init(|| compare_key(self.0.head.compare_name(), self.compare_content()))
    }

    /// `-1`, `0` or `1` as `self` sorts before, with, or after `other` in the
    /// order of the arguments of sums and products.
    pub fn compare(&self, other: &Basic) -> i32 {
        if self.ptr_eq(other) {
            return 0;
        }
        match self.compare_key().cmp(other.compare_key()) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }

    // -- traversal

    /// The nodes in preorder (the node, then its arguments in order).
    pub fn preorder(&self) -> Vec<Basic> {
        let mut result: Vec<Basic> = Vec::new();
        let mut stack: Vec<Basic> = vec![self.clone()];
        while let Some(node) = stack.pop() {
            for a in node.args().iter().rev() {
                stack.push(a.clone());
            }
            result.push(node);
        }
        result
    }

    /// The number of nodes of the tree.
    pub fn count_nodes(&self) -> usize {
        1 + self.0.args.iter().map(|a| a.count_nodes()).sum::<usize>()
    }

    /// The atoms of the tree (`atoms()`), or the nodes satisfying `filter`.
    pub fn atoms(&self, filter: Option<&dyn Fn(&Basic) -> bool>) -> Vec<Basic> {
        let mut seen: std::collections::HashSet<Basic> = std::collections::HashSet::new();
        let mut result: Vec<Basic> = Vec::new();
        for node in self.preorder() {
            let keep: bool = match filter {
                Some(f) => f(&node),
                None => node.args().is_empty(),
            };
            if keep && seen.insert(node.clone()) {
                result.push(node);
            }
        }
        result
    }

    /// The symbols of the tree (`free_symbols`; nothing binds symbols yet).
    pub fn free_symbols(&self) -> BTreeSet<Basic> {
        match &self.0.head {
            Head::Subs => {
                // the variables are bound
                let variables: BTreeSet<Basic> = self.0.args[1].args().iter().cloned().collect();
                let mut result: BTreeSet<Basic> = self.0.args[0]
                    .free_symbols()
                    .into_iter()
                    .filter(|s| !variables.contains(s))
                    .collect();
                for p in self.0.args[2].args() {
                    result.extend(p.free_symbols());
                }
                result
            }
            Head::IndexedRoot => {
                // the variable is bound
                let variable: &Basic = &self.0.args[1];
                self.0.args[0]
                    .free_symbols()
                    .into_iter()
                    .filter(|s| s != variable)
                    .collect()
            }
            Head::ConditionSet => crate::sets::conditionset::condition_set_free_symbols(self),
            _ if self.is_symbol() => std::iter::once(self.clone()).collect(),
            _ => {
                let mut result: BTreeSet<Basic> = BTreeSet::new();
                for a in &self.0.args {
                    result.extend(a.free_symbols());
                }
                result
            }
        }
    }

    /// Whether the tree contains `pattern` as a subtree.
    pub fn has(&self, pattern: &Basic) -> bool {
        self.preorder().iter().any(|n| n == pattern)
    }

    /// The tree with the subtrees in `rules` replaced, purely structurally
    /// (the rebuilt nodes are not evaluated); the largest subtrees first.
    pub fn xreplace(&self, rules: &HashMap<Basic, Basic>) -> Basic {
        if let Some(value) = rules.get(self) {
            return value.clone();
        }
        if self.0.args.is_empty() {
            return self.clone();
        }
        let args: Vec<Basic> = self.0.args.iter().map(|a| a.xreplace(rules)).collect();
        if args.iter().zip(&self.0.args).all(|(a, b)| a.ptr_eq(b)) {
            return self.clone();
        }
        self.rebuild(args)
    }
}

impl PartialEq for Basic {
    fn eq(&self, other: &Basic) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        if self.0.hash != other.0.hash {
            return false;
        }
        if !self.0.head.same(&other.0.head) {
            // Reals and Interval(-oo, oo, True, True)
            let whole = |x: &Basic| matches!(x.head(), Head::Reals) || crate::sets::sets::is_whole_line(x);
            return matches!(self.head(), Head::Reals | Head::Interval)
                && matches!(other.head(), Head::Reals | Head::Interval)
                && whole(self)
                && whole(other);
        }
        if self.0.head.args_are_a_set() {
            let unique = |args: &[Basic]| -> Vec<Basic> {
                let mut out: Vec<Basic> = Vec::new();
                for a in args {
                    if !out.contains(a) {
                        out.push(a.clone());
                    }
                }
                out
            };
            let (a, b) = (unique(&self.0.args), unique(&other.0.args));
            return a.len() == b.len() && a.iter().all(|x| b.contains(x));
        }
        self.0.args == other.0.args
    }
}

impl Eq for Basic {}

impl Hash for Basic {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash.hash(state);
    }
}

/// The `compare` order (for sets of nodes; equal nodes compare equal).
impl PartialOrd for Basic {
    fn partial_cmp(&self, other: &Basic) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Basic {
    fn cmp(&self, other: &Basic) -> std::cmp::Ordering {
        if self == other {
            return std::cmp::Ordering::Equal;
        }
        self.compare_key()
            .cmp(other.compare_key())
            .then_with(|| self.0.hash.cmp(&other.0.hash))
    }
}

impl fmt::Debug for Basic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The printed form of the atoms (Python's `str`); a compound node prints
/// as `Class(arg, ...)` until the printer is ported.
impl fmt::Display for Basic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0.head {
            Head::Rational { value, .. } => {
                if value.is_int() {
                    write!(f, "{}", value.numerator())
                } else {
                    write!(f, "{}/{}", value.numerator(), denom(value))
                }
            }
            Head::Infinity => write!(f, "oo"),
            Head::NegativeInfinity => write!(f, "-oo"),
            Head::ComplexInfinity => write!(f, "zoo"),
            Head::NaN => write!(f, "nan"),
            Head::Pi => write!(f, "pi"),
            Head::Exp1 => write!(f, "E"),
            Head::EulerGamma => write!(f, "EulerGamma"),
            Head::Catalan => write!(f, "Catalan"),
            Head::GoldenRatio => write!(f, "GoldenRatio"),
            Head::TribonacciConstant => write!(f, "TribonacciConstant"),
            Head::ImaginaryUnit => write!(f, "I"),
            Head::Symbol(s) => write!(f, "{}", s.name),
            Head::Dummy(s, _) => write!(f, "_{}", s.name),
            Head::Wild(w) => write!(f, "{}_", w.symbol.name),
            // Python's str: the printer
            _ => write!(f, "{}", super::printing::sstr(self)),
        }
    }
}
