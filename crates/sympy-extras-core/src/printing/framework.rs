//! Port of `sympy_extras/printing/framework.py`: the typed framework of
//! the two-dimensional printers. A rule returns a [`PrintResult`] (the
//! output and the binding strength of its outermost operator as printed),
//! a parent passes each child a [`Context`] (where it stands, what the
//! position requires), and [`needs_brackets`] is the only place where
//! brackets are decided. The Python `Dispatcher` (a table from classes to
//! rules) is a `match` on the head of the node in the concrete printers.

/// The binding strength of the outermost operator of a printed form, from
/// the loosest to the tightest (`Binding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Binding {
    /// what a delimited position requires: nothing
    Lowest = 0,
    /// a block which is bracketed wherever it is not delimited (cases)
    Block = 5,
    /// `a => b`, `a <=> b` (and `a xor b` in LaTeX)
    Implication = 10,
    /// `a xor b` in a picture
    Xor = 15,
    /// `a or b`
    Or = 20,
    /// `a and b`
    And = 30,
    /// `a = b`, `a < b`, `x in S`
    Relation = 35,
    /// `a mod b`
    Modulo = 37,
    /// `A union B`, `A \ B`, `A x B`
    SetOperation = 38,
    /// `a + b`
    Sum = 40,
    /// `-a`: a sign in front of a product or a number
    Negation = 45,
    /// `a / b` on one line
    Quotient = 48,
    /// `a b`, a built-up fraction of LaTeX, an operator with its body
    Product = 50,
    /// a fraction with a bar drawn in a picture
    Fraction = 55,
    /// the one-character root sign of a picture
    Root = 58,
    /// `a^b`, a function with an exponent attached to its name
    Power = 60,
    /// a prefix or postfix sign in a picture
    Affix = 65,
    /// `f(x)`, and `x!` in LaTeX
    Application = 70,
    /// `\neg a` in LaTeX
    Not = 100,
    /// a name, a number, anything delimited
    Atom = 1000,
}

impl Binding {
    /// The binding of a name (`"POWER"`), [`Binding::Lowest`] for an
    /// unknown one (`binding_named`).
    pub fn named(name: &str) -> Binding {
        match name {
            "LOWEST" => Binding::Lowest,
            "BLOCK" => Binding::Block,
            "IMPLICATION" => Binding::Implication,
            "XOR" => Binding::Xor,
            "OR" => Binding::Or,
            "AND" => Binding::And,
            "RELATION" => Binding::Relation,
            "MODULO" => Binding::Modulo,
            "SET_OPERATION" => Binding::SetOperation,
            "SUM" => Binding::Sum,
            "NEGATION" => Binding::Negation,
            "QUOTIENT" => Binding::Quotient,
            "PRODUCT" => Binding::Product,
            "FRACTION" => Binding::Fraction,
            "ROOT" => Binding::Root,
            "POWER" => Binding::Power,
            "AFFIX" => Binding::Affix,
            "APPLICATION" => Binding::Application,
            "NOT" => Binding::Not,
            "ATOM" => Binding::Atom,
            _ => Binding::Lowest,
        }
    }
}

/// Where a child stands in its parent (`Position`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Position {
    /// the whole expression
    Top,
    /// delimited by its parent: an argument of a function, an element of a
    /// container, a bound, the body of a root, the content of brackets
    Delimited,
    /// a term of a sum
    Term,
    /// a factor of a product
    Factor,
    /// the numerator of a fraction
    Numerator,
    /// the denominator of a fraction
    Denominator,
    /// the base of a power
    Base,
    /// the exponent of a power
    Exponent,
    /// a subscript
    Subscript,
    /// a superscript which is not an exponent
    Superscript,
    /// the argument of a function printed without brackets (`sin x`)
    BareArgument,
    /// the operand of a postfix or prefix operator
    Operand,
    /// the left operand of a non-associative infix operator
    Left,
    /// the right operand of a non-associative infix operator
    Right,
    /// a side of a relation
    Side,
    /// an operand of a logical connective
    Connective,
    /// the operand of a negation
    Negated,
    /// the body of a sum, a product, an integral, a limit, a derivative
    Body,
    /// the point of a limit, which carries the mark of the direction
    Point,
}

/// What a parent passes down to a child: where the child stands and what
/// the position requires; never modified, the methods derive the contexts
/// of the children keeping the inherited state (`Context`).
#[derive(Clone, Debug, PartialEq)]
pub struct Context<T: Clone> {
    pub position: Position,
    /// the weakest binding the position takes without brackets
    pub level: Binding,
    /// a result binding exactly as `level` is bracketed too
    pub strict: bool,
    /// something follows the child on its line, at the binding of a product or tighter
    pub followed: bool,
    /// a script is attached to the child
    pub scripts_follow: bool,
    /// the printed exponent which a base may render itself (`sin^2(x)`)
    pub exponent_offered: Option<T>,
    /// the child may render the negation of its parent itself
    pub negation_offered: bool,
    /// inside an exponent, a subscript or a superscript
    pub in_exponent: bool,
    /// on a line of text: no built-up constructs where a flat one exists
    pub inline: bool,
    /// a result binding exactly as this is bracketed too, whatever `level` is
    pub excluded: Option<Binding>,
    /// the parent asks for the built-up form of a number which has a flat one; not inherited
    pub display: bool,
}

impl<T: Clone> Default for Context<T> {
    fn default() -> Context<T> {
        Context {
            position: Position::Top,
            level: Binding::Lowest,
            strict: false,
            followed: false,
            scripts_follow: false,
            exponent_offered: None,
            negation_offered: false,
            in_exponent: false,
            inline: false,
            excluded: None,
            display: false,
        }
    }
}

/// The optional parts of a child context (the arguments of `_child` and
/// `at` after the position and the level).
#[derive(Clone, Debug)]
pub struct ChildOptions<T: Clone> {
    pub strict: bool,
    pub followed: bool,
    pub scripts_follow: bool,
    pub exponent_offered: Option<T>,
    pub negation_offered: bool,
    pub in_exponent: Option<bool>,
    pub excluded: Option<Binding>,
}

impl<T: Clone> Default for ChildOptions<T> {
    fn default() -> ChildOptions<T> {
        ChildOptions {
            strict: false,
            followed: false,
            scripts_follow: false,
            exponent_offered: None,
            negation_offered: false,
            in_exponent: None,
            excluded: None,
        }
    }
}

impl<T: Clone> Context<T> {
    /// A context on a line of text from the start (`Context(inline=True)`).
    pub fn inline_top() -> Context<T> {
        Context {
            inline: true,
            ..Context::default()
        }
    }

    fn child(&self, position: Position, level: Binding, o: ChildOptions<T>) -> Context<T> {
        Context {
            position,
            level,
            strict: o.strict,
            followed: o.followed,
            scripts_follow: o.scripts_follow,
            exponent_offered: o.exponent_offered,
            negation_offered: o.negation_offered,
            in_exponent: o.in_exponent.unwrap_or(self.in_exponent),
            inline: self.inline,
            excluded: o.excluded,
            display: false,
        }
    }

    /// A position whose requirements are those of the notation of one
    /// printer: the child must bind at least as `level`; an exponent, a
    /// subscript, a superscript and the point of a limit are inside an
    /// exponent (`at`).
    pub fn at(&self, position: Position, level: Binding, o: ChildOptions<T>) -> Context<T> {
        let scripts: bool = matches!(
            position,
            Position::Exponent | Position::Subscript | Position::Superscript | Position::Point
        );
        let o = ChildOptions {
            scripts_follow: false,
            in_exponent: if scripts { Some(true) } else { None },
            ..o
        };
        self.child(position, level, o)
    }

    /// A position delimited by the parent: nothing is required.
    pub fn delimited(&self) -> Context<T> {
        self.child(Position::Delimited, Binding::Lowest, ChildOptions::default())
    }

    /// A term of a sum: a nested sum, a relation or a `mod` are bracketed.
    pub fn term(&self) -> Context<T> {
        self.child(
            Position::Term,
            Binding::Sum,
            ChildOptions {
                strict: true,
                ..ChildOptions::default()
            },
        )
    }

    /// A factor of a product; `followed` when it is not the last.
    pub fn factor(&self, followed: bool) -> Context<T> {
        self.child(
            Position::Factor,
            Binding::Product,
            ChildOptions {
                followed,
                ..ChildOptions::default()
            },
        )
    }

    /// The numerator of a fraction: delimited when the fraction is built
    /// up, a product before the slash of an inline one.
    pub fn numerator(&self, inline: bool) -> Context<T> {
        if inline {
            return self.child(
                Position::Numerator,
                Binding::Product,
                ChildOptions {
                    followed: true,
                    ..ChildOptions::default()
                },
            );
        }
        self.child(Position::Numerator, Binding::Lowest, ChildOptions::default())
    }

    /// The denominator of a fraction: after a slash a power or tighter.
    pub fn denominator(&self, inline: bool) -> Context<T> {
        if inline {
            return self.child(Position::Denominator, Binding::Power, ChildOptions::default());
        }
        self.child(Position::Denominator, Binding::Lowest, ChildOptions::default())
    }

    /// The base of a power: an atom, or something which renders the
    /// exponent itself.
    pub fn base(&self, exponent: Option<T>) -> Context<T> {
        self.child(
            Position::Base,
            Binding::Atom,
            ChildOptions {
                scripts_follow: true,
                exponent_offered: exponent,
                ..ChildOptions::default()
            },
        )
    }

    pub fn exponent(&self) -> Context<T> {
        self.child(
            Position::Exponent,
            Binding::Lowest,
            ChildOptions {
                in_exponent: Some(true),
                ..ChildOptions::default()
            },
        )
    }

    pub fn subscript(&self) -> Context<T> {
        self.child(
            Position::Subscript,
            Binding::Lowest,
            ChildOptions {
                in_exponent: Some(true),
                ..ChildOptions::default()
            },
        )
    }

    pub fn superscript(&self) -> Context<T> {
        self.child(
            Position::Superscript,
            Binding::Lowest,
            ChildOptions {
                in_exponent: Some(true),
                ..ChildOptions::default()
            },
        )
    }

    /// The argument of a function printed without brackets.
    pub fn bare_argument(&self) -> Context<T> {
        self.child(Position::BareArgument, Binding::Product, ChildOptions::default())
    }

    /// The operand of a postfix or prefix operator: an atom.
    pub fn operand(&self) -> Context<T> {
        self.child(
            Position::Operand,
            Binding::Atom,
            ChildOptions {
                followed: true,
                ..ChildOptions::default()
            },
        )
    }

    /// The left operand of an infix operator binding as `level`.
    pub fn left(&self, level: Binding) -> Context<T> {
        self.child(
            Position::Left,
            level,
            ChildOptions {
                strict: true,
                followed: true,
                ..ChildOptions::default()
            },
        )
    }

    /// The right operand of an infix operator binding as `level`.
    pub fn right(&self, level: Binding) -> Context<T> {
        self.child(
            Position::Right,
            level,
            ChildOptions {
                strict: true,
                ..ChildOptions::default()
            },
        )
    }

    /// A side of a relation: a nested relation or a connective is bracketed.
    pub fn side(&self) -> Context<T> {
        self.child(
            Position::Side,
            Binding::Relation,
            ChildOptions {
                strict: true,
                ..ChildOptions::default()
            },
        )
    }

    /// An operand of a logical connective: a nested connective is bracketed.
    pub fn connective(&self) -> Context<T> {
        self.child(Position::Connective, Binding::Relation, ChildOptions::default())
    }

    /// The operand of a negation: an application or tighter, or a
    /// connective which renders the negation itself.
    pub fn negated(&self) -> Context<T> {
        self.child(
            Position::Negated,
            Binding::Application,
            ChildOptions {
                negation_offered: true,
                ..ChildOptions::default()
            },
        )
    }

    /// The body of a sum, a product, an integral, a limit or a derivative.
    pub fn body(&self) -> Context<T> {
        self.child(Position::Body, Binding::Product, ChildOptions::default())
    }

    /// The point of a limit, to which the mark of the direction is attached.
    pub fn point(&self) -> Context<T> {
        self.child(
            Position::Point,
            Binding::Atom,
            ChildOptions {
                scripts_follow: true,
                in_exponent: Some(true),
                ..ChildOptions::default()
            },
        )
    }

    /// The context itself: for a node which prints as its argument.
    pub fn same(&self) -> Context<T> {
        self.clone()
    }

    /// The context without the exponent and the negation offered.
    pub fn without_offers(&self) -> Context<T> {
        Context {
            exponent_offered: None,
            negation_offered: false,
            ..self.clone()
        }
    }

    /// The context asking for the built-up form of a number.
    pub fn displayed(&self) -> Context<T> {
        Context {
            display: true,
            ..self.clone()
        }
    }
}

/// What a rule returns: the output and how it binds, as printed (`Result`).
#[derive(Clone, Debug, PartialEq)]
pub struct PrintResult<T: Clone> {
    pub output: T,
    pub binding: Binding,
    /// the output without its leading minus sign, when it starts with one
    pub unsigned: Option<T>,
    /// ends with a construct whose body extends to the right
    pub open_ended: bool,
    /// ends with a script of its own, so that a further script needs brackets
    pub scripted: bool,
    /// the rule rendered the operator its parent offered
    pub absorbed: bool,
    /// contains a function applied to its arguments outside any brackets
    pub applies: bool,
    /// the outermost layout is a fraction bar drawn in two dimensions
    pub stacked: bool,
}

impl<T: Clone> PrintResult<T> {
    /// A result with the output and the binding, the flags unset.
    pub fn new(output: T, binding: Binding) -> PrintResult<T> {
        PrintResult {
            output,
            binding,
            unsigned: None,
            open_ended: false,
            scripted: false,
            absorbed: false,
            applies: false,
            stacked: false,
        }
    }
}

/// Whether the printed child must be bracketed in its position: the single
/// place where brackets are decided (`needs_brackets`).
pub fn needs_brackets<T: Clone>(result: &PrintResult<T>, context: &Context<T>) -> bool {
    if result.absorbed {
        return false;
    }
    if result.binding < context.level || (context.strict && result.binding == context.level) {
        return true;
    }
    if result.open_ended && context.followed {
        return true;
    }
    if result.scripted && context.scripts_follow {
        return true;
    }
    if result.applies && context.position == Position::BareArgument {
        return true;
    }
    if context.excluded == Some(result.binding) {
        return true;
    }
    if result.stacked && matches!(context.position, Position::Numerator | Position::Denominator) {
        return true;
    }
    false
}

/// The descent shared by the printers: [`TreePrinter::print`] prints a
/// child in its context and brackets it when [`needs_brackets`] says so
/// (`TreePrinter`).
pub trait TreePrinter<E, T: Clone> {
    /// The node printed in the context, without brackets around it.
    fn rule(&self, node: &E, context: &Context<T>) -> crate::error::PolyResult<PrintResult<T>>;

    /// The output of `result` in brackets.
    fn bracket(&self, result: &PrintResult<T>, context: &Context<T>) -> T;

    /// A result put in a position: in brackets, an atom then, when
    /// [`needs_brackets`] says so.
    fn enclose(&self, result: PrintResult<T>, context: &Context<T>) -> PrintResult<T> {
        if needs_brackets(&result, context) {
            return PrintResult::new(self.bracket(&result, context), Binding::Atom);
        }
        result
    }

    /// The node printed in the context, in brackets when its position
    /// requires them.
    fn print(&self, node: &E, context: &Context<T>) -> crate::error::PolyResult<PrintResult<T>> {
        let result: PrintResult<T> = self.rule(node, context)?;
        Ok(self.enclose(result, context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brackets_as_the_docstrings() {
        let top: Context<String> = Context::default();
        let total = PrintResult::new("a + b".to_string(), Binding::Sum);
        let got: Vec<bool> = [top.clone(), top.delimited(), top.term(), top.factor(true), top.side()]
            .iter()
            .map(|c| needs_brackets(&total, c))
            .collect();
        assert_eq!(got, vec![false, false, true, true, false]);
        let mut series = PrintResult::new("sum x".to_string(), Binding::Product);
        series.open_ended = true;
        assert!(needs_brackets(&series, &top.factor(true)));
        assert!(!needs_brackets(&series, &top.factor(false)));
        let mut minus_x = PrintResult::new("- x".to_string(), Binding::Negation);
        minus_x.unsigned = Some("x".to_string());
        assert!(!needs_brackets(&minus_x, &top.term()));
        assert!(needs_brackets(&minus_x, &top.factor(false)));
        assert!(needs_brackets(&minus_x, &top.base(None)));
        let raised = top.at(
            Position::Exponent,
            Binding::Lowest,
            ChildOptions {
                excluded: Some(Binding::Power),
                ..ChildOptions::default()
            },
        );
        assert!(raised.in_exponent);
        assert!(needs_brackets(
            &PrintResult::new("y**z".to_string(), Binding::Power),
            &raised
        ));
        assert!(!needs_brackets(
            &PrintResult::new("y + z".to_string(), Binding::Sum),
            &raised
        ));
        let base = top.exponent().base(Some("2".to_string()));
        assert_eq!(
            (
                base.position,
                base.level,
                base.exponent_offered.clone(),
                base.in_exponent
            ),
            (Position::Base, Binding::Atom, Some("2".to_string()), true)
        );
        assert_eq!(Binding::named("SUM"), Binding::Sum);
        assert_eq!(Binding::named("?"), Binding::Lowest);
    }
}
