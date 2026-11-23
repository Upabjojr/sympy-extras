"""The typed framework of the two-dimensional printers (LaTeX, pretty).

Both printers (:mod:`sympy_extras.printing.latex`, whose output is a
string, and :mod:`sympy_extras.printing.pretty`, whose output is a
:class:`~sympy_extras.printing.pretty.layout.Picture`) walk the tree with
the three typed pieces of this module, none of which knows the output
type:

- a :class:`Result`: what a rule returns, the output together with the
  **binding strength of its outermost operator as printed** (a negative
  number binds as a negation, a power printed as a fraction as a product,
  a function which took the exponent of its parent as a power) and the few
  flags a parent needs (the output without its leading minus sign, an
  open-ended construct on the right, a superscript at the end, a fraction
  bar drawn in two dimensions);
- a :class:`Context`: what a parent passes down to each child, immutable:
  the position of the child, the binding strength the position requires,
  what follows the child, the operator the child may render itself (an
  exponent, a negation) and the inherited state (inside an exponent,
  inline);
- :func:`needs_brackets`: the **only** place where brackets are decided,
  from the result of the child and its context. No rule adds brackets of
  its own: :meth:`TreePrinter.print` prints a child, asks
  :func:`needs_brackets` and has the concrete printer draw them.

The rules are found by the class of the node in a :class:`Dispatcher`
(a table from classes to typed functions, looked up along the bases of
the class), never by the name of the class: in a statically typed
language this is a ``match`` on the node.

There is one scale of binding strengths (:class:`Binding`) and one list
of positions (:class:`Position`) for both printers. A printer says how
**its own notation** binds, so that the same node may bind differently in
the two: a quotient is a built-up ``\\frac`` in LaTeX, which binds as a
product, and a fraction bar in a picture, which binds as a
:attr:`Binding.FRACTION`; some levels and flags are produced by one
printer only (they are marked below).

SymPy's printers decide the brackets in the parent from a precedence
number computed from the *class* of the child (``precedence_traditional``),
whatever the child printed; most of the defects listed in sympy/sympy#30844
come from there. This design is not SymPy's.

References
==========

.. [1] SymPy issue 30844, *Printers: systematic review of precedence and
       other defects*, https://github.com/sympy/sympy/issues/30844
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Callable, Generic, Optional, TypeVar

__all__ = ['Binding', 'Position', 'Context', 'Result', 'needs_brackets', 'Dispatcher', 'TreePrinter']

#: the output of a printer
T = TypeVar('T')
#: the objects a printer prints
E = TypeVar('E')
#: a printer
P = TypeVar('P')
#: a class of nodes
N = TypeVar('N')


class Binding(IntEnum):
    """The binding strength of the outermost operator of a printed form,
    from the loosest to the tightest.

    Examples
    ========

    >>> from sympy_extras.printing.framework import Binding
    >>> Binding.SUM < Binding.NEGATION < Binding.PRODUCT < Binding.POWER < Binding.ATOM
    True
    """

    #: what a delimited position requires: nothing
    LOWEST = 0
    #: a block which is bracketed wherever it is not delimited (cases)
    BLOCK = 5
    #: ``a => b``, ``a <=> b`` (and ``a xor b`` in LaTeX)
    IMPLICATION = 10
    #: ``a xor b`` in a picture
    XOR = 15
    #: ``a or b``
    OR = 20
    #: ``a and b``
    AND = 30
    #: ``a = b``, ``a < b``, ``x in S``
    RELATION = 35
    #: ``a mod b``
    MODULO = 37
    #: ``A union B``, ``A \\ B``, ``A x B``
    SET_OPERATION = 38
    #: ``a + b``
    SUM = 40
    #: ``-a``: a sign in front of a product or a number
    NEGATION = 45
    #: ``a / b`` on one line
    QUOTIENT = 48
    #: ``a b``, a built-up fraction of LaTeX, an operator with its body (a sum, an integral, a derivative)
    PRODUCT = 50
    #: a fraction with a bar drawn in a picture
    FRACTION = 55
    #: the one-character root sign of a picture (``√x``), which no bar closes
    ROOT = 58
    #: ``a^b``, a function with an exponent attached to its name
    POWER = 60
    #: a prefix or postfix sign in a picture: ``¬x``, ``x!``
    AFFIX = 65
    #: ``f(x)``, and ``x!`` in LaTeX
    APPLICATION = 70
    #: ``\neg a`` in LaTeX
    NOT = 100
    #: a name, a number, anything delimited
    ATOM = 1000


class Position(Enum):
    """Where a child stands in its parent."""

    #: the whole expression
    TOP = 'top'
    #: delimited by its parent: an argument of a function, an element of a
    #: container, a bound, the body of a root under its bar, the content of brackets
    DELIMITED = 'delimited'
    #: a term of a sum
    TERM = 'term'
    #: a factor of a product
    FACTOR = 'factor'
    #: the numerator of a fraction
    NUMERATOR = 'numerator'
    #: the denominator of a fraction
    DENOMINATOR = 'denominator'
    #: the base of a power
    BASE = 'base'
    #: the exponent of a power
    EXPONENT = 'exponent'
    #: a subscript
    SUBSCRIPT = 'subscript'
    #: a superscript which is not an exponent
    SUPERSCRIPT = 'superscript'
    #: the argument of a function printed without brackets (``sin x``)
    BARE_ARGUMENT = 'bare argument'
    #: the operand of a postfix or prefix operator (``x!``, the variable after the ``d`` of an integral)
    OPERAND = 'operand'
    #: the left operand of a non-associative infix operator (``mod``, a set difference)
    LEFT = 'left'
    #: the right operand of a non-associative infix operator
    RIGHT = 'right'
    #: a side of a relation
    SIDE = 'side'
    #: an operand of a logical connective
    CONNECTIVE = 'connective'
    #: the operand of a negation
    NEGATED = 'negated'
    #: the body of a sum, a product, an integral, a limit, a derivative
    BODY = 'body'
    #: the point of a limit, which carries the mark of the direction
    POINT = 'point'


@dataclass(frozen=True)
class Context(Generic[T]):
    """What a parent passes down to a child: where the child stands and
    what the position requires. A context is never modified; a rule
    derives the contexts of its children with the methods below, which
    keep the inherited state (``in_exponent``, ``inline``).

    Examples
    ========

    >>> from sympy_extras.printing.framework import Binding, Context, Position
    >>> top: Context[str] = Context()
    >>> base = top.exponent().base('2')
    >>> base.position.name, base.level.name, base.exponent_offered, base.in_exponent
    ('BASE', 'ATOM', '2', True)
    """

    position: Position = Position.TOP
    #: the weakest binding the position takes without brackets
    level: Binding = Binding.LOWEST
    #: a result binding exactly as ``level`` is bracketed too
    strict: bool = False
    #: something follows the child on its line, at the binding of a product or tighter
    followed: bool = False
    #: a script (an exponent, the mark of a direction) is attached to the child
    scripts_follow: bool = False
    #: the printed exponent which a base may render itself (``sin^2(x)``)
    exponent_offered: Optional[T] = None
    #: the child may render the negation of its parent itself (``a =/=> b``)
    negation_offered: bool = False
    #: inside an exponent, a subscript or a superscript
    in_exponent: bool = False
    #: on a line of text: no built-up constructs where a flat one exists
    inline: bool = False
    #: a result binding exactly as this is bracketed too, whatever
    #: ``level`` is (a power in a raised exponent of a picture)
    excluded: Optional[Binding] = None
    #: the parent asks for the built-up form of a number which has a flat
    #: one (a fraction among the tall terms of a sum in a picture); not inherited
    display: bool = False

    def _child(self, position: Position, level: Binding, strict: bool = False, followed: bool = False,
               scripts_follow: bool = False, exponent_offered: Optional[T] = None, negation_offered: bool = False,
               in_exponent: Optional[bool] = None, excluded: Optional[Binding] = None) -> Context[T]:
        return Context(position, level, strict, followed, scripts_follow, exponent_offered, negation_offered,
                       self.in_exponent if in_exponent is None else in_exponent, self.inline, excluded)

    def at(self, position: Position, level: Binding, strict: bool = False, followed: bool = False,
           exponent_offered: Optional[T] = None, negation_offered: bool = False,
           excluded: Optional[Binding] = None) -> Context[T]:
        """A position whose requirements are those of the notation of one
        printer rather than the ones of the methods below (the base of a
        power in a picture takes a function application, the one of LaTeX
        an atom): the child must bind at least as ``level``. The inherited
        state is kept; an exponent, a subscript, a superscript and the
        point of a limit are inside an exponent.

        Examples
        ========

        >>> from sympy_extras.printing.framework import Binding, Context, Position, Result, needs_brackets
        >>> top: Context[str] = Context()
        >>> raised = top.at(Position.EXPONENT, Binding.LOWEST, excluded=Binding.POWER)
        >>> raised.in_exponent, needs_brackets(Result('y**z', Binding.POWER), raised)
        (True, True)
        >>> needs_brackets(Result('y + z', Binding.SUM), raised)
        False
        """
        scripts = position in (Position.EXPONENT, Position.SUBSCRIPT, Position.SUPERSCRIPT, Position.POINT)
        return self._child(position, level, strict, followed, False, exponent_offered, negation_offered,
                           True if scripts else None, excluded)

    def delimited(self) -> Context[T]:
        """A position delimited by the parent: nothing is required."""
        return self._child(Position.DELIMITED, Binding.LOWEST)

    def term(self) -> Context[T]:
        """A term of a sum: a nested sum, a relation or a ``mod`` are
        bracketed, a negation is not (the sum writes its sign)."""
        return self._child(Position.TERM, Binding.SUM, strict=True)

    def factor(self, followed: bool) -> Context[T]:
        """A factor of a product; ``followed`` when it is not the last."""
        return self._child(Position.FACTOR, Binding.PRODUCT, followed=followed)

    def numerator(self, inline: bool) -> Context[T]:
        """The numerator of a fraction: delimited when the fraction is
        built up, a product before the slash of an inline one."""
        if inline:
            return self._child(Position.NUMERATOR, Binding.PRODUCT, followed=True)
        return self._child(Position.NUMERATOR, Binding.LOWEST)

    def denominator(self, inline: bool) -> Context[T]:
        """The denominator of a fraction: after a slash it must be a
        power or tighter (``a / (b c)``, ``a / (b / c)``)."""
        if inline:
            return self._child(Position.DENOMINATOR, Binding.POWER)
        return self._child(Position.DENOMINATOR, Binding.LOWEST)

    def base(self, exponent: Optional[T]) -> Context[T]:
        """The base of a power: an atom, or something which renders the
        exponent itself."""
        return self._child(Position.BASE, Binding.ATOM, scripts_follow=True, exponent_offered=exponent)

    def exponent(self) -> Context[T]:
        return self._child(Position.EXPONENT, Binding.LOWEST, in_exponent=True)

    def subscript(self) -> Context[T]:
        return self._child(Position.SUBSCRIPT, Binding.LOWEST, in_exponent=True)

    def superscript(self) -> Context[T]:
        return self._child(Position.SUPERSCRIPT, Binding.LOWEST, in_exponent=True)

    def bare_argument(self) -> Context[T]:
        """The argument of a function printed without brackets."""
        return self._child(Position.BARE_ARGUMENT, Binding.PRODUCT)

    def operand(self) -> Context[T]:
        """The operand of a postfix or prefix operator: an atom."""
        return self._child(Position.OPERAND, Binding.ATOM, followed=True)

    def left(self, level: Binding) -> Context[T]:
        """The left operand of an infix operator binding as ``level``."""
        return self._child(Position.LEFT, level, strict=True, followed=True)

    def right(self, level: Binding) -> Context[T]:
        """The right operand of an infix operator binding as ``level``."""
        return self._child(Position.RIGHT, level, strict=True)

    def side(self) -> Context[T]:
        """A side of a relation: a nested relation or a connective is bracketed."""
        return self._child(Position.SIDE, Binding.RELATION, strict=True)

    def connective(self) -> Context[T]:
        """An operand of a logical connective: a nested connective is bracketed."""
        return self._child(Position.CONNECTIVE, Binding.RELATION)

    def negated(self) -> Context[T]:
        """The operand of a negation: an application or tighter, or a
        connective which renders the negation itself."""
        return self._child(Position.NEGATED, Binding.APPLICATION, negation_offered=True)

    def body(self) -> Context[T]:
        """The body of a sum, a product, an integral, a limit or a
        derivative: a product or tighter."""
        return self._child(Position.BODY, Binding.PRODUCT)

    def point(self) -> Context[T]:
        """The point of a limit, to which the mark of the direction is attached."""
        return self._child(Position.POINT, Binding.ATOM, scripts_follow=True, in_exponent=True)

    def same(self) -> Context[T]:
        """The context itself: for a node which prints as its argument."""
        return self

    def without_offers(self) -> Context[T]:
        """The context without the exponent and the negation offered."""
        return replace(self, exponent_offered=None, negation_offered=False)

    def displayed(self) -> Context[T]:
        """The context asking for the built-up form of a number."""
        return replace(self, display=True)


@dataclass(frozen=True)
class Result(Generic[T]):
    """What a rule returns: the output and how it binds, as printed.

    Examples
    ========

    >>> from sympy_extras.printing.framework import Binding, Context, Result, needs_brackets
    >>> top: Context[str] = Context()
    >>> minus_x = Result('- x', Binding.NEGATION, unsigned='x')
    >>> needs_brackets(minus_x, top.term()), needs_brackets(minus_x, top.factor(False)), needs_brackets(minus_x, top.base(None))
    (False, True, True)
    """

    output: T
    binding: Binding
    #: the output without its leading minus sign, when it starts with one
    unsigned: Optional[T] = None
    #: ends with a construct whose body extends to the right (a sum, a
    #: limit, a derivative, an integral)
    open_ended: bool = False
    #: ends with a script of its own, so that a further script needs brackets
    scripted: bool = False
    #: the rule rendered the operator its parent offered (the exponent,
    #: the negation): the result stands for the parent
    absorbed: bool = False
    #: contains a function applied to its arguments outside any brackets
    applies: bool = False
    #: the outermost layout is a fraction bar drawn in two dimensions
    #: (also under a minus sign): as a part of another fraction it is
    #: bracketed, two bars of the same length do not say which one is the
    #: main one. Never set for LaTeX, which sizes the nested fractions.
    stacked: bool = False


def needs_brackets(result: Result[T], context: Context[T]) -> bool:
    """Whether the printed child must be bracketed in its position: the
    single place where brackets are decided.

    Examples
    ========

    >>> from sympy_extras.printing.framework import Binding, Context, Result, needs_brackets
    >>> top: Context[str] = Context()
    >>> total = Result('a + b', Binding.SUM)
    >>> [needs_brackets(total, c) for c in (top, top.delimited(), top.term(), top.factor(True), top.side())]
    [False, False, True, True, False]
    >>> series = Result('sum x', Binding.PRODUCT, open_ended=True)
    >>> needs_brackets(series, top.factor(followed=True)), needs_brackets(series, top.factor(followed=False))
    (True, False)
    >>> bar = Result('a over b', Binding.FRACTION, stacked=True)
    >>> needs_brackets(bar, top.numerator(False)), needs_brackets(bar, top.factor(False))
    (True, False)
    """
    if result.absorbed:
        return False
    if result.binding < context.level or (context.strict and result.binding == context.level):
        return True
    if result.open_ended and context.followed:
        return True
    if result.scripted and context.scripts_follow:
        return True
    if result.applies and context.position is Position.BARE_ARGUMENT:
        return True
    if context.excluded is not None and result.binding == context.excluded:
        return True
    if result.stacked and context.position in (Position.NUMERATOR, Position.DENOMINATOR):
        return True
    return False


class Dispatcher(Generic[P, E, T]):
    """The rules of a printer ``P`` by the class of the node: a table
    from classes to typed functions ``rule(printer, node, context)``. The
    rule of a node is the one of the first of its classes (the class,
    then its bases in the order of the method resolution) which has one.

    Examples
    ========

    >>> from sympy_extras.printing.framework import Binding, Context, Dispatcher, Result
    >>> rules: Dispatcher[None, object, str] = Dispatcher()
    >>> @rules.of(int)
    ... def _integer(printer: None, n: int, context: Context[str]) -> Result[str]:
    ...     return Result(str(n), Binding.ATOM if n >= 0 else Binding.NEGATION)
    >>> rule = rules.lookup(bool)
    >>> rule is not None and rule(None, True, Context()).output, rules.lookup(str) is None
    ('True', True)
    """

    def __init__(self) -> None:
        self._rules: dict[type[object], Callable[[P, E, Context[T]], Result[T]]] = {}
        self._resolved: dict[type[object], Optional[Callable[[P, E, Context[T]], Result[T]]]] = {}

    def register(self, cls: type[N], rule: Callable[[P, N, Context[T]], Result[T]]) -> None:
        """Print the instances of ``cls`` with ``rule``."""
        def narrowed(printer: P, node: E, context: Context[T]) -> Result[T]:
            if not isinstance(node, cls):
                raise TypeError('the rule of %s got %r' % (cls.__name__, node))
            return rule(printer, node, context)
        self._rules[cls] = narrowed
        self._resolved.clear()

    def of(self, *classes: type[N]) -> Callable[[Callable[[P, N, Context[T]], Result[T]]],
                                                 Callable[[P, N, Context[T]], Result[T]]]:
        """The decorator form of :meth:`register`, for one or more classes."""
        def decorator(rule: Callable[[P, N, Context[T]], Result[T]]) -> Callable[[P, N, Context[T]], Result[T]]:
            for cls in classes:
                self.register(cls, rule)
            return rule
        return decorator

    def lookup(self, cls: type[object]) -> Optional[Callable[[P, E, Context[T]], Result[T]]]:
        """The rule of the instances of ``cls``, ``None`` when neither the
        class nor one of its bases has one."""
        if cls in self._resolved:
            return self._resolved[cls]
        found: Optional[Callable[[P, E, Context[T]], Result[T]]] = None
        for base in cls.__mro__:
            if base in self._rules:
                found = self._rules[base]
                break
        self._resolved[cls] = found
        return found


class TreePrinter(Generic[E, T]):
    """The descent shared by the printers: :meth:`print` prints a child
    in its context and brackets it when :func:`needs_brackets` says so.
    A concrete printer gives :meth:`rule` (the rule of the node, usually
    from a :class:`Dispatcher`) and :meth:`bracket` (how brackets are
    drawn)."""

    def rule(self, node: E, context: Context[T]) -> Result[T]:
        """The node printed in the context, without brackets around it."""
        raise NotImplementedError

    def bracket(self, result: Result[T], context: Context[T]) -> T:
        """The output of ``result`` in brackets."""
        raise NotImplementedError

    def enclose(self, result: Result[T], context: Context[T]) -> Result[T]:
        """A result put in a position: in brackets, an atom then, when
        :func:`needs_brackets` says so. For the rules which look at the
        result of a child before it is bracketed (a sum takes the term
        without its sign), and for results which are not those of a node."""
        if needs_brackets(result, context):
            return Result(self.bracket(result, context), Binding.ATOM)
        return result

    def print(self, node: E, context: Context[T]) -> Result[T]:
        """The node printed in the context, in brackets when its position
        requires them."""
        return self.enclose(self.rule(node, context), context)
