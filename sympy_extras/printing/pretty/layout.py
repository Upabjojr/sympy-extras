"""Two-dimensional pictures and the typed results and contexts of the
pretty printer.

A :class:`Picture` is an immutable block of text rows of equal width with
a baseline; pictures are put next to, above and below each other. It is
a typed port of the ``stringPict`` of SymPy's
``sympy/printing/pretty/stringpict.py``.

What SymPy's ``prettyForm`` adds to a picture, a *binding* which says
how the picture must be bracketed, is here a separate value, and it is
not SymPy's: SymPy stores the binding in the picture and many rules
build a composite picture without setting it, so that it stays ``ATOM``
and ``Rational(2, 3)**x`` prints ``2/3`` with the exponent ``x``
(sympy/sympy#30844, root cause 5). Here

* every rule returns a :class:`PrettyResult`, whose constructor requires
  the :class:`Binding` of the outermost operator *as printed* (a negated
  product binds as a negation, a stacked fraction as a quotient, ``sin²(x)``
  as a power): there is no default, so no composite picture is an atom
  by accident;
* every parent hands each child a :class:`Context`: the
  :class:`Position` of the child and the binding the position requires;
* :func:`needs_brackets` decides from the two, and it is the only place
  where the decision is taken.

These three types do not depend on pictures (the output of a
``PrettyResult`` aside) and are meant to be shared with the other
printers.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/stringpict.py (stringPict) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; pictures are immutable; the binding is not a
# field of the picture but of the typed result of a rule; Binding,
# Position, Context, PrettyResult and needs_brackets are original code.
# ---------------------------------------------------------------------
from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Iterator, Optional, Sequence

from .symbology import Symbology, center, line_width

__all__ = ['Picture', 'Binding', 'Position', 'Context', 'Applied', 'PrettyResult', 'needs_brackets', 'TOP']


class Binding(IntEnum):
    """How tightly the outermost operator of a printed form binds; a
    larger value binds more tightly."""

    #: nothing holds it together: ``x ↦ y``, a brace of cases
    OPEN = 0
    #: ``→``, ``⇔``
    IMPLIES = 10
    XOR = 15
    OR = 20
    AND = 30
    #: ``=``, ``<``, ``∈``
    RELATION = 35
    #: ``∪``, ``∩``, ``\``, ``×`` between sets
    SET = 37
    #: ``x mod y``
    MOD = 38
    #: a sum
    ADD = 40
    #: what starts with a minus sign: ``-x``, ``-2``, ``-x/2``; it is
    #: bracketed wherever a sum is, except as the operand of a sum sign
    NEG = 42
    #: a quotient written on one row: ``1/2``
    INLINE_QUOTIENT = 45
    #: a product, and what reads as one: a sum sign, an integral, a limit
    MUL = 50
    #: a fraction with a horizontal bar
    DIV = 55
    #: the one-character root sign ``√x``, which no bar closes
    ROOT = 58
    #: a power
    POW = 60
    #: ``¬x``, ``x!``
    AFFIX = 65
    #: a function application ``f(x)``
    FUNC = 70
    #: a single token or a form closed by its own delimiters
    ATOM = 100


class Position(Enum):
    """Where a child stands in its parent."""

    TOP = 'top'
    #: operands of an infix operator: the first, the last, the others
    INFIX_FIRST = 'infix-first'
    INFIX_MIDDLE = 'infix-middle'
    INFIX_LAST = 'infix-last'
    #: the operand of a prefix or postfix operator
    AFFIX_OPERAND = 'affix-operand'
    BASE = 'base'
    EXPONENT = 'exponent'
    SUBSCRIPT = 'subscript'
    NUMERATOR = 'numerator'
    DENOMINATOR = 'denominator'
    RADICAND = 'radicand'
    RELATION_LEFT = 'relation-left'
    RELATION_RIGHT = 'relation-right'
    #: an argument between the delimiters of its parent
    ARGUMENT = 'argument'
    #: the operand of a sum sign, a product sign, an integral sign, a limit
    BODY = 'body'
    #: the bound of a sum, a product or an integral
    BOUND = 'bound'
    #: the point of a limit, which carries the direction mark
    LIMIT_POINT = 'limit-point'
    #: the variable of an integral or a derivative, after the ``d``
    DIFFERENTIAL = 'differential'


@dataclass(frozen=True)
class Context:
    """What a parent tells a child: where it stands, what the place
    requires, and the inherited state. Contexts are immutable and the
    settings of the printer are never changed during the descent."""

    position: Position
    #: the child is bracketed when it binds less tightly than this
    minimum: Binding
    #: ... or exactly as tightly
    strict: bool = False
    #: a child which is open to the right (the body of a sum sign, ...)
    #: is bracketed: something follows it
    closed: bool = False
    #: a child which binds exactly like this is bracketed too (a power
    #: in an exponent)
    also: Optional[Binding] = None
    #: the rational numbers are stacked fractions (in a sum with tall terms)
    display: bool = False
    #: the number of ancestors
    depth: int = 0
    #: inside an exponent
    in_exponent: bool = False

    def child(self, position: Position, minimum: Binding, strict: bool = False, closed: bool = False,
              also: Optional[Binding] = None, display: bool = False) -> Context:
        """The context of a child of the node printed in this context."""
        return Context(position, minimum, strict, closed, also, display, self.depth + 1,
                       self.in_exponent or position is Position.EXPONENT)

    def argument(self) -> Context:
        """The context of a child between delimiters: nothing is required."""
        return self.child(Position.ARGUMENT, Binding.OPEN)

    def displayed(self) -> Context:
        """This context with the rational numbers stacked."""
        return replace(self, display=True)


#: the context of the expression given to the printer
TOP = Context(Position.TOP, Binding.OPEN)


class Picture:
    """An immutable block of text: rows of equal width, and the index of
    the row which is the baseline.

    Examples
    ========

    >>> from sympy_extras.printing.pretty.layout import Picture
    >>> half = Picture.text('1').below(Picture.text('-'), Picture.text('2')).with_baseline(1)
    >>> print(Picture.beside(Picture.text('10'), Picture.text(' + '), half))
         1
    10 + -
         2
    """

    __slots__ = ('rows', 'baseline')

    def __init__(self, rows: Sequence[str], baseline: int = 0) -> None:
        lines = list(rows) if rows else ['']
        width = max(line_width(line) for line in lines)
        self.rows: tuple[str, ...] = tuple(center(line, width) for line in lines)
        self.baseline: int = baseline

    @staticmethod
    def text(s: str, baseline: int = 0) -> Picture:
        """The picture of a string; the lines of a multiline string are centred."""
        return Picture(s.splitlines(), baseline)

    @staticmethod
    def blank(width: int, height: int = 1) -> Picture:
        """A block of spaces."""
        return Picture([' '*width]*height)

    def height(self) -> int:
        return len(self.rows)

    def width(self) -> int:
        return line_width(self.rows[0])

    def with_baseline(self, baseline: int) -> Picture:
        return Picture(self.rows, baseline)

    @staticmethod
    def beside(*pictures: Picture) -> Picture:
        """The pictures next to each other, their baselines aligned."""
        if not pictures:
            return Picture([''])
        baseline = max(p.baseline for p in pictures)
        below = max(p.height() - p.baseline for p in pictures)
        height = baseline + below
        columns: list[list[str]] = []
        for p in pictures:
            empty = [' '*p.width()]
            top = baseline - p.baseline
            columns.append(empty*top + list(p.rows) + empty*(height - p.height() - top))
        return Picture([''.join(column[i] for column in columns) for i in range(height)], baseline)

    def right(self, *pictures: Picture) -> Picture:
        """The pictures put at the right of this one."""
        return Picture.beside(self, *pictures)

    def left(self, *pictures: Picture) -> Picture:
        """The pictures (from left to right) put at the left of this one."""
        return Picture.beside(*pictures, self)

    def rtext(self, s: str) -> Picture:
        """This picture followed by a string on the baseline."""
        return Picture.beside(self, Picture.text(s))

    def ltext(self, s: str) -> Picture:
        """This picture preceded by a string on the baseline."""
        return Picture.beside(Picture.text(s), self)

    @staticmethod
    def stacked(pictures: Sequence[Picture]) -> list[str]:
        """The rows of the pictures on top of each other, centred."""
        width = max(p.width() for p in pictures)
        return [center(row, width) for p in pictures for row in p.rows]

    def below(self, *pictures: Picture) -> Picture:
        """The pictures put under this one, which keeps the baseline."""
        return Picture(Picture.stacked([self, *pictures]), self.baseline)

    def above(self, *pictures: Picture) -> Picture:
        """The pictures (from top to bottom) put over this one, which
        keeps the baseline."""
        rows = Picture.stacked([*pictures, self])
        return Picture(rows, len(rows) - self.height() + self.baseline)

    def parens(self, symbology: Symbology, left: str = '(', right: str = ')', one_row_in_ascii: bool = False
               ) -> Picture:
        """This picture between delimiters of its height (an empty string
        is no delimiter). ``one_row_in_ascii`` keeps the delimiters one
        row high in ASCII, where the tall braces are ugly."""
        height, baseline = self.height(), self.baseline
        if one_row_in_ascii and not symbology.unicode:
            height, baseline = 1, 0
        result = self
        if left:
            result = Picture.text(symbology.vobj(left, height), baseline).right(result)
        if right:
            result = result.right(Picture.text(symbology.vobj(right, height), baseline))
        return result

    def __str__(self) -> str:
        return '\n'.join(self.rows)

    def __repr__(self) -> str:
        return 'Picture(%r, %d)' % ('\n'.join(self.rows), self.baseline)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Picture) and other.rows == self.rows and other.baseline == self.baseline

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((self.rows, self.baseline))

    def render(self, wrap_line: bool, num_columns: Optional[int], unicode: bool) -> str:
        """The picture as a string, cut into screens of ``num_columns``
        columns (of the width of the terminal when ``None``) when it is
        wider and ``wrap_line`` is set; a cut is marked on both sides."""
        if not wrap_line:
            return '\n'.join(self.rows)
        columns = num_columns if num_columns is not None else shutil.get_terminal_size(fallback=(0, 0)).columns
        if columns <= 0:
            columns = 80
        if self.width() <= columns:
            return '\n'.join(self.rows)
        start, end = ('\N{RIGHTWARDS ARROW WITH HOOK} ', ' \N{RIGHTWARDS ARROW WITH HOOK}') if unicode else ('> ', ' >')
        screens = zip(*[_chunks(row, columns, start, end) for row in self.rows])
        return '\n\n'.join('\n'.join(screen) for screen in screens)


def _chunks(line: str, columns: int, overflow_start: str, overflow_end: str) -> Iterator[str]:
    """The consecutive pieces of ``line`` which fit ``columns`` columns."""
    prefix = ''
    width, start = line_width(prefix + overflow_end), 0
    for i, c in enumerate(line):
        w = line_width(c)
        # flush only when the character overflows: a combining mark can
        # still be appended at the full width
        if width + w > columns:
            yield prefix + line[start:i] + overflow_end
            prefix = overflow_start
            width, start = line_width(prefix + overflow_end), i
        width += w
    yield prefix + line[start:]


@dataclass(frozen=True)
class Applied:
    """The two parts of a function application printed as ``head(args)``:
    a power puts its exponent between them (``sin²(x)``)."""

    head: Picture
    arguments: Picture


@dataclass(frozen=True)
class PrettyResult:
    """What a rule returns: the picture and what a parent must know
    about it. ``binding`` has no default: it is the binding of the
    operator which produced the picture."""

    picture: Picture
    binding: Binding
    #: the picture without its leading minus sign, when the picture is
    #: the negation of it: a sum writes ``a - m`` instead of ``a + -m``
    magnitude: Optional[Picture] = None
    #: the picture ends with the operand of a sum sign, a product sign,
    #: an integral or a limit: what is written after it is read as part
    #: of that operand
    open_right: bool = False
    #: the parts of a function application
    applied: Optional[Applied] = None
    #: the outermost layout is a fraction bar (also under a minus sign):
    #: as a part of another fraction it is bracketed, two bars of the
    #: same length do not say which one is the main one
    stacked: bool = False


def needs_brackets(result: PrettyResult, context: Context) -> bool:
    """Whether a child printed as ``result`` is bracketed in ``context``.
    This is the only place where brackets are decided.

    Examples
    ========

    >>> from sympy_extras.printing.pretty.layout import (Binding, Context, Picture, Position, PrettyResult,
    ...     needs_brackets)
    >>> minus_x = PrettyResult(Picture.text('-x'), Binding.NEG)
    >>> needs_brackets(minus_x, Context(Position.BASE, Binding.POW, strict=True))
    True
    >>> needs_brackets(minus_x, Context(Position.INFIX_FIRST, Binding.ADD))
    False
    """
    if result.binding < context.minimum:
        return True
    if context.strict and result.binding == context.minimum:
        return True
    if context.also is not None and result.binding == context.also:
        return True
    if result.stacked and context.position in (Position.NUMERATOR, Position.DENOMINATOR):
        return True
    return context.closed and result.open_right
