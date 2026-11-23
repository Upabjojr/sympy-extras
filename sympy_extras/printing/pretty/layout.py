"""Two-dimensional pictures: the output of the pretty printer.

A :class:`Picture` is an immutable block of text rows of equal width with
a baseline; pictures are put next to, above and below each other. It is
a typed port of the ``stringPict`` of SymPy's
``sympy/printing/pretty/stringpict.py``.

What SymPy's ``prettyForm`` adds to a picture, a *binding* which says
how the picture must be bracketed, is not a field of the picture here:
SymPy stores the binding in the picture and many rules build a composite
picture without setting it, so that it stays ``ATOM`` and
``Rational(2, 3)**x`` prints ``2/3`` with the exponent ``x``
(sympy/sympy#30844, root cause 5). The pretty printer uses the typed
framework which it shares with the LaTeX printer
(:mod:`sympy_extras.printing.framework`):

* every rule returns a :class:`~sympy_extras.printing.framework.Result`
  of a picture (:data:`PrettyResult`), whose constructor requires the
  :class:`~sympy_extras.printing.framework.Binding` of the outermost
  operator *as printed* (a negated product binds as a negation, a stacked
  fraction as a fraction, ``sin²(x)`` as a power): there is no default,
  so no composite picture is an atom by accident;
* every parent hands each child a
  :class:`~sympy_extras.printing.framework.Context`
  (:data:`PrettyContext`): the position of the child and the binding the
  position requires;
* :func:`~sympy_extras.printing.framework.needs_brackets` decides from
  the two, and it is the only place where the decision is taken.

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
# field of the picture but of the typed result of a rule.
# ---------------------------------------------------------------------
from __future__ import annotations

import shutil
from typing import Iterator, Optional, Sequence

from ..framework import Context, Result
from .symbology import Symbology, center, line_width

__all__ = ['Picture', 'PrettyResult', 'PrettyContext', 'TOP']


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


#: what a rule of the pretty printer returns: the picture with the
#: binding of its outermost operator as printed (``output`` is the
#: picture, ``unsigned`` the picture without its leading minus sign,
#: ``open_ended`` says that it ends with the operand of a sum sign, a
#: product sign, an integral or a limit, ``stacked`` that its outermost
#: layout is a fraction bar)
PrettyResult = Result[Picture]

#: what a parent hands a child
PrettyContext = Context[Picture]

#: the context of the expression given to the printer
TOP: PrettyContext = Context()
