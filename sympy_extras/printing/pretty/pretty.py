"""``pretty`` and ``pprint``: expressions as two-dimensional text.

``pretty(expr)`` draws ``expr`` the way it is written by hand, with
fraction bars, raised exponents, root signs, sum and integral signs, in
Unicode or in ASCII. It is a typed port of SymPy's pretty printer
(``sympy/printing/pretty``) with another design of the dispatch and of
the bracketing (see :mod:`~sympy_extras.printing.pretty.engine` and
:mod:`~sympy_extras.printing.pretty.layout`), which removes the defects
of sympy/sympy#30844 by construction.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (pretty, pretty_print, the printing of the Python
# containers and of matrices), sympy/printing/pretty/pretty_symbology.py (pretty_use_unicode,
# pretty_try_use_unicode) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; typed settings; the default of ``use_unicode``
# is the only module state, read once when a printer is made.
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from typing import AbstractSet, Mapping, Optional, Sequence, Union

from ...core.basic import Basic, TextSink
from ...core.sorting import default_sort_key
from ...core.sympify import sympify
from ...matrices.dense import ExprMatrix
from ...polys.polytools import Poly
from .engine import PrettyPrinter, PrettyRules, PrettySettings, rule_table
from ..framework import Binding
from .layout import TOP, Picture, PrettyContext, PrettyResult
from .rules_calculus import CALCULUS_RULES
from .rules_core import CORE_RULES, integer_result
from .rules_functions import FUNCTION_RULES
from .rules_other import OTHER_RULES
from .rules_sets import SET_RULES
from .symbology import greek_unicode

__all__ = ['pretty', 'pprint', 'pretty_print', 'pretty_picture', 'pretty_use_unicode', 'pretty_try_use_unicode',
           'pretty_repr', 'Printable', 'TextSink', 'RULES', 'matrix_picture']

#: every rule of the pretty printer
RULES: PrettyRules = rule_table(CORE_RULES + CALCULUS_RULES + FUNCTION_RULES + SET_RULES + OTHER_RULES)

#: what can be pretty printed: a node, a matrix, a polynomial, and the
#: Python numbers, strings and containers of them
Printable = Union[Basic, ExprMatrix, Poly, bool, int, float, str, None, Sequence['Printable'],
                  Mapping['Printable', 'Printable'], AbstractSet['Printable']]


class _Defaults:
    """The default of ``use_unicode`` (what ``init_printing`` sets)."""

    def __init__(self) -> None:
        self.use_unicode: bool = False


_defaults = _Defaults()


def pretty_use_unicode(flag: Optional[bool] = None) -> bool:
    """Set whether the pretty printer uses Unicode when ``use_unicode``
    is not given; returns the previous value (the current one for
    ``None``).

    Examples
    ========

    >>> from sympy_extras.printing.pretty import pretty_use_unicode
    >>> pretty_use_unicode()
    False
    """
    previous = _defaults.use_unicode
    if flag is not None:
        _defaults.use_unicode = flag
    return previous


def pretty_try_use_unicode() -> None:
    """Make Unicode the default when the standard output can encode the
    characters of the printer."""
    encoding = sys.stdout.encoding
    if encoding is None:
        return
    try:
        ''.join(greek_unicode.values()).encode(encoding)
        '\N{DOUBLE-STRUCK ITALIC SMALL I}\N{LEFT PARENTHESIS UPPER HOOK}\N{INTEGRAL EXTENSION}'.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return
    _defaults.use_unicode = True


def _sort_key(item: Printable) -> tuple[int, str]:
    """Python sets and dictionaries are printed in a fixed order."""
    if isinstance(item, Basic):
        return (0, repr(default_sort_key(item)))
    return (1, repr(item))


def matrix_picture(p: PrettyPrinter, rows: Sequence[Sequence[Picture]]) -> Picture:
    """The cells in a grid: the columns two spaces apart, the rows one
    row apart, each cell centred in its column."""
    if not rows or not rows[0]:
        return Picture.text('')
    columns = len(rows[0])
    widths = [max(row[j].width() for row in rows) for j in range(columns)]
    grid: Optional[Picture] = None
    for row in rows:
        cells: list[Picture] = []
        for j, cell in enumerate(row):
            delta = widths[j] - cell.width()
            left = delta//2
            cells.append(Picture.beside(Picture.blank(left), cell, Picture.blank(delta - left)))
        line = p.join(cells, '  ')
        grid = line if grid is None else grid.below(Picture.text(' '), line)
    if grid is None:
        return Picture.text('')
    return grid.with_baseline(grid.height()//2)


def _printable(p: PrettyPrinter, obj: Printable, context: PrettyContext) -> PrettyResult:
    if isinstance(obj, Basic):
        return p.rule(obj, context)
    if isinstance(obj, ExprMatrix):
        if obj.rows == 0 or obj.cols == 0:
            return p.atom('[]')
        cells = [[p.operand(e, context.delimited()) for e in obj.row(i)] for i in range(obj.rows)]
        return PrettyResult(p.parens(matrix_picture(p, cells), '[', ']'), Binding.ATOM)
    if isinstance(obj, Poly):
        parts = [p.operand(obj.as_expr(), context.delimited())]
        parts.extend(p.operand(g, context.delimited()) for g in obj.gens)
        parts.append(Picture.text("domain='%s'" % obj.domain))
        return p.applied(Picture.text('Poly'), p.join(parts, ', '), context)
    if obj is None or isinstance(obj, bool):
        return p.atom(str(obj))
    if isinstance(obj, int):
        return integer_result(obj)
    if isinstance(obj, float):
        if obj != obj or obj in (float('inf'), float('-inf')):
            # (the infinity sign and nan of the nodes, as SymPy; these printed as Python's inf and nan)
            return p.rule(sympify(obj), context)
        return p.opaque(repr(obj))
    if isinstance(obj, str):
        # (as SymPy: a string is printed as it is)
        return PrettyResult(Picture.text(obj), Binding.ATOM if obj.isalnum() else Binding.LOWEST)
    if isinstance(obj, Mapping):
        items: list[Picture] = []
        for key in sorted(obj, key=_sort_key):
            items.append(Picture.beside(_closed(p, key, context), Picture.text(': '), _closed(p, obj[key], context)))
        return PrettyResult(p.parens(p.join(items, ', '), '{', '}', True), Binding.ATOM)
    if isinstance(obj, AbstractSet):
        if not obj:
            return p.atom('frozenset()' if isinstance(obj, frozenset) else 'set()')
        inside = p.parens(p.join([_closed(p, item, context) for item in sorted(obj, key=_sort_key)], ', '),
                          '{', '}', True)
        if isinstance(obj, frozenset):
            return PrettyResult(p.parens(inside, '(', ')', True).ltext('frozenset'), Binding.APPLICATION)
        return PrettyResult(inside, Binding.ATOM)
    pictures = [_closed(p, item, context) for item in obj]
    if isinstance(obj, tuple):
        if len(pictures) == 1:
            return PrettyResult(p.parens(pictures[0].rtext(','), '(', ')', True), Binding.ATOM)
        return PrettyResult(p.parens(p.join(pictures, ', '), '(', ')', True), Binding.ATOM)
    return PrettyResult(p.parens(p.join(pictures, ', '), '[', ']', True), Binding.ATOM)


def _closed(p: PrettyPrinter, obj: Printable, context: PrettyContext) -> Picture:
    c = context.delimited()
    return p.enclose(_printable(p, obj, c), c).output


def _settings(use_unicode: Optional[bool], wrap_line: bool, num_columns: Optional[int], order: Optional[str],
              root_notation: bool, use_unicode_sqrt_char: bool, imaginary_unit: str,
              full_prec: Optional[bool]) -> PrettySettings:
    return PrettySettings(use_unicode=_defaults.use_unicode if use_unicode is None else use_unicode,
                          wrap_line=wrap_line, num_columns=num_columns, order=order, root_notation=root_notation,
                          use_unicode_sqrt_char=use_unicode_sqrt_char, imaginary_unit=imaginary_unit,
                          full_prec=full_prec)


def pretty_picture(expr: Printable, settings: PrettySettings) -> Picture:
    """The picture of ``expr`` (see :func:`pretty`), not cut into screens.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.pretty import pretty_picture, PrettySettings
    >>> x, y = symbols('x y')
    >>> picture = pretty_picture(x/y, PrettySettings())
    >>> picture.rows, picture.baseline
    (('x', '-', 'y'), 1)
    """
    return _printable(PrettyPrinter(settings, RULES), expr, TOP).output


def pretty(expr: Printable, use_unicode: Optional[bool] = None, wrap_line: bool = True,
           num_columns: Optional[int] = None, order: Optional[str] = None, root_notation: bool = True,
           use_unicode_sqrt_char: bool = True, imaginary_unit: str = 'i', full_prec: Optional[bool] = None) -> str:
    """``expr`` as two-dimensional text.

    ``use_unicode`` chooses between Unicode and ASCII (``None``: the
    default set by :func:`pretty_use_unicode`, ASCII at the start);
    ``wrap_line`` cuts a picture wider than ``num_columns`` columns (the
    width of the terminal for ``None``) into screens; ``order`` is the
    order of the terms of the sums, as in SymPy (``'lex'``, ``'grlex'``,
    ``'grevlex'``, these after ``'rev-'``, ``'old'``; see
    :func:`sympy_extras.core.printing.printed_terms`), and ``order='none'``
    keeps the terms of the sums and the factors of the products in the
    order of the arguments; ``root_notation=False`` writes ``x**(1/n)``
    as a power instead of a root; ``use_unicode_sqrt_char=False`` draws
    the root sign of a square root always with its bar; ``imaginary_unit``
    is the letter of the imaginary unit in Unicode, ``'i'`` or ``'j'``;
    ``full_prec`` prints the floats with all the digits of their precision
    (``True``), without the trailing zeros (``False``), or with all the
    digits when the float is the whole expression (``None``, SymPy's
    ``'auto'``).
    The rows of the result have the same width (they end with spaces).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Rational, Pow, Not
    >>> from sympy_extras.core.mod import Mod
    >>> from sympy_extras.printing.pretty import pretty
    >>> x, y, n = symbols('x y n')
    >>> print(pretty((x**2 + 1)/(y - 1)))  # doctest: +NORMALIZE_WHITESPACE
     2
    x  + 1
    ------
    y - 1
    >>> print(pretty(x**Rational(1, 3) + 1/x, use_unicode=True))  # doctest: +NORMALIZE_WHITESPACE
    3 ___   1
    ╲╱ x  + ─
            x

    The brackets follow what is printed: a fraction on one row, a negative
    number, a relation and a remainder are not atoms.

    >>> print(pretty(Pow(Rational(2, 3), n)))  # doctest: +NORMALIZE_WHITESPACE
         n
    (2/3)
    >>> print(pretty(Not(x < y), use_unicode=True))
    ¬(x < y)
    >>> print(pretty(Mod(x, y + n)), '|', pretty(-Mod(x, y)))
    x mod (n + y) | -(x mod y)
    """
    settings = _settings(use_unicode, wrap_line, num_columns, order, root_notation, use_unicode_sqrt_char,
                         imaginary_unit, full_prec)
    return pretty_picture(expr, settings).render(settings.wrap_line, settings.num_columns, settings.use_unicode)


def pretty_print(expr: Printable, use_unicode: Optional[bool] = None, wrap_line: bool = True,
                 num_columns: Optional[int] = None, order: Optional[str] = None, root_notation: bool = True,
                 use_unicode_sqrt_char: bool = True, imaginary_unit: str = 'i', full_prec: Optional[bool] = None
                 ) -> None:
    """Print :func:`pretty` of ``expr`` (``pprint`` is the same function).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.pretty import pprint
    >>> x = symbols('x')
    >>> pprint(1/(x + 1))  # doctest: +NORMALIZE_WHITESPACE
      1
    -----
    x + 1
    """
    print(pretty(expr, use_unicode, wrap_line, num_columns, order, root_notation, use_unicode_sqrt_char,
                 imaginary_unit, full_prec))


pprint = pretty_print


def pretty_repr(expr: Printable, sink: TextSink, cycle: bool, settings: Optional[PrettySettings] = None) -> None:
    """The body of a ``_repr_pretty_(self, p, cycle)`` method, which
    ``init_printing`` installs for IPython: writes the pretty form of
    ``expr`` to ``sink``. A picture of several rows starts on a new
    line, so that its first row is not shifted by the prompt.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.pretty import pretty_repr
    >>> class Sink:
    ...     def text(self, s: str) -> None:
    ...         print(s)
    >>> x = symbols('x')
    >>> pretty_repr(x**2, Sink(), False)  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
     2
    x
    """
    if cycle:
        sink.text('...')
        return
    chosen = settings if settings is not None else PrettySettings(use_unicode=_defaults.use_unicode)
    text = pretty_picture(expr, chosen).render(chosen.wrap_line, chosen.num_columns, chosen.use_unicode)
    sink.text('\n' + text if '\n' in text else text)
