"""The pretty printer: typed dispatch, the one place where brackets are
inserted, and the layouts shared by the rules.

Design
======

SymPy's ``PrettyPrinter`` finds the method ``_print_<ClassName>`` by the
name of the class and lets each parent decide on brackets from the class
of the child. Here

* the dispatch is by class: a table of :class:`RuleEntry` (a class and
  the rule which prints its instances) is searched with ``isinstance``,
  the most specific class winning; in a statically typed language it is
  a ``match`` on the kind of the node. There is no lookup by name;
* a rule is a function ``(printer, node, context) -> PrettyResult``: it
  receives the :class:`~.layout.Context` its parent made for it and
  returns the picture with the binding of its outermost operator;
* a rule prints a child with :meth:`PrettyPrinter.operand`, giving the
  context of the child, and :meth:`PrettyPrinter.bracketed` is the only
  code which adds brackets, when :func:`~.layout.needs_brackets` says so;
* the settings are immutable.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (the layouts: _print_nth_root, _print_seq,
# _hprint_vseparator, prettyForm.__pow__, prettyForm.__truediv__) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; dispatch by class instead of by name; typed
# results and contexts; brackets decided in one place.
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TypeVar

from ...core.basic import Basic
from ...core.printing import sstr
from .layout import Applied, Binding, Context, Picture, PrettyResult, needs_brackets
from .symbology import Symbology

__all__ = ['PrettySettings', 'PrettyPrinter', 'RuleEntry', 'Rule', 'rule_for']


@dataclass(frozen=True)
class PrettySettings:
    """The settings of a pretty printer (see :func:`~.pretty.pretty`)."""

    use_unicode: bool = False
    wrap_line: bool = True
    num_columns: Optional[int] = None
    #: ``None`` for the printing order, ``'none'`` for the order of the arguments
    order: Optional[str] = None
    root_notation: bool = True
    use_unicode_sqrt_char: bool = True
    imaginary_unit: str = 'i'

    def __post_init__(self) -> None:
        if self.imaginary_unit not in ('i', 'j'):
            raise ValueError("'imaginary_unit' must be either 'i' or 'j', not '%s'" % self.imaginary_unit)
        if self.order not in (None, 'lex', 'none'):
            raise ValueError("'order' must be None, 'lex' or 'none', not '%s'" % self.order)


Rule = Callable[['PrettyPrinter', Basic, Context], PrettyResult]


@dataclass(frozen=True)
class RuleEntry:
    """A class and the rule which prints its instances."""

    cls: type[Basic]
    rule: Rule


N = TypeVar('N', bound=Basic)


def rule_for(cls: type[N], rule: Callable[[PrettyPrinter, N, Context], PrettyResult]) -> RuleEntry:
    """The table entry of a rule written for the class ``cls``."""
    def narrowed(printer: PrettyPrinter, e: Basic, context: Context) -> PrettyResult:
        if not isinstance(e, cls):
            raise TypeError('%s is not a %s' % (type(e).__name__, cls.__name__))
        return rule(printer, e, context)
    return RuleEntry(cls, narrowed)


class PrettyPrinter:
    """Prints nodes as pictures with the rules of a table.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.pretty import PrettyPrinter, PrettySettings, RULES, TOP
    >>> x, y = symbols('x y')
    >>> result = PrettyPrinter(PrettySettings(), RULES).result(-x*y, TOP)
    >>> str(result.picture), result.binding.name, str(result.magnitude)
    ('-x*y', 'NEG', 'x*y')
    """

    def __init__(self, settings: PrettySettings, rules: Sequence[RuleEntry]) -> None:
        self.settings: PrettySettings = settings
        self.symbology: Symbology = Symbology(settings.use_unicode, settings.imaginary_unit)
        self._rules: tuple[RuleEntry, ...] = tuple(rules)
        self._by_class: dict[type[Basic], Optional[Rule]] = {}

    @property
    def unicode(self) -> bool:
        return self.settings.use_unicode

    # ------------------------------------------------------------------
    # dispatch

    def _rule(self, cls: type[Basic]) -> Optional[Rule]:
        if cls in self._by_class:
            return self._by_class[cls]
        best: Optional[RuleEntry] = None
        for entry in self._rules:
            if issubclass(cls, entry.cls) and (best is None or (entry.cls is not best.cls
                                                                 and issubclass(entry.cls, best.cls))):
                best = entry
        rule = None if best is None else best.rule
        self._by_class[cls] = rule
        return rule

    def has_rule(self, cls: type[Basic]) -> bool:
        """Whether the table has a rule for the instances of ``cls``
        (they are printed by :meth:`generic` otherwise)."""
        return self._rule(cls) is not None

    def result(self, e: Basic, context: Context) -> PrettyResult:
        """``e`` printed in ``context``, without the brackets the context
        may require (see :meth:`operand`)."""
        rule = self._rule(type(e))
        if rule is None:
            return self.generic(e, context)
        return rule(self, e, context)

    def generic(self, e: Basic, context: Context) -> PrettyResult:
        """A node without a rule: its string form (what SymPy's printer
        does too), on one row."""
        return self.opaque(sstr(e))

    def opaque(self, text: str) -> PrettyResult:
        """A string of unknown structure: an atom when it is one token,
        an application when it reads ``name(...)``, and otherwise open:
        it is then bracketed wherever it is an operand."""
        if _is_token(text):
            return PrettyResult(Picture.text(text), Binding.ATOM)
        return PrettyResult(Picture.text(text), Binding.FUNC if _is_application(text) else Binding.OPEN)

    # ------------------------------------------------------------------
    # brackets: the only place

    def bracketed(self, result: PrettyResult, context: Context) -> PrettyResult:
        """``result``, between parentheses when its place requires them."""
        if not needs_brackets(result, context):
            return result
        return PrettyResult(result.picture.parens(self.symbology), Binding.ATOM)

    def operand(self, e: Basic, context: Context) -> Picture:
        """The picture of the child ``e`` in ``context``, bracketed if needed."""
        return self.bracketed(self.result(e, context), context).picture

    # ------------------------------------------------------------------
    # layouts

    def text(self, s: str) -> Picture:
        return Picture.text(s)

    def atom(self, s: str) -> PrettyResult:
        """A single token."""
        return PrettyResult(Picture.text(s), Binding.ATOM)

    def join(self, pictures: Sequence[Picture], delimiter: str) -> Picture:
        """The pictures in a row, separated by ``delimiter``."""
        parts: list[Picture] = []
        for p in pictures:
            if parts:
                parts.append(Picture.text(delimiter))
            parts.append(p)
        return Picture.beside(*parts)

    def parens(self, picture: Picture, left: str = '(', right: str = ')', one_row_in_ascii: bool = False) -> Picture:
        """``picture`` between the delimiters of a notation (the
        parentheses of a function call, the bars of an absolute value):
        these are not the brackets of precedence."""
        return picture.parens(self.symbology, left, right, one_row_in_ascii)

    def arguments(self, args: Sequence[Basic], context: Context, delimiter: str = ', ') -> Picture:
        """The arguments in a row, each one in the context of an argument."""
        return self.join([self.operand(a, context.argument()) for a in args], delimiter)

    def sequence(self, args: Sequence[Basic], context: Context, left: str, right: str, delimiter: str = ', '
                 ) -> PrettyResult:
        """A delimited sequence: ``(a, b)``, ``[a, b]``, ``{a, b}``."""
        return PrettyResult(self.parens(self.arguments(args, context, delimiter), left, right, True), Binding.ATOM)

    def applied(self, head: Picture, arguments: Picture) -> PrettyResult:
        """``head(arguments)``: a function application, which a power
        may print as ``head^n(arguments)``."""
        inside = self.parens(arguments)
        return PrettyResult(Picture.beside(head, inside), Binding.FUNC, applied=Applied(head, inside))

    def function(self, name: str, args: Sequence[Basic], context: Context) -> PrettyResult:
        """``name(args)``."""
        return self.applied(Picture.text(name), self.arguments(args, context))

    def fraction(self, numerator: Picture, denominator: Picture) -> Picture:
        """A fraction with a horizontal bar."""
        width = max(numerator.width(), denominator.width())
        bar = Picture.text(self.symbology.hobj('-', width))
        return Picture(Picture.stacked([numerator, bar, denominator]), numerator.height())

    def power(self, base: Picture, exponent: Picture) -> Picture:
        """``base`` with ``exponent`` raised at its right."""
        top = Picture.beside(Picture.blank(base.width()), exponent)
        bottom = Picture.beside(base, Picture.blank(exponent.width()))
        return bottom.above(top)

    def subscripted(self, base: Picture, subscript: Picture) -> Picture:
        """``base`` with ``subscript`` lowered at its right."""
        bottom = Picture.beside(Picture.blank(base.width()), subscript)
        top = Picture.beside(base, Picture.blank(subscript.width()))
        return top.below(bottom)

    def separated(self, left_part: Picture, right_part: Picture, left: str, right: str) -> Picture:
        """``left_part | right_part`` between delimiters, the bar as tall
        as the parts (one row high in ASCII when the delimiters are)."""
        if left and not self.unicode:
            row = Picture.beside(left_part, Picture.text(' '), Picture.text('|'), Picture.text(' '), right_part)
            return self.parens(row, left, right, True)
        spacer = ' ' if left else ''
        probe = Picture.beside(left_part, Picture.text(spacer), right_part)
        if left:
            # the bar is as tall as the delimiters (a brace has an odd height)
            probe = self.parens(probe, left, right, True)
        bar = Picture.text(self.symbology.vobj('|', probe.height()), probe.baseline)
        row = Picture.beside(left_part, Picture.text(spacer), bar, Picture.text(spacer), right_part)
        return self.parens(row, left, right, True) if left else row

    # PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_nth_root
    def root(self, radicand: Picture, index: Optional[Picture], small: bool) -> PrettyResult:
        """The root sign over ``radicand``; ``index`` is ``None`` for a
        square root. ``small`` allows the one-character sign ``√`` (the
        caller says whether the radicand is simple enough)."""
        symbology = self.symbology
        if index is None and small and self.settings.use_unicode_sqrt_char and radicand.height() == 1:
            sign = symbology.root(2)
            if sign is not None:
                # no bar closes the radicand: it binds like a prefix operator
                return PrettyResult(radicand.ltext(sign), Binding.ROOT)
        slash = symbology.xobj('/', 1)[0]
        rootsign = symbology.xobj('\\', 1)[0] + slash
        exp = '' if index is None else str(index).ljust(2)
        if len(exp) > 2:
            rootsign = ' '*(len(exp) - 2) + rootsign
        drawn = Picture.text(exp + '\n' + rootsign, 0) if exp else Picture(['', rootsign], 0)
        length = radicand.height() - 1
        diagonal = Picture([' '*(length - i - 1) + slash + ' '*i for i in range(length)], length - 1)
        drawn = drawn.right(diagonal).with_baseline(max(1, radicand.baseline))
        bar = Picture.text(symbology.hobj('_', 2 + radicand.width()))
        return PrettyResult(radicand.above(bar).left(drawn), Binding.ATOM)


def _is_token(text: str) -> bool:
    return text != '' and all(c.isalnum() or c == '_' for c in text)


def _is_application(text: str) -> bool:
    """Whether ``text`` is a name followed by one bracketed group: ``f(x, (y))``, not ``f(x) + g(y)``."""
    start = text.find('(')
    if start <= 0 or not _is_token(text[:start]) or not text.endswith(')'):
        return False
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i == len(text) - 1
    return False
