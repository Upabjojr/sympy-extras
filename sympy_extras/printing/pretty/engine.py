"""The pretty printer on the typed framework of the printers, and the
layouts shared by its rules.

Design
======

SymPy's ``PrettyPrinter`` finds the method ``_print_<ClassName>`` by the
name of the class and lets each parent decide on brackets from the class
of the child. This printer is a
:class:`~sympy_extras.printing.framework.TreePrinter` of pictures, as the
LaTeX printer is one of strings:

* the dispatch is by class: the rules are in a
  :class:`~sympy_extras.printing.framework.Dispatcher`, filled from tables
  of :class:`RuleEntry` (a class and the rule which prints its
  instances); in a statically typed language it is a ``match`` on the
  kind of the node. There is no lookup by name;
* a rule is a function ``(printer, node, context) -> PrettyResult``: it
  receives the context its parent made for it and returns the picture
  with the binding of its outermost operator;
* a rule prints a child with :meth:`PrettyPrinter.operand`, giving the
  context of the child; the brackets are added by the
  :meth:`~sympy_extras.printing.framework.TreePrinter.enclose` of the
  framework, when its ``needs_brackets`` says so, and nowhere else;
* a function application takes the exponent which its parent offers in
  the context (``sin²(x)``), a connective the negation, as in the LaTeX
  printer;
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

from ...core.basic import Basic, ForeignPrint, PrintSetting
from ...core.printing import check_print_order
from ...core.printing import sstr
from ..framework import Binding, Dispatcher, TreePrinter, binding_named
from .layout import Picture, PrettyContext, PrettyResult
from .symbology import Symbology

__all__ = ['PrettySettings', 'PrettyPrinter', 'PrettyRules', 'RuleEntry', 'Rule', 'rule_for', 'rule_table']


@dataclass(frozen=True)
class PrettySettings:
    """The settings of a pretty printer (see :func:`~.pretty.pretty`)."""

    use_unicode: bool = False
    wrap_line: bool = True
    num_columns: Optional[int] = None
    #: the order of the terms of the sums (``core.printing.printed_terms``): ``None`` for the printing order,
    #: ``'none'`` for the order of the arguments, ``'grlex'``, ``'rev-lex'``, ``'old'``, ... as in SymPy
    order: Optional[str] = None
    root_notation: bool = True
    use_unicode_sqrt_char: bool = True
    imaginary_unit: str = 'i'
    #: the floats with all the digits of their precision; ``None`` (SymPy's
    #: ``'auto'``) for a float which is the whole expression only
    full_prec: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.imaginary_unit not in ('i', 'j'):
            raise ValueError("'imaginary_unit' must be either 'i' or 'j', not '%s'" % self.imaginary_unit)
        check_print_order(self.order)


Rule = Callable[['PrettyPrinter', Basic, PrettyContext], PrettyResult]

#: the rules of a pretty printer by the class of the node
PrettyRules = Dispatcher['PrettyPrinter', Basic, Picture]


@dataclass(frozen=True)
class RuleEntry:
    """A class and how the rule which prints its instances is put into a
    table of rules."""

    cls: type[Basic]
    install: Callable[[PrettyRules], None]


N = TypeVar('N', bound=Basic)


def rule_for(cls: type[N], rule: Callable[[PrettyPrinter, N, PrettyContext], PrettyResult]) -> RuleEntry:
    """The entry of a rule written for the class ``cls``."""
    def install(table: PrettyRules) -> None:
        table.register(cls, rule)
    return RuleEntry(cls, install)


def rule_table(entries: Sequence[RuleEntry]) -> PrettyRules:
    """The dispatcher of the rules of ``entries``.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Symbol
    >>> from sympy_extras.printing.pretty.engine import PrettyPrinter, PrettySettings, rule_for, rule_table
    >>> from sympy_extras.printing.pretty.layout import TOP
    >>> rules = rule_table([rule_for(Symbol, lambda p, e, context: p.atom('<%s>' % e.name))])
    >>> printer = PrettyPrinter(PrettySettings(), rules)
    >>> str(printer.operand(Symbol('x'), TOP)), printer.has_rule(Integer)
    ('<x>', False)
    """
    table: PrettyRules = Dispatcher()
    for entry in entries:
        entry.install(table)
    return table


class PrettyPrinter(TreePrinter[Basic, Picture]):
    """Prints nodes as pictures with the rules of a table.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.pretty import PrettyPrinter, PrettySettings, RULES, TOP
    >>> x, y = symbols('x y')
    >>> result = PrettyPrinter(PrettySettings(), RULES).rule(-x*y, TOP)
    >>> str(result.output), result.binding.name, str(result.unsigned)
    ('-x*y', 'NEGATION', 'x*y')
    """

    def __init__(self, settings: PrettySettings, rules: PrettyRules) -> None:
        self.settings: PrettySettings = settings
        self.symbology: Symbology = Symbology(settings.use_unicode, settings.imaginary_unit)
        self._rules: PrettyRules = rules
        #: the settings handed to the system of a wrapped foreign object (Basic._foreign_pretty)
        self.foreign_settings: dict[str, PrintSetting] = {
            'use_unicode': settings.use_unicode, 'order': settings.order, 'root_notation': settings.root_notation,
            'use_unicode_sqrt_char': settings.use_unicode_sqrt_char, 'imaginary_unit': settings.imaginary_unit}

    @property
    def unicode(self) -> bool:
        return self.settings.use_unicode

    # ------------------------------------------------------------------
    # dispatch

    def has_rule(self, cls: type[Basic]) -> bool:
        """Whether the table has a rule for the instances of ``cls``
        (they are printed by :meth:`generic` otherwise)."""
        return self._rules.lookup(cls) is not None

    def rule(self, node: Basic, context: PrettyContext) -> PrettyResult:
        """``node`` printed in ``context``, without the brackets the
        context may require (see :meth:`operand`)."""
        foreign = node._foreign_pretty(self.foreign_settings)
        if foreign is not None:
            return self.foreign(foreign)
        found = self._rules.lookup(type(node))
        if found is None:
            return self.generic(node, context)
        return found(self, node, context)

    def foreign(self, printed: ForeignPrint) -> PrettyResult:
        """The picture which the system of a wrapped foreign object drew
        for it (SymPy's, for the nodes of :mod:`sympy_extras.interop`),
        binding as that system says."""
        picture = Picture(printed.rows, printed.baseline)
        binding = binding_named(printed.binding)
        return PrettyResult(picture, binding, stacked=binding == Binding.FRACTION and picture.height() > 1)

    def generic(self, e: Basic, context: PrettyContext) -> PrettyResult:
        """A node without a rule: its string form (what SymPy's printer
        does too), on one row."""
        return self.opaque(sstr(e))

    def opaque(self, text: str) -> PrettyResult:
        """A string of unknown structure: an atom when it is one token,
        an application when it reads ``name(...)``, and otherwise open:
        it is then bracketed wherever it is an operand."""
        if _is_token(text):
            return PrettyResult(Picture.text(text), Binding.ATOM)
        return PrettyResult(Picture.text(text), Binding.APPLICATION if _is_application(text) else Binding.LOWEST)

    # ------------------------------------------------------------------
    # brackets: drawn here, decided by the framework

    def bracket(self, result: PrettyResult, context: PrettyContext) -> Picture:
        """The picture of ``result`` between parentheses."""
        return result.output.parens(self.symbology)

    def operand(self, e: Basic, context: PrettyContext) -> Picture:
        """The picture of the child ``e`` in ``context``, bracketed if needed."""
        return self.print(e, context).output

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

    def arguments(self, args: Sequence[Basic], context: PrettyContext, delimiter: str = ', ') -> Picture:
        """The arguments in a row, each one delimited by its parent."""
        return self.join([self.operand(a, context.delimited()) for a in args], delimiter)

    def sequence(self, args: Sequence[Basic], context: PrettyContext, left: str, right: str, delimiter: str = ', '
                 ) -> PrettyResult:
        """A delimited sequence: ``(a, b)``, ``[a, b]``, ``{a, b}``."""
        return PrettyResult(self.parens(self.arguments(args, context, delimiter), left, right, True), Binding.ATOM)

    def applied(self, head: Picture, arguments: Picture, context: PrettyContext) -> PrettyResult:
        """``head(arguments)``: a function application. The exponent
        which the parent offers in ``context`` goes between the function
        and its arguments (``sin²(x)``); the result then stands for the
        power and binds as one: nothing is attached to it."""
        inside = self.parens(arguments)
        exponent = context.exponent_offered
        if exponent is not None:
            raised = exponent.with_baseline(head.baseline + exponent.height())
            return PrettyResult(Picture.beside(head, raised, inside), Binding.POWER, absorbed=True)
        return PrettyResult(Picture.beside(head, inside), Binding.APPLICATION)

    def function(self, name: str, args: Sequence[Basic], context: PrettyContext) -> PrettyResult:
        """``name(args)``."""
        return self.applied(Picture.text(name), self.arguments(args, context), context)

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
