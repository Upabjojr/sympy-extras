r"""The LaTeX form of the native objects.

:func:`latex` prints an expression of the typed core (and the matrices,
the polynomials, the Python containers of them) as LaTeX, with the
settings of SymPy's ``latex`` which make sense here. It is a typed port of
SymPy's ``printing/latex.py`` for the classes this package has, on a
different design (:mod:`sympy_extras.printing.framework`):

- the rule of a node is found by its class in a table (``RULES``), not by
  the name of the class;
- a rule returns a :class:`~sympy_extras.printing.framework.Result`: the
  LaTeX together with the binding strength of what was *printed* (``- x``
  binds as a negation whatever the class of the node, ``\frac{1}{x}`` as a
  product, ``\sin^{2}{\left(x \right)}`` as a power);
- a parent gives each child a
  :class:`~sympy_extras.printing.framework.Context` (a term, a factor, a
  base with the exponent it may render itself, the side of a relation,
  the point of a limit, ...); the settings are never modified while
  printing;
- the brackets are added in one place
  (:func:`~sympy_extras.printing.framework.needs_brackets`), from the
  result of the child and its context.

The defects of SymPy's printer listed in sympy/sympy#30844 which concern
the classes of this package are not reproduced
(``tests/test_latex_issue_30844.py``); where SymPy's output is right the
output is the same, up to the differences listed in ``docs/printing.md``.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html#module-sympy.printing.latex
.. [2] SymPy issue 30844, *Printers: systematic review of precedence and
       other defects*, https://github.com/sympy/sympy/issues/30844
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/latex.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; dispatch by class in a table; rules returning typed
# results with the binding strength of the printed form, typed contexts passed
# down, brackets decided in one place; the defects of sympy/sympy#30844 fixed.
# ---------------------------------------------------------------------
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import AbstractSet, Mapping, Optional, Sequence, Union

import mpmath

from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.add import Add, add
from sympy_extras.core.assume import AppliedPredicate, Predicate
from sympy_extras.core.basic import Basic, ForeignPrint, PrintSetting
from sympy_extras.core.sympify import sympify
from sympy_extras.core.containers import Dict, Tuple
from sympy_extras.core.expr import Expr, UnevaluatedExpr
from sympy_extras.core.function import Application, AppliedUndef, Derivative, Lambda, Subs, WildFunction
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import ITE, And, BooleanAtom, Equivalent, Implies, Not, Or, Xor
from sympy_extras.core.mod import Mod
from sympy_extras.core.mul import Mul, neg
from sympy_extras.core.numbers import (AlgebraicNumber, CatalanType, ComplexInfinity, EulerGammaType, Exp1, Float,
    GoldenRatioType, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity, Number, Pi, Rational,
    TribonacciConstantType, fraction_to_mpf, number_from_fraction)
from sympy_extras.core.order import Order
from sympy_extras.core.power import Pow
from sympy_extras.core.printing import check_print_order, printed_factors, printed_terms
from sympy_extras.core.relational import Relational
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polytools import Poly

from ._latex_names import escape_name, escape_text, name_latex, operator_name, split_super_sub, text, translate
from .framework import Binding, Context, Dispatcher, Position, Result, TreePrinter, binding_named, needs_brackets

__all__ = ['latex', 'LatexSettings', 'LatexPrinter', 'LatexInput', 'RULES']

#: the context and the result of the LaTeX printer
Ctx = Context[str]
Res = Result[str]

#: what :func:`latex` prints
LatexInput = Union[Basic, ExprMatrix, Poly, bool, int, float, Fraction, str, None, Sequence['LatexInput'],
                   AbstractSet['LatexInput'], Mapping[Basic, 'LatexInput']]

_MODES = ('plain', 'inline', 'equation', 'equation*')
_MUL_SYMBOLS = {None: ' ', 'ldot': r' \,.\, ', 'dot': r' \cdot ', 'times': r' \times '}
_IMAGINARY_UNITS = {'i': 'i', 'ri': r'\mathrm{i}', 'ti': r'\text{i}', 'j': 'j', 'rj': r'\mathrm{j}',
                    'tj': r'\text{j}'}
_DIFF_OPERATORS = {'d': 'd', 'rd': r'\mathrm{d}', 'td': r'\text{d}'}
_SPACES = ('', ' ', '\\', '\\,', '\\:', '\\;', '\\quad')

_ENDS_WITH_NUMBER = re.compile(r'[0-9][} ]*$')
_STARTS_WITH_NUMBER = re.compile(r'(\d|\\frac{\d+}{\d+})')
_STARTS_WITH_DIGIT = re.compile(r'\d')


@dataclass(frozen=True)
class LatexSettings:
    """The settings of :func:`latex` (SymPy's, see :func:`latex`)."""

    mode: str = 'plain'
    fold_frac_powers: bool = False
    fold_func_brackets: bool = False
    fold_short_frac: Optional[bool] = None
    inv_trig_style: str = 'abbreviated'
    itex: bool = False
    ln_notation: bool = False
    long_frac_ratio: Optional[float] = None
    mat_delim: Optional[str] = '['
    mat_str: Optional[str] = None
    mul_symbol: Optional[str] = None
    order: Optional[str] = None
    symbol_names: Mapping[Symbol, str] = field(default_factory=dict)
    root_notation: bool = True
    imaginary_unit: str = 'i'
    gothic_re_im: bool = False
    decimal_separator: str = 'period'
    parenthesize_super: bool = True
    min: Optional[int] = None
    max: Optional[int] = None
    full_prec: bool = False
    diff_operator: str = 'd'


class LatexPrinter(TreePrinter[LatexInput, str]):
    r"""The printer of :func:`latex`: the settings (never modified) and
    the helpers the rules share.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.latex import LatexPrinter, LatexSettings
    >>> x, y = symbols('x y')
    >>> LatexPrinter(LatexSettings(mul_symbol='dot')).doprint(2*x*y)
    2 \cdot x \cdot y
    """

    def __init__(self, settings: Optional[LatexSettings] = None) -> None:
        s = LatexSettings() if settings is None else settings
        if s.mode not in _MODES:
            raise ValueError("'mode' must be one of 'inline', 'plain', 'equation' or 'equation*'")
        if s.decimal_separator not in ('period', 'comma'):
            raise ValueError("'decimal_separator' must be 'period' or 'comma'")
        if s.inv_trig_style not in ('abbreviated', 'full', 'power'):
            raise ValueError("'inv_trig_style' must be 'abbreviated', 'full' or 'power'")
        check_print_order(s.order)
        if s.mat_delim not in (None, '', '[', '('):
            raise ValueError("'mat_delim' must be '[', '(' or ''")
        self.settings: LatexSettings = s
        self.fold_short_frac: bool = s.mode == 'inline' if s.fold_short_frac is None else s.fold_short_frac
        #: between two factors, and between two numbers
        self.separator: str = self._mul_symbol(s.mul_symbol)
        if s.mul_symbol is None or s.mul_symbol.strip() in _SPACES:
            self.number_separator: str = _MUL_SYMBOLS['dot']
        else:
            self.number_separator = self.separator
        self.imaginary_unit: str = _IMAGINARY_UNITS.get(s.imaginary_unit, s.imaginary_unit)
        self.diff_operator: str = _DIFF_OPERATORS.get(s.diff_operator, s.diff_operator)
        #: between the elements of a container and the arguments of a function
        self.comma: str = ';' if s.decimal_separator == 'comma' else ','
        #: the settings handed to the system of a wrapped foreign object (Basic._foreign_latex)
        self.foreign_settings: dict[str, PrintSetting] = {
            'fold_frac_powers': s.fold_frac_powers, 'fold_func_brackets': s.fold_func_brackets,
            'fold_short_frac': self.fold_short_frac, 'inv_trig_style': s.inv_trig_style, 'itex': s.itex,
            'ln_notation': s.ln_notation, 'long_frac_ratio': s.long_frac_ratio, 'mat_delim': s.mat_delim,
            'mat_str': s.mat_str, 'mul_symbol': s.mul_symbol, 'order': s.order, 'root_notation': s.root_notation,
            'imaginary_unit': s.imaginary_unit, 'gothic_re_im': s.gothic_re_im, 'decimal_separator': s.decimal_separator,
            'parenthesize_super': s.parenthesize_super, 'min': s.min, 'max': s.max, 'diff_operator': s.diff_operator}

    @staticmethod
    def _mul_symbol(symbol: Optional[str]) -> str:
        if symbol in _MUL_SYMBOLS:
            return _MUL_SYMBOLS[symbol]
        if symbol is None or symbol.strip() == '':
            return ' '
        # a symbol of the user stands between spaces (sympy/sympy#30844: 2\starx)
        return ' %s ' % symbol.strip()

    # ------------------------------------------------------------------
    # the entry points

    def doprint(self, expr: LatexInput) -> str:
        """The LaTeX of ``expr`` in the mode of the settings."""
        s = self.settings
        tex = self.print(expr, Context(inline=s.mode == 'inline')).output
        if s.mode == 'plain':
            return tex
        if s.mode == 'inline':
            return '$%s$' % tex
        if s.itex:
            return '$$%s$$' % tex
        return r'\begin{%s}%s\end{%s}' % (s.mode, tex, s.mode)

    def rule(self, node: LatexInput, context: Ctx) -> Res:
        if isinstance(node, Basic):
            foreign = node._foreign_latex(self.foreign_settings)
            if foreign is not None:
                return self.foreign(foreign)
            found = RULES.lookup(type(node))
            if found is not None:
                return found(self, node, context)
            return _basic(self, node, context)
        return _python_value(self, node, context)

    def foreign(self, printed: ForeignPrint) -> Res:
        """The LaTeX which the system of a wrapped foreign object wrote
        for it (SymPy's, for the nodes of :mod:`sympy_extras.interop`),
        binding as that system says; an output which starts with a minus
        sign binds as a negation at most."""
        text = ' '.join(printed.rows)
        binding = binding_named(printed.binding)
        if text.lstrip().startswith('-') and binding > Binding.NEGATION:
            binding = Binding.NEGATION
        return Result(text, binding, applies=binding == Binding.APPLICATION)

    def bracket(self, result: Res, context: Ctx) -> str:
        scripts_only = result.scripted and context.scripts_follow and result.binding >= context.level
        if scripts_only and not self.settings.parenthesize_super:
            return '{%s}' % result.output
        return r'\left(%s\right)' % result.output

    def place(self, result: Res, context: Ctx) -> Res:
        """A result which was not printed by :meth:`print` (a product of
        some of the factors of a node) put in a position."""
        return self.enclose(result, context)

    # ------------------------------------------------------------------
    # helpers of the rules

    def inside(self, node: LatexInput, context: Ctx) -> str:
        """``node`` printed where its parent delimits it."""
        return self.print(node, context.delimited()).output

    def join(self, nodes: Sequence[LatexInput], context: Ctx, separator: Optional[str] = None) -> str:
        """The nodes printed as the elements of a list."""
        sep = self.comma + ' ' if separator is None else separator
        return sep.join(self.inside(n, context) for n in nodes)

    def product(self, factors: Sequence[Expr], context: Ctx, sole: Ctx) -> Res:
        """The factors side by side, a number separated from the number
        before it; a single factor is printed in the context ``sole``."""
        if not factors:
            return Result('1', Binding.ATOM)
        if len(factors) == 1:
            return self.print(factors[0], sole)
        tex = ''
        last = ''
        applies = False
        open_ended = False
        for i, f in enumerate(factors):
            r = self.print(f, context.factor(followed=i < len(factors) - 1))
            if tex:
                if _ENDS_WITH_NUMBER.search(last) and _STARTS_WITH_NUMBER.match(r.output):
                    tex += self.number_separator
                elif _STARTS_WITH_DIGIT.match(r.output):
                    tex += self.number_separator  # x \cdot 3, never x 3
                else:
                    tex += self.separator
            tex += r.output
            last = r.output
            applies = applies or r.applies
            open_ended = r.open_ended
        return Result(tex, Binding.PRODUCT, open_ended=open_ended, applies=applies)

    def attach(self, head: str, tail: str, context: Ctx, scripted_head: bool = False) -> Res:
        """A function printed as ``head`` and ``tail`` (its subscripts
        and bracketed arguments): the exponent its parent offers goes on
        the head (``J^{2}_{n}\\left(x\\right)``) unless the head has a
        superscript already."""
        exponent = context.exponent_offered
        if exponent is not None and not scripted_head:
            return Result('%s^{%s}%s' % (head, exponent, tail), Binding.POWER, absorbed=True, applies=True)
        return Result(head + tail, Binding.APPLICATION, applies=True)

    def apply(self, head: str, args: Sequence[Expr], context: Ctx, foldable: bool = True,
              scripted_head: bool = False) -> Res:
        """``head`` applied to the arguments as SymPy prints a function:
        ``\\sin{\\left(x \\right)}``, ``\\sin^{2}{\\left(x \\right)}`` when
        the parent offers an exponent, ``\\sin {x}`` with
        ``fold_func_brackets`` when the argument can stand bare."""
        exponent = context.exponent_offered
        absorbed = exponent is not None
        if exponent is not None:
            # the head is the base of the exponent: bracketed, where all brackets are, when it has a script
            base = self.place(Result(head, Binding.ATOM, scripted=scripted_head or '^' in head), context.base(None))
            head = '%s^{%s}' % (base.output, exponent)
        body = r'{\left(%s \right)}' % self.join(args, context, self.comma)
        if self.settings.fold_func_brackets and foldable and len(args) == 1:
            bare = context.bare_argument()
            raw = self.rule(args[0], bare)
            if not needs_brackets(raw, bare):
                body = ' {%s}' % raw.output
        return Result(head + body, Binding.POWER if absorbed else Binding.APPLICATION, absorbed=absorbed,
                      applies=True)

    def undefined_head(self, name: str) -> tuple[str, bool]:
        """The head of an undefined function and whether it has a
        superscript: a single letter as it is, a longer name in
        ``\\operatorname``, the scripts of the name after it. The name is
        never taken for a command of LaTeX (``frac``, ``root``)."""
        if '{' in name:
            escaped = escape_name(name)
            return (escaped, '^' in escaped)
        base, supers, subs = split_super_sub(name)
        translated = translate(base)
        if len(base) == 1 or translated != base:
            head = escape_name(translated)
        else:
            head = r'\operatorname{%s}' % escape_text(base)
        if supers:
            head += '^{%s}' % ' '.join(escape_name(translate(s)) for s in supers)
        if subs:
            head += '_{%s}' % ' '.join(escape_name(translate(s)) for s in subs)
        return (head, bool(supers))

    def ordered(self, e: Add) -> Sequence[Expr]:
        """The terms of a sum in the order of the settings."""
        return printed_terms(e.args, self.settings.order)

    def float_string(self, e: Float) -> str:
        """The digits of a float as SymPy's LaTeX printer shows them."""
        s = self.settings
        with mpmath.workprec(e.precision):
            value = fraction_to_mpf(e.as_fraction())
            return str(mpmath.libmp.to_str(value._mpf_, e.dps, strip_zeros=not s.full_prec, min_fixed=s.min,
                                           max_fixed=s.max))


#: the rules of the LaTeX printer by the class of the node
RULES: Dispatcher[LatexPrinter, Basic, str] = Dispatcher()


# ----------------------------------------------------------------------
# what has no rule


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Basic
def _basic(p: LatexPrinter, e: Basic, ctx: Ctx) -> Res:
    name = escape_text(type(e).__name__)
    if e.args:
        return Result(r'\operatorname{%s}\left(%s\right)' % (name, p.join(e.args, ctx)), Binding.APPLICATION,
                      applies=True)
    return Result(r'\text{%s}' % name, Binding.ATOM)


def _python_value(p: LatexPrinter, e: LatexInput, ctx: Ctx) -> Res:
    """The Python values: numbers, strings, containers, and the matrices
    and polynomials, which are not nodes."""
    if e is None or isinstance(e, bool):
        return Result(r'\text{%s}' % e, Binding.ATOM)
    if isinstance(e, int):
        return p.rule(Integer(e), ctx)
    if isinstance(e, float):
        # (through sympify: an infinite float is \infty, nan is \text{NaN}; Float(inf) raised OverflowError)
        return p.rule(sympify(e), ctx)
    if isinstance(e, Fraction):
        return p.rule(number_from_fraction(e), ctx)
    if isinstance(e, str):
        return Result(text(e), Binding.ATOM)
    if isinstance(e, ExprMatrix):
        return _matrix(p, e, ctx)
    if isinstance(e, Poly):
        return _poly(p, e, ctx)
    if isinstance(e, Basic):
        return p.rule(e, ctx)
    if isinstance(e, Mapping):
        keys = sorted(e, key=default_sort_key)
        items = ['%s : %s' % (p.inside(k, ctx), p.inside(e[k], ctx)) for k in keys]
        return Result(r'\left\{ %s\right\}' % (p.comma + r' \  ').join(items), Binding.ATOM)
    if isinstance(e, AbstractSet):
        elements = sorted(e, key=_element_key)
        return Result(r'\left\{%s\right\}' % p.join(elements, ctx), Binding.ATOM)
    if isinstance(e, tuple):
        return _tuple(p, e, ctx)
    return Result(r'\left[ %s\right]' % p.join(e, ctx, p.comma + r' \  '), Binding.ATOM)


def _element_key(e: LatexInput) -> tuple[int, str]:
    """The order of the elements of a Python set: the nodes in the
    printing order, the other values by their text."""
    if isinstance(e, Basic):
        return (0, str(default_sort_key(e)))
    return (1, str(e))


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_tuple
def _tuple(p: LatexPrinter, elements: Sequence[LatexInput], ctx: Ctx) -> Res:
    if len(elements) == 1:
        return Result(r'\left( %s%s\right)' % (p.inside(elements[0], ctx), p.comma), Binding.ATOM)
    return Result(r'\left( %s\right)' % p.join(elements, ctx, p.comma + r' \  '), Binding.ATOM)


# ----------------------------------------------------------------------
# numbers and symbols


@RULES.of(Rational)
def _rational(p: LatexPrinter, e: Rational, ctx: Ctx) -> Res:
    negative = e.p < 0
    if e.q == 1:
        if negative:
            return Result(str(e.p), Binding.NEGATION, unsigned=str(-e.p))
        return Result(str(e.p), Binding.ATOM)
    if ctx.position is Position.EXPONENT and p.settings.fold_frac_powers:
        return Result('%d/%d' % (e.p, e.q), Binding.QUOTIENT)
    if p.fold_short_frac:
        magnitude = '%d / %d' % (abs(e.p), e.q)
        binding = Binding.QUOTIENT
    else:
        magnitude = r'\frac{%d}{%d}' % (abs(e.p), e.q)
        binding = Binding.PRODUCT
    if negative:
        return Result('- ' + magnitude, Binding.NEGATION, unsigned=magnitude)
    return Result(magnitude, binding)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Float
@RULES.of(Float)
def _float(p: LatexPrinter, e: Float, ctx: Ctx) -> Res:
    digits = p.float_string(e)
    negative = digits.startswith('-')
    if negative:
        digits = digits[1:]
    binding = Binding.ATOM
    if 'e' in digits:
        mantissa, exponent = digits.split('e')
        if exponent.startswith('+'):
            exponent = exponent[1:]
        if p.settings.decimal_separator == 'comma':
            mantissa = mantissa.replace('.', '{,}')
        # a product as printed: bracketed as the base of a power (SymPy: {2.5 \cdot 10^{20}}^{2})
        digits = '%s%s10^{%s}' % (mantissa, p.number_separator, exponent)
        binding = Binding.PRODUCT
    elif p.settings.decimal_separator == 'comma':
        digits = digits.replace('.', '{,}')
    if negative:
        return Result('-' + digits, Binding.NEGATION, unsigned=digits)
    return Result(digits, binding)


@RULES.of(Infinity)
def _infinity(p: LatexPrinter, e: Infinity, ctx: Ctx) -> Res:
    return Result(r'\infty', Binding.ATOM)


@RULES.of(NegativeInfinity)
def _negative_infinity(p: LatexPrinter, e: NegativeInfinity, ctx: Ctx) -> Res:
    return Result(r'-\infty', Binding.NEGATION, unsigned=r'\infty')


@RULES.of(ComplexInfinity)
def _complex_infinity(p: LatexPrinter, e: ComplexInfinity, ctx: Ctx) -> Res:
    return Result(r'\tilde{\infty}', Binding.ATOM)


@RULES.of(NaN)
def _nan(p: LatexPrinter, e: NaN, ctx: Ctx) -> Res:
    return Result(r'\text{NaN}', Binding.ATOM)


@RULES.of(Pi)
def _pi(p: LatexPrinter, e: Pi, ctx: Ctx) -> Res:
    return Result(r'\pi', Binding.ATOM)


@RULES.of(Exp1)
def _exp1(p: LatexPrinter, e: Exp1, ctx: Ctx) -> Res:
    return Result('e', Binding.ATOM)


@RULES.of(GoldenRatioType)
def _golden_ratio(p: LatexPrinter, e: GoldenRatioType, ctx: Ctx) -> Res:
    return Result(r'\phi', Binding.ATOM)


@RULES.of(EulerGammaType)
def _euler_gamma(p: LatexPrinter, e: EulerGammaType, ctx: Ctx) -> Res:
    return Result(r'\gamma', Binding.ATOM)


@RULES.of(CatalanType)
def _catalan_constant(p: LatexPrinter, e: CatalanType, ctx: Ctx) -> Res:
    return Result('G', Binding.ATOM)


@RULES.of(TribonacciConstantType)
def _tribonacci_constant(p: LatexPrinter, e: TribonacciConstantType, ctx: Ctx) -> Res:
    return Result(r'\text{TribonacciConstant}', Binding.ATOM)


@RULES.of(ImaginaryUnit)
def _imaginary_unit(p: LatexPrinter, e: ImaginaryUnit, ctx: Ctx) -> Res:
    return Result(p.imaginary_unit, Binding.ATOM)


@RULES.of(AlgebraicNumber)
def _algebraic_number(p: LatexPrinter, e: AlgebraicNumber, ctx: Ctx) -> Res:
    # printed as its expression, bracketed as what that prints (SymPy: an atom)
    return p.rule(e.as_expr(), ctx.same())


@RULES.of(UnevaluatedExpr)
def _unevaluated(p: LatexPrinter, e: UnevaluatedExpr, ctx: Ctx) -> Res:
    return p.rule(e.args[0], ctx.same())


@RULES.of(Symbol)
def _symbol(p: LatexPrinter, e: Symbol, ctx: Ctx) -> Res:
    given = p.settings.symbol_names.get(e)
    if given is not None:
        return Result(given, Binding.ATOM, scripted='^' in given)
    tex, scripted = name_latex(e.name)
    return Result(tex, Binding.ATOM, scripted=scripted)


@RULES.of(BooleanAtom)
def _boolean_atom(p: LatexPrinter, e: BooleanAtom, ctx: Ctx) -> Res:
    return Result(r'\text{%s}' % e, Binding.ATOM)


# ----------------------------------------------------------------------
# sums, products, powers


@RULES.of(Add)
def _add(p: LatexPrinter, e: Add, ctx: Ctx) -> Res:
    terms = p.ordered(e)
    if len(terms) == 1:
        return p.rule(terms[0], ctx.same())
    tex = ''
    open_ended = False
    applies = False
    for i, term in enumerate(terms):
        r = p.print(term, ctx.term())
        if i == 0:
            tex = r.output
        elif r.unsigned is not None:
            tex += ' - ' + r.unsigned
        else:
            tex += ' + ' + r.output
        open_ended = r.open_ended
        applies = applies or r.applies
    return Result(tex, Binding.SUM, open_ended=open_ended, applies=applies)


def _is_one(e: Expr) -> bool:
    return isinstance(e, Rational) and e.p == 1 and e.q == 1


def _negative_number(e: Expr) -> bool:
    """A negative constant: what SymPy's ``fraction(exact=True)`` moves
    to the denominator when it is an exponent."""
    if isinstance(e, Rational):
        return e.p < 0
    if isinstance(e, Float):
        return e.as_fraction() < 0
    return e.is_number and e.is_negative is True and not e.free_symbols


def _split_fraction(factors: Sequence[Expr]) -> tuple[bool, list[Expr], list[Expr]]:
    """The sign, the factors of the numerator and those of the
    denominator of a product in the printing order: the denominator of a
    leading rational coefficient and the commutative powers with a
    negative constant exponent."""
    negative = False
    numerator: list[Expr] = []
    denominator: list[Expr] = []
    for i, f in enumerate(factors):
        if i == 0 and isinstance(f, Rational):
            negative = f.p < 0
            if abs(f.p) != 1 or (f.q == 1 and len(factors) == 1):
                numerator.append(Integer(abs(f.p)))
            if f.q != 1:
                denominator.append(Integer(f.q))
        elif i == 0 and isinstance(f, Float) and f.as_fraction() < 0:
            negative = True
            numerator.append(Float(number_from_fraction(-f.as_fraction()), precision=f.precision))
        elif isinstance(f, Pow) and f.base.is_commutative is not False and _negative_number(f.exp):
            exponent = neg(f.exp)
            denominator.append(f.base if _is_one(exponent) else Pow(f.base, exponent))
        elif isinstance(f, Exp) and _negative_number(f.args[0]):
            denominator.append(Exp(neg(f.args[0])))
        else:
            numerator.append(f)
    return (negative, numerator, denominator)


def _signed(sign: bool, r: Res) -> Res:
    if not sign:
        return r
    return Result('- ' + r.output, Binding.NEGATION, unsigned=r.output, open_ended=r.open_ended, applies=r.applies)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Mul
@RULES.of(Mul)
def _mul(p: LatexPrinter, e: Mul, ctx: Ctx) -> Res:
    args = e.args
    if not args:
        return Result('1', Binding.ATOM)
    if _is_one(args[0]) or any(isinstance(a, Number) for a in args[1:]):
        # an unevaluated product: the factors as they are
        return p.product(args, ctx, ctx.factor(False))
    factors = printed_factors(args, p.settings.order)
    negative, numerator, denominator = _split_fraction(factors)
    if not denominator:
        return _signed(negative, p.place(p.product(numerator, ctx, ctx.factor(False)), ctx.factor(False)))
    return _signed(negative, _fraction(p, numerator, denominator, ctx))


def _fraction(p: LatexPrinter, numerator: Sequence[Expr], denominator: Sequence[Expr], ctx: Ctx) -> Res:
    """``numerator/denominator``: built up, on one line when the
    denominator is short and ``fold_short_frac`` is set, with a part of
    the numerator after the fraction when it is long (``long_frac_ratio``)."""
    built_den = p.product(denominator, ctx, ctx.denominator(False))
    sdenom = built_den.output
    ldenom = len(sdenom.split())
    built_num = p.product(numerator, ctx, ctx.numerator(False))
    snumer = built_num.output
    ratio = p.settings.long_frac_ratio
    if p.fold_short_frac and ldenom <= 2 and '^' not in sdenom:
        num = p.place(p.product(numerator, ctx, ctx.numerator(True)), ctx.numerator(True))
        den = p.place(p.product(denominator, ctx, ctx.denominator(True)), ctx.denominator(True))
        return Result('%s / %s' % (num.output, den.output), Binding.QUOTIENT, applies=num.applies or den.applies)
    if ratio is not None and len(snumer.split()) > ratio * ldenom:
        if len(numerator) <= 1:
            after = p.product(numerator, ctx, ctx.factor(False))
            return Result(r'\frac{1}{%s}%s%s' % (sdenom, p.separator, after.output), Binding.PRODUCT,
                          open_ended=after.open_ended, applies=True)
        kept: list[Expr] = []
        moved: list[Expr] = []
        for f in numerator:
            followed = ctx.factor(True)
            long = len(p.product(kept + [f], ctx, ctx.numerator(False)).output.split()) > ratio * ldenom
            noncommutative = bool(moved) and moved[-1].is_commutative is False and f.is_commutative is False
            if needs_brackets(p.rule(f, followed), followed) or long or noncommutative:
                moved.append(f)
            else:
                kept.append(f)
        if moved:
            after = p.place(p.product(moved, ctx, ctx.factor(False)), ctx.factor(False))
            head = p.product(kept, ctx, ctx.numerator(False)).output
            return Result(r'\frac{%s}{%s}%s%s' % (head, sdenom, p.separator, after.output), Binding.PRODUCT,
                          open_ended=after.open_ended, applies=True)
    return Result(r'\frac{%s}{%s}' % (snumer, sdenom), Binding.PRODUCT)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Pow
@RULES.of(Pow)
def _pow(p: LatexPrinter, e: Pow, ctx: Ctx) -> Res:
    base, exp = e.base, e.exp
    s = p.settings
    if isinstance(exp, Rational):
        if abs(exp.p) == 1 and exp.q != 1 and s.root_notation:
            radicand = p.inside(base, ctx)
            if exp.q == 2:
                tex = r'\sqrt{%s}' % radicand
            elif s.itex:
                tex = r'\root{%d}{%s}' % (exp.q, radicand)
            else:
                tex = r'\sqrt[%d]{%s}' % (exp.q, radicand)
            if exp.p < 0:
                return _reciprocal(p, Result(tex, Binding.POWER), ctx)
            return Result(tex, Binding.POWER)
        if exp.p < 0 and base.is_commutative is not False and not _is_one(base) \
                and not (s.fold_frac_powers and exp.q != 1):
            positive = neg(exp)
            inverse = p.rule(base if _is_one(positive) else Pow(base, positive), ctx.denominator(False))
            return _reciprocal(p, inverse, ctx)
    exponent = p.print(exp, ctx.exponent()).output
    printed = p.print(base, ctx.base(exponent))
    if printed.absorbed:
        # the base rendered the exponent: what it printed is this power, not the parent of this power
        return replace(printed, absorbed=False)
    return Result('%s^{%s}' % (printed.output, exponent), Binding.POWER, applies=printed.applies)


def _reciprocal(p: LatexPrinter, denominator: Res, ctx: Ctx) -> Res:
    """``1/denominator`` for a denominator printed without brackets."""
    if p.fold_short_frac and len(denominator.output.split()) <= 2 and '^' not in denominator.output:
        placed = p.place(denominator, ctx.denominator(True))
        return Result('1 / %s' % placed.output, Binding.QUOTIENT, applies=placed.applies)
    return Result(r'\frac{1}{%s}' % denominator.output, Binding.PRODUCT)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Mod
@RULES.of(Mod)
def _mod(p: LatexPrinter, e: Mod, ctx: Ctx) -> Res:
    # the operands are products or tighter: a negative operand, a sum and a
    # nested mod are bracketed (SymPy: - x \bmod y, x \bmod y \bmod z)
    left = p.print(e.args[0], ctx.left(Binding.NEGATION))
    right = p.print(e.args[1], ctx.right(Binding.NEGATION))
    return Result(r'%s \bmod %s' % (left.output, right.output), Binding.MODULO)


# ----------------------------------------------------------------------
# relations and logic

_RELATIONS = {'==': '=', '>': '>', '<': '<', '>=': r'\geq', '<=': r'\leq', '!=': r'\neq'}


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Relational
@RULES.of(Relational)
def _relational(p: LatexPrinter, e: Relational, ctx: Ctx) -> Res:
    sign = _RELATIONS[e.rel_op]
    if p.settings.itex and sign in ('<', '>'):
        sign = r'\gt' if sign == '>' else r'\lt'
    # a relation or a connective on a side is bracketed (SymPy: x = y = z)
    lhs = p.print(e.lhs, ctx.side())
    rhs = p.print(e.rhs, ctx.side())
    return Result('%s %s %s' % (lhs.output, sign, rhs.output), Binding.RELATION, open_ended=rhs.open_ended)


def _connective(p: LatexPrinter, operands: Sequence[Basic], sign: str, binding: Binding, ctx: Ctx,
                absorbed: bool = False) -> Res:
    parts = [p.print(a, ctx.connective()).output for a in operands]
    return Result((' %s ' % sign).join(parts), binding, absorbed=absorbed)


@RULES.of(And)
def _and(p: LatexPrinter, e: And, ctx: Ctx) -> Res:
    return _connective(p, sorted(e.args, key=default_sort_key), r'\wedge', Binding.AND, ctx)


@RULES.of(Or)
def _or(p: LatexPrinter, e: Or, ctx: Ctx) -> Res:
    return _connective(p, sorted(e.args, key=default_sort_key), r'\vee', Binding.OR, ctx)


@RULES.of(Xor)
def _xor(p: LatexPrinter, e: Xor, ctx: Ctx) -> Res:
    return _connective(p, sorted(e.args, key=default_sort_key), r'\veebar', Binding.IMPLICATION, ctx)


@RULES.of(Implies)
def _implies(p: LatexPrinter, e: Implies, ctx: Ctx) -> Res:
    if ctx.negation_offered:
        return _connective(p, e.args, r'\not\Rightarrow', Binding.IMPLICATION, ctx, absorbed=True)
    return _connective(p, e.args, r'\Rightarrow', Binding.IMPLICATION, ctx)


@RULES.of(Equivalent)
def _equivalent(p: LatexPrinter, e: Equivalent, ctx: Ctx) -> Res:
    operands = sorted(e.args, key=default_sort_key)
    if ctx.negation_offered:
        return _connective(p, operands, r'\not\Leftrightarrow', Binding.IMPLICATION, ctx, absorbed=True)
    return _connective(p, operands, r'\Leftrightarrow', Binding.IMPLICATION, ctx)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Not
@RULES.of(Not)
def _not(p: LatexPrinter, e: Not, ctx: Ctx) -> Res:
    # the operand is bracketed by what it printed (SymPy: \neg x > y)
    operand = p.print(e.args[0], ctx.negated())
    if operand.absorbed:
        return Result(operand.output, operand.binding)
    return Result(r'\neg %s' % operand.output, Binding.NOT)


@RULES.of(ITE)
def _ite(p: LatexPrinter, e: ITE, ctx: Ctx) -> Res:
    return Result(r'\operatorname{ITE}{\left(%s \right)}' % p.join(e.args, ctx), Binding.APPLICATION, applies=True)


@RULES.of(Predicate)
def _predicate(p: LatexPrinter, e: Predicate, ctx: Ctx) -> Res:
    return Result(r'\operatorname{Q}_{\text{%s}}' % escape_text(e.name), Binding.ATOM, applies=False)


@RULES.of(AppliedPredicate)
def _applied_predicate(p: LatexPrinter, e: AppliedPredicate, ctx: Ctx) -> Res:
    head = p.inside(e.function, ctx)
    return Result(r'%s\left(%s\right)' % (head, p.join(e.arguments, ctx)), Binding.APPLICATION, applies=True)


# ----------------------------------------------------------------------
# functions


@RULES.of(Application)
def _function(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    """A function without a form of its own: ``\\sin{\\left(x \\right)}``,
    ``\\operatorname{erf}{\\left(x \\right)}``."""
    return p.apply(operator_name(e.name), e.args, ctx)


@RULES.of(AppliedUndef)
def _applied_undef(p: LatexPrinter, e: AppliedUndef, ctx: Ctx) -> Res:
    head, scripted = p.undefined_head(e.name)
    # never without brackets: f x reads as a product
    return p.apply(head, e.args, ctx, foldable=False, scripted_head=scripted)


@RULES.of(WildFunction)
def _wild_function(p: LatexPrinter, e: WildFunction, ctx: Ctx) -> Res:
    return Result(r'\operatorname{%s\_}' % escape_text(e.name), Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Lambda
@RULES.of(Lambda)
def _lambda(p: LatexPrinter, e: Lambda, ctx: Ctx) -> Res:
    variables = e.variables
    symbols = p.inside(variables[0], ctx) if len(variables) == 1 else _tuple(p, variables, ctx).output
    return Result(r'\left( %s \mapsto %s \right)' % (symbols, p.inside(e.expr, ctx)), Binding.ATOM)


# ----------------------------------------------------------------------
# calculus


# PORTED FROM SYMPY: sympy/printing/conventions.py:requires_partial
def _requires_partial(e: Expr) -> bool:
    if isinstance(e, Derivative):
        return _requires_partial(e.expr)
    return sum(1 for s in e.free_symbols if s.is_integer is not True) > 1


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Derivative
@RULES.of(Derivative)
def _derivative(p: LatexPrinter, e: Derivative, ctx: Ctx) -> Res:
    d = r'\partial' if _requires_partial(e.expr) else p.diff_operator
    tex = ''
    # the orders are integers or symbolic (\frac{d^{n}}{d x^{n}}); their sum is the order of the operator
    total: Expr = Integer(0)
    for x, count in reversed(e.variable_orders):
        total = add(total, count)
        if count == 1:
            tex += '%s %s' % (d, p.inside(x, ctx))
        else:
            tex += '%s %s^{%s}' % (d, p.print(x, ctx.base(None)).output, p.inside(count, ctx))
    if total == 1:
        tex = r'\frac{%s}{%s}' % (d, tex)
    else:
        tex = r'\frac{%s^{%s}}{%s}' % (d, p.inside(total, ctx), tex)
    # the body is a product or tighter, and the derivative is bracketed
    # when a factor follows it (SymPy: \frac{d}{d x} f \frac{d}{d x} g)
    body = p.print(e.expr, ctx.body())
    return Result('%s %s' % (tex, body.output), Binding.PRODUCT, open_ended=True, applies=body.applies)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Subs
@RULES.of(Subs)
def _subs(p: LatexPrinter, e: Subs, ctx: Ctx) -> Res:
    below = ctx.subscript()
    pairs = ['%s=%s' % (p.print(v, below).output, p.print(n, below).output) for v, n in zip(e.variables, e.point)]
    tex = r'\left. %s \right|_{\substack{ %s }}' % (p.inside(e.expr, ctx), r'\\ '.join(pairs))
    # ends with a subscript: bracketed as a base (SymPy: \left. f \right|_{x=1}^{2})
    return Result(tex, Binding.ATOM, scripted=True)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Limit
@RULES.of(Limit)
def _limit(p: LatexPrinter, e: Limit, ctx: Ctx) -> Res:
    z0 = e.point
    if e.direction == '+-' or isinstance(z0, (Infinity, NegativeInfinity)):
        point = p.print(z0, ctx.subscript()).output
    else:
        # the mark of the direction is a script of the point (SymPy: a + b^+, a^{2}^+)
        point = '%s^%s' % (p.print(z0, ctx.point()).output, e.direction)
    body = p.print(e.expr, ctx.body())
    tex = r'\lim_{%s \to %s} %s' % (p.print(e.variable, ctx.subscript()).output, point, body.output)
    return Result(tex, Binding.PRODUCT, open_ended=True, applies=body.applies)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Order
@RULES.of(Order)
def _order(p: LatexPrinter, e: Order, ctx: Ctx) -> Res:
    tex = p.inside(e.expr, ctx)
    variables, point = e.variables, e.point
    if (point and any(not (isinstance(z, Rational) and z.p == 0) for z in point)) or len(variables) > 1:
        tex += '; '
        if len(variables) > 1:
            tex += _tuple(p, variables, ctx).output
        elif variables:
            tex += p.inside(variables[0], ctx)
        tex += r'\rightarrow '
        if len(point) > 1:
            tex += _tuple(p, point, ctx).output
        else:
            tex += p.inside(point[0], ctx)
    return Result(r'O\left(%s\right)' % tex, Binding.APPLICATION, applies=True)


@RULES.of(AccumBounds)
def _accumulation_bounds(p: LatexPrinter, e: AccumBounds, ctx: Ctx) -> Res:
    return Result(r'\left\langle %s%s %s\right\rangle' % (p.inside(e.min, ctx), p.comma, p.inside(e.max, ctx)),
                  Binding.ATOM)


# ----------------------------------------------------------------------
# containers, matrices, polynomials


@RULES.of(Tuple)
def _tuple_node(p: LatexPrinter, e: Tuple, ctx: Ctx) -> Res:
    return _tuple(p, e.args, ctx)


@RULES.of(Dict)
def _dict_node(p: LatexPrinter, e: Dict, ctx: Ctx) -> Res:
    keys = sorted(e.keys(), key=default_sort_key)
    values = dict(e.items())
    items = ['%s : %s' % (p.inside(k, ctx), p.inside(values[k], ctx)) for k in keys]
    return Result(r'\left\{ %s\right\}' % (p.comma + r' \  ').join(items), Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_MatrixBase
def _matrix(p: LatexPrinter, m: ExprMatrix, ctx: Ctx) -> Res:
    s = p.settings
    lines = [' & '.join(p.inside(m[i, j], ctx) for j in range(m.cols)) for i in range(m.rows)]
    environment = s.mat_str
    if environment is None:
        if ctx.inline:
            environment = 'smallmatrix'
        else:
            environment = 'matrix' if m.cols <= 10 else 'array'
    columns = '{%s}' % ('c' * m.cols) if environment == 'array' else ''
    tex = r'\begin{%s}%s%s\end{%s}' % (environment, columns, r'\\'.join(lines), environment)
    if s.mat_delim:
        right = {'(': ')', '[': ']'}[s.mat_delim]
        tex = r'\left%s%s\right%s' % (s.mat_delim, tex, right)
        return Result(tex, Binding.ATOM)
    return Result(tex, Binding.BLOCK)


_DOMAIN = re.compile(r'^(ZZ|QQ|ZZ_I|QQ_I|RR|CC|EX)(?:([\[(])(.*)[\])])?$')
_DOMAIN_NAMES = {'ZZ': r'\mathbb{Z}', 'QQ': r'\mathbb{Q}', 'RR': r'\mathbb{R}', 'CC': r'\mathbb{C}',
                 'ZZ_I': r'\mathbb{Z}\left[i\right]', 'QQ_I': r'\mathbb{Q}\left(i\right)'}


def _domain(name: str) -> str:
    """The LaTeX of the name of a domain (``ZZ``, ``QQ[a,b]``, ``ZZ(a)``)."""
    m = _DOMAIN.match(name.replace(' ', ''))
    if m is None or m.group(1) not in _DOMAIN_NAMES:
        return text(name)
    ground = _DOMAIN_NAMES[m.group(1)]
    if m.group(2) is None:
        return ground
    symbols = ', '.join(name_latex(s)[0] for s in m.group(3).split(','))
    if m.group(2) == '[':
        return r'%s\left[%s\right]' % (ground, symbols)
    return r'%s\left(%s\right)' % (ground, symbols)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Poly
def _poly(p: LatexPrinter, poly: Poly, ctx: Ctx) -> Res:
    tex = ''
    for monomial, coefficient in poly.terms():
        factors: list[Expr] = []
        for gen, exponent in zip(poly.gens, monomial):
            if exponent == 1:
                factors.append(gen)
            elif exponent > 1:
                factors.append(Pow(gen, Integer(exponent)))
        if not factors:
            r = p.print(coefficient, ctx.term())
        elif _is_one(coefficient):
            r = p.product(factors, ctx, ctx.factor(False))
        elif isinstance(coefficient, Rational) and coefficient.p == -1 and coefficient.q == 1:
            r = _signed(True, p.place(p.product(factors, ctx, ctx.factor(False)), ctx.factor(False)))
        else:
            c = p.rule(coefficient, ctx.term())
            if c.unsigned is not None:
                r = _signed(True, _monomial(p, c.unsigned, factors, ctx))
            else:
                r = _monomial(p, p.place(c, ctx.factor(True)).output, factors, ctx)
        if not tex:
            tex = r.output
        elif r.unsigned is not None:
            tex += ' - ' + r.unsigned
        else:
            tex += ' + ' + r.output
    arguments = [tex or '0'] + [p.inside(g, ctx) for g in poly.gens] + ['domain=%s' % _domain(poly.get_domain())]
    return Result(r'\operatorname{Poly}{\left( %s \right)}' % ', '.join(arguments), Binding.APPLICATION,
                  applies=True)


def _monomial(p: LatexPrinter, coefficient: str, factors: Sequence[Expr], ctx: Ctx) -> Res:
    monomial = p.place(p.product(factors, ctx, ctx.factor(False)), ctx.factor(False))
    return Result('%s %s' % (coefficient, monomial.output), Binding.PRODUCT)


# ----------------------------------------------------------------------
# the entry point


def latex(expr: LatexInput, *, mode: str = 'plain', fold_frac_powers: bool = False,
          fold_func_brackets: bool = False, fold_short_frac: Optional[bool] = None,
          inv_trig_style: str = 'abbreviated', itex: bool = False, ln_notation: bool = False,
          long_frac_ratio: Optional[float] = None, mat_delim: Optional[str] = '[', mat_str: Optional[str] = None,
          mul_symbol: Optional[str] = None, order: Optional[str] = None,
          symbol_names: Optional[Mapping[Symbol, str]] = None, root_notation: bool = True,
          imaginary_unit: str = 'i', gothic_re_im: bool = False, decimal_separator: str = 'period',
          parenthesize_super: bool = True, min: Optional[int] = None, max: Optional[int] = None,
          full_prec: bool = False, diff_operator: str = 'd') -> str:
    r"""The LaTeX form of ``expr``.

    Parameters
    ==========

    expr : a node, a matrix, a polynomial, a Python number or container
    mode : 'plain' (default), 'inline' (``$...$``, with short fractions on
        one line), 'equation' or 'equation*' (in the environment)
    fold_frac_powers : ``x^{2/3}`` instead of ``x^{\frac{2}{3}}``
    fold_func_brackets : ``\sin {x}`` where the argument can stand bare
    fold_short_frac : ``x / 2`` when the denominator is short (the default
        in the inline mode)
    inv_trig_style : 'abbreviated' (``asin``), 'full' (``\arcsin``) or
        'power' (``\sin^{-1}``)
    itex : the dialect of itex (``$$...$$``, ``\root``, ``\gt``)
    ln_notation : ``\ln`` instead of ``\log``
    long_frac_ratio : the ratio of the widths of the numerator and the
        denominator above which a part of the numerator is put after the
        fraction
    mat_delim : the delimiter of matrices, '[' (default), '(' or ''
    mat_str : the environment of matrices ('matrix', 'smallmatrix',
        'array', ...)
    mul_symbol : None (a space), 'ldot', 'dot', 'times' or any LaTeX
    order : the order of the terms of the sums, as in SymPy: None, 'lex',
        'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex', one of these
        after 'rev-' (reversed), 'old' or 'none' (as stored); see
        :func:`sympy_extras.core.printing.printed_terms`
    symbol_names : the LaTeX of some symbols
    root_notation : ``\sqrt[3]{x}`` instead of ``x^{\frac{1}{3}}``
    imaginary_unit : 'i' (default), 'ri', 'ti', 'j', 'rj', 'tj' or any LaTeX
    gothic_re_im : ``\Re`` and ``\Im``
    decimal_separator : 'period' or 'comma' (``2{,}5``, the elements and
        the arguments then separated by semicolons everywhere)
    parenthesize_super : ``\left(x^{i}\right)^{2}`` (default) or ``{x^{i}}^{2}``
    min, max : the exponents below and above which a float is printed in
        the scientific notation
    full_prec : the floats with all the digits of their precision
    diff_operator : 'd' (default), 'rd', 'td' or any LaTeX

    Examples
    ========

    >>> from sympy_extras.core import symbols, Rational, Mul, Pow, Eq, Not
    >>> from sympy_extras.functions.elementary.trigonometric import sin
    >>> from sympy_extras.printing import latex
    >>> x, y = symbols('x y')
    >>> print(latex((2*x + 1)**2/y + sin(x)**2))
    \sin^{2}{\left(x \right)} + \frac{\left(2 x + 1\right)^{2}}{y}
    >>> print(latex(x**Rational(2, 3), fold_frac_powers=True), latex(3*x/2, mode='inline'))
    x^{2/3} $3 x / 2$

    The brackets follow what is printed, not the classes:

    >>> print(latex(Mul(2, Mul(-1, y)))), print(latex(Not(Eq(x, y)))), print(latex(Pow(Rational(-1, 2), x)))
    2 \left(- y\right)
    \neg \left(x = y\right)
    \left(- \frac{1}{2}\right)^{x}
    (None, None, None)
    """
    settings = LatexSettings(mode=mode, fold_frac_powers=fold_frac_powers, fold_func_brackets=fold_func_brackets,
                             fold_short_frac=fold_short_frac, inv_trig_style=inv_trig_style, itex=itex,
                             ln_notation=ln_notation, long_frac_ratio=long_frac_ratio, mat_delim=mat_delim,
                             mat_str=mat_str, mul_symbol=mul_symbol, order=order,
                             symbol_names={} if symbol_names is None else symbol_names,
                             root_notation=root_notation, imaginary_unit=imaginary_unit, gothic_re_im=gothic_re_im,
                             decimal_separator=decimal_separator, parenthesize_super=parenthesize_super, min=min,
                             max=max, full_prec=full_prec, diff_operator=diff_operator)
    return LatexPrinter(settings).doprint(expr)


# the rules of the functions and of the other objects register themselves
from . import _latex_functions, _latex_objects  # noqa: E402

_RULE_MODULES = (_latex_functions, _latex_objects)
