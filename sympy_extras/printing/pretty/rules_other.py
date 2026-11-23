"""The pretty-printing rules of the remaining nodes: quantified
formulas, roots of polynomials, sequences and series.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_SeqFormula, _print_FourierSeries,
# _print_FormalPowerSeries, _print_ComplexRootOf, _print_RootSum) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; typed results and contexts; a sequence with
# symbolic bounds is printed in its constructor form instead of raising;
# the quantified formulas are original code.
# ---------------------------------------------------------------------
from __future__ import annotations

from ...assumptions.quantifiers import Exists, ForAll, Quantifier
from ...core.expr import Expr
from ...core.numbers import Infinity, Integer, NegativeInfinity, oo
from ...integrals.integral import Integral
from ...integrals.risch.risch import NonElementaryIntegral
from ...polys.cad.indexedroot import IndexedRoot
from ...polys.complexrootof import ComplexRootOf
from ...polys.rootoftools import RootSum
from ...series.formal import FiniteFormalPowerSeries, FormalPowerSeries
from ...series.fourier import FiniteFourierSeries, FourierSeries
from ...series.sequences import EmptySequence, RecursiveSeq, SeqBase
from .engine import PrettyPrinter, RuleEntry, rule_for
from .layout import Binding, Context, Picture, Position, PrettyResult

__all__ = ['OTHER_RULES']


def _quantifier(p: PrettyPrinter, e: Quantifier, context: Context) -> PrettyResult:
    if not p.unicode or not isinstance(e, (ForAll, Exists)):
        variables = e.variables
        first = p.operand(variables[0], context.argument()) if len(variables) == 1 else \
            p.sequence(variables, context, '(', ')').picture
        return p.applied(Picture.text(type(e).__name__),
                         p.join([first, p.operand(e.formula, context.argument())], ', '))
    sign = p.symbology.atom('ForAll' if isinstance(e, ForAll) else 'Exists', type(e).__name__)
    head = p.arguments(e.variables, context).ltext(sign)
    # a prefix operator: its operand is bracketed unless it is an atom,
    # an application, a negation or another quantified formula
    formula = p.operand(e.formula, context.child(Position.AFFIX_OPERAND, Binding.AFFIX))
    return PrettyResult(Picture.beside(head, Picture.text(' '), formula), Binding.AFFIX)


def _complex_root_of(p: PrettyPrinter, e: ComplexRootOf, context: Context) -> PrettyResult:
    inside = p.join([p.operand(e.expr, context.argument()), Picture.text(str(e.index))], ', ')
    return p.applied(Picture.text('CRootOf'), inside)


def _root_sum(p: PrettyPrinter, e: RootSum, context: Context) -> PrettyResult:
    return p.applied(Picture.text('RootSum'), p.join([p.operand(e.expr, context.argument()),
                                                      p.operand(e.fun, context.argument())], ', '))


def _indexed_root(p: PrettyPrinter, e: IndexedRoot, context: Context) -> PrettyResult:
    inside = p.join([p.operand(e.expr, context.argument()), p.operand(e.variable, context.argument()),
                     Picture.text(str(e.index))], ', ')
    return p.applied(Picture.text('IndexedRoot'), inside)


def _non_elementary_integral(p: PrettyPrinter, e: NonElementaryIntegral, context: Context) -> PrettyResult:
    # an integral which is proved not to be elementary is written as an integral
    return p.result(Integral(e.function, e.variable), context)


def _empty_sequence(p: PrettyPrinter, e: EmptySequence, context: Context) -> PrettyResult:
    return p.atom('EmptySequence')


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_SeqFormula
def _sequence(p: PrettyPrinter, e: SeqBase, context: Context) -> PrettyResult:
    start, stop = e.start, e.stop
    if isinstance(e, RecursiveSeq) or start.free_symbols or stop.free_symbols:
        # (SymPy raises NotImplementedError for symbolic bounds)
        return p.generic(e, context)
    dots = Picture.text(p.symbology.atom('Dots', '...'))
    items: list[Expr]
    leading = False
    trailing = False
    if isinstance(start, NegativeInfinity):
        if isinstance(stop, Infinity):
            return p.generic(e, context)
        items = [e.coeff(stop - Integer(k)) for k in (3, 2, 1, 0)]
        leading = True
    elif isinstance(stop, Infinity) or (isinstance(e.length, Integer) and e.length.p > 4):
        items = e[:4]
        trailing = True
    else:
        items = list(e)
    pictures = [p.operand(item, context.argument()) for item in items]
    if leading:
        pictures.insert(0, dots)
    if trailing:
        pictures.append(dots)
    return PrettyResult(p.parens(p.join(pictures, ', '), '[', ']', True), Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_FormalPowerSeries
def _formal_power_series(p: PrettyPrinter, e: FormalPowerSeries, context: Context) -> PrettyResult:
    return p.result(e.infinite, context)


def _finite_formal_power_series(p: PrettyPrinter, e: FiniteFormalPowerSeries, context: Context) -> PrettyResult:
    # a product, a composition or an inverse of series has no closed
    # form: its first terms with the order term
    return p.result(e.truncate(), context)


def _finite_fourier_series(p: PrettyPrinter, e: FiniteFourierSeries, context: Context) -> PrettyResult:
    return p.result(e.truncate(oo), context)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_FourierSeries
def _fourier_series(p: PrettyPrinter, e: FourierSeries, context: Context) -> PrettyResult:
    an, bn = e.an.formula, e.bn.formula
    if an.is_Number and an == 0 and bn.is_Number and bn == 0:
        return p.result(e.a0, context)
    c = context.child(Position.INFIX_FIRST, Binding.ADD)
    partial = p.bracketed(p.result(e.truncate(), c), c).picture
    dots = p.symbology.atom('Dots', '...')
    return PrettyResult(partial.rtext(' + ' + dots), Binding.ADD)


OTHER_RULES: tuple[RuleEntry, ...] = (
    rule_for(Quantifier, _quantifier),
    rule_for(ComplexRootOf, _complex_root_of),
    rule_for(RootSum, _root_sum),
    rule_for(SeqBase, _sequence),
    rule_for(EmptySequence, _empty_sequence),
    rule_for(FormalPowerSeries, _formal_power_series),
    rule_for(FourierSeries, _fourier_series),
    rule_for(FiniteFormalPowerSeries, _finite_formal_power_series),
    rule_for(FiniteFourierSeries, _finite_fourier_series),
    rule_for(IndexedRoot, _indexed_root),
    rule_for(NonElementaryIntegral, _non_elementary_integral),
)
