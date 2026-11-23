r"""The LaTeX rules of the objects which are not functions: piecewise
expressions, sums, products and integrals, sets, quantifiers, sequences
and series, roots of polynomials.

A typed port of the corresponding printers of SymPy's
``printing/latex.py``; the rules of the classes SymPy does not have
(quantifiers, ``IndexedRoot``, ``TranscendentalRoot``, ``IntegralByRanges``,
sets of points and of sets) are original.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/latex.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; rules in a table by class returning the binding of
# the printed form; the defects of sympy/sympy#30844 fixed.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Sequence, Union

from sympy_extras.assumptions.quantifiers import Exists, ForAll
from sympy_extras.concrete.expr_with_limits import ExprWithLimits, Product, Sum
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import BooleanTrue
from sympy_extras.core.numbers import Infinity, NegativeInfinity
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy_extras.integrals.integral import Integral
from sympy_extras.integrals.regions import IntegralByRanges
from sympy_extras.integrals.risch.risch import NonElementaryIntegral
from sympy_extras.polys.cad.indexedroot import IndexedRoot
from sympy_extras.polys.complexrootof import ComplexRootOf
from sympy_extras.polys.rootoftools import RootSum
from sympy_extras.series.formal import FiniteFormalPowerSeries, FormalPowerSeries
from sympy_extras.series.fourier import FiniteFourierSeries, FourierSeries
from sympy_extras.series.sequences import EmptySequence, RecursiveSeq, SeqAdd, SeqBase, SeqFormula, SeqMul, SeqPer
from sympy_extras.sets.conditionset import ConditionSet
from sympy_extras.sets.fancysets import (ComplexesClass, ImageSet, IntegersClass, Naturals0Class, NaturalsClass,
    RationalsClass)
from sympy_extras.sets.powerset import PowerSet
from sympy_extras.sets.sets import (Complement, Contains, EmptySetClass, FiniteSet, Intersection, Interval, PointSet,
    ProductSet, RealsClass, Set, SetOfSets, Union as SetUnion, UniversalSetClass)
from sympy_extras.solvers.isolation import TranscendentalRoot

from .framework import Binding, Result
from .latex import RULES, Ctx, LatexPrinter, Res

__all__: list[str] = []


# ----------------------------------------------------------------------
# piecewise


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Piecewise
@RULES.of(Piecewise)
def _piecewise(p: LatexPrinter, e: Piecewise, ctx: Ctx) -> Res:
    rows: list[str] = []
    pairs = e.args
    for i, pair in enumerate(pairs):
        if i == len(pairs) - 1 and isinstance(pair.cond, BooleanTrue):
            rows.append(r'%s & \text{otherwise}' % p.inside(pair.expr, ctx))
        else:
            rows.append(r'%s & \text{for}\: %s' % (p.inside(pair.expr, ctx), p.inside(pair.cond, ctx)))
    # open on the right: bracketed wherever it is not delimited (SymPy raises under a power)
    return Result(r'\begin{cases} %s \end{cases}' % r' \\'.join(rows), Binding.BLOCK)


@RULES.of(ExprCondPair)
def _expr_cond_pair(p: LatexPrinter, e: ExprCondPair, ctx: Ctx) -> Res:
    return Result(r'\left( %s%s \  %s\right)' % (p.inside(e.expr, ctx), p.comma, p.inside(e.cond, ctx)),
                  Binding.ATOM)


# ----------------------------------------------------------------------
# sums, products, integrals


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Sum, _print_Product
def _with_limits(p: LatexPrinter, e: ExprWithLimits, sign: str, ctx: Ctx) -> Res:
    below = ctx.subscript()
    if len(e.limits) == 1:
        v, a, b = e.limits[0]
        tex = r'%s_{%s=%s}^{%s} ' % (sign, p.print(v, below).output, p.print(a, below).output,
                                     p.print(b, ctx.superscript()).output)
    else:
        rows = [r'%s \leq %s \leq %s' % (p.print(a, below).output, p.print(v, below).output,
                                         p.print(b, below).output) for v, a, b in e.limits]
        tex = r'%s_{\substack{%s}} ' % (sign, r'\\'.join(rows))
    # the body is a product or tighter (SymPy: \sum n \bmod 3), and the whole
    # is bracketed when a factor follows it
    body = p.print(e.function, ctx.body())
    return Result(tex + body.output, Binding.PRODUCT, open_ended=True, applies=body.applies)


@RULES.of(Sum)
def _sum(p: LatexPrinter, e: Sum, ctx: Ctx) -> Res:
    return _with_limits(p, e, r'\sum', ctx)


@RULES.of(Product)
def _product(p: LatexPrinter, e: Product, ctx: Ctx) -> Res:
    return _with_limits(p, e, r'\prod', ctx)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Integral
@RULES.of(Integral)
def _integral(p: LatexPrinter, e: Integral, ctx: Ctx) -> Res:
    tex = r'\int'
    bounds = e.bounds
    if bounds is not None:
        if not ctx.inline and not p.settings.itex:
            tex += r'\limits'
        tex += '_{%s}^{%s}' % (p.print(bounds[0], ctx.subscript()).output,
                               p.print(bounds[1], ctx.superscript()).output)
    return _integral_form(p, tex, e.function, [e.variable], ctx)


@RULES.of(NonElementaryIntegral)
def _non_elementary_integral(p: LatexPrinter, e: NonElementaryIntegral, ctx: Ctx) -> Res:
    return _integral_form(p, r'\int', e.function, [e.variable], ctx)


@RULES.of(IntegralByRanges)
def _integral_by_ranges(p: LatexPrinter, e: IntegralByRanges, ctx: Ctx) -> Res:
    tex = r'\int\limits_{%s}' % p.print(e.condition, ctx.subscript()).output
    return _integral_form(p, tex, e.integrand, e.variables, ctx)


def _integral_form(p: LatexPrinter, sign: str, integrand: Expr, variables: Sequence[Symbol], ctx: Ctx) -> Res:
    body = p.print(integrand, ctx.body())
    differentials = ''.join(r'\, %s%s' % (p.diff_operator, p.inside(v, ctx)) for v in variables)
    return Result('%s %s%s' % (sign, body.output, differentials), Binding.PRODUCT, open_ended=True,
                  applies=body.applies)


# ----------------------------------------------------------------------
# sets


def _operation(p: LatexPrinter, operands: Sequence[Basic], sign: str, ctx: Ctx) -> Res:
    """The operands of a set operation: a nested operation is bracketed
    (``A \\ (B \\ C)``)."""
    inner = ctx.right(Binding.SET_OPERATION)
    return Result((' %s ' % sign).join(p.print(s, inner).output for s in operands), Binding.SET_OPERATION)


@RULES.of(SetUnion)
def _union(p: LatexPrinter, e: SetUnion, ctx: Ctx) -> Res:
    return _operation(p, e.args, r'\cup', ctx)


@RULES.of(Intersection)
def _intersection(p: LatexPrinter, e: Intersection, ctx: Ctx) -> Res:
    return _operation(p, e.args, r'\cap', ctx)


@RULES.of(Complement)
def _complement(p: LatexPrinter, e: Complement, ctx: Ctx) -> Res:
    return _operation(p, e.args, r'\setminus', ctx)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_ProductSet
@RULES.of(ProductSet)
def _product_set(p: LatexPrinter, e: ProductSet, ctx: Ctx) -> Res:
    sets = e.args
    if len(sets) >= 1 and all(s == sets[0] for s in sets):
        return Result('%s^{%d}' % (p.print(sets[0], ctx.base(None)).output, len(sets)), Binding.POWER)
    return _operation(p, sets, r'\times', ctx)


@RULES.of(EmptySetClass)
def _empty_set(p: LatexPrinter, e: EmptySetClass, ctx: Ctx) -> Res:
    return Result(r'\emptyset', Binding.ATOM)


@RULES.of(UniversalSetClass)
def _universal_set(p: LatexPrinter, e: UniversalSetClass, ctx: Ctx) -> Res:
    return Result(r'\mathbb{U}', Binding.ATOM)


_NAMED_SETS: tuple[tuple[type[Set], str], ...] = (
    (NaturalsClass, r'\mathbb{N}'), (Naturals0Class, r'\mathbb{N}_0'), (IntegersClass, r'\mathbb{Z}'),
    (RationalsClass, r'\mathbb{Q}'), (RealsClass, r'\mathbb{R}'), (ComplexesClass, r'\mathbb{C}'),
)

_NAMED_SET_OF: dict[type[Set], str] = dict(_NAMED_SETS)


def _named_set(p: LatexPrinter, e: Set, ctx: Ctx) -> Res:
    name = _NAMED_SET_OF[type(e)]
    return Result(name, Binding.ATOM, scripted='_' in name)


for _named, _ in _NAMED_SETS:
    RULES.register(_named, _named_set)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Interval
@RULES.of(Interval)
def _interval(p: LatexPrinter, e: Interval, ctx: Ctx) -> Res:
    if e.start == e.end:
        return Result(r'\left\{%s\right\}' % p.inside(e.start, ctx), Binding.ATOM)
    # an infinite end is open, however the interval was built
    left = '(' if e.left_open or isinstance(e.start, NegativeInfinity) else '['
    right = ')' if e.right_open or isinstance(e.end, Infinity) else ']'
    return Result(r'\left%s%s%s %s\right%s' % (left, p.inside(e.start, ctx), p.comma, p.inside(e.end, ctx), right),
                  Binding.ATOM)


def _braces(p: LatexPrinter, elements: Sequence[Basic], ctx: Ctx) -> Res:
    ordered: list[Basic] = sorted(elements, key=default_sort_key)
    return Result(r'\left\{%s\right\}' % p.join(ordered, ctx), Binding.ATOM)


@RULES.of(FiniteSet)
def _finite_set(p: LatexPrinter, e: FiniteSet, ctx: Ctx) -> Res:
    return _braces(p, e.elements, ctx)


@RULES.of(PointSet)
def _point_set(p: LatexPrinter, e: PointSet, ctx: Ctx) -> Res:
    return _braces(p, e.points, ctx)


@RULES.of(SetOfSets)
def _set_of_sets(p: LatexPrinter, e: SetOfSets, ctx: Ctx) -> Res:
    return _braces(p, e.elements, ctx)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_ImageSet
@RULES.of(ImageSet)
def _image_set(p: LatexPrinter, e: ImageSet, ctx: Ctx) -> Res:
    side = ctx.side()
    members = ', '.join(r'%s \in %s' % (p.inside(v, ctx), p.print(s, side).output)
                        for v, s in zip(e.variables, e.base_sets))
    return Result(r'\left\{%s\; \middle|\; %s\right\}' % (p.inside(e.image, ctx), members), Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_ConditionSet
@RULES.of(ConditionSet)
def _condition_set(p: LatexPrinter, e: ConditionSet, ctx: Ctx) -> Res:
    sym = e.sym
    variables = p.join(sym.args, ctx, ', ') if sym.args and not isinstance(sym, Expr) else p.inside(sym, ctx)
    if isinstance(e.base_set, UniversalSetClass):
        return Result(r'\left\{%s\; \middle|\; %s \right\}' % (variables, p.inside(e.condition, ctx)), Binding.ATOM)
    # the condition is an operand of the conjunction (SymPy: x \in S \wedge a \vee b)
    condition = p.print(e.condition, ctx.connective()).output
    base = p.print(e.base_set, ctx.side()).output
    return Result(r'\left\{%s\; \middle|\; %s \in %s \wedge %s \right\}' % (variables, variables, base, condition),
                  Binding.ATOM)


@RULES.of(PowerSet)
def _power_set(p: LatexPrinter, e: PowerSet, ctx: Ctx) -> Res:
    return Result(r'\mathcal{P}\left(%s\right)' % p.inside(e.arg, ctx), Binding.APPLICATION, applies=True)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Contains
@RULES.of(Contains)
def _contains(p: LatexPrinter, e: Contains, ctx: Ctx) -> Res:
    element, container = e.args
    return Result(r'%s \in %s' % (p.print(element, ctx.side()).output, p.print(container, ctx.side()).output),
                  Binding.RELATION)


# ----------------------------------------------------------------------
# quantifiers


def _quantifier(p: LatexPrinter, e: Union[ForAll, Exists], sign: str, ctx: Ctx) -> Res:
    variables = ', '.join(p.inside(v, ctx) for v in e.variables)
    # the formula extends to the right: bracketed wherever it is not delimited
    return Result(r'%s %s\colon %s' % (sign, variables, p.inside(e.formula, ctx)), Binding.BLOCK)


@RULES.of(ForAll)
def _for_all(p: LatexPrinter, e: ForAll, ctx: Ctx) -> Res:
    return _quantifier(p, e, r'\forall', ctx)


@RULES.of(Exists)
def _exists(p: LatexPrinter, e: Exists, ctx: Ctx) -> Res:
    return _quantifier(p, e, r'\exists', ctx)


# ----------------------------------------------------------------------
# sequences and series


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_SeqFormula
@RULES.of(SeqFormula, SeqPer, SeqAdd, SeqMul)
def _sequence(p: LatexPrinter, e: SeqBase, ctx: Ctx) -> Res:
    start, stop = e.start, e.stop
    if isinstance(e, SeqFormula) and (start.free_symbols or stop.free_symbols):
        tex = r'\left\{%s\right\}_{%s=%s}^{%s}' % (
            p.inside(e.formula, ctx), p.print(e.variables[0], ctx.subscript()).output,
            p.print(start, ctx.subscript()).output, p.print(stop, ctx.superscript()).output)
        return Result(tex, Binding.ATOM, scripted=True)
    dots = r'\ldots'
    if isinstance(start, NegativeInfinity):
        shown = [dots] + [p.inside(e.coeff(stop - k), ctx) for k in (3, 2, 1, 0)]
    elif isinstance(stop, Infinity) or e.length.is_number is not True or e.length > 4:
        shown = [p.inside(c, ctx) for c in e[:4]] + [dots]
    else:
        shown = [p.inside(c, ctx) for c in e]
    return Result(r'\left[%s\right]' % (p.comma + ' ').join(shown), Binding.ATOM)


@RULES.of(EmptySequence)
def _empty_sequence(p: LatexPrinter, e: EmptySequence, ctx: Ctx) -> Res:
    return Result(r'\left[\right]', Binding.ATOM)


@RULES.of(RecursiveSeq)
def _recursive_sequence(p: LatexPrinter, e: RecursiveSeq, ctx: Ctx) -> Res:
    conditions = [p.inside(e.recurrence, ctx)] + [p.inside(c, ctx) for c in e.initial.args]
    tex = r'\left\{%s\; \middle|\; %s\right\}' % (p.inside(e.yn, ctx), (p.comma + ' ').join(conditions))
    return Result(tex, Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_FourierSeries
@RULES.of(FourierSeries)
def _fourier_series(p: LatexPrinter, e: FourierSeries, ctx: Ctx) -> Res:
    if e.an.formula.is_zero is True and e.bn.formula.is_zero is True:
        return p.rule(e.a0, ctx.same())
    return Result(p.rule(e.truncate(3), ctx.delimited()).output + r' + \ldots', Binding.SUM)


@RULES.of(FiniteFourierSeries)
def _finite_fourier_series(p: LatexPrinter, e: FiniteFourierSeries, ctx: Ctx) -> Res:
    return p.rule(e.truncate(3), ctx.same())


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_FormalPowerSeries
@RULES.of(FormalPowerSeries)
def _formal_power_series(p: LatexPrinter, e: FormalPowerSeries, ctx: Ctx) -> Res:
    return p.rule(e.infinite, ctx.same())


@RULES.of(FiniteFormalPowerSeries)
def _finite_formal_power_series(p: LatexPrinter, e: FiniteFormalPowerSeries, ctx: Ctx) -> Res:
    return p.rule(e.infinite, ctx.same())


# ----------------------------------------------------------------------
# roots


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_ComplexRootOf
@RULES.of(ComplexRootOf)
def _complex_root_of(p: LatexPrinter, e: ComplexRootOf, ctx: Ctx) -> Res:
    return Result(r'\operatorname{CRootOf} {\left(%s, %d\right)}' % (p.inside(e.expr, ctx), e.index),
                  Binding.APPLICATION, applies=True)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_RootSum
@RULES.of(RootSum)
def _root_sum(p: LatexPrinter, e: RootSum, ctx: Ctx) -> Res:
    return Result(r'\operatorname{RootSum} {\left(%s, %s\right)}' % (p.inside(e.expr, ctx), p.inside(e.fun, ctx)),
                  Binding.APPLICATION, applies=True)


@RULES.of(IndexedRoot)
def _indexed_root(p: LatexPrinter, e: IndexedRoot, ctx: Ctx) -> Res:
    tex = r'\operatorname{Root}_{%d} {\left(%s, %s\right)}' % (e.index, p.inside(e.expr, ctx),
                                                              p.inside(e.variable, ctx))
    return Result(tex, Binding.APPLICATION, applies=True)


@RULES.of(TranscendentalRoot)
def _transcendental_root(p: LatexPrinter, e: TranscendentalRoot, ctx: Ctx) -> Res:
    tex = r'\operatorname{Root} {\left(%s, %s \in \left(%s, %s\right)\right)}' % (
        p.inside(e.function, ctx), p.inside(e.variable, ctx), p.inside(e.lower, ctx), p.inside(e.upper, ctx))
    return Result(tex, Binding.APPLICATION, applies=True)
