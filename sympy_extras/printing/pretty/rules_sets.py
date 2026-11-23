"""The pretty-printing rules of the sets.

The layouts are SymPy's (``sympy/printing/pretty/pretty.py``). The set
operators all bind alike and an operand which is itself printed with a
set operator is bracketed, whatever its class: SymPy decides from a list
of classes per operator, which misses ``X \\ (Y \\ Z)``,
``(X \\ Y) × Z`` and ``(X ∪ Y)²`` (sympy/sympy#30844).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_ProductSet, _print_FiniteSet, _print_Interval,
# _print_Intersection, _print_Union, _print_Complement, _print_ImageSet, _print_ConditionSet,
# _print_Contains, the named sets) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; typed results and contexts; brackets from the
# binding of the printed operand.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Sequence

from ...core.basic import Basic
from ...core.numbers import Infinity, NegativeInfinity
from ...core.sorting import default_sort_key
from ...sets.conditionset import ConditionSet
from ...sets.fancysets import (ComplexesClass, ImageSet, IntegersClass, Naturals0Class, NaturalsClass,
    RationalsClass)
from ...sets.powerset import PowerSet
from ...sets.sets import (Complement, Contains, EmptySetClass, FiniteSet, Intersection, Interval, PointSet,
    ProductSet, RealsClass, SetOfSets, Union, UniversalSetClass)
from .engine import PrettyPrinter, RuleEntry, rule_for
from .layout import Binding, Context, Picture, Position, PrettyResult
from .rules_core import infix, raised

__all__ = ['SET_RULES']


def _set_operation(p: PrettyPrinter, args: Sequence[Basic], context: Context, operator: str) -> PrettyResult:
    return infix(p, args, context, operator, Binding.SET, Binding.SET, strict=True)


def _empty_set(p: PrettyPrinter, e: EmptySetClass, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('EmptySet', 'EmptySet'))


def _universal_set(p: PrettyPrinter, e: UniversalSetClass, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('Universe', 'UniversalSet'))


def _reals(p: PrettyPrinter, e: RealsClass, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('Reals', '(-oo, oo)'))


def _naturals(p: PrettyPrinter, e: NaturalsClass, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('Naturals', 'Naturals'))


def _naturals0(p: PrettyPrinter, e: Naturals0Class, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('Naturals0', 'Naturals0'))


def _integers(p: PrettyPrinter, e: IntegersClass, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('Integers', 'Integers'))


def _rationals(p: PrettyPrinter, e: RationalsClass, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('Rationals', 'Rationals'))


def _complexes(p: PrettyPrinter, e: ComplexesClass, context: Context) -> PrettyResult:
    return p.atom(p.symbology.atom('Complexes', 'Complexes'))


def _interval(p: PrettyPrinter, e: Interval, context: Context) -> PrettyResult:
    if e.start == e.end:
        return p.sequence([e.start], context, '{', '}')
    # an infinite end is not attained: it is drawn open
    left_open = e.left_open or isinstance(e.start, (Infinity, NegativeInfinity))
    right_open = e.right_open or isinstance(e.end, (Infinity, NegativeInfinity))
    return p.sequence([e.start, e.end], context, '(' if left_open else '[', ')' if right_open else ']')


def _finite_set(p: PrettyPrinter, e: FiniteSet, context: Context) -> PrettyResult:
    return p.sequence(sorted(e.args, key=default_sort_key), context, '{', '}')


def _point_set(p: PrettyPrinter, e: PointSet, context: Context) -> PrettyResult:
    return p.sequence(sorted(e.args, key=default_sort_key), context, '{', '}')


def _set_of_sets(p: PrettyPrinter, e: SetOfSets, context: Context) -> PrettyResult:
    return p.sequence(sorted(e.args, key=default_sort_key), context, '{', '}')


def _union(p: PrettyPrinter, e: Union, context: Context) -> PrettyResult:
    return _set_operation(p, e.args, context, p.symbology.atom('Union', 'U'))


def _intersection(p: PrettyPrinter, e: Intersection, context: Context) -> PrettyResult:
    return _set_operation(p, e.args, context, p.symbology.atom('Intersection', 'n'))


def _complement(p: PrettyPrinter, e: Complement, context: Context) -> PrettyResult:
    return _set_operation(p, e.args, context, '\\')


def _product_set(p: PrettyPrinter, e: ProductSet, context: Context) -> PrettyResult:
    sets = e.sets
    if len(sets) >= 1 and all(s == sets[0] for s in sets):
        # the power of a set: the base is bracketed like any base
        return raised(p, sets[0], Picture.text(str(len(sets))), context)
    return _set_operation(p, sets, context, p.symbology.atom('Multiplication', 'x'))


def _power_set(p: PrettyPrinter, e: PowerSet, context: Context) -> PrettyResult:
    return p.function('PowerSet', e.args, context)


def _contains(p: PrettyPrinter, e: Contains, context: Context) -> PrettyResult:
    if not p.unicode:
        return p.function('Contains', e.args, context)
    # a relation: ¬(x ∈ S), and the element is bracketed like a side of a relation
    element = p.operand(e.args[0], context.child(Position.RELATION_LEFT, Binding.RELATION, strict=True))
    domain = p.operand(e.args[1], context.child(Position.RELATION_RIGHT, Binding.RELATION, strict=True))
    return PrettyResult(Picture.beside(element, Picture.text(' %s ' % p.symbology.atom('ElementOf', 'in')), domain),
                        Binding.RELATION)


def _membership(p: PrettyPrinter, variable: Basic, domain: Basic, context: Context) -> Picture:
    inn = p.symbology.atom('SmallElementOf', 'in')
    return Picture.beside(p.operand(variable, context.argument()), Picture.text(' %s ' % inn),
                          p.operand(domain, context.child(Position.RELATION_RIGHT, Binding.RELATION, strict=True)))


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_ImageSet
def _image_set(p: PrettyPrinter, e: ImageSet, context: Context) -> PrettyResult:
    image = p.operand(e.image, context.argument())
    members = [_membership(p, v, s, context) for v, s in zip(e.variables, e.base_sets)]
    return PrettyResult(p.separated(image, p.join(members, ', '), '{', '}'), Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_ConditionSet
def _condition_set(p: PrettyPrinter, e: ConditionSet, context: Context) -> PrettyResult:
    variables = p.operand(e.sym, context.argument())
    if isinstance(e.base_set, UniversalSetClass):
        condition = p.operand(e.condition, context.argument())
        return PrettyResult(p.separated(variables, condition, '{', '}'), Binding.ATOM)
    # the condition is an operand of "and": a formula with a connective is bracketed
    condition = p.operand(e.condition, context.child(Position.INFIX_LAST, Binding.RELATION))
    described = Picture.beside(_membership(p, e.sym, e.base_set, context),
                               Picture.text(' %s ' % p.symbology.atom('And', 'and')), condition)
    return PrettyResult(p.separated(variables, described, '{', '}'), Binding.ATOM)


SET_RULES: tuple[RuleEntry, ...] = (
    rule_for(EmptySetClass, _empty_set),
    rule_for(UniversalSetClass, _universal_set),
    rule_for(Interval, _interval),
    rule_for(RealsClass, _reals),
    rule_for(NaturalsClass, _naturals),
    rule_for(Naturals0Class, _naturals0),
    rule_for(IntegersClass, _integers),
    rule_for(RationalsClass, _rationals),
    rule_for(ComplexesClass, _complexes),
    rule_for(FiniteSet, _finite_set),
    rule_for(PointSet, _point_set),
    rule_for(SetOfSets, _set_of_sets),
    rule_for(Union, _union),
    rule_for(Intersection, _intersection),
    rule_for(Complement, _complement),
    rule_for(ProductSet, _product_set),
    rule_for(PowerSet, _power_set),
    rule_for(Contains, _contains),
    rule_for(ImageSet, _image_set),
    rule_for(ConditionSet, _condition_set),
)
