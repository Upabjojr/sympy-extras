"""Sets of the elements of a set satisfying a condition on the typed core:
:class:`ConditionSet` (unevaluated) and :func:`condition_set`
(evaluated), as SymPy's ``sympy.sets.conditionset``.

The unknown of the condition is a symbol, or a tuple of symbols for a
set of points (``ConditionSet((x, y), x < y, ProductSet(Reals, Reals))``).
:func:`condition_set` applies the rules of SymPy's constructor: a true
condition gives the base set, a false one or an empty base set the empty
set, the elements of a finite base set at which the condition is decided
are sorted out, nested condition sets on the same unknown are merged and
``Contains(x, A)`` as the condition is the intersection with ``A``.

Examples
========

>>> from sympy_extras.core import Symbol, Tuple
>>> from sympy_extras.sets import Reals, finite_set, interval
>>> from sympy_extras.sets.conditionset import ConditionSet, condition_set
>>> x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
>>> condition_set(x, x > 0, finite_set(-1, 2, a))
Union(ConditionSet(x, x > 0, {a}), {2})
>>> condition_set(x, x > 0, interval(-1, 1)), condition_set(x, x > 0, finite_set(3))
(ConditionSet(x, x > 0, Interval(-1, 1)), {3})
>>> c = ConditionSet(Tuple(x, y), x < y, Reals)
>>> c, c.sym, c.free_symbols
(ConditionSet((x, y), x < y, Reals), (x, y), set())
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/sets/conditionset.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the class never evaluates and condition_set holds the
# rules of SymPy's __new__ (no deprecated set conditions, no signature
# check of nested tuples); a tuple unknown only takes points.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional

from sympy_extras.core.basic import Basic, NodePrinter
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, BooleanLike, and_, false, to_boolean, true
from sympy_extras.core.symbol import Symbol

from .sets import Contains, EmptySet, EmptySetClass, FiniteSet, Set, UniversalSet, finite_set, intersection, union

__all__ = ['ConditionSet', 'condition_set']


def _unknowns(sym: Basic) -> tuple[Symbol, ...]:
    """The symbols of the unknown of a condition set: a symbol or a tuple
    of symbols."""
    if isinstance(sym, Symbol):
        return (sym,)
    if isinstance(sym, Tuple) and all(isinstance(s, Symbol) for s in sym.args):
        return tuple(s for s in sym.args if isinstance(s, Symbol))
    raise TypeError('the unknown of a ConditionSet is a symbol or a tuple of symbols, got %s' % (sym,))


# PORTED FROM SYMPY: sympy/sets/conditionset.py:ConditionSet
class ConditionSet(Set):
    """The elements of ``base_set`` at which ``condition`` holds, as a
    condition on ``sym`` (unevaluated: :func:`condition_set` evaluates)."""

    __slots__ = ('_sym', '_condition', '_base_set')

    def __init__(self, sym: Basic, condition: BooleanLike, base_set: Set = UniversalSet) -> None:
        _unknowns(sym)
        self._sym: Final[Basic] = sym
        self._condition: Final[Boolean] = to_boolean(condition)
        self._base_set: Final[Set] = base_set
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._sym, self._condition, self._base_set)

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        sym, condition, base_set = args
        if not isinstance(condition, Boolean) or not isinstance(base_set, Set):
            raise TypeError('ConditionSet takes an unknown, a condition and a set')
        return ConditionSet(sym, condition, base_set)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        sym, condition, base_set = args
        if not isinstance(condition, Boolean) or not isinstance(base_set, Set):
            raise TypeError('ConditionSet takes an unknown, a condition and a set')
        return condition_set(sym, condition, base_set)

    @property
    def sym(self) -> Basic:
        """The unknown: a symbol, or a tuple of symbols."""
        return self._sym

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """The symbols of the unknown."""
        return _unknowns(self._sym)

    @property
    def condition(self) -> Boolean:
        return self._condition

    @property
    def base_set(self) -> Set:
        return self._base_set

    # PORTED FROM SYMPY: sympy/sets/conditionset.py:ConditionSet.free_symbols
    @property
    def free_symbols(self) -> set[Symbol]:
        return (self._condition.free_symbols - set(self.symbols)) | self._base_set.free_symbols

    def _sympystr(self, printer: NodePrinter) -> str:
        return 'ConditionSet(%s, %s, %s)' % (printer.doprint(self._sym), printer.doprint(self._condition),
                                             printer.doprint(self._base_set))

    # PORTED FROM SYMPY: sympy/sets/conditionset.py:ConditionSet._contains
    def _contains(self, other: Expr) -> Optional[Boolean]:
        if not isinstance(self._sym, Symbol):
            # an expression is not a point
            return false
        # SymPy's Contains(other, base_set): kept unevaluated unless decided
        decided = self._base_set.contains(other)
        base_cond = decided if decided is true or decided is false else Contains(other, self._base_set)
        if base_cond is false:
            return false
        return and_(base_cond, to_boolean(self._condition.subs(self._sym, other)))

    # PORTED FROM SYMPY: sympy/sets/conditionset.py:ConditionSet.as_relational
    def as_relational(self, x: Expr) -> Boolean:
        if not isinstance(self._sym, Symbol):
            raise TypeError('the condition of a set of points is not a relation of an expression')
        return and_(to_boolean(self._condition.subs(self._sym, x)), self._base_set.contains(x))


def _fuzzy(b: Basic) -> Optional[bool]:
    if b is true:
        return True
    if b is false:
        return False
    return None


# PORTED FROM SYMPY: sympy/sets/conditionset.py:ConditionSet.__new__
def condition_set(sym: Basic, condition: BooleanLike, base_set: Set = UniversalSet) -> Set:
    """The elements of ``base_set`` at which ``condition`` holds, evaluated
    as SymPy's ``ConditionSet`` constructor does."""
    unknowns = _unknowns(sym)
    if len(set(unknowns)) != len(unknowns):
        raise ValueError('Duplicate symbols detected')
    cond = to_boolean(condition)
    if cond is true:
        return base_set
    if cond is false:
        return EmptySet
    if isinstance(base_set, EmptySetClass):
        return EmptySet
    if isinstance(sym, Symbol) and base_set.contains(sym) is false:
        raise TypeError('sym `%s` is not in base_set `%s`' % (sym, base_set))
    know: Optional[Set] = None
    base = base_set
    if isinstance(base, FiniteSet) and isinstance(sym, Symbol):
        kept: list[Expr] = []
        undecided: list[Expr] = []
        for e in base.elements:
            value = _fuzzy(cond.subs(sym, e))
            if value is True:
                kept.append(e)
            elif value is None:
                undecided.append(e)
        if not undecided:
            return finite_set(*kept)
        know = finite_set(*kept)
        base = finite_set(*undecided)
    if isinstance(base, ConditionSet):
        s, c, b = base.sym, base.condition, base.base_set
        inner = set(base.symbols)
        if s == sym:
            cond = and_(cond, c)
            base = b
        elif not (c.free_symbols & set(unknowns)):
            reps: dict[Basic, Basic] = {v: k for k, v in zip(unknowns, base.symbols)}
            cond = and_(cond, to_boolean(c._xreplace_evaluated(reps)))
            base = b
        elif not (cond.free_symbols & inner):
            reps_: dict[Basic, Basic] = {k: v for k, v in zip(unknowns, base.symbols)}
            sym = sym._xreplace_evaluated(reps_)
            cond = and_(to_boolean(cond._xreplace_evaluated(reps_)), c)
            base = b
    if isinstance(cond, Contains) and sym == cond.args[0]:
        member_of = cond.args[1]
        if isinstance(member_of, Set):
            return intersection(member_of, base)
    rv = ConditionSet(sym, cond, base)
    return rv if know is None else union(know, rv)
