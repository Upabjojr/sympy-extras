"""The power set of a set, the set of its subsets [1]_ [2]_.

A typed port of SymPy's ``sympy.sets.powerset``: :class:`PowerSet` is an
unevaluated node, as SymPy's (its constructor never evaluates either);
``Set.powerset`` gives the finite set of the subsets of a finite set and
``PowerSet(s).rewrite(FiniteSet)`` the same; a set is an element when it
is a subset of the argument, and a power set is a subset of another when
its argument is.

Examples
========

>>> from sympy_extras.sets import EmptySet, FiniteSet, Reals, finite_set, interval
>>> from sympy_extras.sets.powerset import PowerSet
>>> p = PowerSet(finite_set(1, 2, 3))
>>> p, PowerSet(EmptySet), PowerSet(Reals), len(p)
(PowerSet({1, 2, 3}), PowerSet(EmptySet), PowerSet(Reals), 8)
>>> p.rewrite(FiniteSet)
FiniteSet(EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3})
>>> p.contains(finite_set(1, 3)), p.contains(finite_set(4)), PowerSet(interval(0, 1)).contains(interval(0, 2))
(True, False, False)
>>> PowerSet(finite_set(1)).is_subset(p), p.is_subset(PowerSet(finite_set(1)))
(True, False)

References
==========

.. [1] https://en.wikipedia.org/wiki/Power_set
.. [2] https://en.wikipedia.org/wiki/Axiom_of_power_set
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/sets/powerset.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the membership of a set by Set._contains_set; no kind.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Iterator, Optional

from sympy_extras.core.basic import Basic, NodePrinter
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, false, true

from .sets import EmptySet, Set, _finite_of, finite_elements

__all__ = ['PowerSet']


# PORTED FROM SYMPY: sympy/sets/powerset.py:PowerSet
class PowerSet(Set):
    """The set of the subsets of a set, unevaluated.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.sets import EmptySet, finite_set
    >>> from sympy_extras.sets.powerset import PowerSet
    >>> x = Symbol('x')
    >>> p = PowerSet(finite_set(1, 2))
    >>> list(p), p.contains(EmptySet), p.contains(finite_set(x)), p.contains(x)
    ([EmptySet, {1}, {2}, {1, 2}], True, Contains({x}, PowerSet({1, 2})), Contains(x, PowerSet({1, 2})))
    >>> PowerSet(PowerSet(EmptySet)), p.arg
    (PowerSet(PowerSet(EmptySet)), {1, 2})
    """

    __slots__ = ('_arg',)

    def __init__(self, arg: Set) -> None:
        if not isinstance(arg, Set):
            raise ValueError('{} must be a set.'.format(arg))
        self._arg: Final[Set] = arg
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._arg,)

    @property
    def arg(self) -> Set:
        """The set whose subsets are the elements."""
        return self._arg

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        arg = args[0]
        if not isinstance(arg, Set):
            raise ValueError('{} must be a set.'.format(arg))
        return PowerSet(arg)

    def _sympystr(self, printer: NodePrinter) -> str:
        return 'PowerSet(%s)' % printer.doprint(self._arg)

    # PORTED FROM SYMPY: sympy/sets/powerset.py:PowerSet._contains
    def _contains(self, other: Expr) -> Optional[Boolean]:
        # an expression is not a set
        return None

    def _contains_set(self, other: Set) -> Optional[Boolean]:
        superset = self._arg.is_superset(other)
        if superset is None:
            return None
        return true if superset else false

    # PORTED FROM SYMPY: sympy/sets/powerset.py:PowerSet._eval_is_subset
    def _eval_is_subset(self, other: Set) -> Optional[bool]:
        if isinstance(other, PowerSet):
            return self._arg.is_subset(other.arg)
        return None

    def _elements(self) -> tuple[Basic, ...]:
        elements = finite_elements(self._arg)
        if elements is None:
            raise TypeError('the power set of %s is not iterable' % (self._arg,))
        return elements

    # PORTED FROM SYMPY: sympy/sets/powerset.py:PowerSet.__len__
    def __len__(self) -> int:
        return 2 ** len(self._elements())

    # PORTED FROM SYMPY: sympy/sets/powerset.py:PowerSet.__iter__
    def __iter__(self) -> Iterator[Set]:
        found: list[list[Basic]] = [[]]
        yield EmptySet
        for x in self._elements():
            temp: list[list[Basic]] = []
            for y in found:
                new = [x] + y
                yield _finite_of(new)
                temp.append(new)
            found.extend(temp)
