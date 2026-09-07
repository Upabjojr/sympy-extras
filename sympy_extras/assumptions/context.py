"""Global assumptions and the ``assuming`` context manager.

:data:`global_assumptions` plays the role of Mathematica's ``$Assumptions``:
the assumptions it contains are used by every function of
:mod:`sympy_extras.assumptions` in addition to the ones passed explicitly.
:func:`assuming` adds assumptions for the duration of a ``with`` block, like
``Assuming[assum, expr]``; the equivalent predicates of
:mod:`sympy.assumptions` are also pushed to SymPy's own global assumptions,
so that :func:`sympy.ask` and :func:`sympy.refine` see them too.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Union

from sympy.assumptions import global_assumptions as _sympy_global_assumptions
from sympy.core.sympify import sympify
from sympy.logic.boolalg import And, Boolean

from sympy_extras._typing import as_boolean

from .facts import Facts

__all__ = ['AssumptionsContext', 'global_assumptions', 'assuming']


class AssumptionsContext:
    """An ordered collection of assumptions.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions import global_assumptions, ask
    >>> global_assumptions.add(x > 0)
    >>> global_assumptions
    AssumptionsContext([x > 0])
    >>> ask(x**3 > 0)
    True
    >>> global_assumptions.clear()
    >>> ask(x**3 > 0) is None
    True
    """

    def __init__(self, assumptions: Iterable[Union[Boolean, bool]] = ()) -> None:
        self._items: list[Boolean] = []
        for a in assumptions:
            self.add(a)

    def add(self, *assumptions: Union[Boolean, bool]) -> None:
        """Add one or more assumptions."""
        for a in assumptions:
            b = as_boolean(a)
            if b not in self._items:
                self._items.append(b)

    def remove(self, *assumptions: Union[Boolean, bool]) -> None:
        """Remove assumptions; missing ones are ignored."""
        for a in assumptions:
            b = as_boolean(a)
            if b in self._items:
                self._items.remove(b)

    def clear(self) -> None:
        """Remove all assumptions."""
        self._items.clear()

    def __iter__(self) -> Iterator[Boolean]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: object) -> bool:
        return sympify(item) in self._items

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return "AssumptionsContext(%s)" % (self._items,)

    def as_boolean(self) -> Boolean:
        """The conjunction of the assumptions."""
        return And(*self._items)


#: The assumptions in force everywhere, like Mathematica's ``$Assumptions``.
global_assumptions = AssumptionsContext()


@contextmanager
def assuming(*assumptions: Union[Boolean, bool]) -> Iterator[None]:
    """Context manager adding assumptions to :data:`global_assumptions`
    for the duration of a ``with`` block, the counterpart of Mathematica's
    ``Assuming[assum, expr]``.

    The predicates of :mod:`sympy.assumptions` implied by the assumptions
    are added to SymPy's global assumptions as well.

    Examples
    ========

    >>> from sympy import Abs, sqrt, ask, Q
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions import assuming, refine
    >>> with assuming(x > 1):
    ...     print(refine(Abs(x - 1) + sqrt(x**2)))
    ...     print(ask(Q.positive(x)))
    2*x - 1
    True
    >>> refine(Abs(x - 1))
    Abs(x - 1)
    """
    items = [as_boolean(a) for a in assumptions]
    predicates = Facts(items).predicates
    added = [a for a in items if a not in global_assumptions]
    global_assumptions.add(*added)
    old_sympy = set(_sympy_global_assumptions)
    _sympy_global_assumptions.add(predicates)
    try:
        yield
    finally:
        global_assumptions.remove(*added)
        for p in list(_sympy_global_assumptions):
            if p not in old_sympy:
                _sympy_global_assumptions.remove(p)
