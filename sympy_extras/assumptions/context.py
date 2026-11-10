"""Global assumptions and the ``assuming`` context manager, on the core.

:data:`global_assumptions` plays the role of Mathematica's ``$Assumptions``:
the assumptions it contains are used by every function of
:mod:`sympy_extras.assumptions` in addition to the ones passed explicitly.
:func:`assuming` adds assumptions for the duration of a ``with`` block, like
``Assuming[assum, expr]``; the equivalent predicates of ``Q`` are also
pushed to the global assumptions of the core
(:data:`sympy_extras.core.assume.global_assumptions`), so that the core's
``ask`` and ``refine`` see them too.

This module imports no SymPy. :mod:`sympy_extras.assumptions.context_sympy`
is its SymPy face: the same :data:`global_assumptions`, seen with SymPy
objects, and an ``assuming`` which pushes the predicates to SymPy's global
assumptions instead.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Final, Iterable, Iterator

from sympy_extras.core.assume import global_assumptions as _core_global_assumptions
from sympy_extras.core.logic import Boolean, BooleanLike, and_, to_boolean

from .facts import Facts, conjuncts

__all__ = ['AssumptionsContext', 'global_assumptions', 'assuming']


class AssumptionsContext:
    """An ordered collection of assumptions.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.assumptions.context import AssumptionsContext
    >>> x, y = symbols('x y')
    >>> c = AssumptionsContext([x > 0, y > 0, x > 0])
    >>> c, len(c), (x > 0) in c, c.as_boolean()
    (AssumptionsContext([x > 0, y > 0]), 2, True, (x > 0) & (y > 0))
    """

    __slots__ = ('_items',)

    def __init__(self, assumptions: Iterable[BooleanLike] = ()) -> None:
        self._items: Final[list[Boolean]] = []
        for a in assumptions:
            self.add(a)

    def add(self, *assumptions: BooleanLike) -> None:
        """Add one or more assumptions."""
        for a in assumptions:
            b = to_boolean(a)
            if b not in self._items:
                self._items.append(b)

    def remove(self, *assumptions: BooleanLike) -> None:
        """Remove assumptions; missing ones are ignored."""
        for a in assumptions:
            b = to_boolean(a)
            if b in self._items:
                self._items.remove(b)

    def clear(self) -> None:
        """Remove all assumptions."""
        self._items.clear()

    def __iter__(self) -> Iterator[Boolean]:
        return iter(list(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, (Boolean, bool)):
            return to_boolean(item) in self._items
        return False

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return "AssumptionsContext([%s])" % ", ".join(str(a) for a in self._items)

    def as_boolean(self) -> Boolean:
        """The conjunction of the assumptions."""
        return and_(*self._items)


#: The assumptions in force everywhere, like Mathematica's ``$Assumptions``.
global_assumptions: Final[AssumptionsContext] = AssumptionsContext()


@contextmanager
def assuming(*assumptions: BooleanLike) -> Iterator[None]:
    """Context manager adding assumptions to :data:`global_assumptions`
    for the duration of a ``with`` block, the counterpart of Mathematica's
    ``Assuming[assum, expr]``.

    The predicates of ``Q`` implied by the assumptions are added to the
    global assumptions of the core as well.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Q, ask
    >>> from sympy_extras.assumptions.context import assuming, global_assumptions
    >>> x = Symbol('x')
    >>> with assuming(x > 1):
    ...     print(global_assumptions, ask(Q.positive(x)))
    AssumptionsContext([x > 1]) True
    >>> global_assumptions, ask(Q.positive(x))
    (AssumptionsContext([]), None)
    """
    items = [to_boolean(a) for a in assumptions]
    predicates = conjuncts(Facts(items).predicates)
    added = [a for a in items if a not in global_assumptions]
    global_assumptions.add(*added)
    pushed = [p for p in predicates if p not in _core_global_assumptions]
    _core_global_assumptions.add(*pushed)
    try:
        yield
    finally:
        global_assumptions.remove(*added)
        _core_global_assumptions.remove(*pushed)
