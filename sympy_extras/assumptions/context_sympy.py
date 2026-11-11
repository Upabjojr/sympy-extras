"""The SymPy face of the global assumptions and of ``assuming``
(:mod:`sympy_extras.assumptions.context`).

:data:`global_assumptions` is the same collection as
:data:`sympy_extras.assumptions.context.global_assumptions`, seen with
SymPy objects (they are converted by :mod:`sympy_extras.interop` when
added and when read). :func:`assuming` adds assumptions to it for the
duration of a ``with`` block, like ``Assuming[assum, expr]``, and pushes
the equivalent predicates of :mod:`sympy.assumptions` to SymPy's own
global assumptions, so that :func:`sympy.ask` and :func:`sympy.refine`
see them too.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Final, Iterable, Iterator, Optional, Union

from sympy.assumptions import global_assumptions as _sympy_global_assumptions
from sympy.core.basic import Basic
from sympy.core.sympify import sympify
from sympy.logic.boolalg import And, Boolean

from sympy_extras._typing_sympy import as_boolean
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.interop import from_sympy, to_sympy

from . import context as native
from .facts_sympy import Facts

__all__ = ['AssumptionsContext', 'global_assumptions', 'assuming']


def _core(b: Union[Basic, bool]) -> CoreBoolean:
    converted = from_sympy(sympify(b))
    if not isinstance(converted, CoreBoolean):
        raise TypeError("expected a Boolean, got %s" % (b,))
    return converted


class AssumptionsContext:
    """An ordered collection of assumptions, SymPy objects in and out
    (kept as the nodes of the core in a native
    :class:`sympy_extras.assumptions.context.AssumptionsContext`).

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions.context_sympy import global_assumptions
    >>> from sympy_extras.assumptions.ask_sympy import ask
    >>> global_assumptions.add(x > 0)
    >>> global_assumptions
    AssumptionsContext([x > 0])
    >>> ask(x**3 > 0)
    True
    >>> global_assumptions.clear()
    >>> ask(x**3 > 0) is None
    True
    """

    __slots__ = ('_native',)

    def __init__(self, assumptions: Iterable[Union[Boolean, bool]] = (),
                 native_context: Optional[native.AssumptionsContext] = None) -> None:
        self._native: Final[native.AssumptionsContext] = \
            native.AssumptionsContext() if native_context is None else native_context
        for a in assumptions:
            self.add(a)

    @property
    def native(self) -> native.AssumptionsContext:
        """The collection of the core holding the assumptions."""
        return self._native

    def add(self, *assumptions: Union[Boolean, bool]) -> None:
        """Add one or more assumptions."""
        self._native.add(*[_core(a) for a in assumptions])

    def remove(self, *assumptions: Union[Boolean, bool]) -> None:
        """Remove assumptions; missing ones are ignored."""
        self._native.remove(*[_core(a) for a in assumptions])

    def clear(self) -> None:
        """Remove all assumptions."""
        self._native.clear()

    def __iter__(self) -> Iterator[Boolean]:
        return iter([as_boolean(to_sympy(a)) for a in self._native])

    def __len__(self) -> int:
        return len(self._native)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, (Basic, bool)):
            return _core(item) in self._native
        return False

    def __bool__(self) -> bool:
        return bool(self._native)

    def __repr__(self) -> str:
        return "AssumptionsContext(%s)" % (list(self),)

    def as_boolean(self) -> Boolean:
        """The conjunction of the assumptions."""
        return as_boolean(And(*self))


#: The assumptions in force everywhere, like Mathematica's ``$Assumptions``
#: (the native ``global_assumptions`` seen with SymPy objects).
global_assumptions: Final[AssumptionsContext] = AssumptionsContext(native_context=native.global_assumptions)


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
    >>> from sympy_extras.assumptions.context_sympy import assuming
    >>> from sympy_extras.assumptions.refine_sympy import refine
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
