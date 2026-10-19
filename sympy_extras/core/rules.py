"""Replacement rules given by functions: :class:`Transform`.

``Transform(transform, filter)`` behaves as a mapping for ``xreplace``:
a node is "in" the rule when ``filter(node)`` holds, and the rule maps it
to ``transform(node)``, as SymPy's ``Transform`` [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy source: sympy/core/rules.py* (the
       documented interface only).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .basic import Basic

__all__ = ['Transform']


class Transform:
    """A rule ``node -> transform(node)`` on the nodes accepted by ``filter``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.core.rules import Transform
    >>> x = Symbol('x')
    >>> plus_one = Transform(lambda e: e + 1, lambda e: e.is_Integer)
    >>> Integer(2) in plus_one, x in plus_one, plus_one[Integer(2)]
    (True, False, 3)
    >>> (x + 2).xreplace(plus_one).doit()
    x + 3
    """

    __slots__ = ('transform', 'filter')

    def __init__(self, transform: Callable[[Basic], Basic],
                 filter: Callable[[Basic], bool] = lambda node: True) -> None:
        self.transform: Callable[[Basic], Basic] = transform
        self.filter: Callable[[Basic], bool] = filter

    def __contains__(self, item: Basic) -> bool:
        return self.filter(item)

    def __getitem__(self, item: Basic) -> Basic:
        if not self.filter(item):
            raise KeyError(item)
        return self.transform(item)
