"""Monomial orders extending :mod:`sympy.polys.orderings`.

SymPy's orders are ``lex``, ``grlex``, ``grevlex`` (and their inverses) and
product orders built with :func:`~sympy.polys.orderings.build_product_order`,
which are not hashable and cannot be used with :func:`sympy.groebner`. The
orders here are hashable :class:`~sympy.polys.orderings.MonomialOrder`
instances usable both with :func:`sympy.groebner` and with the ring level
functions of :mod:`sympy.polys.groebnertools`.
"""
from __future__ import annotations

from typing import Sequence

from sympy.polys.orderings import MonomialOrder, monomial_key

from sympy_extras._typing import Monomial, OrderSpec

__all__ = ['WeightOrder', 'BlockOrder', 'elimination_order', 'as_order']


def as_order(order: OrderSpec) -> MonomialOrder:
    """The :class:`~sympy.polys.orderings.MonomialOrder` for a name or an
    order."""
    result = monomial_key(order)
    if not isinstance(result, MonomialOrder):
        raise TypeError("a monomial order or its name is expected, got %s" % (order,))
    return result


class WeightOrder(MonomialOrder):
    """A weight vector refined by another order.

    Monomials are compared first by the weighted degree ``sum(w_i*e_i)``
    and ties are broken with ``tail`` (``'lex'`` by default). With
    non-negative weights this is a monomial order. These are the orders
    used along a Gröbner walk.

    Examples
    ========

    >>> from sympy import groebner
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.orderings import WeightOrder
    >>> list(groebner([x**2 - y, x*y - 1], x, y, order=WeightOrder((1, 3), 'lex')))
    [x**3 - 1, -x**2 + y]
    """
    alias = 'weight'
    is_global = True

    def __init__(self, weights: Sequence[int], tail: OrderSpec = 'lex') -> None:
        self.weights = tuple(weights)
        if any(w < 0 for w in self.weights):
            raise ValueError("the weights must be non-negative")
        self.tail = as_order(tail)

    def __call__(self, monomial: Monomial) -> tuple[int, object]:
        return (sum(w*e for w, e in zip(self.weights, monomial)), self.tail(monomial))

    def __hash__(self) -> int:
        return hash((self.__class__, self.weights, self.tail))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WeightOrder) and \
            (self.weights, self.tail) == (other.weights, other.tail)

    def __str__(self) -> str:
        return "WeightOrder(%s, %s)" % (self.weights, self.tail)

    __repr__ = __str__


class BlockOrder(MonomialOrder):
    """A block (product) order: the variables are split into consecutive
    blocks, each compared with its own order, the first block first.

    ``blocks`` is a sequence of pairs ``(order, size)``. A block order whose
    first block contains the variables to eliminate is an *elimination
    order*: the elements of a Gröbner basis without those variables
    generate the elimination ideal, see :func:`elimination_order`.

    Examples
    ========

    >>> from sympy import groebner
    >>> from sympy.abc import t, x, y
    >>> from sympy_extras.polys.orderings import BlockOrder
    >>> order = BlockOrder([('grevlex', 1), ('grevlex', 2)])
    >>> list(groebner([t - x**2, t - y], t, x, y, order=order))
    [t - y, x**2 - y]
    """
    alias = 'block'
    is_global = True

    def __init__(self, blocks: Sequence[tuple[OrderSpec, int]]) -> None:
        self.blocks = tuple((as_order(order), int(size)) for order, size in blocks)
        if any(size <= 0 for _, size in self.blocks):
            raise ValueError("block sizes must be positive")

    def __call__(self, monomial: Monomial) -> tuple[object, ...]:
        key: list[object] = []
        start = 0
        for order, size in self.blocks:
            key.append(order(monomial[start:start + size]))
            start += size
        return tuple(key)

    def __hash__(self) -> int:
        return hash((self.__class__, self.blocks))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BlockOrder) and self.blocks == other.blocks

    def __str__(self) -> str:
        return "BlockOrder(%s)" % ", ".join("(%s, %d)" % (o, s) for o, s in self.blocks)

    __repr__ = __str__


def elimination_order(neliminate: int, nvars: int, order: OrderSpec = 'grevlex') -> MonomialOrder:
    """The block order eliminating the first ``neliminate`` of ``nvars``
    variables, with ``order`` on each block."""
    if neliminate <= 0:
        return as_order(order)
    if neliminate >= nvars:
        return as_order(order)
    return BlockOrder([(order, neliminate), (order, nvars - neliminate)])
