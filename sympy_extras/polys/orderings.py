"""Monomial orders of the typed polynomial layer (no SymPy): ``lex``,
``grlex``, ``grevlex``, weight vectors refined by another order
(:class:`WeightOrder`, the orders along a Gröbner walk) and block orders
(:class:`BlockOrder`, :func:`elimination_order`), as key functions on
exponent tuples (the counterpart of SymPy's :mod:`sympy.polys.orderings`).

These are the orders of :class:`~sympy_extras.polys.rings.PolyRing`, which
re-exports them; :mod:`sympy_extras.polys.orderings_sympy` is the SymPy
face: the same weight and block orders as SymPy ``MonomialOrder``
instances, for SymPy's ``groebner`` and rings.

Examples
========

>>> from sympy_extras.polys.orderings import BlockOrder, WeightOrder, grevlex, lex
>>> W = WeightOrder((1, 3), lex)
>>> W((2, 0)) < W((0, 1))
True
>>> B = BlockOrder([(grevlex, 1), (grevlex, 2)])
>>> sorted([(0, 2, 0), (1, 0, 0), (0, 1, 1)], key=B)
[(0, 1, 1), (0, 2, 0), (1, 0, 0)]

References
==========

.. [1] D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*,
       4th ed., Springer, 2015, Chapter 2, Section 2 (monomial orders) and
       Chapter 3, Section 1 (elimination).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Sequence, Union

__all__ = ['Monomial', 'OrderValue', 'OrderKeyValue', 'MonomialOrder', 'lex', 'grlex', 'grevlex', 'as_order',
           'WeightOrder', 'BlockOrder', 'elimination_order']

#: the exponents of a monomial (the same alias as ``sympy_extras._typing.Monomial``)
Monomial = tuple[int, ...]
#: an entry of the key of a monomial order (weights may be rational)
OrderValue = Union[int, Fraction]
#: the key of a monomial under an order: compared lexicographically
OrderKeyValue = tuple[OrderValue, ...]


class MonomialOrder:
    """A monomial order given by a key function on exponent tuples (a
    larger key is a larger monomial) and a name; ``is_graded`` says
    whether the total degree comes first."""

    def __init__(self, name: str, key: Callable[[Monomial], OrderKeyValue], is_graded: bool) -> None:
        self.name: str = name
        self.key: Callable[[Monomial], OrderKeyValue] = key
        self.is_graded: bool = is_graded

    def __call__(self, monomial: Monomial) -> OrderKeyValue:
        return self.key(monomial)

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MonomialOrder) and other.name == self.name

    def __hash__(self) -> int:
        return hash(('MonomialOrder', self.name))


def _lex_key(m: Monomial) -> OrderKeyValue:
    return m


def _grlex_key(m: Monomial) -> OrderKeyValue:
    return (sum(m),) + m


def _grevlex_key(m: Monomial) -> OrderKeyValue:
    return (sum(m),) + tuple(-e for e in reversed(m))


lex = MonomialOrder('lex', _lex_key, False)
grlex = MonomialOrder('grlex', _grlex_key, True)
grevlex = MonomialOrder('grevlex', _grevlex_key, True)

_named_orders: dict[str, MonomialOrder] = {'lex': lex, 'grlex': grlex, 'grevlex': grevlex}


def as_order(order: Union[str, MonomialOrder]) -> MonomialOrder:
    """The order of a name (``'lex'``, ``'grlex'``, ``'grevlex'``), or the
    order itself.

    >>> from sympy_extras.polys.orderings import as_order, grevlex
    >>> as_order('grevlex') is grevlex, as_order(grevlex) is grevlex
    (True, True)
    """
    if isinstance(order, MonomialOrder):
        return order
    found = _named_orders.get(order)
    if found is None:
        raise ValueError("unknown monomial order %r" % (order,))
    return found


def _memoised(key: Callable[[Monomial], OrderKeyValue]) -> Callable[[Monomial], OrderKeyValue]:
    """``key`` with its values remembered (the Gröbner basis algorithms
    compare the same monomials many times)."""
    cache: dict[Monomial, OrderKeyValue] = {}

    def memoised(monomial: Monomial) -> OrderKeyValue:
        found = cache.get(monomial)
        if found is None:
            found = key(monomial)
            cache[monomial] = found
        return found
    return memoised


def _probably_graded(key: Callable[[Monomial], OrderKeyValue], nvars: int) -> bool:
    """Whether the square of every variable is above every variable."""
    variables = [tuple(1 if k == i else 0 for k in range(nvars)) for i in range(nvars)]
    squares = [tuple(2 if k == i else 0 for k in range(nvars)) for i in range(nvars)]
    return all(key(s) > key(v) for s in squares for v in variables)


class WeightOrder(MonomialOrder):
    """A vector of non-negative integer weights refined by another order
    (``tail``, ``lex`` by default): the weighted degrees are compared
    first, then ``tail`` breaks the ties. These are the orders along a
    Gröbner walk; the name is that of
    :class:`sympy_extras.polys.orderings_sympy.WeightOrder`.

    >>> from sympy_extras.polys.orderings import WeightOrder, lex
    >>> W = WeightOrder((1, 3))
    >>> W, W((2, 0)), W((2, 0)) < W((0, 1)), W == WeightOrder((1, 3), lex)
    (WeightOrder((1, 3), lex), (2, 2, 0), True, True)
    """

    def __init__(self, weights: Sequence[int], tail: MonomialOrder = lex) -> None:
        self.weights: tuple[int, ...] = tuple(weights)
        if any(w < 0 for w in self.weights):
            raise ValueError("the weights must be non-negative")
        self.tail: MonomialOrder = tail
        weighted = self.weights
        tail_key = tail.key

        def key(m: Monomial) -> OrderKeyValue:
            return (sum(w*e for w, e in zip(weighted, m)),) + tail_key(m)
        graded = bool(weighted) and len(set(weighted)) == 1 and weighted[0] > 0
        super().__init__('WeightOrder(%s, %s)' % (self.weights, tail), _memoised(key), graded)


class BlockOrder(MonomialOrder):
    """A block (product) order: consecutive blocks of variables, each
    compared with its own order, the first block first; ``blocks`` are
    pairs ``(order, size)``. With the variables to eliminate in the first
    block it is an elimination order (:func:`elimination_order`). The name
    is that of :class:`sympy_extras.polys.orderings_sympy.BlockOrder`.

    >>> from sympy_extras.polys.orderings import BlockOrder, grevlex
    >>> B = BlockOrder([(grevlex, 1), (grevlex, 2)])
    >>> B, B((0, 1, 1)) < B((1, 0, 0))
    (BlockOrder((grevlex, 1), (grevlex, 2)), True)
    """

    def __init__(self, blocks: Sequence[tuple[MonomialOrder, int]]) -> None:
        self.blocks: tuple[tuple[MonomialOrder, int], ...] = tuple((order, size) for order, size in blocks)
        if any(size <= 0 for _, size in self.blocks):
            raise ValueError("block sizes must be positive")
        slices: list[tuple[Callable[[Monomial], OrderKeyValue], int, int]] = []
        start = 0
        for order, size in self.blocks:
            slices.append((order.key, start, start + size))
            start += size
        nvars = start

        def key(m: Monomial) -> OrderKeyValue:
            result: OrderKeyValue = ()
            for block_key, begin, end in slices:
                result = result + block_key(m[begin:end])
            return result
        memoised = _memoised(key)
        name = 'BlockOrder(%s)' % ', '.join('(%s, %d)' % (o, s) for o, s in self.blocks)
        super().__init__(name, memoised, _probably_graded(memoised, nvars))


def elimination_order(neliminate: int, nvars: int, order: MonomialOrder = grevlex) -> MonomialOrder:
    """The block order eliminating the first ``neliminate`` of ``nvars``
    variables, ``order`` on each block (``order`` itself when nothing or
    everything is eliminated).

    >>> from sympy_extras.polys.orderings import elimination_order
    >>> elimination_order(1, 3), elimination_order(0, 3)
    (BlockOrder((grevlex, 1), (grevlex, 2)), grevlex)
    """
    if neliminate <= 0 or neliminate >= nvars:
        return order
    return BlockOrder([(order, neliminate), (order, nvars - neliminate)])
