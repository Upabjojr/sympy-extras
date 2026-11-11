"""Tests of the native monomial orders (:mod:`sympy_extras.polys.orderings`)
against their definitions; the SymPy face (``orderings_sympy``) is tested
with SymPy's ``groebner`` in ``test_ideals.py``."""
from __future__ import annotations

from functools import cmp_to_key
from itertools import product
from typing import Callable

from sympy_extras._testing import untyped
from sympy_extras.polys import rings
from sympy_extras.polys.orderings import (BlockOrder, Monomial, MonomialOrder, WeightOrder, as_order,
                                          elimination_order, grevlex, grlex, lex)

MONOMIALS = [m for m in product(range(4), repeat=3) if sum(m) <= 4]


def _sorted(cmp: Callable[[Monomial, Monomial], int]) -> list[Monomial]:
    return sorted(MONOMIALS, key=cmp_to_key(cmp))


def _sign(a: int) -> int:
    return (a > 0) - (a < 0)


def _lex_cmp(a: Monomial, b: Monomial) -> int:
    for i, j in zip(a, b):
        if i != j:
            return _sign(i - j)
    return 0


def _grevlex_cmp(a: Monomial, b: Monomial) -> int:
    # total degree, then the smaller exponent of the last differing variable wins
    if sum(a) != sum(b):
        return _sign(sum(a) - sum(b))
    for i, j in zip(reversed(a), reversed(b)):
        if i != j:
            return _sign(j - i)
    return 0


def test_orders_against_their_definitions() -> None:
    assert sorted(MONOMIALS, key=lex) == _sorted(_lex_cmp)
    assert sorted(MONOMIALS, key=grevlex) == _sorted(_grevlex_cmp)
    assert sorted(MONOMIALS, key=grlex) == _sorted(
        lambda a, b: _sign(sum(a) - sum(b)) or _lex_cmp(a, b))
    W = WeightOrder((2, 0, 1), grevlex)
    assert sorted(MONOMIALS, key=W) == _sorted(
        lambda a, b: _sign(2*a[0] + a[2] - 2*b[0] - b[2]) or _grevlex_cmp(a, b))
    B = BlockOrder([(lex, 1), (grevlex, 2)])
    assert sorted(MONOMIALS, key=B) == _sorted(
        lambda a, b: _sign(a[0] - b[0]) or _grevlex_cmp(a[1:], b[1:]))
    # the first block dominates: an elimination order for x
    E = elimination_order(1, 3)
    assert all(E(m) > E(n) for m in MONOMIALS for n in MONOMIALS if m[0] > 0 and n[0] == 0)


def test_names_errors_and_rings() -> None:
    assert str(WeightOrder((1, 3))) == 'WeightOrder((1, 3), lex)'
    assert str(BlockOrder([(grevlex, 1), (grevlex, 2)])) == 'BlockOrder((grevlex, 1), (grevlex, 2))'
    assert elimination_order(0, 3) is grevlex and elimination_order(3, 3, lex) is lex
    assert as_order('grlex') is grlex and isinstance(as_order(lex), MonomialOrder)
    assert WeightOrder((1, 1, 1), grevlex).is_graded and not WeightOrder((1, 0)).is_graded
    assert BlockOrder([(grevlex, 2)]).is_graded and not lex.is_graded
    for bad in (lambda: WeightOrder((1, -1)), lambda: BlockOrder([(lex, 0)]), lambda: untyped(as_order)('ilex')):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('ValueError expected')
    # the rings use these orders
    assert rings.lex is lex and rings.WeightOrder is WeightOrder and rings.elimination_order is elimination_order
