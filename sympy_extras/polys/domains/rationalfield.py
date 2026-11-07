"""The field of rational numbers, with ``fractions.Fraction`` as the
element type.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.domains import QQ
>>> QQ.add(Fraction(1, 2), Fraction(1, 3)), QQ.invert(Fraction(-2, 3))
(Fraction(5, 6), Fraction(-3, 2))
>>> QQ.numer(Fraction(-6, 4)), QQ.denom(Fraction(-6, 4))
(-3, 2)
>>> QQ.from_expr(QQ.to_expr(Fraction(3, 7))) == Fraction(3, 7)
True
"""
from __future__ import annotations

import heapq
from fractions import Fraction
from math import gcd
from typing import Optional

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Rational, number_from_fraction

from .domain import Domain, U
from .integerring import IntegerRing

__all__ = ['RationalField', 'QQ']


class RationalField(Domain[Fraction]):
    """The rational numbers ``QQ``."""

    name = 'QQ'
    is_field = True
    is_ordered = True
    characteristic = 0

    @property
    def zero(self) -> Fraction:
        return Fraction(0)

    @property
    def one(self) -> Fraction:
        return Fraction(1)

    def is_zero(self, a: Fraction) -> bool:
        return a == 0

    def is_one(self, a: Fraction) -> bool:
        return a == 1

    def is_negative(self, a: Fraction) -> bool:
        return a < 0

    def is_unit(self, a: Fraction) -> bool:
        return a != 0

    def add(self, a: Fraction, b: Fraction) -> Fraction:
        return a + b

    def sub(self, a: Fraction, b: Fraction) -> Fraction:
        return a - b

    def neg(self, a: Fraction) -> Fraction:
        return -a

    def mul(self, a: Fraction, b: Fraction) -> Fraction:
        return a*b

    def pow(self, a: Fraction, n: int) -> Fraction:
        return a**n

    def exquo(self, a: Fraction, b: Fraction) -> Fraction:
        if b == 0:
            raise ZeroDivisionError('division by zero in QQ')
        return a/b

    def quo(self, a: Fraction, b: Fraction) -> Fraction:
        return self.exquo(a, b)

    def rem(self, a: Fraction, b: Fraction) -> Fraction:
        if b == 0:
            raise ZeroDivisionError('division by zero in QQ')
        return Fraction(0)

    def invert(self, a: Fraction) -> Fraction:
        if a == 0:
            raise ZeroDivisionError('0 has no inverse')
        return 1/a

    def gcd(self, a: Fraction, b: Fraction) -> Fraction:
        return Fraction(0) if a == 0 and b == 0 else Fraction(1)

    def lcm(self, a: Fraction, b: Fraction) -> Fraction:
        return Fraction(0) if a == 0 or b == 0 else Fraction(1)

    def gcdex(self, a: Fraction, b: Fraction) -> tuple[Fraction, Fraction, Fraction]:
        if a != 0:
            return 1/a, Fraction(0), Fraction(1)
        if b != 0:
            return Fraction(0), 1/b, Fraction(1)
        return Fraction(0), Fraction(0), Fraction(0)

    def abs(self, a: Fraction) -> Fraction:
        return abs(a)

    def canonical_unit(self, a: Fraction) -> Fraction:
        return Fraction(1) if a == 0 else 1/a

    def size(self, a: Fraction) -> int:
        return a.numerator.bit_length() + a.denominator.bit_length()

    def sparse_reduce(self, rows: list[dict[int, Fraction]], pivots: dict[int, dict[int, Fraction]]
                      ) -> list[dict[int, Fraction]]:
        """Fraction-free: rows and pivots as primitive integer rows, each
        step ``r = (L/g) r - (r_c/g) p`` with ``g = gcd(L, r_c)``, the
        content removed once per row (no rational normalisation inside
        the elimination)."""
        integer_pivots = {c: _primitive_integer_row(p) for c, p in pivots.items()}
        result: list[dict[int, Fraction]] = []
        for row in rows:
            reduced = _integer_reduce(_primitive_integer_row(row), integer_pivots)
            if reduced:
                result.append(_monic_rational_row(reduced))
        return result

    def sparse_echelon(self, rows: list[dict[int, Fraction]]) -> list[dict[int, Fraction]]:
        """The reduced row echelon form, fraction-free over the integers
        and made monic at the end."""
        pivots: dict[int, dict[int, int]] = {}
        for row in rows:
            reduced = _integer_reduce(_primitive_integer_row(row), pivots)
            if reduced:
                pivots[min(reduced)] = reduced
        leads = sorted(pivots, reverse=True)
        for c in leads:
            pivot = pivots[c]
            lead_value = pivot[c]
            for other in leads:
                if other >= c or c not in pivots[other]:
                    continue
                pivots[other] = _eliminate(pivots[other], c, pivot, lead_value)
        return [_monic_rational_row(pivots[c]) for c in sorted(pivots)]

    def numer(self, a: Fraction) -> int:
        return a.numerator

    def denom(self, a: Fraction) -> int:
        return a.denominator

    def factor_dense(self, f: list[Fraction]) -> tuple[Fraction, list[tuple[list[Fraction], int]]]:
        from ..factortools import dup_qq_factor
        return dup_qq_factor(f, self)

    def isolate_real_roots_dense(self, f: list[Fraction], eps: Optional[Fraction] = None
                                 ) -> list[tuple[tuple[Fraction, Fraction], int]]:
        from ..rootisolation import dup_isolate_real_roots
        return dup_isolate_real_roots(f, eps)

    def count_real_roots_dense(self, f: list[Fraction], inf: Optional[Fraction], sup: Optional[Fraction]) -> int:
        from ..rootisolation import dup_count_real_roots
        return dup_count_real_roots(f, inf, sup)

    def from_int(self, n: int) -> Fraction:
        return Fraction(n)

    def convert(self, a: U, K: Domain[U]) -> Fraction:
        if isinstance(K, RationalField) and isinstance(a, Fraction):
            return a
        if isinstance(K, IntegerRing) and isinstance(a, int):
            return Fraction(a)
        return self.from_expr(K.to_expr(a))

    def to_expr(self, a: Fraction) -> Expr:
        return number_from_fraction(a)

    def from_expr(self, e: Expr) -> Fraction:
        if isinstance(e, Rational):
            return Fraction(e.p, e.q)
        raise ValueError('%s is not a rational number' % (e,))



def _primitive_integer_row(row: dict[int, Fraction]) -> dict[int, int]:
    """The row times the lcm of its denominators, divided by the gcd of
    the resulting integers, with a positive leading entry."""
    common = 1
    for value in row.values():
        d = value.denominator
        common = common*d//gcd(common, d)
    integers = {c: (v.numerator*(common//v.denominator)) for c, v in row.items()}
    return _primitive(integers)


def _primitive(row: dict[int, int]) -> dict[int, int]:
    content = 0
    for value in row.values():
        content = gcd(content, value)
        if content == 1:
            break
    if row and row[min(row)] < 0:
        content = -content
    if content in (0, 1):
        return row
    return {c: v//content for c, v in row.items()}


def _eliminate(row: dict[int, int], c: int, pivot: dict[int, int], lead_value: int) -> dict[int, int]:
    """``row`` with column ``c`` cleared by ``pivot`` (whose entry there is
    ``lead_value``), fraction-free, then made primitive."""
    factor = row[c]
    g = gcd(lead_value, factor)
    scale, multiple = lead_value//g, factor//g
    result = {column: value*scale for column, value in row.items()} if scale != 1 else dict(row)
    for column, value in pivot.items():
        updated = result.get(column, 0) - multiple*value
        if updated:
            result[column] = updated
        elif column in result:
            del result[column]
    return _primitive(result)


def _integer_reduce(row: dict[int, int], pivots: dict[int, dict[int, int]]) -> dict[int, int]:
    """The integer row with every pivot column eliminated, in increasing
    column order."""
    heap = list(row)
    heapq.heapify(heap)
    while heap:
        c = heapq.heappop(heap)
        if c not in row:
            continue
        pivot = pivots.get(c)
        if pivot is None:
            continue
        before = set(row)
        row = _eliminate(row, c, pivot, pivot[c])
        for column in row:
            if column not in before:
                heapq.heappush(heap, column)
    return row


def _monic_rational_row(row: dict[int, int]) -> dict[int, Fraction]:
    lead = row[min(row)]
    return {c: Fraction(v, lead) for c, v in row.items()}


QQ = RationalField()
