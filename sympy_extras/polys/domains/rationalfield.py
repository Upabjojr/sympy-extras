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
>>> QQ.from_sympy(QQ.to_sympy(Fraction(3, 7))) == Fraction(3, 7)
True
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from sympy.core.expr import Expr
from sympy.core.numbers import Rational

from sympy_extras._typing import as_expr

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
        return self.from_sympy(K.to_sympy(a))

    def to_sympy(self, a: Fraction) -> Expr:
        return Rational(a.numerator, a.denominator)

    def from_sympy(self, e: Expr) -> Fraction:
        e = as_expr(e)
        if isinstance(e, Rational):
            return Fraction(int(e.p), int(e.q))
        raise ValueError('%s is not a rational number' % (e,))


QQ = RationalField()
