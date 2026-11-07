"""The ring of integers, with Python's ``int`` as the element type.

Examples
========

>>> from sympy_extras.polys.domains import ZZ
>>> ZZ.gcd(12, -18), ZZ.gcdex(12, 18), ZZ.quo(-7, 2), ZZ.rem(-7, 2)
(6, (-1, 1, 6), -3, -1)
>>> ZZ.exquo(12, 4)
3
>>> ZZ.from_sympy(ZZ.to_sympy(5))
5
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Optional

from sympy.core.expr import Expr
from sympy.core.numbers import Integer

from sympy_extras._typing import as_expr

from .domain import Domain, U

__all__ = ['IntegerRing', 'ZZ']


class IntegerRing(Domain[int]):
    """The integers ``ZZ``."""

    name = 'ZZ'
    is_field = False
    is_ordered = True
    characteristic = 0

    @property
    def zero(self) -> int:
        return 0

    @property
    def one(self) -> int:
        return 1

    def is_zero(self, a: int) -> bool:
        return a == 0

    def is_one(self, a: int) -> bool:
        return a == 1

    def is_negative(self, a: int) -> bool:
        return a < 0

    def is_unit(self, a: int) -> bool:
        return a == 1 or a == -1

    def add(self, a: int, b: int) -> int:
        return a + b

    def sub(self, a: int, b: int) -> int:
        return a - b

    def neg(self, a: int) -> int:
        return -a

    def mul(self, a: int, b: int) -> int:
        return a*b

    def pow(self, a: int, n: int) -> int:
        if n < 0:
            raise ValueError('negative exponent in ZZ')
        return a**n

    def exquo(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError('division by zero in ZZ')
        q, r = divmod(a, b)
        if r:
            raise ValueError('%d does not divide %d' % (b, a))
        return q

    def quo(self, a: int, b: int) -> int:
        """The quotient of the truncated division (towards zero, so that
        ``a = q b + r`` with ``|r| < |b|`` and ``r`` of the sign of
        ``a``)."""
        if b == 0:
            raise ZeroDivisionError('division by zero in ZZ')
        q = abs(a)//abs(b)
        return q if (a < 0) == (b < 0) else -q

    def rem(self, a: int, b: int) -> int:
        return a - self.quo(a, b)*b

    def invert(self, a: int) -> int:
        if a == 1 or a == -1:
            return a
        raise ValueError('%d is not a unit of ZZ' % a)

    def gcd(self, a: int, b: int) -> int:
        return gcd(a, b)

    def lcm(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return abs(a*b)//gcd(a, b)

    def gcdex(self, a: int, b: int) -> tuple[int, int, int]:
        """``(s, t, g)`` with ``s a + t b = g = gcd(a, b) >= 0`` (the
        extended Euclidean algorithm)."""
        s0, s1 = 1, 0
        t0, t1 = 0, 1
        r0, r1 = a, b
        while r1:
            q = r0//r1
            r0, r1 = r1, r0 - q*r1
            s0, s1 = s1, s0 - q*s1
            t0, t1 = t1, t0 - q*t1
        if r0 < 0:
            return -s0, -t0, -r0
        return s0, t0, r0

    def abs(self, a: int) -> int:
        return abs(a)

    def canonical_unit(self, a: int) -> int:
        return -1 if a < 0 else 1

    has_dense_kernels = True

    def dense_mul(self, f: list[int], g: list[int]) -> list[int]:
        if not f or not g:
            return []
        result = [0]*(len(f) + len(g) - 1)
        for i, a in enumerate(f):
            if a:
                for j, b in enumerate(g):
                    result[i + j] += a*b
        return _strip(result)

    def dense_div(self, f: list[int], g: list[int]) -> tuple[list[int], list[int]]:
        """Division with the quotient coefficients obtained by exact
        division by the leading coefficient of ``g`` (``ValueError``
        when one is not exact)."""
        if not g:
            raise ZeroDivisionError('polynomial division by zero')
        df, dg = len(f) - 1, len(g) - 1
        if df < dg:
            return [], list(f)
        lc = g[0]
        quotient = [0]*(df - dg + 1)
        remainder = list(f)
        for i in range(df - dg + 1):
            c = remainder[i]
            if c:
                q, r = divmod(c, lc)
                if r:
                    raise ValueError('%d does not divide %d' % (lc, c))
                quotient[i] = q
                for j in range(1, dg + 1):
                    remainder[i + j] -= q*g[j]
        return _strip(quotient), _strip(remainder[df - dg + 1:])

    def factor_dense(self, f: list[int]) -> tuple[int, list[tuple[list[int], int]]]:
        from ..factortools import dup_zz_factor
        return dup_zz_factor(f, self)

    def isolate_real_roots_dense(self, f: list[int], eps: Optional[Fraction] = None
                                 ) -> list[tuple[tuple[Fraction, Fraction], int]]:
        from ..rootisolation import dup_isolate_real_roots
        return dup_isolate_real_roots([Fraction(c) for c in f], eps)

    def count_real_roots_dense(self, f: list[int], inf: Optional[Fraction], sup: Optional[Fraction]) -> int:
        from ..rootisolation import dup_count_real_roots
        return dup_count_real_roots([Fraction(c) for c in f], inf, sup)

    def from_int(self, n: int) -> int:
        return n

    def convert(self, a: U, K: Domain[U]) -> int:
        if isinstance(K, IntegerRing) and isinstance(a, int):
            return a
        e = K.to_sympy(a)
        return self.from_sympy(e)

    def to_sympy(self, a: int) -> Expr:
        return Integer(a)

    def from_sympy(self, e: Expr) -> int:
        e = as_expr(e)
        if isinstance(e, Integer):
            return int(e)
        raise ValueError('%s is not an integer' % (e,))



def _strip(f: list[int]) -> list[int]:
    i = 0
    while i < len(f) and f[i] == 0:
        i += 1
    return f[i:]


ZZ = IntegerRing()
