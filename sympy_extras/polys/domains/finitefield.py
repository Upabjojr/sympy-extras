"""The prime fields ``GF(p)``, with the integers ``0, ..., p - 1`` as the
elements (plain ``int``, reduced by every operation).

Examples
========

>>> from sympy_extras.polys.domains import GF
>>> F = GF(7)
>>> F.add(5, 4), F.mul(3, 5), F.invert(3), F.pow(3, 100)
(2, 1, 5, 4)
>>> F.from_int(-1), F.to_sympy(6)
(6, 6)
>>> list(GF(3).elements())
[0, 1, 2]
"""
from __future__ import annotations

from typing import Iterator

from sympy.core.expr import Expr
from sympy.core.numbers import Integer

from sympy_extras._typing import as_expr

from .domain import Domain, U
from .integerring import IntegerRing

__all__ = ['FiniteField', 'GF']


class FiniteField(Domain[int]):
    """The field of integers modulo the prime ``p``."""

    is_field = True
    is_ordered = False

    def __init__(self, p: int) -> None:
        if p < 2:
            raise ValueError('the modulus must be a prime, got %d' % p)
        self.p: int = p
        self.name: str = 'GF(%d)' % p
        self.characteristic: int = p

    @property
    def zero(self) -> int:
        return 0

    @property
    def one(self) -> int:
        return 1 % self.p

    def is_zero(self, a: int) -> bool:
        return a == 0

    def is_one(self, a: int) -> bool:
        return a == 1 % self.p

    def is_unit(self, a: int) -> bool:
        return a != 0

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p

    def neg(self, a: int) -> int:
        return (-a) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a*b) % self.p

    def pow(self, a: int, n: int) -> int:
        return pow(a, n, self.p)

    def exquo(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return (a*pow(b, -1, self.p)) % self.p

    def quo(self, a: int, b: int) -> int:
        return self.exquo(a, b)

    def rem(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return 0

    def invert(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError('0 has no inverse')
        return pow(a, -1, self.p)

    def gcd(self, a: int, b: int) -> int:
        return 0 if a == 0 and b == 0 else self.one

    def lcm(self, a: int, b: int) -> int:
        return 0 if a == 0 or b == 0 else self.one

    def gcdex(self, a: int, b: int) -> tuple[int, int, int]:
        if a != 0:
            return self.invert(a), 0, self.one
        if b != 0:
            return 0, self.invert(b), self.one
        return 0, 0, 0

    def canonical_unit(self, a: int) -> int:
        return self.one if a == 0 else self.invert(a)

    has_dense_kernels = True

    def dense_mul(self, f: list[int], g: list[int]) -> list[int]:
        if not f or not g:
            return []
        p = self.p
        result = [0]*(len(f) + len(g) - 1)
        for i, a in enumerate(f):
            if a:
                for j, b in enumerate(g):
                    result[i + j] += a*b
        return _strip([c % p for c in result])

    def dense_div(self, f: list[int], g: list[int]) -> tuple[list[int], list[int]]:
        if not g:
            raise ZeroDivisionError('polynomial division by zero')
        df, dg = len(f) - 1, len(g) - 1
        if df < dg:
            return [], list(f)
        p = self.p
        inverse = pow(g[0], -1, p)
        quotient = [0]*(df - dg + 1)
        remainder = list(f)
        for i in range(df - dg + 1):
            c = remainder[i] % p
            if c:
                q = (c*inverse) % p
                quotient[i] = q
                for j in range(1, dg + 1):
                    remainder[i + j] -= q*g[j]
        return _strip(quotient), _strip([c % p for c in remainder[df - dg + 1:]])

    def factor_dense(self, f: list[int]) -> tuple[int, list[tuple[list[int], int]]]:
        from ..galoistools import gf_factor
        return gf_factor(f, self)

    def from_int(self, n: int) -> int:
        return n % self.p

    def convert(self, a: U, K: Domain[U]) -> int:
        if isinstance(K, FiniteField) and isinstance(a, int):
            if K.p == self.p:
                return a
            raise ValueError('no conversion from %s to %s' % (K.name, self.name))
        if isinstance(K, IntegerRing) and isinstance(a, int):
            return a % self.p
        return self.from_sympy(K.to_sympy(a))

    def to_sympy(self, a: int) -> Expr:
        return Integer(a)

    def from_sympy(self, e: Expr) -> int:
        e = as_expr(e)
        if isinstance(e, Integer):
            return int(e) % self.p
        raise ValueError('%s is not an integer' % (e,))

    def elements(self) -> Iterator[int]:
        return iter(range(self.p))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FiniteField) and other.p == self.p

    def __hash__(self) -> int:
        return hash(('FiniteField', self.p))


def _strip(f: list[int]) -> list[int]:
    i = 0
    while i < len(f) and f[i] == 0:
        i += 1
    return f[i:]


def GF(p: int) -> FiniteField:
    """The prime field with ``p`` elements."""
    return FiniteField(p)
