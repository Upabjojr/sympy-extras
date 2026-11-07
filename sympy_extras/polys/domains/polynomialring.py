"""Polynomial rings ``K[x]`` as coefficient domains: a polynomial in
several variables is a polynomial over a polynomial ring (``K[y][x]``),
and the generic dense algorithms run on the tower.

Examples
========

>>> from sympy_extras.polys.domains import ZZ, PolynomialRing
>>> R = PolynomialRing(ZZ, 'y')
>>> R
ZZ[y]
>>> R.mul([1, 1], [1, -1])
[1, 0, -1]
>>> R.gcd([2, 0, -2], [4, 4])
[2, 2]
>>> R.to_sympy([1, 0, -1])
y**2 - 1
>>> S = PolynomialRing(R, 'x')
>>> S
ZZ[y][x]
>>> S.to_sympy([[1], [2, 0], []])
x**2 + 2*x*y
"""
from __future__ import annotations

from typing import Generic, TypeVar

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly

from sympy_extras._typing import as_expr

from .. import densearith, densebasic, densetools, euclidtools
from .domain import Domain, U
from .integerring import IntegerRing

__all__ = ['PolynomialRing']

T = TypeVar('T')


class PolynomialRing(Domain[list[T]], Generic[T]):
    """The ring ``K[x]`` of dense univariate polynomials over the domain
    ``K`` (the ``ground`` domain) in the variable named ``variable``."""

    is_field = False

    def __init__(self, ground: Domain[T], variable: str) -> None:
        self.ground: Domain[T] = ground
        self.variable: str = variable
        self.symbol: Symbol = Symbol(variable)
        self.name: str = '%s[%s]' % (ground.name, variable)
        self.is_ordered: bool = ground.is_ordered
        self.characteristic: int = ground.characteristic

    @property
    def zero(self) -> list[T]:
        return []

    @property
    def one(self) -> list[T]:
        return [self.ground.one]

    def is_zero(self, a: list[T]) -> bool:
        return not a

    def is_one(self, a: list[T]) -> bool:
        return len(a) == 1 and self.ground.is_one(a[0])

    def is_negative(self, a: list[T]) -> bool:
        """Whether the leading coefficient is negative (the sign of a
        polynomial, for the canonical forms)."""
        return bool(a) and self.ground.is_negative(a[0])

    def is_unit(self, a: list[T]) -> bool:
        return len(a) == 1 and self.ground.is_unit(a[0])

    def add(self, a: list[T], b: list[T]) -> list[T]:
        return densearith.dup_add(a, b, self.ground)

    def sub(self, a: list[T], b: list[T]) -> list[T]:
        return densearith.dup_sub(a, b, self.ground)

    def neg(self, a: list[T]) -> list[T]:
        return densearith.dup_neg(a, self.ground)

    def mul(self, a: list[T], b: list[T]) -> list[T]:
        return densearith.dup_mul(a, b, self.ground)

    def pow(self, a: list[T], n: int) -> list[T]:
        if n < 0:
            raise ValueError('negative exponent in %s' % self.name)
        return densearith.dup_pow(a, n, self.ground)

    def exquo(self, a: list[T], b: list[T]) -> list[T]:
        if not b:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        if self.ground.is_field:
            return densearith.dup_exquo(a, b, self.ground)
        if not a:
            return []
        if len(a) < len(b):
            raise ValueError('%s does not divide %s' % (b, a))
        # over a ring: lc(b)**(deg a - deg b + 1) a = q b + r, and the
        # quotient is q divided exactly by that power when r vanishes
        q, r = densearith.dup_pdiv(a, b, self.ground)
        if r:
            raise ValueError('%s does not divide %s' % (b, a))
        scale = self.ground.pow(b[0], len(a) - len(b) + 1)
        return densearith.dup_exquo_ground(q, scale, self.ground)

    def quo(self, a: list[T], b: list[T]) -> list[T]:
        if self.ground.is_field:
            return densearith.dup_quo(a, b, self.ground)
        return densearith.dup_pquo(a, b, self.ground)

    def rem(self, a: list[T], b: list[T]) -> list[T]:
        if self.ground.is_field:
            return densearith.dup_rem(a, b, self.ground)
        return densearith.dup_prem(a, b, self.ground)

    def invert(self, a: list[T]) -> list[T]:
        if self.is_unit(a):
            return [self.ground.invert(a[0])]
        raise ValueError('%s is not a unit of %s' % (a, self.name))

    def gcd(self, a: list[T], b: list[T]) -> list[T]:
        return euclidtools.dup_gcd(a, b, self.ground)

    def lcm(self, a: list[T], b: list[T]) -> list[T]:
        return euclidtools.dup_lcm(a, b, self.ground)

    def gcdex(self, a: list[T], b: list[T]) -> tuple[list[T], list[T], list[T]]:
        return euclidtools.dup_gcdex(a, b, self.ground)

    def abs(self, a: list[T]) -> list[T]:
        return self.neg(a) if self.is_negative(a) else list(a)

    def canonical_unit(self, a: list[T]) -> list[T]:
        if not a:
            return self.one
        return [self.ground.canonical_unit(a[0])]

    def from_int(self, n: int) -> list[T]:
        c = self.ground.from_int(n)
        return [] if self.ground.is_zero(c) else [c]

    def convert(self, a: U, K: Domain[U]) -> list[T]:
        if isinstance(K, PolynomialRing) and K.name == self.name and isinstance(a, list):
            return [self.ground.convert(c, K.ground) for c in a]
        if isinstance(K, PolynomialRing) and K.variable == self.variable and isinstance(a, list):
            return densebasic.dup_strip([self.ground.convert(c, K.ground) for c in a], self.ground)
        if isinstance(K, IntegerRing) and isinstance(a, int):
            return self.from_int(a)
        # a ground element
        c = self.ground.convert(a, K)
        return [] if self.ground.is_zero(c) else [c]

    def to_sympy(self, a: list[T]) -> Expr:
        x = self.symbol
        n = len(a) - 1
        total: Expr = as_expr(0)
        for i, c in enumerate(a):
            total = as_expr(total + self.ground.to_sympy(c)*x**(n - i))
        return total

    def from_sympy(self, e: Expr) -> list[T]:
        p = Poly(e, self.symbol)
        coefficients = [self.ground.from_sympy(as_expr(c)) for c in p.all_coeffs()]
        return densebasic.dup_strip(coefficients, self.ground)

    def to_str(self, a: list[T]) -> str:
        return str(self.to_sympy(a))

    def diff(self, a: list[T]) -> list[T]:
        return densetools.dup_diff(a, 1, self.ground)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolynomialRing) and other.ground == self.ground and other.variable == self.variable

    def __hash__(self) -> int:
        return hash(('PolynomialRing', self.ground, self.variable))
