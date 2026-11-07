"""Fields of fractions ``K(x)`` of polynomial rings: elements are pairs
``(numerator, denominator)`` of polynomials over ``K``, reduced (coprime,
the denominator in canonical form).

Examples
========

>>> from sympy_extras.polys.domains import ZZ, PolynomialRing, FractionField
>>> F = FractionField(PolynomialRing(ZZ, 'x'))
>>> F
ZZ(x)
>>> a = F.new([1, 0, -1], [1, 1])
>>> a
FracElement([1, -1], [1])
>>> F.to_sympy(F.invert(F.new([2, 0], [1, 1])))
(x + 1)/(2*x)
>>> F.add(F.new([1], [1, 0]), F.new([1], [1, 1]))
FracElement([2, 1], [1, 1, 0])
"""
from __future__ import annotations

from typing import Generic, TypeVar

from sympy.core.expr import Expr

from sympy_extras._typing import as_expr

from .. import densearith, densebasic, euclidtools
from .domain import Domain, U
from .polynomialring import PolynomialRing

__all__ = ['FractionField', 'FracElement']

T = TypeVar('T')


class FracElement(Generic[T]):
    """A reduced fraction of two dense polynomials over ``T``."""

    __slots__ = ('numer', 'denom')

    def __init__(self, numer: list[T], denom: list[T]) -> None:
        self.numer: list[T] = numer
        self.denom: list[T] = denom

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FracElement) and other.numer == self.numer and other.denom == self.denom

    def __hash__(self) -> int:
        return hash((tuple(self.numer), tuple(self.denom)))

    def __repr__(self) -> str:
        return 'FracElement(%r, %r)' % (self.numer, self.denom)


class FractionField(Domain[FracElement[T]], Generic[T]):
    """The field of fractions ``K(x)`` of the polynomial ring ``K[x]``."""

    is_field = True

    def __init__(self, ring: PolynomialRing[T]) -> None:
        self.ring: PolynomialRing[T] = ring
        self.ground: Domain[T] = ring.ground
        self.name: str = '%s(%s)' % (ring.ground.name, ring.variable)
        self.is_ordered: bool = ring.is_ordered
        self.characteristic: int = ring.characteristic

    def new(self, numer: list[T], denom: list[T]) -> FracElement[T]:
        """The reduced fraction ``numer/denom``."""
        if not denom:
            raise ZeroDivisionError('zero denominator in %s' % self.name)
        if not numer:
            return FracElement([], self.ring.one)
        K = self.ground
        g = euclidtools.dup_gcd(numer, denom, K)
        if not densebasic.dup_is_one(g, K):
            numer = densearith.dup_exquo(numer, g, K)
            denom = densearith.dup_exquo(denom, g, K)
        u = self.ring.canonical_unit(denom)
        if not self.ring.is_one(u):
            numer = self.ring.mul(numer, u)
            denom = self.ring.mul(denom, u)
        if K.is_field:
            lc = denom[0]
            numer = densearith.dup_exquo_ground(numer, lc, K)
            denom = densearith.dup_exquo_ground(denom, lc, K)
        return FracElement(numer, denom)

    @property
    def zero(self) -> FracElement[T]:
        return FracElement([], self.ring.one)

    @property
    def one(self) -> FracElement[T]:
        return FracElement(self.ring.one, self.ring.one)

    def is_zero(self, a: FracElement[T]) -> bool:
        return not a.numer

    def is_one(self, a: FracElement[T]) -> bool:
        return self.ring.is_one(a.numer) and self.ring.is_one(a.denom)

    def is_negative(self, a: FracElement[T]) -> bool:
        return self.ring.is_negative(a.numer)

    def is_unit(self, a: FracElement[T]) -> bool:
        return bool(a.numer)

    def add(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        R = self.ring
        if a.denom == b.denom:
            return self.new(R.add(a.numer, b.numer), a.denom)
        return self.new(R.add(R.mul(a.numer, b.denom), R.mul(b.numer, a.denom)), R.mul(a.denom, b.denom))

    def sub(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        return self.add(a, self.neg(b))

    def neg(self, a: FracElement[T]) -> FracElement[T]:
        return FracElement(self.ring.neg(a.numer), a.denom)

    def mul(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        R = self.ring
        return self.new(R.mul(a.numer, b.numer), R.mul(a.denom, b.denom))

    def exquo(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        if not b.numer:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        R = self.ring
        return self.new(R.mul(a.numer, b.denom), R.mul(a.denom, b.numer))

    def quo(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        return self.exquo(a, b)

    def rem(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        if not b.numer:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return self.zero

    def invert(self, a: FracElement[T]) -> FracElement[T]:
        if not a.numer:
            raise ZeroDivisionError('0 has no inverse')
        return self.new(a.denom, a.numer)

    def gcd(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        return self.zero if not a.numer and not b.numer else self.one

    def lcm(self, a: FracElement[T], b: FracElement[T]) -> FracElement[T]:
        return self.zero if not a.numer or not b.numer else self.one

    def gcdex(self, a: FracElement[T], b: FracElement[T]) -> tuple[FracElement[T], FracElement[T], FracElement[T]]:
        if a.numer:
            return self.invert(a), self.zero, self.one
        if b.numer:
            return self.zero, self.invert(b), self.one
        return self.zero, self.zero, self.zero

    def canonical_unit(self, a: FracElement[T]) -> FracElement[T]:
        return self.one if not a.numer else self.invert(a)

    def numer(self, a: FracElement[T]) -> list[T]:
        return a.numer

    def denom(self, a: FracElement[T]) -> list[T]:
        return a.denom

    def from_int(self, n: int) -> FracElement[T]:
        return FracElement(self.ring.from_int(n), self.ring.one)

    def from_ring(self, a: list[T]) -> FracElement[T]:
        return FracElement(list(a), self.ring.one)

    def convert(self, a: U, K: Domain[U]) -> FracElement[T]:
        if isinstance(K, FractionField) and isinstance(a, FracElement):
            if K == self:
                return a
            return self.from_sympy(K.to_sympy(a))
        return self.from_ring(self.ring.convert(a, K))

    def to_sympy(self, a: FracElement[T]) -> Expr:
        return as_expr(self.ring.to_sympy(a.numer)/self.ring.to_sympy(a.denom))

    def from_sympy(self, e: Expr) -> FracElement[T]:
        from sympy.polys.polytools import cancel
        numer, denom = as_expr(cancel(e)).as_numer_denom()
        return self.new(self.ring.from_sympy(as_expr(numer)), self.ring.from_sympy(as_expr(denom)))

    def to_str(self, a: FracElement[T]) -> str:
        return str(self.to_sympy(a))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FractionField) and other.ring == self.ring

    def __hash__(self) -> int:
        return hash(('FractionField', self.ring))
