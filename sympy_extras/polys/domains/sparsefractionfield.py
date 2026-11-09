"""Fields of rational functions in several variables, ``K(y1, ..., yk)``,
on the sparse polynomials of :mod:`~sympy_extras.polys.domains.
sparsepolynomialring`: elements are pairs ``(numerator, denominator)``
in lowest terms, the leading coefficient of the denominator nonnegative
(SymPy's ``FracField`` of ``sympy.polys.fields``).

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
>>> from sympy_extras.polys.domains.sparsefractionfield import SparseFractionField
>>> x, y = symbols('x y')
>>> F = SparseFractionField(RationalPolynomialRing(['x', 'y']))
>>> F
QQ(x,y)
>>> a = F.from_expr((x**2 - y**2)/(2*x + 2*y))
>>> F.to_expr(a)
x/2 - y/2
>>> F.to_expr(F.add(F.from_expr(1/x), F.from_expr(1/y)))
(x + y)/(x*y)
>>> F.to_expr(F.diff(F.from_expr(x/(x + y)), 0))
y/(x**2 + 2*x*y + y**2)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/fields.py (FracField, FracElement: new,
# the arithmetic, diff, evaluate) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: a typed domain over SparsePolynomialRing (the cancellation is
# the ring's ``cancel``, SymPy's PolyElement.cancel); no mixed operations
# with the ring or the ground domain (convert first).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Generic, TypeVar

from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow

from ..rings import PolyElement
from .domain import Domain, U
from .sparsepolynomialring import SparsePolynomialRing

__all__ = ['SparseFractionField', 'SparseFracElement']

T = TypeVar('T')


class SparseFracElement(Generic[T]):
    """``numer/denom`` in lowest terms (built by
    :meth:`SparseFractionField.new`)."""

    __slots__ = ('numer', 'denom')

    def __init__(self, numer: PolyElement[T], denom: PolyElement[T]) -> None:
        self.numer: PolyElement[T] = numer
        self.denom: PolyElement[T] = denom

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SparseFracElement) and self.numer == other.numer and self.denom == other.denom

    def __hash__(self) -> int:
        return hash((self.numer, self.denom))

    def __repr__(self) -> str:
        return 'SparseFracElement(%s, %s)' % (self.numer, self.denom)


class SparseFractionField(Domain[SparseFracElement[T]], Generic[T]):
    """The field of fractions ``K(y1, ..., yk)`` of the sparse polynomial
    ring ``K[y1, ..., yk]``."""

    is_field = True

    def __init__(self, ring: SparsePolynomialRing[T]) -> None:
        self.ring: SparsePolynomialRing[T] = ring
        self.ground: Domain[T] = ring.ground
        self.names: list[str] = list(ring.names)
        self.name: str = '%s(%s)' % (ring.ground.name, ','.join(ring.names))
        self.characteristic: int = ring.characteristic

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SparseFractionField) and other.ring == self.ring

    def __hash__(self) -> int:
        return hash(('SparseFractionField', self.ring))

    # PORTED FROM SYMPY: sympy/polys/fields.py:FracElement.new
    def new(self, numer: PolyElement[T], denom: PolyElement[T]) -> SparseFracElement[T]:
        """The fraction ``numer/denom`` in lowest terms."""
        if not denom.terms:
            raise ZeroDivisionError('zero denominator in %s' % self.name)
        p, q = self.ring.cancel(numer, denom)
        return SparseFracElement(p, q)

    @property
    def zero(self) -> SparseFracElement[T]:
        return SparseFracElement(self.ring.zero, self.ring.one)

    @property
    def one(self) -> SparseFracElement[T]:
        return SparseFracElement(self.ring.one, self.ring.one)

    @property
    def gens(self) -> list[SparseFracElement[T]]:
        return [self.from_ring(g) for g in self.ring.ring.gens]

    def is_zero(self, a: SparseFracElement[T]) -> bool:
        return not a.numer.terms

    def is_one(self, a: SparseFracElement[T]) -> bool:
        return self.ring.is_one(a.numer) and self.ring.is_one(a.denom)

    def is_negative(self, a: SparseFracElement[T]) -> bool:
        return self.ring.is_negative(a.numer)

    def is_unit(self, a: SparseFracElement[T]) -> bool:
        return bool(a.numer.terms)

    # PORTED FROM SYMPY: sympy/polys/fields.py:FracElement.__add__
    def add(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        if not b.numer.terms:
            return a
        if not a.numer.terms:
            return b
        if a.denom == b.denom:
            return self.new(a.numer + b.numer, a.denom)
        return self.new(a.numer * b.denom + a.denom * b.numer, a.denom * b.denom)

    def sub(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        return self.add(a, self.neg(b))

    def neg(self, a: SparseFracElement[T]) -> SparseFracElement[T]:
        return SparseFracElement(-a.numer, a.denom)

    # PORTED FROM SYMPY: sympy/polys/fields.py:FracElement.__mul__
    def mul(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        if not a.numer.terms or not b.numer.terms:
            return self.zero
        return self.new(a.numer * b.numer, a.denom * b.denom)

    # PORTED FROM SYMPY: sympy/polys/fields.py:FracElement.__truediv__
    def exquo(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        if not b.numer.terms:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return self.new(a.numer * b.denom, a.denom * b.numer)

    def quo(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        return self.exquo(a, b)

    def rem(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        if not b.numer.terms:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return self.zero

    def invert(self, a: SparseFracElement[T]) -> SparseFracElement[T]:
        if not a.numer.terms:
            raise ZeroDivisionError('0 has no inverse')
        return self.new(a.denom, a.numer)

    # PORTED FROM SYMPY: sympy/polys/fields.py:FracElement.__pow__
    def pow(self, a: SparseFracElement[T], n: int) -> SparseFracElement[T]:
        """``a**n``: the powers of a fraction in lowest terms are in lowest
        terms (the sign of the denominator made canonical)."""
        if n >= 0:
            p, q = self.ring._canonical_denominator(a.numer**n, a.denom**n)
            return SparseFracElement(p, q)
        if not a.numer.terms:
            raise ZeroDivisionError('0 has no inverse')
        p, q = self.ring._canonical_denominator(a.denom**-n, a.numer**-n)
        return SparseFracElement(p, q)

    def gcd(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        return self.zero if not a.numer.terms and not b.numer.terms else self.one

    def lcm(self, a: SparseFracElement[T], b: SparseFracElement[T]) -> SparseFracElement[T]:
        return self.zero if not a.numer.terms or not b.numer.terms else self.one

    def gcdex(self, a: SparseFracElement[T], b: SparseFracElement[T]
              ) -> tuple[SparseFracElement[T], SparseFracElement[T], SparseFracElement[T]]:
        if a.numer.terms:
            return self.invert(a), self.zero, self.one
        if b.numer.terms:
            return self.zero, self.invert(b), self.one
        return self.zero, self.zero, self.zero

    def canonical_unit(self, a: SparseFracElement[T]) -> SparseFracElement[T]:
        return self.one if not a.numer.terms else self.invert(a)

    # PORTED FROM SYMPY: sympy/polys/fields.py:FracElement.diff
    def diff(self, a: SparseFracElement[T], i: int) -> SparseFracElement[T]:
        """The derivative in the ``i``-th variable."""
        return self.new(a.numer.diff(i) * a.denom - a.numer * a.denom.diff(i), a.denom**2)

    def numer(self, a: SparseFracElement[T]) -> PolyElement[T]:
        return a.numer

    def denom(self, a: SparseFracElement[T]) -> PolyElement[T]:
        return a.denom

    def from_int(self, n: int) -> SparseFracElement[T]:
        return SparseFracElement(self.ring.from_int(n), self.ring.one)

    def from_ring(self, a: PolyElement[T]) -> SparseFracElement[T]:
        return SparseFracElement(a, self.ring.one)

    def convert(self, a: U, K: Domain[U]) -> SparseFracElement[T]:
        if isinstance(K, SparseFractionField) and isinstance(a, SparseFracElement):
            if K == self:
                return self.new(self.ring.convert(a.numer, K.ring), self.ring.convert(a.denom, K.ring))
            return self.from_expr(K.to_expr(a))
        return self.from_ring(self.ring.convert(a, K))

    def to_expr(self, a: SparseFracElement[T]) -> Expr:
        numer = self.ring.to_expr(a.numer)
        if self.ring.is_one(a.denom):
            return numer
        return mul(numer, pow(self.ring.to_expr(a.denom), Integer(-1)))

    def from_expr(self, e: Expr) -> SparseFracElement[T]:
        """A rational function of the variables (its numerator and
        denominator, polynomials, converted by the ring)."""
        from sympy_extras.polys.polytools import cancel
        numer, denom = cancel(e).as_numer_denom()
        return self.new(self.ring.from_expr(numer), self.ring.from_expr(denom))

    def to_str(self, a: SparseFracElement[T]) -> str:
        return str(self.to_expr(a))

    def size(self, a: SparseFracElement[T]) -> int:
        return self.ring.size(a.numer) + self.ring.size(a.denom)
