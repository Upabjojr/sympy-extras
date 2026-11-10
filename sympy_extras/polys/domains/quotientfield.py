r"""Simple algebraic extensions of any field: $K[\lambda]/(f)$ for a monic
irreducible $f$ over the field $K$.

The elements are dense polynomials over $K$ of degree below that of $f$
(lists, highest degree first), reduced after every product; inverses come
from the extended Euclidean algorithm against $f$. This is
:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField` with
the rationals replaced by a type variable, so that the ground field may
be a field of rational functions in parameters: linear algebra over it
computes with *all* the roots of $f$ at once (the eigenvectors of a
matrix for the eigenvalues which are the roots of one irreducible factor
of its characteristic polynomial).

When $f$ is not irreducible the ring has zero divisors, and
:meth:`QuotientField.invert` raises ``ValueError`` on meeting one.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.domains.quotientfield import QuotientField
>>> F = QuotientField(QQ, [Fraction(1), Fraction(0), Fraction(-2)])        # QQ(sqrt(2))
>>> a = F.add(F.generator, F.one)
>>> F.mul(a, a) == [2, 3] and F.mul(a, F.invert(a)) == F.one
True
>>> F.name
QQ[t]/(t**2 - 2)

References
==========

.. [1] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Section 8.8 (algebraic extensions).
"""
from __future__ import annotations

from typing import Generic, TypeVar

from .. import densearith, densebasic, euclidtools
from .domain import Domain

__all__ = ['QuotientField']

T = TypeVar('T')


class QuotientField(Domain[list[T]], Generic[T]):
    """``K[t]/(modulus)`` with ``modulus`` monic of positive degree and
    irreducible over the field ``K``."""

    is_field = True
    is_ordered = False

    def __init__(self, ground: Domain[T], modulus: list[T]) -> None:
        if not ground.is_field:
            raise ValueError('%s is not a field' % ground.name)
        if densebasic.dup_degree(modulus) < 1 or not ground.is_one(modulus[0]):
            raise ValueError('the modulus must be monic of positive degree')
        self.ground: Domain[T] = ground
        self.modulus: list[T] = list(modulus)
        self.degree: int = densebasic.dup_degree(modulus)
        self.characteristic = ground.characteristic
        terms: list[str] = []
        for k, c in enumerate(modulus):
            if ground.is_zero(c):
                continue
            n = self.degree - k
            power = 't**%d' % n if n > 1 else ('t' if n == 1 else '')
            text = ground.to_str(c)
            if power and ground.is_one(c):
                text = power
            elif power:
                text = '%s*%s' % (text, power)
            terms.append(text)
        self.name = '%s[t]/(%s)' % (ground.name, ' + '.join(terms).replace('+ -', '- '))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuotientField) and other.ground == self.ground and other.modulus == self.modulus

    def __hash__(self) -> int:
        return hash(('QuotientField', self.ground, self.degree))

    @property
    def zero(self) -> list[T]:
        return []

    @property
    def one(self) -> list[T]:
        return [self.ground.one]

    @property
    def generator(self) -> list[T]:
        """The class of ``t``."""
        return self.reduce([self.ground.one, self.ground.zero])

    def reduce(self, a: list[T]) -> list[T]:
        """The representative of degree below that of the modulus."""
        if len(a) <= self.degree:
            return densebasic.dup_strip(a, self.ground)
        return densearith.dup_rem(a, self.modulus, self.ground)

    def is_zero(self, a: list[T]) -> bool:
        return not a

    def is_one(self, a: list[T]) -> bool:
        return len(a) == 1 and self.ground.is_one(a[0])

    def is_unit(self, a: list[T]) -> bool:
        return bool(a)

    def add(self, a: list[T], b: list[T]) -> list[T]:
        return densearith.dup_add(a, b, self.ground)

    def sub(self, a: list[T], b: list[T]) -> list[T]:
        return densearith.dup_sub(a, b, self.ground)

    def neg(self, a: list[T]) -> list[T]:
        return densearith.dup_neg(a, self.ground)

    def mul(self, a: list[T], b: list[T]) -> list[T]:
        if not a or not b:
            return []
        return self.reduce(densearith.dup_mul(a, b, self.ground))

    def invert(self, a: list[T]) -> list[T]:
        if not a:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        if len(a) == 1:
            return [self.ground.invert(a[0])]
        return euclidtools.dup_invert(a, self.modulus, self.ground)

    def exquo(self, a: list[T], b: list[T]) -> list[T]:
        if not a:
            if not b:
                raise ZeroDivisionError('division by zero in %s' % self.name)
            return []
        return self.mul(a, self.invert(b))

    def quo(self, a: list[T], b: list[T]) -> list[T]:
        return self.exquo(a, b)

    def rem(self, a: list[T], b: list[T]) -> list[T]:
        if not b:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return []

    def gcd(self, a: list[T], b: list[T]) -> list[T]:
        return [] if not a and not b else self.one

    def from_int(self, n: int) -> list[T]:
        return densebasic.dup_strip([self.ground.from_int(n)], self.ground)

    def from_ground(self, c: T) -> list[T]:
        return densebasic.dup_strip([c], self.ground)

    def size(self, a: list[T]) -> int:
        return sum(self.ground.size(c) for c in a) + len(a)

    def to_str(self, a: list[T]) -> str:
        return '[%s]' % ', '.join(self.ground.to_str(c) for c in a)
