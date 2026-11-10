r"""Multiquadratic extensions $K(\sqrt{d_1}, \dots, \sqrt{d_k})$ of a field
$K$ of characteristic zero, for integers $d_i$ which are independent
modulo squares (distinct primes, and $-1$ for the imaginary unit).

An element is a combination of the $2^k$ products of the generators: a
dictionary from the subset (a bit mask) to its nonzero coefficient in
$K$. The product of two basis elements is the basis element of the
symmetric difference of the subsets times the product of the $d_i$ in
their intersection. The inverse comes from the conjugations: with
$\sigma$ the automorphism changing the sign of the last generator present
in $a$, the norm $a\, \sigma(a)$ lies in the field without that
generator, and $a^{-1} = \sigma(a)\, (a\, \sigma(a))^{-1}$ [1]_.

The $d_i$ being distinct primes (and $-1$), the $2^k$ products are
linearly independent over $K \supseteq \mathbb{Q}$ whenever $K$ is a
field of rational functions over $\mathbb{Q}$ (Besicovitch), so the
structure is a field and a nonzero dictionary is a nonzero number.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.domains.multiquadratic import MultiquadraticField
>>> F = MultiquadraticField(QQ, [2, 3])
>>> a = F.add(F.generator(0), F.generator(1))            # sqrt(2) + sqrt(3)
>>> F.to_str(F.mul(a, a)), F.to_str(F.invert(a))
('5 + 2*sqrt(6)', '-1*sqrt(2) + sqrt(3)')
>>> F.mul(a, F.invert(a)) == F.one
True

References
==========

.. [1] H. Cohen, *A Course in Computational Algebraic Number Theory*,
       Springer, 1993, Section 4.2 (arithmetic in number fields) and
       Section 4.3 (norms and conjugates).
.. [2] A. S. Besicovitch, *On the linear independence of fractional
       powers of integers*, Journal of the London Mathematical Society 15
       (1940), 3-6.
"""
from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from .domain import Domain

__all__ = ['MultiquadraticField']

T = TypeVar('T')


class MultiquadraticField(Domain[dict[int, T]], Generic[T]):
    """``K(sqrt(d_1), ..., sqrt(d_k))`` over the field ``ground``, with
    ``radicands`` distinct primes, possibly with ``-1``."""

    is_field = True
    is_ordered = False

    def __init__(self, ground: Domain[T], radicands: Sequence[int]) -> None:
        if not ground.is_field or ground.characteristic != 0:
            raise ValueError('%s is not a field of characteristic zero' % ground.name)
        if len(set(radicands)) != len(radicands) or any(d in (0, 1) for d in radicands):
            raise ValueError('the radicands %s are not independent' % (list(radicands),))
        self.ground: Domain[T] = ground
        self.radicands: list[int] = list(radicands)
        self.name = '%s(%s)' % (ground.name, ', '.join('sqrt(%d)' % d for d in radicands))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultiquadraticField) and other.ground == self.ground and other.radicands == self.radicands

    def __hash__(self) -> int:
        return hash(('MultiquadraticField', self.ground, tuple(self.radicands)))

    @property
    def zero(self) -> dict[int, T]:
        return {}

    @property
    def one(self) -> dict[int, T]:
        return {0: self.ground.one}

    def generator(self, i: int) -> dict[int, T]:
        """``sqrt(radicands[i])``."""
        return {1 << i: self.ground.one}

    def from_ground(self, c: T) -> dict[int, T]:
        return {} if self.ground.is_zero(c) else {0: c}

    def from_int(self, n: int) -> dict[int, T]:
        return self.from_ground(self.ground.from_int(n))

    def is_zero(self, a: dict[int, T]) -> bool:
        return not a

    def is_one(self, a: dict[int, T]) -> bool:
        return len(a) == 1 and 0 in a and self.ground.is_one(a[0])

    def is_unit(self, a: dict[int, T]) -> bool:
        return bool(a)

    def add(self, a: dict[int, T], b: dict[int, T]) -> dict[int, T]:
        K = self.ground
        result = dict(a)
        for mask, c in b.items():
            known = result.get(mask)
            if known is None:
                result[mask] = c
                continue
            total = K.add(known, c)
            if K.is_zero(total):
                del result[mask]
            else:
                result[mask] = total
        return result

    def neg(self, a: dict[int, T]) -> dict[int, T]:
        return {mask: self.ground.neg(c) for mask, c in a.items()}

    def sub(self, a: dict[int, T], b: dict[int, T]) -> dict[int, T]:
        return self.add(a, self.neg(b))

    def _square_part(self, common: int) -> int:
        product = 1
        i = 0
        while common:
            if common & 1:
                product *= self.radicands[i]
            common >>= 1
            i += 1
        return product

    def mul(self, a: dict[int, T], b: dict[int, T]) -> dict[int, T]:
        K = self.ground
        result: dict[int, T] = {}
        for m1, c1 in a.items():
            for m2, c2 in b.items():
                value = K.mul(c1, c2)
                common = m1 & m2
                if common:
                    value = K.mul(value, K.from_int(self._square_part(common)))
                mask = m1 ^ m2
                known = result.get(mask)
                if known is not None:
                    value = K.add(known, value)
                if K.is_zero(value):
                    result.pop(mask, None)
                else:
                    result[mask] = value
        return result

    def conjugate(self, a: dict[int, T], i: int) -> dict[int, T]:
        """The image under the automorphism ``sqrt(d_i) -> -sqrt(d_i)``."""
        bit = 1 << i
        return {mask: (self.ground.neg(c) if mask & bit else c) for mask, c in a.items()}

    def invert(self, a: dict[int, T]) -> dict[int, T]:
        if not a:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        highest = max(a).bit_length() - 1
        if highest < 0:
            return {0: self.ground.invert(a[0])}
        conjugate = self.conjugate(a, highest)
        return self.mul(conjugate, self.invert(self.mul(a, conjugate)))

    def exquo(self, a: dict[int, T], b: dict[int, T]) -> dict[int, T]:
        if not a:
            if not b:
                raise ZeroDivisionError('division by zero in %s' % self.name)
            return {}
        return self.mul(a, self.invert(b))

    def quo(self, a: dict[int, T], b: dict[int, T]) -> dict[int, T]:
        return self.exquo(a, b)

    def rem(self, a: dict[int, T], b: dict[int, T]) -> dict[int, T]:
        if not b:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return {}

    def gcd(self, a: dict[int, T], b: dict[int, T]) -> dict[int, T]:
        return {} if not a and not b else self.one

    def size(self, a: dict[int, T]) -> int:
        return sum(self.ground.size(c) for c in a.values()) + len(a)

    def to_str(self, a: dict[int, T]) -> str:
        if not a:
            return '0'
        parts: list[str] = []
        for mask in sorted(a):
            coefficient = self.ground.to_str(a[mask])
            if not mask:
                parts.append(coefficient)
                continue
            square = self._square_part(mask)
            radical = 'sqrt(%d)' % square
            parts.append(radical if self.ground.is_one(a[mask]) else '%s*%s' % (coefficient, radical))
        return ' + '.join(parts)
