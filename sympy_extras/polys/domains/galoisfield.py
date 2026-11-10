r"""The finite fields $GF(p^n)$ as coefficient domains.

$GF(p^n) = GF(p)[a]/(m(a))$ for an irreducible $m$ of degree $n$ over the
prime field: the elements are polynomials in the generator $a$ of degree
below $n$ (dense lists of integers in $[0, p)$, highest degree first, no
leading zeros), reduced after every product; inverses come from the
extended Euclidean algorithm against $m$. Without a modulus, the first
monic irreducible polynomial of degree $n$ is used (its lower
coefficients read as a number in base $p$).

Examples
========

>>> from sympy_extras.polys.domains.galoisfield import GaloisField
>>> K = GaloisField(2, 3)
>>> K, K.modulus, K.order
(GF(2**3), [1, 0, 1, 1], 8)
>>> a = K.generator
>>> K.mul(a, a), K.pow(a, 3), K.pow(a, 7)
([1, 0, 0], [1, 1], [1])
>>> K.mul(K.invert([1, 1, 0]), [1, 1, 0])
[1]
>>> len(list(K.elements()))
8
>>> F = GaloisField(3, 2, [1, 0, 1])          # GF(9) = GF(3)[i]
>>> F.pow(F.generator, 2), F.to_str([2, 1])
([2], '2*a + 1')

References
==========

.. [1] R. Lidl, H. Niederreiter, *Finite Fields*, 2nd ed., Cambridge
       University Press, 1997, Chapters 2 and 3.
.. [2] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
       Cambridge University Press, 2013, Chapter 14.
"""
from __future__ import annotations

from random import Random
from typing import Iterator, Optional

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol

from .. import densearith, densebasic, euclidtools
from ..galoistools import gf_irreducible_p
from .domain import Domain, U
from .finitefield import FiniteField
from .integerring import IntegerRing

__all__ = ['GaloisField']

def _first_irreducible(n: int, F: FiniteField) -> list[int]:
    """The first monic irreducible polynomial of degree ``n`` over the
    prime field, the lower coefficients read as a number in base ``p``."""
    if n == 1:
        return [1, 0]
    index = 1
    while True:
        digits: list[int] = []
        value = index
        for _ in range(n):
            digits.append(value % F.p)
            value //= F.p
        candidate = [1] + digits[::-1]
        if gf_irreducible_p(candidate, F):
            return candidate
        index += 1


class GaloisField(Domain[list[int]]):
    """The field with ``p**n`` elements, as ``GF(p)[a]/(modulus)``.

    Parameters
    ==========

    p : the characteristic, a prime.
    n : the degree over the prime field.
    modulus : a monic irreducible polynomial of degree ``n`` over
        ``GF(p)`` (highest degree first); the first one when omitted.
    generator_expr : the expression standing for the generator in the
        conversions (the symbol ``a`` by default).
    """

    is_field = True
    is_ordered = False

    def __init__(self, p: int, n: int, modulus: Optional[list[int]] = None,
                 generator_expr: Optional[Expr] = None) -> None:
        if n < 1:
            raise ValueError('the degree must be positive')
        self.prime_field: FiniteField = FiniteField(p)
        self.p: int = p
        self.characteristic = p
        self.degree: int = n
        self.order: int = p**n
        if modulus is None:
            modulus = _first_irreducible(n, self.prime_field)
        modulus = [c % p for c in modulus]
        if len(modulus) != n + 1 or modulus[0] != 1:
            raise ValueError('the modulus must be monic of degree %d' % n)
        if n > 1 and not gf_irreducible_p(modulus, self.prime_field):
            raise ValueError('the modulus is not irreducible over GF(%d)' % p)
        self.modulus: list[int] = modulus
        self.generator_expr: Expr = Symbol('a') if generator_expr is None else generator_expr
        self.name: str = 'GF(%d**%d)' % (p, n) if n > 1 else 'GF(%d)' % p

    @property
    def zero(self) -> list[int]:
        return []

    @property
    def one(self) -> list[int]:
        return [1]

    @property
    def generator(self) -> list[int]:
        """The class of ``a``."""
        return self._reduce([1, 0])

    def _reduce(self, a: list[int]) -> list[int]:
        a = densebasic.dup_strip([c % self.p for c in a], self.prime_field)
        if len(a) <= self.degree:
            return a
        return densearith.dup_rem(a, self.modulus, self.prime_field)

    def is_zero(self, a: list[int]) -> bool:
        return not a

    def is_one(self, a: list[int]) -> bool:
        return a == [1]

    def is_unit(self, a: list[int]) -> bool:
        return bool(a)

    def add(self, a: list[int], b: list[int]) -> list[int]:
        return densearith.dup_add(a, b, self.prime_field)

    def sub(self, a: list[int], b: list[int]) -> list[int]:
        return densearith.dup_sub(a, b, self.prime_field)

    def neg(self, a: list[int]) -> list[int]:
        return densearith.dup_neg(a, self.prime_field)

    def mul(self, a: list[int], b: list[int]) -> list[int]:
        return self._reduce(densearith.dup_mul(a, b, self.prime_field))

    def exquo(self, a: list[int], b: list[int]) -> list[int]:
        return self.mul(a, self.invert(b))

    def quo(self, a: list[int], b: list[int]) -> list[int]:
        return self.exquo(a, b)

    def rem(self, a: list[int], b: list[int]) -> list[int]:
        if not b:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return []

    def invert(self, a: list[int]) -> list[int]:
        if not a:
            raise ZeroDivisionError('0 has no inverse')
        return euclidtools.dup_invert(a, self.modulus, self.prime_field)

    def gcd(self, a: list[int], b: list[int]) -> list[int]:
        return [] if not a and not b else self.one

    def lcm(self, a: list[int], b: list[int]) -> list[int]:
        return [] if not a or not b else self.one

    def gcdex(self, a: list[int], b: list[int]) -> tuple[list[int], list[int], list[int]]:
        if a:
            return self.invert(a), [], self.one
        if b:
            return [], self.invert(b), self.one
        return [], [], []

    def canonical_unit(self, a: list[int]) -> list[int]:
        return self.one if not a else self.invert(a)

    def frobenius(self, a: list[int]) -> list[int]:
        """``a**p``."""
        return self.pow(a, self.p)

    def pth_root(self, a: list[int]) -> list[int]:
        """The element whose ``p``-th power is ``a``: ``a**(q/p)``."""
        return self.pow(a, self.order//self.p)

    def trace(self, a: list[int]) -> int:
        """The trace to the prime field: ``a + a**p + ... + a**(p**(n-1))``."""
        total = self.zero
        power = a
        for _ in range(self.degree):
            total = self.add(total, power)
            power = self.frobenius(power)
        return total[0] if total else 0

    def random(self, rng: Random) -> list[int]:
        """A uniformly random element."""
        return densebasic.dup_strip([rng.randrange(self.p) for _ in range(self.degree)], self.prime_field)

    def from_int(self, n: int) -> list[int]:
        return [n % self.p] if n % self.p else []

    def convert(self, a: U, K: Domain[U]) -> list[int]:
        if isinstance(K, GaloisField) and isinstance(a, list) and K.modulus == self.modulus and K.p == self.p:
            return [c for c in a if isinstance(c, int)]
        if isinstance(K, (FiniteField, IntegerRing)) and isinstance(a, int):
            return self.from_int(a)
        raise ValueError('no conversion from %s to %s' % (K, self.name))

    def to_expr(self, a: list[int]) -> Expr:
        n = len(a) - 1
        parts: list[Expr] = []
        for i, c in enumerate(a):
            if c:
                k = n - i
                power = self.generator_expr if k == 1 else pow(self.generator_expr, Integer(k))
                parts.append(Integer(c) if k == 0 else (power if c == 1 else mul(Integer(c), power)))
        return add(*parts) if parts else Integer(0)

    def from_expr(self, e: Expr) -> list[int]:
        raise ValueError('%s is not converted to %s' % (e, self.name))

    def to_str(self, a: list[int]) -> str:
        return str(self.to_expr(a))

    def elements(self) -> Iterator[list[int]]:
        for index in range(self.order):
            digits: list[int] = []
            while index:
                digits.append(index % self.p)
                index //= self.p
            yield digits[::-1]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaloisField) and other.p == self.p and other.modulus == self.modulus

    def __hash__(self) -> int:
        return hash(('GaloisField', self.p, tuple(self.modulus)))
