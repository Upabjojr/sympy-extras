"""The base class of the coefficient domains.

A domain is a ring (or field) whose elements are plain Python values of
one type: ``int`` for the integers and the finite fields, ``Fraction`` for
the rationals, a list of coefficients for a polynomial ring, a pair of
those for a fraction field. The domain object carries the arithmetic as
methods, so that the polynomial algorithms are written once for a type
variable ``T`` and a ``Domain[T]``, exactly as a generic function over a
ring trait in a statically typed language. Nothing is coerced silently: a
domain only accepts its own elements, and conversions between domains are
explicit (:meth:`Domain.convert`).

References
==========

.. [1] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Chapter 2 (the algebra of polynomials and
       the domain hierarchy).
"""
from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from sympy.core.expr import Expr

T = TypeVar('T')
U = TypeVar('U')


class Domain(Generic[T]):
    """A ring whose elements have the Python type ``T``.

    Subclasses define the constants, the arithmetic and the conversions;
    ``is_field`` and ``characteristic`` say which algorithms apply.
    """

    #: a short name for printing (``ZZ``, ``QQ``, ``GF(7)``, ``ZZ[x]``)
    name: str = 'Domain'
    #: whether every nonzero element is invertible
    is_field: bool = False
    #: whether the domain is ordered (``is_negative``, ``abs`` are defined)
    is_ordered: bool = False
    #: the characteristic (``0`` or a prime)
    characteristic: int = 0

    # -- constants and predicates

    @property
    def zero(self) -> T:
        raise NotImplementedError

    @property
    def one(self) -> T:
        raise NotImplementedError

    def is_zero(self, a: T) -> bool:
        return a == self.zero

    def is_one(self, a: T) -> bool:
        return a == self.one

    def is_negative(self, a: T) -> bool:
        """Whether ``a < 0`` (ordered domains)."""
        raise NotImplementedError('%s is not ordered' % self.name)

    def is_unit(self, a: T) -> bool:
        """Whether ``a`` has an inverse in the domain."""
        if self.is_field:
            return not self.is_zero(a)
        raise NotImplementedError

    # -- arithmetic

    def add(self, a: T, b: T) -> T:
        raise NotImplementedError

    def sub(self, a: T, b: T) -> T:
        raise NotImplementedError

    def neg(self, a: T) -> T:
        raise NotImplementedError

    def mul(self, a: T, b: T) -> T:
        raise NotImplementedError

    def pow(self, a: T, n: int) -> T:
        """``a**n`` for ``n >= 0`` by repeated squaring (``n < 0`` in a
        field)."""
        if n < 0:
            return self.pow(self.invert(a), -n)
        result = self.one
        base = a
        while n:
            if n & 1:
                result = self.mul(result, base)
            n >>= 1
            if n:
                base = self.mul(base, base)
        return result

    def exquo(self, a: T, b: T) -> T:
        """The exact quotient ``a/b`` (``b`` divides ``a``); raises
        ``ZeroDivisionError`` for ``b = 0`` and ``ValueError`` when the
        division is not exact."""
        raise NotImplementedError

    def quo(self, a: T, b: T) -> T:
        """The quotient of the division with remainder (``a`` itself
        divided by ``b`` in a field)."""
        raise NotImplementedError

    def rem(self, a: T, b: T) -> T:
        """The remainder of the division (``0`` in a field)."""
        raise NotImplementedError

    def divmod(self, a: T, b: T) -> tuple[T, T]:
        return self.quo(a, b), self.rem(a, b)

    def invert(self, a: T) -> T:
        """The inverse of a unit."""
        raise NotImplementedError

    def gcd(self, a: T, b: T) -> T:
        """A greatest common divisor, normalised (nonnegative in an ordered
        domain, ``1`` for nonzero elements of a field)."""
        raise NotImplementedError

    def lcm(self, a: T, b: T) -> T:
        if self.is_zero(a) or self.is_zero(b):
            return self.zero
        return self.exquo(self.mul(a, b), self.gcd(a, b))

    def gcdex(self, a: T, b: T) -> tuple[T, T, T]:
        """``(s, t, g)`` with ``s a + t b = g = gcd(a, b)`` (Euclidean
        domains)."""
        raise NotImplementedError

    def abs(self, a: T) -> T:
        """The absolute value (ordered domains)."""
        return self.neg(a) if self.is_negative(a) else a

    def sign(self, a: T) -> int:
        """``-1``, ``0`` or ``1`` (ordered domains)."""
        if self.is_zero(a):
            return 0
        return -1 if self.is_negative(a) else 1

    def canonical_unit(self, a: T) -> T:
        """The unit ``u`` such that ``u a`` is in canonical form (positive
        in an ordered domain, monic in a polynomial ring over a field, ``1``
        otherwise); ``1`` for ``a = 0``."""
        return self.one

    # -- dense polynomial kernels: a domain whose elements are machine
    # values may run the inner loops of the dense arithmetic on them
    # directly (``has_dense_kernels``); the generic code of
    # :mod:`sympy_extras.polys.densearith` is used otherwise

    #: whether ``dense_mul`` and ``dense_div`` are provided
    has_dense_kernels: bool = False

    def dense_mul(self, f: list[T], g: list[T]) -> list[T]:
        """The product of two dense polynomials (highest degree first,
        no leading zeros), for domains with ``has_dense_kernels``."""
        raise NotImplementedError

    def dense_div(self, f: list[T], g: list[T]) -> tuple[list[T], list[T]]:
        """The quotient and remainder of dense polynomials, for domains
        with ``has_dense_kernels`` (a field, or the division is exact)."""
        raise NotImplementedError

    # -- conversions

    def from_int(self, n: int) -> T:
        """The image of an integer."""
        raise NotImplementedError

    def convert(self, a: U, K: Domain[U]) -> T:
        """The element ``a`` of the domain ``K`` as an element of this
        domain; ``ValueError`` when there is no exact conversion."""
        raise NotImplementedError

    def to_sympy(self, a: T) -> Expr:
        """The element as a SymPy expression."""
        raise NotImplementedError

    def from_sympy(self, e: Expr) -> T:
        """A SymPy expression which is an element of the domain, as an
        element; ``ValueError`` otherwise."""
        raise NotImplementedError

    def to_str(self, a: T) -> str:
        return str(a)

    # -- enumeration (finite domains)

    def elements(self) -> Iterator[T]:
        raise NotImplementedError('%s is not finite' % self.name)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Domain) and type(self) is type(other) and self.name == other.name

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.name))
