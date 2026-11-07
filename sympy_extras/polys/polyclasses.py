"""A typed polynomial class over the dense representation: a list of
coefficients in a :class:`~sympy_extras.polys.domains.domain.Domain`, a
generator symbol, and the algorithms of the dense modules as methods.

A polynomial in several variables is a ``DensePoly`` over a
:class:`~sympy_extras.polys.domains.polynomialring.PolynomialRing`
domain (``K[y][x]``), whose coefficients are themselves dense
polynomials; the conversions to and from SymPy expressions go through
the domain, so ``to_sympy`` and ``from_sympy`` handle every variable.

Examples
========

>>> from sympy.abc import x
>>> from sympy_extras.polys.domains import ZZ, QQ
>>> from sympy_extras.polys.polyclasses import DensePoly
>>> f = DensePoly.from_sympy(x**4 - 1, x, ZZ)
>>> f
DensePoly([1, 0, 0, 0, -1], ZZ, x)
>>> f.factor_list()
(1, [(DensePoly([1, -1], ZZ, x), 1), (DensePoly([1, 1], ZZ, x), 1), (DensePoly([1, 0, 1], ZZ, x), 1)])
>>> (f*f).sqf_list()
(1, [(DensePoly([1, 0, 0, 0, -1], ZZ, x), 2)])
>>> f.gcd(DensePoly.from_sympy(x**2 - 1, x, ZZ)).to_sympy()
x**2 - 1
>>> f.count_real_roots(), f.real_root_intervals()
(2, [((Fraction(-2, 1), Fraction(0, 1)), 1), ((Fraction(0, 1), Fraction(2, 1)), 1)])
>>> DensePoly.from_sympy(x**2 - 2, x, QQ).resultant(DensePoly.from_sympy(x - 3, x, QQ)) == 7
True
"""
from __future__ import annotations

from fractions import Fraction
from typing import Generic, Iterator, Optional, Sequence, TypeVar

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from . import densearith, densebasic, densetools, euclidtools, sqfreetools
from .domains.domain import Domain
from .domains.polynomialring import PolynomialRing

__all__ = ['DensePoly']

T = TypeVar('T')
U = TypeVar('U')


class DensePoly(Generic[T]):
    """A univariate polynomial ``rep`` (coefficients highest degree first,
    no leading zeros) over ``domain`` in the variable ``gen``."""

    __slots__ = ('rep', 'domain', 'gen')

    def __init__(self, rep: Sequence[T], domain: Domain[T], gen: Symbol) -> None:
        self.rep: list[T] = densebasic.dup_strip(list(rep), domain)
        self.domain: Domain[T] = domain
        self.gen: Symbol = gen

    # -- construction and conversion

    @classmethod
    def from_sympy(cls, expr: Expr, gen: Symbol, domain: Domain[U]) -> DensePoly[U]:
        """The polynomial of an expression in ``gen`` whose coefficients
        belong to ``domain`` (``ValueError`` otherwise)."""
        ring: PolynomialRing[U] = PolynomialRing(domain, gen.name)
        return DensePoly(ring.from_sympy(expr), domain, gen)

    @classmethod
    def from_list(cls, coefficients: Sequence[int], gen: Symbol, domain: Domain[U]) -> DensePoly[U]:
        """The polynomial with the given integer coefficients (highest
        degree first) over ``domain``."""
        return DensePoly(densebasic.dup_from_list(coefficients, domain), domain, gen)

    def to_sympy(self) -> Expr:
        return PolynomialRing(self.domain, self.gen.name).to_sympy(self.rep)

    def convert(self, domain: Domain[U]) -> DensePoly[U]:
        """The same polynomial over another domain."""
        return DensePoly(densebasic.dup_convert(self.rep, self.domain, domain), domain, self.gen)

    def new(self, rep: list[T]) -> DensePoly[T]:
        return DensePoly(rep, self.domain, self.gen)

    def _same(self, other: DensePoly[T]) -> None:
        if other.domain != self.domain or other.gen != self.gen:
            raise ValueError('polynomials over different domains or in different variables')

    def __repr__(self) -> str:
        return 'DensePoly(%r, %s, %s)' % (self.rep, self.domain, self.gen)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DensePoly) and other.rep == self.rep and other.domain == self.domain and other.gen == self.gen

    def __hash__(self) -> int:
        return hash((tuple(self.rep), self.domain, self.gen))

    def __iter__(self) -> Iterator[T]:
        return iter(self.rep)

    # -- basic data

    @property
    def degree(self) -> int:
        return densebasic.dup_degree(self.rep)

    @property
    def LC(self) -> T:
        return densebasic.dup_LC(self.rep, self.domain)

    @property
    def TC(self) -> T:
        return densebasic.dup_TC(self.rep, self.domain)

    @property
    def is_zero(self) -> bool:
        return not self.rep

    @property
    def is_one(self) -> bool:
        return densebasic.dup_is_one(self.rep, self.domain)

    @property
    def is_monic(self) -> bool:
        return bool(self.rep) and self.domain.is_one(self.rep[0])

    @property
    def is_squarefree(self) -> bool:
        return sqfreetools.dup_sqf_p(self.rep, self.domain)

    def coeff(self, n: int) -> T:
        """The coefficient of ``gen**n``."""
        d = self.degree
        if n < 0 or n > d:
            return self.domain.zero
        return self.rep[d - n]

    def terms(self) -> list[tuple[int, T]]:
        return densebasic.dup_terms(self.rep, self.domain)

    # -- arithmetic

    def __neg__(self) -> DensePoly[T]:
        return self.new(densearith.dup_neg(self.rep, self.domain))

    def __add__(self, other: DensePoly[T]) -> DensePoly[T]:
        self._same(other)
        return self.new(densearith.dup_add(self.rep, other.rep, self.domain))

    def __sub__(self, other: DensePoly[T]) -> DensePoly[T]:
        self._same(other)
        return self.new(densearith.dup_sub(self.rep, other.rep, self.domain))

    def __mul__(self, other: DensePoly[T]) -> DensePoly[T]:
        self._same(other)
        return self.new(densearith.dup_mul(self.rep, other.rep, self.domain))

    def __pow__(self, n: int) -> DensePoly[T]:
        return self.new(densearith.dup_pow(self.rep, n, self.domain))

    def mul_ground(self, c: T) -> DensePoly[T]:
        return self.new(densearith.dup_mul_ground(self.rep, c, self.domain))

    def exquo_ground(self, c: T) -> DensePoly[T]:
        return self.new(densearith.dup_exquo_ground(self.rep, c, self.domain))

    def div(self, other: DensePoly[T]) -> tuple[DensePoly[T], DensePoly[T]]:
        """Quotient and remainder (over a field, or with exact divisions
        of the leading coefficients)."""
        self._same(other)
        q, r = densearith.dup_div(self.rep, other.rep, self.domain)
        return self.new(q), self.new(r)

    def quo(self, other: DensePoly[T]) -> DensePoly[T]:
        return self.div(other)[0]

    def rem(self, other: DensePoly[T]) -> DensePoly[T]:
        return self.div(other)[1]

    def exquo(self, other: DensePoly[T]) -> DensePoly[T]:
        """The exact quotient; ``ValueError`` when ``other`` does not
        divide ``self``."""
        self._same(other)
        return self.new(densearith.dup_exquo(self.rep, other.rep, self.domain))

    def pdiv(self, other: DensePoly[T]) -> tuple[DensePoly[T], DensePoly[T]]:
        self._same(other)
        q, r = densearith.dup_pdiv(self.rep, other.rep, self.domain)
        return self.new(q), self.new(r)

    def prem(self, other: DensePoly[T]) -> DensePoly[T]:
        return self.pdiv(other)[1]

    # -- calculus and transformations

    def diff(self, m: int = 1) -> DensePoly[T]:
        return self.new(densetools.dup_diff(self.rep, m, self.domain))

    def integrate(self, m: int = 1) -> DensePoly[T]:
        return self.new(densetools.dup_integrate(self.rep, m, self.domain))

    def eval(self, a: T) -> T:
        return densetools.dup_eval(self.rep, a, self.domain)

    def shift(self, a: T) -> DensePoly[T]:
        """``f(x + a)``."""
        return self.new(densetools.dup_shift(self.rep, a, self.domain))

    def scale(self, a: T) -> DensePoly[T]:
        """``f(a x)``."""
        return self.new(densetools.dup_scale(self.rep, a, self.domain))

    def compose(self, other: DensePoly[T]) -> DensePoly[T]:
        self._same(other)
        return self.new(densetools.dup_compose(self.rep, other.rep, self.domain))

    def mirror(self) -> DensePoly[T]:
        """``f(-x)``."""
        return self.new(densetools.dup_mirror(self.rep, self.domain))

    def reverse(self) -> DensePoly[T]:
        return self.new(densebasic.dup_reverse(self.rep, self.domain))

    # -- contents and normal forms

    def content(self) -> T:
        return densetools.dup_content(self.rep, self.domain)

    def primitive(self) -> tuple[T, DensePoly[T]]:
        c, part = densetools.dup_primitive(self.rep, self.domain)
        return c, self.new(part)

    def monic(self) -> DensePoly[T]:
        return self.new(densetools.dup_monic(self.rep, self.domain))

    # -- gcd, resultants

    def gcd(self, other: DensePoly[T]) -> DensePoly[T]:
        self._same(other)
        return self.new(euclidtools.dup_gcd(self.rep, other.rep, self.domain))

    def lcm(self, other: DensePoly[T]) -> DensePoly[T]:
        self._same(other)
        return self.new(euclidtools.dup_lcm(self.rep, other.rep, self.domain))

    def gcdex(self, other: DensePoly[T]) -> tuple[DensePoly[T], DensePoly[T], DensePoly[T]]:
        """``(s, t, h)`` with ``s f + t g = h`` the monic gcd (fields)."""
        self._same(other)
        s, t, h = euclidtools.dup_gcdex(self.rep, other.rep, self.domain)
        return self.new(s), self.new(t), self.new(h)

    def invert(self, other: DensePoly[T]) -> DensePoly[T]:
        """The inverse of ``self`` modulo ``other`` (fields)."""
        self._same(other)
        return self.new(euclidtools.dup_invert(self.rep, other.rep, self.domain))

    def resultant(self, other: DensePoly[T]) -> T:
        self._same(other)
        return euclidtools.dup_resultant(self.rep, other.rep, self.domain)

    def discriminant(self) -> T:
        return euclidtools.dup_discriminant(self.rep, self.domain)

    def subresultants(self, other: DensePoly[T]) -> list[DensePoly[T]]:
        self._same(other)
        return [self.new(r) for r in euclidtools.dup_subresultants(self.rep, other.rep, self.domain)]

    # -- factorisation

    def sqf_part(self) -> DensePoly[T]:
        return self.new(sqfreetools.dup_sqf_part(self.rep, self.domain))

    def sqf_list(self) -> tuple[T, list[tuple[DensePoly[T], int]]]:
        c, factors = sqfreetools.dup_sqf_list(self.rep, self.domain)
        return c, [(self.new(g), k) for g, k in factors]

    def factor_list(self) -> tuple[T, list[tuple[DensePoly[T], int]]]:
        """The irreducible factorisation over the domain (``ZZ``, ``QQ``,
        ``GF(p)`` and number fields)."""
        c, factors = self.domain.factor_dense(self.rep)
        return c, [(self.new(g), k) for g, k in factors]

    @property
    def is_irreducible(self) -> bool:
        if self.degree < 1:
            return False
        _, factors = self.factor_list()
        return len(factors) == 1 and factors[0][1] == 1 and factors[0][0].degree == self.degree

    # -- real roots (ZZ and QQ)

    def real_root_intervals(self, eps: Optional[Fraction] = None) -> list[tuple[tuple[Fraction, Fraction], int]]:
        """Isolating intervals of the real roots with their multiplicities
        (refined below the width ``eps`` when given)."""
        return self.domain.isolate_real_roots_dense(self.rep, eps)

    def count_real_roots(self, inf: Optional[Fraction] = None, sup: Optional[Fraction] = None) -> int:
        return self.domain.count_real_roots_dense(self.rep, inf, sup)

