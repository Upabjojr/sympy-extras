r"""Algebraic number fields $\mathbb{Q}(\alpha)$ as coefficient domains:
elements are polynomials in $\alpha$ with rational coefficients of degree
below that of the minimal polynomial (dense lists, highest degree first),
reduced after every product; inverses come from the extended Euclidean
algorithm against the minimal polynomial.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.core import sqrt, Integer
>>> from sympy_extras.polys.domains.algebraicfield import AlgebraicField
>>> K = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], sqrt(Integer(2)))
>>> K
QQ<sqrt(2)>
>>> a = K.from_expr(1 + sqrt(Integer(2)))
>>> a, K.mul(a, a), K.invert(a)
([Fraction(1, 1), Fraction(1, 1)], [Fraction(2, 1), Fraction(3, 1)], [Fraction(1, 1), Fraction(-1, 1)])
>>> K.to_expr(K.invert(a))
-1 + sqrt(2)
>>> K.norm(a), K.minimal_polynomial(a)
(Fraction(-1, 1), [Fraction(1, 1), Fraction(-2, 1), Fraction(-1, 1)])

References
==========

.. [1] H. Cohen, *A Course in Computational Algebraic Number Theory*,
       Springer, 1993, Section 4.2 (representation of algebraic numbers
       and their arithmetic).
"""
from __future__ import annotations

from fractions import Fraction

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow, pow

from .. import densearith, densebasic, densetools, euclidtools, sqfreetools
from .domain import Domain, U
from .integerring import IntegerRing
from .rationalfield import QQ, RationalField

__all__ = ['AlgebraicField']


class AlgebraicField(Domain[list[Fraction]]):
    """The field ``QQ(alpha)`` with ``alpha`` a root of the monic
    irreducible ``minpoly`` (coefficients highest degree first) and
    ``alpha_expr`` the expression of the root in the typed core, used by
    the conversions."""

    is_field = True
    is_ordered = False
    characteristic = 0

    def __init__(self, minpoly: list[Fraction], alpha_expr: Expr) -> None:
        if densebasic.dup_degree(minpoly) < 1 or minpoly[0] != 1:
            raise ValueError('the minimal polynomial must be monic of positive degree')
        self.minpoly: list[Fraction] = list(minpoly)
        self.degree: int = densebasic.dup_degree(minpoly)
        self.alpha_expr: Expr = alpha_expr
        self.name: str = 'QQ<%s>' % (alpha_expr,)

    @property
    def zero(self) -> list[Fraction]:
        return []

    @property
    def one(self) -> list[Fraction]:
        return [Fraction(1)]

    @property
    def alpha(self) -> list[Fraction]:
        """The generator itself."""
        return [Fraction(1), Fraction(0)] if self.degree > 1 else self._reduce([Fraction(1), Fraction(0)])

    def _reduce(self, a: list[Fraction]) -> list[Fraction]:
        """A list of coefficients as an element: leading zeros removed,
        reduced modulo the minimal polynomial."""
        a = densebasic.dup_strip(a, QQ)
        if densebasic.dup_degree(a) < self.degree:
            return a
        return densearith.dup_rem(a, self.minpoly, QQ)

    def is_zero(self, a: list[Fraction]) -> bool:
        return not a

    def is_one(self, a: list[Fraction]) -> bool:
        return a == [Fraction(1)]

    def is_unit(self, a: list[Fraction]) -> bool:
        return bool(a)

    def add(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return densearith.dup_add(a, b, QQ)

    def sub(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return densearith.dup_sub(a, b, QQ)

    def neg(self, a: list[Fraction]) -> list[Fraction]:
        return densearith.dup_neg(a, QQ)

    def mul(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return self._reduce(densearith.dup_mul(a, b, QQ))

    def exquo(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return self.mul(a, self.invert(b))

    def quo(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return self.exquo(a, b)

    def rem(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        if not b:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return []

    def invert(self, a: list[Fraction]) -> list[Fraction]:
        if not a:
            raise ZeroDivisionError('0 has no inverse')
        return euclidtools.dup_invert(a, self.minpoly, QQ)

    def gcd(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return [] if not a and not b else self.one

    def lcm(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return [] if not a or not b else self.one

    def gcdex(self, a: list[Fraction], b: list[Fraction]) -> tuple[list[Fraction], list[Fraction], list[Fraction]]:
        if a:
            return self.invert(a), [], self.one
        if b:
            return [], self.invert(b), self.one
        return [], [], []

    def canonical_unit(self, a: list[Fraction]) -> list[Fraction]:
        return self.one if not a else self.invert(a)

    def norm(self, a: list[Fraction]) -> Fraction:
        """The norm ``N(a)``, the product of the conjugates: the resultant
        of the minimal polynomial and ``a`` as a polynomial in
        ``alpha``."""
        if not a:
            return Fraction(0)
        return euclidtools.dup_resultant(self.minpoly, a, QQ)

    def characteristic_polynomial(self, a: list[Fraction]) -> list[Fraction]:
        """The characteristic polynomial of ``a`` (in a new variable
        ``t``): the resultant in ``alpha`` of the minimal polynomial and
        ``t - a(alpha)``, a polynomial in ``t`` of degree ``[K : QQ]``."""
        from .polynomialring import PolynomialRing
        R = PolynomialRing(QQ, 't')
        # t - a(alpha) as a polynomial in alpha with coefficients in QQ[t]
        constant: list[Fraction] = [Fraction(1), Fraction(0)]  # t
        coefficients: list[list[Fraction]] = [[-c] if c else [] for c in a]
        if coefficients:
            coefficients[-1] = densearith.dup_add(coefficients[-1], constant, QQ)
        else:
            coefficients = [constant]
        m_t: list[list[Fraction]] = [[c] if c else [] for c in self.minpoly]
        result = euclidtools.dup_resultant(m_t, densebasic.dup_strip(coefficients, R), R)
        return densetools.dup_monic(result, QQ)

    def minimal_polynomial(self, a: list[Fraction]) -> list[Fraction]:
        """The minimal polynomial of ``a`` over ``QQ``: the square-free part
        of its characteristic polynomial."""
        return sqfreetools.dup_sqf_part(self.characteristic_polynomial(a), QQ)

    def factor_dense(self, f: list[list[Fraction]]) -> tuple[list[Fraction], list[tuple[list[list[Fraction]], int]]]:
        from ..factortools import dup_ext_factor
        return dup_ext_factor(f, self)

    def from_int(self, n: int) -> list[Fraction]:
        return [Fraction(n)] if n else []

    def from_rational(self, q: Fraction) -> list[Fraction]:
        return [q] if q else []

    def convert(self, a: U, K: Domain[U]) -> list[Fraction]:
        if isinstance(K, AlgebraicField) and isinstance(a, list):
            if K.minpoly == self.minpoly:
                return [c for c in a if isinstance(c, Fraction)]
            return self.from_expr(K.to_expr(a))
        if isinstance(K, RationalField) and isinstance(a, Fraction):
            return self.from_rational(a)
        if isinstance(K, IntegerRing) and isinstance(a, int):
            return self.from_int(a)
        return self.from_expr(K.to_expr(a))

    def to_expr(self, a: list[Fraction]) -> Expr:
        """The element as a core expression in ``alpha_expr``."""
        n = len(a) - 1
        parts: list[Expr] = []
        for i, c in enumerate(a):
            if c:
                k = n - i
                power = self.alpha_expr if k == 1 else pow(self.alpha_expr, Integer(k))
                parts.append(QQ.to_expr(c) if k == 0 else mul(QQ.to_expr(c), power))
        return add(*parts) if parts else Integer(0)

    def from_expr(self, e: Expr) -> list[Fraction]:
        """A core expression built from rational numbers and ``alpha_expr``
        (and, when ``alpha_expr`` is a radical ``b**(1/n)``, the powers
        ``b**(k/n)``) by sums, products and integer powers."""
        alpha = self.alpha_expr
        radical_base: Expr = Integer(0)
        radical_index = 0
        if isinstance(alpha, Pow) and isinstance(alpha.exp, Rational) and alpha.exp.p == 1 and alpha.exp.q > 1:
            radical_base, radical_index = alpha.base, alpha.exp.q
        if e == alpha:
            return self.alpha
        if isinstance(e, Rational):
            return self.from_rational(QQ.from_expr(e))
        if radical_index and isinstance(e, Pow) and e.base == radical_base and isinstance(e.exp, Rational) \
                and e.exp.q == radical_index:
            k = e.exp.p
            return self.pow(self.alpha, k) if k >= 0 else self.pow(self.invert(self.alpha), -k)
        if isinstance(e, Add):
            total = self.zero
            for a in e.args:
                total = self.add(total, self.from_expr(a))
            return total
        if isinstance(e, Mul):
            product = self.one
            for a in e.args:
                product = self.mul(product, self.from_expr(a))
            return product
        if isinstance(e, Pow) and isinstance(e.exp, Integer):
            base = self.from_expr(e.base)
            n = e.exp.p
            return self.pow(base, n) if n >= 0 else self.pow(self.invert(base), -n)
        raise ValueError('%s is not an element of %s' % (e, self.name))

    def to_str(self, a: list[Fraction]) -> str:
        return str(self.to_expr(a))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AlgebraicField) and other.minpoly == self.minpoly and other.alpha_expr == self.alpha_expr

    def __hash__(self) -> int:
        return hash(('AlgebraicField', tuple(self.minpoly), self.alpha_expr))
