"""The Gaussian integers ``ZZ_I`` and the Gaussian rationals ``QQ_I``.

The elements are ``x + y*I`` with integer (``GaussianInteger``) or
rational (``GaussianRational``) parts. ``ZZ_I`` is a Euclidean domain:
the division rounds the exact quotient to the nearest Gaussian integer,
the gcd is the last nonzero remainder of the Euclidean algorithm made
canonical (in the first quadrant: the unit of the quadrant divided out).
``QQ_I`` is a field whose ring is ``ZZ_I`` (the denominators of a
polynomial are cleared over ``ZZ_I``, as for ``QQ`` and ``ZZ``).

Examples
========

>>> from sympy_extras.polys.domains.gaussiandomains import ZZ_I, QQ_I, GaussianInteger
>>> a, b = GaussianInteger(3, 4), GaussianInteger(1, 2)
>>> ZZ_I.mul(a, b), ZZ_I.divmod(a, b), ZZ_I.gcd(GaussianInteger(4, 2), GaussianInteger(2, 6))
(GaussianInteger(-5, 10), (GaussianInteger(2, 0), GaussianInteger(1, 0)), GaussianInteger(4, 2))
>>> ZZ_I.canonical_unit(GaussianInteger(-2, 3)), QQ_I.to_expr(QQ_I.invert(QQ_I.convert(a, ZZ_I)))
(GaussianInteger(0, -1), 3/25 - 4*I/25)

References
==========

.. [1] SymPy Development Team, *sympy.polys.domains.gaussiandomains*,
       https://docs.sympy.org/latest/modules/polys/domainsref.html#gaussian-domains
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/domains/gaussiandomains.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the arithmetic is on the domains (Domain[T] methods)
# with immutable element classes.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from math import gcd as integer_gcd
from typing import Final

from sympy_extras.core.expr import Expr
from .domain import Domain, U

__all__ = ['GaussianInteger', 'GaussianRational', 'GaussianIntegerRing', 'GaussianRationalField', 'ZZ_I', 'QQ_I']


class GaussianInteger:
    """``x + y*I`` with integers ``x``, ``y``."""

    __slots__ = ('x', 'y')

    def __init__(self, x: int, y: int = 0) -> None:
        self.x: Final[int] = x
        self.y: Final[int] = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussianInteger) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash(('ZZ_I', self.x, self.y))

    def __repr__(self) -> str:
        return 'GaussianInteger(%d, %d)' % (self.x, self.y)

    def __bool__(self) -> bool:
        return bool(self.x) or bool(self.y)

    # PORTED FROM SYMPY: GaussianElement.quadrant
    def quadrant(self) -> int:
        """``0`` to ``3``: the quadrant (``0``: ``x > 0, y >= 0``)."""
        return _quadrant(self.x, self.y)


class GaussianRational:
    """``x + y*I`` with rationals ``x``, ``y``."""

    __slots__ = ('x', 'y')

    def __init__(self, x: Fraction, y: Fraction = Fraction(0)) -> None:
        self.x: Final[Fraction] = x
        self.y: Final[Fraction] = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussianRational) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash(('QQ_I', self.x, self.y))

    def __repr__(self) -> str:
        return 'GaussianRational(%s, %s)' % (self.x, self.y)

    def __bool__(self) -> bool:
        return bool(self.x) or bool(self.y)

    def quadrant(self) -> int:
        return _quadrant(self.x, self.y)


def _quadrant(x: Fraction | int, y: Fraction | int) -> int:
    # PORTED FROM SYMPY: GaussianElement.quadrant
    if y > 0:
        return 0 if x > 0 else 1
    if y < 0:
        return 2 if x < 0 else 3
    return 0 if x >= 0 else 2


def _gaussian_expr(x: Expr, y: Expr) -> Expr:
    """``x + I*y`` (SymPy's ``to_sympy``)."""
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.numbers import I
    return add(x, mul(I, y))


def _gaussian_parts(e: Expr) -> tuple[Expr, Expr]:
    """``(x, y)`` of ``x + y*I`` (SymPy's ``from_sympy``); ``ValueError``
    when ``e`` is not of that form."""
    from sympy_extras.core.numbers import I, Integer, Rational
    r, b = e.as_coeff_Add()
    if isinstance(b, Rational) and b.p == 0:
        return r, Integer(0)
    c, rest = b.as_coeff_Mul()
    if rest == I:
        return r, c
    raise ValueError("%s is not Gaussian" % (e,))


class GaussianIntegerRing(Domain[GaussianInteger]):
    """The Gaussian integers ``ZZ_I`` (a Euclidean domain)."""

    name = 'ZZ_I'
    is_field = False
    is_ordered = False
    characteristic = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussianIntegerRing)

    def __hash__(self) -> int:
        return hash('ZZ_I')

    @property
    def zero(self) -> GaussianInteger:
        return GaussianInteger(0, 0)

    @property
    def one(self) -> GaussianInteger:
        return GaussianInteger(1, 0)

    @property
    def units(self) -> tuple[GaussianInteger, ...]:
        """The powers of ``I``: ``1, I, -1, -I``."""
        return (GaussianInteger(1, 0), GaussianInteger(0, 1), GaussianInteger(-1, 0), GaussianInteger(0, -1))

    def is_zero(self, a: GaussianInteger) -> bool:
        return not a

    def is_negative(self, a: GaussianInteger) -> bool:
        return False  # SymPy's GaussianDomain.is_negative

    def is_unit(self, a: GaussianInteger) -> bool:
        return a in self.units

    def add(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        return GaussianInteger(a.x + b.x, a.y + b.y)

    def sub(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        return GaussianInteger(a.x - b.x, a.y - b.y)

    def neg(self, a: GaussianInteger) -> GaussianInteger:
        return GaussianInteger(-a.x, -a.y)

    def mul(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        return GaussianInteger(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)

    # PORTED FROM SYMPY: GaussianInteger.__divmod__
    def divmod(self, a: GaussianInteger, b: GaussianInteger) -> tuple[GaussianInteger, GaussianInteger]:
        if not b:
            raise ZeroDivisionError('divmod({}, 0)'.format(a))
        x, y = b.x, b.y
        p, q = a.x*x + a.y*y, -a.x*y + a.y*x
        c = x*x + y*y
        qx = (2*p + c) // (2*c)  # -c <= 2*p - qx*2*c < c
        qy = (2*q + c) // (2*c)
        quotient = GaussianInteger(qx, qy)
        return quotient, self.sub(a, self.mul(quotient, b))  # |r| < |b|

    def quo(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        return self.divmod(a, b)[0]

    def rem(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        return self.divmod(a, b)[1]

    def exquo(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        if not b:
            raise ZeroDivisionError('exquo({}, 0)'.format(a))
        q, r = self.divmod(a, b)
        if r:
            raise ValueError('%s does not divide %s' % (b, a))
        return q

    def invert(self, a: GaussianInteger) -> GaussianInteger:
        if a not in self.units:
            raise ValueError('%s is not a unit' % (a,))
        return GaussianInteger(a.x, -a.y)

    # PORTED FROM SYMPY: GaussianDomain.canonical_unit
    def canonical_unit(self, a: GaussianInteger) -> GaussianInteger:
        return self.units[-a.quadrant()]  # the inverse of the unit of the quadrant

    # PORTED FROM SYMPY: GaussianIntegerRing.normalize
    def normalize(self, a: GaussianInteger) -> GaussianInteger:
        return self.mul(a, self.canonical_unit(a))

    # PORTED FROM SYMPY: GaussianIntegerRing.gcd
    def gcd(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        while b:
            a, b = b, self.rem(a, b)
        return self.normalize(a)

    # PORTED FROM SYMPY: GaussianIntegerRing.lcm
    def lcm(self, a: GaussianInteger, b: GaussianInteger) -> GaussianInteger:
        return self.quo(self.mul(a, b), self.gcd(a, b))

    # PORTED FROM SYMPY: GaussianIntegerRing.gcdex
    def gcdex(self, a: GaussianInteger, b: GaussianInteger) -> tuple[GaussianInteger, GaussianInteger, GaussianInteger]:
        x_a, x_b, y_a, y_b = self.one, self.zero, self.zero, self.one
        while b:
            q = self.quo(a, b)
            a, b = b, self.sub(a, self.mul(q, b))
            x_a, x_b = x_b, self.sub(x_a, self.mul(q, x_b))
            y_a, y_b = y_b, self.sub(y_a, self.mul(q, y_b))
        unit = self.canonical_unit(a)
        return self.mul(x_a, unit), self.mul(y_a, unit), self.mul(a, unit)

    def size(self, a: GaussianInteger) -> int:
        return max(abs(a.x).bit_length(), abs(a.y).bit_length(), 1)

    def from_int(self, n: int) -> GaussianInteger:
        return GaussianInteger(n, 0)

    def convert(self, a: U, K: Domain[U]) -> GaussianInteger:
        if isinstance(a, GaussianInteger):
            return a
        if isinstance(a, GaussianRational):
            if a.x.denominator != 1 or a.y.denominator != 1:
                raise ValueError('%s is not a Gaussian integer' % (a,))
            return GaussianInteger(a.x.numerator, a.y.numerator)
        return self.from_expr(K.to_expr(a))

    def to_expr(self, a: GaussianInteger) -> Expr:
        from sympy_extras.core.numbers import Integer
        return _gaussian_expr(Integer(a.x), Integer(a.y))

    def from_expr(self, e: Expr) -> GaussianInteger:
        from sympy_extras.core.numbers import Integer
        x, y = _gaussian_parts(e)
        if not (isinstance(x, Integer) and isinstance(y, Integer)):
            raise ValueError('%s is not a Gaussian integer' % (e,))
        return GaussianInteger(x.p, y.p)

    # the denominators of the Gaussian rationals (for QQ_I)
    def denom_of(self, a: GaussianRational) -> GaussianInteger:
        """SymPy's ``QQ_I.denom``: the lcm of the denominators of the parts."""
        d1, d2 = a.x.denominator, a.y.denominator
        return GaussianInteger(d1 * d2 // integer_gcd(d1, d2), 0)


class GaussianRationalField(Domain[GaussianRational]):
    """The Gaussian rationals ``QQ_I`` (a field whose ring is ``ZZ_I``)."""

    name = 'QQ_I'
    is_field = True
    is_ordered = False
    characteristic = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussianRationalField)

    def __hash__(self) -> int:
        return hash('QQ_I')

    @property
    def zero(self) -> GaussianRational:
        return GaussianRational(Fraction(0), Fraction(0))

    @property
    def one(self) -> GaussianRational:
        return GaussianRational(Fraction(1), Fraction(0))

    @property
    def units(self) -> tuple[GaussianRational, ...]:
        return (self.one, GaussianRational(Fraction(0), Fraction(1)), GaussianRational(Fraction(-1), Fraction(0)),
                GaussianRational(Fraction(0), Fraction(-1)))

    def is_zero(self, a: GaussianRational) -> bool:
        return not a

    def is_negative(self, a: GaussianRational) -> bool:
        return False

    def add(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        return GaussianRational(a.x + b.x, a.y + b.y)

    def sub(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        return GaussianRational(a.x - b.x, a.y - b.y)

    def neg(self, a: GaussianRational) -> GaussianRational:
        return GaussianRational(-a.x, -a.y)

    def mul(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        return GaussianRational(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)

    # PORTED FROM SYMPY: GaussianRational.__truediv__
    def exquo(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        if not b:
            raise ZeroDivisionError('{} / 0'.format(a))
        x, y = b.x, b.y
        c = x*x + y*y
        return GaussianRational((a.x*x + a.y*y)/c, (-a.x*y + a.y*x)/c)

    def quo(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        return self.exquo(a, b)

    def rem(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        if not b:
            raise ZeroDivisionError('{} % 0'.format(a))
        return self.zero

    def invert(self, a: GaussianRational) -> GaussianRational:
        return self.exquo(self.one, a)

    def gcd(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        return self.zero if not a and not b else self.one

    def lcm(self, a: GaussianRational, b: GaussianRational) -> GaussianRational:
        return self.zero if not a or not b else self.one

    def gcdex(self, a: GaussianRational, b: GaussianRational) -> tuple[GaussianRational, GaussianRational, GaussianRational]:
        if a:
            return self.invert(a), self.zero, self.one
        if b:
            return self.zero, self.invert(b), self.one
        return self.zero, self.zero, self.zero

    def canonical_unit(self, a: GaussianRational) -> GaussianRational:
        return self.units[-a.quadrant()]

    def size(self, a: GaussianRational) -> int:
        return max(a.x.numerator.bit_length(), a.x.denominator.bit_length(),
                   a.y.numerator.bit_length(), a.y.denominator.bit_length(), 1)

    def from_int(self, n: int) -> GaussianRational:
        return GaussianRational(Fraction(n), Fraction(0))

    def convert(self, a: U, K: Domain[U]) -> GaussianRational:
        if isinstance(a, GaussianRational):
            return a
        if isinstance(a, GaussianInteger):
            return GaussianRational(Fraction(a.x), Fraction(a.y))
        return self.from_expr(K.to_expr(a))

    def to_expr(self, a: GaussianRational) -> Expr:
        from sympy_extras.core.numbers import number_from_fraction
        return _gaussian_expr(number_from_fraction(a.x), number_from_fraction(a.y))

    def from_expr(self, e: Expr) -> GaussianRational:
        from sympy_extras.core.numbers import Rational
        x, y = _gaussian_parts(e)
        if not (isinstance(x, Rational) and isinstance(y, Rational)):
            raise ValueError('%s is not a Gaussian rational' % (e,))
        return GaussianRational(x.as_fraction(), y.as_fraction())


ZZ_I = GaussianIntegerRing()
QQ_I = GaussianRationalField()
