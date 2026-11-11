r"""The field of algebraic numbers: exact arithmetic on the roots of
polynomials with rational coefficients.

An :class:`Algebraic` is a complex algebraic number, represented by its
minimal polynomial over $\mathbb{Q}$ (primitive with integer
coefficients, positive leading coefficient) and by a region which
isolates it among the roots of that polynomial: a rational point, an
open interval with rational ends for a real number, a rectangle with
rational corners otherwise. The representation is canonical in its
first half and exact in the second, so that equality, the order of the
real numbers and signs are decided, never estimated.

**Arithmetic.** For $\gamma = \alpha + \beta$ the resultant
$\operatorname{Res}_y(p(x - y), q(y))$ of the minimal polynomials
vanishes at $\gamma$ (for a product, $\operatorname{Res}_y(y^d p(x/y),
q(y))$; for an inverse, the reversed polynomial). It is factored over
$\mathbb{Z}$, and the factor and the root which are $\gamma$ are found
by enclosures: $\alpha$ and $\beta$ are refined, interval arithmetic
gives a box containing $\gamma$, and the roots of each irreducible factor
in the box are counted exactly (winding numbers,
:mod:`sympy_extras.polys.complexroots`), with smaller and smaller boxes
until a single root of a single factor remains. That root is real
exactly when the box made symmetric about the real axis still contains
one root (the roots which are not real come in conjugate pairs).

A power $\alpha^n$ uses $\operatorname{Res}_y(p(y), x - y^n)$, of the
degree of $p$. The resultant of a sum or a product has the product of
the degrees, which is the cost of this representation: numbers of
degree beyond a few dozen are slow to combine.

**Decisions.** Two numbers with the same minimal polynomial are equal
when a box containing both isolating regions contains one root, and
different when the regions, refined, are disjoint; real numbers are
compared by refining their intervals until they separate.

Examples
========

>>> from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots
>>> sqrt2, = [r for r in algebraic_roots([1, 0, -2]) if r > 0]
>>> sqrt3, = [r for r in algebraic_roots([1, 0, -3]) if r > 0]
>>> s = sqrt2 + sqrt3
>>> s
Algebraic([1, 0, -10, 0, 1], 3.14626437)
>>> s*s == 5 + 2*sqrt2*sqrt3
True
>>> (sqrt2*sqrt2).minpoly, sqrt2*sqrt2 == 2
([1, -2], True)
>>> 1/(sqrt3 - sqrt2) == s
True
>>> i = Algebraic.imaginary_unit()
>>> (1 + i)**8
Algebraic([1, -16], 16)
>>> roots = algebraic_roots([1, 0, 0, -2])            # the cube roots of 2
>>> roots
[Algebraic([1, 0, 0, -2], 1.25992105), Algebraic([1, 0, 0, -2], -0.6299605249 - 1.091123636*I), Algebraic([1, 0, 0, -2], -0.6299605249 + 1.091123636*I)]
>>> roots[0] + roots[1] + roots[2] == 0, roots[0]*roots[1]*roots[2] == 2
(True, True)
>>> (roots[1]*roots[2]).is_real, roots[1].conjugate() == roots[2]
(True, True)

References
==========

.. [1] R. Loos, *Computing in algebraic extensions*, in Computer Algebra:
       Symbolic and Algebraic Computation, Springer, 1982, 173-187.
.. [2] H. Cohen, *A Course in Computational Algebraic Number Theory*,
       Springer, 1993, Section 4.2 (representations of algebraic
       numbers).
.. [3] S. Basu, R. Pollack, M.-F. Roy, *Algorithms in Real Algebraic
       Geometry*, 2nd ed., Springer, 2006, Section 10.4.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Final, Optional, Sequence, Union

from sympy_extras.polys.complexroots import Rectangle, _refine, _winding_number, dup_isolate_complex_roots_sqf
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.euclidtools import dup_gcd
from sympy_extras.polys.domains.polynomialring import PolynomialRing
from sympy_extras.polys.euclidtools import dup_resultant
from sympy_extras.polys.factortools import dup_zz_factor
from sympy_extras.polys.numberfields.minpoly import _product_resultant, _sum_resultant
from sympy_extras.polys.rootisolation import Interval, dup_isolate_real_roots_sqf, dup_refine_real_root

__all__ = ['Algebraic', 'algebraic_roots', 'AlgebraicLike', 'primitive_element']

#: a box ``[re_min, re_max] x [im_min, im_max]``, possibly flat
Box = tuple[Fraction, Fraction, Fraction, Fraction]

#: what the operations accept besides algebraic numbers
AlgebraicLike = Union['Algebraic', int, Fraction]

_ZERO = Fraction(0)

_ZX: Final[PolynomialRing[int]] = PolynomialRing(ZZ, 'x')


def _normalised(f: list[int]) -> list[int]:
    """Without common factor, the leading coefficient positive."""
    content = 0
    for c in f:
        content = _gcd(content, c)
    if content == 0:
        return []
    if f[0] < 0:
        content = -content
    return [c//content for c in f]


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _interval_product(a: Fraction, b: Fraction, c: Fraction, d: Fraction) -> tuple[Fraction, Fraction]:
    """The product of the intervals ``[a, b]`` and ``[c, d]``."""
    products = (a*c, a*d, b*c, b*d)
    return min(products), max(products)


def _box_sum(p: Box, q: Box) -> Box:
    return p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3]


def _box_product(p: Box, q: Box) -> Box:
    rr = _interval_product(p[0], p[1], q[0], q[1])
    ii = _interval_product(p[2], p[3], q[2], q[3])
    ri = _interval_product(p[0], p[1], q[2], q[3])
    ir = _interval_product(p[2], p[3], q[0], q[1])
    return rr[0] - ii[1], rr[1] - ii[0], ri[0] + ir[0], ri[1] + ir[1]


def _smallest_square(low: Fraction, high: Fraction) -> Fraction:
    """The minimum of ``t**2`` on ``[low, high]``."""
    if low <= 0 <= high:
        return _ZERO
    return min(low*low, high*high)


def _box_inverse(p: Box) -> Optional[Box]:
    """A box containing the inverses of the points of ``p``; ``None``
    when ``p`` contains zero."""
    norm_low = _smallest_square(p[0], p[1]) + _smallest_square(p[2], p[3])
    if norm_low == 0:
        return None
    norm_high = max(p[0]*p[0], p[1]*p[1]) + max(p[2]*p[2], p[3]*p[3])
    # (x - i y)/(x**2 + y**2)
    re = [p[0]/norm_low, p[0]/norm_high, p[1]/norm_low, p[1]/norm_high]
    im = [-p[2]/norm_low, -p[2]/norm_high, -p[3]/norm_low, -p[3]/norm_high]
    return min(re), max(re), min(im), max(im)


class Algebraic:
    """An algebraic number: its minimal polynomial and an isolating
    region.

    Build the numbers with :func:`algebraic_roots`,
    :meth:`Algebraic.from_rational`, :meth:`Algebraic.imaginary_unit` and
    the arithmetic operations (``+ - * / **`` with integer exponents,
    also with integers and fractions); ``minpoly`` is the minimal
    polynomial over ``QQ`` (integers, highest degree first, primitive,
    positive leading coefficient).

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots
    >>> low, high = algebraic_roots([1, -1, -1])          # x**2 = x + 1
    >>> high
    Algebraic([1, -1, -1], 1.618033989)
    >>> high**2 == high + 1, low*high == -1, low < 0 < high
    (True, True, True)
    >>> box = high.enclosure(Fraction(1, 10**6))
    >>> box[0] < Fraction(16180339, 10**7) < Fraction(16180340, 10**7) < box[1] or box[1] - box[0] < Fraction(1, 10**6)
    True
    >>> Algebraic.from_rational(Fraction(3, 4)).minpoly
    [4, -3]
    """

    def __init__(self, minpoly: list[int], interval: Optional[Interval], rectangle: Optional[Rectangle]) -> None:
        if (interval is None) == (rectangle is None):
            raise ValueError('an interval for a real number, a rectangle otherwise')
        self._minpoly: list[int] = minpoly
        self._rational_minpoly: list[Fraction] = [Fraction(c) for c in minpoly]
        self._interval: Optional[Interval] = interval
        self._rectangle: Optional[Rectangle] = rectangle

    # -- constructors

    @staticmethod
    def from_rational(q: Union[int, Fraction]) -> Algebraic:
        """The rational number ``q``."""
        value = Fraction(q)
        return Algebraic([value.denominator, -value.numerator], (value, value), None)

    @staticmethod
    def imaginary_unit() -> Algebraic:
        """The root ``i`` of ``x**2 + 1`` in the upper half-plane."""
        return Algebraic([1, 0, 1], None, Rectangle(Fraction(-1), Fraction(1), Fraction(1, 2), Fraction(2)))

    # -- description

    @property
    def minpoly(self) -> list[int]:
        return list(self._minpoly)

    @property
    def degree(self) -> int:
        return len(self._minpoly) - 1

    @property
    def is_real(self) -> bool:
        return self._interval is not None

    @property
    def is_rational(self) -> bool:
        return len(self._minpoly) == 2

    @property
    def is_zero(self) -> bool:
        return self._minpoly == [1, 0]

    def rational_value(self) -> Fraction:
        """The value of a rational number (``ValueError`` otherwise)."""
        if not self.is_rational:
            raise ValueError('not a rational number')
        return Fraction(-self._minpoly[1], self._minpoly[0])

    def _box(self, eps: Fraction) -> Box:
        """A box containing the number, with sides shorter than ``eps``
        (the isolating region is refined and kept)."""
        if self._interval is not None:
            low, high = self._interval
            if high - low >= eps:
                low, high = dup_refine_real_root(self._rational_minpoly, low, high, eps)
                self._interval = (low, high)
            return low, high, _ZERO, _ZERO
        rectangle = self._rectangle
        if rectangle is None:
            raise ValueError('no isolating region')
        if rectangle.width >= eps or rectangle.height >= eps:
            rectangle = _refine(self._minpoly, rectangle, eps)
            self._rectangle = rectangle
        return rectangle.corners()

    def real_interval(self) -> Interval:
        """The current isolating interval ``(a, b)`` of a real number: the
        number is in the open interval when ``a < b``, the number is ``a``
        when ``a == b`` (``TypeError`` for a number which is not real)."""
        if self._interval is None:
            raise TypeError('not a real number')
        return self._interval

    def refine_real(self) -> None:
        """Halve the isolating interval of a real number (a bisection:
        the interval becomes the point when its midpoint is the number)."""
        if self._interval is None:
            raise TypeError('not a real number')
        low, high = self._interval
        if low != high:
            self._interval = dup_refine_real_root(self._rational_minpoly, low, high, high - low)

    def enclosure(self, eps: Fraction) -> Box:
        """``(re_min, re_max, im_min, im_max)``: a box with rational
        corners and sides shorter than ``eps`` containing the number (a
        real number has ``im_min == im_max == 0``)."""
        if eps <= 0:
            raise ValueError('eps must be positive')
        return self._box(Fraction(eps))

    def __complex__(self) -> complex:
        box = self._box(Fraction(1, 10**18))
        return complex(float((box[0] + box[1])/2), float((box[2] + box[3])/2))

    def __float__(self) -> float:
        if not self.is_real:
            raise TypeError('not a real number')
        return complex(self).real

    def __repr__(self) -> str:
        if self.is_rational:
            shown = str(self.rational_value())
        else:
            value = complex(self)
            if self.is_real:
                shown = '%.10g' % value.real
            else:
                shown = '%.10g %s %.10g*I' % (value.real, '+' if value.imag > 0 else '-', abs(value.imag))
        return 'Algebraic(%r, %s)' % (self._minpoly, shown)

    # -- unary operations

    def __neg__(self) -> Algebraic:
        n = self.degree
        minpoly = _normalised([c if (n - k) % 2 == 0 else -c for k, c in enumerate(self._minpoly)])
        if self._interval is not None:
            return Algebraic(minpoly, (-self._interval[1], -self._interval[0]), None)
        r = self._rectangle
        if r is None:
            raise ValueError('no isolating region')
        return Algebraic(minpoly, None, Rectangle(-r.re_max, -r.re_min, -r.im_max, -r.im_min))

    def __pos__(self) -> Algebraic:
        return self

    def conjugate(self) -> Algebraic:
        """The complex conjugate."""
        if self._rectangle is None:
            return self
        return Algebraic(self._minpoly, None, self._rectangle.conjugate())

    def inverse(self) -> Algebraic:
        """``1/self`` (``ZeroDivisionError`` for zero)."""
        if self.is_zero:
            raise ZeroDivisionError('the inverse of zero')
        if self.is_rational:
            return Algebraic.from_rational(1/self.rational_value())
        reversed_minpoly = _normalised(self._minpoly[::-1])

        def enclosure(eps: Fraction) -> Optional[Box]:
            return _box_inverse(self._box(eps))

        return _select([reversed_minpoly], enclosure)

    # -- binary operations

    def __add__(self, other: AlgebraicLike) -> Algebraic:
        b = _coerce(other)
        if self.is_rational and b.is_rational:
            return Algebraic.from_rational(self.rational_value() + b.rational_value())
        resultant = _sum_resultant(self._minpoly, b._minpoly)

        def enclosure(eps: Fraction) -> Optional[Box]:
            return _box_sum(self._box(eps), b._box(eps))

        return _select(_irreducible_factors(resultant), enclosure)

    def __radd__(self, other: AlgebraicLike) -> Algebraic:
        return self.__add__(other)

    def __sub__(self, other: AlgebraicLike) -> Algebraic:
        return self.__add__(-_coerce(other))

    def __rsub__(self, other: AlgebraicLike) -> Algebraic:
        return (-self).__add__(other)

    def __mul__(self, other: AlgebraicLike) -> Algebraic:
        b = _coerce(other)
        if self.is_zero or b.is_zero:
            return Algebraic.from_rational(0)
        if b is self:
            return self.__pow__(2)
        if self.is_rational and b.is_rational:
            return Algebraic.from_rational(self.rational_value()*b.rational_value())
        resultant = _product_resultant(self._minpoly, b._minpoly)

        def enclosure(eps: Fraction) -> Optional[Box]:
            return _box_product(self._box(eps), b._box(eps))

        return _select(_irreducible_factors(resultant), enclosure)

    def __rmul__(self, other: AlgebraicLike) -> Algebraic:
        return self.__mul__(other)

    def __truediv__(self, other: AlgebraicLike) -> Algebraic:
        return self.__mul__(_coerce(other).inverse())

    def __rtruediv__(self, other: AlgebraicLike) -> Algebraic:
        return self.inverse().__mul__(other)

    def __pow__(self, exponent: int) -> Algebraic:
        if exponent < 0:
            return self.inverse().__pow__(-exponent)
        if exponent == 0:
            return Algebraic.from_rational(1)
        if exponent == 1:
            return self
        if self.is_rational:
            return Algebraic.from_rational(self.rational_value()**exponent)
        # Res_y(m(y), x - y**n) vanishes at the n-th power and has the
        # degree of m (a product would square the degree)
        target: list[list[int]] = [[-1]] + [[] for _ in range(exponent - 1)] + [[1, 0]]
        resultant = dup_resultant([[c] if c else [] for c in self._minpoly], target, _ZX)

        def enclosure(eps: Fraction) -> Optional[Box]:
            base = self._box(eps)
            box = base
            for _ in range(exponent - 1):
                box = _box_product(box, base)
            return box

        return _select(_irreducible_factors(resultant), enclosure)

    def polynomial_value(self, coefficients: Sequence[Union[int, Fraction]]) -> Algebraic:
        """``g(self)`` for the polynomial ``g`` with these rational
        coefficients (highest degree first), through one resultant:
        ``Res_y(m(y), D x - G(y))`` for ``g = G/D`` has the degree of the
        minimal polynomial ``m``, whereas Horner's rule in this class
        would take a resultant of squared degree at every step.

        Examples
        ========

        >>> from sympy_extras.polys.numberfields.qqbar import algebraic_roots
        >>> sqrt2 = algebraic_roots([1, 0, -2])[1]
        >>> sqrt2.polynomial_value([1, 0, 0, 1])               # sqrt(2)**3 + 1
        Algebraic([1, -2, -7], 3.828427125)
        >>> sqrt2.polynomial_value([1, 0, -2]).is_zero
        True
        """
        g = [Fraction(c) for c in coefficients]
        while g and g[0] == 0:
            g = g[1:]
        if len(g) <= 1 or self.is_rational:
            value = _ZERO
            point = self.rational_value() if self.is_rational else _ZERO
            for c in g:
                value = value*point + c
            return Algebraic.from_rational(value)
        common = 1
        for c in g:
            common = common*c.denominator//_gcd(common, c.denominator)
        integers = [int(c*common) for c in g]
        target: list[list[int]] = [[-c] if c else [] for c in integers[:-1]] + [[common, -integers[-1]]]
        resultant = dup_resultant([[c] if c else [] for c in self._minpoly], target, _ZX)

        def enclosure(eps: Fraction) -> Optional[Box]:
            base = self._box(eps)
            box: Box = (g[0], g[0], _ZERO, _ZERO)
            for c in g[1:]:
                box = _box_product(box, base)
                box = (box[0] + c, box[1] + c, box[2], box[3])
            return box

        return _select(_irreducible_factors(resultant), enclosure)

    def nth_root(self, n: int) -> Algebraic:
        """The principal ``n``-th root: the root of ``z**n = self`` whose
        argument is in ``(-pi/n, pi/n]``.

        Its minimal polynomial divides ``m(x**n)``. Among the roots of
        that polynomial, those whose ``n``-th power is this number are
        found by enclosures (there are ``n`` of them), and the principal
        one has the largest real part; for a negative real number two
        conjugate roots share it, and the one above the real axis is
        taken.

        Examples
        ========

        >>> from sympy_extras.polys.numberfields.qqbar import Algebraic
        >>> two = Algebraic.from_rational(2)
        >>> two.nth_root(2)
        Algebraic([1, 0, -2], 1.414213562)
        >>> (two + two.nth_root(2)).nth_root(2)             # sqrt(2 + sqrt(2)) = 2 cos(pi/8)
        Algebraic([1, 0, -4, 0, 2], 1.847759065)
        >>> Algebraic.from_rational(-1).nth_root(2) == Algebraic.imaginary_unit()
        True
        >>> Algebraic.from_rational(-8).nth_root(3)         # the principal cube root, not -2
        Algebraic([1, -2, 4], 1 + 1.732050808*I)
        """
        if n < 1:
            raise ValueError('the index of the root must be positive')
        if n == 1 or self.is_zero:
            return self
        expanded: list[int] = []
        for c in self._minpoly[:-1]:
            expanded.extend([c] + [0]*(n - 1))
        expanded.append(self._minpoly[-1])
        real_roots, other_roots = _roots(expanded)
        candidates = real_roots + other_roots
        eps = Fraction(1, 4)
        while len(candidates) > n:
            target = self._box(eps)
            kept: list[Algebraic] = []
            for candidate in candidates:
                base = candidate._box(eps)
                box = base
                for _ in range(n - 1):
                    box = _box_product(box, base)
                if not (box[1] < target[0] or target[1] < box[0] or box[3] < target[2] or target[3] < box[2]):
                    kept.append(candidate)
            candidates = kept
            eps = eps/16
        if len(candidates) != n:
            raise ArithmeticError('%d roots of index %d were found' % (len(candidates), n))
        best = candidates[0]
        for candidate in candidates[1:]:
            best = _larger_real_part(best, candidate)
        return best

    def sqrt(self) -> Algebraic:
        """The principal square root."""
        return self.nth_root(2)

    def real_part(self) -> Algebraic:
        """``(z + conjugate(z))/2``."""
        if self.is_real:
            return self
        return (self + self.conjugate())*Fraction(1, 2)

    def imaginary_part(self) -> Algebraic:
        """``(z - conjugate(z))/(2 i)``, a real number."""
        if self.is_real:
            return Algebraic.from_rational(0)
        return (self - self.conjugate())*Fraction(1, 2)*(-Algebraic.imaginary_unit())

    # -- decisions

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, Fraction)):
            return self.is_rational and self.rational_value() == other
        if not isinstance(other, Algebraic):
            return NotImplemented
        if self._minpoly != other._minpoly or self.is_real != other.is_real:
            return False
        if self.is_rational:
            return True
        eps = Fraction(1, 4)
        while True:
            p, q = self._box(eps), other._box(eps)
            if p[1] < q[0] or q[1] < p[0] or p[3] < q[2] or q[3] < p[2]:
                return False
            hull = (min(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), max(p[3], q[3]))
            count = _roots_in(self._minpoly, hull, eps)
            if count == 1:
                return True
            eps = eps/16

    def __hash__(self) -> int:
        return hash((tuple(self._minpoly), self.is_real))

    def sign(self) -> int:
        """The sign of a real number."""
        if self._interval is None:
            raise TypeError('not a real number')
        if self.is_zero:
            return 0
        eps = Fraction(1, 4)
        while True:
            low, high, _, _ = self._box(eps)
            if low > 0 or (low == 0 and high > 0):
                return 1
            if high < 0 or (high == 0 and low < 0):
                return -1
            eps = eps/16

    def _compare(self, other: AlgebraicLike) -> int:
        b = _coerce(other)
        if not (self.is_real and b.is_real):
            raise TypeError('complex numbers are not ordered')
        if self == b:
            return 0
        eps = Fraction(1, 4)
        while True:
            p, q = self._box(eps), b._box(eps)
            if p[1] < q[0]:
                return -1
            if q[1] < p[0]:
                return 1
            if p[1] == q[0] and p[0] < q[1]:
                # open intervals (or a point and an interval) which touch
                return -1
            if q[1] == p[0] and q[0] < p[1]:
                return 1
            eps = eps/16

    def __lt__(self, other: AlgebraicLike) -> bool:
        return self._compare(other) < 0

    def __le__(self, other: AlgebraicLike) -> bool:
        return self._compare(other) <= 0

    def __gt__(self, other: AlgebraicLike) -> bool:
        return self._compare(other) > 0

    def __ge__(self, other: AlgebraicLike) -> bool:
        return self._compare(other) >= 0


def _larger_real_part(a: Algebraic, b: Algebraic) -> Algebraic:
    """The one of the two numbers with the larger real part; when the
    real parts are equal, the one with the larger imaginary part."""
    eps = Fraction(1, 4)
    while eps > Fraction(1, 2**64):
        p, q = a._box(eps), b._box(eps)
        if p[0] > q[1]:
            return a
        if q[0] > p[1]:
            return b
        eps = eps/16
    # close real parts: conjugate numbers have the same one; otherwise
    # they are compared exactly
    if not a.is_real and not b.is_real and a.conjugate() == b:
        return a if _above(a) else b
    ra, rb = a.real_part(), b.real_part()
    if ra != rb:
        return a if ra > rb else b
    return a if a.imaginary_part() > b.imaginary_part() else b


def _coerce(value: AlgebraicLike) -> Algebraic:
    if isinstance(value, Algebraic):
        return value
    return Algebraic.from_rational(value)


def _irreducible_factors(f: list[int]) -> list[list[int]]:
    """The distinct irreducible factors of positive degree."""
    _, factors = dup_zz_factor(f, ZZ)
    return [_normalised(g) for g, _ in factors]


def _roots_in(g: list[int], box: Box, pad: Fraction) -> Optional[int]:
    """The number of roots of ``g`` in the box widened by ``pad`` (or a
    little more when a root lies on its boundary); ``None`` when every
    widening tried has a root on its boundary."""
    for k in range(4):
        margin = pad*(4 + k)/4
        count = _winding_number(g, Rectangle(box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin))
        if count is not None:
            return count
    return None


def _select(factors: list[list[int]], enclosure: Callable[[Fraction], Optional[Box]]) -> Algebraic:
    """The root of one of the irreducible ``factors`` which the boxes
    ``enclosure(eps)`` contain for every ``eps``."""
    eps = Fraction(1, 4)
    while True:
        box = enclosure(eps)
        if box is None:
            eps = eps/16
            continue
        counts = [_roots_in(g, box, eps) for g in factors]
        found = [(g, count) for g, count in zip(factors, counts) if count != 0]
        if len(found) != 1 or found[0][1] != 1:
            eps = eps/16
            continue
        g = found[0][0]
        if len(g) == 2:
            return Algebraic.from_rational(Fraction(-g[1], g[0]))
        margin = eps
        if box[2] - margin > 0 or box[3] + margin < 0:
            # away from the real axis; the rectangle which was counted
            for k in range(4):
                margin = eps*(4 + k)/4
                rectangle = Rectangle(box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin)
                if _winding_number(g, rectangle) == 1 and (rectangle.im_min > 0 or rectangle.im_max < 0):
                    return Algebraic(g, None, rectangle)
            eps = eps/16
            continue
        # the box meets the real axis: real exactly when the symmetric
        # box still has one root
        height = max(abs(box[2]), abs(box[3])) + 2*eps
        symmetric = (box[0], box[1], -height, height)
        if _roots_in(g, symmetric, eps) == 1:
            for k in range(4):
                margin = eps*(4 + k)/4
                rectangle = Rectangle(box[0] - margin, box[1] + margin, -height - margin, height + margin)
                if _winding_number(g, rectangle) == 1:
                    return Algebraic(g, (rectangle.re_min, rectangle.re_max), None)
        eps = eps/16


def algebraic_roots(f: list[int]) -> list[Algebraic]:
    """The distinct complex roots of a non-zero polynomial with integer
    coefficients (highest degree first): the real roots in increasing
    order, then the others by real part, the root below the real axis
    before its conjugate.

    Examples
    ========

    >>> from sympy_extras.polys.numberfields.qqbar import algebraic_roots
    >>> algebraic_roots([1, 0, -3, 0, 2])                  # (x**2 - 1)(x**2 - 2)
    [Algebraic([1, 0, -2], -1.414213562), Algebraic([1, 1], -1), Algebraic([1, -1], 1), Algebraic([1, 0, -2], 1.414213562)]
    >>> algebraic_roots([1, 0, 1])
    [Algebraic([1, 0, 1], 0 - 1*I), Algebraic([1, 0, 1], 0 + 1*I)]
    """
    real, others = _roots(f)
    real.sort(key=_RealKey)
    others.sort(key=_ComplexKey)
    return real + others


def _roots(f: list[int]) -> tuple[list[Algebraic], list[Algebraic]]:
    """The real roots and the other roots, in no particular order."""
    if not any(f):
        raise ValueError('the zero polynomial')
    real: list[Algebraic] = []
    others: list[Algebraic] = []
    for g in _irreducible_factors(f):
        if len(g) == 2:
            real.append(Algebraic.from_rational(Fraction(-g[1], g[0])))
            continue
        rational = [Fraction(c) for c in g]
        for interval in dup_isolate_real_roots_sqf(rational):
            real.append(Algebraic(g, interval, None))
        for rectangle in dup_isolate_complex_roots_sqf(rational):
            others.append(Algebraic(g, None, rectangle.conjugate()))
            others.append(Algebraic(g, None, rectangle))
    return real, others


def _above(a: Algebraic) -> bool:
    """Whether a number which is not real is above the real axis."""
    eps = Fraction(1, 4)
    while True:
        box = a._box(eps)
        if box[2] > 0:
            return True
        if box[3] < 0:
            return False
        eps = eps/16


class _RealKey:
    """Sorting key: the order of the real numbers."""

    def __init__(self, value: Algebraic) -> None:
        self.value = value

    def __lt__(self, other: _RealKey) -> bool:
        return self.value < other.value


class _ComplexKey:
    """Sorting key: by real part, then by imaginary part."""

    def __init__(self, value: Algebraic) -> None:
        self.value = value

    def __lt__(self, other: _ComplexKey) -> bool:
        a, b = self.value, other.value
        eps = Fraction(1, 4)
        while eps > Fraction(1, 2**200):
            p, q = a._box(eps), b._box(eps)
            if p[1] < q[0]:
                return True
            if q[1] < p[0]:
                return False
            eps = eps/16
            if eps < Fraction(1, 2**40):
                # the real parts may be equal: those of conjugate numbers
                # are, the others are compared exactly
                if not a.is_real and not b.is_real and a.conjugate() == b:
                    return not _above(a)
                ra, rb = a.real_part(), b.real_part()
                if ra != rb:
                    return ra < rb
                return a.imaginary_part() < b.imaginary_part()
        return False


def primitive_element(a: Algebraic, b: Algebraic) -> tuple[Algebraic, list[Fraction], list[Fraction]]:
    """A primitive element of the field generated by two algebraic
    numbers: ``(theta, p, q)`` with ``theta = a + c b`` for a small
    integer ``c``, and polynomials with rational coefficients (highest
    degree first) such that ``a = p(theta)`` and ``b = q(theta)``.

    For ``c`` outside a finite set, ``b`` is the only common root of
    ``m_b(x)`` and ``m_a(theta - c x)``, so that their gcd over
    ``QQ(theta)`` is ``x - b`` and gives ``b`` as a polynomial in
    ``theta`` (the theorem of the primitive element, made effective).

    Examples
    ========

    >>> from sympy_extras.polys.numberfields.qqbar import algebraic_roots, primitive_element
    >>> sqrt2 = algebraic_roots([1, 0, -2])[1]
    >>> sqrt3 = algebraic_roots([1, 0, -3])[1]
    >>> theta, p, q = primitive_element(sqrt2, sqrt3)
    >>> theta, p == [Fraction(1, 2), 0, Fraction(-9, 2), 0], q == [Fraction(-1, 2), 0, Fraction(11, 2), 0]
    (Algebraic([1, 0, -10, 0, 1], 3.14626437), True, True)
    >>> theta.polynomial_value(p) == sqrt2, theta.polynomial_value(q) == sqrt3
    (True, True)
    """
    if b.is_rational:
        return a, [Fraction(1), Fraction(0)], [b.rational_value()]
    if a.is_rational:
        return b, [a.rational_value()], [Fraction(1), Fraction(0)]
    ma = [Fraction(v) for v in a.minpoly]
    mb = [Fraction(v) for v in b.minpoly]
    for c in range(1, 4*a.degree*b.degree + 2):
        theta = a + c*b
        K = AlgebraicField([Fraction(v, theta.minpoly[0]) for v in theta.minpoly], Symbol('theta'))
        # m_a(theta - c x) as a polynomial in x over QQ(theta), by Horner
        shifted: list[list[Fraction]] = []
        argument: list[list[Fraction]] = [[Fraction(-c)], K.alpha]
        for coefficient in ma:
            product: list[list[Fraction]] = [K.zero]*(len(shifted) + 1) if shifted else []
            for i, u in enumerate(shifted):
                product[i] = K.add(product[i], K.mul(u, argument[0]))
                product[i + 1] = K.add(product[i + 1], K.mul(u, argument[1]))
            if not product:
                product = [K.from_rational(coefficient)] if coefficient else []
            else:
                product[-1] = K.add(product[-1], K.from_rational(coefficient))
            shifted = product
        common = dup_gcd(shifted, [K.from_rational(v) for v in mb], K)
        if len(common) != 2:
            continue
        q = K.neg(K.quo(common[1], common[0]))
        p = K.sub(K.alpha, K.mul(K.from_int(c), q))
        return theta, p, q
    raise ArithmeticError('no primitive element of the form a + c b was found')
