r"""Isolation and counting of the complex roots of dense polynomials with
rational coefficients, by exact winding numbers.

**Counting.** The number of roots of $f$ inside a rectangle with no root
on its boundary is the winding number of $f$ along the boundary
(the argument principle). On an edge $z = z_0 + t w$, $0 \le t \le 1$,
write $c f(z) = u(t) + i v(t)$ with real polynomials $u, v$, for a
constant $c$ chosen so that $u$ does not vanish at the corners. Each
time the path crosses the imaginary axis counter-clockwise the quotient
$v/u$ jumps from $+\infty$ to $-\infty$, so that the winding number is
$-\frac12 \sum_{\text{edges}} \operatorname{Ind}_0^1 (v/u)$, the Cauchy
indices being counted exactly by Sturm's theorem: $\operatorname{Ind}_a^b
(v/u) = V(a) - V(b)$ for the sign variations of the signed remainder
sequence $u, v, -\operatorname{rem}(u, v), \dots$ The last polynomial of
the sequence is $\gcd(u, v)$, whose roots in $[0, 1]$ are the roots of
$f$ on the edge: they are detected (again by Sturm) and the rectangle is
refused. Everything is computed with integers (the rectangle and the
polynomial scaled by their denominators, pseudo-remainders made
primitive).

**Isolation.** For a square-free real $f$ the roots which are not real
come in conjugate pairs, and their number is known from the count of the
real roots. A strip $|\operatorname{Im} z| < \varepsilon$ around the
real axis free of them is found by halving $\varepsilon$ until the
rectangle $[-B, B] \times [\varepsilon, B]$ ($B$ a root bound) contains
all the pairs; it is then bisected along its longer side (the cut being
moved when it meets a root) until every piece contains one root. The
rectangles returned have rational corners, positive imaginary parts,
disjoint interiors and no root on their boundaries; each contains one
root, and its mirror image the conjugate root.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.complexroots import (Rectangle, dup_count_complex_roots,
...     dup_isolate_complex_roots_sqf, dup_isolate_all_roots_sqf)
>>> f = [Fraction(1), Fraction(0), Fraction(0), Fraction(-1)]      # x**3 - 1
>>> dup_isolate_complex_roots_sqf(f)
[Rectangle(-3, 3, 3/4, 3)]
>>> dup_isolate_complex_roots_sqf(f, Fraction(1, 10))
[Rectangle(-33/64, -31/64, 109/128, 113/128)]
>>> dup_isolate_all_roots_sqf(f)
([(Fraction(0, 1), Fraction(2, 1))], [Rectangle(-3, 3, 3/4, 3)])
>>> dup_count_complex_roots(f, Rectangle(Fraction(-2), Fraction(2), Fraction(-2), Fraction(2)))
3
>>> dup_count_complex_roots(f, Rectangle(Fraction(-2), Fraction(0), Fraction(-2), Fraction(2)))
2

References
==========

.. [1] H. S. Wilf, *A global bisection algorithm for computing the zeros
       of polynomials in the complex plane*, Journal of the ACM 25
       (1978), 415-420.
.. [2] G. E. Collins, W. Krandick, *An efficient algorithm for infallible
       polynomial complex root isolation*, ISSAC '92, 189-194.
.. [3] S. Basu, R. Pollack, M.-F. Roy, *Algorithms in Real Algebraic
       Geometry*, 2nd ed., Springer, 2006, Sections 2.2.2 (Cauchy index,
       signed remainder sequences) and 9.1.
.. [4] P. Henrici, *Applied and Computational Complex Analysis*, vol. 1,
       Wiley, 1974, Section 6.11 (zeros in a region by Sturm sequences).
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd, lcm
from typing import Final, Optional

from .densebasic import dup_degree
from .densetools import dup_clear_denoms
from .domains.rationalfield import QQ
from .rootisolation import Interval, dup_count_real_roots, dup_fujiwara_bound, dup_isolate_real_roots_sqf
from .sqfreetools import dup_sqf_list, dup_sqf_p

__all__ = ['Rectangle', 'dup_count_complex_roots', 'dup_isolate_complex_roots_sqf', 'dup_refine_complex_root',
           'dup_isolate_all_roots_sqf', 'dup_isolate_complex_roots']

#: the constants ``c = a + b i`` tried so that ``Re(c f)`` does not vanish
#: at the corners: pairwise non-parallel, and a corner excludes one
#: direction, so that one of five suits four corners
_DIRECTIONS: list[tuple[int, int]] = [(1, 0), (1, 1), (1, -1), (1, 2), (2, 1)]


class Rectangle:
    """The rectangle ``[re_min, re_max] x [im_min, im_max]`` of the
    complex plane, with rational corners.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.complexroots import Rectangle
    >>> r = Rectangle(Fraction(-1), Fraction(1, 2), Fraction(1), Fraction(3))
    >>> r
    Rectangle(-1, 1/2, 1, 3)
    >>> r.width == Fraction(3, 2), r.height == 2
    (True, True)
    >>> r.conjugate()
    Rectangle(-1, 1/2, -3, -1)
    >>> r.contains(Fraction(0), Fraction(2))
    True
    """

    def __init__(self, re_min: Fraction, re_max: Fraction, im_min: Fraction, im_max: Fraction) -> None:
        if not (re_min < re_max and im_min < im_max):
            raise ValueError('an empty rectangle')
        self.re_min: Final[Fraction] = Fraction(re_min)
        self.re_max: Final[Fraction] = Fraction(re_max)
        self.im_min: Final[Fraction] = Fraction(im_min)
        self.im_max: Final[Fraction] = Fraction(im_max)

    @property
    def width(self) -> Fraction:
        return self.re_max - self.re_min

    @property
    def height(self) -> Fraction:
        return self.im_max - self.im_min

    @property
    def center(self) -> tuple[Fraction, Fraction]:
        """The real and imaginary parts of the centre."""
        return (self.re_min + self.re_max)/2, (self.im_min + self.im_max)/2

    def corners(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return self.re_min, self.re_max, self.im_min, self.im_max

    def conjugate(self) -> Rectangle:
        """The mirror image in the real axis."""
        return Rectangle(self.re_min, self.re_max, -self.im_max, -self.im_min)

    def contains(self, re: Fraction, im: Fraction) -> bool:
        """Whether the point ``re + im i`` is in the closed rectangle."""
        return self.re_min <= re <= self.re_max and self.im_min <= im <= self.im_max

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Rectangle) and self.corners() == other.corners()

    def __hash__(self) -> int:
        return hash(self.corners())

    def __repr__(self) -> str:
        return 'Rectangle(%s, %s, %s, %s)' % self.corners()


# ---------------------------------------------------------------------------
# integer polynomials (highest degree first)

def _strip(f: list[int]) -> list[int]:
    i = 0
    while i < len(f) and f[i] == 0:
        i += 1
    return f[i:]


def _primitive(f: list[int]) -> list[int]:
    """``f`` divided by the (positive) gcd of its coefficients."""
    g = 0
    for c in f:
        g = gcd(g, c)
        if g == 1:
            return f
    return [c//g for c in f] if g > 1 else f


def _negated_remainder(a: list[int], b: list[int]) -> list[int]:
    """A positive multiple of ``-rem(a, b)``, primitive."""
    da, db = len(a) - 1, len(b) - 1
    if da < db:
        return [-c for c in a]
    lc = b[0]
    scale = abs(lc)**(da - db + 1)
    r = [scale*c for c in a]
    for i in range(da - db + 1):
        q = r[i]//lc
        if q:
            for j in range(1, db + 1):
                r[i + j] -= q*b[j]
    return _primitive([-c for c in _strip(r[da - db + 1:])])


def _signed_remainder_sequence(u: list[int], v: list[int]) -> list[list[int]]:
    """``u, v, -rem(u, v), ...`` up to positive factors, without the
    final zero: the last polynomial is the gcd."""
    sequence = [_primitive(u), _primitive(v)]
    while sequence[-1]:
        sequence.append(_negated_remainder(sequence[-2], sequence[-1]))
    sequence.pop()
    return sequence


def _variations(sequence: list[list[int]], at_one: bool) -> int:
    """The sign variations of the values at 1 (the sums of the
    coefficients) or at 0 (the constant terms)."""
    count = 0
    previous = 0
    for g in sequence:
        value = sum(g) if at_one else g[-1]
        if value == 0:
            continue
        s = 1 if value > 0 else -1
        if previous and s != previous:
            count += 1
        previous = s
    return count


def _has_root_in_unit_interval(g: list[int]) -> bool:
    """Whether ``g`` has a root in ``[0, 1]``."""
    if len(g) <= 1:
        return False
    if g[-1] == 0 or sum(g) == 0:
        return True
    n = len(g) - 1
    derivative = [c*(n - i) for i, c in enumerate(g[:-1])]
    sequence = _signed_remainder_sequence(g, derivative)
    return _variations(sequence, False) != _variations(sequence, True)


def _on_segment(a: list[int], scale: int, z0: tuple[int, int], w: tuple[int, int]) -> tuple[list[int], list[int]]:
    """The real and imaginary parts of ``sum a_k scale**k Z**(n - k)``
    at ``Z = z0 + t w`` (Gaussian integers), as polynomials in ``t``,
    lowest degree first: ``scale**n f((z0 + t w)/scale)`` by Horner's
    rule."""
    x0, y0 = z0
    wx, wy = w
    re: list[int] = [a[0]]
    im: list[int] = [0]
    power = 1
    for c in a[1:]:
        power *= scale
        size = len(re)
        new_re = [0]*(size + 1)
        new_im = [0]*(size + 1)
        for j in range(size):
            r, s = re[j], im[j]
            new_re[j] += r*x0 - s*y0
            new_im[j] += r*y0 + s*x0
            new_re[j + 1] += r*wx - s*wy
            new_im[j + 1] += r*wy + s*wx
        new_re[0] += c*power
        re, im = new_re, new_im
    return re, im


def _winding_number(a: list[int], rectangle: Rectangle) -> Optional[int]:
    """The number of roots of the integer polynomial ``a`` inside the
    rectangle; ``None`` when a root lies on the boundary."""
    x0, x1, y0, y1 = rectangle.corners()
    scale = lcm(x0.denominator, x1.denominator, y0.denominator, y1.denominator)
    X0, X1, Y0, Y1 = int(x0*scale), int(x1*scale), int(y0*scale), int(y1*scale)
    # counter-clockwise from the lower left corner
    segments = [((X0, Y0), (X1 - X0, 0)), ((X1, Y0), (0, Y1 - Y0)), ((X1, Y1), (X0 - X1, 0)), ((X0, Y1), (0, Y0 - Y1))]
    edges = [_on_segment(a, scale, z0, w) for z0, w in segments]
    direction: Optional[tuple[int, int]] = None
    for ca, cb in _DIRECTIONS:
        if all(ca*re[0] - cb*im[0] for re, im in edges):
            direction = (ca, cb)
            break
    if direction is None:
        # a corner is a root (four corners exclude four directions at most)
        return None
    ca, cb = direction
    total = 0
    for re, im in edges:
        u = _strip([ca*r - cb*s for r, s in zip(re, im)][::-1])
        v = _strip([ca*s + cb*r for r, s in zip(re, im)][::-1])
        if not v:
            # c f is real and without zero at the ends; its zeros on the
            # edge are those of u
            if _has_root_in_unit_interval(u):
                return None
            continue
        sequence = _signed_remainder_sequence(u, v)
        if _has_root_in_unit_interval(sequence[-1]):
            return None
        total += _variations(sequence, False) - _variations(sequence, True)
    if total % 2:
        raise ArithmeticError('odd total Cauchy index on a closed path')
    return -total//2


def _integer_coefficients(f: list[Fraction]) -> list[int]:
    if dup_degree(f) < 0:
        raise ValueError('the zero polynomial')
    _, cleared = dup_clear_denoms(f, QQ)
    return [int(c) for c in cleared]


def dup_count_complex_roots(f: list[Fraction], rectangle: Rectangle) -> int:
    """The number of complex roots of ``f``, with multiplicities, in the
    interior of the rectangle.

    Raises ``ValueError`` when a root lies on the boundary.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.complexroots import Rectangle, dup_count_complex_roots
    >>> f = [Fraction(1), Fraction(0), Fraction(2), Fraction(0), Fraction(1)]     # (x**2 + 1)**2
    >>> dup_count_complex_roots(f, Rectangle(Fraction(-1), Fraction(1), Fraction(0), Fraction(2)))
    2
    >>> dup_count_complex_roots(f, Rectangle(Fraction(1, 2), Fraction(1), Fraction(0), Fraction(2)))
    0
    """
    count = _winding_number(_integer_coefficients(f), rectangle)
    if count is None:
        raise ValueError('a root lies on the boundary of %s' % (rectangle,))
    return count


def _cuts(count: int) -> list[Fraction]:
    """Positions of a cut along a side, as fractions of its length: the
    middle first, then ``2 count`` others in the middle half (a
    polynomial with ``count`` roots in the rectangle spoils at most
    ``count`` of them)."""
    result = [Fraction(1, 2)]
    for j in range(1, count + 1):
        step = Fraction(j, 4*(count + 1))
        result.append(Fraction(1, 2) + step)
        result.append(Fraction(1, 2) - step)
    return result


def _split(a: list[int], rectangle: Rectangle, count: int) -> tuple[Rectangle, int, Rectangle, int]:
    """The rectangle (with ``count`` roots inside, none on the boundary)
    cut along its longer side into two with no root on the cut, and
    their numbers of roots."""
    x0, x1, y0, y1 = rectangle.corners()
    for position in _cuts(count):
        if rectangle.width >= rectangle.height:
            middle = x0 + position*(x1 - x0)
            first, second = Rectangle(x0, middle, y0, y1), Rectangle(middle, x1, y0, y1)
        else:
            middle = y0 + position*(y1 - y0)
            first, second = Rectangle(x0, x1, y0, middle), Rectangle(x0, x1, middle, y1)
        inside = _winding_number(a, first)
        if inside is not None:
            return first, inside, second, count - inside
    raise ArithmeticError('no cut of %s avoids the roots' % (rectangle,))


def _newton(a: list[int], rectangle: Rectangle, precision: int) -> tuple[int, int]:
    """Newton's iteration from the centre of the rectangle in fixed
    point arithmetic: the last iterate ``(X + Y i)/2**precision`` (no
    promise that it is near a root)."""
    n = len(a) - 1
    re, im = rectangle.center
    X = int(re*2**precision)
    Y = int(im*2**precision)
    for _ in range(precision + 30):
        # G(Z) = 2**(precision n) f(Z/2**precision) and G'(Z) by Horner
        gr, gi, dr, di = a[0], 0, 0, 0
        for k in range(1, n + 1):
            dr, di = dr*X - di*Y + gr, dr*Y + di*X + gi
            gr, gi = gr*X - gi*Y + (a[k] << (precision*k)), gr*Y + gi*X
        norm = dr*dr + di*di
        if norm == 0:
            break
        # the step G/G', rounded
        step_x = (2*(gr*dr + gi*di) + norm)//(2*norm)
        step_y = (2*(gi*dr - gr*di) + norm)//(2*norm)
        if step_x == 0 and step_y == 0:
            break
        X -= step_x
        Y -= step_y
    return X, Y


def _one_root_near(a: list[int], X: int, Y: int, precision: int) -> bool:
    """A sufficient test (Rouché) that the square of half-side ``4/2**
    precision`` centred at ``(X + Y i)/2**precision`` has exactly one
    root inside and none on its boundary.

    With the Taylor coefficients ``B_k`` at the centre (in the unit
    ``2**-precision``), ``|B_1| r > |B_0| + sum_{k >= 2} |B_k| r**k``
    means that the disc of radius ``r`` contains exactly one root. It is
    asked for ``r = 4`` (the root is in the square) and for ``r = 6``
    (a disc which contains the square has no other root); the absolute
    values are bounded by ``max(|re|, |im|) <= |B| <= |re| + |im|``."""
    n = len(a) - 1
    re = [c << (precision*k) for k, c in enumerate(a)]
    im = [0]*(n + 1)
    # Taylor shift by repeated synthetic division: B_k ends at index n - k
    for i in range(n):
        for j in range(1, n - i + 1):
            r, t = re[j - 1], im[j - 1]
            re[j] += r*X - t*Y
            im[j] += r*Y + t*X
    lower = max(abs(re[n - 1]), abs(im[n - 1]))
    for radius in (4, 6):
        rest = abs(re[n]) + abs(im[n])
        power = radius
        for k in range(2, n + 1):
            power *= radius
            rest += (abs(re[n - k]) + abs(im[n - k]))*power
        if lower*radius <= rest:
            return False
    return True


def _refine(a: list[int], rectangle: Rectangle, eps: Optional[Fraction]) -> Rectangle:
    """A rectangle with one root shrunk until its sides are below
    ``eps``: Newton's iteration from the centre proposes a small square,
    which is accepted when it lies in the rectangle and is proved to
    contain the root (by Rouché's theorem on the Taylor expansion at its
    centre, or else by its winding number); otherwise the rectangle is
    bisected and the iteration starts again."""
    if eps is None:
        return rectangle
    precision = 8
    while Fraction(1, 2**precision) > eps/16:
        precision += 1
    unit = Fraction(1, 2**precision)
    while rectangle.width >= eps or rectangle.height >= eps:
        X, Y = _newton(a, rectangle, precision)
        x, y = X*unit, Y*unit
        x0, x1, y0, y1 = rectangle.corners()
        if x0 < x - 4*unit and x + 4*unit < x1 and y0 < y - 4*unit and y + 4*unit < y1:
            square = Rectangle(x - 4*unit, x + 4*unit, y - 4*unit, y + 4*unit)
            if _one_root_near(a, X, Y, precision) or _winding_number(a, square) == 1:
                return square
        first, inside, second, _ = _split(a, rectangle, 1)
        rectangle = first if inside else second
    return rectangle


def dup_refine_complex_root(f: list[Fraction], rectangle: Rectangle, eps: Fraction) -> Rectangle:
    """A rectangle inside the given one, with sides shorter than ``eps``,
    containing the same root: ``rectangle`` must contain exactly one root
    of ``f`` in its interior and none on its boundary.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.complexroots import Rectangle, dup_refine_complex_root
    >>> f = [Fraction(1), Fraction(0), Fraction(1)]
    >>> dup_refine_complex_root(f, Rectangle(Fraction(-1), Fraction(2), Fraction(0), Fraction(3)), Fraction(1, 4))
    Rectangle(-1/64, 1/64, 63/64, 65/64)
    """
    if eps <= 0:
        raise ValueError('eps must be positive')
    a = _integer_coefficients(f)
    if _winding_number(a, rectangle) != 1:
        raise ValueError('%s does not isolate a root' % (rectangle,))
    return _refine(a, rectangle, eps)


def dup_isolate_complex_roots_sqf(f: list[Fraction], eps: Optional[Fraction] = None) -> list[Rectangle]:
    """Isolating rectangles of the roots with positive imaginary part of
    a square-free ``f`` with rational coefficients.

    Each rectangle has rational corners, lies above the real axis and
    contains exactly one root in its interior, none on its boundary; the
    interiors are disjoint, and the conjugate rectangles isolate the
    roots with negative imaginary part. The rectangles are sorted by
    their lower left corners; with ``eps`` their sides are shorter than
    ``eps``.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.complexroots import dup_isolate_complex_roots_sqf
    >>> f = [Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(4)]     # roots +-1 +- i
    >>> dup_isolate_complex_roots_sqf(f)
    [Rectangle(-5, 0, 5/8, 5), Rectangle(0, 5, 5/8, 5)]
    >>> dup_isolate_complex_roots_sqf([Fraction(1), Fraction(0), Fraction(-2)])
    []
    """
    n = dup_degree(f)
    if n < 1:
        return []
    if not dup_sqf_p(f, QQ):
        raise ValueError('the polynomial is not square-free')
    if eps is not None and eps <= 0:
        raise ValueError('eps must be positive')
    pairs = (n - dup_count_real_roots(f))//2
    if pairs == 0:
        return []
    a = _integer_coefficients(f)
    bound = Fraction(int(dup_fujiwara_bound(f)) + 1)
    # a strip around the real axis without roots off the axis
    low = bound
    while True:
        low = low/2
        whole = Rectangle(-bound, bound, low, bound)
        if _winding_number(a, whole) == pairs:
            break
    pending: list[tuple[Rectangle, int]] = [(whole, pairs)]
    isolated: list[Rectangle] = []
    while pending:
        rectangle, count = pending.pop()
        if count == 1:
            isolated.append(_refine(a, rectangle, eps))
            continue
        first, inside, second, outside = _split(a, rectangle, count)
        if inside:
            pending.append((first, inside))
        if outside:
            pending.append((second, outside))
    isolated.sort(key=lambda r: (r.re_min, r.im_min))
    return isolated


def dup_isolate_all_roots_sqf(f: list[Fraction], eps: Optional[Fraction] = None
                              ) -> tuple[list[Interval], list[Rectangle]]:
    """The real roots of a square-free ``f`` by isolating intervals
    (:func:`sympy_extras.polys.rootisolation.dup_isolate_real_roots_sqf`)
    and the roots above the real axis by isolating rectangles: with the
    conjugates of the rectangles, all the ``deg f`` roots.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.complexroots import dup_isolate_all_roots_sqf
    >>> real, upper = dup_isolate_all_roots_sqf([Fraction(c) for c in (1, 0, 0, 0, 0, -1)])
    >>> len(real), len(upper)
    (1, 2)
    """
    return dup_isolate_real_roots_sqf(f, eps), dup_isolate_complex_roots_sqf(f, eps)


def _overlap(r: Rectangle, s: Rectangle) -> bool:
    """Whether the closed rectangles meet."""
    return not (r.re_max < s.re_min or s.re_max < r.re_min or r.im_max < s.im_min or s.im_max < r.im_min)


def dup_isolate_complex_roots(f: list[Fraction], eps: Optional[Fraction] = None) -> list[tuple[Rectangle, int]]:
    """Isolating rectangles of the roots with positive imaginary part of
    any ``f`` with rational coefficients, with their multiplicities.

    The square-free factors are isolated one by one, and the rectangles
    of different factors (which have no common root) are refined until
    they are disjoint.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.complexroots import dup_isolate_complex_roots
    >>> f = [Fraction(c) for c in (1, 0, 3, 0, 3, 0, 1)]          # (x**2 + 1)**3
    >>> dup_isolate_complex_roots(f)
    [(Rectangle(-3, 3, 3/4, 3), 3)]
    """
    if dup_degree(f) < 1:
        return []
    _, factors = dup_sqf_list(f, QQ)
    found: list[tuple[Rectangle, int, list[int]]] = []
    for g, multiplicity in factors:
        if dup_degree(g) < 1:
            continue
        a = _integer_coefficients(g)
        for rectangle in dup_isolate_complex_roots_sqf(g, eps):
            found.append((rectangle, multiplicity, a))
    while True:
        clash = False
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                r, s = found[i][0], found[j][0]
                if found[i][2] is not found[j][2] and _overlap(r, s):
                    clash = True
                    size = min(r.width, r.height, s.width, s.height)/2
                    found[i] = (_refine(found[i][2], r, size), found[i][1], found[i][2])
                    found[j] = (_refine(found[j][2], s, size), found[j][1], found[j][2])
        if not clash:
            break
    result = [(rectangle, multiplicity) for rectangle, multiplicity, _ in found]
    result.sort(key=lambda item: (item[0].re_min, item[0].im_min))
    return result
