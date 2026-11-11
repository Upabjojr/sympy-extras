"""Exact real algebraic sample points for cylindrical algebraic decomposition.

The lifting phase of CAD needs to evaluate polynomials at points whose
coordinates are real algebraic numbers, to find the real roots of the
resulting univariate polynomials and to compare them exactly. Here a real
algebraic number is an :class:`~sympy_extras.polys.numberfields.qqbar.Algebraic`
(its minimal polynomial over `\\mathbb{Z}` and an isolating interval with
rational ends, a point for a rational number), and a sample point is a
single real algebraic number `\\theta` together with the coordinates as
elements of the field `\\mathbb{Q}(\\theta)`
(:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField`).
Signs and comparisons are decided by interval arithmetic, refining the
isolating interval of `\\theta` as needed, and the field is extended by a
primitive element whenever a new algebraic coordinate is added.

The module is native: it works on the typed polynomials of
:mod:`sympy_extras.polys` and does not import SymPy; the SymPy face, on
SymPy's ``CRootOf``, is :mod:`sympy_extras.polys.cad.samplepoints_sympy`.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.core import symbols
>>> from sympy_extras.polys.cad.samplepoints import SamplePoint, real_roots_of
>>> x, y = symbols('x y')
>>> sqrt2 = real_roots_of([1, 0, -2])[1]
>>> q = SamplePoint().extend(sqrt2)
>>> q.real_roots(x**2 + y**2 - 1, [x, y])
[]
>>> q.real_roots(x**2 - y**2 - 1, [x, y])
[Algebraic([1, 1], -1), Algebraic([1, -1], 1)]
>>> p = q.extend(Fraction(1, 2))
>>> p
SamplePoint(Algebraic([1, 0, -2], 1.414213562), Algebraic([2, -1], 1/2))
>>> p.sign(x**2 + y**2 - 2, [x, y]), p.sign(2*y - 1, [x, y])
(1, 0)

References
==========

.. [1] G. E. Collins, R. Loos, Real zeros of polynomials, in: Computer
       Algebra: Symbolic and Algebraic Computation, Springer, 1982,
       pp. 83-94 (Sturm sequences over real algebraic number fields).
"""
from __future__ import annotations

from fractions import Fraction
from functools import cmp_to_key
from itertools import count
from math import floor
from typing import Optional, Sequence, TypeVar, Union

from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Rational
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.densearith import dup_div, dup_neg
from sympy_extras.polys.densebasic import dup_degree, dup_strip
from sympy_extras.polys.densetools import dup_diff, dup_eval
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.integerring import ZZ
from sympy_extras.polys.domains.rationalfield import QQ
from sympy_extras.polys.euclidtools import dup_gcd
from sympy_extras.polys.factortools import dup_ext_norm, dup_zz_factor
from sympy_extras.polys.numberfields.minpoly import _sum_resultant
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.rings import Monomial
from sympy_extras.polys.rootisolation import dup_count_real_roots, dup_isolate_real_roots_sqf
from sympy_extras.polys.sqfreetools import dup_sqf_list, dup_sqf_p

__all__ = ['RealAlgebraic', 'SamplePoint', 'Specialization', 'compare_real', 'rational_real_roots',
           'real_roots_of', 'simplest_between', 'rational_between', 'rational_below', 'rational_above',
           'floor_scaled', 'as_poly', 'Sign']

T = TypeVar('T')

#: a sign: -1, 0 or 1
Sign = int

#: a real algebraic number (a rational number is one of minimal
#: polynomial of degree one)
RealAlgebraic = Algebraic

#: what the functions accept for a real algebraic number
RealLike = Union[Algebraic, Fraction, int]

#: the polynomial arguments
PolyLike = Union[Poly, ExprLike]


def _algebraic(r: RealLike) -> Algebraic:
    if isinstance(r, Algebraic):
        if not r.is_real:
            raise ValueError("%s is not real" % (r,))
        return r
    return Algebraic.from_rational(Fraction(r))


def _bounds(r: Algebraic) -> tuple[Fraction, Fraction]:
    """Rational lower and upper bounds of the real algebraic number ``r``
    (equal for a rational number)."""
    return r.real_interval()


def _normalised(f: list[int]) -> list[int]:
    """``f`` primitive with a positive leading coefficient."""
    g = 0
    for c in f:
        g = _gcd(g, c)
    if f and f[0] < 0:
        g = -g
    return [c // g for c in f] if g not in (0, 1) else list(f)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _irreducible_factors(f: list[int]) -> list[tuple[list[int], int]]:
    """The irreducible factors of positive degree of the integer
    polynomial ``f``, primitive with positive leading coefficients, with
    their multiplicities."""
    _, factors = dup_zz_factor(f, ZZ)
    return [(_normalised(g), k) for g, k in factors if len(g) > 1]


def _real_roots_of_irreducible(g: list[int]) -> list[Algebraic]:
    """The real roots of the irreducible primitive ``g`` in increasing
    order."""
    if len(g) == 2:
        return [Algebraic.from_rational(Fraction(-g[1], g[0]))]
    return [Algebraic(g, interval, None) for interval in dup_isolate_real_roots_sqf([Fraction(c) for c in g])]


def _integer_dense(f: Sequence[Fraction]) -> list[int]:
    """The rational polynomial ``f`` times the lcm of its denominators."""
    d = 1
    for c in f:
        d = d*c.denominator//_gcd(d, c.denominator)
    return [int(c*d) for c in f]


def compare_real(a: RealLike, b: RealLike) -> Sign:
    """Compare two real algebraic numbers exactly: ``-1``, ``0`` or ``1``.

    The isolating intervals are refined, the wider one first, until they
    are disjoint; two irrational numbers with one minimal polynomial are
    equal when an interval containing both holds a single root of it
    (counted by Sturm's theorem), which refinement alone never finds.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.cad.samplepoints import compare_real, real_roots_of
    >>> sqrt2, cbrt2 = real_roots_of([1, 0, -2])[1], real_roots_of([1, 0, 0, -2])[0]
    >>> compare_real(sqrt2, Fraction(3, 2)), compare_real(sqrt2, cbrt2)
    (-1, 1)
    >>> compare_real(sqrt2, real_roots_of([2, 0, -4])[1])
    0
    """
    x, y = _algebraic(a), _algebraic(b)
    if x is y:
        return 0
    same = x.minpoly == y.minpoly and x.degree > 1
    rational = [Fraction(c) for c in x.minpoly] if same else []
    while True:
        alo, ahi = _bounds(x)
        blo, bhi = _bounds(y)
        if ahi < blo or (ahi == blo and (alo < ahi or blo < bhi)):
            return -1
        if bhi < alo or (bhi == alo and (blo < bhi or alo < ahi)):
            return 1
        if alo == ahi and blo == bhi:
            return 0
        if same and dup_count_real_roots(rational, min(alo, blo), max(ahi, bhi)) == 1:
            return 0
        if ahi - alo >= bhi - blo:
            x.refine_real()
        else:
            y.refine_real()


def _order(values: Sequence[Algebraic]) -> list[list[int]]:
    """The positions of the real algebraic numbers ``values`` in increasing
    order, the positions of equal numbers grouped together.

    The order of two numbers whose isolating intervals are disjoint is read
    off the intervals; only the others are compared exactly, which refines
    their intervals.

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.cad.samplepoints import _order, real_roots_of
    >>> m, p = real_roots_of([1, 0, -2])
    >>> _order([Algebraic.from_rational(1), p, Algebraic.from_rational(0), p, m])
    [[4], [2], [0], [1, 3]]
    """
    bounds = [_bounds(v) for v in values]

    def compare(i: int, j: int) -> int:
        (alo, ahi), (blo, bhi) = bounds[i], bounds[j]
        if ahi < blo:
            return -1
        if bhi < alo:
            return 1
        c = compare_real(values[i], values[j])
        bounds[i], bounds[j] = _bounds(values[i]), _bounds(values[j])
        return c

    groups: list[list[int]] = []
    for i in sorted(range(len(values)), key=cmp_to_key(compare)):
        if groups and compare(groups[-1][-1], i) == 0:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _dense_real_roots(f: Sequence[Fraction]) -> list[tuple[Algebraic, int]]:
    """The distinct real roots of the rational polynomial ``f`` (dense,
    highest degree first), in increasing order, with their
    multiplicities: each irreducible factor isolated on its own, the
    roots merged by :func:`_order`."""
    g = dup_strip(list(f), QQ)
    if dup_degree(g) < 1:
        return []
    values: list[Algebraic] = []
    multiplicities: list[int] = []
    for h, m in _irreducible_factors(_integer_dense(g)):
        for r in _real_roots_of_irreducible(h):
            values.append(r)
            multiplicities.append(m)
    return [(values[group[0]], sum(multiplicities[i] for i in group)) for group in _order(values)]


def real_roots_of(f: Sequence[Union[int, Fraction]]) -> list[Algebraic]:
    """The distinct real roots, in increasing order, of the polynomial
    with the rational coefficients ``f`` (highest degree first).

    >>> from sympy_extras.polys.cad.samplepoints import real_roots_of
    >>> real_roots_of([1, 0, -3, 0, 2])
    [Algebraic([1, 0, -2], -1.414213562), Algebraic([1, 1], -1), Algebraic([1, -1], 1), Algebraic([1, 0, -2], 1.414213562)]
    """
    return [r for r, _ in _dense_real_roots([Fraction(c) for c in f])]


def as_poly(f: PolyLike, gens: Sequence[Expr]) -> Poly:
    """``f`` as a :class:`~sympy_extras.polys.polytools.Poly` in ``gens``."""
    generators = tuple(gens)
    if isinstance(f, Poly):
        if f.gens == generators:
            return f
        return Poly(f.as_expr(), *generators)
    return Poly(f, *generators)


def _univariate(f: Poly) -> list[Fraction]:
    """The univariate ``f`` as a dense list, highest degree first."""
    if len(f.gens) != 1:
        raise PolynomialError("a univariate polynomial expected, got %s" % (f,))
    n = f.degree()
    dense = [Fraction(0)]*(n + 1)
    for m, c in f.rep.terms.items():
        dense[n - m[0]] = c
    return dense


def rational_real_roots(f: Poly) -> list[tuple[Algebraic, int]]:
    """The distinct real roots of the univariate polynomial ``f`` over
    ``QQ``, in increasing order, with their multiplicities.

    Each irreducible factor is isolated on its own and the roots of the
    factors are merged by :func:`_order`.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.polytools import Poly
    >>> from sympy_extras.polys.cad.samplepoints import rational_real_roots
    >>> x = Symbol('x')
    >>> [(float(r), m) for r, m in rational_real_roots(Poly((2*x - 1)**2*(x**2 - 2)*(x**2 + 1), x))]
    [(-1.4142135623730951, 1), (0.5, 2), (1.4142135623730951, 1)]
    """
    return _dense_real_roots(_univariate(f))


def floor_scaled(a: RealLike, k: int) -> int:
    """Exact value of `\\lfloor 2^k a \\rfloor` for a real algebraic ``a``.

    >>> from sympy_extras.polys.cad.samplepoints import floor_scaled, real_roots_of
    >>> [floor_scaled(real_roots_of([1, 0, -2])[1], k) for k in range(4)]
    [1, 2, 5, 11]
    """
    r = _algebraic(a)
    scale = Fraction(2)**k
    while True:
        lo, hi = _bounds(r)
        flo, fhi = floor(lo*scale), floor(hi*scale)
        if flo == fhi:
            return flo
        r.refine_real()


def simplest_between(lo: Fraction, hi: Optional[Fraction], lo_open: bool = True,
                     hi_open: bool = True) -> Fraction:
    """The rational number with the smallest denominator in the interval
    with rational endpoints ``lo`` and ``hi``. The endpoints are excluded
    unless ``lo_open`` or ``hi_open`` is ``False``; ``hi`` may be ``None``
    for `+\\infty`.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.cad.samplepoints import simplest_between
    >>> simplest_between(Fraction(1, 3), Fraction(1, 2))
    2/5
    >>> simplest_between(Fraction(1, 3), Fraction(1, 2), hi_open=False)
    1/2
    >>> simplest_between(Fraction(5, 2), Fraction(3))
    8/3
    """
    if hi is None:
        if lo < 0 or (lo == 0 and not lo_open):
            return Fraction(0)
        n = floor(lo)
        return Fraction(n if n == lo and not lo_open else n + 1)
    if lo > hi or (lo == hi and (lo_open or hi_open)):
        raise ValueError("empty interval")
    if (lo < 0 or (lo == 0 and not lo_open)) and (hi > 0 or (hi == 0 and not hi_open)):
        return Fraction(0)
    if hi < 0 or (hi == 0 and hi_open):
        return -simplest_between(-hi, -lo, hi_open, lo_open)
    # now 0 <= lo < hi, with lo excluded if it is 0
    n = floor(lo)
    c = n if (n == lo and not lo_open) else n + 1
    if c < hi or (c == hi and not hi_open):
        return Fraction(c)
    # no integer is admissible: both bounds lie in [n, n + 1] and the
    # result is n + 1/q with q in the reciprocal interval
    inner_hi = 1/(lo - n) if lo > n else None
    inner = simplest_between(1/(hi - n), inner_hi, hi_open, lo_open)
    return n + 1/inner


def _dyadic_bounds(r: Algebraic, k: int) -> tuple[Fraction, Fraction, bool, bool]:
    """Rational bounds of the real algebraic number ``r`` at the scale
    `2^{-k}`, and whether they are attained.

    For a rational number both bounds are the number itself and are
    excluded; for an irrational one they are the dyadic numbers just below
    and above it, and are included.
    """
    if r.is_rational:
        c = r.rational_value()
        return c, c, True, True
    scale = Fraction(2)**k
    lo = Fraction(floor_scaled(r, k))/scale
    return lo, lo + 1/scale, False, False


def rational_between(a: RealLike, b: RealLike) -> Fraction:
    """A simple rational number strictly between the real algebraic
    numbers ``a < b``.

    The result only depends on ``a`` and ``b``: it is the rational number
    with the smallest denominator between the dyadic approximations of the
    irrational endpoints, at the coarsest scale that separates them.

    Examples
    ========

    >>> from sympy_extras.polys.cad.samplepoints import rational_between, real_roots_of
    >>> m, p = real_roots_of([1, 0, -2])
    >>> rational_between(m, p), rational_between(1, p), rational_between(-1, 1)
    (Fraction(0, 1), Fraction(5, 4), Fraction(0, 1))
    """
    x, y = _algebraic(a), _algebraic(b)
    if compare_real(x, y) >= 0:
        raise ValueError("%s is not smaller than %s" % (a, b))
    for k in count(0):
        _, hi, _, hi_open = _dyadic_bounds(x, k)
        lo, _, lo_open, _ = _dyadic_bounds(y, k)
        if hi < lo or (hi == lo and not (hi_open or lo_open)):
            return simplest_between(hi, lo, hi_open, lo_open)
    raise RuntimeError("unreachable")  # pragma: no cover


def rational_below(a: RealLike) -> Fraction:
    """A simple rational number smaller than the real algebraic number ``a``.

    >>> from sympy_extras.polys.cad.samplepoints import rational_below, real_roots_of
    >>> rational_below(real_roots_of([1, 0, -2])[1]), rational_below(-3)
    (Fraction(0, 1), Fraction(-4, 1))
    """
    r = _algebraic(a)
    if compare_real(r, 0) > 0:
        return Fraction(0)
    fl = floor_scaled(r, 0)
    if r.is_rational and fl == r.rational_value():
        fl -= 1
    return Fraction(fl)


def rational_above(a: RealLike) -> Fraction:
    """A simple rational number greater than the real algebraic number ``a``.

    >>> from sympy_extras.polys.cad.samplepoints import rational_above, real_roots_of
    >>> rational_above(real_roots_of([1, 0, -2])[1])
    2
    """
    r = _algebraic(a)
    if compare_real(r, 0) < 0:
        return Fraction(0)
    return Fraction(floor_scaled(r, 0) + 1)


def _sign_in_field(theta: Algebraic, a: list[Fraction]) -> Sign:
    """Sign of the element ``a`` of `\\mathbb{Q}(\\theta)` (a polynomial
    in ``theta``, highest degree first)."""
    if not a:
        return 0
    while True:
        lo, hi = _bounds(theta)
        vlo = vhi = Fraction(0)
        for c in a:
            products = [vlo*lo, vlo*hi, vhi*lo, vhi*hi]
            vlo, vhi = min(products) + c, max(products) + c
        if vlo > 0:
            return 1
        if vhi < 0:
            return -1
        theta.refine_real()


def _field_of(theta: Algebraic) -> AlgebraicField:
    """The field ``QQ(theta)`` (its generator printed ``theta``)."""
    lc = theta.minpoly[0]
    return AlgebraicField([Fraction(c, lc) for c in theta.minpoly], Symbol('theta'))


def _join(theta: Algebraic, beta: Algebraic) -> tuple[Algebraic, AlgebraicField, list[Fraction], list[Fraction]]:
    """Primitive element of `\\mathbb{Q}(\\theta, \\beta)`.

    Returns ``(gamma, K, theta_K, beta_K)`` where ``gamma = theta + k
    beta`` for the first ``k = 1, 2, ...`` for which the resultant of the
    minimal polynomials is squarefree (so that ``gamma`` is primitive),
    ``K`` is the field ``QQ(gamma)`` and ``theta_K``, ``beta_K`` are
    ``theta`` and ``beta`` as elements of ``K``.
    """
    m1, m2 = theta.minpoly, beta.minpoly
    for k in count(1):
        # the minimal polynomial of k beta, then the polynomial whose roots
        # are the sums of a conjugate of theta and k times one of beta
        scaled = [c*k**i for i, c in enumerate(m2)]
        N = _sum_resultant(m1, scaled)
        if dup_sqf_p(N, ZZ):
            break
    candidates = [r for g, _ in _irreducible_factors(N) for r in _real_roots_of_irreducible(g)]
    while True:
        tlo, thi = _bounds(theta)
        blo, bhi = _bounds(beta)
        lo, hi = tlo + k*blo, thi + k*bhi
        alive = []
        for r in candidates:
            rlo, rhi = _bounds(r)
            if not (rhi < lo or hi < rlo):
                alive.append(r)
        if len(alive) == 1:
            break
        # the candidates come from different factors, so their isolating
        # intervals may overlap: refine them as well
        theta.refine_real()
        beta.refine_real()
        for r in alive:
            r.refine_real()
    gamma = alive[0]
    if gamma.is_rational:
        raise PolynomialError("unexpected rational primitive element")
    K = _field_of(gamma)
    # beta is the only common root of m2(t) and m1(gamma - k*t)
    u: list[list[Fraction]] = [K.from_int(-k), K.alpha]
    f: list[list[Fraction]] = []
    for c in m1:
        product: list[list[Fraction]] = [K.zero]*(len(f) + 1) if f else []
        for i, v in enumerate(f):
            product[i] = K.add(product[i], K.mul(v, u[0]))
            product[i + 1] = K.add(product[i + 1], K.mul(v, u[1]))
        if not product:
            product = [K.from_int(c)] if c else []
        else:
            product[-1] = K.add(product[-1], K.from_int(c))
        f = dup_strip(product, K)
    g = dup_gcd([K.from_int(c) for c in m2], f, K)
    if dup_degree(g) != 1:
        raise PolynomialError("failed to compute a primitive element")
    beta_K = K.neg(K.quo(g[1], g[0]))
    theta_K = K.sub(K.alpha, K.mul(K.from_int(k), beta_K))
    return gamma, K, theta_K, beta_K


def _separate(candidates: Sequence[Algebraic]) -> None:
    """Refine the isolating intervals of the real algebraic numbers
    ``candidates``, which are pairwise distinct, until they are pairwise
    disjoint (the roots of different polynomials may have overlapping
    intervals)."""
    while True:
        bounds = [_bounds(r) for r in candidates]
        crowded: set[int] = set()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if not (bounds[i][1] < bounds[j][0] or bounds[j][1] < bounds[i][0]):
                    crowded.add(i)
                    crowded.add(j)
        if not crowded:
            return
        for i in crowded:
            candidates[i].refine_real()


def _sturm(f: list[T], K: Domain[T]) -> list[list[T]]:
    """The Sturm sequence ``f, f', -rem(f, f'), ...`` over the field ``K``."""
    sequence = [list(f), dup_diff(f, 1, K)]
    while sequence[-1]:
        sequence.append(dup_neg(dup_div(sequence[-2], sequence[-1], K)[1], K))
    sequence.pop()
    return sequence


def _variations(sturm: Sequence[list[list[Fraction]]], value: Fraction, theta: Algebraic, K: AlgebraicField) -> int:
    """The number of sign variations of the Sturm sequence ``sturm`` over
    `\\mathbb{Q}(\\theta)` at the rational ``value``, the signs decided
    exactly."""
    point = K.from_rational(value)
    signs = [_sign_in_field(theta, dup_eval(f, point, K)) for f in sturm]
    nonzero = [sign for sign in signs if sign]
    return sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)


def _real_roots_over(theta: Algebraic, K: AlgebraicField, f: list[list[Fraction]]) -> list[tuple[Algebraic, int]]:
    """The distinct real roots, with their multiplicities and in increasing
    order, of the univariate polynomial ``f`` over the field ``K =
    QQ(theta)``.

    The roots are among the real roots of the norm of each squarefree
    factor (a polynomial over the rationals); each candidate is kept or
    dropped by counting the roots of the factor in its isolating interval
    with a Sturm sequence over the field, whose signs are decided by
    interval arithmetic on ``theta`` [1]_.
    """
    found: list[tuple[Algebraic, int]] = []
    for g, multiplicity in dup_sqf_list(f, K)[1]:
        if dup_degree(g) < 1:
            continue
        candidates: list[Algebraic] = [r for r, _ in _dense_real_roots(dup_ext_norm(g, K))]
        if not candidates:
            continue
        sturm = _sturm(g, K)
        _separate(candidates)
        total = 0
        for r in candidates:
            if r.is_rational:
                if not dup_eval(g, K.from_rational(r.rational_value()), K):
                    found.append((r, multiplicity))
                    total += 1
                continue
            lo, hi = _bounds(r)
            # the endpoints are not roots of the norm, so not of g: the
            # count is of the open interval, which holds one root of the
            # norm and so at most one of g
            number = _variations(sturm, lo, theta, K) - _variations(sturm, hi, theta, K)
            if number == 1:
                found.append((r, multiplicity))
                total += 1
            elif number:
                raise PolynomialError("%d roots of a factor in the isolating interval of %s" % (number, r))
        if total != _variations(sturm, _bounds(candidates[0])[0] - 1, theta, K) - \
                _variations(sturm, _bounds(candidates[-1])[1] + 1, theta, K):
            raise PolynomialError("the real roots of a factor are not among those of its norm")
    return [found[group[0]] for group in _order([root for root, _ in found])]


class Specialization:
    """A polynomial with the coordinates of a sample point put for all its
    variables but the last: what the lifting phase needs to know of it.

    Attributes
    ==========

    degree : int
        The degree in the last variable at the point, ``-1`` for a
        polynomial which vanishes identically there.
    leading_sign : int
        The sign of the leading coefficient at the point (0 for the zero
        polynomial).
    roots : list of pairs
        The distinct real roots in increasing order, each with its
        multiplicity.

    The sign of the polynomial on the line over the point follows from
    these: it is ``leading_sign`` beyond the last root and changes at the
    roots of odd multiplicity (see :meth:`sign_before`).

    >>> from sympy_extras.polys.cad.samplepoints import Specialization
    >>> from sympy_extras.polys.numberfields.qqbar import Algebraic
    >>> s = Specialization(3, -1, [(Algebraic.from_rational(-1), 1), (Algebraic.from_rational(1), 2)])
    >>> [s.sign_before(i) for i in range(3)]
    [1, -1, -1]
    """

    __slots__ = ('degree', 'leading_sign', 'roots')

    def __init__(self, degree: int, leading_sign: Sign, roots: Sequence[tuple[Algebraic, int]]) -> None:
        self.degree: int = degree
        self.leading_sign: Sign = leading_sign
        self.roots: list[tuple[Algebraic, int]] = list(roots)

    def sign_before(self, position: int) -> Sign:
        """The sign of the polynomial between its roots of index
        ``position - 1`` and ``position`` (below the first root for 0,
        above the last one for ``len(roots)``)."""
        if not 0 <= position <= len(self.roots):
            raise ValueError("there are %d roots" % len(self.roots))
        odd = sum(multiplicity for _, multiplicity in self.roots[position:]) % 2
        return -self.leading_sign if odd else self.leading_sign


def _evaluate(terms: dict[Monomial, Fraction], values: Sequence[T], K: Domain[T], last: bool) -> list[T]:
    """The polynomial with the rational coefficients ``terms`` at the
    point ``values`` of ``K``: with ``last`` a dense polynomial (highest
    degree first) in the variable after them, otherwise a list holding the
    value."""
    n = len(values)
    powers: list[dict[int, T]] = [{0: K.one, 1: v} for v in values]

    def power(i: int, e: int) -> T:
        found = powers[i].get(e)
        if found is None:
            found = K.mul(power(i, e - 1), values[i])
            powers[i][e] = found
        return found

    collected: dict[int, T] = {}
    for m, c in terms.items():
        value = K.convert(c, QQ)
        for i in range(n):
            if m[i]:
                value = K.mul(value, power(i, m[i]))
        e = m[n] if last else 0
        previous = collected.get(e)
        collected[e] = value if previous is None else K.add(previous, value)
    if not collected:
        return []
    top = max(collected)
    return dup_strip([collected.get(top - i, K.zero) for i in range(top + 1)], K)


class SamplePoint:
    """A point with real algebraic coordinates.

    The coordinates are elements of the field ``K = QQ(theta)`` (or of
    ``QQ``, when ``theta`` is ``None``), kept as polynomials in ``theta``
    (highest degree first), and as :class:`Algebraic` numbers in
    ``values``. Points are built by extending the origin one coordinate at
    a time with :meth:`extend`.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.samplepoints import SamplePoint, real_roots_of
    >>> x, y = symbols('x y')
    >>> q = SamplePoint().extend(real_roots_of([1, 0, -2])[1])
    >>> q.degree, len(q)
    (2, 1)
    >>> s = q.specialization((x**2 - y**2 - 1)*(1 - y)*(x**2 - 3), [x, y])
    >>> s.degree, s.leading_sign, s.roots
    (3, -1, [(Algebraic([1, 1], -1), 1), (Algebraic([1, -1], 1), 2)])
    """

    __slots__ = ('field', 'theta', 'coords', 'values', '_specializations')

    def __init__(self, field: Optional[AlgebraicField] = None, theta: Optional[Algebraic] = None,
                 coords: Sequence[list[Fraction]] = (), values: Sequence[Algebraic] = ()) -> None:
        if (field is None) != (theta is None) or len(coords) != len(values):
            raise ValueError("a field comes with its generator, a coordinate with its value")
        self.field: Optional[AlgebraicField] = field
        self.theta: Optional[Algebraic] = theta
        self.coords: tuple[list[Fraction], ...] = tuple(coords)
        self.values: tuple[Algebraic, ...] = tuple(values)
        # the polynomials already specialized at the point: the lifting
        # phase, the cylindrical descriptions and the region integrals ask
        # for the roots of the same polynomials over the same point
        self._specializations: dict[tuple[Poly, tuple[Expr, ...]], Specialization] = {}

    def __len__(self) -> int:
        return len(self.coords)

    def __repr__(self) -> str:
        return "SamplePoint(%s)" % ", ".join(repr(v) for v in self.values)

    @property
    def degree(self) -> int:
        """Degree of the field of the coordinates over the rationals."""
        if self.theta is None:
            return 1
        return self.theta.degree

    def extend(self, root: RealLike) -> SamplePoint:
        """The point with the coordinate ``root`` (a real
        :class:`Algebraic`, or a rational number) appended."""
        r = _algebraic(root)
        values = self.values + (r,)
        if not r.is_rational:
            if self.theta is not None and (r is self.theta or (r.minpoly == self.theta.minpoly
                                                               and compare_real(r, self.theta) == 0)):
                return SamplePoint(self.field, self.theta, self.coords + (self.field_alpha(),), values)
            if self.theta is None:
                K = _field_of(r)
                return SamplePoint(K, r, self.coords + (K.alpha,), values)
            gamma, K, theta_K, beta_K = _join(self.theta, r)
            coords = [dup_eval([K.from_rational(c) for c in coord], theta_K, K) if coord else K.zero
                      for coord in self.coords]
            return SamplePoint(K, gamma, coords + [beta_K], values)
        q = r.rational_value()
        return SamplePoint(self.field, self.theta, self.coords + ([q] if q else [],), values)

    def field_alpha(self) -> list[Fraction]:
        """The generator ``theta`` as an element of the field."""
        if self.field is None:
            raise ValueError("the coordinates are rational")
        return self.field.alpha

    def _rational(self) -> list[Fraction]:
        return [c[0] if c else Fraction(0) for c in self.coords]

    def _check(self, poly: PolyLike, gens: Sequence[Expr], extra: int) -> Poly:
        if len(gens) != len(self.coords) + extra:
            raise ValueError("expected %d generators" % (len(self.coords) + extra))
        return as_poly(poly, gens)

    def sign(self, poly: PolyLike, gens: Sequence[Expr]) -> Sign:
        """Exact sign of the polynomial ``poly`` in the generators ``gens``
        at the point. There must be as many generators as coordinates."""
        if not gens:
            if self.coords or isinstance(poly, Poly):
                raise ValueError("expected %d generators" % len(self.coords))
            number = to_expr(poly)
            if not isinstance(number, Rational):
                raise ValueError("a rational number expected, got %s" % (number,))
            return QQ.sign(number.as_fraction())
        f = self._check(poly, gens, 0)
        if self.field is None or self.theta is None:
            value = _evaluate(f.rep.terms, self._rational(), QQ, False)
            return QQ.sign(value[0]) if value else 0
        element = _evaluate(f.rep.terms, self.coords, self.field, False)
        return _sign_in_field(self.theta, element[0]) if element else 0

    def specialization(self, poly: PolyLike, gens: Sequence[Expr]) -> Specialization:
        """The polynomial ``poly`` in the last generator of ``gens``,
        after substituting the point for the other generators: its degree,
        the sign of its leading coefficient and its distinct real roots
        with their multiplicities (a :class:`Specialization`). The result
        is kept: the point does not change."""
        f = self._check(poly, gens, 1)
        key = (f, f.gens)
        found = self._specializations.get(key)
        if found is not None:
            return found
        if self.field is None or self.theta is None:
            rational = _evaluate(f.rep.terms, self._rational(), QQ, True)
            if not rational:
                result = Specialization(-1, 0, [])
            else:
                pairs = _dense_real_roots(rational) if len(rational) > 1 else []
                result = Specialization(len(rational) - 1, QQ.sign(rational[0]), pairs)
        else:
            dense = _evaluate(f.rep.terms, self.coords, self.field, True)
            if not dense:
                result = Specialization(-1, 0, [])
            else:
                pairs = _real_roots_over(self.theta, self.field, dense) if len(dense) > 1 else []
                result = Specialization(len(dense) - 1, _sign_in_field(self.theta, dense[0]), pairs)
        self._specializations[key] = result
        return result

    def real_roots(self, poly: PolyLike, gens: Sequence[Expr]) -> Optional[list[Algebraic]]:
        """Sorted distinct real roots of ``poly`` in the last generator of
        ``gens``, after substituting the point for the other generators.

        Returns ``None`` if the polynomial vanishes identically at the
        point.
        """
        found = self.specialization(poly, gens)
        if found.degree < 0:
            return None
        return [r for r, _ in found.roots]
