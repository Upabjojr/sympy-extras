r"""Rational approximation and interpolation: Padé approximants, rational
(Cauchy) interpolation, and the reversion of power series.

**Padé approximants.** The $[m/n]$ approximant of a power series $f$ at
$x_0$ is the rational function $p/q$ with $\deg p \le m$, $\deg q \le n$
whose expansion agrees with $f$ as far as possible:

.. math:: q(x) f(x) - p(x) = O\bigl((x - x_0)^{m + n + 1}\bigr).

It is computed by the extended Euclidean algorithm [1]_ [2]_: with $N =
m + n + 1$ and $c$ the Taylor polynomial of $f$ of degree below $N$, the
remainder sequence of $(x^N, c)$ has, at the first remainder $r_j$ of
degree at most $m$, the cofactor $t_j$ with $r_j \equiv t_j c \pmod{x^N}$
and $\deg t_j \le n$; $p/q = r_j/t_j$ in lowest terms. This needs no
linear system and works over any field, so that the coefficients may
contain parameters. The rational function so defined always exists and is
unique; when $r_j$ and $t_j$ have the common factor $x$ its contact with
$f$ is of lower order than $m + n + 1$ (the approximant in the sense of
Baker does not exist: ``strict=True`` raises ``ValueError`` then).

**Rational interpolation.** The same algorithm applied to $\prod (x -
x_i)$ and the interpolating polynomial gives the rational function
$r/t$ with $\deg r \le k$, $\deg t \le N - 1 - k$ through $N$ points
(Cauchy's interpolation problem [2]_); when $r_j$ and $t_j$ have a common
root $x_i$ that point is *unattainable* and there is no interpolant with
these degrees.

**Reversion.** For $f(x) = y_0 + a_1 (x - x_0) + \dots$ with $a_1 \ne 0$
the inverse function has the expansion $x = x_0 + \sum_k b_k (y -
y_0)^k$ with, by Lagrange's inversion formula [3]_,

.. math:: b_k = \frac{1}{k}\, [h^{k-1}]\,
          \Bigl(\frac{h}{f(x_0 + h) - y_0}\Bigr)^{k}.

The coefficients are taken in the field of the rational functions of
the symbols and of the other kernels they contain (``exp(1)``, ``pi``,
``sin(a)``); a leading coefficient with kernels which are not symbols is
submitted to the zero test of :mod:`sympy_extras.simplify.structure`
before a division by it.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import exp, log, tan
>>> from sympy_extras.series.approximants import pade, rational_interpolate, series_reversion
>>> x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
>>> pade(exp(x), x, 2, 2)
(x**2 + 6*x + 12)/(x**2 - 6*x + 12)
>>> pade(log(1 + x), x, 2, 1), pade(tan(x), x, 3, 2)
((x**2 + 6*x)/(4*x + 6), (x**3 - 15*x)/(6*x**2 - 15))
>>> pade(exp(a*x), x, 1, 1)
(-a*x - 2)/(a*x - 2)
>>> rational_interpolate([(0, 1), (1, 2), (2, 5), (3, 10)], x, 2)
x**2 + 1
>>> rational_interpolate([(0, 1), (1, 1), (2, 2)], x, 1) is None
True
>>> series_reversion(x*exp(x), x, y, 5)
-8*y**4/3 + 3*y**3/2 - y**2 + y

References
==========

.. [1] G. A. Baker, P. Graves-Morris, *Padé Approximants*, 2nd ed.,
       Cambridge University Press, 1996, Sections 1.1 and 1.4 (the two
       definitions, existence).
.. [2] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
       Cambridge University Press, 2013, Sections 5.8 (Cauchy
       interpolation) and 5.9 (Padé approximation by the extended
       Euclidean algorithm).
.. [3] P. Henrici, *Applied and Computational Complex Analysis*, vol. 1,
       Wiley, 1974, Section 1.9 (the Lagrange-Bürmann formula).
"""
from __future__ import annotations

from typing import Final, Generic, Optional, Sequence, TypeVar

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.matrices.eigen import _Field, _FunctionField, _GaussianField, _RationalField, _coefficients
from sympy_extras.matrices.linalg import _PolynomialLift, _complex_lift, _kernels, normal
from sympy_extras.polys import densearith, euclidtools
from sympy_extras.polys.densetools import dup_eval

__all__ = ['taylor_coefficients', 'pade', 'pade_from_coefficients', 'rational_interpolate', 'series_reversion',
           'revert_coefficients']

T = TypeVar('T')


class _Approximation:
    """The interface which does not depend on the type of the elements
    of the field."""

    def pade(self, coefficients: Sequence[Expr], m: int, n: int, strict: bool) -> tuple[list[Expr], list[Expr]]:
        raise NotImplementedError

    def interpolate(self, nodes: Sequence[Expr], values: Sequence[Expr], k: int) -> Optional[tuple[list[Expr], list[Expr]]]:
        raise NotImplementedError

    def revert(self, coefficients: Sequence[Expr]) -> list[Expr]:
        raise NotImplementedError


class _Worker(_Approximation, Generic[T]):
    """Polynomials are dense lists over the field, highest degree
    first."""

    def __init__(self, field: _Field[T], exact: bool) -> None:
        self.field: Final[_Field[T]] = field
        self.exact: Final[bool] = exact

    def _nonzero(self, f: list[T]) -> list[T]:
        """The polynomial, its leading coefficient proved nonzero as a
        quantity."""
        if f and not self.exact:
            from sympy_extras.simplify.structure import is_zero
            e = self.field.expr(f[0])
            if not (isinstance(e, Rational) and e.p != 0) and is_zero(e) is not False:
                raise NotImplementedError('the coefficient %s is not proved nonzero: simplify the coefficients first' % (e,))
        return f

    def _dense(self, ascending: Sequence[Expr]) -> list[T]:
        K = self.field.domain
        f = [self.field.element(c) for c in reversed(ascending)]
        while f and K.is_zero(f[0]):
            del f[0]
        return f

    def _ascending(self, f: list[T]) -> list[Expr]:
        return [self.field.expr(c) for c in reversed(f)]

    def _euclid(self, modulus: list[T], g: list[T], bound: int) -> tuple[list[T], list[T], list[T]]:
        """``(r, t, cofactor)``: ``cofactor*g`` is modulo ``modulus`` the
        first remainder of degree at most ``bound``, and ``r/t`` is that
        quotient in lowest terms."""
        K = self.field.domain
        r0, r1 = modulus, g
        t0: list[T] = []
        t1: list[T] = [K.one]
        while len(r1) - 1 > bound:
            quotient, remainder = densearith.dup_div(r0, self._nonzero(r1), K)
            r0, r1 = r1, remainder
            t0, t1 = t1, densearith.dup_sub(t0, densearith.dup_mul(quotient, t1, K), K)
        cofactor = t1
        if r1:
            common = euclidtools.dup_gcd(self._nonzero(r1), t1, K)
            if len(common) > 1:
                r1, t1 = densearith.dup_exquo(r1, common, K), densearith.dup_exquo(t1, common, K)
        else:
            t1 = [K.one]
        return r1, t1, cofactor

    def _normalised(self, r: list[T], t: list[T]) -> tuple[list[Expr], list[Expr]]:
        """The denominator with constant term one (or leading coefficient
        one when it vanishes at the origin)."""
        K = self.field.domain
        scale = t[-1] if not K.is_zero(t[-1]) else t[0]
        if not K.is_one(scale):
            inverse = K.invert(scale)
            r, t = densearith.dup_mul_ground(r, inverse, K), densearith.dup_mul_ground(t, inverse, K)
        return self._ascending(r), self._ascending(t)

    def pade(self, coefficients: Sequence[Expr], m: int, n: int, strict: bool) -> tuple[list[Expr], list[Expr]]:
        K = self.field.domain
        N = m + n + 1
        modulus = [K.one] + [K.zero]*N
        r, t, cofactor = self._euclid(modulus, self._dense(coefficients[:N]), m)
        # q*f - p = O(x**N) holds for the cofactor: dividing by a power of x loses as many orders
        if strict and K.is_zero(cofactor[-1]):
            raise ValueError('the [%d/%d] Pade approximant does not exist' % (m, n))
        return self._normalised(r, t)

    def interpolate(self, nodes: Sequence[Expr], values: Sequence[Expr], k: int) -> Optional[tuple[list[Expr], list[Expr]]]:
        K = self.field.domain
        points = [self.field.element(e) for e in nodes]
        ordinates = [self.field.element(e) for e in values]
        if len(set(points)) != len(points):
            raise ValueError('the points of interpolation are not distinct')
        # Newton's interpolating polynomial, with the product of the (x - x_i) built on the way
        modulus: list[T] = [K.one]
        g: list[T] = []
        for point, value in zip(points, ordinates):
            difference = K.sub(value, dup_eval(g, point, K))
            if not K.is_zero(difference):
                weight = K.exquo(difference, dup_eval(modulus, point, K))
                g = densearith.dup_add(g, densearith.dup_mul_ground(modulus, weight, K), K)
            modulus = densearith.dup_mul(modulus, [K.one, K.neg(point)], K)
        r, t, _ = self._euclid(modulus, g, k)
        for point, value in zip(points, ordinates):
            denominator = dup_eval(t, point, K)
            if K.is_zero(denominator) or K.mul(denominator, value) != dup_eval(r, point, K):
                return None
        return self._normalised(r, t)

    def revert(self, coefficients: Sequence[Expr]) -> list[Expr]:
        K = self.field.domain
        n = len(coefficients)
        series = [self.field.element(c) for c in coefficients]           # ascending, series[0] is ignored
        if n < 2 or K.is_zero(series[1]):
            raise ValueError('the series has no inverse: the coefficient of the first power is zero')
        self._nonzero([series[1]])

        def product(f: Sequence[T], g: Sequence[T]) -> list[T]:
            out = [K.zero]*(n - 1)
            for i, a in enumerate(f):
                if K.is_zero(a):
                    continue
                for j in range(n - 1 - i):
                    if not K.is_zero(g[j]):
                        out[i + j] = K.add(out[i + j], K.mul(a, g[j]))
            return out

        # phi = h/(a1 h + a2 h**2 + ...) modulo h**(n-1), by the recurrence of the reciprocal
        inverse = K.invert(series[1])
        phi: list[T] = []
        for k in range(n - 1):
            total = K.one if k == 0 else K.zero
            for j in range(1, k + 1):
                total = K.sub(total, K.mul(series[j + 1], phi[k - j]))
            phi.append(K.mul(total, inverse))
        result: list[Expr] = [Integer(0)]
        power = list(phi)
        for k in range(1, n):
            # b_k = [h**(k-1)] phi**k / k
            result.append(self.field.expr(K.exquo(power[k - 1], K.from_int(k))))
            if k + 1 < n:
                power = product(power, phi)
        return result


def _approximation(coefficients: Sequence[Expr]) -> _Approximation:
    kernels = _kernels([ExprMatrix(1, len(coefficients), coefficients)])
    gaussian = _complex_lift(kernels)
    if gaussian is not None:
        return _Worker(_GaussianField(gaussian), True)
    if kernels:
        lift = _PolynomialLift(kernels)
        return _Worker(_FunctionField(lift), lift.exact)
    return _Worker(_RationalField(), True)


def _polynomial(ascending: Sequence[Expr], h: Expr) -> Expr:
    return add(*[mul(c, pow(h, Integer(k))) for k, c in enumerate(ascending)])


def _quotient(numerator: Sequence[Expr], denominator: Sequence[Expr], h: Expr) -> Expr:
    """The quotient of two polynomials in ``h`` (ascending coefficients),
    without fractions inside."""
    return normal(_polynomial(numerator, h)/_polynomial(denominator, h))


def taylor_coefficients(f: ExprLike, x: Symbol, n: int, x0: ExprLike = 0) -> list[Expr]:
    """The coefficients of ``(x - x0)**k`` for ``k = 0, ..., n - 1`` in
    the expansion of ``f`` at ``x0``; ``ValueError`` when the expansion is
    not a power series (a pole, a logarithm, a fractional power).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import cos, log
    >>> from sympy_extras.series.approximants import taylor_coefficients
    >>> x = Symbol('x')
    >>> taylor_coefficients(cos(x), x, 5), taylor_coefficients(log(x), x, 3, 1)
    ([1, 0, -1/2, 0, 1/24], [0, 1, -1/2])
    """
    expression, point = to_expr(f), to_expr(x0)
    if n <= 0:
        return []
    shifted = expression.subs(x, x + point) if point != 0 else expression
    truncated = expand(shifted.series(x, 0, n).removeO())
    coefficients = [Integer(0)] if truncated == 0 else _coefficients(truncated, x)
    if any(c.has(x) for c in coefficients):
        raise ValueError('the expansion of %s at %s is not a power series' % (expression, point))
    ascending = list(reversed(coefficients))[:n]
    zero: Expr = Integer(0)
    return ascending + [zero]*(n - len(ascending))


def pade_from_coefficients(coefficients: Sequence[ExprLike], x: Expr, m: int, n: int, strict: bool = False) -> Expr:
    """The ``[m/n]`` Padé approximant, in ``x``, of the power series with
    the given coefficients (of ``x**0``, ``x**1``, ...; at least ``m + n
    + 1`` of them).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.series.approximants import pade_from_coefficients
    >>> x = Symbol('x')
    >>> pade_from_coefficients([1, 1, 2, 3, 5, 8, 13], x, 1, 2)        # the Fibonacci numbers
    -1/(x**2 + x - 1)
    """
    if m < 0 or n < 0:
        raise ValueError('the degrees of a Pade approximant are not negative')
    known = [to_expr(c) for c in coefficients]
    if len(known) < m + n + 1:
        raise ValueError('%d coefficients are needed for the [%d/%d] approximant' % (m + n + 1, m, n))
    numerator, denominator = _approximation(known).pade(known, m, n, strict)
    return _quotient(numerator, denominator, x)


def pade(f: ExprLike, x: Symbol, m: int, n: int, x0: ExprLike = 0, strict: bool = False) -> Expr:
    """The ``[m/n]`` Padé approximant of ``f`` at ``x = x0``: the
    rational function ``p/q`` with numerator of degree at most ``m`` and
    denominator of degree at most ``n`` such that ``q*f - p`` vanishes to
    the order ``m + n + 1`` at ``x0``. With ``strict``, ``ValueError``
    when no such ``p/q`` has ``f - p/q`` vanishing to that order.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import cos, sqrt
    >>> from sympy_extras.series.approximants import pade
    >>> x = Symbol('x')
    >>> pade(cos(x), x, 2, 2), pade(sqrt(x), x, 1, 1, 1)
    ((12 - 5*x**2)/(x**2 + 12), (3*x + 1)/(x + 3))
    >>> pade(1 + x**2, x, 1, 1)
    1
    """
    if m < 0 or n < 0:
        raise ValueError('the degrees of a Pade approximant are not negative')
    point = to_expr(x0)
    coefficients = taylor_coefficients(f, x, m + n + 1, point)
    numerator, denominator = _approximation(coefficients).pade(coefficients, m, n, strict)
    return _quotient(numerator, denominator, x - point)


def rational_interpolate(points: Sequence[tuple[ExprLike, ExprLike]], x: Expr, degree: int) -> Optional[Expr]:
    """The rational function of ``x`` through the ``N`` points ``(x_i,
    y_i)`` with numerator of degree at most ``degree`` and denominator of
    degree at most ``N - 1 - degree``, or ``None`` when there is none
    (unattainable points). With ``degree = N - 1`` this is the
    interpolating polynomial.

    >>> from sympy_extras.core import Rational, Symbol
    >>> from sympy_extras.series.approximants import rational_interpolate
    >>> x = Symbol('x')
    >>> rational_interpolate([(1, 1), (2, Rational(1, 2)), (3, Rational(1, 3))], x, 0)
    1/x
    >>> rational_interpolate([(0, 0), (1, Rational(1, 2)), (2, Rational(2, 3)), (3, Rational(3, 4))], x, 1)
    x/(x + 1)
    """
    count = len(points)
    if not 0 <= degree < max(count, 1):
        raise ValueError('the degree of the numerator is between 0 and %d' % (count - 1))
    nodes = [to_expr(p) for p, _ in points]
    values = [to_expr(v) for _, v in points]
    found = _approximation(nodes + values).interpolate(nodes, values, degree)
    if found is None:
        return None
    return _quotient(found[0], found[1], x)


def revert_coefficients(coefficients: Sequence[ExprLike]) -> list[Expr]:
    """The coefficients ``[0, b_1, ..., b_(n-1)]`` of the inverse series
    of ``a_1 h + a_2 h**2 + ...`` given by ``[a_0, a_1, ..., a_(n-1)]``
    (the constant term is ignored: it is the point of expansion of the
    inverse).

    >>> from sympy_extras.series.approximants import revert_coefficients
    >>> revert_coefficients([0, 1, 1, 1, 1, 1])              # h/(1 - h) has the inverse y/(1 + y)
    [0, 1, -1, 1, -1, 1]
    """
    known = [to_expr(c) for c in coefficients]
    return _approximation(known).revert(known)


def series_reversion(f: ExprLike, x: Symbol, y: Expr, n: int = 6, x0: ExprLike = 0) -> Expr:
    """The expansion of the inverse function of ``f`` at ``y0 = f(x0)``:
    the polynomial ``g`` in ``y`` with ``g(f(x)) = x + O((x - x0)**n)``.
    ``ValueError`` when the derivative of ``f`` at ``x0`` is zero.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.series.approximants import series_reversion
    >>> x, y = Symbol('x'), Symbol('y')
    >>> series_reversion(sin(x), x, y, 6)                        # the series of asin
    3*y**5/40 + y**3/6 + y
    >>> series_reversion(exp(x), x, y, 4)                        # the series of log at 1
    y + (y - 1)**3/3 - (y - 1)**2/2 - 1
    """
    point = to_expr(x0)
    coefficients = taylor_coefficients(f, x, n, point)
    if n < 2:
        return point
    reverted = _approximation(coefficients).revert(coefficients)
    return add(point, _polynomial(reverted, y - coefficients[0]))
