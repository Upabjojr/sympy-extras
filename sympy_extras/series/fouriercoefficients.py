r"""Fourier series of piecewise polynomial functions, with the
coefficients in closed form for a symbolic index.

For a function $f$ on $(a, b)$, extended with the period $T = b - a$,
and $\omega = 2\pi/T$,

.. math:: f(x) \sim \frac{a_0}{2} + \sum_{n \ge 1} a_n \cos n \omega x + b_n \sin n \omega x,
          \qquad
          a_n = \frac{2}{T} \int_a^b f(x) \cos n \omega x\, dx, \quad
          b_n = \frac{2}{T} \int_a^b f(x) \sin n \omega x\, dx.

When $f$ is a polynomial on each piece of a subdivision the integrals
are elementary: integrating by parts until the polynomial is
differentiated away,

.. math:: \int p(x) \cos kx\, dx = \sum_{j \ge 0} (-1)^j p^{(j)}(x)\, C_{j+1}(x),

with $C_1 = \sin kx / k$, $C_2 = -\cos kx / k^2$, ... the repeated
antiderivatives of $\cos kx$ [1]_. The index $n$ stays a symbol (a
positive integer), so that the values at the ends simplify ($\sin n\pi =
0$, $\cos n\pi = (-1)^n$) and $a_n$, $b_n$ are returned as formulas in
$n$; the mean $a_0$ is computed separately, the formulas dividing by
$n$.

At a point of discontinuity the series converges to the mean of the
one-sided limits (Dirichlet) [2]_; the partial sums show the Gibbs
overshoot there.

Examples
========

>>> from sympy_extras.core import Symbol, pi
>>> from sympy_extras.functions import piecewise
>>> from sympy_extras.series.fouriercoefficients import fourier_coefficients
>>> x = Symbol('x')
>>> saw = fourier_coefficients(x, x, (-pi, pi))
>>> saw.a0, saw.an, saw.bn
(0, 0, -2*(-1)**n/n)
>>> saw.partial_sum(3)
2*sin(x) - sin(2*x) + 2*sin(3*x)/3
>>> square = fourier_coefficients(piecewise((-1, x < 0), (1, True)), x, (-pi, pi))
>>> square.bn, square.partial_sum(3)
((2 - 2*(-1)**n)/(pi*n), 4*sin(x)/pi + 4*sin(3*x)/(3*pi))
>>> parabola = fourier_coefficients(x**2, x, (-pi, pi))
>>> parabola.a0, parabola.an
(2*pi**2/3, 4*(-1)**n/n**2)

References
==========

.. [1] G. P. Tolstov, *Fourier Series*, Dover, 1976, Chapter 1 (the
       coefficients of piecewise smooth functions) and Chapter 3.
.. [2] A. Zygmund, *Trigonometric Series*, 3rd ed., Cambridge University
       Press, 2002, Chapter II (convergence at a point; Dirichlet and
       Jordan).
"""
from __future__ import annotations

from typing import Final, Optional

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import BooleanTrue
from sympy_extras.core.numbers import Integer, pi
from sympy_extras.core.relational import Le, Lt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import cos, sin
from sympy_extras.functions.elementary.piecewise import Piecewise

__all__ = ['FourierCoefficients', 'fourier_coefficients', 'fourier_partial_sum']


class FourierCoefficients:
    """The series ``a0/2 + sum(an*cos(n*omega*x) + bn*sin(n*omega*x))``:
    ``an`` and ``bn`` are expressions in the index ``n``."""

    __slots__ = ('a0', 'an', 'bn', 'n', 'x', 'omega')

    def __init__(self, a0: Expr, an: Expr, bn: Expr, n: Symbol, x: Symbol, omega: Expr) -> None:
        self.a0: Final[Expr] = a0
        self.an: Final[Expr] = an
        self.bn: Final[Expr] = bn
        self.n: Final[Symbol] = n
        self.x: Final[Symbol] = x
        self.omega: Final[Expr] = omega

    def coefficients(self, k: int) -> tuple[Expr, Expr]:
        """``(a_k, b_k)`` for an integer ``k >= 1``."""
        if k < 1:
            raise ValueError('the index of a harmonic is at least one')
        return self.an.subs(self.n, Integer(k)), self.bn.subs(self.n, Integer(k))

    def term(self, k: int) -> Expr:
        """The harmonic of order ``k`` (the constant ``a0/2`` for
        ``k = 0``)."""
        if k == 0:
            return self.a0/2
        a, b = self.coefficients(k)
        angle = k*self.omega*self.x
        return a*cos(angle) + b*sin(angle)

    def partial_sum(self, order: int) -> Expr:
        """The sum of the harmonics up to ``order``."""
        return add(*[self.term(k) for k in range(order + 1)])

    def __repr__(self) -> str:
        return 'FourierCoefficients(a0=%s, an=%s, bn=%s)' % (self.a0, self.an, self.bn)


def _pieces(f: Expr, x: Symbol, low: Expr, high: Expr) -> list[tuple[Expr, Expr, Expr]]:
    """``(expression, from, to)`` on the consecutive pieces of the
    interval."""
    if not isinstance(f, Piecewise):
        return [(f, low, high)]
    result: list[tuple[Expr, Expr, Expr]] = []
    start = low
    for pair in f.args:
        expression, condition = pair.args[0], pair.args[1]
        if not isinstance(expression, Expr):
            raise TypeError('%s is not an expression' % (expression,))
        if isinstance(condition, BooleanTrue):
            end = high
        elif isinstance(condition, (Lt, Le)) and condition.lhs == x and isinstance(condition.rhs, Expr) \
                and not condition.rhs.has(x):
            end = condition.rhs
        else:
            raise NotImplementedError('the condition %s: the pieces must be given by x < c or x <= c in increasing order'
                                      % (condition,))
        if (end - high).is_positive:
            end = high
        if (end - start).is_positive:
            result.append((expression, start, end))
            start = end
        elif (end - start).is_zero is not True and (end - start).is_negative is not True:
            raise NotImplementedError('the order of the points %s and %s is not known' % (start, end))
        if start == high:
            break
    if start != high:
        raise ValueError('the function is not defined up to %s' % (high,))
    return result


def _definite(p: Expr, x: Symbol, k: Expr, sine: bool, low: Expr, high: Expr) -> Expr:
    """The integral of ``p(x)*cos(k*x)`` (``sin`` with ``sine``) over
    ``(low, high)`` for a polynomial ``p``, by repeated integration by
    parts."""
    polynomial = expand(p)
    derivatives: list[Expr] = []
    current = polynomial
    for _ in range(64):
        if current == 0:
            break
        derivatives.append(current)
        current = expand(current.diff(x))
    else:
        raise NotImplementedError('%s is not a polynomial in %s' % (p, x))
    if any(not d.is_polynomial(x) for d in derivatives):
        raise NotImplementedError('%s is not a polynomial in %s' % (p, x))

    def antiderivative(point: Expr) -> Expr:
        # the repeated antiderivatives of cos(k x): sin/k, -cos/k**2, -sin/k**3, cos/k**4, ...
        # and of sin(k x): -cos/k, -sin/k**2, cos/k**3, sin/k**4, ...
        s, c = sin(k*point), cos(k*point)
        cycle = [s, -c, -s, c] if not sine else [-c, -s, c, s]
        total: list[Expr] = []
        for j, derivative in enumerate(derivatives):
            sign = -1 if j % 2 else 1
            total.append(sign*derivative.subs(x, point)*cycle[j % 4]/k**(j + 1))
        return add(*total)

    return antiderivative(high) - antiderivative(low)


def _polynomial_integral(p: Expr, x: Symbol, low: Expr, high: Expr) -> Expr:
    polynomial = expand(p)
    if not polynomial.is_polynomial(x):
        raise NotImplementedError('%s is not a polynomial in %s' % (p, x))
    terms = polynomial.args if polynomial.is_Add else (polynomial,)
    total: list[Expr] = []
    for term in terms:
        if not isinstance(term, Expr):
            raise TypeError('%s is not an expression' % (term,))
        coefficient, power = term.as_independent(x)
        degree = 0
        while power != 1 and degree < 4096:
            degree += 1
            power = power/x
        primitive = coefficient*x**(degree + 1)/(degree + 1)
        total.append(primitive.subs(x, high) - primitive.subs(x, low))
    return add(*total)


def fourier_coefficients(f: ExprLike, x: Symbol, interval: tuple[ExprLike, ExprLike], n: Optional[Symbol] = None
                         ) -> FourierCoefficients:
    """The Fourier coefficients on ``interval = (a, b)`` (the period is
    ``b - a``) of a polynomial in ``x``, or of a ``Piecewise`` of
    polynomials whose conditions are ``x < c`` or ``x <= c`` in
    increasing order (the last one may be ``True``). ``n`` is the index
    of the formulas (a positive integer symbol named ``n`` by default).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import piecewise
    >>> from sympy_extras.series.fouriercoefficients import fourier_coefficients
    >>> x = Symbol('x')
    >>> triangle = fourier_coefficients(piecewise((x, x < 1), (2 - x, True)), x, (0, 2))
    >>> triangle.a0, triangle.an, triangle.bn
    (1, (2*(-1)**n - 2)/(pi**2*n**2), 0)
    """
    from sympy_extras.simplify.normalform import normal_form
    expression = to_expr(f)
    low, high = to_expr(interval[0]), to_expr(interval[1])
    period = high - low
    if period.is_positive is not True:
        raise ValueError('the interval (%s, %s) is not increasing' % (low, high))
    index = Symbol('n', integer=True, positive=True) if n is None else n
    if index.is_integer is not True or index.is_positive is not True:
        raise ValueError('the index must be a symbol declared a positive integer')
    omega = 2*pi/period
    pieces = _pieces(expression, x, low, high)
    k = index*omega
    mean = add(*[_polynomial_integral(p, x, start, end) for p, start, end in pieces])
    cosine = add(*[_definite(p, x, k, False, start, end) for p, start, end in pieces])
    sine = add(*[_definite(p, x, k, True, start, end) for p, start, end in pieces])
    scale = 2/period
    return FourierCoefficients(normal_form(scale*mean), normal_form(scale*cosine), normal_form(scale*sine), index, x, omega)


def fourier_partial_sum(f: ExprLike, x: Symbol, interval: tuple[ExprLike, ExprLike], order: int) -> Expr:
    """The partial sum up to the harmonic ``order`` of the Fourier
    series of ``f`` on ``interval`` (see :func:`fourier_coefficients`).

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.series.fouriercoefficients import fourier_partial_sum
    >>> x = Symbol('x')
    >>> fourier_partial_sum(x**2, x, (-pi, pi), 2)
    -4*cos(x) + cos(2*x) + pi**2/3
    """
    return fourier_coefficients(f, x, interval).partial_sum(order)
