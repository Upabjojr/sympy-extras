"""Fourier series: the classical expansions, and every coefficient
against the integral computed by SymPy on the pieces."""
from __future__ import annotations

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol, pi
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, piecewise, sin
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.series.fouriercoefficients import fourier_coefficients, fourier_partial_sum

x = Symbol('x')
sx = sympy.Symbol('x')
n = Symbol('n', integer=True, positive=True)


def _check(pieces: list[tuple[Expr, Expr, Expr]], f: Expr, low: Expr, high: Expr) -> None:
    """a_0 and the coefficients of orders 1 to 5 are the integrals over
    the pieces."""
    series = fourier_coefficients(f, x, (low, high))
    period = sympy_expr(to_sympy(high - low))
    omega = 2*sympy.pi/period

    def integral(weight: sympy.Expr) -> sympy.Expr:
        total = sympy.Integer(0)
        for expression, start, end in pieces:
            total += sympy.integrate(sympy_expr(to_sympy(expression))*weight, (sx, to_sympy(start), to_sympy(end)))
        return 2*total/period

    assert sympy.simplify(sympy_expr(to_sympy(series.a0)) - integral(sympy.Integer(1))) == 0
    for k in range(1, 6):
        a, b = series.coefficients(k)
        assert sympy.simplify(sympy_expr(to_sympy(a)) - integral(sympy.cos(k*omega*sx))) == 0
        assert sympy.simplify(sympy_expr(to_sympy(b)) - integral(sympy.sin(k*omega*sx))) == 0


def test_classical_series() -> None:
    # Tolstov, Chapter 1: the sawtooth, the square wave, x**2 and |x| on (-pi, pi)
    saw = fourier_coefficients(x, x, (-pi, pi), n)
    assert saw.a0 == 0 and saw.an == 0 and saw.bn == -2*(-1)**n/n
    square = fourier_coefficients(piecewise((-1, x < 0), (1, True)), x, (-pi, pi), n)
    assert square.an == 0 and [square.coefficients(k)[1] for k in (1, 2, 3)] == [4/pi, 0, 4/(3*pi)]
    parabola = fourier_coefficients(x**2, x, (-pi, pi), n)
    assert parabola.a0 == 2*pi**2/3 and parabola.an == 4*(-1)**n/n**2 and parabola.bn == 0
    absolute = fourier_coefficients(piecewise((-x, x < 0), (x, True)), x, (-pi, pi), n)
    assert absolute.a0 == pi and absolute.bn == 0
    assert [absolute.coefficients(k)[0] for k in (1, 2, 3)] == [-4/pi, 0, -4/(9*pi)]
    # the sum at x = 0 of the series of x**2 gives zeta(2): pi**2/3 + 4 sum (-1)**n/n**2 = 0
    assert fourier_partial_sum(x**2, x, (-pi, pi), 2) == pi**2/3 - 4*cos(x) + cos(2*x)


def test_coefficients_against_integrals() -> None:
    one, two = Integer(1), Integer(2)
    _check([(x**3 - x, -one, one)], x**3 - x, -one, one)
    _check([(x**2 + 1, Integer(0), Integer(3))], x**2 + 1, Integer(0), Integer(3))
    f = piecewise((x, x < 1), (2 - x, True))
    _check([(x, Integer(0), one), (2 - x, one, two)], f, Integer(0), two)
    g = piecewise((Integer(0), x <= -1), (x**2, x < Rational(1, 2)), (Integer(3), True))
    _check([(Integer(0), -two, -one), (x**2, -one, Rational(1, 2)), (Integer(3), Rational(1, 2), two)], g, -two, two)
    # the interval may stop before the last breakpoint
    _check([(x, Integer(0), one), (2 - x, one, Rational(3, 2))], f, Integer(0), Rational(3, 2))
    _check([(x*(pi - x), Integer(0), pi)], x*(pi - x), Integer(0), pi)


def test_partial_sums_and_rejections() -> None:
    series = fourier_coefficients(x, x, (-pi, pi))
    assert series.term(0) == 0 and series.term(2) == -sin(2*x)
    assert series.partial_sum(3) == 2*sin(x) - sin(2*x) + 2*sin(3*x)/3
    with pytest.raises(ValueError):
        series.coefficients(0)
    with pytest.raises(ValueError):
        fourier_coefficients(x, x, (1, 0))
    with pytest.raises(ValueError):
        fourier_coefficients(x, x, (0, 1), Symbol('m'))
    with pytest.raises(NotImplementedError):
        fourier_coefficients(sin(x**2), x, (0, 1))
    with pytest.raises(NotImplementedError):
        fourier_coefficients(piecewise((x, x > 0), (Integer(0), True)), x, (-1, 1))
    with pytest.raises(ValueError):
        fourier_coefficients(piecewise((x, x < 0), (Integer(1), x < 1)), x, (-2, 2))
