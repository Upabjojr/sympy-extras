"""Integrals over a period with values from the tables (Gradshteyn and
Ryzhik, 7th ed., 3.915 and 3.937; Maxima's regression file rtestint) and
from the generating function of the modified Bessel functions, each
checked numerically as well."""
from __future__ import annotations

from sympy import symbols, exp, cos, sin, pi, besseli, besselj, sqrt, simplify, Rational

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.periodic import mean_value_integral

x = symbols('x')
a = symbols('a', positive=True)


def _check(f: ExprLike, expected: ExprLike, lo: ExprLike = 0, hi: ExprLike = 2 * pi) -> None:
    f_, expected_, lo_, hi_ = as_expr(f), as_expr(expected), as_expr(lo), as_expr(hi)
    found = mean_value_integral(f_, x, lo_, hi_)
    assert found is not None, f_
    difference = as_expr(found.value - expected_)
    assert simplify(difference) == 0 or abs(complex(difference.evalf(20, subs={a: 1.7}))) < 1e-15, found
    assert verify_numerically(expected_, f_, x, lo_, hi_) is not False


def test_gradshteyn_ryzhik() -> None:
    # GR 3.915.1 and the generating function exp(a cos x) = sum I_n(a) e^{i n x}:
    # Integral(exp(a cos x) cos(n x), (x, 0, 2 pi)) = 2 pi I_n(a)
    _check(exp(a * cos(x)), 2 * pi * besseli(0, a))
    _check(exp(a * cos(x)) * cos(x), 2 * pi * besseli(1, a))
    _check(exp(a * cos(x)) * cos(2 * x), 2 * pi * besseli(2, a))
    _check(exp(a * cos(x)) * cos(3 * x), 2 * pi * besseli(3, a))
    _check(exp(a * cos(x)) * sin(x), 0)
    # GR 3.937.1-type: exp(a cos x + b sin x) over a period is 2 pi I_0(sqrt(a^2 + b^2))
    _check(exp(cos(x) + sin(x)), 2 * pi * besseli(0, sqrt(2)))
    _check(exp(2 * cos(x) + sin(x)), 2 * pi * besseli(0, sqrt(5)))


def test_maxima_and_classical() -> None:
    # Maxima's rtestint: Integral(exp(cos x) cos(sin x), (x, 0, 2 pi)) = 2 pi
    _check(exp(cos(x)) * cos(sin(x)), 2 * pi)
    _check(exp(cos(x)) * sin(sin(x)), 0)
    # Bessel's integral: Integral(cos(cos x), (x, 0, 2 pi)) = 2 pi J_0(1)
    _check(cos(cos(x)), 2 * pi * besselj(0, 1))
    _check(cos(sin(x)), 2 * pi * besselj(0, 1))
    # Wallis: Integral(cos(x)^4, (x, 0, 2 pi)) = 3 pi/4; Integral(sin(x)^2 cos(x)^2) = pi/4
    _check(cos(x)**4, 3 * pi / 4)
    _check(sin(x)**2 * cos(x)**2, pi / 4)
    # two periods
    _check(exp(cos(x)), 4 * pi * besseli(0, 1), 0, 4 * pi)
    _check(cos(x)**2, Rational(3, 1) * pi, -pi, 5 * pi)
