"""Integrals evaluated by series against the tables (Gradshteyn–Ryzhik,
7th ed., [GR]) and the classical Euler-sum values, each also checked
numerically."""
from __future__ import annotations

from sympy import symbols, log, exp, oo, pi, S, atan, Catalan, polygamma, zeta, simplify, sin, cos, tan, I, polylog

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.series import series_integral

x = symbols('x')
p = symbols('p', positive=True)
n = symbols('n', integer=True, positive=True)
t = symbols('t', positive=True)


def _check(f: object, b: object, expected: object) -> None:
    found = series_integral(as_expr(f), x, S.Zero, as_expr(b))
    assert found is not None, f
    assert simplify(found.value - as_expr(expected)) == 0, (found, expected)
    assert verify_numerically(as_expr(expected), as_expr(f), x, S.Zero, as_expr(b)) is not False


def test_logarithmic_integrals() -> None:
    # GR 4.291.1: Integral(log(1 - x)/x, (x, 0, 1)) = -pi^2/6
    _check(log(1 - x) / x, 1, -pi**2 / 6)
    # GR 4.291.2: Integral(log(1 + x)/x, (x, 0, 1)) = pi^2/12
    _check(log(1 + x) / x, 1, pi**2 / 12)
    # GR 4.231.1: Integral(log(x)/(1 + x), (x, 0, 1)) = -pi^2/12
    _check(log(x) / (1 + x), 1, -pi**2 / 12)
    # GR 4.221.1? Integral(log(x) log(1 - x), (x, 0, 1)) = 2 - pi^2/6 (Euler)
    _check(log(x) * log(1 - x), 1, 2 - pi**2 / 6)
    # GR 4.251.4: Integral(x^p log(x)/(1 - x), (x, 0, 1)) = -psi'(p + 1)
    found = series_integral(x**p * log(x) / (1 - x), x, S.Zero, S.One)
    assert found is not None and simplify(found.value + polygamma(1, p + 1)) == 0


def test_exponential_and_catalan() -> None:
    # GR 3.411.1: Integral(x^(n-1)/(e^x - 1), (x, 0, oo)) = Gamma(n) zeta(n)
    _check(x / (exp(x) - 1), oo, pi**2 / 6)
    _check(x**3 / (exp(x) - 1), oo, pi**4 / 15)
    # GR 3.411.3: Integral(x/(e^x + 1), (x, 0, oo)) = pi^2/12
    _check(x / (exp(x) + 1), oo, pi**2 / 12)
    _check(x**2 * exp(-x) / (exp(x) - 1), oo, 2 * zeta(3) - 2)
    # Catalan's constant (GR 4.531.1)
    _check(atan(x) / x, 1, Catalan)


def _check_range(f: ExprLike, a: ExprLike, b: ExprLike, expected: ExprLike,
                 assumptions: Assumptions = None) -> None:
    found = series_integral(as_expr(f), x, as_expr(a), as_expr(b), assumptions)
    assert found is not None, f
    assert simplify(found.value - as_expr(expected)) == 0, (f, found.value)
    assert verify_numerically(found.value, as_expr(f), x, as_expr(a), as_expr(b), assumptions) is not False


def test_fourier_log_sine_integrals() -> None:
    # GR 4.224.3, 4.224.6, 4.224.7: the log-sine integrals
    _check_range(log(sin(x)), 0, pi, -pi * log(2))
    _check_range(log(sin(x)), 0, pi / 2, -pi * log(2) / 2)
    _check_range(log(1 + cos(x)), 0, pi, -pi * log(2))
    _check_range(log(tan(x)), 0, pi / 2, 0)
    _check_range(log(2 * cos(x / 2)), -pi, pi, 0)
    # GR 4.224.9, 4.224.11 and [BorweinStraub]: moments of log(sin x)
    _check_range(x * log(sin(x)), 0, pi, -pi**2 * log(2) / 2)
    _check_range(x * log(sin(x)), 0, pi / 2, -pi**2 * log(2) / 8 + 7 * zeta(3) / 16)
    _check_range(x**2 * log(sin(x)), 0, pi, -pi**3 * log(2) / 3 - pi * zeta(3) / 2)
    _check_range(x**2 * log(2 * sin(x / 2)), 0, 2 * pi, -4 * pi * zeta(3))
    # GR 4.225.1: the square and the product
    _check_range(log(sin(x))**2, 0, pi, pi * log(2)**2 + pi**3 / 12)
    _check_range(log(sin(x)) * log(cos(x)), 0, pi / 2, pi * log(2)**2 / 2 - pi**3 / 48)
    # GR 4.384.3: the Fourier coefficients themselves
    _check_range(log(2 * sin(x / 2)) * cos(n * x), 0, 2 * pi, -pi / n)
    # GR 4.226.1
    _check_range(sin(x) * log(sin(x)), 0, pi, 2 * log(2) - 2)
    # the Clausen function Cl_2(t) = Sum(sin(k t)/k**2) as polylogarithms
    _check_range(log(2 * sin(x / 2)), 0, t, I * (polylog(2, exp(I * t)) - polylog(2, exp(-I * t))) / 2, [t < 6])
