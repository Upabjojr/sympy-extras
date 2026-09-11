"""Integrals evaluated by series against the tables (Gradshteyn–Ryzhik,
7th ed., [GR]) and the classical Euler-sum values, each also checked
numerically."""
from __future__ import annotations

from sympy import symbols, log, exp, oo, pi, S, atan, Catalan, polygamma, zeta, simplify

from sympy_extras._typing import as_expr
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.series import series_integral

x = symbols('x')
p = symbols('p', positive=True)


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
