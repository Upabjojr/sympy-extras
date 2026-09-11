"""Algebraic integrals with known values (Gradshteyn–Ryzhik 2.26–2.28 and
textbook exercises), each also checked numerically."""
from __future__ import annotations

from sympy import symbols, sqrt, oo, pi, log, simplify, asinh, Rational

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals import definite_integral
from sympy_extras.integrals.algebraic import algebraic_integral
from sympy_extras.integrals.definite import verify_numerically

x = symbols('x')


def _check(f: ExprLike, lo: ExprLike, hi: ExprLike, expected: ExprLike) -> None:
    found = algebraic_integral(as_expr(f), x, as_expr(lo), as_expr(hi))
    assert found is not None, f
    difference = as_expr(found.value) - as_expr(expected)
    assert simplify(difference) == 0 or abs(complex(difference.evalf(20))) < 1e-15, (found.value, expected)
    assert verify_numerically(as_expr(expected), as_expr(f), x, as_expr(lo), as_expr(hi)) is not False


def test_quadratic_radicands() -> None:
    # GR 2.261: Integral(dx/sqrt(x^2 + 1)) = asinh(x)
    _check(1 / sqrt(x**2 + 1), 0, 1, asinh(1))
    # GR 2.262.1: Integral(sqrt(x^2 + 1)) = (x sqrt(x^2+1) + asinh(x))/2
    _check(sqrt(x**2 + 1), 0, 1, (sqrt(2) + asinh(1)) / 2)
    # GR 2.266: Integral(dx/(x sqrt(x^2 - 1))) = asec(x): pi/3 between 1 and 2
    _check(1 / (x * sqrt(x**2 - 1)), 1, 2, pi / 3)
    # Integral(x/sqrt(1 - x^2), (x, 0, 1)) = 1
    _check(x / sqrt(1 - x**2), 0, 1, 1)
    # Integral(x^2 sqrt(1 - x^2), (x, 0, 1)) = pi/16 (Beta)
    _check(x**2 * sqrt(1 - x**2), 0, 1, pi / 16)
    # GR 2.281-type: Integral(dx/((x + 1) sqrt(x^2 + x + 1)), (x, 0, oo)) = log(3)
    _check(1 / ((x + 1) * sqrt(x**2 + x + 1)), 0, oo, log(3))


def test_roots_of_linear_and_moebius_functions() -> None:
    # Integral(sqrt(x)/(1 + x)^2, (x, 0, oo)) = pi/2 (Beta(3/2, 1/2))
    _check(sqrt(x) / (1 + x)**2, 0, oo, pi / 2)
    # Integral(x^(1/3)/(1 + x), (x, 0, 1)) = 3 - log(2) - sqrt(3) pi/3
    _check(x**Rational(1, 3) / (1 + x), 0, 1, 3 - log(2) - sqrt(3) * pi / 3)
    # Integral(sqrt((1 - x)/(1 + x)), (x, 0, 1)) = pi/2 - 1
    _check(sqrt((1 - x) / (1 + x)), 0, 1, pi / 2 - 1)
    # Integral(1/(1 + sqrt(x)), (x, 0, 1)) = 2 - 2 log 2
    _check(1 / (1 + sqrt(x)), 0, 1, 2 - 2 * log(2))


def test_through_the_driver() -> None:
    value = definite_integral(1 / (x * sqrt(x**2 - 1)), (x, 1, 2))
    assert simplify(value - pi / 3) == 0
    assert simplify(definite_integral(sqrt((1 - x) / (1 + x)), (x, 0, 1)) - (pi / 2 - 1)) == 0
