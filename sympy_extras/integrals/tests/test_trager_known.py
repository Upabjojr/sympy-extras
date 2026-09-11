"""Definite integrals through Trager's antiderivatives against
quadrature, and the classical examples of the literature."""
from __future__ import annotations

import mpmath

from sympy import symbols, sqrt, N, log, Rational, asinh

from sympy_extras.integrals.trager import trager_antiderivative

x = symbols('x')


def _check(f: object, lower: float, upper: float) -> None:
    from sympy_extras._typing import as_expr
    f_ = as_expr(f)
    F = trager_antiderivative(f_, x)
    assert F is not None, f_
    value = complex(N(F.subs(x, upper) - F.subs(x, lower), 20))
    expected = mpmath.quad(lambda t: complex(N(f_.subs(x, t), 20)), [lower, upper])
    assert abs(value - complex(expected)) < 1e-12, (f_, value, expected)


def test_definite_values() -> None:
    _check(x / sqrt(x**4 + 1), 0, 2)
    _check(1 / (x * sqrt(x**2 + 1)), 1, 3)
    _check((x**2 - 1) / ((x**2 + 1) * sqrt(x**4 + 1)), Rational(1, 2), 3)
    _check(1 / ((x**2 + 1) * sqrt(x**2 + 2)), -1, 2)
    _check(1 / (x * sqrt(x**3 + 1)), 1, 4)
    _check(sqrt(x**2 + 1) / x, 1, 2)
    _check(x**2 / sqrt(x**2 + 1), 0, 1)


def test_classical_forms() -> None:
    # Gradshteyn-Ryzhik 2.271.4 and 2.275: the standard logarithmic forms
    F = trager_antiderivative(x / sqrt(x**4 + 1), x)
    assert F is not None
    difference = F - asinh(x**2) / 2
    assert abs(float(N(difference.subs(x, 2) - difference.subs(x, 1), 20))) < 1e-15
    F = trager_antiderivative(1 / sqrt(x**2 - 1), x)
    assert F == log(x + sqrt(x**2 - 1))
    # Integral(1/(x sqrt(x^2 + 1)), (x, 1, oo)) = log(1 + sqrt(2))
    F = trager_antiderivative(1 / (x * sqrt(x**2 + 1)), x)
    assert F is not None
    assert abs(float(N(-F.subs(x, 1) - log(1 + sqrt(2)), 20))) < 1e-15
    assert abs(float(N(F.subs(x, 10**8), 20))) < 1e-7
