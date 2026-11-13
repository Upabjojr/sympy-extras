"""Tests of the native recognition of constants
(:mod:`sympy_extras.integrals.recognize` on the objects of the core); the
old API on SymPy objects is tested in test_recognize.py."""
from __future__ import annotations

import mpmath

from sympy_extras.core import Float, Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.functions import atan, exp, log, zeta
from sympy_extras.integrals.recognize import _Integrand, quadrature, recognize_constant, recognize_integral
from sympy_extras.settings import configure

x = Symbol('x')


def test_constants() -> None:
    assert recognize_constant(Float('1.6449340668482264364724151666460251892', 38)) == pi**2 / 6
    assert recognize_constant(pi**2 / 6 + zeta(3) / 7) == pi**2 / 6 + zeta(3) / 7
    assert recognize_constant(Rational(3, 7)) == Rational(3, 7)
    assert recognize_constant('1.0986122886681096913952452369225257046') == log(3)
    assert recognize_constant(0.5) is None
    assert recognize_constant(Float('0.123456789012345678901234567890', 30)) is None
    assert recognize_constant(pi * Symbol('a')) is None
    with configure(numerical_checks=False):
        assert recognize_constant(pi**2 / 6) is None


def test_integrals_and_quadrature() -> None:
    assert recognize_integral(1 / (x**3 + 1), (x, 0, 1)) == log(2) / 3 + sqrt(3) * pi / 9
    assert recognize_integral(atan(x) / x, (x, 0, 1)) == S.Catalan
    assert recognize_integral(x * exp(-x), (x, 0, oo)) == 1
    assert recognize_integral(exp(-Symbol('a') * x), (x, 0, oo)) is None
    value = quadrature(exp(-x**2), x, -oo, oo, 30)
    assert value is not None
    with mpmath.workdps(30):
        assert abs(value - mpmath.sqrt(mpmath.pi)) < mpmath.mpf(10)**-25


def test_the_integrand_is_evaluated_by_the_rules_of_the_nodes() -> None:
    g = _Integrand(sqrt(x) * exp(-x) + pi, x)
    with mpmath.workdps(20):
        assert abs(g(mpmath.mpf(2)) - (mpmath.sqrt(2) * mpmath.exp(-2) + mpmath.pi)) < mpmath.mpf(10)**-18
        # a negative base under a square root is complex, as with lambdify
        assert mpmath.im(_Integrand(sqrt(x), x)(mpmath.mpf(-4))) == 2
    assert quadrature(Integer(1) / x, x, Integer(-1), Integer(1)) is None
