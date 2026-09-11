"""Tests of the mean value (Laurent coefficient) method for periodic
integrands."""
from __future__ import annotations

from sympy import symbols, exp, cos, sin, tan, pi, S, I, besseli, besselj, Rational, log
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.periodic import to_laurent, laurent_coefficient, mean_value_integral

x, z = symbols('x z')
a = symbols('a', positive=True)


def test_to_laurent() -> None:
    assert to_laurent(cos(2 * x) + sin(x), x, z) == z**2 / 2 + 1 / (2 * z**2) - I * (z - 1 / z) / 2
    assert to_laurent(exp(3 * I * x), x, z) == z**3
    assert to_laurent(tan(x), x, z) == (z - 1 / z) / (2 * I) / ((z + 1 / z) / 2)
    # an outer function of the trigonometric functions becomes an exponential
    outer = to_laurent(cos(sin(x)), x, z)
    assert outer is not None and outer.has(exp)
    # x occurring otherwise
    assert to_laurent(x * cos(x), x, z) is None
    assert to_laurent(cos(x / 2), x, z) is None
    assert to_laurent(log(x), x, z) is None


def test_laurent_coefficients() -> None:
    assert laurent_coefficient((z + 1 / z)**4, z) == 6
    assert laurent_coefficient((z + 1 / z)**4, z, 2) == 4
    assert laurent_coefficient(exp(z), z, 3) == Rational(1, 6)
    assert laurent_coefficient(exp(z), z, -1) == 0
    assert laurent_coefficient(exp(z / 2) * exp(1 / (2 * z)), z) == besseli(0, 1)
    # I_2(1) through the recurrence I_2 = I_0 - 2 I_1
    assert abs(complex((laurent_coefficient(exp(z / 2) * exp(1 / (2 * z)), z, 2) - besseli(2, 1)).evalf(20))) < 1e-15
    # the bug: exponentials of the same power coming from different
    # factors were not merged, and exp(cos x + sin x) gave nothing
    assert laurent_coefficient(exp(1 / (2 * z)) * exp(z / 2) * exp(I / (2 * z)) * exp(-I * z / 2), z) == besseli(0, S(2)**S.Half)
    # a rational function of z: the residues, not the series
    assert laurent_coefficient(1 / (z**2 + 3 * z + 1), z) is None
    # three functions of z
    assert laurent_coefficient(cos(z) * sin(z) * exp(1 / z), z) is None


def test_mean_value_integral() -> None:
    assert mean_value_integral(exp(cos(x)) * cos(sin(x)), x, 0, 2 * pi) == ConditionalValue(2 * pi)
    assert mean_value_integral(exp(cos(x)), x, 0, 2 * pi) == ConditionalValue(2 * pi * besseli(0, 1))
    assert mean_value_integral(exp(cos(x)) * sin(sin(x)), x, -pi, pi) == ConditionalValue(0)
    assert mean_value_integral(cos(x)**4, x, 0, 4 * pi) == ConditionalValue(3 * pi / 2)
    assert mean_value_integral(cos(cos(x)), x, 0, 2 * pi) == ConditionalValue(2 * pi * besselj(0, 1))
    # not a number of periods, a rational integrand (for the residues), x elsewhere
    assert mean_value_integral(exp(cos(x)), x, 0, pi) is None
    assert mean_value_integral(1 / (2 + cos(x)), x, 0, 2 * pi) is None
    assert mean_value_integral(x * cos(x), x, 0, 2 * pi) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(mean_value_integral)([1], x, 0, 2 * pi))
    raises(TypeError, lambda: untyped(laurent_coefficient)([z], z))
