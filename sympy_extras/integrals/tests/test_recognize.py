"""Tests of the symbolic-numeric recognition of constants."""
from __future__ import annotations

from sympy import symbols, S, pi, log, sqrt, zeta, atan, exp, oo, Rational, Float

from sympy_extras.integrals.recognize import recognize_constant, recognize_integral, quadrature
from sympy_extras.settings import configure

x = symbols('x')


def test_constants_from_digits() -> None:
    # zeta(2), Catalan's constant and log(3) to 30 digits and more
    assert recognize_constant(Float('1.6449340668482264364724151666460251892', 38)) == pi**2 / 6
    assert recognize_constant(Float('0.91596559417721901505460351493238411077', 38)) == S.Catalan
    assert recognize_constant(Float('1.0986122886681096913952452369225257046', 38)) == log(3)
    # an exact expression is checked at a higher precision
    assert recognize_constant(pi**2 / 6 + zeta(3) / 7) == pi**2 / 6 + zeta(3) / 7
    assert recognize_constant(Rational(3, 7)) == Rational(3, 7)
    # a number with no relation to the constants, and too few digits
    assert recognize_constant(Float('0.123456789012345678901234567890', 30)) is None
    assert recognize_constant(0.5) is None
    assert recognize_constant(Float('0.7071', 4)) is None
    # a parameter cannot be recognised
    assert recognize_constant(pi * symbols('a')) is None


def test_integrals() -> None:
    # Integral(1/(x^3 + 1), (x, 0, 1)) = log(2)/3 + sqrt(3) pi / 9 (a classic partial fraction integral)
    assert recognize_integral(1 / (x**3 + 1), (x, 0, 1)) == log(2) / 3 + sqrt(3) * pi / 9
    # Catalan's constant, GR 4.531.1
    assert recognize_integral(atan(x) / x, (x, 0, 1)) == S.Catalan
    # GR 4.291.8: Integral(log(1 + x)/(1 + x^2), (x, 0, 1)) = pi log(2) / 8
    assert recognize_integral(log(1 + x) / (1 + x**2), (x, 0, 1)) == pi * log(2) / 8
    # GR 3.411.1: Integral(x^2/(e^x - 1), (x, 0, oo)) = 2 zeta(3)
    assert recognize_integral(x**2 / (exp(x) - 1), (x, 0, oo)) == 2 * zeta(3)
    # nothing to recognise, and parameters are refused
    assert recognize_integral(exp(-x**x), (x, 0, 1)) is None
    assert recognize_integral(exp(-symbols('a') * x), (x, 0, oo)) is None


def test_quadrature_and_the_switch() -> None:
    value = quadrature(exp(-x), x, S.Zero, oo)
    assert value is not None and abs(value - 1) < 1e-25
    # a complex-valued integrand is not trusted
    assert quadrature(sqrt(x - 2), x, S.Zero, S.One) is None
    with configure(numerical_checks=False):
        assert recognize_constant(Float('1.6449340668482264364724151666460251892', 38)) is None
        assert recognize_integral(atan(x) / x, (x, 0, 1)) is None
