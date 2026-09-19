"""Identities from the literature, as facts to decide: the arctangent
formulas for pi (Machin 1706, Euler, Gauss, Størmer), the classical
identities between the inverse functions on their intervals, and
constants which textbooks on the constant problem use as examples."""
from __future__ import annotations

from sympy import I, Rational, acos, acosh, asin, asinh, atan, cos, exp, log, pi, sin, sqrt, symbols, tan

from sympy_extras.simplify import is_zero

x = symbols('x')


def test_arctangent_formulas_for_pi() -> None:
    a = atan
    # Machin (1706)
    assert is_zero(pi / 4 - 4 * a(Rational(1, 5)) + a(Rational(1, 239))) is True
    # Euler
    assert is_zero(pi / 4 - a(Rational(1, 2)) - a(Rational(1, 3))) is True
    # Hermann
    assert is_zero(pi / 4 - 2 * a(Rational(1, 2)) + a(Rational(1, 7))) is True
    # Hutton
    assert is_zero(pi / 4 - 2 * a(Rational(1, 3)) - a(Rational(1, 7))) is True
    # Gauss
    assert is_zero(pi / 4 - 12 * a(Rational(1, 18)) - 8 * a(Rational(1, 57)) + 5 * a(Rational(1, 239))) is True
    # Størmer
    assert is_zero(pi / 4 - 6 * a(Rational(1, 8)) - 2 * a(Rational(1, 57)) - a(Rational(1, 239))) is True
    # a near miss is not an identity
    assert is_zero(pi / 4 - a(Rational(1, 2)) - a(Rational(1, 4))) is False
    assert is_zero(pi / 4 - 4 * a(Rational(1, 5)) + a(Rational(1, 238))) is False


def test_inverse_functions_on_their_intervals() -> None:
    inside = [x > -1, x < 1]
    assert is_zero(acos(x) - 2 * atan(sqrt(1 - x**2) / (1 + x)), inside) is True
    assert is_zero(asin(x) - atan(x / sqrt(1 - x**2)), inside) is True
    assert is_zero(asin(x) - atan(x / sqrt(1 - x**2))) is False        # x = 2
    assert is_zero(acosh(3) - log(3 + 2 * sqrt(2))) is True
    assert is_zero(asinh(1) - log(1 + sqrt(2))) is True
    assert is_zero(2 * atan(x) - atan(2 * x / (1 - x**2)), inside) is True
    assert is_zero(2 * atan(x) - atan(2 * x / (1 - x**2)) - pi, x > 1) is True


def test_constants_of_the_constant_problem() -> None:
    # de Moivre and the roots of unity
    assert is_zero((cos(pi / 7) + I * sin(pi / 7))**7 + 1) is True
    assert is_zero(exp(2 * I * pi / 5) + exp(4 * I * pi / 5) + exp(6 * I * pi / 5) + exp(8 * I * pi / 5) + 1) is True
    assert is_zero(tan(pi / 8) - sqrt(2) + 1) is True
    assert is_zero(sin(pi / 10) - (sqrt(5) - 1) / 4) is True
    # nested radicals which denest
    assert is_zero(sqrt(3 + 2 * sqrt(2)) - 1 - sqrt(2)) is True
    assert is_zero(sqrt(5 + 2 * sqrt(6)) - sqrt(2) - sqrt(3)) is True
    # logarithms of units and their powers
    assert is_zero(log(7 + 4 * sqrt(3)) - 2 * log(2 + sqrt(3))) is True
    assert is_zero(log(sqrt(2) - 1) + log(sqrt(2) + 1)) is True
    # I**I and the Gelfond constant
    assert is_zero(I**I - exp(-pi / 2)) is True
    assert is_zero(exp(pi) - (-1)**(-I)) is True
    assert is_zero(exp(pi * sqrt(163)) - 262537412640768744) is False  # Ramanujan's near integer
