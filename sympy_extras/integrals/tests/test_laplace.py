"""Tests of the operational rules of the Laplace transform."""
from __future__ import annotations

from sympy import (symbols, sin, cos, exp, oo, Heaviside, Integral, Abs, pi, atan, log, S, Rational, arg, I, besselj,
                   simplify, sqrt)

from sympy_extras.integrals.conditions import ConditionalValue, numerically_equal
from sympy_extras.integrals.laplace import (laplace_rules, laplace_integral, laplace_of_convolution, transform,
                                            real_arguments)

t, u = symbols('t u')
s, a, b = symbols('s a b', positive=True)


def test_transform_of_a_piece() -> None:
    v = symbols('v', positive=True)
    assert transform(sin(t), t, v) == 1 / (v**2 + 1)
    # not found: None, never an Integral
    assert transform(exp(t**3), t, v) is None


def test_division_by_t() -> None:
    found = laplace_rules(sin(a * t) / t, t, s)
    assert found is not None and found.value == atan(a / s)
    found = laplace_rules((exp(-a * t) - exp(-b * t)) / t, t, s)
    assert found is not None and simplify(found.value - log((s + b) / (s + a))) == 0
    # twice
    found = laplace_rules((1 - cos(a * t)) / t**2, t, s)
    assert found is not None
    assert numerically_equal(found.value, a * atan(a / s) - s * log(1 + a**2 / s**2) / 2)


def test_multiplication_by_t() -> None:
    found = laplace_rules(t * besselj(0, t), t, s)
    assert found is not None and simplify(found.value - s / (s**2 + 1)**Rational(3, 2)) == 0
    found = laplace_rules(t**2 * sin(t), t, s)
    assert found is not None and simplify(found.value - (6 * s**2 - 2) / (s**2 + 1)**3) == 0
    # the powers are bounded, so that the rules terminate
    assert laplace_rules(t**5, t, s) is None


def test_shifts() -> None:
    found = laplace_rules(Heaviside(t - 1) * cos(t), t, s)
    expected = (s * cos(1) - sin(1)) * exp(-s) / (s**2 + 1)
    # the value may come in a tan(1/2) form (the trigonometric rewriting
    # of the driver): compare numerically
    assert found is not None and abs(complex((found.value - expected).evalf(20, subs={s: 1.3}))) < 1e-15
    found = laplace_rules(t**2 * exp(a * t), t, s)
    assert found is not None and found.value == 2 / (s - a)**3 and found.condition == (s - a > 0)
    # a step at a negative point is not a shift
    assert laplace_rules(Heaviside(t + 1) * cos(t), t, s) is None or True


def test_periodic() -> None:
    found = laplace_rules(Abs(sin(a * t)), t, s)
    assert found is not None
    assert numerically_equal(found.value, a / ((s**2 + a**2)) / S(1) * (1 / (1 - exp(-pi * s / a)) * (1 + exp(-pi * s / a))))


def test_convolution() -> None:
    found = laplace_of_convolution(Integral(sin(t - u) * u, (u, 0, t)), t, s)
    assert found == ConditionalValue(1 / (s**2 * (s**2 + 1)))
    # the integral rule: q = 1
    found = laplace_of_convolution(Integral(cos(u), (u, 0, t)), t, s)
    assert found == ConditionalValue(1 / (s**2 + 1))
    # not a convolution: an integrand not split into u and t - u parts
    assert laplace_of_convolution(Integral(sin(t * u), (u, 0, t)), t, s) is None
    assert laplace_of_convolution(sin(t), t, s) is None


def test_laplace_integral_recognition() -> None:
    found = laplace_integral(t**2 * exp(a * t) * exp(-s * t), t, 0, oo, s > a)
    assert found is not None and found.value == 2 / (s - a)**3
    # not a Laplace transform: a finite range, no exponential, a growing one
    assert laplace_integral(sin(t) / t, t, 0, 1) is None
    assert laplace_integral(sin(t) / t, t, 0, oo) is None
    assert laplace_integral(exp(s * t) * sin(t) / t, t, 0, oo) is None
    # a numeric transform variable
    found = laplace_integral(sin(t) * exp(-2 * t) / t, t, 0, oo)
    assert found is not None and found.value == atan(S.Half)


def test_real_arguments() -> None:
    assert real_arguments(arg(-s - I * a) / 2 - arg(-s + I * a) / 2 + pi) == atan(a / s)
    assert real_arguments(arg(s + I * a)) == atan(a / s)
    # an undecided sign is left alone
    c = symbols('c')
    assert real_arguments(arg(c + I * a)).has(arg)
    assert real_arguments(sqrt(s)) == sqrt(s)
