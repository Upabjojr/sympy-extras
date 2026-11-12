"""Tests of the operational rules of the Laplace transform on the objects
of the core (:mod:`sympy_extras.integrals.laplace`): the pairs of
Abramowitz and Stegun 29.3 (independent values), the rules which do not
apply, and the agreement with the SymPy face."""
from __future__ import annotations

import sympy

from sympy_extras.core import I, Rational, S, Symbol, oo, pi
from sympy_extras.functions import Arg, abs_, atan, besselj, cos, exp, heaviside, log, sin, sqrt
from sympy_extras.integrals.conditions import ConditionalValue, numerically_equal
from sympy_extras.integrals.laplace import (has_integral, laplace_integral, laplace_of_convolution, laplace_rules,
                                            real_arguments, transform)
from sympy_extras.interop import expr_from_sympy, from_sympy

t, u = Symbol('t'), Symbol('u')
s, a, b = Symbol('s', positive=True), Symbol('a', positive=True), Symbol('b', positive=True)


def test_transform_of_a_piece() -> None:
    v = Symbol('v', positive=True)
    assert transform(sin(t), t, v) == 1 / (v**2 + 1)
    # not found: None, never an Integral
    assert transform(exp(t**3), t, v) is None


def test_division_and_multiplication() -> None:
    # atan(a/s) (AS 29.3.112) and the difference of two exponentials
    found = laplace_rules(sin(a * t) / t, t, s)
    assert found == ConditionalValue(atan(a / s))
    found = laplace_rules((exp(-a * t) - exp(-b * t)) / t, t, s)
    assert found is not None and numerically_equal(found.value, log((s + b) / (s + a)))
    found = laplace_rules((1 - cos(a * t)) / t**2, t, s)
    assert found is not None and numerically_equal(found.value, a * atan(a / s) - s * log(1 + a**2 / s**2) / 2)
    # the transform 1/sqrt(s**2 + 1) of J0 differentiated: s/(s**2 + 1)**(3/2)
    found = laplace_rules(t * besselj(0, t), t, s)
    assert found is not None and numerically_equal(found.value, s / (s**2 + 1)**Rational(3, 2))
    # the powers are bounded, so that the rules terminate
    assert laplace_rules(t**5, t, s) is None


def test_shifts_and_periods() -> None:
    found = laplace_rules(heaviside(t - 1) * exp(-t), t, s)
    assert found is not None and numerically_equal(found.value, exp(-s - 1) / (s + 1))
    found = laplace_rules(t**2 * exp(a * t), t, s)
    assert found == ConditionalValue(2 / (s - a)**3, s - a > 0)
    # the rectified sine, a period pi/a: a coth(pi s/(2 a))/(s**2 + a**2)
    found = laplace_rules(abs_(sin(a * t)), t, s)
    assert found is not None
    assert numerically_equal(found.value, a * (1 + exp(-pi * s / a)) / ((s**2 + a**2) * (1 - exp(-pi * s / a))))
    # a step at a negative point is not a shift, an exponential of t**2 no
    # exponential shift
    assert laplace_rules(heaviside(t + 1) * cos(t), t, s) is None
    assert laplace_rules(exp(t**2), t, s) is None


def test_convolution() -> None:
    # the Integral of the rule is SymPy's, wrapped in a node of the core
    T, U = sympy.symbols('t u')
    integral = expr_from_sympy(sympy.Integral(sympy.sin(T - U) * U, (U, 0, T)))
    found = laplace_of_convolution(integral, t, s)
    assert found == ConditionalValue(1 / (s**2 * (s**2 + 1)))
    assert laplace_of_convolution(sin(t), t, s) is None
    assert not has_integral(sin(t))


def test_laplace_integral() -> None:
    found = laplace_integral(t**2 * exp(a * t) * exp(-s * t), t, 0, oo, s > a)
    assert found is not None and found.value == 2 / (s - a)**3
    assert laplace_integral(sin(t) / t, t, 0, 1) is None
    assert laplace_integral(sin(t) / t, t, 0, oo) is None
    assert laplace_integral(exp(s * t) * sin(t) / t, t, 0, oo) is None
    found = laplace_integral(sin(t) * exp(-2 * t) / t, t, 0, oo)
    assert found == ConditionalValue(atan(S.Half))


def test_real_arguments() -> None:
    assert real_arguments(Arg(-s - I * a) / 2 - Arg(-s + I * a) / 2 + pi) == atan(a / s)
    assert real_arguments(Arg(s + I * a)) == atan(a / s)
    # an undecided sign is left alone
    c = Symbol('c')
    assert real_arguments(Arg(c + I * a)).has(Arg)
    assert real_arguments(sqrt(s)) == sqrt(s)


def test_the_face_agrees() -> None:
    from sympy_extras.integrals.laplace_sympy import laplace_rules as face_rules
    found = face_rules(sympy.sin(sympy.Symbol('a', positive=True) * sympy.Symbol('t')) / sympy.Symbol('t'),
                       sympy.Symbol('t'), sympy.Symbol('s', positive=True))
    native = laplace_rules(sin(a * t) / t, t, s)
    assert found is not None and native is not None and from_sympy(found.value) == native.value
