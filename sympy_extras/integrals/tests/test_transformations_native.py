"""Tests of Frullani's theorem and Glasser's master theorem on the objects
of the core (:mod:`sympy_extras.integrals.transformations`), with the
values of Boros–Moll, *Irresistible Integrals*, 5.2 and 13, and Glasser
1983."""
from __future__ import annotations

from sympy_extras.core import S, Symbol, oo, pi, sqrt
from sympy_extras.functions import atan, cos, exp, log
from sympy_extras.integrals.conditions import ConditionalValue, numerically_equal
from sympy_extras.integrals.transformations import frullani, glasser, transformation_integral

x, k = Symbol('x'), Symbol('k')
a, b = Symbol('a', positive=True), Symbol('b', positive=True)


def test_frullani() -> None:
    # Boros-Moll 5.2: (f(0) - f(oo)) log(b/a)
    assert frullani((exp(-a * x) - exp(-b * x)) / x, x) == ConditionalValue(log(b / a))
    found = frullani((atan(a * x) - atan(b * x)) / x, x)
    assert found is not None and numerically_equal(found.value, -pi * log(b / a) / 2)
    # the condition on the scales is reported
    c, d = Symbol('c'), Symbol('d')
    found = frullani((atan(c * x) - atan(d * x)) / x, x)
    assert found is not None and found.condition == ((c > 0) & (d > 0))
    # no finite limit at infinity, or not a Frullani shape
    assert frullani((cos(a * x) - cos(b * x)) / x, x) is None
    assert frullani((exp(-a * x) - exp(-b * x**2)) / x, x) is None
    assert frullani(exp(-a * x) / x, x) is None


def test_glasser() -> None:
    # the Cauchy-Schlomilch transformation over the half line (even F)
    assert glasser(exp(-(x - 1 / x)**2), x, 0, oo) == ConditionalValue(sqrt(pi) / 2)
    assert glasser(1 / (1 + (x - 1 / x)**2), x, 0, oo) == ConditionalValue(pi / 2)
    # Glasser's theorem over the real line with two poles
    assert glasser(exp(-(x - 1 / (x - 1) - 2 / (x + 1))**2), x, -oo, oo) == ConditionalValue(sqrt(pi))
    # an odd F over the half line, a negative coefficient, a map which is
    # not rational (apart raises PolynomialError on exp(x)) are refused
    assert glasser(exp(-(x - 1 / x)**2) * (x - 1 / x), x, 0, oo) is None
    assert glasser(exp(-(x + 1 / x)**2), x, -oo, oo) is None
    assert glasser(x * exp(x) * exp(k * x) / (exp(x) + 3), x, -oo, oo) is None
    assert transformation_integral(exp(-(x + exp(x))**2), x, -oo, oo) is None
    assert transformation_integral(exp(-x), x, S.Zero, S.One) is None


def test_parameters_in_the_map() -> None:
    # a pole with a parameter: x - a/(x - b) maps each interval onto the line
    found = glasser(exp(-(x - a / (x - b))**2), x, -oo, oo)
    assert found == ConditionalValue(sqrt(pi))
