"""Tests of the summability methods on the objects of the core
(:mod:`sympy_extras.integrals.summability`), with the regularised
integrals of Hardy, *Divergent Series*, 5.14."""
from __future__ import annotations

from sympy.testing.pytest import raises

from sympy_extras.core import I, S, Symbol, oo, pi
from sympy_extras.functions import besselj, cos, exp, sin
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.summability import METHODS, _cesaro_mean, oscillatory_tail, summable_integral

x, k = Symbol('x'), Symbol('k')
a = Symbol('a', positive=True)


def test_abel_and_gaussian_means() -> None:
    assert summable_integral(sin(x), x, 0, oo, 'abel') == ConditionalValue(1)
    assert summable_integral(x * cos(x), x, 0, oo, 'abel') == ConditionalValue(-1)
    assert summable_integral(exp(I * x), x, 0, oo, 'gaussian') == ConditionalValue(I)
    assert summable_integral(sin(a * x), x, 0, oo, 'abel') == ConditionalValue(1 / a)


def test_cesaro_means() -> None:
    # (C, 1) of x sin(x) oscillates, (C, 2) exists
    t = Symbol('t', positive=True)
    assert _cesaro_mean(t * sin(t), t, 1, None) is None
    assert _cesaro_mean(t * sin(t), t, 2, None) == ConditionalValue(0)
    assert summable_integral(cos(x), x, 0, oo, 'cesaro') == ConditionalValue(0)


def test_convergent_and_divergent() -> None:
    # a convergent integral is its own mean; sin(x)**2 and x have none
    assert summable_integral(besselj(0, x), x, 0, oo, 'abel') == ConditionalValue(1)
    assert summable_integral(sin(x)**2, x, 0, oo, 'abel') is None
    assert summable_integral(x, x, 0, oo, 'gaussian') is None


def test_ranges() -> None:
    assert summable_integral(cos(x), x, -oo, oo, 'abel') == ConditionalValue(0)
    assert summable_integral(sin(x), x, -oo, 0, 'abel') == ConditionalValue(-1)
    assert summable_integral(sin(x), x, oo, 0, 'abel') == ConditionalValue(-1)
    assert summable_integral(sin(x), x, 0, pi, 'abel') == ConditionalValue(2)
    assert summable_integral(sin(x), x, 2, 2, 'abel') == ConditionalValue(S.Zero)
    assert oscillatory_tail(sin(x), x, pi, 'abel') == ConditionalValue(-1)


def test_a_parameter_named_like_the_dummy() -> None:
    # the variable of the tail is a dummy of its own name: a parameter t
    # of the integrand is not taken for it by the typed polynomials
    t = Symbol('t', positive=True)
    assert summable_integral(sin(t * x), x, 0, oo, 'abel') == ConditionalValue(1 / t)


def test_conditions_and_methods() -> None:
    assert summable_integral(sin(k * x), x, 0, oo, 'abel', k > 0) == ConditionalValue(1 / k)
    raises(ValueError, lambda: summable_integral(sin(x), x, 0, oo, 'borel'))
    raises(ValueError, lambda: oscillatory_tail(sin(x), x, 0, 'abel '))
    assert METHODS == ('abel', 'cesaro', 'gaussian')
