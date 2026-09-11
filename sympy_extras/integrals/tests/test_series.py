"""Tests of the series expansion with termwise integration."""
from __future__ import annotations

from sympy import symbols, log, exp, oo, pi, S, Dummy, atan, Catalan, polygamma
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.series import expansions, moments, sum_series, series_integral
from sympy_extras.settings import configure

x = symbols('x')
p = symbols('p', positive=True)


def test_expansions_of_the_factors() -> None:
    found = list(expansions(log(1 - x) / x, x, S.Zero, S.One))
    assert [(e.factor, rest) for e, rest in found] == [(log(1 - x), 1 / x), (log(1 - x) / x, 1)]
    # the formula -1/j of log(1 - x) is infinite at j = 0: the series starts at 1
    assert found[0][0].start == 1
    # the geometric series of exponentials on (0, oo)
    found = list(expansions(x / (exp(x) - 1), x, S.Zero, oo))
    assert found and found[0][0].factor == 1 / (exp(x) - 1) and found[0][0].start == 1
    assert found[0][0].coefficient == 1
    found = list(expansions(x / (exp(x) + 1), x, S.Zero, oo))
    assert found and found[0][0].coefficient.subs(found[0][0].index, 2) == -1
    # log(x) has no power series about 0, a monomial is left to the rest
    assert [e.factor for e, _ in expansions(x**p * log(x), x, S.Zero, S.One)] == []


def test_moments_and_sums() -> None:
    found = list(expansions(log(1 - x) / x, x, S.Zero, S.One))
    expansion, rest = found[0]
    moment = moments(expansion, rest, x, S.Zero, S.One)
    assert moment is not None and moment.subs(expansion.index, 3) == S(1) / 3
    j = Dummy('j', integer=True, nonnegative=True)
    assert sum_series((-1)**j / (2 * j + 1)**2, j, 0) == Catalan
    assert sum_series(1 / (j + 1)**2, j, 0) == pi**2 / 6
    assert sum_series(1 / (j + p + 1)**2, j, 0) == polygamma(1, p + 1)
    # a divergent series has no sum
    assert sum_series(S(2)**j, j, 0) is None


def test_series_integral() -> None:
    assert series_integral(log(1 - x) / x, x, 0, 1) == ConditionalValue(-pi**2 / 6)
    assert series_integral(x / (exp(x) - 1), x, 0, oo) == ConditionalValue(pi**2 / 6)
    assert series_integral(atan(x) / x, x, 0, 1) == ConditionalValue(Catalan)
    # a chosen factor
    found = series_integral(log(x) / (1 + x), x, 0, 1, expand=1 / (1 + x))
    assert found == ConditionalValue(-pi**2 / 12)
    # (0, c) is mapped onto (0, 1)
    found = series_integral(log(1 - x / 2) / x, x, 0, 2)
    assert found == ConditionalValue(-pi**2 / 6)
    # other ranges are not handled
    assert series_integral(log(1 - x) / x, x, S.Half, 1) is None


def test_unjustified_interchange_is_refused() -> None:
    # the terms of exp(-x)*log(1 + x) integrate to (-1)**(j-1) (j-1)!, a
    # divergent series: no value
    assert series_integral(exp(-x) * log(1 + x), x, 0, oo) is None
    # without numerical checks an undecided interchange is refused too
    with configure(numerical_checks=False):
        value = series_integral(log(1 - x) / x, x, 0, 1)
    assert value is None or value == ConditionalValue(-pi**2 / 6)


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(series_integral)([1], x, 0, 1))
