"""Tests of the Dirichlet, Frullani and Borwein integrals on the objects of
the core (:mod:`sympy_extras.integrals.dirichlet`)."""
from __future__ import annotations

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import I, Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, exp, log, sin
from sympy_extras.functions.elementary.trigonometric import Cos, Sin
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.dirichlet import dirichlet_integral, trigonometric_sum

x, a = Symbol('x'), Symbol('a')


def test_trigonometric_sums() -> None:
    assert trigonometric_sum(sin(x)**2, x) == (S.Half, {(Integer(2), Cos): -S.Half})
    assert trigonometric_sum(sin(x) * sin(3 * x), x) == (S.Zero, {(Integer(2), Cos): S.Half,
                                                                    (Integer(4), Cos): -S.Half})
    # a phase expanded, a negative frequency reflected (cos even, sin odd)
    assert trigonometric_sum(sin(x + pi / 4), x) == (S.Zero, {(S.One, Sin): sqrt(2) / 2, (S.One, Cos): sqrt(2) / 2})
    assert trigonometric_sum(sin(-2 * x) + cos(-3 * x), x) == (S.Zero, {(Integer(2), Sin): -S.One,
                                                                          (Integer(3), Cos): S.One})
    assert trigonometric_sum(exp(2 * I * x), x) == (S.Zero, {(Integer(2), Cos): S.One, (Integer(2), Sin): I})
    assert trigonometric_sum(sin(a * x), x) is None
    assert trigonometric_sum(sin(a * x), x, a > 0) == (S.Zero, {(a, Sin): S.One})
    assert trigonometric_sum(sin(x**2), x) is None
    assert trigonometric_sum(sin(x) * exp(-x), x) is None


def test_dirichlet_frullani_and_powers_of_sinc() -> None:
    assert dirichlet_integral(sin(x) / x, x, 0, oo) == ConditionalValue(pi / 2)
    assert dirichlet_integral(sin(x) / x, x, -oo, oo) == ConditionalValue(pi)
    assert dirichlet_integral(sin(a * x) / x, x, 0, oo, a < 0) == ConditionalValue(-pi / 2)
    assert dirichlet_integral((cos(x) - cos(2 * x)) / x, x, 0, oo) == ConditionalValue(log(2))
    assert dirichlet_integral((cos(3 * x) - cos(x / 2)) / x, x, 0, oo) == ConditionalValue(-log(6))
    assert dirichlet_integral(sin(x)**4 / x**4, x, 0, oo) == ConditionalValue(pi / 3)
    assert dirichlet_integral((1 - cos(x)) / x**2, x, -oo, oo) == ConditionalValue(pi)
    # GR 3.741.3; the bug of the port: the logarithms combined to
    # log(3**(3/4)), the core's expand not expanding logarithms by default
    assert dirichlet_integral(sin(x)**3 / x**2, x, 0, oo) == ConditionalValue(3 * log(3) / 4)
    assert dirichlet_integral(x * sin(x)**2 / x**3, x, 0, oo) == ConditionalValue(pi / 2)


def test_borwein_integrals() -> None:
    def sinc_product(n: int) -> Expr:
        f: Expr = S.One
        for j in range(n):
            f = f * sin(x / (2 * j + 1)) / (x / (2 * j + 1))
        return f
    assert dirichlet_integral(sinc_product(7), x, 0, oo) == ConditionalValue(pi / 2)
    found = dirichlet_integral(sinc_product(8), x, 0, oo)
    assert found is not None
    assert found.value == Rational(467807924713440738696537864469, 935615849440640907310521750000) * pi


def test_divergent_and_foreign_shapes_are_refused() -> None:
    assert dirichlet_integral(sin(x) / x**2, x, 0, oo) is None
    assert dirichlet_integral(sin(x)**2 / x, x, -oo, oo) is None
    assert dirichlet_integral(cos(x) / x, x, 0, oo) is None
    assert dirichlet_integral(sin(x) / x, x, 1, oo) is None
    assert dirichlet_integral(sin(x) / (x**2 + 1), x, 0, oo) is None
    assert dirichlet_integral(sin(x) / sqrt(x), x, 0, oo) is None
    assert dirichlet_integral(exp(-x) / x, x, 0, oo) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(dirichlet_integral)([1], x, 0, oo))
