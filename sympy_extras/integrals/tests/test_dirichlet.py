"""Tests of the Dirichlet, Frullani and Borwein integrals."""
from __future__ import annotations

from sympy import symbols, sin, cos, exp, log, oo, pi, I, Rational, S, Integer, sqrt
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals import definite_integral
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.dirichlet import dirichlet_integral, trigonometric_sum

x, a, k = symbols('x a k')


def test_trigonometric_sums() -> None:
    assert trigonometric_sum(sin(x)**2, x) == (S.Half, {(Integer(2), cos): -S.Half})
    assert trigonometric_sum(sin(x) * sin(3 * x), x) == (S.Zero, {(Integer(2), cos): S.Half, (Integer(4), cos): -S.Half})
    # a phase expanded, a negative frequency reflected (cos even, sin odd)
    assert trigonometric_sum(sin(x + pi / 4), x) == (S.Zero, {(S.One, sin): sqrt(2) / 2, (S.One, cos): sqrt(2) / 2})
    assert trigonometric_sum(sin(-2 * x) + cos(-3 * x), x) == (S.Zero, {(Integer(2), sin): -S.One, (Integer(3), cos): S.One})
    # exp(I*w*x) as cosine and sine
    assert trigonometric_sum(exp(2 * I * x), x) == (S.Zero, {(Integer(2), cos): S.One, (Integer(2), sin): I})
    # a symbolic frequency needs its sign
    assert trigonometric_sum(sin(a * x), x) is None
    assert trigonometric_sum(sin(a * x), x, a > 0) == (S.Zero, {(a, sin): S.One})
    # not a trigonometric sum
    assert trigonometric_sum(sin(x**2), x) is None
    assert trigonometric_sum(sin(x) * exp(-x), x) is None


def test_dirichlet_frullani_and_powers_of_sinc() -> None:
    assert dirichlet_integral(sin(x) / x, x, 0, oo) == ConditionalValue(pi / 2)
    assert dirichlet_integral(sin(x) / x, x, -oo, oo) == ConditionalValue(pi)
    assert dirichlet_integral(sin(a * x) / x, x, 0, oo, a > 0) == ConditionalValue(pi / 2)
    assert dirichlet_integral(sin(a * x) / x, x, 0, oo, a < 0) == ConditionalValue(-pi / 2)
    assert dirichlet_integral((cos(x) - cos(2 * x)) / x, x, 0, oo) == ConditionalValue(log(2))
    assert dirichlet_integral((cos(3 * x) - cos(x / 2)) / x, x, 0, oo) == ConditionalValue(-log(6))
    assert dirichlet_integral(sin(x)**2 / x**2, x, 0, oo) == ConditionalValue(pi / 2)
    assert dirichlet_integral(sin(x)**3 / x**3, x, 0, oo) == ConditionalValue(3 * pi / 8)
    assert dirichlet_integral(sin(x)**4 / x**4, x, 0, oo) == ConditionalValue(pi / 3)
    assert dirichlet_integral((1 - cos(x)) / x**2, x, -oo, oo) == ConditionalValue(pi)
    # GR 3.741.3: sin(x)**3/x**2 over (0, oo) is 3 log(3)/4
    assert dirichlet_integral(sin(x)**3 / x**2, x, 0, oo) == ConditionalValue(3 * log(3) / 4)
    assert dirichlet_integral(sin(x) * sin(2 * x) / x**2, x, 0, oo) == ConditionalValue(pi / 2)
    # the constant in the denominator, and a numerator to cancel
    assert dirichlet_integral(sin(x)**2 / (3 * x**2), x, 0, oo) == ConditionalValue(pi / 6)
    assert dirichlet_integral(x * sin(x)**2 / x**3, x, 0, oo) == ConditionalValue(pi / 2)


def test_borwein_integrals() -> None:
    # the products of sinc functions equal pi/2 up to the factor 1/13 and
    # not from 1/15 on (Borwein and Borwein, 2001)
    def sinc_product(n: int) -> Expr:
        f: Expr = S.One
        for j in range(n):
            f = f * sin(x / (2 * j + 1)) / (x / (2 * j + 1))
        return f
    assert dirichlet_integral(sinc_product(1), x, 0, oo) == ConditionalValue(pi / 2)
    assert dirichlet_integral(sinc_product(3), x, 0, oo) == ConditionalValue(pi / 2)
    assert dirichlet_integral(sinc_product(7), x, 0, oo) == ConditionalValue(pi / 2)
    found = dirichlet_integral(sinc_product(8), x, 0, oo)
    assert found is not None
    assert found.value == Rational(467807924713440738696537864469, 935615849440640907310521750000) * pi
    # the odd multiples of 1, 3, 5, 7 (Maxima's test suite, rtestint 63)
    assert dirichlet_integral(sin(x) * sin(3 * x) * sin(5 * x) * sin(7 * x) / (105 * x**4), x, -oo, oo) \
        == ConditionalValue(44 * pi / 315)


def test_divergent_and_foreign_shapes_are_refused() -> None:
    # sin(x)/x**2 is 1/x at 0, sin(x)**2/x is 1/(2 x) at infinity, cos(x)/x
    # is 1/x at 0: none converges
    assert dirichlet_integral(sin(x) / x**2, x, 0, oo) is None
    assert dirichlet_integral(sin(x)**2 / x, x, 0, oo) is None
    assert dirichlet_integral(sin(x)**2 / x, x, -oo, oo) is None
    assert dirichlet_integral(cos(x) / x, x, 0, oo) is None
    assert dirichlet_integral(exp(I * x) * sin(x) / x, x, -oo, oo) is None
    # not the shape: another range, another denominator, no trigonometry
    assert dirichlet_integral(sin(x) / x, x, 1, oo) is None
    assert dirichlet_integral(sin(x) / (x**2 + 1), x, 0, oo) is None
    assert dirichlet_integral(sin(x) / sqrt(x), x, 0, oo) is None
    assert dirichlet_integral(exp(-x) / x, x, 0, oo) is None


def test_through_the_driver() -> None:
    assert definite_integral(sin(x) * sin(3 * x) * sin(5 * x) * sin(7 * x) / (105 * x**4), (x, -oo, oo)) == 44 * pi / 315
    assert definite_integral((cos(x) - cos(2 * x)) / x, (x, 0, oo)) == log(2)
    # Maxima's wester 31: exp(2 I t) sin(t)/t over the real line, the real
    # part (sin(3t) - sin(t))/(2t) and the imaginary (cos(t) - cos(3t))/(2t)
    # both integrating to 0
    assert definite_integral(exp(2 * I * x) * sin(x) / x, (x, -oo, oo)) == 0
    value = definite_integral(sin(x)**2 * cos(x) / x**2, (x, 0, oo))
    assert value == pi / 4 and verify_numerically(value, sin(x)**2 * cos(x) / x**2, x, S.Zero, oo) is True


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(dirichlet_integral)([1], x, 0, oo))
