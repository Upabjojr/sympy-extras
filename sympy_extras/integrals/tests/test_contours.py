"""Tests of the rectangular, sector and indented contours."""
from __future__ import annotations

from sympy import symbols, exp, sinh, cosh, sin, cos, oo, pi, S, sqrt, simplify

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.contours import (contour_integral, rectangular_integral, sector_integral,
                                             indented_integral)
from sympy_extras.integrals.definite import verify_numerically

x = symbols('x')
a, b, n = symbols('a b n', positive=True)


def _same(u: ExprLike, v: ExprLike) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0


def test_rectangle_without_exponential_factor() -> None:
    # k = 0: the shifted system (the equation for z**(n+1))
    found = rectangular_integral(1 / cosh(x), x)
    assert found is not None and found.value == pi and found.condition is S.true
    found = rectangular_integral(1 / (2 * cosh(x) + 1), x)
    assert found is not None and _same(found.value, 2 * pi / (3 * sqrt(3)))
    # a rational function of exp(x) which is not a Mellin kernel, checked numerically
    found = rectangular_integral(1 / (cosh(x)**2 + 1), x)
    assert found is not None and verify_numerically(found.value, 1 / (cosh(x)**2 + 1), x, -oo, oo) is True


def test_rectangle_with_powers_of_x() -> None:
    # the simple pole of R at u = 1 (z = 0) is cancelled by the factor x
    found = rectangular_integral(x / sinh(x), x)
    assert found is not None and found.value == pi**2 / 2
    found = rectangular_integral(x**2 / cosh(x), x)
    assert found is not None and found.value == pi**3 / 4
    found = rectangular_integral(x * exp(x / 2) / (1 + exp(x))**2, x)
    assert found is not None and verify_numerically(found.value, x * exp(x / 2) / (1 + exp(x))**2, x, -oo, oo) is True


def test_rectangle_with_exponential_factor() -> None:
    found = rectangular_integral(exp(a * x) / (1 + exp(x)), x, a < 1)
    assert found is not None and found.value == pi / sin(pi * a) and found.condition == (a < 1)
    # a numeric exponent inside the window of decay, checked numerically
    found = rectangular_integral(exp(x / 3) / (1 + exp(x)), x)
    assert found is not None and _same(found.value, pi / sin(pi / 3))
    # outside the window: divergent, nothing claimed
    assert rectangular_integral(exp(2 * x) / (1 + exp(x)), x) is None
    # a pole of R on the positive axis which is not cancelled: refused
    assert rectangular_integral(1 / (1 - exp(x)), x) is None


def test_sector() -> None:
    found = sector_integral(1 / (1 + x**n), x, n > 1)
    assert found is not None and found.value == pi / (n * sin(pi / n))
    found = sector_integral(x**a / (1 + x**n), x, a < n - 1)
    assert found is not None and found.condition == (a < n - 1)
    found = sector_integral(x**2 / (3 + 2 * x**6), x)
    assert found is not None and _same(found.value, sqrt(6) * pi / 36)
    assert verify_numerically(sqrt(6) * pi / 36, x**2 / (3 + 2 * x**6), x, S.Zero, oo) is True
    assert sector_integral(1 / (1 - x**n), x) is None


def test_indented() -> None:
    found = indented_integral(sin(x) / x, x, -oo, oo)
    assert found is not None and found.value == pi
    found = indented_integral(sin(a * x) / (x * (x**2 + b**2)), x, S.Zero, oo)
    assert found is not None and _same(found.value, pi * (1 - exp(-a * b)) / (2 * b**2))
    # the integrand is not regular at the pole: refused
    assert indented_integral(cos(x) / x, x, -oo, oo) is None
    # not even over (0, oo)
    assert indented_integral(sin(x) / (x + 1)**2, x, S.Zero, oo) is None


def test_dispatch() -> None:
    assert contour_integral(x / sinh(x), x, -oo, oo) is not None
    assert contour_integral(1 / (1 + x**n), x, S.Zero, oo, n > 1) is not None
    assert contour_integral(sin(x) / x, x, -oo, oo) is not None
    assert contour_integral(exp(-x**2), x, -oo, oo) is None
